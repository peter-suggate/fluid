#!/usr/bin/env node
/** Focused Dawn trajectory for Sparse CM12 ocean-seiche volume regressions. */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { createOceanSeicheScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const argument = (name: string, fallback: string): string => process.argv.slice(2)
  .find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const steps = Number(argument("steps", "24"));
const brickFineResolution = Number(argument("brick-fine", "8")) as 4 | 8 | 16;
const resolutionMode = argument("resolution-mode", "adaptive") as
  "adaptive" | "all-fine" | "all-coarse";
const out = resolve(argument("out", "artifacts/sparse-cm12-ocean-volume.json"));
if (!Number.isSafeInteger(steps) || steps < 1) throw new RangeError("steps must be positive");
if (![4, 8, 16].includes(brickFineResolution)) throw new RangeError("brick-fine must be 4, 8, or 16");
if (!["adaptive", "all-fine", "all-coarse"].includes(resolutionMode)) {
  throw new RangeError("resolution-mode must be adaptive, all-fine, or all-coarse");
}

const densityReceipt = (density: Float32Array) => {
  let mass = 0, positiveMass = 0, minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY, wetCells = 0;
  for (const value of density) {
    mass += value;
    positiveMass += Math.max(0, value);
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    if (value > 0.005) wetCells += 1;
  }
  return { mass, positiveMass, minimum, maximum, wetCells };
};

const fieldReceipt = (field: Float32Array) => {
  let maximumAbsolute = 0, sumAbsolute = 0, nonFinite = 0;
  for (const value of field) {
    if (!Number.isFinite(value)) nonFinite += 1;
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
    sumAbsolute += Math.abs(value);
  }
  return { maximumAbsolute, meanAbsolute: sumAbsolute / field.length, nonFinite };
};

let device: GPUDevice | undefined;
await acquireWebGPUExclusiveLock("ocean-volume",
  `Sparse CM12 ocean B${brickFineResolution} compiled transport`);
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn,
    [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert(adapter, "ocean volume probe requires a Dawn adapter");
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();
    validationErrors.push(event.error.message);
  });
  device.pushErrorScope("validation");
  const scene = createOceanSeicheScene();
  const options = {
    resolutionMode,
    brickFineResolution,
    presentationPageResolution: brickFineResolution,
    timeStep: "scene" as const,
  };
  const solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
    device, scene, "balanced", undefined, options, () => {});
  try {
    const initialFields = await solver.readDiagnosticFields();
    const initial = densityReceipt(initialFields.density);
    const dt_s = scene.numerics.maxDt_s;
    const trajectory: Record<string, unknown>[] = [];
    const gateErrors: string[] = [];
    for (let step = 1; step <= steps; step += 1) {
      assert.equal(solver.advanceTo(step * dt_s, []), true,
        `ocean step ${step} did not encode`);
      await device.queue.onSubmittedWorkDone();
      const [fields, activity, stats, frameControl, finalScalarMasks, pcm] = await Promise.all([
        solver.readDiagnosticFields(), solver.readGPUActivityPolicy(), solver.readStats(),
        solver.readFrameControlQA(), solver.readFinalScalarMaskHeaderQA(),
        solver.readPressureCanonicalMembershipQA(),
      ]);
      const density = densityReceipt(fields.density);
      if ((stats.pressureIterationsExecuted ?? 0) === 0) {
        gateErrors.push(`step ${step}: pressure executed zero iterations`);
      }
      if (pcm.cell.fault !== 0 || pcm.row.fault !== 0) {
        gateErrors.push(`step ${step}: PCM fault ${pcm.cell.fault}/${pcm.row.fault}`);
      }
      const scheduledTransitions = activity.bricks.flatMap((brick, brickId) =>
        brick.topologyPreparationScheduled ? [{ brickId,
          acceptedResolution: brick.acceptedResolution,
          candidateResolution: brick.candidateResolution,
          active: brick.active,
        }] : []);
      const faultCellRanges: Record<string, unknown>[] = [];
      if (pcm.cell.fault !== 0 && pcm.cell.firstFault !== 0xffff_ffff) {
        const resident = (solver as unknown as { resident: {
          templateWords: Uint32Array; brickFineResolution: number;
        } }).resident;
        const levels = Array.from({ length: Math.log2(resident.brickFineResolution) + 1 },
          (_, level) => 2 ** level);
        const rangesAt = resident.templateWords[11]!;
        for (let brickId = 0; brickId < activity.bricks.length; brickId += 1) {
          for (let level = 0; level < levels.length; level += 1) {
            const at = rangesAt + 2 * (levels.length * brickId + level);
            const first = resident.templateWords[at]!;
            const count = resident.templateWords[at + 1]!;
            if (pcm.cell.firstFault >= first && pcm.cell.firstFault < first + count) {
              const brick = activity.bricks[brickId]!;
              faultCellRanges.push({ brickId, resolution: levels[level], first, count,
                active: brick.active, acceptedResolution: brick.acceptedResolution,
                candidateResolution: brick.candidateResolution,
                topologyPreparationScheduled: brick.topologyPreparationScheduled });
            }
          }
        }
      }
      trajectory.push({
        step, time_s: step * dt_s, ...density,
        relativeMassDrift: (density.mass - initial.mass) / initial.mass,
        relativePositiveMassDrift:
          (density.positiveMass - initial.positiveMass) / initial.positiveMass,
        activeBricks: activity.bricks.filter((brick) => brick.active).length,
        acceptedCells: stats.adaptiveAcceptedCellCount,
        preparedBricks: stats.adaptiveTopologyPreparedBrickCount,
        committedBricks: stats.adaptiveTopologyCommittedBrickCount,
        topologyGeneration: stats.adaptiveTopologyShadowGeneration,
        gamma: fieldReceipt(fields.gamma),
        velocity: fieldReceipt(fields.velocity),
        pressure: fieldReceipt(fields.pressure),
        divergence: fieldReceipt(fields.divergence),
        pressureRelativeResidual: stats.pressureRelativeResidual,
        pressureIterationsExecuted: stats.pressureIterationsExecuted,
        pressureCells: stats.adaptivePressureCellCount,
        pressureRows: stats.adaptivePressureActiveRowCount,
        frameControl,
        scheduledTransitions,
        faultCellRanges,
        finalScalarMasks,
        pcm: {
          mode: pcm.mode,
          cell: {
            phase: pcm.cell.phase, fault: pcm.cell.fault,
            firstFault: pcm.cell.firstFault, dirtyCount: pcm.cell.dirtyCount,
            directWriteCount: pcm.cell.directWriteCount,
            directCauseMask: pcm.cell.directCauseMask,
            conflictPacket: pcm.cell.conflictPacket,
            acceptedGeneration: pcm.cell.acceptedGeneration,
            candidateGeneration: pcm.cell.candidateGeneration,
            totalCount: pcm.cell.totalCount, activeBitCount: pcm.cell.activeBitCount,
          },
          row: {
            phase: pcm.row.phase, fault: pcm.row.fault,
            firstFault: pcm.row.firstFault, dirtyCount: pcm.row.dirtyCount,
            directWriteCount: pcm.row.directWriteCount,
            directCauseMask: pcm.row.directCauseMask,
            conflictPacket: pcm.row.conflictPacket,
            acceptedGeneration: pcm.row.acceptedGeneration,
            candidateGeneration: pcm.row.candidateGeneration,
            totalCount: pcm.row.totalCount, activeBitCount: pcm.row.activeBitCount,
          },
        },
        pressureTopologyRepair: stats.adaptivePressureTopologyRepair,
        pressureTopologyAttribution: stats.adaptivePressureTopologyAttribution,
      });
    }
    const adaptiveRepresentation = await solver.readAdaptiveRepresentationQA();
    const final = trajectory[trajectory.length - 1] as {
      relativePositiveMassDrift: number; wetCells: number;
    };
    if (Math.abs(final.relativePositiveMassDrift) > 5e-4) {
      gateErrors.push(`mass drift ${final.relativePositiveMassDrift} exceeds 0.05%`);
    }
    if (final.wetCells < 0.98 * initial.wetCells) {
      gateErrors.push(`represented volume fell from ${initial.wetCells} to ${final.wetCells}`);
    }
    const validation = await device.popErrorScope();
    if (validation) validationErrors.push(validation.message);
    const report = {
      passed: validationErrors.length === 0 && gateErrors.length === 0,
      kind: "sparse-cm12-ocean-volume",
      brickFineResolution, resolutionMode, transport: "compiled-topology", steps,
      dt_s: scene.numerics.maxDt_s,
      initial, trajectory, adaptiveRepresentation, gateErrors, validationErrors,
    };
    await mkdir(dirname(out), { recursive: true });
    await writeFile(out, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  } finally {
    solver.destroy();
  }
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
