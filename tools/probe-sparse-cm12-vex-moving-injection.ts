#!/usr/bin/env node
/** Standalone Dawn behavior lane for current-frame VEX moving-solid/injection seeds. */
import { pathToFileURL } from "node:url";

import { createRigidFloatScene } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import {
  SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  sparseCM12VelocityExtensionMaskDensity,
} from "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const INVALID = 0xffff_ffff;
const argument = (name: string, fallback: string): string => process.argv.slice(2)
  .find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const steps = Number(argument("steps", "3"));
if (!Number.isSafeInteger(steps) || steps < 2) {
  throw new RangeError("--steps must be an integer of at least 2");
}

const fieldReceipt = (values: Float32Array) => {
  let nonFinite = 0, maximumAbsolute = 0, positiveSum = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) nonFinite += 1;
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
    positiveSum += Math.max(0, value);
  }
  return { nonFinite, maximumAbsolute, positiveSum };
};

type VexQA = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readVelocityExtensionQA"]>>;
type ActivityQA = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>;
type ResidentAddressSource = {
  readonly brickFineResolution: number;
  readonly templateWords: Uint32Array;
};

function verifyExtendedMask(
  vex: VexQA,
  activity: ActivityQA,
  resident: ResidentAddressSource,
  failures: string[],
) {
  const h = SPARSE_CM12_VELOCITY_EXTENSION_HEADER;
  const finalMask = (SPARSE_CM12_VELOCITY_EXTENSION_DEPTH & 1) === 0
    ? vex.validityA : vex.validityB;
  const packetCapacity = vex.header[h.packetCapacity]!;
  const density = sparseCM12VelocityExtensionMaskDensity(
    finalMask, packetCapacity, vex.dispatchPacketCount);
  const velocity = new Float32Array(vex.velocityBits.buffer,
    vex.velocityBits.byteOffset, vex.velocityBits.length);
  const levels = Math.log2(resident.brickFineResolution) + 1;
  const rangesAt = resident.templateWords[11]!;
  let mappedCells = 0, mappedValidCells = 0, mismatchedCells = 0;
  let invalidDepthCells = 0, nonFiniteVelocityCells = 0;

  for (let brick = 0; brick < activity.bricks.length; brick += 1) {
    const record = activity.bricks[brick]!;
    if (!record.active) continue;
    const resolution = record.acceptedResolution;
    const level = Math.log2(resolution);
    if (!Number.isInteger(level) || level < 0 || level >= levels) {
      failures.push(`brick ${brick}: invalid accepted resolution ${resolution}`);
      continue;
    }
    const range = rangesAt + 2 * (levels * brick + level);
    const first = resident.templateWords[range]!;
    const count = resident.templateWords[range + 1]!;
    if (count !== resolution ** 3) {
      failures.push(`brick ${brick}: expected full ${resolution}^3 range, got ${count}`);
      continue;
    }
    const packetAxis = Math.max(1, Math.ceil(resolution / 4));
    for (let localPacket = 0; localPacket < packetAxis ** 3; localPacket += 1) {
      const pz = Math.floor(localPacket / packetAxis ** 2);
      const remainder = localPacket - pz * packetAxis ** 2;
      const py = Math.floor(remainder / packetAxis), px = remainder - py * packetAxis;
      const packet = 64 * brick + localPacket;
      for (let lane = 0; lane < 64; lane += 1) {
        const lx = lane & 3, ly = (lane >>> 2) & 3, lz = lane >>> 4;
        const x = 4 * px + lx, y = 4 * py + ly, z = 4 * pz + lz;
        if (x >= resolution || y >= resolution || z >= resolution) continue;
        const cell = first + x + resolution * (y + resolution * z);
        const depth = vex.acceptedDepth[cell]!;
        const depthValid = depth !== INVALID && depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH;
        if (depth !== INVALID && depth > SPARSE_CM12_VELOCITY_EXTENSION_DEPTH) {
          invalidDepthCells += 1;
        }
        const at = 4 * cell;
        const finiteVelocity = Number.isFinite(velocity[at])
          && Number.isFinite(velocity[at + 1]) && Number.isFinite(velocity[at + 2])
          && Number.isFinite(velocity[at + 3]);
        if (!finiteVelocity) nonFiniteVelocityCells += 1;
        const velocityValid = finiteVelocity && velocity[at + 3]! > 0.5;
        const word = finalMask[2 * packet + (lane >>> 5)] ?? 0;
        const maskValid = ((word >>> (lane & 31)) & 1) !== 0;
        mappedCells += 1;
        if (maskValid) mappedValidCells += 1;
        if (maskValid !== depthValid || depthValid !== velocityValid) mismatchedCells += 1;
      }
    }
  }
  if (vex.header[h.faultCount] !== 0) {
    failures.push(`VEX fault count ${vex.header[h.faultCount]}`);
  }
  if (invalidDepthCells !== 0 || nonFiniteVelocityCells !== 0 || mismatchedCells !== 0) {
    failures.push(`EXTENDED mismatch/depth/non-finite ${mismatchedCells}/${
      invalidDepthCells}/${nonFiniteVelocityCells}`);
  }
  if (mappedValidCells !== density.validCellCount) {
    failures.push(`EXTENDED mask has ${density.validCellCount} valid lanes but ${
      mappedValidCells} map to accepted cells`);
  }
  if (vex.header[h.validCellCount] !== density.validCellCount
    || vex.header[h.emptyPacketCount] !== density.emptyPacketCount) {
    failures.push("VEX header density disagrees with its final packet mask");
  }
  return {
    sourceFrameGeneration: vex.header[h.sourceFrameGeneration],
    topologyGeneration: vex.header[h.topologyGeneration],
    packetCapacity, dispatchPacketCount: vex.dispatchPacketCount,
    mappedCells, validCells: density.validCellCount,
    emptyPackets: density.emptyPacketCount, mismatchedCells,
    invalidDepthCells, nonFiniteVelocityCells,
  };
}

async function main() {
  const backend = argument("backend", process.env.FLUID_WEBGPU_BACKEND ?? "metal");
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  await acquireWebGPUExclusiveLock("dawn-acceptance",
    "tools/probe-sparse-cm12-vex-moving-injection.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  let validationScopeOpen = false;
  try {
    const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${backend}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error(`no Dawn adapter for ${backend}`);
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });
    device.pushErrorScope("validation");
    validationScopeOpen = true;

    const scene = createRigidFloatScene();
    scene.sceneId = "sparse-cm12-vex-moving-injection";
    const bodies = initializeRigidBodies(scene.rigidBodies);
    const body = bodies[0];
    if (!body) throw new Error("rigid-coupling scene has no body");
    body.held = true;
    const startX = body.position_m.x;
    const dt_s = scene.numerics.maxDt_s;
    const motionPerStep_m = 0.02;
    const injection = { centre_m: { x: 0.22, y: 0.60, z: 0 }, radius_m: 0.09 };
    solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
      device, scene, "balanced", undefined,
      { resolutionMode: "all-fine", brickFineResolution: 8,
        presentationPageResolution: 8, timeStep: "scene" }, () => {});
    const resident = (solver as unknown as { resident: ResidentAddressSource }).resident;
    const initialFields = await solver.readDiagnosticFields();
    const initialDensity = fieldReceipt(initialFields.density);
    const failures: string[] = [];
    const checkpoints: Array<Record<string, unknown>> = [];
    let beforeInjectionPositiveMass = initialDensity.positiveSum;
    let injectedPositiveMass = initialDensity.positiveSum;
    let previousVexGeneration = 0;

    for (let step = 1; step <= steps; step += 1) {
      if (step === 2) {
        beforeInjectionPositiveMass = fieldReceipt(
          (await solver.readDiagnosticFields()).density).positiveSum;
        solver.injectLiquidBall(injection);
        await device.queue.onSubmittedWorkDone();
        injectedPositiveMass = fieldReceipt(
          (await solver.readDiagnosticFields()).density).positiveSum;
        if (!(injectedPositiveMass > beforeInjectionPositiveMass)) {
          failures.push("liquid injection did not increase represented positive mass");
        }
      }
      body.position_m.x = startX + motionPerStep_m * step;
      body.linearVelocity_m_s.x = motionPerStep_m / dt_s;
      if (!solver.advanceTo(step * dt_s, bodies)) {
        failures.push(`step ${step}: advance did not encode`);
        break;
      }
      await device.queue.onSubmittedWorkDone();
      const [fields, vex, activity] = await Promise.all([
        solver.readDiagnosticFields(), solver.readVelocityExtensionQA(),
        solver.readGPUActivityPolicy(),
      ]);
      const fieldReceipts = {
        density: fieldReceipt(fields.density), gamma: fieldReceipt(fields.gamma),
        velocity: fieldReceipt(fields.velocity), pressure: fieldReceipt(fields.pressure),
        divergence: fieldReceipt(fields.divergence),
      };
      for (const [name, receipt] of Object.entries(fieldReceipts)) {
        if (receipt.nonFinite !== 0) failures.push(
          `step ${step}: ${name} has ${receipt.nonFinite} non-finite values`);
      }
      const extended = verifyExtendedMask(vex, activity, resident, failures);
      if (extended.sourceFrameGeneration <= previousVexGeneration) failures.push(
        `step ${step}: VEX generation did not advance (${extended.sourceFrameGeneration})`);
      previousVexGeneration = extended.sourceFrameGeneration ?? 0;
      checkpoints.push({ step, phase: step === 1 ? "body-motion" : step === 2
        ? "post-injection" : "body-motion-after-injection",
      bodyX_m: body.position_m.x, extended, fields: fieldReceipts });
    }

    const scopedError = await device.popErrorScope();
    validationScopeOpen = false;
    if (scopedError) validationErrors.push(scopedError.message);
    failures.push(...validationErrors.map((message) => `WebGPU validation: ${message}`));
    const report = {
      passed: failures.length === 0,
      kind: "sparse-cm12-vex-moving-injection", backend, steps, dt_s,
      bodyMotion_m: bodies[0]!.position_m.x - startX,
      injection: { ...injection, positiveMassBefore: beforeInjectionPositiveMass,
        positiveMassAfter: injectedPositiveMass },
      checkpoints, validationErrors, failures,
    };
    console.log(JSON.stringify(report));
    if (!report.passed) process.exitCode = 1;
  } finally {
    if (validationScopeOpen) await device?.popErrorScope().catch(() => null);
    solver?.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
}

await main().catch((error: unknown) => {
  console.error(JSON.stringify({ passed: false, kind: "sparse-cm12-vex-moving-injection",
    error: error instanceof Error ? error.message : String(error) }));
  process.exitCode = 1;
});
