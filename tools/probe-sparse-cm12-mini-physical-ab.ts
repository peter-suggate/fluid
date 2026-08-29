/**
 * Stepwise physical-parity probe for the same mini dam represented as:
 * A. the 32^3 authored lattice with a two-fine-cell minimum-size region; and
 * B. the 16^3 authored lattice.
 *
 * The A fields are conservatively restricted to the 16^3 physical lattice
 * before comparison. This intentionally leaves the brick partition different.
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreakScene,
} from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;
const steps = Number(argument("steps", "8"));
if (!Number.isSafeInteger(steps) || steps < 0) throw new RangeError(
  "steps must be a non-negative integer",
);
const dt_s = 0.004;
const stageLimit = argument("stage-limit", "");
const stageFromStep = Number(argument("stage-from-step", "1"));
const sharpeningLimit = argument("sharpening-limit", "");
const topologyMode = argument("topology", "adaptive");
if (topologyMode !== "adaptive" && topologyMode !== "matched-uniform") {
  throw new RangeError("topology must be adaptive or matched-uniform");
}
const FIELD_NAMES = ["density", "gamma", "sharpeningDelta",
  "sharpeningReceiptMass", "solidOpenFraction", "velocity", "pressureRhs",
  "pressure", "divergence"] as const;
type FieldName = typeof FIELD_NAMES[number];
type DiagnosticFields = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>;

interface Snapshot {
  readonly step: number;
  readonly fields: DiagnosticFields;
  readonly topology: {
    readonly residentBricks: number;
    readonly resolutionHistogram: Readonly<Record<string, number>>;
    readonly bricks: readonly Readonly<Record<string, unknown>>[];
  };
  readonly stats: Readonly<Record<string, number | string | boolean | undefined>>;
}

function physicalRegion(scene: ReturnType<typeof createMinimalPowerDamBreak32Scene>) {
  scene.fluid.refinementRegions = [{
    id: "mini-physical-parity-two-cell-floor",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 2,
    min_m: { x: -0.5 * scene.container.width_m, y: 0,
      z: -0.5 * scene.container.depth_m },
    max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
      z: 0.5 * scene.container.depth_m },
  }];
  return scene;
}

async function runArm(
  device: GPUDevice,
  scene: ReturnType<typeof createMinimalPowerDamBreakScene>,
  resolutionMode: "adaptive" | "all-fine" | "all-coarse",
): Promise<readonly Snapshot[]> {
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    timeStep: "scene",
    resolutionMode,
    brickFineResolution: "8",
    presentationPageResolution: "8",
    pressureIterations: Number(argument("pressure-iterations", "40")),
    pressureRelativeTolerance: Number(argument("pressure-tolerance", "0.001")),
    surfaceSharpening: argument("surface-sharpening", "on"),
  });
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {},
  ) as WebGPUAdaptiveMassSolver;
  const snapshots: Snapshot[] = [];
  try {
    for (let step = 0; step <= steps; step += 1) {
      if (step > 0) {
        if (stageLimit !== "" && step >= stageFromStep) {
          solver.sparseWorldTrace.setStageLimitForQA(stageLimit as never);
        }
        if (sharpeningLimit !== "" && step >= stageFromStep) {
          solver.sparseWorldTrace.setSharpeningPhaseLimitForQA(sharpeningLimit as never);
        }
        while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
        await device.queue.onSubmittedWorkDone();
      }
      const [fields, activity, stats] = await Promise.all([
        solver.readDiagnosticFields(false, stageLimit !== "" && step >= stageFromStep
          ? "candidate" : "accepted"),
        solver.readGPUActivityPolicy(), solver.readStats(),
      ]);
      const active = activity.bricks.filter((brick) => brick.active);
      const resolutions = [...new Set(active.map((brick) => brick.acceptedResolution))]
        .sort((left, right) => left - right);
      snapshots.push({
        step,
        fields,
        topology: {
          residentBricks: activity.residentBrickCount,
          resolutionHistogram: Object.fromEntries(resolutions.map((resolution) => [
            String(resolution),
            active.filter((brick) => brick.acceptedResolution === resolution).length,
          ])),
          bricks: active.map((brick) => ({
            coordinate: brick.coordinate,
            resolution: brick.acceptedResolution,
            planned: brick.plannedResolution,
            reasons: brick.reasons,
            planReasons: brick.planReasons,
            meanDensity: brick.meanDensity,
          })),
        },
        stats: {
          acceptedCells: stats.adaptiveAcceptedCellCount,
          acceptedRows: stats.adaptiveAcceptedRowCount,
          pressureCells: stats.adaptivePressureCellCount,
          pressureRows: stats.adaptivePressureActiveRowCount,
          pressureIterations: stats.pressureIterationsExecuted,
          pressureRelativeResidual: stats.pressureRelativeResidual,
          maxDivergenceAfter_s: stats.maxDivergenceAfter_s,
          maximumSpeed_m_s: stats.maxSpeed_m_s,
          topologyGeneration: stats.adaptiveTopologyShadowGeneration,
          preparedBricks: stats.adaptiveTopologyPreparedBrickCount,
          committedBricks: stats.adaptiveTopologyCommittedBrickCount,
        },
      });
    }
    return snapshots;
  } finally {
    solver.destroy();
  }
}

function restrictScalar2(source: Float32Array): Float32Array {
  const result = new Float32Array(16 ** 3);
  for (let z = 0; z < 16; z += 1) for (let y = 0; y < 16; y += 1) {
    for (let x = 0; x < 16; x += 1) {
      let sum = 0;
      for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
        for (let dx = 0; dx < 2; dx += 1) {
          sum += source[2 * x + dx + 32 * ((2 * y + dy) + 32 * (2 * z + dz))]!;
        }
      }
      result[x + 16 * (y + 16 * z)] = sum / 8;
    }
  }
  return result;
}

function restrictField2(name: FieldName, source: Float32Array): Float32Array {
  if (name !== "velocity") {
    const restricted = restrictScalar2(source);
    // RHS is an integrated control-volume flux in authored finest-cell units.
    // One Mini16 unit is the volume of eight Mini32 units.
    if (name === "pressureRhs") {
      for (let index = 0; index < restricted.length; index += 1) {
        restricted[index] /= 8;
      }
    }
    return restricted;
  }
  const result = new Float32Array(4 * 16 ** 3);
  for (let component = 0; component < 4; component += 1) {
    const scalar = new Float32Array(32 ** 3);
    for (let index = 0; index < scalar.length; index += 1) {
      scalar[index] = source[4 * index + component]!;
    }
    const restricted = restrictScalar2(scalar);
    for (let index = 0; index < restricted.length; index += 1) {
      result[4 * index + component] = restricted[index]!;
    }
  }
  return result;
}

function difference(
  left: Float32Array,
  right: Float32Array,
  liquidMask?: Float32Array,
) {
  assert.equal(left.length, right.length);
  let absolute = 0, squared = 0, referenceAbsolute = 0, referenceSquared = 0;
  let maximumAbsolute = 0, differing = 0, firstDiffering = -1;
  let maximumIndex = -1;
  const examples: Array<{ index: number; left: number; right: number }> = [];
  for (let index = 0; index < left.length; index += 1) {
    const cell = liquidMask && left.length === 4 * liquidMask.length
      ? Math.floor(index / 4) : index;
    if (liquidMask && liquidMask[cell]! <= 1e-6) continue;
    const delta = left[index]! - right[index]!;
    absolute += Math.abs(delta);
    squared += delta * delta;
    referenceAbsolute += Math.abs(right[index]!);
    referenceSquared += right[index]! ** 2;
    if (Math.abs(delta) > maximumAbsolute) {
      maximumAbsolute = Math.abs(delta);
      maximumIndex = index;
    }
    if (delta !== 0) {
      differing += 1;
      if (firstDiffering < 0) firstDiffering = index;
      if (examples.length < 24) examples.push({
        index, left: left[index]!, right: right[index]!,
      });
    }
  }
  return {
    relativeL1: absolute / Math.max(referenceAbsolute, 1e-30),
    relativeL2: Math.sqrt(squared / Math.max(referenceSquared, 1e-30)),
    maximumAbsolute,
    differing,
    firstDiffering,
    firstValues: firstDiffering < 0 ? undefined
      : { left: left[firstDiffering], right: right[firstDiffering] },
    maximumIndex,
    maximumValues: maximumIndex < 0 ? undefined
      : { left: left[maximumIndex], right: right[maximumIndex] },
    examples,
  };
}

function scalarSum(values: Float32Array): number {
  let sum = 0, correction = 0;
  for (const value of values) {
    const adjusted = value - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

await acquireWebGPUExclusiveLock("dawn-acceptance",
  "tools/probe-sparse-cm12-mini-physical-ab.ts");
let device: GPUDevice | undefined;
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();
    validationErrors.push(event.error.message);
  });
  const mini32 = await runArm(device, physicalRegion(createMinimalPowerDamBreak32Scene()),
    topologyMode === "adaptive" ? "adaptive" : "all-coarse");
  const mini16 = await runArm(device, createMinimalPowerDamBreakScene(),
    topologyMode === "adaptive" ? "adaptive" : "all-fine");
  const trajectory = mini32.map((fine, step) => {
    const restricted = Object.fromEntries(FIELD_NAMES.map((name) => [
      name, restrictField2(name, fine.fields[name]),
    ])) as Record<FieldName, Float32Array>;
    const liquidMask = Float32Array.from(restricted.density, (density, index) =>
      Math.max(density, mini16[step]!.fields.density[index]!));
    const bulkLiquidMask = Float32Array.from(liquidMask, (density) =>
      density >= 0.5 ? density : 0);
    return {
      step,
      time_s: step * dt_s,
      mini32Topology: fine.topology,
      mini16Topology: mini16[step]!.topology,
      mini32Stats: fine.stats,
      mini16Stats: mini16[step]!.stats,
      physicalMass_m3: {
        mini32Restricted: scalarSum(restricted.density) * 0.05 ** 3,
        mini16: scalarSum(mini16[step]!.fields.density) * 0.05 ** 3,
        difference: (scalarSum(restricted.density)
          - scalarSum(mini16[step]!.fields.density)) * 0.05 ** 3,
      },
      fields: Object.fromEntries(FIELD_NAMES.map((name) => [name, difference(
        restricted[name], mini16[step]!.fields[name],
      )])),
      liquidFields: Object.fromEntries(FIELD_NAMES.map((name) => [name, difference(
        restricted[name], mini16[step]!.fields[name], liquidMask,
      )])),
      bulkLiquidFields: Object.fromEntries(FIELD_NAMES.map((name) => [name, difference(
        restricted[name], mini16[step]!.fields[name], bulkLiquidMask,
      )])),
    };
  });
  console.log(JSON.stringify({
    probe: "sparse-cm12-mini-physical-ab",
    arms: {
      A: "mini32 + whole-domain minimumCellSize_cells=2",
      B: "mini16",
    },
    dt_s,
    topologyMode,
    stageLimit: stageLimit || undefined,
    trajectory,
    validationErrors,
  }, null, 2));
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
