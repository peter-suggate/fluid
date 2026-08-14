import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
// Composition root for this entry point: importing the method catalog installs
// the simulation methods and the octree coarse-dynamics lanes, without which
// constructing a solver throws rather than silently running the wrong backend.
import "../lib/methods";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";
import { losassoMethod } from "../lib/methods/losasso/method";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import type { PaperPhaseId, PerformanceTrace } from "../lib/core/performance-trace";
import { usePerformanceInstrumentationStore } from
  "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

await acquireWebGPUExclusiveLock("dawn-benchmark", "tools/benchmark-octree-leaf-sizes.ts");
try {
usePerformanceInstrumentationStore.getState().setEnabled(true);
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "WebGPU did not expose an adapter");
const requestedFeatures: GPUFeatureName[] = ["subgroups"];
if (adapter.features.has("timestamp-query")) requestedFeatures.push("timestamp-query");
const device = await adapter.requestDevice({
  requiredFeatures: requestedFeatures,
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
assert.ok(device.features.has("timestamp-query"), "leaf-size benchmark requires timestamp-query support");
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

/** Widens the calm tank laterally (default 1) so the stable deep interior
 * spans many topology tiles, the regime vast-ocean scaling cares about. */
const tankScale = Number(process.env.FLUID_BENCH_TANK_SCALE ?? 1);
assert.ok(Number.isInteger(tankScale) && tankScale >= 1 && tankScale <= 8);

function calmDeepScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "benchmark-octree-leaf-sizes";
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: 1.6 * tankScale,
    height_m: 2.4,
    depth_m: 1.6 * tankScale,
    fillFraction: 0.75,
    top: "open",
    fluidWallMode: "no-slip",
  };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  scene.voxelDomain.finestCellSize_m = 0.025;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.005;
  const refinementFloor = Number(process.env.FLUID_REFINEMENT_REGION_FLOOR ?? 0);
  if (refinementFloor > 0) {
    assert.ok(Number.isSafeInteger(refinementFloor)
      && (refinementFloor & (refinementFloor - 1)) === 0);
    scene.fluid.refinementRegions = [{
      id: "benchmark-full-domain-floor",
      rule: "minimum-cell-size",
      minimumCellSize_cells: refinementFloor,
      min_m: { x: -0.5 * scene.container.width_m, y: 0,
        z: -0.5 * scene.container.depth_m },
      max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
        z: 0.5 * scene.container.depth_m },
    }];
  }
  return scene;
}

const leafSizes = (process.env.FLUID_LEAF_SIZES ?? "2,4,8,16,32")
  .split(",")
  .map(Number)
  .filter((size) => [2, 4, 8, 16, 32].includes(size));
assert.ok(leafSizes.length > 0, "FLUID_LEAF_SIZES must contain one of 2,4,8,16,32");
const warmupSteps = Number(process.env.FLUID_BENCHMARK_WARMUP_STEPS ?? 3);
const sampleSteps = Number(process.env.FLUID_BENCHMARK_SAMPLE_STEPS ?? 7);
assert.ok(Number.isInteger(warmupSteps) && warmupSteps >= 1);
assert.ok(Number.isInteger(sampleSteps) && sampleSteps >= 1);

const phaseIds = [
  "coarse-grid",
  "power-topology",
  "velocity-advection",
  "pressure-system",
  "pressure-solve",
  "velocity-projection",
  "velocity-extrapolation",
  "fine-sdf-advection",
  "fine-sdf-redistance",
  "adaptive-publication",
  "other",
] as const satisfies readonly PaperPhaseId[];
type TraceField = typeof phaseIds[number] | "total";
const timingFields: readonly TraceField[] = [...phaseIds, "total"];

const phaseDuration_ms = (trace: PerformanceTrace, field: TraceField) => field === "total"
  ? trace.total_ms
  : trace.phases.filter((phase) => phase.id === field)
    .reduce((sum, phase) => sum + phase.duration_ms, 0);

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? 0.5 * (sorted[middle - 1] + sorted[middle]) : sorted[middle];
};
const rounded = (value: number) => Number(value.toFixed(3));

// Optional in-process A/B: "|"-separated arms of ";"-separated KEY=VALUE env
// assignments, applied round-robin ahead of each solver build. Interleaving
// arms inside one process keeps GPU clock/thermal state comparable.
const abArms = (process.env.FLUID_AB_ENV ?? "").split("|").filter(Boolean);

const results: Array<Record<string, unknown>> = [];
for (const [configIndex, maximumLeafSize] of leafSizes.entries()) {
  const arm = abArms.length > 0 ? abArms[configIndex % abArms.length] : undefined;
  for (const assignment of arm?.split(";") ?? []) {
    const [key, value] = assignment.split("=");
    process.env[key] = value;
  }
  const scene = calmDeepScene();
  const values = {
    ...losassoMethod.presetFor("balanced"),
    maximumLeafSize: String(maximumLeafSize),
    secondaryParticles: "off",
  };
  const solver = await losassoMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {}) as GPUSolverInstance;
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [64 * tankScale, 96, 64 * tankScale]);
  const samples = Object.fromEntries(timingFields.map((field) => [field, [] as number[]])) as Record<typeof timingFields[number], number[]>;
  const rowSamples: number[] = [];
  const iterationSamples: number[] = [];
  const wallSamples: number[] = [];
  const totalSteps = warmupSteps + sampleSteps;
  for (let step = 1; step <= totalSteps; step += 1) {
    const startedAt = performance.now();
    while (!solver.advanceTo(step * scene.numerics.fixedDt_s!, [])) await new Promise((resolve) => setImmediate(resolve));
    await device.queue.onSubmittedWorkDone();
    // The timestamp map resolves asynchronously after queue completion and
    // publishes through a promise continuation. Give that continuation one
    // host turn before sampling `solver.info`, otherwise this benchmark reads
    // the cold trace whose only interval is the unclassified setup tail.
    await new Promise((resolve) => setImmediate(resolve));
    const wall_ms = performance.now() - startedAt;
    const info = await solver.readStats();
    if (step <= warmupSteps) continue;
    assert.ok(info.physicsTrace, "generic physics trace was not published");
    for (const field of timingFields) samples[field].push(phaseDuration_ms(info.physicsTrace, field));
    rowSamples.push(info.quadtreeLiquidDofCount ?? info.activeSampleCount ?? 0);
    iterationSamples.push(info.quadtreePressureIterationsUsed ?? 0);
    wallSamples.push(wall_ms);
  }
  const medians = Object.fromEntries(timingFields.map((field) => [`${field}_ms`, rounded(median(samples[field]))]));
  results.push({
    ...(arm !== undefined ? { arm } : {}),
    maximumLeafSize,
    grid: [solver.info.nx, solver.info.ny, solver.info.nz],
    liquidPressureRows: Math.round(median(rowSamples)),
    solvePasses: Math.round(median(iterationSamples)),
    wall_ms: rounded(median(wallSamples)),
    ...medians,
  });
  solver.destroy();
}

const leaf2 = results.find((result) => result.maximumLeafSize === 2);
for (const result of results) {
  const baseline = Number(leaf2?.total_ms ?? results[0].total_ms);
  result.speedupVsSmallest = rounded(baseline / Number(result.total_ms));
}
console.table(results);
console.log(JSON.stringify({
  phase: "octree-leaf-size-benchmark",
  warmupSteps,
  sampleSteps,
  results,
  validationErrors,
}));
assert.deepEqual(validationErrors, [], `WebGPU validation errors: ${validationErrors.join("; ")}`);
device.destroy();
} finally {
  usePerformanceInstrumentationStore.getState().setEnabled(false);
  await releaseWebGPUExclusiveLock();
}
