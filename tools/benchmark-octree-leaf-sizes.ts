import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { octreeMethod } from "../lib/methods/octree";
import type { GPUSolverInstance } from "../lib/methods/types";
import type { GPUPhysicsTimings } from "../lib/webgpu-eulerian";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

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
const requestedFeatures = optionalFluidDeviceFeatures(adapter.features);
if (adapter.features.has("timestamp-query")) requestedFeatures.push("timestamp-query");
const device = await adapter.requestDevice({
  requiredFeatures: requestedFeatures,
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
assert.ok(device.features.has("timestamp-query"), "leaf-size benchmark requires timestamp-query support");
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

function calmDeepScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "benchmark-octree-leaf-sizes";
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: 1.6,
    height_m: 2.4,
    depth_m: 1.6,
    fillFraction: 0.75,
    top: "open",
    fluidWallMode: "no-slip",
  };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  scene.numerics.surfaceColumnsOverride = 4096;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.005;
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

const timingFields = [
  "layerConstruction_ms",
  "advection_ms",
  "pressure_ms",
  "projection_ms",
  "extrapolation_ms",
  "surfaceUpdate_ms",
  "fluidResidency_ms",
  "sparsePublication_ms",
  "total_ms",
] as const satisfies readonly (keyof GPUPhysicsTimings)[];

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? 0.5 * (sorted[middle - 1] + sorted[middle]) : sorted[middle];
};
const rounded = (value: number) => Number(value.toFixed(3));

const results: Array<Record<string, unknown>> = [];
for (const maximumLeafSize of leafSizes) {
  const scene = calmDeepScene();
  const values = {
    ...octreeMethod.presetFor("balanced"),
    maximumLeafSize: String(maximumLeafSize),
    secondaryParticles: "off",
  };
  const solver = octreeMethod.createSolver!(device, scene, "balanced", values) as GPUSolverInstance;
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [64, 96, 64]);
  const samples = Object.fromEntries(timingFields.map((field) => [field, [] as number[]])) as Record<typeof timingFields[number], number[]>;
  const rowSamples: number[] = [];
  const wallSamples: number[] = [];
  const totalSteps = warmupSteps + sampleSteps;
  for (let step = 1; step <= totalSteps; step += 1) {
    const startedAt = performance.now();
    while (!solver.advanceTo(step * scene.numerics.fixedDt_s!, [])) await new Promise((resolve) => setImmediate(resolve));
    await device.queue.onSubmittedWorkDone();
    const wall_ms = performance.now() - startedAt;
    const info = await solver.readStats();
    if (step <= warmupSteps) continue;
    assert.ok(info.gpuTimings, "timestamp query results were not published");
    for (const field of timingFields) samples[field].push(info.gpuTimings[field]);
    rowSamples.push(info.quadtreeLiquidDofCount ?? info.activeSampleCount ?? 0);
    wallSamples.push(wall_ms);
  }
  const medians = Object.fromEntries(timingFields.map((field) => [field, rounded(median(samples[field]))]));
  results.push({
    maximumLeafSize,
    grid: [solver.info.nx, solver.info.ny, solver.info.nz],
    liquidPressureRows: Math.round(median(rowSamples)),
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
