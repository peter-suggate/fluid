import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { octreeMethod } from "../lib/methods/octree";
import type { GPUSolverInstance } from "../lib/methods/types";
import { getScenePreset } from "../lib/scenes";
import type { OctreeTopologyLeafCensus } from "../lib/webgpu-octree";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { compareScalarFields } from "./webgpu-smoke-scenarios";
import { readCubicVolumeField } from "./webgpu-smoke-readbacks";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "./webgpu-smoke-isolation";

interface Result {
  readonly gated: boolean;
  readonly construction_ms: number;
  readonly topology: OctreeTopologyLeafCensus;
  readonly field: Float32Array;
  readonly pressureRows: number;
  readonly pressureIterations: number;
  readonly representedVolumeCellSum: number;
}

const rounded = (value: number, digits = 3) => Number(value.toFixed(digits));
const coarseCount = (counts: Readonly<Record<string, number>>) =>
  Object.entries(counts).reduce((total, [size, count]) =>
    total + (Number(size) > 1 ? count : 0), 0);

await acquireWebGPUExclusiveLock(
  "dawn-benchmark",
  "tools/benchmark-water-box-boundary-adaptivity.ts",
);
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
    ...(process.env.FLUID_WEBGPU_ADAPTER
      ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : []),
  ]);
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu },
  });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  assert.ok(adapter.features.has("subgroups"),
    "water-box boundary benchmark requires a subgroup-capable adapter");
  const requiredFeatures: GPUFeatureName[] = ["subgroups"];
  if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  const results: Result[] = [];
  for (const gated of [false, true]) {
    const scene = getScenePreset("water-box-dam-break").create();
    const values = {
      ...octreeMethod.presetFor("balanced"),
      maximumLeafSize: "32",
      globalFineLevelSetFactor: "1",
      interfaceRefinementBandCells: 3,
      fluidGatedBoundaryRefinement: gated,
    };
    const started = performance.now();
    const solver = await octreeMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {},
    ) as GPUSolverInstance;
    await device.queue.onSubmittedWorkDone();
    const construction_ms = performance.now() - started;
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [24, 18, 16]);
    const projection = (solver as unknown as {
      octreeProjection?: {
        readSolveDiagnostics(): Promise<void>;
        readTopologyLeafCensus(): Promise<OctreeTopologyLeafCensus>;
      };
    }).octreeProjection;
    assert.ok(projection);
    await projection.readSolveDiagnostics();
    const topology = await projection.readTopologyLeafCensus();
    const field = await readCubicVolumeField(device, solver, true);
    const info = await solver.readStats();
    results.push({
      gated,
      construction_ms,
      topology,
      field: field.field,
      pressureRows: info.pressureRequiredRows ?? 0,
      pressureIterations: info.quadtreePressureIterationsUsed ?? 0,
      representedVolumeCellSum: info.representedVolumeCellSum ?? 0,
    });
    solver.destroy();
  }

  const control = results.find(({ gated }) => !gated)!;
  const adaptive = results.find(({ gated }) => gated)!;
  const difference = compareScalarFields(adaptive.field, control.field, 24, 18, 16);
  const report = results.map((result) => ({
    gated: result.gated,
    construction_ms: rounded(result.construction_ms),
    topologyLeaves: result.topology.topologyLeaves,
    leafCountsBySize: result.topology.leafCountsBySize,
    boundaryStripLeafCountsBySize: result.topology.boundaryStripLeafCountsBySize,
    pressureRows: result.pressureRows,
    pressureIterations: result.pressureIterations,
    representedVolumeCellSum: rounded(result.representedVolumeCellSum, 6),
    fieldDifference: result.gated ? difference : null,
  }));
  console.log(JSON.stringify({
    phase: "water-box-boundary-adaptivity-benchmark",
    grid: [24, 18, 16],
    surfaceTrackingFactor: 1,
    report,
    validationErrors,
  }));

  assert.equal(difference.meanAbsoluteError, 0,
    "dry boundary coarsening must preserve the initial water-box field exactly");
  assert.equal(adaptive.pressureRows, control.pressureRows,
    "dry boundary coarsening must preserve the initial water-box pressure rows");
  assert.equal(adaptive.representedVolumeCellSum, control.representedVolumeCellSum,
    "dry boundary coarsening must preserve represented water-box volume");
  assert.ok(adaptive.topology.topologyLeaves < control.topology.topologyLeaves,
    "the adaptive water-box must remove dry structural leaves");
  for (const face of ["xHigh", "zHigh"] as const) {
    assert.ok(coarseCount(adaptive.topology.boundaryStripLeafCountsBySize[face]) > 0,
      `the dry far ${face} wall must contain coarse leaves`);
  }
  assert.deepEqual(validationErrors, []);
  device.destroy();
} finally {
  await releaseWebGPUExclusiveLock();
}
