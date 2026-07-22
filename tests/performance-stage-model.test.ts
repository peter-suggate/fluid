import assert from "node:assert/strict";
import test from "node:test";
import { adaptiveTopologyPerformanceStages, physicsPerformanceStages } from "../lib/performance-stage-model";
import { emptyPerformance, measuredGPUTime_ms, type PerformanceSnapshot } from "../lib/stores/diagnostics-store";
import {
  categorizedGPUPhysicsTime_ms,
  decodeGPUPhysicsTimestampSegments,
  emptyGPUPhysicsTimings,
  GPU_PHYSICS_TIMESTAMP_CAPACITY,
  type GPUPhysicsStageId,
  type GPUPhysicsTimestampSegment,
} from "../lib/webgpu-eulerian";

const snapshot = (methodId: string, activeStages: GPUPhysicsStageId[]): PerformanceSnapshot => ({
  ...emptyPerformance,
  methodId,
  gpuPhysicsTimingAvailable: true,
  gpuActiveStages: activeStages,
  gpuPreparation_ms: methodId === "tall-cell" ? 1 : 0,
  gpuLayerConstruction_ms: methodId === "quadtree-tall-cell" || methodId === "octree" ? 2 : 0,
  gpuAdvection_ms: 3,
  gpuConditioning_ms: methodId === "uniform" ? 4 : 0,
  gpuRemeshing_ms: methodId === "tall-cell" ? 5 : 0,
  gpuPressure_ms: 6,
  gpuPowerAssembly_ms: methodId === "octree" ? 2 : 0,
  gpuPressureSolve_ms: methodId === "octree" ? 4 : 0,
  gpuProjection_ms: methodId === "quadtree-tall-cell" ? 0 : 7,
  gpuPowerProjection_ms: methodId === "octree" ? 3 : 0,
  gpuVelocityProjection_ms: methodId === "octree" ? 4 : 0,
  gpuFaceBand_ms: methodId === "octree" ? 1 : 0,
  gpuFaceMarch_ms: methodId === "octree" ? 1 : 0,
  gpuPowerPublication_ms: methodId === "octree" ? 1 : 0,
  gpuExtrapolation_ms: methodId === "octree" ? 12 : 0,
  gpuMaterialization_ms: methodId === "octree" ? 13 : 0,
  gpuSurfaceUpdate_ms: methodId === "quadtree-tall-cell" || methodId === "octree" ? 8 : 0,
  gpuFineTopology_ms: methodId === "octree" ? 3 : 0,
  gpuFineTransport_ms: methodId === "octree" ? 2 : 0,
  gpuFineRedistance_ms: methodId === "octree" ? 3 : 0,
  gpuRigid_ms: 9,
  gpuSpraySimulation_ms: methodId === "octree" ? 14 : 0,
  gpuFluidResidency_ms: methodId === "octree" ? 15 : 0,
  gpuSparsePublication_ms: methodId === "octree" ? 16 : 0,
  gpuDiagnostics_ms: 10,
  gpuOverhead_ms: 11
});

const stagesFor = (methodId: string, activeStages: GPUPhysicsStageId[]) => {
  const sample = snapshot(methodId, activeStages);
  return { sample, stages: physicsPerformanceStages({ methodId, snapshot: sample, contextMatches: true, pressureSolver: "test solver" }) };
};

test("uniform trace exposes VOF conditioning and preserves queue order", () => {
  const { sample, stages } = stagesFor("uniform", ["advection", "conditioning", "pressure", "projection", "rigidCoupling", "diagnostics"]);
  assert.deepEqual(stages.map((stage) => stage.key), ["advection", "conditioning", "pressure", "projection", "rigid", "diagnostics", "overhead"]);
  assert.deepEqual(stages.map((stage) => stage.dependsOn[0]), ["uploads", "advection", "conditioning", "pressure", "projection", "rigid", "diagnostics"]);
  assert.equal(stages.reduce((sum, stage) => sum + stage.value, 0), measuredGPUTime_ms(sample));
});

test("restricted tall-cell trace exposes preparation and cadence-driven remeshing before its pressure solve", () => {
  const { sample, stages } = stagesFor("tall-cell", ["preparation", "advection", "remeshing", "rigidCoupling", "pressure", "projection", "diagnostics"]);
  assert.deepEqual(stages.map((stage) => stage.key), ["preparation", "advection", "remesh", "rigid", "pressure", "projection", "diagnostics", "overhead"]);
  assert.deepEqual(stages.find((stage) => stage.key === "pressure")?.dependsOn, ["rigid"]);
  assert.equal(stages.reduce((sum, stage) => sum + stage.value, 0), measuredGPUTime_ms(sample));
});

test("quadtree trace exposes inline topology and post-projection surface maintenance", () => {
  const { sample, stages } = stagesFor("quadtree-tall-cell", ["topology", "advection", "pressure", "surfaceUpdate", "rigidCoupling", "diagnostics"]);
  assert.deepEqual(stages.map((stage) => stage.key), ["topology", "advection", "pressure", "surface-update", "rigid", "diagnostics", "overhead"]);
  assert.deepEqual(stages.find((stage) => stage.key === "advection")?.dependsOn, ["topology"]);
  assert.deepEqual(stages.find((stage) => stage.key === "surface-update")?.dependsOn, ["pressure"]);
  assert.equal(stages.reduce((sum, stage) => sum + stage.value, 0), measuredGPUTime_ms(sample));
});

test("octree trace exposes its resident pipeline and immersed-body coupling", () => {
  const { sample, stages } = stagesFor("octree", ["topology", "advection", "pressure", "projection", "extrapolation", "materialization", "surfaceUpdate", "rigidCoupling", "spray", "fluidResidency", "sparsePublication", "diagnostics"]);
  assert.deepEqual(stages.map((stage) => stage.key), ["topology", "advection", "pressure", "projection", "extrapolation", "materialization", "surface-update", "rigid", "spray-sim", "fluid-residency", "sparse-publication", "diagnostics", "overhead"]);
  assert.deepEqual(stages.map((stage) => stage.dependsOn[0]), ["uploads", "topology", "advection", "pressure", "projection", "extrapolation", "materialization", "surface-update", "rigid", "spray-sim", "fluid-residency", "sparse-publication", "diagnostics"]);
  assert.equal(stages.find((stage) => stage.key === "spray-sim")?.label, "Spray breakup + transport");
  assert.equal(stages.find((stage) => stage.key === "fluid-residency")?.label, "Fluid brick residency");
  assert.equal(stages.find((stage) => stage.key === "sparse-publication")?.label, "Sparse scene fluid publication");
  assert.equal(stages.reduce((sum, stage) => sum + stage.value, 0), measuredGPUTime_ms(sample));
});

test("power-diagram timestamps split assembly, solve, and projection without double-counting", () => {
  const { sample, stages } = stagesFor("octree", ["topology", "advection", "pressure", "powerAssembly", "pressureSolve", "projection", "powerProjection", "velocityProjection", "extrapolation", "materialization", "surfaceUpdate", "rigidCoupling", "spray", "fluidResidency", "sparsePublication", "diagnostics"]);
  assert.deepEqual(stages.slice(2, 6).map((stage) => stage.key), ["power-assembly", "pressure", "power-projection", "projection"]);
  assert.deepEqual(stages.slice(2, 6).map((stage) => stage.value), [2, 4, 3, 4]);
  assert.deepEqual(stages.slice(2, 6).map((stage) => stage.dependsOn[0]), ["advection", "power-assembly", "pressure", "power-projection"]);
  assert.equal(stages.reduce((sum, stage) => sum + stage.value, 0), measuredGPUTime_ms(sample));
});

test("Section 5 timestamps replace their aggregates without double-counting", () => {
  const activeStages: GPUPhysicsStageId[] = ["topology", "advection", "pressure", "powerAssembly", "pressureSolve",
    "projection", "powerProjection", "velocityProjection", "faceBand", "faceMarch", "powerPublication",
    "extrapolation", "materialization", "surfaceUpdate", "fineTopology", "fineTransport", "fineRedistance",
    "rigidCoupling", "spray", "fluidResidency", "sparsePublication", "diagnostics"];
  const { sample, stages } = stagesFor("octree", activeStages);
  assert.deepEqual(stages.filter((stage) => ["face-band", "face-march", "power-publication"].includes(stage.key))
    .map((stage) => [stage.key, stage.value, stage.dependsOn[0]]), [
      ["face-band", 1, "pressure"],
      ["face-march", 1, "face-band"],
      ["power-publication", 1, "face-march"],
    ]);
  assert.deepEqual(stages.filter((stage) => ["fine-transport", "fine-topology", "fine-redistance"].includes(stage.key))
    .map((stage) => [stage.key, stage.value, stage.dependsOn[0]]), [
      ["fine-transport", 2, "materialization"],
      ["fine-topology", 3, "fine-transport"],
      ["fine-redistance", 3, "fine-topology"],
    ]);
  assert.equal(stages.some((stage) => stage.key === "power-projection" || stage.key === "surface-update"), false);
  assert.equal(stages.reduce((sum, stage) => sum + stage.value, 0), measuredGPUTime_ms(sample));
});

test("asynchronous quadtree topology is not repeated in the per-advance physics total", () => {
  const sample = snapshot("quadtree-tall-cell", ["advection", "pressure", "surfaceUpdate", "diagnostics"]);
  const stages = physicsPerformanceStages({ methodId: "quadtree-tall-cell", snapshot: sample, contextMatches: true, topologyPath: "async" });
  assert.equal(stages.some((stage) => stage.key === "topology"), false);
  assert.deepEqual(stages.find((stage) => stage.key === "advection")?.dependsOn, ["uploads"]);
  assert.equal(stages.reduce((sum, stage) => sum + stage.value, 0), measuredGPUTime_ms(sample) - sample.gpuLayerConstruction_ms);
});

test("asynchronous topology exposes every measured rebuild phase", () => {
  const sample = snapshot("quadtree-tall-cell", []);
  Object.assign(sample, {
    adaptiveRebuildCompletedCount: 1,
    adaptiveGPUConstructionKernel_ms: 1,
    adaptiveGPUSparsePack_ms: 2,
    adaptiveCPUTopologyPack_ms: 3,
    adaptiveCPURedistance_ms: 4,
    adaptiveCPUQuadtreeDecode_ms: 5,
    adaptiveCPUTallGrid_ms: 6,
    adaptiveCPUVariationalAssembly_ms: 7,
    adaptiveCPUSystemPack_ms: 8,
    adaptiveCPUICFactorization_ms: 9,
    adaptiveCPUResourceUpload_ms: 10
  });
  const stages = adaptiveTopologyPerformanceStages({ snapshot: sample, contextMatches: true });
  assert.deepEqual(stages.map((stage) => stage.key), ["adaptive-gpu-build", "adaptive-gpu-pack", "adaptive-topology-pack", "adaptive-redistance", "adaptive-decode", "adaptive-tall-grid", "adaptive-assembly", "adaptive-system-pack", "adaptive-factor", "adaptive-upload"]);
  assert.equal(stages.reduce((sum, stage) => sum + stage.value, 0), 55);
  assert.ok(stages.every((stage) => stage.timer === "async" && stage.active));
});

test("conditional method stages remain visible when idle", () => {
  const quadtree = stagesFor("quadtree-tall-cell", ["advection", "pressure", "surfaceUpdate", "diagnostics"]).stages;
  const tall = stagesFor("tall-cell", ["preparation", "advection", "rigidCoupling", "pressure", "projection", "diagnostics"]).stages;
  assert.equal(quadtree.find((stage) => stage.key === "topology")?.active, false);
  assert.equal(tall.find((stage) => stage.key === "remesh")?.active, false);
});

test("expanded physics accounting includes every named category", () => {
  const timings = emptyGPUPhysicsTimings();
  Object.assign(timings, { preparation_ms: 1, layerConstruction_ms: 2, advection_ms: 3, conditioning_ms: 4, remeshing_ms: 5, pressure_ms: 6, projection_ms: 7, surfaceUpdate_ms: 8, rigidCoupling_ms: 9, diagnostics_ms: 10, extrapolation_ms: 11, materialization_ms: 12, spray_ms: 13, fluidResidency_ms: 14, sparsePublication_ms: 15 });
  assert.equal(categorizedGPUPhysicsTime_ms(timings), 120);
});

test("timestamp capacity covers the worst 64-substep wrapper trace", () => {
  // Pressure uses one shared split boundary. Nested power/face projection
  // uses five values and fine surface uses four, adding six values per step.
  const worstCaseQueries = 2 + 2 + 64 * (9 * 2 + 6) + 3 * 2;
  assert.ok(GPU_PHYSICS_TIMESTAMP_CAPACITY >= worstCaseQueries);
  assert.equal(GPU_PHYSICS_TIMESTAMP_CAPACITY * 8, 16384);
});

test("shared timestamp partitions decode only when every boundary is resolved and monotonic", () => {
  const requiredBoundaries = [0, 1, 2, 3];
  const segments: GPUPhysicsTimestampSegment[] = [
    { name: "surfaceUpdate_ms", start: 0, end: 3, requiredBoundaries },
    { name: "fineTransport_ms", start: 0, end: 1, requiredBoundaries },
    { name: "fineTopology_ms", start: 1, end: 2, requiredBoundaries },
    { name: "fineRedistance_ms", start: 2, end: 3, requiredBoundaries },
  ];
  assert.deepEqual(decodeGPUPhysicsTimestampSegments(
    new BigUint64Array([1_000_000n, 3_000_000n, 7_000_000n, 10_000_000n]),
    segments,
  ), {
    surfaceUpdate_ms: 9,
    fineTransport_ms: 2,
    fineTopology_ms: 4,
    fineRedistance_ms: 3,
  });

  for (const timestamps of [
    new BigUint64Array([0n, 3_000_000n, 7_000_000n, 10_000_000n]),
    new BigUint64Array([1_000_000n, 8_000_000n, 7_000_000n, 10_000_000n]),
  ]) {
    assert.deepEqual(decodeGPUPhysicsTimestampSegments(timestamps, segments), {
      surfaceUpdate_ms: 0,
      fineTransport_ms: 0,
      fineTopology_ms: 0,
      fineRedistance_ms: 0,
    });
  }
});

test("one invalid repeated timestamp range suppresses the whole timing field", () => {
  const segments: GPUPhysicsTimestampSegment[] = [
    { name: "pressure_ms", start: 0, end: 1 },
    { name: "pressure_ms", start: 2, end: 3 },
  ];
  assert.deepEqual(decodeGPUPhysicsTimestampSegments(
    new BigUint64Array([1_000_000n, 2_000_000n, 0n, 4_000_000n]),
    segments,
  ), { pressure_ms: 0 });
});
