import assert from "node:assert/strict";
import test from "node:test";
import { adaptiveTopologyPerformanceStages, physicsPerformanceStages } from "../lib/performance-stage-model";
import { emptyPerformance, measuredGPUTime_ms, type PerformanceSnapshot } from "../lib/stores/diagnostics-store";
import { categorizedGPUPhysicsTime_ms, emptyGPUPhysicsTimings, GPU_PHYSICS_TIMESTAMP_CAPACITY, type GPUPhysicsStageId } from "../lib/webgpu-eulerian";

const snapshot = (methodId: string, activeStages: GPUPhysicsStageId[]): PerformanceSnapshot => ({
  ...emptyPerformance,
  methodId,
  gpuPhysicsTimingAvailable: true,
  gpuActiveStages: activeStages,
  gpuPreparation_ms: methodId === "tall-cell" ? 1 : 0,
  gpuLayerConstruction_ms: methodId === "quadtree-tall-cell" ? 2 : 0,
  gpuAdvection_ms: 3,
  gpuConditioning_ms: methodId === "uniform" ? 4 : 0,
  gpuRemeshing_ms: methodId === "tall-cell" ? 5 : 0,
  gpuPressure_ms: 6,
  gpuProjection_ms: methodId === "quadtree-tall-cell" ? 0 : 7,
  gpuSurfaceUpdate_ms: methodId === "quadtree-tall-cell" ? 8 : 0,
  gpuRigid_ms: 9,
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
  Object.assign(timings, { preparation_ms: 1, layerConstruction_ms: 2, advection_ms: 3, conditioning_ms: 4, remeshing_ms: 5, pressure_ms: 6, projection_ms: 7, surfaceUpdate_ms: 8, rigidCoupling_ms: 9, diagnostics_ms: 10 });
  assert.equal(categorizedGPUPhysicsTime_ms(timings), 55);
});

test("timestamp capacity covers the worst 64-substep wrapper trace", () => {
  const worstCaseQueries = 2 * (1 + 64 * 5 + 1); // total + five possible timed categories/substep + diagnostics
  assert.ok(GPU_PHYSICS_TIMESTAMP_CAPACITY >= worstCaseQueries);
  assert.equal(GPU_PHYSICS_TIMESTAMP_CAPACITY * 8, 6144);
});
