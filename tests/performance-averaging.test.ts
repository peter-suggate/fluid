import assert from "node:assert/strict";
import test from "node:test";
import { averagePerformanceSnapshots, rollingPerformanceSnapshots } from "../lib/performance-averaging";
import { emptyPerformance, type PerformanceSnapshot } from "../lib/stores/diagnostics-store";

const sample = (cpuFrame_ms: number, gpuPressure_ms: number, physicsAvailable = true): PerformanceSnapshot => ({
  ...emptyPerformance,
  methodId: "tall-cell",
  gpuPhysicsTimingAvailable: physicsAvailable,
  cpuFrame_ms,
  gpuPressure_ms
});

test("frame averaging includes every CPU frame but excludes unavailable GPU timings", () => {
  const averaged = averagePerformanceSnapshots([
    sample(3, 0, false),
    sample(6, 9),
    sample(12, 15)
  ], emptyPerformance);

  assert.equal(averaged.cpuFrame_ms, 7);
  assert.equal(averaged.gpuPressure_ms, 12);
  assert.equal(averaged.gpuPhysicsTimingAvailable, true);
});

test("rolling frame averages use a trailing window at every history point", () => {
  const rolling = rollingPerformanceSnapshots([sample(3, 3), sample(6, 6), sample(12, 12)], 2);
  assert.deepEqual(rolling.map((entry) => entry.cpuFrame_ms), [3, 4.5, 9]);
  assert.deepEqual(rolling.map((entry) => entry.gpuPressure_ms), [3, 4.5, 9]);
});

test("timing epoch and accepted-sample identities remain discrete metadata", () => {
  const first = { ...sample(3, 3), renderTimingEpoch: 7, renderTimingSampleId: 40 };
  const second = { ...sample(6, 6), renderTimingEpoch: 7, renderTimingSampleId: 44 };
  const averaged = averagePerformanceSnapshots([first, second], emptyPerformance);
  assert.equal(averaged.renderTimingEpoch, 7);
  assert.equal(averaged.renderTimingSampleId, 44);
});

test("repeated UI frames do not overweight one asynchronous physics sample", () => {
  const first = { ...sample(3, 3), gpuPhysicsTimingSampleId: 10 };
  const repeated = { ...sample(6, 3), gpuPhysicsTimingSampleId: 10 };
  const second = { ...sample(9, 9), gpuPhysicsTimingSampleId: 11 };
  const averaged = averagePerformanceSnapshots([first, repeated, second], emptyPerformance);
  assert.equal(averaged.gpuPressure_ms, 6);
  assert.equal(averaged.gpuPhysicsTimingSampleId, 11);
});

test("averaging preserves the union of conditional stages in the selected window", () => {
  const first = { ...sample(3, 3), gpuActiveStages: ["preparation", "advection", "remeshing"] as PerformanceSnapshot["gpuActiveStages"] };
  const second = { ...sample(6, 6), gpuActiveStages: ["preparation", "advection"] as PerformanceSnapshot["gpuActiveStages"] };
  const averaged = averagePerformanceSnapshots([first, second], emptyPerformance);
  assert.deepEqual(averaged.gpuActiveStages, ["preparation", "advection", "remeshing"]);
});

test("averaging carries sparse residency and publication timestamp stages", () => {
  const first = { ...sample(3, 3), gpuFluidResidency_ms: 1, gpuSparsePublication_ms: 3 };
  const second = { ...sample(6, 6), gpuFluidResidency_ms: 3, gpuSparsePublication_ms: 7 };
  const averaged = averagePerformanceSnapshots([first, second], emptyPerformance);
  assert.equal(averaged.gpuFluidResidency_ms, 2);
  assert.equal(averaged.gpuSparsePublication_ms, 5);
});

test("averaging carries the power-diagram timing subdivisions", () => {
  const first = { ...sample(3, 5), gpuPowerAssembly_ms: 2, gpuPressureSolve_ms: 3, gpuPowerProjection_ms: 4, gpuVelocityProjection_ms: 1 };
  const second = { ...sample(6, 9), gpuPowerAssembly_ms: 4, gpuPressureSolve_ms: 5, gpuPowerProjection_ms: 6, gpuVelocityProjection_ms: 3 };
  const averaged = averagePerformanceSnapshots([first, second], emptyPerformance);
  assert.deepEqual(
    [averaged.gpuPowerAssembly_ms, averaged.gpuPressureSolve_ms, averaged.gpuPowerProjection_ms, averaged.gpuVelocityProjection_ms],
    [3, 4, 5, 2],
  );
});

test("averaging carries the Section 5 timing subdivisions", () => {
  const first = { ...sample(3, 5), gpuFaceBand_ms: 1, gpuFaceMarch_ms: 2, gpuPowerPublication_ms: 3,
    gpuFineTopology_ms: 4, gpuFineTransport_ms: 5, gpuFineRedistance_ms: 6 };
  const second = { ...sample(6, 9), gpuFaceBand_ms: 3, gpuFaceMarch_ms: 4, gpuPowerPublication_ms: 5,
    gpuFineTopology_ms: 6, gpuFineTransport_ms: 7, gpuFineRedistance_ms: 8 };
  const averaged = averagePerformanceSnapshots([first, second], emptyPerformance);
  assert.deepEqual(
    [averaged.gpuFaceBand_ms, averaged.gpuFaceMarch_ms, averaged.gpuPowerPublication_ms,
      averaged.gpuFineTopology_ms, averaged.gpuFineTransport_ms, averaged.gpuFineRedistance_ms],
    [2, 3, 4, 5, 6, 7],
  );
});
