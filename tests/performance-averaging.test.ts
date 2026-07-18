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
