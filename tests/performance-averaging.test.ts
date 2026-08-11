import assert from "node:assert/strict";
import test from "node:test";
import { averagePerformanceTraces, performanceTraceIsExact, type PerformanceTrace } from "../lib/performance-trace";

const trace = (sampleId: number, advection_ms: number, solve_ms: number): PerformanceTrace => ({
  sampleId,
  domain: "gpu",
  lane: "physics",
  context: "octree",
  capturedAt_ms: sampleId,
  total_ms: advection_ms + solve_ms,
  phases: [
    { id: "velocity-advection", label: "Velocity advection", duration_ms: advection_ms },
    { id: "pressure-solve", label: "Pressure solve", duration_ms: solve_ms },
  ],
});

test("generic trace averaging preserves exact additive accounting", () => {
  const averaged = averagePerformanceTraces([trace(1, 2, 4), trace(2, 4, 8)]);
  assert.ok(averaged);
  assert.equal(averaged.sampleId, 2);
  assert.equal(averaged.total_ms, 9);
  assert.deepEqual(averaged.phases.map((phase) => phase.duration_ms), [3, 6]);
  assert.equal(performanceTraceIsExact(averaged), true);
});

test("intermittent phases keep per-encode durations with an encoded fraction", () => {
  const first = trace(1, 2, 4);
  const second: PerformanceTrace = {
    ...trace(2, 4, 0),
    phases: [{ id: "velocity-advection", label: "Velocity advection", duration_ms: 4 }],
  };
  const averaged = averagePerformanceTraces([first, second]);
  assert.ok(averaged);
  assert.equal(averaged.total_ms, 5);
  // Per-encode cost, not diluted over frames that never ran the phase: the
  // solve ran once at 4 ms, so it reads 4 ms at fraction 0.5 — not 2 ms.
  assert.deepEqual(averaged.phases.map((phase) => phase.duration_ms), [3, 4]);
  assert.deepEqual(averaged.phases.map((phase) => phase.encodedFraction), [1, 0.5]);
  assert.equal(performanceTraceIsExact(averaged), true);
});

test("cached observations are averaged once and independent lanes cannot be combined", () => {
  const first = trace(1, 2, 4);
  const averaged = averagePerformanceTraces([first, first, trace(2, 4, 8)]);
  assert.ok(averaged);
  assert.equal(averaged.total_ms, 9,
    "re-publishing one cached trace must not weight it as a second observation");
  assert.throws(() => averagePerformanceTraces([
    first,
    { ...first, domain: "cpu", lane: "main-thread" },
  ]), /independent performance lanes/);
});
