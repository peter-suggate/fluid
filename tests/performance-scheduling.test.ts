import assert from "node:assert/strict";
import test from "node:test";
import { advanceWallBreakdown, measuredGPUUtilization, performanceSchedule } from "../lib/performance-scheduling";

test("performance scheduling separates a presentation frame from a submission batch", () => {
  const schedule = performanceSchedule({
    targetFps: 60,
    gpuAdvance_s: 0.016,
    submissionBatchDepth: 5,
    physicsPerAdvance_ms: 6.166,
    renderPerFrame_ms: 0.437,
    pressureSolvesPerAdvance: 1
  });

  assert.ok(Math.abs(schedule.advancesPerFrame - 1.0416667) < 1e-6);
  assert.ok(Math.abs(schedule.pressureSolvesPerFrame - 1.0416667) < 1e-6);
  assert.equal(schedule.pressureSolvesPerBatch, 5);
  assert.equal(schedule.batchSimulation_ms, 80);
  assert.ok(Math.abs(schedule.realtimeFramesPerBatch - 4.8) < 1e-9);
  assert.ok(Math.abs(schedule.gpuDemandPerFrame_ms - 6.860) < 0.001);
  assert.ok(Math.abs(schedule.demandPercent - 41.16) < 0.01);
});

test("pressure defect correction counts as a second solve per advance", () => {
  const schedule = performanceSchedule({ targetFps: 60, gpuAdvance_s: 0.016, submissionBatchDepth: 5, physicsPerAdvance_ms: 10, renderPerFrame_ms: 1, pressureSolvesPerAdvance: 2 });
  assert.ok(Math.abs(schedule.pressureSolvesPerFrame - 2.0833333) < 1e-6);
  assert.equal(schedule.pressureSolvesPerBatch, 10);
});

test("measured utilization combines independently timestamped physics and presentation cadence", () => {
  const utilization = measuredGPUUtilization({
    physics_ms: 6,
    physicsCompletionInterval_ms: 12,
    presentation_ms: 4,
    presentationInterval_ms: 16
  });
  assert.deepEqual(utilization, { physics: 0.5, presentation: 0.25, total: 0.75 });
  assert.equal(measuredGPUUtilization({}), null);
  assert.equal(measuredGPUUtilization({ physics_ms: 20, physicsCompletionInterval_ms: 10 })?.total, 1, "queued work is reported as saturated, never above 100%");
});

test("last advance wall time preserves CPU encode and untimestamped queue delay", () => {
  assert.deepEqual(advanceWallBreakdown({
    cpuAdvanceEncode_ms: 48,
    gpuBatchWall_ms: 210,
    gpuAdvanceWall_ms: 258,
    gpuStep_ms: 0.08,
  }), {
    encode_ms: 48,
    queueFence_ms: 210,
    timestampedGPU_ms: 0.08,
    untimestampedQueue_ms: 209.92,
    wall_ms: 258,
  });
  assert.equal(advanceWallBreakdown({}), null);
});
