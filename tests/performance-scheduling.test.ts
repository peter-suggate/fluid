import assert from "node:assert/strict";
import test from "node:test";
import { performanceSchedule } from "../lib/performance-scheduling";

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
