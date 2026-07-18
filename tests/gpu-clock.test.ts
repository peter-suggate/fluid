import assert from "node:assert/strict";
import test from "node:test";
import { commitGPUCompletion, gpuCanAcceptNextStep } from "../lib/simulation/gpu-clock";

test("GPU transport grants only one explicit single step at a time", () => {
  assert.equal(gpuCanAcceptNextStep(0, 0), true);
  assert.equal(gpuCanAcceptNextStep(1 / 30, 0), false);
  assert.equal(gpuCanAcceptNextStep(1 / 30, 1 / 30), true);
});

test("GPU completion never publishes beyond the requested transport time", () => {
  assert.equal(commitGPUCompletion(1 / 30, 0, 1 / 30), 1 / 30);
  assert.equal(commitGPUCompletion(1 / 30, 1 / 30, 1 / 60), 1 / 30, "stale callbacks are ignored");
  assert.equal(commitGPUCompletion(1 / 30, 0, 2 / 30), 1 / 30, "completion is clamped to the prepared CPU state");
});
