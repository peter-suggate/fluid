import assert from "node:assert/strict";
import test from "node:test";
import { commitGPUCompletion, gpuBatchDepth, gpuCanAcceptNextStep } from "../lib/simulation/gpu-clock";

test("tall-cell batches one presentation quantum without changing dt", () => {
  assert.equal(gpuBatchDepth("tall-cell", 0.004, false), 5);
  assert.equal(gpuBatchDepth("tall-cell", 1 / 30, false), 1);
  assert.equal(gpuBatchDepth("tall-cell", 0.0005, false), 8, "batch latency remains bounded");
  assert.equal(gpuBatchDepth("tall-cell", 0.004, true), 5, "partitioned rigid feedback retains presentation throughput");
});

test("other GPU methods preserve their existing submission depths", () => {
  assert.equal(gpuBatchDepth("quadtree-tall-cell", 0.004, false), 2);
  assert.equal(gpuBatchDepth("quadtree-tall-cell", 0.004, true), 1);
  assert.equal(gpuBatchDepth("uniform-grid", 0.004, false), 1);
  assert.equal(gpuBatchDepth("tall-cell", Number.NaN, false), 1);
});

test("GPU transport grants only one submitted step at a time", () => {
  assert.equal(gpuCanAcceptNextStep(0, 0), true);
  assert.equal(gpuCanAcceptNextStep(1 / 30, 0), false);
  assert.equal(gpuCanAcceptNextStep(1 / 30, 1 / 30), true);
});

test("GPU completion never publishes beyond the requested transport time", () => {
  assert.equal(commitGPUCompletion(1 / 30, 0, 1 / 30), 1 / 30);
  assert.equal(commitGPUCompletion(1 / 30, 1 / 30, 1 / 60), 1 / 30, "stale callbacks are ignored");
  assert.equal(commitGPUCompletion(1 / 30, 0, 2 / 30), 1 / 30, "completion is clamped to the prepared CPU state");
});
