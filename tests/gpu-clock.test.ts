import assert from "node:assert/strict";
import test from "node:test";
import { commitGPUCompletion, gpuBatchDepth, gpuCanAcceptNextStep, gpuInFlightStepLimit } from "../lib/simulation/gpu-clock";

test("tall-cell batches one presentation quantum without changing dt", () => {
  assert.equal(gpuBatchDepth("tall-cell", 0.004, false), 5);
  assert.equal(gpuBatchDepth("tall-cell", 1 / 30, false), 1);
  assert.equal(gpuBatchDepth("tall-cell", 0.0005, false), 32, "batch latency remains bounded");
  assert.equal(gpuBatchDepth("tall-cell", 0.004, true), 5, "partitioned rigid feedback retains presentation throughput");
  assert.equal(gpuBatchDepth("tall-cell", 0.004, false, 30), 9, "30 Hz batches enough stable substeps for real time");
  assert.equal(gpuBatchDepth("tall-cell", 0.004, false, 120), 3, "120 Hz uses a shorter presentation quantum");
});

test("every selectable frame rate can advance the default tall-cell step in real time", () => {
  const dt = 0.004;
  for (const fps of [24, 30, 60, 90, 120]) {
    assert.ok(gpuBatchDepth("tall-cell", dt, false, fps) * dt >= 1 / fps - 1e-9, `${fps} Hz batch covers its wall-clock interval`);
  }
});

test("adaptive methods use bounded batching when their coupling policy permits it", () => {
  assert.equal(gpuBatchDepth("quadtree-tall-cell", 0.004, false), 2);
  assert.equal(gpuBatchDepth("quadtree-tall-cell", 0.004, true), 1);
  assert.equal(gpuBatchDepth("octree", 0.004, false), 5, "uncoupled octree work fills one presentation quantum");
  assert.equal(gpuBatchDepth("octree", 0.004, true), 5, "lagged octree feedback keeps presentation throughput");
  assert.equal(gpuBatchDepth("uniform-grid", 0.004, false), 1);
  assert.equal(gpuBatchDepth("tall-cell", Number.NaN, false), 1);
});

test("eligible GPU methods prepare a second bounded batch to prevent queue starvation", () => {
  assert.equal(gpuInFlightStepLimit("tall-cell", 0.0065, true, 60), 6);
  assert.equal(gpuInFlightStepLimit("tall-cell", 0.004, false, 30), 18);
  assert.equal(gpuInFlightStepLimit("quadtree-tall-cell", 0.004, true, 60), 1, "adaptive rigid coupling remains single-step");
  assert.equal(gpuInFlightStepLimit("quadtree-tall-cell", 0.004, false, 60), 2, "adaptive topology retains its shallow window");
  assert.equal(gpuInFlightStepLimit("octree", 0.004, false, 60), 10, "uncoupled octree keeps a second presentation batch queued");
  assert.equal(gpuInFlightStepLimit("octree", 0.004, true, 60), 10, "lagged octree coupling stays bounded to two batches");
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
