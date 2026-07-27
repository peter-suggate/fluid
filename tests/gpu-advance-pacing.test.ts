import assert from "node:assert/strict";
import test from "node:test";
import { GPUAdvanceWallEstimator } from "../lib/gpu-advance-pacing";

test("an unpipelined completion samples submit-to-completion wall time", () => {
  const estimator = new GPUAdvanceWallEstimator();
  assert.equal(estimator.estimate_ms, undefined);
  estimator.observeCompletion(110, 100, 0);
  assert.equal(estimator.estimate_ms, 10);
});

test("saturated completions sample completion-to-completion spacing", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(110, 100, 0);
  // Two advances were in flight; the fence spacing is the per-advance cost,
  // not the inflated submit-to-done interval that includes the predecessor.
  estimator.observeCompletion(118, 100, 1);
  assert.ok(Math.abs((estimator.estimate_ms ?? 0) - (10 + 0.25 * (8 - 10))) < 1e-9);
});

test("the first pipelined completion without history falls back to submit time", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(112, 100, 3);
  assert.equal(estimator.estimate_ms, 12);
});

test("non-positive and non-finite samples are ignored", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(100, 100, 0);
  assert.equal(estimator.estimate_ms, undefined);
  estimator.observeCompletion(Number.NaN, 100, 0);
  assert.equal(estimator.estimate_ms, undefined);
  estimator.observeCompletion(104, 100, 0);
  assert.equal(estimator.estimate_ms, 4);
});

test("reset drops the model so a new solver re-bootstraps", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(105, 100, 0);
  estimator.reset();
  assert.equal(estimator.estimate_ms, undefined);
  // After reset the completion history is also gone: a pipelined sample must
  // not difference against a completion from the previous timeline.
  estimator.observeCompletion(220, 200, 2);
  assert.equal(estimator.estimate_ms, 20);
});
