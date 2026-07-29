import assert from "node:assert/strict";
import test from "node:test";
import { GPUAdvanceWallEstimator } from "../lib/gpu-advance-pacing";

const observation = (
  pendingTargetWorkAtSubmit: number,
  interveningWorkSequence = 0,
  queueWasIdleAtSubmit = pendingTargetWorkAtSubmit === 0,
) => ({ pendingTargetWorkAtSubmit, interveningWorkSequence, queueWasIdleAtSubmit });

test("an unpipelined completion samples submit-to-completion wall time", () => {
  const estimator = new GPUAdvanceWallEstimator();
  assert.equal(estimator.estimate_ms, undefined);
  estimator.observeCompletion(110, 100, observation(0));
  assert.equal(estimator.estimate_ms, 10);
});

test("saturated completions sample completion-to-completion spacing", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(110, 100, observation(0));
  // Two advances were in flight; the fence spacing is the per-advance cost,
  // not the inflated submit-to-done interval that includes the predecessor.
  estimator.observeCompletion(118, 100, observation(1));
  assert.ok(Math.abs((estimator.estimate_ms ?? 0) - (10 + 0.25 * (8 - 10))) < 1e-9);
});

test("the first pipelined completion without an idle queue is ignored", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(112, 100, observation(3, 0, false));
  assert.equal(estimator.estimate_ms, undefined);
});

test("presentation work between physics completions is never charged to physics", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(120, 100, observation(0, 1, false));
  assert.equal(estimator.estimate_ms, undefined, "presentation-ahead first sample is contaminated");
  estimator.observeCompletion(128, 101, observation(1, 1, false));
  assert.equal(estimator.estimate_ms, 8, "the second completion isolates an uninterrupted physics step");
  estimator.observeCompletion(150, 129, observation(1, 2, false));
  assert.equal(estimator.estimate_ms, 8, "a new presentation epoch must not inflate the EMA");
});

test("non-positive and non-finite samples are ignored", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(100, 100, observation(0));
  assert.equal(estimator.estimate_ms, undefined);
  estimator.observeCompletion(Number.NaN, 100, observation(0));
  assert.equal(estimator.estimate_ms, undefined);
  estimator.observeCompletion(104, 100, observation(0));
  assert.equal(estimator.estimate_ms, 4);
});

test("reset drops the model so a new solver re-bootstraps", () => {
  const estimator = new GPUAdvanceWallEstimator();
  estimator.observeCompletion(105, 100, observation(0));
  estimator.reset();
  assert.equal(estimator.estimate_ms, undefined);
  // After reset the completion history is also gone: a pipelined sample must
  // not difference against a completion from the previous timeline.
  estimator.observeCompletion(220, 200, observation(2, 0, false));
  assert.equal(estimator.estimate_ms, undefined);
});
