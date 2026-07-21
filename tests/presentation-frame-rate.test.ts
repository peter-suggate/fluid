import assert from "node:assert/strict";
import test from "node:test";
import { PresentationFrameRateTracker } from "../lib/presentation-frame-rate";

test("presentation frame rate measures submitted-frame timestamps", () => {
  const tracker = new PresentationFrameRateTracker();
  for (let frame = 0; frame <= 30; frame += 1) tracker.record(frame * (1_000 / 60));
  assert.ok(Math.abs((tracker.sample(500) ?? 0) - 60) < 0.001);
});

test("presentation frame rate does not count time without submissions", () => {
  const tracker = new PresentationFrameRateTracker();
  tracker.record(0);
  tracker.record(16.67);
  assert.equal(tracker.sample(1_100), null);
});

test("presentation frame rate carries the boundary interval into the next sample", () => {
  const tracker = new PresentationFrameRateTracker();
  tracker.record(0);
  tracker.record(20);
  assert.equal(tracker.sample(20), 50);
  tracker.record(40);
  tracker.record(60);
  assert.equal(tracker.sample(60), 50);
});
