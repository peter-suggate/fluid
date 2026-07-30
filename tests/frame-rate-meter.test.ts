import assert from "node:assert/strict";
import test from "node:test";
import { SmoothedFrameRate } from "../lib/frame-rate-meter";

test("FPS is derived from the mean frame interval", () => {
  const meter = new SmoothedFrameRate(5);
  assert.equal(meter.sample(0), undefined);
  assert.equal(meter.sample(10), 100);
  assert.equal(meter.sample(30), 1000 / 15);
});

test("FPS smoothing retains only the latest five frame intervals", () => {
  const meter = new SmoothedFrameRate(5);
  [0, 100, 110, 120, 130, 140, 150].forEach((time_ms) => meter.sample(time_ms));
  assert.equal(meter.framesPerSecond, 100, "the old 100 ms interval should leave the five-frame window");
});

test("reset prevents an inactive gap from contaminating resumed FPS", () => {
  const meter = new SmoothedFrameRate(5);
  meter.sample(0);
  meter.sample(10);
  meter.reset();
  assert.equal(meter.sample(10_000), undefined);
  assert.equal(meter.sample(10_020), 50);
});

test("completed FPS uses cumulative GPU completions rather than callback spacing", () => {
  const meter = new SmoothedFrameRate(5);
  assert.equal(meter.sampleCompleted(0, 0), undefined);
  assert.equal(meter.sampleCompleted(3, 50), 60);
  assert.equal(meter.sampleCompleted(6, 100), 60);
});

test("batched GPU completion delivery does not create an FPS spike", () => {
  const meter = new SmoothedFrameRate(5);
  meter.sampleCompleted(0, 0);
  meter.sampleCompleted(1, 16);
  assert.equal(meter.sampleCompleted(6, 100), 60);
});

test("completed FPS falls to zero after the rolling window contains no completions", () => {
  const meter = new SmoothedFrameRate(5);
  meter.sampleCompleted(0, 0);
  meter.sampleCompleted(6, 100);
  meter.sampleCompleted(6, 700);
  assert.equal(meter.completedFramesPerSecond, 0);
});
