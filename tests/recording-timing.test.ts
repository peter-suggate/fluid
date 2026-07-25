import assert from "node:assert/strict";
import test from "node:test";
import { SIMULATION_VIDEO_FRAME_DURATION_S, realTimePlaybackRate, simulationFramesDue, sourceDurationForPlayback } from "../lib/recording-timing";

test("recording playback maps wall-clock capture duration onto simulation time", () => {
  assert.equal(realTimePlaybackRate(12, 3), 4);
  assert.equal(realTimePlaybackRate(3, 3), 1);
  assert.equal(realTimePlaybackRate(1.5, 3), 0.5);
});

test("recording playback falls back safely for incomplete timing", () => {
  assert.equal(realTimePlaybackRate(0, 3), 1);
  assert.equal(realTimePlaybackRate(3, 0), 1);
  assert.equal(realTimePlaybackRate(Number.NaN, 3), 1);
});

test("non-finite WebM metadata preserves the measured capture duration", () => {
  assert.equal(sourceDurationForPlayback(Number.POSITIVE_INFINITY, 12), 12);
  assert.equal(sourceDurationForPlayback(Number.NaN, 12), 12);
  assert.equal(sourceDurationForPlayback(11.8, 12), 11.8);
});

test("30 fps capture samples every crossed 0.033 simulation seconds", () => {
  const next = 5 + SIMULATION_VIDEO_FRAME_DURATION_S;
  assert.equal(simulationFramesDue(5.02, next), 0);
  assert.equal(simulationFramesDue(next, next), 1);
  assert.equal(simulationFramesDue(5.1, next), 3);
});
