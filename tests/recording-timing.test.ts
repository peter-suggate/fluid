import assert from "node:assert/strict";
import test from "node:test";
import { realTimePlaybackRate } from "../lib/recording-timing";

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
