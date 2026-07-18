import assert from "node:assert/strict";
import test from "node:test";
import { advancePresentationClock, frameInterval_ms, presentationFrameDue, presentationStateChanged } from "../lib/frame-pacing";

test("presentation pacing uses a fixed 60 Hz cadence", () => {
  assert.ok(Math.abs(frameInterval_ms() - 1000 / 60) < 1e-12);
});

test("presentation pacing tolerates requestAnimationFrame timestamp jitter", () => {
  assert.equal(presentationFrameDue(-Infinity, 0), true);
  assert.equal(presentationFrameDue(0, 8.3), false);
  assert.equal(presentationFrameDue(0, 16.2), true);
});

test("the nominal clock selects 60 of 120 display callbacks", () => {
  let clock = -Infinity;
  const presented: number[] = [];
  for (let index = 0; index < 8; index += 1) {
    const now = index * (1000 / 120);
    if (!presentationFrameDue(clock, now)) continue;
    presented.push(index);
    clock = advancePresentationClock(clock, now);
  }
  assert.deepEqual(presented, [0, 2, 4, 6]);
});

test("paused presentation state changes only when a rendered input changes", () => {
  const scene = {};
  const camera = {};
  const previous = [scene, camera, 1280, 720];
  assert.equal(presentationStateChanged(undefined, previous), true);
  assert.equal(presentationStateChanged(previous, [scene, camera, 1280, 720]), false);
  assert.equal(presentationStateChanged(previous, [scene, {}, 1280, 720]), true);
  assert.equal(presentationStateChanged(previous, [scene, camera, 1920, 1080]), true);
});
