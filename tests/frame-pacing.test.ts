import assert from "node:assert/strict";
import test from "node:test";
import { advancePresentationClock, clampTargetFps, frameInterval_ms, presentationFrameDue } from "../lib/frame-pacing";

test("presentation pacing defaults and clamps to supported rates", () => {
  assert.equal(clampTargetFps(Number.NaN), 60);
  assert.equal(clampTargetFps(10), 24);
  assert.equal(clampTargetFps(144), 120);
  assert.ok(Math.abs(frameInterval_ms(60) - 1000 / 60) < 1e-12);
});

test("presentation pacing tolerates requestAnimationFrame timestamp jitter", () => {
  assert.equal(presentationFrameDue(-Infinity, 0, 60), true);
  assert.equal(presentationFrameDue(0, 8.3, 60), false);
  assert.equal(presentationFrameDue(0, 16.2, 60), true);
  assert.equal(presentationFrameDue(0, 16.2, 30), false);
  assert.equal(presentationFrameDue(0, 32.9, 30), true);
});

test("the nominal clock can select 90 of 120 display callbacks", () => {
  let clock = -Infinity;
  const presented: number[] = [];
  for (let index = 0; index < 8; index += 1) {
    const now = index * (1000 / 120);
    if (!presentationFrameDue(clock, now, 90)) continue;
    presented.push(index);
    clock = advancePresentationClock(clock, now, 90);
  }
  assert.deepEqual(presented, [0, 2, 3, 4, 6, 7]);
});
