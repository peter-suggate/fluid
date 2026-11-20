import test from "node:test";
import assert from "node:assert/strict";
import { boundedPresentationScale, PlaybackPixelBudget } from "../lib/core/presentation-resolution";

test("playback adjusts to sustained missed refreshes and recovers with queue headroom", () => {
  const budget = new PlaybackPixelBudget();
  assert.equal(budget.resolve(10000, 0, 0), 10000);
  assert.equal(budget.resolve(10000, 2000, 110), 7744);
  assert.equal(budget.resolve(10000, 4000, 220), 5929);
  assert.equal(budget.resolve(undefined, 12000, 400), undefined);
  assert.equal(budget.resolve(10000, 15000, 500), 5929);
  for (let i = 0; i < 100; i++) budget.observeCompletion(6);
  for (let i = 1; i <= 6; i++) budget.resolve(10000, 15000 + 2000*i, 500 + 120*i);
  assert.equal(budget.resolve(10000, 27001, 1220), 7744);
  assert.equal(budget.resolve(20000, 27002, 1220), 20000);
});

test("the playback budget preserves the pixel count across display densities and aspect ratios", () => {
  const pixels = 688 * 387;
  assert.equal(boundedPresentationScale(1280, 720, .6, 8192, pixels), 688 / 1280);
  assert.equal(boundedPresentationScale(3840, 2160, .6, 8192, pixels), 688 / 3840);
  assert.ok(1800 * 1800 * boundedPresentationScale(1800, 1800, .6, 8192, pixels) ** 2 <= pixels + 1e-6);
  assert.equal(boundedPresentationScale(640, 480, .6, 8192, pixels), .6);
});

test("supersampling respects texture dimensions and aggregate pixel storage", () => {
  assert.equal(boundedPresentationScale(1000, 700, 2, 8192), 2);
  const retina = boundedPresentationScale(3840, 2160, 2, 8192);
  assert.ok(3840 * retina * 2160 * retina <= 8 * 1024 * 1024 + 1);
  const constrained = boundedPresentationScale(6000, 2000, 2, 4096);
  assert.ok(Math.floor(6000 * constrained) <= 4096);
  assert.equal(boundedPresentationScale(6000, 3000, 1, 8192), 1, "legacy native resolution is preserved");
  assert.equal(boundedPresentationScale(1000, 700, NaN, 8192), 1);
});
