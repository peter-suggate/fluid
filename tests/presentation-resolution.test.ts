import test from "node:test";
import assert from "node:assert/strict";
import { boundedPresentationScale } from "../lib/core/presentation-resolution";

test("supersampling respects texture dimensions and aggregate pixel storage", () => {
  assert.equal(boundedPresentationScale(1000, 700, 2, 8192), 2);
  const retina = boundedPresentationScale(3840, 2160, 2, 8192);
  assert.ok(3840 * retina * 2160 * retina <= 8 * 1024 * 1024 + 1);
  const constrained = boundedPresentationScale(6000, 2000, 2, 4096);
  assert.ok(Math.floor(6000 * constrained) <= 4096);
  assert.equal(boundedPresentationScale(6000, 3000, 1, 8192), 1, "legacy native resolution is preserved");
  assert.equal(boundedPresentationScale(1000, 700, NaN, 8192), 1);
});
