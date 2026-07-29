import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../tools/benchmark-svo-split-cumulative-gpu.ts", import.meta.url), "utf8");

test("split cumulative benchmark fixes every control except shading path", () => {
  assert.match(source, /FLUID_SVO_DRY_FRAME_SCENE: "hose-tank"/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_TRAVERSAL: "canonical-parametric"/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_BRICK_OCCUPANCY: "off"/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_BRICK_SIZE: "8"/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS: "0"/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_CONE_SCALE: "0.5"/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_COHERENCE: "off"/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_SHADING: shadingPath/);
  assert.match(source, /soleVariable: "shadingPath"/);
  assert.match(source, /repeat % 2 === 0 \? \["inline", "split"\] : \["split", "inline"\]/);
  assert.match(source, /aggregateTiming/);
  assert.match(source, /timingCaveat/);
});

test("split cumulative benchmark reports full-frame raw and visible error", () => {
  assert.match(source, /differingHalfWords/);
  assert.match(source, /differingPixels/);
  assert.match(source, /differingTonePixels/);
  assert.match(source, /maximumToneByteDelta/);
  assert.match(source, /relativeLuminance/);
  assert.match(source, /changedPixelBounds/);
  assert.match(source, /configuredPackedRgba16FloatPath/);
});
