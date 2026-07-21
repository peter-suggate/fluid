import assert from "node:assert/strict";
import test from "node:test";
import { narrowVerticalSlitMetrics } from "../tools/raster-slit-metrics";

function filledRectangle(width: number, height: number) {
  const mask = new Uint8Array(width * height);
  for (let y = 2; y < height - 2; y += 1) mask.fill(1, 2 + y * width, width - 2 + y * width);
  return mask;
}

test("narrow vertical slit metric detects persistent internal gaps", () => {
  const width = 20, height = 18;
  const mask = filledRectangle(width, height);
  for (let y = 4; y < 16; y += 1) mask[9 + y * width] = 0;
  assert.deepEqual(narrowVerticalSlitMetrics(mask, width, height), {
    count: 1, pixels: 12, maximumLength_px: 12,
  });
});

test("narrow vertical slit metric accepts three-pixel diagonal cracks", () => {
  const width = 24, height = 20;
  const mask = filledRectangle(width, height);
  for (let y = 4; y < 16; y += 1) {
    const x = 10 + Math.floor((y - 4) / 6);
    mask.fill(0, x + y * width, x + 3 + y * width);
  }
  const metrics = narrowVerticalSlitMetrics(mask, width, height);
  assert.equal(metrics.count, 1);
  assert.equal(metrics.pixels, 36);
  assert.equal(metrics.maximumLength_px, 12);
});

test("narrow vertical slit metric ignores silhouettes and isolated noise", () => {
  const width = 20, height = 18;
  const mask = filledRectangle(width, height);
  for (let y = 5; y < 14; y += 1) mask.fill(0, 2 + y * width, 4 + y * width);
  mask[9 + 5 * width] = 0;
  mask[10 + 10 * width] = 0;
  assert.deepEqual(narrowVerticalSlitMetrics(mask, width, height), {
    count: 0, pixels: 0, maximumLength_px: 0,
  });
});
