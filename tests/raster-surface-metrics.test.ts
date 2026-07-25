import assert from "node:assert/strict";
import test from "node:test";
import { enclosedSurfaceHoleMetrics, surfaceStepMetrics } from "../tools/raster-surface-metrics";

test("enclosed surface holes exclude exterior notches and retain missing interior patches", () => {
  const width = 9, height = 8;
  const mask = new Uint8Array(width * height);
  for (let y = 1; y < 7; y += 1) mask.fill(1, 1 + y * width, 8 + y * width);
  mask[4 + 3 * width] = 0;
  mask[5 + 3 * width] = 0;
  mask[4 + 4 * width] = 0;
  mask[5 + 4 * width] = 0;
  // This absent run reaches the image boundary and is a silhouette notch.
  mask[1 + 2 * width] = 0;
  mask[0 + 2 * width] = 0;
  assert.deepEqual(enclosedSurfaceHoleMetrics(mask, width, height), {
    count: 1,
    pixels: 4,
    maximumPixels: 4,
    maximumWidth_px: 2,
    maximumHeight_px: 2,
  });
});

test("surface step metrics distinguish a cell terrace from a smooth sheet", () => {
  const width = 6, height = 4, cell = 0.05;
  const mask = new Uint8Array(width * height).fill(1);
  const smooth = new Float32Array(mask.length * 3);
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const at = (x + y * width) * 3;
    smooth.set([x * 0.002, y * 0.002, 0], at);
  }
  const smoothMetrics = surfaceStepMetrics(mask, smooth, width, height, cell);
  assert.equal(smoothMetrics.cellScaleHeightJumps, 0);
  assert.ok(smoothMetrics.terraceEdgeFraction < 0.12);

  const stepped = smooth.slice();
  for (let y = 0; y < height; y += 1) for (let x = 3; x < width; x += 1) {
    stepped[(x + y * width) * 3 + 1] += cell;
  }
  const metrics = surfaceStepMetrics(mask, stepped, width, height, cell);
  assert.equal(metrics.cellScaleHeightJumps, height);
  assert.ok(metrics.terraceEdgeFraction > 0.12);
  assert.ok(metrics.maximumHeightJump_m >= 0.99 * cell);
});
