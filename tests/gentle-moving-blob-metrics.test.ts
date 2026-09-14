import assert from "node:assert/strict";
import test from "node:test";

import { contourMetrics } from "../tools/gentle-moving-blob-metrics";

function ellipse(cx: number, cy: number, rx: number, ry: number, count = 512) {
  const result: number[] = [];
  for (let i = 0; i < count; i++) {
    const a = 2 * Math.PI * i / count, b = 2 * Math.PI * (i + 1) / count;
    result.push(cx + rx * Math.cos(a), cy + ry * Math.sin(a),
      cx + rx * Math.cos(b), cy + ry * Math.sin(b));
  }
  return result;
}

test("circle contour metrics are translation invariant", () => {
  const first = contourMetrics(ellipse(10, 12, 6, 6), Math.PI * 36, [10, 12], 6, .05, [32, 24]);
  const moved = contourMetrics(ellipse(17, 9, 6, 6), Math.PI * 36, [17, 9], 6, .05, [32, 24]);
  for (const metric of [first, moved]) {
    assert.ok(Math.abs(metric.circularity! - 1) < 1e-4);
    assert.ok(metric.radialRmsErrorM! < 1e-5);
    assert.ok(Math.abs(metric.boundaryAxisRatio! - 1) < 1e-4);
  }
});

test("ellipse contour reports its boundary axis ratio", () => {
  const metric = contourMetrics(ellipse(16, 12, 8, 4), Math.PI * 32, [16, 12], 6, 1, [32, 24]);
  assert.ok(metric.boundaryAxisRatio! > 1.5 && metric.boundaryAxisRatio! < 1.8);
  assert.ok(metric.circularity! < 0.9);
});

test("absent contour metrics are null rather than false zeroes", () => {
  const metric = contourMetrics([], 0, [0, 0], 1, 1, [2, 2]);
  assert.equal(metric.segmentCount, 0);
  assert.equal(metric.circularity, null);
  assert.equal(metric.radialRmsErrorM, null);
  assert.equal(metric.boundaryAxisRatio, null);
});
