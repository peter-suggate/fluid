import assert from "node:assert/strict";
import test from "node:test";
import {
  clipPowerPolygonHalfSpace,
  movingSolidNormalVelocity,
  powerFaceHalfSpaceAperture,
  powerPolygonMetric,
} from "../lib/octree-power-solid";

const square = [
  [-1, -1, 0],
  [1, -1, 0],
  [1, 1, 0],
  [-1, 1, 0],
] as const;

test("power polygon metric and exact half-space aperture retain physical area", () => {
  assert.deepEqual(powerPolygonMetric(square), { area: 4, centroid: [0, 0, 0] });

  const half = powerFaceHalfSpaceAperture(square, [1, 0, 0], 0);
  assert.equal(half.openFraction, 0.5);
  assert.equal(half.openArea, 2);
  assert.equal(half.closedArea, 2);
  assert.deepEqual(half.closedCentroid, [-0.5, 0, 0]);

  assert.equal(powerFaceHalfSpaceAperture(square, [1, 0, 0], -2).openFraction, 1);
  assert.equal(powerFaceHalfSpaceAperture(square, [1, 0, 0], 2).openFraction, 0);
});

test("power polygon clipping preserves winding and coplanarity", () => {
  const clipped = clipPowerPolygonHalfSpace(square, [1, 1, 0], 0, true);
  assert.equal(clipped.length, 3);
  const metric = powerPolygonMetric(clipped);
  assert.ok(Math.abs(metric.area - 2) < 1e-12);
  assert.ok(metric.centroid[0] + metric.centroid[1] > 0);
  assert.ok(clipped.every((point) => point[2] === 0));
});

test("moving-solid normal velocity includes angular motion at face centroid", () => {
  assert.equal(movingSolidNormalVelocity([1, 2, 3], [0, 0, 2], [0, 0, 0], [0, 3, 0], [1, 0, 0]), -5);
  assert.ok(Math.abs(movingSolidNormalVelocity([0, 0, 0], [0, 2, 0], [1, 0, 1], [3, 0, 1], [0, 0, -2]) - 4) < 1e-12);
});
