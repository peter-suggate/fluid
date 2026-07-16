import assert from "node:assert/strict";
import test from "node:test";
import { gridOverlayShader } from "../lib/webgpu-grid-overlay";

test("grid overlay is an independent alpha-composited presentation layer", () => {
  assert.match(gridOverlayShader, /@group\(0\) @binding\(2\) var fluidField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(3\) var tallCellBases: texture_2d<f32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(4\) var adaptiveCells: texture_3d<u32>/);
  assert.match(gridOverlayShader, /let axis = i32\(round\(u\.debug\.x\)\)/);
  assert.match(gridOverlayShader, /@fragment fn fragmentMain/);
  assert.match(gridOverlayShader, /return vec4f\(displayColor\(overlay\.color\), overlay\.alpha\)/);
});

test("grid overlay suppresses dense backing-grid lines inside adaptive cells", () => {
  assert.match(gridOverlayShader, /let adaptiveGrid = u\.debug\.z > 0\.5/);
  assert.match(gridOverlayShader, /any\(adaptiveCellKey\(lowerHorizontal, dims\) != own\)/);
  assert.match(gridOverlayShader, /any\(adaptiveCellKey\(below, dims\) != own\)/);
  assert.match(gridOverlayShader, /let leafSize = i32\(\(key\.x >> 20u\) & 1023u\)/);
});

test("grid overlay preserves rigid-body occlusion independently of the water renderer", () => {
  assert.match(gridOverlayShader, /fn nearestBodyDistance/);
  assert.match(gridOverlayShader, /distance >= nearestBodyDistance\(origin, direction\) && !overlay\.solid/);
});

test("grid overlay renders rigid bodies as complete represented cells", () => {
  assert.match(gridOverlayShader, /fn bodySignedDistance/);
  assert.match(gridOverlayShader, /let sphereDistance = length\(closest - bodies\[index\]\.positionRadius\.xyz\) - bodies\[index\]\.positionRadius\.w/);
  assert.match(gridOverlayShader, /gridBodySample\(representedCell\(cell, dims, boundsMin, size, adaptiveGrid, tallGrid\)\)/);
  assert.match(gridOverlayShader, /return GridSample\(color, alpha, gridBody\.occupied\)/);
});
