import assert from "node:assert/strict";
import test from "node:test";
import { gridOverlayShader } from "../lib/webgpu-grid-overlay";

test("grid overlay is an independent alpha-composited presentation layer", () => {
  assert.match(gridOverlayShader, /@group\(0\) @binding\(2\) var fluidField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(3\) var tallCellBases: texture_2d<f32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(4\) var adaptiveCells: texture_3d<u32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(6\) var pressureSamples: texture_3d<u32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(7\) var divergenceField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(8\) var mappedPressureField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /let axis = i32\(round\(u\.debug\.x\)\)/);
  assert.match(gridOverlayShader, /@fragment fn fragmentMain/);
  assert.match(gridOverlayShader, /return vec4f\(displayColor\(overlay\.color\), overlay\.alpha\)/);
});

test("adaptive diagnostic modes expose coverage, level set, divergence, and pressure", () => {
  assert.match(gridOverlayShader, /fieldMode == 3/);
  assert.match(gridOverlayShader, /fieldMode == 4/);
  assert.match(gridOverlayShader, /fieldMode == 5/);
  assert.match(gridOverlayShader, /fieldMode == 6/);
  assert.match(gridOverlayShader, /let unrepresented = adaptiveGrid && wet && !hasLiquidPressureDof\(cell\)/);
  assert.match(gridOverlayShader, /divergence \* max\(u\.environment\.y, 1e-6\)/);
  assert.match(gridOverlayShader, /textureLoad\(mappedPressureField, cell, 0\)\.x/);
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

test("grid overlay field modes sample live GPU velocity without readback", () => {
  assert.match(gridOverlayShader, /@group\(0\) @binding\(5\) var velocityField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /let fieldMode = i32\(round\(u\.debug\.w\)\)/);
  // CFL mode uses the solver's substep dt carried in the shared uniform.
  assert.match(gridOverlayShader, /let dt = max\(u\.environment\.y, 1e-6\)/);
  assert.match(gridOverlayShader, /abs\(velocity\.x\) \* dt \/ h\.x/);
  // Speed mode normalizes by the last reported liquid maximum.
  assert.match(gridOverlayShader, /max\(u\.environment\.z, 1e-4\)/);
});

test("grid overlay velocity sampling honours the packed tall-cell layout", () => {
  // Piecewise tall-cell reconstruction: top world cell = top endpoint dof,
  // every other interior row = bottom dof — the field projection controls.
  assert.match(gridOverlayShader, /let row = select\(0, 1, q\.y == base - 1\)/);
  assert.match(gridOverlayShader, /let packedY = 2 \+ q\.y - base;\n  let stored = vec3i\(textureDimensions\(velocityField\)\)/);
});
