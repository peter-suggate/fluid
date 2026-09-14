import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { createInitialLevelSetGeometryWGSL } from
  "../lib/methods/adaptive-volume/levelset-initial-geometry";

const authoredReturn = (
  top: "open" | "closed",
  shape: "box" | "sphere" = "box",
  maximumY = 8,
) => {
  const scene = cloneScene(defaultScene);
  scene.container = { ...scene.container, width_m: 8, height_m: 8, depth_m: 8,
    fillFraction: 0, top, shape };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialHeightField;
  scene.fluid.initialLiquidVolumes = [{
    shape: "box",
    min_m: { x: -1, y: 0, z: -4 },
    max_m: { x: 1, y: maximumY, z: 4 },
  }];
  const wgsl = createInitialLevelSetGeometryWGSL(scene, [32, 32, 32], .25);
  const match = wgsl.match(/return (.+)\/0\.25;\n}/);
  assert.ok(match, "authored phi expression must remain inspectable");
  return match[1]!;
};

test("closed physical wall planes are continuation boundaries, not authored box surfaces", () => {
  const expression = authoredReturn("closed");
  assert.match(expression, /point\.x/);
  assert.doesNotMatch(expression, /point\.y/);
  assert.doesNotMatch(expression, /point\.z/);
});

test("an open top remains a real authored free surface", () => {
  const expression = authoredReturn("open");
  assert.match(expression, /point\.y-8\.0/);
  assert.doesNotMatch(expression, /0\.0-point\.y/,
    "the physical floor still continues the liquid");
  assert.doesNotMatch(expression, /point\.z/);
});

test("an interior box plane remains a real free surface in a closed tank", () => {
  assert.match(authoredReturn("closed", "box", 6), /point\.y-6\.0/);
});

test("axis planes are retained when rectangular wall continuation does not apply", () => {
  const expression = authoredReturn("closed", "sphere");
  for (const axis of ["x", "y", "z"]) assert.match(expression, new RegExp(`point\\.${axis}`));
});
