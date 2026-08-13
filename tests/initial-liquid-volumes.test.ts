import assert from "node:assert/strict";
import test from "node:test";
import {
  initialLiquidFractionAtCell,
  initialLiquidVolumeContainsPoint,
  initialLiquidVolumeSignedDistance,
} from "../lib/initial-fluid";
import { cloneScene, defaultScene, validateScene, type InitialLiquidHemisphere } from "../lib/model";
import { cm12Scene } from "../lib/cm12-paper-scenes";

const leftHemisphere: InitialLiquidHemisphere = {
  shape: "hemisphere",
  center_m: { x: 0, y: 1, z: 0 },
  radius_m: 1,
  outwardNormal: { x: 1, y: 0, z: 0 },
};

test("a hemisphere is a ball intersected with its retained half-space", () => {
  assert.equal(initialLiquidVolumeContainsPoint(leftHemisphere, { x: -0.5, y: 1, z: 0 }), true);
  assert.equal(initialLiquidVolumeContainsPoint(leftHemisphere, { x: 0.5, y: 1, z: 0 }), false);
  assert.equal(initialLiquidVolumeContainsPoint(leftHemisphere, { x: -1.1, y: 1, z: 0 }), false);
  assert.ok(initialLiquidVolumeSignedDistance(leftHemisphere, { x: -0.5, y: 1, z: 0 }) < 0);
  assert.ok(initialLiquidVolumeSignedDistance(leftHemisphere, { x: 0.1, y: 1, z: 0 }) > 0);
});

test("analytic volume seeding retains fractional curved boundary cells", () => {
  const scene = cloneScene(defaultScene);
  scene.container = { ...scene.container, width_m: 2, height_m: 2, depth_m: 2, fillFraction: 0 };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.initialLiquidVolumes = [leftHemisphere];
  // This left-side cell straddles the curved surface: it is neither discarded
  // by a centre-only test nor promoted to a whole cell.
  const fraction = initialLiquidFractionAtCell(scene, 0, 1, 1, [4, 4, 4], false);
  assert.ok(fraction > 0 && fraction < 1, `fraction ${fraction}`);
});

test("the spherical paper cases use the unified initial-volume vocabulary", () => {
  const dam = cm12Scene("cm12-figure-8");
  const drop = cm12Scene("cm12-figure-12");
  assert.equal(dam.fluid.initialLiquidVolumes?.[0]?.shape, "hemisphere");
  assert.equal(drop.fluid.initialLiquidVolumes?.[0]?.shape, "sphere");
  assert.deepEqual(validateScene(dam), []);
  assert.deepEqual(validateScene(drop), []);
});

test("hemisphere normals are validated as geometry", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialLiquidVolumes = [{ ...leftHemisphere, outwardNormal: { x: 0, y: 0, z: 0 } }];
  assert.match(validateScene(scene).join("\n"), /outward normal must be finite and non-zero/);
});
