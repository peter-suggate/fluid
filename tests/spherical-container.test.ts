import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, validateScene } from "../lib/model";
import {
  pointInsideSceneContainer,
  sphericalContainerGeometry,
  sphericalContainerOpenFractionAtCell,
} from "../lib/spherical-container";

test("spherical container geometry is centred, closed, and metric", () => {
  const scene = cloneScene(defaultScene);
  scene.container = { ...scene.container, width_m: 8, height_m: 6, depth_m: 10,
    shape: "sphere", top: "closed" };
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(sphericalContainerGeometry(scene), {
    center_m: { x: 0, y: 3, z: 0 }, radius_m: 3,
  });
  assert.equal(pointInsideSceneContainer(scene, { x: 0, y: 3, z: 0 }), true);
  assert.equal(pointInsideSceneContainer(scene, { x: 0, y: 6.01, z: 0 }), false);
  assert.equal(pointInsideSceneContainer(scene, { x: 3.01, y: 3, z: 0 }), false);
});

test("spherical cut-cell quadrature preserves open interior and solid corners", () => {
  const scene = cloneScene(defaultScene);
  scene.container = { ...scene.container, width_m: 8, height_m: 8, depth_m: 8,
    shape: "sphere", top: "closed" };
  const dimensions = [8, 8, 8] as const;
  assert.equal(sphericalContainerOpenFractionAtCell(scene, 3, 3, 3, dimensions), 1);
  assert.equal(sphericalContainerOpenFractionAtCell(scene, 0, 0, 0, dimensions), 0);
  const boundary = sphericalContainerOpenFractionAtCell(scene, 0, 3, 3, dimensions);
  assert.ok(boundary > 0 && boundary < 1);
});

test("an open spherical boundary is rejected", () => {
  const scene = cloneScene(defaultScene);
  scene.container.shape = "sphere";
  scene.container.top = "open";
  assert.match(validateScene(scene).join("\n"), /spherical container must be closed/i);
});
