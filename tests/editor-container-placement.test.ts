import assert from "node:assert/strict";
import test from "node:test";
import { sceneContainerBox } from "../lib/core/editor-entity";
import { hoverSceneAt, restInContainer } from "../lib/core/editor-hover";
import { defaultFluidBallRadius_m } from "../lib/core/editor-fluid-volume";
import { sceneDefinitionCamera } from "../lib/core/scene-definition";
import { findSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { projectToViewport, viewportRayForPointer } from "../lib/core/webgpu-camera";
import type { Vec3 } from "../lib/core/model";

/**
 * What a click puts in the tank, and where.
 *
 * The regression these cover is one the picture gives no sign of: when the
 * house set became a stage, the ray that leaves the top of the tank stopped
 * meeting nothing and started meeting the stage floor, metres away and outside
 * the container. Every placement aimed above the waterline resolved out there
 * and was refused, and the ones that still worked all landed on the floor —
 * the tank's whole upper volume, and every corner of it, quietly stopped
 * accepting anything. Nothing threw, so only an assertion about *where a pixel
 * puts an object* can catch it coming back.
 */

const scene = getScenePreset("water-box-dam-break").create();
const camera = sceneDefinitionCamera(findSceneDefinition("water-box-dam-break")!);
const rect = { left: 0, top: 0, width: 1542, height: 784 };
const box = sceneContainerBox(scene);
const radius_m = defaultFluidBallRadius_m(scene);

/** Where a ball dropped on the pixel that shows `target` would come to rest. */
function dropAt(target: Vec3): Vec3 | undefined {
  const projected = projectToViewport(target, camera, rect.width, rect.height);
  const ray = viewportRayForPointer(camera,
    projected.leftFraction * rect.width, projected.topFraction * rect.height, rect);
  return restInContainer(scene, ray, hoverSceneAt(scene, [], ray), radius_m);
}

const corners = (["x", "y", "z"] as const).reduce<Vec3[]>(
  (points, axis) => points.flatMap((point) =>
    [box.min[axis], box.max[axis]].map((value) => ({ ...point, [axis]: value }))),
  [{ x: 0, y: 0, z: 0 }]);

test("a ball dropped at any container corner lands whole, inside the tank", () => {
  for (const corner of corners) {
    const centre = dropAt(corner);
    assert.ok(centre, `no placement for the corner at ${JSON.stringify(corner)}`);
    for (const axis of ["x", "y", "z"] as const) {
      // The whole ball, not merely its centre: authoring the centre on the
      // corner left an eighth of the ball in the tank and the rest in the wall.
      assert.ok(centre[axis] >= box.min[axis] + radius_m - 1e-9
        && centre[axis] <= box.max[axis] - radius_m + 1e-9,
      `corner ${JSON.stringify(corner)} placed ${axis}=${centre[axis]} outside the tank`);
    }
  }
});

test("each corner is reachable as its own placement, not one shared point", () => {
  const placed = corners.map(dropAt).filter((centre) => centre !== undefined);
  assert.equal(placed.length, corners.length);
  // Seven of the eight sit in their own corner; the near top one looks down the
  // length of the tank, where the sight-line is genuinely ambiguous.
  const nestled = placed.filter((centre) => (["x", "y", "z"] as const).every((axis) =>
    Math.abs(centre[axis] - (box.min[axis] + radius_m)) < 1e-6
    || Math.abs(centre[axis] - (box.max[axis] - radius_m)) < 1e-6));
  assert.ok(nestled.length >= 7, `only ${nestled.length} corners nestled into the corner`);
});

test("aiming up the tank places up the tank, rather than always on the floor", () => {
  const heights = new Set<number>();
  for (let step = 0; step <= 8; step += 1) {
    const centre = dropAt({ x: 0, y: (step / 8) * box.max.y, z: 0 });
    if (centre) heights.add(Number(centre.y.toFixed(3)));
  }
  // Every reachable drop resolved to the single floor height while the stage
  // floor was answering for the tank's upper volume.
  assert.ok(heights.size >= 5, `aiming up the tank produced ${heights.size} distinct heights`);
});

test("a press that is not aimed at the tank is still refused", () => {
  const ray = viewportRayForPointer(camera, 8, rect.height - 8, rect);
  assert.equal(restInContainer(scene, ray, hoverSceneAt(scene, [], ray), radius_m), undefined);
});
