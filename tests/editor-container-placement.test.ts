import assert from "node:assert/strict";
import test from "node:test";
import { sceneContainerBox } from "../lib/core/editor-entity";
import { hoverSceneAt, restFluidInWorld, restInContainer } from "../lib/core/editor-hover";
import { defaultFluidBallRadius_m, fluidInteractionDropVolume } from "../lib/core/editor-fluid-volume";
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

test("a ball aimed at the larger hydrostatic pool starts above its free surface", () => {
  const definition = findSceneDefinition("hydrostatic-power-large-offset")!;
  const hydrostatic = definition.build();
  const hydroCamera = sceneDefinitionCamera(definition);
  const surfaceY = hydrostatic.container.fillFraction * hydrostatic.container.height_m;
  const target = { x: 0, y: surfaceY, z: 0 };
  const projected = projectToViewport(target, hydroCamera, rect.width, rect.height);
  const ray = viewportRayForPointer(hydroCamera,
    projected.leftFraction * rect.width, projected.topFraction * rect.height, rect);
  const hover = hoverSceneAt(hydrostatic, [], ray);
  const radius = defaultFluidBallRadius_m(hydrostatic);
  const centre = restInContainer(hydrostatic, ray, hover, radius);

  assert.equal(hover?.kind, "fluid");
  assert.ok(centre);
  assert.ok(centre.y - radius >= surfaceY + hydrostatic.voxelDomain.finestCellSize_m - 1e-9,
    `ball bottom ${centre.y - radius} must start clear of surface ${surfaceY}`);
});

test("a press that is not aimed at the tank is still refused", () => {
  const ray = viewportRayForPointer(camera, 8, rect.height - 8, rect);
  assert.equal(restInContainer(scene, ray, hoverSceneAt(scene, [], ray), radius_m), undefined);
});

test("a liquid interaction may land on the open world floor outside the tank", () => {
  const outsideX = scene.container.width_m / 2 + 4 * radius_m;
  const ray = {
    origin: { x: outsideX, y: scene.container.height_m, z: 0 },
    direction: { x: 0, y: -1, z: 0 },
  };
  const centre = restFluidInWorld(scene, ray,
    hoverSceneAt(scene, [], ray, { scenery: false }), radius_m);
  assert.ok(centre);
  assert.equal(centre.x, outsideX);
  assert.equal(centre.y, radius_m);
  assert.ok(Math.abs(centre.x) > scene.container.width_m / 2);
  assert.equal(fluidInteractionDropVolume(scene, centre, radius_m).center_m.x, outsideX,
    "the runtime interaction must not clamp back to the vessel");
});

test("the tank remains promoted over a later outside-scene hit", () => {
  const projected = projectToViewport({ x: 0, y: box.max.y, z: 0 },
    camera, rect.width, rect.height);
  const ray = viewportRayForPointer(camera,
    projected.leftFraction * rect.width, projected.topFraction * rect.height, rect);
  const outsideHover = {
    kind: "floor" as const,
    position_m: { x: 8, y: 0, z: 8 },
    normal: { x: 0, y: 1, z: 0 },
    distance_m: 20,
    label: "stage floor",
  };
  const promoted = restFluidInWorld(scene, ray, outsideHover, radius_m);
  const established = restInContainer(scene, ray, outsideHover, radius_m);
  assert.deepEqual(promoted, established);
  assert.ok(promoted);
  assert.ok(Math.abs(promoted.x) <= scene.container.width_m / 2 - radius_m + 1e-9);
  assert.ok(Math.abs(promoted.z) <= scene.container.depth_m / 2 - radius_m + 1e-9);
});
