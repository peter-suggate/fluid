import assert from "node:assert/strict";
import test from "node:test";
import { containerViewPoint, viewportCentrePivot, viewportCentreRay } from "../lib/core/camera-pivot";
import { CAMERA_DISTANCE_RANGE, cameraBasis, cameraPosition, orbit, retarget, zoom } from "../lib/core/math";
import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "../lib/core/model";
import { projectToViewport } from "../lib/core/webgpu-camera";
import { targetAtRay } from "../lib/core/editor-probe-catalog";
import type { EditorEntityContext } from "../lib/core/editor-entity";

const VIEW: CameraState = {
  azimuth_rad: 0.6, elevation_rad: 0.35, distance_m: 2.4,
  target_m: { x: 0.1, y: 0.3, z: -0.05 },
};

function context(scene: SceneDescription): EditorEntityContext {
  return { scene, bodies: [], pickingAvailable: true };
}

const away = (a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }) =>
  Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

// The property the whole design rests on: moving the pivot onto the point the
// frame is centred on is invisible. If this fails, every orbit and every wheel
// notch begins with a jump.
test("retargeting onto the centre of the viewport does not move the frame", () => {
  const ray = viewportCentreRay(VIEW);
  for (const depth_m of [0.4, 1.2, 2.4, 9]) {
    const pivot = {
      x: ray.origin.x + ray.direction.x * depth_m,
      y: ray.origin.y + ray.direction.y * depth_m,
      z: ray.origin.z + ray.direction.z * depth_m,
    };
    const anchored = retarget(VIEW, pivot);
    assert.ok(away(cameraPosition(anchored), cameraPosition(VIEW)) < 1e-9, "the eye moved");
    const before = cameraBasis(VIEW).forward, after = cameraBasis(anchored).forward;
    assert.ok(away(before, after) < 1e-9, "the view direction moved");
    assert.ok(Math.abs(anchored.distance_m - depth_m) < 1e-9, "the pivot is not the new distance");
  }
});

// What lets the wheel anchor on every event instead of tracking a gesture.
test("retargeting is idempotent once the camera is on the pivot", () => {
  const ray = viewportCentreRay(VIEW);
  const pivot = { x: ray.origin.x + ray.direction.x, y: ray.origin.y + ray.direction.y, z: ray.origin.z + ray.direction.z };
  const once = retarget(VIEW, pivot);
  assert.deepEqual(retarget(once, pivot), once);
});

test("a pivot beyond the wheel's reach is declined, never clamped onto", () => {
  const ray = viewportCentreRay(VIEW);
  const far = CAMERA_DISTANCE_RANGE.maximum_m * 2;
  const pivot = { x: ray.origin.x + ray.direction.x * far, y: ray.origin.y + ray.direction.y * far, z: ray.origin.z + ray.direction.z * far };
  assert.deepEqual(retarget(VIEW, pivot), VIEW, "a declined pivot must leave the camera exactly as it was");
});

// Orbiting about the pivot keeps it in the middle of the frame — the visible
// meaning of "the centre of rotation is what is at the centre of the viewport".
test("the pivot stays centred through an orbit", () => {
  const ray = viewportCentreRay(VIEW);
  const pivot = { x: ray.origin.x + ray.direction.x * 1.5, y: ray.origin.y + ray.direction.y * 1.5, z: ray.origin.z + ray.direction.z * 1.5 };
  let view = retarget(VIEW, pivot);
  for (let step = 0; step < 12; step += 1) view = orbit(view, 17, -9);
  const projection = projectToViewport(pivot, view, 800, 460);
  assert.ok(Math.abs(projection.leftFraction - 0.5) < 1e-6, `drifted to ${projection.leftFraction}`);
  assert.ok(Math.abs(projection.topFraction - 0.5) < 1e-6, `drifted to ${projection.topFraction}`);
});

test("the pivot stays centred through a zoom, and the eye closes on it", () => {
  const ray = viewportCentreRay(VIEW);
  const pivot = { x: ray.origin.x + ray.direction.x * 1.5, y: ray.origin.y + ray.direction.y * 1.5, z: ray.origin.z + ray.direction.z * 1.5 };
  const anchored = retarget(VIEW, pivot);
  const closer = zoom(anchored, -400);
  assert.ok(away(cameraPosition(closer), pivot) < away(cameraPosition(anchored), pivot));
  const projection = projectToViewport(pivot, closer, 800, 460);
  assert.ok(Math.abs(projection.leftFraction - 0.5) < 1e-6);
  assert.ok(Math.abs(projection.topFraction - 0.5) < 1e-6);
});

test("the wheel reaches closer and further than the old 0.65..12 m range", () => {
  assert.ok(CAMERA_DISTANCE_RANGE.minimum_m < 0.65);
  assert.ok(CAMERA_DISTANCE_RANGE.maximum_m > 15, "the widest authored preset opens at 15 m");
  let view = { ...VIEW };
  for (let step = 0; step < 200; step += 1) view = zoom(view, -100);
  assert.equal(view.distance_m, CAMERA_DISTANCE_RANGE.minimum_m);
  for (let step = 0; step < 400; step += 1) view = zoom(view, 100);
  assert.equal(view.distance_m, CAMERA_DISTANCE_RANGE.maximum_m);
});

// An empty tank has nothing a CPU probe can pick but its own far wall, and
// `pickRoomExitFace` answers with the wall the ray *leaves* through: orbiting
// about the back of the tank swings everything in front of it.
test("an empty tank pivots on its volume rather than on the wall behind it", () => {
  const scene = cloneScene(defaultScene);
  scene.systems = { ...scene.systems, fluid: false };
  const camera: CameraState = { azimuth_rad: 0.5, elevation_rad: 0.2, distance_m: 2, target_m: { x: 0, y: 0.3, z: 0 } };
  const ray = viewportCentreRay(camera);
  const wall = targetAtRay(context(scene), ray);
  assert.equal(wall.kind, "tank-wall", "this scene is the case the rule is for");
  const pivot = viewportCentrePivot(context(scene), ray);
  assert.deepEqual(pivot, containerViewPoint(scene, ray));
  assert.ok(away(pivot, ray.origin) < wall.distance_m, "the pivot is nearer than the wall behind it");
  const c = scene.container, slack = 1e-6;
  assert.ok(pivot.x >= -0.5 * c.width_m - slack && pivot.x <= 0.5 * c.width_m + slack, "pivot left the tank in x");
  assert.ok(pivot.y >= -slack && pivot.y <= c.height_m + slack, "pivot left the tank in y");
  assert.ok(pivot.z >= -0.5 * c.depth_m - slack && pivot.z <= 0.5 * c.depth_m + slack, "pivot left the tank in z");
});

// The other half of the rule: anything the reader would be told they are
// pointing at is the pivot, and the container fallback never overrides it.
test("a surface in the middle of the frame is the pivot", () => {
  const scene = cloneScene(defaultScene);
  const camera: CameraState = { azimuth_rad: 0.5, elevation_rad: 0.2, distance_m: 2, target_m: { x: 0, y: 0.3, z: 0 } };
  const ray = viewportCentreRay(camera);
  const target = targetAtRay(context(scene), ray);
  assert.equal(target.kind, "entity", "the dam block is what this view is centred on");
  assert.deepEqual(viewportCentrePivot(context(scene), ray), target.point_m);
});

test("a ray that misses the container has no container point to fall back to", () => {
  const scene = cloneScene(defaultScene);
  assert.equal(containerViewPoint(scene, { origin: { x: 0, y: 40, z: 0 }, direction: { x: 0, y: 1, z: 0 } }), undefined);
});
