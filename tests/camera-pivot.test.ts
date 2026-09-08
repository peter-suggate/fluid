import assert from "node:assert/strict";
import test from "node:test";
import { containerViewPoint, gesturePivot, viewDepthPoint } from "../lib/core/camera-pivot";
import { CAMERA_DISTANCE_RANGE, cameraBasis, cameraPosition, dot, orbit, orbitAbout, pan, sub, zoom, zoomToward } from "../lib/core/math";
import { cloneScene, defaultScene, type CameraState, type SceneDescription, type Vec3 } from "../lib/core/model";
import { projectToViewport, viewportRayForPixel, viewportWorldPerPixel } from "../lib/core/webgpu-camera";
import { targetAtRay } from "../lib/core/editor-probe-catalog";
import type { EditorEntityContext } from "../lib/core/editor-entity";

const VIEW: CameraState = {
  azimuth_rad: 0.6, elevation_rad: 0.35, distance_m: 2.4,
  target_m: { x: 0.1, y: 0.3, z: -0.05 },
};
const WIDTH = 800, HEIGHT = 460;

function context(scene: SceneDescription): EditorEntityContext {
  return { scene, bodies: [], pickingAvailable: true };
}

const away = (a: Vec3, b: Vec3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

/** A world point behind a given canvas pixel, a given way along its ray. */
function pointBehindPixel(camera: CameraState, pixelX: number, pixelY: number, along_m: number): Vec3 {
  const ray = viewportRayForPixel(camera, pixelX, pixelY, WIDTH, HEIGHT);
  return { x: ray.origin.x + ray.direction.x * along_m, y: ray.origin.y + ray.direction.y * along_m, z: ray.origin.z + ray.direction.z * along_m };
}

// The persistent-pivot case must be the plain spherical orbit, so nothing
// about the feel of a drag changed for a press on empty space.
test("orbiting about the look-at target is the spherical orbit, bit for bit", () => {
  let rigid = VIEW, spherical = VIEW;
  for (let step = 0; step < 12; step += 1) {
    rigid = orbitAbout(rigid, VIEW.target_m, 17, -9);
    spherical = orbit(spherical, 17, -9);
  }
  assert.ok(away(cameraPosition(rigid), cameraPosition(spherical)) < 1e-9, "the eye diverged");
  assert.ok(away(rigid.target_m, VIEW.target_m) < 1e-9, "the target moved");
  assert.equal(rigid.azimuth_rad, spherical.azimuth_rad);
  assert.equal(rigid.elevation_rad, spherical.elevation_rad);
  assert.equal(rigid.distance_m, VIEW.distance_m);
});

// The visible meaning of "orbit about what you grabbed": the grabbed point
// does not move on screen, however far off-centre it was.
test("an off-centre pivot stays under the cursor through an orbit", () => {
  for (const [px, py, along_m] of [[120, 80, 1.1], [700, 400, 3.2], [400, 230, 2.4], [40, 440, 0.6]] as const) {
    const pivot = pointBehindPixel(VIEW, px, py, along_m);
    const before = projectToViewport(pivot, VIEW, WIDTH, HEIGHT);
    let view = VIEW;
    for (let step = 0; step < 12; step += 1) view = orbitAbout(view, pivot, 17, -9);
    const after = projectToViewport(pivot, view, WIDTH, HEIGHT);
    assert.ok(Math.abs(after.leftFraction - before.leftFraction) < 1e-6, `pixel (${px},${py}) drifted to ${after.leftFraction}`);
    assert.ok(Math.abs(after.topFraction - before.topFraction) < 1e-6, `pixel (${px},${py}) drifted to ${after.topFraction}`);
    assert.ok(Math.abs(away(cameraPosition(view), pivot) - along_m) < 1e-9, "the eye changed its distance from the pivot");
    assert.ok(Math.abs(view.distance_m - VIEW.distance_m) < 1e-12, "the target's depth changed");
  }
});

test("an orbit about any pivot is a turn by the requested angles", () => {
  const pivot = pointBehindPixel(VIEW, 700, 400, 3.2);
  const turned = orbitAbout(VIEW, pivot, 50, -30);
  assert.ok(Math.abs(turned.azimuth_rad - (VIEW.azimuth_rad - 50 * 0.007)) < 1e-12);
  assert.ok(Math.abs(turned.elevation_rad - (VIEW.elevation_rad - 30 * 0.007)) < 1e-12);
  const pole = orbitAbout(VIEW, pivot, 0, 10_000);
  assert.equal(pole.elevation_rad, 1.35, "the turntable stops short of the pole");
  assert.ok(Math.abs(away(cameraPosition(pole), pivot) - 3.2) < 1e-9, "clamping the tilt still moved rigidly");
});

// Zoom to cursor: the point under the pointer is the one the eye closes on,
// and it does not move on screen while everything around it grows.
test("the point under the cursor stays put through a zoom, and the eye closes on it", () => {
  for (const [px, py, along_m] of [[120, 80, 1.1], [700, 400, 3.2], [400, 230, 2.4]] as const) {
    const pivot = pointBehindPixel(VIEW, px, py, along_m);
    const before = projectToViewport(pivot, VIEW, WIDTH, HEIGHT);
    let view = VIEW;
    for (let step = 0; step < 5; step += 1) view = zoomToward(view, pivot, -100);
    const after = projectToViewport(pivot, view, WIDTH, HEIGHT);
    assert.ok(Math.abs(after.leftFraction - before.leftFraction) < 1e-6, `pixel (${px},${py}) drifted to ${after.leftFraction}`);
    assert.ok(Math.abs(after.topFraction - before.topFraction) < 1e-6, `pixel (${px},${py}) drifted to ${after.topFraction}`);
    assert.ok(away(cameraPosition(view), pivot) < away(cameraPosition(VIEW), pivot), "the eye did not close on the pivot");
    assert.ok(away(cameraBasis(view).forward, cameraBasis(VIEW).forward) < 1e-9, "the view direction turned");
    assert.ok(Math.abs(view.distance_m - VIEW.distance_m * Math.exp(-0.5)) < 1e-9, "the target did not scale with the frame");
  }
});

test("zooming toward a surface never pushes the eye through it", () => {
  const pivot = pointBehindPixel(VIEW, 700, 400, 0.3);
  let view = VIEW;
  for (let step = 0; step < 400; step += 1) view = zoomToward(view, pivot, -100);
  assert.ok(away(cameraPosition(view), pivot) >= CAMERA_DISTANCE_RANGE.minimum_m - 1e-9);
  assert.ok(dot(sub(pivot, cameraPosition(view)), cameraBasis(view).forward) > 0, "the pivot ended up behind the eye");
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

// Pan is "grab the world": the surface under the cursor moves with the
// cursor pixel for pixel, at whatever depth it was grabbed.
test("a pan carries the grabbed surface with the cursor one-for-one", () => {
  for (const along_m of [0.5, 2.4, 7]) {
    const grabbed = pointBehindPixel(VIEW, 300, 200, along_m);
    const depth_m = dot(sub(grabbed, cameraBasis(VIEW).position), cameraBasis(VIEW).forward);
    const before = projectToViewport(grabbed, VIEW, WIDTH, HEIGHT);
    const dx = 140, dy = -60;
    const after = projectToViewport(grabbed, pan(VIEW, dx, dy, viewportWorldPerPixel(VIEW, depth_m, HEIGHT)), WIDTH, HEIGHT);
    assert.ok(Math.abs((after.leftFraction - before.leftFraction) * WIDTH - dx) < 1e-6, `moved ${(after.leftFraction - before.leftFraction) * WIDTH} px, not ${dx}`);
    assert.ok(Math.abs((after.topFraction - before.topFraction) * HEIGHT - dy) < 1e-6, `moved ${(after.topFraction - before.topFraction) * HEIGHT} px, not ${dy}`);
  }
});

// An empty tank has nothing a CPU probe can pick but its own far wall, and
// `pickRoomExitFace` answers with the wall the ray *leaves* through: orbiting
// about the back of the tank swings everything in front of it.
test("an empty tank pivots on its volume rather than on the wall behind it", () => {
  const scene = cloneScene(defaultScene);
  scene.systems = { ...scene.systems, fluid: false };
  const camera: CameraState = { azimuth_rad: 0.5, elevation_rad: 0.2, distance_m: 2, target_m: { x: 0, y: 0.3, z: 0 } };
  const ray = viewportRayForPixel(camera, 400, 230, WIDTH, HEIGHT);
  const wall = targetAtRay(context(scene), ray);
  assert.equal(wall.kind, "tank-wall", "this scene is the case the rule is for");
  const pivot = gesturePivot(context(scene), camera, ray);
  assert.deepEqual(pivot, containerViewPoint(scene, ray));
  assert.ok(away(pivot, ray.origin) < wall.distance_m, "the pivot is nearer than the wall behind it");
  const c = scene.container, slack = 1e-6;
  assert.ok(pivot.x >= -0.5 * c.width_m - slack && pivot.x <= 0.5 * c.width_m + slack, "pivot left the tank in x");
  assert.ok(pivot.y >= -slack && pivot.y <= c.height_m + slack, "pivot left the tank in y");
  assert.ok(pivot.z >= -0.5 * c.depth_m - slack && pivot.z <= 0.5 * c.depth_m + slack, "pivot left the tank in z");
});

// The other half of the rule: anything the reader would be told they are
// pointing at is the pivot, and the container fallback never overrides it.
test("a surface under the cursor is the pivot", () => {
  const scene = cloneScene(defaultScene);
  const camera: CameraState = { azimuth_rad: 0.5, elevation_rad: 0.2, distance_m: 2, target_m: { x: 0, y: 0.3, z: 0 } };
  const ray = viewportRayForPixel(camera, 400, 230, WIDTH, HEIGHT);
  const target = targetAtRay(context(scene), ray);
  assert.equal(target.kind, "entity", "the dam block is what this view is centred on");
  assert.deepEqual(gesturePivot(context(scene), camera, ray), target.point_m);
});

// A cursor over the sky, past the tank: no surface, no container span. The
// pivot is still under the cursor, at the depth the camera was working at.
test("a cursor over nothing pivots on the cursor's ray at the view depth", () => {
  const scene = cloneScene(defaultScene);
  const camera: CameraState = { azimuth_rad: 0.5, elevation_rad: 0.2, distance_m: 2, target_m: { x: 0, y: 0.3, z: 0 } };
  const ray = viewportRayForPixel(camera, 790, 2, WIDTH, HEIGHT);
  assert.equal(targetAtRay(context(scene), ray).kind, "room", "this pixel must be over nothing");
  assert.equal(containerViewPoint(scene, ray), undefined, "and past the tank");
  const pivot = gesturePivot(context(scene), camera, ray);
  assert.deepEqual(pivot, viewDepthPoint(camera, ray));
  const depth_m = dot(sub(pivot, cameraBasis(camera).position), cameraBasis(camera).forward);
  assert.ok(Math.abs(depth_m - camera.distance_m) < 1e-9, "the pivot is not at the target's depth");
  const projection = projectToViewport(pivot, camera, WIDTH, HEIGHT);
  assert.ok(Math.abs(projection.leftFraction * WIDTH - 790.5) < 1e-6 && Math.abs(projection.topFraction * HEIGHT - 2.5) < 1e-6, "the pivot left the cursor");
});

test("a ray that misses the container has no container point to fall back to", () => {
  const scene = cloneScene(defaultScene);
  assert.equal(containerViewPoint(scene, { origin: { x: 0, y: 40, z: 0 }, direction: { x: 0, y: 1, z: 0 } }), undefined);
});
