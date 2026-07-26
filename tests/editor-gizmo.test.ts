import assert from "node:assert/strict";
import test from "node:test";
import {
  closestPointOnAxis,
  gizmoAxisDragPosition,
  gizmoHandleAtPointer,
  gizmoHandleLength_m,
  projectGizmo,
  GIZMO_AXIS_DIRECTIONS,
} from "../lib/editor-gizmo";
import { add, cameraBasis, normalize, scale, sub } from "../lib/math";
import { defaultCamera, type Vec3 } from "../lib/model";
import { projectToViewport, viewportRay, viewportRayForPointer } from "../lib/webgpu-camera";

const WIDTH = 960;
const HEIGHT = 540;

/** Ray through a world point, so the constraint has a known exact answer. */
function rayThrough(target: Vec3) {
  const basis = cameraBasis(defaultCamera);
  return { origin: basis.position, direction: normalize(sub(target, basis.position)) };
}

test("viewport ray and projection are mutual inverses", () => {
  for (const [ndcX, ndcY] of [[0, 0], [0.5, -0.25], [-0.8, 0.6]] as const) {
    const ray = viewportRay(defaultCamera, ndcX, ndcY, WIDTH / HEIGHT);
    const point = add(ray.origin, scale(ray.direction, 3.1));
    const projected = projectToViewport(point, defaultCamera, WIDTH, HEIGHT);
    assert.ok(Math.abs(projected.leftFraction - 0.5 * (ndcX + 1)) < 1e-9);
    assert.ok(Math.abs(projected.topFraction - 0.5 * (1 - ndcY)) < 1e-9);
    assert.ok(projected.depth_m > 0);
  }
});

test("pointer rays agree with the canvas fractions the overlay lays out with", () => {
  const rect = { left: 40, top: 12, width: WIDTH, height: HEIGHT };
  const ray = viewportRayForPointer(defaultCamera, rect.left + 0.75 * WIDTH, rect.top + 0.25 * HEIGHT, rect);
  const projected = projectToViewport(add(ray.origin, scale(ray.direction, 2)), defaultCamera, WIDTH, HEIGHT);
  assert.ok(Math.abs(projected.leftFraction - 0.75) < 1e-9);
  assert.ok(Math.abs(projected.topFraction - 0.25) < 1e-9);
});

test("axis constraint recovers the exact point a ray was aimed at", () => {
  const axisOrigin = { x: 0.1, y: 0.4, z: -0.2 };
  for (const axis of ["x", "y", "z"] as const) {
    const expected = add(axisOrigin, scale(GIZMO_AXIS_DIRECTIONS[axis], 0.37));
    const ray = rayThrough(expected);
    const point = closestPointOnAxis(ray.origin, ray.direction, axisOrigin, GIZMO_AXIS_DIRECTIONS[axis]);
    assert.ok(point, `${axis} axis must resolve`);
    for (const component of ["x", "y", "z"] as const) {
      assert.ok(Math.abs(point[component] - expected[component]) < 1e-9, `${axis}.${component}`);
    }
  }
});

test("a ray parallel to the axis is rejected instead of shooting the body away", () => {
  const axisOrigin = { x: 0, y: 0.4, z: 0 };
  const parallel = { origin: { x: -4, y: 0.4, z: 0 }, direction: { x: 1, y: 0, z: 0 } };
  assert.equal(closestPointOnAxis(parallel.origin, parallel.direction, axisOrigin, GIZMO_AXIS_DIRECTIONS.x), undefined);
  assert.equal(gizmoAxisDragPosition(parallel.origin, parallel.direction, "x", axisOrigin, { x: 0, y: 0, z: 0 }), undefined);
});

test("the grab offset keeps an axis drag from jumping on the first move", () => {
  const start = { x: 0, y: 0.4, z: 0 };
  const grabbed = add(start, { x: 0.25, y: 0, z: 0 });
  const downRay = rayThrough(grabbed);
  const grabPoint = closestPointOnAxis(downRay.origin, downRay.direction, start, GIZMO_AXIS_DIRECTIONS.x);
  assert.ok(grabPoint);
  const grabOffset = sub(start, grabPoint);
  const held = gizmoAxisDragPosition(downRay.origin, downRay.direction, "x", start, grabOffset);
  assert.ok(held);
  assert.ok(Math.hypot(held.x - start.x, held.y - start.y, held.z - start.z) < 1e-9, "no motion without pointer motion");

  const moveRay = rayThrough(add(grabbed, { x: 0.5, y: 0, z: 0 }));
  const moved = gizmoAxisDragPosition(moveRay.origin, moveRay.direction, "x", start, grabOffset);
  assert.ok(moved);
  assert.ok(Math.abs(moved.x - (start.x + 0.5)) < 1e-9, "motion tracks the pointer along the axis");
  assert.ok(Math.abs(moved.y - start.y) < 1e-12 && Math.abs(moved.z - start.z) < 1e-12, "off-axis components are pinned");
});

test("handle length holds a constant on-screen size across camera distance", () => {
  const near = gizmoHandleLength_m(1);
  const far = gizmoHandleLength_m(4);
  assert.ok(Math.abs(far / near - 4) < 1e-12, "world length must scale with eye depth");
  const position = { x: 0, y: 0.4, z: 0 };
  const gizmo = projectGizmo(position, defaultCamera, WIDTH, HEIGHT);
  const spans = gizmo.handles.map((handle) => Math.hypot(
    (handle.tip.leftFraction - gizmo.origin.leftFraction) * WIDTH,
    (handle.tip.topFraction - gizmo.origin.topFraction) * HEIGHT,
  ));
  for (const span of spans) assert.ok(span > 12 && span < 0.5 * HEIGHT, `handle span ${span}px must be grabbable but not huge`);
});

test("pointer hit test prefers the centre and picks the nearest axis", () => {
  const position = { x: 0, y: 0.4, z: 0 };
  const gizmo = projectGizmo(position, defaultCamera, WIDTH, HEIGHT);
  const origin = { x: gizmo.origin.leftFraction * WIDTH, y: gizmo.origin.topFraction * HEIGHT };
  assert.equal(gizmoHandleAtPointer(gizmo, origin, WIDTH, HEIGHT), "free");
  for (const handle of gizmo.handles) {
    const tip = { x: handle.tip.leftFraction * WIDTH, y: handle.tip.topFraction * HEIGHT };
    assert.equal(gizmoHandleAtPointer(gizmo, tip, WIDTH, HEIGHT), handle.axis);
  }
  assert.equal(gizmoHandleAtPointer(gizmo, { x: origin.x + 400, y: origin.y + 260 }, WIDTH, HEIGHT), undefined);
});

test("a gizmo behind the camera cannot be grabbed", () => {
  const basis = cameraBasis(defaultCamera);
  const behind = add(basis.position, scale(basis.forward, -1.5));
  const gizmo = projectGizmo(behind, defaultCamera, WIDTH, HEIGHT);
  assert.ok(gizmo.origin.depth_m < 0);
  assert.equal(gizmoHandleAtPointer(gizmo, { x: 0.5 * WIDTH, y: 0.5 * HEIGHT }, WIDTH, HEIGHT), undefined);
});
