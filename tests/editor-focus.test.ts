import assert from "node:assert/strict";
import test from "node:test";
import type { AxisConstraint } from "../lib/core/editor-axis-constraint";
import type { EditorRay } from "../lib/core/editor-entity";
import type { Vec3 } from "../lib/core/model";
import {
  FOCUS_KEY_HOLD_MS, FocusKeyHold, FocusSession, focusAxes, focusGuide, focusLockHint, focusLockKeysLive,
  focusLockLabel, focusOrPick, projectFocus, resolveFocus, type FocusPick,
} from "../lib/core/editor-focus";

const close = (actual: Vec3, expected: Vec3, message?: string) => {
  for (const axis of ["x", "y", "z"] as const) assert.ok(Math.abs(actual[axis] - expected[axis]) < 1e-9, `${message ?? ""} ${axis}: ${actual[axis]} vs ${expected[axis]}`);
};
/** A camera up and back, looking at `target`. */
const aim = (target: Vec3, eye: Vec3 = { x: 0, y: 3, z: 5 }): EditorRay => {
  const d = { x: target.x - eye.x, y: target.y - eye.y, z: target.z - eye.z };
  return { origin: eye, direction: d };
};
/** A floor at y = 0 as the consumer's own pick. */
const floor: FocusPick = (ray) => {
  if (!(ray.direction.y < 0)) return undefined;
  const t = -ray.origin.y / ray.direction.y;
  return { point: { x: ray.origin.x + t * ray.direction.x, y: 0, z: ray.origin.z + t * ray.direction.z },
    normal: { x: 0, y: 1, z: 0 }, distance_m: t };
};

test("only one- and two-axis constraints lock the focus", () => {
  assert.equal(focusAxes(undefined), undefined);
  assert.equal(focusAxes([]), undefined);
  assert.equal(focusAxes(["x", "y", "z"]), undefined);
  assert.deepEqual(focusAxes(["y"]), ["y"]);
  assert.deepEqual(focusAxes(["x", "z"]), ["x", "z"]);
});

test("unlocked resolution is exactly the consumer's pick", () => {
  const ray = aim({ x: 1, y: 0, z: -1 });
  const picked = floor(ray);
  assert.deepEqual(resolveFocus(ray, undefined, floor), picked);
  assert.deepEqual(focusOrPick(undefined, ray, floor), picked);
  const session = new FocusSession({ constraint: () => undefined });
  assert.deepEqual(session.resolve(ray, floor), picked);
  assert.equal(focusOrPick(session, aim({ x: 0, y: 10, z: 0 }), floor), undefined, "no ground plane is invented");
});

test("a plane lock holds the locked coordinate exactly and faces the viewer", () => {
  const anchor = { x: 0.3, y: 1.25, z: -0.4 };
  const target = { x: 2, y: 1.25, z: -3 };
  const focus = projectFocus(aim(target), ["x", "z"], anchor)!;
  close(focus.point, target);
  assert.equal(focus.point.y, anchor.y, "height copied, not recomputed");
  assert.deepEqual(focus.normal, { x: 0, y: 1, z: 0 }, "eye is above the plane");
  assert.equal(focus.locked, true);
  const side = projectFocus(aim({ x: 0.3, y: 2, z: 1 }), ["y", "z"], anchor)!;
  assert.equal(side.point.x, 0.3);
  assert.deepEqual(side.normal, { x: -1, y: 0, z: 0 }, "the eye is on the -x side of x = 0.3");
});

test("an axis lock is the closest approach to the line, with the other coordinates pinned", () => {
  const anchor = { x: 1, y: 0, z: -1 };
  const focus = projectFocus(aim({ x: 1, y: 2, z: -1 }), ["y"], anchor)!;
  close(focus.point, { x: 1, y: 2, z: -1 });
  assert.equal(focus.point.x, 1);
  assert.equal(focus.point.z, -1);
  // The normal is perpendicular to the axis and points back to the eye.
  assert.ok(Math.abs(focus.normal.y) < 1e-12);
  assert.ok(focus.normal.z > 0);
});

test("grazing, behind-the-eye and out-of-reach locks resolve to nothing", () => {
  const anchor = { x: 0, y: 1, z: 0 };
  assert.equal(projectFocus({ origin: { x: 0, y: 1, z: 5 }, direction: { x: 0, y: 0, z: -1 } }, ["x", "z"], anchor), undefined);
  assert.equal(projectFocus({ origin: { x: 0, y: 3, z: 5 }, direction: { x: 0, y: 1, z: -1 } }, ["x", "z"], anchor), undefined);
  assert.equal(projectFocus({ origin: { x: 0, y: 5, z: 0 }, direction: { x: 0, y: -1, z: 0 } }, ["y"], anchor), undefined);
  assert.ok(projectFocus(aim({ x: 0, y: 1, z: -40 }), ["x", "z"], anchor));
  assert.equal(projectFocus(aim({ x: 0, y: 1, z: -40 }), ["x", "z"], anchor, 10), undefined);
});

test("a session anchors at the press point, then holds the lock through the drag", () => {
  let constraint: AxisConstraint = ["x", "z"];
  const session = new FocusSession({ constraint: () => constraint });
  const press = session.resolve(aim({ x: 0.5, y: 0, z: 0 }), floor)!;
  assert.deepEqual(session.anchor, { x: 0.5, y: 0, z: 0 });
  assert.deepEqual(press.normal, { x: 0, y: 1, z: 0 }, "the anchoring sample keeps the surface normal");
  // Lift the anchor onto Y by switching mid-drag: the hand stays where it is.
  const slid = session.resolve(aim({ x: 1, y: 0, z: -1 }), floor)!;
  close(slid.point, { x: 1, y: 0, z: -1 });
  constraint = ["y"];
  const lifted = session.resolve(aim({ x: 1, y: 2, z: -1 }), floor)!;
  close(lifted.point, { x: 1, y: 2, z: -1 }, "re-anchored at the last focus, not at the press");
  // Back to a horizontal plane at the lifted height.
  constraint = ["x", "z"];
  const level = session.resolve(aim({ x: -1, y: 2, z: -2 }), floor)!;
  assert.ok(Math.abs(level.point.y - 2) < 1e-9);
  constraint = undefined;
  assert.deepEqual(session.resolve(aim({ x: 0, y: 0, z: 0 }), floor), floor(aim({ x: 0, y: 0, z: 0 })));
});

test("a lock engaged mid-stroke anchors at the current focus; a press in open air uses the last anchor", () => {
  let constraint: AxisConstraint = undefined;
  const session = new FocusSession({ constraint: () => constraint });
  session.resolve(aim({ x: 0, y: 0, z: -1 }), floor);
  constraint = ["y"];
  const up = session.resolve(aim({ x: 0, y: 1, z: -1 }), floor)!;
  close(up.point, { x: 0, y: 1, z: -1 });
  const air = new FocusSession({ constraint: () => ["x", "z"], fallback: { x: 0, y: 1.5, z: 0 } });
  const sky = aim({ x: 0.5, y: 1.5, z: 0 }, { x: 0, y: 2, z: 5 });
  const stood = air.resolve(sky, () => undefined)!;
  close(stood.point, { x: 0.5, y: 1.5, z: 0 });
  assert.equal(new FocusSession({ constraint: () => ["y"] }).resolve(sky, () => undefined), undefined,
    "no pick and no previous anchor: nothing to stand on");
});

test("lock keys tap to latch, hold to spring, ignore repeats", () => {
  const keys = new FocusKeyHold();
  const yLock = keys.down({ key: "y", timeStamp: 0 }, undefined);
  assert.deepEqual(yLock, ["y"]);
  assert.equal(keys.holding, true);
  assert.equal(keys.down({ key: "y", repeat: true, timeStamp: 30 }, yLock as AxisConstraint), "ignore");
  assert.equal(keys.up({ key: "y", timeStamp: 100 }, yLock as AxisConstraint), "keep", "a tap latches");
  assert.equal(keys.holding, false);
  const plane = keys.down({ key: "Y", shiftKey: true, timeStamp: 1000 }, yLock as AxisConstraint);
  assert.deepEqual(plane, ["x", "z"]);
  assert.equal(keys.up({ key: "y", timeStamp: 1000 + FOCUS_KEY_HOLD_MS }, plane as AxisConstraint), "release", "a hold springs back");
  assert.equal(keys.down({ key: "y", timeStamp: 0 }, ["y"]), undefined, "same key releases");
  assert.equal(keys.down({ key: "q", timeStamp: 0 }, undefined), "ignore");
});

test("lock keys are live with a selection or a placing tool in EDIT", () => {
  assert.equal(focusLockKeysLive({ viewportMode: "interact" }), false);
  assert.equal(focusLockKeysLive({ viewportMode: "camera", selection: { kind: "tank" } }), true);
  assert.equal(focusLockKeysLive({ viewportMode: "interact", voxelToolId: "pen" }), true);
  assert.equal(focusLockKeysLive({ viewportMode: "camera", voxelToolId: "pen" }), false);
  assert.equal(focusLockKeysLive({ viewportMode: "interact", armedGesture: "fluid-ball" }), true);
  assert.equal(focusLockKeysLive({ viewportMode: "interact", armedGesture: "body-drag" }), true);
  assert.equal(focusLockKeysLive({ viewportMode: "interact", armedGesture: "fluid-erase" }), false,
    "Y still toggles ERASE where nothing places");
});

test("guide and caption name the lock and its release", () => {
  const anchor = { x: 0, y: 1, z: 0 };
  const axis = focusGuide({ axes: ["y"], anchor }, 2);
  assert.equal(axis.kind, "paths");
  assert.deepEqual(axis.kind === "paths" && axis.paths, [[{ x: 0, y: -1, z: 0 }, { x: 0, y: 3, z: 0 }]]);
  const plane = focusGuide({ axes: ["x", "z"], anchor }, 2);
  assert.ok(plane.kind === "paths" && plane.paths.length === 3
    && plane.paths.every((path) => path.every((point) => point.y === 1)), "a plane guide lies in the plane");
  assert.equal(focusLockLabel(["y"]), "ALONG Y");
  assert.equal(focusLockLabel(["z", "x"]), "ON XZ PLANE");
  assert.equal(focusLockHint(["y"]), "Y or Esc releases");
  assert.equal(focusLockHint(["x", "z"]), "Shift+Y or Esc releases");
});
