import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  BOUNDS_AXES,
  boundsAxisConstraintLabel,
  boundsAxisForKey,
  boundsDragAxes,
  boundsDragAxisDirection,
  constrainBoundsHandle,
  toggleBoundsAxisConstraint,
} from "../lib/editor-bounds-axis";
import { dragFluidBodyBox, fluidBodyBox, fluidBodyHandleById } from "../lib/editor-fluid-body";
import { dragTankExtents, tankBox } from "../lib/editor-tank";
import { getScenePreset } from "../lib/scenes";
import { sceneCellSizes_m } from "../lib/scene-lattice";
import { useUIStore } from "../lib/stores/ui-store";
import type { SceneDescription } from "../lib/model";

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

test("axis keys name axes, and nothing else does", () => {
  assert.equal(boundsAxisForKey("x"), "x");
  assert.equal(boundsAxisForKey("Y"), "y");
  assert.equal(boundsAxisForKey("z"), "z");
  assert.equal(boundsAxisForKey("b"), undefined);
  assert.equal(boundsAxisForKey("Escape"), undefined);
});

test("an axis key locks its axis, and the same key again releases it", () => {
  // The key that enters the state leaves it, matching the tool shortcuts.
  const locked = toggleBoundsAxisConstraint(undefined, "y", "axis");
  assert.deepEqual(locked, ["y"]);
  assert.equal(toggleBoundsAxisConstraint(locked, "y", "axis"), undefined);
  // A different axis switches outright: "X then Y" is a change of mind, not a
  // request for the XY plane — that is what shift is for.
  assert.deepEqual(toggleBoundsAxisConstraint(locked, "x", "axis"), ["x"]);
});

test("shift locks the named axis out and leaves the other two", () => {
  const plane = toggleBoundsAxisConstraint(undefined, "y", "plane");
  assert.deepEqual(plane, ["x", "z"]);
  assert.equal(toggleBoundsAxisConstraint(plane, "y", "plane"), undefined);
  // The two modes are distinct states, so one does not silently release the
  // other: shift+Y over an existing Y lock becomes the plane.
  assert.deepEqual(toggleBoundsAxisConstraint(["y"], "y", "plane"), ["x", "z"]);
  assert.deepEqual(toggleBoundsAxisConstraint(["x", "z"], "y", "axis"), ["y"]);
});

test("a constrained handle is the handle with its locked-out sides dropped", () => {
  const box = { min: { x: -1, y: 0, z: -1 }, max: { x: 1, y: 2, z: 1 } };
  const corner = fluidBodyHandleById(box, "+++")!;
  assert.deepEqual(constrainBoundsHandle(corner, ["y"]).sides, { y: "max" });
  assert.deepEqual(constrainBoundsHandle(corner, ["x", "z"]).sides, { x: "max", z: "max" });
  assert.deepEqual(constrainBoundsHandle(corner, undefined).sides, corner.sides);
  // The position is still the handle's own, so the drag keeps resolving where
  // the user grabbed rather than jumping to a face centre.
  assert.deepEqual(constrainBoundsHandle(corner, ["y"]).position_m, corner.position_m);

  const minCorner = fluidBodyHandleById(box, "--+")!;
  assert.deepEqual(constrainBoundsHandle(minCorner, ["y"]).sides, { y: "min" },
    "a min side stays a min side under the lock");
});

test("a lock naming nothing the handle owns leaves it with no sides at all", () => {
  const box = { min: { x: -1, y: 0, z: -1 }, max: { x: 1, y: 2, z: 1 } };
  const face = fluidBodyHandleById(box, "+00")!;
  assert.deepEqual(boundsDragAxes(face, ["y"]), []);
  assert.deepEqual(constrainBoundsHandle(face, ["y"]).sides, {});
  assert.equal(boundsDragAxisDirection(face, ["y"]), undefined);
});

test("one remaining degree of freedom names the axis line the drag rides", () => {
  const box = { min: { x: -1, y: 0, z: -1 }, max: { x: 1, y: 2, z: 1 } };
  // Unconstrained, this is the old face-normal rule: a face rides its normal,
  // an edge or corner has no single axis and drags in the camera plane.
  assert.deepEqual(boundsDragAxisDirection(fluidBodyHandleById(box, "0+0")!, undefined), { x: 0, y: 1, z: 0 });
  assert.deepEqual(boundsDragAxisDirection(fluidBodyHandleById(box, "00-")!, undefined), { x: 0, y: 0, z: 1 });
  assert.equal(boundsDragAxisDirection(fluidBodyHandleById(box, "++0")!, undefined), undefined);
  assert.equal(boundsDragAxisDirection(fluidBodyHandleById(box, "+++")!, undefined), undefined);
  // A lock that leaves one axis makes a corner behave as that face does.
  assert.deepEqual(boundsDragAxisDirection(fluidBodyHandleById(box, "+++")!, ["y"]), { x: 0, y: 1, z: 0 });
  assert.equal(boundsDragAxisDirection(fluidBodyHandleById(box, "+++")!, ["x", "z"]), undefined);
});

test("a Y-locked corner lowers the water without touching its width or depth", () => {
  // The gesture the lock exists for: a corner is the easiest handle to hit and
  // the only one on some camera angles, and unlocked it grows all three axes.
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene)!;
  const handle = fluidBodyHandleById(box, "+++")!;
  const [, cellY] = sceneCellSizes_m(scene);
  const target = { x: box.max.x + 0.4, y: box.max.y - 3 * cellY, z: box.max.z + 0.4 };

  const free = dragFluidBodyBox(box, handle, target, scene);
  assert.ok(free.max.x > box.max.x && free.max.z > box.max.z,
    "unlocked, the corner grows the footprint too — the problem the lock solves");

  const locked = dragFluidBodyBox(box, constrainBoundsHandle(handle, ["y"]), target, scene);
  assert.deepEqual([locked.min, { x: locked.max.x, z: locked.max.z }],
    [box.min, { x: box.max.x, z: box.max.z }], "only the +y side may move");
  // The moved side still snaps to the lattice, so the drop matches the request
  // to within the half cell the snap is entitled to.
  assert.ok(Math.abs((box.max.y - locked.max.y) - 3 * cellY) <= 0.5 * cellY,
    `${box.max.y - locked.max.y} should be within half a cell of ${3 * cellY}`);
});

test("a plane lock reshapes the footprint and holds the height", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene)!;
  const handle = constrainBoundsHandle(fluidBodyHandleById(box, "+++")!, ["x", "z"]);
  const dragged = dragFluidBodyBox(box, handle, { x: box.max.x + 0.3, y: 0, z: box.max.z + 0.3 }, scene);
  assert.equal(dragged.max.y, box.max.y);
  assert.ok(dragged.max.x > box.max.x && dragged.max.z > box.max.z);
});

test("an inert handle holds the box exactly where it was", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene)!;
  // The +x face under a Y lock: honest nothing, rather than a silent fallback
  // to an axis the user did not ask for.
  const handle = constrainBoundsHandle(fluidBodyHandleById(box, "+00")!, ["y"]);
  assert.deepEqual(dragFluidBodyBox(box, handle, { x: 99, y: 99, z: 99 }, scene), box);
});

test("the tank takes the same lock, so a corner can change height alone", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  const handle = fluidBodyHandleById(box, "+++")!;
  const target = { x: box.max.x + 0.5, y: 0.5 * scene.container.height_m, z: box.max.z + 0.5 };

  const free = dragTankExtents(scene, handle, target);
  assert.ok(free.width_m > scene.container.width_m && free.depth_m > scene.container.depth_m);

  const locked = dragTankExtents(scene, constrainBoundsHandle(handle, ["y"]), target);
  assert.equal(locked.width_m, scene.container.width_m);
  assert.equal(locked.depth_m, scene.container.depth_m);
  assert.ok(locked.height_m < scene.container.height_m);
});

test("the lock reads the way the key that set it was pressed", () => {
  assert.equal(boundsAxisConstraintLabel(["y"]), "ALONG Y");
  // Shift names the axis it locks out, so that is the axis the label states.
  assert.equal(boundsAxisConstraintLabel(["x", "z"]), "LOCKING Y");
  assert.equal(boundsAxisConstraintLabel(undefined), "");
  assert.equal(boundsAxisConstraintLabel(BOUNDS_AXES), "", "all three axes is no constraint at all");
});

test("the lock cannot survive the mode that draws it", () => {
  const initial = useUIStore.getInitialState();
  useUIStore.setState(initial, true);
  assert.equal(initial.boundsAxisConstraint, undefined);

  useUIStore.getState().setActiveTool("bounds");
  useUIStore.getState().setBoundsAxisConstraint(["y"]);
  assert.deepEqual(useUIStore.getState().boundsAxisConstraint, ["y"]);

  useUIStore.getState().setActiveTool("select");
  assert.equal(useUIStore.getState().boundsAxisConstraint, undefined,
    "a lock left armed outside BOUNDS would eat the next drag invisibly");
  useUIStore.setState(initial, true);
});

test("the keyboard claims the axis letters only while BOUNDS is armed", () => {
  const shortcuts = readFileSync(new URL("../lib/use-editor-shortcuts.ts", import.meta.url), "utf8");
  assert.match(shortcuts, /if \(ui\.activeTool === "bounds"\)[\s\S]{0,400}?boundsAxisForKey\(event\.key\)/,
    "outside BOUNDS, y must still arm the erase tool");
  assert.match(shortcuts, /event\.shiftKey \? "plane" : "axis"/,
    "shift is what asks for the plane constraint");
  assert.match(shortcuts, /if \(ui\.boundsAxisConstraint\) \{ ui\.setBoundsAxisConstraint\(undefined\); return; \}/,
    "Escape must drop the lock before it drops the tool");
});
