import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  EDITOR_AXES,
  axisConstraintLabel,
  axisDragDirection,
  constrainedAxes,
  editorAxisForKey,
  toggleAxisConstraint,
} from "../lib/editor-axis-constraint";
import { dragFluidBodyBox, fluidBodyBox } from "../lib/editor-fluid-body";
import { dragTankExtents, tankBox } from "../lib/editor-tank";
import { getScenePreset } from "../lib/scenes";
import { sceneCellSizes_m } from "../lib/scene-lattice";
import { useUIStore } from "../lib/stores/ui-store";
import { sides, handleById } from "./helpers/editor-entities";
import type { SceneDescription } from "../lib/model";

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

test("axis keys name axes, and nothing else does", () => {
  assert.equal(editorAxisForKey("x"), "x");
  assert.equal(editorAxisForKey("Y"), "y");
  assert.equal(editorAxisForKey("z"), "z");
  assert.equal(editorAxisForKey("b"), undefined);
  assert.equal(editorAxisForKey("Escape"), undefined);
});

test("an axis key locks its axis, and the same key again releases it", () => {
  // The key that enters the state leaves it, matching the tool shortcuts.
  const locked = toggleAxisConstraint(undefined, "y", "axis");
  assert.deepEqual(locked, ["y"]);
  assert.equal(toggleAxisConstraint(locked, "y", "axis"), undefined);
  // A different axis switches outright: "X then Y" is a change of mind, not a
  // request for the XY plane — that is what shift is for.
  assert.deepEqual(toggleAxisConstraint(locked, "x", "axis"), ["x"]);
});

test("shift locks the named axis out and leaves the other two", () => {
  const plane = toggleAxisConstraint(undefined, "y", "plane");
  assert.deepEqual(plane, ["x", "z"]);
  assert.equal(toggleAxisConstraint(plane, "y", "plane"), undefined);
  // The two modes are distinct states, so one does not silently release the
  // other: shift+Y over an existing Y lock becomes the plane.
  assert.deepEqual(toggleAxisConstraint(["y"], "y", "plane"), ["x", "z"]);
  assert.deepEqual(toggleAxisConstraint(["x", "z"], "y", "axis"), ["y"]);
});

test("a constrained handle is the handle with its locked-out axes dropped", () => {
  assert.deepEqual(constrainedAxes(["x", "y", "z"], ["y"]), ["y"]);
  assert.deepEqual(constrainedAxes(["x", "y", "z"], ["x", "z"]), ["x", "z"]);
  assert.deepEqual(constrainedAxes(["x", "y", "z"], undefined), ["x", "y", "z"]);
  // A lock naming nothing the handle owns leaves it with no axes at all, which
  // is the honest outcome rather than a silent fallback to an axis the user did
  // not ask for.
  assert.deepEqual(constrainedAxes(["x"], ["y"]), []);
  assert.equal(axisDragDirection(["x"], ["y"]), undefined);
});

test("one remaining degree of freedom names the axis line the drag rides", () => {
  // Unconstrained, this is the face-normal rule: a face rides its normal, an
  // edge or corner has no single axis and drags in the camera plane.
  assert.deepEqual(axisDragDirection(["y"], undefined), { x: 0, y: 1, z: 0 });
  assert.deepEqual(axisDragDirection(["z"], undefined), { x: 0, y: 0, z: 1 });
  assert.equal(axisDragDirection(["x", "y"], undefined), undefined);
  assert.equal(axisDragDirection(["x", "y", "z"], undefined), undefined);
  // A lock that leaves one axis makes a corner behave as that face does.
  assert.deepEqual(axisDragDirection(["x", "y", "z"], ["y"]), { x: 0, y: 1, z: 0 });
  assert.equal(axisDragDirection(["x", "y", "z"], ["x", "z"]), undefined);
});

test("a Y-locked corner lowers the water without touching its width or depth", () => {
  // The gesture the lock exists for: a corner is the easiest handle to hit and
  // the only one on some camera angles, and unlocked it grows all three axes.
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene)!;
  const [, cellY] = sceneCellSizes_m(scene);
  const target = { x: box.max.x + 0.4, y: box.max.y - 3 * cellY, z: box.max.z + 0.4 };

  const free = dragFluidBodyBox(box, sides("+++"), target, scene);
  assert.ok(free.max.x > box.max.x && free.max.z > box.max.z,
    "unlocked, the corner grows the footprint too — the problem the lock solves");

  const locked = dragFluidBodyBox(box, sides("0+0"), target, scene);
  assert.deepEqual([locked.min, { x: locked.max.x, z: locked.max.z }],
    [box.min, { x: box.max.x, z: box.max.z }], "only the +y side may move");
  // The moved side still snaps to the lattice, so the drop matches the request
  // to within the half cell the snap is entitled to.
  assert.ok(Math.abs((box.max.y - locked.max.y) - 3 * cellY) <= 0.5 * cellY,
    `${box.max.y - locked.max.y} should be within half a cell of ${3 * cellY}`);
});

test("the lock reaches the drag through the handle, not around it", () => {
  // End to end: the constraint is applied inside `handle.drag`, so nothing the
  // viewport does can disagree with what the handle promises.
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene)!;
  const [, cellY] = sceneCellSizes_m(scene);
  const corner = handleById(scene, { kind: "fluid-body", id: "fluid-body" }, "+++");
  const target = { x: box.max.x + 0.4, y: box.max.y - 3 * cellY, z: box.max.z + 0.4 };

  const locked = corner.drag(target, ["y"]);
  assert.ok(locked);
  const reshaped = fluidBodyBox({ ...scene, ...locked })!;
  assert.ok(Math.abs(reshaped.max.x - box.max.x) < 1e-9, "x must not move under a Y lock");
  assert.ok(reshaped.max.y < box.max.y);

  // A lock naming nothing this handle owns holds the scene still rather than
  // proposing a patch that changes nothing.
  const face = handleById(scene, { kind: "fluid-body", id: "fluid-body" }, "+00");
  assert.equal(face.drag(target, ["y"]), undefined);
});

test("a plane lock reshapes the footprint and holds the height", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene)!;
  const dragged = dragFluidBodyBox(box, sides("+0+"),
    { x: box.max.x + 0.3, y: 0, z: box.max.z + 0.3 }, scene);
  assert.equal(dragged.max.y, box.max.y);
  assert.ok(dragged.max.x > box.max.x && dragged.max.z > box.max.z);
});

test("an inert handle holds the box exactly where it was", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene)!;
  // The +x face under a Y lock: honest nothing, rather than a silent fallback
  // to an axis the user did not ask for.
  assert.deepEqual(dragFluidBodyBox(box, {}, { x: 99, y: 99, z: 99 }, scene), box);
});

test("the tank takes the same lock, so a corner can change height alone", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  const target = { x: box.max.x + 0.5, y: 0.5 * scene.container.height_m, z: box.max.z + 0.5 };

  const free = dragTankExtents(scene, sides("+++"), target);
  assert.ok(free.width_m > scene.container.width_m && free.depth_m > scene.container.depth_m);

  const locked = dragTankExtents(scene, sides("0+0"), target);
  assert.equal(locked.width_m, scene.container.width_m);
  assert.equal(locked.depth_m, scene.container.depth_m);
  assert.ok(locked.height_m < scene.container.height_m);
});

test("the lock reads the way the key that set it was pressed", () => {
  assert.equal(axisConstraintLabel(["y"]), "ALONG Y");
  // Shift names the axis it locks out, so that is the axis the label states.
  assert.equal(axisConstraintLabel(["x", "z"]), "LOCKING Y");
  assert.equal(axisConstraintLabel(undefined), "");
  assert.equal(axisConstraintLabel(EDITOR_AXES), "", "all three axes is no constraint at all");
});

test("the lock cannot survive the mode that draws it", () => {
  const initial = useUIStore.getInitialState();
  useUIStore.setState(initial, true);
  assert.equal(initial.axisConstraint, undefined);

  useUIStore.getState().setAxisConstraint(["y"]);
  assert.deepEqual(useUIStore.getState().axisConstraint, ["y"]);

  useUIStore.getState().setActiveTool("fluid-paint");
  assert.equal(useUIStore.getState().axisConstraint, undefined,
    "a lock left armed into the next tool would eat its first drag invisibly");
  useUIStore.setState(initial, true);
});

test("the keyboard claims the axis letters only while something is selected", () => {
  const shortcuts = readFileSync(new URL("../lib/use-editor-shortcuts.ts", import.meta.url), "utf8");
  assert.match(shortcuts, /if \(ui\.selection\) \{[\s\S]{0,400}?editorAxisForKey\(event\.key\)/,
    "with nothing selected, y must still arm the erase tool");
  assert.match(shortcuts, /event\.shiftKey \? "plane" : "axis"/,
    "shift is what asks for the plane constraint");
  assert.match(shortcuts, /if \(ui\.axisConstraint\) \{ ui\.setAxisConstraint\(undefined\); return; \}/,
    "Escape must drop the lock before it drops the tool");
});
