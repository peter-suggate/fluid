import assert from "node:assert/strict";
import test from "node:test";
import type { EditorEntityContext } from "../lib/core/editor-entity";
import {
  ARMABLE_GESTURES,
  EDITOR_GESTURES,
  editorGestureForShortcut,
  gestureForPress,
  getEditorGesture,
  type EditorGestureId,
} from "../lib/core/editor-gesture-catalog";
import { targetAtRay } from "../lib/core/editor-probe-catalog";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";

const NONE = { shift: false, middleButton: false };

function context(scene: SceneDescription): EditorEntityContext {
  return { scene, bodies: [], pickingAvailable: true };
}

/** A ray from outside the container aimed away from it: the room fallback. */
function skyTarget(scene: SceneDescription) {
  const c = scene.container;
  return targetAtRay(context(scene), {
    origin: { x: 0, y: 4 * c.height_m, z: 4 * c.depth_m },
    direction: { x: 0, y: 0.7071, z: 0.7071 },
  });
}

function wallTarget(scene: SceneDescription) {
  const c = scene.container;
  return targetAtRay(context(scene), {
    origin: { x: 0, y: 0.5 * c.height_m, z: -2 * c.depth_m },
    direction: { x: 0, y: 0, z: 1 },
  });
}

function voxelTarget() {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [{
    operation: "fill", minimum: [3, 4, 2], maximumExclusive: [4, 5, 3], materialId: 2,
  }];
  const [hx, hy, hz] = sceneCellSizes_m(scene);
  return targetAtRay(context(scene), {
    origin: { x: -0.5 * scene.container.width_m + 2 * hx, y: 4.5 * hy,
      z: -0.5 * scene.container.depth_m + 2.5 * hz },
    direction: { x: 1, y: 0, z: 0 },
  });
}

test("orbit is last and claims everything, so resolution never falls through", () => {
  const last = EDITOR_GESTURES.at(-1);
  assert.equal(last?.id, "orbit");
  assert.equal(last.armable, undefined, "the fallback must not be armable");
  assert.equal(last.needsPresentation, undefined,
    "a presentation-gated fallback would leave a press with no meaning at all");
});

test("gesture ids are unique and every armable one has a distinct key", () => {
  const ids = EDITOR_GESTURES.map((gesture) => gesture.id);
  assert.equal(new Set(ids).size, ids.length);
  const keys = ARMABLE_GESTURES.map((gesture) => gesture.shortcut);
  assert.ok(keys.every((key) => typeof key === "string"));
  assert.equal(new Set(keys).size, keys.length);
  // The reverse direction, because a key that armed nothing would look broken
  // rather than unbound.
  for (const gesture of ARMABLE_GESTURES) {
    assert.equal(editorGestureForShortcut(gesture.shortcut!), gesture.id);
  }
});

// The one thing that makes an implicit gesture implicit: it can never be armed,
// so it can never be reached by a key or left stuck on the mode chip.
test("only armable gestures carry a key", () => {
  for (const gesture of EDITOR_GESTURES) {
    if (gesture.armable) continue;
    assert.equal(gesture.shortcut, undefined, `${gesture.id} must not be armable by key`);
  }
});

test("everything solid sweeps and the room orbits", () => {
  const scene = cloneScene(defaultScene);
  assert.equal(gestureForPress(undefined, voxelTarget(), NONE, true), "voxel-sweep");
  assert.equal(gestureForPress(undefined, wallTarget(scene), NONE, true), "voxel-sweep");
  assert.equal(gestureForPress(undefined, skyTarget(scene), NONE, true), "orbit");
});

test("shift and the middle button always mean the camera", () => {
  const target = voxelTarget();
  assert.equal(gestureForPress(undefined, target, { shift: true, middleButton: false }, true), "pan");
  assert.equal(gestureForPress(undefined, target, { shift: false, middleButton: true }, true), "pan");
  // Even over an armed brush, so a mode always has a way to move the camera
  // without being left first.
  assert.equal(gestureForPress("fluid-paint", target, { shift: true, middleButton: false }, true), "pan");
});

test("an armed gesture outranks the implicit one under the cursor", () => {
  const target = voxelTarget();
  assert.equal(gestureForPress("fluid-paint", target, NONE, true), "fluid-paint");
  assert.equal(gestureForPress("region-draw", target, NONE, true), "region-draw");
});

// A reader who armed WATER and pressed while the renderer was rebuilding asked
// for water; answering with a voxel selection instead would be the interface
// doing something else in their name.
test("an armed gesture blocked by presentation yields to the camera, not to a sweep", () => {
  const target = voxelTarget();
  assert.equal(gestureForPress("fluid-paint", target, NONE, false), "orbit");
  assert.equal(gestureForPress("fluid-ball", target, NONE, false), "orbit");
  // REGION is not presentation-gated: a region is a box in the document, not
  // something dropped onto a published surface.
  assert.equal(gestureForPress("region-draw", target, NONE, false), "region-draw");
});

test("an unpresented body cannot be grabbed, and the press orbits instead", () => {
  const target = { ...skyTarget(cloneScene(defaultScene)),
    selection: { kind: "body", id: "cup" } as const };
  assert.equal(gestureForPress(undefined, target, NONE, true), "body-throw");
  assert.equal(gestureForPress(undefined, target, NONE, false), "orbit");
});

test("every gesture id resolves to its own definition", () => {
  for (const gesture of EDITOR_GESTURES) {
    assert.equal(getEditorGesture(gesture.id).id, gesture.id);
  }
  assert.throws(() => getEditorGesture("not-a-gesture" as EditorGestureId));
});
