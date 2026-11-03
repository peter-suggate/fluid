import assert from "node:assert/strict";
import test from "node:test";
import type { AxisConstraint } from "../lib/core/editor-axis-constraint";
import type { EditorRay } from "../lib/core/editor-entity";
import { FocusSession, bodyPlacementCentre, placementCentre } from "../lib/core/editor-focus";
import { createEmptyScene, minimalRoomSceneryGraph } from "../lib/core/empty-scene";
import { voxelToolKeyControls } from "../lib/core/editor-voxel-tool-actions";
import { cloneScene, defaultScene, type SceneDescription, type Vec3 } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { toolValues, type ToolContext } from "../lib/core/voxel-editor/plugin";
import { penTool } from "../lib/core/voxel-editor/plugins/pen";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { visibleToolSurface } from "../lib/core/voxel-editor/surface";
import { fluidShapeBounds } from "../lib/core/voxel-editor/fluid-geometry";

/** A slab of solid, cells 2..29 in x and z, 1..3 in y: its top face is at cell y = 4. */
function slab(): SceneDescription {
  const result = createEmptyScene();
  result.scenery = { ...minimalRoomSceneryGraph, nodes: [{ ...minimalRoomSceneryGraph.nodes.find(node => node.kind === "room-shell")!, faces: [] }] };
  result.solidVoxels = [{ operation: "fill", minimum: [2, 1, 2], maximumExclusive: [30, 4, 30], materialId: 2 }];
  return result;
}
/** Straight down through the centre of column (x, z). */
function down(scene: SceneDescription, x: number, z = 8): EditorRay {
  const h = sceneCellSizes_m(scene);
  return { origin: { x: -scene.container.width_m / 2 + (x + .5) * h[0], y: 2,
    z: -scene.container.depth_m / 2 + (z + .5) * h[2] }, direction: { x: 0, y: -1, z: 0 } };
}
const pen = (size = 1) => toolValues(penTool, { size });
const focus = (constraint: AxisConstraint, fallback?: ToolContext["ray"]["origin"]) =>
  new FocusSession({ constraint: () => constraint, ...(fallback ? { fallback } : {}) });

test("no tool takes X, Y or Z: they are the focus locks under every tool", () => {
  for (const plugin of voxelTools.tools) {
    const values = toolValues(plugin);
    for (const key of ["x", "y", "z"]) {
      const claimed = voxelToolKeyControls(plugin.id).some((control) => control.kind === "choice"
        ? control.options.some((option) => option.shortcut === key)
        : control.shortcut === key);
      assert.equal(claimed, false, `${plugin.id} claims ${key} (${JSON.stringify(values)})`);
    }
  }
});

test("unlocked, the Pen's focus is exactly the nearest visible surface: same patches with or without a host focus", () => {
  for (const size of [1, 5]) {
    const base = slab();
    const path = [4, 5, 9, 12, 12, 40, 14].map((x, i) => down(base, x, 8 + (i % 3)));
    const bare = penTool.begin({ scene: base, ray: path[0]!, values: pen(size) })!;
    const hosted = penTool.begin({ scene: base, ray: path[0]!, values: pen(size), focus: focus(undefined) })!;
    for (const ray of path) assert.deepEqual(hosted.update(ray), bare.update(ray), `size ${size}`);
    assert.equal(penTool.begin({ scene: base, ray: down(base, 40), values: pen(size), focus: focus(undefined) }), undefined,
      "a miss with no lock still goes to the camera: no ground plane");
  }
  const base = slab();
  assert.ok(visibleToolSurface(base, down(base, 4)), "the slab is pickable through the shared path");
});

test("an X lock makes the Pen's stroke straight whatever the pointer's z", () => {
  const base = slab();
  const gesture = penTool.begin({ scene: base, ray: down(base, 4, 8), values: pen(1), focus: focus(["x"]) })!;
  gesture.update(down(base, 4, 8));
  const stroke = gesture.update(down(base, 12, 13))!;
  assert.deepEqual(stroke.patches.map((patch) => patch.minimum[0]), [4, 5, 6, 7, 8, 9, 10, 11, 12]);
  assert.ok(stroke.patches.every((patch) => patch.minimum[1] === 4 && patch.minimum[2] === 8), "y and z held at the press");
  const free = penTool.begin({ scene: base, ray: down(base, 4, 8), values: pen(1) })!;
  free.update(down(base, 4, 8));
  assert.ok(free.update(down(base, 12, 13))!.patches.some((patch) => patch.minimum[2] !== 8), "unlocked it would wander");
});

test("a Shift+Y lock holds the Pen at the press height, and draws on into open air past the edge", () => {
  const base = slab();
  const gesture = penTool.begin({ scene: base, ray: down(base, 26), values: pen(1), focus: focus(["x", "z"]) })!;
  gesture.update(down(base, 26));
  const out = gesture.update(down(base, 36))!;
  assert.deepEqual(out.patches.map((patch) => patch.minimum[0]), [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]);
  assert.ok(out.patches.every((patch) => patch.minimum[1] === 4 && patch.operation === "fill"), "a level course, in air beyond x = 29");
  // A later press over open air stands on the last stroke's anchor.
  const anchor = { x: down(base, 26).origin.x, y: 4 * sceneCellSizes_m(base)[1], z: down(base, 26).origin.z };
  const air = penTool.begin({ scene: base, ray: down(base, 40), values: pen(1), focus: focus(["x", "z"], anchor) })!;
  assert.equal(air.update(down(base, 40))!.patches[0]!.minimum[1], 4);
});

const tank = () => {
  const result = cloneScene(defaultScene);
  result.terrain = undefined;
  result.container = { ...result.container, width_m: 4, height_m: 4, depth_m: 4 };
  return result;
};
const fromAbove = (x = 0, z = 0): EditorRay => ({ origin: { x, y: 10, z }, direction: { x: 0, y: -1, z: 0 } });
const towards = (target: Vec3, eye: Vec3 = { x: 3, y: 3, z: 3 }): EditorRay =>
  ({ origin: eye, direction: { x: target.x - eye.x, y: target.y - eye.y, z: target.z - eye.z } });

test("unlocked, a water drop stays on its Height plane exactly as before", () => {
  for (const plugin of voxelTools.tools.filter((candidate) => candidate.execution === "release")) {
    const base = tank(), values = toolValues(plugin, { height: 3 });
    const bare = plugin.begin({ scene: base, ray: fromAbove(), values })!;
    const hosted = plugin.begin({ scene: base, ray: fromAbove(), values, focus: focus(undefined) })!;
    for (const ray of [fromAbove(), fromAbove(.25, .3), towards({ x: -.5, y: 0, z: .4 })]) {
      assert.deepEqual(hosted.update(ray), bare.update(ray), plugin.id);
    }
  }
});

test("a Y lock lifts a water drop straight up from where the press met its plane", () => {
  for (const plugin of voxelTools.tools.filter((candidate) => candidate.execution === "release")) {
    const base = tank(), values = toolValues(plugin, { height: 3 });
    const voxel = Math.min(...sceneCellSizes_m(base));
    // A press from straight above could not address a vertical lock at all: the eye looks along it.
    const press = towards({ x: .5, y: 0, z: -.25 });
    const gesture = plugin.begin({ scene: base, ray: press, values, focus: focus(["y"]) })!;
    const pressed = gesture.update(press)!.action!.edit;
    const lifted = gesture.update(towards({ ...pressed.center_m, y: 2.5 }))!.action!.edit;
    assert.equal(lifted.center_m.x, pressed.center_m.x, plugin.id);
    assert.equal(lifted.center_m.z, pressed.center_m.z, plugin.id);
    assert.ok(lifted.center_m.y > pressed.center_m.y + voxel, `${plugin.id} rises`);
    const bottom = fluidShapeBounds(lifted).min.y / voxel;
    assert.ok(Math.abs(bottom - Math.round(bottom)) < 1e-9, `${plugin.id} keeps its bottom on the grid`);
    assert.throws(() => gesture.update(towards({ ...pressed.center_m, y: 40 }, { x: 3, y: 30, z: 3 })), /inside the tank/);
  }
});

test("BALL and DRAG place on the resting point unlocked, and on the lock through it when one is up", () => {
  const base = tank();
  const rest = (ray: EditorRay): Vec3 | undefined => {
    if (!(ray.direction.y < 0)) return undefined;
    const t = (0.25 - ray.origin.y) / ray.direction.y;
    return { x: ray.origin.x + t * ray.direction.x, y: 0.25, z: ray.origin.z + t * ray.direction.z };
  };
  const ray = towards({ x: .5, y: .25, z: -.5 });
  assert.deepEqual(placementCentre(undefined, ray, rest), { centre: rest(ray)!, locked: false });
  assert.deepEqual(placementCentre(focus(undefined), ray, rest), { centre: rest(ray)!, locked: false });
  const lift = focus(["y"]);
  assert.deepEqual(placementCentre(lift, ray, rest)!.centre, rest(ray), "the press anchors at the rest point");
  const up = placementCentre(lift, towards({ x: .5, y: 1.5, z: -.5 }), rest)!;
  assert.equal(up.locked, true);
  assert.ok(Math.abs(up.centre.y - 1.5) < 1e-9 && Math.abs(up.centre.x - .5) < 1e-12 && Math.abs(up.centre.z + .5) < 1e-12);
  // A body stays inside the container, as a resting one does.
  const body = focus(["x"]);
  bodyPlacementCentre(base, body, ray, .2, rest);
  const pushed = bodyPlacementCentre(base, body, towards({ x: 9, y: .25, z: -.5 }, { x: 0, y: 3, z: 6 }), .2, rest)!;
  assert.ok(Math.abs(pushed.x - (base.container.width_m / 2 - .2)) < 1e-12, "clamped to the wall less its radius");
  assert.equal(pushed.y, .25);
});
