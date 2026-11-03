import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { voxelRailToolForKey, voxelToolRail } from "../lib/core/editor-voxel-tool-actions";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { PASTE_SCENERY, pasteTool } from "../lib/core/voxel-editor/plugins/paste";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { PASTE_TOOL_ID, pasteVerb } from "../lib/core/voxel-editor/verbs/paste";

const clipboard = { minimum: [0, 0, 0] as const, block: { size: [2, 1, 1] as const,
  solidFraction: new Uint8Array([255, 255]), signedDistanceQ8: new Int16Array([-128, -128]), materialId: new Uint16Array([2, 3]) } };

/** A tank wall of scenery (face at z = 1.3 m) beside a voxel slab (top at y = 4 cells). */
function fixture() {
  const scene = createEmptyScene({ extents_m: { x: 12.8, y: .8, z: 2.4 }, finestCellSize_m: .025 });
  scene.scenery = { ...scene.scenery!, nodes: [
    { ...scene.scenery!.nodes.find(node => node.kind === "room-shell")!, faces: [] },
    { id: "wall", kind: "box" as const, place: { units: "metres" as const, position: { x: 0, y: .4, z: 1.25 } },
      halfSize: { x: 6.5, y: .4, z: .05 },
      material: { colorLinear: [.8, .75, .6] as [number, number, number], surface: "ceramic" as const } },
  ] };
  scene.solidVoxels = [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [64, 4, 64], materialId: 2 }];
  const h = sceneCellSizes_m(scene);
  const atWall = (x = 0) => ({ origin: { x, y: .4, z: 2 }, direction: { x: 0, y: 0, z: -1 } });
  const onSlab = (x: number) => ({ origin: { x: -scene.container.width_m / 2 + (x + .5) * h[0], y: .5,
    z: -scene.container.depth_m / 2 + 8.5 * h[2] }, direction: { x: 0, y: -1, z: 0 } });
  return { scene, h, atWall, onSlab };
}

test("Paste is the verb's tool: armed by Paste, not a rail mark, and Pattern, Face and Line are gone", () => {
  assert.equal(PASTE_TOOL_ID, "paste");
  assert.equal(voxelTools.get(PASTE_TOOL_ID), pasteTool);
  assert.equal(pasteVerb.run({} as never).armTool, "paste");
  for (const retired of ["pattern", "face", "line", "center", "box", "voxel"]) assert.equal(voxelTools.get(retired), undefined, retired);
  const { scene } = fixture();
  assert.ok(!voxelToolRail(scene, undefined, clipboard).some((group) => group.tools.some((tool) => tool.id === "paste")));
  for (const key of "123456789") assert.notEqual(voxelRailToolForKey(key), "paste");
  assert.ok(pasteTool.ui.controls.some((control) => control.id === "turns"), "the quarter-turn control stays");
});

test("a pasted block stamps where you click, turned by quarter turns", () => {
  const { scene, onSlab } = fixture();
  const straight = pasteTool.begin({ scene, ray: onSlab(10), values: toolValues(pasteTool, { operation: 0 }), clipboard })!
    .update(onSlab(10))!;
  assert.ok(straight.patches.length > 0 && straight.patches.every((patch) => patch.operation === "fill" && patch.minimum[1] === 4));
  const size = (patches: typeof straight.patches) => [0, 1, 2].map((a) =>
    Math.max(...patches.map((p) => p.maximumExclusive[a]!)) - Math.min(...patches.map((p) => p.minimum[a]!)));
  assert.deepEqual(size(straight.patches), [2, 1, 1]);
  const turned = pasteTool.begin({ scene, ray: onSlab(10), values: toolValues(pasteTool, { operation: 0, turns: 1 }), clipboard })!
    .update(onSlab(10))!;
  assert.deepEqual(size(turned.patches), [1, 1, 2]);
});

test("attaching a paste builds on scenery; erasing or repainting with one says it cannot reach scenery instead of doing nothing", () => {
  const { scene, h, atWall } = fixture();
  const built = pasteTool.begin({ scene, ray: atWall(), values: toolValues(pasteTool, { operation: 0 }), clipboard })!.update(atWall())!;
  assert.ok(built.patches.length > 0);
  assert.ok(built.patches.every((patch) => patch.operation === "fill" && patch.minimum[2] * h[2] - 1.2 >= 1.3 - 1e-9),
    "the block stands on the wall's face, in front of it");
  for (const operation of [1, 2]) {
    assert.throws(() => pasteTool.begin({ scene, ray: atWall(), values: toolValues(pasteTool, { operation }), clipboard }),
      (error: Error) => error.message === PASTE_SCENERY);
  }
  // Pressed on voxels, then dragged onto scenery: the proposal is withdrawn, with the reason on the caption.
  const { onSlab } = fixture();
  const erase = pasteTool.begin({ scene, ray: onSlab(10), values: toolValues(pasteTool, { operation: 1 }), clipboard })!;
  assert.ok(erase.update(onSlab(10))!.patches.length > 0);
  const over = erase.update(atWall())!;
  assert.equal(over.patches.length, 0);
  assert.match(over.caption, /scenery is cut with Pen or Shape/);
});
