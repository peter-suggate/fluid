import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createEmptyScene, minimalRoomSceneryGraph } from "../lib/core/empty-scene";
import { voxelRailToolForKey } from "../lib/core/editor-voxel-tool-actions";
import type { SceneDescription } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { selectDepthControl, selectTool } from "../lib/core/voxel-editor/plugins/select";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { beginToolTransaction, type ToolHost } from "../lib/core/voxel-editor/transaction";
import type { VoxelRegion } from "../lib/core/voxel-editor/voxel-access";
import { regionCellCount } from "../lib/core/voxel-editor/voxel-grid";

/** Two separate pieces standing on nothing: A over cells x 2..5, B over x 10..13; both z 2..5, y 1..3. */
function pieces() {
  const scene = createEmptyScene();
  scene.scenery = { ...minimalRoomSceneryGraph, nodes: [{ ...minimalRoomSceneryGraph.nodes.find(node => node.kind === "room-shell")!, faces: [] }] };
  scene.solidVoxels = [
    { operation: "fill", minimum: [2, 1, 2], maximumExclusive: [6, 4, 6], materialId: 2 },
    { operation: "fill", minimum: [10, 1, 2], maximumExclusive: [14, 4, 6], materialId: 3 },
  ];
  const h = sceneCellSizes_m(scene);
  const down = (x: number, z = 3) => ({ origin: { x: -scene.container.width_m / 2 + (x + .5) * h[0], y: 2,
    z: -scene.container.depth_m / 2 + (z + .5) * h[2] }, direction: { x: 0, y: -1, z: 0 } });
  return { scene, down };
}

test("Select merges Wand and Marquee: one rail mark on W", () => {
  assert.equal(voxelTools.get("select-voxels"), selectTool);
  assert.equal(voxelTools.get("wand"), undefined);
  assert.equal(voxelTools.get("marquee"), undefined);
  assert.equal(selectTool.ui.key, "w");
  assert.equal(voxelRailToolForKey("w"), "select-voxels");
  assert.deepEqual(selectTool.ui.controls.map((control) => control.id), ["selectionMode", "depth", "byMaterial", "shell"]);
});

test("a click selects the connected piece, and only that piece", () => {
  const { scene, down } = pieces();
  const gesture = selectTool.begin({ scene, ray: down(3), values: toolValues(selectTool) })!;
  const click = gesture.update(down(3))!;
  assert.equal(regionCellCount(click.selection!), 4 * 3 * 4);
  assert.deepEqual(click.selection!.minimum, [2, 1, 2]);
  assert.equal(click.patches.length, 0, "selecting authors nothing");
  assert.equal(gesture.update(down(3)), click, "the search runs once");
});

test("a drag selects a box in one gesture, its depth from the control, with no second click", () => {
  const { scene, down } = pieces();
  for (const depth of [1, 3]) {
    const gesture = selectTool.begin({ scene, ray: down(3), values: toolValues(selectTool, { depth }) })!;
    gesture.update(down(3));
    const box = gesture.update(down(12, 5))!.selection!;
    assert.deepEqual(box.minimum, [3, 4 - depth, 3]);
    assert.deepEqual(box.maximumExclusive, [13, 4, 6]);
    assert.equal(box.mask, undefined);
    assert.equal(gesture.advance, undefined, "there is no release-then-click phase");
    // Back to the pressed cell is a click again.
    assert.equal(regionCellCount(gesture.update(down(3))!.selection!), 48);
  }
  assert.equal(selectDepthControl.adjust, "primary", "brackets set the depth");
});

test("add and subtract are the shared selection mode, for clicks and boxes alike", () => {
  const { scene, down } = pieces();
  const values = toolValues(selectTool, { selectionMode: 1 });
  const gesture = selectTool.begin({ scene, ray: down(3), values })!;
  assert.equal(gesture.update(down(3))!.selectionMode, "add");
  assert.equal(gesture.update(down(12))!.selectionMode, "add");
  assert.equal(selectTool.begin({ scene, ray: down(3), values: toolValues(selectTool, { selectionMode: 2 }) })!
    .update(down(12))!.selectionMode, "subtract");
});

test("the release stands the selection through the transaction host, and a hover only marks the cell", async () => {
  const { scene, down } = pieces();
  const selected: { region: VoxelRegion; mode?: string }[] = [];
  const host: ToolHost = {
    scene: () => scene, publish: () => Promise.reject(new Error("selecting publishes nothing")),
    select: (region, mode) => { selected.push({ region, mode }); },
    begin: () => {}, finish: () => {}, cancel: () => {},
  };
  const transaction = beginToolTransaction(selectTool, host, down(3), toolValues(selectTool))!;
  await transaction.update(down(12, 5));
  await transaction.finish();
  assert.equal(selected.length, 1);
  assert.deepEqual(selected[0]!.region.minimum, [3, 3, 3]);
  const hover = selectTool.begin({ scene: scene as SceneDescription, ray: down(3), values: toolValues(selectTool), hover: true })!
    .update(down(3))!;
  assert.equal(hover.selection, undefined);
  assert.equal(hover.highlight.kind, "box");
});
