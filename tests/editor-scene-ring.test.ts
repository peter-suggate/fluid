import assert from "node:assert/strict";
import test from "node:test";
import type { EditorAction } from "../lib/core/editor-action";
import { sceneActionsAt } from "../lib/core/editor-entity-catalog";
import { cloneScene, defaultScene } from "../lib/core/model";
import { voxelTools } from "../lib/core/voxel-editor/registry";

// The scene's ring after the shelf: the document verbs and the sculpt tools
// are contextual — composed onto the ring rather than standing in a corner —
// so this pins what the ring now promises: every capability the persistent
// Scene/Tools shelf used to hold has a wedge, and a tool that does not apply
// is drawn disabled with its own reason rather than omitted.

const POINT = { x: 0, y: 0, z: 0 };

function wedge(actions: readonly EditorAction[], id: string): EditorAction {
  const found = actions.find((action) => action.id === id);
  assert.ok(found, `no ${id} wedge`);
  return found;
}

test("the scene wedge carries the document verbs, and add-water only while dry", () => {
  const wet = sceneActionsAt(cloneScene(defaultScene), POINT);
  const wetChildren = (wedge(wet, "scene").children ?? []).map((child) => child.id);
  assert.deepEqual(wetChildren,
    ["choose-scene", "scene-new", "scene-save", "scene-export", "scene-import"]);

  const dryScene = cloneScene(defaultScene);
  dryScene.systems = { ...dryScene.systems, fluid: false };
  const dry = sceneActionsAt(dryScene, POINT);
  const dryChildren = (wedge(dry, "scene").children ?? []).map((child) => child.id);
  assert.ok(dryChildren.includes("scene-enable-water"), "a dry document offers water");
});

test("every registered sculpt tool has a wedge, disabled with its own reason when it cannot run", () => {
  // The default scene runs water, and no method is supplied — so the solid
  // tools are unavailable and must say why, not disappear.
  const actions = sceneActionsAt(cloneScene(defaultScene), POINT, undefined, { methodId: "uniform" });
  const children = actions.flatMap((action) => action.children ?? []);
  for (const tool of voxelTools.tools) {
    const child = children.find((candidate) => candidate.id === `voxel-tool-${tool.id}`);
    assert.ok(child, `no wedge for ${tool.id}`);
    const reason = tool.unavailable({ scene: cloneScene(defaultScene), methodId: "uniform" });
    assert.equal(child.enabled, !reason, `${tool.id} enabled state disagrees with its own predicate`);
    if (reason) assert.equal(child.hint, reason, `${tool.id} must teach the way in`);
    assert.equal(child.effect?.kind, "voxel-tool");
    assert.ok(child.iconPath, `${tool.id} carries its plugin icon`);
  }
});

test("the LOOK ring keeps the document verbs but withholds everything that edits", () => {
  const actions = sceneActionsAt(cloneScene(defaultScene), POINT, undefined,
    { placement: false, methodId: "adaptive-mass" });
  assert.ok(actions.find((action) => action.id === "scene"), "LOOK still reaches the document");
  const ids = actions.map((action) => action.id);
  assert.ok(!ids.some((id) => id.startsWith("sculpt-")), "LOOK must not offer sculpt tools");
  assert.ok(!ids.includes("water"), "LOOK must not offer placement");
});
