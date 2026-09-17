import assert from "node:assert/strict";
import test from "node:test";
import type { EditorAction } from "../lib/core/editor-action";
import { entityActionsAt, sceneActionsAt } from "../lib/core/editor-entity-catalog";
import { labRegionWedges } from "../advance-lab/lab-ring";
import { refinementRegionSelectionId } from "../lib/features/refinement-region/definition";
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneDocumentActions, sceneDocumentVerbs } from "../lib/core/editor-scene-document";
import { gravityFeature } from "../lib/features/gravity/definition";
import { surfaceDisplayFeature } from "../lib/features/surface-display/definition";
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
    { placement: false, methodId: "adaptive-volume" });
  assert.ok(actions.find((action) => action.id === "scene"), "LOOK still reaches the document");
  const ids = actions.map((action) => action.id);
  assert.ok(!ids.some((id) => id.startsWith("sculpt-")), "LOOK must not offer sculpt tools");
  assert.ok(!ids.includes("water"), "LOOK must not offer placement");
});

// The standing priority direction (2026-09-08): solver, surface method and
// gravity are high; the document's file operations are low. The rank lives in
// the colocated declarations — these assertions read them, never a component.
test("declared priorities match the standing direction", () => {
  const gravity = gravityFeature.placements.find((placement) => placement.slot === "scene.physics");
  assert.equal(gravity?.priority, "high", "gravity's strip placement is high priority");
  const surface = surfaceDisplayFeature.placements.find((placement) => placement.slot === "scene.surface");
  assert.equal(surface?.priority, "high", "the surface mode's strip placement is high priority");

  const dryScene = cloneScene(defaultScene);
  dryScene.systems = { ...dryScene.systems, fluid: false };
  for (const verb of sceneDocumentVerbs(dryScene)) {
    const expected = verb.id === "scene-enable-water" ? "high" : "low";
    assert.equal(verb.priority, expected, `${verb.id} priority`);
  }
});

/**
 * The `host` arm belongs to the page that composed it, and the studio composes none.
 *
 * `EditorActionEffect` gained a `{ kind: "host" }` arm so a second page — the
 * 2-D advance lab — could put its own verbs on the shared ring without naming
 * them in `lib/core`. The bargain that makes that safe is one-directional:
 * `performEditorAction` answers `host` with a warning and no work, so a studio
 * wedge that ever carried one would be a wedge that silently does nothing. This
 * is the assertion that keeps that from being a discovery.
 */
test("no studio ring wedge carries a host effect", () => {
  const scenes = [cloneScene(defaultScene), (() => {
    const dry = cloneScene(defaultScene);
    dry.systems = { ...dry.systems, fluid: false };
    return dry;
  })()];
  const rings: readonly (readonly EditorAction[])[] = scenes.flatMap((scene) => [
    sceneActionsAt(scene, POINT),
    sceneActionsAt(scene, POINT, undefined, { methodId: "uniform" }),
    sceneActionsAt(scene, POINT, undefined, { placement: false, methodId: "adaptive-volume" }),
    sceneDocumentActions(scene),
  ]);
  const walk = (actions: readonly EditorAction[]): void => {
    for (const action of actions) {
      assert.notEqual(action.effect?.kind, "host",
        `${action.id} carries a host effect, which performEditorAction cannot run`);
      walk(action.children ?? []);
    }
  };
  for (const ring of rings) walk(ring);
});

/**
 * One capability, two hosts, one ring.
 *
 * The plugin exercise's claim on the ring, made checkable on the one capability
 * both pages have: pointing at a refinement region offers *the same wedges* in
 * the studio and in the 2-D advance lab — the same ids, in the same order, with
 * the same labels, icons and tones — because both lists are composed by
 * `lib/core/editor-entity-wedges.ts` rather than written out twice. Before WP6
 * the lab's pair were `slice-region-select` ("Select") and `slice-region-remove`
 * ("Remove"), which is two pages disagreeing about what the word for "delete
 * this" is on one kind of object.
 *
 * The *effects* are deliberately excluded from the comparison and are the one
 * thing that must differ: the studio writes a new `SceneDescription`, the lab
 * sends a whole-list command to a running Rust world, and neither could express
 * the other's. That is the seam working, not a gap in the pin — and the test
 * above this one is what keeps the studio from ever emitting the lab's arm.
 */
test("a selected region offers the same wedges in the studio and in the lab", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid = { ...scene.fluid, refinementRegions: [{
    id: "region-1",
    min_m: { x: 0.2, y: 0, z: 0.2 },
    max_m: { x: 0.6, y: 0.4, z: 0.6 },
    rule: "minimum-cell-size",
    minimumCellSize_cells: 2,
  }] };
  const studio = entityActionsAt({ scene, bodies: [] }, {
    selection: { kind: "refinement-region", id: refinementRegionSelectionId("region-1") },
    point_m: POINT,
  });
  const lab = labRegionWedges(
    { regions: [{ id: "region-1", minimumFine: [8, 8], maximumFine: [24, 24],
      minimumCellWidth: 2 }], nx: 128, ny: 64 },
    "region-1");

  const shape = (actions: readonly EditorAction[]) => actions.map((action) => ({
    id: action.id, label: action.label, icon: action.icon,
    tone: action.tone, hint: action.hint, enabled: action.enabled,
  }));
  assert.deepEqual(shape(lab), shape(studio));
  assert.deepEqual(studio.map((action) => action.id), ["select", "delete"],
    "and the pair is still Edit then Delete, with the irreversible one last");

  // The effects differ, and must: this is the seam, not a hole in the pin.
  assert.equal(studio[1]?.effect?.kind, "scene");
  assert.equal(lab[1]?.effect?.kind, "host");
});
