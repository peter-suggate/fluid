import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { simulation } from "../lib/core/simulation/controller";
import { defaultScene, type SceneDescription } from "../lib/core/model";
import type { SceneCard } from "../lib/core/scene-definition";

/**
 * A pane has one edit gesture at a time, and only its handle closes it. These
 * are the regressions the shared `pendingEdit`/draft slot had: an undo in the
 * middle of a drag cleared the slot, so the release landed an edit with no undo
 * entry, and an async continuation closed whatever gesture was open by then.
 */
function freshPane(context: { mock: { method: typeof test.mock.method } }) {
  const session = simulation.session();
  session.scene.getState().setScene({ ...defaultScene, sceneId: "start" }, "test");
  session.history.getState().clear();
  // The solver half of a reset needs a GPU; its document half is what history sees.
  context.mock.method(simulation, "reset", (scene?: SceneDescription, presetId?: string) => {
    if (scene) session.scene.getState().setScene(scene, presetId);
  });
  return session;
}

test("undo is refused while a gesture is open, so the release still records", (context) => {
  const session = freshPane(context);
  simulation.edit("Renamed once", { sceneId: "first" });
  const drag = simulation.beginDraft("terrain", "Shaped terrain");
  assert.ok(drag);
  session.sceneDraft.getState().updateDraft({ sceneId: "dragged" });
  assert.equal(simulation.undo(), false);
  assert.equal(session.scene.getState().scene.sceneId, "first");
  assert.equal(drag.commit({ reseed: true }), true);
  assert.equal(session.scene.getState().scene.sceneId, "dragged");
  assert.equal(session.history.getState().past.length, 2);
  assert.equal(simulation.undo(), true);
  assert.equal(session.scene.getState().scene.sceneId, "first");
});

test("a second gesture is refused while one is open", (context) => {
  freshPane(context);
  const first = simulation.beginDraft("fluid-body", "Dropped a ball of water");
  assert.ok(first);
  assert.equal(simulation.beginDraft("terrain", "Shaped terrain"), undefined);
  assert.equal(simulation.beginEdit("Adjusted"), undefined);
  first.cancel();
  assert.equal(first.open, false);
  const second = simulation.beginEdit("Adjusted");
  assert.ok(second);
  second.cancel();
});

test("a stale handle closes nothing: the late continuation cannot steal a newer gesture", (context) => {
  const session = freshPane(context);
  const drop = simulation.beginDraft("fluid-body", "Dropped a ball of water");
  assert.ok(drop);
  // A scene opened over the drop abandons it…
  simulation.openSceneCard({ id: "other", source: "starter", name: "Other", blurb: "", shelf: "",
    open: () => ({ scene: { ...defaultScene, sceneId: "other" }, presetId: "other" }) } as unknown as SceneCard);
  assert.equal(drop.open, false);
  const next = simulation.beginDraft("terrain", "Shaped terrain");
  assert.ok(next);
  session.sceneDraft.getState().updateDraft({ sceneId: "terrain" });
  // …so the injection resolving afterwards cannot cancel the terrain drag.
  drop.cancel();
  assert.equal(drop.commit(), false);
  assert.equal(next.open, true);
  assert.equal(session.sceneDraft.getState().draft?.patch.sceneId, "terrain");
  next.cancel();
});

test("an atomic edit inside an open gesture joins it, and a cancel still records it", (context) => {
  const session = freshPane(context);
  const hold = simulation.beginEdit("Adjusted a dial");
  assert.ok(hold);
  session.scene.getState().patchScene({ sceneId: "dial" });
  assert.equal(simulation.edit("Toggled gravity", { sceneId: "gravity" }, { reseed: true }), true);
  assert.equal(session.history.getState().past.length, 0);
  hold.cancel();
  assert.equal(session.history.getState().past.length, 1);
  assert.equal(session.scene.getState().scene.sceneId, "gravity");
  assert.equal(simulation.undo(), true);
  assert.equal(session.scene.getState().scene.sceneId, "start");
});
