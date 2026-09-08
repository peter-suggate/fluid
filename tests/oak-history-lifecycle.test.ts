import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { simulation } from "../lib/core/simulation/controller";
import { createSceneryNodeAt, scenerySelectionId } from "../lib/core/editor-scenery";
import { withOakParameters } from "../lib/core/oak-tree-controls";
import { defaultScene } from "../lib/core/model";

test("tree commit, undo and redo retain selection and the running timeline", context => {
  const session = simulation.session();
  const node = createSceneryNodeAt(defaultScene, "oak-v2", { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
  const initial = { ...defaultScene, scenery: { palettes: {}, nodes: [node] } };
  session.scene.getState().setScene(initial, "test");
  session.history.getState().clear();
  const selection = { kind: "scenery" as const, id: scenerySelectionId(node.id) };
  session.ui.getState().select(selection);
  session.runtime.getState().setRunState("running");
  context.mock.method(simulation, "reset", () => { assert.fail("Scenery history reset the fluid timeline"); });
  simulation.beginEdit("Changed oak");
  const edited = withOakParameters(initial, node.id, { twigDepth: 1 });
  simulation.commitEdit(edited, { reseed: true });
  assert.equal(session.history.getState().past.length, 1);
  assert.equal(simulation.undo(), true);
  assert.deepEqual(session.scene.getState().scene.scenery, initial.scenery);
  assert.deepEqual(session.ui.getState().selection, selection);
  assert.equal(session.runtime.getState().runState, "running");
  assert.equal(simulation.redo(), true);
  assert.deepEqual(session.scene.getState().scene.scenery, edited.scenery);
  assert.deepEqual(session.ui.getState().selection, selection);
  assert.equal(session.runtime.getState().runState, "running");
});

test("undoing tree placement clears a selection whose tree no longer exists", context => {
  const session = simulation.session();
  const initial = { ...defaultScene, scenery: { palettes: {}, nodes: [] } };
  session.scene.getState().setScene(initial, "test");
  session.history.getState().clear();
  context.mock.method(simulation, "reset", () => { assert.fail("Tree placement history reset the scene"); });
  simulation.addScenery("oak-v2", { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
  assert.equal(session.ui.getState().selection?.kind, "scenery");
  simulation.undo();
  assert.equal(session.ui.getState().selection, undefined);
  assert.deepEqual(session.scene.getState().scene.scenery, initial.scenery);
});
