import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../../core/model";
import { createEditorHistoryStore } from "../../core/stores/history-store";
import { gravityEnabled, setGravity, toggleGravity } from "./state";

test("toggle memory survives serialization and independent UI representations", () => {
  const original = { ...defaultScene.fluid, gravity_m_s2: { x: 2, y: -4, z: 3 } };
  const disabled = toggleGravity(original);
  assert.equal(gravityEnabled(disabled.gravity_m_s2), false);
  const saved = JSON.parse(JSON.stringify(disabled));
  assert.deepEqual(toggleGravity(saved).gravity_m_s2, original.gravity_m_s2);
  assert.deepEqual(original.gravity_m_s2, { x: 2, y: -4, z: 3 });
});

test("direct vector editor and toggle share latest enabled acceleration", () => {
  const disabled = toggleGravity(defaultScene.fluid);
  const edited = setGravity(disabled, { x: 0, y: -2, z: 0 });
  assert.deepEqual(toggleGravity(toggleGravity(edited)).gravity_m_s2, { x: 0, y: -2, z: 0 });
  const zeroed = setGravity(edited, { x: 0, y: 0, z: 0 });
  assert.deepEqual(toggleGravity(zeroed).gravity_m_s2, edited.gravity_m_s2);
});

test("undo and redo restore both gravity and remembered vector", () => {
  const history = createEditorHistoryStore();
  const initial = cloneScene(defaultScene);
  initial.fluid = setGravity(initial.fluid, { x: 1, y: -3, z: 2 });
  const disabled = { ...initial, fluid: toggleGravity(initial.fluid) };
  const snapshot = (scene: typeof initial) => ({ scene, presetId: "test", label: "Gravity" });
  history.getState().record(snapshot(initial));
  assert.deepEqual(history.getState().undo(snapshot(disabled))?.scene.fluid, initial.fluid);
  const redone = history.getState().redo(snapshot(initial))!.scene;
  assert.deepEqual(toggleGravity(redone.fluid).gravity_m_s2, initial.fluid.gravity_m_s2);
});

test("fresh zero-gravity scene cannot inherit another scene's remembered vector", () => {
  const first = toggleGravity(setGravity(defaultScene.fluid, { x: 3, y: -2, z: 0 }));
  const fresh = { ...defaultScene.fluid, gravity_m_s2: { x: 0, y: 0, z: 0 } };
  assert.deepEqual(toggleGravity(fresh).gravity_m_s2, defaultScene.fluid.gravity_m_s2);
  assert.deepEqual(toggleGravity(first).gravity_m_s2, { x: 3, y: -2, z: 0 });
});

test("direction preserves strength and off-state memory", async () => {
  const { setGravityDirection, gravityDirection } = await import("./state");
  const authored = setGravity(defaultScene.fluid, { x: 3, y: -4, z: 0 });
  const side = setGravityDirection(authored, "positive-z");
  assert.deepEqual(side.gravity_m_s2, { x: 0, y: 0, z: 5 });
  assert.equal(gravityDirection(side), "positive-z");
  const off = setGravityDirection(toggleGravity(side), "up");
  assert.deepEqual(off.gravity_m_s2, { x: 0, y: 0, z: 0 });
  assert.equal(gravityDirection(off), "up");
  assert.deepEqual(toggleGravity(off).gravity_m_s2, { x: 0, y: 5, z: 0 });
  assert.equal(gravityDirection(authored), "custom");
});
