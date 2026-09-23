import assert from "node:assert/strict";
import test from "node:test";
import { createUIStore } from "../lib/core/stores/ui-store";

test("selecting a rigid body never implicitly picks it up", () => {
  const ui = createUIStore();
  ui.getState().selectBody("sphere-1");
  assert.equal(ui.getState().selectedBodyId, "sphere-1");
  assert.equal(ui.getState().carry, undefined);
  ui.getState().select({ kind: "body", id: "sphere-2" });
  assert.equal(ui.getState().selectedBodyId, "sphere-2");
  assert.equal(ui.getState().carry, undefined);
  ui.getState().beginCarry("sphere-2", "Sphere 2");
  const carry = ui.getState().carry;
  ui.getState().selectBody("sphere-2");
  assert.equal(ui.getState().carry, carry, "explicit carry survives re-selection");
  ui.getState().endCarry();
  ui.getState().selectBody("sphere-2");
  assert.equal(ui.getState().carry, undefined, "re-selection after a drop stays passive");
  ui.getState().beginCarry("sphere-2", "Sphere 2");
  ui.getState().selectBody("sphere-1");
  assert.equal(ui.getState().carry, undefined, "selecting another body ends the old carry");
});
