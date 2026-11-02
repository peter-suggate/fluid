import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { simulation } from "../lib/core/simulation/controller";
import { createUIStore } from "../lib/core/stores/ui-store";
import { defaultScene } from "../lib/core/model";
import { add, scale } from "../lib/core/math";

test("selecting a rigid body never implicitly picks it up", () => {
  const ui = createUIStore();
  ui.getState().selectBody("sphere-1");
  assert.equal(ui.getState().selectedBodyId, "sphere-1");
  assert.equal(ui.getState().carry, undefined);
  ui.getState().select({ kind: "body", id: "sphere-2" });
  assert.equal(ui.getState().selectedBodyId, "sphere-2");
  assert.equal(ui.getState().carry, undefined);
});

/**
 * The controller is the one owner of a kinematic hold. These are the bugs the
 * old split (viewport ref, store carry, controller drag) had: switching the
 * carry to another body left the first one weightless, and the store could say
 * "carrying" with nothing held.
 */
function twoBodies() {
  const session = simulation.session();
  simulation.releaseBodyHold();
  simulation.reset({ ...defaultScene, rigidBodies: [] }, "test");
  simulation.addBody("sphere");
  simulation.addBody("box");
  const [a, b] = session.scene.getState().scene.rigidBodies;
  assert.ok(a && b);
  const held = (id: string) => session.diagnostics.getState().bodies.find(body => body.description.id === id)?.held;
  const at = (id: string) => session.diagnostics.getState().bodies.find(body => body.description.id === id)!.position_m;
  return { session, a: a.id, b: b.id, held, at };
}

test("a carry holds, selects and publishes; selecting another body puts it down", () => {
  const { session, a, b, held } = twoBodies();
  session.ui.getState().select(undefined);
  assert.equal(simulation.beginCarry(a, "Sphere"), true);
  assert.equal(held(a), true);
  assert.equal(session.ui.getState().selectedBodyId, a);
  assert.deepEqual(session.ui.getState().carry, { bodyId: a, label: "Sphere", tiltDegrees: 0 });
  session.ui.getState().selectBody(a);
  assert.equal(simulation.carrying(), true, "re-selecting the carried body keeps it in hand");
  session.ui.getState().selectBody(b);
  assert.equal(simulation.carrying(), false);
  assert.equal(held(a), false);
  assert.equal(session.ui.getState().carry, undefined);
});

test("carrying a second body lets go of the first", () => {
  const { session, a, b, held } = twoBodies();
  simulation.beginCarry(a);
  simulation.beginCarry(b);
  assert.equal(held(a), false);
  assert.equal(held(b), true);
  assert.equal(session.ui.getState().carry?.bodyId, b);
  simulation.endCarry();
  assert.equal(held(b), false);
});

test("a throw takes over from a carry, and a stale end releases nothing", () => {
  const { session, a, b, held, at } = twoBodies();
  simulation.beginCarry(a);
  simulation.dragBody(b, at(b), { x: 0, y: 0, z: 0 }, "start");
  assert.equal(simulation.carrying(), false);
  assert.equal(session.ui.getState().carry, undefined);
  assert.equal(held(a), false);
  assert.equal(held(b), true);
  // A gizmo release for a hold that is not the current one changes nothing.
  simulation.manipulateBody(b, at(b), "end");
  assert.equal(held(b), true);
  simulation.dragBody(a, at(a), { x: 0, y: 0, z: 0 }, "end");
  assert.equal(held(b), true);
  simulation.dragBody(b, at(b), { x: 0, y: 0, z: 0 }, "end");
  assert.equal(held(b), false);
  assert.equal(simulation.bodyHold(), undefined);
});

test("a carry follows the pointer on its plane, tilts, and Escape puts it back", () => {
  const { session, a, at } = twoBodies();
  const origin = { ...at(a) };
  simulation.beginCarry(a);
  const hold = simulation.bodyHold();
  assert.equal(hold?.kind, "carry");
  if (hold?.kind !== "carry") return;
  const normal = hold.grip.plane.normal;
  const target = add(origin, { x: 0, y: 0.1, z: 0 });
  simulation.carryTo({ origin: add(target, scale(normal, -1)), direction: normal }, performance.now() + 100);
  assert.ok(Math.abs(at(a).y - target.y) < 1e-9, "the body rises with the pointer");
  simulation.tiltCarry(2);
  assert.equal(session.ui.getState().carry?.tiltDegrees, 15);
  simulation.endCarry(true);
  assert.deepEqual(at(a), origin);
  assert.equal(session.ui.getState().carry, undefined);
});

test("a viewport going away releases whatever its pane holds", () => {
  const { a, held, at } = twoBodies();
  simulation.manipulateBody(a, at(a), "start");
  simulation.releaseBodyHold();
  assert.equal(held(a), false);
  assert.equal(simulation.bodyHold(), undefined);
});
