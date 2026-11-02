import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { SimulationController } from "../lib/core/simulation/controller";
import { createPaneSession } from "../lib/core/session/session";

/**
 * A pane's realm and its place on the host clock join and leave together. The
 * regression: compare mode attached the realm in one component and the
 * viewport registered the clock in another, so a viewport remount left the host
 * out of lockstep with pane B's realm still attached, and a compare verb could
 * register the clock for a pane that had no realm.
 */
test("a pane joins the clock with its realm and leaves with it", () => {
  const controller = new SimulationController();
  const b = createPaneSession("b");
  assert.deepEqual(controller.paneIds(), ["a"]);
  controller.attachPane("b", b);
  controller.attachPane("b", b);
  assert.deepEqual(controller.paneIds(), ["a", "b"]);
  assert.equal(controller.session("b"), b);
  assert.equal(controller.detachPane("b"), true);
  assert.deepEqual(controller.paneIds(), ["a"]);
  assert.notEqual(controller.session("b"), b);
  assert.equal(controller.detachPane("b"), false);
  assert.equal(controller.detachPane("a"), false);
  assert.deepEqual(controller.paneIds(), ["a"]);
});
