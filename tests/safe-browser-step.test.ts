import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { SimulationController } from "../lib/core/simulation/controller";
import { createPaneSession } from "../lib/core/session/session";
import { PRIMARY_PANE_ID } from "../lib/core/simulation/pane-clock";
import { createInteractivePlayground } from "../lib/interaction/playground";

const settle = async (predicate: () => boolean) => {
  for (let i = 0; i < 200 && !predicate(); i++) await new Promise(resolve => setTimeout(resolve, 5));
  assert.ok(predicate());
};

/**
 * Safe bring-up allows one STEP per page load, and the controller's latch is
 * the only record of it. The transport used to keep its own flag, set on the
 * click, so a STEP the host refused still locked the button for good.
 */
test("under safe bring-up a refused STEP spends nothing and an accepted one spends the only one", async () => {
  const had = Object.getOwnPropertyDescriptor(globalThis, "location");
  Object.defineProperty(globalThis, "location", { configurable: true, value: { search: "?gpu=safe" } });
  const controller = new SimulationController(), pane = createPaneSession(PRIMARY_PANE_ID);
  pane.scene.getState().setScene(createInteractivePlayground());
  controller.attachPane(PRIMARY_PANE_ID, pane);
  const host = controller.interaction(PRIMARY_PANE_ID);
  try {
    controller.singleStep();
    assert.equal(controller.safeBrowserStepSpent(), false, "the host was not ready, so no step was taken");
    await host.ready;
    controller.singleStep();
    assert.equal(controller.safeBrowserStepSpent(), true);
    await settle(() => controller.completedTime() > 0);
    const after = controller.completedTime();
    controller.singleStep();
    await new Promise(resolve => setTimeout(resolve, 50));
    assert.equal(controller.completedTime(), after, "the second STEP is refused");
  } finally {
    await host.dispose();
    if (had) Object.defineProperty(globalThis, "location", had);
    else delete (globalThis as { location?: unknown }).location;
  }
});
