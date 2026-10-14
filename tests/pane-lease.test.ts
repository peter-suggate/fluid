import assert from "node:assert/strict";
import test from "node:test";
import { manualGPUControlTargetsPane, shutdownBrowserGPUSession } from "../lib/core/gpu-startup";
import { createPaneLeaseBroker } from "../lib/core/session/pane-lease";

test("rapid viewport replacements retain the original shutdown barrier and lease", async () => {
  const broker = createPaneLeaseBroker();
  const original = broker.acquire("a");
  assert.equal(original.status, "acquired");
  if (original.status !== "acquired") return;
  const pendingCompilation = Promise.withResolvers<void>();
  const originalShutdown = shutdownBrowserGPUSession({ shutdown: () => pendingCompilation.promise }, original.release);
  let replacementDestroyed = false;
  // This replacement unmounts while still waiting to acquire the old lease.
  const replacementShutdown = shutdownBrowserGPUSession({ shutdown: async () => {
    replacementDestroyed = true;
  } }, undefined, originalShutdown);
  let thirdStarted = false;
  const third = replacementShutdown.then(() => {
    const lease = broker.acquire("a");
    assert.equal(lease.status, "acquired");
    thirdStarted = true;
    if (lease.status === "acquired") lease.release();
  });
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(replacementDestroyed, true);
  assert.equal(thirdStarted, false);
  assert.deepEqual(broker.heldPanes(), ["a"]);
  // Compare's other pane is independent of this pane's shutdown chain.
  const paneB = broker.acquire("b");
  assert.equal(paneB.status, "acquired");
  pendingCompilation.resolve();
  await third;
  assert.equal(thirdStarted, true);
  if (paneB.status === "acquired") paneB.release();
  assert.deepEqual(broker.heldPanes(), []);
});

test("a manual start/stop acts on the pane it names, or on every pane", () => {
  const everyPane = new CustomEvent("fluid-lab:start-gpu", { detail: {} });
  assert.equal(manualGPUControlTargetsPane(everyPane, "a"), true);
  assert.equal(manualGPUControlTargetsPane(everyPane, "b"), true);

  const paneB = new CustomEvent("fluid-lab:stop-gpu", { detail: { paneId: "b" } });
  assert.equal(manualGPUControlTargetsPane(paneB, "a"), false);
  assert.equal(manualGPUControlTargetsPane(paneB, "b"), true);

  // The page's START/STOP buttons dispatch with no detail at all.
  assert.equal(manualGPUControlTargetsPane(new Event("fluid-lab:start-gpu"), "a"), true);
});
