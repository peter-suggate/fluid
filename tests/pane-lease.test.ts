import assert from "node:assert/strict";
import test from "node:test";
import { manualGPUControlTargetsPane } from "../lib/core/gpu-startup";

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
