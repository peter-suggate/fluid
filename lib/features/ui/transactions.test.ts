import assert from "node:assert/strict";
import test from "node:test";
import { createUIStore } from "../../core/stores/ui-store";

test("invalid SVO variants fail before UI state mutation or notification", () => {
  const store = createUIStore();
  const before = store.getState();
  let notifications = 0;
  store.subscribe(() => notifications++);
  assert.throws(() => store.getState().setSvoPrimaryTraversal("missing" as never), /supported variant/);
  assert.throws(() => store.getState().setSvoConeTracingMode("missing" as never), /supported variant/);
  assert.equal(store.getState(), before);
  assert.equal(notifications, 0);
  store.getState().setSvoPrimaryTraversal("traced");
  store.getState().setSvoConeTracingMode("exact");
  assert.equal(store.getState().svoPrimaryTraversal, "traced");
  assert.equal(store.getState().svoConeTracingMode, "exact");
});
