import assert from "node:assert/strict";
import test from "node:test";
import { useMethodStore } from "../lib/core/stores/method-store";

test("default method store can be imported before the entry point installs methods", async () => {
  await import("../lib/methods");
  const first = useMethodStore.getState();
  assert.equal(first.methodId, "uniform-volume");
  assert.equal(useMethodStore.getState(), first);
  let notifications = 0;
  const unsubscribe = useMethodStore.subscribe(() => notifications++);
  first.setParam("uniform-volume", "volumeStorage", "pages32");
  assert.equal(useMethodStore.getState().overrides["uniform-volume"]?.volumeStorage, "pages32");
  assert.equal(notifications, 1);
  unsubscribe();
  useMethodStore.getState().resetParams("uniform-volume");
});
