import assert from "node:assert/strict";
import test from "node:test";
import { useMethodStore } from "../lib/core/stores/method-store";

test("default method store can be imported before the entry point installs methods", async () => {
  await import("../lib/methods");
  const first = useMethodStore.getState();
  assert.equal(first.methodId, "uniform-narrow-band-flip");
  assert.equal(useMethodStore.getState(), first);
  let notifications = 0;
  const unsubscribe = useMethodStore.subscribe(() => notifications++);
  first.setParam("uniform-narrow-band-flip", "fineGridPadding", 2);
  assert.equal(useMethodStore.getState().overrides["uniform-narrow-band-flip"]?.fineGridPadding, 2);
  assert.equal(notifications, 1);
  unsubscribe();
  useMethodStore.getState().resetParams("uniform-narrow-band-flip");
});

test("retired URL methods fall back and current parameters have no solver namespace", async () => {
  await import("../lib/methods");
  const { parseQueryState, serializeQueryState } = await import("../lib/core/url-state");
  const parsed = parseQueryState("?method=losasso&param.losasso.maximumLeafSize=32&param.fineGridPadding=2");
  assert.equal(parsed.methodId, "uniform-narrow-band-flip");
  assert.deepEqual(parsed.overrides, { "uniform-narrow-band-flip": { fineGridPadding: 2 } });
  const query = new URLSearchParams(serializeQueryState("?method=losasso", parsed, parsed, parsed.ui));
  assert.equal(query.has("method"), false);
  assert.equal(query.get("param.fineGridPadding"), "2");
  assert.equal(query.has("param.losasso.maximumLeafSize"), false);
});
