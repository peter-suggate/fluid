import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { defaultMethodId, getMethod, registeredSimulationMethods } from "../lib/core/method-registry";
import { SEEP_METHOD_ID, seepMethod } from "../lib/methods/seep-method";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";

test("seep-m1max is registered beside NB-FLIP, which stays the default", () => {
  assert.equal(defaultMethodId(), "uniform-narrow-band-flip");
  assert.ok(registeredSimulationMethods().some(m => m.id === SEEP_METHOD_ID));
  assert.equal(getMethod(SEEP_METHOD_ID), seepMethod);
  assert.match(seepMethod.label, /experimental/i);
});

test("seep-m1max refuses every scene but reservoir-seep, before touching the device", async () => {
  const scene = { ...sceneDocument(getSceneDefinition("reservoir-seep")), sceneId: "dam-break" };
  await assert.rejects(seepMethod.createSolverAsync!({} as GPUDevice, scene, "balanced", {}, undefined, () => {}), /runs only reservoir-seep/);
});

test("?method=seep-m1max opens the seep under the new solver and round-trips", async () => {
  const { parseQueryState, serializeQueryState } = await import("../lib/core/url-state");
  const parsed = parseQueryState("?scene=reservoir-seep&method=seep-m1max");
  assert.equal(parsed.methodId, SEEP_METHOD_ID);
  const query = new URLSearchParams(serializeQueryState("?scene=reservoir-seep&method=seep-m1max", parsed, parsed, parsed.ui));
  assert.equal(query.get("method"), SEEP_METHOD_ID);
  const fallback = parseQueryState("?scene=reservoir-seep");
  assert.equal(fallback.methodId, "uniform-narrow-band-flip");
});
