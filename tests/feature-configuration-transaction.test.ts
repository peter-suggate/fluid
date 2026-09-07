import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createMethodStore } from "../lib/core/stores/method-store";

test("invalid surface selection leaves the entire method store unchanged", () => {
  const store = createMethodStore();
  const before = store.getState();
  assert.throws(() => before.setParam("power-liquids", "globalFineLevelSetFactor", "2"), /supported variant/);
  assert.equal(store.getState(), before);
  assert.throws(() => before.setMethodId("nonexistent"), /Unknown simulation method/);
  assert.equal(store.getState(), before);
});
test("inactive method settings survive switching while only supported configurations commit", () => {
  const store = createMethodStore();
  store.getState().setParam("power-liquids", "globalFineLevelSetFactor", "8");
  store.getState().setMethodId("power-liquids");
  store.getState().setMethodId("adaptive-mass");
  assert.equal(store.getState().overrides["power-liquids"]?.globalFineLevelSetFactor, "8");
});

test("controller rejects incompatible variants before announcing GPU work", async () => {
  const { simulation } = await import("../lib/core/simulation/controller");
  const { defaultSession } = await import("../lib/core/session/session");
  const methodBefore = defaultSession.method.getState();
  const statusBefore = defaultSession.diagnostics.getState().gpuStatus;
  assert.throws(() => simulation.setMethodParam("power-liquids", "globalFineLevelSetFactor", "2"), /supported variant/);
  assert.equal(defaultSession.method.getState(), methodBefore);
  assert.equal(defaultSession.diagnostics.getState().gpuStatus, statusBefore);
});
