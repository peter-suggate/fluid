import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createMethodStore, resolvedMethodValues } from "../lib/core/stores/method-store";

test("a forbidden pressure override leaves the entire method store unchanged", () => {
  const store = createMethodStore();
  const before = store.getState();
  assert.throws(() => before.setParam("uniform-volume", "pressureCycleBudget", 1), /fixes "pressureCycleBudget"/);
  assert.equal(store.getState(), before);
  assert.throws(() => before.setMethodId("nonexistent"), /Unknown simulation method/);
  assert.equal(store.getState(), before);
});
test("unsupported coarsening values resolve to the declared Uniform default", () => {
  const store = createMethodStore();
  const defaults = resolvedMethodValues(store.getState());
  store.getState().setParam("uniform-volume", "coarsening", "octree");
  assert.equal(resolvedMethodValues(store.getState()).coarsening, defaults.coarsening);
});
test("method settings survive reselecting the maintained method", () => {
  const store = createMethodStore();
  store.getState().setParam("uniform-volume", "coarsening", "regions");
  store.getState().setMethodId("uniform-volume");
  store.getState().setMethodId("uniform-volume");
  assert.equal(store.getState().overrides["uniform-volume"]?.coarsening, "regions");
});

test("controller rejects forbidden overrides before announcing GPU work", async () => {
  const { simulation } = await import("../lib/core/simulation/controller");
  const { defaultSession } = await import("../lib/core/session/session");
  const methodBefore = defaultSession.method.getState();
  const statusBefore = defaultSession.diagnostics.getState().gpuStatus;
  assert.throws(() => simulation.setMethodParam("uniform-volume", "pressureCycleBudget", 1), /fixes "pressureCycleBudget"/);
  assert.equal(defaultSession.method.getState(), methodBefore);
  assert.equal(defaultSession.diagnostics.getState().gpuStatus, statusBefore);
});

test("restoring an explicit default removes the override without resetting the active timeline", async () => {
  const { simulation } = await import("../lib/core/simulation/controller");
  const { defaultSession: session } = await import("../lib/core/session/session");
  const savedMethod = session.method.getState();
  const savedRuntime = session.runtime.getState();
  const savedDiagnostics = session.diagnostics.getState();
  try {
    session.method.setState({ methodId: "uniform-volume", overrides: { "uniform-volume": { phiCubicAdvection: "off" } } });
    session.runtime.setState({ simulationTime: 2, topologyFrozen: true });
    simulation.resetMethodParam("uniform-volume", "phiCubicAdvection");
    assert.equal(session.method.getState().overrides["uniform-volume"]?.phiCubicAdvection, undefined);
    assert.equal(session.runtime.getState().simulationTime, 2);
    assert.equal(session.runtime.getState().topologyFrozen, true);
    assert.equal(session.diagnostics.getState().gpuStatus, savedDiagnostics.gpuStatus);
    simulation.setMethodParam("uniform-volume", "phiCubicAdvection", "off");
    assert.equal(session.runtime.getState().simulationTime, 2);
  } finally {
    session.method.setState(savedMethod, true);
    session.runtime.setState(savedRuntime, true);
    session.diagnostics.setState(savedDiagnostics, true);
  }
});
