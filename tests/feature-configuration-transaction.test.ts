import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createMethodStore } from "../lib/core/stores/method-store";

test("a forbidden pressure override leaves the entire method store unchanged", () => {
  const store = createMethodStore();
  const before = store.getState();
  assert.throws(() => before.setParam("uniform-narrow-band-flip", "pressureCycleBudget", 1), /fixes "pressureCycleBudget"/);
  assert.equal(store.getState(), before);
  assert.equal(store.getState(), before);
});
test("controller rejects forbidden overrides before announcing GPU work", async () => {
  const { simulation } = await import("../lib/core/simulation/controller");
  const { defaultSession } = await import("../lib/core/session/session");
  const methodBefore = defaultSession.method.getState();
  const statusBefore = defaultSession.diagnostics.getState().gpuStatus;
  assert.throws(() => simulation.setMethodParam("uniform-narrow-band-flip", "pressureCycleBudget", 1), /fixes "pressureCycleBudget"/);
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
    session.method.setState({ methodId: "uniform-narrow-band-flip", overrides: { "uniform-narrow-band-flip": { phiCubicAdvection: "off" } } });
    session.runtime.setState({ simulationTime: 2 });
    simulation.resetMethodParam("uniform-narrow-band-flip", "phiCubicAdvection");
    assert.equal(session.method.getState().overrides["uniform-narrow-band-flip"]?.phiCubicAdvection, undefined);
    assert.equal(session.runtime.getState().simulationTime, 2);
    assert.equal(session.diagnostics.getState().gpuStatus, savedDiagnostics.gpuStatus);
    simulation.setMethodParam("uniform-narrow-band-flip", "phiCubicAdvection", "off");
    assert.equal(session.runtime.getState().simulationTime, 2);
  } finally {
    session.method.setState(savedMethod, true);
    session.runtime.setState(savedRuntime, true);
    session.diagnostics.setState(savedDiagnostics, true);
  }
});
