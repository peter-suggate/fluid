import "../lib/methods";
import { uniformMethod } from "../lib/methods/uniform/method";
import test from "node:test";
import assert from "node:assert/strict";
import { createPaneSession } from "../lib/core/session/session";
import { defaultScene } from "../lib/core/model";
import { createCoarseFirstPoolImpactQuarterScene } from "../lib/core/scenes";
import { hostTransportFailure } from "../lib/core/simulation/host-transport-status";
import { effectiveSimulationStep_s } from "../lib/core/simulation-step";

// Analytic study scenes may explicitly override the default document timestep.
test("the default scene document uses the CM12 paper timestep", () => {
  assert.equal(defaultScene.numerics.fixedDt_s, 1 / 30);
  assert.equal(defaultScene.numerics.maxDt_s, 1 / 30);
});

test("quarter pool and Uniform now agree by default; explicit incompatible steps are rejected", () => {
  const a = createPaneSession("a"), b = createPaneSession("b");
  a.scene.setState({ scene: createCoarseFirstPoolImpactQuarterScene() });
  b.scene.setState({ scene: createCoarseFirstPoolImpactQuarterScene() });
  a.method.setState({ methodId: "adaptive-mass" });
  b.method.setState({ methodId: "uniform" });
  assert.equal(effectiveSimulationStep_s(a.scene.getState().scene, a.method.getState()), 1 / 30);
  assert.equal(effectiveSimulationStep_s(b.scene.getState().scene, b.method.getState()), 1 / 30);
  assert.equal(hostTransportFailure([a, b]), undefined);
  a.method.setState({ overrides: { "adaptive-mass": { timeStep: "scene" } } });
  a.scene.getState().patchNumerics({ fixedDt_s: 1 / 60, maxDt_s: 1 / 60 });
  assert.match(hostTransportFailure([a, b])!, /A = 16.67 ms, B = 33.33 ms/);
  assert.equal(a.scene.getState().scene.numerics.fixedDt_s, 1 / 60, "preflight must not silently repair user settings");
});

test("pane B failure blocks the whole experiment while cleanup is pending", () => {
  const a = createPaneSession("a"), b = createPaneSession("b");
  b.diagnostics.setState({ gpuStatus: { state: "stopping", label: "Invalid bind group; draining", resource: uniformMethod.resource } });
  assert.match(hostTransportFailure([a, b])!, /Pane B: Invalid bind group; draining/);
});
