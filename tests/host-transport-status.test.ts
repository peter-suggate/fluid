import "../lib/methods";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import test from "node:test";
import assert from "node:assert/strict";
import { createPaneSession } from "../lib/core/session/session";
import { defaultScene } from "../lib/core/model";
import { createCoarseFirstPoolImpactQuarterScene } from "../lib/core/scenes";
import { hostTransportFailure } from "../lib/core/simulation/host-transport-status";
import { effectiveSimulationStep_s } from "../lib/core/simulation-step";

// Analytic study scenes may explicitly override the default document timestep.
test("the default scene document uses its authored timestep", () => {
  assert.equal(defaultScene.numerics.fixedDt_s, 1 / 30);
  assert.equal(defaultScene.numerics.maxDt_s, 1 / 30);
});

test("Uniform Geometric panes agree by default; explicit incompatible steps are rejected", () => {
  const a = createPaneSession("a"), b = createPaneSession("b");
  a.scene.setState({ scene: createCoarseFirstPoolImpactQuarterScene() });
  b.scene.setState({ scene: createCoarseFirstPoolImpactQuarterScene() });
  a.method.setState({ methodId: "uniform-volume" });
  b.method.setState({ methodId: "uniform-volume" });
  assert.equal(effectiveSimulationStep_s(a.scene.getState().scene, a.method.getState()), 1 / 60);
  assert.equal(effectiveSimulationStep_s(b.scene.getState().scene, b.method.getState()), 1 / 60);
  assert.equal(hostTransportFailure([a, b]), undefined);
  a.method.setState({ overrides: { "uniform-volume": { timeStep: "scene" } } });
  a.scene.getState().patchNumerics({ fixedDt_s: 1 / 30, maxDt_s: 1 / 30 });
  assert.match(hostTransportFailure([a, b])!, /A = 33.33 ms, B = 16.67 ms/);
  assert.equal(a.scene.getState().scene.numerics.fixedDt_s, 1 / 30, "preflight must not silently repair user settings");
});

test("pane B failure blocks the whole experiment while cleanup is pending", () => {
  const a = createPaneSession("a"), b = createPaneSession("b");
  b.diagnostics.setState({ gpuStatus: { state: "stopping", label: "Invalid bind group; draining", resource: uniformVolumeMethod.resource } });
  assert.match(hostTransportFailure([a, b])!, /Pane B: Invalid bind group; draining/);
});

test("superseded initialization is cancellation rather than a terminal transport failure", () => {
  const pane = createPaneSession("a");
  pane.diagnostics.setState({ gpuStatus: {
    state: "cancelled", label: "Previous scene initialization superseded", resource: uniformVolumeMethod.resource,
  } });
  assert.equal(hostTransportFailure([pane]), undefined);
});

test("a stale owner's cancellation keeps the active runtime, new owner and running timeline", async () => {
  const { liveSvoSceneResourcePlugin: oldOwner } = await import("../lib/svo/features/scene-publication/webgpu-live-svo-scene");
  const pane = createPaneSession("a");
  pane.runtime.getState().setRunState("running");
  pane.diagnostics.getState().set({ gpuStatus: { state: "initializing", label: "Old scene", resource: oldOwner } });
  pane.diagnostics.getState().set({ gpuStatus: { state: "ready", label: "Current fluid", adapter: "mock", resource: uniformVolumeMethod.resource } });
  const before = pane.diagnostics.getState();
  pane.diagnostics.getState().set({ gpuStatus: { state: "cancelled", label: "Old scene superseded", resource: oldOwner } });
  const after = pane.diagnostics.getState();
  assert.equal(after.gpuStatus, before.gpuStatus);
  assert.equal(after.resourceReadiness.activeLane, before.resourceReadiness.activeLane);
  assert.equal(after.resourceReadiness.plugins[uniformVolumeMethod.resource.id], before.resourceReadiness.plugins[uniformVolumeMethod.resource.id]);
  assert.equal(after.resourceReadiness.plugins[oldOwner.id].activity, undefined);
  assert.equal(after.resourceReadiness.plugins[oldOwner.id].usable, false);
  assert.equal(hostTransportFailure([pane]), undefined);
  assert.equal(pane.runtime.getState().runState, "running");
});


test("superseded preparation retires its activity without halting accepted fluid", async () => {
  const { reduceGPUResourceStatus, initialResourceReadiness, resourceActivities } = await import("../lib/core/resource-readiness");
  const resource = uniformVolumeMethod.resource;
  let readiness = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "ready", label: "Accepted fluid", adapter: "mock", resource,
  });
  readiness = reduceGPUResourceStatus(readiness, { state: "initializing", label: "Replacement", resource });
  const cancelled = { state: "cancelled", label: "Previous scene initialization superseded", resource } as const;
  readiness = reduceGPUResourceStatus(readiness, cancelled);
  assert.equal(resourceActivities(readiness).length, 0);
  assert.equal(readiness.plugins[resource.id].usable, true);
  assert.equal(readiness.plugins[resource.id].state, "ready");
  const pane = createPaneSession("a");
  pane.diagnostics.setState({ gpuStatus: cancelled, resourceReadiness: readiness });
  assert.equal(hostTransportFailure([pane]), undefined);
  pane.diagnostics.setState({ gpuStatus: { state: "stopping", label: "Real failure draining", resource } });
  assert.match(hostTransportFailure([pane])!, /Real failure draining/);
});
