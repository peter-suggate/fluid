import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { FluidLabRenderer, type SimulationRunConfig } from "../lib/core/webgpu-renderer";
import { createEmptyScene } from "../lib/core/empty-scene";
import type { SceneDescription } from "../lib/core/model";
import { initialResourceReadiness, reduceGPUResourceStatus, resourceActivities } from "../lib/core/resource-readiness";
import { liveSvoSceneResourcePlugin } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { uniformMethod } from "../lib/methods/uniform/method";
import { createPaneSession } from "../lib/core/session/session";
import { hostTransportFailure } from "../lib/core/simulation/host-transport-status";

test("replacing a pending initializer retires the superseded resource's progress", () => {
  const statuses: import("../lib/core/gpu-status").GPUStatus[] = [];
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, (status) => statuses.push(status));
  const abort = new AbortController();
  const pending = new Promise<void>(() => {});
  Object.assign(renderer, {
    device: {}, gpuFluidPending: pending,
    gpuFluidInitializationAbort: abort,
    gpuFluidInitializationResource: liveSvoSceneResourcePlugin,
  });
  const config = { methodId: "adaptive-volume", quality: "balanced", values: {} } as SimulationRunConfig;
  const access = renderer as unknown as {
    beginGPUFluidInitialization(scene: SceneDescription, config: SimulationRunConfig, key: string, mode: "full-scene"): void;
  };
  access.beginGPUFluidInitialization(createEmptyScene(), config, "replacement", "full-scene");
  assert.equal(abort.signal.aborted, true);
  assert.equal(statuses[0].state, "cancelled");
  let state = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "initializing", label: "Old source", startedAt_ms: 1, resource: liveSvoSceneResourcePlugin,
  });
  state = reduceGPUResourceStatus(state, statuses[0]);
  assert.equal(resourceActivities(state).length, 0);
  assert.equal(state.plugins[liveSvoSceneResourcePlugin.id].state, "idle");
});

test("cancelling replacement work preserves the previously usable resource", () => {
  let state = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "ready", label: "Original scene", adapter: "mock", resource: liveSvoSceneResourcePlugin,
  });
  state = reduceGPUResourceStatus(state, {
    state: "initializing", label: "Replacement scene", resource: liveSvoSceneResourcePlugin,
  });
  state = reduceGPUResourceStatus(state, {
    state: "cancelled", label: "Replacement superseded", resource: liveSvoSceneResourcePlugin,
  });
  assert.equal(resourceActivities(state).length, 0);
  assert.equal(state.plugins[liveSvoSceneResourcePlugin.id].usable, true);
  assert.equal(state.plugins[liveSvoSceneResourcePlugin.id].state, "ready");
});

test("superseded initialization is cancellation rather than a terminal transport failure", () => {
  const pane = createPaneSession("a");
  pane.diagnostics.setState({ gpuStatus: {
    state: "cancelled", label: "Previous scene initialization superseded", resource: uniformMethod.resource,
  } });
  assert.equal(hostTransportFailure([pane]), undefined);
});

test("a stale owner's cancellation keeps the active runtime, new owner and running timeline", async () => {
  const { liveSvoSceneResourcePlugin: oldOwner } = await import("../lib/svo/features/scene-publication/webgpu-live-svo-scene");
  const pane = createPaneSession("a");
  pane.runtime.getState().setRunState("running");
  pane.diagnostics.getState().set({ gpuStatus: { state: "initializing", label: "Old scene", resource: oldOwner } });
  pane.diagnostics.getState().set({ gpuStatus: { state: "ready", label: "Current fluid", adapter: "mock", resource: uniformMethod.resource } });
  const before = pane.diagnostics.getState();
  pane.diagnostics.getState().set({ gpuStatus: { state: "cancelled", label: "Old scene superseded", resource: oldOwner } });
  const after = pane.diagnostics.getState();
  assert.equal(after.gpuStatus, before.gpuStatus);
  assert.equal(after.resourceReadiness.activeLane, before.resourceReadiness.activeLane);
  assert.equal(after.resourceReadiness.plugins[uniformMethod.resource.id], before.resourceReadiness.plugins[uniformMethod.resource.id]);
  assert.equal(after.resourceReadiness.plugins[oldOwner.id].activity, undefined);
  assert.equal(after.resourceReadiness.plugins[oldOwner.id].usable, false);
  assert.equal(hostTransportFailure([pane]), undefined);
  assert.equal(pane.runtime.getState().runState, "running");
});
