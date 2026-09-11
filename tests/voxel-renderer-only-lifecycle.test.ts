import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { sceneryConstructionKey } from "../lib/core/scenery-construction-key";
import { createEmptyScene } from "../lib/core/empty-scene";
import { FluidLabRenderer, gpuSceneSolverKey, type SimulationRunConfig } from "../lib/core/webgpu-renderer";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import type { SceneDescription } from "../lib/core/model";
import { initialResourceReadiness, reduceGPUResourceStatus, reduceGPUResourceEvidence, resourceActivities } from "../lib/core/resource-readiness";
import { liveSvoSceneResourcePlugin } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import type { ResourcePluginDefinition } from "../lib/core/resource-plugin";
import type { EffectiveRendererStatus } from "../lib/core/renderer-status";

test("a dry scene adopts voxel edits without alternating rebuild identities", () => {
  const scene = createEmptyScene();
  const config = { methodId: "adaptive-volume", quality: "balanced", values: {} } as SimulationRunConfig;
  const staged: SceneDescription[] = [];
  const source = { stageSceneUpdate(next: SceneDescription) { staged.push(next); }, info: {} } as unknown as GPUSolverInstance;
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  Object.assign(renderer, {
    device: {}, gpuFluid: source,
    gpuFluidKey: `${gpuSceneSolverKey(scene, config)}:presentation-full-scene:scenery-${sceneryConstructionKey(scene)}`,
    appliedSceneUniformKey: "previous-fluid-scene-values",
    beginGPUFluidInitialization() { assert.fail("Renderer-only edit started a source rebuild"); },
  });
  const access = renderer as unknown as {
    currentGPUFluid(scene: SceneDescription, config: SimulationRunConfig, mode: "full-scene"): GPUSolverInstance;
  };
  for (let frame = 0; frame < 3; frame++) assert.equal(access.currentGPUFluid(scene, config, "full-scene"), source);
  assert.deepEqual(staged, [scene]);
  const edited: SceneDescription = { ...scene, solidVoxels: [...scene.solidVoxels,
    { operation: "fill", minimum: [1, 0, 1], maximumExclusive: [2, 1, 2] }] };
  for (let frame = 0; frame < 3; frame++) assert.equal(access.currentGPUFluid(edited, config, "full-scene"), source);
  assert.deepEqual(staged, [scene, edited]);
});

test("pending presentation does not copy another resource's activity into its own task", () => {
  const presentation: ResourcePluginDefinition = {
    id: "test.presentation", lane: "svo", label: "Presentation",
    provides: ["sparse-voxel-presentation"], blocks: "nothing",
  };
  let state = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "ready", label: "Presentation ready", adapter: "mock", resource: presentation,
  });
  state = reduceGPUResourceStatus(state, {
    state: "initializing", label: "Loading live scene", startedAt_ms: 1, resource: liveSvoSceneResourcePlugin,
  });
  state = reduceGPUResourceEvidence(state, undefined,
    { state: "pending", detail: "Waiting for scene", failureReason: "pipeline-compiling" } as EffectiveRendererStatus);
  const activities = resourceActivities(state);
  assert.equal(activities.length, 1);
  assert.equal(activities[0].pluginId, liveSvoSceneResourcePlugin.id);
  assert.equal(state.plugins[presentation.id].activity, undefined);
});

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
