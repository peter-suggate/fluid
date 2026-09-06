import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import type { AdaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { createRuntimeStore } from "../lib/core/stores/runtime-store";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { FluidLabRenderer, gpuSceneSolverKey, gpuSceneUniformKey, type SimulationRunConfig } from "../lib/core/webgpu-renderer";

test("topology freeze is pane-local, reversible, and cleared on a new timeline", () => {
  const a = createRuntimeStore();
  const b = createRuntimeStore();
  a.getState().setSimulationTime(0.15);
  a.getState().setTopologyFrozen(true);
  assert.equal(a.getState().topologyFrozen, true);
  assert.equal(a.getState().simulationTime, 0.15);
  assert.equal(a.getState().runState, "running", "freezing topology does not pause physics");
  assert.equal(b.getState().topologyFrozen, false);
  a.getState().setTopologyFrozen(false);
  assert.equal(a.getState().topologyFrozen, false);
  a.getState().setTopologyFrozen(true);
  a.getState().resetSimulationTime();
  assert.equal(a.getState().topologyFrozen, false);
  assert.equal(a.getState().simulationEpoch, 1);
  assert.equal(a.getState().simulationTime, 0);
});

test("renderer applies freeze transitions to the retained solver without rebuilding", () => {
  const scene = createMinimalPowerDamBreak64Scene();
  const config: SimulationRunConfig = { methodId: "adaptive-mass", quality: "balanced", values: {} };
  const key = gpuSceneSolverKey(scene, config);
  assert.equal(gpuSceneSolverKey(scene, { ...config, topologyFrozen: true }), key);
  const calls: boolean[] = [];
  const solver = { setTopologyFrozen: (frozen: boolean) => calls.push(frozen) };
  const renderer = Object.assign(Object.create(FluidLabRenderer.prototype), {
    device: {}, gpuFluid: solver, gpuFluidKey: key,
    solverKey: () => key,
    appliedSceneUniformKey: gpuSceneUniformKey(scene),
  }) as {
    gpuFluid: typeof solver;
    currentGPUFluid: (document: typeof scene, config: SimulationRunConfig, mode: "full-scene") => unknown;
  };
  assert.equal(renderer.currentGPUFluid(scene, config, "full-scene"), solver);
  renderer.currentGPUFluid(scene, { ...config, topologyFrozen: true }, "full-scene");
  renderer.currentGPUFluid(scene, { ...config, topologyFrozen: true }, "full-scene");
  renderer.currentGPUFluid(scene, config, "full-scene");
  assert.deepEqual(calls, [false, true, false], "repeated frames do not cancel preparation again");
  renderer.gpuFluid = { setTopologyFrozen: (frozen) => calls.push(frozen) };
  renderer.currentGPUFluid(scene, config, "full-scene");
  assert.deepEqual(calls, [false, true, false, false], "a replacement solver receives current state");
});


test("live renderer parameter uploads retain a frozen topology until the toggle is released", () => {
  const scene = createMinimalPowerDamBreak64Scene();
  const config: SimulationRunConfig = { methodId: "adaptive-mass", quality: "balanced", values: {} };
  let cancelledPreparations = 0;
  // Exercise the production solver methods without constructing any GPU resources.
  // The browser uploads runtime values on every draw; offline probes do not.
  const solver = Object.assign(Object.create(WebGPUAdaptiveMassSolver.prototype), {
    options: {} as AdaptiveMassSolverOptions,
    info: {},
    pressureIterationControlGeneration: 0,
    sparseRuntime: { cancelTopologyPreparation: () => { cancelledPreparations += 1; } },
  }) as WebGPUAdaptiveMassSolver;
  const policy = () => (solver as unknown as { options: AdaptiveMassSolverOptions }).options.activityPolicy;
  const renderer = Object.assign(Object.create(FluidLabRenderer.prototype), {
    device: {}, gpuFluid: solver, gpuFluidKey: gpuSceneSolverKey(scene, config),
    solverKey: () => gpuSceneSolverKey(scene, config),
    appliedSceneUniformKey: gpuSceneUniformKey(scene),
  }) as {
    currentGPUFluid: (document: typeof scene, config: SimulationRunConfig, mode: "full-scene") => unknown;
  };
  renderer.currentGPUFluid(scene, config, "full-scene");
  renderer.currentGPUFluid(scene, { ...config, topologyFrozen: true }, "full-scene");
  assert.equal(policy()?.freezeTopology, true);
  for (let frame = 0; frame < 5; frame += 1) {
    renderer.currentGPUFluid(scene, {
      ...config, topologyFrozen: true,
      values: { selectorMode: "coarse-first", energyThreshold: 2 + frame },
    }, "full-scene");
    assert.equal(policy()?.freezeTopology, true, `frame ${frame}: runtime values must not clear freeze`);
    assert.equal(policy()?.energyThreshold, 2 + frame, "live settings still update while frozen");
  }
  assert.equal(cancelledPreparations, 1, "holding freeze must not repeatedly cancel preparation");
  renderer.currentGPUFluid(scene, config, "full-scene");
  assert.equal(policy()?.freezeTopology, false, "the toggle releases the current solver");
});
