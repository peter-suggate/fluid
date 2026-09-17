import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import type { AdaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { createRuntimeStore } from "../lib/core/stores/runtime-store";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { FluidLabRenderer, gpuSceneSolverKey, gpuSceneUniformKey, type SimulationRunConfig } from "../lib/core/webgpu-renderer";

test("topology freeze is pane-local, reversible, and cleared on a new timeline", () => {
  const a = createRuntimeStore();
  const b = createRuntimeStore();
  a.getState().setSimulationTime(0.15);
  // A store opens paused, so the clock is started here to have something a
  // freeze could stop.
  a.getState().setRunState("running");
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
  const config: SimulationRunConfig = { methodId: "adaptive-volume", quality: "balanced", values: {} };
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



test("a second paused drop supersedes the support receipt before either dose is applied", async () => {
  let revision = 0, pending = 0, checks = 0;
  const applied: number[] = [];
  let enterFirst!: () => void, finishFirst!: (needed: boolean) => void;
  const entered = new Promise<void>(resolve => { enterFirst = resolve; });
  const firstReceipt = new Promise<boolean>(resolve => { finishFirst = resolve; });
  const atlas = {};
  const solver = Object.assign(Object.create(WebGPUAdaptiveMassSolver.prototype), {
    scene: createMinimalPowerDamBreak64Scene(),
    options: { activityPolicy: { freezeTopology: true } },
    info: { encodedSteps: 0 }, atlas, presentation: { allocatedBytes: 0 },
    frozenFrontierPending: false,
    topologyGenerationLimits: { maximumSpanBricks: 1 },
    assertSimulationHealthy: async () => {},
    sparseWorld: { edit: () => { revision++; pending++; } },
    sparseRuntime: {
      acceptedAtlas: atlas, allocatedBytes: 0, generationPlanningRequired: false,
      get pendingLiquidInteractions() { return pending > 0; },
      get pendingLiquidInteractionRevision() { return revision; },
      needsDetailedGenerationPlanning: async () => {
        checks++;
        if (checks === 1) { enterFirst(); return firstReceipt; }
        return false;
      },
      completePendingLiquidInteractions: () => { applied.push(pending); pending = 0; },
    },
  }) as WebGPUAdaptiveMassSolver;
  const ball = { centre_m: { x: 0, y: .4, z: 0 }, radius_m: .1 };
  solver.injectLiquidBall(ball);
  await entered;
  solver.injectLiquidBall({ ...ball, centre_m: { x: .2, y: .4, z: 0 } });
  assert.deepEqual(applied, []);
  finishFirst(false);
  await solver.waitForTopologyReady();
  assert.equal(checks, 2, "the older receipt cannot authorize the second drop");
  assert.deepEqual(applied, [2], "both doses apply once after current support is verified");
  assert.equal(pending, 0);
  assert.equal(solver.info.encodedSteps, 0);
});
