import assert from "node:assert/strict";
import test from "node:test";

import "../lib/methods";
import { createMinimalPowerDamBreak64Scene, getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import {
  FluidLabRenderer,
  gpuSceneSolverKey,
  gpuSceneUniformKey,
  sceneEditRequiresReset,
  type SimulationRunConfig,
} from "../lib/core/webgpu-renderer";

test("Sparse CM12 refinement regions are a live policy edit", () => {
  const before = createMinimalPowerDamBreak64Scene();
  const after = structuredClone(before);
  after.fluid.refinementRegions = [{
    id: "live-whole-domain-two-cell-floor",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 2,
    min_m: { x: -0.5 * after.container.width_m, y: 0,
      z: -0.5 * after.container.depth_m },
    max_m: { x: 0.5 * after.container.width_m, y: after.container.height_m,
      z: 0.5 * after.container.depth_m },
  }];
  const config: SimulationRunConfig = {
    methodId: "adaptive-mass",
    quality: "balanced",
    values: {},
  };

  assert.equal(sceneEditRequiresReset(before, after, "adaptive-mass"), false,
    "drawing a refinement box must not reset the simulation timeline");
  assert.equal(gpuSceneSolverKey(before, config), gpuSceneSolverKey(after, config),
    "a region edit must retain the attached Sparse CM12 solver");
  assert.notEqual(gpuSceneUniformKey(before), gpuSceneUniformKey(after),
    "the retained solver must still observe and upload the new policy");
});

test("dropping the first moving body keeps a refined renderer-only world attached", () => {
  const before = sceneDocument(getSceneDefinition("garden-svo-lighting"));
  const after = structuredClone(before);
  const template = after.rigidBodies[0];
  assert.ok(template);
  after.rigidBodies.push({
    ...template,
    id: "live-drop",
    name: "Live drop",
    motion: "dynamic",
    position_m: { x: 0, y: before.container.height_m + 0.2, z: 0 },
  });
  const config: SimulationRunConfig = {
    methodId: "adaptive-mass",
    quality: "balanced",
    values: { svoEnvironmentRefinementDepth: 3 },
  };

  assert.equal(sceneEditRequiresReset(before, after, config.methodId), false);
  assert.equal(gpuSceneSolverKey(before, config), gpuSceneSolverKey(after, config),
    "the live rigid arena can adopt a dropped ball without rebuilding the environment");
  assert.notEqual(gpuSceneUniformKey(before), gpuSceneUniformKey(after),
    "the retained renderer must still upload the new rigid roster");
});

test("a live liquid edit retires the retained water mesh", () => {
  let injected = 0;
  let invalidated = 0;
  const renderer = Object.create(FluidLabRenderer.prototype) as {
    disposed: boolean;
    deviceLost: boolean;
    gpuFluid: {
      simulationReady: boolean;
      injectLiquidBall: () => void;
    };
    waterPipeline: { invalidateSurface: () => void };
    pausedPresentationRevision: number;
  };
  renderer.disposed = false;
  renderer.deviceLost = false;
  renderer.gpuFluid = {
    simulationReady: true,
    injectLiquidBall: () => { injected += 1; },
  };
  renderer.waterPipeline = {
    invalidateSurface: () => { invalidated += 1; },
  };
  renderer.pausedPresentationRevision = 7;

  assert.equal(FluidLabRenderer.prototype.injectLiquidBall.call(
    renderer as unknown as FluidLabRenderer, {
    centre_m: { x: 0, y: 0.8, z: 0 },
    radius_m: 0.1,
  }), true);
  assert.equal(injected, 1, "the edit must reach the attached solver once");
  assert.equal(invalidated, 1, "the next frame must re-extract the water surface");
  assert.equal((renderer as unknown as FluidLabRenderer).presentationRevision, 8,
    "a paused viewport must be woken for the replacement mesh");
});
