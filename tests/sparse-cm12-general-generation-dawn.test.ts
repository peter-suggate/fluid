import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { cloneScene, defaultScene } from "../lib/core/model";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("ordinary policy merges quiet siblings in a non-ocean scene", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-general-generation-dawn");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = (await gpu.requestAdapter())!;
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    assert.ok(device);
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    const scene = cloneScene(defaultScene);
    scene.rigidBodies = [];
    scene.container = { ...scene.container, width_m: 3.2, height_m: 3.2, depth_m: 3.2, fillFraction: 1 };
    scene.voxelDomain.finestCellSize_m = 0.05;
    scene.fluid.initialCondition = "dam-break";
    scene.fluid.initialDamBreakDimensions_m = { x: 3.2, y: 3.2, z: 3.2 };
    scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined, {
      ...adaptiveMassSolverOptions({}), initialResolutionForQA: 1, topologyPageBudget: 0,
    }, () => {});
    await solver.waitForSimulationReady();
    const initial = await solver.readGPUActivityPolicy();
    const initialMass = (await solver.readDiagnosticFields()).density.reduce((sum, rho) => sum + rho, 0);
    console.log(JSON.stringify({initialLeaves:initial.bricks.length, spans:[...new Set(initial.bricks.map(b=>b.spanBricks))]}));
    for (let step = 1; step <= 129; step++) {
      await solver.waitForTopologyReady();
      while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion?.();
      await device.queue.onSubmittedWorkDone();
    }
    await solver.waitForTopologyReady();
    const after = await solver.readGPUActivityPolicy();
    console.log(JSON.stringify({generations:solver.info.topologyGenerationCount,error:solver.info.topologyGenerationError,
      leaves:after.bricks.length, spans:[...new Set(after.bricks.map(b=>b.spanBricks))],
      quiet:after.bricks.filter(b=>b.quietEpochs>=64).length,reasons:[...new Set(after.bricks.map(b=>b.reasons))]}));
    assert.equal(solver.info.topologyGenerationError, undefined);
    assert.ok((solver.info.topologyGenerationCount ?? 0) > 0);
    assert.ok(after.bricks.some(b=>b.active && 8*b.spanBricks/b.acceptedResolution>=16));
    assert.equal(after.faultFlags,0);
    const fields = await solver.readDiagnosticFields();
    const mass = fields.density.reduce((sum, rho) => sum + rho, 0);
    assert.ok(Math.abs(mass - initialMass) < 0.1, `quiet replacement mass ${initialMass} -> ${mass}`);
    assert.deepEqual(errors,[]);
  } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
