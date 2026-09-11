import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene } from "../lib/core/model";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("coarse bulk macros refine in place with conservative 2:1 faces", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "coarse-macro-rerung");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    assert.ok(device);
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const scene = cloneScene(defaultScene);
    scene.rigidBodies = [];
    scene.solidVoxels = [];
    scene.container = { ...scene.container, width_m: 1.6, height_m: 0.8, depth_m: 0.8, fillFraction: 1, top: "closed" };
    scene.voxelDomain.finestCellSize_m = 0.05;
    scene.fluid.initialCondition = "tank-fill";
    scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined, {
      ...adaptiveMassSolverOptions({}), maximumMacroSpanBricks: 2,
      gammaDiffusionEnabled: false, surfaceSharpeningEnabled: false, pressureIterations: 8,
    }, () => {});
    await solver.waitForSimulationReady();
    const initial = await solver.readGPUActivityPolicy();
    assert.equal(initial.bricks.filter(b => b.active && b.spanBricks === 2 && b.acceptedResolution === 1).length, 2, JSON.stringify(initial.bricks.map(b => [b.coordinate,b.spanBricks,b.acceptedResolution,b.meanDensity])));
    const before = (await solver.readDiagnosticFields()).density.reduce((n, rho) => n + rho, 0);
    const edited = structuredClone(scene);
    edited.fluid.refinementRegions = [{ id: "left-ceiling", rule: "minimum-cell-size", minimumCellSize_cells: 1,
      maximumCellSize_cells: 2, min_m: { x: -0.8, y: 0, z: -0.4 }, max_m: { x: 0, y: 0.8, z: 0.4 } }];
    solver.applySceneUniforms(edited);
    for (let step = 1; step <= 3; step++) {
      while (!solver.advanceTo(step / 30, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
    }
    const after = await solver.readGPUActivityPolicy();
    assert.ok(after.bricks.some(b => b.active && b.coordinate[0] === 0 && b.spanBricks === 2 && b.acceptedResolution === 8));
    assert.ok(after.bricks.some(b => b.active && b.coordinate[0] === 2 && b.spanBricks === 2 && b.acceptedResolution >= 4));
    assert.equal(solver.info.topologyGenerationCount ?? 0, 0, "backed macro rerungs must stay on the in-place path");
    const fields = await solver.readDiagnosticFields();
    assert.ok(fields.density.every(rho => Number.isFinite(rho) && rho >= 0));
    assert.ok(Math.abs(fields.density.reduce((n, rho) => n + rho, 0) - before) < 1e-4);
    assert.equal(after.faultFlags, 0);
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
