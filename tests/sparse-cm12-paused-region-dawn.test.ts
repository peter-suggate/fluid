import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { readPublishedCM12Field } from "../tools/sparse-cm12-published-field";
import { cloneScene, defaultScene } from "../lib/core/model";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
for (const shape of ["flat", "curved"] as const) (dawnModule ? test : test.skip)(`${shape}: paused region edits publish latest bounds without a physics step`, { timeout: 180_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "paused-region");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device!.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const scene = cloneScene(defaultScene);
    scene.rigidBodies = []; scene.solidVoxels = [];
    scene.container = { ...scene.container, width_m: 0.8, height_m: 0.8, depth_m: 0.8, fillFraction: 0.5 };
    scene.voxelDomain.finestCellSize_m = 0.05;
    scene.fluid.initialCondition = "tank-fill";
    if (shape === "curved") scene.fluid.initialHeightField = {
      kind: "quadratic", baseHeight_m: 0.35, center_m: { x: 0, z: 0 },
      curvatureX_mInv: 0.5, curvatureZ_mInv: 0.5,
    };
    solver = await WebGPUAdaptiveMassSolver.createAsync(device!, scene, "balanced", undefined,
      { ...adaptiveMassSolverOptions({ selectorMode: "coarse-first" }), pressureIterations: 8 }, () => {});
    await solver.waitForSimulationReady();
    const volume = async () => (await solver!.readDiagnosticFields(true)).density.reduce((sum, rho) => sum + rho, 0);
    let before = await volume();
    let beforeSteps = solver.info.encodedSteps ?? 0;
    const residentEditTimes_ms: number[] = [];
    const edit = (width: number, leftHalf = false) => {
      const next = structuredClone(scene);
      next.fluid.refinementRegions = [{ id: "paused", rule: "minimum-cell-size",
        minimumCellSize_cells: width, maximumCellSize_cells: width,
        min_m: { x: -0.4, y: 0, z: -0.4 }, max_m: { x: leftHalf ? 0 : 0.4, y: 0.8, z: 0.4 } }];
      solver!.applySceneUniforms(next);
      const started = performance.now();
      return solver!.refreshSceneTopology().then(() => {
        if (beforeSteps === 0) residentEditTimes_ms.push(performance.now() - started);
      });
    };
    const check = async (width: number, leftHalf = false) => {
      const activity = await solver!.readGPUActivityPolicy();
      const inside = activity.bricks.filter(b => b.active && b.coordinate.every(q => q >= 0 && q < 2) && (!leftHalf || b.coordinate[0] === 0));
      assert.ok(inside.length > 0);
      assert.ok(inside.every(b => 8 * b.spanBricks / b.acceptedResolution === width),
        JSON.stringify({ bricks: inside.map(b => [b.coordinate, b.spanBricks, b.acceptedResolution]), deferred: solver!.info.topologyGenerationDeferred, error: solver!.info.topologyGenerationError }));
      assert.equal(activity.faultFlags, 0);
      assert.equal(activity.acceptedSteps, beforeSteps, "the GPU activity clock is unchanged");
      assert.equal(solver!.info.encodedSteps ?? 0, beforeSteps);
      assert.ok(Math.abs(await volume() - before) < 1e-3, "liquid volume survives topology transfer");
      const fields = await solver!.readDiagnosticFields(true);
      const published = await readPublishedCM12Field(device!, solver!);
      assert.ok(published.values.some(phi => Number.isFinite(phi) && phi < 0)
        && published.values.some(phi => Number.isFinite(phi) && phi > 0),
        "the actual published surface contains both sides of the liquid interface");
      assert.ok(fields.velocity.every(Number.isFinite));
      if (beforeSteps === 0) assert.ok(fields.velocity.every(v => v === 0), "gravity did not advance the paused fluid");
    };
    await edit(4); await check(4);
    const coarseSource = solver.globalFineLevelSetSource;
    await edit(2); await check(2);
    assert.equal(solver.globalFineLevelSetSource.metadata, coarseSource.metadata,
      "region edits reuse resident storage rather than rebuilding the world");
    const superseded = edit(4);
    const latest = edit(1);
    await Promise.all([superseded, latest]); await check(1);
    assert.equal(solver.info.topologyGenerationCount ?? 0, 0,
      "backed edits must not replace the resident world");
    console.log(JSON.stringify({ shape, residentEditTimes_ms }));
    // Resume normally after changing topology at t=0.
    while (!solver.advanceTo(1 / 30, [])) await new Promise(setImmediate);
    await solver.waitForTopologyReady();
    assert.equal(solver.info.encodedSteps, beforeSteps + 1);
    await solver.assertSimulationHealthy();
    beforeSteps = solver.info.encodedSteps!;
    before = await volume();
    await edit(4); await check(4);
    await edit(2); await check(2);
    await edit(1, true); await check(1, true);
    // Region edits share the topology transaction with liquid insertion.
    // Verify the insertion branch still adds mass without advancing time.
    const massBeforeInjection = await volume();
    solver.injectLiquidBall({ centre_m: { x: -0.15, y: 0.65, z: 0 }, radius_m: 0.06 });
    await solver.waitForTopologyReady();
    assert.ok(await volume() > massBeforeInjection);
    assert.equal(solver.info.encodedSteps, beforeSteps);
    await solver.assertSimulationHealthy();
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
