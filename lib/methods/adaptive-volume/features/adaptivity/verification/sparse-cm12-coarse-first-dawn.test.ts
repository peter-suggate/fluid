import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { getSceneDefinition } from "../../../../../core/scenes";
import { sceneAtContainerExtents } from "../../../../../core/scene-scale";
import { sceneDocument } from "../../../../../core/scene-definition";
import { requiredFluidDeviceLimits } from "../../../../../core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../../../../../harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../../../method";
import { resolveMethodValues } from "../../../../../core/method-contract";
import type { WebGPUAdaptiveMassSolver } from "../../../webgpu-adaptive-mass-solver";
const dawnModule = process.env.WEBGPU_NODE_MODULE;
for (const scenario of ["still", "impact", "settling"] as const) (dawnModule ? test : test.skip)(scenario === "settling"
  ? "coarse-first mildly disturbed surface returns to coarse while moving"
  : scenario === "impact"
  ? "coarse-first pool requires local evidence while preserving far still water"
  : "coarse-first hydrostatic pool stays at B1", { timeout: 240_000 }, async () => {
  const impact = scenario === "impact", settling = scenario === "settling";
  await acquireWebGPUExclusiveLock("dawn-test", "coarse-first-pool-impact");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device!.addEventListener("uncapturederror", (event) => { event.preventDefault(); errors.push(event.error.message); });
    let scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact"));
    if (settling) {
      scene = sceneAtContainerExtents(scene, { width_m: 1.6, height_m: 1.6, depth_m: 1.6 });
      scene.container.fillFraction = 0.5;
      scene.voxelDomain.finestCellSize_m = 0.05;
      scene.fluid.initialLiquidVolumes = [
        { shape: "sphere", center_m: { x: 0, y: 0.84, z: 0 }, radius_m: 0.12 },
      ];
    }
    if (!impact && !settling) delete scene.fluid.initialLiquidVolumes;
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      selectorMode: "coarse-first", timeStep: "scene",
    });
    solver = await adaptiveMassMethod.createSolverAsync!(device!, scene, "balanced", values,
      undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const surfaceY = settling ? 1 : 3;
    const centerXZ = settling ? 1.5 : 7.5;
    const initial = await solver.readGPUActivityPolicy();
    const pool = initial.bricks.filter(b => b.active && b.coordinate[1] === surfaceY);
    if (!settling) assert.ok(pool.filter(b => b.acceptedResolution === 1).length > 100);
    const initialFine = pool.filter(b => b.acceptedResolution >= 4).length;
    let settledCoarse = false;
    const initialFields = await solver.readDiagnosticFields();
    const mass = (density: Float32Array) => density.reduce((a, b) => a + b, 0);
    const initialMass = mass(initialFields.density);
    let refinedBeforeContact = false;
    let minimumFarCoarse = Infinity;
    const trace: unknown[] = [];
    for (let step = 1; step <= (settling ? 150 : impact ? 75 : 18); step++) {
      while (!solver.advanceTo(step / 60, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion?.();
      await solver.waitForTopologyReady();
      if (impact && step % 15 === 0) console.log(JSON.stringify({ step, generations: solver.info.topologyGenerationCount, preparationMs: solver.info.topologyPreparationDurationMs, error: solver.info.topologyGenerationError }));
      if (step % 3 === 0) {
        const snapshot = await solver.readGPUActivityPolicy();
        const surface = snapshot.bricks.filter(b => b.active && b.coordinate[1] === surfaceY);
        if (settling && step >= 60) {
          const coarse = surface.filter(b => b.acceptedResolution <= 2).length;
          const moving = surface.some(b => b.maximumVelocityTravelFineCells > 1e-4);
          settledCoarse ||= coarse >= surface.length * 0.75 && moving;
        }
        const center = surface.filter(b => Math.abs(b.coordinate[0] - centerXZ) < 1
          && Math.abs(b.coordinate[2] - centerXZ) < 1);
        const far = surface.filter(b => Math.abs(b.coordinate[0] - centerXZ) > 4);
        // Ball bottom starts 1.8 m above the pool. Ballistic contact ~0.61 s.
        if (step <= 27 && center.some(b => b.acceptedResolution === 8)) refinedBeforeContact = true;
        if (step <= 27) minimumFarCoarse = Math.min(minimumFarCoarse,
          far.filter(b => b.acceptedResolution === 1).length / Math.max(1, far.length));
        trace.push({ step, center: center.map(b => [b.acceptedResolution, b.plannedResolution,
          b.reasons, b.maximumVelocityTravelFineCells]), farB1: far.filter(b => b.acceptedResolution === 1).length, far: far.slice(0, 2).map(b => [b.coordinate, b.acceptedResolution, b.plannedResolution, b.reasons, b.planReasons, b.maximumVelocityTravelFineCells]) });
      }
    }
    if (!impact && !settling) {
      solver.applyRuntimeValues({ ...values, energyThreshold: 3, curvatureTolerance: 0.2 });
      assert.equal(solver.advanceTo(18 / 60, []), false, "live controls must preserve simulation time");
      while (!solver.advanceTo(19 / 60, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion?.();
    }
    const fields = await solver.readDiagnosticFields();
    console.log(JSON.stringify({ scenario, initialFine, settledCoarse, initialCells: initial.bricks.filter(b => b.active).reduce((n, b) => n + b.acceptedResolution ** 3, 0), trace, relativeMassError: Math.abs(mass(fields.density) / initialMass - 1) }));
    assert.deepEqual(errors, []);
    if (impact) assert.equal(refinedBeforeContact, false,
      "remote impact prediction refined a calm pool without local evidence: " + JSON.stringify(trace));
    if (settling) {
      assert.ok(initialFine > 0, "curved disturbance must start fine");
      assert.ok(settledCoarse, "moving surface did not recover B1/B2 coverage");
    }
    if (!settling) assert.ok(minimumFarCoarse > 0.8, `far pool lost coarse topology: ${minimumFarCoarse}`);
    assert.ok(fields.density.every(Number.isFinite));
    assert.ok(Math.abs(mass(fields.density) / initialMass - 1) < (impact || settling ? 0.005 : 1e-5));
  } finally {
    solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  }
});

(dawnModule ? test : test.skip)("half-pool impact lookahead retains B1 without local evidence on step three",
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "coarse-first-half-pool-evidence");
    let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-half"));
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        selectorMode: "coarse-first", timeStep: "paper",
      });
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values,
        undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      for (let step = 1; step <= 3; step++) {
        while (!solver.advanceTo(step / 30, [])) await new Promise(setImmediate);
        await solver.awaitFrameCompletion?.();
        await solver.waitForTopologyReady();
      }
      const activity = await solver.readGPUActivityPolicy();
      const centerSurface = activity.bricks.filter(brick => brick.active
        && brick.coordinate[1] === 1
        && Math.abs(brick.coordinate[0] - 3.5) < 1
        && Math.abs(brick.coordinate[2] - 3.5) < 1);
      assert.ok(centerSurface.length > 0);
      assert.ok(centerSurface.every(brick => brick.acceptedResolution === 1),
        "remote motion alone must not refine the calm pool surface: "
          + JSON.stringify(centerSurface));
      const approachingSurface = activity.bricks.filter(brick => brick.active
        && brick.coordinate[1] >= 3 && (brick.reasons & 1) !== 0
        && brick.densityMoments[1] < -16);
      assert.ok(approachingSurface.length > 0,
        "the fixture must carry a surface moving more than one brick over the lookahead horizon");
    } finally {
      solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
