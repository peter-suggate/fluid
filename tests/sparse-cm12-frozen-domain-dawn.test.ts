import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from
  "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const dawnTest = modulePath ? test : test.skip;

dawnTest("fixed-domain frozen mini32 retains initial cell sizes and supports the advancing front", {
  timeout: 180_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-frozen-domain");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  const live = new Set<GPU>();
  Object.assign(globalThis, { frozenDomainGPU: live });
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => {
      event.preventDefault(); errors.push(event.error.message);
    });
    const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
    const dt = 1 / 30;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      selectorMode: "coarse-first", timeStep: "scene",
    });
    const options = adaptiveMassSolverOptions(values);
    solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
      device, scene, "balanced", undefined, options, () => {});
    await solver.waitForSimulationReady();
    const initialActivity = await solver.readGPUActivityPolicy();
    const initialFields = await solver.readDiagnosticFields(true);
    assert.equal(initialActivity.bricks.filter(b => b.active).length, 60);
    const initialCellSizes = initialActivity.bricks.map(b =>
      [b.leafId, b.coordinate, b.spanBricks, b.acceptedResolution]);
    solver.destroy(); solver = undefined;

    solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
      device, scene, "balanced", undefined, { ...options, initialAtlasResidentForQA: true }, () => {});
    await solver.waitForSimulationReady();
    solver.setTopologyFrozen(true);
    const fixedActivity = await solver.readGPUActivityPolicy();
    assert.equal(fixedActivity.bricks.filter(b => b.active).length, 64);
    assert.deepEqual(fixedActivity.bricks.map(b =>
      [b.leafId, b.coordinate, b.spanBricks, b.acceptedResolution]), initialCellSizes,
    "air residency must preserve every initial cell size, including inactive bricks");
    const fixedFields = await solver.readDiagnosticFields(true);
    assert.deepEqual(fixedFields.density, initialFields.density);
    assert.deepEqual(fixedFields.solidOpenFraction, initialFields.solidOpenFraction);
    assert.equal(solver.info.encodedSteps, 0, "full residency must not warm up the fluid");
    const mass = (rho: Float32Array) => rho.reduce((sum, value) => sum + value, 0);
    const initialMass = mass(fixedFields.density);
    let enteredDryCorner = false;
    // Ordinary frozen residency failed at step 5. The front must enter that
    // previously inactive corner without any change to membership or rungs.
    for (let step = 1; step <= 12; step++) {
      while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion?.();
      await solver.waitForTopologyReady();
      await solver.assertSimulationHealthy();
      assert.equal(solver.info.encodedSteps, step);
      const activity = await solver.readGPUActivityPolicy();
      assert.equal(activity.faultFlags, 0);
      assert.equal(activity.commitFailed, false);
      assert.equal(activity.bricks.filter(b => b.active).length, 64);
      assert.deepEqual(activity.bricks.map(b =>
        [b.leafId, b.coordinate, b.spanBricks, b.acceptedResolution]), initialCellSizes);
      const fields = await solver.readDiagnosticFields(true);
      assert.ok(fields.density.every(Number.isFinite));
      assert.ok(fields.velocity.every(Number.isFinite));
      assert.ok(Math.abs(mass(fields.density) / initialMass - 1) < 0.0001);
      for (let z = 24; z < 32; z++) for (let y = 0; y < 32; y++) {
        for (let x = 24; x < 32; x++) {
          enteredDryCorner ||= fields.density[x + 32 * (y + 32 * z)]! > 0.001;
        }
      }
    }
    assert.ok(enteredDryCorner, "nonzero liquid must actually use the added air support");
    assert.deepEqual(errors, []);
  } finally {
    solver?.destroy(); device?.destroy(); live.clear();
    await releaseWebGPUExclusiveLock();
  }
});
