import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("fluid torus begins hollow and falls onto the floor", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "falling-water-torus");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device!.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const definition = getSceneDefinition("falling-water-torus");
    const scene = sceneDocument(definition);
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", definition.methodProfile?.overrides ?? {});
    solver = await adaptiveMassMethod.createSolverAsync!(device!, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    console.log("torus solver ready");
    const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
    const summarize = (density: Float32Array) => {
      let mass = 0, momentY = 0, floorMass = 0;
      for (let i = 0; i < density.length; i++) {
        assert.ok(Number.isFinite(density[i]));
        const rho = Math.max(0, density[i]!); const y = Math.floor(i / nx) % ny;
        mass += rho; momentY += rho * (y + .5) * .05;
        if (y < 4) floorMass += rho;
      }
      return { mass, centreY: momentY / mass, floorMass };
    };
    const initial = await solver.readDiagnosticFields();
    const worldMass = async () => (await solver!.readGPUActivityPolicy()).bricks
      .filter(brick => brick.active)
      .reduce((sum, brick) => sum + Math.max(0, brick.meanDensity) * (8 * brick.spanBricks) ** 3, 0);
    const before = summarize(initial.density);
    // All authored liquid starts inside the tank. The activity census has
    // not run at reset, so the initialized diagnostic field is the mass oracle.
    const beforeWorldMass = before.mass;
    assert.ok(before.mass > 0);
    for (let y = 0; y < ny; y++) {
      assert.equal(initial.density[Math.floor(nx/2) + nx * (y + ny * Math.floor(nz/2))], 0,
        "the torus hole must begin empty");
    }
    for (let step = 1; step <= 48; step++) {
      while (!solver.advanceTo(step / 60, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      if (step % 12 === 0) console.log(`torus step ${step}`);
    }
    const after = summarize((await solver.readDiagnosticFields()).density);
    const afterWorldMass = await worldMass();
    console.log(JSON.stringify({ before, after, beforeWorldMass, afterWorldMass }));
    assert.ok(after.centreY < before.centreY - .5, "the torus must fall");
    assert.ok(after.floorMass > .1 * before.mass, "liquid must reach the floor");
    // Diagnostic fields cover the authored tank only. Sparse transport may
    // move water outside it; conservation must count all active world leaves.
    assert.ok(afterWorldMass > .9 * beforeWorldMass, "impact must retain the liquid");
    assert.deepEqual(errors, []);
  } finally {
    solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  }
});
