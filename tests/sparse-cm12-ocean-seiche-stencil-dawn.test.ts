// Run with WEBGPU_NODE_MODULE and node --max-old-space-size=12288.
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
import type { SceneDescription } from "../lib/core/model";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const fixture = "sparse-cm12-ocean-seiche-ui-stencil.json";
(modulePath ? test : test.skip)("ocean-seiche defaults retain geometric transport support through the reported frame-35 halt", { timeout: 600_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "ocean-seiche-stencil");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  const live = new Set<GPU>(); Object.assign(globalThis, { oceanSeicheTestGPU: live });
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const configuration = JSON.parse(readFileSync(new URL(
      `./fixtures/${fixture}`, import.meta.url), "utf8"));
    const scene = configuration.scene as SceneDescription;
    const values = resolveMethodValues(adaptiveMassMethod, configuration.method.quality,
      configuration.method.overrides["adaptive-volume"]);
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const initial = await solver.readDiagnosticFields(true);
    const mass = (rho: Float32Array) => rho.reduce((sum, value) => sum + value, 0);
    const initialMass = mass(initial.density);
    // The receipt is zero-based: frame 35 is the 36th encoded step (1.2 s).
    // A full-duration trajectory can be requested separately for investigation.
    const steps = Number(process.env.FLUID_OCEAN_STEPS ?? 36);
    assert.ok(Number.isSafeInteger(steps) && steps >= 36);
    for (let step = 1; step <= steps; step++) {
      while (!solver.advanceTo(step * scene.numerics.fixedDt_s, [])) await new Promise(setImmediate);
      if (step < steps) await solver.waitForTopologyReady();
      await solver.assertSimulationHealthy();
      assert.equal(solver.info.encodedSteps, step);
      if (step % 30 !== 0 && step !== steps) continue;
      console.log(`ocean-seiche: ${step}/${steps} frames healthy`);
      const fields = await solver.readDiagnosticFields(true);
      assert.ok(fields.density.every(value => Number.isFinite(value) && value >= 0),
        `density must remain nonnegative at step ${step}`); assert.ok(fields.velocity.every(Number.isFinite));
      assert.ok(Math.abs(mass(fields.density) / initialMass - 1) < 0.001,
        `transport must retain mass at step ${step}`);
    }
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});
