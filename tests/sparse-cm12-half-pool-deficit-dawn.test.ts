import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
import type { SceneDescription } from "../lib/core/model";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const modulePath = process.env.WEBGPU_NODE_MODULE;
for (const [variant, fixture] of [
  ["adaptive", "sparse-cm12-half-pool-deficit.json"],
  ["forced finest", "sparse-cm12-half-pool-fine-negative-density.json"],
] as const) {
(modulePath ? test : test.skip)(`half pool ${variant} retains nonnegative mass and donor support for four seconds`, { timeout: 180_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "half-pool-deficit-support");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  const live = new Set<GPU>(); Object.assign(globalThis, { halfPoolDeficitTestGPU: live });
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
      configuration.method.overrides["adaptive-mass"]);
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const initial = await solver.readDiagnosticFields(true);
    const mass = (rho: Float32Array) => rho.reduce((sum, value) => sum + value, 0);
    const initialMass = mass(initial.density);
    // The receipts stopped at frames 42 and 43: missing forward support,
    // then stale gamma scratch consumed as density under forced finest cells.
    // Cover the whole authored duration, including residue after impact.
    const steps = Math.round(scene.duration_s / scene.numerics.fixedDt_s);
    assert.equal(steps, 120);
    for (let step = 1; step <= steps; step++) {
      while (!solver.advanceTo(step * scene.numerics.fixedDt_s, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady(); await solver.assertSimulationHealthy();
      assert.equal(solver.info.encodedSteps, step);
      const activity = await solver.readGPUActivityPolicy();
      assert.equal(activity.faultFlags, 0); assert.equal(activity.commitFailed, false);
      if (variant === "forced finest") {
        const active = activity.bricks.filter(b => b.active);
        assert.ok(active.every(b => b.acceptedResolution === 8));
        const indirect = await solver.readAcceptedIndirectQA();
        assert.equal(indirect[0], active.length * 8,
          `accepted cell worklist must contain every active B8 page at step ${step}`);
      }

      if (step === 42) {
        assert.ok(activity.bricks.some(b => b.active
          && b.coordinate[0] === 3 && b.coordinate[1] === 6 && b.coordinate[2] === 3),
        "the retired upper receiver must be allocated again before frame 42 transports mass into it");
      }
      const fields = await solver.readDiagnosticFields(true);
      assert.ok(fields.density.every(value => Number.isFinite(value) && value >= 0),
        `density must remain nonnegative at step ${step}`); assert.ok(fields.velocity.every(Number.isFinite));
      assert.ok(Math.abs(mass(fields.density) / initialMass - 1) < 0.001,
        `transport must retain mass at step ${step}`);
    }
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});
}
