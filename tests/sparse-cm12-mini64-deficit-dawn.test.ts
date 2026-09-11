import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import type { SceneDescription } from "../lib/core/model";
import { pathToFileURL } from "node:url";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const modulePath = process.env.WEBGPU_NODE_MODULE;
for (const uiCadence of [false, true]) {
(modulePath ? test : test.skip)(`mini64 ${uiCadence ? "UI default cadence" : "colliding frontier requests"} retains donor support`, { timeout: 180_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "mini64-deficit-support");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  const live = new Set<GPU>(); Object.assign(globalThis, { mini64DeficitTestGPU: live });
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const configuration = JSON.parse(readFileSync(new URL("./fixtures/sparse-cm12-mini64-ui-deficit.json", import.meta.url), "utf8"));
    const scene = uiCadence ? configuration.scene as SceneDescription : createMinimalPowerDamBreak64Scene();
    if (!uiCadence) scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", uiCadence ? configuration.method.overrides["adaptive-volume"] : {});
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const initial = await solver.readDiagnosticFields(true);
    const mass = (rho: Float32Array) => rho.reduce((sum, value) => sum + value, 0);
    const initialMass = mass(initial.density);
    // In the supplied failure the absent page (7,0,5) collided with (6,1,4)
    // while allocating step 5 support. The next step halted with a completely
    // unsupported forward deficit. Demand must be fulfilled in that epoch.
    for (let step = 1; step <= 12; step++) {
      while (!solver.advanceTo(step * scene.numerics.fixedDt_s, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion?.();
      if (!uiCadence) await solver.waitForTopologyReady();
      await solver.assertSimulationHealthy();
      assert.equal(solver.info.encodedSteps, step);
      // Extra topology waits or per-frame field readbacks hide the UI admission race.
      if (uiCadence && step < 12) continue;
      const activity = await solver.readGPUActivityPolicy();
      assert.equal(activity.faultFlags, 0); assert.equal(activity.commitFailed, false);
      if (step === 5) for (const coordinate of [[7,0,5], [6,1,4]]) {
        assert.ok(activity.bricks.some(b => b.active && coordinate.every((q, axis) => q >= b.coordinate[axis]! && q < b.coordinate[axis]! + b.spanBricks)),
          `colliding requested coordinate ${coordinate} must already be resident`);
      }
      const fields = await solver.readDiagnosticFields(true);
      assert.ok(fields.density.every(value => Number.isFinite(value) && value >= 0));
      assert.ok(fields.velocity.every(Number.isFinite));
      assert.ok(Math.abs(mass(fields.density) / initialMass - 1) < 0.001,
        `allocation must retain transported mass at step ${step}`);
    }
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});

}
