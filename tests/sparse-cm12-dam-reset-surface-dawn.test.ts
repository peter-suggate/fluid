import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { sceneDamBreakBox } from "../lib/core/initial-fluid";
import { resolveMethodValues } from "../lib/core/method-contract";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { readPublishedCM12Field } from "../tools/sparse-cm12-published-field";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const nativeInstances = new Set<GPU>();
(modulePath ? test : test.skip)("coarse-first reset publishes a flat dam top and straight sides from accepted density", { timeout: 180_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "coarse-first dam reset surface");
  let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    nativeInstances.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const scene = createMinimalPowerDamBreak32Scene();
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {});
    assert.equal(values.selectorMode, "coarse-first");
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const { values: phi } = await readPublishedCM12Field(device, solver);
    const { nx, ny, nz } = solver.info;
    const box = sceneDamBreakBox(scene);
    const top = box.max.y * ny, sideX = box.max.x * nx, sideZ = box.max.z * nz;
    const at = (x: number, y: number, z: number) => phi[x + nx * (y + ny * z)]!;
    const crossing = (lo: number, hi: number, coordinate: number) => coordinate + .5 - lo / (hi - lo);
    let topSamples = 0, sideSamples = 0, maximumTopError = 0, maximumSideError = 0;
    // Stay two finest cells inside the authored corners; their deliberate
    // interpolation rounding is not a vertical-wall shape error.
    for (let z = 2; z < sideZ - 2; z++) for (let x = 2; x < sideX - 2; x++) {
      let found = false;
      for (let y = 0; y < ny - 1; y++) {
        const lo = at(x, y, z), hi = at(x, y + 1, z);
        if (!(lo < 0 && hi >= 0)) continue;
        maximumTopError = Math.max(maximumTopError, Math.abs(crossing(lo, hi, y) - top));
        topSamples++; found = true;
      }
      assert.ok(found, `missing dam top at ${x},${z}`);
    }
    for (let y = 2; y < top - 3; y++) for (let z = 2; z < sideZ - 3; z++) {
      let found = false;
      for (let x = 0; x < nx - 1; x++) {
        const lo = at(x, y, z), hi = at(x + 1, y, z);
        if (!(lo < 0 && hi >= 0)) continue;
        maximumSideError = Math.max(maximumSideError, Math.abs(crossing(lo, hi, x) - sideX));
        sideSamples++; found = true;
      }
      assert.ok(found, `missing dam side at ${y},${z}`);
    }
    assert.ok(topSamples > 100 && sideSamples > 100);
    assert.ok(maximumTopError < .002, `top displaced ${maximumTopError} finest cells`);
    assert.ok(maximumSideError < .25, `vertical side displaced ${maximumSideError} finest cells`);
    assert.equal(solver.info.encodedSteps, 0, "reset geometry must require no physics advance");
    const healthEncoder = device.createCommandEncoder();
    const readHealth = solver.captureSimulationHealth(healthEncoder);
    device.queue.submit([healthEncoder.finish()]);
    await readHealth();
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ topSamples, sideSamples, maximumTopError, maximumSideError }));
  } finally {
    solver?.destroy(); device?.destroy(); if (gpu) nativeInstances.delete(gpu);
    await releaseWebGPUExclusiveLock();
  }
});
