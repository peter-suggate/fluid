import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { captureNativeUniformVexMaps } from "../tools/implicit-density/native-uniform-vex-capture";

const modulePath = process.env.WEBGPU_NODE_MODULE, live = new Set<GPU>();
(modulePath ? test : test.skip)("actual full-fine imposed-flow VEX supplies covered GPU maps before gather", { timeout: 240_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "actual-native-uniform-vex-capture");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = []; device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const receipts = await captureNativeUniformVexMaps(device,
      process.env.FLUID_NATIVE_UNIFORM_VEX_OUT ?? "artifacts/retained-imposed-flow/native-uniform-vex-map");
    assert.equal(receipts.length, 2); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
  } finally { device?.destroy(); if (gpu) live.delete(gpu); await releaseWebGPUExclusiveLock(); }
});
