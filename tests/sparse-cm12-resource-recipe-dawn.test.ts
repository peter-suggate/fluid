import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createSolidWorld, SolidWorldDirectory } from "../lib/core/solid-world";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { realizeCM12ResourceRecipe } from "../lib/methods/adaptive-volume/sparse-cm12-resource-recipe";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;

test("a worker-prepared resident hydrates auto pipeline layouts on Dawn", {
  skip: !dawnModule && "set WEBGPU_NODE_MODULE for Dawn recipe hydration",
  timeout: 120_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-resource-recipe");
  let gpu: GPU | undefined;
  let device: GPUDevice | undefined;
  let resident: WebGPUSparseCM12Resident | undefined;
  let realized: Awaited<ReturnType<typeof realizeCM12ResourceRecipe>> | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
    const atlas=createSparseAdaptiveMassAtlas([8,8,8],[{
      key:0,coordinate:[0,0,0],resolution:1,
      density:new Float64Array([1]),gamma:new Float64Array([1]),
    }],0,8);
    const solidWorld=createSolidWorld([{operation:"fill",minimum:[0,0,0],maximumExclusive:[1,1,1]}]);
    Object.setPrototypeOf(solidWorld.directory,SolidWorldDirectory.prototype);
    const recipe=structuredClone(await WebGPUSparseCM12Resident.recordPreparedGeneration({
      atlas,active:new Set([0]),finestCellSize_m:.05,solidWorld,
      maximumBytes:64*1024*1024,topologyPageCapacityMaximum:0,
      symmetry:{scalar:false,face:false},limits:adapter.limits,
    }));
    assert.ok(recipe.operations.some(operation=>operation.method==="getBindGroupLayout"));
    realized=await realizeCM12ResourceRecipe(device,recipe);
    resident=Object.assign(Object.create(WebGPUSparseCM12Resident.prototype),
      (realized.state as {resident:object}).resident,{device,currentSolidWorld:solidWorld,
        simulationCompilationSnapshot:()=>({state:"idle",generation:0,queued:0,active:0,cached:0})}) as WebGPUSparseCM12Resident;
    const encoder=device.createCommandEncoder();
    resident.encodeInitialPresentation(encoder,.05);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await resident.assertSimulationHealthy();
  } finally {
    resident?.destroy();
    realized?.destroy();
    device?.destroy();
    (gpu as unknown as {destroy?():void})?.destroy?.();
    await releaseWebGPUExclusiveLock();
  }
});
