import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sparseCM12DawnDefaultOptions, sparseCM12DawnDefaultValues } from "../lib/harness/sparse-cm12-dawn-defaults";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("momentum snapshot invalidates on liquid edits and runtime off/on preserves simulation and allocation",{timeout:90_000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","immutable momentum lifecycle");let device:GPUDevice|undefined,solver:WebGPUAdaptiveMassSolver|undefined;
  try{
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});assert.ok(device);
    const errors:string[]=[];device.addEventListener("uncapturederror",event=>{event.preventDefault();errors.push(event.error.message);});
    const scene=getScenePreset("coarse-first-pool-impact-half").create();const dt=1/30;
    solver=await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(device,scene,"balanced",undefined,sparseCM12DawnDefaultOptions(),()=>{});
    await solver.waitForSimulationReady();
    const step=async(n:number)=>{while(!solver!.advanceTo(n*dt,[]))await new Promise<void>(setImmediate);
      await solver!.awaitFrameCompletion();await solver!.waitForTopologyReady();};
    await step(1);await step(2);
    const initial=await solver.readAirExtensionReceiptQA();assert.ok(initial.momentumSnapshotReady);assert.ok(initial.momentumSnapshotCells>0);
    solver.injectLiquidBall({centre_m:{x:0,y:.4,z:0},radius_m:.12});
    await solver.waitForTopologyReady();await device.queue.onSubmittedWorkDone();
    assert.equal((await solver.readAirExtensionReceiptQA()).momentumSnapshotReady,false,"authored liquid edit invalidates previous momentum");
    solver.applyRuntimeValues({...sparseCM12DawnDefaultValues(),airExtension:"off"});await step(3);
    const off=await solver.readAirExtensionReceiptQA();assert.equal(off.enabled,false);assert.equal(off.encodedDispatches,0);
    assert.equal(off.allocatedBytes,initial.allocatedBytes,"switching off retains reusable storage");
    solver.applyRuntimeValues({...sparseCM12DawnDefaultValues(),airExtension:"on"});await step(4);
    const on=await solver.readAirExtensionReceiptQA();assert.ok(on.momentumSnapshotReady);assert.equal(on.allocatedBytes,initial.allocatedBytes);
    assert.ok(on.ready&&on.converged);assert.deepEqual(errors,[]);
  }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
