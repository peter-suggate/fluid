import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";

async function read(device:GPUDevice,texture:GPUTexture) {
 const row=Math.ceil(texture.width*(texture.format==='rgba32float'?16:4)/256)*256;
 const staging=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try {
  const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:staging,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
  device.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);
  return new Uint8Array(staging.getMappedRange()).slice();
 } finally {staging.unmap();staging.destroy();}
}
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("MiniDam32 pressure prefixes preserve fields and grow from asynchronous residual evidence",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","MiniDam32 pressure budget");let device:GPUDevice|undefined;
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
  const options=uniformGeometricSolverOptions({},scene);
  const adaptive=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,options,()=>{});
  const fixed=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...options,pressureCycleBudget:"fixed"},()=>{});
  // Synchronize only the dedicated asynchronous demand readback, never readStats.
  // This makes the test deterministic without supplying an alternative demand signal.
  const settle=async()=>{
   await device!.queue.onSubmittedWorkDone();
   const deadline=performance.now()+5000;
   while((adaptive as any).pressureCycleDemandPending){
    assert.ok(performance.now()<deadline,"dedicated demand readback completed");
    await new Promise(resolve=>setTimeout(resolve,1));
   }
  };
  try {
   for(let frame=1;frame<=60;frame++) {
    adaptive.applyRuntimeValues({});
    assert.ok(adaptive.advanceTo(frame/30));await adaptive.awaitFrameCompletion();
    assert.ok(fixed.advanceTo(frame/30));await fixed.awaitFrameCompletion();await settle();
    assert.equal(adaptive.info.uniformPressureCyclesEncoded,1);
    assert.equal(adaptive.info.uniformPressureCyclesConverged,true);
    assert.equal(fixed.info.uniformPressureCyclesEncoded,7);
    assert.ok(adaptive.info.uniformPressurePassesEncoded!<fixed.info.uniformPressurePassesEncoded!);
    if([1,30,60].includes(frame)) for(const field of ['volumeTexture','vertexPhiTexture','velocityTexture','gridPressureTexture'] as const)
     assert.deepEqual(await read(device,adaptive[field]!),await read(device,fixed[field]!),`${field} frame ${frame}`);
   }
   // A tolerance far below the default creates measured unmet demand. It must
   // expand the encoded prefix without changing the prebuilt configured plan.
   const configured=adaptive.info.uniformPressurePassesConfigured;
   const observed:number[]=[];
   for(let frame=61;frame<=66;frame++) {
    adaptive.applyRuntimeValues({pressureResidualTolerance:1e-8});
    assert.ok(adaptive.advanceTo(frame/30));await adaptive.awaitFrameCompletion();await settle();
    observed.push(adaptive.info.uniformPressureCyclesEncoded!);
    assert.equal(adaptive.info.uniformPressurePassesConfigured,configured);
   }
   assert.equal(observed[0],1);
   assert.ok(observed.some(n=>n>1),`unmet tolerance must increase encoding: ${observed}`);
   assert.ok(observed.every(n=>n>=1&&n<=7));
   const stats=await adaptive.readStats() as unknown as Record<string,number>;
   assert.ok(Number.isFinite(stats.uniformCM11aFineResidualInfinity));
   assert.ok(stats.uniformPressureAcceptedResidual!<=stats.uniformPressureInitialResidual!);
   adaptive.applyRuntimeValues({pressureResidualTolerance:0});
   assert.ok(adaptive.advanceTo(67/30));await adaptive.awaitFrameCompletion();await settle();
   assert.equal(adaptive.info.uniformPressureCyclesEncoded,7,"disabled convergence gate preserves full schedule");
   adaptive.applyRuntimeValues({pressureCycleBudget:"fixed"});
   assert.ok(adaptive.advanceTo(68/30));await adaptive.awaitFrameCompletion();
   assert.equal(adaptive.info.uniformPressureCyclesEncoded,7,"explicit fixed mode preserves full schedule");
   console.log(JSON.stringify({tightToleranceEncodedCycles:observed,passesConfigured:configured}));
  } finally {adaptive.destroy();fixed.destroy();}
  assert.deepEqual(errors,[]);
 } finally {device?.destroy();await releaseWebGPUExclusiveLock();}
});
