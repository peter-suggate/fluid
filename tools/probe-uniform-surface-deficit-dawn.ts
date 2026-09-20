/** Short MiniDam64 smoke/timing comparison; not a long-run settling claim. */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readRgbaTexture3D } from "../lib/harness/webgpu-smoke-readbacks";
await acquireWebGPUExclusiveLock("dawn-probe","MiniDam64 surface-deficit balance");
let device:GPUDevice|undefined;
try {
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 for(const mode of ["off","on"]){
  const scene=sceneDocument(getSceneDefinition("minimal-power-dam-break-64"));
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({surfaceDeficitBalancing:mode},scene),()=>{});
  try{
   const times:number[]=[];
   for(let frame=1;frame<=60;frame++){
    const start=performance.now();assert.ok(solver.advanceTo(frame/30,[]));await device.queue.onSubmittedWorkDone();
    if(frame>10)times.push(performance.now()-start);
   }
   const velocity=await readRgbaTexture3D(device,solver.velocityTexture,solver.info.nx,solver.info.ny,solver.info.nz);
   assert.ok(velocity.every(Number.isFinite));assert.deepEqual(errors,[]);
   times.sort((a,b)=>a-b);
   console.log(JSON.stringify({scene:scene.sceneId,mode,frames:60,medianWallMs:times[Math.floor(times.length/2)],p95WallMs:times[Math.floor(times.length*.95)],finiteVelocity:true}));
  }finally{solver.destroy();}
 }
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
