import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { GPUTimestampPhase } from "../lib/core/performance-trace";
interface Access {
  liquidBalanceIndirect:boolean; conditioningScratch:GPUBuffer;
  encodeGeometricVolume(encoder:GPUCommandEncoder,seam?:(phase:GPUTimestampPhase)=>void):void;
}
const median=(xs:number[])=>[...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)]!;
await acquireWebGPUExclusiveLock("dawn-probe","uniform balancing dispatch benchmark");
let device:GPUDevice|undefined;
try {
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);assert.ok(adapter.features.has("timestamp-query"));
 device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 for(const size of [32,64]) for(const indirect of [false,true,true,false]){
  const scene=structuredClone(sceneDocument(getSceneDefinition(`minimal-power-dam-break-${size}`)));
  if(process.argv.includes("--rest")){scene.fluid.initialCondition="tank-fill";scene.fluid.initialLiquidVolumes=[];scene.container.fillFraction=0.5;}
  const solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",resolveMethodValues(uniformVolumeMethod,"balanced",{liquidCapacityBalancing:"on",velocityTransport:"semi-lagrangian"}),undefined,()=>{}) as WebGPUUniformReferenceSolver;
  const access=solver as unknown as Access;access.liquidBalanceIndirect=indirect;
  const query=device.createQuerySet({type:"timestamp",count:2});
  const resolved=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
  const readback=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const original=access.encodeGeometricVolume.bind(access);
  access.encodeGeometricVolume=(encoder,seam)=>original(encoder,phase=>{
   seam?.(phase);const index=phase.label==="Dense geometric volume coupling"?0:phase.label==="Dense liquid capacity balancing"?1:-1;
   if(index>=0){const p=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:index}});p.end();}
  });
  const samples:{frame:number,wall_ms:number,balance_ms:number,rounds:number,error:number}[]=[];
  for(let frame=1;frame<=30;frame++){
   const start=performance.now();assert.ok(solver.advanceTo(frame/30));await device.queue.onSubmittedWorkDone();const wall_ms=performance.now()-start;
   const encoder=device.createCommandEncoder();encoder.resolveQuerySet(query,0,2,resolved,0);encoder.copyBufferToBuffer(resolved,0,readback,0,16);
   encoder.copyBufferToBuffer(access.conditioningScratch,(2*size**3+5)*4,readback,16,8);device.queue.submit([encoder.finish()]);
   await readback.mapAsync(GPUMapMode.READ);const mapped=readback.getMappedRange();const times=new BigUint64Array(mapped,0,2);const words=new Uint32Array(mapped,16,2);const floats=new Float32Array(mapped,16,2);
   samples.push({frame,wall_ms,balance_ms:Number(times[1]!-times[0]!)/1e6,rounds:words[0]!,error:floats[1]!});readback.unmap();
  }
  const measured=samples.slice(3);console.log(JSON.stringify({case:process.argv.includes("--rest")?"rest":"dam",size,dispatch:indirect?"indirect":"fast-exit",medianWall_ms:median(measured.map(s=>s.wall_ms)),medianBalance_ms:median(measured.map(s=>s.balance_ms)),meanBalance_ms:measured.reduce((a,s)=>a+s.balance_ms,0)/measured.length,stats:await solver.readStats(),samples}));
  solver.destroy();query.destroy();resolved.destroy();readback.destroy();assert.deepEqual(errors,[]);
 }
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
