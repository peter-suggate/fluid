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
interface Access {
  pressureMultigrid:{diagnostics:GPUBuffer;encode(encoder:GPUCommandEncoder,group:GPUBindGroup,boundary?:unknown):void};
}
const median=(xs:number[])=>[...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)]!;
await acquireWebGPUExclusiveLock("dawn-probe","uniform pressure tolerance benchmark");
let device:GPUDevice|undefined;
try {
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);assert.ok(adapter.features.has("timestamp-query"));
 device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const marker=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:"@compute @workgroup_size(1) fn marker(){}"}),entryPoint:"marker"}});
 const requested=Number(process.env.PRESSURE_TOLERANCE??0.0001);
 for(const size of [64]) for(const tolerance of [0,requested,requested,0]){
  const scene=structuredClone(sceneDocument(getSceneDefinition(`minimal-power-dam-break-${size}`)));
  if(process.argv.includes("--rest")){scene.fluid.initialCondition="tank-fill";scene.fluid.initialLiquidVolumes=[];scene.container.fillFraction=0.5;}
  const solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",resolveMethodValues(uniformVolumeMethod,"balanced",{liquidCapacityBalancing:"off",velocityTransport:"semi-lagrangian",pressureResidualTolerance:tolerance}),undefined,()=>{}) as WebGPUUniformReferenceSolver;
  const access=solver as unknown as Access;
  const query=device.createQuerySet({type:"timestamp",count:2});
  const resolved=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
  const readback=device.createBuffer({size:48,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const original=access.pressureMultigrid.encode.bind(access.pressureMultigrid);
  access.pressureMultigrid.encode=(encoder,group,boundary)=>{
   const before=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:0}});before.setPipeline(marker);before.dispatchWorkgroups(1);before.end();
   original(encoder,group,boundary);
   const after=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:1}});after.setPipeline(marker);after.dispatchWorkgroups(1);after.end();
  };
  const samples:{frame:number,wall_ms:number,pressure_ms:number,fullCycles:number,vCycles:number,residual:number}[]=[];
  for(let frame=1;frame<=12;frame++){
   const start=performance.now();assert.ok(solver.advanceTo(frame/30));await device.queue.onSubmittedWorkDone();const wall_ms=performance.now()-start;
   const encoder=device.createCommandEncoder();encoder.resolveQuerySet(query,0,2,resolved,0);encoder.copyBufferToBuffer(resolved,0,readback,0,16);
   encoder.copyBufferToBuffer(access.pressureMultigrid.diagnostics,64,readback,16,12);encoder.copyBufferToBuffer(access.pressureMultigrid.diagnostics,40,readback,28,4);device.queue.submit([encoder.finish()]);
   await readback.mapAsync(GPUMapMode.READ);const mapped=readback.getMappedRange();const times=new BigUint64Array(mapped,0,2);const words=new Uint32Array(mapped,16,3);const floats=new Float32Array(mapped,28,1);
   samples.push({frame,wall_ms,pressure_ms:Number(times[1]!-times[0]!)/1e6,fullCycles:words[1]!,vCycles:words[2]!,residual:floats[0]!});readback.unmap();
  }
  const measured=samples.slice(3);console.log(JSON.stringify({case:process.argv.includes("--rest")?"rest":"dam",size,tolerance,medianWall_ms:median(measured.map(s=>s.wall_ms)),medianPressure_ms:median(measured.map(s=>s.pressure_ms)),meanPressure_ms:measured.reduce((a,s)=>a+s.pressure_ms,0)/measured.length,stats:await solver.readStats(),samples}));
  solver.destroy();query.destroy();resolved.destroy();readback.destroy();assert.deepEqual(errors,[]);
 }
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
