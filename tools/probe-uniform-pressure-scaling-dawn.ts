/** Diagnostic pressure kernel timings. Run with FLUID_UNIFORM_MG_LEVEL_LABELS=1.
 * This disables production pass batching to attribute each dispatch; compare
 * stage totals with profile-uniform-geometric-dawn.ts before drawing conclusions.
 * --scene=<id> --steps=30 --out=/tmp/pressure-kernels --values='<JSON>'
 * --coarse-tolerance=0.1 changes only the inner tolerance (diagnostic, not a fix).
 */
import assert from "node:assert/strict";
import { mkdir,writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU,type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const passFilter=new RegExp(arg("passes","^Uniform CM11a"));
const out=arg("out","/tmp/pressure-kernels"),steps=Number(arg("steps","120"));
assert.equal(process.env.FLUID_UNIFORM_MG_LEVEL_LABELS,"1","Set FLUID_UNIFORM_MG_LEVEL_LABELS=1 for per-dispatch attribution");
assert.ok(Number.isInteger(steps)&&steps>4);
await acquireWebGPUExclusiveLock("dawn-probe","uniform pressure scaling");
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
try {
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal", "disable-dawn-features=timestamp_quantization"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);assert.ok(adapter.features.has("timestamp-query"));
 const base=managedGPUDevice(await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 let active=false;let labels:string[]=[];
 const queries=base.createQuerySet({type:"timestamp",count:4096});
 const result=base.createBuffer({size:32768,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
 const read=base.createBuffer({size:32768,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 device=new Proxy(base,{get(target,key){if(key==="createCommandEncoder")return (descriptor?:GPUCommandEncoderDescriptor)=>{
  const encoder=target.createCommandEncoder(descriptor);let sampled=false;
  return new Proxy(encoder,{get(enc,field){
   if(field==="beginComputePass")return (desc?:GPUComputePassDescriptor)=>{
    const label=desc?.label??"";
    if(active&&passFilter.test(label)){
     const index=labels.length*2;assert.ok(index+1<4096);labels.push(label);sampled=true;
     return enc.beginComputePass({...desc,timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
    }
    return enc.beginComputePass(desc);
   };
   if(field==="finish")return ()=>{if(sampled){enc.resolveQuerySet(queries,0,labels.length*2,result,0);enc.copyBufferToBuffer(result,0,read,0,labels.length*16);}return enc.finish();};
   const value=Reflect.get(enc,field);return typeof value==="function"?value.bind(enc):value;
  }});
 };const value=Reflect.get(target,key);return typeof value==="function"?value.bind(target):value;}});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 // Diagnostic only: alter the inner solve tolerance, retaining the production
 // fine-grid acceptance, rejection, recovery and lagged budget mechanisms.
 const coarseToleranceArg=arg("coarse-tolerance","auto");
 const coarseTolerance=coarseToleranceArg === "auto" ? undefined : Number(coarseToleranceArg);
 assert.ok(coarseTolerance === undefined || (Number.isFinite(coarseTolerance)&&coarseTolerance>0));
 const compiler=gpuCompilationManagerFor(device),originalModule=compiler.createShaderModule;
 let replaced=0;
 compiler.createShaderModule=function(descriptor){
  const needle="bitcast<f32>(atomicLoad(&mgCoarseResidualBits))<=mgCoarseTarget";
  if(coarseTolerance !== undefined && descriptor.code.includes(needle)){
   replaced++;
   descriptor={...descriptor,code:descriptor.code.replaceAll(needle,
    `bitcast<f32>(atomicLoad(&mgCoarseResidualBits))<=${coarseTolerance}`)};
  }
  return originalModule.call(this,descriptor);
 };
 const scene=sceneDocument(getSceneDefinition(arg("scene","cm12-figure-9")));const values=resolveMethodValues(uniformVolumeMethod,"balanced",JSON.parse(arg("values","{}")));
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 compiler.createShaderModule=originalModule;assert.ok(coarseTolerance === undefined || replaced>0,"coarse tolerance shader hook did not match");
 await mkdir(out,{recursive:true});await writeFile(`${out}/config.json`,JSON.stringify({capturedAt:new Date().toISOString(),adapter:{vendor:adapter.info.vendor,device:adapter.info.device,description:adapter.info.description},coarseTolerance,values,scene,lattice:{nx:solver.info.nx,ny:solver.info.ny,nz:solver.info.nz},scope:"Per-dispatch timestamp instrumentation with production batching disabled; rendering excluded."},null,2));
 const rows:unknown[]=[];
 for(let frame=1;frame<=steps;frame++){
  labels=[];active=true;const start=performance.now();assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();active=false;await device.queue.onSubmittedWorkDone();const wallMs=performance.now()-start;
  await read.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(read.getMappedRange());const passes=labels.map((label,i)=>({label,ms:Number(times[2*i+1]!-times[2*i]!)/1e6}));read.unmap();
  const info=await solver.readStats();const work=Object.fromEntries(Object.entries(info).filter(([k])=>/uniformPressure|uniformCM11a|volumeCellSum|maxSpeed|encodedSteps/i.test(k)));
  const row={frame,t:frame/30,wallMs,passes,work};rows.push(row);console.log(JSON.stringify({frame,wallMs,pressureMs:passes.reduce((sum,p)=>sum+p.ms,0),work}));await writeFile(`${out}/trace.json`,JSON.stringify(rows,null,2));assert.deepEqual(errors,[]);
 }
 queries.destroy();result.destroy();read.destroy();
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
