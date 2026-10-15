/** Same-state surface-splat parity and timestamp probe. Replays both kernels
 * at the actual reconstruction dispatch, restoring its distance plane each time.
 * node --import tsx tools/probe-nbflip-surface-splat.ts [--frames=1,120] [--repeats=7]
 * Browser GPU must be off. Times exclude simulation and readback. */
import { uniformMixedCertifiedEntriesWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const frames=(option("frames")??"1,120").split(",").map(Number).sort((a,b)=>a-b);
const repeats=Number(option("repeats")??7);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
let frame=0, solver:WebGPUUniformReferenceSolver|undefined;
type Sample={saved:GPUBuffer;results:GPUBuffer;queries:GPUQuerySet;resolved:GPUBuffer;times:GPUBuffer;size:number};
let sample:Sample|undefined;
const intercepted=new Proxy(raw,{get:(target,key)=>{
 if(key==="createShaderModule")return (d:GPUShaderModuleDescriptor)=>{
  // Detail-module rewriting turns scene-dependent constants into functions.
  // This interception runs after that rewrite, so use its emitted spelling.
  const band=/fn\s+NB_BAND\s*\(/.test(d.code)?"NB_BAND()":"NB_BAND";
  const code=d.code.includes("fn surfaceSplatReference(")?uniformMixedCertifiedEntriesWGSL(d.code,["surfaceSplatReference"],`atomicLoad(&bins[${band}+3u])`):d.code;
  return target.createShaderModule({...d,code});
 };
 if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
  const p=await target.createComputePipelineAsync(d);
  if(d.compute.entryPoint==="surfaceSplat")references.set(p,await target.createComputePipelineAsync({...d,label:"Surface splat scalar reference",compute:{...d.compute,entryPoint:"surfaceSplatReference"}}));
  return p;
 };
 if(key==="createCommandEncoder")return (d?:GPUCommandEncoderDescriptor)=>{
  const e=target.createCommandEncoder(d);
  return new Proxy(e,{get:(encoder,k)=>{
   if(k==="beginComputePass")return (descriptor?:GPUComputePassDescriptor)=>{
    let pipeline:GPUComputePipeline|undefined;const bindings=new Map<number,unknown[]>();
    const dispatch=(kind:"dispatchWorkgroups"|"dispatchWorkgroupsIndirect",args:unknown[])=>{
     assert.ok(pipeline);
     const run=(selected:GPUComputePipeline,timestampWrites?:GPUComputePassTimestampWrites)=>{
      const pass=encoder.beginComputePass({...descriptor,timestampWrites});pass.setPipeline(selected);
      for(const [index,values] of bindings)Reflect.apply(pass.setBindGroup,pass,[index,...values]);
      Reflect.apply(pass[kind],pass,args);pass.end();
     };
     const reference=references.get(pipeline);
     if(reference&&frames.includes(frame)){
      assert.ok(!sample,"one surface splat dispatch per frame");
      const band=(solver as unknown as {mixedFrame:{narrowBandFlip:{bins:GPUBuffer;cells:number;vertices:number}}}).mixedFrame.narrowBandFlip;
      const offset=4*(2*band.cells+2*(band.cells/64)),size=4*band.vertices,bins=band.bins;
      const saved=raw.createBuffer({size,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      const results=raw.createBuffer({size:2*size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const queries=raw.createQuerySet({type:"timestamp",count:4*repeats});
      const resolved=raw.createBuffer({size:32*repeats,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const times=raw.createBuffer({size:32*repeats,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      sample={saved,results,queries,resolved,times,size};encoder.copyBufferToBuffer(bins,offset,saved,0,size);
      for(let repeat=0;repeat<repeats;repeat++)for(let variant=0;variant<2;variant++){
       encoder.copyBufferToBuffer(saved,0,bins,offset,size);
       const scalar=(variant+repeat)%2===0,index=4*repeat+2*variant;
       run(scalar?reference:pipeline,{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1});
       if(repeat===0)encoder.copyBufferToBuffer(bins,offset,results,scalar?0:size,size);
      }
      // Leave the ordinary optimized result for subsequent reconstruction.
      encoder.copyBufferToBuffer(saved,0,bins,offset,size);run(pipeline);
      encoder.resolveQuerySet(queries,0,4*repeats,resolved,0);encoder.copyBufferToBuffer(resolved,0,times,0,32*repeats);
     }else run(pipeline);
    };
    return {setPipeline:(p:GPUComputePipeline)=>{pipeline=p;},setBindGroup:(index:number,...values:unknown[])=>{bindings.set(index,values);},
     dispatchWorkgroups:(...args:unknown[])=>dispatch("dispatchWorkgroups",args),dispatchWorkgroupsIndirect:(...args:unknown[])=>dispatch("dispatchWorkgroupsIndirect",args),
     end:()=>{},pushDebugGroup:()=>{},popDebugGroup:()=>{},insertDebugMarker:()=>{}} as unknown as GPUComputePassEncoder;
   };
   const value=Reflect.get(encoder,k,encoder);return typeof value==="function"?value.bind(encoder):value;
  }});
 };
 const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
}});
const device=managedGPUDevice(intercepted,{requireWorkerRealm:false});
const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
try{
 const definition=getSceneDefinition("nbflip-water-slide-long"),quality=definition.methodProfile!.quality;
 const scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 assert.ok(frames.at(-1)!*dt<=scene.duration_s);
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(frame=1;frame<=frames.at(-1)!;frame++){
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();
  assert.equal(solver.info.simulationPipelineError,undefined);
  if(!frames.includes(frame))continue;
  assert.ok(sample,"surface splat probe dispatched");const current:Sample=sample;
  try{
   await Promise.all([current.results.mapAsync(GPUMapMode.READ),current.times.mapAsync(GPUMapMode.READ)]);
   const words=new Uint32Array(current.results.getMappedRange()),vertices=current.size/4;let mismatches=0;
   for(let i=0;i<vertices;i++)if(words[i]!==words[vertices+i])mismatches++;
   const timestamps=new BigUint64Array(current.times.getMappedRange()),reference_ms:number[]=[],cached_ms:number[]=[];
   for(let repeat=0;repeat<repeats;repeat++)for(let variant=0;variant<2;variant++){
    const index=4*repeat+2*variant;((variant+repeat)%2===0?reference_ms:cached_ms).push(Number(timestamps[index+1]!-timestamps[index]!)/1e6);
   }
   console.log(JSON.stringify({frame,time_s:frame*dt,vertices,mismatches,particles:solver.narrowBandFlipInfo?.particles,reference_ms,cached_ms}));
   current.results.unmap();current.times.unmap();assert.equal(mismatches,0,"all nearest-distance words match scalar reference");assert.deepEqual(errors,[]);
  }finally{current.saved.destroy();current.results.destroy();current.queries.destroy();current.resolved.destroy();current.times.destroy();sample=undefined;}
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
