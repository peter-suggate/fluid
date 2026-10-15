/** Same-state particle advection parity and timestamp probe. Replays both
 * kernels on completed scene fields and restores every mutated buffer.
 * node --import tsx tools/probe-nbflip-advect.ts [--frames=1,120,480] [--repeats=7]
 * Browser GPU must be off. Times exclude simulation and readback. */
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
const frames=(option("frames")??"1,120,480").split(",").map(Number).sort((a,b)=>a-b);
const repeats=Number(option("repeats")??7);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const bindResources=new WeakMap<GPUBindGroup,readonly GPUBindGroupEntry[]>();
const references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
type Capture={pipeline:GPUComputePipeline;reference:GPUComputePipeline;bindings:Map<number,unknown[]>};
let capture:Capture|undefined;
const intercepted=new Proxy(raw,{get:(target,key)=>{
 if(key==="createBindGroup")return (d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);bindResources.set(group,[...d.entries]);return group;};
 if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
  const p=await target.createComputePipelineAsync(d);
  if(d.compute.entryPoint==="advectParticles")references.set(p,await target.createComputePipelineAsync({...d,label:"Particle advection eager collision reference",compute:{...d.compute,constants:{...d.compute.constants,nbAdvectScalarCollision:1}}}));
  return p;
 };
 if(key==="createCommandEncoder")return (d?:GPUCommandEncoderDescriptor)=>{
  const e=target.createCommandEncoder(d);
  return new Proxy(e,{get:(encoder,k)=>{
   if(k==="beginComputePass")return (descriptor?:GPUComputePassDescriptor)=>{
    const pass=encoder.beginComputePass(descriptor);let pipeline:GPUComputePipeline|undefined;
    const bindings=new Map<number,unknown[]>();
    return new Proxy(pass,{get:(p,property)=>{
     if(property==="setPipeline")return (value:GPUComputePipeline)=>{pipeline=value;p.setPipeline(value);};
     if(property==="setBindGroup")return (index:number,...values:unknown[])=>{bindings.set(index,values);Reflect.apply(p.setBindGroup,p,[index,...values]);};
     if(property==="dispatchWorkgroups")return (...args:unknown[])=>{
      const reference=pipeline&&references.get(pipeline);if(pipeline&&reference)capture={pipeline,reference,bindings:new Map(bindings)};
      Reflect.apply(p.dispatchWorkgroups,p,args);
     };
     const value=Reflect.get(p,property,p);return typeof value==="function"?value.bind(p):value;
    }});
   };
   const value=Reflect.get(encoder,k,encoder);return typeof value==="function"?value.bind(encoder):value;
  }});
 };
 const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
}});
const device=managedGPUDevice(intercepted,{requireWorkerRealm:false});
const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
let solver:WebGPUUniformReferenceSolver|undefined;
try{
 const definition=getSceneDefinition("nbflip-water-slide-long"),quality=definition.methodProfile!.quality;
 const scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 assert.ok(frames.at(-1)!*dt<=scene.duration_s);
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(let frame=1;frame<=frames.at(-1)!;frame++){
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();
  assert.equal(solver.info.simulationPipelineError,undefined);
  if(!frames.includes(frame))continue;
  assert.ok(capture,"production advectParticles was dispatched");
  const resourceEntries=bindResources.get(capture.bindings.get(1)![0] as GPUBindGroup);assert.ok(resourceEntries);
  const buffers=[0,2,4].map(binding=>{const resource=resourceEntries.find(e=>e.binding===binding)!.resource as GPUBufferBinding;assert.ok(!resource.offset);return resource.buffer;});
  const sizes=buffers.map(buffer=>buffer.size),total=sizes.reduce((a,b)=>a+b,0);
  const saved=raw.createBuffer({size:total,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const results=raw.createBuffer({size:2*total,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const queries=raw.createQuerySet({type:"timestamp",count:4*repeats});
  const resolved=raw.createBuffer({size:32*repeats,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
  const timesRead=raw.createBuffer({size:32*repeats,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const e=raw.createCommandEncoder();let cursor=0;
   for(const buffer of buffers){e.copyBufferToBuffer(buffer,0,saved,cursor,buffer.size);cursor+=buffer.size;}
   for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){
    // Alternate order to avoid systematically favoring a warm cache.
    const reference=(kind+repeat)%2===0,index=4*repeat+2*kind;
    cursor=0;for(const buffer of buffers){e.copyBufferToBuffer(saved,cursor,buffer,0,buffer.size);cursor+=buffer.size;}
    const pass=e.beginComputePass({label:reference?"advectReference":"advectParticles",timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
    pass.setPipeline(reference?capture.reference:capture.pipeline);
    for(const [index,bindings] of capture.bindings)Reflect.apply(pass.setBindGroup,pass,[index,...bindings]);
    pass.dispatchWorkgroups(4096);pass.end();
    if(repeat===0){cursor=reference?0:total;for(const buffer of buffers){e.copyBufferToBuffer(buffer,0,results,cursor,buffer.size);cursor+=buffer.size;}}
   }
   cursor=0;for(const buffer of buffers){e.copyBufferToBuffer(saved,cursor,buffer,0,buffer.size);cursor+=buffer.size;}e.resolveQuerySet(queries,0,4*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,timesRead,0,32*repeats);
   raw.queue.submit([e.finish()]);await Promise.all([results.mapAsync(GPUMapMode.READ),timesRead.mapAsync(GPUMapMode.READ)]);
   const words=new Uint32Array(results.getMappedRange());let mismatches=0;const firstMismatches:number[]=[];
   for(let word=0;word<total/4;word++)if(words[word]!==words[total/4+word]){mismatches++;if(firstMismatches.length<8)firstMismatches.push(word);}
   const timestamps=new BigUint64Array(timesRead.getMappedRange()),reference_ms:number[]=[],optimized_ms:number[]=[];
   for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){
    const index=4*repeat+2*kind;((kind+repeat)%2===0?reference_ms:optimized_ms).push(Number(timestamps[index+1]!-timestamps[index]!)/1e6);
   }
   console.log(JSON.stringify({frame,time_s:frame*dt,bufferBytes:sizes,mismatches,firstMismatches,particles:solver.narrowBandFlipInfo?.particles,fineTiles:solver.info.uniformMixedFineTiles,reference_ms,optimized_ms}));
   results.unmap();timesRead.unmap();assert.equal(mismatches,0,"all particle records, bin counts and state words match eager reference");assert.deepEqual(errors,[]);
  }finally{saved.destroy();results.destroy();queries.destroy();resolved.destroy();timesRead.destroy();}
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
