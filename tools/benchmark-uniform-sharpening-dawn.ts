/** Production Figure 9 stage medians and final-field hashes, under the GPU lease.
 * Compare separate processes with FLUID_UNIFORM_AB_OFF=sharpenflux,edgeplanes.
 * UNIFORM_BENCH_STAGE=extension selects velocity extension.
 * UNIFORM_BENCH_PASSES=on measures individual kernels instead of stages:
 * the two timestamp modes must not overwrite one another's pass descriptors.
 */
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
const median=(v:number[])=>{const s=[...v].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]!+s[Math.floor(s.length/2)]!)/2;};
const frames=Number(process.env.UNIFORM_BENCH_FRAMES??180);
const warmup=Number(process.env.UNIFORM_BENCH_WARMUP??120);
assert.ok(Number.isSafeInteger(frames)&&Number.isSafeInteger(warmup)&&warmup>=0&&frames>warmup);
await acquireWebGPUExclusiveLock("dawn-probe",`Uniform ${process.env.UNIFORM_BENCH_STAGE??"sharpening"} benchmark`);
let device:GPUDevice|undefined;
const results:unknown[]=[];
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 const rawDevice=await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 if(process.env.UNIFORM_BENCH_SHADER_ERRORS){
  const create=rawDevice.createShaderModule.bind(rawDevice);
  rawDevice.createShaderModule=(descriptor)=>{const module=create(descriptor);void module.getCompilationInfo().then(info=>{for(const m of info.messages)if(m.type==="error")console.error(descriptor.label,m.lineNum,m.message,descriptor.code.split("\n").slice(m.lineNum-2,m.lineNum+1).join("\n"));});return module;};
 }
 device=managedGPUDevice(rawDevice,{requireWorkerRealm:false});
 usePerformanceInstrumentationStore.getState().setEnabled(process.env.UNIFORM_BENCH_PASSES!=="on");
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 for(const sceneId of (process.env.UNIFORM_BENCH_SCENE?[process.env.UNIFORM_BENCH_SCENE]:["cm12-figure-9"])){
  const scene=sceneDocument(getSceneDefinition(sceneId));
  const arms:{mode:string;full:number[];stages:Record<string,number[]>;pages:number|undefined}[]=[];
  for(const mode of ["pages"]){
   const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,
    uniformGeometricSolverOptions({},scene),()=>{});
   console.log(JSON.stringify({grid:[solver.info.nx,solver.info.ny,solver.info.nz]}));
   const query=device.createQuerySet({type:"timestamp",count:256});
   const resolved=device.createBuffer({size:2048,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size:2048,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   let labels:string[]=[];
   const access=solver as any;
   const method=process.env.UNIFORM_BENCH_STAGE==="extension"?"encodeVelocityExtrapolation":"encodeGeometricVolume";
   const original=access[method].bind(solver);
   access[method]=(encoder:GPUCommandEncoder,...args:unknown[])=>{
    labels=[];
    const proxy=new Proxy(encoder,{get(target,key){
     if(key==="beginComputePass")return (desc:GPUComputePassDescriptor)=>{
      if(process.env.UNIFORM_BENCH_PASSES==="on"){
       assert.equal(desc.timestampWrites,undefined,"per-pass and stage timestamp modes must be separate");
       const index=labels.length*2;assert.ok(index+1<256,"timestamp query capacity");labels.push(desc.label!);
       return target.beginComputePass({...desc,timestampWrites:{querySet:query,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
      }
      return target.beginComputePass(desc);
     };
     const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
    }});
    original(proxy,...args);
    if(labels.length)encoder.resolveQuerySet(query,0,labels.length*2,resolved,0);
    encoder.copyBufferToBuffer(resolved,0,read,0,labels.length*16);
   };
   const full:number[]=[],stages:Record<string,number[]>={};let priorSample=-1;
   try{
    for(let frame=1;frame<=frames;frame++){
     const start=performance.now();(solver as any).lastPhysicsTraceAt_ms=-Infinity;assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();await device.queue.onSubmittedWorkDone();
     if(frame>warmup)full.push(performance.now()-start);
     await read.mapAsync(GPUMapMode.READ);
     const times=new BigUint64Array(read.getMappedRange());const sums:Record<string,number>={};
     labels.forEach((label,i)=>{assert.ok(times[2*i]!>0n&&times[2*i+1]!>=times[2*i]!,`invalid timestamp for ${label}`);sums[label]=(sums[label]??0)+Number(times[2*i+1]!-times[2*i]!)/1e6;});
     if(frame>warmup)for(const [label,time] of Object.entries(sums))(stages[label]??=[]).push(time);
     read.unmap();
     await solver.readStats();
     const trace=solver.info.physicsTrace;
     if(frame>warmup&&trace?.measurementSource==="gpu-hardware-timestamp"&&trace.sampleId!==priorSample){
      for(const phase of trace.phases)(stages[phase.label]??=[]).push(phase.duration_ms);
      priorSample=trace.sampleId;
     }
    }
    const hashes:Record<string,string>={};
    for(const name of ["volumeTexture","vertexPhiTexture","velocityTexture"] as const){
     const texture=solver[name]!;const row=Math.ceil(texture.width*(texture.format==="rgba32float"?16:4)/256)*256;
     const buffer=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
     const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture},{buffer,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);device.queue.submit([encoder.finish()]);
     await buffer.mapAsync(GPUMapMode.READ);hashes[name]=createHash("sha256").update(new Uint8Array(buffer.getMappedRange())).digest("hex");buffer.unmap();buffer.destroy();
    }
    const arm={hashes,volumeDrift:solver.info.volumeDrift,representedVolumeDrift:solver.info.representedVolumeDrift,mode,full,stages,pages:solver.info.uniformVolumePagesActive,transportTiles:solver.info.uniformVolumeTransportWorkgroups,sharpenTiles:solver.info.uniformVolumeSharpenWorkgroups};arms.push(arm);console.log(JSON.stringify({sceneId,mode,full_ms:median(full),stages:Object.fromEntries(Object.entries(stages).map(([k,v])=>[k,median(v)])),pages:arm.pages,hashes}));
   }finally{solver.destroy();query.destroy();resolved.destroy();read.destroy();}
  }
  
  const result={sceneId,scope:"queue-fenced production simulation excluding rendering",profiledStage:process.env.UNIFORM_BENCH_STAGE??"volume",timestampMode:process.env.UNIFORM_BENCH_PASSES==="on"?"passes":"stages",frames,warmup,disabledOptimizations:process.env.FLUID_UNIFORM_AB_OFF??"",arms};
  results.push(result);console.log(JSON.stringify({...result,arms:undefined}));
 }
 assert.deepEqual(errors,[]);
 writeFileSync(process.env.UNIFORM_BENCH_OUTPUT??"/tmp/uniform-sharpening.json",JSON.stringify(results,null,2)+"\n");
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
