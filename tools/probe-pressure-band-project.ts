/** Frozen live pressure projection: committed copy/project/present versus
 * fused projection/presentation and cached solid predicates. Both texture homes, negative planes and every
 * allocated presentation word are compared, including untouched values.
 * node --import tsx tools/probe-pressure-band-project.ts [--steps=120]
 * [--rounds=8] [--scene=nbflip-figure-9-teaser] [--reference=ead0c168]
 * [--out=artifacts/nbflip-ui-perf/band-project.json]
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync,mkdtempSync,rmSync,writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname,join,resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformPressureBand,UniformPressureBandFields,UniformPressureBandSchedule } from "../lib/methods/uniform/uniform-pressure-band";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

type LiveBand={
 index:GPUBuffer;rows:GPUBuffer;coarse:GPUBuffer;solve:GPUBuffer;workSlots:number;
 fields:UniformPressureBandFields;simulation:UniformMixedOwnership;pressure:UniformMixedOwnership;
 capacity:number;schedule:UniformPressureBandSchedule;solid?:UniformMixedSolid;spatialOrder:boolean;
 pipelines:Map<string,GPUComputePipeline>;launches:Map<string,unknown>;
 encodeSolve(e:GPUCommandEncoder):void;
 dispatch(p:GPUComputePassEncoder,name:string,launch:"slots"):void;
};
type RecordCopy={name:string;size:number;source:GPUBuffer|GPUTexture;output:boolean;capture(e:GPUCommandEncoder,to:GPUBuffer):void;restore(e:GPUCommandEncoder,from:GPUBuffer,reference:boolean):void};
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option("steps")??120),rounds=Number(option("rounds")??8),revision=option("reference")??"ead0c168";
assert.ok([steps,rounds].every(n=>Number.isInteger(n)&&n>0));
const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),scene=structuredClone(sceneDocument(definition));
assert.equal(definition.methodProfile?.methodId,method.id);
const quality=definition.methodProfile!.quality,values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s,out=option("out")??"artifacts/nbflip-ui-perf/band-project.json";
const temporary=mkdtempSync(join(tmpdir(),"fluid-band-project-"));
// Keep the reference in the same ESM module graph: detail field handles
// are registered in module-local WeakMaps, so a CommonJS copy cannot bind them.
writeFileSync(join(temporary,"package.json"),JSON.stringify({type:"module"}));
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined,referenceBand:UniformPressureBand|undefined;
const errors:string[]=[],buffers=new Set<GPUBuffer>();
try{
 const sourcePath="lib/methods/uniform/uniform-pressure-band.ts";
 let source=execFileSync("git",["show",`${revision}:${sourcePath}`],{encoding:"utf8",maxBuffer:2**20});
 assert.ok(source.includes('this.dispatch(project,"copy","slots")')&&source.includes('this.dispatch(project,"present","slots")'),"Reference must contain the three-pass projection");
 // Compile only projection kernels; the reference consumes the candidate's
 // frozen rows and completed solve, without repeating any physics step.
 source=source.replace(/from\s*"(\.\/[^"\n]+)"/g,(_,name:string)=>`from ${JSON.stringify(resolve(dirname(sourcePath),name)+".ts")}`)
  .replace("Object.entries(sources).map(","Object.entries(sources).filter(([name])=>[\"copy\",\"project\",\"present\"].includes(name)).map(");
 const referencePath=join(temporary,"reference.ts");writeFileSync(referencePath,source);
 const Reference=(await import(pathToFileURL(referencePath).href)).UniformPressureBand as typeof UniformPressureBand;
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);
 const shaderSources=new WeakMap<GPUShaderModule,string>(),descriptors=new WeakMap<GPUComputePipeline,GPUComputePipelineDescriptor>();
 const copyable=new Proxy(profile.device,{get:(target,key)=>{
  if(key==="createBuffer")return(d:GPUBufferDescriptor)=>target.createBuffer(["Uniform pressure band rows","Uniform pressure band aggregates"].includes(d.label??"")?{...d,usage:d.usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}:d);
  if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);shaderSources.set(module,d.code);return module;};
  if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{const pipeline=await target.createComputePipelineAsync(d);descriptors.set(pipeline,d);return pipeline;};
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});
 device=managedGPUDevice(copyable,{requireWorkerRealm:false});
 device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 // A structural type avoids granting diagnostics access in production.
 const band=(solver as unknown as {mixedFrame:{band:LiveBand}}).mixedFrame.band;
 const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.COPY_SRC)});buffers.add(b);return b;};
 const release=(b:GPUBuffer)=>{b.destroy();buffers.delete(b);};
 let reference:LiveBand|undefined;
 const bufferRecord=(name:string,buffer:GPUBuffer,offset=0,size=buffer.size-offset,output=false):RecordCopy=>({name,size,source:buffer,output,
  capture:(e,to)=>e.copyBufferToBuffer(buffer,offset,to,0,size),restore:(e,from,ref)=>{
   const target=ref&&["index","rows","solve","coarse"].includes(name)?reference![name as "index"|"rows"|"solve"|"coarse"]:buffer;
   e.copyBufferToBuffer(from,0,target,offset,size);
  }});
 const recordsOf=():RecordCopy[]=>{
  const f=band.fields;
  const records=[bufferRecord("index",band.index),bufferRecord("rows",band.rows),bufferRecord("solve",band.solve),bufferRecord("coarse",band.coarse),bufferRecord("negative",f.negative,0,f.negative.size,true),bufferRecord("presentation",f.presentation.buffer,4*f.presentation.word,4*64*band.capacity,true)];
  const field=uniformDetailField(f.velocity),textures=[...new Set(field?[field.storage.physical(field.handle),field.storage.baseOf(field.handle)!]:[f.velocity])];
  for(const [bank,texture] of textures.entries()){
   assert.equal(texture.format,"rgba32float");const bytesPerRow=Math.ceil(texture.width*16/256)*256,rowsPerImage=texture.height,size=bytesPerRow*rowsPerImage*texture.depthOrArrayLayers;
   const extent:[number,number,number]=[texture.width,texture.height,texture.depthOrArrayLayers];
   records.push({name:`velocity-${bank}`,size,source:texture,output:true,capture:(e,to)=>e.copyTextureToBuffer({texture},{buffer:to,bytesPerRow,rowsPerImage},extent),restore:(e,from)=>e.copyBufferToTexture({buffer:from,bytesPerRow,rowsPerImage},{texture},extent)});
  }
  return records;
 };
 let records:RecordCopy[]=[],snapshots:GPUBuffer[]=[],snapshotsTaken=0,frame=0;
 const solve=band.encodeSolve.bind(band);
 band.encodeSolve=(encoder:GPUCommandEncoder)=>solve(new Proxy(encoder,{get:(e,key)=>{
  if(key==="beginComputePass")return(d?:GPUComputePassDescriptor)=>{
   if(frame===steps&&d?.label==="Uniform pressure band projection"){
    const next=recordsOf();
    if(next.length!==records.length||next.some((r,i)=>r.source!==records[i]?.source||r.size!==records[i]?.size)){
     snapshots.forEach(release);records=next;snapshots=records.map(r=>allocate(r.size));
    }
    records.forEach((r,i)=>r.capture(e,snapshots[i]!));snapshotsTaken++;
   }
   return e.beginComputePass(d);
  };
  const value=Reflect.get(e,key,e);return typeof value==="function"?value.bind(e):value;
 }}));
 for(frame=1;frame<=steps;frame++){
  solver.advanceTo(frame*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  if(frame%12===0||frame===steps)console.log(JSON.stringify({phase:"advance",frame}));
 }
 band.encodeSolve=solve;assert.ok(snapshotsTaken>0);
 // Membership is GPU-owned now. The cached initial layout supplies only
 // invariant dimensions to shader construction; live capacities/bindings
 // and the restored compact band list determine all projection work.
 const layoutView=(ownership:UniformMixedOwnership)=>new Proxy(ownership,{get:(target,key)=>{
  if(key==="layout")return(target as unknown as {currentLayout:unknown}).currentLayout;
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});
 referenceBand=new Reference(device,layoutView(band.simulation),layoutView(band.pressure),{...band.fields,capacity:band.capacity},band.schedule,band.solid,band.spatialOrder);
 await referenceBand.initialize();reference=referenceBand as unknown as LiveBand;reference.workSlots=band.workSlots;
 assert.deepEqual(errors,[],"Reference construction must be valid before replay");
 const cachedPipeline=band.pipelines.get("project")!;assert.ok(cachedPipeline);
 const descriptor=descriptors.get(cachedPipeline)!;assert.ok(descriptor);
 let fusionSource=shaderSources.get(descriptor.compute.module)!;assert.ok(fusionSource.includes("fn umCellIsOpen("));
 fusionSource=fusionSource.replace("let own=umCellIsOpen(p);","let own=umCellOpen(p)>1e-5;")
  .replace("own==umCellIsOpen(q)","own==(umCellOpen(q)>1e-5)")
  .replace("if(!umCellIsOpen(p)||","if(umCellOpen(p)<=1e-5||");
 const fusionModule=raw.createShaderModule({label:"Band presentation fusion without cached solid predicates",code:fusionSource});
 const fusionPipeline=await raw.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module:fusionModule}});
 const restore=(e:GPUCommandEncoder,ref:boolean)=>records.forEach((r,i)=>r.restore(e,snapshots[i]!,ref));
 type Variant="reference"|"fusion"|"cached";
 const encode=(e:GPUCommandEncoder,variant:Variant)=>{
  const ref=variant==="reference";
  if(!ref){band.pipelines.set("project",variant==="fusion"?fusionPipeline:cachedPipeline);band.launches.delete("project");}
  const p=e.beginComputePass({label:ref?"Projection copy + project + present":variant==="fusion"?"Projection copy + fused project/present":"Projection copy + fused project/present + cached solid predicates"}),target=ref?reference!:band;
  for(const name of ref?["copy","project","present"]:["copy","project"])target.dispatch(p,name,"slots");p.end();
 };
 const outputRecords=records.filter(r=>r.output);
 const output=async(ref:boolean)=>{
  const result=outputRecords.map(r=>allocate(r.size)),e=raw.createCommandEncoder();restore(e,ref);encode(e,ref?"reference":"cached");outputRecords.forEach((r,i)=>r.capture(e,result[i]!));raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();return result;
 };
 const left=await output(true),right=await output(false);
 assert.deepEqual(errors,[],"Replay commands must be valid before comparing outputs");
 let words=0,mismatches=0;const banks:unknown[]=[],examples:unknown[]=[];
 for(let bank=0;bank<outputRecords.length;bank++){
  const size=outputRecords[bank]!.size,a=allocate(size,true),b=allocate(size,true),e=raw.createCommandEncoder();e.copyBufferToBuffer(left[bank]!,0,a,0,size);e.copyBufferToBuffer(right[bank]!,0,b,0,size);raw.queue.submit([e.finish()]);
  await Promise.all([a.mapAsync(GPUMapMode.READ),b.mapAsync(GPUMapMode.READ)]);const av=new Uint32Array(a.getMappedRange()),bv=new Uint32Array(b.getMappedRange());let differences=0;
  for(let i=0;i<av.length;i++)if(av[i]!==bv[i]){differences++;if(examples.length<12)examples.push({bank:outputRecords[bank]!.name,word:i,reference:av[i],candidate:bv[i]});}
  banks.push({name:outputRecords[bank]!.name,words:av.length,mismatches:differences});words+=av.length;mismatches+=differences;
  a.unmap();b.unmap();release(a);release(b);release(left[bank]!);release(right[bank]!);
 }
 console.log(JSON.stringify({phase:"parity",words,mismatches,banks,examples}));
 for(const variant of ["reference","fusion","cached"] as const){const e=raw.createCommandEncoder();restore(e,variant==="reference");encode(e,variant);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();}
 const timings:unknown[]=[],totals:Record<Variant,number[]>={reference:[],fusion:[],cached:[]};
 for(let round=0;round<rounds;round++)for(const variant of (round%2?["cached","fusion","reference"]:["reference","fusion","cached"]) as Variant[]){
  profile.start();const e=device.createCommandEncoder();restore(e,variant==="reference");encode(e,variant);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();const passes=await profile.finish();
  const ms=passes.reduce((v,p)=>v+p.ms,0);totals[variant].push(ms);timings.push({round,label:variant,ms,passes});
 }
 const means=Object.fromEntries(Object.entries(totals).map(([name,ms])=>[name,ms.reduce((a,b)=>a+b,0)/ms.length]));
 const result={date:new Date().toISOString(),sceneId:definition.id,referenceRevision:revision,values,dt,steps,rounds,snapshotsTaken,snapshotBytes:records.reduce((n,r)=>n+r.size,0),capacity:band.capacity,workSlots:band.workSlots,words,mismatches,banks,examples,means_ms:means,timingIsolation:"same live pre-projection snapshot restored; shared outputs; variants separately submitted and fenced; inclusive copy/project/present",timings,errors};
 mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));
 assert.equal(mismatches,0,"Every pressure projection output bit must match");assert.deepEqual(errors,[]);
}finally{buffers.forEach(b=>b.destroy());referenceBand?.destroy();solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();rmSync(temporary,{recursive:true,force:true});}
