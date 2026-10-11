/** Frozen live cut-tile plane fit: original serial per-tile work versus a
 * cooperative 128-lane fit. Captures all bound inputs immediately before the
 * last frame's coarse pressure authority, then compares both fitted phi and
 * the downstream pressure authority. No candidate advances the simulation.
 * node --import tsx tools/probe-pressure-cut-fit.ts --steps=96 --rounds=12
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync,writeFileSync } from "node:fs";
import { dirname,resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";
import { groupedCutFitWGSL,hybridCutFitWGSL,maskCutFitWGSL,parallelCutFitWGSL,referenceCutFitWGSL,streamedCutFitWGSL } from "./parallel-cut-fit";

type Authority={ownership:UniformMixedOwnership;solid:UniformMixedSolid;pipelines:Map<string,GPUComputePipeline>;
 variant(p:GPUComputePipeline):GPUComputePipeline;dispatchOwners(p:GPUComputePassEncoder,pipeline:GPUComputePipeline):void;
 encode(e:GPUCommandEncoder,g:UniformDetailGroup):void};
type RecordCopy={name:string;source:GPUBuffer|GPUTexture;size:number;offset?:number;snapshot:GPUBuffer;
 capture(e:GPUCommandEncoder,to:GPUBuffer):void;restore(e:GPUCommandEncoder):void};
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option("steps")??96),rounds=Number(option("rounds")??12),repeats=Number(option("repeats")??4);
type Variant="reference"|"parallel"|"grouped"|"empty"|"masks"|"hybrid"|"streamed";
const variants:Variant[]=process.argv.includes("--streamed")?["reference","masks","streamed","hybrid"]:process.argv.includes("--masks")?["reference","empty","masks","grouped","hybrid"]:process.argv.includes("--both")?["reference","parallel","grouped"]:process.argv.includes("--grouped")?["reference","grouped"]:["reference","parallel"];
const out=option("out")??"artifacts/nbflip-ui-perf/parallel-pressure-cut-fit-96.json";
const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),scene=structuredClone(sceneDocument(definition)),quality=definition.methodProfile!.quality;
const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
assert.ok([steps,rounds,repeats].every(n=>Number.isInteger(n)&&n>0));
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
const errors:string[]=[],buffers=new Set<GPUBuffer>();
try{
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);
 const sources=new WeakMap<GPUShaderModule,string>(),descriptors=new WeakMap<GPUComputePipeline,GPUComputePipelineDescriptor>();
 const groups=new WeakMap<GPUBindGroup,GPUBindGroupEntry[]>(),views=new WeakMap<GPUTextureView,GPUTexture>();
 const proxy=new Proxy(profile.device,{get:(target,key)=>{
  if(key==="createBuffer")return(d:GPUBufferDescriptor)=>target.createBuffer((d.usage&(GPUBufferUsage.MAP_READ|GPUBufferUsage.MAP_WRITE))?d:{...d,usage:d.usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  if(key==="createTexture")return(d:GPUTextureDescriptor)=>{
   const texture=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
   const original=texture.createView.bind(texture);
   texture.createView=(d?:GPUTextureViewDescriptor)=>{const view=original(d);views.set(view,texture);return view;};return texture;
  };
  if(key==="createBindGroup")return(d:GPUBindGroupDescriptor)=>{const entries=[...d.entries],group=target.createBindGroup({...d,entries});groups.set(group,entries);return group;};
  if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
  if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{const pipeline=await target.createComputePipelineAsync(d);descriptors.set(pipeline,d);return pipeline;};
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});
 device=managedGPUDevice(proxy,{requireWorkerRealm:false});device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.COPY_SRC)});buffers.add(b);return b;};
 const release=(buffer:GPUBuffer)=>{buffer.destroy();buffers.delete(buffer);};
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const split=(solver as unknown as {mixedFrame:{split:{authority:Authority;authorityGroup:UniformDetailGroup}}}).mixedFrame.split;
 const authority=split.authority,tiles=authority.ownership.capacity.tiles,h=Math.min(...authority.ownership.capacity.lattice.cellSize_m);
 assert.ok(authority.pipelines.has("cut")&&authority.solid.coarse,"Scene must exercise the coarse solid authority");
 const originalEncode=authority.encode.bind(authority);let records:RecordCopy[]=[],savedGroups:GPUBindGroup[]=[],baseGroup:GPUBindGroup|undefined,snapshotsTaken=0,frame=0;
 const recordBuffer=(name:string,binding:GPUBufferBinding)=>{
  const {buffer,offset=0,size=buffer.size-offset}=binding;
  if(records.some(r=>r.source===buffer&&r.offset===offset&&r.size===size))return;
  const snapshot=allocate(size);records.push({name,source:buffer,size,offset,snapshot,
   capture:(e,to)=>e.copyBufferToBuffer(buffer,offset,to,0,size),restore:e=>e.copyBufferToBuffer(snapshot,0,buffer,offset,size)});
 };
 const recordTexture=(name:string,texture:GPUTexture)=>{
  if(records.some(r=>r.source===texture))return;
  const bytes=texture.format==="r32float"?4:texture.format==="rg32float"?8:texture.format==="rgba32float"?16:0;assert.ok(bytes,texture.format);
  const extent:[number,number,number]=[texture.width,texture.height,texture.depthOrArrayLayers],bytesPerRow=Math.ceil(texture.width*bytes/256)*256,rowsPerImage=texture.height,size=bytesPerRow*rowsPerImage*texture.depthOrArrayLayers;
  const snapshot=allocate(size);records.push({name,source:texture,size,snapshot,
   capture:(e,to)=>e.copyTextureToBuffer({texture},{buffer:to,bytesPerRow,rowsPerImage},extent),restore:e=>e.copyBufferToTexture({buffer:snapshot,bytesPerRow,rowsPerImage},{texture},extent)});
 };
 authority.encode=(encoder,group)=>{
  if(frame===steps){
   const next=[authority.ownership.bindGroup,group.group,authority.solid.tileGroup];
   if(!savedGroups.length||next.some((g,i)=>g!==savedGroups[i])||group.base!==baseGroup){
    records.forEach(r=>release(r.snapshot));records=[];savedGroups=next;baseGroup=group.base;
    [...next,group.base].forEach((g,index)=>{
     const entries=groups.get(g);assert.ok(entries,"Bind group descriptor must be recorded");
     for(const entry of entries){const resource=entry.resource,name=`group${index}.binding${entry.binding}`;
      if("buffer" in resource)recordBuffer(name,resource as GPUBufferBinding);
      else{const texture=views.get(resource as GPUTextureView);if(texture)recordTexture(name,texture);}
     }
    });
   }
   records.forEach(r=>r.capture(encoder,r.snapshot));snapshotsTaken++;
  }
  originalEncode(encoder,group);
 };
 for(frame=1;frame<=steps;frame++){solver.advanceTo(frame*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(frame%12===0||frame===steps)console.log(JSON.stringify({phase:"advance",frame}));}
 authority.encode=originalEncode;assert.ok(snapshotsTaken>0&&baseGroup);
 let baseline=authority.variant(authority.pipelines.get("cut")!);const descriptor=descriptors.get(baseline);assert.ok(descriptor);
 const productionSource=sources.get(descriptor.compute.module);assert.ok(productionSource);
 const originalSource=referenceCutFitWGSL(productionSource);
 if(originalSource!==productionSource){const module=raw.createShaderModule({label:"Original serial cut fit",code:originalSource});baseline=await raw.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module}});}
 const pipelines=new Map<Variant,GPUComputePipeline>([["reference",baseline]]);
 for(const variant of variants.filter(v=>v!=="reference")){
  const code=variant==="empty"||variant==="masks"?maskCutFitWGSL(originalSource,variant==="masks"):(variant==="streamed"?streamedCutFitWGSL:variant==="hybrid"?hybridCutFitWGSL:variant==="grouped"?groupedCutFitWGSL:parallelCutFitWGSL)(originalSource);writeFileSync(`/tmp/fluid-${variant}-cut-fit.wgsl`,code);
  const module=raw.createShaderModule({label:`${variant} cut tile least-squares fit`,code});
  const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(messages.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
  pipelines.set(variant,await raw.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module}}));
 }
 const bind=(pass:GPUComputePassEncoder,base=false)=>{savedGroups.forEach((g,i)=>pass.setBindGroup(i,i===1&&base?baseGroup!:g));};
 const restore=(encoder:GPUCommandEncoder)=>records.forEach(r=>r.restore(encoder));
 const encode=(encoder:GPUCommandEncoder,variant:Variant,full=true)=>{
  let pass=encoder.beginComputePass({label:`cut:${variant}`});bind(pass);pass.setPipeline(pipelines.get(variant)!);pass.dispatchWorkgroups(Math.max(1,Math.min(1024,Math.ceil(tiles/64))));pass.end();
  if(full){
   pass=encoder.beginComputePass({label:"Coarse authority after cut fit"});bind(pass,true);
   pass.setPipeline(authority.variant(authority.pipelines.get("absent")!));const dims=authority.ownership.capacity.lattice.dimensions;pass.dispatchWorkgroups(dims.reduce((n,d)=>n*Math.ceil(d/16),1));
   authority.dispatchOwners(pass,authority.variant(authority.pipelines.get("surface")!));pass.end();
  }
 };
 const binding=(at:number)=>{const entry=groups.get(savedGroups[1]!)!.find(e=>e.binding===at);assert.ok(entry&&"buffer" in entry.resource);return entry.resource as GPUBufferBinding;};
 const phi=binding(3),balance=binding(6);
 const readBinding=async(binding:GPUBufferBinding,from?:GPUBuffer)=>{
  const offset=binding.offset??0,size=binding.size??binding.buffer.size-offset,read=allocate(size,true),e=raw.createCommandEncoder();
  e.copyBufferToBuffer(from??binding.buffer,from?0:offset,read,0,size);raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange().slice(0));read.unmap();release(read);return words;
 };
 const solidRecord=records.find(r=>r.source===authority.solid.coarse!.record);assert.ok(solidRecord);
 const solidWords=await readBinding({buffer:solidRecord.source as GPUBuffer,size:solidRecord.size},solidRecord.snapshot),solidFloats=new Float32Array(solidWords.buffer);
 const cutTiles:number[]=[],liveHistogram:Record<number,number>={};let fullOpenTiles=0;
 for(let t=0;t<tiles;t++)if(solidFloats[4*(authority.solid.coarse!.count+t)]!>0.5&&solidFloats[4*(authority.solid.coarse!.count+t)+1]!>0.5){
  cutTiles.push(t);const word=4*(authority.solid.coarse!.count+tiles+3*t),open=[(solidFloats[word]!|(solidFloats[word+1]!<<16))>>>0,(solidFloats[word+2]!|(solidFloats[word+3]!<<16))>>>0];let live=0;
  if(open.every(word=>word===0xffffffff))fullOpenTiles++;
  for(let v=0;v<125;v++){const g=[v%5,Math.floor(v/5)%5,Math.floor(v/25)];let yes=false;
   for(let k=0;k<8;k++){const c=g.map((p,a)=>p-((k>>a)&1));if(c.every(p=>p>=0&&p<4)){const i=c[0]!+4*c[1]!+16*c[2]!;yes ||= !!((open[i>>>5]!>>>(i&31))&1);}}
   live+=Number(yes);
  }liveHistogram[live]=(liveHistogram[live]??0)+1;
 }
 console.log(JSON.stringify({phase:"coverage",tiles,cutTiles:cutTiles.length,fullOpenTiles,liveHistogram,snapshotBytes:records.reduce((n,r)=>n+r.size,0)}));assert.ok(cutTiles.length>0);
 const capture=async(variant:Variant)=>{const e=raw.createCommandEncoder();restore(e);encode(e,variant);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();return {phi:await readBinding(phi),balance:await readBinding(balance)};};
 const left=await capture("reference");
 const liquidThreshold=Math.fround(Math.fround(-.001)*Math.fround(4*Math.fround(h)));
 const compare=(a:Uint32Array,b:Uint32Array,indices?:number[])=>{
  const af=new Float32Array(a.buffer),bf=new Float32Array(b.buffer);let differences=0,maxAbsolute=0,maxRelative=0,signChanges=0,liquidChanges=0,nonfinite=0;const examples:unknown[]=[];
  for(const i of indices??a.keys()){if(a[i]===b[i])continue;differences++;const x=af[i]!,y=bf[i]!,absolute=Math.abs(x-y);maxAbsolute=Math.max(maxAbsolute,absolute);
   if(Math.abs(x)>h*1e-4)maxRelative=Math.max(maxRelative,absolute/Math.abs(x));if((x<0)!==(y<0))signChanges++;if((x<liquidThreshold)!==(y<liquidThreshold))liquidChanges++;if(!Number.isFinite(x)||!Number.isFinite(y))nonfinite++;
   if(examples.length<8)examples.push({word:i,reference:x,candidate:y,absolute_cells:absolute/h});
  }
  return {words:indices?.length??a.length,differences,maxAbsolute_cells:maxAbsolute/h,maxRelativeAwayFromZero:maxRelative,signChanges,liquidChanges,nonfinite,examples};
 };
 const comparisons=[];
 for(const variant of variants.filter(v=>v!=="reference")){
  const right=await capture(variant),cut=compare(left.balance,right.balance,cutTiles.map(t=>2*(1+t))),allBalance=compare(left.balance,right.balance),authorityPhi=compare(left.phi,right.phi);
  comparisons.push({variant,cut,allBalance,authorityPhi});console.log(JSON.stringify({phase:"parity",variant,cut,allBalance,authorityPhi,errors}));
  assert.equal(allBalance.differences,cut.differences,"Only active cut-fit output words may differ");
  if(variant==="empty"){assert.equal(cut.differences,0);assert.equal(authorityPhi.differences,0);}
  assert.equal(cut.nonfinite,0);assert.equal(authorityPhi.nonfinite,0);assert.deepEqual(errors,[]);
 }
 const waitForIdle=async()=>{if(!process.argv.includes("--wait-idle"))return;while(true){
  const other=execFileSync("ps",["-axo","pid=,command="],{encoding:"utf8"}).split("\n").filter(line=>/node .*\b(probe-|benchmark-|validate-)/.test(line)&&!line.includes(` ${process.pid} `)&&!line.includes("probe-pressure-cut-fit.ts"));
  if(!other.length)return;console.log(JSON.stringify({phase:"waiting-for-other-processes",other}));await new Promise(resolve=>setTimeout(resolve,2000));
 }};
 await waitForIdle();
 for(let n=0;n<3;n++)for(const variant of variants){const e=raw.createCommandEncoder();restore(e);encode(e,variant);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();}
 const timings:{round:number;variant:Variant;ms:number;cut_ms:number;authority_ms:number}[]=[];
 console.log(JSON.stringify({phase:"timing",rounds,repeats}));
 for(let round=0;round<rounds;round++)for(const variant of round%2?[...variants].reverse():variants){
  let e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  profile.start();e=device.createCommandEncoder();for(let n=0;n<repeats;n++)encode(e,variant);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  const passes=await profile.finish(),cut_ms=passes.filter(p=>p.label.startsWith("cut:")).reduce((n,p)=>n+p.ms,0)/repeats,authority_ms=passes.filter(p=>p.label==="Coarse authority after cut fit").reduce((n,p)=>n+p.ms,0)/repeats;
  timings.push({round,variant,ms:cut_ms+authority_ms,cut_ms,authority_ms});
 }
 const mean=(variant:Variant,key:"ms"|"cut_ms"|"authority_ms")=>timings.filter(t=>t.variant===variant).reduce((n,t)=>n+t[key],0)/rounds;
 const means=variants.map(variant=>({variant,ms:mean(variant,"ms"),cut_ms:mean(variant,"cut_ms"),authority_ms:mean(variant,"authority_ms"),saved_ms:mean("reference","ms")-mean(variant,"ms")}));
 const result={date:new Date().toISOString(),sceneId:definition.id,steps,dt,values,rounds,repeats,tiles,cutTiles:cutTiles.length,fullOpenTiles,liveHistogram,comparisons,means,timings,errors};
 mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));
 assert.deepEqual(errors,[]);
}finally{solver?.destroy();await device?.queue.onSubmittedWorkDone();buffers.forEach(b=>b.destroy());profile?.destroy();device?.destroy();}
