/** Frozen-input copy/reconstruct/redistance replay. Compares every live phi
 * word (base and fine/mixed physical texels), metric and crossing mask, with
 * other physical texels reported separately. Includes the eliminated copy
 * in the timestamp interval. No scene values are changed.
 * node --import tsx tools/probe-narrow-band-inplace.ts --steps=96 --rounds=8
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField, type UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { GPUPassProfile } from "./gpu-pass-profile";

class Variants {
 readonly device:GPUDevice;
 readonly baseline=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
 constructor(raw:GPUDevice){
  const sources=new WeakMap<GPUShaderModule,string>(),originals=new WeakMap<GPUShaderModule,GPUShaderModule>();
  this.device=new Proxy(raw,{get:(target,key)=>{
   if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);sources.set(m,d.code);return m;};
   if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
    const p=await target.createComputePipelineAsync(d);
    if(["redistanceFine","redistanceCoarse"].includes(d.compute.entryPoint??"")){
     let m=originals.get(d.compute.module);
     if(!m){
      const source=sources.get(d.compute.module);assert.ok(source);
      const own=source.match(/var value=udLoad_outputPhi\([^;]+\)\.x;/g);
      if(own?.length!==1)writeFileSync("/tmp/narrow-band-inplace-reference-source.wgsl",source);
      assert.equal(own?.length,1,"Own-texel storage load must be unique");
      m=target.createShaderModule({label:"Reference separate-field redistance",code:source.replace(own![0]!,"var value=umSampleVertex(vec3f(p));")});
      const messages=(await m.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(messages.map(m=>`${m.lineNum}: ${m.message}`),[]);originals.set(d.compute.module,m);
     }
     this.baseline.set(p,await target.createComputePipelineAsync({...d,compute:{...d.compute,module:m}}));
    }
    return p;
   };
   const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;
  }});
 }
}
type Stage={
 bins:GPUBuffer;parity:number;ownership:UniformMixedOwnership;solid?:UniformMixedSolid;
 pipelines:Map<string,GPUComputePipeline>;groups:Record<string,[UniformDetailGroup,UniformDetailGroup]>;
 measureBand(e:GPUCommandEncoder,group?:string):void;
 dispatchBatch(e:GPUCommandEncoder,entries:readonly string[],group:string|undefined,label:string):void;
 reconstruct(e:GPUCommandEncoder):void;redistance(e:GPUCommandEncoder):void;
};
type Frame={
 narrowBandFlip:Stage;ownership:UniformMixedOwnership;
 fields:{phi:GPUTexture;phiScratch:GPUTexture};
 surface:{encode(e:GPUCommandEncoder,entry:"advect",group:UniformDetailGroup):void};surfaceGroups:readonly UniformDetailGroup[];
 phiResolve:{encode(e:GPUCommandEncoder,group:UniformDetailGroup):void};phiResolveGroups:{phi:UniformDetailGroup;scratch:UniformDetailGroup};
 copyWhole(e:GPUCommandEncoder,a:GPUTexture,b:GPUTexture):void;
};
async function run(){
 const option=(k:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3);
 const steps=Number(option("steps")??96),rounds=Number(option("rounds")??8),allowedUlp=Number(option("max-ulp")??0);
 const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser");assert.equal(definition.methodProfile?.methodId,method.id);
 const quality=definition.methodProfile!.quality,scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides,timeStep:"sixtieth"});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s,out=option("out")??"artifacts/nbflip-ui-perf/inplace-redistance.json";
 let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
 const errors:string[]=[],buffers:GPUBuffer[]=[];
 try{
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  profile=new GPUPassProfile(raw);const variants=new Variants(profile.device);device=managedGPUDevice(variants.device,{requireWorkerRealm:false});
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  for(let n=1;n<=steps;n++){solver.advanceTo(n*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(n%12===0||n===steps)console.log(JSON.stringify({phase:"advance",frame:n,time:n*dt}));}
  const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame,stage=frame.narrowBandFlip;
  // Dawn/Metal omits timestamps for an empty pass; a real one-lane dispatch
  // brackets the first copy without changing any solver state.
  const marker=raw.createBuffer({size:4,usage:GPUBufferUsage.STORAGE});buffers.push(marker);
  const markerPipeline=await raw.createComputePipelineAsync({layout:"auto",compute:{module:raw.createShaderModule({code:"@group(0) @binding(0) var<storage,read_write> count:array<u32>; @compute @workgroup_size(1) fn main(){count[0]+=1u;}"}),entryPoint:"main"}});
  const markerGroup=raw.createBindGroup({layout:markerPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:marker}}]});
  // Freeze an actual advected input, without moving the particle bank again.
  let e=device.createCommandEncoder();frame.surface.encode(e,"advect",frame.surfaceGroups[0]!);frame.phiResolve.encode(e,frame.phiResolveGroups.scratch);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.COPY_SRC)});buffers.push(b);return b;};
  const dims=[solver.info.nx,solver.info.ny,solver.info.nz],cells=dims.reduce((a,b)=>a*b,1),tiles=cells/64,vertices=dims.reduce((a,b)=>a*(b+1),1);
  const records=[frame.fields.phi,frame.fields.phiScratch].flatMap((handle,fieldIndex)=>{
   const field=uniformDetailField(handle);assert.ok(field);
   return [...new Set([field.storage.physical(handle),field.storage.baseOf(handle)!])].map((texture,home)=>{
    const bytesPerRow=Math.ceil(texture.width*4/256)*256;
    return {texture,fieldIndex,home,bytesPerRow,rowsPerImage:texture.height,size:bytesPerRow*texture.height*texture.depthOrArrayLayers,extent:[texture.width,texture.height,texture.depthOrArrayLayers] as [number,number,number]};
   });
  });
  const original=records.map(r=>allocate(r.size)),binsOriginal=allocate(stage.bins.size),topologyRead=allocate(16*tiles,true);
  e=raw.createCommandEncoder();records.forEach((r,i)=>e.copyTextureToBuffer({texture:r.texture},{buffer:original[i]!,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},r.extent));
  e.copyBufferToBuffer(stage.bins,0,binsOriginal,0,stage.bins.size);e.copyBufferToBuffer(frame.ownership.presentation.buffer,frame.ownership.presentation.offset??0,topologyRead,0,16*tiles);raw.queue.submit([e.finish()]);
  await topologyRead.mapAsync(GPUMapMode.READ);const topology=new Uint32Array(topologyRead.getMappedRange()).slice();topologyRead.unmap();
  const restore=(e:GPUCommandEncoder)=>{records.forEach((r,i)=>e.copyBufferToTexture({buffer:original[i]!,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},{texture:r.texture},r.extent));e.copyBufferToBuffer(binsOriginal,0,stage.bins,0,stage.bins.size);};
  const metricOffset=4*(2*cells+2*tiles),metricSize=4*vertices,crossingOffset=4*(3*cells+vertices+2*tiles+4),crossingSize=12*tiles;
  const encode=(e:GPUCommandEncoder,baseline:boolean)=>{
   const markerPass=e.beginComputePass({label:`inplace interval begin ${baseline?"baseline":"candidate"}`});markerPass.setPipeline(markerPipeline);markerPass.setBindGroup(0,markerGroup);markerPass.dispatchWorkgroups(1);markerPass.end();
   if(!baseline){stage.reconstruct(e);frame.phiResolve.encode(e,frame.phiResolveGroups.phi);stage.redistance(e);frame.phiResolve.encode(e,frame.phiResolveGroups.phi);return;}
   frame.copyWhole(e,frame.fields.phiScratch,frame.fields.phi);
   stage.measureBand(e,"surface");e.clearBuffer(stage.bins,metricOffset,metricSize);
   stage.dispatchBatch(e,["surfaceCells","surfaceSplat",...(!stage.ownership.coarseOnly?["coupleFine"]:[]),"couple"],"surface","reference surface reconstruction");
   frame.phiResolve.encode(e,frame.phiResolveGroups.scratch);stage.measureBand(e,"reconstruct");
   const pass=e.beginComputePass({label:"reference separate-field redistance"});pass.setBindGroup(0,stage.ownership.bindGroup);if(stage.solid)pass.setBindGroup(2,stage.solid.tileGroup);
   pass.setBindGroup(1,stage.groups.reconstruct![stage.parity]!.group);
   const metric=stage.pipelines.get("buildDistance")!;pass.setPipeline(stage.solid?.select(metric)??metric);pass.dispatchWorkgroups(1024);
   for(const entry of [...(!stage.ownership.coarseOnly?["redistanceFine"]:[]),"redistanceCoarse"]){
    const compiled=stage.pipelines.get(entry)!;const selected=stage.solid?.select(compiled)??compiled;const reference=variants.baseline.get(selected);assert.ok(reference);
    pass.setPipeline(reference);pass.dispatchWorkgroups(1024);
   }pass.end();frame.phiResolve.encode(e,frame.phiResolveGroups.phi);
  };
  const outputRecords=records.filter(r=>r.fieldIndex===0);
  const capture=async(baseline:boolean)=>{
   const reads=[...outputRecords.map(r=>allocate(r.size,true)),allocate(metricSize,true),allocate(crossingSize,true)];
   const e=device!.createCommandEncoder();restore(e);encode(e,baseline);
   outputRecords.forEach((r,i)=>e.copyTextureToBuffer({texture:r.texture},{buffer:reads[i]!,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},r.extent));
   e.copyBufferToBuffer(stage.bins,metricOffset,reads[reads.length-2]!,0,metricSize);e.copyBufferToBuffer(stage.bins,crossingOffset,reads[reads.length-1]!,0,crossingSize);raw.queue.submit([e.finish()]);
   return Promise.all(reads.map(async b=>{await b.mapAsync(GPUMapMode.READ);const a=new Uint32Array(b.getMappedRange()).slice();b.unmap();b.destroy();return a;}));
  };
  const baseline=await capture(true),candidate=await capture(false);const comparison:unknown[]=[];
  // Reusable real-mask fixture for the independent integer reach benchmark.
  const maskOut=option("mask-out");if(maskOut){mkdirSync(dirname(maskOut),{recursive:true});writeFileSync(maskOut,Buffer.from(baseline[baseline.length-1]!.buffer));}
  const ordered=(word:number)=>word&0x80000000?0x80000000-(word&0x7fffffff):0x80000000+word;
  baseline.forEach((a,bank)=>{
   const b=candidate[bank]!,af=new Float32Array(a.buffer),bf=new Float32Array(b.buffer),record=outputRecords[bank];let liveWords=0,mismatches=0,inactiveMismatches=0,maxUlp=0,maxAbsolute=0;const first:unknown[]=[];
   for(let i=0;i<a.length;i++){
    let live=true;
    if(record){
     const pitch=record.bytesPerRow/4,x=i%pitch,y=Math.floor(i/pitch)%record.rowsPerImage,z=Math.floor(i/(pitch*record.rowsPerImage));
     live=x<record.texture.width;
     if(live&&record.home===0&&record.texture.width>=dims[0]!+1){
      live=x<=dims[0]!&&y<=dims[1]!&&z<=dims[2]!;
      if(live){const tx=Math.floor(Math.min(x,dims[0]!-1)/4),ty=Math.floor(Math.min(y,dims[1]!-1)/4),tz=Math.floor(Math.min(z,dims[2]!-1)/4),tile=tx+dims[0]!/4*(ty+dims[1]!/4*tz);live=(topology[2*tiles+2*tile+1]!>>>27)===1;}
     }
    }
    if(live)liveWords++;
    if(a[i]===b[i])continue;
    if(!live){inactiveMismatches++;continue;}
    mismatches++;const ulp=Math.abs(ordered(a[i]!)-ordered(b[i]!));maxUlp=Math.max(maxUlp,ulp);maxAbsolute=Math.max(maxAbsolute,Math.abs(af[i]!-bf[i]!));
    if(first.length<8)first.push({word:i,baseline:a[i],candidate:b[i],baselineFloat:af[i],candidateFloat:bf[i],ulp});
   }
   comparison.push({bank,kind:record?`phi ${record.home===0?"physical":"base"}`:bank===baseline.length-2?"metric":"crossing masks/reach",words:a.length,liveWords,mismatches,inactiveMismatches,maxUlp,maxAbsolute,first});
  });
  // Restore identical input before every timing interval, outside its marks.
  for(let n=0;n<2;n++)for(const b of [true,false]){e=device.createCommandEncoder();restore(e);encode(e,b);raw.queue.submit([e.finish()]);}await raw.queue.onSubmittedWorkDone();
  const times:{baseline:boolean;ms:number;passes:Awaited<ReturnType<GPUPassProfile["finish"]>>}[]=[];
  for(let round=0;round<rounds;round++)for(const b of round%2?[false,true]:[true,false]){
   profile.start();e=device.createCommandEncoder();restore(e);encode(e,b);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();const passes=await profile.finish();
   assert.ok(passes.every(p=>p.idle_ms>=0&&p.idle_ms<1000),"Every interval marker must have valid timestamps");
   times.push({baseline:b,ms:passes.reduce((n,p,i)=>n+p.ms+(i?p.idle_ms:0),0),passes});
  }
  const means=Object.fromEntries([true,false].map(b=>[b?"baseline":"candidate",times.filter(t=>t.baseline===b).reduce((n,t)=>n+t.ms,0)/rounds]));
  e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  const result={date:new Date().toISOString(),sceneId:definition.id,values,dt,steps,rounds,dims,comparison,means_ms:means,times,errors};mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify({comparison,means_ms:means,errors,out}));
  assert.deepEqual(errors,[]);assert.ok(comparison.every(r=>(r as {maxUlp:number}).maxUlp<=allowedUlp),"Live outputs must match the requested ULP allowance");
 }finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await run();
