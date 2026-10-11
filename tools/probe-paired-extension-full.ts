/** Full same-input extension replay: two scalar seed/sweep/hierarchy/publication
 * sequences versus shared paired seed/sweeps followed by both publications.
 * Both intermediate old and final new fields are compared as u32, including
 * metadata, physical/base textures and negative boundaries. Timings execute
 * the actual two-reference sequence; no 2x extrapolation. P2G retention copy
 * and unchanged particle snapshot/update are excluded here.
 */
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField, type UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformMixedExtension } from "../lib/methods/uniform/uniform-mixed-extension";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";


type Frame = {
 ownership:UniformMixedOwnership;extension:UniformMixedExtension;
 extensionGroups:readonly [UniformDetailGroup,UniformDetailGroup];stageScratch:GPUBuffer;
 pairedExtensionGroups:Map<GPUBuffer,unknown>;
 params:{extension:GPUBuffer};
 fields:{velocity:GPUTexture;velocityScratch:GPUTexture;departure:GPUTexture;phase:GPUTexture;negative:GPUBuffer;negativeScratch:GPUBuffer;negativeDeparture:GPUBuffer};
};

async function run():Promise<void>{
 const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
 const steps=Number(option("steps")??96),rounds=Number(option("rounds")??12);
 const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser");
 const quality=definition.methodProfile!.quality,scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides,timeStep:"sixtieth"});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 const out=option("out")??"artifacts/nbflip-ui-perf/paired-extension-full.json";
 let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
 const errors:string[]=[],buffers:GPUBuffer[]=[];
 try{
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  profile=new GPUPassProfile(raw);device=managedGPUDevice(profile.device,{requireWorkerRealm:false});
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame;
  for(let n=1;n<=steps;n++){solver.advanceTo(n*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(n%24===0||n===steps)console.log(JSON.stringify({phase:"advance",frame:n,time:n*dt}));}
  const f=frame.fields,ext=frame.extension;
  const sweepCounts=(option("sweeps")??String(values.extensionFrontSweeps??2)).split(",").map(Number);
  assert.ok(sweepCounts.every(n=>Number.isSafeInteger(n)&&n>=0));let sweeps=sweepCounts[0]!;
  const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC)});buffers.push(b);return b;};
  const companion=allocate(ext.companionBytesAt(frame.ownership.capacity.fineTiles));
  const common={phase:f.phase,output:f.velocityScratch,outputNegative:f.negativeScratch,scratch:{buffer:frame.stageScratch},params:frame.params.extension};
  const old=ext.bind({...common,physical:f.departure,negative:f.negativeDeparture});
  const paired={scratch:companion,physical:f.departure,negative:f.negativeDeparture};
  const pairCurrent=ext.bind({...common,physical:f.velocity,negative:f.negative,companion:paired});
  const pairOld=ext.bind({...common,physical:f.departure,negative:f.negativeDeparture,companion:paired});
  const field=uniformDetailField(f.velocityScratch);
  const textures=[...new Set(field?[field.storage.physical(field.handle),field.storage.baseOf(field.handle)!]:[f.velocityScratch])];
  const records=textures.map(texture=>{const bytesPerRow=Math.ceil(texture.width*16/256)*256;return{texture,bytesPerRow,rowsPerImage:texture.height,size:bytesPerRow*texture.height*texture.depthOrArrayLayers,extent:[texture.width,texture.height,texture.depthOrArrayLayers] as [number,number,number]};});
  const originals=records.map(r=>allocate(r.size)),originalNegative=allocate(f.negativeScratch.size);
  let e=raw.createCommandEncoder();records.forEach((r,i)=>e.copyTextureToBuffer({texture:r.texture},{buffer:originals[i]!,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},r.extent));e.copyBufferToBuffer(f.negativeScratch,0,originalNegative,0,originalNegative.size);raw.queue.submit([e.finish()]);
  const restore=(e:GPUCommandEncoder)=>{records.forEach((r,i)=>e.copyBufferToTexture({buffer:originals[i]!,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},{texture:r.texture},r.extent));e.copyBufferToBuffer(originalNegative,0,f.negativeScratch,0,originalNegative.size);};
  const first=(e:GPUCommandEncoder,pair:boolean)=>{if(pair){ext.preparePair(e,pairCurrent,sweeps);ext.finishPair(e,pairOld,sweeps,true);}else ext.encode(e,old,sweeps,false);};
  const second=(e:GPUCommandEncoder,pair:boolean)=>{if(pair)ext.finishPair(e,frame.extensionGroups,sweeps,false);else ext.encode(e,frame.extensionGroups,sweeps,true);};
  const capture=async(pair:boolean)=>{
   const snapshots=[0,1].map(()=>[...records.map(r=>allocate(r.size,true)),allocate(f.negativeScratch.size,true)]);
   const e=raw.createCommandEncoder();restore(e);first(e,pair);
   for(let stage=0;stage<2;stage++){
    if(stage)second(e,pair);
    records.forEach((r,i)=>e.copyTextureToBuffer({texture:r.texture},{buffer:snapshots[stage]![i]!,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},r.extent));e.copyBufferToBuffer(f.negativeScratch,0,snapshots[stage]!.at(-1)!,0,f.negativeScratch.size);
   }
   raw.queue.submit([e.finish()]);return Promise.all(snapshots.flat().map(async read=>{await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();return data;}));
  };
  const results:unknown[]=[];
  for(sweeps of sweepCounts){
  const reference=await capture(false),candidate=await capture(true);let words=0,mismatches=0;const examples:unknown[]=[];
  for(let bank=0;bank<reference.length;bank++){const a=reference[bank]!,b=candidate[bank]!;words+=a.length;for(let i=0;i<a.length;i++)if(a[i]!==b[i]){mismatches++;if(examples.length<12)examples.push({bank,word:i,reference:a[i],candidate:b[i]});}}
  console.log(JSON.stringify({phase:"parity",sweeps,words,mismatches,examples,errors}));
  const trials:{pair:boolean;ms:number;passes:Awaited<ReturnType<GPUPassProfile["finish"]>>}[]=[];
  for(let round=0;round<rounds+2;round++)for(const pair of round%2?[true,false]:[false,true]){
   e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
   profile.start();e=device.createCommandEncoder();first(e,pair);second(e,pair);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();const passes=await profile.finish();if(round>=2)trials.push({pair,ms:passes.reduce((s,p)=>s+p.ms,0),passes});
  }
  const average=(pair:boolean)=>{const t=trials.filter(t=>t.pair===pair);return t.reduce((s,t)=>s+t.ms,0)/t.length;};
  const result={date:new Date().toISOString(),sceneId:definition.id,values,dt,steps,rounds,sweeps,words,mismatches,examples,means_ms:{twoScalar:average(false),paired:average(true),saving:average(false)-average(true)},scope:"Actual complete two-scalar extension sequence versus paired seed/sweeps plus both full hierarchy/publication sequences. Excludes old-P2G retention copy and unchanged particle sampling.",trials,errors};
  results.push(result);console.log(JSON.stringify(result));
  assert.deepEqual(errors,[]);assert.equal(mismatches,0,"Both old and new complete extension fields must be exact");
  }
  mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(results.length===1?results[0]:{cases:results},null,2)+"\n");
 }finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await run();
