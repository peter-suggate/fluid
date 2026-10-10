/** At the live pressure-setup boundary, compare retained simulation authority
 * outputs with the original redundant authority's outputs. Every physical and
 * base phase/correction texel and every phi capacity word is compared bitwise.
 * node --import tsx tools/probe-uniform-authority-reuse.ts [--steps=36]
 * [--frames=1,12,36] [--scene=nbflip-figure-9-teaser] [--out=FILE]
 * The original pass is replayed before pressure consumes the outputs. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
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
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

type Frame={
 authority:{surfaceOnly:boolean;encode(e:GPUCommandEncoder,group:UniformDetailGroup,balance?:boolean):void};
 authorityGroup:UniformDetailGroup;authorityPhi:GPUBuffer;
 fields:{phase:GPUTexture;correction:GPUTexture};
 band:{encodePrepare(e:GPUCommandEncoder):void};
};
type RecordCopy={name:string;size:number;encode(e:GPUCommandEncoder,to:GPUBuffer):void};
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option("steps")??36),selected=new Set((option("frames")??`1,${Math.min(12,steps)},${steps}`).split(",").map(Number));
assert.ok(Number.isInteger(steps)&&steps>0&&[...selected].every(n=>Number.isInteger(n)&&n>=1&&n<=steps));
const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),scene=structuredClone(sceneDocument(definition));
assert.equal(definition.methodProfile?.methodId,method.id);assert.equal(scene.fluid.surfaceTension_N_m,0,"This probe exercises the no-capillarity reuse branch");
const quality=definition.methodProfile!.quality,values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
const out=option("out")??"artifacts/nbflip-ui-perf/authority-reuse.json";
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
const errors:string[]=[],rows:unknown[]=[],buffers=new Set<GPUBuffer>();
try{
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);device=managedGPUDevice(profile.device,{requireWorkerRealm:false});
 device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame;assert.ok(frame.authority.surfaceOnly);
 let index=0,pending:{records:RecordCopy[];before:GPUBuffer[];after:GPUBuffer[]}|undefined;
 const prepare=frame.band.encodePrepare.bind(frame.band);
 frame.band.encodePrepare=(encoder)=>{
  // advanceTo can encode several CFL-limited substeps. Capture the first
  // pressure setup of each selected advance without retaining duplicate
  // full-capacity readbacks for every substep.
  if(selected.has(index)&&pending===undefined){
   const phi=frame.authorityPhi;
   const records:RecordCopy[]=[{name:"authorityPhi",size:phi.size,encode:(e,to)=>e.copyBufferToBuffer(phi,0,to,0,phi.size)}];
   for(const name of ["phase","correction"] as const){
    const handle=frame.fields[name],field=uniformDetailField(handle);
    const textures=[...new Set(field?[field.storage.physical(field.handle),field.storage.baseOf(field.handle)!]:[handle])];
    for(const [bank,texture] of textures.entries()){
     const bytesPerRow=Math.ceil(texture.width*4/256)*256,rowsPerImage=texture.height;
     const size=bytesPerRow*rowsPerImage*texture.depthOrArrayLayers;
     records.push({name:`${name}-${bank}`,size,encode:(e,to)=>e.copyTextureToBuffer({texture},{buffer:to,bytesPerRow,rowsPerImage},[texture.width,texture.height,texture.depthOrArrayLayers])});
    }
   }
   const allocate=()=>records.map(r=>{const b=raw.createBuffer({size:r.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});buffers.add(b);return b;});
   const before=allocate(),after=allocate();records.forEach((r,i)=>r.encode(encoder,before[i]!));
   // This is the pass removed by the reuse optimization, at its old location.
   profile!.start();frame.authority.encode(encoder,frame.authorityGroup);
   records.forEach((r,i)=>r.encode(encoder,after[i]!));pending={records,before,after};
  }
  prepare(encoder);
 };
 for(index=1;index<=steps;index++){
  solver.advanceTo(index*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  if(selected.has(index)){
   assert.ok(pending,"The selected frame must reach pressure setup");
   const held=pending as {records:RecordCopy[];before:GPUBuffer[];after:GPUBuffer[]};pending=undefined;
   const banks:unknown[]=[],examples:unknown[]=[];let mismatches=0,words=0;
   for(let bank=0;bank<held.records.length;bank++){
    const a=held.before[bank]!,b=held.after[bank]!;await Promise.all([a.mapAsync(GPUMapMode.READ),b.mapAsync(GPUMapMode.READ)]);
    const left=new Uint32Array(a.getMappedRange()),right=new Uint32Array(b.getMappedRange());let differences=0;
    for(let k=0;k<left.length;k++)if(left[k]!==right[k]){differences++;if(examples.length<8)examples.push({bank:held.records[bank]!.name,word:k,before:left[k],after:right[k]});}
    words+=left.length;mismatches+=differences;banks.push({name:held.records[bank]!.name,words:left.length,mismatches:differences});
    a.unmap();b.unmap();a.destroy();b.destroy();buffers.delete(a);buffers.delete(b);
   }
   const passes=await profile.finish(),authority=passes.filter(p=>p.label==="NB-FLIP surface pressure authority");
   const row={frame:index,words,mismatches,examples,banks,replayedAuthority_ms:authority[0]?.ms};rows.push(row);console.log(JSON.stringify(row));
   assert.equal(mismatches,0,"Retained authority outputs must be identical at the original pass location");
  }
 }
 assert.deepEqual(errors,[]);
 const result={date:new Date().toISOString(),sceneId:definition.id,values,dt,steps,rows,errors};mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(result,null,2)+"\n");
}catch(error){mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify({sceneId:definition.id,values,dt,steps,rows,errors,error:String(error)},null,2)+"\n");throw error;}
finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
