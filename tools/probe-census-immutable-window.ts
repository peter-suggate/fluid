/** Frozen-input immutable world-header census diagnostic. No physics policy changes.
 * node --import tsx tools/probe-census-immutable-window.ts --both --steps=120 --rounds=21
 * --both compares fine and coarse census kernels on one frozen solver state.
 */
import assert from "node:assert/strict";
import {experimentalImmutableWindow} from "./experimental-immutable-window";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import type { UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

type Dynamic={work:GPUBuffer;group:UniformDetailGroup;ownership:UniformMixedOwnership;pipelines:Map<string,GPUComputePipeline>};
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const both=process.argv.includes("--both"),fineOnly=process.argv.includes("--fine");
const steps=Number(option("steps")??120),rounds=Number(option("rounds")??21);
const root=resolve(option('root')??'/tmp/fluid-history-final'),load=(file:string)=>import(pathToFileURL(resolve(root,file)).href);
const [{resolveMethodValues},{sceneDocument},{getSceneDefinition},{managedGPUDevice},{uniformNarrowBandMethod:method},{sceneFluidLatticeDimensions,sceneFluidBoxPage},{uniformWindowDerivedWords}]=await Promise.all([
 load('lib/core/method-contract.ts'),load('lib/core/scene-definition.ts'),load('lib/core/scenes.ts'),load('lib/core/gpu-compilation-manager.ts'),load('lib/methods/uniform/uniform-narrow-band-method.ts'),load('lib/core/scene-lattice-dimensions.ts'),load('lib/methods/uniform/uniform-window-address.ts')]);
const definition=getSceneDefinition(option('scene')??'sparse-cm12-long-dam-break'),scene=structuredClone(sceneDocument(definition)),quality=definition.methodProfile?.quality??'balanced';
const policy=scene.container.walls==='none'?3:scene.container.top==='open'?2:1;
const header=uniformWindowDerivedWords(policy,policy===3?sceneFluidBoxPage(scene):[0,0,0],sceneFluidLatticeDimensions(scene,Number.MAX_SAFE_INTEGER).map((x:number)=>x/32));
const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile?.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
const errors:string[]=[],buffers:GPUBuffer[]=[];
try {
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);const sources=new WeakMap<GPUShaderModule,string>(),candidates=new Map<string,GPUComputePipeline>();
 const proxy=new Proxy(profile.device,{get:(target,key)=>{
  if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
  if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
   const pipeline=await target.createComputePipelineAsync(d);
   const entry=d.compute.entryPoint;
   if((entry===(fineOnly?"classify":"classifyCoarse")||(both&&entry==="classify"))&&sources.get(d.compute.module)?.includes("fn umClassifyOwner(")){
    const source=sources.get(d.compute.module);assert.ok(source);
    const code=experimentalImmutableWindow(source,header);assert.notEqual(code,source);
    writeFileSync(`/tmp/fluid-census-${entry}.wgsl`,code);
    const module=target.createShaderModule({label:"Immutable window census diagnostic",code});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
    candidates.set(entry,await target.createComputePipelineAsync({...d,compute:{...d.compute,module}}));
   }return pipeline;
  };
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});
 device=managedGPUDevice(proxy,{requireWorkerRealm:false}) as GPUDevice;device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(let n=1;n<=steps;n++){solver.advanceTo(n*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(n%12===0)console.log(JSON.stringify({phase:"advance",frame:n}));}
 const dynamic=(solver as unknown as {mixedDynamic:Dynamic}).mixedDynamic;
 for(const fine of both?[false,true]:[fineOnly]){
 const entry=fine?"classify":"classifyCoarse",candidate=candidates.get(entry);assert.ok(candidate);
 const stem=option("out")??"artifacts/dam-history-2026-10-15/census-immutable-window";
 const out=`${stem.replace(/\.json$/,"")}-${entry}.json`;
 const baseline=dynamic.pipelines.get(entry)!;
 const buffer=(read=false)=>{const b=raw.createBuffer({size:dynamic.work.size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.COPY_SRC)});buffers.push(b);return b;};
 const original=buffer();let e=raw.createCommandEncoder();e.copyBufferToBuffer(dynamic.work,0,original,0,original.size);raw.queue.submit([e.finish()]);
 const restore=(e:GPUCommandEncoder)=>e.copyBufferToBuffer(original,0,dynamic.work,0,original.size);
 const encode=(e:GPUCommandEncoder,isBaseline:boolean)=>{
  const pass=e.beginComputePass({label:isBaseline?"Census baseline":"Census immutable window"});pass.setBindGroup(0,dynamic.ownership.bindGroup);pass.setBindGroup(1,dynamic.group.group);pass.setPipeline(isBaseline?baseline:candidate!);
  const n=fine?Math.min(4096,dynamic.ownership.capacity.tiles):Math.ceil(dynamic.ownership.capacity.tiles/64),x=dynamic.ownership.dispatchX;pass.dispatchWorkgroups(Math.min(n,x),Math.ceil(n/x));pass.end();
 };
 const capture=async(isBaseline:boolean)=>{const read=buffer(true),e=raw.createCommandEncoder();restore(e);encode(e,isBaseline);e.copyBufferToBuffer(dynamic.work,0,read,0,read.size);raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange().slice(0));read.unmap();return words;};
 const a=await capture(true),b=await capture(false);let mismatches=0;const first:unknown[]=[];
 for(let i=0;i<a.length;i++)if(a[i]!==b[i]){mismatches++;if(first.length<8)first.push({word:i,baseline:a[i],candidate:b[i]});}
 e=raw.createCommandEncoder();for(let n=0;n<4;n++){restore(e);encode(e,true);restore(e);encode(e,false);}raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
 const timings:{baseline:boolean;ms:number}[]=[];
 for(let n=0;n<rounds;n++)for(const isBaseline of n%2?[false,true]:[true,false]){
  e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  profile.start();e=device.createCommandEncoder();encode(e,isBaseline);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  timings.push({baseline:isBaseline,ms:(await profile.finish()).reduce((s,p)=>s+p.ms,0)});
 }
 const mean=(v:boolean)=>timings.filter(t=>t.baseline===v).reduce((s,t)=>s+t.ms,0)/rounds;
 const result={entry,dt,values,steps,rounds,words:a.length,mismatches,first,baseline_ms:mean(true),candidate_ms:mean(false),saved_ms:mean(true)-mean(false),timings,errors};mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));assert.equal(mismatches,0);assert.deepEqual(errors,[]);
 e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
 }
}finally{solver?.destroy();await device?.queue.onSubmittedWorkDone();for(const b of buffers)b.destroy();profile?.destroy();device?.destroy();}
