/** Frozen-input parallel census diagnostic. No physics policy changes.
 * node --import tsx tools/probe-census-coarse.ts [--steps=96] [--rounds=12]
 * --fine instead reduces shared atomics once per SIMD subgroup.
 * --both tests each change separately using one solver fixture.
 */
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
import type { UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

function parallelFine(code:string):string {
 const at=code.indexOf("@compute @workgroup_size(64) fn classify("),end=code.indexOf("// Single-owner (4h) tiles",at);assert.ok(at>=0&&end>at);
 let source=code.slice(at,end);
 const replace=(a:string,b:string)=>{assert.equal(source.split(a).length,2,`Expected one ${a}`);source=source.replace(a,b);};
 replace("for(var a=0u;a<3u;a++){atomicMin(&tileLow[a],c.low[a]);atomicMax(&tileHigh[a],c.high[a]);}","for(var a=0u;a<3u;a++){let lo=subgroupMin(c.low[a]);let hi=subgroupMax(c.high[a]);if(subgroupElect()){atomicMin(&tileLow[a],lo);atomicMax(&tileHigh[a],hi);}}");
 replace("for(var k=0u;k<6u;k++){atomicMin(&tileGap[k],c.gap[k]);atomicMin(&tileReach[k],c.reach[k]);}","for(var k=0u;k<6u;k++){let gap=subgroupMin(c.gap[k]);let reach=subgroupMin(c.reach[k]);if(subgroupElect()){atomicMin(&tileGap[k],gap);atomicMin(&tileReach[k],reach);}}");
 replace("atomicOr(&mixedTile,c.flags);","let flags=subgroupOr(c.flags);if(subgroupElect()){atomicOr(&mixedTile,flags);}");
 replace("for(var k=0u;k<2u;k++){if(c.crossing[k]!=0u){atomicOr(&tileCrossing[k],c.crossing[k]);}}","let crossing=subgroupOr(c.crossing);if(subgroupElect()){for(var k=0u;k<2u;k++){if(crossing[k]!=0u){atomicOr(&tileCrossing[k],crossing[k]);}}}");
 replace("if((atomicLoad(&mixedTile)&2u)!=0u&&policy.step.y>0.0){atomicMax(&tileError,bitcast<u32>(umResolutionError(tile,lane,width,atomicLoad(&tileWide))));}","var ownError=0u;if((atomicLoad(&mixedTile)&2u)!=0u&&policy.step.y>0.0){ownError=bitcast<u32>(umResolutionError(tile,lane,width,atomicLoad(&tileWide)));}let error=subgroupMax(ownError);if(subgroupElect()){atomicMax(&tileError,error);}");
 replace("if((atomicLoad(&mixedTile)&2u)!=0u&&thinWanted()){let deep=umDepthSample(tile,lane);for(var k=0u;k<2u;k++){atomicMax(&tileDeep[k],deep[k]);}}","var ownDeep=vec2u(0u);if((atomicLoad(&mixedTile)&2u)!=0u&&thinWanted()){ownDeep=umDepthSample(tile,lane);}let deep=subgroupMax(ownDeep);if(subgroupElect()){for(var k=0u;k<2u;k++){atomicMax(&tileDeep[k],deep[k]);}}");
 return (code.includes("enable subgroups;")?"":"enable subgroups;\n")+code.slice(0,at)+source+code.slice(end);
}

function parallelCoarse(code:string):string {
 const first=code.indexOf("@compute @workgroup_size(64) fn classifyCoarse("), end=code.indexOf("// The neighbour criteria",first);
 assert.ok(first>=0&&end>first);
 let replacement=/* wgsl */`
var<workgroup> pcClass:array<TileClass,4>;
var<workgroup> pcError:array<atomic<u32>,4>;
var<workgroup> pcDepth:array<atomic<u32>,8>;
@compute @workgroup_size(64) fn classifyCoarse(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let slot=lane>>4u;let local=lane&15u;
 for(var base=4u*(group.x+umDispatchX*group.y);base<umCounts.y;base+=4u*groups.x*groups.y){
  if(lane==0u){resetGroupCensus();}
  if(local==0u){atomicStore(&pcError[slot],0u);atomicStore(&pcDepth[2u*slot],0u);atomicStore(&pcDepth[2u*slot+1u],0u);}
  workgroupBarrier();
  let index=base+slot;
  if(index<umCounts.y&&local==0u){var c=umEmptyClass();umClassifyOwner(umTopology[UM_TILES+umCounts.x+index],4u,0u,&c);pcClass[slot]=c;}
  workgroupBarrier();
  if(index<umCounts.y&&(pcClass[slot].flags&2u)!=0u){
   let tile=umTopology[UM_TILES+umCounts.x+index];
   if(policy.step.y>0.0&&local<8u){atomicMax(&pcError[slot],bitcast<u32>(umResolutionError(tile,local,4u,0u)));}
   if(thinWanted()){
    var deep=vec2u(0u);for(var i=local;i<64u;i+=16u){deep=max(deep,umDepthSample(tile,i));}
    atomicMax(&pcDepth[2u*slot],deep.x);atomicMax(&pcDepth[2u*slot+1u],deep.y);
   }
  }
  workgroupBarrier();
  if(index<umCounts.y&&local==0u){
   var c=pcClass[slot];c.error=atomicLoad(&pcError[slot]);c.deep=vec2u(atomicLoad(&pcDepth[2u*slot]),atomicLoad(&pcDepth[2u*slot+1u]));
   umFinishTile(umTopology[UM_TILES+umCounts.x+index],4u,c);
  }
  workgroupBarrier();if(lane==0u){flushGroupCensus();}workgroupBarrier();
 }
}
`;
 if(code.includes("fn UM_TILES()"))replacement=replacement.replace(/\bUM_TILES\b/g,"UM_TILES()");
 if(code.includes("struct UMSceneCounts"))replacement=replacement.replace(/\bumCounts\./g,"umCounts.work.");
 return code.slice(0,first)+replacement+code.slice(end);
}

type Dynamic={work:GPUBuffer;group:UniformDetailGroup;ownership:UniformMixedOwnership;pipelines:Map<string,GPUComputePipeline>};
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const both=process.argv.includes("--both"),fineOnly=process.argv.includes("--fine");
const steps=Number(option("steps")??96),rounds=Number(option("rounds")??12);
const definition=getSceneDefinition("nbflip-figure-9-teaser"),scene=structuredClone(sceneDocument(definition)),quality=definition.methodProfile!.quality;
const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
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
    const code=entry==="classify"?parallelFine(source):parallelCoarse(source);
    writeFileSync(`/tmp/fluid-census-${entry}.wgsl`,code);
    const module=target.createShaderModule({label:"Parallel census diagnostic",code});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
    candidates.set(entry,await target.createComputePipelineAsync({...d,compute:{...d.compute,module}}));
   }return pipeline;
  };
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});
 device=managedGPUDevice(proxy,{requireWorkerRealm:false});device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(let n=1;n<=steps;n++){solver.advanceTo(n*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(n%12===0)console.log(JSON.stringify({phase:"advance",frame:n}));}
 const dynamic=(solver as unknown as {mixedDynamic:Dynamic}).mixedDynamic;
 for(const fine of both?[false,true]:[fineOnly]){
 const entry=fine?"classify":"classifyCoarse",candidate=candidates.get(entry);assert.ok(candidate);
 const out=option("out")??`artifacts/nbflip-ui-perf/${fine?"fine":"coarse"}-census-96.json`;
 const baseline=dynamic.pipelines.get(entry)!;
 const buffer=(read=false)=>{const b=raw.createBuffer({size:dynamic.work.size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.COPY_SRC)});buffers.push(b);return b;};
 const original=buffer();let e=raw.createCommandEncoder();e.copyBufferToBuffer(dynamic.work,0,original,0,original.size);raw.queue.submit([e.finish()]);
 const restore=(e:GPUCommandEncoder)=>e.copyBufferToBuffer(original,0,dynamic.work,0,original.size);
 const encode=(e:GPUCommandEncoder,isBaseline:boolean)=>{
  const pass=e.beginComputePass({label:isBaseline?"Coarse census baseline":"Coarse census parallel"});pass.setBindGroup(0,dynamic.ownership.bindGroup);pass.setBindGroup(1,dynamic.group.group);pass.setPipeline(isBaseline?baseline:candidate!);
  const n=fine?Math.min(4096,dynamic.ownership.capacity.tiles):Math.ceil(dynamic.ownership.capacity.tiles/(isBaseline?64:4)),x=dynamic.ownership.dispatchX;pass.dispatchWorkgroups(Math.min(n,x),Math.ceil(n/x));pass.end();
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
