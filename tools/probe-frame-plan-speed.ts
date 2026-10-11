/** Frozen proof of reusing the mid-frame support planes at the no-body tail.
 * The normal simulation observes that phi/ownership-derived support planes
 * remain identical across every tail plan, then compares a speed-only seed. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
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
import { uniformDetailPick } from "../lib/methods/uniform/uniform-detail-fields";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

function replaceExact(source:string,from:string,to:string):string{
 assert.equal(source.split(from).length-1,1,from);return source.replace(from,to);
}
function speedOnlySource(source:string):string{
 // The occupancy section is bounded by its semantic comments/statements,
 // before runtime detail accessors are allowed to obscure individual loads.
 const begin=source.indexOf("  // A 4h owner's origin and corners");assert.ok(begin>=0);
 const stop="  if(occupied){atomicOr(&seeded[slot],3u);}";const end=source.indexOf(stop,begin);assert.ok(end>begin);
 source=source.slice(0,begin)+source.slice(end+stop.length);
 source=replaceExact(source,"atomicStore(&seeded[lane],0u);","");
 const store=/ if\(owner\.width!=0u&&owner\.lane==0u\)\{atomicStore\(&umSupport\[owner\.tile\],atomicLoad\(&seeded\[slot\]\)\);\}/;
 assert.equal([...source.matchAll(new RegExp(store,"g"))].length,1);
 return source.replace(store,"");
}
class Capture{
 readonly sources=new WeakMap<GPUShaderModule,string>();readonly pipelines=new WeakMap<GPUComputePipeline,GPUComputePipelineDescriptor>();readonly device:GPUDevice;
 constructor(readonly raw:GPUDevice){this.device=new Proxy(raw,{get:(target,key)=>{
  if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);this.sources.set(m,d.code);return m;};
  if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{const p=await target.createComputePipelineAsync(d);this.pipelines.set(p,d);return p;};
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});}
}
const distribution=(values:number[])=>{const sorted=[...values].sort((a,b)=>a-b),mean=values.reduce((a,b)=>a+b,0)/values.length;return {mean,median:(sorted[Math.floor((sorted.length-1)/2)]!+sorted[Math.ceil((sorted.length-1)/2)]!)/2,min:sorted[0],max:sorted.at(-1),stddev:Math.sqrt(values.reduce((sum,value)=>sum+(value-mean)**2,0)/values.length)};};
async function run():Promise<void>{
 const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
 const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),quality=definition.methodProfile!.quality,scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s,steps=Number(option("steps")??96),rounds=Number(option("rounds")??12);
 const out=option("out")??"artifacts/nbflip-ui-perf/frame-plan-speed.json";
 let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
 const buffers:GPUBuffer[]=[],errors:string[]=[];
 try{
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  profile=new GPUPassProfile(raw);const capture=new Capture(profile.device);device=managedGPUDevice(capture.device,{requireWorkerRealm:false});
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  const frame=(solver as any).mixedFrame,plan=frame.plan,ownership=frame.ownership,flip=frame.narrowBandFlip;
  const original=uniformDetailPick(plan.pipelines.get("seed")!),descriptor=capture.pipelines.get(original);assert.ok(descriptor);
  const integrated=plan.pipelines.has("speed");
  const code=integrated?capture.sources.get(descriptor.compute.module)!:speedOnlySource(capture.sources.get(descriptor.compute.module)!);
  const module=raw.createShaderModule({label:"Frame tail speed only",code});const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(messages.length){writeFileSync("/tmp/fluid-frame-plan-speed.wgsl",code);throw new Error(messages.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));}
  const candidate=integrated?uniformDetailPick(plan.pipelines.get("speed")):await raw.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module}});
  const marker=await raw.createComputePipelineAsync({layout:"auto",compute:{module:raw.createShaderModule({code:"@compute @workgroup_size(1) fn marker() {}"}),entryPoint:"marker"}});
  const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC)});buffers.push(b);return b;};
  const compareModule=raw.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> expected:array<u32>;
@group(0) @binding(1) var<storage,read> actual:array<u32>;
@group(0) @binding(2) var<storage,read_write> stats:array<atomic<u32>>;
@compute @workgroup_size(256) fn compare(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<arrayLength(&expected);i+=65536u){if(expected[i]!=actual[i]){atomicAdd(&stats[0],1u);atomicMin(&stats[1],i);}}
}`});
  const compare=await raw.createComputePipelineAsync({layout:"auto",compute:{module:compareModule,entryPoint:"compare"}});
  const stats=allocate(8),read=allocate(8,true),supportBytes=ownership.support.size,topologyBytes=ownership.topology.size;
  const planes=16*ownership.capacity.tiles,heldPlanes=allocate(planes),heldTopology=allocate(topologyBytes);
  const group=(expected:GPUBuffer,actual:GPUBuffer,size:number)=>raw.createBindGroup({layout:compare.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:expected,size}},{binding:1,resource:{buffer:actual,size}},{binding:2,resource:{buffer:stats}}]});
  const comparePlanes=group(heldPlanes,ownership.support,planes),compareTopology=group(heldTopology,ownership.topology,topologyBytes);
  const verify=(encoder:GPUCommandEncoder,groups:GPUBindGroup[])=>{const pass=encoder.beginComputePass({label:"Frozen plan comparison"});pass.setPipeline(compare);for(const group of groups){pass.setBindGroup(0,group);pass.dispatchWorkgroups(256);}pass.end();};
  const results=async()=>{const e=raw.createCommandEncoder();e.copyBufferToBuffer(stats,0,read,0,8);raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const data=[...new Uint32Array(read.getMappedRange())];read.unmap();return {mismatches:data[0]!,firstMismatch:data[1]!};};
  raw.queue.writeBuffer(stats,0,new Uint32Array([0,0xffffffff]));
  let afterUpdate=false,tailCount=0,havePlan=false,latestPolicy:unknown;
  const update=flip.update.bind(flip),full=plan.encode.bind(plan),refresh=plan.encodeSpeed?.bind(plan);
  flip.update=(encoder:GPUCommandEncoder)=>{update(encoder);afterUpdate=true;};
  plan.encode=(encoder:GPUCommandEncoder,policy:unknown)=>{
   if(afterUpdate){
    assert.ok(havePlan,"tail needs a preceding support plan");assert.deepEqual(policy,latestPolicy,"support policy must stay unchanged within frame");
    // Saved immediately after the latest full plan in this frame. Any width
    // change or support mutation before the tail is visible to this observer.
    verify(encoder,[comparePlanes,compareTopology]);full(encoder,policy);verify(encoder,[comparePlanes,compareTopology]);tailCount++;afterUpdate=false;
   }else{full(encoder,policy);encoder.copyBufferToBuffer(ownership.support,0,heldPlanes,0,planes);encoder.copyBufferToBuffer(ownership.topology,0,heldTopology,0,topologyBytes);havePlan=true;latestPolicy=policy;}
  };
  if(refresh)plan.encodeSpeed=plan.encode;
  console.log(JSON.stringify({phase:"source-preflight",integrated}));
  for(let n=1;n<=steps;n++){solver.advanceTo(n*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(n%24===0||n===steps)console.log(JSON.stringify({phase:"advance",frame:n}));}
  const observation={tails:tailCount,words:tailCount*2*(planes+topologyBytes)/4,...await results()};console.log(JSON.stringify({phase:"tail-observation",...observation}));assert.equal(observation.mismatches,0);assert.equal(tailCount,steps);
  plan.encode=full;if(refresh)plan.encodeSpeed=refresh;flip.update=update;
  const saved=allocate(supportBytes),reference=allocate(supportBytes),comparison=group(reference,ownership.support,supportBytes);
  let e=raw.createCommandEncoder();e.copyBufferToBuffer(ownership.support,0,saved,0,supportBytes);raw.queue.submit([e.finish()]);
  const speed=(encoder:GPUCommandEncoder)=>{
   if(refresh){refresh(encoder,latestPolicy);return;}
   encoder.clearBuffer(ownership.support,ownership.capacity.tiles*16,64);
   const pass=encoder.beginComputePass({label:"Uniform tail speed only"});pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,plan.group.group);pass.setBindGroup(2,plan.extendedGroup.group);ownership.dispatchAllCounted(pass,candidate);pass.end();
  };
  const restore=(encoder:GPUCommandEncoder)=>encoder.copyBufferToBuffer(saved,0,ownership.support,0,supportBytes);
  raw.queue.writeBuffer(stats,0,new Uint32Array([0,0xffffffff]));e=raw.createCommandEncoder();restore(e);full(e,latestPolicy);e.copyBufferToBuffer(ownership.support,0,reference,0,supportBytes);restore(e);speed(e);verify(e,[comparison]);raw.queue.submit([e.finish()]);
  const parity={words:supportBytes/4,...await results()};console.log(JSON.stringify({phase:"parity",...parity}));assert.equal(parity.mismatches,0);
  if(process.argv.includes("--wait-for-gpu-quiet")){let seconds=0;for(;;){const others=execFileSync("ps",["-Ao","pid=,comm=,args="],{encoding:"utf8"}).split("\n").filter(line=>{const row=line.match(/^\s*(\d+)\s+(\S+)\s+(.*)$/);return row&&Number(row[1])!==process.pid&&/(?:^|\/)node$/.test(row[2]!)&&/tools\/(?:probe|benchmark)[^\s]*\.ts/.test(row[3]!);});if(!others.length)break;if(seconds++%60===0)console.log(JSON.stringify({phase:"waiting-before-timing",others}));await new Promise(resolve=>setTimeout(resolve,1000));}}
  const trials:{speedOnly:boolean;ms:number;compute_ms:number}[]=[];
  for(let round=0;round<rounds+2;round++)for(const speedOnly of round%2?[true,false]:[false,true]){
   e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
   profile.start();e=device.createCommandEncoder();const start=e.beginComputePass({label:"Tail plan interval start"});start.setPipeline(marker);start.dispatchWorkgroups(1);start.end();if(speedOnly)speed(e);else full(e,latestPolicy);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();const passes=await profile.finish();if(round>=2)trials.push({speedOnly,ms:passes.reduce((sum,p)=>sum+p.ms+p.idle_ms,0),compute_ms:passes.reduce((sum,p)=>sum+p.ms,0)});
  }
  const originalTimes=distribution(trials.filter(t=>!t.speedOnly).map(t=>t.ms)),candidateTimes=distribution(trials.filter(t=>t.speedOnly).map(t=>t.ms));
  const saving=distribution(Array.from({length:rounds},(_,r)=>{const pair=trials.slice(2*r,2*r+2);return pair.find(t=>!t.speedOnly)!.ms-pair.find(t=>t.speedOnly)!.ms;}));
  const report={date:new Date().toISOString(),sceneId:definition.id,values,dt,steps,rounds,integrated,observation,parity,distributions_ms:{original:originalTimes,speedOnly:candidateTimes,saving},trials,errors};
  mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify(report));assert.deepEqual(errors,[]);
 }finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await run();
