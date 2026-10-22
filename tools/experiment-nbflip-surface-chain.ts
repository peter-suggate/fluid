/** Same-state replay of derived gather -> metric -> final phi publication.
 * Restores every captured writable channel and compares every word, including
 * physical phi textures and revision state. Restores/comparisons are untimed.
 */
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {createHash} from "node:crypto";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {dirname,resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {resolveMethodValues} from "../lib/core/method-contract";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from "../lib/harness/node-dawn-provider";
import {uniformNarrowBandMethod as method} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
for(const arg of process.argv.slice(2))assert.match(arg,/^--(?:frames|repeats|out|scene|dt)=.+$/);
const frames=(option("frames")??"80,120").split(",").map(Number).sort((a,b)=>a-b),repeats=Number(option("repeats")??7);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>=3);
const out=resolve(option("out")??"artifacts/surface-chain.json");mkdirSync(dirname(out),{recursive:true});
const variants=["baseline","control","compactCanonical"];
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal","disable-dawn-features=timestamp_quantization"]).requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:(["timestamp-query","float32-filterable","subgroups"] as GPUFeatureName[]).filter(f=>adapter.features.has(f)),requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const errors:string[]=[];raw.addEventListener("uncapturederror",event=>{event.preventDefault();errors.push(event.error.message);console.error(event.error.message);});
const compareModule=raw.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> live:array<u32>;
@group(0) @binding(1) var<storage,read> reference:array<u32>;
@group(0) @binding(2) var<storage,read_write> mismatches:array<atomic<u32>>;
struct Params{range:vec4u,skip:vec4u,revisions:vec4u};
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> examples:array<vec4u>;
@group(0) @binding(5) var<storage,read> initial:array<u32>;
fn epochClass(x:u32)->u32{return select(select(1u,2u,x>=0xfffffffeu),0u,x==0u);}
@compute @workgroup_size(256) fn compare(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u){
 let p=params.range;for(var i=id.x;i<p.z;i+=256u*groups.x){
  if(i>=params.skip.x&&i<params.skip.y){continue;}
  let a=live[p.x+i];let b=reference[p.x+i];var wrong=a!=b;
  if(i>=params.revisions.x&&i<params.revisions.y){
   if(wrong){atomicAdd(&mismatches[p.y+p.w],1u);}
   let old=initial[p.x+i];wrong=(a==old)!=(b==old)||epochClass(a)!=epochClass(b)||a<old||b<old;
  }
  if(wrong){let j=atomicAdd(&mismatches[p.w],1u);if(j<64u){examples[64u*p.w+j]=vec4u(i,b,a,0u);}}
 }
}`});
const compare=await raw.createComputePipelineAsync({layout:"auto",compute:{module:compareModule,entryPoint:"compare"}});
const groupInfo=new WeakMap<GPUBindGroup,{entries:GPUBindGroupEntry[];layout:GPUBindGroupLayout}>();
const layouts=new WeakMap<GPUBindGroupLayout,GPUBindGroupLayoutEntry[]>(),textures=new WeakMap<GPUTextureView,GPUTexture>();
const pipelineInfo=new WeakMap<GPUComputePipeline,{name:string;derived:boolean;extra?:Record<string,GPUComputePipeline>}>();
const moduleSources=new WeakMap<GPUShaderModule,string>();const shaderHashes:Record<string,string>={};let revisionRange=[0,0];
type Binding={group:GPUBindGroup;dynamic:number[]};
type Command={pipeline:GPUComputePipeline;name:string;bindings:Map<number,Binding>;size:[number,number,number]};
type Part={name:string;bytes:number;at:number;buffer?:GPUBuffer;offset?:number;texture?:GPUTexture;row?:number;height?:number;depth?:number};
type Capture={frame:number;commands:Command[];parts:Part[];total:number;initial:GPUBuffer;saved:GPUBuffer;live:GPUBuffer;resources:GPUBuffer[]};
type Pending=Capture & {times:GPUBuffer;result:GPUBuffer;queries:GPUQuerySet;order:number[][];examples:GPUBuffer};
let frame=0,capture:Capture|undefined;const captured=new Set<number>(),pending:Pending[]=[];
const apply=(pass:GPUComputePassEncoder,c:Command,pipeline=c.pipeline)=>{pass.setPipeline(pipeline);for(const [i,b] of c.bindings)pass.setBindGroup(i,b.group,b.dynamic);pass.dispatchWorkgroups(...c.size);};
const copy=(e:GPUCommandEncoder,c:Capture,b:GPUBuffer,restore=false)=>{
 for(const p of c.parts){
  if(p.buffer){if(restore)e.copyBufferToBuffer(b,p.at,p.buffer,p.offset??0,p.bytes);else e.copyBufferToBuffer(p.buffer,p.offset??0,b,p.at,p.bytes);}
  else {const texture={texture:p.texture!},buffer={buffer:b,offset:p.at,bytesPerRow:p.row!,rowsPerImage:p.height!},extent={width:p.texture!.width,height:p.height!,depthOrArrayLayers:p.depth!};if(restore)e.copyBufferToTexture(buffer,texture,extent);else e.copyTextureToBuffer(texture,buffer,extent);}
 }
};
function start(e:GPUCommandEncoder,command:Command):Capture{
 const parts:Part[]=[];const seen=new Set<GPUBuffer|GPUTexture>();let total=0;
 for(const [index,binding] of command.bindings){
  const info=groupInfo.get(binding.group)!;assert.ok(info);const entries=layouts.get(info.layout)!;
  for(const entry of info.entries){
   const layout=entries.find(x=>x.binding===entry.binding)!;
   if(index===0&&layout.buffer?.type==="storage"||index===1&&[2,3,4].includes(entry.binding)){
    const r=entry.resource as GPUBufferBinding;if(seen.has(r.buffer))continue;seen.add(r.buffer);
    const bytes=r.size??r.buffer.size-(r.offset??0);total=Math.ceil(total/256)*256;
    parts.push({name:`g${index}b${entry.binding}:${r.buffer.label}`,bytes,at:total,buffer:r.buffer,offset:r.offset??0});total+=bytes;
   }else if(index===1&&[7,14,39,46].includes(entry.binding)){
    const t=textures.get(entry.resource as GPUTextureView);assert.ok(t,`texture ${entry.binding}`);if(seen.has(t))continue;seen.add(t);assert.equal(t.format,"r32float");
    const row=Math.ceil(t.width*4/256)*256,bytes=row*t.height*t.depthOrArrayLayers;total=Math.ceil(total/256)*256;
    parts.push({name:`g1b${entry.binding}:${t.label}`,bytes,at:total,texture:t,row,height:t.height,depth:t.depthOrArrayLayers});total+=bytes;
   }
  }
 }
 assert.ok(total<raw.limits.maxStorageBufferBindingSize);const resources:GPUBuffer[]=[];
 const make=()=>{const b=raw.createBuffer({size:total,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|GPUBufferUsage.STORAGE});resources.push(b);return b;};
 const c={frame,commands:[],parts,total,initial:make(),saved:make(),live:make(),resources};copy(e,c,c.initial);return c;
}
function finish(e:GPUCommandEncoder,c:Capture){
 assert.deepEqual(c.commands.map(c=>c.name),["arenaGather","arenaDistance","arenaFar","arenaStore"]);
 const store=c.commands[3]!;
 copy(e,c,c.saved);
 const count=variants.length*c.parts.length,queryCount=2*repeats*variants.length;
 const make=(size:number,usage:number)=>{const b=raw.createBuffer({size,usage});c.resources.push(b);return b;};
 const diff=make(8*count,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
 const result=make(8*count,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 const exampleStore=make(64*16*count,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
 const examples=make(64*16*count,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 const queries=raw.createQuerySet({type:"timestamp",count:queryCount});
 const resolved=make(8*queryCount,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);
 const times=make(8*queryCount,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 const comparisons=variants.map((_,v)=>c.parts.map((p,i)=>{
  const flip=(solver as unknown as {mixedFrame:{narrowBandFlip:{brickWord:number;address:{blocks:number}}}}).mixedFrame.narrowBandFlip;
  const skip=variants[v]==="compactCanonical"&&p.name.startsWith("g1b2:")?[flip.brickWord+4,flip.brickWord+4+64*flip.address.blocks]:[0,0];
  const revisions=p.name.startsWith("g1b2:")?revisionRange:[0,0];
  const params=make(48,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);raw.queue.writeBuffer(params,0,new Uint32Array([p.at/4,count,p.bytes/4,v*c.parts.length+i,...skip,0,0,...revisions,0,0]));
  return {n:Math.min(4096,Math.ceil(p.bytes/1024)),group:raw.createBindGroup({layout:compare.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:c.live}},{binding:1,resource:{buffer:c.saved}},{binding:2,resource:{buffer:diff}},{binding:3,resource:{buffer:params}},{binding:4,resource:{buffer:exampleStore}},{binding:5,resource:{buffer:c.initial}}]})};
 }));
 const order:number[][]=[];
 for(let r=0;r<repeats;r++){
  const row=variants.map((_,i)=>(i+r)%variants.length);if(r%2)row.reverse();order.push(row);
  for(let k=0;k<row.length;k++){
   const v=row[k]!,q=2*(r*variants.length+k);copy(e,c,c.initial,true);
   const pass=e.beginComputePass({label:`surface chain ${variants[v]}`,timestampWrites:{querySet:queries,beginningOfPassWriteIndex:q,endOfPassWriteIndex:q+1}});
   if(v<2)for(const cmd of c.commands)apply(pass,cmd);
   else{
    const entries=pipelineInfo.get(store.pipeline)!.extra!;const gather=pipelineInfo.get(c.commands[0]!.pipeline)!.extra!.arenaGatherPublication!;
    assert.deepEqual([gather,c.commands[2]!.pipeline,entries.arenaDistanceCanonical!,entries.arenaPublishFinish!].map(p=>pipelineInfo.get(p)?.name),["arenaGather","arenaFar","arenaDistanceCanonical","arenaPublishFinish"]);
    apply(pass,c.commands[0]!,gather);apply(pass,c.commands[2]!);apply(pass,store,entries.arenaDistanceCanonical);apply(pass,store,entries.arenaPublishFinish);
   }
   pass.end();copy(e,c,c.live);
   const verify=e.beginComputePass();verify.setPipeline(compare);for(const cmp of comparisons[v]!){verify.setBindGroup(0,cmp.group);verify.dispatchWorkgroups(cmp.n);}verify.end();
  }
 }
 copy(e,c,c.saved,true);e.resolveQuerySet(queries,0,queryCount,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,8*queryCount);e.copyBufferToBuffer(diff,0,result,0,8*count);
 e.copyBufferToBuffer(exampleStore,0,examples,0,64*16*count);
 pending.push({...c,times,result,queries,order,examples});captured.add(frame);
}
const intercepted=new Proxy(raw,{get(target,key){
 if(key==="createTexture")return(d:GPUTextureDescriptor)=>{
  const texture=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});const create=texture.createView.bind(texture);
  texture.createView=(d?:GPUTextureViewDescriptor)=>{const view=create(d);textures.set(view,texture);return view;};return texture;
 };
 if(key==="createBindGroupLayout")return(d:GPUBindGroupLayoutDescriptor)=>{const layout=target.createBindGroupLayout(d);layouts.set(layout,[...d.entries]);return layout;};
 if(key==="createBindGroup")return(d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);groupInfo.set(group,{entries:[...d.entries],layout:d.layout});return group;};
 if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);moduleSources.set(module,d.code);return module;};
 if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d);const name=d.compute.entryPoint??"";const extra:Record<string,GPUComputePipeline>={};
  if(name==="arenaGather"&&d.compute.constants?.nbArenaDerived===1){
   extra.arenaGatherPublication=await target.createComputePipelineAsync({...d,label:"surface gather and publication init",compute:{...d.compute,constants:{...d.compute.constants,nbArenaClearPublication:1}}});
  }
  if(name==="arenaStore"){
   const source=moduleSources.get(d.compute.module)!;assert.ok(source.includes("fn arenaDistanceCanonical")&&source.includes("fn arenaPublishFinish"));
   const revisionStart=Number(source.match(/const NB_REUSE_LIVE:u32=(\d+)u/)?.[1]),revisionCount=Number(source.match(/const NB_REUSE_REGIONS:u32=(\d+)u/)?.[1]);assert.ok(revisionStart>0&&revisionCount>0);revisionRange=[revisionStart,revisionStart+revisionCount];
   shaderHashes.source=createHash("sha256").update(source).digest("hex");writeFileSync(`${out}.wgsl`,source);
   for(const entryPoint of ["arenaDistanceCanonical","arenaPublishFinish"])extra[entryPoint]=await target.createComputePipelineAsync({...d,label:`surface compact ${entryPoint}`,compute:{...d.compute,entryPoint}});
  }
  for(const [entry,p] of Object.entries(extra))pipelineInfo.set(p,{name:entry==="arenaGatherPublication"?"arenaGather":entry,derived:entry==="arenaGatherPublication"});
  pipelineInfo.set(pipeline,{name,derived:d.compute.constants?.nbArenaDerived===1,extra});return pipeline;
 };
 if(key==="createCommandEncoder")return(d?:GPUCommandEncoderDescriptor)=>{
  const encoder=target.createCommandEncoder(d);
  return new Proxy(encoder,{get(e,key){
   if(key==="beginComputePass")return(descriptor?:GPUComputePassDescriptor)=>{
    let pass=e.beginComputePass(descriptor),pipeline:GPUComputePipeline|undefined;const bindings=new Map<number,Binding>();
    const rebind=()=>{if(pipeline)pass.setPipeline(pipeline);for(const [i,b]of bindings)pass.setBindGroup(i,b.group,b.dynamic);};
    return new Proxy(pass,{get(_p,key){
     if(key==="setPipeline")return(value:GPUComputePipeline)=>{pipeline=value;pass.setPipeline(value);};
     if(key==="setBindGroup")return(i:number,group:GPUBindGroup,dynamic?:Iterable<number>|Uint32Array,start?:number,length?:number)=>{
      const all=dynamic?[...dynamic]:[];const values=start===undefined?all:all.slice(start,start+length!);bindings.set(i,{group,dynamic:values});pass.setBindGroup(i,group,values);
     };
     if(key==="dispatchWorkgroups")return(x:number,y=1,z=1)=>{
      assert.ok(pipeline);const info=pipelineInfo.get(pipeline)!;const command:Command={pipeline,name:info?.name??"unknown",bindings:new Map(bindings),size:[x,y,z]};
      if(info?.name==="arenaGather"&&info.derived&&frames.includes(frame)&&!captured.has(frame)){
       assert.equal(capture,undefined);pass.end();capture=start(e,command);pass=e.beginComputePass();rebind();
      }
      if(capture)capture.commands.push(command);
      pass.dispatchWorkgroups(x,y,z);
      if(capture&&info?.name==="arenaStore"){
       pass.end();finish(e,capture);capture=undefined;pass=e.beginComputePass();rebind();
      }
     };
     const value=Reflect.get(pass,key,pass);return typeof value==="function"?value.bind(pass):value;
    }});
   };
   const value=Reflect.get(e,key,e);return typeof value==="function"?value.bind(e):value;
  }});
 };
 const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
}});
const device=managedGPUDevice(intercepted,{requireWorkerRealm:false});
const sceneId=option("scene")??"nbflip-figure-7-pour",scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
const dt=Number(option("dt")??scene.numerics.fixedDt_s);assert.ok(Number.isFinite(dt)&&dt>0);scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
const values=resolveMethodValues(method,"balanced",{...method.appDefaults,...getSceneDefinition(sceneId).methodProfile?.overrides,timeStep:"scene"});
const results:unknown[]=[];let solver:WebGPUUniformReferenceSolver|undefined;
const metadata={date:new Date().toISOString(),revision:execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8"}).trim(),shaderFileHash:createHash("sha256").update(readFileSync("lib/methods/uniform/uniform-surface-brick-arena.wgsl.ts")).digest("hex"),scene:sceneId,dt,frames,repeats,values,variants,shaderHashes,implementation:"product compact canonical publication with native capacity masks",trajectoryImplementation:"baseline: harness disables private canonical batch after initialization",oracleEntries:["arenaGather","arenaDistance","arenaFar","arenaStore"],candidateEntries:["arenaGather@publication@derived","arenaFar","arenaDistanceCanonical","arenaPublishFinish"],deadScratchExcluded:"candidate UB_MARK physical-brick mark words only",scope:"same-state complete derived-gather/metric/store chain; exact fields, metric and non-epoch state; opaque revision tokens compare changed/unchanged, validity class and monotonicity; baseline outputs restored; no unit tests"};
const save=(status:string,error?:string)=>writeFileSync(out,JSON.stringify({...metadata,status,error,errors,results},null,2)+"\n");
try{
 const begin=performance.now();solver=await method.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;// Keep the ordinary trajectory and untimed oracle on the preserved baseline.
 // This harness-only override adds no public solver setting. Candidate replay
 // still compiles the production entrypoints with the production Store layout.
 const flip=(solver as unknown as {mixedFrame:{narrowBandFlip:{canonicalSurface?:unknown}}}).mixedFrame.narrowBandFlip;
 assert.ok(flip.canonicalSurface,"product compact batches exist for this scene");flip.canonicalSurface=undefined;
 console.log(JSON.stringify({phase:"constructed",ms:performance.now()-begin}));
 for(frame=1;frame<=frames.at(-1)!;frame++){
  if(frame===frames[0])(solver as unknown as {mixedFrame:{surfaceWorkControl:{framePlan:{coalescePasses:boolean}}}}).mixedFrame.surfaceWorkControl.framePlan.coalescePasses=false;
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  if(frame%20===0)console.log(JSON.stringify({phase:"progress",frame}));
  if(!frames.includes(frame))continue;assert.ok(captured.has(frame),`captured ${frame}`);
  for(const p of pending.splice(0)){
   await Promise.all([p.times.mapAsync(GPUMapMode.READ),p.result.mapAsync(GPUMapMode.READ),p.examples.mapAsync(GPUMapMode.READ)]);const ticks=new BigUint64Array(p.times.getMappedRange()),diff=new Uint32Array(p.result.getMappedRange()),examples=new Uint32Array(p.examples.getMappedRange());const samples=variants.map(()=>[] as number[]);
   p.order.forEach((row,r)=>row.forEach((v,k)=>{const q=2*(r*variants.length+k);samples[v]!.push(Number(ticks[q+1]!-ticks[q]!)/1e6);}));
   const arms=variants.map((name,i)=>{const sorted=[...samples[i]!].sort((a,b)=>a-b);return {name,ms:samples[i],medianMs:sorted[sorted.length>>1],mismatches:p.parts.map((part,k)=>({part:part.name,words:diff[i*p.parts.length+k],opaqueRevisionWordDifferences:diff[variants.length*p.parts.length+i*p.parts.length+k],examples:Array.from({length:Math.min(64,diff[i*p.parts.length+k]!)},(_,j)=>Array.from(examples.subarray(4*(64*(i*p.parts.length+k)+j),4*(64*(i*p.parts.length+k)+j)+3)))}))};});
   const record={frame,time_s:frame*dt,particles:solver.narrowBandFlipInfo?.particles,snapshotBytes:p.total,parts:p.parts.map(p=>({name:p.name,bytes:p.bytes})),arms};results.push(record);console.log(JSON.stringify(record));save("running");
   p.times.unmap();p.result.unmap();p.examples.unmap();p.queries.destroy();p.resources.forEach(b=>b.destroy());
   assert.ok(arms.every(a=>a.mismatches.every(m=>m.words===0)),"all fields and observable control decisions match on every replay");
  }
  assert.deepEqual(errors,[]);
 }
 save("complete");
}catch(error){save("failed",String(error));throw error;}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
