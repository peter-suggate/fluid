/** Same-state replay of coarse return -> fine return -> preserved-component copy -> band projection.
 * Works with the product implementation (explicit full-return oracle override
 * 0 versus deletion override 1) and archived sources (isolated shader rewrite).
 * Restores every captured writable channel and compares every word, including
 * physical velocity textures, boundary values, pressure pool, presentation,
 * and shared writable state. Restores/comparisons are untimed.
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
import {pressureDirectTransferWGSL} from "./nbflip-pressure-return-experiment";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
for(const arg of process.argv.slice(2))assert.match(arg,/^--(?:frames|repeats|out|scene|dt)=.+$/);
const frames=(option("frames")??"80,120").split(",").map(Number).sort((a,b)=>a-b),repeats=Number(option("repeats")??7);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>=3);
const out=resolve(option("out")??"artifacts/pressure-chain.json");mkdirSync(dirname(out),{recursive:true});
const variants=["baseline","control","skipOverwrittenBandReturn"];
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal","disable-dawn-features=timestamp_quantization"]).requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:(["timestamp-query","float32-filterable","subgroups"] as GPUFeatureName[]).filter(f=>adapter.features.has(f)),requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const errors:string[]=[];raw.addEventListener("uncapturederror",event=>{event.preventDefault();errors.push(event.error.message);console.error(event.error.message);});
const compareModule=raw.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> live:array<u32>;
@group(0) @binding(1) var<storage,read> reference:array<u32>;
@group(0) @binding(2) var<storage,read_write> mismatches:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:vec4u;
@compute @workgroup_size(256) fn compare(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u){
 var n=0u;for(var i=id.x;i<p.z;i+=256u*groups.x){n+=u32(live[p.x+i]!=reference[p.x+i]);}
 if(n>0u){atomicAdd(&mismatches[p.w],n);}
}`});
const compare=await raw.createComputePipelineAsync({layout:"auto",compute:{module:compareModule,entryPoint:"compare"}});
const groupInfo=new WeakMap<GPUBindGroup,{entries:GPUBindGroupEntry[];layout:GPUBindGroupLayout}>();
const layouts=new WeakMap<GPUBindGroupLayout,GPUBindGroupLayoutEntry[]>(),textures=new WeakMap<GPUTextureView,GPUTexture>();
type PipelineInfo={name:string;oracle?:GPUComputePipeline;candidate?:GPUComputePipeline;candidateLayout?:GPUBindGroupLayout;implementation?:"product"|"archive-rewrite";oracleConstants?:Record<string,number>;candidateConstants?:Record<string,number>};
const pipelineInfo=new WeakMap<GPUComputePipeline,PipelineInfo>();
const pipelineLayouts=new WeakMap<GPUPipelineLayout,(GPUBindGroupLayout|null|undefined)[]>();
const EXTRA_BINDING=99;
const moduleSources=new WeakMap<GPUShaderModule,string>();const shaderHashes:Record<string,string>={};
type Binding={group:GPUBindGroup;dynamic:number[]};
type Command={pipeline:GPUComputePipeline;name:string;bindings:Map<number,Binding>;size:[number,number,number]};
type Part={name:string;bytes:number;at:number;buffer?:GPUBuffer;offset?:number;texture?:GPUTexture;row?:number;height?:number;depth?:number};
type Capture={frame:number;commands:Command[];parts:Part[];total:number;initial:GPUBuffer;saved:GPUBuffer;live:GPUBuffer;resources:GPUBuffer[]};
type Pending=Capture & {times:GPUBuffer;result:GPUBuffer;queries:GPUQuerySet;order:number[][]};
let frame=0,capture:Capture|undefined;const captured=new Set<number>(),pending:Pending[]=[];
const apply=(pass:GPUComputePassEncoder,c:Command,candidate=false)=>{
 const info=pipelineInfo.get(c.pipeline)!;
 pass.setPipeline(c.name==="toSimulationFine"?(candidate?info.candidate!:info.oracle!):c.pipeline);
 for(const [i,b] of c.bindings){
  if(candidate&&info.candidateLayout&&i===1){
   const original=groupInfo.get(b.group)!;assert.ok(original);
   const band=getBand();
   const group=raw.createBindGroup({layout:info.candidateLayout!,entries:[...original.entries,{binding:EXTRA_BINDING,resource:{buffer:band.index}}]});
   pass.setBindGroup(i,group,b.dynamic);
  }else pass.setBindGroup(i,b.group,b.dynamic);
 }
 pass.dispatchWorkgroups(...c.size);
};
type BandView={index:GPUBuffer;slotMapOffset:number;groups:Map<string,{group:GPUBindGroup;base:GPUBindGroup}>};
const getBand=()=>{assert.ok(solver);return (solver as unknown as {mixedFrame:{band:BandView}}).mixedFrame.band;};
const copy=(e:GPUCommandEncoder,c:Capture,b:GPUBuffer,restore=false)=>{
 for(const p of c.parts){
  if(p.buffer){if(restore)e.copyBufferToBuffer(b,p.at,p.buffer,p.offset??0,p.bytes);else e.copyBufferToBuffer(p.buffer,p.offset??0,b,p.at,p.bytes);}
  else {const texture={texture:p.texture!},buffer={buffer:b,offset:p.at,bytesPerRow:p.row!,rowsPerImage:p.height!},extent={width:p.texture!.width,height:p.height!,depthOrArrayLayers:p.depth!};if(restore)e.copyBufferToTexture(buffer,texture,extent);else e.copyTextureToBuffer(texture,buffer,extent);}
 }
};
function start(e:GPUCommandEncoder,command:Command):Capture{
 const parts:Part[]=[];const seen=new Set<GPUBuffer|GPUTexture>();let total=0;
 const currentGroups=[...command.bindings].map(([index,binding])=>({index,group:binding.group}));
 const returned=(solver as unknown as {mixedFrame:{split:{toSimulation:{group:GPUBindGroup;base:GPUBindGroup}}}}).mixedFrame.split.toSimulation;
 currentGroups.push({index:1,group:returned.group},{index:1,group:returned.base});
 for(const name of ["copy","project"]){const bound=getBand().groups.get(name);assert.ok(bound,`arena ${name} group`);currentGroups.push({index:1,group:bound.group},{index:1,group:bound.base});}
 for(const {index,group} of currentGroups){
  const info=groupInfo.get(group)!;assert.ok(info);const entries=layouts.get(info.layout)!;assert.ok(entries);
  for(const entry of info.entries){
   const layout=entries.find(x=>x.binding===entry.binding)!;
   if(layout.buffer?.type==="storage"){
    const r=entry.resource as GPUBufferBinding;if(seen.has(r.buffer))continue;seen.add(r.buffer);
    // Full backing buffer: aliases may expose different ranges later in the chain.
    const bytes=r.buffer.size;total=Math.ceil(total/256)*256;
    parts.push({name:`g${index}b${entry.binding}:${r.buffer.label}`,bytes,at:total,buffer:r.buffer,offset:0});total+=bytes;
   }else if(layout.storageTexture){
    const t=textures.get(entry.resource as GPUTextureView);assert.ok(t,`texture ${entry.binding}`);if(seen.has(t))continue;seen.add(t);
    assert.ok(["r32float","rgba32float"].includes(t.format),`snapshot format ${t.format}`);
    const pixel=t.format==="rgba32float"?16:4,row=Math.ceil(t.width*pixel/256)*256,bytes=row*t.height*t.depthOrArrayLayers;total=Math.ceil(total/256)*256;
    parts.push({name:`g${index}b${entry.binding}:${t.label}`,bytes,at:total,texture:t,row,height:t.height,depth:t.depthOrArrayLayers});total+=bytes;
   }
  }
 }
 assert.ok(total<raw.limits.maxStorageBufferBindingSize);const resources:GPUBuffer[]=[];
 const make=()=>{const b=raw.createBuffer({size:total,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|GPUBufferUsage.STORAGE});resources.push(b);return b;};
 const c={frame,commands:[],parts,total,initial:make(),saved:make(),live:make(),resources};copy(e,c,c.initial);return c;
}
function finish(e:GPUCommandEncoder,c:Capture){
 assert.deepEqual(c.commands.map(c=>c.name),["toSimulationCoarse","toSimulationFine","pCopy","pProject"]);
 const fine=c.commands[1]!,fineInfo=pipelineInfo.get(fine.pipeline)!;assert.ok(fineInfo.candidate&&fineInfo.oracle);
 if(fineInfo.implementation==="product"){
  assert.equal(fineInfo.oracleConstants?.umSkipProjectedBand,0,"baseline must force full return");
  assert.equal(fineInfo.candidateConstants?.umSkipProjectedBand,1,"candidate must enable deletion");
  const bandGroup=fine.bindings.get(2);assert.ok(bandGroup,"product replay retains the band-index group");
  const indexBinding=groupInfo.get(bandGroup.group)?.entries.find(entry=>entry.binding===99)?.resource as GPUBufferBinding|undefined;
  assert.equal(indexBinding?.buffer,getBand().index,"product group 2 reads this frame's band index");
  assert.ok(c.parts.some(part=>part.buffer===getBand().index),"band index is snapshotted through arena state");
 }
 // Fail before submission if a later dispatch writes an un-snapshotted
 // resource, including a detail-field allocation or binding change.
 for(const cmd of c.commands)for(const bound of cmd.bindings.values()){
  assert.equal(bound.dynamic.length,0,"snapshot ranges currently require non-dynamic bindings");
  const info=groupInfo.get(bound.group)!;assert.ok(info);const entries=layouts.get(info.layout)!;
  for(const entry of info.entries){const layout=entries.find(x=>x.binding===entry.binding)!;
   if(layout.buffer?.type==="storage"){const resource=entry.resource as GPUBufferBinding;assert.ok(c.parts.some(p=>p.buffer===resource.buffer),`uncaptured buffer in ${cmd.name}`);}
   if(layout.storageTexture){const texture=textures.get(entry.resource as GPUTextureView);assert.ok(texture&&c.parts.some(p=>p.texture===texture),`uncaptured texture in ${cmd.name}`);}
  }
 }
 copy(e,c,c.saved);
 const count=variants.length*c.parts.length,queryCount=2*repeats*variants.length;
 const make=(size:number,usage:number)=>{const b=raw.createBuffer({size,usage});c.resources.push(b);return b;};
 const diff=make(4*count,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
 const result=make(4*count,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 const queries=raw.createQuerySet({type:"timestamp",count:queryCount});
 const resolved=make(8*queryCount,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);
 const times=make(8*queryCount,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 const comparisons=variants.map((_,v)=>c.parts.map((p,i)=>{
  const params=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);raw.queue.writeBuffer(params,0,new Uint32Array([p.at/4,0,p.bytes/4,v*c.parts.length+i]));
  return {n:Math.min(4096,Math.ceil(p.bytes/1024)),group:raw.createBindGroup({layout:compare.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:c.live}},{binding:1,resource:{buffer:c.saved}},{binding:2,resource:{buffer:diff}},{binding:3,resource:{buffer:params}}]})};
 }));
 const order:number[][]=[];
 for(let r=0;r<repeats;r++){
  const row=variants.map((_,i)=>(i+r)%variants.length);if(r%2)row.reverse();order.push(row);
  for(let k=0;k<row.length;k++){
   const v=row[k]!,q=2*(r*variants.length+k);copy(e,c,c.initial,true);
   const pass=e.beginComputePass({label:`pressure return chain ${variants[v]}`,timestampWrites:{querySet:queries,beginningOfPassWriteIndex:q,endOfPassWriteIndex:q+1}});
   for(const cmd of c.commands)apply(pass,cmd,v===2&&cmd.name==="toSimulationFine");
   pass.end();copy(e,c,c.live);
   const verify=e.beginComputePass();verify.setPipeline(compare);for(const cmp of comparisons[v]!){verify.setBindGroup(0,cmp.group);verify.dispatchWorkgroups(cmp.n);}verify.end();
  }
 }
 copy(e,c,c.saved,true);e.resolveQuerySet(queries,0,queryCount,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,8*queryCount);e.copyBufferToBuffer(diff,0,result,0,4*count);
 pending.push({...c,times,result,queries,order});captured.add(frame);
}
const intercepted=new Proxy(raw,{get(target,key){
 if(key==="createBuffer")return(d:GPUBufferDescriptor)=>target.createBuffer({...d,usage:(d.usage&(GPUBufferUsage.MAP_READ|GPUBufferUsage.MAP_WRITE))?d.usage:d.usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 if(key==="createTexture")return(d:GPUTextureDescriptor)=>{
  const texture=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});const create=texture.createView.bind(texture);
  texture.createView=(d?:GPUTextureViewDescriptor)=>{const view=create(d);textures.set(view,texture);return view;};return texture;
 };
 if(key==="createBindGroupLayout")return(d:GPUBindGroupLayoutDescriptor)=>{const layout=target.createBindGroupLayout(d);layouts.set(layout,[...d.entries]);return layout;};
 if(key==="createBindGroup")return(d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);groupInfo.set(group,{entries:[...d.entries],layout:d.layout});return group;};
 if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);moduleSources.set(module,d.code);return module;};
 if(key==="createPipelineLayout")return(d:GPUPipelineLayoutDescriptor)=>{const layout=target.createPipelineLayout(d);pipelineLayouts.set(layout,[...d.bindGroupLayouts]);return layout;};
 if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d);const name=d.compute.entryPoint??"";
  const info:PipelineInfo={name};
  if(name==="toSimulationFine"){
   const source=moduleSources.get(d.compute.module)!;assert.ok(source);writeFileSync(`${out}.baseline.wgsl`,source);
   const product=/\btProjectedBand\s*:/.test(source)&&/override\s+umSkipProjectedBand\b/.test(source);
   info.implementation=product?"product":"archive-rewrite";
   info.oracleConstants={...d.compute.constants,...(product?{umSkipProjectedBand:0}:{})};
   info.candidateConstants={...d.compute.constants,...(product?{umSkipProjectedBand:1}:{})};
   shaderHashes.baseline=createHash("sha256").update(source).digest("hex");
   if(product){
    // The normal product pipeline may already skip the band, including on the
    // capture frame. Compile an explicit full-return oracle and never infer
    // baseline behavior from that pipeline's default or current constants.
    info.oracle=await target.createComputePipelineAsync({...d,label:"pressure full-return oracle",compute:{...d.compute,constants:info.oracleConstants}});
    info.candidate=await target.createComputePipelineAsync({...d,label:"pressure product band-return deletion",compute:{...d.compute,constants:info.candidateConstants}});
    shaderHashes.candidate=shaderHashes.baseline!;writeFileSync(`${out}.candidate.wgsl`,source);
   }else{
    info.oracle=pipeline;
    // Archive compatibility: add the isolated index binding and WGSL rewrite.
    const arenaSource=readFileSync("lib/methods/uniform/uniform-pressure-arena.ts","utf8");
    const header=Number(arenaSource.match(/const HEADER=(\d+)/)?.[1]);assert.ok(Number.isInteger(header));
    const code=pressureDirectTransferWGSL(source,{slotMapOffset:4*header,binding:EXTRA_BINDING});
    shaderHashes.candidate=createHash("sha256").update(code).digest("hex");writeFileSync(`${out}.candidate.wgsl`,code);
    assert.ok(d.layout!=="auto");const existing=pipelineLayouts.get(d.layout);assert.ok(existing);
    const original=layouts.get(existing[1]!);assert.ok(original);
    info.candidateLayout=target.createBindGroupLayout({entries:[...original,{binding:EXTRA_BINDING,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
    const bindGroupLayouts=[...existing];bindGroupLayouts[1]=info.candidateLayout;
    info.candidate=await target.createComputePipelineAsync({...d,label:"pressure skip overwritten band return",layout:target.createPipelineLayout({bindGroupLayouts}),compute:{...d.compute,module:target.createShaderModule({code})}});
   }
  }
  pipelineInfo.set(pipeline,info);return pipeline;
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
      if(info?.name==="toSimulationCoarse"&&frames.includes(frame)&&!captured.has(frame)){
       assert.equal(capture,undefined);assert.equal(getBand().slotMapOffset,4*Number(readFileSync("lib/methods/uniform/uniform-pressure-arena.ts","utf8").match(/const HEADER=(\d+)/)?.[1]));pass.end();capture=start(e,command);pass=e.beginComputePass();rebind();
      }
      if(capture)capture.commands.push(command);
      // The untimed reference chain also uses the explicit full-return
      // oracle. Its final output is saved and restored after every replay.
      if(capture&&info?.name==="toSimulationFine"){assert.ok(info.oracle);pass.setPipeline(info.oracle);pass.dispatchWorkgroups(x,y,z);pass.setPipeline(pipeline);}
      else pass.dispatchWorkgroups(x,y,z);
      if(capture&&info?.name==="pProject"){
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
const metadata={date:new Date().toISOString(),revision:execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8"}).trim(),shaderFileHash:createHash("sha256").update(readFileSync("lib/methods/uniform/uniform-mixed-remap.ts")).digest("hex"),scene:sceneId,dt,frames,repeats,values,variants,shaderHashes,scope:"same-state complete pressure return/copy/band projection chain; exact all-written-channel comparison; explicit full-return oracle outputs restored; product constants 0 versus 1 or archive rewrite; no unit tests"};
const save=(status:string,error?:string)=>writeFileSync(out,JSON.stringify({...metadata,status,error,errors,results},null,2)+"\n");
try{
 const begin=performance.now();solver=await method.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;console.log(JSON.stringify({phase:"constructed",ms:performance.now()-begin}));
 for(frame=1;frame<=frames.at(-1)!;frame++){
  if(frame===frames[0])(solver as unknown as {mixedFrame:{surfaceWorkControl:{framePlan:{coalescePasses:boolean}}}}).mixedFrame.surfaceWorkControl.framePlan.coalescePasses=false;
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  if(frame%20===0)console.log(JSON.stringify({phase:"progress",frame}));
  if(!frames.includes(frame))continue;assert.ok(captured.has(frame),`captured ${frame}`);
  for(const p of pending.splice(0)){
   await Promise.all([p.times.mapAsync(GPUMapMode.READ),p.result.mapAsync(GPUMapMode.READ)]);const ticks=new BigUint64Array(p.times.getMappedRange()),diff=new Uint32Array(p.result.getMappedRange());const samples=variants.map(()=>[] as number[]);
   p.order.forEach((row,r)=>row.forEach((v,k)=>{const q=2*(r*variants.length+k);samples[v]!.push(Number(ticks[q+1]!-ticks[q]!)/1e6);}));
   const arms=variants.map((name,i)=>{const sorted=[...samples[i]!].sort((a,b)=>a-b);return {name,ms:samples[i],medianMs:sorted[sorted.length>>1],mismatches:p.parts.map((part,k)=>({part:part.name,words:diff[i*p.parts.length+k]}))};});
   const fineInfo=pipelineInfo.get(p.commands.find(c=>c.name==="toSimulationFine")!.pipeline)!;
   const record={pipelineContract:{implementation:fineInfo.implementation,oracleConstants:fineInfo.oracleConstants,candidateConstants:fineInfo.candidateConstants},frame,time_s:frame*dt,particles:solver.narrowBandFlipInfo?.particles,snapshotBytes:p.total,parts:p.parts.map(p=>({name:p.name,bytes:p.bytes})),arms};results.push(record);console.log(JSON.stringify(record));save("running");
   p.times.unmap();p.result.unmap();p.queries.destroy();p.resources.forEach(b=>b.destroy());
   assert.ok(arms.every(a=>a.mismatches.every(m=>m.words===0)),"all physical outputs match on every replay");
  }
  assert.deepEqual(errors,[]);
 }
 save("complete");
}catch(error){save("failed",String(error));throw error;}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
