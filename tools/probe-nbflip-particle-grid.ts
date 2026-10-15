/** Frozen actual snapshot shader inputs, alternate4096 vs --grid=256 groups.
 * Every particle output word must match; restores production result before frame continues.
 * --scene-file supports isolated32³ fixtures, --frames=1,12 and --repeats=9.
 */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {resolveMethodValues} from '../lib/core/method-contract';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {managedGPUDevice} from '../lib/core/gpu-compilation-manager';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {uniformNarrowBandMethod as method} from '../lib/methods/uniform/uniform-narrow-band-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const selectedEntries=['snapshot'];
const candidateGrid=Number(option('grid')??256),batch=Number(option('batch')??16);
const entryNames=new WeakMap<GPUComputePipeline,string>();
const frames=(option('frames')??'1,24,60,120').split(',').map(Number).sort((a,b)=>a-b),repeats=Number(option('repeats')??5);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const indirectPipeline=await raw.createComputePipelineAsync({layout:'auto',compute:{entryPoint:'main',module:raw.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> state:array<u32>;
@group(0) @binding(1) var<storage,read_write> dispatch:array<u32>;
@compute @workgroup_size(1) fn main(){dispatch[0]=max(1u,min(4096u,(state[1]+63u)/64u));dispatch[1]=1u;dispatch[2]=1u;}`})}});
const sources=new WeakMap<GPUShaderModule,string>(),references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
const entries=new WeakMap<GPUBindGroup,readonly GPUBindGroupEntry[]>();
let frame=0;const captured=new Set<string>();
type Probe={entry:string;frame:number;results:[GPUBuffer,GPUBuffer];times:GPUBuffer;queries:GPUQuerySet;resources:GPUBuffer[];total:number;parts:{name:string;bytes:number}[]};
const pending:Probe[]=[];
const intercepted=new Proxy(raw,{get(target,key){
 if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
 if(key==='createBindGroup')return (d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);entries.set(group,[...d.entries]);return group;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d),source=sources.get(d.compute.module);
  if(selectedEntries.includes(d.compute.entryPoint??'')&&source){
   entryNames.set(pipeline,d.compute.entryPoint!);
   references.set(pipeline,pipeline);
  }return pipeline;
 };
 if(key==='createCommandEncoder')return (d?:GPUCommandEncoderDescriptor)=>{
  const encoder=target.createCommandEncoder(d);
  return new Proxy(encoder,{get(e,key){
   if(key==='beginComputePass')return (descriptor?:GPUComputePassDescriptor)=>{
    let pass=e.beginComputePass(descriptor),pipeline:GPUComputePipeline|undefined;
    const bindings=new Map<number,unknown[]>();
    const bind=()=>{assert.ok(pipeline);pass.setPipeline(pipeline);for(const [index,args] of bindings)Reflect.apply(pass.setBindGroup,pass,[index,...args]);};
    const dispatch=(method:'dispatchWorkgroups'|'dispatchWorkgroupsIndirect',args:unknown[])=>{
     const reference=pipeline&&references.get(pipeline),actualEntry=pipeline&&entryNames.get(pipeline);
     const captureKey=`${frame}:${actualEntry}`;
     if(!reference||!actualEntry||!frames.includes(frame)||captured.has(captureKey)){Reflect.apply(pass[method],pass,args);return;}
     captured.add(captureKey);pass.end();
     const groupEntries=entries.get(bindings.get(1)![0] as GPUBindGroup);assert.ok(groupEntries);
     const bufferParts=[1].map(binding=>{const bindingResource=groupEntries.find(entry=>entry.binding===binding)!.resource as GPUBufferBinding;return {name:'particles',...bindingResource,size:bindingResource.size??bindingResource.buffer.size-(bindingResource.offset??0)};});
     const indirect=target.createBuffer({size:12,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
     const counts=groupEntries.find(entry=>entry.binding===4)!.resource as GPUBufferBinding;
     const indirectGroup=target.createBindGroup({layout:indirectPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:counts},{binding:1,resource:{buffer:indirect}}]});
     {const prepare=e.beginComputePass();prepare.setPipeline(indirectPipeline);prepare.setBindGroup(0,indirectGroup);prepare.dispatchWorkgroups(1);prepare.end();}
     let total=0;const locations=bufferParts.map(part=>{const at=total;total+=part.size;return {part,at};});
     const parts=bufferParts.map(part=>({name:part.name,bytes:part.size}));
     const make=(size:number,usage:number)=>target.createBuffer({size,usage});
     assert.ok(total<=target.limits.maxBufferSize,`particle residency snapshot ${total} exceeds buffer limit ${target.limits.maxBufferSize}`);
     const initial=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),saved=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),results:[GPUBuffer,GPUBuffer]=[make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ)];
     const queries=target.createQuerySet({type:'timestamp',count:4*repeats}),resolved=make(32*repeats,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(32*repeats,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
     const snapshot=(buffer:GPUBuffer,offset:number)=>{
      for(const {part,at} of locations)e.copyBufferToBuffer(part.buffer,part.offset??0,buffer,offset+at,part.size);
     };
     const restore=(buffer:GPUBuffer)=>{
      for(const {part,at} of locations)e.copyBufferToBuffer(buffer,at,part.buffer,part.offset??0,part.size);
     };
     snapshot(initial,0);
     // First execute the production dispatch and retain its exact outputs.
     pass=e.beginComputePass();bind();Reflect.apply(pass[method],pass,args);pass.end();snapshot(saved,0);
     for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){
      const original=(repeat+kind)%2===0,index=4*repeat+2*kind;restore(initial);
      pass=e.beginComputePass({label:original?'QA snapshot4096':'QA snapshot reducedgrid',timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});bind();pass.setPipeline(original?reference:pipeline!);for(let run=0;run<batch;run++){if(!original&&candidateGrid===0)pass.dispatchWorkgroupsIndirect(indirect,0);else Reflect.apply(pass[method],pass,original?args:[candidateGrid]);}pass.end();
      if(repeat===0)snapshot(results[original?0:1],0);
     }
     restore(saved);
     e.resolveQuerySet(queries,0,4*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,32*repeats);
     pending.push({entry:actualEntry,frame,results,times,queries,resources:[initial,saved,resolved,indirect],total,parts});
     pass=e.beginComputePass();bind();
    };
    return new Proxy(pass,{get(_p,key){
     if(key==='setPipeline')return (value:GPUComputePipeline)=>{pipeline=value;pass.setPipeline(value);};
     if(key==='setBindGroup')return (index:number,...args:unknown[])=>{bindings.set(index,args);Reflect.apply(pass.setBindGroup,pass,[index,...args]);};
     if(key==='dispatchWorkgroups'||key==='dispatchWorkgroupsIndirect')return (...args:unknown[])=>dispatch(key,args);
     const value=Reflect.get(pass,key,pass);return typeof value==='function'?value.bind(pass):value;
    }});
   };
   const value=Reflect.get(e,key,e);return typeof value==='function'?value.bind(e):value;
  }});
 };
 const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
}});
const device=managedGPUDevice(intercepted,{requireWorkerRealm:false}),errors:string[]=[];
device.addEventListener('uncapturederror',event=>{event.preventDefault();errors.push(event.error.message);});
let solver:WebGPUUniformReferenceSolver|undefined;
try{
 const definition=getSceneDefinition(option('scene')??'sparse-cm12-long-dam-break'),quality=definition.methodProfile?.quality??'balanced',scene=option('scene-file')?JSON.parse(readFileSync(option('scene-file')!,'utf8')):structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile?.overrides,particleTransfer:'flip'}),dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(frame=1;frame<=frames.at(-1)!;frame++){
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  if(!frames.includes(frame))continue;for(const selected of selectedEntries)assert.ok(captured.has(`${frame}:${selected}`),`${selected} dispatch was captured`);
  for(const probe of pending.splice(0)){
   await Promise.all([...probe.results.map(buffer=>buffer.mapAsync(GPUMapMode.READ)),probe.times.mapAsync(GPUMapMode.READ)]);
   const words=probe.results.map(buffer=>new Uint32Array(buffer.getMappedRange())),timestamps=new BigUint64Array(probe.times.getMappedRange());
   let cursor=0;const mismatches=probe.parts.map(part=>{let count=0;
    for(let i=0;i<part.bytes/4;i++)if(words[0]![cursor+i]!==words[1]![cursor+i])count++;
    cursor+=part.bytes/4;return {name:part.name,bytes:part.bytes,mismatches:count};});
   const reference_ms:number[]=[],optimized_ms:number[]=[];
   for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){const index=4*repeat+2*kind;((repeat+kind)%2===0?reference_ms:optimized_ms).push(Number(timestamps[index+1]!-timestamps[index]!)/(1e6*batch));}
   console.log(JSON.stringify({entry:probe.entry,frame,grid:candidateGrid,batch,particles:solver.narrowBandFlipInfo,time_s:frame*dt,mismatches,reference_ms,optimized_ms}));
   for(const result of probe.results){result.unmap();result.destroy();}probe.times.unmap();probe.times.destroy();probe.queries.destroy();for(const resource of probe.resources)resource.destroy();
   assert.ok(mismatches.every(part=>part.mismatches===0),'snapshot outputs match bitwise');assert.deepEqual(errors,[]);
  }
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
