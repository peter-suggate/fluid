/** Per-dispatch GPU time and grid of every compute dispatch over a few frames.
 * Every dispatch of the selected frames runs in its own timestamped pass
 * (pipeline and bindings reapplied), so pass overheads are not the frame's:
 * this locates launch cost by entry and grid, it does not report speed.
 * node_modules/.bin/tsx tools/probe-nbflip-dispatch-grid.ts [--root=DIR] [--scene=ID] [--scene-file=JSON]
 *   [--frames=20,21,22] [--top=60]
 * Run with the shared GPU lock. */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import type {uniformNarrowBandMethod} from '../lib/methods/uniform/uniform-narrow-band-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const root=resolve(option('root')??'.'),sceneId=option('scene')??'sparse-cm12-long-dam-break';
const frames=(option('frames')??'20,21,22').split(',').map(Number).sort((a,b)=>a-b),top=Number(option('top')??60);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0));
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const errors:string[]=[];raw.addEventListener('uncapturederror',event=>{event.preventDefault();errors.push(event.error.message);});
const CAPACITY=2048,queries=raw.createQuerySet({type:'timestamp',count:2*CAPACITY});
const names=new WeakMap<GPUComputePipeline,string>();
let active=false;const rows:{key:string;grid:string}[]=[];
const device=new Proxy(raw,{get(target,key){
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d);const c=d.compute.constants??{};
  names.set(pipeline,`${d.compute.entryPoint??'main'}${Object.keys(c).length?' '+JSON.stringify(c):''}`);return pipeline;
 };
 if(key==='createCommandEncoder')return (d?:GPUCommandEncoderDescriptor)=>{
  const encoder=target.createCommandEncoder(d);
  return new Proxy(encoder,{get(e,k){
   if(k==='beginComputePass')return (descriptor?:GPUComputePassDescriptor)=>{
    if(!active)return e.beginComputePass(descriptor);
    let pipeline:GPUComputePipeline|undefined;const bindings=new Map<number,unknown[]>();
    const dispatch=(kind:'dispatchWorkgroups'|'dispatchWorkgroupsIndirect',args:unknown[])=>{
     assert.ok(pipeline);const index=rows.length;assert.ok(index<CAPACITY,'dispatch capacity');
     rows.push({key:`${descriptor?.label??'unlabelled'} / ${names.get(pipeline)??pipeline.label}`,grid:kind==='dispatchWorkgroups'?(args as number[]).map(n=>n??1).join('x'):'indirect'});
     const pass=e.beginComputePass({label:descriptor?.label,timestampWrites:{querySet:queries,beginningOfPassWriteIndex:2*index,endOfPassWriteIndex:2*index+1}});
     pass.setPipeline(pipeline);for(const [i,values] of bindings)Reflect.apply(pass.setBindGroup,pass,[i,...values]);
     Reflect.apply(pass[kind],pass,args);pass.end();
    };
    return {
     setPipeline:(p:GPUComputePipeline)=>{pipeline=p;},
     setBindGroup:(i:number,...values:unknown[])=>{bindings.set(i,values);},
     dispatchWorkgroups:(...args:unknown[])=>dispatch('dispatchWorkgroups',args),
     dispatchWorkgroupsIndirect:(...args:unknown[])=>dispatch('dispatchWorkgroupsIndirect',args),
     end:()=>{},pushDebugGroup:()=>{},popDebugGroup:()=>{},insertDebugMarker:()=>{},
    } as unknown as GPUComputePassEncoder;
   };
   const v=Reflect.get(e,k,e);return typeof v==='function'?v.bind(e):v;
  }});
 };
 const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
}});
const load=(file:string)=>import(pathToFileURL(resolve(root,file)).href);
const [{uniformNarrowBandMethod:method},{resolveMethodValues},{sceneDocument},{getSceneDefinition},{managedGPUDevice}]=await Promise.all([
 load('lib/methods/uniform/uniform-narrow-band-method.ts'),load('lib/core/method-contract.ts'),load('lib/core/scene-definition.ts'),load('lib/core/scenes.ts'),load('lib/core/gpu-compilation-manager.ts')]);
const typed=method as typeof uniformNarrowBandMethod;
const definition=getSceneDefinition(sceneId),scene=option('scene-file')?JSON.parse(readFileSync(option('scene-file')!,'utf8')):structuredClone(sceneDocument(definition));
scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
const quality=definition.methodProfile?.quality??'balanced',values=resolveMethodValues(typed,quality,{...typed.appDefaults,...definition.methodProfile?.overrides});
const dt=typed.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
let solver:WebGPUUniformReferenceSolver|undefined;
const totals=new Map<string,{ms:number;count:number;grids:Set<string>}>();
try{
 solver=await typed.createSolverAsync!(managedGPUDevice(device,{requireWorkerRealm:false}),scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 // Rows accumulate over the selected frames; one more frame runs before the
 // resolve, so any submission a frame leaves pending has completed.
 const frameOf:number[]=[];
 for(let frame=1;frame<=frames.at(-1)!+1;frame++){
  active=frames.includes(frame);const first=rows.length;
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  active=false;for(let i=first;i<rows.length;i++)frameOf[i]=frame;
 }
 await raw.queue.onSubmittedWorkDone();
 const n=rows.length,resolved=raw.createBuffer({size:16*n,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read=raw.createBuffer({size:16*n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=raw.createCommandEncoder();e.resolveQuerySet(queries,0,2*n,resolved,0);e.copyBufferToBuffer(resolved,0,read,0,16*n);raw.queue.submit([e.finish()]);
 await read.mapAsync(GPUMapMode.READ);const stamps=new BigUint64Array(read.getMappedRange());
 assert.ok(!stamps.every(t=>t===0n),'GPU timestamps were not written');
 const perFrame=new Map<number,{n:number;ms:number}>();
 rows.forEach((row,i)=>{const ms=Number(stamps[2*i+1]!-stamps[2*i]!)/1e6;const f=perFrame.get(frameOf[i]!)??{n:0,ms:0};f.n++;f.ms+=ms;perFrame.set(frameOf[i]!,f);
  const t=totals.get(row.key)??{ms:0,count:0,grids:new Set<string>()};t.ms+=ms;t.count++;t.grids.add(row.grid);totals.set(row.key,t);});
 read.unmap();read.destroy();resolved.destroy();
 for(const [frame,f] of perFrame)console.log(JSON.stringify({frame,dispatches:f.n,dispatch_ms:+f.ms.toFixed(3)}));
 const sorted=[...totals].sort((a,b)=>b[1].ms-a[1].ms),k=frames.length;
 for(const [key,t] of sorted.slice(0,top))console.log(`${(t.ms/k).toFixed(4)} ms/frame  ${(t.count/k).toFixed(1)}x  grids=${[...t.grids].slice(0,4).join(',')}  ${key}`);
 assert.deepEqual(errors,[]);
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
