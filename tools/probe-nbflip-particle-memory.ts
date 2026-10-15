/** Same-input GPU replay of the packed three-float neighbor motion cache
 * against a four-float cache. A diagnostic-only tail holds the reference;
 * classify fills both from the same particles. Restores every output before
 * each replay and requires bitwise equality of fine/base velocity and receipt.
 * node --import tsx tools/probe-nbflip-particle-memory.ts --frames=120,600 --repeats=9
 * Run alone on the GPU. This probe uses the long-slide FLIP profile. */
import assert from 'node:assert/strict';
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
const frames=(option('frames')??'120,600').split(',').map(Number).sort((a,b)=>a-b),repeats=Number(option('repeats')??9);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const sources=new WeakMap<GPUShaderModule,string>(),references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
const entries=new WeakMap<GPUBindGroup,readonly GPUBindGroupEntry[]>(),textures=new WeakMap<GPUTextureView,GPUTexture>();
let frame=0,capturedFrame=-1,particleCapacity=0;
type Probe={frame:number;results:[GPUBuffer,GPUBuffer];work:GPUBuffer;times:GPUBuffer;queries:GPUQuerySet;resources:GPUBuffer[];total:number;parts:{name:string;bytes:number}[]};
const pending:Probe[]=[];
const intercepted=new Proxy(raw,{get(target,key){
 if(key==='createBuffer')return (d:GPUBufferDescriptor)=>{if(d.label==='Narrow-band FLIP particles A')particleCapacity=d.size/48;if(d.label==='Narrow-band FLIP cell starts and compact neighbor samples')d={...d,size:d.size+16*particleCapacity};return target.createBuffer(d);};
 if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{
  const anchor='particles[i].before.w=nbParticleState(nbParticleAge(particles[i]),select(1.0,0.0,supported));';
  const code=d.code.replace(anchor,anchor+'let ca=UM_CELL_KEYS()+7u*arrayLength(&particles)+4u*i;let cv=bitcast<vec4u>(vec4f(particles[i].velocity.xyz,nbParticleMode(particles[i])));links[ca]=cv.x;links[ca+1u]=cv.y;links[ca+2u]=cv.z;links[ca+3u]=cv.w;');
  const module=target.createShaderModule({...d,code});sources.set(module,code);return module;
 };
 if(key==='createTexture')return (d:GPUTextureDescriptor)=>{
  const texture=target.createTexture(d),createView=texture.createView.bind(texture);
  Object.defineProperty(texture,'createView',{value:(descriptor?:GPUTextureViewDescriptor)=>{const view=createView(descriptor);textures.set(view,texture);return view;}});return texture;
 };
 if(key==='createBindGroup')return (d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);entries.set(group,[...d.entries]);return group;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d),source=sources.get(d.compute.module);
  if(d.compute.entryPoint==='transfer'){
   const code=source!.replace('fn nbMotion(i:u32)->vec4f{let a=UM_CELL_KEYS()+4u*arrayLength(&particles)+3u*i;let mode=links[UM_CELL_KEYS()+4u*i+3u]>>31u;return vec4f(bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u])),f32(mode));}', 'fn nbMotion(i:u32)->vec4f{let a=UM_CELL_KEYS()+7u*arrayLength(&particles)+4u*i;return bitcast<vec4f>(vec4u(links[a],links[a+1u],links[a+2u],links[a+3u]));}');assert.notEqual(code,source);
   references.set(pipeline,await target.createComputePipelineAsync({...d,label:'QA cached motion gather',compute:{...d.compute,module:target.createShaderModule({code})}}));
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
     const reference=pipeline&&references.get(pipeline);
     if(!reference||!frames.includes(frame)||capturedFrame===frame){Reflect.apply(pass[method],pass,args);return;}
     capturedFrame=frame;pass.end();
     const groupEntries=entries.get(bindings.get(1)![0] as GPUBindGroup);assert.ok(groupEntries);
     const phi=groupEntries.find(entry=>entry.binding===4)!.resource as GPUBufferBinding;
     const imageParts=[9,73].map(binding=>{const texture=textures.get(groupEntries.find(entry=>entry.binding===binding)!.resource as GPUTextureView);assert.ok(texture);const bytesPerRow=Math.ceil(texture.width*16/256)*256;return {texture,bytesPerRow,bytes:bytesPerRow*texture.height*texture.depthOrArrayLayers};});
     const phiBytes=phi.size??phi.buffer.size-(phi.offset??0),imageOffset=Math.ceil(phiBytes/256)*256,parts=[{name:'receipt',bytes:phiBytes},{name:'padding',bytes:imageOffset-phiBytes},...imageParts.map((p,i)=>({name:i?'baseVelocity':'fineVelocity',bytes:p.bytes}))],total=parts.reduce((n,p)=>n+p.bytes,0);
     const make=(size:number,usage:number)=>target.createBuffer({size,usage});
     const work=make(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),workBinding=groupEntries.find(entry=>entry.binding===4)!.resource as GPUBufferBinding;
     e.copyBufferToBuffer(workBinding.buffer,workBinding.offset??0,work,0,16);
     assert.ok(total<=target.limits.maxBufferSize,`particle motion gather snapshot ${total} exceeds buffer limit ${target.limits.maxBufferSize}`);
     const initial=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),saved=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),results:[GPUBuffer,GPUBuffer]=[make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ)];
     const queries=target.createQuerySet({type:'timestamp',count:4*repeats}),resolved=make(32*repeats,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(32*repeats,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
     const snapshot=(buffer:GPUBuffer,offset:number)=>{e.copyBufferToBuffer(phi.buffer,phi.offset??0,buffer,offset,phiBytes);offset+=imageOffset;for(const p of imageParts){e.copyTextureToBuffer({texture:p.texture},{buffer,offset,bytesPerRow:p.bytesPerRow,rowsPerImage:p.texture.height},[p.texture.width,p.texture.height,p.texture.depthOrArrayLayers]);offset+=p.bytes;}};
     const restore=(buffer:GPUBuffer)=>{e.copyBufferToBuffer(buffer,0,phi.buffer,phi.offset??0,phiBytes);let offset=imageOffset;for(const p of imageParts){e.copyBufferToTexture({buffer,offset,bytesPerRow:p.bytesPerRow,rowsPerImage:p.texture.height},{texture:p.texture},[p.texture.width,p.texture.height,p.texture.depthOrArrayLayers]);offset+=p.bytes;}};
     // Restore the same pre-dispatch outputs too: otherwise a missing store
     // could inherit the reference result and falsely pass this comparison.
     snapshot(initial,0);
     // First execute the production dispatch and retain its exact outputs.
     pass=e.beginComputePass();bind();Reflect.apply(pass[method],pass,args);pass.end();snapshot(saved,0);
     for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){
      const original=(repeat+kind)%2===0,index=4*repeat+2*kind;restore(initial);
      pass=e.beginComputePass({label:original?'QA four-float motion cache':'QA packed three-float motion cache',timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});bind();pass.setPipeline(original?reference:pipeline!);Reflect.apply(pass[method],pass,args);pass.end();
      if(repeat===0)snapshot(results[original?0:1],0);
     }
     restore(saved);
     e.resolveQuerySet(queries,0,4*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,32*repeats);
     pending.push({frame,results,work,times,queries,resources:[initial,saved,resolved],total,parts});
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
 const definition=getSceneDefinition('nbflip-water-slide-long'),quality=definition.methodProfile!.quality,scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides}),dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(frame=1;frame<=frames.at(-1)!;frame++){
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  if(!frames.includes(frame))continue;assert.equal(capturedFrame,frame,'particle motion gather dispatch was captured');
  for(const probe of pending.splice(0)){
   await Promise.all([...probe.results.map(buffer=>buffer.mapAsync(GPUMapMode.READ)),probe.work.mapAsync(GPUMapMode.READ),probe.times.mapAsync(GPUMapMode.READ)]);
   const work=Array.from(new Uint32Array(probe.work.getMappedRange()));probe.work.unmap();probe.work.destroy();
   const words=probe.results.map(buffer=>new Uint32Array(buffer.getMappedRange())),timestamps=new BigUint64Array(probe.times.getMappedRange());
   let cursor=0;const mismatches=probe.parts.map(part=>{let count=0;for(let i=0;i<part.bytes/4;i++)if(words[0]![cursor+i]!==words[1]![cursor+i])count++;cursor+=part.bytes/4;return {name:part.name,bytes:part.bytes,mismatches:count};});
   const reference_ms:number[]=[],optimized_ms:number[]=[];
   for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){const index=4*repeat+2*kind;((repeat+kind)%2===0?reference_ms:optimized_ms).push(Number(timestamps[index+1]!-timestamps[index]!)/1e6);}
   console.log(JSON.stringify({frame,time_s:frame*dt,work,mismatches,reference_ms,optimized_ms}));
   for(const result of probe.results){result.unmap();result.destroy();}probe.times.unmap();probe.times.destroy();probe.queries.destroy();for(const resource of probe.resources)resource.destroy();
   assert.ok(mismatches.every(part=>part.mismatches===0),'particle motion gather outputs match bitwise');assert.deepEqual(errors,[]);
  }
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
