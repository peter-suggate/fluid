/** Same-dispatch pressure-authority parity and GPU timing. Inserts replay passes
 * immediately after the first simulation authority dispatch at selected frames,
 * before its scratch inputs can be overwritten. Restores production outputs.
 * node --import tsx tools/probe-nbflip-pressure-authority.ts --frames=1,120,480 --repeats=5
 * Run with the shared GPU lock. */
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
const frames=(option('frames')??'1,120,480').split(',').map(Number).sort((a,b)=>a-b),repeats=Number(option('repeats')??5);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const sources=new WeakMap<GPUShaderModule,string>(),references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
const entries=new WeakMap<GPUBindGroup,readonly GPUBindGroupEntry[]>(),textures=new WeakMap<GPUTextureView,GPUTexture>();
let frame=0,capturedFrame=-1;
type Probe={frame:number;results:[GPUBuffer,GPUBuffer];times:GPUBuffer;queries:GPUQuerySet;resources:GPUBuffer[];total:number;parts:{name:string;bytes:number}[]};
const pending:Probe[]=[];
function referenceSource(source:string){
 const phase=/fn umPhase\(o:UMOwner,liquid:bool\)->f32\{[\s\S]*?\n\}/;
 assert.match(source,phase);source=source.replace(phase,'fn umPhase(o:UMOwner,liquid:bool)->f32{return select(0.0,1.0,liquid&&umCapacity(o)>1e-5); }');
 const guard='terms[i]=0.0;weights[i]=0.0;if(!umCellIsOpen(q)){continue;}';assert.ok(source.includes(guard));
 source=source.replace(guard,'terms[i]=0.0;weights[i]=0.0;');return source;
}
const intercepted=new Proxy(raw,{get(target,key){
 if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
 if(key==='createTexture')return (d:GPUTextureDescriptor)=>{
  const texture=target.createTexture(d),createView=texture.createView.bind(texture);
  Object.defineProperty(texture,'createView',{value:(descriptor?:GPUTextureViewDescriptor)=>{const view=createView(descriptor);textures.set(view,texture);return view;}});return texture;
 };
 if(key==='createBindGroup')return (d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);entries.set(group,[...d.entries]);return group;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d),source=sources.get(d.compute.module);
  // Simulation authority only: resident pressure ownership has a binding 9.
  if(d.compute.entryPoint==='surface'&&source?.includes('fn umPressurePhiCell')&&!source.includes('var<storage,read> umSimulationTiles')){
   const module=target.createShaderModule({code:referenceSource(source)});
   references.set(pipeline,await target.createComputePipelineAsync({...d,label:'QA original pressure authority',compute:{...d.compute,module}}));
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
     const phi=groupEntries.find(entry=>entry.binding===3)!.resource as GPUBufferBinding;
     const imageParts=[4,5].map(binding=>{const texture=textures.get(groupEntries.find(entry=>entry.binding===binding)!.resource as GPUTextureView);assert.ok(texture);const bytesPerRow=Math.ceil(texture.width*4/256)*256;return {texture,bytesPerRow,bytes:bytesPerRow*texture.height*texture.depthOrArrayLayers};});
     const phiBytes=phi.size??phi.buffer.size-(phi.offset??0),imageOffset=Math.ceil(phiBytes/256)*256,parts=[{name:'phi',bytes:phiBytes},{name:'padding',bytes:imageOffset-phiBytes},...imageParts.map((p,i)=>({name:i?'correction':'phase',bytes:p.bytes}))],total=parts.reduce((n,p)=>n+p.bytes,0);
     const make=(size:number,usage:number)=>target.createBuffer({size,usage});
     assert.ok(total<=target.limits.maxBufferSize,`authority snapshot ${total} exceeds buffer limit ${target.limits.maxBufferSize}`);
     const saved=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),results:[GPUBuffer,GPUBuffer]=[make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ)];
     const queries=target.createQuerySet({type:'timestamp',count:4*repeats}),resolved=make(32*repeats,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(32*repeats,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
     const snapshot=(buffer:GPUBuffer,offset:number)=>{e.copyBufferToBuffer(phi.buffer,phi.offset??0,buffer,offset,phiBytes);offset+=imageOffset;for(const p of imageParts){e.copyTextureToBuffer({texture:p.texture},{buffer,offset,bytesPerRow:p.bytesPerRow,rowsPerImage:p.texture.height},[p.texture.width,p.texture.height,p.texture.depthOrArrayLayers]);offset+=p.bytes;}};
     // First execute the production dispatch and retain its exact outputs.
     pass=e.beginComputePass();bind();Reflect.apply(pass[method],pass,args);pass.end();snapshot(saved,0);
     for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){
      const original=(repeat+kind)%2===0,index=4*repeat+2*kind;
      pass=e.beginComputePass({label:original?'QA authority original':'QA authority optimized',timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});bind();pass.setPipeline(original?reference:pipeline!);Reflect.apply(pass[method],pass,args);pass.end();
      if(repeat===0)snapshot(results[original?0:1],0);
     }
     e.copyBufferToBuffer(saved,0,phi.buffer,phi.offset??0,phiBytes);let offset=imageOffset;
     for(const p of imageParts){e.copyBufferToTexture({buffer:saved,offset,bytesPerRow:p.bytesPerRow,rowsPerImage:p.texture.height},{texture:p.texture},[p.texture.width,p.texture.height,p.texture.depthOrArrayLayers]);offset+=p.bytes;}
     e.resolveQuerySet(queries,0,4*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,32*repeats);
     pending.push({frame,results,times,queries,resources:[saved,resolved],total,parts});
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
  if(!frames.includes(frame))continue;assert.equal(capturedFrame,frame,'authority dispatch was captured');
  for(const probe of pending.splice(0)){
   await Promise.all([...probe.results.map(buffer=>buffer.mapAsync(GPUMapMode.READ)),probe.times.mapAsync(GPUMapMode.READ)]);
   const words=probe.results.map(buffer=>new Uint32Array(buffer.getMappedRange())),timestamps=new BigUint64Array(probe.times.getMappedRange());
   let cursor=0;const mismatches=probe.parts.map(part=>{let count=0;for(let i=0;i<part.bytes/4;i++)if(words[0]![cursor+i]!==words[1]![cursor+i])count++;cursor+=part.bytes/4;return {name:part.name,bytes:part.bytes,mismatches:count};});
   const reference_ms:number[]=[],optimized_ms:number[]=[];
   for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){const index=4*repeat+2*kind;((repeat+kind)%2===0?reference_ms:optimized_ms).push(Number(timestamps[index+1]!-timestamps[index]!)/1e6);}
   console.log(JSON.stringify({frame,time_s:frame*dt,mismatches,reference_ms,optimized_ms}));
   for(const result of probe.results){result.unmap();result.destroy();}probe.times.unmap();probe.times.destroy();probe.queries.destroy();for(const resource of probe.resources)resource.destroy();
   assert.ok(mismatches.every(part=>part.mismatches===0),'authority outputs match bitwise');assert.deepEqual(errors,[]);
  }
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
