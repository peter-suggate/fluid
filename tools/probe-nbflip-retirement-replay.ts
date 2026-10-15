/** Same-dispatch coarse-retirement heat-test parity and GPU timing. Inserts replay passes
 * immediately after the coarse redistance dispatch at selected frames,
 * before its scratch inputs can be overwritten. Restores production outputs.
 * node --import tsx tools/probe-nbflip-retirement-replay.ts --frames=1,120,600 --repeats=5
 * Run with the shared GPU lock. */
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
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
const selectedEntries=['redistanceCoarse'];
const entryNames=new WeakMap<GPUComputePipeline,string>();
const frames=(option('frames')??'1,24,60').split(',').map(Number).sort((a,b)=>a-b),repeats=Number(option('repeats')??5);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const sources=new WeakMap<GPUShaderModule,string>(),references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
const entries=new WeakMap<GPUBindGroup,readonly GPUBindGroupEntry[]>(),textures=new WeakMap<GPUTextureView,GPUTexture>();
let frame=0;const captured=new Set<string>();
type Probe={entry:string;frame:number;results:[GPUBuffer,GPUBuffer];times:GPUBuffer;queries:GPUQuerySet;resources:GPUBuffer[];total:number;parts:{name:string;bytes:number}[]};
const pending:Probe[]=[];
const intercepted=new Proxy(raw,{get(target,key){
 if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
 if(key==='createTexture')return (d:GPUTextureDescriptor)=>{
  const texture=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST}),createView=texture.createView.bind(texture);
  Object.defineProperty(texture,'createView',{value:(descriptor?:GPUTextureViewDescriptor)=>{const view=createView(descriptor);textures.set(view,texture);return view;}});return texture;
 };
 if(key==='createBindGroup')return (d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);entries.set(group,[...d.entries]);return group;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d),source=sources.get(d.compute.module);
  if(selectedEntries.includes(d.compute.entryPoint??'')&&source){
   entryNames.set(pipeline,d.compute.entryPoint!);
   const original=source.replace('!nbSurfaceHotCornerFrom(vec3i(origin>>vec3u(2u)),p)','nbSurfaceThetaFrom(vec3i(origin>>vec3u(2u)),vec3f(p))<=0.0');
   assert.notEqual(original,source);
   const referenceModule=target.createShaderModule({label:`QA original ${d.compute.entryPoint}`,code:original});
   const failures=(await referenceModule.getCompilationInfo()).messages.filter(message=>message.type==='error');
   if(failures.length){const path=`/tmp/fluid-retirement-reference-${d.compute.entryPoint}.wgsl`;writeFileSync(path,original);throw new Error(`${path}: ${failures.map(message=>`${message.lineNum}:${message.linePos}: ${message.message}`).join('\n')}`);}
   references.set(pipeline,await target.createComputePipelineAsync({...d,label:'QA original coarse retirement',compute:{...d.compute,module:referenceModule}}));
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
     const bufferParts:{name:string;buffer:GPUBuffer;offset?:number;size:number}[]=[];
     const imageParts=[14,78].flatMap(binding=>{
      const view=groupEntries.find(resource=>resource.binding===binding)?.resource as GPUTextureView|undefined;if(!view)return [];
      const texture=textures.get(view);assert.ok(texture);const bytesPerRow=Math.ceil(texture.width*4/256)*256;
      return [{name:binding===14?'outputPhi':'outputPhiBase',texture,bytesPerRow,bytes:bytesPerRow*texture.height*texture.depthOrArrayLayers}];
     });
     let bufferBytes=0;const locations=bufferParts.map(part=>{const at=bufferBytes;bufferBytes+=part.size;return {part,at};});
     const imageOffset=Math.ceil(bufferBytes/256)*256;let total=imageOffset;
     const images=imageParts.map(part=>{const at=total;total+=part.bytes;return {...part,at};});
     const parts=[...bufferParts.map(part=>({name:part.name,bytes:part.size})),{name:'padding',bytes:imageOffset-bufferBytes},...images.map(part=>({name:part.name,bytes:part.bytes}))];
     const make=(size:number,usage:number)=>target.createBuffer({size,usage});
     assert.ok(total<=target.limits.maxBufferSize,`coarse retirement snapshot ${total} exceeds buffer limit ${target.limits.maxBufferSize}`);
     const initial=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),saved=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),results:[GPUBuffer,GPUBuffer]=[make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ)];
     const queries=target.createQuerySet({type:'timestamp',count:4*repeats}),resolved=make(32*repeats,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(32*repeats,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
     const snapshot=(buffer:GPUBuffer,offset:number)=>{
      for(const {part,at} of locations)e.copyBufferToBuffer(part.buffer,part.offset??0,buffer,offset+at,part.size);
      for(const image of images)e.copyTextureToBuffer({texture:image.texture},{buffer,offset:offset+image.at,bytesPerRow:image.bytesPerRow,rowsPerImage:image.texture.height},[image.texture.width,image.texture.height,image.texture.depthOrArrayLayers]);
     };
     const restore=(buffer:GPUBuffer)=>{
      for(const {part,at} of locations)e.copyBufferToBuffer(buffer,at,part.buffer,part.offset??0,part.size);
      for(const image of images)e.copyBufferToTexture({buffer,offset:image.at,bytesPerRow:image.bytesPerRow,rowsPerImage:image.texture.height},{texture:image.texture},[image.texture.width,image.texture.height,image.texture.depthOrArrayLayers]);
     };
     snapshot(initial,0);
     // First execute the production dispatch and retain its exact outputs.
     pass=e.beginComputePass();bind();Reflect.apply(pass[method],pass,args);pass.end();snapshot(saved,0);
     for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){
      const original=(repeat+kind)%2===0,index=4*repeat+2*kind;restore(initial);
      pass=e.beginComputePass({label:original?'QA coarse retirement original':'QA coarse retirement optimized',timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});bind();pass.setPipeline(original?reference:pipeline!);Reflect.apply(pass[method],pass,args);pass.end();
      if(repeat===0)snapshot(results[original?0:1],0);
     }
     restore(saved);
     e.resolveQuerySet(queries,0,4*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,32*repeats);
     pending.push({entry:actualEntry,frame,results,times,queries,resources:[initial,saved,resolved],total,parts});
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
 const definition=getSceneDefinition(option('scene')??'minimal-power-dam-break-32'),quality=definition.methodProfile?.quality??'balanced',scene=structuredClone(sceneDocument(definition));scene.duration_s=4;scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile?.overrides}),dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
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
   for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){const index=4*repeat+2*kind;((repeat+kind)%2===0?reference_ms:optimized_ms).push(Number(timestamps[index+1]!-timestamps[index]!)/1e6);}
   console.log(JSON.stringify({entry:probe.entry,frame,time_s:frame*dt,mismatches,reference_ms,optimized_ms}));
   for(const result of probe.results){result.unmap();result.destroy();}probe.times.unmap();probe.times.destroy();probe.queries.destroy();for(const resource of probe.resources)resource.destroy();
   assert.ok(mismatches.every(part=>part.mismatches===0),'coarse retirement outputs match bitwise');assert.deepEqual(errors,[]);
  }
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
