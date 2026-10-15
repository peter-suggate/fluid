/** Frozen-phi halo prototype; each candidate dispatch includes a full halo refresh, with refresh also timed separately. Inserts replay passes
 * immediately after the first surface advection dispatch at selected frames,
 * before its scratch inputs can be overwritten. Restores production outputs.
 * node --import tsx tools/probe-nbflip-surface-halo.ts --entry=all --frames=1,120,600 --repeats=5
 * Run with the shared GPU lock.
 * --audit-taps=true is diagnostic only: every consumed cubic halo tap is
 * compared bitwise with the original loader; a mismatch marks final phi
 * -1e20 after wall/source continuation. The replay still enforces bitwise
 * output parity, so a final float arithmetic discrepancy also exits 1.
 * The second original replay (controlMismatches) distinguishes a baseline
 * race from a candidate difference. Deferred-list ordering is normalized.
 */
import assert from 'node:assert/strict';
import {surfaceHaloWGSL,SURFACE_HALO_STRIDE} from './experimental-surface-halo';
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
const entry=option('entry')??'all';assert.ok(['all','advectOwners','advectFine','advect','advectDeferred'].includes(entry));
const selectedEntries=entry==='all'?['advectOwners','advectFine','advect','advectDeferred']:[entry];
const entryNames=new WeakMap<GPUComputePipeline,string>();
const frames=(option('frames')??'1,24,60').split(',').map(Number).sort((a,b)=>a-b),repeats=Number(option('repeats')??5);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const layoutEntries=new WeakMap<GPUBindGroupLayout,GPUBindGroupLayoutEntry[]>();
const pipelineLayouts=new WeakMap<GPUPipelineLayout,GPUBindGroupLayout[]>();
const halos=new WeakMap<GPUComputePipeline,{layout:GPUBindGroupLayout;fill:GPUComputePipeline}>();
const sources=new WeakMap<GPUShaderModule,string>(),references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
const entries=new WeakMap<GPUBindGroup,readonly GPUBindGroupEntry[]>(),textures=new WeakMap<GPUTextureView,GPUTexture>();
let frame=0;const captured=new Set<string>();
type Probe={entry:string;frame:number;results:[GPUBuffer,GPUBuffer,GPUBuffer];times:GPUBuffer;queries:GPUQuerySet;resources:(GPUBuffer|GPUTexture)[];total:number;parts:{name:string;bytes:number}[]};
const pending:Probe[]=[];
const intercepted=new Proxy(raw,{get(target,key){
 if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
 if(key==='createTexture')return (d:GPUTextureDescriptor)=>{
  const texture=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST}),createView=texture.createView.bind(texture);
  Object.defineProperty(texture,'createView',{value:(descriptor?:GPUTextureViewDescriptor)=>{const view=createView(descriptor);textures.set(view,texture);return view;}});return texture;
 };
 if(key==='createBindGroupLayout')return (d:GPUBindGroupLayoutDescriptor)=>{const layout=target.createBindGroupLayout(d);layoutEntries.set(layout,[...d.entries]);return layout;};
 if(key==='createPipelineLayout')return (d:GPUPipelineLayoutDescriptor)=>{const layout=target.createPipelineLayout(d);pipelineLayouts.set(layout,[...d.bindGroupLayouts].map(item=>{assert.ok(item);return item;}));return layout;};
 if(key==='createBindGroup')return (d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);entries.set(group,[...d.entries]);return group;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d),source=sources.get(d.compute.module);
  if(selectedEntries.includes(d.compute.entryPoint??'')&&source){
   entryNames.set(pipeline,d.compute.entryPoint!);
   assert.notEqual(d.layout,'auto');
   const layouts=pipelineLayouts.get(d.layout as GPUPipelineLayout);assert.ok(layouts);
   const resourceEntries=layoutEntries.get(layouts[1]!);assert.ok(resourceEntries);assert.ok(!resourceEntries.some(e=>e.binding===14));
   const resourceLayout=target.createBindGroupLayout({entries:[...resourceEntries,{binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d'}},{binding:15,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'r32float',viewDimension:'2d'}}]});
   const alternative=[...layouts];alternative[1]=resourceLayout;
   const layout=target.createPipelineLayout({bindGroupLayouts:alternative}),code=surfaceHaloWGSL(source,option('audit-taps')==='true');
   const module=target.createShaderModule({label:'QA frozen-phi surface halo',code});
   const failures=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
   if(failures.length){writeFileSync('/tmp/fluid-surface-halo.wgsl',code);throw new Error(failures.map(x=>`${x.lineNum}: ${x.message}`).join('\n'));}
   const candidate=await target.createComputePipelineAsync({...d,layout,compute:{...d.compute,module}});
   const fill=await target.createComputePipelineAsync({...d,layout,compute:{...d.compute,module,entryPoint:'qaFillSurfaceHalo'}});
   references.set(pipeline,candidate);halos.set(pipeline,{layout:resourceLayout,fill});
  }
  return pipeline;
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
     const configuration=halos.get(pipeline!);assert.ok(configuration);
     const inputView=groupEntries.find(x=>x.binding===0)!.resource as GPUTextureView,input=textures.get(inputView);assert.ok(input);
     const edge=input.width%33===0?33:9;
     const blocks=(input.width/edge)*(input.height/edge)*(input.depthOrArrayLayers/edge);
     const halo=target.createTexture({size:[4096,Math.ceil(blocks*SURFACE_HALO_STRIDE/4096)],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});const haloView=halo.createView();
     const dummyHalo=target.createTexture({size:[1,1],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}),dummyView=dummyHalo.createView();
     const haloGroup=target.createBindGroup({layout:configuration.layout,entries:[...groupEntries,{binding:14,resource:haloView},{binding:15,resource:dummyView}]});
     const fillGroup=target.createBindGroup({layout:configuration.layout,entries:[...groupEntries,{binding:14,resource:dummyView},{binding:15,resource:haloView}]});
     const haloJobs=Math.ceil(blocks*SURFACE_HALO_STRIDE/64),haloX=Math.min(haloJobs,target.limits.maxComputeWorkgroupsPerDimension),haloY=Math.ceil(haloJobs/haloX);
     const bufferParts=[10,12,13].map(binding=>{const bindingResource=groupEntries.find(entry=>entry.binding===binding)!.resource as GPUBufferBinding;return {name:binding===10?'deferred':binding===12?'claims':'status',...bindingResource,size:bindingResource.size??bindingResource.buffer.size-(bindingResource.offset??0)};});
     const imageParts=[1,65].flatMap(binding=>{
      const view=groupEntries.find(resource=>resource.binding===binding)?.resource as GPUTextureView|undefined;if(!view)return [];
      const texture=textures.get(view);assert.ok(texture);const bytesPerRow=Math.ceil(texture.width*4/256)*256;
      return [{name:binding===1?'phi':'phiBase',texture,bytesPerRow,bytes:bytesPerRow*texture.height*texture.depthOrArrayLayers}];
     });
     let bufferBytes=0;const locations=bufferParts.map(part=>{const at=bufferBytes;bufferBytes+=part.size;return {part,at};});
     const imageOffset=Math.ceil(bufferBytes/256)*256;let total=imageOffset;
     const images=imageParts.map(part=>{const at=total;total+=part.bytes;return {...part,at};});
     const parts=[...bufferParts.map(part=>({name:part.name,bytes:part.size})),{name:'padding',bytes:imageOffset-bufferBytes},...images.map(part=>({name:part.name,bytes:part.bytes}))];
     const make=(size:number,usage:number)=>target.createBuffer({size,usage});
     assert.ok(total<=target.limits.maxBufferSize,`surface advection snapshot ${total} exceeds buffer limit ${target.limits.maxBufferSize}`);
     const initial=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),saved=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),results:[GPUBuffer,GPUBuffer,GPUBuffer]=[make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ)];
     const queries=target.createQuerySet({type:'timestamp',count:6*repeats}),resolved=make(48*repeats,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(48*repeats,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
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
     for(let repeat=0;repeat<repeats;repeat++){
      for(let kind=0;kind<2;kind++){
       const original=(repeat+kind)%2===0,index=6*repeat+2*kind;restore(initial);
       pass=e.beginComputePass({label:original?'QA generic surface':'QA halo refresh plus surface',timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});bind();
       if(!original){pass.setBindGroup(1,fillGroup);pass.setPipeline(configuration.fill);pass.dispatchWorkgroups(haloX,haloY);pass.setBindGroup(1,haloGroup);pass.setPipeline(reference);}
       Reflect.apply(pass[method],pass,args);pass.end();
       if(repeat===0)snapshot(results[original?0:1],0);if(repeat===1&&original)snapshot(results[2],0);
      }
      const index=6*repeat+4;pass=e.beginComputePass({label:'QA halo refresh alone',timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});bind();pass.setBindGroup(1,fillGroup);pass.setPipeline(configuration.fill);pass.dispatchWorkgroups(haloX,haloY);pass.end();
     }
     restore(saved);
     e.resolveQuerySet(queries,0,6*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,48*repeats);
     pending.push({entry:actualEntry,frame,results,times,queries,resources:[initial,saved,resolved,halo,dummyHalo],total,parts});
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
device.addEventListener('uncapturederror',event=>{event.preventDefault();errors.push(event.error.message);console.error(event.error.message);});
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
   let cursor=0;const mismatchExamples:{part:string;word:number;before:number;after:number;beforeFloat:number;afterFloat:number}[]=[];const floats=words.map(w=>new Float32Array(w.buffer,w.byteOffset,w.length));const mismatches=probe.parts.map(part=>{let count=0,controlMismatches=0;if(repeats>1&&part.name!=='deferred')for(let i=0;i<part.bytes/4;i++)if(words[0]![cursor+i]!==words[2]![cursor+i])controlMismatches++;
    if(part.name==='deferred'){
     const first=words[0]!.subarray(cursor,cursor+part.bytes/4).slice(),second=words[1]!.subarray(cursor,cursor+part.bytes/4).slice();
     assert.equal(first[0],second[0],'deferred vertex count');const vertices=first[0]!;assert.ok(vertices+4<=first.length);
     first.subarray(4,4+vertices).sort();second.subarray(4,4+vertices).sort();
     for(let i=0;i<first.length;i++)if(first[i]!==second[i])count++;
    }else{for(let i=0;i<part.bytes/4;i++)if(words[0]![cursor+i]!==words[1]![cursor+i]){count++;if(mismatchExamples.length<8)mismatchExamples.push({part:part.name,word:i,before:words[0]![cursor+i]!,after:words[1]![cursor+i]!,beforeFloat:floats[0]![cursor+i]!,afterFloat:floats[1]![cursor+i]!});}}
    cursor+=part.bytes/4;return {name:part.name,bytes:part.bytes,mismatches:count,controlMismatches};});
   const reference_ms:number[]=[],optimized_ms:number[]=[];
   for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){const index=6*repeat+2*kind;((repeat+kind)%2===0?reference_ms:optimized_ms).push(Number(timestamps[index+1]!-timestamps[index]!)/1e6);}
   const refresh_ms=Array.from({length:repeats},(_,r)=>Number(timestamps[6*r+5]!-timestamps[6*r+4]!)/1e6);
   console.log(JSON.stringify({entry:probe.entry,frame,auditTaps:option('audit-taps')==='true',time_s:frame*dt,mismatches,mismatchExamples,reference_ms,refresh_plus_consumer_ms:optimized_ms,refresh_ms}));
   for(const result of probe.results){result.unmap();result.destroy();}probe.times.unmap();probe.times.destroy();probe.queries.destroy();for(const resource of probe.resources)resource.destroy();
   assert.ok(mismatches.every(part=>part.mismatches===0),'surface advection outputs match bitwise');assert.deepEqual(errors,[]);
  }
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
