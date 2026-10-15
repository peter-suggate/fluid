/** Same-state immutable window-header specialization on frozen solver sources. Inserts replay passes
 * immediately after the first surface advection dispatch at selected frames,
 * before its scratch inputs can be overwritten. Restores production outputs.
 * node --import tsx tools/probe-nbflip-immutable-window.ts --entry=all --frames=1,120,600 --repeats=5
 * Run with the shared GPU lock. */
import assert from 'node:assert/strict';
import {experimentalImmutableWindow} from './experimental-immutable-window';
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const root=resolve(option('root')??'/tmp/fluid-history-final'),load=(file:string)=>import(pathToFileURL(resolve(root,file)).href);
const [{resolveMethodValues},{sceneDocument},{getSceneDefinition},{managedGPUDevice},{uniformNarrowBandMethod:method},{sceneFluidLatticeDimensions,sceneFluidBoxPage},{uniformWindowDerivedWords}]=await Promise.all([
 load('lib/core/method-contract.ts'),load('lib/core/scene-definition.ts'),load('lib/core/scenes.ts'),load('lib/core/gpu-compilation-manager.ts'),load('lib/methods/uniform/uniform-narrow-band-method.ts'),load('lib/core/scene-lattice-dimensions.ts'),load('lib/methods/uniform/uniform-window-address.ts')]);
const definition=getSceneDefinition(option('scene')??'sparse-cm12-long-dam-break'),quality=definition.methodProfile?.quality??'balanced',scene=structuredClone(sceneDocument(definition));
scene.duration_s=4;scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
const policy=scene.container.walls==='none'?3:scene.container.top==='open'?2:1;
const header=uniformWindowDerivedWords(policy,policy===3?sceneFluidBoxPage(scene):[0,0,0],sceneFluidLatticeDimensions(scene,Number.MAX_SAFE_INTEGER).map((x:number)=>x/32));
const implementation=option('implementation')??'literal';assert.ok(['literal','overrides','uniform'].includes(implementation));
const entry=option('entry')??'all';assert.ok(['all','advectOwners','advectFine','advect','advectDeferred','advectWalls'].includes(entry));
const self=option('self')==='true',allowMismatch=option('allow-mismatch')==='true';
const selectedEntries=entry==='all'?['advectOwners','advectFine','advect','advectDeferred','advectWalls']:[entry];
const entryNames=new WeakMap<GPUComputePipeline,string>();
const frames=(option('frames')??'1,24,60').split(',').map(Number).sort((a,b)=>a-b),repeats=Number(option('repeats')??5);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&Number.isInteger(repeats)&&repeats>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const evidencePipelines=new Map<string,GPUComputePipeline>();
const sources=new WeakMap<GPUShaderModule,string>(),references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
const entries=new WeakMap<GPUBindGroup,readonly GPUBindGroupEntry[]>(),textures=new WeakMap<GPUTextureView,GPUTexture>();
let frame=0;const captured=new Set<string>();
type Probe={headers?:GPUBuffer;entry:string;frame:number;results:[GPUBuffer,GPUBuffer,GPUBuffer];times:GPUBuffer;queries:GPUQuerySet;resources:GPUBuffer[];total:number;parts:{name:string;bytes:number}[]};
const pending:Probe[]=[];
const intercepted=new Proxy(raw,{get(target,key){
 if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
 if(key==='createTexture')return (d:GPUTextureDescriptor)=>{
  const texture=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST}),createView=texture.createView.bind(texture);
  Object.defineProperty(texture,'createView',{value:(descriptor?:GPUTextureViewDescriptor)=>{const view=createView(descriptor);textures.set(view,texture);return view;}});return texture;
 };
 if(key==='createBindGroup')return (d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);entries.set(group,[...d.entries]);return group;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const original=sources.get(d.compute.module);
  const selected=selectedEntries.includes(d.compute.entryPoint??'')&&original!==undefined;
  let descriptor=d;
  if(selected){
   if(implementation==='literal'){
    const candidate=experimentalImmutableWindow(original,header);assert.notEqual(candidate,original);
    descriptor={...d,compute:{...d.compute,module:target.createShaderModule({label:'QA immutable window',code:candidate})}};
   }else if(implementation==='overrides'||implementation==='uniform'){
    const names=[...original.matchAll(/override (\w+FixedWorldGeometry):bool=false/g)].map(match=>match[1]!);
    assert.ok(names.length>0,'production source has immutable-world overrides');
    for(const name of names)assert.equal(d.compute.constants?.[name],1,'production descriptor opts in to fixed world');
   }
  }
  const pipeline=await target.createComputePipelineAsync(descriptor);
  if(selected){
   entryNames.set(pipeline,d.compute.entryPoint!);
   let referenceCode=original;
   if(implementation==='uniform'){
    let restored=0;
    referenceCode=original.replace(/\b(\w*)umCounts\.window\[(\d+)u\]\[(\d+)u\]/g,(_read,prefix:string,vector:string,component:string)=>{
     const word=4*Number(vector)+Number(component);assert.ok(word>=0&&word<32);restored++;
     const table=`${prefix}UM_PAGE_TABLE`,call=new RegExp(`\\bfn ${table}\\(`).test(original)?'()':'';
     return `${prefix}umTopology[${table}${call}+${262144+word}u]`;
    });
    assert.ok(restored>0,'uniform header reads were restored to storage for the reference');
   }
   const referenceModule=target.createShaderModule({label:`QA original ${d.compute.entryPoint}`,code:referenceCode});
   const failures=(await referenceModule.getCompilationInfo()).messages.filter(message=>message.type==='error');
   if(failures.length){const path=`/tmp/fluid-surface-reference-${d.compute.entryPoint}.wgsl`;writeFileSync(path,referenceCode);throw new Error(`${path}: ${failures.map(message=>`${message.lineNum}:${message.linePos}: ${message.message}`).join('\n')}`);}
   const constants={...d.compute.constants};
   if(implementation==='overrides')for(const match of original.matchAll(/override (\w+FixedWorldGeometry):bool=false/g))constants[match[1]!]=0;
   references.set(pipeline,self?pipeline:await target.createComputePipelineAsync({...d,label:'QA original surface advection',compute:{...d.compute,module:referenceModule,constants}}));
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
     const queries=target.createQuerySet({type:'timestamp',count:4*repeats}),resolved=make(32*repeats,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(32*repeats,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
     const snapshot=(buffer:GPUBuffer,offset:number)=>{
      for(const {part,at} of locations)e.copyBufferToBuffer(part.buffer,part.offset??0,buffer,offset+at,part.size);
      for(const image of images)e.copyTextureToBuffer({texture:image.texture},{buffer,offset:offset+image.at,bytesPerRow:image.bytesPerRow,rowsPerImage:image.texture.height},[image.texture.width,image.texture.height,image.texture.depthOrArrayLayers]);
     };
     const restore=(buffer:GPUBuffer)=>{
      for(const {part,at} of locations)e.copyBufferToBuffer(buffer,at,part.buffer,part.offset??0,part.size);
      for(const image of images)e.copyBufferToTexture({buffer,offset:image.at,bytesPerRow:image.bytesPerRow,rowsPerImage:image.texture.height},{texture:image.texture},[image.texture.width,image.texture.height,image.texture.depthOrArrayLayers]);
     };
     let headers:GPUBuffer|undefined;
     if(implementation==='uniform'){
      const ownerEntries=entries.get(bindings.get(0)![0] as GPUBindGroup)!;
      const topology=ownerEntries.find(entry=>entry.binding===0)!.resource as GPUBufferBinding,counts=ownerEntries.find(entry=>entry.binding===1)!.resource as GPUBufferBinding;
      const tiles=(solver as unknown as {mixedFrame:{ownership:{capacity:{tiles:number}}}}).mixedFrame.ownership.capacity.tiles;
      headers=make(256,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      e.copyBufferToBuffer(counts.buffer,(counts.offset??0)+48,headers,0,128);
      e.copyBufferToBuffer(topology.buffer,(topology.offset??0)+4*(5*tiles+262144),headers,128,128);
     }
     snapshot(initial,0);
     // First execute the production dispatch and retain its exact outputs.
     pass=e.beginComputePass();bind();Reflect.apply(pass[method],pass,args);pass.end();snapshot(saved,0);
     for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){
      const original=(repeat+kind)%2===0,index=4*repeat+2*kind;restore(initial);
      pass=e.beginComputePass({label:original?'QA surface advection original':'QA surface advection optimized',timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});bind();pass.setPipeline(original?reference:pipeline!);Reflect.apply(pass[method],pass,args);pass.end();
      if(repeat===0)snapshot(results[original?0:1],0);
      if(repeat===repeats-1&&!original)snapshot(results[2],0);
     }
     restore(saved);
     e.resolveQuerySet(queries,0,4*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,32*repeats);
     pending.push({headers,entry:actualEntry,frame,results,times,queries,resources:[initial,saved,resolved],total,parts});
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
const device:GPUDevice=managedGPUDevice(intercepted,{requireWorkerRealm:false}),errors:string[]=[];
device.addEventListener('uncapturederror',event=>{event.preventDefault();errors.push(event.error.message);});
let solver:WebGPUUniformReferenceSolver|undefined;
try{

 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile?.overrides}),dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(frame=1;frame<=frames.at(-1)!;frame++){
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  if(!frames.includes(frame))continue;for(const selected of selectedEntries)assert.ok(captured.has(`${frame}:${selected}`),`${selected} dispatch was captured`);
  for(const probe of pending.splice(0)){
   await Promise.all([...probe.results.map(buffer=>buffer.mapAsync(GPUMapMode.READ)),probe.times.mapAsync(GPUMapMode.READ),...(probe.headers?[probe.headers.mapAsync(GPUMapMode.READ)]:[])]);
   const allWords=probe.results.map(buffer=>new Uint32Array(buffer.getMappedRange())),timestamps=new BigUint64Array(probe.times.getMappedRange());
   const compare=(a:number,b:number)=>{const words=[allWords[a]!,allWords[b]!];let cursor=0;return probe.parts.map(part=>{let count=0,maxUlp=0,maxAbsolute=0;const first:{index:number;reference:number;candidate:number;referenceBits:number;candidateBits:number;ulp:number}[]=[];
    if(part.name==='deferred'){
     const first=words[0]!.subarray(cursor,cursor+part.bytes/4).slice(),second=words[1]!.subarray(cursor,cursor+part.bytes/4).slice();
     assert.equal(first[0],second[0],'deferred vertex count');const vertices=first[0]!;assert.ok(vertices+4<=first.length);
     first.subarray(4,4+vertices).sort();second.subarray(4,4+vertices).sort();
     for(let i=0;i<first.length;i++)if(first[i]!==second[i])count++;
    }else{for(let i=0;i<part.bytes/4;i++)if(words[0]![cursor+i]!==words[1]![cursor+i]){
     count++;const referenceBits=words[0]![cursor+i]!,candidateBits=words[1]![cursor+i]!,values=new Float32Array(Uint32Array.of(referenceBits,candidateBits).buffer);maxUlp=Math.max(maxUlp,Math.abs(referenceBits-candidateBits));maxAbsolute=Math.max(maxAbsolute,Math.abs(values[1]!-values[0]!));if(first.length<10){first.push({index:i,reference:values[0]!,candidate:values[1]!,referenceBits,candidateBits,ulp:Math.abs(referenceBits-candidateBits)});}
    }}
    cursor+=part.bytes/4;return {name:part.name,bytes:part.bytes,mismatches:count,maxUlp,maxAbsolute,first};});};
   const mismatches=compare(0,1),selfMismatches=compare(1,2);
   const reference_ms:number[]=[],optimized_ms:number[]=[];
   for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){const index=4*repeat+2*kind;((repeat+kind)%2===0?reference_ms:optimized_ms).push(Number(timestamps[index+1]!-timestamps[index]!)/1e6);}
   const headerWords=probe.headers?Array.from(new Uint32Array(probe.headers.getMappedRange())):undefined;
   const headerDifferences=headerWords?headerWords.slice(0,32).flatMap((word,index)=>word===headerWords[32+index]?[]:[{index,uniform:word,storage:headerWords[32+index]}]):undefined;
   console.log(JSON.stringify({implementation,self,headerDifferences,entry:probe.entry,frame,time_s:frame*dt,mismatches,selfMismatches,reference_ms,optimized_ms}));
   if(headerDifferences)assert.deepEqual(headerDifferences,[],'uniform and storage metadata agree at replay input');
   if(probe.headers){probe.headers.unmap();probe.headers.destroy();}
   for(const result of probe.results){result.unmap();result.destroy();}probe.times.unmap();probe.times.destroy();probe.queries.destroy();for(const resource of probe.resources)resource.destroy();
   if(!allowMismatch)assert.ok(mismatches.every(part=>part.mismatches===0),'surface advection outputs match bitwise');assert.deepEqual(errors,[]);
  }
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
