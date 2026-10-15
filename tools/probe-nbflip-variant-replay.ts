/** Same-state GPU replay of one or more text variants of selected NB-FLIP entries.
 * At each selected frame the first dispatch of each selected entry is captured:
 * every writable resource its shader declares is snapshotted, the production
 * dispatch runs and its outputs are kept, then each variant runs once from the
 * same inputs (outputs compared word by word, with the largest f32 ULP gap),
 * then timed passes alternate the variants in rotating order, each pass
 * dispatching its pipeline `inner` times back to back (Dawn timestamps are
 * quantized to 65.536 us, so one dispatch is too coarse). Production outputs
 * are restored before the frame continues.
 * node --import tsx tools/probe-nbflip-variant-replay.ts --scene=sparse-cm12-long-dam-break \
 *   --frames=60 --entries=advectParticles --variants=aa,ctx --repeats=5 --inner=8 [--dump=DIR] [--nth=0] [--scene-file=JSON]
 * Variants live in tools/probe-nbflip-variants.ts. Run with the shared GPU lock. */
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
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
import {probeVariants} from './probe-nbflip-variants';
import {GPUPassProfile} from './gpu-pass-profile';
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const selected=(option('entries')??'advectParticles').split(',');
const variantNames=(option('variants')??'aa').split(',');
for(const name of variantNames)assert.ok(name==='aa'||probeVariants[name],`unknown variant ${name}`);
const frames=(option('frames')??'60').split(',').map(Number).sort((a,b)=>a-b),repeats=Number(option('repeats')??5),inner=Number(option('inner')??8),nth=Number(option('nth')??0);
const dump=option('dump');if(dump)mkdirSync(dump,{recursive:true});
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0)&&repeats>0&&inner>0);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const rawDevice=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const splitFrames=(option('split-frames')??'').split(',').filter(Boolean).map(Number),entryOf=new WeakMap<GPUComputePipeline,string>();
const profileFrames=[...(option('profile-frames')??'').split(',').filter(Boolean).map(Number),...splitFrames],profiler=profileFrames.length?new GPUPassProfile(rawDevice):undefined,raw=profiler?.device??rawDevice;
const sources=new WeakMap<GPUShaderModule,string>(),variantsOf=new WeakMap<GPUComputePipeline,{entry:string;writable:Set<string>;pipelines:GPUComputePipeline[]}>();
const entries=new WeakMap<GPUBindGroup,readonly GPUBindGroupEntry[]>(),textures=new WeakMap<GPUTextureView,GPUTexture>();
let frame=0;const seen=new Map<string,number>();const captured=new Set<string>();
type Part={name:string;bytes:number;words:number};
type Probe={entry:string;frame:number;results:GPUBuffer[];times:GPUBuffer;queries:GPUQuerySet;resources:GPUBuffer[];parts:Part[];total:number};
const pending:Probe[]=[];const dumped=new Set<string>();
/** group:binding of every resource the module declares writable. */
function writableBindings(code:string):Set<string>{
 const out=new Set<string>();
 for(const m of code.matchAll(/@group\((\d+)\)\s*@binding\((\d+)\)\s*var<storage,\s*read_write>/g))out.add(`${m[1]}:${m[2]}`);
 for(const m of code.matchAll(/@group\((\d+)\)\s*@binding\((\d+)\)\s*var\s+\w+\s*:\s*texture_storage_\w+<\w+,\s*(?:write|read_write)>/g))out.add(`${m[1]}:${m[2]}`);
 return out;
}
const intercepted=new Proxy(raw,{get(target,key){
 if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
 if(key==='createTexture')return (d:GPUTextureDescriptor)=>{
  const texture=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST}),createView=texture.createView.bind(texture);
  Object.defineProperty(texture,'createView',{value:(descriptor?:GPUTextureViewDescriptor)=>{const view=createView(descriptor);textures.set(view,texture);return view;}});return texture;
 };
 // Snapshots copy every writable storage buffer: give each copy usages (never on a mappable buffer).
 if(key==='createBuffer')return (d:GPUBufferDescriptor)=>target.createBuffer((d.usage&GPUBufferUsage.STORAGE)&&!(d.usage&(GPUBufferUsage.MAP_READ|GPUBufferUsage.MAP_WRITE))?{...d,usage:d.usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}:d);
 if(key==='createBindGroup')return (d:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(d);entries.set(group,[...d.entries]);return group;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const pipeline=await target.createComputePipelineAsync(d),source=sources.get(d.compute.module),entry=d.compute.entryPoint??'';entryOf.set(pipeline,entry);
  if(selected.includes(entry)&&source){
   if(dump&&!dumped.has(entry)){dumped.add(entry);writeFileSync(`${dump}/${entry}.wgsl`,source);writeFileSync(`${dump}/${entry}.constants.json`,JSON.stringify(d.compute.constants??{}));}
   const pipelines:GPUComputePipeline[]=[];
   for(const name of variantNames){
    if(name==='aa'){pipelines.push(await target.createComputePipelineAsync({...d,label:`QA ${entry} aa`}));continue;}
    const code=probeVariants[name]!(source,entry);
    if(code===source){console.log(JSON.stringify({warning:'variant leaves the source unchanged',entry,variant:name}));}
    const module=target.createShaderModule({label:`QA ${entry} ${name}`,code});
    const failures=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
    if(failures.length){const path=`${dump??'/tmp'}/variant-${name}-${entry}.wgsl`;writeFileSync(path,code);throw new Error(`${path}: ${failures.map(message=>`${message.lineNum}:${message.linePos}: ${message.message}`).join('\n')}`);}
    if(dump)writeFileSync(`${dump}/variant-${name}-${entry}.wgsl`,code);
    const started=performance.now();
    pipelines.push(await target.createComputePipelineAsync({...d,label:`QA ${entry} ${name}`,compute:{...d.compute,module}}));
    console.log(JSON.stringify({compiled:name,entry,ms:Math.round(performance.now()-started)}));
   }
   variantsOf.set(pipeline,{entry,writable:writableBindings(source),pipelines});
  }return pipeline;
 };
 if(key==='createCommandEncoder')return (d?:GPUCommandEncoderDescriptor)=>{
  const encoder=target.createCommandEncoder(d);
  return new Proxy(encoder,{get(e,key){
   if(key==='beginComputePass')return (descriptor?:GPUComputePassDescriptor)=>{
    let pass=e.beginComputePass(descriptor),pipeline:GPUComputePipeline|undefined;
    const bindings=new Map<number,unknown[]>();
    const bind=(use?:GPUComputePipeline)=>{assert.ok(pipeline);pass.setPipeline(use??pipeline);for(const [index,args] of bindings)Reflect.apply(pass.setBindGroup,pass,[index,...args]);};
    const dispatch=(method:'dispatchWorkgroups'|'dispatchWorkgroupsIndirect',args:unknown[])=>{
     // --split-frames: each dispatch in a pass of its own, labelled by its pass and entry.
     if(pipeline&&splitFrames.includes(frame)){pass.end();pass=e.beginComputePass({label:`${descriptor?.label??'?'} | ${entryOf.get(pipeline)??'?'}`});bind();Reflect.apply(pass[method],pass,args);pass.end();pass=e.beginComputePass({label:'(split rest)'});bind();return;}
     const info=pipeline&&variantsOf.get(pipeline);
     if(!info||!frames.includes(frame)){Reflect.apply(pass[method],pass,args);return;}
     const seenKey=`${frame}:${info.entry}`,count=seen.get(seenKey)??0;seen.set(seenKey,count+1);
     if(count!==nth||captured.has(seenKey)){Reflect.apply(pass[method],pass,args);return;}
     captured.add(seenKey);pass.end();
     // Writable resources, whole buffers and textures, deduplicated.
     const buffers=new Map<GPUBuffer,string>(),images=new Map<GPUTexture,string>();
     for(const [group,args] of bindings){
      const groupEntries=entries.get(args[0] as GPUBindGroup);if(!groupEntries)continue;
      for(const entry of groupEntries){
       if(!info.writable.has(`${group}:${entry.binding}`))continue;
       const resource=entry.resource as GPUBufferBinding|GPUTextureView;
       if('buffer' in resource){if(!buffers.has(resource.buffer))buffers.set(resource.buffer,`${group}:${entry.binding}`);}
       else{const texture=textures.get(resource as GPUTextureView);assert.ok(texture,`texture of ${group}:${entry.binding}`);if(!images.has(texture))images.set(texture,`${group}:${entry.binding}`);}
      }
     }
     const texel=(t:GPUTexture)=>t.format==='rgba32float'?16:t.format==='r32float'||t.format==='r32uint'?4:(()=>{throw new Error(`format ${t.format}`);})();
     let total=0;const parts:Part[]=[];
     const bufferAt=[...buffers].map(([buffer,name])=>{const at=total;total+=Math.ceil(buffer.size/256)*256;parts.push({name:`${name} buffer`,bytes:buffer.size,words:buffer.size/4});return {buffer,at};});
     const imageAt=[...images].map(([texture,name])=>{const bytesPerRow=Math.ceil(texture.width*texel(texture)/256)*256,bytes=bytesPerRow*texture.height*texture.depthOrArrayLayers,at=total;total+=bytes;parts.push({name:`${name} ${texture.format} ${texture.width}x${texture.height}x${texture.depthOrArrayLayers}`,bytes,words:bytes/4});return {texture,at,bytesPerRow};});
     const make=(size:number,usage:number)=>target.createBuffer({size,usage});
     assert.ok(total<=target.limits.maxBufferSize,`snapshot ${total} exceeds buffer limit`);
     const initial=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),saved=make(total,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
     const results=info.pipelines.map(()=>make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ)),production=make(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
     const passes=repeats*info.pipelines.length;
     const queries=target.createQuerySet({type:'timestamp',count:2*passes}),resolved=make(16*passes,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(16*passes,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
     const snapshot=(buffer:GPUBuffer)=>{
      for(const {buffer:b,at} of bufferAt)e.copyBufferToBuffer(b,0,buffer,at,b.size);
      for(const {texture,at,bytesPerRow} of imageAt)e.copyTextureToBuffer({texture},{buffer,offset:at,bytesPerRow,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
     };
     const restore=(buffer:GPUBuffer)=>{
      for(const {buffer:b,at} of bufferAt)e.copyBufferToBuffer(buffer,at,b,0,b.size);
      for(const {texture,at,bytesPerRow} of imageAt)e.copyBufferToTexture({buffer,offset:at,bytesPerRow,rowsPerImage:texture.height},{texture},[texture.width,texture.height,texture.depthOrArrayLayers]);
     };
     snapshot(initial);
     pass=e.beginComputePass();bind();Reflect.apply(pass[method],pass,args);pass.end();snapshot(saved);e.copyBufferToBuffer(saved,0,production,0,total);
     info.pipelines.forEach((p,v)=>{restore(initial);pass=e.beginComputePass();bind(p);Reflect.apply(pass[method],pass,args);pass.end();snapshot(results[v]!);});
     let slot=0;
     for(let repeat=0;repeat<repeats;repeat++)for(let k=0;k<info.pipelines.length;k++){
      const v=(k+repeat)%info.pipelines.length;restore(initial);
      pass=e.beginComputePass({label:`QA ${info.entry} variant ${v}`,timestampWrites:{querySet:queries,beginningOfPassWriteIndex:2*slot,endOfPassWriteIndex:2*slot+1}});
      bind(info.pipelines[v]);for(let n=0;n<inner;n++)Reflect.apply(pass[method],pass,args);pass.end();slot++;
     }
     restore(saved);
     e.resolveQuerySet(queries,0,2*passes,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,16*passes);
     pending.push({entry:info.entry,frame,results:[production,...results],times,queries,resources:[initial,saved,resolved],parts,total});
     pass=e.beginComputePass();bind();
    };
    return new Proxy(pass,{get(_p,key){
     if(key==='setPipeline')return (value:GPUComputePipeline)=>{pipeline=value;pass.setPipeline(value);};
     if(key==='setBindGroup')return (index:number,...args:unknown[])=>{bindings.set(index,args);Reflect.apply(pass.setBindGroup,pass,[index,...args]);};
     if(key==='dispatchWorkgroups'||key==='dispatchWorkgroupsIndirect')return (...args:unknown[])=>dispatch(key,args);
     if(key==='end')return ()=>pass.end();
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
const ulp=(a:number,b:number)=>{const x=a|0,y=b|0;const fa=x<0?(0x80000000-x)|0:x,fb=y<0?(0x80000000-y)|0:y;return Math.abs(fa-fb);};
try{
 const definition=getSceneDefinition(option('scene')??'sparse-cm12-long-dam-break'),quality=definition.methodProfile?.quality??'balanced',scene=option('scene-file')?JSON.parse(readFileSync(option('scene-file')!,'utf8')):structuredClone(sceneDocument(definition));scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile?.overrides}),dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 const started=performance.now();
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 console.log(JSON.stringify({constructed_ms:Math.round(performance.now()-started),dt}));
 for(frame=1;frame<=frames.at(-1)!;frame++){
  if(profiler&&profileFrames.includes(frame))profiler.start();
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();
  if(profiler&&profileFrames.includes(frame)){const passes=await profiler.finish();const sum=new Map<string,number>();for(const p of passes)sum.set(p.label,(sum.get(p.label)??0)+p.ms);console.log(JSON.stringify({profile:frame,total:+passes.reduce((a,p)=>a+p.ms,0).toFixed(2),passes:[...sum].sort((a,b)=>b[1]-a[1]).map(([l,ms])=>[l,+ms.toFixed(3)])}));}assert.equal(solver.info.simulationPipelineError,undefined);
  if(!frames.includes(frame))continue;
  for(const entry of selected)if(!captured.has(`${frame}:${entry}`))console.log(JSON.stringify({warning:'not captured',entry,frame,dispatches:seen.get(`${frame}:${entry}`)??0}));
  for(const probe of pending.splice(0)){
   await Promise.all([...probe.results.map(buffer=>buffer.mapAsync(GPUMapMode.READ)),probe.times.mapAsync(GPUMapMode.READ)]);
   const words=probe.results.map(buffer=>new Int32Array(buffer.getMappedRange())),stamps=new BigUint64Array(probe.times.getMappedRange());
   const variants=variantNames.map((name,v)=>{
    let cursor=0;const parts=probe.parts.map(part=>{let count=0,worst=0,abs=0;const a=words[0]!,b=words[v+1]!,fa=new Float32Array(a.buffer,a.byteOffset,a.length),fb=new Float32Array(b.buffer,b.byteOffset,b.length);
     for(let i=cursor;i<cursor+part.words;i++)if(a[i]!==b[i]){count++;worst=Math.max(worst,ulp(a[i]!,b[i]!));const d=Math.abs(fa[i]!-fb[i]!);if(Number.isFinite(d))abs=Math.max(abs,d);}
     cursor+=Math.ceil(part.bytes/256)*256/4;return count?{part:part.name,mismatches:count,maxUlp:worst,maxAbs:abs}:undefined;}).filter(Boolean);
    const ms:number[]=[];
    for(let slot=0;slot<repeats*variantNames.length;slot++){const repeat=Math.floor(slot/variantNames.length),k=slot%variantNames.length;if((k+repeat)%variantNames.length===v)ms.push(Number(stamps[2*slot+1]!-stamps[2*slot]!)/1e6/inner);}
    const sorted=[...ms].sort((x,y)=>x-y);
    return {variant:name,min_ms:+sorted[0]!.toFixed(4),median_ms:+sorted[Math.floor(sorted.length/2)]!.toFixed(4),ms:ms.map(x=>+x.toFixed(4)),diff:parts};
   });
   console.log(JSON.stringify({entry:probe.entry,frame,inner,parts:probe.parts.map(p=>`${p.name} ${p.bytes}`),variants}));
   for(const result of probe.results){result.unmap();result.destroy();}probe.times.unmap();probe.times.destroy();probe.queries.destroy();for(const resource of probe.resources)resource.destroy();
  }
  assert.deepEqual(errors,[]);
 }
 console.log(JSON.stringify({info:solver.narrowBandFlipInfo,fineTiles:solver.workDomainInfo?.fineTiles}));
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();rawDevice.destroy();}
