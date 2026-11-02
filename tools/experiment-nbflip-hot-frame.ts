/** Replay one complete captured GPU frame from a real scene. No product changes.
 * node --import tsx tools/experiment-nbflip-hot-frame.ts --frame=120 --repeats=12
 * --root=SNAPSHOT --rewrites=transferDeposit:lean128
 * Inputs/restores and output comparisons are outside the wall-clock interval.
 * Wall includes encoding, submission, all stage kernels/copies and GPU completion.
 */
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync,readFileSync,readdirSync,rmSync} from 'node:fs';
import {cpus} from 'node:os';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
import type {UniformMixedFrameTrace} from '../lib/methods/uniform/uniform-mixed-frame';
import {GPUPassProfile} from './gpu-pass-profile';
import {hotVariants} from './nbflip-hot-variants';

const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const root=resolve(option('root')??'.');
const load=(file:string)=>import(pathToFileURL(resolve(root,file)).href);
const [{resolveMethodValues},{sceneDocument},{getSceneDefinition},{managedGPUDevice},{uniformNarrowBandMethod:method}]=await Promise.all([
 load('lib/core/method-contract.ts'), load('lib/core/scene-definition.ts'), load('lib/core/scenes.ts'), load('lib/core/gpu-compilation-manager.ts'), load('lib/methods/uniform/uniform-narrow-band-method.ts')
]);
const targetFrame=Number(option('frame')??120),repeats=Number(option('repeats')??9);
const scope='frame';
const fuseSurface=option('fuse-surface')==='true',activeDispatch=option('active-dispatch')==='true';
const rewriteMap=new Map((option('rewrites')??'').split(',').filter(Boolean).map(s=>{const [entry,variant]=s.split(':');assert.ok(entry&&variant&&hotVariants[variant]);return [entry,variant] as [string,string];}));
assert.ok(rewriteMap.size||fuseSurface||activeDispatch,'explicit experiment required');
const kernelProfile=option('kernels')==='true';
const out=resolve(option('out')??'artifacts/nbflip-hot-frame');mkdirSync(out,{recursive:true});
assert.ok(Number.isInteger(targetFrame)&&targetFrame>0&&Number.isInteger(repeats)&&repeats>=3);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal','disable-dawn-features=timestamp_quantization']).requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:(['timestamp-query','float32-filterable','subgroups'] as GPUFeatureName[]).filter(f=>adapter.features.has(f)),requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const errors:string[]=[];raw.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
const sources=new WeakMap<GPUShaderModule,string>();
const pipelines=new WeakMap<GPUComputePipeline,GPUComputePipelineDescriptor>();
const layouts=new WeakMap<GPUBindGroupLayout,GPUBindGroupLayoutEntry[]>();
const groups=new WeakMap<GPUBindGroup,{entries:GPUBindGroupEntry[];layout:GPUBindGroupLayout}>();
const views=new WeakMap<GPUTextureView,GPUTexture>();
type Binding={group:GPUBindGroup;offsets:number[]};
type Dispatch={kind:'dispatch';pipeline:GPUComputePipeline;bindings:Map<number,Binding>;method:'dispatchWorkgroups'|'dispatchWorkgroupsIndirect';args:unknown[];pass:number;label:string;phase?:string};
type Copy={kind:'copy';method:string;args:unknown[]};
type Op=Dispatch|Copy|{kind:'submit'}|{kind:'write';buffer:GPUBuffer;offset:number;bytes:Uint8Array<ArrayBuffer>};
type Part={name:string;buffer?:GPUBuffer;texture?:GPUTexture;bytes:number;row?:number;initial:GPUBuffer;expected:GPUBuffer;live:GPUBuffer;writable:boolean};
type Capture={name:string;ops:Op[];parts:Map<GPUBuffer|GPUTexture,Part>};
const captures:Capture[]=[];let active:Capture|undefined;let frame=0,passId=0;
const storageUsage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
const make=(size:number,usage=storageUsage)=>raw.createBuffer({size:Math.max(4,size),usage});
const pixelBytes=(format:GPUTextureFormat)=>{const n:Record<string,number>={r32float:4,r32uint:4,r32sint:4,rg32float:8,rgba32float:16,rgba32uint:16,rgba16float:8,rgba8unorm:4,r8unorm:1};assert.ok(n[format],`snapshot format ${format}`);return n[format]!;};
function copyPart(e:GPUCommandEncoder,p:Part,target:GPUBuffer,restore=false){
 if(p.buffer){if(restore)e.copyBufferToBuffer(target,0,p.buffer,0,p.bytes);else e.copyBufferToBuffer(p.buffer,0,target,0,p.bytes);}
 else{const t=p.texture!,b={buffer:target,bytesPerRow:p.row!,rowsPerImage:t.height},size=[t.width,t.height,t.depthOrArrayLayers];
  if(restore)e.copyBufferToTexture(b,{texture:t},size);else e.copyTextureToBuffer({texture:t},b,size);}
}
function ensure(e:GPUCommandEncoder,resource:GPUBuffer|GPUTexture,writable:boolean){
 if('usage' in resource&&!('format' in resource)&&(resource.usage&(GPUBufferUsage.MAP_READ|GPUBufferUsage.MAP_WRITE)))return;
 if(!active)return;const existing=active.parts.get(resource);if(existing){existing.writable||=writable;return;}
 const texture='format' in resource?resource as GPUTexture:undefined,buffer=texture?undefined:resource as GPUBuffer;
 const row=texture?Math.ceil(texture.width*pixelBytes(texture.format)/256)*256:undefined;
 const bytes=texture?row!*texture.height*texture.depthOrArrayLayers:buffer!.size;
 assert.ok(bytes<=raw.limits.maxStorageBufferBindingSize,`comparison binding ${resource.label} ${bytes}`);
 const p:Part={name:resource.label,buffer,texture,row,bytes,initial:make(bytes),expected:make(bytes),live:make(bytes),writable};
 active.parts.set(resource,p);copyPart(e,p,p.initial);
}
function bindingResources(bindings:Map<number,Binding>,pipeline:GPUComputePipeline){
 const resources=new Map<GPUBuffer|GPUTexture,boolean>();
 for(const [index,b] of bindings){
  const info=groups.get(b.group);assert.ok(info,'registered bind group');const entries=layouts.get(info.layout);
  for(const entry of info.entries){const layout=entries?.find(x=>x.binding===entry.binding);const r=entry.resource;
   const declaration=layout?'':sources.get(pipelines.get(pipeline)!.compute.module)!.match(new RegExp(`@group\\(\\s*${index}\\s*\\)\\s*@binding\\(\\s*${entry.binding}\\s*\\)[^;]+;`))?.[0];
   assert.ok(layout||declaration,`binding declaration ${index}:${entry.binding}`);
   const writable=layout?layout.buffer?.type==='storage'||!!layout.storageTexture:/read_write|texture_storage/.test(declaration!);
   if('buffer' in r){resources.set(r.buffer,(resources.get(r.buffer)??false)||writable);}
   else {const texture=views.get(r as GPUTextureView);if(texture)resources.set(texture,(resources.get(texture)??false)||writable);}
  }
 }
 return resources;
}
function endCapture(e:GPUCommandEncoder){
 assert.ok(active);for(const p of active.parts.values())copyPart(e,p,p.expected);
 active.ops.push({kind:'submit'});console.log(JSON.stringify({captured:active.name,commands:active.ops.length,resources:active.parts.size,bytes:[...active.parts.values()].reduce((s,p)=>s+p.bytes,0)}));active=undefined;
}
function startCapture(name:string){assert.equal(active,undefined);active={name,ops:[],parts:new Map()};captures.push(active);}
const originalWrite=raw.queue.writeBuffer.bind(raw.queue);
// queue writes remain ordered before the recorded encoder's submission.
const queue=new Proxy(raw.queue,{get(target,key){
 if(key==='writeBuffer')return(buffer:GPUBuffer,offset:number,data:GPUAllowSharedBufferSource,dataOffset=0,size?:number)=>{
  if(active){const unit=ArrayBuffer.isView(data)?((data as unknown as {BYTES_PER_ELEMENT?:number}).BYTES_PER_ELEMENT??1):1;
   const start=(ArrayBuffer.isView(data)?data.byteOffset:0)+dataOffset*unit;
   const length=size===undefined?(ArrayBuffer.isView(data)?data.byteLength:data.byteLength)-dataOffset*unit:size*unit;
   const bytes=new Uint8Array(ArrayBuffer.isView(data)?data.buffer:data,start,length).slice();active.ops.push({kind:'write',buffer,offset,bytes});}
  originalWrite(buffer,offset,data,dataOffset,size);
 };
 const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
}});
const intercepted=new Proxy(raw,{get(target,key){
 if(key==='queue')return queue;
 if(key==='createBuffer')return(d:GPUBufferDescriptor)=>target.createBuffer(d.usage&(GPUBufferUsage.MAP_READ|GPUBufferUsage.MAP_WRITE)?d:{...d,usage:d.usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 if(key==='createTexture')return(d:GPUTextureDescriptor)=>{const t=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});const create=t.createView.bind(t);t.createView=(d?:GPUTextureViewDescriptor)=>{const v=create(d);views.set(v,t);return v;};return t;};
 if(key==='createShaderModule')return(d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);sources.set(m,d.code);return m;};
 if(key==='createBindGroupLayout')return(d:GPUBindGroupLayoutDescriptor)=>{const l=target.createBindGroupLayout(d);layouts.set(l,[...d.entries]);return l;};
 if(key==='createBindGroup')return(d:GPUBindGroupDescriptor)=>{const g=target.createBindGroup(d);groups.set(g,{entries:[...d.entries],layout:d.layout});return g;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{const p=await target.createComputePipelineAsync(d);pipelines.set(p,d);return p;};
 if(key==='createComputePipeline')return(d:GPUComputePipelineDescriptor)=>{const p=target.createComputePipeline(d);pipelines.set(p,d);return p;};
 const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
}});
// Only the target frame is instrumented, and only the trace's stage commands.
function instrument(e:GPUCommandEncoder):GPUCommandEncoder{
 return new Proxy(e,{get(target,key){
  if(key==='beginComputePass')return(d?:GPUComputePassDescriptor)=>{
   let pass=target.beginComputePass(d),pipeline:GPUComputePipeline|undefined;const bindings=new Map<number,Binding>();const id=++passId,overrides=new Map<PropertyKey,unknown>();
   const rebind=()=>{if(pipeline)pass.setPipeline(pipeline);for(const [i,b]of bindings)pass.setBindGroup(i,b.group,b.offsets);};
   return new Proxy(pass,{set(_p,k,value){overrides.set(k,value);return true;},get(_p,k){
    if(overrides.has(k))return overrides.get(k);
    if(k==='setPipeline')return(p:GPUComputePipeline)=>{pipeline=p;pass.setPipeline(p);};
    if(k==='setBindGroup')return(i:number,g:GPUBindGroup|null,dynamic?:Iterable<number>,start?:number,length?:number)=>{const all=dynamic?[...dynamic]:[],offsets=start===undefined?all:all.slice(start,start+length!);if(g)bindings.set(i,{group:g,offsets});else bindings.delete(i);pass.setBindGroup(i,g,offsets);};
    if(k==='dispatchWorkgroups'||k==='dispatchWorkgroupsIndirect')return(...args:unknown[])=>{
     const entry=pipeline&&pipelines.get(pipeline)?.compute.entryPoint;
     if(active){assert.ok(pipeline);const resources=bindingResources(bindings,pipeline);if(k==='dispatchWorkgroupsIndirect')resources.set(args[0] as GPUBuffer,false);
      const missing=[...resources].some(([r])=>!active!.parts.has(r));
      if(missing){pass.end();for(const [r,w]of resources)ensure(target,r,w);pass=target.beginComputePass(d);rebind();}
      else for(const [r,w]of resources)if(w)active.parts.get(r)!.writable=true;
      active.ops.push({kind:'dispatch',pipeline,bindings:new Map(bindings),method:k,args,pass:id,label:d?.label??'unlabelled'});
     }
     Reflect.apply(pass[k],pass,args);
    };
    if(k==='end')return()=>pass.end();
    const value=Reflect.get(pass,k,pass);return typeof value==='function'?value.bind(pass):value;
   }});
  };
  if(['clearBuffer','copyBufferToBuffer','copyTextureToTexture','copyBufferToTexture','copyTextureToBuffer'].includes(String(key)))return(...args:unknown[])=>{
   if(active){
    if(key==='clearBuffer')ensure(target,args[0] as GPUBuffer,true);
    if(key==='copyBufferToBuffer'){ensure(target,args[0] as GPUBuffer,false);ensure(target,args[2] as GPUBuffer,true);}
    if(key==='copyTextureToTexture'){ensure(target,(args[0] as GPUTexelCopyTextureInfo).texture,false);ensure(target,(args[1] as GPUTexelCopyTextureInfo).texture,true);}
    if(key==='copyBufferToTexture'){ensure(target,(args[0] as GPUTexelCopyBufferInfo).buffer,false);ensure(target,(args[1] as GPUTexelCopyTextureInfo).texture,true);}
    if(key==='copyTextureToBuffer'){ensure(target,(args[0] as GPUTexelCopyTextureInfo).texture,false);ensure(target,(args[1] as GPUTexelCopyBufferInfo).buffer,true);}
    active.ops.push({kind:'copy',method:String(key),args});
   }
   Reflect.apply(Reflect.get(target,key,target),target,args);
  };
  const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
 }});
}
const trace:UniformMixedFrameTrace={
 instrument(e){if(!active&&captures.length===0)startCapture('frame');return instrument(e);},
 phase(_e,p){if(active)for(const op of active.ops)if(op.kind==='dispatch'&&!op.phase)op.phase=p.label;},
 submit(e){(solver as unknown as {mixedFrame:{blit:{finishStage(e:GPUCommandEncoder):void}}}).mixedFrame.blit.finishStage(e);if(active&&active.ops.length&&active.ops.at(-1)?.kind!=='submit')active.ops.push({kind:'submit'});},submitted(){},abort(){console.error('capture frame aborted');},
};
const compareModule=raw.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> a:array<u32>;
@group(0) @binding(1) var<storage,read> b:array<u32>;
@group(0) @binding(2) var<storage,read_write> counts:array<atomic<u32>>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 var n=0u;for(var i=gid.x;i<arrayLength(&a);i+=256u*groups.x){if(a[i]!=b[i]){n++;atomicMin(&counts[1],i);}}
 if(n>0u){atomicAdd(&counts[0],n);}
}`});
const comparator=await raw.createComputePipelineAsync({layout:'auto',compute:{module:compareModule,entryPoint:'main'}});
async function restore(c:Capture){const e=raw.createCommandEncoder();for(const p of c.parts.values())copyPart(e,p,p.initial,true);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();}
async function compare(c:Capture){
 const parts=[...c.parts.values()].filter(p=>p.writable),counts=parts.map(()=>make(8)),read=make(8*parts.length,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
 counts.forEach(b=>raw.queue.writeBuffer(b,0,new Uint32Array([0,0xffffffff])));
 const e=raw.createCommandEncoder();parts.forEach(p=>copyPart(e,p,p.live));const pass=e.beginComputePass();pass.setPipeline(comparator);
 parts.forEach((p,i)=>{pass.setBindGroup(0,raw.createBindGroup({layout:comparator.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:p.live}},{binding:1,resource:{buffer:p.expected}},{binding:2,resource:{buffer:counts[i]!}}]}));pass.dispatchWorkgroups(Math.min(4096,Math.ceil(p.bytes/1024)));});pass.end();
 counts.forEach((b,i)=>e.copyBufferToBuffer(b,0,read,8*i,8));raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
 const words=new Uint32Array(read.getMappedRange());const result=parts.map((p,i)=>({name:p.name,kind:p.texture?'texture':'buffer',bytes:p.bytes,mismatchedWords:words[2*i]!,firstWord:words[2*i+1]!}));read.unmap();read.destroy();counts.forEach(b=>b.destroy());return result;
}
let jobArgs:GPUBuffer|undefined,jobPipeline:GPUComputePipeline|undefined,jobGroup:GPUBindGroup|undefined;
function replay(c:Capture,replacements:Map<GPUComputePipeline,GPUComputePipeline>,device:GPUDevice=raw,candidate=false){
 let e=device.createCommandEncoder(),pass:GPUComputePassEncoder|undefined,last=-1,dirty=false;let bound=new Set<number>();
 let jobsBuilt=false;
 const close=()=>{pass?.end();pass=undefined;last=-1;bound=new Set();};
 for(const op of c.ops){
  if(op.kind==='write'){raw.queue.writeBuffer(op.buffer,op.offset,op.bytes);continue;}
  if(op.kind==='submit'){close();if(dirty){raw.queue.submit([e.finish()]);e=device.createCommandEncoder();dirty=false;}continue;}
  if(op.kind==='copy'){close();Reflect.apply(Reflect.get(e,op.method,e),e,op.args);dirty=true;continue;}
  const entry=pipelines.get(op.pipeline)?.compute.entryPoint;
  if(candidate&&fuseSurface&&entry==='arenaPublishCanonical')continue;
  const sized=candidate&&activeDispatch&&['separationPrepass','advectParticlesDeferred'].includes(entry??'');
  if(sized&&!jobsBuilt){close();const p=e.beginComputePass();p.setPipeline(jobPipeline!);p.setBindGroup(0,jobGroup!);p.dispatchWorkgroups(1);p.end();jobsBuilt=true;dirty=true;}
  if(!pass||last!==op.pass){close();pass=e.beginComputePass({label:op.phase??op.label});last=op.pass;}
  pass.setPipeline(replacements.get(op.pipeline)??op.pipeline);for(const i of bound)if(!op.bindings.has(i))pass.setBindGroup(i,null);for(const [i,b]of op.bindings)pass.setBindGroup(i,b.group,b.offsets);bound=new Set(op.bindings.keys());if(sized)pass.dispatchWorkgroupsIndirect(jobArgs!,entry==='separationPrepass'?0:12);else Reflect.apply(pass[op.method],pass,op.args);dirty=true;
 }
 close();if(dirty)raw.queue.submit([e.finish()]);
}
const choices=[{name:'baseline-A'},{name:'candidate'},{name:'baseline-B'}];
const results:unknown[]=[];const sourceHash=createHash('sha256');
for(const f of readdirSync(resolve(root,'lib/methods/uniform')).filter(f=>f.endsWith('.ts')).sort())sourceHash.update(f).update(readFileSync(resolve(root,`lib/methods/uniform/${f}`)));
const report={root,fuseSurface,activeDispatch,status:'running',date:new Date().toISOString(),revision:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),sourceHash:sourceHash.digest('hex'),cpu:cpus()[0]?.model,scene:option('scene')??'sparse-cm12-long-dam-break',frame:targetFrame,time_s:targetFrame/60,dt_s:1/60,repeats,features:[...raw.features],adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},scope,rewrites:Object.fromEntries(rewriteMap),contract:'Whole captured GPU frame commands, copies and writes; wall encoding through queue completion; restore/verification excluded; receipt destinations excluded; CPU planning frozen; every other captured writable word compared',values:{} as unknown,telemetry:{} as unknown,results,errors};
const save=()=>writeFileSync(`${out}/results.json`,JSON.stringify(report,null,2)+'\n');
let solver:WebGPUUniformReferenceSolver|undefined;
try{
 const definition=getSceneDefinition(report.scene),scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,'balanced',{...method.appDefaults,...definition.methodProfile?.overrides,timeStep:'sixtieth'});
 report.values=values;
 solver=await method.createSolverAsync!(managedGPUDevice(intercepted,{requireWorkerRealm:false}),scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const mf=(solver as unknown as {mixedFrame:{advance:(...args:unknown[])=>unknown}}).mixedFrame;
 const advance=mf.advance.bind(mf);mf.advance=(...args)=>{if(frame===targetFrame)args[1]=trace;return advance(...args);};
 console.log(JSON.stringify({phase:'constructed'}));
 for(frame=1;frame<=targetFrame;frame++){assert.ok(solver.advanceTo(frame/60,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(frame%30===0)console.log(JSON.stringify({frame,particles:solver.narrowBandFlipInfo?.particles}));}
 const finalEncoder=raw.createCommandEncoder();endCapture(finalEncoder);raw.queue.submit([finalEncoder.finish()]);await raw.queue.onSubmittedWorkDone();
 report.telemetry={...solver.narrowBandFlipInfo,...solver.workDomainInfo};
 assert.equal(captures.length,1);assert.deepEqual(errors,[]);save();
 for(const c of captures){
  if(activeDispatch){
   const op=c.ops.find(op=>op.kind==='dispatch'&&pipelines.get(op.pipeline)?.compute.entryPoint==='advectParticlesDeferred');assert.ok(op?.kind==='dispatch'&&op.method==='dispatchWorkgroups');
   const source=sources.get(pipelines.get(op.pipeline)!.compute.module)!;
   const number=(name:string)=>{const n=Number(source.match(new RegExp(`const ${name}:u32=(\\d+)u`))?.[1]);assert.ok(n>0,name);return n;};
   const capacity=number('NB_PARTICLE_CAPACITY'),limit=number('NB_SLAB_LIMIT'),chunk=number('NB_MOVE_CHUNK'),ceiling=op.args[0] as number;
   const state=groups.get(op.bindings.get(1)!.group)!.entries.find(e=>e.binding===4)!.resource as GPUBufferBinding;
   jobArgs=raw.createBuffer({size:24,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
   const code=`@group(0) @binding(0) var<storage,read> s:array<u32>;@group(0) @binding(1) var<storage,read_write> a:array<u32>;
   @compute @workgroup_size(1) fn main(){let flat=min(s[21],${capacity}u-min(s[0],${limit}u));let tail=(flat+63u)/64u;
   a[0]=min(${chunk/64}u*s[22]+tail,${ceiling}u);a[1]=1u;a[2]=1u;a[3]=min(s[22]+tail,${ceiling}u);a[4]=1u;a[5]=1u;}`;
   jobPipeline=await raw.createComputePipelineAsync({layout:'auto',compute:{module:raw.createShaderModule({code}),entryPoint:'main'}});
   jobGroup=raw.createBindGroup({layout:jobPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:state},{binding:1,resource:{buffer:jobArgs}}]});
  }
  if(fuseSurface){
   const names=c.ops.filter(op=>op.kind==='dispatch').map(op=>pipelines.get(op.pipeline)?.compute.entryPoint);
   assert.equal(names.filter(n=>n==='arenaDistance').length,1,'fusion experiment requires one canonical metric');
   assert.equal(names.filter(n=>n==='arenaPublishCanonical').length,1,'fusion experiment requires one publisher');
   assert.equal(names.indexOf('arenaDistance')+1,names.indexOf('arenaPublishCanonical'));
  }
  const compiled:Map<GPUComputePipeline,GPUComputePipeline>[]=[];
  for(const v of choices){
   const replacements=new Map<GPUComputePipeline,GPUComputePipeline>();
   for(const op of c.ops){if(op.kind!=='dispatch'||replacements.has(op.pipeline))continue;const d=pipelines.get(op.pipeline);assert.ok(d,op.label);
    const entry=d.compute.entryPoint??'',rewrite=v.name==='candidate'?rewriteMap.get(entry):undefined;
    const fused=v.name==='candidate'&&fuseSurface&&entry==='arenaDistance';
    if(!rewrite&&!fused)continue;const code=fused?sources.get(d.compute.module)!:hotVariants[rewrite!]!(sources.get(d.compute.module)!,entry);const m=raw.createShaderModule({code});
    writeFileSync(`${out}/${c.name}-${v.name}-${entry}.wgsl`,code);
    const p=await raw.createComputePipelineAsync({...d,label:`replay ${v.name}`,compute:{...d.compute,module:m,...(fused?{entryPoint:'arenaDistanceCanonical'}:{})}});replacements.set(op.pipeline,p);
   }
   compiled.push(replacements);
   if(v.name==='candidate')for(const entry of rewriteMap.keys())assert.ok([...replacements.keys()].some(p=>pipelines.get(p)?.compute.entryPoint===entry),`captured replacement ${entry}`);
  }
  const samples=choices.map(()=>[] as number[]),diffs:unknown[][]=choices.map(()=>[]);
  // One untimed warmup for every variant, followed by rotated/reversed trials.
  for(let round=-1;round<repeats;round++){
   const order=choices.map((_,i)=>(i+Math.max(0,round))%choices.length);if(round%2===1)order.reverse();
   for(const i of order){await restore(c);const start=performance.now();replay(c,compiled[i]!,raw,choices[i]!.name==='candidate');await raw.queue.onSubmittedWorkDone();const ms=performance.now()-start;
    if(round>=0)samples[i]!.push(ms);
    if(round<=0||round===repeats-1){const diff=await compare(c);diffs[i]!.push({round,parts:diff.filter(p=>p.mismatchedWords),checkedWords:diff.reduce((s,p)=>s+p.bytes/4,0),textureMismatchedWords:diff.filter(p=>p.kind==='texture').reduce((s,p)=>s+p.mismatchedWords,0),mismatchedWords:diff.reduce((s,p)=>s+p.mismatchedWords,0)});}
   }
   console.log(JSON.stringify({stage:c.name,round}));
  }
  const profiler=new GPUPassProfile(raw);await restore(c);profiler.start();replay(c,new Map(),profiler.device);await raw.queue.onSubmittedWorkDone();const passes=await profiler.finish();profiler.destroy();
  let kernels:unknown[]|undefined;
  if(kernelProfile){
   const recorder=new GPUPassProfile(raw,4096,true);await restore(c);recorder.start();replay(c,new Map(),recorder.device);await raw.queue.onSubmittedWorkDone();const timed=await recorder.finish();recorder.destroy();
   const dispatches=c.ops.filter(op=>op.kind==='dispatch');assert.equal(timed.length,dispatches.length);
   kernels=timed.map((p,i)=>({...p,entry:pipelines.get(dispatches[i]!.pipeline)?.compute.entryPoint,phase:dispatches[i]!.phase}));
  }
  const mean=(xs:number[])=>xs.reduce((a,b)=>a+b,0)/xs.length;
  const statistics=(xs:number[])=>{const s=[...xs].sort((a,b)=>a-b);return {mean_ms:mean(xs),median_ms:s[Math.floor(s.length/2)],min_ms:s[0],max_ms:s.at(-1),samples_ms:xs};};
  const result={stage:c.name,resources:[...c.parts.values()].map(p=>({name:p.name,kind:p.texture?'texture':'buffer',bytes:p.bytes,writable:p.writable})),dispatches:c.ops.filter(op=>op.kind==='dispatch').map(op=>({label:op.label,entry:pipelines.get(op.pipeline)?.compute.entryPoint,phase:op.phase,args:op.method==='dispatchWorkgroups'?op.args:['indirect'],constants:pipelines.get(op.pipeline)?.compute.constants})),variants:choices.map((v,i)=>({name:v.name,...statistics(samples[i]!),checks:diffs[i]})),passes,kernels};
  results.push(result);save();console.log(JSON.stringify({stage:c.name,variants:result.variants.map(v=>({...v,checks:undefined,samples_ms:undefined}))}));
  for(const p of c.parts.values()){p.initial.destroy();p.expected.destroy();p.live.destroy();}
 }
 assert.deepEqual(errors,[]);report.status='complete';rmSync(`${out}/failure.txt`,{force:true});save();
}catch(error){report.status='failed';writeFileSync(`${out}/failure.txt`,String(error)+'\n'+(error instanceof Error?error.stack:''));save();throw error;}
finally{jobArgs?.destroy();solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
