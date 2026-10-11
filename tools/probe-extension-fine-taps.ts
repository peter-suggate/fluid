/** Same-input current extension finalization + update replay. The candidate
 * writes h fine taps at publication, leaving the unchanged coarse tap fill. */
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {mkdirSync,writeFileSync} from "node:fs";
import {dirname,resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {resolveMethodValues} from "../lib/core/method-contract";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {uniformNarrowBandMethod as method} from "../lib/methods/uniform/uniform-narrow-band-method";
import {uniformDetailField,uniformDetailPick} from "../lib/methods/uniform/uniform-detail-fields";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {GPUPassProfile} from "./gpu-pass-profile";
import {Capture} from "./probe-extension-direct-snapshot";
import {coarseOnlyTapsSource,extensionFineTapsSource} from "./extension-fine-taps";

const stats=(values:number[])=>{const sorted=[...values].sort((a,b)=>a-b),mean=values.reduce((s,v)=>s+v,0)/values.length;return{mean,median:(sorted[Math.floor((sorted.length-1)/2)]!+sorted[Math.ceil((sorted.length-1)/2)]!)/2,min:sorted[0],max:sorted.at(-1),stddev:Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/values.length)};};
async function run():Promise<void>{
 const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
 const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),quality=definition.methodProfile!.quality,scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s,steps=Number(option("steps")??96),rounds=Number(option("rounds")??12),sweeps=Number(values.extensionFrontSweeps??2);
 const warmup=Number(option("warmup")??8);
 const out=option("out")??"artifacts/nbflip-ui-perf/extension-fine-taps-96.json";
 let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
 const buffers:GPUBuffer[]=[],errors:string[]=[];
 try{
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  profile=new GPUPassProfile(raw);
  const captureDevice=new Proxy(profile.device,{get(target,key){if(key==="createTexture")return(d:GPUTextureDescriptor)=>target.createTexture({...d,usage:d.usage|(/Narrow-band FLIP stage (?:sampling cache|unit taps)/.test(d.label??"")?GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST:0)});const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}});
  const capture=new Capture(captureDevice);device=managedGPUDevice(capture.device,{requireWorkerRealm:false});
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  const frame=(solver as any).mixedFrame,ext=frame.extension,flip=frame.narrowBandFlip,taps=flip.stageTaps;
  const descriptor=capture.pipelines.get(ext.regularPipelines.get("publish"))!,tapDescriptor=capture.pipelines.get(taps.builder.pipeline)!;
  const originalSource=capture.sources.get(descriptor.compute.module)!,tapSource=capture.sources.get(tapDescriptor.compute.module)!;
  writeFileSync("/tmp/fluid-fine-taps-original.wgsl",originalSource);writeFileSync("/tmp/fluid-fine-taps-taps.wgsl",tapSource);
  const candidates=[extensionFineTapsSource(originalSource,tapSource),coarseOnlyTapsSource(tapSource)];
  const modules=await Promise.all(candidates.map(async(code,index)=>{const module=raw.createShaderModule({label:`Fine taps fusion${index}`,code});const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(messages.length){writeFileSync(`/tmp/fluid-fine-taps-${index}.wgsl`,code);throw new Error(messages.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));}return module;}));
  const variants=await Promise.all([ext.regularPipelines.get("publish"),ext.seamPipelines.get("publish"),taps.builder.pipeline].map(async(p,index)=>{const d=capture.pipelines.get(p)!;return raw.createComputePipelineAsync({...d,compute:{...d.compute,module:modules[index<2?0:1]!}});}));
  console.log(JSON.stringify({phase:"source-preflight",storageBindingsAdded:0}));
  for(let n=1;n<=steps;n++){solver.advanceTo(n*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(n%24===0||n===steps)console.log(JSON.stringify({phase:"advance",frame:n}));}
  const o=frame.ownership,f=frame.fields;
  const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC)});buffers.push(b);return b;};
  const physical=(t:GPUTexture)=>{const field=uniformDetailField(t);return field?field.storage.physical(field.handle):t;};
  const homes=(t:GPUTexture)=>{const field=uniformDetailField(t);return [...new Set(field?[field.storage.physical(field.handle),field.storage.baseOf(field.handle)!]:[t])].filter(Boolean);};
  const outputTextures=[...homes(f.velocityScratch),physical(taps.unit),flip.coarseVelocity] as GPUTexture[];
  const textureRecords=outputTextures.map(texture=>{const bytesPerRow=Math.ceil(texture.width*16/256)*256,extent=[texture.width,texture.height,texture.depthOrArrayLayers] as [number,number,number],size=bytesPerRow*texture.height*texture.depthOrArrayLayers;return{texture,bytesPerRow,rowsPerImage:texture.height,extent,original:allocate(size),reference:allocate(size)};});
  const particles=flip.activeParticles as GPUBuffer;flip.parity^=1;
  const stateRead=allocate(flip.state.size,true);let e=raw.createCommandEncoder();e.copyBufferToBuffer(flip.state,0,stateRead,0,stateRead.size);raw.queue.submit([e.finish()]);await stateRead.mapAsync(GPUMapMode.READ);const state=new Uint32Array(stateRead.getMappedRange().slice(0));stateRead.unmap();const count=state[0]!;assert.ok(count>0);state[1]=count;raw.queue.writeBuffer(flip.state,0,state);
  const bufferRecords=[{buffer:particles,size:48*count},{buffer:f.negativeScratch as GPUBuffer,size:f.negativeScratch.size},{buffer:o.hanging as GPUBuffer,size:o.hanging.size}].map(r=>({...r,original:allocate(r.size),reference:allocate(r.size)}));
  const originals=(encoder:GPUCommandEncoder,which:"original"|"reference")=>{textureRecords.forEach(r=>encoder.copyTextureToBuffer({texture:r.texture},{buffer:r[which],bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},r.extent));bufferRecords.forEach(r=>encoder.copyBufferToBuffer(r.buffer,0,r[which],0,r.size));};
  const restore=(encoder:GPUCommandEncoder)=>{textureRecords.forEach(r=>encoder.copyBufferToTexture({buffer:r.original,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},{texture:r.texture},r.extent));bufferRecords.forEach(r=>encoder.copyBufferToBuffer(r.original,0,r.buffer,0,r.size));};
  e=raw.createCommandEncoder();originals(e,"original");raw.queue.submit([e.finish()]);
  const groups=frame.extensionGroups.map((group:any)=>{const d=capture.groups.get(group.group)!;return raw.createBindGroup({...d,entries:[...d.entries].map(entry=>entry.binding===11?{binding:11,resource:physical(taps.unit).createView()}:entry)});});
  const marker=await raw.createComputePipelineAsync({layout:"auto",compute:{module:raw.createShaderModule({code:"@compute @workgroup_size(1) fn marker(){}"}),entryPoint:"marker"}});
  const comparison=raw.createShaderModule({code:`
@group(0) @binding(0) var input:texture_3d<f32>;
@group(0) @binding(1) var<storage,read> expected:array<u32>;
@group(0) @binding(2) var<storage,read_write> result:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> current:array<u32>;
override rowWords:u32=0u;
fn note(i:u32,a:u32,b:u32){if(a!=b){atomicAdd(&result[0],1u);atomicMin(&result[1],i);}}
@compute @workgroup_size(256) fn textureCompare(@builtin(global_invocation_id) gid:vec3u){let at=gid.x+65535u*256u*gid.y;let d=textureDimensions(input);if(at>=d.x*d.y*d.z){return;}let q=vec3u(at%d.x,(at/d.x)%d.y,at/(d.x*d.y));let a=bitcast<vec4u>(textureLoad(input,vec3i(q),0));let offset=4u*q.x+rowWords*(q.y+d.y*q.z);for(var c=0u;c<4u;c++){note(4u*at+c,a[c],expected[offset+c]);}}
@compute @workgroup_size(256) fn bufferCompare(@builtin(global_invocation_id) gid:vec3u){let i=gid.x+65535u*256u*gid.y;if(i<arrayLength(&expected)){note(i,current[i],expected[i]);}}
`});
  const results=allocate(16),read=allocate(16,true);
  const comparisons=await Promise.all([...textureRecords.map(async r=>{const pipeline=await raw.createComputePipelineAsync({layout:"auto",compute:{module:comparison,entryPoint:"textureCompare",constants:{rowWords:r.bytesPerRow/4}}});return{words:4*r.extent.reduce((a,b)=>a*b,1),invocations:r.extent.reduce((a,b)=>a*b,1),pipeline,group:raw.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:r.texture.createView()},{binding:1,resource:{buffer:r.reference}},{binding:2,resource:{buffer:results}}]})};}),...bufferRecords.map(async r=>{const pipeline=await raw.createComputePipelineAsync({layout:"auto",compute:{module:comparison,entryPoint:"bufferCompare"}});return{words:r.size/4,invocations:r.size/4,pipeline,group:raw.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:r.reference}},{binding:2,resource:{buffer:results}},{binding:3,resource:{buffer:r.buffer,size:r.size}}]})};})]);
  const finalize=(encoder:GPUCommandEncoder,fused:boolean,activeSweeps:number)=>{
   if(!fused){ext.finishPair(encoder,frame.extensionGroups,activeSweeps,false);return;}
   const final=frame.extensionGroups[activeSweeps%2],pass=encoder.beginComputePass({label:"Uniform mixed extension"});pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,final.group);pass.setPipeline(uniformDetailPick(ext.restrictPipeline));pass.dispatchWorkgroups(...o.capacity.tileDimensions.map((n:number)=>Math.ceil(n/4)) as [number,number,number]);ext.hierarchy.encode(pass,true);
   pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,groups[activeSweeps%2]);o.dispatchTierCounted(pass,variants[0],0);
   pass.setBindGroup(1,final.base);o.dispatchRegularCoarseCounted(pass,ext.listPipelines.get("publish"));
   pass.setBindGroup(1,groups[activeSweeps%2]);o.dispatchCounted(pass,variants[1],o.capacity.tiles,"hanging");pass.end();
  };
  const encode=(encoder:GPUCommandEncoder,fused:boolean,activeSweeps:number)=>{
   finalize(encoder,fused,activeSweeps);const pass=encoder.beginComputePass({label:"Narrow-band FLIP update sample"});flip.cache.encode(encoder,flip.cacheGroups.update,pass);
   const old=taps.builder.pipeline;if(fused)taps.builder.pipeline=variants[2];flip.encodeStageTaps(encoder,pass);taps.builder.pipeline=old;
   flip.encodeParticleWork(encoder,pass);flip.dispatchBatch(encoder,["update"],undefined,"update",pass);pass.end();
  };
  const parity=[];
  for(const activeSweeps of [...new Set([sweeps,0,1,3])]){
   // Prepare the scalar seed/sweep state once. Finalization reads these slots
   // only; replayHierarchy uses the same immutable source/distance topology.
   e=raw.createCommandEncoder();ext.encode(e,frame.extensionGroups,activeSweeps,false);raw.queue.submit([e.finish()]);
   raw.queue.writeBuffer(results,0,new Uint32Array([0,0xffffffff,0,0]));
   for(const fused of [false,true]){e=raw.createCommandEncoder();restore(e);encode(e,fused,activeSweeps);if(!fused)originals(e,"reference");else{const pass=e.beginComputePass({label:"Compare fused publication outputs"});for(const c of comparisons){pass.setPipeline(c.pipeline);pass.setBindGroup(0,c.group);const n=Math.ceil(c.invocations/256);pass.dispatchWorkgroups(Math.min(65535,n),Math.ceil(n/65535));}pass.end();}raw.queue.submit([e.finish()]);}
   e=raw.createCommandEncoder();e.copyBufferToBuffer(results,0,read,0,16);raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const resultWords=[...new Uint32Array(read.getMappedRange())];read.unmap();const result={sweeps:activeSweeps,words:comparisons.reduce((s,c)=>s+c.words,0),mismatches:resultWords[0],firstMismatch:resultWords[1]};parity.push(result);console.log(JSON.stringify({phase:"parity",...result}));assert.equal(result.mismatches,0);
  }
  e=raw.createCommandEncoder();ext.encode(e,frame.extensionGroups,sweeps,false);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  if(process.argv.includes("--wait-for-gpu-quiet")){let seconds=0;for(;;){const others=execFileSync("ps",["-Ao","pid=,args="],{encoding:"utf8"}).split("\n").filter(line=>{const row=line.match(/^\s*(\d+)\s+(.*)$/);return row&&Number(row[1])!==process.pid&&/^(?:\S*\/)?node\s/.test(row[2]!)&&/tools\/(?:probe|benchmark)[^\s]*\.ts/.test(row[2]!);});if(!others.length)break;if(seconds++%60===0)console.log(JSON.stringify({phase:"waiting-before-timing",others}));await new Promise(resolve=>setTimeout(resolve,1000));}}
  const trials:{fused:boolean;ms:number;passes:Awaited<ReturnType<GPUPassProfile["finish"]>>}[]=[];
  for(let round=0;round<rounds+warmup;round++)for(const fused of round%2?[true,false]:[false,true]){
   e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();profile.start();e=device.createCommandEncoder();const start=e.beginComputePass({label:"Fine tap fusion interval start"});start.setPipeline(marker);start.dispatchWorkgroups(1);start.end();encode(e,fused,sweeps);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();const passes=await profile.finish();if(round>=warmup)trials.push({fused,ms:passes.reduce((s,p)=>s+p.ms+p.idle_ms,0),passes});
  }
  const saving=stats(Array.from({length:rounds},(_,i)=>{const pair=trials.slice(2*i,2*i+2);return pair.find(t=>!t.fused)!.ms-pair.find(t=>t.fused)!.ms;}));
  const report={date:new Date().toISOString(),sceneId:definition.id,values,dt,steps,rounds,sweeps,count,parity,distributions_ms:{original:stats(trials.filter(t=>!t.fused).map(t=>t.ms)),fused:stats(trials.filter(t=>t.fused).map(t=>t.ms)),saving},trials,errors};mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify(report));assert.deepEqual(errors,[]);
  if(process.argv.includes("--snapshot-after")){
   const companion=allocate(ext.companionBytesAt(o.capacity.fineTiles));
   const common={physical:f.velocity,phase:f.phase,negative:f.negative,output:f.velocityScratch,outputNegative:f.negativeScratch,scratch:{buffer:frame.stageScratch},params:frame.params.extension,companion:{scratch:companion,physical:f.velocity,negative:f.negative}};
   const pair=ext.bind(common),snapshot=ext.bind({...common,snapshot:taps.unit});
   const prepareSnapshot=(encoder:GPUCommandEncoder)=>{ext.retainPair(encoder,pair);ext.preparePair(encoder,pair,sweeps);};
   const encodeSnapshot=(encoder:GPUCommandEncoder,split:boolean)=>{
    if(!split){ext.finishSnapshot(encoder,snapshot,sweeps);return;}
    const pipelines=ext.snapshotPipelines;
    let pass=encoder.beginComputePass({label:"Slot snapshot restrict"});pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,snapshot[sweeps%2].group);pass.setPipeline(pipelines.restrict);pass.dispatchWorkgroups(...o.capacity.tileDimensions.map((n:number)=>Math.ceil(n/4)) as [number,number,number]);pass.end();
    pass=encoder.beginComputePass({label:"Slot snapshot hierarchy"});ext.hierarchy.encode(pass,false);pass.end();
    pass=encoder.beginComputePass({label:"Slot snapshot fine"});pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,snapshot[sweeps%2].group);if(o.capacity.fineTiles>0){pass.setPipeline(pipelines.fine);pass.dispatchWorkgroups(Math.min(8192,o.capacity.fineTiles));}pass.end();
    pass=encoder.beginComputePass({label:"Slot snapshot coarse"});pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,snapshot[sweeps%2].group);const n=Math.ceil(3*o.capacity.tiles/64);pass.setPipeline(pipelines.coarse);pass.dispatchWorkgroups(Math.min(65535,n),Math.ceil(n/65535));pass.end();
   };
   const expected=allocate(companion.size),slotCompare=await raw.createComputePipelineAsync({layout:"auto",compute:{module:comparison,entryPoint:"bufferCompare"}}),slotGroup=raw.createBindGroup({layout:slotCompare.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:expected}},{binding:2,resource:{buffer:results}},{binding:3,resource:{buffer:companion}}]});
   raw.queue.writeBuffer(results,0,new Uint32Array([0,0xffffffff,0,0]));
   e=raw.createCommandEncoder();prepareSnapshot(e);encodeSnapshot(e,false);e.copyBufferToBuffer(companion,0,expected,0,expected.size);originals(e,"reference");raw.queue.submit([e.finish()]);
   e=raw.createCommandEncoder();prepareSnapshot(e);encodeSnapshot(e,true);const comparePass=e.beginComputePass({label:"Compare split snapshot"});
   for(const c of comparisons){comparePass.setPipeline(c.pipeline);comparePass.setBindGroup(0,c.group);const n=Math.ceil(c.invocations/256);comparePass.dispatchWorkgroups(Math.min(65535,n),Math.ceil(n/65535));}
   comparePass.setPipeline(slotCompare);comparePass.setBindGroup(0,slotGroup);const n=Math.ceil(companion.size/4/256);comparePass.dispatchWorkgroups(Math.min(65535,n),Math.ceil(n/65535));comparePass.end();e.copyBufferToBuffer(results,0,read,0,16);raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const compared=[...new Uint32Array(read.getMappedRange())];read.unmap();assert.equal(compared[0],0);console.log(JSON.stringify({phase:"split-snapshot-parity",words:comparisons.reduce((s,c)=>s+c.words,0)+companion.size/4,mismatches:compared[0]}));
   const timings:{split:boolean;ms:number;passes:Awaited<ReturnType<GPUPassProfile["finish"]>>}[]=[];
   for(let round=0;round<rounds+warmup;round++)for(const split of round%2?[true,false]:[false,true]){
    e=raw.createCommandEncoder();prepareSnapshot(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();profile.start();e=device.createCommandEncoder();encodeSnapshot(e,split);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();const passes=await profile.finish();if(round>=warmup)timings.push({split,ms:passes.reduce((sum,p)=>sum+p.ms+p.idle_ms,0),passes});
   }
   const stages=Object.fromEntries(["restrict","hierarchy","fine","coarse"].map(name=>[name,stats(timings.filter(t=>t.split).map(t=>t.passes.find(p=>p.label===`Slot snapshot ${name}`)!.ms))]));
   const result={sceneId:definition.id,steps,sweeps,payload:"Frozen current values used for both paired payloads; same topology, masks and distances as current extension",parity:{words:comparisons.reduce((s,c)=>s+c.words,0)+companion.size/4,mismatches:compared[0]},stages_ms:stages,unsplit_ms:stats(timings.filter(t=>!t.split).map(t=>t.ms)),split_ms:stats(timings.filter(t=>t.split).map(t=>t.ms)),timings,errors};writeFileSync(out.replace(/\.json$/,"-snapshot.json"),JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));assert.deepEqual(errors,[]);
  }
 }finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await run();
