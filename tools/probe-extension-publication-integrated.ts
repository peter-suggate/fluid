/** Frozen whole old/current publication replay of retained geometric tap masks. */
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
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

const stats=(values:number[])=>{const sorted=[...values].sort((a,b)=>a-b),mean=values.reduce((s,v)=>s+v,0)/values.length;return{mean,median:(sorted[Math.floor((sorted.length-1)/2)]!+sorted[Math.ceil((sorted.length-1)/2)]!)/2,min:sorted[0],max:sorted.at(-1),stddev:Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/values.length)};};
async function run():Promise<void>{
 const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
 const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),quality=definition.methodProfile!.quality,scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s,steps=Number(option("steps")??96),rounds=Number(option("rounds")??12),sweeps=Number(values.extensionFrontSweeps??2);
 const warmup=Number(option("warmup")??8);
 const out=option("out")??"artifacts/nbflip-ui-perf/extension-publication-integrated-96.json";
 let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
 const buffers:GPUBuffer[]=[],errors:string[]=[];
 try{
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  profile=new GPUPassProfile(raw);
  const captureDevice=new Proxy(profile.device,{get(target,key){if(key==="createTexture")return(d:GPUTextureDescriptor)=>target.createTexture({...d,usage:d.usage|(/Narrow-band FLIP stage (?:sampling cache|unit taps)|Uniform (?:Sec. 3.3 hierarchy|nearest-source origins)/.test(d.label??"")?GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST:0)});const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}});
  const capture=new Capture(captureDevice);device=managedGPUDevice(capture.device,{requireWorkerRealm:false});
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  const frame=(solver as any).mixedFrame,ext=frame.extension,flip=frame.narrowBandFlip,taps=flip.stageTaps,extrap=(solver as any).velocityExtrapolator,mixed=extrap.mixedPipelines;
  const originalPipelines={snapshotFine:ext.snapshotPipelines.fine,snapshotCoarse:ext.snapshotPipelines.coarse,regular:ext.regularPipelines.get("publish"),seam:ext.seamPipelines.get("publish"),list:ext.listPipelines.get("publish")};
  const productionSnapshot={fine:originalPipelines.snapshotFine,coarse:originalPipelines.snapshotCoarse},productionReplay=ext.replayPublishers;
  assert.ok(productionReplay,"Production selection replay is unavailable");
  const variants=[productionSnapshot.fine,productionSnapshot.coarse,productionReplay.regular.get("publish"),productionReplay.seams.get("publish"),productionReplay.list.get("publish")];
  const baselineSnapshot=readFileSync(option("baseline-snapshot")??"/tmp/fluid-publication-baseline-old.wgsl","utf8");
  const baselineModule=raw.createShaderModule({label:"Original unretained snapshot",code:baselineSnapshot});
  const messages=(await baselineModule.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(messages,[]);
  const baseline=await Promise.all([productionSnapshot.fine,productionSnapshot.coarse].map(pipeline=>{const d=capture.pipelines.get(pipeline)!;return raw.createComputePipelineAsync({...d,compute:{...d.compute,module:baselineModule}});}));
  originalPipelines.snapshotFine=baseline[0];originalPipelines.snapshotCoarse=baseline[1];
  const productionSource=capture.sources.get(capture.pipelines.get(variants[2])!.compute.module)!;
  const fallbackCondition="if(retained.y!=-1.0)";assert.equal(productionSource.split(fallbackCondition).length-1,1);
  const fallbackCode=productionSource.replace(fallbackCondition,"if(publicationForceFallback||retained.y!=-1.0)")+"\noverride publicationForceFallback:bool=false;";
  const fallbackModule=raw.createShaderModule({label:"Production forced missing-selection fallback",code:fallbackCode});
  const fallbackMessages=(await fallbackModule.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(fallbackMessages,[]);
  const fallbackVariants=await Promise.all(variants.slice(2).map(pipeline=>{const d=capture.pipelines.get(pipeline)!;return raw.createComputePipelineAsync({...d,compute:{...d.compute,module:fallbackModule,constants:{...d.compute.constants,publicationForceFallback:1}}});}));
  console.log(JSON.stringify({phase:"source-preflight",storageBindingsAdded:0,scratch:"Inactive extension epoch; one tagged selection per canonical positive face"}));
  const rawOld=raw.createBuffer({size:ext.companionBytesAt(frame.ownership.capacity.fineTiles),usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(rawOld);
  const originalRetain=ext.retainPair.bind(ext);
  for(let n=1;n<=steps;n++){
   if(n===steps)ext.retainPair=(encoder:GPUCommandEncoder,groups:any)=>{originalRetain(encoder,groups);encoder.copyBufferToBuffer(flip.extensionScratch,0,rawOld,0,rawOld.size);};
   solver.advanceTo(n*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(n%24===0||n===steps)console.log(JSON.stringify({phase:"advance",frame:n}));
  }
  ext.retainPair=originalRetain;
  const o=frame.ownership,f=frame.fields;
  const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC)});buffers.push(b);return b;};
  const physical=(t:GPUTexture)=>{const field=uniformDetailField(t);return field?field.storage.physical(field.handle):t;};
  const homes=(t:GPUTexture)=>{const field=uniformDetailField(t);return [...new Set(field?[field.storage.physical(field.handle),field.storage.baseOf(field.handle)!]:[t])].filter(Boolean);};
  const outputTextures=[...new Set([...homes(f.velocityScratch),physical(taps.unit),physical(frame.hanging.unitVelocity),flip.coarseVelocity,...extrap.hierarchyLevels.slice(1).flatMap((level:any)=>[level.down,level.up,level.originsDown,level.originsUp])])] as GPUTexture[];
  const textureRecords=outputTextures.map(texture=>{const bytesPerRow=Math.ceil(texture.width*16/256)*256,extent=[texture.width,texture.height,texture.depthOrArrayLayers] as [number,number,number],size=bytesPerRow*texture.height*texture.depthOrArrayLayers;return{texture,bytesPerRow,rowsPerImage:texture.height,extent,original:allocate(size),reference:allocate(size)};});
  const particles=flip.activeParticles as GPUBuffer;flip.parity^=1;
  const stateRead=allocate(flip.state.size,true);let e=raw.createCommandEncoder();e.copyBufferToBuffer(flip.state,0,stateRead,0,stateRead.size);raw.queue.submit([e.finish()]);await stateRead.mapAsync(GPUMapMode.READ);const state=new Uint32Array(stateRead.getMappedRange().slice(0));stateRead.unmap();const count=state[0]!;assert.ok(count>0);state[1]=count;raw.queue.writeBuffer(flip.state,0,state);
  const companion=allocate(rawOld.size);
  const common={phase:f.phase,output:f.velocityScratch,outputNegative:f.negativeScratch,scratch:{buffer:frame.stageScratch},params:frame.params.extension,companion:{scratch:companion,physical:f.departure,negative:f.negativeDeparture}};
  const pairGroups=ext.bind({...common,physical:f.velocity,negative:f.negative});
  const snapshotGroups=ext.bind({...common,physical:f.departure,negative:f.negativeDeparture,snapshot:taps.unit});
  const particleSnapshots=ext.companionBindings(companion).map((slot:GPUBufferBinding)=>flip.bindSnapshotSlots(slot));
  const bufferRecords=[{buffer:companion,size:companion.size},{buffer:particles,size:48*count},{buffer:f.negativeScratch as GPUBuffer,size:f.negativeScratch.size},{buffer:o.hanging as GPUBuffer,size:o.hanging.size}].map(r=>({...r,original:allocate(r.size),reference:allocate(r.size)}));
  const originals=(encoder:GPUCommandEncoder,which:"original"|"reference")=>{textureRecords.forEach(r=>encoder.copyTextureToBuffer({texture:r.texture},{buffer:r[which],bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},r.extent));bufferRecords.forEach(r=>encoder.copyBufferToBuffer(r.buffer,0,r[which],0,r.size));};
  const restore=(encoder:GPUCommandEncoder)=>{textureRecords.forEach(r=>encoder.copyBufferToTexture({buffer:r.original,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},{texture:r.texture},r.extent));bufferRecords.forEach(r=>encoder.copyBufferToBuffer(r.original,0,r.buffer,0,r.size));};
  e=raw.createCommandEncoder();originals(e,"original");raw.queue.submit([e.finish()]);
  const marker=await raw.createComputePipelineAsync({layout:"auto",compute:{module:raw.createShaderModule({code:"@compute @workgroup_size(1) fn marker(){}"}),entryPoint:"marker"}});
  const comparisonCode=`
@group(0) @binding(0) var input:texture_3d<f32>;
@group(0) @binding(1) var<storage,read> expected:array<u32>;
@group(0) @binding(2) var<storage,read_write> result:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> current:array<u32>;
override rowWords:u32=0u;
fn note(i:u32,a:u32,b:u32){if(a!=b){atomicAdd(&result[0],1u);atomicMin(&result[1],i);}}
@compute @workgroup_size(256) fn textureCompare(@builtin(global_invocation_id) gid:vec3u){let at=gid.x+65535u*256u*gid.y;let d=textureDimensions(input);if(at>=d.x*d.y*d.z){return;}let q=vec3u(at%d.x,(at/d.x)%d.y,at/(d.x*d.y));let a=bitcast<vec4u>(textureLoad(input,vec3i(q),0));let offset=4u*q.x+rowWords*(q.y+d.y*q.z);for(var c=0u;c<4u;c++){note(4u*at+c,a[c],expected[offset+c]);}}
@compute @workgroup_size(256) fn bufferCompare(@builtin(global_invocation_id) gid:vec3u){let i=gid.x+65535u*256u*gid.y;if(i<arrayLength(&expected)){note(i,current[i],expected[i]);}}
`;
  const comparison=raw.createShaderModule({code:comparisonCode}),integerComparison=raw.createShaderModule({code:comparisonCode.replace("texture_3d<f32>","texture_3d<u32>").replace("bitcast<vec4u>(textureLoad(input,vec3i(q),0))","textureLoad(input,vec3i(q),0)")});
  const results=allocate(16),read=allocate(16,true);
  const comparisons=await Promise.all([...textureRecords.map(async r=>{const pipeline=await raw.createComputePipelineAsync({layout:"auto",compute:{module:r.texture.format==="rgba32uint"?integerComparison:comparison,entryPoint:"textureCompare",constants:{rowWords:r.bytesPerRow/4}}});return{words:4*r.extent.reduce((a,b)=>a*b,1),invocations:r.extent.reduce((a,b)=>a*b,1),pipeline,group:raw.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:r.texture.createView()},{binding:1,resource:{buffer:r.reference}},{binding:2,resource:{buffer:results}}]})};}),...bufferRecords.map(async r=>{const pipeline=await raw.createComputePipelineAsync({layout:"auto",compute:{module:comparison,entryPoint:"bufferCompare"}});return{words:r.size/4,invocations:r.size/4,pipeline,group:raw.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:r.reference}},{binding:2,resource:{buffer:results}},{binding:3,resource:{buffer:r.buffer,size:r.size}}]})};})]);
  const checkpointTextures=textureRecords.map((record,index)=>({record,comparison:comparisons[index]!})).filter(({record})=>record.texture===physical(taps.unit)||record.texture===flip.coarseVelocity).map(item=>({...item,reference:allocate(item.record.reference.size)}));
  const checkpointBuffers=bufferRecords.slice(0,2).map((record,index)=>({record,comparison:comparisons[textureRecords.length+index]!,reference:allocate(record.size)}));
  const oldComparisons=[...checkpointTextures.map(({record,comparison,reference})=>({...comparison,group:raw.createBindGroup({layout:comparison.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:record.texture.createView()},{binding:1,resource:{buffer:reference}},{binding:2,resource:{buffer:results}}]})})),...checkpointBuffers.map(({record,comparison,reference})=>({...comparison,group:raw.createBindGroup({layout:comparison.pipeline.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:reference}},{binding:2,resource:{buffer:results}},{binding:3,resource:{buffer:record.buffer,size:record.size}}]})}))];
  const slotBytes=ext.scratchBytesAt(o.capacity.fineTiles)/2,liveSlots=allocate(slotBytes);
  const livePipeline=await raw.createComputePipelineAsync({layout:"auto",compute:{module:comparison,entryPoint:"bufferCompare"}});
  const liveComparisons=[0,1].map(epoch=>({words:slotBytes/4,invocations:slotBytes/4,pipeline:livePipeline,group:raw.createBindGroup({layout:livePipeline.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:liveSlots}},{binding:2,resource:{buffer:results}},{binding:3,resource:{buffer:frame.stageScratch,offset:epoch*slotBytes,size:slotBytes}}]})}));
  const compare=(encoder:GPUCommandEncoder,items:typeof comparisons)=>{const pass=encoder.beginComputePass({label:"Compare publication replay outputs"});for(const c of items){pass.setPipeline(c.pipeline);pass.setBindGroup(0,c.group);const n=Math.ceil(c.invocations/256);pass.dispatchWorkgroups(Math.min(65535,n),Math.ceil(n/65535));}pass.end();};
  let checkpoint:((encoder:GPUCommandEncoder)=>void)|undefined;
  const prepare=(encoder:GPUCommandEncoder,activeSweeps:number)=>{restore(encoder);encoder.copyBufferToBuffer(rawOld,0,companion,0,companion.size);ext.preparePair(encoder,pairGroups,activeSweeps);encoder.copyBufferToBuffer(frame.stageScratch,(activeSweeps%2)*slotBytes,liveSlots,0,slotBytes);};
  let forceFallback=false;
  const encode=(encoder:GPUCommandEncoder,fused:boolean,activeSweeps:number)=>{
   ext.snapshotPipelines.fine=fused?productionSnapshot.fine:originalPipelines.snapshotFine;ext.snapshotPipelines.coarse=fused?productionSnapshot.coarse:originalPipelines.snapshotCoarse;
   ext.replayPublishers=!fused?undefined:forceFallback?{regular:new Map([["publish",fallbackVariants[0]]]),seams:new Map([["publish",fallbackVariants[1]]]),list:new Map([["publish",fallbackVariants[2]]])}:productionReplay;
   ext.finishSnapshot(encoder,snapshotGroups,activeSweeps);flip.snapshot(encoder,particleSnapshots[activeSweeps%2]);checkpoint?.(encoder);
   ext.finishPair(encoder,frame.extensionGroups,activeSweeps,false);
   const pass=encoder.beginComputePass({label:"Narrow-band FLIP update sample"});flip.cache.encode(encoder,flip.cacheGroups.update,pass);flip.encodeStageTaps(encoder,pass);flip.encodeParticleWork(encoder,pass);flip.dispatchBatch(encoder,["update"],undefined,"update",pass);pass.end();
   ext.snapshotPipelines.fine=productionSnapshot.fine;ext.snapshotPipelines.coarse=productionSnapshot.coarse;ext.replayPublishers=productionReplay;
  };
  const parity=[];
  for(const variant of [...new Set([sweeps,0,1,3])].map(sweeps=>({sweeps,fallback:false})).concat([{sweeps,fallback:true}])){
   const activeSweeps=variant.sweeps;forceFallback=variant.fallback;
   raw.queue.writeBuffer(results,0,new Uint32Array([0,0xffffffff,0,0]));
   for(const fused of [false,true]){
    checkpoint=fused?(encoder=>compare(encoder,oldComparisons)):(encoder=>{checkpointTextures.forEach(({record:r,reference})=>encoder.copyTextureToBuffer({texture:r.texture},{buffer:reference,bytesPerRow:r.bytesPerRow,rowsPerImage:r.rowsPerImage},r.extent));checkpointBuffers.forEach(({record:r,reference})=>encoder.copyBufferToBuffer(r.buffer,0,reference,0,r.size));});
    e=raw.createCommandEncoder();prepare(e,activeSweeps);encode(e,fused,activeSweeps);if(!fused)originals(e,"reference");else compare(e,[...comparisons,liveComparisons[activeSweeps%2]!]);raw.queue.submit([e.finish()]);
   }
   checkpoint=undefined;
   e=raw.createCommandEncoder();e.copyBufferToBuffer(results,0,read,0,16);raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const resultWords=[...new Uint32Array(read.getMappedRange())];read.unmap();const result={sweeps:activeSweeps,forcedFallback:forceFallback,words:[...comparisons,...oldComparisons,liveComparisons[0]!].reduce((s,c)=>s+c.words,0),mismatches:resultWords[0],firstMismatch:resultWords[1]};parity.push(result);console.log(JSON.stringify({phase:"parity",...result}));assert.equal(result.mismatches,0);
  }
  forceFallback=false;
  if(process.argv.includes("--wait-for-gpu-quiet")){let seconds=0;for(;;){const others=execFileSync("ps",["-Ao","pid=,args="],{encoding:"utf8"}).split("\n").filter(line=>{const row=line.match(/^\s*(\d+)\s+(.*)$/);return row&&Number(row[1])!==process.pid&&/^(?:\S*\/)?node\s/.test(row[2]!)&&/tools\/(?:probe|benchmark)[^\s]*\.ts/.test(row[2]!);});if(!others.length)break;if(seconds++%60===0)console.log(JSON.stringify({phase:"waiting-before-timing",others}));await new Promise(resolve=>setTimeout(resolve,1000));}}
  const trials:{fused:boolean;ms:number;passes:Awaited<ReturnType<GPUPassProfile["finish"]>>}[]=[];
  for(let round=0;round<rounds+warmup;round++)for(const fused of round%2?[true,false]:[false,true]){
   e=raw.createCommandEncoder();prepare(e,sweeps);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();profile.start();e=device.createCommandEncoder();const start=e.beginComputePass({label:"Publication replay interval start"});start.setPipeline(marker);start.dispatchWorkgroups(1);start.end();encode(e,fused,sweeps);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();const passes=await profile.finish();if(round>=warmup)trials.push({fused,ms:passes.reduce((s,p)=>s+p.ms+p.idle_ms,0),passes});
  }
  const saving=stats(Array.from({length:rounds},(_,i)=>{const pair=trials.slice(2*i,2*i+2);return pair.find(t=>!t.fused)!.ms-pair.find(t=>t.fused)!.ms;}));
  const report={date:new Date().toISOString(),sceneId:definition.id,values,dt,steps,rounds,sweeps,count,parity,distributions_ms:{original:stats(trials.filter(t=>!t.fused).map(t=>t.ms)),fused:stats(trials.filter(t=>t.fused).map(t=>t.ms)),saving},trials,errors};mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify(report));assert.deepEqual(errors,[]);
 }finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await run();
