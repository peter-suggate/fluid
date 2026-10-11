/** Frozen whole momentum/authority/P2G interval, including compact queue construction.
 * node --import tsx tools/probe-compact-momentum.ts --steps=96 --rounds=10
 */
import assert from "node:assert/strict";
import {existsSync,mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {dirname,resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {resolveMethodValues} from "../lib/core/method-contract";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {uniformNarrowBandMethod as method} from "../lib/methods/uniform/uniform-narrow-band-method";
import {uniformDetailField,uniformDetailBindLayout,uniformDetailGroup} from "../lib/methods/uniform/uniform-detail-fields";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {GPUPassProfile} from "./gpu-pass-profile";
import {compactMomentumSource,compactMomentumCellSource,createCompactMomentumReplay} from "./compact-momentum";
import {createTransferReachReplay} from "./transfer-reach-cache";

type RecordCopy={name:string;size:number;output:boolean;capture(e:GPUCommandEncoder,to:GPUBuffer):void;restore(e:GPUCommandEncoder,from:GPUBuffer):void};
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option("steps")??96),rounds=Number(option("rounds")??8);
assert.ok(Number.isInteger(steps)&&steps>0&&Number.isInteger(rounds)&&rounds>0);
const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),quality=definition.methodProfile!.quality;
const scene=structuredClone(sceneDocument(definition));
const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
const out=option("out")??"artifacts/nbflip-ui-perf/compact-momentum.json";
let solver:WebGPUUniformReferenceSolver|undefined,device:GPUDevice|undefined,profile:GPUPassProfile|undefined;
const errors:string[]=[],buffers=new Set<GPUBuffer>();
try{
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);
 const sources=new WeakMap<GPUShaderModule,string>(),descriptors=new WeakMap<GPUComputePipeline,GPUComputePipelineDescriptor>(),layouts=new WeakMap<GPUPipelineLayout,GPUPipelineLayoutDescriptor>(),groups=new WeakMap<GPUBindGroup,GPUBindGroupDescriptor>();
 const capture=new Proxy(profile.device,{get:(target,key)=>{
  if(key==="createBuffer")return(d:GPUBufferDescriptor)=>target.createBuffer((d.usage&(GPUBufferUsage.MAP_READ|GPUBufferUsage.MAP_WRITE))?d:{...d,usage:d.usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  if(key==="createTexture")return(d:GPUTextureDescriptor)=>target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
  if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);sources.set(m,d.code);return m;};
  if(key==="createPipelineLayout")return(d:GPUPipelineLayoutDescriptor)=>{const l=target.createPipelineLayout(d);layouts.set(l,{...d,bindGroupLayouts:[...d.bindGroupLayouts]});return l;};
  if(key==="createBindGroup")return(d:GPUBindGroupDescriptor)=>{const g=target.createBindGroup(d);groups.set(g,{...d,entries:[...d.entries]});return g;};
  if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{const p=await target.createComputePipelineAsync(d);descriptors.set(p,d);return p;};
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});
 device=managedGPUDevice(capture,{requireWorkerRealm:false});
 device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const frame=(solver as any).mixedFrame,flip=frame.narrowBandFlip,momentum=frame.momentum;
 const mode=option("mode")??"compact";
 assert.ok(["compact","compact-cells","properties-cache","properties-empty"].includes(mode));
 const compactMode=mode.startsWith("compact"),directCells=mode==="compact-cells";
 const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.COPY_SRC|GPUBufferUsage.STORAGE)});buffers.add(b);return b;};
 const release=(b:GPUBuffer)=>{b.destroy();buffers.delete(b);};
 const bufferRecord=(name:string,b:GPUBuffer,output=false,offset=0,size=b.size-offset):RecordCopy=>({name,size,output,capture:(e,to)=>e.copyBufferToBuffer(b,offset,to,0,size),restore:(e,from)=>e.copyBufferToBuffer(from,0,b,offset,size)});
 const textureRecords=(name:string,handle:GPUTexture,output=false):RecordCopy[]=>{
  const field=uniformDetailField(handle),textures=[...new Set(field?[field.storage.physical(field.handle),field.storage.baseOf(field.handle)!]:[handle])];
  return textures.map((texture,bank)=>{const pixel=texture.format==="rgba32float"?16:4,bytesPerRow=Math.ceil(texture.width*pixel/256)*256,rowsPerImage=texture.height;
   const extent:[number,number,number]=[texture.width,texture.height,texture.depthOrArrayLayers],size=bytesPerRow*rowsPerImage*texture.depthOrArrayLayers;
   return{name:`${name}-${bank}`,size,output,capture:(e,to)=>e.copyTextureToBuffer({texture},{buffer:to,bytesPerRow,rowsPerImage},extent),restore:(e,from)=>e.copyBufferToTexture({buffer:from,bytesPerRow,rowsPerImage},{texture},extent)};
  });
 };
 let records:RecordCopy[]=[],snapshots:GPUBuffer[]=[],parameters:any,parity=0,captured=0,step=0;
 const original=frame.encodeParticleMomentum.bind(frame);
 const tiles=frame.ownership.capacity.tiles;
 const producerCertificate=allocate(frame.ownership.speeds.size);
 const originalCertificate=frame.plan.encodeCertificate.bind(frame.plan);
 frame.plan.encodeCertificate=(encoder:GPUCommandEncoder,dt:number)=>{
  originalCertificate(encoder,dt);
  if(step===steps)encoder.copyBufferToBuffer(frame.ownership.speeds,0,producerCertificate,0,producerCertificate.size);
 };
 frame.encodeParticleMomentum=(encoder:GPUCommandEncoder,p:any)=>{
  if(step===steps){
   assert.equal(captured++,0);parameters=p;parity=flip.parity;
   records=[bufferRecord("bins",flip.bins),bufferRecord("links",flip.next),bufferRecord("particles",flip.particles[1-parity],true),bufferRecord("state",flip.state,true),bufferRecord("authority",frame.authorityPhi,true),bufferRecord("support",frame.ownership.support),bufferRecord("speeds",frame.ownership.speeds),bufferRecord("negativeDeparture",frame.fields.negativeDeparture,true),bufferRecord("negativeScratch",frame.fields.negativeScratch),
    ...textureRecords("extended",frame.fields.velocityScratch),...textureRecords("coarseCache",frame.coarseCache),...textureRecords("unitVelocity",frame.hanging.unitVelocity),...textureRecords("velocity",frame.fields.velocity,true),...textureRecords("departure",frame.fields.departure,true),...textureRecords("phase",frame.fields.phase,true),...textureRecords("correction",frame.fields.correction,true)];
   snapshots=records.map(r=>allocate(r.size));records.forEach((r,i)=>r.capture(encoder,snapshots[i]!));
  }
  original(encoder,p);
 };
 const momentumDescriptor=descriptors.get(momentum.generalPipeline)!;
 if(compactMode)writeFileSync("/tmp/fluid-compact-momentum.wgsl",(directCells?compactMomentumCellSource:compactMomentumSource)(sources.get(momentumDescriptor.compute.module)!));
 const compact=compactMode?await createCompactMomentumReplay(raw,frame,{sources,descriptors,groups},false,directCells):undefined;
 const reach=!compactMode?await createTransferReachReplay(raw,frame,{sources,descriptors,groups},mode==="properties-empty"):undefined;
 writeFileSync(`/tmp/fluid-${mode}.wgsl`,(compact??reach)!.code);
 console.log(JSON.stringify({phase:"compiled",mode,queueBytes:compact?.bytes??0}));
 for(step=1;step<=steps;step++){solver.advanceTo(step*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(step%12===0||step===steps)console.log(JSON.stringify({phase:"advance",frame:step}));}
 frame.encodeParticleMomentum=original;frame.plan.encodeCertificate=originalCertificate;assert.equal(captured,1);flip.parity=parity;
 compact?.bind();
 assert.deepEqual(errors,[]);
 const restore=(e:GPUCommandEncoder)=>records.forEach((r,i)=>r.restore(e,snapshots[i]!));
 let splitCompact=false;
 const encode=(e:GPUCommandEncoder,candidate:boolean)=>{
  if(!candidate){original(e,parameters);return;}
  if(reach){reach.run(()=>original(e,parameters));return;}
  const previous=momentum.encode;momentum.encode=(encoder:GPUCommandEncoder)=>compact!.encode(encoder,splitCompact);
  try{original(e,parameters);}finally{momentum.encode=previous;}
 };
 const read=async(buffer:GPUBuffer,size=buffer.size,offset=0)=>{const staging=allocate(size,true),e=raw.createCommandEncoder();e.copyBufferToBuffer(buffer,offset,staging,0,size);raw.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);const words=new Uint32Array(staging.getMappedRange().slice(0));staging.unmap();release(staging);return words;};
 const stateSnapshot=snapshots[records.findIndex(r=>r.name==="state")]!;
 const counts=await read(stateSnapshot),liveCount=counts[1]!;
 // Capture the certificate independently of transfer metadata. This also
 // detects a stale/overwritten class word instead of trusting a sign tag.
 const reachWords=await read(snapshots[records.findIndex(r=>r.name==="speeds")]!,4*tiles,4*tiles);
 const supportHeader=await read(snapshots[records.findIndex(r=>r.name==="support")]!,64,16*tiles);
 const producedReachWords=await read(producerCertificate,4*tiles,4*tiles);
 let overwrittenCertificateWords=0;const certificateOverwriteExamples:any[]=[];
 for(let i=0;i<tiles;i++)if(reachWords[i]!==producedReachWords[i]){overwrittenCertificateWords++;if(certificateOverwriteExamples.length<8)certificateOverwriteExamples.push({tile:i,produced:producedReachWords[i],beforeMomentum:reachWords[i]});}
 const classHistogram=[0,0,0,0],reachHistogram=Array(32).fill(0);
 for(const word of reachWords){classHistogram[word>>>30]!++;for(let axis=0;axis<6;axis++)reachHistogram[(word>>>(5*axis))&31]!++;}
 const certificateSnapshot={tiles,classHistogram,reachHistogram,overwrittenCertificateWords,certificateOverwriteExamples,supportHeader:[...supportHeader],regularListCount:supportHeader[1],generalListCount:supportHeader[2]};
 console.log(JSON.stringify({phase:"captured-certificate",...certificateSnapshot}));
 // Only the working prefix can be touched by classification. Unallocated
 // particle capacity remains saved for repeatable restoration but need not be compared.
 const outputs=records.filter(r=>r.output).map(r=>r.name==="particles"?{...r,size:48*liveCount,capture:(e:GPUCommandEncoder,to:GPUBuffer)=>e.copyBufferToBuffer(flip.particles[1-parity],0,to,0,48*liveCount)}:r);
 const resultBuffers:GPUBuffer[][]=[];
 for(const candidate of [false,true]){
  const e=raw.createCommandEncoder();restore(e);encode(e,candidate);const result=outputs.map(r=>allocate(r.size));outputs.forEach((r,i)=>r.capture(e,result[i]!));raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();resultBuffers.push(result);
 }
 let words=0,mismatches=0,maxAbsolute=0,maxULP=0,signedZeros=0;const banks:any[]=[],examples:any[]=[];
 for(let bank=0;bank<outputs.length;bank++){
  const record=outputs[bank]!;let differences=0;
  for(let offset=0;offset<record.size;offset+=16*1024*1024){
   const size=Math.min(record.size-offset,16*1024*1024),a=await read(resultBuffers[0]![bank]!,size,offset),b=await read(resultBuffers[1]![bank]!,size,offset);
   const af=new Float32Array(a.buffer),bf=new Float32Array(b.buffer);
   for(let i=0;i<a.length;i++)if(a[i]!==b[i]){differences++;if(af[i]===0&&bf[i]===0)signedZeros++;else if(Number.isFinite(af[i])&&Number.isFinite(bf[i])){const order=(u:number)=>(u&0x80000000)?0x80000000-(u&0x7fffffff):0x80000000+u;maxULP=Math.max(maxULP,Math.abs(order(a[i]!)-order(b[i]!)));}const absolute=Math.abs(af[i]!-bf[i]!);if(Number.isFinite(absolute))maxAbsolute=Math.max(maxAbsolute,absolute);if(examples.length<16)examples.push({bank:record.name,word:offset/4+i,baselineBits:a[i],candidateBits:b[i],baseline:af[i],candidate:bf[i]});}
  }
  words+=record.size/4;mismatches+=differences;banks.push({name:record.name,words:record.size/4,mismatches:differences});resultBuffers.forEach(a=>release(a[bank]!));
 }
 console.log(JSON.stringify({phase:"parity",words,mismatches,maxAbsolute,maxULP,signedZeros,banks,examples}));
 const kept=compact?await read(compact.scratch,16,4*compact.inputBase):new Uint32Array(4),escaped=compact?await read(compact.scratch,16):new Uint32Array(4);
 const coverage={queuedTiles:directCells?undefined:kept[0],queuedCells:directCells?kept[0]:kept[1],escapedTiles:directCells?undefined:escaped[0],escapedCells:directCells?escaped[0]:undefined,generalFineCells:64*certificateSnapshot.generalListCount!,regularFineCells:64*certificateSnapshot.regularListCount!};
 console.log(JSON.stringify({phase:"coverage",...coverage}));
 let e:GPUCommandEncoder;
 for(const candidate of [false,true]){e=raw.createCommandEncoder();restore(e);encode(e,candidate);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();}
 const timingReady=option("timing-ready");
 if(timingReady&&!existsSync(timingReady)){
  console.log(JSON.stringify({phase:"waiting-for-exclusive-timing",readyFile:timingReady}));
  while(!existsSync(timingReady))await new Promise(resolve=>setTimeout(resolve,1000));
 }
 // The timing gate may leave the GPU idle long enough to change its power
 // state. Warm both variants after the gate, outside the measured pairs.
 for(let warm=0;warm<3;warm++)for(const candidate of warm%2?[true,false]:[false,true]){e=raw.createCommandEncoder();restore(e);encode(e,candidate);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();}
 const timings:any[]=[],totals={baseline:[] as number[],candidate:[] as number[]};
 for(let round=0;round<rounds;round++)for(const candidate of round%2?[true,false]:[false,true]){
  e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  profile.start();e=device.createCommandEncoder();encode(e,candidate);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();const passes=await profile.finish();
  const ms=passes.reduce((n,p)=>n+p.ms,0),elapsed=passes.reduce((n,p)=>n+p.ms+p.idle_ms,0),variant=candidate?"candidate":"baseline";
  totals[variant].push(elapsed);timings.push({round,variant,ms,elapsed_ms:elapsed,passes});
 }
 // Separate diagnostic dispatches reveal where staging moves cost. These
 // timestamps are not included in the unchanged whole-interval A/B above.
 const dispatchBatch=flip.dispatchBatch.bind(flip),kernelBreakdown:any[]=[];
 flip.dispatchBatch=(encoder:GPUCommandEncoder,entries:string[],group:string|undefined,label:string,shared?:GPUComputePassEncoder)=>{
  if(shared)return dispatchBatch(encoder,entries,group,label,shared);
  for(const entry of entries)dispatchBatch(encoder,[entry],group,entry);
 };
 try{for(const candidate of [false,true]){
  splitCompact=true;e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
  profile.start();e=device.createCommandEncoder();encode(e,candidate);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();
  kernelBreakdown.push({variant:candidate?"candidate":"baseline",passes:await profile.finish()});
 }}finally{flip.dispatchBatch=dispatchBatch;}
 const mean=(a:number[])=>a.reduce((n,v)=>n+v,0)/a.length;
 const result={date:new Date().toISOString(),sceneId:definition.id,mode,values,dt,steps,rounds,liveCount,words,mismatches,maxAbsolute,maxULP,signedZeros,banks,examples,certificateSnapshot,coverage,means_ms:{baseline:mean(totals.baseline),candidate:mean(totals.candidate)},timings,kernelBreakdown,errors};
 mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));
 assert.deepEqual(errors,[]);assert.equal(mismatches,0,"Frozen combined outputs must be bit exact");
}finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
