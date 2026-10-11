/** Inclusive replay of raw-canonical retention versus the whole old RGBA copy. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailPick } from "../lib/methods/uniform/uniform-detail-fields";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";
import { coarseSweepSource } from "./extension-coarse-sweep";
import { Capture } from "./probe-extension-direct-snapshot";
import { retainCanonicalSource, retainedCurrentSeed, retainedSnapshot, originalCurrentSeed, originalSnapshot, packedCanonicalSource } from "./extension-retained-seed";

const distribution=(values:number[])=>{const sorted=[...values].sort((a,b)=>a-b),mean=values.reduce((a,b)=>a+b,0)/values.length;return {mean,median:(sorted[Math.floor((sorted.length-1)/2)]!+sorted[Math.ceil((sorted.length-1)/2)]!)/2,min:sorted[0],max:sorted.at(-1),stddev:Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/values.length)};};
async function run():Promise<void>{
 const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
 const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),quality=definition.methodProfile!.quality,scene=structuredClone(sceneDocument(definition));
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s,steps=Number(option("steps")??96),rounds=Number(option("rounds")??12),sweeps=Number(values.extensionFrontSweeps??2);
 const coarseAfter=process.argv.includes("--coarse-after"),packedAfter=process.argv.includes("--packed-after"),retainGrid=Number(option("retain-grid")??4096);
 let snapshotCap=0;
 let packing=process.argv.includes("--packed"),sweepMode:"none"|"early"|"direct"="none";
 const out=option("out")??"artifacts/nbflip-ui-perf/extension-retained-seed.json";
 let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
 const buffers:GPUBuffer[]=[],errors:string[]=[];
 try{
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  profile=new GPUPassProfile(raw);const capture=new Capture(profile.device);device=managedGPUDevice(capture.device,{requireWorkerRealm:false});
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  const initial=(solver as any).mixedFrame,ext=initial.extension;
  assert.ok(ext.supportsPair);
  const initialSeed=uniformDetailPick(ext.pairedRegular.get("seed")),seedDescriptor=capture.pipelines.get(initialSeed)!;
  const productionPairSource=capture.sources.get(seedDescriptor.compute.module)!;
  const integrated=productionPairSource.includes("fn retainFine("),pairSource=originalCurrentSeed(productionPairSource);
  const snapshotDescriptor=capture.pipelines.get(uniformDetailPick(ext.snapshotPipelines.fine))!;
  const snapshotSource=originalSnapshot(capture.sources.get(snapshotDescriptor.compute.module)!);
  const modules=new Map<string,GPUShaderModule>();
  const sources:readonly (readonly [string,string])[]=[["original-seed",pairSource],["original-snapshot",snapshotSource],["seed",retainedCurrentSeed(pairSource)],["retain",retainCanonicalSource(pairSource)],["compare",retainCanonicalSource(pairSource,true,coarseAfter)],["snapshot",retainedSnapshot(snapshotSource)],...((packing||packedAfter)?[["packed",packedCanonicalSource(integrated?productionPairSource:retainCanonicalSource(pairSource))] as const]:[]),...(coarseAfter?[["early",coarseSweepSource(productionPairSource)],["direct",coarseSweepSource(productionPairSource,true)]] as const:[])];
  for(const [name,code] of sources){
   const module=raw.createShaderModule({label:`Retained raw ${name}`,code}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(messages.length){writeFileSync(`/tmp/fluid-retained-${name}.wgsl`,code);throw new Error(messages.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));}modules.set(name,module);
  }
  const seedMaps=[ext.pairedRegular,ext.pairedList,ext.pairedSeams] as Map<string,GPUComputePipeline>[];
  const productionSeeds=seedMaps.map(map=>map.get("seed")!);
  const originalSweep=ext.pairedSeams.get("sweep") as GPUComputePipeline,sweepDescriptor=capture.pipelines.get(originalSweep)!;
  const sweepVariants=Object.fromEntries(await Promise.all((coarseAfter?["early","direct"]:[]).map(async name=>[name,await raw.createComputePipelineAsync({...sweepDescriptor,compute:{...sweepDescriptor.compute,module:modules.get(name)!}})])));
  const compileSeeds=(name:string)=>Promise.all(productionSeeds.map(p=>{const descriptor=capture.pipelines.get(p)!;return raw.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module:modules.get(name)!}});}));
  const seeds=integrated&&!coarseAfter?await compileSeeds("original-seed"):productionSeeds,seedVariants=integrated?productionSeeds:await compileSeeds("seed");
  const productionSnapshot=ext.snapshotPipelines;
  const compileSnapshot=async(module:string)=>Object.fromEntries(await Promise.all(Object.entries(productionSnapshot).map(async([name,p])=>{const descriptor=capture.pipelines.get(p as GPUComputePipeline)!;return [name,await raw.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module:modules.get(module)!}})] as const;})));
  const referenceSnapshot=integrated&&!coarseAfter?await compileSnapshot("original-snapshot"):productionSnapshot,snapshotVariants=integrated?productionSnapshot:await compileSnapshot("snapshot");
  const originalLayout=seedDescriptor.layout as GPUPipelineLayout;
  const compareLayout=raw.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...(coarseAfter?[{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}]:[])]});
  const proofSlotsLayout=raw.createBindGroupLayout({entries:[{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...(coarseAfter?[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}]:[])]});
  const proofLayout=raw.createPipelineLayout({bindGroupLayouts:[initial.ownership.bindLayout,proofSlotsLayout,compareLayout]});
  const makeJobs=async(name:string,layout:GPUPipelineLayout)=>Promise.all(["retainFine","retainCoarse"].map(entryPoint=>raw.createComputePipelineAsync({layout,compute:{module:modules.get(name)!,entryPoint}})));
  const retain=integrated?ext.retainPipelines:await makeJobs("retain",originalLayout),compare=await makeJobs("compare",proofLayout),packed=(packing||packedAfter)?await makeJobs("packed",originalLayout):undefined;
  const marker=await raw.createComputePipelineAsync({layout:"auto",compute:{module:raw.createShaderModule({code:"@compute @workgroup_size(1) fn marker() {}"}),entryPoint:"marker"}});
  console.log(JSON.stringify({phase:"source-preflight"}));
  const lifetimeBytes=ext.companionBytesAt(initial.ownership.capacity.fineTiles);
  const lifetimeReads=integrated?[0,1].map(()=>raw.createBuffer({size:lifetimeBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})):[];buffers.push(...lifetimeReads);
  let liveBank:GPUBuffer|undefined;
  for(let n=1;n<=steps;n++){
   // Capture the final pre-force raw input solely for this frozen reference.
   // Production snapshot reads retained slots; departure has no later reader.
   if(integrated&&n===steps){
    const retain=ext.retainPair.bind(ext),prepare=ext.preparePair.bind(ext);
    ext.retainPair=(encoder:GPUCommandEncoder,groups:any)=>{
     initial.copyWhole(encoder,initial.fields.velocity,initial.fields.departure);retain(encoder,groups);
     liveBank=initial.narrowBandFlip.extensionScratch;encoder.copyBufferToBuffer(liveBank!,0,lifetimeReads[0]!,0,lifetimeBytes);
    };
    ext.preparePair=(encoder:GPUCommandEncoder,groups:any,sweeps:number)=>{
     assert.ok(liveBank);encoder.copyBufferToBuffer(liveBank,0,lifetimeReads[1]!,0,lifetimeBytes);prepare(encoder,groups,sweeps);
    };
   }
   solver.advanceTo(n*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(n%24===0||n===steps)console.log(JSON.stringify({phase:"advance",frame:n}));}
  let lifetime:undefined|{words:number;mismatches:number;firstMismatch:number};
  if(integrated){
   await Promise.all(lifetimeReads.map(b=>b.mapAsync(GPUMapMode.READ)));
   const before=new Uint32Array(lifetimeReads[0]!.getMappedRange()),after=new Uint32Array(lifetimeReads[1]!.getMappedRange());
   lifetime={words:before.length,mismatches:0,firstMismatch:-1};
   for(let i=0;i<before.length;i++)if(before[i]!==after[i]){lifetime.mismatches++;if(lifetime.firstMismatch<0)lifetime.firstMismatch=i;}
   lifetimeReads.forEach(b=>b.unmap());console.log(JSON.stringify({phase:"borrowed-bank-lifetime",...lifetime}));assert.equal(lifetime.mismatches,0);
   const prototype=Object.getPrototypeOf(ext);ext.retainPair=prototype.retainPair.bind(ext);ext.preparePair=prototype.preparePair.bind(ext);
  }
  const frame=(solver as any).mixedFrame,flip=frame.narrowBandFlip,ownership=frame.ownership,f=frame.fields;
  assert.equal(frame.extension,ext,"fixture must keep compiled extension");
  const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC)});buffers.push(b);return b;};
  const particles=flip.activeParticles as GPUBuffer;flip.parity^=1;
  const countRead=allocate(8,true);let e=raw.createCommandEncoder();e.copyBufferToBuffer(flip.state,0,countRead,0,8);raw.queue.submit([e.finish()]);
  await countRead.mapAsync(GPUMapMode.READ);const count=new Uint32Array(countRead.getMappedRange())[0]!;countRead.unmap();assert.ok(count>0);raw.queue.writeBuffer(flip.state,4,new Uint32Array([count]));
  const companion=allocate(ext.companionBytesAt(ownership.capacity.fineTiles)),slots=ext.companionBindings(companion) as GPUBufferBinding[];
  const stateBytes=ext.scratchBytesAt(ownership.capacity.fineTiles)/2,heldState=coarseAfter?allocate(stateBytes):undefined;
  const heldSlots=allocate(slots[0]!.size!),slotStats=allocate(16),slotRead=allocate(16,true),original=allocate(count*48),particleReference=allocate(count*16),particleStats=allocate(16),particleRead=allocate(16,true);
  const common={phase:f.phase,output:f.velocityScratch,outputNegative:f.negativeScratch,scratch:{buffer:frame.stageScratch},params:frame.params.extension};
  const companionField={scratch:companion,physical:f.departure,negative:f.negativeDeparture};
  const current=ext.bind({...common,physical:f.velocity,negative:f.negative,companion:companionField});
  const old=ext.bind({...common,physical:f.departure,negative:f.negativeDeparture,companion:companionField,snapshot:flip.snapshotUnitVelocity});
  const gather=ext.bind({...common,physical:f.velocityScratch,negative:f.negativeDeparture,output:f.departure,companion:{...companionField,physical:f.velocity}});
  const snapshots=slots.map(s=>flip.bindSnapshotSlots(s));
  const proofSlots=slots.map((resource,epoch)=>raw.createBindGroup({layout:proofSlotsLayout,entries:[{binding:2,resource},...(coarseAfter?[{binding:0,resource:{buffer:frame.stageScratch,offset:epoch*stateBytes,size:stateBytes}}]:[])]}));
  const proofGroup=raw.createBindGroup({layout:compareLayout,entries:[{binding:0,resource:{buffer:heldSlots}},{binding:1,resource:{buffer:slotStats}},...(heldState?[{binding:2,resource:{buffer:heldState}}]:[])]});
  const particleModule=raw.createShaderModule({code:`
struct Particle{position:vec4f,velocity:vec4f,before:vec4u}
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<storage,read_write> expected:array<vec4u>;
@group(0) @binding(2) var<storage,read_write> stats:array<atomic<u32>>;
override capture:bool=false;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&expected)){return;}let value=particles[i].before;
 if(capture){expected[i]=value;return;}for(var axis=0u;axis<4u;axis++){if(value[axis]!=expected[i][axis]){atomicAdd(&stats[0],1u);atomicMin(&stats[1],i);}}
}`});
  const particlePipelines=await Promise.all([true,false].map(capture=>raw.createComputePipelineAsync({layout:"auto",compute:{module:particleModule,entryPoint:"main",constants:{capture:+capture}}})));
  const particleGroups=particlePipelines.map(p=>raw.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:particles}},{binding:1,resource:{buffer:particleReference}},{binding:2,resource:{buffer:particleStats}}]}));
  // velocityScratch retains the old raw input for this frozen replay. No arm
  // publishes a materialized field there; only unit taps and scalar slots.
  e=raw.createCommandEncoder();frame.copyWhole(e,f.departure,f.velocityScratch);e.copyBufferToBuffer(particles,0,original,0,original.size);raw.queue.submit([e.finish()]);
  const jobs=(encoder:GPUCommandEncoder,pipelines:GPUComputePipeline[],group:GPUBindGroup,proof=false,grid=65535)=>{
   const pass=encoder.beginComputePass({label:proof?"Compare canonical old slots":"Retain canonical old values"});pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group);if(proof)pass.setBindGroup(2,proofGroup);
   if(ownership.capacity.fineTiles>0){pass.setPipeline(pipelines[0]!);pass.dispatchWorkgroups(Math.min(grid,ownership.capacity.fineTiles));}
   pass.setPipeline(pipelines[1]!);pass.dispatchWorkgroups(Math.min(grid,ownership.capacity.tiles));pass.end();
  };
  const prepare=(encoder:GPUCommandEncoder)=>{encoder.copyBufferToBuffer(original,0,particles,0,original.size);encoder.clearBuffer(companion);if(coarseAfter)encoder.clearBuffer(frame.stageScratch);frame.copyWhole(encoder,f.velocity,f.departure);};
  const encode=(encoder:GPUCommandEncoder,retained:boolean,activeSweeps=sweeps)=>{
   const retainedPath=retained||packing||coarseAfter;
   if(retainedPath){if(retained&&packing){jobs(encoder,packed!,gather[0].group,false,retainGrid);}else if(integrated){
    // Call the integrated encoder directly; bypass the fixture-only capture.
    const prototype=Object.getPrototypeOf(ext);prototype.retainPair.call(ext,encoder,gather);
   }else jobs(encoder,retain,gather[1].group);}else frame.copyWhole(encoder,f.velocityScratch,f.departure);
   for(let i=0;i<seedMaps.length;i++)seedMaps[i]!.set("seed",(retainedPath?seedVariants:seeds)[i]!);
   ext.pairedSeams.set("sweep",retained&&sweepMode!=="none"?sweepVariants[sweepMode]:originalSweep);
   ext.preparePair(encoder,current,activeSweeps);
   ext.snapshotPipelines=retainedPath?snapshotVariants:referenceSnapshot;
   if(retained&&snapshotCap){
    const pass=encoder.beginComputePass({label:"Uniform mixed extension slot snapshot"}),p=ext.snapshotPipelines;
    pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,old[activeSweeps%2].group);pass.setPipeline(uniformDetailPick(p.restrict));
    pass.dispatchWorkgroups(...ownership.capacity.tileDimensions.map((n:number)=>Math.ceil(n/4)) as [number,number,number]);ext.hierarchy.encode(pass,false);
    pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,old[activeSweeps%2].group);
    if(ownership.capacity.fineTiles>0){pass.setPipeline(uniformDetailPick(p.fine));pass.dispatchWorkgroups(Math.min(snapshotCap,ownership.capacity.fineTiles));}
    const count=Math.ceil(3*ownership.capacity.tiles/64);pass.setPipeline(uniformDetailPick(p.coarse));pass.dispatchWorkgroups(Math.min(65535,count),Math.ceil(count/65535));pass.end();
   }else ext.finishSnapshot(encoder,old,activeSweeps);
   flip.snapshot(encoder,snapshots[activeSweeps%2]);
   for(let i=0;i<seedMaps.length;i++)seedMaps[i]!.set("seed",productionSeeds[i]!);ext.snapshotPipelines=productionSnapshot;ext.pairedSeams.set("sweep",originalSweep);
  };
  const readStats=async(buffer:GPUBuffer,read:GPUBuffer)=>{const encoder=raw.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,read,0,16);raw.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=[...new Uint32Array(read.getMappedRange())];read.unmap();return {mismatches:values[0]!,firstMismatch:values[1]!,words:values[2]!};};
  for(const mode of coarseAfter?["early","direct","snapshot4096","snapshot8192"]:packedAfter?["copy","packed"]:[packing?"packed":"copy"]){
  snapshotCap=mode.startsWith("snapshot")?Number(mode.slice(8)):0;packing=mode==="packed";sweepMode=mode==="early"||mode==="direct"?mode:"none";const parity=[];
  for(const activeSweeps of [...new Set([sweeps,0,1,3])]){
   raw.queue.writeBuffer(particleStats,0,new Uint32Array([0,0xffffffff,0,0]));raw.queue.writeBuffer(slotStats,0,new Uint32Array([0,0xffffffff,0,0]));
   for(const retained of [false,true]){
    e=raw.createCommandEncoder();prepare(e);encode(e,retained,activeSweeps);
    if(!retained){e.copyBufferToBuffer(companion,slots[activeSweeps%2]!.offset??0,heldSlots,0,heldSlots.size);if(heldState)e.copyBufferToBuffer(frame.stageScratch,(activeSweeps%2)*stateBytes,heldState,0,stateBytes);}else jobs(e,compare,proofSlots[activeSweeps%2]!,true);
    const index=+retained,pass=e.beginComputePass({label:"Compare particle before"});pass.setPipeline(particlePipelines[index]!);pass.setBindGroup(0,particleGroups[index]!);pass.dispatchWorkgroups(Math.ceil(count/256));pass.end();raw.queue.submit([e.finish()]);
   }
   const result={sweeps:activeSweeps,slots:await readStats(slotStats,slotRead),particles:{...await readStats(particleStats,particleRead),words:count*4}};parity.push(result);console.log(JSON.stringify({phase:"parity",packing,sweepMode,snapshotCap,...result}));assert.equal(result.slots.mismatches,0);assert.equal(result.particles.mismatches,0);
  }
  if(process.argv.includes("--wait-for-gpu-quiet")){let seconds=0;for(;;){const others=execFileSync("ps",["-Ao","pid=,args="],{encoding:"utf8"}).split("\n").filter(line=>{const row=line.match(/^\s*(\d+)\s+(.*)$/);return row&&Number(row[1])!==process.pid&&/^(?:\S*\/)?node\s/.test(row[2]!)&&/tools\/(?:probe|benchmark)[^\s]*\.ts/.test(row[2]!);});if(!others.length)break;if(seconds++%60===0)console.log(JSON.stringify({phase:"waiting-before-timing",others}));await new Promise(resolve=>setTimeout(resolve,1000));}}
  const trials:{retained:boolean;ms:number;compute_ms:number;passes:Awaited<ReturnType<GPUPassProfile["finish"]>>}[]=[];
  for(let round=0;round<rounds+2;round++)for(const retained of round%2?[true,false]:[false,true]){
   e=raw.createCommandEncoder();prepare(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
   profile.start();e=device.createCommandEncoder();const start=e.beginComputePass({label:"Old retention interval start"});start.setPipeline(marker);start.dispatchWorkgroups(1);start.end();encode(e,retained);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();const passes=await profile.finish();if(round>=2)trials.push({retained,ms:passes.reduce((sum,p)=>sum+p.ms+p.idle_ms,0),compute_ms:passes.reduce((sum,p)=>sum+p.ms,0),passes});
  }
  const originalTimes=distribution(trials.filter(t=>!t.retained).map(t=>t.ms)),retainedTimes=distribution(trials.filter(t=>t.retained).map(t=>t.ms));
  const saving=distribution(Array.from({length:rounds},(_,r)=>{const pair=trials.slice(2*r,2*r+2);return pair.find(t=>!t.retained)!.ms-pair.find(t=>t.retained)!.ms;}));
  const snapshotTime=(trial:typeof trials[number])=>trial.passes.filter(p=>p.label.includes("slot snapshot")||p.label.includes("FLIP snapshot")).reduce((sum,p)=>sum+p.ms+p.idle_ms,0);
  const snapshotTiming={original:distribution(trials.filter(t=>!t.retained).map(snapshotTime)),retained:distribution(trials.filter(t=>t.retained).map(snapshotTime)),saving:distribution(Array.from({length:rounds},(_,r)=>{const pair=trials.slice(2*r,2*r+2);return snapshotTime(pair.find(t=>!t.retained)!)-snapshotTime(pair.find(t=>t.retained)!);} ))};
  const report={integrated,packing,sweepMode,snapshotCap,retainGrid,lifetime,date:new Date().toISOString(),sceneId:definition.id,values,dt,steps,rounds,sweeps,count,parity,distributions_ms:{original:originalTimes,retained:retainedTimes,saving},snapshot_ms:snapshotTiming,trials,errors};const reportPath=coarseAfter?out.replace(/\.json$/,`-${mode}.json`):packing&&packedAfter?out.replace(/\.json$/,"-packed.json"):out;mkdirSync(dirname(reportPath),{recursive:true});writeFileSync(reportPath,JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify(report));assert.deepEqual(errors,[]);
  }
 }catch(error){console.error(JSON.stringify({phase:"failure",errors}));throw error;}finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await run();
