/** Profile Uniform Geometric ownership scaling, using Figure 7 at 256³.
 * --scene=cm12-figure-7-256 --case=adaptive|coarse|mixed|fine --frames=40
 * --coarsening=dynamic|regions --out=/tmp/stage-scaling.json
 * --sharpening-baseline restores the previous boundary sweep scheduler for A/B.
 * --surface-tolerance=0.125 probes smooth surface coarsening (h error units).
 * --quality-every=12 captures canonical quality and diagnostic slices after timing.
 * --pressure-reserve=1 adds one slot to the lagged plan instead of the full envelope.
 * --warmup=1200 times late settling in throughput mode.
 * --coarse-extension skips mixed front sweeps and seeds the 4h hierarchy directly.
 * --case=focus --focus-box=0.75,0,0,1,0.5,1 uses a 4h background and an h box.
 * --focus-travel=0.5 moves that box by half the domain width over the run.
 * --coarse-cadence selects the existing travel/cached-curvature mode (coupled).
 * --detail-census --quality-every=30 measures frozen h/2h/4h shape error.
 * --band-target=5 exercises the existing fine-pressure residual exit.
 * --elide-empty-band omits fine-pressure launches in a fixed all-4h run.
 * --atlas=table|affine|dense --atlas-edge=32 tests full-occupancy field placement.
 * --field-hashes checks complete canonical volume/velocity/phi and ownership.
 * These experiments do not alter production policy or acceptance thresholds.
 * Run serially under the WebGPU lease. Adds timestamp boundaries around
 * existing encodes, never simulation passes. Mixed pins x<0 at 4h and x>=0
 * at h. All cases use the same scene, timestep and numerical parameters.
 */
import assert from "node:assert/strict";
import {writeFileSync, mkdirSync, readFileSync, readdirSync} from "node:fs";
import {createHash} from "node:crypto";
import {dirname, resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {managedGPUDevice,gpuCompilationManagerFor} from "../lib/core/gpu-compilation-manager";
import {GPUStageTimestampRecorder} from "../lib/core/performance-trace";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {resolveMethodValues} from "../lib/core/method-contract";
import {initializeRigidBodies} from "../lib/core/rigid-body";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import {createProcessRetainedDawnGPU, type NodeDawnProvider} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import {UniformMixedExtension} from "../lib/methods/uniform/uniform-mixed-extension";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {UniformMixedFrameTrace} from "../lib/methods/uniform/uniform-mixed-frame";
import {readMixedTexture,readMixedTileWords} from "../tests/helpers/uniform-mixed-native-fields";
import {uniformQualityCensus} from "./uniform-quality-census";
import {uniformDetailCensus} from "./uniform-detail-census";
import {UniformAtlasAddressExperiment,type AtlasExperimentMode} from "./uniform-atlas-address-experiment";
import {refinementRegionLattice} from "../lib/core/refinement-regions";

import {redistanceEveryStepReference} from "./uniform-redistance-reference";

const fingerprint=()=>{
 const files=[...readdirSync("lib/methods/uniform",{recursive:true}).map(String).filter(p=>p.endsWith(".ts")).map(p=>`lib/methods/uniform/${p}`),"lib/core/scenes.ts","lib/core/cm12-paper-scenes.ts"].sort();
 return createHash("sha256").update(files.map(path=>`${path}:${createHash("sha256").update(readFileSync(path)).digest("hex")}`).join("\n")).digest("hex");
};
const sourceFingerprint=fingerprint();
const experimentFingerprint=()=>createHash("sha256").update(["tools/probe-uniform-stage-scaling-dawn.ts","tools/uniform-atlas-address-experiment.ts"].map(p=>readFileSync(p)).reduce((a,b)=>Buffer.concat([a,b]),Buffer.alloc(0))).digest("hex");
const experimentFingerprintBefore=experimentFingerprint();
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const kind=arg("case","adaptive"), sceneId=arg("scene","cm12-figure-7-256"), frames=Number(arg("frames","40")), dt=1/Number(arg("hz","60"));
const sharpeningBaseline=process.argv.includes("--sharpening-baseline");
const coarseExtension=process.argv.includes("--coarse-extension");
const valueOverrides=JSON.parse(arg("values","{}"));
const surfaceTolerance=Math.min(2,Math.max(0,Number(arg("surface-tolerance",String(valueOverrides.detailShapeTolerance??0)))||0)), qualityEvery=Number(arg("quality-every","0"));
const detailCensus=process.argv.includes("--detail-census");
assert.ok(!detailCensus||qualityEvery>0,"Detail census needs --quality-every");
const bandTarget=process.argv.some(a=>a.startsWith("--band-target="))?Number(arg("band-target","0")):undefined;
assert.ok(bandTarget===undefined||(Number.isFinite(bandTarget)&&bandTarget>=0));
const focusBox=arg("focus-box","0.75,0,0,1,0.5,1").split(",").map(Number),focusTravel=Number(arg("focus-travel","0"));
const coarseCadence=process.argv.includes("--coarse-cadence");
const elideEmptyBand=process.argv.includes("--elide-empty-band");
const atlasMode=arg("atlas","native"),atlasEdge=Number(arg("atlas-edge","32"));
assert.ok(["native","dense","table","affine"].includes(atlasMode)&&[16,32].includes(atlasEdge));
const fieldHashes=process.argv.includes("--field-hashes");
const fieldDump=arg("field-dump-dir","");
assert.ok(!fieldDump||fieldHashes,"Field dumps require --field-hashes");
let atlas:UniformAtlasAddressExperiment|undefined,atlasResources:{destroy:()=>void}|undefined;
assert.ok(focusBox.length===6&&focusBox.every(x=>Number.isFinite(x)&&x>=0&&x<=1)&&[0,1,2].every(a=>focusBox[a]!<focusBox[a+3]!)&&Number.isFinite(focusTravel));
const pressureReserve=process.argv.some(a=>a.startsWith("--pressure-reserve="))?Number(arg("pressure-reserve","0")):undefined;
assert.ok(pressureReserve===undefined||(Number.isInteger(pressureReserve)&&pressureReserve>=0&&pressureReserve<=7));
assert.ok(pressureReserve===undefined||!process.argv.includes("--full-pressure-envelope"),"Choose a reserve or the full envelope");
assert.ok(Number.isInteger(qualityEvery)&&qualityEvery>=0);
const rebuildEveryStep=process.argv.includes("--rebuild-every-step"),inlineCurvature=process.argv.includes("--inline-curvature");
const warmup=Number(arg("warmup","8"));
assert.ok(Number.isInteger(warmup)&&warmup>=0);
const throughput=process.argv.includes("--throughput"), out=resolve(arg("out",`/tmp/${kind}.json`));
assert.ok(!throughput||qualityEvery===0,"Quality readbacks must not enter throughput measurements");
assert.ok(["adaptive","coarse","mixed","fine","focus"].includes(kind));
assert.ok(Number.isInteger(frames)&&frames>8&&dt>0&&Number.isFinite(dt));
const stats=(values:number[])=>{const s=values.toSorted((a,b)=>a-b);return {n:s.length,mean:s.reduce((a,b)=>a+b,0)/s.length,median:s[Math.floor(s.length/2)]!,p10:s[Math.floor(s.length*.1)]!,p90:s[Math.min(s.length-1,Math.floor(s.length*.9))]!};};
const rows:{frame:number;wall_ms:number;trace:NonNullable<WebGPUUniformReferenceSolver["info"]["physicsTrace"]>;work:Record<string,unknown>}[]=[];
const qualitySnapshots:unknown[]=[];
let context:Record<string,unknown>={kind,sceneId,frames,dt_s:dt,throughput,rebuildEveryStep,inlineCurvature,warmup,sharpeningBaseline,coarseExtension,sourceFingerprint,surfaceTolerance,pressureReserve,qualitySnapshots,detailCensus,bandTarget,focusBox,focusTravel,coarseCadence,elideEmptyBand,experimentFingerprintBefore};
const save=(value:unknown)=>{mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(value,null,2)+"\n");};
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const leaseDeadline=Date.now()+300_000;let waiting=false;
for(;;){
 try{await acquireWebGPUExclusiveLock("dawn-benchmark",`Uniform stage scaling: ${sceneId}/${kind}`);break;}
 catch(error){
  if(!(error instanceof Error)||!(error.cause instanceof Error)||!("code" in error.cause)||error.cause.code!=="EEXIST"||Date.now()>leaseDeadline)throw error;
  if(!waiting){console.log("Waiting for the repository WebGPU lease");waiting=true;}
  await new Promise(resolve=>setTimeout(resolve,250));
 }
}
try {
 const dawn=await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE??"node_modules/webgpu/index.js")).href) as NodeDawnProvider;
 Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
 const adapterInfo={vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description};
 const raw=await adapter.requestDevice({requiredFeatures:throughput?[]:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 // The one-tile-per-workgroup seam sweep this arm rewrote production into is
 // retired with its text-rewriting helper: tests/helpers/uniform-sharpening-reference.ts
 // is now a dense reference (UniformSharpeningReference), not a timing baseline.
 assert.ok(!sharpeningBaseline,"--sharpening-baseline is retired: the seam sweep it restored no longer exists");
 device=managedGPUDevice(raw,{requireWorkerRealm:false});
 if(rebuildEveryStep){const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=d=>create({...d,code:redistanceEveryStepReference(d.code)});
 }
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition(sceneId));
 if(atlasMode!=="native"){
  atlas=new UniformAtlasAddressExperiment(refinementRegionLattice(scene).dimensions,atlasEdge as 16|32,atlasMode as AtlasExperimentMode);
  atlasResources=atlas.install(raw);
 }
 context={...context,atlasMode,atlasEdge};
 const inflow=arg("inflow","scene");assert.ok(["scene","on","off"].includes(inflow));
 if(inflow!=="scene"){assert.ok(scene.fluid.inflow,"Scene has no inflow to override");scene.fluid.inflow.enabled=inflow==="on";}
 const roster=initializeRigidBodies(scene.rigidBodies);
 if(kind==="focus")assert.ok(!scene.fluid.inflow?.enabled&&roster.every(b=>b.description.motion==="static"),
  "This selection prototype has not implemented moving-body or inflow priority requests");
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 const halfX=scene.container.width_m/2,halfZ=scene.container.depth_m/2;
 const bounds={min_m:{x:-halfX,y:0,z:-halfZ},max_m:{x:halfX,y:scene.container.height_m,z:halfZ}};
 const region=(id:string,width:number,minX:number,maxX:number)=>({id,rule:"minimum-cell-size" as const,minimumCellSize_cells:width,maximumCellSize_cells:width,min_m:{...bounds.min_m,x:minX},max_m:{...bounds.max_m,x:maxX}});
 if(kind!=="adaptive")scene.fluid.refinementRegions=kind==="mixed"?[region("coarse-half",4,-halfX,0),region("fine-half",1,0,halfX)]:[region("whole-domain",kind==="coarse"||kind==="focus"?4:1,-halfX,halfX)];
 const coarsening=arg("coarsening","dynamic");assert.ok(["regions","dynamic"].includes(coarsening));
 assert.ok(kind!=="focus"||coarsening==="dynamic","Focus probe needs the GPU relayout path");
 const values=resolveMethodValues(uniformVolumeMethod,"balanced",{...valueOverrides,timeStep:"scene",coarsening,
  ...(process.argv.some(a=>a.startsWith("--surface-tolerance="))?{detailShapeTolerance:surfaceTolerance}:{})});
 if(coarseExtension){
  const initialize=UniformMixedExtension.prototype.initialize;
  UniformMixedExtension.prototype.initialize=function(){
   // Probe-only selection of the existing coarse-source mode, before shader compilation.
   (this as unknown as {regularBulk:boolean}).regularBulk=true;
   return initialize.call(this);
  };
 }
 usePerformanceInstrumentationStore.getState().setEnabled(false);
 const start=performance.now();
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 await device.queue.onSubmittedWorkDone();
 const setup_ms=performance.now()-start,initial={...await solver.readStats()};
 const capacity=initial.nx*initial.ny*initial.nz;
 if(kind==="focus")assert.ok(initial.uniformMixedCoarseTiles!>0,"Focus experiment must start on a 4h background");
 const expectedOwners=kind==="adaptive"||kind==="focus"?undefined:kind==="fine"?capacity:kind==="mixed"?(capacity+capacity/64)/2:capacity/64;
 if(expectedOwners!==undefined)assert.equal(initial.uniformMixedOwners,expectedOwners,"Requested fixed ownership must be installed");
 const lattice={nx:initial.nx,ny:initial.ny,nz:initial.nz,cellSize_m:initial.cellSize_m};
 if(atlas)assert.deepEqual(atlas.dimensions,[lattice.nx,lattice.ny,lattice.nz]);
 const readCanonical=async(texture:GPUTexture)=>{
  const data=await readMixedTexture(device!,texture);
  return atlas?atlas.reorder(data,[texture.width,texture.height,texture.depthOrArrayLayers],texture.format==="rgba32float"?4:1):data;
 };
 const hashFields=async(dump=false)=>{
  const f=(solver as unknown as {mixedFrame:{fields:Record<string,GPUTexture>}}).mixedFrame.fields;
  const result:Record<string,string>={};
  if(dump&&fieldDump)mkdirSync(fieldDump,{recursive:true});
  for(const name of ["volume","velocity","phi"]){const data=await readCanonical(f[name]!),bytes=new Uint8Array(data.buffer,data.byteOffset,data.byteLength);result[name]=createHash("sha256").update(bytes).digest("hex");if(dump&&fieldDump)writeFileSync(resolve(fieldDump,`${name}.f32`),bytes);}
  const tiles=await readMixedTileWords(device!,solver);result.tiles=createHash("sha256").update(new Uint8Array(tiles.buffer)).digest("hex");
  if(dump&&fieldDump)writeFileSync(resolve(fieldDump,"tiles.u32"),new Uint8Array(tiles.buffer));
  return result;
 };
 context={...context,scene,values,lattice,setup_ms,initial,adapter:adapterInfo};
 console.log(JSON.stringify({kind,setup_ms,lattice,owners:initial.uniformMixedOwners,bytes:initial.allocatedBytes}));
 // QA seams only: split the existing trace without adding simulation passes.
 const target=solver as unknown as {mixedFrameTrace:()=>UniformMixedFrameTrace|undefined;mixedFrame:Record<string,any>;lastPhysicsTraceAt_ms:number};
 if(elideEmptyBand){
  assert.ok(kind==="coarse"&&coarsening==="regions"&&initial.uniformMixedFineTiles===0&&roster.length===0,
   "Empty-band elision is only certified for fixed all-4h ownership without bodies");
  const band=target.mixedFrame.band;
  band.encodePrepare=(encoder:GPUCommandEncoder)=>{encoder.clearBuffer(band.index,0,26*4);encoder.clearBuffer(band.index,band.slotMapOffset);};
  band.encodeSolve=()=>{};
 }
 if(bandTarget!==undefined){
  // Diagnostic only: exercise the band's existing residual exit. The root
  // pressure tolerance and all acceptance checks are unchanged.
  const frame=target.mixedFrame,write=frame.write.bind(frame);
  frame.write=(p:unknown)=>{write(p);device!.queue.writeBuffer(frame.bandParams,32,new Float32Array([bandTarget]));};
 }
 if(coarseCadence){
  const frame=target.mixedFrame,advance=frame.advance.bind(frame);
  frame.advance=(p:Record<string,unknown>,...args:unknown[])=>advance({...p,coarseSurfaceTravel:true},...args);
 }
 if(kind==="focus"){
  // Deliberately isolated selection prototype: 4h is authoritative outside
  // the box; h owns ALL fields inside. Existing conservative remap and local
  // pressure coupling are exercised. This is not a separate fine overlay.
  // Whole-domain coarse region suppresses automatic surface refinement;
  // the production builder's solid exclusions remain in force.
  const host=solver as unknown as Record<string,any>,builder=host.mixedBuilder,dynamic=host.mixedDynamic;
  const setStatic=builder.setStatic.bind(builder),encode=dynamic.encode.bind(dynamic);
  let fine:Uint8Array|undefined,coarse:Uint8Array|undefined,step=0;
  builder.setStatic=(f:Uint8Array,c?:Uint8Array)=>{fine=f.slice();coarse=c?.slice();setStatic(f,c);};
  host.mixedBuilderStaticKey=undefined;
  dynamic.encode=(encoder:GPUCommandEncoder,...args:unknown[])=>{
   assert.ok(fine&&coarse,"Production region/solid masks must arrive before patch selection");
   const selected=fine.slice(),[tx,ty,tz]=[lattice.nx/4,lattice.ny/4,lattice.nz/4];
   const shift=focusTravel*Math.min(1,step++/Math.max(1,(throughput?warmup:0)+frames-1));
   for(let t=0;t<selected.length;t++){
    const p=[(t%tx+.5)/tx,(Math.floor(t/tx)%ty+.5)/ty,(Math.floor(t/(tx*ty))+.5)/tz];
    if(p.every((q,a)=>q>=focusBox[a]!+(a===0?shift:0)&&q<focusBox[a+3]!+(a===0?shift:0)))selected[t]=1;
   }
   // Selection must precede the residency census, including entirely dry
   // requested patches. Selecting only at the builder is too late: the
   // next frame correctly rejects fine tiles in absent residency pages.
   setStatic(selected,coarse);dynamic.join(selected);return encode(encoder,...args);
  };
 }
 const workDevice=device;
 const readWork=async()=>{const finalWork:Record<string,number[]>={};
 // Diagnostic copies outside the measured GPU and wall intervals.
 for(const [name,buffer,words] of [
  ["sharpening",target.mixedFrame.sharpen.work.list,8],
  ["transport",target.mixedFrame.transport.live,20],
  ["pressureBand",target.mixedFrame.band.index,26],
 ] as [string,GPUBuffer,number][]){
  const read=workDevice.createBuffer({size:4*words,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const e=workDevice.createCommandEncoder();e.copyBufferToBuffer(buffer,0,read,0,4*words);workDevice.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);finalWork[name]=Array.from(new Uint32Array(read.getMappedRange()));}
  finally{if(read.mapState==="mapped")read.unmap();read.destroy();}
 }
 return finalWork;};
 if(inlineCurvature){
  const force=target.mixedFrame.forces,encode=force.encode.bind(force);
  force.encode=(e:GPUCommandEncoder,g:GPUBindGroup,capillarity:boolean)=>encode(e,g,capillarity,false);
 }
 if(pressureReserve!==undefined){
  const frame=target.mixedFrame,advance=frame.advance.bind(frame);
  frame.advance=(p:Record<string,unknown>,...args:unknown[])=>advance({...p,fullPressureEnvelope:false,pressureReserve},...args);
 }
 if(process.argv.includes("--full-pressure-envelope")){
  const frame=target.mixedFrame,advance=frame.advance.bind(frame);
  frame.advance=(p:Record<string,unknown>,...args:unknown[])=>advance({...p,fullPressureEnvelope:true},...args);
  context.fullPressureEnvelope=true;
 }
 if(!throughput){
  await GPUStageTimestampRecorder.prepare(device);usePerformanceInstrumentationStore.getState().setMode("timeline");
  let active:UniformMixedFrameTrace|undefined;const original=target.mixedFrameTrace.bind(target);
  target.mixedFrameTrace=()=>{active=original();return active;};
  for(const [name,method,label] of [
   ["phiResolve","encode","Resolve hanging phi"],["surfaceVolume","encode","Global surface volume correction"],
   ["geometry","encode","Surface geometry"],["momentum","encode","Momentum transport"],["forces","encode","Forces"],
   ["plan","encode","Support classification"],["plan","encodeCertificate","Transport reach certificate"],
   ["extension","encode","Velocity extension / hierarchy"],["cache","encode","Velocity cache"],["hanging","encode","Hanging velocity"],
   ["transport","encodeCopy","Volume copy"],["transport","encodeTransport","Volume transport"],
   ["cleanup","encode","Volume cleanup"],["sharpen","encodeGeometry","Sharpening geometry"],["sharpen","encodeSweeps","Sharpening sweeps"],
   ["surfaceBand","encode","Pressure surface band"],["authority","encode","Simulation pressure phase"],
   ["split.transfer","encodeToPressure","Transfer to coarse pressure"],["split.transfer","encodeToSimulation","Transfer to simulation"],
   ["band","encodePrepare","Fine pressure preparation"],["band","encodeSolve","Fine pressure solve"],
  ]){
   const object=name!.split(".").reduce((object,key)=>object[key],target.mixedFrame),encode=object[method!];
   object[method!]=function(...args:any[]){const r=encode.apply(this,args);active?.phase(args[0],{id:"other",label:label!});return r;};
  }
  const host=solver as unknown as Record<string,any>;
  for(const [object,method,label] of [
   [host.mixedDynamic,"encode","Resolution classifier"],
   [host.mixedBuilder,"encode","Ownership builder"],
   [target.mixedFrame.remap,"applyGpu","Ownership remap and adoption"],
  ] as [Record<string,any>,string,string][]){
   const encode=object[method];object[method]=function(...args:any[]){const r=encode.apply(this,args);active?.phase(args[0],{id:"other",label});return r;};
  }
  const surface=target.mixedFrame.surface,encode=surface.encode;
  surface.encode=function(...args:any[]){const r=encode.apply(this,args);active?.phase(args[0],{id:"other",label:`Surface ${args[1]}`});return r;};
 }
 let elapsed_ms=0,lastSample=-1,dustMass=0;
 if(throughput){
  const advance=async(first:number,count:number)=>{for(let i=first;i<first+count;i+=2){for(let j=i;j<Math.min(i+2,first+count);j++)assert.ok(solver!.advanceTo(j*dt,roster));await solver!.awaitFrameCompletion();}await device!.queue.onSubmittedWorkDone();};
  await advance(1,warmup);const start=performance.now();await advance(warmup+1,frames);elapsed_ms=performance.now()-start;
 }else for(let frame=1;frame<=frames;frame++){
  target.lastPhysicsTraceAt_ms=-Infinity;
  const start=performance.now();assert.ok(solver.advanceTo(frame*dt,roster));await solver.awaitFrameCompletion();await device.queue.onSubmittedWorkDone();const wall_ms=performance.now()-start;
  let info:WebGPUUniformReferenceSolver["info"]=await solver.readStats();
  for(let n=0;n<100&&(!info.physicsTrace||info.physicsTrace.sampleId===lastSample);n++){await new Promise(r=>setTimeout(r,5));info=await solver.readStats();}
  assert.ok(info.physicsTrace&&info.physicsTrace.sampleId!==lastSample,`No fresh trace: frame ${frame}`);
  assert.equal(info.physicsTrace.measurementSource,"gpu-hardware-timestamp");lastSample=info.physicsTrace.sampleId;
  assert.ok(Math.abs(info.lastDt_s!-dt)<1e-9);assert.equal(info.simulationPipelineError,undefined);
  if(expectedOwners!==undefined)assert.equal(info.uniformMixedOwners,expectedOwners,"Enforcement regions must retain the same owner count");
  dustMass+=info.uniformVolumeDustMass_cells??0;
  const work=Object.fromEntries(Object.entries(info).filter(([key])=>/^(allocatedBytes|lastDt|encodedSteps|uniformMixed|uniformPressure|uniformVolumeDust|maxSpeed|volumeCellSum|volumeDrift)/.test(key)));
  if(process.argv.includes("--work-census"))work.stageLists=await readWork();
  rows.push({frame,wall_ms,trace:info.physicsTrace,work});
  if(qualityEvery>0&&(frame===1||frame%qualityEvery===0||frame===frames)){
   const fields=target.mixedFrame.fields;
   const volume=await readCanonical(fields.volume),phi=await readCanonical(fields.phi);
   const tiles=await readMixedTileWords(device,solver);
   const quality=uniformQualityCensus([lattice.nx,lattice.ny,lattice.nz],tiles,volume,phi);
   assert.equal(quality.nonfinite,0,"Canonical state must stay finite");
   qualitySnapshots.push({frame,time_s:frame*dt,...quality,...(fieldHashes?{fieldHashes:await hashFields()}:{}),...(detailCensus?{detail:uniformDetailCensus([lattice.nx,lattice.ny,lattice.nz],tiles,volume,phi,lattice.cellSize_m)}:{})});
  }
  if(frame%10===0){console.log(JSON.stringify({kind,frame,wall_ms,gpu_ms:info.physicsTrace.total_ms}));save({...context,rows});}
 }
 const final={...await solver.readStats()};assert.equal(final.simulationPipelineError,undefined);assert.deepEqual(errors,[]);
 if(expectedOwners!==undefined)assert.equal(final.uniformMixedOwners,expectedOwners,"Enforcement regions must retain the same owner count");
 const finalWork=await readWork();
 const selected=rows.filter(r=>r.frame>8),labels=[...new Set(selected.flatMap(r=>r.trace.phases.map(p=>p.label)))];
 const summary=throughput?{msPerStep:elapsed_ms/frames,elapsed_ms}: {wall_ms:stats(selected.map(r=>r.wall_ms)),gpu_ms:stats(selected.map(r=>r.trace.total_ms)),stages:labels.map(label=>({label,...stats(selected.map(r=>r.trace.phases.filter(p=>p.label===label).reduce((s,p)=>s+p.duration_ms,0)))})).sort((a,b)=>b.mean-a.mean)};
 const report={capturedAt:new Date().toISOString(),sourceFingerprint,sourceFingerprintAfter:fingerprint(),kind,sceneId,adapter:adapterInfo,backend:"Dawn/Metal",method:uniformVolumeMethod.id,dt_s:dt,frames,discardFrames:throughput?warmup:8,throughput,rebuildEveryStep,inlineCurvature,warmup,sharpeningBaseline,scope:"Simulation only; rendering excluded. Throughput uses two frames in flight without timestamps or per-frame stats. Trace mode fences each frame; GPU stage seams exclude CPU waits. Warmup frames are reported in discardFrames.",scene,values,lattice,setup_ms,initial,final,finalWork,dustMass,summary,rows,validationErrors:errors};
 save({...report,experimentFingerprintBefore,experimentFingerprintAfter:experimentFingerprint(),surfaceTolerance,pressureReserve:pressureReserve??(coarsening==="dynamic"&&surfaceTolerance>0?1:0),coarseExtension,qualitySnapshots,detailCensus,bandTarget,focusBox,focusTravel,coarseCadence,elideEmptyBand,atlasMode,atlasEdge,atlasCounts:atlas?.counts,atlasMetadataBytes:atlasMode==="table"?atlas?.offsets.byteLength:0,...(fieldHashes?{fieldHashes:await hashFields(true)}:{}),fullPressureEnvelope:process.argv.includes("--full-pressure-envelope")});console.log(JSON.stringify({out,summary}));
}catch(error){save({...context,sourceFingerprintAfter:fingerprint(),rows,failure:error instanceof Error?error.message:String(error),final:solver?.info});throw error;
}finally{solver?.destroy();atlasResources?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
