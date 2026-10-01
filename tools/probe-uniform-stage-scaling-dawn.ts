/** Profile Uniform Geometric ownership scaling, using Figure 7 at 256³.
 * --scene=cm12-figure-7-256 --case=adaptive|coarse|mixed|fine --frames=40
 * --coarsening=dynamic|regions --out=/tmp/stage-scaling.json
 * --sharpening-baseline restores the previous boundary sweep scheduler for A/B.
 * --surface-tolerance=0.125 probes smooth surface coarsening (h error units).
 * --quality-every=12 captures canonical quality and diagnostic slices after timing.
 * --pressure-reserve=1 adds one slot to the lagged plan instead of the full envelope.
 * --warmup=1200 times late settling in throughput mode.
 * --coarse-extension skips mixed front sweeps and seeds the 4h hierarchy directly.
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
import {uniformGeometricCoarseSurfaceTolerance} from "../lib/methods/uniform/uniform-geometric-parameters";
import {UniformMixedExtension} from "../lib/methods/uniform/uniform-mixed-extension";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {UniformMixedFrameTrace} from "../lib/methods/uniform/uniform-mixed-frame";
import {readMixedTexture,readMixedTileWords} from "../tests/helpers/uniform-mixed-native-fields";
import {uniformQualityCensus} from "./uniform-quality-census";

import {redistanceEveryStepReference} from "./uniform-redistance-reference";

const fingerprint=()=>{
 const files=[...readdirSync("lib/methods/uniform",{recursive:true}).map(String).filter(p=>p.endsWith(".ts")).map(p=>`lib/methods/uniform/${p}`),"lib/core/scenes.ts","lib/core/cm12-paper-scenes.ts"].sort();
 return createHash("sha256").update(files.map(path=>`${path}:${createHash("sha256").update(readFileSync(path)).digest("hex")}`).join("\n")).digest("hex");
};
const sourceFingerprint=fingerprint();
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const kind=arg("case","adaptive"), sceneId=arg("scene","cm12-figure-7-256"), frames=Number(arg("frames","40")), dt=1/Number(arg("hz","60"));
const sharpeningBaseline=process.argv.includes("--sharpening-baseline");
const coarseExtension=process.argv.includes("--coarse-extension");
const valueOverrides=JSON.parse(arg("values","{}"));
const surfaceTolerance=uniformGeometricCoarseSurfaceTolerance(Number(arg("surface-tolerance",String(valueOverrides.coarseningSurfaceTolerance??0.5)))), qualityEvery=Number(arg("quality-every","0"));
const pressureReserve=process.argv.some(a=>a.startsWith("--pressure-reserve="))?Number(arg("pressure-reserve","0")):undefined;
assert.ok(pressureReserve===undefined||(Number.isInteger(pressureReserve)&&pressureReserve>=0&&pressureReserve<=7));
assert.ok(pressureReserve===undefined||!process.argv.includes("--full-pressure-envelope"),"Choose a reserve or the full envelope");
assert.ok(Number.isInteger(qualityEvery)&&qualityEvery>=0);
const rebuildEveryStep=process.argv.includes("--rebuild-every-step"),inlineCurvature=process.argv.includes("--inline-curvature");
const warmup=Number(arg("warmup","8"));
assert.ok(Number.isInteger(warmup)&&warmup>=0);
const throughput=process.argv.includes("--throughput"), out=resolve(arg("out",`/tmp/${kind}.json`));
assert.ok(!throughput||qualityEvery===0,"Quality readbacks must not enter throughput measurements");
assert.ok(["adaptive","coarse","mixed","fine"].includes(kind));
assert.ok(Number.isInteger(frames)&&frames>8&&dt>0&&Number.isFinite(dt));
const stats=(values:number[])=>{const s=values.toSorted((a,b)=>a-b);return {n:s.length,mean:s.reduce((a,b)=>a+b,0)/s.length,median:s[Math.floor(s.length/2)]!,p10:s[Math.floor(s.length*.1)]!,p90:s[Math.min(s.length-1,Math.floor(s.length*.9))]!};};
const rows:{frame:number;wall_ms:number;trace:NonNullable<WebGPUUniformReferenceSolver["info"]["physicsTrace"]>;work:Record<string,unknown>}[]=[];
const qualitySnapshots:unknown[]=[];
let context:Record<string,unknown>={kind,sceneId,frames,dt_s:dt,throughput,rebuildEveryStep,inlineCurvature,warmup,sharpeningBaseline,coarseExtension,sourceFingerprint,surfaceTolerance,pressureReserve,qualitySnapshots};
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
 if(sharpeningBaseline){
  const {sharpeningReference}=await import("../tests/helpers/uniform-sharpening-reference");
  const create=raw.createShaderModule.bind(raw);
  Object.defineProperty(raw,"createShaderModule",{configurable:true,writable:true,value:(descriptor:GPUShaderModuleDescriptor)=>create({...descriptor,code:descriptor.code.includes("fn shSweepJobs(")?sharpeningReference(descriptor.code):descriptor.code})});
 }
 device=managedGPUDevice(raw,{requireWorkerRealm:false});
 if(rebuildEveryStep){const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=d=>create({...d,code:redistanceEveryStepReference(d.code)});
 }
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition(sceneId));
 const roster=initializeRigidBodies(scene.rigidBodies);
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 const halfX=scene.container.width_m/2,halfZ=scene.container.depth_m/2;
 const bounds={min_m:{x:-halfX,y:0,z:-halfZ},max_m:{x:halfX,y:scene.container.height_m,z:halfZ}};
 const region=(id:string,width:number,minX:number,maxX:number)=>({id,rule:"minimum-cell-size" as const,minimumCellSize_cells:width,maximumCellSize_cells:width,min_m:{...bounds.min_m,x:minX},max_m:{...bounds.max_m,x:maxX}});
 if(kind!=="adaptive")scene.fluid.refinementRegions=kind==="mixed"?[region("coarse-half",4,-halfX,0),region("fine-half",1,0,halfX)]:[region("whole-domain",kind==="coarse"?4:1,-halfX,halfX)];
 const coarsening=arg("coarsening","dynamic");assert.ok(["regions","dynamic"].includes(coarsening));
 const values=resolveMethodValues(uniformVolumeMethod,"balanced",{...valueOverrides,timeStep:"scene",coarsening,
  ...(process.argv.some(a=>a.startsWith("--surface-tolerance="))?{coarseningSurfaceTolerance:surfaceTolerance}:{})});
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
 const expectedOwners=kind==="adaptive"?undefined:kind==="fine"?capacity:kind==="mixed"?(capacity+capacity/64)/2:capacity/64;
 if(expectedOwners!==undefined)assert.equal(initial.uniformMixedOwners,expectedOwners,"Requested fixed ownership must be installed");
 const lattice={nx:initial.nx,ny:initial.ny,nz:initial.nz,cellSize_m:initial.cellSize_m};
 context={...context,scene,values,lattice,setup_ms,initial,adapter:adapterInfo};
 console.log(JSON.stringify({kind,setup_ms,lattice,owners:initial.uniformMixedOwners,bytes:initial.allocatedBytes}));
 // QA seams only: split the existing trace without adding simulation passes.
 const target=solver as unknown as {mixedFrameTrace:()=>UniformMixedFrameTrace|undefined;mixedFrame:Record<string,any>;lastPhysicsTraceAt_ms:number};
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
   const volume=await readMixedTexture(device,fields.volume),phi=await readMixedTexture(device,fields.phi);
   const tiles=await readMixedTileWords(device,solver);
   const quality=uniformQualityCensus([lattice.nx,lattice.ny,lattice.nz],tiles,volume,phi);
   assert.equal(quality.nonfinite,0,"Canonical state must stay finite");
   qualitySnapshots.push({frame,time_s:frame*dt,...quality});
  }
  if(frame%10===0){console.log(JSON.stringify({kind,frame,wall_ms,gpu_ms:info.physicsTrace.total_ms}));save({...context,rows});}
 }
 const final={...await solver.readStats()};assert.equal(final.simulationPipelineError,undefined);assert.deepEqual(errors,[]);
 if(expectedOwners!==undefined)assert.equal(final.uniformMixedOwners,expectedOwners,"Enforcement regions must retain the same owner count");
 const finalWork=await readWork();
 const selected=rows.filter(r=>r.frame>8),labels=[...new Set(selected.flatMap(r=>r.trace.phases.map(p=>p.label)))];
 const summary=throughput?{msPerStep:elapsed_ms/frames,elapsed_ms}: {wall_ms:stats(selected.map(r=>r.wall_ms)),gpu_ms:stats(selected.map(r=>r.trace.total_ms)),stages:labels.map(label=>({label,...stats(selected.map(r=>r.trace.phases.filter(p=>p.label===label).reduce((s,p)=>s+p.duration_ms,0)))})).sort((a,b)=>b.mean-a.mean)};
 const report={capturedAt:new Date().toISOString(),sourceFingerprint,sourceFingerprintAfter:fingerprint(),kind,sceneId,adapter:adapterInfo,backend:"Dawn/Metal",method:uniformVolumeMethod.id,dt_s:dt,frames,discardFrames:throughput?warmup:8,throughput,rebuildEveryStep,inlineCurvature,warmup,sharpeningBaseline,scope:"Simulation only; rendering excluded. Throughput uses two frames in flight without timestamps or per-frame stats. Trace mode fences each frame; GPU stage seams exclude CPU waits. Warmup frames are reported in discardFrames.",scene,values,lattice,setup_ms,initial,final,finalWork,dustMass,summary,rows,validationErrors:errors};
 save({...report,surfaceTolerance,pressureReserve:pressureReserve??(coarsening==="dynamic"&&surfaceTolerance>0?1:0),coarseExtension,qualitySnapshots,fullPressureEnvelope:process.argv.includes("--full-pressure-envelope")});console.log(JSON.stringify({out,summary}));
}catch(error){save({...context,sourceFingerprintAfter:fingerprint(),rows,failure:error instanceof Error?error.message:String(error),final:solver?.info});throw error;
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
