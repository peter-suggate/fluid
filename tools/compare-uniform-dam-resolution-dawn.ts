/** Compare the 128³ dam held at 4h with the physical 32³ mini dam.
 * --case=high-regions|mini-regions|high-dynamic|mini-fine-dynamic
 * --frames=120 --hz=60 --throughput --out=/tmp/comparison.json
 * Without --throughput, records hardware stage timestamps, with finer seams.
 * --field-census reads canonical fields after timing. --full-pressure-envelope
 * is diagnostic only: retains the existing cap and gates, bypassing lagged plans.
 * The production solver, tuning and pressure acceptance are unchanged.
 */
import assert from "node:assert/strict";
import {writeFileSync, mkdirSync} from "node:fs";
import {dirname, resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {GPUStageTimestampRecorder} from "../lib/core/performance-trace";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {resolveMethodValues} from "../lib/core/method-contract";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import {createProcessRetainedDawnGPU, type NodeDawnProvider} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {UniformMixedFrameTrace} from "../lib/methods/uniform/uniform-mixed-frame";
import {readMixedTexture} from "../tests/helpers/uniform-mixed-native-fields";

const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const kind=arg("case","high-regions"), frames=Number(arg("frames","120")), dt=1/Number(arg("hz","60"));
const throughput=process.argv.includes("--throughput"), out=resolve(arg("out",`/tmp/${kind}.json`));
assert.ok(["high-regions","mini-regions","high-dynamic","mini-fine-dynamic"].includes(kind));
assert.ok(Number.isInteger(frames)&&frames>8&&dt>0&&Number.isFinite(dt));
const stats=(values:number[])=>{const s=values.toSorted((a,b)=>a-b);return {n:s.length,mean:s.reduce((a,b)=>a+b,0)/s.length,median:s[Math.floor(s.length/2)]!,p10:s[Math.floor(s.length*.1)]!,p90:s[Math.min(s.length-1,Math.floor(s.length*.9))]!};};
const rows:{frame:number;wall_ms:number;trace:NonNullable<WebGPUUniformReferenceSolver["info"]["physicsTrace"]>;work:Record<string,unknown>}[]=[];
let context:Record<string,unknown>={kind,frames,dt_s:dt,throughput};
const save=(value:unknown)=>{mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(value,null,2)+"\n");};
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
await acquireWebGPUExclusiveLock("dawn-benchmark",`Uniform dam resolution comparison: ${kind}`);
try {
 const dawn=await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE??"node_modules/webgpu/index.js")).href) as NodeDawnProvider;
 Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
 const adapterInfo={vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description};
 device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:throughput?[]:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition(kind.startsWith("high")?"high-resolution-dam-break":"minimal-power-dam-break-32"));
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 if(kind.startsWith("high"))scene.fluid.refinementRegions=[{id:"whole-domain-4h",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:{x:-.4,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4}}];
 if(kind==="mini-fine-dynamic")scene.fluid.refinementRegions=[{id:"whole-domain-h",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,min_m:{x:-.4,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4}}];
 const values=resolveMethodValues(uniformVolumeMethod,"balanced",{...JSON.parse(arg("values","{}")),timeStep:"scene",coarsening:kind.endsWith("dynamic")?"dynamic":"regions"});
 usePerformanceInstrumentationStore.getState().setEnabled(false);
 const start=performance.now();
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 await device.queue.onSubmittedWorkDone();
 const setup_ms=performance.now()-start,initial={...await solver.readStats()};
 assert.equal(initial.uniformMixedOwners,32**3,"Both comparison arms must have 32³ physical owners");
 const lattice={nx:initial.nx,ny:initial.ny,nz:initial.nz,cellSize_m:initial.cellSize_m};
 context={...context,scene,values,lattice,setup_ms,initial,adapter:adapterInfo};
 console.log(JSON.stringify({kind,setup_ms,lattice,owners:initial.uniformMixedOwners,bytes:initial.allocatedBytes}));
 // QA seams only: split the existing trace without adding simulation passes.
 const target=solver as unknown as {mixedFrameTrace:()=>UniformMixedFrameTrace|undefined;mixedFrame:Record<string,any>;lastPhysicsTraceAt_ms:number};
 if(process.argv.includes("--full-pressure-envelope")){
  const frame=target.mixedFrame,get=frame.lagged.get.bind(frame.lagged);
  frame.lagged.get=(key:number)=>get(key)?frame.initialPlan:undefined;
  context.fullPressureEnvelope=true;
 }
 if(!throughput){
  await GPUStageTimestampRecorder.prepare(device);usePerformanceInstrumentationStore.getState().setMode("timeline");
  let active:UniformMixedFrameTrace|undefined;const original=target.mixedFrameTrace.bind(target);
  target.mixedFrameTrace=()=>{active=original();return active;};
  for(const [name,label] of [["phiResolve","Resolve hanging phi"],["surfaceVolume","Global surface volume correction"],["geometry","Surface geometry"],["momentum","Momentum transport"],["forces","Forces"]]){
   const object=target.mixedFrame[name!],encode=object.encode;
   object.encode=function(...args:any[]){const r=encode.apply(this,args);active?.phase(args[0],{id:"other",label:label!});return r;};
  }
  const surface=target.mixedFrame.surface,encode=surface.encode;
  surface.encode=function(...args:any[]){const r=encode.apply(this,args);active?.phase(args[0],{id:"other",label:`Surface ${args[1]}`});return r;};
 }
 let elapsed_ms=0,lastSample=-1,dustMass=0;
 if(throughput){
  const advance=async(first:number,count:number)=>{for(let i=first;i<first+count;i+=2){for(let j=i;j<Math.min(i+2,first+count);j++)assert.ok(solver!.advanceTo(j*dt,[]));await solver!.awaitFrameCompletion();}await device!.queue.onSubmittedWorkDone();};
  await advance(1,8);const start=performance.now();await advance(9,frames);elapsed_ms=performance.now()-start;
 }else for(let frame=1;frame<=frames;frame++){
  target.lastPhysicsTraceAt_ms=-Infinity;
  const start=performance.now();assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();await device.queue.onSubmittedWorkDone();const wall_ms=performance.now()-start;
  let info:WebGPUUniformReferenceSolver["info"]=await solver.readStats();
  for(let n=0;n<100&&(!info.physicsTrace||info.physicsTrace.sampleId===lastSample);n++){await new Promise(r=>setTimeout(r,5));info=await solver.readStats();}
  assert.ok(info.physicsTrace&&info.physicsTrace.sampleId!==lastSample,`No fresh trace: frame ${frame}`);
  assert.equal(info.physicsTrace.measurementSource,"gpu-hardware-timestamp");lastSample=info.physicsTrace.sampleId;
  assert.ok(Math.abs(info.lastDt_s!-dt)<1e-9);assert.equal(info.simulationPipelineError,undefined);
  assert.equal(info.uniformMixedOwners,32**3,"Enforcement regions must retain the same owner count");
  dustMass+=info.uniformVolumeDustMass_cells??0;
  const work=Object.fromEntries(Object.entries(info).filter(([key])=>/^(allocatedBytes|lastDt|encodedSteps|uniformMixed|uniformPressure|uniformVolumeDust|maxSpeed|volumeCellSum|volumeDrift)/.test(key)));
  rows.push({frame,wall_ms,trace:info.physicsTrace,work});
  if(frame%30===0){console.log(JSON.stringify({kind,frame,wall_ms,gpu_ms:info.physicsTrace.total_ms}));save({...context,rows});}
 }
 const final={...await solver.readStats()};assert.equal(final.simulationPipelineError,undefined);assert.deepEqual(errors,[]);
 assert.equal(final.uniformMixedOwners,32**3,"Enforcement regions must retain the same owner count");
 let fieldCensus:Record<string,unknown>|undefined;
 if(process.argv.includes("--field-census")){
  // Compare canonical samples on the common physical 32³ lattice. Hanging
  // texels in the 128³ representation are not authoritative state.
  const fields=target.mixedFrame.fields,volume=await readMixedTexture(device,fields.volume),phi=await readMixedTexture(device,fields.phi);
  const w=kind.startsWith("high")?4:1,n=32*w,vn=n+1,h=.025;
  let mass=0,mx=0,my=0,mz=0,phiNegativeVertices=0,frontCell=-1;
  for(let z=0;z<32;z++)for(let y=0;y<32;y++)for(let x=0;x<32;x++){
   const v=volume[w*x+n*(w*y+n*w*z)]!;mass+=v;mx+=v*((x+.5)*h-.4);my+=v*(y+.5)*h;mz+=v*((z+.5)*h-.4);
  }
  for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++)if(phi[w*x+vn*(w*y+vn*w*z)]!<0){phiNegativeVertices++;if(z===16&&y>0)frontCell=Math.max(frontCell,x);}
  fieldCensus={physicalVolume_m3:mass*h**3,centerOfMass_m:{x:mx/mass,y:my/mass,z:mz/mass},phiNegativeVertices,centerPlaneFront_m:frontCell<0?null:frontCell*h-.4,note:"Post-timing canonical samples at identical physical spacing; front uses negative phi vertices on z=0 above the floor."};
 }
 const selected=rows.filter(r=>r.frame>8),labels=[...new Set(selected.flatMap(r=>r.trace.phases.map(p=>p.label)))];
 const summary=throughput?{msPerStep:elapsed_ms/frames,elapsed_ms}: {wall_ms:stats(selected.map(r=>r.wall_ms)),gpu_ms:stats(selected.map(r=>r.trace.total_ms)),stages:labels.map(label=>({label,...stats(selected.map(r=>r.trace.phases.filter(p=>p.label===label).reduce((s,p)=>s+p.duration_ms,0)))})).sort((a,b)=>b.mean-a.mean)};
 const report={capturedAt:new Date().toISOString(),kind,adapter:adapterInfo,backend:"Dawn/Metal",method:uniformVolumeMethod.id,dt_s:dt,frames,discardFrames:8,throughput,scope:"Simulation only; rendering excluded. Throughput uses two frames in flight without timestamps or per-frame stats. Trace mode fences each frame; GPU stage seams exclude CPU waits. First eight frames excluded.",scene,values,lattice,setup_ms,initial,final,dustMass,summary,rows,validationErrors:errors};
 save({...report,fieldCensus,fullPressureEnvelope:process.argv.includes("--full-pressure-envelope")});console.log(JSON.stringify({out,summary,fieldCensus}));
}catch(error){save({...context,rows,failure:error instanceof Error?error.message:String(error),final:solver?.info});throw error;
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
