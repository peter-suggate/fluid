/** Matched production advances and hardware stage timings; exclusive GPU. */
import {UniformTexturePages} from "../lib/methods/uniform/uniform-texture-pages";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
const median=(v:number[])=>{const s=[...v].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]!+s[Math.floor(s.length/2)]!)/2;};
await acquireWebGPUExclusiveLock("dawn-probe","Uniform page-first volume benchmark");
let device:GPUDevice|undefined;
const results:unknown[]=[];
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 usePerformanceInstrumentationStore.getState().setEnabled(true);
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 for(const sceneId of (process.env.UNIFORM_BENCH_SCENE?[process.env.UNIFORM_BENCH_SCENE]:["water-box-dam-break"])){
  const scene=sceneDocument(getSceneDefinition(sceneId));
  const arms:{mode:string;full:number[];stages:Record<string,number[]>;pages:number|undefined}[]=[];
  for(const mode of (process.env.ARMS??"production,root-dense,direct,one-cycle,production").split(",")){
   const compiler=gpuCompilationManagerFor(device);
   const originalModule=compiler.createShaderModule;
   compiler.createShaderModule=function(descriptor){
    if(mode==="tight-domain" && descriptor.label==="Publish accepted uniform page domain") {
     assert.equal(sceneId,"water-box-dam-break");
     descriptor={...descriptor,code:descriptor.code
      .replace('accepted[B]=n*(EDGE/4u);accepted[B+1u]=EDGE/4u;accepted[B+2u]=EDGE/4u;',
       'accepted[B]=n*((accepted[B+12u]+3u)/4u);accepted[B+1u]=(accepted[B+13u]+3u)/4u;accepted[B+2u]=(accepted[B+14u]+3u)/4u;')
      .replace('accepted[B+4u]=n*v;accepted[B+5u]=v;accepted[B+6u]=v;',
       'accepted[B+4u]=n*((accepted[B+12u]+4u)/4u);accepted[B+5u]=(accepted[B+13u]+4u)/4u;accepted[B+6u]=(accepted[B+14u]+4u)/4u;')};
    }
    return originalModule.call(this,descriptor);
   };
   const originalShader=UniformTexturePages.prototype.shader;
   UniformTexturePages.prototype.shader=function(source,fixed,audit){
    if(mode==="raw-dense")return source;
    let result=originalShader.call(this,source,fixed,mode.includes("no-audit")?false:audit);
    if(mode==="literal-loops")result=result.replace(/uniformFieldPages\[35\]\[([0-3])\]/g,(_m,i)=>`${[2,3,6,8][Number(i)]}u`);
    return result;
   };
   const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,
    {...uniformGeometricSolverOptions({},scene),
     ...(["root-dense","raw-dense"].includes(mode)?{fieldStorageForQA:"dense" as const}:{}),
     ...(mode==="direct"?{pressureCycleDispatch:"direct" as const}:{}),
     ...(mode==="one-cycle"?{pressureSchedule:{fullCycles:1,vCycles:0,preSweeps:6,postSweeps:6,residualTolerance:10}}:{}),
    },()=>{});
   UniformTexturePages.prototype.shader=originalShader;
   compiler.createShaderModule=originalModule;
   if(mode.includes("direct-domain")) {assert.equal(sceneId,"water-box-dam-break");(solver as any).pageDomainDispatch=undefined;}
   if(mode==="tight-pressure") {
    assert.equal(sceneId,"water-box-dam-break");
    const pressure=(solver as any).pressureMultigrid;
    const geometry=new Uint32Array((pressure.levels.length+1)*3);
    pressure.levels.forEach((l:any,i:number)=>geometry.set(l.dimensions.map((n:number)=>Math.ceil(n/4)),i*3));
    geometry.set([1,1,1],pressure.levels.length*3);
    device.queue.writeBuffer(pressure.cycleDispatch,0,geometry);
    for(const pass of pressure.plan)if(pass.workgroups.some((n:number)=>n>1))
     pass.workgroups=pressure.levels[pass.activeLevel].dimensions.map((n:number)=>Math.ceil(n/4));
   }
   const full:number[]=[],stages:Record<string,number[]>={};let priorSample=-1;
   try{
    for(let frame=1;frame<=24;frame++){
     const start=performance.now();(solver as any).lastPhysicsTraceAt_ms=-Infinity;assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();await device.queue.onSubmittedWorkDone();
     if(frame>4)full.push(performance.now()-start);
     await solver.readStats();
     const trace=solver.info.physicsTrace;
     if(frame>4&&trace?.measurementSource==="gpu-hardware-timestamp"&&trace.sampleId!==priorSample){
      for(const phase of trace.phases)(stages[phase.label]??=[]).push(phase.duration_ms);
      priorSample=trace.sampleId;
     }
    }
    const arm={mode,full,stages,allocatedBytes:solver.info.allocatedBytes,
     cyclesEncoded:solver.info.uniformPressureCyclesEncoded,cyclesExecuted:solver.info.uniformPressureCyclesExecuted,
     passesEncoded:solver.info.uniformPressurePassesEncoded,pages:solver.info.uniformVolumePagesActive,transportTiles:solver.info.uniformVolumeTransportWorkgroups,sharpenTiles:solver.info.uniformVolumeSharpenWorkgroups};arms.push(arm);console.log(JSON.stringify({sceneId,mode,full_ms:median(full),stages:Object.fromEntries(Object.entries(stages).map(([k,v])=>[k,median(v)])),pages:arm.pages,pressure:solver.info.pressureSolver}));
   }finally{solver.destroy();}
  }
  results.push({sceneId,scope:"Queue-fenced simulation, rendering excluded. Readbacks after timed interval. Ablations are diagnostic only.",arms});
 }
 assert.deepEqual(errors,[]);
 writeFileSync(process.env.UNIFORM_BENCH_OUTPUT??"/tmp/uniform-small-frame-cost.json",JSON.stringify(results,null,2)+"\n");
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
