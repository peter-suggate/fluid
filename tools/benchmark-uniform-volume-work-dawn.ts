/** Matched production advances and hardware stage timings; exclusive GPU. */
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
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
 for(const sceneId of (process.env.UNIFORM_BENCH_SCENE?[process.env.UNIFORM_BENCH_SCENE]:["hero-garden-hose","minimal-power-dam-break-64"])){
  const scene=sceneDocument(getSceneDefinition(sceneId));
  const arms:{mode:string;full:number[];stages:Record<string,number[]>;pages:number|undefined}[]=[];
  for(const mode of ["window","pages","pages","window"]){
   const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,
    {...uniformGeometricSolverOptions({volumeStorage:sceneId==="minimal-power-dam-break-64"?(mode==="window"?"dense":"auto"):"pages32",pressureWindow:"domain"},scene),volumePageWork:mode==="pages",pageDomain:mode==="pages",activeRegion:mode==="window",pressureWindow:false},()=>{});
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
    const arm={mode,full,stages,pages:solver.info.uniformVolumePagesActive,transportTiles:solver.info.uniformVolumeTransportWorkgroups,sharpenTiles:solver.info.uniformVolumeSharpenWorkgroups};arms.push(arm);console.log(JSON.stringify({sceneId,mode,full_ms:median(full),stages:Object.fromEntries(Object.entries(stages).map(([k,v])=>[k,median(v)])),pages:arm.pages}));
   }finally{solver.destroy();}
  }
  const measures=(mode:string,field:"full")=>median(arms.filter(a=>a.mode===mode).flatMap(a=>a[field]));
  const result={sceneId,scope:"queue-fenced simulation excluding rendering; matched fixed pressure cycles; ABBA",arms,window_ms:measures("window","full"),pages_ms:measures("pages","full"),throughputRatio:measures("window","full")/measures("pages","full")};
  results.push(result);console.log(JSON.stringify({...result,arms:undefined}));
 }
 assert.deepEqual(errors,[]);
 writeFileSync(process.env.UNIFORM_BENCH_OUTPUT??"docs/research/uniform-page-domain-2026-09-21.json",JSON.stringify(results,null,2)+"\n");
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
