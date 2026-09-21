/** Matched production advances and hardware stage timings; exclusive GPU. */

import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
// QA-only census after timing: read dispatch metadata without changing scheduling.
async function workCensus(solver: WebGPUUniformReferenceSolver) {
 const internal=solver as any, pressure=internal.pressureMultigrid;
 const device=internal.device as GPUDevice;
 const staging=device.createBuffer({size:1024,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(internal.activeRegion,0,staging,0,1024);
 device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
 const activeRegion=Array.from(new Uint32Array(staging.getMappedRange()));staging.unmap();staging.destroy();
 const plan:Record<string,{passes:number;workgroups:number}>= {};
 for(const pass of pressure.plan??[]) {
  const key=`${pass.stage}/L${pass.activeLevel}/${pass.entryPoint}/gate${pass.cycleGate}`;
  const bucket=plan[key]??={passes:0,workgroups:0};bucket.passes++;
  bucket.workgroups+=pass.workgroups.reduce((a:number,b:number)=>a*b,1);
 }
 const describe=(t:GPUTexture,dims?:readonly number[])=>({label:t.label,format:t.format,
  logical:dims,physical:[t.width,t.height,t.depthOrArrayLayers]});
 return {activeRegion,pressureWindowLevelGroups:pressure.windowLevelGroups,dimensions:[solver.info.nx,solver.info.ny,solver.info.nz],
  pageDomain:internal.pageDomain?{edge:internal.pageDomain.edge,count:internal.pageDomain.count,capacity:internal.pageDomain.capacity}:null,
  pressureLevels:pressure.levels.map((l:any)=>describe(l.pressure[0],l.dimensions)),pressurePlan:plan,
  fields:internal.fieldPages?[...internal.fieldPages.fields].map(([t,f]:any)=>describe(t,f.dims)):[],
  pressureWindow:internal.pressureWindowCapacity,activeRegionEnabled:internal.activeRegionEnabled};
}
const median=(v:number[])=>{const s=[...v].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]!+s[Math.floor(s.length/2)]!)/2;};
await acquireWebGPUExclusiveLock("dawn-probe","Uniform Geometric long dam paging A/B");
let device:GPUDevice|undefined;
const frames=Number(process.env.FRAMES??24);
assert.ok(Number.isInteger(frames)&&frames>4);
const revision=execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8"}).trim();
const sourceHashes=Object.fromEntries(readdirSync("lib/methods/uniform").filter(p=>p.endsWith(".ts")).map(p=>[
 p,createHash("sha256").update(readFileSync(`lib/methods/uniform/${p}`)).digest("hex")]));
const results:unknown[]=[];
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 usePerformanceInstrumentationStore.getState().setEnabled(true);
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 for(const sceneId of (process.env.UNIFORM_BENCH_SCENE?[process.env.UNIFORM_BENCH_SCENE]:["sparse-cm12-long-dam-break"])){
  const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
  if(process.env.TILE_EDGE){
   const edge=Number(process.env.TILE_EDGE);assert.ok([16,32].includes(edge));
   scene.container.width_m=scene.container.height_m=scene.container.depth_m=0.4;
   scene.voxelDomain.finestCellSize_m=0.4/edge;
   scene.fluid.initialDamBreakDimensions_m={x:0.2,y:0.2,z:0.4};
   scene.container.fillFraction=0.25;scene.fluid.initialLiquidVolumes=[];
   scene.solidVoxels=[...solidVoxelShellForScene(scene)];
  }
  const arms:{mode:string;full:number[];stages:Record<string,number[]>;pages:number|undefined}[]=[];
  for(const mode of (process.env.ARMS??"production").split(",")){
   const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,
    {...uniformGeometricSolverOptions({},scene),
     ...(process.env.ONE_CYCLE==='1'?{pressureCycleBudget:'fixed' as const,
      pressureSchedule:{fullCycles:1,vCycles:0,preSweeps:6,postSweeps:6,residualTolerance:10}}:{}),
     ...(process.env.FULL_DOMAIN==='1'?{activeRegion:false,pressureWindow:false}:{}),
    },()=>{});
   const full:number[]=[],stages:Record<string,number[]>={},pressureEvidence:unknown[]=[];let priorSample=-1;
   try{
    for(let frame=1;frame<=frames;frame++){
     const start=performance.now();(solver as any).lastPhysicsTraceAt_ms=-Infinity;assert.ok(solver.advanceTo(frame/30));await (solver as any).awaitFrameCompletion?.();await device.queue.onSubmittedWorkDone();
     if(frame>4)full.push(performance.now()-start);
     if(process.env.ASYNC_DEMAND!=="1") await solver.readStats();
     pressureEvidence.push({frame,encoded:solver.info.uniformPressureCyclesEncoded,executed:solver.info.uniformPressureCyclesExecuted,converged:solver.info.uniformPressureCyclesConverged,passes:solver.info.uniformPressurePassesEncoded});
     const trace=solver.info.physicsTrace;
     if(frame>4&&trace?.measurementSource==="gpu-hardware-timestamp"&&trace.sampleId!==priorSample){
      for(const phase of trace.phases)(stages[phase.label]??=[]).push(phase.duration_ms);
      priorSample=trace.sampleId;
     }
    }
    await solver.readStats();
    assert.equal(solver.info.uniformPageMissingReads??0,0);
    assert.ok(Number.isFinite(solver.info.maxSpeed_m_s));
    const arm={mode,full,stages,pressureEvidence,census:await workCensus(solver),finalInfo:solver.info,allocatedBytes:solver.info.allocatedBytes,
     cyclesEncoded:solver.info.uniformPressureCyclesEncoded,cyclesExecuted:solver.info.uniformPressureCyclesExecuted,
     passesEncoded:solver.info.uniformPressurePassesEncoded,pages:solver.info.uniformVolumePagesActive,transportTiles:solver.info.uniformVolumeTransportWorkgroups,sharpenTiles:solver.info.uniformVolumeSharpenWorkgroups};arms.push(arm);console.log(JSON.stringify({sceneId,fixture:process.env.TILE_EDGE?`single-tile-${process.env.TILE_EDGE}-dam`:sceneId,mode,full_ms:median(full),stages:Object.fromEntries(Object.entries(stages).map(([k,v])=>[k,median(v)])),pages:arm.pages,pressure:solver.info.pressureSolver}));
   }finally{solver.destroy();}
  }
  results.push({revision,sourceHashes,adapter:adapter.info,asyncDemand:process.env.ASYNC_DEMAND==='1',frames,warmup:4,tileEdge:process.env.TILE_EDGE?Number(process.env.TILE_EDGE):undefined,oneCycle:process.env.ONE_CYCLE==='1',fullDomain:process.env.FULL_DOMAIN==='1',sceneId,fixture:process.env.TILE_EDGE?`single-tile-${process.env.TILE_EDGE}-dam`:sceneId,scope:"Queue-fenced simulation, rendering excluded. Readbacks after timed interval. Production defaults, balanced quality, one advance per 1/30 second.",arms});
 }
 assert.deepEqual(errors,[]);
 writeFileSync(process.env.UNIFORM_BENCH_OUTPUT??"/tmp/uniform-long-dam-paging.json",JSON.stringify(results,null,2)+"\n");
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
