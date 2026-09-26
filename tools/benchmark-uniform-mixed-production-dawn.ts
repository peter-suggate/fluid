import assert from "node:assert/strict";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {GPUSolverInstance} from "../lib/core/method-contract";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";

const median=(values:number[])=>{const v=[...values].sort((a,b)=>a-b),i=Math.floor(v.length/2);return v.length%2?v[i]!:(v[i-1]!+v[i]!)/2;};
const diagnostic=process.argv.includes("--diagnostic");
const profile=process.argv.includes("--profile");
if(profile)usePerformanceInstrumentationStore.getState().setEnabled(true);
const mode=process.argv.find(v=>v.startsWith("--mode="))?.slice(7)??"all";
assert.ok(["all","fine","air","half"].includes(mode));
const modes=mode==="all"?["fine","air","half"]:[mode];
const base=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
const lattice=refinementRegionLattice(base),axes=["x","y","z"] as const;
const warmup=8,samples=16;
await acquireWebGPUExclusiveLock("dawn-benchmark","Uniform production mixed defaults ABBA");
let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??`${process.cwd()}/node_modules/webgpu/index.js`).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits),requiredFeatures:profile?["timestamp-query"]:[]}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 for(const mode of modes){
  const results:{arm:string;times:number[];mass:number|undefined;front:number|undefined;bytes:number}[]=[];
  for(const arm of ["native","mixed","mixed","native"]){
   const scene=structuredClone(base);scene.fluid.refinementRegions=[];
   if(arm==="mixed"&&mode!=="fine"){
    const min=Object.fromEntries(axes.map((a,i)=>[a,lattice.origin_m[a]+(mode==="air"?lattice.dimensions[i]!-8:i===0?lattice.dimensions[i]!/2:0)*lattice.cellSize_m[i]!]));
    const max=Object.fromEntries(axes.map((a,i)=>[a,lattice.origin_m[a]+(lattice.dimensions[i]!-(mode==="air"?4:0))*lattice.cellSize_m[i]!]));
    scene.fluid.refinementRegions=[{id:mode,rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:min as {x:number;y:number;z:number},max_m:max as {x:number;y:number;z:number}}];
   }
   solver=arm==="native"
    ?await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({},scene),()=>{})
    :await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{},undefined,()=>{});
   if(arm==="mixed"&&mode==="air"){
    assert.equal(solver.info.uniformMixedCoarseTiles,1);assert.equal(solver.info.uniformMixedTransitionTiles,26);
   }
   const times:number[]=[];
   for(let step=1;step<=warmup+samples;step++){
    const start=performance.now();assert.ok(solver.advanceTo(step/30,[]));await solver.awaitFrameCompletion?.();
    assert.equal(solver.info.simulationPipelineError,undefined);
    if(profile&&step>warmup)console.log(JSON.stringify({mode,arm,step,cycles:solver.info.uniformPressureCyclesExecuted,residual:solver.info.uniformPressureAcceptedResidual,trace:solver.info.physicsTrace}));
    if(step>warmup)times.push(performance.now()-start);
   }
   const info=await solver.readStats();
   const row={arm,times,mass:info.volumeCellSum,front:info.front_m,bytes:info.allocatedBytes};results.push(row);
   console.log(JSON.stringify({mode,...row,maxSpeed_m_s:info.maxSpeed_m_s,median_ms:median(times),cleanup:"production defaults",completedTime_s:info.completedTime_s,
    fineTiles:info.uniformMixedFineTiles,transitionTiles:info.uniformMixedTransitionTiles,coarseTiles:info.uniformMixedCoarseTiles,regularTiles:info.uniformMixedRegularTiles,generalTiles:info.uniformMixedGeneralTiles,fineSupportTiles:info.uniformTwoLevelFineTiles}));
   solver.destroy();solver=undefined;await device.queue.onSubmittedWorkDone();
  }
  const native=median(results.filter(r=>r.arm==="native").flatMap(r=>r.times)),mixed=median(results.filter(r=>r.arm==="mixed").flatMap(r=>r.times));
  const throughput=native/mixed,required=mode==="half"?1:.98;
  const residentRatio=Math.max(...results.filter(r=>r.arm==="mixed").map(r=>r.bytes))/Math.min(...results.filter(r=>r.arm==="native").map(r=>r.bytes));
  console.log(JSON.stringify({mode,nativeMedian_ms:native,mixedMedian_ms:mixed,throughput,required,residentRatio,maximumResidentRatio:1.03,passed:throughput>=required&&residentRatio<=1.03,
   boundary:"advanceTo through accepted frame completion; canonical publication included, renderer excluded",warmup,samples,repeats:2}));
  if(!diagnostic){
   assert.ok(residentRatio<=1.03,`${mode}: resident memory ${residentRatio} exceeds 1.03`);
   assert.ok(throughput>=required,`${mode}: throughput ${throughput} < ${required}`);
  }
 }
 assert.deepEqual(errors,[]);
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
