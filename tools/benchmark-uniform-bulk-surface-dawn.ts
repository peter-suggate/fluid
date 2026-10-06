/** Whole-frame gate for the regular-bulk/surface architecture. No renderer,
 * no timestamp instrumentation, identical physical parameters in every arm.
 * Includes classification, layout adoption, remap and accepted completion. */
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
import type {GPUSolverInstance} from "../lib/core/method-contract";

const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const diagnostic=process.argv.includes("--diagnostic");
const selected=arg("case","all");
const cases=["one-tile","long-region","dynamic"] as const;
assert.ok(selected==="all"||cases.some(c=>c===selected));
const windows=[[9,24],[25,48],[49,72],[73,120]] as const;
const median=(a:number[])=>{const v=a.toSorted((a,b)=>a-b);return (v[(v.length-1)>>1]!+v[v.length>>1]!)/2;};
/** Read the visible center-plane toe outside the timed frame. Mass alone
 * cannot catch velocity changes that leave the dam in a compact block. */
async function surfaceFront(device:GPUDevice,solver:GPUSolverInstance):Promise<number>{
 const phi=(solver as unknown as {vertexPhiField:GPUTexture}).vertexPhiField;
 assert.ok(phi,"Front validation requires the solver's vertex level set");
 const stride=Math.ceil(phi.width*4/256)*256;
 const readback=device.createBuffer({size:stride*phi.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{
  const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:phi,origin:[0,0,Math.floor(phi.depthOrArrayLayers/2)]},{buffer:readback,bytesPerRow:stride,rowsPerImage:phi.height},[phi.width,phi.height,1]);
  device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
  const values=new Float32Array(readback.getMappedRange());let front=-1;
  for(let y=1;y<phi.height;y++)for(let x=0;x<phi.width;x++)if(values[x+y*stride/4]!<0)front=Math.max(front,x);
  return front;
 }finally{readback.unmap();readback.destroy();}
}
await acquireWebGPUExclusiveLock("dawn-benchmark","Uniform bulk/surface 2% frame-time gate");
let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??`${process.cwd()}/node_modules/webgpu/index.js`).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const failures:string[]=[];
 for(const kind of cases.filter(c=>selected==="all"||c===selected)){
  const captures:{arm:string;times:number[];cycles:number[];fine:number[];coarse:number[];relayouts:number;fronts:number[]}[]=[];
  for(const arm of ["fine","candidate","candidate","fine"]){
   const scene=structuredClone(sceneDocument(getSceneDefinition(kind==="one-tile"?"minimal-power-dam-break-64":"sparse-cm12-long-dam-break")));
   scene.fluid.refinementRegions=[];
   const lattice=refinementRegionLattice(scene),axes=["x","y","z"] as const;
   if(arm==="candidate"&&kind!=="dynamic"){
    const percent=kind==="one-tile"?[87.5,87.5,87.5,93.75,93.75,93.75]:[91.6667,0,75,100,8.3333,100];
    const point=(offset:number)=>Object.fromEntries(axes.map((a,i)=>[a,lattice.origin_m[a]+percent[i+offset]!/100*lattice.dimensions[i]!*lattice.cellSize_m[i]!])) as {x:number;y:number;z:number};
    scene.fluid.refinementRegions=[{id:kind,rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:point(0),max_m:point(3)}];
   }
   solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{detailPolicy:arm==="candidate"?kind==="dynamic"?"dynamic":"requested":"full"},undefined,()=>{});
   // FOLLOW-UP: the h background with 4h boxes is retired (Requested is 4h outside Fine
   // boxes); the box arms need a Fine complement to measure what they did.
   if(arm==="candidate"&&kind==="one-tile")assert.equal(solver.info.uniformMixedCoarseTiles,1);
   const initial=await solver.readStats();const initialMass=initial.volumeCellSum;
   assert.ok(initialMass!==undefined&&Number.isFinite(initialMass)&&initialMass>0,"Initial liquid mass is required");
   let discardedMass=0;
   const row={arm,times:[] as number[],cycles:[] as number[],fine:[] as number[],coarse:[] as number[],relayouts:0,fronts:[] as number[]};
   for(let step=1;step<=120;step++){
    const start=performance.now();assert.ok(solver.advanceTo(step/30,[]));await solver.awaitFrameCompletion?.();
    row.times.push(performance.now()-start);assert.equal(solver.info.simulationPipelineError,undefined);
    discardedMass+=solver.info.uniformVolumeDustMass_cells??0;
    row.cycles.push(solver.info.uniformPressureCyclesExecuted??0);row.fine.push(solver.info.uniformMixedFineTiles??0);row.coarse.push(solver.info.uniformMixedCoarseTiles??0);
    if(step===10||step===20)row.fronts.push(await surfaceFront(device,solver));
   }
   const info=await solver.readStats();
   assert.ok(info.volumeCellSum!==undefined&&Number.isFinite(info.volumeCellSum),"Final liquid mass is required");
   const accountedMassDrift=(info.volumeCellSum+discardedMass-initialMass)/initialMass;
   row.relayouts=info.uniformMixedDynamicRelayouts??0;
   if(arm==="candidate"&&kind==="dynamic"){
    assert.ok(row.relayouts>0,"Dynamic arm must change ownership");assert.ok(row.coarse.some(n=>n>0),"Dynamic arm must do coarse work");
    assert.ok(new Set(row.fine).size>1,"Dynamic surface mask must move");
   }
   console.log(JSON.stringify({kind,...row,volume:info.volumeCellSum,initialMass,discardedMass,accountedMassDrift,volumeDrift:info.volumeDrift,front_m:info.front_m,bytes:info.allocatedBytes}));captures.push(row);
   assert.ok(Math.abs(accountedMassDrift)<1e-4,`${kind}/${arm}: unaccounted mass drift ${accountedMassDrift}`);
   solver.destroy();solver=undefined;await device.queue.onSubmittedWorkDone();
  }
  // The existing long-dam regression allows two h cells of front movement.
  // Apply that same limit against the contemporaneous all-fine control,
  // including in diagnostic mode; timing is meaningless after a physics regression.
  if(kind!=="one-tile")for(const candidate of captures.filter(r=>r.arm==="candidate"))for(let i=0;i<2;i++){
   const reference=median(captures.filter(r=>r.arm==="fine").map(r=>r.fronts[i]!));
   assert.ok(Math.abs(candidate.fronts[i]!-reference)<=2,`${kind} frame ${(i+1)*10}: front ${candidate.fronts[i]}, all-fine ${reference} ± 2 h`);
  }
  const comparisons=windows.map(([start,end])=>{
   const take=(arm:string)=>captures.filter(r=>r.arm===arm).flatMap(r=>r.times.slice(start-1,end));
   const fine=median(take("fine")),candidate=median(take("candidate"));
   const result={frames:[start,end],fine_ms:fine,candidate_ms:candidate,ratio:candidate/fine,maximumRatio:1.02,passed:candidate<=fine*1.02};
   if(!result.passed)failures.push(`${kind} frames ${start}–${end}: ${result.ratio.toFixed(4)} > 1.02`);return result;
  });
  console.log(JSON.stringify({kind,comparisons,boundary:"accepted frame completion including dynamic census/remap; renderer excluded",order:"ABBA",repeats:2}));
 }
 assert.deepEqual(errors,[]);
 if(!diagnostic)assert.deepEqual(failures,[],"Uniform bulk/surface frame-time gate");
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
