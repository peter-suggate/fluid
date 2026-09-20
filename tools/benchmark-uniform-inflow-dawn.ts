/** Mini64 regression check for canonical surface-deficit reduction ordering. */
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
await acquireWebGPUExclusiveLock("dawn-test","Uniform inflow scheduling mini64 regression");
let device:GPUDevice|undefined;
const results:{mode:string;milliseconds:number[];median_ms:number;allocatedBytes:number}[]=[];
try {
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();if(!adapter)throw Error("No adapter");
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition("minimal-power-dam-break-64"));
 // ABBA order; fresh identical scenes, three warmups followed by twelve steps.
 for(const mode of ["prior","fixed","fixed","prior"]){
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({volumeStorage:"pages32"},scene),()=>{});
  if(mode==="prior"){
   const prior=solver as any;
   prior.encodeSurfaceDeficitBalance=(encoder:GPUCommandEncoder)=>{
    if(!prior.geometricVolume)return;
    if(!prior.surfaceDeficitBalancing){encoder.clearBuffer(prior.conditioningScratch,prior.info.cellCount*12,4);return;}
    prior.run(encoder,"Surface-deficit partial sums",prior.volumePipelines.uvBalanceMeasure,prior.sharpenComputeGroup);
    prior.runDirect(encoder,"Surface-deficit global balance",prior.volumePipelines.uvBalanceReduce,prior.sharpenComputeGroup,[1,1,1]);
   };
  }
  const times:number[]=[];
  try { for(let frame=1;frame<=15;frame++){
   const start=performance.now();solver.advanceTo(frame/30);await solver.awaitFrameCompletion();await device.queue.onSubmittedWorkDone();
   if(frame>3)times.push(performance.now()-start);
  }
  results.push({mode,milliseconds:times,median_ms:median(times),allocatedBytes:solver.info.allocatedBytes});
  console.log(JSON.stringify(results.at(-1)));
  }finally{solver.destroy();}
 }
 if(errors.length)throw new Error(errors.join("\n"));
 const prior=median(results.filter(r=>r.mode==="prior").flatMap(r=>r.milliseconds));
 const fixed=median(results.filter(r=>r.mode==="fixed").flatMap(r=>r.milliseconds));
 const report={scene:"minimal-power-dam-break-64",scope:"full simulation steps, queue completion; excludes presentation",warmup:3,samplesPerRun:12,results,prior_ms:prior,fixed_ms:fixed,throughputRatio:prior/fixed,meets95Percent:prior/fixed>=.95};
 writeFileSync("docs/research/uniform-inflow-mini64-2026-09-21.json",JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify(report));
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
