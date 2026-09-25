/** Sequential fresh-instance ABBA check: fine physics with and without prepared
 * coarse resources. Compilation and startup are outside fenced step timings. */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
const median=(a:number[])=>{const s=[...a].sort((x,y)=>x-y);return (s[Math.floor((s.length-1)/2)]!+s[Math.floor(s.length/2)]!)/2;};
const frames=30,warmup=5;
// Wait for another local GPU job without launching Dawn concurrently. The
// timeout concerns lease acquisition only, never a numerical/performance lane.
const leaseStart=performance.now();
for(;;) {
 try {await acquireWebGPUExclusiveLock("dawn-benchmark","Uniform prepared coarse fine-mode ABBA");break;}
 catch(error) {
  if(performance.now()-leaseStart>120000 || !String(error).includes("Refusing concurrent GPU execution"))throw error;
  await new Promise(resolve=>setTimeout(resolve,250));
 }
}
let device:GPUDevice|undefined;
try {
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const runs=[];
 for(const prepareCoarseSimulation of [false,true,true,false]) {
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,
   {...uniformGeometricSolverOptions({},scene),prepareCoarseSimulation},()=>{});
  try {
   const times=[];
   for(let frame=1;frame<=frames;frame++){
    const start=performance.now();assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();
    if(frame>warmup)times.push(performance.now()-start);
   }
   runs.push({prepareCoarseSimulation,medianMs:median(times),times,allocatedBytes:solver.info.allocatedBytes});
   console.log(JSON.stringify(runs.at(-1)));
  }finally{solver.destroy();}
 }
 const baseline=median(runs.filter(r=>!r.prepareCoarseSimulation).map(r=>r.medianMs));
 const prepared=median(runs.filter(r=>r.prepareCoarseSimulation).map(r=>r.medianMs));
 console.log(JSON.stringify({baselineMs:baseline,preparedMs:prepared,ratio:prepared/baseline,tolerance:1.02}));
 assert.deepEqual(errors,[]);
 assert.ok(prepared<=baseline*1.02,`Prepared fine mode exceeded 2% tolerance: ${prepared/baseline}`);
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
