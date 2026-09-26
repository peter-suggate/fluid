/** Sequential fresh-instance ABBA check: fine physics with and without prepared
 * coarse resources, optionally including the mixed transport preparation.
 * This measures dormant preparation, never active mixed simulation.
 * Compilation and startup are outside fenced step timings. */
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
const mixedTransportPreparation=process.argv.includes("--mixed-transport-preparation");
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
 const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 let residentBytes=0;
 for(const name of ["createBuffer","createTexture"] as const){
  const original=(raw[name] as Function).bind(raw);
  Object.defineProperty(raw,name,{configurable:true,writable:true,value:(descriptor:GPUBufferDescriptor|GPUTextureDescriptor)=>{
   const resource=original(descriptor) as GPUBuffer|GPUTexture;
   let bytes:number;
   if(name==="createBuffer")bytes=(resource as GPUBuffer).size;
   else {const t=resource as GPUTexture;const channels:Record<string,number>={r32float:4,rg32float:8,rgba32float:16,r32uint:4,rg32uint:8,rgba32uint:16,rgba8unorm:4,rgba16float:8,r16float:2};
    assert.ok(t.format in channels,`Unaccounted texture format ${t.format}`);assert.equal(t.mipLevelCount,1);
    bytes=t.width*t.height*t.depthOrArrayLayers*t.sampleCount*channels[t.format]!;
   }
   residentBytes+=bytes;const destroy=resource.destroy.bind(resource);let alive=true;
   Object.defineProperty(resource,"destroy",{configurable:true,writable:true,value:()=>{if(alive){residentBytes-=bytes;alive=false;}destroy();}});
   return resource;
  }});
 }
 device=managedGPUDevice(raw,{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const runs=[];
 for(const prepareCoarseSimulation of [false,true,true,false]) {
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,
   {...uniformGeometricSolverOptions({},scene),prepareCoarseSimulation},()=>{});
  try {
   if(prepareCoarseSimulation && mixedTransportPreparation) {
    const c=scene.container;
    await solver.prepareMixedTransportForQA([{id:"benchmark-fine",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,
     min_m:{x:-c.width_m/2,y:0,z:-c.depth_m/2},max_m:{x:-c.width_m/4,y:c.height_m,z:c.depth_m/2}}]);
   }
   const times=[];
   for(let frame=1;frame<=frames;frame++){
    const start=performance.now();assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();
    if(frame>warmup)times.push(performance.now()-start);
   }
   runs.push({prepareCoarseSimulation,prepareMixedTransport:prepareCoarseSimulation&&mixedTransportPreparation,medianMs:median(times),times,allocatedBytes:solver.info.allocatedBytes,residentBytes});
   console.log(JSON.stringify(runs.at(-1)));
  }finally{solver.destroy();}
 }
 const baseline=median(runs.filter(r=>!r.prepareCoarseSimulation).map(r=>r.medianMs));
 const prepared=median(runs.filter(r=>r.prepareCoarseSimulation).map(r=>r.medianMs));
 const baselineBytes=runs.find(r=>!r.prepareCoarseSimulation)!.residentBytes, preparedBytes=runs.find(r=>r.prepareCoarseSimulation)!.residentBytes;
 assert.ok(preparedBytes<=baselineBytes*1.03,`Actual GPU payload exceeds 3%: ${preparedBytes/baselineBytes}`);
 console.log(JSON.stringify({baselineBytes,preparedBytes,memoryRatio:preparedBytes/baselineBytes,baselineMs:baseline,preparedMs:prepared,ratio:prepared/baseline,tolerance:1.02}));
 assert.deepEqual(errors,[]);
 assert.ok(prepared<=baseline*1.02,`Prepared fine mode exceeded 2% tolerance: ${prepared/baseline}`);
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
