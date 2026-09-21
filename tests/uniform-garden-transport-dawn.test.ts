import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";

async function readTexture(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const components=texture.format==="rgba32float"?4:1;
  const width=texture.width*components;
  const row=Math.ceil(width*4/256)*256;
  const b=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:b,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
    device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
    const src=new Float32Array(b.getMappedRange()), result=new Float32Array(width*texture.height*texture.depthOrArrayLayers);
    for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)result.set(src.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+width),(z*texture.height+y)*width);
    return result;
  } finally {if(b.mapState==="mapped")b.unmap();b.destroy();}
}

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("garden transport excludes dry terrain and matches full-domain transport",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","garden fluid support");
 let device:GPUDevice|undefined;const solvers:WebGPUUniformReferenceSolver[]=[];
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=sceneDocument(getSceneDefinition("hero-garden-hose"));
  const bodies=initializeRigidBodies(scene.rigidBodies);
  const dry={...scene,fluid:{...scene.fluid,inflow:undefined}};
  for(const transportTiles of [true,false])solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,dry,"balanced",undefined,{
   ...uniformGeometricSolverOptions({},dry),transportTiles,activeRegion:false,pressureWindow:false,phiWindowForQA:false,
   pressureCycleBudget:"fixed",pressureSchedule:{fullCycles:1,vCycles:0,preSweeps:6,postSweeps:6,residualTolerance:0},
  },()=>{}));
  const maxima:Record<string,number>={};
  for(let frame=1;frame<=12;frame++){
   for(const solver of solvers){
    if(frame===3)solver.applySceneUniforms(scene);
    if(frame===8)solver.injectLiquidBall({centre_m:{x:.65,y:.65,z:-.35},radius_m:.035});
    if(frame===10)solver.applySceneUniforms(dry);
    assert.ok(solver.advanceTo(frame/30,bodies));await solver.awaitFrameCompletion();await solver.readStats();
   }
   const info=solvers[0]!.info;
   assert.equal(info.uniformTwoLevelFineTiles,solvers[1]!.info.uniformTwoLevelFineTiles,"transport selection must not change fine velocity support");
   assert.equal((info as typeof info & {uniformTwoLevelShellTiles?:number}).uniformTwoLevelShellTiles,(solvers[1]!.info as typeof info & {uniformTwoLevelShellTiles?:number}).uniformTwoLevelShellTiles,"transport selection must not change extension support");
   const binding=solvers[0]!.volumePageSource!.transportRecords!;
   const bytes=binding.size??binding.buffer.size-(binding.offset??0);
   const staging:GPUBuffer=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(binding.buffer,binding.offset??0,staging,0,bytes);device.queue.submit([encoder.finish()]);
   await staging.mapAsync(GPUMapMode.READ);const pages:Uint32Array=new Uint32Array(staging.getMappedRange()).slice();staging.unmap();staging.destroy();
   const transportPages=pages[4]!;
   assert.equal(transportPages,Array.from(pages.subarray(8,8+pages[1]!*pages[2]!*pages[3]!)).filter(Boolean).length);
   if(frame<=2){
    assert.equal(info.uniformTransportTiles,0,"dry terrain alone must not activate transport");
    assert.equal(info.uniformVolumeTransportWorkgroups,0);
    assert.equal(transportPages,0,"page visual must publish no dry transport pages");
    assert.ok((info.uniformTransportTilesTotal??0)>0);
   }
   if(frame===3){
    assert.ok(info.uniformTransportTiles!>0,"hose activates transport");
    assert.ok(info.uniformTransportTiles!<info.uniformTransportTilesTotal!/4,"hose support stays local");
   }
   for(const field of ["volumeTexture","vertexPhiTexture","velocityTexture"] as const){
    const a=await readTexture(device,solvers[0]![field]!),b=await readTexture(device,solvers[1]![field]!);
    let max=0;
    for(let i=0;i<a.length;i++){
     assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]));
     max=Math.max(max,Math.abs(a[i]!-b[i]!)/Math.max(1,Math.abs(b[i]!)));
    }
    maxima[field]=Math.max(maxima[field]??0,max);
    assert.ok(max<=1e-5,`${field} frame ${frame}: normalized discrepancy ${max}`);
   }
   console.log(JSON.stringify({frame,transportPages,transportTiles:info.uniformTransportTiles,totalTiles:info.uniformTransportTilesTotal,
    transportGroups:info.uniformVolumeTransportWorkgroups,volume:info.volumeCellSum}));
  }
  console.log(JSON.stringify({maxima}));assert.deepEqual(errors,[]);
 }finally{for(const solver of solvers)solver.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
