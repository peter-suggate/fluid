import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
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
(modulePath?test:test.skip)("liquid-tile pressure smoothing matches dense sweeps through 64³ impact",{timeout:600000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform pressure tile work");
 let device:GPUDevice|undefined;
 const solvers:WebGPUUniformReferenceSolver[]=[];
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("cm12-figure-7")));
  scene.voxelDomain.finestCellSize_m*=2;scene.nominalResolution.length_m*=2;
  scene.solidVoxels=[...solidVoxelShellForScene(scene)];
  for(const pressureSmoothingForQA of ["dense",undefined] as const){
   solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
    ...uniformGeometricSolverOptions({},scene),pressureSmoothingForQA,pressureCycleBudget:"fixed",
   },()=>{}));
  }
  assert.equal(solvers[0]!.pressureSmoothingWorkSourceForQA.length,0);
  assert.ok(solvers[1]!.pressureSmoothingWorkSourceForQA.length>0);
  for(let frame=1;frame<=40;frame++){
   for(const solver of solvers){assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();}
   if([1,2,5,24,30,40].includes(frame))for(const field of ["gridPressureTexture","volumeTexture","velocityTexture","vertexPhiTexture"] as const){
    const a=await readTexture(device,solvers[0]![field]!);
    const b=await readTexture(device,solvers[1]![field]!);
    assert.equal(a.length,b.length);
    // The two colour passes commute within each colour. Skipping tiles only
    // after their air constraints have been projected should be bit-exact.
    const aBits=new Uint32Array(a.buffer),bBits=new Uint32Array(b.buffer);
    for(let i=0;i<a.length;i++)assert.ok(Number.isFinite(a[i])&&aBits[i]===bBits[i],`${field} frame ${frame} index ${i}: ${a[i]} != ${b[i]}`);
   }
  }
  assert.deepEqual(errors,[]);
 } finally {for(const solver of solvers)solver.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
