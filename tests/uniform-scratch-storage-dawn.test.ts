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
(modulePath?test:test.skip)("shared uniform scratch matches separate backing on identical inputs through the 64³ ball drop",{timeout:1200000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform shared stage storage");
 let device:GPUDevice|undefined;
 const solvers:WebGPUUniformReferenceSolver[]=[];
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("cm12-figure-7")));
  scene.voxelDomain.finestCellSize_m*=2;
  scene.nominalResolution.length_m*=2;
  scene.solidVoxels=[...solidVoxelShellForScene(scene)];
  for(const [scratchStorageForQA,retainStageDiagnosticsForQA] of [["separate",true],[undefined,true],[undefined,false]] as const){
   const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
    ...uniformGeometricSolverOptions({},scene),scratchStorageForQA,retainStageDiagnosticsForQA,
    pressureCycleBudget:"fixed",
   },()=>{});
   solvers.push(solver);
   assert.equal(solver.info.nx,64);
   if(retainStageDiagnosticsForQA)solver.enableStageDiagnosticsForQA();
  }
  assert.ok(solvers[2]!.info.allocatedBytes<solvers[0]!.info.allocatedBytes*.6);
  const fields=["volumeTexture","velocityTexture","gridPressureTexture","extrapolatedVelocityTexture","vertexPhiTexture","surfaceFieldTexture",
   "preProjectionVelocityTexture","extrapolationActiveStateTexture","advectedVertexPhiTexture"] as const;
  const failures:string[]=[];
  for(let frame=1;frame<=40;frame++){
   for(const solver of solvers){assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();}
   if([1,2,3,4,5,24,30,40].includes(frame)){
   for(const field of fields){
    const a=await readTexture(device,solvers[0]![field]!);
    for(const arm of (field.startsWith("preProjection")||field.startsWith("extrapolation")||field.startsWith("advected") ? [1] : [1,2])){
    const b=await readTexture(device,solvers[arm]![field]!);
    assert.equal(a.length,b.length,`${field} retains its logical dimensions`);
    let squared=0,scale=0,maxError=0,maxValue=0;
    for(let i=0;i<a.length;i++){
     assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]),`${field} finite at ${i}`);
     const delta=Math.abs(a[i]!-b[i]!);squared+=delta*delta;scale+=a[i]!*a[i]!;
     maxError=Math.max(maxError,delta);maxValue=Math.max(maxValue,Math.abs(a[i]!));
    }
    const relativeL2=Math.sqrt(squared/Math.max(scale,1e-20));
    console.log(JSON.stringify({frame,field,arm,maxError,relativeL2}));
    // Storage access can change f32 contraction in the Metal compiler.
    // Identical physical input must produce equivalent stage output.
    // Free-fall pressure is nearly zero. Use a 1 mPa absolute floor there;
    // relative error alone is ill-conditioned for that field.
    const absoluteTolerance=field==="gridPressureTexture" ? 1e-3 : 1e-6;
    if(!(relativeL2<1e-5 || maxError<absoluteTolerance))failures.push(`${field} relative error ${relativeL2} frame ${frame}`);
    if(!(maxError<Math.max(1e-5,absoluteTolerance,maxValue*1e-4)))failures.push(`${field} maximum error ${maxError} frame ${frame}`);
    }
   }
   }
   // Use the control trajectory as the next step's physical input. This
   // isolates storage equivalence from amplification of f32 roundoff by
   // subsequent nonlinear sharpening/impact steps. The profiler separately
   // records an uninterrupted 60-frame production trajectory and mass drift.
   const encoder=device.createCommandEncoder();
   const physicalFields=["velocityA","velocityB","velocityD","volumeA","volumeB",
    "gammaA","gammaB","surfaceA","surfaceB","vertexPhiField","vertexPhiScratch","transportA","gridPressureTexture"];
   const control=solvers[0] as unknown as Record<string,GPUTexture>;
   for(const solver of solvers.slice(1))for(const name of physicalFields){
    const source=control[name]!,destination=(solver as unknown as Record<string,GPUTexture>)[name]!;
    encoder.copyTextureToTexture({texture:source},{texture:destination},[source.width,source.height,source.depthOrArrayLayers]);
   }
   device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
  }
  assert.deepEqual(errors,[]);
  assert.deepEqual(failures,[]);
 } finally {for(const solver of solvers)solver.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
