import type { MethodParamValues } from "../lib/core/method-contract";
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
(modulePath?test:test.skip)("shared receiver and donor work matches dense dispatch through impact, dust changes and remote insertion",{timeout:600000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform compiled support");
 let device:GPUDevice|undefined;const solvers:WebGPUUniformReferenceSolver[]=[];
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("cm12-figure-7")));
  scene.voxelDomain.finestCellSize_m*=128/96;scene.nominalResolution.length_m*=128/96;
  scene.solidVoxels=[...solidVoxelShellForScene(scene)];
  for(const volumePageWork of [false,true])solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
   ...uniformGeometricSolverOptions({},scene),volumePageWork,compiledSupportForQA:volumePageWork,pressureCycleBudget:"fixed",
  },()=>{}));
  assert.equal(solvers[0]!.info.nx,96);
  const runtime:MethodParamValues={pressureCycleBudget:"fixed"};
  for(let frame=1;frame<=40;frame++){
   if(frame===9)runtime.volumeDustThreshold=0;
   if(frame===13)runtime.volumeDustThreshold=0.003;
   if(frame===17)runtime.volumeDustThreshold=0.001;
   for(const solver of solvers){
    // Match the renderer's repeated calls, including mutation of the caller's
    // existing record when a control actually changes.
    solver.applyRuntimeValues(runtime);
    if(frame===21)solver.injectLiquidBall({centre_m:{x:2,y:4,z:1.5},radius_m:.2});
    assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();
   }
   if([2,14,18,22].includes(frame))assert.equal((solvers[1] as unknown as {receiverWorkEncoded:boolean}).receiverWorkEncoded,true,"unchanged controls retain compiled receiver work");
   if([1,2,9,13,14,17,18,21,22,30,40].includes(frame)){
    for(const field of ["volumeTexture","vertexPhiTexture","velocityTexture"] as const){
     const a=await readTexture(device,solvers[0]![field]!),b=await readTexture(device,solvers[1]![field]!);
     let error=0,norm=0,maximum=0,scale=0;
     for(let i=0;i<a.length;i++){
      assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]));
      const delta=Math.abs(a[i]!-b[i]!);error+=delta*delta;norm+=a[i]!*a[i]!;maximum=Math.max(maximum,delta);scale=Math.max(scale,Math.abs(a[i]!));
     }
     const relativeL2=Math.sqrt(error/Math.max(norm,1e-20));
     console.log(JSON.stringify({frame,field,maximum,relativeL2}));
     assert.ok(relativeL2<1e-5 || maximum<1e-6,`${field} relative error at ${frame}`);
     assert.ok(maximum<Math.max(1e-5,scale*1e-4),`${field} maximum error at ${frame}`);
    }
   }
   // Give both schedules identical physical input for the next comparison.
   const e=device.createCommandEncoder();
   for(const name of ["velocityA","velocityB","velocityD","volumeA","volumeB","gammaA","gammaB","surfaceA","surfaceB","vertexPhiField","vertexPhiScratch","transportA","gridPressureTexture"]){
    const a=(solvers[0] as unknown as Record<string,GPUTexture>)[name]!,b=(solvers[1] as unknown as Record<string,GPUTexture>)[name]!;
    e.copyTextureToTexture({texture:a},{texture:b},[a.width,a.height,a.depthOrArrayLayers]);
   }
   device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();
   assert.deepEqual(errors,[]);
  }
 }finally{for(const solver of solvers)solver.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
