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
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";

async function readTexture(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const row=Math.ceil(texture.width*4/256)*256;
  const b=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:b,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
    device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
    const src=new Float32Array(b.getMappedRange()), result=new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers);
    for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)result.set(src.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+texture.width),(z*texture.height+y)*texture.width);
    return result;
  } finally {if(b.mapState==="mapped")b.unmap();b.destroy();}
}

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("continuing inflow releases dense launches and pressure capacity without losing source support",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform inflow window");
 let device:GPUDevice|undefined;const solvers:WebGPUUniformReferenceSolver[]=[];
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=sceneDocument(getSceneDefinition("hero-garden-hose"));
  for(const pressureWindow of ["domain","domain","window"])
   solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({volumeStorage:"pages32",pressureWindow},scene),()=>{}));
  // Retain the old every-frame external-source path as a scheduling oracle.
  const oracle=solvers[1] as any;
  const oldPlan=oracle.planWindowDispatch.bind(oracle),oldScan=oracle.encodeActiveRegion.bind(oracle);
  oracle.planWindowDispatch=(dt:number)=>oldPlan(dt,true);
  oracle.encodeActiveRegion=(encoder:GPUCommandEncoder)=>oldScan(encoder,true);
  let denseAt12=0,windowPressureSteps=0;
  for(let frame=1;frame<=32;frame++){
   for(const solver of solvers){
    if(frame===25)solver.injectLiquidBall({centre_m:{x:.65,y:.5,z:.35},radius_m:.04});
    assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();await solver.readStats();
   }
   if(solvers[2]!.info.uniformPressureLatticeWindowed)windowPressureSteps++;
   if(frame===12)denseAt12=solvers[0]!.info.uniformSolveWindowDenseSteps!;
   if(frame===24){
    assert.equal(solvers[0]!.info.uniformSolveWindowDenseSteps,denseAt12,"continuous hose must stop renewing dense fallback");

   }
   if([1,12,24,25,32].includes(frame)){
    for(const field of ["volumeTexture","vertexPhiTexture"] as const){
     const a=await readTexture(device,solvers[0]![field]!),b=await readTexture(device,solvers[1]![field]!);
     let maxError=0;
     for(let i=0;i<a.length;i++){
      assert.ok(Number.isFinite(a[i]) && Number.isFinite(b[i]));
      maxError=Math.max(maxError,Math.abs(a[i]!-b[i]!));
     }
     console.log(JSON.stringify({frame,field,maxError}));
     assert.equal(maxError,0,`${field} must match the dense-launch oracle at frame ${frame}`);
    }
    console.log(JSON.stringify({frame,arms:solvers.map(s=>({dense:s.info.uniformSolveWindowDenseSteps,clipped:s.info.uniformSolveWindowClippedSteps,lattice:s.info.uniformPressureLattice,windowed:s.info.uniformPressureLatticeWindowed}))}));
   }
  }
  assert.ok(windowPressureSteps>0,"pressure must be allowed to leave domain capacity during continuous inflow");
  assert.ok(solvers[0]!.info.uniformSolveWindowDenseSteps!>denseAt12,"live drop must retain conservative fallback");
  for(const solver of solvers)assert.equal(solver.info.uniformSolveWindowClippedSteps,0);
  assert.deepEqual(errors,[]);
 }finally{for(const s of solvers)s.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
