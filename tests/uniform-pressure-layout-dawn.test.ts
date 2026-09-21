import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";

async function read(device:GPUDevice,texture:GPUTexture) {
 const components=texture.format==='rgba32float'?4:1;
 const row=Math.ceil(texture.width*components*4/256)*256;
 const staging=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try {
  const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:staging,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
  device.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);
  return new Uint8Array(staging.getMappedRange()).slice();
 } finally {if(staging.mapState==='mapped')staging.unmap();staging.destroy();}
}
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("native pressure workspace equals tiled and logical-dispatch atlas on identical inputs",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","native pressure layout equivalence");let device:GPUDevice|undefined;
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const fixture of ["partial-pages","long-dam"]){
   const scene=structuredClone(sceneDocument(getSceneDefinition("sparse-cm12-long-dam-break")));
   if(fixture==='partial-pages'){
    scene.container.width_m=1;scene.container.height_m=.5;scene.container.depth_m=.4;
    scene.voxelDomain.finestCellSize_m=.025;
    scene.fluid.initialDamBreakDimensions_m={x:.25,y:.25,z:.4};
    scene.solidVoxels=[...solidVoxelShellForScene(scene)];
   }
   const solvers:WebGPUUniformReferenceSolver[]=[];
   try {
    for(const pressureStorageForQA of [undefined,"paged-logical","paged"] as const){
     solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
      ...uniformGeometricSolverOptions({},scene),pressureStorageForQA,pressureCycleDispatch:"direct",pressureCycleBudget:"fixed",
      pressureSchedule:{fullCycles:1,vCycles:0,preSweeps:6,postSweeps:6,residualTolerance:0},
     },()=>{}));
    }
    const native=(solvers[0] as any).pressureMultigrid;
    const logical=(solvers[1] as any).pressureMultigrid;
    const tiled=(solvers[2] as any).pressureMultigrid;
    assert.equal(native.pressurePublication,undefined,"projection directly consumes the native solve result");
    assert.doesNotMatch(native.shaderFragment,/mgPageAddress/);
    assert.ok(native.allocatedBytes<tiled.allocatedBytes);
    assert.deepEqual(native.plan.map((p:any)=>p.workgroups),logical.plan.map((p:any)=>p.workgroups));
    for(let frame=1;frame<=12;frame++){
     for(const solver of solvers){
      if(frame===7)solver.injectLiquidBall({centre_m:{x:0,y:.3,z:0},radius_m:.05});
      assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();
     }
     // Equal output after each step guarantees the next solve sees identical
     // topology/RHS inputs, rather than comparing diverged trajectories.
     for(const field of ['gridPressureTexture','volumeTexture','velocityTexture','vertexPhiTexture'] as const){
      const expected=await read(device,solvers[0]![field]!);
      for(let arm=1;arm<solvers.length;arm++)assert.deepEqual(await read(device,solvers[arm]![field]!),expected,`${fixture} ${field} frame ${frame} arm ${arm}`);
     }
    }
    console.log(JSON.stringify({fixture,nativeBytes:native.allocatedBytes,atlasBytes:tiled.allocatedBytes}));
   }finally{for(const solver of solvers)solver.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
