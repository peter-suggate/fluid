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
 } finally {staging.unmap();staging.destroy();}
}
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("compiled single-page execution matches native operators through motion and insertion",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","compiled single-page execution");let device:GPUDevice|undefined;
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const edge of [16,32]) {
   const scene=structuredClone(sceneDocument(getSceneDefinition("sparse-cm12-long-dam-break")));
   scene.container.width_m=scene.container.height_m=scene.container.depth_m=.4;
   scene.voxelDomain.finestCellSize_m=.4/edge;
   scene.fluid.initialDamBreakDimensions_m={x:.2,y:.2,z:.4};scene.container.fillFraction=.25;
   scene.fluid.initialLiquidVolumes=[];scene.solidVoxels=[...solidVoxelShellForScene(scene)];
   const options={...uniformGeometricSolverOptions({},scene),activeRegion:false,pressureCycleBudget:"fixed" as const,
    pressureSchedule:{fullCycles:1,vCycles:0,preSweeps:6,postSweeps:6,residualTolerance:10}};
   const native=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...options,pageDomain:false,volumePages:undefined},()=>{});
   const compiled=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,options,()=>{});
   try {
    const internals=compiled as any;
    assert.equal(internals.nativePageCoordinates,true);
    assert.equal(internals.fieldPages,undefined);
    assert.equal(internals.pageDomainDispatch,undefined);
    assert.equal(internals.volumePageConfig,undefined);
    assert.equal(compiled.volumeTexture.width,edge);
    assert.equal(compiled.vertexPhiTexture!.width,edge+1);
    assert.equal(compiled.gridPressureTexture.width,edge+2);
    assert.ok(compiled.volumePageSource,"accepted page overlay remains available");
    for(let frame=1;frame<=24;frame++) {
     for(const solver of [native,compiled]) {
      if(frame===9)solver.injectLiquidBall({centre_m:{x:.1,y:.3,z:0},radius_m:.04});
      assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();
     }
     if([1,8,24].includes(frame)) for(const field of ['volumeTexture','vertexPhiTexture','velocityTexture','gridPressureTexture'] as const)
      assert.deepEqual(await read(device,compiled[field]!),await read(device,native[field]!),`${edge} ${field} frame ${frame}`);
    }
   } finally {native.destroy();compiled.destroy();}
  }
  assert.deepEqual(errors,[]);
 } finally {device?.destroy();await releaseWebGPUExclusiveLock();}
});
