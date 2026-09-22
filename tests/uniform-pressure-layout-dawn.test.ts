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
/** The same read, unpadded and as f32, for a per-texel comparison. */
async function floats(device:GPUDevice,texture:GPUTexture):Promise<Float32Array> {
 const components=texture.format==='rgba32float'?4:1;
 const width=texture.width*components;
 const row=Math.ceil(width*4/256)*256;
 const staging=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try {
  const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:staging,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
  device.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);
  const src=new Float32Array(staging.getMappedRange()),result=new Float32Array(width*texture.height*texture.depthOrArrayLayers);
  for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)
   result.set(src.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+width),(z*texture.height+y)*width);
  return result;
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
    // Native smoothing visits compact liquid tiles, and its finest per-cycle
    // operators run from the cycle list. Ignore the list's own setup and the
    // dense passes that seed the far field it skips, then compare: every
    // pressure operator outside smoothing must still run, in order, and every
    // one that is not listed must keep its launch.
    const listSetup=(p:any)=>/^mg(Smooth|BuildSmooth|PublishSmooth|BuildCycle)/.test(p.entryPoint)||p.cycleSetup;
    const dense=(entryPoint:string)=>entryPoint.replace(/Tiles$/,"");
    const listed=new Set(native.plan.filter((p:any)=>p.tileList).map((p:any)=>dense(p.entryPoint)));
    const schedule=(plan:any[])=>plan.filter(p=>!listSetup(p)).map(p=>dense(p.entryPoint));
    const launches=(plan:any[])=>plan.filter(p=>!listSetup(p)&&!listed.has(dense(p.entryPoint)))
      .map(p=>[p.entryPoint,p.workgroups]);
    assert.deepEqual(schedule(native.plan),schedule(logical.plan));
    assert.deepEqual(launches(native.plan),launches(logical.plan));
    assert.ok(native.plan.some((p:any)=>p.entryPoint==='mgSmoothVisitInPlace'),'the fixture exercises the fused visit');
    for(let frame=1;frame<=12;frame++){
     for(const solver of solvers){
      if(frame===7)solver.injectLiquidBall({centre_m:{x:0,y:.3,z:0},radius_m:.05});
      assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();
     }
     // Equal output after each step guarantees the next solve sees identical
     // topology/RHS inputs, rather than comparing diverged trajectories.
     for(const field of ['volumeTexture','velocityTexture','vertexPhiTexture'] as const){
      const expected=await read(device,solvers[0]![field]!);
      for(let arm=1;arm<solvers.length;arm++)assert.deepEqual(await read(device,solvers[arm]![field]!),expected,`${fixture} ${field} frame ${frame} arm ${arm}`);
     }
     // Pressure is bit-for-bit on every row the native arm's cycle list
     // carries -- all liquid and all constrained rows, plus a tile of margin.
     // Outside it the native arm keeps the exact zero mgBuildFinestRhs stored
     // where an unlisted arm accumulates the trilinear prolongation of COARSE
     // air rows. Those rows are dead in both: the smoother, mgApply and
     // mgResidual mask non-liquid neighbours, the p_min downsample saturates
     // at -FLT_MAX whatever p is, and the projection reads air pressure as 0
     // -- which is what the three fields above, bit-for-bit on a coupled
     // trajectory, establish.
     const listedPressure=await floats(device,solvers[0]!.gridPressureTexture);
     for(let arm=1;arm<solvers.length;arm++){
      const other=await floats(device,solvers[arm]!.gridPressureTexture);
      assert.equal(other.length,listedPressure.length);
      const listedBits=new Uint32Array(listedPressure.buffer),otherBits=new Uint32Array(other.buffer);
      let scale=0,worst=0,unlisted=0;
      for(let i=0;i<other.length;i++){
       assert.ok(Number.isFinite(other[i])&&Number.isFinite(listedPressure[i]),`${fixture} pressure frame ${frame} arm ${arm} index ${i}`);
       scale=Math.max(scale,Math.abs(other[i]!));
       if(listedBits[i]===otherBits[i])continue;
       unlisted+=1;
       assert.ok(listedPressure[i]===0,`${fixture} pressure frame ${frame} arm ${arm} index ${i}: native ${listedPressure[i]} != ${other[i]} on a listed row`);
       worst=Math.max(worst,Math.abs(other[i]!));
      }
      assert.ok(scale===0||worst<scale,`${fixture} frame ${frame} arm ${arm}: unlisted pressure ${worst} is the field maximum ${scale}`);
     }
    }
    console.log(JSON.stringify({fixture,nativeBytes:native.allocatedBytes,atlasBytes:tiled.allocatedBytes}));
   }finally{for(const solver of solvers)solver.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
