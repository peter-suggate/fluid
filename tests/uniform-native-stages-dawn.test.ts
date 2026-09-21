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
(modulePath?test:test.skip)("native rectangular stages match the domain-free numerical oracle",{timeout:240000},async()=>{
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
    // Match the domain-free phi code form here; the dedicated phi oracle
    // separately checks atlas/native phi equivalence.
    for(const pageDomain of [false,true]){
     solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
      ...uniformGeometricSolverOptions({},scene),pageDomain,activeRegion:false,pressureWindow:false,phiWindowForQA:false,phiLiteralLoopsForQA:true,geometricRedistance:false,pressureCycleDispatch:"direct",pressureCycleBudget:"fixed",
      pressureSchedule:{fullCycles:1,vCycles:0,preSweeps:6,postSweeps:6,residualTolerance:0},
     },()=>{}));
    }
    const compiled=(solvers[1] as any);
    assert.equal(compiled.pageDomainDispatch,undefined,"native root dispatch has exact logical dimensions");
    assert.doesNotMatch(compiled.shaderSource,/fn pageDomainCell/);
    for(let frame=1;frame<=12;frame++){
     // Start each advance from the domain-free arm's canonical fields. This
     // isolates layout/codegen error instead of amplifying earlier f32 noise.
     if(frame>1){
      const e=device.createCommandEncoder(),a=solvers[0] as any,b=solvers[1] as any;
      for(const key of Object.keys(a)){
       const source=a[key] as GPUTexture|undefined,target=b[key] as GPUTexture|undefined;
       if(source && target && typeof source.createView==='function' && typeof target.createView==='function'
         && (source.usage&GPUTextureUsage.COPY_SRC) && (target.usage&GPUTextureUsage.COPY_DST)){
        assert.deepEqual([target.width,target.height,target.depthOrArrayLayers],[source.width,source.height,source.depthOrArrayLayers],key);
        e.copyTextureToTexture({texture:source},{texture:target},[source.width,source.height,source.depthOrArrayLayers]);
       }
      }
      for(const key of ['boundaryVelocityA','boundaryVelocityB'])
       e.copyBufferToBuffer(a[key],0,b[key],0,a[key].size);
      device.queue.submit([e.finish()]);
     }
     for(const solver of solvers){
      if(frame===7)solver.injectLiquidBall({centre_m:{x:0,y:.3,z:0},radius_m:.05});
      assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();
     }
     // Same-input advances may differ in f32 rounding after native address
     // folding; report absolute and normalized error separately.
     for(const field of ['gridPressureTexture','volumeTexture','velocityTexture','vertexPhiTexture'] as const){
      const expected=await read(device,solvers[0]![field]!);
      for(let arm=1;arm<solvers.length;arm++){
       const actual=await read(device,solvers[arm]![field]!);
       const a=new Float32Array(actual.buffer),b=new Float32Array(expected.buffer);
       let maximum=0,relative=0;
       for(let i=0;i<a.length;i++){
        assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]));
        maximum=Math.max(maximum,Math.abs(a[i]!-b[i]!));
        relative=Math.max(relative,Math.abs(a[i]!-b[i]!)/Math.max(1,Math.abs(b[i]!)));
       }
       console.log(JSON.stringify({fixture,field,frame,maximum,relative}));
       if(field==='volumeTexture'||field==='vertexPhiTexture')assert.equal(maximum,0,`${fixture} ${field} frame ${frame}`);
       assert.ok(relative<=1e-5,`${fixture} ${field} frame ${frame}: normalized error ${relative}`);
      }
     }
    }

   }finally{for(const solver of solvers)solver.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
