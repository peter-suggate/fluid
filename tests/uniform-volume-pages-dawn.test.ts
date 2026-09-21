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
import { GridOverlayPipeline } from "../lib/core/webgpu-grid-overlay";
import { visualLayers } from "../lib/core/visual-layers";

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
(modulePath?test:test.skip)("production volume pages preserve the dense step and publish renderable page records",{timeout:240000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","Uniform Geometric volume pages");
  let device:GPUDevice|undefined;
  const solvers:WebGPUUniformReferenceSolver[]=[], resources:{destroy():void}[]=[];
  try {
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    const scene=sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
    for(const storage of ["dense","pages16","pages32"]) {
      solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,
        uniformGeometricSolverOptions({volumeStorage:storage},scene),()=>{}));
    }
    for(let frame=1;frame<=5;frame++) {
      for(const solver of solvers) {
        assert.ok(solver.advanceTo(frame/30));
        assert.equal(solver.info.hostSchedulingUsesReadback,true);
        assert.equal(solver.framePending,false);
        await solver.awaitFrameCompletion();await device.queue.onSubmittedWorkDone();
        await solver.readStats();
        assert.equal(solver.framePending,false);
        assert.equal(solver.info.submittedTime_s,frame/30);
      }
      const expected=await readTexture(device,solvers[0]!.volumeTexture);
      const phi=await readTexture(device,solvers[0]!.vertexPhiTexture!);
      for(const solver of solvers.slice(1)) {
        assert.deepEqual(await readTexture(device,solver.volumeTexture),expected,`V frame ${frame}`);
        assert.deepEqual(await readTexture(device,solver.vertexPhiTexture!),phi,`phi frame ${frame}`);
        assert.ok((solver.info.uniformVolumePagesActive??0)>0);
      }
    }
    const solver=solvers[1]!;
    assert.ok(solvers[0]!.volumePageSource,"page domain is visible even with dense scratch backing");
    assert.equal(solver.info.uniformVolumePageEdge,32);
    const tex=(format:GPUTextureFormat)=>{const t=device!.createTexture({size:[1,1,1],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING});resources.push(t);return t;};
    const target=device.createTexture({size:[16,16],format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT});
    const uniforms=device.createBuffer({size:416,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const bodies=device.createBuffer({size:768,usage:GPUBufferUsage.STORAGE});
    const columns=device.createTexture({size:[1,1],format:"rg32float",usage:GPUTextureUsage.TEXTURE_BINDING});
    resources.push(target,uniforms,bodies,columns);
    const overlay=new GridOverlayPipeline(device,"rgba8unorm",uniforms,bodies);resources.push(overlay);
    device.pushErrorScope("validation");await overlay.initialize();
    overlay.setDenseLevelSetVolumeSource(solver.denseLevelSetVolumeSource);
    overlay.setVolume(solver.volumeTexture,columns,tex("r32uint"),tex("rgba32float"),tex("rg32uint"),tex("r32float"),tex("r32float"),solver.volumeTexture);
    overlay.setLayers(visualLayers(["pages","surface"]),solver.tileClassSource,solver.solveWindowSource,undefined,undefined,solver.volumePageSource);
    const e=device.createCommandEncoder();assert.ok(overlay.encode(e,target.createView()));device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();assert.equal((await device.popErrorScope())?.message,undefined);
    assert.deepEqual(errors,[]);
  } finally {for(const r of resources)r.destroy();for(const s of solvers)s.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});

(modulePath?test:test.skip)("garden hose page work preserves live liquid insertion",{timeout:240000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","Uniform pages garden hose");
  let device:GPUDevice|undefined;
  const solvers:WebGPUUniformReferenceSolver[]=[];
  try {
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    const scene=sceneDocument(getSceneDefinition("hero-garden-hose"));
    for(const pageSize of ["32","16"])solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({pageSize,pressureWindow:"domain"},scene),()=>{}));
    assert.equal(solvers[0]!.info.uniformDomainPages,45);
    assert.equal(solvers[1]!.info.uniformDomainPages,324);
    for(let frame=1;frame<=12;frame++) {
      for(const solver of solvers) {
        if(frame===5)solver.applyRuntimeValues({sharpeningWorkMap:"off",transportWorkMap:"dense"});
        if(frame===9)solver.applyRuntimeValues({sharpeningWorkMap:"on",transportWorkMap:"tiles"});
        if(frame===2 || frame===7)solver.injectLiquidBall({centre_m:{x:0.65,y:0.5,z:0.35},radius_m:0.04});
        assert.ok(solver.advanceTo(frame/30));
        assert.equal(solver.framePending,false);
        assert.equal(solver.info.hostSchedulingUsesReadback,true);
        await solver.awaitFrameCompletion();await solver.readStats();
      }
      for(const field of ["volumeTexture","vertexPhiTexture"] as const) {
        const dense=await readTexture(device,solvers[0]![field]!);
        const pages=await readTexture(device,solvers[1]![field]!);
        let error=0;
        for(let i=0;i<dense.length;i++) {assert.ok(Number.isFinite(pages[i]));error=Math.max(error,Math.abs(dense[i]!-pages[i]!));}
        assert.equal(error,0,`${field} garden frame ${frame}`);
      }
      const info=solvers[1]!.info;
      console.log(JSON.stringify({scene:"hero-garden-hose",frame,edge:info.uniformVolumePageEdge,pages:info.uniformVolumePagesActive,total:info.uniformVolumePagesTotal,arenaBytes:info.uniformVolumePageBytes,denseArenaBytes:info.nx*info.ny*info.nz*80}));
    }
    assert.deepEqual(errors,[]);
  } finally {for(const s of solvers)s.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});


(modulePath?test:test.skip)("page size switches Mini32 between single and multiple pages without material field drift",{timeout:120000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform page-size single-page boundary");
 let device:GPUDevice|undefined;const solvers:WebGPUUniformReferenceSolver[]=[];
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
  for(const pageSize of ["32","16"])solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({pageSize},scene),()=>{}));
  assert.equal(solvers[0]!.info.uniformDomainPages,1);assert.equal(solvers[1]!.info.uniformDomainPages,8);
  const maxima={volumeTexture:0,vertexPhiTexture:0};
  for(let frame=1;frame<=5;frame++){
   for(const solver of solvers){assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();await solver.readStats();}
   for(const field of ["volumeTexture","vertexPhiTexture"] as const){
    const a=await readTexture(device,solvers[0]![field]!),b=await readTexture(device,solvers[1]![field]!);
    for(let i=0;i<a.length;i++){assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]));maxima[field]=Math.max(maxima[field],Math.abs(a[i]!-b[i]!));}
   }
  }
  // Single-page and native multi-page phi kernels compile differently. Bound
  // accumulated float-rounding drift to < 0.0001 cell and one tenth of the
  // configured 0.001-cell-volume dust floor. This is a cross-kernel comparison;
  // the existing same-kernel and Garden checks above remain bit-exact.
  assert.ok(maxima.vertexPhiTexture < .025*1e-4,JSON.stringify(maxima));
  assert.ok(maxima.volumeTexture < 1e-4,JSON.stringify(maxima));
  console.log(JSON.stringify({pageSizeBoundaryMaxima:maxima}));assert.deepEqual(errors,[]);
 }finally{for(const solver of solvers)solver.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
