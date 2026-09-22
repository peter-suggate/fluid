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
   if([1,2,5,24,30,40].includes(frame)){
    // The simulation state itself is bit-for-bit. The two colour passes
    // commute within each colour, and skipping tiles only after their air
    // constraints have been projected leaves every liquid row untouched, so
    // any pressure that reaches the projection has to survive this unchanged.
    for(const field of ["volumeTexture","velocityTexture","vertexPhiTexture"] as const){
     const a=await readTexture(device,solvers[0]![field]!);
     const b=await readTexture(device,solvers[1]![field]!);
     assert.equal(a.length,b.length);
     const aBits=new Uint32Array(a.buffer),bBits=new Uint32Array(b.buffer);
     for(let i=0;i<a.length;i++)assert.ok(Number.isFinite(a[i])&&aBits[i]===bBits[i],`${field} frame ${frame} index ${i}: ${a[i]} != ${b[i]}`);
    }
    // Pressure is bit-for-bit on every row the cycle list carries: all liquid
    // and all constrained rows, plus a tile of margin. Outside it the tiled
    // arm keeps the exact zero mgBuildFinestRhs stored, where the dense arm
    // accumulates the trilinear prolongation of COARSE air rows -- noise
    // eleven orders below the field. Those rows are dead in both arms: the
    // smoother, mgApply and mgResidual mask non-liquid neighbours, the p_min
    // downsample saturates at -FLT_MAX whatever p is, and the projection
    // reads air pressure as 0. So a difference is allowed only where the
    // tiled arm is still that zero, and only at far-field magnitude.
    const a=await readTexture(device,solvers[0]!.gridPressureTexture);
    const b=await readTexture(device,solvers[1]!.gridPressureTexture);
    assert.equal(a.length,b.length);
    const aBits=new Uint32Array(a.buffer),bBits=new Uint32Array(b.buffer);
    let scale=0;for(let i=0;i<a.length;i++)scale=Math.max(scale,Math.abs(a[i]!));
    let denseOnly=0,worst=0,worstIndex=-1;
    for(let i=0;i<a.length;i++){
     assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]),`pressure frame ${frame} index ${i}: ${a[i]} / ${b[i]}`);
     if(aBits[i]===bBits[i])continue;
     denseOnly+=1;
     assert.ok(b[i]===0,`pressure frame ${frame} index ${i}: tiled ${b[i]} != dense ${a[i]} on a row the cycle list carries`);
     if(Math.abs(a[i]!)>worst){worst=Math.abs(a[i]!);worstIndex=i;}
    }
    // The strongest pressure in the field is always on a row both arms agree
    // on. A dense-only row carrying the maximum would mean the list had
    // dropped a live row, not far field.
    assert.ok(scale===0||worst<scale,`dense-only pressure ${worst} at ${worstIndex} is the field maximum ${scale}`);
    console.log(JSON.stringify({frame,cells:a.length,denseOnlyCells:denseOnly,denseOnlyMax:worst,fieldMax:scale}));
   }
  }
  assert.deepEqual(errors,[]);
 } finally {for(const solver of solvers)solver.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
