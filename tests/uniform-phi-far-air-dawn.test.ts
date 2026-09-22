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
/**
 * E4-E7: the four host certificates of the geometric phi and transport stages
 * against the unconditional arm, on figure 7 through floor impact.
 *
 * The scene is deliberately the one the certificates are written for -- a dry
 * closed tank with no solid voxel, no terrain and no body, so `uvSolidFree`
 * holds and the solid walks, the embedded-wall terms and the far-air arm are
 * all supposed to be identity rather than approximations. Every field is
 * compared each step, and the experiment is re-seeded from the control after
 * the comparison, so what is measured is the step's own difference rather than
 * sixty steps of a nonlinear system amplifying one rounding.
 *
 * The transport tile counts are compared too, in the other direction: the
 * per-tile reach must actually SHRINK the live set once the splash makes the
 * domain-maximum displacement unrepresentative, or the experiment is doing
 * nothing and the field agreement means nothing.
 */
(modulePath?test:test.skip)("far-air phi arm, solid-free trace, windowed census and per-tile reach reproduce the unconditional arm",{timeout:1200000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform Geometric far-air phi arm");
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
  for(const leanPhiArmsForQA of ["unconditional",undefined] as const){
   solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
    ...uniformGeometricSolverOptions({},scene),leanPhiArmsForQA,pressureCycleBudget:"fixed",
   },()=>{}));
  }
  assert.deepEqual([solvers[0]!.info.nx,solvers[0]!.info.ny,solvers[0]!.info.nz],[64,64,64]);
  const fields=["volumeTexture","velocityTexture","vertexPhiTexture","surfaceFieldTexture",
   "extrapolatedVelocityTexture","advectedVertexPhiTexture"] as const;
  const failures:string[]=[];
  let shrunkSteps=0;
  // Thirty steps at this resolution carries the ball through free fall, the
  // floor impact and the first of the spreading sheet -- the window in which
  // the domain-maximum displacement and the local one diverge.
  for(let frame=1;frame<=30;frame++){
   for(const solver of solvers){assert.ok(solver.advanceTo(frame/30,[]));await solver.awaitFrameCompletion();}
   const control=await solvers[0]!.readStats(),lean=await solvers[1]!.readStats();
   const tiles=(info:typeof control)=>(info as unknown as Record<string,number|undefined>).uniformTransportTiles;
   assert.equal(typeof tiles(control),"number","transport tile telemetry is populated");
   assert.ok(tiles(lean)!<=tiles(control)!,`per-tile reach never grows the live set (frame ${frame})`);
   if(tiles(lean)!<tiles(control)!)shrunkSteps+=1;
   for(const field of fields){
    const a=await readTexture(device,solvers[0]![field]!);
    const b=await readTexture(device,solvers[1]![field]!);
    assert.equal(a.length,b.length,`${field} retains its logical dimensions`);
    let squared=0,scale=0,maxError=0,maxValue=0;
    for(let i=0;i<a.length;i++){
     assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]),`${field} finite at ${i} frame ${frame}`);
     const delta=Math.abs(a[i]!-b[i]!);squared+=delta*delta;scale+=a[i]!*a[i]!;
     maxError=Math.max(maxError,delta);maxValue=Math.max(maxValue,Math.abs(a[i]!));
    }
    const relativeL2=Math.sqrt(squared/Math.max(scale,1e-20));
    if(maxError>0)console.log(JSON.stringify({frame,field,maxError,relativeL2}));
    if(!(relativeL2<1e-5||maxError<1e-6))failures.push(`${field} relative error ${relativeL2} frame ${frame}`);
    if(!(maxError<Math.max(1e-5,maxValue*1e-4)))failures.push(`${field} maximum error ${maxError} frame ${frame}`);
   }
   // Same isolation the system/extension lane uses: compare the step, then
   // hand the control's state to the experiment so the next comparison starts
   // from identical physical input.
   const encoder=device.createCommandEncoder();
   const physicalFields=["velocityA","velocityB","velocityD","volumeA","volumeB",
    "gammaA","gammaB","surfaceA","surfaceB","vertexPhiField","vertexPhiScratch","transportA","gridPressureTexture"];
   const source=solvers[0] as unknown as Record<string,GPUTexture>;
   for(const name of physicalFields){
    const from=source[name],to=(solvers[1] as unknown as Record<string,GPUTexture>)[name];
    if(!from||!to)continue;
    encoder.copyTextureToTexture({texture:from},{texture:to},[from.width,from.height,from.depthOrArrayLayers]);
   }
   device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
  }
  assert.deepEqual(failures,[],"the certified arms reproduce the unconditional fields");
  assert.ok(shrunkSteps>0,"the per-tile reach shrank the live transport set on at least one step");
  console.log(JSON.stringify({shrunkSteps}));
  assert.deepEqual(errors,[]);
 } finally {
  for(const solver of solvers)solver.destroy();
  device?.destroy();
  releaseWebGPUExclusiveLock();
 }
});
