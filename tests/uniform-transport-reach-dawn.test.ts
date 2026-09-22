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
const steps=Number(process.env.UNIFORM_TRANSPORT_REACH_STEPS ?? 60);
/**
 * E7 ALONE, against the domain-wide reach, demanding bit equality.
 *
 * The far-air lane compares all four host certificates at once at 64³ with a
 * 1e-5 tolerance, which is too coarse and too small to price the one thing
 * that can go wrong with a per-tile reach: a receiver whose trace departs
 * further than its OWN neighbourhood's displacement predicted, because the
 * velocity a far-air cell traces with is not a start-of-step fine velocity at
 * all -- it is whatever Sec. 3.3's extension copied there. Nothing about the
 * tile it sits in bounds that.
 *
 * So this lane isolates E7 with `transportReachPerTileForQA` (the other three
 * certificates are on in BOTH arms, so they cancel), runs figure 7 at its
 * authored 128³ through the floor impact and the spreading sheet, and asserts
 * that every compared element is EQUAL, not close. The narrowed set is a
 * subset of the domain-wide one by construction, so any difference at all is
 * liquid the restricted arm refused to move. The experiment is re-seeded from
 * the control after each comparison, so a failure names the first step whose
 * own transport differs rather than the first step an earlier rounding grew
 * large enough to see.
 *
 * The lane also asserts the set really shrinks, so that agreement means the
 * restriction is exact rather than absent.
 */
(modulePath?test:test.skip)("the per-tile transport reach reproduces the domain-wide reach exactly",{timeout:2400000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform Geometric per-tile transport reach");
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
  for(const transportReachPerTileForQA of [false,true] as const){
   solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
    ...uniformGeometricSolverOptions({},scene),transportReachPerTileForQA,pressureCycleBudget:"fixed",
   },()=>{}));
  }
  assert.deepEqual([solvers[0]!.info.nx,solvers[0]!.info.ny,solvers[0]!.info.nz],[128,128,128]);
  // Volume is the transported quantity, phi decides the set's own seeds, and
  // the projected velocity is what the next step's reach is measured from.
  const fields=["volumeTexture","vertexPhiTexture","velocityTexture"] as const;
  const failures:string[]=[];
  let shrunkSteps=0;
  for(let frame=1;frame<=steps;frame++){
   for(const solver of solvers){assert.ok(solver.advanceTo(frame/30,[]));await solver.awaitFrameCompletion();}
   const control=await solvers[0]!.readStats(),narrow=await solvers[1]!.readStats();
   const scalars=(info:typeof control)=>({
    volumeCellSum:info.volumeCellSum,representedVolumeCellSum:info.representedVolumeCellSum,
    maxSpeed_m_s:info.maxSpeed_m_s,front_m:info.front_m,
   });
   if(JSON.stringify(scalars(control))!==JSON.stringify(scalars(narrow)))
    failures.push(`frame ${frame} telemetry ${JSON.stringify(scalars(control))} vs ${JSON.stringify(scalars(narrow))}`);
   const tiles=(info:typeof control)=>(info as unknown as Record<string,number|undefined>).uniformTransportTiles!;
   assert.equal(typeof tiles(control),"number","transport tile telemetry is populated");
   assert.ok(tiles(narrow)<=tiles(control),`per-tile reach never grows the live set (frame ${frame})`);
   if(tiles(narrow)<tiles(control))shrunkSteps+=1;
   for(const field of fields){
    const a=await readTexture(device,solvers[0]![field]!);
    const b=await readTexture(device,solvers[1]![field]!);
    assert.equal(a.length,b.length,`${field} retains its logical dimensions`);
    let differing=0,first=-1,maxError=0;
    for(let i=0;i<a.length;i++){
     if(a[i]!==b[i]){differing+=1;if(first<0)first=i;maxError=Math.max(maxError,Math.abs(a[i]!-b[i]!));}
    }
    if(differing>0){
     console.log(JSON.stringify({frame,field,differing,first,maxError,
      control:a[first],narrowed:b[first],
      controlTiles:tiles(control),narrowedTiles:tiles(narrow)}));
     failures.push(`${field} differs in ${differing} elements at frame ${frame} (max ${maxError})`);
    }
   }
   if(failures.length>0)break;
   // Compare the step, then hand the control's state to the experiment so the
   // next comparison starts from identical physical input.
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
  assert.deepEqual(failures,[],"the per-tile reach reproduces the domain-wide transport bit for bit");
  assert.ok(shrunkSteps>0,"the per-tile reach shrank the live transport set on at least one step");
  console.log(JSON.stringify({steps,shrunkSteps}));
  assert.deepEqual(errors,[]);
 } finally {
  for(const solver of solvers)solver.destroy();
  device?.destroy();
  releaseWebGPUExclusiveLock();
 }
});
