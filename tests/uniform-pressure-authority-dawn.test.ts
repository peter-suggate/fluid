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
import { createBodyDescription, initializeRigidBodies } from "../lib/core/rigid-body";
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
(modulePath?test:test.skip)("cooperative tile census and pressure authority reuse match serial/raw queries through terrain, edits and moving bodies",{timeout:1200000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform pressure authority reuse");
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
  // Partial atlas pages exercise shared scratch while multiples of four keep
  // the two-level census enabled (partial cell tiles deliberately disable it).
  scene.container.width_m+=4*scene.voxelDomain.finestCellSize_m;
  scene.container.depth_m-=4*scene.voxelDomain.finestCellSize_m;
  scene.solidVoxels=[...solidVoxelShellForScene(scene)];
  scene.terrain={baseHeight_m:2.25*scene.voxelDomain.finestCellSize_m,features:[]};
  for(const pressureAuthorityForQA of ["raw",undefined] as const){
   const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
    ...uniformGeometricSolverOptions({},scene),pressureAuthorityForQA,tileSeedForQA:pressureAuthorityForQA === "raw" ? "serial" : undefined,
    pressureCycleBudget:"fixed",
   },()=>{});
   solvers.push(solver);
   assert.deepEqual([solver.info.nx,solver.info.ny,solver.info.nz],[68,64,60]);
  }

  assert.equal((solvers[0] as unknown as {cooperativeTileSeed:boolean}).cooperativeTileSeed,false);
  assert.equal((solvers[1] as unknown as {cooperativeTileSeed:boolean}).cooperativeTileSeed,true);
  const fields=["volumeTexture","velocityTexture","gridPressureTexture","extrapolatedVelocityTexture","vertexPhiTexture","surfaceFieldTexture",
   "preProjectionVelocityTexture","advectedVertexPhiTexture"] as const;
  const failures:string[]=[];
  for(let frame=1;frame<=12;frame++){
   for(const solver of solvers){
    if(frame===3)solver.applySceneUniforms({...scene,solidVoxels:[...scene.solidVoxels,{operation:"fill",minimum:[30,0,0],maximumExclusive:[32,solver.info.ny,solver.info.nz]}]});
    if(frame===6)solver.applySceneUniforms(scene);
    const bodies=frame>=8&&frame<=9 ? initializeRigidBodies([{...createBodyDescription("sphere",1,scene.container.height_m),
      dimensions_m:{x:scene.container.width_m/8,y:scene.container.width_m/8,z:scene.container.width_m/8},
      position_m:{x:(frame-8)*scene.container.width_m/32,y:scene.container.height_m/2,z:0},
      linearVelocity_m_s:{x:.1,y:0,z:0}}]) : [];
    assert.ok(solver.advanceTo(frame/30,bodies));await solver.awaitFrameCompletion();
   }
   const info=await solvers[0]!.readStats(),cachedInfo=await solvers[1]!.readStats();
   assert.ok(Number.isInteger(info.uniformTwoLevelFineTiles),"tile telemetry is populated");
   assert.equal(cachedInfo.uniformTwoLevelFineTiles,info.uniformTwoLevelFineTiles,`fine tile classes frame ${frame}`);
   assert.equal((cachedInfo as typeof cachedInfo & {uniformTwoLevelShellTiles?:number}).uniformTwoLevelShellTiles,
    (info as typeof info & {uniformTwoLevelShellTiles?:number}).uniformTwoLevelShellTiles,`shell tile classes frame ${frame}`);
   const workKey="uniformTransportMaxDisplacement_cells";
   assert.equal(typeof (info as unknown as Record<string,unknown>)[workKey],"number","travel telemetry is populated");
   assert.equal((cachedInfo as unknown as Record<string,unknown>)[workKey],
    (info as unknown as Record<string,unknown>)[workKey],`transport reach frame ${frame}`);
   {
   for(const field of fields){
    const a=await readTexture(device,solvers[0]![field]!);
    const b=await readTexture(device,solvers[1]![field]!);
    assert.equal(a.length,b.length,`${field} retains its logical dimensions`);
    let squared=0,scale=0,maxError=0,maxValue=0;
    for(let i=0;i<a.length;i++){
     assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]),`${field} finite at ${i}`);
     const delta=Math.abs(a[i]!-b[i]!);squared+=delta*delta;scale+=a[i]!*a[i]!;
     maxError=Math.max(maxError,delta);maxValue=Math.max(maxValue,Math.abs(a[i]!));
    }
    const relativeL2=Math.sqrt(squared/Math.max(scale,1e-20));
    console.log(JSON.stringify({frame,field,maxError,relativeL2}));
    // Shader specialization can change FP32 contraction. Compare each step
    // on identical physical input, comparing rebuilt and reused pressure data.
    // Free-fall pressure is nearly zero. Use a 1 mPa absolute floor there;
    // relative error alone is ill-conditioned for that field.
    const absoluteTolerance=field==="gridPressureTexture" ? 1e-3 : 1e-6;
    if(!(relativeL2<1e-5 || maxError<absoluteTolerance))failures.push(`${field} relative error ${relativeL2} frame ${frame}`);
    if(!(maxError<Math.max(1e-5,absoluteTolerance,maxValue*1e-4)))failures.push(`${field} maximum error ${maxError} frame ${frame}`);
   }
   }
   // Use the control trajectory as the next step's physical input. This
   // isolates pressure input reuse from amplification of f32 roundoff by
   // subsequent nonlinear steps. The 256³ profiler separately records an
   // uninterrupted production trajectory and mass drift.
   const encoder=device.createCommandEncoder();
   const physicalFields=["velocityA","velocityB","velocityD","volumeA","volumeB",
    "gammaA","gammaB","surfaceA","surfaceB","vertexPhiField","vertexPhiScratch","transportA","gridPressureTexture"];
   const control=solvers[0] as unknown as Record<string,GPUTexture>;
   for(const solver of solvers.slice(1))for(const name of physicalFields){
    const source=control[name]!,destination=(solver as unknown as Record<string,GPUTexture>)[name]!;
    encoder.copyTextureToTexture({texture:source},{texture:destination},[source.width,source.height,source.depthOrArrayLayers]);
   }
   device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
  }
  assert.deepEqual(errors,[]);
  assert.deepEqual(failures,[]);
 } finally {for(const solver of solvers)solver.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
