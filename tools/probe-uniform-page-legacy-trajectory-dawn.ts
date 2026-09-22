/** Observational comparison against the retired direct-texture shader interface.
 * Float reassociation can change long trajectories. This probe reports mass,
 * centroid and interface differences; passing TAP only means the probe ran.
 * Run serially with WEBGPU_NODE_MODULE and node --import tsx --test.
 */
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
(modulePath?test:test.skip)("diagnose legacy versus page-operator trajectories (not a correctness gate)",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform page-first volume");
 let device:GPUDevice|undefined;const solvers:WebGPUUniformReferenceSolver[]=[];
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message); console.error(e.error.message);});
  const scene=sceneDocument(getSceneDefinition("hero-garden-hose"));
  for(const volumePageWork of [true,false])
   solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...uniformGeometricSolverOptions({volumeStorage:"pages32",pressureWindow:"domain"},scene),volumePageWork,pageDomain:volumePageWork},()=>{}));
  // A saved legacy setting must not re-enable readback-driven page scheduling.
  solvers[0]!.applyRuntimeValues({pressureCycleBudget:"lagged"});
  solvers[1]!.applyRuntimeValues({pressureCycleBudget:"fixed"});
  for(let frame=1;frame<=64;frame++){
   for(const solver of solvers){
    if(frame===41)solver.applyRuntimeValues({redistance:"off",pressureCycleBudget:"fixed"});
    if(frame===44)solver.applyRuntimeValues({redistance:"on",pressureCycleBudget:"fixed"});
    if(frame===33)solver.applySceneUniforms({...scene,fluid:{...scene.fluid,inflow:undefined}});
    if(frame===45)solver.applySceneUniforms({...scene,fluid:{...scene.fluid,inflow:{...scene.fluid.inflow!,center_m:{x:.15,y:.45,z:-.2}}}});
    if(frame===25)solver.injectLiquidBall({centre_m:{x:.65,y:.5,z:.35},radius_m:.04});
    assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();await solver.readStats();
   }
   if([1,12,24,25,32,40,43,44,45,64].includes(frame)||(process.env.PHI_DIAGNOSTIC&&frame>32)){
    for(const field of ["volumeTexture","advectedVertexPhiTexture","vertexPhiTexture","velocityTexture","surfaceFieldTexture"] as const){
     const a=await readTexture(device,solvers[0]![field]!),b=await readTexture(device,solvers[1]![field]!);
     let maxError=0,at=-1;
     for(let i=0;i<a.length;i++){
      assert.ok(Number.isFinite(a[i]) && Number.isFinite(b[i]));
      if(field==="vertexPhiTexture" || field==="advectedVertexPhiTexture"){}
      const delta=Math.abs(a[i]!-b[i]!);
      if(delta>maxError){maxError=delta;at=i;}
     }
     console.log(JSON.stringify({frame,field,maxError,...(process.env.PHI_DIAGNOSTIC?{at,a:a[at],b:b[at],groups:solvers.map(s=>({g:(s as any).windowVertexGroups,lag:(s as any).windowLagged}))}:{})}));
     if(field==="vertexPhiTexture"){
      const h=scene.container.width_m/solvers[0]!.info.nx;let count=0,error=0,max=0;
      for(let i=0;i<a.length;i++)if(Math.min(Math.abs(a[i]!),Math.abs(b[i]!))<h){count++;const e=Math.abs(a[i]!-b[i]!)/h;error+=e;max=Math.max(max,e);}
      console.log(JSON.stringify({frame,interfaceMeanCells:error/count,interfaceMaxCells:max}));
     }
     if(field==="volumeTexture"){
      let l1=0,total=0;for(let i=0;i<a.length;i++){l1+=Math.abs(a[i]!-b[i]!);total+=Math.abs(b[i]!);}
      const moments=(v:Float32Array)=>{let mass=0;const centre=[0,0,0];const {nx,ny}=solvers[0]!.info;for(let i=0;i<v.length;i++){mass+=v[i]!;centre[0]+=v[i]!*(i%nx);centre[1]+=v[i]!*(Math.floor(i/nx)%ny);centre[2]+=v[i]!*Math.floor(i/(nx*ny));}return {mass,centre:centre.map(n=>n/mass)};};
      console.log(JSON.stringify({frame,relativeVolumeL1:l1/Math.max(total,1e-30),paged:moments(a),reference:moments(b)}));
     }
    }

   }
  }
  for(const solver of solvers){
   assert.equal(solver.info.hostSchedulingUsesReadback,false);
   assert.equal(solver.solveWindowSource,undefined);
  }
  assert.equal(solvers[0]!.info.uniformDomainAuthority,"pages");
  assert.equal(solvers[0]!.info.uniformPressureCycleBudget,"fixed");
  assert.equal(solvers[0]!.info.uniformPressureCyclesEncoded,solvers[0]!.info.uniformPressureCyclesConfigured);
  assert.equal(solvers[1]!.info.uniformDomainAuthority,undefined);

  assert.deepEqual(errors,[]);
 }finally{for(const s of solvers)s.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
