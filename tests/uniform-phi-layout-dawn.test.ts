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
(modulePath?test:test.skip)("native phi kernels agree with atlas on frozen identical inputs",{timeout:360000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","native phi layout equivalence");let device:GPUDevice|undefined;
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const layout of (process.env.PHI_LAYOUT?[process.env.PHI_LAYOUT]:["atlas","window"]))for(const fixture of ["partial-pages","long-dam"]){
   const scene=structuredClone(sceneDocument(getSceneDefinition("sparse-cm12-long-dam-break")));
   if(fixture==='partial-pages'){
    scene.container.width_m=1;scene.container.height_m=.5;scene.container.depth_m=.4;
    scene.voxelDomain.finestCellSize_m=.025;
    scene.fluid.initialDamBreakDimensions_m={x:.25,y:.25,z:.4};
    scene.solidVoxels=[...solidVoxelShellForScene(scene)];
   }
   const solvers:WebGPUUniformReferenceSolver[]=[];
   try {
    for(const arm of [0,1]){
     const phiStorageForQA=layout==="atlas"&&arm===1?"paged":undefined;
     solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
      ...uniformGeometricSolverOptions({},scene),phiStorageForQA,...(layout==="window"&&arm===0?{}:{phiWindowForQA:false as const}),pressureCycleDispatch:"direct",pressureCycleBudget:"fixed",
      pressureSchedule:{fullCycles:1,vCycles:0,preSweeps:6,postSweeps:6,residualTolerance:0},
     },()=>{}));
    }
    const native=solvers[0] as any, atlas=solvers[1] as any;
    const logicalRead=async(solver:any,texture:GPUTexture)=>{
     const field=solver.fieldPages.fields.get(texture), [nx,ny,nz]=field.dims;
     const raw=new Float32Array((await read(device!,texture)).buffer);
     const components=texture.format==='rgba32float'?4:1;
     const row=Math.ceil(texture.width*components*4/256)*64;
     const result=new Float32Array(nx*ny*nz*components);
     for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
      let px=x,py=y,pz=z;
      if(field.paged){const page=Math.floor(x/16)+Math.ceil(nx/16)*(Math.floor(y/16)+Math.ceil(ny/16)*Math.floor(z/16));
       const ax=texture.width/16,ay=texture.height/16;
       px=x%16+16*(page%ax);py=y%16+16*(Math.floor(page/ax)%ay);pz=z%16+16*Math.floor(page/(ax*ay));}
      for(let c=0;c<components;c++)result[(x+nx*(y+ny*z))*components+c]=raw[px*components+row*(py+texture.height*pz)+c]!;
     }
     return result;
    };
    let maximumError=0,minimumWindowVertices=Infinity;
    for(let frame=1;frame<=12;frame++){
     if(frame===7)solvers[1]!.injectLiquidBall({centre_m:{x:0,y:.3,z:0},radius_m:.05});
     assert.ok(solvers[1]!.advanceTo(frame/30));await solvers[1]!.awaitFrameCompletion();
     const encoder=device.createCommandEncoder();
     const nativeFields=new Map([...native.fieldPages.fields.keys()].map((t:any)=>[t.label,t]));
     let originalPhi:Float32Array|undefined;
     for(const [texture,metadata] of atlas.fieldPages.fields as Map<GPUTexture,any>){
      const target=nativeFields.get(texture.label) as GPUTexture;
      if(!(target.usage & GPUTextureUsage.COPY_DST))continue;
      if(texture===atlas.vertexPhiField)originalPhi=await logicalRead(atlas,texture);
      if(texture.width===target.width&&texture.height===target.height&&texture.depthOrArrayLayers===target.depthOrArrayLayers)
       encoder.copyTextureToTexture({texture},{texture:target},[texture.width,texture.height,texture.depthOrArrayLayers]);
      else {const values=await logicalRead(atlas,texture);native.fieldPages.upload(target,values);
       if(texture===atlas.vertexPhiField)originalPhi=values;}
     }
     for(const key of ['params','activeRegion','boundaryVelocityA','boundaryVelocityB'])
      encoder.copyBufferToBuffer(atlas[key],0,native[key],0,atlas[key].size);
     // Phi samples the shared three conditioning planes (coarse velocity and
     // tile classes). The native balance/page-record tail is smaller and is
     // neither an input to these kernels nor layout-compatible with the atlas.
     encoder.copyBufferToBuffer(atlas.conditioningScratch,0,native.conditioningScratch,0,
      native.info.nx*native.info.ny*native.info.nz*3*4);
     encoder.copyBufferToBuffer(atlas.activeScratch,0,native.activeScratch,0,1024);
     encoder.copyBufferToBuffer(atlas.activeScratch,atlas.solidVoxelScratchOffsetWords*4,native.activeScratch,native.solidVoxelScratchOffsetWords*4,
      atlas.activeScratch.size-atlas.solidVoxelScratchOffsetWords*4);
     device.queue.submit([encoder.finish()]);
     device.queue.writeBuffer(native.params,42*4,new Float32Array([native.solidVoxelScratchOffsetWords]));
     for(const key of ['velocityA','transportA','volumeA','vertexPhiField','gammaA']){
      const a=await logicalRead(native,native[key]),b=await logicalRead(atlas,atlas[key]);
      assert.equal(Buffer.compare(Buffer.from(a.buffer),Buffer.from(b.buffer)),0,`identical input ${key} frame ${frame}`);
     }
     await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);
     let bounds:Uint32Array|undefined;
     if(layout==='window'){
      if(frame===10)native.applyRuntimeValues({volumeDustThreshold:0});
      if(frame===11)native.applyRuntimeValues({volumeDustThreshold:uniformGeometricSolverOptions({},scene).volumeDustThreshold!});
      const e=device.createCommandEncoder();native.encodePhiRegion(e);device.queue.submit([e.finish()]);
      const b:GPUBuffer=device.createBuffer({size:1024,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const copy=device.createCommandEncoder();copy.copyBufferToBuffer(native.phiRegion,0,b,0,1024);device.queue.submit([copy.finish()]);
      await b.mapAsync(GPUMapMode.READ);bounds=new Uint32Array(b.getMappedRange().slice(0));b.unmap();b.destroy();
      if(frame===10)assert.deepEqual(Array.from(bounds.subarray(7,13)),[0,0,0,native.info.nx,native.info.ny,native.info.nz],"live zero dust floor uses full-domain fallback");
      minimumWindowVertices=Math.min(minimumWindowVertices,[0,1,2].reduce((n,a)=>n*(bounds![10+a]!-bounds![7+a]!+1),1));
     }
     for(const [entry,group,field] of [
      ['uvAdvectPhi','densityTraceGroup','vertexPhiScratch'],
      ['uvRedistancePhi','phiReverseGroup','vertexPhiField'],
     ]){
      for(const solver of [native,atlas]){const e=device.createCommandEncoder();
       solver.runVertex(e,entry,solver.volumePipelines[entry],solver[group]);device.queue.submit([e.finish()]);}
      const expected=await logicalRead(atlas,atlas[field]);const actual=await logicalRead(native,native[field]);
      let maxError=0;const [nx,ny,nz]=[native.info.nx+1,native.info.ny+1,native.info.nz+1];
      for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
       if(bounds&&[x,y,z].some((v,a)=>v<bounds![7+a]!||v>bounds![10+a]!))continue;
       const i=x+nx*(y+ny*z);maxError=Math.max(maxError,Math.abs(expected[i]!-actual[i]!));
      }
      maximumError=Math.max(maximumError,maxError);
      // Compiled advection may fuse arithmetic differently: allow 1/10000 cell.
      // The iterative redistance oracle remains bounded at one micrometre.
      const tolerance=entry==='uvAdvectPhi'?Math.min(scene.container.width_m/native.info.nx,scene.container.height_m/native.info.ny,scene.container.depth_m/native.info.nz)*1e-4:1e-6;
      assert.ok(Number.isFinite(maxError)&&maxError<=tolerance,`${layout} ${fixture} ${entry} frame ${frame}: max error ${maxError}`);
      if(layout==='atlas'&&entry==='uvAdvectPhi')native.fieldPages.upload(native.vertexPhiScratch,expected);
     }
     // Preserve the production oracle's trajectory after the isolated replay.
     atlas.fieldPages.upload(atlas.vertexPhiField,originalPhi!);
    }
    if(layout==='window'&&fixture==='long-dam')assert.ok(minimumWindowVertices<(native.info.nx+1)*(native.info.ny+1)*(native.info.nz+1));
    console.log(JSON.stringify({layout,fixture,verifiedFrames:12,maximumError,minimumWindowVertices}));
   }finally{for(const solver of solvers)solver.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
