import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sampleSolidWorld, solidWorldForScene } from "../lib/core/solid-world";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";

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

// phi inside a solid is not solver state (uvBuried). The hose fill once pulled
// the terrain's buried vertex planes into the redistance band through the
// source plug, the redistance walked the contour into the terrain one plane a
// step, and the published surface crossed 0.5 inside closed cells: the water
// was drawn below the pond floor. Both halves are pinned here on the scene
// that showed it, with the method's defaults.
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("hero garden hose never publishes liquid inside terrain",{timeout:400000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","garden buried phi");
 let device:GPUDevice|undefined;let solver:WebGPUUniformReferenceSolver|undefined;
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=sceneDocument(getSceneDefinition("hero-garden-hose"));
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...uniformGeometricSolverOptions({},scene)},()=>{});
  const {nx,ny,nz}=solver.info;
  const c=scene.container;const band=4*Math.min(c.width_m/nx,c.height_m/ny,c.depth_m/nz);
  const world=solidWorldForScene(scene);
  const closed=new Uint8Array(nx*ny*nz);let closedCells=0;
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const solid=sampleSolidWorld(world,[x,y,z]).solidFraction>0;closed[x+nx*(y+ny*z)]=solid?1:0;if(solid)closedCells++;}
  assert.ok(closedCells>0,"the garden terrain must close cells");
  const buried:number[]=[];
  for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++){
   let all=true,any=false;
   for(const cz of [z-1,z])for(const cy of [y-1,y])for(const cx of [x-1,x]){
    if(cx<0||cy<0||cz<0||cx>=nx||cy>=ny||cz>=nz)continue;any=true;if(!closed[cx+nx*(cy+ny*cz)])all=false;}
   if(any&&all)buried.push(x+(nx+1)*(y+(ny+1)*z));
  }
  assert.ok(buried.length>0);
  for(let frame=1;frame<=60;frame++){
   assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();await solver.readStats();
   if(frame!==1&&frame%10!==0)continue;
   const phi=await readTexture(device,solver.vertexPhiTexture!);const surface=await readTexture(device,solver.surfaceFieldTexture!);
   let belowBand=0,minBuried=Infinity;
   for(const i of buried){const v=phi[i]!;assert.ok(Number.isFinite(v));minBuried=Math.min(minBuried,v);if(v<band)belowBand++;}
   let closedWet=0,wetCells=0;
   for(let i=0;i<nx*ny*nz;i++){const wet=surface[i]!>=0.5;if(wet)wetCells++;if(wet&&closed[i])closedWet++;}
   console.log(JSON.stringify({frame,volume:solver.info.volumeCellSum,buried:buried.length,belowBand,minBuried,wetCells,closedWet}));
   assert.equal(belowBand,0,`frame ${frame}: ${belowBand} buried vertices entered the redistance band (min ${minBuried} m)`);
   assert.equal(closedWet,0,`frame ${frame}: the published surface crosses 0.5 inside ${closedWet} closed cells`);
   if(frame===60)assert.ok(wetCells>100,"the hose must have filled some of the pond");
  }
  assert.deepEqual(errors,[]);
 } finally {solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
