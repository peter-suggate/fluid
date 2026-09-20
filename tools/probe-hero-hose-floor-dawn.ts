import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { writeFileSync, mkdirSync } from 'node:fs';
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { uniformVolumeMethod } from '../lib/methods/uniform/uniform-volume-method';
import { resolveMethodValues } from '../lib/core/method-contract';
import { sampleSolidWorld, solidWorldForScene } from '../lib/core/solid-world';
import { terrainColumnHeights } from '../lib/core/terrain';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
function tetra(v:number[]):number {
 const n=v.filter(x=>x<0),o=v.filter(x=>x>=0);
 if(n.length===0)return 0;if(n.length===4)return 1;
 if(n.length===1)return o.reduce((p,x)=>p*(-n[0]/(x-n[0])),1);
 if(n.length===3)return 1-n.reduce((p,x)=>p*o[0]/(o[0]-x),1);
 const a=-n[0]/(o[0]-n[0]),b=-n[0]/(o[1]-n[0]),c=-n[1]/(o[0]-n[1]),d=-n[1]/(o[1]-n[1]);
 return Math.max(0,Math.min(1,a*b+b*c*(1-a)+c*d*(1-b)));
}
const tets=[[0,1,3,7],[0,1,5,7],[0,2,3,7],[0,2,6,7],[0,4,5,7],[0,4,6,7]];
const arg=(name:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${name}=`))?.slice(name.length+3)??fallback;
await acquireWebGPUExclusiveLock('dawn-probe','hero hose floor');
let device:GPUDevice|undefined, solver:WebGPUUniformReferenceSolver|undefined;
try {
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href) as NodeDawnProvider;
 Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);
 const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition('hero-garden-hose'));
 const overrides=JSON.parse(arg('values','{}'));
 const values=resolveMethodValues(uniformVolumeMethod,'balanced',overrides);
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{},new AbortController().signal) as WebGPUUniformReferenceSolver;
 const [nx,ny,nz]=[solver.info.nx,solver.info.ny,solver.info.nz];
 const h=scene.container.height_m/ny;
 const world=solidWorldForScene(scene), terrain=terrainColumnHeights(scene,nx,nz);
 const solid=new Uint8Array(nx*ny*nz);
 for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)solid[x+nx*(y+ny*z)]=Number(sampleSolidWorld(world,[x,y,z]).solidFraction>0);
 async function read(texture:GPUTexture){
  const row=Math.ceil(texture.width*4/256)*256;
  const buffer=device!.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const encoder=device!.createCommandEncoder();encoder.copyTextureToBuffer({texture},{buffer,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);device!.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);const data=new Float32Array(buffer.getMappedRange().slice(0));buffer.unmap();buffer.destroy();
  return (x:number,y:number,z:number)=>data[(z*texture.height+y)*row/4+x]!;
 }
 const samples:unknown[]=[];
 console.log(JSON.stringify({phase:'ready',dimensions:[nx,ny,nz],values}));
 const frames=Number(arg('frames','90'));
 for(let frame=0;frame<=frames;frame++){
  if(frame){while(!solver.advanceTo(frame/30,[]))await new Promise(setImmediate);await device.queue.onSubmittedWorkDone();}
  const stats=await solver.readStats();
  if(frame>12&&frame%10!==0&&frame!==frames)continue;
  const v=await read(solver.volumeTexture),phi=await read(solver.vertexPhiTexture!);
  let geometricSurface=0;
  let total=0,insideSolid=0,belowTerrain=0,orphaned=0,excess=0,max=0,nearFloor=0;let maxCell:number[]=[];
  const rows=new Array(ny).fill(0);
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
   const value=v(x,y,z);if(!Number.isFinite(value))throw new Error('nonfinite volume');total+=value;rows[y]+=value;
   const floor=terrain[x+nx*z]/h;
   const open=solid[x+nx*(y+ny*z)]?0:1-Math.max(0,Math.min(1,floor-y));
   if(frame===frames&&open>0){
    const corners=Array.from({length:8},(_,k)=>phi(x+(k&1),y+((k>>1)&1),z+(k>>2)));
    if(corners.some(p=>p<0))geometricSurface+=open*(corners.every(p=>p<0)?1:tets.reduce((s,t)=>s+tetra(t.map(k=>corners[k])),0)/6);
   }
   if(value<=1e-8)continue;
   if(open===0)insideSolid+=value;
   if(y+.5<floor)belowTerrain+=value;
   if(y>=floor&&y<floor+2)nearFloor+=value;
   excess+=Math.max(0,value-open);
   let p=0;for(let k=0;k<8;k++)p+=phi(x+(k&1),y+((k>>1)&1),z+(k>>2))/8;
   if(p>=0)orphaned+=value;
   if(value>max){max=value;maxCell=[x,y,z];}
  }
  const sample={geometricSurface:frame===frames?geometricSurface:undefined,frame,time:frame/30,total,reference:stats.referenceLiquidVolume_cells,represented:stats.representedVolumeCellSum,insideSolid,belowTerrain,orphaned,excess,max,maxCell,nearFloor,rows};
  samples.push(sample);console.log(JSON.stringify(sample));
 }
 assert.deepEqual(errors,[]);
 const out=arg('out','artifacts/hero-hose-floor/baseline.json');mkdirSync(resolve(out,'..'),{recursive:true});writeFileSync(out,JSON.stringify({scene:scene.sceneId,dimensions:[nx,ny,nz],values,samples},null,2));
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
