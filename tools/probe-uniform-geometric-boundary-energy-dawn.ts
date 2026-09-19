/** Read-only Dawn stage census for figure 12. Energy fields are owner-volume
 * proxies, not the variational face kinetic energy or a stability assertion. */
import assert from "node:assert/strict";
import { mkdir,writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readBufferBinding,readFloatTexture3D,readRgbaTexture3D } from "../lib/harness/webgpu-smoke-readbacks";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { solidWorldForScene,sampleSolidWorld } from "../lib/core/solid-world";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const n=Number(arg("n","64")),steps=Number(arg("steps","45")),out=arg("out","/tmp/figure12-current");
process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT="1";
await acquireWebGPUExclusiveLock("dawn-probe","figure12 boundary energy");
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
try {
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
 const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=structuredClone(sceneDocument(getSceneDefinition(arg("scene","cm12-figure-12"))));
 scene.voxelDomain.finestCellSize_m=scene.container.width_m/n;scene.solidVoxels=[...solidVoxelShellForScene(scene)];
 const values=resolveMethodValues(uniformVolumeMethod,"balanced",JSON.parse(arg("values","{}")));
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 if(Number(arg("coarse","0"))>0)solver.enableCM11aCoarsestCapture(Number(arg("coarse","0")));
 const {nx,ny,nz}=solver.info;const world=solidWorldForScene(scene);const open=new Uint8Array(nx*ny*nz);
 for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)open[x+nx*(y+ny*z)]=sampleSolidWorld(world,[x,y,z]).solidFraction===0?1:0;
 await mkdir(out,{recursive:true});await writeFile(`${out}/config.json`,JSON.stringify({n,values,scene},null,2));
 const xyz=(i:number)=>[i%nx,Math.floor(i/nx)%ny,Math.floor(i/(nx*ny))];
 const rows:unknown[]=[];
 for(let frame=1;frame<=steps;frame++){
  if(frame===Number(arg("switch-frame","0")))solver.applyRuntimeValues({...values,...JSON.parse(arg("switch-values","{}"))});
  assert.ok(solver.advanceTo(frame/30));await device.queue.onSubmittedWorkDone();
  const a=solver.symmetryStageAuditTextures!;
  const volume=await readFloatTexture3D(device,solver.volumeTexture,nx,ny,nz);
  const stages:Record<string,unknown>={};let post:Float32Array|undefined,pre:Float32Array|undefined;
  for(const [name,texture] of [["previous",a.preExtrapolationVelocity],["advection",a.velocityAdvection],["projection",a.pressureProjection]] as const){
   const v=await readRgbaTexture3D(device,texture,nx,ny,nz);let max=0,at=0,axis=0,energy=0,allEnergy=0,solidEnergy=0;
   for(let i=0;i<volume.length;i++)for(let c=0;c<3;c++){
    const speed=v[4*i+c]!;if(Math.abs(speed)>max){max=Math.abs(speed);at=i;axis=c;}
    allEnergy+=speed*speed;energy+=volume[i]!*speed*speed;if(!open[i])solidEnergy+=speed*speed;
   }
   stages[name]={max,at:xyz(at),axis,solid:!open[at],energy,allEnergy,solidEnergy};
   if(name==="projection")post=v;if(name==="advection")pre=v;
  }
  const pressureTexture=solver.physicsFieldsForQA.pressure;
  const pressure=await readFloatTexture3D(device,pressureTexture,pressureTexture.width,pressureTexture.height,pressureTexture.depthOrArrayLayers);
  const phi=await readFloatTexture3D(device,solver.vertexPhiTexture!,nx+1,ny+1,nz+1);
  let minP=Infinity,maxP=-Infinity,mass=0,maxV=0,orphans=0;
  for(const p of pressure){minP=Math.min(minP,p);maxP=Math.max(maxP,p);}
  for(let i=0;i<volume.length;i++){
   const v=volume[i]!;mass+=v;maxV=Math.max(maxV,v);const [x,y,z]=xyz(i);let f=0;
   for(let k=0;k<8;k++)f+=phi[x!+(k&1)+(nx+1)*(y!+((k>>1)&1)+(ny+1)*(z!+(k>>2)))]!/8;
   if(v>.5&&f>=0)orphans++;
  }
  const boundary=await readBufferBinding(device,{buffer:solver.negativeBoundaryVelocityBuffer},solver.negativeBoundaryVelocityBytes);
  let negativeBoundaryMax=0;for(const u of new Float32Array(boundary.buffer,boundary.byteOffset,boundary.byteLength/4))negativeBoundaryMax=Math.max(negativeBoundaryMax,Math.abs(u));
  const stats=await solver.readStats();
  const pressureStats=Object.fromEntries(Object.entries(stats).filter(([key])=>/uniformCM11a|uniformPressure/.test(key)));
  const row={frame,t:frame/30,pressureStats,negativeBoundaryMax,latticeOrigin:solver.physicsFieldsForQA.latticeOrigin,latticeDimensions:solver.physicsFieldsForQA.latticeDimensions,mass,maxV,orphans,minP,maxP,stages,residual:stats.uniformCM11aFineResidualInfinity,cap:stats.uniformCM11aCapFailure,coarseResidual:stats.uniformCM11aResidualInfinity,cycles:stats.uniformPressureCyclesExecuted};
  rows.push(row);console.log(JSON.stringify(row));await writeFile(`${out}/trace.json`,JSON.stringify(rows,null,2));
  const capture=arg("capture","").split(",").map(Number).includes(frame);
  if(capture){if(Number(arg("coarse","0"))>0)await writeFile(`${out}/${frame}-coarse.json`,JSON.stringify(await solver.readCM11aCoarsestCapture(),null,2));await writeFile(`${out}/${frame}-boundary.bin`,boundary);for(const [key,data] of Object.entries({volume,phi,pressure,pre:pre!,post:post!}))await writeFile(`${out}/${frame}-${key}.bin`,new Uint8Array(data.buffer,data.byteOffset,data.byteLength));await writeFile(`${out}/open.bin`,open);}
  assert.deepEqual(errors,[]);if((stages.projection as {max:number}).max>10000)break;
 }
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
