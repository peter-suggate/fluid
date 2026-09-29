import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import type {Quaternion,RigidBodyDescription,SceneDescription,Vec3} from "../lib/core/model";
import {initializeRigidBodies} from "../lib/core/rigid-body";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;

/** Rigid bodies on the mixed frame (dynamic coarsening): a static crate
 * submerged in the collapsing dam column and a light dynamic crate dropped
 * onto it. No device errors, owner mass conserved, no liquid left inside
 * either body, finite residuals, and the dynamic crate falls and stays in
 * the tank. */
const edge=.15;
const crate=(id:string,position:Vec3,motion:"static"|"dynamic"):RigidBodyDescription=>({id,name:id,shape:"box",dimensions_m:{x:edge,y:edge,z:edge},density_kg_m3:500,
 position_m:position,orientation:{x:0,y:0,z:0,w:1},linearVelocity_m_s:{x:0,y:0,z:0},angularVelocity_rad_s:{x:0,y:0,z:0},restitution:.2,friction:.5,motion});
const scene=():SceneDescription=>{
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 s.container.width_m=s.container.height_m=s.container.depth_m=.8;s.voxelDomain.finestCellSize_m=.025;
 s.fluid.inflow=undefined;s.fluid.initialDamBreakDimensions_m={x:.3,y:.4,z:.8};s.solidVoxels=[];
 s.rigidBodies=[crate("static",{x:-.25,y:.15,z:.2},"static"),crate("falling",{x:-.25,y:.6,z:-.2},"dynamic")];
 return s;
};
const rotate=(q:Quaternion,v:Vec3):Vec3=>{
 const u={x:q.x,y:q.y,z:q.z},c=(a:Vec3,b:Vec3)=>({x:a.y*b.z-a.z*b.y,y:a.z*b.x-a.x*b.z,z:a.x*b.y-a.y*b.x});
 const t=c(u,v),tt=c(u,t);return {x:v.x+2*(q.w*t.x+tt.x),y:v.y+2*(q.w*t.y+tt.y),z:v.z+2*(q.w*t.z+tt.z)};
};

async function ownerFields(device:GPUDevice,solver:WebGPUUniformReferenceSolver,s:SceneDescription,poses:readonly {position_m:Vec3;orientation:Quaternion}[]){
 const texture=solver.volumeTexture,volume=await readMixedTexture(device,texture);
 const [nx,ny,nz]=[texture.width,texture.height,texture.depthOrArrayLayers];
 const words=(solver as unknown as {mixedFrame:{ownership:{presentation:{buffer:GPUBuffer}}}}).mixedFrame.ownership.presentation.buffer;
 const bytes=4*(nx>>2)*(ny>>2)*(nz>>2),staging=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(words,0,staging,0,bytes);device.queue.submit([encoder.finish()]);
 await staging.mapAsync(GPUMapMode.READ);const tiles=new Uint32Array(staging.getMappedRange().slice(0));staging.destroy();
 const h=s.container.width_m/nx;
 // Cells well inside a body (1.5 cells in from its faces): fully covered
 // at this pose and the one before it.
 const deep=poses.map(p=>(w:Vec3)=>{
  const inverse={x:-p.orientation.x,y:-p.orientation.y,z:-p.orientation.z,w:p.orientation.w};
  const l=rotate(inverse,{x:w.x-p.position_m.x,y:w.y-p.position_m.y,z:w.z-p.position_m.z});
  return Math.max(Math.abs(l.x),Math.abs(l.y),Math.abs(l.z))<=.5*edge-1.5*h;
 });
 let mass=0;const inside=poses.map(()=>0),insideCells=poses.map(()=>0),insideCoarse=poses.map(()=>0);
 for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
  const word=tiles[(x>>2)+(nx>>2)*((y>>2)+(ny>>2)*(z>>2))]!,width=word&0x80000000?1:4;
  const v=volume[x-x%width+nx*(y-y%width+ny*(z-z%width))]!;
  if(!(x%width||y%width||z%width))mass+=v*width**3;
  const world={x:-.5*s.container.width_m+(x+.5)*h,y:(y+.5)*h,z:-.5*s.container.depth_m+(z+.5)*h};
  deep.forEach((d,i)=>{if(d(world)){inside[i]=Math.max(inside[i]!,v);insideCells[i]!++;if(width!==1)insideCoarse[i]!++;}});
 }
 return {mass,inside,insideCells,insideCoarse};
}

(modulePath?test:test.skip)("rigid bodies couple on the mixed frame",{timeout:1800000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed rigid bodies");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const failures:string[]=[];
  const s=scene(),roster=initializeRigidBodies(s.rigidBodies);
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,s,"balanced",undefined,uniformGeometricSolverOptions({coarsening:"dynamic"},s),()=>{});
  try{
   const residuals:number[]=[],heights:number[]=[];let start:number|undefined;
   const poses=async()=>{const read=await solver.readRigidBodyPoses();assert.ok(read&&read.length===2,"two body poses");return read;};
   for(let frame=1;frame<=45;frame++){
    assert.ok(solver.advanceTo(frame/30,roster),`advance ${frame}`);await solver.awaitFrameCompletion();
    const stats=await solver.readStats();residuals.push(stats.uniformPressureAcceptedResidual!);
    if(frame===1)start=(await ownerFields(device,solver,s,await poses())).mass;
    heights.push((await poses())[1]!.position_m.y);
   }
   const end=await poses(),fields=await ownerFields(device,solver,s,end);
   const report={start,fields,staticPose:end[0],fallingPose:end[1],heights:heights.filter((_,i)=>i%5===4).map(y=>+y.toFixed(4)),residuals:residuals.filter((_,i)=>i%5===4)};
   console.log(JSON.stringify(report));
   if(!(Math.abs(fields.mass-start!)<=1e-3*start!))failures.push(`owner mass ${fields.mass} vs ${start}`);
   fields.inside.forEach((v,i)=>{if(!(v<=1e-4))failures.push(`body ${i}: ${v} of liquid inside (${fields.insideCells[i]} deep cells)`);});
   if(!residuals.every(Number.isFinite))failures.push(`non-finite residuals ${residuals}`);
   const s0=s.rigidBodies[0]!.position_m,p0=end[0]!.position_m;
   if(Math.hypot(p0.x-s0.x,p0.y-s0.y,p0.z-s0.z)>1e-5)failures.push(`static crate moved to ${JSON.stringify(p0)}`);
   const p1=end[1]!.position_m;
   if(![p1.x,p1.y,p1.z].every(Number.isFinite))failures.push(`falling crate pose is not finite ${JSON.stringify(p1)}`);
   if(!(p1.y<s.rigidBodies[1]!.position_m.y-.05))failures.push(`falling crate did not fall (${p1.y})`);
   if(!(p1.y>.5*edge-.02&&Math.abs(p1.x)<.4&&Math.abs(p1.z)<.4))failures.push(`falling crate left the tank (${JSON.stringify(p1)})`);
  }finally{solver.destroy();}
  assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
