/** Rigid-body energy reproduction on the production NB-FLIP solver.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js SCENE=pool BODY_DENSITY=500 node --import tsx tools/probe-nbflip-rigid-energy.ts dynamic 360
 * SCENE=pool uses coarse-first-pool-impact's tank with a 1 m rigid sphere;
 * KEEP_DROP=1 also retains its original liquid drop. BODY_Y=1.6 starts at
 * floating equilibrium. Without SCENE, use the small 32^3 hydrostatic tank.
 * CELL_SIZE, DT, BODY_RADIUS, BODY_Y and BODY_DENSITY override the fixture.
 * TRACE_STEP captures P2G, forces, pressure and pre-G2P velocity.
 * nearRms/nearMax measure wet cells outside the sphere within 3h; gridMax
 * also includes air, solid extension and unused coarse backing texels.
 */
import assert from "node:assert/strict";
import { createRigidCouplingOracleScene, createCoarseFirstPoolImpactScene } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";

const mode=process.argv[2]??"static",steps=Number(process.argv[3]??60);
assert.ok(["static","dynamic","none"].includes(mode));
assert.ok(Number.isInteger(steps)&&steps>0);
await withUniformDevice("rigid energy",async device=>{
 device.addEventListener("uncapturederror",event=>console.error(event.error.message));
 const pool=process.env.SCENE==="pool";
 const scene=pool?createCoarseFirstPoolImpactScene():createRigidCouplingOracleScene("rigid-hydrostatic");
 if(pool){scene.rigidBodies=createRigidCouplingOracleScene("rigid-hydrostatic").rigidBodies;if(!process.env.KEEP_DROP)delete scene.fluid.initialLiquidVolumes;}
 scene.voxelDomain.finestCellSize_m=Number(process.env.CELL_SIZE??(pool?0.05:0.025));
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=Number(process.env.DT??(1/60));
 scene.rigidBodies[0]!.position_m.y=Number(process.env.BODY_Y??(pool?3.65:0.4));
 scene.rigidBodies[0]!.density_kg_m3=Number(process.env.BODY_DENSITY??1000);
 const radius=Number(process.env.BODY_RADIUS??(pool?1:0.15));
 scene.rigidBodies[0]!.dimensions_m={x:radius,y:radius,z:radius};
 if(mode==="none")scene.rigidBodies=[];
 else scene.rigidBodies[0]!.motion=mode==="dynamic"?"dynamic":"static";
 const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene,JSON.parse(process.env.METHOD_VALUES??"{}"));
 const flip=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
 const frame=(solver as unknown as {mixedFrame:{fields:{velocity:GPUTexture;velocityScratch:GPUTexture;phase:GPUTexture;centerPhi:GPUTexture};forces:object;band:object}}).mixedFrame;
 let captures:{stage:string;read:()=>Promise<Float32Array>}[]|undefined;
 const hook=(target:object,key:string,texture:GPUTexture,before=false)=>{
  const object=target as Record<string,(...args:unknown[])=>unknown>,original=object[key]!.bind(object);
  object[key]=(...args:unknown[])=>{
   const take=()=>{if(captures)captures.push({stage:key+(before?".before":""),read:uniformDetailField(texture)!.storage.capture(args[0] as GPUCommandEncoder,texture)});};
   if(before)take();const result=original(...args);if(!before)take();return result;
  };
 };
 if(process.env.TRACE_STEP){
  hook(flip,"transfer",frame.fields.velocity);hook(frame.forces,"encode",frame.fields.velocityScratch);
  hook(frame.band,"encodeSolve",frame.fields.velocity);hook(flip,"update",frame.fields.velocity,true);
 }
 try{
  for(let step=1;step<=steps;step++){
   captures=step===Number(process.env.TRACE_STEP)?[]:undefined;
   assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,bodies));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
   for(const capture of captures??[]){const v=await capture.read();let max=0,at=0;for(let i=0;i<v.length;i++)if(i%4!==3&&Math.abs(v[i]!)>max){max=Math.abs(v[i]!);at=i;}console.log({step,stage:capture.stage,max,at});}
   if(step>3&&step%10!==0&&!captures)continue;
   const p=await readMixedBuffer(device,flip.activeParticles),v=await readMixedTexture(device,solver.velocityTexture);
   let max=0,sum=0,gridMax=0,gridAt=0;
   for(let i=0;i<flip.count;i++){const j=i*flip.particleStride,s=p[j+4]!**2+p[j+5]!**2+p[j+6]!**2;max=Math.max(max,Math.sqrt(s));sum+=s;}
   for(let i=0;i<v.length;i++)if(i%4!==3&&Math.abs(v[i]!)>gridMax){gridMax=Math.abs(v[i]!);gridAt=i;}
   const {nx,ny}=solver.info,index=gridAt>>2,cell=[index%nx,Math.floor(index/nx)%ny,Math.floor(index/(nx*ny))];
   const poses=await solver.readRigidBodyPoses(),phase=await readMixedTexture(device,frame.fields.phase);
   const origin=(solver as unknown as {frame:{origin_m:number[]}}).frame.origin_m,h=scene.voxelDomain.finestCellSize_m,body=poses?.[0]?.position_m;
   let nearMax=0,nearSum=0,nearCount=0;
   if(body)for(let i=0;i<phase.length;i++){
    if(phase[i]!<=0)continue;const x=i%nx,y=Math.floor(i/nx)%ny,z=Math.floor(i/(nx*ny));
    const distance=Math.hypot(origin[0]!+(x+0.5)*h-body.x,origin[1]!+(y+0.5)*h-body.y,origin[2]!+(z+0.5)*h-body.z);
    if(distance<radius||distance>radius+3*h)continue;
    const speed2=v[4*i]!**2+v[4*i+1]!**2+v[4*i+2]!**2;nearMax=Math.max(nearMax,Math.sqrt(speed2));nearSum+=speed2;nearCount++;
   }
   console.log(JSON.stringify({mode,step,nearMax,nearRms:Math.sqrt(nearSum/Math.max(1,nearCount)),nearCount,particles:flip.count,max,rms:Math.sqrt(sum/Math.max(1,flip.count)),gridMax,cell,axis:gridAt%4,poses,rigidState:Array.from((await readMixedBuffer(device,(solver as unknown as {rigidSystem:{stateBuffer:GPUBuffer}}).rigidSystem.stateBuffer)).slice(0,32)),rootResidual:solver.info.uniformPressureAcceptedResidual,bandResidual:solver.info.uniformPressureBandResidual,exchange:Array.from(new Int32Array((await readMixedBuffer(device,(solver as unknown as {rigidExchange:GPUBuffer}).rigidExchange)).buffer).slice(0,12))}));
  }
 }finally{solver.destroy();}
});
