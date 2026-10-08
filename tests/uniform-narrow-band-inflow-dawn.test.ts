import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {UniformNarrowBandFlip} from "../lib/methods/uniform/uniform-narrow-band-flip";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
gpuTest("NB inflow keeps its particles when interior reseeding exceeds the budget",{timeout:120_000},async()=>{
 await withUniformDevice("NB inflow particle priority",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  const h=1/32,dt=1/60;
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5,top:"closed",fluidWallMode:"free-slip"});
  scene.voxelDomain.finestCellSize_m=h;scene.rigidBodies=[];scene.solidVoxels=[];
  scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
  Object.assign(scene.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialVelocity_m_s:{x:0,y:0,z:0},
   gravity_m_s2:{x:0,y:0,z:0},surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,
   inflow:{center_m:{x:0,y:27*h,z:0},radius_m:3*h,length_m:2*h,velocity_m_s:{x:0,y:-3*h/dt,z:0},start_s:0,end_s:1,ramp_s:0}});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,dt);
   const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
   // Keep a crowded, persistent pool surface and enough room for the nozzle,
   // but not for all of the pool's optional inner-band seeds. This is the
   // state Figure 7 reaches after several seconds, without a long warm-up.
   const persistent=stage.capacity-2048,samples=new Float32Array(persistent*12);
   for(let i=0;i<persistent;i++)samples.set([0.25+0.5*(i%64),15.75,0.25+0.5*(Math.floor(i/64)%64),1,0,0,0,-0.25,0,0,0,0],12*i);
   device.queue.writeBuffer(stage.activeParticles,0,samples);
   device.queue.writeBuffer(stage.state,0,new Uint32Array([persistent,persistent,0,0]));
   await advanceUniform(solver,2*dt);
   const result=await readMixedBuffer(device,stage.activeParticles);
   let retained=0,injected=0;const occupied=new Set<string>();
   for(let i=0;i<stage.count;i++){
    const x=result[12*i]!,y=result[12*i+1]!,z=result[12*i+2]!;
    if(y<18)retained++;
    if(y>22&&y<27){injected++;occupied.add(`${Math.floor(x)},${Math.floor(z)}`);}
   }
   assert.equal(stage.count,stage.capacity,"the fixture exercises the full allocation");
   assert.ok(stage.reseedClipped>0,"optional reseeding exceeds the available slots");
   assert.ok(retained>=persistent,"existing surface particles retain their slots");
   assert.ok(injected>300,`the source has a particle band, got ${injected} particles`);
   assert.ok(occupied.size>=24,`the band covers the nozzle aperture, got ${occupied.size} columns`);
  }finally{solver.destroy();}
 });
});
