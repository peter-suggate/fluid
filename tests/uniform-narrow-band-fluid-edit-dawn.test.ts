import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
for(const fillFraction of [0,0.25])gpuTest(`NB editor adds a ball with dynamic coverage, fill=${fillFraction}`,{timeout:120_000},async()=>{
 await withUniformDevice("NB editor ball",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction,top:"closed",fluidWallMode:"free-slip"});
  scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
  scene.numerics={...scene.numerics,fixedDt_s:0.05,maxDt_s:0.05};
  Object.assign(scene.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialBrickSeeds_m:undefined,initialHeightField:undefined,
   initialVelocity_m_s:{x:0,y:0,z:0},gravity_m_s2:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{...uniformNarrowBandMethod.appDefaults,timeStep:"scene",detailPolicy:"dynamic"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,0.05);
   const before=(await solver.readStats()).narrowBandTargetVolume_cells!;
   const edit={operation:"add",shape:"ball",center_m:{x:0,y:0.625,z:0},radius_m:0.125} as const;
   assert.deepEqual(await solver.editFluid(edit),{accepted:true});
   assert.equal((await solver.editFluid(edit)).accepted,false,"a pending ball cannot be silently overwritten");
   for(let step=2;step<=5;step++)await advanceUniform(solver,step*0.05);
   const fields=await readUniformFields(device,solver);
   assert.ok(fields.vertex(16,20,16)<0,"the inserted ball has a liquid interior");
   const after=await solver.readStats();
   const added=4/3*Math.PI*(edit.radius_m*32)**3;
   assert.ok(Math.abs(after.narrowBandTargetVolume_cells!-before-added)<0.01,"the edit adds to the volume budget once");
   assert.ok((solver.narrowBandFlipInfo?.particles??0)>0,"new liquid gets surface particles");
   assert.equal(solver.info.encodedSteps,5,"insertion preserves the requested large timesteps");
  }finally{solver.destroy();}
 });
});
