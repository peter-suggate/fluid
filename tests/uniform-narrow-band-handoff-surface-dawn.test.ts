import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";

// Reduced from the rectangular trenches around the first Letters impact.
// There is no motion to explain a surface change: only partial activation,
// the particle overlap, and h/4h ownership. Inspect canonical phi, before any
// render reconstruction can conceal an error. The old raster cuts >4h in six
// steps where it assigns particle influence beyond the seeded footprint.
(process.env.WEBGPU_NODE_MODULE?test:test.skip)("adaptive NB handoff preserves a flat surface across partial activation and retirement",{timeout:180_000},async()=>{
 await withUniformDevice("NB handoff surface",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:17/32,top:"closed",fluidWallMode:"free-slip"});
  scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
  scene.numerics={...scene.numerics,fixedDt_s:1/60,maxDt_s:1/60};
  Object.assign(scene.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialBrickSeeds_m:undefined,initialHeightField:undefined,
   scheduledDrops:[],initialVelocity_m_s:{x:0,y:0,z:0},gravity_m_s2:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0});
  const values={...uniformNarrowBandMethod.appDefaults,timeStep:"scene",detailShapeTolerance:0,adaptiveBudgetPercent:25,adaptiveFadeSeconds:.1};
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   let step=0,maxError=0;
   for(const [budget,steps] of [[25,6],[0,8],[25,6]] as const){
    solver.applyRuntimeValues({...values,adaptiveBudgetPercent:budget});await solver.pipelinesPrepared();
    for(let i=0;i<steps;i++){
     await advanceUniform(solver,++step/60);
     const fields=await readUniformFields(device,solver);
     for(let z=0;z<=32;z++)for(let x=0;x<=32;x++){
      let height=-1;
      for(let y=8;y<24;y++){
       const a=fields.vertex(x,y,z),b=fields.vertex(x,y+1,z);
       if(a<=0&&b>0){height=y+a/(a-b);break;}
      }
      const error=Math.abs(height-17);maxError=Math.max(error,maxError);
      assert.ok(error<.01,`step ${step}, budget ${budget}, (${x},${z}): flat pool changed by ${error}h`);
     }
    }
    if(budget===0){assert.equal(solver.narrowBandFlipInfo!.particles,0);assert.equal(solver.info.uniformMixedFineTiles,0);}
    else{assert.ok(solver.narrowBandFlipInfo!.particles>0);assert.ok(solver.info.uniformMixedFineTiles!>0&&solver.info.uniformMixedFineTiles!<512);}
   }
   console.log(JSON.stringify({steps:step,maxHeightError_h:maxError}));
  }finally{solver.destroy();}
 });
});

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("Letters impact has no rectangular handoff trench at 0.6 and 0.7 seconds",{timeout:300_000},async()=>{
 await withUniformDevice("Letters handoff reproduction",async device=>{
  const {createNbflipLetters}=await import("../lib/core/nbflip-paper-scenes");
  const scene=createNbflipLetters();scene.numerics={...scene.numerics,fixedDt_s:1/60,maxDt_s:1/60};
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{...uniformNarrowBandMethod.appDefaults,timeStep:"scene"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const rings:number[]=[];
   for(let step=1;step<=42;step++){
    await advanceUniform(solver,step/60);
    if(step!==36&&step!==42)continue;
    const f=await readUniformFields(device,solver);
    // The expanding rectangular trenches reproduced in the user's scene,
    // outside the impact cavity. These rings dip >2h below the resting pool
    // with the old heat raster, despite the outward wave raising the water.
    const [x0,x1,z0,z1]=step===36?[95,161,43,85]:[87,179,27,103];
    let lowest=Infinity;
    for(let z=z0!;z<=z1!;z++)for(let x=x0!;x<=x1!;x++){
     if(x!==x0&&x!==x1&&z!==z0&&z!==z1)continue;
     let height=-1;
     for(let y=20;y<60;y++){
      const a=f.vertex(x,y,z),b=f.vertex(x,y+1,z);
      if(a<=0&&b>0){height=y+a/(a-b);break;}
     }
     lowest=Math.min(lowest,height);

    }
    rings.push(lowest);
    if(process.env.FLUID_TRANSITION_REPORT){
     const {mkdir,writeFile}=await import("node:fs/promises");const heights:number[]=[];
     for(let z=0;z<=128;z++)for(let x=0;x<=256;x++){
      let height=0;for(let y=20;y<60;y++){const a=f.vertex(x,y,z),b=f.vertex(x,y+1,z);if(a<=0&&b>0){height=y+a/(a-b);break;}}heights.push(height);
     }
     await mkdir(process.env.FLUID_TRANSITION_REPORT,{recursive:true});
     await writeFile(`${process.env.FLUID_TRANSITION_REPORT}/letters-${step}.json`,JSON.stringify({nx:256,ny:192,nz:128,heights}));
    }
    console.log(JSON.stringify({step,lowestRingHeight_h:lowest,fineTiles:solver.info.uniformMixedFineTiles,particles:solver.narrowBandFlipInfo!.particles}));
   }
   // The default shape tolerance is 0.5h. Ordinary sub-cell ripple is
   // allowed; the reproduced rectangular trenches exceed 2h.
   for(const lowest of rings)assert.ok(lowest>36.5,`Letters outer ring reaches ${lowest}h below the 37h pool`);
  }finally{solver.destroy();}
 });
});
