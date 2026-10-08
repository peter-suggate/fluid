import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("NB-FLIP Figure 9 keeps a coarse liquid interior after 15 seconds",{timeout:300_000},async()=>{
 await withUniformDevice("NB-FLIP Figure 9 long residency",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("cm12-figure-9")));
  scene.numerics={...scene.numerics,fixedDt_s:1/60,maxDt_s:1/60};
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{...uniformNarrowBandMethod.appDefaults,timeStep:"scene"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   for(let step=1;step<=900;step++){
    await advanceUniform(solver,step/60);
    if(step%180===0)console.log(JSON.stringify({step,time:solver.info.completedTime_s,fineTiles:solver.info.uniformMixedFineTiles,...solver.narrowBandFlipInfo}));
   }
   assert.ok(Math.abs(solver.info.completedTime_s!-15)<1e-8);
   const fields=await readUniformFields(device,solver),{nx,ny,nz}=solver.info;
   let wet=0,fineWet=0;
   for(let z=0;z<nz;z+=4)for(let y=0;y<ny;y+=4)for(let x=0;x<nx;x+=4){
    let full=true;
    for(let k=0;k<64;k++)if(fields.density[x+k%4+nx*(y+Math.floor(k/4)%4+ny*(z+Math.floor(k/16)))]!<0.99){full=false;break;}
    if(full){wet++;if(fields.widthAt(x,y,z)===1)fineWet++;}
   }
   console.log(JSON.stringify({wetTiles:wet,fineWetTiles:fineWet,fineWetFraction:fineWet/wet,...solver.narrowBandFlipInfo}));
   assert.ok(wet>1000,"the dam still has a substantial liquid interior");
   assert.ok(fineWet/wet<0.8,`surface refinement swallowed the liquid interior: ${fineWet}/${wet} fully wet tiles at h`);
  }finally{solver.destroy();}
 });
});
