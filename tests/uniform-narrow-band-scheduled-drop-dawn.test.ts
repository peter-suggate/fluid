import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {liquidExtrusionDistance,liquidExtrusionEdges,liquidExtrusionVolume_m3,type LiquidExtrusion} from "../lib/core/liquid-extrusion";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;

// The schedule and the outline are both host data the shader has to agree
// with: the step that ends at a drop's time adds it once, to the field and to
// the volume budget alike, and the liquid it adds is the extrusion's -- hole,
// outward offset and rounded faces included -- rather than its bounding ball.
gpuTest("NB adds a scheduled extrusion once, in the shape its outline gives",{timeout:120_000},async()=>{
 await withUniformDevice("NB scheduled extrusion",async device=>{
  const n=32,h=1/n,dt=1/60;
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.125,top:"closed",fluidWallMode:"free-slip"});
  scene.voxelDomain.finestCellSize_m=h;scene.rigidBodies=[];scene.solidVoxels=[];
  scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
  // A frame 12 cells square around a 4-cell hole, off the lattice's symmetry
  // planes so no axis mix-up or sign error can pass by coincidence.
  const cx=0.5*h,cy=18.5*h,frame:LiquidExtrusion={shape:"extrusion",
   contours_m:[[-5,-5,5,-5,5,5,-5,5],[-3,-3,3,-3,3,3,-3,3]].map(c=>c.map((v,i)=>(i%2?cy:cx)+v*h)),
   centerZ_m:2.5*h,halfDepth_m:3*h,offset_m:h,edgeRadius_m:1.5*h};
  Object.assign(scene.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialVelocity_m_s:{x:0,y:0,z:0},gravity_m_s2:{x:0,y:0,z:0},
   inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,scheduledDrops:[{time_s:2*dt,volume:frame}]});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{...uniformNarrowBandMethod.appDefaults,timeStep:"scene"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const budget=()=>readMixedBuffer(device,(solver as unknown as {mixedFrame:{narrowBandVolumeBudget:GPUBuffer}}).mixedFrame.narrowBandVolumeBudget);
   await advanceUniform(solver,dt);
   const before=(await budget())[0]!;
   assert.ok(Math.abs(before-4*n*n)<0.01,"the pool alone before the drop's step");
   const pool=await readUniformFields(device,solver);
   assert.ok(pool.vertex(16,18,18)>0&&pool.vertex(21,18,18)>0,"no liquid where the frame will be");
   await advanceUniform(solver,2*dt);
   const added=liquidExtrusionVolume_m3(frame,h/4)/h**3;
   assert.ok(added>650&&added<760,`the frame carries ${added} cells`);
   assert.ok(Math.abs((await budget())[0]!-before-added)<0.01,"the frame's own volume joins the budget");
   const fields=await readUniformFields(device,solver),edges=liquidExtrusionEdges(frame);
   let liquid=0,air=0;
   for(let z=8;z<=28;z++)for(let y=8;y<=n;y++)for(let x=6;x<=26;x++){
    const d=liquidExtrusionDistance(frame,{x:(x-16)*h,y:y*h,z:(z-16)*h},edges)/h,phi=fields.vertex(x,y,z);
    // A cell of slack either side: the volume correction may move the surface by a fraction of one.
    if(d<-1){liquid++;assert.ok(phi<0,`liquid at ${[x,y,z]}, ${d.toFixed(2)} cells inside: phi ${phi}`);}
    else if(d>1&&d<3){air++;assert.ok(phi>0,`air at ${[x,y,z]}, ${d.toFixed(2)} cells outside: phi ${phi}`);}
   }
   assert.ok(liquid>150&&air>800,`the check covers the frame (${liquid} liquid, ${air} air vertices)`);
   assert.ok(fields.vertex(16,18,18)>0&&fields.vertex(17,19,18)>0,"the hole is air");
   assert.ok(fields.vertex(21,18,18)<0&&fields.vertex(12,18,18)<0,"both uprights are liquid");
   await advanceUniform(solver,3*dt);
   assert.ok(Math.abs((await budget())[0]!-before-added)<0.01,"the frame is not added twice");
  }finally{solver.destroy();}
 });
});
