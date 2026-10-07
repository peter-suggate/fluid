import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { withUniformDevice, advanceUniform, readUniformFields } from "./helpers/uniform-geometric";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("narrow-band FLIP preserves a resting pool and advances a dam at 1/30 s",{timeout:180_000},async()=>{
 await withUniformDevice("narrow-band FLIP",async device=>{
  for(const rest of [true,false]){
   const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
   Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5,top:"closed",fluidWallMode:"free-slip"});
   scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
   Object.assign(scene.fluid,{initialVelocity_m_s:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,
    gravity_m_s2:{x:0,y:rest?0:-9.81,z:0}});
   if(rest)Object.assign(scene.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialBrickSeeds_m:undefined,initialHeightField:undefined});
   const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"paper",retainStageDiagnosticsForQA:true},undefined,()=>{}) as WebGPUUniformReferenceSolver;
   try{
    // Read simulation authority, not the optional presentation snapshot.
    const raw=solver as unknown as {volumeA:GPUTexture;vertexPhiField:GPUTexture;velocityA:GPUTexture;mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}};
    const fields=()=>readUniformFields(device,{volumeTexture:raw.volumeA,vertexPhiTexture:raw.vertexPhiField,gridPressureTexture:solver.gridPressureTexture,info:solver.info,mixedFrame:raw.mixedFrame,awaitFrameCompletion:()=>solver.awaitFrameCompletion()} as unknown as WebGPUUniformReferenceSolver);
    const initial=await fields();const sum=(v:Float32Array)=>v.reduce((a,b)=>a+b,0);const mass=sum(initial.density);
    for(let frame=1;frame<=6;frame++)await advanceUniform(solver,frame/30);
    const final=await fields(),info=solver.narrowBandFlipInfo!;
    console.log(JSON.stringify({rest,mass,finalMass:sum(final.density),...info,massRatio:sum(final.density)/mass,ny:solver.info.ny}));
    assert.ok(info.particles>1000,`populated surface band: ${JSON.stringify(info)}`);
    assert.ok(info.particles<mass*8,"do not carry particles throughout the liquid interior");
    assert.ok(Math.abs(sum(final.density)/mass-1)<1e-5,"particle reseeding must not create material");
    assert.ok(final.phi.every(Number.isFinite));
    if(rest)assert.ok(final.phi.every((p,i)=>Math.abs(p-initial.phi[i]!)<1e-5),"zero-force pool remains still");
    if(rest){
     // A particle-only impulse must reach the projected grid. This fails for
     // decorative tracers or a transfer that is overwritten by momentum.
     const stage=raw.mixedFrame.narrowBandFlip,data=await readMixedBuffer(device,stage.activeParticles);
     for(let i=0;i<info.particles;i++){const at=i*12;data[at+4]=0.3*(data[at+2]!/32-0.5);data[at+6]=-0.3*(data[at]!/32-0.5);}
     device.queue.writeBuffer(stage.activeParticles,0,new Float32Array(data));
     await advanceUniform(solver,7/30);
     const velocity=await readMixedTexture(device,raw.velocityA);
     assert.ok(velocity.some((v,i)=>i%4!==3&&Math.abs(v)>0.01),"surface particle momentum reaches the grid");
     assert.ok(Math.abs(sum((await fields()).density)/mass-1)<1e-5);
    }
   }finally{solver.destroy();}
  }
 });
});
