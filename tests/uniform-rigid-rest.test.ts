import assert from "node:assert/strict";
import test from "node:test";
import { createRigidCouplingOracleScene, createCoarseFirstPoolImpactScene } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedTexture, readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("a stationary sphere crossing the NB-FLIP waterline does not generate flow", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice("rigid waterline rest",async device=>{
  // The equator crosses open cut cells; the shallow cap also has completely
  // blocked faces whose dual cells still contain water.
  for(const height of [0.4,0.5]){
   const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
   scene.voxelDomain.finestCellSize_m=0.025;
   scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
   const sphere=scene.rigidBodies[0]!;
   sphere.position_m.y=height;
   sphere.dimensions_m={x:0.15,y:0.15,z:0.15};
   const bodies=initializeRigidBodies(scene.rigidBodies);
   const solver=await createUniformSolver(device,scene);
   try{
    for(let step=1;step<=60;step++){
     // Passing [] silently removes the body and turns this into an empty-tank test.
     assert.ok(solver.advanceTo(step/60,bodies));
     await solver.awaitFrameCompletion();
     assert.equal(solver.info.simulationPipelineError,undefined);
     if(step!==1&&step!==10&&step!==60)continue;
     const velocity=await readMixedTexture(device,solver.velocityTexture);
     let peak=0;
     for(let i=0;i<velocity.length;i++)if(i%4!==3){assert.ok(Number.isFinite(velocity[i]));peak=Math.max(peak,Math.abs(velocity[i]!));}
     assert.ok(peak<0.01,`stationary sphere at y=${height}, frame ${step}: ${peak} m/s`);
    }
   }finally{solver.destroy();}
  }
 });
});

test("a freely floating sphere near equilibrium does not boil", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice("floating sphere rest",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
  scene.voxelDomain.finestCellSize_m=0.025;
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  const sphere=scene.rigidBodies[0]!;
  Object.assign(sphere,{motion:"dynamic",density_kg_m3:500,position_m:{x:0,y:0.4,z:0},dimensions_m:{x:0.15,y:0.15,z:0.15}});
  const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene);
  try{
   for(let step=1;step<=120;step++){
    assert.ok(solver.advanceTo(step/60,bodies));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
    if(step!==1&&step!==10&&step!==60&&step!==120)continue;
    const velocity=await readMixedTexture(device,solver.velocityTexture);
    let peak=0;for(let i=0;i<velocity.length;i++)if(i%4!==3)peak=Math.max(peak,Math.abs(velocity[i]!));
    assert.ok(peak<0.1,`floating sphere at frame ${step}: ${peak} m/s`);
   }
   const poses=await solver.readRigidBodyPoses();
   assert.equal(poses?.length,1);
   assert.ok(Math.abs(poses![0]!.position_m.y-0.4)<0.005,"floating sphere remains near its equilibrium height");
  }finally{solver.destroy();}
 });
});


test("a large floating sphere retains hydrostatic support after buried phi is retired", {skip:!process.env.WEBGPU_NODE_MODULE}, async t=>{
 await withUniformDevice("large pool sphere equilibrium",async device=>{
  const scene=createCoarseFirstPoolImpactScene();
  delete scene.fluid.initialLiquidVolumes;
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  scene.rigidBodies=createRigidCouplingOracleScene("rigid-hydrostatic").rigidBodies;
  Object.assign(scene.rigidBodies[0]!,{motion:"dynamic",density_kg_m3:500,position_m:{x:0,y:1.6,z:0},dimensions_m:{x:1,y:1,z:1}});
  const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene);
  const inner=solver as unknown as {frame:{origin_m:number[]};mixedFrame:{fields:{phase:GPUTexture}};rigidExchange:GPUBuffer};
  const {nx,ny}=solver.info,h=scene.voxelDomain.finestCellSize_m;
  try{
   for(let step=1;step<=240;step++){
    assert.ok(solver.advanceTo(step/60,bodies));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
    if(step!==1&&step%60!==0)continue;
    const pose=(await solver.readRigidBodyPoses())![0]!.position_m;
    assert.ok(Math.abs(pose.y-1.6)<0.03,`frame ${step}: equilibrium sphere sank or rose to ${pose.y} m`);
    if(step===1){
     const exchange=new Int32Array((await readMixedBuffer(device,inner.rigidExchange)).buffer);
     const weightImpulse=bodies[0]!.mass_kg*Math.abs(scene.fluid.gravity_m_s2.y)/60;
     assert.equal(exchange[10],1,"support must come from fluid pressure, not buried level-set volume");
     assert.ok(Math.abs(exchange[1]!*1e-6/weightImpulse-1)<0.05,"hydrostatic pressure must support the sphere's weight");
    }
    const [velocity,phase]=await Promise.all([readMixedTexture(device,solver.velocityTexture),readMixedTexture(device,inner.mixedFrame.fields.phase)]);
    let sum=0,count=0,peak=0;
    // Only physical liquid outside the sphere in its three-cell contact
    // collar. Far coarse texels and solid/air extensions are not fluid energy.
    for(let i=0;i<phase.length;i++){
     if(phase[i]!<=0)continue;
     const x=i%nx,y=Math.floor(i/nx)%ny,z=Math.floor(i/(nx*ny));
     const d=Math.hypot(inner.frame.origin_m[0]!+(x+.5)*h-pose.x,inner.frame.origin_m[1]!+(y+.5)*h-pose.y,inner.frame.origin_m[2]!+(z+.5)*h-pose.z);
     if(d<1||d>1+3*h)continue;
     const speed2=velocity[4*i]!**2+velocity[4*i+1]!**2+velocity[4*i+2]!**2;
     assert.ok(Number.isFinite(speed2));sum+=speed2;count++;peak=Math.max(peak,Math.sqrt(speed2));
    }
    assert.ok(count>0);const rms=Math.sqrt(sum/count);
    t.diagnostic(`frame ${step}: height ${pose.y.toFixed(4)} m, near-body RMS ${rms.toFixed(4)} m/s, peak ${peak.toFixed(4)} m/s`);
    assert.ok(rms<0.15,`frame ${step}: sphere generated ${rms} m/s RMS flow from rest`);
    assert.ok(peak<0.75,`frame ${step}: sphere generated a ${peak} m/s jet from rest`);
   }
  }finally{solver.destroy();}
 });
});
