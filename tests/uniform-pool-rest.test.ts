import assert from "node:assert/strict";
import test from "node:test";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedTexture } from "./helpers/uniform-mixed-native-fields";

test("a hydrostatic voxel pool does not seed wall particles or develop edge flow", {skip:!process.env.WEBGPU_NODE_MODULE}, async t=>{
 await withUniformDevice("voxel pool at rest",async device=>{
  // Keep the actual pool dimensions, padded storage, procedural board and
  // moving ball. The ball remains above the water during these first two seconds.
  const scene=createBackyardPoolPartyScene();
  const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene);
  const {mixedFrame:frame}=solver as unknown as {mixedFrame:UniformMixedFrame};
  try{
   for(let step=1;step<=120;step++){
    assert.ok(solver.advanceTo(step/60,bodies));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
    assert.equal(frame.narrowBandFlip!.count,0,`frame ${step}: no physical interface detail requires particles`);
    if(step!==1&&step!==10&&step!==120)continue;
    const velocity=await readMixedTexture(device,solver.velocityTexture);
    let peak=0;
    for(let i=0;i<velocity.length;i++)if(i%4!==3){assert.ok(Number.isFinite(velocity[i]));peak=Math.max(peak,Math.abs(velocity[i]!));}
    t.diagnostic(`frame ${step}: peak speed component ${peak} m/s`);
    assert.ok(peak<0.0001,`frame ${step}: hydrostatic pool developed ${peak} m/s flow`);
   }
  }finally{solver.destroy();}
 });
});

test("solid-aware refinement still seeds a real curved liquid surface", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice("voxel pool with a falling drop",async device=>{
  const scene=createBackyardPoolPartyScene();
  scene.container={...scene.container,width_m:3.2,height_m:3.2,depth_m:3.2};
  scene.solidBoundary=undefined;scene.rigidBodies=[];
  scene.solidVoxels=[[[0,0,0],[64,4,64]],[[0,0,0],[12,40,64]],[[52,0,0],[64,40,64]],
   [[12,0,0],[52,40,12]],[[12,0,52],[52,40,64]]].map(([minimum,maximumExclusive])=>({
    operation:"fill",minimum:minimum as [number,number,number],maximumExclusive:maximumExclusive as [number,number,number],materialId:1,
   }));
  scene.fluid.initialLiquidVolumes=[
   {shape:"sphere",center_m:{x:0,y:2.4,z:0},radius_m:0.2},
  ];
  const solver=await createUniformSolver(device,scene);
  const {mixedFrame:frame}=solver as unknown as {mixedFrame:UniformMixedFrame};
  try{
   for(let step=1;step<=10;step++){
    assert.ok(solver.advanceTo(step/60,[]));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
   }
   assert.ok(frame.narrowBandFlip!.count>0,"the curved drop must retain particle detail");
   const velocity=await readMixedTexture(device,solver.velocityTexture);
   let peak=0;for(let i=1;i<velocity.length;i+=4)peak=Math.max(peak,Math.abs(velocity[i]!));
   assert.ok(peak>0.1,"gravity must still accelerate the drop");
  }finally{solver.destroy();}
 });
});
