import assert from "node:assert/strict";
import test from "node:test";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { createUniformSolver, withUniformDevice, readUniformFields } from "./helpers/uniform-geometric";
import { readMixedTexture,readMixedTileWords } from "./helpers/uniform-mixed-native-fields";

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

test("curved free-falling liquid retains fine shape and follows gravity", {skip:!process.env.WEBGPU_NODE_MODULE}, async t=>{
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
  try{
   for(let step=1;step<=10;step++){
    assert.ok(solver.advanceTo(step/60,[]));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
   }
   // Smooth translation can use fine Eulerian transport without particles.
   // Check the physical interface instead of requiring a particular carrier.
   const fields=await readUniformFields(device,solver),crossings:number[]=[];
   const {nx,ny,nz}=solver.info,x=nx>>1,z=nz>>1,h=.05;
   for(let y=1;y<=ny;y++){
    const low=fields.vertex(x,y-1,z),high=fields.vertex(x,y,z);
    if((low<0)!==(high<0))crossings.push(h*(y-1+low/(low-high)));
   }
   assert.equal(crossings.length,2,"the drop must retain a closed liquid interval");
   const centre=(crossings[0]!+crossings[1]!)/2,radius=(crossings[1]!-crossings[0]!)/2;
   const expected=2.4+.5*scene.fluid.gravity_m_s2.y*(10/60)**2;
   assert.ok(Math.abs(centre-expected)<.5*h,`free fall centre ${centre} m vs ${expected} m`);
   assert.ok(Math.abs(radius-.2)<.5*h,`the translating drop changed radius to ${radius} m`);
   for(const y of crossings)assert.equal(fields.widthAt(x,Math.min(ny-1,Math.floor(y/h)),z),1,"the curved interface must retain h tiles");
   t.diagnostic(`drop centre ${centre} m; radius ${radius} m`);
   const velocity=await readMixedTexture(device,solver.velocityTexture);
   let peak=0;for(let i=1;i<velocity.length;i+=4)peak=Math.max(peak,Math.abs(velocity[i]!));
   assert.ok(peak>0.1,"gravity must still accelerate the drop");
  }finally{solver.destroy();}
 });
});

test("stationary curved liquid retains spatial detail without particle update activity",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("stationary spatial detail",async device=>{
  const scene=createBackyardPoolPartyScene();
  scene.container={...scene.container,width_m:3.2,height_m:3.2,depth_m:3.2};
  scene.solidBoundary=undefined;scene.solidVoxels=[];scene.rigidBodies=[];
  scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.fluid.surfaceTension_N_m=0;
  scene.fluid.initialLiquidVolumes=[{shape:"sphere",center_m:{x:0,y:1.6,z:0},radius_m:.3}];
  const solver=await createUniformSolver(device,scene,{detailShapeTolerance:.1,adaptiveBudgetPercent:100});
  try{
   for(let step=1;step<=60;step++){
    assert.ok(solver.advanceTo(step/60,[]));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
   }
   const tiles=await readMixedTileWords(device,solver);
   assert.ok(tiles.some(word=>(word&0x80000000)!==0),"curvature must keep h owners even without update activity");
   assert.equal(solver.narrowBandFlipInfo!.particles,0,"stationary geometry alone must not continuously seed particles");
   const velocity=await readMixedTexture(device,solver.velocityTexture);
   let peak=0;for(let i=0;i<velocity.length;i++)if(i%4!==3)peak=Math.max(peak,Math.abs(velocity[i]!));
   assert.ok(peak<1e-4,`stationary detail developed flow: ${peak}`);
  }finally{solver.destroy();}
 });
});
