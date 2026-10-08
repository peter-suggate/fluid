import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { advanceUniform, withUniformDevice, readUniformFields } from "./helpers/uniform-geometric";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
const adaptive={timeStep:"scene",adaptiveBudgetPercent:100,adaptiveSurface:"on",detailPolicy:"dynamic",detailSolidContact:"off",detailSurface:"on",detailSurfaceDistance:0,
 detailShape:"on",detailShapeTolerance:0.5,detailShapeMetric:"displacement",detailThin:"on",detailThinThickness:4,detailStrain:"on",detailStrainThreshold:0.1,
 detailImpact:"on",detailApproach:"on",detailApproachSteps:2,detailRotation:"off",detailNearFocus:"off",detailBulk:"off",detailMarginTiles:0,detailHoldSteps:0};
function pool(gravity=0){
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 Object.assign(s.container,{width_m:1,height_m:1,depth_m:1,fillFraction:17/32,top:"closed",fluidWallMode:"free-slip"});
 s.voxelDomain.finestCellSize_m=1/32;s.rigidBodies=[];s.solidVoxels=[];
 s.numerics={...s.numerics,fixedDt_s:1/30,maxDt_s:1/30};
 Object.assign(s.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialVelocity_m_s:{x:0,y:0,z:0},gravity_m_s2:{x:0,y:gravity,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,initialBrickSeeds_m:undefined,initialHeightField:undefined});
 return s;
}
gpuTest("adaptive NB percentage counts requesting tiles and changes live from zero to full",{timeout:180_000},async()=>{
 await withUniformDevice("adaptive NB quota lifecycle",async device=>{
  const values={...adaptive,detailShapeTolerance:0,adaptiveBudgetPercent:25,adaptiveFadeSeconds:0.1};
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,pool(),"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   let step=0;const advance=async(n:number)=>{for(let i=0;i<n;i++)await advanceUniform(solver,++step/30);};
   const census=async()=>{
    const source=(solver as unknown as {mixedDynamic:{importance:{buffer:GPUBuffer;offset:number}}}).mixedDynamic.importance;
    const read=device.createBuffer({size:512*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const e=device.createCommandEncoder();e.copyBufferToBuffer(source.buffer,source.offset,read,0,512*8);device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange());let wanted=0,kept=0,dropped=0;
    for(let t=0;t<512;t++){const flags=words[2*t+1]!;if((flags>>>16)&63)wanted++;if(flags&(1<<25))kept++;if(flags&(1<<27))dropped++;}
    read.unmap();read.destroy();return {wanted,kept,dropped};
   };
   await advance(2);const quarter=await census();
   assert.ok(quarter.wanted>0&&quarter.wanted<512,"the denominator is requesting surface tiles, not the domain");
   assert.equal(quarter.kept,Math.floor(quarter.wanted/4));assert.equal(quarter.dropped,quarter.wanted-quarter.kept);
   solver.applyRuntimeValues({...values,adaptiveBudgetPercent:0});await solver.pipelinesPrepared();await advance(10);
   const zero=await census();assert.equal(zero.kept,0);assert.equal(solver.narrowBandFlipInfo!.particles,0);
   assert.equal(solver.info.uniformMixedFineTiles,0,"expired support must not bypass a zero automatic budget");
   solver.applyRuntimeValues({...values,adaptiveBudgetPercent:100});await solver.pipelinesPrepared();await advance(3);
   const full=await census();assert.equal(full.kept,full.wanted);assert.equal(full.dropped,0);
   assert.ok(solver.narrowBandFlipInfo!.particles>0,"100% reactivates all requesting surface tiles");
   console.log(JSON.stringify({quarter,zero,full}));
  }finally{solver.destroy();}
 });
});
gpuTest("adaptive NB retirement time tunes live particle and fine-tile release",{timeout:180_000},async()=>{
 await withUniformDevice("adaptive NB retirement time",async device=>{
  const counts=[];
  for(const adaptiveFadeSeconds of [0.1,1]){
   const values={...adaptive,detailShapeTolerance:0,adaptiveFadeSeconds};
   const solver=await uniformNarrowBandMethod.createSolverAsync!(device,pool(),"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
   try{
    await advanceUniform(solver,1/30);assert.ok(solver.narrowBandFlipInfo!.particles>0);
    solver.applyRuntimeValues({...values,adaptiveBudgetPercent:0});await solver.pipelinesPrepared();
    for(let step=2;step<=9;step++)await advanceUniform(solver,step/30);
    counts.push(solver.narrowBandFlipInfo!.particles);
    if(adaptiveFadeSeconds===0.1)assert.equal(solver.info.uniformMixedFineTiles,0);
    else assert.ok(solver.info.uniformMixedFineTiles!>0,"cooling samples retain support");
   }finally{solver.destroy();}
  }
  assert.equal(counts[0],0);assert.ok(counts[1]!>0);
 });
});
gpuTest("adaptive NB calm pool runs at 4h with no particles under gravity",{timeout:180_000},async()=>{
 await withUniformDevice("adaptive NB cold pool",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,pool(-9.81),"balanced",adaptive,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   for(let step=1;step<=30;step++)await advanceUniform(solver,step/30);
   assert.equal(solver.narrowBandFlipInfo!.particles,0,"cold liquid needs no velocity or geometry particles");
   assert.equal(solver.info.uniformMixedFineTiles,0,"cold surface releases every h tile");
   const f=await readUniformFields(device,solver);
   let height=0;for(let z=0;z<=32;z++)for(let x=0;x<=32;x++)height=Math.max(height,Math.abs(f.vertex(x,17,z))*32);
   assert.ok(height<0.05,`hydrostatic surface drift ${height}h`);
   const stats=await solver.readStats();
   console.log(JSON.stringify({cold:solver.narrowBandFlipInfo,fineTiles:solver.info.uniformMixedFineTiles,height,volumeDrift:stats.volumeDrift}));
   assert.ok(stats.volumeDrift!==undefined&&Math.abs(stats.volumeDrift)<0.002,"cold Eulerian pool retains its volume");
  }finally{solver.destroy();}
 });
});
gpuTest("adaptive NB reactivates from zero particles and returns to a cold 4h surface",{timeout:180_000},async()=>{
 await withUniformDevice("adaptive NB lifecycle",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,pool(),"balanced",adaptive,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   let step=0;const advance=async(n:number)=>{for(let i=0;i<n;i++)await advanceUniform(solver,++step/30);};
   await advance(2);assert.equal(solver.narrowBandFlipInfo!.particles,0);
   for(let cycle=0;cycle<2;cycle++){
    solver.applyRuntimeValues({...adaptive,detailShapeTolerance:0});await solver.pipelinesPrepared();await advance(3);
    assert.ok(solver.narrowBandFlipInfo!.particles>0,"new activity bootstraps the outer surface, not only the inner collar");
    assert.ok(solver.info.uniformMixedFineTiles!>0,"activity promotes the surface");
    const hot=solver.narrowBandFlipInfo!.particles;
    const cold=cycle===0?adaptive:{...adaptive,detailShape:"off",detailThin:"off",detailStrain:"off",detailImpact:"off",detailApproach:"off"};
    solver.applyRuntimeValues(cold);await solver.pipelinesPrepared();await advance(1);
    assert.ok(solver.info.uniformMixedFineTiles!>0,"cooling particles retain fine support even with every criterion disabled");
    await advance(29);
    console.log(JSON.stringify({cycle,hot,cold:solver.narrowBandFlipInfo,fineTiles:solver.info.uniformMixedFineTiles}));
    assert.equal(solver.narrowBandFlipInfo!.particles,0,"cooling stops reseeding and retires particles");
    assert.equal(solver.info.uniformMixedFineTiles,0,"particle requests do not pin cold tiles");
    const f=await readUniformFields(device,solver);
    for(let z=4;z<=28;z++)for(let x=4;x<=28;x++)assert.ok(Math.abs(f.vertex(x,17,z))*32<0.05,"handoff preserves a resting plane");
   }
   solver.applyRuntimeValues({...adaptive,detailPolicy:"full"});await solver.pipelinesPrepared();await advance(2);
   assert.equal(solver.narrowBandFlipInfo!.adaptiveSurface,false,"Full keeps its fixed-band semantics");
   assert.ok(solver.narrowBandFlipInfo!.particles>0,"leaving adaptive mode fills the cold surface band");
   solver.applyRuntimeValues(adaptive);await solver.pipelinesPrepared();await advance(30);
   assert.equal(solver.narrowBandFlipInfo!.particles,0,"returning to Dynamic resumes cooling");
   assert.equal(solver.info.uniformMixedFineTiles,0);
  }finally{solver.destroy();}
 });
});
gpuTest("adaptive NB retains a quiet thin sheet while releasing the surrounding pool",{timeout:180_000},async()=>{
 await withUniformDevice("adaptive NB thin feature",async device=>{
  const s=pool();s.container.fillFraction=8/32;
  s.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-0.25,y:19/32,z:-0.1875},max_m:{x:0.125,y:21/32,z:0.1875}}];
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,s,"balanced",adaptive,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   for(let step=1;step<=12;step++)await advanceUniform(solver,step/30);
   const f=await readUniformFields(device,solver);const crossings:number[]=[];
   for(let y=16;y<26;y++){
    const a=f.vertex(16,y,16),b=f.vertex(16,y+1,16);
    if(a*b<=0&&a!==b)crossings.push(y+a/(a-b));
   }
   assert.equal(crossings.length,2,"low velocity must not remove a thin component");
   assert.ok(crossings[1]!-crossings[0]!>1.5,"the two-cell sheet remains resolved");
   assert.ok(solver.narrowBandFlipInfo!.particles>0,"thin geometry retains particles without motion heat");
   assert.ok(solver.info.uniformMixedFineTiles!>0&&solver.info.uniformMixedFineTiles!<512,"detail remains local");
   console.log(JSON.stringify({sheet:crossings,particles:solver.narrowBandFlipInfo!.particles,fineTiles:solver.info.uniformMixedFineTiles}));
  }finally{solver.destroy();}
 });
});
gpuTest("adaptive NB seeds new source liquid above a particle-free pool",{timeout:180_000},async()=>{
 await withUniformDevice("adaptive NB source activation",async device=>{
  const s=pool();s.container.fillFraction=8/32;
  s.fluid.scheduledDrops=[{time_s:3/30,volume:{shape:"sphere",center_m:{x:0,y:23/32,z:0},radius_m:3/32}}];
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,s,"balanced",adaptive,undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/30);assert.equal(solver.narrowBandFlipInfo!.particles,0);
   for(let step=2;step<=5;step++)await advanceUniform(solver,step/30);
   const f=await readUniformFields(device,solver);
   assert.ok(f.vertex(16,23,16)<0,"scheduled drop has a liquid interior");
   assert.ok(f.vertex(16,28,16)>0,"source keeps its extent");
   assert.ok(solver.narrowBandFlipInfo!.particles>100,"a cold simulation can create a new particle band");
   assert.equal(f.widthAt(0,7,0),4,"distant resting pool stays coarse");
  }finally{solver.destroy();}
 });
});
gpuTest("adaptive NB carries a thin translating component through cold tiles",{timeout:180_000},async()=>{
 await withUniformDevice("adaptive NB moving feature",async device=>{
  const s=pool();s.container.fillFraction=0;
  s.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-0.25,y:19/32,z:-0.1875},max_m:{x:0.125,y:21/32,z:0.1875}}];
  s.fluid.initialVelocity_m_s={x:0.2,y:0,z:0};
  const results=[];
  for(const adaptiveSurface of ["off","on"]){
   const solver=await uniformNarrowBandMethod.createSolverAsync!(device,s,"balanced",{...adaptive,adaptiveSurface},undefined,()=>{}) as WebGPUUniformReferenceSolver;
   try{
    const velocity=new Float32Array(32**3*4);for(let i=0;i<32**3;i++)velocity[4*i]=0.2;solver.initializeVelocityForQA(velocity);
    let first=0;let startVolume=0;
    for(let step=1;step<=7;step++){
     await advanceUniform(solver,step/30);
     const f=await readUniformFields(device,solver);let volume=0,moment=0;
     for(let i=0;i<f.density.length;i++){volume+=f.density[i]!;moment+=(i%32+0.5)*f.density[i]!;}
     if(step===1){first=moment/volume;startVolume=volume;}
     if(step===7){
      const displacement=moment/volume-first;
      assert.ok(Math.abs(displacement-1.28)<0.35,`${adaptiveSurface}: transported centroid ${displacement}h`);
      results.push({adaptiveSurface,volumeRatio:volume/startVolume,displacement,particles:solver.narrowBandFlipInfo!.particles});
     }
    }
   }finally{solver.destroy();}
  }
  console.log(JSON.stringify({movingSheet:results}));
  assert.ok(Math.abs(results[0]!.volumeRatio-results[1]!.volumeRatio)<0.02,"adaptive handoff adds less than 2% volume error to the fixed-band trajectory");
  assert.ok(results[1]!.particles>0,"translation alone must not cool away a thin feature");
 });
});
