import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {UniformNarrowBandFlip} from "../lib/methods/uniform/uniform-narrow-band-flip";
import {advanceUniform,withUniformDevice,readUniformFields} from "./helpers/uniform-geometric";
import {readMixedBuffer,readMixedTileWords} from "./helpers/uniform-mixed-native-fields";
const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
const controls={timeStep:"scene",detailPolicy:"dynamic",detailSolidContact:"off",detailSurface:"on",detailSurfaceDistance:0,detailShape:"on",detailShapeTolerance:0,detailThin:"off",detailStrain:"off",detailRotation:"off",detailImpact:"off",detailApproach:"off",detailNearFocus:"off",detailBulk:"off",detailMarginTiles:0,detailHoldSteps:0};
function scene(){
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 Object.assign(s.container,{width_m:1,height_m:1,depth_m:1,fillFraction:17/32,top:"closed",fluidWallMode:"free-slip"});
 s.voxelDomain.finestCellSize_m=1/32;s.rigidBodies=[];s.solidVoxels=[];
 s.numerics={...s.numerics,fixedDt_s:0.1,maxDt_s:0.1};
 Object.assign(s.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialVelocity_m_s:{x:0,y:0,z:0},gravity_m_s2:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,initialBrickSeeds_m:undefined,initialHeightField:undefined});
 return s;
}
gpuTest("NB particle width stays 4h while surface grid padding changes live",{timeout:180_000},async()=>{
 await withUniformDevice("NB independent widths",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene(),"balanced",{...controls,fineGridPadding:0},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
   let frame=0;const rows=[];
   for(const padding of [0,1,2,0]){
    solver.applyRuntimeValues({...controls,fineGridPadding:padding});await solver.pipelinesPrepared();
    for(let i=0;i<3;i++)await advanceUniform(solver,++frame*0.1);
    const data=await readMixedBuffer(device,stage.activeParticles),tiles=await readMixedTileWords(device,solver);
    let coarse=0,minimum=Infinity,maximum=-Infinity;
    for(let i=0;i<stage.count;i++){
     const x=data[12*i]!,y=data[12*i+1]!,z=data[12*i+2]!;minimum=Math.min(minimum,y);maximum=Math.max(maximum,y);
     if((tiles[Math.floor(x/4)+8*(Math.floor(y/4)+8*Math.floor(z/4))]!>>>31)===0)coarse++;
    }
    assert.equal(stage.count,32*32*4*8,"padding does not truncate or duplicate particles");
    assert.equal(minimum,13.25);assert.ok(Math.abs(maximum-(17-Math.sqrt(0.875**2-0.125)))<1e-5);
    if(padding===0)assert.ok(coarse>0,"the inner band survives on 4h owners");
    const fields=await readUniformFields(device,solver);
    for(let z=0;z<=32;z++)for(let x=0;x<=32;x++)assert.ok(Math.abs(fields.vertex(x,17,z))<1e-5,"resting surface is unchanged");
    rows.push({padding,fineTiles:solver.info.uniformMixedFineTiles,particles:stage.count,coarse});
   }
   assert.equal(rows[0]!.fineTiles,64,"zero padding refines only the surface-crossing tile layer");
   assert.equal(rows[3]!.fineTiles,rows[0]!.fineTiles,"removing padding releases extra grid coverage");
   assert.ok(rows[2]!.fineTiles!>rows[0]!.fineTiles!);console.log(JSON.stringify({coverage:rows}));
  }finally{solver.destroy();}
 });
});
gpuTest("NB zero-padding grid coverage follows a drop moving 2.56h per pressure step",{timeout:180_000},async()=>{
 await withUniformDevice("NB swept surface coverage",async device=>{
  const separations:number[]=[];
  for(const padding of [0,1,2]){
   const s=scene();s.container.fillFraction=0;
   Object.assign(s.fluid,{initialVelocity_m_s:{x:0.8,y:0,z:0},initialLiquidVolumes:[{shape:"sphere",center_m:{x:-0.15,y:0.6,z:0},radius_m:0.12}]});
   const solver=await uniformNarrowBandMethod.createSolverAsync!(device,s,"balanced",{...controls,fineGridPadding:padding},undefined,()=>{}) as WebGPUUniformReferenceSolver;
   try{
    const velocity=new Float32Array(32**3*4);for(let i=0;i<32**3;i++)velocity[4*i]=0.8;solver.initializeVelocityForQA(velocity);
    const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
    let first=0;
    for(let frame=1;frame<=4;frame++){
     await advanceUniform(solver,frame*0.1);
     assert.equal(solver.info.encodedSteps,frame);assert.ok(Math.abs(solver.info.lastDt_s!-0.1)<1e-8);
     const data=await readMixedBuffer(device,stage.activeParticles);let x=0,error=0;
     for(let i=0;i<stage.count;i++){x+=data[12*i]!;error=Math.max(error,Math.abs(data[12*i+4]!-0.8),Math.abs(data[12*i+5]!),Math.abs(data[12*i+6]!));}
     x/=stage.count;if(frame===1)first=x;
     assert.ok(error<1e-3,`padding ${padding}: translation velocity error ${error}`);
     assert.ok(Math.abs(x-first-(frame-1)*2.56)<0.25,`padding ${padding}: centroid ${x}`);
     separations.push(stage.diagnostics.afterMaxOutside);
     console.log(JSON.stringify({padding,frame,x,error,...solver.narrowBandFlipInfo,fineTiles:solver.info.uniformMixedFineTiles}));
    }
    console.log(JSON.stringify({padding,...solver.narrowBandFlipInfo,fineTiles:solver.info.uniformMixedFineTiles}));
   }finally{solver.destroy();}
  }
  assert.ok(separations.every(v=>v<0.5),`surface separation ${separations}`);
 });
});
