import assert from "node:assert/strict";
import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { narrowBandFlipValues, uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readMixedBuffer, readMixedTexture, readMixedTileWords } from "./helpers/uniform-mixed-native-fields";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { withUniformDevice, advanceUniform, readUniformFields } from "./helpers/uniform-geometric";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("narrow-band FLIP preserves a resting pool and advances a dam at 1/30 s",{timeout:240_000},async()=>{
 await withUniformDevice("narrow-band FLIP",async device=>{
  for(const coarseParticles of [false,true])for(const rest of [true,false]){
   const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
   Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5,top:"closed",fluidWallMode:"free-slip"});
   scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
   Object.assign(scene.fluid,{initialVelocity_m_s:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,
    gravity_m_s2:{x:0,y:rest?0:-9.81,z:0}});
   if(rest)Object.assign(scene.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialBrickSeeds_m:undefined,initialHeightField:undefined});
   const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"paper",retainStageDiagnosticsForQA:true,...(coarseParticles?{coarseParticleMode:"on",detailPolicy:"requested",detailSolidContact:"off"}:{})},undefined,()=>{}) as WebGPUUniformReferenceSolver;
   try{
    // Read simulation authority, not the optional presentation snapshot.
    const raw=solver as unknown as {volumeA:GPUTexture;vertexPhiField:GPUTexture;velocityA:GPUTexture;mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}};
    const fields=()=>readUniformFields(device,{volumeTexture:raw.volumeA,vertexPhiTexture:raw.vertexPhiField,gridPressureTexture:solver.gridPressureTexture,info:solver.info,mixedFrame:raw.mixedFrame,awaitFrameCompletion:()=>solver.awaitFrameCompletion()} as unknown as WebGPUUniformReferenceSolver);
    assert.equal(raw.mixedFrame.narrowBandFlip.coarseParticles,coarseParticles);
    if(coarseParticles)assert.equal(raw.mixedFrame.narrowBandFlip.coarseOnly,true,"particles must not reserve simulation h tiles");
    const initial=await fields();const sum=(v:Float32Array)=>v.reduce((a,b)=>a+b,0);const mass=sum(initial.density);
    for(let frame=1;frame<=6;frame++)await advanceUniform(solver,frame/30);
    const final=await fields(),info=solver.narrowBandFlipInfo!;
    console.log(JSON.stringify({coarseParticles,rest,mass,finalMass:sum(final.density),...info,massRatio:sum(final.density)/mass,ny:solver.info.ny}));
    assert.ok(info.particles>1000,`populated surface band: ${JSON.stringify(info)}`);
    assert.ok(info.particles<mass*8,"do not carry particles throughout the liquid interior");
    assert.ok(Math.abs(sum(final.density)/mass-1)<1e-5,"particle reseeding must not create material");
    assert.ok(final.phi.every(Number.isFinite));
    if(coarseParticles){
     assert.equal(solver.info.uniformMixedFineTiles,0);assert.equal(solver.info.uniformPressureBandTiles,0);
     assert.equal(raw.mixedFrame.narrowBandFlip.coarseOnly,true);
    }else{
     assert.ok((solver.info.uniformMixedFineTiles??0)>0,"default retains fine surface tiles");
     assert.ok((solver.info.uniformPressureBandTiles??0)>0,"default solves fine band pressure");
     assert.equal(solver.denseLevelSetVolumeSource!.contourVertexPhi,undefined,"default publishes geometric surface");
     assert.equal(raw.mixedFrame.narrowBandFlip.surfaceSource.vertexPhi.width,1,"default does not allocate particle reconstruction");
    }
    const surface=await readMixedTexture(device,solver.denseLevelSetVolumeSource!.vertexPhi);
    assert.ok(surface.every(Number.isFinite));assert.ok(surface.some(p=>p<0)&&surface.some(p=>p>0));
    if(coarseParticles)assert.equal(solver.denseLevelSetVolumeSource!.mixedOwnership,undefined,"render detail is independent of simulation ownership");
    if(rest)assert.ok(final.phi.every((p,i)=>Math.abs(p-initial.phi[i]!)<1e-5),"zero-force pool remains still");
    if(rest){
     // Exercise the actual consumer: a nodal particle surface must not be
     // mistaken for the packed simulation volume (which drew an inverted box).
     const uniform=device.createBuffer({size:400,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
     const bodies=device.createBuffer({size:768,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
     const column=device.createTexture({size:[1,1],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING});
     const water=new RasterWaterPipeline(device,"rgba8unorm",uniform,bodies);
     try{
      await water.initialize();const packed=new Float32Array(100);
      packed.set([1,1,1,0.5],12);packed.set([0,1/32,0,0],16);packed.set([32,32,32,1],20);device.queue.writeBuffer(uniform,0,packed);
      water.setVolume(solver.surfaceFieldTexture??solver.volumeTexture,solver.columnBaseTexture??column,solver.denseLevelSetVolumeSource);
      const encoder=device.createCommandEncoder();water.encodeDenseSurfaceExtractionForQA(encoder,32,32,32,"full",false);device.queue.submit([encoder.finish()]);
      const mesh=await water.readDenseSurfaceExtractionForQA();assert.ok(mesh.vertexCount>0);
      let top=-Infinity;for(let v=0;v<mesh.vertices.length;v+=8){assert.ok(Number.isFinite(mesh.vertices[v+1]!));top=Math.max(top,mesh.vertices[v+1]!);}
      assert.ok(top>0.45&&top<0.55,`the resting particle surface remains at half height, got ${top}`);
     }finally{water.destroy();uniform.destroy();bodies.destroy();column.destroy();}
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
    if(rest&&!coarseParticles){
     // Exercise the same normalized values and live region edits as the UI.
     // Every surviving sample must belong to an accepted h tile, including
     // after moving a region across the tank and removing it altogether.
     let frame=7,controls=narrowBandFlipValues({timeStep:"paper"});
     const tune=async(values:Record<string,string|number>)=>{
      controls=narrowBandFlipValues({...controls,...values});solver.applyRuntimeValues(controls);await solver.pipelinesPrepared();
     };
     const run=async(label:string)=>{
      for(let i=0;i<3;i++)await advanceUniform(solver,++frame/30);
      const count=solver.narrowBandFlipInfo!.particles;
      const tiles=await readMixedTileWords(device,solver),data=await readMixedBuffer(device,raw.mixedFrame.narrowBandFlip.activeParticles);
      for(let i=0;i<count;i++){
       const at=12*i,x=Math.floor(data[at]!/4),y=Math.floor(data[at+1]!/4),z=Math.floor(data[at+2]!/4);
       assert.ok((tiles[x+8*(y+8*z)]!&0x80000000)!==0,`${label}: particle ${i} lies outside h ownership`);
      }
      console.log(JSON.stringify({label,particles:count,fineTiles:solver.info.uniformMixedFineTiles}));return count;
     };
     const region=(right:boolean)=>({id:"particle-region",rule:"minimum-cell-size" as const,minimumCellSize_cells:1,maximumCellSize_cells:1,
      min_m:{x:right?0:-0.5,y:0,z:-0.5},max_m:{x:right?0.5:0,y:1,z:0.5}});
     await tune({detailPolicy:"requested",detailSolidContact:"off"});
     const drawn=structuredClone(scene);drawn.fluid.refinementRegions=[region(false)];solver.applySceneUniforms(drawn);await solver.pipelinesPrepared();
     const left=await run("left Fine region");assert.ok(left>1000&&left<info.particles*0.8,"a half-tank h region limits particle generation");
     drawn.fluid.refinementRegions=[region(true)];solver.applySceneUniforms(drawn);await solver.pipelinesPrepared();
     assert.ok(await run("moved Fine region")>1000,"moving a region seeds its new surface coverage");
     solver.applySceneUniforms(scene);await solver.pipelinesPrepared();
     assert.equal(await run("no Fine regions"),0,"removing h regions retires all fine-band particles");assert.equal(solver.info.uniformMixedFineTiles,0);
     await tune({detailPolicy:"full"});assert.ok(await run("Full")>left,"Full restores complete surface particle coverage");
     await tune({detailPolicy:"dynamic",detailShape:"off",detailThin:"off",detailStrain:"off",detailRotation:"off",detailImpact:"off",detailApproach:"off",detailNearFocus:"off",detailMarginTiles:0,detailHoldSteps:0});
     assert.equal(await run("Dynamic criteria off"),0);
     await tune({detailShape:"on",detailShapeTolerance:0});assert.ok(await run("Dynamic shape on")>1000,"shape criteria regenerate surface particles");
     const relayoutMassRatio=sum((await fields()).density)/mass;console.log(JSON.stringify({relayoutMassRatio}));
     // Match uniform-detail-policy-dawn's repeated live-remap bound. The
     // stricter 1e-5 fixed-layout particle/reseeding checks above stay intact.
     assert.ok(Math.abs(relayoutMassRatio-1)<2e-3,`live refinement conserves material within the existing remap bound: ${relayoutMassRatio}`);
    }
    if(!rest&&coarseParticles){
     // Refinement remains a live accuracy option, not a prerequisite for
     // particle admission. Exercise both directions without a reset.
     solver.applyRuntimeValues({detailPolicy:"dynamic",detailShape:"on",detailShapeTolerance:0,detailSolidContact:"off",sharpeningDistance:0});
     await solver.pipelinesPrepared();
     for(let frame=7;frame<=9;frame++)await advanceUniform(solver,frame/30);
     assert.ok((solver.info.uniformMixedFineTiles??0)>0,"dynamic criteria request h tiles");
     assert.ok(solver.narrowBandFlipInfo!.particles>1000,"particles survive refinement");
     solver.applyRuntimeValues({detailPolicy:"requested",detailSolidContact:"off",sharpeningDistance:0});await solver.pipelinesPrepared();
     for(let frame=10;frame<=12;frame++)await advanceUniform(solver,frame/30);
     assert.equal(solver.info.uniformMixedFineTiles,0,"removing requests returns to coarse only");
     assert.equal(solver.info.uniformPressureBandTiles,0);
     assert.equal(raw.mixedFrame.narrowBandFlip.coarseOnly,true,"release fine capacity after retirement");
     assert.ok(solver.narrowBandFlipInfo!.particles>1000);
    }

   }finally{solver.destroy();}
  }
 });
});
