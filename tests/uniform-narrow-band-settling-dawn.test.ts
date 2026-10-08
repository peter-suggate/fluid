import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {SecondaryParticleRenderPipeline} from "../lib/core/webgpu-secondary-particles";
import {uniformDetailField} from "../lib/methods/uniform/uniform-detail-fields";
import {withUniformDevice,advanceUniform} from "./helpers/uniform-geometric";
import {readMixedBuffer,readMixedTexture,readMixedTileWords} from "./helpers/uniform-mixed-native-fields";
const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
function pool(gravity=0){
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 Object.assign(s.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5,top:"closed",fluidWallMode:"free-slip"});
 s.voxelDomain.finestCellSize_m=1/32;s.rigidBodies=[];s.solidVoxels=[];
 s.numerics={...s.numerics,fixedDt_s:1/30,maxDt_s:1/30};
 Object.assign(s.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialVelocity_m_s:{x:0,y:0,z:0},gravity_m_s2:{x:0,y:gravity,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0});
 return s;
}
function frameOf(s:WebGPUUniformReferenceSolver){return (s as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame;}
const vertex=(x:number,y:number,z:number)=>x+33*(y+33*z);

gpuTest("NB calm level set ignores tangential particle disorder and smooths pre-existing small wrinkles",{timeout:180_000},async()=>{
 await withUniformDevice("NB calm surface",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,pool(),"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/30);const frame=frameOf(solver),stage=frame.narrowBandFlip!;
   const samples=await readMixedBuffer(device,stage.activeParticles);
   for(let i=0;i<stage.count;i++){
    const at=12*i;
    samples[at]=Math.max(0.01,Math.min(31.99,samples[at]!+0.2*Math.sin(i*13.17)));
    samples[at+2]=Math.max(0.01,Math.min(31.99,samples[at+2]!+0.2*Math.cos(i*7.31)));
   }
   device.queue.writeBuffer(stage.activeParticles,0,new Float32Array(samples));
   for(let step=2;step<=12;step++)await advanceUniform(solver,step/30);
   const raw=frame as unknown as {fields:{phi:GPUTexture;volume:GPUTexture;velocity:GPUTexture}};
   let phi=await readMixedTexture(device,raw.fields.phi);let error=0;
   for(let z=2;z<=30;z++)for(let x=2;x<=30;x++)error=Math.max(error,Math.abs(phi[vertex(x,16,z)]!)*32);
   assert.ok(error<1e-4,`horizontal sample disorder must not wrinkle calm water: ${error}h`);
   // Manufacture a zero-mean cell-scale ripple in the tracked interface.
   // This cannot disappear through advection: every velocity is zero.
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++){
    const ripple=x>2&&x<30&&z>2&&z<30?0.015*Math.sin(x*Math.PI/2)*Math.sin(z*Math.PI/2):0;
    phi[vertex(x,y,z)]=(y-16-ripple)/32;
   }
   uniformDetailField(raw.fields.phi)!.storage.upload(raw.fields.phi,phi);
   const rough=(v:Float32Array)=>{let sum=0;for(let z=4;z<28;z++)for(let x=4;x<28;x++)sum+=(v[vertex(x,16,z)]!*32)**2;return Math.sqrt(sum/(24*24));};
   const before=rough(phi);
   for(let step=13;step<=42;step++)await advanceUniform(solver,step/30);
   phi=await readMixedTexture(device,raw.fields.phi);const after=rough(phi);
   console.log(JSON.stringify({particleDisorderError_h:error,wrinkleRmsBefore_h:before,wrinkleRmsAfter_h:after}));
   assert.ok(after<before*0.4,`calm wrinkles decay: ${before} -> ${after}`);
   const volume=await readMixedTexture(device,raw.fields.volume);
   assert.ok(Math.abs(volume.reduce((a,b)=>a+b,0)/16384-1)<0.002,"filter keeps represented volume within 0.2%");
  }finally{solver.destroy();}
 });
});

gpuTest("NB escaped spray has optical geometry without liquid geometry and re-enters once",{timeout:180_000},async()=>{
 await withUniformDevice("NB independent spray",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,pool(-9.81),"balanced",{timeStep:"scene",detailSurfaceDistance:0,fineGridPadding:0,detailMarginTiles:0,detailHoldSteps:0,detailSolidContact:"off",detailImpact:"off",detailApproach:"off"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/30);const frame=frameOf(solver),stage=frame.narrowBandFlip!;
   const n=stage.count;
   device.queue.writeBuffer(stage.activeParticles,n*48,new Float32Array([16,26,16,3,0,0,0,10,0,0,0,1]));
   device.queue.writeBuffer(stage.state,0,new Uint32Array([n+1,n+1,0,0]));
   await advanceUniform(solver,2/30);
   const raw=frame as unknown as {fields:{phi:GPUTexture;volume:GPUTexture}};
   const phi=await readMixedTexture(device,raw.fields.phi);
   assert.ok(phi[vertex(16,26,16)]!>0.1,"spray never forces its enclosing liquid surface");
   const tiles=await readMixedTileWords(device,solver);
   assert.equal(tiles[4+8*(6+8*4)]!>>>31,0,"spray does not request fine simulation support");
   const source=solver.secondaryParticles!;
   const draw=new Uint32Array((await readMixedBuffer(device,source.indirectBuffer!)).buffer);
   assert.equal(draw[1],1,"only spray is published to the production water renderer");
   const drops=await readMixedBuffer(device,source.buffer);
   assert.ok(drops[3]!>0&&drops[14]===1,"optical drop has positive radius and enabled shape");
   assert.ok(Math.abs(drops[0]!)<1e-6&&Math.abs(drops[2]!)<1e-6,"render positions are world-space");
   // Exercise the real optical renderer, including both depth intersections.
   const uniforms=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
   device.queue.writeBuffer(uniforms,0,new Float32Array([32,32,0,0,0,drops[1]!,-0.2,0.2,0,drops[1]!,0,0,1,1,1,0]));
   const renderer=new SecondaryParticleRenderPipeline(device,uniforms);await renderer.initialize();renderer.setSource(source);
   const texture=(format:GPUTextureFormat)=>device.createTexture({size:[32,32],format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
   const front=texture("rgba32float"),back=texture("rgba32float"),normal=texture("rgba16float"),depth=texture("depth24plus");
   const read=device.createBuffer({size:32*32*16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   const values:Float32Array[]=[];
   try{
    for(const [side,position] of [["front",front],["back",back]] as const){
     const e=device.createCommandEncoder(),pass=e.beginRenderPass({colorAttachments:[position,normal].map(t=>({view:t.createView(),loadOp:"clear" as const,storeOp:"store" as const,clearValue:[0,0,0,0]})),depthStencilAttachment:{view:depth.createView(),depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});
     assert.ok(renderer.encodeOpticalInterface(pass,side));pass.end();e.copyTextureToBuffer({texture:position},{buffer:read,bytesPerRow:32*16},[32,32]);device.queue.submit([e.finish()]);
     await read.mapAsync(GPUMapMode.READ);values.push(new Float32Array(read.getMappedRange()).slice());read.unmap();
    }
    const at=4*(16+32*16);
    assert.ok(values[0]![at+3]!>0&&values[1]![at+3]!>0,"spray draws both optical interfaces");
    assert.ok(values[0]![at+2]!<values[1]![at+2]!,"front/back depth gives nonzero liquid thickness for water shading");
   }finally{for(const t of [front,back,normal,depth])t.destroy();read.destroy();uniforms.destroy();}

   const samples=await readMixedBuffer(device,stage.activeParticles);let index=-1;
   for(let i=0;i<stage.count;i++)if(samples[12*i+3]!>=3){assert.equal(index,-1);index=i;}
   assert.ok(index>=0);assert.ok(samples[12*index+5]!<0,"spray accelerates under gravity");
   device.queue.writeBuffer(stage.activeParticles,index*48,new Float32Array([16,14.5,16,3,0,-5,0,-1.5,0,0,0,1]));
   await advanceUniform(solver,3/30);
   const returned=new Uint32Array((await readMixedBuffer(device,source.indirectBuffer!)).buffer);
   assert.equal(returned[1],0,"re-entry removes the optical spray representation");
   assert.equal(stage.diagnostics.unsupported,0,"re-entry has resolved liquid support");
  }finally{solver.destroy();}
 });
});


gpuTest("NB disturbed pool settles under gravity without a particle-spacing noise floor",{timeout:180_000},async()=>{
 await withUniformDevice("NB gravity settling",async device=>{
  // Compare with the same pressure/advection solver at the same duration.
  // The existing short hydrostatic absolute-volume bound remains in the
  // contract suite; here isolate volume changes introduced by the disturbance.
  let referenceVolume=0;
  const reference=await uniformNarrowBandMethod.createSolverAsync!(device,pool(-9.81),"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   for(let step=1;step<=91;step++)await advanceUniform(reference,step/30);
   const fields=(frameOf(reference) as unknown as {fields:{volume:GPUTexture}}).fields;
   referenceVolume=(await readMixedTexture(device,fields.volume)).reduce((a,b)=>a+b,0);
  }finally{reference.destroy();}

  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,pool(-9.81),"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/30);const frame=frameOf(solver),stage=frame.narrowBandFlip!;
   const raw=frame as unknown as {fields:{phi:GPUTexture;volume:GPUTexture;velocity:GPUTexture}};
   const phi=await readMixedTexture(device,raw.fields.phi),samples=await readMixedBuffer(device,stage.activeParticles);
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++){
    const ripple=x>2&&x<30&&z>2&&z<30?0.03*Math.sin(x*Math.PI/2)*Math.sin(z*Math.PI/2):0;
    phi[vertex(x,y,z)]=(y-16-ripple)/32;
   }
   uniformDetailField(raw.fields.phi)!.storage.upload(raw.fields.phi,phi);
   for(let i=0;i<stage.count;i++){
    const at=12*i;samples[at]=Math.max(0.01,Math.min(31.99,samples[at]!+0.2*Math.sin(i*13.17)));
    samples[at+2]=Math.max(0.01,Math.min(31.99,samples[at+2]!+0.2*Math.cos(i*7.31)));
   }
   device.queue.writeBuffer(stage.activeParticles,0,new Float32Array(samples));
   for(let step=2;step<=91;step++)await advanceUniform(solver,step/30);
   const final=await readMixedTexture(device,raw.fields.phi),velocity=await readMixedTexture(device,raw.fields.velocity),volume=await readMixedTexture(device,raw.fields.volume);
   const heights:number[]=[];for(let z=4;z<28;z++)for(let x=4;x<28;x++)heights.push(final[vertex(x,16,z)]!*32);
   const mean=heights.reduce((a,b)=>a+b,0)/heights.length;
   const rms=Math.sqrt(heights.reduce((sum,h)=>sum+(h-mean)**2,0)/heights.length);
   let speed=0;for(let i=0;i<velocity.length;i++)if(i%4!==3)speed=Math.max(speed,Math.abs(velocity[i]!));
   const drift=Math.abs(volume.reduce((a,b)=>a+b,0)/16384-1);
   const disturbanceVolumeError=Math.abs(volume.reduce((a,b)=>a+b,0)/referenceVolume-1);
   console.log(JSON.stringify({settledRms_h:rms,settledMean_h:mean,settledSpeed:speed,settledVolumeDrift:drift,referenceVolumeDrift:Math.abs(referenceVolume/16384-1),disturbanceVolumeError}));
   assert.ok(rms<0.01,`settled gravity surface roughness ${rms}h`);
   assert.ok(speed<1e-3,`settled gravity speed ${speed}m/s`);
   assert.ok(disturbanceVolumeError<0.0002,`disturbance adds less than 0.02% volume error: ${disturbanceVolumeError}`);
  }finally{solver.destroy();}
 });
});

gpuTest("NB retains a resolved translating thin sheet in the level set",{timeout:180_000},async()=>{
 await withUniformDevice("NB coherent thin sheet",async device=>{
  const scene=pool();scene.container.fillFraction=0;
  scene.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-0.25,y:0.59375,z:-0.1875},max_m:{x:0.125,y:0.65625,z:0.1875}}];
  scene.fluid.initialVelocity_m_s={x:0.15,y:0,z:0};
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const raw=frameOf(solver) as unknown as {fields:{phi:GPUTexture;volume:GPUTexture}};
   const initial=(await readMixedTexture(device,raw.fields.volume)).reduce((a,b)=>a+b,0);
   for(let step=1;step<=12;step++)await advanceUniform(solver,step/30);
   const phi=await readMixedTexture(device,raw.fields.phi),volume=await readMixedTexture(device,raw.fields.volume);
   const crossings:number[]=[];
   for(let y=0;y<32;y++){
    const a=phi[vertex(16,y,16)]!,b=phi[vertex(16,y+1,16)]!;
    if(a*b<=0&&a!==b)crossings.push(y+a/(a-b));
   }
   assert.equal(crossings.length,2,"coherent sheet has two liquid interfaces");
   const thickness=crossings[1]!-crossings[0]!;
   const ratio=volume.reduce((a,b)=>a+b,0)/initial;
   console.log(JSON.stringify({sheetThickness_h:thickness,sheetVolumeRatio:ratio}));
   assert.ok(thickness>1.5&&thickness<2.5,"surface blending retains the two-cell sheet");
   assert.ok(Math.abs(ratio-1)<0.1,"short sheet translation retains its represented volume within 10%");
   assert.ok(phi[vertex(16,20,16)]!<0,"resolved liquid is not demoted to optical spray");
  }finally{solver.destroy();}
 });
});
