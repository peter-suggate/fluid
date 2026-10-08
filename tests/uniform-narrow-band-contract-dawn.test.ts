import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {UniformNarrowBandFlip} from "../lib/methods/uniform/uniform-narrow-band-flip";
import {advanceUniform,withUniformDevice} from "./helpers/uniform-geometric";
import {narrowBandMembershipWGSL} from "../lib/methods/uniform/uniform-narrow-band-membership.wgsl";
import type {UniformMixedPressureAuthority} from "../lib/methods/uniform/uniform-mixed-pressure-authority";
import type {UniformDetailGroup} from "../lib/methods/uniform/uniform-detail-fields";
import {uniformDetailField} from "../lib/methods/uniform/uniform-detail-fields";
import {readMixedBuffer,readMixedTexture,readMixedTileWords} from "./helpers/uniform-mixed-native-fields";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
function movingDrop(dt:number){
 const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0,top:"closed",fluidWallMode:"free-slip"});
 scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
 scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
 Object.assign(scene.fluid,{initialVelocity_m_s:{x:0.8,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,
  gravity_m_s2:{x:0,y:0,z:0},initialCondition:"tank-fill",initialLiquidVolumes:[{shape:"sphere",center_m:{x:-0.15,y:0.6,z:0},radius_m:0.12}],initialBrickSeeds_m:undefined,initialHeightField:undefined});
 return scene;
}

gpuTest("NB-FLIP surface authority bypasses volume transport and recovery, including direct runtime overrides",{timeout:120_000},async()=>{
 await withUniformDevice("NB-FLIP surface authority",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,movingDrop(1/30),"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   assert.equal(solver.info.volumeControl,false,"NB-FLIP reports no volume-control authority");
   const frame=(solver as unknown as {mixedFrame:{surface:{encode(...args:unknown[]):void};transport:{encodeTransport(...args:unknown[]):void};surfaceVolume:{encode(...args:unknown[]):void};cleanup:{encode(...args:unknown[]):void};fields:{target:GPUTexture;volume:GPUTexture;correction:GPUTexture}}}).mixedFrame;
   for(const key of ["transport","cleanup"] as const){
    assert.equal(frame[key],undefined,`NB-FLIP must not construct ${key}`);
   }
   assert.ok(frame.surfaceVolume,"NB-FLIP keeps global surface correction with a scalar target");
   const encodeSurface=frame.surface.encode.bind(frame.surface);
   frame.surface.encode=(...args:unknown[])=>{assert.notEqual(args[1],"traceCells","NB-FLIP must not trace unused volume departures");encodeSurface(...args);};
   solver.applyRuntimeValues({totalSurfaceVolume:"on",surfaceVolumeRounds:4,surfaceDeficitBalancing:"on",volumeDustThreshold:0.1,orphanDustThreshold:0.1});
   await solver.pipelinesPrepared();
   for(let step=1;step<=3;step++){
    await advanceUniform(solver,step/30);
    const target=await readMixedTexture(device,frame.fields.target),volume=await readMixedTexture(device,frame.fields.volume);
    assert.deepEqual(volume,target,"occupancy is measured directly from the current surface");
    const correction=await readMixedTexture(device,frame.fields.correction);
    assert.ok(correction.every(v=>v===0),"projection has no volume-recovery source");
   }
  }finally{solver.destroy();}
 });
});

gpuTest("NB-FLIP coarse pressure and donor phase ignore independently corrupted volume",{timeout:120_000},async()=>{
 await withUniformDevice("NB-FLIP surface-only pressure",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,movingDrop(1/30),"balanced",{timeStep:"scene",coarseParticleMode:"on",detailPolicy:"requested",detailSolidContact:"off"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/30);
   const frame=(solver as unknown as {mixedFrame:{
    authority:UniformMixedPressureAuthority;authorityGroup:UniformDetailGroup;authorityPhi:GPUBuffer;
    split:{authority:UniformMixedPressureAuthority;authorityGroup:UniformDetailGroup};
    levels:{phi:GPUBufferBinding}[];params:{authority:GPUBuffer};
    fields:{volume:GPUTexture;volumeScratch:GPUTexture;phase:GPUTexture;correction:GPUTexture};
   }}).mixedFrame;
   const classify=async()=>{
    const encoder=device.createCommandEncoder();frame.authority.encode(encoder,frame.authorityGroup);frame.split.authority.encode(encoder,frame.split.authorityGroup);
    device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
    return {phi:await readMixedBuffer(device,frame.authorityPhi),root:await readMixedBuffer(device,frame.levels[0]!.phi.buffer),phase:await readMixedTexture(device,frame.fields.phase)};
   };
   const baseline=await classify();
   // Simulate arbitrarily stale/overfilled geometric mass and re-enable its
   // old recovery controls. Neither may alter NB pressure rows or donor phase.
   for(const volume of [frame.fields.volume,frame.fields.volumeScratch])uniformDetailField(volume)!.storage.upload(volume,new Float32Array(32**3).fill(10));
   device.queue.writeBuffer(frame.params.authority,0,new Float32Array([1/30,0,0,0]));
   assert.deepEqual(await classify(),baseline,"the reconstructed surface is the only pressure and phase authority");
   assert.ok((await readMixedTexture(device,frame.fields.correction)).every(v=>v===0),"no mass recovery even with nonzero dt and enabled balancing");
  }finally{solver.destroy();}
 });
});

gpuTest("NB-FLIP translates a resolved drop across multiple h cells per pressure step",{timeout:180_000},async()=>{
 await withUniformDevice("NB-FLIP large timestep",async device=>{
  const endpoints:number[]=[];
  for(const dt of [0.1,0.05]){
   const solver=await uniformNarrowBandMethod.createSolverAsync!(device,movingDrop(dt),"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
   try{
    const velocity=new Float32Array(32**3*4);for(let i=0;i<32**3;i++)velocity[4*i]=0.8;
    solver.initializeVelocityForQA(velocity);
    const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
    const steps=Math.round(0.4/dt);let first=0;
    for(let step=1;step<=steps;step++){
     await advanceUniform(solver,step*dt);
     assert.equal(solver.info.simulationPipelineError,undefined);
     assert.ok(Math.abs(solver.info.lastDt_s!-dt)<1e-8,"keep the requested global timestep");
     assert.equal(solver.info.encodedSteps,step,"trajectory refinement does not add global fluid steps");
     const particles=await readMixedBuffer(device,stage.activeParticles);
     let x=0,velocityError=0;
     for(let i=0;i<stage.count;i++){x+=particles[12*i]!;velocityError=Math.max(velocityError,Math.abs(particles[12*i+4]!-0.8),Math.abs(particles[12*i+5]!),Math.abs(particles[12*i+6]!));}
     x/=stage.count;if(step===1)first=x;
     assert.ok(velocityError<1e-3,`constant translation velocity error ${velocityError}`);
     assert.ok(stage.diagnostics.afterMaxOutside<0.5,`particle/surface separation ${JSON.stringify(stage.diagnostics)}`);
     assert.ok(stage.diagnostics.beforeMaxOutside<1,"forward and backward traces agree within the paper's h allowance");
     if(step===steps){
      const expected=first+0.8*(steps-1)*dt*32;
      assert.ok(Math.abs(x-expected)<0.25,`centroid ${x}, expected ${expected}`);endpoints.push(x);
     }
    }
    console.log(JSON.stringify({dt,steps,cfl:0.8*dt*32,...solver.narrowBandFlipInfo}));
   }finally{solver.destroy();}
  }
  assert.ok(Math.abs(endpoints[0]!-endpoints[1]!)<0.25,"halving the macro timestep preserves the translated surface position");
 });
});

gpuTest("NB-FLIP keeps a hydrostatic pool stable without surface-volume recovery",{timeout:120_000},async()=>{
 await withUniformDevice("NB-FLIP hydrostatic pool",async device=>{
  const scene=movingDrop(1/30);
  Object.assign(scene.container,{fillFraction:0.5});
  Object.assign(scene.fluid,{initialVelocity_m_s:{x:0,y:0,z:0},initialLiquidVolumes:[],gravity_m_s2:{x:0,y:-9.81,z:0}});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   // Keep this a transport/projection regression even though production now
   // corrects total volume. A global shift must not hide hydrostatic drift.
   (solver as unknown as {mixedFrame:{surfaceVolume:{encode:()=>void}}}).mixedFrame.surfaceVolume.encode=()=>{};
   for(let step=1;step<=20;step++)await advanceUniform(solver,step/30);
   const frame=(solver as unknown as {mixedFrame:{fields:{phi:GPUTexture;velocity:GPUTexture;volume:GPUTexture};narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame;
   const phi=await readMixedTexture(device,frame.fields.phi),velocity=await readMixedTexture(device,frame.fields.velocity),volume=await readMixedTexture(device,frame.fields.volume);
   let surfaceError=0,speed=0;
   for(let z=0;z<=32;z++)for(let x=0;x<=32;x++)surfaceError=Math.max(surfaceError,Math.abs(phi[x+33*(16+33*z)]!));
   for(let i=0;i<velocity.length;i++)if(i%4!==3)speed=Math.max(speed,Math.abs(velocity[i]!));
   console.log(JSON.stringify({hydrostaticSurfaceError:surfaceError,speed,volumeError:Math.abs(volume.reduce((a,b)=>a+b,0)/16384-1)}));
   // Unlike the zero-gravity lattice test, this includes projection and
   // sampling error. Bound drift to 5% of h and volume error to 0.2%;
   // no artificial mass target is allowed to enforce exact conservation.
   assert.ok(surfaceError<0.05/32,`hydrostatic surface drift ${surfaceError} m`);
   assert.ok(speed<1e-3,`hydrostatic velocity ${speed} m/s`);
   assert.ok(Math.abs(volume.reduce((a,b)=>a+b,0)/16384-1)<0.002,"hydrostatic volume drift stays below 0.2% without enforcing it");
   assert.equal(frame.narrowBandFlip.diagnostics.unsupported,0,"resting surface particles stay grid-coupled");
  }finally{solver.destroy();}
 });
});


gpuTest("NB-FLIP stale interior distances cannot seed particles or hold fine bulk",{timeout:120_000},async()=>{
 await withUniformDevice("NB-FLIP geometric band membership",async device=>{
  const scene=movingDrop(1/30);scene.container.fillFraction=0.5;
  Object.assign(scene.fluid,{initialLiquidVolumes:[],initialVelocity_m_s:{x:0,y:0,z:0}});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",detailPolicy:"full",detailSolidContact:"off"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/30);
   const frame=(solver as unknown as {mixedFrame:{fields:{phi:GPUTexture};narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame;
   const stage=frame.narrowBandFlip,phi=await readMixedTexture(device,frame.fields.phi),stale=phi.slice();
   for(let z=0;z<=32;z++)for(let y=0;y<15;y++)for(let x=0;x<=32;x++)stale[x+33*(y+33*z)]=-0.25/32;
   const field=uniformDetailField(frame.fields.phi)!;field.storage.upload(frame.fields.phi,stale);
   // Emulate a former surface sample carried into the bulk, with a stale
   // shallow depth. The zero set stays y=16 throughout this manufactured case.
   device.queue.writeBuffer(stage.activeParticles,stage.count*48,new Float32Array([16,2,16,1,0,0,0,-0.25,0,0,0,0]));
   device.queue.writeBuffer(stage.state,0,new Uint32Array([stage.count+1,stage.count+1,0,0]));
   const encoder=device.createCommandEncoder();stage.update(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
   const count=new Uint32Array((await readMixedBuffer(device,stage.state)).buffer)[0]!;
   const particles=await readMixedBuffer(device,stage.activeParticles);
   assert.ok(count>1000,"retain the actual surface band");
   for(let i=0;i<count;i++)assert.ok(particles[12*i+1]!>=11,`particle ${i} at y=${particles[12*i+1]} escaped the conservative 4h band around y=16`);
   await advanceUniform(solver,2/30);
   const guarded=await readMixedTexture(device,frame.fields.phi);
   for(let z=0;z<=32;z++)for(let y=0;y<10;y++)for(let x=0;x<=32;x++)assert.ok(guarded[x+33*(y+33*z)]!<0,"retirement and erosion must not hollow the deep liquid");
   field.storage.upload(frame.fields.phi,phi);
   solver.applyRuntimeValues({detailPolicy:"dynamic",detailShape:"on",detailShapeTolerance:0,detailSurface:"on",detailSurfaceDistance:2,detailSolidContact:"off",detailMarginTiles:0,detailHoldSteps:0});await solver.pipelinesPrepared();
   for(let step=3;step<=6;step++)await advanceUniform(solver,step/30);
   const tiles=await readMixedTileWords(device,solver);
   for(let z=0;z<8;z++)for(let y=0;y<2;y++)for(let x=0;x<8;x++)assert.equal(tiles[x+8*(y+8*z)]!>>>31,0,"deep liquid returns to 4h after Full is disabled");
  }finally{solver.destroy();}
 });
});


gpuTest("NB-FLIP bounded distance sweeps match a brute-force nearest-crossing search",async()=>{
 await withUniformDevice("NB-FLIP bounded distance sweeps",async device=>{
  const dims=[16,12,8],cells=dims.reduce((a,b)=>a*b,1),index=(x:number,y:number,z:number)=>x+dims[0]!*(y+dims[1]!*z);
  const seeds=[[0,0,0],[15,11,7],[8,6,4],[2,10,3],[12,1,2],[4,4,7],[9,10,0]];
  // The shader's banks with no surface-tile bank before them: nodal distances, nearest cells, then the search's tile banks.
  const tiles=cells/64,vertices=dims.reduce((a,b)=>a*(b+1),1),nearest=tiles+vertices,band=nearest+cells,mask=band+4,search=mask+4*tiles;
  const words=new Uint32Array(band+4+6*tiles);words.fill(0xffffffff,0,band);
  for(const [x,y,z] of seeds){const tile=(x!>>2)+(dims[0]!/4)*((y!>>2)+(dims[1]!/4)*(z!>>2)),bit=(x!&3)+4*(y!&3)+16*(z!&3);words[mask+2*tile+(bit>>5)]!|=1<<(bit&31);}
  for(let t=0;t<tiles;t++)words[search+t]=t;
  const bins=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(bins,0,words);
   const module=device.createShaderModule({code:`
const UM_D=vec3u(16u,12u,8u);const UM_T=UM_D/4u;const NB_SURFACE_TILES=0u;
// This fixture measures the exhaustive distance search, with activity pruning off.
const NB_ACTIVITY_TARGET=0u;fn nbAdaptive()->bool{return false;}
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
fn cellIndex(p:vec3i)->u32{return u32(p.x)+UM_D.x*(u32(p.y)+UM_D.y*u32(p.z));}
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%UM_T.x,(t/UM_T.x)%UM_T.y,t/(UM_T.x*UM_T.y));}
fn umTileMinimumWidth(t:u32)->u32{return 1u;}
fn umSampleVertex(p:vec3f)->f32{return p.y-6.0;}
fn umCellOpen(p:vec3i)->f32{return 1.0;}
fn bandPhi(p:vec3f)->f32{return p.y-6.0;}
${narrowBandMembershipWGSL}`});
   const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:bins}}]});
   // One workgroup a listed tile: every tile here.
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"depthNearest"}});
   const encoder=device.createCommandEncoder();{const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(tiles);pass.end();}device.queue.submit([encoder.finish()]);
   const result=new Uint32Array((await readMixedBuffer(device,bins)).buffer);
   for(let z=0;z<dims[2]!;z++)for(let y=0;y<dims[1]!;y++)for(let x=0;x<dims[0]!;x++){
    const eligible=seeds.filter(s=>Math.abs(s[0]!-x)<=5&&Math.abs(s[1]!-y)<=5&&Math.abs(s[2]!-z)<=5);
    const actual=result[nearest+index(x,y,z)]!;
    if(!eligible.length){assert.equal(actual,0xffffffff);continue;}
    const cost=(s:number[])=> (s[0]!-x)**2+(s[1]!-y)**2+(s[2]!-z)**2;
    assert.notEqual(actual,0xffffffff,`missing crossing near ${x},${y},${z}`);
    const selected=[actual%dims[0]!,Math.floor(actual/dims[0]!)%dims[1]!,Math.floor(actual/(dims[0]!*dims[1]!))];
    assert.equal(cost(selected),Math.min(...eligible.map(cost)),`nearest crossing near ${x},${y},${z}`);
   }
  }finally{bins.destroy();}
 });
});
