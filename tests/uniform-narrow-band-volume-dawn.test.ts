import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,withUniformDevice} from "./helpers/uniform-geometric";

import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import type {UniformMixedSurfaceVolume} from "../lib/methods/uniform/uniform-mixed-surface-volume";
import {uniformDetailField,type UniformDetailGroup} from "../lib/methods/uniform/uniform-detail-fields";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {readMixedBuffer,readMixedTexture} from "./helpers/uniform-mixed-native-fields";
type ScalarFrame=Pick<UniformMixedFrame,"ownership"|"setRelayout"|"updateLayout">&{
 narrowBandVolumeBudget:GPUBuffer;surfaceVolume:UniformMixedSurfaceVolume;surfaceVolumeGroup:UniformDetailGroup;
 fields:{phi:GPUTexture;volume:GPUTexture;velocityScratch:GPUTexture};
};
function upload(texture:GPUTexture,values:Float32Array){uniformDetailField(texture)!.storage.upload(texture,values);}
const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;

// Exercise flight, impact, spreading and settling. A final-only measurement
// can hide early loss followed by reconstruction gain (the old no-reseed
// control lost 49% before returning to within 3% of its initial volume).
for(const dt of [0.017,0.05])gpuTest(`NB Figure 2 bounds geometric volume throughout 5.1s at dt=${dt}`,{timeout:180_000},async()=>{
 await withUniformDevice("NB Figure 2 volume",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("cm12-figure-2")));
  scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{...uniformNarrowBandMethod.appDefaults,timeStep:"scene"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const samples:{time:number;drift:number;shift:number;before:number}[]=[];
   for(let step=1;step<=Math.round(5.1/dt);step++){
    await advanceUniform(solver,step*dt);
    assert.ok(Math.abs((solver.info.completedTime_s??0)-step*dt)<1e-8,"retain the requested global timestep");
    {
     const stats=await solver.readStats();const drift=Number(stats.volumeDrift);
     if(step%10===0||step===Math.round(5.1/dt))samples.push({time:step*dt,drift,shift:stats.narrowBandVolumeShift_cells!,before:stats.narrowBandVolumeBeforeCorrection_cells!});
     assert.ok(Math.abs(stats.narrowBandVolumeShift_cells!)<=4.000001,"bounded surface correction");
     assert.equal(solver.info.encodedSteps,step,"no additional fluid steps");
     assert.ok(Number.isFinite(drift)&&Math.abs(drift)<0.1,`geometric volume at ${step*dt}s: ${(100*drift).toFixed(3)}%`);
    }
   }
   console.log(JSON.stringify({dt,volumeSamples:samples}));
  }finally{solver.destroy();}
 });
});

// Isolate correction from transport: it must repair both signs of error,
// ignore stale cell mass, preserve its budget through remaps, and debit exits.
gpuTest("NB scalar correction repairs loss and gain and accounts sources/outflow",{timeout:120_000},async()=>{
 await withUniformDevice("NB scalar volume contract",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5,top:"closed",fluidWallMode:"free-slip"});
  scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
  scene.numerics={...scene.numerics,fixedDt_s:1/60,maxDt_s:1/60};
  Object.assign(scene.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialVelocity_m_s:{x:0,y:0,z:0},gravity_m_s2:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/60);
   const frame=(solver as unknown as {mixedFrame:ScalarFrame}).mixedFrame;
   const initial=await readMixedBuffer(device,frame.narrowBandVolumeBudget);
   assert.ok(Math.abs(initial[0]!-16*32*32)<0.01,"capture initialized geometry once");
   const phi=await readMixedTexture(device,frame.fields.phi);
   for(const offset of [-0.25,0.25]){
    upload(frame.fields.phi,Float32Array.from(phi,v=>v+offset/32));
    upload(frame.fields.volume,new Float32Array(32**3).fill(offset<0?0:10));
    const e=device.createCommandEncoder();frame.surfaceVolume.encode(e,frame.surfaceVolumeGroup,2);device.queue.submit([e.finish()]);
    const corrected=await readMixedTexture(device,frame.fields.phi);
    for(let z=0;z<=32;z++)for(let x=0;x<=32;x++)assert.ok(Math.abs(corrected[x+33*(16+33*z)]!)<1e-5,"restore the original planar surface for both error signs");
    const state=await readMixedBuffer(device,frame.narrowBandVolumeBudget);
    assert.equal(state[0],initial[0],"corrupted geometric mass never changes the target");
    assert.ok(Math.abs(state[3]!*32-offset)<0.005,"report the applied normal shift");
   }
   frame.setRelayout();
   assert.equal(frame.updateLayout(createUniformMixedLayout(frame.ownership.capacity.lattice,[],4)),undefined);
   assert.equal((await readMixedBuffer(device,frame.narrowBandVolumeBudget))[0],initial[0],"remapping does not reset the target");
   assert.equal(frame.updateLayout(createUniformMixedLayout(frame.ownership.capacity.lattice,[],1)),undefined);
   // A real solver injection must reach the scalar budget exactly once.
   const radius=0.05;solver.injectLiquidBall({centre_m:{x:0,y:0.75,z:0},radius_m:radius});
   await advanceUniform(solver,2/60);
   const added=(4/3)*Math.PI*radius**3*32**3;
   const afterSource=await readMixedBuffer(device,frame.narrowBandVolumeBudget);
   assert.ok(Math.abs(afterSource[0]!-initial[0]!-added)<0.01,"explicit source adds to the scalar budget");
   await advanceUniform(solver,3/60);
   assert.equal((await readMixedBuffer(device,frame.narrowBandVolumeBudget))[0],afterSource[0],"source is not added twice");
   // A top-attached 4h layer moving up by 2h loses two complete layers,
   // despite a timestep larger than a one-cell crossing.
   const topPhi=Float32Array.from(phi,(_,i)=>(28-((Math.floor(i/33))%33))/32);
   upload(frame.fields.phi,topPhi);
   const velocity=new Float32Array(32**3*4);for(let i=0;i<32**3;i++)velocity[4*i+1]=1;
   upload(frame.fields.velocityScratch,velocity);
   const e=device.createCommandEncoder();frame.surfaceVolume.beginStep(e,frame.surfaceVolumeGroup,2/32,0,true);device.queue.submit([e.finish()]);
   const afterExit=await readMixedBuffer(device,frame.narrowBandVolumeBudget);
   assert.ok(Math.abs(afterExit[2]!-2*32*32)<0.01,"debit swept boundary volume at large timesteps");
   assert.ok(Math.abs(afterExit[0]!-(afterSource[0]!-2*32*32))<0.01,"outflow cannot be restored by correction");
  }finally{solver.destroy();}
 });
});
