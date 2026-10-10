import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {uniformDetailField} from "../lib/methods/uniform/uniform-detail-fields";
import {withUniformDevice,advanceUniform} from "./helpers/uniform-geometric";
import {readMixedBuffer,readMixedTexture} from "./helpers/uniform-mixed-native-fields";
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
   // Seed the actual MAC field; this backend does not consume the scene vector.
   const velocity=new Float32Array(32**3*4);
   for(let i=0;i<32**3;i++)velocity[4*i]=0.15;
   solver.initializeVelocityForQA(velocity);
   const raw=frameOf(solver) as unknown as {fields:{phi:GPUTexture;volume:GPUTexture}};
   const initial=(await readMixedTexture(device,raw.fields.volume)).reduce((a,b)=>a+b,0);
   for(let step=1;step<=12;step++)await advanceUniform(solver,step/30);
   const phi=await readMixedTexture(device,raw.fields.phi),volume=await readMixedTexture(device,raw.fields.volume);
   const crossings:number[]=[];
   for(let y=0;y<32;y++){
    const a=phi[vertex(16,y,16)]!,b=phi[vertex(16,y+1,16)]!;
    if(a*b<=0&&a!==b){
     const crossing=y+a/(a-b);
     // An exact nodal zero belongs to both incident edges, but is one interface.
     if(crossings.length===0||Math.abs(crossing-crossings[crossings.length-1]!)>1e-6)crossings.push(crossing);
    }
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
