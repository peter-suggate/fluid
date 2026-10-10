import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import type {UniformMixedSurface} from "../lib/methods/uniform/uniform-mixed-surface";
import type {UniformMixedPhiResolve} from "../lib/methods/uniform/uniform-mixed-phi-resolve";
import type {UniformMixedMomentumCache,UniformMixedHangingTaps} from "../lib/methods/uniform/uniform-mixed-momentum-cache";
import type {UniformDetailGroup} from "../lib/methods/uniform/uniform-detail-fields";
import {uniformDetailField} from "../lib/methods/uniform/uniform-detail-fields";
import {advanceUniform,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
type Frame=Pick<UniformMixedFrame,"ownership"|"setRelayout"|"updateLayout"|"narrowBandFlip">&{fields:{phi:GPUTexture;phiScratch:GPUTexture;volume:GPUTexture;target:GPUTexture}};
function scene(){
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 Object.assign(s.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5,top:"closed",fluidWallMode:"free-slip"});
 s.voxelDomain.finestCellSize_m=1/32;s.rigidBodies=[];s.solidVoxels=[];
 s.numerics={...s.numerics,fixedDt_s:1/60,maxDt_s:1/60};
 Object.assign(s.fluid,{initialCondition:"tank-fill",initialLiquidVolumes:[],initialVelocity_m_s:{x:0,y:0,z:0},gravity_m_s2:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0});
 return s;
}
function upload(texture:GPUTexture,values:Float32Array){uniformDetailField(texture)!.storage.upload(texture,values);}
const vertex=(x:number,y:number,z:number)=>x+33*(y+33*z);

gpuTest("NB remap regenerates occupancy from phi instead of redistributing a corrupted mass budget",{timeout:120_000},async()=>{
 await withUniformDevice("NB surface remap",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene(),"balanced",{timeStep:"scene",detailPolicy:"full",detailSolidContact:"off"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/60);
   const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame;frame.setRelayout();
   const lattice=frame.ownership.capacity.lattice;
   const phi=await readMixedTexture(device,frame.fields.phi);
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++)phi[vertex(x,y,z)]=(y+0.2*x-16.125)/32;
   upload(frame.fields.phi,phi);
   for(const width of [4,1,4,1] as const){
    upload(frame.fields.volume,new Float32Array(32**3).fill(10));
    assert.equal(frame.updateLayout(createUniformMixedLayout(lattice,[],width)),undefined);
    await device.queue.onSubmittedWorkDone();
    const volume=await readMixedTexture(device,frame.fields.volume),target=await readMixedTexture(device,frame.fields.target);
    assert.deepEqual(volume,target,"occupancy must be regenerated at the remap boundary, before an advance");
    assert.ok(volume.every(v=>v>=0&&v<=1),"the overfilled input has no authority");
    const remapped=await readMixedTexture(device,frame.fields.phi);
    for(let i=0;i<phi.length;i++)assert.ok(Math.abs(phi[i]!-remapped[i]!)<1e-6,"affine surface survives round-trip ownership changes");
   }
  }finally{solver.destroy();}
 });
});

gpuTest("NB wall advection neither repairs liquid from stored volume nor changes a stationary surface",{timeout:120_000},async()=>{
 await withUniformDevice("NB wall surface authority",async device=>{
  const s=scene();s.solidVoxels=[{operation:"fill",minimum:[10,3,10],maximumExclusive:[14,12,14]}];
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,s,"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/60);
   const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame;
   const privateFrame=frame as unknown as {surface:UniformMixedSurface;surfaceGroups:UniformDetailGroup[];params:{surface:GPUBuffer};displacement:unknown;cache:UniformMixedMomentumCache;cacheGroup:UniformDetailGroup;hanging:UniformMixedHangingTaps;hangingGroup:UniformDetailGroup};
   assert.equal(privateFrame.displacement,undefined,"NB has no independent solid-displacement mass deposits");
   const phi=await readMixedTexture(device,frame.fields.phi);
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++)phi[vertex(x,y,z)]=(y+0.2*x-16.125)/32;
   upload(frame.fields.phi,phi);
   // A stationary characteristic must be the identity beside embedded
   // solids as well. Keep the ordinary positive frame timestep.
   let baseline:Float32Array|undefined;
   for(const mass of [0,10]){
    upload(frame.fields.volume,new Float32Array(32**3).fill(mass));
    upload(frame.fields.phiScratch,phi);
    const e=device.createCommandEncoder();privateFrame.surface.encode(e,"advect",privateFrame.surfaceGroups[0]!);device.queue.submit([e.finish()]);
    const out=await readMixedTexture(device,frame.fields.phiScratch);
    if(baseline)assert.deepEqual(out,baseline,"liquid mass must not affect wall surface transport");baseline=out;
    for(let i=0;i<phi.length;i++)assert.ok(Math.abs(phi[i]!-out[i]!)<1e-6,`stationary vertex ${i} moved: ${phi[i]} -> ${out[i]}`);
   }
   // The wall-normal face is zero, so tracing the wall itself cannot replace
   // its old air value. Incoming liquid must continue from the interior,
   // without needing stored mass as evidence (or manufacturing it at rest).
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++)phi[vertex(x,y,z)]=Math.max(1.25-y,y-5)/32;
   upload(frame.fields.phi,phi);
   // Surface transport reads the extended velocity scratch, not the physical MAC field.
   const velocity=(frame as unknown as {fields:{velocityScratch:GPUTexture}}).fields.velocityScratch;
   const falling=new Float32Array(32**3*4);for(let i=0;i<32**3;i++)falling[4*i+1]=-2;
   upload(velocity,falling);baseline=undefined;
   // Production resolves both interpolation caches after extending velocity
   // and before advection. A direct field upload must refresh those too.
   const prepare=device.createCommandEncoder();
   privateFrame.cache.encode(prepare,privateFrame.cacheGroup);
   privateFrame.hanging.encode(prepare,privateFrame.hangingGroup);
   device.queue.submit([prepare.finish()]);
   for(const mass of [0,10]){
    upload(frame.fields.volume,new Float32Array(32**3).fill(mass));
    upload(frame.fields.phiScratch,phi);
    const e=device.createCommandEncoder();privateFrame.surface.encode(e,"advect",privateFrame.surfaceGroups[0]!);device.queue.submit([e.finish()]);
    const out=await readMixedTexture(device,frame.fields.phiScratch);
    assert.ok(out[vertex(16,0,16)]!<0,"incoming liquid must wet the closed floor");
    if(baseline)assert.deepEqual(out,baseline,"incoming wall continuation is independent of stored mass");baseline=out;
   }
  }finally{solver.destroy();}
 });
});

gpuTest("NB shared distance rebuild preserves every zero-crossing cell across an h/4h seam",{timeout:120_000},async()=>{
 await withUniformDevice("NB shared distance",async device=>{
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene(),"balanced",{timeStep:"scene",detailPolicy:"full",detailSolidContact:"off"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/60);
   const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame;frame.setRelayout();
   const lattice=frame.ownership.capacity.lattice;
   assert.equal(frame.updateLayout(createUniformMixedLayout(lattice,[{id:"left",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,min_m:{x:-0.5,y:0,z:-0.5},max_m:{x:0,y:1,z:0.5}}],4)),undefined);
   const phi=new Float32Array(33**3);
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++){
    const d=Math.hypot(x-15.2,y-15.7,z-16.3)-9.3;
    phi[vertex(x,y,z)]=d/32*(d<0?0.15:1.7);
   }
   upload(frame.fields.phiScratch,phi);
   const resolve=frame as unknown as {phiResolve:UniformMixedPhiResolve;phiResolveGroups:{phi:UniformDetailGroup;scratch:UniformDetailGroup}};
   let e=device.createCommandEncoder();resolve.phiResolve.encode(e,resolve.phiResolveGroups.scratch);device.queue.submit([e.finish()]);
   // readMixedTexture returns stored texels; regular 4h hanging texels are
   // intentionally stale. Reconstruct the represented field on the CPU.
   const represented=(raw:Float32Array)=>{
    const out=raw.slice();
    for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=16;x<=32;x++){
     const origin=[x,y,z].map(v=>4*Math.floor(Math.min(v,31)/4));const q=[x,y,z].map((v,a)=>(v-origin[a]!)/4);let value=0;
     for(let k=0;k<8;k++){const b=[k%2,(k>>1)%2,k>>2];let w=1;for(let a=0;a<3;a++)w*=b[a]?q[a]!:1-q[a]!;value+=w*raw[vertex(origin[0]!+4*b[0]!,origin[1]!+4*b[1]!,origin[2]!+4*b[2]!)]!;}
     out[vertex(x,y,z)]=value;
    }return out;
   };
   const before=represented(await readMixedTexture(device,frame.fields.phiScratch));
   e=device.createCommandEncoder();frame.narrowBandFlip!.redistance(e);resolve.phiResolve.encode(e,resolve.phiResolveGroups.phi);device.queue.submit([e.finish()]);
   const after=represented(await readMixedTexture(device,frame.fields.phi));let crossings=0,changed=0;
   for(let z=0;z<32;z++)for(let y=0;y<32;y++)for(let x=0;x<32;x++){
    const corners=Array.from({length:8},(_,k)=>vertex(x+k%2,y+(k>>1)%2,z+(k>>2)));
    const values=corners.map(i=>before[i]!);
    if(Math.min(...values)<=0&&Math.max(...values)>=0){
     crossings++;assert.ok(corners.some(i=>after[i]!<=0)&&corners.some(i=>after[i]!>=0),"a cached crossing cell must remain crossing");
     for(const i of corners)assert.ok(Math.abs(after[i]!-before[i]!)<1e-6,"redistance must preserve the interpolated surface, not only vertex signs");
    }
   }
   for(let i=0;i<after.length;i++){assert.ok(Number.isFinite(after[i]));assert.equal(Math.sign(after[i]!),Math.sign(before[i]!));if(Math.abs(after[i]!-before[i]!)>1e-4)changed++;}
   assert.ok(crossings>100&&changed>100,"exercise both preserved interface and rebuilt distance");
  }finally{solver.destroy();}
 });
});
