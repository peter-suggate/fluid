import assert from "node:assert/strict";
import test from "node:test";
import {createRigidCouplingOracleScene} from "../lib/core/scenes";
import type {UniformMixedFramePlan} from "../lib/methods/uniform/uniform-mixed-frame-plan";
import type {UniformMixedHangingTaps,UniformMixedMomentumCache} from "../lib/methods/uniform/uniform-mixed-momentum-cache";
import {uniformDetailField} from "../lib/methods/uniform/uniform-detail-fields";
import type {UniformMixedSurface} from "../lib/methods/uniform/uniform-mixed-surface";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {createUniformSolver,withUniformDevice,advanceUniform} from "./helpers/uniform-geometric";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

test("production approximate surface follows current moving input and agrees with full work",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("surface moving input cache",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
  // 64 cells across leave a true interior beyond the 4h stencil and solid-clear halo.
  scene.voxelDomain.finestCellSize_m=.0125;
  scene.rigidBodies=[];scene.solidVoxels=[{operation:"fill",minimum:[2,2,2],maximumExclusive:[4,4,4],materialId:1}];scene.solidBoundary=undefined;
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.fluid.dynamicViscosity_Pa_s=0;
  const solver=await createUniformSolver(device,scene);
  const frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame;
  type Seam={plan:UniformMixedFramePlan;cache:UniformMixedMomentumCache;cacheGroup:Parameters<UniformMixedMomentumCache["encode"]>[1];hanging:UniformMixedHangingTaps;hangingGroup:Parameters<UniformMixedHangingTaps["encode"]>[1];fields:{phi:GPUTexture;phiScratch:GPUTexture;velocityScratch:GPUTexture};params:{surface:GPUBuffer};surface:UniformMixedSurface;surfaceGroup:Parameters<UniformMixedSurface["encode"]>[1]};
  const seam=frame as unknown as Seam;
  try{
   await advanceUniform(solver,1/60);
   const base=await readMixedTexture(device,seam.fields.phi),field=uniformDetailField(seam.fields.phi)!;
   const h=scene.voxelDomain.finestCellSize_m;
   // Only the signed-distance band moves; far-air residency markers retain
   // their original values. Surface.encode uses real velocity/owner caches
   // established by the frame above, without advancing pressure or particles.
   const upload=(offset:number)=>{const v=base.slice();for(let i=0;i<v.length;i++)if(Math.abs(v[i]!)<8*h)v[i]!+=offset;field.storage.upload(seam.fields.phi,v);frame.surfaceWorkControl.invalidate();};
   device.queue.writeBuffer(seam.params.surface,24,new Uint32Array([1]));
   const readWords=async(source:GPUBuffer,offset:number,count:number)=>{const read=device.createBuffer({size:4*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=device.createCommandEncoder();e.copyBufferToBuffer(source,offset,read,0,4*count);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);return new Uint32Array(read.getMappedRange()).slice();}finally{if(read.mapState==="mapped")read.unmap();read.destroy();}};
   let lastCounts=new Uint32Array();
   const run=async()=>{const e=device.createCommandEncoder();seam.surface.encode(e,seam.surfaceGroup);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();
    const counts=await readWords(seam.surface.claims,seam.surface.workCountersOffset,seam.surface.workCounterWords);lastCounts=counts;return counts[2]!+counts[8]!;};
   const models=async()=>new Float32Array((await readWords(frame.surfaceWorkControl.surface,4*frame.surfaceWorkControl.modelWordOffset,8*frame.surfaceWorkControl.regions)).buffer);
   upload(0);await run();await run();assert.ok(lastCounts[0]!>0,"production far band must execute the early endpoint proof");assert.ok(lastCounts[2]!>0,"whole-affine work must stay local with a remote solid present");const initial=await models();
   const stationary=await readMixedTexture(device,seam.fields.phiScratch);
   let movingHits=0;
   for(const offset of [.0002*h,.0004*h,.0006*h,.0008*h]){upload(offset);movingHits+=await run();}
   assert.ok(movingHits>0,"changed input must execute the production predictor");
   const small=await models();
   assert.ok(small.some((value,i)=>i%8===3&&Number.isFinite(value)&&value!==initial[i]),"regional affine models must consume current moving phi every frame");
   const predicted=await readMixedTexture(device,seam.fields.phiScratch);
   let movement=0;for(let i=0;i<predicted.length;i++)movement=Math.max(movement,Math.abs(predicted[i]!-stationary[i]!));
   assert.ok(movement>.0004*h,"surface output must move while its approximation consumes current input");
   upload(.0016*h);await run();const spent=await models();
   assert.ok(spent.some((value,i)=>i%8===3&&Number.isFinite(value)&&value!==small[i]),"continued small motion must update the current model instead of retaining stale evidence");
   upload(.0008*h);frame.surfaceWorkControl.enabled=false;await run();const full=await readMixedTexture(device,seam.fields.phiScratch);
   let difference=0;for(let i=0;i<full.length;i++){assert.ok(Number.isFinite(full[i]));difference=Math.max(difference,Math.abs(full[i]!-predicted[i]!));}
   assert.ok(difference<.01*h,`moving-input full-work disagreement ${difference}`);
   // Manufacture current extended fields directly, then rebuild the real
   // production certificate, coarse cache and hanging taps. Pressure and
   // particles do not participate in this isolated surface comparison.
   const velocityField=uniformDetailField(seam.fields.velocityScratch)!;
   const [vx,vy,vz]=velocityField.logical,[px,py,pz]=field.logical;
   const curved=new Float32Array(base.length);
   for(let z=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++)curved[x+px*(y+py*z)]=h*(y-.5*(py-1))+.125*h*Math.cos(2*Math.PI*x/(px-1))*Math.cos(2*Math.PI*z/(pz-1));
   for(const affine of [false,true]){
    const velocity=new Float32Array(4*vx*vy*vz);
    for(let z=0;z<vz;z++)for(let y=0;y<vy;y++)for(let x=0;x<vx;x++){
     const at=4*(x+vx*(y+vy*z));velocity[at]=.00015;velocity[at+1]=.0003+(affine?.0002*((x+.5)/vx-.5):0);velocity[at+2]=.0001;
    }
    velocityField.storage.upload(seam.fields.velocityScratch,velocity);field.storage.upload(seam.fields.phi,curved);frame.surfaceWorkControl.invalidate();
    const e=device.createCommandEncoder();seam.plan.encodeCertificate(e,1/60);seam.cache.encode(e,seam.cacheGroup);if(frame.ownership.capacity.fineTiles>0)seam.hanging.encode(e,seam.hangingGroup);device.queue.submit([e.finish()]);
    frame.surfaceWorkControl.enabled=true;await run();const accepted=lastCounts[2]!,rejected=lastCounts[3]!;assert.ok(lastCounts[2]!>0,`${affine?"affine":"uniform"} nonzero velocity must execute whole-affine work with a remote solid`);
    const approximate=await readMixedTexture(device,seam.fields.phiScratch);
    frame.surfaceWorkControl.enabled=false;await run();const reference=await readMixedTexture(device,seam.fields.phiScratch);
    let error=0,moved=0;for(let i=0;i<reference.length;i++){assert.ok(Number.isFinite(reference[i]));error=Math.max(error,Math.abs(reference[i]!-approximate[i]!));}
    for(let z=16;z<=48;z+=4)for(let x=16;x<=48;x+=4){const i=x+px*(32+py*z);moved=Math.max(moved,Math.abs(approximate[i]!-curved[i]!));}
    console.log(JSON.stringify({fixture:affine?"affine-velocity-curved-phi":"uniform-velocity-curved-phi",wholeAffineAccepted:accepted,wholeAffineRejected:rejected,maxError_m:error,maxMovement_m:moved,budget_m:.02*h}));
    assert.ok(error<=.0201*h,`${affine?"affine":"uniform"} curved-phi disagreement ${error}`);
    assert.ok(moved>1e-7*h,"nonzero certified-midpoint transport must move the current curved surface");
   }

  }finally{solver.destroy();}
 });
});
