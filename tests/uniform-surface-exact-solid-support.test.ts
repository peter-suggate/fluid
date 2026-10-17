// Candidate and reference share the same fields, ownership, controller, and scratch.
import assert from "node:assert/strict";
import test from "node:test";
import {createRigidCouplingOracleScene} from "../lib/core/scenes";
import {UniformMixedSurface} from "../lib/methods/uniform/uniform-mixed-surface";
import type {UniformMixedSurfaceFields} from "../lib/methods/uniform/uniform-mixed-surface";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {uniformDetailField} from "../lib/methods/uniform/uniform-detail-fields";
import {createUniformSolver,withUniformDevice,advanceUniform} from "./helpers/uniform-geometric";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
import {withExactSolidSupport} from "./helpers/uniform-exact-solid-support-device";

test("exact solid support preserves current surface work at nonzero transport",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("exact surface solid support",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
  scene.voxelDomain.finestCellSize_m=.0125;scene.rigidBodies=[];
  scene.solidVoxels=[{operation:"fill",minimum:[2,2,2],maximumExclusive:[4,4,4],materialId:1}];scene.solidBoundary=undefined;
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.fluid.dynamicViscosity_Pa_s=0;
  const solver=await createUniformSolver(device,scene);
  const frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame;
  // Test seam only: preserve native objects and bind the second surface to
  // the exact same current inputs, output, and serially reused scratch.
  const seam=frame as any;
  const candidate:UniformMixedSurface=seam.surface;
  const original=candidate as any;
  const reference=new UniformMixedSurface(original.device,frame.ownership,original.sourceParams,original.solid,original.farGate,frame.surfaceWorkControl);
  try{
   await advanceUniform(solver,1/60);
   const fields:UniformMixedSurfaceFields={narrowBandState:seam.narrowBandFlip.state,unitVelocity:seam.hanging.unitVelocity,unitHalo:seam.hanging.haloOf(seam.hanging.unitVelocity),velocity:seam.fields.velocityScratch,coarseVelocity:seam.coarseCache,negative:seam.fields.negativeScratch,params:seam.params.surface,evidence:{buffer:seam.stageScratch},phi:seam.fields.phi,outputPhi:seam.fields.phiScratch};
   const referenceGroup=reference.bind(fields);await withExactSolidSupport(original.device,false,()=>reference.initialize());
   const phiField=uniformDetailField(fields.phi)!,velocityField=uniformDetailField(fields.velocity)!;
   const [px,py,pz]=phiField.logical,[vx,vy,vz]=velocityField.logical;
   const h=scene.voxelDomain.finestCellSize_m,dt=1/60;
   device.queue.writeBuffer(seam.params.surface,24,new Uint32Array([1]));
   const run=async(stage:UniformMixedSurface,group:Parameters<UniformMixedSurface["encode"]>[1])=>{
    const e=device.createCommandEncoder();stage.encode(e,group);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();
    return readMixedTexture(device,fields.outputPhi);
   };
   const finalCounts=async()=>{const r=candidate.regionCounters;const b=device.createBuffer({size:20*r.regions,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=device.createCommandEncoder();e.copyBufferToBuffer(r.buffer,r.offset,b,0,b.size);device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const words=new Uint32Array(b.getMappedRange());let fullCubic=0;for(let i=4;i<words.length;i+=5)fullCubic+=words[i]!;return fullCubic;}finally{if(b.mapState==="mapped")b.unmap();b.destroy();}};
   let fullCubicSeen=0;
   for(const travel of [.0004,.4])for(const affine of [false,true]){
    const phi=new Float32Array(px*py*pz),velocity=new Float32Array(4*vx*vy*vz);
    // Strong current curvature deliberately preserves a full cubic fallback
    // around the interface; smooth far regions remain approximation eligible.
    for(let z=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++)phi[x+px*(y+py*z)]=h*(y-.5*(py-1))+.5*h*Math.cos(2*Math.PI*x/16)*Math.cos(2*Math.PI*z/16);
    const speed=travel*h/dt;
    for(let z=0;z<vz;z++)for(let y=0;y<vy;y++)for(let x=0;x<vx;x++){
     const i=4*(x+vx*(y+vy*z));velocity[i]=.25*speed;velocity[i+1]=speed*(affine?.75+.25*(x+.5)/vx:1);velocity[i+2]=speed/3;
    }
    phiField.storage.upload(fields.phi,phi);frame.surfaceWorkControl.invalidate();velocityField.storage.upload(fields.velocity,velocity);
    const e=device.createCommandEncoder();seam.plan.encodeCertificate(e,dt);seam.cache.encode(e,seam.cacheGroup);if(frame.ownership.capacity.fineTiles>0)seam.hanging.encode(e,seam.hangingGroup);device.queue.submit([e.finish()]);
    frame.surfaceWorkControl.enabled=true;
    const expected=await run(reference,referenceGroup),actual=await run(candidate,seam.surfaceGroup);
    fullCubicSeen+=await finalCounts();
    assert.deepEqual(new Uint32Array(actual.buffer),new Uint32Array(expected.buffer),`exact support changed output at ${travel} cells/step, affine=${affine}`);
    // Also compare actual approximation to the exact full operator at useful
    // nonzero displacement, keeping the .02h local budget unchanged.
    frame.surfaceWorkControl.enabled=false;const full=await run(reference,referenceGroup);
    let maxError=0,moved=0;for(let i=0;i<full.length;i++){assert.ok(Number.isFinite(full[i]));maxError=Math.max(maxError,Math.abs(full[i]!-actual[i]!));}
    for(let z=16;z<=48;z+=4)for(let x=16;x<=48;x+=4){const i=x+px*(32+py*z);moved=Math.max(moved,Math.abs(actual[i]!-phi[i]!));}
    assert.ok(maxError<=.0201*h,`local error ${maxError} at ${travel} cells/step`);assert.ok(moved>1e-7*h);
    console.log(JSON.stringify({travelCells:travel,affine,maxError_m:maxError,moved_m:moved}));
   }
   assert.ok(fullCubicSeen>0,"comparison must exercise actual full cubic fallback");
  }finally{reference.destroy();solver.destroy();}
 });
});
