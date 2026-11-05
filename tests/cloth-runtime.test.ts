import test from 'node:test';
import assert from 'node:assert/strict';
import { ClothRuntime } from '../lib/cloth/runtime';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { sourceClothColliders } from '../lib/cloth/source-contact';
import { sampleClothCollider } from '../lib/cloth/contact';
import type { ClothDocument } from '../lib/cloth/types';
import type { WetClothSourceField } from '../lib/cloth/source-types';
const doc = (): ClothDocument => ({schemaVersion:'1.0.0',id:'test',materials:[{...DEFAULT_CLOTH_MATERIAL,mechanicalDamping_s:0}],geometry:{kind:'inline',data:{restPositions_m:[0,1,0,.1,1,0,0,1,.1],triangles:[0,2,1],rodOffsets:[0],rodIndices:[]}}});
const close = (a:number,b:number,t=1e-8) => assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);

test('cloth exchange consumes impulse exactly once with wet inertial mass and isolates publication',()=>{
  const runtime=new ClothRuntime(doc()), old=runtime.identity, impulse=new Float64Array(9);
  for(let i=0;i<3;i++) {runtime.state.waterVolume_m3[i]=runtime.state.restControlVolume_m3[i]!*.3;impulse[3*i]=.2*(runtime.state.dryMass_kg[i]!+1000*runtime.state.waterVolume_m3[i]!);}
  runtime.acceptFluidImpulse(impulse,old);
  assert.throws(()=>runtime.acceptFluidImpulse(impulse,old),/already pending/);
  runtime.step(.001,[],{gravity_m_s2:[0,0,0]});
  for(let i=0;i<3;i++) close(runtime.state.velocities_m_s[3*i]!,.2);
  assert.throws(()=>runtime.acceptFluidImpulse(impulse,old),/Stale/);
  const published=runtime.publish(), x=published.positions_m[0]!;
  close(published.saturation![0]!, .3/(1-DEFAULT_CLOTH_MATERIAL.solidFraction),1e-7);
  runtime.step(.001,[],{gravity_m_s2:[0,0,0]});
  close(runtime.state.velocities_m_s[0]!,.2);close(published.positions_m[0]!,x);
});

test('failed nonlinear steps restore source motion, physical arrays and pending impulses',()=>{
  const document=doc(); const runtime=new ClothRuntime({...document,scripts:[{kind:'translate',group:0,start_s:0,end_s:1,vector:[.1,0,0]}]});
  runtime.state.positions_m[3]! += .02;
  const impulse=new Float64Array(9);impulse[1]=1e-5;runtime.acceptFluidImpulse(impulse);
  const before=runtime.checkpoint();
  assert.throws(()=>runtime.step(.001,[],{iterations:1,tolerance_m:0}),/did not converge/);
  assert.deepEqual(runtime.checkpoint(),before);
});

test('source collision uses exact rounded moving SDF and reconstructs swept primitive motion',()=>{
  const old:WetClothSourceField={index:0,kind:'box',usage:'solid',group:0,inside:false,sampled:false,root:true,center_m:[0,0,0],halfExtents_m:[.2,.2,.2],roundingRadius_m:.02,sourceAttributes:{}};
  const next={...old,center_m:[.1,0,0] as const};
  const [collider]=sourceClothColliders([old],[next],.1);
  close(sampleClothCollider(collider!,[.22,0,0],.1).distance_m,0);
  close(sampleClothCollider(collider!,[.32,0,0]).distance_m,0);
  assert.equal(collider!.kind,'sampled');
  if(collider!.kind==='sampled') close(collider!.velocityAt([.32,0,0])[0],1);
});

test('rigid external impulses preserve SI momentum and rotate anisotropic inverse inertia', async()=>{
  const {applyGPURigidExternalImpulses}=await import('../lib/core/webgpu-rigid-body');
  const values=new Float32Array(64), rho=1000;
  // 2 kg dynamic body, rotation of 90 degrees around Z; principal inverse
  // inertias (1,2,3) rotate to world (2,1,3).
  values.set([Math.SQRT1_2,0,0,Math.SQRT1_2],8); values[20]=rho/2;values.set([rho,2*rho,3*rho],21);values[30]=1;
  values[62]=2; // held second body
  const before=values.slice();
  const changed=applyGPURigidExternalImpulses(values,['body','held'],[{bodyId:'body',linearImpulse_N_s:[2,4,6],angularImpulse_N_m_s:[1,0,0]},{bodyId:'held',linearImpulse_N_s:[2,4,6],angularImpulse_N_m_s:[1,0,0]}],rho);
  assert.deepEqual(changed,[0]);assert.deepEqual([...values.slice(12,15)],[1,2,3]);
  close(values[16]!,2,1e-6);close(values[17]!,0,1e-6);close(values[24]!,1);
  assert.deepEqual(values.slice(32),before.slice(32));
  const accepted=values.slice();
  assert.throws(()=>applyGPURigidExternalImpulses(values,['body','held'],[{bodyId:'missing',linearImpulse_N_s:[1,0,0],angularImpulse_N_m_s:[0,0,0]}],rho),/Unknown/);
  assert.deepEqual(values,accepted);
});

test('source solveSolid=false holds the mesh while prescribed supports and water state remain active',()=>{
 const document=doc(),data=document.geometry.kind==='inline'?document.geometry.data:undefined;assert.ok(data);
 const runtime=new ClothRuntime({...document,physics:{solveSolid:false,varyingSolidFraction:false},geometry:{kind:'inline',data:{...data,fixed:[1,0,0]}},scripts:[{kind:'translate',group:0,start_s:0,end_s:1,vector:[.1,0,0]}]});
 const original=runtime.state.positions_m.slice(),water=runtime.state.restControlVolume_m3[1]!*.1;runtime.state.waterVolume_m3[1]=water;
 runtime.acceptFluidImpulse(new Float64Array(9).fill(1));
 const result=runtime.step(.01);
 assert.equal(result.iterations,0);assert.ok(runtime.state.positions_m[0]!>original[0]!);
 assert.deepEqual(runtime.state.positions_m.slice(3),original.slice(3));assert.deepEqual([...runtime.state.velocities_m_s.slice(3)],new Array(6).fill(0));
 assert.equal(runtime.state.waterVolume_m3[1],water);assert.equal(runtime.state.time_s,.01);
 assert.deepEqual(runtime.state.controlVolume_m3,runtime.state.restControlVolume_m3);
});
