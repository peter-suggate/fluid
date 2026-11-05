import test from 'node:test';import assert from 'node:assert/strict';
import {DEFAULT_CLOTH_MATERIAL,rodRigidity} from '../lib/cloth/material';
import {createClothState,prepareClothTopology,sourceRodReferenceDirectors} from '../lib/cloth/topology';
import {clothElasticEnergy,stepCloth} from '../lib/cloth/dynamics';
import {GPUClothMechanics} from '../lib/cloth/gpu-mechanics';
import {withUniformDevice} from './helpers/uniform-geometric';
const data={restPositions_m:[0,0,0,1,0,0,2,0,0],triangles:[],rodOffsets:[0,3],rodIndices:[0,1,2]};
const material={...DEFAULT_CLOTH_MATERIAL,rodRadius_m:.01,rodRadiusB_m:.02,solidFraction:1,mechanicalDamping_s:0};
const near=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);
test('source DER retains separate adjacent curvature components and ellipse bending axes',()=>{
 const p=prepareClothTopology(data,[material]),s=createClothState(p);s.positions_m.set([0,0,0,1,0,0,1,1,0]);const r=rodRigidity(material),untwisted=clothElasticEnergy(p,s);near(untwisted,2*r.bend1_N_m2);
 s.twist_rad.set([Math.PI/2,Math.PI/2]);near(clothElasticEnergy(p,s),2*r.bend2_N_m2);
 s.twist_rad.set([Math.PI/2,-Math.PI/2]);near(clothElasticEnergy(p,s)-.5*r.twist_N_m2*Math.PI**2,2*r.bend2_N_m2);
});
test('source central-node packing fraction scales DER bending and source first frame seed is persisted',()=>{
 const source={...material,attachmentMultiplier:.001},p=prepareClothTopology({...data,solidFraction:[.1,.4,.9]},[source]),s=createClothState(p);s.positions_m.set([0,0,0,1,0,0,1,1,0]);near(clothElasticEnergy(p,s),2*rodRigidity(source,.4).bend1_N_m2);
 assert.deepEqual(sourceRodReferenceDirectors({...data,restPositions_m:[0,0,0,0,1,0,0,2,0]}),[1,0,0,1,0,0]);
});
test('reference twist retains full-turn history instead of injecting a branch-cut torque',()=>{
 const p=prepareClothTopology({...data,initialTwist_rad:[0,-2*Math.PI],initialReferenceTwist_rad:[0,2*Math.PI]},[material]),s=createClothState(p);
 near(clothElasticEnergy(p,s),0);const report=stepCloth(p,s,.001,{gravity_m_s2:[0,0,0],iterations:128,tolerance_m:1e-9});assert.ok(report.converged);near(s.rodReferenceTwist_rad[1]!,2*Math.PI);near(s.twist_rad[1]!,-2*Math.PI);
});
test('source viscous rod extension divides by the previous length and keeps original elastic length',()=>{
 const m={...material,attachmentMultiplier:.001,rodStretchDamping_s:.003},geometry={restPositions_m:[0,0,0,.01,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},p=prepareClothTopology(geometry,[m]),s=createClothState(p),dt=.0001,L=.02,elastic=rodRigidity(m).stretch_N/.01,viscous=rodRigidity(m).stretch_N*m.rodStretchDamping_s/(dt*L),inertia=s.dryMass_kg[0]!/2/dt**2,expected=(inertia*L+elastic*.01+viscous*L)/(inertia+elastic+viscous);
 s.positions_m.set([-.005,0,0,.015,0,0]);const report=stepCloth(p,s,dt,{gravity_m_s2:[0,0,0],iterations:256,tolerance_m:1e-10});assert.ok(report.converged,JSON.stringify(report));near(s.positions_m[3]!-s.positions_m[0]!,expected);
});
test('actual GPU four-component elliptical DER and distinct viscosity modes match Float64',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('elliptical four-component DER',async device=>{
 const m={...material,attachmentMultiplier:.001,rodStretchDamping_s:.001,rodBendingDamping_s:.004,rodTwistDamping_s:.002},p=prepareClothTopology(data,[m]),a=createClothState(p),b=createClothState(p);a.positions_m.set([0,0,0,1,0,0,1.9,.2,0]);a.twist_rad.set([.1,-.1-2*Math.PI]);a.rodReferenceTwist_rad[1]=2*Math.PI;b.positions_m.set(a.positions_m);b.twist_rad.set(a.twist_rad);b.rodReferenceTwist_rad.set(a.rodReferenceTwist_rad);const solver=new GPUClothMechanics(device,p),options={gravity_m_s2:[0,0,0] as const,iterations:128,tolerance_m:1e-9};
 try{await solver.initialize();const cpu=stepCloth(p,a,.001,options),gpu=await solver.step(b,.001,options);assert.ok(cpu.converged&&gpu.converged);for(let k=0;k<a.positions_m.length;k++)assert.ok(Math.abs(a.positions_m[k]!-b.positions_m[k]!)<1e-7);for(let k=0;k<2;k++)assert.ok(Math.abs(a.twist_rad[k]!-b.twist_rad[k]!)<1e-5);}finally{solver.destroy();}
 });
});
