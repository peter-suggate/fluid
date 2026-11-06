import test from 'node:test';
import assert from 'node:assert/strict';
import {GPUClothMechanics} from '../lib/cloth/gpu-mechanics';
import {GPUClothNewton} from '../lib/cloth/gpu-newton';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState,prepareClothTopology} from '../lib/cloth/topology';
import {stepCloth,clothElasticEnergy} from '../lib/cloth/dynamics';
import {withUniformDevice} from './helpers/uniform-geometric';

test('GPU colored shell and DER energies agree with Float64 mechanics',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('cloth energy solver',async device=>{
  const errors:string[]=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
  const material={...DEFAULT_CLOTH_MATERIAL,mechanicalDamping_s:0};
  for(const data of [
   {restPositions_m:[0,1,0,.1,1,0,0,1,.1,.1,1,.1],triangles:[0,2,1,1,2,3],rodOffsets:[0],rodIndices:[],fixed:[1,1,0,0]},
   {restPositions_m:[0,0,0,.01,0,0,.02,0,0,.03,0,0],triangles:[],rodOffsets:[0,4],rodIndices:[0,1,2,3],fixed:[3,0,0,0]},
  ]){
   const topology=prepareClothTopology(data,[material]),reference=createClothState(topology),actual=createClothState(topology),solver=new GPUClothMechanics(device,topology);
   if(topology.rods.length){reference.positions_m[7]=actual.positions_m[7]=.0002;reference.twist_rad[2]=actual.twist_rad[2]=.05;}
   else{reference.positions_m[10]=actual.positions_m[10]=.01+1;reference.positions_m[9]!+=.02;actual.positions_m[9]!+=.02;}
   try{await solver.initialize();const initialEnergy=clothElasticEnergy(topology,reference),dt=topology.rods.length?.00002:.0002;const options={gravity_m_s2:[0,-9.81,0] as const,iterations:128,tolerance_m:1e-9};const cpu=stepCloth(topology,reference,dt,options),gpu=await solver.step(actual,dt,options);assert.ok(cpu.converged);assert.ok(gpu.elasticEnergy_J<initialEnergy*.999,'elastic solve must materially reduce released strain energy');assert.ok(gpu.converged,`max GPU update ${gpu.maximumUpdate_m}`);
    for(let i=0;i<actual.positions_m.length;i++)assert.ok(Math.abs(actual.positions_m[i]!-reference.positions_m[i]!)<2e-7,`position ${i}: ${actual.positions_m[i]} vs ${reference.positions_m[i]}`);
    assert.ok(Math.abs(gpu.elasticEnergy_J-cpu.elasticEnergy_J)<Math.max(1e-10,cpu.elasticEnergy_J*.002));
   }finally{solver.destroy();}
  }
  assert.deepEqual(errors,[]);
 });
});

test('millimetre shell hinges remain finite and resolve motion below world coordinate precision',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('small physical cloth scales',async device=>{
  const data={restPositions_m:[2,1,0,2.001,1,0,2,1,.001,2.001,1,.001],triangles:[0,2,1,1,2,3],rodOffsets:[0],rodIndices:[],fixed:[1,1,0,0]},topology=prepareClothTopology(data,[DEFAULT_CLOTH_MATERIAL]),reference=createClothState(topology),actual=createClothState(topology),solver=new GPUClothMechanics(device,topology);
  try{await solver.initialize();const options={iterations:128,tolerance_m:1e-9};const cpu=stepCloth(topology,reference,.00003,options),gpu=await solver.step(actual,.00003,options);assert.ok(cpu.converged);assert.ok(gpu.converged,`residual ${gpu.maximumUpdate_m}`);for(let i=0;i<actual.positions_m.length;i++)assert.ok(Math.abs(actual.positions_m[i]!-reference.positions_m[i]!)<1e-9);assert.ok(actual.positions_m[7]!<1,'free node must physically move below f32 world-coordinate ULP');}finally{solver.destroy();}
 });
});

test('implicit cohesion steps on the GPU Newton owner; the Gauss-Seidel sweep refuses it',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('implicit cohesive GPU candidate',async device=>{
  const data={restPositions_m:[0,0,0,.01,0,0,0,0,.01,0,.02,0,.01,.02,0,0,.02,.01],triangles:[0,2,1,3,5,4],rodOffsets:[0],rodIndices:[]},topology=prepareClothTopology(data,[DEFAULT_CLOTH_MATERIAL]),reference=createClothState(topology),actual=createClothState(topology),solver=new GPUClothNewton(device,topology),sweep=new GPUClothMechanics(device,topology),pair={vertices:[0,1,2,3,4,5],weights:[1/3,1/3,1/3,-1/3,-1/3,-1/3],force_N:1e-5,startElement:0,endElement:1},options={gravity_m_s2:[0,0,0] as const,iterations:128,tolerance_m:1e-10,cohesionPairs:[pair]};
  try{
   await solver.initialize();await sweep.initialize();
   const cpu=stepCloth(topology,reference,.001,options),gpu=await solver.step(actual,.001,{...options,tolerance_m:1e-8});
   assert.ok(cpu.converged&&gpu.converged);assert.equal(gpu.refinementIterations,0);assert.ok(gpu.cohesionEnergy_J!>0);
   const difference=Math.max(...actual.positions_m.map((x,i)=>Math.abs(x-reference.positions_m[i]!)));
   assert.ok(difference<1e-7,`GPU Newton vs Float64: ${difference} m`);assert.equal(actual.time_s,.001);
   const before=createClothState(topology);await assert.rejects(sweep.step(before,.001,options),/no cohesion energy/);
  }finally{solver.destroy();sweep.destroy();}
 });
});
