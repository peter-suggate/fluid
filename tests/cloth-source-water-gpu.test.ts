import assert from 'node:assert/strict';
import test from 'node:test';
import {createRigidCouplingOracleScene} from '../lib/core/scenes';
import {UniformSourceWater} from '../lib/methods/uniform/uniform-source-water';
import {createUniformSolver,withUniformDevice} from './helpers/uniform-geometric';
import type {UniformMixedFrame} from '../lib/methods/uniform/uniform-mixed-frame';
import {readMixedBuffer} from './helpers/uniform-mixed-native-fields';
import {NB_VOLUME_COMPENSATION_FLOAT} from '../lib/methods/uniform/uniform-mixed-surface-volume';
import {uniformDetailField} from '../lib/methods/uniform/uniform-detail-fields';
import {tetrahedralWaterFill} from '../lib/cloth/grid-water-transfer';
import type {WetClothSourceField} from '../lib/cloth/source-types';

test('source SDF adds actual NB-FLIP water once and terminator removes the same volume',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('source NB-FLIP volume',async device=>{
  const scene=createRigidCouplingOracleScene('rigid-hydrostatic');scene.rigidBodies=[];scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.numerics.fixedDt_s=scene.numerics.maxDt_s=.001;
  scene.fluid.refinementRegions=[{id:'source-support',min_m:{x:-.35,y:.1,z:-.35},max_m:{x:.35,y:.8,z:.35},rule:'minimum-cell-size',minimumCellSize_cells:1,maximumCellSize_cells:1}];
  const solver=await createUniformSolver(device,scene),frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame,l=frame.ownership.capacity.lattice;
  const fields=frame.clothCouplingFields([l.origin_m.x,l.origin_m.y,l.origin_m.z],l.cellSize_m),plugin=new UniformSourceWater(fields);
  const source:WetClothSourceField={index:0,kind:'sphere',usage:'source',group:0,inside:false,sampled:true,root:true,center_m:[0,.55,0],radius_m:.06,sourceAttributes:{}};
  try{
   await plugin.initialize();assert.ok(solver.advanceTo(.001,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
   const budget=async()=>{const b=await readMixedBuffer(device,fields.waterBudget);return (b[0]!+b[NB_VOLUME_COMPENSATION_FLOAT]!)*l.cellSize_m.reduce((a,b)=>a*b,1);};
   const before=await budget(),beforePhi=await uniformDetailField(fields.phi)!.storage.read(fields.phi);
   plugin.prepare([source],[{fieldIndex:0,kind:'emit',velocity_m_s:[1,0,0]}]);
   const emit=async()=>{const e=device.createCommandEncoder();plugin.encodeSupport(e);plugin.encode(e);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();};
   await emit();const added=await budget()-before;let expected=0;const nx=solver.info.nx+1,ny=solver.info.ny+1,volume=new Float64Array(solver.info.nx*solver.info.ny*solver.info.nz);
   for(let z=0;z<solver.info.nz;z++)for(let y=0;y<solver.info.ny;y++)for(let x=0;x<solver.info.nx;x++){
    const before=Array.from({length:8},(_,i)=>beforePhi[x+(i&1)+nx*(y+((i>>1)&1)+ny*(z+((i>>2)&1)))]!);
    const after=before.map((v,i)=>{const distance=Math.fround(Math.hypot(l.origin_m.x+(x+(i&1))*l.cellSize_m[0],l.origin_m.y+(y+((i>>1)&1))*l.cellSize_m[1]-.55,l.origin_m.z+(z+((i>>2)&1))*l.cellSize_m[2])-.06);return distance<=.1?Math.min(v,distance):v;});
    const cellVolume=l.cellSize_m.reduce((a,b)=>a*b,1);volume[x+solver.info.nx*(y+solver.info.ny*z)]=tetrahedralWaterFill(after)*cellVolume;
    expected+=(tetrahedralWaterFill(after)-tetrahedralWaterFill(before))*cellVolume;
   }
   assert.ok(Math.abs(added-expected)<1e-8,`GPU volume ${added} versus exact native tetrahedral ${expected}`);
   assert.ok(Math.abs((await plugin.readLedger()).externalVolume_m3-added)<1e-8);
   const velocity=await uniformDetailField(fields.projectedVelocity)!.storage.read(fields.projectedVelocity);let momentum=0;
   for(let z=0;z<solver.info.nz;z++)for(let y=0;y<solver.info.ny;y++)for(let x=0;x<solver.info.nx-1;x++){
    const i=x+solver.info.nx*(y+solver.info.ny*z);momentum+=velocity[4*i]!*.5*(volume[i]!+volume[i+1]!);
   }
   assert.ok(Math.abs(momentum-added)<1e-8,`source x momentum/rho ${momentum} versus ${added}`);
   await emit();assert.ok(Math.abs(await budget()-before-added)<1e-9,'fully occupied source creates no duplicate volume');
   plugin.prepare([source],[{fieldIndex:0,kind:'terminate',velocity_m_s:[0,0,0]}]);await emit();
   assert.ok(await budget()<before+added*.1,'terminator removes the source water');
  }finally{plugin.destroy();solver.destroy();}
 });
});

test('resident terminator removes native water and retires real FLIP samples before reconstruction',{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 const {UniformSourceResidentTermination}=await import('../lib/methods/uniform/uniform-source-resident-termination');
 await withUniformDevice('resident source termination',async device=>{
  const scene=createRigidCouplingOracleScene('rigid-hydrostatic');scene.rigidBodies=[];scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.fluid.initialVelocity_m_s={x:.4,y:0,z:0};scene.numerics.fixedDt_s=scene.numerics.maxDt_s=.001;
  scene.fluid.refinementRegions=[{id:'termination-boundary',min_m:{x:-.4,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4},rule:'minimum-cell-size',minimumCellSize_cells:1,maximumCellSize_cells:1}];
  const solver=await createUniformSolver(device,scene,{bulkTier:'off'}),frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame,l=frame.ownership.capacity.lattice;
  const fields=frame.clothCouplingFields([l.origin_m.x,l.origin_m.y,l.origin_m.z],l.cellSize_m),plugin=new UniformSourceResidentTermination(fields,frame.narrowBandFlip);
  const box:WetClothSourceField={index:0,kind:'box',usage:'terminator',group:0,inside:false,sampled:true,root:true,center_m:[0,.35,0],halfExtents_m:[.12,.1,.12],roundingRadius_m:0,sourceAttributes:{}};
  try{
   await plugin.initialize();assert.ok(solver.advanceTo(.001,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);assert.ok(solver.advanceTo(.002,[]));await solver.awaitFrameCompletion();
   const records=await readMixedBuffer(device,frame.narrowBandFlip.activeParticles),stateWords=new Uint32Array((await readMixedBuffer(device,frame.narrowBandFlip.state)).buffer);
   const held=[12,13,14].map(i=>32*(stateWords[i]!|0));const live:number[][]=[];
   for(let i=0;i<frame.narrowBandFlip.capacity;i++)if(records[12*i+3]!>0)live.push([0,1,2].map(a=>fields.origin_m[a]!+(records[12*i+a]!+held[a]!)*fields.cellSize_m[a]!));
   t.diagnostic(`Actual live native FLIP records before termination: ${live.length}`);
   assert.ok(live.length>0,'test must exercise production live FLIP records');
   const before=await readMixedBuffer(device,fields.waterBudget);plugin.prepare([box]);
   const encoder=device.createCommandEncoder();assert.equal(plugin.encode(encoder),true);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
   const after=await readMixedBuffer(device,fields.waterBudget),ledger=await plugin.readLedger(),killed=await readMixedBuffer(device,plugin.particles.diagnostics);
   assert.ok(ledger.externalVolume_m3<-.005,'box must remove actual resident water');
   const delta=((after[0]!+after[NB_VOLUME_COMPENSATION_FLOAT]!)-(before[0]!+before[NB_VOLUME_COMPENSATION_FLOAT]!))*l.cellSize_m.reduce((a,b)=>a*b,1);
   assert.ok(Math.abs(delta-ledger.externalVolume_m3)<1e-7,'native budget and external removal agree');
   assert.ok(new Uint32Array(killed.buffer)[0]!>0,'actual FLIP records inside terminator retire');
   const p=await uniformDetailField(fields.phi)!.storage.read(fields.phi),nx=solver.info.nx+1,ny=solver.info.ny+1;
   assert.ok(p[8+nx*(7+ny*8)]!>0,'water surface opens at box centre');
   // The released fixtures mostly use inverted boxes. Removing their
   // exterior must cover every current owner and live record, not just a
   // CPU-selected initial source envelope.
   plugin.prepare([{...box,inside:true}]);const inverted=device.createCommandEncoder();plugin.encode(inverted);device.queue.submit([inverted.finish()]);await device.queue.onSubmittedWorkDone();
   const allKilled=new Uint32Array((await readMixedBuffer(device,plugin.particles.diagnostics)).buffer);
   assert.equal(allKilled[0],live.length,'box plus its inverse retire every originally live record exactly once');
   const erased=await uniformDetailField(fields.phi)!.storage.read(fields.phi);
   assert.ok(erased[12+nx*(6+ny*12)]!>=0,'inverted terminator also removes water outside the original box');

  }finally{plugin.destroy();solver.destroy();}
 });
});
