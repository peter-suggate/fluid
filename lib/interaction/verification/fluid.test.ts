import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createEmptyScene } from '../../core/empty-scene';
import { interactionCatalog, interactionRuntimes } from '../installed';
import { Box3DWorld } from '../adapters/box3d/world';
import type { FluidExchange } from '../contracts';
test('native feedback is atomic, generation checked and applied exactly once',async()=>{
 const scene=createEmptyScene({extents_m:{x:4,y:3,z:4},finestCellSize_m:.25});scene.scenery=undefined;
 const entities=[-1,1].map((x,i)=>interactionCatalog.createEntity({id:`body${i}`,kind:'fluid.crate',transform:{position_m:{x,y:1,z:0},orientation:{x:0,y:0,z:0,w:1}}}));
 const world=await Box3DWorld.create(scene,entities.map(entity=>({entity,recipe:interactionRuntimes.find(f=>f.type==='fluid.rigid-body')!.body!(entity)})),7);
 const identity={generation:7,tick:1,solidRevision:1};
 const packets:FluidExchange[]=world.renderBodies().map(b=>({...identity,bodyId:b.description.id,bodyGeneration:7,linearImpulse_N_s:{x:75,y:0,z:0},angularImpulse_N_m_s:{x:0,y:0,z:0},displacedVolume_m3:0,fluidVelocity_m_s:{x:0,y:0,z:0},policy:'pressure-traction'}));
 try{
  assert.throws(()=>world.applyFluid([packets[0],{...packets[1],generation:6}],identity),/Stale/);assert.equal(world.snapshots()[0].linearVelocity_m_s.x,0);
  assert.throws(()=>world.applyFluid([packets[0]],identity),/Incomplete/);assert.equal(world.snapshots()[0].linearVelocity_m_s.x,0);
  world.applyFluid(packets,identity);assert.ok(Math.abs(world.snapshots()[0].linearVelocity_m_s.x-1)<1e-6);
  assert.throws(()=>world.applyFluid(packets,identity),/already applied/);
 }finally{world.destroy();}
});
