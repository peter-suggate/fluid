import test from 'node:test';
import assert from 'node:assert/strict';
import { ClothRuntime } from '../lib/cloth/runtime';
import { rebaseClothRuntime } from '../lib/cloth/rebase';
import { cutClothShellBox } from '../lib/cloth/edit';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import type { ClothDocument } from '../lib/cloth/types';

test('live cloth topology adoption preserves the source clock, surviving rod frames and water after a cut',()=>{
 const document:ClothDocument={schemaVersion:'1.0.0',id:'edit',materials:[DEFAULT_CLOTH_MATERIAL],scripts:[{kind:'translate',group:0,start_s:0,end_s:1,vector:[0,.01,0]}],geometry:{kind:'inline',data:{
  restPositions_m:[-1,1,-1,1,1,-1,-1,1,1,1,1,1],triangles:[0,2,1,1,2,3],rodOffsets:[0,2],rodIndices:[0,1],fixed:[3,3,1,1]}}};
 const runtime=new ClothRuntime(document);runtime.step(.001,[],{gravity_m_s2:[0,0,0]});
 runtime.state.twist_rad[0]=.23;runtime.state.angularVelocity_rad_s[0]=.03;
 const data={...runtime.prepared.data,initialPositions_m:Array.from(runtime.state.positions_m),initialVelocities_m_s:Array.from(runtime.state.velocities_m_s),initialWaterVolume_m3:Array.from(runtime.state.waterVolume_m3)};
 const cut=cutClothShellBox(document,data,runtime.state.positions_m,{minimum_m:[-.2,.9,-.2],maximum_m:[.2,1.1,.2]});
 const rebased=rebaseClothRuntime(runtime,cut.document,cut.topology);
 assert.equal(rebased.identity.tick,runtime.identity.tick);assert.equal(rebased.state.time_s,.001);
 assert.equal(rebased.state.topologyRevision,runtime.state.topologyRevision+1);
 assert.equal(rebased.state.twist_rad[0],.23);assert.equal(rebased.state.angularVelocity_rad_s[0],.03);
 assert.deepEqual(rebased.state.positions_m,cut.positions_m);
 assert.equal(rebased.state.waterVolume_m3.reduce((a,b)=>a+b),0);
 rebased.step(.001,[],{gravity_m_s2:[0,0,0]});assert.equal(rebased.state.time_s,.002);
 assert.ok(rebased.state.positions_m[1]!>runtime.state.positions_m[1]!);
});
