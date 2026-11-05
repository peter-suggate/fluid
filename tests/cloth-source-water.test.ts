import assert from 'node:assert/strict';
import test from 'node:test';
import {planSourceWater,sourceWaterOperations,sourceTerminationBoundaryBounds} from '../lib/cloth/source-water';
import type {ClothVector} from '../lib/cloth/types';
import type {WetClothSourceField,WetClothSourceProgram} from '../lib/cloth/source-types';
const source:WetClothSourceField={index:0,kind:'box',usage:'source',group:0,inside:false,sampled:true,root:true,center_m:[.5,.5,.5],halfExtents_m:[2,2,2],sourceAttributes:{}};
const vertices=Array.from({length:8},(_,i)=>[i&1,(i>>1)&1,(i>>2)&1] as ClothVector),cells=[{vertices:[0,1,2,3,4,5,6,7],capacity_m3:1}];
test('source transaction credits only new water momentum and removes only local mixture',()=>{
 const initial=Float32Array.from(vertices,p=>p[0]-.25),emit={fieldIndex:0,kind:'emit' as const,velocity_m_s:[2,-1,0] as ClothVector};
 const added=planSourceWater(initial,vertices,cells,[source],[emit],1);
 assert.equal(added.freeDelta_m3,.75);assert.deepEqual([...added.cellAddedMomentum_m4_s],[1.5,-.75,0]);assert.equal(added.cellRetainedVolume_m3[0],.25);
 const again=planSourceWater(added.phi_m,vertices,cells,[source],[emit],1);assert.equal(again.freeDelta_m3,0);
 const removed=planSourceWater(initial,vertices,cells,[source],[emit,{...emit,kind:'terminate'}],1);
 assert.equal(removed.freeDelta_m3,-.25);assert.ok(removed.cellAddedMomentum_m4_s.every(v=>v===0));assert.equal(removed.cellRetainedVolume_m3[0],0);
});
test('source scheduling prevents initial/native-drop duplication while continuous sources replenish',()=>{
 const program:WetClothSourceProgram={schemaVersion:1,sourceId:'test',fields:[source],scripts:[],sourceOffset_m:[0,0,0],timeStep_s:.01,requiredCapabilities:[],initialSdfFieldIndices:[0]};
 const instant={fieldIndex:0,kind:'instant' as const,start_s:0,end_s:0,velocity_m_s:[1,0,0] as ClothVector};
 assert.equal(sourceWaterOperations(program,[instant]).length,0);
 assert.equal(sourceWaterOperations(program,[{...instant,kind:'interval',end_s:.01}]).length,1);
 assert.equal(sourceWaterOperations({...program,scheduledFieldIndices:[0]},[{...instant,start_s:2,end_s:2}]).length,0);
});

test('inverted terminators pin six finite boundary bands rather than the full domain',()=>{
 const field:WetClothSourceField={...source,usage:'terminator',inside:true,center_m:[0,0,0],halfExtents_m:[100,200,300],roundingRadius_m:.2,axis:[0,1,0],angle_rad:0};
 const bounds=sourceTerminationBoundaryBounds([field],[.01,.02,.03]);assert.equal(bounds.length,6);
 for(let a=0;a<3;a++)for(let side=0;side<2;side++){const b=bounds[2*a+side]!,face=(side?1:-1)*field.halfExtents_m![a]!;assert.ok(Math.abs((b.max[a]!-b.min[a]!)-.64)<1e-10);assert.ok(b.min[a]!<face&&b.max[a]!>face);}
 assert.equal(bounds.some(b=>b.min.every(v=>v<0)&&b.max.every(v=>v>0)),false,'interior is not unnecessarily pinned fine');
 assert.throws(()=>sourceTerminationBoundaryBounds([{...field,kind:'file'}],[.01,.01,.01]),/Unknown/);
});
