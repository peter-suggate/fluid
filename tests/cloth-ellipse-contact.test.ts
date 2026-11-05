import test from 'node:test';import assert from 'node:assert/strict';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {prepareClothTopology,createClothState} from '../lib/cloth/topology';
import {resolveClothContacts,sampleClothCollider} from '../lib/cloth/contact';
import {clothRodCapsuleGeometry} from '../lib/cloth/rod-geometry';
import {sampleClippedRodCollider} from '../lib/cloth/clipped-contact';
const material={...DEFAULT_CLOTH_MATERIAL,rodRadius_m:.1,rodRadiusB_m:.02};
const data={restPositions_m:[-1,0,0,1,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]};
test('elliptical yarn contact preserves narrow-axis clearance and uses rotated physical material extent',()=>{
 const p=prepareClothTopology(data,[material]),s=createClothState(p),old=s.positions_m.slice(),plane={kind:'plane' as const,normal:[0,0,1] as const,offset_m:-.03};
 const clear=resolveClothContacts(p,s,old,.01,[plane],false);assert.equal(clear.contacts,0);assert.deepEqual(s.positions_m,old);
 s.twist_rad[0]=Math.PI/2;const hit=resolveClothContacts(p,s,old,.01,[plane],false);assert.ok(hit.contacts>0);assert.ok(s.positions_m[2]!>=.07-1e-8&&s.positions_m[5]!>=.07-1e-8);
});
test('exact ellipse-to-sphere separation is shared with analytic surface geometry',()=>{
 const p=prepareClothTopology(data,[material]),s=createClothState(p),shape=clothRodCapsuleGeometry(p,data,s.positions_m,0,0,{rodTwist_rad:s.twist_rad,rodDirectors:s.rodDirectors,rodTangents:s.rodTangents}),sphere={kind:'sphere' as const,center_m:[0,0,.06] as const,radius_m:.01},distance=sampleClippedRodCollider(shape,sphere,0,sampleClothCollider);
 assert.ok(Math.abs(distance.distance_m-.03)<1e-9);assert.ok(distance.lowerBound_m>0);assert.ok(Math.abs(distance.point[2]!-.02)<1e-9);
});
test('an initially interior sphere separates through the ellipse closest boundary',()=>{
 const p=prepareClothTopology(data,[material]),s=createClothState(p),old=s.positions_m.slice(),sphere={kind:'sphere' as const,center_m:[0,0,0] as const,radius_m:.01};
 const report=resolveClothContacts(p,s,old,.01,[sphere],false);assert.ok(report.contacts>0);assert.ok(Math.abs(s.positions_m[0]!-old[0]!)<1e-8&&Math.abs(s.positions_m[3]!-old[3]!)<1e-8);
 assert.ok(Math.abs(s.positions_m[2]!)>=.03-1e-8&&Math.abs(s.positions_m[5]!)>=.03-1e-8);
});
