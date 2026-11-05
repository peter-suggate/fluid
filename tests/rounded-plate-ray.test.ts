import test from 'node:test';import assert from 'node:assert/strict';
import {intersectRoundedPlate} from '../lib/core/rounded-plate-ray';
import type {SolidBoxCut} from '../lib/core/solid-cuts';
const base={dimensions_m:{x:.08,y:.003,z:.04},position_m:{x:0,y:0,z:0},orientation:{x:0,y:0,z:0,w:1}};
test('shared rounded plate ray hits exact face, cylinder, spherical corner and grazing rim',()=>{
 const face=intersectRoundedPlate(base,{origin:{x:0,y:.1,z:0},direction:{x:0,y:-1,z:0}})!;assert.ok(Math.abs(face.t_m-.097)<1e-13);assert.deepEqual(face.normal,{x:0,y:1,z:0});
 for(const y of[.002,.003*(1-1e-8)]){const hit=intersectRoundedPlate(base,{origin:{x:0,y,z:.1},direction:{x:0,y:0,z:-1}})!;assert.ok(hit);assert.ok(Math.abs(hit.t_m-(.06-Math.sqrt(.003**2-y*y)))<1e-12);}
 const x=.081,z=.041,hit=intersectRoundedPlate(base,{origin:{x,y:.1,z},direction:{x:0,y:-1,z:0}})!;assert.ok(Math.abs(hit.t_m-(.1-Math.sqrt(.003**2-.001**2-.001**2)))<1e-12);
});
test('local cut planes expose inward normals and overlapping subtractions hide internal boundaries',()=>{
 const cut=(x:number,half:number):SolidBoxCut=>({center_m:{x,y:0,z:0},halfExtents_m:{x:half,y:.02,z:.08},orientation:{x:0,y:0,z:0,w:1}}),ray={origin:{x:0,y:0,z:0},direction:{x:1,y:0,z:0}};
 const a=intersectRoundedPlate({...base,cuts:[cut(0,.02)]},ray)!;assert.equal(a.t_m,.02);assert.deepEqual(a.normal,{x:-1,y:0,z:0});
 const b=intersectRoundedPlate({...base,cuts:[cut(0,.02),cut(.02,.01)]},ray)!;assert.equal(b.t_m,.03);assert.deepEqual(b.normal,a.normal);
 assert.equal(intersectRoundedPlate({...base,cuts:[cut(0,.02)]},{origin:{x:0,y:.1,z:0},direction:{x:0,y:-1,z:0}}),null);
});
