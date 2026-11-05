import test from 'node:test';
import assert from 'node:assert/strict';
import {rigidRayIntervals,rigidRayUnionLength} from '../lib/core/rigid-ray-intervals';
import {SCENE_SHAPE_TABLE,type SceneShapeName} from '../lib/core/scene-shape';
import {quaternionInverseRotate} from '../lib/core/rigid-body';
import {applySolidBoxCuts} from '../lib/core/solid-cuts';
import type {RigidBodyDescription} from '../lib/core/model';

const plate:Parameters<typeof rigidRayIntervals>[0]={shape:'rounded-plate',dimensions_m:{x:.09,y:.015,z:.09},position_m:{x:0,y:.375,z:0},orientation:{w:1,x:0,y:0,z:0}};

test('analytic slab thickness survives grid phase where eight-point cell coverage vanishes',()=>{
 const h=.05;
 for(const center of [.35,.36,.375,.384,.399]){
  const body={...plate,position_m:{x:0,y:center,z:0}};
  let length=0;for(let cell=0;cell<16;cell++)length+=rigidRayUnionLength([body],{x:0,y:cell*h,z:0},{x:0,y:1,z:0},[0,h]);
  assert.ok(Math.abs(length-.03)<2e-16,`${center}: ${length}`);
 }
 const samples=[.375-.4*h,.375+.4*h];assert.ok(samples.every(y=>Math.abs(y-.375)>.015),'regression must exercise the old missed-thickness case');
});

test('interval union counts overlapping bodies once and retains narrow cuts',()=>{
 const ray={x:0,y:1,z:0},origin={x:0,y:.3,z:0};
 const shifted={...plate,position_m:{x:0,y:.385,z:0}};
 assert.ok(Math.abs(rigidRayUnionLength([plate,shifted],origin,ray,[0,.2])-.04)<1e-15);
 const cut={center_m:{x:0,y:0,z:0},halfExtents_m:{x:.001,y:.002,z:.001},orientation:{w:1,x:0,y:0,z:0}};
 const intervals=rigidRayIntervals({...plate,cuts:[cut]},origin,ray,[0,.2]);
 assert.equal(intervals.length,2);assert.ok(Math.abs(intervals.reduce((s,[a,b])=>s+b-a,0)-.026)<1e-15);
 assert.deepEqual(rigidRayIntervals({...plate,cuts:[cut]},{x:.04,y:.3,z:0},ray,[0,.2]),rigidRayIntervals(plate,{x:.04,y:.3,z:0},ray,[0,.2]));
});

test('all shape intervals agree with authoritative distance membership under rotation, cuts and inversion',()=>{
 let seed=17;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 const angle=.7,axisLength=Math.hypot(1,2,3),orientation={w:Math.cos(angle/2),x:Math.sin(angle/2)/axisLength,y:2*Math.sin(angle/2)/axisLength,z:3*Math.sin(angle/2)/axisLength};
 for(const shape of Object.keys(SCENE_SHAPE_TABLE) as SceneShapeName[])for(const inverted of [false,true]){
  const body:Parameters<typeof rigidRayIntervals>[0]={shape,dimensions_m:{x:.3,y:.4,z:shape==='cup'?.03:.25},position_m:{x:.07,y:-.04,z:.11},orientation,fluidBoundary:inverted?'interior':undefined,cuts:[{center_m:{x:0,y:.03,z:0},halfExtents_m:{x:.08,y:.04,z:.35},orientation:{w:Math.cos(.3),x:0,y:0,z:Math.sin(.3)}}]};
  for(let ray=0;ray<45;ray++){
   const origin={x:random()-.5,y:random()-.5,z:random()-.5},direction={x:random()*2-1,y:random()*2-1,z:random()*2-1},intervals=rigidRayIntervals(body,origin,direction,[-1,1]);
   for(let sample=0;sample<51;sample++){
    const t=-1+2*(sample+.31)/51,p=quaternionInverseRotate(orientation,{x:origin.x+t*direction.x-body.position_m.x,y:origin.y+t*direction.y-body.position_m.y,z:origin.z+t*direction.z-body.position_m.z});
    const distance=applySolidBoxCuts((inverted?-1:1)*SCENE_SHAPE_TABLE[shape].distance_m(body.dimensions_m,p),body.cuts,p);
    if(Math.abs(distance)<1e-10)continue;
    assert.equal(intervals.some(([a,b])=>t>a&&t<b),distance<0,`${shape} inverted=${inverted} ray=${ray} t=${t}`);
   }
  }
 }
});

test('unknown interval shape fails explicitly',()=>{
 assert.throws(()=>rigidRayIntervals({...plate,shape:'unknown' as RigidBodyDescription['shape']},{x:0,y:0,z:0},{x:0,y:1,z:0},[0,1]),/Unsupported interval shape/);
});
