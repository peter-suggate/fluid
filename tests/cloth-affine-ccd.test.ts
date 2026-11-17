import test from 'node:test';
import assert from 'node:assert/strict';
import {affineVertexTriangleCCD as vt,affineEdgeEdgeCCD as ee,type AffineVertex,type Vec3} from '../tools/cloth-affine-ccd';
const p=(x:Vec3,d:Vec3=[0,0,0]):AffineVertex=>({x,d});
const tri=[p([-2,-2,0]),p([2,-2,0]),p([0,2,0])] as const;

test('affine CCD finds whole-triangle crossing despite separated endpoints',()=>{
  const r=vt(p([0,0,1],[0,0,-2]),...tri,{thickness:.1});
  assert.equal(r.reason,'unresolved');
  assert.ok(r.alpha<=.45 && r.alpha>.45-1e-6,JSON.stringify(r));
  assert.equal(vt(p([0,0,1],[1,0,0]),...tri,{thickness:.1}).alpha,1);
  assert.equal(vt(p([0,0,.1]),...tri,{thickness:.1}).alpha,0);
  assert.equal(vt(p([0,0,0]),...tri,{thickness:.1}).alpha,0);
});

test('affine CCD covers edge interiors, tangency, and collapsed primitives',()=>{
  const a=p([-1,0,0]),b=p([1,0,0]);
  const r=ee(a,b,p([0,-1,1],[0,0,-2]),p([0,1,1],[0,0,-2]),{thickness:.1});
  assert.ok(r.alpha<=.45 && r.alpha>.45-1e-6,JSON.stringify(r));
  // Two collapsed segments graze at t=.5, while both endpoints are safe.
  const fixed=p([0,0,0]),moving=p([-1,.1,0],[2,0,0]);
  const grazing=ee(fixed,fixed,moving,moving,{thickness:.1});
  assert.ok(grazing.alpha<.5 && grazing.alpha>.499,JSON.stringify(grazing));
  assert.equal(ee(fixed,fixed,p([0,1,0]),p([0,1,0]),{thickness:.1}).alpha,1);
  const collapsed=p([0,0,0]);
  assert.equal(vt(p([0,0,1]),collapsed,collapsed,collapsed,{thickness:.1}).alpha,1);
});

test('affine CCD removes common translation and fails closed on budgets and invalid data',()=>{
  const v=p([0,0,1],[0,0,-2]),base=vt(v,...tri,{thickness:.125});
  const shift=(v:AffineVertex):AffineVertex=>({x:v.x.map(x=>x+2**40) as unknown as Vec3,d:v.d.map(x=>x+2**30) as unknown as Vec3});
  assert.deepEqual(vt(shift(v),...tri.map(shift) as [AffineVertex,AffineVertex,AffineVertex],{thickness:.125}),base);
  const exhausted=vt(v,...tri,{thickness:.125,maxIntervals:1});
  assert.equal(exhausted.alpha,0);assert.equal(exhausted.reason,'budget');
  assert.equal(vt(p([NaN,0,0]),...tri,{thickness:.1}).reason,'nonfinite');
  assert.throws(()=>vt(v,...tri,{thickness:-1}));
  assert.throws(()=>vt(v,...tri,{thickness:.1,maxIntervals:0}));
});

test('affine CCD prefixes agree with analytic point-pair roots across scales',()=>{
  // Independent quadratic root, not dense samples of the oracle's geometry.
  for(const scale of [1e-8,1,1e8]) for(let i=0;i<32;i++) {
    const radius=(.1+i*.01)*scale,offset=radius*.7;
    const origin=p([0,0,0]),moving=p([-2*scale,offset,0],[4*scale,0,0]);
    const root=(2*scale-Math.sqrt(radius*radius-offset*offset))/(4*scale);
    const r=ee(origin,origin,moving,moving,{thickness:radius});
    assert.ok(r.alpha<=root && r.alpha>root-1e-6,JSON.stringify({scale,i,r,root}));
  }
});

test('affine support tests cover rotated crossings and independently collapsing triangles',()=>{
  for(let i=0;i<32;i++){
    const a=.173*i,b=.317*i,ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);
    const rotate=([x,y,z]:Vec3):Vec3=>[cb*x+sb*(sa*y+ca*z),ca*y-sa*z,-sb*x+cb*(sa*y+ca*z)];
    const transform=(v:AffineVertex):AffineVertex=>({x:rotate(v.x),d:rotate(v.d)});
    const collapsing=tri.map(v=>p(v.x,[-2*v.x[0],-2*v.x[1],0])).map(transform) as [AffineVertex,AffineVertex,AffineVertex];
    const r=vt(transform(p([0,0,.4],[0,0,-1])),...collapsing,{thickness:.1});
    assert.ok(r.alpha<=.3 && r.alpha>.3-1e-6,JSON.stringify({i,r}));
  }
});

test('affine prefix cannot bridge a gap, including exhaustion after safe progress',()=>{
  const r=vt(p([0,0,3],[0,0,-4]),...tri,{thickness:.1,maxIntervals:3});
  assert.equal(r.reason,'budget');assert.equal(r.alpha,.5);
  assert.equal(vt(p([0,0,1],[0,0,-2]),...tri,{thickness:.1,maxDepth:0}).alpha,0);
  const a=p([-1,0,0]),b=p([1,0,0]);
  assert.equal(ee(a,b,p([0,-1,0]),p([0,1,0]),{thickness:0}).alpha,0);
  assert.equal(vt(p([0,0,0],[0,0,1]),...tri,{thickness:0}).alpha,0,'initially touching cannot certify a strict prefix');
  const huge=p([1e308,0,0]),opposite=p([-1e308,0,0]);
  assert.equal(ee(huge,huge,opposite,opposite,{thickness:0}).reason,'nonfinite');
  const tiny=p([1e-310,0,0]);
  assert.equal(ee(tiny,tiny,tiny,tiny,{thickness:0}).alpha,0);
});

test('same-time support planes remove correlated-motion subdivision without allowing orientation reversal',()=>{
  const points=[p([0,0,0]),p([1,0,0],[-1,0,0]),p([2,0,0],[-1.5,0,0]),p([3,0,0],[-1.5,0,0])] as const;
  const old=ee(...points,{thickness:.1}),next=ee(...points,{thickness:.1,sameTimeEndpoints:true});
  assert.equal(next.alpha,1);assert.equal(old.alpha,1);assert.equal(next.intervals,1);assert.ok(old.intervals>next.intervals);
  const left=p([-1,0,0],[2,0,0]),right=p([1,0,0],[-2,0,0]);
  const crossing=ee(left,left,right,right,{thickness:.1,sameTimeEndpoints:true});
  assert.ok(crossing.alpha<=.475 && crossing.alpha>.475-1e-6,'opposite endpoint orientations cannot certify crossing');
});
