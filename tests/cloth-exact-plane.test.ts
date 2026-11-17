import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareExactAffinePlaneCheck} from '../tools/cloth-exact-plane';
import {affineEdgeEdgeCCD,type AffineVertex,type Vec3} from '../tools/cloth-affine-ccd';
const p=(x:Vec3,d:Vec3=[0,0,0]):AffineVertex=>({x,d});
test('exact dyadic support test preserves strict boundaries without normalization or overflow',()=>{
  const a=[p([0,0,0])],b=[p([1,0,0])];
  for(const scale of [Number.MIN_VALUE,1e-200,1,1e200,Number.MAX_VALUE]){
    assert.equal(prepareExactAffinePlaneCheck(a,b,1)([scale,0,0],0,1),false);
    assert.equal(prepareExactAffinePlaneCheck(a,b,1-Number.EPSILON)([scale,0,0],0,1),true);
    assert.equal(prepareExactAffinePlaneCheck(a,b,1+Number.EPSILON)([scale,0,0],0,1),false);
  }
  assert.equal(prepareExactAffinePlaneCheck(a,b,0)([0,0,0],0,1),false);
  const large=[p([Number.MAX_VALUE,0,0])],negative=[p([-Number.MAX_VALUE,0,0])];
  assert.equal(prepareExactAffinePlaneCheck(large,negative,1)([1,0,0],0,1),true);
});
test('exact endpoint supports require one orientation and original affine motion',()=>{
  const a=[p([-1,0,0],[2,0,0])],b=[p([1,0,0],[-2,0,0])];
  for(const same of [false,true])assert.equal(prepareExactAffinePlaneCheck(a,b,.1,same)([1,0,0],0,1),false);
  assert.equal(prepareExactAffinePlaneCheck(a,b,.1,true)([1,0,0],0,.4),true);
  const origin=[p([2**52,0,0])],moving=[p([2**52+1,0,0],[-.25,0,0])];
  // Rounded x+t*d can erase this quarter-unit movement; exact inputs retain it.
  assert.equal(prepareExactAffinePlaneCheck(origin,moving,.8,true)([1,0,0],0,1),false);
  assert.equal(prepareExactAffinePlaneCheck(origin,moving,.7,true)([1,0,0],0,1),true);
});
test('exact affine prefix validation covers the entire requested prefix',()=>{
  const a=p([-1,0,0],[2,0,0]),b=p([1,0,0],[-2,0,0]);
  const safe=affineEdgeEdgeCCD(a,a,b,b,{thickness:.1,maxAlpha:.4,exactEndpoints:true,sameTimeEndpoints:true});
  assert.equal(safe.alpha,.4);assert.equal(safe.reason,'separated');
  const unsafe=affineEdgeEdgeCCD(a,a,b,b,{thickness:.1,maxAlpha:1,exactEndpoints:true,sameTimeEndpoints:true});
  assert.equal(unsafe.reason,'unresolved');assert.ok(unsafe.alpha<=.475);
});
