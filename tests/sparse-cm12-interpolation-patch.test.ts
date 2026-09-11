import assert from 'node:assert/strict';
import test from 'node:test';
import {compileCM12InterpolationPatch, evaluateCM12InterpolationPatch, type PatchPoint} from '../lib/methods/adaptive-mass/sparse-cm12-interpolation-patch';

export const patchFixtures: PatchPoint[][] = [];
for (const width of [1,2,4,8]) for (const kind of ['box','wedge','pyramid','frustum']) {
  const points = Array.from({length:8},(_,c):PatchPoint => {
    const x=c&1,y=(c>>1)&1,z=c>>2;
    const a = x ? width : kind==='box' ? width : kind==='frustum' ? 2*width : 0;
    const b = x || kind==='box' || kind==='wedge' ? width : kind==='frustum' ? 2*width : 0;
    return [12+1.5*width*x,16+a*(y-.5),20+b*(z-.5)];
  });
  for (const permutation of [[0,1,2],[1,2,0],[2,0,1]]) for (const flip of [false,true]) {
    const transformed=Array.from({length:8},(_,c):PatchPoint=>{
      const bits=[c&1,(c>>1)&1,c>>2];
      const originalBits=permutation.map(axis=>bits[axis]!);
      if(flip) originalBits[0]=1-originalBits[0]!;
      const point=points[originalBits[0]!+2*originalBits[1]!+4*originalBits[2]!]!;
      const result=[0,0,0];
      permutation.forEach((axis,k)=>{result[axis]=point[k]!*(flip&&k===0?-1:1);});
      return result as unknown as PatchPoint;
    });
    patchFixtures.push(transformed);
  }
}
test('compiled boxes, wedges, pyramids and frusta reproduce trilinear geometry and affine fields',()=>{
  for(const nodes of patchFixtures){
    const patch=compileCM12InterpolationPatch(nodes);assert.ok(patch);
    for(const t of [0,.25,.5,.75,1]) for(const u of [0,.25,.5,1]) for(const v of [0,.25,1]){
      const param=[t,u,v];
      const truth=nodes.map((_,c)=>param.reduce((w,q,k)=>w*(c&(1<<k)?q:1-q),1));
      const q=[0,1,2].map(k=>nodes.reduce((s,n,i)=>s+truth[i]!*n[k]!,0)) as unknown as PatchPoint;
      const weights=evaluateCM12InterpolationPatch(patch,q);assert.ok(weights);
      assert.ok(weights.every(x=>x>=0));assert.ok(Math.abs(weights.reduce((a,b)=>a+b)-1)<1e-12);
      for(let k=0;k<3;k++) assert.ok(Math.abs(nodes.reduce((s,n,i)=>s+weights[i]!*n[k]!,0)-q[k]!)<1e-11);
    }
    for(const node of nodes){const w=evaluateCM12InterpolationPatch(patch,node);assert.ok(w);}
    assert.equal(evaluateCM12InterpolationPatch(patch,[1000,1000,1000]),null);
  }
});
test('compiler refuses warped and degenerate geometry',()=>{
 const bad=patchFixtures[0]!.map(p=>[...p]) as [number,number,number][];bad[7]![0]+=.125;
 assert.equal(compileCM12InterpolationPatch(bad),null);
 assert.equal(compileCM12InterpolationPatch(Array(8).fill([0,0,0])),null);
});
