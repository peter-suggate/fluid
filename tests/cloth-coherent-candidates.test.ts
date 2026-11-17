import test from 'node:test';
import assert from 'node:assert/strict';
import {CoherentCandidateReference} from '../tools/cloth-coherent-candidates';
import {clothSweptCandidates} from '../tools/cloth-swept-candidates';
import type {AffineVertex,Vec3} from '../tools/cloth-affine-ccd';
const p=(x:Vec3,d:Vec3=[0,0,0]):AffineVertex=>({x,d});
const key=(c:{kind:string;ids:readonly number[]})=>c.kind+':'+c.ids.join(',');
function verify(cache:CoherentCandidateReference,vs:AffineVertex[],ts:number[],es:number[]){
 const r=cache.update(vs),keys=new Set(r.pairs.map(key)),boxes=cache.snapshot().boxes;
 // Independent exhaustive padded-box oracle after EVERY transaction.
 const overlap=(a:number[],b:number[])=>[0,1,2].every(k=>Math.min(...a.map(i=>boxes[i]!.lo[k]!))<=Math.max(...b.map(i=>boxes[i]!.hi[k]!))&&Math.min(...b.map(i=>boxes[i]!.lo[k]!))<=Math.max(...a.map(i=>boxes[i]!.hi[k]!)));
 for(let v=0;v<vs.length;v++)for(let t=0;t<ts.length;t+=3){const f=ts.slice(t,t+3);if(!f.includes(v)&&overlap([v],f))assert.ok(keys.has('vt:'+[v,...f].join(',')));}
 for(let i=0;i<es.length;i+=2)for(let j=i+2;j<es.length;j+=2){const a=es.slice(i,i+2),b=es.slice(j,j+2);if(!a.some(v=>b.includes(v))&&overlap(a,b))assert.ok(keys.has('ee:'+[...a,...b].join(',')));}
 for(const c of clothSweptCandidates(vs,ts,es,cache.thickness))assert.ok(keys.has(key(c)),'Actual swept candidate omitted');
 return r;
}
test('coherent candidate cache preserves padded and actual coverage through escapes, shrink and reappearance',()=>{
 const ts=[0,1,2,3,4,5],es=[0,1,2,3,4,5],cache=new CoherentCandidateReference(ts,es,.5,.001);
 let seed=712;const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
 for(let iteration=0;iteration<80;iteration++){
  const vs=Array.from({length:7},()=>p([rand()*4,rand()*4,rand()*4],[rand()-.5,rand()-.5,rand()-.5]));
  verify(cache,vs,ts,es);
  const stable=verify(cache,vs,ts,es);assert.equal(stable.changedVertices,0);assert.equal(stable.appended,0);
 }
 cache.reset();assert.equal(cache.snapshot().pairs.length,0);
});
test('unchanged padded boxes retain a pair becoming relevant inside the skin',()=>{
 const ts=[0,1,2],cache=new CoherentCandidateReference(ts,[],1,0);
 const a=[p([-1,-1,0]),p([1,-1,0]),p([0,1,0]),p([0,0,.75])];
 verify(cache,a,ts,[]);const b=[...a];b[3]=p([0,0,.5],[0,0,-.6]);
 const r=verify(cache,b,ts,[]);assert.equal(r.changedVertices,0);assert.ok(r.pairs.some(c=>c.ids[0]===3));
});
test('overflow performs a complete rebuild or leaves the old transaction intact',()=>{
 const ts=[0,1,2],cache=new CoherentCandidateReference(ts,[],0,0,1);
 const base=[p([-1,-1,0]),p([1,-1,0]),p([0,1,0]),p([0,0,0]),p([0,0,10])];
 verify(cache,base,ts,[]);
 const swapped=[...base];swapped[3]=p([0,0,10]);swapped[4]=p([0,0,0]);
 assert.equal(verify(cache,swapped,ts,[]).rebuilt,true);
 const before=cache.snapshot(),tooMany=[...swapped];tooMany[3]=p([0,0,0]);
 assert.throws(()=>cache.update(tooMany),/exceeds capacity/);assert.deepEqual(cache.snapshot(),before);
 assert.throws(()=>cache.update([...swapped,p([0,0,0])]),/topology change/);
 const bad=[...swapped];bad[4]=p([NaN,0,0]);assert.throws(()=>cache.update(bad));assert.deepEqual(cache.snapshot(),before);
});
