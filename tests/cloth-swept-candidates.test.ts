import test from 'node:test';
import assert from 'node:assert/strict';
import {clothSweptCandidates,affineAxisBounds} from '../tools/cloth-swept-candidates';
import type {AffineVertex,Vec3} from '../tools/cloth-affine-ccd';
const p=(x:Vec3,d:Vec3=[0,0,0]):AffineVertex=>({x,d});
test('affine bounds round outward even when an endpoint increment is lost',()=>{
  const [lo,hi]=affineAxisBounds(2**52,.25,0);assert.ok(lo<=2**52);assert.ok(hi>2**52);
  const [negativeLo,negativeHi]=affineAxisBounds(-(2**52),-.25,0);assert.ok(negativeLo<-(2**52));assert.ok(negativeHi>=-(2**52));
  const [tinyLo,tinyHi]=affineAxisBounds(0,Number.MIN_VALUE,Number.MIN_VALUE);assert.ok(tinyLo<=-Number.MIN_VALUE);assert.ok(tinyHi>=2*Number.MIN_VALUE);
  assert.throws(()=>affineAxisBounds(Number.MAX_VALUE,Number.MAX_VALUE,0));
});
test('swept broadphase includes interior crossings whose start and end bounds are disjoint',()=>{
  const vs=[p([-2,-2,0]),p([2,-2,0]),p([0,2,0]),p([0,0,1],[0,0,-2]),p([-1,0,0]),p([1,0,0]),p([0,-1,1],[0,0,-2]),p([0,1,1],[0,0,-2])];
  const c=[...clothSweptCandidates(vs,[0,1,2],[4,5,6,7],.01)];
  assert.ok(c.some(c=>c.kind==='vt'&&c.ids[0]===3));
  assert.deepEqual(c.filter(c=>c.kind==='ee'),[{kind:'ee',ids:[4,5,6,7]}]);
  assert.ok(c.filter(c=>c.kind==='vt').every(c=>!c.ids.slice(1).includes(c.ids[0])));
});
test('swept hierarchy retains all analytically crossing pairs across many leaves',()=>{
  const vs:AffineVertex[]=[],triangles:number[]=[],edges:number[]=[];
  for(let i=0;i<40;i++){
    const k=vs.length,x=10*i;
    vs.push(p([x-2,-2,0]),p([x+2,-2,0]),p([x,2,0]),p([x,0,1],[0,0,-2]),p([x,-1,1],[0,0,-2]),p([x,1,1],[0,0,-2]));
    triangles.push(k,k+1,k+2);edges.push(k,k+1,k+4,k+5);
  }
  const c=[...clothSweptCandidates(vs,triangles,edges,.1)];
  for(let i=0;i<40;i++)assert.ok(c.some(c=>c.kind==='vt'&&c.ids[0]===6*i+3&&c.ids[1]===6*i));
  assert.equal(new Set(c.map(c=>c.kind+':'+c.ids.join(','))).size,c.length);
  assert.throws(()=>[...clothSweptCandidates([p([Infinity,0,0])],[],[],0)]);
  assert.throws(()=>[...clothSweptCandidates([p([0,0,0])],[0,0,2],[],0)]);
  assert.deepEqual([...clothSweptCandidates([],[],[],.1)],[]);
});

test('swept tree has no omissions against exhaustive small-mesh interval bounds',()=>{
  let seed=1729;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed%21)-10;};
  const vs=Array.from({length:54},()=>p([random(),random(),random()],[random(),random(),random()]));
  const triangles=Array.from({length:54},(_,i)=>i),edges=Array.from({length:54},(_,i)=>i);
  const actual=new Set([...clothSweptCandidates(vs,triangles,edges,.125)].map(c=>c.kind+':'+c.ids.join(',')));
  // Independent O(n²) reference, no hierarchy and no production box helper.
  const intersects=(a:number[],b:number[])=>[0,1,2].every(k=>{
    const endpoints=(ids:number[])=>ids.flatMap(i=>[vs[i]!.x[k]!,vs[i]!.x[k]!+vs[i]!.d[k]!]);
    const aa=endpoints(a),bb=endpoints(b);
    return Math.max(...aa)+.25>=Math.min(...bb)&&Math.max(...bb)+.25>=Math.min(...aa);
  });
  for(let v=0;v<vs.length;v++)for(let t=0;t<triangles.length;t+=3){
    const face=triangles.slice(t,t+3);
    if(!face.includes(v)&&intersects([v],face))assert.ok(actual.has('vt:'+[v,...face].join(',')));
  }
  for(let i=0;i<edges.length;i+=2)for(let j=i+2;j<edges.length;j+=2){
    const a=edges.slice(i,i+2),b=edges.slice(j,j+2);
    if(!a.some(v=>b.includes(v))&&intersects(a,b))assert.ok(actual.has('ee:'+[...a,...b].join(',')));
  }
});
