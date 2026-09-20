import test from "node:test";
import assert from "node:assert/strict";
import {initialUniformPageDomain} from "../lib/methods/uniform/uniform-page-domain";

test("page-owned cells and vertices have one owner under reordered and partial pages",()=>{
 for(const dims of [[32,32,32],[64,64,64],[80,48,96]] as const){
  const d=initialUniformPageDomain(dims,32,true);
  const cells=new Uint8Array(dims[0]*dims[1]*dims[2]);
  const vertices=new Uint8Array((dims[0]+1)*(dims[1]+1)*(dims[2]+1));
  for(let page=0;page<d.count;page++){
   const slot=d.words[16+16*d.capacity+page]!;
   const origin=[0,1,2].map(a=>d.words[16+16*slot+a]!*32);
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++){
    const q=[origin[0]!+x,origin[1]!+y,origin[2]!+z];
    if(q.every((v,a)=>v<dims[a]!)&&x<32&&y<32&&z<32)cells[q[0]!+dims[0]*(q[1]!+dims[1]*q[2]!)]!++;
    if(q.some((v,a)=>v>dims[a]!))continue;
    if([x,y,z].some((v,a)=>v>=32&&q[a]!==dims[a]))continue;
    vertices[q[0]!+(dims[0]+1)*(q[1]!+(dims[1]+1)*q[2]!)]!++;
   }
  }
  assert.ok(cells.every(n=>n===1),`cell ownership ${dims}`);
  assert.ok(vertices.every(n=>n===1),`vertex ownership ${dims}`);
 }
});

test("mini64 contiguous specialization derives its dispatch from the page generation",()=>{
 const d=initialUniformPageDomain([64,64,64]);
 assert.equal(d.count,8);assert.equal(d.contiguous,true);
 assert.deepEqual(Array.from(d.words.slice(0,3)),[16,16,16]);
 assert.deepEqual(Array.from(d.words.slice(4,7)),[17,17,17]);
});
