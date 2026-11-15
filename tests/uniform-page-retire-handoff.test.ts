import assert from "node:assert/strict";
import test from "node:test";
import {uniformPageRetireNext} from "../lib/methods/uniform/uniform-page-initialise";

// The retire handoff (uniformPageRetireNext, the host twin of prNext): a
// vertex a retired block held passes to the block that holds it after the
// commit, so no reader falls back to a skirt texel an earlier holder left.
// The reader here is the vertex slot rule (uniformVertexSlot, umVertexSlot,
// umWideVertexSlot): the vertex's own page (v-1)>>bits, else the first
// stored page of the walk over the plane axes, at local 0 on the stepped axes.

type Page=readonly number[];
const key=(q:Page)=>q.join(",");
const AIR=0.1;

interface World{
 readonly bits:number;
 /** Stored pages and their blocks. */
 readonly blocks:Map<string,number>;
 /** Block texels, keyed by block and local vertex. */
 readonly texels:Map<string,number>;
}
const texel=(block:number,l:Page)=>`${block}:${l.join(",")}`;
/** The vertex slot rule: the block and local texel a reader of vertex v takes, or undefined (the AIR sentinel). */
function holder(w:World,v:Page):{block:number;local:number[]}|undefined{
 const mask=(1<<w.bits)-1,own=v.map(x=>(x-1)>>w.bits),local=v.map(x=>((x-1)&mask)+1),plane=v.map(x=>(x&mask)===0);
 for(let k=0;k<8;k++){
  if([0,1,2].some(a=>((k>>a)&1)!==0&&!plane[a]))continue;
  const q=own.map((x,a)=>x+((k>>a)&1)),b=w.blocks.get(key(q));
  if(b!==undefined)return {block:b,local:local.map((x,a)=>((k>>a)&1)!==0?0:x)};
 }
 return undefined;
}
function read(w:World,v:Page):number{
 const h=holder(w,v);return h?w.texels.get(texel(h.block,h.local))??AIR:AIR;
}
/** A commit: claim pages (the AIR image over every texel of a fresh block), retire pages, then the handoff (with handoff). */
function commit(w:World,claim:readonly Page[],retire:readonly Page[],handoff:boolean,next:{block:number}):Set<number>{
 const before=new Set(w.blocks.keys()),retired=retire.map(q=>({q,b:w.blocks.get(key(q))!}));
 for(const {q} of retired)w.blocks.delete(key(q));
 const fresh=new Set<number>(),n=(1<<w.bits)+1;
 for(const q of claim){
  const b=next.block++;w.blocks.set(key(q),b);fresh.add(b);
  for(let x=0;x<n;x++)for(let y=0;y<n;y++)for(let z=0;z<n;z++)w.texels.set(texel(b,[x,y,z]),AIR);
 }
 if(!handoff)return fresh;
 const writes=new Set<string>();
 for(const {q,b} of retired)for(let x=0;x<n;x++)for(let y=0;y<n;y++)for(let z=0;z<n;z++){
  const l=[x,y,z];
  // Interior vertices have no other holder (the GPU pass skips them).
  if(l.every(c=>c>=1&&c<=n-2))continue;
  const to=uniformPageRetireNext(q,l,w.bits,p=>before.has(key(p)),p=>w.blocks.get(key(p))??-1,c=>fresh.has(c));
  if(!to)continue;
  const at=texel(to.block,to.local);
  // A holder texel takes one write in the pass, never a fresh block's AIR.
  assert.ok(!writes.has(at),`two retired blocks write ${at}`);writes.add(at);
  assert.ok(!fresh.has(to.block));
  w.texels.set(at,w.texels.get(texel(b,l))??AIR);
 }
 return fresh;
}

test("a retired block's vertex passes to its next holder, not a stale skirt (the 128^3 dam's phantom liquid)",()=>{
 for(const bits of [3,5]){
  const s=1<<bits,w:World={bits,blocks:new Map([["0,0,0",1],["1,0,0",2]]),texels:new Map()};
  // Vertex (s, 3, 3): page 0's high plane, page 1's low skirt. A BULK16
  // demote of page 0 once handed its deep phi to page 1's skirt; the
  // promote copied it back and left the skirt holding it.
  const v=[s,3,3];w.texels.set(texel(2,[0,3,3]),-0.1169);
  // Page 0 now holds air there, and the census retires it.
  w.texels.set(texel(1,[s,3,3]),AIR);
  assert.equal(read(w,v),AIR);
  const stale:World={bits,blocks:new Map(w.blocks),texels:new Map(w.texels)};
  commit(stale,[],[[0,0,0]],false,{block:3});
  assert.equal(read(stale,v),-0.1169,"without the handoff the reader takes the stale skirt");
  commit(w,[],[[0,0,0]],true,{block:3});
  assert.equal(read(w,v),AIR,`bits ${bits}`);
 }
});

test("every vertex keeps its value across claims and retires, AIR only where a fresh block or none now holds it",()=>{
 let seed=12345;const rand=()=>{seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x80000000;};
 for(const bits of [3,5]){
  const s=1<<bits,pages=3,next={block:1};
  const w:World={bits,blocks:new Map(),texels:new Map()};
  for(let x=0;x<pages;x++)for(let y=0;y<pages;y++)for(let z=0;z<pages;z++)if(rand()<0.6)w.blocks.set(key([x,y,z]),next.block++);
  // Every texel of every block, skirts included, holds its own value: a skirt a reader skips is stale.
  for(const b of w.blocks.values())for(let x=0;x<=s;x++)for(let y=0;y<=s;y++)for(let z=0;z<=s;z++)w.texels.set(texel(b,[x,y,z]),rand()-0.5);
  // The vertices on page planes (all the handoff can move), in the pages' span.
  const vertices:number[][]=[],coords=Array.from({length:pages*s+1},(_,i)=>i).filter(i=>[0,1,2,s-1].includes(i%s));
  for(const x of coords)for(const y of coords)for(const z of coords)if([x,y,z].some(c=>c%s===0))vertices.push([x,y,z]);
  for(let round=0;round<40;round++){
   const all:number[][]=[];for(let x=0;x<pages;x++)for(let y=0;y<pages;y++)for(let z=0;z<pages;z++)all.push([x,y,z]);
   const retire=all.filter(q=>w.blocks.has(key(q))&&rand()<0.25),claim=all.filter(q=>!w.blocks.has(key(q))&&rand()<0.25);
   const before=new Map(vertices.map(v=>[key(v),read(w,v)]));
   const fresh=commit(w,claim,retire,true,next);
   for(const v of vertices){
    const h=holder(w,v),expected=!h||fresh.has(h.block)?AIR:before.get(key(v))!;
    assert.equal(read(w,v),expected,`bits ${bits} round ${round} vertex ${v}`);
   }
  }
 }
});
