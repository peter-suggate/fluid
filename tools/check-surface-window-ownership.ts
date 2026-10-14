/** Host oracle for collectWindowsMain: exhaustive independent tile/window
 * incidence versus the block/local ownership rule, including outside pages. */
import assert from 'node:assert/strict';
import { surfaceWindowCapacity } from '../lib/core/webgpu-water-pipeline';
import { uniformPagePoolLayout, uniformPagePoolSeed, uniformPagePoolPublished, UNIFORM_PAGE_POOL_PUBLISHED as P } from '../lib/methods/uniform/uniform-page-pool';
type V = [number,number,number];
let randomState=0x19730921;
const random=()=>{randomState=(Math.imul(randomState,1664525)+1013904223)>>>0;return randomState/4294967296;};
const key=(p:readonly number[])=>p.join(',');
let cases=0,total=0;
try {
for(const extent of [2,3,5,9])for(let trial=0;trial<8;trial++){
 const pages:V[]=[];
 for(let i=0;i<10+trial*3;i++){
  const p:V=[Math.floor(random()*extent)+28,Math.floor(random()*extent),Math.floor(random()*extent)+28];
  if(!pages.some(q=>key(q)===key(p)))pages.push(p);
 }
 const resident=new Map(pages.map((p,i)=>[key(p),i+1]));
 const expected=new Set<string>();
 // Independent oracle: each resident tile touches these eight cube windows.
 for(const p of pages)for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++)for(let k=0;k<8;k++)
  expected.add(key([8*p[0]+x+(k&1),8*p[1]+y+((k>>1)&1),8*p[2]+z+(k>>2)]));
 const actual=new Map<string,number>();let duplicate=0;
 for(const [index,p] of pages.entries()){
  const block=index+1;
  for(let group=0;group<8;group++)for(let lane=0;lane<64;lane++){
   const own=[4*(group&1)+(lane&3),4*((group>>1)&1)+((lane>>2)&3),4*(group>>2)+(lane>>4)];
   for(let k=0;k<8;k++){
    const up=[k&1,(k>>1)&1,k>>2];if(up.some((v,a)=>v&&own[a]!==7))continue;
    const l=own.map((v,a)=>up[a]?8:v),high=l.map(v=>v===8?1:0),plane=l.map(v=>(v&7)===0);
    const mine=high[0]!|(high[1]!<<1)|(high[2]!<<2);let list=true;
    for(let m=0;m<mine;m++){
     const step=[m&1,(m>>1)&1,m>>2];if(step.some((v,a)=>v&&!plane[a]))continue;
     if(resident.has(key(p.map((v,a)=>v+high[a]!-step[a]!)))){list=false;break;}
    }
    if(!list)continue;
    const word=(block<<12)|l[0]!|(l[1]!<<4)|(l[2]!<<8);
    const decoded=[word&15,(word>>4)&15,(word>>8)&15];assert.equal(word>>>12,block);
    const g=key(p.map((v,a)=>8*v+decoded[a]!));if(actual.has(g))duplicate++;actual.set(g,word);
   }
  }
 }
 const missing=[...expected].filter(k=>!actual.has(k)).length,extra=[...actual.keys()].filter(k=>!expected.has(k)).length;
 assert.equal(missing+extra+duplicate,0);assert.ok(actual.size<=surfaceWindowCapacity(2048,2048,2048,pages.length+2));
 // Same blocks published with full signed coordinates, never keyed by seed box.
 const layout=uniformPagePoolLayout({blocks:pages.length+2,idleSteps:2,box:{origin:[30,0,30],pages:[2,1,2]}});
 const seed=uniformPagePoolSeed(layout,pages.map((p,i)=>({x:p[0],y:p[1],z:p[2],block:i+1})),()=>1);
 const published=uniformPagePoolPublished(seed),signed=new Int32Array(published.buffer);let drawn=0;
 for(let b=1;b<layout.blocks-1;b++){const r=P.records+4*b;if(published[r+3]===1){drawn++;assert.deepEqual([...signed.subarray(r,r+3)],pages[b-1]);}}
 assert.equal(drawn,pages.length);
 console.log(`PASS ownership extent=${extent} trial=${trial} pages=${pages.length} windows=${actual.size} missing=${missing} extra=${extra} duplicate=${duplicate} overlay=${drawn}`);cases++;total+=actual.size;
}
// Negative coordinates in the overlay record are preserved (not modulo keys).
{
 const l=uniformPagePoolLayout({blocks:4,idleSteps:2,box:{origin:[0,0,0],pages:[1,1,1]}});
 const p=uniformPagePoolPublished(uniformPagePoolSeed(l,[{x:-2,y:3,z:-4,block:1}],()=>1));
 assert.deepEqual([...new Int32Array(p.buffer).subarray(P.records+4,P.records+7)],[-2,3,-4]);
}
console.log(`PASS ownership ${cases} cases, ${total} windows; signed overlay record`);
} catch(error) { console.error(`FAIL ownership after ${cases} cases and ${total} windows`,error);process.exitCode=1; }
