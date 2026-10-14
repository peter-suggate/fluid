import assert from "node:assert/strict";
import test from "node:test";
import { uniformSeamSideOrders, compileUniformBlendMask } from "../lib/methods/uniform/uniform-compiled-topology";

type V = [number, number, number];
test("minimal blending masks preserve box distance throughout the center tile",()=>{
 const weight=(mask:number,p:V)=>{
  let fine=0;
  for(let k=0;k<27;k++)if(mask&(1<<k)){
   const o=[k%3-1,Math.floor(k/3)%3-1,Math.floor(k/9)-1].map(v=>4*v);
   const distance=Math.max(...p.map((q,a)=>Math.max(0,o[a]!-q,q-o[a]!-4)));
   fine=Math.max(fine,Math.max(0,1-distance/2));
  }
  return fine;
 };
 let state=9137;const masks=[0,0x07ffffff,...Array.from({length:27},(_,i)=>1<<i)];
 for(let i=0;i<256;i++){state=(Math.imul(state,1664525)+1013904223)>>>0;masks.push(state&0x07ffffff);}
 for(const original of masks){const reduced=compileUniformBlendMask(original);assert.equal(reduced&~original,0);
  for(const x of [0,.125,1,2,3,3.875,4])for(const y of [0,.25,2,3.75,4])for(const z of [0,.25,2,3.75,4]){
   const p:V=[x,y,z];assert.equal(weight(reduced,p),weight(original,p),`mask ${original} at ${p}`);
  }
 }
});
test("compiled seam schedules preserve side and patch ordering for all 64 side configurations",()=>{
 for(let sides=0;sides<64;sides++){
  const sequence:{side:number;part:number}[]=[];
  for(const fine of [true,false])for(let side=0;side<6;side++)if(Boolean(sides&(1<<side))===fine){
   for(let part=0;part<(fine?16:1);part++)sequence.push({side,part});
  }
  const fine=[0,1,2,3,4,5].filter(side=>sides&(1<<side)).length;
  for(let q=0;q<192;q++){
   const onFine=q<16*fine,ordinal=onFine?Math.floor(q/16):q-15*fine;
   const actual=ordinal>=6?undefined:{side:(uniformSeamSideOrders[sides]!>>>(3*ordinal))&7,part:onFine?q%16:0};
   assert.deepEqual(actual,sequence[q],`side mask ${sides}, lane ${q}`);
  }
 }
});
