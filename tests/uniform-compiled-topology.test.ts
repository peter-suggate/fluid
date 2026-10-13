import assert from "node:assert/strict";
import test from "node:test";
import { createUniformMixedLayoutFromWidths, type UniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
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
function layout(d:V,widths:Uint8Array):UniformMixedLayout {
 return createUniformMixedLayoutFromWidths({dimensions:d,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},widths,[]);
}
function owner(l:UniformMixedLayout,p:V){
 if(p.some((v,a)=>v<0||v>=l.lattice.dimensions[a]!))return undefined;
 const t=p.map(v=>v>>2),d=l.tileDimensions,tile=t[0]!+d[0]*(t[1]!+d[1]*t[2]!);
 const word=l.tiles[tile]!,width=(word>>>31)?1:4,n=4/width;
 const local=p.map(v=>Math.floor((v%4)/width));const lane=local[0]!+n*(local[1]!+n*local[2]!);
 return {tile,width,lane,index:(word&0x3fffffff)+lane,origin:p.map(v=>v-v%width) as V};
}
test("compiled face patches match both incident owners including sixteen-to-one seams",()=>{
 for(let bits=0;bits<256;bits++){
  const l=layout([8,8,8],Uint8Array.from({length:8},(_,k)=>bits&(1<<k)?1:4));
  for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
   const a=owner(l,[x,y,z])!;if(a.origin.some((v,i)=>v!==[x,y,z][i]))continue;
   for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){
    const probe=[...a.origin] as V;probe[axis]!+=sign>0?a.width:-1;
    const first=owner(l,probe),width=Math.min(a.width,first?.width??a.width),side=a.width/width;
    for(let part=0;part<side*side;part++){
     const u=(axis+1)%3,v=(axis+2)%3,q=[...probe] as V;q[u]!+=(part%side)*width;q[v]!+=Math.floor(part/side)*width;
     const b=owner(l,q);if(!b){assert.equal(part,0);continue;}
     const offset=part%side*[1,4,16][u]!+Math.floor(part/side)*[1,4,16][v]!;
     assert.equal(first!.index+offset,b.index);assert.equal(first!.lane+offset,b.lane);
     // The neighbor's reciprocal patch has the same normal plane and area.
     const plane=a.origin[axis]!+(sign>0?a.width:0);
     assert.equal(plane,b.origin[axis]!+(sign<0?b.width:0));
     assert.equal(Math.min(a.width,b.width),width);
    }
   }
  }
 }
});
