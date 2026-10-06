import assert from "node:assert/strict";
import test from "node:test";
import { createUniformMixedLayoutFromWidths, type UniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { uniformVertexIncidentMasks, uniformStencilOctant, uniformSeamSideOrders, compileUniformBlendMask, UNIFORM_COMPILED_TOPOLOGY as R } from "../lib/methods/uniform/uniform-compiled-topology";

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
const corner = (k:number):V => [k&1,(k>>1)&1,k>>2];
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
function reference(l:UniformMixedLayout,p:V){
 let best:ReturnType<typeof owner>;
 for(let k=0;k<8;k++){
  const c=corner(k),o=owner(l,p.map((v,a)=>v+c[a]!-1) as V);
  if(o&&(!best||o.width>best.width||o.width===best.width&&o.index<best.index))best=o;
 }
 return best!;
}
function compiled(l:UniformMixedLayout,p:V){
 const c=p.map((v,a)=>Math.min(v,l.lattice.dimensions[a]!-1)>>2),d=l.tileDimensions;
 const t=c[0]!+d[0]*(c[1]!+d[1]*c[2]!);const local=p.map((v,a)=>v-4*c[a]!);
 const required=Number(local[0]!==0)|(Number(local[1]!==0)<<1)|(Number(local[2]!==0)<<2);
 const mask=((l.stencils[2*t+1]!>>>R.incident)&255)&uniformVertexIncidentMasks[required]!;
 if(mask){const k=31-Math.clz32(mask&-mask),b=corner(k);return owner(l,c.map((v,a)=>(v+b[a]!-1)*4) as V)!;}
 return owner(l,p.map((v,a)=>Math.max(0,Math.min(v-1,l.lattice.dimensions[a]!-1))) as V)!;
}
function check(l:UniformMixedLayout){
 const d=l.lattice.dimensions;
 for(let z=0;z<=d[2];z++)for(let y=0;y<=d[1];y++)for(let x=0;x<=d[0];x++){
  const p:V=[x,y,z],a=reference(l,p),b=compiled(l,p);
  assert.equal(b.index,a.index,`authority at ${p}`);assert.equal(b.width,a.width);
 }
 for(let tile=0;tile<l.tiles.length;tile++){
  const t=l.tileDimensions,o:V=[tile%t[0],Math.floor(tile/t[0])%t[1],Math.floor(tile/(t[0]*t[1]))];
  const word=l.stencils[2*tile+1]!,own=owner(l,o.map(v=>4*v) as V)!;
  for(let k=0;k<8;k++){
   const c=corner(k),p=o.map((v,a)=>4*(v+c[a]!)) as V;
   const expected=own.width===4&&reference(l,p).tile===tile;
   assert.equal(Boolean((word>>>R.corners)&(1<<k)),expected,`corner ${tile}/${k}`);
   if(k){const q=o.map((v,a)=>4*(v+c[a]!)) as V,other=owner(l,q);
    assert.equal(Boolean((word>>>R.positive)&(1<<k)),other?.width===4,`positive octant ${tile}/${k}`);}
  }
  // Reconstruct the 4³ redistance footprint using eight packed octants.
  // Ignore out-of-domain cells, as window sampling clamps to the domain.
  for(let k=0;k<64;k++){
   const d:V=[k%4,Math.floor(k/4)%4,Math.floor(k/16)];
   const q=o.map((v,a)=>4*(v+d[a]!-1)) as V,expected=owner(l,q);if(!expected)continue;
   const c=d.map(v=>v>>1),center=o.map((v,a)=>v+c[a]!);
   const key=center[0]!+t[0]*(center[1]!+t[1]*center[2]!);
   const octant=c[0]!+2*(c[1]!+2*c[2]!);
   const mask=uniformStencilOctant(l.stencils[2*key]!,octant);
   const bit=(d[0]&1)+2*((d[1]&1)+2*(d[2]&1));
   assert.equal(Boolean(mask&(1<<bit)),expected.width===1,`window ${tile}/${k}`);
  }
 }
}

test("compiled vertex authority and writers match incident-owner traversal for every 2³ layout",()=>{
 for(let bits=0;bits<256;bits++)check(layout([8,8,8],Uint8Array.from({length:8},(_,k)=>bits&(1<<k)?1:4)));
});

test("compiled vertices handle thin domains and interior seams with arbitrary remote numbering",()=>{
 let seed=0x12345678;
 for(const d of [[4,8,12],[12,4,8],[8,12,4],[16,12,8]] as V[]){
  const n=d.reduce((s,v)=>s*v/4,1);
  for(let i=0;i<12;i++)check(layout(d,Uint8Array.from({length:n},()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed&0x80000000?1:4;})));
 }
});

test("remote refinement renumbers compact owners without changing a local compiled recipe",()=>{
 const widths=new Uint8Array(125).fill(4);widths[0]=1;
 const before=layout([20,20,20],widths);widths[0]=4;const after=layout([20,20,20],widths);
 assert.notEqual(before.tiles[124],after.tiles[124]);
 assert.equal(before.stencils[249],after.stencils[249]);
 check(before);check(after);
});

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
