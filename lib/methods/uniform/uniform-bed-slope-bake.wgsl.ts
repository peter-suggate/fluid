/** Kernels of the bed-slope bake (uniform-bed-slope-bake.ts has the rule and
 * the launches). Every kernel is one invocation a column of the lattice,
 * workgroups of 8x8 columns over a window of columns given by the params
 * block; a column's floor cells are its slots, lowest first.
 *
 * Scratch, by slot s = layers*(x + nx*z) + k:
 *   meta   one word: bits 0..10 the floor cell's y, bit 11 a drop beside it,
 *          then five bits a direction (-x, +x, -z, +z): the neighbour's slot
 *          in its column (3 bits) and the step in y plus one (2 bits; 3 for
 *          no neighbour). All ones for a slot with no floor cell.
 *   a, b   four words: the nearest nosing of the levels y-1, y, y+1 and y+2,
 *          each X | Z<<12 in half cells (a cell's centre is (2x+1, 2z+1)),
 *          all ones in 24 bits for none; bit 31 of the first word is the
 *          drop flag of the cell's tread. The spread reads one and writes the
 *          other. Before the links b's first word is the slot's y; after the
 *          spread b's first two words are the bed heights, which the
 *          relaxation takes from one to the other.
 */
import { solidOccupancyMaskWGSL } from "../../core/solid-occupancy-mask";
import type { UniformBedSlopeBakeLayout } from "./uniform-bed-slope-bake";

export type UniformBedSlopeBakeStage="build"|"spread"|"resolve";

const BINDINGS:Readonly<Record<UniformBedSlopeBakeStage,string>>={
 build:`@group(0) @binding(1) var<storage,read> bsMask:array<u32>;
@group(0) @binding(2) var<storage,read_write> bsMeta:array<u32>;
@group(0) @binding(3) var<storage,read_write> bsA:array<u32>;
@group(0) @binding(4) var<storage,read_write> bsB:array<u32>;`,
 spread:`@group(0) @binding(1) var<storage,read> bsMeta:array<u32>;
@group(0) @binding(2) var<storage,read> bsFrom:array<u32>;
@group(0) @binding(3) var<storage,read_write> bsTo:array<u32>;`,
 // The mask's buffer again, for its lane: two cells share a word and belong
 // to different invocations, so a cell's sixteen bits are cleared and set
 // with atomics.
 resolve:`@group(0) @binding(1) var<storage,read_write> bsLane:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> bsMeta:array<u32>;
@group(0) @binding(3) var<storage,read> bsB:array<u32>;`,
};

const COMMON=(l:UniformBedSlopeBakeLayout)=>/* wgsl */`
// Columns: x0, z0 and the count along x and along z.
struct BsParams{window:vec4u,bake:vec4u}
@group(0) @binding(0) var<uniform> bsParams:BsParams;
const BS_SIZE=vec3u(${l.size[0]}u,${l.size[1]}u,${l.size[2]}u);
const BS_SHAPE=vec3u(${l.size[0]+2}u,${l.size[1]+2}u,${l.size[2]+2}u);
const BS_LAYERS:u32=${l.layers}u;
const BS_LANE:u32=${l.laneBaseWord}u;
const BS_NONE:u32=0xffffffffu;
const BS_POINT:u32=0xffffffu;
const BS_STEP=array<vec2i,4>(vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
fn bsCell(p:vec3i)->u32{let q=vec3u(p+vec3i(1));return q.x+BS_SHAPE.x*(q.y+BS_SHAPE.y*q.z);}
fn bsColumn(x:u32,z:u32)->u32{return x+BS_SIZE.x*z;}
fn bsLink(info:u32,d:u32)->u32{return (info>>(12u+5u*d))&31u;}
/** The slot a link names, in the column one step along d. */
fn bsNeighbour(x:u32,z:u32,d:u32,link:u32)->u32{
 let q=vec2i(i32(x),i32(z))+BS_STEP[d];
 return BS_LAYERS*bsColumn(u32(q.x),u32(q.y))+(link&7u);
}
/** A nosing nearer the cell centre c than the one held; a tie goes to the smaller point, whichever way it arrived. */
fn bsBetter(cand:u32,held:u32,c:vec2i)->bool{
 if(cand==BS_POINT){return false;}
 if(held==BS_POINT){return true;}
 let a=vec2i(i32(cand&4095u),i32(cand>>12u))-c;let b=vec2i(i32(held&4095u),i32(held>>12u))-c;
 let da=dot(a,a);let db=dot(b,b);
 return da<db||(da==db&&cand<held);
}
fn bsRoot(v:u32)->u32{
 var r=u32(sqrt(f32(v)));
 while(r*r>v){r-=1u;}
 while((r+1u)*(r+1u)<=v){r+=1u;}
 return r;
}`;

const BUILD=(l:UniformBedSlopeBakeLayout)=>/* wgsl */`
// The solid mask's image (SolidOccupancyMask) by page: a cell of a page the
// lattice walls off is solid, so the lattice's walls and floor are.
${solidOccupancyMaskWGSL("bsMask","bsWindowPage","bsWindowSolid")}
// A cell of the lattice, which lies at BS_ORIGIN in the image's window.
const BS_ORIGIN=vec3i(${l.origin[0]},${l.origin[1]},${l.origin[2]});
fn bsSolid(p:vec3i)->bool{return bsWindowSolid(p+BS_ORIGIN);}
/** A column's floor cells: open over solid, the lowest BS_LAYERS of them. */
@compute @workgroup_size(8,8,1) fn bsLayers(@builtin(global_invocation_id) id:vec3u){
 if(any(id.xy>=bsParams.window.zw)){return;}
 let x=bsParams.window.x+id.x;let z=bsParams.window.y+id.y;let base=BS_LAYERS*bsColumn(x,z);
 var k=0u;var below=bsSolid(vec3i(i32(x),-1,i32(z)));
 for(var y=0u;y<BS_SIZE.y;y++){
  let here=bsSolid(vec3i(i32(x),i32(y),i32(z)));
  if(!here&&below&&k<BS_LAYERS){bsB[4u*(base+k)]=y;k++;}
  below=here;
 }
 for(;k<BS_LAYERS;k++){bsB[4u*(base+k)]=BS_NONE;}
}
/** Links to the floor cells a film reaches in one step, and the nosings beside the cell. */
@compute @workgroup_size(8,8,1) fn bsLinks(@builtin(global_invocation_id) id:vec3u){
 if(any(id.xy>=bsParams.window.zw)){return;}
 let x=bsParams.window.x+id.x;let z=bsParams.window.y+id.y;let base=BS_LAYERS*bsColumn(x,z);
 let c=vec2i(i32(2u*x+1u),i32(2u*z+1u));
 let low=vec2i(bsParams.window.xy);let high=low+vec2i(bsParams.window.zw);
 for(var k=0u;k<BS_LAYERS;k++){
  let s=base+k;let y=bsB[4u*s];
  if(y==BS_NONE){bsMeta[s]=BS_NONE;continue;}
  var word=y;var drop=0u;var own=BS_POINT;var up=BS_POINT;
  let openAbove=!bsSolid(vec3i(i32(x),i32(y)+1,i32(z)));
  for(var d=0u;d<4u;d++){
   let q=vec2i(i32(x),i32(z))+BS_STEP[d];var link=24u;
   let openBeside=!bsSolid(vec3i(q.x,i32(y),q.y));
   // A column outside the window is no neighbour: the window is grown past
   // every column whose state can reach a cell the bake writes.
   if(all(q>=low)&&all(q<high)){
    let other=BS_LAYERS*bsColumn(u32(q.x),u32(q.y));
    for(var j=0u;j<BS_LAYERS;j++){
     let y1=bsB[4u*(other+j)];
     if(y1==BS_NONE){break;}
     // Level, one up over a riser this cell's own air clears, or one down over the neighbour's.
     if(y1==y||(y1==y+1u&&openAbove)||(y1+1u==y&&openBeside)){link=j|((y1+1u-y)<<3u);break;}
    }
   }
   let rise=link>>3u;let point=u32(c.x+BS_STEP[d].x)|(u32(c.y+BS_STEP[d].y)<<12u);
   if(rise==0u&&bsBetter(point,own,c)){own=point;}
   if(rise==2u&&bsBetter(point,up,c)){up=point;}
   // Open beside and below that: a fall of a cell or more that is not this tread's own riser.
   if(openBeside&&rise!=0u&&!bsSolid(vec3i(q.x,i32(y)-1,q.y))){drop=1u;}
   word|=link<<(12u+5u*d);
  }
  bsMeta[s]=word|(drop<<11u);
  bsA[4u*s]=BS_POINT|(drop<<31u);bsA[4u*s+1u]=own;bsA[4u*s+2u]=up;bsA[4u*s+3u]=BS_POINT;
 }
}
/** A nosing's distance from the centre c, in 1/128 cell, rounded down. */
fn bsReach(p:u32,c:vec2i)->i32{
 let a=vec2i(i32(p&4095u),i32(p>>12u))-c;
 return i32(bsRoot(u32(dot(a,a))*4096u));
}
/** The bed height of every floor cell from its nosings, in 1/4096 cell above the lattice's floor less one. */
@compute @workgroup_size(8,8,1) fn bsHeight(@builtin(global_invocation_id) id:vec3u){
 if(any(id.xy>=bsParams.window.zw)){return;}
 let x=bsParams.window.x+id.x;let z=bsParams.window.y+id.y;let base=BS_LAYERS*bsColumn(x,z);
 let c=vec2i(i32(2u*x+1u),i32(2u*z+1u));
 for(var k=0u;k<BS_LAYERS;k++){
  let s=base+k;let info=bsMeta[s];
  if(info==BS_NONE){break;}
  let first=bsA[4u*s];let below=first&BS_POINT;let own=bsA[4u*s+1u]&BS_POINT;let up=bsA[4u*s+2u]&BS_POINT;let above=bsA[4u*s+3u]&BS_POINT;
  var height=BS_NONE;let level=i32(((info&2047u)+1u)*4096u);
  if(own!=BS_POINT&&up!=BS_POINT){
   // Between its own nosing and the riser above: the tread's place on the chord from one to the other.
   let d=bsReach(own,c);let u=bsReach(up,c);
   if(u+d>=${l.minRun}){height=u32(level+(d*8192+u+d)/(2*(u+d)));}
  }else if(own!=BS_POINT&&below!=BS_POINT){
   // The top landing: the run of the tread below, carried up from its own nosing.
   let d=bsReach(own,c);let run=bsReach(below,c)-d;
   if(run>=${l.minRun}&&4*d<=${4+l.band4}*run){height=u32(level+(d*8192+run)/(2*run));}
  }else if(up!=BS_POINT&&above!=BS_POINT&&(first>>31u)!=0u){
   // The last tread before a drop: the run of the tread above, carried down from the riser.
   let u=bsReach(up,c);let run=bsReach(above,c)-u;
   if(run>=${l.minRun}&&4*u<=${4+l.band4}*run){height=u32(level+4096-(u*8192+run)/(2*run));}
  }
  bsB[4u*s]=height;
 }
}
fn bsRelax(id:vec3u,src:u32,dst:u32){
 if(any(id.xy>=bsParams.window.zw)){return;}
 let x=bsParams.window.x+id.x;let z=bsParams.window.y+id.y;let base=BS_LAYERS*bsColumn(x,z);
 for(var k=0u;k<BS_LAYERS;k++){
  let s=base+k;let info=bsMeta[s];
  if(info==BS_NONE){break;}
  let h=bsB[4u*s+src];var next=h;
  if(h!=BS_NONE){
   var sum=0u;var axes=0u;
   for(var axis=0u;axis<2u;axis++){
    let a=bsLink(info,2u*axis);let b=bsLink(info,2u*axis+1u);
    if(a>=24u||b>=24u){continue;}
    let low=bsB[4u*bsNeighbour(x,z,2u*axis,a)+src];let high=bsB[4u*bsNeighbour(x,z,2u*axis+1u,b)+src];
    // An axis with a height on one side only is left out: a height that is linear across its neighbours never moves.
    if(low!=BS_NONE&&high!=BS_NONE){sum+=low+high;axes++;}
   }
   if(axes>0u){next=(2u*axes*h+sum+2u*axes)/(4u*axes);}
  }
  bsB[4u*s+dst]=next;
 }
}
@compute @workgroup_size(8,8,1) fn bsRelaxOut(@builtin(global_invocation_id) id:vec3u){bsRelax(id,0u,1u);}
@compute @workgroup_size(8,8,1) fn bsRelaxBack(@builtin(global_invocation_id) id:vec3u){bsRelax(id,1u,0u);}`;

const SPREAD=/* wgsl */`
/** One step of the nearest nosings along the links: the neighbour's nosing of a level is this cell's of the same level. */
@compute @workgroup_size(8,8,1) fn bsSpread(@builtin(global_invocation_id) id:vec3u){
 if(any(id.xy>=bsParams.window.zw)){return;}
 let x=bsParams.window.x+id.x;let z=bsParams.window.y+id.y;let base=BS_LAYERS*bsColumn(x,z);
 let c=vec2i(i32(2u*x+1u),i32(2u*z+1u));
 for(var k=0u;k<BS_LAYERS;k++){
  let s=base+k;let info=bsMeta[s];
  if(info==BS_NONE){break;}
  let first=bsFrom[4u*s];var drop=first>>31u;
  var f=array<u32,4>(first&BS_POINT,bsFrom[4u*s+1u],bsFrom[4u*s+2u],bsFrom[4u*s+3u]);
  for(var d=0u;d<4u;d++){
   let link=bsLink(info,d);let rise=link>>3u;
   if(rise==3u){continue;}
   let n=4u*bsNeighbour(x,z,d,link);
   for(var i=0u;i<4u;i++){
    // Level i here is level i+1-rise there; out of the four (the subtraction wraps), not held.
    let j=i+1u-rise;
    if(j>3u){continue;}
    let cand=bsFrom[n+j]&BS_POINT;
    if(bsBetter(cand,f[i],c)){f[i]=cand;}
   }
   // The drop flag belongs to a tread: it crosses level links alone.
   if(rise==1u){drop|=bsFrom[n]>>31u;}
  }
  bsTo[4u*s]=f[0]|(drop<<31u);bsTo[4u*s+1u]=f[1];bsTo[4u*s+2u]=f[2];bsTo[4u*s+3u]=f[3];
 }
}`;

const RESOLVE=(l:UniformBedSlopeBakeLayout)=>/* wgsl */`
const BS_HEIGHT:u32=${l.relax&1}u;
fn bsSide(info:u32,x:u32,z:u32,d:u32)->u32{
 let link=bsLink(info,d);
 if(link>=24u){return BS_NONE;}
 return bsB[4u*bsNeighbour(x,z,d,link)+BS_HEIGHT];
}
/** The lane of a column's cells: zero, but for its floor cells with a bed height, the height's gradient. */
@compute @workgroup_size(8,8,1) fn bsResolve(@builtin(global_invocation_id) id:vec3u){
 if(any(id.xy>=bsParams.bake.zw)){return;}
 let x=bsParams.bake.x+id.x;let z=bsParams.bake.y+id.y;let base=BS_LAYERS*bsColumn(x,z);
 for(var y=0u;y<BS_SIZE.y;y++){
  let i=bsCell(vec3i(i32(x),i32(y),i32(z)));
  atomicAnd(&bsLane[BS_LANE+(i>>1u)],~(0xffffu<<(16u*(i&1u))));
 }
 for(var k=0u;k<BS_LAYERS;k++){
  let s=base+k;let info=bsMeta[s];
  if(info==BS_NONE){break;}
  let h=bsB[4u*s+BS_HEIGHT];
  if(h==BS_NONE){continue;}
  // In 1/8192 cell a cell: the centred difference, or the one side that has a height.
  var g=vec2i(0);
  for(var axis=0u;axis<2u;axis++){
   let low=bsSide(info,x,z,2u*axis);let high=bsSide(info,x,z,2u*axis+1u);
   if(low!=BS_NONE&&high!=BS_NONE){g[axis]=i32(high)-i32(low);}
   else if(high!=BS_NONE){g[axis]=2*(i32(high)-i32(h));}
   else if(low!=BS_NONE){g[axis]=2*(i32(h)-i32(low));}
  }
  let m2=u32(dot(g,g));
  if(m2<${l.minSlope2}u||m2>${l.maxSlope2}u){continue;}
  if(m2>${4096*4096}u){
   // Past half a cell a cell by rounding alone: brought back to it.
   var r=bsRoot(m2);
   if(r*r<m2){r+=1u;}
   g=g*4096/i32(r);
  }
  let size=min((254*abs(g)+4096)/8192,vec2i(127));let lane=select(size,-size,g<vec2i(0));
  let entry=(u32(lane.x)&255u)|((u32(lane.y)&255u)<<8u);
  let i=bsCell(vec3i(i32(x),i32(info&2047u),i32(z)));
  if(entry!=0u){atomicOr(&bsLane[BS_LANE+(i>>1u)],entry<<(16u*(i&1u)));}
 }
}`;

/** The module of one stage: its bindings, the shared helpers and its kernels. */
export function uniformBedSlopeBakeWGSL(layout:UniformBedSlopeBakeLayout,stage:UniformBedSlopeBakeStage):string{
 return `${BINDINGS[stage]}${COMMON(layout)}${stage==="build"?BUILD(layout):stage==="spread"?SPREAD:RESOLVE(layout)}`;
}
