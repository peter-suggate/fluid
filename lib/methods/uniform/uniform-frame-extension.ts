/** B4's velocity extension (docs/plans/brick-resident-step-2026-11-08.md
 * §3.3): two tile-window passes on the frame brick store, 4h then h.
 *
 * The work follows the band, not the scene. ufxF flags each held page that
 * has a tile with the support bit (the surface band plus one tile) in a
 * region the COLD freeze does not hold; a page is active where it or one of
 * its 26 neighbours is flagged. Elsewhere every face is a source or blocked,
 * which no sweep changes, and the canonical texel and its subfaces already
 * hold that answer: those pages and the bricks with no supported tile
 * publish their own values and stage nothing.
 * 1. ufxR: a thread a 4h face of an active page, its seed (value, distance)
 *    to the coarse buffer's seed half.
 *    - Its value is its canonical texel on a 4h tile, or the mean of its 16
 *      h subfaces on an h tile: the reflux restriction.
 *    - The face is a source where any subface is one. A subface is a source
 *      where the phase of the owner on either side is above 0.5. The two
 *      owners are the face's two tiles, looked up once a face.
 * 2. ufx4: one workgroup a held page, a lane a 4h cell.
 *    - A flagged page stages its window (the page and a halo of two, 12³
 *      cells, an axis at a time) with one load a face from the seed half:
 *      its neighbours are active, so ufxR wrote theirs too. An active page
 *      that is not flagged takes its seeds as its state.
 *    - Two Jacobi sweeps at 4h spacing (ufxBSweep0 on the window inset by
 *      one, then the page's cells): exactly two global sweeps, whatever the
 *      neighbours hold. The (value, distance) of every face goes to the
 *      coarse buffer.
 *    - 4h tiles then publish their +faces: the three anchor texels (the
 *      canonical faces) by the cell's lane, and, on a plane against h tiles,
 *      each subface's, the page's seam cells shared out over the workgroup.
 *      A source subface keeps its own value, and so does a face no sweep
 *      reached.
 * 3. ufxRH: a thread four cells (one tile row) of a live brick of an active
 *    page: their h face seeds, 2 bits each (unreached, source, blocked, a
 *    closed wall), packed per slot (the result buffer), the tile looked up
 *    once a row.
 * 4. ufxH: one workgroup a live brick, a lane an h cell.
 *    - A brick with a supported h tile stages the 4h state of its 4h cells
 *      and their low neighbours (5³ an axis) once, then its window an axis
 *      at a time (12³ cells): a core cell is one texel load (the
 *      neighbour's h-store texel, resolved once at record time, the texel
 *      and seed word held in registers across the axes); a seam (a 4h tile
 *      inside a brick, a brick that is not live) prolongs the staged 4h
 *      state linearly along the axis, value and distance alike. Two sweeps
 *      at h spacing, as at 4h. A tile without support holds sources and
 *      blocked faces, which the sweep leaves.
 *    - Its h cells publish their three +faces in one store. A face no sweep
 *      reached takes the 4h band's where that reached it, else keeps its
 *      own; a closed wall keeps its own. A brick without a supported tile
 *      publishes its own.
 *
 * Why one workgroup a block, an axis at a time (B4 round 4, counters on
 * letters): the first brick pass's four 12³ windows (28 KiB) left two
 * 128-lane workgroups a core, 8 % occupancy, and each brick then ran three
 * components of staging, two sweeps and a write in series (18 barriers),
 * every texel load a page-table walk: integer-latency bound (ALU 21 %, 1 %
 * F32, 3 GB/s). Tile windows (8 a brick, all three axes) doubled occupancy
 * but staged 2.3x the cells and repeated the record and its dependent
 * loads 8 times a brick. One window an axis is 14 KiB for 512 lanes: the
 * record, the flags and the 4h state load once a brick, the halo is
 * 1216 cells a brick against 3584, and a seam reads the staged 4h state.
 *
 * Held regions (UniformColdFreeze, regions of 16h: 8 a page, one a brick).
 * A frozen region's faces stay put in the velocity and the forced field
 * (velocityScratch): every stage leaves them, so the extension publishes
 * no texel a frozen cell owns, and sweeps no brick of one. Its pages flag
 * nothing; a frozen page beside a flagged one restricts (its neighbours'
 * windows stage its seeds) and takes those as its state. The work is then
 * the band's and the warm volume's: the warm volume's velocity changes
 * every step and the forced field overwrites velocityScratch there, so its
 * own values are republished (an rgba32float face field has no read-write
 * storage binding, so no in-place publication is possible).
 *
 * The old field (UniformFrameExtensionFields.old, the departure's
 * extension of P2G's field for the tail's FLIP delta). Only the supported
 * band's faces are kept: swept pages and bricks publish into the departure
 * field (free from P2G's read until the next momentum) the values the
 * ordinary extension would. Nothing else is written. The tail's extension
 * (the same seeds and reach: phase and topology are unchanged across
 * pressure) then publishes the current field, and
 * UniformNarrowBandFlip.recordDelta forms current minus old at each
 * sample's supported faces. No sentinel marks a face no sweep reached: the
 * transfer reads the departure field's faces next step where momentum
 * wrote none, so every texel published there is a velocity (both sides
 * keep such a face's own value, as the old field's snapshot did).
 *
 * Domain, as the old slot extension had it:
 * - A face whose across cell is SOLID (or bulk) is a closed wall: never swept
 *   and never a neighbour (blocked), and published as its own value.
 * - A face of a tile without the extension support bit (umTileSupport & 2)
 *   is blocked unless it is a source.
 * - Low faces (the -face against a SOLID page below) are the negative
 *   buffer's: published as they are (a copy), outside every pass.
 * A face no sweep reaches keeps its value: the far field holds the last
 * frame's state, so no pyramid runs (§5 risk 1, the lagged fixed point).
 *
 * Cited: Zhao 2005 (fast sweeping); Adalsteinsson & Sethian 1999 (extension
 * velocity); Bridson 2015 §4.4 (extrapolation band). */
import { UniformColdFreeze } from "./uniform-cold-freeze";
import { uniformDetailBindLayout, uniformDetailClass, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailShader, type UniformDetailGroup } from "./uniform-detail-fields";
import type { UniformFrameBricks } from "./uniform-frame-bricks";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_LIVE_GROUP_GROUPS } from "./uniform-live-lists";
import { uniformStageListsWGSL } from "./uniform-stage-lists";

export interface UniformFrameExtensionFields{
 /** The face velocity to extend (rgba32float face field). */
 physical:GPUTexture;
 /** Phase (r32float cell field): above 0.5 is liquid. */
 phase:GPUTexture;
 /** The extended velocity (rgba32float face field, disjoint from physical). */
 output:GPUTexture;
 /** The low-face buffers, copied (not for an old field). */
 negative:GPUBuffer;outputNegative:GPUBuffer;
 /** 16 bytes: the cell size in metres (xyz). */
 params:GPUBuffer;
 /** The old field for the tail's FLIP delta: only the supported band's
  * faces of warm regions are written. */
 old?:boolean;
}
export interface UniformFrameExtensionGroup{readonly detail:UniformDetailGroup;readonly negative:GPUBuffer;readonly outputNegative:GPUBuffer;readonly old:boolean}

const PASS:GPUComputePassDescriptor={label:"Uniform frame extension"};
/** A seed distance: unreached, and (negative) outside the domain. */
const UNREACHED=1e20,BLOCKED=-1;
/** A live slot's header (ufxRH, for ufxH): its mode, its origin (3), and its
 * 27-neighbour record: words, h tile masks, packed h-store texels. */
const UFX_HEAD=88;
const U=UNREACHED.toExponential(),HALF=(0.5*UNREACHED).toExponential();

/** A window's sweep helpers, cells c from -2 an axis, width w: name At(c)
 * its index, name Upwind(i), and name SweepAt(c, base, h), the Jacobi
 * update of the window's cell c at window offset base. Scalar throughout:
 * the six neighbours are fixed strides, so no array is indexed. */
function sweep(name:string,V:string,D:string,w:number):string{
 const at=w===8?"q.x|(q.y<<3u)|(q.z<<6u)":`q.x+${w}u*(q.y+${w}u*q.z)`;
 return /* wgsl */`fn ${name}At(c:vec3i)->u32{let q=vec3u(c+vec3i(2));return ${at};}
// A blocked face is never an upwind neighbour.
fn ${name}Upwind(i:u32)->vec2f{let d=${D}[i];return select(vec2f(${V}[i],d),vec2f(0.0,${U}),d<0.0);}
fn ${name}SweepAt(c:vec3i,base:u32,h:vec3f)->vec2f{
 let i=base+${name}At(c);let old=vec2f(${V}[i],${D}[i]);if(old.y<=0.0){return old;}
 return ufxCombine(old,${name}Upwind(i-1u),${name}Upwind(i+1u),${name}Upwind(i-${w}u),${name}Upwind(i+${w}u),${name}Upwind(i-${w*w}u),${name}Upwind(i+${w*w}u),h);
}
`;
}

export class UniformFrameExtension{
 private layoutOf?:GPUBindGroupLayout;
 private freeze?:UniformColdFreeze;
 /** (value, distance) of every 4h face of every block, 512 cells × 3 axes a
  * block: the swept state, then (from UFX_SEEDS) ufxR's seeds, then a
  * page's flag and its active bit (UFX_FLAGS, UFX_ACTIVE: .x 1 or 0). */
 readonly coarse:GPUBuffer;
 private readonly blocks:number;
 /** ufxRH's packed h seeds, 128 words a slot, then each slot's header
  * (UFX_HEAD words). The h results never leave threadgroup memory. */
 private readonly result:GPUBuffer;
 private pipelines?:{flag:GPUComputePipeline;restrict:GPUComputePipeline;coarse:readonly [GPUComputePipeline,GPUComputePipeline];seed:GPUComputePipeline;fine:readonly [GPUComputePipeline,GPUComputePipeline]};
 private readonly pageGroups:number;private readonly brickGroups:number;
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly bricks:UniformFrameBricks){
  const blocks=ownership.capacity.address.blocks;
  // A workgroup-per-item visit's grid (UNIFORM_LIVE_GROUP_GROUPS), not the
  // capacity: each launch strides its live count, and a 512-lane group past
  // it still costs its launch (the letters skeleton: 0.42 ms of 12288).
  this.blocks=blocks;this.pageGroups=Math.min(blocks,UNIFORM_LIVE_GROUP_GROUPS);this.brickGroups=Math.min(bricks.capacity,UNIFORM_LIVE_GROUP_GROUPS);
  this.coarse=device.createBuffer({label:"Uniform frame extension 4h faces",size:8*(2*3*512+2)*blocks,usage:GPUBufferUsage.STORAGE});
  this.result=device.createBuffer({label:"Uniform frame extension h seeds",size:4*(128+UFX_HEAD)*bricks.capacity,usage:GPUBufferUsage.STORAGE});
 }
 /** The COLD freeze, before bind and initialize: its frozen regions are held. */
 attachFreeze(freeze:UniformColdFreeze|undefined):void{
  if(this.layoutOf||this.freeze)throw new Error("Frame extension: the freeze attaches once, before bind");
  this.freeze=freeze;
 }
 private get layout():GPUBindGroupLayout{
  const storage=(binding:number):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}});
  const sampled=(binding:number,viewDimension:GPUTextureViewDimension="3d"):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension}});
  // Binding 8 is the velocity again, read by h-store texel (udTexel_): no twin.
  return this.layoutOf??=uniformDetailBindLayout(this.device,{label:"Uniform frame extension",entries:[
   sampled(0),sampled(1),
   {binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
   storage(3),storage(4),storage(5),
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...(this.freeze?[sampled(7,"2d")]:[]),
   sampled(8),
  ]},[8]);
 }
 get allocatedBytes():number{return this.coarse.size+this.result.size;}
 /** The module's WGSL (for its regression too). */
 code():string{
  const o=this.ownership;
  return uniformMixedTopologyWGSL(o.capacity,0)+uniformStageListsWGSL+/* wgsl */`
@group(1) @binding(0) var physical:texture_3d<f32>;
@group(1) @binding(1) var phase:texture_3d<f32>;
@group(1) @binding(2) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(3) var<storage,read_write> ufbTable:array<atomic<u32>>;
@group(1) @binding(4) var<storage,read_write> ufxCoarse:array<vec2f>;
@group(1) @binding(5) var<storage,read_write> ufxResult:array<u32>;
@group(1) @binding(6) var<uniform> ufxParams:vec4f;
@group(1) @binding(8) var physicalTexels:texture_3d<f32>;
`+UniformColdFreeze.wgsl(this.freeze,1,7)+this.bricks.tableWGSL("ufbTable")+/* wgsl */`
// The old field (UniformFrameExtensionFields.old): the supported band alone.
override ufxOld:bool=false;
const UFX_SEEDS:u32=${3*512*this.blocks}u;
const UFX_FLAGS:u32=${2*3*512*this.blocks}u;
const UFX_ACTIVE:u32=${(2*3*512+1)*this.blocks}u;
const UFX_HEADER:u32=${128*this.bricks.capacity}u;
fn ufxHeader(slot:u32)->u32{return UFX_HEADER+${UFX_HEAD}u*slot;}
fn ufxFlagged(block:u32)->bool{return ufxCoarse[UFX_FLAGS+block].x>0.5;}
fn ufxActive(block:u32)->bool{return ufxCoarse[UFX_ACTIVE+block].x>0.5;}
fn ufxReached(state:vec2f)->bool{return state.y>=0.0&&state.y<${HALF};}
fn ufxAxis(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
// Cell p's tile key, or UM_TILES where no live tile holds it.
fn ufxTile(p:vec3i)->u32{if(any(p<vec3i(0))||!umCellPresent(p)){return UM_TILES;}let t=umTileAt(vec3u(p)>>vec3u(2u));return select(UM_TILES,t,umTileLive(t));}
fn ufxFine(t:u32)->bool{return t<UM_TILES&&umTileWidth(t)==1u;}
// A tile key known to lie in a stored block: live and h.
fn ufxFineTile(t:u32)->bool{return umTileLive(t)&&umTileWidth(t)==1u;}
// The phase of cell p's owner (its own on an h tile, its tile's origin on a 4h one); -1 where nothing is stored.
// t is ufxTile(p): callers that know it pass it.
fn ufxPhase(p:vec3i,t:u32)->f32{if(t>=UM_TILES){return -1.0;}return textureLoad(phase,select((p>>vec3u(2u))<<vec3u(2u),p,umTileWidth(t)==1u),0).x;}
// Face (p, a) between cells on tiles t (p's) and n (p + e_a's).
fn ufxSource(p:vec3i,a:u32,t:u32,n:u32)->bool{return ufxPhase(p,t)>0.5||ufxPhase(p+ufxAxis(a),n)>0.5;}
fn ufxClosed(p:vec3i,a:u32)->bool{return umAcrossKind(p,a,1)>=UM_SOLID;}
// The packed seed of h face (p, a) on live tile t: 1 a source, 3 a closed
// wall, 2 blocked (a tile without support), 0 unreached.
fn ufxSeedCodeOf(p:vec3i,a:u32,t:u32)->u32{
 if(ufxClosed(p,a)){return 3u;}
 if(ufxSource(p,a,t,ufxTile(p+ufxAxis(a)))){return 1u;}
 return select(2u,0u,(umTileSupport(t)&2u)!=0u);
}
// A packed seed's distance: 0 a source, blocked a wall or a tile without support, else unreached.
fn ufxSeedCode(c:u32)->f32{return select(select(${U},${BLOCKED}.0,c>=2u),0.0,c==1u);}
// ---- 4h: face (4h cell C, axis a).
fn ufxSub(C:vec3i,a:u32,i:u32)->vec3i{var q=4*C;q[a]+=3;q[(a+1u)%3u]+=i32(i&3u);q[(a+2u)%3u]+=i32(i>>2u);return q;}
// The seed of 4h face (C, a), as (value, distance). A face between 4h
// tiles: its canonical texel. Otherwise (an h tile, or a 4h tile's face
// against one, whose 16 subface texels it stores) the mean of the source
// subfaces where there is one: the liquid faces' restriction (an air
// subface holds no fluid velocity, and averaging it in would break the
// extension's maximum principle), else of all 16 (a value only a sweep can
// replace, never prolonged unreached). Every subface lies on C's tile and
// its +e_a cell on C + e_a's, so the two tiles are looked up once.
fn ufxRestrict4(C:vec3i,a:u32)->vec2f{
 let t=ufxTile(4*C);if(t>=UM_TILES){return vec2f(0.0,${BLOCKED}.0);}
 let e=ufxAxis(a);let n=ufxTile(4*(C+e));let p=4*C+3*e;
 var value=0.0;var source=false;
 if(umTileWidth(t)!=1u&&!ufxFine(n)){value=textureLoad(physical,p,0)[a];source=ufxSource(p,a,t,n);}
 else{
  var all=0.0;var liquid=0.0;var count=0.0;
  for(var i=0u;i<16u;i++){let q=ufxSub(C,a,i);let v=textureLoad(physical,q,0)[a];all+=v;if(ufxSource(q,a,t,n)){liquid+=v;count+=1.0;}}
  value=select(all/16.0,liquid/count,count>0.0);source=count>0.0;
 }
 if(ufxClosed(p,a)){return vec2f(value,${BLOCKED}.0);}
 if(source){return vec2f(value,0.0);}
 return vec2f(value,select(${BLOCKED}.0,${U},(umTileSupport(t)&2u)!=0u));
}
fn ufxStateIndex(block:u32,l:vec3u,a:u32)->u32{return 3u*(512u*block+l.x+8u*l.y+64u*l.z)+a;}
fn ufxSeedIndex(block:u32,l:vec3u,a:u32)->u32{return UFX_SEEDS+ufxStateIndex(block,l,a);}
// The 4h state of face (C, a) on block b (C's page), (value, distance); blocked where b is not stored.
fn ufxCoarseOf(C:vec3i,a:u32,b:u32)->vec2f{if(!umBlockStored(b)){return vec2f(0.0,${BLOCKED}.0);}return ufxCoarse[ufxStateIndex(b,vec3u(C&vec3i(7)),a)];}
// The upwind root over the nearer neighbour of each axis (Zhao 2005; the
// first-order upwind root of Sethian 1999), the band cut-off at two cells
// (Bridson 2015 §4.4), then the neighbours below the root as the extension
// value, weighted by (root - d_n)/h_n² (Adalsteinsson & Sethian 1999).
// One axis's term of the extension value: (weight × the mean of its
// neighbours at the axis minimum m, weight), nothing at or above the root.
fn ufxTerm(m:f32,lo:vec2f,hi:vec2f,spacing:f32,root:f32,epsilon:f32)->vec2f{
 if(m>=root-epsilon){return vec2f(0.0);}
 var value=0.0;var count=0.0;
 if(abs(lo.y-m)<=epsilon){value+=lo.x;count+=1.0;}
 if(abs(hi.y-m)<=epsilon){value+=hi.x;count+=1.0;}
 let weight=(root-m)/(spacing*spacing);return vec2f(weight*value/max(count,1.0),weight);
}
// Scalar throughout: no array a lane indexes, so nothing spills.
fn ufxCombine(old:vec2f,l0:vec2f,h0:vec2f,l1:vec2f,h1:vec2f,l2:vec2f,h2:vec2f,h:vec3f)->vec2f{
 let m0=min(l0.y,h0.y);let m1=min(l1.y,h1.y);let m2=min(l2.y,h2.y);
 // The minima ascending, each with its spacing.
 var s0=m0;var s1=m1;var s2=m2;var g0=h.x;var g1=h.y;var g2=h.z;
 if(s1<s0){let d=s0;s0=s1;s1=d;let g=g0;g0=g1;g1=g;}
 if(s2<s0){let d=s0;s0=s2;s2=d;let g=g0;g0=g2;g2=g;}
 if(s2<s1){let d=s1;s1=s2;s2=d;let g=g1;g1=g2;g2=g;}
 if(s0>=${HALF}){return old;}
 var root=s0+g0;
 if(s1<${HALF}&&root>s1){
  let w0=1.0/(g0*g0);let w1=1.0/(g1*g1);
  var a=w0+w1;var b=s0*w0+s1*w1;var c=-1.0+s0*s0*w0+s1*s1*w1;
  root=(b+sqrt(max(b*b-a*c,0.0)))/a;
  if(s2<${HALF}&&root>s2){
   let w2=1.0/(g2*g2);a=a+w2;b=b+s2*w2;c=c+s2*s2*w2;
   root=(b+sqrt(max(b*b-a*c,0.0)))/a;
  }
 }
 let cut=2.0*min(h.x,min(h.y,h.z));
 if(root>cut||root>old.y){return old;}
 let epsilon=max(root,cut)*1.1920929e-7;
 let t=ufxTerm(m0,l0,h0,h.x,root,epsilon)+ufxTerm(m1,l1,h1,h.y,root,epsilon)+ufxTerm(m2,l2,h2,h.z,root,epsilon);
 return vec2f(select(0.0,t.x/t.y,t.y>0.0),root);
}
// ---- The block window, one axis at a time, shared by both passes: a
// block's 8³ cells (an h brick's, or a 4h page's) and a halo of two, 12³
// cells c (-2..9 an axis, from its first cell), (value, distance) each. Two
// Jacobi sweeps: the first over the window inset by one (10³), whose
// stencils stay inside it, the second over the block's cells, a lane each,
// whose stencils stay inside the first's. Those are then exactly two global
// sweeps, whatever the window's edge holds.
var<workgroup> ufxBV:array<f32,1728>;
var<workgroup> ufxBD:array<f32,1728>;
fn ufxBrickCell(k:u32)->vec3i{return vec3i(vec3u(k%12u,(k/12u)%12u,k/144u))-vec3i(2);}
fn ufxBrickInset(j:u32)->vec3i{return vec3i(vec3u(j%10u,(j/10u)%10u,j/100u))-vec3i(1);}
${sweep("ufxB","ufxBV","ufxBD",12)}// The first sweep (two cells a lane), held until every lane has read. Holds barriers.
fn ufxBSweep0(lane:u32,h:vec3f){
 var held:array<vec2f,2>;
 for(var r=0u;r<2u;r++){let k=lane+512u*r;if(k<1000u){held[r]=ufxBSweepAt(ufxBrickInset(k),0u,h);}}
 workgroupBarrier();
 for(var r=0u;r<2u;r++){let k=lane+512u*r;if(k<1000u){let i=ufxBAt(ufxBrickInset(k));ufxBV[i]=held[r].x;ufxBD[i]=held[r].y;}}
 workgroupBarrier();
}
// The 27-neighbour record (offset x+3y+9z of (-1..1)³): word, h tile mask, h-store texel of its first cell.
var<workgroup> ufxNW:array<u32,27>;
var<workgroup> ufxNM:array<u32,27>;
var<workgroup> ufxNT:array<vec3i,27>;
var<workgroup> ufxMode:u32;
var<workgroup> ufxCount:u32;
var<workgroup> ufxSupported:atomic<u32>;
// The record neighbour holding window cell r (-8..15 an axis, from the block's first cell).
fn ufxNeighbourIndex(r:vec3i)->u32{let d=vec3u(r+vec3i(8))>>vec3u(3u);return d.x+3u*d.y+9u*d.z;}
// ---- h: a live brick's cells. The h-store texel of brick id's first cell
// (its page's h slot) packed, UB_NONE_ID where the page has none: the slot
// and the brick's place in its page. A cell's texel is then that plus its
// offset in the brick (a brick lies in one page); x < 0 unpacked of none.
fn ufxTexelPack(id:u32)->u32{let f=umFineSlot(id>>6u);if(f==0u){return UB_NONE_ID;}return (f<<6u)|(id&63u);}
fn ufxTexelOf(t:u32)->vec3i{if(t==UB_NONE_ID){return vec3i(-1);}let l=t&63u;return (umBlockCoord(t>>6u)<<vec3u(5u))|vec3i(8u*vec3u(l&3u,(l>>2u)&3u,l>>4u));}
fn ufxLoad(T:vec3i,l:vec3u,p:vec3i)->vec4f{if(T.x<0){return textureLoad(physical,p,0);}return udTexel_physicalTexels(T+vec3i(l));}
// Cell l's packed seeds (2 bits an axis) in live slot w.
fn ufxCodes(w:u32,l:vec3u)->u32{return (ufxResult[128u*w+((l.x+8u*l.y+64u*l.z)>>2u)]>>(8u*(l.x&3u)))&63u;}
// The 4h state of the brick's 4h cells and their -e_a neighbours, by
// window cell (-2..2 an axis from the brick's first 4h cell), 125 an axis:
// every 4h face an h face of the brick window prolongs from.
var<workgroup> ufxP4:array<vec2f,375>;
fn ufxAt4(C:vec3i,base:vec3i,a:u32)->vec2f{let d=vec3u(C-base+vec3i(2));return ufxP4[125u*a+d.x+5u*d.y+25u*d.z];}
// The 4h state prolonged to h face (p, a), base the brick's first 4h cell:
// linear between the 4h owner's two faces on the axis, constant across it
// (the 4x4 h subfaces of a 4h face carry its value). A coarse face weighs
// in only where its sweep reached it, for value and distance alike; with
// neither reached, the value is the owner's own face's and the distance
// unreached.
fn ufxProlong(p:vec3i,a:u32,base:vec3i)->vec2f{
 let C=p>>vec3u(2u);let hi=ufxAt4(C,base,a);let khi=ufxReached(hi);let own=vec2f(hi.x,select(${U},hi.y,khi));
 let q=(p[a]&3)+1;if(q==4){return own;}
 let lo=ufxAt4(C-ufxAxis(a),base,a);
 if(!ufxReached(lo)){return own;}
 if(khi){return mix(lo,hi,f32(q)*0.25);}
 return lo;
}
// The brick's 4h state. Holds a barrier.
fn ufxStage4(lane:u32,base:vec3i){
 if(lane<375u){let d=lane%125u;let C=base+vec3i(vec3u(d%5u,(d/5u)%5u,d/25u))-vec3i(2);ufxP4[lane]=ufxCoarseOf(C,lane/125u,umPageWord(C>>vec3u(3u)));}
 workgroupBarrier();
}
var<workgroup> ufxOrigin:array<u32,3>;
// An h face's published value from its state s (value, distance): one no
// sweep reached takes the 4h band's where that reached it, else keeps its
// own; a closed wall (code 3) keeps its own. base: the brick's first 4h cell.
fn ufxFinishH(s:vec2f,p:vec3i,a:u32,code:u32,base:vec3i)->f32{
 if(ufxReached(s)||code==3u){return s.x;}
 let far=ufxProlong(p,a,base);return select(s.x,far.x,ufxReached(far));
}
// ---- 4h publication: the page's 4h cells (bit 0 a 4h tile; bit 1 + a its
// +e_a neighbour an h tile, a seam; bit 4 + a that face a closed wall; bit
// 8 + a the swept state reached it), the phase of each seam cell's owner,
// the swept values, and the list of seam cells.
var<workgroup> ufxCell:array<u32,512>;
var<workgroup> ufxCellPhase:array<f32,512>;
var<workgroup> ufxS:array<f32,1536>;
var<workgroup> ufxSeams:array<u32,512>;
// A 4h tile's +face texel L (any coordinate 3) of page cell i: each axis on
// whose plane L lies gives a component, the canonical face's value or,
// against an h tile, the subface's own (a source, a closed wall, or a face
// no sweep reached) or the 4h one. An unswept page publishes its own
// values: there each face's state is its seed, which this rule publishes as
// the texel's own (the canonical texel, or a source or blocked subface).
fn ufxPublish4(C:vec3i,L:vec3i,i:u32,info:u32,swept:bool){
 let q=4*C+L;let own=textureLoad(physical,q,0);var value=vec4f(0.0,0.0,0.0,own.w);
 for(var a=0u;a<3u;a++){
  if(L[a]!=3){continue;}
  if(!swept){value[a]=own[a];continue;}
  let s=ufxS[512u*a+i];
  if(((info>>(1u+a))&1u)==0u){value[a]=s;continue;}
  let source=((info>>(4u+a))&1u)!=0u||ufxCellPhase[i]>0.5||textureLoad(phase,q+ufxAxis(a),0).x>0.5;
  value[a]=select(own[a],s,!source&&((info>>(8u+a))&1u)!=0u);
 }
 textureStore(output,q,value);
}
// One workgroup a held page: its flag, whether any of its 512 tiles (all
// live in a held block) has the support bit in a region the freeze does
// not hold.
@compute @workgroup_size(128) fn ufxF(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ufxCount=ubaLevelCount(2u);atomicStore(&ufxSupported,0u);}
 let n=workgroupUniformLoad(&ufxCount);
 for(var e=group.x;e<n;e+=groups.x){
  let block=ubaLevelSlot(2u,e);
  for(var j=lane;j<512u;j+=128u){let t=512u*block+j;if((umTileSupport(t)&2u)!=0u&&!umFrozenRegion(umPageOfTile(t))){atomicStore(&ufxSupported,1u);}}
  workgroupBarrier();
  if(lane==0u){ufxCoarse[UFX_FLAGS+block]=vec2f(f32(atomicLoad(&ufxSupported)),0.0);atomicStore(&ufxSupported,0u);}
  workgroupBarrier();
 }
}
// A thread a face of a held page: (pages, 12) groups of 128 faces, cells
// fastest, an axis 4 groups. The page's 27 flags give its active bit, and
// only an active page restricts.
@compute @workgroup_size(128) fn ufxR(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ufxCount=ubaLevelCount(2u);atomicStore(&ufxSupported,0u);}
 let n=workgroupUniformLoad(&ufxCount);let a=group.y>>2u;let k=((group.y&3u)<<7u)|lane;let l=vec3u(k&7u,(k>>3u)&7u,k>>6u);
 for(var e=group.x;e<n;e+=groups.x){
  let block=ubaLevelSlot(2u,e);let page=vec3i(umBlockPage(block));
  if(lane<27u){let b=umPageWord(page+vec3i(i32(lane%3u),i32((lane/3u)%3u),i32(lane/9u))-vec3i(1));if(umBlockStored(b)&&ufxFlagged(b)){atomicStore(&ufxSupported,1u);}}
  workgroupBarrier();if(lane==0u){ufxMode=atomicLoad(&ufxSupported);atomicStore(&ufxSupported,0u);}
  let nearby=workgroupUniformLoad(&ufxMode)!=0u;
  if(nearby){ufxCoarse[ufxSeedIndex(block,l,a)]=ufxRestrict4(vec3i((page<<vec3u(3u))+vec3i(l)),a);}
  if(group.y==0u&&lane==0u){ufxCoarse[UFX_ACTIVE+block]=vec2f(select(0.0,1.0,nearby),0.0);}
 }
}
// One workgroup a held page, a lane a 4h cell: (pages) groups (the pool's
// blocks; a group past the count exits).
// - A flagged page stages and sweeps its window an axis at a time, its
//   neighbours' seeds (they are active, so ufxR wrote them). A page none of
//   whose warm tiles has the support bit holds only sources and blocked
//   faces, which no sweep changes (a source's root never beats 0, a blocked
//   face is never swept): an active one's state is its seeds, copied.
// - Publication, none of a frozen region, and an old field's only on a swept
//   page: a 4h tile's lane writes its three anchors (origin + 3 on each
//   axis, the canonical texels); on an axis whose +neighbour is an h tile,
//   the plane's 16 subface texels instead, the page's seam cells shared out
//   over the workgroup.
@compute @workgroup_size(512) fn ufx4(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ufxCount=ubaLevelCount(2u);}
 let n=workgroupUniformLoad(&ufxCount);let h=4.0*ufxParams.xyz;
 let l=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);let o=(l.x>>2u)|((l.y>>2u)<<1u)|((l.z>>2u)<<2u);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);let page=vec3i(umBlockPage(slot)<<vec3u(3u));let C=page+vec3i(l);
  if(lane<27u){ufxNW[lane]=ubaLevelRecord(2u,slot,lane);}
  if(lane==27u){ufxMode=select(0u,1u,ufxFlagged(slot))|select(0u,2u,ufxActive(slot));atomicStore(&ufxSupported,0u);}
  let mode=workgroupUniformLoad(&ufxMode);let swept=(mode&1u)!=0u;var info=0u;
  if(swept){
   for(var a=0u;a<3u;a++){
    for(var r=0u;r<4u;r++){
     let k=lane+512u*r;if(k>=1728u){continue;}
     let c=ufxBrickCell(k);let w=ufxNW[ufxNeighbourIndex(c)];var s=vec2f(0.0,${BLOCKED}.0);
     if(w!=UB_NONE_ID){s=ufxCoarse[ufxSeedIndex(w,vec3u(c&vec3i(7)),a)];}
     ufxBV[k]=s.x;ufxBD[k]=s.y;
    }
    workgroupBarrier();
    ufxBSweep0(lane,h);
    let s=ufxBSweepAt(vec3i(l),0u,h);ufxCoarse[ufxStateIndex(slot,l,a)]=s;ufxS[512u*a+lane]=s.x;
    if(ufxReached(s)){info|=256u<<a;}
    workgroupBarrier();
   }
  }
  else if((mode&2u)!=0u){for(var a=0u;a<3u;a++){let k=ufxStateIndex(slot,l,a);ufxCoarse[k]=ufxCoarse[UFX_SEEDS+k];}}
  var ph=0.0;
  if(!umFrozenRegion(8u*slot+o)&&(swept||!ufxOld)){
   let t=512u*slot+umTileLocal(l);
   if(umTileLive(t)&&umTileWidth(t)!=1u){
    info|=1u;
    for(var a=0u;a<3u;a++){
     let e=ufxAxis(a);let lC=vec3i(l);var nt=512u*slot+umTileLocal(vec3u(lC+e));
     if(lC[a]==7){let w=ufxNW[13u+select(select(9u,3u,a==1u),1u,a==0u)];nt=select(512u*w+umTileLocal(vec3u(lC+e)),UM_TILES,w==UB_NONE_ID);}
     if(nt<UM_TILES&&ufxFineTile(nt)){info|=2u<<a;if(ufxClosed(4*C+3*e,a)){info|=16u<<a;}}
    }
    if((info&14u)!=0u){ph=textureLoad(phase,4*C,0).x;ufxSeams[atomicAdd(&ufxSupported,1u)]=lane;}
   }
  }
  ufxCell[lane]=info;ufxCellPhase[lane]=ph;
  // The anchors off a seam plane, by their own lane (its writes above are its own to read).
  if((info&1u)!=0u){for(var a=0u;a<3u;a++){if(((info>>(1u+a))&1u)==0u){ufxPublish4(C,3*ufxAxis(a),lane,info,swept);}}}
  workgroupBarrier();
  let seams=atomicLoad(&ufxSupported);
  for(var k=lane;k<64u*seams;k+=512u){
   let i=ufxSeams[k>>6u];let cell=ufxCell[i];let j=k&63u;let L=vec3i(vec3u(j&3u,(j>>2u)&3u,j>>4u));
   let seam=vec3<bool>((cell&2u)!=0u,(cell&4u)!=0u,(cell&8u)!=0u);
   if(any((L==vec3i(3))&seam)){ufxPublish4(page+vec3i(vec3u(i&7u,(i>>3u)&7u,i>>6u)),L,i,cell,swept);}
  }
  workgroupBarrier();
 }
}
// A thread a tile row (4 cells, x fastest) of a live brick: (bricks)
// groups of 128 rows. First the slot's header, every live brick's: its mode
// (bit 0 a supported h tile, bit 1 + i tile i (x, y, z bits) h-owned; 0 in
// a frozen region), its origin, and its record (lanes 32..58). Then, on an
// active page, the seeds. Only a live h tile's cells are ever staged as a
// core (the record's tile mask), and only by a brick with a supported
// tile, whose neighbours lie in active pages.
@compute @workgroup_size(128) fn ufxRH(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let n=ubaCount(UBA_LIVE_COUNT);let k=4u*lane;let l=vec3i(vec3u(k&7u,(k>>3u)&7u,k>>6u));
 for(var e=group.x;e<n;e+=groups.x){
  let slot=atomicLoad(&ufbTable[UBA_LIVE+e]);let id=ubaBrick(slot);let H=ufxHeader(slot);
  if(lane==0u){
   let origin=ubaOrigin(id);var mode=0u;
   if(!umFrozenRegion(ubaParentKey(id))){
    for(var i=0u;i<8u;i++){
     let t=512u*(id>>6u)+umTileLocal((origin>>vec3u(2u))+vec3u(i&1u,(i>>1u)&1u,i>>2u));
     if(ufxFineTile(t)){mode|=(2u<<i)|select(0u,1u,(umTileSupport(t)&2u)!=0u);}
    }
   }
   ufxResult[H]=mode;ufxResult[H+1u]=origin.x;ufxResult[H+2u]=origin.y;ufxResult[H+3u]=origin.z;
  }
  if(lane>=32u&&lane<59u){
   let j=lane-32u;let w=ubaLevelRecord(0u,slot,j);var m=0u;var t=UB_NONE_ID;
   if(w<UBA_SEAM){m=ubaRecordWord(w,28u);t=ufxTexelPack(ubaBrick(w));}
   ufxResult[H+4u+j]=w;ufxResult[H+31u+j]=m;ufxResult[H+58u+j]=t;
  }
  if(!ufxActive(id>>6u)){continue;}
  let p=vec3i(ubaOrigin(id))+l;let t=ufxTile(p);var word=0x2a2a2a2au;
  if(ufxFine(t)){word=0u;for(var i=0u;i<4u;i++){for(var a=0u;a<3u;a++){word|=ufxSeedCodeOf(p+vec3i(i32(i),0,0),a,t)<<(8u*i+2u*a);}}}
  ufxResult[128u*slot+lane]=word;
 }
}
// One workgroup a live brick, a lane a cell: (bricks) groups, its header
// (ufxRH) loaded in one round. A brick of a frozen region is held: no
// sweep, no texel. A brick with a supported h
// tile stages and sweeps its window an axis at a time; its h cells publish.
// (A tile without support holds sources and blocked faces, which the sweep
// leaves.) An old field publishes only a swept brick's.
@compute @workgroup_size(512) fn ufxH(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ufxCount=ubaCount(UBA_LIVE_COUNT);}
 let n=workgroupUniformLoad(&ufxCount);let h=ufxParams.xyz;
 let l=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);let tile=(l.x>>2u)|((l.y>>2u)<<1u)|((l.z>>2u)<<2u);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=atomicLoad(&ufbTable[UBA_LIVE+e]);let H=ufxHeader(slot);
  if(lane<27u){ufxNW[lane]=ufxResult[H+4u+lane];ufxNM[lane]=ufxResult[H+31u+lane];ufxNT[lane]=ufxTexelOf(ufxResult[H+58u+lane]);}
  else if(lane==27u){ufxMode=ufxResult[H];}
  else if(lane<31u){ufxOrigin[lane-28u]=ufxResult[H+lane-27u];}
  let tiles=workgroupUniformLoad(&ufxMode);let owned=((tiles>>(1u+tile))&1u)!=0u;
  let origin=vec3i(vec3u(ufxOrigin[0],ufxOrigin[1],ufxOrigin[2]));let p=origin+vec3i(l);
  if((tiles&1u)!=0u){
   let base=origin>>vec3u(2u);ufxStage4(lane,base);
   // The lane's window cells (four at most), classified once: a core cell's
   // texel and seeds (kind 1), a seam (kind 2), nothing (0).
   var kind:array<u32,4>;var word:array<u32,4>;var texel:array<vec4f,4>;
   for(var r=0u;r<4u;r++){
    let k=lane+512u*r;kind[r]=0u;
    if(k<1728u){
     let c=ufxBrickCell(k);let nb=ufxNeighbourIndex(c);let lr=vec3u(c&vec3i(7));let w=ufxNW[nb];
     if(w<UBA_SEAM&&((ufxNM[nb]>>((lr.x>>2u)|((lr.y>>2u)<<1u)|((lr.z>>2u)<<2u)))&1u)!=0u){kind[r]=1u;texel[r]=ufxLoad(ufxNT[nb],lr,origin+c);word[r]=ufxCodes(w,lr);}
     else if(w!=UB_NONE_ID){kind[r]=2u;}
    }
   }
   let codes=ufxCodes(slot,l);var out=vec3f(0.0);
   for(var a=0u;a<3u;a++){
    for(var r=0u;r<4u;r++){
     let k=lane+512u*r;if(k>=1728u){continue;}
     var s=vec2f(0.0,${BLOCKED}.0);
     if(kind[r]==1u){s=vec2f(texel[r][a],ufxSeedCode((word[r]>>(2u*a))&3u));}
     else if(kind[r]==2u){s=ufxProlong(origin+ufxBrickCell(k),a,base);}
     ufxBV[k]=s.x;ufxBD[k]=s.y;
    }
    workgroupBarrier();
    ufxBSweep0(lane,h);
    let s=ufxBSweepAt(vec3i(l),0u,h);
    if(owned){out[a]=ufxFinishH(s,p,a,(codes>>(2u*a))&3u,base);}
    workgroupBarrier();
   }
   if(owned){textureStore(output,p,vec4f(out,ufxLoad(ufxNT[13u],l,p).w));}
  }
  else if(owned&&!ufxOld){textureStore(output,p,ufxLoad(ufxNT[13u],l,p));}
  workgroupBarrier();
 }
}
`;
 }
 async initialize():Promise<void>{
  // The passes address the fields by lattice coordinate, which only the
  // detail rewrite resolves to their paged texels: a device without the
  // solver's live storage (another wrapper of it) would leave the loads and
  // stores raw, reading zeros and dropping every store.
  const o=this.ownership,code=this.code();
  if(uniformDetailShader(code,this.device)===code)throw new Error("Frame extension: no live detail storage on this device, so the field accesses would not be rewritten");
  const module=uniformDetailModule(this.device,{label:"Uniform frame extension",code});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(`Frame extension: ${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,this.layout]});
  const make=(entryPoint:string,old=false)=>uniformDetailPipeline(this.device,o,{layout,compute:{module,entryPoint,constants:{umDispatchX:o.dispatchX,...(old?{ufxOld:1}:{})}}});
  const [flag,restrict,coarse,coarseOld,seed,fine,fineOld]=await Promise.all([make("ufxF"),make("ufxR"),make("ufx4"),make("ufx4",true),make("ufxRH"),make("ufxH"),make("ufxH",true)]);
  this.pipelines={flag:flag!,restrict:restrict!,coarse:[coarse!,coarseOld!],seed:seed!,fine:[fine!,fineOld!]};
 }
 bind(f:UniformFrameExtensionFields):UniformFrameExtensionGroup{
  for(const [i,t] of [f.physical,f.phase,f.output].entries())if(t.format!==(i===1?"r32float":"rgba32float")||uniformDetailClass(t)!==(i===1?"cell":"face"))throw new Error("Frame extension requires native canonical fields");
  if(f.physical===f.output||f.negative===f.outputNegative)throw new Error("Frame extension outputs must be disjoint");
  const detail=uniformDetailGroup(this.device,{label:"Uniform frame extension",layout:this.layout,entries:[
   {binding:0,resource:f.physical},{binding:1,resource:f.phase},{binding:2,resource:f.output},
   {binding:3,resource:{buffer:this.bricks.table}},{binding:4,resource:{buffer:this.coarse}},{binding:5,resource:{buffer:this.result}},
   {binding:6,resource:{buffer:f.params,size:16}},
   ...(this.freeze?[{binding:7,resource:this.freeze.regions.createView()}]:[]),
   {binding:8,resource:f.physical},
  ]});
  return {detail,negative:f.negative,outputNegative:f.outputNegative,old:f.old===true};
 }
 /** The extension of the bound fields, on the frame bricks' current table
  * (encodeUpdate this step). seeded: an extension since that update, with
  * the same phase, topology and freeze, wrote the flags, the h seeds and
  * the headers (the tail's, after the old field's: pressure changes none
  * of them), so only the restriction, the sweeps and publication run. */
 encode(encoder:GPUCommandEncoder,group:UniformFrameExtensionGroup,seeded=false):void{
  const p=this.pipelines;if(!p)throw new Error("Frame extension is not initialized");
  // An old field holds no low face: the delta reads none.
  if(!group.old)encoder.copyBufferToBuffer(group.negative,0,group.outputNegative,0,Math.min(group.negative.size,group.outputNegative.size));
  const pass=encoder.beginComputePass(PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.detail.group);
  // Flags and the restriction (pages), the 4h sweep (a workgroup a
  // page), the h seeds and headers (bricks) and sweep (a workgroup a
  // brick): each launch strides its live count over a fixed grid.
  const old=group.old?1:0;
  if(!seeded){pass.setPipeline(p.flag);pass.dispatchWorkgroups(this.pageGroups);}
  pass.setPipeline(p.restrict);pass.dispatchWorkgroups(this.pageGroups,12);
  pass.setPipeline(p.coarse[old]);pass.dispatchWorkgroups(this.pageGroups);
  if(!seeded){pass.setPipeline(p.seed);pass.dispatchWorkgroups(this.brickGroups);}
  pass.setPipeline(p.fine[old]);pass.dispatchWorkgroups(this.brickGroups);
  pass.end();
 }
 destroy():void{this.coarse.destroy();this.result.destroy();}
}
