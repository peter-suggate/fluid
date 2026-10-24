import { uniformCompiledVertexResolveWGSL } from "./uniform-compiled-topology";
import { uniformDetailBindLayout, uniformDetailClass, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedPageCount,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {levelSetFillWGSL} from "../../core/level-set-fill.wgsl";
import {type UniformLatticeAddress} from "./uniform-lattice-address";
import {UNIFORM_LIVE_KEY_GROUPS} from "./uniform-live-lists";
import {uniformStageListsWGSL} from "./uniform-stage-lists";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,uniformMixedVertexBuriedWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";

/** NB-FLIP's scalar volume budget: the liquid's filled volume measured on a
 * native four-cell band, and the global normal shift that would restore the
 * budget, found by secant Newton rounds. Phi is read, never written. Fields
 * and scratch are borrowed. With static
 * solids a unit owner's capacity is its native open fraction: closed cells
 * neither seed nor carry the band, and slopes use only open-sided axes.
 *
 * Only the seed visits every owner. Each dilation moves the band at most one
 * owner, so at most one tile: a step visits the tiles within one tile of a
 * tile already carrying band, and every other owner keeps the zero the seed
 * wrote into both parities. Measure shifts the owners one tile around the
 * final band tiles (a vertex's incident cells lie in adjacent tiles); metric
 * adds every tile with a wider owner in its stencil. An owner of a skipped
 * tile has all-zero corner scales, so both its samples are one
 * constant: the seed writes those workgroup partials once, in the measure's
 * own reduction order, and the refinements rebuild only workgroups that can
 * differ. Every value equals the dense evaluation's.
 *
 * Residency (uniformMixedResidencyWord): owner passes stride the h tiles and
 * the 4h tiles of resident pages (UNIFORM_MIXED_COUNTED.residentAll), tile
 * passes the tiles of resident pages. An absent page's 4h owners are far air
 * (V=0, corner phi at least 16h, audited by the next census): unseeded, no
 * band, no liquid or volume, so their dense rows are zeros the sums lose
 * nothing without (only the grouping of the partial sums changes). clearBand
 * clears the band, visit and measure flags densely; a measured tile with a
 * neighbour in an absent page would read an owner the seed skipped, so it
 * sets the sticky closure bit (a fatal at the next census).
 *
 * bind()'s volume is the measured geometry output: read once at
 * initialization, then a 36-byte budget owns the target. The band spans four
 * h cells and the shift is bounded by 4h, including large timesteps; the
 * required shift is the budget's diagnostic. */
/** Floats of the NB scalar volume budget, and where its control rate sits. */
export const NB_VOLUME_BUDGET_FLOATS=9,NB_VOLUME_RATE_FLOAT=8;
/** The most volume control may ask of the liquid, in 1/s: the long dam break
 * at 1/30 s loses its volume at about 1/s where its front lands. */
const NB_VOLUME_RATE_LIMIT=4;
/** The share of the budget volume control leaves alone. The gauge is not the
 * capture: a resting film over a voxel ledge measures half a percent off the
 * budget it was captured at, and a flat pool a cell in ten thousand. Liquid
 * at rest must be asked for nothing, or the divergence moves its waterline
 * and wakes the adaptive surface. */
const NB_VOLUME_DEAD_BAND=0.01;
const VOLUME_PASS:GPUComputePassDescriptor={label:"Uniform mixed global surface volume"},BUDGET_PASS:GPUComputePassDescriptor={label:"NB scalar volume sources"};
export class UniformMixedSurfaceVolume {
 /** NB budget: target, initialized, a word no stage writes, last normal shift,
  * volume before correction, initial volume, last measured volume, target,
  * then the gauge's volume control rate in 1/s. */
 readonly budget:GPUBuffer;
 private readonly budgetParams:GPUBuffer;
 /** beginStep's parameter words: dt, added cells, open top, control rate. */
 private readonly budgetWords=new Float32Array(4);
 private budgetCaptured=false;
 get allocatedBytes():number{return 48;}
 /** Bytes of bind()'s scratch at that many h tiles. The sections the lattice
  * fixes come first: the corner scales (a word a vertex), the partial rows
  * and their chunk sums, the Newton state and three flags a tile. Then the
  * band's two parities, by owner index. */
 scratchBytesAt(fineTiles:number):number{return 4*(this.vertices+4*(this.groups+this.chunks)+8+3*this.tiles+2*(63*fineTiles+this.tiles));}
 get scratchBytes():number{return this.scratchBytesAt(this.ownership.capacity.fineTiles);}
 private readonly vertices:number;
 private readonly groups:number;
 private readonly chunks:number;
 private readonly tiles:number;
 /** Residency pages: the bound of the page launches. */
 private readonly pages:number;
 private readonly address:UniformLatticeAddress;
 /** clearBand's fixed grid, a lane a live tile (strided). */
 private readonly clearGroups:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** The caller runs UniformMixedPhiResolve on phi before encode: every owner
  * corner is then a stored or resolved texel; resolveScale completes the
  * corner scales the same way after metric. */
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid){
  const lattice=ownership.capacity.lattice;
  const address=this.address=ownership.capacity.address;
  this.vertices=address.vertexKeys;
  this.tiles=ownership.capacity.tiles;this.pages=uniformMixedPageCount(this.tiles);
  this.clearGroups=UNIFORM_LIVE_KEY_GROUPS;
  // Partial rows: the residentAll launch's job bound (svJobs): every tile h,
  // plus a job per page. Rows sized to the tiles alone let an all-h layout's
  // page jobs write the chunk sums, and its extra chunk the Newton state.
  this.groups=this.tiles+this.pages;this.chunks=Math.ceil(this.groups/64);
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  const scratch={binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}};
  this.budget=device.createBuffer({label:"NB scalar volume budget",size:4*NB_VOLUME_BUDGET_FLOATS,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.budgetParams=device.createBuffer({label:"NB volume budget parameters",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.resources=uniformDetailBindLayout(device,{entries:[texture(0),texture(1),scratch,{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]});
 }
 /** The stage measures phi and volume and writes neither. */
 bind(phi:GPUTexture,volume:GPUTexture,scratch:GPUBufferBinding):UniformDetailGroup{
  const address=this.address;
  for(const [i,t] of [phi,volume].entries())if(t.format!=="r32float"||uniformDetailClass(t)!==(i===1?"cell":"vertex"))throw new Error("Mixed surface constraint requires native vertex and cell fields");
  if((scratch.size??scratch.buffer.size-(scratch.offset??0))<this.scratchBytes)throw new Error("Mixed surface constraint needs sufficient scratch");
  const work={...scratch,size:this.scratchBytes};
  return uniformDetailGroup(this.device,{layout:this.resources,entries:[{binding:0,resource:phi},{binding:1,resource:volume},{binding:3,resource:work},{binding:5,resource:{buffer:this.budget}},{binding:6,resource:{buffer:this.budgetParams}}]});
 }
 async initialize():Promise<void>{
  const topology=uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformStageListsWGSL;
  const sourceScale=uniformMixedVertexSamplingSource("",false,undefined,undefined,true).replace(/\bum(Vertex(?!(?:At|Coord|Present|Near|Texel|CellAbove|CellBelow)\b)\w*|SampleVertex|LoadVertex8?)\b/g,name=>name.replace("um","umScale"));
  // Owner entries stride every live owner (all), resolveScale every seam
  // tile (fused): GPU-counted fixed grids, sized by capacity alone.
  const module=uniformDetailModule(this.device,{label:"Uniform mixed surface volume",code:uniformMixedCountedEntriesWGSL(topology+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var volume:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> scratch:array<f32>;
override parity:u32=0u;
@group(1) @binding(5) var<storage,read_write> budget:array<f32>;
@group(1) @binding(6) var<uniform> budgetParams:vec4f;
override captureBudget:bool=false;override gaugeBudget:bool=false;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// A tile corner, and below a 4h owner's own texel (its tile origin): the
// base blocks (UNIFORM_DETAIL_4H_LOAD).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
// Only the metric's nonzero-band gradient calls this scalar reader. Audit
// its actual storage slot, including boundary vertices, rather than a cube
// around every measured tile. Seed's far-air classification uses the 8-load
// helpers and does not create a residency dependency of its own.
fn umLoadVertexW(p:vec3u,width:u32)->f32{
 let slot=umVertexSlot(vec3i(p));
 if(slot.w==0u){umSupport[UM_RESIDENCY+1u]=1u;}
 if(width==4u){return umLoadCorner(p);}return umLoadVertex(p);}
// The same loads of the eight vertices p + s*umCorner(k,2u): one page-table read where one block holds them.
fn umLoadVertex8(p:vec3u,s:u32)->array<f32,8>{return udLoad8_phi(vec3i(p),s);}
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return udLoad48_phi(vec3i(p),s);}
fn umLoadVertexW8(p:vec3u,width:u32)->array<f32,8>{if(width==4u){return umLoadCorner8(p,4u);}return umLoadVertex8(p,width);}
${uniformMixedVertexSamplingSource("",true,undefined,"umLoadCorner",true)}
const UM_H=vec3f(${this.ownership.capacity.lattice.cellSize_m.join(",")});
// The corner scales come first, a word a vertex (umVertexAt). Then the
// partial rows, a job each: the tiles and the residency pages (UM_PAGES).
// Band tiles, visited tiles, measured tiles.
const SV_P:u32=UM_VERTEX_KEYS;
const SV_GROUPS:u32=UM_TILES+UM_PAGES;
const SV_CHUNKS:u32=(SV_GROUPS+63u)/64u;
const SV_R:u32=SV_P+4u*SV_GROUPS;const SV_S:u32=SV_R+4u*SV_CHUNKS;
const SV_BAND:u32=SV_S+8u;const SV_VISIT:u32=SV_S+8u+UM_TILES;const SV_MEASURE:u32=SV_S+8u+2u*UM_TILES;
// The band by owner index: two parities (svBandAt), the last section.
const SV_OWNERS:u32=SV_S+8u+3u*UM_TILES;
fn svVisited(o:UMOwner)->bool{return scratch[SV_VISIT+o.tile]!=0.0;}
// Every incident owner of a unit-stencil owner's corners has unit width: the
// corners are stored, and a corner belongs to the lowest incident cell, so
// the owner holds its positive corner and its corners on negative walls.
fn svUnit(o:UMOwner)->bool{return umTileMaximumWidth(o.tile)==1u;}
// low: umLowOpen of the owner's origin (read for a unit-stencil owner).
fn svAuthority(o:UMOwner,k:u32,low:vec3<bool>)->bool{
 if(o.width==4u){return (umCoarseCornerMask(o.tile)&(1u<<k))!=0u;}
 let corner=umCorner(k,2u);
 if(svUnit(o)){return all((corner!=vec3u(0))|low);}
 return umVertexAuthority(umOrigin(o)+corner*o.width).index==o.index;
}
// Tile of a unit-stencil owner's corner authority, clamp(p-1)'s owner.
fn svUnitAuthorityTile(p:vec3u)->u32{return umTileAt(vec3u(umVertexCellBelow(vec3i(p)))/4u);}
fn svMeasured(t:u32)->bool{return scratch[SV_MEASURE+t]!=0.0;}
// An absent vertex has no metric contribution. Shared scratch is not a
// persistent zero image for the AIR/SOLID sentinel blocks.
fn umScaleLoadVertex(p:vec3u)->f32{if(!umVertexPresent(vec3i(p))){return 0.0;}return scratch[umVertexAt(p)];}
// The same words of the eight vertices p + s*umCorner(k,2u): where one block holds them, its first key and fixed steps.
fn umScaleLoadVertex8(p:vec3u,s:u32)->array<f32,8>{
 var v:array<f32,8>;let run=umVertexRun(vec3i(p),s);
 if(run.x!=0u){let first=umVertexRunAt(run);for(var k=0u;k<8u;k++){let c=umCorner(k,2u)*s;v[k]=scratch[first+c.x+33u*c.y+1089u*c.z];}return v;}
 for(var k=0u;k<8u;k++){v[k]=umScaleLoadVertex(p+umCorner(k,2u)*s);}
 return v;
}
${sourceScale}
${levelSetFillWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}${this.solid?uniformMixedVertexBuriedWGSL:""}
// What an owner can hold: its open fraction (an h cell's, a 4h owner's tile's).
fn umCapacity(o:UMOwner)->f32{if(!umSolidEnabled()){return 1.0;}if(o.width==1u){return umCellOpen(vec3i(umOrigin(o)));}return umTileOpen(o.tile);}
// A cut 4h owner's filled volume in h cells: the tetrahedral fill of each h
// cell of its tile under the owner's trilinear field, times the cell's open
// fraction.
fn svOpenFill(v:array<f32,8>,origin:vec3i)->f32{
 var sum=0.0;
 for(var s=0u;s<64u;s++){
  let cell=vec3u(s&3u,(s>>2u)&3u,s>>4u);let open=umCellOpen(origin+vec3i(cell));
  if(open<=0.0){continue;}
  let base=vec3f(cell);var c:array<f32,8>;var negative=0u;
  for(var k=0u;k<8u;k++){
   let f=0.25*(base+vec3f(umCorner(k,2u)));
   let low=mix(mix(v[0],v[1],f.x),mix(v[2],v[3],f.x),f.y);let high=mix(mix(v[4],v[5],f.x),mix(v[6],v[7],f.x),f.y);
   c[k]=mix(low,high,f.z);negative+=select(0u,1u,c[k]<0.0);
  }
  if(negative==8u){sum+=open;}else if(negative!=0u){sum+=open*fill(c);}
 }
 return sum;
}
// Partial rows: one per job of the residentAll launch (h tiles, then resident pages).
fn svJobs()->u32{return umCounts.x+umResidentPageCount();}
// A band parity's base: the second follows the live owners of the first.
fn svBandAt(side:u32)->u32{return SV_OWNERS+side*(64u*umCounts.x+umCounts.y);}
fn umShiftLimit()->f32{return min(UM_H.x,min(UM_H.y,UM_H.z))*4.0;}
// A lane's row: filled mass at the shift and one secant step ahead; the
// budget is the target, so the row's third word stays zero.
var<workgroup> sums:array<vec4f,64>;
var<workgroup> measureLive:atomic<u32>;
var<workgroup> measureBand:atomic<u32>;
// Seed's corner values die at the measureBand uniform barrier. Reuse the
// same 512 bytes for its 64 two-float reduction rows after that barrier.
var<workgroup> seedScratch:array<f32,128>;
fn sumGroup(l:u32){workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l<stride){sums[l]+=sums[l+stride];}workgroupBarrier();}}
fn storeSum(at:u32,value:vec4f){for(var c=0u;c<4u;c++){scratch[at+c]=value[c];}}
fn loadSum(at:u32)->vec4f{return vec4f(scratch[at],scratch[at+1u],scratch[at+2u],scratch[at+3u]);}
// The workgroup's partial from its lanes' rows. A group without liquid or
// volume sums to zero: skip its tree.
fn storePartial(l:u32,group:vec3u){
 workgroupBarrier();
 if(any(sums[l]!=vec4f(0))){atomicOr(&measureLive,1u);}
 let at=SV_P+4u*(group.x+umDispatchX*group.y);
 if(workgroupUniformLoad(&measureLive)==0u){if(l<4u){scratch[at+l]=0.0;}return;}
 sumGroup(l);
 if(l==0u){storeSum(at,sums[0]);}
}
// Clears the live tiles' flags (a fixed grid striding them); the first
// thread also starts the Newton state: shift, the bracket's low and high,
// converged, secant step.
@compute @workgroup_size(64) fn clearBand(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var j=gid.x;j<umLiveTileCount();j+=64u*groups.x){let t=umLiveTile(j);scratch[SV_BAND+t]=0.0;scratch[SV_VISIT+t]=0.0;scratch[SV_MEASURE+t]=0.0;}
 if(gid.x==0u){let limit=umShiftLimit();scratch[SV_S]=0.0;scratch[(SV_S+1u)]=-limit;scratch[(SV_S+2u)]=limit;scratch[(SV_S+3u)]=0.0;scratch[(SV_S+4u)]=limit/64.0;}
}
// A seed workgroup that is one unit-stencil fine tile stages its 5^3 stored
// vertices once; every lane then reads its eight corners from them.
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let job=group.x+umDispatchX*group.y;
 let staged=job<umCounts.x&&umTileMaximumWidth(umTopology[UM_TILES+job])==1u;
 // A workgroup runs several jobs (counted launch): clear the flags per job.
 if(l==0u){atomicStore(&measureLive,0u);atomicStore(&measureBand,0u);}
 if(staged){let base=umTileCoord(umTopology[UM_TILES+job])*4u;for(var i=l;i<125u;i+=64u){seedScratch[i]=umLoadVertex(base+umCorner(i,5u));}}
 workgroupBarrier();
 let o=umResidentAllOwner(gid);var inside=false;var seeded=false;
 if(o.width!=0u){
  if(umCapacity(o)>0.0){
   var low=1e30;var high=-1e30;
   if(staged){
    let local=umCorner(o.lane,4u);
    for(var k=0u;k<8u;k++){let c=local+umCorner(k,2u);let v=seedScratch[c.x+5u*(c.y+5u*c.z)];low=min(low,v);high=max(high,v);}
   }else{
    let c8=umLoadVertexW8(umOrigin(o),o.width);for(var k=0u;k<8u;k++){let v=c8[k];low=min(low,v);high=max(high,v);}
   }
   seeded=low<=0.0&&high>=0.0;inside=high<0.0;
  }
  let band=select(0.0,5.0,seeded);scratch[svBandAt(0u)+o.index]=band;scratch[svBandAt(1u)+o.index]=band;
 }
 if(seeded){scratch[SV_BAND+o.tile]=1.0;atomicOr(&measureBand,1u);}
 // A seeded group is always measured. Otherwise every corner scale of every
 // lane stays zero: each sample is the owner's whole mass or nothing. Both
 // samples are then one value per lane, and the measure tree adds every
 // component in the same order: reduce the sample once and replicate.
 if(workgroupUniformLoad(&measureBand)!=0u){return;}
 var row=vec2f(0);
 if(o.width!=0u){
  let mass=f32(o.width*o.width*o.width)*umCapacity(o);row=vec2f(select(0.0,mass,inside),0.0);
 }
 seedScratch[2u*l]=row.x;seedScratch[2u*l+1u]=row.y;
 if(any(row!=vec2f(0))){atomicOr(&measureLive,1u);}
 let at=SV_P+4u*(group.x+umDispatchX*group.y);
 if(workgroupUniformLoad(&measureLive)==0u){if(l<4u){scratch[at+l]=0.0;}return;}
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){seedScratch[2u*l]+=seedScratch[2u*(l+stride)];seedScratch[2u*l+1u]+=seedScratch[2u*(l+stride)+1u];}workgroupBarrier();}
 let total=vec2f(seedScratch[0],seedScratch[1]);
 // The partial: (s, s, 0, 0).
 if(l<4u){scratch[at+l]=select(select(0.0,total.y,l==2u),total.x,l<2u);}
}
// Visited: tiles within one tile of band. Every corner scale of an owner
// outside them is zero. Dilate tests the same rule inline; grow writes the
// final visit flags once the band is complete.
fn svNearBand(t:u32,k:u32)->bool{
 let q=vec3i(umTileCoord(t))+vec3i(umCorner(k,3u))-vec3i(1);
 return umTilePresent(q)&&scratch[SV_BAND+umTileAt(vec3u(q))]!=0.0;
}
@compute @workgroup_size(64) fn grow(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let t=umResidentPageTile(group.x,lane);if(!umTileLive(t)){return;}
 let centre=vec3i(umTileCoord(t));var visit=0.0;
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let q=centre+vec3i(x,y,z);if(!umTilePresent(q)){continue;}
  if(scratch[SV_BAND+umTileAt(vec3u(q))]!=0.0){visit=1.0;}
 }}}
 scratch[SV_VISIT+t]=visit;
}
// Measured (metric's set): the visited tiles plus every tile with a wider
// owner in its stencil within two tiles of band. A visited tile's hanging
// corners interpolate the corners of an adjacent wider owner, whose
// authorities lie one tile farther. Within two tiles of band is within one
// of a visited tile (the in-domain step toward the band tile is one), so
// this reads the final visit flags' 27 neighbours, not band's 125.
@compute @workgroup_size(64) fn measureGrow(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let t=umResidentPageTile(group.x,lane);if(!umTileLive(t)){return;}
 var measured=scratch[SV_VISIT+t];
 if(measured==0.0&&umTileMaximumWidth(t)>1u){
  let centre=vec3i(umTileCoord(t));
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
   let q=centre+vec3i(x,y,z);if(!umTilePresent(q)){continue;}
   if(scratch[SV_VISIT+umTileAt(vec3u(q))]!=0.0){measured=1.0;}
  }}}
 }
 scratch[SV_MEASURE+t]=measured;
 // The residency audit belongs at the reads in metric, after the band and
 // solid tests. A measured tile with zero incident band reads no phi slope;
 // requiring all 27 neighbours here kept unrelated diagonal air resident.
}
// NB measures this band in h cells, independent of owner width.
// A step dilates the owners of visited tiles (grow's rule on the band flags
// as this step reads them). Band flags another workgroup sets during the
// step only add tiles whose neighbourhood is zero in the input parity: their
// owners write the zero the output parity already holds (seed, monotone band).
// An h tile's lanes test its 27 neighbour tiles together, a 4h lane its own.
// A 4h owner's face is one patch, or the sixteen unit owners of an h tile's
// facing layer: that tile's lanes, with one ownership lookup a side.
var<workgroup> svVisit:atomic<u32>;
@compute @workgroup_size(64) fn dilate(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let fine=group.x<umCounts.x;
 if(l==0u){atomicStore(&svVisit,0u);}
 workgroupBarrier();
 let o=umResidentAllOwner(gid);
 if(fine&&l<27u&&umTileResident(o.tile)&&svNearBand(o.tile,l)){atomicOr(&svVisit,1u);}
 var visited=workgroupUniformLoad(&svVisit)!=0u;
 if(o.width==0u){return;}
 if(!fine){for(var k=0u;k<27u&&!visited;k++){visited=svNearBand(o.tile,k);}}
 if(!visited){return;}let input=svBandAt(parity);let out=svBandAt(parity^1u);var band=scratch[input+o.index];
 if(o.width==1u){
  // A unit owner's every face is one unit patch onto the adjacent owner.
  // Inside its own tile that owner is lane +-1, +-4 or +-16 (umOwnerAt's
  // index); only tile-face neighbours look ownership up.
  let origin=vec3i(umOrigin(o));let local=umCorner(o.lane,4u);
  for(var k=0u;k<6u;k++){
   let axis=k/2u;let up=(k&1u)==1u;let stride=select(select(16u,4u,axis==1u),1u,axis==0u);
   if(select(local[axis]>0u,local[axis]<3u,up)){band=max(band,scratch[input+select(o.index-stride,o.index+stride,up)]-1.0);continue;}
   var q=origin;q[axis]+=select(-1,1,up);let other=umOwnerAt(q);if(other.width!=0u){band=max(band,scratch[input+other.index]-f32(max(o.width,other.width)));}
  }
 }else{
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let other=umFace(o,axis,select(-1,1,side==1u),0u).neighbor;if(other.width==0u){continue;}
  if(other.width==o.width){band=max(band,scratch[input+other.index]-f32(max(o.width,other.width)));continue;}
  let stride=vec3u(1u,4u,16u);let layer=input+other.index-other.lane+select(3u,0u,side==1u)*stride[axis];
  for(var part=0u;part<16u;part++){band=max(band,scratch[layer+(part%4u)*stride[(axis+1u)%3u]+(part/4u)*stride[(axis+2u)%3u]]-f32(o.width));}
 }}}
 let value=select(0.0,max(0.0,band),umCapacity(o)>0.0);scratch[out+o.index]=value;
 if(value>0.0){scratch[SV_BAND+o.tile]=1.0;}
}
// Scales over the measured tiles, which hold every visited tile. A corner
// whose authority lies outside them has no band in any incident cell: every
// owner there writes the same zero. Nothing reads scales farther away.
// A measured unit-stencil fine tile stages the bands of its 5^3 cell block
// (its cells and the next layer up); its authority corners read only those.
var<workgroup> metricBand:array<f32,125>;
@compute @workgroup_size(64) fn metric(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let job=group.x+umDispatchX*group.y;let fineTile=umTopology[UM_TILES+min(job,UM_TILES-1u)];
 let staged=job<umCounts.x&&umTileMaximumWidth(fineTile)==1u&&svMeasured(fineTile);let block=umTileCoord(fineTile)*4u;
 if(staged){for(var i=l;i<125u;i+=64u){
  let c=block+umCorner(i,5u);var band=0.0;
  if(umCellPresent(vec3i(c))){let m=c%4u;band=scratch[SV_OWNERS+(umTopology[umTileAt(c/4u)]&0x3fffffffu)+m.x+4u*(m.y+4u*m.z)];}
  metricBand[i]=band;
 }}
 workgroupBarrier();
 let o=umResidentAllOwner(gid);if(o.width==0u||!svMeasured(o.tile)){return;}let unit=svUnit(o);
 // An owner's corner lies on a wall along an axis with no cell stored past
 // it: below the owner at a low corner, above it at a high one.
 var low=vec3<bool>(false);var high=vec3<bool>(false);
 {let origin=vec3i(umOrigin(o));low=umLowOpen(origin);for(var a=0u;a<3u;a++){high[a]=umAcrossKind(origin,a,i32(o.width))!=UM_RESIDENT;}}
 for(var k=0u;k<umCounts.w;k++){
  let p=umOrigin(o)+umCorner(k,2u)*o.width;
  if(unit){
   if(!svAuthority(o,k,low)){if(!svMeasured(svUnitAuthorityTile(p))){scratch[umVertexAt(p)]=0.0;}continue;}
  }else{
   let authority=umVertexAuthority(p);
   if(authority.index!=o.index){if(!svMeasured(authority.tile)){scratch[umVertexAt(p)]=0.0;}continue;}
  }
  var band=0.0;
  if(staged){
   // Block coordinates below zero lie outside the lattice (wall corners).
   let local=vec3i(p)-vec3i(block)-vec3i(1);
   for(var j=0u;j<umCounts.w;j++){let c=local+vec3i(umCorner(j,2u));if(all(c>=vec3i(0))){band=max(band,metricBand[u32(c.x+5*(c.y+5*c.z))]);}}
  }else{
   for(var j=0u;j<umCounts.w;j++){let c=umOwnerAt(vec3i(p)+vec3i(umCorner(j,2u))-vec3i(1));if(c.width!=0u){band=max(band,scratch[SV_OWNERS+c.index]);}}
  }
  if(band<=0.0){scratch[umVertexAt(p)]=0.0;continue;}
  var gradient=vec3f(0);
  ${this.solid?`// Native centred slopes need an open incident cell on both sides.
  var openLow=vec3<bool>(false);var openHigh=vec3<bool>(false);
  for(var j=0u;j<8u;j++){let corner=umCorner(j,2u);if(umCellOpen(vec3i(p)+vec3i(corner)-vec3i(1))>0.0){
   for(var a=0u;a<3u;a++){if(corner[a]==0u){openLow[a]=true;}else{openHigh[a]=true;}}}}`:""}
  // Both ends are width-aligned lattice vertices in tiles whose stencil holds
  // this owner: stored or resolved texels, so the samples are direct loads.
  // A wide owner's wall vertex takes the one-sided slope. Dropping the wall
  // axis (the native rule) leaves a floor vertex under a layer thinner than
  // one owner at the 0.1 floor: it moved a tenth as far as the rest, and the
  // global shift piled that layer's volume onto steep and tall parts.
  for(var axis=0u;axis<3u;axis++){
   if(o.width==1u&&select(low[axis],high[axis],((k>>axis)&1u)!=0u)){continue;}
   ${this.solid?"if(o.width==1u&&!(openLow[axis]&&openHigh[axis])){continue;}":""}
   let top=((k>>axis)&1u)!=0u;var lo=p;var hi=p;
   if(top||!low[axis]){lo[axis]=p[axis]-o.width;}if(!top||!high[axis]){hi[axis]=p[axis]+o.width;}
   // A wide slope is one-sided at a buried end: phi there is not state.
   ${this.solid?"if(o.width!=1u){if(umVertexBuried(vec3i(lo))){lo=p;}if(umVertexBuried(vec3i(hi))){hi=p;}if(lo[axis]==hi[axis]){continue;}}":""}
   gradient[axis]=(umLoadVertexW(hi,o.width)-umLoadVertexW(lo,o.width))/(f32(hi[axis]-lo[axis])*UM_H[axis]);
  }
  scratch[umVertexAt(p)]=band*0.2*max(0.1,length(gradient));
 }
}
// Completes the unstored scale texels of mixed-stencil tiles after metric,
// as UniformMixedPhiResolve does for phi (one group per seam tile, compiled
// authority and staged 4h lattice): reconstruction reads only stored scales
// and this writes only unstored ones.
var<workgroup> svResolveLattice:array<f32,27>;
@compute @workgroup_size(125) fn resolveScale(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let header=7u*UM_TILES+16u;let job=group.x+umDispatchX*group.y;
 let valid=job<umSupport[header]+umSupport[header+1u];
 let tile=select(0u,umSupport[header+4u+select(0u,job,valid)],valid);let base=vec3i(umTileCoord(tile));
 if(valid&&lane>=9u&&lane<35u){let v=(base+vec3i(umCorner(lane-8u,3u))-vec3i(1))*4;if(umVertexPresent(v)){svResolveLattice[lane-8u]=umScaleLoadVertex(vec3u(v));}}
 workgroupBarrier();
 if(!valid){return;}
${uniformCompiledVertexResolveWGSL("svResolveLattice","umScaleVertexWeight","umScaleVertexSum8")}let at2=umVertexAt(p);
 if(bitcast<u32>(value)!=bitcast<u32>(scratch[at2])){scratch[at2]=value;}
}
// Each lane writes its filled mass at the shift and one secant step ahead
// into its own workgroup row. A group with no visited tile
// keeps the constant partial its seed wrote; a converged solve skips the
// round. An unvisited lane's scales are zero: its texels may hold an earlier
// frame's scales or interpolate them.
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let o=umResidentAllOwner(gid);
 if(l==0u){atomicStore(&measureLive,0u);atomicStore(&measureBand,0u);}
 workgroupBarrier();
 if(o.width!=0u&&svVisited(o)&&scratch[(SV_S+3u)]==0.0){atomicOr(&measureBand,1u);}
 if(workgroupUniformLoad(&measureBand)==0u){return;}
 var row=vec4f(0);
 if(o.width!=0u){
  let origin=umOrigin(o);let capacity=umCapacity(o);let mass=f32(o.width*o.width*o.width)*capacity;
  var raw:array<f32,8>;var scale:array<f32,8>;var low=1e30;var high=-1e30;
  let shift=scratch[SV_S];let step=scratch[(SV_S+4u)];
  // Every corner is a stored (or resolved) vertex: eight direct loads.
  raw=umLoadVertexW8(origin,o.width);scale=umScaleLoadVertex8(origin,o.width);
  if(!svVisited(o)){scale=array<f32,8>();}
  for(var k=0u;k<8u;k++){low=min(low,raw[k]-(shift+step)*scale[k]);high=max(high,raw[k]-shift*scale[k]);}
  // An owner inside or outside the surface at both shifts has a constant
  // fraction; only cut owners evaluate the two shifted fills.
  if(high<0.0){row.x=mass;row.y=mass;}
  else if(low<0.0){
   for(var sample=0u;sample<2u;sample++){
    let at=shift+f32(sample)*step;var values:array<f32,8>;var negative=0u;
    for(var k=0u;k<8u;k++){values[k]=raw[k]-at*scale[k];negative+=select(0u,1u,values[k]<0.0);}
    // A cut 4h owner fills its open cells, not a share of its box.
    if(negative==8u){row[sample]=mass;}
    else if(negative!=0u){if(o.width==4u&&capacity<0.99999){row[sample]=svOpenFill(values,vec3i(umOrigin(o)));}else{row[sample]=fill(values)*mass;}}
   }
  }
 }
 sums[l]=row;
 storePartial(l,group);
}
// A fixed grid strides the chunks of 64 measure partials the GPU count
// holds; solve reads exactly those chunk sums.
var<workgroup> svReduceJobs:u32;
@compute @workgroup_size(64) fn reduce(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 // The job count lives in storage: made workgroup-uniform for the barriers.
 if(l==0u){svReduceJobs=svJobs();}
 let count=workgroupUniformLoad(&svReduceJobs);let chunks=(count+63u)/64u;
 for(var chunk=group.x;chunk<chunks;chunk+=groups.x){
  let i=chunk*64u+l;
  sums[l]=vec4f(0);if(i<count){sums[l]=loadSum(SV_P+4u*i);}sumGroup(l);
  if(l==0u){storeSum(SV_R+4u*chunk,sums[0]);}
  workgroupBarrier();
 }
}
// Filled volume never decreases with the shift. Secant Newton, clamped into
// the bracket the measured side narrows; converged once within tolerance or
// when no shift inside the limit can change the filled volume.
@compute @workgroup_size(64) fn solve(@builtin(local_invocation_index) l:u32){
 if(l==0u){svReduceJobs=svJobs();}
 let count=(workgroupUniformLoad(&svReduceJobs)+63u)/64u;
 var sum=vec4f(0);for(var i=l;i<count;i+=64u){sum+=loadSum(SV_R+4u*i);}sums[l]=sum;sumGroup(l);

 // Volume control (Kim et al. 2007, the proportional term): the divergence
 // that closes the liquid's gap to its budget over the control time. The
 // pressure authority asks it of every liquid owner, so the samples and the
 // surface they rebuild are carried to the budget together.
 if(gaugeBudget){
  if(l==0u){let filled=sums[0].x;let gap=budget[0]-filled;budget[6]=filled;budget[7]=budget[0];
   // The gauge is a step behind what it asks for: closing the gap in under
   // three steps would overshoot it.
   let gain=min(budgetParams.w,1.0/(3.0*max(budgetParams.x,1e-6)));
   let excess=sign(gap)*max(abs(gap)-${NB_VOLUME_DEAD_BAND}*budget[0],0.0);
   budget[${NB_VOLUME_RATE_FLOAT}u]=select(0.0,clamp(excess*gain/max(filled,0.5*budget[0]),-${NB_VOLUME_RATE_LIMIT},${NB_VOLUME_RATE_LIMIT}),budget[1]>0.0&&budget[0]>0.0);}
  return;
 }
 if(l==0u&&scratch[(SV_S+3u)]==0.0){
  let filled=sums[0].x;
  if(captureBudget){budget[0]=filled;budget[1]=1.0;budget[5]=filled;scratch[(SV_S+3u)]=1.0;return;}
  if(scratch[SV_S]==0.0){budget[4]=filled;}
  budget[6]=filled;budget[7]=budget[0];
  let desired=budget[0];let shift=scratch[SV_S];let step=scratch[(SV_S+4u)];
  let rise=sums[0].y-filled;
  if(abs(filled-desired)<=max(1e-5,1e-7*abs(desired))||rise*(2.0*umShiftLimit()/step)<=1e-6){scratch[(SV_S+3u)]=1.0;}
  else{
   var low=scratch[(SV_S+1u)];var high=scratch[(SV_S+2u)];
   if(filled<desired){low=shift;}else{high=shift;}
   scratch[(SV_S+1u)]=low;scratch[(SV_S+2u)]=high;scratch[SV_S]=clamp(shift+(desired-filled)*step/max(rise,1e-30),low,high);
  }
  budget[3]=scratch[SV_S];
 }
}
// Initial budget uses geometry alone, before any transport or metric scratch.
@compute @workgroup_size(64) fn measureInitial(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let o=umResidentAllOwner(gid);var amount=0.0;
 if(l==0u){atomicStore(&measureLive,0u);}workgroupBarrier();
 if(o.width!=0u){
  // The frame has just measured occupancy from the initialized surface.
  // NB binds that geometry output here, never transported cell mass.
  let p=vec3i(umOrigin(o));var occupied=0.0;
  if(o.width==4u){occupied=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,p,0).x;}
  else{occupied=textureLoad(volume,p,0).x;}
  amount=occupied*f32(o.width*o.width*o.width);
 }
 sums[l]=vec4f(amount,amount,0,0);storePartial(l,group);
}
// The step's explicit sources, in h-cell volume units.
@compute @workgroup_size(1) fn updateBudget(){budget[0]=max(0.0,budget[0]+budgetParams.y);budget[3]=0.0;}
`,["seed","dilate","metric","measure","grow","measureGrow","measureInitial","resolveScale"])});
  const errors=(await module.getCompilationInfo()).messages.filter(e=>e.type==="error");if(errors.length)throw new Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[])]});
  const constants={umDispatchX:this.ownership.dispatchX};
  const resident={umCountedJobs:UNIFORM_MIXED_COUNTED.residentAll},pages={umCountedJobs:UNIFORM_MIXED_COUNTED.residentPages};
  const create=(key:string,entryPoint:string,extra:Record<string,number>={})=>uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{...constants,...extra,...s}}})).then(p=>{this.pipelines.set(key,p);});
  await Promise.all([
   ...["clearBand","reduce","solve"].map(entry=>create(entry,entry)),
   create("capture","solve",{captureBudget:1}),create("gauge","solve",{gaugeBudget:1}),create("measureInitial","measureInitial",resident),create("updateBudget","updateBudget"),
   ...["seed","metric","measure"].map(entry=>create(entry,entry,resident)),
   create("resolveScale","resolveScale",{umCountedJobs:UNIFORM_MIXED_COUNTED.fused}),
   create("dilate0","dilate",{parity:0,...resident}),create("dilate1","dilate",{parity:1,...resident}),
   create("grow","grow",pages),create("measureGrow","measureGrow",pages),
  ]);
 }
 /** Measures the volume and the shift that would restore the budget; phi is
  * left alone. rounds: secant Newton rounds (measure,
  * reduce, solve); a converged solve skips the rest's work. */
 encode(encoder:GPUCommandEncoder,group:UniformDetailGroup,rounds=2,capture:boolean|"gauge"=false):void{
  if(!Number.isInteger(rounds)||rounds<1)throw new Error(`Mixed surface constraint needs a positive whole round count, not ${rounds}`);
  if(this.pipelines.size!==15)throw new Error("Mixed surface constraint is not initialized");
  const pass=encoder.beginComputePass(VOLUME_PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  if(capture){this.run(pass,"clearBand");this.run(pass,"measureInitial");this.run(pass,"reduce");this.run(pass,capture==="gauge"?"gauge":"capture");pass.end();return;}
  this.run(pass,"clearBand");this.run(pass,"seed");for(let i=0;i<4;i++)this.run(pass,i%2?"dilate1":"dilate0");this.run(pass,"grow");this.run(pass,"measureGrow");this.run(pass,"metric");this.run(pass,"resolveScale");
  for(let i=0;i<rounds;i++){this.run(pass,"measure");this.run(pass,"reduce");this.run(pass,"solve");}
  pass.end();
 }
 /** One entry's launch in encode()'s pass. */
 private run(pass:GPUComputePassEncoder,entry:string):void{
  const own=this.pipelines.get(entry)!;const pipeline=uniformDetailPick(this.solid?.select(own)??own);pass.setPipeline(pipeline);
  if(entry==="solve"||entry==="capture"||entry==="gauge")pass.dispatchWorkgroups(1);
  else if(entry==="reduce")this.ownership.dispatchCounted(pass,pipeline,this.chunks);
  else if(entry==="resolveScale")this.ownership.dispatchFusedCounted(pass,pipeline);
  else if(entry==="grow"||entry==="measureGrow")this.ownership.dispatchCounted(pass,pipeline,this.pages);
  else if(entry==="clearBand")pass.dispatchWorkgroups(this.clearGroups);
  else this.ownership.dispatchAllCounted(pass,pipeline);
 }
 /** Capture the initialized surface once; thereafter only explicit sources
  * change the target: liquid that rises through an open top is stored like
  * any other and stays in it. No CPU readback feeds correction. controlRate
  * (1/s, zero for none) gauges the step's opening occupancy against the
  * budget as it stood before this step's sources, and leaves the volume
  * control rate in the budget. */
 beginStep(encoder:GPUCommandEncoder,group:UniformDetailGroup,dt:number,addedCells:number,controlRate=0):void{
  if(!this.budgetCaptured){this.encode(encoder,group,1,true);this.budgetCaptured=true;}
  if(addedCells===0&&controlRate===0)return;
  const words=this.budgetWords;words[0]=dt;words[1]=addedCells;words[3]=controlRate;
  this.device.queue.writeBuffer(this.budgetParams,0,words);
  if(controlRate>0)this.encode(encoder,group,1,"gauge");
  if(addedCells===0)return;
  const pass=encoder.beginComputePass(BUDGET_PASS);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  this.launch(pass,"updateBudget",1);pass.end();
 }
 /** One fixed-width entry in beginStep()'s pass. */
 private launch(pass:GPUComputePassEncoder,entry:string,n:number):void{const own=this.pipelines.get(entry)!;pass.setPipeline(uniformDetailPick(this.solid?.select(own)??own));pass.dispatchWorkgroups(n);}
 destroy():void{this.budget.destroy();this.budgetParams.destroy();}

}
