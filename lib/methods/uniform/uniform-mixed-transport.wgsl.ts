import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { uniformMixedSolidWGSL } from "./uniform-mixed-solid.wgsl";
import { uniformMixedDustAccountingWGSL } from "./uniform-mixed-dust-accounting.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";

/** Native-texture conservative transport on mixed h/4h Uniform owners. With static
 * solids (group 2), rows carry cut-cell capacity: a unit row its h cell's open
 * fraction, a 4h row its tile's (`record`, the all-4h record's count; without
 * it every 4h row is uncut). Edge weights by the open part of each overlap,
 * open row and donor targets and fallbacks, and sealed rows keep their V. */
export function uniformMixedTransportWGSL(layout: UniformMixedLayout,sources=false,solid=false,resolved=false,record?:number): string {
  const [nx, ny, nz] = layout.lattice.dimensions;
  if (Math.max(nx, ny, nz) > 1020) throw new Error("Mixed transport packs row bases in 10 bits per axis: at most 1020 cells");
  // Donor sums hold every row weight of a round, at most one lattice cell per
  // cell: 2^range bits above the binary point, the rest below.
  const cells = nx * ny * nz, fraction = 64 - (Math.ceil(Math.log2(cells)) + 1);
  if (fraction < 24) throw new Error("Mixed transport donor sums need at least 24 fraction bits");
  // Unit-row weights are below 2, so fraction+1 bits in 22-bit window limbs.
  const limbs = Math.ceil((fraction + 1) / 22);
  return /* wgsl */ `
${uniformMixedTopologyWGSL(layout, 0)}
@group(1) @binding(0) var<storage,read_write> sampling:array<atomic<u32>>;
@group(1) @binding(1) var<storage,read_write> edges:array<u32>;
@group(1) @binding(2) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(1) @binding(3) var<storage,read_write> sums:array<f32>;
@group(1) @binding(4) var volume:texture_3d<f32>;
@group(1) @binding(5) var output:texture_storage_3d<r32float,write>;
@group(1) @binding(6) var departure:texture_3d<f32>;
${sources?uniformMixedSourceWGSL(7):""}
@group(1) @binding(8) var<storage,read_write> live:array<atomic<u32>>;
// The regular dust floor, UniformMixedCleanup's tuning ABI: strength,
// distance, regular floor, orphan floor; accounting words 5/6.
@group(1) @binding(9) var phi:texture_3d<f32>;
@group(1) @binding(10) var<uniform> tuning:array<vec4f,2>;
@group(1) @binding(11) var<storage,read_write> reductions:array<atomic<u32>>;
${uniformMixedDustAccountingWGSL(nx*ny*nz)}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// A tile corner: the base block (UNIFORM_DETAIL_4H_LOAD).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",resolved,undefined,"umLoadCorner")}
// A nonzero V below the floor is discarded unless it is positive with the
// surface inside the owner's band (the cleanup floor, pointwise per owner).
fn tpFloor(origin:vec3u,width:u32,value:f32)->f32{
 let floor=tuning[0].z;
 if(value!=0.0&&abs(value)<floor&&!(value>0.0&&umSampleVertex(vec3f(origin)+vec3f(0.5*f32(width)))<${4*Math.max(...layout.lattice.cellSize_m)}*f32(width))){
  umAccountDust(value,width*width*width,floor,5u);return 0.0;
 }
 return value;
}
const D=vec3u(${nx},${ny},${nz});const T=D/4u;
// Donor sums: exact 64-bit fixed point, multiples of 2^-TP_F, in two planar
// words (low, then high) per owner index. Every stored weight is first rounded
// up onto that grid (tpQuantize; only weights below 2^(23-TP_F) move), so a
// sum is exactly its weights in any order, and gather divides by that sum.
// Sampled flags, a third plane: a row's own cell is sampled when some row
// has a positive raw weight on it (the fallback test). A plane is one word
// per owner the capacity holds: the decoded sums' length.
const TP_F:i32=${fraction};
fn tpSumHigh(i:u32)->u32{return arrayLength(&sums)+i;}
fn tpFlag(i:u32)->u32{return 2u*arrayLength(&sums)+i;}
fn tpFixed(value:f32)->vec2u{
 let bits=bitcast<u32>(value);if(bits==0u){return vec2u(0u);}
 let exponent=bits>>23u;let mantissa=select(bits&0x7fffffu,(bits&0x7fffffu)|0x800000u,exponent!=0u);
 let shift=i32(max(exponent,1u))-150+TP_F;
 if(shift<0){let k=u32(-shift);if(k>=32u){return vec2u(1u,0u);}return vec2u((mantissa+(1u<<k)-1u)>>k,0u);}
 let s=u32(shift);
 if(s>=32u){return vec2u(0u,mantissa<<(s-32u));}
 return vec2u(mantissa<<s,select(0u,mantissa>>(32u-s),s!=0u));
}
fn tpQuantize(value:f32)->f32{let x=tpFixed(value);if(x.y!=0u||x.x>=0x800000u){return value;}return ldexp(f32(x.x),-TP_F);}
fn tpDelta(a:vec2u,b:vec2u)->vec2u{return vec2u(a.x-b.x,a.y-b.y-select(0u,1u,a.x<b.x));}
fn tpAdd(i:u32,x:vec2u){
 var high=x.y;
 if(x.x!=0u){let old=bitcast<u32>(atomicAdd(&rigidExchange[i],bitcast<i32>(x.x)));high+=select(0u,1u,old>0xffffffffu-x.x);}
 if(high!=0u){atomicAdd(&rigidExchange[tpSumHigh(i)],bitcast<i32>(high));}
}
fn uvAddDonor(donor:u32,value:f32){tpAdd(donor,tpFixed(value));}
// A weight's part of its donor's capacity. The donor's only sampler takes it
// whole, exactly: a row at rest keeps V bit for bit at any open fraction (a
// quotient rounds once the weight is not a power of two).
fn tpShare(weight:f32,capacity:f32,sum:f32)->f32{
 if(weight==sum&&weight!=0.0){return capacity;}
 return weight*capacity/max(sum,1e-20);
}
// Reads and clears a sum.
fn tpTake(i:u32)->f32{
 let low=bitcast<u32>(atomicExchange(&rigidExchange[i],0));let high=bitcast<u32>(atomicExchange(&rigidExchange[tpSumHigh(i)],0));
 return ldexp(f32(high)*4294967296.0+f32(low),-TP_F);
}
${uniformMixedSolidWGSL(solid?2:undefined,solid?record:undefined)}
${uniformMixedTransportLiveWGSL(sources)}
struct Row {tile:u32,lane:u32,width:u32,index:u32,address:u32,stride:u32,grain:u32,side:u32,count:u32}
fn rowAt(gid:vec3u)->Row {
 let o=tpOwner(gid);if(o.width==0u){return Row();}
 let t=o.tile;let lane=o.lane;let w=o.width;
 var grain=1u;
 if(w>1u){grain=((atomicLoad(&sampling[t])>>(2u*lane))&3u)+1u;}
 let side=w/grain+1u;
 // Rows by owner rank, h tiles first. An h tile's 64 unit rows store their
 // ten words planar across the tile (word k of every lane adjacent): 640
 // words from 10 * its first owner. A 4h row keeps its base, 125 donors and
 // self edge contiguous: 128 words a 4h owner, after every h tile's.
 let unit=w==1u;let first=o.index-lane;let fine=64u*umCounts.x;
 return Row(t,lane,w,o.index,select(10u*fine+128u*(first-fine),10u*first+lane,unit),select(1u,64u,unit),grain,side,side*side*side+1u);
}
fn corner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn umRowOrigin(r:Row)->vec3u{return umTileCoord(r.tile)*4u+corner(r.lane,4u/r.width)*r.width;}
// A 4h tile's open fraction as a row or donor holds it: sealed at <=1e-5.
fn tpTileOpen(t:u32)->f32{let open=umTileOpen(t);return select(0.0,open,open>1e-5);}
// The row's open fraction: uvOpen(id) for a unit row, the tile's mean for a 4h row.
fn umRowOpen(r:Row)->f32{if(r.width==1u){return umCellOpen(vec3i(umRowOrigin(r)));}return tpTileOpen(r.tile);}
// The open part of a box's overlap with a 4h tile: the overlap whole, none of
// a sealed tile, and a cut tile's by its h cells (the open overlap with a
// staircase is not a function of the tile's open fraction).
fn tpOpenOverlap(tile:vec3i,lower:vec3f,upper:vec3f,whole:f32)->f32{
 let open=umTileOpen(umTileAt(vec3u(tile)));
 if(open>=1.0||whole==0.0){return whole;}
 if(open<=1e-5){return 0.0;}
 let first=max(4*tile,vec3i(floor(lower)));let last=min(4*tile+3,vec3i(ceil(upper))-1);var sum=0.0;
 for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
  let c=vec3i(x,y,z);let lengths=max(vec3f(0),min(upper,vec3f(c)+1.0)-max(lower,vec3f(c)));
  sum+=lengths.x*lengths.y*lengths.z*umCellOpen(c);
 }}}
 return sum;
}
// The open volume the row is normalized to, in h cells.
fn umRowCapacity(r:Row)->f32{return f32(r.width*r.width*r.width)*umRowOpen(r);}
struct Donor {index:u32,capacity:f32,origin:vec3u}
// The volume a donor's samplers divide between them, in h cells: a 4h
// donor's less its tile's solid, the capacity its own row is normalized to
// (gather still shares V by the cell volume: V is a fraction of the cell).
fn tpDonorOpen(d:Donor)->f32{${solid?"if(d.capacity!=1.0){return d.capacity*tpTileOpen(umTileAt(d.origin/4u));}":""}return d.capacity;}
// The row's first donor cell in grain units, packed 10 bits per axis with a
// bias of 4: a partially intersecting 4h box sampled at h can start at -4.
// Fully out-of-domain rows have no positive donor weights; clamp their unused
// address so it cannot overflow into the next packed coordinate. Decoded once per row.
// Only nonzero-weight donors, all in the lattice, are resolved from it; u32
// wrap-around makes base+corner exact for them.
fn packRowBase(base:vec3i)->u32{let b=vec3u(clamp(base,vec3i(-4),vec3i(D)-vec3i(1))+vec3i(4));return b.x|(b.y<<10u)|(b.z<<20u);}
fn rowBase(r:Row)->vec3u{
 let word=edges[r.address];
 return (vec3u(word&1023u,(word>>10u)&1023u,word>>20u)-vec3u(4u))*r.grain;
}
// umOwnerAt and umOrigin for an in-lattice cell, by shifts: owners are
// aligned to their width inside their tile.
fn donorAt(p:vec3u)->Donor {
 let tile=umTileAt(p/4u);let word=umTopology[tile];
 let shift=select(2u,0u,(word&0x80000000u)!=0u);
 let local=(p%4u)>>vec3u(shift);let n=4u>>shift;
 return Donor((word&0x3fffffffu)+local.x+n*(local.y+n*local.z),f32(1u<<(3u*shift)),p&vec3u(~((1u<<shift)-1u)));
}
fn donorWord(r:Row,base:vec3u,k:u32,word:u32)->Donor {
 if(k==r.count-1u||word==0u){return Donor(r.index,f32(r.width*r.width*r.width),umRowOrigin(r));}
 return donorAt(base+corner(k,r.side)*r.grain);
}
fn donorFrom(r:Row,base:vec3u,k:u32)->Donor {return donorWord(r,base,k,edges[r.address+(1u+k)*r.stride]);}
fn buildAt(gid:vec3u){
 var r=rowAt(gid);if(r.width==0u){return;}
 let origin=umTileCoord(r.tile)*4u+corner(r.lane,4u/r.width)*r.width;
 // A 4h row's origin is its tile's: the base block.
 var departed=vec3f(0);if(r.width==4u){departed=${UNIFORM_DETAIL_4H_LOAD}textureLoad(departure,vec3i(origin),0).xyz;}else{departed=textureLoad(departure,vec3i(origin),0).xyz;}
 let lower=departed-vec3f(0.5*f32(r.width));
 let upper=lower+f32(r.width);var grain=r.width;
 let lo=max(vec3i(0),vec3i(floor(lower/4.0)));let hi=min(vec3i(T)-1,vec3i(ceil(upper/4.0))-1);
 // A width <=4 box touches at most two 4h tiles per axis, regardless of travel.
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
   grain=min(grain,umTileWidth(umTileAt(vec3u(vec3i(x,y,z)))));
 }}}
 if(r.width>1u){atomicOr(&sampling[r.tile],(grain-1u)<<(2u*r.lane));}
 r.grain=grain;r.side=r.width/grain+1u;r.count=r.side*r.side*r.side+1u;
 let base=vec3i(floor(lower/f32(grain)));let dims=vec3i(D/grain);
 edges[r.address]=packRowBase(base);
 for(var k=0u;k<r.count-1u;k++){
  let q=base+vec3i(corner(k,r.side));var weight=0.0;
  if(all(q>=vec3i(0))&&all(q<dims)){
   let a=vec3f(q)*f32(grain);let lengths=max(vec3f(0),min(upper,a+f32(grain))-max(lower,a));
   weight=lengths.x*lengths.y*lengths.z;
   ${solid?"// The open part of the overlap: an h cell's at grain 1 (min(open) with a unit\n   // row's own), a 4h tile's at grain 4.\n   if(grain==1u){weight*=min(select(1.0,umCellOpen(vec3i(origin)),r.width==1u),umCellOpen(q));}else{weight=tpOpenOverlap(q,lower,upper,weight);}":""}
  }
  edges[r.address+(1u+k)*r.stride]=bitcast<u32>(weight);
  if(bitcast<u32>(weight)!=0u){atomicStore(&rigidExchange[tpFlag(donorAt(vec3u(q)*grain).index)],1);}
 }
 edges[r.address+r.count*r.stride]=0u;
}
fn clearDonor(index:u32){atomicStore(&rigidExchange[index],0);atomicStore(&rigidExchange[tpSumHigh(index)],0);atomicStore(&rigidExchange[tpFlag(index)],0);}
fn tpSampled(index:u32)->bool{return atomicLoad(&rigidExchange[tpFlag(index)])!=0;}
fn decodeAt(gid:vec3u){
 let o=tpOwner(gid);if(o.width==0u){return;}sums[o.index]=tpTake(o.index);
}
// Four coarse rows per workgroup, sixteen lanes per row. The same dynamic
// donor loop covers regular 4h rows and 126-edge refinement-seam rows.
// Shared weights preserve the original float32 rounding boundary and each
// row's ascending summation order; donor accumulation remains exact.
var<workgroup> coarseWeights:array<f32,504>;
var<workgroup> coarseDonors:array<u32,504>;
var<workgroup> coarseScale:array<f32,4>;
fn normalizeCoarseRow(job:u32,lane:u32,divide:bool){
 let row=lane/16u;let local=lane%16u;let offset=row*126u;
 let r=rowAt(vec3u(4u*job+row,0u,0u));let base=rowBase(r);
 for(var k=local;k<126u;k+=16u){if(r.width!=0u&&k<r.count){
  let donor=donorFrom(r,base,k);var weight=bitcast<f32>(edges[r.address+(1u+k)*r.stride]);
  if(divide){weight=tpShare(weight,tpDonorOpen(donor),sums[donor.index]);}
  else if(k==r.count-1u&&!tpSampled(r.index)){weight=${solid?"max(umRowCapacity(r),1e-6)":"f32(r.width*r.width*r.width)"};}
  coarseWeights[offset+k]=weight;coarseDonors[offset+k]=donor.index;
 }}
 workgroupBarrier();
 if(local==0u&&r.width!=0u){
  var sum=0.0;for(var k=0u;k<r.count;k++){sum+=coarseWeights[offset+k];}
  coarseScale[row]=umRowCapacity(r)/max(sum,1e-20);
 }
 workgroupBarrier();
 for(var k=local;k<126u;k+=16u){if(r.width!=0u&&k<r.count){
  let weight=tpQuantize(coarseWeights[offset+k]*coarseScale[row]);
  edges[r.address+(1u+k)*r.stride]=bitcast<u32>(weight);uvAddDonor(coarseDonors[offset+k],weight);
 }}
}
var<workgroup> tpCoarseJobs:u32;
fn normalizeCoarseRows(group:u32,groups:u32,lane:u32,divide:bool){
 if(lane==0u){tpCoarseJobs=(atomicLoad(&live[tpCountWord(umTransportList,1u)])+3u)/4u;}
 let jobs=workgroupUniformLoad(&tpCoarseJobs);
 for(var job=group;job<jobs;job+=groups){normalizeCoarseRow(job,lane,divide);workgroupBarrier();}
}
@compute @workgroup_size(64) fn rowsFallbackCoarse(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){normalizeCoarseRows(group.x,groups.x,lane,false);}
@compute @workgroup_size(64) fn rowsDivideCoarse(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){normalizeCoarseRows(group.x,groups.x,lane,true);}
// Fine row tiles: one h tile of 64 unit rows per job. Their donors usually
// lie in a box of at most eight tiles; the job then sums them in workgroup
// memory (the same exact words) and adds each touched donor once. A wider
// box adds straight to the donors. build flags sampled donors the same way.
fn tpOverlap(q:vec3i,lower:vec3f,upper:vec3f,open:f32)->f32{
 if(any(q<vec3i(0))||any(q>=vec3i(D))){return 0.0;}
 let a=vec3f(q);let lengths=max(vec3f(0),min(upper,a+1.0)-max(lower,a));
 return lengths.x*lengths.y*lengths.z${solid?"*min(open,umCellOpen(q))":""};
}
// Window limbs: a word gets at most 512 adds per job (64 rows, 8 donor cells
// each in one 4h owner) below 2^22, so plain adds never carry.
const TP_LIMBS:u32=${limbs}u;
fn tpLimb(x:vec2u,i:u32)->u32{
 let s=22u*i;var v=x.x;
 if(s>=32u){v=x.y>>(s-32u);}else if(s!=0u){v=(x.x>>s)|(x.y<<(32u-s));}
 return v&0x3fffffu;
}
fn tpWide(w:u32,i:u32)->vec2u{
 let s=22u*i;if(s==0u){return vec2u(w,0u);}
 if(s>=32u){return vec2u(0u,w<<(s-32u));}return vec2u(w<<s,w>>(32u-s));
}
var<workgroup> tpWindow:array<atomic<u32>,${512*limbs}>;
var<workgroup> tpBoxLow:array<atomic<u32>,3>;
var<workgroup> tpBoxHigh:array<atomic<u32>,3>;
var<workgroup> tpFineJobs:u32;
fn tpContribute(d:Donor,weight:f32,fits:bool,boxLow:vec3u,extent:vec3u,mode:u32){
 if(fits){
  let t=d.origin/4u-boxLow;let cell=d.origin%4u;
  let word=64u*(t.x+extent.x*(t.y+extent.y*t.z))+cell.x+4u*(cell.y+4u*cell.z);
  let x=tpFixed(weight);for(var i=0u;i<TP_LIMBS;i++){let v=tpLimb(x,i);if(v!=0u){atomicAdd(&tpWindow[512u*i+word],v);}}
 }
 else{if(mode==0u){atomicStore(&rigidExchange[tpFlag(d.index)],1);}tpAdd(d.index,tpFixed(weight));}
}
// Each lane commits one cell of every box tile; a 4h tile's owner is cell 0.
fn tpCommit(lane:u32,fits:bool,boxLow:vec3u,extent:vec3u,mode:u32){
 if(fits){for(var slot=0u;slot<extent.x*extent.y*extent.z;slot++){
  let word=64u*slot+lane;var x=vec2u(0u);
  for(var i=0u;i<TP_LIMBS;i++){let w=tpWide(atomicLoad(&tpWindow[512u*i+word]),i);atomicStore(&tpWindow[512u*i+word],0u);
   let low=x.x+w.x;x=vec2u(low,x.y+w.y+select(0u,1u,low<w.x));}
  if(any(x!=vec2u(0u))){
   let tile=umTileAt(boxLow+vec3u(slot%extent.x,(slot/extent.x)%extent.y,slot/(extent.x*extent.y)));
   let index=(umTopology[tile]&0x3fffffffu)+lane;
   if(mode==0u){atomicStore(&rigidExchange[tpFlag(index)],1);}tpAdd(index,x);
  }
 }}
 if(lane<3u){atomicStore(&tpBoxLow[lane],0xffffffffu);atomicStore(&tpBoxHigh[lane],0u);}
}
${tpFineJobWGSL(0,solid)}${tpFineJobWGSL(2,solid)}
// The fallback round of fine rows: build normalized every row without its
// self edge and added it, exactly, to the round-zero sums. A row whose own
// cell no row sampled takes its self edge now and moves its other weights
// by exact differences.
fn rowsFallbackFineAt(gid:vec3u){
 let r=rowAt(gid);if(r.width==0u||tpSampled(r.index)){return;}
 let origin=umRowOrigin(r);
 let lower=textureLoad(departure,vec3i(origin),0).xyz-vec3f(0.5);let upper=lower+1.0;
 let base=vec3i(floor(lower));let open=${solid?"umCellOpen(vec3i(origin))":"1.0"};
 let own=${solid?"max(umRowCapacity(r),1e-6)":"1.0"};
 var sum=0.0;for(var k=0u;k<8u;k++){sum+=tpOverlap(base+vec3i(corner(k,2u)),lower,upper,open);}
 let scale=umRowCapacity(r)/max(sum+own,1e-20);
 for(var k=0u;k<8u;k++){
  let q=base+vec3i(corner(k,2u));let old=bitcast<f32>(edges[r.address+(1u+k)*64u]);
  let weight=tpQuantize(tpOverlap(q,lower,upper,open)*scale);
  if(weight!=old){edges[r.address+(1u+k)*64u]=bitcast<u32>(weight);tpAdd(donorAt(vec3u(q)).index,tpDelta(tpFixed(weight),tpFixed(old)));}
 }
 let weight=tpQuantize(own*scale);edges[r.address+576u]=bitcast<u32>(weight);tpAdd(r.index,tpFixed(weight));
}
@compute @workgroup_size(64) fn rowsFallbackFine(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let owners=tpLiveOwners();for(var slot=gid.x;slot<owners;slot+=64u*groups.x){rowsFallbackFineAt(vec3u(slot,0u,0u));}
}
fn tpFineRows(group:u32,groups:u32,lane:u32,mode:u32){
 for(var i=lane;i<512u*TP_LIMBS;i+=64u){atomicStore(&tpWindow[i],0u);}
 if(lane<3u){atomicStore(&tpBoxLow[lane],0xffffffffu);atomicStore(&tpBoxHigh[lane],0u);}
 if(lane==0u){tpFineJobs=atomicLoad(&live[tpCountWord(umTransportList,0u)]);}
 let jobs=workgroupUniformLoad(&tpFineJobs);
 for(var job=group;job<jobs;job+=groups){
  if(mode==0u){tpFineJob0(job,lane);}else{tpFineJob2(job,lane);}
  workgroupBarrier();
 }
}
@compute @workgroup_size(64) fn buildFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){tpFineRows(group.x,groups.x,lane,0u);}
@compute @workgroup_size(64) fn rowsDivideFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){tpFineRows(group.x,groups.x,lane,2u);}
fn gatherAt(gid:vec3u){
 let r=rowAt(gid);if(r.width==0u){return;}var value=0.0;
 ${solid?`// A sealed row's V is an unplaceable reservoir; preserve it.
 if(umRowCapacity(r)<=0.0){let p=umRowOrigin(r);textureStore(output,vec3i(p),vec4f(tpFloor(p,r.width,textureLoad(volume,vec3i(p),0).x)));return;}`:""}
 let base=rowBase(r);
 for(var k=0u;k<r.count;k++){let word=edges[r.address+(1u+k)*r.stride];let d=donorWord(r,base,k,word);
  // A 4h donor (capacity 64) is read at its tile origin: the base block.
  var donated=0.0;if(d.capacity!=1.0){donated=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,vec3i(d.origin),0).x;}else{donated=textureLoad(volume,vec3i(d.origin),0).x;}
  value+=tpShare(bitcast<f32>(word),d.capacity,sums[d.index])*donated;}
 let origin=umTileCoord(r.tile)*4u+corner(r.lane,4u/r.width)*r.width;
 var fill=value/f32(r.width*r.width*r.width);
 ${sources?`if(umSourceParams.drop.w>0.0||umSourceinflowStrength()>0.0){
  var added=0.0;
  for(var z=0u;z<r.width;z++){for(var y=0u;y<r.width;y++){for(var x=0u;x<r.width;x++){
   let q=vec3i(origin+vec3u(x,y,z));${solid?`// The drop fills the h cell's share of the row's room: open-fill for a unit row.
   let open=umCellOpen(q);var room=max(0.0,open-fill);if(r.width!=1u){room=open*max(0.0,1.0-fill*f32(r.width*r.width*r.width)/umRowCapacity(r));}
   added+=min(umSourcedropSource(q),room)+select(0.0,umSourceinflowSweptPlugSource(q,umSourceParams.dimsDt.w),open>0.0);`:`added+=min(umSourcedropSource(q),max(0.0,1.0-fill))+umSourceinflowSweptPlugSource(q,umSourceParams.dimsDt.w);`}
  }}}
  fill+=added/f32(r.width*r.width*r.width);
 }`:""}
 textureStore(output,vec3i(origin),vec4f(tpFloor(origin,r.width,fill)));
}
// Rows outside R1 keep V (the seeds stored it); R1 rows return to volume.
fn copyVolumeAt(gid:vec3u){
 let o=tpOwner(gid);if(o.width!=0u){let p=vec3i(umOrigin(o));var v=0.0;if(umCellWidth==4u){v=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,p,0).x;}else{v=textureLoad(volume,p,0).x;}textureStore(output,p,vec4f(v));}
}
// Live-list entries: a fixed grid-stride grid over the listed owners (a
// lane per owner, tiers in separate launches).
${["build","decode","gather","copyVolume"].map(entry=>`@compute @workgroup_size(64) fn ${entry}(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let owners=tpLiveOwners();for(var slot=gid.x;slot<owners;slot+=64u*groups.x){${entry}At(vec3u(slot,0u,0u));}
}`).join("\n")}
`;
}

/** One fine row tile job, unrolled over its nine edges so the weights and
 * donors stay in registers. mode 0: build, normalized without the self edge
 * (rowsFallbackFine adds it where needed), flags and round-zero sums; 2: division. */
function tpFineJobWGSL(mode: 0 | 2, solid: boolean): string {
  const ks = mode === 0 ? [0, 1, 2, 3, 4, 5, 6, 7] : [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const each = (f: (k: number) => string) => ks.map(f).join("\n ");
  const word = (k: number) => `edges[r.address+${(1 + k) * 64}u]`;
  const body = `${mode === 0 ? `let origin=umRowOrigin(r);
 let lower=textureLoad(departure,vec3i(origin),0).xyz-vec3f(0.5);let upper=lower+1.0;
 let base=vec3i(floor(lower));let open=${solid ? "umCellOpen(vec3i(origin))" : "1.0"};
 edges[r.address]=packRowBase(base);
 ${each((k) => `let q${k}=base+vec3i(${k & 1},${(k >> 1) & 1},${k >> 2});var w${k}=tpOverlap(q${k},lower,upper,open);var d${k}=Donor();if(w${k}!=0.0){d${k}=donorAt(vec3u(q${k}));}`)}
 ${word(8)}=0u;` : `let base=rowBase(r);
 ${each((k) => `let e${k}=${word(k)};let d${k}=donorWord(r,base,${k}u,e${k});var w${k}=bitcast<f32>(e${k});`)}
 ${each((k) => `w${k}=tpShare(w${k},tpDonorOpen(d${k}),sums[d${k}.index]);`)}`}
 let scale=umRowCapacity(r)/max(${ks.map((k) => `w${k}`).join("+")},1e-20);
 ${each((k) => `w${k}=tpQuantize(w${k}*scale);${word(k)}=bitcast<u32>(w${k});`)}`;
  return /* wgsl */ `
fn tpFineJob${mode}(job:u32,lane:u32){
 let r=rowAt(vec3u(job*64u+lane,0u,0u));
 ${body}
 var low=vec3u(0xffffffffu);var high=vec3u(0u);
 ${each((k) => `if(w${k}!=0.0){let t=d${k}.origin/4u;low=min(low,t);high=max(high,t);}`)}
 if(all(high>=low)){for(var a=0u;a<3u;a++){atomicMin(&tpBoxLow[a],low[a]);atomicMax(&tpBoxHigh[a],high[a]);}}
 workgroupBarrier();
 let boxLow=vec3u(atomicLoad(&tpBoxLow[0]),atomicLoad(&tpBoxLow[1]),atomicLoad(&tpBoxLow[2]));
 let boxHigh=vec3u(atomicLoad(&tpBoxHigh[0]),atomicLoad(&tpBoxHigh[1]),atomicLoad(&tpBoxHigh[2]));
 let extent=select(vec3u(9u),boxHigh+vec3u(1u)-boxLow,all(boxHigh>=boxLow));
 let fits=all(extent<=vec3u(8u))&&extent.x*extent.y*extent.z<=8u;
 ${each((k) => `if(w${k}!=0.0){tpContribute(d${k},w${k},fits,boxLow,extent,${mode}u);}`)}
 workgroupBarrier();
 tpCommit(lane,fits,boxLow,extent,${mode}u);
}`;
}

/** Header words of the transport live set (live buffer). */
export const UNIFORM_MIXED_TRANSPORT_LIVE_HEADER=20;
/** The transport live set, rebuilt every frame from the current ownership,
 * equal to the dense transport. One hop, row to donors, is bounded per tile by
 * the tiles that tile's rows sample, itself included (its donor box). Sets:
 *   A(X): donors of rows in X, the union of their donor boxes (scatter);
 *   B(Y): rows sampling Y, the tiles whose donor box meets Y (gather).
 * Gather gives a row V only from donors holding V (seed S, tiles holding V or
 * a source), so only rows R1 = B(S) can receive V. Their V needs, through the
 * rounds sampled flag, fallback and two divisions: third-round sums of S (rows
 * R1), second-round sums of A(R1) (rows Q2 = B(A(R1))), first-round sums of
 * A(Q2) (rows Q1 = B(A(Q2))), and the sampled flags of Q1 (rows Q0 = B(Q1)).
 * Every row sampling a donor whose sum is read lies in the set that reads it,
 * so each of those sums is the dense sum. Each round runs only the rows
 * whose values a later round reads: build (flags) Q0, the fallback round Q1,
 * the first division Q2, the second division and gather R1 (nested:
 * R1 in Q2 in Q1 in Q0, since every box holds its own tile). A round's
 * donor sums are decoded (and cleared) over the donors its rows add to:
 * every donor after build and fallback (build adds to A(Q0)), then
 * D1 = A(Q2), then D2 = A(R1). Rows outside R1 sample no V and hold none, so
 * the output there stays zero, as the dense gather writes it.
 * Words: [0,10) the counts of lists 3..7, [12,16) the counts of lists 1, 2
 * (h then 4h), [16,20) unused; then the set bits plane, the donor box low
 * and high planes, and the tier lists: 1 rows Q0, 2 donors (all), 3 rows Q1,
 * 4 rows Q2, 5 rows R1, 6 donors D1, 7 donors D2. A list entry is two words:
 * the tile and its owner base (topology word), so owners resolve in one hop. */
function uniformMixedTransportLiveWGSL(sources:boolean):string{
 return /* wgsl */`
const TP_HEADER:u32=${UNIFORM_MIXED_TRANSPORT_LIVE_HEADER}u;
override umTransportList:u32=0u;
fn tpPlane(k:u32,t:u32)->u32{return TP_HEADER+k*UM_TILES+t;}
fn tpList(list:u32,tier:u32)->u32{return TP_HEADER+(3u+4u*(list-1u)+2u*tier)*UM_TILES;}
fn tpCountWord(list:u32,tier:u32)->u32{return select(2u*(list-3u)+tier,12u+2u*(list-1u)+tier,list<=2u);}
fn tpTier(width:u32)->u32{return select(1u,0u,width==1u);}
// Owners of this launch's tier in its live list (rows 1, donors 2).
fn tpLiveOwners()->u32{
 let tier=tpTier(umCellWidth);return atomicLoad(&live[tpCountWord(umTransportList,tier)])*(64u/(umCellWidth*umCellWidth*umCellWidth));
}
// Owner of a listed tile.
fn tpOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;let per=64u/(umCellWidth*umCellWidth*umCellWidth);let tier=tpTier(umCellWidth);
 let job=slot/per;if(job>=atomicLoad(&live[tpCountWord(umTransportList,tier)])){return UMOwner();}
 let at=tpList(umTransportList,tier)+2u*job;let lane=slot%per;
 return UMOwner(atomicLoad(&live[at]),lane,umCellWidth,atomicLoad(&live[at+1u])+lane);
}
// Set bits in plane 0.
const TP_S:u32=1u;const TP_R1:u32=2u;const TP_D2:u32=4u;const TP_Q2:u32=8u;
const TP_D1:u32=16u;const TP_Q1:u32=32u;const TP_Q0:u32=64u;const TP_DONOR:u32=128u;
var<workgroup> tpSeeded:atomic<u32>;
var<workgroup> tpLow:array<atomic<u32>,3>;
var<workgroup> tpHigh:array<atomic<u32>,3>;
// Seed tiles holding V (or a source) and record the tiles this tile's rows
// can sample. A row's donors with nonzero weight are grain cells overlapping
// its box [lower, lower+width), and a grain divides 4, so they lie in tiles
// floor(lower/4) through floor((ceil(upper)-1)/4) at every grain. The
// fallback self edge adds the tile itself (the 512 start). Stored as offsets
// from the tile, biased by 512 in 10 bits.
fn tpOwnerReach(tile:u32,origin:vec3u,width:u32)->array<vec3u,2>{
 let coord=vec3i(umTileCoord(tile));
 // A 4h owner's origin is its tile's: the base block (width is a constant at both callers).
 var departed=vec3f(0);if(width==4u){departed=${UNIFORM_DETAIL_4H_LOAD}textureLoad(departure,vec3i(origin),0).xyz;}else{departed=textureLoad(departure,vec3i(origin),0).xyz;}
 let lower=departed-vec3f(0.5*f32(width));let upper=lower+f32(width);
 // A non-finite departure reaches the whole lattice.
 let finite=all(abs(lower)<vec3f(1.0e8));
 let first=select(vec3i(0),clamp(vec3i(floor(lower/4.0)),vec3i(0),vec3i(UM_T)-1),finite);
 let last=select(vec3i(UM_T)-1,clamp((vec3i(ceil(upper))-1)/4,vec3i(0),vec3i(UM_T)-1),finite);
 return array<vec3u,2>(vec3u(first-coord+512),vec3u(last-coord+512));
}
fn tpSourcesActive()->bool{return ${sources?"umSourceParams.drop.w>0.0||umSourceinflowStrength()>0.0":"false"};}
// Callers test tpSourcesActive first.
fn tpSourced(q:vec3i)->bool{
 ${sources?"return umSourcedropSource(q)>0.0||umSourceinflowSweptPlugSource(q,umSourceParams.dimsDt.w)>0.0;":"return false;"}
}
fn tpStoreSeed(tile:u32,seeded:u32,low:vec3u,high:vec3u){
 atomicStore(&live[tpPlane(0u,tile)],seeded);
 atomicStore(&live[tpPlane(1u,tile)],low.x|(low.y<<10u)|(low.z<<20u));
 atomicStore(&live[tpPlane(2u,tile)],high.x|(high.y<<10u)|(high.z<<20u));
}
// liveSeed: one workgroup per h tile, a lane per owner; liveSeedCoarse: one
// lane per 4h tile (its single owner, then its 64 cells for sources).
// Together they cover the h/4h partition once. They also clear every
// owner's donor sums and flags and every 4h tile's sampling widths, a
// superset of the donors (list 2) the rounds add to, and store every
// owner's V in output: gather overwrites the R1 rows there.
@compute @workgroup_size(64) fn liveSeed(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=wid.x+umDispatchX*wid.y;if(job>=umCounts.x){return;}
 let tile=umTopology[UM_TILES+job];clearDonor((umTopology[tile]&0x3fffffffu)+lane);
 if(lane<3u){atomicStore(&tpLow[lane],512u);atomicStore(&tpHigh[lane],512u);}
 if(lane==0u){atomicStore(&tpSeeded,0u);}workgroupBarrier();
 let origin=umTileCoord(tile)*4u+umCorner(lane,4u);
 let reach=tpOwnerReach(tile,origin,1u);
 for(var axis=0u;axis<3u;axis++){atomicMin(&tpLow[axis],reach[0][axis]);atomicMax(&tpHigh[axis],reach[1][axis]);}
 let v=textureLoad(volume,vec3i(origin),0).x;textureStore(output,vec3i(origin),vec4f(v));
 if(v!=0.0||(tpSourcesActive()&&tpSourced(vec3i(origin)))){atomicOr(&tpSeeded,TP_S);}
 workgroupBarrier();
 if(lane==0u){
  tpStoreSeed(tile,atomicLoad(&tpSeeded),vec3u(atomicLoad(&tpLow[0]),atomicLoad(&tpLow[1]),atomicLoad(&tpLow[2])),
   vec3u(atomicLoad(&tpHigh[0]),atomicLoad(&tpHigh[1]),atomicLoad(&tpHigh[2])));
 }
}
@compute @workgroup_size(64) fn liveSeedCoarse(@builtin(global_invocation_id) gid:vec3u){
 let job=gid.x+umDispatchX*64u*gid.y;if(job>=umCounts.y){return;}
 let tile=umTopology[UM_TILES+umCounts.x+job];let origin=umTileCoord(tile)*4u;
 clearDonor(umTopology[tile]&0x3fffffffu);atomicStore(&sampling[tile],0u);
 let reach=tpOwnerReach(tile,origin,4u);
 let v=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,vec3i(origin),0).x;textureStore(output,vec3i(origin),vec4f(v));var seeded=v!=0.0;
 if(!seeded&&tpSourcesActive()){for(var k=0u;k<64u&&!seeded;k++){seeded=tpSourced(vec3i(origin+umCorner(k,4u)));}}
 tpStoreSeed(tile,select(0u,TP_S,seeded),min(reach[0],vec3u(512u)),max(reach[1],vec3u(512u)));
}
override tpFrom:u32=0u;
override tpInto:u32=0u;
fn tpOffsets(word:u32)->vec3i{return vec3i(vec3u(word&1023u,(word>>10u)&1023u,word>>20u))-vec3i(512);}
fn tpBox(tile:u32)->array<vec3i,2>{
 let p=vec3i(umTileCoord(tile));
 return array<vec3i,2>(p+tpOffsets(atomicLoad(&live[tpPlane(1u,tile)])),p+tpOffsets(atomicLoad(&live[tpPlane(2u,tile)])));
}
// B: this tile's rows sample a tile in tpFrom.
@compute @workgroup_size(64) fn liveGather(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let box=tpBox(tile);
 for(var z=box[0].z;z<=box[1].z;z++){for(var y=box[0].y;y<=box[1].y;y++){for(var x=box[0].x;x<=box[1].x;x++){
  if((atomicLoad(&live[tpPlane(0u,umTileAt(vec3u(vec3i(x,y,z))))])&tpFrom)!=0u){atomicOr(&live[tpPlane(0u,tile)],tpInto);return;}
 }}}
}
// A: every tile this tpFrom tile's rows can sample.
@compute @workgroup_size(64) fn liveScatter(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 if((atomicLoad(&live[tpPlane(0u,tile)])&tpFrom)==0u){return;}
 let box=tpBox(tile);
 for(var z=box[0].z;z<=box[1].z;z++){for(var y=box[0].y;y<=box[1].y;y++){for(var x=box[0].x;x<=box[1].x;x++){
  let at=tpPlane(0u,umTileAt(vec3u(vec3i(x,y,z))));
  if((atomicLoad(&live[at])&tpInto)==0u){atomicOr(&live[at],tpInto);}
 }}}
}
// Membership of each list; the nested sets also admit their subsets' bits.
fn tpListed(bits:u32,list:u32)->bool{
 let rows1=TP_R1|TP_S;let rows2=rows1|TP_Q2;let rows3=rows2|TP_Q1;
 switch list {
  case 1u:{return (bits&(rows3|TP_Q0))!=0u;}
  case 2u:{return (bits&(rows3|TP_Q0|TP_DONOR))!=0u;}
  case 3u:{return (bits&rows3)!=0u;}
  case 4u:{return (bits&rows2)!=0u;}
  case 5u:{return (bits&rows1)!=0u;}
  case 6u:{return (bits&(rows2|TP_D2|TP_D1))!=0u;}
  default:{return (bits&(rows1|TP_D2))!=0u;}
 }
}
// A workgroup's tiles reserve each list's slots with one global add.
var<workgroup> tpCompactCount:array<atomic<u32>,14>;
var<workgroup> tpCompactBase:array<u32,14>;
@compute @workgroup_size(64) fn liveCompact(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=gid.x+umDispatchX*64u*gid.y;
 if(lane<14u){atomicStore(&tpCompactCount[lane],0u);}
 workgroupBarrier();
 var bits=0u;var tier=0u;var word=0u;var local:array<u32,7>;
 if(tile<UM_TILES){bits=atomicLoad(&live[tpPlane(0u,tile)]);word=umTopology[tile];tier=select(1u,0u,(word&0x80000000u)!=0u);}
 for(var list=1u;list<=7u;list++){if(bits!=0u&&tpListed(bits,list)){local[list-1u]=atomicAdd(&tpCompactCount[2u*(list-1u)+tier],1u);}}
 workgroupBarrier();
 if(lane<14u){let n=atomicLoad(&tpCompactCount[lane]);if(n!=0u){tpCompactBase[lane]=atomicAdd(&live[tpCountWord(lane/2u+1u,lane%2u)],n);}}
 workgroupBarrier();
 for(var list=1u;list<=7u;list++){if(bits!=0u&&tpListed(bits,list)){let at=tpList(list,tier)+2u*(tpCompactBase[2u*(list-1u)+tier]+local[list-1u]);atomicStore(&live[at],tile);atomicStore(&live[at+1u],word&0x3fffffffu);}}
}
`;
}
