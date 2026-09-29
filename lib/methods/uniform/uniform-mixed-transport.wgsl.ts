import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import { volumeNormalizeRowsWGSL, volumeNormalizeDonorsWGSL } from "./uniform-volume-normalization.wgsl";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { uniformMixedSolidWGSL } from "./uniform-mixed-solid.wgsl";

/** Native-texture conservative transport on mixed h/4h Uniform owners. With static
 * solids (group 2), unit rows carry native cut-cell capacity: edge weights by
 * min(open), open row targets and fallbacks, and sealed cells keep their V. */
export function uniformMixedTransportWGSL(layout: UniformMixedLayout,sources=false,solid=false): string {
  const [nx, ny, nz] = layout.lattice.dimensions;
  if (Math.max(nx, ny, nz) > 1020) throw new Error("Mixed transport packs row bases in 10 bits per axis: at most 1020 cells");
  const expressions = { count: "r.count", weight: "bitcast<f32>(edges[r.address+(1u+k)*r.stride])", storeWeight: (v: string) => `edges[r.address+(1u+k)*r.stride]=bitcast<u32>(tpQuantize(${v}));`, donor: "donorOf(r,k).index", target: "umRowCapacity(r)" };
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
const D=vec3u(${nx},${ny},${nz});const T=D/4u;
// Donor sums: exact 64-bit fixed point, multiples of 2^-TP_F, in two planar
// words (low, then high) per owner index. Every stored weight is first rounded
// up onto that grid (tpQuantize; only weights below 2^(23-TP_F) move), so a
// sum is exactly its weights in any order, and gather divides by that sum.
const TP_F:i32=${fraction};const TP_PLANE:u32=${cells}u;
// Sampled flags, a third plane: a row's own cell is sampled when some row
// has a positive raw weight on it (the fallback test).
const TP_FLAGS:u32=${2*cells}u;
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
 if(high!=0u){atomicAdd(&rigidExchange[TP_PLANE+i],bitcast<i32>(high));}
}
fn uvAddDonor(donor:u32,value:f32){tpAdd(donor,tpFixed(value));}
// Reads and clears a sum.
fn tpTake(i:u32)->f32{
 let low=bitcast<u32>(atomicExchange(&rigidExchange[i],0));let high=bitcast<u32>(atomicExchange(&rigidExchange[TP_PLANE+i],0));
 return ldexp(f32(high)*4294967296.0+f32(low),-TP_F);
}
${uniformMixedSolidWGSL(solid?2:undefined)}
${uniformMixedTransportLiveWGSL(sources)}
struct Row {tile:u32,lane:u32,width:u32,index:u32,address:u32,stride:u32,grain:u32,side:u32,count:u32}
fn rowAt(gid:vec3u)->Row {
 let o=tpOwner(gid);if(o.width==0u){return Row();}
 let t=o.tile;let lane=o.lane;let w=o.width;
 var grain=1u;
 if(w>1u){grain=((atomicLoad(&sampling[t])>>(2u*lane))&3u)+1u;}
 let side=w/grain+1u;let maxSide=w+1u;
 // Unit rows store their words planar across the tile (word k of every lane
 // adjacent); coarse rows keep theirs contiguous.
 let unit=w==1u;
 return Row(t,lane,w,(umTopology[t]&0x3fffffffu)+lane,t*640u+select(lane*(maxSide*maxSide*maxSide+2u),lane,unit),select(1u,64u,unit),grain,side,side*side*side+1u);
}
fn corner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn umRowOrigin(r:Row)->vec3u{return umTileCoord(r.tile)*4u+corner(r.lane,4u/r.width)*r.width;}
// uvOpen(id) for a unit row; coarse rows are uncut (promotion certificate).
fn umRowCapacity(r:Row)->f32{if(r.width==1u){return umCellOpen(vec3i(umRowOrigin(r)));}return f32(r.width*r.width*r.width);}
struct Donor {index:u32,capacity:f32,origin:vec3u}
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
fn donorOf(r:Row,k:u32)->Donor {return donorFrom(r,rowBase(r),k);}
fn buildAt(gid:vec3u){
 var r=rowAt(gid);if(r.width==0u){return;}
 let origin=umTileCoord(r.tile)*4u+corner(r.lane,4u/r.width)*r.width;
 let lower=textureLoad(departure,vec3i(origin),0).xyz-vec3f(0.5*f32(r.width));
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
   ${solid?"if(grain==1u){weight*=min(select(1.0,umCellOpen(vec3i(origin)),r.width==1u),umCellOpen(q));}":""}
  }
  edges[r.address+(1u+k)*r.stride]=bitcast<u32>(weight);
  if(bitcast<u32>(weight)!=0u){atomicStore(&rigidExchange[TP_FLAGS+donorAt(vec3u(q)*grain).index],1);}
 }
 edges[r.address+r.count*r.stride]=0u;
}
fn clearDonor(index:u32){atomicStore(&rigidExchange[index],0);atomicStore(&rigidExchange[TP_PLANE+index],0);atomicStore(&rigidExchange[TP_FLAGS+index],0);}
fn tpSampled(index:u32)->bool{return atomicLoad(&rigidExchange[TP_FLAGS+index])!=0;}
fn clearAt(gid:vec3u){
 let o=tpOwner(gid);if(o.width==0u){return;}clearDonor(o.index);
 if(o.lane==0u){atomicStore(&sampling[o.tile],0u);}
}
fn decodeAt(gid:vec3u){
 let o=tpOwner(gid);if(o.width==0u){return;}sums[o.index]=tpTake(o.index);
}
// The native fused schedule: fallback joins round zero; donor division joins
// the next row reader. Only the final gather consumes the last decoded sums.
fn normalizeRow(r:Row,divide:bool){
 if(r.count==9u){
  // A two-cell-per-axis row (every unit row, and coarse rows sampled at
  // their own width): the same three passes on register copies of its nine
  // weights and donors, instead of re-reading edges and re-resolving owners.
  var weights:array<f32,9>;var donors:array<u32,9>;var capacities:array<f32,9>;let base=rowBase(r);
  for(var k=0u;k<9u;k++){weights[k]=bitcast<f32>(edges[r.address+(1u+k)*r.stride]);let d=donorFrom(r,base,k);donors[k]=d.index;capacities[k]=d.capacity;}
  if(divide){for(var k=0u;k<9u;k++){weights[k]=weights[k]*capacities[k]/max(sums[donors[k]],1e-20);}}
  else if(!tpSampled(r.index)){weights[8]=${solid?"select(f32(r.width*r.width*r.width),max(umRowCapacity(r),1e-6),r.width==1u)":"f32(r.width*r.width*r.width)"};}
  var sum=0.0;for(var k=0u;k<9u;k++){sum+=weights[k];}
  let scale=umRowCapacity(r)/max(sum,1e-20);
  for(var k=0u;k<9u;k++){let weight=tpQuantize(weights[k]*scale);edges[r.address+(1u+k)*r.stride]=bitcast<u32>(weight);uvAddDonor(donors[k],weight);}
  return;
 }
 if(divide){${volumeNormalizeDonorsWGSL(expressions, "sums[donor]", "donorOf(r,k).capacity")}}
 else if(!tpSampled(r.index)){edges[r.address+r.count*r.stride]=bitcast<u32>(${solid?"select(f32(r.width*r.width*r.width),max(umRowCapacity(r),1e-6),r.width==1u)":"f32(r.width*r.width*r.width)"});}
 ${volumeNormalizeRowsWGSL(expressions)}
}
fn rowsFallbackAt(gid:vec3u){
 let r=rowAt(gid);if(r.width==0u){return;}normalizeRow(r,false);
}
fn rowsDivideAt(gid:vec3u){
 let r=rowAt(gid);if(r.width==0u){return;}normalizeRow(r,true);
}
// One coarse row per workgroup: distribute up to 125 donor overlaps,
// retaining the scalar row's summation order and exact integer donor sums.
var<workgroup> coarseWeights:array<f32,126>;
var<workgroup> coarseDonors:array<u32,126>;
var<workgroup> coarseScale:f32;
fn normalizeCoarseRow(job:u32,lane:u32,divide:bool){
 let r=rowAt(vec3u(job,0u,0u));let base=rowBase(r);
 for(var k=lane;k<126u;k+=64u){if(r.width!=0u&&k<r.count){
  let donor=donorFrom(r,base,k);var weight=bitcast<f32>(edges[r.address+(1u+k)*r.stride]);
  if(divide){weight=weight*donor.capacity/max(sums[donor.index],1e-20);}
  else if(k==r.count-1u&&!tpSampled(r.index)){weight=f32(r.width*r.width*r.width);}
  coarseWeights[k]=weight;coarseDonors[k]=donor.index;
 }}
 workgroupBarrier();
 if(lane==0u&&r.width!=0u){
  var sum=0.0;for(var k=0u;k<r.count;k++){sum+=coarseWeights[k];}
  coarseScale=umRowCapacity(r)/max(sum,1e-20);
 }
 workgroupBarrier();
 for(var k=lane;k<126u;k+=64u){if(r.width!=0u&&k<r.count){
  let weight=tpQuantize(coarseWeights[k]*coarseScale);
  edges[r.address+(1u+k)*r.stride]=bitcast<u32>(weight);uvAddDonor(coarseDonors[k],weight);
 }}
}
// Grid-stride over the listed coarse rows, one row per workgroup job.
var<workgroup> tpCoarseJobs:u32;
fn normalizeCoarseRows(group:u32,groups:u32,lane:u32,divide:bool){
 if(lane==0u){tpCoarseJobs=atomicLoad(&live[13u]);}
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
 else{if(mode==0u){atomicStore(&rigidExchange[TP_FLAGS+d.index],1);}tpAdd(d.index,tpFixed(weight));}
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
   if(mode==0u){atomicStore(&rigidExchange[TP_FLAGS+index],1);}tpAdd(index,x);
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
 if(lane==0u){tpFineJobs=atomicLoad(&live[12u]);}
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
 ${solid?`// A sealed cell's V is an unplaceable reservoir; preserve it.
 if(r.width==1u&&umRowCapacity(r)<=0.0){let p=vec3i(umRowOrigin(r));textureStore(output,p,vec4f(textureLoad(volume,p,0).x));return;}`:""}
 let base=rowBase(r);
 for(var k=0u;k<r.count;k++){let word=edges[r.address+(1u+k)*r.stride];let d=donorWord(r,base,k,word);
  let weight=bitcast<f32>(word)*d.capacity/max(sums[d.index],1e-20);
  value+=weight*textureLoad(volume,vec3i(d.origin),0).x;}
 let origin=umTileCoord(r.tile)*4u+corner(r.lane,4u/r.width)*r.width;
 var fill=value/f32(r.width*r.width*r.width);
 ${sources?`if(umSourceParams.drop.w>0.0||umSourceinflowStrength()>0.0){
  var added=0.0;
  for(var z=0u;z<r.width;z++){for(var y=0u;y<r.width;y++){for(var x=0u;x<r.width;x++){
   let q=vec3i(origin+vec3u(x,y,z));${solid?`let open=select(1.0,umCellOpen(q),r.width==1u);
   added+=min(umSourcedropSource(q),max(0.0,open-fill))+select(0.0,umSourceinflowSweptPlugSource(q,umSourceParams.dimsDt.w),open>0.0);`:`added+=min(umSourcedropSource(q),max(0.0,1.0-fill))+umSourceinflowSweptPlugSource(q,umSourceParams.dimsDt.w);`}
  }}}
  fill+=added/f32(r.width*r.width*r.width);
 }`:""}
 textureStore(output,vec3i(origin),vec4f(fill));
}
// Live-list entries: a fixed grid-stride grid over the listed owners (a
// lane per owner, tiers in separate launches); dense (list 0) otherwise.
@compute @workgroup_size(64) fn build(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(umTransportList==0u){buildAt(gid);return;}
 let owners=tpLiveOwners();for(var slot=gid.x;slot<owners;slot+=64u*groups.x){buildAt(vec3u(slot,0u,0u));}
}
@compute @workgroup_size(64) fn clear(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(umTransportList==0u){clearAt(gid);return;}
 let owners=tpLiveOwners();for(var slot=gid.x;slot<owners;slot+=64u*groups.x){clearAt(vec3u(slot,0u,0u));}
}
@compute @workgroup_size(64) fn decode(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(umTransportList==0u){decodeAt(gid);return;}
 let owners=tpLiveOwners();for(var slot=gid.x;slot<owners;slot+=64u*groups.x){decodeAt(vec3u(slot,0u,0u));}
}
@compute @workgroup_size(64) fn rowsFallback(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(umTransportList==0u){rowsFallbackAt(gid);return;}
 let owners=tpLiveOwners();for(var slot=gid.x;slot<owners;slot+=64u*groups.x){rowsFallbackAt(vec3u(slot,0u,0u));}
}
@compute @workgroup_size(64) fn rowsDivide(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(umTransportList==0u){rowsDivideAt(gid);return;}
 let owners=tpLiveOwners();for(var slot=gid.x;slot<owners;slot+=64u*groups.x){rowsDivideAt(vec3u(slot,0u,0u));}
}
@compute @workgroup_size(64) fn gather(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(umTransportList==0u){gatherAt(gid);return;}
 let owners=tpLiveOwners();for(var slot=gid.x;slot<owners;slot+=64u*groups.x){gatherAt(vec3u(slot,0u,0u));}
}
@compute @workgroup_size(64) fn copyVolume(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width!=0u){let p=vec3i(umOrigin(o));textureStore(output,p,vec4f(textureLoad(volume,p,0).x));}
}
@compute @workgroup_size(64) fn restrictVolume(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}let origin=umOrigin(o);var value=0.0;
 for(var z=0u;z<o.width;z++){for(var y=0u;y<o.width;y++){for(var x=0u;x<o.width;x++){
  value+=textureLoad(volume,vec3i(origin+vec3u(x,y,z)),0).x;
 }}}
 textureStore(output,vec3i(origin),vec4f(value/f32(o.width*o.width*o.width)));
}
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
 ${each((k) => `w${k}=w${k}*d${k}.capacity/max(sums[d${k}.index],1e-20);`)}`}
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
 * so each of those sums is the dense sum. Rows are S, Q1 and Q0; donors are
 * the rows and their donors. Any other tile samples no V and holds none, and
 * the dense transport leaves it at zero, as skipping it does.
 * Words: [0,12) unused, [12,16) the list counts (rows then donors; h then
 * 4h), [16,20) unused; then the set bits plane,
 * the donor box low and high planes, and four tier lists. */
function uniformMixedTransportLiveWGSL(sources:boolean):string{
 return /* wgsl */`
const TP_HEADER:u32=${UNIFORM_MIXED_TRANSPORT_LIVE_HEADER}u;
override umTransportList:u32=0u;
fn tpPlane(k:u32,t:u32)->u32{return TP_HEADER+k*UM_TILES+t;}
fn tpList(list:u32,tier:u32)->u32{return TP_HEADER+(3u+2u*(list-1u)+tier)*UM_TILES;}
fn tpTier(width:u32)->u32{return select(1u,0u,width==1u);}
// Owners of this launch's tier in its live list (rows 1, donors 2).
fn tpLiveOwners()->u32{
 let tier=tpTier(umCellWidth);return atomicLoad(&live[12u+2u*(umTransportList-1u)+tier])*(64u/(umCellWidth*umCellWidth*umCellWidth));
}
// Owner of a listed tile (rows 1, donors 2); the full tier lists otherwise.
fn tpOwner(gid:vec3u)->UMOwner {
 if(umTransportList==0u){return umOwner(gid);}
 let slot=gid.x+umDispatchX*64u*gid.y;let per=64u/(umCellWidth*umCellWidth*umCellWidth);let tier=tpTier(umCellWidth);
 let job=slot/per;if(job>=atomicLoad(&live[12u+2u*(umTransportList-1u)+tier])){return UMOwner();}
 let tile=atomicLoad(&live[tpList(umTransportList,tier)+job]);let lane=slot%per;
 return UMOwner(tile,lane,umCellWidth,(umTopology[tile]&0x3fffffffu)+lane);
}
// Set bits in plane 0.
const TP_S:u32=1u;const TP_R1:u32=2u;const TP_D2:u32=4u;const TP_Q2:u32=8u;
const TP_D1:u32=16u;const TP_Q1:u32=32u;const TP_Q0:u32=64u;const TP_DONOR:u32=128u;
var<workgroup> tpSeeded:atomic<u32>;
var<workgroup> tpLow:array<atomic<u32>,3>;
var<workgroup> tpHigh:array<atomic<u32>,3>;
// One workgroup per tile, a lane per owner: seed tiles holding V (or a source)
// and record the tiles this tile's rows can sample. A row's donors with
// nonzero weight are grain cells overlapping its box [lower, lower+width),
// and a grain divides 4, so they lie in tiles floor(lower/4) through
// floor((ceil(upper)-1)/4) at every grain. The fallback self edge adds the
// tile itself. Stored as offsets from the tile, biased by 512 in 10 bits.
@compute @workgroup_size(64) fn liveSeed(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=wid.x+umDispatchX*wid.y;if(tile>=UM_TILES){return;}
 if(lane<3u){atomicStore(&tpLow[lane],512u);atomicStore(&tpHigh[lane],512u);}
 if(lane==0u){atomicStore(&tpSeeded,0u);}workgroupBarrier();
 let width=umTileWidth(tile);let side=4u/width;let coord=vec3i(umTileCoord(tile));
 if(lane<side*side*side){
  let origin=umTileCoord(tile)*4u+umCorner(lane,side)*width;
  let lower=textureLoad(departure,vec3i(origin),0).xyz-vec3f(0.5*f32(width));let upper=lower+f32(width);
  // A non-finite departure reaches the whole lattice.
  let finite=all(abs(lower)<vec3f(1.0e8));
  let first=select(vec3i(0),clamp(vec3i(floor(lower/4.0)),vec3i(0),vec3i(UM_T)-1),finite);
  let last=select(vec3i(UM_T)-1,clamp((vec3i(ceil(upper))-1)/4,vec3i(0),vec3i(UM_T)-1),finite);
  for(var axis=0u;axis<3u;axis++){
   atomicMin(&tpLow[axis],u32(first[axis]-coord[axis]+512));atomicMax(&tpHigh[axis],u32(last[axis]-coord[axis]+512));
  }
  if(textureLoad(volume,vec3i(origin),0).x!=0.0){atomicOr(&tpSeeded,TP_S);}
 }
 ${sources?`if(umSourceParams.drop.w>0.0||umSourceinflowStrength()>0.0){
  let q=vec3i(umTileCoord(tile)*4u+umCorner(lane,4u));
  if(umSourcedropSource(q)>0.0||umSourceinflowSweptPlugSource(q,umSourceParams.dimsDt.w)>0.0){atomicOr(&tpSeeded,TP_S);}
 }`:""}
 workgroupBarrier();
 if(lane==0u){
  atomicStore(&live[tpPlane(0u,tile)],atomicLoad(&tpSeeded));
  atomicStore(&live[tpPlane(1u,tile)],atomicLoad(&tpLow[0])|(atomicLoad(&tpLow[1])<<10u)|(atomicLoad(&tpLow[2])<<20u));
  atomicStore(&live[tpPlane(2u,tile)],atomicLoad(&tpHigh[0])|(atomicLoad(&tpHigh[1])<<10u)|(atomicLoad(&tpHigh[2])<<20u));
 }
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
@compute @workgroup_size(64) fn liveCompact(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let bits=atomicLoad(&live[tpPlane(0u,tile)]);let tier=tpTier(umTileWidth(tile));
 let rows=(bits&(TP_S|TP_Q1|TP_Q0))!=0u;let donor=rows||(bits&TP_DONOR)!=0u;
 for(var list=1u;list<=2u;list++){
  if(select(donor,rows,list==1u)){let slot=atomicAdd(&live[12u+2u*(list-1u)+tier],1u);atomicStore(&live[tpList(list,tier)+slot],tile);}
 }
}
`;
}
