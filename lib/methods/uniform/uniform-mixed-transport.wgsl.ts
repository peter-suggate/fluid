import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import { uniformVolumeDonorArithmeticWGSL } from "./uniform-volume-donor-sum.wgsl";
import { volumeNormalizeRowsWGSL, volumeNormalizeDonorsWGSL } from "./uniform-volume-normalization.wgsl";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { uniformMixedSolidWGSL } from "./uniform-mixed-solid.wgsl";

/** Native-texture conservative transport on graded Uniform owners. With static
 * solids (group 2), unit rows carry native cut-cell capacity: edge weights by
 * min(open), open row targets and fallbacks, and sealed cells keep their V. */
export function uniformMixedTransportWGSL(layout: UniformMixedLayout,sources=false,solid=false): string {
  const [nx, ny, nz] = layout.lattice.dimensions;
  if (Math.max(nx, ny, nz) > 1020) throw new Error("Mixed transport packs row bases in 10 bits per axis: at most 1020 cells");
  const expressions = { count: "r.count", weight: "bitcast<f32>(edges[r.address+1u+k])", storeWeight: (v: string) => `edges[r.address+1u+k]=bitcast<u32>(${v});`, donor: "donorOf(r,k).index", target: "umRowCapacity(r)" };
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
fn uvLimbBase(i:u32)->u32{return i;}
fn uvLimbPlane()->u32{return ${nx*ny*nz}u;}
${uniformVolumeDonorArithmeticWGSL}
${uniformMixedSolidWGSL(solid?2:undefined)}
${uniformMixedTransportLiveWGSL(sources)}
struct Row {tile:u32,lane:u32,width:u32,index:u32,address:u32,grain:u32,side:u32,count:u32}
fn rowAt(gid:vec3u)->Row {
 let o=tpOwner(gid);if(o.width==0u){return Row();}
 let t=o.tile;let lane=o.lane;let w=o.width;
 var grain=1u;
 if(w>1u){grain=((atomicLoad(&sampling[t])>>(2u*lane))&3u)+1u;}
 let side=w/grain+1u;let maxSide=w+1u;
 return Row(t,lane,w,(umTopology[t]&0x3fffffffu)+lane,t*640u+lane*(maxSide*maxSide*maxSide+2u),grain,side,side*side*side+1u);
}
fn corner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn umRowOrigin(r:Row)->vec3u{return umTileCoord(r.tile)*4u+corner(r.lane,4u/r.width)*r.width;}
// uvOpen(id) for a unit row; coarse rows are uncut (promotion certificate).
fn umRowCapacity(r:Row)->f32{if(r.width==1u){return umCellOpen(vec3i(umRowOrigin(r)));}return f32(r.width*r.width*r.width);}
struct Donor {index:u32,capacity:f32,origin:vec3u}
// The row's first donor cell in grain units, packed 10 bits per axis with a
// bias of 2: a box starting past a low wall has base >= -2 (the departure is
// clamped to the lattice and the box is at most 4 wide). Decoded once per row.
// Only nonzero-weight donors, all in the lattice, are resolved from it; u32
// wrap-around makes base+corner exact for them.
fn packRowBase(base:vec3i)->u32{let b=vec3u(base+vec3i(2));return b.x|(b.y<<10u)|(b.z<<20u);}
fn rowBase(r:Row)->vec3u{
 let word=edges[r.address];
 return (vec3u(word&1023u,(word>>10u)&1023u,word>>20u)-vec3u(2u))*r.grain;
}
// umOwnerAt and umOrigin for an in-lattice cell, by shifts: owners are
// aligned to their width inside their tile.
fn donorAt(p:vec3u)->Donor {
 let tile=umTileAt(p/4u);let word=umTopology[tile];
 let shift=select(select(2u,1u,(word&0x40000000u)!=0u),0u,(word&0x80000000u)!=0u);
 let local=(p%4u)>>vec3u(shift);let n=4u>>shift;
 return Donor((word&0x3fffffffu)+local.x+n*(local.y+n*local.z),f32(1u<<(3u*shift)),p&vec3u(~((1u<<shift)-1u)));
}
fn donorFrom(r:Row,base:vec3u,k:u32)->Donor {
 if(k==r.count-1u||edges[r.address+1u+k]==0u){return Donor(r.index,f32(r.width*r.width*r.width),umRowOrigin(r));}
 return donorAt(base+corner(k,r.side)*r.grain);
}
fn donorOf(r:Row,k:u32)->Donor {return donorFrom(r,rowBase(r),k);}
@compute @workgroup_size(64) fn build(@builtin(global_invocation_id) gid:vec3u){
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
  edges[r.address+1u+k]=bitcast<u32>(weight);
  // donorsampled: round zero reads these sums only through the fallback's
  // zero test, and a limb sum of nonnegative weights is zero exactly when no
  // positive weight arrived. A flag in limb 0 is that test; decodeSampled
  // reads and clears it.
  if(bitcast<u32>(weight)!=0u){atomicStore(&rigidExchange[uvLimbBase(donorAt(vec3u(q)*grain).index)],1);}
 }
 edges[r.address+r.count]=0u;
}
fn clearDonor(index:u32){for(var limb=0u;limb<6u;limb++){atomicStore(&rigidExchange[index+limb*uvLimbPlane()],0);}}
@compute @workgroup_size(64) fn clear(@builtin(global_invocation_id) gid:vec3u){
 let o=tpOwner(gid);if(o.width==0u){return;}clearDonor(o.index);
 if(o.lane==0u){atomicStore(&sampling[o.tile],0u);}
}
@compute @workgroup_size(64) fn decodeSampled(@builtin(global_invocation_id) gid:vec3u){
 let o=tpOwner(gid);if(o.width==0u){return;}let at=uvLimbBase(o.index);
 sums[o.index]=select(0.0,1.0,atomicLoad(&rigidExchange[at])!=0);atomicStore(&rigidExchange[at],0);
}
@compute @workgroup_size(64) fn decode(@builtin(global_invocation_id) gid:vec3u){
 let o=tpOwner(gid);if(o.width==0u){return;}sums[o.index]=uvDonorSum(o.index);clearDonor(o.index);
}
// The native fused schedule: fallback joins round zero; donor division joins
// the next row reader. Only the final gather consumes the last decoded sums.
fn normalizeRow(r:Row,divide:bool){
 if(r.count==9u){
  // A two-cell-per-axis row (every unit row, and coarse rows sampled at
  // their own width): the same three passes on register copies of its nine
  // weights and donors, instead of re-reading edges and re-resolving owners.
  var weights:array<f32,9>;var donors:array<u32,9>;var capacities:array<f32,9>;let base=rowBase(r);
  for(var k=0u;k<9u;k++){weights[k]=bitcast<f32>(edges[r.address+1u+k]);let d=donorFrom(r,base,k);donors[k]=d.index;capacities[k]=d.capacity;}
  if(divide){for(var k=0u;k<9u;k++){weights[k]=weights[k]*capacities[k]/max(sums[donors[k]],1e-20);}}
  else if(sums[r.index]==0.0){weights[8]=${solid?"select(f32(r.width*r.width*r.width),max(umRowCapacity(r),1e-6),r.width==1u)":"f32(r.width*r.width*r.width)"};}
  var sum=0.0;for(var k=0u;k<9u;k++){sum+=weights[k];}
  let scale=umRowCapacity(r)/max(sum,1e-20);
  for(var k=0u;k<9u;k++){let weight=weights[k]*scale;edges[r.address+1u+k]=bitcast<u32>(weight);uvAddDonor(donors[k],weight);}
  return;
 }
 if(divide){${volumeNormalizeDonorsWGSL(expressions, "sums[donor]", "donorOf(r,k).capacity")}}
 else if(sums[r.index]==0.0){edges[r.address+r.count]=bitcast<u32>(${solid?"select(f32(r.width*r.width*r.width),max(umRowCapacity(r),1e-6),r.width==1u)":"f32(r.width*r.width*r.width)"});}
 ${volumeNormalizeRowsWGSL(expressions)}
}
@compute @workgroup_size(64) fn rowsFallback(@builtin(global_invocation_id) gid:vec3u){
 let r=rowAt(gid);if(r.width==0u){return;}normalizeRow(r,false);
}
@compute @workgroup_size(64) fn rowsDivide(@builtin(global_invocation_id) gid:vec3u){
 let r=rowAt(gid);if(r.width==0u){return;}normalizeRow(r,true);
}
@compute @workgroup_size(64) fn gather(@builtin(global_invocation_id) gid:vec3u){
 let r=rowAt(gid);if(r.width==0u){return;}var value=0.0;
 ${solid?`// A sealed cell's V is an unplaceable reservoir; preserve it.
 if(r.width==1u&&umRowCapacity(r)<=0.0){let p=vec3i(umRowOrigin(r));textureStore(output,p,vec4f(textureLoad(volume,p,0).x));return;}`:""}
 let base=rowBase(r);
 for(var k=0u;k<r.count;k++){let d=donorFrom(r,base,k);
  let weight=bitcast<f32>(edges[r.address+1u+k])*d.capacity/max(sums[d.index],1e-20);
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

/** Header words of the transport live set (live buffer). */
export const UNIFORM_MIXED_TRANSPORT_LIVE_HEADER=32;
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
 * Words: [0,18) indirect args (rows then donors, per tier), [18,24) counts,
 * 24 unused; then the set bits plane, the donor box low and high planes, and
 * six tier lists. */
function uniformMixedTransportLiveWGSL(sources:boolean):string{
 return /* wgsl */`
const TP_HEADER:u32=${UNIFORM_MIXED_TRANSPORT_LIVE_HEADER}u;
override umTransportList:u32=0u;
fn tpPlane(k:u32,t:u32)->u32{return TP_HEADER+k*UM_TILES+t;}
fn tpList(list:u32,tier:u32)->u32{return TP_HEADER+(3u+3u*(list-1u)+tier)*UM_TILES;}
fn tpTier(width:u32)->u32{return select(select(2u,1u,width==2u),0u,width==1u);}
// Owner of a listed tile (rows 1, donors 2); the full tier lists otherwise.
fn tpOwner(gid:vec3u)->UMOwner {
 if(umTransportList==0u){return umOwner(gid);}
 let slot=gid.x+umDispatchX*64u*gid.y;let per=64u/(umCellWidth*umCellWidth*umCellWidth);let tier=tpTier(umCellWidth);
 let job=slot/per;if(job>=atomicLoad(&live[18u+3u*(umTransportList-1u)+tier])){return UMOwner();}
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
  if(select(donor,rows,list==1u)){let slot=atomicAdd(&live[18u+3u*(list-1u)+tier],1u);atomicStore(&live[tpList(list,tier)+slot],tile);}
 }
}
@compute @workgroup_size(1) fn livePublish(){
 for(var k=0u;k<6u;k++){
  let tier=k%3u;let owners=atomicLoad(&live[18u+k])*(64u>>(3u*tier));let groups=(owners+63u)/64u;
  atomicStore(&live[3u*k],min(groups,umDispatchX));atomicStore(&live[3u*k+1u],(groups+umDispatchX-1u)/umDispatchX);atomicStore(&live[3u*k+2u],1u);
 }
}
`;
}
