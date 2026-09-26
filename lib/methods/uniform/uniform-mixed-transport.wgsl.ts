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
const D=vec3u(${nx},${ny},${nz});const T=D/4u;
fn uvLimbBase(i:u32)->u32{return i;}
fn uvLimbPlane()->u32{return ${nx*ny*nz}u;}
${uniformVolumeDonorArithmeticWGSL}
${uniformMixedSolidWGSL(solid?2:undefined)}
struct Row {tile:u32,lane:u32,width:u32,index:u32,address:u32,grain:u32,side:u32,count:u32}
fn rowAt(gid:vec3u)->Row {
 let o=umOwner(gid);if(o.width==0u){return Row();}
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
fn donorOf(r:Row,k:u32)->Donor {
 if(k==r.count-1u||edges[r.address+1u+k]==0u){return Donor(r.index,f32(r.width*r.width*r.width),umTileCoord(r.tile)*4u+corner(r.lane,4u/r.width)*r.width);}
 let o=corner(k,r.side);let dims=D/r.grain;
 let linear=edges[r.address]+o.x+dims.x*(o.y+dims.y*o.z);
 let p=vec3u(linear%dims.x,(linear/dims.x)%dims.y,linear/(dims.x*dims.y))*r.grain;
 let owner=umOwnerAt(vec3i(p));
 return Donor(owner.index,f32(owner.width*owner.width*owner.width),umOrigin(owner));
}
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
 edges[r.address]=u32(base.x+dims.x*(base.y+dims.y*base.z));
 for(var k=0u;k<r.count-1u;k++){
  let q=base+vec3i(corner(k,r.side));var weight=0.0;
  if(all(q>=vec3i(0))&&all(q<dims)){
   let a=vec3f(q)*f32(grain);let lengths=max(vec3f(0),min(upper,a+f32(grain))-max(lower,a));
   weight=lengths.x*lengths.y*lengths.z;
   ${solid?"if(grain==1u){weight*=min(select(1.0,umCellOpen(vec3i(origin)),r.width==1u),umCellOpen(q));}":""}
  }
  edges[r.address+1u+k]=bitcast<u32>(weight);uvAddDonor(donorOf(r,k).index,weight);
 }
 edges[r.address+r.count]=0u;
}
fn clearDonor(index:u32){for(var limb=0u;limb<6u;limb++){atomicStore(&rigidExchange[index+limb*uvLimbPlane()],0);}}
@compute @workgroup_size(64) fn clear(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}clearDonor(o.index);
 if(o.lane==0u){atomicStore(&sampling[o.tile],0u);}
}
@compute @workgroup_size(64) fn decode(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}sums[o.index]=uvDonorSum(o.index);clearDonor(o.index);
}
// The native fused schedule: fallback joins round zero; donor division joins
// the next row reader. Only the final gather consumes the last decoded sums.
fn normalizeRow(r:Row,divide:bool){
 if(r.count==9u){
  // A two-cell-per-axis row (every unit row, and coarse rows sampled at
  // their own width): the same three passes on register copies of its nine
  // weights and donors, instead of re-reading edges and re-resolving owners.
  var weights:array<f32,9>;var donors:array<u32,9>;var capacities:array<f32,9>;
  for(var k=0u;k<9u;k++){weights[k]=bitcast<f32>(edges[r.address+1u+k]);let d=donorOf(r,k);donors[k]=d.index;capacities[k]=d.capacity;}
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
 for(var k=0u;k<r.count;k++){let d=donorOf(r,k);
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
