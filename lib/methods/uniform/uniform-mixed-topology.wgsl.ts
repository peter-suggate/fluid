import { uniformMidpointTraceWGSL } from "./uniform-midpoint-trace.wgsl";
import { uniformMixedVelocitySamplingWGSL } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedFacesWGSL } from "./uniform-mixed-faces.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";

/** Regular tiers with at most this many tiles join the fused interface launch. */
export const UNIFORM_MIXED_FUSED_REGULAR_TILES=64;
/** A regular 4h tier with at most this many owners (256 groups) is
 * launch-bound: they ride another launch as packed lanes (umRegularCoarseOwner)
 * instead of a width-specialized launch of their own. */
export const UNIFORM_MIXED_PACKED_REGULAR_OWNERS=16384;

/** One packed topology buffer: tile records, h/4h worklists, then frozen stencil masks.
 * Ownership and tracing share this ABI. Two tiers: 0 = h (width 1, 64 owners
 * per tile), 1 = 4h (width 4, one owner per tile). umCounts = (h tiles, 4h
 * tiles, 0, 8). A 64-lane group visits one h tile or 64 coarse tiles; coarse dispatch never
 * pays 63 idle lanes.
 */
export function uniformMixedTopologyWGSL(layout: UniformMixedLayout, group: number, prefix = ""): string {
  if (prefix && !/^[a-zA-Z][a-zA-Z0-9]*$/.test(prefix)) throw new Error("Invalid mixed topology namespace");
  const source = /* wgsl */ `
@group(${group}) @binding(0) var<storage,read> umTopology:array<u32>;
@group(${group}) @binding(1) var<uniform> umCounts:vec4u;
@group(${group}) @binding(2) var<storage,read_write> umSupport:array<u32>;
override umCellWidth:u32=1u;
override umDispatchX:u32=65535u;
override umRegularFine:bool=false;
override umPlannedFine:u32=0u;
override umInterfaceTiles:bool=false;
override umRegularTiles:bool=false;
override umMergedTiles:bool=false;
// With umMergedTiles, take the fused jobs (every seam tile, then small regular tiers) instead.
override umFusedJobs:bool=false;
const UM_D=vec3u(${layout.lattice.dimensions.map(n => `${n}u`).join(',')});const UM_T=UM_D/4u;
const UM_TILES:u32=${layout.tiles.length}u;
fn umTileWidth(t:u32)->u32{return select(4u,1u,(umTopology[t]&0x80000000u)!=0u);}
fn umTileSupport(t:u32)->u32{return umSupport[3u*UM_TILES+t];}
fn umTileStencil(t:u32)->vec2u{return vec2u(umTopology[2u*UM_TILES+2u*t],umTopology[2u*UM_TILES+2u*t+1u]);}
fn umTileMaximumWidth(t:u32)->u32{return umTileStencil(t).x>>27u;}
fn umTileMinimumWidth(t:u32)->u32{return umTileStencil(t).y>>27u;}
fn umTileAt(p:vec3u)->u32{return p.x+UM_T.x*(p.y+UM_T.y*p.z);}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%UM_T.x,(t/UM_T.x)%UM_T.y,t/(UM_T.x*UM_T.y));}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
struct UMOwner {tile:u32,lane:u32,width:u32,index:u32}
fn umAllOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;let fine=umCounts.x*64u;
 if(slot<fine){let tile=umTopology[UM_TILES+slot/64u];let lane=slot%64u;return UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);}
 let local=slot-fine;if(local>=umCounts.y){return UMOwner();}
 let tile=umTopology[UM_TILES+umCounts.x+local];return UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
}
// Regular 4h owner s: tiles whose 3x3x3 stencil has one width, listed by the
// ownership (support 8n+20: count, three zero words, then the list).
fn umRegularCoarseOwner(slot:u32)->UMOwner {
 let base=8u*UM_TILES+20u;if(slot>=umSupport[base]){return UMOwner();}
 let tile=umSupport[base+4u+slot];return UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
}
fn umOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;
 if(umMergedTiles){
  // 64 slots per merged tile job (see umTileJobOwner); then 64 regular
  // coarse owners per packed job.
  let tiles=umMergedTileJobs();
  if(!umFusedJobs&&slot/64u>=tiles){return umRegularCoarseOwner(slot-64u*tiles);}
  var owner=umMergedTileJob(slot/64u);let lane=slot%64u;
  if(owner.width==0u||lane>=64u/(owner.width*owner.width*owner.width)){return UMOwner();}
  owner.lane=lane;owner.index+=lane;return owner;
 }
 let cells=64u/(umCellWidth*umCellWidth*umCellWidth);
 let job=slot/cells;var offset=0u;var count=umCounts.x;
 if(umInterfaceTiles){
  let header=7u*UM_TILES+16u;var tier=0u;var start=0u;
  if(umCellWidth==4u){tier=1u;start=umSupport[header];}
  if(job>=umSupport[header+tier]){return UMOwner();}
  let tile=umSupport[header+4u+start+job];let lane=slot%cells;
  return UMOwner(tile,lane,umCellWidth,(umTopology[tile]&0x3fffffffu)+lane);
 }
 if(umCellWidth==4u){offset=umCounts.x;count=umCounts.y;}
 if(umCellWidth==1u&&umPlannedFine!=0u){
  count=umSupport[4u*UM_TILES+umPlannedFine];if(job>=count){return UMOwner();}
  let tile=umSupport[(4u+umPlannedFine)*UM_TILES+16u+job];let lane=slot%64u;
  return UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);
 }
 if(job>=count){return UMOwner();}
 let tile=umTopology[UM_TILES+offset+job];let lane=slot%cells;
 if(umRegularTiles&&umTileMaximumWidth(tile)!=umTileMinimumWidth(tile)){return UMOwner();}
 return UMOwner(tile,lane,umCellWidth,(umTopology[tile]&0x3fffffffu)+lane);
}
// One workgroup per interface tile of every tier, then every tile of each
// small regular tier (UNIFORM_MIXED_FUSED_REGULAR_TILES). The owner carries
// its runtime width, so a single launch replaces the tiny per-tier launches.
fn umFusedRegularTier(tier:u32)->bool{
 let count=umCounts[tier];return count>umSupport[7u*UM_TILES+16u+tier]&&count<=${UNIFORM_MIXED_FUSED_REGULAR_TILES}u;
}
// Not a small fused tier: see UNIFORM_MIXED_PACKED_REGULAR_OWNERS.
fn umPackedRegularTier(tier:u32)->bool{
 let owners=(umCounts[tier]-umSupport[7u*UM_TILES+16u+tier])*(64u>>(6u*tier));
 return !umFusedRegularTier(tier)&&owners<=${UNIFORM_MIXED_PACKED_REGULAR_OWNERS}u;
}
fn umFusedOwner(group:vec3u,lane:u32,regular:bool)->UMOwner {
 let header=7u*UM_TILES+16u;var job=group.x+umDispatchX*group.y;
 let seams=umSupport[header]+umSupport[header+1u];
 var tile=0u;
 if(job<seams){tile=umSupport[header+4u+job];}
 else{
  if(!regular){return UMOwner();}
  job-=seams;var offset=0u;var found=false;
  for(var tier=0u;tier<2u;tier++){
   let count=umCounts[tier];
   if(!found&&umFusedRegularTier(tier)){if(job<count){tile=umTopology[UM_TILES+offset+job];found=true;}else{job-=count;}}
   offset+=count;
  }
  if(!found||umTileMaximumWidth(tile)!=umTileMinimumWidth(tile)){return UMOwner();}
 }
 let width=umTileWidth(tile);let side=4u/width;if(lane>=side*side*side){return UMOwner();}
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
// One workgroup per tile. Merged launches take the certificate's general h
// list, then every seam 4h tile, so their serial latencies overlap instead
// of paying dependent launches; otherwise the umCellWidth tier.
// Jobs from umMergedTileJobs() on pack 64 regular coarse owners each
// (umRegularCoarseOwner), one lane per owner as regular fine work runs.
fn umMergedTileJobs()->u32 {
 return umSupport[4u*UM_TILES+2u]+umSupport[7u*UM_TILES+17u];
}
fn umMergedTileJob(index:u32)->UMOwner {
 if(umFusedJobs){return umFusedOwner(vec3u(index,0u,0u),0u,true);}
 var job=index;var tile=0u;var width=1u;let general=umSupport[4u*UM_TILES+2u];
 // No nested umSupport index: the frame plan rewrites these reads as atomics.
 let header=7u*UM_TILES+16u;let fine=umSupport[header];
 if(job<general){tile=umSupport[6u*UM_TILES+16u+job];}
 else{job-=general;width=4u;if(job>=umSupport[header+1u]){return UMOwner();}tile=umSupport[header+4u+fine+job];}
 return UMOwner(tile,0u,width,umTopology[tile]&0x3fffffffu);
}
fn umTileJobOwner(group:vec3u)->UMOwner {
 let job=group.x+umDispatchX*group.y;
 if(umMergedTiles){return umMergedTileJob(job);}
 let cells=64u/(umCellWidth*umCellWidth*umCellWidth);return umOwner(vec3u(job*cells,0u,0u));
}
fn umOrigin(o:UMOwner)->vec3u{return umTileCoord(o.tile)*4u+umCorner(o.lane,4u/o.width)*o.width;}
${uniformMixedFacesWGSL}
`;
  // Transfers read two ownership levels in one shader. Namespace their entire
  // ABI, retaining the same lookup implementation and buffer representation.
  return prefix ? source.replace(/\b(?:um[A-Z]\w*|UM_\w+|UMOwner|UMFace)\b/g, name => prefix + name) : source;
}

/** Appended after native storage specialization, for certified boxes
 * without interior solids. Canonical mixed MAC input is bound as transportIn without a halo.
 * RK2 is shared with native; interior-solid collision walking is unnecessary.
 */
export function uniformMixedNativeTraceWGSL(layout: UniformMixedLayout): string {
  return uniformMixedTopologyWGSL(layout, 1) + /* wgsl */ `
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32 {
 if(anchor[axis]<0){return boundaryVelocityIn[boundaryFaceIndex(max(anchor,vec3i(0)),axis)];}
 return textureLoad(transportIn,anchor,0)[axis];
}
${uniformMixedVelocitySamplingWGSL}
fn umTraceMixed(p:vec3f,dt:f32)->vec3f {
${uniformMidpointTraceWGSL("umSampleVelocity")}
 return end;
}
@compute @workgroup_size(64) fn uvMixedTrace(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}
 let origin=umOrigin(o);let p=vec3f(origin)+vec3f(0.5*f32(o.width));
 textureStore(velocityOut,vec3i(origin),vec4f(umTraceMixed(p,params.dimsDt.w),f32(o.width)));
}
`;
}
