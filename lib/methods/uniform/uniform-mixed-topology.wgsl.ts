import { uniformMidpointTraceWGSL } from "./uniform-midpoint-trace.wgsl";
import { uniformMixedVelocitySamplingWGSL } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedFacesWGSL } from "./uniform-mixed-faces.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";

/** One packed topology buffer: tile records, h/2h/4h worklists, then frozen stencil masks.
 * Ownership and tracing share this ABI. A 64-lane group visits one h tile,
 * eight 2h tiles, or 64 coarse tiles; coarse dispatch never pays 63 idle lanes.
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
const UM_D=vec3u(${layout.lattice.dimensions.map(n => `${n}u`).join(',')});const UM_T=UM_D/4u;
const UM_TILES:u32=${layout.tiles.length}u;
fn umTileWidth(t:u32)->u32{let word=umTopology[t];if((word&0x80000000u)!=0u){return 1u;}if((word&0x40000000u)!=0u){return 2u;}return 4u;}
fn umTileSupport(t:u32)->u32{return umSupport[3u*UM_TILES+t];}
fn umTileStencil(t:u32)->vec2u{return vec2u(umTopology[2u*UM_TILES+2u*t],umTopology[2u*UM_TILES+2u*t+1u]);}
fn umTileMaximumWidth(t:u32)->u32{return umTileStencil(t).x>>27u;}
fn umTileMinimumWidth(t:u32)->u32{return umTileStencil(t).y>>27u;}
fn umTileAt(p:vec3u)->u32{return p.x+UM_T.x*(p.y+UM_T.y*p.z);}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%UM_T.x,(t/UM_T.x)%UM_T.y,t/(UM_T.x*UM_T.y));}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
struct UMOwner {tile:u32,lane:u32,width:u32,index:u32}
fn umAllOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;let fine=umCounts.x*64u;let middle=umCounts.y*8u;
 var local=slot;var width=1u;var offset=0u;
 if(slot>=fine){local=slot-fine;width=2u;offset=umCounts.x;}
 if(slot>=fine+middle){local=slot-fine-middle;width=4u;offset=umCounts.x+umCounts.y;}
 if(slot>=fine+middle+umCounts.z){return UMOwner();}
 let cells=64u/(width*width*width);let tile=umTopology[UM_TILES+offset+local/cells];let lane=local%cells;
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
fn umOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;let cells=64u/(umCellWidth*umCellWidth*umCellWidth);
 let job=slot/cells;var offset=0u;var count=umCounts.x;
 if(umInterfaceTiles){
  let header=7u*UM_TILES+16u;var tier=0u;var start=0u;
  if(umCellWidth>=2u){tier=1u;start=umSupport[header];}
  if(umCellWidth==4u){tier=2u;start+=umSupport[header+1u];}
  if(job>=umSupport[header+tier]){return UMOwner();}
  let tile=umSupport[header+4u+start+job];let lane=slot%cells;
  return UMOwner(tile,lane,umCellWidth,(umTopology[tile]&0x3fffffffu)+lane);
 }
 if(umCellWidth==2u){offset=umCounts.x;count=umCounts.y;}
 if(umCellWidth==4u){offset=umCounts.x+umCounts.y;count=umCounts.z;}
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
