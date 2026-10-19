import {NARROW_BAND_ACTIVITY_WORDS_PER_TILE} from "./uniform-narrow-band-activity.wgsl";
import { UNIFORM_DETAIL_CANONICAL_LOAD } from "../../core/uniform-detail-abi";
/** Retire stale far-field phi without changing vertices used by physical liquid.
 * Four tile planes: physical seeds, then a separable protected-region dilation.
 * The 16-cell collar covers the distance/volume readers' four-owner band.
 * The residency census still adds its full directional travel/stencil closure. */
export const NARROW_BAND_AIR_WORDS_PER_TILE=5;
export const NARROW_BAND_AIR_COLLAR_TILES=4;
/** The far-air gate plane (NB_AIR+4 tiles): nonzero where a seed lies within
 * the collar and one tile more. Outside it the next surface advect keeps
 * a vertex's far marker (UniformMixedSurface's farGate). */
export const NARROW_BAND_AIR_GATE_PLANE=4;
/** The state word clearAir sets where it changes a 4h-lattice vertex: the
 * only vertices a hanging texel's resolved value reads, so with it zero the
 * resolve after clearAir would write nothing (UniformMixedPhiResolve's
 * gated launch skips it). */
export const NARROW_BAND_AIR_CHANGED_WORD=15;
export const narrowBandAirWGSL=/* wgsl */`
const NB_AIR=NB_ACTIVITY_SURFACE_HEAT+${NARROW_BAND_ACTIVITY_WORDS_PER_TILE-6}u*UM_TILES;
override nbAirListed:bool=false;
override nbAirFused:bool=false;
override nbAirLanes:u32=64u;
// A trilinear field attains its extrema on a box at its corners. Restrict
// the box to a conservative superset of the cell's open part. In particular,
// negative corners buried under a heightfield are not physical liquid.
// Nine terrain samples include the corners of every bilinear patch where
// cell-centre patch boundaries cut the cell, so their minimum is a bound.
fn nbAirCellWet(c:vec3i,origin:vec3u,width:u32,values:array<f32,8>)->bool{
 if(umCellOpen(c)<=0.0){return false;}
 var bottom=f32(c.y);
 if(umSolidHasTerrain()){
  var bed=3.0e38;
  for(var z=0u;z<3u;z++){for(var x=0u;x<3u;x++){
   bed=min(bed,umTerrainSurface(vec3f(c)+vec3f(0.5*f32(x),0,0.5*f32(z))).x);
  }}
  bottom=max(bottom,bed);
  if(bottom>=f32(c.y+1)){return false;}
 }
 let low=(vec3f(f32(c.x),bottom,f32(c.z))-vec3f(origin))/f32(width);
 let high=(vec3f(c+1)-vec3f(origin))/f32(width);
 for(var k=0u;k<8u;k++){
  let q=select(low,high,umCorner(k,2u)!=vec3u(0u));
  if(nbTrilinear(values,q).w<=0.0){return true;}
 }
 return false;
}
fn nbAirTileWet(t:u32,first:u32,end:u32)->bool{
 // Includes ballistic samples and their reconstruction footprint. Never
 // require grid liquid as evidence for an unresolved particle's support.
 if(atomicLoad(&bins[NB_SURFACE_TILES+t])!=0u){return true;}
 let origin=4u*umTileCoord(t);let width=umTileWidth(t);
 var coarse:array<f32,8>;
 if(width==4u){
  coarse=umLoadCorner8(origin,4u);
  var negative=false;for(var k=0u;k<8u;k++){negative=negative||coarse[k]<=0.0;}
  if(!negative){return false;}
 }
 for(var lane=first;lane<end;lane++){
  let c=origin+umCorner(lane,4u);
  var values=coarse;var base=origin;
  if(width==1u){values=umLoadVertex8(c,1u);base=c;}
  var negative=false;for(var k=0u;k<8u;k++){negative=negative||values[k]<=0.0;}
  if(negative&&nbAirCellWet(vec3i(c),base,width,values)){return true;}
 }
 return false;
}
// A tile's seed: a COLD tile's phi is held, its seed stands, joined by any sample.
fn nbAirSeed(t:u32,first:u32,end:u32)->bool{
 if(nbTileCold(t)){return atomicLoad(&bins[NB_AIR+t])!=0u||atomicLoad(&bins[NB_SURFACE_TILES+t])!=0u;}
 return nbAirTileWet(t,first,end);
}
// 4h tiles (and absent ones) a lane each; h tiles are airSeedsFine's.
@compute @workgroup_size(64) fn airSeeds(@builtin(global_invocation_id) id:vec3u){
 for(var t=id.x;t<UM_TILES;t+=65536u){
  var wet=false;
  if(umTileLive(t)){if(umTileWidth(t)==1u){continue;}wet=nbAirSeed(t,0u,64u);}
  atomicStore(&bins[NB_AIR+t],u32(wet));
 }
}
// The h tiles (the ownership's h list), a workgroup a tile and a lane a
// cell: an h tile's 64 cells are 512 vertex loads, which held a lane per
// tile behind ~64 serial evaluations.
var<workgroup> nbAirFine:u32;var<workgroup> nbAirAny:atomic<u32>;
@compute @workgroup_size(64) fn airSeedsFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){nbAirFine=umCounts.x;}
 let count=workgroupUniformLoad(&nbAirFine);
 for(var j=group.x;j<count;j+=groups.x){
  let t=umTopology[UM_TILES+j];
  if(lane==0u){atomicStore(&nbAirAny,0u);}
  workgroupBarrier();
  if(nbAirSeed(t,lane,lane+1u)){atomicStore(&nbAirAny,1u);}
  workgroupBarrier();
  if(lane==0u){atomicStore(&bins[NB_AIR+t],atomicLoad(&nbAirAny));}
 }
}
// One shared-plan launch covers coarse regions and fine owners. They write
// disjoint seed words; no inter-dispatch dependency separates the two widths.
var<workgroup> nbAirSeedRegions:u32;
var<workgroup> nbAirSeedFine:u32;
var<workgroup> nbAirSeedRegion:u32;
@compute @workgroup_size(64) fn airPlanSeeds(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(nbAirFused&&group.x==0u&&lane==0u){atomicStore(&state[15u],0u);}
 if(lane==0u){nbAirSeedRegions=atomicLoad(&bins[NB_FRAME_PLAN+4u+6u*NB_REUSE_REGIONS]);nbAirSeedFine=umCounts.x;}
 let regions=workgroupUniformLoad(&nbAirSeedRegions);let fine=workgroupUniformLoad(&nbAirSeedFine);
 for(var job=group.x;job<regions+fine;job+=groups.x){
  if(job<regions){
   if(lane==0u){nbAirSeedRegion=atomicLoad(&bins[NB_FRAME_PLAN+5u+6u*NB_REUSE_REGIONS+job]);}
   let region=workgroupUniformLoad(&nbAirSeedRegion);let t=nbReuseTile(region,lane);
   if(umTileWidth(t)!=1u){atomicStore(&bins[NB_AIR+t],u32(nbAirSeed(t,0u,64u)));}
  }else{
   let t=umTopology[UM_TILES+job-regions];
   if(lane==0u){atomicStore(&nbAirAny,0u);}workgroupBarrier();
   if(nbAirSeed(t,lane,lane+1u)){atomicStore(&nbAirAny,1u);}workgroupBarrier();
   if(lane==0u){atomicStore(&bins[NB_AIR+t],atomicLoad(&nbAirAny));}
  }
  workgroupBarrier();
 }
}
${[0,1,2].map(axis=>/* wgsl */`
@compute @workgroup_size(64) fn airSpread${axis}(@builtin(global_invocation_id) id:vec3u){
 for(var t=id.x;t<UM_TILES;t+=65536u){
  var keep=0u;
  if(umTileLive(t)){
   let p=vec3i(umTileCoord(t));
   for(var d=-${NARROW_BAND_AIR_COLLAR_TILES};d<=${NARROW_BAND_AIR_COLLAR_TILES};d++){
    var q=p;q[${axis}]+=d;let key=umTileKey(q);
    if(umKeyStored(key)){keep|=atomicLoad(&bins[NB_AIR+${axis}u*UM_TILES+key]);}
   }
  }
  atomicStore(&bins[NB_AIR+${axis+1}u*UM_TILES+t],keep);
 }
}`).join("\n")}
// The far-air gate: the collar (plane 3) dilated by one tile. A tile it
// leaves out is a tile beyond the collar's edge: the liquid's next step
// cannot bring the collar to it, so this step's clear would hold every
// value its advection could take at the far marker or above.
@compute @workgroup_size(64) fn airGate(@builtin(global_invocation_id) id:vec3u){
 for(var t=id.x;t<UM_TILES;t+=65536u){
  var near=0u;
  if(umTileLive(t)){
   let p=vec3i(umTileCoord(t));
   for(var k=0u;k<27u;k++){
    let key=umTileKey(p+vec3i(umCorner(k,3u))-vec3i(1));
    if(umKeyStored(key)){near|=atomicLoad(&bins[NB_AIR+3u*UM_TILES+key]);}
   }
  }
  atomicStore(&bins[NB_AIR+${4}u*UM_TILES+t],near);
 }
}
// One workgroup computes all three separable dilations and the gate for
// a region. Intermediate rows live in workgroup memory, not three pool-wide
// planes. The live masks reproduce the old sparse-hole semantics at EVERY
// intermediate axis (a plain box dilation would cross absent pages).
var<workgroup> nbAirRegionBases:array<u32,125>;
var<workgroup> nbAirX:array<u32,196>;
var<workgroup> nbAirLive:array<u32,196>;
var<workgroup> nbAirY:array<u32,84>;
var<workgroup> nbAirZ:array<u32,36>;
var<workgroup> nbAirPlanCount:u32;
var<workgroup> nbAirPlanRegion:u32;
@compute @workgroup_size(nbAirLanes) fn airPlanPropagate(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){nbAirPlanCount=atomicLoad(&bins[NB_FRAME_PLAN+3u]);}
 let count=workgroupUniformLoad(&nbAirPlanCount);
 for(var job=group.x;job<count;job+=groups.x){
  if(lane==0u){nbAirPlanRegion=atomicLoad(&bins[NB_FRAME_PLAN+4u+5u*NB_REUSE_REGIONS+job]);}
  let region=workgroupUniformLoad(&nbAirPlanRegion);let world=vec3i(umPageCoord(region));
  // Compile the 5³ region address closure once. The 14³ tile gathers below
  // use physical tile arithmetic, avoiding 2,744 page-table searches.
  for(var k=lane;k<125u;k+=nbAirLanes){
   let key=umTileKey(4*(world+vec3i(umCorner(k,5u))-2));var base=0xffffffffu;
   if(umKeyStored(key)){base=umPageOfTile(key);}nbAirRegionBases[k]=base;
  }
  workgroupBarrier();
  for(var row=lane;row<196u;row+=nbAirLanes){
   let yz=vec2i(i32(row%14u)-5,i32(row/14u)-5);var seeds=0u;var live=0u;
   for(var x=0u;x<14u;x++){
    let p=vec3i(i32(x)-5,yz);let r=vec3u((p>>vec3u(2u))+2);
    let base=nbAirRegionBases[r.x+5u*(r.y+5u*r.z)];
    if(base!=0xffffffffu){
     let local=vec3u(p)&vec3u(3u);let key=nbReuseTile(base,local.x+4u*(local.y+4u*local.z));
     live|=1u<<x;if(atomicLoad(&bins[NB_AIR+key])!=0u){seeds|=1u<<x;}
    }
   }
   var spread=0u;for(var x=0u;x<6u;x++){if(((seeds>>x)&511u)!=0u){spread|=1u<<x;}}
   nbAirX[row]=spread&((live>>4u)&63u);nbAirLive[row]=live;
  }
  workgroupBarrier();
  for(var row=lane;row<84u;row+=nbAirLanes){
   let y=row%6u;let z=row/6u;var spread=0u;
   for(var k=0u;k<9u;k++){spread|=nbAirX[y+k+14u*z];}
   nbAirY[row]=spread&((nbAirLive[y+4u+14u*z]>>4u)&63u);
  }
  workgroupBarrier();
  if(lane<36u){
   let y=lane%6u;let z=lane/6u;var spread=0u;
   for(var k=0u;k<9u;k++){spread|=nbAirY[y+6u*(z+k)];}
   nbAirZ[lane]=spread&((nbAirLive[y+4u+14u*(z+4u)]>>4u)&63u);
  }
  workgroupBarrier();
  if(lane<64u){
  let local=umCorner(lane,4u);let tile=nbReuseTile(region,lane);
  atomicStore(&bins[NB_AIR+3u*UM_TILES+tile],(nbAirZ[local.y+1u+6u*(local.z+1u)]>>(local.x+1u))&1u);
  var near=0u;
  for(var z=0u;z<3u;z++){for(var y=0u;y<3u;y++){near|=nbAirZ[local.y+y+6u*(local.z+z)];}}
  atomicStore(&bins[NB_AIR+4u*UM_TILES+tile],u32(((near>>local.x)&7u)!=0u));
  }
  workgroupBarrier();
  if(nbAirFused){nbAirClearRegion(region,world,lane);}
 }
}
var<workgroup> nbAirOrigin:vec3i;
var<workgroup> nbAirClearTiles:array<vec2u,64>;
var<workgroup> nbAirClearFine:atomic<u32>;
var<workgroup> nbAirClearCoarse:atomic<u32>;
fn nbAirClearRegion(region:u32,world:vec3i,lane:u32){
 if(lane==0u){nbAirOrigin=4*world;atomicStore(&nbAirClearFine,0u);atomicStore(&nbAirClearCoarse,0u);}workgroupBarrier();
 // The same seed-region list that governed coarse clear; fine owners are
 // necessarily requests. No vertex outside the old write domain is added.
 let selected=(atomicLoad(&bins[NB_FRAME_PLAN+4u+4u*NB_REUSE_REGIONS+region])&8u)!=0u||(atomicLoad(&bins[NB_FRAME_PLAN+4u+region])&1u)!=0u;
 if(lane<64u&&selected){
  let t=nbReuseTile(region,lane);let width=umTileWidth(t);
  if(!nbTileCold(t)&&atomicLoad(&bins[NB_AIR+t])==0u){
   let regular=umTileMinimumWidth(t)==width&&umTileMaximumWidth(t)==width;
   var low=vec3<bool>(false);if(regular){low=umLowOpen(vec3i(4u*umTileCoord(t)));}
   let flags=u32(regular)|(u32(low.x)<<1u)|(u32(low.y)<<2u)|(u32(low.z)<<3u);
   var slot=0u;if(width==1u){slot=atomicAdd(&nbAirClearFine,1u);}else{slot=63u-atomicAdd(&nbAirClearCoarse,1u);}
   nbAirClearTiles[slot]=vec2u(t,flags);
  }
 }
 workgroupBarrier();
 let fine=125u*atomicLoad(&nbAirClearFine);let jobs=fine+8u*atomicLoad(&nbAirClearCoarse);
 for(var job=lane;job<jobs;job+=nbAirLanes){
  let isFine=job<fine;var slot=63u-(job-fine)/8u;var corner=umCorner((job-fine)%8u,2u);var width=4u;
  if(isFine){slot=job/125u;corner=umCorner(job%125u,5u);width=1u;}
  let entry=nbAirClearTiles[slot];let t=entry.x;let p=4u*umTileCoord(t)+width*corner;
  if((entry.y&1u)!=0u){let low=(vec3u(entry.y)>>vec3u(1u,2u,3u))&vec3u(1u);if(!all((corner!=vec3u(0u))|(low!=vec3u(0u)))){continue;}}
  else{let a=umVertexAuthority(p);if(a.width!=width||a.tile!=t){continue;}}
  nbAirStore(p,width==4u);
 }
 workgroupBarrier();
}
fn nbAirProtected(p:vec3u,plane:u32)->bool{
 // Shared canonical vertices belong to every incident tile, even when a
 // different tile is their storage authority. Keep both sides of seams.
 for(var k=0u;k<8u;k++){
  let c=vec3i(p)-vec3i(umCorner(k,2u));
  if(nbAirFused&&plane==3u){
   // The six-cubed local dilation includes every incident tile of this
   // region's canonical vertices. No cross-workgroup output is consumed.
   let q=(c>>vec3u(2u))-nbAirOrigin+1;
   if(all(q>=vec3i(0))&&all(q<vec3i(6))&&((nbAirZ[u32(q.y+6*q.z)]>>u32(q.x))&1u)!=0u){return true;}
  }else{let t=umTileKey(c>>vec3u(2u));if(umKeyStored(t)&&atomicLoad(&bins[NB_AIR+plane*UM_TILES+t])!=0u){return true;}}
 }
 return false;
}
fn nbAirStore(p:vec3u,coarse:bool){
 // Fused regions include coarse vertices outside the resident h ring.
 var value=0.0;
 if(nbAirFused&&coarse){value=${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(outputPhi,vec3i(p)).x;}
 else{value=textureLoad(outputPhi,vec3i(p)).x;}
 let far=16.0*max(params.hDt.x,max(params.hDt.y,params.hDt.z));
 // Already retired values cannot change. Avoid the incident-tile searches,
 // revision checks and write-through stores for this retained air state.
 if(nbAirFused&&value>=far){return;}
 // Every tile with possible open liquid is a seed. Its incident vertices
 // stay exact; negative values outside that set can only describe buried
 // liquid. Clear those even inside the wider numerical collar, otherwise
 // coarse handoff can lift a buried zero into open cells.
 if(nbAirProtected(p,select(3u,0u,value<=0.0))){return;}
 // Refinement prolongs a 4h cell's trilinear field into its h vertices,
 // including the solid side of a liquid contact. Cleanup must protect that
 // same interpolation footprint: clearing a negative descendant while its
 // coarse endpoints are retained moves the wall solely because it refined.
 // Aligned vertices still use the physical-seed test above, so isolated
 // buried fields retain no support and are retired normally.
 if(value<=0.0&&any((p&vec3u(3u))!=vec3u(0u))){
  let low=(p/4u)*4u;let high=((p+3u)/4u)*4u;
  for(var k=0u;k<8u;k++){
   if(nbAirProtected(select(low,high,umCorner(k,2u)!=vec3u(0u)),0u)){return;}
  }
 }
 // The frame plan/census use phi < 16*max(h) for coarse near support.
 // Six-cell redistance sentinels cannot certify far air for that predicate.
 if(value<far&&all(p%vec3u(4u)==vec3u(0u))){atomicStore(&state[${NARROW_BAND_AIR_CHANGED_WORD}u],1u);}
 if(nbVertexCold(p)){return;}
 let next=max(value,far);
 if(nbReuseEnabled()&&bitcast<u32>(value)!=bitcast<u32>(next)){nbMarkPhiChanged(p);}
 textureStore(outputPhi,vec3i(p),vec4f(next));
}
${[1,4].map(width=>/* wgsl */`
${width===1?`// An h tile's 125 vertices (the ownership's h list) are a workgroup's, a lane each (two rounds).
@compute @workgroup_size(64) fn airClearFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var j=group.x;j<umCounts.x;j+=groups.x){
  let t=umTopology[UM_TILES+j];
  if(!umTileLive(t)||umTileWidth(t)!=1u||nbTileCold(t)){continue;}
  if(atomicLoad(&bins[NB_AIR+t])!=0u){continue;}
  let width=1u;let origin=4u*umTileCoord(t);let side=5u;
  let regular=umTileMinimumWidth(t)==width&&umTileMaximumWidth(t)==width;
  var low=vec3<bool>(false);if(regular){low=umLowOpen(vec3i(origin));}
  for(var k=lane;k<125u;k+=64u){`:`@compute @workgroup_size(64) fn airClearCoarse(@builtin(global_invocation_id) id:vec3u){
 let count=select(UM_TILES,64u*atomicLoad(&bins[NB_FRAME_PLAN+4u+6u*NB_REUSE_REGIONS]),nbAirListed);
 for(var job=id.x;job<count;job+=65536u){
  var t=job;if(nbAirListed){let region=atomicLoad(&bins[NB_FRAME_PLAN+5u+6u*NB_REUSE_REGIONS+job/64u]);t=nbReuseTile(region,job%64u);}
  if(!umTileLive(t)||umTileWidth(t)!=${width}u||nbTileCold(t)){continue;}
  if(atomicLoad(&bins[NB_AIR+t])!=0u){continue;}
  let width=${width}u;let origin=4u*umTileCoord(t);let side=4u/width+1u;
  let regular=umTileMinimumWidth(t)==width&&umTileMaximumWidth(t)==width;
  var low=vec3<bool>(false);if(regular){low=umLowOpen(vec3i(origin));}
  for(var k=0u;k<side*side*side;k++){`}
   let corner=umCorner(k,side);let p=origin+width*corner;
   if(regular){if(!all((corner!=vec3u(0u))|low)){continue;}}
   else{let a=umVertexAuthority(p);if(a.width!=width||a.tile!=t){continue;}}
   nbAirStore(p,width==4u);
  }
 }
}
`).join("\n")}
`;
