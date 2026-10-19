import {NARROW_BAND_ACTIVITY_WORDS_PER_TILE} from "./uniform-narrow-band-activity.wgsl";
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
fn nbAirProtected(p:vec3u,plane:u32)->bool{
 // Shared canonical vertices belong to every incident tile, even when a
 // different tile is their storage authority. Keep both sides of seams.
 for(var k=0u;k<8u;k++){
  let c=vec3i(p)-vec3i(umCorner(k,2u));let t=umTileKey(c>>vec3u(2u));
  if(umKeyStored(t)&&atomicLoad(&bins[NB_AIR+plane*UM_TILES+t])!=0u){return true;}
 }
 return false;
}
fn nbAirStore(p:vec3u){
 let value=textureLoad(outputPhi,vec3i(p)).x;
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
 let far=16.0*max(params.hDt.x,max(params.hDt.y,params.hDt.z));
 if(value<far&&all(p%vec3u(4u)==vec3u(0u))){atomicStore(&state[${NARROW_BAND_AIR_CHANGED_WORD}u],1u);}
 nbStoreInPlace(p,max(value,far));
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
 for(var t=id.x;t<UM_TILES;t+=65536u){
  if(!umTileLive(t)||umTileWidth(t)!=${width}u||nbTileCold(t)){continue;}
  if(atomicLoad(&bins[NB_AIR+t])!=0u){continue;}
  let width=${width}u;let origin=4u*umTileCoord(t);let side=4u/width+1u;
  let regular=umTileMinimumWidth(t)==width&&umTileMaximumWidth(t)==width;
  var low=vec3<bool>(false);if(regular){low=umLowOpen(vec3i(origin));}
  for(var k=0u;k<side*side*side;k++){`}
   let corner=umCorner(k,side);let p=origin+width*corner;
   if(regular){if(!all((corner!=vec3u(0u))|low)){continue;}}
   else{let a=umVertexAuthority(p);if(a.width!=width||a.tile!=t){continue;}}
   nbAirStore(p);
  }
 }
}
`).join("\n")}
`;
