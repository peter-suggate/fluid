/** Retire stale far-field phi without changing vertices used by physical liquid.
 * Four tile planes: physical seeds, then a separable protected-region dilation.
 * The 16-cell collar covers the distance/volume readers' four-owner band.
 * The residency census still adds its full directional travel/stencil closure. */
export const NARROW_BAND_AIR_WORDS_PER_TILE=4;
export const NARROW_BAND_AIR_COLLAR_TILES=4;
export const narrowBandAirWGSL=/* wgsl */`
const NB_AIR=NB_ACTIVITY_SURFACE_HEAT+UM_TILES;
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
fn nbAirTileWet(t:u32)->bool{
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
 for(var lane=0u;lane<64u;lane++){
  let c=origin+umCorner(lane,4u);
  var values=coarse;var base=origin;
  if(width==1u){values=umLoadVertex8(c,1u);base=c;}
  var negative=false;for(var k=0u;k<8u;k++){negative=negative||values[k]<=0.0;}
  if(negative&&nbAirCellWet(vec3i(c),base,width,values)){return true;}
 }
 return false;
}
@compute @workgroup_size(64) fn airSeeds(@builtin(global_invocation_id) id:vec3u){
 for(var t=id.x;t<UM_TILES;t+=65536u){
  var wet=false;if(umTileLive(t)){wet=nbAirTileWet(t);}
  atomicStore(&bins[NB_AIR+t],u32(wet));
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
 // The frame plan/census use phi < 16*max(h) for coarse near support.
 // Six-cell redistance sentinels cannot certify far air for that predicate.
 let far=16.0*max(params.hDt.x,max(params.hDt.y,params.hDt.z));
 textureStore(outputPhi,vec3i(p),vec4f(max(value,far)));
}
${[1,4].map(width=>/* wgsl */`
@compute @workgroup_size(64) fn airClear${width===1?"Fine":"Coarse"}(@builtin(global_invocation_id) id:vec3u){
 for(var t=id.x;t<UM_TILES;t+=65536u){
  if(!umTileLive(t)||umTileWidth(t)!=${width}u){continue;}
  if(atomicLoad(&bins[NB_AIR+t])!=0u){continue;}
  let width=${width}u;let origin=4u*umTileCoord(t);let side=4u/width+1u;
  let regular=umTileMinimumWidth(t)==width&&umTileMaximumWidth(t)==width;
  var low=vec3<bool>(false);if(regular){low=umLowOpen(vec3i(origin));}
  for(var k=0u;k<side*side*side;k++){
   let corner=umCorner(k,side);let p=origin+width*corner;
   if(regular){if(!all((corner!=vec3u(0u))|low)){continue;}}
   else{let a=umVertexAuthority(p);if(a.width!=width||a.tile!=t){continue;}}
   nbAirStore(p);
  }
 }
}
`).join("\n")}
`;
