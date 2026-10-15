/** Reuse the reconstructed surface's bounded crossing search for redistance
 * and particle membership. Canonical vertex signs stay fixed, and fine
 * redistance leaves every interface vertex alone: its crossing set is final.
 * Fixed-band coarse values remain advected surface samples. Adaptive coarse
 * vertices beside retiring h owners recover metric magnitudes before handoff;
 * that can move a hanging zero, so they are stored first (depthCoarse, then
 * redistanceCoarse) and the fine band's search and metric follow them. */
export const narrowBandRedistanceWGSL=/* wgsl */`
// Sort cursors expire after particle ordering. Before each metric build,
// borrow that bank for eight exact normalized corners per crossing cell.
// Transfer properties may overwrite it later; the next metric rebuilds it.
override nbCachedGeometry:bool=true;
fn nbGeometryCapacity()->u32{return (UM_CELL_KEYS-1u)/8u;}
fn nbGeometryRank(mask:vec2u,lane:u32)->u32{
 let word=lane>>5u;let below=(1u<<(lane&31u))-1u;
 return select(0u,countOneBits(mask.x),word!=0u)+countOneBits(mask[word]&below);
}
var<workgroup> nbGeometryBase:u32;
var<workgroup> nbGeometryMask:vec2u;
@compute @workgroup_size(64) fn depthGeometry(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+group.x]);
 if(lane==0u){
  nbGeometryMask=vec2u(atomicLoad(&bins[NB_BAND_MASK+2u*tile]),atomicLoad(&bins[NB_BAND_MASK+2u*tile+1u]));
  let count=countOneBits(nbGeometryMask.x)+countOneBits(nbGeometryMask.y);
  nbGeometryBase=atomicAdd(&bins[UM_CELL_KEYS],count);
  // The candidate list has been consumed by depthSeeds; use it as a map.
  atomicStore(&bins[NB_BAND_CANDIDATES+tile],nbGeometryBase);
 }
 let base=workgroupUniformLoad(&nbGeometryBase);let mask=nbGeometryMask;
 if((mask[lane>>5u]&(1u<<(lane&31u)))==0u){return;}
 let slot=base+nbGeometryRank(mask,lane);if(slot>=nbGeometryCapacity()){return;}
 let origin=4u*umTileCoord(tile);let p=origin+umCorner(lane,4u);
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 for(var k=0u;k<8u;k++){
  let value=nbCrossingVertex(tile,origin,p+umCorner(k,2u))/h;
  atomicStore(&bins[UM_CELL_KEYS+1u+8u*slot+k],bitcast<u32>(value));
 }
}
fn nbTrilinear(v:array<f32,8>,q:vec3f)->vec4f{
 let x00=mix(v[0],v[1],q.x);let x10=mix(v[2],v[3],q.x);
 let x01=mix(v[4],v[5],q.x);let x11=mix(v[6],v[7],q.x);
 let y0=mix(x00,x10,q.y);let y1=mix(x01,x11,q.y);
 let dx=mix(mix(v[1]-v[0],v[3]-v[2],q.y),mix(v[5]-v[4],v[7]-v[6],q.y),q.z);
 return vec4f(dx,mix(x10-x00,x11-x01,q.z),y1-y0,mix(y0,y1,q.z));
}
// Differentiate the represented trilinear field directly. A gradient needs
// its eight corners, rather than six additional interpolated phi queries.
// c: the cell q is read in, q a point of its closed extent.
fn nbPhiGradient(c:vec3u,q:vec3f)->vec4f{
 let width=umTileWidth(umTileAt(c/4u));let origin=(c/width)*width;
 let f=(q-vec3f(origin))/f32(width);var v:array<f32,8>;var terms:array<f32,8>;
 if(width==4u){v=umLoadCorner8(origin,4u);}else{v=umLoadVertex8(origin,width);}
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);
  let w=select(vec3f(1)-f,f,corner!=vec3u(0));terms[k]=v[k]*w.x*w.y*w.z;
 }
 let value=umVertexSum8(terms);
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 return vec4f(nbTrilinear(v,f).xyz/(h*f32(width)),value/h);
}
fn nbSurfaceDistance(p:vec3f,cell:u32)->f32{
 let base=umCellCoord(cell);let origin=vec3f(base);var v:array<f32,8>;
 // Use the same represented field that marked this cell as a crossing.
 // At h/4h seams umVertexValue can reconstruct a hanging vertex from a
 // different owner and erase the zero that the crossing search found.
 let tile=nbBandTile(base);let tileOrigin=4u*umTileCoord(tile);
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 var slot=nbGeometryCapacity();
 if(nbCachedGeometry){
  let first=atomicLoad(&bins[NB_BAND_CANDIDATES+tile]);
  if(first<nbGeometryCapacity()){
   let mask=vec2u(atomicLoad(&bins[NB_BAND_MASK+2u*tile]),atomicLoad(&bins[NB_BAND_MASK+2u*tile+1u]));
   let local=base&vec3u(3u);slot=first+nbGeometryRank(mask,local.x+4u*(local.y+4u*local.z));
  }
 }
 if(slot<nbGeometryCapacity()){
  for(var k=0u;k<8u;k++){v[k]=bitcast<f32>(atomicLoad(&bins[UM_CELL_KEYS+1u+8u*slot+k]));}
 }else{
  for(var k=0u;k<8u;k++){v[k]=nbCrossingVertex(tile,tileOrigin,base+umCorner(k,2u))/h;}
 }
 // An absent zero is a failed query, never a physical distance. Preserve
 // the caller's finite fallback if a stale/degenerate cell is supplied.
 var low=v[0];var high=v[0];
 for(var k=1u;k<8u;k++){low=min(low,v[k]);high=max(high,v[k]);}
 if(low>0.0||high<0.0||low==high){return -1.0;}
 // Edge crossings supply a zero-set fallback if a clamped Newton solve
 // cannot reach zero inside this cell (thin or badly conditioned patches).
 // Every point in this unit cell is within this finite geometric bound.
 var distance=length(p-origin)+sqrt(3.0);
 for(var a=0u;a<3u;a++){for(var k=0u;k<8u;k++){
  let b=1u<<a;if((k&b)!=0u){continue;}let other=k|b;
  if(v[k]*v[other]>0.0||v[k]==v[other]){continue;}
  let t=clamp(v[k]/(v[k]-v[other]),0.0,1.0);
  let q=origin+mix(vec3f(umCorner(k,2u)),vec3f(umCorner(other,2u)),t);
  distance=min(distance,length(p-q));
 }}
 var q=clamp(p-origin,vec3f(0),vec3f(1));
 for(var i=0u;i<6u;i++){
  let f=nbTrilinear(v,q);let g2=dot(f.xyz,f.xyz);
  if(g2<1e-12){break;}
  q=clamp(q-f.w*f.xyz/g2,vec3f(0),vec3f(1));
 }
 if(abs(nbTrilinear(v,q).w)<1e-4){distance=min(distance,length(p-origin-q));}
 return distance;
}
// A vertex is its cell's tile's; one on an upper face a wall lies across
// goes with the last cell. c: that cell, of the tile of key tile. The tiles
// within three of a crossing are written: a sample in a tile within two
// reads only those.
fn nbBuildDistance(p:vec3u,c:vec3u,tile:u32){
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 {
  let initial=umSampleVertex(vec3f(p));
  var value=initial/h;
  if(abs(initial)>1e-8){
   var nearest=NB_NO_SURFACE;
   if((atomicLoad(&bins[NB_BAND_REACH+tile])&1u)!=0u){nearest=atomicLoad(&bins[NB_DEPTH_B+umCellAt(c)]);}
   // With no crossing cell within five of the vertex's, the nearest box is
   // five cells off or more: the level set stands, no nearer than that. Six
   // lifted the air five cells over a resting surface, and with it the 4h
   // owner whose phi sets the surface's pressure theta.
   var distance=max(abs(value),5.0);
   if(nearest!=NB_NO_SURFACE){
    let candidate=nbSurfaceDistance(vec3f(p),nearest);
    if(candidate>=0.0){distance=candidate;}
   }
   // Cell-centre proximity alone can select the wrong interface beside a
   // tiny ripple. Also trace the local phi gradient across cell boundaries;
   // unlike a cell-clamped solve this reaches the surface above that vertex.
   if(abs(value)<4.5){
    // The trace moves a cell and a half at most a turn: its cell is asked
    // again from the cell it has reached, and a wall holds both.
    var q=vec3f(p);var cell=vec3i(c);
    for(var iteration=0u;iteration<5u;iteration++){
     // A solid's phi is not the liquid's: traced into a wall, the vertices on
     // it found the wall for a surface, and no samples were seeded beside it.
     if(umCellOpen(cell)<=1e-5){break;}
     let sample=nbPhiGradient(vec3u(cell),q);let f=sample.w;if(abs(f)<1e-5){distance=min(distance,length(q-vec3f(p)));break;}
     let g=sample.xyz;
     let g2=dot(g,g);if(g2<1e-8){break;}
     q=nbPointNear(cell,q-clamp(f/sqrt(g2),-1.5,1.5)*g*inverseSqrt(g2));cell=nbCellNear(cell,vec3i(floor(q)));
    }
   }
   value=sign(initial)*max(distance,1e-8/h);
  }
  atomicStore(&bins[NB_DEPTH_A+umVertexAt(p)],bitcast<u32>(value));
 }
}
// The 64 interior vertices use every lane. A tile on a high face of its page
// has the vertices of that face too where a wall lies across it: their cell
// is held in the tile. Evaluate those once after the dense cube.
@compute @workgroup_size(64) fn buildDistance(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_DISTANCE+group.x]);let origin=4u*umTileCoord(tile);
 let inner=origin+umCorner(lane,4u);nbBuildDistance(inner,inner,tile);
 if(any(((origin+4u)&vec3u(31u))==vec3u(0u))){
  for(var k=lane;k<125u;k+=64u){
   let local=umCorner(k,5u);let p=origin+local;
   if(!any(local==vec3u(4u))){continue;}
   let c=vec3u(umCellNearFromPage(tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));
   if(any((local==vec3u(4u))&(c==p))){continue;}
   nbBuildDistance(p,c,tile);
  }
 }
}
fn nbInterfaceVertex(p:vec3u)->bool{
 // A vertex touching a crossing cell (its bit in the tile mask)
 // any such cell determines its interpolated zero set: changing its magnitude
 // would move that surface even if its sign stayed fixed.
 let local=p&vec3u(3u);
 if(all(local!=vec3u(0u))){
  // Interior vertices touch one tile only. Test all incident cells with one
  // mask load, or two when the z pair crosses the packed word boundary.
  let x=(1u<<local.x)|(1u<<(local.x-1u));
  let xy=(x<<(4u*local.y))|(x<<(4u*(local.y-1u)));
  let at=NB_BAND_MASK+2u*nbBandTile(p);
  if(local.z==2u){return (((atomicLoad(&bins[at])>>16u)|atomicLoad(&bins[at+1u]))&xy)!=0u;}
  let bits=atomicLoad(&bins[at+(local.z>>1u)]);
  return (bits&(xy|(xy<<16u)))!=0u;
 }
 for(var k=0u;k<8u;k++){
  let c=vec3i(p)-vec3i(umCorner(k,2u));
  if(!umCellPresent(c)){continue;}
  if(nbBandCrossing(vec3u(c))){return true;}
 }
 return false;
}
// p: a vertex of the tile of key own and first cell origin; c: its cell,
// held in the tile's page where a wall lies across an upper face.
fn nbRedistanceVertex(own:u32,origin:vec3u,p:vec3u,width:u32){
 let c=vec3u(umCellNearFromPage(own>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 // Before retiring adjacent h owners, recover the metric coarse samples.
 // Particle sphere unions preserve their fine zero set but distort interior
 // magnitudes; inheriting those magnitudes as a 4h chord moves a flat surface.
 // Only a vertex whose tile has an h neighbour can retire. Test its
 // stencil first, then the eight heat words; this zero predicate needs no phi.
 let retiring=width==4u&&umTileMinimumWidth(umTileAt(c/4u))==1u&&!nbSurfaceHotCornerFrom(vec3i(origin>>vec3u(2u)),p);
 let normalize=(width==1u&&!nbInterfaceVertex(p))||retiring;
 // The metric replacement needs no sample of the old field. Keep its exact
 // multiply, and read phi only where the reconstructed value is consumed.
 if(normalize&&(nbBandReach(c)&2u)!=0u){
  let value=bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+umVertexAt(p)]))*h;
  textureStore(outputPhi,vec3i(p),vec4f(value));return;
 }
 // Coarse/interface samples retain their reconstructed values: nothing to
 // write. Past reach, preserve the old field's sign and its minimum
 // six-cell magnitude. Every dispatched vertex is canonical and has one
 // writer. The metric bank was completed before this pass, so only this
 // vertex's old value is needed. Coarse dispatches bind its base home; fine
 // ones its resident h home. No neighbor can be read while another
 // invocation overwrites it.
 if(!normalize){return;}
 let value=textureLoad(outputPhi,vec3i(p)).x;
 if(abs(value)>1e-8&&abs(value)<6.0*h){textureStore(outputPhi,vec3i(p),vec4f(sign(value)*6.0*h));}
}
@compute @workgroup_size(128) fn redistanceFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(vec3u(group.x*64u,0u,0u));let origin=umTileCoord(owner.tile)*4u;
 if(lane>=125u){return;}
 let local=vec3u(lane%5u,(lane/5u)%5u,lane/25u);let p=origin+local;
 let regular=umTileMaximumWidth(owner.tile)==1u&&umTileMinimumWidth(owner.tile)==1u;
 // A low vertex is this tile's only where nothing is stored below it.
 if(regular){if(any(local==vec3u(0))&&!all((local>vec3u(0))|umLowOpen(vec3i(origin)))){return;}}
 else{let a=umVertexAuthority(p);if(a.width!=1u||a.tile!=owner.tile){return;}}
 nbRedistanceVertex(owner.tile,origin,p,1u);
}
// The metric of the coarse corners retirement can normalize, where
// redistanceCoarse reads it: one writer each. Every corner beside an h owner
// is built only if its heat can retire: the corner's zero-heat predicate
// depends on heat alone, so it is identical in this pass and the store.
// Compiled without the geometry
// cache, which the fine band's metric builds after its own search.
@compute @workgroup_size(64) fn depthCoarse(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<umCounts.y;i+=65536u){
  let owner=umAllOwner(vec3u(umCounts.x*64u+i,0u,0u));let origin=umOrigin(owner);
  let regular=umTileMaximumWidth(owner.tile)==4u&&umTileMinimumWidth(owner.tile)==4u;
  // Keep the regular owner's low-wall rule. At seams the layout already
  // compiled the canonical coarse corners; no authority lookup is needed.
  var low=vec3<bool>(false);if(regular){low=umLowOpen(vec3i(origin));}
  let corners=umCoarseCornerMask(owner.tile);
  for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let p=origin+corner*4u;
   if(regular){if(!all((corner!=vec3u(0))|low)){continue;}}
   else if((corners&(1u<<k))==0u){continue;}
   let c=vec3u(umCellNearFromPage(owner.tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));
   if(umTileMinimumWidth(umTileAt(c/4u))!=1u||(nbBandReach(c)&2u)==0u){continue;}
   if(nbSurfaceHotCornerFrom(vec3i(origin>>vec3u(2u)),p)){continue;}
   nbBuildDistance(p,c,nbBandTile(c));
  }
 }
}
@compute @workgroup_size(64) fn redistanceCoarse(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<umCounts.y;i+=65536u){
  let owner=umAllOwner(vec3u(umCounts.x*64u+i,0u,0u));let origin=umOrigin(owner);
  let regular=umTileMaximumWidth(owner.tile)==4u&&umTileMinimumWidth(owner.tile)==4u;
  // Keep the regular owner's low-wall rule. At seams the layout already
  // compiled the canonical coarse corners; no authority lookup is needed.
  var low=vec3<bool>(false);if(regular){low=umLowOpen(vec3i(origin));}
  let corners=umCoarseCornerMask(owner.tile);
  for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let p=origin+corner*4u;
   if(regular){if(!all((corner!=vec3u(0))|low)){continue;}}
   else if((corners&(1u<<k))==0u){continue;}
   nbRedistanceVertex(owner.tile,origin,p,4u);
  }
 }
}
`;
