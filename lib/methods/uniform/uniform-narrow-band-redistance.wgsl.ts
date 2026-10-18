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
var<workgroup> nbMetricReuseFlag:u32;
@compute @workgroup_size(64) fn depthGeometry(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+group.x]);
 // A tile whose metric is retained builds no distance: no geometry. Its map
 // entry says so (capacity), and a reader gathers the cell itself.
 if(lane==0u){nbMetricReuseFlag=u32(nbMetricReuseAvailable(tile));if(nbMetricReuseFlag!=0u){atomicStore(&bins[NB_BAND_CANDIDATES+tile],nbGeometryCapacity());}}
 if(workgroupUniformLoad(&nbMetricReuseFlag)!=0u){return;}
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
 // A current crossing-cell plane also shortens the detailed path at tile
 // boundaries. Its residual bounds the whole trilinear cell; the projected
 // foot's residual-sized normal segment must stay inside that cell. This avoids
 // the edge/Newton work without needing persistent geometry or temporal error.
 if(nbReuseCacheEnabled()){
 var centre=0.0;var gradient=vec3f(0);var scale=0.0;
 for(var k=0u;k<8u;k++){
  centre+=0.125*v[k];gradient+=0.25*v[k]*(2.0*vec3f(umCorner(k,2u))-1.0);scale=max(scale,abs(v[k]));
 }
 let slope=length(gradient);
 if(slope>1e-6&&slope<1e6){
  var residual=0.0000152587890625*max(1.0,scale);
  for(var k=0u;k<8u;k++){
   residual=max(residual,abs(v[k]-centre-dot(gradient,vec3f(umCorner(k,2u))-0.5))+0.0000152587890625*max(1.0,scale));
  }
  let error=residual/slope;let q=p-origin;
  let signed=(centre+dot(gradient,q-0.5))/slope;
  let foot=q-signed*gradient/slope;
  let margin=(error+0.00001)*abs(gradient/slope);
  if(error>=0.0&&error<=0.02&&abs(signed)<=3.402823e38&&all(foot>=margin)&&all(foot<=vec3f(1)-margin)){return abs(signed);}
 }
 }
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
// Current-field approximation, with no temporal freezing: each new crossing
// scan refreshes its plane and residual, while exact unchanged inputs retain
// that same proof. A whole ball around p must remain inside the certified
// tile; checking only the projected foot could miss a nearer unseen zero.
fn nbMetricReuseAvailable(tile:u32)->bool{
 let region=nbReuseRegion(tile);
 if(region>=NB_REUSE_METRIC_CAPACITY||!nbMetricRegionReusable(tile)){return false;}
 return atomicLoad(&bins[NB_REUSE_METRIC_VALID+64u*region+nbReuseLocal(tile)])!=0u;
}
fn nbCurrentPlaneEvidence(p:vec3u,tile:u32)->vec3f{
 let region=nbReuseRegion(tile);
 if(!nbReuseCacheEnabled()||region>=NB_REUSE_METRIC_CAPACITY){return vec3f(0);}
 let at=NB_REUSE_REGION_PLANE+8u*region;
 if(atomicLoad(&bins[at+7u])==0u){return vec3f(0);}
 var plane:vec4f;for(var k=0u;k<4u;k++){plane[k]=bitcast<f32>(atomicLoad(&bins[at+k]));}
 let residual=bitcast<f32>(atomicLoad(&bins[at+4u]));let slope=length(plane.xyz);
 if(!(slope>1e-6&&slope<1e6)){return vec3f(0);}
 let q=vec3f(p)-vec3f(16u*(umTileCoord(tile)/4u));
 let phi=plane.w+dot(plane.xyz,q-8.0);let error=residual/slope;
 if(!(residual>=0.0&&error>=0.0&&error<=0.02&&abs(phi)<=3.402823e38&&abs(phi)>residual)){return vec3f(0);}
 let distance=phi/slope;let radius=abs(distance)+error+0.00001;
 // The whole nearest-distance ball must be covered: otherwise an unseen
 // zero beyond the certificate could be closer than the predicted surface.
 if(any(q-vec3f(radius)<vec3f(0))||any(q+vec3f(radius)>vec3f(16))){return vec3f(0);}
 let mask=vec2u(atomicLoad(&bins[at+5u]),atomicLoad(&bins[at+6u]));
 if(any(mask!=vec2u(0xffffffffu))){
  let lo=vec3u(floor((q-radius)/4.0));let hi=min(vec3u(3),vec3u(floor((q+radius)/4.0)));
  let xmask=((1u<<(hi.x-lo.x+1u))-1u)<<lo.x;
  for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){
   let needed=xmask<<(4u*y+16u*(z&1u));
   if((mask[z>>1u]&needed)!=needed){return vec3f(0);}
  }}
 }
 let defect=residual+abs(phi-distance)+error;
 return vec3f(distance,1.0,defect);
}
fn nbCurrentPlaneMetric(p:vec3u,tile:u32)->vec2f{return nbCurrentPlaneEvidence(p,tile).xy;}
// A vertex is its cell's tile's; one on an upper face a wall lies across
// goes with the last cell. c: that cell, of the tile of key tile. The tiles
// within three of a crossing are written: a sample in a tile within two
// reads only those.
fn nbBuildDistance(p:vec3u,c:vec3u,tile:u32){
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 {
  let plane=nbCurrentPlaneMetric(p,tile);
  if(plane.y!=0.0){
   atomicStore(&bins[NB_DEPTH_A+umVertexAt(p)],bitcast<u32>(plane.x));return;
  }
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
// is held in the tile. Those are a second round, a lane each (61 of them:
// x = 4, then y = 4, then z = 4), through the same call site: a second
// inlined nbBuildDistance, trace and all, cost the whole pass 1.4 ms of a
// 6 ms late-splash frame.
// Its h-side pipeline is launched as a solid-free twin and a general
// pipeline (umSolidSkip); the others test nothing here.
@compute @workgroup_size(64) fn buildDistance(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umSolidSkip()){return;}
 let tile=atomicLoad(&bins[NB_BAND_DISTANCE+group.x]);let origin=4u*umTileCoord(tile);
 // The 16h producer certificate covers the whole metric dependency halo.
 // A retained tile restores the borrowed metric bank before any phi/solid
 // gathers or zero-set iterations. Cache validity survives scratch reuse.
 let region=nbReuseRegion(tile);
 if(lane==0u){nbMetricReuseFlag=u32(nbMetricReuseAvailable(tile));}
 let cached=workgroupUniformLoad(&nbMetricReuseFlag)!=0u;
 let metricBase=NB_REUSE_METRIC+125u*(64u*region+nbReuseLocal(tile));
 var p=origin+umCorner(lane,4u);var c=p;
 let face=any(((origin+4u)&vec3u(31u))==vec3u(0u));
 for(var round=0u;round<select(1u,2u,face);round++){
  if(round==1u){
   if(lane>=61u){break;}
   var local=vec3u(4u,lane%5u,lane/5u);
   if(lane>=45u){let m=lane-45u;local=vec3u(m&3u,m>>2u,4u);}else if(lane>=25u){let m=lane-25u;local=vec3u(m&3u,4u,m>>2u);}
   p=origin+local;c=vec3u(umCellNearFromPage(tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));
   if(any((local==vec3u(4u))&(c==p))){break;}
  }
  let local=p-origin;let metricSlot=local.x+5u*(local.y+5u*local.z);
  if(cached){
   atomicStore(&bins[NB_DEPTH_A+umVertexAt(p)],atomicLoad(&bins[metricBase+metricSlot]));
  }else{
   nbBuildDistance(p,c,tile);
   if(nbReuseCacheEnabled()&&region<NB_REUSE_METRIC_CAPACITY){
    atomicStore(&bins[metricBase+metricSlot],atomicLoad(&bins[NB_DEPTH_A+umVertexAt(p)]));
   }
  }
 }
 // Workgroup completion makes every canonical payload available before a
 // later search can accept the tile. Noncached/overflow tiles run fully.
 storageBarrier();
 if(lane==0u&&nbReuseCacheEnabled()&&region<NB_REUSE_METRIC_CAPACITY){
  atomicStore(&bins[NB_REUSE_METRIC_VALID+64u*region+nbReuseLocal(tile)],1u);
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
  nbStoreInPlace(p,value);return;
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
 if(abs(value)>1e-8&&abs(value)<6.0*h){nbStoreInPlace(p,sign(value)*6.0*h);}
}
@compute @workgroup_size(128) fn redistanceFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(vec3u(group.x*64u,0u,0u));let origin=umTileCoord(owner.tile)*4u;
 // Every vertex of a COLD region's tile is COLD: redistance holds it.
 if(lane>=125u||nbTileCold(owner.tile)){return;}
 let local=vec3u(lane%5u,(lane/5u)%5u,lane/25u);let p=origin+local;
 // A current SDF-defect bound includes any previously skipped drift. The
 // next current producer scan reopens work when that defect exceeds budget.
 // Far-reach clamping retains its original semantics below.
 if(all(local>vec3u(0))&&all(local<vec3u(4))&&(atomicLoad(&bins[NB_BAND_REACH+owner.tile])&2u)!=0u){
  let plane=nbCurrentPlaneEvidence(p,owner.tile);
  if(plane.y!=0.0&&plane.z<=0.02){return;}
 }
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
// A lane to a corner (eight an owner): a seam owner's corners each run a
// search and trace, which held one lane behind eight of them.
@compute @workgroup_size(64) fn depthCoarse(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<8u*umCounts.y;i+=65536u){
  let owner=umAllOwner(vec3u(umCounts.x*64u+(i>>3u),0u,0u));let origin=umOrigin(owner);
  if(nbTileCold(owner.tile)){continue;}
  let regular=umTileMaximumWidth(owner.tile)==4u&&umTileMinimumWidth(owner.tile)==4u;
  // Keep the regular owner's low-wall rule. At seams the layout already
  // compiled the canonical coarse corners; no authority lookup is needed.
  let k=i&7u;let corner=umCorner(k,2u);let p=origin+corner*4u;
  if(regular){if(!all((corner!=vec3u(0))|umLowOpen(vec3i(origin)))){continue;}}
  else if((umCoarseCornerMask(owner.tile)&(1u<<k))==0u){continue;}
  let c=vec3u(umCellNearFromPage(owner.tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));
  if(umTileMinimumWidth(umTileAt(c/4u))!=1u||(nbBandReach(c)&2u)==0u){continue;}
  if(nbSurfaceHotCornerFrom(vec3i(origin>>vec3u(2u)),p)){continue;}
  nbBuildDistance(p,c,nbBandTile(c));
  nbBoundCoarseMetric(p);
 }
}
// A retiring corner's metric is no farther than the zero the represented
// field has on each of its six 4h lattice edges: the search builds it from
// the band's crossings, which miss a surface lying along a coarse tile's
// own edges (calm water under an h tile a falling drop activated: the
// corner took the distance to the drop and a 2.4 h hole opened beneath it).
// The current field is this pass's input (redistanceCoarse stores later).
fn nbBoundCoarseMetric(p:vec3u){
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));let at=NB_DEPTH_A+umVertexAt(p);
 let d=bitcast<f32>(atomicLoad(&bins[at]));var v=0.0;var bound=abs(d);
 for(var a=0u;a<7u;a++){
  var q=vec3i(p);if(a>0u){q[(a-1u)>>1u]+=select(4,-4,((a-1u)&1u)!=0u);}
  if(any(q<vec3i(0))){continue;}
  let n=umSampleVertex(vec3f(q))/h;
  if(a==0u){v=n;}else if(v*n<0.0){bound=min(bound,4.0*abs(v)/(abs(v)+abs(n)));}
 }
 atomicStore(&bins[at],bitcast<u32>(select(-bound,bound,d>=0.0)));
}
@compute @workgroup_size(64) fn redistanceCoarse(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<umCounts.y;i+=65536u){
  let owner=umAllOwner(vec3u(umCounts.x*64u+i,0u,0u));let origin=umOrigin(owner);
  if(nbTileCold(owner.tile)){continue;}
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
