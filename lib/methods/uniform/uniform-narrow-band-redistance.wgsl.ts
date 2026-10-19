/** Coarse retirement's metric and store, and the zero-set distance both it
 * and the surface brick arena's redistance (uniform-surface-brick-arena) use.
 * Canonical vertex signs stay fixed, and fine redistance leaves every
 * interface vertex alone: its crossing set is final. Fixed-band coarse
 * values remain advected surface samples. Adaptive coarse vertices beside
 * retiring h owners recover metric magnitudes before handoff; that can move
 * a hanging zero, so they are stored first (depthCoarse, then
 * redistanceCoarse) and the arena's metric follows them. */
export const narrowBandRedistanceWGSL=/* wgsl */`
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
// The distance from q (cell units, from the cell's low corner) to the zero
// of the trilinear cell of corners v (phi/h), or -1 where it holds none.
fn nbCellDistance(q:vec3f,v:array<f32,8>)->f32{
 // An absent zero is a failed query, never a physical distance. Preserve
 // the caller's finite fallback if a stale/degenerate cell is supplied.
 var low=v[0];var high=v[0];
 for(var k=1u;k<8u;k++){low=min(low,v[k]);high=max(high,v[k]);}
 if(low>0.0||high<0.0||low==high){return -1.0;}
 // Edge crossings supply a zero-set fallback if a clamped Newton solve
 // cannot reach zero inside this cell (thin or badly conditioned patches).
 // Every point in this unit cell is within this finite geometric bound.
 var distance=length(q)+sqrt(3.0);
 for(var a=0u;a<3u;a++){for(var k=0u;k<8u;k++){
  let b=1u<<a;if((k&b)!=0u){continue;}let other=k|b;
  if(v[k]*v[other]>0.0||v[k]==v[other]){continue;}
  let t=clamp(v[k]/(v[k]-v[other]),0.0,1.0);
  distance=min(distance,length(q-mix(vec3f(umCorner(k,2u)),vec3f(umCorner(other,2u)),t)));
 }}
 var r=clamp(q,vec3f(0),vec3f(1));
 for(var i=0u;i<6u;i++){
  let f=nbTrilinear(v,r);let g2=dot(f.xyz,f.xyz);
  if(g2<1e-12){break;}
  r=clamp(r-f.w*f.xyz/g2,vec3f(0),vec3f(1));
 }
 if(abs(nbTrilinear(v,r).w)<1e-4){distance=min(distance,length(q-r));}
 return distance;
}
// Use the same represented field that marked this cell as a crossing.
// At h/4h seams umVertexValue can reconstruct a hanging vertex from a
// different owner and erase the zero that the crossing search found.
fn nbSurfaceDistance(p:vec3f,cell:u32)->f32{
 let base=umCellCoord(cell);var v:array<f32,8>;
 let tile=nbBandTile(base);let tileOrigin=4u*umTileCoord(tile);
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 for(var k=0u;k<8u;k++){v[k]=nbCrossingVertex(tile,tileOrigin,base+umCorner(k,2u))/h;}
 return nbCellDistance(p-vec3f(base),v);
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
// p: a corner of the coarse owner of key own and first cell origin. Before
// retiring adjacent h owners, recover the metric coarse samples. Particle
// sphere unions preserve their fine zero set but distort interior
// magnitudes; inheriting those magnitudes as a 4h chord moves a flat surface.
// Only a vertex whose tile has an h neighbour can retire. Test its stencil
// first, then the eight heat words; this zero predicate needs no phi.
fn nbRedistanceVertex(own:u32,origin:vec3u,p:vec3u){
 let c=vec3u(umCellNearFromPage(own>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 if(umTileMinimumWidth(umTileAt(c/4u))!=1u||nbSurfaceHotCornerFrom(vec3i(origin>>vec3u(2u)),p)){return;}
 // A buried vertex is not a distance sample. In particular, padding past a
 // non-page-aligned tank wall must keep its far marker: replacing it with
 // the distance to the waterline pushes the interpolated wall contour into
 // the solid on every retirement. Use owner capacity, so corners needed by
 // partially open 4h owners still participate in the metric reconstruction.
 if(umVertexBuried(vec3i(p))){return;}
 if((nbBandReach(c)&2u)!=0u){nbStoreInPlace(p,bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+umVertexAt(p)]))*h);return;}
 // Past reach, preserve the old field's sign and its minimum six-cell
 // magnitude. Every dispatched vertex is canonical and has one writer.
 let value=textureLoad(outputPhi,vec3i(p)).x;
 if(abs(value)>1e-8&&abs(value)<6.0*h){nbStoreInPlace(p,sign(value)*6.0*h);}
}
// The metric of the coarse corners retirement can normalize, where
// redistanceCoarse reads it: one writer each. Every corner beside an h owner
// is built only if its heat can retire: the corner's zero-heat predicate
// depends on heat alone, so it is identical in this pass and the store.
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
// A sign change into a buried far marker is a solid contact, not a free
// surface. Match the crossing search's open-cell rule, including both sides
// of an integer plane (the zero can lie exactly on a voxel wall or edge).
fn nbMetricZeroOpen(p:vec3f)->bool{
 let low=vec3i(ceil(p))-vec3i(1);let high=vec3i(floor(p));
 for(var k=0u;k<8u;k++){
  let cell=select(low,high,umCorner(k,2u)!=vec3u(0u));
  if(umCellOpen(cell)>1e-5){return true;}
 }
 return false;
}
fn nbBoundCoarseMetric(p:vec3u){
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));let at=NB_DEPTH_A+umVertexAt(p);
 let d=bitcast<f32>(atomicLoad(&bins[at]));var v=0.0;var bound=abs(d);
 for(var a=0u;a<7u;a++){
  var q=vec3i(p);if(a>0u){q[(a-1u)>>1u]+=select(4,-4,((a-1u)&1u)!=0u);}
  if(any(q<vec3i(0))){continue;}
  let n=umSampleVertex(vec3f(q))/h;
  if(a==0u){v=n;}else if(v*n<0.0){
   let t=abs(v)/(abs(v)+abs(n));
   // Keep the two stationary coordinates exactly on their lattice planes.
   let axis=(a-1u)>>1u;var zero=vec3f(p);zero[axis]=mix(f32(p[axis]),f32(q[axis]),t);
   if(nbMetricZeroOpen(zero)){bound=min(bound,4.0*t);}
  }
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
   nbRedistanceVertex(owner.tile,origin,p);
  }
 }
}
`;
