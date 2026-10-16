/** Shared bounded crossing search, over the tiles the surface reaches.
 * A flat pass lists the tiles that can hold a crossing cell; each writes its
 * crossings as a 64-bit mask and marks the tiles within three of it. A cell
 * has a crossing within five cells only in a tile within two of one, so the
 * search runs on those alone: a separable squared-distance minimization
 * (11 taps per axis) over the masks, sharing the X/XY minima in scratch, that
 * finds the exact nearest crossing-cell centre. Reconstruction uses its
 * cell-box guard against stale bulk phi. Redistance then writes nodal metric
 * distances (bank A) on the tiles within three; particle membership samples
 * that field independently of h/4h ownership. Outside the reach nothing is
 * stored: a reader there has the level set alone, no nearer than six cells. */
import { narrowBandNearestWGSL } from "./uniform-narrow-band-nearest.wgsl";

export const narrowBandMembershipWGSL=/* wgsl */`
const NB_DEPTH_A:u32=NB_SURFACE_TILES+UM_TILES;
// Bank A holds a word a vertex key, or the two packed nearest banks (16 words a tile key each) where those are longer.
const NB_DEPTH_B:u32=NB_DEPTH_A+max(UM_VERTEX_KEYS,UM_CELL_KEYS/2u);
const NB_NO_SURFACE:u32=0xffffffffu;
// Tile banks of the search, rebuilt with it: three list counts (candidates,
// search, distance), a crossing mask of two words a tile (bit x+4y+16z), the
// reach (1: within two tiles of a crossing, 2: within three), the lists.
// The fourth count and list are the transfer's: its 4h owners with a gather.
const NB_BAND_TILES:u32=UM_TILES;
const NB_BAND:u32=NB_DEPTH_B+UM_CELL_KEYS;
const NB_BAND_MASK:u32=NB_BAND+4u;
const NB_BAND_REACH:u32=NB_BAND_MASK+2u*NB_BAND_TILES;
const NB_BAND_CANDIDATES:u32=NB_BAND_REACH+NB_BAND_TILES;
const NB_BAND_SEARCH:u32=NB_BAND_CANDIDATES+NB_BAND_TILES;
const NB_BAND_DISTANCE:u32=NB_BAND_SEARCH+NB_BAND_TILES;
const NB_BAND_SEAM:u32=NB_BAND_DISTANCE+NB_BAND_TILES;
fn nbBandTile(c:vec3u)->u32{return umTileAt(c/4u);}
fn nbBandReach(c:vec3u)->u32{return atomicLoad(&bins[NB_BAND_REACH+nbBandTile(c)]);}
fn nbBandCrossing(c:vec3u)->bool{
 let l=c&vec3u(3u);
 return ((atomicLoad(&bins[NB_BAND_MASK+2u*nbBandTile(c)+(l.z>>1u)])>>(l.x+4u*l.y+16u*(l.z&1u)))&1u)!=0u;
}
// Crossing scans visit the integer vertices of a tile (key tile, first
// cell origin), its upper faces with them. Use the positive-side sampling
// owner, held in the tile where a wall lies across an upper face.
// Fine vertices and 4-aligned coarse corners have one nonzero weight; other
// coarse lattice vertices retain the sampler's complete ordered D4 sum.
fn nbCrossingVertex(tile:u32,origin:vec3u,p:vec3u)->f32{
 let cell=vec3u(umCellNearFromPage(tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));let width=umTileWidth(umTileAt(cell/4u));
 if(width==1u){return umLoadVertex(p);}
 if(all((p&vec3u(3u))==vec3u(0u))){return umLoadCorner(p);}
 return umSampleVertex(vec3f(p));
}
// A 4h tile among 4h tiles interpolates its corners throughout: it holds a
// crossing only if they straddle zero. Beside an h tile its upper faces may
// be stored h vertices, so it is listed with the h tiles and decided there.
@compute @workgroup_size(64) fn depthTiles(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){
  if(!umTileLive(t)){continue;}
  // Pure 4h Eulerian surfaces neither reconstruct particles nor redistance
  // h vertices. Search only where fine geometry or live/activating samples
  // need it; the existing reach dilation supplies the inner overlap collar.
  if(umTileMinimumWidth(t)==4u&&atomicLoad(&bins[NB_SURFACE_TILES+t])==0u&&bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+t]))<=0.0){continue;}
  var candidate=true;
  if(umTileMinimumWidth(t)==4u){
   let origin=4u*umTileCoord(t);var low=3.0e38;var high=-3.0e38;
   for(var k=0u;k<8u;k++){let value=nbCrossingVertex(t,origin,origin+4u*umCorner(k,2u));low=min(low,value);high=max(high,value);}
   candidate=low<=0.0&&high>=0.0;
  }
  if(candidate){let slot=atomicAdd(&bins[NB_BAND],1u);atomicStore(&bins[NB_BAND_CANDIDATES+slot],t);}
 }
}
var<workgroup> nbBandVertices:array<f32,125>;
var<workgroup> nbBandMask:array<atomic<u32>,2>;
@compute @workgroup_size(128) fn depthSeeds(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_CANDIDATES+group.x]);let coord=umTileCoord(tile);let origin=4u*coord;
 if(lane<125u){nbBandVertices[lane]=nbCrossingVertex(tile,origin,origin+umCorner(lane,5u));}
 if(lane<2u){atomicStore(&nbBandMask[lane],0u);}
 workgroupBarrier();
 if(lane<64u){
  let l=umCorner(lane,4u);var low=3.0e38;var high=-3.0e38;
  for(var k=0u;k<8u;k++){let v=l+umCorner(k,2u);let value=nbBandVertices[v.x+5u*(v.y+5u*v.z)];low=min(low,value);high=max(high,value);}
  // Exact zero belongs to either incident cell; a flat zero plateau alone
  // is not an interface. Domain and solid walls are not free surfaces.
  if(low<=0.0&&high>=0.0&&low<high&&umCellOpen(vec3i(origin+l))>1e-5){atomicOr(&nbBandMask[l.z>>1u],1u<<(l.x+4u*l.y+16u*(l.z&1u)));}
 }
 workgroupBarrier();
 let mask=vec2u(atomicLoad(&nbBandMask[0]),atomicLoad(&nbBandMask[1]));
 if((mask.x|mask.y)==0u){return;}
 if(lane<2u){atomicStore(&bins[NB_BAND_MASK+2u*tile+lane],mask[lane]);}
 for(var k=lane;k<343u;k+=128u){
  let o=vec3i(umCorner(k,7u))-3;let t=vec3i(coord)+o;
  let key=umTileKey(t);if(key>=UM_TILES){continue;}
  atomicOr(&bins[NB_BAND_REACH+key],select(3u,2u,any(abs(o)>vec3i(2))));
 }
}
@compute @workgroup_size(64) fn depthLists(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){
  if(!umTileLive(t)){continue;}
  let reach=atomicLoad(&bins[NB_BAND_REACH+t]);
  if((reach&1u)!=0u){let slot=atomicAdd(&bins[NB_BAND+1u],1u);atomicStore(&bins[NB_BAND_SEARCH+slot],t);}
  if((reach&2u)!=0u){let slot=atomicAdd(&bins[NB_BAND+2u],1u);atomicStore(&bins[NB_BAND_DISTANCE+slot],t);}
 }
}
// The crossings of one x row, 14 cells from five below the tile: its own
// four and five either side, read from the masks of the five tiles across.
fn nbBandRow(coord:vec3u,y:i32,z:i32)->u32{
 if(!umCellPresent(vec3i(4*i32(coord.x),y,z))){return 0u;}
 let shift=4u*(u32(y)&3u)+16u*(u32(z)&1u);let word=(u32(z)&3u)>>1u;var bits=0u;
 for(var o=-2;o<=2;o++){
  let t=vec3i(i32(coord.x)+o,y>>2u,z>>2u);let key=umTileKey(t);if(key>=UM_TILES){continue;}
  let nibble=(atomicLoad(&bins[NB_BAND_MASK+2u*key+word])>>shift)&15u;
  if(o==-2){bits|=nibble>>3u;}else{bits|=nibble<<u32(4*o+5);}
 }
 return bits&0x3fffu;
}
${narrowBandNearestWGSL}
// c: the cell p is read in, the caller's clamp of it.
fn bulkDepth(c:vec3u,p:vec3f)->f32{
 let phiValue=bandPhi(p);if(phiValue>=0.0){return phiValue;}
 if((nbBandReach(c)&1u)==0u){return min(phiValue,-8.0);}
 let nearest=atomicLoad(&bins[NB_DEPTH_B+umCellAt(c)]);
 // No crossing cell within five of this one: its box is five cells off or
 // more, six only from below. A deeper bound sank the resting bath's phi
 // from seven cells down, under the 4h owners that read it.
 if(nearest==NB_NO_SURFACE){return min(phiValue,-5.0);}
 let low=vec3f(umCellCoord(nearest));let delta=max(max(low-p,p-(low+1.0)),vec3f(0));
 return min(phiValue,-length(delta));
}
// Bank A is the nodal metric field redistance leaves on the reached tiles.
// Particle membership does not depend on the resolution of simulation owners.
// Past the reach (two tiles from any crossing) the level set stands in, held
// at least six cells away. p: a point of a cell of a page that is not SOLID
// (a sample's position, a point of a stored cell, the centre of a stored
// face), read in that cell.
fn particleDepth(p:vec3f)->f32{
 let c=vec3u(p);let f=p-vec3f(c);
 if((nbBandReach(c)&1u)==0u){let far=bandPhi(p);return select(max(far,6.0),min(far,-6.0),far<0.0);}
 var value=0.0;
 // The eight vertex keys: where one block holds them (umVertexRun) the first's plus 1, 33 and 1089 a step, by one
 // page-table read; else umVertexAt of each, in a loop kept rolled (its bound is not a constant to the compiler:
 // run is zero there): eight inlined copies of umVertexAt's page search cost every caller registers, and a third
 // of a late splash's samples lie on a page plane (the floor sheet).
 var keys:array<u32,8>;let run=umVertexRun(vec3i(c),1u);
 if(run.x!=0u){let first=umVertexRunAt(run);for(var k=0u;k<8u;k++){let corner=umCorner(k,2u);keys[k]=first+corner.x+33u*corner.y+1089u*corner.z;}}
 else{for(var k=0u;k<8u+run.y;k++){keys[k]=umVertexAt(c+umCorner(k,2u));}}
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let w=select(1.0-f,f,corner!=vec3u(0));
  value+=w.x*w.y*w.z*bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+keys[k]]));
 }
 return value;
}
`;
