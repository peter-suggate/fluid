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
// coarse lattice vertices take the sampler's arithmetic on its 4h path (one
// or two loads on a lattice edge, else the ordered D4 sum of the cell's 4h
// corners), without the sampler: its bounds and owner search, which the cell
// here already settles, cost a 4h candidate tile's 117 such vertices most of
// depthSeeds.
// Set on the reconstruct group's crossing search alone: its phi is the
// scratch and its outputPhi the current field.
override nbColdCurrentPhi:bool=false;
fn nbCrossingVertex(tile:u32,origin:vec3u,p:vec3u)->f32{
 let cell=vec3u(umCellNearFromPage(tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));let key=umTileAt(cell/4u);let width=umTileWidth(key);
 // Reconstruction's search (nbColdCurrentPhi): a COLD h vertex reads the
 // current field (outputPhi, f.phi there), which transport leaves equal to
 // its scratch, so a COLD region need not store that scratch.
 if(width==1u){if(nbColdCurrentPhi&&nbTileCold(key)){return textureLoad(outputPhi,vec3i(p)).x;}return umLoadVertex(p);}
 if(all((p&vec3u(3u))==vec3u(0u))){return umLoadCorner(p);}
 let low=(cell/4u)*4u;let t=(vec3f(p)-vec3f(low))/4.0;
 {let zt=t==vec3f(0.0);if(u32(zt.x)+u32(zt.y)+u32(zt.z)>=2u){
  let a=select(select(2u,1u,!zt.y),0u,!zt.x);var e=vec3u(0);e[a]=4u;
  return umLoadCorner(low)*(1.0-t[a])+umLoadCorner(low+e)*t[a];
 }}
 let c8=umLoadCorner8(low,4u);var values:array<f32,8>;
 for(var k=0u;k<8u;k++){let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));values[k]=c8[k]*w.x*w.y*w.z;}
 return umVertexSum8(values);
}
// A 4h tile among 4h tiles interpolates its corners throughout: it holds a
// crossing only if they straddle zero. Beside an h tile its upper faces may
// be stored h vertices, so it is listed with the h tiles and decided there.
// Eligibility depends on ownership, moved-particle support and activity,
// all immutable for the three phi epochs of this frame. Build it once.
fn nbCrossingRequested(t:u32)->bool{
 return umTileLive(t)&&(umTileMinimumWidth(t)!=4u||atomicLoad(&bins[NB_SURFACE_TILES+t])!=0u||!(bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+t]))<=0.0));
}
var<workgroup> nbPlanMask:array<atomic<u32>,2>;
@compute @workgroup_size(64) fn depthPlanSeed(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var region=group.x;region<NB_REUSE_REGIONS;region+=groups.x){
  if(lane<2u){atomicStore(&nbPlanMask[lane],0u);}workgroupBarrier();
  let tile=nbReuseTile(region,lane);
  if(nbCrossingRequested(tile)){atomicOr(&nbPlanMask[lane>>5u],1u<<(lane&31u));}
  // Clear once for the whole frame; subsequent searches clear only this
  // plan's halo. Retired requests cannot leave stale masks or reach behind.
  atomicStore(&bins[NB_BAND_MASK+2u*tile],0u);atomicStore(&bins[NB_BAND_MASK+2u*tile+1u],0u);atomicStore(&bins[NB_BAND_REACH+tile],0u);
  workgroupBarrier();
  if(lane<2u){atomicStore(&bins[NB_FRAME_PLAN+4u+NB_REUSE_REGIONS+2u*region+lane],atomicLoad(&nbPlanMask[lane]));}
  if(lane==0u){
   atomicStore(&bins[NB_FRAME_PLAN+4u+region],u32((atomicLoad(&nbPlanMask[0])|atomicLoad(&nbPlanMask[1]))!=0u));
   atomicAdd(&bins[NB_FRAME_PLAN+2u],countOneBits(atomicLoad(&nbPlanMask[0]))+countOneBits(atomicLoad(&nbPlanMask[1])));
  }
  workgroupBarrier();
 }
}
var<workgroup> nbPlanHalo:atomic<u32>;
var<workgroup> nbAirPlanHalo:atomic<u32>;
@compute @workgroup_size(64) fn depthPlanExpand(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var region=group.x;region<NB_REUSE_REGIONS;region+=groups.x){
  if(lane==0u){atomicStore(&nbPlanHalo,0u);atomicStore(&nbAirPlanHalo,0u);}workgroupBarrier();
  let tile=nbReuseTile(region,lane);
  if(lane<27u&&umTileLive(tile)){
   let q=vec3i(umPageCoord(region))+vec3i(umCorner(lane,3u))-1;
   let key=umTileKey(4*q);if(umKeyStored(key)){atomicOr(&nbPlanHalo,atomicLoad(&bins[NB_FRAME_PLAN+4u+umPageOfTile(key)])&1u);}
  }
  // Air support reads up to five tiles from a changed seed. SurfaceHalo
  // includes phi closure changes; particle requests include cold arrivals.
  if(umTileLive(tile)){for(var k=lane;k<125u;k+=64u){
   let q=vec3i(umPageCoord(region))+vec3i(umCorner(k,5u))-2;let key=umTileKey(4*q);
   if(umKeyStored(key)){let r=umPageOfTile(key);
    let changed=(atomicLoad(&bins[NB_FRAME_PLAN+4u+4u*NB_REUSE_REGIONS+r])&8u)!=0u;
    let requested=(atomicLoad(&bins[NB_FRAME_PLAN+4u+r])&1u)!=0u;
    if(changed||requested){atomicStore(&nbAirPlanHalo,1u);}
   }
  }}
  workgroupBarrier();let included=atomicLoad(&nbPlanHalo)!=0u;
  if(lane==0u&&umTileLive(tile)&&((atomicLoad(&bins[NB_FRAME_PLAN+4u+4u*NB_REUSE_REGIONS+region])&8u)!=0u||(atomicLoad(&bins[NB_FRAME_PLAN+4u+region])&1u)!=0u)){
   let rank=atomicAdd(&bins[NB_FRAME_PLAN+4u+6u*NB_REUSE_REGIONS],1u);atomicStore(&bins[NB_FRAME_PLAN+5u+6u*NB_REUSE_REGIONS+rank],region);
  }
  if(lane==0u&&atomicLoad(&nbAirPlanHalo)!=0u){let rank=atomicAdd(&bins[NB_FRAME_PLAN+3u],1u);atomicStore(&bins[NB_FRAME_PLAN+4u+5u*NB_REUSE_REGIONS+rank],region);}
  if(lane==0u){
   if(included){
    atomicOr(&bins[NB_FRAME_PLAN+4u+region],2u);
    let rank=atomicAdd(&bins[NB_FRAME_PLAN+1u],1u);atomicStore(&bins[NB_FRAME_PLAN+4u+3u*NB_REUSE_REGIONS+rank],region);
   }else if(region<NB_REUSE_CAPACITY){
    // An omitted region cannot supply an old nearest/metric certificate
    // after its request set or world mapping changed.
    atomicStore(&bins[NB_REUSE_ALLOWED+region],0u);atomicStore(&bins[NB_REUSE_NEAREST_VALID+region],0u);atomicStore(&bins[NB_REUSE_NEAREST_FLAGS+region],0u);
    atomicStore(&bins[NB_REUSE_REGION_PLANE+8u*region+7u],0u);
   }
  }
  if(!included&&region<NB_REUSE_CAPACITY){
   atomicStore(&bins[NB_REUSE_VALID+64u*region+lane],0u);atomicStore(&bins[NB_REUSE_METRIC_VALID+64u*region+lane],0u);atomicStore(&bins[NB_REUSE_PLANE+384u*region+6u*lane+5u],0u);
  }
  workgroupBarrier();
 }
 if(group.x==0u&&lane==0u){atomicStore(&bins[NB_FRAME_PLAN],1u);}
}
fn nbDepthTile(t:u32){
  if(!umTileLive(t)){return;}
  // Pure 4h Eulerian surfaces neither reconstruct particles nor redistance
  // h vertices. Search only where fine geometry or live/activating samples
  // need it; the existing reach dilation supplies the inner overlap collar.
  if(nbPlanned()){
   let region=nbReuseRegion(t);let lane=nbReuseLocal(t);
   if((atomicLoad(&bins[NB_FRAME_PLAN+4u+NB_REUSE_REGIONS+2u*region+(lane>>5u)])&(1u<<(lane&31u)))==0u){return;}
  }else if(!nbCrossingRequested(t)){return;}
  var candidate=true;
  if(nbCrossingReuse(t)){
   let mask=nbCrossingCachedMask(t);if((mask.x|mask.y)==0u){return;}
  }else
  if(umTileMinimumWidth(t)==4u){
   let origin=4u*umTileCoord(t);var low=3.0e38;var high=-3.0e38;
   for(var k=0u;k<8u;k++){let value=nbCrossingVertex(t,origin,origin+4u*umCorner(k,2u));low=min(low,value);high=max(high,value);}
   candidate=low<=0.0&&high>=0.0;
  }
  if(candidate){let slot=atomicAdd(&bins[NB_BAND],1u);atomicStore(&bins[NB_BAND_CANDIDATES+slot],t);}
}
@compute @workgroup_size(64) fn depthTiles(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let regions=nbPlanRegions(lane);
 for(var slot=group.x;slot<regions;slot+=groups.x){
  let region=nbPlanRegion(slot,lane);let tile=nbReuseTile(region,lane);
  if(nbPlanned()){atomicStore(&bins[NB_BAND_MASK+2u*tile],0u);atomicStore(&bins[NB_BAND_MASK+2u*tile+1u],0u);atomicStore(&bins[NB_BAND_REACH+tile],0u);}
  nbPrepareReuse(region,lane,tile);nbDepthTile(tile);
  workgroupBarrier();
 }
}
var<workgroup> nbBandVertices:array<f32,125>;
// A tile's 125 samples select only eight positive-side owners. Resolve each
// owner once, including a page's high-wall clamp, and share the 3³ coarse
// corners covering their interpolation stencils. Fine and COLD-current reads
// retain their original homes; interpolation keeps nbCrossingVertex's sums.
var<workgroup> nbSeedOwner:array<vec4u,8>;
var<workgroup> nbSeedCold:array<u32,8>;
var<workgroup> nbSeedCorners:array<f32,27>;
fn nbSeedCorner(p:vec3u)->f32{return nbSeedCorners[p.x+3u*(p.y+3u*p.z)];}
fn nbGatherCrossing(tile:u32,origin:vec3u,lane:u32){
 if(lane<8u){
  let c=vec3u(umCellNearFromPage(tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(origin+4u*umCorner(lane,2u))));
  let key=umTileAt(c/4u);nbSeedOwner[lane]=vec4u((c/4u)*4u,umTileWidth(key));nbSeedCold[lane]=u32(nbColdCurrentPhi&&nbTileCold(key));
 }
 if(lane<27u){nbSeedCorners[lane]=umLoadCorner(origin+4u*umCorner(lane,3u));}
 workgroupBarrier();
 if(lane<125u){
  let local=umCorner(lane,5u);let p=origin+local;let high=vec3u(local==vec3u(4u));let owner=nbSeedOwner[high.x+2u*high.y+4u*high.z];var value=0.0;
  if(owner.w==1u){
   if(nbSeedCold[high.x+2u*high.y+4u*high.z]!=0u){value=textureLoad(outputPhi,vec3i(p)).x;}else{value=umLoadVertex(p);}
  }else if(all((p&vec3u(3u))==vec3u(0u))){value=nbSeedCorner(local/4u);}
  else{
   let low=owner.xyz;let base=(low-origin)/4u;let t=(vec3f(p)-vec3f(low))/4.0;let zero=t==vec3f(0);
   if(u32(zero.x)+u32(zero.y)+u32(zero.z)>=2u){
    let a=select(select(2u,1u,!zero.y),0u,!zero.x);var e=vec3u(0);e[a]=1u;
    value=nbSeedCorner(base)*(1.0-t[a])+nbSeedCorner(base+e)*t[a];
   }else{
    var terms:array<f32,8>;
    for(var k=0u;k<8u;k++){let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));terms[k]=nbSeedCorner(base+corner)*w.x*w.y*w.z;}
    value=umVertexSum8(terms);
   }
  }
  nbBandVertices[lane]=value;if((bitcast<u32>(value)&0x7fffffffu)>=0x7f800000u){atomicStore(&nbCrossingFinite,0u);}
 }
 workgroupBarrier();
}
var<workgroup> nbBandMask:array<atomic<u32>,2>;
var<workgroup> nbSeedReuse:u32;
var<workgroup> nbCrossingFinite:atomic<u32>;
@compute @workgroup_size(128) fn depthSeeds(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_CANDIDATES+group.x]);umLoadAddressPlan(tile>>9u,lane);let coord=umTileCoord(tile);let origin=4u*coord;
 if(lane==0u){
  nbSeedReuse=select(2u,1u,nbCrossingReuse(tile));
  atomicStore(&nbCrossingFinite,1u);
 }
 let decision=workgroupUniformLoad(&nbSeedReuse);let cached=(decision&1u)!=0u;
 let gather=(decision&2u)!=0u;
 if(cached){if(lane<2u){atomicStore(&nbBandMask[lane],nbCrossingCachedMask(tile)[lane]);}}
 if(gather){
  nbGatherCrossing(tile,origin,lane);
 }
 if(!cached){
  if(lane<2u){atomicStore(&nbBandMask[lane],0u);}
  workgroupBarrier();
  if(lane<64u){
   let l=umCorner(lane,4u);var low=3.0e38;var high=-3.0e38;
   for(var k=0u;k<8u;k++){let v=l+umCorner(k,2u);let value=nbBandVertices[v.x+5u*(v.y+5u*v.z)];low=min(low,value);high=max(high,value);}
   // Exact zero belongs to either incident cell; flat plateaus and walls do not.
   if(low<=0.0&&high>=0.0&&low<high&&umCellOpen(vec3i(origin+l))>1e-5){atomicOr(&nbBandMask[l.z>>1u],1u<<(l.x+4u*l.y+16u*(l.z&1u)));}
  }
 }

 workgroupBarrier();
 let mask=vec2u(atomicLoad(&nbBandMask[0]),atomicLoad(&nbBandMask[1]));
 if(lane==0u&&atomicLoad(&nbCrossingFinite)!=0u){nbCommitCrossing(tile,mask);}
 if((mask.x|mask.y)==0u){return;}
 if(lane<2u){atomicStore(&bins[NB_BAND_MASK+2u*tile+lane],mask[lane]);}
 for(var k=lane;k<343u;k+=128u){
  let o=vec3i(umCorner(k,7u))-3;let t=vec3i(coord)+o;
  let key=umTileKey(t);if(key>=UM_TILES){continue;}
  atomicOr(&bins[NB_BAND_REACH+key],select(3u,2u,any(abs(o)>vec3i(2))));
 }
}
fn nbDepthList(t:u32){
 if(!umTileLive(t)){return;}
 let reach=atomicLoad(&bins[NB_BAND_REACH+t]);
 if((reach&1u)!=0u){let slot=atomicAdd(&bins[NB_BAND+1u],1u);atomicStore(&bins[NB_BAND_SEARCH+slot],t);}
 if((reach&2u)!=0u){let slot=atomicAdd(&bins[NB_BAND+2u],1u);atomicStore(&bins[NB_BAND_DISTANCE+slot],t);}
}
@compute @workgroup_size(64) fn depthLists(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){nbDepthList(t);}
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
 // page-table read. Where the vertices' own pages (the pages of the cells below them) span two pages on one axis
 // (a third of a late splash's samples lie on a page plane: the floor sheet), both pages' words are read once and
 // each key is a select of them: a vertex whose own page is not stored on the face between them is the upper
 // page's skirt (local 0) where that page is stored, as umVertexSlot walks. Else umVertexAt of each, in a loop
 // kept rolled (its bound is not a constant to the compiler: run is zero there): eight inlined copies of
 // umVertexAt's page search cost every caller registers.
 var keys:array<u32,8>;let run=umVertexRun(vec3i(c),1u);
 if(run.x!=0u){let first=umVertexRunAt(run);for(var k=0u;k<8u;k++){let corner=umCorner(k,2u);keys[k]=first+corner.x+33u*corner.y+1089u*corner.z;}}
 else{
  let q=vec3i(c);let page=(q-vec3i(1))>>vec3u(5u);let cross=vec3i(((q+vec3i(1))>>vec3u(5u))!=page);
  if(cross.x+cross.y+cross.z==1){
   let w0=umPageWord(page);let w1=umPageWord(page+cross);let walk=!umBlockStored(w0)&&umBlockStored(w1);
   // Keys of local (1,1,1) in each block; the skirt is 32 planes below local 32 on the crossing axis.
   let k0=35937u*min(w0,UM_BLOCKS-1u)+1123u;let k1=35937u*min(w1,UM_BLOCKS-1u)+1123u;let skirt=u32(dot(cross,vec3i(32,1056,34848)));
   for(var k=0u;k<8u;k++){
    let at=q+vec3i(umCorner(k,2u))-vec3i(1);let l=at&vec3i(31);let o=u32(l.x+33*l.y+1089*l.z);
    let upper=any((at>>vec3u(5u))!=page);let face=walk&&any((l==vec3i(31))&(cross!=vec3i(0)));
    keys[k]=select(select(k0+o,k1+o-skirt,face),k1+o,upper);
   }
  }else{for(var k=0u;k<8u+run.y;k++){keys[k]=umVertexAt(c+umCorner(k,2u));}}
 }
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let w=select(1.0-f,f,corner!=vec3u(0));
  value+=w.x*w.y*w.z*bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+keys[k]]));
 }
 return value;
}
`;
