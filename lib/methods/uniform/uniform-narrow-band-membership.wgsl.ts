/** Shared bounded crossing search, over the tiles the surface reaches.
 * A flat pass lists the tiles that can hold a crossing cell; each writes its
 * crossings as a 64-bit mask and marks the tiles within three of it. A cell
 * has a crossing within five cells only in a tile within two of one, so the
 * search runs on those alone: a tile-local separable squared-distance
 * minimization (11 taps per axis) over the masks, in workgroup memory, that
 * finds the exact nearest crossing-cell centre. Reconstruction uses its
 * cell-box guard against stale bulk phi. Redistance then writes nodal metric
 * distances (bank A) on the tiles within three; particle membership samples
 * that field independently of h/4h ownership. Outside the reach nothing is
 * stored: a reader there has the level set alone, no nearer than six cells. */
export const narrowBandMembershipWGSL=/* wgsl */`
const NB_DEPTH_A:u32=NB_SURFACE_TILES+UM_T.x*UM_T.y*UM_T.z;
const NB_DEPTH_B:u32=NB_DEPTH_A+(UM_D.x+1u)*(UM_D.y+1u)*(UM_D.z+1u);
const NB_NO_SURFACE:u32=0xffffffffu;
// Tile banks of the search, rebuilt with it: three list counts (candidates,
// search, distance), a crossing mask of two words a tile (bit x+4y+16z), the
// reach (1: within two tiles of a crossing, 2: within three), the lists.
// The fourth count and list are the transfer's: its 4h owners with a gather.
const NB_BAND_TILES:u32=UM_T.x*UM_T.y*UM_T.z;
const NB_BAND:u32=NB_DEPTH_B+UM_D.x*UM_D.y*UM_D.z;
const NB_BAND_MASK:u32=NB_BAND+4u;
const NB_BAND_REACH:u32=NB_BAND_MASK+2u*NB_BAND_TILES;
const NB_BAND_CANDIDATES:u32=NB_BAND_REACH+NB_BAND_TILES;
const NB_BAND_SEARCH:u32=NB_BAND_CANDIDATES+NB_BAND_TILES;
const NB_BAND_DISTANCE:u32=NB_BAND_SEARCH+NB_BAND_TILES;
const NB_BAND_SEAM:u32=NB_BAND_DISTANCE+NB_BAND_TILES;
fn nbCell(i:u32)->vec3u{return vec3u(i%UM_D.x,(i/UM_D.x)%UM_D.y,i/(UM_D.x*UM_D.y));}
fn nbBandTile(c:vec3u)->u32{let t=c/4u;return t.x+UM_T.x*(t.y+UM_T.y*t.z);}
fn nbBandReach(c:vec3u)->u32{return atomicLoad(&bins[NB_BAND_REACH+nbBandTile(c)]);}
fn nbBandCrossing(c:vec3u)->bool{
 let l=c&vec3u(3u);
 return ((atomicLoad(&bins[NB_BAND_MASK+2u*nbBandTile(c)+(l.z>>1u)])>>(l.x+4u*l.y+16u*(l.z&1u)))&1u)!=0u;
}
// A 4h tile among 4h tiles interpolates its corners throughout: it holds a
// crossing only if they straddle zero. Beside an h tile its upper faces may
// be stored h vertices, so it is listed with the h tiles and decided there.
@compute @workgroup_size(64) fn depthTiles(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){
  var candidate=true;
  if(umTileMinimumWidth(t)==4u){
   let origin=4u*umTileCoord(t);var low=3.0e38;var high=-3.0e38;
   for(var k=0u;k<8u;k++){let value=umSampleVertex(vec3f(origin+4u*umCorner(k,2u)));low=min(low,value);high=max(high,value);}
   candidate=low<=0.0&&high>=0.0;
  }
  if(candidate){let slot=atomicAdd(&bins[NB_BAND],1u);atomicStore(&bins[NB_BAND_CANDIDATES+slot],t);}
 }
}
var<workgroup> nbBandVertices:array<f32,125>;
var<workgroup> nbBandMask:array<atomic<u32>,2>;
@compute @workgroup_size(128) fn depthSeeds(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_CANDIDATES+group.x]);let coord=umTileCoord(tile);let origin=4u*coord;
 if(lane<125u){nbBandVertices[lane]=umSampleVertex(vec3f(origin+umCorner(lane,5u)));}
 if(lane<2u){atomicStore(&nbBandMask[lane],0u);}
 workgroupBarrier();
 if(lane<64u){
  let l=umCorner(lane,4u);var low=3.0e38;var high=-3.0e38;
  for(var k=0u;k<8u;k++){let v=l+umCorner(k,2u);let value=nbBandVertices[v.x+5u*(v.y+5u*v.z)];low=min(low,value);high=max(high,value);}
  // Exact zero belongs to either incident cell; a flat zero plateau alone
  // is not an interface. Domain and solid walls are not free surfaces.
  if(low<=0.0&&high>=0.0&&low<high&&umCellOpen(vec3i(origin+l))>=0.5){atomicOr(&nbBandMask[l.z>>1u],1u<<(l.x+4u*l.y+16u*(l.z&1u)));}
 }
 workgroupBarrier();
 let mask=vec2u(atomicLoad(&nbBandMask[0]),atomicLoad(&nbBandMask[1]));
 if((mask.x|mask.y)==0u){return;}
 if(lane<2u){atomicStore(&bins[NB_BAND_MASK+2u*tile+lane],mask[lane]);}
 for(var k=lane;k<343u;k+=128u){
  let o=vec3i(umCorner(k,7u))-3;let t=vec3i(coord)+o;
  if(any(t<vec3i(0))||any(t>=vec3i(UM_T))){continue;}
  atomicOr(&bins[NB_BAND_REACH+u32(t.x)+UM_T.x*(u32(t.y)+UM_T.y*u32(t.z))],select(3u,2u,any(abs(o)>vec3i(2))));
 }
}
@compute @workgroup_size(64) fn depthLists(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){
  let reach=atomicLoad(&bins[NB_BAND_REACH+t]);
  if((reach&1u)!=0u){let slot=atomicAdd(&bins[NB_BAND+1u],1u);atomicStore(&bins[NB_BAND_SEARCH+slot],t);}
  if((reach&2u)!=0u){let slot=atomicAdd(&bins[NB_BAND+2u],1u);atomicStore(&bins[NB_BAND_DISTANCE+slot],t);}
 }
}
// The crossings of one x row, 14 cells from five below the tile: its own
// four and five either side, read from the masks of the five tiles across.
fn nbBandRow(coord:vec3u,y:i32,z:i32)->u32{
 if(y<0||z<0||y>=i32(UM_D.y)||z>=i32(UM_D.z)){return 0u;}
 let shift=4u*(u32(y)&3u)+16u*(u32(z)&1u);let word=(u32(z)&3u)>>1u;var bits=0u;
 for(var o=-2;o<=2;o++){
  let x=i32(coord.x)+o;if(x<0||x>=i32(UM_T.x)){continue;}
  let nibble=(atomicLoad(&bins[NB_BAND_MASK+2u*(u32(x)+UM_T.x*(u32(y)/4u+UM_T.y*(u32(z)/4u)))+word])>>shift)&15u;
  if(o==-2){bits|=nibble>>3u;}else{bits|=nibble<<u32(4*o+5);}
 }
 return bits&0x3fffu;
}
// One byte per cell of a tile row: the x offset of its nearest crossing plus
// five, 255 for none. Then one per cell of a tile column: that offset and
// the y offset plus five in the high nibble. Equal distances keep the lower
// offset, x before y before z.
var<workgroup> nbBandRows:array<u32,196>;
var<workgroup> nbBandColumns:array<u32,56>;
@compute @workgroup_size(64) fn depthNearest(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+group.x]);let coord=umTileCoord(tile);let origin=vec3i(4u*coord);
 for(var row=lane;row<196u;row+=64u){
  let bits=nbBandRow(coord,origin.y+i32(row%14u)-5,origin.z+i32(row/14u)-5);var packed=0xffffffffu;
  if(bits!=0u){
   packed=0u;
   for(var x=0u;x<4u;x++){
    var code=255u;
    for(var d=0u;d<=5u;d++){
     if(((bits>>(x+5u-d))&1u)!=0u){code=5u-d;break;}
     if(((bits>>(x+5u+d))&1u)!=0u){code=5u+d;break;}
    }
    packed|=code<<(8u*x);
   }
  }
  nbBandRows[row]=packed;
 }
 workgroupBarrier();
 if(lane<56u){
  let y=lane%4u;let z=lane/4u;var packed=0u;
  for(var x=0u;x<4u;x++){
   var best=255u;var distance=1000;
   for(var o=-5;o<=5;o++){
    let code=(nbBandRows[u32(i32(y)+o+5)+14u*z]>>(8u*x))&255u;if(code==255u){continue;}
    let dx=i32(code)-5;let d=dx*dx+o*o;
    if(d<distance){distance=d;best=code|(u32(o+5)<<4u);}
   }
   packed|=best<<(8u*x);
  }
  nbBandColumns[lane]=packed;
 }
 workgroupBarrier();
 let l=umCorner(lane,4u);var nearest=NB_NO_SURFACE;var distance=1000;
 for(var o=-5;o<=5;o++){
  let code=(nbBandColumns[l.y+4u*u32(i32(l.z)+o+5)]>>(8u*l.x))&255u;if(code==255u){continue;}
  let delta=vec3i(i32(code&15u)-5,i32(code>>4u)-5,o);let d=dot(delta,delta);
  if(d<distance){distance=d;nearest=cellIndex(origin+vec3i(l)+delta);}
 }
 atomicStore(&bins[NB_DEPTH_B+cellIndex(origin+vec3i(l))],nearest);
}
fn bulkDepth(p:vec3f)->f32{
 let phiValue=bandPhi(p);if(phiValue>=0.0){return phiValue;}
 let c=clamp(vec3i(floor(p)),vec3i(0),vec3i(UM_D)-1);
 if((nbBandReach(vec3u(c))&1u)==0u){return min(phiValue,-8.0);}
 let nearest=atomicLoad(&bins[NB_DEPTH_B+cellIndex(c)]);
 if(nearest==NB_NO_SURFACE){return min(phiValue,-8.0);}
 let low=vec3f(nbCell(nearest));let delta=max(max(low-p,p-(low+1.0)),vec3f(0));
 return min(phiValue,-length(delta));
}
// Bank A is the nodal metric field redistance leaves on the reached tiles.
// Particle membership does not depend on the resolution of simulation owners.
// Past the reach the level set stands in, held at least six cells away as
// the metric field holds a vertex with no crossing near it.
fn nbVertexIndex(p:vec3u)->u32{return p.x+(UM_D.x+1u)*(p.y+(UM_D.y+1u)*p.z);}
fn particleDepth(p:vec3f)->f32{
 let q=clamp(p,vec3f(0),vec3f(UM_D));let c=min(vec3u(floor(q)),UM_D-1u);let f=q-vec3f(c);
 if((nbBandReach(c)&1u)==0u){let far=bandPhi(q);return select(max(far,6.0),min(far,-6.0),far<0.0);}
 var value=0.0;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let w=select(1.0-f,f,corner!=vec3u(0));
  value+=w.x*w.y*w.z*bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+nbVertexIndex(c+corner)]));
 }
 return value;
}
`;
