/** The surface brick arena (docs/plans/surface-brick-arena-2026-10-19.md),
 * phase 1: the end-step crossing search and fine redistance on dense 8³-cell
 * bricks of the shared brick table (uniform-brick-table.ts). A brick's
 * window is its 9³ represented vertices (8g .. 8g+8 an axis: every corner of
 * its cells), gathered once from paged phi into the links buffer from
 * NB_ARENA, slot·729.
 *
 * arenaMark: a thread a tile; a crossing candidate (depthTiles' rule) marks
 * its brick arena core; the table build rings it (class arena).
 * arenaGather (G): paged phi → the window, the value nbCrossingVertex gives
 * each vertex from its positive-side cell's owner. @derived after retire:
 * only what the coarse stores move (4h corners, interpolated and hanging
 * vertices, by the resolve's formula), so no phi resolve runs between.
 * arenaCross (X): core bricks; the crossing masks of their candidate tiles
 * from the window in workgroup memory, and the reach of their crossing tiles.
 * depthLists then lists the search and distance tiles as before.
 * arenaNearest: the exact nearest crossing cell of each cell (bank B) of a
 * reach-1 tile, separable in workgroup memory with depthNearest*'s ties.
 * arenaDistance (R): bank A of the vertices whose cell lies in the brick, in
 * a reach-2 tile: the nearest represented zero of the nearest crossing cells
 * of its eight incident cells, from their bricks' windows.
 * arenaFar: bank A of reach-2 tiles outside the arena (eight cells or more
 * from any crossing candidate): today's no-crossing value.
 * arenaStore (W): the fine redistance store; changed values only. */
import { UNIFORM_COMPILED_TOPOLOGY } from "./uniform-compiled-topology";
export const narrowBandArenaWGSL=/* wgsl */`
override nbArenaDerived:bool=false;
// The brick list is complete; publication can reuse its dead mark words.
override nbArenaClearPublication:bool=false;
// The redistance's reach in tiles from a crossing tile (surfaceRedistanceReach): reach 3 (bits 1 and 2) to tiles within min(2, it), reach 2 at three.
override nbReachTiles:u32=3u;
const NB_ARENA_WINDOW:u32=729u;
var<workgroup> nbArenaWord:u32;
// lane 0's value, uniform to every lane.
fn nbArenaUniform(value:u32,lane:u32)->u32{workgroupBarrier();if(lane==0u){nbArenaWord=value;}return workgroupUniformLoad(&nbArenaWord);}
// umVertexAt(p) of a vertex whose cell below (p-1) lies in page page of block
// block: no table read (umVertexSlot's first answer).
fn nbArenaVertexKey(p:vec3u,block:u32,page:vec3i)->u32{
 let c=vec3i(p)-1;if(any((c>>vec3u(5u))!=page)){return umVertexAt(p);}
 let own=vec3u(c&vec3i(31))+1u;return 35937u*block+own.x+33u*(own.y+33u*own.z);
}
// The arena: windows in the links buffer from NB_ARENA.
fn nbArenaLoad(i:u32)->f32{return bitcast<f32>(links[NB_ARENA+i]);}
fn nbArenaH()->f32{return min(params.hDt.x,min(params.hDt.y,params.hDt.z));}
@compute @workgroup_size(64) fn arenaMark(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var t=gid.x;t<UM_TILES;t+=64u*groups.x){
  if(!nbCrossingRequested(t)){continue;}
  // A 4h tile among 4h tiles holds a crossing only if its corners straddle zero.
  if(umTileMinimumWidth(t)==4u){
   let origin=4u*umTileCoord(t);var low=3.0e38;var high=-3.0e38;
   for(var k=0u;k<8u;k++){let value=nbCrossingVertex(t,origin,origin+4u*umCorner(k,2u));low=min(low,value);high=max(high,value);}
   if(!(low<=0.0&&high>=0.0)){continue;}
  }
  ubMark(ubBrickOfTile(t),UB_ARENA_CORE);
 }
}
// G. Workgroup: the 27 tiles of the window's cells (key, width | seam<<8) and the 5³ 4h corners from the tile below.
var<workgroup> nbArenaTile:array<vec2u,27>;
var<workgroup> nbArenaCorner:array<f32,125>;
fn nbArenaCornerAt(q:vec3u,t0:vec3u)->f32{let n=q+1u-t0;return nbArenaCorner[n.x+5u*(n.y+5u*n.z)];}
@compute @workgroup_size(256) fn arenaGather(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=nbArenaUniform(ubCount(),lane);
 for(var slot=group.x;slot<count;slot+=groups.x){
  if((nbArenaUniform(ubClass(slot),lane)&UB_ARENA)==0u){continue;}
  let id=ubBrick(slot);
  if(nbArenaClearPublication&&lane==0u){atomicStore(&bins[UB_MARK+id],0u);}
  let block=id>>6u;let origin=ubBrickOrigin(id);let page=vec3i(origin>>vec3u(5u));let t0=origin/4u;
  umLoadAddressPlan(block,lane);
  if(lane<27u){let t=umTileAt(t0+umCorner(lane,3u));nbArenaTile[lane]=vec2u(t,umTileWidth(t)|select(0u,256u,umTileMaximumWidth(t)!=umTileMinimumWidth(t)));}
  if(lane<125u){let q=vec3i(t0)+vec3i(umCorner(lane,5u))-1;nbArenaCorner[lane]=select(0.0,umLoadCorner(vec3u(4*q)),all(q>=vec3i(0)));}
  workgroupBarrier();
  // The h texel of the brick's local 0: an owned vertex (every index 1..8) is that plus its index.
  let texel=33*umBlockCoord(block)+vec3i(origin&vec3u(31u));
  for(var k=lane;k<NB_ARENA_WINDOW;k+=256u){
   let l=umCorner(k,9u);let p=origin+l;
   var c=p;if(any(l==vec3u(8u))){c=vec3u(umCellNearFromPage(block,page,vec3i(p)));}
   let tc=c/4u;let r=tc-t0;let info=nbArenaTile[r.x+3u*(r.y+3u*r.z)];
   let aligned=all((p&vec3u(3u))==vec3u(0u));var value=0.0;var write=true;
   if((info.y&255u)==1u){
    if(nbArenaDerived){
     // A hanging vertex of a seam tile: the phi resolve's sum of its 4h corners.
     var incident=0u;if((info.y&256u)!=0u&&!aligned){incident=umCoarseIncident(info.x,p-4u*tc);}
     if(incident!=0u){
      let at=umCorner(firstTrailingBit(incident),2u);let local=p-4u*tc;
      let t=vec3f(local+(vec3u(1)-at)*4u)/4.0;var values:array<f32,8>;
      for(var j=0u;j<8u;j++){
       let corner=umCorner(j,2u);let w=umVertexWeight(t,corner);let m=at+corner;
       if(w>0.0){values[j]=w*nbArenaCornerAt(tc+m-1u,t0);}
      }
      value=umVertexSum8(values);
     }else if(aligned){value=umLoadVertex(p);}else{write=false;}
    }else if(all(l!=vec3u(0u))){value=udTexel_phi(texel+vec3i(l)).x;}
    else{value=umLoadVertex(p);}
   }else if(aligned){value=nbArenaCornerAt(p/4u,t0);}
   else{
    // nbCrossingVertex's 4h arithmetic, from the cached corners.
    let low=tc*4u;let t=(vec3f(p)-vec3f(low))/4.0;let zero=t==vec3f(0.0);
    if(u32(zero.x)+u32(zero.y)+u32(zero.z)>=2u){
     let a=select(select(2u,1u,!zero.y),0u,!zero.x);var e=vec3u(0);e[a]=1u;
     value=nbArenaCornerAt(tc,t0)*(1.0-t[a])+nbArenaCornerAt(tc+e,t0)*t[a];
    }else{
     var values:array<f32,8>;
     for(var j=0u;j<8u;j++){let corner=umCorner(j,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));values[j]=nbArenaCornerAt(tc+corner,t0)*w.x*w.y*w.z;}
     value=umVertexSum8(values);
    }
   }
   if(write){links[NB_ARENA+NB_ARENA_WINDOW*slot+k]=bitcast<u32>(value);}
  }
 }
}
// X and R stage the window.
var<workgroup> nbArenaWindow:array<f32,729>;
fn nbArenaVertex(l:vec3u)->f32{return nbArenaWindow[l.x+9u*(l.y+9u*l.z)];}
var<workgroup> nbArenaMask:array<atomic<u32>,16>;
var<workgroup> nbArenaCandidate:array<u32,8>;
var<workgroup> nbArenaCrossed:atomic<u32>;
@compute @workgroup_size(256) fn arenaCross(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=nbArenaUniform(ubCount(),lane);
 for(var slot=group.x;slot<count;slot+=groups.x){
  if((nbArenaUniform(ubClass(slot),lane)&UB_ARENA_CORE)==0u){continue;}
  let id=ubBrick(slot);let origin=ubBrickOrigin(id);let t0=origin/4u;
  for(var k=lane;k<NB_ARENA_WINDOW;k+=256u){nbArenaWindow[k]=nbArenaLoad(NB_ARENA_WINDOW*slot+k);}
  if(lane<16u){atomicStore(&nbArenaMask[lane],0u);}if(lane==0u){atomicStore(&nbArenaCrossed,0u);}
  workgroupBarrier();
  if(lane<8u){
   let t=umTileAt(t0+umCorner(lane,2u));var candidate=nbCrossingRequested(t);
   if(candidate&&umTileMinimumWidth(t)==4u){
    var low=3.0e38;var high=-3.0e38;
    for(var k=0u;k<8u;k++){let value=nbArenaVertex(4u*(umCorner(lane,2u)+umCorner(k,2u)));low=min(low,value);high=max(high,value);}
    candidate=low<=0.0&&high>=0.0;
   }
   nbArenaCandidate[lane]=select(UM_TILES,t,candidate);
  }
  workgroupBarrier();
  for(var k=lane;k<512u;k+=256u){
   let l=umCorner(k,8u);let r=l/4u;let tile=r.x|(r.y<<1u)|(r.z<<2u);
   if(nbArenaCandidate[tile]==UM_TILES){continue;}
   var low=3.0e38;var high=-3.0e38;
   for(var j=0u;j<8u;j++){let value=nbArenaVertex(l+umCorner(j,2u));low=min(low,value);high=max(high,value);}
   // Exact zero belongs to either incident cell; flat plateaus and walls do not.
   let m=l&vec3u(3u);
   if(low<=0.0&&high>=0.0&&low<high&&umCellOpen(vec3i(origin+l))>1e-5){atomicOr(&nbArenaMask[2u*tile+(m.z>>1u)],1u<<(m.x+4u*m.y+16u*(m.z&1u)));}
  }
  workgroupBarrier();
  if(lane<8u){
   let t=nbArenaCandidate[lane];let mask=vec2u(atomicLoad(&nbArenaMask[2u*lane]),atomicLoad(&nbArenaMask[2u*lane+1u]));
   if(t!=UM_TILES&&(mask.x|mask.y)!=0u){
    atomicStore(&bins[NB_BAND_MASK+2u*t],mask.x);atomicStore(&bins[NB_BAND_MASK+2u*t+1u],mask.y);atomicOr(&nbArenaCrossed,1u<<lane);
   }
  }
  workgroupBarrier();
  // The reach of the brick's crossing tiles, a target tile once: 3 within two tiles of one (nbReachTiles 1: one), 2 at three (nbReachTiles 3).
  let crossed=atomicLoad(&nbArenaCrossed);
  if(crossed!=0u){for(var k=lane;k<512u;k+=256u){
   let r=vec3i(umCorner(k,8u))-3;var reach=0u;
   for(var j=0u;j<8u;j++){
    if((crossed&(1u<<j))==0u){continue;}
    let o=abs(r-vec3i(umCorner(j,2u)));let d=max(o.x,max(o.y,o.z));
    if(d<=min(2,i32(nbReachTiles))){reach=3u;}else if(d==3&&nbReachTiles==3u){reach|=2u;}
   }
   if(reach==0u){continue;}
   let key=umTileKey(vec3i(t0)+r);if(key>=UM_TILES){continue;}
   if((atomicLoad(&bins[NB_BAND_REACH+key])&reach)!=reach){atomicOr(&bins[NB_BAND_REACH+key],reach);}
  }}
 }
}
// Nearest crossing cell (bank B). Rows: the 18×18 (y,z) rows from five
// cells below the brick; masks of the 6³ tiles from two below.
var<workgroup> nbArenaKey:array<u32,216>;
var<workgroup> nbArenaMasks:array<vec2u,216>;
var<workgroup> nbArenaX:array<u32,648>;
var<workgroup> nbArenaY:array<u32,288>;
var<workgroup> nbArenaOwnReach:array<u32,8>;
fn nbArenaCode(words:u32,x:u32)->u32{return (words>>(8u*(x&3u)))&255u;}
@compute @workgroup_size(256) fn arenaNearest(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=nbArenaUniform(ubCount(),lane);
 for(var slot=group.x;slot<count;slot+=groups.x){
  if((nbArenaUniform(ubClass(slot),lane)&UB_ARENA)==0u){continue;}
  let id=ubBrick(slot);let origin=ubBrickOrigin(id);let t0=origin/4u;
  if(lane<216u){
   let key=umTileKey(vec3i(t0)+vec3i(umCorner(lane,6u))-2);nbArenaKey[lane]=key;var mask=vec2u(0u);
   if(key<UM_TILES){mask=vec2u(atomicLoad(&bins[NB_BAND_MASK+2u*key]),atomicLoad(&bins[NB_BAND_MASK+2u*key+1u]));}
   nbArenaMasks[lane]=mask;
  }
  // The reach of the brick's own tiles.
  if(lane<8u){let key=umTileKey(vec3i(t0+umCorner(lane,2u)));nbArenaOwnReach[lane]=select(0u,atomicLoad(&bins[NB_BAND_REACH+key]),key<UM_TILES);}
  workgroupBarrier();
  // X: the nearest crossing along x within five, ties to the left (depthNearestX).
  for(var row=lane;row<324u;row+=256u){
   let ry=i32(row%18u)-5;let rz=i32(row/18u)-5;let ty=u32(2+(ry>>2u));let tz=u32(2+(rz>>2u));
   let shift=4u*u32(ry&3)+16u*u32(rz&1);let word=u32(rz&3)>>1u;
   // The row's cells from eight below the brick: bit i, cell origin.x - 8 + i.
   var bits=0u;
   if(nbArenaKey[2u+6u*ty+36u*tz]<UM_TILES){for(var tx=0u;tx<6u;tx++){bits|=((nbArenaMasks[tx+6u*ty+36u*tz][word]>>shift)&15u)<<(4u*tx);}}
   var packed=vec2u(0xffffffffu);
   if(bits!=0u){
    packed=vec2u(0u);
    for(var x=0u;x<8u;x++){
     let centre=x+8u;let left=bits&((1u<<(centre+1u))-1u);let right=bits>>centre;
     let dl=select(255u,centre-firstLeadingBit(left),left!=0u);let dr=select(255u,firstTrailingBit(right),right!=0u);
     let delta=min(dl,dr);let code=select(255u,select(5u+dr,5u-dl,dl<=dr),delta<=5u);
     packed[x>>2u]|=code<<(8u*(x&3u));
    }
   }
   nbArenaX[2u*row]=packed.x;nbArenaX[2u*row+1u]=packed.y;
  }
  workgroupBarrier();
  // Y: the brick's 8 rows of y, every z; strict, lower offsets first (depthNearestY).
  for(var job=lane;job<288u;job+=256u){
   let half=job&1u;let row=job>>1u;let yi=row%8u;let rz=row/8u;var packed=0u;
   for(var x=4u*half;x<4u*half+4u;x++){
    var best=255u;var distance=1000;
    for(var o=-5;o<=5;o++){
     let code=nbArenaCode(nbArenaX[2u*(u32(i32(yi)+5+o)+18u*rz)+half],x);
     let dx=i32(code)-5;let d=dx*dx+o*o;
     if(code!=255u&&d<distance){best=code|(u32(o+5)<<4u);distance=d;}
    }
    packed|=best<<(8u*(x&3u));
   }
   nbArenaY[job]=packed;
  }
  workgroupBarrier();
  // Z: every cell of the brick in a reach-1 tile (depthNearestZ).
  for(var k=lane;k<512u;k+=256u){
   let l=umCorner(k,8u);let cell=origin+l;let r=l/4u+2u;let tile=nbArenaKey[r.x+6u*r.y+36u*r.z];
   let own=l/4u;if(tile>=UM_TILES||(nbArenaOwnReach[own.x|(own.y<<1u)|(own.z<<2u)]&1u)==0u){continue;}
   var distance=1000;var best=vec3i(0);
   for(var o=-5;o<=5;o++){
    let code=nbArenaCode(nbArenaY[2u*(l.y+8u*u32(i32(l.z)+5+o))+(l.x>>2u)],l.x);
    if(code==255u){continue;}
    let dx=i32(code&15u)-5;let dy=i32(code>>4u)-5;let d=dx*dx+dy*dy+o*o;
    if(d<distance){distance=d;best=vec3i(cell)+vec3i(dx,dy,o);}
   }
   var nearest=NB_NO_SURFACE;if(distance<1000){nearest=umCellAt(vec3u(best));}
   atomicStore(&bins[NB_DEPTH_B+64u*tile+((l.x&3u)|((l.y&3u)<<2u)|((l.z&3u)<<4u))],nearest);
  }
 }
}
// R's cell distance: nbCellDistance (the zero of the trilinear cell of
// corners v, phi/h, from q in cell units; -1 where it holds none) with its
// twelve edges unrolled and the clamped Newton solve stopped once converged.
fn nbArenaEdge(q:vec3f,a:f32,b:f32,corner:vec3f,axis:u32,distance:f32)->f32{
 if(a*b>0.0||a==b){return distance;}
 var at=corner;at[axis]+=clamp(a/(a-b),0.0,1.0);
 return min(distance,length(q-at));
}
fn nbArenaCellDistance(q:vec3f,v:array<f32,8>)->f32{
 let low=min(min(min(v[0],v[1]),min(v[2],v[3])),min(min(v[4],v[5]),min(v[6],v[7])));
 let high=max(max(max(v[0],v[1]),max(v[2],v[3])),max(max(v[4],v[5]),max(v[6],v[7])));
 if(low>0.0||high<0.0||low==high){return -1.0;}
 var d=length(q)+sqrt(3.0);
 d=nbArenaEdge(q,v[0],v[1],vec3f(0,0,0),0u,d);d=nbArenaEdge(q,v[2],v[3],vec3f(0,1,0),0u,d);
 d=nbArenaEdge(q,v[4],v[5],vec3f(0,0,1),0u,d);d=nbArenaEdge(q,v[6],v[7],vec3f(0,1,1),0u,d);
 d=nbArenaEdge(q,v[0],v[2],vec3f(0,0,0),1u,d);d=nbArenaEdge(q,v[1],v[3],vec3f(1,0,0),1u,d);
 d=nbArenaEdge(q,v[4],v[6],vec3f(0,0,1),1u,d);d=nbArenaEdge(q,v[5],v[7],vec3f(1,0,1),1u,d);
 d=nbArenaEdge(q,v[0],v[4],vec3f(0,0,0),2u,d);d=nbArenaEdge(q,v[1],v[5],vec3f(1,0,0),2u,d);
 d=nbArenaEdge(q,v[2],v[6],vec3f(0,1,0),2u,d);d=nbArenaEdge(q,v[3],v[7],vec3f(1,1,0),2u,d);
 // A converged step moves r by under 1e-6 of a cell: stop there.
 var r=clamp(q,vec3f(0),vec3f(1));var f=nbTrilinear(v,r);
 for(var i=0u;i<6u;i++){
  if(abs(f.w)<1e-6){break;}
  let g2=dot(f.xyz,f.xyz);if(g2<1e-12){break;}
  // A clamped step that stays put stays put.
  let next=clamp(r-f.w*f.xyz/g2,vec3f(0),vec3f(1));if(all(next==r)){break;}
  r=next;f=nbTrilinear(v,r);
 }
 if(abs(f.w)<1e-4){d=min(d,length(q-r));}
 return d;
}
// R. Workgroup: the window, the record, the reach of the 4³ tiles from one
// below, and the nearest crossing cell (bank B; none outside reach 1) of the
// 10³ cells from one below, as that cell's offset from the brick's first
// cell (nbArenaPack).
var<workgroup> nbArenaRecord:array<u32,27>;
var<workgroup> nbArenaReach:array<vec2u,64>;
var<workgroup> nbArenaNearestCell:array<u32,1000>;
const NB_ARENA_NONE:u32=0xffffffffu;
// An offset of -8..23 an axis, five bits an axis.
fn nbArenaPack(o:vec3i)->u32{let u=vec3u(o+8);return u.x|(u.y<<5u)|(u.z<<10u);}
fn nbArenaUnpack(c:u32)->vec3i{return vec3i(vec3u(c&31u,(c>>5u)&31u,c>>10u))-8;}
// Window vertex j (0..216) of the high faces: x=8, then y=8, then z=8.
fn nbArenaFace(j:u32)->vec3u{
 if(j<81u){return vec3u(8u,j%9u,j/9u);}
 if(j<153u){let i=j-81u;return vec3u(i%8u,8u,i/8u);}
 let i=j-153u;return vec3u(i%8u,i/8u,8u);
}
// The nearest crossing cell of p's incident cell e (window vertex l).
fn nbArenaIncident(l:vec3u,e:u32)->u32{let cell=umCorner(e,2u);return nbArenaNearestCell[(l.x+1u-cell.x)+10u*((l.y+1u-cell.y)+10u*(l.z+1u-cell.z))];}
// The gap from p (window vertex l) to the box of a cell at offset at.
fn nbArenaGap(l:vec3u,at:vec3i)->f32{let q=vec3f(l)-vec3f(at);return length(max(max(-q,q-vec3f(1.0)),vec3f(0.0)));}
// R, compacted: a round of 256 vertices queues its (vertex, candidate)
// pairs, every vertex's nearest box first, and the workgroup solves them
// lane by lane: a vertex with eight candidates no longer holds 31 lanes idle.
var<workgroup> nbArenaQueue:array<u32,2048>;
var<workgroup> nbArenaQueued:array<atomic<u32>,2>;
var<workgroup> nbArenaFound:array<atomic<u32>,256>;
fn nbArenaVertexOf(k:u32)->vec3u{if(k<512u){return umCorner(k,8u);}return nbArenaFace(k-512u);}
@compute @workgroup_size(256) fn arenaDistance(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=nbArenaUniform(ubCount(),lane);let h=nbArenaH();let inverse=1.0/h;
 for(var slot=group.x;slot<count;slot+=groups.x){
  if((nbArenaUniform(ubClass(slot),lane)&UB_ARENA)==0u){continue;}
  let id=ubBrick(slot);let block=id>>6u;let origin=ubBrickOrigin(id);let page=vec3i(origin>>vec3u(5u));let t0=vec3i(origin/4u);
  umLoadAddressPlan(block,lane);
  for(var k=lane;k<NB_ARENA_WINDOW;k+=256u){nbArenaWindow[k]=nbArenaLoad(NB_ARENA_WINDOW*slot+k);}
  if(lane<27u){nbArenaRecord[lane]=ubNeighbour(slot,lane);}
  if(lane<64u){let key=umTileKey(t0+vec3i(umCorner(lane,4u))-1);nbArenaReach[lane]=vec2u(key,select(0u,atomicLoad(&bins[NB_BAND_REACH+key]),key<UM_TILES));}
  workgroupBarrier();
  for(var k=lane;k<1000u;k+=256u){
   let c=vec3i(origin)+vec3i(umCorner(k,10u))-1;let r=vec3u((c>>vec3u(2u))-t0+1);let tile=nbArenaReach[r.x+4u*(r.y+4u*r.z)];
   var nearest=NB_ARENA_NONE;
   if(tile.x<UM_TILES&&(tile.y&1u)!=0u){
    let n=atomicLoad(&bins[NB_DEPTH_B+64u*tile.x+u32((c.x&3)|((c.y&3)<<2u)|((c.z&3)<<4u))]);
    if(n!=NB_NO_SURFACE){nearest=nbArenaPack(vec3i(umCellCoord(n))-vec3i(origin));}
   }
   nbArenaNearestCell[k]=nearest;
  }
  let vertices=nbArenaUniform(512u+select(0u,217u,any((origin&vec3u(31u))==vec3u(24u))),lane);
  for(var base=0u;base<vertices;base+=256u){
   workgroupBarrier();
   if(lane<2u){atomicStore(&nbArenaQueued[lane],0u);}
   atomicStore(&nbArenaFound[lane],0x7f7fffffu);
   workgroupBarrier();
   // The round's vertex: one whose cell lies in the brick, in a reach-2 tile, and its pairs.
   let k=base+lane;let l=nbArenaVertexOf(k);let p=origin+l;var write=false;var initial=0.0;
   if(k<vertices){
    var c=p;if(k>=512u){c=vec3u(umCellNearFromPage(block,page,vec3i(p)));}
    let r=c/4u-vec3u(t0)+1u;
    if(all(c<origin+8u)&&(nbArenaReach[r.x+4u*(r.y+4u*r.z)].y&2u)!=0u){
     write=true;initial=nbArenaWindow[l.x+9u*(l.y+9u*l.z)];
     if(abs(initial)>1e-8){
      var best=8u;var gap=3.402823e38;
      for(var e=0u;e<8u;e++){let n=nbArenaIncident(l,e);if(n==NB_ARENA_NONE){continue;}let g=nbArenaGap(l,nbArenaUnpack(n));if(g<gap){gap=g;best=e;}}
      if(best<8u){
       let first=nbArenaIncident(l,best);
       for(var e=0u;e<8u;e++){
        let n=nbArenaIncident(l,e);if(n==NB_ARENA_NONE){continue;}
        if(e!=best){var repeat=n==first;for(var j=0u;j<e;j++){if(nbArenaIncident(l,j)==n){repeat=true;}}if(repeat){continue;}}
        let b=(nbArenaUnpack(n)>>vec3u(3u))+1;if(any(b<vec3i(0))||any(b>vec3i(2))||nbArenaRecord[u32(b.x+3*b.y+9*b.z)]==UB_NONE){continue;}
        if(e==best){nbArenaQueue[atomicAdd(&nbArenaQueued[0],1u)]=lane|(n<<8u);}
        else{nbArenaQueue[256u+atomicAdd(&nbArenaQueued[1],1u)]=lane|(n<<8u);}
       }
      }
     }
    }
   }
   // The nearest boxes, then the rest: a box no nearer than the vertex's best zero is skipped.
   for(var queue=0u;queue<2u;queue++){
    // Lane 0 reads the count once every lane's appends have landed.
    workgroupBarrier();let n=nbArenaUniform(atomicLoad(&nbArenaQueued[queue]),lane);
    for(var i=lane;i<n;i+=256u){
     let entry=nbArenaQueue[select(256u+i,i,queue==0u)];let v=entry&255u;let at=nbArenaUnpack(entry>>8u);let lv=nbArenaVertexOf(base+v);
     if(nbArenaGap(lv,at)>=bitcast<f32>(atomicLoad(&nbArenaFound[v]))){continue;}
     let b=(at>>vec3u(3u))+1;let s=nbArenaRecord[u32(b.x+3*b.y+9*b.z)];
     let o=vec3u(at-8*(b-1));let wo=o.x+9u*(o.y+9u*o.z);var values:array<f32,8>;
     if(all(b==vec3i(1))){for(var j=0u;j<8u;j++){let corner=umCorner(j,2u);values[j]=nbArenaWindow[wo+corner.x+9u*(corner.y+9u*corner.z)]*inverse;}}
     else{let at0=NB_ARENA_WINDOW*s+wo;for(var j=0u;j<8u;j++){let corner=umCorner(j,2u);values[j]=nbArenaLoad(at0+corner.x+9u*(corner.y+9u*corner.z))*inverse;}}
     let d=nbArenaCellDistance(vec3f(lv)-vec3f(at),values);if(d>=0.0){atomicMin(&nbArenaFound[v],bitcast<u32>(d));}
    }
   }
   workgroupBarrier();
   if(write){
    var value=initial/h;
    if(abs(initial)>1e-8){
     // With no crossing cell within five of an incident cell the level set stands, no nearer than five cells.
     let found=bitcast<f32>(atomicLoad(&nbArenaFound[lane]));var distance=max(abs(value),5.0);if(found<3.402823e38){distance=found;}
     value=sign(initial)*max(distance,1e-8/h);
    }
    atomicStore(&bins[NB_DEPTH_A+nbArenaVertexKey(p,block,page)],bitcast<u32>(value));
   }
  }
 }
}
// Bank A of a distance tile outside the arena: no crossing cell lies within
// eight of its vertices, so none within five of their cells.
@compute @workgroup_size(128) fn arenaFar(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let h=nbArenaH();
 for(var j=group.x;j<atomicLoad(&bins[NB_BAND+2u]);j+=groups.x){
  let tile=atomicLoad(&bins[NB_BAND_DISTANCE+j]);let s=ubSlot(ubBrickOfTile(tile));
  if(lane>=125u||(s!=UB_NONE&&(ubClass(s)&UB_ARENA)!=0u)){continue;}
  let origin=4u*umTileCoord(tile);let p=origin+umCorner(lane,5u);
  let c=vec3u(umCellNearFromPage(tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(p)));if(any(c/4u!=origin/4u)){continue;}
  let initial=umSampleVertex(vec3f(p));var value=initial/h;
  if(abs(initial)>1e-8){value=sign(initial)*max(abs(value),5.0);}
  atomicStore(&bins[NB_DEPTH_A+umVertexAt(p)],bitcast<u32>(value));
 }
}
// W. Workgroup: the 4³ tiles from one below (key; NB_STORE_* bits with
// the reach from bit 8; the band mask), the capacity (umVertexCellCapacity)
// of the 10³ cells from one below, and the reuse regions a change touched.
var<workgroup> nbStoreTile:array<vec4u,64>;
var<workgroup> nbStoreIncident:array<u32,64>;
var<workgroup> nbStoreCapacity:array<u32,1000>;
var<workgroup> nbStoreRegions:atomic<u32>;
var<workgroup> nbArenaFine:atomic<u32>;
const NB_STORE_STORED:u32=1u;const NB_STORE_H:u32=2u;const NB_STORE_REGULAR:u32=4u;const NB_STORE_COLD:u32=64u;const NB_STORE_WALL:u32=128u;
// Bits 3..5: umLowOpen of a regular h tile's first cell, an axis a bit.
fn nbStoreAt(q:vec3i,t0:vec3i)->vec4u{let r=vec3u(q-t0+1);return nbStoreTile[r.x+4u*(r.y+4u*r.z)];}
// The fine tile redistanceFine wrote p from: a regular h tile its high
// corners and the low faces nothing is stored below, any other h tile the
// vertices it is the authority of. xyz its tile coordinate, w its key (UB_NONE: none).
fn nbStoreWriter(p:vec3u,t0:vec3i)->vec4u{
 let plane=(p&vec3u(3u))==vec3u(0u);let sides=u32(plane.x)|(u32(plane.y)<<1u)|(u32(plane.z)<<2u);
 for(var k=0u;k<=sides;k=select((k-sides)&sides,8u,k==sides)){
  let q=vec3i(p>>vec3u(2u))-vec3i(umCorner(k,2u));let tile=nbStoreAt(q,t0);
  if((tile.y&(NB_STORE_STORED|NB_STORE_H))!=(NB_STORE_STORED|NB_STORE_H)){continue;}
  let local=vec3u(vec3i(p)-4*q);
  if((tile.y&NB_STORE_REGULAR)!=0u){
   let low=(vec3u(tile.y)>>vec3u(3u,4u,5u))&vec3u(1u);
   if(any(local==vec3u(0u))&&!all((local>vec3u(0u))|(low!=vec3u(0u)))){continue;}
  }else if(!nbStoreAuthority(p,t0,tile.x)){continue;}
  return vec4u(vec3u(q),tile.x);
 }
 return vec4u(0u,0u,0u,UB_NONE);
}
// umVertexAuthority(p) is the h tile key, from the staged tiles: away from a
// wall page the cell above is p's and the cell below p-1's.
fn nbStoreAuthority(p:vec3u,t0:vec3i,key:u32)->bool{
 // umVertexIncidentCell moves a cell only out of a wall page (the lattice's SOLID).
 if(any((p&vec3u(31u))==vec3u(0u))&&((nbStoreAt(vec3i(p>>vec3u(2u)),t0).y|nbStoreAt((vec3i(p)-1)>>vec3u(2u),t0).y)&NB_STORE_WALL)!=0u){let a=umVertexAuthority(p);return a.width==1u&&a.tile==key;}
 if(!umRegularFine){
  // A coarse incident tile of the cell above is the authority: 4h, never the key.
  let q=vec3i(p>>vec3u(2u));let local=p-4u*vec3u(q);
  let required=u32(local.x!=0u)|(u32(local.y!=0u)<<1u)|(u32(local.z!=0u)<<2u);
  let r=vec3u(q-t0+1);if((nbStoreIncident[r.x+4u*(r.y+4u*r.z)]&UM_VERTEX_INCIDENT[required])!=0u){return false;}
 }
 // umOwnerAt(the cell below): its tile, if stored, and its width.
 let below=nbStoreAt((vec3i(p)-1)>>vec3u(2u),t0);
 return (below.y&(NB_STORE_STORED|NB_STORE_H))==(NB_STORE_STORED|NB_STORE_H)&&below.x==key;
}
// The same physical incident-cell rule as nbVertexCold; staged cold bits
// alone include solid padding, which must not pin a moving liquid surface.
fn nbStoreCold(p:vec3u,t0:vec3i)->bool{
 for(var k=0u;k<8u;k++){
  let cell=vec3i(p)-vec3i(umCorner(k,2u));let tile=nbStoreAt(cell>>vec3u(2u),t0);
  if((tile.y&(NB_STORE_STORED|NB_STORE_COLD))==(NB_STORE_STORED|NB_STORE_COLD)&&umVertexCellCapacity(cell,tile.x)){return true;}
 }
 return false;
}
// nbInterfaceVertex from the staged band masks.
fn nbStoreInterface(p:vec3u,t0:vec3i)->bool{
 let local=p&vec3u(3u);
 if(all(local!=vec3u(0u))){
  let x=(1u<<local.x)|(1u<<(local.x-1u));
  let xy=(x<<(4u*local.y))|(x<<(4u*(local.y-1u)));
  let mask=nbStoreAt(vec3i(p>>vec3u(2u)),t0).zw;
  if(local.z==2u){return (((mask.x>>16u)|mask.y)&xy)!=0u;}
  return (mask[local.z>>1u]&(xy|(xy<<16u)))!=0u;
 }
 for(var k=0u;k<8u;k++){
  let c=vec3i(p)-vec3i(umCorner(k,2u));let tile=nbStoreAt(c>>vec3u(2u),t0);
  if((tile.y&NB_STORE_STORED)==0u){continue;}
  let m=vec3u(c)&vec3u(3u);
  if(((tile[2u+(m.z>>1u)]>>(m.x+4u*m.y+16u*(m.z&1u)))&1u)!=0u){return true;}
 }
 return false;
}
@compute @workgroup_size(256) fn arenaStore(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=nbArenaUniform(ubCount(),lane);let h=nbArenaH();
 for(var slot=group.x;slot<count;slot+=groups.x){
  if((nbArenaUniform(ubClass(slot),lane)&UB_ARENA)==0u){continue;}
  let id=ubBrick(slot);let block=id>>6u;let origin=ubBrickOrigin(id);let t0=vec3i(origin/4u);
  if(lane==0u){atomicStore(&nbArenaFine,0u);atomicStore(&nbStoreRegions,0u);}
  umLoadAddressPlan(block,lane);workgroupBarrier();
  if(lane<8u&&umTileWidth(umTileAt(vec3u(t0)+umCorner(lane,2u)))==1u){atomicOr(&nbArenaFine,1u<<lane);}
  // Only h tiles write: fine, bit x+2y+4z of the brick's h tiles.
  workgroupBarrier();let fine=nbArenaUniform(atomicLoad(&nbArenaFine),lane);if(fine==0u){continue;}
  if(lane<64u){
   let q=t0+vec3i(umCorner(lane,4u))-1;let key=umTileKey(q);var bits=0u;var mask=vec2u(0u);var incident=0u;
   if(umKeyStored(key)){
    bits=NB_STORE_STORED|(atomicLoad(&bins[NB_BAND_REACH+key])<<8u);
    if(umTileWidth(key)==1u){bits|=NB_STORE_H;}
    if(umTileMaximumWidth(key)==1u&&umTileMinimumWidth(key)==1u){let low=umLowOpen(4*q);bits|=NB_STORE_REGULAR|(8u*u32(low.x))|(16u*u32(low.y))|(32u*u32(low.z));}
    if(nbRegionCold(umPageOfTile(key))){bits|=NB_STORE_COLD;}
    mask=vec2u(atomicLoad(&bins[NB_BAND_MASK+2u*key]),atomicLoad(&bins[NB_BAND_MASK+2u*key+1u]));
    incident=(umVertexRecipe(key)>>${UNIFORM_COMPILED_TOPOLOGY.incident}u)&255u;
   }
   if(umTileKind(q)==UM_SOLID){bits|=NB_STORE_WALL;}
   nbStoreTile[lane]=vec4u(key,bits,mask);nbStoreIncident[lane]=incident;
  }
  workgroupBarrier();
  for(var k=lane;k<1000u;k+=256u){let cell=vec3i(origin)+vec3i(umCorner(k,10u))-1;nbStoreCapacity[k]=u32(umVertexCellCapacity(cell,nbStoreAt(cell>>vec3u(2u),t0).x));}
  workgroupBarrier();
  let page=vec3i(origin>>vec3u(5u));let reuse=nbReuseEnabled();let regions=(t0-1)>>vec3u(2u);
  for(var k=lane;k<NB_ARENA_WINDOW;k+=256u){
   let l=umCorner(k,9u);let p=origin+l;
   // The brick's tiles holding p (an axis: tile 0 at 0..4, tile 1 at 4..8); none an h tile, no writer.
   let a=select(vec3u(1u),vec3u(3u),l==vec3u(4u))<<select(vec3u(0u),vec3u(1u),l>vec3u(4u));
   var held=0u;for(var j=0u;j<8u;j++){let q=umCorner(j,2u);if(all(((a>>q)&vec3u(1u))!=vec3u(0u))){held|=1u<<j;}}
   if((held&fine)==0u){continue;}
   let writer=nbStoreWriter(p,t0);
   if(writer.w==UB_NONE||any(vec3i(writer.xyz)<t0)||any(vec3i(writer.xyz)>t0+1)){continue;}
   // Every vertex of a COLD region's tile is COLD: redistance holds it.
   if((nbStoreAt(vec3i(writer.xyz),t0).y&NB_STORE_COLD)!=0u||nbStoreCold(p,t0)){continue;}
   // An interface vertex's magnitude places its zero; a buried one (no incident capacity) is no distance sample.
   if(nbStoreInterface(p,t0)){continue;}
   var open=0u;for(var j=0u;j<8u;j++){let m=l+umCorner(j,2u);open|=nbStoreCapacity[m.x+10u*(m.y+10u*m.z)];}
   if(open==0u){continue;}
   // A writer tile lies in the brick, so its page is the brick's.
   var c=p;if(any(l==vec3u(8u))){c=vec3u(umCellNearFromPage(block,page,vec3i(p)));}
   let tile=nbStoreAt(vec3i(c>>vec3u(2u)),t0).y;
   // The window holds the h texel wherever the vertex's cell is an h tile's.
   var old=0.0;if((tile&NB_STORE_H)!=0u){old=nbArenaLoad(NB_ARENA_WINDOW*slot+k);}else{old=textureLoad(outputPhi,vec3i(p)).x;}
   var value=old;
   if(((tile>>8u)&2u)!=0u){value=bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+nbArenaVertexKey(p,block,page)]))*h;}
   // Past reach: the old sign, at least six cells.
   else if(abs(old)>1e-8&&abs(old)<6.0*h){value=sign(old)*6.0*h;}
   if(bitcast<u32>(value)==bitcast<u32>(old)){continue;}
   // nbMarkPhiChanged once a brick: the regions of p's incident cells (those across its 16-cell planes).
   if(reuse){
    let boundary=(p&vec3u(15u))==vec3u(0u);let sides=u32(boundary.x)|(2u*u32(boundary.y))|(4u*u32(boundary.z));
    for(var j=0u;j<=sides;j=select((j-sides)&sides,8u,j==sides)){
     let q=(vec3i(p)-vec3i(umCorner(j,2u)))>>vec3u(2u);if((nbStoreAt(q,t0).y&NB_STORE_STORED)==0u){continue;}
     let r=vec3u((q>>vec3u(2u))-regions);atomicOr(&nbStoreRegions,1u<<(r.x|(r.y<<1u)|(r.z<<2u)));
    }
   }
   textureStore(outputPhi,vec3i(p),vec4f(value));
  }
  workgroupBarrier();
  // A region's tiles share its page: the first of them in the 4³ tiles stands for it.
  if(lane<8u&&(atomicLoad(&nbStoreRegions)&(1u<<lane))!=0u){
   let key=umTileKey(max(t0-1,4*(regions+vec3i(umCorner(lane,2u)))));
   if(umKeyStored(key)){
    let at=NB_REUSE_LIVE+nbReuseRegion(key);let old=atomicLoad(&bins[at]);
    if(old>0u){atomicMax(&bins[at],min(old,0xfffffffdu)+1u);}
   }
  }
 }
}
`;

/** Publish through the metric's canonical vertex owner. Keep capacity in its
 * native 64-bit tile masks, and route revision bits to the original writer.
 * Gather initializes marks; Finish handles missing owners and commits once
 * per writer brick. Requires the solid tile record (umSolidOpenCells).
 */
export const narrowBandArenaCanonicalWGSL=/* wgsl */`
var<workgroup> nbCapacityTileMask:array<vec2u,64>;
// Exactly umVertexCellCapacity for all64 cells of a4³ tile, kept in the
// solid record's existing bit representation instead of expanded scalars.
// umSolidValid is page-constant, and a4³ tile never crosses a32³ page.
fn nbCapacityMaskAt(q:vec3i,key:u32)->vec2u{
 if(!umSolidValid(4*q)){return vec2u(0u);}
 if(!umKeyStored(key)){return select(vec2u(0u),vec2u(0xffffffffu),umTileOpenAt(q)>1e-5);}
 if(umTileWidth(key)==4u){return select(vec2u(0u),vec2u(0xffffffffu),umTileOpen(key)>1e-5);}
 if(!umSolidEnabled()){return vec2u(0xffffffffu);}
 return umSolidOpenCells(key);
}
fn nbCapacityOpen(p:vec3u,t0:vec3i)->u32{
 for(var j=0u;j<8u;j++){
  let c=vec3i(p)+vec3i(umCorner(j,2u))-1;let r=(c>>vec3u(2u))-t0+1;
  let mask=nbCapacityTileMask[u32(r.x+4*r.y+16*r.z)];let m=vec3u(c)&vec3u(3u);let bit=m.x+4u*m.y+16u*m.z;
  if((mask[bit>>5u]&(1u<<(bit&31u)))!=0u){return 1u;}
 }
 return 0u;
}

var<workgroup> nbCanonicalFine:atomic<u32>;
var<workgroup> nbCanonicalRegions:array<atomic<u32>,27>;
// UB_MARK is dead after brickTableList/Record and every later table build
// clears it. Reuse one word per physical brick for the original writer's
// eight incident-region change bits. No live table keys/classes are changed.
fn nbArenaCanonicalWriter(p:vec3u,t0:vec3i)->vec4u{
   let writer=nbStoreWriter(p,t0);
   if(writer.w==UB_NONE){return vec4u(0u,0u,0u,UB_NONE);}

    let offset=vec3i(writer.xyz>>vec3u(1u))-(t0>>vec3u(1u))+1;
    let writerSlot=nbArenaRecord[u32(offset.x+3*offset.y+9*offset.z)];
    if(writerSlot==UB_NONE||(ubClass(writerSlot)&UB_ARENA)==0u){return vec4u(0u,0u,0u,UB_NONE);}

   // Every vertex of a COLD region's tile is COLD: redistance holds it.
   if((nbStoreAt(vec3i(writer.xyz),t0).y&NB_STORE_COLD)!=0u||nbStoreCold(p,t0)){return vec4u(0u,0u,0u,UB_NONE);}
   // An interface vertex's magnitude places its zero; a buried one (no incident capacity) is no distance sample.
   if(nbStoreInterface(p,t0)){return vec4u(0u,0u,0u,UB_NONE);}
   if(nbCapacityOpen(p,t0)==0u){return vec4u(0u,0u,0u,UB_NONE);}
 return writer;
}
fn nbArenaCanonicalPublish(p:vec3u,l:vec3u,t0:vec3i,block:u32,page:vec3i,slot:u32,h:f32,metric:f32,reuse:bool,writer:vec4u){
 let k=l.x+9u*(l.y+9u*l.z);
   // A writer tile lies in the brick, so its page is the brick's.
   var c=p;if(any(l==vec3u(8u))){c=vec3u(umCellNearFromPage(block,page,vec3i(p)));}
   let tile=nbStoreAt(vec3i(c>>vec3u(2u)),t0).y;
   // The window holds the h texel wherever the vertex's cell is an h tile's.
   var old=0.0;if((tile&NB_STORE_H)!=0u){old=nbArenaLoad(NB_ARENA_WINDOW*slot+k);}else{old=textureLoad(outputPhi,vec3i(p)).x;}
   var value=old;
   if(((tile>>8u)&2u)!=0u){value=metric*h;}
   // Past reach: the old sign, at least six cells.
   else if(abs(old)>1e-8&&abs(old)<6.0*h){value=sign(old)*6.0*h;}
   if(bitcast<u32>(value)==bitcast<u32>(old)){return;}
   // nbMarkPhiChanged once a brick: the regions of p's incident cells (those across its 16-cell planes).
   if(reuse){
    let writerT0=vec3i(writer.xyz&~vec3u(1u));let regions=(writerT0-1)>>vec3u(2u);
    let offset=(writerT0>>vec3u(1u))-(t0>>vec3u(1u))+1;let route=u32(offset.x+3*offset.y+9*offset.z);
    let boundary=(p&vec3u(15u))==vec3u(0u);let sides=u32(boundary.x)|(2u*u32(boundary.y))|(4u*u32(boundary.z));
    for(var j=0u;j<=sides;j=select((j-sides)&sides,8u,j==sides)){
     let q=(vec3i(p)-vec3i(umCorner(j,2u)))>>vec3u(2u);if((nbStoreAt(q,t0).y&NB_STORE_STORED)==0u){continue;}
     let r=vec3u((q>>vec3u(2u))-regions);atomicOr(&nbCanonicalRegions[route],1u<<(r.x|(r.y<<1u)|(r.z<<2u)));
    }
   }
   textureStore(outputPhi,vec3i(p),vec4f(value));
}
@compute @workgroup_size(256) fn arenaDistanceCanonical(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=nbArenaUniform(ubCount(),lane);let h=nbArenaH();let inverse=1.0/h;
 for(var slot=group.x;slot<count;slot+=groups.x){
  if((nbArenaUniform(ubClass(slot),lane)&UB_ARENA)==0u){continue;}
  let id=ubBrick(slot);let block=id>>6u;let origin=ubBrickOrigin(id);let page=vec3i(origin>>vec3u(5u));let t0=vec3i(origin/4u);
  umLoadAddressPlan(block,lane);
  if(lane<27u){atomicStore(&nbCanonicalRegions[lane],0u);}
   if(lane==0u){atomicStore(&nbCanonicalFine,0u);}workgroupBarrier();
   // Metric always needs key/reach. Fine publication needs the larger
   // descriptor and cell capacities only if an incident writer is fine.
   if(lane<64u){
    let key=umTileKey(t0+vec3i(umCorner(lane,4u))-1);var reach=0u;
    if(umKeyStored(key)){reach=atomicLoad(&bins[NB_BAND_REACH+key]);if(umTileWidth(key)==1u){atomicOr(&nbCanonicalFine,1u);}}
    nbArenaReach[lane]=vec2u(key,reach);
   }
   workgroupBarrier();let publishFine=nbArenaUniform(atomicLoad(&nbCanonicalFine),lane)!=0u;

  if(publishFine&&lane<64u){
   let q=t0+vec3i(umCorner(lane,4u))-1;let key=nbArenaReach[lane].x;
   var bits=0u;var mask=vec2u(0u);var incident=0u;
   if(umKeyStored(key)){
    let reach=nbArenaReach[lane].y;
    bits=NB_STORE_STORED|(reach<<8u);
    if(umTileWidth(key)==1u){bits|=NB_STORE_H;}
    if(umTileMaximumWidth(key)==1u&&umTileMinimumWidth(key)==1u){let low=umLowOpen(4*q);bits|=NB_STORE_REGULAR|(8u*u32(low.x))|(16u*u32(low.y))|(32u*u32(low.z));}
    if(nbRegionCold(umPageOfTile(key))){bits|=NB_STORE_COLD;}
    mask=vec2u(atomicLoad(&bins[NB_BAND_MASK+2u*key]),atomicLoad(&bins[NB_BAND_MASK+2u*key+1u]));
    incident=(umVertexRecipe(key)>>${UNIFORM_COMPILED_TOPOLOGY.incident}u)&255u;
   }
   if(umTileKind(q)==UM_SOLID){bits|=NB_STORE_WALL;}
   nbStoreTile[lane]=vec4u(key,bits,mask);nbStoreIncident[lane]=incident;
   nbCapacityTileMask[lane]=nbCapacityMaskAt(q,key);
  }
  workgroupBarrier();
  let reuse=nbReuseEnabled();
  for(var k=lane;k<NB_ARENA_WINDOW;k+=256u){nbArenaWindow[k]=nbArenaLoad(NB_ARENA_WINDOW*slot+k);}
  if(lane<27u){nbArenaRecord[lane]=ubNeighbour(slot,lane);}
  workgroupBarrier();
  for(var k=lane;k<1000u;k+=256u){
   let c=vec3i(origin)+vec3i(umCorner(k,10u))-1;let r=vec3u((c>>vec3u(2u))-t0+1);let tile=nbArenaReach[r.x+4u*(r.y+4u*r.z)];
   var nearest=NB_ARENA_NONE;
   if(tile.x<UM_TILES&&(tile.y&1u)!=0u){
    let n=atomicLoad(&bins[NB_DEPTH_B+64u*tile.x+u32((c.x&3)|((c.y&3)<<2u)|((c.z&3)<<4u))]);
    if(n!=NB_NO_SURFACE){nearest=nbArenaPack(vec3i(umCellCoord(n))-vec3i(origin));}
   }
   nbArenaNearestCell[k]=nearest;
  }
  let metricVertices=512u+select(0u,217u,any((origin&vec3u(31u))==vec3u(24u)));
  let vertices=nbArenaUniform(metricVertices,lane);
  for(var base=0u;base<vertices;base+=256u){
   workgroupBarrier();
   if(lane<2u){atomicStore(&nbArenaQueued[lane],0u);}
   atomicStore(&nbArenaFound[lane],0x7f7fffffu);
   workgroupBarrier();
   // The round's vertex: one whose cell lies in the brick, in a reach-2 tile, and its pairs.
   let k=base+lane;let l=nbArenaVertexOf(k);let p=origin+l;var write=false;var metricWrite=false;var initial=0.0;
   var writer=vec4u(0u,0u,0u,UB_NONE);
   if(k<vertices){
    var c=p;if(k>=512u){c=vec3u(umCellNearFromPage(block,page,vec3i(p)));}
    let r=c/4u-vec3u(t0)+1u;
    let reached=(nbArenaReach[r.x+4u*(r.y+4u*r.z)].y&2u)!=0u;
    let owned=all(c<origin+8u);
    if(owned&&publishFine){writer=nbArenaCanonicalWriter(p,t0);}
    metricWrite=owned&&reached;
    if(metricWrite){
     write=true;initial=nbArenaWindow[l.x+9u*(l.y+9u*l.z)];
     if(abs(initial)>1e-8){
      var best=8u;var gap=3.402823e38;
      for(var e=0u;e<8u;e++){let n=nbArenaIncident(l,e);if(n==NB_ARENA_NONE){continue;}let g=nbArenaGap(l,nbArenaUnpack(n));if(g<gap){gap=g;best=e;}}
      if(best<8u){
       let first=nbArenaIncident(l,best);
       for(var e=0u;e<8u;e++){
        let n=nbArenaIncident(l,e);if(n==NB_ARENA_NONE){continue;}
        if(e!=best){var repeat=n==first;for(var j=0u;j<e;j++){if(nbArenaIncident(l,j)==n){repeat=true;}}if(repeat){continue;}}
        let b=(nbArenaUnpack(n)>>vec3u(3u))+1;if(any(b<vec3i(0))||any(b>vec3i(2))||nbArenaRecord[u32(b.x+3*b.y+9*b.z)]==UB_NONE){continue;}
        if(e==best){nbArenaQueue[atomicAdd(&nbArenaQueued[0],1u)]=lane|(n<<8u);}
        else{nbArenaQueue[256u+atomicAdd(&nbArenaQueued[1],1u)]=lane|(n<<8u);}
       }
      }
     }
    }
   }
   // The nearest boxes, then the rest: a box no nearer than the vertex's best zero is skipped.
   for(var queue=0u;queue<2u;queue++){
    // Lane 0 reads the count once every lane's appends have landed.
    workgroupBarrier();let n=nbArenaUniform(atomicLoad(&nbArenaQueued[queue]),lane);
    for(var i=lane;i<n;i+=256u){
     let entry=nbArenaQueue[select(256u+i,i,queue==0u)];let v=entry&255u;let at=nbArenaUnpack(entry>>8u);let lv=nbArenaVertexOf(base+v);
     if(nbArenaGap(lv,at)>=bitcast<f32>(atomicLoad(&nbArenaFound[v]))){continue;}
     let b=(at>>vec3u(3u))+1;let s=nbArenaRecord[u32(b.x+3*b.y+9*b.z)];
     let o=vec3u(at-8*(b-1));let wo=o.x+9u*(o.y+9u*o.z);var values:array<f32,8>;
     if(all(b==vec3i(1))){for(var j=0u;j<8u;j++){let corner=umCorner(j,2u);values[j]=nbArenaWindow[wo+corner.x+9u*(corner.y+9u*corner.z)]*inverse;}}
     else{let at0=NB_ARENA_WINDOW*s+wo;for(var j=0u;j<8u;j++){let corner=umCorner(j,2u);values[j]=nbArenaLoad(at0+corner.x+9u*(corner.y+9u*corner.z))*inverse;}}
     let d=nbArenaCellDistance(vec3f(lv)-vec3f(at),values);if(d>=0.0){atomicMin(&nbArenaFound[v],bitcast<u32>(d));}
    }
   }
   workgroupBarrier();
   if(write){
    var value=initial/h;
    if(abs(initial)>1e-8){
     // With no crossing cell within five of an incident cell the level set stands, no nearer than five cells.
     let found=bitcast<f32>(atomicLoad(&nbArenaFound[lane]));var distance=max(abs(value),5.0);if(found<3.402823e38){distance=found;}
     value=sign(initial)*max(distance,1e-8/h);
    }
    if(metricWrite){atomicStore(&bins[NB_DEPTH_A+nbArenaVertexKey(p,block,page)],bitcast<u32>(value));}
    if(writer.w!=UB_NONE){nbArenaCanonicalPublish(p,l,t0,block,page,slot,h,value,reuse,writer);}
   }else if(writer.w!=UB_NONE){nbArenaCanonicalPublish(p,l,t0,block,page,slot,h,0.0,reuse,writer);}
  }
  workgroupBarrier();
  // Aggregate once per original writer brick, not once per vertex.
  if(lane<27u){
   let changed=atomicLoad(&nbCanonicalRegions[lane]);let writerSlot=nbArenaRecord[lane];
   if(changed!=0u&&writerSlot!=UB_NONE){atomicOr(&bins[UB_MARK+ubBrick(writerSlot)],changed);}
  }
 }
}
// arenaDistanceCanonical's publication alone, after arenaDistance stored
// the metric (bank A) of every vertex it would have published: the same
// writers, values and routed regions from a lean kernel, which the metric's
// windows and queues do not hold to one workgroup a core.
@compute @workgroup_size(256) fn arenaPublishCanonical(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=nbArenaUniform(ubCount(),lane);let h=nbArenaH();
 for(var slot=group.x;slot<count;slot+=groups.x){
  if((nbArenaUniform(ubClass(slot),lane)&UB_ARENA)==0u){continue;}
  let id=ubBrick(slot);let block=id>>6u;let origin=ubBrickOrigin(id);let page=vec3i(origin>>vec3u(5u));let t0=vec3i(origin/4u);
  umLoadAddressPlan(block,lane);
  if(lane<27u){atomicStore(&nbCanonicalRegions[lane],0u);}
  if(lane==0u){atomicStore(&nbCanonicalFine,0u);}workgroupBarrier();
  let q=t0+vec3i(umCorner(lane&63u,4u))-1;var key=UM_TILES;
  if(lane<64u){key=umTileKey(q);if(umKeyStored(key)&&umTileWidth(key)==1u){atomicOr(&nbCanonicalFine,1u);}}
  // Only an incident fine writer publishes.
  if(nbArenaUniform(atomicLoad(&nbCanonicalFine),lane)==0u){continue;}
  if(lane<64u){
   var bits=0u;var mask=vec2u(0u);var incident=0u;
   if(umKeyStored(key)){
    bits=NB_STORE_STORED|(atomicLoad(&bins[NB_BAND_REACH+key])<<8u);
    if(umTileWidth(key)==1u){bits|=NB_STORE_H;}
    if(umTileMaximumWidth(key)==1u&&umTileMinimumWidth(key)==1u){let low=umLowOpen(4*q);bits|=NB_STORE_REGULAR|(8u*u32(low.x))|(16u*u32(low.y))|(32u*u32(low.z));}
    if(nbRegionCold(umPageOfTile(key))){bits|=NB_STORE_COLD;}
    mask=vec2u(atomicLoad(&bins[NB_BAND_MASK+2u*key]),atomicLoad(&bins[NB_BAND_MASK+2u*key+1u]));
    incident=(umVertexRecipe(key)>>${UNIFORM_COMPILED_TOPOLOGY.incident}u)&255u;
   }
   if(umTileKind(q)==UM_SOLID){bits|=NB_STORE_WALL;}
   nbStoreTile[lane]=vec4u(key,bits,mask);nbStoreIncident[lane]=incident;
   nbCapacityTileMask[lane]=nbCapacityMaskAt(q,key);
  }
  if(lane<27u){nbArenaRecord[lane]=ubNeighbour(slot,lane);}
  workgroupBarrier();
  let reuse=nbReuseEnabled();
  let vertices=512u+select(0u,217u,any((origin&vec3u(31u))==vec3u(24u)));
  for(var k=lane;k<vertices;k+=256u){
   let l=nbArenaVertexOf(k);let p=origin+l;
   var c=p;if(k>=512u){c=vec3u(umCellNearFromPage(block,page,vec3i(p)));}
   if(!all(c<origin+8u)){continue;}
   let writer=nbArenaCanonicalWriter(p,t0);if(writer.w==UB_NONE){continue;}
   // arenaDistance wrote bank A wherever the vertex's cell lies in a reach-2 tile.
   var metric=0.0;
   if(((nbStoreAt(vec3i(c>>vec3u(2u)),t0).y>>8u)&2u)!=0u){metric=bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+nbArenaVertexKey(p,block,page)]));}
   nbArenaCanonicalPublish(p,l,t0,block,page,slot,h,metric,reuse,writer);
  }
  workgroupBarrier();
  if(lane<27u){
   let changed=atomicLoad(&nbCanonicalRegions[lane]);let writerSlot=nbArenaRecord[lane];
   if(changed!=0u&&writerSlot!=UB_NONE){atomicOr(&bins[UB_MARK+ubBrick(writerSlot)],changed);}
  }
 }
}
// Only a missing positive-side neighbour can leave a writer's high-face
// vertex outside the canonical arena-owner domain.
fn nbArenaBoundaryMissing(slot:u32)->u32{
 for(var k=1u;k<8u;k++){
  let q=umCorner(k,2u)+1u;let other=ubNeighbour(slot,q.x+3u*(q.y+3u*q.z));
  if(other==UB_NONE||(ubClass(other)&UB_ARENA)==0u){return 1u;}
 }
 return 0u;
}
// Canonical publication has finished globally. This workgroup owns the
// original writer brick, so its boundary changes and routed changes can
// commit together without another mask materialization/table traversal.
fn nbArenaFinishCommit(t0:vec3i,changed:u32,lane:u32){
 if(lane>=8u||(changed&(1u<<lane))==0u){return;}
 let regions=(t0-1)>>vec3u(2u);let key=umTileKey(max(t0-1,4*(regions+vec3i(umCorner(lane,2u)))));
 if(umKeyStored(key)){
  let at=NB_REUSE_LIVE+nbReuseRegion(key);let old=atomicLoad(&bins[at]);
  if(old>0u){atomicMax(&bins[at],min(old,0xfffffffdu)+1u);}
 }
}
@compute @workgroup_size(256) fn arenaPublishFinish(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=nbArenaUniform(ubCount(),lane);let h=nbArenaH();
 for(var slot=group.x;slot<count;slot+=groups.x){
  if((nbArenaUniform(ubClass(slot),lane)&UB_ARENA)==0u){continue;}
  let id=ubBrick(slot);let block=id>>6u;let origin=ubBrickOrigin(id);let t0=vec3i(origin/4u);
  let routed=nbArenaUniform(atomicLoad(&bins[UB_MARK+id]),lane);
  if(nbArenaUniform(nbArenaBoundaryMissing(slot),lane)==0u){nbArenaFinishCommit(t0,routed,lane);continue;}
  if(lane==0u){atomicStore(&nbArenaFine,0u);atomicStore(&nbStoreRegions,0u);}
  umLoadAddressPlan(block,lane);workgroupBarrier();
  if(lane<8u&&umTileWidth(umTileAt(vec3u(t0)+umCorner(lane,2u)))==1u){atomicOr(&nbArenaFine,1u<<lane);}
  // Only h tiles write: fine, bit x+2y+4z of the brick's h tiles.
  workgroupBarrier();let fine=nbArenaUniform(atomicLoad(&nbArenaFine),lane);if(fine==0u){nbArenaFinishCommit(t0,routed,lane);continue;}
  if(lane<64u){
   let q=t0+vec3i(umCorner(lane,4u))-1;let key=umTileKey(q);var bits=0u;var mask=vec2u(0u);var incident=0u;
   if(umKeyStored(key)){
    bits=NB_STORE_STORED|(atomicLoad(&bins[NB_BAND_REACH+key])<<8u);
    if(umTileWidth(key)==1u){bits|=NB_STORE_H;}
    if(umTileMaximumWidth(key)==1u&&umTileMinimumWidth(key)==1u){let low=umLowOpen(4*q);bits|=NB_STORE_REGULAR|(8u*u32(low.x))|(16u*u32(low.y))|(32u*u32(low.z));}
    if(nbRegionCold(umPageOfTile(key))){bits|=NB_STORE_COLD;}
    mask=vec2u(atomicLoad(&bins[NB_BAND_MASK+2u*key]),atomicLoad(&bins[NB_BAND_MASK+2u*key+1u]));
    incident=(umVertexRecipe(key)>>${UNIFORM_COMPILED_TOPOLOGY.incident}u)&255u;
   }
   if(umTileKind(q)==UM_SOLID){bits|=NB_STORE_WALL;}
   nbStoreTile[lane]=vec4u(key,bits,mask);nbStoreIncident[lane]=incident;
   nbCapacityTileMask[lane]=nbCapacityMaskAt(q,key);
  }
  workgroupBarrier();
  let page=vec3i(origin>>vec3u(5u));let reuse=nbReuseEnabled();let regions=(t0-1)>>vec3u(2u);
  for(var face=lane;face<217u;face+=256u){
   let l=nbArenaFace(face);let p=origin+l;let k=l.x+9u*(l.y+9u*l.z);
   let ownerCell=umCellNearFromPage(block,page,vec3i(p));let ownerId=ubBrickAt(ownerCell);
   if(ownerId!=UB_NONE){let ownerSlot=ubSlot(ownerId);if(ownerSlot!=UB_NONE&&(ubClass(ownerSlot)&UB_ARENA)!=0u){continue;}}
   // The brick's tiles holding p (an axis: tile 0 at 0..4, tile 1 at 4..8); none an h tile, no writer.
   let a=select(vec3u(1u),vec3u(3u),l==vec3u(4u))<<select(vec3u(0u),vec3u(1u),l>vec3u(4u));
   var held=0u;for(var j=0u;j<8u;j++){let q=umCorner(j,2u);if(all(((a>>q)&vec3u(1u))!=vec3u(0u))){held|=1u<<j;}}
   if((held&fine)==0u){continue;}
   let writer=nbStoreWriter(p,t0);
   if(writer.w==UB_NONE||any(vec3i(writer.xyz)<t0)||any(vec3i(writer.xyz)>t0+1)){continue;}
   // Every vertex of a COLD region's tile is COLD: redistance holds it.
   if((nbStoreAt(vec3i(writer.xyz),t0).y&NB_STORE_COLD)!=0u||nbStoreCold(p,t0)){continue;}
   // An interface vertex's magnitude places its zero; a buried one (no incident capacity) is no distance sample.
   if(nbStoreInterface(p,t0)){continue;}
   if(nbCapacityOpen(p,t0)==0u){continue;}
   // A writer tile lies in the brick, so its page is the brick's.
   var c=p;if(any(l==vec3u(8u))){c=vec3u(umCellNearFromPage(block,page,vec3i(p)));}
   let tile=nbStoreAt(vec3i(c>>vec3u(2u)),t0).y;
   // The window holds the h texel wherever the vertex's cell is an h tile's.
   var old=0.0;if((tile&NB_STORE_H)!=0u){old=nbArenaLoad(NB_ARENA_WINDOW*slot+k);}else{old=textureLoad(outputPhi,vec3i(p)).x;}
   var value=old;
   if(((tile>>8u)&2u)!=0u){value=bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+nbArenaVertexKey(p,block,page)]))*h;}
   // Past reach: the old sign, at least six cells.
   else if(abs(old)>1e-8&&abs(old)<6.0*h){value=sign(old)*6.0*h;}
   if(bitcast<u32>(value)==bitcast<u32>(old)){continue;}
   // nbMarkPhiChanged once a brick: the regions of p's incident cells (those across its 16-cell planes).
   if(reuse){
    let boundary=(p&vec3u(15u))==vec3u(0u);let sides=u32(boundary.x)|(2u*u32(boundary.y))|(4u*u32(boundary.z));
    for(var j=0u;j<=sides;j=select((j-sides)&sides,8u,j==sides)){
     let q=(vec3i(p)-vec3i(umCorner(j,2u)))>>vec3u(2u);if((nbStoreAt(q,t0).y&NB_STORE_STORED)==0u){continue;}
     let r=vec3u((q>>vec3u(2u))-regions);atomicOr(&nbStoreRegions,1u<<(r.x|(r.y<<1u)|(r.z<<2u)));
    }
   }
   textureStore(outputPhi,vec3i(p),vec4f(value));
  }
  workgroupBarrier();
  nbArenaFinishCommit(t0,routed|atomicLoad(&nbStoreRegions),lane);
 }
}

`;
