/**
 * LANE E OWNS THIS FILE (T4). Two exits of the old surface advect, ported verbatim with the producers of their
 * inputs: kept deep air (umDeepAirKept) and the released walls (umReleasedWalls, umReleasedMayChange). φ is in h
 * (cells) and positions are in cells. Velocities are in m/s (the pools' unit): a travel dt·v is converted to cells
 * by / h_m where the old engine kept metres. Every old metre threshold therefore reads in h. The old (6 + 3w)·min h
 * becomes 6 + 3w, the old 1e-4·h[a]·w becomes 1e-4·w, the slack 1e-5·max h becomes 1e-5, and the half width
 * 0.5·w·min h becomes 0.5·w.
 *
 * Sources, old engine (by TS file:line):
 * - The tile reach words. uniform-mixed-frame-plan.ts:
 *   - localSpeed (269-325), umSignedFaceExtent (241-252), umReach / umPackReach / umReachPlus / umReachMinus
 *     (149-155) and umCertificateRadius (136-139);
 *   - umLowOpen (uniform-lattice-address.ts 311-313);
 *   - the separable spread0..2 (uniform-frame-plan-spread.ts 12-33). The x pass goes local→REACH1, y REACH1→REACH0
 *     and z REACH0→REACH1, as spread0..2 alternate the two halves of speeds[]. The reader reads the last half
 *     (uniform-mixed-surface.ts 1993: speeds + tiles copied into the claims' evidence words).
 * - The wall reach: wallReach (uniform-mixed-surface.ts 1682-1700).
 * - The readers (uniform-mixed-surface.ts):
 *   - umStagePlaneReach (825);
 *   - umReleasedWalls (829-870), applied after the wall contact (umAdvected 1035);
 *   - umReleasedMayChange (1316-1333);
 *   - umDeepAirKept (1398-1406), through umKeptAir (1412-1417: far air first, deep air only off the wall
 *     vertices, umAdvectStore 1504-1520), with umAirSource the identity (1343; no sources, levelset-coarse
 *     deviation 9).
 *   - umWallAhead and umFaceTile: uniform-mixed-walls.wgsl.ts 80-103 and uniform-lattice-address.ts 643-644.
 *
 * The brick mapping:
 * - Page kinds (umPageKind). Out of the lattice is SOLID: the box, as the old window header's solid bounds make
 *   every page outside them. A resident page in the lattice is stored. A non-resident one is AIR. Any other word
 *   (BULK16, CLAIM) is 16h or mid-update state this port does not read. It latches FAIL.UNSUPPORTED
 *   (NR_INFO.PAGE_KIND) and never reads as air. The brick's skirt pages lie outside the lattice, so they are SOLID,
 *   as the old pages beyond the box were.
 * - The faces a reach word reads are those of the extended field: US, the old velocityScratch / `extended`.
 *   - An h cell's owner faces are its +face (the old texel at the cell: the brick's face stored at c + e_a) and,
 *     where umLowOpen holds, its low face (stored at c).
 *   - A 4h tile's +face is its upper neighbour's C face, or that neighbour's 16 h faces when the neighbour is
 *     h-owned (umFace's patches). Its low face (umLowOpen at 4T) is the C face at T.
 *   - nbFacePlane over US reads each of these.
 * - An h-owned tile with no F slot (all-h engines, outside the brick band) reads its faces as the ABI serves them:
 *   0 wherever the absent brick stores them. Its reach is the same formula over the faces it can read: the
 *   +faces of its top layer where they cross into a resident brick, together with the 0 faces. That reduction is
 *   exact, not a stand-in.
 * - The wall faces and their release:
 *   - A wall face is a resident in-lattice page's face with the box across (umWallFace's list: a stored page with
 *     SOLID across).
 *   - The probe cell q (the wall's in-lattice cell) gives the owner width, 1 or 4. The owner's wall face sits at
 *     the lower-face storage (brick-abi.md §2.4): plane 0 at q, plane N at q + e_a (the skirt cell, or C tile N/4).
 *   - The speed is U there: the projected field, the old wall texel's `velocity`. No pass between project and
 *     the surface advect rewrites U (F's tail writes US and UP only).
 *   - The release bit is FLAGS bit 8 + a at the same storage, F's cell or C's tile. After project, lane P writes
 *     it as umRelease (uniform-mixed-pressure-velocity.ts 485-492), agreed with P on 2026-11-15 and landed in
 *     9e5d2b14. Between the rows and project the same bits are P's solver wall state, so nothing here runs inside
 *     the pressure stage: both producers and the readers run before it.
 *
 * Where each producer runs (BrickEngine.scheduleFor):
 * - nrReach and nrSpread x3 run in part 1, after the head and before move. That is the old frame plan's point:
 *   encodeCertificate follows the relayout and the extension, before move.
 * - nrWallReach runs in part 2, after the lists and before stages.levelSet. That is the old point: the surface
 *   advect's first dispatch.
 * - Part 2 can claim pages after part 1's plan (setClaimPages; the old layout is fixed from the plan through the
 *   advect). Each C slot therefore carries the step its reach words were made for (REACH_STEP, the ring's step
 *   + 1). A slot whose words are not this step's reads saturated, as the old spread treats an AIR tile. The
 *   wall words are per step and per resident page at the advect, so they need no stamp.
 *
 * ABI (brick-abi.md §5.1, §5.4): REACH0 and REACH1 (512 words a C slot, a word a tile, by the tile's C local),
 * REACH_STEP (a word a C slot), WALL_TILE (384 words a C slot: 64 a face direction f = 2a + upper, by umFaceTile's
 * uv), and the header words REL_SPEED0/1 (the plan's speed, f32 bits, m/s) and WALL_REACH0/1 (6 words each: a face
 * direction's largest dt·away / h, f32 bits, cells). The two copies of each header word alternate by cfg.y (the
 * ring parity): each producer max-reduces into this step's copy and zeroes the other, which the next step reduces
 * into.
 *
 * Dispatch: nrReach is FC_G1 (an F brick or a C page a workgroup, 256 lanes). nrSpread is C_ALL G1, cfg.z the
 * axis. nrWallReach is C_ALL G1, a lane per face tile, 384 a page. The readers add no barrier. The six plane words
 * are read where used: uniform loads, not staged (lcAdvect holds no barrier).
 */
import { ARGS, type BrickLayout, brickAbiWGSL, LIST } from "./brick-abi";
import { type BrickBinding, type BrickKernel, type BrickPlacement, CFG, declareBindings } from "./brick-kernel";
import { brickOwnerWGSL } from "./brick-owner.wgsl";

/** FAIL_INFO codes of FAIL.UNSUPPORTED latched here. */
export const NR_INFO = {
  /** A page word in the lattice that is neither a C slot nor NONE (BULK16, CLAIM): the page kinds read none. */
  PAGE_KIND: 0x4e520001,
} as const;

/**
 * The readers lane E's lcAdvect and lane F's lAdvect include: nbrDeepAirKept, nbrReleasedWalls and
 * nbrReleasedMayChange. The module must hold brickAbiWGSL with set (atomic, named nbSet) and dir, the cfg uniform,
 * and `vel` (f32) and `phi` (u32) bindings. Self-contained: every name is nbr-prefixed, and the tier owners come in
 * as their own nbr copy over U.
 */
export const levelSetReleaseWGSL = () => [brickOwnerWGSL({ phi: "phi", vel: "vel", prefix: "nbr" }), NR_READERS_WGSL].join("\n");

const NR_READERS_WGSL = /* wgsl */`
const NBR_STORED:u32=0u;const NBR_AIR:u32=1u;const NBR_SOLID:u32=2u;
fn nbrDt()->f32{return nbRingF(cfg.x,NB_RING_DT);}
fn nbrHm()->f32{return nbRingF(cfg.x,NB_RING_H);}
// umPageKind of page q in the box (q in pages).
fn nbrPageKind(q:vec3i)->u32{
 if(any(q<vec3i(0))||any(32*q>=NB_LATTICE)){return NBR_SOLID;}
 let w=nbPageWord(q);
 if(w<NB_SEAM){return NBR_STORED;}
 if(w!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,${NR_INFO.PAGE_KIND}u);}
 return NBR_AIR;
}
// umReachPlus/umReachMinus's largest field: any field over 2 means the tile's departures move.
fn nbrReachMax(w:u32)->vec3u{return max(vec3u(w&31u,(w>>5u)&31u,(w>>10u)&31u),vec3u((w>>15u)&31u,(w>>20u)&31u,(w>>25u)&31u));}
// The spread reach word of tile t (umDeepAirKept's claims word, 30 bits). Saturated where no page holds t, or where
// the page's words are not this step's (REACH_STEP: claimed after the plan, read as the spread's AIR).
fn nbrTileReach(t:vec3i)->u32{
 let cs=nbCSlotOfPage(t>>vec3u(3u));
 if(cs==NB_NONE||nbSetWord(NB_S_REACH_STEP+cs)!=nbRing(cfg.x,NB_RING_STEP)+1u){return 0x3fffffffu;}
 return nbSetWord(NB_S_REACH1+512u*cs+nbrTileIndex(t))&0x3fffffffu;
}
// umPlaneReach[k]: this step's largest dt·away / h over face direction k's released wall tiles.
fn nbrPlaneReach(k:u32)->f32{return bitcast<f32>(nbSetWord(NB_H_WALL_REACH0+6u*cfg.y+k));}
// umFaceTile's word of C slot b's face f at tile t (only t's in-plane axes count).
fn nbrFaceTile(b:u32,f:u32,t:vec3i)->u32{let a=f>>1u;let q=vec3u(t&vec3i(7));return NB_S_WALL_TILE+384u*b+64u*f+q[(a+1u)%3u]+8u*q[(a+2u)%3u];}
// umWallAhead (uniform-mixed-walls.wgsl.ts 80-103) in cells: blocks hold the C slot of each probe column's page at
// the wall, NB_NONE for none.
struct NbrWallAhead {plane:i32,blocks:array<u32,4>}
fn nbrWallAhead(p:vec3i,k:u32,reach:f32,bound:f32)->NbrWallAhead{
 var ahead=NbrWallAhead(0,array<u32,4>(NB_NONE,NB_NONE,NB_NONE,NB_NONE));
 let axis=k>>1u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;let upper=(k&1u)!=0u;let inward=select(1.0,-1.0,upper);
 let slack=1e-5;
 let mask=u32((p[u]&31)==0)|(u32((p[v]&31)==0)<<1u);
 var plane=select(p[axis]&(-32),(p[axis]+31)&(-32),upper);
 var first=(p[axis]&31)==0;
 for(var n=0u;n<64u;n++){
  if(reach-inward*f32(p[axis]-plane)+slack<bound){return ahead;}
  var blocks=array<u32,4>(NB_NONE,NB_NONE,NB_NONE,NB_NONE);var open=false;var wall=false;
  for(var c=0u;c<4u;c++){
   if((c&mask)!=c){blocks[c]=blocks[c&mask];continue;}
   var q=p;q[axis]=plane-select(0,1,upper);q[u]-=i32(1u-(c&1u));q[v]-=i32(1u-(c>>1u));
   q=q>>vec3u(5u);let b=nbrPageKind(q);let bs=q;q[axis]+=select(-1,1,upper);let across=nbrPageKind(q);
   if(across==NBR_STORED){open=open||first||b==NBR_STORED;}
   else if(across!=NBR_AIR&&b==NBR_STORED){blocks[c]=nbCSlotOfPage(bs);wall=true;}
  }
  if(wall){ahead.plane=plane;ahead.blocks=blocks;return ahead;}
  if(!open){return ahead;}
  plane+=select(-32,32,upper);first=false;
 }
 return ahead;
}
// The wall face of in-lattice cell q's owner on axis a (upper: plane N), as wallReach and umReleasedWalls read it
// (umOwnerAt, umLoadWallTexel, negative[], the .w bit): x the owner's width, y U at the face's storage (m/s), z 1
// where FLAGS bit 8 + a there is set (lane P's release).
fn nbrWallOwner(q:vec3i,a:u32,upper:bool)->vec3f{
 let w=select(4.0,1.0,nbrCellOwnerH(q));
 var s=q;if(upper){s[a]+=1;}
 var speed=0.0;var flags=0u;
 if(w==1.0){
  let fs=nbFSlotOfCell(s);
  if(fs<NB_SEAM){speed=vel[NB_VEL_F_U0+NB_STRIDE*(fs+a*NB_FCAP)+nbLocal(s)];flags=phi[NB_PHI_F_FLAGS+NB_STRIDE*fs+nbLocal(s)];}
 } else {
  let T=s>>vec3u(2u);let cs=nbCSlotOfPage(T>>vec3u(3u));
  if(cs!=NB_NONE){speed=vel[NB_VEL_C_U0+NB_STRIDE*(cs+a*NB_CCAP)+nbrTileIndex(T)];flags=phi[NB_PHI_C_FLAGS+NB_STRIDE*cs+nbrTileIndex(T)];}
 }
 return vec3f(w,speed,f32((flags>>(8u+a))&1u));
}
// umReleasedWalls (uniform-mixed-surface.ts 829-870): the vertex at p (value: its advected φ after the wall
// contact) rises to the released walls' gaps.
fn nbrReleasedWalls(p:vec3f,value:f32)->f32{
 var result=value;let dt=nbrDt();let hm=nbrHm();let at=vec3i(p);
 for(var k=0u;k<6u;k++){
  let axis=k>>1u;let upper=(k&1u)!=0u;let inward=select(1.0,-1.0,upper);
  let reach=nbrPlaneReach(k);
  if(reach<=0.0){continue;}
  let wall=nbrWallAhead(at,k,reach,result);
  let depth=inward*(p[axis]-f32(wall.plane));
  var whole=1e30;var probes=0u;
  for(var c=0u;c<4u;c++){
   if(wall.blocks[c]==NB_NONE){continue;}
   var cell=at;cell[axis]=wall.plane-select(0,1,upper);cell[(axis+1u)%3u]-=i32(1u-(c&1u));cell[(axis+2u)%3u]-=i32(1u-(c>>1u));
   let o=nbrWallOwner(cell,axis,upper);
   let away=inward*o.y;
   let resolved=o.x==1.0||value>-0.5*o.x;
   let g=dt*away/hm;
   let gap=select(0.0,g,o.z!=0.0&&g>1e-4*o.x);
   if(resolved&&gap>0.0){result=max(result,gap-depth);}
   whole=min(whole,gap);probes++;
  }
  if(probes>0u&&whole>0.0){result=max(result,whole-depth);}
 }
 return result;
}
// umReleasedMayChange (uniform-mixed-surface.ts 1316-1333): some released wall tile can raise value at p.
fn nbrReleasedMayChange(p:vec3f,value:f32)->bool{
 let slack=1e-5;let at=vec3i(p);
 for(var k=0u;k<6u;k++){
  let reach=nbrPlaneReach(k);
  if(!(reach>0.0)){continue;}
  let axis=k>>1u;let inward=select(1.0,-1.0,(k&1u)!=0u);
  let wall=nbrWallAhead(at,k,reach,value);
  let depth=inward*(p[axis]-f32(wall.plane));
  var local=0u;
  for(var c=0u;c<4u;c++){
   if(wall.blocks[c]==NB_NONE){continue;}
   var cell=at;cell[(axis+1u)%3u]-=i32(1u-(c&1u));cell[(axis+2u)%3u]-=i32(1u-(c>>1u));
   local=max(local,nbSetWord(nbrFaceTile(wall.blocks[c],k,cell>>vec3u(2u))));
  }
  let tileReach=bitcast<f32>(local);
  if(tileReach>0.0&&!(tileReach-depth+slack<value)){return true;}
 }
 return false;
}
// umDeepAirKept (uniform-mixed-surface.ts 1398-1406): the stored value (value, φ at vertex v of width w) where v is
// deep air of a still tile and no released wall can raise it, else -1 (traced). umAirSource is the identity.
fn nbrDeepAirKept(v:vec3i,width:u32,value:f32)->f32{
 if(!(value>=6.0+3.0*f32(width))){return -1.0;}
 let t=clamp(v,vec3i(0),NB_LATTICE-vec3i(1))>>vec3u(2u);
 if(any(nbrReachMax(nbrTileReach(t))>vec3u(2u))||nbrReleasedMayChange(vec3f(v),value)){return -1.0;}
 return value;
}
`;

/** The producers' shared code: the readers (page kinds, wall owners) and the US owners (nrs prefix). */
const NR_PRODUCER_WGSL = () => [
  levelSetReleaseWGSL(),
  brickOwnerWGSL({ phi: "phi", vel: "vel", prefix: "nrs", faceF: "NB_VEL_F_US0", faceC: "NB_VEL_C_US0" }), /* wgsl */`
const NR_SATURATED:u32=31u;
// umReach and umPackReach (uniform-mixed-frame-plan.ts 149-153).
fn nrReachOf(travel:f32)->u32{
 let tiles=ceil(travel*1.00001/4.0)+2.0;
 return select(NR_SATURATED,u32(tiles),tiles<f32(NR_SATURATED));
}
fn nrPack(plus:vec3u,minus:vec3u)->u32{return plus.x|(plus.y<<5u)|(plus.z<<10u)|(minus.x<<15u)|(minus.y<<20u)|(minus.z<<25u);}
// umCertificateRadius (136-139): the spread's tile radius from the plan's speed (m/s).
fn nrRadius(speed:f32)->u32{return u32(min(ceil(speed*nbrDt()/nbrHm()*1.00001/4.0)+2.0,1e8));}
fn nrUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
// umCellPresent: c's page is resident (in the lattice).
fn nrPresent(c:vec3i)->bool{return nbInLattice(c)&&nbCSlotOfPage(c>>vec3u(5u))!=NB_NONE;}
// umSignedFaceExtent's accumulation: x lo, y hi, z 0 once a face is non-finite.
fn nrTake(e:vec3f,x:f32)->vec3f{return vec3f(min(e.x,x),max(e.y,x),select(0.0,e.z,abs(x)<=3.402823e38));}
fn nrEmpty()->vec3f{return vec3f(3.402823e38,-3.402823e38,1.0);}
// The owner's contribution: reach words (plus a at 0..2, minus at 3..5) and the plan speed's bits.
struct NrReach {r:array<u32,6>,fastest:u32}
fn nrContribution(e:array<vec3f,3>,scale:f32)->NrReach{
 var o=NrReach(array<u32,6>(0u,0u,0u,0u,0u,0u),0u);
 for(var a=0u;a<3u;a++){
  let finite=e[a].z!=0.0&&e[a].x<=e[a].y;
  o.r[a]=select(NR_SATURATED,nrReachOf(max(0.0,-e[a].x)*scale),finite);
  o.r[3u+a]=select(NR_SATURATED,nrReachOf(max(0.0,e[a].y)*scale),finite);
  o.fastest=max(o.fastest,select(0x7f800000u,bitcast<u32>(max(abs(e[a].x),abs(e[a].y))),finite));
 }
 return o;
}
// An h cell's extents (umSignedFaceExtent, unit owner): its +face and, where umLowOpen holds, its low face.
// fs: c's F slot (c is h-owned); a +face inside the brick is this slot's own word.
fn nrCellExtent(fs:u32,c:vec3i)->array<vec3f,3>{
 var e:array<vec3f,3>;
 for(var a=0u;a<3u;a++){
  var x=nrEmpty();
  let up=c+nrUnit(a);
  if(((c[a]+1)&7)!=0){x=nrTake(x,vel[NB_VEL_F_US0+NB_STRIDE*(fs+a*NB_FCAP)+nbLocal(up)]);}
  else{x=nrTake(x,nrsFacePlane(up,a));}
  if((c[a]&31)==0&&!nrPresent(c-nrUnit(a))){x=nrTake(x,vel[NB_VEL_F_US0+NB_STRIDE*(fs+a*NB_FCAP)+nbLocal(c)]);}
  e[a]=x;
 }
 return e;
}
// A 4h tile's extents: its +face (the upper neighbour's C face, or its 16 h faces) and its open low face (C at T).
fn nrTileExtent(T:vec3i)->array<vec3f,3>{
 var e:array<vec3f,3>;
 for(var a=0u;a<3u;a++){
  var x=nrEmpty();let u=(a+1u)%3u;let v=(a+2u)%3u;
  let up=4*(T+nrUnit(a));
  if(nrsCellOwnerH(up)){for(var i=0;i<16;i++){var c=up;c[u]+=i&3;c[v]+=i>>2u;x=nrTake(x,nrsFFace(c,a));}}
  else{x=nrTake(x,nrsCFace(T+nrUnit(a),a));}
  if(((4*T[a])&31)==0&&!nrPresent(4*T-nrUnit(a))){x=nrTake(x,nrsCFace(T,a));}
  e[a]=x;
 }
 return e;
}
// An h tile with no F slot: every face its absent brick stores reads 0; the rest are its top layer's +faces that
// cross into the next brick (where the tile is the brick's upper tile on that axis).
fn nrAbsentExtent(T:vec3i)->array<vec3f,3>{
 var e:array<vec3f,3>;
 for(var a=0u;a<3u;a++){
  var x=nrTake(nrEmpty(),0.0);let u=(a+1u)%3u;let v=(a+2u)%3u;
  if((T[a]&1)==1){for(var i=0;i<16;i++){var c=4*T;c[a]+=4;c[u]+=i&3;c[v]+=i>>2u;x=nrTake(x,nrsFacePlane(c,a));}}
  e[a]=x;
 }
 return e;
}
fn nrParityClear(g:u32,lid:u32,base:u32,n:u32){
 if(g==0u&&lid<n){atomicStore(&nbSet[base+n*(1u-cfg.y)+lid],0u);}
}
`].join("\n");

const NR_REACH_WGSL = /* wgsl */`
var<workgroup> wReach:array<atomic<u32>,48>;
var<workgroup> wFastest:atomic<u32>;
var<workgroup> wSplit:u32;
fn nrMaxInto(i:u32,o:NrReach){for(var k=0u;k<6u;k++){atomicMax(&wReach[6u*i+k],o.r[k]);}atomicMax(&wFastest,o.fastest);}
fn nrPage(cs:u32,lid:u32,scale:f32){
 let O=nbCOrigin(cs)>>vec3u(2u);
 if(lid==0u){atomicStore(&nbSet[NB_S_REACH_STEP+cs],nbRing(cfg.x,NB_RING_STEP)+1u);}
 var fastest=0u;
 for(var r=0u;r<2u;r++){
  let j=lid+256u*r;let T=O+nbLocalCoord(j);let at=NB_S_REACH0+512u*cs+j;
  if(!nrsTileInLattice(T)){atomicStore(&nbSet[at],0u);continue;}
  var e:array<vec3f,3>;
  if(nrsTileOwnerH(T)){
   if(nbFSlotOfCell(4*T)<NB_SEAM){continue;}
   e=nrAbsentExtent(T);
  } else {e=nrTileExtent(T);}
  let o=nrContribution(e,scale);
  atomicStore(&nbSet[at],nrPack(vec3u(o.r[0],o.r[1],o.r[2]),vec3u(o.r[3],o.r[4],o.r[5])));
  fastest=max(fastest,o.fastest);
 }
 atomicMax(&nbSet[NB_H_REL_SPEED0+cfg.y],fastest);
}
@compute @workgroup_size(256) fn nrReach(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 if(lid==0u){wSplit=nbListCount(NB_LIST_F_ALL);atomicStore(&wFastest,0u);}
 if(lid<48u){atomicStore(&wReach[lid],0u);}
 let split=workgroupUniformLoad(&wSplit);
 let g=nbGroup(wid);
 nrParityClear(g,lid,NB_H_REL_SPEED0,1u);
 let scale=nbrDt()/nbrHm();
 if(g>=split){if(g-split<nbListCount(NB_LIST_C_ALL)){nrPage(nbListSlot(NB_LIST_C_ALL,g-split),lid,scale);}return;}
 let fs=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(fs);
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let q=nbLocalCoord(l);let c=o+q;
  if(!nrsTileOwnerH(c>>vec3u(2u))){continue;}
  let i=u32(q.x>>2u)|(u32(q.y>>2u)<<1u)|(u32(q.z>>2u)<<2u);
  nrMaxInto(i,nrContribution(nrCellExtent(fs,c),scale));
 }
 workgroupBarrier();
 if(lid<8u){
  let T=(o>>vec3u(2u))+vec3i(i32(lid&1u),i32((lid>>1u)&1u),i32(lid>>2u));
  if(nrsTileOwnerH(T)){
   let cs=nbCSlotOfPage(T>>vec3u(3u));let w=lid*6u;
   atomicStore(&nbSet[NB_S_REACH0+512u*cs+nrsTileIndex(T)],nrPack(vec3u(atomicLoad(&wReach[w]),atomicLoad(&wReach[w+1u]),atomicLoad(&wReach[w+2u])),vec3u(atomicLoad(&wReach[w+3u]),atomicLoad(&wReach[w+4u]),atomicLoad(&wReach[w+5u]))));
  }
 }
 if(lid==0u){atomicMax(&nbSet[NB_H_REL_SPEED0+cfg.y],atomicLoad(&wFastest));}
}
`;

const NR_SPREAD_WGSL = /* wgsl */`
@compute @workgroup_size(256) fn nrSpread(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_C_ALL)){return;}
 let cs=nbListSlot(NB_LIST_C_ALL,g);let axis=cfg.z;
 let src=select(NB_S_REACH0,NB_S_REACH1,axis==1u);let dst=select(NB_S_REACH1,NB_S_REACH0,axis==1u);
 let reach=i32(min(nrRadius(bitcast<f32>(nbSetWord(NB_H_REL_SPEED0+cfg.y))),0x7fffffffu>>2u));
 let last=(NB_LATTICE[axis]>>2u)-1;
 let O=nbCOrigin(cs)>>vec3u(2u);
 for(var r=0u;r<2u;r++){
  let j=lid+256u*r;let p=O+nbLocalCoord(j);
  if(!nrsTileInLattice(p)){atomicStore(&nbSet[dst+512u*cs+j],0u);continue;}
  var plus=vec3u(0u);var minus=vec3u(0u);
  // The window held to the lattice: a tile beyond it is SOLID, the spread's zero boundary.
  for(var q=max(p[axis]-reach,0);q<=min(p[axis]+reach,last);q++){
   var t=p;t[axis]=q;let page=t>>vec3u(3u);
   if(nbrPageKind(page)!=NBR_STORED){plus=vec3u(NR_SATURATED);minus=plus;break;}
   let value=nbSetWord(src+512u*nbCSlotOfPage(page)+nrsTileIndex(t));
   plus=max(plus,vec3u(value&31u,(value>>5u)&31u,(value>>10u)&31u));
   minus=max(minus,vec3u((value>>15u)&31u,(value>>20u)&31u,(value>>25u)&31u));
  }
  atomicStore(&nbSet[dst+512u*cs+j],nrPack(plus,minus));
 }
}
`;

const NR_WALL_WGSL = /* wgsl */`
// wallReach (uniform-mixed-surface.ts 1682-1700): a lane a tile of each wall face of the page.
@compute @workgroup_size(256) fn nrWallReach(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);
 nrParityClear(g,lid,NB_H_WALL_REACH0,6u);
 if(g>=nbListCount(NB_LIST_C_ALL)){return;}
 let cs=nbListSlot(NB_LIST_C_ALL,g);let P=nbCOrigin(cs)>>vec3u(5u);
 if(nbrPageKind(P)==NBR_SOLID){return;}
 let dt=nbrDt();let hm=nbrHm();
 for(var i=lid;i<384u;i+=256u){
  let f=i>>6u;let lane=i&63u;let axis=f>>1u;let upper=(f&1u)!=0u;let inward=select(1.0,-1.0,upper);
  var across=P;across[axis]+=select(-1,1,upper);
  if(nbrPageKind(across)!=NBR_SOLID){continue;}
  var tile=8*P;tile[axis]+=select(0,7,upper);tile[(axis+1u)%3u]+=i32(lane&7u);tile[(axis+2u)%3u]+=i32(lane>>3u);
  let width=select(4u,1u,nbrTileOwnerH(tile));var bits=0u;
  for(var j=0u;j<16u;j+=width*width){
   var cell=4*tile;cell[axis]+=select(0,3,upper);cell[(axis+1u)%3u]+=i32(j&3u);cell[(axis+2u)%3u]+=i32(j>>2u);
   let o=nbrWallOwner(cell,axis,upper);
   let gap=dt*(inward*o.y)/hm;
   if(o.z!=0.0&&gap>1e-4*o.x){bits=max(bits,bitcast<u32>(gap));}
  }
  atomicStore(&nbSet[NB_S_WALL_TILE+384u*cs+i],bits);
  if(bits!=0u){atomicMax(&nbSet[NB_H_WALL_REACH0+6u*cfg.y+f],bits);}
 }
}
`;

export interface LevelSetReleaseKernels { readonly nrReach: BrickKernel; readonly nrSpread: BrickKernel; readonly nrWallReach: BrickKernel }

/** The producers' bindings: the readers' (set, dir, vel, phi), read-only but the set. */
const NR_BINDINGS: readonly BrickBinding[] = [CFG, { name: "set", kind: "rw", type: "array<atomic<u32>>" }, { name: "dir", kind: "ro", type: "array<u32>" },
  { name: "vel", kind: "ro", type: "array<f32>" }, { name: "phi", kind: "ro", type: "array<u32>" }];

/** The producers. nrReach: FC_G1; nrSpread: C_ALL G1, cfg.z the axis; nrWallReach: C_ALL G1. */
export function levelSetReleaseKernels(L: BrickLayout): LevelSetReleaseKernels {
  const k = (label: string, body: string): BrickKernel => ({
    label, entry: label, bindings: NR_BINDINGS,
    code: [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: false } }), declareBindings(NR_BINDINGS), NR_PRODUCER_WGSL(), body].join("\n"),
  });
  return { nrReach: k("nrReach", NR_REACH_WGSL), nrSpread: k("nrSpread", NR_SPREAD_WGSL), nrWallReach: k("nrWallReach", NR_WALL_WGSL) };
}

/** The producers' placements: part 1 (after the head, before move) and part 2 (after the lists, before levelSet). */
export function levelSetReleaseSchedule(L: BrickLayout, kernels = levelSetReleaseKernels(L)): { part1: BrickPlacement[]; part2: BrickPlacement[] } {
  const listC = { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G1 };
  return {
    part1: [
      { kernel: kernels.nrReach, dispatch: { indirect: ARGS.FC_G1 } },
      { kernel: kernels.nrSpread, dispatch: listC, z: 0 },
      { kernel: kernels.nrSpread, dispatch: listC, z: 1 },
      { kernel: kernels.nrSpread, dispatch: listC, z: 2 },
    ],
    part2: [{ kernel: kernels.nrWallReach, dispatch: listC }],
  };
}
