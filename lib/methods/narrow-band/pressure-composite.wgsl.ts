/**
 * Lane P's composite pressure rows on the brick tiers (brick-abi.md §2.5, §6.1): NB-FLIP's
 * system, from the old arena's pRows, pInit, pRebase and pNative, and from buildRhs and the
 * root authority. The formulas are pressure-rows.wgsl.ts's, checked bit for bit against the
 * old resolved text. This file supplies the addressing on the tier storage, through the owner
 * readers (brick-owner.wgsl.ts).
 *
 * Units are the slot contract's: X = p dt/ρ, and every old row is multiplied by h² dt/ρ. So
 * an h face's coupling is V/θ, a seam's V/(2.5θ'), a native 4h face's 4V (64 V/(4h)², the
 * band-equivalent row), and a native wall's 4·fraction. The rows are written in total
 * variables. The old rows are solved about a start, and the start enters here only through
 * the seam heads and the walls' initial state.
 *
 * - **rowsF** (F_ALL, the h rows of h-owned tiles). It writes X (the start), RHS, DIAG,
 *   K0..2 (the lower face's coupling, to a liquid h row or a native row below) and KU0..2 (a
 *   seam to a native row above).
 *   - Face kinds are npBandFace's. Across the lattice's faces is the separating wall (V 0.5).
 *     Across a non-resident page is K_OPEN.
 *   - The band sees a native row at X_n + head (pRebase: `PN_B + bSeamHead`), so a liquid
 *     row's RHS gains C·head.
 *   - RHS = −h Σ sign·fraction·US + h²·rate·capacity + Σ C·head + Σ_attached C·lift, where rate
 *     is lane F's FH.RATE (header word 64) and capacity is 1 with no solids.
 * - **rowsC** (C_ALL, the native rows of 4h-owned tiles). It writes X (the start), RHS, DIAG,
 *   K0..2 (native–native, lower face) and PPHI (rootPhi, the mean of the tile's 8 corners).
 *   - RHS = −16h Σ sign·V4·u4 + 64h²·rate + Σ_attached c·lift − Σ_seam ch·head. Here u4 is
 *     the 4h face's u*: the C face, the wall face, or the mean of its 16 h subfaces toward an
 *     h tile (the old to-pressure transfer).
 *   - The seam subfaces' coefficients are recomputed with the band's own face formula, so both
 *     sides hold the same number. An air subface adds to the diagonal only (a zero-pressure
 *     boundary).
 *   - The native ghost's θ is the root authority's (umFaceSurfaceTheta): the first zero
 *     crossing of the h vertex φ, marched from the liquid tile's centre in h steps. Its
 *     fallback is the centres' linear θ.
 * - **Walls** (K_WALL, N_WALL) are the separating wall's linearisation (Batty, Bertails &
 *   Bridson 2007; Chentanez & Müller 2012).
 *   - An attached wall's ghost is X + lift, so its coupling leaves the diagonal and the RHS
 *     gains C·lift. That cancels the wall's flux, which is the Neumann wall.
 *   - A separated wall's ghost is 0: the coupling stays on the diagonal.
 *   - The row build takes the state at the start (attached when start + lift ≥ 0, pRebase's
 *     first measure). The solver's outer loop re-decides it.
 *   - N_HELD (a native row deeper than its own width, rootPhi < −4) is always attached.
 * - **Starts** (pInit, buildRhs).
 *   - A warm cell (FLAGS bit 7: it held a liquid row of its tier last step) starts from X.
 *   - A tile's start is its C cell's X, the tile's last pressure: the native row's solution,
 *     or an h tile's mean over its liquid rows (083's band-tile mean), which the solve's finish
 *     writes. Air starts at 0. Reading C X rather than averaging the h rows' X keeps the row
 *     passes free of read-after-write races: rowsF writes F X, and only rowsC (after it) writes
 *     C X, and only on dry native rows.
 *   - A cold h row prolongs the tiles' starts trilinearly, with the band's surface carry
 *     (bCoarse, bCoarseCarry), verbatim. Tile coordinates clamp at held pages, which are the
 *     pages beyond the lattice.
 *
 * Unported pieces latch FAIL.UNSUPPORTED when they would activate: a cut brick, or a BULK16
 * page across a face. There are no solids, so bounded rows (bInside, N_CLOSED) never arise.
 */
import { BOX_TOKEN, boxPadded, type BrickLayout, resolveBox } from "./brick-abi";
import type { BrickBinding } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { brickCellKernel } from "./brick-pressure.wgsl";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { CENSUS, RG } from "./brick-census.wgsl";
import { PRESSURE_ROWS_WGSL } from "./pressure-rows.wgsl";
import { BRICK_SOURCE_RING_WGSL } from "./brick-source.wgsl";

/** The root's far air in cells (UNIFORM_MIXED_ROOT_FAR_CELLS, uniform-mixed-root-owners.ts): RO_FAR. */
const RO_FAR_CELLS = 16;

/**
 * FLAGS bits of lane P (brick-abi.md §2.4). WALL0 << a is the separating wall on the lower face along a, at the
 * face's storage (plane 0: the cell or tile itself; plane N: the skirt cell or tile N/4). From the rows to the
 * solve's finish it is the solver's state (1 = separated), which the build sets and the MGPCG's outer loop
 * re-decides. After project it is the old umRelease of the face (lane E's released walls): not held, the
 * projection's wall ghost <= 0, and -sign U dt > 1e-4 width h.
 */
export const P_FLAG = { WARM: 1 << 7, WALL0: 1 << 8, WALLS: 7 << 8 } as const;
/** The separating wall's coupling in the slot units: an h row's K_WALL (npBandFace, V 0.5), a native row's 4·0.5. */
export const P_WALL_C = { F: 0.5, C: 2 } as const;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const presRW: BrickBinding = { name: "pres", kind: "rw", type: "array<f32>" };
const cutRO: BrickBinding = { name: "cut", kind: "ro", type: "array<f32>" };
const velRW: BrickBinding = { name: "vel", kind: "rw", type: "array<f32>" };
const phiRW: BrickBinding = { name: "phi", kind: "rw", type: "array<u32>" };
// The rows write the wall state's FLAGS bits, so phi is read-write in every composite kernel.
const ROWS = [CFG, setRW, dirRO, velRO, phiRW, presRW, cutRO];
const PROJECT = [CFG, setRW, dirRO, velRW, phiRW, presRW, cutRO];
const FACES: BrickBinding = { name: "pmgF", kind: "rw", type: "array<f32>" };
// Still-tank A: projectC stores a STILL page's discarded correction in the census's region words (RG FROZEN).
const CENSUS_RW: BrickBinding = { name: "census", kind: "rw", type: "array<u32>" };

/** The shared readers: centre and face φ, root φ, starts, the marched θ and the band face. */
// The readers' sources: poVertex (nbVertexPhi), poCentre (poCentreG) and poOwnerH (nbCellOwnerH). rowsF stages
// its brick's halo in workgroup memory (vertices o-1..o+9, centres o-1..o+8, the tiles 2B-1..2B+2 of a brick at
// o = 8B), each value computed by the same function as the direct reader, so the stage changes no value.
const STAGE_DIRECT = /* wgsl */`
fn poVertex(v:vec3i)->f32{return nbVertexPhi(v);}
fn poCentre(c:vec3i)->f32{return poCentreG(c);}
fn poOwnerH(c:vec3i)->bool{return nbCellOwnerH(c);}
fn poRootPhi(T:vec3i)->f32{return poRootPhiG(T);}
fn poTileStart(T:vec3i)->f32{return poTileStartG(T);}
fn poStartRoot(T:vec3i)->f32{return poRootPhiG(T);}
fn poStartTile(T:vec3i)->f32{return poTileStartG(T);}
fn poSlotOfCell(c:vec3i)->u32{return nbFSlotOfCell(c);}
fn poSlotRec(c:vec3i,k:u32)->u32{let s=nbFSlotOfCell(c);if(s>=NB_SEAM){return NB_NONE;}return nbFRec(s,k);}
fn poGuardVertex(v:vec3i){poGuardVertexG(v);}
fn poFacePlane(c:vec3i,a:u32)->f32{return nbFacePlane(c,a);}
`;
const STAGE_BRICK = /* wgsl */`
var<workgroup> poSV:array<f32,1331>;
var<workgroup> poSC:array<f32,1000>;
var<workgroup> poST:array<u32,64>;
var<private> poSO:vec3i;
// poStage stages the vertices and centres (poVS); projectF's light stage (poStageLight) stages only the owners and
// slots, and its few band faces then read the direct functions, whose values the stage reproduces.
var<private> poVS:bool;
fn poIndex(d:vec3i,n:i32)->u32{return u32(d.x+n*(d.y+n*d.z));}
fn poCoord(i:u32,n:u32)->vec3i{return vec3i(i32(i%n),i32((i/n)%n),i32(i/(n*n)));}
fn poVertex(v:vec3i)->f32{let d=v-poSO+vec3i(1);if(poVS&&all(d>=vec3i(0))&&all(d<vec3i(11))){return poSV[poIndex(d,11)];}return nbVertexPhi(v);}
fn poCentre(c:vec3i)->f32{let d=c-poSO+vec3i(1);if(poVS&&all(d>=vec3i(0))&&all(d<vec3i(10))){return poSC[poIndex(d,10)];}return poCentreG(c);}
fn poOwnerH(c:vec3i)->bool{let d=(c>>vec3u(2u))-(poSO>>vec3u(2u))+vec3i(1);if(all(d>=vec3i(0))&&all(d<vec3i(4))){return poST[poIndex(d,4)]!=0u;}return nbCellOwnerH(c);}
// The tiles a cold row's start reads (poStart: 2B-1..2B+2, the carry's beyond 2B-2..2B+3 along y; a face halo
// cell's taps too, poTileNear clamping only toward its own tile): their root φ and start, staged only when the brick
// or its face halo has a cold row (poTiles).
var<workgroup> poSR:array<f32,96>;
var<workgroup> poSS:array<f32,96>;
var<private> poRC:bool;
fn poTileWindow(T:vec3i)->i32{let d=T-(poSO>>vec3u(2u))+vec3i(1,2,1);if(all(d>=vec3i(0))&&all(d<vec3i(4,6,4))){return d.x+4*(d.y+6*d.z);}return -1;}
fn poRootPhi(T:vec3i)->f32{if(poRC){let i=poTileWindow(T);if(i>=0){return poSR[i];}}return poRootPhiG(T);}
fn poTileStart(T:vec3i)->f32{if(poRC){let i=poTileWindow(T);if(i>=0){return poSS[i];}}return poTileStartG(T);}
// The cold start's reads: the window alone. Inline at each of the start's taps, the global fallback set nbOperator's
// register peak (Metal's cap 384 threads under its 512 lanes, so it spilled; 512 without). The window holds every
// tile the start reaches, so a read outside it (or unstaged) is a defect and latches.
fn poWindowAt(T:vec3i)->u32{let i=poTileWindow(T);if(i<0||!poRC){nbFail(NB_FAIL_UNSUPPORTED,0xa6000000u);return 0u;}return u32(i);}
fn poStartRoot(T:vec3i)->f32{return poSR[poWindowAt(T)];}
fn poStartTile(T:vec3i)->f32{return poSS[poWindowAt(T)];}
// A face halo cell of the brick at origin o for lanes 0..383 (six faces of 64), as the window's 10^3 offset.
fn poHaloD(lid:u32)->vec3i{
  let side=lid/64u;let a=side/2u;let k=lid%64u;var d=vec3i(0);
  d[(a+1u)%3u]=i32(k%8u)+1;d[(a+2u)%3u]=i32(k/8u)+1;d[a]=select(0,9,(side&1u)==1u);
  return d;
}
// Called in uniform control flow (cold uniform) by every lane of the brick's workgroup, after poStage.
fn poTiles(lid:u32,cold:bool){
  poRC=false;
  if(cold){
    if(lid<96u){
      let T=(poSO>>vec3u(2u))-vec3i(1,2,1)+vec3i(i32(lid%4u),i32((lid/4u)%6u),i32(lid/24u));
      let root=poRootPhiG(T);poSR[lid]=root;poSS[lid]=poTileStartR(T,root);
    }
    workgroupBarrier();
  }
  poRC=cold;
}
// nbVertexPhi over the brick's vertex window with its lookups staged once (lane F's remedy for owner resolution per
// sample): the 5^3 tiles T0-2..T0+2 that can hold a vertex's authority (in the lattice, resident, 4h-owned), the C
// vertices of the 6^3 tiles T0-2..T0+3 an authority's corners reach, and the F slots of the 27 bricks around. The
// same candidate order and arithmetic as nbVertexPhi, so the stage changes no value.
var<workgroup> poSK:array<u32,125>;
var<workgroup> poSQ:array<f32,216>;
var<workgroup> poSB:array<u32,27>;
// The staged bricks' K slots (NB_REC_CUT) and classes (NB_REC_CLASS), NB_NONE off a slot: the cuts' readers ask them
// once a face of a row, about ten times a cell.
var<workgroup> poSBK:array<u32,27>;
var<workgroup> poSBC:array<u32,27>;
fn poVertexS(v:vec3i,o:vec3i)->f32{
  if(any(v<vec3i(0))||any(v>NB_LATTICE)){return NB_PHI_FAR;}
  let T0=o>>vec3u(2u);let tA=v>>vec3u(2u);let inside=(v-4*tA)!=vec3i(0);var T=vec3i(-1);
  for(var k=0u;k<8u;k++){
    let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);
    if(any(inside&(corner==vec3u(0u)))){continue;}
    let t=tA+vec3i(corner)-vec3i(1);
    if(poSK[poIndex(t-T0+vec3i(2),5)]!=0u){T=t;break;}
  }
  if(T.x<0){
    let s=poSB[poIndex((v>>vec3u(3u))-(o>>vec3u(3u))+vec3i(1),3)];if(s>=NB_SEAM){return NB_PHI_AIR_C;}
    return bitcast<f32>(phi[nbFPhiBase(s)+NB_STRIDE*s+nbLocal(v)]);
  }
  let t=vec3f(v-4*T)/4.0;var values:array<f32,8>;
  for(var k=0u;k<8u;k++){
    let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));let weight=w.x*w.y*w.z;
    if(weight>0.0){values[k]=weight*poSQ[poIndex(T+vec3i(corner)-T0+vec3i(2),6)];}
  }
  return ((values[0]+values[5])+(values[1]+values[4]))+((values[2]+values[7])+(values[3]+values[6]));
}
// The F slot of cell c from the staged bricks (nbFSlotOfCell outside them or at a negative brick).
fn poSlotOfCell(c:vec3i)->u32{
  let b=c>>vec3u(3u);let d=b-(poSO>>vec3u(3u))+vec3i(1);
  if(all(b>=vec3i(0))&&all(d>=vec3i(0))&&all(d<vec3i(3))){return poSB[poIndex(d,3)];}
  return nbFSlotOfCell(c);
}
// nbFRec(poSlotOfCell(c), k) for k NB_REC_CUT or NB_REC_CLASS from the staged bricks, NB_NONE off a slot.
fn poSlotRec(c:vec3i,k:u32)->u32{
  let b=c>>vec3u(3u);let d=b-(poSO>>vec3u(3u))+vec3i(1);
  if(all(b>=vec3i(0))&&all(d>=vec3i(0))&&all(d<vec3i(3))){let i=poIndex(d,3);return select(poSBC[i],poSBK[i],k==NB_REC_CUT);}
  let s=nbFSlotOfCell(c);if(s>=NB_SEAM){return NB_NONE;}return nbFRec(s,k);
}
// poGuardVertexG with the pages of the brick's vertices (o..o+9: pages o>>5 + {0,1}^3) staged.
var<workgroup> poSP:array<u32,8>;
fn poGuardVertex(v:vec3i){
  if(PO_TIERED&&all(v>vec3i(0))&&all(v<NB_LATTICE)){
    let d=(v>>vec3u(5u))-(poSO>>vec3u(5u));
    var held=false;if(all(d>=vec3i(0))&&all(d<vec3i(2))){held=poSP[poIndex(d,2)]!=0u;}else{held=nbCSlotOfPage(v>>vec3u(5u))!=NB_NONE;}
    if(!held){nbFail(NB_FAIL_TIER,nbKey(v>>vec3u(5u)));}
  }
}
// nbFacePlane with the owners (poOwnerH) and F slots (poSlotOfCell) staged.
fn poFacePlane(c:vec3i,a:u32)->f32{
  if(!nbFaceInLattice(c,a)){return 0.0;}
  var e=vec3i(0);e[a]=1;
  if(poOwnerH(c)||poOwnerH(c-e)){let s=poSlotOfCell(c);if(s>=NB_SEAM){return 0.0;}return vel[NB_VEL_F_US0+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(c)];}
  return nbCFace(c>>vec3u(2u),a);
}
// Called in uniform control flow by every lane of the brick's workgroup.
// n: the workgroup's lanes.
fn poStage(o:vec3i,lid:u32,n:u32){
  poSO=o;poVS=true;
  {
    if(lid<8u){poSP[lid]=u32(nbCSlotOfPage((o>>vec3u(5u))+poCoord(lid,2u))!=NB_NONE);}
    let T0=o>>vec3u(2u);
    if(lid<125u){let t=T0-vec3i(2)+poCoord(lid,5u);poSK[lid]=u32(nbTileInLattice(t)&&nbCSlotOfPage(t>>vec3u(3u))!=NB_NONE&&!nbTileOwnerH(t));}
    if(lid<216u){poSQ[lid]=nbCVertex(T0-vec3i(2)+poCoord(lid,6u));}
    if(lid<27u){let b=(o>>vec3u(3u))-vec3i(1)+poCoord(lid,3u);let s=select(nbFSlotOfBrick(b),NB_NONE,any(b<vec3i(0)));poSB[lid]=s;var k=NB_NONE;var cls=NB_NONE;if(s<NB_SEAM){k=nbFRec(s,NB_REC_CUT);cls=nbFRec(s,NB_REC_CLASS);}poSBK[lid]=k;poSBC[lid]=cls;}
    workgroupBarrier();
  }
  for(var i=lid;i<1331u;i+=n){poSV[i]=poVertexS(o-vec3i(1)+poCoord(i,11u),o);}
  if(lid<64u){poST[lid]=u32(nbTileOwnerH((o>>vec3u(2u))-vec3i(1)+poCoord(lid,4u)));}
  workgroupBarrier();
  for(var i=lid;i<1000u;i+=n){
    let d=poCoord(i,10u);var v:array<f32,8>;
    for(var j=0u;j<8u;j++){v[j]=poSV[poIndex(d+poCorner(j),11)];}
    poSC[i]=poMean8(v);
  }
  workgroupBarrier();
}
// The owners (poST) and F slots (poSB) only. Called in uniform control flow by every lane.
fn poStageLight(o:vec3i,lid:u32){
  poSO=o;poVS=false;
  if(lid<27u){let b=(o>>vec3u(3u))-vec3i(1)+poCoord(lid,3u);let s=select(nbFSlotOfBrick(b),NB_NONE,any(b<vec3i(0)));poSB[lid]=s;var k=NB_NONE;var cls=NB_NONE;if(s<NB_SEAM){k=nbFRec(s,NB_REC_CUT);cls=nbFRec(s,NB_REC_CLASS);}poSBK[lid]=k;poSBC[lid]=cls;}
  if(lid<64u){poST[lid]=u32(nbTileOwnerH((o>>vec3u(2u))-vec3i(1)+poCoord(lid,4u)));}
  workgroupBarrier();
}
`;

/** The cuts' readers (M5 design §5; brick-abi.md §2.3 with A1): a cell's open fraction FRAC, a lower face's aperture
 * AP and moving-wall velocity W from its brick's K slot. A brick without a slot is open (FRAC 1, AP 1, W 0), a cell
 * outside the lattice solid (FRAC 0, as pbStampCell's out-of-box voxel), and a lattice wall face is
 * 0.5 (FRAC low + FRAC high) (umPressureFaceV, pbStampFace): stored at plane 0, formed here at plane N, where no
 * brick holds the face. */
const CUTS_WGSL = /* wgsl */`
fn poCutK(c:vec3i)->u32{if(!nbInWallBox(c)){return NB_NONE;}return poSlotRec(c,NB_REC_CUT);}
fn poFrac(c:vec3i)->f32{if(!nbInWallBox(c)){return 0.0;}let k=poCutK(c);if(k==NB_NONE){return 1.0;}return cut[NB_CUT_FRAC+NB_STRIDE*k+nbLocal(c)];}
fn poCutAP(a:u32)->u32{return select(select(NB_CUT_AP2,NB_CUT_AP1,a==1u),NB_CUT_AP0,a==0u);}
fn poCutW(a:u32)->u32{return select(select(NB_CUT_W2,NB_CUT_W1,a==1u),NB_CUT_W0,a==0u);}
// The lower face of c along a (c may lie on the lattice's plane N, one past its last cell).
fn poFaceAP(c:vec3i,a:u32)->f32{
  let low=c-poUnit(a);
  if(!nbInWallBox(c)){return 0.5*(poFrac(low)+0.0);}
  let k=poCutK(c);if(k==NB_NONE){return select(0.5,1.0,nbInWallBox(low));}
  return cut[poCutAP(a)+NB_STRIDE*k+nbLocal(c)];
}
fn poFaceW(c:vec3i,a:u32)->f32{let k=poCutK(c);if(k==NB_NONE){return 0.0;}return cut[poCutW(a)+NB_STRIDE*k+nbLocal(c)];}
fn poCutBmask(k:u32)->u32{if(k==NB_NONE){return 0u;}return bitcast<u32>(cut[NB_CUT_BMASK+k]);}
// A static solid cell (lane E's 0xa2 hand-off, the coordinator's option (b), 2026-11-17): a resident cell of a static
// cut brick whose static open fraction is <= 1e-5 (umCellIsOpen's shut; FLAGS.STATIC_SHUT, the stamps'). It is never
// a row: the pressure sees it as the box's shell, so its faces to the liquid are separating walls (poAcross),
// re-decided as the box walls are (pmgWalls*). With no body (BMASK 0) the bit is the slot's FRAC <= 1e-5 (the test
// before 2026-11-18); with a body it tells the static cells from the body's, which stay the body's rows.
fn poStaticShut(q:vec3i)->bool{
  let k=poCutK(q);if(k==NB_NONE){return false;}
  if((poSlotRec(q,NB_REC_CLASS)&NB_CLASS_CUT_STATIC)==0u){return false;}
  return (poFlagsF(q)&NB_FLAG_STATIC_SHUT)!=0u;
}
// The pressure's liquid test of an h cell with centre φ p: wet, and not a static solid cell.
fn poRowWet(c:vec3i,p:f32)->bool{return poWet(p,1.0)&&!poStaticShut(c);}
// umCellInsideSolid of an h cell (FLAGS.INSIDE: a cut brick's stamps; 0 in a brick with no K slot, a C tile or past the box).
fn poInside(q:vec3i)->bool{if(poCutK(q)==NB_NONE){return false;}return (poFlagsF(q)&NB_FLAG_INSIDE)!=0u;}
`;

const compositeWGSL = (cuts: boolean) => /* wgsl */`
const PO_WARM:u32=${P_FLAG.WARM}u;const PO_WALL0:u32=${P_FLAG.WALL0}u;const PO_WALLS:u32=${P_FLAG.WALLS}u;
// The wall state's storage word: h cell c's FLAGS (a cell, or the skirt cell at plane N), tile T's C FLAGS (T
// up to N/4). A liquid row's wall storage is always resident (the residency dilates the skirt); missing, it
// latches FAIL.UNSUPPORTED.
fn poFlagsWordF(c:vec3i)->u32{let s=poSlotOfCell(c);if(s>=NB_SEAM){nbFail(NB_FAIL_UNSUPPORTED,0xa0000000u);return NB_NONE;}return NB_PHI_F_FLAGS+NB_STRIDE*s+nbLocal(c);}
fn poFlagsWordC(T:vec3i)->u32{let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,0xa1000000u);return NB_NONE;}return NB_PHI_C_FLAGS+NB_STRIDE*cs+nbTileIndex(T);}
// Writes the separated bit of the wall on axis a at FLAGS word w (one writer a word: the face's liquid row).
fn poSetWall(w:u32,a:u32,separated:bool){if(w==NB_NONE){return;}let bit=PO_WALL0<<a;phi[w]=(phi[w]&~bit)|select(0u,bit,separated);}
// The root's far air (RO_FAR, uniform-mixed-root-owners.ts): a tile no page stores reads 16 cells, in h.
const PO_RO_FAR:f32=${RO_FAR_CELLS.toFixed(1)};
fn poUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn poCorner(j:u32)->vec3i{return vec3i(i32(j&1u),i32((j>>1u)&1u),i32(j>>2u));}
// The old geometry's corner mean (gcEvaluate: each corner times 1/8, then d4Sum8's pairing).
fn poMean8(v:array<f32,8>)->f32{var c:array<f32,8>;for(var j=0u;j<8u;j++){c[j]=v[j]*0.125;}return ((c[0]+c[5])+(c[1]+c[4]))+((c[2]+c[7])+(c[3]+c[6]));}
// An h cell's centre φ: umPressurePhiCell, which with no solids is the cell's centerPhi texel, the
// geometry's corner mean of its 8 vertices (hanging ones trilinear in the 4h corners). Its closed-cell
// branch needs solids, and a cut brick latches UNSUPPORTED.
fn poCentreG(c:vec3i)->f32{var v:array<f32,8>;for(var j=0u;j<8u;j++){v[j]=nbVertexPhi(c+poCorner(j));}return poMean8(v);}
// The φ at the centre of c's face on side sign along a (bFacePhi): the mean of its 4 vertices.
fn poFacePhi(c:vec3i,a:u32,sign:i32)->f32{
  var o=c;if(sign>0){o[a]+=1;}let u=poUnit((a+1u)%3u);let v=poUnit((a+2u)%3u);
  return 0.25*(((poVertex(o)+poVertex(o+u))+poVertex(o+v))+poVertex(o+u+v));
}
fn poWet(phi:f32,width:f32)->bool{return npWet(phi,width,1.0);}
// Tiered, residency keeps every page within reach of liquid (setMarkPages: at least 7 cells), so a liquid
// row's own corner vertices are never in an absent page. If one is, the stencil read FAR: FAIL.TIER.
// All-h (PO_TIERED false), an absent page is h air like an absent brick (brick-abi.md §2.5 Residency): no check.
// Vertices on the lattice's faces (the wall skirt) are not checked.
fn poGuardVertexG(v:vec3i){if(PO_TIERED&&all(v>vec3i(0))&&all(v<NB_LATTICE)){if(nbCSlotOfPage(v>>vec3u(5u))==NB_NONE){nbFail(NB_FAIL_TIER,nbKey(v>>vec3u(5u)));}}}
fn poPageHeld(p:vec3i)->bool{return any(p<vec3i(0))||any(32*p>=NB_LATTICE);}
fn poTileOpen(t:vec3i)->bool{return nbTileInWallBox(t);}
// A tile's root φ (umAuthority): a 4h tile's corner mean (d4Sum8 of vertices·0.125); an h tile's
// held fit, the mean of its 64 h centres (pcHeldFit: weights 1/2 on the tile's faces, /64);
// RO_FAR off the lattice or in a non-resident page.
fn poRootPhi4(T:vec3i)->f32{var v:array<f32,8>;for(var j=0u;j<8u;j++){v[j]=nbCVertex(T+poCorner(j));}return poMean8(v);}
// An h tile's held fit is computed once a step by tileRoots (before the operator) into its C PPHI, which rowsC
// keeps; a tile whose brick holds no F slot (never on a resident h tile) sums its vertices here.
fn poRootPhiG(T:vec3i)->f32{
  if(!nbTileInLattice(T)){return PO_RO_FAR;}
  let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){return PO_RO_FAR;}
  if(!nbTileH(cs,nbTileIndex(T))){return poRootPhi4(T);}
  if(nbFSlotOfBrick(T>>vec3u(1u))<NB_SEAM){return pres[NB_PRES_C_PPHI+NB_STRIDE*cs+nbTileIndex(T)];}
  var total=0.0;
  for(var z=0;z<=4;z++){for(var y=0;y<=4;y++){for(var x=0;x<=4;x++){
    let g=vec3i(x,y,z);let w=select(vec3f(1.0),vec3f(0.5),(g==vec3i(0))|(g==vec3i(4)));
    total+=w.x*w.y*w.z*poVertex(4*T+g);
  }}}
  return total/64.0;
}
fn poFlagsF(c:vec3i)->u32{let s=poSlotOfCell(c);if(s>=NB_SEAM){return 0u;}return phi[NB_PHI_F_FLAGS+NB_STRIDE*s+nbLocal(c)];}
fn poXF(c:vec3i)->f32{let s=poSlotOfCell(c);if(s>=NB_SEAM){return 0.0;}return pres[NB_PRES_F_X+NB_STRIDE*s+nbLocal(c)];}
// A tile's start (R_PRESSURE, clamped as buildRhs: air 0): its C cell's X, which the solve's finish
// leaves as the tile's last pressure (a native row's solution; an h tile's mean over its liquid rows,
// 083's band-tile mean). No row pass writes it while another reads it.
fn poTileStartG(T:vec3i)->f32{return poTileStartR(T,poRootPhi(T));}
fn poTileStartR(T:vec3i,root:f32)->f32{
  if(!nbTileInLattice(T)){return 0.0;}
  let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){return 0.0;}
  if(!poWet(root,4.0)){return 0.0;}
  return pres[NB_PRES_C_X+NB_STRIDE*cs+nbTileIndex(T)];
}
// bCoarseAt (umTileNearFromPage): a tile beyond the home page clamps into it on each axis whose
// face neighbour page is held (beyond the lattice).
fn poTileNear(page:vec3i,t:vec3i)->vec3i{
  let low=page<<vec3u(3u);let high=low+vec3i(7);let d=vec3i(t>high)-vec3i(t<low);
  if(all(d==vec3i(0))){return t;}
  let held=vec3<bool>(d.x!=0&&poPageHeld(page+vec3i(d.x,0,0)),d.y!=0&&poPageHeld(page+vec3i(0,d.y,0)),d.z!=0&&poPageHeld(page+vec3i(0,0,d.z)));
  return select(t,clamp(t,low,high),held);
}
fn poCoarse(t:vec3i)->vec3f{
  let phi_=poStartRoot(t);
  return select(vec3f(0.0),vec3f(poStartTile(t),phi_,1.0),poWet(phi_,4.0)&&poTileOpen(t));
}
fn poCoarseKind(t:vec3i)->vec3f{
  let open=poTileOpen(t);let wet=poWet(poStartRoot(t),4.0);
  return vec3f(select(0.0,poStartTile(t),open&&wet),f32(open&&wet),f32(open&&!wet));
}
struct PoCarry {rise:vec2f,height:vec2f,air:f32}
fn poCoarseCarry(page:vec3i,b:vec3i,w:vec3f)->PoCarry{
  var carry:PoCarry;
  for(var k=0u;k<4u;k++){
    let bit=vec2u(k&1u,k>>1u);let sides=select(1.0-w.xz,w.xz,bit==vec2u(1u));let side=sides.x*sides.y;
    let c=b+vec3i(i32(bit.x),0,i32(bit.y));
    let t0=poTileNear(page,c);let t1=poTileNear(page,c+vec3i(0,1,0));let y0=t0.y;let y1=t1.y;
    let low=poCoarseKind(t0);var high=low;if(y1!=y0){high=poCoarseKind(t1);}
    let lower=side*(1.0-w.y);let upper=side*w.y;let liquid=lower*low.y+upper*high.y;
    carry.air+=lower*low.z+upper*high.z;
    carry.height+=vec2f(lower*low.y*f32(y0-b.y)+upper*high.y*f32(y1-b.y),liquid);
    if(y1!=y0&&low.y>0.0&&high.y>0.0){carry.rise+=liquid*vec2f(high.x-low.x,1.0);continue;}
    var own=low;var at=y0;var beyond=select(y0-1,y0+1,y0!=c.y);
    if(y1!=y0){if(high.y>0.0){own=high;at=y1;beyond=y1+1;}else{beyond=y0-1;}}
    if(own.y==0.0){continue;}
    let t2=poTileNear(page,vec3i(c.x,beyond,c.z));if(t2.y!=beyond){continue;}
    let other=poCoarseKind(t2);
    if(other.y>0.0){carry.rise+=liquid*vec2f((other.x-own.x)*f32(beyond-at),1.0);}
  }
  return carry;
}
// An h row's start (pInit): X when warm, else the tiles' starts prolonged with the surface carry.
fn poStart(c:vec3i,own:f32)->f32{
  if(!poWet(own,1.0)){return 0.0;}
  if((poFlagsF(c)&PO_WARM)!=0u){return poXF(c);}
  let y=(vec3f(c)+0.5)/4.0-0.5;let b=vec3i(floor(y));let w=y-floor(y);var sum=vec3f(0.0);
  let page=c>>vec3u(5u);
  for(var k=0u;k<8u;k++){
    let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let weights=select(1.0-w,w,bit==vec3u(1u));
    sum+=weights.x*weights.y*weights.z*poCoarse(poTileNear(page,b+vec3i(bit)));
  }
  var start=0.0;
  if(sum.z>=0.99999){start=sum.x/sum.z;}
  else if(sum.z>0.0){
    let carry=poCoarseCarry(page,b,w);
    if(carry.air>0.0||carry.rise.y==0.0){start=sum.x/sum.z*clamp(own*sum.z/min(sum.y,-1e-30),0.0,1000.0);}
    else{
      let mean=sum.x/sum.z;let bound=1000.0*mean;
      start=clamp(mean+(w.y-carry.height.x/carry.height.y)*carry.rise.x/carry.rise.y,min(bound,0.0),max(bound,0.0));
    }
  }
  return start;
}
// The native ghost's θ across tile t's upper face along a, n = t + e_a (umFaceSurfaceTheta).
fn poNativeTheta(t:vec3i,n:vec3i,a:u32)->f32{
  let pa=poRootPhi(t);let pb=poRootPhi(n);let la=poWet(pa,4.0);
  if(la==poWet(pb,4.0)){return 1.0;}
  var q=4*select(n,t,la)+vec3i(2);var previous=nbVertexPhi(q);let direction=select(-1,1,la);
  if(previous<0.0){
    for(var j=1u;j<=4u;j++){
      q[a]+=direction;let next=nbVertexPhi(q);
      if(next>=0.0){return clamp((f32(j-1u)+previous/(previous-next))/4.0,0.001,1000.0);}
      previous=next;
    }
  }
  return npTheta(select(pb,pa,la),select(pa,pb,la));
}
// What lies across an in-lattice cell's page: resident, or air (BULK16 latches UNSUPPORTED).
fn poAcross(q:vec3i)->u32{
  if(!nbInWallBox(q)){return NP_ACROSS_WALL;}
  let w=nbPageWord(q>>vec3u(5u));
  if(w==NB_NONE){return NP_ACROSS_AIR;}
  if(w>=NB_SEAM){nbFail(NB_FAIL_UNSUPPORTED,w);return NP_ACROSS_WALL;}${cuts ? `
  if(poStaticShut(q)){return NP_ACROSS_WALL;}` : ""}
  return NP_ACROSS_RESIDENT;
}
// The band face of h cell c (own φ, liquid) on face f (pRows' S path, no solids).
struct PoFace {face:NpFace,qWet:bool,native:bool}
fn poBandFace(c:vec3i,own:f32,liquid:bool,f:u32)->PoFace{
  let a=f/2u;let sign=npSign(f);let q=c+sign*poUnit(a);
  let across=poAcross(q);
  var band=false;var native=false;var n=NpCell(NB_PHI_AIR,false,false);
  if(across==NP_ACROSS_RESIDENT){
    band=poOwnerH(q);
    if(!band){let root=poRootPhi(q>>vec3u(2u));native=poWet(root,4.0);if(native){n=NpCell(root,true,true);}}
    if(!native){let p=poCentre(q);let wet=poWet(p,1.0);n=NpCell(p,wet,wet);}
  }
${cuts ? `  // The cuts (§5): the face's aperture, AP of its upper cell's lower face, and both cells' open fractions; the
  // solid flags as the old rows take them: shut = FRAC <= 1e-5 (umCellIsOpen), inside = FLAGS.INSIDE (the stamps'
  // umCellInsideSolid), read only where npBandFace's θ rule can take it (a wet-dry face, both open fractions < 0.1).
  let volume=poFaceAP(select(c,q,sign>0),a);let open=poFrac(c);var qOpen=1.0;if(across==NP_ACROSS_RESIDENT){qOpen=poFrac(q);}
  var fp=0.0;if(across==NP_ACROSS_RESIDENT&&liquid!=n.wet){fp=poFacePhi(c,a,sign);}
  var inside=false;var qInside=false;
  if(across==NP_ACROSS_RESIDENT&&liquid!=n.wet&&open<0.1&&qOpen<0.1){inside=poInside(c);qInside=poInside(q);}
  let face=npBandFace(across,own,liquid,1.0,1.0,band,native,n,fp,volume,false,open<=1e-5,inside,open,qOpen>1e-5,qInside,qOpen);` : `  let volume=select(0.5,1.0,nbInWallBox(q));
  // The face φ enters only the marched θ of a wet-dry face (npBandFace with solidClear): the other faces skip its
  // four vertices.
  var fp=0.0;if(across==NP_ACROSS_RESIDENT&&liquid!=n.wet){fp=poFacePhi(c,a,sign);}
  let face=npBandFace(across,own,liquid,1.0,1.0,band,native,n,fp,volume,true,false,false,1.0,true,false,1.0);`}
  return PoFace(face,n.wet,native);
}${cuts ? CUTS_WGSL : ""}
// poBandFace of h cell m on face f where the cell across lies in a liquid 4h-owned tile of a resident page with root
// φ root (a seam, seen from the native row): across RESIDENT, not band, native, n = (root, wet, liquid), volume 1.
fn poSeamFace(m:vec3i,own:f32,liquid:bool,f:u32,root:f32)->NpFace{
  let a=f/2u;let sign=npSign(f);
  return npBandFace(NP_ACROSS_RESIDENT,own,liquid,1.0,1.0,false,true,NpCell(root,true,true),poFacePhi(m,a,sign),1.0,true,false,false,1.0,true,false,1.0);
}
`;

/** The projection's readers: the rows' stored φ and start, and an h face's projected velocity. */
const PROJECT_WGSL = /* wgsl */`
fn poPhiF(c:vec3i)->f32{let s=poSlotOfCell(c);if(s>=NB_SEAM){return NB_PHI_AIR;}return pres[NB_PRES_F_PPHI+NB_STRIDE*s+nbLocal(c)];}
fn poBF(c:vec3i)->f32{let s=poSlotOfCell(c);if(s>=NB_SEAM){return 0.0;}return pres[NB_PRES_F_B+NB_STRIDE*s+nbLocal(c)];}
fn poXT(T:vec3i)->f32{
  if(!nbTileInLattice(T)){return 0.0;}
  let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){return 0.0;}
  return pres[NB_PRES_C_X+NB_STRIDE*cs+nbTileIndex(T)];
}
// An h row's seam head on face f (bSeamHead: its tangential neighbours' starts) from the rows' stored starts (B) and
// centre φ (PPHI): the head the row's RHS carried (rowsF's poHeadS).
fn poHeadB(c:vec3i,own:f32,base:f32,f:u32)->f32{
  var usable=0u;var bases:array<f32,6>;
  for(var g=0u;g<6u;g++){
    if(g/2u==f/2u){continue;}
    let m=c+npSign(g)*poUnit(g/2u);
    // poPhiF and poBF of m, loaded together (one round of loads, not a chain a neighbour).
    if(nbInWallBox(m)&&poOwnerH(m)){
      let s=poSlotOfCell(m);
      if(s<NB_SEAM){let at=NB_STRIDE*s+nbLocal(m);let pm=pres[NB_PRES_F_PPHI+at];let bm=pres[NB_PRES_F_B+at];if(poWet(pm,1.0)){usable|=1u<<g;bases[g]=bm;}}
    }
  }
  return npSeamHead(poWet(own,1.0),base,f,vec3u(c&vec3i(7)),usable,bases);
}
// The projected velocity of h cell m's face f (umProject's bFace in the slot units): U = US - sign (c/V) (ghost - X)/h,
// the ghost by the face's kind: a band row's X, a native row's X plus the seam head, 0 at air, and at a separating
// wall X + lift while attached in the solve's final state (separated: the FLAGS bit at the face), else 0. A Neumann
// face keeps US and a closed one is 0. From a dry h cell (liquid false) only a seam to a liquid native row moves
// (the native row's zero ghost); every other face keeps US.
fn poProjectFace(m:vec3i,own:f32,liquid:bool,f:u32,us:f32,h:f32,separated:bool)->f32{
  let face=poBandFace(m,own,liquid,f).face;
  var head=0.0;if(face.kind==NP_K_SEAM&&liquid){head=poHeadB(m,own,poBF(m),f);}
  return poProjectFaceK(m,own,liquid,f,us,h,separated,face.kind,face.coefficient/face.fraction,head);
}
// A liquid h row's seam ratio (coefficient / fraction): npBandFace's native face from a liquid side, θ unused (1),
// volume 1 (the native tile is resident, so in the lattice). rowsF's face cache holds a seam face's head in its
// ratio word, so projectF takes the ratio from here.
fn poSeamRatio()->f32{
  let face=npBandFace(NP_ACROSS_RESIDENT,-1.0,true,1.0,1.0,false,true,NpCell(-1.0,true,true),0.0,1.0,true,false,false,1.0,true,false,1.0);
  return face.coefficient/face.fraction;
}
// poProjectFace on a face's kind and coefficient / fraction (rowsF's face cache, or the formula's).
// head: a liquid side's seam head (rowsF's poHeadS, cached; poHeadB from the stored rows on the formula's path).
fn poProjectFaceK(m:vec3i,own:f32,liquid:bool,f:u32,us:f32,h:f32,separated:bool,kind:u32,ratio:f32,head:f32)->f32{
  let a=f/2u;let sign=npSign(f);
  if(kind==NP_K_NEUMANN){return us;}
  if(kind==NP_K_CLOSED){return 0.0;}
  if(!liquid&&kind!=NP_K_SEAM){return us;}
  let x=select(0.0,poXF(m),liquid);let q=m+sign*poUnit(a);
  var ghost=0.0;
  if(kind==NP_K_WALL){let lift=npLift(f,1.0,us,h,1.0);if(!separated){ghost=x+lift;}}
  else if(kind==NP_K_BAND){ghost=poXF(q);}
  else if(kind==NP_K_SEAM){ghost=poXT(q>>vec3u(2u));if(liquid){ghost+=head;}}
  return us-f32(sign)*ratio*(ghost-x)/h;
}
// The old umRelease (uniform-mixed-pressure-velocity.ts) of a lattice wall face with owner width w (1 or 4 cells):
// not held (the owner's centre or root φ >= -w), the projection's wall ghost (the wall slot's pressure) <= 0, and
// -sign U dt > 1e-4 w h, sign the face's side of its owner. A dry owner's ghost is 0.
fn poRelease(phiOwner:f32,ghost:f32,sign:f32,u:f32,w:f32,h:f32)->bool{
  let dt=nbRingF(cfg.x,NB_RING_DT);
  return !(phiOwner< -w)&&ghost<=0.0&&-sign*u*dt>1e-4*w*h;
}
`;

/**
 * The walls at the box (M3 padding; brick-abi.ts BOX_TOKEN): on a lattice padded past the container, the rows see the
 * box as their lattice. A cell or tile past it is no row and lies across a WALL face (poAcross, the native rows' wall
 * slot), the wall planes (the separated state, its release) are the box's, a cut reads it solid, and the owners'
 * nbOnWall / nbFaceInLattice / nbFacePlane are the box's (brickOwnerWGSL's box). The storage reads keep the lattice:
 * the staged vertex φ and authorities (nbVertexPhi's), the residency guards, the page clamps and the starts' slots.
 * Unpadded, every token resolves to the lattice's own name, so the text is byte-identical.
 */
export const PO_TILE_BOX_TOKEN = "nbTileInWallBox(";
/** poBandFace's last line, where a detail layout's coupled faces take their kind (pressureDetailFaces). */
const PO_BAND_RETURN = "  return PoFace(face,n.wet,native);\n}";

/**
 * A detail engine's coupled box faces (brick-detail.wgsl.ts): a face of the box that lies inside the parent's container
 * is the parent's velocity, prescribed (K_NEUMANN: U = US, the forces' relaxed value, whose flux enters the row's
 * divergence), not a wall. The nested domain's inflow condition (a one-way nest: the outer solution's normal velocity
 * on the inner boundary), so liquid crosses the box with the parent's flux; a wall would hold the box's volume fixed.
 * The flux is the cell's open fraction's (the whole face's), not the wall's half aperture. A closed or sealed face
 * (a static solid there) stays closed. Off a detail layout, or in a kernel with no band face, the code is unchanged.
 */
export function pressureDetailFaces(L: Pick<BrickLayout, "detail">, code: string): string {
  if (!L.detail || !code.includes("fn poBandFace(")) return code;
  if (code.split(PO_BAND_RETURN).length !== 2) throw new Error("Pressure: poBandFace's return moved (the detail faces)");
  const cuts = code.includes("fn poFrac(");
  return code.replace(PO_BAND_RETURN, "  return PoFace(poDetailFace(c,f,face),n.wet,native);\n}") + /* wgsl */`
const PO_DTL_COUPLED:u32=${L.detail.coupled}u;
fn poDetailFace(c:vec3i,f:u32,face:NpFace)->NpFace{
  if(((PO_DTL_COUPLED>>f)&1u)==0u||face.kind==NP_K_CLOSED){return face;}
  let q=c+npSign(f)*poUnit(f/2u);if(nbInWallBox(q)){return face;}
  return NpFace(NP_K_NEUMANN,0.0,${cuts ? "poFrac(c)" : "1.0"});
}
`;
}

export function pressureResolveBox(L: Pick<BrickLayout, "lattice" | "box" | "detail"> & { readonly staticTiles?: boolean }, code: string, tiered = true): string {
  code = pressureDetailFaces(L, code);
  const padded = boxPadded(L);
  // A native row is a whole 4h tile: a box plane inside a tile has no row form (no partial native row; refused).
  if (padded && tiered && L.box.some(n => n % 4 !== 0)) throw new Error(`Pressure: the walls at the box ${L.box.join("x")} need whole 4h tiles (each axis a multiple of 4) on a tiered lattice`);
  const tile = padded ? "fn poTileInBox(t:vec3i)->bool{return all(t>=vec3i(0))&&all(4*t<NB_BOX);}\n" : "";
  const out = resolveBox(L, (code.includes(PO_TILE_BOX_TOKEN) ? tile : "") + code.replaceAll(PO_TILE_BOX_TOKEN, L.staticTiles ? "nbTileInWallS(" : padded ? "poTileInBox(" : "nbTileInLattice("));
  if (out.includes(BOX_TOKEN) || out.includes("WallBox(")) throw new Error("Pressure: a wall token is unresolved");
  return out;
}

// Still-tank A: a STILL page's held face, as the solve reads it, is the old COLD freeze's u* = F + g dt (umColdForced:
// F the held face, g dt where its last transfer's gravity rule held, FLAG.KICK0; nbfGravityKick's value + dt g).
const FORCED_C_WGSL = /* wgsl */`
fn poForcedC(cs:u32,i:u32,a:u32,raw:f32)->f32{
  if((nbCRec(cs,NB_REC_CLASS)&NB_CLASS_STILL)==0u||(phi[NB_PHI_C_FLAGS+NB_STRIDE*cs+i]&(NB_FLAG_KICK0<<a))==0u){return raw;}
  return raw+nbRingF(cfg.x,NB_RING_DT)*nbRingF(cfg.x,NB_RING_GRAVITY+a);
}
`;
const headerWGSL = (L: BrickLayout, staged = false, tiered = true, cuts = false) => `const PO_TIERED:bool=${tiered};\n` + (tiered ? FORCED_C_WGSL : "") + brickOwnerWGSL({ phi: "phi", vel: "vel", faceF: "NB_VEL_F_US0", faceC: "NB_VEL_C_US0", box: BOX_TOKEN, ...(tiered ? { forcedC: "poForcedC" } : {}), ...(L.wallView ? { view: true } : {}) }) + PRESSURE_ROWS_WGSL + (staged ? STAGE_BRICK : STAGE_DIRECT) + compositeWGSL(cuts);

/** The composite kernels. rowsF's label and entry are the engine's operator's when pressure-operator.wgsl.ts builds it.
 * `tiered: false` (the all-h engine) drops the absent-page guard: all-h, an absent page is air (brick-abi.md §2.5). */
export function pressureCompositeKernels(L: BrickLayout, names: { rowsF?: [string, string]; projectF?: [string, string] } = {}, opt: { readonly tiered?: boolean; readonly faces?: number; readonly live?: { readonly hb: number; readonly hp: number; readonly rl: number; readonly rln: number; readonly dry: number }; readonly cuts?: boolean } = {}) {
  const tiered = opt.tiered !== false;
  // The cuts (opt.cuts, default off; M5 design §5): rowsF and projectF read the K slots' FRAC / AP / W (static and
  // body), so a brick with a cut no longer latches FAIL.UNSUPPORTED there. Off, their text is the cut-free rows'.
  // They need the face cache: the walls' re-decision (pmgWallsF) takes a wall face's coupling from it.
  const cuts = opt.cuts === true;
  if (cuts && opt.faces === undefined) throw new Error("Pressure composite: the cuts need the face cache (pmgF)");
  // The face cache (opt.faces: its pmgF word, mgLayout(L).FC; lane P's own buffer, bound by the MGPCG solvers): rowsF
  // stores each liquid h row's six band faces (coefficient / fraction, and its kind + 1 in 4 bits a face of a seventh
  // word as an exact f32 integer), and projectF reads them on every face with a liquid h side instead of recomputing
  // the formula, so it stages only the owners and slots (poStageLight). The values are the formula's, bit for bit.
  const faces = opt.faces;
  const FC = faces === undefined ? "" : `const PO_FC:u32=${faces}u;\nfn poFcAt(s:u32,k:u32,l:u32)->u32{return PO_FC+NB_STRIDE*(7u*s+k)+l;}\n`;
  const rowsB = faces === undefined ? ROWS : [...ROWS, FACES], projectB = faces === undefined ? PROJECT : [...PROJECT, FACES];
  // The live lists (opt.live, with the face cache: pmgF words of mgLayout(L) HBL, HPL, RLL and RLN, written by the
  // MGPCG's pressureRowsList over F_ALL): tileRoots and rowsF run a workgroup a brick of HB (a brick with an h-owned
  // tile) and projectF one of HP (an h-owned tile in the brick or across its lower faces), each list's indirect
  // arguments exact, so the other F_ALL bricks (5 in 6 on the tiered dam) launch none of these kernels' workgroups
  // (their shared memory held a core's residency to a few workgroups: 3492 empty workgroups took 296 / 202 us of
  // rowsF / projectF). Their zero rows are pressureRowsZero's, their FLAGS bits pressureProjectRest's.
  const live = opt.live;
  if (live && faces === undefined) throw new Error("Pressure composite: the live lists need the face cache (pmgF)");
  const LIVE = live === undefined ? "" : `const PO_HBL:u32=${live.hb}u;const PO_HPL:u32=${live.hp}u;const PO_RLL:u32=${live.rl}u;const PO_RLN:u32=${live.rln}u;const PO_DRY:u32=${live.dry}u;\n`;
  const head = (list: "hb" | "hp", go: string, fill = "") => live === undefined
    ? `let g=nbGroup(wid);if(lid==0u){${fill}${go}=select(0u,1u,g<nbListCount(NB_LIST_F_ALL));}if(workgroupUniformLoad(&${go})==0u){return;}\n  let slot=nbListSlot(NB_LIST_F_ALL,g);`
    : `let g=nbGroup(wid);${fill ? `if(lid==0u){${fill}}` : ""}\n  let slot=u32(pmgF[${list === "hb" ? "PO_HBL" : "PO_HPL"}+g]);`;
  const k = (label: string, body: string, entry = label, bindings: readonly BrickBinding[] = ROWS, staged = false, cut = false) => brickCellKernel(L, label, bindings, pressureResolveBox(L, headerWGSL(L, staged, tiered, cut) + body, tiered), entry);
  const [rowsLabel, rowsEntry] = names.rowsF ?? ["pressureRowsF", "pressureRowsF"];
  const [projectLabel, projectEntry] = names.projectF ?? ["pressureProjectF", "pressureProjectF"];

  // tileRoots: a workgroup a brick of F_ALL (G1), before the operator (PressureSolver.before). The held fit of each
  // h tile of the brick (umAuthority's h root, pcHeldFit: the mean of its 64 h centres, the vertices 4T..4T+4 at
  // weights 1/2 on the tile's faces, /64) into its C PPHI, from the brick's 9^3 vertices staged once. It was summed
  // by every cold brick for each of the 96 tiles its starts read (up to 12 bricks a tile, 125 vertices each: 7 of
  // the tiered dam operator's 8.7 ms, 2026-11-15).
  const tileRoots = k("pressureTileRoots", LIVE + /* wgsl */`
var<workgroup> trGo:u32;
var<workgroup> trV:array<f32,729>;
var<workgroup> trP:array<f32,200>;
@compute @workgroup_size(256) fn pressureTileRoots(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  ${head("hb", "trGo")}let o=nbFOrigin(slot);let T0=o>>vec3u(2u);
  if(lid==0u){var m=0u;for(var j=0u;j<8u;j++){if(nbTileOwnerH(T0+poCorner(j))){m|=1u<<j;}}trGo=m;}
  let mask=workgroupUniformLoad(&trGo);if(mask==0u){return;}
  for(var i=lid;i<729u;i+=256u){trV[i]=nbVertexPhi(o+vec3i(i32(i%9u),i32((i/9u)%9u),i32(i/81u)));}
  workgroupBarrier();
  // A lane a row of 5 vertices along x of tile j (25 rows a tile), then a lane a tile.
  if(lid<200u){
    let j=lid/25u;let y=i32((lid%25u)%5u);let z=i32((lid%25u)/5u);let base=4*poCorner(j);
    var row=0.0;
    for(var x=0;x<=4;x++){let d=base+vec3i(x,y,z);row+=select(1.0,0.5,x==0||x==4)*trV[u32(d.x+9*(d.y+9*d.z))];}
    trP[lid]=select(1.0,0.5,y==0||y==4)*select(1.0,0.5,z==0||z==4)*row;
  }
  workgroupBarrier();
  if(lid<8u&&((mask>>lid)&1u)!=0u){
    var total=0.0;for(var r=0u;r<25u;r++){total+=trP[25u*lid+r];}
    let T=T0+poCorner(lid);
    pres[NB_PRES_C_PPHI+NB_STRIDE*nbCSlotOfPage(T>>vec3u(3u))+nbTileIndex(T)]=total/64.0;
  }
}`, "pressureTileRoots", live === undefined ? ROWS : [...ROWS, FACES]);

  // rowsF: a workgroup a brick of F_ALL (G1), a lane a cell (two cells a lane of 256 before: each lane's chain of
  // face formulas twice over, at the same staged memory a workgroup).
  const rowsF = k(rowsLabel, FC + LIVE + /* wgsl */`
var<workgroup> poGo:u32;
var<workgroup> poCold:atomic<u32>;
// The h rows' starts over the brick's centre window (o-1..o+8, poSC's index): the brick's own 512 and the 384 face
// halo cells, 0 off the liquid h rows. A seam head reads its tangential neighbours' starts from here (poHeadS).
var<workgroup> poSH:array<f32,1000>;
// An h row's seam head on face f (bSeamHead, npSeamHead's sums in its order): the tangential neighbours' starts from
// poSH, without npSeamHead's array of bases. The usable test is the staged owners and centres, and each start is
// poStart's value (a warm row's X is its start, so the rows' X writes leave it), so the head is the direct poStart
// walk's bit for bit (the composite rows' dump, 701 heads).
fn poHeadS(c:vec3i,own:f32,base:f32,f:u32)->f32{
  if(!poWet(own,1.0)){return 0.0;}
  let lb=vec3u(c&vec3i(7));var head=0.0;
  for(var axis=0u;axis<3u;axis++){
    if(axis==f/2u){continue;}
    var rise=0.0;var count=0.0;
    for(var side=0u;side<2u;side++){
      let g=2u*axis+side;let m=c+npSign(g)*poUnit(axis);
      if(nbInWallBox(m)&&poOwnerH(m)){if(${cuts ? "poRowWet(m,poCentre(m))" : "poWet(poCentre(m),1.0)"}){rise+=f32(npSign(g))*(poSH[poIndex(m-poSO+vec3i(1),10)]-base);count+=1.0;}}
    }
    if(count>0.0){head+=rise/count*(f32(lb[axis]&3u)-1.5);}
  }
  return head;
}
@compute @workgroup_size(512) fn ${rowsEntry}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  ${live === undefined ? `  let g=nbGroup(wid);if(lid==0u){poGo=select(0u,1u,g<nbListCount(NB_LIST_F_ALL));}if(workgroupUniformLoad(&poGo)==0u){return;}
  let slot=nbListSlot(NB_LIST_F_ALL,g);
  if(lid==0u){atomicStore(&poCold,0u);}
  // A brick of 4h-owned tiles only (the tiered dam's native interior: 5 in 6 of F_ALL) holds no h row: its zero rows
  // are written without staging the brick's phi (the same rows the full path writes for cells it does not own).
  workgroupBarrier();
  if(lid<8u){if(nbTileOwnerH((nbFOrigin(slot)>>vec3u(2u))+poCorner(lid))){atomicOr(&poCold,2u);}}
  workgroupBarrier();
  if(lid==0u){poGo=atomicLoad(&poCold);atomicStore(&poCold,0u);}
  if(workgroupUniformLoad(&poGo)==0u){
    ${cuts ? `// A cut brick of 4h tiles only latches when it is a body's (§4 keeps a body's tiles h). A static cut's 4h tile
    // is dry (lane E's static rule leaves dry cut bricks at 4h) or latches in rowsC (a liquid native row there).
    if(lid==0u&&poCutBmask(nbFRec(slot,NB_REC_CUT))!=0u){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}` : "if(lid==0u&&nbFRec(slot,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}"}
    for(var j=0u;j<1u;j++){
      let at=NB_STRIDE*slot+lid+512u*j;
      pres[NB_PRES_F_B+at]=0.0;pres[NB_PRES_F_X+at]=0.0;pres[NB_PRES_F_RHS+at]=0.0;pres[NB_PRES_F_DIAG+at]=0.0;pres[NB_PRES_F_PPHI+at]=0.0;
      pres[NB_PRES_F_K0+at]=0.0;pres[NB_PRES_F_K1+at]=0.0;pres[NB_PRES_F_K2+at]=0.0;pres[NB_PRES_F_KU0+at]=0.0;pres[NB_PRES_F_KU1+at]=0.0;pres[NB_PRES_F_KU2+at]=0.0;
    }
    return;
  }
` : `${head("hb", "poGo", "atomicStore(&poCold,0u);")}
  workgroupBarrier();
`}  poStage(nbFOrigin(slot),lid,512u);
  for(var j=0u;j<1u;j++){
    let c=nbFOrigin(slot)+nbLocalCoord(lid+512u*j);
    if(nbInWallBox(c)&&poOwnerH(c)){if(${cuts ? "poRowWet(c,poCentre(c))" : "poWet(poCentre(c),1.0)"}&&(poFlagsF(c)&PO_WARM)==0u){atomicOr(&poCold,1u);}}
  }
  // A cold face halo row's start reads the window too.
  if(lid<384u){
    let m=nbFOrigin(slot)-vec3i(1)+poHaloD(lid);
    if(nbInWallBox(m)&&poOwnerH(m)){if(${cuts ? "poRowWet(m,poCentre(m))" : "poWet(poCentre(m),1.0)"}&&(poFlagsF(m)&PO_WARM)==0u){atomicOr(&poCold,1u);}}
  }
  workgroupBarrier();
  if(lid==0u){poGo=atomicLoad(&poCold);}
  poTiles(lid,workgroupUniformLoad(&poGo)!=0u);
  ${cuts ? `// A cut brick's liquid rows are all h (the native rows never read a cut). A body's brick has every tile h (§4's h
  // ring): a lattice tile of it that is not h-owned latches. A static cut's 4h tile may stay (lane E's static rule,
  // the old solidPromote, leaves dry cut bricks at 4h); a liquid native row in one latches in rowsC.
  let kslot=nbFRec(slot,NB_REC_CUT);let bmask=poCutBmask(kslot);
  if(bmask!=0u&&lid<8u){let T=(nbFOrigin(slot)>>vec3u(2u))+poCorner(lid);if(nbTileInLattice(T)&&!nbTileOwnerH(T)){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}}
  let staticCut=(nbFRec(slot,NB_REC_CLASS)&NB_CLASS_CUT_STATIC)!=0u;` : "if(lid==0u&&nbFRec(slot,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}"}
  let o=nbFOrigin(slot);let r=cfg.x;let h=nbRingF(r,NB_RING_H);
  let rate=bitcast<f32>(atomicLoad(&nbSet[NB_H_LANE_F]));let capacity=1.0;
  // The starts first (a lane its cell's, lanes 0..383 a face halo cell's), so that the seam heads read them staged.
  let l=lid;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
  let owned=nbInWallBox(c)&&poOwnerH(c);
  var own=NB_PHI_AIR;if(owned){own=poCentre(c);}
  let liquid=owned&&${cuts ? "poRowWet(c,own)" : "poWet(own,1.0)"};
  var start=0.0;if(liquid){start=poStart(c,own);}
  poSH[poIndex(nbLocalCoord(l)+vec3i(1),10)]=start;
  if(lid<384u){
    let d=poHaloD(lid);let m=o-vec3i(1)+d;var sm=0.0;
    if(nbInWallBox(m)&&poOwnerH(m)){let pm=poCentre(m);if(${cuts ? "poRowWet(m,pm)" : "poWet(pm,1.0)"}){sm=poStart(m,pm);}}
    poSH[poIndex(d,10)]=sm;
  }
  workgroupBarrier();
  {
    var diag=0.0;var div=0.0;var extra=0.0;var kk=vec3f(0.0);var ku=vec3f(0.0);
    ${faces === undefined ? "" : "var kinds=0u;"}
    ${cuts ? `// A static solid cell under liquid φ beside a liquid native row: rowsC's seam would read it as air (no row, PPHI
    // air), so it latches (lane E's static rule keeps a wet cut brick's neighbours h).${L.staticTiles ? " A closed static tile's cell is a\n    // wall to rowsC (nbInWallC), as is a neighbour in one." : ""}
    if(owned&&!liquid&&poWet(own,1.0)&&poStaticShut(c)${L.staticTiles ? "&&nbInWallC(c)" : ""}){
      for(var f=0u;f<6u;f++){
        let q=c+npSign(f)*poUnit(f/2u);
        if(nbInWallC(q)&&!poOwnerH(q)&&nbPageWord(q>>vec3u(5u))<NB_SEAM&&poWet(poRootPhi(q>>vec3u(2u)),4.0)){nbFail(NB_FAIL_UNSUPPORTED,0xa5000000u|nbFRec(slot,NB_REC_KEY));}
      }
    }
    ` : ""}if(liquid){
      for(var j=0u;j<8u;j++){poGuardVertex(c+poCorner(j));}${cuts ? `
      // The row's capacity (bHeadCapacity, the old head capacity: its own FRAC on all six faces). A static solid cell
      // is no row (poRowWet, option (b)); a shut cell of a brick with a body is the body's (a body brick's row, as in a
      // brick with no static cut). With no body a shut row is a static one poRowWet missed: it latches.
      let capacity=poFrac(c);
      if(staticCut&&bmask==0u&&capacity<=1e-5){nbFail(NB_FAIL_UNSUPPORTED,0xa2000000u|nbFRec(slot,NB_REC_KEY));}` : ""}
      for(var f=0u;f<6u;f++){
        let a=f/2u;let sign=npSign(f);let pf=poBandFace(c,own,true,f);let face=pf.face;${faces === undefined ? "" : cuts ? `
        // A wall face's cache word is its coupling (V, the walls' re-decision reads it; its ratio is 1), a closed
        // face's 0.
        kinds|=(face.kind+1u)<<(4u*f);if(face.kind!=NP_K_SEAM){pmgF[poFcAt(slot,f,l)]=select(select(face.coefficient/face.fraction,face.coefficient,face.kind==NP_K_WALL),0.0,face.kind==NP_K_CLOSED);}` : `
        kinds|=(face.kind+1u)<<(4u*f);if(face.kind!=NP_K_SEAM){pmgF[poFcAt(slot,f,l)]=face.coefficient/face.fraction;}`}
        var fc=c;if(sign>0){fc[a]+=1;}
        let us=poFacePlane(fc,a);
        if(face.kind!=NP_K_CLOSED){div+=f32(sign)*face.fraction*us;}${cuts ? `
        // The moving wall (pRows 1618): sign (cap - A) W on every face but a wall or air face; W is 0 off bodies.
        if(face.kind!=NP_K_WALL&&face.kind!=NP_K_OPEN){div+=f32(sign)*(capacity-face.fraction)*poFaceW(fc,a);}
        // The native rows (rowsC) take a seam at volume 1, and the faces into an absent brick are open: a cut there
        // would be one-sided, so it latches.
        if(pf.native&&face.fraction!=1.0){nbFail(NB_FAIL_UNSUPPORTED,0xa3000000u|nbFRec(slot,NB_REC_KEY));}
        if(bmask!=0u&&sign>0&&nbInLattice(fc)&&poSlotOfCell(fc)>=NB_SEAM){nbFail(NB_FAIL_UNSUPPORTED,0xa4000000u|nbFRec(slot,NB_REC_KEY));}` : ""}
        if(face.kind==NP_K_NEUMANN||face.kind==NP_K_CLOSED){continue;}
        if(face.kind==NP_K_WALL){
          // The build's state (pRebase's first measure, start + lift >= 0), stored at the face for the outer loop.
          let lift=npLift(f,1.0,us,h,1.0);let separated=start+lift<0.0;
          if(!separated){extra+=face.coefficient*lift;}else{diag+=face.coefficient;}
          poSetWall(poFlagsWordF(fc),a,separated);
          continue;
        }
        diag+=face.coefficient;
        if(face.kind==NP_K_BAND&&pf.qWet&&sign<0){kk[a]=face.coefficient;}
        if(face.kind==NP_K_SEAM){
          if(sign<0){kk[a]=face.coefficient;}else{ku[a]=face.coefficient;}
          let head=poHeadS(c,own,start,f);extra+=face.coefficient*head;${faces === undefined ? "" : `
          pmgF[poFcAt(slot,f,l)]=head;`}
        }
      }
    }
    // Every cell's X is its start, 0 off the liquid rows (the slot contract's zero rows, cells of 4h-owned tiles and
    // the skirt included: they hold no state, and a refined tile's rows start cold).
    pres[NB_PRES_F_B+at]=start;pres[NB_PRES_F_X+at]=start;
    pres[NB_PRES_F_RHS+at]=select(0.0,-h*div+h*h*rate*${cuts ? "poFrac(c)" : "capacity"}+extra,liquid);
    pres[NB_PRES_F_DIAG+at]=diag;pres[NB_PRES_F_PPHI+at]=select(0.0,${cuts ? "select(own,NB_PHI_AIR,poStaticShut(c))" : "own"},owned);
    pres[NB_PRES_F_K0+at]=kk.x;pres[NB_PRES_F_K1+at]=kk.y;pres[NB_PRES_F_K2+at]=kk.z;
    pres[NB_PRES_F_KU0+at]=ku.x;pres[NB_PRES_F_KU1+at]=ku.y;pres[NB_PRES_F_KU2+at]=ku.z;${faces === undefined ? "" : `
    pmgF[poFcAt(slot,6u,l)]=f32(kinds);`}
  }
}`, rowsEntry, rowsB, true, cuts);

  // rowsC: a workgroup a page of C_ALL (G1), a lane a tile. After rowsF, whose stored rows it reads: a seam
  // subface's h cell m takes its centre φ from PPHI and its head from poHeadB (the h rows' starts in B and centre φ
  // in PPHI), the values rowsF computed, so the subface's coefficient and head are the h row's own, unrecomputed.
  // The seams are the pass's work (16 subfaces a face, each a band face and a head), so they are spread over the
  // workgroup: each lane lists its tile's seam faces, then the lanes take the listed faces' subfaces 512 at a time
  // (32 faces), and each tile's lane adds its faces' 16 sums in subface order. One lane walked its two tiles' seams
  // alone before (2.4 ms on the tiered dam, the slowest lane's chain).
  const rowsC = k("pressureRowsC", PROJECT_WGSL + /* wgsl */`
var<workgroup> rcN:atomic<u32>;var<workgroup> rcCount:u32;
var<workgroup> rcE:array<u32,3072>;
var<workgroup> rcR:array<f32,512>;
var<workgroup> rcD:array<f32,512>;var<workgroup> rcX:array<f32,512>;var<workgroup> rcP:array<f32,512>;
@compute @workgroup_size(512) fn pressureRowsC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){rcCount=nbListCount(NB_LIST_C_ALL);atomicStore(&rcN,0u);}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&rcCount)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  let O=nbCOrigin(cs)>>vec3u(2u);let r=cfg.x;let h=nbRingF(r,NB_RING_H);
  let rate=bitcast<f32>(atomicLoad(&nbSet[NB_H_LANE_F]));let capacity=1.0;
  let i=lid;let T=O+nbLocalCoord(i);let at=NB_STRIDE*cs+i;
  var diag=0.0;var div=0.0;var extra=0.0;var kk=vec3f(0.0);var start=0.0;
  let owned=nbTileInWallBox(T)&&!nbTileH(cs,i);
  var root=0.0;if(owned){root=poRootPhi4(T);}
  rcR[i]=root;
  let liquid=owned&&poWet(root,4.0);
  ${cuts ? `// A liquid native row in a cut brick (a static cut's tile left at 4h) would not read the cut: the old band
  // certificate's failure, a latch.
${L.staticTiles ? "  // With the static tiles an exact tile of a cut brick is a 4h row (its closed neighbours are walls): only an inexact\n  // one, or one in a body's brick, latches.\n" : ""}  if(liquid){let fs=nbFSlotOfBrick(T>>vec3u(1u));if(fs<NB_SEAM){if(nbFRec(fs,NB_REC_CUT)!=NB_NONE${L.staticTiles ? "&&(nbTileInexact(T)||(nbFRec(fs,NB_REC_CLASS)&NB_CLASS_NEAR_BODY)!=0u)" : ""}){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(fs,NB_REC_KEY));}}}` : ""}
  var seams=0u;
  if(liquid){
    for(var j=0u;j<8u;j++){poGuardVertex(4*(T+poCorner(j)));}
    start=poTileStart(T);
    for(var f=0u;f<6u;f++){
      let a=f/2u;let side=f&1u;let sign=npSign(f);let e=poUnit(a);let Tn=T+sign*e;
      let plane=4*select(T,Tn,side==1u);
      if(!nbTileInWallBox(Tn)){${L.detail ? `
        // A detail engine's coupled box face (pressureDetailFaces' native twin): the parent's velocity, prescribed. Its
        // flux (the whole 4h face's, the forces' relaxed value) enters the divergence; no coupling, never a wall.
        if(((${L.detail.coupled}u>>f)&1u)!=0u){div+=f32(sign)*nbFacePlane(plane,a);poSetWall(poFlagsWordC(select(T,Tn,side==1u)),a,false);continue;}` : ""}
        // The root's wall slot: fraction 0.5 (the wall face's mean h V), b its forced flux.
        let fraction=0.5;let u=nbFacePlane(plane,a);let c=4.0*fraction;let lift=f32(sign)*4.0*h*u;
        div+=f32(sign)*fraction*u;
        let separated=!(root< -4.0)&&start+lift<0.0;
        if(!separated){extra+=c*lift;}else{diag+=c;}
        poSetWall(poFlagsWordC(select(T,Tn,side==1u)),a,separated);
        continue;
      }
      let w=nbPageWord(Tn>>vec3u(3u));
      if(w>=NB_SEAM&&w!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,w);continue;}
      let low=select(Tn,T,side==1u);let high=select(T,Tn,side==1u);
      if(w==NB_NONE){
        let c=4.0/poNativeTheta(low,high,a);diag+=c;div+=f32(sign)*nbFacePlane(plane,a);continue;
      }
      // A seam (the 16 h cells of Tn beside the face): listed, summed below.
      if(nbTileOwnerH(Tn)){seams|=1u<<f;continue;}
      let un=nbFacePlane(plane,a);div+=f32(sign)*un;
      if(poWet(poRootPhi4(Tn),4.0)){let c=4.0;diag+=c;if(side==0u){kk[a]=c;}}
      else{diag+=4.0/poNativeTheta(low,high,a);}
    }
  }
  // The tile's seam faces, listed contiguously in face order (one atomic a tile), so that its lane sums exactly its
  // own entries of each chunk below instead of scanning the chunk's 32.
  let sCount=countOneBits(seams);var sBase=0u;
  if(sCount>0u){sBase=atomicAdd(&rcN,sCount);var e=sBase;for(var f=0u;f<6u;f++){if(((seams>>f)&1u)!=0u){rcE[e]=(i<<3u)|f;e++;}}}
  workgroupBarrier();
  if(lid==0u){rcCount=atomicLoad(&rcN);}
  let n=workgroupUniformLoad(&rcCount);
  // Each listed face's subface s, with the band's own face formula from the h side (poSeamFace: the cell across is
  // the native row's tile, whose root φ is rcR's).
  for(var c0=0u;c0<n;c0+=32u){
    let ent=c0+(lid>>4u);var dc=0.0;var xc=0.0;var pc=0.0;
    if(ent<n){
      let wv=rcE[ent];let ti=wv>>3u;let f=wv&7u;let a=f/2u;let side=f&1u;
      let Ts=O+nbLocalCoord(ti);let Tn=Ts+npSign(f)*poUnit(a);
      let u=poUnit((a+1u)%3u);let v=poUnit((a+2u)%3u);let g2=f^1u;let sq=lid&15u;
      var layer=4*Tn;if(side==0u){layer[a]+=3;}
      let m=layer+i32(sq&3u)*u+i32(sq>>2u)*v;
      let pm=poPhiF(m);let lm=poWet(pm,1.0);let pf=poSeamFace(m,pm,lm,g2,rcR[ti]);
      if(pf.kind==NP_K_SEAM){dc=pf.coefficient;if(lm){xc=pf.coefficient*poHeadB(m,pm,poBF(m),g2);}}
      var fm=m;if(side==0u){fm[a]+=1;}
      pc=nbFacePlane(fm,a);
    }
    rcD[lid]=dc;rcX[lid]=xc;rcP[lid]=pc;
    workgroupBarrier();
    // No break or continue: the barrier below stays in uniform control flow.
    for(var e=max(sBase,c0);e<min(sBase+sCount,c0+32u);e++){
      let q=e-c0;
      {
        let wv=rcE[e];var sd=0.0;var sx=0.0;var sp=0.0;
        for(var sq=0u;sq<16u;sq++){sd+=rcD[16u*q+sq];sx+=rcX[16u*q+sq];sp+=rcP[16u*q+sq];}
        diag+=sd;extra-=sx;div+=f32(npSign(wv&7u))*(sp/16.0);
      }
    }
    workgroupBarrier();
  }
  // X keeps the tile's last pressure (the start) on liquid native rows, and on h tiles for the finish.
  if(owned&&!liquid){pres[NB_PRES_C_X+at]=0.0;}
  pres[NB_PRES_C_B+at]=start;
  pres[NB_PRES_C_RHS+at]=select(0.0,-16.0*h*div+64.0*h*h*rate*capacity+extra,liquid);
  pres[NB_PRES_C_DIAG+at]=diag;
  // An h tile's PPHI is tileRoots' held fit, which this pass's seams read across tiles: kept.
  if(!(nbTileInLattice(T)&&nbTileH(cs,i))){pres[NB_PRES_C_PPHI+at]=root;}
  pres[NB_PRES_C_K0+at]=kk.x;pres[NB_PRES_C_K1+at]=kk.y;pres[NB_PRES_C_K2+at]=kk.z;
}`);

  // projectF: a workgroup a brick (G1 of the engine's project list, F_ALL), a lane two cells' lower faces (a cell a
  // lane of 512 measured 1584 -> 1911 us a call on the tiered dam).
  // Writes U on every h face (a face with an h-owned cell on either side, stored at its upper cell), raises
  // BOUND_F, and writes the WARM bit (FLAGS bit 7) of every cell: an h-owned cell that held a liquid row.
  // A dry h side m whose cell across q is not h-owned: poProjectFace(m, own, false, ...) is US unless the face is a
  // seam (q resident, its tile's root wet: npBandFace's native), so only a seam runs the formula; poAcross runs first
  // either way (its FAIL latch).
  const DRY = faces === undefined ? "" : /* wgsl */`
fn poDrySeam(q:vec3i)->bool{return poAcross(q)==NP_ACROSS_RESIDENT&&poWet(poRootPhi(q>>vec3u(2u)),4.0);}
fn poProjectDry(m:vec3i,own:f32,f:u32,us:f32,h:f32,separated:bool,q:vec3i)->f32{
  if(!poDrySeam(q)){return us;}
  return poProjectFace(m,own,false,f,us,h,separated);
}
`;
  const projectF = k(projectLabel, FC + LIVE + PROJECT_WGSL + DRY + BRICK_SOURCE_RING_WGSL + /* wgsl */`
var<workgroup> wSpeed:atomic<u32>;
// The brick's signed per-axis bound (BOUND_F's six words: a the largest -u_a, 3 + a the largest u_a).
var<workgroup> wBound:array<atomic<u32>,6>;
var<workgroup> poGo:u32;${live === undefined ? "" : `
// A dry seam face here, left to projectDry (PO_DRY: this brick's flag).
var<workgroup> wDry:atomic<u32>;`}
@compute @workgroup_size(256) fn ${projectEntry}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  ${live === undefined ? `  let g=nbGroup(wid);
  if(lid==0u){atomicStore(&wSpeed,0u);for(var k=0u;k<6u;k++){atomicStore(&wBound[k],0u);}poGo=select(0u,1u,g<nbListCount(NB_LIST_F_ALL));}
  if(workgroupUniformLoad(&poGo)==0u){return;}
  let slot=nbListSlot(NB_LIST_F_ALL,g);
  let o=nbFOrigin(slot);
  // No h-owned tile in the brick or across its lower faces (the 3^3 tiles from its origin tile - 1): no h face is
  // stored here, so only the cells' FLAGS change (WARM and the wall bits cleared, as the full path for cells it does
  // not own), without staging the brick's phi.
  if(lid<27u){if(nbTileOwnerH((o>>vec3u(2u))-vec3i(1)+poCoord(lid,3u))){atomicOr(&wSpeed,1u);}}
  workgroupBarrier();
  if(lid==0u){poGo=atomicLoad(&wSpeed);atomicStore(&wSpeed,0u);for(var k=0u;k<6u;k++){atomicStore(&wBound[k],0u);}}
  if(workgroupUniformLoad(&poGo)==0u){
    for(var j=0u;j<2u;j++){let at=NB_STRIDE*slot+lid+256u*j;phi[NB_PHI_F_FLAGS+at]&=~(PO_WARM|PO_WALLS);}
    return;
  }
` : `${head("hp", "poGo", "atomicStore(&wSpeed,0u);atomicStore(&wDry,0u);for(var k=0u;k<6u;k++){atomicStore(&wBound[k],0u);}")}
  let o=nbFOrigin(slot);
  workgroupBarrier();
`}  ${faces === undefined ? "poStage(o,lid,256u);" : "poStageLight(o,lid);"}
  var bound=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  {
    let h=nbRingF(cfg.x,NB_RING_H);
    for(var j=0u;j<2u;j++){
      let l=lid+256u*j;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
      let hc=nbInWallBox(c)&&poOwnerH(c);
      var pc=NB_PHI_AIR;if(hc){pc=pres[NB_PRES_F_PPHI+at];}
      let lc=hc&&poWet(pc,1.0);
      // This cell's FLAGS hold the solver's wall state for its lower faces on the lattice's planes (read here, by
      // this lane only), and leave with the faces' release bits.
      let flags=phi[NB_PHI_F_FLAGS+at];var release=0u;
      for(var a=0u;a<3u;a++){
        if(!nbFaceInLattice(c,a)){continue;}
        let n=c-poUnit(a);let hn=nbInWallBox(n)&&poOwnerH(n);
        if(!hc&&!hn){continue;}
        let fa=NB_STRIDE*(slot+a*NB_FCAP)+l;let us=vel[NB_VEL_F_US0+fa];
        var pn=NB_PHI_AIR;if(hn){pn=poPhiF(n);}
        let ln=hn&&poWet(pn,1.0);
        let wall=c[a]==0||c[a]==NB_WALL_BOX[a];${cuts && faces !== undefined ? "var" : "let"} separated=wall&&(flags&(PO_WALL0<<a))!=0u;
        var u=us;${faces === undefined ? `
        if(lc){u=poProjectFace(c,pc,true,2u*a,us,h,separated);}
        else if(ln){u=poProjectFace(n,pn,true,2u*a+1u,us,h,separated);}` : `${cuts ? `
        // The cuts: the face's aperture and moving-wall velocity (this brick's lower face), and its closure: a liquid
        // side's kind, else A <= 1e-6. A closed face moves with the solid (bFace 3217: U = W).
        let ap=poFaceAP(c,a);let wv=poFaceW(c,a);var closed=ap<=1e-6;` : ""}
        if(lc){
          let kind=((u32(pmgF[poFcAt(slot,6u,l)])>>(8u*a))&15u)-1u;${cuts ? "var" : "let"} word=pmgF[poFcAt(slot,2u*a,l)];${cuts ? `closed=kind==NP_K_CLOSED;
          // A wall face's state is the solve's (FLAGS at the face), a static solid's face (option (b)) as the box's.
          if(kind==NP_K_WALL){word=1.0;separated=(flags&(PO_WALL0<<a))!=0u;}` : ""}
          u=poProjectFaceK(c,pc,true,2u*a,us,h,separated,kind,select(word,poSeamRatio(),kind==NP_K_SEAM),select(0.0,word,kind==NP_K_SEAM));
        }
        else if(ln){
          let sn=poSlotOfCell(n);let lnl=nbLocal(n);
          let kind=((u32(pmgF[poFcAt(sn,6u,lnl)])>>(8u*a+4u))&15u)-1u;${cuts ? "var" : "let"} word=pmgF[poFcAt(sn,2u*a+1u,lnl)];${cuts ? `closed=kind==NP_K_CLOSED;
          if(kind==NP_K_WALL){word=1.0;separated=(flags&(PO_WALL0<<a))!=0u;}` : ""}
          u=poProjectFaceK(n,pn,true,2u*a+1u,us,h,separated,kind,select(word,poSeamRatio(),kind==NP_K_SEAM),select(0.0,word,kind==NP_K_SEAM));
        }
${live === undefined ? `
        else if(hc&&!hn){u=poProjectDry(c,pc,2u*a,us,h,separated,n);}
        else if(hn&&!hc){u=poProjectDry(n,pn,2u*a+1u,us,h,separated,c);}` : `
        else if(hc!=hn&&poDrySeam(select(c,n,hc))){atomicOr(&wDry,1u);continue;}`}${cuts ? `
        if(closed){u=wv;}` : ""}`}
${faces === undefined ? `
        else if(hc&&!hn){u=poProjectFace(c,pc,false,2u*a,us,h,separated);}
        else if(hn&&!hc){u=poProjectFace(n,pn,false,2u*a+1u,us,h,separated);}` : ""}
        // The nozzle (umProjectWithSource): applyInflowVelocity of the face's texel, the cell below it; a box-low face
        // is not the nozzle's.
        if(nbInflowOn()&&c[a]!=0){u=nbInflowProjected(n,a,u);}
        vel[NB_VEL_F_U0+fa]=u;bound[a]=max(bound[a],-u);bound[3u+a]=max(bound[3u+a],u);
        if(wall){
          // The owner: c on plane 0 (its low face), n on plane N (its high face).
          let low=c[a]==0;let sign=select(1.0,-1.0,low);let liquid=select(ln,lc,low);let m=select(n,c,low);
          var ghost=0.0;if(liquid&&!separated){ghost=poXF(m)+sign*us*h;}
          if(poRelease(select(pn,pc,low),ghost,sign,u,1.0,h)${cuts ? "&&poFrac(m)>1e-5&&ap>1e-6" : ""}){release|=PO_WALL0<<a;}
        }${cuts ? `
        // A solid face (bReleased 3359-3367): between an open and a shut cell, from the band row below it, released
        // when the shut side's pressure is <= 0 and the fluid leaves the solid, -+(U - W) dt > 1e-4 h.
        else if(hn){
          let openN=poFrac(n)>1e-5;
          if(openN!=(poFrac(c)>1e-5)&&ap>1e-6){
            var pressure=0.0;if(select(ln,lc,openN)){pressure=poXF(select(n,c,openN));}
            if(pressure<=0.0&&select(1.0,-1.0,openN)*(u-wv)*nbRingF(cfg.x,NB_RING_DT)>1e-4*h){release|=PO_WALL0<<a;}
          }
        }` : ""}
      }
      phi[NB_PHI_F_FLAGS+at]=(flags&~(PO_WARM|PO_WALLS))|select(0u,PO_WARM,lc&&pres[NB_PRES_F_DIAG+at]>0.0)|release;
    }
    // A side no face moved toward stays 0, and a lane below the workgroup's word so far takes no atomic.
    for(var k=0u;k<6u;k++){let w=bitcast<u32>(bound[k]);if(w>atomicLoad(&wBound[k])){atomicMax(&wBound[k],w);}}
  }
  workgroupBarrier();
  if(lid<6u){atomicMax(&nbSet[NB_S_BOUND_F+NB_BOUND_WORDS*slot+lid],atomicLoad(&wBound[lid]));}${live === undefined ? "" : `
  if(lid==0u){pmgF[PO_DRY+slot]=f32(atomicLoad(&wDry));}`}
}`, projectEntry, projectB, true, cuts);

  // projectDry (opt.live: right after projectF, over HP): the dry seam faces projectF left, a dry h side facing a
  // resident wet 4h tile (poProjectDry's formula case). Rare, and poProjectFace's npBandFace held projectF to Metal's
  // 448-thread register cap (one 256-lane workgroup a core; 768 without it): projectF 3.00 -> 1.55 ms on the pour
  // (go/no-go, 2026-11-21). Their U, bound and release bits as projectF's: its FLAGS write keeps a cell's other bits,
  // and a dry face's projection and release do not read the wall state it clears (separated: a liquid side's only).
  // 64 lanes, so a brick without one (the flag) leaves at once.
  const projectDry = live === undefined ? undefined : k(`${projectLabel}Dry`, FC + LIVE + PROJECT_WGSL + DRY + BRICK_SOURCE_RING_WGSL + /* wgsl */`
var<workgroup> poGo:u32;
@compute @workgroup_size(64) fn ${projectEntry}Dry(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);let slot=u32(pmgF[PO_HPL+g]);
  if(lid==0u){poGo=u32(pmgF[PO_DRY+slot]);}
  if(workgroupUniformLoad(&poGo)==0u){return;}
  let o=nbFOrigin(slot);
  poStageLight(o,lid);
  let h=nbRingF(cfg.x,NB_RING_H);
  var bound=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  for(var j=0u;j<8u;j++){
    let l=lid+64u*j;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
    let hc=nbInWallBox(c)&&poOwnerH(c);
    var pc=NB_PHI_AIR;if(hc){pc=pres[NB_PRES_F_PPHI+at];}
    if(hc&&poWet(pc,1.0)){continue;}
    var release=0u;
    for(var a=0u;a<3u;a++){
      if(!nbFaceInLattice(c,a)){continue;}
      let n=c-poUnit(a);let hn=nbInWallBox(n)&&poOwnerH(n);
      if(hc==hn){continue;}
      var pn=NB_PHI_AIR;if(hn){pn=poPhiF(n);}
      if(hn&&poWet(pn,1.0)){continue;}
      if(!poDrySeam(select(c,n,hc))){continue;}
      let fa=NB_STRIDE*(slot+a*NB_FCAP)+l;let us=vel[NB_VEL_F_US0+fa];
      let wall=c[a]==0||c[a]==NB_WALL_BOX[a];
      var u=poProjectFace(select(n,c,hc),select(pn,pc,hc),false,2u*a+select(1u,0u,hc),us,h,false);${cuts ? `
      let ap=poFaceAP(c,a);let wv=poFaceW(c,a);if(ap<=1e-6){u=wv;}` : ""}
      if(nbInflowOn()&&c[a]!=0){u=nbInflowProjected(n,a,u);}
      vel[NB_VEL_F_U0+fa]=u;bound[a]=max(bound[a],-u);bound[3u+a]=max(bound[3u+a],u);
      if(wall){
        let low=c[a]==0;let sign=select(1.0,-1.0,low);let m=select(n,c,low);
        if(poRelease(select(pn,pc,low),0.0,sign,u,1.0,h)${cuts ? "&&poFrac(m)>1e-5&&ap>1e-6" : ""}){release|=PO_WALL0<<a;}
      }${cuts ? `
      else if(hn){
        let openN=poFrac(n)>1e-5;
        if(openN!=(poFrac(c)>1e-5)&&ap>1e-6&&select(1.0,-1.0,openN)*(u-wv)*nbRingF(cfg.x,NB_RING_DT)>1e-4*h){release|=PO_WALL0<<a;}
      }` : ""}
    }
    if(release!=0u){phi[NB_PHI_F_FLAGS+at]|=release;}
  }
  for(var k=0u;k<6u;k++){let w=bitcast<u32>(bound[k]);if(w!=0u){atomicMax(&nbSet[NB_S_BOUND_F+NB_BOUND_WORDS*slot+k],w);}}
}`, `${projectEntry}Dry`, projectB, true, cuts);

  // projectC: a workgroup a page of C_ALL (G1), a lane two tiles, after the solve's finish.
  // - An h tile's C X becomes the mean X of its liquid h rows (083's band-tile mean): the tile's start next step.
  // - Every 4h face (both sides 4h-owned, or a 4h tile on a lattice wall) stored at this page's tiles gets U:
  //   a native–native face the 4h gradient, a native–air face its ghost at the marched θ, a native wall
  //   the separating wall in the state the rows took (attached when start + lift >= 0, or N_HELD), and
  //   a face with no liquid side U = US. In the slot units a native face's coupling c gives
  //   U = US - sign c/(16 h V) (ghost - X), so 4V/(16hV) = 1/(4h).
  const projectC = k("pressureProjectC", PROJECT_WGSL + BRICK_SOURCE_RING_WGSL + /* wgsl */`
fn poNativeX(T:vec3i)->vec2f{
  if(!nbTileInWallBox(T)){return vec2f(0.0,-1.0);}
  let w=nbPageWord(T>>vec3u(3u));if(w==NB_NONE||w>=NB_SEAM){return vec2f(0.0,-2.0);}
  let i=nbTileIndex(T);if(nbTileH(w,i)){return vec2f(0.0,-3.0);}
  let at=NB_STRIDE*w+i;
  return vec2f(pres[NB_PRES_C_X+at],select(0.0,1.0,pres[NB_PRES_C_DIAG+at]>0.0&&poWet(pres[NB_PRES_C_PPHI+at],4.0)));
}
// Still-tank A: a STILL page's projection writes no velocity and no wall release (its faces and FLAGS hold, the old
// COLD freeze's): it stores each region's largest discarded correction |projected - F| (umColdCorrect's), in m/s,
// for the next region heat's frozen term (census RG FROZEN).
var<workgroup> pcFrozen:array<atomic<u32>,8>;
var<workgroup> pcStill:u32;
@compute @workgroup_size(256) fn pressureProjectC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){pcStill=0u;}
  if(lid<8u){atomicStore(&pcFrozen[lid],0u);}
  let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_C_ALL);
  var cs=0u;if(live){cs=nbListSlot(NB_LIST_C_ALL,g);if(lid==0u){pcStill=select(0u,1u,(nbCRec(cs,NB_REC_CLASS)&NB_CLASS_STILL)!=0u);}}
  let still=workgroupUniformLoad(&pcStill)!=0u;
  if(live){pcProjectPage(cs,lid,still);}
  if(!still){return;}
  workgroupBarrier();
  if(lid<8u){census[${CENSUS.STRIDE}u*cs+${CENSUS.REG}u+${RG.WORDS}u*lid+${RG.FROZEN}u]=atomicLoad(&pcFrozen[lid]);}
}
// umProjectWithSource at a 4h face (T, a): the old +face of the owner below, its texels 4T - e_a + i e_u + j e_v.
fn pcInflowProjected(T:vec3i,a:u32,value:f32)->f32{
  let u=select(select(vec3i(1,0,0),vec3i(0,1,0),a==0u),vec3i(1,0,0),a==1u);let v=select(select(vec3i(0,1,0),vec3i(0,0,1),a==0u),vec3i(0,0,1),a==1u);
  let base=4*T-poUnit(a);var sum=0.0;
  for(var j=0;j<4;j++){for(var i=0;i<4;i++){sum+=nbInflowProjected(base+i*u+j*v,a,value);}}
  return sum/16.0;
}
fn pcProjectPage(cs:u32,lid:u32,still:bool){
  let O=nbCOrigin(cs)>>vec3u(2u);let h=nbRingF(cfg.x,NB_RING_H);
  for(var j=0u;j<2u;j++){
    let i=lid+256u*j;let T=O+nbLocalCoord(i);let at=NB_STRIDE*cs+i;
    let tIn=nbTileInWallBox(T);
    // This tile's C FLAGS: the solver's wall state for its lower faces on the lattice's planes, then their release.
    let fw=NB_PHI_C_FLAGS+at;let flags=phi[fw];var release=0u;
    if(tIn&&nbTileH(cs,i)){
      if(still){continue;}
      phi[fw]=flags&~PO_WALLS;
      var sum=0.0;var count=0.0;
      for(var m=0u;m<64u;m++){
        let c=4*T+vec3i(i32(m&3u),i32((m>>2u)&3u),i32(m>>4u));let s=nbFSlotOfCell(c);
        if(s>=NB_SEAM){continue;}
        let q=NB_STRIDE*s+nbLocal(c);
        if(pres[NB_PRES_F_DIAG+q]>0.0){sum+=pres[NB_PRES_F_X+q];count+=1.0;}
      }
      pres[NB_PRES_C_X+at]=select(0.0,sum/count,count>0.0);
      continue;
    }
    let own=poNativeX(T);
    for(var a=0u;a<3u;a++){
      if(!nbFaceInLattice(4*T,a)){continue;}
      let e=poUnit(a);let Lt=T-e;
      if(nbTileInLattice(Lt)&&nbTileOwnerH(Lt)){continue;}
      let fa=NB_STRIDE*(cs+a*NB_CCAP)+i;let held=vel[NB_VEL_C_US0+fa];let us=poForcedC(cs,i,a,held);
      let low=poNativeX(Lt);
      let wall=${L.detail ? `(T[a]==0||4*T[a]==NB_WALL_BOX[a])&&((${L.detail.coupled}u>>(2u*a+select(0u,1u,T[a]!=0)))&1u)==0u` : "T[a]==0||4*T[a]==NB_WALL_BOX[a]"};let separated=wall&&(flags&(PO_WALL0<<a))!=0u;
      var u=us;
      if(own.y>0.0&&low.y>0.0){u=us-(own.x-low.x)/(4.0*h);}
      else if(own.y>0.0||low.y>0.0){
        // The liquid side m, the face's sign from m (T's lower face -1, Lt's upper +1), the other side.
        let mine=own.y>0.0;let sign=select(1.0,-1.0,mine);let x=select(low.x,own.x,mine);let other=select(own.y,low.y,mine);
        let m=select(Lt,T,mine);
        if(other==-1.0${L.detail ? "&&wall" : ""}){
          let mw=nbPageWord(m>>vec3u(3u));let mat=NB_STRIDE*mw+nbTileIndex(m);
          let lift=sign*4.0*h*us;
          if(pres[NB_PRES_C_PPHI+mat]< -4.0||!separated){u=us-sign*lift/(4.0*h);}
          else{u=us+sign*x/(4.0*h);}
        }
        ${L.detail ? "// A detail engine's coupled box face (not a wall): the prescribed velocity, US.\n        else if(other==-1.0){u=us;}\n        " : ""}else{u=us+sign*x/(4.0*h*poNativeTheta(Lt,T,a));}
      }
      // A still page's discarded correction: its liquid faces' only (a face with no liquid side is the extension's, not
      // the projection's, so its kick is no correction).
      if(still){if(own.y>0.0||low.y>0.0){atomicMax(&pcFrozen[((i>>2u)&1u)|(((i>>5u)&1u)<<1u)|(((i>>8u)&1u)<<2u)],bitcast<u32>(abs(u-held)));}continue;}
      // The nozzle (umProjectWithSource): applyInflowVelocity averaged over the 4h face's 16 texels, the cells below it.
      if(nbInflowOn()&&T[a]!=0){u=pcInflowProjected(T,a,u);}
      vel[NB_VEL_C_U0+fa]=u;
      if(wall){
        // The owner, a 4h-owned tile: T on plane 0 (its low face), Lt on plane N (its high face). Held is N_HELD.
        let atLow=T[a]==0;let m=select(Lt,T,atLow);let mx=select(low,own,atLow);let sign=select(1.0,-1.0,atLow);
        let mw=nbPageWord(m>>vec3u(3u));let mat=NB_STRIDE*mw+nbTileIndex(m);let root=pres[NB_PRES_C_PPHI+mat];
        var ghost=0.0;if(mx.y>0.0&&(root< -4.0||!separated)){ghost=mx.x+sign*4.0*h*us;}
        if(poRelease(root,ghost,sign,u,4.0,h)){release|=PO_WALL0<<a;}
      }
    }
    if(!still){phi[fw]=(flags&~PO_WALLS)|release;}
  }
}`, "pressureProjectC", [...PROJECT, CENSUS_RW]);

  // projectRest (opt.live: the engine's project, F_ALL G1, after the solve's finish placed projectF over HP): a brick
  // off HP (no h-owned tile in it or across its lower faces) stores no h face, so only its cells' FLAGS change, WARM
  // and the wall bits cleared (projectF's path for a cell it does not own); a brick of RL (off HP, its bits not
  // certified clear since the last step) clears them. Workgroup g takes RL's entry g, so the F_ALL dispatch's other
  // workgroups exit at once (64 lanes, no shared memory).
  const projectRest = live === undefined ? undefined : k("pressureProjectRest", LIVE + /* wgsl */`
@compute @workgroup_size(64) fn pressureProjectRest(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(g>=u32(pmgF[PO_RLN])){return;}
  let slot=u32(pmgF[PO_RLL+g]);
  for(var j=0u;j<8u;j++){let at=NB_STRIDE*slot+lid+64u*j;phi[NB_PHI_F_FLAGS+at]&=~(PO_WARM|PO_WALLS);}
}`, "pressureProjectRest", projectB);

  return { tileRoots, rowsF, rowsC, projectF, projectDry, projectC, projectRest };
}
