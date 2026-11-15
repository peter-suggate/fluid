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
import type { BrickLayout } from "./brick-abi";
import type { BrickBinding } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { brickCellKernel } from "./brick-pressure.wgsl";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { PRESSURE_ROWS_WGSL } from "./pressure-rows.wgsl";

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
fn poSlotOfCell(c:vec3i)->u32{return nbFSlotOfCell(c);}
fn poGuardVertex(v:vec3i){poGuardVertexG(v);}
fn poFacePlane(c:vec3i,a:u32)->f32{return nbFacePlane(c,a);}
`;
const STAGE_BRICK = /* wgsl */`
var<workgroup> poSV:array<f32,1331>;
var<workgroup> poSC:array<f32,1000>;
var<workgroup> poST:array<u32,64>;
var<private> poSO:vec3i;
fn poIndex(d:vec3i,n:i32)->u32{return u32(d.x+n*(d.y+n*d.z));}
fn poCoord(i:u32,n:u32)->vec3i{return vec3i(i32(i%n),i32((i/n)%n),i32(i/(n*n)));}
fn poVertex(v:vec3i)->f32{let d=v-poSO+vec3i(1);if(all(d>=vec3i(0))&&all(d<vec3i(11))){return poSV[poIndex(d,11)];}return nbVertexPhi(v);}
fn poCentre(c:vec3i)->f32{let d=c-poSO+vec3i(1);if(all(d>=vec3i(0))&&all(d<vec3i(10))){return poSC[poIndex(d,10)];}return poCentreG(c);}
fn poOwnerH(c:vec3i)->bool{let d=(c>>vec3u(2u))-(poSO>>vec3u(2u))+vec3i(1);if(all(d>=vec3i(0))&&all(d<vec3i(4))){return poST[poIndex(d,4)]!=0u;}return nbCellOwnerH(c);}
// The tiles a cold row's start reads (poStart: 2B-1..2B+2, the carry's beyond 2B-2..2B+3 along y): their root φ
// and start, staged only when the brick has a cold row (poTiles).
var<workgroup> poSR:array<f32,96>;
var<workgroup> poSS:array<f32,96>;
var<private> poRC:bool;
fn poTileWindow(T:vec3i)->i32{let d=T-(poSO>>vec3u(2u))+vec3i(1,2,1);if(all(d>=vec3i(0))&&all(d<vec3i(4,6,4))){return d.x+4*(d.y+6*d.z);}return -1;}
fn poRootPhi(T:vec3i)->f32{if(poRC){let i=poTileWindow(T);if(i>=0){return poSR[i];}}return poRootPhiG(T);}
fn poTileStart(T:vec3i)->f32{if(poRC){let i=poTileWindow(T);if(i>=0){return poSS[i];}}return poTileStartG(T);}
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
    let s=poSB[poIndex((v>>vec3u(3u))-(o>>vec3u(3u))+vec3i(1),3)];if(s>=NB_SEAM){return NB_PHI_FAR;}
    return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
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
fn poStage(o:vec3i,lid:u32){
  poSO=o;
  {
    if(lid<8u){poSP[lid]=u32(nbCSlotOfPage((o>>vec3u(5u))+poCoord(lid,2u))!=NB_NONE);}
    let T0=o>>vec3u(2u);
    if(lid<125u){let t=T0-vec3i(2)+poCoord(lid,5u);poSK[lid]=u32(nbTileInLattice(t)&&nbCSlotOfPage(t>>vec3u(3u))!=NB_NONE&&!nbTileOwnerH(t));}
    if(lid<216u){poSQ[lid]=nbCVertex(T0-vec3i(2)+poCoord(lid,6u));}
    if(lid<27u){let b=(o>>vec3u(3u))-vec3i(1)+poCoord(lid,3u);poSB[lid]=select(nbFSlotOfBrick(b),NB_NONE,any(b<vec3i(0)));}
    workgroupBarrier();
  }
  for(var i=lid;i<1331u;i+=256u){poSV[i]=poVertexS(o-vec3i(1)+poCoord(i,11u),o);}
  if(lid<64u){poST[lid]=u32(nbTileOwnerH((o>>vec3u(2u))-vec3i(1)+poCoord(lid,4u)));}
  workgroupBarrier();
  for(var i=lid;i<1000u;i+=256u){
    let d=poCoord(i,10u);var v:array<f32,8>;
    for(var j=0u;j<8u;j++){v[j]=poSV[poIndex(d+poCorner(j),11)];}
    poSC[i]=poMean8(v);
  }
  workgroupBarrier();
}
`;

const COMPOSITE_WGSL = /* wgsl */`
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
fn poTileOpen(t:vec3i)->bool{return nbTileInLattice(t);}
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
  let phi_=poRootPhi(t);
  return select(vec3f(0.0),vec3f(poTileStart(t),phi_,1.0),poWet(phi_,4.0)&&poTileOpen(t));
}
fn poCoarseKind(t:vec3i)->vec3f{
  let open=poTileOpen(t);let wet=poWet(poRootPhi(t),4.0);
  return vec3f(select(0.0,poTileStart(t),open&&wet),f32(open&&wet),f32(open&&!wet));
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
  if(!nbInLattice(q)){return NP_ACROSS_WALL;}
  let w=nbPageWord(q>>vec3u(5u));
  if(w==NB_NONE){return NP_ACROSS_AIR;}
  if(w>=NB_SEAM){nbFail(NB_FAIL_UNSUPPORTED,w);return NP_ACROSS_WALL;}
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
  let volume=select(0.5,1.0,nbInLattice(q));
  let face=npBandFace(across,own,liquid,1.0,1.0,band,native,n,poFacePhi(c,a,sign),volume,true,false,false,1.0,true,false,1.0);
  return PoFace(face,n.wet,native);
}
// An h row's seam head on face f (bSeamHead): its tangential neighbours' starts.
fn poHead(c:vec3i,own:f32,base:f32,f:u32)->f32{
  var usable=0u;var bases:array<f32,6>;
  for(var g=0u;g<6u;g++){
    if(g/2u==f/2u){continue;}
    let m=c+npSign(g)*poUnit(g/2u);
    if(nbInLattice(m)&&poOwnerH(m)){let pm=poCentre(m);if(poWet(pm,1.0)){usable|=1u<<g;bases[g]=poStart(m,pm);}}
  }
  return npSeamHead(poWet(own,1.0),base,f,vec3u(c&vec3i(7)),usable,bases);
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
// poHead with the rows' stored starts (B) and centre φ (PPHI): the head the row's RHS carried.
fn poHeadB(c:vec3i,own:f32,base:f32,f:u32)->f32{
  var usable=0u;var bases:array<f32,6>;
  for(var g=0u;g<6u;g++){
    if(g/2u==f/2u){continue;}
    let m=c+npSign(g)*poUnit(g/2u);
    if(nbInLattice(m)&&nbCellOwnerH(m)){let pm=poPhiF(m);if(poWet(pm,1.0)){usable|=1u<<g;bases[g]=poBF(m);}}
  }
  return npSeamHead(poWet(own,1.0),base,f,vec3u(c&vec3i(7)),usable,bases);
}
// The projected velocity of h cell m's face f (umProject's bFace in the slot units): U = US - sign (c/V) (ghost - X)/h,
// the ghost by the face's kind: a band row's X, a native row's X plus the seam head, 0 at air, and at a separating
// wall X + lift while attached in the solve's final state (separated: the FLAGS bit at the face), else 0. A Neumann
// face keeps US and a closed one is 0. From a dry h cell (liquid false) only a seam to a liquid native row moves
// (the native row's zero ghost); every other face keeps US.
fn poProjectFace(m:vec3i,own:f32,liquid:bool,f:u32,us:f32,h:f32,separated:bool)->f32{
  let a=f/2u;let sign=npSign(f);let face=poBandFace(m,own,liquid,f).face;
  if(face.kind==NP_K_NEUMANN){return us;}
  if(face.kind==NP_K_CLOSED){return 0.0;}
  if(!liquid&&face.kind!=NP_K_SEAM){return us;}
  let x=select(0.0,poXF(m),liquid);let q=m+sign*poUnit(a);
  var ghost=0.0;
  if(face.kind==NP_K_WALL){let lift=npLift(f,1.0,us,h,1.0);if(!separated){ghost=x+lift;}}
  else if(face.kind==NP_K_BAND){ghost=poXF(q);}
  else if(face.kind==NP_K_SEAM){ghost=poXT(q>>vec3u(2u));if(liquid){ghost+=poHeadB(m,own,poBF(m),f);}}
  return us-f32(sign)*(face.coefficient/face.fraction)*(ghost-x)/h;
}
// The old umRelease (uniform-mixed-pressure-velocity.ts) of a lattice wall face with owner width w (1 or 4 cells):
// not held (the owner's centre or root φ >= -w), the projection's wall ghost (the wall slot's pressure) <= 0, and
// -sign U dt > 1e-4 w h, sign the face's side of its owner. A dry owner's ghost is 0.
fn poRelease(phiOwner:f32,ghost:f32,sign:f32,u:f32,w:f32,h:f32)->bool{
  let dt=nbRingF(cfg.x,NB_RING_DT);
  return !(phiOwner< -w)&&ghost<=0.0&&-sign*u*dt>1e-4*w*h;
}
`;

const headerWGSL = (L: BrickLayout, staged = false, tiered = true) => `const PO_TIERED:bool=${tiered};\n` + brickOwnerWGSL({ phi: "phi", vel: "vel", faceF: "NB_VEL_F_US0", faceC: "NB_VEL_C_US0" }) + PRESSURE_ROWS_WGSL + (staged ? STAGE_BRICK : STAGE_DIRECT) + COMPOSITE_WGSL;

/** The composite kernels. rowsF's label and entry are the engine's operator's when pressure-operator.wgsl.ts builds it.
 * `tiered: false` (the all-h engine) drops the absent-page guard: all-h, an absent page is air (brick-abi.md §2.5). */
export function pressureCompositeKernels(L: BrickLayout, names: { rowsF?: [string, string]; projectF?: [string, string] } = {}, opt: { readonly tiered?: boolean } = {}) {
  const tiered = opt.tiered !== false;
  const k = (label: string, body: string, entry = label, bindings: readonly BrickBinding[] = ROWS, staged = false) => brickCellKernel(L, label, bindings, headerWGSL(L, staged, tiered) + body, entry);
  const [rowsLabel, rowsEntry] = names.rowsF ?? ["pressureRowsF", "pressureRowsF"];
  const [projectLabel, projectEntry] = names.projectF ?? ["pressureProjectF", "pressureProjectF"];

  // tileRoots: a workgroup a brick of F_ALL (G1), before the operator (PressureSolver.before). The held fit of each
  // h tile of the brick (umAuthority's h root, pcHeldFit: the mean of its 64 h centres, the vertices 4T..4T+4 at
  // weights 1/2 on the tile's faces, /64) into its C PPHI, from the brick's 9^3 vertices staged once. It was summed
  // by every cold brick for each of the 96 tiles its starts read (up to 12 bricks a tile, 125 vertices each: 7 of
  // the tiered dam operator's 8.7 ms, 2026-11-15).
  const tileRoots = k("pressureTileRoots", /* wgsl */`
var<workgroup> trGo:u32;
var<workgroup> trV:array<f32,729>;
var<workgroup> trP:array<f32,200>;
@compute @workgroup_size(256) fn pressureTileRoots(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(lid==0u){trGo=select(0u,1u,g<nbListCount(NB_LIST_F_ALL));}if(workgroupUniformLoad(&trGo)==0u){return;}
  let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);let T0=o>>vec3u(2u);
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
}`);

  // rowsF: a workgroup a brick of F_ALL (G1), a lane two cells.
  const rowsF = k(rowsLabel, /* wgsl */`
var<workgroup> poGo:u32;
var<workgroup> poCold:atomic<u32>;
@compute @workgroup_size(256) fn ${rowsEntry}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(lid==0u){poGo=select(0u,1u,g<nbListCount(NB_LIST_F_ALL));}if(workgroupUniformLoad(&poGo)==0u){return;}
  let slot=nbListSlot(NB_LIST_F_ALL,g);
  if(lid==0u){atomicStore(&poCold,0u);}
  // A brick of 4h-owned tiles only (the tiered dam's native interior: 5 in 6 of F_ALL) holds no h row: its zero rows
  // are written without staging the brick's phi (the same rows the full path writes for cells it does not own).
  workgroupBarrier();
  if(lid<8u){if(nbTileOwnerH((nbFOrigin(slot)>>vec3u(2u))+poCorner(lid))){atomicOr(&poCold,2u);}}
  workgroupBarrier();
  if(lid==0u){poGo=atomicLoad(&poCold);atomicStore(&poCold,0u);}
  if(workgroupUniformLoad(&poGo)==0u){
    if(lid==0u&&nbFRec(slot,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}
    for(var j=0u;j<2u;j++){
      let at=NB_STRIDE*slot+lid+256u*j;
      pres[NB_PRES_F_B+at]=0.0;pres[NB_PRES_F_X+at]=0.0;pres[NB_PRES_F_RHS+at]=0.0;pres[NB_PRES_F_DIAG+at]=0.0;pres[NB_PRES_F_PPHI+at]=0.0;
      pres[NB_PRES_F_K0+at]=0.0;pres[NB_PRES_F_K1+at]=0.0;pres[NB_PRES_F_K2+at]=0.0;pres[NB_PRES_F_KU0+at]=0.0;pres[NB_PRES_F_KU1+at]=0.0;pres[NB_PRES_F_KU2+at]=0.0;
    }
    return;
  }
  poStage(nbFOrigin(slot),lid);
  for(var j=0u;j<2u;j++){
    let c=nbFOrigin(slot)+nbLocalCoord(lid+256u*j);
    if(nbInLattice(c)&&poOwnerH(c)){if(poWet(poCentre(c),1.0)&&(poFlagsF(c)&PO_WARM)==0u){atomicOr(&poCold,1u);}}
  }
  workgroupBarrier();
  if(lid==0u){poGo=atomicLoad(&poCold);}
  poTiles(lid,workgroupUniformLoad(&poGo)!=0u);
  if(lid==0u&&nbFRec(slot,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}
  let o=nbFOrigin(slot);let r=cfg.x;let h=nbRingF(r,NB_RING_H);
  let rate=bitcast<f32>(atomicLoad(&nbSet[NB_H_LANE_F]));let capacity=1.0;
  for(var j=0u;j<2u;j++){
    let l=lid+256u*j;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
    var diag=0.0;var div=0.0;var extra=0.0;var kk=vec3f(0.0);var ku=vec3f(0.0);
    let owned=nbInLattice(c)&&poOwnerH(c);
    var own=NB_PHI_AIR;if(owned){own=poCentre(c);}
    let liquid=owned&&poWet(own,1.0);
    var start=0.0;
    if(liquid){
      for(var j=0u;j<8u;j++){poGuardVertex(c+poCorner(j));}
      start=poStart(c,own);
      for(var f=0u;f<6u;f++){
        let a=f/2u;let sign=npSign(f);let pf=poBandFace(c,own,true,f);let face=pf.face;
        var fc=c;if(sign>0){fc[a]+=1;}
        let us=poFacePlane(fc,a);
        if(face.kind!=NP_K_CLOSED){div+=f32(sign)*face.fraction*us;}
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
          extra+=face.coefficient*poHead(c,own,start,f);
        }
      }
    }
    // Every cell's X is its start, 0 off the liquid rows (the slot contract's zero rows, cells of 4h-owned tiles and
    // the skirt included: they hold no state, and a refined tile's rows start cold).
    pres[NB_PRES_F_B+at]=start;pres[NB_PRES_F_X+at]=start;
    pres[NB_PRES_F_RHS+at]=select(0.0,-h*div+h*h*rate*capacity+extra,liquid);
    pres[NB_PRES_F_DIAG+at]=diag;pres[NB_PRES_F_PPHI+at]=select(0.0,own,owned);
    pres[NB_PRES_F_K0+at]=kk.x;pres[NB_PRES_F_K1+at]=kk.y;pres[NB_PRES_F_K2+at]=kk.z;
    pres[NB_PRES_F_KU0+at]=ku.x;pres[NB_PRES_F_KU1+at]=ku.y;pres[NB_PRES_F_KU2+at]=ku.z;
  }
}`, rowsEntry, ROWS, true);

  // rowsC: a workgroup a page of C_ALL (G1), a lane two tiles. After rowsF, whose stored rows it reads: a seam
  // subface's h cell m takes its centre φ from PPHI and its head from poHeadB (the h rows' starts in B and centre φ
  // in PPHI), the values rowsF computed, so the subface's coefficient and head are the h row's own, unrecomputed.
  const rowsC = k("pressureRowsC", PROJECT_WGSL + /* wgsl */`
@compute @workgroup_size(256) fn pressureRowsC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  let O=nbCOrigin(cs)>>vec3u(2u);let r=cfg.x;let h=nbRingF(r,NB_RING_H);
  let rate=bitcast<f32>(atomicLoad(&nbSet[NB_H_LANE_F]));let capacity=1.0;
  for(var j=0u;j<2u;j++){
    let i=lid+256u*j;let T=O+nbLocalCoord(i);let at=NB_STRIDE*cs+i;
    var diag=0.0;var div=0.0;var extra=0.0;var kk=vec3f(0.0);var start=0.0;
    let owned=nbTileInLattice(T)&&!nbTileH(cs,i);
    var root=0.0;if(owned){root=poRootPhi4(T);}
    let liquid=owned&&poWet(root,4.0);
    if(liquid){
      for(var j=0u;j<8u;j++){poGuardVertex(4*(T+poCorner(j)));}
      start=poTileStart(T);
      for(var f=0u;f<6u;f++){
        let a=f/2u;let side=f&1u;let sign=npSign(f);let e=poUnit(a);let Tn=T+sign*e;
        let plane=4*select(T,Tn,side==1u);
        if(!nbTileInLattice(Tn)){
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
        if(nbTileOwnerH(Tn)){
          // A seam: the 16 h cells of Tn beside the face, each with the band's own face formula.
          let u=poUnit((a+1u)%3u);let v=poUnit((a+2u)%3u);let g2=f^1u;
          var layer=4*Tn;if(side==0u){layer[a]+=3;}
          var sum=0.0;
          for(var s=0u;s<16u;s++){
            let m=layer+i32(s&3u)*u+i32(s>>2u)*v;
            let pm=poPhiF(m);let lm=poWet(pm,1.0);let pf=poBandFace(m,pm,lm,g2);
            if(pf.face.kind==NP_K_SEAM){
              diag+=pf.face.coefficient;
              if(lm){extra-=pf.face.coefficient*poHeadB(m,pm,poBF(m),g2);}
            }
            var fm=m;if(side==0u){fm[a]+=1;}
            sum+=nbFacePlane(fm,a);
          }
          div+=f32(sign)*(sum/16.0);
          continue;
        }
        let un=nbFacePlane(plane,a);div+=f32(sign)*un;
        if(poWet(poRootPhi4(Tn),4.0)){let c=4.0;diag+=c;if(side==0u){kk[a]=c;}}
        else{diag+=4.0/poNativeTheta(low,high,a);}
      }
    }
    // X keeps the tile's last pressure (the start) on liquid native rows, and on h tiles for the finish.
    if(owned&&!liquid){pres[NB_PRES_C_X+at]=0.0;}
    pres[NB_PRES_C_B+at]=start;
    pres[NB_PRES_C_RHS+at]=select(0.0,-16.0*h*div+64.0*h*h*rate*capacity+extra,liquid);
    pres[NB_PRES_C_DIAG+at]=diag;
    // An h tile's PPHI is tileRoots' held fit, which this pass's seams read across tiles: kept.
    if(!(nbTileInLattice(T)&&nbTileH(cs,i))){pres[NB_PRES_C_PPHI+at]=root;}
    pres[NB_PRES_C_K0+at]=kk.x;pres[NB_PRES_C_K1+at]=kk.y;pres[NB_PRES_C_K2+at]=kk.z;
  }
}`);

  // projectF: a workgroup a brick (G1 of the engine's project list, F_ALL), a lane two cells' lower faces.
  // Writes U on every h face (a face with an h-owned cell on either side, stored at its upper cell), raises
  // BOUND_F, and writes the WARM bit (FLAGS bit 7) of every cell: an h-owned cell that held a liquid row.
  const projectF = k(projectLabel, PROJECT_WGSL + /* wgsl */`
var<workgroup> wSpeed:atomic<u32>;
var<workgroup> poGo:u32;
@compute @workgroup_size(256) fn ${projectEntry}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);
  if(lid==0u){atomicStore(&wSpeed,0u);poGo=select(0u,1u,g<nbListCount(NB_LIST_F_ALL));}
  if(workgroupUniformLoad(&poGo)==0u){return;}
  let slot=nbListSlot(NB_LIST_F_ALL,g);
  let o=nbFOrigin(slot);
  // No h-owned tile in the brick or across its lower faces (the 3^3 tiles from its origin tile - 1): no h face is
  // stored here, so only the cells' FLAGS change (WARM and the wall bits cleared, as the full path for cells it does
  // not own), without staging the brick's phi.
  if(lid<27u){if(nbTileOwnerH((o>>vec3u(2u))-vec3i(1)+poCoord(lid,3u))){atomicOr(&wSpeed,1u);}}
  workgroupBarrier();
  if(lid==0u){poGo=atomicLoad(&wSpeed);atomicStore(&wSpeed,0u);}
  if(workgroupUniformLoad(&poGo)==0u){
    for(var j=0u;j<2u;j++){let at=NB_STRIDE*slot+lid+256u*j;phi[NB_PHI_F_FLAGS+at]&=~(PO_WARM|PO_WALLS);}
    return;
  }
  poStage(o,lid);
  var speed=0.0;
  {
    let h=nbRingF(cfg.x,NB_RING_H);
    for(var j=0u;j<2u;j++){
      let l=lid+256u*j;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
      let hc=nbInLattice(c)&&poOwnerH(c);
      var pc=NB_PHI_AIR;if(hc){pc=pres[NB_PRES_F_PPHI+at];}
      let lc=hc&&poWet(pc,1.0);
      // This cell's FLAGS hold the solver's wall state for its lower faces on the lattice's planes (read here, by
      // this lane only), and leave with the faces' release bits.
      let flags=phi[NB_PHI_F_FLAGS+at];var release=0u;
      for(var a=0u;a<3u;a++){
        if(!nbFaceInLattice(c,a)){continue;}
        let n=c-poUnit(a);let hn=nbInLattice(n)&&poOwnerH(n);
        if(!hc&&!hn){continue;}
        let fa=NB_STRIDE*(slot+a*NB_FCAP)+l;let us=vel[NB_VEL_F_US0+fa];
        var pn=NB_PHI_AIR;if(hn){pn=poPhiF(n);}
        let ln=hn&&poWet(pn,1.0);
        let wall=c[a]==0||c[a]==NB_LATTICE[a];let separated=wall&&(flags&(PO_WALL0<<a))!=0u;
        var u=us;
        if(lc){u=poProjectFace(c,pc,true,2u*a,us,h,separated);}
        else if(ln){u=poProjectFace(n,pn,true,2u*a+1u,us,h,separated);}
        else if(hc&&!hn){u=poProjectFace(c,pc,false,2u*a,us,h,separated);}
        else if(hn&&!hc){u=poProjectFace(n,pn,false,2u*a+1u,us,h,separated);}
        vel[NB_VEL_F_U0+fa]=u;speed=max(speed,abs(u));
        if(wall){
          // The owner: c on plane 0 (its low face), n on plane N (its high face).
          let low=c[a]==0;let sign=select(1.0,-1.0,low);let liquid=select(ln,lc,low);let m=select(n,c,low);
          var ghost=0.0;if(liquid&&!separated){ghost=poXF(m)+sign*us*h;}
          if(poRelease(select(pn,pc,low),ghost,sign,u,1.0,h)){release|=PO_WALL0<<a;}
        }
      }
      phi[NB_PHI_F_FLAGS+at]=(flags&~(PO_WARM|PO_WALLS))|select(0u,PO_WARM,lc&&pres[NB_PRES_F_DIAG+at]>0.0)|release;
    }
    atomicMax(&wSpeed,bitcast<u32>(speed));
  }
  workgroupBarrier();
  if(lid==0u){atomicMax(&nbSet[NB_S_BOUND_F+slot],atomicLoad(&wSpeed));}
}`, projectEntry, PROJECT, true);

  // projectC: a workgroup a page of C_ALL (G1), a lane two tiles, after the solve's finish.
  // - An h tile's C X becomes the mean X of its liquid h rows (083's band-tile mean): the tile's start next step.
  // - Every 4h face (both sides 4h-owned, or a 4h tile on a lattice wall) stored at this page's tiles gets U:
  //   a native–native face the 4h gradient, a native–air face its ghost at the marched θ, a native wall
  //   the separating wall in the state the rows took (attached when start + lift >= 0, or N_HELD), and
  //   a face with no liquid side U = US. In the slot units a native face's coupling c gives
  //   U = US - sign c/(16 h V) (ghost - X), so 4V/(16hV) = 1/(4h).
  const projectC = k("pressureProjectC", PROJECT_WGSL + /* wgsl */`
fn poNativeX(T:vec3i)->vec2f{
  if(!nbTileInLattice(T)){return vec2f(0.0,-1.0);}
  let w=nbPageWord(T>>vec3u(3u));if(w==NB_NONE||w>=NB_SEAM){return vec2f(0.0,-2.0);}
  let i=nbTileIndex(T);if(nbTileH(w,i)){return vec2f(0.0,-3.0);}
  let at=NB_STRIDE*w+i;
  return vec2f(pres[NB_PRES_C_X+at],select(0.0,1.0,pres[NB_PRES_C_DIAG+at]>0.0&&poWet(pres[NB_PRES_C_PPHI+at],4.0)));
}
@compute @workgroup_size(256) fn pressureProjectC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  let O=nbCOrigin(cs)>>vec3u(2u);let h=nbRingF(cfg.x,NB_RING_H);
  for(var j=0u;j<2u;j++){
    let i=lid+256u*j;let T=O+nbLocalCoord(i);let at=NB_STRIDE*cs+i;
    let tIn=nbTileInLattice(T);
    // This tile's C FLAGS: the solver's wall state for its lower faces on the lattice's planes, then their release.
    let fw=NB_PHI_C_FLAGS+at;let flags=phi[fw];var release=0u;
    if(tIn&&nbTileH(cs,i)){
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
      let fa=NB_STRIDE*(cs+a*NB_CCAP)+i;let us=vel[NB_VEL_C_US0+fa];
      let low=poNativeX(Lt);
      let wall=T[a]==0||4*T[a]==NB_LATTICE[a];let separated=wall&&(flags&(PO_WALL0<<a))!=0u;
      var u=us;
      if(own.y>0.0&&low.y>0.0){u=us-(own.x-low.x)/(4.0*h);}
      else if(own.y>0.0||low.y>0.0){
        // The liquid side m, the face's sign from m (T's lower face -1, Lt's upper +1), the other side.
        let mine=own.y>0.0;let sign=select(1.0,-1.0,mine);let x=select(low.x,own.x,mine);let other=select(own.y,low.y,mine);
        let m=select(Lt,T,mine);
        if(other==-1.0){
          let mw=nbPageWord(m>>vec3u(3u));let mat=NB_STRIDE*mw+nbTileIndex(m);
          let lift=sign*4.0*h*us;
          if(pres[NB_PRES_C_PPHI+mat]< -4.0||!separated){u=us-sign*lift/(4.0*h);}
          else{u=us+sign*x/(4.0*h);}
        }
        else{u=us+sign*x/(4.0*h*poNativeTheta(Lt,T,a));}
      }
      vel[NB_VEL_C_U0+fa]=u;
      if(wall){
        // The owner, a 4h-owned tile: T on plane 0 (its low face), Lt on plane N (its high face). Held is N_HELD.
        let atLow=T[a]==0;let m=select(Lt,T,atLow);let mx=select(low,own,atLow);let sign=select(1.0,-1.0,atLow);
        let mw=nbPageWord(m>>vec3u(3u));let mat=NB_STRIDE*mw+nbTileIndex(m);let root=pres[NB_PRES_C_PPHI+mat];
        var ghost=0.0;if(mx.y>0.0&&(root< -4.0||!separated)){ghost=mx.x+sign*4.0*h*us;}
        if(poRelease(root,ghost,sign,u,4.0,h)){release|=PO_WALL0<<a;}
      }
    }
    phi[fw]=(flags&~PO_WALLS)|release;
  }
}`, "pressureProjectC", PROJECT);

  return { tileRoots, rowsF, rowsC, projectF, projectC };
}
