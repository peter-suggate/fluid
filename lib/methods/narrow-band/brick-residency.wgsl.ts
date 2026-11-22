/**
 * The head's residency (brick-abi.md §5.4, lane E): a local, per-source, per-axis reach (seep's S2j, the per-axis CFL
 * activation band: seep-techniques.md §6, progress-lane-s.md 2026-11-14; Bridson 2015 §6.3, Ferstl et al. 2016 §3).
 *
 * Timing is the old frame head's: part 1's head, after the census and before stages.move. It reads the completed
 * step's active class (ACT, setMark's, brick-abi.md §5.5), particle support (PSUPPORT, F's pJoin) and velocity bound
 * (BOUND_F: at the head, project's signed per-axis face maxima of the completed step, six words a slot; setMark clears pCount's particle speeds,
 * brick-set.wgsl.ts setMark's last store, before project raises it, pressure-composite.wgsl.ts project). Both engines
 * run it.
 *
 * resMark: a workgroup (64 lanes) a source brick of F_ALL (G1; tiered FC_G1, F_ALL's bricks then C_ALL's pages).
 * A brick source has up to four source classes, each a box: the vertex box of the class's tiles grown per axis and
 * side by the class's move and its margin (RES_MARGIN), the margin the largest distance a stage reads past the class's
 * tiles (lane E's read table, 2026-11-16, progress-lane-e.md; file:line in the table's rows below, L levelset-kernels,
 * P particles-kernels, F particles-formulas, V velocity-kernels, C levelset-coarse, all .wgsl.ts):
 * - Owners, ACT.RES (an h-owned tile, tiered; a φ < 0 vertex; a particle). All-h: bricks by flow + 20. Tiered (R3,
 *   lane E, the coordinator's ruling 2026-11-16): the h ring, bricks by RES_MARGIN.own.tiered = 4 cells (one tile)
 *   with no move, half-open (cells lo .. hi - 1: the bricks of the 27 tiles about each), since the h ownership is
 *   fixed through the step (the head's transfer commits it, after this residency, and the census keeps each band
 *   tile's same ring); pages by flow + 20. flow is the reach rule's velocity term (reachSet, brick-abi.md §5.4), per axis and side
 *   (2026-11-19, the old localSpeed's signed per-axis reach, umSignedFaceExtent): a box's low side along a grows by
 *   (BOUND_F's -a word + max(-g_a, 0) dt) dt / h, its high side by the +a word and max(g_a, 0). An isotropic flow
 *   grew a box on all six sides by the brick's fastest face: a cut sliver's 14 m/s face (θ 0.03 at aperture 0.02)
 *   claimed 9 bricks every way about each shore brick (hero-garden-hose-x10, 19k F bricks at step 2). Tiered, the tiles with particles (any F slot's, owner or not: a tile coarsened at this head keeps its records for this move) (TILE_OCC bits 0..7, pScan's) are a fourth
 *   class, bricks by flow + RES_MARGIN.particles = flow + 4 (the particles move). The rows (an F tile's reads past
 *   the owners), tiered:
 *   - lcRetire 4 (C, lcRetired): it builds (lcBuildDistance: fgReach1 and FC_NEAR at the 4h cell) only within
 *     lcStencilFine's 1 tile of an h-owned tile; FC_NEAR's cell is an F slot's (lNear finds crossings of F slots
 *     only). lNear's ±5 window may now hold tiles no F slot held: lcNearKnown (C deviation 10) reads FC_NEAR only
 *     where every crossing tile of the window was in an F slot (else the C terms bound the distance), so the window
 *     sets no brick reach.
 *   - lcCouple 0 (C, lcCoupled): its F reads (fgReach1, FC_NEAR, SURFACE_TILE, FC_KEY, the bin counts) are taken
 *     only where known (C deviation 10: c's slot resident and not fresh, lNear's window known; an absent bin cell is
 *     open with no particles, as a present empty one), else the C level set stands in; θ is 0 at a tile no slot
 *     holds (deviation 7). (It was 7 before R3: the reads at an absent tile read as far.)
 *   - pTiles' FT_THETA and SURFACE_TILE 4 (P, pTiles: the particle's tile and the neighbour toward its half; a
 *     tile no F slot holds is skipped): the particle class, flow + 4. The DEPTH reads of a particle (fgDepth at its
 *     position and sites, about 1.25 cells) and the move's taps (1.5 cells) lie inside it.
 *   - uExtend 3 (V l.695-762), the sampler's UNIT taps 3 (fluid-grid fsUnit: a tap lies within 3 cells of an h owner
 *     of the opening tiers; since c2f9bffa uAdvect reads a tap with no F slot as 0, fsUnitAbsent, so this row is a
 *     correctness bound, not a cost: tell lane F before any change that could thin it), the + faces, vertex rows,
 *     tile-word stencils and setMark's 1-ring 1 (V l.304-831, L l.256-296, 394-398): all within the h ring's 4.
 *   - pSeed's re-entry seed (P, pSeedCell cfg.w = 2, part 1): the tiles the head refines, inside their census ring
 *     (the same 27 tiles), and their seeds' first move (flow, under a tile) too.
   - lDilate 20 (R = 5 tiles, separable x, y, z with intermediates DA, DB per brick, L l.346-410) reads through the
 *     C pages tiered: a tile no F slot holds reads its page's DIL4 word (lDilateC, L l.412-511, its sources, DA in
 *     the y pass, DB in the z pass), so it sets no brick reach. lDilateC latches FAIL.TIER 0x4c440001 at an h-owned
 *     tile with no F slot (L l.498; an h-owned tile is an owner, so its brick is resident). All-h there is no DIL4
 *     (the intermediates are per brick only), so the all-h owners keep lDilate's 20.
 *   The pages' row (tiered): lDilate through C 20. Every lDilate source (AIR_SEED, OCCUPIED, CROSSING: a φ ≤ 0
 *   corner, or SURFACE_TILE at a particle's tile) is an h-owned tile, a particle's tile or a 4h liquid tile, and a
 *   tile whose DA or DB is short of R lies within 20 cells (per axis) of a source; the owners' pages (flow + 20) and
 *   the page sources' (flow + 20, below) hold every such tile, so lDilateC's staged sources and DIL4 words match the
 *   whole lattice's (a tile in no resident page reads as no source, the saturated value it would have).
 * - Support, PSUPPORT word 0 (pJoin's support tiles, floor((x - 2)/4) + {0, 1}, P l.417-426): max(travel, flow) + 0,
 *   travel PSUPPORT's particle travel that way (cells, words 1..6). Every read of the next move lies within 1.5 cells
 *   of the moved particle (G2P's taps, pDeposit's quadratic B-spline, pSupported, separation, P l.108-210, 285-301,
 *   F l.68-73, velocity-formulas l.33-48), inside the ±2 support pJoin recorded about the same particle; the move is
 *   max(travel, flow) (pJoin's travel or the flow, whichever is larger), so the margin past it is 0. A shortfall
 *   latches pJoin's REACH (it fails explicitly).
 * - Seeds, ACT.SEED (a tile with heat at the last setMark or the previous head's TRIG) and this head's TRIG
 *   (censusDecide's, earlier in this head; not in ACT): + 6, no move. pSeed seeds a cell where FT_T > 0 (P l.320),
 *   in h-owned tiles only (R1); FT_T is half the 27-tile max of lHeatNow (L l.394-409), and lHeatNow, the last FT_H
 *   cooled by 4 dt or 2 on this head's TRIG (L l.362-365), is FT_T's only source, so a seed's cell is within a tile (4
 *   cells) of a seed tile; its site lies in the cell (F l.229-235) and its first pJoin, in the same tail before any
 *   move, records the ±2 support about it: 6 cells. Without this head's TRIG a fresh seed in a 4h tile latched
 *   pJoin's REACH at the tiered dam's far walls (brick-abi.md §5.4).
 * Each box keeps its bricks and, tiered, its pages.
 * - A page source (tiered, pass 0) is a page with 4h-owned liquid tiles (a C corner φ < 0, setMarkPages' count) at
 *   the head; its box is those tiles' box grown by flow + 20, the flow's speed the largest |u| on their C plane
 *   faces (the completed step's projected 4h velocity; the 4h interpolant inside a tile is bounded by its planes).
 *   20 is lDilate through C (the owners' pages row above: a 4h liquid tile is a source). The rest are inside it:
 *   lcAdvect 16 (the zero set is inside the box, so a vertex of lcAdvect's band, |φ| < 8, is within 8 cells of it,
 *   and its foot stencil, φ at ±4 and ±8, C l.300-303, and cubic taps, ±8 about the foot, C l.333-340, read 8 past
 *   that), lcClear's collar 16 (C l.231), lcRetire 12 (C l.548-553; driven by the brick owners), uExt4Seed's writes
 *   12 and uExt4's sweeps 8 (V l.787-915), the pressure rows 8 by storage (pressure-composite l.599-620),
 *   vcAdvectPage 4 past the trace (V l.272-289). It
 *   keeps pages only: 4h liquid is C storage, and requesting bricks for it would make the deep 4h interior h storage
 *   (the tiered dam filled its 4913-brick pool by step 42 that way, against 4820 at the peak for the old rule).
 * A brick or page no source covers leaves at this commit (setLeave, setLeavePages; no hysteresis, as the old rule).
 * - cfg.z = 0: pass 0 keeps present bricks (MARK_F) and claims absent bricks of resident pages; a brick in an absent
 *   page claims the page; tiered, the box's pages are kept or claimed (MARK_C).
 * - cfg.z = 1: pass 1, after setClaimPages, claims the box's bricks in the pages pass 0 entered (C_ENTER), then zeroes
 *   the source's PSUPPORT words 0..6.
 * - RES_RADIUS: the largest class reach this head, in tiles (rounded up).
 * The work scales with the sources and their reach, never with the lattice.
 *
 * Deviation, stated (lane E, 2026-11-15, coordinator's ruling: storage, residency, lists and kernel shape follow
 * seep-m1max): the old NB-FLIP residency (pageSeed and lrAsk, uniform-mixed-dynamic.ts l.2097-2148,
 * uniform-local-residency.ts l.140-158) is retired: its per-stage seed records, the global departureTravel and
 * sampledFlow envelope and the 5-tile RESIDENCY_MARGIN_TILES. The reach is the source's own (seep S2j), not the
 * region's envelope. Kept latches: setMark's 1-ring FAIL.REACH (a liquid vertex's ring absent), pScatter's REACH (an
 * unbinned particle), and setLeavePages' REACH (a leaving page that held 4h liquid tiles).
 * - All-h: every tile is h, so the h-ownership source is dropped (ACT's rule); the census's WANT does not seed (all-h
 *   WANT is not committed; censusDecide). Tiered: censusDecide keeps each band tile's present ring (R3: the bricks of
 *   its 27 tiles) itself and appends its missing bricks as candidates.
 * - Destinations are bricks and pages; a destination off the brick range is the lattice's wall.
 */
import { BOX_TOKEN, type BrickLayout } from "./brick-abi";
import { CFG, type BrickBinding } from "./brick-kernel";
import { censusKernel } from "./brick-census.wgsl";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { BRICK_SOURCE_BINDING, BRICK_SOURCE_RING_WGSL } from "./brick-source.wgsl";
import { brickDetailWGSL } from "./brick-detail.wgsl";

/**
 * Each source class's margin past its move, in cells, the largest distance a stage reads past the class's tiles (the
 * header's table): owners (ACT.RES) as bricks (`own`: tiered the h ring with no move, all-h past the flow) and,
 * tiered, as pages past the flow (`ownPages`); tiered, the tiles with particles past the flow (`particles`); support (PSUPPORT) past the travel or the flow; seeds (ACT.SEED and this head's TRIG) with no move;
 * page sources (tiered, 4h-owned liquid) past their flow.
 */
export const RES_MARGIN = { own: { tiered: 4, allH: 20 }, ownPages: 20, particles: 4, support: 0, seed: 6, page: 20 } as const;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRW: BrickBinding = { name: "dir", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const phiRW: BrickBinding = { name: "phi", kind: "rw", type: "array<u32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };

export function brickResidencyKernels(L: BrickLayout, opt: { readonly tiered?: boolean } = {}) {
  const tiered = opt.tiered === true;
  // The claim helpers resMark and resSource share.
  const RQ_WGSL = /* wgsl */`
// NONE -> CLAIM by compare-exchange (weak: retried while the word is still NONE). True for the one winner.
fn rqTake(i:u32)->bool{
  loop{
    let x=atomicCompareExchangeWeak(&dir[i],NB_NONE,NB_CLAIM);
    if(x.exchanged){return true;}
    if(x.old_value!=NB_NONE){return false;}
  }
}
// The winner's candidate (KEY_CLAIMED); past the list's capacity its word goes back to NONE and LIST latches.
fn rqAppend(key:u32,word:u32){
  let e=atomicAdd(&nbSet[NB_H_N_CAND],1u);
  if(e<NB_FCAP){atomicStore(&nbSet[NB_S_CAND+e],key|NB_KEY_CLAIMED);}else{atomicStore(&dir[word],NB_NONE);nbFail(NB_FAIL_LIST,key);}
}
fn rqPage(p:vec3i){
  let pw=NB_D_PAGE+nbPageIndex(p);let ps=atomicLoad(&dir[pw]);
  if(ps<NB_SEAM){if(atomicLoad(&nbSet[NB_S_MARK_C+ps])==0u){atomicOr(&nbSet[NB_S_MARK_C+ps],1u);}return;}
  if(ps==NB_NONE){if(rqTake(pw)){rqAppend(nbKey(4*p)|NB_KEY_PAGE,pw);atomicAdd(&nbSet[NB_H_RES_CLAIMED_C],1u);}return;}
  if(ps!=NB_CLAIM){nbFail(NB_FAIL_UNSUPPORTED,nbKey(4*p));}
}
// A destination no meet can change, tested before the meets (which loop over seed bits): a kept brick, or one a
// claim already holds; in pass 0 a brick whose page is claimed (pass 1 takes it); in pass 1 any brick outside the
// pages pass 0 entered. Any other word falls through to rqBrick, which latches what it does not support.
fn rqSettled(q:vec3i,rpass:u32)->bool{
  let ps=atomicLoad(&dir[NB_D_PAGE+nbPageIndex(q>>vec3u(2u))]);
  if(ps<NB_SEAM){
    if(rpass==1u&&(atomicLoad(&dir[NB_D_CREC+32u*ps+NB_REC_CLASS])&NB_CLASS_ENTERING)==0u){return true;}
    let s=atomicLoad(&dir[NB_D_CHILD+64u*ps+nbBrickLocal(q)]);
    return s==NB_CLAIM||(s<NB_SEAM&&atomicLoad(&nbSet[NB_S_MARK_F+s])!=0u);
  }
  return ps==NB_CLAIM||(ps==NB_NONE&&rpass==1u);
}
fn rqPageSettled(p:vec3i)->bool{
  let ps=atomicLoad(&dir[NB_D_PAGE+nbPageIndex(p)]);
  return ps==NB_CLAIM||(ps<NB_SEAM&&atomicLoad(&nbSet[NB_S_MARK_C+ps])!=0u);
}
fn rqBrick(q:vec3i,rpass:u32){
  let pw=NB_D_PAGE+nbPageIndex(q>>vec3u(2u));let ps=atomicLoad(&dir[pw]);
  if(ps<NB_SEAM){
    let cw=NB_D_CHILD+64u*ps+nbBrickLocal(q);let s=atomicLoad(&dir[cw]);
    if(s<NB_SEAM){if(atomicLoad(&nbSet[NB_S_MARK_F+s])==0u){atomicOr(&nbSet[NB_S_MARK_F+s],1u);}return;}
    if(s==NB_NONE){${L.detail ? "if(!nbDtlBrickKeep(q)){return;}" : ""}if(rqTake(cw)){rqAppend(nbKey(q),cw);atomicAdd(&nbSet[NB_H_RES_CLAIMED_F],1u);}return;}
    if(s!=NB_CLAIM){nbFail(NB_FAIL_UNSUPPORTED,nbKey(q));}
    return;
  }
  if(ps==NB_NONE){${L.detail ? "if(!nbDtlBrickKeep(q)){return;}" : ""}if(rpass==0u&&rqTake(pw)){rqAppend(nbKey(4*(q>>vec3u(2u)))|NB_KEY_PAGE,pw);atomicAdd(&nbSet[NB_H_RES_CLAIMED_C],1u);}return;}
  if(ps!=NB_CLAIM){nbFail(NB_FAIL_UNSUPPORTED,nbKey(q));}
}
`;
  const pages = L.lattice.map(n => Math.ceil((Math.floor(n / 8) + 1) / 4));
  const resMark = censusKernel(L, "resMark", tiered ? [CFG, setRW, dirRW, phiRO, velRO] : [CFG, setRW, dirRW], (tiered ? brickOwnerWGSL({ phi: "phi", vel: "vel", box: BOX_TOKEN }) : "") + /* wgsl */`
const RQ_PAGES:vec3i=vec3i(${pages.join(",")});
const RM_OWN:f32=${tiered ? RES_MARGIN.own.tiered : RES_MARGIN.own.allH}.0;
const RM_OWN_PAGES:f32=${RES_MARGIN.ownPages}.0;
const RM_SUPPORT:f32=${RES_MARGIN.support}.0;
const RM_PARTICLES:f32=${RES_MARGIN.particles}.0;
const RM_SEED:f32=${RES_MARGIN.seed}.0;
const RM_PAGE:f32=${RES_MARGIN.page}.0;
var<workgroup> rmLo:array<vec3i,5>;
var<workgroup> rmHi:array<vec3i,5>;
var<workgroup> rmN:u32;
var<workgroup> rmPLo:array<vec3i,5>;
var<workgroup> rmPHi:array<vec3i,5>;
var<workgroup> rmGo:u32;
var<workgroup> rmPage:u32;
var<workgroup> rmTlo:array<atomic<u32>,3>;
var<workgroup> rmThi:array<atomic<u32>,3>;
var<workgroup> rmSpeed:atomic<u32>;
${RQ_WGSL}// A brick source (slot s of F_ALL with ACT.RES, particle support, ACT.SEED or this head's TRIG): a box a class, its tiles'
// vertex box grown per axis and side by the class's move and margin, as bricks; pass 1 then zeroes the support. Tiered, a page
// source (pass 0, a page of C_ALL after F_ALL's bricks): the box of its 4h-owned liquid tiles (a C corner φ < 0, as
// setMarkPages counts them) grown by flow + RM_PAGE, the flow from the largest |u| on those tiles' C plane faces;
// it keeps or claims pages only (4h storage). Both raise RES_RADIUS in pass 0.
fn rmFlow(speed:f32)->f32{
  let dt=nbRingF(cfg.x,NB_RING_DT);let h=nbRingF(cfg.x,NB_RING_H);
  let gv=vec3f(nbRingF(cfg.x,NB_RING_GRAVITY),nbRingF(cfg.x,NB_RING_GRAVITY+1u),nbRingF(cfg.x,NB_RING_GRAVITY+2u));
  // The reach rule's velocity term (reachSet): (bound + |g| dt) dt / h cells.
  return (speed+length(gv)*dt)*dt/h;
}
// A brick source's flow toward one side (side 0 the axis' low side, 1 its high side): BOUND_F's word for that side and
// gravity's component that way, (bound + max(±g_a, 0) dt) dt / h cells (the old localSpeed's signed per-axis reach).
fn rmFlowSide(s:u32,a:u32,side:u32)->f32{
  let dt=nbRingF(cfg.x,NB_RING_DT);let h=nbRingF(cfg.x,NB_RING_H);
  let ga=nbRingF(cfg.x,NB_RING_GRAVITY+a);let gs=max(select(-ga,ga,side==1u),0.0);
  return (bitcast<f32>(atomicLoad(&nbSet[NB_S_BOUND_F+NB_BOUND_WORDS*s+3u*side+a]))+gs*dt)*dt/h;
}
@compute @workgroup_size(64) fn resMark(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);let rpass=cfg.z;
  // Kind: 0 nothing, 1 a brick source, 2 a page source.
  if(lane==0u){
    var kind=0u;let nF=nbListCount(NB_LIST_F_ALL);rmN=0u;
    if(g<nF){
      let s=nbListSlot(NB_LIST_F_ALL,g);let at=NB_S_PSUPPORT+NB_PSUPPORT_WORDS*s;
      let act=atomicLoad(&nbSet[NB_S_ACT+s]);
      // The source classes (c 0 owners ACT.RES, 1 support PSUPPORT word 0, 2 seeds ACT.SEED | this head's TRIG, 3 tiered the tiles with particles),
      // a box each, grown by the class's move and its read-by margin (RES_MARGIN, the header's table).
      let mo=(act>>10u)&0xffu;let ms=atomicLoad(&nbSet[at])&0xffu;let md=((act>>18u)&0xffu)|(atomicLoad(&nbSet[NB_S_TRIG+s])&0xffu);
      // Tiered, class 3: the tiles with particles (any F slot's, owner or not: a tile coarsened at this head keeps its records for this move) (TILE_OCC bits 0..7, pScan's), past their move (R3).
      let mp=${tiered ? "atomicLoad(&nbSet[NB_S_TILE_OCC+s])&0xffu" : "0u"};
      let o=nbFOrigin(s);var reach=0.0;var nb=0u;
      // Tiered, class 4: the tiles this head refines from 4h (WANT h, CMASK still 4h: the transfer commits later in the
      // head). pSeedCell's re-entry seed (cfg.w = 2, part 1) fills their band after this pass and moves the seeds in this
      // step's move at the post-transfer grid velocity (the trilinear of the C faces), so they are the particle class
      // past the larger of the slot's flow and the tiles' C face flow (their records, as class 3's, reach pJoin's support).
      var mr=0u;var sr=0.0;
      ${tiered ? `{
        let T0=o>>vec3u(2u);
        for(var j=0u;j<8u;j++){
          let T=T0+vec3i(vec3u(j&1u,(j>>1u)&1u,j>>2u));if(!nbTileInLattice(T)){continue;}
          let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){continue;}let i=nbTileIndex(T);
          if(((atomicLoad(&nbSet[NB_S_WANT+16u*cs+(i>>5u)])>>(i&31u))&1u)==0u||nbTileH(cs,i)){continue;}
          mr|=1u<<j;
          for(var a=0u;a<3u;a++){var e=vec3i(0);e[a]=1;sr=max(sr,max(abs(nbCFace(T,a)),abs(nbCFace(T+e,a))));}
        }
      }` : ""}
      let flowC=rmFlow(sr);
      for(var c=0u;c<5u;c++){
        let m=select(select(select(select(md,ms,c==1u),mo,c==0u),mp,c==3u),mr,c==4u);
        if(m==0u){continue;}
        var tlo=vec3i(1);var thi=vec3i(0);
        for(var t=0u;t<8u;t++){if(((m>>t)&1u)!=0u){let tt=vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u));tlo=min(tlo,tt);thi=max(thi,tt);}}
        var lo=vec3i(0);var hi=vec3i(0);var pLo=vec3i(0);var pHi=vec3i(0);
        for(var a=0u;a<3u;a++){
          // Owners: tiered, the h ring, one tile and no move (R3: the h ownership is fixed through the step; the rows
          // are the h-anchored reads, the header's table), half-open; all-h, flow + RM_OWN.
          let fl=rmFlowSide(s,a,0u);let fh=rmFlowSide(s,a,1u);
          var rl=${tiered ? "RM_OWN" : "RM_OWN+fl"};var rh=${tiered ? "RM_OWN" : "RM_OWN+fh"};
          if(c==3u){rl=RM_PARTICLES+fl;rh=RM_PARTICLES+fh;}
          if(c==4u){rl=RM_PARTICLES+max(fl,flowC);rh=RM_PARTICLES+max(fh,flowC);}
          if(c==1u){rl=RM_SUPPORT+max(bitcast<f32>(atomicLoad(&nbSet[at+1u+a])),fl);rh=RM_SUPPORT+max(bitcast<f32>(atomicLoad(&nbSet[at+4u+a])),fh);}
          if(c==2u){rl=RM_SEED;rh=RM_SEED;}
          lo[a]=i32(floor(f32(o[a]+4*tlo[a])-rl));hi[a]=i32(ceil(f32(o[a]+4*thi[a]+4)+rh));
          // Tiered owners' cells are [lo, hi): the last brick is the one holding cell hi - 1.
          ${tiered ? "if(c==0u){hi[a]=hi[a]-1;}" : ""}
          reach=max(reach,max(rl,rh));
          // Tiered, the pages: the owners' by flow + RM_OWN_PAGES (lDilate's reach through the C pages), the others' as
          // their bricks'.
          pLo[a]=lo[a];pHi[a]=hi[a];
          ${tiered ? `if(c==0u){let pl=RM_OWN_PAGES+fl;let ph=RM_OWN_PAGES+fh;pLo[a]=i32(floor(f32(o[a]+4*tlo[a])-pl));pHi[a]=i32(ceil(f32(o[a]+4*thi[a]+4)+ph));reach=max(reach,max(pl,ph));}` : ""}
        }
        rmLo[nb]=max(lo>>vec3u(3u),vec3i(0));rmHi[nb]=min(hi>>vec3u(3u),NB_BRICKS-vec3i(1));
        rmPLo[nb]=max(pLo>>vec3u(3u),vec3i(0));rmPHi[nb]=min(pHi>>vec3u(3u),NB_BRICKS-vec3i(1));nb++;
      }
      rmN=nb;
      if(nb!=0u){
        kind=1u;
        if(rpass==0u){atomicMax(&nbSet[NB_H_RES_RADIUS],u32(ceil(reach/4.0)));}
        else{for(var k=0u;k<7u;k++){atomicStore(&nbSet[at+k],0u);}}
      }
      if(rpass==1u&&nbListCount(NB_LIST_C_ENTER)==0u){kind=0u;}
    }${tiered ? `
    else if(rpass==0u&&g<nF+nbListCount(NB_LIST_C_ALL)){
      kind=2u;rmPage=nbListSlot(NB_LIST_C_ALL,g-nF);
      for(var a=0u;a<3u;a++){atomicStore(&rmTlo[a],8u);atomicStore(&rmThi[a],0u);}
      atomicStore(&rmSpeed,0u);
    }` : ""}
    rmGo=kind;
  }
  let kind=workgroupUniformLoad(&rmGo);
  if(kind==0u){return;}
  if(kind==1u){
    let nb=workgroupUniformLoad(&rmN);
    for(var c=0u;c<nb;c++){
      let a=rmLo[c];let b=rmHi[c];
      if(all(a<=b)){
        let ext=vec3u(b-a+vec3i(1));let n=ext.x*ext.y*ext.z;
        for(var j=lane;j<n;j+=64u){
          let q=a+vec3i(vec3u(j%ext.x,(j/ext.x)%ext.y,j/(ext.x*ext.y)));
          if(!rqSettled(q,rpass)){rqBrick(q,rpass);}
        }
      }${tiered ? `
      if(rpass==0u){
        let pa=max(rmPLo[c]>>vec3u(2u),vec3i(0));let pb=min(rmPHi[c]>>vec3u(2u),RQ_PAGES-vec3i(1));
        if(all(pa<=pb)){
          let ext=vec3u(pb-pa+vec3i(1));let n=ext.x*ext.y*ext.z;
          for(var j=lane;j<n;j+=64u){
            let p=pa+vec3i(vec3u(j%ext.x,(j/ext.x)%ext.y,j/(ext.x*ext.y)));
            if(!rqPageSettled(p)){rqPage(p);}
          }
        }
      }` : ""}
    }
  }
  ${tiered ? `
  if(kind!=2u){return;}
  // The page source: its 4h-owned liquid tiles' box and their largest C plane face speed.
  let cs=rmPage;let P=nbKeyCoord(atomicLoad(&dir[NB_D_CREC+32u*cs+NB_REC_KEY]));
  for(var j=0u;j<8u;j++){
    let i=lane+64u*j;let t=vec3u(i&7u,(i>>3u)&7u,i>>6u);let T=8*P+vec3i(t);
    if(!nbTileInLattice(T)||nbTileH(cs,i)){continue;}
    var mn=NB_PHI_AIR;
    for(var q=0u;q<8u;q++){mn=min(mn,nbCVertex(T+vec3i(vec3u(q&1u,(q>>1u)&1u,q>>2u))));}
    if(mn>=0.0){continue;}
    var sp=0.0;
    for(var a=0u;a<3u;a++){
      atomicMin(&rmTlo[a],t[a]);atomicMax(&rmThi[a],t[a]);
      var e=vec3i(0);e[a]=1;sp=max(sp,max(abs(nbCFace(T,a)),abs(nbCFace(T+e,a))));
    }
    atomicMax(&rmSpeed,bitcast<u32>(sp));
  }
  workgroupBarrier();
  if(lane==0u){
    var go=0u;
    if(atomicLoad(&rmTlo[0])<=atomicLoad(&rmThi[0])){
      let r=rmFlow(bitcast<f32>(atomicLoad(&rmSpeed)))+RM_PAGE;
      var lo=vec3i(0);var hi=vec3i(0);
      for(var a=0u;a<3u;a++){
        lo[a]=i32(floor(f32(32*P[a]+4*i32(atomicLoad(&rmTlo[a])))-r));hi[a]=i32(ceil(f32(32*P[a]+4*i32(atomicLoad(&rmThi[a]))+4)+r));
      }
      rmLo[0]=max(lo>>vec3u(5u),vec3i(0));rmHi[0]=min(hi>>vec3u(5u),RQ_PAGES-vec3i(1));
      atomicMax(&nbSet[NB_H_RES_RADIUS],u32(ceil(r/4.0)));
      go=1u;
    }
    rmGo=go;
  }
  if(workgroupUniformLoad(&rmGo)==0u){return;}
  let pa=rmLo[0];let pb=rmHi[0];
  if(all(pa<=pb)){
    let ext=vec3u(pb-pa+vec3i(1));let n=ext.x*ext.y*ext.z;
    for(var j=lane;j<n;j+=64u){
      let p=pa+vec3i(vec3u(j%ext.x,(j/ext.x)%ext.y,j/(ext.x*ext.y)));
      if(!rqPageSettled(p)){rqPage(p);}
    }
  }` : ""}
}`);
  // resSource (lane F, M4; brick-source.ts): the ring's drop boxes (this step's drop and the next step's, each its
  // bounding box grown by the pin margin: requestDropPages and uniformPagePoolDropAhead) as one more brick source,
  // after resMark's pass (cfg.z the pass): their bricks kept or claimed as resMark's boxes are, tiered also their
  // pages in pass 0. The census of the drop's own step then finds its tiles resident (a drop of the first step: the
  // boot's set holds its box, BrickEngineOptions.bootBoxes). The nozzle's plug box (k = 2) is claimed every step, as
  // the old pool pins its pages (uniformPageSources). A workgroup strides the bricks of every box.
  const resSource = censusKernel(L, "resSource", [CFG, setRW, dirRW], BRICK_SOURCE_RING_WGSL + RQ_WGSL + /* wgsl */`
@compute @workgroup_size(64) fn resSource(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) nw:vec3u){
  let rpass=cfg.z;
  for(var k=0u;k<3u;k++){
    let box=nbSourceBox(k);if(!all(box[0]<box[1])){continue;}
    let a=box[0]>>vec3u(3u);let b=(box[1]-vec3i(1))>>vec3u(3u);
    let ext=vec3u(b-a+vec3i(1));let n=ext.x*ext.y*ext.z;
    for(var j=wid.x*64u+lane;j<n;j+=64u*nw.x){
      let q=a+vec3i(vec3u(j%ext.x,(j/ext.x)%ext.y,j/(ext.x*ext.y)));
      if(!rqSettled(q,rpass)){rqBrick(q,rpass);}
    }${tiered ? `
    if(rpass==0u){
      let pa=a>>vec3u(2u);let pb=b>>vec3u(2u);
      let pe=vec3u(pb-pa+vec3i(1));let m=pe.x*pe.y*pe.z;
      for(var j=wid.x*64u+lane;j<m;j+=64u*nw.x){
        let p=pa+vec3i(vec3u(j%pe.x,(j/pe.x)%pe.y,j/(pe.x*pe.y)));
        if(!rqPageSettled(p)){rqPage(p);}
      }
    }` : ""}
  }
}`);
  // resDetail (a detail engine's, brick-detail.wgsl.ts): the shell's and the fresh tiles' storage where the parent's
  // liquid is near (its φ at the tile's centre under NB_DTL_REACH and a tile), one more source after resMark's pass,
  // so the shell holds storage for the liquid the relaxation brings in and a fresh tile for the parent's state it
  // starts from. Tiered it keeps or claims their pages alone (pass 0): the parent's liquid is 4h storage until the
  // census refines it, as a page source's is; all-h, the tile's bricks (both passes, as resSource's). A workgroup
  // strides the active tiles (META.ACTIVE of src's keys); a core tile is the engine's own residency's.
  const d = L.detail;
  const resDetail = d ? censusKernel(L, "resDetail", [CFG, setRW, dirRW, BRICK_SOURCE_BINDING], brickDetailWGSL(L) + RQ_WGSL + /* wgsl */`
@compute @workgroup_size(64) fn resDetail(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) nw:vec3u){
  let rpass=cfg.z;${tiered ? "if(rpass!=0u){return;}" : ""}
  let n=min(nbDtlWord(NB_DTL_META),NB_DTL_BUDGET);
  for(var j=wid.x*64u+lane;j<n;j+=64u*nw.x){
    let i=nbDtlWord(NB_DTL_KEYS+j);let T=vec3i(i32(i%u32(NB_DTL_TILES.x)),i32((i/u32(NB_DTL_TILES.x))%u32(NB_DTL_TILES.y)),i32(i/u32(NB_DTL_TILES.x*NB_DTL_TILES.y)));
    if(nbDtlBit(2u,T)){continue;}
    let lo=NB_DTL_U*vec3f(T);
    if(nbDtlPhi(lo+vec3f(0.5*NB_DTL_U))>NB_DTL_REACH+NB_DTL_U){continue;}
    ${tiered ? "let p=vec3i(lo)>>vec3u(5u);if(!rqPageSettled(p)){rqPage(p);}" : `for(var k=0u;k<${(d.factor / 2) ** 3}u;k++){let q=(vec3i(lo)>>vec3u(3u))+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));if(!rqSettled(q,rpass)){rqBrick(q,rpass);}}`}
  }
}`) : undefined;
  // detailRefresh (a detail engine's, the step's first kernel): the cells of the tiles the mask turned this step get their
  // OPEN back (fresh: FLAG_DTL_OUT set, a container cell that is no static solid's) or withhold it (gone), in their F
  // slots and their pages' C tiles, so a slot made while its tile was inactive (a ring brick) opens with the tile. A
  // workgroup a turned tile (META.FRESH of the fresh list, then META.GONE of the gone list).
  const unit = d ? 4 * d.factor : 0, sub = d ? (unit / 8) ** 3 : 0, subC = d ? (unit / 4) ** 3 : 0;
  const detailRefresh = d ? censusKernel(L, "detailRefresh", [CFG, setRW, dirRO, phiRW, BRICK_SOURCE_BINDING], brickDetailWGSL(L) + /* wgsl */`
fn drFlag(at:u32,fresh:bool){
  let f=phi[at];
  if(fresh&&(f&NB_FLAG_DTL_OUT)!=0u){phi[at]=(f|NB_FLAG_OPEN)&~NB_FLAG_DTL_OUT;}
  if(!fresh&&(f&NB_FLAG_OPEN)!=0u){phi[at]=(f&~NB_FLAG_OPEN)|NB_FLAG_DTL_OUT;}
}
@compute @workgroup_size(64) fn detailRefresh(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) nw:vec3u){
  let nf=min(nbDtlWord(NB_DTL_META+${2}u),NB_DTL_BUDGET);let ng=min(nbDtlWord(NB_DTL_META+${3}u),NB_DTL_BUDGET);
  for(var j=wid.x;j<nf+ng;j+=nw.x){
    let fresh=j<nf;let i=select(nbDtlWord(NB_DTL_GONEL+j-nf),nbDtlWord(NB_DTL_FRESHL+j),fresh);
    let T=vec3i(i32(i%u32(NB_DTL_TILES.x)),i32((i/u32(NB_DTL_TILES.x))%u32(NB_DTL_TILES.y)),i32(i/u32(NB_DTL_TILES.x*NB_DTL_TILES.y)));
    let lo=${unit}*T;
    for(var k=0u;k<${sub}u;k++){
      let b=(lo>>vec3u(3u))+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
      if(!nbBrickInRange(b)){continue;}
      let s=nbFSlotOfBrick(b);if(s>=NB_SEAM){continue;}
      for(var l=lane;l<512u;l+=64u){drFlag(NB_PHI_F_FLAGS+NB_STRIDE*s+l,fresh);}
    }
    for(var k=lane;k<${subC}u;k+=64u){
      let t=(lo>>vec3u(2u))+vec3i(i32(k%${unit / 4}u),i32((k/${unit / 4}u)%${unit / 4}u),i32(k/${(unit / 4) ** 2}u));
      let cs=nbCSlotOfPage(t>>vec3u(3u));if(cs==NB_NONE){continue;}
      drFlag(NB_PHI_C_FLAGS+NB_STRIDE*cs+nbLocal(t),fresh);
    }
  }
}`) : undefined;
  return { resMark, resSource, ...(resDetail ? { resDetail } : {}), ...(detailRefresh ? { detailRefresh } : {}) };
}
