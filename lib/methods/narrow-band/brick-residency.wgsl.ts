/**
 * The head's residency (brick-abi.md §5.4, lane E): a local, per-source, per-axis reach (seep's S2j, the per-axis CFL
 * activation band: seep-techniques.md §6, progress-lane-s.md 2026-11-14; Bridson 2015 §6.3, Ferstl et al. 2016 §3).
 *
 * Timing is the old frame head's: part 1's head, after the census and before stages.move. It reads the completed
 * step's active class (ACT, setMark's, brick-abi.md §5.5), particle support (PSUPPORT, F's pJoin) and velocity bound
 * (BOUND_F: at the head, project's face maximum of the completed step; setMark clears pCount's particle speeds,
 * brick-set.wgsl.ts setMark's last store, before project raises it, pressure-composite.wgsl.ts project). Both engines
 * run it.
 *
 * resMark: a workgroup (64 lanes) a source brick of F_ALL (G1; tiered FC_G1, F_ALL's bricks then C_ALL's pages).
 * A brick source has up to three source classes, each a box: the vertex box of the class's tiles grown per axis and
 * side by the class's move and its margin (RES_MARGIN), the margin the largest distance a stage reads past the class's
 * tiles (lane E's read table, 2026-11-16, progress-lane-e.md; file:line in the table's rows below, L levelset-kernels,
 * P particles-kernels, F particles-formulas, V velocity-kernels, C levelset-coarse, all .wgsl.ts):
 * - Owners, ACT.RES (an h-owned tile, tiered; a φ < 0 vertex; a particle): bricks by flow + 9 tiered, flow + 20
 *   all-h; tiered, pages by flow + 20. flow = (BOUND_F + |g| dt) dt / h is the reach rule's velocity term (reachSet,
 *   brick-abi.md §5.4), the owner region's move this step. The bricks' rows (an F tile's reads past the owners):
 *   - lcRetire 9 (C l.537-553): it builds (lcBuildDistance, C l.490-511, fgReach1 and FC_NEAR at the 4h cell) only
 *     within lcStencilFine's 1 tile of an h-owned tile, then reads FC_NEAR's cell; the largest of the rows, the
 *     tiered owners' margin.
 *   - The DEPTH chain through lDistance and lNear 7 (lNear's box 5, L l.575-586).
 *   - lcCouple 7 (C l.400-436): it reads fgReach1, FC_NEAR, the tile flags (SURFACE_TILE), FC_KEY and the bin counts
 *     at a vertex only where θ > 0, and θ (nbfInterpolatedHeat over the 8 tiles about the vertex, C l.389-396) is
 *     0 farther than a tile from a FT_THETA tile, which pTiles writes within a tile of a particle's tile (P
 *     l.285-301): 4 + 2 + 1 cells past the particle owners' tiles.
 *   - pTiles' FT_THETA and SURFACE_TILE 4 (P l.285-301: the particle's tile and the neighbour toward its half; a
 *     tile no F slot holds is skipped, so the 4h F tiles about the particles need their bricks).
 *   - uExtend 3 (V l.695-762), the sampler's UNIT taps 3 (fluid-grid l.470-474), the + faces, vertex rows,
 *     tile-word stencils and setMark's 1-ring 1 (V l.304-831, L l.256-296, 394-398).
 *   - lDilate 20 (R = 5 tiles, separable x, y, z with intermediates DA, DB per brick, L l.346-410) reads through the
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
 *   deep sites in 4h owners too; FT_T is half the 27-tile max of lHeatNow (L l.394-409), and lHeatNow, the last FT_H
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
 *   WANT is not committed; censusDecide). Tiered: censusDecide keeps each band tile's present 27-brick ring itself and
 *   appends its missing bricks as candidates.
 * - Destinations are bricks and pages; a destination off the brick range is the lattice's wall.
 */
import type { BrickLayout } from "./brick-abi";
import { CFG, type BrickBinding } from "./brick-kernel";
import { censusKernel } from "./brick-census.wgsl";
import { brickOwnerWGSL } from "./brick-owner.wgsl";

/**
 * Each source class's margin past its move, in cells, the largest distance a stage reads past the class's tiles (the
 * header's table): owners (ACT.RES) past the flow, as bricks (`own`, tiered or all-h) and, tiered, as pages
 * (`ownPages`); support (PSUPPORT) past the travel or the flow; seeds (ACT.SEED and this head's TRIG) with no move;
 * page sources (tiered, 4h-owned liquid) past their flow.
 */
export const RES_MARGIN = { own: { tiered: 9, allH: 20 }, ownPages: 20, support: 0, seed: 6, page: 20 } as const;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRW: BrickBinding = { name: "dir", kind: "rw", type: "array<atomic<u32>>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };

export function brickResidencyKernels(L: BrickLayout, opt: { readonly tiered?: boolean } = {}) {
  const tiered = opt.tiered === true;
  const pages = L.lattice.map(n => Math.ceil((Math.floor(n / 8) + 1) / 4));
  const resMark = censusKernel(L, "resMark", tiered ? [CFG, setRW, dirRW, phiRO, velRO] : [CFG, setRW, dirRW], (tiered ? brickOwnerWGSL({ phi: "phi", vel: "vel" }) : "") + /* wgsl */`
const RQ_PAGES:vec3i=vec3i(${pages.join(",")});
const RM_OWN:f32=${tiered ? RES_MARGIN.own.tiered : RES_MARGIN.own.allH}.0;
const RM_OWN_PAGES:f32=${RES_MARGIN.ownPages}.0;
const RM_SUPPORT:f32=${RES_MARGIN.support}.0;
const RM_SEED:f32=${RES_MARGIN.seed}.0;
const RM_PAGE:f32=${RES_MARGIN.page}.0;
var<workgroup> rmLo:array<vec3i,3>;
var<workgroup> rmHi:array<vec3i,3>;
var<workgroup> rmN:u32;
var<workgroup> rmPLo:array<vec3i,3>;
var<workgroup> rmPHi:array<vec3i,3>;
var<workgroup> rmGo:u32;
var<workgroup> rmPage:u32;
var<workgroup> rmTlo:array<atomic<u32>,3>;
var<workgroup> rmThi:array<atomic<u32>,3>;
var<workgroup> rmSpeed:atomic<u32>;
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
    if(s==NB_NONE){if(rqTake(cw)){rqAppend(nbKey(q),cw);atomicAdd(&nbSet[NB_H_RES_CLAIMED_F],1u);}return;}
    if(s!=NB_CLAIM){nbFail(NB_FAIL_UNSUPPORTED,nbKey(q));}
    return;
  }
  if(ps==NB_NONE){if(rpass==0u&&rqTake(pw)){rqAppend(nbKey(4*(q>>vec3u(2u)))|NB_KEY_PAGE,pw);atomicAdd(&nbSet[NB_H_RES_CLAIMED_C],1u);}return;}
  if(ps!=NB_CLAIM){nbFail(NB_FAIL_UNSUPPORTED,nbKey(q));}
}
// A brick source (slot s of F_ALL with ACT.RES, particle support, ACT.SEED or this head's TRIG): a box a class, its tiles'
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
@compute @workgroup_size(64) fn resMark(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);let rpass=cfg.z;
  // Kind: 0 nothing, 1 a brick source, 2 a page source.
  if(lane==0u){
    var kind=0u;let nF=nbListCount(NB_LIST_F_ALL);rmN=0u;
    if(g<nF){
      let s=nbListSlot(NB_LIST_F_ALL,g);let at=NB_S_PSUPPORT+NB_PSUPPORT_WORDS*s;
      let act=atomicLoad(&nbSet[NB_S_ACT+s]);
      // The three source classes (c 0 owners ACT.RES, 1 support PSUPPORT word 0, 2 seeds ACT.SEED | this head's TRIG),
      // a box each, grown by the class's move and its read-by margin (RES_MARGIN, the header's table).
      let mo=(act>>10u)&0xffu;let ms=atomicLoad(&nbSet[at])&0xffu;let md=((act>>18u)&0xffu)|(atomicLoad(&nbSet[NB_S_TRIG+s])&0xffu);
      let flow=rmFlow(bitcast<f32>(atomicLoad(&nbSet[NB_S_BOUND_F+s])));
      let o=nbFOrigin(s);var reach=0.0;var nb=0u;
      for(var c=0u;c<3u;c++){
        let m=select(select(md,ms,c==1u),mo,c==0u);
        if(m==0u){continue;}
        var tlo=vec3i(1);var thi=vec3i(0);
        for(var t=0u;t<8u;t++){if(((m>>t)&1u)!=0u){let tt=vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u));tlo=min(tlo,tt);thi=max(thi,tt);}}
        var lo=vec3i(0);var hi=vec3i(0);var pLo=vec3i(0);var pHi=vec3i(0);
        for(var a=0u;a<3u;a++){
          var rl=RM_OWN+flow;var rh=rl;
          if(c==1u){rl=RM_SUPPORT+max(bitcast<f32>(atomicLoad(&nbSet[at+1u+a])),flow);rh=RM_SUPPORT+max(bitcast<f32>(atomicLoad(&nbSet[at+4u+a])),flow);}
          if(c==2u){rl=RM_SEED;rh=RM_SEED;}
          lo[a]=i32(floor(f32(o[a]+4*tlo[a])-rl));hi[a]=i32(ceil(f32(o[a]+4*thi[a]+4)+rh));
          reach=max(reach,max(rl,rh));
          // Tiered, the pages: the owners' by flow + RM_OWN_PAGES (lDilate's reach through the C pages), the others' as
          // their bricks'.
          pLo[a]=lo[a];pHi[a]=hi[a];
          ${tiered ? `if(c==0u){let rp=RM_OWN_PAGES+flow;pLo[a]=i32(floor(f32(o[a]+4*tlo[a])-rp));pHi[a]=i32(ceil(f32(o[a]+4*thi[a]+4)+rp));reach=max(reach,rp);}` : ""}
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
  return { resMark };
}
