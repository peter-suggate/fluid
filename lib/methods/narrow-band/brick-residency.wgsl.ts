/**
 * The head's residency (brick-abi.md §5.4, lane E): a local, per-source, per-axis reach (seep's S2j, the per-axis CFL
 * activation band: seep-techniques.md §6, progress-lane-s.md 2026-11-14; Bridson 2015 §6.3, Ferstl et al. 2016 §3).
 *
 * Timing is the old frame head's: part 1's head, after the census and before stages.move. It reads the completed
 * step's active class (ACT, setMark's, brick-abi.md §5.5), particle support (PSUPPORT, F's pJoin) and velocity bound
 * (BOUND_F: project's faces and pScatter's particle speeds of the completed step). Both engines run it.
 *
 * resMark: a workgroup (64 lanes) a source brick of F_ALL (G1; tiered FC_G1, F_ALL's bricks then C_ALL's pages).
 * - A brick source has residency tiles (ACT.RES: the ACTIVE tiles less those ACTIVE only by their 4h corners),
 *   particle support (PSUPPORT word 0) or this head's census trigger (TRIG bits 0..7, censusDecide's, earlier in
 *   this head); its source box is the vertex box of those tiles. TRIG is the seed reach: pSeed seeds where the
 *   tile heat FT_T > 0, deep sites in 4h owners too, and FT_T is half the 27-tile heat max (lDilate) with heat 2 on
 *   this head's triggered tiles (lHeatNow). A tile with heat at the last setMark is ACTIVE, so a seed's 2-cell
 *   support is within 6 cells of a source box; a tile triggered at this head is not in ACT (the previous head's
 *   TRIG). Without it a fresh seed in a 4h tile latched pJoin's REACH at the tiered dam's far walls (brick-abi.md
 *   §5.4). Per axis a and side:
 *     reach = max(travel, flow) + 4 ACT_DILATION cells,
 *   where travel is PSUPPORT's particle travel that way (cells, words 1..6) and flow = (BOUND_F + |g| dt) dt / h is
 *   the reach rule's velocity term (reachSet, brick-abi.md §5.4). The 12-cell margin is F_ACT's dilation, so a brick
 *   F_ACT may list next step is resident. It keeps every brick the grown box covers and, tiered, every page.
 * - A page source (tiered, pass 0) is a page with 4h-owned liquid tiles (a C corner φ < 0, setMarkPages' count) at
 *   the head; its box is those tiles' box grown by flow + 12 cells, the flow's speed the largest |u| on their C plane
 *   faces (the completed step's projected 4h velocity; the 4h interpolant inside a tile is bounded by its planes). It
 *   keeps pages only: 4h liquid is C storage, and requesting bricks for it would make the deep 4h interior h storage
 *   (the tiered dam filled its 4913-brick pool by step 42 that way, against 4820 at the peak for the old rule).
 * A brick or page no source covers leaves at this commit (setLeave, setLeavePages; no hysteresis, as the old rule).
 * - cfg.z = 0: pass 0 keeps present bricks (MARK_F) and claims absent bricks of resident pages; a brick in an absent
 *   page claims the page; tiered, the box's pages are kept or claimed (MARK_C).
 * - cfg.z = 1: pass 1, after setClaimPages, claims the box's bricks in the pages pass 0 entered (C_ENTER), then zeroes
 *   the source's PSUPPORT words 0..6.
 * - RES_RADIUS: the largest reach this head, in tiles (rounded up).
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
import { ACT_DILATION, type BrickLayout } from "./brick-abi";
import { CFG, type BrickBinding } from "./brick-kernel";
import { censusKernel } from "./brick-census.wgsl";
import { brickOwnerWGSL } from "./brick-owner.wgsl";

/** The reach's margin past the travel, in cells: F_ACT's dilation (brick-abi.md §5.5). */
export const RES_MARGIN_CELLS = 4 * ACT_DILATION;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRW: BrickBinding = { name: "dir", kind: "rw", type: "array<atomic<u32>>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };

export function brickResidencyKernels(L: BrickLayout, opt: { readonly tiered?: boolean } = {}) {
  const tiered = opt.tiered === true;
  const pages = L.lattice.map(n => Math.ceil((Math.floor(n / 8) + 1) / 4));
  const resMark = censusKernel(L, "resMark", tiered ? [CFG, setRW, dirRW, phiRO, velRO] : [CFG, setRW, dirRW], (tiered ? brickOwnerWGSL({ phi: "phi", vel: "vel" }) : "") + /* wgsl */`
const RQ_PAGES:vec3i=vec3i(${pages.join(",")});
const RM_MARGIN:f32=${RES_MARGIN_CELLS}.0;
var<workgroup> rmLo:vec3i;
var<workgroup> rmHi:vec3i;
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
// A brick source (slot s of F_ALL: ACT.RES tiles | particle support | this head's TRIG, m != 0): the vertex box of those tiles grown
// per axis and side by max(travel, flow) + RM_MARGIN, as bricks; pass 1 then zeroes the support. Tiered, a page
// source (pass 0, a page of C_ALL after F_ALL's bricks): the box of its 4h-owned liquid tiles (a C corner φ < 0, as
// setMarkPages counts them) grown by flow + RM_MARGIN, the flow from the largest |u| on those tiles' C plane faces;
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
    var kind=0u;let nF=nbListCount(NB_LIST_F_ALL);
    if(g<nF){
      let s=nbListSlot(NB_LIST_F_ALL,g);let at=NB_S_PSUPPORT+NB_PSUPPORT_WORDS*s;
      let m=((atomicLoad(&nbSet[NB_S_ACT+s])>>10u)&0xffu)|(atomicLoad(&nbSet[at])&0xffu)|(atomicLoad(&nbSet[NB_S_TRIG+s])&0xffu);
      if(m!=0u){
        var tlo=vec3i(1);var thi=vec3i(0);
        for(var t=0u;t<8u;t++){if(((m>>t)&1u)!=0u){let tt=vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u));tlo=min(tlo,tt);thi=max(thi,tt);}}
        let flow=rmFlow(bitcast<f32>(atomicLoad(&nbSet[NB_S_BOUND_F+s])));
        let o=nbFOrigin(s);var lo=vec3i(0);var hi=vec3i(0);var reach=0.0;
        for(var a=0u;a<3u;a++){
          let rl=max(bitcast<f32>(atomicLoad(&nbSet[at+1u+a])),flow)+RM_MARGIN;
          let rh=max(bitcast<f32>(atomicLoad(&nbSet[at+4u+a])),flow)+RM_MARGIN;
          lo[a]=i32(floor(f32(o[a]+4*tlo[a])-rl));hi[a]=i32(ceil(f32(o[a]+4*thi[a]+4)+rh));
          reach=max(reach,max(rl,rh));
        }
        rmLo=max(lo>>vec3u(3u),vec3i(0));rmHi=min(hi>>vec3u(3u),NB_BRICKS-vec3i(1));
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
    let a=rmLo;let b=rmHi;
    if(all(a<=b)){
      let ext=vec3u(b-a+vec3i(1));let n=ext.x*ext.y*ext.z;
      for(var j=lane;j<n;j+=64u){
        let q=a+vec3i(vec3u(j%ext.x,(j/ext.x)%ext.y,j/(ext.x*ext.y)));
        if(!rqSettled(q,rpass)){rqBrick(q,rpass);}
      }
    }
  }
  ${tiered ? `if(kind==1u&&rpass==0u){
    let pa=max(rmLo>>vec3u(2u),vec3i(0));let pb=min(rmHi>>vec3u(2u),RQ_PAGES-vec3i(1));
    if(all(pa<=pb)){
      let ext=vec3u(pb-pa+vec3i(1));let n=ext.x*ext.y*ext.z;
      for(var j=lane;j<n;j+=64u){
        let p=pa+vec3i(vec3u(j%ext.x,(j/ext.x)%ext.y,j/(ext.x*ext.y)));
        if(!rqPageSettled(p)){rqPage(p);}
      }
    }
  }
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
      let r=rmFlow(bitcast<f32>(atomicLoad(&rmSpeed)))+RM_MARGIN;
      var lo=vec3i(0);var hi=vec3i(0);
      for(var a=0u;a<3u;a++){
        lo[a]=i32(floor(f32(32*P[a]+4*i32(atomicLoad(&rmTlo[a])))-r));hi[a]=i32(ceil(f32(32*P[a]+4*i32(atomicLoad(&rmThi[a]))+4)+r));
      }
      rmLo=max(lo>>vec3u(5u),vec3i(0));rmHi=min(hi>>vec3u(5u),RQ_PAGES-vec3i(1));
      atomicMax(&nbSet[NB_H_RES_RADIUS],u32(ceil(r/4.0)));
      go=1u;
    }
    rmGo=go;
  }
  if(workgroupUniformLoad(&rmGo)==0u){return;}
  let pa=rmLo;let pb=rmHi;
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
