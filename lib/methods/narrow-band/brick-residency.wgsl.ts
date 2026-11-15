/**
 * The head's residency (brick-abi.md §5.4, lane E): NB-FLIP's local residency rule, ported verbatim on the brick
 * storage. These are the old engine's pageSeed (uniform-mixed-dynamic.ts l.2097-2148) and lrAsk
 * (uniform-local-residency.ts l.140-158; launched at uniform-mixed-dynamic.ts l.2221, committed at
 * webgpu-uniform-reference.ts l.2214).
 *
 * Timing is the old frame head's: part 1's head, after the census and before stages.move. It reads the completed
 * step's projected and extended velocity (US), as the old census does when encoded after a completed frame
 * (uniform-mixed-dynamic.ts l.2353-2355; uniform-narrow-band-flip.ts l.2204). Both engines run it.
 *
 * Kernels (brickResidencyKernels):
 * - resSeed is pageSeedOf: a workgroup a C page of C_ALL, its 8 regions of 16 cells (4^3 tiles), lanes over the
 *   tiles. Per region and stage it writes a record (two seed-mask words, then the -xyz and +xyz radii in tiles, one
 *   byte an axis, lrRadii) into the page's census scratch (CENSUS.RES).
 *   - The flow is the old envelope's: sampledFlow (SAMPLE_REACH 2.5, shift 0) over the region grown by
 *     departureTravel (H.CENSUS_REACH, the old census[18]; no gravity term, steps 1) plus RESIDENCY_MARGIN_TILES
 *     (5) tiles.
 *   - Particle travel joins stages >= 2 (samples = held && stage >= 2), each stage with its RESIDENCY_STAGES margin.
 * - resRequest is lrAsk over the destinations of the brick storage: F bricks (8 cells, 2^3 tiles) and C pages
 *   (32 cells, 8^3 tiles). A destination is kept when some stage's record meets it (lrStageMeets, verbatim).
 *   - Present: its keep mark is set (MARK_F, MARK_C), checked before the atomicOr.
 *   - Absent: its directory word goes NONE -> CLAIM, and the one winner appends its key with KEY_CLAIMED for
 *     setClaimPages and setClaimBricks.
 *   - cfg.z = 0: pass 0 takes pages first; a brick in an absent page takes the page. cfg.z = 1: pass 1, after
 *     setClaimPages, takes the bricks of the pages that pass 0 entered (and only those).
 *   - Candidates come from each record's own box, so the work scales with the sources and their reach, never with
 *     the lattice.
 * - Leaving follows the old rule: a block no record reaches retires at this commit (idleSteps 1, no hysteresis;
 *   uniform-lattice-address.ts l.931, uniform-page-pool.ts l.1105). setLeave and setLeavePages run after the
 *   claims, at the head.
 *
 * Seeds (pageSeedOf, at M2: nothing is authored and there are no sources):
 * - a tile is considered when it is near (classify's near, CENSUS.NEAR), band (WANT) or a sample (PSUPPORT);
 * - ownership: an h tile now (width 1) or band;
 * - transport and pressure: wet (the census's NC_WET) or sample;
 * - surface: crossing (NC_CROSSES, the physical flag) or sample;
 * - particles: sample.
 * The particle support and travel are lane F's DEPB hook (PSUPPORT, pJoin: the old transferRefine prediction,
 * uniform-particle-brick.wgsl.ts l.33-38, 240-262). resSeed reads it and zeroes it.
 *
 * Deviations, all stated:
 * - All-h has no old counterpart. The old engine has no all-h mode: every tile is h by construction there, and a
 *   verbatim width-1 ownership seed would keep every brick. All-h ownership seeds are this census's band tiles
 *   (WANT) alone, and every stage is requested at F granularity (a brick is the h storage).
 * - Tiered: pages are requested at page granularity for every stage (the 4h storage, verbatim). Bricks are
 *   requested for the ownership and particle stages only (the h storage of h tiles and the particle bins).
 *   censusDecide also keeps a band tile's present 27-brick ring (the brick storage's h ring, which censusWant
 *   checks).
 * - Destinations are bricks and pages, not 32^3 blocks. The old blocks keep cells resident beyond the footprints,
 *   and old stages may read or propagate into them (extension sweeps, level-set dilation), while our owner readers
 *   return far or air there. If a dam gate row moves in a way that traces to such a cell, revisit the granularity,
 *   not the gate.
 * - No closure-cap or window failure (UP_FATAL_CLOSURE_CAP, UP_FATAL_EDGE): there is no window to outgrow. A
 *   destination off the brick range is the lattice's wall.
 * - Only resident pages have census words, so only they seed. A sample's support in an absent page is latched by
 *   F's hook (FAIL.REACH).
 * - sampledFlow's rangeKeys scans C_ALL when a box spans over 64 pages: this is the old rule, and its cost grows
 *   with C_ALL for a fast region.
 */
import { BrickLayout } from "./brick-abi";
import { CFG, type BrickBinding } from "./brick-kernel";
import { CENSUS, CENSUS_FORMULAS, CENSUS_HOOKS, CENSUS_OWNER, censusKernel } from "./brick-census.wgsl";
import { RESIDENCY_STAGES, RESIDENCY_STENCIL_TILES, localResidencyWGSL } from "../uniform/uniform-local-residency";

/** The stages the head requests (RESIDENCY_STAGES without legacy, which the staged policy never seeds). */
export const RES_STAGES = 5;
/** The old RESIDENCY_MARGIN_TILES (uniform-mixed-dynamic.ts l.120). */
export const RESIDENCY_MARGIN_TILES = RESIDENCY_STENCIL_TILES;
if (RESIDENCY_STAGES.slice(0, RES_STAGES).map(s => s.name).join() !== "ownership,transport,pressure,surface,particles") throw new Error("Residency: RESIDENCY_STAGES changed");
/** Census scratch words of a page's residency records: 4 a (region, stage), region-major. */
export const RES_RECORD_WORDS = 4 * 8 * RES_STAGES;
if (CENSUS.RES + RES_RECORD_WORDS > CENSUS.STRIDE) throw new Error("Residency: the census stride holds no records");

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const dirRW: BrickBinding = { name: "dir", kind: "rw", type: "array<atomic<u32>>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const censusRW: BrickBinding = { name: "census", kind: "rw", type: "array<u32>" };
const censusRO: BrickBinding = { name: "census", kind: "ro", type: "array<u32>" };

export function brickResidencyKernels(L: BrickLayout, opt: { readonly tiered?: boolean } = {}) {
  const tiered = opt.tiered === true;
  const margins = RESIDENCY_STAGES.slice(0, RES_STAGES).map(s => `${s.margin}.0`).join(",");
  const pages = L.lattice.map(n => Math.ceil((Math.floor(n / 8) + 1) / 4));

  // resSeed: pageSeedOf (l.2097) for the page's 8 regions.
  const resSeed = censusKernel(L, "resSeed", [CFG, setRW, dirRO, velRO, phiRO, censusRW], CENSUS_OWNER + CENSUS_HOOKS + CENSUS_FORMULAS + localResidencyWGSL + /* wgsl */`
const RS_STAGES:u32=${RES_STAGES}u;
const RS_MARGIN_TILES:f32=${RESIDENCY_MARGIN_TILES}.0;
var<workgroup> rsMask:array<atomic<u32>,${2 * 8 * RES_STAGES}>;
var<workgroup> rsTravel:array<atomic<u32>,48>;
var<workgroup> rsFound:atomic<u32>;
var<workgroup> rsSupport:array<u32,64>;
var<workgroup> rsGo:u32;
// departureTravel (l.1763): one step, so no gravity term.
fn rsDepartureTravel()->f32{return min(bitcast<f32>(atomicLoad(&nbSet[NB_H_CENSUS_REACH]))+0.0,1.0e6);}
@compute @workgroup_size(64) fn resSeed(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);if(lane==0u){rsGo=select(0u,1u,g<nbListCount(NB_LIST_C_ALL));}if(workgroupUniformLoad(&rsGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);let o=NC_S*cs;
  for(var j=lane;j<${2 * 8 * RES_STAGES}u;j+=64u){atomicStore(&rsMask[j],0u);}
  if(lane<48u){atomicStore(&rsTravel[lane],0u);}
  if(lane==0u){atomicStore(&rsFound,0u);}
  // The particle support (PSUPPORT), lane j the page's child brick j: its tiles' bits, and its travel joined into its
  // region's (the old region words: the travel of every support block that meets the region). Read, then zeroed.
  let fs=nbChild(cs,lane);var support=0u;
  if(fs<NB_SEAM){
    let at=NB_S_PSUPPORT+NB_PSUPPORT_WORDS*fs;
    support=atomicLoad(&nbSet[at])&0xffu;
    if(support!=0u){
      let r=((lane>>1u)&1u)|(((lane>>3u)&1u)<<1u)|(((lane>>5u)&1u)<<2u);
      for(var k=0u;k<6u;k++){let v=atomicLoad(&nbSet[at+1u+k]);if(v!=0u){atomicMax(&rsTravel[6u*r+k],v);}}
      for(var k=0u;k<7u;k++){atomicStore(&nbSet[at+k],0u);}
    }
  }
  rsSupport[lane]=support;
  workgroupBarrier();
  for(var j=0u;j<8u;j++){
    let i=lane+64u*j;let t=xTileOf(cs,i);
    if(!nbTileInLattice(t)){continue;}
    let l=vec3u(i&7u,(i>>3u)&7u,i>>6u);
    let sample=((rsSupport[(l.x>>1u)|((l.y>>1u)<<2u)|((l.z>>1u)<<4u)]>>((l.x&1u)|((l.y&1u)<<1u)|((l.z&1u)<<2u)))&1u)!=0u;
    let near=census[o+${CENSUS.NEAR}u+i]!=0u;
    let band=((atomicLoad(&nbSet[NB_S_WANT+16u*cs+(i>>5u)])>>(i&31u))&1u)!=0u;
    if(!near&&!band&&!sample){continue;}
    let fin=ncFin(cs,i);let wet=(fin&NC_WET)!=0u;let crossing=(fin&NC_CROSSES)!=0u;
    let width1=${tiered ? "nbTileH(cs,i)" : "false"};
    let seeds=array<bool,${RES_STAGES}>(width1||band,wet||sample,wet||sample,crossing||sample,sample);
    let r=(l.x>>2u)|((l.y>>2u)<<1u)|((l.z>>2u)<<2u);let q=(l.x&3u)|((l.y&3u)<<2u)|((l.z&3u)<<4u);
    atomicOr(&rsFound,1u<<r);
    for(var s=0u;s<RS_STAGES;s++){if(seeds[s]){atomicOr(&rsMask[2u*(RS_STAGES*r+s)+(q>>5u)],1u<<(q&31u));}}
  }
  workgroupBarrier();
  if(lane>=8u){return;}
  // Lane r: region r's flow and records. The global envelope only locates local extrema; it never dilates seeds.
  let r=lane;
  let origin=vec3f(32*nbKeyCoord(nbCRec(cs,NB_REC_KEY))+16*vec3i(i32(r&1u),i32((r>>1u)&1u),i32(r>>2u)));
  let envelope=rsDepartureTravel()+4.0*RS_MARGIN_TILES;
  var flow=NcFlow(vec3f(0),vec3f(0));
  if(((atomicLoad(&rsFound)>>r)&1u)!=0u){flow=ncSampledFlow(origin-vec3f(envelope),origin+vec3f(16.0+envelope),vec3f(0.0));}
  var pl=vec3f(0);var ph=vec3f(0);
  for(var a=0u;a<3u;a++){pl[a]=bitcast<f32>(atomicLoad(&rsTravel[6u*r+a]));ph[a]=bitcast<f32>(atomicLoad(&rsTravel[6u*r+3u+a]));}
  let margins=array<f32,${RES_STAGES}>(${margins});
  let scale=vec3f(ncDt()/ncH());
  var radius=0u;
  for(var stage=0u;stage<RS_STAGES;stage++){
    let m=vec2u(atomicLoad(&rsMask[2u*(RS_STAGES*r+stage)]),atomicLoad(&rsMask[2u*(RS_STAGES*r+stage)+1u]));
    var radii=vec2u(0u);
    if((m.x|m.y)!=0u&&stage!=0u){
      // Ballistic travel belongs to particle-created fluid, not every air tile in the same region.
      let samples=stage>=2u;
      radii=lrRadii(flow.low,flow.high,scale,select(vec3f(0),pl,samples),select(vec3f(0),ph,samples),margins[stage]);
      let e=max(lrUnpack(radii.x),lrUnpack(radii.y));radius=max(radius,u32(max(e.x,max(e.y,e.z))));
    }
    let at=o+${CENSUS.RES}u+4u*(RS_STAGES*r+stage);
    census[at]=m.x;census[at+1u]=m.y;census[at+2u]=radii.x;census[at+3u]=radii.y;
  }
  if(radius!=0u){atomicMax(&nbSet[NB_H_RES_RADIUS],radius);}
}`);

  // resRequest: lrAsk (l.140) for the page's 8 regions, lanes over each region's candidate destinations.
  const fLevel = tiered ? [4, 0] : [1, 4, 0, 2, 3];
  const meets = (stages: number[], low: string, high: string) =>
    stages.map(s => `(((rqRec[RQ_STAGES*r+${s}u].x|rqRec[RQ_STAGES*r+${s}u].y)!=0u)&&lrStageMeets(rqRec[RQ_STAGES*r+${s}u],${s}u,${low},${high}))`).join("||");
  const box = (stages: number[], shift: number) => stages.map(s => `
      {let rec=rqRec[RQ_STAGES*r+${s}u];if((rec.x|rec.y)!=0u){a=min(a,(R-lrUnpack(rec.z))>>vec3u(${shift}u));b=max(b,(R+vec3i(3)+lrUnpack(rec.w))>>vec3u(${shift}u));}}`).join("");
  const resRequest = censusKernel(L, "resRequest", [CFG, setRW, dirRW, censusRO], localResidencyWGSL + /* wgsl */`
const NC_S:u32=${CENSUS.STRIDE}u;
const RQ_STAGES:u32=${RES_STAGES}u;
const RQ_PAGES:vec3i=vec3i(${pages.join(",")});
var<workgroup> rqRec:array<vec4u,${8 * RES_STAGES}>;
var<workgroup> rqGo:u32;
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
@compute @workgroup_size(64) fn resRequest(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);let rpass=cfg.z;
  if(lane==0u){rqGo=select(0u,1u,g<nbListCount(NB_LIST_C_ALL)&&(rpass==0u||nbListCount(NB_LIST_C_ENTER)!=0u));}
  if(workgroupUniformLoad(&rqGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);let o=NC_S*cs;
  if(lane<${8 * RES_STAGES}u){let at=o+${CENSUS.RES}u+4u*lane;rqRec[lane]=vec4u(census[at],census[at+1u],census[at+2u],census[at+3u]);}
  workgroupBarrier();
  let P=8*nbKeyCoord(nbCRec(cs,NB_REC_KEY));
  for(var r=0u;r<8u;r++){
    // R: the region's low tile; records are relative to it (lrAsk's origin), destinations as tile boxes.
    let R=P+4*vec3i(i32(r&1u),i32((r>>1u)&1u),i32(r>>2u));
    {
      var a=vec3i(1<<20);var b=vec3i(-(1<<20));${box(fLevel, 1)}
      a=max(a,vec3i(0));b=min(b,NB_BRICKS-vec3i(1));
      if(all(a<=b)){
        let ext=vec3u(b-a+vec3i(1));let n=ext.x*ext.y*ext.z;
        for(var j=lane;j<n;j+=64u){
          let q=a+vec3i(vec3u(j%ext.x,(j/ext.x)%ext.y,j/(ext.x*ext.y)));
          if(rqSettled(q,rpass)){continue;}
          let low=2*q-R;let high=low+vec3i(1);
          if(${meets(fLevel, "low", "high")}){rqBrick(q,rpass);}
        }
      }
    }
    ${tiered ? `if(rpass==0u){
      var a=vec3i(1<<20);var b=vec3i(-(1<<20));${box([0, 1, 2, 3, 4], 3)}
      a=max(a,vec3i(0));b=min(b,RQ_PAGES-vec3i(1));
      if(all(a<=b)){
        let ext=vec3u(b-a+vec3i(1));let n=ext.x*ext.y*ext.z;
        for(var j=lane;j<n;j+=64u){
          let p=a+vec3i(vec3u(j%ext.x,(j/ext.x)%ext.y,j/(ext.x*ext.y)));
          if(rqPageSettled(p)){continue;}
          let low=8*p-R;let high=low+vec3i(7);
          if(${meets([1, 4, 0, 2, 3], "low", "high")}){rqPage(p);}
        }
      }
    }` : ""}
  }
}`);

  return { resSeed, resRequest };
}
