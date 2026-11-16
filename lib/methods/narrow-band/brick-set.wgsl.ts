/**
 * The brick-set kernels of lane E (brick-abi.md §7): setMark, setUpdate (as
 * setClaimPages, setClaimBricks, setLeave, setLeavePages, setRecords and
 * setInit, one dispatch each, for the grid-wide barriers between their
 * phases), setLists, setArgs and stepEnd, plus bootPhi for the initial liquid.
 *
 * Every dispatch is sized from the lists, the candidate count or the pool
 * capacities, never from the lattice. Kernels that hold a workgroup barrier
 * do not return early: work is gated by an `live` flag so every barrier
 * stays in uniform control flow.
 */
import { BrickLayout, LIST, POOLS, brickAbiWGSL } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, declareBindings } from "./brick-kernel";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import type { BrickStages } from "./brick-stages";

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const setRO: BrickBinding = { name: "set", kind: "ro", type: "array<u32>" };
const dirRW: BrickBinding = { name: "dir", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const argsRW: BrickBinding = { name: "args", kind: "rw", type: "array<u32>" };
const velRW: BrickBinding = { name: "vel", kind: "rw", type: "array<f32>" };
const phiRW: BrickBinding = { name: "phi", kind: "rw", type: "array<u32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const presRW: BrickBinding = { name: "pres", kind: "rw", type: "array<f32>" };
const bootRO: BrickBinding = { name: "boot", kind: "ro", type: "array<f32>" };

/** The lists setLists rebuilds every step (all but the churn lists setUpdate writes). */
export const REBUILT_LISTS: readonly number[] = Object.values(LIST).filter(k => k !== LIST.F_ENTER && k !== LIST.F_LEAVE && k !== LIST.C_ENTER && k !== LIST.C_LEAVE);
/** Workgroups of the grid-stride candidate and churn kernels: enough to fill the GPU, bounded by the pool. */
export const churnGroups = (cap: number, lanes: number) => Math.max(1, Math.min(256, Math.ceil(cap / lanes)));

const kernel = (L: BrickLayout, label: string, bindings: readonly BrickBinding[], body: string): BrickKernel => {
  const set = bindings.find(b => b.name === "set"), dir = bindings.find(b => b.name === "dir");
  const code = [
    brickAbiWGSL(L, { ...(set ? { set: { name: "nbSet", atomic: set.kind === "rw" } } : {}), ...(dir ? { dir: { name: "dir", atomic: dir.kind === "rw" } } : {}) }),
    declareBindings(bindings),
    body,
  ].join("\n");
  return { label, entry: label, code, bindings };
};

/** Free-stack pop and push, list append, and the directory claim (setUpdate's helpers). */
const SET_HELPERS = /* wgsl */`
fn nbAppend(k:u32,slot:u32){
  let e=atomicAdd(&nbSet[NB_H_LIST_COUNT+k],1u);
  if(e<NB_S_LIST_CAP[k]){atomicStore(&nbSet[NB_S_LIST[k]+e],slot);}else{nbFail(NB_FAIL_LIST,k);}
}
// A pop that underflows adds 1 back, counts the refusal and latches the kind's POOL bit with the key.
fn nbPop(top:u32,free:u32,cap:u32,high:u32,refused:u32,fail:u32,key:u32)->u32{
  let t=atomicSub(&nbSet[top],1u);
  if(t==0u||t>cap){atomicAdd(&nbSet[top],1u);atomicAdd(&nbSet[refused],1u);nbFail(fail,key);return NB_NONE;}
  let s=atomicLoad(&nbSet[free+t-1u]);atomicMax(&nbSet[high],s+1u);return s;
}
fn nbPush(top:u32,free:u32,slot:u32){let t=atomicAdd(&nbSet[top],1u);atomicStore(&nbSet[free+t],slot);}
// NONE -> CLAIM by compare-exchange (weak: retried while the word is still NONE). True for the one winner.
fn nbClaim(i:u32)->bool{
  loop{
    let r=atomicCompareExchangeWeak(&dir[i],NB_NONE,NB_CLAIM);
    if(r.exchanged){return true;}
    if(r.old_value!=NB_NONE){return false;}
  }
}
// The C-local tile bits of brick-local bl (8 tiles, 2^3 a brick) in CMASK.
fn nbTileMask(cs:u32,bl:u32,set_:bool){
  let b=vec3u(bl&3u,(bl>>2u)&3u,bl>>4u);
  for(var j=0u;j<8u;j++){
    let t=2u*b+vec3u(j&1u,(j>>1u)&1u,j>>2u);let i=t.x|(t.y<<3u)|(t.z<<6u);
    let w=NB_D_CMASK+16u*cs+(i>>5u);let bit=1u<<(i&31u);
    if(set_){atomicOr(&dir[w],bit);}else{atomicAnd(&dir[w],~bit);}
  }
}
`;

/** The brick set's options. `tiered`: the tier census and transfer run (brick-abi.md §2.5); F bricks are kept by
 * core, ring and bin reach, 4h liquid keeps C pages, and bricks entered in a step start 4h. Off: every brick is h. */
export interface BrickSetOptions {
  readonly tiered?: boolean;
  /** The stage set's tile heat (BrickStages.activity): setMark's heat criterion of the ACTIVE tiles. */
  readonly activity?: BrickStages["activity"];
}

export function brickSetKernels(L: BrickLayout, opt: BrickSetOptions = {}) {
  const k = (label: string, bindings: readonly BrickBinding[], body: string) => kernel(L, label, bindings, body);
  const tiered = opt.tiered === true;

  // setMark: one workgroup a brick of F_ALL (G1), in part 2 after F's move and rebin. It writes:
  // - the brick's class word: LIQUID (an h-owned φ < 0 vertex: the brick holds pressure rows), SURFACE, PARTICLES, SKIRT;
  // - the active class (brick-abi.md §5.5, ACT): the ACTIVE tiles of this step into its own word, and NOW into every
  //   resident brick within ACT_DILATION tiles of one (a lane an offset of the 5^3 about it, at tile precision);
  // - the residency check below; the bound word clears.
  // Residency itself is the head's (brick-residency.wgsl.ts, brick-abi.md §5.4): setMark keeps and requests nothing.
  // Tiered (§2.5): the scan reads h-owned tiles only (a 4h tile's F cells hold no state); a 4h tile's φ is its C
  // corners' (the tier authority's interpolant, whose minimum over the tile is a corner's).
  const act = opt.activity;
  const heat = act ? act.heat.map(w => `${act.buffer}[${act.tile}u+${act.words}u*(8u*slot+t)+${w}u]!=0u`).join("||") : "false";
  const markBindings: BrickBinding[] = [CFG, setRW, dirRW, phiRO, ...(tiered ? [velRO] : []), ...(act ? [{ name: act.buffer, kind: "ro", type: "array<u32>" } as BrickBinding] : [])];
  const setMark = k("setMark", markBindings, (tiered ? brickOwnerWGSL({ phi: "phi", vel: "vel" }) : "") + /* wgsl */`
var<workgroup> wNeg:atomic<u32>;
var<workgroup> wPos:atomic<u32>;
var<workgroup> wTiles:atomic<u32>;
var<workgroup> wRes:atomic<u32>;
var<workgroup> wSeed:atomic<u32>;
var<workgroup> wNb:array<u32,27>;
@compute @workgroup_size(64) fn setMark(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){atomicStore(&wNeg,0u);atomicStore(&wPos,0u);atomicStore(&wTiles,0u);atomicStore(&wRes,0u);atomicStore(&wSeed,0u);}
  let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
  var slot=0u;if(live){slot=nbListSlot(NB_LIST_F_ALL,g);}
  // The brick's 27 neighbour words (F record words 0-26), for the residency check below.
  if(live&&lid<27u){wNb[lid]=atomicLoad(&dir[NB_D_FREC+32u*slot+lid]);}
  workgroupBarrier();
  var o=vec3i(0);
  if(live){
    o=nbFOrigin(slot);
    let tiles=${tiered ? "atomicLoad(&dir[NB_D_FREC+32u*slot+NB_REC_MASK])&0xffu" : "0xffu"};
    var neg=0u;var pos=0u;var wet=0u;
    for(var j=0u;j<8u;j++){
      let l=lid+64u*j;let c=nbLocalCoord(l);
      if(any(o+c>NB_LATTICE)){continue;}
      let t=u32((c.x>>2u)|((c.y>>2u)<<1u)|((c.z>>2u)<<2u));
      if(((tiles>>t)&1u)==0u){continue;}
      let p=bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*slot+l]);
      if(p<0.0){neg++;wet|=1u<<t;}else{pos=1u;}
      // The residency check (brick-abi.md §5.4, §8): a liquid vertex's 1-ring (every vertex of the cells it bounds)
      // is resident. φ is this step's advected level set on the directory the head committed, so a miss is a reach
      // defect of this step's head: FAIL.REACH with the absent brick's key, never read as air. Only the brick's face
      // vertices reach a neighbour.
      if(p<0.0&&(any(c==vec3i(0))||any(c==vec3i(7)))){
        for(var n=0u;n<27u;n++){
          let d=nbNeighbourOffset(n);
          if(any((d<vec3i(0))&(c!=vec3i(0)))||any((d>vec3i(0))&(c!=vec3i(7)))||n==NB_REC_SELF){continue;}
          let w=o+c+d;let nb=(o>>vec3u(3u))+d;
          if(any(w<vec3i(0))||any(w>NB_LATTICE)||!nbBrickInRange(nb)){continue;}
          if(wNb[n]>=NB_SEAM){nbFail(NB_FAIL_REACH,nbKey(nb));}
        }
      }
    }
    if(neg>0u){atomicAdd(&wNeg,neg);}
    if(pos!=0u){atomicOr(&wPos,1u);}
    // ACT.RES's liquid tiles (an h-owned φ < 0 vertex; tiered they are h-owned tiles, listed below).
    if(wet!=0u){atomicOr(&wRes,wet);}
    // The ACTIVE tiles (brick-abi.md §5.5): lane t < 8 tests tile t's particles, heat${tiered ? ", h ownership and 4h corners" : ""}.
    if(lid<8u){
      let t=lid;var on=(nbSetWord(NB_S_TILE_OCC+slot)>>t)&1u;
      // The residency's brick sources (brick-residency.wgsl.ts): ACT.RES, the h-owned${tiered ? "" : " (liquid)"} tiles and the tiles
      // with particles; ACT.SEED, the tiles with heat or the census's trigger (the seeds' sources). A tile ACTIVE only
      // by its 4h corners is 4h storage, which the residency requests by page (resMark's page sources), not by brick.
      if(on!=0u${tiered ? "||((tiles>>t)&1u)!=0u" : ""}){on=1u;atomicOr(&wRes,1u<<t);}
      if(((nbSetWord(NB_S_TRIG+slot)>>t)&1u)!=0u||${heat}){on=1u;atomicOr(&wSeed,1u<<t);}
      ${tiered ? `if(on==0u){
        let T=(o>>vec3u(2u))+vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u));
        for(var q=0u;q<8u;q++){if(nbCVertex(T+vec3i(vec3u(q&1u,(q>>1u)&1u,q>>2u)))<0.0){on=1u;}}
      }` : ""}
      if(on!=0u){wet|=1u<<t;}
    }
    if(wet!=0u){atomicOr(&wTiles,wet);}
  }
  workgroupBarrier();
  let m=atomicLoad(&wTiles);
  // NOW on the bricks within ACT_DILATION tiles: offset d (a lane each of the 5^3) is covered when an ACTIVE tile t
  // (brick-local tile coordinate in {0,1}^3) lies within the dilation of the brick's tiles on every axis,
  // 2d - ACT_DILATION <= t <= 2d + 1 + ACT_DILATION (with ACT_DILATION 3: t - 2 <= d <= t + 1).
  if(live&&m!=0u){
    let b=o>>vec3u(3u);
    for(var e=lid;e<125u;e+=64u){
      let d=vec3i(i32(e%5u),i32((e/5u)%5u),i32(e/25u))-vec3i(2);
      var hit=false;
      for(var t=0u;t<8u;t++){
        if(((m>>t)&1u)==0u){continue;}
        let tt=vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u));
        if(all(2*d-i32(NB_ACT_DILATION)<=tt)&&all(tt<=2*d+1+i32(NB_ACT_DILATION))){hit=true;}
      }
      if(!hit){continue;}
      let nb=b+d;if(!nbBrickInRange(nb)){continue;}
      let s2=nbFSlotOfBrick(nb);
      if(s2<NB_SEAM){atomicOr(&nbSet[NB_S_ACT+s2],NB_ACT_NOW);}
    }
  }
  if(!live||lid!=0u){return;}
  // ACT.RES (bits 10..17) and ACT.SEED (bits 18..25): the residency's brick sources.
  atomicOr(&nbSet[NB_S_ACT+slot],m|(atomicLoad(&wRes)<<10u)|(atomicLoad(&wSeed)<<18u));
  var cls=0u;
  if(atomicLoad(&wNeg)>0u){cls|=NB_CLASS_LIQUID;}
  if(atomicLoad(&wNeg)>0u&&atomicLoad(&wPos)!=0u){cls|=NB_CLASS_SURFACE;}
  if(nbSetWord(NB_S_BIN_COUNT+slot)>0u){cls|=NB_CLASS_PARTICLES;}
  if(any(o>=NB_LATTICE)){cls|=NB_CLASS_SKIRT;}
  atomicStore(&dir[NB_D_FREC+32u*slot+NB_REC_CLASS],cls);
  atomicStore(&nbSet[NB_S_BOUND_F+slot],0u);
}`);

  // setMarkPages (tiered only): one workgroup a page of C_ALL (G1), a lane 8 tiles. The count of its 4h-owned tiles
  // with a liquid corner (a C vertex phi < 0) is the C record's LIQUID_TILES (C_LIQ). It keeps and requests nothing
  // (the head's residency does).
  const setMarkPages = k("setMarkPages", [CFG, setRW, dirRW, phiRO, velRO], brickOwnerWGSL({ phi: "phi", vel: "vel" }) + /* wgsl */`
var<workgroup> wN:atomic<u32>;
@compute @workgroup_size(64) fn setMarkPages(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){atomicStore(&wN,0u);}
  workgroupBarrier();
  let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_C_ALL);
  var cs=0u;
  if(live){
    cs=nbListSlot(NB_LIST_C_ALL,g);let P=nbKeyCoord(atomicLoad(&dir[NB_D_CREC+32u*cs+NB_REC_KEY]));
    var n=0u;
    for(var j=0u;j<8u;j++){
      let i=lid+64u*j;let t=vec3u(i&7u,(i>>3u)&7u,i>>6u);let T=8*P+vec3i(t);
      if(!nbTileInLattice(T)||nbTileH(cs,i)){continue;}
      var m=NB_PHI_AIR;
      for(var q=0u;q<8u;q++){m=min(m,nbCVertex(T+vec3i(vec3u(q&1u,(q>>1u)&1u,q>>2u))));}
      if(m<0.0){n++;}
    }
    if(n>0u){atomicAdd(&wN,n);}
  }
  workgroupBarrier();
  if(live&&lid==0u){atomicStore(&dir[NB_D_CREC+32u*cs+NB_REC_C_LIQUID_TILES],atomicLoad(&wN));}
}`);

  // setClaimPages: a candidate's page takes a C slot (NONE -> CLAIM -> cslot), once. A KEY_CLAIMED page key's word
  // is already CLAIM (resMark's winner); a KEY_CLAIMED brick key's page is resident.
  const setClaimPages = k("setClaimPages", [CFG, setRW, dirRW], SET_HELPERS + /* wgsl */`
@compute @workgroup_size(64) fn setClaimPages(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) nw:vec3u){
  let n=min(atomicLoad(&nbSet[NB_H_N_CAND]),NB_FCAP);
  for(var e=gid.x;e<n;e+=64u*nw.x){
    let key=atomicLoad(&nbSet[NB_S_CAND+e]);let p=nbKeyCoord(key)>>vec3u(2u);
    let claimed=(key&NB_KEY_CLAIMED)!=0u;let page=(key&NB_KEY_PAGE)!=0u;
    if(claimed&&!page){continue;}
    let pi=NB_D_PAGE+nbPageIndex(p);
    if(!claimed&&!nbClaim(pi)){continue;}
    let cs=nbPop(NB_H_TOP_C,NB_S_FREE_C,NB_CCAP,NB_H_HIGH_C,NB_H_REFUSED_C,NB_FAIL_POOL_C,key);
    if(cs==NB_NONE){atomicStore(&dir[pi],NB_NONE);continue;}
    for(var j=0u;j<64u;j++){atomicStore(&dir[NB_D_CHILD+64u*cs+j],NB_NONE);}
    // All-h: every tile of a resident page is h-owned, so a tile whose brick is not resident is an h region with
    // no slot (NB_PHI_FAR, faces 0: air) and C is never an authority. Tiered: CMASK is the bricks' (§2.5).
    for(var j=0u;j<16u;j++){atomicStore(&dir[NB_D_CMASK+16u*cs+j],${tiered ? "0u" : "0xffffffffu"});}
    for(var j=0u;j<16u;j++){atomicStore(&nbSet[NB_S_WANT+16u*cs+j],0u);}
    for(var j=0u;j<16u;j++){atomicStore(&nbSet[NB_S_AIR_GATE+16u*cs+j],0u);}
    let r=NB_D_CREC+32u*cs;
    for(var j=0u;j<27u;j++){atomicStore(&dir[r+j],NB_NONE);}
    atomicStore(&dir[r+NB_REC_C_CHILDREN],0u);atomicStore(&dir[r+NB_REC_C_LIQUID_TILES],0u);
    atomicStore(&dir[r+NB_REC_KEY],nbKey(p));atomicStore(&dir[r+NB_REC_CUT],NB_NONE);
    atomicStore(&dir[r+NB_REC_CLASS],NB_CLASS_ENTERING);
    atomicStore(&dir[pi],cs);
    nbAppend(NB_LIST_C_ENTER,cs);
  }
}`);

  // setClaimBricks: a candidate brick takes an F slot under its page's C slot, once. A KEY_CLAIMED key's child word
  // is already CLAIM (resMark's winner). An entering brick's particle support (PSUPPORT) starts empty.
  const setClaimBricks = k("setClaimBricks", [CFG, setRW, dirRW], SET_HELPERS + /* wgsl */`
@compute @workgroup_size(64) fn setClaimBricks(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) nw:vec3u){
  let n=min(atomicLoad(&nbSet[NB_H_N_CAND]),NB_FCAP);
  for(var e=gid.x;e<n;e+=64u*nw.x){
    let key=atomicLoad(&nbSet[NB_S_CAND+e]);let b=nbKeyCoord(key);
    if((key&NB_KEY_PAGE)!=0u){continue;}
    let cs=atomicLoad(&dir[NB_D_PAGE+nbPageIndex(b>>vec3u(2u))]);
    if(cs>=NB_SEAM){continue;}
    let bl=nbBrickLocal(b);let ci=NB_D_CHILD+64u*cs+bl;
    if((key&NB_KEY_CLAIMED)==0u&&!nbClaim(ci)){continue;}
    let fs=nbPop(NB_H_TOP_F,NB_S_FREE_F,NB_FCAP,NB_H_HIGH_F,NB_H_REFUSED_F,NB_FAIL_POOL_F,key);
    if(fs==NB_NONE){atomicStore(&dir[ci],NB_NONE);continue;}
    let r=NB_D_FREC+32u*fs;
    for(var j=0u;j<27u;j++){atomicStore(&dir[r+j],NB_NONE);}
    let h=${tiered ? "cfg.z==1u" : "true"};
    atomicStore(&dir[r+NB_REC_PARENT],cs);atomicStore(&dir[r+NB_REC_MASK],select(0u,0xffu,h)|(bl<<8u));
    atomicStore(&dir[r+NB_REC_KEY],key&~NB_KEY_CLAIMED);atomicStore(&dir[r+NB_REC_CUT],NB_NONE);
    for(var j=0u;j<NB_PSUPPORT_WORDS;j++){atomicStore(&nbSet[NB_S_PSUPPORT+NB_PSUPPORT_WORDS*fs+j],0u);}
    atomicStore(&dir[r+NB_REC_CLASS],NB_CLASS_ENTERING|select(0u,NB_CLASS_SKIRT,any(8*b>=NB_LATTICE)));
    atomicStore(&nbSet[NB_S_BIN_COUNT+fs],0u);atomicStore(&nbSet[NB_S_BOUND_F+fs],0u);atomicStore(&nbSet[NB_S_MARK_F+fs],0u);
    atomicStore(&nbSet[NB_S_TILE_OCC+fs],0u);atomicStore(&nbSet[NB_S_TRIG+fs],0u);atomicStore(&nbSet[NB_S_ACT+fs],0u);
    atomicAdd(&dir[NB_D_CREC+32u*cs+NB_REC_C_CHILDREN],1u);
    if(${tiered ? "h" : "false"}){nbTileMask(cs,bl,true);}
    atomicStore(&dir[ci],fs);
    nbAppend(NB_LIST_F_ENTER,fs);
  }
}`);

  // setLeave: a live F slot of last step's F_ALL without a keep bit leaves (G64:
  // a thread an entry). Its directory word clears and its slot goes back on the
  // stack after every pop of the step. Every mark clears for the next step.
  const setLeave = k("setLeave", [CFG, setRW, dirRW], SET_HELPERS + /* wgsl */`
@compute @workgroup_size(64) fn setLeave(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let e=64u*nbGroup(wid)+lid;if(e>=nbListCount(NB_LIST_F_ALL)){return;}
  let s=nbListSlot(NB_LIST_F_ALL,e);
  if(atomicExchange(&nbSet[NB_S_MARK_F+s],0u)!=0u){return;}
  let r=NB_D_FREC+32u*s;let cs=atomicLoad(&dir[r+NB_REC_PARENT]);let bl=(atomicLoad(&dir[r+NB_REC_MASK])>>8u)&63u;
  atomicStore(&dir[NB_D_CHILD+64u*cs+bl],NB_NONE);
  atomicSub(&dir[NB_D_CREC+32u*cs+NB_REC_C_CHILDREN],1u);
  ${tiered ? "nbTileMask(cs,bl,false);" : "// All-h: CMASK stays all-ones (setClaimPages)."}
  atomicStore(&dir[r+NB_REC_CLASS],NB_CLASS_LEAVING);
  nbPush(NB_H_TOP_F,NB_S_FREE_F,s);
  nbAppend(NB_LIST_F_LEAVE,s);
}`);

  // setLeavePages: a live page of last step's C_ALL with no F child left and no keep mark (resMark) leaves.
  const setLeavePages = k("setLeavePages", [CFG, setRW, dirRW], SET_HELPERS + /* wgsl */`
@compute @workgroup_size(64) fn setLeavePages(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let e=64u*nbGroup(wid)+lid;if(e>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);let r=NB_D_CREC+32u*cs;
  let kept=atomicExchange(&nbSet[NB_S_MARK_C+cs],0u)!=0u;
  if(kept||atomicLoad(&dir[r+NB_REC_C_CHILDREN])!=0u){return;}
  // A leaving page that held 4h liquid tiles (setMarkPages' LIQUID_TILES, last step) is a reach defect: no source
  // brick's reach covered liquid (brick-residency.wgsl.ts). FAIL.REACH with its key, never dropped as air.
  if(atomicLoad(&dir[r+NB_REC_C_LIQUID_TILES])!=0u){nbFail(NB_FAIL_REACH,atomicLoad(&dir[r+NB_REC_KEY]));}
  atomicStore(&dir[NB_D_PAGE+nbPageIndex(nbKeyCoord(atomicLoad(&dir[r+NB_REC_KEY])))],NB_NONE);
  atomicStore(&dir[r+NB_REC_CLASS],NB_CLASS_LEAVING);
  nbPush(NB_H_TOP_C,NB_S_FREE_C,cs);
  nbAppend(NB_LIST_C_LEAVE,cs);
}`);

  // setRecords: a workgroup an entering or leaving brick, a lane a neighbour.
  // An entering brick writes its own 27 words from the (final) directory, and
  // each live neighbour's opposite word points back at it; a leaving brick's
  // live neighbours point at NONE. Each (neighbour, word) pair has one writer.
  // Workgroup 0 also clears the counts setLists rebuilds (nothing reads them
  // between here and setLists).
  const setRecords = k("setRecords", [CFG, setRW, dirRW], /* wgsl */`
@compute @workgroup_size(32) fn setRecords(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32,@builtin(num_workgroups) nw:vec3u){
  if(wid.x==0u&&lid==0u){${REBUILT_LISTS.map(j => `atomicStore(&nbSet[NB_H_LIST_COUNT+${j}u],0u);`).join("")}}
  let nE=nbListCount(NB_LIST_F_ENTER);let nL=nbListCount(NB_LIST_F_LEAVE);
  if(lid>=27u){return;}
  for(var e=wid.x;e<nE+nL;e+=nw.x){
    let entering=e<nE;
    var s=0u;if(entering){s=nbListSlot(NB_LIST_F_ENTER,e);}else{s=nbListSlot(NB_LIST_F_LEAVE,e-nE);}
    let b=nbKeyCoord(atomicLoad(&dir[NB_D_FREC+32u*s+NB_REC_KEY]));
    var s2=NB_NONE;
    if(lid==NB_REC_SELF){s2=s;}else{let nb=b+nbNeighbourOffset(lid);if(nbBrickInRange(nb)){s2=nbFSlotOfBrick(nb);}}
    if(entering){atomicStore(&dir[NB_D_FREC+32u*s+lid],s2);}
    if(lid!=NB_REC_SELF&&s2<NB_SEAM){atomicStore(&dir[NB_D_FREC+32u*s2+26u-lid],select(NB_NONE,s,entering));}
  }
}`);

  // setInit: entering bricks' and pages' channels to air (brick-abi.md §2.3).
  const zero = (pool: "vel" | "pres") => POOLS[pool].map(ch => `${pool}[NB_${pool.toUpperCase()}_F_${ch}+at]=0.0;`).join("");
  const zeroC = (pool: "vel" | "pres") => POOLS[pool].map(ch => `${pool}[NB_${pool.toUpperCase()}_C_${ch}+at]=0.0;`).join("");
  const setInit = k("setInit", [CFG, setRO, dirRO, velRW, phiRW, presRW], /* wgsl */`
@compute @workgroup_size(128) fn setInit(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32,@builtin(num_workgroups) nw:vec3u){
  let n=nbListCount(NB_LIST_F_ENTER);
  for(var e=wid.x;e<n;e+=nw.x){
    let s=nbListSlot(NB_LIST_F_ENTER,e);let o=nbFOrigin(s);
    for(var j=0u;j<4u;j++){
      let l=lid+128u*j;let at=NB_STRIDE*s+l;
      ${zero("vel")}
      ${zero("pres")}
      phi[NB_PHI_F_PHI+at]=bitcast<u32>(NB_PHI_AIR_C);phi[NB_PHI_F_PHI1+at]=bitcast<u32>(NB_PHI_AIR_C);
      phi[NB_PHI_F_FLAGS+at]=select(0u,NB_FLAG_OPEN,nbInLattice(o+nbLocalCoord(l)))|NB_FLAG_OWNER_H;
    }
  }
  // Entering C slots: a cell a tile (C local = tile index), open inside the lattice. φ is NB_PHI_AIR_C (16 max h, the
  // old far-air value and the least AIR an old claim takes; the old claim's own AIR is the box's far φ,
  // uniform-page-initialise.ts 5-24, which would make the residency depend on the box). F's entering bricks take it
  // too: nbAirStore's 16 max h holds at h vertices as at 4h ones (uniform-narrow-band-air.wgsl.ts 261, hDt.xyz the
  // fine cell size).
  let nC=nbListCount(NB_LIST_C_ENTER);
  for(var e=wid.x;e<nC;e+=nw.x){
    let s=nbListSlot(NB_LIST_C_ENTER,e);let o=nbCOrigin(s);
    for(var j=0u;j<4u;j++){
      let l=lid+128u*j;let at=NB_STRIDE*s+l;
      ${zeroC("vel")}
      ${zeroC("pres")}
      phi[NB_PHI_C_PHI+at]=bitcast<u32>(NB_PHI_AIR_C);phi[NB_PHI_C_PHI1+at]=bitcast<u32>(NB_PHI_AIR_C);
      phi[NB_PHI_C_FLAGS+at]=select(0u,NB_FLAG_OPEN,nbInLattice(o+4*nbLocalCoord(l)));
    }
  }
}`);

  // bootPhi: the initial liquid's signed distance (in h) on the entering bricks.
  // boot holds cfg.z boxes of 6 floats (lo xyz, hi xyz, in cells); the union is the min. Uncapped, as the old engine's
  // t = 0 φ (initialLiquidVertexPhi: the analytic distance, its empty value the lattice's far φ, NB_PHI_FAR).
  const bootPhi = k("bootPhi", [CFG, setRO, dirRO, phiRW, bootRO], /* wgsl */`
fn sdBox(p:vec3f,lo:vec3f,hi:vec3f)->f32{let q=abs(p-0.5*(lo+hi))-0.5*(hi-lo);return length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);}
@compute @workgroup_size(128) fn bootPhi(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32,@builtin(num_workgroups) nw:vec3u){
  // The box count is the host's, held to the boot binding's boxes (the engine's loop bound, 2026-11-17).
  let nBoxes=min(cfg.z,arrayLength(&boot)/6u);
  let n=nbListCount(NB_LIST_F_ENTER);
  for(var e=wid.x;e<n;e+=nw.x){
    let s=nbListSlot(NB_LIST_F_ENTER,e);let o=nbFOrigin(s);
    for(var j=0u;j<4u;j++){
      let l=lid+128u*j;let p=vec3f(o+nbLocalCoord(l));
      var d=NB_PHI_FAR;
      for(var i=0u;i<nBoxes;i++){d=min(d,sdBox(p,vec3f(boot[6u*i],boot[6u*i+1u],boot[6u*i+2u]),vec3f(boot[6u*i+3u],boot[6u*i+4u],boot[6u*i+5u])));}
      phi[NB_PHI_F_PHI+NB_STRIDE*s+l]=bitcast<u32>(d);phi[NB_PHI_F_PHI1+NB_STRIDE*s+l]=bitcast<u32>(d);
    }
  }
  // The entering pages' C vertices (h vertex 4T), the same distance: a tiered boot commits the census's tiers
  // without a remap (U = 0, and F and C hold one analytic φ, which is the remap's injection at a corner).
  let nC=nbListCount(NB_LIST_C_ENTER);
  for(var e=wid.x;e<nC;e+=nw.x){
    let s=nbListSlot(NB_LIST_C_ENTER,e);let o=nbCOrigin(s);
    for(var j=0u;j<4u;j++){
      let l=lid+128u*j;let p=vec3f(o+4*nbLocalCoord(l));
      var d=NB_PHI_FAR;
      for(var i=0u;i<nBoxes;i++){d=min(d,sdBox(p,vec3f(boot[6u*i],boot[6u*i+1u],boot[6u*i+2u]),vec3f(boot[6u*i+3u],boot[6u*i+4u],boot[6u*i+5u])));}
      phi[NB_PHI_C_PHI+NB_STRIDE*s+l]=bitcast<u32>(d);phi[NB_PHI_C_PHI1+NB_STRIDE*s+l]=bitcast<u32>(d);
    }
  }
}`);

  // setLists: a thread a slot below the high-water mark. A live slot (key not NONE, not leaving) joins its lists from
  // its class word (setMark's; F_LIQ: the bricks with an h-owned liquid vertex) and its active class; a leaving
  // slot's key clears. F_ACT (brick-abi.md §5.5) lists NOW | PREV; at the head (cfg.z = 1) also an entering brick
  // (its first step's part 1 has no class yet), and the word then moves NOW to PREV for this step's setMark. The
  // boot (cfg.z = 2) ORs NOW onto every live brick, so step 1 runs the moved stages on all of F_ALL; its ACTIVE tiles
  // (the boot's class pass, run before the boot's last lists) are kept for step 1's head residency.
  const fLists: [number, string][] = [[LIST.F_LIQ, "LIQUID"], [LIST.F_SURF, "SURFACE"], [LIST.F_PART, "PARTICLES"], [LIST.F_SEAM, "SEAM"], [LIST.F_SOLID, "NEAR_SOLID"], [LIST.F_BODY, "NEAR_BODY"], [LIST.F_COLD, "COLD"], [LIST.F_CHANGED, "CHANGED"]];
  const setLists = k("setLists", [CFG, setRW, dirRW], SET_HELPERS + /* wgsl */`
@compute @workgroup_size(64) fn setLists(@builtin(global_invocation_id) gid:vec3u){
  let i=gid.x;
  if(i<min(atomicLoad(&nbSet[NB_H_HIGH_F]),NB_FCAP)){
    let r=NB_D_FREC+32u*i;
    if(atomicLoad(&dir[r+NB_REC_KEY])!=NB_NONE){
      let cls=atomicLoad(&dir[r+NB_REC_CLASS]);
      if((cls&NB_CLASS_LEAVING)!=0u){atomicStore(&dir[r+NB_REC_KEY],NB_NONE);}
      else{
        let c=cls&~NB_CLASS_ENTERING;atomicStore(&dir[r+NB_REC_CLASS],c);
        nbAppend(NB_LIST_F_ALL,i);
        var a=atomicLoad(&nbSet[NB_S_ACT+i]);
        if(cfg.z==2u){a|=NB_ACT_NOW;atomicStore(&nbSet[NB_S_ACT+i],a);}
        if((a&(NB_ACT_NOW|NB_ACT_PREV))!=0u||(cfg.z==1u&&(cls&NB_CLASS_ENTERING)!=0u)){nbAppend(NB_LIST_F_ACT,i);}
        if(cfg.z==1u){atomicStore(&nbSet[NB_S_ACT+i],select(0u,NB_ACT_PREV,(a&NB_ACT_NOW)!=0u));}
        ${fLists.map(([j, bit]) => `if((c&NB_CLASS_${bit})!=0u){nbAppend(${j}u,i);}`).join("\n        ")}
      }
    }
  }
  if(i<min(atomicLoad(&nbSet[NB_H_HIGH_C]),NB_CCAP)){
    let r=NB_D_CREC+32u*i;
    if(atomicLoad(&dir[r+NB_REC_KEY])!=NB_NONE){
      let cls=atomicLoad(&dir[r+NB_REC_CLASS]);
      if((cls&NB_CLASS_LEAVING)!=0u){atomicStore(&dir[r+NB_REC_KEY],NB_NONE);}
      else{
        atomicStore(&dir[r+NB_REC_CLASS],cls&~NB_CLASS_ENTERING);
        nbAppend(NB_LIST_C_ALL,i);
        if(atomicLoad(&dir[r+NB_REC_C_CHILDREN])!=0u){nbAppend(NB_LIST_C_SEAM,i);}
        if(atomicLoad(&dir[r+NB_REC_C_LIQUID_TILES])!=0u){nbAppend(NB_LIST_C_LIQ,i);}
      }
    }
  }
}`);

  // setArgs: one workgroup. Each list's G1, G8, G64 and X8 (eight workgroups an entry: setCensus's
  // C passes, a workgroup a 64-tile slab of a page) under the fold rule, the FC and FA triples, and the
  // live, churn and peak counters.
  const setArgs = k("setArgs", [CFG, setRW, argsRW], /* wgsl */`
fn fold(at:u32,n:u32){args[at]=min(n,NB_FOLD);args[at+1u]=max(1u,(n+NB_FOLD-1u)/NB_FOLD);args[at+2u]=1u;}
fn count(k:u32)->u32{
  let n=atomicLoad(&nbSet[NB_H_LIST_COUNT+k]);
  if(n>NB_S_LIST_CAP[k]){nbFail(NB_FAIL_LIST,k);return NB_S_LIST_CAP[k];}
  return n;
}
fn triple(at:u32,n:u32){fold(at+NB_ARGS_G1,n);fold(at+NB_ARGS_G8,(n+7u)/8u);fold(at+NB_ARGS_G64,(n+63u)/64u);fold(at+NB_ARGS_X8,8u*n);}
@compute @workgroup_size(64) fn setArgs(@builtin(local_invocation_index) lid:u32){
  if(lid<=NB_LIST_F_ACT){triple(NB_ARGS_PER_LIST*lid,count(lid));}
  if(lid==NB_LIST_F_ACT+2u){let nA=count(NB_LIST_F_ACT)+count(NB_LIST_C_ALL);fold(NB_ARGS_FA_G1,nA);fold(NB_ARGS_FA_G8,(nA+7u)/8u);fold(NB_ARGS_FA_G64,(nA+63u)/64u);}
  if(lid==NB_LIST_F_ACT+1u){
    let nF=count(NB_LIST_F_ALL);let nC=count(NB_LIST_C_ALL);
    fold(NB_ARGS_FC_G1,nF+nC);fold(NB_ARGS_FC_G8,(nF+nC+7u)/8u);fold(NB_ARGS_FC_G64,(nF+nC+63u)/64u);
    atomicStore(&nbSet[NB_H_LIVE_F],nF);atomicStore(&nbSet[NB_H_LIVE_C],nC);
    atomicMax(&nbSet[NB_H_PEAK_F],nF);atomicMax(&nbSet[NB_H_PEAK_C],nC);
    // cfg.z = 1 (part 2): no churn (the head's counts stay for the receipt).
    if(cfg.z==0u){
      atomicStore(&nbSet[NB_H_N_ENTER_F],count(NB_LIST_F_ENTER));atomicStore(&nbSet[NB_H_N_LEAVE_F],count(NB_LIST_F_LEAVE));
      atomicStore(&nbSet[NB_H_N_ENTER_C],count(NB_LIST_C_ENTER));atomicStore(&nbSet[NB_H_N_LEAVE_C],count(NB_LIST_C_LEAVE));
      atomicMax(&nbSet[NB_H_PEAK_CAND],atomicLoad(&nbSet[NB_H_N_CAND]));
    }
  }
}`);

  // stepEnd: one lane. The step counter (cfg.z = 1 at boot: no step), and the
  // churn counts and candidates cleared for the next step.
  const stepEnd = k("stepEnd", [CFG, setRW], /* wgsl */`
@compute @workgroup_size(1) fn stepEnd(){
  if(cfg.z==0u){atomicAdd(&nbSet[NB_H_STEP],1u);}
  atomicStore(&nbSet[NB_H_N_CAND],0u);
  atomicStore(&nbSet[NB_H_LIST_COUNT+NB_LIST_F_ENTER],0u);atomicStore(&nbSet[NB_H_LIST_COUNT+NB_LIST_F_LEAVE],0u);
  atomicStore(&nbSet[NB_H_LIST_COUNT+NB_LIST_C_ENTER],0u);atomicStore(&nbSet[NB_H_LIST_COUNT+NB_LIST_C_LEAVE],0u);
}`);

  return { setMark, setMarkPages, setClaimPages, setClaimBricks, setLeave, setLeavePages, setRecords, setInit, bootPhi, setLists, setArgs, stepEnd };
}

