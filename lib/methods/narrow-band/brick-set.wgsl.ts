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

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const setRO: BrickBinding = { name: "set", kind: "ro", type: "array<u32>" };
const dirRW: BrickBinding = { name: "dir", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const argsRW: BrickBinding = { name: "args", kind: "rw", type: "array<u32>" };
const velRW: BrickBinding = { name: "vel", kind: "rw", type: "array<f32>" };
const phiRW: BrickBinding = { name: "phi", kind: "rw", type: "array<u32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
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

export function brickSetKernels(L: BrickLayout) {
  const k = (label: string, bindings: readonly BrickBinding[], body: string) => kernel(L, label, bindings, body);

  // setMark: one workgroup a brick of F_ALL (G1). The brick's liquid vertices'
  // box (or the whole brick for a non-empty bin), grown by the reach, keeps
  // the live bricks it overlaps and lists the others as candidates.
  const setMark = k("setMark", [CFG, setRW, dirRW, phiRO], /* wgsl */`
var<workgroup> wNeg:atomic<u32>;
var<workgroup> wPos:atomic<u32>;
var<workgroup> wLo:array<atomic<u32>,3>;
var<workgroup> wHi:array<atomic<u32>,3>;
@compute @workgroup_size(64) fn setMark(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){atomicStore(&wNeg,0u);atomicStore(&wPos,0u);for(var a=0u;a<3u;a++){atomicStore(&wLo[a],8u);atomicStore(&wHi[a],0u);}}
  workgroupBarrier();
  let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
  var slot=0u;var o=vec3i(0);var bound=0.0;
  if(live){
    slot=nbListSlot(NB_LIST_F_ALL,g);o=nbFOrigin(slot);
    bound=bitcast<f32>(nbSetWord(NB_S_BOUND_F+slot));
    var neg=0u;var pos=0u;var lo=vec3u(8u);var hi=vec3u(0u);
    for(var j=0u;j<8u;j++){
      let l=lid+64u*j;let c=nbLocalCoord(l);
      if(any(o+c>NB_LATTICE)){continue;}
      let p=bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*slot+l]);
      if(p<0.0){neg++;lo=min(lo,vec3u(c));hi=max(hi,vec3u(c));}else{pos=1u;}
    }
    if(neg>0u){atomicAdd(&wNeg,neg);for(var a=0u;a<3u;a++){atomicMin(&wLo[a],lo[a]);atomicMax(&wHi[a],hi[a]);}}
    if(pos!=0u){atomicOr(&wPos,1u);}
  }
  workgroupBarrier();
  if(!live){return;}
  let neg=atomicLoad(&wNeg);let parts=nbSetWord(NB_S_BIN_COUNT+slot)>0u;
  if(lid==0u){
    var cls=0u;
    if(neg>0u&&atomicLoad(&wPos)!=0u){cls|=NB_CLASS_SURFACE;}
    if(parts){cls|=NB_CLASS_PARTICLES;}
    if(any(o>=NB_LATTICE)){cls|=NB_CLASS_SKIRT;}
    atomicStore(&dir[NB_D_FREC+32u*slot+NB_REC_CLASS],cls);
    atomicStore(&nbSet[NB_S_BOUND_F+slot],0u);
  }
  if(neg==0u&&!parts){return;}
  var lo=vec3f(f32(atomicLoad(&wLo[0])),f32(atomicLoad(&wLo[1])),f32(atomicLoad(&wLo[2])));
  var hi=vec3f(f32(atomicLoad(&wHi[0])),f32(atomicLoad(&wHi[1])),f32(atomicLoad(&wHi[2])));
  if(parts){lo=vec3f(0.0);hi=vec3f(7.0);}
  let r=cfg.x;let dt=nbRingF(r,NB_RING_DT);let h=nbRingF(r,NB_RING_H);
  let gv=vec3f(nbRingF(r,NB_RING_GRAVITY),nbRingF(r,NB_RING_GRAVITY+1u),nbRingF(r,NB_RING_GRAVITY+2u));
  let reach=(bound+length(gv)*dt)*dt/h+NB_REACH_MARGIN;
  let b0=max(vec3i(floor((vec3f(o)+lo-reach)/8.0)),vec3i(0));
  let b1=min(vec3i(floor((vec3f(o)+hi+reach)/8.0)),NB_BRICKS-vec3i(1));
  let ext=b1-b0+vec3i(1);let total=u32(ext.x*ext.y*ext.z);
  for(var i=lid;i<total;i+=64u){
    let nb=b0+vec3i(i32(i%u32(ext.x)),i32((i/u32(ext.x))%u32(ext.y)),i32(i/u32(ext.x*ext.y)));
    let s=nbFSlotOfBrick(nb);
    if(s<NB_SEAM){atomicOr(&nbSet[NB_S_MARK_F+s],1u);}
    else if(s==NB_NONE){
      let e=atomicAdd(&nbSet[NB_H_N_CAND],1u);
      if(e<NB_FCAP){atomicStore(&nbSet[NB_S_CAND+e],nbKey(nb));}else{nbFail(NB_FAIL_LIST,nbKey(nb));}
    }
    else{nbFail(NB_FAIL_UNSUPPORTED,nbKey(nb));}
  }
}`);

  // setClaimPages: a candidate's page takes a C slot (NONE -> CLAIM -> cslot), once.
  const setClaimPages = k("setClaimPages", [CFG, setRW, dirRW], SET_HELPERS + /* wgsl */`
@compute @workgroup_size(64) fn setClaimPages(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) nw:vec3u){
  let n=min(atomicLoad(&nbSet[NB_H_N_CAND]),NB_FCAP);
  for(var e=gid.x;e<n;e+=64u*nw.x){
    let key=atomicLoad(&nbSet[NB_S_CAND+e]);let p=nbKeyCoord(key)>>vec3u(2u);
    let pi=NB_D_PAGE+nbPageIndex(p);
    if(!nbClaim(pi)){continue;}
    let cs=nbPop(NB_H_TOP_C,NB_S_FREE_C,NB_CCAP,NB_H_HIGH_C,NB_H_REFUSED_C,NB_FAIL_POOL_C,key);
    if(cs==NB_NONE){atomicStore(&dir[pi],NB_NONE);continue;}
    for(var j=0u;j<64u;j++){atomicStore(&dir[NB_D_CHILD+64u*cs+j],NB_NONE);}
    for(var j=0u;j<16u;j++){atomicStore(&dir[NB_D_CMASK+16u*cs+j],0u);}
    let r=NB_D_CREC+32u*cs;
    for(var j=0u;j<27u;j++){atomicStore(&dir[r+j],NB_NONE);}
    atomicStore(&dir[r+NB_REC_C_CHILDREN],0u);atomicStore(&dir[r+NB_REC_C_LIQUID_TILES],0u);
    atomicStore(&dir[r+NB_REC_KEY],nbKey(p));atomicStore(&dir[r+NB_REC_CUT],NB_NONE);
    atomicStore(&dir[r+NB_REC_CLASS],NB_CLASS_ENTERING);
    atomicStore(&dir[pi],cs);
    nbAppend(NB_LIST_C_ENTER,cs);
  }
}`);

  // setClaimBricks: a candidate brick takes an F slot under its page's C slot, once.
  const setClaimBricks = k("setClaimBricks", [CFG, setRW, dirRW], SET_HELPERS + /* wgsl */`
@compute @workgroup_size(64) fn setClaimBricks(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) nw:vec3u){
  let n=min(atomicLoad(&nbSet[NB_H_N_CAND]),NB_FCAP);
  for(var e=gid.x;e<n;e+=64u*nw.x){
    let key=atomicLoad(&nbSet[NB_S_CAND+e]);let b=nbKeyCoord(key);
    let cs=atomicLoad(&dir[NB_D_PAGE+nbPageIndex(b>>vec3u(2u))]);
    if(cs>=NB_SEAM){continue;}
    let bl=nbBrickLocal(b);let ci=NB_D_CHILD+64u*cs+bl;
    if(!nbClaim(ci)){continue;}
    let fs=nbPop(NB_H_TOP_F,NB_S_FREE_F,NB_FCAP,NB_H_HIGH_F,NB_H_REFUSED_F,NB_FAIL_POOL_F,key);
    if(fs==NB_NONE){atomicStore(&dir[ci],NB_NONE);continue;}
    let r=NB_D_FREC+32u*fs;
    for(var j=0u;j<27u;j++){atomicStore(&dir[r+j],NB_NONE);}
    atomicStore(&dir[r+NB_REC_PARENT],cs);atomicStore(&dir[r+NB_REC_MASK],0xffu|(bl<<8u));
    atomicStore(&dir[r+NB_REC_KEY],key);atomicStore(&dir[r+NB_REC_CUT],NB_NONE);
    atomicStore(&dir[r+NB_REC_CLASS],NB_CLASS_ENTERING|select(0u,NB_CLASS_SKIRT,any(8*b>=NB_LATTICE)));
    atomicStore(&nbSet[NB_S_BIN_COUNT+fs],0u);atomicStore(&nbSet[NB_S_BOUND_F+fs],0u);atomicStore(&nbSet[NB_S_MARK_F+fs],0u);
    atomicAdd(&dir[NB_D_CREC+32u*cs+NB_REC_C_CHILDREN],1u);
    nbTileMask(cs,bl,true);
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
  nbTileMask(cs,bl,false);
  atomicStore(&dir[r+NB_REC_CLASS],NB_CLASS_LEAVING);
  nbPush(NB_H_TOP_F,NB_S_FREE_F,s);
  nbAppend(NB_LIST_F_LEAVE,s);
}`);

  // setLeavePages: a live page of last step's C_ALL with no F child left leaves (M1: pages are containers).
  const setLeavePages = k("setLeavePages", [CFG, setRW, dirRW], SET_HELPERS + /* wgsl */`
@compute @workgroup_size(64) fn setLeavePages(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let e=64u*nbGroup(wid)+lid;if(e>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);let r=NB_D_CREC+32u*cs;
  atomicStore(&nbSet[NB_S_MARK_C+cs],0u);
  if(atomicLoad(&dir[r+NB_REC_C_CHILDREN])!=0u){return;}
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

  // setInit: entering bricks' channels to air (brick-abi.md §2.3).
  const zero = (pool: "vel" | "pres") => POOLS[pool].map(ch => `${pool}[NB_${pool.toUpperCase()}_F_${ch}+at]=0.0;`).join("");
  const setInit = k("setInit", [CFG, setRO, dirRO, velRW, phiRW, presRW], /* wgsl */`
@compute @workgroup_size(128) fn setInit(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32,@builtin(num_workgroups) nw:vec3u){
  let n=nbListCount(NB_LIST_F_ENTER);
  for(var e=wid.x;e<n;e+=nw.x){
    let s=nbListSlot(NB_LIST_F_ENTER,e);let o=nbFOrigin(s);
    for(var j=0u;j<4u;j++){
      let l=lid+128u*j;let at=NB_STRIDE*s+l;
      ${zero("vel")}
      ${zero("pres")}
      phi[NB_PHI_F_PHI+at]=bitcast<u32>(NB_PHI_AIR);phi[NB_PHI_F_PHI1+at]=bitcast<u32>(NB_PHI_AIR);
      phi[NB_PHI_F_FLAGS+at]=select(0u,NB_FLAG_OPEN,nbInLattice(o+nbLocalCoord(l)))|NB_FLAG_OWNER_H;
    }
  }
}`);

  // bootPhi: the initial liquid's signed distance (in h) on the entering bricks.
  // boot holds cfg.z boxes of 6 floats (lo xyz, hi xyz, in cells); the union is the min.
  const bootPhi = k("bootPhi", [CFG, setRO, dirRO, phiRW, bootRO], /* wgsl */`
fn sdBox(p:vec3f,lo:vec3f,hi:vec3f)->f32{let q=abs(p-0.5*(lo+hi))-0.5*(hi-lo);return length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);}
@compute @workgroup_size(128) fn bootPhi(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32,@builtin(num_workgroups) nw:vec3u){
  let n=nbListCount(NB_LIST_F_ENTER);
  for(var e=wid.x;e<n;e+=nw.x){
    let s=nbListSlot(NB_LIST_F_ENTER,e);let o=nbFOrigin(s);
    for(var j=0u;j<4u;j++){
      let l=lid+128u*j;let p=vec3f(o+nbLocalCoord(l));
      var d=NB_PHI_AIR;
      for(var i=0u;i<cfg.z;i++){d=min(d,sdBox(p,vec3f(boot[6u*i],boot[6u*i+1u],boot[6u*i+2u]),vec3f(boot[6u*i+3u],boot[6u*i+4u],boot[6u*i+5u])));}
      phi[NB_PHI_F_PHI+NB_STRIDE*s+l]=bitcast<u32>(d);phi[NB_PHI_F_PHI1+NB_STRIDE*s+l]=bitcast<u32>(d);
    }
  }
}`);

  // setLists: a thread a slot below the high-water mark. A live slot (key not
  // NONE, not leaving) joins its lists; a leaving slot's key clears. M1: every
  // live F brick is LIQUID (F_LIQ = F_ALL, brick-abi.md §5.5).
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
        let c=(cls&~NB_CLASS_ENTERING)|NB_CLASS_LIQUID;atomicStore(&dir[r+NB_REC_CLASS],c);
        nbAppend(NB_LIST_F_ALL,i);
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

  // setArgs: one workgroup. Each list's G1, G8 and G64 under the fold rule,
  // the FC triples, and the live, churn and peak counters.
  const setArgs = k("setArgs", [CFG, setRW, argsRW], /* wgsl */`
fn fold(at:u32,n:u32){args[at]=min(n,NB_FOLD);args[at+1u]=max(1u,(n+NB_FOLD-1u)/NB_FOLD);args[at+2u]=1u;}
fn count(k:u32)->u32{
  let n=atomicLoad(&nbSet[NB_H_LIST_COUNT+k]);
  if(n>NB_S_LIST_CAP[k]){nbFail(NB_FAIL_LIST,k);return NB_S_LIST_CAP[k];}
  return n;
}
fn triple(at:u32,n:u32){fold(at+NB_ARGS_G1,n);fold(at+NB_ARGS_G8,(n+7u)/8u);fold(at+NB_ARGS_G64,(n+63u)/64u);}
@compute @workgroup_size(64) fn setArgs(@builtin(local_invocation_index) lid:u32){
  if(lid<17u){triple(NB_ARGS_PER_LIST*lid,count(lid));}
  if(lid==17u){
    let nF=count(NB_LIST_F_ALL);let nC=count(NB_LIST_C_ALL);
    fold(NB_ARGS_FC_G1,nF+nC);fold(NB_ARGS_FC_G8,(nF+nC+7u)/8u);fold(NB_ARGS_FC_G64,(nF+nC+63u)/64u);
    atomicStore(&nbSet[NB_H_LIVE_F],nF);atomicStore(&nbSet[NB_H_LIVE_C],nC);
    atomicMax(&nbSet[NB_H_PEAK_F],nF);atomicMax(&nbSet[NB_H_PEAK_C],nC);
    atomicStore(&nbSet[NB_H_N_ENTER_F],count(NB_LIST_F_ENTER));atomicStore(&nbSet[NB_H_N_LEAVE_F],count(NB_LIST_F_LEAVE));
    atomicStore(&nbSet[NB_H_N_ENTER_C],count(NB_LIST_C_ENTER));atomicStore(&nbSet[NB_H_N_LEAVE_C],count(NB_LIST_C_LEAVE));
    atomicMax(&nbSet[NB_H_PEAK_CAND],atomicLoad(&nbSet[NB_H_N_CAND]));
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

  return { setMark, setClaimPages, setClaimBricks, setLeave, setLeavePages, setRecords, setInit, bootPhi, setLists, setArgs, stepEnd };
}

