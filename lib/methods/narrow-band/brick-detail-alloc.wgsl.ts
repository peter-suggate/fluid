/**
 * The detail engine's allocator (brick-detail.ts), the parent side: every parent step, which mask tiles the fine
 * engine runs (brick-detail.wgsl.ts), chosen by the parent's heat under a fixed budget, and the parent's state about
 * them. Six dispatches on the parent's layout, one pass, after the parent's step and before the fine one:
 *
 * - dtlClear: the last step's mask off (its actives' bits), its slot map kept as the previous one, the counters reset;
 * - dtlHeat: a workgroup a parent F brick. The heat of a 4h tile is the deformation a step at the parent's surface,
 *   dt ‖∇u‖ (the velocity gradient's Frobenius norm, its face differences between liquid cells: the air's extended
 *   velocity is no flow) at the liquid cells within DETAIL_SURFACE cells of the surface, the tile's largest, held
 *   with a decay: H = max(raw, H_last e^(−Δt / τ)). A splash's crown, a sheet stretching and breaking, a jet's neck
 *   are where the surface deforms fastest; a still pool, a falling blob, a uniform flow have none. A tile with
 *   H ≥ DETAIL_HEAT_MIN is a candidate, counted in a log histogram (four bins an octave), a tile on last step
 *   DETAIL_HOLD_BINS bins up (hysteresis: the mask turns over only for markedly hotter tiles);
 * - dtlThreshold: the histogram's cut, the hottest bins whose candidates' tiles fit the budget, and the quota of the
 *   first bin that does not fit whole (its incumbents first, then its first newcomers to come, so the budget fills): a candidate claims its 3³
 *   tiles (so the hot tile is core and the relaxation shell lies about it), κ tiles a candidate, κ the last steps'
 *   ratio smoothed (the tiles they asked for, past the budget too; between 1 and 27);
 * - dtlMark: the chosen candidates' tiles and the authored ones (forced, no dilation) on: a slot each in the map, at
 *   most the budget's (past it, an overflow, counted, no tile);
 * - dtlRing: each active's since (its activation step: fresh while under DETAIL_FRESH steps), core bit, and ring (its
 *   inactive container neighbours, a slot each for the nodes the edge's faces read); the tiles turned on (fresh list)
 *   and off (gone list) this step;
 * - dtlNodes: the parent's φ and faces on every slot's block that is not core (a core tile never reads them).
 * Every store is the budget's size or the parent's pools' (the heat is a C slot's tiles', tagged with the tile), none
 * the container's but the mask's bit planes (3 bits a 4h tile).
 */
import { BrickLayout, boxWGSL, brickAbiWGSL, foldArgs, resolveBox } from "./brick-abi";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { DETAIL_BLOCK, DETAIL_FRESH, DETAIL_HASH_WGSL, DETAIL_META, brickDetailHash, brickDetailPlaneWords, brickDetailSrc, type BrickDetailLayout } from "./brick-detail.wgsl";

/** A cell is the parent's surface within this many cells (|φ| in h). */
export const DETAIL_SURFACE = 1.5;
/** The least heat (dt ‖∇u‖, a step's deformation) that asks for detail. */
export const DETAIL_HEAT_MIN = 0.05;
/** The heat's memory (s): a tile stays hot this long after its deformation stops (e-folding). */
export const DETAIL_HEAT_TAU = 0.25;
/** The histogram's bins (four an octave above DETAIL_HEAT_MIN). */
export const DETAIL_BINS = 64;

/** The allocator state's words (its own buffer). */
export function brickDetailAllocWords(d: BrickDetailLayout, parent: Pick<BrickLayout, "cap">, forced: number) {
  const cand = 8 * parent.cap.f, hash = brickDetailHash(d);
  const st = 0, hist = 16, histHeld = hist + DETAIL_BINS, candT = histHeld + DETAIL_BINS, candB = candT + cand, phash = candB + cand, pkeys = phash + 2 * hash;
  const psince = pkeys + d.budget, since = psince + d.budget, heat = since + d.budget, forcedAt = heat + 3 * 512 * parent.cap.c;
  return { st, hist, histHeld, candT, candB, cand, phash, pkeys, psince, since, heat, forced: forcedAt, words: forcedAt + Math.max(1, forced) };
}
/** The state words (ST). */
export const DETAIL_ST = { STEP: 0, CAND: 1, SELECTED: 2, PREV_ACTIVE: 3, PREV_SELECTED: 4, THRESHOLD: 5, QUOTA: 6, KAPPA: 7, DTL: 8, TAKEN: 9, PREV_WANT: 10, QUOTA_NEW: 11, TAKEN_NEW: 12 } as const;
/** The heat bins (a quarter octave each) a tile on last step gains: a newcomer must be 2^(HOLD/4) hotter to take its place. */
export const DETAIL_HOLD_BINS = 3;

export interface BrickDetailAllocOptions {
  readonly dt_s: number;
  readonly h_m: number;
  /** The authored tiles (always on). */
  readonly forced: number;
}

/** The allocator's WGSL on the parent's layout: entry points dtlClear, dtlHeat, dtlThreshold, dtlMark, dtlRing, dtlNodes. */
export function brickDetailAllocWGSL(parent: BrickLayout, d: BrickDetailLayout, o: BrickDetailAllocOptions): string {
  const s = brickDetailSrc(d), a = brickDetailAllocWords(d, parent, o.forced), H = brickDetailHash(d), M = DETAIL_META, ST = DETAIL_ST;
  const block3 = DETAIL_BLOCK ** 3;
  return resolveBox(parent, /* wgsl */`
${brickAbiWGSL(parent, { set: { name: "nbSet", atomic: false }, dir: { name: "dir", atomic: false } })}
@group(0) @binding(0) var<storage,read> nbSet:array<u32>;
@group(0) @binding(1) var<storage,read> dir:array<u32>;
@group(0) @binding(2) var<storage,read> phi:array<u32>;
@group(0) @binding(3) var<storage,read> vel:array<f32>;
@group(0) @binding(4) var<storage,read_write> src:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> fdir:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> al:array<atomic<u32>>;
${brickOwnerWGSL({ phi: "phi", vel: "vel", box: boxWGSL(parent) })}
${DETAIL_HASH_WGSL(H)}
const DT_TILES:vec3i=vec3i(${d.tiles.join(",")});const DT_B:u32=${d.budget}u;const DT_S:u32=${d.slots}u;const DT_H:u32=${H}u;
const DT_HEAT_B:f32=${Math.max(1, d.budget - o.forced)}.0;const DT_NF:u32=${o.forced}u;const DT_CAND:u32=${a.cand}u;
const DT_PLANE:u32=${brickDetailPlaneWords(d)}u;const DT_R:f32=${d.factor}.0;
const DT_DT:f32=${o.dt_s};const DT_HM:f32=${o.h_m};const DT_TAU:f32=${DETAIL_HEAT_TAU};const DT_HMIN:f32=${DETAIL_HEAT_MIN};const DT_SURF:f32=${DETAIL_SURFACE};
const S_META:u32=${s.meta}u;const S_MAP:u32=${s.map}u;const S_KEYS:u32=${s.keys}u;const S_FRESH:u32=${s.fresh}u;const S_GONE:u32=${s.gone}u;const S_NODES:u32=${s.nodes}u;
const A_ST:u32=${a.st}u;const A_HIST:u32=${a.hist}u;const A_HISTH:u32=${a.histHeld}u;const A_CT:u32=${a.candT}u;const A_CB:u32=${a.candB}u;const A_PHASH:u32=${a.phash}u;
const A_PKEYS:u32=${a.pkeys}u;const A_PSINCE:u32=${a.psince}u;const A_SINCE:u32=${a.since}u;const A_HEAT:u32=${a.heat}u;const A_FORCED:u32=${a.forced}u;
fn dtIn(T:vec3i)->bool{return all(T>=vec3i(0))&&all(T<DT_TILES);}
fn dtIdx(T:vec3i)->u32{return u32(T.x+DT_TILES.x*(T.y+DT_TILES.y*T.z));}
fn dtTile(i:u32)->vec3i{let n=vec3u(DT_TILES);return vec3i(i32(i%n.x),i32((i/n.x)%n.y),i32(i/(n.x*n.y)));}
fn dtSt(k:u32)->u32{return atomicLoad(&al[A_ST+k]);}
fn dtMeta(k:u32)->u32{return atomicLoad(&src[S_META+k]);}
// The fine engine's mask planes in its dir (ST.DTL, written at bind).
fn dtDtl()->u32{return atomicLoad(&al[A_ST+${ST.DTL}u]);}
fn dtBit(p:u32,i:u32)->bool{return ((atomicLoad(&fdir[dtDtl()+p*DT_PLANE+(i>>5u)])>>(i&31u))&1u)!=0u;}
fn dtSetBit(p:u32,i:u32){atomicOr(&fdir[dtDtl()+p*DT_PLANE+(i>>5u)],1u<<(i&31u));}
fn dtClearBits(i:u32){let m=~(1u<<(i&31u));for(var p=0u;p<3u;p++){atomicAnd(&fdir[dtDtl()+p*DT_PLANE+(i>>5u)],m);}}
// Insert tile i into the slot map: its entry and whether this call put it there (false: it was there).
fn dtInsert(i:u32)->vec2u{
  var e=nbDtlHash(i);var k=0u;
  loop{
    if(k>=DT_H){return vec2u(NB_NONE,0u);}
    let r=atomicCompareExchangeWeak(&src[S_MAP+2u*e],0u,i+1u);
    if(r.exchanged){return vec2u(e,1u);}
    if(r.old_value==i+1u){return vec2u(e,0u);}
    if(r.old_value!=0u){e=(e+1u)&(DT_H-1u);k++;}
  }
}
// Tile i's slot in the previous step's map, NB_NONE without one.
fn dtPrevSlot(i:u32)->u32{
  var e=nbDtlHash(i);
  for(var k=0u;k<DT_H;k++){
    let key=atomicLoad(&al[A_PHASH+2u*e]);
    if(key==i+1u){return atomicLoad(&al[A_PHASH+2u*e+1u]);}
    if(key==0u){return NB_NONE;}
    e=(e+1u)&(DT_H-1u);
  }
  return NB_NONE;
}

@compute @workgroup_size(64) fn dtlClear(@builtin(global_invocation_id) g:vec3u){
  let j=g.x;
  if(j<dtSt(${ST.PREV_ACTIVE}u)){
    let i=atomicLoad(&src[S_KEYS+j]);dtClearBits(i);
    atomicStore(&al[A_PKEYS+j],i);atomicStore(&al[A_PSINCE+j],atomicLoad(&al[A_SINCE+j]));
  }
  if(j<DT_H){
    atomicStore(&al[A_PHASH+2u*j],atomicLoad(&src[S_MAP+2u*j]));atomicStore(&al[A_PHASH+2u*j+1u],atomicLoad(&src[S_MAP+2u*j+1u]));
    atomicStore(&src[S_MAP+2u*j],0u);
  }
  if(j<${64}u){atomicStore(&al[A_HIST+j],0u);atomicStore(&al[A_HISTH+j],0u);}
  if(j==0u){
    for(var k=0u;k<8u;k++){atomicStore(&src[S_META+k],0u);}
    atomicStore(&al[A_ST+${ST.CAND}u],0u);atomicStore(&al[A_ST+${ST.SELECTED}u],0u);atomicStore(&al[A_ST+${ST.TAKEN}u],0u);atomicStore(&al[A_ST+${ST.TAKEN_NEW}u],0u);atomicAdd(&al[A_ST+${ST.STEP}u],1u);
  }
}

var<workgroup> wgGo:u32;
var<workgroup> wgMax:array<atomic<u32>,8>;
var<workgroup> wgN:atomic<u32>;
var<workgroup> wgCells:array<u32,512>;
// The face and the vertex φ (h) by their owners (brick-owner.wgsl.ts): an F brick's storage holds them only where an
// h tile owns them; a tiered parent's 4h tiles inside an F brick leave it stale, and read raw it was a falling drop's
// heat.
fn dtFace(c:vec3i,a:u32)->f32{return nbFaceU(c,a);}
fn dtPhi(c:vec3i)->f32{return nbVertexPhi(c);}
@compute @workgroup_size(64) fn dtlHeat(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);
  if(lane==0u){wgGo=select(0u,1u,g<nbListCount(NB_LIST_F_ALL));}
  if(lane<8u){atomicStore(&wgMax[lane],0u);}
  if(workgroupUniformLoad(&wgGo)==0u){return;}
  let s=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(s);
  // The surface cells first, listed, then the lanes share them: a surface lies on few of a brick's cells, scattered
  // over the lanes' eight, and walked in place every lane waited on its SIMD's slowest (0.28 of 0.36 ms a step on
  // cm12-figure-6's parent, 2026-11-23).
  for(var j=0u;j<8u;j++){
    let l=lane+64u*j;let c=o+vec3i(i32(l&7u),i32((l>>3u)&7u),i32(l>>6u));
    if(!nbInWallBox(c)||!nbCellOwnerH(c)){continue;}
    let p=dtPhi(c);
    if(p>=0.0||p<=-DT_SURF){continue;}
    wgCells[atomicAdd(&wgN,1u)]=l;
  }
  workgroupBarrier();
  let n=atomicLoad(&wgN);
  for(var k=lane;k<n;k+=64u){
    let l=wgCells[k];let q=vec3i(i32(l&7u),i32((l>>3u)&7u),i32(l>>6u));let c=o+q;
    // The liquid's own gradient: a difference across b only to a liquid vertex, along a's own axis too (face a of
    // c + e_a has c + e_a for its corner; past the surface it is an air cell's face, which holds no flow, and the
    // difference would read a falling drop's speed as deformation).
    var sum=0.0;var wet=vec3<bool>(true);
    for(var b=0u;b<3u;b++){var e=vec3i(0);e[b]=1;wet[b]=nbInWallBox(c+e)&&dtPhi(c+e)<0.0;}
    for(var a=0u;a<3u;a++){
      let u0=dtFace(c,a);
      for(var b=0u;b<3u;b++){if(!wet[b]){continue;}var e=vec3i(0);e[b]=1;let du=dtFace(c+e,a)-u0;sum+=du*du;}
    }
    let raw=DT_DT*sqrt(sum)/DT_HM;
    let t=(q>>vec3u(2u));atomicMax(&wgMax[u32(t.x+2*t.y+4*t.z)],bitcast<u32>(raw));
  }
  workgroupBarrier();
  if(lane>=8u){return;}
  let T=(o>>vec3u(2u))+vec3i(i32(lane&1u),i32((lane>>1u)&1u),i32(lane>>2u));
  if(!dtIn(T)){return;}
  let raw=bitcast<f32>(atomicLoad(&wgMax[lane]));let i=dtIdx(T);let step=dtSt(${ST.STEP}u);
  var heat=raw;
  let cs=nbCSlotOfPage(T>>vec3u(3u));
  if(cs!=NB_NONE){
    let at=A_HEAT+3u*(512u*cs+nbTileIndex(T));
    if(atomicLoad(&al[at+2u])==i+1u){
      let last=bitcast<f32>(atomicLoad(&al[at]));let age=f32(step-atomicLoad(&al[at+1u]));
      heat=max(heat,last*exp(-age*DT_DT/DT_TAU));
    }
    atomicStore(&al[at],bitcast<u32>(heat));atomicStore(&al[at+1u],step);atomicStore(&al[at+2u],i+1u);
  }
  if(heat<DT_HMIN){return;}
  // Hysteresis: a tile on last step holds its place against a newcomer up to DETAIL_HOLD_BINS hotter.
  let held=select(0u,${DETAIL_HOLD_BINS}u,dtPrevSlot(i)<dtSt(${ST.PREV_ACTIVE}u));
  let bin=min(u32(4.0*log2(heat/DT_HMIN))+held,${63}u);
  let k=atomicAdd(&al[A_ST+${ST.CAND}u],1u);
  if(k<DT_CAND){atomicStore(&al[A_CT+k],i);atomicStore(&al[A_CB+k],bin);atomicAdd(&al[A_HIST+bin],1u);if(held!=0u){atomicAdd(&al[A_HISTH+bin],1u);}}
}

@compute @workgroup_size(1) fn dtlThreshold(){
  let pa=dtSt(${ST.PREV_WANT}u);let ps=dtSt(${ST.PREV_SELECTED}u);
  // κ, the tiles a selected seed turns on (its 3³ less the overlap), smoothed over steps (a step's is noisy).
  var kappa=bitcast<f32>(dtSt(${ST.KAPPA}u));
  if(!(kappa>=1.0)){kappa=27.0;}
  if(ps>0u){kappa=clamp(0.5*kappa+0.5*f32(max(pa,DT_NF)-DT_NF)/f32(ps),1.0,27.0);}
  atomicStore(&al[A_ST+${ST.KAPPA}u],bitcast<u32>(kappa));
  // The bins that fit whole, hottest first; the first that does not takes its quota of seeds (dtlMark, first come).
  var cut=${64}u;var sum=0.0;var quota=0u;
  for(var b=${63}i;b>=0;b--){
    let n=f32(atomicLoad(&al[A_HIST+u32(b)]));
    if(kappa*(sum+n)>DT_HEAT_B){quota=u32(max(DT_HEAT_B/kappa-sum,0.0));break;}
    sum+=n;cut=u32(b);
  }
  // The partial bin's quota goes to its incumbents (on last step) first, so the mask holds still at the budget.
  var held=0u;if(cut>0u&&cut<=${63}u){held=min(quota,atomicLoad(&al[A_HISTH+cut-1u]));}
  atomicStore(&al[A_ST+${ST.THRESHOLD}u],cut);atomicStore(&al[A_ST+${ST.QUOTA}u],held);atomicStore(&al[A_ST+${ST.QUOTA_NEW}u],quota-held);
  atomicStore(&src[S_META+${M.THRESHOLD}u],cut);atomicStore(&src[S_META+${M.HOT}u],dtSt(${ST.CAND}u));
}

fn dtActivate(i:u32){
  let r=dtInsert(i);if(r.y==0u){return;}
  let k=atomicAdd(&src[S_META+${M.ACTIVE}u],1u);
  if(k<DT_B){atomicStore(&src[S_KEYS+k],i);atomicStore(&src[S_MAP+2u*r.x+1u],k);dtSetBit(0u,i);}
  else{atomicStore(&src[S_MAP+2u*r.x+1u],NB_NONE);atomicAdd(&src[S_META+${M.OVERFLOW}u],1u);}
}
@compute @workgroup_size(64) fn dtlMark(@builtin(global_invocation_id) g:vec3u){
  let j=g.x;
  if(j<min(dtSt(${ST.CAND}u),DT_CAND)){
    let bin=atomicLoad(&al[A_CB+j]);let cut=dtSt(${ST.THRESHOLD}u);
    if(bin<cut){
      if(bin+1u!=cut){return;}
      let i=atomicLoad(&al[A_CT+j]);
      if(dtPrevSlot(i)<dtSt(${ST.PREV_ACTIVE}u)){if(atomicAdd(&al[A_ST+${ST.TAKEN}u],1u)>=dtSt(${ST.QUOTA}u)){return;}}
      else if(atomicAdd(&al[A_ST+${ST.TAKEN_NEW}u],1u)>=dtSt(${ST.QUOTA_NEW}u)){return;}
    }
    atomicAdd(&al[A_ST+${ST.SELECTED}u],1u);
    let T=dtTile(atomicLoad(&al[A_CT+j]));
    for(var k=0u;k<27u;k++){let Q=T+nbNeighbourOffset(k);if(dtIn(Q)){dtActivate(dtIdx(Q));}}
    return;
  }
  if(j>=DT_CAND&&j-DT_CAND<DT_NF){dtActivate(atomicLoad(&al[A_FORCED+j-DT_CAND]));}
}

@compute @workgroup_size(64) fn dtlRing(@builtin(global_invocation_id) g:vec3u){
  let j=g.x;let step=dtSt(${ST.STEP}u);
  if(j<min(dtMeta(${M.ACTIVE}u),DT_B)){
    let i=atomicLoad(&src[S_KEYS+j]);let T=dtTile(i);
    let ps=dtPrevSlot(i);let was=ps<dtSt(${ST.PREV_ACTIVE}u);
    let since=select(step,atomicLoad(&al[A_PSINCE+min(ps,DT_B-1u)]),was);
    atomicStore(&al[A_SINCE+j],since);
    if(!was){let f=atomicAdd(&src[S_META+${M.FRESH}u],1u);if(f<DT_B){atomicStore(&src[S_FRESH+f],i);}}
    if(step-since<${DETAIL_FRESH}u){dtSetBit(1u,i);}
    var core=true;
    for(var k=0u;k<27u;k++){
      let Q=T+nbNeighbourOffset(k);if(!dtIn(Q)){continue;}
      let q=dtIdx(Q);if(q==i||dtBit(0u,q)){continue;}
      core=false;
      let r=dtInsert(q);if(r.y==0u){continue;}
      let n=atomicAdd(&src[S_META+${M.RING}u],1u);
      if(DT_B+n<DT_S){atomicStore(&src[S_KEYS+DT_B+n],q);atomicStore(&src[S_MAP+2u*r.x+1u],DT_B+n);}
      else{atomicStore(&src[S_MAP+2u*r.x+1u],NB_NONE);atomicAdd(&src[S_META+${M.OVERFLOW}u],1u);}
    }
    if(core){dtSetBit(2u,i);}
  }
  if(j<dtSt(${ST.PREV_ACTIVE}u)){
    let i=atomicLoad(&al[A_PKEYS+j]);
    if(!dtBit(0u,i)){let k=atomicAdd(&src[S_META+${M.GONE}u],1u);if(k<DT_B){atomicStore(&src[S_GONE+k],i);}}
  }
}

@compute @workgroup_size(64) fn dtlNodes(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let slot=nbGroup(wid);
  if(slot==0u&&lane==0u){
    atomicStore(&al[A_ST+${ST.PREV_ACTIVE}u],min(dtMeta(${M.ACTIVE}u),DT_B));atomicStore(&al[A_ST+${ST.PREV_SELECTED}u],dtSt(${ST.SELECTED}u));atomicStore(&al[A_ST+${ST.PREV_WANT}u],dtMeta(${M.ACTIVE}u));
    atomicStore(&src[S_META+${M.STEP}u],dtSt(${ST.STEP}u));
  }
  if(slot<DT_B){if(slot>=min(dtMeta(${M.ACTIVE}u),DT_B)){return;}}
  else if(slot>=DT_B+min(dtMeta(${M.RING}u),DT_S-DT_B)){return;}
  let i=atomicLoad(&src[S_KEYS+slot]);
  if(slot<DT_B&&dtBit(2u,i)){return;}
  let base=4*dtTile(i)-vec3i(1);
  for(var n=lane;n<${block3}u;n+=64u){
    let v=base+vec3i(i32(n%${DETAIL_BLOCK}u),i32((n/${DETAIL_BLOCK}u)%${DETAIL_BLOCK}u),i32(n/${DETAIL_BLOCK ** 2}u));
    let at=S_NODES+4u*(${block3}u*slot+n);
    atomicStore(&src[at],bitcast<u32>(nbVertexPhi(v)*DT_R));
    for(var a=0u;a<3u;a++){atomicStore(&src[at+1u+a],bitcast<u32>(nbFaceU(v,a)));}
  }
}`);
}

/** The allocator's dispatches (workgroups), in order. */
export function brickDetailAllocDispatches(parent: Pick<BrickLayout, "cap">, d: BrickDetailLayout, forced: number): Record<string, [number, number, number]> {
  const a = brickDetailAllocWords(d, parent, forced), H = brickDetailHash(d);
  const g = (n: number): [number, number, number] => [Math.max(1, Math.ceil(n / 64)), 1, 1];
  return {
    dtlClear: g(Math.max(H, d.budget, 64)),
    dtlHeat: foldArgs(Math.max(1, parent.cap.f)),
    dtlThreshold: [1, 1, 1],
    dtlMark: g(a.cand + forced),
    dtlRing: g(d.budget),
    dtlNodes: foldArgs(d.slots),
  };
}
