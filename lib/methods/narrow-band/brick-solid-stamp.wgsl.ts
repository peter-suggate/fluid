/**
 * solidStamp (M3 stage 2b, lane E; brick-abi.md §2.3, m5-bodies-design-2026-11-17.md A5/A6): the static solid's cuts
 * on the bricks that enter, from the static voxel mask through the five hooks (brick-static.wgsl.ts).
 *
 * A workgroup an F_ENTER brick (G1), at the stamp stage (after the head's lists, before P's bodyStamp):
 * - the brick is cut when one of its cells is a mask voxel, or one of its lower faces has a mask voxel below it (its
 *   lower faces are its own storage). The box rule alone cuts nothing: a brick with no K slot is open but for the box
 *   rule's faces (§2.3), and a stamped brick keeps the same box faces, so the two conventions agree at every wall;
 * - a NEAR_BODY brick is P's (A5: bodyStamp writes its static and body cuts together); solidStamp skips it;
 * - a cut brick takes a K slot (nbKTake: POOL_K and an uncut brick when the pool is exhausted), is CUT_STATIC, and
 *   gets the whole slot: FRAC = nbStaticCellOpen, AP_a = nbStaticFaceOpen of the face below, W = 0 and BMASK = 0 (A1:
 *   the first stamp to write a fresh slot writes all of it). At a box wall nbStaticFaceOpen is the old engine's shell
 *   voxel face, 0.5 (open(-1) + open(0)) = 0.5 FRAC of the cell inside: the cut-free rows' K_WALL fraction 0.5 and
 *   P's pbStampFace agree with it (lane P's review, 2026-11-17); two cells past the box give 0;
 * - FLAGS.OPEN clears on every cell whose FRAC is 0 (setInit set it by the box rule).
 * - NEAR_SOLID (brick-solid-walk.wgsl.ts, every entering brick, a body brick's too): a mask voxel within
 *   BRICK_SOLID_RING cells of the brick on every axis; setMark keeps it while the brick is resident.
 *
 * solidRestamp (M3 stage 2c, the live edit): between two steps, after the host uploads the mask words an edit rewrote
 * (BrickEngine.restampStatics), a workgroup an F_ALL brick restamps the bricks the edit can reach, and every other brick
 * returns at once and keeps every word:
 * - the reach: the edit's changed cells `solidEdit` = [lo, hi), grown by BRICK_SOLID_RING. A brick's NEAR_SOLID reads
 *   the mask within the ring, and its cuts read its cells and the cell below each (inside the ring);
 * - NEAR_SOLID is set or cleared, from the same scan solidStamp makes;
 * - a NEAR_BODY or BODY_WAS brick's cuts and FLAGS are P's (bodyStamp rewrites them each step from the same mask);
 * - otherwise a cut brick takes its slot (nbKTake: the one it holds, else a fresh one, FAIL.POOL_K when none), gets the
 *   whole slot as solidStamp writes it, and CUT_STATIC. A brick no longer cut loses CUT_STATIC, and the setKRelease
 *   that follows (cfg.z = 0 over F_ALL) gives its slot back, so no pop and push share a dispatch;
 * - FLAGS.OPEN of each cell is the box rule (setInit) and not a mask voxel: a fill closes a cell, a clear opens it.
 * The liquid a fill buries is the walk's: pSolid retires a record whose start is closed, and phiSolid keeps a buried
 * vertex's φ (umVertexBuried), as the old engine does.
 */
import { brickAbiWGSL, resolveBox, type BrickLayout } from "./brick-abi";
import { declareBindings, type BrickBinding, type BrickKernel } from "./brick-kernel";
import { BRICK_SET_HELPERS_WGSL } from "./brick-set.wgsl";
import { brickStaticWGSL } from "./brick-static.wgsl";
import { BRICK_SOLID_RING as BRICK_SOLID_RING_CELLS } from "./brick-solid-walk.wgsl";
import type { SolidOccupancyMaskWalls } from "../../core/solid-occupancy-mask";

/** The static mask the engine binds as `solid` (brickStaticMask's words and walls; brick-engine.ts). */
export interface BrickStaticMaskBinding {
  readonly words: Uint32Array;
  readonly walls: SolidOccupancyMaskWalls;
}

/** The edit's changed cells, [lo, hi) as six i32 words (two padding), bound as `solidEdit` (solidRestamp). */
export const BRICK_SOLID_EDIT_WORDS = 8;

export function brickSolidStampKernels(L: BrickLayout, mask: Pick<BrickStaticMaskBinding, "walls">): { solidStamp: BrickKernel; solidRestamp: BrickKernel } {
  const bindings: BrickBinding[] = [
    { name: "cfg", kind: "uniform" },
    { name: "set", kind: "rw", type: "array<atomic<u32>>" },
    { name: "dir", kind: "rw", type: "array<atomic<u32>>" },
    { name: "phi", kind: "rw", type: "array<u32>" },
    { name: "cut", kind: "rw", type: "array<f32>" },
    { name: "solid", kind: "ro", type: "array<u32>" },
  ];
  const statics = brickStaticWGSL({ box: L.box, openTop: mask.walls.policy === 2, frame: "vec3f(0.0)", h: "vec3f(1.0)", mask: { buffer: "solid", walls: mask.walls } });
  const code = [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: true } }), declareBindings(bindings), BRICK_SET_HELPERS_WGSL, statics, /* wgsl */`
var<workgroup> wCut:atomic<u32>;
var<workgroup> wNear:atomic<u32>;
var<workgroup> wSlot:u32;
var<workgroup> wHead:vec4u;
// Whether the box rule alone closes cell c (outside the box): the mask's voxels are the cut, the box is not.
fn nbStampMaskVoxel(c:vec3i)->bool{return nbStaticInBox(c)&&nbStaticVoxel(c);}
@compute @workgroup_size(64) fn solidStamp(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  // The brick, read by lane 0 and shared (uniform control flow for the barriers): x live, y slot, z key.
  if(lid==0u){
    let e=nbGroup(wid);var h=vec4u(0u);
    if(e<nbListCount(NB_LIST_F_ENTER)){
      let s=nbListSlot(NB_LIST_F_ENTER,e);let r=NB_D_FREC+32u*s;
      h=vec4u(1u,s,atomicLoad(&dir[r+NB_REC_KEY]),select(0u,1u,(atomicLoad(&dir[r+NB_REC_CLASS])&NB_CLASS_NEAR_BODY)!=0u));
    }
    wHead=h;atomicStore(&wCut,0u);atomicStore(&wNear,0u);
  }
  let h=workgroupUniformLoad(&wHead);
  if(h.x==0u){return;}
  let s=h.y;let r=NB_D_FREC+32u*s;let o=8*nbKeyCoord(h.z);
  // NEAR_SOLID (brick-solid-walk.wgsl.ts): a mask voxel among the 24^3 cells of the brick and its 26 neighbours, a
  // body brick's too (pSolid and phiSolid move every brick a record or a trace can reach a voxel from).
  for(var l=lid;l<${(3 * BRICK_SOLID_RING_CELLS) ** 3}u;l+=64u){
    let c=o-vec3i(${BRICK_SOLID_RING_CELLS})+vec3i(i32(l%${3 * BRICK_SOLID_RING_CELLS}u),i32((l/${3 * BRICK_SOLID_RING_CELLS}u)%${3 * BRICK_SOLID_RING_CELLS}u),i32(l/${(3 * BRICK_SOLID_RING_CELLS) ** 2}u));
    if(nbStampMaskVoxel(c)){atomicOr(&wNear,1u);break;}
  }
  workgroupBarrier();
  if(lid==0u&&atomicLoad(&wNear)!=0u){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_NEAR_SOLID);}
  // A NEAR_BODY brick's cuts are P's bodyStamp's (A5).
  if(h.w!=0u){return;}
  for(var l=lid;l<512u;l+=64u){
    let c=o+nbLocalCoord(l);
    var cut=nbStampMaskVoxel(c);
    for(var a=0u;a<3u;a++){cut=cut||nbStampMaskVoxel(c-nbStaticAxis(a));}
    if(cut){atomicOr(&wCut,1u);}
  }
  workgroupBarrier();
  if(lid==0u){
    var ks=NB_NONE;
    if(atomicLoad(&wCut)!=0u){ks=nbKTake(s);if(ks!=NB_NONE){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_CUT_STATIC);}}
    wSlot=ks;
  }
  let ks=workgroupUniformLoad(&wSlot);
  if(ks==NB_NONE){return;}
  if(lid==0u){cut[NB_CUT_BMASK+ks]=0.0;}
  for(var l=lid;l<512u;l+=64u){
    let c=o+nbLocalCoord(l);let at=NB_STRIDE*ks+l;
    let frac=nbStaticCellOpen(c);
    cut[NB_CUT_FRAC+at]=frac;
    for(var a=0u;a<3u;a++){
      let b=c-nbStaticAxis(a);
      let ap=nbStaticFaceOpen(b,a);
      // Constant channel bases (no dynamic index into a channel table).
      if(a==0u){cut[NB_CUT_AP0+at]=ap;cut[NB_CUT_W0+at]=0.0;}else if(a==1u){cut[NB_CUT_AP1+at]=ap;cut[NB_CUT_W1+at]=0.0;}else{cut[NB_CUT_AP2+at]=ap;cut[NB_CUT_W2+at]=0.0;}
    }
    if(frac==0.0){let f=NB_PHI_F_FLAGS+NB_STRIDE*s+l;phi[f]=phi[f]&~NB_FLAG_OPEN;}
  }
}`].join("\n");
  return { solidStamp: { label: "solidStamp", entry: "solidStamp", code, bindings }, solidRestamp: solidRestampKernel(L, statics, bindings) };
}

function solidRestampKernel(L: BrickLayout, statics: string, stamp: readonly BrickBinding[]): BrickKernel {
  const bindings: BrickBinding[] = [...stamp, { name: "solidEdit", kind: "ro", type: "array<i32>" }];
  const R = BRICK_SOLID_RING_CELLS, N = 3 * R;
  const code = [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: true } }), declareBindings(bindings), BRICK_SET_HELPERS_WGSL, statics, /* wgsl */`
var<workgroup> wCut:atomic<u32>;
var<workgroup> wNear:atomic<u32>;
var<workgroup> wSlot:u32;
var<workgroup> wHead:vec4u;
fn nbStampMaskVoxel(c:vec3i)->bool{return nbStaticInBox(c)&&nbStaticVoxel(c);}
@compute @workgroup_size(64) fn solidRestamp(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  // The brick, read by lane 0 and shared: x live and in the edit's reach, y slot, z key, w class.
  if(lid==0u){
    let e=nbGroup(wid);var h=vec4u(0u);
    if(e<nbListCount(NB_LIST_F_ALL)){
      let s=nbListSlot(NB_LIST_F_ALL,e);let r=NB_D_FREC+32u*s;let key=atomicLoad(&dir[r+NB_REC_KEY]);let o=8*nbKeyCoord(key);
      let lo=vec3i(solidEdit[0],solidEdit[1],solidEdit[2])-vec3i(${R});let hi=vec3i(solidEdit[3],solidEdit[4],solidEdit[5])+vec3i(${R});
      if(all(o+vec3i(8)>lo)&&all(o<hi)){h=vec4u(1u,s,key,atomicLoad(&dir[r+NB_REC_CLASS]));}
    }
    wHead=h;atomicStore(&wCut,0u);atomicStore(&wNear,0u);
  }
  let h=workgroupUniformLoad(&wHead);
  if(h.x==0u){return;}
  let s=h.y;let r=NB_D_FREC+32u*s;let o=8*nbKeyCoord(h.z);
  for(var l=lid;l<${N ** 3}u;l+=64u){
    let c=o-vec3i(${R})+vec3i(i32(l%${N}u),i32((l/${N}u)%${N}u),i32(l/${N * N}u));
    if(nbStampMaskVoxel(c)){atomicOr(&wNear,1u);break;}
  }
  workgroupBarrier();
  if(lid==0u){
    if(atomicLoad(&wNear)!=0u){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_NEAR_SOLID);}else{atomicAnd(&dir[r+NB_REC_CLASS],~NB_CLASS_NEAR_SOLID);}
  }
  // A body brick's cuts and FLAGS are P's bodyStamp's (A5).
  if((h.w&(NB_CLASS_NEAR_BODY|NB_CLASS_BODY_WAS))!=0u){return;}
  for(var l=lid;l<512u;l+=64u){
    let c=o+nbLocalCoord(l);
    var cut=nbStampMaskVoxel(c);
    for(var a=0u;a<3u;a++){cut=cut||nbStampMaskVoxel(c-nbStaticAxis(a));}
    if(cut){atomicOr(&wCut,1u);}
    // FLAGS.OPEN: the box rule (setInit) and not a mask voxel.
    let f=NB_PHI_F_FLAGS+NB_STRIDE*s+l;let open=nbInWallBox(c)&&!nbStampMaskVoxel(c);
    phi[f]=select(phi[f]&~NB_FLAG_OPEN,phi[f]|NB_FLAG_OPEN,open);
  }
  workgroupBarrier();
  if(lid==0u){
    var ks=NB_NONE;
    if(atomicLoad(&wCut)!=0u){ks=nbKTake(s);if(ks!=NB_NONE){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_CUT_STATIC);}}
    else{atomicAnd(&dir[r+NB_REC_CLASS],~NB_CLASS_CUT_STATIC);}
    wSlot=ks;
  }
  let ks=workgroupUniformLoad(&wSlot);
  if(ks==NB_NONE){return;}
  if(lid==0u){cut[NB_CUT_BMASK+ks]=0.0;}
  for(var l=lid;l<512u;l+=64u){
    let c=o+nbLocalCoord(l);let at=NB_STRIDE*ks+l;
    cut[NB_CUT_FRAC+at]=nbStaticCellOpen(c);
    for(var a=0u;a<3u;a++){
      let ap=nbStaticFaceOpen(c-nbStaticAxis(a),a);
      if(a==0u){cut[NB_CUT_AP0+at]=ap;cut[NB_CUT_W0+at]=0.0;}else if(a==1u){cut[NB_CUT_AP1+at]=ap;cut[NB_CUT_W1+at]=0.0;}else{cut[NB_CUT_AP2+at]=ap;cut[NB_CUT_W2+at]=0.0;}
    }
  }
}`].join("\n");
  return { label: "solidRestamp", entry: "solidRestamp", code: resolveBox(L, code), bindings };
}
