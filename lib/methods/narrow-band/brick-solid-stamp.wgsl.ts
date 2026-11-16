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
 *
 * A live edit (ring flag SOLID_EDIT) restamps F_ALL from the updated mask: next, with the edit path (M3 stage 2c).
 */
import { brickAbiWGSL, type BrickLayout } from "./brick-abi";
import { declareBindings, type BrickBinding, type BrickKernel } from "./brick-kernel";
import { BRICK_SET_HELPERS_WGSL } from "./brick-set.wgsl";
import { brickStaticWGSL } from "./brick-static.wgsl";
import type { SolidOccupancyMaskWalls } from "../../core/solid-occupancy-mask";

/** The static mask the engine binds as `solid` (brickStaticMask's words and walls; brick-engine.ts). */
export interface BrickStaticMaskBinding {
  readonly words: Uint32Array;
  readonly walls: SolidOccupancyMaskWalls;
}

export function brickSolidStampKernels(L: BrickLayout, mask: Pick<BrickStaticMaskBinding, "walls">): { solidStamp: BrickKernel } {
  const bindings: BrickBinding[] = [
    { name: "cfg", kind: "uniform" },
    { name: "set", kind: "rw", type: "array<atomic<u32>>" },
    { name: "dir", kind: "rw", type: "array<atomic<u32>>" },
    { name: "phi", kind: "rw", type: "array<u32>" },
    { name: "cut", kind: "rw", type: "array<f32>" },
    { name: "solid", kind: "ro", type: "array<u32>" },
  ];
  const statics = brickStaticWGSL({ box: L.lattice, openTop: mask.walls.policy === 2, frame: "vec3f(0.0)", h: "vec3f(1.0)", mask: { buffer: "solid", walls: mask.walls } });
  const code = [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: true } }), declareBindings(bindings), BRICK_SET_HELPERS_WGSL, statics, /* wgsl */`
var<workgroup> wCut:atomic<u32>;
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
      h=vec4u(select(1u,0u,(atomicLoad(&dir[r+NB_REC_CLASS])&NB_CLASS_NEAR_BODY)!=0u),s,atomicLoad(&dir[r+NB_REC_KEY]),0u);
    }
    wHead=h;atomicStore(&wCut,0u);
  }
  let h=workgroupUniformLoad(&wHead);
  if(h.x==0u){return;}
  let s=h.y;let r=NB_D_FREC+32u*s;let o=8*nbKeyCoord(h.z);
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
  return { solidStamp: { label: "solidStamp", entry: "solidStamp", code, bindings } };
}
