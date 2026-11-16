/** Lane P's body stages in the brick engine (M5 design note, docs/plans/nbflip-port/m5-bodies-design-2026-11-17.md
 * §3, §4, §7.2; amendments A2-A6): the `bodies` / `bodyCuts` bindings, setMark's body criterion (pbBodyMarkTile, A2)
 * and the bodyStamp kernel at the stamp stage (A6).
 *
 * Bindings (A3): `bodies` is the rigid state buffer itself (WebGPURigidBodySystem's, 24 records of 128 B,
 * webgpu-rigid-body.ts:73-92), read-only; `bodyCuts` the body-local cut atlas (packRigidBodyCutAtlas, solid-cuts.ts),
 * read-only, uploaded on change after pbCheckCutAtlas. The ring's words 64-68 are the header (RING.BODY_*); FLAGS bit
 * BODIES gates every body test (A3: E's ring word 3 bit 1).
 *
 * bodyStamp: a workgroup a brick (G1), 128 lanes.
 * - cfg.z = 0, over F_BODY (NEAR_BODY | BODY_WAS): a NEAR_BODY brick's BMASK is the bodies whose bounding sphere,
 *   dilated by length(2h), one cell and the swept reach (|v| + |w| r) dt, meets the brick's box (pbBrickCandidates);
 *   a BODY_WAS-only brick is the clean-up pass, BMASK 0: the static values and W = 0 (bitwise the static hooks').
 * - cfg.z = 1, over F_ENTER (A2's entry path): an entering brick whose BMASK is not 0 becomes NEAR_BODY and is stamped
 *   in the same step, so no brick enters open inside a body. It runs after solidStamp, so the brick's CUT_STATIC (and
 *   with it the K slot's hold once the body leaves) is solidStamp's own.
 * Each stamped brick takes its K slot (nbKTake, A5: POOL_K latched and the brick left uncut when the pool is out), and
 * the stamp writes the whole slot: BMASK, FRAC, AP0-2, W0-2 (A1), and FLAGS.OPEN (A8) on its cells.
 * Latches (FAIL.UNSUPPORTED, info tag | value): 0xb1 a candidate body's shape tag is not stamped (a rounded plate or
 * an unknown tag; info the body); 0xb2 a body's cut header is out of range (info the body); 0xb3 (setMark, tiered) a
 * marked tile is not h-owned (info the brick key). */
import type { BrickLayout } from "./brick-abi";
import type { BrickBinding, BrickKernel } from "./brick-kernel";
import { brickAbiWGSL } from "./brick-abi";
import { CFG, declareBindings } from "./brick-kernel";
import { BRICK_SET_HELPERS_WGSL } from "./brick-set.wgsl";
import { brickStaticWGSL } from "./brick-static.wgsl";
import { pbBodyStampWGSL } from "./bodies-shapes.wgsl";
import { BODIES_RO, BODY_CUTS_RO, pbBrickBodiesWGSL, pbWgslVec3, type BrickBodiesFrame } from "./bodies-binding.wgsl";
import { pbStampBrickWGSL } from "./bodies-kernels.wgsl";
import type { SolidOccupancyMaskWalls } from "../../core/solid-occupancy-mask";

/** The static solid the stamp reads (E's hooks, A7): the box rule and, when the scene binds it, the voxel mask. */
export interface BrickBodyStatics {
  readonly openTop: boolean;
  readonly mask?: { readonly walls: SolidOccupancyMaskWalls };
}

export function brickBodyKernels(L: BrickLayout, frame: BrickBodiesFrame, statics: BrickBodyStatics): { bodyStamp: BrickKernel } {
  const bindings: BrickBinding[] = [
    CFG,
    { name: "set", kind: "rw", type: "array<atomic<u32>>" },
    { name: "dir", kind: "rw", type: "array<atomic<u32>>" },
    { name: "phi", kind: "rw", type: "array<u32>" },
    { name: "cut", kind: "rw", type: "array<f32>" },
    BODIES_RO, BODY_CUTS_RO,
    ...(statics.mask ? [{ name: "solid", kind: "ro", type: "array<u32>" } as BrickBinding] : []),
  ];
  const h = pbWgslVec3([frame.h_m, frame.h_m, frame.h_m]);
  // The hooks in world metres (solidStamp's own run in cells: frame 0, h 1; nbStaticSolidAt is the only one that
  // takes a world point, and the body queries ask it in metres).
  const hooks = brickStaticWGSL({ box: L.lattice, openTop: statics.openTop, frame: pbWgslVec3(frame.frame_m), h, ...(statics.mask ? { mask: { buffer: "solid", walls: statics.mask.walls } } : {}) });
  const code = [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: true } }), declareBindings(bindings), BRICK_SET_HELPERS_WGSL, hooks,
    pbBrickBodiesWGSL(frame), pbBodyStampWGSL(), /* wgsl */`
var<private> bsSlot:u32;
var<private> bsK:u32;
fn bsPut(l:u32,ch:u32,v:f32){
 var b=NB_CUT_FRAC;
 if(ch==1u){b=NB_CUT_AP0;}else if(ch==2u){b=NB_CUT_AP1;}else if(ch==3u){b=NB_CUT_AP2;}
 else if(ch==4u){b=NB_CUT_W0;}else if(ch==5u){b=NB_CUT_W1;}else if(ch==6u){b=NB_CUT_W2;}
 cut[b+NB_STRIDE*bsK+l]=v;
}
fn bsOpen(l:u32,open:bool){let f=NB_PHI_F_FLAGS+NB_STRIDE*bsSlot+l;phi[f]=select(phi[f]&~NB_FLAG_OPEN,phi[f]|NB_FLAG_OPEN,open);}
${pbStampBrickWGSL({ putCut: "bsPut", putOpen: "bsOpen" })}
// The brick's BMASK (§3): pbBodyBoxCandidates with each body's own reach, length(2h) + one cell + its swept reach
// (|v| + |w| r) horizon: the mark's margin, so every body setMark marked the brick for is a candidate.
fn pbBrickCandidates(o:vec3i)->u32{
 let h=pbH();let lo=pbWorldCell(o)-0.5*h;let hi=lo+8.0*h;let cell=max(h.x,max(h.y,h.z));let dt=pbBrickHorizon();
 var mask=0u;
 for(var i=0u;i<pbBodyCount();i++){
  let b=bodies[i];let c=b.positionShape.xyz;let radius=max(b.dimensions.w,0.0);
  let reach=length(2.0*h)+cell+(length(b.linearVelocity.xyz)+length(b.angularVelocity.xyz)*radius)*dt;
  if(b.positionShape.w<0.0||distance(clamp(c,lo,hi),c)<=radius+reach){mask|=1u<<i;}
 }
 return mask;
}
var<workgroup> wHead:vec4u;
@compute @workgroup_size(128) fn bodyStamp(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  // Lane 0 decides the brick and takes its slot (one nbKTake caller a brick); x stamped, y F slot, z K slot, w BMASK.
  if(lid==0u){
    var head=vec4u(0u,0u,NB_NONE,0u);
    let entry=cfg.z==1u;let list=select(NB_LIST_F_BODY,NB_LIST_F_ENTER,entry);let e=nbGroup(wid);
    if(e<nbListCount(list)){
      let s=nbListSlot(list,e);let r=NB_D_FREC+32u*s;let cls=atomicLoad(&dir[r+NB_REC_CLASS]);
      var go=false;var mask=0u;
      if(entry){
        // A skirt brick (no cell in the lattice) is never a body brick (pbBrickMarkTile's rule).
        if((cls&NB_CLASS_NEAR_BODY)==0u&&nbInLattice(nbFOrigin(s))){mask=pbBrickCandidates(nbFOrigin(s));go=mask!=0u;}
        if(go){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_NEAR_BODY);}
      }else{
        go=(cls&(NB_CLASS_NEAR_BODY|NB_CLASS_BODY_WAS))!=0u;
        if((cls&NB_CLASS_NEAR_BODY)!=0u){mask=pbBrickCandidates(nbFOrigin(s));}
      }
      var bad=mask;while(bad!=0u){let i=firstTrailingBit(bad);bad&=bad-1u;if(!pbBodyShapeSupported(i)){nbFail(NB_FAIL_UNSUPPORTED,0xb1000000u|i);}}
      if(go){let ks=nbKTake(s);if(ks!=NB_NONE){head=vec4u(1u,s,ks,mask);}}
    }
    wHead=head;
  }
  let head=workgroupUniformLoad(&wHead);
  if(head.x==0u){return;}
  bsSlot=head.y;bsK=head.z;
  if(lid==0u){cut[NB_CUT_BMASK+bsK]=bitcast<f32>(head.w);}
  pbStampBrick(nbFOrigin(head.y),head.w,lid,128u);
}`].join("\n");
  return { bodyStamp: { label: "bodyStamp", entry: "bodyStamp", code, bindings, cfgZ: [0, 1] } };
}
