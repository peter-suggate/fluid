/** Lane P's body stages in the brick engine (M5 design note, docs/plans/nbflip-port/m5-bodies-design-2026-11-17.md
 * §3, §4, §7.2; amendments A2-A6): the `bodies` / `bodyCuts` bindings, setMark's body criterion (pbBodyMarkTile, A2)
 * and the bodyStamp kernel at the stamp stage (A6).
 *
 * Bindings (A3): `bodies` is the rigid state buffer itself (WebGPURigidBodySystem's, 24 records of 128 B,
 * webgpu-rigid-body.ts:73-92), read-only; `bodyCuts` the body-local cut atlas (packRigidBodyCutAtlas, solid-cuts.ts),
 * read-only, uploaded on change after pbCheckCutAtlas. The ring's words 64-68 are the header (RING.BODY_*); FLAGS bit
 * BODIES gates every body test (A3: E's ring word 3 bit 1).
 *
 * bodyStamp: a workgroup a brick (G1), 512 lanes, a cell a lane (2026-11-17: at 128 lanes it was 0.50-0.59 ms a step
 *   on the rigid hydrostatic and sink at h = 0.025, at 512 0.20-0.26 ms, the words bitwise the same; few bricks are
 *   stamped, so the brick's latency is the stage's).
 * - cfg.z = 0, over F_BODY (NEAR_BODY | BODY_WAS): a NEAR_BODY brick's BMASK is the bodies whose bounding sphere,
 *   dilated by length(2h), one cell and the swept reach (|v| + |w| r) dt, meets the brick's box (pbBrickCandidates);
 *   a BODY_WAS-only brick is the clean-up pass, BMASK 0: the static values and W = 0 (bitwise the static hooks').
 * - cfg.z = 1, over F_ENTER (A2's entry path): an entering brick with an h tile whose BMASK is not 0 becomes NEAR_BODY
 *   and is stamped in the same step, so no brick enters open inside a body. One with no h tile (part 2's claims start
 *   4h) waits for the census to hold it h; a tile of it the body marks latches 0xb3 at setMark, never open inside. It runs after solidStamp, so the brick's CUT_STATIC (and
 *   with it the K slot's hold once the body leaves) is solidStamp's own.
 * Each stamped brick takes its K slot (nbKTake, A5: POOL_K latched and the brick left uncut when the pool is out), and
 * the stamp writes the whole slot: BMASK, FRAC, AP0-2, W0-2 (A1), and FLAGS.OPEN (A8) on its cells.
 * Latches (FAIL.UNSUPPORTED, info tag | value): 0xb1 a candidate body's shape tag is not stamped (a rounded plate or
 * an unknown tag; info the body); 0xb2 a body's cut header is out of range (info the body); 0xb3 (setMark, tiered) a
 * marked tile is not h-owned (info the brick key).
 *
 * bodyCouple (§6, brickBodyCoupleKernel): part 2, after F's tail (X final), over F_BODY: a workgroup a brick whose K slot
 * the head stamped with BMASK != 0 (every such brick is NEAR_BODY or BODY_WAS at part 2's setMark, so F_BODY holds
 * it). The pressure is the old couple's p = X rho / dt (the slot contract's X = p dt / rho), 0 where the stamp's
 * FRAC <= 1e-5 (pbCoupleBrick) and off the liquid rows (X is 0 there); the wet weight is pbOwnerWet of the cell's 8
 * corner phi. The words go into the rigid exchange (`bodyExchange`, the rigid system's, GPU_RIGID_EXCHANGE_WORDS a
 * body), which the host clears after the integrator reads it (BrickBodiesBinding.integrate). Gated on the ring's
 * BODIES flag and BODY_FLAG.COUPLING. */
import type { BrickLayout } from "./brick-abi";
import type { BrickBinding, BrickKernel } from "./brick-kernel";
import { brickAbiWGSL } from "./brick-abi";
import { CFG, declareBindings } from "./brick-kernel";
import { BRICK_SET_HELPERS_WGSL } from "./brick-set.wgsl";
import { brickStaticWGSL } from "./brick-static.wgsl";
import { pbBodyCoupleWGSL, pbBodyStampWGSL } from "./bodies-shapes.wgsl";
import { BODIES_RO, BODY_CUTS_RO, pbBrickBodiesWGSL, pbWgslVec3, type BrickBodiesFrame, PB_BRICK_CANDIDATES_WGSL } from "./bodies-binding.wgsl";
import { pbCoupleBrickWGSL, pbStampBrickWGSL } from "./bodies-kernels.wgsl";
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
  // takes a world point, and the body queries ask it in metres). The box is the container's (L.box, as solidStamp's):
  // on a padded lattice the cells past it are wall, not open (unpadded, L.box is the lattice and the text the same).
  const hooks = brickStaticWGSL({ box: L.box, openTop: statics.openTop, frame: pbWgslVec3(frame.frame_m), h, ...(statics.mask ? { mask: { buffer: "solid", walls: statics.mask.walls } } : {}) });
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
${PB_BRICK_CANDIDATES_WGSL}
var<workgroup> wHead:vec4u;
@compute @workgroup_size(512) fn bodyStamp(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  // Lane 0 decides the brick and takes its slot (one nbKTake caller a brick); x stamped, y F slot, z K slot, w BMASK.
  if(lid==0u){
    var head=vec4u(0u,0u,NB_NONE,0u);
    let entry=cfg.z==1u;let list=select(NB_LIST_F_BODY,NB_LIST_F_ENTER,entry);let e=nbGroup(wid);
    if(e<nbListCount(list)){
      let s=nbListSlot(list,e);let r=NB_D_FREC+32u*s;let cls=atomicLoad(&dir[r+NB_REC_CLASS]);
      var go=false;var mask=0u;
      if(entry){
        // A skirt brick (no cell in the lattice) is never a body brick (pbBrickMarkTile's rule). Nor is a brick with no
        // h tile (REC.MASK): one part 2 claimed starts 4h (setClaimBricks), after the census that would hold it h, and
        // a body brick's tiles are h (the pressure latches one that is not). The census holds it h from its next head
        // (cdBodyHeld), and a tile the body marks before then latches 0xb3 in setMark. Backyard's board bricks under the
        // resting ball entered so and latched UNSUPPORTED at frame 1 (2026-11-18).
        let hTiles=(atomicLoad(&dir[r+NB_REC_MASK])&0xffu)!=0u;
        if((cls&NB_CLASS_NEAR_BODY)==0u&&hTiles&&nbInLattice(nbFOrigin(s))){mask=pbBrickCandidates(nbFOrigin(s));go=mask!=0u;}
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
  pbStampBrick(nbFOrigin(head.y),head.w,lid,512u);
  // FLAGS.STATIC_SHUT (brick-abi.md §2.4): the cell's static open fraction alone, so the pressure tells a static solid
  // cell from a body's (poStaticShut). The lane's own cell, as pbStampBrick's FLAGS.OPEN.
  {let c=nbFOrigin(head.y)+nbLocalCoord(lid);let f=NB_PHI_F_FLAGS+NB_STRIDE*bsSlot+lid;
   phi[f]=select(phi[f]&~NB_FLAG_STATIC_SHUT,phi[f]|NB_FLAG_STATIC_SHUT,nbStaticCellOpen(c)<=1e-5);}
}`].join("\n");
  return { bodyStamp: { label: "bodyStamp", entry: "bodyStamp", code, bindings, cfgZ: [0, 1] } };
}

/** The fluid numbers couple needs: the density (p = X rho / dt). */
export interface BrickBodyFluid { readonly density_kg_m3: number }

/** The rigid exchange binding (the rigid system's buffer: GPU_RIGID_EXCHANGE_WORDS i32 a body, atomics). */
export const BODY_EXCHANGE_RW: BrickBinding = { name: "bodyExchange", kind: "rw", type: "array<atomic<i32>>" };

export function brickBodyCoupleKernel(L: BrickLayout, frame: BrickBodiesFrame, fluid: BrickBodyFluid): BrickKernel {
  if (!(Number.isFinite(fluid.density_kg_m3) && fluid.density_kg_m3 > 0)) throw new Error(`bodyCouple: density ${fluid.density_kg_m3} is not a fluid density`);
  const bindings: BrickBinding[] = [
    CFG,
    { name: "set", kind: "rw", type: "array<atomic<u32>>" },
    { name: "dir", kind: "ro", type: "array<u32>" },
    { name: "phi", kind: "ro", type: "array<u32>" },
    { name: "cut", kind: "ro", type: "array<f32>" },
    { name: "pres", kind: "ro", type: "array<f32>" },
    BODIES_RO, BODY_CUTS_RO, BODY_EXCHANGE_RW,
  ];
  const rho = Math.fround(fluid.density_kg_m3).toPrecision(9);
  const code = [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: false } }), declareBindings(bindings),
    pbBrickBodiesWGSL(frame), pbBodyCoupleWGSL, /* wgsl */`
const BC_RHO:f32=${/[.e]/.test(rho) ? rho : `${rho}.0`};
var<private> bcS:u32;
var<private> bcK:u32;
var<private> bcScale:f32;
fn bcFrac(l:u32)->f32{return cut[NB_CUT_FRAC+NB_STRIDE*bcK+l];}
fn bcPressure(c:vec3i,l:u32)->f32{return pres[NB_PRES_F_X+NB_STRIDE*bcS+l]*bcScale;}
// A vertex's phi (h) as nbVertexPhi reads an h-owned vertex (every vertex of a body brick's cells is: §4 keeps the
// body tiles and their one-tile ring h, and setMark latches 0xb3 otherwise): NB_PHI_FAR outside the lattice's
// vertices, NB_PHI_AIR_C where no F slot holds it.
fn bcVertex(v:vec3i)->f32{
 if(any(v<vec3i(0))||any(v>NB_LATTICE)){return NB_PHI_FAR;}
 let s=nbFSlotOfBrick(v>>vec3u(3u));if(s>=NB_SEAM){return NB_PHI_AIR_C;}
 return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
}
// umBodyOwnerWet's corner sum, in umCorner's order.
fn bcWet(c:vec3i,l:u32)->f32{
 var sum=0.0;
 for(var k=0u;k<8u;k++){sum+=bcVertex(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
 return pbOwnerWet(sum);
}
// pbStampBrickWGSL's cell of a brick lane (pbCoupleBrick reads it; this kernel has no stamp).
fn pbBrickCell(l:u32)->vec3i{return vec3i(i32(l&7u),i32((l>>3u)&7u),i32(l>>6u));}
${pbCoupleBrickWGSL({ frac: "bcFrac", pressure: "bcPressure", wet: "bcWet" }, "bodyExchange")}
var<workgroup> bcHead:vec4u;
@compute @workgroup_size(128) fn bodyCouple(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  // Lane 0: x coupled, y F slot, z K slot, w BMASK (the head's stamp).
  if(lid==0u){
    var head=vec4u(0u,0u,NB_NONE,0u);
    let e=nbGroup(wid);
    let on=pbBrickBodiesOn()&&(nbRing(cfg.x,NB_RING_BODY_FLAGS)&NB_BODY_FLAG_COUPLING)!=0u;
    if(on&&e<nbListCount(NB_LIST_F_BODY)){
      let s=nbListSlot(NB_LIST_F_BODY,e);let k=nbFRec(s,NB_REC_CUT);
      if(k!=NB_NONE){let mask=bitcast<u32>(cut[NB_CUT_BMASK+k]);if(mask!=0u){head=vec4u(1u,s,k,mask);}}
    }
    bcHead=head;
  }
  pbCoupleClear(lid,128u);
  let head=workgroupUniformLoad(&bcHead);
  if(head.x==0u){return;}
  bcS=head.y;bcK=head.z;
  let dt=nbRingF(cfg.x,NB_RING_DT);bcScale=BC_RHO/dt;
  pbCoupleBrick(nbFOrigin(head.y),head.w,lid,128u,dt);
  workgroupBarrier();
  pbCoupleFlush(head.w,lid,128u);
}`].join("\n");
  return { label: "bodyCouple", entry: "bodyCouple", code, bindings };
}
