/** Lane P's per-brick body passes for the brick engine (M5 design note, docs/plans/nbflip-port/m5-bodies-design-2026-11-17.md
 * §2.3, §3, §6): bodyStamp's FRAC / AP / W / OPEN over one NEAR_BODY (or BODY_WAS) brick, and couple's per-brick
 * reduction into the rigid exchange. Both are built on bodies-shapes.wgsl.ts (the verbatim old formulas) and fix no
 * binding: the kernel supplies small sink and source functions, so the same text runs in the engine (on E's cut slot,
 * amendments A1/A5/A6) and in the parity tests.
 *
 * Layout (brick-abi.md §2.3 with A1): a channel word is a cell of the brick (nbLocal order, l = x + 8y + 64z); the
 * AP_a and W_a words of cell c are its lower face along a, the face between c - e_a and c. The shared module's face
 * functions take the face on the +a side of a cell, so the lower face of c is evaluated at c - e_a.
 *
 * Masks: every query takes the brick's BMASK (pbBodyBoxCandidates of the brick's box dilated by at least length(2h)).
 * The cells and faces bodyStamp evaluates lie within one cell of the box, and every old reach test is inside that
 * dilation, so the values equal the all-bodies ones bit for bit (bodies-shapes R6 checks both masks). */
import { GPU_RIGID_BODY_CAPACITY } from "../../core/rigid-capacity";
import { GPU_RIGID_EXCHANGE_WORDS } from "../../core/webgpu-eulerian";

/** The cut channels a stamp writes, in the order of pbStampBrick's `ch` argument (A1: POOLS.cut FRAC, AP0..2, W0..2). */
export const PB_CUT_CHANNELS = ["FRAC", "AP0", "AP1", "AP2", "W0", "W1", "W2"] as const;

/** Where a stamp writes. Each is the name of a WGSL function the kernel defines. */
export interface PbStampSink {
  /** `(l:u32,ch:u32,v:f32)`: cut channel ch (PB_CUT_CHANNELS) of the brick's cell l. */
  readonly putCut: string;
  /** `(l:u32,open:bool)`: FLAGS.OPEN of the brick's cell l (FRAC > 1e-5, the umCellIsOpen predicate). */
  readonly putOpen: string;
}

/** bodyStamp over one brick (§2.3): cells l = lane, lane + lanes, ... of the brick with origin o, for a NEAR_BODY brick
 * with candidate mask `mask`, or for a BODY_WAS clean-up pass with mask 0 (the static values, W = 0: the body branch of
 * every formula is then skipped, and the value is the static hook's, as solidStamp writes it). Needs pbBodyShapesWGSL
 * and pbBodyStampWGSL. */
export function pbStampBrickWGSL(sink: PbStampSink): string {
  return /* wgsl */`
fn pbBrickCell(l:u32)->vec3i{return vec3i(i32(l&7u),i32((l>>3u)&7u),i32(l>>6u));}
fn pbStampBrick(o:vec3i,mask:u32,lane:u32,lanes:u32){
 for(var l=lane;l<512u;l+=lanes){
  let c=o+pbBrickCell(l);
  let frac=pbStampCell(c,mask);
  ${sink.putCut}(l,0u,frac);${sink.putOpen}(l,frac>1e-5);
  for(var a=0u;a<3u;a++){
   // The lower face of c is the +a face of q = c - e_a; its two cells' FRAC are q's and c's (each the same function
   // of the same mask, so a neighbour brick's own stamp of q is this value).
   let q=c-pbEi(a);
   ${sink.putCut}(l,1u+a,pbStampFace(q,a,mask,pbStampCell(q,mask),frac));
   ${sink.putCut}(l,4u+a,pbStampFaceVelocity(q,a,mask));
  }
 }
}`;
}

/** What couple reads. Each is the name of a WGSL function the kernel defines. */
export interface PbCoupleSource {
  /** `(l:u32)->f32`: the brick's stamped FRAC at cell l (couple's p is 0 where FRAC <= 1e-5, umBodyPressure). */
  readonly frac: string;
  /** `(c:vec3i,l:u32)->f32`: p (Pa) = X rho / dt at a liquid h row, 0 at any other cell. */
  readonly pressure: string;
  /** `(c:vec3i,l:u32)->f32`: the cell's wet fraction, umBodyOwnerWet: in the engine pbOwnerWet of the sum of its 8
   * corner phi (in h). */
  readonly wet: string;
}

/** The rigid exchange's words a body (webgpu-eulerian.ts GPU_RIGID_EXCHANGE_WORDS): 0-5 reaction and torque x1e6, 6 the
 * wet weight x65536, 10 the near flag. */
export const PB_EXCHANGE_WORDS = GPU_RIGID_EXCHANGE_WORDS;

/** couple over one brick (§6), in two calls with a workgroup barrier between them:
 * - pbCoupleBrick(o, mask, lane, lanes, dt): each cell's pbCoupleBody for each body of mask, rounded per cell to the
 *   exchange's fixed point as the old couple does (webgpu-eulerian.ts), summed into workgroup atomics (integer sums, so
 *   the brick's total is the old per-cell atomics' total bit for bit);
 * - pbCoupleFlush(mask, lane, lanes): one atomicAdd a nonzero word into `exchange` (the name of an
 *   `array<atomic<i32>>` storage variable, GPU_RIGID_EXCHANGE_WORDS a body; cleared before couple), and word 10 = 1
 *   for a near body.
 * pbCoupleClear(lane, lanes) zeroes the workgroup words first (then a barrier). Needs pbBodyShapesWGSL and
 * pbBodyCoupleWGSL. */
export function pbCoupleBrickWGSL(src: PbCoupleSource, exchange: string): string {
  const n = GPU_RIGID_BODY_CAPACITY;
  return /* wgsl */`
var<workgroup> pbXw:array<atomic<i32>,${7 * n}>;
var<workgroup> pbXnear:atomic<u32>;
fn pbCoupleClear(lane:u32,lanes:u32){
 for(var i=lane;i<${7 * n}u;i+=lanes){atomicStore(&pbXw[i],0);}
 if(lane==0u){atomicStore(&pbXnear,0u);}
}
fn pbCoupleBrick(o:vec3i,mask:u32,lane:u32,lanes:u32,dt:f32){
 let bodies=mask&pbBodyMaskAll();
 if(bodies==0u){return;}
 for(var l=lane;l<512u;l+=lanes){
  let c=o+pbBrickCell(l);
  let p=select(${src.pressure}(c,l),0.0,${src.frac}(l)<=1e-5);
  let wet=${src.wet}(c,l);
  var remaining=bodies;
  while(remaining!=0u){
   let i=firstTrailingBit(remaining);remaining&=remaining-1u;
   let r=pbCoupleBody(i,c,p,wet,dt);
   if(!r.near){continue;}
   atomicOr(&pbXnear,1u<<i);
   let w=7u*i;
   let x=vec3i(pbFixed(r.reaction.x),pbFixed(r.reaction.y),pbFixed(r.reaction.z));
   let t=vec3i(pbFixed(r.torque.x),pbFixed(r.torque.y),pbFixed(r.torque.z));
   if(x.x!=0){atomicAdd(&pbXw[w],x.x);}if(x.y!=0){atomicAdd(&pbXw[w+1u],x.y);}if(x.z!=0){atomicAdd(&pbXw[w+2u],x.z);}
   if(t.x!=0){atomicAdd(&pbXw[w+3u],t.x);}if(t.y!=0){atomicAdd(&pbXw[w+4u],t.y);}if(t.z!=0){atomicAdd(&pbXw[w+5u],t.z);}
   let q=pbFixedWet(r.wet);if(q!=0){atomicAdd(&pbXw[w+6u],q);}
  }
 }
}
fn pbCoupleFlush(mask:u32,lane:u32,lanes:u32){
 let near=atomicLoad(&pbXnear)&mask;
 for(var k=lane;k<${7 * n}u;k+=lanes){
  let i=k/7u;let word=k%7u;
  if(((near>>i)&1u)==0u){continue;}
  let v=atomicLoad(&pbXw[k]);
  if(v!=0){atomicAdd(&${exchange}[${GPU_RIGID_EXCHANGE_WORDS}u*i+word],v);}
  if(word==0u){atomicStore(&${exchange}[${GPU_RIGID_EXCHANGE_WORDS}u*i+10u],1);}
 }
}`;
}
