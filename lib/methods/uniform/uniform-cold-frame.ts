import { UNIFORM_REGION_TIER } from "./uniform-region-heat";

/** Words of the region heat header (tiers[R..R+5)) a frame receipt carries:
 * live COLD, WARM, HOT, free region counts of the frame head, then the
 * regions a mid-frame hook (remap, page claim) raised. */
export const UNIFORM_COLD_FRAME_HEAT_WORDS=5;
/** The evidence: the heat header words, then the frame's live NB-FLIP
 * particle count (particle allocation seeds census page claims). */
export const UNIFORM_COLD_FRAME_EVIDENCE_WORDS=UNIFORM_COLD_FRAME_HEAT_WORDS+1;

/** Host cold-frame evidence: which region work tiers may be present this
 * frame, so stages whose launches do work only in warmer regions are not
 * encoded at all (each empty launch costs ~12 us GPU + ~7 us host).
 *
 * ceiling: the highest UNIFORM_REGION_TIER any live region may hold this
 * advance, fixed from advance start to end. quiet: ceiling is COLD.
 * Consumers: launches that work only in non-COLD regions may be left out
 * when quiet; launches that work only in HOT regions when ceiling < HOT.
 * A stage that leaves its launches out must hold its retained outputs
 * already (no copies): exactly what the launches leave in an all-COLD frame.
 *
 * Evidence and lag: no blocking readback. Frame N reads the region heat
 * header of frame N-UNIFORM_MIXED_RECEIPT_RING (2), carried by that frame's
 * receipt, which the ring guarantees is checked before N encodes: the same
 * deterministic lag as the pressure plan, so free-running and awaited runs
 * encode alike. A region that heats with no host trigger at frame M (its
 * head's tier rises) is fully encoded from frame M+2 on; frames M and M+1
 * may leave a stage out for it. Remaps and page claims are mid-frame
 * promotions: a frame whose evidence holds one, or live particles (their
 * allocation seeds page claims), is not quiet, and the census keeps the
 * widths of deep-COLD regions (no remap without a warmer tier).
 * Host triggers (every edit, invalidation, relayout attach, capacity change,
 * kick, parameter change, added source volume, drop, externally moved
 * bodies) force HOT for their frame and until a receipt of a frame from the
 * trigger on is the evidence.
 * enabled false (NB-FLIP coldFrameSkip "off"): ceiling is always HOT, so no
 * stage leaves a launch out. */
export class UniformColdFrame {
 constructor(readonly enabled=true){}
 /** UNIFORM_REGION_TIER: the highest tier any live region may hold this advance. */
 ceiling:number=UNIFORM_REGION_TIER.hot;
 get quiet():boolean{return this.ceiling===UNIFORM_REGION_TIER.cold;}
 /** The first frame whose receipt reflects the last host trigger. */
 private triggered=0;
 /** A host trigger pending for the next advance (between frames). */
 private pending=true;
 /** Any host state change between frames: the next advance and every
  * advance until its own receipt is the evidence encode in full. */
 trigger():void{this.pending=true;}
 /** At an advance's start. frame: its index (a kick passes its frame's,
  * which also triggers). evidence: frame evidenceFrame's header words, or
  * undefined when no checked receipt of that frame exists. trigger: a host
  * trigger this advance knows itself. */
 begin(frame:number,evidenceFrame:number,evidence:Uint32Array|undefined,trigger:boolean):void{
  if(trigger||this.pending){this.pending=false;this.triggered=frame;}
  const hot=UNIFORM_REGION_TIER.hot;
  if(!this.enabled||!evidence||evidenceFrame<this.triggered||evidence[4]!==0||evidence[5]!==0){this.ceiling=hot;return;}
  this.ceiling=evidence[2]!==0?hot:evidence[1]!==0?UNIFORM_REGION_TIER.warm:UNIFORM_REGION_TIER.cold;
 }
}
