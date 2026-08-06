/**
 * The hero bonsai's crown, as the scene actually publishes it: a handful of
 * cloud pads, each an `SvoFieldProgram` tape with two scales of packed cellular
 * relief on it.
 *
 * Drawn by `tools/shape-lab.ts canopy`, on the CPU, at no voxel size at all, so
 * the form can be judged before any of it reaches a scene.
 *
 * **This module holds no numbers.** It used to carry its own copy of the tape
 * and its own ladder, and the copy drifted: the lab kept a 48 mm head shell
 * while the scene shipped 13.5, so the lab was drawing a canopy the renderer had
 * never been asked for and every comparison between them was meaningless. The
 * pads and the tape both come from `lib/voxel-scenery/bonsai-canopy-field.ts`
 * now, which is the module the scene reads, so a change to the form is visible
 * here by construction and the two cannot disagree again.
 */
import {
  bonsaiCanopyField,
  bonsaiCanopyPadProgram,
  type BonsaiCanopyField,
} from "../lib/voxel-scenery/bonsai-canopy-field";
import { bonsaiCanopyPads, type BonsaiCanopyPad } from "../lib/voxel-scenery/bonsai-canopy-pads";
import { BONSAI_POND_CANOPY } from "../lib/voxel-scenery/bonsai";
import type { SvoFieldProgram } from "../lib/svo-field-program";
import type { Vec3 } from "../lib/model";

/**
 * The leaf the lab draws against — production's refinement depth 3.
 *
 * The tape's ladder stops where the lattice does, so a lab tracing it at the
 * wrong leaf would draw a form the renderer is never asked for. That is the
 * exact drift this module was rewritten to remove.
 */
const LEAF_M = 0.00625 / 2 ** 3;

export interface PlacedProgram {
  readonly program: SvoFieldProgram;
  readonly at: Vec3;
}

/** The hero specimen's crown, centred on the origin. */
export function heroCanopyPads(): PlacedProgram[] {
  const form = BONSAI_POND_CANOPY;
  const field: BonsaiCanopyField = bonsaiCanopyField(LEAF_M);
  const pads: BonsaiCanopyPad[] = bonsaiCanopyPads({
    crownRadius_m: form.crownRadius_m,
    crownThickness_m: form.crownThickness_m,
    crownDroop: form.crownDroop,
    center_m: { x: 0, y: 0, z: 0 },
    seed: 0x51b0_1a7e,
  });
  return pads.map((pad, index) => ({
    at: pad.center_m,
    program: bonsaiCanopyPadProgram(pad.radius_m, field, (0x51b0_1a7e + 7919 * index) >>> 0),
  }));
}
