/**
 * Legacy box-leaf field-program experiment for `tools/shape-lab.ts canopy`.
 * Kept for comparison; production bonsai now publishes the shared leaf density
 * field in bonsai.ts. Use tools/preview/foliage.ts for production captures.
 */
import {
  bonsaiCanopyField,
  bonsaiCanopyPadProgram,
  type BonsaiCanopyField,
} from "../lib/core/voxel-scenery/bonsai-canopy-field";
import { bonsaiCanopyPads, type BonsaiCanopyPad } from "../lib/core/voxel-scenery/bonsai-canopy-pads";
import { BONSAI_POND_CANOPY } from "../lib/core/voxel-scenery/bonsai";
import type { SvoFieldProgram } from "../lib/svo/features/scene-publication/svo-field-program";
import type { Vec3 } from "../lib/core/model";

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
