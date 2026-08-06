/**
 * Where the cloud pads go.
 *
 * The plate's canopy is five to seven distinct rounded masses with sky and
 * shadow between them, at a range of sizes, sitting on a common top plane and
 * drooping at the rim. This lays that out; `bonsai-canopy-field.ts` says what
 * each one is made of.
 *
 * It is a module of its own rather than a branch inside `bonsai.ts` because the
 * shape lab draws the same pads the scene publishes, and the last time a lab and
 * a scene each kept their own copy of a form they drifted by a factor of three
 * and every comparison between them was worthless.
 *
 * ## Why pads are records and relief is a tape
 *
 * The split is not arbitrary and it is the correction that mattered. A
 * displacement can roughen a surface; it cannot turn one mass into six. The
 * first field canopy put the whole crown in a single ellipsoid and roughened it,
 * and the frame showed a smooth flat disc on a stem at every relief setting that
 * was tried — because the silhouette was the ellipsoid's, and there was only
 * ever one ellipsoid. So the scale at which the object stops being connected is
 * geometry, and everything below it is relief.
 */
import type { Vec3 } from "../model";

export interface BonsaiCanopyPad {
  readonly center_m: Vec3;
  /** The **finished** pad — relief included. The tape insets its own core. */
  readonly radius_m: Vec3;
}

export interface BonsaiCanopyPadRequest {
  readonly crownRadius_m: readonly [number, number];
  readonly crownThickness_m: number;
  readonly crownDroop: number;
  readonly center_m: Vec3;
  readonly seed: number;
}

/**
 * Seven, which is the plate's count and also as many as the crown can hold at a
 * size that still reads as a cloud.
 *
 * The count and the size are one decision: `n` masses of plan share
 * `fill / sqrt(n)` pave a crown, so seven pads on this specimen are 336 mm
 * across and five would be 397. Below about five they stop being pads and become
 * lobes of one shape; above about nine they are the *clump* scale, which the
 * tape already draws for the cost of one op and draws better, because a clump on
 * a tape is packed against its neighbours and a clump as a record is not.
 */
export const CANOPY_PAD_COUNT = 7;

/**
 * **Below** paving, and deliberately.
 *
 * At 1.0 equal ellipses tile the crown's plan; the heads elsewhere in this
 * canopy use 1.20 because they have to be one fused mass. Pads must not fuse —
 * the sky between them is the thing the plate has and the old single lens did
 * not — and the first trace at 1.04 showed why paving is already too much: seven
 * pads covered 108 % of the crown's plan and came back as one continuous slab
 * with a bumpy edge, which is the exact failure the pads were introduced to fix,
 * reached by a different route.
 *
 * 0.92 puts the coverage at about 85 %, which leaves gaps a viewer reads as
 * separate clouds. A canopy is allowed to have sky in it; a canopy with no sky
 * in it is a disc.
 */
export const CANOPY_PAD_FILL = 0.92;

/**
 * A pad's half-thickness as a share of its own plan radius.
 *
 * The plate's pads are roughly 250 wide by 180 tall: near enough round, and
 * nothing like the crown they collectively make up, which is four times wider
 * than it is thick. Deriving a pad's thickness from the *crown's* thickness is
 * what made the first version's masses into plates — 336 mm across and 75 thick,
 * a disc, and seven discs at one height are one disc.
 *
 * The floor under it is the relief rather than the drawing. The tape insets its
 * core by the sum of its shells — about 41 mm — on **every** axis, so the
 * thinnest axis is where that has to fit, and a pad whose half-thickness is 74 mm
 * would be left with a 33 mm core carrying 42 mm of relief. That is the "the tape
 * drew a slab" failure the old single-plate version hit, one scale down.
 *
 * At 0.86 the half-thickness is about 104 mm, the core keeps 62, and even the
 * smallest pad in the set holds a 90 mm-thick core — enough for the 32 mm floret
 * lattice to have three cells across it rather than two. The pads come out very
 * nearly round, which is what the plate's are.
 */
export const CANOPY_PAD_FLATTEN = 0.86;

/** Range the pads are dealt over, as a share either side of the nominal size. */
export const CANOPY_PAD_SWELL = 0.34;

/**
 * How far a pad's own level wanders, as a share of the crown's thickness.
 *
 * This is what makes the crown's upper outline scalloped rather than flat, and
 * it was 0.09 — plus or minus 12 mm on a 135 mm crown, which is less than the
 * spread the size jitter alone produces and invisible beside it. The plate's
 * pads stand at *clearly* different levels; two of them are most of a pad above
 * their neighbours, and that stepping is a good part of what stops the canopy
 * reading as one object.
 */
export const CANOPY_PAD_LIFT = 0.30;

/** Golden angle, so no two pads share a bearing. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * How the pads crowd toward the rim.
 *
 * A third rather than the equal-area half: a canopy seen from the side is mostly
 * its rim, and an equal-area sunflower puts too many masses in a middle that is
 * hidden behind them. The endpoints stay exact — the first pad sits on the
 * crown's own axis and the last on the rim.
 */
const RIM_CROWDING = 1 / 3;

const hash01 = (n: number): number => {
  let h = Math.imul(n ^ 0x27d4_eb2d, 0x9e37_79b1) >>> 0;
  h ^= h >>> 15; h = Math.imul(h, 0x85eb_ca6b) >>> 0; h ^= h >>> 13;
  return (h >>> 0) / 4_294_967_296;
};
const hashSigned = (n: number): number => 2 * hash01(n) - 1;

export function bonsaiCanopyPads(request: BonsaiCanopyPadRequest): BonsaiCanopyPad[] {
  const { crownRadius_m: [crownX, crownZ], crownThickness_m, crownDroop, center_m, seed } = request;
  const planShare = CANOPY_PAD_FILL / Math.sqrt(CANOPY_PAD_COUNT);
  // The plane the pads hang from. Not the mid-plane: a pad is placed by its
  // *top* and grows downward, so the swell that gives the crown its clumped
  // masses thickens the underside instead of doming the top — and a flat top
  // over a rounded underside is the one thing the plate is unambiguous about.
  const topY = center_m.y + 0.5 * crownThickness_m;
  const pads: BonsaiCanopyPad[] = [];
  for (let index = 0; index < CANOPY_PAD_COUNT; index += 1) {
    const swell = 1 - 0.5 * CANOPY_PAD_SWELL + CANOPY_PAD_SWELL * hash01(seed + 43 * index + 1);
    // Clamped at the crown's own radius, and the clamp is the bound rather than
    // a tidy-up: the inset below subtracts the pad's radius from the crown's, so
    // a pad the swell had pushed past the crown would inset to zero, sit on the
    // axis and still draw past the authored silhouette.
    const planX = Math.min(crownX, crownX * planShare * swell * (1 + 0.10 * hashSigned(seed + 43 * index + 2)));
    const planZ = Math.min(crownZ, crownZ * planShare * swell * (1 + 0.10 * hashSigned(seed + 43 * index + 4)));
    // Round, off its own plan radius — never off the crown's thickness. See
    // `CANOPY_PAD_FLATTEN`.
    const halfThickness = CANOPY_PAD_FLATTEN * Math.sqrt(planX * planZ)
      * (0.86 + 0.28 * hash01(seed + 43 * index + 3));
    const u = CANOPY_PAD_COUNT > 1 ? (index / (CANOPY_PAD_COUNT - 1)) ** RIM_CROWDING : 0;
    const angle = GOLDEN_ANGLE * index + 1.31 * hash01(seed + 5);
    // Inset by the radius of the thing it carries, so a pad's far edge lands on
    // the authored silhouette rather than past it.
    const offsetX = Math.max(0, crownX - planX) * u * Math.cos(angle);
    const offsetZ = Math.max(0, crownZ - planZ) * u * Math.sin(angle);
    // The droop follows the pad's own distance out along the crown, normalised
    // by the reach a pad centre has, so a rim pad still gets the full authored
    // droop.
    const out = Math.min(1, Math.hypot(offsetX / crownX, offsetZ / crownZ) / Math.max(1e-6, 1 - planShare));
    pads.push({
      center_m: {
        x: center_m.x + offsetX,
        y: topY - halfThickness
          - crownDroop * crownThickness_m * out * out
          + CANOPY_PAD_LIFT * crownThickness_m * hashSigned(seed + 43 * index),
        z: center_m.z + offsetZ,
      },
      radius_m: { x: planX, y: halfThickness, z: planZ },
    });
  }
  return pads;
}
