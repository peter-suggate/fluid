import type { Vec3 } from "../model";
import type { SceneryMaterial, SceneryNode } from "../scenery-graph";
import { alongAxis, V } from "./builder";

/**
 * The planting: air plants and grass tufts, as one rosette generator.
 *
 * Reference: `output/imagegen/garden-pond-hose-fill-simplified.png`. Three of
 * these sit in the frame — one wedged behind a pebble at the near left, one at
 * the back beside the tree, one on the right rim — and they are the same plant
 * at two settings. The right and back ones are air plants: nine or so fat
 * blades, half of them arced right over past horizontal. The left one is
 * grassier, fifteen narrow blades on a tighter fan. Nothing else in the
 * reference is planting, which is why this file exposes two forms and no third.
 *
 * The whole object is 6-9 cm across in a 1.8 m frame, so its entire job is to
 * read as *grown* at a size where it is barely a hundred pixels tall. Two things
 * carry that, and neither is detail:
 *
 *  - **The blade is a curve, and the catalog only has straight primitives.** A
 *    blade that leaves the neck steeply and finishes pointing outward and down
 *    turns through most of a right angle, and a chain of cones has to spend a
 *    segment on every part of that turn the eye can see. `segments` is that
 *    price, paid per blade, and it is the one number in this file that was
 *    settled against renders rather than reasoned about — see the note on
 *    `RosetteForm.segments`.
 *  - **A rosette is a phyllotaxis, not an arrangement.** Blades leave on a
 *    golden-angle spiral and recurve further the older they are, so the upright
 *    young ones scatter through a fan of fallen old ones. Placing the upright
 *    blades in the middle and the flat ones outside — the obvious construction —
 *    produces a shuttlecock, which is what an arrangement of cones looks like.
 *
 * What is *not* the constraint, having been measured: the 25 mm lattice. A 9 mm
 * blade is a third of a cell, and gap G2 of the plan says a primitive thinner
 * than a voxel can lose that voxel to a neighbour and vanish — so the ladder in
 * `tools/preview/rosette.ts` was rendered down to a 2.4 mm blade through both
 * the megakernel, which resolves a surface through the per-voxel material owner,
 * and `raster-primary`, which draws a conservative box per authored record and
 * never consults an owner at all. The two frames agree to 2/255 over the
 * thinnest plants in the ladder, break-up and all. What breaks a blade up is
 * therefore screen resolution and not ownership: a blade holds together while
 * its projected diameter is about three pixels and dashes below two, which at
 * the hero camera is 13 mm and 9 mm of blade at a 520-line frame, and 4.5 mm and
 * 3 mm at 1500. That is why `halfWidth_m` is authored where the reference puts
 * it rather than being inflated to clear a cell: the cell was never the floor.
 *
 * Determinism is the same contract the procedural tree keeps: the seed is the
 * only entropy, it enters through an integer avalanche hash, and the same spec
 * grows the same plant on every rebuild — the sparse publication cache is keyed
 * on the geometry that comes out of here.
 */

/**
 * What makes an air plant an air plant and a tuft a tuft: the shape, with no
 * position, seed or key in it, so a set can be planted from one form.
 */
export interface RosetteForm {
  /** Blades springing from the neck. Nine reads as an air plant, fifteen as grass. */
  readonly bladeCount: number;
  /** The longest blade, root to tip, along its own arc rather than as a height. */
  readonly length_m: number;
  /** Half-width of a blade at its widest, which is near the base. */
  readonly halfWidth_m: number;
  /**
   * Cone segments per blade.
   *
   * The interesting number here, and the one settled against renders rather
   * than reasoned about — `ROSETTE_PREVIEW=segments` is the ladder that settled
   * it, one to six on one seed. What the eye reads is not the chord sag, which
   * is under a millimetre everywhere: it is the **kink at the widest joint**,
   * and because `recurve` pushes the turn outward that joint is always the last
   * one. Measured on the air plant, whose blades turn 100 degrees from neck to
   * tip:
   *
   * | segments | last kink | reads as |
   * |---:|---:|---|
   * | 1 | — | a starburst; no recurve exists at all |
   * | 2 | 66 deg | two sticks and an elbow |
   * | 3 | 52 deg | an arc with one obvious hinge in the outer blades |
   * | 4 | 41 deg | curved, with a faint kink left on the outermost blade |
   * | 5 | 34 deg | clean |
   * | 6 | 29 deg | indistinguishable from five |
   *
   * So the threshold is a joint of about **35 degrees**, and the segment count a
   * blade needs is its own turn divided by that — not a constant. The air plant
   * turns furthest and pays five; the tuft turns 20 degrees less and pays four.
   * Six is refused not because it looks worse but because a sixth of the whole
   * planting budget bought nothing anyone could point at.
   */
  readonly segments: number;
  /** Tilt off vertical the *oldest* blade's tip reaches. Past pi/2 is a recurve. */
  readonly splay_rad: number;
  /**
   * How much of a blade's tilt it already has where it leaves the neck, as a
   * fraction of its own tip tilt. This is what separates a rosette from a
   * fountain: at 0 every blade starts vertical and the base becomes a hard knot
   * of parallel cones, and at 1 the blade is straight and the recurve is gone.
   */
  readonly launchShare: number;
  /**
   * How the turn is distributed along the blade. Above 1 it accumulates toward
   * the tip, which is what a recurve *is* — a blade that runs nearly straight
   * out of the neck and then curls over its last third.
   */
  readonly recurve: number;
  /** Clay-palette value at the blade's shoulder; base and tip ramp off it. */
  readonly value: number;
}

export interface RosetteSpec extends RosetteForm {
  /** Key prefix. Every emitted node is published as `${key}/...`. */
  readonly key: string;
  /** Where the plant stands, in world metres on the ground plane. */
  readonly at_m: readonly [number, number];
  /**
   * Ground height at a world point. The plan curve's own height function for a
   * pond bank, `terrainHeightAt` for anything else — the generator never assumes
   * a datum, because every one of the reference's plants is wedged into a slope
   * or a stone rather than standing on a lawn.
   */
  readonly groundHeightAt: (x_m: number, z_m: number) => number;
  /**
   * How far the neck is pushed under the ground. Defaults to most of the neck,
   * because none of the reference's three plants shows a base: each is jammed
   * into a gap between the coping and a pebble, and a rosette resting *on* a
   * surface reads as a cut flower laid down.
   */
  readonly bed_m?: number;
  /** The only entropy. Any integer. */
  readonly seed: number;
}

/** One tapered run of a blade, in world metres. */
export interface RosetteSegment {
  readonly from_m: Vec3;
  readonly to_m: Vec3;
  readonly fromRadius_m: number;
  readonly toRadius_m: number;
  readonly value: number;
}

export interface RosetteBlade {
  /** Position on the golden-angle spiral: 0 is the oldest and most recurved. */
  readonly order: number;
  readonly azimuth_rad: number;
  readonly length_m: number;
  readonly launchTilt_rad: number;
  readonly tipTilt_rad: number;
  readonly segments: readonly RosetteSegment[];
}

export interface RosettePlan {
  readonly spec: RosetteSpec;
  /** Where the neck meets the ground, in world metres, before bedding. */
  readonly root_m: Vec3;
  readonly neckCenter_m: Vec3;
  readonly neckRadius_m: Vec3;
  readonly blades: readonly RosetteBlade[];
  /** Widest horizontal reach of any blade tip from `root_m`. */
  readonly radius_m: number;
  /** Highest point above `root_m`. */
  readonly height_m: number;
  /** Primitives this plan publishes: every blade segment, plus the neck. */
  readonly leafCount: number;
}

/**
 * The air plant of the back and right banks: nine fat blades, most of them
 * fallen past horizontal, on a fan barely taller than it is wide.
 */
export const ROSETTE_AIR_PLANT: RosetteForm = Object.freeze({
  bladeCount: 9,
  length_m: 0.060,
  halfWidth_m: 0.0045,
  segments: 5,
  splay_rad: 2.20,
  launchShare: 0.22,
  recurve: 1.85,
  value: 0.88,
});

/**
 * The near-left plant: two thirds the blade width, half as many again of them,
 * and a longer run, so the same generator reads as grass rather than as a bigger
 * air plant. Width relative to length is what carries the difference at this
 * size — the air plant's blades are a seventh of their own length across and the
 * tuft's a twenty-fifth — and the splay is pulled in behind it because a fan of
 * needles that opens as far as an air plant's reads as a shuttlecock.
 */
export const ROSETTE_GRASS_TUFT: RosetteForm = Object.freeze({
  bladeCount: 15,
  length_m: 0.076,
  halfWidth_m: 0.0030,
  segments: 4,
  splay_rad: 1.90,
  launchShare: 0.20,
  recurve: 1.90,
  value: 0.87,
});

/** Where along a blade it is widest, as a fraction of its length. */
const BLADE_SHOULDER = 0.22;
/** Half-width where the blade leaves the neck, as a fraction of the widest. */
const BLADE_BASE_SHARE = 0.5;
/** How the outer run narrows. Above 1 holds the width, then drops to a point. */
const BLADE_TAPER = 1.15;
/**
 * Smallest tip radius, as a fraction of the widest.
 *
 * A cone that closes to nothing is the honest shape and is what the profile
 * asks for, but a zero-radius apex has no normal, and the last segment of every
 * blade in the frame would be shaded from a degenerate one. A fifth of a
 * millimetre is invisible at this scale and is a surface.
 */
const BLADE_TIP_FLOOR = 0.05;
/** How far a cone runs past a joint into its neighbour, in local radii. */
const JOINT_OVERLAP_SHARE = 0.9;
/** Bundle radius the blades leave from, as a fraction of a blade's half-width. */
const NECK_BUNDLE_SHARE = 0.85;
/** How far up the neck the blades leave, as a fraction of a blade's half-width. */
const NECK_RISE_SHARE = 1.25;
/**
 * How tilt is shared out across the spiral. Above 1 holds more of the rosette
 * upright and puts the whole recurve into the oldest few — which is what a
 * rosette actually does, and it is also what keeps a legible upright core in a
 * silhouette a hundred pixels tall.
 */
const BLADE_AGE_CURVE = 1.15;
/**
 * Length of the most recurved blade, as a fraction of the most upright one.
 *
 * The two ends of one trade. At 1 a blade that falls to horizontal reaches as
 * far sideways as the plant is tall and the rosette opens into a starfish; low
 * enough and the fallen blades become stubs and the plant is a fistful of
 * spikes. The reference's air plants are about a fifth wider than they are
 * tall, and that ratio is what this number is set by.
 */
const BLADE_UPRIGHT_LENGTH_SHARE = 0.72;
/** Palette values a blade ramps between, relative to the form's own value. */
const BLADE_BASE_VALUE_OFFSET = -0.11;
const BLADE_TIP_VALUE_OFFSET = 0.05;
/** The clay palette's declared band. A value outside it is not a white plant. */
const CLAY_VALUE_MINIMUM = 0.62;
const CLAY_VALUE_MAXIMUM = 0.92;

const ROSETTE_TAGS = Object.freeze(["plant", "rosette"]);

/** Integer avalanche hash in [0, 1). The same plant on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

/** Hash mapped to [-1, 1). */
const hashSigned = (n: number): number => 2 * hash01(n) - 1;

const clamp = (value: number, low: number, high: number): number => Math.min(high, Math.max(low, value));

/**
 * Half-width a fraction `s` along a blade.
 *
 * A blade widens off the neck, is broadest a fifth of the way out, and then
 * tapers over the whole remaining run — which is why the reference's blades read
 * as blades and not as spikes. A linear taper from a fat base is the shape that
 * looks like a carrot; the shoulder is what stops it.
 */
export function rosetteBladeHalfWidth_m(s: number, halfWidth_m: number): number {
  const t = clamp(s, 0, 1);
  const rise = Math.min(1, t / BLADE_SHOULDER);
  const fall = Math.min(1, ((1 - t) / (1 - BLADE_SHOULDER)) ** BLADE_TAPER);
  const profile = (BLADE_BASE_SHARE + (1 - BLADE_BASE_SHARE) * rise) * fall;
  return halfWidth_m * Math.max(BLADE_TIP_FLOOR, profile);
}

/**
 * Grow one plant's geometry. Pure: it allocates nothing, touches no builder, and
 * can be asked how far it reaches before a single node is published — which is
 * what lets a caller check a rosette clears its neighbours without building one.
 */
export function planRosette(spec: RosetteSpec): RosettePlan {
  const { bladeCount, length_m, halfWidth_m, segments, splay_rad, launchShare, recurve, seed } = spec;
  if (!Number.isInteger(bladeCount) || bladeCount < 3) throw new RangeError("A rosette needs at least three blades");
  if (!Number.isInteger(segments) || segments < 1) throw new RangeError("A rosette blade needs at least one segment");
  if (!(length_m > 0) || !(halfWidth_m > 0)) throw new RangeError("Rosette blade length and half-width must be positive");
  if (!(splay_rad > 0)) throw new RangeError("Rosette splay must be positive");
  if (!(launchShare >= 0 && launchShare <= 1)) throw new RangeError("Rosette launch share must be in [0, 1]");
  if (!(recurve > 0)) throw new RangeError("Rosette recurve must be positive");

  const [x, z] = spec.at_m;
  const bed_m = spec.bed_m ?? NECK_RISE_SHARE * halfWidth_m;
  const root_m = V(x, spec.groundHeightAt(x, z) - bed_m, z);

  // The neck. Blades leave from inside it rather than from its crown, so the
  // bundle fuses into one stem instead of showing nine cone butts on a bulb.
  //
  // It is deliberately barely wider than one blade and mostly under the ground.
  // A neck sized to hold nine blade bases side by side is a bald egg with a
  // plant standing on it, which is what the first render showed and what none of
  // the reference's plants has: theirs are narrow necks wedged into a gap, and
  // the bases are simply not visible.
  const neckRise_m = NECK_RISE_SHARE * halfWidth_m;
  const neckCenter_m = V(root_m.x, root_m.y + 0.55 * halfWidth_m, root_m.z);
  const neckRadius_m = V(0.9 * halfWidth_m, 1.25 * halfWidth_m, 0.9 * halfWidth_m);

  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const bundle_m = NECK_BUNDLE_SHARE * halfWidth_m;
  const value = clamp(spec.value, CLAY_VALUE_MINIMUM, CLAY_VALUE_MAXIMUM);
  const baseValue = clamp(value + BLADE_BASE_VALUE_OFFSET, CLAY_VALUE_MINIMUM, CLAY_VALUE_MAXIMUM);
  const tipValue = clamp(value + BLADE_TIP_VALUE_OFFSET, CLAY_VALUE_MINIMUM, CLAY_VALUE_MAXIMUM);

  const blades: RosetteBlade[] = [];
  let radius_m = 0;
  let height_m = 0;

  for (let order = 0; order < bladeCount; order += 1) {
    const jitter = (salt: number): number => hashSigned(seed + 131 * order + salt);
    // Age runs from 1 at the first blade to 0 at the last. Pairing it with the
    // golden angle is the whole trick: successive blades are both a step younger
    // and two-fifths of a turn round, so the upright ones never end up adjacent.
    const age = bladeCount > 1 ? 1 - order / (bladeCount - 1) : 1;
    const azimuth_rad = goldenAngle * order + 0.20 * jitter(1);
    const tipTilt_rad = splay_rad * clamp(0.10 + 0.90 * age ** BLADE_AGE_CURVE + 0.09 * jitter(2), 0.06, 1.04);
    const launchTilt_rad = launchShare * tipTilt_rad;
    // An upright blade is the longest one. Without this coupling a blade that
    // falls to horizontal reaches as far sideways as the plant is tall, and the
    // rosette spreads into a starfish instead of the reference's tight fan.
    const upright = 1 - tipTilt_rad / splay_rad;
    const bladeLength_m = length_m * (BLADE_UPRIGHT_LENGTH_SHARE + (1 - BLADE_UPRIGHT_LENGTH_SHARE) * upright) * (1 + 0.07 * jitter(3));
    // A little azimuthal drift, so a blade does not stay in one vertical plane
    // and the fan keeps a side to it from every direction.
    const twist_rad = 0.24 * jitter(4);

    const tiltAt = (s: number): number => launchTilt_rad + (tipTilt_rad - launchTilt_rad) * s ** recurve;
    const valueAt = (s: number): number => baseValue + (tipValue - baseValue) * clamp(s, 0, 1);

    let at = V(
      root_m.x + bundle_m * Math.cos(azimuth_rad),
      root_m.y + neckRise_m,
      root_m.z + bundle_m * Math.sin(azimuth_rad),
    );
    const step_m = bladeLength_m / segments;
    const runs: RosetteSegment[] = [];
    for (let index = 0; index < segments; index += 1) {
      // Midpoint rule. Taking the tangent at the middle of the step rather than
      // at its start is what keeps a four-segment blade on its own arc instead
      // of drifting consistently outside it, and it costs nothing.
      const middle = (index + 0.5) / segments;
      const tilt = tiltAt(middle);
      const azimuth = azimuth_rad + twist_rad * middle;
      const to_m = V(
        at.x + step_m * Math.sin(tilt) * Math.cos(azimuth),
        at.y + step_m * Math.cos(tilt),
        at.z + step_m * Math.sin(tilt) * Math.sin(azimuth),
      );
      runs.push({
        from_m: at,
        to_m,
        fromRadius_m: rosetteBladeHalfWidth_m(index / segments, halfWidth_m),
        toRadius_m: rosetteBladeHalfWidth_m((index + 1) / segments, halfWidth_m),
        value: valueAt(middle),
      });
      radius_m = Math.max(radius_m, Math.hypot(to_m.x - root_m.x, to_m.z - root_m.z));
      height_m = Math.max(height_m, to_m.y - root_m.y);
      at = to_m;
    }
    blades.push({ order, azimuth_rad, length_m: bladeLength_m, launchTilt_rad, tipTilt_rad, segments: runs });
  }

  return {
    spec,
    root_m,
    neckCenter_m,
    neckRadius_m,
    blades,
    radius_m: Math.max(radius_m, neckRadius_m.x),
    height_m: Math.max(height_m, neckCenter_m.y - root_m.y + neckRadius_m.y),
    leafCount: bladeCount * segments + 1,
  };
}

/**
 * Publish a planned rosette as scenery.
 *
 * One group, so a plant is one click target and moving it moves every blade;
 * children are offsets from the group's own origin for the same reason.
 *
 * The group the primitives name is *not* the group node's id, and the extra word
 * is load-bearing: `roughnessFor` in the proxy builder reads the group name and
 * gives anything matching `leaf` the matte finish foliage wants, and everything
 * else a semi-gloss that on a white blade reads as wet plastic.
 */
export function rosetteSceneryNodes(plan: RosettePlan): SceneryNode[] {
  const { spec, root_m } = plan;
  const group = `${spec.key}-leaf`;
  const local = (p: Vec3): Vec3 => V(p.x - root_m.x, p.y - root_m.y, p.z - root_m.z);
  const clay = (value: number): SceneryMaterial => ({ palette: "clay", value });

  const children: SceneryNode[] = [{
    kind: "ellipsoid",
    id: `${spec.key}/neck`,
    group,
    tags: [...ROSETTE_TAGS, "neck"],
    place: { position: local(plan.neckCenter_m) },
    radius: plan.neckRadius_m,
    material: clay(clamp(spec.value + BLADE_BASE_VALUE_OFFSET, CLAY_VALUE_MINIMUM, CLAY_VALUE_MAXIMUM)),
  }];

  for (const blade of plan.blades) {
    const last = blade.segments.length - 1;
    blade.segments.forEach((run, index) => {
      const axis = V(run.to_m.x - run.from_m.x, run.to_m.y - run.from_m.y, run.to_m.z - run.from_m.z);
      const run_m = Math.hypot(axis.x, axis.y, axis.z);
      // Run each cone past the joint into its neighbour. Two cones that merely
      // meet at a point share only that point's plane, and their end faces are
      // perpendicular to *different* axes — so the union of a bent chain has a
      // wedge missing on the outside of every bend, exactly where a recurving
      // blade is most visible. Overlapping by the local radius closes it, and
      // the cost is that the blade is a fraction of a millimetre fatter inside
      // the joint, which is under the taper's own resolution.
      const behind = index > 0 ? JOINT_OVERLAP_SHARE * run.fromRadius_m : 0;
      const ahead = index < last ? JOINT_OVERLAP_SHARE * run.toRadius_m : 0;
      const halfHeight = 0.5 * (run_m + behind + ahead);
      const shift = 0.5 * (ahead - behind) / run_m;
      children.push({
        kind: "cone",
        id: `${spec.key}/blade-${blade.order}-${index}`,
        group,
        tags: [...ROSETTE_TAGS, "blade"],
        place: {
          position: local(V(
            0.5 * (run.from_m.x + run.to_m.x) + shift * axis.x,
            0.5 * (run.from_m.y + run.to_m.y) + shift * axis.y,
            0.5 * (run.from_m.z + run.to_m.z) + shift * axis.z,
          )),
          orientation: alongAxis(axis),
        },
        baseRadius: run.fromRadius_m,
        topRadius: run.toRadius_m,
        halfHeight,
        material: clay(run.value),
      });
    });
  }

  return [{
    kind: "group",
    id: spec.key,
    tags: ROSETTE_TAGS,
    place: { units: "metres", position: root_m },
    children,
  }];
}

/** Plan and publish in one call, which is how a set plants one of these. */
export function rosetteNodes(spec: RosetteSpec): SceneryNode[] {
  return rosetteSceneryNodes(planRosette(spec));
}
