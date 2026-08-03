import type { Quaternion, Vec3 } from "../model";
import { quaternionMultiply } from "../rigid-body";
import type { SceneryMaterial, SceneryNode } from "../scenery-graph";
import { alongAxis, V } from "./builder";
import {
  pondVesselHeightAt,
  pondVesselInnerFaceRun,
  pondVesselPlanCurve,
  type PondVesselSpec,
} from "./pond-vessel";

/**
 * The stone species: capped boulders, packed pebble beds and stepping stones.
 *
 * Reference: `output/imagegen/garden-pond-hose-fill-simplified.png`. Three
 * families that look nothing alike in the frame and are one vocabulary
 * underneath — a graded ellipsoid bedded onto whatever ground it is given, with
 * the boulder adding a stem under its cap and the stepping stone trading its
 * dome for a flat tread. That is deliberate: the reference's stones read as one
 * quarry, and the cheapest way to guarantee that is for them to come out of one
 * generator. They share a file for the same reason they share a look.
 *
 * **These are species, not props.** Each takes a *form* — shape only, no
 * position, no seed, no id — and a *spec* that adds where this one goes, which
 * one it is and what it stands on. Nothing here knows what a pond is: a bed
 * takes an outline and a ground query, so it can ring a pool, line a path or
 * bank against a wall footing without being retyped. See
 * `lib/voxel-scenery/README.md`.
 *
 * The hero pond's own arrangement lives at the bottom of this file, behind
 * `stoneSet`, and is the one part that takes a `PondVesselSpec`. It is a *set*:
 * four boulder specs, two bed specs and one path, all authored against the
 * vessel's plan curve so that a control point moving in the pond moves the whole
 * arrangement with it.
 *
 * **Bedding is delegated, not baked.** Every ground-standing stone is published
 * with `anchor: "terrain"`, so the datum it sits on is resolved from the
 * heightfield the renderer actually draws rather than from a caller's idea of
 * it. The `groundHeightAt` query is still needed while planning — to know which
 * candidates stand in water and how the ground tilts under them — but it decides
 * *whether and how* a stone is placed, never *how high* it ends up. A bake and a
 * bilinear fetch of that bake disagree by up to two millimetres at a coping's
 * foot, which is a fifth of a pebble.
 *
 * **No bedded stone stands in the water.** Scenery is invisible to the solver
 * (`VoxelSceneSource` has no scenery term), so a pebble below the still-water
 * level is a pebble the water flows through. Beds are therefore culled wherever
 * the stone would sit under the level they are given, which costs the
 * reference's few fully submerged shore pebbles and buys a shoreline that cannot
 * be wrong. Stepping stones are the deliberate exception: they are the one
 * family whose whole subject is standing in water.
 *
 * Determinism is the contract the procedural tree already keeps: the seed is the
 * only entropy, it enters through an integer avalanche hash, and the same spec
 * emits the same nodes on every rebuild — the sparse publication cache is keyed
 * on the geometry that comes out of here.
 */

/**
 * Ground height at a world point, in metres.
 *
 * Callers pass the heightfield the scene actually bakes — `terrainHeightAt` for
 * an ordinary terrain, a vessel's own height function for a generated one — so
 * a generator can never disagree with the surface its stones will be drawn on.
 */
export type GroundHeightAt = (x_m: number, z_m: number) => number;

/** A closed plan outline, as the polyline the beds and paths follow. */
export type PlanOutline = readonly (readonly [number, number])[];

/** A patch of plan a bed must leave clear: what a boulder or a post occupies. */
export interface StoneFootprint {
  readonly at_m: readonly [number, number];
  readonly radius_m: number;
}

// ---------------------------------------------------------------------------
// Seeded arithmetic, shared with the vessel and the tree so that "another
// garden" is one re-seed rather than three unrelated notions of randomness.
// ---------------------------------------------------------------------------

/** Integer avalanche hash in [0, 1). The same stones on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

const hashSigned = (n: number): number => 2 * hash01(n) - 1;

/** Linear pick inside an authored `[low, high]` band. */
const band = (range: readonly [number, number], t: number): number => range[0] + (range[1] - range[0]) * t;

/** Shortest distance between two fractions of a turn, in [0, 0.5]. */
function turnDelta(a: number, b: number): number {
  const d = Math.abs(a - b) % 1;
  return Math.min(d, 1 - d);
}

/** Smoothstep on [0, 1], clamped outside it. The same ramp the vessel uses. */
function ramp(value: number): number {
  const t = Math.min(1, Math.max(0, value));
  return t * t * (3 - 2 * t);
}

const quaternionAboutY = (angle_rad: number): Quaternion =>
  ({ w: Math.cos(0.5 * angle_rad), x: 0, y: Math.sin(0.5 * angle_rad), z: 0 });

const stone = (value: number): SceneryMaterial => ({ palette: "stone", value });

/**
 * The palette band the whole quarry is authored in.
 *
 * Outside it a stone stops reading as the same fired white as everything around
 * it and starts reading as a different material, which is the one thing a
 * monochrome set cannot survive. Every form's values are clamped into it rather
 * than trusted, because a form is data a caller can write.
 */
export const STONE_VALUE_MINIMUM = 0.62;
export const STONE_VALUE_MAXIMUM = 0.92;
const value = (v: number): SceneryMaterial => stone(Math.min(STONE_VALUE_MAXIMUM, Math.max(STONE_VALUE_MINIMUM, v)));

// ---------------------------------------------------------------------------
// The rail: any closed plan outline, walkable by arc length and offsettable by
// plan distance.
// ---------------------------------------------------------------------------

/**
 * A plan outline, with an outward normal and a cumulative arc length per sample.
 *
 * Arc length rather than angle is what a *packing* needs: pebbles are laid end
 * to end along a bank, and stepping the angle instead would pack them tightly on
 * a pond's narrow ends and leave gaps on its flanks. An ellipse sampled evenly
 * in angle differs from one sampled evenly in arc by its own aspect — a factor
 * of 1.37 on the hero pond, which is a third of a pebble's width per step and
 * reads immediately as a bed that thins at the sides.
 */
export interface PlanRail {
  readonly points: PlanOutline;
  /** Unit outward normal at each sample: positive plan distance runs along it. */
  readonly outward: readonly (readonly [number, number])[];
  /** Cumulative arc length at each sample, plus the closing length at the end. */
  readonly arc_m: readonly number[];
  readonly length_m: number;
}

/** Prepare an outline for walking. Cheap, and shared by every bed that follows it. */
export function planRail(points: PlanOutline): PlanRail {
  const count = points.length;
  if (count < 3) throw new RangeError("A plan rail needs at least three points");
  let cx = 0, cz = 0;
  for (const [x, z] of points) { cx += x / count; cz += z / count; }
  const outward: (readonly [number, number])[] = [];
  const arc_m: number[] = [0];
  for (let index = 0; index < count; index += 1) {
    const [ax, az] = points[(index - 1 + count) % count];
    const [bx, bz] = points[(index + 1) % count];
    const tx = bx - ax, tz = bz - az;
    const magnitude = Math.hypot(tx, tz) || 1;
    // Either perpendicular is a normal; the one pointing away from the outline's
    // own centroid is the one that means "outside" on a star-shaped outline,
    // which is all this curve is ever allowed to be.
    let nx = tz / magnitude, nz = -tx / magnitude;
    const [px, pz] = points[index];
    if (nx * (px - cx) + nz * (pz - cz) < 0) { nx = -nx; nz = -nz; }
    outward.push([nx, nz]);
    const [qx, qz] = points[(index + 1) % count];
    arc_m.push(arc_m[index] + Math.hypot(qx - px, qz - pz));
  }
  return { points, outward, arc_m, length_m: arc_m[count] };
}

/**
 * A point on the rail at `arc_m` along it, offset `distance_m` outward.
 *
 * The `turn` handed back is the rail's own *parameter* — the fraction of the way
 * round the sample list — and not a bearing about any centre. On a circle the
 * two agree and on the hero pond's ellipse they differ by up to 0.03 of a turn,
 * so anything that has to meet a bearing-indexed quantity converts explicitly
 * rather than assuming.
 */
export function railAt(rail: PlanRail, arc_m: number, distance_m: number): {
  readonly x: number; readonly z: number; readonly turn: number;
} {
  const count = rail.points.length;
  const wrapped = ((arc_m % rail.length_m) + rail.length_m) % rail.length_m;
  // Binary search over the cumulative table; the rail is walked hundreds of
  // times per bed and a linear scan would make the packing quadratic.
  let low = 0, high = count;
  while (low + 1 < high) {
    const middle = (low + high) >> 1;
    if (rail.arc_m[middle] <= wrapped) low = middle; else high = middle;
  }
  const span = rail.arc_m[low + 1] - rail.arc_m[low];
  const t = span > 0 ? (wrapped - rail.arc_m[low]) / span : 0;
  const [ax, az] = rail.points[low];
  const [bx, bz] = rail.points[(low + 1) % count];
  const [anx, anz] = rail.outward[low];
  const [bnx, bnz] = rail.outward[(low + 1) % count];
  let nx = anx + (bnx - anx) * t, nz = anz + (bnz - anz) * t;
  const magnitude = Math.hypot(nx, nz) || 1;
  nx /= magnitude; nz /= magnitude;
  return {
    x: ax + (bx - ax) * t + distance_m * nx,
    z: az + (bz - az) * t + distance_m * nz,
    turn: (low + t) / count,
  };
}

/** Where on the rail a fraction of a turn falls, in arc length. */
export function railArcForTurn(rail: PlanRail, turn: number): number {
  const count = rail.points.length;
  const position = (((turn % 1) + 1) % 1) * count;
  const index = Math.min(count - 1, Math.floor(position));
  const t = position - index;
  return rail.arc_m[index] + t * (rail.arc_m[index + 1] - rail.arc_m[index]);
}

/** A point on the rail at a fraction of a turn, offset outward. */
const railTurnAt = (rail: PlanRail, turn: number, distance_m: number) =>
  railAt(rail, railArcForTurn(rail, turn), distance_m);

/**
 * The ground under a point and the way it leans, as the unit surface normal.
 *
 * The lean is what stops a bedded stone from floating. A coping's inner face can
 * fall forty degrees, and a flat-lying pebble on it would stand a centimetre
 * proud on its uphill side and hang clear of the ground on its downhill one — on
 * a stone whose whole height is two centimetres. The step is a millimetre-scale
 * central difference deliberately wider than a plaster's relief noise: the point
 * is the shape of the bank, not the shape of its grain.
 */
const GROUND_NORMAL_STEP_M = 0.006;

function groundLie(groundHeightAt: GroundHeightAt, x: number, z: number): {
  readonly height_m: number; readonly normal: Vec3;
} {
  const height_m = groundHeightAt(x, z);
  const step = GROUND_NORMAL_STEP_M;
  const dx = (groundHeightAt(x + step, z) - groundHeightAt(x - step, z)) / (2 * step);
  const dz = (groundHeightAt(x, z + step) - groundHeightAt(x, z - step)) / (2 * step);
  const magnitude = Math.hypot(dx, 1, dz);
  return { height_m, normal: V(-dx / magnitude, 1 / magnitude, -dz / magnitude) };
}

/**
 * How far off plumb a bedded stone is ever laid.
 *
 * A basin wall is very nearly vertical, so a stone that followed the bank's
 * normal without limit would be laid on its side the moment it reached the lip
 * of a drop. A stone at the edge of a drop rests on the flat it is standing on,
 * not on the face beside it, and a little over two fifths of a radian is about
 * where a lying stone stops reading as lying.
 */
const STONE_MAXIMUM_BED_TILT_RAD = 0.42;

/** Lean a stone's up-axis part of the way onto the bank's normal, and no further. */
function beddedOrientation(normal: Vec3, follow: number, yaw_rad: number): Quaternion {
  const horizontal = Math.hypot(normal.x, normal.z);
  const tilt = Math.min(follow * Math.atan2(horizontal, Math.max(1e-9, normal.y)), STONE_MAXIMUM_BED_TILT_RAD);
  const scale = horizontal > 1e-9 ? Math.sin(tilt) / horizontal : 0;
  return quaternionMultiply(
    alongAxis(V(normal.x * scale, Math.cos(tilt), normal.z * scale)),
    quaternionAboutY(yaw_rad),
  );
}

// ---------------------------------------------------------------------------
// Species 1: the capped boulder
// ---------------------------------------------------------------------------

/**
 * What makes a mushroom-cap boulder one, with no position, seed or key in it.
 *
 * The proportions are read straight off the reference and are the whole of what
 * makes these read as *capped* rather than as lumps: the cap is a little under
 * twice the stem's width, it is flattened to a bit over a third of its own
 * width, and it overhangs far enough that the undercut is a visible dark line
 * all the way round. Get any one of those wrong and the silhouette becomes a
 * mound — which is exactly what `BOULDER_ROUNDED_COBBLE` below is.
 *
 * Nothing here may be named "mushroom": `svoMaterialFunctionIdForEnvironmentProxy`
 * tests that word on the same line as `hose` and `cloth`, so a boulder described
 * as a mushroom is shaded as an organic one.
 */
export interface CappedBoulderForm {
  /** Cap semi-axis on its long horizontal direction: the boulder's own scale. */
  readonly capRadius_m: number;
  /** Cap semi-height as a fraction of `capRadius_m`. The reference reads 0.36-0.39. */
  readonly capFlatten: number;
  /** Cap semi-axis across, as a fraction of along. Under 1 keeps the plan oval. */
  readonly capDepthShare: number;
  /** Stem top radius as a fraction of the cap's. Under a half is what makes the overhang. */
  readonly stemTopShare: number;
  /** Stem base radius as a fraction of the cap's: the taper, wider at the ground. */
  readonly stemBaseShare: number;
  /** Stem height as a fraction of the cap radius. Near zero gives a seated cobble. */
  readonly stemHeightShare: number;
  /** Footing radius as a fraction of the cap's: the fillet where the stone meets the bank. */
  readonly footingShare: number;
  /** Footing semi-height as a fraction of the cap radius. */
  readonly footingHeightShare: number;
  /**
   * A second lobe swelling out of the cap, as a fraction of the cap's radius.
   *
   * The whole reason it exists: a lone ellipsoid draws a perfect ellipse against
   * the sky, and a perfect ellipse is the one silhouette a weathered stone never
   * has. Zero omits it and saves a leaf.
   */
  readonly shoulderShare: number;
  /** How far the shoulder sits off the cap's axis, as a fraction of the cap radius. */
  readonly shoulderOffsetShare: number;
  /**
   * How far off plumb the stone is set, in radians, before the seed's own
   * scatter. Slight is the point: the reference's stones lean just enough that
   * no two verticals in a group are parallel, and a stone leaning further than
   * about eight degrees stops looking placed and starts looking dropped.
   */
  readonly lean_rad: number;
  /** Palette value at the cap. The stem and footing ramp down off it. */
  readonly value: number;
}

/** The reference's near-left group: a flat wide cap on a short tapered stem. */
export const BOULDER_MUSHROOM_CAP: CappedBoulderForm = Object.freeze({
  capRadius_m: 0.095,
  capFlatten: 0.375,
  capDepthShare: 0.92,
  stemTopShare: 0.50,
  stemBaseShare: 0.61,
  stemHeightShare: 1.05,
  footingShare: 0.84,
  footingHeightShare: 0.20,
  shoulderShare: 0.72,
  shoulderOffsetShare: 0.34,
  lean_rad: 0.07,
  value: 0.86,
});

/**
 * The same species with its stem taken away: the rounded stones the reference
 * banks against the tree's plinth on the far side, which are the same quarry seen
 * as boulders rather than as caps. Kept because a bank of these is what most
 * scenes actually want, and because it is the proof that the form carries the
 * silhouette rather than the generator.
 */
export const BOULDER_ROUNDED_COBBLE: CappedBoulderForm = Object.freeze({
  capRadius_m: 0.070,
  capFlatten: 0.62,
  capDepthShare: 0.86,
  stemTopShare: 0.86,
  stemBaseShare: 0.78,
  stemHeightShare: 0.22,
  footingShare: 0.80,
  footingHeightShare: 0.24,
  shoulderShare: 0.74,
  shoulderOffsetShare: 0.30,
  lean_rad: 0.10,
  value: 0.84,
});

export interface CappedBoulderSpec extends CappedBoulderForm {
  /** Key prefix. Every emitted node is published as `${key}` or `${key}/...`. */
  readonly key: string;
  /** Where the stone stands, in world metres on the plan. */
  readonly at_m: readonly [number, number];
  /** The only entropy. Any integer. */
  readonly seed: number;
}

const BOULDER_TAGS = Object.freeze(["stone", "boulder"]);

/**
 * One boulder, bedded on whatever ground the scene resolves under it.
 *
 * There is no `groundHeightAt` here on purpose: the stone's shape does not
 * depend on the ground, only its datum does, and the datum is delegated to the
 * terrain anchor. A caller that needs to know whether this ground is dry enough
 * to stand a stone on asks its own height query before building one — which is
 * what the hero arrangement does.
 */
export function cappedBoulderNodes(spec: CappedBoulderSpec): SceneryNode[] {
  const { key, seed, capRadius_m } = spec;
  if (!(capRadius_m > 0)) throw new RangeError("A boulder needs a positive cap radius");
  const group = `${key}-stone`;

  const capSemiHeight = capRadius_m * spec.capFlatten * (0.94 + 0.12 * hash01(seed + 1));
  const stemHeight = capRadius_m * spec.stemHeightShare * (0.90 + 0.22 * hash01(seed + 2));
  const stemTop = capRadius_m * spec.stemTopShare * (0.94 + 0.12 * hash01(seed + 3));
  const stemBase = capRadius_m * spec.stemBaseShare * (0.94 + 0.12 * hash01(seed + 4));
  const footingRadius = capRadius_m * spec.footingShare * (0.96 + 0.10 * hash01(seed + 7));
  const footingSemiHeight = capRadius_m * spec.footingHeightShare;

  const leanAzimuth = 2 * Math.PI * hash01(seed + 5);
  const lean = spec.lean_rad * (0.6 + 0.8 * hash01(seed + 6));
  const tilt = alongAxis(V(Math.sin(lean) * Math.cos(leanAzimuth), Math.cos(lean), Math.sin(lean) * Math.sin(leanAzimuth)));

  // The cap sits down over the stem's top rather than balancing on it, so the
  // two fuse into one solid instead of meeting in a seam the voxelizer would
  // have to decide the ownership of.
  const capCenterY = stemHeight + 0.42 * capSemiHeight;
  const shoulderAzimuth = 2 * Math.PI * hash01(seed + 13);

  const children: SceneryNode[] = [
    // The footing. Half in the ground, wider than the stem, and the reason the
    // boulder meets the bank in a shadowed fillet rather than in the clean circle
    // a cone drawn on a plane gives you.
    {
      kind: "ellipsoid",
      id: `${key}/footing`,
      group,
      tags: [...BOULDER_TAGS, "footing"],
      place: { position: V(0, -0.42 * footingSemiHeight, 0) },
      radius: V(footingRadius, footingSemiHeight, footingRadius * (0.86 + 0.10 * hash01(seed + 8))),
      material: value(spec.value - 0.14 + 0.05 * hash01(seed + 9)),
    },
    {
      kind: "cone",
      id: `${key}/stem`,
      group,
      tags: [...BOULDER_TAGS, "stem"],
      place: { position: V(0, 0.5 * stemHeight, 0) },
      baseRadius: stemBase,
      topRadius: stemTop,
      halfHeight: 0.5 * stemHeight,
      material: value(spec.value - 0.12 + 0.05 * hash01(seed + 10)),
    },
    {
      kind: "ellipsoid",
      id: `${key}/cap`,
      group,
      tags: [...BOULDER_TAGS, "cap"],
      place: { position: V(0, capCenterY, 0) },
      radius: V(capRadius_m, capSemiHeight, capRadius_m * spec.capDepthShare * (0.96 + 0.10 * hash01(seed + 11))),
      material: value(spec.value - 0.02 + 0.06 * hash01(seed + 12)),
    },
  ];

  if (spec.shoulderShare > 0) {
    children.push({
      // Fused rather than placed — it sits well inside the cap's own surface and
      // only shows where it pushes past it, so what the eye reads is one stone
      // with a shoulder rather than two stones touching.
      kind: "ellipsoid",
      id: `${key}/cap-lobe`,
      group,
      tags: [...BOULDER_TAGS, "cap"],
      place: {
        position: V(
          capRadius_m * spec.shoulderOffsetShare * Math.cos(shoulderAzimuth),
          capCenterY + capSemiHeight * (0.10 * hashSigned(seed + 14)),
          capRadius_m * spec.shoulderOffsetShare * Math.sin(shoulderAzimuth),
        ),
      },
      radius: V(
        capRadius_m * spec.shoulderShare * (0.94 + 0.16 * hash01(seed + 15)),
        capSemiHeight * (0.86 + 0.14 * hash01(seed + 16)),
        capRadius_m * spec.shoulderShare * (0.86 + 0.20 * hash01(seed + 17)),
      ),
      material: value(spec.value - 0.02 + 0.06 * hash01(seed + 18)),
    });
  }

  return [{
    kind: "group",
    id: key,
    group,
    tags: [...BOULDER_TAGS, "bank"],
    place: {
      units: "metres",
      anchor: "terrain",
      position: V(spec.at_m[0], 0, spec.at_m[1]),
      ground: [spec.at_m[0], spec.at_m[1]],
      orientation: tilt,
    },
    children,
  }];
}

// ---------------------------------------------------------------------------
// Species 2: the pebble bed
// ---------------------------------------------------------------------------

/**
 * What a bed is made of: the grain, the packing and the scatter about it.
 *
 * The band the stones are laid in is *not* here. A bed's width and grade are
 * properties of the run it follows — wide where a bank opens out, a single course
 * where it is a wall — so they arrive as functions of the turn on the spec, and
 * the same form serves both.
 */
export interface PebbleBedForm {
  /** Mean stone radius at the rich end of the grade, and at the lean end. */
  readonly radiusRich_m: number;
  readonly radiusLean_m: number;
  /**
   * Multipliers on the mean that one stone's size is drawn from.
   *
   * Deliberately wide, because the reference's beds are not graded stone: a stone
   * twice its neighbour's size sits next to it and the small ones fill in around
   * it. Narrowing this is what made the first bed read as gravel rather than as
   * pebbles.
   */
  readonly sizeSpread: readonly [number, number];
  /**
   * How far a course advances per stone, and how far the run advances per course,
   * as fractions of a diameter.
   *
   * These decide whether the result is a bed or a necklace, and a tenth of a
   * diameter of overlap gives the necklace. Stones in a real bed rest in the
   * hollows between the stones under and beside them, so a course has to advance
   * by appreciably less than a diameter; a quarter is where the bed closes up and
   * no ground shows through it from a raised camera.
   */
  readonly alongPacking: number;
  readonly acrossPacking: number;
  /** Semi-axis bands as fractions of a stone's own radius: along, up, across. */
  readonly lengthShare: readonly [number, number];
  readonly heightShare: readonly [number, number];
  readonly widthShare: readonly [number, number];
  /**
   * How much the band's width breathes along the run, as a fraction of itself.
   *
   * Two slow octaves keyed on the turn. Without it the bed's outer edge is a
   * clean offset of the rail, which is the one line in a set that could not have
   * been laid by hand.
   */
  readonly widthWander: number;
  /**
   * Mean radius at the band's far edge as a fraction of at its start. Under 1
   * gives a beach: cobbles at the top of the shelf fining to shingle at the water.
   */
  readonly crossGrade: number;
  /** How far a stone is pushed into the bed, as a fraction of its own semi-height. */
  readonly sinkShare: number;
  /**
   * How high the middle of a wide band heaps, as a fraction of a stone's radius.
   *
   * A bed is heaped, not laid: stones toward the middle rest on the ones under
   * them, and the band's cross-section rises to a low crown. That is what turns a
   * mosaic seen from above into a bank of stones seen from the side. Narrow bands
   * get none of it — there is nothing under a single course to heap it on, which
   * is what `moundWidthFloor_m` measures.
   */
  readonly moundShare: number;
  readonly moundWidthFloor_m: number;
  /**
   * How steeply the bed thins toward its outer edge, and the power the thinning
   * climbs by.
   *
   * A packing that runs to a width and halts leaves a machined edge along its
   * whole outer run. Thinning it instead gives the reference's actual ending: a
   * solid bed, then stones with plaster between them, then singles.
   */
  readonly edgeThinning: number;
  readonly edgeThinningPower: number;
  /** How far a stone leans onto the ground's own normal, and the per-stone jitter on it. */
  readonly bedTiltFollow: number;
  readonly bedNormalJitter: number;
  /** Palette value band the bed's stones are drawn from. */
  readonly value: readonly [number, number];
}

/**
 * The bank bed: cobbles heaped where a bed is rich, thinning to shingle where it
 * is lean. This is the form the reference's upper-left bank is made of.
 *
 * The grain is measured off the reference against the pond itself: its coping
 * spans 1.15 m across 1 630 pixels, and the cobbles beside its largest boulder
 * come out 27-44 mm across. The packing is deliberately loose for that grain, and
 * that half is not aesthetic. The lighting hierarchy binds
 * `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK = 64` primitives per 200 mm brick and
 * silently drops the surplus, so the densest stones in a bed are the ones that
 * stop being lit — and the first version of this bed, a continuous double band at
 * a tighter packing, put more than 64 into three bricks. Stone count in a bed
 * goes as the inverse square of the spacing, so a tenth more of it is a fifth
 * fewer stones per brick at no visible cost. `tests/stone-set.test.ts` measures
 * the worst brick rather than trusting this note. See
 * `docs/SVO_FINE_VOXEL_CAPACITY.md`.
 */
export const PEBBLE_GRADED_COBBLE: PebbleBedForm = Object.freeze({
  radiusRich_m: 0.0160,
  radiusLean_m: 0.0085,
  sizeSpread: [0.60, 1.55] as const,
  alongPacking: 0.80,
  acrossPacking: 0.88,
  lengthShare: [0.94, 1.22] as const,
  heightShare: [0.66, 1.02] as const,
  widthShare: [0.82, 1.08] as const,
  widthWander: 0.30,
  // Coarsening *outward*, because the bank band grows away from the water: the
  // reference heaps its biggest cobbles up against the boulders and fines to
  // shingle at the shoreline, and a bed of one grain reads as gravel however
  // wide the size spread inside it is.
  crossGrade: 1.35,
  sinkShare: 0.34,
  moundShare: 0.55,
  moundWidthFloor_m: 0.09,
  edgeThinning: 0.52,
  edgeThinningPower: 2.2,
  bedTiltFollow: 0.72,
  bedNormalJitter: 0.42,
  value: [0.70, 0.90] as const,
});

/**
 * The water's-edge course: two thirds the grain, flatter stones, laid tighter and
 * fining as it runs down. This is what the reference shows all the way round its
 * right and back, where the bed is one course of small stones on the rim's inner
 * slope, and what its left shore fines *to* as the shelf runs into the water.
 */
export const PEBBLE_FINE_SHINGLE: PebbleBedForm = Object.freeze({
  radiusRich_m: 0.0135,
  radiusLean_m: 0.0075,
  sizeSpread: [0.74, 1.36] as const,
  alongPacking: 0.82,
  acrossPacking: 0.90,
  lengthShare: [0.96, 1.26] as const,
  heightShare: [0.62, 0.98] as const,
  widthShare: [0.84, 1.10] as const,
  widthWander: 0.16,
  crossGrade: 0.62,
  sinkShare: 0.36,
  moundShare: 0.40,
  moundWidthFloor_m: 0.07,
  edgeThinning: 0.52,
  edgeThinningPower: 1.9,
  bedTiltFollow: 0.78,
  bedNormalJitter: 0.46,
  value: [0.72, 0.91] as const,
});

export interface PebbleBedSpec extends PebbleBedForm {
  /** Key prefix. Every stone is published as `${key}/pebble-n` under `${key}`. */
  readonly key: string;
  /** The only entropy. Any integer. */
  readonly seed: number;
  /** The outline the bed follows. */
  readonly rail: PlanRail;
  /** Ground under a candidate: the surface the bed will be drawn on. */
  readonly groundHeightAt: GroundHeightAt;
  /** Plan distance the band's near edge sits at. Positive is outside the rail. */
  readonly start_m: number;
  /** Which way the band grows from `start_m`. */
  readonly direction: 1 | -1;
  /** Band width at a fraction of a turn round the rail, before the wander. */
  readonly widthAt: (turn: number) => number;
  /** Where on the grade this part of the run sits, in [0, 1]. 1 is `radiusRich_m`. */
  readonly gradeAt?: (turn: number) => number;
  /**
   * A level no stone may sit under, if this bed meets water.
   *
   * Omit it and the bed ignores water entirely, which is right for a path edging
   * or a wall footing. Give it and the band's lower edge becomes the shoreline
   * exactly, because it is the shoreline that decides it.
   */
  readonly level_m?: number;
  /** Footprints the bed packs around: boulders, posts, anything already standing. */
  readonly avoid?: readonly StoneFootprint[];
  /** The region the bed may occupy. Default: everywhere the rail reaches. */
  readonly within?: (x_m: number, z_m: number) => boolean;
}

const PEBBLE_TAGS = Object.freeze(["stone", "pebble"]);

/**
 * A pebble bed, packed rather than scattered.
 *
 * A Poisson-disc scatter produces a field of stones with ground showing between
 * them, and the reference has no ground showing: its beds are stones resting on
 * stones. So this walks the band instead. At each step along the rail it lays a
 * *course* across the band, each pebble advanced by its own radius and its
 * neighbour's, and steps along by the same measure; alternate courses start half
 * a stone across so they interlock like brickwork rather than lining up in rows.
 * The packing fractions are under one, which means neighbours actually touch — a
 * bed of exactly-tangent spheres still reads as beads.
 *
 * Every candidate is then tested against the ground it would occupy, and the ones
 * standing in water, outside the region or under a footprint are dropped.
 */
export function pebbleBedNodes(spec: PebbleBedSpec): SceneryNode[] {
  const { key, rail, groundHeightAt, direction, start_m } = spec;
  if (direction !== 1 && direction !== -1) throw new RangeError("A pebble band grows outward or inward, nothing else");
  const gradeAt = spec.gradeAt ?? (() => 1);
  const children: SceneryNode[] = [];
  const group = `${key}-stone`;
  let published = 0;
  let arc = 0;
  let course = 0;

  while (arc < rail.length_m) {
    const turn = railAt(rail, arc, 0).turn;
    const grade = Math.min(1, Math.max(0, gradeAt(turn)));
    // A little of the bed's wandering is its own rather than its run's: two slow
    // octaves keyed on the course so the width breathes along the bank.
    const noise = 0.5 + 0.5 * Math.sin(2 * Math.PI * (3.3 * turn + hash01(spec.seed)))
      * (0.6 + 0.4 * Math.sin(2 * Math.PI * (7.7 * turn + hash01(spec.seed + 1))));
    const meanRadius = spec.radiusLean_m + (spec.radiusRich_m - spec.radiusLean_m) * grade;
    const width = Math.max(0, spec.widthAt(turn)) * (1 + spec.widthWander * (2 * noise - 1));

    let across = start_m + direction * (course % 2 === 0 ? 0 : spec.acrossPacking * meanRadius);
    const limit = start_m + direction * width;
    let lane = 0;
    while (direction * (limit - across) > 0) {
      const seed = spec.seed + 0x3f_00 + 31 * published + 7919 * lane + 131 * course;
      const bandFraction = Math.abs(across - start_m) / Math.max(1e-6, width);
      // The grade also runs *across* the band, which is what a beach does: the
      // cobbles are at the top of the shelf and the shingle is at the water.
      const laneRadius = meanRadius * (1 + (spec.crossGrade - 1) * Math.min(1, bandFraction));
      const radius = laneRadius * band(spec.sizeSpread, hash01(seed));
      across += direction * spec.acrossPacking * radius;
      const centre = across;
      const at = railAt(rail, arc + (hashSigned(seed + 1) * 0.35 * radius), centre);
      across += direction * spec.acrossPacking * radius;
      lane += 1;

      // A wide aspect band, not just a wide size band. The reference's beds hold
      // flat slabs lying beside near-round cobbles, and a bed whose stones all
      // share one aspect reads as a tray of eggs however much their sizes differ.
      const semi = V(
        radius * band(spec.lengthShare, hash01(seed + 2)),
        radius * band(spec.heightShare, hash01(seed + 3)),
        radius * band(spec.widthShare, hash01(seed + 4)),
      );
      const sink = spec.sinkShare * semi.y;
      const edge = Math.abs(centre - start_m) / Math.max(1e-6, width);
      if (hash01(seed + 7) < spec.edgeThinning * edge ** spec.edgeThinningPower) continue;
      if (spec.within && !spec.within(at.x, at.z)) continue;
      if (spec.avoid?.some(({ at_m, radius_m }) => Math.hypot(at.x - at_m[0], at.z - at_m[1]) < radius_m + 0.55 * radius)) continue;
      const lie = groundLie(groundHeightAt, at.x, at.z);
      // The level test is on the stone's own centre, not on the ground: a pebble
      // bedded on a bank that is barely clear of the water still has most of
      // itself under it, and "no bedded stone in the water" has to mean the stone
      // rather than the point it stands on.
      if (spec.level_m !== undefined && lie.height_m - sink <= spec.level_m) continue;
      const mound = spec.moundShare * radius * Math.sin(Math.PI * Math.min(1, edge))
        * Math.min(1, Math.max(0, (width - 2.2 * meanRadius) / spec.moundWidthFloor_m));

      children.push({
        kind: "ellipsoid",
        id: `${key}/pebble-${published}`,
        group,
        tags: PEBBLE_TAGS,
        place: {
          units: "metres",
          anchor: "terrain",
          position: V(at.x, mound - sink, at.z),
          ground: [at.x, at.z],
          // The bank's own normal, jittered per stone. Bedding every pebble
          // square to the ground makes a bed whose stones all catch the light
          // identically: the giveaway that they came out of a loop. A stone lying
          // against its neighbour instead of on the plaster is what the jitter
          // stands in for, and it is the cheapest form of the contact relaxation
          // this packing deliberately does not run.
          orientation: beddedOrientation(
            V(
              lie.normal.x + spec.bedNormalJitter * hashSigned(seed + 8),
              lie.normal.y,
              lie.normal.z + spec.bedNormalJitter * hashSigned(seed + 9),
            ),
            spec.bedTiltFollow,
            2 * Math.PI * hash01(seed + 5),
          ),
        },
        radius: semi,
        material: value(band(spec.value, hash01(seed + 6))),
      });
      published += 1;
    }
    arc += 2 * spec.alongPacking * meanRadius;
    course += 1;
  }

  return [{
    kind: "group",
    id: key,
    group,
    tags: PEBBLE_TAGS,
    // No position: the children carry world metres and resolve their own ground,
    // and a terrain anchor samples the heightfield under a node's *local* place.
    // A group with an offset would move the stones and leave their datums behind.
    place: { units: "metres" },
    children,
  }];
}

// ---------------------------------------------------------------------------
// Species 3: stepping stones
// ---------------------------------------------------------------------------

/**
 * What a stepping stone is: a flat tread on a footing, and how much of it shows.
 *
 * The tread is a cylinder with a very flat ellipsoid over it, the ellipsoid's
 * equator exactly on the cylinder's top face. That is tangent-continuous — the
 * side runs vertically into a crown that lifts a tenth of a radius — so the stone
 * reads as cut flat with a softened edge, which is what the reference shows,
 * without the two-primitive seam a smaller dome would leave.
 */
export interface SteppingStoneForm {
  /**
   * Tread thickness.
   *
   * Measured off the reference against the disc's own width: the near stone shows
   * a side about a fifth of its diameter, of which the top 17 mm is above the
   * water. A thinner tread reads as a tile laid on the water.
   */
  readonly tread_m: number;
  /** Dome rise on the tread, as a fraction of its radius. */
  readonly dome: number;
  /**
   * Footing radii as fractions of the tread's, at the bed and under the tread.
   *
   * Wider at the bed than the tread it carries. A footing narrower than its tread
   * is a stalk, and a stalk under a disc is a capped boulder — which is the one
   * thing these must not read as when there are actual capped boulders on the
   * bank behind them. Under water none of this shows; on a dry scene all of it
   * does, which is the condition to author for.
   */
  readonly footingBaseShare: number;
  readonly footingTopShare: number;
  /**
   * How far the footing reaches under the bed.
   *
   * It always reaches below, so a stone cannot hang in the water on a heightfield
   * that samples two millimetres away from the one this was planned against.
   */
  readonly bed_m: number;
  /** How far the tread's top clears the level it wades through. */
  readonly freeboard_m: number;
  /**
   * How much of the tread stands proud where the bed is already above the level,
   * as a fraction of the tread.
   *
   * This is what makes the shore end of a path read as *nearly dry*: a stone on
   * ground above the waterline shows its whole thickness rather than sinking to
   * meet a water surface that is not there.
   */
  readonly emergenceShare: number;
  /** Per-stone scatter on the radius, as a fraction of itself. */
  readonly radiusJitter: number;
  /** Palette value at the tread. The footing ramps down off it. */
  readonly value: number;
}

/**
 * The reference's five: flat pale *plates*, barely proud of the water.
 *
 * Plate, not drum, is the whole of the form. The reference's stones are about a
 * fifth of their own width thick and show nothing under them: what stands above
 * the bed is a disc with a softened edge and no visible support at all. So the
 * tread is thin, and the footing is narrower than the tread at every height —
 * `footingBaseShare` under 1 — which is the opposite of what a boulder's footing
 * does. A footing that flared past its tread would put a visible collar round
 * every plate on a scene that opens dry.
 */
export const STEPPING_DISC: SteppingStoneForm = Object.freeze({
  tread_m: 0.018,
  dome: 0.075,
  footingBaseShare: 0.94,
  footingTopShare: 0.78,
  bed_m: 0.052,
  freeboard_m: 0.011,
  emergenceShare: 0.85,
  radiusJitter: 0.03,
  value: 0.84,
});

export interface SteppingStonePathSpec extends SteppingStoneForm {
  /** Key prefix. Every stone is published as `${key}/step-n`. */
  readonly key: string;
  /** The only entropy. Any integer. */
  readonly seed: number;
  /**
   * The line the stones are laid on, as control points in world metres.
   *
   * Two points give a straight run and three or more a curve — a Catmull-Rom
   * through them, walked by *arc length* rather than by parameter. Equal steps in
   * parameter would put the first pair of stones inside one another wherever the
   * legs of the path differ in length, which on any wading path they do.
   */
  readonly path: PlanOutline;
  readonly count: number;
  /** Tread radius at the first stone and at the last. */
  readonly radiusStart_m: number;
  readonly radiusEnd_m: number;
  /** Clear water between consecutive treads: the stride the path is laid at. */
  readonly stride_m: number;
  /** Bed under a stone: the surface its footing stands on. */
  readonly groundHeightAt: GroundHeightAt;
  /** The still-water level the path wades through. Omit for a dry path. */
  readonly level_m?: number;
}

const STEPPING_TAGS = Object.freeze(["stone", "stepping"]);

/** Bezier-equivalent samples used to walk the path by arc length. */
const STEPPING_PATH_SAMPLES = 192;

/** Uniform Catmull-Rom through the control points, with the ends held. */
function pathPoint(points: PlanOutline, s: number): readonly [number, number] {
  const spans = points.length - 1;
  const position = Math.min(spans - 1e-9, Math.max(0, s * spans));
  const index = Math.floor(position);
  const t = position - index, t2 = t * t, t3 = t2 * t;
  const at = (offset: number) => points[Math.min(points.length - 1, Math.max(0, index + offset))];
  const [ax, az] = at(-1), [bx, bz] = at(0), [cx, cz] = at(1), [dx, dz] = at(2);
  const spline = (a: number, b: number, c: number, d: number) =>
    0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - 3 * c + d - a) * t3);
  return [spline(ax, bx, cx, dx), spline(az, bz, cz, dz)];
}

/**
 * A path of stepping stones, each set against the bed under it.
 *
 * This is the one species whose height is authored against the *water* rather
 * than against the ground. A stepping stone's whole subject is the constant
 * finger's breadth of freeboard it holds as the pond deepens under it — a chain
 * that instead held a constant height above the bed would climb out of the water
 * at the shore and drown at the far end, which is the opposite of the reference.
 * So the tread's top is placed on the level and the footing under it is as long
 * as the local depth demands.
 *
 * Where the bed is *above* the level the rule inverts and the stone sits on the
 * ground with its whole thickness showing, which is what the shore end of a path
 * wading in over a shelf actually looks like — and, on a scene that opens dry,
 * what most of the path looks like.
 */
export function steppingStoneNodes(spec: SteppingStonePathSpec): SceneryNode[] {
  const { key, seed, count, groundHeightAt } = spec;
  if (!Number.isInteger(count) || count < 1) throw new RangeError("A stepping path needs at least one stone");
  if (spec.path.length < 2) throw new RangeError("A stepping path needs at least two control points");

  const walk: number[] = [0];
  for (let step = 1; step <= STEPPING_PATH_SAMPLES; step += 1) {
    const [ax, az] = pathPoint(spec.path, (step - 1) / STEPPING_PATH_SAMPLES);
    const [bx, bz] = pathPoint(spec.path, step / STEPPING_PATH_SAMPLES);
    walk.push(walk[step - 1] + Math.hypot(bx - ax, bz - az));
  }
  const pointAtArc = (target: number): readonly [number, number] => {
    // Past the end the path is continued along its own last tangent rather than
    // clamped: a chain longer than the line it was given should run off the end
    // of it, not pile up on the last point.
    if (target > walk[STEPPING_PATH_SAMPLES]) {
      const [ax, az] = pathPoint(spec.path, 1 - 1 / STEPPING_PATH_SAMPLES);
      const [bx, bz] = pathPoint(spec.path, 1);
      const run = Math.hypot(bx - ax, bz - az) || 1;
      const over = target - walk[STEPPING_PATH_SAMPLES];
      return [bx + (bx - ax) * over / run, bz + (bz - az) * over / run];
    }
    let step = 1;
    while (step < STEPPING_PATH_SAMPLES && walk[step] < target) step += 1;
    const span = walk[step] - walk[step - 1];
    return pathPoint(spec.path, (step - 1 + (span > 0 ? (target - walk[step - 1]) / span : 0)) / STEPPING_PATH_SAMPLES);
  };

  const radiusOf = (index: number): number => {
    const t = count > 1 ? index / (count - 1) : 0;
    return (spec.radiusStart_m + (spec.radiusEnd_m - spec.radiusStart_m) * t)
      * (1 - 0.5 * spec.radiusJitter + spec.radiusJitter * hash01(seed + 613 * index));
  };

  const nodes: SceneryNode[] = [];
  let arc = 0;
  for (let index = 0; index < count; index += 1) {
    const stoneSeed = seed + 0x77_00 + 613 * index;
    const radius = radiusOf(index);
    if (index > 0) arc += radiusOf(index - 1) + radius + spec.stride_m;
    const [x, z] = pointAtArc(arc);
    const group = `${key}-stone`;

    const ground = groundHeightAt(x, z);
    const level = spec.level_m ?? -Infinity;
    const treadTop = Math.max(
      level + spec.freeboard_m * (0.85 + 0.3 * hash01(stoneSeed + 1)),
      ground + spec.emergenceShare * spec.tread_m,
    );
    const treadBottom = treadTop - spec.tread_m;
    const footingBottom = ground - spec.bed_m;
    const footingTop = treadBottom + 0.004;

    nodes.push({
      kind: "group",
      id: `${key}/step-${index}`,
      group,
      tags: [...STEPPING_TAGS, "step"],
      // The stone's own origin, and the yaw is about *that*. It has to be: a
      // group's orientation rotates its children's offsets, so a group left at
      // the world origin with children carrying world coordinates spins every
      // stone round the middle of the scene instead of round itself. That is not
      // a hypothetical — it is what this generator did, and the five plates
      // landed scattered across the far side of the basin standing on 110 mm
      // plinths, which read exactly like five drums that had been placed there on
      // purpose. Nothing in the numbers was wrong; the frame was.
      place: {
        units: "metres",
        position: V(x, 0, z),
        orientation: quaternionAboutY(2 * Math.PI * hash01(stoneSeed + 2)),
      },
      children: [
        {
          // A tapered plinth, not a post. A straight-sided column under a wider
          // tread is a stalk, and a foot that narrows into the bed reads as the
          // stone it is under either condition.
          kind: "cone",
          id: `${key}/step-${index}/footing`,
          group,
          tags: [...STEPPING_TAGS, "footing"],
          place: { position: V(0, 0.5 * (footingBottom + footingTop), 0) },
          baseRadius: radius * spec.footingBaseShare,
          topRadius: radius * spec.footingTopShare,
          halfHeight: 0.5 * (footingTop - footingBottom),
          material: value(spec.value - 0.16 + 0.04 * hash01(stoneSeed + 3)),
        },
        {
          kind: "cylinder",
          id: `${key}/step-${index}/tread`,
          group,
          tags: [...STEPPING_TAGS, "tread"],
          place: { position: V(0, 0.5 * (treadBottom + treadTop), 0) },
          radius,
          halfHeight: 0.5 * spec.tread_m,
          material: value(spec.value - 0.04 + 0.06 * hash01(stoneSeed + 4)),
        },
        {
          kind: "ellipsoid",
          id: `${key}/step-${index}/crown`,
          group,
          tags: [...STEPPING_TAGS, "crown"],
          place: { position: V(0, treadTop, 0) },
          radius: V(radius, radius * spec.dome, radius * (0.97 + 0.05 * hash01(stoneSeed + 5))),
          material: value(spec.value - 0.02 + 0.06 * hash01(stoneSeed + 6)),
        },
      ],
    });
  }
  return nodes;
}
// ---------------------------------------------------------------------------
// The hero pond's arrangement
// ---------------------------------------------------------------------------

/**
 * The hero garden's stone set: four boulders, a few beds and one wading path.
 *
 * The one type here that takes a `PondVesselSpec`, and the reason the species
 * above do not. Everything is authored in the vessel's own coordinates — a
 * fraction of a turn round the plan curve and a plan distance either side of it —
 * so a control point moving in the pond carries the whole set with it, and
 * nothing is a world coordinate that could drift from the coping it hugs.
 */
export interface StoneSetSpec {
  /** The vessel the stones follow. Its plan curve is the only layout authority. */
  readonly vessel: PondVesselSpec;
  /** Still-water level in metres. Bedded stones stay above it; the path wades through it. */
  readonly waterline_m: number;
  /** The only entropy. Any integer. */
  readonly seed: number;
  /** Key prefix for every node emitted. Defaults to `stone`. */
  readonly key?: string;
}

/**
 * Turns here are the *rail's* parameter, not a bearing.
 *
 * The vessel indexes its beach by the bearing about the plan's centre and the
 * rail indexes itself by arc, and on an ellipse of this aspect the two differ by
 * up to 0.03 of a turn — a third of the beach sector's half-width. Everything
 * authored below is a rail turn, and `beachRunAt` converts before it asks the
 * vessel anything.
 */
function bearingTurn(spec: StoneSetSpec, x: number, z: number): number {
  const [cx, cz] = spec.vessel.center_m;
  return Math.atan2(z - cz, x - cx) / (2 * Math.PI);
}

/** The vessel's own inner-face run under a point on the rail. Never re-derived. */
function beachRunAt(spec: StoneSetSpec, rail: PlanRail, turn: number): number {
  const at = railTurnAt(rail, turn, 0);
  return pondVesselInnerFaceRun(spec.vessel, bearingTurn(spec, at.x, at.z));
}

/** The vessel's height function, as the query every species here is given. */
function vesselGround(spec: StoneSetSpec): GroundHeightAt {
  const curve = pondVesselPlanCurve(spec.vessel);
  return (x, z) => pondVesselHeightAt(spec.vessel, curve, x, z);
}

/**
 * The four boulders, as a small graded family on the left bank.
 *
 * Two mistakes are worth recording, because both looked right in the numbers and
 * wrong in the frame.
 *
 * The first put all four at turns 0.41-0.52, which at the camera the scene has
 * now is the pond's *far* side: the group clumped at the left edge at the
 * greatest depth in the shot, which is the one place a boulder cannot read as
 * large. The second overcorrected — the group came round to the near bank at
 * turns 0.30-0.45, half a metre from the eye, and four stones the size of the
 * pond's own rim filled the near-left quarter of the frame.
 *
 * The reference settles it. Its largest cap is about a *sixth* of the pond's
 * width and it sits **behind the near coping**, up and to the left, with all four
 * reading as one small family rather than as foreground objects. So the group
 * lives on the far half of the left bank, where the camera is a metre away
 * instead of half of one, and the caps grade 2.4 : 1 across it — the receding
 * line the reference draws, rather than a row.
 */
interface BoulderPlacement {
  /** Fraction of a turn round the rail. */
  readonly turn: number;
  /** Plan distance outside the coping's crest centreline. */
  readonly offset_m: number;
  /** Cap semi-axis on its long horizontal direction: the boulder's scale. */
  readonly capRadius_m: number;
}

const HERO_BOULDERS: readonly BoulderPlacement[] = Object.freeze([
  { turn: 0.702, offset_m: 0.142, capRadius_m: 0.080 },
  { turn: 0.656, offset_m: 0.146, capRadius_m: 0.060 },
  { turn: 0.612, offset_m: 0.150, capRadius_m: 0.044 },
  { turn: 0.570, offset_m: 0.150, capRadius_m: 0.033 },
]);

/**
 * The plaster the set may stand on, as a plan rectangle.
 *
 * The hero container is 1.8 x 1.2 m and the pond's plan reaches 0.52 x 0.38, so
 * on the near bearings there is barely 200 mm of ground between the coping and
 * the wall. A bed that ran to its full width there would put stones over the edge
 * — off the plaster, and out of the container the extent oracle pins. This is the
 * one number here that is about the room rather than about the pond, and it is
 * inset by a large cobble so a stone's *body* stays inside too.
 */
const HERO_PLASTER_HALF_X_M = 0.852;
const HERO_PLASTER_HALF_Z_M = 0.552;
const onPlaster = (x: number, z: number): boolean =>
  Math.abs(x) <= HERO_PLASTER_HALF_X_M && Math.abs(z) <= HERO_PLASTER_HALF_Z_M;

/**
 * Where the pebbles actually are, as the three or four places the reference puts
 * them — and, just as importantly, the long runs of coping where it puts none.
 *
 * This is the correction that mattered most. An earlier bed ringed the whole
 * coping in an unbroken double band, inside and out, all the way round. The
 * reference does nothing of the sort: it has a dense bed at the upper left
 * between the boulders and the water, a course along the right-hand rim by the
 * tree, a loose scatter outside the coping at the lower right, and two or three
 * singles at the lower left. Most of the rim has no stones against it at all, and
 * that bare white run is a large part of why the reference reads as composed
 * rather than as decorated.
 *
 * There is a measured reason as well as a compositional one. The lighting
 * hierarchy binds 64 primitives per 200 mm brick and drops the rest silently, and
 * a continuous double band put more than that into three bricks. Clusters plus
 * the coarser grain halve the set.
 */
interface PebbleCluster {
  /** Middle of the cluster, as a rail turn. */
  readonly turn: number;
  /** Half its run in turns. The bed fades to nothing over the outer part of it. */
  readonly halfLength: number;
  /** Widest the bank band gets here, in plan metres. Zero leaves the outside bare. */
  readonly bankWidth_m: number;
  /** Where on the grade this cluster sits, in [0, 1]. 1 is the coarsest cobble. */
  readonly grade: number;
  /** Whether the water's-edge course runs here too. */
  readonly shore: boolean;
}

const HERO_PEBBLE_CLUSTERS: readonly PebbleCluster[] = Object.freeze([
  // The dense bed under the boulders, running down over the beach into the
  // water. The reference's one large bed, and the only place both bands meet.
  { turn: 0.560, halfLength: 0.150, bankWidth_m: 0.120, grade: 1.00, shore: true },
  // The course along the right-hand rim, where the tree's terrace comes down to
  // the coping. Narrow, and the one the reference runs *inside* the rim.
  { turn: 0.080, halfLength: 0.095, bankWidth_m: 0.062, grade: 0.45, shore: true },
  // The loose scatter outside the coping at the near right: a dozen stones, no
  // bed under them, which is what a bank width of one cobble produces.
  { turn: 0.230, halfLength: 0.062, bankWidth_m: 0.038, grade: 0.30, shore: false },
  // Two or three singles at the near left, and nothing else on that whole run.
  { turn: 0.395, halfLength: 0.038, bankWidth_m: 0.028, grade: 0.25, shore: false },
]);

/** How much of a cluster's run is full width before it fades out. */
function clusterWeight(cluster: PebbleCluster, turn: number): number {
  return ramp(1 - turnDelta(turn, cluster.turn) / cluster.halfLength);
}

/**
 * The bank band's start, in plan distance from the coping's crest centreline.
 *
 * Beyond the widest the coping's section ever swells to — `rimHalfWidth_m` is
 * modulated by up to 22 % as it runs — so a bed never climbs the rim's outer
 * shoulder.
 */
const HERO_BANK_START_SHARE = 1.14;

/**
 * The shore band, and the thing that changed when the pond grew a beach.
 *
 * The plan wanted a bed hugging the coping's inner foot, which assumed the
 * shallow dish the vessel started as. Everywhere the inner face is a wall — 155
 * mm of drop in 35 mm of run — there is no shore inside this pond at all, and
 * what survives of the reference's inner bed is the one course of small stones
 * resting on the rim's inner slope right at the water's edge, which the reference
 * does show along its right-hand rim.
 *
 * Over the beach sector there is now a real shelf, and the reference's wide bed
 * running down into the water can exist. So the band's width is a share of the
 * *vessel's own* inner-face run at that bearing: 35 mm of run gives a single
 * course and 300 mm gives a bed a hand's breadth wide, out of one expression,
 * with the waterline test deciding per stone where it stops. The outline of what
 * is left is the vessel's shoreline rather than a number authored here.
 */
const HERO_SHORE_START_SHARE = -0.84;
const HERO_SHORE_RUN_SHARE = 0.62;
const HERO_SHORE_WIDTH_FLOOR_M = 0.022;

/**
 * The wading path, as rail turns and how deep the bed is under the water there.
 *
 * Authored as *submergence* rather than as a plan distance, and solved against
 * the beach at build time, because the thing a stepping stone cares about is how
 * far its bed lies under the surface — not how far out it stands. `shorelineRun`
 * below inverts the shelf for each control point, so the path is literally a
 * contour of the pond's own shore and follows it if the vessel changes.
 *
 * That inversion is what turns the plates back into plates. Laid at a fixed plan
 * distance instead, the same chain crosses the shelf's toe halfway along and its
 * last two stones stand on 110 mm plinths on the basin floor — five drums in open
 * water, which is exactly what this arrangement replaced. Held on a contour, the
 * bed under every stone is 10-45 mm down, so what shows under each tread is a few
 * millimetres of taper. That matters here more than it would elsewhere: the scene
 * opens **dry**, so everything the water was going to hide is on show.
 *
 * The shape is the reference's — a shallow S entering the water from the left,
 * bowing out into the pond in the middle and turning back toward the bank at the
 * near end.
 */
const HERO_PATH: readonly (readonly [number, number])[] = Object.freeze([
  [0.575, 0.020],
  [0.520, 0.027],
  [0.462, 0.035],
  [0.412, 0.042],
  [0.374, 0.048],
  // A tail past the last stone, so the walk stays on the authored curve instead
  // of running off the end of it along a straight tangent.
  [0.344, 0.054],
]);

/**
 * Tread radius at the shore end and at the deep end. Wading in, they shrink.
 *
 * Measured against the pond rather than against the frame: the reference's coping
 * spans 1.15 m across 1 630 pixels, which puts its nearest plate at about 100 mm
 * across and its furthest at 75 mm. An earlier reading of 145 mm came from
 * measuring the nearest stone as though it stood at the pond's own depth.
 */
const HERO_STEP_RADIUS_SHORE_M = 0.042;
const HERO_STEP_RADIUS_DEEP_M = 0.032;
const HERO_STEP_COUNT = 5;
/** Clear water between plates: about a third of a plate, which is what the shelf affords. */
const HERO_STEP_STRIDE_M = 0.026;

/** How far inward the shelf is searched before a contour is called unreachable. */
const HERO_SHORE_SEARCH_M = 0.34;

/**
 * The plan distance at which the bed lies `submerged_m` under the level.
 *
 * A bisection inward from the coping's inner foot, which is legitimate because
 * the ground falls monotonically from there to the basin floor — the crest is the
 * only rise in the section and the search starts inside it. Clamped at both ends:
 * a stone may never come nearer the rim than `minimum_m`, and a contour deeper
 * than the basin floor resolves to the toe of the shelf rather than to nothing.
 */
function shorelineRun(
  groundHeightAt: GroundHeightAt,
  rail: PlanRail,
  turn: number,
  target_m: number,
  minimum_m: number,
): number {
  const arc = railArcForTurn(rail, turn);
  const groundAt = (run: number): number => {
    const at = railAt(rail, arc, -run);
    return groundHeightAt(at.x, at.z);
  };
  let low = minimum_m, high = HERO_SHORE_SEARCH_M;
  if (groundAt(low) <= target_m) return low;
  if (groundAt(high) >= target_m) return high;
  for (let step = 0; step < 24; step += 1) {
    const middle = 0.5 * (low + high);
    if (groundAt(middle) > target_m) low = middle; else high = middle;
  }
  return 0.5 * (low + high);
}

/** The four boulders, dropped wherever the bank they want is under water. */
export function stoneSetBoulderNodes(spec: StoneSetSpec): SceneryNode[] {
  const key = spec.key ?? "stone";
  const rail = planRail(pondVesselPlanCurve(spec.vessel));
  const groundHeightAt = vesselGround(spec);
  const nodes: SceneryNode[] = [];
  for (const [index, placement] of HERO_BOULDERS.entries()) {
    const at = railTurnAt(rail, placement.turn, placement.offset_m);
    if (groundHeightAt(at.x, at.z) <= spec.waterline_m) continue;
    nodes.push(...cappedBoulderNodes({
      ...BOULDER_MUSHROOM_CAP,
      capRadius_m: placement.capRadius_m,
      key: `${key}/boulder-${index}`,
      at_m: [at.x, at.z],
      seed: spec.seed + 0x51_00 + 977 * index,
    }));
  }
  return nodes;
}

/** Plan footprint a boulder claims, so the beds can be laid around it. */
function boulderFootprints(spec: StoneSetSpec, rail: PlanRail): StoneFootprint[] {
  return HERO_BOULDERS.map((placement) => {
    const at = railTurnAt(rail, placement.turn, placement.offset_m);
    return { at_m: [at.x, at.z] as const, radius_m: placement.capRadius_m * 0.88 };
  });
}

/** The beds: cobbles heaped on the bank, shingle running down the shore. */
export function stoneSetPebbleNodes(spec: StoneSetSpec): SceneryNode[] {
  const key = spec.key ?? "stone";
  const rail = planRail(pondVesselPlanCurve(spec.vessel));
  const rimHalfWidth = spec.vessel.rimHalfWidth_m;
  const groundHeightAt = vesselGround(spec);
  const avoid = boulderFootprints(spec, rail);

  const bankWidthAt = (turn: number): number =>
    HERO_PEBBLE_CLUSTERS.reduce((widest, cluster) =>
      Math.max(widest, cluster.bankWidth_m * clusterWeight(cluster, turn)), 0);
  const shoreWeightAt = (turn: number): number =>
    HERO_PEBBLE_CLUSTERS.reduce((most, cluster) =>
      Math.max(most, cluster.shore ? clusterWeight(cluster, turn) : 0), 0);
  // The grade is the clusters' own, weighted by how much of each is here, so a
  // stone in the overlap between two of them is graded between the two rather
  // than jumping at the join.
  const gradeAt = (turn: number): number => {
    let weight = 0, graded = 0;
    for (const cluster of HERO_PEBBLE_CLUSTERS) {
      const w = clusterWeight(cluster, turn);
      weight += w; graded += w * cluster.grade;
    }
    return weight > 0 ? graded / weight : 0;
  };

  return [
    ...pebbleBedNodes({
      ...PEBBLE_GRADED_COBBLE,
      key: `${key}/bank-bed`,
      seed: spec.seed + 0x9e_00,
      rail,
      groundHeightAt,
      start_m: HERO_BANK_START_SHARE * rimHalfWidth,
      direction: 1,
      widthAt: bankWidthAt,
      gradeAt,
      level_m: spec.waterline_m,
      avoid,
      within: onPlaster,
    }),
    ...pebbleBedNodes({
      ...PEBBLE_FINE_SHINGLE,
      key: `${key}/shore-bed`,
      seed: spec.seed + 0x9e_00 + 104_729,
      rail,
      groundHeightAt,
      // Laid deliberately *across* the coping's inner foot, so the waterline test
      // decides per stone which side of it each one ends up on rather than a
      // number authored here.
      start_m: HERO_SHORE_START_SHARE * rimHalfWidth,
      direction: -1,
      widthAt: (turn) => shoreWeightAt(turn)
        * (HERO_SHORE_WIDTH_FLOOR_M + HERO_SHORE_RUN_SHARE * beachRunAt(spec, rail, turn)),
      gradeAt,
      level_m: spec.waterline_m,
      avoid,
      within: onPlaster,
    }),
  ];
}

/** The five plates, wading in along a contour of the shore. */
export function stoneSetSteppingNodes(spec: StoneSetSpec): SceneryNode[] {
  const key = spec.key ?? "stone";
  const rail = planRail(pondVesselPlanCurve(spec.vessel));
  const groundHeightAt = vesselGround(spec);
  // A plate may never come nearer the rim than its own radius plus the widest the
  // coping's section swells to, or it grows out of the rim it is supposed to have
  // stepped off. Taken against the *largest* plate and with a stone's-edge of
  // margin on top, because the walk lays stones between the control points and
  // the shore's own contour curls back toward the rim past the beach's near edge:
  // the clamp has to hold for the interpolated positions too, not just for the
  // ones solved here.
  const minimumRun = 1.30 * spec.vessel.rimHalfWidth_m + HERO_STEP_RADIUS_SHORE_M + 0.012;
  return steppingStoneNodes({
    ...STEPPING_DISC,
    key: `${key}/path`,
    seed: spec.seed,
    path: HERO_PATH.map(([turn, submerged_m]) => {
      const run = shorelineRun(groundHeightAt, rail, turn, spec.waterline_m - submerged_m, minimumRun);
      const at = railTurnAt(rail, turn, -run);
      return [at.x, at.z] as const;
    }),
    count: HERO_STEP_COUNT,
    radiusStart_m: HERO_STEP_RADIUS_SHORE_M,
    radiusEnd_m: HERO_STEP_RADIUS_DEEP_M,
    stride_m: HERO_STEP_STRIDE_M,
    groundHeightAt,
    level_m: spec.waterline_m,
  });
}

/**
 * The whole set, in the order it should be published.
 *
 * Boulders first because they claim the ground the beds then pack around, and the
 * path last because it is the only family that does not consult the others. The
 * order is also the owner-index order downstream, so it stays fixed.
 */
export function stoneSet(spec: StoneSetSpec): SceneryNode[] {
  return [...stoneSetBoulderNodes(spec), ...stoneSetPebbleNodes(spec), ...stoneSetSteppingNodes(spec)];
}
