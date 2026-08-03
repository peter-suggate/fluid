import type { Quaternion, RigidBodyDescription, Vec3 } from "../model";
import { quaternionMultiply } from "../rigid-body";
import type { SceneryMaterial, SceneryNode, SceneryPlacement } from "../scenery-graph";
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
 * underneath — an irregular graded mass bedded onto whatever ground it is
 * given, with the boulder adding a stem under its cap and the stepping stone
 * trading its dome for a flat tread. That is deliberate: the reference's stones
 * read as one quarry, and the cheapest way to guarantee that is for them to
 * come out of one generator. They share a file for the same reason they share a
 * look.
 *
 * **Nothing that is meant to read as rock is a bare ellipsoid.** That was the
 * one thing this file got wrong for longest and the hardest to see from inside
 * it: every number below spends itself on variety — size, aspect, bedding
 * normal, grade — and all of it arrived at the eye as a tray of eggs, because
 * an ellipsoid has no silhouette irregularity at all and no amount of authored
 * spread can give it one. Every mass is now built by {@link stoneMass}, which is
 * also the single seam where a better field gets swapped in. The exception is
 * deliberate and measured: below the size at which a lumpy outline can be
 * resolved, a marched field is cost with nothing on the other end of it, so the
 * fine shingle stays an ellipsoid and earns its variety from aspect instead.
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

/**
 * Every stone in the set, declared as stone.
 *
 * `surface` is what un-loads the naming rule this file used to carry. Before
 * it, the closure was picked by matching a regular expression against the
 * object's group and tags, so a boulder had to avoid a vocabulary it had no
 * other reason to care about — the comment on `CappedBoulderForm` below is the
 * scar. Saying `stone` here says it once, for the whole quarry.
 */

/**
 * **The seam.** Every solid mass in this quarry — a cap, a shoulder, a cobble —
 * is built here and nowhere else, so the whole set changes shape together.
 *
 * *What it is today.* A jittered sphere lattice clipped to the ellipsoid it
 * would otherwise have been. The parameter that matters is `lobesAcross`: how
 * many spheres of the lattice span the stone. Around seven, which is what the
 * caps used to be authored at, the packing granulates a surface and leaves the
 * ellipse underneath intact — and an ellipse against the sky is the one
 * silhouette a weathered stone never has. Around three or four, each sphere is
 * a third of the stone, the union bulges past the envelope in some directions
 * and falls short of it in others, and what is left is lumpy rather than
 * merely textured. That is the whole reason this file no longer publishes a
 * bare ellipsoid for anything that is meant to read as rock.
 *
 * *What it will be.* An explicit set of oriented anisotropic lobes placed from
 * the seed — `field: "seeded-lobes"` on `SceneryClusterNode` — which is a
 * strictly better fit for a stone than a lattice, because a lattice is
 * isotropic and periodic and a stone is neither. That field is not wired
 * through to the shader yet. When it is, this function is the only edit: every
 * caller hands it an envelope, a coarseness and a seed, which is exactly the
 * vocabulary the seeded field takes.
 *
 * The three ratios below are the field's safety contract, discharged once here
 * rather than trusted per call site — see {@link SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD}
 * in `lib/svo-primitive-abi.ts`, which throws on a violation.
 */
interface StoneMassSpec {
  readonly id: string;
  readonly group: string;
  readonly tags: readonly string[];
  /** The whole placement, so a bedded stone keeps its terrain anchor and its lie. */
  readonly place: SceneryPlacement;
  /** Half-axes of the envelope. The drawn solid is inside it by construction. */
  readonly radius: Vec3;
  /**
   * Lattice spheres spanning the stone's mean horizontal reach.
   *
   * Non-positive publishes the bare ellipsoid instead, which is what the fine
   * shingle takes: below a couple of millimetres on screen a marched field buys
   * a silhouette nobody can resolve.
   */
  readonly lobesAcross: number;
  readonly seed: number;
  readonly material: SceneryMaterial;
}

/**
 * Sphere radius as a fraction of the lattice period.
 *
 * Above `sqrt(3)/2 = 0.866` every cell corner is inside a sphere, so the mass
 * is closed rather than a bag of separate beads. Just under one keeps the
 * overlap small enough that the union still shows where its spheres are.
 */
const STONE_MASS_FLORET_SHARE = 0.98;
/**
 * Sphere displacement from its own cell centre, as a fraction of the period.
 *
 * This is the amplitude of the irregularity: a sphere wandering by `j·p` moves
 * the surface it owns by up to `j·p·sqrt(3)`, which at the coarse periods below
 * is a sixth of the stone. Well under the ABI's 0.3 ceiling, and under the
 * value at which a sphere wanders far enough to open a hole in its own row.
 */
const STONE_MASS_JITTER = 0.16;
/** Smooth-minimum radius, as a fraction of the sphere. Fuses the lattice without filling it. */
const STONE_MASS_SMOOTH_SHARE = 0.30;

/**
 * The ABI's condition, evaluated at the ratios above so a violation is a
 * compile-time-visible number rather than a runtime throw:
 *
 *   (2 - jitter)·period >= floretRadius + smoothRadius
 *   (2 - 0.16)·p        >= (0.98 + 0.30·0.98)·p
 *   1.84·p              >= 1.274·p                     ✓ 44 % margin
 *
 * It holds for every period because both sides are proportional to it, which is
 * why the caller may choose `lobesAcross` freely.
 */
const STONE_MASS_NEIGHBOURHOOD_MARGIN =
  (2 - STONE_MASS_JITTER) / (STONE_MASS_FLORET_SHARE * (1 + STONE_MASS_SMOOTH_SHARE));
if (!(STONE_MASS_NEIGHBOURHOOD_MARGIN >= 1)) {
  throw new RangeError(
    `Stone mass ratios violate the cluster neighbourhood condition by ${(1 / STONE_MASS_NEIGHBOURHOOD_MARGIN).toFixed(3)}x`,
  );
}

/** A cluster's seed is a u32, and a caller's may be anything an integer hash produced. */
const seed32 = (seed: number): number => (seed >>> 0) || 1;

function stoneMass(spec: StoneMassSpec): SceneryNode {
  const { id, group, tags, place, radius, material } = spec;
  if (!(spec.lobesAcross > 0)) {
    return { kind: "ellipsoid", id, group, tags, place, radius, material };
  }
  // The mean *horizontal* reach, not the largest semi-axis. A bedded stone is
  // flattened, and a period taken off its height would put thirty spheres
  // across its plan; one taken off its length would put a third of a sphere
  // through its thickness and clip the lot back to the bare ellipsoid.
  const reach = radius.x + radius.z;
  const latticePeriod = reach / spec.lobesAcross;
  const floretRadius = STONE_MASS_FLORET_SHARE * latticePeriod;
  return {
    kind: "cluster", id, group, tags,
    place,
    lobe: radius,
    floretRadius,
    latticePeriod,
    jitter: STONE_MASS_JITTER,
    smoothRadius: STONE_MASS_SMOOTH_SHARE * floretRadius,
    seed: seed32(spec.seed),
    material,
  };
}

const stone = (value: number): SceneryMaterial => ({ palette: "stone", value, surface: "stone" });

/**
 * The palette band the whole quarry is authored in.
 *
 * Outside it a stone stops reading as the same fired white as everything around
 * it and starts reading as a different material, which is the one thing a
 * monochrome set cannot survive. Every form's values are clamped into it rather
 * than trusted, because a form is data a caller can write.
 *
 * The band moved up once and it is worth recording why, because the numbers
 * below were not wrong when they were written. The hero scene used to shade
 * every prop through a closure with its own colour grain, so an authored 0.70
 * arrived at the eye with the grain's own highlights on top of it. It now
 * renders every surface through a flat `plaster` closure with no grain at all
 * and a key light raised so that white reads as white — which means the
 * authored value *is* the albedo, and the set's old floor of 0.62 came back as
 * a grey pebble beside a white rim. Everything here is now authored for a
 * creamy white: the darkest thing in the quarry is a boulder's footing in its
 * own shadow, and even that is 0.78. The ceiling stays under 1 because
 * `canonicalSvoMaterialRecord` rejects a channel above it.
 */
export const STONE_VALUE_MINIMUM = 0.78;
export const STONE_VALUE_MAXIMUM = 0.94;
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
 * This used to carry a naming prohibition — nothing here could be called a
 * "mushroom", because `svoMaterialFunctionIdForEnvironmentProxy` tested that
 * word on the same line as `hose` and `cloth` and would have shaded a boulder
 * as an organic one. The forms now declare `surface: "stone"` outright, so the
 * word is free again and the shape can be described by what it looks like.
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
   * How many lobes of the mass span the cap. See {@link StoneMassSpec}.
   *
   * This replaced a `capRumple` depth, and the change is not a rename. A rumple
   * depth of a quarter of the cap put seven spheres across it, which granulated
   * the *surface* and left the ellipse under it exactly where it was — so the
   * stone still drew a perfect ellipse against the sky, which is precisely what
   * the parameter was added to stop. Three or four says the same thing in the
   * units that decide it: each lobe is a third of the stone, so the silhouette
   * itself goes lumpy.
   *
   * A count rather than a length because it is a *proportion* of the stone.
   * Weathering that stayed a fixed size would make a small cobble look like a
   * scale model of a large one. Zero leaves the cap the smooth ellipsoid it
   * once was, and nothing in this file asks for that any more.
   */
  readonly capLobesAcross: number;
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

/**
 * Four forms, because a family is proportions and not sizes.
 *
 * The set used to be one form at four `capRadius_m`, and the render said what
 * that is: four photocopies at four enlargements, which the eye reads as one
 * object repeated rather than as a group of stones. The reference's group is
 * nothing like it — it holds a tall stone on a slim stem under a wide flat cap,
 * a squat broad one whose cap is nearly as thick as it is wide, and a low
 * seated cobble with barely a stem at all, and *those* differences are what
 * make it read as a quarry. So the four below differ in every ratio that draws
 * a silhouette: cap flattening runs 0.30 to 0.62, stem height 0.18 to 1.75 cap
 * radii, and the overhang from a full brim to none.
 *
 * `capRadius_m` on a form is only its natural size; the hero set overrides it
 * per stone. What a caller is choosing here is the *shape*.
 */

/** Tall: a slim stem under a wide flat cap, with the deepest undercut in the set. */
export const BOULDER_TALL_PARASOL: CappedBoulderForm = Object.freeze({
  capRadius_m: 0.085,
  capFlatten: 0.30,
  capDepthShare: 0.90,
  stemTopShare: 0.40,
  stemBaseShare: 0.54,
  stemHeightShare: 1.75,
  footingShare: 0.72,
  footingHeightShare: 0.18,
  shoulderShare: 0.58,
  shoulderOffsetShare: 0.40,
  capLobesAcross: 4.2,
  lean_rad: 0.09,
  value: 0.90,
});

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
  capLobesAcross: 3.6,
  lean_rad: 0.07,
  value: 0.89,
});

/**
 * Squat and broad: the mass in the reference's group that reads as *heavy*.
 *
 * Its cap is half as thick as it is wide and sits on a stem two thirds the
 * cap's own width, so there is a lip rather than a brim and the whole stone is
 * one boulder rather than a plate on a post. The oval plan is doing work here —
 * a squat stone drawn on a circle is a drum — and the shoulder is the largest
 * in the set, because at three lobes across the mass this is the individual
 * whose silhouette is most obviously a stone.
 */
export const BOULDER_SQUAT_ANVIL: CappedBoulderForm = Object.freeze({
  capRadius_m: 0.080,
  capFlatten: 0.50,
  capDepthShare: 0.76,
  stemTopShare: 0.68,
  stemBaseShare: 0.82,
  stemHeightShare: 0.42,
  footingShare: 0.96,
  footingHeightShare: 0.26,
  shoulderShare: 0.84,
  shoulderOffsetShare: 0.46,
  capLobesAcross: 3.0,
  lean_rad: 0.05,
  value: 0.88,
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
  capDepthShare: 0.82,
  // A lip rather than a brim, and the stem still has to taper: a cobble sits on
  // a foot wider than the waist under its own crown, like every other stone in
  // the quarry, or the two solids meet in a re-entrant corner.
  stemTopShare: 0.86,
  stemBaseShare: 0.92,
  stemHeightShare: 0.18,
  footingShare: 0.98,
  footingHeightShare: 0.24,
  shoulderShare: 0.78,
  shoulderOffsetShare: 0.34,
  // A cobble is a river stone, and a river stone is worn rounder than a
  // weathered cap: fewer, larger lobes rather than none.
  capLobesAcross: 2.8,
  lean_rad: 0.10,
  value: 0.89,
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
      // The ramps off the cap's value are a *shading* device — a stone is darker
      // where it turns away from the sky — and they are a third of what they
      // were, because the closure under them no longer adds any grain of its
      // own. A 0.14 step used to read as one stone in two lights; on flat
      // plaster it reads as two stones.
      material: value(spec.value - 0.06 + 0.04 * hash01(seed + 9)),
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
      material: value(spec.value - 0.045 + 0.03 * hash01(seed + 10)),
    },
    stoneMass({
      id: `${key}/cap`, group, tags: [...BOULDER_TAGS, "cap"],
      place: { position: V(0, capCenterY, 0) },
      radius: V(capRadius_m, capSemiHeight, capRadius_m * spec.capDepthShare * (0.96 + 0.10 * hash01(seed + 11))),
      lobesAcross: spec.capLobesAcross,
      seed: seed + 12,
      material: value(spec.value - 0.01 + 0.04 * hash01(seed + 12)),
    }),
  ];

  if (spec.shoulderShare > 0) {
    // Fused rather than placed — it sits well inside the cap's own surface and
    // only shows where it pushes past it, so what the eye reads is one stone
    // with a shoulder rather than two stones touching.
    children.push(
      stoneMass({
        id: `${key}/cap-lobe`, group, tags: [...BOULDER_TAGS, "cap"],
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
        // A shoulder is a smaller mass than the cap it grows out of, so the
        // same lobe *count* would make it finer-grained than its own stone and
        // give the join away. Scaled by the two envelopes instead, the two
        // halves of one boulder are packed at one period.
        lobesAcross: spec.capLobesAcross * spec.shoulderShare,
        seed: seed + 18,
        material: value(spec.value - 0.01 + 0.04 * hash01(seed + 18)),
      }),
    );
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
  /**
   * Drawn *half-width* at or above which a stone is published as an irregular
   * mass rather than as an ellipsoid, and how many lobes span one when it is.
   *
   * The half-width the stone actually ends up with — `max(semi.x, semi.z)` —
   * and not the lane radius it was drawn from. Those differ by the aspect band,
   * which reaches 1.58, so a floor applied to the lane radius let a dozen
   * stones out at 30 mm wide with a bare elliptical outline while the ones
   * beside them at 25 mm got a silhouette. The rule is about what is on screen,
   * so it is measured on what is on screen.
   *
   * The line this file was told to draw, and it is a *cost* line rather than a
   * taste one. Every stone in the reference is an irregular ovoid and none of
   * them is a sphere, so on looks alone every stone here would be a
   * {@link stoneMass}. But a cluster is a marched field — up to
   * `SVO_PRIMITIVE_MARCH_ITERATIONS` field evaluations against an ellipsoid's
   * one closed-form root — and a bed is the densest population in the scene.
   *
   * So the split is by *legibility*. At the hero camera the pond spans about
   * 1.15 m across the frame, so a stone 24 mm across covers on the order of ten
   * pixels: enough that a lumpy outline is a lumpy outline, and enough that a
   * round one reads as a bead. Below that the silhouette is a smudge whatever
   * shape it is, and what still reads is the stone's *aspect* — which the
   * spreads above deliver for free. The floor therefore buys the irregularity
   * exactly where it can be seen and pays nothing for it where it cannot.
   *
   * `clusterLobesAcross` is a band, not a number, because the lattice is
   * anchored on the node's own origin: two stones at one lobe count and one
   * envelope would carry the same arrangement of lumps, modulated only by the
   * per-cell jitter. Drawing the count per stone re-phases the lattice against
   * the envelope and is the cheapest decorrelation available.
   */
  readonly clusterFloor_m: number;
  readonly clusterLobesAcross: readonly [number, number];
  /** Palette value band the bed's stones are drawn from. */
  readonly value: readonly [number, number];
}

/**
 * The bank bed: cobbles heaped where a bed is rich, thinning to shingle where it
 * is lean. This is the form the reference's upper-left bank is made of.
 *
 * **The grade is the subject.** The first version of this bed graded its mean
 * radius 16 mm to 8.5 mm — a factor of 1.9 — and the render said what a factor
 * of 1.9 looks like from a metre away, which is nothing: an even necklace of
 * identical beads round the pool. The reference is not remotely that. It heaps
 * cobbles the size of a plum against the foot of its big stones and runs out
 * into shingle you could not pick up individually, and measured against the
 * coping — 1.15 m across 1 630 pixels — that is 80 mm down to about 15 mm.
 *
 * So the mean now grades 42 mm to 14 mm across, a factor of 2.9, and the size
 * spread inside it widens to 0.46-1.55 of that mean. Compounded with
 * `crossGrade` running the same way outward, the *population* runs from a 6 mm
 * flake to an 86 mm cobble: a factor of fourteen, against the four the bed used
 * to hold. That is the whole difference between "graded" as a word in a comment
 * and graded as something the frame shows.
 *
 * The packing stays deliberately loose, and that half is not aesthetic. The
 * lighting hierarchy binds `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK = 64`
 * primitives per 200 mm brick and silently drops the surplus, so the densest
 * stones in a bed are the ones that stop being lit. Coarsening the grain is the
 * cheapest possible relief for that: stone count goes as the inverse square of
 * the spacing, so heaping *larger* cobbles against the boulders costs fewer
 * records than the fine ones it replaces. `tests/stone-set.test.ts` measures the
 * worst brick rather than trusting this note. See `docs/SVO_FINE_VOXEL_CAPACITY.md`.
 */
export const PEBBLE_GRADED_COBBLE: PebbleBedForm = Object.freeze({
  radiusRich_m: 0.0210,
  radiusLean_m: 0.0072,
  sizeSpread: [0.46, 1.55] as const,
  alongPacking: 0.80,
  acrossPacking: 0.88,
  // A wide *aspect* band, not just a wide size band, and much wider than it was.
  // The reference's beds hold flat slabs lying on their side beside near-round
  // cobbles: the length runs to half again the width and the height falls to a
  // third of it, so a stone drawn at the ends of these bands is a flake rather
  // than an egg. This is also what carries the fine shingle, which is below the
  // size at which a marched silhouette is worth paying for — an ellipsoid at
  // 0.34 height share does not read as a sphere.
  lengthShare: [0.88, 1.58] as const,
  heightShare: [0.34, 0.88] as const,
  widthShare: [0.60, 1.16] as const,
  widthWander: 0.34,
  // Coarsening *outward*, because the bank band grows away from the water: the
  // reference heaps its biggest cobbles up against the boulders and fines to
  // shingle at the shoreline, and a bed of one grain reads as gravel however
  // wide the size spread inside it is.
  crossGrade: 1.34,
  sinkShare: 0.34,
  moundShare: 0.55,
  moundWidthFloor_m: 0.09,
  // Deeper than it was but held to the last quarter of the band, and the power
  // is the load-bearing half of that pair. The bed coarsens outward, so a
  // thinning that bit at half width would be deleting precisely the largest
  // cobbles it had just graded up — the first pass at these numbers did exactly
  // that and capped the bed at 58 mm when it was authored to reach 86.
  edgeThinning: 0.74,
  edgeThinningPower: 2.6,
  bedTiltFollow: 0.72,
  bedNormalJitter: 0.42,
  // 24 mm across is about ten pixels at the hero camera: the size at which a
  // round outline starts reading as a bead. Three to four and a half lobes puts
  // one lump per third of the stone.
  clusterFloor_m: 0.012,
  clusterLobesAcross: [3.0, 4.5] as const,
  value: [0.82, 0.93] as const,
});

/**
 * The water's-edge course: two thirds the grain, flatter stones, laid tighter and
 * fining as it runs down. This is what the reference shows all the way round its
 * right and back, where the bed is one course of small stones on the rim's inner
 * slope, and what its left shore fines *to* as the shelf runs into the water.
 *
 * Its cluster floor is deliberately *higher* than the bank bed's, not lower.
 * This band exists to fine out into the water, so most of it is below the size
 * at which a marched outline earns its cost; only the few cobbles at its rich
 * end are worth one, and they are the ones sitting where the bank bed's drift
 * runs down to meet it.
 */
export const PEBBLE_FINE_SHINGLE: PebbleBedForm = Object.freeze({
  radiusRich_m: 0.0125,
  radiusLean_m: 0.0058,
  sizeSpread: [0.58, 1.48] as const,
  alongPacking: 0.82,
  acrossPacking: 0.90,
  lengthShare: [0.92, 1.52] as const,
  heightShare: [0.32, 0.82] as const,
  widthShare: [0.62, 1.14] as const,
  widthWander: 0.22,
  crossGrade: 0.62,
  sinkShare: 0.36,
  moundShare: 0.40,
  moundWidthFloor_m: 0.07,
  edgeThinning: 0.72,
  edgeThinningPower: 1.4,
  bedTiltFollow: 0.78,
  bedNormalJitter: 0.46,
  clusterFloor_m: 0.014,
  clusterLobesAcross: [2.8, 4.0] as const,
  value: [0.83, 0.94] as const,
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
   * Share of the candidate stones kept at a fraction of a turn, in [0, 1].
   * Omitted keeps them all.
   *
   * The *other* half of "a drift", and the half a band width alone cannot say.
   * Narrowing the band toward a drift's tail thins the bed by making it
   * narrower, which from a raised camera reads as a bed that has been trimmed
   * with a straightedge. A drift does not narrow, it *disperses*: the same
   * spread of ground goes from stones resting on stones, to stones with plaster
   * between them, to three stones and then none. This is the knob for that, and
   * it is on the spec rather than the form because it is a property of the run.
   */
  readonly densityAt?: (turn: number) => number;
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
  const densityAt = spec.densityAt ?? (() => 1);
  const children: SceneryNode[] = [];
  const group = `${key}-stone`;
  let published = 0;
  let arc = 0;
  let course = 0;

  while (arc < rail.length_m) {
    const turn = railAt(rail, arc, 0).turn;
    const grade = Math.min(1, Math.max(0, gradeAt(turn)));
    const density = Math.min(1, Math.max(0, densityAt(turn)));
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
      // Dispersal along the run, then thinning across the band. Two independent
      // hashes, because a stone dropped by both is a stone dropped once and the
      // two effects have to compose rather than mask one another.
      if (hash01(seed + 11) >= density) continue;
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

      children.push(stoneMass({
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
        // Tested on the stone's own drawn half-width rather than on the lane's
        // mean, so a large stone in a fine part of the bed still gets a
        // silhouette and a small one in a coarse part does not pay for one it
        // cannot show.
        lobesAcross: Math.max(semi.x, semi.z) >= spec.clusterFloor_m
          ? band(spec.clusterLobesAcross, hash01(seed + 10)) : 0,
        seed: seed + 12,
        material: value(band(spec.value, hash01(seed + 6))),
      }));
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
  value: 0.89,
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
 * One stone's solved place on the path, before anything is drawn or collided.
 *
 * Extracted so that the scenery and the solver's collider cannot disagree. A
 * stepping stone is the one family in this file that has to exist twice —
 * scenery is invisible to the solver, so a disc that is only scenery is a disc
 * water flows through, which on a wading path is the one thing it must not do —
 * and the way two representations of the same object stay in agreement is that
 * there is only ever one solve.
 */
export interface SteppingStonePlacement {
  readonly index: number;
  /** Plan position in world metres. */
  readonly at_m: readonly [number, number];
  /** Tread radius, after the per-stone jitter. */
  readonly radius_m: number;
  readonly yaw_rad: number;
  readonly treadTop_m: number;
  readonly treadBottom_m: number;
  readonly footingBottom_m: number;
  readonly footingTop_m: number;
  /** Seed the drawn stone's per-part variation is taken from. */
  readonly seed: number;
}

/**
 * Walk the path and solve every stone against the bed and the level.
 *
 * The whole of the arithmetic that used to live inside the node emitter. See
 * `steppingStoneNodes` for why the tread's *top* is what is placed and the
 * footing under it is whatever the local depth demands.
 */
export function steppingStonePlacements(spec: SteppingStonePathSpec): SteppingStonePlacement[] {
  const { seed, count, groundHeightAt } = spec;
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

  const placements: SteppingStonePlacement[] = [];
  let arc = 0;
  for (let index = 0; index < count; index += 1) {
    const stoneSeed = seed + 0x77_00 + 613 * index;
    const radius = radiusOf(index);
    if (index > 0) arc += radiusOf(index - 1) + radius + spec.stride_m;
    const [x, z] = pointAtArc(arc);
    const ground = groundHeightAt(x, z);
    const level = spec.level_m ?? -Infinity;
    const treadTop = Math.max(
      level + spec.freeboard_m * (0.85 + 0.3 * hash01(stoneSeed + 1)),
      ground + spec.emergenceShare * spec.tread_m,
    );
    const treadBottom = treadTop - spec.tread_m;
    placements.push({
      index,
      at_m: [x, z],
      radius_m: radius,
      yaw_rad: 2 * Math.PI * hash01(stoneSeed + 2),
      treadTop_m: treadTop,
      treadBottom_m: treadBottom,
      footingBottom_m: ground - spec.bed_m,
      footingTop_m: treadBottom + 0.004,
      seed: stoneSeed,
    });
  }
  return placements;
}

/**
 * How far inside the drawn footing's narrowest radius the collider sits.
 *
 * Three per cent, which on the largest plate is 1.2 mm. Exact tangency would
 * leave the two surfaces coincident along a whole circle, and which one a ray
 * resolves there is a coin toss taken per pixel.
 */
const STEPPING_COLLIDER_INSET = 0.97;

/**
 * The collider under a stepping stone: one static cylinder, inscribed.
 *
 * **Why one cylinder.** The drawn stone is three primitives — a tapered footing,
 * the tread, and a crown that softens its edge — and a rigid body is one shape.
 * At the lattice this scene ships on, 25 mm, a 32-42 mm plate is 2.6 to 3.4
 * cells across and the footing's taper is entirely sub-cell: the solver cannot
 * represent the difference between the plate and its foot, so paying two of the
 * twelve available body slots per stone to state it would buy nothing.
 *
 * **Why inscribed rather than circumscribed.** The scene opens dry, so the whole
 * footing is on show, and the rigid renderer colours bodies from a hard-coded
 * shape palette in `lib/webgpu-rigid-body.ts` — a cylinder comes out purple.
 * A collider at the tread's own radius would therefore stand a purple collar
 * proud of the footing on every dry frame. Taken at the narrowest the drawn
 * solid ever gets over the height it spans — the footing's top radius, less a
 * little — the collider is strictly inside the stone at every height, so it can
 * never be the nearest surface and there is nothing to z-fight with. The water
 * reaches about three millimetres nearer the disc's foot than the stone does,
 * which is a fifth of a cell.
 */
export function steppingStoneBodies(spec: SteppingStonePathSpec, keyPrefix = spec.key): RigidBodyDescription[] {
  // A stone's own material, so a body that is ever asked what it weighs answers
  // in granite rather than in water.
  const density_kg_m3 = 2650;
  return steppingStonePlacements(spec).map((placement) => {
    const radius = placement.radius_m * spec.footingTopShare * STEPPING_COLLIDER_INSET;
    const height = Math.max(1e-4, placement.treadTop_m - placement.footingBottom_m);
    return {
      id: `${keyPrefix}/step-${placement.index}`,
      name: `Stepping stone ${placement.index + 1}`,
      shape: "cylinder" as const,
      dimensions_m: { x: radius, y: height, z: radius },
      density_kg_m3,
      position_m: { x: placement.at_m[0], y: 0.5 * (placement.footingBottom_m + placement.treadTop_m), z: placement.at_m[1] },
      orientation: quaternionAboutY(placement.yaw_rad),
      linearVelocity_m_s: V(0, 0, 0),
      angularVelocity_rad_s: V(0, 0, 0),
      restitution: 0.05,
      friction: 0.85,
      motion: "static" as const,
    };
  });
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
  const { key } = spec;
  const nodes: SceneryNode[] = [];
  for (const placement of steppingStonePlacements(spec)) {
    const { index, radius_m: radius, seed: stoneSeed } = placement;
    const [x, z] = placement.at_m;
    const { treadTop_m: treadTop, treadBottom_m: treadBottom, footingBottom_m: footingBottom, footingTop_m: footingTop } = placement;
    const group = `${key}-stone`;

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
        orientation: quaternionAboutY(placement.yaw_rad),
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
          material: value(spec.value - 0.07 + 0.03 * hash01(stoneSeed + 3)),
        },
        (() => {
          // This used to be a cylinder with a paper-thin ellipsoid intersecting
          // its top plane. At the hero camera the crown's entire upward-to-side
          // normal transition occupied about one pixel, so sampling turned its
          // dark grazing band into a broken ring of dots. One rounded-cylinder
          // SDF has a single zero set and a fillet large enough to resolve; its
          // bottom and crown apex remain exactly where the old pair put them.
          const crownRise = radius * spec.dome;
          const top = treadTop + crownRise;
          const halfHeight = 0.5 * (top - treadBottom);
          const edgeRadius = Math.min(
            0.48 * halfHeight,
            Math.max(crownRise, radius * 0.12),
          );
          return {
            kind: "cylinder" as const,
            id: `${key}/step-${index}/tread`,
            group,
            tags: [...STEPPING_TAGS, "tread", "crown"],
            place: { position: V(0, 0.5 * (treadBottom + top), 0) },
            radius,
            halfHeight,
            edgeRadius,
            material: value(spec.value - 0.015 + 0.04 * hash01(stoneSeed + 4)),
          };
        })(),
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
 * instead of half of one, and the caps grade 2.7 : 1 across it — the receding
 * line the reference draws, rather than a row.
 *
 * The third mistake is the one this table now carries the fix for, and it is not
 * about *where*. All four used to be `BOULDER_MUSHROOM_CAP` at four radii and
 * four even steps of the rail, and both halves of that were wrong. Four
 * enlargements of one drawing read as one object repeated, whatever their sizes;
 * and four even steps read as a fence, because a group of stones the eye
 * accepts as *placed* has no interval in it that repeats. So each row now names
 * its own form, and the turns are 0.030, 0.038 and 0.062 apart — a close pair,
 * a third leaning in, and the small one set out on its own.
 */
interface BoulderPlacement {
  /** Fraction of a turn round the rail. */
  readonly turn: number;
  /** Plan distance outside the coping's crest centreline. */
  readonly offset_m: number;
  /** Cap semi-axis on its long horizontal direction: the boulder's scale. */
  readonly capRadius_m: number;
  /** Which stone this is. The whole point of the family: they are not one shape. */
  readonly form: CappedBoulderForm;
}

const HERO_BOULDERS: readonly BoulderPlacement[] = Object.freeze([
  // The anchor: broad, heavy, seated, and the nearest of the four to the rim so
  // it reads in front of the rest.
  { turn: 0.688, offset_m: 0.126, capRadius_m: 0.082, form: BOULDER_SQUAT_ANVIL },
  // Its partner, half a cap away and standing more than twice as tall on a stem
  // two fifths the width of what it carries. This is the reference's parasol,
  // and the pair is what makes the group a group.
  { turn: 0.658, offset_m: 0.168, capRadius_m: 0.068, form: BOULDER_TALL_PARASOL },
  // The third leans back into the pair rather than continuing the line.
  { turn: 0.620, offset_m: 0.132, capRadius_m: 0.047, form: BOULDER_MUSHROOM_CAP },
  // Small and low, set out along the bank on its own: the thing that stops the
  // group reading as a wall.
  { turn: 0.558, offset_m: 0.152, capRadius_m: 0.030, form: BOULDER_ROUNDED_COBBLE },
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
 *
 * **A drift, not a band with a taper.** The first version of this table gave
 * each place one `grade`, so the grain inside a drift was *constant* — the
 * weighted average of a single contributing entry is that entry, whatever
 * weight it has — and only the band's width faded. What the render showed was
 * exactly what that describes: a necklace of identically sized stones round a
 * pool, narrowing at the ends. The reference's drifts do the opposite of that.
 * They heap their largest cobbles hard against the foot of the big stones and
 * then run *out*, coarse to fine and dense to sparse over the same stretch of
 * ground, with the band still a hand's breadth wide where the last three stones
 * are. So a place now carries both ends of both ramps, and the drift's own
 * weight interpolates them.
 */
interface PebbleDrift {
  /** Middle of the drift, as a rail turn. */
  readonly turn: number;
  /** Half its run in turns. The bed fades to nothing over the outer part of it. */
  readonly halfLength: number;
  /** Widest the bank band gets here, in plan metres. Zero leaves the outside bare. */
  readonly bankWidth_m: number;
  /** Grade at the drift's heart and where it fades out, in [0, 1]. 1 is the coarsest cobble. */
  readonly grade: number;
  readonly gradeTail: number;
  /** Share of the candidate stones kept, at the heart and at the tail. */
  readonly density: number;
  readonly densityTail: number;
  /** Whether the water's-edge course runs here too. */
  readonly shore: boolean;
}

const HERO_PEBBLE_DRIFTS: readonly PebbleDrift[] = Object.freeze([
  // The heap at the boulders' feet. Its heart is at the middle of the family
  // rather than beside it — the whole subject of the reference's main bed is
  // that the biggest cobbles are the ones wedged against the biggest stones —
  // and it runs out both ways to a quarter of the grain and a third of the
  // stones. It is the widest band in the set and the only one that reaches the
  // full 42 mm mean.
  { turn: 0.640, halfLength: 0.108, bankWidth_m: 0.150, grade: 1.00, gradeTail: 0.30, density: 1.00, densityTail: 0.30, shore: false },
  // The beach: where the shelf runs down into the water, and the only place
  // both bands meet. Fine from the start and finer as it goes, because this is
  // the end of the drift above rather than a bed of its own.
  { turn: 0.494, halfLength: 0.098, bankWidth_m: 0.072, grade: 0.50, gradeTail: 0.10, density: 0.92, densityTail: 0.26, shore: true },
  // The course along the right-hand rim, where the tree's terrace comes down to
  // the coping. Narrow, fine, and the one the reference runs *inside* the rim.
  { turn: 0.080, halfLength: 0.086, bankWidth_m: 0.050, grade: 0.24, gradeTail: 0.05, density: 0.84, densityTail: 0.20, shore: true },
  // The loose scatter outside the coping at the near right: a dozen stones, no
  // bed under them, which is what a narrow band at half density produces.
  { turn: 0.230, halfLength: 0.056, bankWidth_m: 0.034, grade: 0.15, gradeTail: 0.02, density: 0.52, densityTail: 0.14, shore: false },
]);

/** How much of a drift's run is full width before it fades out. */
function driftWeight(drift: PebbleDrift, turn: number): number {
  return ramp(1 - turnDelta(turn, drift.turn) / drift.halfLength);
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
      ...placement.form,
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
    HERO_PEBBLE_DRIFTS.reduce((widest, drift) =>
      Math.max(widest, drift.bankWidth_m * driftWeight(drift, turn)), 0);
  const shoreWeightAt = (turn: number): number =>
    HERO_PEBBLE_DRIFTS.reduce((most, drift) =>
      Math.max(most, drift.shore ? driftWeight(drift, turn) : 0), 0);
  // The grade a drift is at *here*, ramped between its heart and its tail by its
  // own weight, and then averaged over the drifts that reach this turn so a
  // stone in the overlap between two of them is graded between the two rather
  // than jumping at the join. The inner ramp is the whole fix: without it the
  // average of one contributing drift is that drift's own number, so every bed
  // came out at one grain.
  const gradeAt = (turn: number): number => {
    let weight = 0, graded = 0;
    for (const drift of HERO_PEBBLE_DRIFTS) {
      const w = driftWeight(drift, turn);
      weight += w; graded += w * (drift.gradeTail + (drift.grade - drift.gradeTail) * w);
    }
    return weight > 0 ? graded / weight : 0;
  };
  // Density takes the *densest* drift rather than the average, to match the way
  // the band width is resolved: where two drifts overlap the bed is the union of
  // them, so it cannot be sparser than either.
  const densityAt = (turn: number): number =>
    HERO_PEBBLE_DRIFTS.reduce((most, drift) => {
      const w = driftWeight(drift, turn);
      return Math.max(most, drift.densityTail + (drift.density - drift.densityTail) * w);
    }, 0);

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
      densityAt,
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
      densityAt,
      level_m: spec.waterline_m,
      avoid,
      within: onPlaster,
    }),
  ];
}

/**
 * The wading path's spec, solved against the shore once.
 *
 * Both the drawn plates and the colliders under them come from here, so a
 * control point moving in the vessel moves the stone and the volume of water it
 * displaces together. Two independent solves of the same contour is exactly the
 * drift the whole file is arranged to prevent.
 */
function heroSteppingPath(spec: StoneSetSpec): SteppingStonePathSpec {
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
  return {
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
  };
}

/** The five plates, wading in along a contour of the shore. */
export function stoneSetSteppingNodes(spec: StoneSetSpec): SceneryNode[] {
  return steppingStoneNodes(heroSteppingPath(spec));
}

/**
 * The same five plates as solid bodies the water has to part around.
 *
 * Scenery has no solver term, so without these the disc path is a picture the
 * jet pours straight through — which on the one family whose whole subject is
 * standing in water is the difference between a pond and a photograph of one.
 */
export function stoneSetSteppingBodies(spec: StoneSetSpec): RigidBodyDescription[] {
  return steppingStoneBodies(heroSteppingPath(spec));
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

// ---------------------------------------------------------------------------
// What the set looks like from outside it
// ---------------------------------------------------------------------------

/**
 * Where one point-placed stone actually stands, for anything that has to lay
 * out *around* the set rather than inside it.
 *
 * Measured off the nodes the generator emits rather than re-derived beside
 * them, because a second derivation is a second answer. `hero-layout.ts` kept
 * one of those: four boulder rows and five disc rows in authored world metres,
 * written against an older pond and then left behind by it — by the time this
 * was added the boulder rows were on the wrong side of the water and a disc row
 * failed its own clearance assertion while the stone it claimed to describe was
 * comfortably clear. A composition that cannot see the set cannot lay anything
 * out around it, and a composition that sees a stale copy is worse than one
 * that sees nothing.
 */
export interface StoneStation {
  /** The node id this describes. */
  readonly key: string;
  /** Plan position in world metres: the stone's own anchor. */
  readonly at_m: readonly [number, number];
  /** Widest full plan extent, and full vertical extent. */
  readonly width_m: number;
  readonly height_m: number;
  /** Lowest point, in the frame the stone's own group is placed in. */
  readonly base_m: number;
}

/** Half-extents of one primitive in its own local frame. A cluster is its envelope. */
function primitiveHalfExtent(node: SceneryNode): Vec3 | undefined {
  switch (node.kind) {
    case "ellipsoid": return node.radius;
    case "cluster": return node.lobe;
    case "cylinder": return V(node.radius, node.halfHeight, node.radius);
    case "cone": return V(Math.max(node.baseRadius, node.topRadius), node.halfHeight, Math.max(node.baseRadius, node.topRadius));
    default: return undefined;
  }
}

/**
 * One published stone, reduced to the box a layout needs.
 *
 * The group's own orientation is deliberately *not* applied. A boulder's tilt is
 * a few degrees and rotating each part's box by it would inflate the envelope by
 * more than the lean actually moves anything, which for a composition — whose
 * whole use of this is "what else fits beside it" — is the wrong error to make.
 */
function stationOf(node: SceneryNode): StoneStation {
  if (node.kind !== "group") throw new TypeError(`${node.id} is not a placed stone`);
  const at = node.place?.position ?? V(0, 0, 0);
  let width = 0, low = Infinity, high = -Infinity;
  for (const child of node.children) {
    const half = primitiveHalfExtent(child);
    if (!half) continue;
    const centre = child.place?.position ?? V(0, 0, 0);
    width = Math.max(width, 2 * (Math.abs(centre.x) + half.x), 2 * (Math.abs(centre.z) + half.z));
    low = Math.min(low, centre.y - half.y);
    high = Math.max(high, centre.y + half.y);
  }
  return { key: node.id, at_m: [at.x, at.z], width_m: width, height_m: high - low, base_m: low };
}

/** The boulder family, as the boxes a composition has to keep clear of. */
export function stoneSetBoulderStations(spec: StoneSetSpec): StoneStation[] {
  return stoneSetBoulderNodes(spec).map(stationOf);
}

/**
 * The wading path, as the boxes a composition has to keep clear of.
 *
 * `base_m` is a *world* height here and a seat-relative one for a boulder, and
 * that asymmetry is the species' rather than an oversight: a stepping stone is
 * the one family authored against the water instead of against the ground, so
 * its parts carry absolute heights and its group carries none.
 */
export function stoneSetSteppingStations(spec: StoneSetSpec): StoneStation[] {
  return stoneSetSteppingNodes(spec).map(stationOf);
}
