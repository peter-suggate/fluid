import type { Quaternion, RigidBodyDescription, Vec3 } from "../core/model";
import { packMaterialOwner, SPARSE_BRICK_NO_OWNER, unpackMaterialOwner } from "./sparse-brick-octree";
import {
  evaluateSvoFieldProgram,
  svoFieldProgramExtent_m,
  type SvoFieldProgram,
} from "./svo-field-program";
import {
  SVO_PRIMITIVE_KIND_FLAGS,
  SVO_PRIMITIVE_KIND_TABLE,
  svoPrimitiveKindConstantsWGSL,
  type SvoPrimitiveKindName,
} from "./svo-primitive-kinds";
import { sampleSvoProceduralNoise } from "./svo-procedural-material";
import { terrainHeightAt, terrainNormalAt, type TerrainDescription } from "../core/terrain";
import { materialIdForRigidShape } from "../core/voxel-scene";

/** Four 16-byte lanes, directly usable as a WebGPU storage-buffer array. */
export const SVO_PRIMITIVE_RECORD_STRIDE_BYTES = 64;
export const SVO_PRIMITIVE_RECORD_WORDS = SVO_PRIMITIVE_RECORD_STRIDE_BYTES / Uint32Array.BYTES_PER_ELEMENT;
export const SVO_PRIMITIVE_INVALID_REFERENCE = 0xffff_ffff;
/** Fixed bisection ceiling shared by the CPU oracle and f32 WGSL closest-point solve. */
export const SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS = 64;
/**
 * Sphere-trace ceiling for kinds whose hit is solved by marching their exact
 * distance rather than by a closed-form root. Callers hand in a bounded
 * interval — one traversed voxel in the renderer — so the march starts within a
 * cell of the surface and this ceiling is never the reason a hit is missed.
 */
export const SVO_PRIMITIVE_MARCH_ITERATIONS = 48;

/**
 * Stable on-GPU primitive tags. Zero remains reserved for an invalid/empty
 * record.
 *
 * Read from the kind table rather than restated, so the numbers a record is
 * packed with and the numbers the shader compares against cannot drift. The key
 * names are the historical camelCase ones because they are already spelled out
 * across the tests and diagnostics that read this object.
 */
export const SVO_PRIMITIVE_KINDS = Object.freeze({
  sphere: SVO_PRIMITIVE_KIND_TABLE.sphere.code,
  box: SVO_PRIMITIVE_KIND_TABLE.box.code,
  capsule: SVO_PRIMITIVE_KIND_TABLE.capsule.code,
  cylinder: SVO_PRIMITIVE_KIND_TABLE.cylinder.code,
  ellipsoid: SVO_PRIMITIVE_KIND_TABLE.ellipsoid.code,
  terrainHeightfield: SVO_PRIMITIVE_KIND_TABLE["terrain-heightfield"].code,
  torus: SVO_PRIMITIVE_KIND_TABLE.torus.code,
  cone: SVO_PRIMITIVE_KIND_TABLE.cone.code,
  smoothUnionCluster: SVO_PRIMITIVE_KIND_TABLE["smooth-union-cluster"].code,
  roundCone: SVO_PRIMITIVE_KIND_TABLE["round-cone"].code,
  roundedCylinder: SVO_PRIMITIVE_KIND_TABLE["rounded-cylinder"].code,
  fieldProgram: SVO_PRIMITIVE_KIND_TABLE["field-program"].code,
} as const);

export const SVO_PRIMITIVE_FLAGS = SVO_PRIMITIVE_KIND_FLAGS;

/**
 * The procedural signed-distance fields kind 9 selects among.
 *
 * One primitive kind, several fields, because what varies between them is the
 * *construction* and not anything a record holds: each is a smooth minimum of
 * analytic lobes over a procedurally generated domain, hard-`max`ed against the
 * record's ellipsoid envelope. A second kind code would duplicate the envelope
 * contract, the march plumbing, the arena reference and the normal policy in
 * order to vary one function.
 *
 * They are geometric constructions and nothing more. What a given parameter
 * regime happens to look like is not decided here and is not describable here:
 * the same three fields have to serve a set that has not been authored yet, so
 * anything that would be true only of one set belongs in the generator that
 * composes them.
 *
 * Every field in this family owes the same two guarantees, and neither is a
 * quality knob:
 *
 * 1. **It stays inside the envelope.** The record's three dimension floats are
 *    an ellipsoid, the field is intersected with it by a hard `max`, and the
 *    solid is therefore provably contained. That is what lets every bounds
 *    formula, the voxelizer and the node-mip oracle keep treating this kind as
 *    an ordinary ellipsoid without knowing any field exists.
 * 2. **It is a Lipschitz-1 lower bound.** The march steps by `|d|`, so a field
 *    that ever oversteps walks through its own surface. The geometric fields
 *    are smooth minima of analytic unit-Lipschitz lobes. The density field is
 *    different: it divides its threshold function by the complete gradient
 *    bound of both noise scales and its radial bias. Raw noise displacement is
 *    still forbidden; measured unbounded it reached L = 60 and tunneled on
 *    58.9 % of rays in `docs/HERO_GARDEN_AGGREGATE_SDF_ASSESSMENT.md` §3.
 *
 * Codes are stable: they are packed into the arena blocks of published scenes.
 * `lattice` is zero so that a scene authored before this family existed — and a
 * block that was never filled — reads as the field it always was.
 */
export const SVO_CLUSTER_FIELD_NAMES = Object.freeze(["lattice", "seeded-lobes", "tapered-sweep", "noise-foliage"] as const);

export type SvoClusterFieldName = typeof SVO_CLUSTER_FIELD_NAMES[number];

export interface SvoClusterFieldEntry {
  readonly name: SvoClusterFieldName;
  /** Stable on-GPU field tag, packed into word 0 of the arena block. */
  readonly code: number;
  /** Name of the matching WGSL constant, generated into the shared shader library. */
  readonly wgslConstant: string;
  /**
   * Words this field's block actually occupies, header included.
   *
   * Per field rather than one global, because a swept tube's polyline is an
   * order of magnitude larger than a lattice's five numbers and sizing every
   * block for the largest would be honest only by accident. See
   * {@link SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS} for why the arena still steps
   * by a uniform slot.
   */
  readonly arenaWords: number;
}

export const SVO_CLUSTER_FIELD_TABLE = Object.freeze({
  lattice: { name: "lattice", code: 0, wgslConstant: "SVO_CLUSTER_FIELD_LATTICE", arenaWords: 16 },
  "seeded-lobes": { name: "seeded-lobes", code: 1, wgslConstant: "SVO_CLUSTER_FIELD_SEEDED_LOBES", arenaWords: 16 },
  "tapered-sweep": { name: "tapered-sweep", code: 2, wgslConstant: "SVO_CLUSTER_FIELD_TAPERED_SWEEP", arenaWords: 48 },
  "noise-foliage": { name: "noise-foliage", code: 3, wgslConstant: "SVO_CLUSTER_FIELD_NOISE_FOLIAGE", arenaWords: 16 },
} as const satisfies Record<SvoClusterFieldName, SvoClusterFieldEntry>);

const clusterFieldByCode = new Map<number, SvoClusterFieldEntry>(
  Object.values(SVO_CLUSTER_FIELD_TABLE).map((entry) => [entry.code, entry]),
);

/** The field a packed block's word 0 names, or undefined for a code no field owns. */
export function svoClusterFieldByCode(code: number): SvoClusterFieldEntry | undefined {
  return clusterFieldByCode.get(code);
}

/**
 * The WGSL field constants, generated rather than transcribed — the same
 * arrangement, and for the same reason, as the kind constants next door.
 */
const svoClusterFieldConstantsWGSL: string = Object.values(SVO_CLUSTER_FIELD_TABLE)
  .map((entry) => `const ${entry.wgslConstant}: u32 = ${entry.code}u;`)
  .join("\n");

/**
 * Slot stride of a cluster's parameter block in the shared scene arena.
 *
 * The largest field's block, so every slot is the same size. That costs a
 * lattice thirty-two unused words and buys two things a variable stride would
 * lose. A reference stays *checkable*: the resolver rejects an offset that is
 * not a multiple of the stride, which is what turns a stale or corrupted
 * word-13 into a reported failure instead of a plausible packing read from the
 * middle of its neighbour. And a reference stays a function of the cluster's
 * index alone, so the publisher does not have to carry a prefix sum of every
 * preceding field's size through a build that assigns references while it is
 * still discovering which fields the scene uses.
 */
export const SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS = Math.max(
  ...Object.values(SVO_CLUSTER_FIELD_TABLE).map((entry) => entry.arenaWords),
);

/** Control points a tapered sweep may carry. Fixed: the block is a fixed-size slot. */
export const SVO_CLUSTER_SWEEP_MAXIMUM_POINTS = 8;

/**
 * Octaves the lattice field may stack.
 *
 * A cost bound rather than a soundness one, and the number is set from
 * measurement. It was **three**, argued as "a fourth would put a covering
 * fragment past 250 distance evaluations for detail an eighth of the first
 * octave's, which is below a pixel at any scale this renderer draws." Both
 * halves of that argument were written at a 25 mm leaf and neither survives the
 * environment-refinement ladder: at `SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM`
 * the hero garden's leaf is 0.78125 mm, a fourth octave of the stone family's
 * 55 mm period is **6.9 mm** and a fifth is 3.4 mm, and at the hero camera's
 * measured 1.99 plate-pixels per screen-millimetre those are 13 px and 7 px.
 * Nothing about them is sub-pixel.
 *
 * ## What an octave costs, measured
 *
 * On the hero garden's own two lattice regimes — a 82x35x74 mm boulder cap on a
 * 55 mm period, and a 240x100x220 mm foliage pad on 90 mm — 4 096 covering-
 * fragment rays each, marched by the same loop and epsilon as
 * `intersectMarchedLocal`:
 *
 *   octaves   finest mm   eval cost   mean march steps   hits/misses/exhausted
 *      1         55.0       x1.00          13.26            2153/1943/114
 *      2         27.5       x1.92          13.04            2154/1942/114
 *      3         13.8       x2.88          13.04            2154/1942/114
 *      4          6.9       x3.96          13.04            2154/1942/114
 *      5          3.4       x4.92          13.04            2154/1942/114
 *      6          1.7       x5.92          13.04            2154/1942/114
 *
 * Three findings, and the middle one is the surprise:
 *
 *  - **Evaluation cost is exactly linear in octave count**, as the construction
 *    says it must be: an octave is one independent neighbourhood minimisation.
 *  - **March step count does not move at all**, and neither do the hit, miss and
 *    exhaustion counts — bit-identical from octave two upward. This is the
 *    Lipschitz argument on the tin: each octave is separately a 1-Lipschitz
 *    lower bound and a smooth minimum of them is a convex combination of their
 *    gradients, so stacking octaves never costs a step and never tunnels.
 *  - **It costs no geometry.** Octaves change no record's envelope, no record
 *    count and no proxy box, so the per-brick owner census is unchanged to the
 *    integer: measured over the whole ladder, the hero garden's busiest 200 mm
 *    brick is 103/103/103/103/116 at 25/6.25/3.125/1.5625/0.78125 mm with the
 *    ceiling at three and *the same five numbers* with it at six.
 *
 * So the honest ceiling is the point past which no generator would ask for one.
 * Every species gates its own octaves on the same three-leaf admission test
 * (`stoneFormOctaves`, `foliageGrainOctaves`): an octave is only taken when its
 * period clears three leaves, which is the aliasing floor. At the finest leaf
 * that can exist — 0.78125 mm — that test asks the stone family for **five**
 * and foliage's coarser 90 mm period for six.
 *
 * **Five**, therefore. It is exactly what the deepest legal refinement admits
 * for the stone family, which is the whole of the hero garden's lattice
 * population (117 of 117 lattice records). Six was rejected on cost: it serves
 * only foliage, whose sixth octave is 2.81 mm — 3.6 leaves, sitting on the
 * admission floor — and it would buy that by putting a further 20 % on the most
 * expensive record kind in the scene.
 *
 * **This is a no-op at the shipped leaf, by construction.** `HERO_GARDEN_CELL_M`
 * is 6.25 mm and the admission test asks for two octaves there, so no authored
 * set moves until a scene spends refinement levels. Verified: `check:scenery`
 * reports identical prop counts on all ten subjects either side of the change.
 *
 * **What is not measured is the frame.** The cost above is per distance
 * evaluation, on the CPU mirror; what fraction of a depth-3 frame a lattice
 * record's march actually is was not obtainable — the Dawn lane's depth-3 run
 * dies on a V8 heap exhaustion before it presents, and frame times in that lane
 * are not currently reproducible in any case. The bound above is what is known:
 * linear in octaves, on the records that carry a lattice, with no step-count and
 * no geometry term.
 */
export const SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES = 5;

/**
 * Lobes the seeded-lobes field may fuse.
 *
 * The floor is where the construction degenerates rather than a matter of
 * taste: with fewer than four, the smooth union is dominated by the centred
 * lobe and the result is an ellipsoid — the shape the field exists to be an
 * alternative to. The ceiling is a cost bound, twelve rotations and twelve
 * scaled lengths per distance evaluation and four of those per normal.
 */
export const SVO_CLUSTER_LOBE_MINIMUM_COUNT = 4;
export const SVO_CLUSTER_LOBE_MAXIMUM_COUNT = 12;

/**
 * Largest ratio between a lobe's longest and shortest half-axis.
 *
 * A soundness ceiling wearing a cost hat. The lobe's distance is scaled by a
 * lower bound on its smallest singular value, so a lobe of ratio `a` understeps
 * by up to `a` near its long axis — and with
 * {@link SVO_PRIMITIVE_MARCH_ITERATIONS} fixed at 48, a grazing ray on a flat
 * enough lobe runs out of iterations before it arrives and reports a miss.
 */
export const SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY = 4;

/**
 * Defaults for the seeded-lobes placement parameters.
 *
 * A *representative* regime and not a definition. The placement rule itself is
 * three numbers — span, spread and displacement — and they exist as parameters
 * rather than as constants because the range they cover is the difference
 * between a compact solid with a slightly irregular skin and a loose cluster of
 * distinct lumps, and this file has no way to know which of those a caller
 * wants. See {@link SvoClusterSeededLobesField} for what each one does and for
 * the containment argument they are constrained by.
 */
export const SVO_CLUSTER_LOBE_DEFAULT_SPAN = 0.48;
export const SVO_CLUSTER_LOBE_DEFAULT_SPAN_SPREAD = 0.24;
export const SVO_CLUSTER_LOBE_DEFAULT_DISPLACEMENT = 0.8;

/**
 * Cells per axis in the neighbourhood a cluster's distance is minimised over.
 *
 * Four, and this number is the whole safety argument for the kind — it is not a
 * quality knob.
 *
 * The obvious choice is two: `floor(p/period - 1/2)` names the eight cell
 * centres that surround the point, so the *nearest* sphere is almost always one
 * of them. That is not enough, and the reason is subtle. The block's membership
 * changes as the point crosses a cell boundary, and at the instant a cell
 * leaves, its sphere is only `(1 - jitter)·period - radius` from the point —
 * 1.2 mm on a 16 mm period. A lobe that close is still inside the smooth
 * minimum's blend, so the field *jumps* when it leaves. A sphere trace stepping
 * by `|d|` relies on the field being Lipschitz-1; measured over 200 000 sample
 * pairs, the eight-cell field's constant is **60**, and 5 of 2 000 rays step
 * clean through the surface as a result. Enlarging to a 27-cell block does not
 * help: it is centred on the point's own cell rather than on the point, so its
 * membership changes at the same boundaries.
 *
 * A four-cell block is the smallest one centred on the point that *contains*
 * the eight surrounding cells with a ring to spare. A leaving sphere is then at
 * least `(2 - jitter)·period - radius` away — 17.2 mm at the same scale, well
 * outside the blend — so the field is continuous. Measured constant: 1.000.
 *
 * Sixty-four cells is not sixty-four sphere evaluations: the fifty-six outer
 * cells are rejected by a distance to their *unjittered* centre, which needs no
 * hash, and only those that can reach the running minimum are evaluated.
 */
export const SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD = 4;

/**
 * The largest jitter a domain-repeated lattice may carry, as a fraction of its
 * own period.
 *
 * With the neighbourhood above, three tenths is a *look* bound rather than a
 * safety one — it is where the packing stops reading as a lattice, and past it
 * a sphere wanders far enough to leave a visible hole in its own row. The
 * safety bound is the period condition enforced beside it, which this value
 * also enters.
 */
export const SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER = 0.3;

/** Stable feature IDs returned beside shading normals. */
export const SVO_PRIMITIVE_FEATURES = Object.freeze({
  smooth: 0,
  boxFaceX: 1,
  boxFaceY: 2,
  boxFaceZ: 3,
  cylinderSide: 4,
  cylinderCap: 5,
  terrain: 6,
} as const);

interface SvoPrimitiveIdentity {
  /** Stable scene-local primitive identity. */
  primitiveId: number;
  /** Stable material-table index. Zero is reserved for empty space. */
  materialId: number;
  /** Stable scene owner, or 0xffff when the primitive has no selectable owner. */
  ownerId?: number;
}

interface SvoLocatedPrimitive extends SvoPrimitiveIdentity {
  center_m: Vec3;
}

interface SvoOrientedPrimitive extends SvoLocatedPrimitive {
  /** Repository quaternion order is wxyz; packed GPU order is xyzw. */
  orientation?: Quaternion;
}

export interface SvoSpherePrimitive extends SvoLocatedPrimitive {
  kind: "sphere";
  radius_m: number;
}

export interface SvoBoxPrimitive extends SvoOrientedPrimitive {
  kind: "box";
  halfExtents_m: Vec3;
}

export interface SvoCapsulePrimitive extends SvoOrientedPrimitive {
  kind: "capsule";
  radius_m: number;
  segmentHalfLength_m: number;
}

export interface SvoCylinderPrimitive extends SvoOrientedPrimitive {
  kind: "cylinder";
  radius_m: number;
  halfHeight_m: number;
}

/** Flat-capped cylinder whose cap/side join is a circular fillet. */
export interface SvoRoundedCylinderPrimitive extends SvoOrientedPrimitive {
  kind: "rounded-cylinder";
  /** Outer radial extent, including the fillet. */
  radius_m: number;
  /** Outer vertical half extent, including the fillet. */
  halfHeight_m: number;
  edgeRadius_m: number;
}

export interface SvoEllipsoidPrimitive extends SvoOrientedPrimitive {
  kind: "ellipsoid";
  radii_m: Vec3;
}

/**
 * A ring swept about the local +Y axis, so its tube lies in the local XZ plane.
 * One record replaces the bead chains scenery used to approximate a coil, and
 * because it is a single owner the surface no longer steps at voxel boundaries
 * where two of them used to meet.
 */
export interface SvoTorusPrimitive extends SvoOrientedPrimitive {
  kind: "torus";
  /** Centreline radius of the ring. */
  majorRadius_m: number;
  /** Tube radius, strictly inside the centreline so the ring never self-intersects. */
  minorRadius_m: number;
}

/**
 * A truncated cone about the local +Y axis: `baseRadius_m` at -Y, `topRadius_m`
 * at +Y. A zero top radius is an ordinary cone. Anything with a taper — a
 * flowerpot, a spout, a stem — is this, not a cylinder.
 */
export interface SvoConePrimitive extends SvoOrientedPrimitive {
  kind: "cone";
  baseRadius_m: number;
  topRadius_m: number;
  halfHeight_m: number;
}

/** Convex hull of spheres at local -Y/+Y: a tapered segment with round ends. */
export interface SvoRoundConePrimitive extends SvoOrientedPrimitive {
  kind: "round-cone";
  baseRadius_m: number;
  topRadius_m: number;
  halfHeight_m: number;
}

/**
 * A terrain record references the shared scene heightfield table. Variable-size
 * terrain features deliberately do not live in every primitive record.
 */
export interface SvoTerrainHeightfieldPrimitive extends SvoPrimitiveIdentity {
  kind: "terrain-heightfield";
  terrainReference: number;
  normalEpsilon_m?: number;
}

/**
 * What every field of the family carries, whatever it is made of.
 *
 * Not in the record, because it does not fit: sixty-four bytes leave three
 * floats of per-kind space and the envelope's half-axes have already spent them.
 */
interface SvoClusterFieldBase {
  /**
   * Radius of the polynomial smooth minimum that fuses this field's lobes.
   *
   * This is what removes the seams a chain of explicit primitives leaves at
   * every junction, and it is also why the trace is safe: a smooth minimum is a
   * strict *lower* bound on the union distance, so the march understeps.
   */
  readonly smoothRadius_m: number;
  /** Stable u32 that decorrelates one cluster's procedural arrangement from its neighbours'. */
  readonly seed: number;
}

/**
 * Field 0: a domain-repeated jittered lobe lattice, `n` octaves deep.
 *
 * One sphere per cell of an infinite cubic lattice, displaced from its cell
 * centre by a hash of the cell index, smooth-unioned over the point's own
 * neighbourhood — then that whole construction repeated at successive halvings
 * and fused onto itself.
 *
 * `field` is optional here and required on every other member, and that is
 * deliberate rather than lax. A packing authored before this family existed is
 * a lattice, and leaving the discriminant off means the call sites that already
 * author one stay untouched and keep publishing exactly the surface they did.
 */
export interface SvoClusterLatticeField extends SvoClusterFieldBase {
  readonly field?: "lattice";
  /** Radius of one lobe of the coarsest octave's lattice. */
  readonly latticeLobeRadius_m: number;
  /** Domain-repetition period of the coarsest octave, the same on all three axes. */
  readonly latticePeriod_m: number;
  /**
   * How far a lobe is displaced from its own cell centre, as a fraction of the
   * period, in [0, {@link SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER}].
   *
   * Zero is a perfect crystal and looks like one. The cap is where the
   * four-cell neighbourhood stops being a lower bound; see the constant.
   *
   * The lobe radius against the period is the other half of what the packing
   * looks like: below half the period the lattice separates into distinct
   * beads, and above `period·√3/2` every cell corner is covered and the result
   * is a solid with a rumpled skin.
   */
  readonly jitter: number;
  /**
   * How many halvings of the lattice are fused onto it, in
   * `1..{@link SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES}`. Absent is one.
   *
   * Octave `k` repeats on `period·2^-k` with lobes of `radius·2^-k`, and is
   * smooth-unioned onto the running result with `smoothRadius·2^-k` — every
   * length scaled together, so an octave is the same lattice seen closer and
   * inherits the same neighbourhood bound that makes the first one sound. Each
   * octave costs one whole lattice evaluation.
   */
  readonly octaves?: number;
}

/**
 * Field 1: a set of oriented anisotropic lobes, placed from a seed and fused.
 *
 * A general irregular solid: `lobeCount` ellipsoids at hashed orientations and
 * axis ratios, positioned inside the envelope by a rule that guarantees the
 * hard `max` never reaches any of them, smooth-unioned in index order. The
 * silhouette it produces has as many distinct curvatures as it has lobes, which
 * is the property that distinguishes it from the ellipsoid it is contained in.
 *
 * The lobes are not stored. They are derived from `seed` and from the record's
 * envelope by a fixed hash, which is what keeps the block sixteen words and
 * what lets the shader recompute the same arrangement on every march step. It
 * is also loop-invariant over a march, so a compiler that hoists it pays for
 * the placement once per covering fragment rather than once per step.
 *
 * All three placement parameters are fractions of the envelope in its own
 * normalised space — the space in which the envelope is the unit ball — so the
 * arrangement is invariant to how eccentric the envelope is and a caller sets
 * proportions rather than lengths.
 */
export interface SvoClusterSeededLobesField extends SvoClusterFieldBase {
  readonly field: "seeded-lobes";
  /** Lobes fused, in `{@link SVO_CLUSTER_LOBE_MINIMUM_COUNT}..{@link SVO_CLUSTER_LOBE_MAXIMUM_COUNT}`. */
  readonly lobeCount: number;
  /**
   * Largest ratio between a lobe's longest and shortest half-axis, in
   * `1..{@link SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY}`.
   *
   * One makes every lobe a sphere. Raising it both varies the silhouette and
   * shrinks the solid, because a lobe's *longest* half-axis is what `lobeSpan`
   * fixes and the other two shorten away from it.
   */
  readonly anisotropy: number;
  /**
   * A lobe's longest half-axis, as a fraction of the envelope, in `(0, 1)`.
   * Defaults to {@link SVO_CLUSTER_LOBE_DEFAULT_SPAN}.
   *
   * The primary size knob, and it trades solidity against irregularity: what a
   * lobe spans it cannot also travel, because the two are constrained to sum to
   * at most one. Near 1 the field is a single centred ellipsoid; small values
   * give distinct lumps a long way apart, which the blend may or may not close.
   */
  readonly lobeSpan?: number;
  /**
   * How much `lobeSpan` varies across the lobes, in `[0, 1 - lobeSpan)`.
   * Defaults to {@link SVO_CLUSTER_LOBE_DEFAULT_SPAN_SPREAD}.
   *
   * Zero makes every lobe the same size, which reads as regular however the
   * orientations scatter.
   */
  readonly lobeSpanSpread?: number;
  /**
   * The share of its available travel a lobe actually spends, in `[0, 1]`.
   * Defaults to {@link SVO_CLUSTER_LOBE_DEFAULT_DISPLACEMENT}.
   *
   * Zero puts every lobe concentric and the field becomes the union of nested
   * ellipsoids. One pushes each lobe until its bounding ball is tangent to the
   * envelope's clearance shell, which is the only way the authored envelope is
   * also the drawn silhouette — but because travel and span are constrained to
   * sum to a constant, at exactly one every lobe reaches the *same* shell and
   * the outline regularises again. Just under one is where both hold.
   */
  readonly displacement?: number;
}

/** One control point of a tapered sweep, in the record's own local frame. */
export interface SvoClusterSweepPoint {
  /** Offset from the record's centre, before its orientation is applied. */
  readonly position_m: Vec3;
  /** Sweep radius at this point. The sweep tapers linearly between neighbours. */
  readonly radius_m: number;
}

/**
 * Field 2: a tapered sweep along a control polyline.
 *
 * Consecutive control points define round cones — the convex hull of the two
 * spheres at their ends — smooth-unioned in authored order. One record for a
 * run that would otherwise be a chain of capsules: a record each, a stepped
 * shading normal at every junction, and a seam wherever two of them met across
 * a voxel boundary. The segment distance is exact, so the field is Lipschitz-1
 * rather than merely bounded by one.
 */
export interface SvoClusterTaperedSweepField extends SvoClusterFieldBase {
  readonly field: "tapered-sweep";
  /** Two to {@link SVO_CLUSTER_SWEEP_MAXIMUM_POINTS} points, in sweep order. */
  readonly points: readonly SvoClusterSweepPoint[];
}

/**
 * Field 3: thresholded two-scale value-noise density.
 *
 * The cluster noise modulates the probability that the detail noise exceeds
 * the iso-value, so occupied voxels gather into broad masses rather than
 * becoming uniform static. `interiorBias` raises density toward the centre of
 * the envelope without closing the voids at its edge.
 */
export interface SvoClusterNoiseFoliageField extends SvoClusterFieldBase {
  readonly field: "noise-foliage";
  readonly clusterPeriod_m: number;
  readonly detailPeriod_m: number;
  readonly threshold: number;
  readonly clusterWeight: number;
  readonly detailWeight: number;
  readonly interiorBias: number;
}

/**
 * The packing a cluster's arena block holds: one of the family's fields.
 *
 * Discriminated on `field`, whose absence is the lattice — see
 * {@link SvoClusterLatticeField}.
 */
export type SvoSmoothUnionClusterPacking =
  | SvoClusterLatticeField
  | SvoClusterSeededLobesField
  | SvoClusterTaperedSweepField
  | SvoClusterNoiseFoliageField;

/** The field a packing names, with the lattice standing in for an absent discriminant. */
export function svoClusterFieldName(packing: SvoSmoothUnionClusterPacking): SvoClusterFieldName {
  return packing.field ?? "lattice";
}

/**
 * An aggregate: one record standing for a procedural field clipped to an
 * ellipsoidal envelope.
 *
 * It exists because the explicit alternative does not fit. An aggregate spelled
 * as fifteen hundred records spends more than a third of the scene's 4 096-record
 * budget, and — worse, because it is silent — averages some eighty of them in
 * each brick against a 64-candidate-per-brick ceiling whose overflow is a
 * *dropped* primitive: absent from the opacity pyramid and the radiance atlas
 * while still drawing in primary visibility. One record owning a coarse region
 * outright is how that budget and that density are both returned.
 *
 * `lobeRadii_m` is the record's three dimension floats, so every bounds formula
 * in the tree — the raster proxy, the voxelizer's world AABB, the candidate BVH
 * leaf, the coverage footprint, the sway budget — reads an ordinary ellipsoid
 * without knowing this kind exists. That is sound rather than convenient: the
 * envelope is applied as a hard `max` against its own distance, so the solid is
 * contained in the ellipsoid by construction, and a voxelizer that treats the
 * whole envelope as solid over-occludes a fissured interior. Over-occluding is
 * the safe direction for shadows and ambient occlusion; under-occluding is
 * indistinguishable from empty space to every consumer of the pyramid.
 */
export interface SvoSmoothUnionClusterPrimitive extends SvoOrientedPrimitive {
  kind: "smooth-union-cluster";
  /**
   * Half-axes of the envelope every field of the family is clipped to.
   *
   * Named for the lattice's lobe because that is what it was when the kind held
   * one field. It is the envelope for all of them, and the containment argument
   * above is what every other consumer of this record is relying on.
   */
  lobeRadii_m: Vec3;
  /** u32 word offset of this cluster's parameter block in the shared scene arena. */
  clusterReference: number;
  /**
   * The block's contents.
   *
   * Present on an authored descriptor and absent on one recovered from a packed
   * record, exactly as a terrain record recovers a reference and not a
   * heightfield. Evaluating a cluster without either is an error rather than a
   * plausible-looking default: a silently degenerate packing renders as a
   * smooth ellipsoid, which is a shape the scene never asked for.
   */
  packing?: SvoSmoothUnionClusterPacking;
}

/**
 * A record whose geometry is a field program: a tape in the scene arena,
 * evaluated as one SDF.
 *
 * The same arrangement as {@link SvoSmoothUnionClusterPrimitive}, and for the
 * same reason — sixty-four bytes leave three floats of per-kind space, and a
 * tape is up to sixteen ops of eight words. What differs is what the three
 * floats mean. A cluster's are an *ellipsoid envelope* that the field is hard
 * `max`ed against, so containment is enforced. A field program is not clipped by
 * anything: its containment is *derived*, by `svoFieldProgramExtent_m`, from the
 * source solid's own radii grown by the total warp amplitude on the chain
 * feeding it. That bound is per axis, so the three floats are a box half-extent.
 *
 * Which makes the envelope below an *authored floor* rather than the shape.
 * `dimensions` takes the componentwise maximum of it and the program's derived
 * extent, so a caller may reserve room for a tape it intends to grow later
 * without the bound ever falling below what the current tape actually needs. A
 * bound that under-covered would delete exactly the silhouette detail the warp
 * exists to create — silently, and worst at the outline the eye reads first.
 */
export interface SvoFieldProgramPrimitive extends SvoOrientedPrimitive {
  kind: "field-program";
  /**
   * Authored conservative half-extent about the centre, along the local axes.
   *
   * A floor on the packed dimensions, never a ceiling: see above. This is also
   * what a descriptor recovered from a packed record carries, because the record
   * holds the widened result and nothing else.
   */
  envelopeRadii_m: Vec3;
  /** u32 word offset of this record's tape block in the shared scene arena. */
  fieldProgramReference: number;
  /**
   * The tape.
   *
   * Present on an authored descriptor and absent on one recovered from a packed
   * record, exactly as a cluster recovers a reference and not a packing.
   * Evaluating a field program without either is an error rather than a
   * plausible-looking default: a tape that silently lost its warp renders a
   * smooth ellipsoid, which is precisely the frame this kind exists to stop
   * producing, and it would do it without a single error.
   */
  program?: SvoFieldProgram;
}

export type SvoPrimitiveDescriptor =
  | SvoSpherePrimitive
  | SvoBoxPrimitive
  | SvoCapsulePrimitive
  | SvoCylinderPrimitive
  | SvoRoundedCylinderPrimitive
  | SvoEllipsoidPrimitive
  | SvoTorusPrimitive
  | SvoConePrimitive
  | SvoRoundConePrimitive
  | SvoSmoothUnionClusterPrimitive
  | SvoFieldProgramPrimitive
  | SvoTerrainHeightfieldPrimitive;

export interface SvoPrimitiveSample {
  signedDistance_m: number;
  normal: Vec3 | null;
  featureId: number;
}

export type SvoFinitePrimitiveDescriptor = Exclude<SvoPrimitiveDescriptor, SvoTerrainHeightfieldPrimitive>;

/** World-space ray whose interval is measured in metres along its normalized direction. */
export interface SvoPrimitiveRay {
  origin_m: Vec3;
  direction: Vec3;
  tMin_m?: number;
  tMax_m?: number;
}

export interface SvoPrimitiveRayHit {
  /** Physical distance from origin_m, independent of direction magnitude. */
  t_m: number;
  position_m: Vec3;
  normal: Vec3;
  /** Smooth primitives interpolate their analytic gradient; hard-feature primitives select one authored feature. */
  normalPolicy: "smooth" | "hard-feature";
  featureId: number;
  primitiveKind: SvoFinitePrimitiveDescriptor["kind"];
  primitiveId: number;
  materialId: number;
  ownerId: number;
}

export type SvoTerrainResolver = (terrainReference: number) => TerrainDescription | undefined;

/**
 * Reads a cluster's packing back out of the scene arena.
 *
 * The mirror of {@link SvoTerrainResolver}, and for the same reason: a record
 * recovered from packed bytes names an arena block it cannot itself contain, so
 * whoever holds the arena supplies the contents. A descriptor that still has
 * its authored `packing` needs no resolver.
 */
export type SvoClusterResolver = (clusterReference: number) => SvoSmoothUnionClusterPacking | undefined;

/**
 * Reads a field program's tape back out of the scene arena.
 *
 * The mirror of {@link SvoClusterResolver}, one kind over. A descriptor that
 * still has its authored `program` needs no resolver.
 */
export type SvoFieldProgramResolver = (fieldProgramReference: number) => SvoFieldProgram | undefined;

function resolveFieldProgram(
  descriptor: SvoFieldProgramPrimitive,
  fieldProgramResolver?: SvoFieldProgramResolver,
): ResolvedFieldProgram {
  const program = descriptor.program ?? fieldProgramResolver?.(descriptor.fieldProgramReference);
  if (!program) {
    throw new Error("Field-program evaluation requires its authored tape or an arena resolver");
  }
  return { ...descriptor, program };
}

function resolveCluster(
  descriptor: SvoSmoothUnionClusterPrimitive,
  clusterResolver?: SvoClusterResolver,
): ResolvedCluster {
  const packing = descriptor.packing ?? clusterResolver?.(descriptor.clusterReference);
  if (!packing) {
    throw new Error("Smooth-union cluster evaluation requires its authored packing or an arena resolver");
  }
  validateSvoClusterPacking(packing, descriptor.lobeRadii_m);
  return { ...descriptor, packing };
}

const NORMAL_EPSILON = 1e-12;

function finiteVec3(value: Vec3, label: string): void {
  if (![value.x, value.y, value.z].every(Number.isFinite)) throw new RangeError(`${label} must be finite`);
}

function positive(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be positive`);
}

function nonNegative(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${label} must be non-negative`);
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function normalizedOrientation(value: Quaternion | undefined): Quaternion {
  if (value === undefined) return { w: 1, x: 0, y: 0, z: 0 };
  if (![value.w, value.x, value.y, value.z].every(Number.isFinite)) throw new RangeError("Primitive orientation must be finite");
  const magnitude = Math.hypot(value.w, value.x, value.y, value.z);
  if (!(magnitude > NORMAL_EPSILON)) throw new RangeError("Primitive orientation must have nonzero length");
  return { w: value.w / magnitude, x: value.x / magnitude, y: value.y / magnitude, z: value.z / magnitude };
}

function validateIdentity(descriptor: SvoPrimitiveDescriptor): void {
  uint32(descriptor.primitiveId, "Primitive ID");
  if (!Number.isInteger(descriptor.materialId) || descriptor.materialId < 1 || descriptor.materialId > 0xffff) {
    throw new RangeError("Primitive material ID must be a nonzero uint16");
  }
  const owner = descriptor.ownerId ?? SPARSE_BRICK_NO_OWNER;
  if (!Number.isInteger(owner) || owner < 0 || owner > 0xffff) throw new RangeError("Primitive owner ID must fit uint16");
}

function dimensions(descriptor: SvoPrimitiveDescriptor): Vec3 {
  if (descriptor.kind === "sphere") {
    positive(descriptor.radius_m, "Sphere radius");
    return { x: descriptor.radius_m, y: 0, z: 0 };
  }
  if (descriptor.kind === "box") {
    finiteVec3(descriptor.halfExtents_m, "Box half extents");
    positive(descriptor.halfExtents_m.x, "Box X half extent");
    positive(descriptor.halfExtents_m.y, "Box Y half extent");
    positive(descriptor.halfExtents_m.z, "Box Z half extent");
    return { ...descriptor.halfExtents_m };
  }
  if (descriptor.kind === "capsule") {
    positive(descriptor.radius_m, "Capsule radius");
    nonNegative(descriptor.segmentHalfLength_m, "Capsule segment half length");
    return { x: descriptor.radius_m, y: descriptor.segmentHalfLength_m, z: 0 };
  }
  if (descriptor.kind === "cylinder") {
    positive(descriptor.radius_m, "Cylinder radius");
    positive(descriptor.halfHeight_m, "Cylinder half height");
    return { x: descriptor.radius_m, y: descriptor.halfHeight_m, z: 0 };
  }
  if (descriptor.kind === "rounded-cylinder") {
    positive(descriptor.radius_m, "Rounded-cylinder radius");
    positive(descriptor.halfHeight_m, "Rounded-cylinder half height");
    positive(descriptor.edgeRadius_m, "Rounded-cylinder edge radius");
    if (descriptor.edgeRadius_m > Math.min(descriptor.radius_m, descriptor.halfHeight_m)) {
      throw new RangeError("Rounded-cylinder edge radius must fit inside its outer radius and half height");
    }
    return { x: descriptor.radius_m, y: descriptor.halfHeight_m, z: descriptor.edgeRadius_m };
  }
  if (descriptor.kind === "ellipsoid") {
    finiteVec3(descriptor.radii_m, "Ellipsoid radii");
    positive(descriptor.radii_m.x, "Ellipsoid X radius");
    positive(descriptor.radii_m.y, "Ellipsoid Y radius");
    positive(descriptor.radii_m.z, "Ellipsoid Z radius");
    return { ...descriptor.radii_m };
  }
  if (descriptor.kind === "torus") {
    positive(descriptor.majorRadius_m, "Torus major radius");
    positive(descriptor.minorRadius_m, "Torus minor radius");
    // A tube thicker than the ring it is swept around passes through the axis,
    // and the swept-circle distance stops being the distance to that solid.
    if (!(descriptor.minorRadius_m < descriptor.majorRadius_m)) {
      throw new RangeError("Torus minor radius must be smaller than its major radius");
    }
    return { x: descriptor.majorRadius_m, y: descriptor.minorRadius_m, z: 0 };
  }
  if (descriptor.kind === "cone" || descriptor.kind === "round-cone") {
    positive(descriptor.halfHeight_m, "Cone half height");
    nonNegative(descriptor.baseRadius_m, "Cone base radius");
    nonNegative(descriptor.topRadius_m, "Cone top radius");
    // Both ends may not close: that degenerates to a segment with no surface.
    if (!(Math.max(descriptor.baseRadius_m, descriptor.topRadius_m) > 0)) {
      throw new RangeError("Cone must have a positive radius at one end");
    }
    if (descriptor.kind === "round-cone"
      && !(2 * descriptor.halfHeight_m > Math.abs(descriptor.topRadius_m - descriptor.baseRadius_m) * SINGLE_PRECISION_SLACK)) {
      throw new RangeError("Round-cone taper must not outrun its segment length");
    }
    return { x: descriptor.baseRadius_m, y: descriptor.halfHeight_m, z: descriptor.topRadius_m };
  }
  if (descriptor.kind === "smooth-union-cluster") {
    finiteVec3(descriptor.lobeRadii_m, "Cluster lobe radii");
    positive(descriptor.lobeRadii_m.x, "Cluster lobe X radius");
    positive(descriptor.lobeRadii_m.y, "Cluster lobe Y radius");
    positive(descriptor.lobeRadii_m.z, "Cluster lobe Z radius");
    uint32(descriptor.clusterReference, "Cluster reference");
    if (descriptor.clusterReference === SVO_PRIMITIVE_INVALID_REFERENCE) {
      throw new RangeError("Cluster reference may not use the invalid sentinel");
    }
    if (descriptor.packing) validateSvoClusterPacking(descriptor.packing, descriptor.lobeRadii_m);
    return { ...descriptor.lobeRadii_m };
  }
  if (descriptor.kind === "field-program") {
    finiteVec3(descriptor.envelopeRadii_m, "Field-program envelope radii");
    positive(descriptor.envelopeRadii_m.x, "Field-program envelope X half extent");
    positive(descriptor.envelopeRadii_m.y, "Field-program envelope Y half extent");
    positive(descriptor.envelopeRadii_m.z, "Field-program envelope Z half extent");
    uint32(descriptor.fieldProgramReference, "Field-program reference");
    if (descriptor.fieldProgramReference === SVO_PRIMITIVE_INVALID_REFERENCE) {
      throw new RangeError("Field-program reference may not use the invalid sentinel");
    }
    // No tape means this descriptor came back out of a packed record, and the
    // record already carries the widened bound: re-deriving it here would need
    // the tape the record deliberately does not hold. Widening on the way in and
    // trusting the record on the way out is what makes the round trip a fixed
    // point rather than a bound that grows every time a scene is republished.
    if (!descriptor.program) return { ...descriptor.envelopeRadii_m };
    const derived = svoFieldProgramExtent_m(descriptor.program);
    return {
      x: Math.max(descriptor.envelopeRadii_m.x, derived[0]),
      y: Math.max(descriptor.envelopeRadii_m.y, derived[1]),
      z: Math.max(descriptor.envelopeRadii_m.z, derived[2]),
    };
  }
  uint32(descriptor.terrainReference, "Terrain reference");
  if (descriptor.terrainReference === SVO_PRIMITIVE_INVALID_REFERENCE) throw new RangeError("Terrain reference may not use the invalid sentinel");
  const epsilon = descriptor.normalEpsilon_m ?? 0.02;
  positive(epsilon, "Terrain normal epsilon");
  return { x: epsilon, y: 0, z: 0 };
}

/**
 * Both bounds below are checked with a single-precision slack, and that is not
 * cosmetic: a packing lives in the scene arena as f32s, so a value authored *at*
 * a limit comes back from the round trip a few ulps above it — `fround(0.3)` is
 * 0.30000001 — and a scene that packed successfully would then fail to unpack.
 * The slack is far below anything that could make an argument here untrue.
 */
const SINGLE_PRECISION_SLACK = 1 + 1e-6;

/**
 * The envelope's own Lipschitz-1 lower bound, read from the inside.
 *
 * How far a point is from the envelope surface, *underestimated*. Used to place
 * and to check every field's lobes, and it is exactly the function the hard
 * `max` clips with — so a lobe that clears this margin is one the `max` provably
 * never touches, and the silhouette it draws is its own rather than an ellipsoid
 * cap. Any other containment test would be either unsound or gratuitously
 * conservative: the ellipsoid with each half-axis reduced by `r` does *not*
 * contain every ball of radius `r` that fits, which is easy to get wrong on an
 * eccentric envelope.
 */
function clusterEnvelopeClearance_m(point: Vec3, lobeRadii_m: Vec3): number {
  return -clusterLobeDistance(point, lobeRadii_m);
}

/**
 * Reject a packing whose field would stop being a Lipschitz-1 lower bound, or
 * would be clipped by the envelope it is meant to fill.
 *
 * Every check here is load-bearing rather than tidy. A field that oversteps
 * makes the march walk through its own surface, and the symptom downstream is a
 * hole rather than an exception; a lobe that pushes past the envelope is sliced
 * flat by the `max`, and the symptom is an ellipsoid cap on a shape that spent
 * every other number avoiding one. Neither is visible from anywhere but here.
 */
export function validateSvoClusterPacking(packing: SvoSmoothUnionClusterPacking, lobeRadii_m: Vec3): void {
  nonNegative(packing.smoothRadius_m, "Cluster smooth-minimum radius");
  uint32(packing.seed, "Cluster seed");
  const shortestEnvelope_m = Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
  if (packing.field === "noise-foliage") {
    positive(packing.clusterPeriod_m, "Noise-foliage cluster period");
    positive(packing.detailPeriod_m, "Noise-foliage detail period");
    const unit = (value: number, label: string): void => {
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        throw new RangeError(`${label} must be in 0..1`);
      }
    };
    unit(packing.threshold, "Noise-foliage threshold");
    unit(packing.clusterWeight, "Noise-foliage cluster weight");
    unit(packing.detailWeight, "Noise-foliage detail weight");
    unit(packing.interiorBias, "Noise-foliage interior bias");
    if (!(packing.clusterWeight + packing.detailWeight > 0)) {
      throw new RangeError("Noise-foliage density needs a positive noise weight");
    }
    return;
  }
  if (packing.field === "seeded-lobes") {
    if (!Number.isInteger(packing.lobeCount)
      || packing.lobeCount < SVO_CLUSTER_LOBE_MINIMUM_COUNT || packing.lobeCount > SVO_CLUSTER_LOBE_MAXIMUM_COUNT) {
      throw new RangeError(
        `Seeded-lobe count must be an integer in ${SVO_CLUSTER_LOBE_MINIMUM_COUNT}..${SVO_CLUSTER_LOBE_MAXIMUM_COUNT}`,
      );
    }
    if (!Number.isFinite(packing.anisotropy)
      || packing.anisotropy < 1 || packing.anisotropy > SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY) {
      throw new RangeError(`Seeded-lobe anisotropy must be in 1..${SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY}`);
    }
    const { span, spanSpread, displacement } = seededLobePlacement(packing);
    positive(span, "Seeded-lobe span");
    nonNegative(spanSpread, "Seeded-lobe span spread");
    if (!Number.isFinite(displacement) || displacement < 0 || displacement > 1) {
      throw new RangeError("Seeded-lobe displacement must be in 0..1");
    }
    // Span, its spread and the blend all come out of the same unit budget: the
    // widest lobe reaches `span + spanSpread` and the blend inflates the union
    // by up to its own radius on top, so exceeding one leaves the hard `max`
    // slicing the lobes it was supposed to contain without touching.
    const widest = span + spanSpread + packing.smoothRadius_m / shortestEnvelope_m;
    if (widest * SINGLE_PRECISION_SLACK >= 1) {
      throw new RangeError(
        `Seeded lobes overrun their envelope: span ${span} plus spread ${spanSpread} plus a blend of `
        + `${(packing.smoothRadius_m / shortestEnvelope_m).toFixed(3)} of the shortest half-axis must stay under 1`,
      );
    }
    return;
  }
  if (packing.field === "tapered-sweep") {
    if (packing.points.length < 2 || packing.points.length > SVO_CLUSTER_SWEEP_MAXIMUM_POINTS) {
      throw new RangeError(`Tapered-sweep control polyline must have 2..${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS} points`);
    }
    packing.points.forEach((point, index) => {
      finiteVec3(point.position_m, `Tapered-sweep point ${index} position`);
      positive(point.radius_m, `Tapered-sweep point ${index} radius`);
      // The margin the hard `max` needs: the convex hull of two contained balls
      // is contained, so checking the balls checks every segment — and adding
      // the blend covers the smooth minimum, which *inflates* the union by up
      // to a quarter of its own radius.
      const clearance_m = clusterEnvelopeClearance_m(point.position_m, lobeRadii_m);
      if (clearance_m * SINGLE_PRECISION_SLACK < point.radius_m + packing.smoothRadius_m) {
        throw new RangeError(
          `Tapered-sweep point ${index} reaches past its envelope: the point is ${clearance_m.toFixed(4)} m inside it `
          + `and needs ${(point.radius_m + packing.smoothRadius_m).toFixed(4)} m for its radius plus the blend`,
        );
      }
    });
    for (let index = 0; index + 1 < packing.points.length; index += 1) {
      const from = packing.points[index];
      const to = packing.points[index + 1];
      const span_m = Math.hypot(
        to.position_m.x - from.position_m.x,
        to.position_m.y - from.position_m.y,
        to.position_m.z - from.position_m.z,
      );
      // A round cone whose taper outruns its own length is one ball inside the
      // other, and the closed-form distance below degenerates on it.
      if (!(span_m > Math.abs(to.radius_m - from.radius_m) * SINGLE_PRECISION_SLACK)) {
        throw new RangeError(
          `Tapered-sweep segment ${index} tapers faster than it runs: `
          + `${span_m.toFixed(4)} m between points against a ${Math.abs(to.radius_m - from.radius_m).toFixed(4)} m radius change`,
        );
      }
    }
    return;
  }
  positive(packing.latticeLobeRadius_m, "Cluster lattice lobe radius");
  positive(packing.latticePeriod_m, "Cluster lattice period");
  nonNegative(packing.jitter, "Cluster jitter");
  const octaves = packing.octaves ?? 1;
  if (!Number.isInteger(octaves) || octaves < 1 || octaves > SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES) {
    throw new RangeError(`Cluster lattice octaves must be an integer in 1..${SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES}`);
  }
  if (packing.jitter > SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER * SINGLE_PRECISION_SLACK) {
    throw new RangeError(`Cluster jitter must not exceed ${SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER} of the lattice period`);
  }
  // The condition that keeps the field continuous, and therefore the trace
  // sound. A lobe leaving the neighbourhood is at least this far away; if the
  // smooth minimum can still feel it there, the field steps and rays tunnel.
  // One check covers every octave: period, lobe radius and blend are all
  // scaled by the same `2^-k`, so the inequality is scale-invariant.
  const departingDistance_m = (SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD / 2 - packing.jitter) * packing.latticePeriod_m;
  if (departingDistance_m * SINGLE_PRECISION_SLACK < packing.latticeLobeRadius_m + packing.smoothRadius_m) {
    throw new RangeError(
      "Cluster lattice lobes reach past their own neighbourhood: "
      + `(${SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD / 2} - jitter) * period must be at least the lattice lobe plus smooth-minimum radius`,
    );
  }
}

function kindCode(kind: SvoPrimitiveKindName): number {
  return SVO_PRIMITIVE_KIND_TABLE[kind].code;
}

function primitiveFlags(kind: SvoPrimitiveKindName): number {
  return SVO_PRIMITIVE_KIND_TABLE[kind].flags;
}

/**
 * Conservative local half-extent of a primitive about its own centre.
 *
 * The one formula behind every bound in the tree. It used to be five, in five
 * files, in five slightly different forms — and a bound that is too small does
 * not fail loudly, it removes geometry from a raster proxy or a dirty region
 * and leaves a clipped silhouette to be noticed by eye.
 */
export function svoPrimitiveLocalExtent_m(descriptor: SvoPrimitiveDescriptor): Vec3 {
  return SVO_PRIMITIVE_KIND_TABLE[descriptor.kind].localExtent_m(dimensions(descriptor));
}

/** Radius of a sphere about the centre containing the primitive at any orientation. */
export function svoPrimitiveBoundingRadius_m(descriptor: SvoPrimitiveDescriptor): number {
  return SVO_PRIMITIVE_KIND_TABLE[descriptor.kind].boundingRadius_m(dimensions(descriptor));
}

function descriptorCenter(descriptor: SvoPrimitiveDescriptor): Vec3 {
  if (descriptor.kind === "terrain-heightfield") return { x: 0, y: 0, z: 0 };
  finiteVec3(descriptor.center_m, "Primitive centre");
  return descriptor.center_m;
}

function descriptorOrientation(descriptor: SvoPrimitiveDescriptor): Quaternion {
  if (descriptor.kind === "sphere" || descriptor.kind === "terrain-heightfield") return normalizedOrientation(undefined);
  return normalizedOrientation(descriptor.orientation);
}

/** Validate and canonicalize a descriptor before hashing, packing, or upload. */
export function canonicalSvoPrimitive(descriptor: SvoPrimitiveDescriptor): SvoPrimitiveDescriptor {
  validateIdentity(descriptor);
  const d = dimensions(descriptor);
  const ownerId = descriptor.ownerId ?? SPARSE_BRICK_NO_OWNER;
  if (descriptor.kind === "sphere") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, radius_m: d.x };
  if (descriptor.kind === "box") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), halfExtents_m: d };
  if (descriptor.kind === "capsule") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), radius_m: d.x, segmentHalfLength_m: d.y };
  if (descriptor.kind === "cylinder") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), radius_m: d.x, halfHeight_m: d.y };
  if (descriptor.kind === "rounded-cylinder") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), radius_m: d.x, halfHeight_m: d.y, edgeRadius_m: d.z };
  if (descriptor.kind === "ellipsoid") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), radii_m: d };
  if (descriptor.kind === "torus") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), majorRadius_m: d.x, minorRadius_m: d.y };
  if (descriptor.kind === "cone" || descriptor.kind === "round-cone") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), baseRadius_m: d.x, halfHeight_m: d.y, topRadius_m: d.z };
  if (descriptor.kind === "smooth-union-cluster") {
    return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), lobeRadii_m: d };
  }
  if (descriptor.kind === "field-program") {
    // The canonical envelope is the *widened* bound, so canonicalizing twice is
    // idempotent and a descriptor and the record packed from it describe the
    // same box. Anything narrower here would let a re-canonicalized descriptor
    // report a smaller extent than the record already published.
    return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), envelopeRadii_m: d };
  }
  return { ...descriptor, ownerId, normalEpsilon_m: d.x };
}

/** The word-13 arena reference a kind carries, or the invalid sentinel. */
function arenaReference(descriptor: SvoPrimitiveDescriptor): number {
  if (descriptor.kind === "terrain-heightfield") return descriptor.terrainReference >>> 0;
  if (descriptor.kind === "smooth-union-cluster") return descriptor.clusterReference >>> 0;
  if (descriptor.kind === "field-program") return descriptor.fieldProgramReference >>> 0;
  return SVO_PRIMITIVE_INVALID_REFERENCE;
}

/**
 * Pack `{center.xyz, kind}`, `{dimensions.xyz, material|owner}`, quaternion
 * `xyzw`, then `{primitive, terrain-reference, flags, reserved}`.
 */
export function packSvoPrimitiveRecords(descriptors: readonly SvoPrimitiveDescriptor[]): Uint32Array<ArrayBuffer> {
  const buffer = new ArrayBuffer(descriptors.length * SVO_PRIMITIVE_RECORD_STRIDE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  descriptors.forEach((input, index) => {
    const descriptor = canonicalSvoPrimitive(input);
    const base = index * SVO_PRIMITIVE_RECORD_WORDS;
    const center = descriptorCenter(descriptor);
    const d = dimensions(descriptor);
    const orientation = descriptorOrientation(descriptor);
    floats.set([center.x, center.y, center.z], base);
    words[base + 3] = kindCode(descriptor.kind);
    floats.set([d.x, d.y, d.z], base + 4);
    words[base + 7] = packMaterialOwner(descriptor.materialId, descriptor.ownerId);
    floats.set([orientation.x, orientation.y, orientation.z, orientation.w], base + 8);
    words.set([
      descriptor.primitiveId >>> 0,
      arenaReference(descriptor),
      primitiveFlags(descriptor.kind),
      0,
    ], base + 12);
  });
  return words;
}

function descriptorFromRecord(words: Uint32Array, floats: Float32Array, base: number): SvoPrimitiveDescriptor {
  const center_m = { x: floats[base], y: floats[base + 1], z: floats[base + 2] };
  const d = { x: floats[base + 4], y: floats[base + 5], z: floats[base + 6] };
  const orientation = { x: floats[base + 8], y: floats[base + 9], z: floats[base + 10], w: floats[base + 11] };
  const { materialId, ownerId } = unpackMaterialOwner(words[base + 7]);
  const identity = { primitiveId: words[base + 12], materialId, ownerId };
  const kind = words[base + 3];
  if (kind === SVO_PRIMITIVE_KINDS.sphere) return { ...identity, kind: "sphere", center_m, radius_m: d.x };
  if (kind === SVO_PRIMITIVE_KINDS.box) return { ...identity, kind: "box", center_m, orientation, halfExtents_m: d };
  if (kind === SVO_PRIMITIVE_KINDS.capsule) return { ...identity, kind: "capsule", center_m, orientation, radius_m: d.x, segmentHalfLength_m: d.y };
  if (kind === SVO_PRIMITIVE_KINDS.cylinder) return { ...identity, kind: "cylinder", center_m, orientation, radius_m: d.x, halfHeight_m: d.y };
  if (kind === SVO_PRIMITIVE_KINDS.roundedCylinder) return { ...identity, kind: "rounded-cylinder", center_m, orientation, radius_m: d.x, halfHeight_m: d.y, edgeRadius_m: d.z };
  if (kind === SVO_PRIMITIVE_KINDS.ellipsoid) return { ...identity, kind: "ellipsoid", center_m, orientation, radii_m: d };
  if (kind === SVO_PRIMITIVE_KINDS.torus) return { ...identity, kind: "torus", center_m, orientation, majorRadius_m: d.x, minorRadius_m: d.y };
  if (kind === SVO_PRIMITIVE_KINDS.cone) return { ...identity, kind: "cone", center_m, orientation, baseRadius_m: d.x, halfHeight_m: d.y, topRadius_m: d.z };
  if (kind === SVO_PRIMITIVE_KINDS.roundCone) return { ...identity, kind: "round-cone", center_m, orientation, baseRadius_m: d.x, halfHeight_m: d.y, topRadius_m: d.z };
  if (kind === SVO_PRIMITIVE_KINDS.smoothUnionCluster) {
    // No `packing`: the record carries only the reference, and inventing a
    // default here would produce a smooth ellipsoid that renders plausibly and
    // is not the authored shape. A caller that needs the packing resolves it.
    return { ...identity, kind: "smooth-union-cluster", center_m, orientation, lobeRadii_m: d, clusterReference: words[base + 13] };
  }
  if (kind === SVO_PRIMITIVE_KINDS.fieldProgram) {
    // No `program`, for the reason above: a default tape is a smooth ellipsoid,
    // which is a shape the scene never asked for and would raise nothing. The
    // record's dimensions are already the widened bound, so the recovered
    // envelope is the published one rather than a floor that has to be re-grown.
    return { ...identity, kind: "field-program", center_m, orientation, envelopeRadii_m: d, fieldProgramReference: words[base + 13] };
  }
  if (kind === SVO_PRIMITIVE_KINDS.terrainHeightfield) {
    return { ...identity, kind: "terrain-heightfield", terrainReference: words[base + 13], normalEpsilon_m: d.x };
  }
  throw new RangeError(`Unknown SVO primitive kind ${kind}`);
}

/** Deterministic CPU unpack mirror used by tests, diagnostics, and capture tools. */
export function unpackSvoPrimitiveRecords(packed: Uint32Array): SvoPrimitiveDescriptor[] {
  if (packed.length % SVO_PRIMITIVE_RECORD_WORDS !== 0) throw new RangeError("Packed SVO primitive data has a partial record");
  const copied = new Uint32Array(packed);
  const floats = new Float32Array(copied.buffer);
  const result: SvoPrimitiveDescriptor[] = [];
  for (let base = 0; base < copied.length; base += SVO_PRIMITIVE_RECORD_WORDS) {
    result.push(canonicalSvoPrimitive(descriptorFromRecord(copied, floats, base)));
  }
  return result;
}

/** Map the repository's existing rigid dimension semantics into the render ABI. */
export function svoPrimitiveForRigidBody(
  body: Pick<RigidBodyDescription, "shape" | "dimensions_m" | "position_m" | "orientation">,
  primitiveId: number,
  ownerId: number,
  materialId = materialIdForRigidShape(body.shape),
): Exclude<SvoPrimitiveDescriptor, SvoEllipsoidPrimitive | SvoTerrainHeightfieldPrimitive> {
  const identity = { primitiveId, materialId, ownerId, center_m: { ...body.position_m } };
  if (body.shape === "sphere") return canonicalSvoPrimitive({ ...identity, kind: "sphere", radius_m: body.dimensions_m.x }) as SvoSpherePrimitive;
  if (body.shape === "box") return canonicalSvoPrimitive({
    ...identity, kind: "box", orientation: { ...body.orientation },
    halfExtents_m: { x: body.dimensions_m.x / 2, y: body.dimensions_m.y / 2, z: body.dimensions_m.z / 2 },
  }) as SvoBoxPrimitive;
  if (body.shape === "capsule") return canonicalSvoPrimitive({
    ...identity, kind: "capsule", orientation: { ...body.orientation },
    radius_m: body.dimensions_m.x, segmentHalfLength_m: body.dimensions_m.y / 2,
  }) as SvoCapsulePrimitive;
  return canonicalSvoPrimitive({
    ...identity, kind: "cylinder", orientation: { ...body.orientation },
    radius_m: body.dimensions_m.x, halfHeight_m: body.dimensions_m.y / 2,
  }) as SvoCylinderPrimitive;
}

function rotate(q: Quaternion, v: Vec3): Vec3 {
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  return {
    x: v.x + q.w * tx + (q.y * tz - q.z * ty),
    y: v.y + q.w * ty + (q.z * tx - q.x * tz),
    z: v.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

function inverseRotate(q: Quaternion, v: Vec3): Vec3 {
  return rotate({ w: q.w, x: -q.x, y: -q.y, z: -q.z }, v);
}

function normalize(v: Vec3): Vec3 | null {
  const length = Math.hypot(v.x, v.y, v.z);
  return length > NORMAL_EPSILON ? { x: v.x / length, y: v.y / length, z: v.z / length } : null;
}

function localPoint(descriptor: Exclude<SvoPrimitiveDescriptor, SvoTerrainHeightfieldPrimitive>, worldPoint_m: Vec3): { point: Vec3; orientation: Quaternion } {
  finiteVec3(worldPoint_m, "Primitive query point");
  const orientation = descriptorOrientation(descriptor);
  return {
    point: inverseRotate(orientation, {
      x: worldPoint_m.x - descriptor.center_m.x,
      y: worldPoint_m.y - descriptor.center_m.y,
      z: worldPoint_m.z - descriptor.center_m.z,
    }),
    orientation,
  };
}

function worldNormal(orientation: Quaternion, normal: Vec3 | null): Vec3 | null {
  return normal ? normalize(rotate(orientation, normal)) : null;
}

interface CanonicalPrimitiveRay {
  origin_m: Vec3;
  direction: Vec3;
  tMin_m: number;
  tMax_m: number;
}

interface LocalPrimitiveRayHit {
  t_m: number;
  normal: Vec3;
  featureId: number;
}

function canonicalPrimitiveRay(ray: SvoPrimitiveRay): CanonicalPrimitiveRay {
  finiteVec3(ray.origin_m, "Primitive ray origin");
  finiteVec3(ray.direction, "Primitive ray direction");
  const direction = normalize(ray.direction);
  if (!direction) throw new RangeError("Primitive ray direction must be non-zero");
  const tMin_m = ray.tMin_m ?? 0;
  const tMax_m = ray.tMax_m ?? Number.POSITIVE_INFINITY;
  if (!Number.isFinite(tMin_m) || tMin_m < 0) throw new RangeError("Primitive ray minimum must be a non-negative finite metre distance");
  if (!(Number.isFinite(tMax_m) || tMax_m === Number.POSITIVE_INFINITY) || tMax_m < tMin_m) {
    throw new RangeError("Primitive ray maximum must be at least its minimum");
  }
  return { origin_m: { ...ray.origin_m }, direction, tMin_m, tMax_m };
}

/** Sorted roots of a*t^2 + 2*b*t + c, retaining exact tangent contact. */
function quadraticRoots(a: number, b: number, c: number): readonly number[] {
  if (!(a > NORMAL_EPSILON)) return [];
  const discriminant = b * b - a * c;
  const tolerance = 1e-12 * Math.max(1, Math.abs(b * b), Math.abs(a * c));
  if (discriminant < -tolerance) return [];
  const root = Math.sqrt(Math.max(0, discriminant));
  if (root === 0) return [-b / a];
  // This form avoids losing the near root when b and sqrt(discriminant) are close.
  const q = -b - Math.sign(b || 1) * root;
  const first = q / a;
  const second = c / q;
  return first <= second ? [first, second] : [second, first];
}

function inPrimitiveRayRange(t_m: number, ray: CanonicalPrimitiveRay): boolean {
  const tolerance = 1e-10 * Math.max(1, Math.abs(t_m), Math.abs(ray.tMin_m), Number.isFinite(ray.tMax_m) ? Math.abs(ray.tMax_m) : 1);
  return Number.isFinite(t_m) && t_m >= ray.tMin_m - tolerance && t_m <= ray.tMax_m + tolerance;
}

function nearestLocalHit(candidates: readonly LocalPrimitiveRayHit[], ray: CanonicalPrimitiveRay): LocalPrimitiveRayHit | null {
  let nearest: LocalPrimitiveRayHit | null = null;
  for (const candidate of candidates) {
    if (!inPrimitiveRayRange(candidate.t_m, ray)) continue;
    const t_m = Math.max(ray.tMin_m, candidate.t_m);
    if (!nearest || t_m < nearest.t_m) nearest = { ...candidate, t_m };
  }
  return nearest;
}

function localPrimitiveRay(
  descriptor: SvoFinitePrimitiveDescriptor,
  ray: CanonicalPrimitiveRay,
): { origin: Vec3; direction: Vec3; orientation: Quaternion } {
  const orientation = descriptorOrientation(descriptor);
  const offset = {
    x: ray.origin_m.x - descriptor.center_m.x,
    y: ray.origin_m.y - descriptor.center_m.y,
    z: ray.origin_m.z - descriptor.center_m.z,
  };
  return { origin: inverseRotate(orientation, offset), direction: inverseRotate(orientation, ray.direction), orientation };
}

function sphereRayCandidates(origin: Vec3, direction: Vec3, radius_m: number, centerY_m = 0): readonly number[] {
  const offset = { x: origin.x, y: origin.y - centerY_m, z: origin.z };
  return quadraticRoots(
    direction.x ** 2 + direction.y ** 2 + direction.z ** 2,
    offset.x * direction.x + offset.y * direction.y + offset.z * direction.z,
    offset.x ** 2 + offset.y ** 2 + offset.z ** 2 - radius_m ** 2,
  );
}

function intersectSphereLocal(
  descriptor: SvoSpherePrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const candidates = sphereRayCandidates(origin, direction, descriptor.radius_m).map((t_m) => {
    const point = { x: origin.x + direction.x * t_m, y: origin.y + direction.y * t_m, z: origin.z + direction.z * t_m };
    return { t_m, normal: normalize(point)!, featureId: SVO_PRIMITIVE_FEATURES.smooth };
  });
  return nearestLocalHit(candidates, ray);
}

function intersectBoxLocal(
  descriptor: SvoBoxPrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const axes = ["x", "y", "z"] as const;
  let enter = Number.NEGATIVE_INFINITY;
  let exit = Number.POSITIVE_INFINITY;
  let enterAxis: typeof axes[number] = "x";
  let exitAxis: typeof axes[number] = "x";
  for (const axis of axes) {
    const extent = descriptor.halfExtents_m[axis];
    if (Math.abs(direction[axis]) <= NORMAL_EPSILON) {
      if (origin[axis] < -extent || origin[axis] > extent) return null;
      continue;
    }
    let near = (-extent - origin[axis]) / direction[axis];
    let far = (extent - origin[axis]) / direction[axis];
    if (near > far) [near, far] = [far, near];
    // Strict comparisons preserve the stable X -> Y -> Z feature tie.
    if (near > enter) { enter = near; enterAxis = axis; }
    if (far < exit) { exit = far; exitAxis = axis; }
    if (exit < enter) return null;
  }
  const candidates = ([{ t_m: enter, axis: enterAxis }, { t_m: exit, axis: exitAxis }]).map(({ t_m, axis }) => {
    const coordinate = origin[axis] + direction[axis] * t_m;
    return {
      t_m,
      normal: {
        x: axis === "x" ? Math.sign(coordinate || -direction.x || 1) : 0,
        y: axis === "y" ? Math.sign(coordinate || -direction.y || 1) : 0,
        z: axis === "z" ? Math.sign(coordinate || -direction.z || 1) : 0,
      },
      featureId: axis === "x" ? SVO_PRIMITIVE_FEATURES.boxFaceX
        : axis === "y" ? SVO_PRIMITIVE_FEATURES.boxFaceY : SVO_PRIMITIVE_FEATURES.boxFaceZ,
    };
  });
  return nearestLocalHit(candidates, ray);
}

function intersectCapsuleLocal(
  descriptor: SvoCapsulePrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const candidates: LocalPrimitiveRayHit[] = [];
  const radialA = direction.x ** 2 + direction.z ** 2;
  const radialB = origin.x * direction.x + origin.z * direction.z;
  const radialC = origin.x ** 2 + origin.z ** 2 - descriptor.radius_m ** 2;
  for (const t_m of quadraticRoots(radialA, radialB, radialC)) {
    const y = origin.y + direction.y * t_m;
    if (y >= -descriptor.segmentHalfLength_m && y <= descriptor.segmentHalfLength_m) {
      const point = { x: origin.x + direction.x * t_m, y: 0, z: origin.z + direction.z * t_m };
      candidates.push({ t_m, normal: normalize(point)!, featureId: SVO_PRIMITIVE_FEATURES.smooth });
    }
  }
  for (const sign of [-1, 1] as const) {
    const centerY = sign * descriptor.segmentHalfLength_m;
    for (const t_m of sphereRayCandidates(origin, direction, descriptor.radius_m, centerY)) {
      const point = {
        x: origin.x + direction.x * t_m,
        y: origin.y + direction.y * t_m - centerY,
        z: origin.z + direction.z * t_m,
      };
      if ((sign < 0 && point.y <= 0) || (sign > 0 && point.y >= 0)) {
        candidates.push({ t_m, normal: normalize(point)!, featureId: SVO_PRIMITIVE_FEATURES.smooth });
      }
    }
  }
  return nearestLocalHit(candidates, ray);
}

function intersectCylinderLocal(
  descriptor: SvoCylinderPrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const candidates: LocalPrimitiveRayHit[] = [];
  const radialA = direction.x ** 2 + direction.z ** 2;
  const radialB = origin.x * direction.x + origin.z * direction.z;
  const radialC = origin.x ** 2 + origin.z ** 2 - descriptor.radius_m ** 2;
  for (const t_m of quadraticRoots(radialA, radialB, radialC)) {
    const y = origin.y + direction.y * t_m;
    if (y >= -descriptor.halfHeight_m && y <= descriptor.halfHeight_m) {
      const point = { x: origin.x + direction.x * t_m, y: 0, z: origin.z + direction.z * t_m };
      candidates.push({ t_m, normal: normalize(point)!, featureId: SVO_PRIMITIVE_FEATURES.cylinderSide });
    }
  }
  if (Math.abs(direction.y) > NORMAL_EPSILON) {
    for (const sign of [-1, 1] as const) {
      const t_m = (sign * descriptor.halfHeight_m - origin.y) / direction.y;
      const x = origin.x + direction.x * t_m;
      const z = origin.z + direction.z * t_m;
      if (x * x + z * z <= descriptor.radius_m ** 2 * (1 + 1e-12)) {
        // Caps are appended after sides, then promoted on an exact rim tie below.
        candidates.push({ t_m, normal: { x: 0, y: sign, z: 0 }, featureId: SVO_PRIMITIVE_FEATURES.cylinderCap });
      }
    }
  }
  const hit = nearestLocalHit(candidates, ray);
  if (!hit || hit.featureId === SVO_PRIMITIVE_FEATURES.cylinderCap) return hit;
  const tiedCap = candidates.find((candidate) => candidate.featureId === SVO_PRIMITIVE_FEATURES.cylinderCap
    && inPrimitiveRayRange(candidate.t_m, ray) && Math.abs(candidate.t_m - hit.t_m) <= 1e-10 * Math.max(1, hit.t_m));
  return tiedCap ? { ...tiedCap, t_m: hit.t_m } : hit;
}

function intersectEllipsoidLocal(
  descriptor: SvoEllipsoidPrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const scaledOrigin = {
    x: origin.x / descriptor.radii_m.x,
    y: origin.y / descriptor.radii_m.y,
    z: origin.z / descriptor.radii_m.z,
  };
  const scaledDirection = {
    x: direction.x / descriptor.radii_m.x,
    y: direction.y / descriptor.radii_m.y,
    z: direction.z / descriptor.radii_m.z,
  };
  const roots = quadraticRoots(
    scaledDirection.x ** 2 + scaledDirection.y ** 2 + scaledDirection.z ** 2,
    scaledOrigin.x * scaledDirection.x + scaledOrigin.y * scaledDirection.y + scaledOrigin.z * scaledDirection.z,
    scaledOrigin.x ** 2 + scaledOrigin.y ** 2 + scaledOrigin.z ** 2 - 1,
  );
  const candidates = roots.map((t_m) => {
    const point = { x: origin.x + direction.x * t_m, y: origin.y + direction.y * t_m, z: origin.z + direction.z * t_m };
    return {
      t_m,
      normal: normalize({
        x: point.x / descriptor.radii_m.x ** 2,
        y: point.y / descriptor.radii_m.y ** 2,
        z: point.z / descriptor.radii_m.z ** 2,
      })!,
      featureId: SVO_PRIMITIVE_FEATURES.smooth,
    };
  });
  return nearestLocalHit(candidates, ray);
}

/** Exact `uint32` mirror of `svoClusterHash`; one avalanche per lattice cell. */
function clusterCellHash(cellX: number, cellY: number, cellZ: number, seed: number): number {
  let hash = seed >>> 0;
  hash = (hash ^ Math.imul(cellX | 0, 0x9e37_79b1)) >>> 0;
  hash = (hash ^ Math.imul(cellY | 0, 0x85eb_ca77)) >>> 0;
  hash = (hash ^ Math.imul(cellZ | 0, 0xc2b2_ae3d)) >>> 0;
  hash = (hash ^ (hash >>> 16)) >>> 0;
  hash = Math.imul(hash, 0x7feb_352d) >>> 0;
  hash = (hash ^ (hash >>> 15)) >>> 0;
  hash = Math.imul(hash, 0x846c_a68b) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
}

/**
 * Three decorrelated offsets in [-1, 1] from one avalanche.
 *
 * Ten bits an axis rather than three separate hashes: the jitter this scales is
 * at most three tenths of a lattice period — millimetres at the scales this
 * renderer draws — so a thousand levels resolves it to microns, and it runs
 * this eight times per evaluation and roughly sixty times per covering ray.
 */
function clusterCellJitter(cellX: number, cellY: number, cellZ: number, seed: number): Vec3 {
  const hash = clusterCellHash(cellX, cellY, cellZ, seed);
  return {
    x: ((hash & 0x3ff) / 1023) * 2 - 1,
    y: (((hash >>> 10) & 0x3ff) / 1023) * 2 - 1,
    z: (((hash >>> 20) & 0x3ff) / 1023) * 2 - 1,
  };
}

/**
 * The polynomial smooth minimum, and the reason the aggregate is traceable at
 * all: it is a strict lower bound on `min(a, b)`, so a sphere trace stepping by
 * it understeps and can never cross the surface it is looking for.
 */
function smoothMinimum(a: number, b: number, k: number): number {
  if (!(k > 0)) return Math.min(a, b);
  const h = Math.min(1, Math.max(0, 0.5 + (0.5 * (b - a)) / k));
  return b + (a - b) * h - k * h * (1 - h);
}

/**
 * One octave of the lattice field: every length the neighbourhood argument
 * involves, at one scale, plus the seed that decorrelates it from the others.
 */
interface ClusterLatticeOctave {
  readonly latticeLobeRadius_m: number;
  readonly latticePeriod_m: number;
  readonly jitter: number;
  readonly smoothRadius_m: number;
  readonly seed: number;
}

/** Surface distance to the jittered sphere of one lattice cell. */
function clusterSphereDistance(
  point: Vec3,
  cellX: number,
  cellY: number,
  cellZ: number,
  octave: ClusterLatticeOctave,
): number {
  const period = octave.latticePeriod_m;
  const jitter = clusterCellJitter(cellX, cellY, cellZ, octave.seed);
  return Math.hypot(
    point.x - (cellX + 0.5 + octave.jitter * jitter.x) * period,
    point.y - (cellY + 0.5 + octave.jitter * jitter.y) * period,
    point.z - (cellZ + 0.5 + octave.jitter * jitter.z) * period,
  ) - octave.latticeLobeRadius_m;
}

/**
 * The fused packing of one octave, minimised over the point's own neighbourhood.
 *
 * Two passes, and the split is what makes sixty-four cells affordable. The
 * inner eight — the cell centres immediately surrounding the point — always
 * matter and are evaluated outright; they also establish a tight running
 * minimum. The outer fifty-six are then rejected by the distance to their
 * *unjittered* centre, which costs a subtract and a dot product and no hash at
 * all, and only the handful that could still reach the running minimum are
 * evaluated. On a representative packing that is roughly fifteen of the fifty-six.
 *
 * The fold order is fixed and shared with the WGSL because a polynomial smooth
 * minimum is not associative: folding the same spheres in a different sequence
 * gives a different answer, by up to a quarter of the smoothing radius.
 */
function clusterLatticeOctaveDistance(point: Vec3, octave: ClusterLatticeOctave): number {
  const period = octave.latticePeriod_m;
  const span = SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD;
  const ring = span / 2 - 1;
  const baseX = Math.floor(point.x / period - 0.5) - ring;
  const baseY = Math.floor(point.y / period - 0.5) - ring;
  const baseZ = Math.floor(point.z / period - 0.5) - ring;
  let distance = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const sphere = clusterSphereDistance(
      point, baseX + ring + (corner & 1), baseY + ring + ((corner >> 1) & 1), baseZ + ring + ((corner >> 2) & 1), octave,
    );
    distance = corner === 0 ? sphere : smoothMinimum(distance, sphere, octave.smoothRadius_m);
  }
  // The furthest a sphere can sit from its own unjittered centre, plus the
  // reach of the lobe and of the blend. A cell whose nominal centre is
  // further than this beyond the running minimum cannot change it.
  const reach = octave.jitter * period * Math.sqrt(3) + octave.latticeLobeRadius_m + octave.smoothRadius_m;
  for (let index = 0; index < span * span * span; index += 1) {
    const stepX = index % span;
    const stepY = Math.floor(index / span) % span;
    const stepZ = Math.floor(index / (span * span)) % span;
    if (stepX >= ring && stepX <= ring + 1 && stepY >= ring && stepY <= ring + 1 && stepZ >= ring && stepZ <= ring + 1) continue;
    const cellX = baseX + stepX, cellY = baseY + stepY, cellZ = baseZ + stepZ;
    const nominal = Math.hypot(
      point.x - (cellX + 0.5) * period,
      point.y - (cellY + 0.5) * period,
      point.z - (cellZ + 0.5) * period,
    );
    if (nominal - reach > distance) continue;
    distance = smoothMinimum(distance, clusterSphereDistance(point, cellX, cellY, cellZ, octave), octave.smoothRadius_m);
  }
  return distance;
}

/**
 * The seed octave `k` hashes its cells with.
 *
 * Zero for octave zero, exactly — `imul(0, ...)` is zero and `seed ^ 0` is
 * `seed` — which is what makes a one-octave field bit-for-bit the field this
 * kind published before octaves existed. Without that the octave parameter
 * would silently re-pack every set already authored against it.
 */
function clusterOctaveSeed(seed: number, octave: number): number {
  return (seed ^ Math.imul(octave, 0x9e37_79b1)) >>> 0;
}

/**
 * The lattice field: octaves of the same packing, each a halving of the last,
 * smooth-unioned onto the running result.
 *
 * Both the scale and the blend halve together. That is what carries the
 * neighbourhood argument up the stack unchanged: octave `k` is the octave-zero
 * packing viewed at `2^k`, so if the first satisfies the period condition every
 * one after it does, and each is separately a Lipschitz-1 lower bound. The
 * smooth minimum that fuses them is a convex combination of their gradients, so
 * the stack is one as well.
 */
function clusterLatticeDistance(point: Vec3, packing: SvoClusterLatticeField): number {
  const octaves = packing.octaves ?? 1;
  let distance = 0;
  for (let octave = 0; octave < octaves; octave += 1) {
    // A negative power of two, so octave zero scales every length by an exact
    // 1 and the single-octave result is unchanged to the last bit.
    const scale = 2 ** -octave;
    const smoothRadius_m = packing.smoothRadius_m * scale;
    const sample = clusterLatticeOctaveDistance(point, {
      latticeLobeRadius_m: packing.latticeLobeRadius_m * scale,
      latticePeriod_m: packing.latticePeriod_m * scale,
      jitter: packing.jitter,
      smoothRadius_m,
      seed: clusterOctaveSeed(packing.seed, octave),
    });
    distance = octave === 0 ? sample : smoothMinimum(distance, sample, smoothRadius_m);
  }
  return distance;
}

/**
 * One anisotropic lobe, in the envelope's normalised space — the space in which
 * the envelope is the unit ball. Both the centre and the half-axes are
 * fractions of the envelope, not metres.
 */
interface ClusterSeededLobe {
  readonly center: Vec3;
  readonly halfAxes: Vec3;
  readonly orientation: Quaternion;
}

/** The three placement parameters, with the defaults an absent one takes. */
function seededLobePlacement(packing: SvoClusterSeededLobesField): {
  span: number; spanSpread: number; displacement: number;
} {
  return {
    span: packing.lobeSpan ?? SVO_CLUSTER_LOBE_DEFAULT_SPAN,
    spanSpread: packing.lobeSpanSpread ?? SVO_CLUSTER_LOBE_DEFAULT_SPAN_SPREAD,
    displacement: packing.displacement ?? SVO_CLUSTER_LOBE_DEFAULT_DISPLACEMENT,
  };
}

/**
 * Where lobe `index` sits, how big it is and which way it faces.
 *
 * Four hashes of the same avalanche the lattice uses, so the arrangement is
 * reproducible from `seed` alone and nothing about it is stored.
 *
 * Everything is in the envelope's *normalised* space, and that is what makes
 * the solid scale with what it was authored against. Sizing a lobe against the
 * envelope's shortest half-axis instead — the obvious reading of "inside it" —
 * caps every lobe of a 3:2 envelope at its narrow dimension and leaves the
 * result occupying a fraction of the volume the voxelizer and the opacity
 * pyramid have already called solid.
 *
 * The containment argument is then one line. A lobe's surface reaches
 * `|v| + span` from the origin in normalised space, the envelope's own
 * Lipschitz-1 lower bound at radius `u` is `(1 - u)·Rmin`, and `v` is placed
 * within `1 - span - smooth/Rmin` — so the clearance at the lobe's surface is
 * at least the blend and the hard `max` never bites.
 *
 * Lobe zero is centred, which is the cheapest guarantee that every other lobe
 * has something to fuse to. It is not a proof of connectedness: a low count, a
 * high anisotropy and a high displacement can still leave a lobe standing off
 * on its own, and the blend is the knob that closes it.
 */
function clusterSeededLobe(index: number, lobeRadii_m: Vec3, packing: SvoClusterSeededLobesField): ClusterSeededLobe {
  const shortest_m = Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
  const placement = seededLobePlacement(packing);
  const offsetHash = clusterCellJitter(index, 0, 0, packing.seed);
  const shapeHash = clusterCellJitter(index, 1, 0, packing.seed);
  const axisHash = clusterCellJitter(index, 2, 0, packing.seed);
  const spreadHash = clusterCellJitter(index, 3, 0, packing.seed);

  // Axis ratios spanning at most `anisotropy` end to end, then rescaled so the
  // longest is one — which is what lets `span` alone stand for the lobe's reach
  // in the containment argument, whatever the anisotropy did to the other two.
  const raw = {
    x: Math.pow(packing.anisotropy, 0.5 * shapeHash.x),
    y: Math.pow(packing.anisotropy, 0.5 * shapeHash.y),
    z: Math.pow(packing.anisotropy, 0.5 * shapeHash.z),
  };
  const longest = Math.max(raw.x, raw.y, raw.z);
  const span = placement.span + placement.spanSpread * 0.5 * (spreadHash.x + 1);
  const halfAxes = { x: span * raw.x / longest, y: span * raw.y / longest, z: span * raw.z / longest };

  // The normalised radius the centre may travel over, and the share of it this
  // lobe spends. Zero for lobe zero, the shared core. The hash gives the
  // direction only and the radius is the budget share, so a lobe is placed *at*
  // a chosen distance rather than wherever a cube-uniform hash happened to fall
  // — which averaged well under half the budget and left the solid clear of its
  // own envelope on every side.
  const budget = index === 0
    ? 0
    : Math.max(0, 1 - span - packing.smoothRadius_m / shortest_m) * placement.displacement;
  const offsetLength = Math.hypot(offsetHash.x, offsetHash.y, offsetHash.z);
  const offsetScale = offsetLength > NORMAL_EPSILON ? budget / offsetLength : 0;
  const center = {
    x: offsetHash.x * offsetScale,
    y: offsetHash.y * offsetScale,
    z: offsetHash.z * offsetScale,
  };

  // Axis-angle rather than four hashed components normalised: a quaternion
  // built this way is a rotation for any hash at all, including the degenerate
  // one, which the shader has no way to reject.
  const axisLength = Math.hypot(axisHash.x, axisHash.y, axisHash.z);
  const axis = axisLength > NORMAL_EPSILON
    ? { x: axisHash.x / axisLength, y: axisHash.y / axisLength, z: axisHash.z / axisLength }
    : { x: 0, y: 1, z: 0 };
  const half = 0.5 * Math.PI * spreadHash.y;
  const sine = Math.sin(half);
  return {
    center,
    halfAxes,
    orientation: { w: Math.cos(half), x: axis.x * sine, y: axis.y * sine, z: axis.z * sine },
  };
}

/**
 * The seeded-lobes field: oriented anisotropic lobes fused into one solid.
 *
 * A lobe is the affine image `M = diag(R)·Q·diag(k)` of the unit ball, and its
 * distance is the same trick the envelope uses one level up: `(|M⁻¹w| - 1)`
 * shares the lobe's exact zero set, and multiplying by a lower bound on `M`'s
 * smallest singular value makes it Lipschitz-1 — `|∇f| ≤ σ / σmin(M) ≤ 1`.
 * `min(R)·min(k)` is that lower bound, because `σmin(AB) ≥ σmin(A)·σmin(B)` and
 * a rotation has none. It is not tight, and being loose here costs march steps
 * rather than correctness: the field understeps, which is the safe direction.
 *
 * The smooth minimum over the lobes is a convex combination of their gradients,
 * so the fused field is unit-Lipschitz as well.
 */
function clusterSeededLobesDistance(
  point: Vec3,
  lobeRadii_m: Vec3,
  packing: SvoClusterSeededLobesField,
): number {
  const shortest_m = Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
  const normalized = { x: point.x / lobeRadii_m.x, y: point.y / lobeRadii_m.y, z: point.z / lobeRadii_m.z };
  let distance = 0;
  for (let index = 0; index < packing.lobeCount; index += 1) {
    const lobe = clusterSeededLobe(index, lobeRadii_m, packing);
    const local = inverseRotate(lobe.orientation, {
      x: normalized.x - lobe.center.x,
      y: normalized.y - lobe.center.y,
      z: normalized.z - lobe.center.z,
    });
    const scaled = Math.hypot(local.x / lobe.halfAxes.x, local.y / lobe.halfAxes.y, local.z / lobe.halfAxes.z);
    const sample = (scaled - 1) * shortest_m * Math.min(lobe.halfAxes.x, lobe.halfAxes.y, lobe.halfAxes.z);
    distance = index === 0 ? sample : smoothMinimum(distance, sample, packing.smoothRadius_m);
  }
  return distance;
}

/**
 * Exact distance to a round cone: the convex hull of two spheres.
 *
 * Exact, and therefore Lipschitz-1 by definition rather than by argument, which
 * is the whole reason a tube is built out of these rather than out of a capsule
 * whose radius is lerped along it — that construction is not a distance at all
 * wherever the taper is steep. `span > |Δradius|` is the condition that keeps
 * one sphere from swallowing the other; the packing validator rejects anything
 * else, so `a2` here is positive.
 */
function clusterRoundConeDistance(
  point: Vec3,
  from: Vec3,
  to: Vec3,
  fromRadius_m: number,
  toRadius_m: number,
): number {
  const axis = { x: to.x - from.x, y: to.y - from.y, z: to.z - from.z };
  const axisSquared = axis.x ** 2 + axis.y ** 2 + axis.z ** 2;
  const taper = fromRadius_m - toRadius_m;
  const lateralSquared = axisSquared - taper * taper;
  const inverseAxisSquared = 1 / axisSquared;
  const offset = { x: point.x - from.x, y: point.y - from.y, z: point.z - from.z };
  const along = offset.x * axis.x + offset.y * axis.y + offset.z * axis.z;
  const beyond = along - axisSquared;
  const perpendicular = {
    x: offset.x * axisSquared - axis.x * along,
    y: offset.y * axisSquared - axis.y * along,
    z: offset.z * axisSquared - axis.z * along,
  };
  const perpendicularSquared = perpendicular.x ** 2 + perpendicular.y ** 2 + perpendicular.z ** 2;
  const alongSquared = along * along * axisSquared;
  const beyondSquared = beyond * beyond * axisSquared;
  // The two cap regions and the lateral band, separated by the tangent plane
  // the taper defines rather than by a clamp along the axis.
  const split = Math.sign(taper) * taper * taper * perpendicularSquared;
  if (Math.sign(beyond) * lateralSquared * beyondSquared > split) {
    return Math.sqrt(perpendicularSquared + beyondSquared) * inverseAxisSquared - toRadius_m;
  }
  if (Math.sign(along) * lateralSquared * alongSquared < split) {
    return Math.sqrt(perpendicularSquared + alongSquared) * inverseAxisSquared - fromRadius_m;
  }
  return (Math.sqrt(perpendicularSquared * lateralSquared * inverseAxisSquared) + along * taper) * inverseAxisSquared
    - fromRadius_m;
}

/**
 * The tapered-sweep field: round-cone segments fused in authored order.
 *
 * Fold order is fixed and mirrored in WGSL for the same reason the lattice's is
 * — a polynomial smooth minimum is not associative — and it is why the polyline
 * is stored in sweep order rather than sorted by anything.
 */
function clusterTaperedSweepDistance(point: Vec3, packing: SvoClusterTaperedSweepField): number {
  let distance = 0;
  for (let index = 0; index + 1 < packing.points.length; index += 1) {
    const from = packing.points[index];
    const to = packing.points[index + 1];
    const sample = clusterRoundConeDistance(point, from.position_m, to.position_m, from.radius_m, to.radius_m);
    distance = index === 0 ? sample : smoothMinimum(distance, sample, packing.smoothRadius_m);
  }
  return distance;
}

const NOISE_FOLIAGE_CLUSTER_SALT = 0x68bc_21eb;
const NOISE_FOLIAGE_DETAIL_SALT = 0x02e5_be93;
/** Gradient bound of smoothstep-trilinear value noise in a unit domain. */
const NOISE_FOLIAGE_VALUE_GRADIENT_BOUND = 1.5 * Math.sqrt(3);
/** Share of the envelope radius reserved for a density fade, not geometry. */
const NOISE_FOLIAGE_EDGE_FADE = 0.38;

/**
 * Density above zero is occupied. Dividing its negation by the complete
 * gradient bound turns the iso-field into a Lipschitz-1 distance lower bound,
 * which is what lets the existing sphere tracer march it without tunnelling.
 */
function clusterNoiseFoliageDistance(
  point: Vec3,
  lobeRadii_m: Vec3,
  packing: SvoClusterNoiseFoliageField,
): number {
  const clusterFrequency = 1 / packing.clusterPeriod_m;
  const detailFrequency = 1 / packing.detailPeriod_m;
  const cluster = sampleSvoProceduralNoise(
    point, [clusterFrequency, clusterFrequency, clusterFrequency],
    (packing.seed ^ NOISE_FOLIAGE_CLUSTER_SALT) >>> 0,
  );
  const detail = sampleSvoProceduralNoise(
    point, [detailFrequency, detailFrequency, detailFrequency],
    (packing.seed ^ NOISE_FOLIAGE_DETAIL_SALT) >>> 0,
  );
  // Smoothstep the low-frequency octave into broad density islands. This is
  // the cluster-of-increased-density layer: values around the middle separate
  // more decisively while extrema stay smooth, so the canopy builds cauliflower
  // masses instead of merely embossing an ellipsoid with gentle noise.
  const clustered = cluster * cluster * (3 - 2 * cluster);
  const normalizedRadius = Math.hypot(
    point.x / lobeRadii_m.x,
    point.y / lobeRadii_m.y,
    point.z / lobeRadii_m.z,
  );
  const interior = Math.max(0, 1 - normalizedRadius);
  const fadeLinear = Math.max(0, Math.min(1, interior / NOISE_FOLIAGE_EDGE_FADE));
  const fade = fadeLinear * fadeLinear * (3 - 2 * fadeLinear);
  const rawDensity = packing.clusterWeight * clustered
    + packing.detailWeight * detail
    + packing.interiorBias;
  // Fade density to zero before the acceleration envelope. The hard ellipsoid
  // clip below is still the conservative bound every raster and octree path
  // expects, but it is now strictly outside the zero surface and can never be
  // the rounded surface the user sees.
  const density = fade * rawDensity - packing.threshold;
  const shortest_m = Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
  const noiseGradient_mInv = NOISE_FOLIAGE_VALUE_GRADIENT_BOUND * (
    1.5 * packing.clusterWeight * clusterFrequency + packing.detailWeight * detailFrequency
  );
  // smoothstep's derivative peaks at 1.5. `rawDensity` is at most the sum of
  // its weights, so this includes the product rule for `fade * rawDensity`.
  const fadeGradient_mInv = 1.5 / (NOISE_FOLIAGE_EDGE_FADE * shortest_m);
  const lipschitz_mInv = noiseGradient_mInv
    + (packing.clusterWeight + packing.detailWeight + packing.interiorBias) * fadeGradient_mInv;
  return -density / lipschitz_mInv;
}

/** The field a packing names, before the envelope clip. */
function clusterFieldDistance(point: Vec3, lobeRadii_m: Vec3, packing: SvoSmoothUnionClusterPacking): number {
  if (packing.field === "noise-foliage") return clusterNoiseFoliageDistance(point, lobeRadii_m, packing);
  if (packing.field === "seeded-lobes") return clusterSeededLobesDistance(point, lobeRadii_m, packing);
  if (packing.field === "tapered-sweep") return clusterTaperedSweepDistance(point, packing);
  return clusterLatticeDistance(point, packing);
}

/**
 * The smallest solid feature a field draws.
 *
 * The gradient step is a quarter of it. Per field because the number that sets
 * it is different in each: a lattice's finest lobe, a lobe set's blend, a
 * sweep's thinnest section. Reading a fixed fraction of the envelope instead
 * would step across several features at once and hand back the envelope's own
 * normal.
 */
export function svoClusterFeatureRadius_m(lobeRadii_m: Vec3, packing: SvoSmoothUnionClusterPacking): number {
  if (packing.field === "noise-foliage") return 0.5 * packing.detailPeriod_m;
  if (packing.field === "seeded-lobes") {
    const shortest_m = Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
    // A lobe set with no blend still has a smallest feature: the thinnest a
    // lobe can be. `lobeSpan` fixes its longest half-axis and the anisotropy
    // shortens the others away from it by at most that ratio, so this is a
    // lower bound on every lobe's smallest half-axis whatever the seed did.
    return Math.max(packing.smoothRadius_m, seededLobePlacement(packing).span * shortest_m / Math.max(packing.anisotropy, 1));
  }
  if (packing.field === "tapered-sweep") {
    let thinnest_m = packing.points[0].radius_m;
    for (const point of packing.points) thinnest_m = Math.min(thinnest_m, point.radius_m);
    return thinnest_m;
  }
  // The finest octave's lobe. An exact power of two, so a one-octave field
  // keeps the step it has always used.
  return packing.latticeLobeRadius_m * 2 ** -((packing.octaves ?? 1) - 1);
}

/**
 * A Lipschitz-1 lower bound on the distance to an ellipsoid.
 *
 * Not the exact closest-point distance, which costs a 64-iteration bisection
 * and is evaluated here on every march step of every covering fragment. The
 * scaled-radius form `(|p/r| - 1) * min(r)` has gradient magnitude at most one
 * by construction and shares the exact ellipsoid's zero set, which is all the
 * clip needs: the surface is in the right place, and the trace understeps
 * toward it.
 *
 * Used twice over: once as the envelope every field is clipped to, and once as
 * the distance to an individual anisotropic lobe of the seeded-lobes field.
 */
function clusterLobeDistance(point: Vec3, lobeRadii_m: Vec3): number {
  const scaled = Math.hypot(point.x / lobeRadii_m.x, point.y / lobeRadii_m.y, point.z / lobeRadii_m.z);
  return (scaled - 1) * Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
}

/** The field, clipped to the envelope. A hard intersection, so the solid is inside the ellipsoid. */
function clusterDistance(point: Vec3, lobeRadii_m: Vec3, packing: SvoSmoothUnionClusterPacking): number {
  return Math.max(clusterFieldDistance(point, lobeRadii_m, packing), clusterLobeDistance(point, lobeRadii_m));
}

/**
 * A field's value and gradient — the CPU twin of `SvoClusterSample`.
 *
 * The WGSL beside this file carries the derivation; the short version is that
 * every leaf of a cluster field is a sphere, an ellipsoid or the convex hull of
 * two spheres, each with a closed-form normal, and both combinators above them
 * pass a gradient through exactly. Noise foliage differentiates its density
 * numerically at the terminal detail scale; all other fields use one analytic
 * pass instead of the four whole evaluations a tetrahedral difference spent.
 *
 * The distance arithmetic in each `...Sample` below is character-for-character
 * its distance-only twin's, because the two must not drift;
 * `tests/svo-cluster-gradient.test.ts` holds them to it and to a central
 * difference of the field itself.
 */
interface ClusterSample {
  readonly distance_m: number;
  readonly gradient: Vec3;
}

/**
 * The gradient a polynomial smooth minimum implies, which is the plain blend.
 *
 * `f = b + (a-b)h - k·h(1-h)` with `h = 0.5 + (b-a)/2k`, so
 * `∂f/∂a = h + (a-b)h' - k(1-2h)h'`; substituting `a-b = -(2h-1)k` and
 * `h' = -1/2k` the last two terms are `(2h-1)/2` and `(1-2h)/2` and cancel
 * identically. `∂f/∂a = h`, `∂f/∂b = 1-h`: the blend weight is the whole
 * derivative, so this is exact and not a smoothed approximation of the seam.
 */
function smoothMinimumSample(a: ClusterSample, b: ClusterSample, k: number): ClusterSample {
  if (!(k > 0)) return a.distance_m <= b.distance_m ? a : b;
  const h = Math.min(1, Math.max(0, 0.5 + (0.5 * (b.distance_m - a.distance_m)) / k));
  return {
    distance_m: b.distance_m + (a.distance_m - b.distance_m) * h - k * h * (1 - h),
    gradient: {
      x: b.gradient.x + (a.gradient.x - b.gradient.x) * h,
      y: b.gradient.y + (a.gradient.y - b.gradient.y) * h,
      z: b.gradient.z + (a.gradient.z - b.gradient.z) * h,
    },
  };
}

/** A degenerate offset yields a zero gradient, never a guessed direction. */
function unitGradient(offset: Vec3): Vec3 {
  const length = Math.hypot(offset.x, offset.y, offset.z);
  return length > NORMAL_EPSILON
    ? { x: offset.x / length, y: offset.y / length, z: offset.z / length }
    : { x: 0, y: 0, z: 0 };
}

function clusterSphereSample(
  point: Vec3,
  cellX: number,
  cellY: number,
  cellZ: number,
  octave: ClusterLatticeOctave,
): ClusterSample {
  const period = octave.latticePeriod_m;
  const jitter = clusterCellJitter(cellX, cellY, cellZ, octave.seed);
  const offset = {
    x: point.x - (cellX + 0.5 + octave.jitter * jitter.x) * period,
    y: point.y - (cellY + 0.5 + octave.jitter * jitter.y) * period,
    z: point.z - (cellZ + 0.5 + octave.jitter * jitter.z) * period,
  };
  return {
    distance_m: Math.hypot(offset.x, offset.y, offset.z) - octave.latticeLobeRadius_m,
    gradient: unitGradient(offset),
  };
}

/** {@link clusterLatticeOctaveDistance}, differentiated in the same fold order. */
function clusterLatticeOctaveSample(point: Vec3, octave: ClusterLatticeOctave): ClusterSample {
  const period = octave.latticePeriod_m;
  const span = SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD;
  const ring = span / 2 - 1;
  const baseX = Math.floor(point.x / period - 0.5) - ring;
  const baseY = Math.floor(point.y / period - 0.5) - ring;
  const baseZ = Math.floor(point.z / period - 0.5) - ring;
  let sample: ClusterSample = { distance_m: 0, gradient: { x: 0, y: 0, z: 0 } };
  for (let corner = 0; corner < 8; corner += 1) {
    const sphere = clusterSphereSample(
      point, baseX + ring + (corner & 1), baseY + ring + ((corner >> 1) & 1), baseZ + ring + ((corner >> 2) & 1), octave,
    );
    sample = corner === 0 ? sphere : smoothMinimumSample(sample, sphere, octave.smoothRadius_m);
  }
  const reach = octave.jitter * period * Math.sqrt(3) + octave.latticeLobeRadius_m + octave.smoothRadius_m;
  for (let index = 0; index < span * span * span; index += 1) {
    const stepX = index % span;
    const stepY = Math.floor(index / span) % span;
    const stepZ = Math.floor(index / (span * span)) % span;
    if (stepX >= ring && stepX <= ring + 1 && stepY >= ring && stepY <= ring + 1 && stepZ >= ring && stepZ <= ring + 1) continue;
    const cellX = baseX + stepX, cellY = baseY + stepY, cellZ = baseZ + stepZ;
    const nominal = Math.hypot(
      point.x - (cellX + 0.5) * period,
      point.y - (cellY + 0.5) * period,
      point.z - (cellZ + 0.5) * period,
    );
    if (nominal - reach > sample.distance_m) continue;
    sample = smoothMinimumSample(sample, clusterSphereSample(point, cellX, cellY, cellZ, octave), octave.smoothRadius_m);
  }
  return sample;
}

function clusterLatticeSample(point: Vec3, packing: SvoClusterLatticeField): ClusterSample {
  const octaves = packing.octaves ?? 1;
  let sample: ClusterSample = { distance_m: 0, gradient: { x: 0, y: 0, z: 0 } };
  for (let octave = 0; octave < octaves; octave += 1) {
    const scale = 2 ** -octave;
    const smoothRadius_m = packing.smoothRadius_m * scale;
    const octaveSample = clusterLatticeOctaveSample(point, {
      latticeLobeRadius_m: packing.latticeLobeRadius_m * scale,
      latticePeriod_m: packing.latticePeriod_m * scale,
      jitter: packing.jitter,
      smoothRadius_m,
      seed: clusterOctaveSeed(packing.seed, octave),
    });
    sample = octave === 0 ? octaveSample : smoothMinimumSample(sample, octaveSample, smoothRadius_m);
  }
  return sample;
}

/**
 * {@link clusterRoundConeDistance}, differentiated branch for branch.
 *
 * Two of the three branches are spheres wearing the segment's algebra:
 * `P + b²A` is `A²|p - to|²` and `P + l²A` is `A²|p - from|²` identically, so
 * those are `|p - c| - r` and their gradients are radial. The side branch has
 * gradient magnitude `√((L + taper²)/A) = 1` by construction — the same
 * identity that makes the distance Lipschitz-1.
 */
function clusterRoundConeSample(
  point: Vec3,
  from: Vec3,
  to: Vec3,
  fromRadius_m: number,
  toRadius_m: number,
): ClusterSample {
  const axis = { x: to.x - from.x, y: to.y - from.y, z: to.z - from.z };
  const axisSquared = axis.x ** 2 + axis.y ** 2 + axis.z ** 2;
  const taper = fromRadius_m - toRadius_m;
  const lateralSquared = axisSquared - taper * taper;
  const inverseAxisSquared = 1 / axisSquared;
  const offset = { x: point.x - from.x, y: point.y - from.y, z: point.z - from.z };
  const along = offset.x * axis.x + offset.y * axis.y + offset.z * axis.z;
  const beyond = along - axisSquared;
  const perpendicular = {
    x: offset.x * axisSquared - axis.x * along,
    y: offset.y * axisSquared - axis.y * along,
    z: offset.z * axisSquared - axis.z * along,
  };
  const perpendicularSquared = perpendicular.x ** 2 + perpendicular.y ** 2 + perpendicular.z ** 2;
  const alongSquared = along * along * axisSquared;
  const beyondSquared = beyond * beyond * axisSquared;
  const split = Math.sign(taper) * taper * taper * perpendicularSquared;
  if (Math.sign(beyond) * lateralSquared * beyondSquared > split) {
    return {
      distance_m: Math.sqrt(perpendicularSquared + beyondSquared) * inverseAxisSquared - toRadius_m,
      gradient: unitGradient({ x: point.x - to.x, y: point.y - to.y, z: point.z - to.z }),
    };
  }
  if (Math.sign(along) * lateralSquared * alongSquared < split) {
    return {
      distance_m: Math.sqrt(perpendicularSquared + alongSquared) * inverseAxisSquared - fromRadius_m,
      gradient: unitGradient(offset),
    };
  }
  const outward = unitGradient(perpendicular);
  const lateral = Math.sqrt(Math.max(0, lateralSquared * inverseAxisSquared));
  return {
    distance_m: (Math.sqrt(perpendicularSquared * lateralSquared * inverseAxisSquared) + along * taper)
      * inverseAxisSquared - fromRadius_m,
    gradient: {
      x: outward.x * lateral + axis.x * taper * inverseAxisSquared,
      y: outward.y * lateral + axis.y * taper * inverseAxisSquared,
      z: outward.z * lateral + axis.z * taper * inverseAxisSquared,
    },
  };
}

function clusterTaperedSweepSample(point: Vec3, packing: SvoClusterTaperedSweepField): ClusterSample {
  let sample: ClusterSample = { distance_m: 0, gradient: { x: 0, y: 0, z: 0 } };
  for (let index = 0; index + 1 < packing.points.length; index += 1) {
    const from = packing.points[index];
    const to = packing.points[index + 1];
    const segment = clusterRoundConeSample(point, from.position_m, to.position_m, from.radius_m, to.radius_m);
    sample = index === 0 ? segment : smoothMinimumSample(sample, segment, packing.smoothRadius_m);
  }
  return sample;
}

/**
 * One seeded lobe, differentiated through the frames it is authored in.
 *
 * `scaled = |R⁻¹(p/r - c) / k|`, so the chain is the ellipsoid gradient in lobe
 * space, rotated back by the lobe's orientation, then divided by the envelope
 * radii the normalised space was built with.
 */
function clusterSeededLobesSample(
  point: Vec3,
  lobeRadii_m: Vec3,
  packing: SvoClusterSeededLobesField,
): ClusterSample {
  const shortest_m = Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
  const normalized = { x: point.x / lobeRadii_m.x, y: point.y / lobeRadii_m.y, z: point.z / lobeRadii_m.z };
  let sample: ClusterSample = { distance_m: 0, gradient: { x: 0, y: 0, z: 0 } };
  for (let index = 0; index < packing.lobeCount; index += 1) {
    const lobe = clusterSeededLobe(index, lobeRadii_m, packing);
    const local = inverseRotate(lobe.orientation, {
      x: normalized.x - lobe.center.x,
      y: normalized.y - lobe.center.y,
      z: normalized.z - lobe.center.z,
    });
    const scaledVector = { x: local.x / lobe.halfAxes.x, y: local.y / lobe.halfAxes.y, z: local.z / lobe.halfAxes.z };
    const scaled = Math.hypot(scaledVector.x, scaledVector.y, scaledVector.z);
    const span = shortest_m * Math.min(lobe.halfAxes.x, lobe.halfAxes.y, lobe.halfAxes.z);
    const localGradient = scaled > NORMAL_EPSILON
      ? {
        x: scaledVector.x / lobe.halfAxes.x / scaled,
        y: scaledVector.y / lobe.halfAxes.y / scaled,
        z: scaledVector.z / lobe.halfAxes.z / scaled,
      }
      : { x: 0, y: 0, z: 0 };
    const world = rotate(lobe.orientation, localGradient);
    const lobeSample: ClusterSample = {
      distance_m: (scaled - 1) * span,
      gradient: {
        x: (world.x * span) / lobeRadii_m.x,
        y: (world.y * span) / lobeRadii_m.y,
        z: (world.z * span) / lobeRadii_m.z,
      },
    };
    sample = index === 0 ? lobeSample : smoothMinimumSample(sample, lobeSample, packing.smoothRadius_m);
  }
  return sample;
}

/** The envelope every field is clipped to, and its gradient. */
function clusterLobeSample(point: Vec3, lobeRadii_m: Vec3): ClusterSample {
  const shortest_m = Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
  const normalized = { x: point.x / lobeRadii_m.x, y: point.y / lobeRadii_m.y, z: point.z / lobeRadii_m.z };
  const magnitude = Math.hypot(normalized.x, normalized.y, normalized.z);
  const gradient = magnitude > NORMAL_EPSILON
    ? {
      x: (normalized.x / lobeRadii_m.x / magnitude) * shortest_m,
      y: (normalized.y / lobeRadii_m.y / magnitude) * shortest_m,
      z: (normalized.z / lobeRadii_m.z / magnitude) * shortest_m,
    }
    : { x: 0, y: 0, z: 0 };
  return { distance_m: (magnitude - 1) * shortest_m, gradient };
}

/** Density fields do not have analytic lobe normals; differentiate the same
 * bounded distance used by the march over a fraction of the detail period. */
function clusterNoiseFoliageSample(
  point: Vec3,
  lobeRadii_m: Vec3,
  packing: SvoClusterNoiseFoliageField,
): ClusterSample {
  const distance_m = clusterNoiseFoliageDistance(point, lobeRadii_m, packing);
  const h = Math.max(1e-6, packing.detailPeriod_m * 0.001);
  const at = (x: number, y: number, z: number): number => clusterNoiseFoliageDistance({ x, y, z }, lobeRadii_m, packing);
  return {
    distance_m,
    gradient: {
      x: (at(point.x + h, point.y, point.z) - at(point.x - h, point.y, point.z)) / (2 * h),
      y: (at(point.x, point.y + h, point.z) - at(point.x, point.y - h, point.z)) / (2 * h),
      z: (at(point.x, point.y, point.z + h) - at(point.x, point.y, point.z - h)) / (2 * h),
    },
  };
}

function clusterFieldSample(point: Vec3, lobeRadii_m: Vec3, packing: SvoSmoothUnionClusterPacking): ClusterSample {
  if (packing.field === "noise-foliage") return clusterNoiseFoliageSample(point, lobeRadii_m, packing);
  if (packing.field === "seeded-lobes") return clusterSeededLobesSample(point, lobeRadii_m, packing);
  if (packing.field === "tapered-sweep") return clusterTaperedSweepSample(point, packing);
  return clusterLatticeSample(point, packing);
}

/** {@link clusterDistance} and its gradient: a `max` passes the winner through. */
function clusterSample(point: Vec3, lobeRadii_m: Vec3, packing: SvoSmoothUnionClusterPacking): ClusterSample {
  const field = clusterFieldSample(point, lobeRadii_m, packing);
  const envelope = clusterLobeSample(point, lobeRadii_m);
  return field.distance_m >= envelope.distance_m ? field : envelope;
}

/**
 * The cluster's shading normal — one differentiated pass, not four evaluations.
 *
 * This was a four-tap tetrahedral difference, and its own comment priced a tap
 * at "roughly sixty sphere distances". That was the frame: on
 * `hero-garden-hose` at refinement depth 3, deleting *only* the cluster
 * gradients took the 800x460 frame from 42.7 ms to 19.1 ms — 55 % of the whole
 * frame in this one function, against 4 % for the field-program tapes beside it
 * and nothing measurable for the ground or the stencil fallback.
 *
 * Every leaf of the fold already knew its own normal. Carrying it through the
 * blend costs one lerp per combination, deletes three quarters of the
 * evaluations. Noise foliage alone takes a local central difference because its
 * trilinear density has no lobe normal to carry through the fold.
 */
function clusterLocalNormal(point: Vec3, lobeRadii_m: Vec3, packing: SvoSmoothUnionClusterPacking): Vec3 | null {
  return normalize(clusterSample(point, lobeRadii_m, packing).gradient);
}

/** A cluster whose arena block has been resolved, which is the only form that can be evaluated. */
type ResolvedCluster = SvoSmoothUnionClusterPrimitive & { packing: SvoSmoothUnionClusterPacking };

/** A field program whose tape has been resolved, which is the only form that can be evaluated. */
type ResolvedFieldProgram = SvoFieldProgramPrimitive & { program: SvoFieldProgram };

type MarchedPrimitive = SvoTorusPrimitive | SvoConePrimitive | SvoRoundConePrimitive | SvoRoundedCylinderPrimitive
  | ResolvedCluster | ResolvedFieldProgram;

/**
 * A field program's distance, already divided by the Lipschitz constant the
 * evaluator returned beside it.
 *
 * **Do not "simplify" this back to `value.distance_m`.** A domain warp evaluates
 * its subtree at a moved point, so the composed field is `L`-Lipschitz rather
 * than 1-Lipschitz and `|g(p)|` *overestimates* the clearance by up to `L`.
 * Every consumer in this ABI — the march loop's step, the voxelizer's occupancy
 * test, the cell-coverage corner sweep — is built on the one contract that a
 * kind's distance is a **lower bound** on the true clearance, and each of them
 * breaks differently when it is not: the trace steps through the surface and
 * punches holes in every warped shape (§2.4 of `docs/hero-fidelity-1000x-
 * handoff.md`), and voxelization reports empty cells that hold solid.
 *
 * Dividing here rather than threading the constant out to each of those call
 * sites is deliberate. It costs nothing — `L` is a property of the tape, not of
 * the point, so the division is a uniform positive scale that leaves the sign
 * and the zero set exactly where they were — and it means a field program is
 * simply another 1-Lipschitz lower bound to everything downstream, with no new
 * rule for a later reader to know about and forget. The only visible effect is
 * that the march's acceptance band stops it up to `L` times its own epsilon
 * outside the surface, which is the safe side of the surface and is far below
 * the band the other marched kinds already accept.
 */
function fieldProgramDistance_m(program: SvoFieldProgram, localPoint: Vec3): number {
  const value = evaluateSvoFieldProgram(program, localPoint);
  return value.distance_m / Math.max(1, value.lipschitz);
}

/**
 * Central-difference step for a field program's shading normal.
 *
 * Scaled to the record's own extent because a tape has no analytic gradient and
 * the field's finest feature is a fraction of the shape, not an absolute size.
 * The absolute floor keeps a millimetre-scale record off the f32 noise shelf.
 */
function fieldProgramNormalStep_m(extent_m: Vec3): number {
  return Math.max(1e-5, 1e-3 * Math.max(extent_m.x, extent_m.y, extent_m.z));
}

function fieldProgramLocalNormal(descriptor: ResolvedFieldProgram, point: Vec3, extent_m: Vec3): Vec3 | null {
  const step = fieldProgramNormalStep_m(extent_m);
  const at = (dx: number, dy: number, dz: number): number =>
    fieldProgramDistance_m(descriptor.program, { x: point.x + dx, y: point.y + dy, z: point.z + dz });
  return normalize({
    x: at(step, 0, 0) - at(-step, 0, 0),
    y: at(0, step, 0) - at(0, -step, 0),
    z: at(0, 0, step) - at(0, 0, -step),
  });
}

function roundedCylinderLocalSample(descriptor: SvoRoundedCylinderPrimitive, point: Vec3): SvoPrimitiveSample {
  const radial = Math.hypot(point.x, point.z);
  const radialCore = descriptor.radius_m - descriptor.edgeRadius_m;
  const verticalCore = descriptor.halfHeight_m - descriptor.edgeRadius_m;
  const q = { x: radial - radialCore, y: Math.abs(point.y) - verticalCore };
  const outside = { x: Math.max(q.x, 0), y: Math.max(q.y, 0) };
  const signedDistance_m = Math.hypot(outside.x, outside.y) + Math.min(Math.max(q.x, q.y), 0) - descriptor.edgeRadius_m;
  const direction = radial > NORMAL_EPSILON ? { x: point.x / radial, z: point.z / radial } : { x: 1, z: 0 };
  let normal: Vec3;
  if (q.x > 0 && q.y > 0) {
    const magnitude = Math.hypot(q.x, q.y);
    normal = { x: direction.x * q.x / magnitude, y: Math.sign(point.y || 1) * q.y / magnitude, z: direction.z * q.x / magnitude };
  } else if (q.x > q.y) {
    normal = { x: direction.x, y: 0, z: direction.z };
  } else {
    normal = { x: 0, y: Math.sign(point.y || 1), z: 0 };
  }
  return { signedDistance_m, normal, featureId: SVO_PRIMITIVE_FEATURES.smooth };
}

function roundConeLocalSample(descriptor: SvoRoundConePrimitive, point: Vec3): SvoPrimitiveSample {
  // This is the local-Y specialization of `clusterRoundConeDistance`. Besides
  // being cheaper during a march, its region test also gives the exact normal:
  // endpoint-sphere radial in either cap region, constant cone gradient in the
  // tangent band. A finite-difference normal made every coping hit pay four
  // more SDF evaluations and introduced avoidable sub-pixel normal jitter.
  const { baseRadius_m: base, topRadius_m: top, halfHeight_m: half } = descriptor;
  const span = 2 * half;
  const taper = (base - top) / span;
  const lateral = Math.sqrt(Math.max(0, 1 - taper * taper));
  const radial = Math.hypot(point.x, point.z);
  const along = point.y + half;
  const region = -taper * radial + lateral * along;
  let signedDistance_m: number;
  let normal: Vec3 | null;
  if (region < 0) {
    const offset = { x: point.x, y: along, z: point.z };
    signedDistance_m = Math.hypot(offset.x, offset.y, offset.z) - base;
    normal = normalize(offset);
  } else if (region > lateral * span) {
    const offset = { x: point.x, y: along - span, z: point.z };
    signedDistance_m = Math.hypot(offset.x, offset.y, offset.z) - top;
    normal = normalize(offset);
  } else {
    signedDistance_m = lateral * radial + taper * along - base;
    const direction = radial > NORMAL_EPSILON ? { x: point.x / radial, z: point.z / radial } : { x: 1, z: 0 };
    normal = { x: lateral * direction.x, y: taper, z: lateral * direction.z };
  }
  return { signedDistance_m, normal, featureId: SVO_PRIMITIVE_FEATURES.smooth };
}

/**
 * Local signed distance, shading normal and feature of the marched kinds.
 *
 * The torus and cone distances are exact, which is what lets one sphere trace
 * stand in for a bespoke root solve per shape. The cluster's is a strict lower
 * bound, which serves the same trace for the same reason — the march only needs
 * a guarantee that it never steps past a surface.
 */
function marchedLocalSample(descriptor: MarchedPrimitive, point: Vec3): SvoPrimitiveSample {
  if (descriptor.kind === "field-program") {
    const extent_m = svoPrimitiveLocalExtent_m(descriptor);
    return {
      signedDistance_m: fieldProgramDistance_m(descriptor.program, point),
      normal: fieldProgramLocalNormal(descriptor, point, extent_m),
      featureId: SVO_PRIMITIVE_FEATURES.smooth,
    };
  }
  if (descriptor.kind === "smooth-union-cluster") {
    return {
      signedDistance_m: clusterDistance(point, descriptor.lobeRadii_m, descriptor.packing),
      normal: clusterLocalNormal(point, descriptor.lobeRadii_m, descriptor.packing),
      featureId: SVO_PRIMITIVE_FEATURES.smooth,
    };
  }
  if (descriptor.kind === "torus") {
    const radial = Math.hypot(point.x, point.z);
    const ring = radial - descriptor.majorRadius_m;
    const signedDistance_m = Math.hypot(ring, point.y) - descriptor.minorRadius_m;
    // On the axis the whole ring is equidistant, so no single normal exists.
    const scale = radial > NORMAL_EPSILON ? ring / radial : 0;
    const normal = radial > NORMAL_EPSILON
      ? normalize({ x: point.x * scale, y: point.y, z: point.z * scale })
      : null;
    return { signedDistance_m, normal, featureId: SVO_PRIMITIVE_FEATURES.smooth };
  }
  if (descriptor.kind === "rounded-cylinder") return roundedCylinderLocalSample(descriptor, point);
  if (descriptor.kind === "round-cone") return roundConeLocalSample(descriptor, point);
  const { baseRadius_m: base, topRadius_m: top, halfHeight_m: half } = descriptor;
  const radial = Math.hypot(point.x, point.z);
  // Exact frustum distance: nearest of the end disc (`cap`) and the lateral
  // band (`side`), signed by whether the point is inside both.
  const capRadius = point.y < 0 ? base : top;
  const cap = { x: radial - Math.min(radial, capRadius), y: Math.abs(point.y) - half };
  const slope = { x: top - base, y: 2 * half };
  const toApex = { x: top - radial, y: half - point.y };
  const projection = Math.min(1, Math.max(0, (toApex.x * slope.x + toApex.y * slope.y) / (slope.x ** 2 + slope.y ** 2)));
  const side = { x: radial - top + slope.x * projection, y: point.y - half + slope.y * projection };
  const capSquared = cap.x ** 2 + cap.y ** 2;
  const sideSquared = side.x ** 2 + side.y ** 2;
  const inside = side.x < 0 && cap.y < 0;
  const signedDistance_m = (inside ? -1 : 1) * Math.sqrt(Math.min(capSquared, sideSquared));
  // Hard features: select the authored end or the authored band, never a blend.
  if (capSquared <= sideSquared) {
    return {
      signedDistance_m,
      normal: { x: 0, y: Math.sign(point.y || 1), z: 0 },
      featureId: SVO_PRIMITIVE_FEATURES.cylinderCap,
    };
  }
  const lateral = normalize({ x: slope.y, y: base - top, z: 0 })!;
  const direction = radial > NORMAL_EPSILON ? { x: point.x / radial, z: point.z / radial } : { x: 1, z: 0 };
  return {
    signedDistance_m,
    normal: { x: direction.x * lateral.x, y: lateral.y, z: direction.z * lateral.x },
    featureId: SVO_PRIMITIVE_FEATURES.cylinderSide,
  };
}

/** Rotation-invariant radius that bounds a marched kind about its own centre. */
function marchedBoundingRadius(descriptor: MarchedPrimitive): number {
  return svoPrimitiveBoundingRadius_m(descriptor);
}

/** Surface acceptance band, shared with WGSL so both solves stop in the same place. */
function marchSurfaceEpsilon(t_m: number): number {
  return Math.max(1e-6, 1e-4 * Math.abs(t_m));
}

/**
 * Bounded sphere trace against an exact distance. Stepping by |distance| walks
 * to the surface from either side, so a ray that starts inside reports the same
 * far surface a closed-form solve would.
 */
function intersectMarchedLocal(
  descriptor: MarchedPrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  // The bounding sphere both rejects misses early and closes an unbounded tMax.
  const bounds = quadraticRoots(
    direction.x ** 2 + direction.y ** 2 + direction.z ** 2,
    origin.x * direction.x + origin.y * direction.y + origin.z * direction.z,
    origin.x ** 2 + origin.y ** 2 + origin.z ** 2 - marchedBoundingRadius(descriptor) ** 2,
  );
  if (bounds.length < 2) return null;
  const start = Math.max(ray.tMin_m, bounds[0]);
  const end = Math.min(ray.tMax_m, bounds[1]);
  if (!(end >= start)) return null;
  let t_m = start;
  for (let iteration = 0; iteration <= SVO_PRIMITIVE_MARCH_ITERATIONS; iteration += 1) {
    const sample = marchedLocalSample(descriptor, {
      x: origin.x + direction.x * t_m,
      y: origin.y + direction.y * t_m,
      z: origin.z + direction.z * t_m,
    });
    const distance = Math.abs(sample.signedDistance_m);
    if (distance <= marchSurfaceEpsilon(t_m)) {
      return sample.normal ? { t_m, normal: sample.normal, featureId: sample.featureId } : null;
    }
    // The floor only guarantees progress; the acceptance band above is what
    // decides a hit, so it can never step across a surface it should have found.
    t_m += Math.max(distance, NORMAL_EPSILON);
    if (t_m > end) return null;
  }
  return null;
}

function intersectCanonicalSvoPrimitive(
  descriptor: SvoFinitePrimitiveDescriptor,
  ray: CanonicalPrimitiveRay,
  clusterResolver?: SvoClusterResolver,
  fieldProgramResolver?: SvoFieldProgramResolver,
): SvoPrimitiveRayHit | null {
  const { origin, direction, orientation } = localPrimitiveRay(descriptor, ray);
  const localHit = descriptor.kind === "sphere" ? intersectSphereLocal(descriptor, origin, direction, ray)
    : descriptor.kind === "box" ? intersectBoxLocal(descriptor, origin, direction, ray)
      : descriptor.kind === "capsule" ? intersectCapsuleLocal(descriptor, origin, direction, ray)
        : descriptor.kind === "cylinder" ? intersectCylinderLocal(descriptor, origin, direction, ray)
          : descriptor.kind === "smooth-union-cluster"
            ? intersectMarchedLocal(resolveCluster(descriptor, clusterResolver), origin, direction, ray)
            : descriptor.kind === "field-program"
              ? intersectMarchedLocal(resolveFieldProgram(descriptor, fieldProgramResolver), origin, direction, ray)
              : descriptor.kind === "torus" || descriptor.kind === "cone" || descriptor.kind === "round-cone" || descriptor.kind === "rounded-cylinder"
                ? intersectMarchedLocal(descriptor, origin, direction, ray)
                : intersectEllipsoidLocal(descriptor, origin, direction, ray);
  if (!localHit) return null;
  const normal = worldNormal(orientation, localHit.normal);
  if (!normal) return null;
  return {
    t_m: localHit.t_m,
    position_m: {
      x: ray.origin_m.x + ray.direction.x * localHit.t_m,
      y: ray.origin_m.y + ray.direction.y * localHit.t_m,
      z: ray.origin_m.z + ray.direction.z * localHit.t_m,
    },
    normal,
    normalPolicy: SVO_PRIMITIVE_KIND_TABLE[descriptor.kind].normalPolicy,
    featureId: localHit.featureId,
    primitiveKind: descriptor.kind,
    primitiveId: descriptor.primitiveId,
    materialId: descriptor.materialId,
    ownerId: descriptor.ownerId ?? SPARSE_BRICK_NO_OWNER,
  };
}

/** Exact analytic finite-primitive hit oracle. Terrain is intentionally handled by its separate heightfield tracer. */
export function intersectSvoPrimitive(
  input: SvoFinitePrimitiveDescriptor,
  rayInput: SvoPrimitiveRay,
  clusterResolver?: SvoClusterResolver,
  fieldProgramResolver?: SvoFieldProgramResolver,
): SvoPrimitiveRayHit | null {
  const descriptor = canonicalSvoPrimitive(input);
  if (descriptor.kind === "terrain-heightfield") throw new TypeError("Terrain heightfield intersection uses the separate terrain tracer");
  return intersectCanonicalSvoPrimitive(descriptor, canonicalPrimitiveRay(rayInput), clusterResolver, fieldProgramResolver);
}

/** Nearest exact finite-primitive hit. Input order is the deterministic tie-breaker. */
export function intersectSvoPrimitives(
  inputs: readonly SvoPrimitiveDescriptor[],
  rayInput: SvoPrimitiveRay,
  clusterResolver?: SvoClusterResolver,
  fieldProgramResolver?: SvoFieldProgramResolver,
): SvoPrimitiveRayHit | null {
  const ray = canonicalPrimitiveRay(rayInput);
  let nearest: SvoPrimitiveRayHit | null = null;
  for (const input of inputs) {
    const descriptor = canonicalSvoPrimitive(input);
    if (descriptor.kind === "terrain-heightfield") continue;
    const hit = intersectCanonicalSvoPrimitive(descriptor, ray, clusterResolver, fieldProgramResolver);
    if (hit && (!nearest || hit.t_m < nearest.t_m)) nearest = hit;
  }
  return nearest;
}

/** Unpack the stable 64-byte ABI and select its nearest finite analytic hit. */
export function intersectPackedSvoPrimitiveRecords(
  packed: Uint32Array,
  ray: SvoPrimitiveRay,
  clusterResolver?: SvoClusterResolver,
  fieldProgramResolver?: SvoFieldProgramResolver,
): SvoPrimitiveRayHit | null {
  return intersectSvoPrimitives(unpackSvoPrimitiveRecords(packed), ray, clusterResolver, fieldProgramResolver);
}

interface SvoEllipsoidClosestPoint {
  point: Vec3;
  ambiguous: boolean;
}

function ellipsoidClosestEquation(extents: readonly number[], point: readonly number[], lambda: number): number {
  let sum = 0;
  for (let axis = 0; axis < extents.length; axis += 1) {
    const extentSquared = extents[axis] ** 2;
    const ratio = extents[axis] * point[axis] / (extentSquared + lambda);
    sum += ratio * ratio;
  }
  return sum - 1;
}

/**
 * Exact Euclidean closest point in the positive orthant. The active-axis
 * reduction handles interior medial-axis singularities; every root solve uses
 * the fixed public bisection ceiling mirrored by WGSL.
 */
function closestEllipsoidPositive(extents: readonly number[], point: readonly number[]): { point: number[]; ambiguous: boolean } {
  if (extents.length === 1) return { point: [extents[0]], ambiguous: point[0] <= NORMAL_EPSILON };
  const last = extents.length - 1;
  const smallestExtent = extents[last];
  const coordinateTolerance = NORMAL_EPSILON * Math.max(1, extents[0]);
  if (point[last] <= coordinateTolerance) {
    const candidate = new Array<number>(extents.length).fill(0);
    let surfaceSum = 0;
    let valid = true;
    for (let axis = 0; axis < last; axis += 1) {
      const denominator = extents[axis] ** 2 - smallestExtent ** 2;
      if (Math.abs(denominator) <= NORMAL_EPSILON * Math.max(1, extents[axis] ** 2)) {
        if (point[axis] > coordinateTolerance) valid = false;
        continue;
      }
      candidate[axis] = extents[axis] ** 2 * point[axis] / denominator;
      surfaceSum += (candidate[axis] / extents[axis]) ** 2;
    }
    if (valid && surfaceSum <= 1 + 32 * Number.EPSILON) {
      candidate[last] = smallestExtent * Math.sqrt(Math.max(0, 1 - surfaceSum));
      return { point: candidate, ambiguous: candidate[last] > coordinateTolerance };
    }
    const reduced = closestEllipsoidPositive(extents.slice(0, last), point.slice(0, last));
    return { point: [...reduced.point, 0], ambiguous: reduced.ambiguous };
  }

  const equationAtZero = ellipsoidClosestEquation(extents, point, 0);
  if (Math.abs(equationAtZero) <= 32 * Number.EPSILON) return { point: [...point], ambiguous: false };
  let lower = equationAtZero < 0 ? -(smallestExtent ** 2) : 0;
  let upper = equationAtZero < 0 ? 0 : Math.max(1, extents[0] * Math.hypot(...point));
  for (let iteration = 0; iteration < SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS; iteration += 1) {
    const middle = 0.5 * (lower + upper);
    if (ellipsoidClosestEquation(extents, point, middle) > 0) lower = middle;
    else upper = middle;
  }
  const lambda = 0.5 * (lower + upper);
  return {
    point: extents.map((extent, axis) => extent ** 2 * point[axis] / (extent ** 2 + lambda)),
    ambiguous: false,
  };
}

function closestEllipsoidPoint(radii: Vec3, point: Vec3): SvoEllipsoidClosestPoint {
  const extents = [radii.x, radii.y, radii.z];
  const coordinates = [point.x, point.y, point.z];
  const axes = [0, 1, 2].sort((a, b) => extents[b] - extents[a]);
  const sortedExtents = axes.map((axis) => extents[axis]);
  const positivePoint = axes.map((axis) => Math.abs(coordinates[axis]));
  const result = closestEllipsoidPositive(sortedExtents, positivePoint);
  const closest = [0, 0, 0];
  for (let sortedAxis = 0; sortedAxis < axes.length; sortedAxis += 1) {
    const originalAxis = axes[sortedAxis];
    closest[originalAxis] = Math.sign(coordinates[originalAxis] || 1) * result.point[sortedAxis];
  }
  return { point: { x: closest[0], y: closest[1], z: closest[2] }, ambiguous: result.ambiguous };
}

/** CPU numerical mirror of the WGSL evaluation and hard-feature normal policy. */
export function sampleSvoPrimitive(
  input: SvoPrimitiveDescriptor,
  worldPoint_m: Vec3,
  terrainResolver?: SvoTerrainResolver,
  clusterResolver?: SvoClusterResolver,
  fieldProgramResolver?: SvoFieldProgramResolver,
): SvoPrimitiveSample {
  const descriptor = canonicalSvoPrimitive(input);
  if (descriptor.kind === "terrain-heightfield") {
    if (!terrainResolver) throw new Error("Terrain primitive evaluation requires a terrain resolver");
    finiteVec3(worldPoint_m, "Primitive query point");
    const terrain = terrainResolver(descriptor.terrainReference);
    return {
      signedDistance_m: worldPoint_m.y - terrainHeightAt(terrain, worldPoint_m.x, worldPoint_m.z),
      normal: terrainNormalAt(terrain, worldPoint_m.x, worldPoint_m.z, descriptor.normalEpsilon_m),
      featureId: SVO_PRIMITIVE_FEATURES.terrain,
    };
  }
  const { point, orientation } = localPoint(descriptor, worldPoint_m);
  if (descriptor.kind === "sphere") {
    const length = Math.hypot(point.x, point.y, point.z);
    return { signedDistance_m: length - descriptor.radius_m, normal: worldNormal(orientation, normalize(point)), featureId: SVO_PRIMITIVE_FEATURES.smooth };
  }
  if (descriptor.kind === "box") {
    const q = {
      x: Math.abs(point.x) - descriptor.halfExtents_m.x,
      y: Math.abs(point.y) - descriptor.halfExtents_m.y,
      z: Math.abs(point.z) - descriptor.halfExtents_m.z,
    };
    const outside = { x: Math.max(q.x, 0), y: Math.max(q.y, 0), z: Math.max(q.z, 0) };
    const signedDistance_m = Math.hypot(outside.x, outside.y, outside.z) + Math.min(Math.max(q.x, q.y, q.z), 0);
    // Select exactly one authored face. Ties are stable X -> Y -> Z and never
    // average normals across a sharp edge or corner.
    let axis: "x" | "y" | "z" = "x";
    if (q.y > q.x) axis = "y";
    if (q.z > q[axis]) axis = "z";
    const localNormal = { x: axis === "x" ? Math.sign(point.x || 1) : 0, y: axis === "y" ? Math.sign(point.y || 1) : 0, z: axis === "z" ? Math.sign(point.z || 1) : 0 };
    const featureId = axis === "x" ? SVO_PRIMITIVE_FEATURES.boxFaceX : axis === "y" ? SVO_PRIMITIVE_FEATURES.boxFaceY : SVO_PRIMITIVE_FEATURES.boxFaceZ;
    return { signedDistance_m, normal: worldNormal(orientation, localNormal), featureId };
  }
  if (descriptor.kind === "capsule") {
    const segmentY = Math.max(-descriptor.segmentHalfLength_m, Math.min(descriptor.segmentHalfLength_m, point.y));
    const offset = { x: point.x, y: point.y - segmentY, z: point.z };
    return {
      signedDistance_m: Math.hypot(offset.x, offset.y, offset.z) - descriptor.radius_m,
      normal: worldNormal(orientation, normalize(offset)), featureId: SVO_PRIMITIVE_FEATURES.smooth,
    };
  }
  if (descriptor.kind === "torus" || descriptor.kind === "cone" || descriptor.kind === "round-cone" || descriptor.kind === "rounded-cylinder") {
    const local = marchedLocalSample(descriptor, point);
    return { ...local, normal: worldNormal(orientation, local.normal) };
  }
  if (descriptor.kind === "smooth-union-cluster") {
    const local = marchedLocalSample(resolveCluster(descriptor, clusterResolver), point);
    return { ...local, normal: worldNormal(orientation, local.normal) };
  }
  if (descriptor.kind === "field-program") {
    const local = marchedLocalSample(resolveFieldProgram(descriptor, fieldProgramResolver), point);
    return { ...local, normal: worldNormal(orientation, local.normal) };
  }
  if (descriptor.kind === "cylinder") {
    const radialLength = Math.hypot(point.x, point.z);
    const radialDistance = radialLength - descriptor.radius_m;
    const capDistance = Math.abs(point.y) - descriptor.halfHeight_m;
    const signedDistance_m = Math.hypot(Math.max(radialDistance, 0), Math.max(capDistance, 0)) + Math.min(Math.max(radialDistance, capDistance), 0);
    const capWins = capDistance >= radialDistance;
    const localNormal = capWins
      ? { x: 0, y: Math.sign(point.y || 1), z: 0 }
      : normalize({ x: point.x, y: 0, z: point.z });
    return {
      signedDistance_m, normal: worldNormal(orientation, localNormal),
      featureId: capWins ? SVO_PRIMITIVE_FEATURES.cylinderCap : SVO_PRIMITIVE_FEATURES.cylinderSide,
    };
  }
  const closestResult = closestEllipsoidPoint(descriptor.radii_m, point);
  const delta = {
    x: point.x - closestResult.point.x,
    y: point.y - closestResult.point.y,
    z: point.z - closestResult.point.z,
  };
  const distance = Math.hypot(delta.x, delta.y, delta.z);
  const inside = Math.hypot(
    point.x / descriptor.radii_m.x,
    point.y / descriptor.radii_m.y,
    point.z / descriptor.radii_m.z,
  ) < 1;
  const signedDistance_m = inside ? -distance : distance;
  const onSurface = distance <= NORMAL_EPSILON * Math.max(1, descriptor.radii_m.x, descriptor.radii_m.y, descriptor.radii_m.z);
  const normal = onSurface
    ? normalize({
      x: point.x / descriptor.radii_m.x ** 2,
      y: point.y / descriptor.radii_m.y ** 2,
      z: point.z / descriptor.radii_m.z ** 2,
    })
    : closestResult.ambiguous ? null : normalize(inside
      ? { x: -delta.x, y: -delta.y, z: -delta.z }
      : delta);
  return { signedDistance_m, normal: worldNormal(orientation, normal), featureId: SVO_PRIMITIVE_FEATURES.smooth };
}

/**
 * The one function {@link svoPrimitiveWGSL} does not carry, and every module
 * that includes it must declare *before* it:
 *
 * ```wgsl
 * fn svoFieldProgramReferenceSample(reference: u32, localPoint: vec3f) -> SvoFieldValue
 * ```
 *
 * A field program's tape is a hundred and thirty-two words, so the shared
 * library cannot take it as a parameter the way it takes `SvoClusterPacking` —
 * that struct is already the largest thing threaded through
 * `svoIntersectPrimitiveExact`, and this one is four times its size and would be
 * copied by every primitive of every kind. It also cannot read the arena
 * itself: three shaders include this library against three different bindings,
 * and the one thing that has kept it reusable is that it names no binding.
 *
 * So the tape is read by the host, which is the same bargain `svoFieldProgram-
 * WGSL` already makes by taking a `loadWord`. `reference` is the record's word
 * 13 verbatim, and each host resolves it against whatever arena it owns and by
 * whatever convention it published — the renderer's is a word offset, the live
 * voxelizer's is a slot. A reference the host cannot resolve must answer with a
 * distance of `1e20`, which the ABI reads as "this record did not resolve" and
 * reports as invalid rather than drawing the envelope.
 *
 * This constant is that answer for a module with no field-program arena at all.
 * It is not a fallback inside the library: a module either has the arena and
 * generates the real evaluator, or it does not and says so once, here, where a
 * reader can see which of the two it is.
 */
export const svoFieldProgramAbsentWGSL = /* wgsl */ `
struct SvoFieldValue{distance_m:f32,lipschitz:f32}
fn svoFieldProgramReferenceSample(reference:u32,localPoint:vec3f)->SvoFieldValue{
  return SvoFieldValue(1e20,1.0);
}
`;

/**
 * Shared WGSL declaration/evaluation library. Terrain height and normal are
 * supplied by the scene's existing terrain evaluator using metadata.y as its
 * stable table reference. Box/cylinder normals select one feature; they never
 * average across hard boundaries.
 *
 * Requires the host module to have declared `svoFieldProgramReferenceSample`
 * first — see {@link svoFieldProgramAbsentWGSL} for the contract and for the
 * declaration a module without a tape arena uses.
 */
export const svoPrimitiveWGSL = /* wgsl */ `
${svoPrimitiveKindConstantsWGSL}
const SVO_FEATURE_SMOOTH: u32 = 0u;
const SVO_FEATURE_BOX_X: u32 = 1u;
const SVO_FEATURE_BOX_Y: u32 = 2u;
const SVO_FEATURE_BOX_Z: u32 = 3u;
const SVO_FEATURE_CYLINDER_SIDE: u32 = 4u;
const SVO_FEATURE_CYLINDER_CAP: u32 = 5u;
const SVO_FEATURE_TERRAIN: u32 = 6u;
const SVO_SAMPLE_NORMAL_VALID: u32 = 1u;
const SVO_PRIMITIVE_RAY_MISS: u32 = 0u;
const SVO_PRIMITIVE_RAY_HIT: u32 = 1u;
const SVO_PRIMITIVE_RAY_INVALID: u32 = 2u;
const SVO_PRIMITIVE_RAY_INFINITY: f32 = 3.402823e38;

struct SvoPrimitiveRecord {
  centerKind: vec4u,
  dimensionsIdentity: vec4u,
  orientation: vec4f,
  metadata: vec4u,
}

struct SvoPrimitiveSample {
  signedDistance_m: f32,
  featureId: u32,
  flags: u32,
  _padding: u32,
  normal: vec4f,
}

struct SvoPrimitiveRayResult {
  t_m: f32,
  featureId: u32,
  status: u32,
  _padding: u32,
  normal: vec4f,
}

struct SvoPrimitiveQuadraticRoots {
  values: vec2f,
  count: u32,
  _padding: u32,
}

fn svoPrimitiveCenter_m(record: SvoPrimitiveRecord) -> vec3f { return bitcast<vec3f>(record.centerKind.xyz); }
fn svoPrimitiveKind(record: SvoPrimitiveRecord) -> u32 { return record.centerKind.w; }
fn svoPrimitiveDimensions_m(record: SvoPrimitiveRecord) -> vec3f { return bitcast<vec3f>(record.dimensionsIdentity.xyz); }
fn svoPrimitiveMaterialId(record: SvoPrimitiveRecord) -> u32 { return record.dimensionsIdentity.w & 0xffffu; }
fn svoPrimitiveOwnerId(record: SvoPrimitiveRecord) -> u32 { return record.dimensionsIdentity.w >> 16u; }
fn svoPrimitiveId(record: SvoPrimitiveRecord) -> u32 { return record.metadata.x; }
// One word, three arena-backed kinds. A heightfield names its samples, a
// cluster names its packing and a field program names its tape; each is a block
// in the shared scene arena, and none of the three fits in the record's three
// dimension floats. What the number *means* is the host's convention, not this
// library's — see svoFieldProgramAbsentWGSL.
fn svoPrimitiveArenaReference(record: SvoPrimitiveRecord) -> u32 { return record.metadata.y; }
fn svoPrimitiveTerrainReference(record: SvoPrimitiveRecord) -> u32 { return record.metadata.y; }
fn svoPrimitiveClusterReference(record: SvoPrimitiveRecord) -> u32 { return record.metadata.y; }
fn svoPrimitiveFieldProgramReference(record: SvoPrimitiveRecord) -> u32 { return record.metadata.y; }

/**
 * The conservative local half-extent of every kind, generated from the same
 * table the CPU reads. A negative component means the kind has no finite local
 * box, which a caller must treat as a refusal rather than clamp to zero.
 */
fn svoPrimitiveLocalExtent_m(kind: u32, dimensions_m: vec3f) -> vec3f {
  if (kind == SVO_KIND_SPHERE) { return vec3f(dimensions_m.x); }
  // A field program's dimensions are already the conservative box its tape's
  // warps can reach, so the extent is the record verbatim, exactly as a box's is.
  if (kind == SVO_KIND_BOX || kind == SVO_KIND_ELLIPSOID || kind == SVO_KIND_SMOOTH_UNION_CLUSTER
    || kind == SVO_KIND_FIELD_PROGRAM) { return dimensions_m; }
  if (kind == SVO_KIND_CAPSULE) { return vec3f(dimensions_m.x, dimensions_m.y + dimensions_m.x, dimensions_m.x); }
  if (kind == SVO_KIND_CYLINDER) { return vec3f(dimensions_m.x, dimensions_m.y, dimensions_m.x); }
  if (kind == SVO_KIND_ROUNDED_CYLINDER) { return vec3f(dimensions_m.x, dimensions_m.y, dimensions_m.x); }
  if (kind == SVO_KIND_TORUS) { return vec3f(dimensions_m.x + dimensions_m.y, dimensions_m.y, dimensions_m.x + dimensions_m.y); }
  if (kind == SVO_KIND_CONE) { let radius = max(dimensions_m.x, dimensions_m.z); return vec3f(radius, dimensions_m.y, radius); }
  if (kind == SVO_KIND_ROUND_CONE) { let radius = max(dimensions_m.x, dimensions_m.z); return vec3f(radius, dimensions_m.y + radius, radius); }
  return vec3f(-1.0);
}

fn svoQuaternionRotate(q: vec4f, point: vec3f) -> vec3f {
  let twiceCross = 2.0 * cross(q.xyz, point);
  return point + q.w * twiceCross + cross(q.xyz, twiceCross);
}

${svoClusterFieldConstantsWGSL}
/** No field owns this code, so a block carrying it evaluates as a miss. */
const SVO_CLUSTER_FIELD_INVALID: u32 = 0xffffffffu;
const SVO_CLUSTER_SWEEP_MAXIMUM_POINTS: u32 = ${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}u;

/**
 * A cluster's arena block, as the shader sees it.
 *
 * The envelope's half-axes are absent because the record already carries them
 * as its dimensions — which is also why every bounds formula in the tree sees a
 * plain ellipsoid and needs to know nothing about this kind.
 *
 * One struct for the whole field family rather than one per field, because
 * there are no unions in WGSL and the alternative is a distinct packing
 * parameter threaded through \`svoIntersectPrimitiveExact\` for each. The
 * polyline dominates it: eight \`vec4f\` against a dozen scalars for everything
 * else. That is the cost of keeping the shared library ignorant of where the
 * arena is bound — a field that read its own control points out of storage
 * would tie this file to one shader's binding layout, and there are three
 * shaders using it. The array is dynamically indexed, so a driver stages it in
 * scratch rather than in registers and the cost lands on the fields that use it.
 */
struct SvoClusterPacking {
  field: u32,
  seed: u32,
  /** Octaves, lobes or control points, depending on the field. */
  count: u32,
  smoothRadius_m: f32,
  latticeLobeRadius_m: f32,
  latticePeriod_m: f32,
  jitter: f32,
  anisotropy: f32,
  /** Seeded-lobe placement, already defaulted by the packer. Fractions of the envelope. */
  lobeSpan: f32,
  lobeSpanSpread: f32,
  displacement: f32,
  /** xyz is a control-point offset from the record's centre, w its sweep radius. */
  points: array<vec4f, ${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}>,
}

/** The encoding of "this record's block was not resolved". Evaluates as a miss. */
fn svoInvalidClusterPacking() -> SvoClusterPacking {
  return SvoClusterPacking(SVO_CLUSTER_FIELD_INVALID, 0u, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    array<vec4f, ${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}>());
}

// A block that was never filled is all zeroes, which reads as the lattice with
// no lobe radius and no period — rejected here. That is deliberate: the "not
// resolved" case has to fail the same test a corrupt one does, because a
// cluster that fell back to its envelope would draw a smooth ellipsoid nobody
// authored and raise nothing anywhere.
fn svoClusterPackingValid(packing: SvoClusterPacking) -> bool {
  if (packing.field == SVO_CLUSTER_FIELD_LATTICE) {
    return packing.latticeLobeRadius_m > 0.0 && packing.latticePeriod_m > 0.0 && packing.count >= 1u;
  }
  if (packing.field == SVO_CLUSTER_FIELD_SEEDED_LOBES) {
    return packing.count >= 1u && packing.anisotropy >= 1.0 && packing.lobeSpan > 0.0;
  }
  if (packing.field == SVO_CLUSTER_FIELD_TAPERED_SWEEP) {
    return packing.count >= 2u && packing.count <= SVO_CLUSTER_SWEEP_MAXIMUM_POINTS && packing.points[0].w > 0.0;
  }
  if (packing.field == SVO_CLUSTER_FIELD_NOISE_FOLIAGE) {
    return packing.latticeLobeRadius_m > 0.0 && packing.latticePeriod_m > 0.0
      && packing.anisotropy + packing.lobeSpan > 0.0;
  }
  return false;
}

/** Three decorrelated offsets in [-1,1] from one avalanche; the CPU mirrors it bit for bit. */
fn svoClusterCellJitter(cell: vec3i, seed: u32) -> vec3f {
  var hash = seed;
  hash = hash ^ (bitcast<u32>(cell.x) * 0x9e3779b1u);
  hash = hash ^ (bitcast<u32>(cell.y) * 0x85ebca77u);
  hash = hash ^ (bitcast<u32>(cell.z) * 0xc2b2ae3du);
  hash = hash ^ (hash >> 16u); hash = hash * 0x7feb352du; hash = hash ^ (hash >> 15u);
  hash = hash * 0x846ca68bu; hash = hash ^ (hash >> 16u);
  let quantized = vec3u(hash & 1023u, (hash >> 10u) & 1023u, (hash >> 20u) & 1023u);
  return vec3f(quantized) / 1023.0 * 2.0 - vec3f(1.0);
}

/** Polynomial smooth minimum: a strict lower bound on min(a,b), so the trace understeps. */
fn svoSmoothMinimum(a: f32, b: f32, k: f32) -> f32 {
  if (!(k > 0.0)) { return min(a, b); }
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

// Lipschitz-1 lower bound on the ellipsoid distance, sharing its exact zero
// set. The closest-point solve is a 64-iteration bisection and this runs on
// every march step of every covering fragment. Used both as the envelope every
// field is clipped to and as one anisotropic lobe of a seeded set.
fn svoClusterLobeDistance_m(point: vec3f, lobeRadii_m: vec3f) -> f32 {
  return (length(point / lobeRadii_m) - 1.0) * min(lobeRadii_m.x, min(lobeRadii_m.y, lobeRadii_m.z));
}

/** One octave's lengths, at one scale, plus the seed that decorrelates it. */
struct SvoClusterLatticeOctave {
  latticeLobeRadius_m: f32,
  latticePeriod_m: f32,
  jitter: f32,
  smoothRadius_m: f32,
  seed: u32,
}

fn svoClusterSphereDistance_m(point: vec3f, cell: vec3i, octave: SvoClusterLatticeOctave) -> f32 {
  let centre = (vec3f(cell) + vec3f(0.5) + octave.jitter * svoClusterCellJitter(cell, octave.seed)) * octave.latticePeriod_m;
  return length(point - centre) - octave.latticeLobeRadius_m;
}

// Two passes over a neighbourhood centred on the point. The inner eight cells
// always matter and set a tight running minimum; the outer shell is rejected by
// the distance to its unjittered centre, which needs no hash. The fold order is
// fixed and mirrored on the CPU because a polynomial smooth minimum is not
// associative. See SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD for why the block is
// this size rather than the obvious eight.
fn svoClusterLatticeOctaveDistance_m(point: vec3f, octave: SvoClusterLatticeOctave) -> f32 {
  const SPAN: i32 = ${SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD};
  const RING: i32 = SPAN / 2 - 1;
  let base = vec3i(floor(point / octave.latticePeriod_m - vec3f(0.5))) - vec3i(RING);
  var distance_m = 0.0;
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let cell = base + vec3i(RING) + vec3i(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
    let sphere = svoClusterSphereDistance_m(point, cell, octave);
    distance_m = select(svoSmoothMinimum(distance_m, sphere, octave.smoothRadius_m), sphere, corner == 0u);
  }
  let reach = octave.jitter * octave.latticePeriod_m * 1.7320508 + octave.latticeLobeRadius_m + octave.smoothRadius_m;
  for (var index = 0; index < SPAN * SPAN * SPAN; index += 1) {
    let step = vec3i(index % SPAN, (index / SPAN) % SPAN, (index / (SPAN * SPAN)) % SPAN);
    if (all(step >= vec3i(RING)) && all(step <= vec3i(RING + 1))) { continue; }
    let cell = base + step;
    let nominal = length(point - (vec3f(cell) + vec3f(0.5)) * octave.latticePeriod_m);
    if (nominal - reach > distance_m) { continue; }
    distance_m = svoSmoothMinimum(distance_m, svoClusterSphereDistance_m(point, cell, octave), octave.smoothRadius_m);
  }
  return distance_m;
}

// Octave k is the octave-zero packing at 2^-k, so the period condition that
// makes the first one a lower bound carries to every one after it unchanged.
// exp2(0) is exactly 1 and the seed mix is exactly the seed at k = 0, which is
// what keeps a one-octave field bit-identical to the field this kind published
// before octaves existed.
fn svoClusterLatticeDistance_m(point: vec3f, packing: SvoClusterPacking) -> f32 {
  var distance_m = 0.0;
  for (var octave = 0u; octave < packing.count; octave += 1u) {
    let scale = exp2(-f32(octave));
    let smoothRadius_m = packing.smoothRadius_m * scale;
    let sample = svoClusterLatticeOctaveDistance_m(point, SvoClusterLatticeOctave(
      packing.latticeLobeRadius_m * scale, packing.latticePeriod_m * scale, packing.jitter,
      smoothRadius_m, packing.seed ^ (octave * 0x9e3779b1u),
    ));
    distance_m = select(svoSmoothMinimum(distance_m, sample, smoothRadius_m), sample, octave == 0u);
  }
  return distance_m;
}

// Centre and half-axes in the envelope's normalised space, where the envelope
// is the unit ball. Fractions of the envelope, not metres.
struct SvoClusterSeededLobe {
  center: vec3f,
  halfAxes: vec3f,
  orientation: vec4f,
}

// Lobe placement, derived from the seed and the envelope and stored nowhere.
// A lobe's longest half-axis is exactly its span and its centre stays inside
// 1 - span - smooth/Rmin of the origin, so the envelope's own Lipschitz-1 lower
// bound at the lobe's surface already covers the blend and the hard max never
// touches it. Normalised space is what makes the solid scale with the envelope
// it was authored against rather than with a ball of its shortest half-axis.
// Lobe zero is centred, the cheapest guarantee that the others have something
// to fuse to. Everything here depends only on the packing and the envelope, so
// it is loop-invariant over a march and a driver may hoist it out.
fn svoClusterSeededLobe(index: u32, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> SvoClusterSeededLobe {
  let shortest_m = min(lobeRadii_m.x, min(lobeRadii_m.y, lobeRadii_m.z));
  let offsetHash = svoClusterCellJitter(vec3i(i32(index), 0, 0), packing.seed);
  let shapeHash = svoClusterCellJitter(vec3i(i32(index), 1, 0), packing.seed);
  let axisHash = svoClusterCellJitter(vec3i(i32(index), 2, 0), packing.seed);
  let spreadHash = svoClusterCellJitter(vec3i(i32(index), 3, 0), packing.seed);

  let raw = vec3f(pow(packing.anisotropy, 0.5 * shapeHash.x), pow(packing.anisotropy, 0.5 * shapeHash.y),
    pow(packing.anisotropy, 0.5 * shapeHash.z));
  let longest = max(raw.x, max(raw.y, raw.z));
  let span = packing.lobeSpan + packing.lobeSpanSpread * 0.5 * (spreadHash.x + 1.0);
  let halfAxes = span * raw / longest;

  let budget = select(
    max(0.0, 1.0 - span - packing.smoothRadius_m / shortest_m) * packing.displacement, 0.0, index == 0u);
  // The hash supplies the direction only; the radius is the budget share, so a
  // lobe is placed at a chosen distance rather than wherever a cube-uniform
  // hash fell — which left the solid clear of its own envelope on every side.
  let offsetLength = length(offsetHash);
  let offsetScale = select(0.0, budget / max(offsetLength, 1e-12), offsetLength > 1e-12);

  // Axis-angle, not four hashed components normalised: this is a rotation for
  // every hash including the degenerate one, which the shader cannot reject.
  let axisLength = length(axisHash);
  let axis = select(vec3f(0.0, 1.0, 0.0), axisHash / max(axisLength, 1e-12), axisLength > 1e-12);
  let half = 0.5 * 3.1415927 * spreadHash.y;
  return SvoClusterSeededLobe(offsetHash * offsetScale, halfAxes, vec4f(axis * sin(half), cos(half)));
}

// A lobe is the affine image diag(R)*Q*diag(k) of the unit ball. (|M^-1 w| - 1)
// shares its exact zero set, and min(R)*min(k) is a lower bound on M's smallest
// singular value — sigma_min(AB) >= sigma_min(A) sigma_min(B), and a rotation
// has none — so the product is Lipschitz-1. Loose rather than tight, which
// costs march steps and never correctness.
fn svoClusterSeededLobesDistance_m(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> f32 {
  let shortest_m = min(lobeRadii_m.x, min(lobeRadii_m.y, lobeRadii_m.z));
  let normalized = point / lobeRadii_m;
  var distance_m = 0.0;
  for (var index = 0u; index < packing.count; index += 1u) {
    let lobe = svoClusterSeededLobe(index, lobeRadii_m, packing);
    let local = svoQuaternionRotate(vec4f(-lobe.orientation.xyz, lobe.orientation.w), normalized - lobe.center);
    let scaled = length(local / lobe.halfAxes);
    let sample = (scaled - 1.0) * shortest_m * min(lobe.halfAxes.x, min(lobe.halfAxes.y, lobe.halfAxes.z));
    distance_m = select(svoSmoothMinimum(distance_m, sample, packing.smoothRadius_m), sample, index == 0u);
  }
  return distance_m;
}

// Exact distance to the convex hull of two spheres, so Lipschitz-1 by
// definition rather than by argument. The validator rejects a segment whose
// taper outruns its own length, which is what keeps lateralSquared positive.
fn svoClusterRoundConeDistance_m(point: vec3f, start: vec4f, finish: vec4f) -> f32 {
  let axis = finish.xyz - start.xyz;
  let axisSquared = dot(axis, axis);
  let taper = start.w - finish.w;
  let lateralSquared = axisSquared - taper * taper;
  let inverseAxisSquared = 1.0 / axisSquared;
  let offset = point - start.xyz;
  let along = dot(offset, axis);
  let beyond = along - axisSquared;
  let perpendicular = offset * axisSquared - axis * along;
  let perpendicularSquared = dot(perpendicular, perpendicular);
  let alongSquared = along * along * axisSquared;
  let beyondSquared = beyond * beyond * axisSquared;
  let split = sign(taper) * taper * taper * perpendicularSquared;
  if (sign(beyond) * lateralSquared * beyondSquared > split) {
    return sqrt(perpendicularSquared + beyondSquared) * inverseAxisSquared - finish.w;
  }
  if (sign(along) * lateralSquared * alongSquared < split) {
    return sqrt(perpendicularSquared + alongSquared) * inverseAxisSquared - start.w;
  }
  return (sqrt(perpendicularSquared * lateralSquared * inverseAxisSquared) + along * taper) * inverseAxisSquared - start.w;
}

fn svoRoundConeDistance_m(point:vec3f,dimensions_m:vec3f)->f32{
  let span=2.0*dimensions_m.y;
  let taper=(dimensions_m.x-dimensions_m.z)/span;
  let lateral=sqrt(max(0.0,1.0-taper*taper));
  let radial=length(point.xz);
  let along=point.y+dimensions_m.y;
  let region=-taper*radial+lateral*along;
  if(region<0.0){return length(vec3f(point.x,along,point.z))-dimensions_m.x;}
  if(region>lateral*span){return length(vec3f(point.x,along-span,point.z))-dimensions_m.z;}
  return lateral*radial+taper*along-dimensions_m.x;
}

fn svoRoundConeNormal(point:vec3f,dimensions_m:vec3f)->vec3f{
  let span=2.0*dimensions_m.y;
  let taper=(dimensions_m.x-dimensions_m.z)/span;
  let lateral=sqrt(max(0.0,1.0-taper*taper));
  let radial=length(point.xz);
  let along=point.y+dimensions_m.y;
  let region=-taper*radial+lateral*along;
  if(region<0.0){let offset=vec3f(point.x,along,point.z);let magnitude=length(offset);return select(vec3f(0.0),offset/magnitude,magnitude>1e-12);}
  if(region>lateral*span){let offset=vec3f(point.x,along-span,point.z);let magnitude=length(offset);return select(vec3f(0.0),offset/magnitude,magnitude>1e-12);}
  let direction=point.xz/max(radial,1e-8);
  return vec3f(lateral*direction.x,taper,lateral*direction.y);
}

fn svoRoundedCylinderDistance_m(point:vec3f,dimensions_m:vec3f)->f32{
  let radial=length(point.xz);
  let core=dimensions_m.xy-vec2f(dimensions_m.z);
  let q=vec2f(radial-core.x,abs(point.y)-core.y);
  return length(max(q,vec2f(0.0)))+min(max(q.x,q.y),0.0)-dimensions_m.z;
}

fn svoRoundedCylinderNormal(point:vec3f,dimensions_m:vec3f)->vec3f{
  let radial=length(point.xz);
  let direction=point.xz/max(radial,1e-8);
  let core=dimensions_m.xy-vec2f(dimensions_m.z);
  let q=vec2f(radial-core.x,abs(point.y)-core.y);
  if(q.x>0.0&&q.y>0.0){let gradient=normalize(q);return vec3f(direction.x*gradient.x,select(-gradient.y,gradient.y,point.y>=0.0),direction.y*gradient.x);}
  if(q.x>q.y){return vec3f(direction.x,0.0,direction.y);}
  return vec3f(0.0,select(-1.0,1.0,point.y>=0.0),0.0);
}

// The polyline is copied into a function-scope var so the loop may index it
// dynamically; the fold order is the authored one, mirrored on the CPU because
// a polynomial smooth minimum is not associative.
fn svoClusterTaperedSweepDistance_m(point: vec3f, packing: SvoClusterPacking) -> f32 {
  var points = packing.points;
  let count = min(packing.count, SVO_CLUSTER_SWEEP_MAXIMUM_POINTS);
  var distance_m = 0.0;
  for (var index = 0u; index + 1u < count; index += 1u) {
    let sample = svoClusterRoundConeDistance_m(point, points[index], points[index + 1u]);
    distance_m = select(svoSmoothMinimum(distance_m, sample, packing.smoothRadius_m), sample, index == 0u);
  }
  return distance_m;
}

// noise-foliage aliases the generic scalar slots as follows:
//   latticePeriod = cluster period, latticeLobeRadius = detail period,
//   jitter = threshold, anisotropy = cluster weight, lobeSpan = detail weight,
//   lobeSpanSpread = interior bias.
// Density above zero is occupied. The division is the complete gradient bound
// of both smoothstep-trilinear noises plus the radial interior term, producing
// a Lipschitz-1 lower bound safe for the shared sphere trace.
fn svoClusterNoiseFoliageDistance_m(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> f32 {
  let clusterFrequency = 1.0 / packing.latticePeriod_m;
  let detailFrequency = 1.0 / packing.latticeLobeRadius_m;
  let cluster = svoProceduralNoise(point, vec3f(clusterFrequency), packing.seed ^ 0x68bc21ebu);
  let detail = svoProceduralNoise(point, vec3f(detailFrequency), packing.seed ^ 0x02e5be93u);
  let clustered = cluster * cluster * (3.0 - 2.0 * cluster);
  let normalizedRadius = length(point / lobeRadii_m);
  let interior = max(0.0, 1.0 - normalizedRadius);
  let fadeLinear = clamp(interior / ${NOISE_FOLIAGE_EDGE_FADE.toFixed(8)}, 0.0, 1.0);
  let fade = fadeLinear * fadeLinear * (3.0 - 2.0 * fadeLinear);
  let rawDensity = packing.anisotropy * clustered + packing.lobeSpan * detail + packing.lobeSpanSpread;
  let density = fade * rawDensity - packing.jitter;
  let shortest_m = min(lobeRadii_m.x, min(lobeRadii_m.y, lobeRadii_m.z));
  let noiseGradient_mInv = ${NOISE_FOLIAGE_VALUE_GRADIENT_BOUND.toFixed(8)}
    * (1.5 * packing.anisotropy * clusterFrequency + packing.lobeSpan * detailFrequency);
  let fadeGradient_mInv = 1.5 / (${NOISE_FOLIAGE_EDGE_FADE.toFixed(8)} * shortest_m);
  let lipschitz_mInv = noiseGradient_mInv
    + (packing.anisotropy + packing.lobeSpan + packing.lobeSpanSpread) * fadeGradient_mInv;
  return -density / lipschitz_mInv;
}

fn svoClusterFieldDistance_m(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> f32 {
  if (packing.field == SVO_CLUSTER_FIELD_NOISE_FOLIAGE) {
    return svoClusterNoiseFoliageDistance_m(point, lobeRadii_m, packing);
  }
  if (packing.field == SVO_CLUSTER_FIELD_SEEDED_LOBES) {
    return svoClusterSeededLobesDistance_m(point, lobeRadii_m, packing);
  }
  if (packing.field == SVO_CLUSTER_FIELD_TAPERED_SWEEP) { return svoClusterTaperedSweepDistance_m(point, packing); }
  return svoClusterLatticeDistance_m(point, packing);
}

// The smallest solid feature a field draws, and a quarter of it is the gradient
// step. Per field because the number that sets it differs in each; a fixed
// fraction of the envelope would step across several features at once and hand
// back the envelope's own.
fn svoClusterFeatureRadius_m(lobeRadii_m: vec3f, packing: SvoClusterPacking) -> f32 {
  if (packing.field == SVO_CLUSTER_FIELD_NOISE_FOLIAGE) { return 0.5 * packing.latticeLobeRadius_m; }
  if (packing.field == SVO_CLUSTER_FIELD_SEEDED_LOBES) {
    let shortest_m = min(lobeRadii_m.x, min(lobeRadii_m.y, lobeRadii_m.z));
    return max(packing.smoothRadius_m, packing.lobeSpan * shortest_m / max(packing.anisotropy, 1.0));
  }
  if (packing.field == SVO_CLUSTER_FIELD_TAPERED_SWEEP) {
    var points = packing.points;
    let count = min(packing.count, SVO_CLUSTER_SWEEP_MAXIMUM_POINTS);
    var thinnest_m = points[0].w;
    for (var index = 1u; index < count; index += 1u) { thinnest_m = min(thinnest_m, points[index].w); }
    return thinnest_m;
  }
  // max(count, 1) rather than count: this is reachable through
  // svoEvaluatePrimitive without the validity gate the march applies, and a
  // zero count would wrap the u32 subtraction into a meaningless step.
  return packing.latticeLobeRadius_m * exp2(-f32(max(packing.count, 1u) - 1u));
}

fn svoClusterDistance_m(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> f32 {
  if (!svoClusterPackingValid(packing)) { return SVO_PRIMITIVE_RAY_INFINITY; }
  return max(svoClusterFieldDistance_m(point, lobeRadii_m, packing), svoClusterLobeDistance_m(point, lobeRadii_m));
}

/**
 * A field's value and gradient at one point.
 *
 * Geometric fields have closed-form outward normals carried through their
 * combinators. Noise foliage differentiates the thresholded density locally at
 * a fraction of its detail period, then the shared hard max still chooses
 * between that result and the envelope exactly.
 */
struct SvoClusterSample {
  distance_m: f32,
  gradient: vec3f,
}

/**
 * The gradient a polynomial smooth minimum implies, which is the plain blend.
 *
 * \`f = b + (a-b)h - k·h(1-h)\` with \`h = 0.5 + (b-a)/2k\` in the interior, so
 * \`∂f/∂a = h + (a-b)h' - k(1-2h)h'\`, and substituting \`a-b = -(2h-1)k\` and
 * \`h' = -1/2k\` the last two terms are \`(2h-1)/2\` and \`(1-2h)/2\`. They cancel
 * identically: \`∂f/∂a = h\`, \`∂f/∂b = 1-h\`. The blend weight is the *whole*
 * derivative, which is what makes this exact rather than an approximation of
 * the seam. Outside the interior \`h\` clamps and the winner's gradient passes
 * through unchanged.
 *
 * The distance arithmetic is character-for-character {@link svoSmoothMinimum}'s
 * so the two folds cannot drift; \`tests/svo-cluster-gradient.test.ts\` holds them
 * to it.
 */
fn svoSmoothMinimumSample(a: SvoClusterSample, b: SvoClusterSample, k: f32) -> SvoClusterSample {
  if (!(k > 0.0)) {
    if (a.distance_m <= b.distance_m) { return a; }
    return b;
  }
  let h = clamp(0.5 + 0.5 * (b.distance_m - a.distance_m) / k, 0.0, 1.0);
  return SvoClusterSample(mix(b.distance_m, a.distance_m, h) - k * h * (1.0 - h),
    mix(b.gradient, a.gradient, h));
}

/** A degenerate gradient is answered with a zero vector, never with a guess. */
fn svoClusterUnitGradient(offset: vec3f) -> vec3f {
  let magnitude = length(offset);
  return select(vec3f(0.0), offset / magnitude, magnitude > 1e-12);
}

fn svoClusterSphereSample(point: vec3f, cell: vec3i, octave: SvoClusterLatticeOctave) -> SvoClusterSample {
  let centre = (vec3f(cell) + vec3f(0.5) + octave.jitter * svoClusterCellJitter(cell, octave.seed)) * octave.latticePeriod_m;
  let offset = point - centre;
  return SvoClusterSample(length(offset) - octave.latticeLobeRadius_m, svoClusterUnitGradient(offset));
}

/** {@link svoClusterLatticeOctaveDistance_m}, differentiated in the same fold order. */
fn svoClusterLatticeOctaveSample(point: vec3f, octave: SvoClusterLatticeOctave) -> SvoClusterSample {
  const SPAN: i32 = ${SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD};
  const RING: i32 = SPAN / 2 - 1;
  let base = vec3i(floor(point / octave.latticePeriod_m - vec3f(0.5))) - vec3i(RING);
  var sample = SvoClusterSample(0.0, vec3f(0.0));
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let cell = base + vec3i(RING) + vec3i(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
    let sphere = svoClusterSphereSample(point, cell, octave);
    if (corner == 0u) { sample = sphere; } else { sample = svoSmoothMinimumSample(sample, sphere, octave.smoothRadius_m); }
  }
  let reach = octave.jitter * octave.latticePeriod_m * 1.7320508 + octave.latticeLobeRadius_m + octave.smoothRadius_m;
  for (var index = 0; index < SPAN * SPAN * SPAN; index += 1) {
    let step = vec3i(index % SPAN, (index / SPAN) % SPAN, (index / (SPAN * SPAN)) % SPAN);
    if (all(step >= vec3i(RING)) && all(step <= vec3i(RING + 1))) { continue; }
    let cell = base + step;
    let nominal = length(point - (vec3f(cell) + vec3f(0.5)) * octave.latticePeriod_m);
    if (nominal - reach > sample.distance_m) { continue; }
    sample = svoSmoothMinimumSample(sample, svoClusterSphereSample(point, cell, octave), octave.smoothRadius_m);
  }
  return sample;
}

fn svoClusterLatticeSample(point: vec3f, packing: SvoClusterPacking) -> SvoClusterSample {
  var sample = SvoClusterSample(0.0, vec3f(0.0));
  for (var octave = 0u; octave < packing.count; octave += 1u) {
    let scale = exp2(-f32(octave));
    let smoothRadius_m = packing.smoothRadius_m * scale;
    let octaveSample = svoClusterLatticeOctaveSample(point, SvoClusterLatticeOctave(
      packing.latticeLobeRadius_m * scale, packing.latticePeriod_m * scale, packing.jitter,
      smoothRadius_m, packing.seed ^ (octave * 0x9e3779b1u),
    ));
    if (octave == 0u) { sample = octaveSample; } else { sample = svoSmoothMinimumSample(sample, octaveSample, smoothRadius_m); }
  }
  return sample;
}

/**
 * {@link svoClusterRoundConeDistance_m}, differentiated branch for branch.
 *
 * Two of its three branches are spheres wearing the segment's algebra:
 * \`P + b²A\` is \`A²|p - finish|²\` and \`P + l²A\` is \`A²|p - start|²\` identically,
 * so those branches are \`|p - c| - r\` and their gradients are the radial
 * direction. The side branch is \`perpendicular·√(L/A) + along·taper/A\` in the
 * unit frame, whose gradient has magnitude \`√((L + taper²)/A) = 1\` by
 * construction — the same identity that makes the distance Lipschitz-1.
 */
fn svoClusterRoundConeSample(point: vec3f, start: vec4f, finish: vec4f) -> SvoClusterSample {
  let axis = finish.xyz - start.xyz;
  let axisSquared = dot(axis, axis);
  let taper = start.w - finish.w;
  let lateralSquared = axisSquared - taper * taper;
  let inverseAxisSquared = 1.0 / axisSquared;
  let offset = point - start.xyz;
  let along = dot(offset, axis);
  let beyond = along - axisSquared;
  let perpendicular = offset * axisSquared - axis * along;
  let perpendicularSquared = dot(perpendicular, perpendicular);
  let alongSquared = along * along * axisSquared;
  let beyondSquared = beyond * beyond * axisSquared;
  let split = sign(taper) * taper * taper * perpendicularSquared;
  if (sign(beyond) * lateralSquared * beyondSquared > split) {
    let radial = point - finish.xyz;
    return SvoClusterSample(sqrt(perpendicularSquared + beyondSquared) * inverseAxisSquared - finish.w,
      svoClusterUnitGradient(radial));
  }
  if (sign(along) * lateralSquared * alongSquared < split) {
    return SvoClusterSample(sqrt(perpendicularSquared + alongSquared) * inverseAxisSquared - start.w,
      svoClusterUnitGradient(offset));
  }
  let side = (sqrt(perpendicularSquared * lateralSquared * inverseAxisSquared) + along * taper) * inverseAxisSquared - start.w;
  let outward = svoClusterUnitGradient(perpendicular) * sqrt(max(0.0, lateralSquared * inverseAxisSquared))
    + axis * (taper * inverseAxisSquared);
  return SvoClusterSample(side, outward);
}

fn svoClusterTaperedSweepSample(point: vec3f, packing: SvoClusterPacking) -> SvoClusterSample {
  var points = packing.points;
  let count = min(packing.count, SVO_CLUSTER_SWEEP_MAXIMUM_POINTS);
  var sample = SvoClusterSample(0.0, vec3f(0.0));
  for (var index = 0u; index + 1u < count; index += 1u) {
    let segment = svoClusterRoundConeSample(point, points[index], points[index + 1u]);
    if (index == 0u) { sample = segment; } else { sample = svoSmoothMinimumSample(sample, segment, packing.smoothRadius_m); }
  }
  return sample;
}

/**
 * One seeded lobe, differentiated through the frames it is authored in.
 *
 * \`scaled = |R⁻¹(p/r - c) / halfAxes|\`, so the chain is the ellipsoid gradient
 * in lobe space, rotated back by the lobe's own orientation, then divided by the
 * envelope radii the normalised space was made with.
 */
fn svoClusterSeededLobesSample(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> SvoClusterSample {
  let shortest_m = min(lobeRadii_m.x, min(lobeRadii_m.y, lobeRadii_m.z));
  let normalized = point / lobeRadii_m;
  var sample = SvoClusterSample(0.0, vec3f(0.0));
  for (var index = 0u; index < packing.count; index += 1u) {
    let lobe = svoClusterSeededLobe(index, lobeRadii_m, packing);
    let local = svoQuaternionRotate(vec4f(-lobe.orientation.xyz, lobe.orientation.w), normalized - lobe.center);
    let scaledVector = local / lobe.halfAxes;
    let scaled = length(scaledVector);
    let span = shortest_m * min(lobe.halfAxes.x, min(lobe.halfAxes.y, lobe.halfAxes.z));
    let localGradient = select(vec3f(0.0), (scaledVector / lobe.halfAxes) / scaled, scaled > 1e-12);
    let lobeSample = SvoClusterSample((scaled - 1.0) * span,
      svoQuaternionRotate(lobe.orientation, localGradient) * span / lobeRadii_m);
    if (index == 0u) { sample = lobeSample; } else { sample = svoSmoothMinimumSample(sample, lobeSample, packing.smoothRadius_m); }
  }
  return sample;
}

/** The envelope every field is clipped to, and its gradient. */
fn svoClusterLobeSample(point: vec3f, lobeRadii_m: vec3f) -> SvoClusterSample {
  let shortest_m = min(lobeRadii_m.x, min(lobeRadii_m.y, lobeRadii_m.z));
  let normalized = point / lobeRadii_m;
  let magnitude = length(normalized);
  let gradient = select(vec3f(0.0), (normalized / lobeRadii_m) / magnitude, magnitude > 1e-12);
  return SvoClusterSample((magnitude - 1.0) * shortest_m, gradient * shortest_m);
}

fn svoClusterNoiseFoliageSample(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> SvoClusterSample {
  let distance_m = svoClusterNoiseFoliageDistance_m(point, lobeRadii_m, packing);
  let h = max(1e-6, packing.latticeLobeRadius_m * 0.001);
  let dx = svoClusterNoiseFoliageDistance_m(point + vec3f(h, 0.0, 0.0), lobeRadii_m, packing)
    - svoClusterNoiseFoliageDistance_m(point - vec3f(h, 0.0, 0.0), lobeRadii_m, packing);
  let dy = svoClusterNoiseFoliageDistance_m(point + vec3f(0.0, h, 0.0), lobeRadii_m, packing)
    - svoClusterNoiseFoliageDistance_m(point - vec3f(0.0, h, 0.0), lobeRadii_m, packing);
  let dz = svoClusterNoiseFoliageDistance_m(point + vec3f(0.0, 0.0, h), lobeRadii_m, packing)
    - svoClusterNoiseFoliageDistance_m(point - vec3f(0.0, 0.0, h), lobeRadii_m, packing);
  return SvoClusterSample(distance_m, vec3f(dx, dy, dz) / (2.0 * h));
}

fn svoClusterFieldSample(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> SvoClusterSample {
  if (packing.field == SVO_CLUSTER_FIELD_NOISE_FOLIAGE) {
    return svoClusterNoiseFoliageSample(point, lobeRadii_m, packing);
  }
  if (packing.field == SVO_CLUSTER_FIELD_SEEDED_LOBES) {
    return svoClusterSeededLobesSample(point, lobeRadii_m, packing);
  }
  if (packing.field == SVO_CLUSTER_FIELD_TAPERED_SWEEP) { return svoClusterTaperedSweepSample(point, packing); }
  return svoClusterLatticeSample(point, packing);
}

/** {@link svoClusterDistance_m} and its gradient: a \`max\` passes the winner through. */
fn svoClusterSample(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> SvoClusterSample {
  if (!svoClusterPackingValid(packing)) { return SvoClusterSample(SVO_PRIMITIVE_RAY_INFINITY, vec3f(0.0)); }
  let field = svoClusterFieldSample(point, lobeRadii_m, packing);
  let envelope = svoClusterLobeSample(point, lobeRadii_m);
  if (field.distance_m >= envelope.distance_m) { return field; }
  return envelope;
}

/**
 * The cluster's shading normal — one differentiated pass, not four evaluations.
 *
 * This used to be a four-tap tetrahedral difference, and its own comment priced
 * a tap at "roughly sixty sphere distances". That was the frame: on
 * \`hero-garden-hose\` at refinement depth 3, deleting *only* the cluster
 * gradients took the 800x460 frame from 42.7 ms to 19.1 ms — **55 % of the whole
 * frame in this one function**, against 4 % for the field-program tapes beside
 * it and nothing measurable for the ground or the stencil fallback. The scene
 * publishes 631 cluster records against 145 tapes, and a lattice record folds
 * sixty-four jittered spheres per octave, so the four taps were spending
 * thousands of sphere distances per shaded pixel.
 *
 * Every leaf of the fold already knew its own normal. Carrying it through the
 * blend costs one \`mix\` per combination and deletes three quarters of the
 * evaluations for geometric fields. Noise foliage is the deliberate exception:
 * it takes six nearby density samples because there is no analytic lobe normal
 * to reuse, and its step is tied to the authored detail period.
 */
fn svoClusterLocalNormal(point: vec3f, lobeRadii_m: vec3f, packing: SvoClusterPacking) -> vec3f {
  return svoClusterUnitGradient(svoClusterSample(point, lobeRadii_m, packing).gradient);
}

fn svoPrimitiveLocalPoint(record: SvoPrimitiveRecord, worldPoint_m: vec3f) -> vec3f {
  let q = record.orientation;
  return svoQuaternionRotate(vec4f(-q.xyz, q.w), worldPoint_m - svoPrimitiveCenter_m(record));
}

fn svoPrimitiveNoRayHit(status: u32) -> SvoPrimitiveRayResult {
  return SvoPrimitiveRayResult(SVO_PRIMITIVE_RAY_INFINITY, SVO_FEATURE_SMOOTH, status, 0u, vec4f(0.0));
}

// Sorted roots of a*t^2 + 2*b*t + c. The stable q form retains near roots.
fn svoPrimitiveQuadraticRoots(a: f32, b: f32, c: f32) -> SvoPrimitiveQuadraticRoots {
  if (!(a > 1e-8)) { return SvoPrimitiveQuadraticRoots(vec2f(0.0), 0u, 0u); }
  let discriminant = b * b - a * c;
  let tolerance = 8e-6 * max(1.0, max(abs(b * b), abs(a * c)));
  if (discriminant < -tolerance) { return SvoPrimitiveQuadraticRoots(vec2f(0.0), 0u, 0u); }
  let root = sqrt(max(0.0, discriminant));
  if (root == 0.0) { return SvoPrimitiveQuadraticRoots(vec2f(-b / a, 0.0), 1u, 0u); }
  let q = -b - select(-root, root, b >= 0.0);
  let first = q / a;
  let second = c / q;
  return SvoPrimitiveQuadraticRoots(vec2f(min(first, second), max(first, second)), 2u, 0u);
}

fn svoPrimitiveRayInRange(t_m: f32, tMin_m: f32, tMax_m: f32) -> bool {
  let tolerance_m = 8e-6 * max(1.0, max(abs(t_m), max(abs(tMin_m), abs(tMax_m))));
  return t_m >= tMin_m - tolerance_m && t_m <= tMax_m + tolerance_m;
}

struct SvoConeSample {
  distance_m: f32,
  featureId: u32,
  normal: vec3f,
}

fn svoTorusDistance_m(point: vec3f, dimensions_m: vec3f) -> f32 {
  let ring = vec2f(length(point.xz) - dimensions_m.x, point.y);
  return length(ring) - dimensions_m.y;
}

fn svoTorusNormal(point: vec3f, dimensions_m: vec3f) -> vec3f {
  let radial = length(point.xz);
  if (!(radial > 1e-8)) { return vec3f(0.0); }
  let scale = (radial - dimensions_m.x) / radial;
  let offset = vec3f(point.x * scale, point.y, point.z * scale);
  let magnitude = length(offset);
  return select(vec3f(0.0), offset / magnitude, magnitude > 1e-12);
}

// Exact frustum distance: nearest of the end disc and the lateral band, signed
// by whether the point is inside both. dimensions are (base, halfHeight, top).
fn svoConeSample(point: vec3f, dimensions_m: vec3f) -> SvoConeSample {
  let base = dimensions_m.x;
  let half = dimensions_m.y;
  let top = dimensions_m.z;
  let radial = length(point.xz);
  let capRadius = select(top, base, point.y < 0.0);
  let cap = vec2f(radial - min(radial, capRadius), abs(point.y) - half);
  let slope = vec2f(top - base, 2.0 * half);
  let toApex = vec2f(top - radial, half - point.y);
  let projection = clamp(dot(toApex, slope) / dot(slope, slope), 0.0, 1.0);
  let side = vec2f(radial - top, point.y - half) + slope * projection;
  let capSquared = dot(cap, cap);
  let sideSquared = dot(side, side);
  let inside = side.x < 0.0 && cap.y < 0.0;
  let distance_m = select(1.0, -1.0, inside) * sqrt(min(capSquared, sideSquared));
  // Hard features: select the authored end or the authored band, never a blend.
  if (capSquared <= sideSquared) {
    return SvoConeSample(distance_m, SVO_FEATURE_CYLINDER_CAP, vec3f(0.0, select(1.0, -1.0, point.y < 0.0), 0.0));
  }
  let lateral = normalize(vec2f(slope.y, base - top));
  let direction = select(vec2f(1.0, 0.0), point.xz / max(radial, 1e-8), radial > 1e-8);
  return SvoConeSample(distance_m, SVO_FEATURE_CYLINDER_SIDE, vec3f(direction.x * lateral.x, lateral.y, direction.y * lateral.x));
}

/**
 * A field program's distance, already divided by the Lipschitz constant the
 * evaluator returns beside it.
 *
 * **Do not "simplify" this to \`.distance_m\`.** A domain warp evaluates its
 * subtree at a moved point, so the composed field is L-Lipschitz rather than
 * 1-Lipschitz and |g(p)| *overestimates* the clearance by up to L. The march
 * below steps by exactly what this returns, and every other consumer of a kind's
 * distance in this tree — the voxelizer's occupancy test and its cell-coverage
 * corner sweep — is built on the same contract that the value is a **lower**
 * bound on the true clearance. Return the undivided distance and the trace steps
 * through its own surface and punches holes in every warped shape, which is the
 * failure §2.4 of docs/hero-fidelity-1000x-handoff.md names as the one that
 * kills naive versions of this design.
 *
 * Dividing here rather than threading the constant out to each call site costs
 * nothing: L is a property of the tape and not of the point, so this is a
 * uniform positive scale that moves neither the sign nor the zero set. What it
 * buys is that a field program is just another 1-Lipschitz lower bound to
 * everything downstream, with no new rule to know about. The only visible effect
 * is that the acceptance band stops the march up to L times its own epsilon
 * outside the surface — the safe side, and below the band the other marched
 * kinds already accept.
 */
fn svoFieldProgramDistance_m(reference: u32, point: vec3f) -> f32 {
  let value = svoFieldProgramReferenceSample(reference, point);
  return value.distance_m / max(1.0, value.lipschitz);
}

/**
 * Central-difference step for a field program's shading normal. Scaled to the
 * record's own extent because a tape has no analytic gradient and its finest
 * feature is a fraction of the shape rather than an absolute size; the floor
 * keeps a millimetre-scale record off the f32 noise shelf.
 */
fn svoFieldProgramNormalStep_m(dimensions_m: vec3f) -> f32 {
  return max(1e-5, 1e-3 * max(dimensions_m.x, max(dimensions_m.y, dimensions_m.z)));
}

/**
 * Tetrahedral rather than central: four field evaluations, not six.
 *
 * A field program's distance is the most expensive sample in this ABI — a whole
 * op chain per tap — and the six-tap central difference spent one third of them
 * on a component the other four already determine. The four offsets are the
 * vertices of a regular tetrahedron, so the summed contributions span all three
 * axes; it is the same construction \`svoClusterLocalNormal\` above already uses,
 * and the only thing that kept the two apart was that they were written at
 * different times.
 *
 * The result is normalised, so the scale factor the two forms differ by falls
 * out. What does not is second-order error, which a tetrahedral stencil places
 * differently rather than more of: both are exact on a linear field, and a
 * displaced field's curvature is what either one smooths.
 */
fn svoFieldProgramNormal(reference: u32, point: vec3f, dimensions_m: vec3f) -> vec3f {
  let step = svoFieldProgramNormalStep_m(dimensions_m);
  let a = vec3f(1.0, -1.0, -1.0); let b = vec3f(-1.0, -1.0, 1.0);
  let c = vec3f(-1.0, 1.0, -1.0); let d = vec3f(1.0, 1.0, 1.0);
  let gradient = a * svoFieldProgramDistance_m(reference, point + a * step)
    + b * svoFieldProgramDistance_m(reference, point + b * step)
    + c * svoFieldProgramDistance_m(reference, point + c * step)
    + d * svoFieldProgramDistance_m(reference, point + d * step);
  let magnitude = length(gradient);
  return select(vec3f(0.0), gradient / magnitude, magnitude > 1e-12);
}

/**
 * Whether a record's tape resolved at all.
 *
 * The evaluator answers 1e20 both for a block past capacity and for one that was
 * never filled, so a single probe separates a real program from an unresolved
 * reference: an authored tape has a finite distance everywhere. Probed once per
 * ray at the local origin rather than per march step, and the local origin is
 * the cheapest point that is guaranteed to be inside the record's own extent.
 */
fn svoFieldProgramResolved(reference: u32) -> bool {
  return svoFieldProgramReferenceSample(reference, vec3f(0.0)).distance_m < 1e19;
}

fn svoMarchedKind(kind: u32) -> bool {
  return kind == SVO_KIND_TORUS || kind == SVO_KIND_CONE || kind == SVO_KIND_ROUND_CONE || kind == SVO_KIND_ROUNDED_CYLINDER || kind == SVO_KIND_SMOOTH_UNION_CLUSTER || kind == SVO_KIND_FIELD_PROGRAM;
}

fn svoMarchedDistance_m(kind: u32, point: vec3f, dimensions_m: vec3f, packing: SvoClusterPacking, fieldProgramReference: u32) -> f32 {
  if (kind == SVO_KIND_TORUS) { return svoTorusDistance_m(point, dimensions_m); }
  if (kind == SVO_KIND_SMOOTH_UNION_CLUSTER) { return svoClusterDistance_m(point, dimensions_m, packing); }
  if (kind == SVO_KIND_FIELD_PROGRAM) { return svoFieldProgramDistance_m(fieldProgramReference, point); }
  if (kind == SVO_KIND_ROUND_CONE) { return svoRoundConeDistance_m(point,dimensions_m); }
  if (kind == SVO_KIND_ROUNDED_CYLINDER) { return svoRoundedCylinderDistance_m(point,dimensions_m); }
  return svoConeSample(point, dimensions_m).distance_m;
}

fn svoMarchedLocalNormal(kind: u32, point: vec3f, dimensions_m: vec3f, packing: SvoClusterPacking, fieldProgramReference: u32) -> vec4f {
  if (kind == SVO_KIND_TORUS) { return vec4f(svoTorusNormal(point, dimensions_m), f32(SVO_FEATURE_SMOOTH)); }
  if (kind == SVO_KIND_SMOOTH_UNION_CLUSTER) {
    return vec4f(svoClusterLocalNormal(point, dimensions_m, packing), f32(SVO_FEATURE_SMOOTH));
  }
  if (kind == SVO_KIND_FIELD_PROGRAM) {
    return vec4f(svoFieldProgramNormal(fieldProgramReference, point, dimensions_m), f32(SVO_FEATURE_SMOOTH));
  }
  if (kind == SVO_KIND_ROUND_CONE) { return vec4f(svoRoundConeNormal(point,dimensions_m),f32(SVO_FEATURE_SMOOTH)); }
  if (kind == SVO_KIND_ROUNDED_CYLINDER) { return vec4f(svoRoundedCylinderNormal(point,dimensions_m),f32(SVO_FEATURE_SMOOTH)); }
  let cone = svoConeSample(point, dimensions_m);
  return vec4f(cone.normal, f32(cone.featureId));
}

/** Rotation-invariant radius that bounds a marched kind about its own centre. */
fn svoMarchedBoundingRadius_m(kind: u32, dimensions_m: vec3f) -> f32 {
  if (kind == SVO_KIND_TORUS) { return dimensions_m.x + dimensions_m.y; }
  // A warp displaces the evaluation point by at most its amplitude per
  // component, so a field program's dimensions bound a *box* and the sphere that
  // contains it is the one through its corner. Reading them as an ellipsoid's
  // half-axes — which is right for the aggregate below — would start the march
  // inside the shape on a diagonal ray and clip the silhouette.
  if (kind == SVO_KIND_FIELD_PROGRAM) { return length(dimensions_m); }
  // An ellipsoidal lobe is contained in a sphere of its longest half-axis, not
  // of its corner: the corner belongs to the box around it, and using it here
  // would start every march that much earlier for nothing.
  if (kind == SVO_KIND_SMOOTH_UNION_CLUSTER) { return max(dimensions_m.x, max(dimensions_m.y, dimensions_m.z)); }
  if (kind == SVO_KIND_ROUND_CONE) { return dimensions_m.y + max(dimensions_m.x,dimensions_m.z); }
  if (kind == SVO_KIND_ROUNDED_CYLINDER) { return length(dimensions_m.xy); }
  return length(vec2f(max(dimensions_m.x, dimensions_m.z), dimensions_m.y));
}

/** Surface acceptance band, shared with the CPU so both solves stop in the same place. */
fn svoMarchSurfaceEpsilon_m(t_m: f32) -> f32 { return max(1e-6, 1e-4 * abs(t_m)); }

/**
 * Bounded ray hit for every finite primitive kind in the shared ABI: a closed
 * form where one exists, and otherwise a sphere trace of the kind's own exact
 * distance. Stepping by |distance| walks to the surface from either side, so a
 * ray that starts inside reports the same far surface a closed form would.
 */
fn svoIntersectPrimitiveExact(
  record: SvoPrimitiveRecord,
  worldOrigin_m: vec3f,
  worldDirectionIn: vec3f,
  tMin_m: f32,
  tMax_m: f32,
  packing: SvoClusterPacking,
) -> SvoPrimitiveRayResult {
  let directionLength = length(worldDirectionIn);
  let orientationLength = length(record.orientation);
  if (!(directionLength > 1e-8) || !(orientationLength > 1e-8) || !(tMin_m >= 0.0) || !(tMax_m >= tMin_m)) {
    return svoPrimitiveNoRayHit(SVO_PRIMITIVE_RAY_INVALID);
  }
  let worldDirection = worldDirectionIn / directionLength;
  let q = record.orientation / orientationLength;
  let inverse = vec4f(-q.xyz, q.w);
  let localOrigin = svoQuaternionRotate(inverse, worldOrigin_m - svoPrimitiveCenter_m(record));
  let localDirection = svoQuaternionRotate(inverse, worldDirection);
  let dimensions_m = svoPrimitiveDimensions_m(record);
  let kind = svoPrimitiveKind(record);
  let fieldProgramReference = svoPrimitiveFieldProgramReference(record);
  let marched = svoMarchedKind(kind);
  let finiteKind = (kind >= SVO_KIND_SPHERE && kind <= SVO_KIND_ELLIPSOID) || marched;
  var dimensionsValid = false;
  if (kind == SVO_KIND_BOX || kind == SVO_KIND_ELLIPSOID) { dimensionsValid = all(dimensions_m > vec3f(0.0)); }
  else if (kind == SVO_KIND_SPHERE) { dimensionsValid = dimensions_m.x > 0.0; }
  else if (kind == SVO_KIND_CAPSULE) { dimensionsValid = dimensions_m.x > 0.0 && dimensions_m.y >= 0.0; }
  else if (kind == SVO_KIND_CYLINDER) { dimensionsValid = dimensions_m.x > 0.0 && dimensions_m.y > 0.0; }
  else if (kind == SVO_KIND_TORUS) { dimensionsValid = dimensions_m.y > 0.0 && dimensions_m.y < dimensions_m.x; }
  else if (kind == SVO_KIND_CONE || kind == SVO_KIND_ROUND_CONE) {
    dimensionsValid = dimensions_m.y > 0.0 && dimensions_m.x >= 0.0 && dimensions_m.z >= 0.0
      && max(dimensions_m.x, dimensions_m.z) > 0.0;
    if(kind==SVO_KIND_ROUND_CONE){dimensionsValid=dimensionsValid&&2.0*dimensions_m.y>abs(dimensions_m.z-dimensions_m.x)*1.000001;}
  }
  else if (kind == SVO_KIND_ROUNDED_CYLINDER) {
    dimensionsValid=dimensions_m.x>0.0&&dimensions_m.y>0.0&&dimensions_m.z>0.0&&dimensions_m.z<=min(dimensions_m.x,dimensions_m.y);
  }
  // A cluster whose arena block did not resolve is invalid, not empty: falling
  // through to a miss would draw a hole where the aggregate is and say nothing.
  else if (kind == SVO_KIND_SMOOTH_UNION_CLUSTER) {
    dimensionsValid = all(dimensions_m > vec3f(0.0)) && svoClusterPackingValid(packing);
  }
  // Same refusal as the aggregate's, one kind over: a tape whose arena block did
  // not resolve is invalid, not empty. Falling through to a miss would draw a
  // hole where the shape is and say nothing anywhere.
  else if (kind == SVO_KIND_FIELD_PROGRAM) {
    dimensionsValid = all(dimensions_m > vec3f(0.0)) && svoFieldProgramResolved(fieldProgramReference);
  }
  if (!finiteKind || !dimensionsValid) { return svoPrimitiveNoRayHit(SVO_PRIMITIVE_RAY_INVALID); }

  var bestT_m = SVO_PRIMITIVE_RAY_INFINITY;
  var bestNormal = vec3f(0.0);
  var bestFeature = SVO_FEATURE_SMOOTH;

  if (kind == SVO_KIND_SPHERE) {
    let roots = svoPrimitiveQuadraticRoots(
      dot(localDirection, localDirection),
      dot(localOrigin, localDirection),
      dot(localOrigin, localOrigin) - dimensions_m.x * dimensions_m.x,
    );
    for (var rootIndex = 0u; rootIndex < 2u; rootIndex += 1u) {
      if (rootIndex >= roots.count) { break; }
      let candidate = roots.values[rootIndex];
      if (svoPrimitiveRayInRange(candidate, tMin_m, tMax_m)) {
        bestT_m = max(tMin_m, candidate);
        bestNormal = normalize(localOrigin + localDirection * bestT_m);
        break;
      }
    }
  } else if (kind == SVO_KIND_ELLIPSOID) {
    let scaledOrigin = localOrigin / dimensions_m;
    let scaledDirection = localDirection / dimensions_m;
    let roots = svoPrimitiveQuadraticRoots(
      dot(scaledDirection, scaledDirection),
      dot(scaledOrigin, scaledDirection),
      dot(scaledOrigin, scaledOrigin) - 1.0,
    );
    for (var rootIndex = 0u; rootIndex < 2u; rootIndex += 1u) {
      if (rootIndex >= roots.count) { break; }
      let candidate = roots.values[rootIndex];
      if (svoPrimitiveRayInRange(candidate, tMin_m, tMax_m)) {
        bestT_m = max(tMin_m, candidate);
        let point_m = localOrigin + localDirection * bestT_m;
        bestNormal = normalize(point_m / (dimensions_m * dimensions_m));
        break;
      }
    }
  } else if (kind == SVO_KIND_BOX) {
    var enter = -SVO_PRIMITIVE_RAY_INFINITY;
    var exit = SVO_PRIMITIVE_RAY_INFINITY;
    var enterAxis = 0u;
    var exitAxis = 0u;
    var valid = true;
    for (var axis = 0u; axis < 3u; axis += 1u) {
      if (abs(localDirection[axis]) <= 1e-8) {
        if (localOrigin[axis] < -dimensions_m[axis] || localOrigin[axis] > dimensions_m[axis]) { valid = false; }
      } else {
        let first = (-dimensions_m[axis] - localOrigin[axis]) / localDirection[axis];
        let second = (dimensions_m[axis] - localOrigin[axis]) / localDirection[axis];
        let nearT = min(first, second);
        let farT = max(first, second);
        if (nearT > enter) { enter = nearT; enterAxis = axis; }
        if (farT < exit) { exit = farT; exitAxis = axis; }
        if (exit < enter) { valid = false; }
      }
    }
    let useEnter = svoPrimitiveRayInRange(enter, tMin_m, tMax_m);
    let candidate = select(exit, enter, useEnter);
    let featureAxis = select(exitAxis, enterAxis, useEnter);
    if (valid && svoPrimitiveRayInRange(candidate, tMin_m, tMax_m)) {
      bestT_m = max(tMin_m, candidate);
      let point_m = localOrigin + localDirection * bestT_m;
      bestNormal[featureAxis] = select(-1.0, 1.0, point_m[featureAxis] >= 0.0);
      bestFeature = SVO_FEATURE_BOX_X + featureAxis;
    }
  } else if (marched) {
    // The bounding sphere both rejects misses early and bounds the march to the
    // part of the interval that can contain this primitive at all.
    let boundingRadius_m = svoMarchedBoundingRadius_m(kind, dimensions_m);
    let bounds = svoPrimitiveQuadraticRoots(
      dot(localDirection, localDirection),
      dot(localOrigin, localDirection),
      dot(localOrigin, localOrigin) - boundingRadius_m * boundingRadius_m,
    );
    if (bounds.count == 2u) {
      let start = max(tMin_m, bounds.values[0]);
      let end = min(tMax_m, bounds.values[1]);
      if (end >= start) {
        var marchT_m = start;
        for (var iteration = 0u; iteration <= ${SVO_PRIMITIVE_MARCH_ITERATIONS}u; iteration += 1u) {
          let point_m = localOrigin + localDirection * marchT_m;
          let distance_m = abs(svoMarchedDistance_m(kind, point_m, dimensions_m, packing, fieldProgramReference));
          if (distance_m <= svoMarchSurfaceEpsilon_m(marchT_m)) {
            let local = svoMarchedLocalNormal(kind, point_m, dimensions_m, packing, fieldProgramReference);
            if (length(local.xyz) > 1e-8) {
              bestT_m = marchT_m;
              bestNormal = local.xyz;
              bestFeature = u32(local.w);
            }
            break;
          }
          // The floor only guarantees progress; the acceptance band above is
          // what decides a hit, so it never steps across a surface.
          marchT_m += max(distance_m, 1e-8);
          if (marchT_m > end) { break; }
        }
      }
    }
  } else {
    let radialRoots = svoPrimitiveQuadraticRoots(
      dot(localDirection.xz, localDirection.xz),
      dot(localOrigin.xz, localDirection.xz),
      dot(localOrigin.xz, localOrigin.xz) - dimensions_m.x * dimensions_m.x,
    );
    for (var rootIndex = 0u; rootIndex < 2u; rootIndex += 1u) {
      if (rootIndex >= radialRoots.count) { break; }
      let candidate = radialRoots.values[rootIndex];
      let y_m = localOrigin.y + localDirection.y * candidate;
      if (abs(y_m) <= dimensions_m.y && svoPrimitiveRayInRange(candidate, tMin_m, tMax_m) && candidate < bestT_m) {
        bestT_m = max(tMin_m, candidate);
        let point_m = localOrigin + localDirection * bestT_m;
        bestNormal = normalize(vec3f(point_m.x, 0.0, point_m.z));
        bestFeature = select(SVO_FEATURE_CYLINDER_SIDE, SVO_FEATURE_SMOOTH, kind == SVO_KIND_CAPSULE);
      }
    }
    if (kind == SVO_KIND_CAPSULE) {
      for (var capIndex = 0u; capIndex < 2u; capIndex += 1u) {
        let capSign = select(-1.0, 1.0, capIndex != 0u);
        let capCenter = vec3f(0.0, capSign * dimensions_m.y, 0.0);
        let offset = localOrigin - capCenter;
        let roots = svoPrimitiveQuadraticRoots(
          dot(localDirection, localDirection), dot(offset, localDirection),
          dot(offset, offset) - dimensions_m.x * dimensions_m.x,
        );
        for (var rootIndex = 0u; rootIndex < 2u; rootIndex += 1u) {
          if (rootIndex >= roots.count) { break; }
          let candidate = roots.values[rootIndex];
          let normalPoint = offset + localDirection * candidate;
          if (capSign * normalPoint.y >= 0.0 && svoPrimitiveRayInRange(candidate, tMin_m, tMax_m) && candidate < bestT_m) {
            bestT_m = max(tMin_m, candidate);
            bestNormal = normalize(offset + localDirection * bestT_m);
          }
        }
      }
    } else if (abs(localDirection.y) > 1e-8) {
      for (var capIndex = 0u; capIndex < 2u; capIndex += 1u) {
        let capSign = select(-1.0, 1.0, capIndex != 0u);
        let candidate = (capSign * dimensions_m.y - localOrigin.y) / localDirection.y;
        let point_m = localOrigin + localDirection * candidate;
        let tieTolerance_m = 8e-6 * max(1.0, abs(candidate));
        if (dot(point_m.xz, point_m.xz) <= dimensions_m.x * dimensions_m.x * 1.000008
          && svoPrimitiveRayInRange(candidate, tMin_m, tMax_m) && candidate <= bestT_m + tieTolerance_m) {
          bestT_m = max(tMin_m, candidate);
          bestNormal = vec3f(0.0, capSign, 0.0);
          bestFeature = SVO_FEATURE_CYLINDER_CAP;
        }
      }
    }
  }

  if (!(bestT_m < SVO_PRIMITIVE_RAY_INFINITY)) { return svoPrimitiveNoRayHit(SVO_PRIMITIVE_RAY_MISS); }
  let worldNormal = normalize(svoQuaternionRotate(q, bestNormal));
  return SvoPrimitiveRayResult(bestT_m, bestFeature, SVO_PRIMITIVE_RAY_HIT, 0u, vec4f(worldNormal, 0.0));
}

fn svoBoxDistance_m(point: vec3f, halfExtents_m: vec3f) -> f32 {
  let q = abs(point) - halfExtents_m;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn svoCapsuleDistance_m(point: vec3f, dimensions_m: vec3f) -> f32 {
  let closestY = clamp(point.y, -dimensions_m.y, dimensions_m.y);
  return length(vec3f(point.x, point.y - closestY, point.z)) - dimensions_m.x;
}

fn svoCylinderDistance_m(point: vec3f, dimensions_m: vec3f) -> f32 {
  let q = vec2f(length(point.xz) - dimensions_m.x, abs(point.y) - dimensions_m.y);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

fn svoEllipsoidEquation2(extents_m: vec2f, point_m: vec2f, lambda: f32) -> f32 {
  let squared = extents_m * extents_m;
  let ratio = extents_m * point_m / (squared + vec2f(lambda));
  return dot(ratio, ratio) - 1.0;
}

fn svoEllipsoidEquation3(extents_m: vec3f, point_m: vec3f, lambda: f32) -> f32 {
  let squared = extents_m * extents_m;
  let ratio = extents_m * point_m / (squared + vec3f(lambda));
  return dot(ratio, ratio) - 1.0;
}

// xy is the positive-orthant closest point; z marks an ambiguous medial-axis normal.
fn svoEllipsoidClosest2(extents_m: vec2f, point_m: vec2f) -> vec3f {
  let tolerance = 1e-6 * max(1.0, extents_m.x);
  if (point_m.y <= tolerance) {
    let denominator = extents_m.x * extents_m.x - extents_m.y * extents_m.y;
    if (abs(denominator) > 1e-6 * max(1.0, extents_m.x * extents_m.x)) {
      let candidateX = extents_m.x * extents_m.x * point_m.x / denominator;
      let surfaceSum = candidateX * candidateX / (extents_m.x * extents_m.x);
      if (surfaceSum <= 1.000004) {
        let candidateY = extents_m.y * sqrt(max(0.0, 1.0 - surfaceSum));
        return vec3f(candidateX, candidateY, select(0.0, 1.0, candidateY > tolerance));
      }
    } else if (point_m.x <= tolerance) {
      return vec3f(0.0, extents_m.y, 1.0);
    }
    return vec3f(extents_m.x, 0.0, select(0.0, 1.0, point_m.x <= tolerance));
  }
  let equationAtZero = svoEllipsoidEquation2(extents_m, point_m, 0.0);
  if (abs(equationAtZero) <= 4e-6) { return vec3f(point_m, 0.0); }
  var lower = select(0.0, -extents_m.y * extents_m.y, equationAtZero < 0.0);
  var upper = select(max(1.0, extents_m.x * length(point_m)), 0.0, equationAtZero < 0.0);
  for (var iteration = 0u; iteration < ${SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS}u; iteration += 1u) {
    let middle = 0.5 * (lower + upper);
    if (svoEllipsoidEquation2(extents_m, point_m, middle) > 0.0) { lower = middle; } else { upper = middle; }
  }
  let lambda = 0.5 * (lower + upper);
  let squared = extents_m * extents_m;
  return vec3f(squared * point_m / (squared + vec2f(lambda)), 0.0);
}

// xyz is the signed-octant closest point; w marks an ambiguous medial-axis normal.
fn svoEllipsoidClosestPoint_m(radii_m: vec3f, point_m: vec3f) -> vec4f {
  var extents_m = radii_m;
  var positivePoint_m = abs(point_m);
  var axes = vec3u(0u, 1u, 2u);
  if (extents_m.x < extents_m.y) {
    let extent = extents_m.x; extents_m.x = extents_m.y; extents_m.y = extent;
    let coordinate = positivePoint_m.x; positivePoint_m.x = positivePoint_m.y; positivePoint_m.y = coordinate;
    let axis = axes.x; axes.x = axes.y; axes.y = axis;
  }
  if (extents_m.y < extents_m.z) {
    let extent = extents_m.y; extents_m.y = extents_m.z; extents_m.z = extent;
    let coordinate = positivePoint_m.y; positivePoint_m.y = positivePoint_m.z; positivePoint_m.z = coordinate;
    let axis = axes.y; axes.y = axes.z; axes.z = axis;
  }
  if (extents_m.x < extents_m.y) {
    let extent = extents_m.x; extents_m.x = extents_m.y; extents_m.y = extent;
    let coordinate = positivePoint_m.x; positivePoint_m.x = positivePoint_m.y; positivePoint_m.y = coordinate;
    let axis = axes.x; axes.x = axes.y; axes.y = axis;
  }

  let tolerance = 1e-6 * max(1.0, extents_m.x);
  var sortedClosest_m = vec3f(0.0);
  var ambiguous = 0.0;
  if (positivePoint_m.z <= tolerance) {
    let squared = extents_m * extents_m;
    let denominator = squared.xy - vec2f(squared.z);
    var reducedValid = true;
    var candidate = vec2f(0.0);
    for (var axis = 0u; axis < 2u; axis += 1u) {
      if (abs(denominator[axis]) <= 1e-6 * max(1.0, squared[axis])) {
        if (positivePoint_m[axis] > tolerance) { reducedValid = false; }
      } else {
        candidate[axis] = squared[axis] * positivePoint_m[axis] / denominator[axis];
      }
    }
    let surfaceSum = dot(candidate / extents_m.xy, candidate / extents_m.xy);
    if (reducedValid && surfaceSum <= 1.000004) {
      let candidateZ = extents_m.z * sqrt(max(0.0, 1.0 - surfaceSum));
      sortedClosest_m = vec3f(candidate, candidateZ);
      ambiguous = select(0.0, 1.0, candidateZ > tolerance);
    } else {
      let reduced = svoEllipsoidClosest2(extents_m.xy, positivePoint_m.xy);
      sortedClosest_m = vec3f(reduced.xy, 0.0);
      ambiguous = reduced.z;
    }
  } else {
    let equationAtZero = svoEllipsoidEquation3(extents_m, positivePoint_m, 0.0);
    if (abs(equationAtZero) <= 4e-6) {
      sortedClosest_m = positivePoint_m;
    } else {
      var lower = select(0.0, -extents_m.z * extents_m.z, equationAtZero < 0.0);
      var upper = select(max(1.0, extents_m.x * length(positivePoint_m)), 0.0, equationAtZero < 0.0);
      for (var iteration = 0u; iteration < ${SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS}u; iteration += 1u) {
        let middle = 0.5 * (lower + upper);
        if (svoEllipsoidEquation3(extents_m, positivePoint_m, middle) > 0.0) { lower = middle; } else { upper = middle; }
      }
      let lambda = 0.5 * (lower + upper);
      let squared = extents_m * extents_m;
      sortedClosest_m = squared * positivePoint_m / (squared + vec3f(lambda));
    }
  }
  var closest_m = vec3f(0.0);
  closest_m[axes.x] = select(-sortedClosest_m.x, sortedClosest_m.x, point_m[axes.x] >= 0.0);
  closest_m[axes.y] = select(-sortedClosest_m.y, sortedClosest_m.y, point_m[axes.y] >= 0.0);
  closest_m[axes.z] = select(-sortedClosest_m.z, sortedClosest_m.z, point_m[axes.z] >= 0.0);
  return vec4f(closest_m, ambiguous);
}

fn svoEllipsoidDistance_m(point: vec3f, radii_m: vec3f) -> f32 {
  if (any(radii_m <= vec3f(0.0))) { return 3.402823e38; }
  let closest = svoEllipsoidClosestPoint_m(radii_m, point);
  let distance_m = length(point - closest.xyz);
  return select(distance_m, -distance_m, dot(point / radii_m, point / radii_m) < 1.0);
}

fn svoPrimitiveDistance_m(record: SvoPrimitiveRecord, worldPoint_m: vec3f, terrainHeight_m: f32, packing: SvoClusterPacking) -> f32 {
  let kind = svoPrimitiveKind(record);
  let dimensions_m = svoPrimitiveDimensions_m(record);
  if (kind == SVO_KIND_TERRAIN) { return worldPoint_m.y - terrainHeight_m; }
  let point = svoPrimitiveLocalPoint(record, worldPoint_m);
  if (kind == SVO_KIND_SPHERE) { return length(point) - dimensions_m.x; }
  if (kind == SVO_KIND_BOX) { return svoBoxDistance_m(point, dimensions_m); }
  if (kind == SVO_KIND_CAPSULE) { return svoCapsuleDistance_m(point, dimensions_m); }
  if (kind == SVO_KIND_CYLINDER) { return svoCylinderDistance_m(point, dimensions_m); }
  if (kind == SVO_KIND_ELLIPSOID) { return svoEllipsoidDistance_m(point, dimensions_m); }
  if (svoMarchedKind(kind)) {
    return svoMarchedDistance_m(kind, point, dimensions_m, packing, svoPrimitiveFieldProgramReference(record));
  }
  return 3.402823e38;
}

fn svoBoxFeatureNormal(point: vec3f, halfExtents_m: vec3f) -> vec4f {
  let q = abs(point) - halfExtents_m;
  var axis = 0u;
  if (q.y > q.x) { axis = 1u; }
  if (q.z > q[axis]) { axis = 2u; }
  var normal = vec3f(0.0);
  normal[axis] = select(-1.0, 1.0, point[axis] >= 0.0);
  return vec4f(normal, f32(SVO_FEATURE_BOX_X + axis));
}

fn svoCylinderFeatureNormal(point: vec3f, dimensions_m: vec3f) -> vec4f {
  let radialDistance = length(point.xz) - dimensions_m.x;
  let capDistance = abs(point.y) - dimensions_m.y;
  if (capDistance >= radialDistance) {
    return vec4f(0.0, select(-1.0, 1.0, point.y >= 0.0), 0.0, f32(SVO_FEATURE_CYLINDER_CAP));
  }
  let radial = point.xz / max(length(point.xz), 1e-8);
  return vec4f(radial.x, 0.0, radial.y, f32(SVO_FEATURE_CYLINDER_SIDE));
}

fn svoPrimitiveLocalNormal(record: SvoPrimitiveRecord, point: vec3f, packing: SvoClusterPacking) -> vec4f {
  let kind = svoPrimitiveKind(record);
  let dimensions_m = svoPrimitiveDimensions_m(record);
  if (kind == SVO_KIND_SPHERE) { return vec4f(normalize(point), f32(SVO_FEATURE_SMOOTH)); }
  if (kind == SVO_KIND_BOX) { return svoBoxFeatureNormal(point, dimensions_m); }
  if (kind == SVO_KIND_CAPSULE) {
    let closestY = clamp(point.y, -dimensions_m.y, dimensions_m.y);
    return vec4f(normalize(vec3f(point.x, point.y - closestY, point.z)), f32(SVO_FEATURE_SMOOTH));
  }
  if (kind == SVO_KIND_CYLINDER) { return svoCylinderFeatureNormal(point, dimensions_m); }
  if (svoMarchedKind(kind)) {
    return svoMarchedLocalNormal(kind, point, dimensions_m, packing, svoPrimitiveFieldProgramReference(record));
  }
  if (kind == SVO_KIND_ELLIPSOID) {
    if (any(dimensions_m <= vec3f(0.0))) { return vec4f(0.0); }
    let closest = svoEllipsoidClosestPoint_m(dimensions_m, point);
    let delta = point - closest.xyz;
    let distance_m = length(delta);
    let surfaceTolerance_m = 1e-6 * max(1.0, max(dimensions_m.x, max(dimensions_m.y, dimensions_m.z)));
    if (distance_m <= surfaceTolerance_m) {
      return vec4f(normalize(point / (dimensions_m * dimensions_m)), f32(SVO_FEATURE_SMOOTH));
    }
    if (closest.w > 0.5) { return vec4f(0.0); }
    let outward = select(delta, -delta, dot(point / dimensions_m, point / dimensions_m) < 1.0);
    return vec4f(normalize(outward), f32(SVO_FEATURE_SMOOTH));
  }
  return vec4f(0.0);
}

fn svoEvaluatePrimitive(record: SvoPrimitiveRecord, worldPoint_m: vec3f, terrainHeight_m: f32, terrainNormal: vec3f, packing: SvoClusterPacking) -> SvoPrimitiveSample {
  let distance_m = svoPrimitiveDistance_m(record, worldPoint_m, terrainHeight_m, packing);
  if (svoPrimitiveKind(record) == SVO_KIND_TERRAIN) {
    let normalLength = length(terrainNormal);
    return SvoPrimitiveSample(distance_m, SVO_FEATURE_TERRAIN, select(0u, SVO_SAMPLE_NORMAL_VALID, normalLength > 1e-8), 0u, vec4f(terrainNormal / max(normalLength, 1e-8), 0.0));
  }
  let localPoint = svoPrimitiveLocalPoint(record, worldPoint_m);
  let local = svoPrimitiveLocalNormal(record, localPoint, packing);
  let localLength = length(local.xyz);
  let worldNormal = svoQuaternionRotate(record.orientation, local.xyz / max(localLength, 1e-8));
  return SvoPrimitiveSample(distance_m, u32(local.w), select(0u, SVO_SAMPLE_NORMAL_VALID, localLength > 1e-8), 0u, vec4f(worldNormal, 0.0));
}
`;
