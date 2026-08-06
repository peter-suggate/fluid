import type { Quaternion, RigidBodyDescription, Vec3 } from "../model";
import { quaternionMultiply } from "../rigid-body";
import type { SceneryMaterial, SceneryNode, SceneryPlacement } from "../scenery-graph";
import { SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES } from "../svo-primitive-abi";
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
 * **Three bands, authored separately, measured against the cell.** The set's
 * surface is not one "weathering" knob; it is three scales that the pipeline
 * treats completely differently, and conflating them is what produced a bank of
 * melted lumps. Measured off `artifacts/plate-crops/boulders.png` against
 * `HERO_GARDEN_CELL_M` = 25 mm:
 *
 *  1. **Silhouette, 100 mm and up.** The oblate cap, the tapered stem, the
 *     plinth slab under them. Primitives, one record each. Under voxel-only
 *     shading this is nearly the whole of what the eye gets: the stone reads by
 *     its outline and by the value step between its parts.
 *  2. **Form, 14-80 mm at 4-8 mm of amplitude.** A slow undulation that stops
 *     the cap's horizon being a conic. {@link stoneMass} carries it, and it is
 *     authored as an absolute length rather than as a share of the stone,
 *     because the thing it has to clear is the cell.
 *
 *     **It is a ladder rather than one period now, and the ladder is the leaf's.**
 *     A lattice field stacks octaves — each a halving of the period, the lobe and
 *     the blend together — inside the *same one record*, so
 *     {@link stoneFormOctaves} spends as many of them as the leaf can draw: one
 *     55 mm undulation at a 25 mm leaf, 55 and 27.5 at 6.25 mm, 55 / 27.5 / 13.75
 *     at 3.125 mm and finer. Nothing about the silhouette, the record count or
 *     the per-brick census moves; what moves is how fine the surface under it is.
 *     {@link stoneFormMinimumSpan_m} follows it down, so smaller stones join the
 *     band as the lattice refines rather than the beds staying bare ellipsoids
 *     forever against a floor baked at 25 mm.
 *  3. **Grain, ~3 mm pitting.** Dense shallow concavities, *subtractive* — the
 *     plate is pocked, not lumpy, and an additive octave reads as plaster. At
 *     25 mm cells this was an eighth of a cell and could appear nowhere; at the
 *     0.78 mm leaf the tree now reaches it is 3.8 leaves and is the right band to
 *     draw. It is declared in {@link STONE_GRAIN_TAPE} and **still emitted by
 *     nothing**, because `SceneryNode` has no `field-program` member — the one
 *     remaining blocker of the three that note used to list. See it.
 *
 * **The old lattice drew the envelope exactly, and that is measured.** Every
 * mass here used to be a `cluster` whose lattice was authored as *lobes across
 * the stone* with a lobe radius of 0.98 of the period. Sampling the published
 * packing round its own equator (`sampleSvoPrimitive`, 1 440 rays, bisected to
 * 0.1 µm) says the drawn horizon departs from the ellipsoid envelope by **0.0
 * mm at every seed and every lobe count**: at 0.98 the lattice covers space
 * outright — the covering threshold is √3/2 = 0.866 — so the hard `max` against
 * the envelope is the whole shape and the field contributes nothing. Two
 * hundred and eight marched records were buying a sphere trace against an
 * ellipsoid that a closed-form root already gives. That is the actual root of
 * "the boulders read as melted lumps": they were ellipsoids, and so was every
 * pebble.
 *
 * So the lobe radius is now **under** the covering threshold and the period is
 * a length rather than a count, and the band a stone gets is decided by whether
 * the cell can show it. Below {@link stoneFormMinimumSpan_m} a stone is
 * published as the ellipsoid it draws anyway, which is honest and costs a
 * closed-form root instead of forty-eight march steps.
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
 * **The seam.** Every solid mass in this quarry — a cap, a plinth, a cobble —
 * is built here and nowhere else, so the whole set changes shape together, and
 * so that band 2 is one measured set of numbers rather than a knob per species.
 *
 * It carries **the form band and nothing else**. The silhouette above it is the
 * caller's primitives; the grain below it is {@link STONE_GRAIN_TAPE} and has
 * no emitter. What is here is a jittered sphere lattice whose period is a
 * *length in metres* — 55 mm, a little over two cells — clipped to the
 * ellipsoid the stone would otherwise have been.
 *
 * The period being absolute is the whole correction. It used to be authored as
 * `lobesAcross`, a count, on the reasoning that weathering is a proportion of a
 * stone rather than a fixed size. That is true of a real stone and false of a
 * *rendered* one: what decides whether an undulation appears is its size
 * against the cell, so a count made the boulders' features three cells wide and
 * the shingle's a twentieth of one, and only the first could ever have shown.
 *
 * Below {@link STONE_FORM_MINIMUM_SPAN_M} the stone cannot carry a period at
 * all and is published as the ellipsoid it draws anyway. That is not a
 * concession: at 25 mm cells a 40 mm pebble is under two cells across, and a
 * marched field on it is forty-eight distance evaluations spent on a feature
 * that cannot survive the resample.
 */
interface StoneMassSpec {
  readonly id: string;
  readonly group: string;
  readonly tags: readonly string[];
  /** The whole placement, so a bedded stone keeps its terrain anchor and its lie. */
  readonly place: SceneryPlacement;
  /** Half-axes of the envelope. The drawn solid is inside it by construction. */
  readonly radius: Vec3;
  readonly seed: number;
  readonly material: SceneryMaterial;
  /**
   * The finest voxel this stone will be drawn at, in metres.
   *
   * Placement, not form — see `SceneryGeneratorRequest.detailCellSize_m`. It
   * decides two things and nothing else: how many octaves of the band the stone
   * carries, and how small a stone gets one at all. Absent is
   * {@link STONE_DEFAULT_LEAF_SIZE_M}, which reproduces the 25 mm set exactly.
   */
  readonly leafSize_m?: number;
}

/**
 * The leaf a stone is drawn at when nobody says, in metres.
 *
 * `HERO_GARDEN_CELL_M`, which is the lattice every number in this file's band
 * table was measured against. A default rather than a constant of the species:
 * a caller that knows its own leaf passes it and the whole band ladder moves.
 */
export const STONE_DEFAULT_LEAF_SIZE_M = 0.025;

/**
 * Leaves a feature must span before it renders as that feature.
 *
 * The project-wide band law — see `bonsai.ts`'s header for the derivation and
 * `LEGIBLE_FEATURE_LEAVES` there for the same constant. Shading is the voxel
 * cell's surface, so under about two leaves a period comes back as aliasing and
 * at three it starts being geometry.
 */
const LEGIBLE_FEATURE_LEAVES = 3;

/**
 * The form band's period, in metres.
 *
 * 55 mm: 2.2 cells at `HERO_GARDEN_CELL_M`, and the middle of the 40-80 mm the
 * plate's caps show. Under about two cells a period is at the resample's
 * Nyquist limit and comes back as aliasing rather than as form; much over three
 * and there is only one undulation on a cap and the horizon is a conic again
 * with a dent in it.
 */
const STONE_FORM_PERIOD_M = 0.055;

/**
 * Lattice sphere radius as a fraction of the period. **The measured operating
 * point, not a taste choice, and the number the whole band lives or dies on.**
 *
 * `sqrt(3)/2 = 0.866` is where every cell corner falls inside a sphere and the
 * union covers space outright — above it the hard `max` against the envelope is
 * the entire shape and the field draws a perfect ellipsoid, which is exactly
 * what the old 0.98 did and exactly why the caps read as melted. Under it the
 * corners open into concavities and the drawn horizon finally leaves the
 * envelope.
 *
 * How far under is a genuine trade and it was resolved by sampling **the
 * published nodes**, not a synthetic envelope: three boulder caps, 720 rays
 * round each equator bisected against `sampleSvoPrimitive`, residual taken
 * against the envelope ellipse rather than against the mean radius so the
 * envelope's own ovality is not mistaken for form.
 *
 *   floret | cap 164 mm       | cap 136 mm        | cap 94 mm
 *   -------|------------------|-------------------|------------------
 *   0.98   | 0.0 mm           | 0.0 mm            | 0.0 mm
 *   0.72   | 1.0 mm  (0.6 %)  | 12.6 mm  (3.3 %)  | 0.3 mm  (0.0 %)
 *   0.70   | 5.4 mm  (1.9 %)  | 22.1 mm  (6.5 %)  | 8.0 mm  (1.8 %)
 *   0.69   | 7.7 mm  (1.7 %)  | 24.6 mm  (9.0 %)  | 11.7 mm (4.9 %)
 *   0.68   | 12.5 mm (3.5 %)  | 27.0 mm (11.0 %)  | 15.3 mm (7.0 %)
 *   0.66   | 27.0 mm (6.4 %)  | 33.2 mm (17.3 %)  | 21.7 mm (15.7 %)
 *
 * where the first figure is the horizon's amplitude and the bracket is the
 * share of the stone's interior left in sealed voids.
 *
 * **0.72 is rejected and the reason is the most useful thing in this table.**
 * On paper it is the safe choice — small inset, almost no voids — and on two of
 * the three caps it produces *one millimetre*, a twenty-fifth of a cell, which
 * under cell-face shading is indistinguishable from the ellipsoid it was
 * supposed to replace. The lattice is anchored on the node's own origin and only
 * three periods span a cap, so whether an uncovered corner lands near the
 * equator is close to a coin toss: 12.6 mm on one stone and 0.3 mm on its
 * neighbour, from the same numbers. A band that works on a third of the set is
 * not a band.
 *
 * 0.69 is where every cap clears a third of a cell — 0.31, 0.98 and 0.47 — which
 * is the least that can move a voxelized horizon, and it is taken *because* the
 * lottery exists rather than in spite of it: the number has to be low enough
 * that the unluckiest phase still shows. What it costs is an inset of 2.6 to
 * 11.7 mm, so a cap draws up to 9 % narrower than its authored envelope, and up
 * to 9 % of a stone's interior in voids. Neither is free and both were checked:
 * the inset is inside the clearance the boulder placements already carry to the
 * container wall, and the voids are sealed — a 64 x 64 plan-column sweep of all
 * three caps finds **no hole through any of them** at 0.66 or above, which is
 * the failure that would actually show.
 */
const STONE_FORM_LOBE_SHARE = 0.69;
/** Smooth-minimum radius, as a fraction of the sphere. Rounds the concavities without filling them. */
const STONE_FORM_BLEND_SHARE = 0.30;
/**
 * Sphere displacement from its own cell centre, as a fraction of the period.
 *
 * What stops the concavities being a *crystal*. At zero the uncovered corners
 * sit on a perfect cubic array and read as machining; the ABI's ceiling is 0.3
 * and the amplitude table above is measured at 0.24, which is where the array
 * has broken up and no two dents on one stone are the same depth.
 */
const STONE_FORM_JITTER = 0.24;

/**
 * The smallest plan diameter that got a form band when the leaf was 25 mm.
 *
 * Three cells. Two is the Nyquist floor for one period and leaves nothing over
 * for the crest either side of a trough, so three is the first width at which a
 * voxelized horizon can hold crest-trough-crest. It is a *plan* measure because
 * every stone here is flattened and its thickness is never the limit.
 *
 * At this floor the hero set's three larger boulder caps and the coarsest bed
 * cobbles carry the band, and the shingle, the plates' shoulders and the fine
 * courses do not. That is about a twentieth of the clusters this file used to
 * publish, for the same record count and the same drawn shape — see the header.
 *
 * **It is now the coarse end of a ladder rather than the rule.** Written as a
 * constant it was a bake of one lattice: 75 mm is `3 × 25 mm`, and at a 6.25 mm
 * leaf it went on excluding every stone under 75 mm from a band that four leaves
 * could by then have drawn easily. {@link stoneFormMinimumSpan_m} is the rule;
 * this is what it returns at {@link STONE_DEFAULT_LEAF_SIZE_M}, kept exported
 * because it is the number the table above was measured at.
 */
export const STONE_FORM_MINIMUM_SPAN_M = 0.075;

/**
 * How many of the finest octave's periods a stone must span to carry the band.
 *
 * Read off the pair the set was authored at rather than invented: 75 mm of
 * minimum span over a 55 mm period. It is a little over one, which is what
 * "crest-trough-crest" costs once the trough is allowed to straddle the stone's
 * own axis, and expressing the floor in *periods* is what lets the floor follow
 * the octave ladder instead of standing still while the leaf moves.
 */
const STONE_FORM_MINIMUM_SPAN_PERIODS = STONE_FORM_MINIMUM_SPAN_M / STONE_FORM_PERIOD_M;

/**
 * Octaves of the form band this leaf can draw, in
 * `1..{@link SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES}`.
 *
 * **This is where "detail follows the lattice" actually lives for the whole
 * quarry, and it costs zero records.** Octave `k` of a lattice field repeats on
 * `period·2^-k` with lobes and blend scaled with it (see
 * `SvoClusterLatticeField.octaves`), so a stone that carried one 55 mm
 * undulation at a 25 mm leaf carries 55 and 27.5 at 6.25 mm and 55, 27.5 and
 * 13.75 at 3.125 mm — inside **the same one record**, evaluated by the shader.
 * Nothing about the silhouette, the record count, the owner palette or the
 * per-brick census moves; what moves is how fine the surface under it is.
 *
 * An octave is admitted while its own period still clears
 * {@link LEGIBLE_FEATURE_LEAVES}, which is the same test every other band in
 * this tree is judged by. At the leaves the ladder is walked at:
 *
 *   leaf      octaves   finest period   floor
 *   25 mm        1          55.00 mm    75.0 mm
 *   12.5 mm      1          55.00 mm    75.0 mm
 *   6.25 mm      2          27.50 mm    37.5 mm
 *   3.125 mm     3          13.75 mm    18.8 mm
 *   1.5625 mm    4           6.88 mm     9.4 mm
 *   0.78125 mm   5           3.44 mm     4.7 mm
 *
 * The first row is today's set unchanged, which is the property that makes this
 * safe to land: a scene that never says what leaf it draws at gets exactly the
 * geometry it got before.
 *
 * **The bottom two rows used to read 13.75 mm as well**, because
 * `SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES` was three — a cost bound argued at a
 * 25 mm leaf, where "detail an eighth of the first octave's" really was below a
 * pixel. It was re-measured rather than raised on that objection: an octave
 * costs exactly one more evaluation and provably zero march steps, zero records
 * and zero per-brick owners, so the ceiling now sits at five, which is what this
 * ladder asks for at the finest leaf the octree can spend. The numbers are on
 * the constant.
 */
export function stoneFormOctaves(leafSize_m: number = STONE_DEFAULT_LEAF_SIZE_M): number {
  if (!(leafSize_m > 0)) throw new RangeError("Stone leaf size must be positive");
  const legible_m = LEGIBLE_FEATURE_LEAVES * leafSize_m;
  let octaves = 1;
  while (octaves < SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES
    && STONE_FORM_PERIOD_M * 2 ** -octaves >= legible_m) octaves += 1;
  return octaves;
}

/** The finest period the band carries at this leaf, in metres. */
export function stoneFormFinestPeriod_m(leafSize_m: number = STONE_DEFAULT_LEAF_SIZE_M): number {
  return STONE_FORM_PERIOD_M * 2 ** -(stoneFormOctaves(leafSize_m) - 1);
}

/**
 * The smallest plan diameter that gets a form band at this leaf, in metres.
 *
 * Two conditions, and the binding one changes as the leaf shrinks. The stone has
 * to span {@link STONE_FORM_MINIMUM_SPAN_PERIODS} of the finest octave it would
 * carry — otherwise the band has nowhere to put a trough — and it has to span
 * {@link LEGIBLE_FEATURE_LEAVES} leaves, or the voxelization cannot hold one
 * whatever its period. At 25 mm both give 75 mm; below about 8 mm the period
 * term takes over and the floor stops falling, which is the octave ceiling
 * showing through.
 */
export function stoneFormMinimumSpan_m(leafSize_m: number = STONE_DEFAULT_LEAF_SIZE_M): number {
  return Math.max(
    LEGIBLE_FEATURE_LEAVES * leafSize_m,
    STONE_FORM_MINIMUM_SPAN_PERIODS * stoneFormFinestPeriod_m(leafSize_m),
  );
}

/**
 * The ABI's neighbourhood condition, evaluated at the ratios above so a
 * violation is a visible number here rather than a throw at publication:
 *
 *   (2 - jitter)·period >= floretRadius + smoothRadius
 *   (2 - 0.24)·p        >= (0.69 + 0.30·0.69)·p
 *   1.76·p              >= 0.897·p                     ✓ 96 % margin
 *
 * It holds for every period because both sides are proportional to it. See
 * {@link SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD} in `lib/svo-primitive-abi.ts`.
 */
const STONE_FORM_NEIGHBOURHOOD_MARGIN =
  (2 - STONE_FORM_JITTER) / (STONE_FORM_LOBE_SHARE * (1 + STONE_FORM_BLEND_SHARE));
if (!(STONE_FORM_NEIGHBOURHOOD_MARGIN >= 1)) {
  throw new RangeError(
    `Stone form ratios violate the cluster neighbourhood condition by ${(1 / STONE_FORM_NEIGHBOURHOOD_MARGIN).toFixed(3)}x`,
  );
}

/** A cluster's seed is a u32, and a caller's may be anything an integer hash produced. */
const seed32 = (seed: number): number => (seed >>> 0) || 1;

/** Whether a stone is wide enough that a form period can appear on its horizon. */
export const stoneCarriesFormBand = (radius: Vec3, leafSize_m: number = STONE_DEFAULT_LEAF_SIZE_M): boolean =>
  radius.x + radius.z >= stoneFormMinimumSpan_m(leafSize_m);

function stoneMass(spec: StoneMassSpec): SceneryNode {
  const { id, group, tags, place, radius, material } = spec;
  const leafSize_m = spec.leafSize_m ?? STONE_DEFAULT_LEAF_SIZE_M;
  if (!stoneCarriesFormBand(radius, leafSize_m)) {
    return { kind: "ellipsoid", id, group, tags, place, radius, material };
  }
  const floretRadius = STONE_FORM_LOBE_SHARE * STONE_FORM_PERIOD_M;
  return {
    kind: "cluster", id, group, tags,
    place,
    lobe: radius,
    floretRadius,
    latticePeriod: STONE_FORM_PERIOD_M,
    jitter: STONE_FORM_JITTER,
    smoothRadius: STONE_FORM_BLEND_SHARE * floretRadius,
    // The one number in this node that the scene's lattice decides. Every length
    // beside it is the plate's; this is how many halvings of them the leaf can
    // show. See `stoneFormOctaves`.
    octaves: stoneFormOctaves(leafSize_m),
    seed: seed32(spec.seed),
    material,
  };
}

// ---------------------------------------------------------------------------
// Band 3: the grain, declared and not yet emitted
// ---------------------------------------------------------------------------

/**
 * One op of the grain tape, in the shape `lib/svo-field-program.ts` takes.
 *
 * Declared locally rather than imported as `SvoFieldOpDescriptor` because when
 * this was written two of the ops below did not exist. **They both do now**, and
 * the mapping is exact: `worley-erosion` is `worley-subtract` (op table code
 * 10, "the grain band — dense subtractive pitting, measured off the plate at
 * about three millimetres"), and `footprint-fade` is not an op at all any more
 * because the fade is *structural* — every op that introduces a spatial period
 * declares it as `coarsestPeriod_m` and `svoFieldBandLimitFade` attenuates its
 * amplitude **and its Lipschitz contribution** to zero as that period
 * approaches twice the sampling length. So parameter `d` below has no home and
 * needs no home.
 *
 * One blocker of the three is left, and it is the same one every module in this
 * directory records: `SceneryNode` has no `field-program` member. See the note
 * on {@link STONE_GRAIN_TAPE}.
 */
interface StoneGrainOp {
  readonly op: string;
  readonly out: number;
  readonly point?: number;
  readonly fieldA?: number;
  readonly fieldB?: number;
  readonly seed?: number;
  readonly parameters?: { readonly a?: number; readonly b?: number; readonly c?: number; readonly d?: number };
}

/**
 * The grain band, as the field-program tape it wants to be. **Emitted by
 * nothing.**
 *
 * Three things stood between this and a rendered pit. **Two of them are gone**,
 * and re-reading them is now the shortest path to the third:
 *
 *  1. `SceneryNode` has no `field-program` member, so the scenery graph cannot
 *     carry a tape at all — `lib/svo-primitive-kinds.ts` has kind 12 and
 *     `lib/scenery-graph.ts` has no node for it. **Still true, and now the only
 *     thing that is.** It is one node kind, one `scenery-expand` arm and one
 *     `ProxyBuilder` method away, against machinery that is otherwise complete
 *     end to end: `SvoFieldProgramPrimitive`, the arena packer, the CPU
 *     evaluator, the generated WGSL and `webgpu-sparse-scene-proxies`' whole
 *     `field-program` path all exist and are exercised by tests.
 *  2. ~~`worley-erosion` and `footprint-fade` are not in
 *     `SVO_FIELD_OP_TABLE`.~~ **Landed.** The table now runs to thirteen ops
 *     including `worley-subtract` (code 10), which is this op under its final
 *     name, and the fade is structural rather than an op — see the note on
 *     {@link StoneGrainOp}.
 *  3. ~~It could not show if both existed: the pits are 3 mm against a 25 mm
 *     cell.~~ **The cell moved.** At the hero garden's authored 6.25 mm leaf a
 *     3 mm pit is half a leaf and would still alias; at the 0.78 mm leaf the
 *     tree now reaches it is **3.8 leaves and squarely legible**, which is the
 *     first rung at which this band is the right thing to draw rather than a
 *     note about the future.
 *
 * It is written down anyway because the *decisions* in it are the expensive
 * part and they are made here, against the plate, where the plate is open:
 *
 *  - **Subtractive, not additive.** `max(solid, -pits)`. The plate's surface is
 *    pocked — thousands of shallow round concavities, none of them proud of the
 *    mean surface. An additive octave set puts bumps on instead and reads as
 *    thrown plaster, which is a different material.
 *  - **Cellular, not value noise.** A pit has a rim and a floor and a distinct
 *    neighbour; fBm has neither, and at this amplitude the difference between
 *    the two is the whole of what "porcelain" means.
 *  - **Footprint-faded.** The op's own amplitude goes to zero as the sample's
 *    footprint approaches the pit period, so the band costs nothing and shows
 *    nothing until the leaf shrinks past it. That is the intended outcome at
 *    25 mm, not a failure.
 *  - **One tape for the whole quarry.** A boulder and a pebble carry the same
 *    grain at the same absolute size, because they came out of the same
 *    weather. The tape takes the solid it erodes as an operand; nothing about
 *    it scales with the stone.
 */
export const STONE_GRAIN_PERIOD_M = 0.003;
export const STONE_GRAIN_DEPTH_M = 0.0008;

export const STONE_GRAIN_TAPE: readonly StoneGrainOp[] = Object.freeze([
  // Register 0 is the solid this erodes: whatever the silhouette and form bands
  // above have already produced, handed in rather than rebuilt.
  {
    op: "worley-erosion",
    out: 1,
    point: 0,
    seed: 0x5701_e5,
    // a: cell period, b: pit depth, c: how much of a cell a pit occupies —
    // under a half leaves porcelain between the pits, which is what the plate
    // shows; at one the surface is a honeycomb.
    // d: the footprint at which the amplitude has faded to nothing, as a
    // multiple of the period. Two, so the band is dark until a leaf is half a
    // pit across.
    parameters: { a: STONE_GRAIN_PERIOD_M, b: STONE_GRAIN_DEPTH_M, c: 0.42, d: 2.0 },
  },
  { op: "subtract", out: 2, fieldA: 0, fieldB: 1 },
]);

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
 * **The proportions below are measured off the plate, and they replace numbers
 * that made the stem geometrically impossible to see.** This is the correction
 * that matters most in the file, so it is worth writing down what was wrong,
 * because every individual number looked reasonable.
 *
 * The cap was a full ellipsoid seated at `stemHeight + 0.42·capSemiHeight`, and
 * the stems were authored at 0.10-0.50 cap radii under caps flattened to
 * 0.46-0.72. Put the two together on the largest boulder — cap radius 82 mm,
 * flatten 0.58, stem 0.30 — and the cap's *lower pole* lands at
 *
 *   stemHeight + (0.42 - 1)·capSemiHeight = 0.30R - 0.58·0.58R = -0.036R
 *
 * that is, **below the ground**. The cap's own half-width at the stem's top was
 * 0.91R against a stem of 0.70R. The stem was not narrow, or short, or badly
 * tapered: it was *inside the cap*, on all four boulders, and the drawn solid
 * was a single squashed ellipsoid resting on the bank. There is no overhang to
 * shadow because there is no overhang, and that is the whole of "the boulders
 * read as melted lumps".
 *
 * Measured on `artifacts/plate-crops/boulders.png`, the largest of the three
 * stones, at native resolution (cap half-width 114 px is the unit):
 *
 *   cap half-width         114 px    1.00 R
 *   cap apex to equator     74 px    0.65 R
 *   cap equator to underside 15 px   0.13 R   → total thickness 0.78 R, 2.56 : 1
 *   stem top half-width     82 px    0.72 R
 *   stem base half-width     87 px   0.76 R
 *   stem height             88 px    0.77 R
 *   whole stone             162 px   1.42 R tall on 2.00 R wide
 *
 * Two things fall out. The plate's cap is **not an ellipsoid** — it is a dome
 * over a tight roll-under, so its equator sits at four fifths of its height
 * rather than at half — and an ellipsoid cap therefore has to choose between
 * matching the dome's curvature (`capFlatten` 0.65) and matching the total
 * thickness (0.39). It matches the *thickness*: 2.5 : 1 is the proportion the
 * brief names, a thicker cap deepens the under-tuck by 0.75·capFlatten·R and
 * that is the part the stem has to stay clear of. And the stem is **0.77 cap
 * radii tall**, not the 0.6 this file previously called "the ceiling a stem
 * stops reading as stone at". That ceiling was inferred from a render in which
 * a 1.75-radii stalk carried a 0.30 plate — a mushroom lamp, correctly
 * rejected — and then applied to a regime three times shorter. The plate is the
 * instruction and the plate has tall stems.
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
  /**
   * Cap semi-height as a fraction of `capRadius_m`: half the cap's thickness.
   *
   * The plate's caps are 2.5 : 1 wide to thick, so 0.40 is the family's centre.
   * It is also the parameter the stem's visibility is most sensitive to: the
   * cap's under-tuck reaches `0.75·capFlatten·R` below its own equator before
   * it has narrowed to the stem's width, and everything under that line is
   * cap rather than stem.
   */
  readonly capFlatten: number;
  /** Cap semi-axis across, as a fraction of along. Under 1 keeps the plan oval. */
  readonly capDepthShare: number;
  /**
   * Where the cap's equator sits above the stem's top, as a fraction of the
   * cap's own semi-height.
   *
   * The number that used to be a hard-coded 0.42 and the reason the stem
   * vanished. It has to be small and positive: positive so the cap's widest
   * circle is above the stem's top and the overhang is a real horizontal step
   * rather than a tangency, small so the cap's lower pole stays well clear of
   * the plinth. At 0.30 the cap overhangs by `R·(1 - sqrt(1 - 0.09)) = 0.05 R`
   * measured at the stem's top face, growing to `R - stemTop` at the equator.
   */
  readonly capSeatShare: number;
  /** Stem top radius as a fraction of the cap's. The plate reads 0.72. */
  readonly stemTopShare: number;
  /** Stem base radius as a fraction of the cap's: the taper, wider at the plinth. */
  readonly stemBaseShare: number;
  /** Stem height as a fraction of the cap radius. The plate reads 0.77. */
  readonly stemHeightShare: number;
  /**
   * Plinth radius as a fraction of the cap's, and its thickness.
   *
   * The plate stands every one of its stones on a slab: thin, near-elliptical,
   * flat on top with a rounded edge, and wide enough that a band of it shows
   * outside the stem's foot all the way round. That band is what carries the
   * contact shadow, and the contact shadow is half of why the stones read as
   * *placed on* something rather than as growing out of the plaster.
   *
   * It replaced a half-buried ellipsoid mound of 0.88-0.98 R at 0.20-0.26 R
   * semi-height, whose top stood 0.13 R proud with a plan half-width of 0.82 R
   * against a stem base of 0.76 R — six per cent of a radius of visible slab,
   * on a stone 30 mm across. Nothing about it could have read.
   *
   * The thickness is a little over the plate's own. Measured on the small
   * left-hand stone in `artifacts/plate-crops/boulders.png` the slab is 17 px
   * against a 70 px half-width, so 0.24 R, and at 0.24 R sunk a third the
   * largest boulder's plinth would stand 13 mm proud — **half a cell.** Under
   * voxel-only shading half a cell is nothing at all, so the slab is authored at
   * 0.30-0.36 R sunk a quarter and stands 21 mm, which is most of a cell and can
   * therefore produce a step. This is one of the few places in the file where
   * the render's own limits legitimately outrank the reference, and it is
   * recorded as such rather than smuggled in as a taste change.
   */
  readonly plinthShare: number;
  readonly plinthThicknessShare: number;
  /** How much of the plinth's thickness is buried, as a fraction of itself. */
  readonly plinthSinkShare: number;
  /**
   * A second lobe swelling out of the cap, as a fraction of the cap's radius.
   *
   * Kept on the interface and authored at zero on all four hero forms. It
   * existed because a lone ellipsoid draws a perfect ellipse against the sky;
   * the form band in {@link stoneMass} now breaks that ellipse *on the cap
   * itself*, at a period the cell can carry, which is both cheaper by a record
   * and a better description of a water-worn stone than a second mass stuck to
   * the side of the first. The plate has no boulder with a lump on it.
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
  /** Palette value at the cap. The stem and plinth ramp down off it. */
  readonly value: number;
}

/**
 * Four forms, because a family is proportions and not sizes — and all four
 * squat, because the alternative turned out to be furniture.
 *
 * The set used to be one form at four `capRadius_m`, and the render said what
 * that is: four photocopies at four enlargements, which the eye reads as one
 * object repeated rather than as a group of stones. So the forms were split.
 *
 * The *second* render said what splitting them badly is. The range opened to
 * stem heights of 0.18 to 1.75 cap radii and cap flattenings of 0.30 to 0.62,
 * and at hero scale the tall thin end of that came back as a **mushroom lamp**:
 * a wide flat disc on a thin vertical stalk, reading as a small side table
 * standing beside a pond. Nothing about it was wrong as a shape. It was wrong
 * as a *stone*, and the reason is that a stone has no reason to be tall. Rock
 * that stands proud of a bank on a narrow waist is rock that has been carved;
 * what weather leaves is a bun on a low plinth with an overhanging lip, and the
 * reference's whole group is that.
 *
 * **And then the correction over-shot, which is the third render.** Reading
 * "not a lamp" as "not tall" put the stems at 0.10-0.50 cap radii under caps of
 * 0.46-0.72 — and, as the arithmetic in {@link CappedBoulderForm} shows, that
 * combination puts the cap's lower pole at or below the ground and swallows the
 * stem whole. The forms had become so squat that the species stopped existing:
 * four ellipsoids on a bank. A lamp and a lump are two different failures and
 * the band between them is where the plate actually sits.
 *
 * So the ranges are now stem heights of **0.26 to 0.95** cap radii and cap
 * flattenings of **0.40 to 0.74**, chosen so that on every one of the four the
 * stem's own outline survives:
 *
 *   form            flatten  stem   cap pole   stem visible   stone h : w
 *   MUSHROOM_CAP     0.40    0.95     0.67 R       0.76 R        1.47 : 2
 *   LIPPED_DOME      0.44    0.78     0.47 R       0.54 R        1.35 : 2
 *   SQUAT_ANVIL      0.52    0.55     0.19 R       0.36 R        1.23 : 2
 *   ROUNDED_COBBLE   0.74    0.22    -0.30 R       none          1.18 : 2
 *
 * "cap pole" is where the cap ellipsoid's lowest point lands above the plinth's
 * top face and "stem visible" is the height over which the stem, not the cap,
 * is the outline: the cap's half-width equals the stem's top radius at
 * `capCentre - capSemiHeight·sqrt(1 - stemTopShare²)`, and everything below
 * that is stem. The plate's own stone is 1.42 : 2 with 0.77 R of visible stem,
 * so the first two forms are the plate and the other two are the family spread
 * either side of it. The cobble is the deliberate exception and keeps its
 * negative pole: it is the plate's small left-hand stone, a dome with a stub
 * under it, and its whole job in the family is to be the one that is *not*
 * capped.
 *
 * `capRadius_m` on a form is only its natural size; the hero set overrides it
 * per stone. What a caller is choosing here is the *shape*.
 */

/**
 * The deepest undercut in the set: a thick cap carried well clear of its stem,
 * so the shadow line runs unbroken round the stone.
 *
 * This is what the tall parasol became, and then what the melted lump became.
 * The lip is what the form was always for — the plate's stones are legible at a
 * glance because a dark ring separates cap from stem — and it needs two things
 * at once, which is why it was so easy to lose: a stem narrow enough that the
 * cap stands past it (0.54 of the cap, the narrowest in the set) *and* a stem
 * tall enough that the cap's own under-tuck does not reach the ground. Getting
 * the first without the second is what produced a stone with a 1.83x overhang
 * and no visible stem at all.
 */
export const BOULDER_LIPPED_DOME: CappedBoulderForm = Object.freeze({
  capRadius_m: 0.085,
  capFlatten: 0.44,
  capDepthShare: 0.88,
  capSeatShare: 0.30,
  stemTopShare: 0.54,
  stemBaseShare: 0.68,
  stemHeightShare: 0.78,
  plinthShare: 1.04,
  plinthThicknessShare: 0.30,
  plinthSinkShare: 0.26,
  shoulderShare: 0,
  shoulderOffsetShare: 0,
  lean_rad: 0.09,
  value: 0.90,
});

/**
 * The classic, and the closest of the four to the plate's own largest stone: a
 * 2.5 : 1 cap on the tallest stem in the set.
 *
 * 0.95 cap radii of stem is not a ceiling being spent, it is the measurement —
 * the plate reads 0.77 and the seed's own scatter takes this to 0.86-1.06, so
 * the family straddles it. This is the individual whose whole subject is that
 * you can see what it stands on, and it is the one to look at first in a render
 * to know whether the correction landed.
 */
export const BOULDER_MUSHROOM_CAP: CappedBoulderForm = Object.freeze({
  capRadius_m: 0.095,
  capFlatten: 0.40,
  capDepthShare: 0.90,
  capSeatShare: 0.30,
  stemTopShare: 0.62,
  stemBaseShare: 0.76,
  stemHeightShare: 0.95,
  plinthShare: 1.08,
  plinthThicknessShare: 0.34,
  plinthSinkShare: 0.26,
  shoulderShare: 0,
  shoulderOffsetShare: 0,
  lean_rad: 0.07,
  value: 0.89,
});

/**
 * Squat and broad: the mass in the plate's group that reads as *heavy*.
 *
 * Its cap is half again as thick as the classic's and sits on a stem three
 * quarters of the cap's own width, so there is a lip rather than a brim and the
 * whole thing is one boulder rather than a plate on a post. The oval plan is
 * doing work here — a squat stone drawn on a circle is a drum — and it carries
 * the widest plinth in the set, because a heavy stone needs to be seen to be
 * standing on something that could take it.
 */
export const BOULDER_SQUAT_ANVIL: CappedBoulderForm = Object.freeze({
  capRadius_m: 0.080,
  capFlatten: 0.52,
  capDepthShare: 0.76,
  capSeatShare: 0.30,
  stemTopShare: 0.74,
  stemBaseShare: 0.88,
  // 0.55 read correctly on paper — the widest cap in the set needs the shortest
  // stem — and came out at 29 mm of visible stem, which is 1.1 cells. This is
  // the group's anchor and the largest stone in the frame; if any boulder's
  // articulation has to survive the resample it is this one, so the stem buys
  // back to 1.7 cells at the cost of some of the family's spread.
  stemHeightShare: 0.72,
  plinthShare: 1.12,
  plinthThicknessShare: 0.36,
  plinthSinkShare: 0.24,
  shoulderShare: 0,
  shoulderOffsetShare: 0,
  lean_rad: 0.05,
  value: 0.88,
});

/**
 * The plate's small left-hand stone: a near-round dome on a stub, on its own
 * slab.
 *
 * The one form in the family that is deliberately *not* capped — its cap pole
 * sits below the plinth's top face, so the dome meets its slab directly and no
 * stem shows. Measured off the crop, that stone's dome is 0.87 of its own
 * half-width tall and its stub is a quarter of it, which is what the numbers
 * here are. It is what stops the group reading as four of one thing, and it is
 * the proof that the form carries the silhouette rather than the generator.
 */
export const BOULDER_ROUNDED_COBBLE: CappedBoulderForm = Object.freeze({
  capRadius_m: 0.070,
  capFlatten: 0.74,
  capDepthShare: 0.82,
  capSeatShare: 0.30,
  // A stub still has to taper: a cobble sits on a foot wider than the waist
  // under its own crown, like every other stone in the quarry, or the two
  // solids meet in a re-entrant corner the voxelizer has to decide.
  stemTopShare: 0.80,
  stemBaseShare: 0.94,
  stemHeightShare: 0.22,
  plinthShare: 1.08,
  plinthThicknessShare: 0.30,
  plinthSinkShare: 0.30,
  shoulderShare: 0,
  shoulderOffsetShare: 0,
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
  /**
   * The finest voxel this stone will be drawn at, in metres.
   *
   * Placement rather than form, and the only number here the scene's lattice
   * decides — see `SceneryGeneratorRequest.detailCellSize_m`. Absent is
   * {@link STONE_DEFAULT_LEAF_SIZE_M} and reproduces the 25 mm set exactly.
   */
  readonly leafSize_m?: number;
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
  const plinthRadius = capRadius_m * spec.plinthShare * (0.96 + 0.10 * hash01(seed + 7));
  const plinthHalfHeight = 0.5 * capRadius_m * spec.plinthThicknessShare;

  const leanAzimuth = 2 * Math.PI * hash01(seed + 5);
  const lean = spec.lean_rad * (0.6 + 0.8 * hash01(seed + 6));
  const tilt = alongAxis(V(Math.sin(lean) * Math.cos(leanAzimuth), Math.cos(lean), Math.sin(lean) * Math.sin(leanAzimuth)));

  // The plinth's top face, which is the datum the stone above it stands on.
  // Everything else is measured from here rather than from the ground, because
  // the ground is a heightfield the terrain anchor resolves and the slab is the
  // thing the stem is actually sitting on.
  const seat = (2 - 2 * spec.plinthSinkShare) * plinthHalfHeight;
  // The cap sits down over the stem's top rather than balancing on it, so the
  // two fuse into one solid instead of meeting in a seam the voxelizer would
  // have to decide the ownership of. `capSeatShare` is small and positive: see
  // {@link CappedBoulderForm}, and see the arithmetic there for what the old
  // hard-coded 0.42 did to the stem.
  const capCenterY = seat + stemHeight + spec.capSeatShare * capSemiHeight;
  const shoulderAzimuth = 2 * Math.PI * hash01(seed + 13);

  const children: SceneryNode[] = [
    // The plinth: a thin slab with a flat top and a rounded edge, sunk a third
    // of its own thickness into the bank. One rounded-cylinder SDF rather than
    // a squashed ellipsoid, because what the plate shows is a *slab* — a flat
    // face with a band of shadow under a rolled edge — and an ellipsoid's lens
    // section has no flat face and no edge to roll.
    //
    // The edge radius is 0.42 of the half-thickness rather than all of it: at
    // the full half-thickness the slab is a lens again, and under 0.3 the edge
    // is a corner the voxelizer aliases into a staircase.
    {
      kind: "cylinder",
      id: `${key}/plinth`,
      group,
      tags: [...BOULDER_TAGS, "footing", "plinth"],
      place: { position: V(0, (1 - 2 * spec.plinthSinkShare) * plinthHalfHeight, 0) },
      radius: plinthRadius,
      halfHeight: plinthHalfHeight,
      edgeRadius: 0.42 * plinthHalfHeight,
      // The ramps off the cap's value are a *shading* device — a stone is darker
      // where it turns away from the sky — and they are a third of what they
      // were, because the closure under them no longer adds any grain of its
      // own. A 0.14 step used to read as one stone in two lights; on flat
      // plaster it reads as two stones. Under voxel-only shading they matter
      // more again, not less: the value step between plinth, stem and cap is
      // most of what separates the three when the surface itself is cell faces.
      material: value(spec.value - 0.06 + 0.04 * hash01(seed + 9)),
    },
    {
      // Started a little under the plinth's top face so the two overlap rather
      // than meet, and run up to the cap. A stem that began exactly on the face
      // would leave a coincident circle for the tracer to resolve per sample.
      kind: "cone",
      id: `${key}/stem`,
      group,
      tags: [...BOULDER_TAGS, "stem"],
      place: { position: V(0, 0.5 * (seat + stemHeight) - 0.15 * plinthHalfHeight, 0) },
      baseRadius: stemBase,
      topRadius: stemTop,
      halfHeight: 0.5 * (seat + stemHeight) + 0.15 * plinthHalfHeight,
      material: value(spec.value - 0.045 + 0.03 * hash01(seed + 10)),
    },
    stoneMass({
      id: `${key}/cap`, group, tags: [...BOULDER_TAGS, "cap"], leafSize_m: spec.leafSize_m,
      place: { position: V(0, capCenterY, 0) },
      radius: V(capRadius_m, capSemiHeight, capRadius_m * spec.capDepthShare * (0.96 + 0.10 * hash01(seed + 11))),
      seed: seed + 12,
      material: value(spec.value - 0.01 + 0.04 * hash01(seed + 12)),
    }),
  ];

  if (spec.shoulderShare > 0) {
    // Fused rather than placed — it sits well inside the cap's own surface and
    // only shows where it pushes past it, so what the eye reads is one stone
    // with a shoulder rather than two stones touching. Every hero form authors
    // this at zero now; see {@link CappedBoulderForm}.
    children.push(
      stoneMass({
        id: `${key}/cap-lobe`, group, tags: [...BOULDER_TAGS, "cap"], leafSize_m: spec.leafSize_m,
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
   * How far a course advances per stone, and how far the run advances per
   * course, as fractions of **the drawn stone's own plan diameter**.
   *
   * Both halves of that sentence changed, and together they are the fix for
   * "the pebble courses merge into one continuous sausage".
   *
   * *The fraction.* It was 0.80 and 0.88, on the reasoning that stones in a real
   * bed rest in the hollows between their neighbours and a bed of exactly
   * tangent spheres reads as beads. The plate says otherwise, plainly:
   * `artifacts/plate-crops/pebbles.png` shows a single-file course down the
   * rim's inner edge in which every stone keeps its whole rounded outline and
   * meets its neighbours in a dark contact line. They are beads. What separates
   * them from *our* beads is that ours interpenetrated by a fifth of a diameter,
   * which fuses two spheres into a peanut with a shallow waist, and a shallow
   * waist under flat light is no line at all. At or just over 1.0 the union has
   * a cusp between the stones instead, which is the darkest crease a hard union
   * can make, and that crease is the contact shadow.
   *
   * *The measure.* This used to advance by the **lane radius** while drawing
   * stones of up to 1.58 times it, so the real overlap was never the authored
   * one: a long stone beside another long stone overlapped by 80 % of a
   * diameter however the fraction was set. The step is now taken against each
   * stone's own plan circumradius, `max(semi.x, semi.z)`, so the number here
   * means what it says at every aspect and every yaw. The circumradius rather
   * than the support in the step direction because the yaw is free and a stone
   * that only clears its neighbour at the yaw it happened to draw is a stone
   * that merges on the next re-seed.
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
 * **Where the "every stone gets a silhouette" rule went, and why.**
 *
 * A bed used to carry `clusterFloor_m` and `clusterLobesAcross` — a size above
 * which a stone was published as a marched aggregate rather than as an
 * ellipsoid, and how many lattice lobes spanned it. Both are gone, replaced by
 * {@link STONE_FORM_MINIMUM_SPAN_M}, and it is worth recording that this is the
 * *third* setting of the same line rather than a first attempt.
 *
 * The floor was first reasoned in pixels and set at a 12 mm half-width, which
 * was four times too coarse: the frame is 1 600 px wide, not the 800 that
 * estimate assumed, and the beds sit nearer the eye than the pond's own middle.
 * It came down to 5 mm, and every stone in the beds went marched.
 *
 * The pixel reasoning is now the wrong reasoning entirely, on two counts. The
 * measured one is in the header: at a lobe radius of 0.98 of the period the
 * lattice covers space, so those marched stones drew the ellipsoid envelope
 * exactly and the floor decided nothing at all. The structural one is that
 * shading is voxels only — the primary returns a cell face, not an analytic
 * intersection — so the question "can the eye resolve this outline" is settled
 * by the *cell* long before it reaches the pixel. At 25 mm cells the whole bed
 * is one to three cells per stone and no field on it can survive the resample.
 *
 * So the rule is one absolute length, applied to every species in the file from
 * one place, and it lands almost everything in these beds on the ellipsoid
 * side. That is not a loss of fidelity against what shipped: it is the same
 * drawn shape, published as the primitive that draws it, at a closed-form root
 * instead of forty-eight march steps.
 */

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
  // Just over tangency, measured against the drawn plan circumradius. See
  // {@link PebbleBedForm} for why these went from under one to over it and why
  // the measure changed with them.
  alongPacking: 1.00,
  acrossPacking: 1.02,
  // The aspect band, narrowed hard toward the plate. It used to run to a 1.58
  // length share against a 0.34 height share, authored as "flat slabs lying on
  // their side beside near-round cobbles" — and that is not what
  // `artifacts/plate-crops/pebbles.png` holds. Every stone in that course is a
  // smoothly rounded, slightly flattened ovoid: plan aspect 1.3 to 1.5, height
  // a little under three quarters of the plan half-width. No flakes, no slabs,
  // nothing lying on an edge.
  //
  // It is also the *other* half of the merging. A stone drawn 1.58 lanes long
  // reaches most of the way across its second neighbour whatever the packing
  // says, so the aspect band and the packing fraction had to be corrected
  // together or one would have undone the other.
  lengthShare: [0.96, 1.42] as const,
  heightShare: [0.52, 0.80] as const,
  widthShare: [0.74, 1.06] as const,
  widthWander: 0.34,
  // Coarsening *outward*, because the bank band grows away from the water: the
  // reference heaps its biggest cobbles up against the boulders and fines to
  // shingle at the shoreline, and a bed of one grain reads as gravel however
  // wide the size spread inside it is.
  crossGrade: 1.34,
  // A fifth of the stone rather than a third. Sinking is what buries the
  // contact line the plate's stones show: at 0.34 the widest circle of a
  // flattened pebble is at or under the plaster, so what stands proud is a
  // shallow cap with no undercut, and a row of shallow caps is a ridge. At 0.20
  // the stone's own equator clears the ground and the crease between neighbours
  // runs down to it.
  sinkShare: 0.20,
  moundShare: 0.55,
  moundWidthFloor_m: 0.09,
  // Deeper than it was but held to the last quarter of the band, and the power
  // is the load-bearing half of that pair. The bed coarsens outward, so a
  // thinning that bit at half width would be deleting precisely the largest
  // cobbles it had just graded up — the first pass at these numbers did exactly
  // that and capped the bed at 58 mm when it was authored to reach 86.
  //
  // Both numbers came down when the packing opened out: a bed laid at tangency
  // already has plaster showing between its stones, so a thinning tuned against
  // a bed that overlapped by a fifth now deletes a course that was reading.
  edgeThinning: 0.52,
  edgeThinningPower: 2.2,
  bedTiltFollow: 0.72,
  bedNormalJitter: 0.42,
  value: [0.82, 0.93] as const,
});

/**
 * The water's-edge course: two thirds the grain, flatter stones, laid tighter and
 * fining as it runs down. This is what the reference shows all the way round its
 * right and back, where the bed is one course of small stones on the rim's inner
 * slope, and what its left shore fines *to* as the shelf runs into the water.
 *
 * This is the bed the plate is clearest about and the one the render got most
 * wrong, so it is the one to judge the change on.
 * `artifacts/plate-crops/pebbles.png` is almost entirely this species: a course
 * running down the rim's inner edge, one stone wide for most of its length and
 * two where it widens, sizes varying about three to one, every stone distinct
 * with a dark line where it meets the next. Ours came back as a continuous
 * sausage lying against the coping — the same white value as the rim it rests
 * on, no crease anywhere along it, no individual stone recoverable.
 */
export const PEBBLE_FINE_SHINGLE: PebbleBedForm = Object.freeze({
  radiusRich_m: 0.0125,
  radiusLean_m: 0.0058,
  sizeSpread: [0.58, 1.48] as const,
  // The tightest packing in the set and still clear of one: this is the course
  // that has to read as separate stones at the front of the frame.
  alongPacking: 0.99,
  acrossPacking: 1.01,
  lengthShare: [0.98, 1.38] as const,
  heightShare: [0.54, 0.82] as const,
  widthShare: [0.76, 1.04] as const,
  widthWander: 0.22,
  crossGrade: 0.62,
  sinkShare: 0.22,
  moundShare: 0.40,
  moundWidthFloor_m: 0.07,
  edgeThinning: 0.50,
  edgeThinningPower: 1.3,
  bedTiltFollow: 0.78,
  bedNormalJitter: 0.46,
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
  /**
   * The finest voxel this bed will be drawn at, in metres.
   *
   * Placement rather than form, and the only number here the scene's lattice
   * decides — see `SceneryGeneratorRequest.detailCellSize_m`. Absent is
   * {@link STONE_DEFAULT_LEAF_SIZE_M} and reproduces the 25 mm set exactly.
   */
  readonly leafSize_m?: number;
}

const PEBBLE_TAGS = Object.freeze(["stone", "pebble"]);

/**
 * The most one course of a bed may coarsen over the one before it.
 *
 * See the use site: this is what keeps a packing that steps by the *last*
 * course's reach from being overrun by the *next* course's stones. A quarter is
 * enough that a bed still runs from cobbles to shingle over a drift — twelve
 * courses at 1.25 is a factor of fourteen — and small enough that no two stones
 * lying against each other differ by more than the plate's own courses do.
 */
const STONE_COURSE_GROWTH = 1.25;

/**
 * How far a stone is staggered along the run, as a share of its own reach.
 *
 * What stops the courses reading as rows. It used to be 0.35 and it used to be
 * spent out of the packing's own clearance: two stones in adjacent courses each
 * staggering 0.35 of a reach toward one another close 0.7 of a reach, against
 * the 0.08 a packing fraction of 1.04 had given them. That is most of a stone of
 * overlap, invisible in every authored number, and it was the largest single
 * source of the merging that survived the packing correction.
 *
 * It is now smaller *and* paid for: the course advance below carries the same
 * factor, so the stagger moves stones without ever eating the gap between them.
 */
const STONE_COURSE_STAGGER = 0.16;

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
  let previousReach = 0;

  // The last course has to clear the first one, and `railAt` wraps: a loop that
  // ran while `arc < length` laid a final course a few millimetres short of the
  // rail's own end, right on top of course zero. Stopping a whole course's reach
  // early closes the ring instead of overlapping it.
  while (arc < rail.length_m - 2 * spec.alongPacking * previousReach) {
    const turn = railAt(rail, arc, 0).turn;
    const grade = Math.min(1, Math.max(0, gradeAt(turn)));
    const density = Math.min(1, Math.max(0, densityAt(turn)));
    // A little of the bed's wandering is its own rather than its run's: two slow
    // octaves keyed on the course so the width breathes along the bank.
    const noise = 0.5 + 0.5 * Math.sin(2 * Math.PI * (3.3 * turn + hash01(spec.seed)))
      * (0.6 + 0.4 * Math.sin(2 * Math.PI * (7.7 * turn + hash01(spec.seed + 1))));
    const meanRadius = spec.radiusLean_m + (spec.radiusRich_m - spec.radiusLean_m) * grade;
    const width = Math.max(0, spec.widthAt(turn)) * (1 + spec.widthWander * (2 * noise - 1));

    // A stone may not be much larger than the course before it. The run advances
    // by the reach of the course just laid, so the *next* course's stones are
    // laid at a spacing chosen before they were drawn — and a stone drawn three
    // times its predecessor (the size spread alone is 3.4 : 1 inside one lane)
    // reaches back over the whole course behind it. Measured before this ratchet
    // existed: 31 of 105 pebbles interpenetrated a neighbour by more than a tenth
    // of themselves and the worst sat entirely inside another stone.
    //
    // A quarter of growth per course rather than a hard cap, because a bed's
    // grain does change along a run — that is what `crossGrade` and the drift
    // table are for — it just does not change by a factor of three between two
    // stones lying against each other, and the plate's courses do not either.
    const reachCeiling = previousReach > 0 ? STONE_COURSE_GROWTH * previousReach : Infinity;
    let across = start_m + direction * (course % 2 === 0 ? 0 : spec.acrossPacking * meanRadius);
    const limit = start_m + direction * width;
    let lane = 0;
    // The widest stone this course actually laid, so the run advances by what
    // was drawn rather than by the lane's mean. A course holding one 40 mm
    // cobble has to clear 40 mm before the next course starts, and stepping by
    // the mean instead is what let a large stone in a fine course swallow both
    // its neighbours and the two stones behind it.
    let courseReach = 0;
    while (direction * (limit - across) > 0) {
      const seed = spec.seed + 0x3f_00 + 31 * published + 7919 * lane + 131 * course;
      const bandFraction = Math.abs(across - start_m) / Math.max(1e-6, width);
      // The grade also runs *across* the band, which is what a beach does: the
      // cobbles are at the top of the shelf and the shingle is at the water.
      const laneRadius = meanRadius * (1 + (spec.crossGrade - 1) * Math.min(1, bandFraction));
      const radius = laneRadius * band(spec.sizeSpread, hash01(seed));

      // Drawn first, stepped by second. The aspect band is wide enough that the
      // stone's own plan circumradius and the lane radius it came from differ by
      // up to half again, and stepping by the lane radius is what made this a
      // packing of *lanes* rather than a packing of stones.
      const drawn = V(
        radius * band(spec.lengthShare, hash01(seed + 2)),
        radius * band(spec.heightShare, hash01(seed + 3)),
        radius * band(spec.widthShare, hash01(seed + 4)),
      );
      // Scaled whole rather than clipped on one axis, so the ratchet takes size
      // off a stone and never changes its aspect — the aspect is the plate's and
      // the size is negotiable.
      const restraint = Math.min(1, reachCeiling / Math.max(drawn.x, drawn.z));
      const semi = restraint < 1 ? V(drawn.x * restraint, drawn.y * restraint, drawn.z * restraint) : drawn;
      const planReach = Math.max(semi.x, semi.z);
      courseReach = Math.max(courseReach, planReach);

      across += direction * spec.acrossPacking * planReach;
      const centre = across;
      const at = railAt(rail, arc + (hashSigned(seed + 1) * STONE_COURSE_STAGGER * planReach), centre);
      across += direction * spec.acrossPacking * planReach;
      lane += 1;

      const sink = spec.sinkShare * semi.y;
      const edge = Math.abs(centre - start_m) / Math.max(1e-6, width);
      // Dispersal along the run, then thinning across the band. Two independent
      // hashes, because a stone dropped by both is a stone dropped once and the
      // two effects have to compose rather than mask one another.
      if (hash01(seed + 11) >= density) continue;
      if (hash01(seed + 7) < spec.edgeThinning * edge ** spec.edgeThinningPower) continue;
      if (spec.within && !spec.within(at.x, at.z)) continue;
      if (spec.avoid?.some(({ at_m, radius_m }) => Math.hypot(at.x - at_m[0], at.z - at_m[1]) < radius_m + 0.35 * planReach)) continue;
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
        leafSize_m: spec.leafSize_m,
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
        seed: seed + 12,
        material: value(band(spec.value, hash01(seed + 6))),
      }));
      published += 1;
    }
    // **The rail's arc is measured on its own centreline, and a band laid off it
    // does not travel that distance.** Offsetting a closed convex outline inward
    // by `d` removes exactly `2*pi*d` of perimeter, so a course spacing stepped
    // on the centreline lands an inward band's stones closer together than it
    // was asked for — on this pond's rail, 2.7 m round, the shore band sits
    // about 60 mm in and loses 14 % of its spacing. That is a seventh of a stone
    // of overlap that no packing fraction can see, because the packing is
    // working in a coordinate the stones are not laid in.
    const bandOffset = start_m + direction * 0.5 * width;
    const railScale = rail.length_m / Math.max(0.25 * rail.length_m, rail.length_m + 2 * Math.PI * bandOffset);
    arc += 2 * spec.alongPacking * (1 + STONE_COURSE_STAGGER) * railScale
      * (courseReach > 0 ? courseReach : meanRadius);
    previousReach = courseReach > 0 ? courseReach : previousReach;
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
 * What a stepping stone is: a tread on a footing, and how much of it shows.
 *
 * The tread is a cylinder with a crown over it, the crown's equator on the
 * cylinder's top face. That is tangent-continuous — the side runs vertically
 * into a dome — so the stone reads as cut flat with a softened edge, which is
 * what the reference shows, without the two-primitive seam a smaller dome would
 * leave.
 *
 * **A plate carries one lobe off its own axis.** These were the last surfaces of
 * revolution in the set, and at hero scale five of them lying in a row came back
 * reading as *pills*: flat white ovals of one size at one spacing, tablets
 * rather than stones set in water. Three causes, and the numbers below fix two —
 * the section was a fifth of the plate where the reference's is a quarter, and
 * the sizes and strides had almost no scatter on them at all. The third is that
 * a rounded cylinder draws a perfect circle in plan whatever else is done to it.
 *
 * The fix is the device the capped boulder already uses: one {@link stoneMass}
 * set off the axis, mostly inside the tread and showing only where it pushes
 * past the rim, so the outline the eye follows round the stone is broken on one
 * side. Deliberately *not* a mass over the whole tread — the tread is a single
 * rounded-cylinder SDF precisely so that its crown has one zero set and a fillet
 * wide enough to resolve, and wrapping it in a second solid whose surface nearly
 * coincides near the rim would put back the broken ring of grazing dots that
 * cost the plates their edge in the first place. The lobe sits low and sticks
 * out; its normals nowhere agree with the rim's.
 */
export interface SteppingStoneForm {
  /**
   * Tread thickness, as a fraction of the tread's own radius.
   *
   * Measured off the reference against the disc's own width: its stones show a
   * side about a quarter of their diameter. Two things were wrong with the
   * 18 mm this used to be. It was a fifth rather than a quarter, which read as a
   * tile laid on the water; and being a *length* it did not scale, so the same
   * 18 mm on the far plates — which are a fifth narrower — made the small end of
   * the chain thinner still, exactly where the perspective was already shrinking
   * it. A share keeps the section constant down the path, which is what "a set
   * of stones" means and what a fixed thickness cannot say.
   */
  readonly treadShare: number;
  /** Dome rise on the crown, as a fraction of the tread radius. */
  readonly dome: number;
  /**
   * The off-axis lobe: its half-width and its offset, both as fractions of the
   * tread radius.
   *
   * `shoulderShare + shoulderOffsetShare` a little over one is the whole trick —
   * that surplus is how far the lobe stands past the rim, and at 0.10 it is a
   * bulge on one flank rather than a second stone stuck to the first. Zero
   * omits it and saves a record.
   *
   * It no longer carries a lobe count: whether the lobe gets a form band is
   * decided by its own plan span against {@link STONE_FORM_MINIMUM_SPAN_M}, like
   * every other mass in the file. At the hero path's 32-42 mm plates the
   * shoulder is about 50 mm across and falls under the floor, so it publishes as
   * the ellipsoid it was always drawing.
   */
  readonly shoulderShare: number;
  readonly shoulderOffsetShare: number;
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
  /**
   * Per-stone scatter on the radius, and on the stride between stones, each as a
   * fraction of itself.
   *
   * Three per cent of radius and none at all on the stride is what made a row of
   * pills: five stones the same size at the same spacing is a *pattern*, and the
   * eye reads a pattern as manufacture however irregular each element is. Real
   * stepping stones are whatever the mason had, set where the wading was good.
   * A fifth on the size and nearly a half on the stride is enough that no two
   * gaps in the chain are alike.
   */
  readonly radiusJitter: number;
  readonly strideJitter: number;
  /** Palette value at the tread. The footing ramps down off it. */
  readonly value: number;
}

/**
 * The reference's five: pale stones set proud of the water, not tablets on it.
 *
 * Plate rather than drum is still the form — what stands above the bed is a
 * broad low stone with a softened edge and no visible support, and the footing
 * stays narrower than the tread at every height so no collar shows on a scene
 * that opens dry. What changed is that "plate" had been read as "thin". The
 * reference's stones are about a quarter of their own width in section and their
 * outlines are irregular; ours were a fifth, perfectly round and identical, and
 * five of those in a row is a bathroom rather than a pond.
 */
export const STEPPING_DISC: SteppingStoneForm = Object.freeze({
  // 0.60 of the radius plus a 0.11 crown is a solid 0.355 of its own width
  // deep, against the 0.284 the old fixed 18 mm gave the nominal plate.
  treadShare: 0.60,
  dome: 0.11,
  shoulderShare: 0.62,
  shoulderOffsetShare: 0.48,
  footingBaseShare: 0.94,
  footingTopShare: 0.78,
  bed_m: 0.052,
  freeboard_m: 0.011,
  emergenceShare: 0.85,
  radiusJitter: 0.20,
  strideJitter: 0.45,
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
  /**
   * The finest voxel this set will be drawn at, in metres.
   *
   * Placement rather than form, and the only number here the scene's lattice
   * decides — see `SceneryGeneratorRequest.detailCellSize_m`. Absent is
   * {@link STONE_DEFAULT_LEAF_SIZE_M} and reproduces the 25 mm set exactly.
   */
  readonly leafSize_m?: number;
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

  // The clear water between one stone and the next, drawn per gap. Keyed on the
  // gap's own index rather than on either stone's, so widening a plate does not
  // silently re-roll the stride beyond it.
  const strideOf = (gap: number): number =>
    spec.stride_m * (1 - 0.5 * spec.strideJitter + spec.strideJitter * hash01(seed + 0x5d_00 + 271 * gap));

  const placements: SteppingStonePlacement[] = [];
  let arc = 0;
  for (let index = 0; index < count; index += 1) {
    const stoneSeed = seed + 0x77_00 + 613 * index;
    const radius = radiusOf(index);
    const tread = spec.treadShare * radius;
    if (index > 0) arc += radiusOf(index - 1) + radius + strideOf(index - 1);
    const [x, z] = pointAtArc(arc);
    const ground = groundHeightAt(x, z);
    const level = spec.level_m ?? -Infinity;
    const treadTop = Math.max(
      level + spec.freeboard_m * (0.85 + 0.3 * hash01(stoneSeed + 1)),
      ground + spec.emergenceShare * tread,
    );
    const treadBottom = treadTop - tread;
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
        ...(spec.shoulderShare > 0 ? [stoneMass({
          leafSize_m: spec.leafSize_m,
          // Set at the tread's own mid-height and on a bearing of its own, so
          // the five plates are not all lopsided the same way. Its top is held a
          // little under the crown's apex: a lobe that broke the *upper* surface
          // would put a lump where the foot goes.
          id: `${key}/step-${index}/shoulder`,
          group,
          tags: [...STEPPING_TAGS, "tread"],
          place: {
            position: V(
              radius * spec.shoulderOffsetShare * Math.cos(2 * Math.PI * hash01(stoneSeed + 7)),
              0.5 * (treadBottom + treadTop),
              radius * spec.shoulderOffsetShare * Math.sin(2 * Math.PI * hash01(stoneSeed + 7)),
            ),
          },
          radius: V(
            radius * spec.shoulderShare * (0.88 + 0.24 * hash01(stoneSeed + 8)),
            0.46 * (treadTop - treadBottom),
            radius * spec.shoulderShare * (0.82 + 0.30 * hash01(stoneSeed + 9)),
          ),
          seed: stoneSeed + 10,
          material: value(spec.value - 0.02 + 0.04 * hash01(stoneSeed + 11)),
        })] : []),
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
  /**
   * The finest voxel this set will be drawn at, in metres.
   *
   * Placement rather than form, and the only number here the scene's lattice
   * decides — see `SceneryGeneratorRequest.detailCellSize_m`. Absent is
   * {@link STONE_DEFAULT_LEAF_SIZE_M} and reproduces the 25 mm set exactly.
   */
  readonly leafSize_m?: number;
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
 * lives on the bank the camera sees across the water, and the caps grade 2.7 : 1
 * across it — the receding line the reference draws, rather than a row.
 *
 * The third mistake was not about *where* at all. All four used to be
 * `BOULDER_MUSHROOM_CAP` at four radii and four even steps of the rail, and both
 * halves of that were wrong. Four enlargements of one drawing read as one object
 * repeated, whatever their sizes; and four even steps read as a fence, because a
 * group of stones the eye accepts as *placed* has no interval in it that
 * repeats. So each row names its own form, and the turns are unevenly spaced —
 * a close pair, a third leaning in, and the small one set out on its own.
 *
 * **The fourth mistake was where, and it is the reason these turns are solved
 * rather than authored.** The group sat at turns 0.558-0.688, which reads on
 * paper as "the far half of the left bank" and reads in the frame as *the left
 * edge*: rendered at 1600x920 the four landed at 12 %, 3 %, -1 % and -6 % of the
 * frame's width, so two of them were outside the crop entirely and the
 * composition's anchor was off camera. Meanwhile the middle of the frame held
 * empty plaster and an empty basin.
 *
 * Two lines of arithmetic decide this and they are worth keeping written down,
 * because "move it left a bit" is how the group got here. At the hero camera —
 * azimuth 5.4, elevation 0.40, distance 1.40 m, target (-0.04, 0.35, 0),
 * `tanHalfFov` 0.24 — the eye stands at (-1.04, 0.90, 0.82) and the bank's own
 * rail sweeps across the frame like this, at the offset these stones stand at:
 *
 *   turn  0.60  0.64  0.68  0.72  0.76  0.80  0.84  0.88
 *   x      -4 %   2 %  10 %  19 %  27 %  35 %  43 %  52 %
 *   y      55 %  50 %  42 %  36 %  32 %  28 %  25 %  24 %
 *
 * Everything below turn 0.66 is against the left edge or off it. The bonsai
 * stands at 46 %, so the run from 0.69 to 0.79 — 12 % to 33 % of the frame,
 * upper left, a metre and a half from the eye and squarely behind the water — is
 * both the empty part of the composition and the last bank wide enough to stand
 * on. That is where the family now is, and `HERO_PEBBLE_DRIFTS` follows it: the
 * coarse drift's heart is pinned to this group's own turn, so the cobbles moved
 * with the stones they are heaped against.
 *
 * The room ran out at the same time. On these bearings the container wall is
 * 0.143 to 0.175 m outside the coping's crest, so a 0.082 m cap at an offset
 * over about 0.14 would put its widest part through the domain — which is why
 * the offsets below are smaller than the ones the old, roomier corner allowed.
 * What that widest part *is* changed with the form: it used to be the cap's
 * off-axis shoulder at 1.30 cap radii, and it is now the plinth slab at 1.12,
 * so the same offsets carry 14 % more clearance than they were authored with.
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

/**
 * **The fifth mistake was that the group had no daylight in it**, and it is the
 * one a viewer names first: the caps merged into a single lump on the bank.
 *
 * "Most of a cap away and set further out so the two overlap in plan" is a
 * description of a pair that reads as a pair when you can see round both of
 * them, and this arrangement could not. Measured in plan at the shipped turns,
 * the anchor and its partner **overlapped by 41 mm of cap** and by 54 mm of
 * plinth, and the anchor and the third touched at exactly 0 mm — so three of the
 * four were one silhouette with two bumps on it, at any resolution and under any
 * light. The plate's group is the opposite: every stone in it is closed by sky
 * or by shadow on both sides, and that separation is most of why four objects
 * read as four.
 *
 * The turns below are therefore solved against a plan clearance rather than
 * spaced by eye, and the quantity solved for is the **cap** gap — what the
 * silhouette does — with the plinth gap carried as a second constraint because a
 * plinth that overlaps merges the feet even when the caps are clear. Every pair
 * now holds at least **47 mm of cap gap and 35 mm of plinth gap**, the tightest
 * being the pair, which is as it should be: they stay the closest two stones in
 * the group without being one stone.
 *
 * Spreading costs room, and the group was already against the domain: the
 * container wall is 0.143-0.175 m outside the crest on these bearings and the
 * plaster is |x| <= 0.852, |z| <= 0.552. The offsets came *in* as the turns went
 * apart to pay for it — the anchor from 0.104 to 0.084 — which also breaks the
 * near-constant radius the four used to sit at. The whole family clears the
 * plaster by at least 7 mm at its widest part.
 *
 * The turn window is unchanged in spirit: 0.68 to 0.83 is still the upper-left
 * bank, still clear of the left crop at 0.66, and still short of the bonsai at
 * 46 % of the frame.
 */
const HERO_BOULDERS: readonly BoulderPlacement[] = Object.freeze([
  // The anchor: broad, heavy, seated, and set nearest the rim so it reads in
  // front of the rest. Lands at about a fifth of the way across the frame.
  { turn: 0.731, offset_m: 0.084, capRadius_m: 0.080, form: BOULDER_SQUAT_ANVIL },
  // Its partner, a cap and a half away along the bank and further out from the
  // rim, so the two are a pair seen past one another rather than a pair fused.
  { turn: 0.680, offset_m: 0.126, capRadius_m: 0.066, form: BOULDER_LIPPED_DOME },
  // The third leans back into the pair rather than continuing the line.
  { turn: 0.782, offset_m: 0.118, capRadius_m: 0.047, form: BOULDER_MUSHROOM_CAP },
  // Small and low, set along the bank on its own: the thing that stops the group
  // reading as a wall, and the one that carries the eye toward the tree.
  { turn: 0.828, offset_m: 0.132, capRadius_m: 0.030, form: BOULDER_ROUNDED_COBBLE },
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
 * **The table was retuned when the packing was corrected, and the retune is not
 * art direction.** Stepping by each stone's own drawn plan reach at a fraction
 * over tangency, instead of by 0.80 of a lane diameter, necessarily lays about
 * half as many stones over the same ground — that is the whole point of it, and
 * it is what buys the contact creases the plate has and the render did not. The
 * bands here are wider and the density tails higher to take about two thirds of
 * that back (161 stone-set records to 139, against 92 if nothing had moved),
 * and the `halfLength` runs deliberately were *not* stretched to take the rest:
 * the bare stretches of coping between the drifts are the composition, they are
 * pinned by `tests/stone-set.test.ts`, and spending them on record count would
 * be trading the thing the plate is admired for against a number.
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
  // The heap at the boulders' feet, and its heart is the *family's* own turn —
  // the whole subject of the reference's main bed is that the biggest cobbles
  // are the ones wedged against the biggest stones, so this number is not free
  // and moves whenever `HERO_BOULDERS` does. It runs out both ways to a quarter
  // of the grain and a third of the stones.
  //
  // Its band is narrower than it was, and that is the far bank rather than the
  // art direction: the container wall is about 0.16 m outside the crest here,
  // so a band that ran the old 0.150 m would have its outer courses culled
  // against `onPlaster` in a straight line, which is a machined edge wearing a
  // bed's clothes.
  // Re-pinned to 0.755 when the family spread from 0.694-0.788 to 0.680-0.828.
  // The heart follows the group's own middle, which is the whole reason it is a
  // number here rather than a constant: "the biggest cobbles are wedged against
  // the biggest stones" is only true if the two move together. `halfLength` is
  // deliberately *not* stretched to cover the wider group — see above.
  { turn: 0.755, halfLength: 0.120, bankWidth_m: 0.132, grade: 1.00, gradeTail: 0.38, density: 1.00, densityTail: 0.60, shore: false },
  // The beach: where the shelf runs down into the water, and the only place
  // both bands meet. Fine from the start and finer as it goes. It no longer
  // joins the heap above — the family moved round the bank and this one did not,
  // because it belongs to the vessel's beach sector rather than to the stones —
  // so what was one long bed is now two places with bare coping between them,
  // which is what the reference shows anyway.
  { turn: 0.494, halfLength: 0.116, bankWidth_m: 0.112, grade: 0.50, gradeTail: 0.12, density: 1.00, densityTail: 0.56, shore: true },
  // The course along the right-hand rim, where the tree's terrace comes down to
  // the coping. Narrow, fine, and the one the reference runs *inside* the rim.
  { turn: 0.080, halfLength: 0.122, bankWidth_m: 0.084, grade: 0.24, gradeTail: 0.06, density: 1.00, densityTail: 0.52, shore: true },
  // The loose scatter outside the coping at the near right: a dozen stones, no
  // bed under them, which is what a narrow band at half density produces.
  { turn: 0.230, halfLength: 0.068, bankWidth_m: 0.052, grade: 0.15, gradeTail: 0.02, density: 0.70, densityTail: 0.26, shore: false },
]);

/** How much of a drift's run is full width before it fades out. */
function driftWeight(drift: PebbleDrift, turn: number): number {
  return ramp(1 - turnDelta(turn, drift.turn) / drift.halfLength);
}

/**
 * The bank band's start, in plan distance from the coping's crest centreline.
 *
 * Beyond the widest the coping's section ever swells to, so a bed never climbs
 * the rim's outer shoulder. It was 1.14, and that did not actually clear it:
 * `sectionWidthVariation` is 0.22 but `periodicSpline` reaches about 1.6x its
 * amplitude, so the widest half-width is **1.35** of the authored mean and a
 * 1.14 start laid cobbles on the rim wherever the section swelled.
 *
 * In absolute terms this moves the bed *toward* the water, not away: the coping
 * narrowed from 64.4 mm of half-width to 30, so the bank now starts 42 mm out
 * where it used to start 73. That is the plate, which butts its cobbles against
 * the rim's foot rather than leaving a bare shelf between the two.
 */
const HERO_BANK_START_SHARE = 1.40;

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
const HERO_SHORE_RUN_SHARE = 0.80;
/**
 * The narrowest the water's-edge course ever runs, in plan metres.
 *
 * It was 22 mm, which at this bed's grain is under one stone: a "single course"
 * that in practice laid one stone and then ran out of band. The plate's course
 * — `artifacts/plate-crops/pebbles.png` — is one stone wide for most of its
 * length and two where it widens, so the floor is now a stone and a half.
 */
const HERO_SHORE_WIDTH_FLOOR_M = 0.046;

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
const HERO_STEP_STRIDE_M = 0.030;

/**
 * Clear water between the nearest plate's rim and the coping's outermost swell.
 *
 * A plate may never come nearer the rim than its own radius plus the widest the
 * coping's section swells to, or it grows out of the rim it is supposed to have
 * stepped off. This margin sits on top of that, and it is taken against the
 * *largest* plate, because the walk lays stones between the control points and
 * the shore's own contour curls back toward the rim past the beach's near edge:
 * the clamp has to hold for the interpolated positions too, not just for the
 * ones solved here.
 *
 * It was 12 mm, which the shore solve never even reached — the contour put the
 * first plate 102 mm out and the clamp only bound at 95 — so the plate sat with
 * about 31 mm of open water between it and the coping's inner foot. That is not
 * enough for the vessel: a deeper beach sweep needs the shelf to pinch further
 * in (`PLAN_BEACH_PINCH` wants to go from 6 % toward the reference's 12 %) and
 * the pinch has nowhere to go while a stone is parked in its way. At 52 mm the
 * clamp binds, the near plate moves out into the water where it reads better
 * anyway, and the clearance is a little under 60 mm — the 25 mm the pool wall
 * asked for, and then some. This number is the *vessel's* room rather than the
 * path's, and it is the only thing standing between the beach and its reference
 * profile.
 */
const HERO_STEP_WALL_MARGIN_M = 0.052;

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
      leafSize_m: spec.leafSize_m,
    }));
  }
  return nodes;
}

/** Plan footprint a boulder claims, so the beds can be laid around it. */
function boulderFootprints(spec: StoneSetSpec, rail: PlanRail): StoneFootprint[] {
  return HERO_BOULDERS.map((placement) => {
    const at = railTurnAt(rail, placement.turn, placement.offset_m);
    // **The stem's foot, not the cap and not the plinth.** This used to be 0.88
    // of the cap radius, which on the largest boulder is a 72 mm circle laid over
    // a band that starts 40 mm out and runs 132 mm — that is, the whole band, at
    // every one of the four boulders, whose exclusion circles overlap one another
    // along the bank. The drift whose entire subject is "the biggest cobbles are
    // the ones wedged against the biggest stones" had no ground to stand on, and
    // the only reason it read at all was that the bed was dense enough to put
    // nine stones in the annulus outside it.
    //
    // `artifacts/plate-crops/boulders.png` settles what the clearance should be:
    // the cobbles butt directly against the boulders' feet and two of them lie
    // over the rolled edge of a slab. So the circle is the stem's base — what a
    // stone would actually have to displace — and the per-stone margin is a third
    // of a pebble rather than a half.
    return { at_m: [at.x, at.z] as const, radius_m: placement.capRadius_m * placement.form.stemBaseShare };
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
      leafSize_m: spec.leafSize_m,
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
      leafSize_m: spec.leafSize_m,
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
  // Against the *largest* plate the chain can actually produce, which is the
  // nominal shore radius plus half the size scatter — not the nominal itself.
  // The scatter went from 3 % to 20 % when the plates stopped reading as pills,
  // and 20 % of a plate is 4 mm of clearance somebody would otherwise have
  // budgeted and not received.
  const largestPlate = HERO_STEP_RADIUS_SHORE_M * (1 + 0.5 * STEPPING_DISC.radiusJitter);
  const minimumRun = 1.30 * spec.vessel.rimHalfWidth_m + largestPlate + HERO_STEP_WALL_MARGIN_M;
  return {
    ...STEPPING_DISC,
    key: `${key}/path`,
    seed: spec.seed,
    leafSize_m: spec.leafSize_m,
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
