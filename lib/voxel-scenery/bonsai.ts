import type { Vec3 } from "../model";
import type { SceneryMaterial, SceneryNode } from "../scenery-graph";
import { alongAxis, V } from "./builder";

/**
 * The bonsai: a fused multi-trunk root mass under a wide crown of cauliflower
 * florets. One species, parameterised — the hero pond's specimen is a form, not
 * the object.
 *
 * Reference: `output/imagegen/garden-pond-hose-fill-simplified.png`, and
 * `docs/HERO_GARDEN_HOSE_SCENE_PLAN.md` §4 and §6b item 3.
 *
 * ## Why this is not `procedural-tree`
 *
 * The existing generator grows a *niwaki*: one tapering trunk, limbs on a
 * golden-angle spiral, and foliage as a handful of flattened pads spaced so the
 * tiers read separately with sky between them. Every one of those decisions is
 * the opposite of what this species needs. Its trunk is a fused root mass that
 * splits low into pale limbs; its crown is one wide plate with no sky in it at
 * all, and the plate's whole character is the floret structure covering it. A
 * re-tune could not get there — the pad count would have to go from twelve to a
 * thousand and the pads would have to stop being pads — so this is a second
 * generator rather than a fork of the first, and `procedural-tree` keeps the
 * garden's cloud-pruned trees.
 *
 * ## The one hard constraint: no smooth union
 *
 * The plan's answer to both halves of this object is one marched SDF primitive
 * that smooth-unions a displaced cluster (§6b item 3). That primitive does not
 * exist, and this module deliberately does not ask for it — it is the control
 * arm of that decision. Everything here is built from the six exact analytic
 * kinds the scenery graph already publishes, so what it can and cannot reach is
 * evidence about whether the new kind is required.
 *
 * Three techniques stand in for the smooth union, and all three are honest
 * unions:
 *
 *  - **A tapered tube is a cone chain with a sphere at every node.** A sphere of
 *    the node's own radius exactly covers the flat cap the cone ends in, so the
 *    chain's union is a round-capped tapered tube with no seam at any bend.
 *    This is the melted-wax profile as far as a union can carry it.
 *  - **A fillet is placed, not solved.** A smooth minimum is a ball rolled along
 *    a seam; where two root fingers leave the stump together, that ball is put
 *    in the notch explicitly and sized from the gap it has to bridge.
 *  - **Overlap does the rest.** Parts are made big enough that the seam between
 *    any two of them is inside a third. It is the reason the root fingers are
 *    nearly as thick as the stump and the florets are seated most of the way
 *    under the surface they sit on.
 *
 * ## Why a rotated ellipsoid here is always a spheroid
 *
 * `alongAxis` returns the *shortest-arc* rotation taking local +Y onto a
 * direction, which fixes one axis and leaves the roll about it unspecified. A
 * triaxial ellipsoid rotated that way would spin its two cross-axes with the
 * direction and read as a different shape at every azimuth. So anything rotated
 * in this module has `radius.x === radius.z` and varies only along local +Y —
 * florets and fillets are both round in plan anyway. Unrotated ellipsoids, which
 * here means the canopy lobes, are free to be triaxial because their axes are
 * the world's.
 *
 * ## What actually limits the canopy, measured
 *
 * Not the lattice. The plan's gap G2 says one material owner per voxel is the
 * real limit on fine detail, and for a *voxelized field* it is — but a scenery
 * primitive's primary visibility never goes through the voxel payload.
 * `traceStatic` in `webgpu-svo-dry-scene.ts` opens with `traceScenePrimitives`,
 * a BVH over the authored records that solves each one exactly, and only then
 * walks the SVO for the fields that have no analytic form. Its own comment says
 * so: "owner payloads only accelerate non-analytic scene fields". Rendering this
 * canopy at 25 mm, 50 mm and 100 mm cells produced three byte-identical frames.
 * A floret is silhouetted exactly at any size; what the lattice coarsens is the
 * occupancy the cone tracer lights it with, not its shape.
 *
 * What limits it is `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES = 4096` — the BVH's
 * capacity, shared by every object in the frame, and a cliff rather than a
 * slope: above it `buildSvoPrimitiveCandidates` is simply not called and the
 * whole set stops being drawn. Covering this crown's envelope at the overlap a
 * floret needs before it fuses costs about `2 * area / (pi * w^2)` records, so
 * halving the floret quadruples the bill. That is the entire economy of this
 * object, and it is why the floret is a flattened cap rather than a bead and why
 * nothing is built that the camera cannot see.
 *
 * ## Determinism
 *
 * One seed in, identical geometry out, every rebuild — the sparse publication
 * cache's static revision depends on it. An integer avalanche hash is the only
 * entropy and it moves jitter within slots without ever changing a count, so a
 * re-seed grows a recognisable sibling rather than a different species.
 *
 * These are plain primitive nodes, so unlike `kind: "tree"` a specimen has no
 * gust: sway is resolved during expansion and only the tree node opts into it.
 */

/**
 * What makes this species this species: shape only, with no position, seed or
 * key in it, so one form can plant a grove.
 *
 * Every number here was settled against a render and says what moving it does.
 * The two that are not parameters are the key and the seed, which are identity.
 */
export interface BonsaiForm {
  /** Ground under the stand to the top of the crown, in metres. */
  readonly height_m: number;
  /** Outer plan extent of the crown from its own centre, on x and z. */
  readonly crownRadius_m: readonly [number, number];
  /**
   * Total vertical thickness of the crown *envelope* — florets included, not the
   * lobes underneath. Against `crownRadius_m` this is the single number that
   * decides whether the specimen reads as a flat overhanging canopy or as a
   * ball; the reference's is a little over four to one.
   */
  readonly crownThickness_m: number;
  /**
   * How far the crown's centre sits from the trunk, along the spec's `lean`.
   * This is the overhang, and it is what puts canopy over water.
   */
  readonly crownOverhang_m: number;
  /**
   * How far the rim lobes hang below the middle ones, in crown thicknesses.
   * Zero is a flat slab; this is what rounds the underside without doming the
   * top.
   */
  readonly crownDroop: number;
  /**
   * Canopy lobes. They are *balls* laid out in a plane, not pancakes — the crown
   * is flat because of the arrangement, not because each lobe is flattened — so
   * this and `crownThickness_m` are not independent: the plate wants to be about
   * one lobe thick, and a count that disagrees reads as a lumpy dinner plate
   * however the rest is tuned. Tiling the crown's plan area is the constraint.
   */
  readonly lobes: number;
  /**
   * Lobe overlap. Near one the lobes merely meet, which is what keeps the clefts
   * the crown is legible by; much above it they dissolve into one mass. A
   * cauliflower head is a set of heads.
   */
  readonly lobeFill: number;
  /** Spread of lobe sizes about the mean, as a fraction. Zero is a paving pattern. */
  readonly lobeSwell: number;
  /**
   * Nodules per lobe — the middle octave, and the clumping the eye reads as
   * cauliflower. Capped internally against the plate's own thickness: a nodule
   * wider than the lobe it sits on has stopped being a bump on a plate and
   * become the plate.
   */
  readonly bumpsPerLobe: number;
  /**
   * Florets cast onto each lobe's nodule envelope, before the occlusion cull.
   * The budget knob: this times `lobes` is very nearly the whole record count.
   */
  readonly floretsPerLobe: number;
  /**
   * Half-width of one floret — the granulation half-period on the crown, and the
   * number this whole approach is judged on. Halving it quadruples the records
   * needed to keep the surface fused; see the header.
   */
  readonly floretRadius_m: number;
  /**
   * How deep a floret is against how wide, as a fraction.
   *
   * A floret is a squashed cap, not a bead. Two reasons, and the second decides
   * it. A real cauliflower floret is a flattened facet pressed against its
   * neighbours, so the shape is simply righter — and a cap covers `1/f^2` times
   * the envelope area of a sphere standing as far proud of the surface, three
   * times over at the authored value, so the same records buy three times the
   * granulation frequency. On an object whose entire problem is records per unit
   * of surface, that is not a detail.
   */
  readonly floretFlatten: number;
  /**
   * How far under the envelope a floret is seated, in its own depth. Sitting
   * them on it leaves them tangent and the crown reads as a bunch of grapes;
   * pushing them in is the only overlap control a union has, and it is what a
   * smooth union would have been asked for.
   */
  readonly floretSeat: number;
  /**
   * Spread of floret size, drawn once per nodule and only jittered within it, so
   * the crown comes out as clumps of like-sized grain rather than one uniform
   * stipple. That correlation is most of what separates a cauliflower from a
   * raspberry, and it costs nothing.
   */
  readonly floretGrain: number;
  /** Stump radius. Every length below the crown is a multiple of it. */
  readonly boleRadius_m: number;
  /** Fraction of `height_m` the fused stump stands before the trunks leave it. */
  readonly stumpShare: number;
  /** Fraction of `height_m` at which the trunks fork into limbs. */
  readonly forkShare: number;
  /**
   * Trunks rising from the stump. This species is a *multi-trunk*: several
   * smooth trunks of much the same thickness leave one low fused stump, splay
   * apart as they climb and fork near the canopy — a candelabra. Two reads as a
   * fork, four or five as the reference.
   */
  readonly trunks: number;
  /** How far a trunk's top stands off the axis, in stump radii. The vase. */
  readonly trunkSplay: number;
  /** Trunk radius as a fraction of the stump's, where it leaves the stump. */
  readonly trunkWidth: number;
  /**
   * Limb radius as a fraction of the trunk's, where the trunk ends. Under one,
   * always: a limb wider than the trunk it forks from shows its own base disc
   * sticking out through the fork, which is the one place in the object where a
   * cone chain's uncapped near end is visible.
   */
  readonly limbWidth: number;
  /**
   * Limbs allowed per trunk. With a lobe apiece a thirteen-lobe crown would be
   * held up by thirteen poles and the gesture under it would be a thicket; the
   * lobes are ordered from the middle outward, so the limbs that do exist reach
   * the inner ones and the rim rests on its neighbours as the reference's does.
   */
  readonly limbsPerTrunk: number;
  /**
   * How far a limb bows outward at mid-run, as a fraction of its own horizontal
   * reach. The reference's limbs are a vase, not a fan, and a straight run
   * between the same two endpoints loses the whole gesture.
   */
  readonly limbBow: number;
  /**
   * Root fingers running out over the ground. Not a buttress skirt: they leave
   * the stump low, run out almost level with the ground and end in blunt toes
   * among whatever is bedded around them.
   */
  readonly roots: number;
  /** Finger length in stump radii. */
  readonly rootReach: number;
  /**
   * Finger radius as a fraction of the stump's. Fingers this thick overlap the
   * stump and one another where they meet it, which is what makes the base read
   * as one fused mass rather than as legs.
   */
  readonly rootWidth: number;
  /**
   * How much a finger narrows along its run. A finger that narrows to a point is
   * a spike and a ring of spikes is a starfish, so this stays small and the toe
   * is fatter than the run that reaches it.
   */
  readonly rootTaper: number;
  /** Palette naming the trunk, root and limb ramp. */
  readonly barkPalette: string;
  /** Palette naming the canopy ramp. */
  readonly canopyPalette: string;
  /** Value of a limb on `barkPalette`. Stump and roots sit just under it. */
  readonly barkValue: number;
  /** Value of a sunlit floret. Crevices and lobe cores ramp down from it. */
  readonly canopyValue: number;
  /**
   * Refuses to publish past this many primitives, so a re-tune cannot silently
   * spend another object class's share of the 4 096-leaf scene budget.
   */
  readonly maximumLeaves: number;
}

export interface BonsaiSpec extends BonsaiForm {
  /** Key prefix. Every emitted node is published as `${key}/...`. */
  readonly key: string;
  /** Where the trunk meets the ground, in world metres on the ground plane. */
  readonly at_m: readonly [number, number];
  /**
   * Ground height at a world point. `terrainHeightAt(scene.terrain, x, z)` for
   * anything standing on a heightfield — the baked surface the renderer draws,
   * not a re-derivation of it — because the toes are placed on whatever this
   * returns and a specimen on a bank is the normal case, not the exception.
   */
  readonly groundHeightAt: (x_m: number, z_m: number) => number;
  /**
   * Direction the crown overhangs toward, in XZ. Need not be normalized.
   *
   * Placement rather than form, and deliberately a bearing rather than a pond:
   * a specimen leans out over whatever it is standing beside, and the generator
   * has no business knowing what that is.
   */
  readonly lean: readonly [number, number];
  /** The only entropy. Any integer. */
  readonly seed: number;
}

/** A canopy lobe, as the plan hands it to a test or to a culling pass. */
export interface BonsaiLobe {
  readonly center_m: Vec3;
  readonly radius_m: Vec3;
}

export interface BonsaiPlan {
  readonly spec: BonsaiSpec;
  /** Ready to splice into `SceneryGraph.nodes`. One group, everything under it. */
  readonly nodes: readonly SceneryNode[];
  /** Published primitives. The group is not a leaf and is not counted. */
  readonly leafCount: number;
  /** World-space extent of the published surface, florets included. */
  readonly bounds_m: { readonly min: Vec3; readonly max: Vec3 };
  /** Where each root finger's toe enters the ground, in world metres. */
  readonly rootFeet_m: readonly Vec3[];
  /** Lobe envelopes in world metres, before nodules and florets thicken them. */
  readonly lobes: readonly BonsaiLobe[];
  /** Radius of one canopy nodule, derived from the lobe size and count. */
  readonly bumpRadius_m: number;
  /**
   * Summed floret cap area over the crown's own envelope area. One is tangent
   * florets with gaps between them; two is the overlap at which they fuse. The
   * one number that says whether a form's `floretsPerLobe` is enough for its
   * `floretRadius_m`, without rendering it.
   */
  readonly floretCoverage: number;
}

/**
 * The reference's specimen: a wide flat-topped multi-trunk overhanging a pond,
 * about 0.65 m across and 0.40 m tall on a 1.8 m stage.
 *
 * Thirteen lobes on a 0.17 m plate is the tiling — one lobe thick — and 106
 * florets a lobe at 29 mm half-width is the finest granulation that both covers
 * that envelope and fits a 1 500-leaf share of the scene. Every one of those
 * three numbers is against a wall; see the header, and `plan.floretCoverage`,
 * which comes out at 2.4 here — barely past the two at which florets stop
 * reading as separate beads.
 *
 * The density is set by the *worst* seed rather than this one. Culling is
 * geometric, so the record count swings about ten per cent as the lobes move,
 * and a form tuned on its own seed overruns the budget on a re-seed — which is
 * what `tests/bonsai.test.ts` found. Every count here is the value at which four
 * hundred seeds all fit and all still cover past two.
 */
export const BONSAI_POND_CANOPY: BonsaiForm = Object.freeze({
  height_m: 0.40,
  crownRadius_m: [0.34, 0.30] as const,
  crownThickness_m: 0.17,
  crownOverhang_m: 0.10,
  crownDroop: 0.30,
  lobes: 13,
  lobeFill: 1.00,
  lobeSwell: 0.48,
  bumpsPerLobe: 16,
  floretsPerLobe: 106,
  floretRadius_m: 0.0293,
  floretFlatten: 0.56,
  floretSeat: 0.55,
  floretGrain: 0.52,
  boleRadius_m: 0.048,
  stumpShare: 0.09,
  forkShare: 0.44,
  trunks: 4,
  trunkSplay: 1.9,
  trunkWidth: 0.50,
  limbWidth: 0.72,
  limbsPerTrunk: 2,
  limbBow: 0.30,
  roots: 8,
  rootReach: 2.5,
  rootWidth: 0.58,
  rootTaper: 0.22,
  barkPalette: "stone",
  canopyPalette: "clay",
  barkValue: 0.91,
  canopyValue: 0.85,
  maximumLeaves: 1_500,
});

/**
 * The same species grown as a standard: half again as tall, a crown a third
 * narrower and deeper, three trunks instead of five, and no overhang to speak
 * of. What a courtyard or a gallery wants — something to stand *beside* rather
 * than under.
 *
 * The florets are coarser and there are more per lobe because the crown is
 * deeper and its envelope larger; the record count comes out close to the pond
 * form's, which is the point of keeping the budget in the form.
 */
export const BONSAI_COURTYARD_STANDARD: BonsaiForm = Object.freeze({
  ...BONSAI_POND_CANOPY,
  height_m: 0.62,
  crownRadius_m: [0.24, 0.22] as const,
  crownThickness_m: 0.24,
  crownOverhang_m: 0.03,
  crownDroop: 0.18,
  lobes: 9,
  bumpsPerLobe: 14,
  floretsPerLobe: 150,
  floretRadius_m: 0.029,
  boleRadius_m: 0.044,
  stumpShare: 0.05,
  forkShare: 0.52,
  trunks: 3,
  trunkSplay: 1.4,
  limbsPerTrunk: 3,
  roots: 6,
  rootReach: 2.1,
});

/**
 * A shelf specimen at a third the scale: seven lobes, three trunks, and florets
 * held at half the pond form's half-width because at this size the granulation
 * is what carries the illusion and the envelope is small enough to afford it.
 *
 * It is the honest end of the record economy in the other direction: shrink the
 * object and the same budget buys a much finer surface, because the area to
 * cover falls with the square.
 */
export const BONSAI_SHELF_MINIATURE: BonsaiForm = Object.freeze({
  ...BONSAI_POND_CANOPY,
  height_m: 0.155,
  crownRadius_m: [0.125, 0.115] as const,
  crownThickness_m: 0.065,
  crownOverhang_m: 0.035,
  lobes: 9,
  bumpsPerLobe: 12,
  floretsPerLobe: 176,
  floretRadius_m: 0.0114,
  boleRadius_m: 0.019,
  trunks: 3,
  limbsPerTrunk: 3,
  roots: 6,
});

/** Integer avalanche hash in [0, 1). The same tree on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

/** Hash mapped to [-1, 1). */
const hashSigned = (n: number): number => 2 * hash01(n) - 1;

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/** The narrow neutral band every white surface in these sets is authored in. */
const clampValue = (value: number): number => Math.min(.92, Math.max(.62, value));

const add = (a: Vec3, b: Vec3): Vec3 => V(a.x + b.x, a.y + b.y, a.z + b.z);
const sub = (a: Vec3, b: Vec3): Vec3 => V(a.x - b.x, a.y - b.y, a.z - b.z);
const scale = (a: Vec3, k: number): Vec3 => V(a.x * k, a.y * k, a.z * k);
const length = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);

function normalize(a: Vec3): Vec3 {
  const magnitude = length(a);
  if (!(magnitude > 0)) throw new Error("Bonsai direction must have nonzero length");
  return scale(a, 1 / magnitude);
}

/** Evenly spread directions on the unit sphere; `spin` decorrelates two sets. */
function fibonacciDirection(index: number, count: number, spin: number): Vec3 {
  const y = 1 - 2 * (index + .5) / count;
  const ring = Math.sqrt(Math.max(0, 1 - y * y));
  const angle = GOLDEN_ANGLE * index + spin;
  return V(ring * Math.cos(angle), y, ring * Math.sin(angle));
}

/** Squared position inside an axis-aligned ellipsoid: below one is inside it. */
function ellipsoidField(point: Vec3, center: Vec3, radius: Vec3): number {
  const dx = (point.x - center.x) / radius.x;
  const dy = (point.y - center.y) / radius.y;
  const dz = (point.z - center.z) / radius.z;
  return dx * dx + dy * dy + dz * dz;
}

/**
 * How far inside a lobe a point sits, in metres, measured out from the lobe's
 * centre. The field is quadratic, so the surface along that ray is at
 * `|d| / sqrt(field)` and the depth is the rest of the way.
 *
 * Radial rather than perpendicular, which overstates depth near a flat face — in
 * the direction that matters, because an overstatement culls a nodule that
 * really was buried rather than sparing one that was not.
 */
function lobeDepth(point: Vec3, lobe: BonsaiLobe): number {
  const field = ellipsoidField(point, lobe.center_m, lobe.radius_m);
  if (!(field < 1)) return 0;
  if (!(field > 0)) return Math.min(lobe.radius_m.x, lobe.radius_m.y, lobe.radius_m.z);
  return length(sub(point, lobe.center_m)) * (1 / Math.sqrt(field) - 1);
}

/**
 * How many voxels across one floret is on a given lattice.
 *
 * Kept because it is the first thing anyone asks about this object, and because
 * the answer turned out to be that it does not matter: primary visibility for a
 * scenery primitive is a BVH over exact records and never consults the voxel
 * payload, so a floret is silhouetted exactly whether it spans four cells or a
 * quarter of one. What the number does bound is how coarsely the cone tracer's
 * occupancy lights the crown. See the header.
 */
export function bonsaiFloretVoxelSpan(form: BonsaiForm, cellSize_m: number): number {
  if (!(cellSize_m > 0)) throw new RangeError("Bonsai floret voxel span needs a positive cell size");
  return 2 * form.floretRadius_m / cellSize_m;
}

/**
 * Grow one specimen.
 *
 * Pure: it allocates nothing, touches no builder and publishes no primitive. The
 * caller splices `plan.nodes` into a scenery graph, and a test can ask where the
 * crown reaches and where the toes landed before anything is voxelized.
 */
export function planBonsai(spec: BonsaiSpec): BonsaiPlan {
  const {
    key, at_m: [standX, standZ], groundHeightAt, height_m: height,
    crownThickness_m: crownThickness, crownOverhang_m: overhang,
    boleRadius_m: boleRadius, trunks: trunkCount, roots: rootCount, lobes: lobeCount,
    bumpsPerLobe, floretsPerLobe, floretRadius_m: floretRadius, seed,
  } = spec;
  if (!(height > 0) || !(boleRadius > 0) || !(crownThickness > 0)) {
    throw new RangeError("Bonsai needs a positive height, stump radius and crown thickness");
  }
  if (!(spec.crownRadius_m[0] > 0) || !(spec.crownRadius_m[1] > 0)) throw new RangeError("Bonsai crown radii must be positive");
  if (!Number.isInteger(trunkCount) || trunkCount < 2) throw new RangeError("Bonsai needs at least two trunks");
  if (!Number.isInteger(rootCount) || rootCount < 3) throw new RangeError("Bonsai needs at least three root fingers");
  if (!Number.isInteger(lobeCount) || lobeCount < 3) throw new RangeError("Bonsai needs at least three canopy lobes");
  if (!Number.isInteger(bumpsPerLobe) || bumpsPerLobe < 1) throw new RangeError("Bonsai needs at least one nodule per lobe");
  if (!Number.isInteger(floretsPerLobe) || floretsPerLobe < 1) throw new RangeError("Bonsai needs at least one floret per lobe");
  if (!Number.isInteger(spec.limbsPerTrunk) || spec.limbsPerTrunk < 1) throw new RangeError("Bonsai needs at least one limb per trunk");
  if (!(floretRadius > 0)) throw new RangeError("Bonsai floret half-width must be positive");
  if (!(spec.floretFlatten > 0 && spec.floretFlatten <= 1)) throw new RangeError("Bonsai floret flatten must be in (0, 1]");
  if (!(spec.lobeFill > 0)) throw new RangeError("Bonsai lobe fill must be positive");
  if (!(spec.limbWidth > 0 && spec.limbWidth < 1)) throw new RangeError("Bonsai limb width must be under its trunk's");

  const groundY = groundHeightAt(standX, standZ);
  /** Ground under a point in the specimen's own frame, relative to the stand. */
  const groundLocal = (x: number, z: number): number => groundHeightAt(standX + x, standZ + z) - groundY;

  const lean = (() => {
    const [x, z] = spec.lean;
    const magnitude = Math.hypot(x, z);
    return magnitude > 0 ? V(x / magnitude, 0, z / magnitude) : V(1, 0, 0);
  })();
  const across = V(-lean.z, 0, lean.x);
  /** A direction in the plan, measured from the lean. */
  const bearing = (angle: number): Vec3 =>
    add(scale(lean, Math.cos(angle)), scale(across, Math.sin(angle)));

  const nodes: SceneryNode[] = [];
  let minimum = V(Infinity, Infinity, Infinity);
  let maximum = V(-Infinity, -Infinity, -Infinity);
  /** Every emitted primitive reports the box that bounds it, in local metres. */
  const cover = (center: Vec3, radius: Vec3 | number): void => {
    const r = typeof radius === "number" ? V(radius, radius, radius) : radius;
    minimum = V(Math.min(minimum.x, center.x - r.x), Math.min(minimum.y, center.y - r.y), Math.min(minimum.z, center.z - r.z));
    maximum = V(Math.max(maximum.x, center.x + r.x), Math.max(maximum.y, center.y + r.y), Math.max(maximum.z, center.z + r.z));
  };

  const bark = (value: number): SceneryMaterial => ({ palette: spec.barkPalette, value: clampValue(value) });
  const canopy = (value: number): SceneryMaterial => ({ palette: spec.canopyPalette, value: clampValue(value) });

  /**
   * Tags, and they are load-bearing twice over.
   *
   * `svoMaterialFunctionIdForEnvironmentProxy` picks a surface closure by
   * matching a regex against the group name followed by the tags, so `porcelain`
   * is on every part on purpose: this specimen is one fired ceramic surface from
   * the toes to the crown, not wood under foliage. It has to be spelled and it
   * has to stay spelled — `tests/bonsai.test.ts` pins the closure for exactly
   * that reason.
   *
   * The group is left unset, so every part inherits the enclosing group node and
   * one specimen is one click target with its parts still individually
   * addressable. That means `key` is in the semantic string, and the wood and
   * stone patterns are tested before the ceramic one: a key spelled `old-tree`
   * or `stone-bonsai` would restyle the whole object. Keys here name the
   * specimen, not its material.
   */
  const partTags = (role: string): string[] => ["bonsai", role, "porcelain"];

  const sphere = (id: string, center: Vec3, radius: number, material: SceneryMaterial, role: string): void => {
    nodes.push({
      kind: "ellipsoid", id: `${key}/${id}`, tags: partTags(role),
      place: { position: center }, radius: V(radius, radius, radius), material,
    });
    cover(center, radius);
  };

  /**
   * One run of a tapered tube, plus the sphere that rounds off its far end.
   *
   * The sphere is not decoration. A cone ends in a flat disc, so a chain of them
   * shows a crescent notch at every bend; a sphere of the end radius covers that
   * disc exactly and the union comes out as one continuous round-capped tube.
   */
  const run = (
    id: string, from: Vec3, to: Vec3, fromRadius: number, toRadius: number,
    material: SceneryMaterial, role: string,
  ): void => {
    const axis = sub(to, from);
    const halfHeight = .5 * length(axis);
    if (!(halfHeight > 0)) throw new Error(`Bonsai run ${id} has zero length`);
    const center = scale(add(from, to), .5);
    nodes.push({
      kind: "cone", id: `${key}/${id}`, tags: partTags(role),
      place: { position: center, orientation: alongAxis(axis) },
      baseRadius: fromRadius, topRadius: toRadius, halfHeight, material,
    });
    // The two end spheres, not one about the centre: a truncated cone lies
    // inside the hull of them, and the loose bound would report a trunk reaching
    // a hand's breadth below the ground it is standing on.
    cover(from, fromRadius);
    cover(to, toRadius);
    sphere(`${id}-node`, to, toRadius, material, role);
  };

  // ---- Stump, root fingers and trunks --------------------------------------
  //
  // Read off the reference rather than assumed, and the assumption was wrong
  // twice. This is not a single bole with a buttressed flare: it is a multi-
  // trunk candelabra, and the roots are not a cone skirt but flat fingers lying
  // on the ground. Building it the other way produced a mushroom with a ring of
  // tabs round its foot; the difference is structural and no amount of retuning
  // a cone reaches it.
  const stumpTopY = spec.stumpShare * height;
  const forkY = spec.forkShare * height;
  // Buried at the base, so the stump never shows a rim where it meets ground the
  // terrain has moved a few millimetres since this was authored.
  run("stump", V(0, -.6 * boleRadius, 0), V(0, stumpTopY, 0),
    1.5 * boleRadius, 1.15 * boleRadius, bark(spec.barkValue + .01), "stem");

  // Root fingers. Each node is placed on the ground under it and then lifted by
  // a share of the stump that decays along the run, so a finger leaves the stump
  // at stump height and is lying on the ground a third of the way out — and it
  // follows the ground down as it goes, which is the whole reason this generator
  // is handed a height query rather than a plane.
  const rootFeet_m: Vec3[] = [];
  const rootPoints: Vec3[][] = [];
  const ROOT_STEPS = 3;
  for (let index = 0; index < rootCount; index += 1) {
    const angle = 2 * Math.PI * (index + .30 * hashSigned(seed + 17 * index)) / rootCount;
    const reach = boleRadius * spec.rootReach * (.85 + .30 * hash01(seed + 17 * index + 1));
    const outward = bearing(angle);
    const width = spec.rootWidth * (.90 + .20 * hash01(seed + 17 * index + 2));
    const points: Vec3[] = [];
    for (let step = 0; step <= ROOT_STEPS; step += 1) {
      const t = step / ROOT_STEPS;
      const radial = reach * Math.max(.22, t) ** .62;
      const x = outward.x * radial, z = outward.z * radial;
      points.push(V(x, groundLocal(x, z) + stumpTopY * (1 - t) ** 2.2 - .25 * boleRadius * t, z));
    }
    rootPoints.push(points);
    const radiusAt = (t: number): number => boleRadius * width * (1 - spec.rootTaper * t);
    for (let step = 0; step < ROOT_STEPS; step += 1) {
      run(`root-${index}-${step}`, points[step], points[step + 1],
        radiusAt(step / ROOT_STEPS), radiusAt((step + 1) / ROOT_STEPS),
        bark(spec.barkValue - .03 + .04 * hash01(seed + 23 * index + step)), "root");
    }
    sphere(`root-${index}-toe`, points[ROOT_STEPS], 1.24 * radiusAt(1),
      bark(spec.barkValue - .02 + .03 * hash01(seed + 29 * index)), "root");
    rootFeet_m.push(V(standX + points[ROOT_STEPS].x, groundY + points[ROOT_STEPS].y, standZ + points[ROOT_STEPS].z));
  }

  // The fillet in each notch between neighbouring fingers — a sphere, and it
  // took two wrong shapes to get there. A flattened spheroid across the pair is
  // what a smooth union would leave, but the roll of `alongAxis` is unspecified
  // so it has to be a disc, and a disc wide enough to bridge two fingers reaches
  // as far above and below them: a ring of those made the base wear a skirt of
  // petals. A sphere in the notch has no plane to stick out of, and a smoothing
  // radius is a rolling ball anyway — this is the ball. Its radius is clamped
  // against the ground under it so a fillet cannot be buried in the bank.
  for (let index = 0; index < rootCount; index += 1) {
    const here = rootPoints[index];
    const next = rootPoints[(index + 1) % rootCount];
    for (let step = 1; step <= 2; step += 1) {
      const center = scale(add(here[step], next[step]), .5);
      const clearance = center.y - groundLocal(center.x, center.z);
      const radius = Math.min(.46 * length(sub(next[step], here[step])), Math.max(.20 * boleRadius, .9 * clearance));
      if (!(radius > 0)) continue;
      sphere(`fillet-${index}-${step}`, center, radius,
        bark(spec.barkValue - .03 + .04 * hash01(seed + 37 * index + step)), "root");
    }
  }

  // The trunks. Radial offset goes as t^1.8 so each leaves the stump nearly
  // upright and is splayed by the time it forks: that curve is what makes the
  // group read as one tree opening out, where a straight splay reads as a wigwam.
  const TRUNK_STEPS = 4;
  const trunkTops: { readonly at: Vec3; readonly radius: number }[] = [];
  for (let index = 0; index < trunkCount; index += 1) {
    const angle = 2 * Math.PI * (index + .34 * hashSigned(seed + 131 * index)) / trunkCount + .4;
    const outward = bearing(angle);
    const splay = boleRadius * spec.trunkSplay * (.80 + .40 * hash01(seed + 131 * index + 1));
    const rise = forkY - stumpTopY;
    if (!(rise > 0)) throw new RangeError("Bonsai fork share must exceed its stump share");
    const at = (t: number): Vec3 => add(
      scale(outward, splay * t ** 1.8),
      add(V(0, stumpTopY + rise * t, 0), scale(lean, .22 * rise * t * t)),
    );
    const radius = boleRadius * spec.trunkWidth * (.90 + .25 * hash01(seed + 131 * index + 2));
    for (let step = 0; step < TRUNK_STEPS; step += 1) {
      const t0 = step / TRUNK_STEPS, t1 = (step + 1) / TRUNK_STEPS;
      run(`trunk-${index}-${step}`, at(t0), at(t1),
        radius * (1 - .40 * t0), radius * (1 - .40 * t1),
        bark(spec.barkValue + .02 * hash01(seed + 137 * index + step)), "stem");
    }
    trunkTops.push({ at: at(1), radius: radius * .60 });
  }

  // ---- Canopy lobes -------------------------------------------------------
  // The crown is one plate. Lobe centres go on a sunflower disc inset by the
  // lobe's own plan radius, so the outermost floret lands on `crownRadius_m`
  // rather than beyond it, and they droop with the square of the radius — which
  // is the difference between a flat top with a rounded underside and a dome.
  const crownCenter = add(scale(lean, overhang), V(0, height - .5 * crownThickness, 0));
  const lobePlanRadius = spec.lobeFill * Math.sqrt(spec.crownRadius_m[0] * spec.crownRadius_m[1] / lobeCount);
  // `crownThickness_m` is the envelope, because that is the number the reference
  // is read for. Nodules stand proud of the lobe by most of their radius and
  // florets proud of those, so the lobe underneath is what is left of the plate
  // after both — which is the equation solved here.
  const lobeHalfThickness = Math.max(.18 * crownThickness, (.5 * crownThickness - .38 * floretRadius) / 1.495);
  // A lobe is a squashed ball, so the area to cover is dominated by its two
  // faces: n nodules of radius r cover it once over when n pi r^2 = 2 pi a^2,
  // and the factor above one is the overlap that fuses them. Capped against the
  // plate's own half-thickness for the reason `bumpsPerLobe` states.
  const bumpRadius = Math.min(1.1 * lobeHalfThickness, 1.14 * lobePlanRadius * Math.sqrt(2 / bumpsPerLobe));
  const ringRadius = V(
    Math.max(0, spec.crownRadius_m[0] - lobePlanRadius), 0,
    Math.max(0, spec.crownRadius_m[1] - lobePlanRadius),
  );
  const lobes: BonsaiLobe[] = [];
  for (let index = 0; index < lobeCount; index += 1) {
    const u = Math.sqrt((index + .5) / lobeCount);
    const angle = GOLDEN_ANGLE * index + 1.31 * hash01(seed + 5);
    const center = add(crownCenter, V(
      ringRadius.x * u * Math.cos(angle),
      -spec.crownDroop * crownThickness * u * u + .11 * crownThickness * hashSigned(seed + 43 * index),
      ringRadius.z * u * Math.sin(angle),
    ));
    const swell = 1 - .5 * spec.lobeSwell + spec.lobeSwell * hash01(seed + 43 * index + 1);
    lobes.push({
      center_m: center,
      radius_m: V(
        lobePlanRadius * swell * (1 + .10 * hashSigned(seed + 43 * index + 2)),
        lobeHalfThickness * (.88 + .20 * hash01(seed + 43 * index + 3)),
        lobePlanRadius * swell * (1 + .10 * hashSigned(seed + 43 * index + 4)),
      ),
    });
  }

  // ---- Limbs --------------------------------------------------------------
  // Each limb ends inside the lobe it holds up, because a canopy resting on
  // nothing is the most obvious way a generated tree reads as generated.
  const LIMB_STEPS = 4;
  const limbCount = Math.min(lobeCount, spec.limbsPerTrunk * trunkCount);
  for (let index = 0; index < limbCount; index += 1) {
    const lobe = lobes[index];
    const trunk = trunkTops[index % trunkCount];
    const target = add(lobe.center_m, scale(sub(trunk.at, lobe.center_m), .42));
    const start = add(trunk.at, V(0, -.22 * (forkY - stumpTopY) * hash01(seed + 59 * index), 0));
    const delta = sub(target, start);
    const outward = normalize(V(delta.x, 0, delta.z));
    const bow = scale(outward, spec.limbBow * Math.hypot(delta.x, delta.z) * (.8 + .4 * hash01(seed + 59 * index + 1)));
    const at = (t: number): Vec3 => add(add(start, scale(delta, t)), scale(bow, Math.sin(Math.PI * t)));
    const baseRadius = trunk.radius * spec.limbWidth * (.88 + .24 * hash01(seed + 59 * index + 2));
    // A cone chain caps its far end at every node and its near end nowhere, so
    // the first segment of a limb would show a bare disc where it leaves the
    // fork. One sphere closes it, and the fork reads as a fork rather than as
    // two parts pushed together.
    sphere(`limb-${index}-fork`, at(0), baseRadius, bark(spec.barkValue), "stem");
    for (let step = 0; step < LIMB_STEPS; step += 1) {
      const t0 = step / LIMB_STEPS, t1 = (step + 1) / LIMB_STEPS;
      run(`limb-${index}-${step}`, at(t0), at(t1),
        baseRadius * (1 - .52 * t0), baseRadius * (1 - .52 * t1),
        bark(spec.barkValue - .01 + .02 * hash01(seed + 61 * index + step)), "stem");
    }
  }

  // ---- Canopy: lobe cores, nodules, florets --------------------------------
  // Three scales, and each earns its records for a different reason. The lobe
  // core is opacity: without it a ray that threads between two florets sees sky
  // through the crown. The nodules are the mass the eye reads as clumping. The
  // florets are the surface, and they are what the budget is actually spent on.
  //
  // Only the shell is built. At the overlap a floret needs before it fuses, most
  // of every nodule is inside another nodule and most of every lobe is inside
  // another lobe; a floret placed there can never be hit. Culling those is the
  // difference between spending the budget on the surface the camera sees and
  // spending nine tenths of it on the inside of a solid.
  //
  // Nodules are therefore all placed before any floret is: a floret has to be
  // tested against every nodule in the crown, not just the ones on its own lobe,
  // and a single pass would only know about the lobes already visited.
  for (let index = 0; index < lobeCount; index += 1) {
    const lobe = lobes[index];
    nodes.push({
      kind: "ellipsoid", id: `${key}/lobe-${index}`, tags: partTags("crown"),
      place: { position: lobe.center_m }, radius: lobe.radius_m,
      material: canopy(spec.canopyValue - .15 + .03 * hash01(seed + 71 * index)),
    });
    cover(lobe.center_m, lobe.radius_m);
  }

  const bumps: { readonly lobe: number; readonly index: number; readonly center: Vec3; readonly radius: number }[] = [];
  for (let index = 0; index < lobeCount; index += 1) {
    const lobe = lobes[index];
    const inset = V(
      Math.max(1e-4, lobe.radius_m.x - .55 * bumpRadius),
      Math.max(1e-4, lobe.radius_m.y - .55 * bumpRadius),
      Math.max(1e-4, lobe.radius_m.z - .55 * bumpRadius),
    );
    for (let bump = 0; bump < bumpsPerLobe; bump += 1) {
      const direction = fibonacciDirection(bump, bumpsPerLobe, 2.4 * hash01(seed + 83 * index));
      const jitter = 1 + .10 * hashSigned(seed + 89 * (index * bumpsPerLobe + bump));
      const center = add(lobe.center_m, V(inset.x * direction.x * jitter, inset.y * direction.y * jitter, inset.z * direction.z * jitter));
      const radius = bumpRadius * (.82 + .34 * hash01(seed + 97 * (index * bumpsPerLobe + bump)));
      // Buried by more than its own radius means it cannot reach any surface.
      if (lobes.some((other, otherIndex) => otherIndex !== index && lobeDepth(center, other) > radius)) continue;
      const rise = .5 + .5 * normalize(sub(center, lobe.center_m)).y;
      sphere(`bump-${index}-${bump}`, center, radius, canopy(spec.canopyValue - .17 + .12 * rise), "crown");
      bumps.push({ lobe: index, index: bump, center, radius });
    }
  }

  // Florets are cast onto the nodule field, not sprinkled over each nodule in
  // turn. Sprinkling was the obvious construction and it is the wrong one: most
  // of a nodule is inside its neighbours, so most of what it is offered is
  // thrown away and the survivors land at whatever density the leftovers
  // happened to be — nowhere near the overlap a floret needs before it fuses.
  //
  // Casting spends the budget where the eye is. One ray per floret leaves the
  // lobe's centre, the floret is seated on the outermost nodule that ray leaves
  // through, and the count is then exactly the number of florets on the visible
  // shell — which is why `floretsPerLobe` is a budget in the units the budget is
  // actually kept in.
  let floretCapArea = 0;
  for (let index = 0; index < lobeCount; index += 1) {
    const lobe = lobes[index];
    const own = bumps.filter((bump) => bump.lobe === index);
    for (let floret = 0; floret < floretsPerLobe; floret += 1) {
      const direction = fibonacciDirection(floret, floretsPerLobe, 2.1 * hash01(seed + 127 * index));
      const surface = V(lobe.radius_m.x * direction.x, lobe.radius_m.y * direction.y, lobe.radius_m.z * direction.z);
      const ray = normalize(surface);
      let reach = length(surface);
      let seat = -1;
      for (const bump of own) {
        const toBump = sub(bump.center, lobe.center_m);
        const along = toBump.x * ray.x + toBump.y * ray.y + toBump.z * ray.z;
        const offSquared = (toBump.x * toBump.x + toBump.y * toBump.y + toBump.z * toBump.z) - along * along;
        const halfChordSquared = bump.radius * bump.radius - offSquared;
        if (!(halfChordSquared > 0)) continue;
        const exit = along + Math.sqrt(halfChordSquared);
        if (exit > reach) { reach = exit; seat = bump.index; }
      }
      const salt = index * bumpsPerLobe + Math.max(0, seat);
      const grain = 1 - .5 * spec.floretGrain + spec.floretGrain * hash01(seed + 113 * salt);
      const wide = floretRadius * grain * (.90 + .20 * hash01(seed + 107 * index + floret));
      const deep = spec.floretFlatten * wide;
      const at = add(lobe.center_m, scale(ray, reach - spec.floretSeat * deep));
      if (lobes.some((other, otherIndex) => otherIndex !== index && lobeDepth(at, other) > deep)) continue;
      if (bumps.some((bump) => bump.lobe !== index && bump.radius - length(sub(at, bump.center)) > deep)) continue;
      const lift = .5 + .5 * ray.y;
      nodes.push({
        kind: "ellipsoid", id: `${key}/floret-${index}-${floret}`, tags: partTags("floret"),
        place: { position: at, orientation: alongAxis(ray) },
        radius: V(wide, deep, wide),
        material: canopy(spec.canopyValue - .05 + .10 * lift + .02 * hashSigned(seed + 109 * index + floret)),
      });
      cover(at, wide);
      floretCapArea += Math.PI * wide * wide;
    }
  }

  const leafCount = nodes.length;
  if (leafCount > spec.maximumLeaves) {
    throw new RangeError(`Bonsai published ${leafCount} primitives, over its ${spec.maximumLeaves}-leaf budget`);
  }

  // The envelope the florets had to cover, as the ellipsoid of the measured
  // extent. Approximate on purpose — it is a ratio to be compared between forms,
  // not a surface integral — using Thomsen's formula, which is within about one
  // per cent for anything as blunt as a crown.
  const half = V(.5 * (maximum.x - minimum.x), .5 * (maximum.y - minimum.y), .5 * (maximum.z - minimum.z));
  const p = 1.6075;
  const envelopeArea = 4 * Math.PI * (((half.x * half.y) ** p + (half.x * half.z) ** p + (half.y * half.z) ** p) / 3) ** (1 / p);

  return {
    spec,
    nodes: [{
      kind: "group", id: key, tags: ["bonsai"],
      place: { units: "metres", position: V(standX, groundY, standZ) },
      children: nodes,
    }],
    leafCount,
    bounds_m: {
      min: V(standX + minimum.x, groundY + minimum.y, standZ + minimum.z),
      max: V(standX + maximum.x, groundY + maximum.y, standZ + maximum.z),
    },
    rootFeet_m,
    lobes: lobes.map(({ center_m, radius_m }) => ({
      center_m: V(standX + center_m.x, groundY + center_m.y, standZ + center_m.z),
      radius_m,
    })),
    bumpRadius_m: bumpRadius,
    floretCoverage: envelopeArea > 0 ? floretCapArea / envelopeArea : 0,
  };
}

/** The graph nodes alone, for a caller that only wants to splice them in. */
export function bonsaiNodes(spec: BonsaiSpec): SceneryNode[] {
  return [...planBonsai(spec).nodes];
}
