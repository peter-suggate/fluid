import type { Vec3 } from "../model";
import type { SceneryMaterial, SceneryNode } from "../scenery-graph";
import {
  OAK_LOBE_ACROSS_SHARE,
  OAK_LOBE_DEPTH_SHARE,
  oakCanopyField,
  oakCanopyPadProgram,
} from "./oak-canopy-field";
import {
  OAK_CROWN_AUTOMATON_STEPS,
  OAK_CROWN_BARE_CELLS,
  OAK_CROWN_BIRTH_NEIGHBOURS,
  OAK_CROWN_CELL_SHARE,
  OAK_CROWN_CROWD_NEIGHBOURS,
  OAK_CROWN_REACH_CELLS,
  OAK_ORDERS_SPREADING,
  oakPublishedOrderCount,
  planOakSkeleton,
  type OakBranchOrder,
  type OakFoliageAttachment,
  type OakSkeleton,
} from "./oak-branching";
import { sweptTubeNodes } from "./swept-tube";

/**
 * The oak: a single dominant bole that dissolves into competing limbs, three
 * branch orders on a phyllotactic spiral, and a billowing crown of foliage
 * masses hung on the limbs themselves rather than floating over them.
 *
 * Reference: <https://80.lv/articles/how-to-create-realistic-3d-oak-tree>, and
 * `docs/oak-tree-species-plan.md` for the whole argument.
 *
 * ## What was taken from the reference, and what could not be
 *
 * The reference is an art pipeline — SpeedTree, Maya LODs, Substance, UE5 — and
 * its three load-bearing techniques have no equivalent here. Its foliage is
 * **branch cards**: alpha-masked planes, which have no interior, cannot be
 * voxelized and have no SDF. Its surfaces are **texture atlases**, and shading
 * here is the voxel cell's surface through a palette and a material closure,
 * with no UVs to atlas. Its levels of detail are **decimated meshes**, and level
 * of detail here is the leaf size fed back into the generator's own derivations.
 *
 * What crossed over is the art direction, and it argues against what
 * `procedural-tree.ts` does:
 *
 *  - a massive base and multiple starts rather than a single dowel;
 *  - branch *orders* planned explicitly, which is the parameter axis this
 *    species exposes;
 *  - foliage placed so **the branches stay visible through it** — which is the
 *    same sentence `bonsai-canopy-pads.ts` arrived at from the other direction,
 *    that a canopy with no sky in it is a disc.
 *
 * ## Why this is a third generator
 *
 * `procedural-tree.ts` grows a niwaki: one trunk, limbs on a golden-angle
 * spiral, foliage as a dozen flattened pads spaced so the tiers read separately.
 * `bonsai.ts` grows a fused multi-trunk under one flat cauliflower plate. An oak
 * is neither, and the difference is not tuning: it is a single dominant bole
 * with strong apical dominance low down, *recursive* branching rather than one
 * limb generation, and a crown whose masses hang on the limbs and are meant to
 * be seen past. Neither existing form re-tunes into that.
 *
 * But nothing new is *built* here. Every run below is a `tapered-sweep` through
 * `sweptTubeNodes` — one record for a whole limb, C¹ through its bends, with its
 * envelope solved rather than authored — and every foliage mass is one
 * `field-program` tape from `oak-canopy-field.ts`. This module is layout and
 * proportion; the machinery is the machinery the bonsai already proved.
 *
 * ## The leaf decides how much oak there is, and that is the whole design
 *
 * The project-wide band law, stated in `bonsai.ts`:
 *
 * > A feature whose period is under about **two leaves** does not render as that
 * > feature; it renders as aliasing. Above about **three leaves** it renders as
 * > geometry.
 *
 * Scaling a real 20 m oak to this specimen's 0.80 m gives a 40 mm bole, 12 mm
 * primaries, 4 mm secondaries, 1.2 mm tertiaries and 0.6 mm twigs, against a
 * three-leaf floor of 2.34 mm at the production leaf and 18.75 mm at the
 * coarsest rung. So:
 *
 *   rung      leaf       floor      branch orders   foliage lattices
 *   depth 0   6.25 mm    18.75 mm   bole, primary   1
 *   depth 1   3.125 mm   9.375 mm   + secondary     2
 *   depth 2   1.5625 mm  4.69 mm    + secondary     3
 *   depth 3   0.78125 mm 2.34 mm    + secondary     3 + the sub-leaf carve
 *
 * Tertiaries and twigs never publish at any rung this stage can hold. They are
 * implied by the foliage's own granulation instead of modelled, which is what
 * the canopy tape's interleaved scatter lattices exist to do.
 *
 * **Two invariants make that a refinement rather than a different tree**, and
 * both are properties of the construction rather than of the tuning:
 *
 *  1. *The crown envelope does not move with the leaf.* A culled order's foliage
 *     is absorbed into its parent's attachment rather than dropped —
 *     `OakFoliageAttachment.share` sums to one at every leaf size — and the pad
 *     radius below is solved from that share by volume, so the total foliage
 *     volume is identical at every rung. A coarse tree has fewer, larger masses
 *     in the same envelope; it is not a smaller tree.
 *  2. *Nothing is ever published under three leaves.* A scale either exists at
 *     the floor or does not exist at all.
 *
 * The bonsai is worth contrasting here, because it is the object being replaced
 * and it does **not** have this property: it publishes exactly 154 records at
 * depth 0 and at depth 3, identical, and only its canopy tape changes with the
 * leaf. Graceful refinement is not inherited from it. It had to be built.
 *
 * ## Determinism
 *
 * One seed in, identical geometry out, every rebuild — the sparse publication
 * cache's static revision depends on it. The only entropy is an integer
 * avalanche hash, in `oak-branching.ts`, and it moves values within slots
 * without ever changing a count, so a re-seed grows a recognisable sibling
 * rather than a different species.
 *
 * These are plain primitive nodes, so like the bonsai and unlike `kind: "tree"`
 * a specimen has no gust. Sway is resolved during expansion and only the tree
 * node opts into it; see `docs/oak-tree-species-plan.md` concern 6.
 */

/**
 * What makes this species this species: shape only, with no position, seed or
 * key in it, so one form can plant a wood.
 *
 * Every number is a parameter and says what moving it does. The numbers here are
 * **starting points to be tuned at `/shape-lab`**, not settled measurements —
 * this is a first specimen, and the loop exists precisely so that the arguments
 * for these values get written after they have been looked at rather than
 * before.
 */
export interface OakForm {
  /** Ground under the bole to the top of the highest foliage mass, in metres. */
  readonly height_m: number;
  /**
   * Bole radius where it leaves the ground.
   *
   * The one number the whole skeleton is measured in: every order's radius is a
   * share of its parent's, so this sets how heavy the tree reads. At the
   * specimen's 0.020 m the bole is 40 mm — 51 leaves at the production leaf and
   * 6.4 at the coarsest, so it is the one part of the tree that draws well at
   * every rung.
   */
  readonly boleRadius_m: number;
  /** Horizontal reach of the widest primary limb, in metres. The crown's radius. */
  readonly spread_m: number;
  /**
   * The branch orders, index 0 being the bole.
   *
   * The species, as data — and the reference's own "schematic planning with
   * colour coding to determine branch levels" made into parameters. See
   * `OakBranchOrder`; `OAK_ORDERS_SPREADING` is the mature specimen's.
   *
   * How many of these actually publish is decided by the leaf and not by the
   * length of this array. Authoring a fourth order is not wasted — it simply
   * waits for a rung fine enough to draw it, and until then its foliage hangs
   * off its parent.
   */
  readonly orders: readonly OakBranchOrder[];
  /**
   * How many branches each order forks, as a multiple of the table's own counts.
   *
   * **The distilled control for branch density**, and it exists because the
   * thing it drives is a nested array: `orders` is four objects of eight fields
   * apiece, which the shape lab renders faithfully as thirty-two controls behind
   * four disclosure triangles. That is the right shape for a *species
   * definition* and the wrong shape for a knob somebody drags while looking at a
   * tree.
   *
   * Multiplicative down the whole ladder, so this is the strongest single
   * control on the object: at 1.0 the hero forks 6 / 5 / 4 for 120 tertiary
   * runs, and at 1.3 it forks 8 / 7 / 5 for 280. The bole is exempt — order 0's
   * `branchCount` is the buttress count, not a fork count.
   *
   * Costs no finer voxel: these are the same diameters, so the same rungs admit
   * them. It costs **records**, roughly linearly, and those go to the cluster
   * arena rather than the field-program one.
   */
  readonly branchDensity: number;
  /**
   * How fast the branches thin, as a multiple of the table's own radius ratios.
   *
   * **The distilled control for how far the branching refines**, because an
   * order is published only when its diameter clears three leaves — so fattening
   * the outer orders does not just thicken them, it *admits* them.
   *
   * The hero sits one rung short of its own tertiaries and this is the number
   * that closes it: order 3 forks at 1.91 mm and needs a 0.64 mm leaf, while
   * production's depth 3 gives 0.78 mm — a miss of 18 %. At about 1.11 the
   * tertiary clears the floor at depth 3 and 120 more runs appear at production,
   * which is the difference between a tree with limbs and a tree with twigs.
   *
   * Above about 1.2 the orders stop reading as a taper and the tree goes woody
   * and stiff. Below 1.0 it loses an order and gets sparser at every rung.
   */
  readonly branchFineness: number;
  /**
   * How long each branch runs, as a multiple of the table's own length fractions.
   *
   * **The distilled control for how far the wood reaches into the crown.** A
   * child's length is a fraction of its parent's, so this compounds down the
   * ladder — at 1.3 a tertiary is 2.2x its authored length, not 1.3x — which is
   * what makes it the right knob for "the branches stop short of the foliage"
   * and the wrong one for small corrections.
   *
   * It does **not** make the tree bigger. `fitToEnvelope` fits the whole object
   * to `height_m` and `spread_m` before anything is published, so lengthening
   * the limbs redistributes the tree rather than growing it: the wood occupies
   * more of the same envelope and the bole shortens against it. That is the
   * point — the crown is seeded from the tips of the *full* skeleton, so limbs
   * that reach further put drawn wood where the foliage already is.
   *
   * Above about 1.4 the orders stop reading as a hierarchy, because a child
   * nearly as long as its parent is a fork rather than a branch.
   */
  readonly branchLength: number;
  /**
   * A foliage mass's plan radius, as a multiple of the crown automaton's cell.
   *
   * **This used to be a conserved volume and it is deliberately not one now.**
   * The tiling it belonged to solved each pad's radius from its share of a fixed
   * total, so a crown with more masses in it had smaller ones — which is exactly
   * backwards once the masses are grown on a lattice rather than dealt from a
   * budget. What decides whether a crown reads as foliage is whether neighbouring
   * masses **overlap**, and overlap is a statement about the lattice: at 0.5 the
   * masses exactly touch, below it the crown speckles into separated beads, and
   * above it they merge into connected, non-convex volumes with sky between the
   * clumps rather than between every mass.
   *
   * 0.72 puts a mass 1.44 cells across — comfortably interpenetrating with its
   * six face neighbours and reaching about halfway to its diagonal ones, which is
   * the regime where the union has both a ragged outline and holes in it. Toward
   * 1.0 the crown closes into the single lump this species exists not to be; the
   * clumping is then coming from `OAK_CROWN_CELL_SHARE` in `oak-branching.ts`,
   * and that is the knob to move, not this one.
   *
   * The crown's total volume is now an *output* — cells times this cubed — rather
   * than an input. That costs nothing in leaf-invariance, because the cell count
   * and the cell pitch are both leaf-independent.
   */
  readonly foliageBulk: number;
  /**
   * A foliage mass's half-thickness as a share of its plan radius.
   *
   * Near round, and for the reason `CANOPY_PAD_FLATTEN` gives one species over:
   * a spray of oak leaves is a roughly equant clump, and deriving its thickness
   * from the *crown's* would make every mass a disc — and a set of discs at one
   * height is one disc.
   */
  readonly foliageFlatten: number;
  /**
   * How far out along its limb a foliage mass sits, in its own radii.
   *
   * Above zero, so the mass hangs off the end of the limb rather than swallowing
   * it. This is the control that keeps the skeleton visible inside the crown,
   * which is the reference's explicit direction and the one thing a canopy of
   * blobs on a stick gets wrong.
   */
  readonly foliageReach: number;
  /** How far a mass sags below its attachment, in its own radii. Oak foliage hangs. */
  readonly foliageDroop: number;
  /**
   * Spread of foliage mass sizes about the size their share asks for.
   *
   * Zero is a paving pattern. This is what turns an even scatter of equal balls
   * into a few large clumps with smaller ones packed against them, and it is
   * volume-preserving: the sizes are dealt symmetrically about the solved
   * radius, so the crown's bulk does not move with it.
   */
  readonly foliageSwell: number;
  /**
   * The crown automaton's cell pitch, as a share of the crown's longest axis.
   *
   * The single knob that decides how many foliage masses a crown has, and — with
   * `foliageBulk`, which sizes a mass in these cells — the one that decides
   * whether the canopy reads as clumps or as a lump. Smaller is more masses and
   * more records; the ceiling that binds is the 64-candidate limit *per brick*,
   * so measure that rather than the count when moving it far.
   */
  readonly crownCell: number;
  /** How far foliage may grow from the skeleton, in automaton cells. Bounds the crown's reach. */
  readonly crownReach: number;
  /** Automaton steps. Two to four; above four the clumps merge. */
  readonly crownStepCount: number;
  /** Live neighbours of 26 that bring a dead cell to life. Lower fills the crown. */
  readonly crownBirthCount: number;
  /**
   * Live neighbours of 26 above which a live cell is buried and dies.
   *
   * The porosity knob, and the one that puts sky in the canopy — with it off the
   * automaton fills its envelope solid. Lower is more open. See
   * `OAK_CROWN_CROWD_NEIGHBOURS`.
   */
  readonly crownCrowdCount: number;
  /**
   * Clearance kept around the bole and the structural limbs, in automaton cells.
   *
   * What keeps the skeleton visible through the crown — the reference article's
   * explicit direction, and the thing `foliageReach` used to do before the crown
   * was grown rather than placed. Zero lets foliage close over the whole tree.
   */
  readonly crownBare: number;
  /**
   * Lobe pitch on a mass's outline, as a share of its plan radius.
   *
   * The knob that decides whether a crown reads as foliage or as a bag of
   * balls, and it acts on the *silhouette* rather than on the relief. Every
   * lattice in `oak-canopy-field.ts` is inside an envelope the tape clips to;
   * this is what stops that envelope being an ellipsoid. Zero turns the lobing
   * off and restores the ellipsoid, which is worth having on a slider precisely
   * because it is the A/B.
   *
   * See `OakLobeShape` for the derivation and for the hard relation between this
   * and `foliageLobeDepth`: the depth must stay under half of this or the mass
   * is removed in its entirety.
   */
  readonly foliageLobe: number;
  /** How deep a lobe bites, as a share of the mass's plan radius. Must be under half `foliageLobe`. */
  readonly foliageLobeDepth: number;
  /** Palette naming the bark ramp. */
  readonly barkPalette: string;
  /** Palette naming the foliage ramp. */
  readonly canopyPalette: string;
  /**
   * Linear albedo of a limb on `barkPalette`.
   *
   * The hero set is one fired porcelain surface from the toes to the crown — the
   * terrain shell declares `materialModel: "porcelain"` and every prop reads it
   * — so this specimen is a **porcelain oak**: realistic in form, ceramic in
   * surface. What separates its parts is form under a raised key rather than
   * value, which is why the two values below sit close together. A naturalistic
   * oak is these two palettes and these two values, and nothing else.
   */
  readonly barkValue: number;
  /** Linear albedo of a sunlit foliage mass. */
  readonly canopyValue: number;
}

/**
 * ## Why this species has no leaf budget
 *
 * Every other generator in `lib/voxel-scenery/` carries a `maximumLeaves` and
 * throws past it, and this one deliberately does not. The budget was here and it
 * was removed, because in the one loop this species is actually used in it fires
 * on the way to the answer rather than at it: `/shape-lab` re-expands the form on
 * every slider frame, so raising a branch count throws mid-drag and the specimen
 * vanishes from the canvas instead of getting bigger. A guard that makes
 * exploration impossible is not protecting the exploration.
 *
 * Nothing is left unguarded by removing it. The count is still *reported* on
 * `OakPlan.leafCount`, and the ceiling that genuinely matters is enforced
 * downstream and scene-wide rather than per object:
 * `SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES` (= `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES`,
 * 16 384) throws in `buildSvoScenePrimitives`, and the whole hero garden publishes
 * around 450 records — so a tree would have to grow by more than an order of
 * magnitude before it is the binding constraint.
 *
 * **The constraint that binds first is the one a per-object count could never
 * have seen anyway.** It is the 64-candidate *per-brick* ceiling, and its
 * overflow is not a silent drop of the surplus: per
 * `webgpu-sparse-scene-proxies.ts`, overfull bricks stall the live generation,
 * the node-mip certification pass reads zero, and cone visibility fails closed
 * over the whole octree domain — a black slab across the container footprint. A
 * crown of overlapping foliage masses is exactly the geometry that provokes it,
 * and whether it happens depends on how the masses *cluster*, not on how many
 * there are. Two hundred records spread over the crown are fine; sixty stacked
 * in one brick are not. So the check is a busiest-brick census, and
 * `docs/oak-tree-species-plan.md` records how to run it.
 */

export interface OakSpec extends OakForm {
  /** Key prefix. Every emitted node is published as `${key}/...`. */
  readonly key: string;
  /** Where the bole meets the ground, in world metres on the ground plane. */
  readonly at_m: readonly [number, number];
  /**
   * Ground height at a world point. `terrainHeightAt(scene.terrain, x, z)` for
   * anything standing on a heightfield — the baked surface the renderer draws,
   * not a re-derivation of it.
   */
  readonly groundHeightAt: (x_m: number, z_m: number) => number;
  /** Direction the crown leans toward, in XZ. Need not be normalized. */
  readonly lean: readonly [number, number];
  /**
   * The leaf the specimen will be drawn at, in metres. Absent takes
   * {@link OAK_DEFAULT_LEAF_SIZE_M}.
   *
   * Placement rather than form, exactly as `groundHeightAt` is: the species has
   * an opinion about proportion and no opinion at all about what resolution a
   * scene runs. Handed the scene's own cell size, the branch orders and the
   * foliage lattices both resolve against it and the specimen converges on the
   * tree as the leaf shrinks, with nothing re-authored anywhere.
   */
  readonly leafSize_m?: number;
  /** The only entropy. Any integer. */
  readonly seed: number;
}

/**
 * The leaf a specimen is drawn at when nobody says, in metres.
 *
 * `HERO_GARDEN_CELL_M`. A default rather than a constant of the species: a
 * caller that knows its own cell size should pass it, and everything moves with
 * it. `SceneryGeneratorRequest.detailCellSize_m` is where the scene's comes
 * from.
 */
export const OAK_DEFAULT_LEAF_SIZE_M = 0.00625;

/** One foliage mass, as the plan hands it to a test or a census. */
export interface OakFoliageMass {
  readonly center_m: Vec3;
  readonly radius_m: Vec3;
  /** Its share of the crown's foliage volume. All shares sum to one. */
  readonly share: number;
}

export interface OakPlan {
  readonly spec: OakSpec;
  /** Ready to splice into `SceneryGraph.nodes`. */
  readonly nodes: readonly SceneryNode[];
  readonly leafCount: number;
  /** The skeleton this crown was hung on, for anything that wants to measure it. */
  readonly skeleton: OakSkeleton;
  readonly masses: readonly OakFoliageMass[];
  /** World-space extent of the whole specimen. */
  readonly bounds_m: { readonly min: Vec3; readonly max: Vec3 };
  /**
   * World-space extent of the foliage alone.
   *
   * Separate because every question about this tree's *silhouette* is a question
   * about the crown, and measured off the whole specimen every answer moves when
   * the bole does. It is also the thing the leaf-invariance contract is about:
   * this box should be very nearly the same at every rung.
   */
  readonly crownBounds_m: { readonly min: Vec3; readonly max: Vec3 };
  /** Branch orders that cleared the leaf floor at this rung. */
  readonly publishedOrderCount: number;
  /** Foliage lattice levels the tape carries at this rung. */
  readonly foliageLevelCount: number;
}

/**
 * The hero specimen: a mature spreading oak about 0.80 m tall with a 0.84 m
 * crown, standing on the garden's back-right terrace and leaning out over the
 * pond.
 *
 * 0.80 m rather than the bonsai's 0.40 m, and that is the decision the whole
 * form hangs on. At 0.40 m a faithful oak gets a bole, one branch order and no
 * individually drawable leaves — everything else is under the three-leaf floor,
 * so what you would get is an oak-shaped bonsai. Doubling it buys a full second
 * branch order *and* a 4 mm leaf blade at 5.1 leaves across, which is the
 * difference between an oak and a silhouette of one.
 *
 * It costs a scene change: the container is the voxel domain and it was 0.6 m
 * tall, against a tree that reaches about 1.01 m from this terrace. See
 * `HERO_GARDEN_CONTAINER`.
 */
export const OAK_HERO_SPREADING: OakForm = Object.freeze({
  height_m: 0.80,
  boleRadius_m: 0.023,
  spread_m: 0.5,
  orders: OAK_ORDERS_SPREADING,
  branchDensity: 1.0,
  branchFineness: 0.6,
  branchLength: 1.2,
  // A mass 1.44 automaton cells across, so the crown's cells interpenetrate.
  foliageBulk: 0.48,
  foliageFlatten: 0.28,
  // Small, and much smaller than the tiling wanted. Reach and droop were a
  // *placement* device there — a mass had to be pushed off the end of the limb
  // that owned it. A grown cell is already where it belongs, so these are now
  // per-mass perturbations that keep the lattice from reading as one, and a large
  // reach would just inflate the whole shell outward.
  foliageReach: 0.10,
  foliageDroop: 0.12,
  foliageSwell: 0.30,
  crownCell: OAK_CROWN_CELL_SHARE,
  crownReach: OAK_CROWN_REACH_CELLS,
  crownStepCount: OAK_CROWN_AUTOMATON_STEPS,
  crownBirthCount: OAK_CROWN_BIRTH_NEIGHBOURS,
  crownCrowdCount: OAK_CROWN_CROWD_NEIGHBOURS,
  crownBare: OAK_CROWN_BARE_CELLS,
  foliageLobe: OAK_LOBE_ACROSS_SHARE,
  foliageLobeDepth: OAK_LOBE_DEPTH_SHARE,
  barkPalette: "stone",
  canopyPalette: "clay",
  barkValue: 0.91,
  canopyValue: 0.90,
});

/**
 * Distinct foliage-mass sizes, and distinct leaf arrangements, a crown may ask
 * the renderer's field-program arena for.
 *
 * Their product is the crown's whole arena cost — 24 blocks of the 256 the
 * renderer holds — and it is independent of how many masses the automaton grows.
 * See the call site for why quantising here costs almost nothing visible.
 */
const OAK_TAPE_RADIUS_STEPS = 6;
const OAK_TAPE_SEEDS = 4;

/**
 * The order table with the two distilled knobs folded in.
 *
 * One function rather than two call sites, because `oakPublishedOrderCount` and
 * `planOakSkeleton` must agree about the ladder exactly — a scaling applied to
 * the grower but not to the admission test would publish an order the census
 * says is culled, and the two are read from different modules.
 *
 * Order 0 is passed through untouched: its `branchCount` is the buttress count
 * and its `radiusRatio` is 1.0 by construction, so scaling either would move the
 * bole rather than the branching.
 */
export function oakScaledOrders(form: OakForm): readonly OakBranchOrder[] {
  const density = Math.max(0, form.branchDensity);
  const fineness = Math.max(0, form.branchFineness);
  const reach = Math.max(1e-3, form.branchLength);
  if (density === 1 && fineness === 1 && reach === 1) return form.orders;
  return form.orders.map((order, index) => (index === 0 ? order : {
    ...order,
    branchCount: Math.max(2, Math.round(order.branchCount * density)),
    // Saturated, not scaled past the limit. A child may not be fatter than its
    // parent — `validateOrder` refuses it, and rightly, since a pipe model that
    // gains area at a fork is not a tree. A knob that throws when dragged is
    // useless in a loop whose whole purpose is dragging knobs, so the ceiling is
    // a ceiling: past about 1.39 the inner orders simply stop tapering and the
    // knob keeps meaning something for the outer ones.
    radiusRatio: Math.min(1, order.radiusRatio * fineness),
    // Unclamped above: `validateOrder` asks only that it be positive, and a
    // child longer than its parent is a legal — if unusual — plant.
    lengthFraction: order.lengthFraction * reach,
  }));
}

const V = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const add = (a: Vec3, b: Vec3): Vec3 => V(a.x + b.x, a.y + b.y, a.z + b.z);
const scale = (a: Vec3, k: number): Vec3 => V(a.x * k, a.y * k, a.z * k);

/** Integer avalanche hash in [0, 1). The same tree on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e37_79b9, 0x85eb_ca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2_ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}
const hashSigned = (n: number): number => 2 * hash01(n) - 1;

const clampValue = (value: number): number => Math.min(1, Math.max(0, value));


/**
 * Grow one specimen.
 *
 * Pure: it allocates nothing, touches no builder and publishes no primitive. The
 * caller splices `plan.nodes` into a scenery graph, and a caller that wants to
 * know where the crown reaches can ask before anything is voxelized.
 */
export function planOak(spec: OakSpec): OakPlan {
  const {
    key, at_m: [standX, standZ], groundHeightAt, height_m: height,
    boleRadius_m: boleRadius, spread_m: spread, seed,
  } = spec;
  const orders = oakScaledOrders(spec);
  if (!(height > 0) || !(boleRadius > 0) || !(spread > 0)) {
    throw new RangeError("Oak needs a positive height, bole radius and spread");
  }
  if (orders.length < 1) throw new RangeError("Oak needs at least one branch order");
  if (!(spec.foliageBulk > 0)) throw new RangeError("Oak foliage bulk must be positive");
  if (!(spec.foliageFlatten > 0 && spec.foliageFlatten <= 1)) {
    throw new RangeError("Oak foliage flatten must be in (0, 1]");
  }
  // Checked here as well as in `oakCanopyLobe` so that the failure names the
  // form's own knob rather than a tape the caller never wrote. The relation
  // rather than the values: a lobe deeper than half its pitch removes the whole
  // mass, and a slider can walk into that in one drag.
  if (!(spec.foliageLobe >= 0)) throw new RangeError("Oak foliage lobe pitch must be zero or positive");
  if (!(spec.foliageLobeDepth >= 0)) throw new RangeError("Oak foliage lobe depth must be zero or positive");
  // Only when the lobe is on: `foliageLobe: 0` is the documented way to turn it
  // off for an A/B, and it must not be second-guessed because the depth beside
  // it was left at its tuned value.
  if (spec.foliageLobe > 0 && spec.foliageLobeDepth > 0 && !(spec.foliageLobeDepth < 0.5 * spec.foliageLobe)) {
    throw new RangeError(
      `Oak foliage lobe depth ${spec.foliageLobeDepth.toFixed(3)} must be under half its pitch ` +
        `${spec.foliageLobe.toFixed(3)}, or every point of a foliage mass is carved away and the crown is empty`,
    );
  }
  if (!(spec.foliageSwell >= 0 && spec.foliageSwell < 1)) {
    throw new RangeError("Oak foliage swell must be in [0, 1)");
  }

  const leafSize_m = spec.leafSize_m ?? OAK_DEFAULT_LEAF_SIZE_M;
  if (!(leafSize_m > 0)) throw new RangeError("Oak leaf size must be positive");

  const groundY = groundHeightAt(standX, standZ);
  const root_m = V(standX, groundY, standZ);

  const skeleton = planOakSkeleton({
    root_m,
    height_m: height,
    boleRadius_m: boleRadius,
    spread_m: spread,
    orders,
    leanXZ: spec.lean,
    seed,
    leafSize_m,
    crownCell: spec.crownCell,
    crownReach: spec.crownReach,
    crownStepCount: spec.crownStepCount,
    crownBirthCount: spec.crownBirthCount,
    crownCrowdCount: spec.crownCrowdCount,
    crownBare: spec.crownBare,
  });

  const field = oakCanopyField(leafSize_m);

  const nodes: SceneryNode[] = [];
  let minimum = V(Infinity, Infinity, Infinity);
  let maximum = V(-Infinity, -Infinity, -Infinity);
  let crownMinimum = V(Infinity, Infinity, Infinity);
  let crownMaximum = V(-Infinity, -Infinity, -Infinity);
  const cover = (center: Vec3, radius: Vec3): void => {
    minimum = V(Math.min(minimum.x, center.x - radius.x), Math.min(minimum.y, center.y - radius.y), Math.min(minimum.z, center.z - radius.z));
    maximum = V(Math.max(maximum.x, center.x + radius.x), Math.max(maximum.y, center.y + radius.y), Math.max(maximum.z, center.z + radius.z));
  };
  const coverCrown = (center: Vec3, radius: Vec3): void => {
    cover(center, radius);
    crownMinimum = V(Math.min(crownMinimum.x, center.x - radius.x), Math.min(crownMinimum.y, center.y - radius.y), Math.min(crownMinimum.z, center.z - radius.z));
    crownMaximum = V(Math.max(crownMaximum.x, center.x + radius.x), Math.max(crownMaximum.y, center.y + radius.y), Math.max(crownMaximum.z, center.z + radius.z));
  };

  /**
   * One fired ceramic surface from the roots to the crown, declared outright
   * rather than spelled into the tags.
   *
   * The bonsai's header records why: the surface closure used to be picked by
   * matching a regular expression against the group name followed by the tags,
   * so a key spelled `old-tree` or `stone-oak` would silently restyle the whole
   * object. `surface` on the material is a declaration; a tag is a description.
   */
  const bark = (value: number): SceneryMaterial =>
    ({ palette: spec.barkPalette, value: clampValue(value), surface: "ceramic" });
  const canopy = (value: number): SceneryMaterial =>
    ({ palette: spec.canopyPalette, value: clampValue(value), surface: "ceramic" });

  // ---- The skeleton --------------------------------------------------------
  // One `tapered-sweep` record per run. `sweptTubeNodes` fits the envelope
  // itself and splits a run whose best envelope still carries too much empty
  // volume, so a long primary may come back as two records; that is its call to
  // make and not this module's.
  for (const run of skeleton.runs) {
    const value = spec.barkValue - 0.02 + 0.04 * hash01(seed + 1_009 * run.order + run.stations.length);
    nodes.push(...sweptTubeNodes({
      id: `${key}/${run.key}`,
      group: key,
      tags: ["oak", run.order === 0 ? "bole" : "limb", "porcelain"],
      material: bark(value),
      stations: run.stations.map((station) => ({ at_m: station.at_m, radius_m: station.radius_m })),
      seed: (seed ^ 0x0a_c0_de) >>> 0,
    }));
    for (const station of run.stations) {
      cover(station.at_m, V(station.radius_m, station.radius_m, station.radius_m));
    }
  }

  // ---- The crown -----------------------------------------------------------
  // Masses hung on the attachments the skeleton reported, whose shares sum to
  // one whether or not the run under each one was published. That is what makes
  // the crown the same crown at every rung.
  const masses: OakFoliageMass[] = [];
  for (const [index, attachment] of skeleton.attachments.entries()) {
    const nominal_m = spec.foliageBulk * skeleton.crownCell_m;
    // Dealt symmetrically about the solved radius, so the spread costs nothing
    // in bulk however wide it is opened.
    const swell = 1 + spec.foliageSwell * hashSigned(attachment.seed + 17);
    const plan_m = Math.max(leafSize_m, nominal_m * swell);
    const half_m = plan_m * spec.foliageFlatten;
    const radius_m = V(plan_m, half_m, plan_m * (0.90 + 0.16 * hash01(attachment.seed + 23)));
    const center_m = add(
      add(attachment.at_m, scale(attachment.direction, spec.foliageReach * plan_m)),
      V(
        0.10 * plan_m * hashSigned(attachment.seed + 31),
        -spec.foliageDroop * plan_m,
        0.10 * plan_m * hashSigned(attachment.seed + 37),
      ),
    );
    masses.push({ center_m, radius_m, share: attachment.share });
    const step = Math.max(1e-6, nominal_m * (1 + spec.foliageSwell) / OAK_TAPE_RADIUS_STEPS);
    const quantised_m = Math.max(step, Math.round(plan_m / step) * step);
    const tapeRadius_m = V(quantised_m, quantised_m * spec.foliageFlatten, quantised_m);
    const tapeSeed = ((attachment.seed ^ 0x0f01) >>> 0) % OAK_TAPE_SEEDS;
    nodes.push({
      kind: "field-program",
      id: `${key}/foliage-${index}`,
      group: key,
      tags: ["oak", "foliage", "porcelain"],
      // A tape is authored in metres and must resolve at a unit scale of exactly
      // one; the record is centred on its mass and takes no orientation.
      place: { units: "metres", position: center_m },
      // Quantised, so a thousand masses ask for a handful of *distinct* tapes.
      //
      // The renderer's field-program arena holds 256 blocks and a record
      // addresses one by index; `svo-scene-primitives.ts` shares a block between
      // records whose serialised tape is identical, so what decides whether a
      // crown publishes is not how many masses it has but how many different
      // shapes they are. Rounding the radius to `OAK_TAPE_RADIUS_STEPS` sizes and
      // the seed to `OAK_TAPE_SEEDS` variants caps the crown at their product
      // whatever the automaton grows.
      //
      // The mass keeps its own *record* — its centre, its shade and its envelope
      // are all per-mass — so what is lost is only that two masses of very
      // similar size carry an identical arrangement of leaves inside them, at a
      // scale where the leaves are a few voxels across.
      program: oakCanopyPadProgram(tapeRadius_m, field, tapeSeed, {
        across: spec.foliageLobe,
        depth: spec.foliageLobeDepth,
      }),
      // A shade per mass, off the attachment's own seed. The set is one flat
      // plaster closure with the key raised, so a crown of identical masses
      // loses its separation the moment two meet edge-on; this is the little
      // help the form gets.
      material: canopy(spec.canopyValue - 0.03 + 0.05 * hash01(attachment.seed + 41)),
    });
    coverCrown(center_m, radius_m);
  }

  // No budget check. See the note above `OakForm`: the count is reported rather
  // than enforced, because the ceiling that binds is per *brick* and scene-wide
  // rather than per object, and a throw here fires on the way to an answer in
  // the one loop this species exists to be tuned in.
  const leafCount = nodes.length;

  return {
    spec,
    nodes,
    leafCount,
    skeleton,
    masses,
    bounds_m: { min: minimum, max: maximum },
    crownBounds_m: { min: crownMinimum, max: crownMaximum },
    publishedOrderCount: skeleton.publishedOrderCount,
    foliageLevelCount: field.levels.length,
  };
}

/** The species, as the generator catalog calls it. */
export function oakNodes(spec: OakSpec): SceneryNode[] {
  return [...planOak(spec).nodes];
}

/**
 * How many branch orders and foliage lattices a given leaf will draw, without
 * growing anything.
 *
 * The ladder as a function, so a caller — the shape lab's depth selector, a
 * census, a person reading — can ask what a rung buys before paying for it.
 */
export function oakDetailLadder(form: OakForm, leafSize_m = OAK_DEFAULT_LEAF_SIZE_M): {
  readonly leafSize_m: number;
  readonly publishedOrderCount: number;
  readonly foliageLevelCount: number;
  // The lobe carve is deliberately not reported. It is a property of the *form*
  // now rather than of the rung — its pitch is a share of a mass's own radius,
  // so it clears the band law at every leaf the lab offers and there is no rung
  // at which knowing it tells a caller anything.
} {
  const field = oakCanopyField(leafSize_m);
  return {
    leafSize_m,
    publishedOrderCount: oakPublishedOrderCount({
      root_m: V(0, 0, 0),
      height_m: form.height_m,
      boleRadius_m: form.boleRadius_m,
      spread_m: form.spread_m,
      orders: oakScaledOrders(form),
      leanXZ: [1, 0],
      seed: 1,
      leafSize_m,
    }),
    foliageLevelCount: field.levels.length,
  };
}

export type { OakBranchOrder, OakFoliageAttachment };
