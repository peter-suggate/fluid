import type { Vec3 } from "../model";
import { SVO_CLUSTER_SWEEP_MAXIMUM_POINTS } from "../svo-primitive-abi";
import { V } from "./builder";

/**
 * The oak's growth model: parameters and a seed in, branch runs and foliage
 * attachment frames out.
 *
 * This file is *planning only*. It allocates nothing, touches no builder,
 * publishes no scenery node and imports no sibling of its own species. What it
 * produces is geometry a caller can measure before anything is drawn — which is
 * the same discipline `planProceduralTree` follows, and for the same reason: the
 * crown's extent has to be known before the record that carries it can be sized.
 *
 * ## Why a third tree generator
 *
 * `procedural-tree.ts` grows a niwaki — one tapering trunk, limbs on a spiral,
 * foliage as a dozen flattened pads with sky deliberately between the tiers.
 * `bonsai.ts` grows a fused multi-trunk under one wide plate of a crown. An oak
 * is neither: a single dominant bole low down that *dissolves* into competing
 * limbs higher up, a nearly-equal first split, three branch orders on a
 * phyllotactic spiral, and a billowing crown of many masses with the skeleton
 * still readable through it. See `docs/oak-tree-species-plan.md` §4, which makes
 * the case at length.
 *
 * ## The law this module is shaped by
 *
 * Project-wide, and stated in `bonsai.ts`:
 *
 * > A feature whose period is under about **two leaves** does not render as that
 * > feature; it renders as aliasing. Above about **three leaves** it renders as
 * > geometry.
 *
 * The specimen is authored for a tree about 0.80 m tall in a scene whose
 * production leaf is 0.78125 mm (`SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT = 3`)
 * and whose coarsest rung is 6.25 mm (depth 0). Scaling a real 20 m oak down to
 * 0.80 m gives a 40 mm bole, a 12 mm primary limb, a 4 mm secondary, a 1.2 mm
 * tertiary and a 0.6 mm twig. Against a three-leaf floor that is 2.34 mm at
 * depth 3 and **18.75 mm at depth 0** — so a faithfully-scaled oak loses its
 * primary limbs on the coarsest rung, and a crown hanging off a bole with
 * nothing in between is worse than any amount of coarseness. The species below is
 * therefore authored *pushed coarser than nature*, exactly as the plan sanctions;
 * see {@link OAK_ORDERS_SPREADING} for the table and the arithmetic.
 *
 * ## The two mechanisms that make it refine gracefully
 *
 * **1. Branch-order admission is derived from the leaf.** An order is published
 * only when the diameter it would naturally be drawn at clears
 * {@link OAK_LEGIBLE_FEATURE_LEAVES} leaves. {@link oakPublishedOrderCount} is
 * that clock and it is pure, so a caller can ask before planning. Orders are a
 * chain, so culling one culls everything under it.
 *
 * **2. Culled foliage is absorbed, never dropped — but the crown has a floor of
 * its own that culling may not push past.** The crown is defined by the tips of
 * the **full** skeleton, grown from `orders` alone and never from the leaf. When
 * an order is culled its attachments bucket into an ancestor at their own
 * share-weighted centroid, and `Σ share` stays exactly 1 because bucketing only
 * ever *sums*. What that alone does not preserve is the *extent*: one point plus
 * one radius has three degrees of freedom and a cluster has six, so collapsing a
 * whole primary's subtree to its centre of mass conserves the bulk and throws
 * away the spread. Measured on the hero specimen it pulled the crown's plan reach
 * from 537 mm to 307 mm — a silhouette moving by two fifths between rungs, which
 * is exactly the pop this is all supposed to prevent, and volume-conserving
 * sizing does not buy it back (a mass carrying five shares is only `5^(1/3)` =
 * 1.7x wider).
 *
 * So bucketing stops at {@link OAK_CROWN_MASS_MAXIMUM_SHARE}: a *composition*
 * rule, not a resolution one — a mass wider than that has no room for sky beside
 * it. It is leaf-independent, and that is sound because a foliage mass is not a
 * branch. A branch is a thin tube and its legibility is its diameter, which is
 * why orders drop out; a mass is a blob two orders of magnitude clear of the
 * floor (at depth 0 the floor is 18.75 mm and the hero's masses are ~260 mm
 * across), so nothing about the leaf has an opinion on how many there should be.
 *
 * The consequence, on a specimen whose finest order is a placement device, is
 * that **the crown is bit-identical at every rung and only the skeleton
 * refines** — measured across leaves 6.25, 3.125, 1.5625 and 0.78125 mm on both
 * forms below, the attachment cloud's bounding box drifts 0.00% on every axis and
 * the share-weighted centroid agrees to every printed digit. `share` is a volume
 * fraction so a consumer sizing radius as `share^(1/3)` reproduces the crown, and
 * that is the contract `oak.ts` is expected to honour.
 *
 * The same discipline runs through the fit: `height_m` and `spread_m` are solved
 * on the full skeleton *and* the full crown, never on the published subset, so
 * the object's size is a constant of the spec rather than a function of the rung.
 * A tree whose silhouette changes with the voxel is a tree that pops; only its
 * internal structure may refine.
 *
 * ## Determinism
 *
 * One seed in, byte-identical geometry out, on every call. The only entropy is an
 * integer avalanche hash of the branch path folded with the seed — the same idiom
 * `procedural-tree.ts` uses, and for the same hard reason: the environment
 * catalog's static revision, and with it the whole sparse publication cache, is
 * keyed on the geometry produced here. Nothing in this file calls `Math.random`.
 *
 * Re-seeding must grow a *sibling*, not a different species. Every variance below
 * moves a value **within a slot**; not one of them changes a count. So a re-seed
 * moves angles, lengths and reach, and leaves the branch topology — and therefore
 * the record count and the admission ladder — exactly where it was.
 */

// ---------------------------------------------------------------------------
// The contract
// ---------------------------------------------------------------------------

/** One branch order's parameters. Index 0 is the bole itself. */
export interface OakBranchOrder {
  /**
   * Children spawned per parent run.
   *
   * At index 0 there is no parent, and this is the count of **buttress roots**
   * at the foot instead — see {@link OakBranchOrder} note below and
   * {@link OAK_BUTTRESS_REACH_RADII}.
   */
  readonly branchCount: number;
  /** Angle off the parent axis at the fork. 0 is along it, π/2 across it. */
  readonly downAngle_rad: number;
  readonly downAngleVariance_rad: number;
  /** Phyllotactic advance between siblings. The golden angle unless you mean otherwise. */
  readonly rotate_rad: number;
  readonly rotateVariance_rad: number;
  /** Run length as a share of the parent's. At index 0, a share of `height_m`. */
  readonly lengthFraction: number;
  readonly lengthVariance: number;
  /** Radius at the fork, as a share of the parent's local radius. At index 0, of `boleRadius_m`. */
  readonly radiusRatio: number;
  /** Fraction of radius lost along the run. Must stay under 1: a sweep station needs a positive radius. */
  readonly taper: number;
  /** Total turn along the run, positive **away from vertical**. The bow. */
  readonly curve_rad: number;
  /** Reverse turn over the second half, positive **back toward vertical**. The S. */
  readonly curveBack_rad: number;
  /** Gravitropism: how much of the run's departure from +Y is blended out over its whole length, [0, 1]. */
  readonly upAttraction: number;
  /** Sweep stations along the run, 2..{@link SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}. */
  readonly segmentCount: number;
  /** Where along the parent children begin, [0, 1]. At index 0, the buttress band on the bole. */
  readonly forkStart: number;
  /** And where they stop. */
  readonly forkEnd: number;
}

export interface OakSkeletonSpec {
  /**
   * The crown automaton's rules, all optional and all defaulted.
   *
   * On the spec rather than baked in because these are the crown's *composition*
   * — how many clumps, how far they reach, how porous they are — and composition
   * is what `/shape-lab` exists to settle. See `growCrownAutomaton`. None of them
   * may consult the leaf, which is what keeps the crown leaf-invariant.
   */
  /** Automaton cell pitch, as a share of the seed cloud's longest axis. */
  readonly crownCell?: number;
  /** How far foliage may grow from the skeleton, in cells. */
  readonly crownReach?: number;
  /** Automaton steps. */
  readonly crownStepCount?: number;
  /** Live neighbours of 26 that bring a dead cell to life. */
  readonly crownBirthCount?: number;
  /** Live neighbours of 26 above which a live cell is buried and dies. */
  readonly crownCrowdCount?: number;
  /** Clearance kept around the structural limbs, in cells beyond their radius. */
  readonly crownBare?: number;
  readonly root_m: Vec3;
  /**
   * Root to the top of the whole planned object, in metres — every branch order
   * and every foliage attachment, published or not. Exact by construction; see
   * {@link fitToEnvelope}, which also says why this bounds the crown rather than
   * the highest *run* the one-line version of this field used to promise.
   *
   * Drawn mass *centres* sit inside it, because the crown a leaf publishes is a
   * bucketing of the crown this is fitted to, and the consumer's own mass radius
   * then fills back out toward it. Anything sizing a container reads `bounds_m`.
   */
  readonly height_m: number;
  readonly boleRadius_m: number;
  /** Plan radius of the whole planned object, in metres. Bounds the crown, not just the widest primary; see {@link height_m}. */
  readonly spread_m: number;
  readonly orders: readonly OakBranchOrder[];
  /** Direction the crown leans toward, in XZ. Need not be normalized. */
  readonly leanXZ: readonly [number, number];
  readonly seed: number;
  /** The leaf this specimen will be drawn at. The order-admission clock. */
  readonly leafSize_m: number;
}

export interface OakStation { readonly at_m: Vec3; readonly radius_m: number; }

export interface OakRun {
  /** "bole", "flare-2", "o1-3", "o2-3-1" — unique across the skeleton, and stable under a re-seed. */
  readonly key: string;
  readonly order: number;
  readonly parentKey?: string;
  /** From the fork outward. Radii are non-increasing along the run. */
  readonly stations: readonly OakStation[];
  readonly tip_m: Vec3;
  readonly tipDirection: Vec3;
  /** Distance from the root: the sway lever. */
  readonly lever_m: number;
}

/** Where a foliage mass hangs, whether or not the run under it was published. */
export interface OakFoliageAttachment {
  readonly at_m: Vec3;
  readonly direction: Vec3;
  /**
   * The order of the run this mass hangs on — the bucket level, not the leaf.
   *
   * It may be *deeper* than `publishedOrderCount - 1`, which is the whole point
   * of "whether or not the run under it was published": culling asks the crown to
   * coarsen and the crown's own composition floor refuses past a point. See
   * {@link crownBucketLevel}. A consumer that wants to style the inner crown
   * differently from the outer one reads this.
   */
  readonly order: number;
  /** Of the crown's total foliage, as a **volume** fraction. All shares sum to 1. */
  readonly share: number;
  readonly lever_m: number;
  /** Stable per attachment, and stable under a change of leaf for an unmerged one. */
  readonly seed: number;
}

export interface OakSkeleton {
  readonly spec: OakSkeletonSpec;
  readonly runs: readonly OakRun[];
  readonly attachments: readonly OakFoliageAttachment[];
  /**
   * The pitch of the automaton lattice the crown was grown on.
   *
   * **A consumer sizes a foliage mass from this, not from `share`.** The masses
   * have to overlap or the crown is a scatter of separated beads, and overlap is
   * a statement about the lattice they sit on rather than about how much volume
   * they were asked to carry between them. Sizing by conserved volume — the
   * tiling's rule — makes a mass *shrink* as the crown gains cells, which is
   * exactly backwards: more cells is a finer canopy, not a sparser one.
   *
   * Leaf-independent, like everything else the automaton produces.
   */
  readonly crownCell_m: number;
  readonly bounds_m: { readonly min: Vec3; readonly max: Vec3 };
  readonly maximumLever_m: number;
  /** Orders that cleared the leaf floor. `orders.length` at a fine leaf, fewer at a coarse one. */
  readonly publishedOrderCount: number;
}

// ---------------------------------------------------------------------------
// The law, and the constants derived under it
// ---------------------------------------------------------------------------

/**
 * Leaves a feature must span to render as that feature rather than as aliasing.
 *
 * Three, not two, and the difference is the whole argument `bonsai.ts` records:
 * two is the *edge* of the band, and a floor set at the edge of the band is not a
 * floor. The previous bonsai crown's coarsest scale stood at 1.8 leaves and the
 * frame came back as bright ridges around black pits.
 *
 * Everything leaf-dependent in this file goes through this one number:
 * {@link oakPublishedOrderCount} multiplies it by the leaf to get the admission
 * floor, and half of that same product is the radius floor a published station
 * may not thin below.
 */
export const OAK_LEGIBLE_FEATURE_LEAVES = 3;

/**
 * How hard a low fork is suppressed relative to a high one.
 *
 * This is apical dominance made into arithmetic, and it is the difference
 * between an oak and a shrub. A child forking at the very foot of its parent gets
 * `1 - this` of the nominal length; one forking at the tip gets all of it. At
 * 0.45 the bole's lowest primary is 55% of nominal and its highest is 98%, so
 * near the top the limbs are as long as what is left of the bole above them — and
 * that is precisely the "dominant low, dissolving high" the species is described
 * by. Take it to 0 and every limb is the same length and the tree reads as a
 * bottlebrush; take it past about 0.7 and the lower crown empties out.
 *
 * A starting point to be tuned in the shape lab, not a measured value.
 */
const OAK_APICAL_SUPPRESSION = 0.45;

/**
 * How much a fork's height on its parent swings its down-angle, as a share of
 * the authored angle.
 *
 * The one control that decides the silhouette, and `procedural-tree.ts` credits
 * the same mechanism with the same thing: the low limbs leave near-horizontally
 * and reach furthest, the high ones climb and stay short. Without it every limb
 * converges on the same cone and the crown collapses into one mass.
 *
 * The swing is measured about the fork window's own midpoint, so it *self-
 * neutralises* for a narrow window: the buttress band at index 0 spans 0.07 of
 * the bole and every root there gets essentially the authored angle, which is
 * what a root flare wants. Only a wide window — a primary's 0.34..0.96 — actually
 * spends it.
 */
const OAK_FORK_HEIGHT_SPLAY = 0.55;

/**
 * Lateral deflection added to a run's heading per station, as a fraction of the
 * unit direction.
 *
 * Small on purpose. A limb is a smooth arc plus a wobble, not a random walk: the
 * `tapered-sweep` record fuses its segments with a smooth minimum, so the wobble
 * shows as a limb that wanders rather than as a kink, and past about 0.15 it
 * starts to fight the curve/curveBack profile that is doing the actual shaping.
 */
const OAK_STATION_WOBBLE = 0.075;

/**
 * The angles a fork is allowed to ask for, in radians off the parent axis.
 *
 * A down-angle at or past π folds a child back through the run that carries it,
 * which is a sweep record intersecting its own parent along its whole length —
 * legal geometry, ugly picture, and a per-brick candidate count doubled where it
 * happens. The floor keeps a "child" from being a colinear extension of its
 * parent, which is a second record drawing a line the first already drew.
 */
const OAK_FORK_MINIMUM_DOWN_RAD = 0.06;
const OAK_FORK_MAXIMUM_DOWN_RAD = 2.60;

// ---- The root flare -------------------------------------------------------
//
// An oak's foot is its most characteristic feature and it is *lobed*: distinct
// roots leaving the trunk at intervals with hollows between them. That is why
// the flare here is short splayed runs rather than a widening radius on the
// bole's own lowest stations. A `tapered-sweep` section is a circle, so an axial
// swell is a bell — axisymmetric, and it reads as a plinth or a candle drip, not
// as a tree. Only separate runs can put a *hollow* between two lobes, which is
// the same argument `bonsai-canopy-pads.ts` makes one scale up: a displacement
// cannot open a void between two masses.
//
// The record cost is bounded by `orders[0].branchCount` and it lands on the
// fattest, most reliably-published geometry in the tree — the foot publishes at
// every leaf, because if the bole is under the floor nothing is above it either.
//
// A buttress is *interpolated* rather than *grown*. It has two fixed ends — one
// inside the bole so its start cap never shows, one under the ground plane so its
// end cap never shows — and a walked run cannot be made to arrive at both.

/** Reach from the bole axis, in bole radii. About 2.4 puts the footprint at 2.75x the bole, which is where a mature oak's is. */
const OAK_BUTTRESS_REACH_RADII = 2.4;
/** How far the spring point sits inside the bole, as a share of the bole's local radius. Well under 1 or the cap shows. */
const OAK_BUTTRESS_INSET_SHARE = 0.35;
/** Base radius, as a share of the bole's local radius at the spring point. */
const OAK_BUTTRESS_RADIUS_SHARE = 0.52;
/** Radius lost from spring to foot. A buttress is thickest where it leaves the trunk. */
const OAK_BUTTRESS_TAPER = 0.62;
/** The foot is pushed at least this far under the root, in bole radii, whatever the down-angle asked for. */
const OAK_BUTTRESS_MINIMUM_BURY_RADII = 0.30;
/** Where the outward shoulder sits along the reach. Under 1: the root bulges out, then dives. */
const OAK_BUTTRESS_SHOULDER_SHARE = 0.62;
/**
 * Stations on a buttress.
 *
 * Four, because the shape is one shoulder and one dive and a quadratic through
 * four points resolves both. Clamped against the ABI like any other run — see
 * {@link clampStationCount} — so this constant can never be the thing that makes
 * a record illegal.
 */
const OAK_BUTTRESS_STATIONS = 4;

// ---- Where the foliage hangs ----------------------------------------------

/**
 * Orders back from the finest that carry foliage.
 *
 * Two: the terminal order and the one above it. An oak is not a hollow shell —
 * there is foliage on shoots coming directly off the secondaries, and it is what
 * fills the crown's interior so the skeleton reads *through* a canopy rather than
 * silhouetted against one. That is the article's own direction ("branches remain
 * visible, not hidden by excessive leaves") read from the other side.
 *
 * Raising this to 3 would put mass on the primaries themselves, which on this
 * species means mass in the vase — the one part of the crown that should be open.
 */
const OAK_FOLIAGE_ORDER_DEPTH = 2;

/** Share of the crown's foliage carried by the inner order. The rest goes to the terminal one. */
const OAK_FOLIAGE_INNER_SHARE = 0.18;


/**
 * How far past a run's tip its foliage mass is centred, as a share of the run's
 * length.
 *
 * A mass centred *on* the tip is half wasted inside the branch. A fifth of the
 * run past it puts the mass where the twigs this module does not model would
 * have been — which is the whole point of the terminal order: it is a placement
 * device, not a thing to be drawn.
 */
const OAK_FOLIAGE_TIP_REACH = 0.22;
/** Lateral scatter of a mass off its run's axis, as a share of the run's length. Breaks the lattice. */
const OAK_FOLIAGE_JITTER = 0.10;



// ---------------------------------------------------------------------------
// Vector helpers. Same style as `procedural-tree.ts`; deliberately local.
// ---------------------------------------------------------------------------

const add = (a: Vec3, b: Vec3): Vec3 => V(a.x + b.x, a.y + b.y, a.z + b.z);
const sub = (a: Vec3, b: Vec3): Vec3 => V(a.x - b.x, a.y - b.y, a.z - b.z);
const scale = (a: Vec3, k: number): Vec3 => V(a.x * k, a.y * k, a.z * k);
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 =>
  V(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const length = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
const UP = V(0, 1, 0);

function normalizeOr(a: Vec3, fallback: Vec3): Vec3 {
  const magnitude = length(a);
  return magnitude > 1e-9 ? scale(a, 1 / magnitude) : fallback;
}

/** Any unit vector perpendicular to `u`, chosen off whichever axis `u` leans on least. */
function perpendicular(u: Vec3): Vec3 {
  const away = Math.abs(u.x) < 0.9 ? V(1, 0, 0) : V(0, 1, 0);
  return normalizeOr(cross(u, away), V(0, 0, 1));
}

/** Rodrigues. `axis` must be unit. */
function rotateAbout(v: Vec3, axis: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1 - c)));
}

/**
 * Blend a direction toward vertical.
 *
 * The same helper `procedural-tree.ts` uses, and the same job: gravitropism. A
 * limb leaves the bole heading outward and finishes heading up, and this is what
 * turns it.
 */
function towardVertical(direction: Vec3, amount: number): Vec3 {
  return normalizeOr(add(scale(direction, 1 - amount), scale(UP, amount)), direction);
}

/**
 * The axis a run bends about: horizontal, and perpendicular to where the run is
 * already heading.
 *
 * The sign convention is worth stating because it is exactly where this kind of
 * code goes wrong. A **positive** rotation about `cross(+Y, d)` tilts `d` *away
 * from vertical, in the direction it is already going*. Check it twice: with
 * `d = +Y` and the reference along `+X` the axis is `-Z`, and rotating `+Y` about
 * `-Z` by `+θ` gives `(sin θ, cos θ, 0)` — a lean into `+X`. With `d = +X` the
 * axis is again `-Z`, and the same rotation gives `(cos θ, -sin θ, 0)` — a dive.
 * One convention, both cases: positive `curve_rad` bows the run out and down, and
 * positive `curveBack_rad` lifts it back.
 */
function bendAxisFor(direction: Vec3, reference: Vec3): Vec3 {
  return normalizeOr(cross(UP, direction), normalizeOr(cross(UP, reference), perpendicular(direction)));
}

// ---------------------------------------------------------------------------
// Entropy. An integer avalanche hash of the branch path, and nothing else.
// ---------------------------------------------------------------------------

/** Integer avalanche hash in [0, 1). The same tree on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

/** Hash mapped to [-1, 1). */
const hashSigned = (n: number): number => 2 * hash01(n) - 1;

/**
 * Fold a branch path into one uint32 with the seed.
 *
 * Order-sensitive on purpose: `[1, 0]` and `[0, 1]` are different branches and
 * must not share a hash, or two limbs on opposite sides of the tree bend
 * identically. A fractional seed truncates, which is stated rather than
 * prevented — the field is typed `number` and a caller reaching it through a
 * slider should not get an exception for a rounding error.
 */
function hashPath(seed: number, path: readonly number[]): number {
  let h = Math.trunc(seed) >>> 0;
  for (const index of path) h = Math.imul(h ^ (index + 0x9e37), 0x85ebca6b) >>> 0;
  return (h ^ (h >>> 15)) >>> 0;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

const finite = (n: number): boolean => Number.isFinite(n);

/**
 * Stations a run may carry.
 *
 * A `tapered-sweep` record's control polyline is a fixed-size slot of
 * {@link SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}, and `sweptTubeNodes` deals with an
 * over-long run by *splitting* it — which costs a second record and, worse, puts
 * a hard union in the middle of a limb where the smooth minimum was doing the
 * work. So a run planned here never exceeds it. Authored input that does is a
 * `RangeError` from {@link validateSpec}; this clamp is the belt to that
 * braces, so a *derived* count — {@link OAK_BUTTRESS_STATIONS}, or anything a
 * later revision computes — can never be the thing that makes a record illegal.
 */
function clampStationCount(count: number): number {
  return Math.max(2, Math.min(SVO_CLUSTER_SWEEP_MAXIMUM_POINTS, Math.floor(count)));
}

function validateOrder(order: OakBranchOrder, index: number): void {
  const at = `Oak branch order ${index}`;
  if (!Number.isInteger(order.branchCount) || order.branchCount < (index === 0 ? 0 : 1)) {
    throw new RangeError(
      index === 0
        ? `${at} branch count is the buttress count and must be a non-negative integer`
        : `${at} branch count must be an integer of at least one, or the orders below it are unreachable`,
    );
  }
  if (!Number.isInteger(order.segmentCount)
    || order.segmentCount < 2
    || order.segmentCount > SVO_CLUSTER_SWEEP_MAXIMUM_POINTS) {
    throw new RangeError(
      `${at} segment count must be an integer in 2..${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}`
      + ` — a run is one tapered-sweep record and SVO_CLUSTER_SWEEP_MAXIMUM_POINTS is the polyline slot`,
    );
  }
  if (!finite(order.downAngle_rad) || !finite(order.rotate_rad)) {
    throw new RangeError(`${at} down angle and rotate must be finite`);
  }
  if (!(order.downAngleVariance_rad >= 0) || !(order.rotateVariance_rad >= 0)) {
    throw new RangeError(`${at} angle variances must be non-negative`);
  }
  if (!(order.lengthFraction > 0)) throw new RangeError(`${at} length fraction must be positive`);
  if (!(order.lengthVariance >= 0) || !(order.lengthVariance < 1)) {
    throw new RangeError(`${at} length variance must be in [0, 1) — at 1 a run can be planned with zero length`);
  }
  if (!(order.radiusRatio > 0) || !(order.radiusRatio <= 1)) {
    throw new RangeError(`${at} radius ratio must be in (0, 1] — a child may not be fatter than its parent`);
  }
  if (!(order.taper >= 0) || !(order.taper < 1)) {
    throw new RangeError(`${at} taper must be in [0, 1) — a sweep station needs a positive radius`);
  }
  if (!finite(order.curve_rad) || !finite(order.curveBack_rad)) {
    throw new RangeError(`${at} curve and curve-back must be finite`);
  }
  if (!(order.upAttraction >= 0) || !(order.upAttraction <= 1)) {
    throw new RangeError(`${at} up attraction must be in [0, 1]`);
  }
  if (!(order.forkStart >= 0) || !(order.forkEnd <= 1) || !(order.forkStart <= order.forkEnd)) {
    throw new RangeError(`${at} fork window must satisfy 0 <= forkStart <= forkEnd <= 1`);
  }
}

function validateSpec(spec: OakSkeletonSpec): void {
  if (!finite(spec.root_m.x) || !finite(spec.root_m.y) || !finite(spec.root_m.z)) {
    throw new RangeError("Oak root must be finite");
  }
  if (!(spec.height_m > 0)) throw new RangeError("Oak height must be positive");
  if (!(spec.boleRadius_m > 0)) throw new RangeError("Oak bole radius must be positive");
  if (!(spec.spread_m > 0)) throw new RangeError("Oak spread must be positive");
  if (!(spec.leafSize_m > 0)) throw new RangeError("Oak leaf size must be positive");
  if (!finite(spec.seed)) throw new RangeError("Oak seed must be a finite number");
  if (!finite(spec.leanXZ[0]) || !finite(spec.leanXZ[1])) throw new RangeError("Oak lean must be finite");
  if (spec.orders.length < 1) throw new RangeError("Oak needs at least one branch order — index 0 is the bole");
  for (const [index, order] of spec.orders.entries()) validateOrder(order, index);
}

// ---------------------------------------------------------------------------
// Order admission: the leaf's clock
// ---------------------------------------------------------------------------

/**
 * How many orders this leaf will draw. Pure, so a caller can ask before planning.
 *
 * The rule is one line: **an order publishes when the diameter it would
 * naturally be drawn at clears three leaves.** What makes it a derivation rather
 * than a table is that the diameter is walked down the radius chain the growth
 * actually uses — `radiusRatio` against the parent's *local* radius at the middle
 * of the fork window, tapered — so moving a `radiusRatio` in the shape lab moves
 * the ladder with it and cannot leave a stale number behind.
 *
 * Two details that are decisions, not accidents:
 *
 *  - **It breaks rather than skips.** Orders are a chain: a secondary hangs off a
 *    primary, so culling the primary culls everything under it whatever its own
 *    arithmetic says. There is no such thing as a floating order.
 *  - **The bole always publishes.** A specimen whose bole is itself under the
 *    floor is a specimen too small for its scene, and returning zero would answer
 *    that with an invisible tree rather than with a coarse one. The caller who
 *    wants to know can compare `2 * boleRadius_m` against the floor directly.
 *
 * The nominal diameter is taken at the fork window's midpoint, so individual runs
 * of an admitted order straddle it — the lowest fork on a wide window leaves a
 * fatter parent than the highest does. That spread is what the *radius floor* in
 * {@link planOakSkeleton} exists to catch: admission decides whether an order
 * exists, and the floor guarantees that nothing which does exist is drawn under
 * three leaves.
 */
export function oakPublishedOrderCount(spec: OakSkeletonSpec): number {
  validateSpec(spec);
  const floor_m = OAK_LEGIBLE_FEATURE_LEAVES * spec.leafSize_m;
  let radius_m = spec.boleRadius_m * spec.orders[0].radiusRatio;
  // What the *bole* is drawn at, which is its natural radius: the bole clears
  // the floor at every rung this stage holds.
  let parentDrawn_m = 2 * radius_m;
  let published = 1;
  for (let order = 1; order < spec.orders.length; order += 1) {
    const entry = spec.orders[order];
    const parent = spec.orders[order - 1];
    const forkMid = 0.5 * (entry.forkStart + entry.forkEnd);
    radius_m = entry.radiusRatio * radius_m * (1 - parent.taper * forkMid);
    // An order under the floor is **fattened to it, not dropped** — and the
    // condition for that is the whole of this change.
    //
    // The old rule broke at the first order under three leaves, on the band law:
    // a tube thinner than that renders as aliasing rather than as a tube. True,
    // and it costs far too much. `planOakSkeleton` already floors every admitted
    // run's radius at the same three leaves, so a run that is *published* is
    // drawn legibly whatever its natural radius — which means the honest coarse
    // answer was never "no branch here", it was "a branch here, chunkier than it
    // should be". Dropping it took the hero from three orders to two between
    // depth 2 and depth 0 and left a bare fork where a tree's whole silhouette
    // should be.
    //
    // What stops that argument running away is the parent. Fattening every
    // remaining order to the floor makes a child exactly as thick as its parent
    // and the hierarchy dissolves into a bundle of equal tubes — a bush, not a
    // tree. So an order publishes while the diameter it would be **drawn** at
    // stays strictly under its parent's drawn diameter, which is self-limiting
    // and needs no count: the ladder stops itself exactly where the taper stops
    // being visible, one order later at every rung than the bare gate did.
    const drawn_m = Math.max(2 * radius_m, floor_m);
    if (!(drawn_m < parentDrawn_m)) break;
    parentDrawn_m = drawn_m;
    published += 1;
  }
  return published;
}

// ---------------------------------------------------------------------------
// Growth
// ---------------------------------------------------------------------------

/** A run as grown, in nominal space, with natural (unfloored) radii. */
interface GrownRun {
  readonly key: string;
  readonly order: number;
  readonly parentKey?: string;
  /** Child-index path from the bole. `[]` is the bole, `[3, 1]` is `o2-3-1`. */
  readonly path: readonly number[];
  /** Bole first, then each order in turn, ending with this run itself. */
  readonly ancestry: GrownRun[];
  readonly hash: number;
  readonly points: Vec3[];
  readonly radii: number[];
  readonly length_m: number;
}

/** Position, tangent and natural radius at a parameter along a run. */
function sampleRun(run: GrownRun, t: number): { at: Vec3; tangent: Vec3; radius_m: number } {
  const spans = run.points.length - 1;
  const u = Math.min(Math.max(t, 0), 1) * spans;
  const index = Math.min(spans - 1, Math.floor(u));
  const frac = u - index;
  const from = run.points[index];
  const to = run.points[index + 1];
  return {
    at: add(from, scale(sub(to, from), frac)),
    // The stations of a grown run are equally spaced by construction, so the
    // index parameter is the arc-length parameter and no re-parameterisation is
    // needed. Buttresses break that — and they are never forked from.
    tangent: normalizeOr(sub(to, from), UP),
    radius_m: run.radii[index] + (run.radii[index + 1] - run.radii[index]) * frac,
  };
}

/**
 * Walk one run out from its fork.
 *
 * The turn profile is the whole shape of the thing. Over the run's parameter the
 * per-segment turn is
 *
 *     δ(u) = (curve · 2(1 − u) − curveBack · 2u) / segments
 *
 * so the two weights each integrate to one over the run and the total net turn
 * is `curve − curveBack` — but the bow is spent early and the recovery late,
 * which is the S an oak limb actually has. A primary on
 * {@link OAK_ORDERS_SPREADING} leaves the bole climbing, bows over to near
 * horizontal by mid-run, and comes back up: the vase.
 *
 * Gravitropism is a separate, always-upward pull applied after the turn. Its
 * per-station amount is `1 − (1 − upAttraction)^(1/segments)`, so the whole-run
 * effect stays near-constant as `segmentCount` moves: raising the station count
 * re-samples the same curve instead of producing a different one. That matters
 * because `segmentCount` is a *cost* parameter — it is how many of the eight
 * polyline slots this run spends — and a cost parameter that changes the shape is
 * a cost parameter nobody can turn.
 */
function growRun(
  key: string,
  order: number,
  parent: GrownRun | undefined,
  path: readonly number[],
  hash: number,
  from: Vec3,
  direction0: Vec3,
  bendReference: Vec3,
  length_m: number,
  baseRadius_m: number,
  entry: OakBranchOrder,
): GrownRun {
  const stations = clampStationCount(entry.segmentCount);
  const spans = stations - 1;
  const step_m = length_m / spans;
  const bendAxis = bendAxisFor(direction0, bendReference);
  const perStationUp = entry.upAttraction >= 1 ? 1 : 1 - (1 - entry.upAttraction) ** (1 / spans);

  const points: Vec3[] = [from];
  const radii: number[] = [baseRadius_m];
  let at = from;
  let direction = direction0;
  for (let span = 0; span < spans; span += 1) {
    const u = (span + 0.5) / spans;
    const turn = (entry.curve_rad * 2 * (1 - u) - entry.curveBack_rad * 2 * u) / spans;
    direction = rotateAbout(direction, bendAxis, turn);
    direction = towardVertical(direction, perStationUp);
    const lateral = perpendicular(direction);
    const binormal = cross(direction, lateral);
    direction = normalizeOr(add(direction, add(
      scale(lateral, OAK_STATION_WOBBLE * hashSigned(hash + 0x51 + 7 * span)),
      scale(binormal, OAK_STATION_WOBBLE * hashSigned(hash + 0x52 + 7 * span)),
    )), direction);
    at = add(at, scale(direction, step_m));
    points.push(at);
    // Linear taper. Non-increasing for any `taper` in [0, 1), which validation
    // guarantees, so requirement "radii never rise along a run" holds by
    // construction rather than by a sort.
    radii.push(baseRadius_m * (1 - entry.taper * ((span + 1) / spans)));
  }
  // The ancestry ends with the run itself, which is what lets absorption pick an
  // anchor with one index and no special case for "not merged".
  const ancestry: GrownRun[] = parent ? [...parent.ancestry] : [];
  const run: GrownRun = { key, order, parentKey: parent?.key, path, ancestry, hash, points, radii, length_m };
  ancestry.push(run);
  return run;
}

/** Everything the growth produces before the leaf has been consulted at all. */
interface GrownTree {
  readonly runs: readonly GrownRun[];
  /** By order, so the emission can walk them coarse-to-fine. */
  readonly byOrder: readonly (readonly GrownRun[])[];
}

function growSkeleton(spec: OakSkeletonSpec): GrownTree {
  const lean = (() => {
    const [x, z] = spec.leanXZ;
    const magnitude = Math.hypot(x, z);
    return magnitude > 0 ? V(x / magnitude, 0, z / magnitude) : V(1, 0, 0);
  })();

  const byOrder: GrownRun[][] = spec.orders.map(() => []);

  // ---- The bole ---------------------------------------------------------
  // Nominal length only: the fit below sets the tree's real size, so what is
  // authored here is the bole's *share* of the tree rather than its metres.
  const boleEntry = spec.orders[0];
  const boleHash = hashPath(spec.seed, []);
  const boleLength_m = spec.height_m * boleEntry.lengthFraction
    * (1 + boleEntry.lengthVariance * hashSigned(boleHash + 0x11));
  const bole = growRun(
    "bole", 0, undefined, [], boleHash,
    spec.root_m, UP, lean, boleLength_m, spec.boleRadius_m * boleEntry.radiusRatio, boleEntry,
  );
  byOrder[0].push(bole);

  // ---- Orders 1..n ------------------------------------------------------
  // Every order is grown, including the ones this leaf will not publish. That is
  // deliberate and it is what makes the crown leaf-invariant: the finest order's
  // tips are where the foliage hangs, and on this specimen that order never
  // publishes at any rung the stage can hold. It exists to place the crown.
  for (let order = 1; order < spec.orders.length; order += 1) {
    const entry = spec.orders[order];
    const forkMid = 0.5 * (entry.forkStart + entry.forkEnd);
    for (const parent of byOrder[order - 1]) {
      // A per-parent phase so siblings of different parents do not all start
      // their spiral at the same azimuth, which would stack the whole tree into
      // one plane at every order.
      const phase = 2 * Math.PI * hash01(parent.hash + 0x21);
      for (let child = 0; child < entry.branchCount; child += 1) {
        const path = [...parent.path, child];
        const hash = hashPath(spec.seed, path);
        const t = entry.branchCount === 1
          ? forkMid
          : entry.forkStart + (entry.forkEnd - entry.forkStart) * (child / (entry.branchCount - 1));
        const fork = sampleRun(parent, t);

        // The fork frame. Azimuth zero points along the lean, projected off the
        // parent's own axis, so a re-seed rotates the spiral without changing
        // which side of the tree the crown hangs on.
        const axis = fork.tangent;
        const reference = normalizeOr(sub(lean, scale(axis, dot(lean, axis))), perpendicular(axis));
        const across = cross(axis, reference);
        const azimuth = phase + entry.rotate_rad * child
          + entry.rotateVariance_rad * hashSigned(hash + 0x31);
        const radial = add(scale(reference, Math.cos(azimuth)), scale(across, Math.sin(azimuth)));

        const down = Math.min(OAK_FORK_MAXIMUM_DOWN_RAD, Math.max(OAK_FORK_MINIMUM_DOWN_RAD,
          entry.downAngle_rad * (1 + OAK_FORK_HEIGHT_SPLAY * (forkMid - t))
          + entry.downAngleVariance_rad * hashSigned(hash + 0x32)));
        const direction0 = normalizeOr(
          add(scale(axis, Math.cos(down)), scale(radial, Math.sin(down))), radial);

        const apical = 1 - OAK_APICAL_SUPPRESSION * (1 - t);
        const length_m = parent.length_m * entry.lengthFraction * apical
          * (1 + entry.lengthVariance * hashSigned(hash + 0x33));
        const baseRadius_m = entry.radiusRatio * fork.radius_m;

        byOrder[order].push(growRun(
          `o${order}-${path.join("-")}`, order, parent, path,
          hash, fork.at, direction0, radial, length_m, baseRadius_m, entry,
        ));
      }
    }
  }

  return { runs: byOrder.flat(), byOrder };
}

// ---------------------------------------------------------------------------
// The fit: `height_m` and `spread_m` made exact
// ---------------------------------------------------------------------------

/**
 * Scale the grown tree about its root so it lands on the authored height and
 * spread.
 *
 * Two factors, one vertical and one horizontal, solved on the **full** skeleton
 * and the **full** crown — every order, published or not, plus every attachment
 * before any bucketing. That is the load-bearing detail: solving on the published
 * subset would make the fit a function of the leaf, and the whole tree would
 * breathe as the rung changed. Solved on the full object it is a constant of the
 * spec, and the silhouette is leaf-invariant by construction.
 *
 * **Both numbers bound the whole object, crown included.** `height_m` is root to
 * the top of the highest thing the tree plans, and `spread_m` is the plan radius
 * of the widest — which for any real form is a foliage attachment and not a
 * primary limb, since the crown is the outermost thing an oak has. Fitting to the
 * primaries alone, which is what the field's one-line description says, was
 * measured on the hero specimen at a `spread_m` of 0.42 m against a crown 1.19 m
 * wide: a "bound" three times under the thing it was meant to bound, and useless
 * to a caller sizing a container. The widest primary now lands well *inside*
 * `spread_m`, which is the direction a caller can actually use.
 *
 * The consumer's foliage radius still adds on top of both, because this module
 * plans attachment *points* and does not know how wide a mass is. Anything
 * sizing a container reads `bounds_m`, never `height_m`.
 *
 * The buttresses are built *after* the fit, from the fitted bole and
 * `boleRadius_m`, so a root flare keeps its authored proportions and cannot drag
 * the fit around.
 *
 * The honest cost: an anisotropic scale is a shear, and it moves the authored
 * angles. A limb authored at 45° comes out at 52° under a 1.3x horizontal
 * factor. So the forms below are proportioned to land the factors near 1, and a
 * factor that strays far from it is a sign the *form* wants re-authoring —
 * `lengthFraction` and `downAngle_rad` are the cure, not a bigger `spread_m`.
 */
function fitToEnvelope(tree: GrownTree, seeds: readonly SeedAttachment[], spec: OakSkeletonSpec): void {
  let top_m = 0;
  let reach_m = 0;
  const swallow = (point: Vec3): void => {
    top_m = Math.max(top_m, point.y - spec.root_m.y);
    reach_m = Math.max(reach_m, Math.hypot(point.x - spec.root_m.x, point.z - spec.root_m.z));
  };
  for (const run of tree.runs) for (const point of run.points) swallow(point);
  for (const seed of seeds) swallow(seed.at_m);

  const vertical = top_m > 1e-9 ? spec.height_m / top_m : 1;
  const horizontal = reach_m > 1e-9 ? spec.spread_m / reach_m : 1;
  const apply = (point: Vec3): Vec3 => V(
    spec.root_m.x + (point.x - spec.root_m.x) * horizontal,
    spec.root_m.y + (point.y - spec.root_m.y) * vertical,
    spec.root_m.z + (point.z - spec.root_m.z) * horizontal,
  );
  for (const run of tree.runs) {
    for (const [index, point] of run.points.entries()) run.points[index] = apply(point);
  }
  for (const seed of seeds) seed.at_m = apply(seed.at_m);
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

function runFrom(grown: GrownRun, radii: readonly number[], root_m: Vec3): OakRun {
  const points = grown.points;
  const tip_m = points[points.length - 1];
  return {
    key: grown.key,
    order: grown.order,
    parentKey: grown.parentKey,
    stations: points.map((at_m, index) => ({ at_m, radius_m: radii[index] })),
    tip_m,
    tipDirection: normalizeOr(sub(tip_m, points[points.length - 2]), UP),
    lever_m: length(sub(tip_m, root_m)),
  };
}

/**
 * The buttress runs, built in final coordinates off the fitted bole.
 *
 * A quadratic through spring → shoulder → foot: the spring point sits inside the
 * bole, the shoulder pushes the root out at trunk height, and the foot is under
 * the ground plane. `orders[0].downAngle_rad` sets how steeply the chord dives —
 * `drop = run · tan(downAngle − π/2)`, which is the same "angle off the parent
 * axis" every other order means by it, read against a vertical bole. Whatever it
 * asks for, the foot is pushed at least
 * {@link OAK_BUTTRESS_MINIMUM_BURY_RADII} under the root: a sweep is capped at
 * both ends, and a cap above the ground is a disc of bark hanging in the air.
 */
function growButtresses(bole: GrownRun, spec: OakSkeletonSpec, lean: Vec3): OakRun[] {
  const entry = spec.orders[0];
  if (entry.branchCount < 1) return [];
  const stations = clampStationCount(OAK_BUTTRESS_STATIONS);
  const across = V(-lean.z, 0, lean.x);
  const forkMid = 0.5 * (entry.forkStart + entry.forkEnd);
  const runs: OakRun[] = [];
  for (let index = 0; index < entry.branchCount; index += 1) {
    const hash = hashPath(spec.seed ^ 0x0b0b, [index]);
    const t = entry.branchCount === 1
      ? forkMid
      : entry.forkStart + (entry.forkEnd - entry.forkStart) * (index / Math.max(1, entry.branchCount - 1));
    const spring = sampleRun(bole, t);
    const azimuth = entry.rotate_rad * index + entry.rotateVariance_rad * hashSigned(hash + 0x41);
    const radial = normalizeOr(
      add(scale(lean, Math.cos(azimuth)), scale(across, Math.sin(azimuth))), lean);

    const reach_m = OAK_BUTTRESS_REACH_RADII * spec.boleRadius_m
      * (1 + entry.lengthVariance * hashSigned(hash + 0x42));
    const inset_m = OAK_BUTTRESS_INSET_SHARE * spring.radius_m;
    const from = add(spring.at, scale(radial, inset_m));
    const run_m = Math.max(1e-6, reach_m - inset_m);
    // Held past horizontal: a root that leaves the trunk heading up is not a
    // root. Past that the angle is the chord's own dive, and the floor under the
    // foot is what guarantees the end cap is buried whatever it asks for.
    const down = Math.min(OAK_FORK_MAXIMUM_DOWN_RAD, Math.max(0.5 * Math.PI + 1e-3,
      entry.downAngle_rad + entry.downAngleVariance_rad * hashSigned(hash + 0x43)));
    const foot = V(
      from.x + radial.x * run_m,
      Math.min(
        from.y - run_m * Math.tan(down - 0.5 * Math.PI),
        spec.root_m.y - OAK_BUTTRESS_MINIMUM_BURY_RADII * spec.boleRadius_m,
      ),
      from.z + radial.z * run_m,
    );
    // The shoulder: out at spring height, so the root bulges before it dives.
    const shoulder = V(
      from.x + radial.x * run_m * OAK_BUTTRESS_SHOULDER_SHARE,
      from.y,
      from.z + radial.z * run_m * OAK_BUTTRESS_SHOULDER_SHARE,
    );
    const baseRadius_m = OAK_BUTTRESS_RADIUS_SHARE * spring.radius_m;

    const points: Vec3[] = [];
    const radii: number[] = [];
    for (let station = 0; station < stations; station += 1) {
      const u = station / (stations - 1);
      const inverse = 1 - u;
      points.push(add(add(
        scale(from, inverse * inverse),
        scale(shoulder, 2 * inverse * u)),
        scale(foot, u * u)));
      radii.push(baseRadius_m * (1 - OAK_BUTTRESS_TAPER * u));
    }
    const tip_m = points[points.length - 1];
    runs.push({
      key: `flare-${index}`,
      order: 0,
      parentKey: bole.key,
      stations: points.map((at_m, station) => ({ at_m, radius_m: radii[station] })),
      tip_m,
      tipDirection: normalizeOr(sub(tip_m, points[points.length - 2]), radial),
      lever_m: length(sub(tip_m, spec.root_m)),
    });
  }
  return runs;
}

/** One foliage mass as born, before absorption has been applied. */
interface SeedAttachment {
  readonly run: GrownRun;
  readonly birthOrder: number;
  /** Nominal space until {@link fitToEnvelope} has run over it. */
  at_m: Vec3;
  readonly direction: Vec3;
  /** Already normalized against the whole crown: these sum to exactly 1. */
  readonly share: number;
}

/**
 * The crown, before absorption: one mass at the tip of every run of the outer
 * {@link OAK_FOLIAGE_ORDER_DEPTH} orders of the **full** skeleton.
 *
 * This set is the crown, and it is computed from `orders` alone — the leaf has
 * not been consulted, and cannot be. Its positions and its total bulk are
 * therefore leaf-invariant by construction, which is the whole point: absorption
 * downstream may only ever *bucket* this cloud, never move it or add to it.
 *
 * Within an order the weight is the **pipe model** — foliage is proportional to
 * the cross-section feeding it, `weight = tipRadius²` — on the *natural* radii,
 * never the floored ones, so the crown cannot redistribute itself when the leaf
 * changes. Between the two orders the split is the flat
 * {@link OAK_FOLIAGE_INNER_SHARE}, because the pipe model cannot arbitrate across
 * a fork without counting the same cross-section twice.
 */
function seedCrown(tree: GrownTree): SeedAttachment[] {
  const orderCount = tree.byOrder.length;
  const firstFoliageOrder = Math.max(0, orderCount - OAK_FOLIAGE_ORDER_DEPTH);
  const seeds: Omit<SeedAttachment, "share">[] = [];
  const weights: number[] = [];
  const weightByOrder = new Map<number, number>();

  for (let order = firstFoliageOrder; order < orderCount; order += 1) {
    for (const run of tree.byOrder[order]) {
      const points = run.points;
      const tip = points[points.length - 1];
      const direction = normalizeOr(sub(tip, points[points.length - 2]), UP);
      const lateral = perpendicular(direction);
      const binormal = cross(direction, lateral);
      seeds.push({
        run,
        birthOrder: order,
        at_m: add(add(tip, scale(direction, OAK_FOLIAGE_TIP_REACH * run.length_m)), add(
          scale(lateral, OAK_FOLIAGE_JITTER * run.length_m * hashSigned(run.hash + 0x61)),
          scale(binormal, OAK_FOLIAGE_JITTER * run.length_m * hashSigned(run.hash + 0x62)),
        )),
        direction,
      });
      const weight = run.radii[run.radii.length - 1] ** 2;
      weights.push(weight);
      weightByOrder.set(order, (weightByOrder.get(order) ?? 0) + weight);
    }
  }
  // Each birth order is normalized to its configured share of the crown here,
  // before anything is bucketed, which is what makes the total exactly 1 whatever
  // the bucketing does to it afterwards. A species with only one foliage order
  // takes all of it.
  const single = weightByOrder.size < 2;
  return seeds.map((seed, index) => {
    const total = weightByOrder.get(seed.birthOrder) ?? 1;
    const orderShare = single ? 1
      : seed.birthOrder === orderCount - 1 ? 1 - OAK_FOLIAGE_INNER_SHARE : OAK_FOLIAGE_INNER_SHARE;
    return { ...seed, share: total > 0 ? orderShare * (weights[index] / total) : 0 };
  });
}



// ---------------------------------------------------------------------------
// The crown, as a cellular automaton
// ---------------------------------------------------------------------------

/**
 * Grow the crown by simulation rather than by placing it.
 *
 * ## Why the placed crown had to go
 *
 * The first crown was thirty masses on a share-weighted tiling, each drawn as a
 * field-program pad clipped to its own ellipsoid, and every render of it came
 * back as a bag of balls. `oak-canopy-field.ts` records the half of that
 * diagnosis which lives in the tape — the clip *is* the silhouette, and lobing it
 * helps but cannot make one convex mass ragged. This is the other half, and it
 * is the bigger one: **thirty masses at 128 mm across a 700 mm crown is five
 * masses wide.** Foliage does not read as foliage because its clumps are
 * detailed; it reads because many small clumps interpenetrate into irregular,
 * connected, non-convex volumes with sky cut through them. No amount of relief
 * inside a mass and no carve on its envelope reaches that, because it is a fact
 * about how the masses are *arranged*.
 *
 * So the arrangement is grown. Seed a cluster at the end of every terminal
 * branch, then run a three-dimensional automaton for a few steps:
 *
 *  - a cell **lives** if it touches a branch, or if it already has neighbours
 *    enough to be part of a clump ({@link OAK_CROWN_BIRTH_NEIGHBOURS});
 *  - a cell **dies** with fewer than {@link OAK_CROWN_SUPPORT_NEIGHBOURS}, which
 *    is what stops the crown speckling into cells hanging in mid-air;
 *  - a cell **dies when it is enclosed**, which hollows the interior out so the
 *    crown is a shell around an open vase rather than a solid lump.
 *
 * The result self-clusters around the limbs, which is exactly the thing the
 * tiling could not do: it wraps.
 *
 * ## The hollowing rule is not the one a 26-neighbourhood would give
 *
 * The rule as usually stated is "dies above six neighbours". That is written for
 * a **six**-neighbourhood, where seven is impossible and the rule is really
 * "dies when enclosed". Transcribed literally onto the 26-neighbourhood the
 * growth rule needs — three neighbours out of six is a very different test from
 * three out of twenty-six — it eats the shell as well as the interior: a cell on
 * a smooth shell has nine to seventeen of its twenty-six neighbours alive, so a
 * threshold of six kills the crown outright rather than hollowing it.
 *
 * So the two tests use the two neighbourhoods, each where it means something:
 * **birth and support count all twenty-six** (clumping is a question about the
 * whole neighbourhood), and **enclosure tests the six faces** (a cell whose six
 * faces are all covered cannot be seen from any axis and is the memory the rule
 * exists to not spend). That is the intent of the original rule, at the only
 * threshold that carries it.
 *
 * ## Leaf-invariance is now structural rather than defended
 *
 * The tiling this replaces needed a floor on the mass count
 * (`OAK_CROWN_MASS_MAXIMUM_SHARE`, since removed) to stop a culled branch order
 * collapsing the crown onto six primaries — a composition rule bolted on to
 * defend a silhouette. The automaton needs nothing of the kind. Its lattice, its
 * seeds and its rules are all in absolute metres off the **full** skeleton, none
 * of them consults `leafSize_m`, and a deterministic rule on a fixed lattice from
 * fixed seeds returns the same set every time. The crown is therefore *identical*
 * at every rung by construction, which is a stronger statement than the one the
 * floor bought and it costs nothing to make.
 *
 * The contact repair went with it. A mass now lives where it does because it
 * grew from a branch or from a neighbour that did, so an orphan is not a case the
 * rules can produce, and the repair — which had to guess the consumer's radius
 * formula to decide what "touching" meant — had nothing left to fix.
 */
interface CrownAutomaton {
  readonly cell_m: number;
  readonly attachments: readonly OakFoliageAttachment[];
}

/**
 * The automaton's cell pitch, as a share of the seed cloud's longest axis.
 *
 * The one number that decides how many masses a crown has, and it is a share
 * rather than a length so that a small specimen and a large one get crowns of the
 * same *composition* — the whole point of a species being a form rather than a
 * model. At 0.085 the hero's ~700 mm crown gets an 8-cell box and 60 mm cells,
 * which after growth and hollowing lands in the low hundreds of masses against
 * the thirty the tiling gave.
 *
 * It trades against nothing in the tape and against two things outside it. Down
 * is more masses, each smaller: better foliage until the masses stop overlapping
 * and the crown speckles, which {@link OAK_CROWN_MASS_CELLS} is what actually
 * governs. Down is also more **records**, and the ceiling that binds there is
 * the 64-candidate limit *per brick* rather than any per-object budget — see
 * `docs/oak-tree-species-plan.md`. Smaller masses touch proportionally fewer
 * bricks, so the count per brick is far flatter in this number than the record
 * count is, but it is not free and it is the thing to measure after moving this.
 */
export const OAK_CROWN_CELL_SHARE = 0.075;

/** Jitter off the lattice, as a share of a cell. */
const OAK_CROWN_CELL_JITTER = 0.22;

/** Cells of the automaton's lattice a seed cluster covers, as a radius. */
const OAK_CROWN_SEED_RADIUS_CELLS = 1.1;

/**
 * How many steps the automaton runs.
 *
 * Three, from the article's "2 to 4". Each step can advance the crown one cell
 * out from its seeds, so this and {@link OAK_CROWN_CELL_SHARE} together set how
 * far foliage reaches past the branch tips — three 60 mm cells is 180 mm, which
 * on a 800 mm tree is a canopy that closes over the gaps between limbs without
 * swallowing the vase. Above four the clumps merge into the single lump the
 * whole design is trying not to be.
 */
export const OAK_CROWN_AUTOMATON_STEPS = 3;

/** Live neighbours of twenty-six that bring a dead cell to life. */
export const OAK_CROWN_BIRTH_NEIGHBOURS = 3;

/** Live neighbours of twenty-six below which a live cell is a floater and dies. */
const OAK_CROWN_SUPPORT_NEIGHBOURS = 2;

/**
 * Live neighbours of twenty-six above which a live cell is buried and dies.
 *
 * **This is the rule that puts sky in the crown, and it is worth recording that
 * it was nearly dropped.** Read as a memory optimisation — "hollow out the
 * inside so you do not waste voxels on hidden geometry" — it looks redundant
 * here, because a mass buried inside a crown costs a record whether or not the
 * eye can see it, and the obvious transcription onto a twenty-six-neighbourhood
 * (where a smooth shell cell has nine to seventeen live neighbours) reads as a
 * rule that would eat the crown rather than hollow it.
 *
 * Both readings are wrong, and the render that settled it is unambiguous. With
 * no crowding rule the automaton fills its envelope solid in three steps: an
 * outline that is genuinely irregular, wrapped around a canopy with no sky
 * through it and no branch visible anywhere inside it — the "excessive leaves"
 * the reference article names as the failure. The crowding rule is not a memory
 * optimisation at all. It is what stops a growth rule from converging on a lump.
 *
 * What it actually does is put the automaton in an equilibrium rather than
 * letting it run to a fixed point: birth needs {@link OAK_CROWN_BIRTH_NEIGHBOURS}
 * and survival is capped here, so a region that fills in thins itself back out
 * and what persists is filigree at the density where those two meet. The crown
 * is porous by dynamics, not by a fill fraction anybody chose.
 */
export const OAK_CROWN_CROWD_NEIGHBOURS = 6;

/**
 * How far from the skeleton a cell may be born, in cells.
 *
 * **The automaton connects and hollows the crown; it does not size it.** Left
 * unbounded it does size it, and measurably: each step advances the front one
 * cell in every direction, so three steps at a 60 mm cell added 180 mm to every
 * side and took the hero's crown from 713 mm across to 1215 mm — on a tree 800 mm
 * tall. Worse than the number is the coupling. The crown's silhouette is set by
 * where the branch tips are, which `fitToEnvelope` has already fitted to
 * `height_m` and `spread_m`; letting the step count move it as well means two
 * unrelated knobs both change the tree's size and neither can be tuned.
 *
 * **A bound is not the same as an envelope, and the first version of this got
 * that wrong.** Clipping growth to the ellipsoid through the seed cloud does
 * hold the size — and it hands the crown a *sphere* for an outline, which is the
 * bag-of-balls failure this whole rewrite exists to fix, moved up one scale and
 * made worse: one ball instead of thirty. So the bound is a dilation of the
 * **skeleton itself** rather than a solid fitted around it. Foliage may grow two
 * cells from a branch tip or from the wood of a foliage-order limb and no
 * further, so the crown's outline is the limb distribution's own — lumpy where
 * the limbs cluster, open where they fork — and it cannot be anything else.
 *
 * Two cells rather than three, so the reach genuinely binds: at three it equals
 * {@link OAK_CROWN_AUTOMATON_STEPS} and constrains nothing the rules would not
 * have done anyway.
 */
export const OAK_CROWN_REACH_CELLS = 0;

/**
 * How far a branch reaches to bring cells to life, in cells beyond its own radius.
 *
 * The article's rule is "touches a branch", and on a lattice this coarse a limb
 * 7 mm across touches almost no cell centre — a literal reading seeds nothing off
 * the wood at all and the crown grows only from the tips. A cell's width of
 * tolerance is what makes the rule mean what it says.
 *
 * Only the runs of the foliage orders are offered. A bole that grew leaves would
 * fill the vase, which is the one part of an oak's crown that must stay open.
 */
const OAK_CROWN_BRANCH_REACH_CELLS = 0.975;

/**
 * Clearance kept around the structural limbs, in cells beyond their own radius.
 *
 * **The rule the reference article states as art direction, and the one the
 * automaton had no way to express.** "Branches remain visible, not hidden by
 * excessive leaves" was enforced in the tiling by `foliageReach`, which pushed
 * every mass off the end of the limb that owned it. A grown cell is already
 * where it belongs, so that control went to almost nothing — and with cells born
 * on the wood by {@link OAK_CROWN_BRANCH_REACH_CELLS} and masses 1.44 cells
 * across, the crown closed over the whole skeleton. The bole and every primary
 * were still published and still exactly where they had been; nothing could see
 * them.
 *
 * So foliage may grow off the **terminal** order's wood, where a real oak's
 * leaves are, and is kept clear of everything structural under it. The clearance
 * is a little over one mass radius, which is what it takes for a limb to read as
 * a limb rather than as a seam between two clumps.
 *
 * This is a veto rather than a birth rule: it kills cells that are already alive
 * as well as refusing new ones, because the seed clusters and the branch mask
 * both run before the first step and either can put a cell on a primary.
 */
export const OAK_CROWN_BARE_CELLS = 0.9;

/**
 * A hard ceiling on live cells, and what it is actually protecting.
 *
 * Every live cell is one `field-program` record in the scene, and records land in
 * bricks that hold **64 candidates** before the live generation stalls, the
 * node-mip certification reads zero and cone visibility fails closed — a black
 * slab where the tree was, with nothing in the failure naming the cause. That is
 * a long way downstream of a number in this file, so the number in this file
 * refuses first and says why.
 *
 * Generous rather than tight: it is a runaway guard on a rule that could in
 * principle fill its box, not a budget. A hero crown at the pitch above is an
 * order of magnitude under it.
 */
const OAK_CROWN_MAXIMUM_CELLS = 2048;

function growCrownAutomaton(
  seeds: readonly SeedAttachment[],
  tree: GrownTree,
  spec: OakSkeletonSpec,
): CrownAutomaton {
  const foliageOrder = Math.max(0, tree.byOrder.length - OAK_FOLIAGE_ORDER_DEPTH);
  const cellShare = spec.crownCell ?? OAK_CROWN_CELL_SHARE;
  const reachCells = spec.crownReach ?? OAK_CROWN_REACH_CELLS;
  const stepCount = Math.max(0, Math.round(spec.crownStepCount ?? OAK_CROWN_AUTOMATON_STEPS));
  const birth = Math.max(1, Math.round(spec.crownBirthCount ?? OAK_CROWN_BIRTH_NEIGHBOURS));
  const crowd = Math.max(birth, Math.round(spec.crownCrowdCount ?? OAK_CROWN_CROWD_NEIGHBOURS));
  const bareCells = Math.max(0, spec.crownBare ?? OAK_CROWN_BARE_CELLS);
  if (seeds.length === 0) return { cell_m: 0, attachments: [] };

  // The lattice, off the seed cloud rather than off the whole tree: the crown is
  // what the automaton grows and the bole is not part of it. Padded by the reach
  // the rules can actually achieve, so growth is never clipped by the box.
  let low = V(Infinity, Infinity, Infinity);
  let high = V(-Infinity, -Infinity, -Infinity);
  for (const seed of seeds) {
    low = V(Math.min(low.x, seed.at_m.x), Math.min(low.y, seed.at_m.y), Math.min(low.z, seed.at_m.z));
    high = V(Math.max(high.x, seed.at_m.x), Math.max(high.y, seed.at_m.y), Math.max(high.z, seed.at_m.z));
  }
  const span = sub(high, low);
  const cell_m = Math.max(1e-4, cellShare * Math.max(span.x, span.y, span.z));
  const pad = stepCount + Math.ceil(OAK_CROWN_SEED_RADIUS_CELLS) + 1;
  const origin = V(low.x - pad * cell_m, low.y - pad * cell_m, low.z - pad * cell_m);
  const dims = [
    Math.ceil(span.x / cell_m) + 2 * pad + 1,
    Math.ceil(span.y / cell_m) + 2 * pad + 1,
    Math.ceil(span.z / cell_m) + 2 * pad + 1,
  ] as const;
  const [nx, ny, nz] = dims;
  const at = (x: number, y: number, z: number): number => (z * ny + y) * nx + x;
  const centre = (x: number, y: number, z: number): Vec3 =>
    V(origin.x + (x + 0.5) * cell_m, origin.y + (y + 0.5) * cell_m, origin.z + (z + 0.5) * cell_m);

  let live = new Uint8Array(nx * ny * nz);
  const branch = new Uint8Array(nx * ny * nz);

  const reachable = new Uint8Array(nx * ny * nz);
  const bare = new Uint8Array(nx * ny * nz);

  // The seed clusters, one at the end of every terminal branch.
  const seedRadius_m = OAK_CROWN_SEED_RADIUS_CELLS * cell_m;
  const reach_m = reachCells * cell_m;
  const stamp = (target: Uint8Array, point: Vec3, radius_m: number): void => {
    const reach = Math.ceil(radius_m / cell_m);
    const gx = Math.floor((point.x - origin.x) / cell_m);
    const gy = Math.floor((point.y - origin.y) / cell_m);
    const gz = Math.floor((point.z - origin.z) / cell_m);
    for (let z = Math.max(0, gz - reach); z <= Math.min(nz - 1, gz + reach); z += 1) {
      for (let y = Math.max(0, gy - reach); y <= Math.min(ny - 1, gy + reach); y += 1) {
        for (let x = Math.max(0, gx - reach); x <= Math.min(nx - 1, gx + reach); x += 1) {
          if (length(sub(centre(x, y, z), point)) <= radius_m) target[at(x, y, z)] = 1;
        }
      }
    }
  };
  for (const seed of seeds) {
    stamp(live, seed.at_m, seedRadius_m);
    stamp(reachable, seed.at_m, reach_m);
  }

  // The wood the rule may grow off — the foliage orders only, so the vase stays
  // open. Stamped once as a mask rather than re-tested per step, which is what
  // keeps the automaton linear in its steps.
  // The structural limbs, which foliage is kept off entirely — every order but
  // the terminal one, plus the buttressed foot, which `growButtresses` publishes
  // outside `byOrder` and which is the most visible wood on the specimen.
  const bare_m = bareCells * cell_m;
  for (let order = 0; order + 1 < tree.byOrder.length; order += 1) {
    for (const run of tree.byOrder[order]) {
      for (let index = 0; index + 1 < run.points.length; index += 1) {
        const from = run.points[index];
        const to = run.points[index + 1];
        const radius_m = Math.max(run.radii[index], run.radii[index + 1]) + bare_m;
        const steps = Math.max(1, Math.ceil(length(sub(to, from)) / (0.5 * cell_m)));
        for (let step = 0; step <= steps; step += 1) {
          stamp(bare, add(from, scale(sub(to, from), step / steps)), radius_m);
        }
      }
    }
  }

  for (let order = foliageOrder; order < tree.byOrder.length; order += 1) {
    for (const run of tree.byOrder[order]) {
      for (let index = 0; index + 1 < run.points.length; index += 1) {
        const from = run.points[index];
        const to = run.points[index + 1];
        const radius_m = Math.max(run.radii[index], run.radii[index + 1])
          + OAK_CROWN_BRANCH_REACH_CELLS * cell_m;
        // Walked along the segment rather than solved against it: a segment is
        // short compared with a cell here, and a stamp per sample is both simpler
        // and tighter than dilating a capsule's bounding box.
        const steps = Math.max(1, Math.ceil(length(sub(to, from)) / (0.5 * cell_m)));
        for (let step = 0; step <= steps; step += 1) {
          const point = add(from, scale(sub(to, from), step / steps));
          stamp(branch, point, radius_m);
          stamp(reachable, point, radius_m + reach_m);
        }
      }
    }
  }

  const neighbours26 = (grid: Uint8Array, x: number, y: number, z: number): number => {
    let count = 0;
    for (let dz = -1; dz <= 1; dz += 1) {
      const z2 = z + dz;
      if (z2 < 0 || z2 >= nz) continue;
      for (let dy = -1; dy <= 1; dy += 1) {
        const y2 = y + dy;
        if (y2 < 0 || y2 >= ny) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          const x2 = x + dx;
          if (x2 < 0 || x2 >= nx) continue;
          count += grid[at(x2, y2, z2)];
        }
      }
    }
    return count;
  };
  for (let step = 0; step < stepCount; step += 1) {
    const next = new Uint8Array(live.length);
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          const index = at(x, y, z);
          const count = neighbours26(live, x, y, z);
          // Read entirely off the *previous* generation, so a step is a pure
          // function of it and the sweep order cannot matter.
          if (bare[index] === 1) { next[index] = 0; continue; }
          next[index] = (live[index] === 1
            ? count >= OAK_CROWN_SUPPORT_NEIGHBOURS && count <= crowd
            : branch[index] === 1
              || (count >= birth && reachable[index] === 1))
            ? 1 : 0;
        }
      }
    }
    live = next;
  }

  // One mass per surviving cell. Positions are the cell centres, jittered off a
  // hash of the cell's own coordinates — a lattice with nothing to break it reads
  // as a lattice however organic the set of occupied cells is.
  const attachments: OakFoliageAttachment[] = [];
  let sum = V(0, 0, 0);
  let count = 0;
  for (let index = 0; index < live.length; index += 1) if (live[index] === 1) count += 1;
  if (count > OAK_CROWN_MAXIMUM_CELLS) {
    throw new RangeError(
      `an oak crown grew ${count} foliage cells, over the ${OAK_CROWN_MAXIMUM_CELLS} this module will publish. ` +
        `Every cell is one field-program record and records overflow a brick's 64-candidate ceiling long before ` +
        `any per-object budget bites. Raise crownCell (currently ${cellShare}) rather than ` +
        "this ceiling, or shorten crownStepCount",
    );
  }
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        if (live[at(x, y, z)] !== 1) continue;
        sum = add(sum, centre(x, y, z));
      }
    }
  }
  const centroid = count > 0 ? scale(sum, 1 / count) : spec.root_m;
  const share = count > 0 ? 1 / count : 0;
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        if (live[at(x, y, z)] !== 1) continue;
        // Off the cell's own coordinates rather than a running counter, so a mass
        // keeps its jitter and its shade when a neighbour appears or leaves.
        const cellSeed = hashPath(spec.seed, [x, y, z]);
        const home = centre(x, y, z);
        const at_m = add(home, V(
          OAK_CROWN_CELL_JITTER * cell_m * hashSigned(cellSeed + 0x11),
          OAK_CROWN_CELL_JITTER * cell_m * hashSigned(cellSeed + 0x12),
          OAK_CROWN_CELL_JITTER * cell_m * hashSigned(cellSeed + 0x13),
        ));
        attachments.push({
          at_m,
          // Outward from the crown's own centre. The tiling's direction was the
          // limb's tangent, because a mass was being *placed* past a tip; a cell
          // is already where it belongs, so the only thing left for a direction
          // to mean is which way is out.
          direction: normalizeOr(sub(home, centroid), UP),
          order: foliageOrder,
          share,
          lever_m: length(sub(at_m, spec.root_m)),
          seed: cellSeed >>> 0,
        });
      }
    }
  }
  return { cell_m, attachments };
}



/**
 * Grow one skeleton. Deterministic: one seed in, identical geometry out.
 *
 * The order of operations is the design:
 *
 *  1. **Grow every order**, published or not, in nominal space, and **seed the
 *     crown** off the deepest tips of that full tree. Nothing to this point has
 *     consulted the leaf, and nothing downstream may move either of them.
 *  2. **Fit** the whole object — skeleton and crown together — to `height_m` and
 *     `spread_m`, so the fit is a constant of the spec rather than a function of
 *     the rung.
 *  3. **Publish** the orders the leaf admits, flooring every station's radius at
 *     one and a half leaves — capped at the run's own base radius, so the floor
 *     only ever *truncates a taper* and can never make a run fatter than it
 *     started or invert its profile.
 *  4. **Flare** the foot, in final coordinates.
 *  5. **Bucket** the crown as coarsely as culling asks and the crown's own
 *     composition allows, then repair whatever contact that cost.
 */
export function planOakSkeleton(spec: OakSkeletonSpec): OakSkeleton {
  const publishedOrderCount = oakPublishedOrderCount(spec); // validates
  const tree = growSkeleton(spec);
  const seeds = seedCrown(tree);
  fitToEnvelope(tree, seeds, spec);

  const lean = (() => {
    const [x, z] = spec.leanXZ;
    const magnitude = Math.hypot(x, z);
    return magnitude > 0 ? V(x / magnitude, 0, z / magnitude) : V(1, 0, 0);
  })();

  // The radius floor. Half of three leaves is the radius that draws a diameter
  // at the legibility floor, and capping it at the run's own base radius is what
  // keeps this a truncation rather than a fattening: an admitted run starts at or
  // above the floor by definition of admission, so the cap only binds on the
  // marginal runs at the bottom of an order's fork window — which is exactly the
  // spread that admission, taken at the window midpoint, cannot see.
  const legibleRadius_m = 0.5 * OAK_LEGIBLE_FEATURE_LEAVES * spec.leafSize_m;

  // Coarse to fine, so the skeleton occupies one contiguous run of owner indices
  // per order — which is what lets a consumer re-pose or re-value a whole order
  // with one contiguous write.
  const runs: OakRun[] = [];
  for (let order = 0; order < publishedOrderCount; order += 1) {
    for (const grown of tree.byOrder[order]) {
      // Floored, not truncated. The cap this used to carry — `min` against the
      // run's own base radius, so the floor could only shorten a taper and never
      // fatten a run — was what made admission all-or-nothing: an order below
      // the floor would have published at its natural sub-voxel radius and drawn
      // as speckle. `oakPublishedOrderCount` now admits such an order
      // deliberately, so the floor has to actually reach it.
      //
      // Nothing changes for an order that was already admitted: it clears the
      // floor at its base by definition of the old gate, so the `min` was
      // returning `legibleRadius_m` there anyway.
      const floor_m = legibleRadius_m;
      runs.push(runFrom(grown, grown.radii.map((radius) => Math.max(radius, floor_m)), spec.root_m));
    }
    if (order === 0) runs.push(...growButtresses(tree.byOrder[0][0], spec, lean));
  }

  // The crown, grown rather than placed. Off the *full* tree, published or not:
  // which limbs a leaf happens to draw is not a fact about the crown's layout,
  // and letting it in would leak the rung into the silhouette.
  const crown = growCrownAutomaton(seeds, tree, spec);
  const attachments = crown.attachments;

  let minimum = V(Infinity, Infinity, Infinity);
  let maximum = V(-Infinity, -Infinity, -Infinity);
  const swallow = (at: Vec3, radius_m: number): void => {
    minimum = V(Math.min(minimum.x, at.x - radius_m), Math.min(minimum.y, at.y - radius_m), Math.min(minimum.z, at.z - radius_m));
    maximum = V(Math.max(maximum.x, at.x + radius_m), Math.max(maximum.y, at.y + radius_m), Math.max(maximum.z, at.z + radius_m));
  };
  for (const run of runs) for (const station of run.stations) swallow(station.at_m, station.radius_m);
  // Attachments are swallowed as points. This module does not know how wide a
  // foliage mass is — that is the consumer's `share^(1/3)` sizing — so a caller
  // fitting a container must add the crown's own radius on top of these bounds.
  for (const attachment of attachments) swallow(attachment.at_m, 0);
  if (!Number.isFinite(minimum.x)) {
    minimum = spec.root_m;
    maximum = spec.root_m;
  }

  return {
    spec,
    runs,
    attachments,
    crownCell_m: crown.cell_m,
    bounds_m: { min: minimum, max: maximum },
    maximumLever_m: Math.max(
      0,
      ...runs.map(({ lever_m }) => lever_m),
      ...attachments.map(({ lever_m }) => lever_m),
    ),
    publishedOrderCount,
  };
}

// ---------------------------------------------------------------------------
// The specimens
// ---------------------------------------------------------------------------

/** Phyllotaxis. Nothing lands on top of anything it has already grown. */
const GOLDEN_ANGLE_RAD = Math.PI * (3 - Math.sqrt(5));

/**
 * A mature spreading oak: the hero specimen's orders.
 *
 * ## The radius ladder, and why it is coarser than nature
 *
 * On the hero's 0.80 m specimen with a 20 mm bole radius, the chain
 * {@link oakPublishedOrderCount} walks produces:
 *
 *   order        radiusRatio   nominal diameter   a scaled real oak
 *   0  bole            1.00         40.0 mm             40 mm
 *   1  primary         0.72         20.9 mm             12 mm
 *   2  secondary       0.52          7.4 mm              4 mm
 *   3  tertiary        0.42          2.0 mm            1.2 mm
 *
 * The right-hand column is a real 20 m oak scaled to 0.80 m, and the middle one
 * is deliberately fatter through the whole middle of the tree. The reason is one
 * number: the depth-0 legibility floor is **18.75 mm**, and a faithfully-scaled
 * 12 mm primary limb is 1.9 leaves there — inside the aliasing band. It would
 * either not publish at all, leaving a crown hanging off a bole with nothing in
 * between, or publish as a broken string of cells. Twenty-one millimetres clears
 * the floor with 11% to spare, and that margin is the thing to watch when tuning:
 * dropping `radiusRatio` on order 1 below about 0.65 takes the primaries out of
 * the coarsest rung entirely.
 *
 * Read as ratios the ladder is 1 : 0.52 : 0.19 : 0.050, against nature's
 * 1 : 0.30 : 0.10 : 0.030. The character it produces is right for the species
 * anyway: **the first split is nearly an equal division of the bole**, which is
 * what makes a spreading oak rather than a fir, and every split under it is a
 * little under half.
 *
 * ## What the ladder publishes, measured
 *
 * On a 0.80 x 0.42 m specimen, seed 0x0ac0de, at the four production rungs:
 *
 *   leaf         rung     floor      orders  runs  masses  Σ share  crown box
 *   6.25 mm      depth 0  18.75 mm     2      12     30    1.000    566 x 553 x 483 mm
 *   3.125 mm     depth 1   9.375 mm    2      12     30    1.000    566 x 553 x 483 mm
 *   1.5625 mm    depth 2   4.688 mm    3      42     30    1.000    566 x 553 x 483 mm
 *   0.78125 mm   depth 3   2.344 mm    3      42     30    1.000    566 x 553 x 483 mm
 *
 * The crown box is the attachment cloud's own bounding box and it agrees to
 * **0.00% on every axis**, as does the share-weighted centroid to every printed
 * digit — the degradation contract, read off the product path rather than
 * asserted. Node count is 42, 42, 72, 72: monotone non-decreasing with the rung.
 *
 * Two structural rungs across four, and depth 1 buys no new order. That is not a
 * miss, it is arithmetic: the floor halves per rung while this ladder drops by
 * about 2.8x per order, so a rung and an order are not the same step and cannot
 * be made to be by tuning. What depth 1 and depth 3 *do* buy is the radius floor
 * relaxing — at depth 0 every station under 9.4 mm radius is held at 9.4 and the
 * tree is visibly chunky, and by depth 3 the floor is 1.2 mm and the authored
 * taper runs to its own end — plus whatever the consumer's foliage tape gates in.
 * The floor also pays for itself in records: the fatter coarse runs fit tighter
 * sweep envelopes, and the same skeleton costs 2.25 records a run at depth 0
 * against 3.3 at depth 3.
 *
 * The tertiary order is the interesting one: it never publishes at any rung this
 * stage can hold, and that is what it is *for*. It places the crown. Its 120 tips
 * are the attachment sites, they are computed at every leaf, and they are the
 * reason the silhouette does not move when the rung does. What a leaf changes is
 * only how coarsely those 120 are bucketed — and on this specimen the
 * composition floor pins the bucketing at the secondaries, so not even that
 * moves.
 *
 * What the coarse rungs genuinely cost, and it is not hidden: with the crown
 * pinned and the skeleton culled, 10 of the 30 masses at depth 0 hang off limbs
 * that are not drawn (20 of 30 still reach the published skeleton through the
 * canopy body — see {@link repairCrownContact}). The alternative is a crown that
 * shrinks to meet the skeleton, which is the pop, so this is the right side of
 * the trade and the numbers are here so nobody has to re-derive that.
 *
 * ## The vase
 *
 * `downAngle 1.02` is 58° off the bole — the limb leaves climbing at 32° — and
 * {@link OAK_FORK_HEIGHT_SPLAY} swings that to 68° at the lowest fork and 48° at
 * the highest, which is the tiering. Then `curve 0.62` bows the run out and down
 * over its first half and `curveBack 0.98` lifts it back over its second, for a
 * net rise of 0.36 rad on top of `upAttraction 0.34`. Out, over, and up: the
 * vase. Set `curveBack` to zero and the same limbs read as a weeping willow.
 *
 * Every number here is a starting point for the shape lab, not a settled one.
 * Nothing in this file has been measured against a render.
 */
export const OAK_ORDERS_SPREADING: readonly OakBranchOrder[] = [
  {
    // The bole, and its root flare. `branchCount` is the buttress count,
    // `downAngle_rad` how steeply a root dives, and the fork window the band of
    // the bole they spring from — the flare is the bole's own branch order,
    // pointing down. Everything else on this entry is the bole's own shape.
    branchCount: 5,
    downAngle_rad: 2.10,
    downAngleVariance_rad: 0.22,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.30,
    lengthFraction: 0.52,
    lengthVariance: 0.06,
    radiusRatio: 1.0,
    taper: 0.42,
    curve_rad: 0.16,
    curveBack_rad: 0.10,
    upAttraction: 0.55,
    segmentCount: 7,
    forkStart: 0.0,
    forkEnd: 0.07,
  },
  {
    branchCount: 6,
    downAngle_rad: 1.02,
    downAngleVariance_rad: 0.22,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.34,
    lengthFraction: 0.88,
    lengthVariance: 0.16,
    radiusRatio: 0.72,
    taper: 0.55,
    curve_rad: 0.62,
    curveBack_rad: 0.98,
    upAttraction: 0.34,
    segmentCount: 7,
    /**
     * Where the crown starts, and the one number here that a neighbouring object
     * can falsify.
     *
     * It was 0.34, and 0.34 put the first primary a third of the way up a 740 mm
     * bole — but a primary leaves at 58 degrees off the axis and its foliage
     * droops from there, so the crown's *underside* came out 79 mm above the
     * ground. That is 11 % of the tree's height, against the 30-to-40 % a mature
     * spreading oak carries, and the frame consequence was not subtle:
     * `tests/hero-layout.test.ts` caught a foliage mass swallowing the
     * `plant/air-1` rosette standing beside the trunk.
     *
     * The fix belongs here rather than on the plant. A tree whose lowest leaves
     * brush the ground is a shrub, and the layout is entitled to stand something
     * next to a trunk — a clear bole is what makes that legal. At 0.52 the first
     * fork is a little over half way up, the crown's underside clears about
     * 210 mm, and the rosette is in the open.
     */
    forkStart: 0.52,
    forkEnd: 0.96,
  },
  {
    branchCount: 5,
    downAngle_rad: 0.92,
    downAngleVariance_rad: 0.26,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.45,
    lengthFraction: 0.62,
    lengthVariance: 0.22,
    radiusRatio: 0.52,
    taper: 0.58,
    curve_rad: 0.48,
    curveBack_rad: 0.66,
    upAttraction: 0.26,
    segmentCount: 5,
    forkStart: 0.26,
    forkEnd: 0.90,
  },
  {
    // Never published at any rung this stage can hold. It exists to place the
    // crown, and the foliage that would have hung on it is absorbed upward.
    branchCount: 4,
    downAngle_rad: 0.86,
    downAngleVariance_rad: 0.30,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.55,
    lengthFraction: 0.55,
    lengthVariance: 0.26,
    radiusRatio: 0.42,
    taper: 0.60,
    curve_rad: 0.40,
    curveBack_rad: 0.44,
    upAttraction: 0.18,
    segmentCount: 4,
    forkStart: 0.30,
    forkEnd: 0.92,
  },
  /**
   * Twigs — the fifth order, and the one this table exists to prove costs
   * nothing to author.
   *
   * `oakPublishedOrderCount` admits an order only while the diameter it would be
   * *drawn* at stays under its parent's, so at a coarse rung this is simply
   * absent and the tree is the four-order tree it was before. It appears when
   * the leaf is fine enough for a twig to be thinner than the tertiary carrying
   * it, and not one rung sooner.
   *
   * Three differences from the orders above, all saying "twig" rather than
   * "small branch":
   *
   *  - **Forked from the outer half only** (`forkStart` 0.55). Every other order
   *    forks along nearly the whole of its parent, because a limb bears limbs
   *    down its length. A twig bears leaves, and it bears them at the end — this
   *    is what puts the new wood *at the tips* rather than distributed along the
   *    tertiaries, which is the whole point of adding it.
   *  - **Splayed and barely drawn upward** (`downAngle_rad` 1.05, `upAttraction`
   *    0.08). Twigs fan into the light rather than reaching for it; the strong
   *    `upAttraction` of a structural limb would comb them into brushes.
   *  - **Two segments and almost no curve.** At this size a run is a few voxels
   *    long and curvature it cannot resolve is march cost for nothing.
   *
   * `radiusRatio` 0.46 rather than the tertiary's 0.42, because the admission
   * rule already refuses this order when it would be as fat as its parent — so
   * the ratio can be honest about a twig being a fair fraction of the shoot it
   * grew from, and the ladder rather than the table decides when it draws.
   */
  {
    branchCount: 3,
    downAngle_rad: 1.05,
    downAngleVariance_rad: 0.34,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.62,
    lengthFraction: 0.48,
    lengthVariance: 0.30,
    radiusRatio: 0.46,
    taper: 0.55,
    curve_rad: 0.22,
    curveBack_rad: 0.18,
    upAttraction: 0.08,
    segmentCount: 2,
    forkStart: 0.55,
    forkEnd: 0.95,
  },
];

/**
 * A younger, more upright oak — the article's "older or younger versions" made
 * into a second form rather than a second module.
 *
 * The point of shipping two is to prove the parameters describe a *species*: the
 * same growth model, the same admission ladder, a recognisably different tree.
 * What moved and why:
 *
 *  - **A longer clean bole** (`lengthFraction 0.66` against 0.52) with the fork
 *    window starting at 0.46 instead of 0.34. A young oak in competition holds a
 *    leader and carries no low limbs; the spreading form is what an open-grown
 *    tree becomes decades later.
 *  - **Steeper limbs** (`downAngle 0.72` — 41° — against 58°) and much stronger
 *    gravitropism (`upAttraction 0.52` against 0.34), with less of an S. That is
 *    the difference between a crown that reaches out and one that reaches up.
 *  - **Fewer, thinner children** at every order, and a slighter flare: four
 *    buttress roots rather than five, and `taper 0.38` on a bole that has not yet
 *    put on its lower mass.
 *
 * The radius ratios were then re-solved rather than copied, because the ladder
 * has to survive the change: with `taper 0.38` and a fork window centred at 0.72,
 * `radiusRatio 0.70` lands the primary at 20.3 mm and it still clears the
 * depth-0 floor. The nominal diameters come out 40.0 / 20.3 / 7.1 / 2.0 mm and
 * the published-order ladder is identical to the spreading form's. **A sibling
 * form that refines differently is a second species wearing this one's
 * parameters**, so that check is the one to run first after changing anything
 * here.
 *
 * Measured on the same 0.80 x 0.42 m specimen: orders 2 / 2 / 3 / 3 across the
 * four production rungs, 10 / 10 / 30 / 30 runs, 20 masses at every rung,
 * `Σ share` exactly 1, and an attachment-cloud box of 435 x 442 x 607 mm that
 * drifts 0.00% on every axis. Same contract, different tree.
 */
export const OAK_ORDERS_UPRIGHT: readonly OakBranchOrder[] = [
  {
    branchCount: 4,
    downAngle_rad: 2.16,
    downAngleVariance_rad: 0.18,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.24,
    lengthFraction: 0.66,
    lengthVariance: 0.05,
    radiusRatio: 1.0,
    taper: 0.38,
    curve_rad: 0.10,
    curveBack_rad: 0.08,
    upAttraction: 0.72,
    segmentCount: 7,
    forkStart: 0.0,
    forkEnd: 0.06,
  },
  {
    branchCount: 5,
    downAngle_rad: 0.72,
    downAngleVariance_rad: 0.18,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.30,
    lengthFraction: 0.72,
    lengthVariance: 0.14,
    radiusRatio: 0.70,
    taper: 0.50,
    curve_rad: 0.34,
    curveBack_rad: 0.52,
    upAttraction: 0.52,
    segmentCount: 6,
    forkStart: 0.46,
    forkEnd: 0.98,
  },
  {
    branchCount: 4,
    downAngle_rad: 0.78,
    downAngleVariance_rad: 0.24,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.40,
    lengthFraction: 0.58,
    lengthVariance: 0.20,
    radiusRatio: 0.50,
    taper: 0.56,
    curve_rad: 0.30,
    curveBack_rad: 0.42,
    upAttraction: 0.40,
    segmentCount: 5,
    forkStart: 0.30,
    forkEnd: 0.92,
  },
  {
    branchCount: 4,
    downAngle_rad: 0.80,
    downAngleVariance_rad: 0.28,
    rotate_rad: GOLDEN_ANGLE_RAD,
    rotateVariance_rad: 0.50,
    lengthFraction: 0.52,
    lengthVariance: 0.24,
    radiusRatio: 0.44,
    taper: 0.58,
    curve_rad: 0.26,
    curveBack_rad: 0.30,
    upAttraction: 0.30,
    segmentCount: 4,
    forkStart: 0.32,
    forkEnd: 0.94,
  },
];
