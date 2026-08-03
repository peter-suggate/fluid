import type { Vec3 } from "../model";
import type { SceneryMaterial, SceneryNode } from "../scenery-graph";
import {
  SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES,
  SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY,
  SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER,
} from "../svo-primitive-abi";
import { alongAxis, V } from "./builder";
import { sweptTubeNodes, type SweptTubeStation } from "./swept-tube";

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
 * ## Where the smooth union went
 *
 * The plan's answer to both halves of this object is one marched SDF primitive
 * that smooth-unions a displaced cluster (§6b item 3). This module was written
 * as the control arm of that decision — everything built from the exact
 * analytic kinds the scenery graph already published, so that what it could and
 * could not reach was evidence about whether the new kind was required. The
 * answer came back *required*, the kind exists now, and this whole object is
 * built from it: the crown is a jittered lattice `cluster` and every run below
 * it — base, buttress, trunk, limb — is a `tapered-sweep` one.
 *
 * The trunk used to be a chain of cones with a sphere at every node, and that
 * construction is *exact*: a sphere of the node's own radius covers the flat cap
 * a cone ends in, so the union is a round-capped tapered tube with no gap at any
 * bend. It still rendered wrong, and the reason is the one thing a union cannot
 * fix. A hard union is C⁰ but not C¹, so the shading normal steps at every
 * junction; a smooth taper came out as a stack of faceted pipes with a seam
 * between each, and no amount of subdividing helps because subdividing adds
 * junctions. A swept field fuses its segments with a smooth minimum, which is C¹
 * through the junction, and it costs one record for a whole run instead of two
 * per segment.
 *
 * Two techniques still stand in for a smooth union *between* records, because
 * two records always meet in a hard one:
 *
 *  - **A fillet is placed, not solved.** A smooth minimum is a ball rolled along
 *    a seam; where two buttress roots leave the base together, that ball is put
 *    in the notch explicitly and sized from the gap it has to bridge.
 *  - **A part starts inside its parent.** A sweep is capped at both ends, so a
 *    run that began on its parent's surface would show a cap disc there. Every
 *    run here begins inside the mass it leaves — a buttress on the axis, a limb
 *    within its trunk — and what shows is the junction rather than the joint.
 *
 * ## Why a rotated ellipsoid here is always a spheroid
 *
 * `alongAxis` returns the *shortest-arc* rotation taking local +Y onto a
 * direction, which fixes one axis and leaves the roll about it unspecified. A
 * triaxial ellipsoid rotated that way would spin its two cross-axes with the
 * direction and read as a different shape at every azimuth. So anything rotated
 * in this module has `radius.x === radius.z` and varies only along local +Y —
 * florets and fillets are both round in plan anyway. Unrotated ellipsoids, which
 * here means the canopy lobes and their closure cores, are free to be triaxial
 * because their axes are the world's — and the same argument is why every
 * `cluster` here is placed unrotated: its envelope is the crown's own plate.
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
 * ## Which is why the crown is two records a lobe
 *
 * A `cluster` evaluates its packing in the shader, so its grain costs no
 * records at all and the economy above stops applying to it — the reference's
 * own six-millimetre floret, which the explicit build could not have reached
 * with the whole scene budget, is free. What the aggregate cannot do is be both
 * granular and closed: a lattice only stops letting rays through at
 * `r >= period * sqrt(3)/2`, and at that ratio the packing is a solid, the
 * envelope's hard `max` cuts the solid off at exactly the envelope, and the
 * lobe draws as a bare ellipsoid. So the two jobs are two records: a cluster at
 * a ratio where the florets are distinct, and an ellipsoid *core* one
 * granulation-depth under it that no ray gets past. That is the same division
 * of labour the explicit build had — florets for the surface, a lobe core for
 * the opacity — spelled in two records rather than a hundred.
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
  /**
   * Outer plan extent of the crown from its own centre, on x and z, and a hard
   * bound rather than a target: every head is inset by its own plan radius, so
   * nothing published reaches past this. A finite set of round heads inscribed
   * in a disc spans a little under the disc, so the measured extent comes out
   * around nine tenths of it; see `CROWN_RIM_CROWDING`.
   */
  readonly crownRadius_m: readonly [number, number];
  /**
   * Total vertical thickness of the crown *envelope* — the crown's own surface,
   * not the lobes underneath. Against `crownRadius_m` this is the single number
   * that decides whether the specimen reads as a flat overhanging canopy or as a
   * ball; the reference's is about five and a half to one, and the crown stops
   * reading as a cloud layer somewhere under four.
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
   * Canopy lobes — the clumped masses the crown reads as, one cauliflower head
   * apiece. They are flattened balls laid out in a plane, and the crown is flat
   * because of the arrangement as much as because of the flattening, so this and
   * `crownThickness_m` are not independent: the plate is exactly one mass thick
   * under the aggregate crown, and a count that disagrees reads as a lumpy
   * dinner plate however the rest is tuned. Tiling the crown's plan area is the
   * constraint, and the head count is what the tiling is judged on — too many
   * and the masses stop being distinguishable, too few and the rim goes from
   * scalloped to polygonal.
   */
  readonly lobes: number;
  /**
   * Lobe overlap, as a multiple of the radius that would pave the crown's plan
   * area exactly. One is that paving, and it is *not* a covering — circles of
   * equal area to their share of a plane always leave the gaps between circles —
   * so a form that wants a continuous crown authors this above one. Much above
   * it and the heads dissolve into a single mass; a cauliflower head is a set of
   * heads.
   */
  readonly lobeFill: number;
  /**
   * Spread of lobe sizes about the mean, as a fraction. Zero is a paving
   * pattern; this is what turns a mound of equal balls into a few large heads
   * with smaller ones packed against them. Each lobe is inset by *its own* plan
   * radius rather than the nominal one, so the spread costs nothing in
   * silhouette however wide it is opened.
   */
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
   * Half-width of one *explicit* floret. Halving it quadruples the records
   * needed to keep the surface fused; see the header. Unused under
   * `canopy: "aggregate"`, which sets its grain with `canopyGrainAcross`
   * instead — this number used to serve both and it should not have, because
   * under the aggregate a floret costs no records and the whole economy this
   * one is expressed in has stopped applying.
   */
  readonly floretRadius_m: number;
  /**
   * Florets across the crown's full width, for the aggregate crown.
   *
   * The granulation, as the number the reference is actually read for: count the
   * florets along the widest part of the silhouette. Sixty and up is the
   * reference's fine broccoli curd; twenty is a cobbled ball. It sets the
   * lattice period directly — `2 * crownRadius_m[0] / canopyGrainAcross` — and
   * every other length of the packing is a fixed ratio to that period, so this
   * is the only granulation knob and it costs nothing to turn.
   */
  readonly canopyGrainAcross: number;
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
  /**
   * How the crown is spelled: as explicit florets, or as one aggregate per lobe.
   *
   * `florets` is the construction every field above describes — lobe cores,
   * nodules and cast florets, three scales of explicit primitive, and very
   * nearly the whole record count. It is honest and it does not scale: the pond
   * canopy spends 1 339 of the scene's 4 096 records on it, and — the part that
   * is not a budget but a correctness wall — averages some eighty-three of them
   * in each sparse brick against a per-brick candidate ceiling of sixty-four
   * whose overflow is a *silent drop*. Past that ceiling the surplus florets
   * never reach the opacity pyramid or the radiance atlas: the crown keeps
   * drawing in primary visibility while quietly ceasing to cast a shadow.
   *
   * `aggregate` publishes two records per lobe instead: a `cluster` — a
   * jittered sphere packing clipped to the lobe's own ellipsoid, evaluated in
   * the shader — for the granulation, and an ellipsoid *core* one granulation
   * depth under it for the closure the packing cannot carry at a ratio where
   * its florets are still distinct. Eight records rather than eleven
   * hundred, and a grain that costs nothing to refine. `floretsPerLobe`,
   * `bumpsPerLobe`, `floretRadius_m`, `floretFlatten` and `floretSeat` are all
   * unused under it; `canopyGrainAcross` is what sets the surface.
   */
  readonly canopy: "florets" | "aggregate";
  /**
   * Stump radius. Every length below the crown is a multiple of it — the flare
   * at its foot, the buttresses that leave it and the trunks that rise from it —
   * so this is the one number that says how heavy the base reads against the
   * crown over it. The reference's is about a sixth of the crown's radius.
   */
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
  /**
   * Trunk radius as a fraction of the stump's, where it leaves the stump. Two
   * thirds, not half: a trunk that starts at half a stump it then tapers, forks
   * and tapers again arrives at the canopy as a dark stick, which is what the
   * first version of this looked like against a white crown.
   */
  readonly trunkWidth: number;
  /**
   * Limb radius as a fraction of the trunk's, where the trunk ends. Under one,
   * always: a limb wider than the trunk it forks from shows its own end cap
   * standing proud of the fork, and a fork that shows a disc is the one junction
   * in this object that reads as assembly rather than as growth.
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
   * Buttress roots. They leave the *axis* — inside the stump, so the joint has
   * no seam to close — climb out through its flare, and run down and away to end
   * in blunt toes bedded in whatever is around them. What the reference shows is
   * a base that grips: the flare is not the stump's own taper but these, thick
   * where they leave it and splayed by the time they land.
   *
   * They were fingers before, and fingers is what they looked like: the run
   * started two fifths of the way out and only `rootWidth` of the stump thick,
   * so the base read as a post with tabs beside it instead of as a root plate.
   */
  readonly roots: number;
  /** Buttress length in stump radii, measured from the axis. */
  readonly rootReach: number;
  /**
   * Buttress radius at the shoulder, as a fraction of the stump's. Near one, and
   * it has to be: a buttress leaves the axis inside the stump, so anything much
   * thinner emerges as a rib on the flare rather than as part of it. Thick
   * enough that neighbours overlap where they leave, which is what makes the
   * base read as one fused mass rather than as legs.
   */
  readonly rootWidth: number;
  /**
   * How much a buttress narrows along its run — which under an anisotropic fin
   * is spelled as how far back toward the trunk its mass sits, because an
   * ellipsoid falls to nothing at both ends and a fin centred on its own run
   * would be fattest in the middle, the one place a root never is. Zero is that
   * leaf shape; near one the fin leaves thick and thins the whole way out.
   */
  readonly rootTaper: number;
  /** Palette naming the trunk, root and limb ramp. */
  readonly barkPalette: string;
  /** Palette naming the canopy ramp. */
  readonly canopyPalette: string;
  /**
   * Linear albedo of a limb on `barkPalette`. Stump and buttresses sit just
   * under it, and the ramps in either direction are narrow: this specimen is
   * fired porcelain standing in plaster, so what separates its parts is form
   * under a raised key rather than value.
   */
  readonly barkValue: number;
  /** Linear albedo of a sunlit floret. Crevices and lobe cores ramp down from it. */
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
  /**
   * World-space extent of the canopy alone — the plate, without the trunk under
   * it or the buttresses running down the bank away from it.
   *
   * Separate because every question anyone asks about this crown is about the
   * crown: whether it is a plate or a ball, whether it lands on the width it was
   * authored at, whether it clears what it overhangs. Measured off the whole
   * specimen, all three answers move when the trunk does.
   */
  readonly crownBounds_m: { readonly min: Vec3; readonly max: Vec3 };
  /** Where each buttress root's toe enters the ground, in world metres. */
  readonly rootFeet_m: readonly Vec3[];
  /**
   * Lobe envelopes in world metres. Under `aggregate` these are the crown's own
   * surface — the packing is clipped to exactly them; under `florets` they are
   * the cores, before nodules and florets thicken them.
   */
  readonly lobes: readonly BonsaiLobe[];
  /** Radius of one canopy nodule, derived from the lobe size and count. */
  readonly bumpRadius_m: number;
  /**
   * Summed surface area over the crown's own envelope area. Under florets the
   * surface is a floret cap: one is tangent caps with gaps between them and two
   * is the overlap at which they fuse. Under `aggregate` it is a whole head, and
   * the ratio is then the heads' own overlap — whether the crown is one fused
   * mass with clefts in it or a heap of separate balls. The one number that says
   * whether a form's counts are enough for its sizes, without rendering it.
   */
  readonly floretCoverage: number;
}

/**
 * The reference's specimen: a wide flat-topped multi-trunk overhanging a pond,
 * about 0.74 m across and 0.40 m tall on a 1.8 m stage.
 *
 * Read off the reference as three proportions and one count. The crown is five
 * and a half times as wide as it is thick, which is what makes it a cloud layer
 * over the trunk rather than a ball on it. The crown is *one mass*: four
 * overlapping plates, each nine tenths of the crown's own radius, so what the
 * union reads as is a single flat-topped cloud with a softly irregular outline
 * rather than a bunch of separate heads with sky between them. Its clumping and
 * its curd both come out of one record's octave stack — masses at 46 mm, lumps
 * at 23 mm, florets at 12 mm, sixty-four of them across the width.
 *
 * The trunk carries a fifth of the crown's radius at the base and splays into
 * seven buttresses, which together are twice as wide as the base they leave.
 *
 * The record count is the *worst* seed's rather than this one's. Culling is
 * geometric, so the count swings as the lobes move, and a form tuned on its own
 * seed overruns the budget on a re-seed — which is what `tests/bonsai.test.ts`
 * found. Every count here is the value at which four hundred seeds all fit.
 */
export const BONSAI_POND_CANOPY: BonsaiForm = Object.freeze({
  height_m: 0.40,
  crownRadius_m: [0.37, 0.32] as const,
  crownThickness_m: 0.135,
  crownOverhang_m: 0.11,
  crownDroop: 0.34,
  lobes: 9,
  lobeFill: 1.38,
  lobeSwell: 0.45,
  bumpsPerLobe: 16,
  floretsPerLobe: 106,
  floretRadius_m: 0.0293,
  canopyGrainAcross: 64,
  floretFlatten: 0.56,
  floretSeat: 0.55,
  floretGrain: 0.52,
  canopy: "aggregate",
  boleRadius_m: 0.058,
  stumpShare: 0.11,
  forkShare: 0.46,
  trunks: 3,
  trunkSplay: 2.5,
  trunkWidth: 0.88,
  limbWidth: 0.72,
  limbsPerTrunk: 2,
  limbBow: 0.30,
  roots: 5,
  rootReach: 2.9,
  rootWidth: 0.92,
  rootTaper: 0.62,
  barkPalette: "stone",
  canopyPalette: "clay",
  barkValue: 0.91,
  canopyValue: 0.90,
  maximumLeaves: 1_500,
});

/**
 * The same species grown as a standard: half again as tall, a crown a third
 * narrower and much deeper, three trunks instead of four, and no overhang to
 * speak of. What a courtyard or a gallery wants — something to stand *beside*
 * rather than under.
 *
 * `canopyGrainAcross` is inherited rather than re-authored, which is the point
 * of counting florets across the crown instead of authoring their radius: the
 * grain scales with the specimen and a variant that is two thirds the width
 * gets florets two thirds the size without saying so.
 */
export const BONSAI_COURTYARD_STANDARD: BonsaiForm = Object.freeze({
  ...BONSAI_POND_CANOPY,
  height_m: 0.62,
  crownRadius_m: [0.24, 0.22] as const,
  crownThickness_m: 0.17,
  crownOverhang_m: 0.03,
  crownDroop: 0.18,
  lobes: 9,
  bumpsPerLobe: 14,
  floretsPerLobe: 150,
  floretRadius_m: 0.029,
  boleRadius_m: 0.048,
  stumpShare: 0.05,
  forkShare: 0.52,
  trunks: 3,
  trunkSplay: 1.4,
  limbsPerTrunk: 3,
  roots: 6,
  rootReach: 2.4,
});

/**
 * A shelf specimen at a third the scale: nine lobes and three trunks, with a
 * grain that follows the crown down without being asked to.
 *
 * It is the honest end of the record economy in the other direction. Under the
 * explicit crown a smaller specimen bought a much finer surface for the same
 * budget, because the area to cover falls with the square; under the aggregate
 * it buys nothing, because the surface never cost records in the first place.
 */
export const BONSAI_SHELF_MINIATURE: BonsaiForm = Object.freeze({
  ...BONSAI_POND_CANOPY,
  height_m: 0.155,
  crownRadius_m: [0.125, 0.115] as const,
  crownThickness_m: 0.055,
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

/**
 * Octaves of lattice the crown's packing stacks.
 *
 * Three, the ABI's ceiling, and it is what makes this a cauliflower rather than
 * a cobbled ball. Octave `k` repeats at half the last one's period with half its
 * floret and half its blend, so one record carries the whole hierarchy the
 * reference shows — head-sized masses at 40 mm, lumps at 20 mm, curd at 10 mm —
 * for one extra lattice evaluation each and no extra records at all. Spelling
 * the same three scales as separate records is what the explicit crown did, and
 * it cost 1 158.
 */
const CANOPY_LATTICE_OCTAVES = 3;
if (CANOPY_LATTICE_OCTAVES > SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES) {
  throw new RangeError(`Bonsai canopy asks for ${CANOPY_LATTICE_OCTAVES} lattice octaves; the render ABI allows ${SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES}`);
}

/**
 * Floret radius as a fraction of its own octave's period.
 *
 * The one number the crown's whole character turns on, and the direction it has
 * to move in is the opposite of what it looks like. At `r = 0.95 P` — what this
 * was — every corner of a lattice cell is inside a sphere, so the packing is a
 * *solid*, and the hard `max` against the envelope then cuts that solid off at
 * exactly the envelope: every lobe drawing as the bare ellipsoid the cluster was
 * published to avoid. Octaves make that worse, not better, because they fill:
 * three stacked lattices are `1 - (1 - phi)^3` dense where one is `phi`, so the
 * ratio that reads as granular under one octave reads as flush under three.
 *
 * Measured by casting rays at the hero crown's own plate envelope through the
 * render ABI's CPU tracer, with the closure core in place and the grain fixed at
 * 64 florets across:
 *
 *              r/P   0.34   0.38   0.42   0.46   0.55   0.62
 *   1 octave relief   —      —     3.48   3.76   2.56   1.65  mm
 *   3 octave relief  8.32   6.53   4.58   3.20   0.86   0.19  mm
 *   3 octave core     6%     3%     1%     0%     0%     0%
 *
 * `relief` is the RMS depth of the drawn surface under the envelope — the
 * granulation, in the units the eye reads it in — and the bottom-right corner is
 * the whole diagnosis: two tenths of a millimetre is not a rumple, it is
 * *nothing*, the ellipsoid exactly. Under three octaves 0.38 is the largest
 * ratio whose relief is still a floret-sized feature and the smallest at which
 * the closure core is not showing through; below 0.34 the core is bare over a
 * tenth of the crown and it reads as beads scattered on a ball.
 */
const CANOPY_FLORET_PERIOD_SHARE = 0.38;

/**
 * Smooth-minimum radius, as a fraction of the floret it fuses.
 *
 * In direct competition with the relief above, because a polynomial smooth
 * minimum rounds away every feature under its own radius. A blend of a third of
 * a floret — what this was — fills most of the valley between two neighbours.
 * Just over a fifth still fuses the ones the jitter has pushed together, which
 * is the clumping that separates a cauliflower from a raspberry, and it scales
 * down the octave stack with everything else, so each scale is blended in its
 * own proportion rather than the coarsest one's.
 */
const CANOPY_FLORET_BLEND_SHARE = 0.22;

/**
 * Lattice displacement from the cell centre, as a fraction of the period.
 *
 * The ABI's ceiling — {@link SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER} — and this
 * crown wants all of it. A crystal reads as a crystal at any ratio, and the
 * jitter is what makes some neighbours merge into a head and others stand apart;
 * that variance *is* the clumping. Past the ceiling a sphere wanders far enough
 * to open a hole in its own row, which here the core would show through.
 */
const CANOPY_JITTER = SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER;

/**
 * How far the closure core sits under the crown's surface, in *coarse* lattice
 * periods.
 *
 * The core is what no ray gets past, and it exists because a lattice that is
 * closed is not granular; see the header. Its depth is the depth of the
 * granulation, so it is a multiple of the period rather than of the plate —
 * scale the specimen and the crevices scale with the curd — and of the coarsest
 * octave's period, because that is the octave whose gaps are deepest.
 *
 * One period, and the number is not obvious in the direction it goes: a *deeper*
 * core shows less of itself, not more, because the lattice's next layer arrives
 * before the core does. Any less and the core swallows the granulation; any more
 * and the valleys start reading as holes rather than as crevices.
 */
const CANOPY_RELIEF_PERIODS = 1.0;

/**
 * How much of its parent a trunk, and then a limb, has spent by the time it
 * ends.
 *
 * Both are steep, and both had to be. The first version tapered a trunk by a
 * third over its whole run and rendered as four near-vertical legs of roughly
 * constant thickness under a plate — table legs, not a tree. A run that arrives
 * at half of what it left with reads as grown, and the exponent under one puts
 * most of that narrowing in the first half, where a tree's actually is.
 */
const TRUNK_TAPER = 0.52;
const LIMB_TAPER = 0.60;

/**
 * How far out a limb's target sits, as a share of the crown's own radius.
 *
 * Well inside it, on purpose. A limb ends *within* the mass it holds up, because
 * a canopy resting on nothing is the most obvious way a generated tree reads as
 * generated — and one that ends at the rim pokes out of the silhouette instead.
 */
const LIMB_TARGET_SHARE = 0.62;

/**
 * How far up its trunk a limb forks, as a share of the trunk's own run.
 *
 * Under one, always, and that is the joint treatment rather than a proportion: a
 * sweep is capped at both ends, so a limb whose first station is anywhere but
 * *inside* its parent shows that cap as a ring with a dark interior. Starting on
 * the trunk's own centreline four fifths of the way up puts the station inside
 * the trunk's section by construction, and the fork reads as a fork.
 */
const LIMB_FORK_SHARE = 0.80;

/**
 * How much empty envelope a run *at ground level* may carry, against
 * {@link TUBE_ENVELOPE_WASTE_BUDGET} for the ones in the air.
 *
 * A cluster's envelope is not decoration: it is what the voxelizer, the bounds
 * formulae and the opacity pyramid read, so a loose one marks bricks occupied
 * and occludes indirect light over a volume the record does not fill. Up in the
 * canopy there is nothing there to occlude. At the foot there is — the set beds
 * air plants and pebbles right against these buttresses, and at the default
 * budget one buttress's envelope reached far enough to swallow a plant whole and
 * `tests/hero-layout.test.ts` said so.
 *
 * Six, measured on the specimen's own 200 mm bricks against the same 64-candidate
 * contract the scene is judged on — records, then the busiest brick the bonsai
 * puts anything in:
 *
 *   budget    12     10      8      6      4
 *   records   53     55     67     84    144
 *   busiest   36     38     45     52     86
 *
 * It is not monotone in the direction it looks: a tighter budget splits a run,
 * and every split is a whole extra record whose *own* envelope still has to
 * clear the containment bound, so past a point the count grows faster than the
 * envelopes shrink and the brick load goes back up. Four is past that point.
 * Six is the loosest budget at which nothing the set beds against this base is
 * swallowed, and it leaves the specimen's busiest brick under the contract.
 */
const GROUND_ENVELOPE_WASTE_BUDGET = 6;

/** How tall a fin stands where it leaves the trunk, as a fraction of its shoulder width. */
const BUTTRESS_HEIGHT_SHARE = 1.45;

/**
 * Envelopes chained along one fin, and how far each overlaps its neighbour.
 *
 * The count is what buys the blade. A single envelope over the whole run makes
 * the *length* the longest half-axis, and the eccentricity cap below then sets
 * the thin axis off that length rather than off the height — which is how one
 * envelope per root came out with a section near enough square. Four short ones
 * put the length under the height for most of the run, so the cap binds on the
 * height instead and the section falls to one in four.
 *
 * The overlap is not optional: two records meet in a hard union, and a chain
 * that merely abutted would show a gap at every junction. A third of a segment
 * either way puts each junction well inside its neighbour, where the two blades
 * already agree on their section and the union is invisible.
 */
const BUTTRESS_SEGMENTS = 4;
const BUTTRESS_SEGMENT_OVERLAP = 0.34;

/**
 * The largest ratio between a fin envelope's longest and shortest half-axis.
 *
 * A soundness bound wearing a proportion's hat, and it is the *envelope's*
 * rather than the lobes'. A cluster is clipped by `(|p/R| - 1)·min R`, a
 * Lipschitz-1 lower bound whose slack goes as the envelope's own eccentricity,
 * so a fin `n` times thinner than it is long understeps by up to `n` and, with
 * {@link SVO_PRIMITIVE_MARCH_ITERATIONS} fixed at 48, a ray runs out before it
 * arrives. That failure is a *hole*, reported nowhere.
 *
 * Measured with the ABI's own tracer on a blade envelope, rays cast at it from
 * 676 directions through its centre — where the field provably is, so a miss can
 * only be the march giving up:
 *
 *   eccentricity    3      4      5      6      8     10
 *   rays that hit  100%   100%  99.7%  97.5%  92.5%  82.3%
 *
 * Four is the last one clean, and it is also enough: it puts the fin's thickness
 * at a quarter of its height, which is the thick end of what reads as a blade.
 * Anything thinner has to come from a cap this measurement does not support.
 */
const BUTTRESS_MAXIMUM_ECCENTRICITY = 4;

/**
 * The fin's blend, as a fraction of its *shortest* half-axis.
 *
 * Which is the axis the ABI measures it against: a seeded-lobes packing must
 * keep `span + spread + blend/shortest` under one, or the hard `max` slices the
 * lobes it was there to contain. The defaults spend 0.72 of that budget, so this
 * has 0.28 to work in and takes half of it — enough to round the lumps into one
 * blade, not enough to swell it back toward its envelope.
 */
const BUTTRESS_BLEND_SHARE = 0.15;

/**
 * How far the ground may fall under a buttress before it has left the bank, as
 * a fraction of the stump radius.
 *
 * The generator is handed a height query and nothing else, on purpose — a
 * specimen has no business knowing whether it is standing beside a pond, a wall
 * or a step. What it *can* see is that the ground under a bearing has dropped
 * away, and that is what the edge of a bank is. On the hero stand the terrace
 * falls 70 mm within 80 mm toward the water while the coping's outer foot is at
 * 100 mm, so a fin that stops at a bole radius of fall stops on soil with room
 * to spare; the bearings that run inland fall 50 mm in 240 mm and never reach
 * the limit, so they are not shortened at all.
 */
const ROOT_MAXIMUM_FALL_SHARE = 0.85;

/**
 * The shortest run worth publishing a buttress along, in stump radii.
 *
 * Below it the four chained envelopes are shorter than they are tall and the
 * chain stops being a blade, so the bearing is left bare. It is a floor rather
 * than a target: on the hero stand the pond-side bearings clamp to about two
 * thirds of a bole and still publish, which is the stub a bank that narrow
 * actually has room for.
 */
const BUTTRESS_MINIMUM_REACH_SHARE = 0.6;

/**
 * Anisotropy of the fin's own lobes, against the ABI's ceiling of four.
 *
 * The lobes live in the envelope's normalised space, so this is irregularity on
 * top of the fin's proportions rather than instead of them: at one every lump is
 * a scaled copy of the fin and the silhouette regularises, and at the ceiling
 * the lobes are flat enough that the march's own lobe bound starts to bite.
 */
const BUTTRESS_LOBE_ANISOTROPY = 2.6;
if (BUTTRESS_LOBE_ANISOTROPY > SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY) {
  throw new RangeError(`Bonsai buttress lobes ask for anisotropy ${BUTTRESS_LOBE_ANISOTROPY}; the render ABI allows ${SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY}`);
}

/**
 * Exponent of the sunflower's radial coordinate — how hard the heads crowd onto
 * the crown's rim.
 *
 * A half is the equal-area spiral, which spreads the heads evenly over the
 * crown's *plan area* and therefore leaves only a third of them anywhere near
 * its edge. That is the right distribution for a dome and the wrong one for a
 * plate: a crown this flat is seen mostly edge-on, so what the silhouette is
 * made of is the rim, and the rim wants to be scalloped by heads rather than
 * polygonal between four of them.
 *
 * The measured effect on the span is small and worth stating so nobody re-tunes
 * this expecting otherwise — worst of eight seeds, as a fraction of the authored
 * 0.74 m width:
 *
 *   exponent  0.50   0.42   0.38   0.34   0.30
 *   span      .885   .901   .909   .919   .927
 *
 * The residual tenth is structural rather than a tuning failure. Every head is
 * inset by its own radius, so the crown is provably inside `crownRadius_m`, and
 * a finite set of round heads inscribed in a disc spans less than the disc:
 * whichever head is nearest the x axis is a few degrees off it and a little
 * short of the rim, and the span is that head's reach rather than the disc's.
 * Crowding buys some of it back and nothing buys the rest without letting a head
 * out past the silhouette the form authored.
 *
 * A third, then: the rim is dense enough that the silhouette no longer depends
 * on which azimuth the seed dealt the widest head, and the middle of the plate
 * is still covered by the innermost head's own radius.
 */
const CROWN_RIM_CROWDING = 0.34;

/**
 * The render ABI's neighbourhood condition, discharged once here at the ratios
 * above rather than trusted per call site — it is a property of the sphere
 * trace, and violating it makes rays tunnel through the crown:
 *
 *   (2 - jitter)·period >= floretRadius + smoothRadius
 *   (2 - 0.30)·P        >= (0.38 + 0.22·0.38)·P
 *   1.70·P              >= 0.464·P                      ✓ 3.7x margin
 *
 * Both sides are proportional to the period, so it holds for every
 * `canopyGrainAcross` a form may author — and for every octave of the lattice,
 * since an octave scales period, floret and blend together.
 */
const CANOPY_NEIGHBOURHOOD_MARGIN =
  (2 - CANOPY_JITTER) / (CANOPY_FLORET_PERIOD_SHARE * (1 + CANOPY_FLORET_BLEND_SHARE));
if (!(CANOPY_NEIGHBOURHOOD_MARGIN >= 1)) {
  throw new RangeError(
    `Bonsai canopy ratios violate the cluster neighbourhood condition by ${(1 / CANOPY_NEIGHBOURHOOD_MARGIN).toFixed(3)}x`,
  );
}

/** The aggregate crown's packing, in metres, derived from the form alone. */
export interface BonsaiCanopyGrain {
  /** Domain-repetition period of the *coarsest* octave — the head-sized masses. */
  readonly latticePeriod_m: number;
  /** Radius of one floret of the coarsest octave. */
  readonly floretRadius_m: number;
  /** Radius of the smooth minimum that fuses the coarsest octave's florets. */
  readonly blendRadius_m: number;
  /** Octaves stacked. Each halves the period, the floret and the blend together. */
  readonly octaves: number;
  /** Period of the finest octave — the curd, and what `canopyGrainAcross` names. */
  readonly finePeriod_m: number;
  /** Radius of one floret of the finest octave. */
  readonly fineFloretRadius_m: number;
  /** How far the closure core is inset under the crown's surface. */
  readonly relief_m: number;
}

/**
 * What the crown's surface is made of, from the one count the form authors.
 *
 * `canopyGrainAcross` names the *finest* octave, because that is the scale the
 * reference is read at — count the florets along the widest part of the crown —
 * and the coarse end then falls out of the octave count rather than being
 * authored beside it. Every length here is a period times a fixed ratio and
 * every period is a power of two apart, so the whole packing is one number and
 * the octave count, and the neighbourhood bound the trace needs is scale-free
 * and holds for all of them at once.
 */
export function bonsaiCanopyGrain(form: BonsaiForm): BonsaiCanopyGrain {
  if (!Number.isInteger(form.canopyGrainAcross) || form.canopyGrainAcross < 4) {
    throw new RangeError("Bonsai canopy grain must be an integer of at least four florets across");
  }
  const finePeriod_m = 2 * form.crownRadius_m[0] / form.canopyGrainAcross;
  const latticePeriod_m = finePeriod_m * 2 ** (CANOPY_LATTICE_OCTAVES - 1);
  const floretRadius_m = CANOPY_FLORET_PERIOD_SHARE * latticePeriod_m;
  return {
    latticePeriod_m,
    floretRadius_m,
    blendRadius_m: CANOPY_FLORET_BLEND_SHARE * floretRadius_m,
    octaves: CANOPY_LATTICE_OCTAVES,
    finePeriod_m,
    fineFloretRadius_m: CANOPY_FLORET_PERIOD_SHARE * finePeriod_m,
    relief_m: CANOPY_RELIEF_PERIODS * latticePeriod_m,
  };
}

/**
 * The largest share of a segment's own length its radius may change over.
 *
 * A round cone whose taper outruns its length is one ball inside the other, and
 * the closed-form sweep distance degenerates on it — the render ABI rejects the
 * record rather than drawing it, which is a *world build failure*, not a bad
 * frame. Nine tenths keeps the flare as steep as it can honestly be and leaves
 * the f32 round trip room.
 */
const SWEEP_MAXIMUM_TAPER_SHARE = 0.9;

/**
 * A station list whose taper never outruns its own run.
 *
 * Discharged here rather than at each call site, because the constraint couples
 * two things a generator naturally authors apart: the *profile*, which wants to
 * flare hard at the foot, and the *stations*, which are spaced by whatever curve
 * the run follows. A base whose lower stations are far apart and whose upper
 * ones are close spends the same radius change over a tenth of the length, and
 * the failure lands at GPU initialisation with a message about segment 1 rather
 * than anywhere near the numbers that caused it.
 *
 * Forward only, and toward the previous radius: every run here tapers
 * monotonically from a thick end, so pulling a station back toward its
 * predecessor keeps the profile and only softens the steepest part of it.
 */
function taperedWithinItsRun(stations: readonly SweptTubeStation[]): SweptTubeStation[] {
  const out: SweptTubeStation[] = [{ ...stations[0] }];
  for (let index = 1; index < stations.length; index += 1) {
    const previous = out[index - 1];
    const station = stations[index];
    const span_m = Math.hypot(
      station.at_m.x - previous.at_m.x,
      station.at_m.y - previous.at_m.y,
      station.at_m.z - previous.at_m.z,
    );
    const allowed_m = SWEEP_MAXIMUM_TAPER_SHARE * span_m;
    const change_m = station.radius_m - previous.radius_m;
    out.push({
      at_m: station.at_m,
      radius_m: Math.abs(change_m) <= allowed_m
        ? station.radius_m
        : previous.radius_m + Math.sign(change_m) * allowed_m,
    });
  }
  return out;
}

/** Thomsen's approximation, within about one percent over these aspect ratios. */
function ellipsoidSurfaceArea(radius: Vec3): number {
  const p = 1.6075;
  const mean = (radius.x ** p * radius.y ** p + radius.x ** p * radius.z ** p + radius.y ** p * radius.z ** p) / 3;
  return 4 * Math.PI * mean ** (1 / p);
}

/** Integer avalanche hash in [0, 1). The same tree on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

/** Hash mapped to [-1, 1). */
const hashSigned = (n: number): number => 2 * hash01(n) - 1;

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * The narrow band every white surface in these sets is authored in.
 *
 * The floor used to be 0.62, from a time when the scene shaded each closure's
 * own colour grain and the ramps below had room to run down into shadow. Every
 * surface in this set now renders through one flat plaster closure with the key
 * light raised so white reads as white, and 0.62 under that reads as grey
 * plastic — so the floor is where the specimen still reads as the fired
 * porcelain it is. The ceiling is a guard rather than a look: a linear albedo is
 * a reflectance, `canonicalSvoMaterialRecord` throws above one, and nothing here
 * has any business being near it.
 */
const clampValue = (value: number): number => Math.min(.94, Math.max(.78, value));

const add = (a: Vec3, b: Vec3): Vec3 => V(a.x + b.x, a.y + b.y, a.z + b.z);
const sub = (a: Vec3, b: Vec3): Vec3 => V(a.x - b.x, a.y - b.y, a.z - b.z);
const scale = (a: Vec3, k: number): Vec3 => V(a.x * k, a.y * k, a.z * k);
const length = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);

function normalize(a: Vec3): Vec3 {
  const magnitude = length(a);
  if (!(magnitude > 0)) throw new Error("Bonsai direction must have nonzero length");
  return scale(a, 1 / magnitude);
}

/**
 * The rotation taking local +X, +Y, +Z onto an orthonormal `u, v, w`.
 *
 * `alongAxis` is not enough for anything triaxial, and the header says why: it
 * returns the shortest-arc rotation onto one direction and leaves the roll about
 * it unspecified, so a fin oriented that way would stand on edge at one azimuth
 * and lie flat at the next. A buttress is a *plate* — long along its run, tall,
 * thin across — which is three different half-axes and therefore needs all three
 * axes named. Standard branch-on-largest-diagonal construction, which is the
 * numerically stable one when the trace is negative.
 */
function basisOrientation(u: Vec3, v: Vec3, w: Vec3): { w: number; x: number; y: number; z: number } {
  const trace = u.x + v.y + w.z;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    return { w: .25 * s, x: (v.z - w.y) / s, y: (w.x - u.z) / s, z: (u.y - v.x) / s };
  }
  if (u.x > v.y && u.x > w.z) {
    const s = Math.sqrt(1 + u.x - v.y - w.z) * 2;
    return { w: (v.z - w.y) / s, x: .25 * s, y: (v.x + u.y) / s, z: (w.x + u.z) / s };
  }
  if (v.y > w.z) {
    const s = Math.sqrt(1 + v.y - u.x - w.z) * 2;
    return { w: (w.x - u.z) / s, x: (v.x + u.y) / s, y: .25 * s, z: (w.y + v.z) / s };
  }
  const s = Math.sqrt(1 + w.z - u.x - v.y) * 2;
  return { w: (u.y - v.x) / s, x: (w.x + u.z) / s, y: (w.y + v.z) / s, z: .25 * s };
}

const cross = (a: Vec3, b: Vec3): Vec3 =>
  V(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);

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
 * How many voxels across one floret of the *published* crown is.
 *
 * Whichever crown the form spells: a cast cap under `florets`, a lattice sphere
 * under `aggregate`. Kept because it is the first thing anyone asks about this
 * object, and because the answer turned out to be that it does not matter —
 * primary visibility for a scenery primitive is a BVH over exact records and
 * never consults the voxel payload, so a floret is silhouetted exactly whether
 * it spans four cells or a quarter of one. What the number does bound is how
 * coarsely the cone tracer's occupancy lights the crown. See the header.
 */
export function bonsaiFloretVoxelSpan(form: BonsaiForm, cellSize_m: number): number {
  if (!(cellSize_m > 0)) throw new RangeError("Bonsai floret voxel span needs a positive cell size");
  const radius_m = form.canopy === "aggregate" ? bonsaiCanopyGrain(form).fineFloretRadius_m : form.floretRadius_m;
  return 2 * radius_m / cellSize_m;
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
  if (!Number.isInteger(rootCount) || rootCount < 3) throw new RangeError("Bonsai needs at least three root buttresses");
  if (!Number.isInteger(lobeCount) || lobeCount < 3) throw new RangeError("Bonsai needs at least three canopy lobes");
  if (!Number.isInteger(bumpsPerLobe) || bumpsPerLobe < 1) throw new RangeError("Bonsai needs at least one nodule per lobe");
  if (!Number.isInteger(floretsPerLobe) || floretsPerLobe < 1) throw new RangeError("Bonsai needs at least one floret per lobe");
  if (!Number.isInteger(spec.limbsPerTrunk) || spec.limbsPerTrunk < 1) throw new RangeError("Bonsai needs at least one limb per trunk");
  if (!(floretRadius > 0)) throw new RangeError("Bonsai floret half-width must be positive");
  if (!(spec.floretFlatten > 0 && spec.floretFlatten <= 1)) throw new RangeError("Bonsai floret flatten must be in (0, 1]");
  if (!(spec.lobeFill > 0)) throw new RangeError("Bonsai lobe fill must be positive");
  if (!(spec.limbWidth > 0 && spec.limbWidth < 1)) throw new RangeError("Bonsai limb width must be under its trunk's");
  if (!(spec.rootWidth > 0)) throw new RangeError("Bonsai buttress width must be positive");
  if (!(spec.rootTaper >= 0 && spec.rootTaper < 1)) throw new RangeError("Bonsai buttress taper must be in [0, 1)");
  // Throws on a bad `canopyGrainAcross` whichever crown is spelled, so a form
  // cannot carry a grain that only fails once someone flips `canopy`.
  const crownGrain = bonsaiCanopyGrain(spec);

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
  // The crown's own extent, kept apart from the specimen's. `floretCoverage` is
  // surface published over surface to cover, and the surface to cover is the
  // *crown* — measuring it off the whole specimen made the ratio a function of
  // how tall the trunk was, so a form with the same crown on a taller trunk
  // scored lower and the number stopped being comparable between forms.
  let crownMinimum = V(Infinity, Infinity, Infinity);
  let crownMaximum = V(-Infinity, -Infinity, -Infinity);
  /** Every emitted primitive reports the box that bounds it, in local metres. */
  const cover = (center: Vec3, radius: Vec3 | number): void => {
    const r = typeof radius === "number" ? V(radius, radius, radius) : radius;
    minimum = V(Math.min(minimum.x, center.x - r.x), Math.min(minimum.y, center.y - r.y), Math.min(minimum.z, center.z - r.z));
    maximum = V(Math.max(maximum.x, center.x + r.x), Math.max(maximum.y, center.y + r.y), Math.max(maximum.z, center.z + r.z));
  };
  /** The same, for a canopy part, so the crown can be measured on its own. */
  const coverCrown = (center: Vec3, radius: Vec3 | number): void => {
    cover(center, radius);
    const r = typeof radius === "number" ? V(radius, radius, radius) : radius;
    crownMinimum = V(Math.min(crownMinimum.x, center.x - r.x), Math.min(crownMinimum.y, center.y - r.y), Math.min(crownMinimum.z, center.z - r.z));
    crownMaximum = V(Math.max(crownMaximum.x, center.x + r.x), Math.max(crownMaximum.y, center.y + r.y), Math.max(crownMaximum.z, center.z + r.z));
  };

  // One fired ceramic surface from the toes to the crown, declared rather than
  // spelled into the tags. See the note on `porcelain` below for what this
  // replaces.
  const bark = (value: number): SceneryMaterial => ({ palette: spec.barkPalette, value: clampValue(value), surface: "ceramic" });
  const canopy = (value: number): SceneryMaterial => ({ palette: spec.canopyPalette, value: clampValue(value), surface: "ceramic" });

  /**
   * Tags, and they are load-bearing twice over.
   *
   * `porcelain` is on every part because that is what the specimen is, and it
   * used to be load-bearing in a second way: the surface closure was picked by
   * matching a regular expression against the group name followed by the tags,
   * so the word had to be spelled and had to stay spelled or a bonsai became
   * wood under foliage. The materials above now declare `surface: "ceramic"`
   * outright, so the tag is a description again rather than an instruction.
   * `tests/bonsai.test.ts` still pins the closure, which is now pinning the
   * declaration instead of the spelling.
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
    (role === "crown" || role === "floret" ? coverCrown : cover)(center, radius);
  };

  /**
   * One continuous run of tapered tube, as the fewest sweep records that carry
   * it — and the reason nothing below the crown is a cone chain any more.
   *
   * A chain of cones with a sphere at every node is *exact*, and it still looked
   * wrong: a hard union is C⁰ but not C¹, so the shading normal steps at every
   * junction and a smooth taper renders as a stack of faceted pipes with a seam
   * between each. `sweptTubeNodes` publishes the whole run as one `tapered-sweep`
   * cluster whose segments are fused by a smooth minimum, which is C¹ through
   * every junction, and it solves its own envelope — see the header of
   * `swept-tube.ts`, which is emphatic that the envelope must not be authored by
   * hand because the containment bound collapses near the ellipsoid's surface.
   *
   * Stations are handed over in the *specimen's own* frame rather than the
   * world's. `sweptTubeNodes` documents itself as emitting at scene root, but
   * what it actually does is put the fitted origin in `place.position` with
   * `units: "metres"`, and `childFrame` composes that against whatever encloses
   * it — so a run authored in local metres inside this generator's group lands
   * where the group is. Nothing in the solver is world-dependent; only the
   * comment is.
   */
  const tube = (
    id: string, authored: readonly SweptTubeStation[], material: SceneryMaterial, role: string,
    envelopeWasteBudget?: number,
  ): void => {
    const stations = taperedWithinItsRun(authored);
    nodes.push(...sweptTubeNodes({
      id: `${key}/${id}`,
      // Explicit, because `sweptTubeNodes` defaults a run's group to its own id.
      // One specimen is one click target; see the note on `partTags`.
      group: key,
      tags: partTags(role),
      material, stations, envelopeWasteBudget,
      seed: (seed ^ 0x51ed_1e5f) >>> 0,
    }));
    // The stations bound the run: a sweep is the union of balls about them, so
    // their own boxes contain it exactly. The fitted *envelope* is much larger
    // and would report a trunk reaching a hand's breadth past what it draws.
    for (const station of stations) cover(station.at_m, station.radius_m);
  };

  // ---- Base, buttress roots and trunks -------------------------------------
  //
  // Read off the reference rather than assumed, and the assumption was wrong
  // three times. This is not a single bole with a skirt, and it is not four
  // poles under a plate either — both of those were built and both read as
  // furniture. It is one sculpted mass at the ground that flares outward into
  // buttress roots and upward into a *few* limbs that taper hard as they climb.
  //
  // Three things carry that and all three are structural rather than tuning.
  // Every run is one swept record, so there is no facet and no seam along it.
  // Every run starts *inside* the mass it leaves, so there is no cap disc where
  // it emerges and no crease that is not a real junction. And the taper is steep
  // enough that nothing has parallel sides: a limb that leaves at four fifths of
  // its parent and arrives at two fifths reads as grown, where one that holds
  // its width reads as pipe.
  const stumpTopY = spec.stumpShare * height;
  const forkY = spec.forkShare * height;
  const rise = forkY - stumpTopY;
  if (!(rise > 0)) throw new RangeError("Bonsai fork share must exceed its stump share");

  // The base mass. Buried at the foot, so no rim shows where it meets ground the
  // terrain has moved a few millimetres since this was authored, and it flares
  // through three stations rather than tapering straight — a cone is a cone at
  // any resolution and this has to read as thrown clay.
  // The stations are spread down into the buried part rather than bunched near
  // the top, because a sweep segment may not taper faster than it runs — see
  // `taperedWithinItsRun`, which enforces that and would otherwise silently
  // flatten this flare into a cylinder.
  const BASE_FOOT_SHARE = 2.15;
  const BASE_WAIST_SHARE = 1.42;
  const BASE_TOP_SHARE = 1.10;
  tube("base", [
    { at_m: V(0, -.85 * boleRadius, 0), radius_m: BASE_FOOT_SHARE * boleRadius },
    { at_m: V(0, -.10 * boleRadius, 0), radius_m: 1.76 * boleRadius },
    { at_m: V(0, .45 * stumpTopY, 0), radius_m: BASE_WAIST_SHARE * boleRadius },
    { at_m: V(0, stumpTopY, 0), radius_m: BASE_TOP_SHARE * boleRadius },
  ], bark(spec.barkValue + .01), "stem", GROUND_ENVELOPE_WASTE_BUDGET);

  // The buttresses. A **fin**, and every decision here is about making a blade
  // rather than a lump.
  //
  // Round sweeps made them tentacles: a sweep is circular in section at every
  // station by construction, so a bundle of them reads as pipes. One `seeded-lobes`
  // envelope apiece made them boulders, and the reason is the eccentricity cap:
  // a cluster's clip is `(|p/R| - 1)·min R`, whose slack goes as the envelope's
  // own longest-over-shortest, so a single envelope spanning the whole run
  // forced the thin axis up to a quarter of that *length* — 0.61 of the fin's
  // height, which is a section near enough square, which is a bread roll.
  //
  // The way past it is the one the crown's rim needed: composition. A fin is
  // several short envelopes chained along its run, each well inside the cap, and
  // their union is longer and thinner than any of them could be alone. With the
  // long axis no longer the largest, the thin axis is set by the *height*
  // instead, and the section comes out one to four — a blade standing on edge.
  //
  // The field is what makes it a root rather than a wedge. Its lobes are placed
  // from the seed inside the envelope by a rule that guarantees the clipping
  // `max` never reaches them, so the silhouette it draws is the lobes' own and
  // *not* the ellipsoid — the property the crown's lattice does not have, and
  // the reason the same trick cannot be used up there. All three placement
  // parameters are fractions of the envelope in its own normalised space, so an
  // envelope this eccentric produces lobes of the same proportions: a fin's
  // lumps are fin-shaped without being told to be.
  const rootFeet_m: Vec3[] = [];
  const ROOT_STEPS = 12;
  for (let index = 0; index < rootCount; index += 1) {
    const angle = 2 * Math.PI * (index + .30 * hashSigned(seed + 17 * index)) / rootCount;
    const outward = bearing(angle);
    const width = spec.rootWidth * (.90 + .20 * hash01(seed + 17 * index + 2));
    // How far this bearing may run before it has left the bank.
    //
    // Terrain, not geometry, and deliberately so: the generator is handed a
    // height query and nothing else, because a specimen has no business knowing
    // what it is standing beside. What it *can* see is that the ground under a
    // bearing has dropped away, which is what the edge of a bank is. On the hero
    // stand the terrace falls 70 mm within 80 mm toward the water and the
    // coping's outer foot is at 100 mm, so a fin that stops where the bank has
    // fallen a bole radius stops on soil; the bearings that run inland never
    // reach the limit at all and are unaffected.
    const wanted = boleRadius * spec.rootReach * (.85 + .30 * hash01(seed + 17 * index + 1));
    const shoulderTall_m = boleRadius * width * BUTTRESS_HEIGHT_SHARE;
    const reach = (() => {
      const fall = ROOT_MAXIMUM_FALL_SHARE * boleRadius;
      for (let step = 1; step <= ROOT_STEPS; step += 1) {
        const radial = wanted * step / ROOT_STEPS;
        if (!(-groundLocal(outward.x * radial, outward.z * radial) > fall)) continue;
        // Back off by the fin's own height at the *toe*, not by a step of the
        // walk. What has to stay on the bank is the envelope, and the last one
        // reaches past its toe by its own longest half-axis — which for a blade
        // is how tall it is there, not how far it runs. Backing off a step left
        // three millimetres of one hanging over the coping; backing off the
        // shoulder's height instead deleted the pond-side roots outright.
        return Math.max(0, wanted * (step - 1) / ROOT_STEPS
          - shoulderTall_m * (1 - spec.rootTaper));
      }
      return wanted;
    })();
    // A bearing with no bank left still gets a buttress, just a short one: the
    // pond side of this specimen genuinely has less ground, and a stub that
    // flares out of the base and stops reads as that. What it must not do is go
    // to zero, which leaves the camera-facing quarter of the base with no root
    // on it at all.
    if (!(reach > BUTTRESS_MINIMUM_REACH_SHARE * boleRadius)) continue;

    // The run: out on `t^0.62`, down on `(1-t)^2`, every station placed on the
    // ground under it, so a buttress follows the bank down as it goes. That last
    // part is the whole reason this generator is handed a height query.
    const at = (t: number): Vec3 => {
      const radial = reach * t ** .62;
      const x = outward.x * radial, z = outward.z * radial;
      return V(x, groundLocal(x, z) + stumpTopY * (1 - t) ** 2 - .30 * boleRadius * t, z);
    };
    // Height along the run, from the shoulder to a knife edge. This is the flare:
    // the fin is at its tallest where it leaves the trunk and thins the whole way
    // out, rather than being fattest in the middle like a leaf.
    const tallAt = (t: number): number => shoulderTall_m * (1 - spec.rootTaper * t ** .7);

    for (let segment = 0; segment < BUTTRESS_SEGMENTS; segment += 1) {
      // Overlapped, because two records meet in a hard union and a gap between
      // them is a hole. A third of a segment either way puts each junction well
      // inside its neighbour, where both blades already agree on their section.
      const t0 = Math.max(0, (segment - BUTTRESS_SEGMENT_OVERLAP) / BUTTRESS_SEGMENTS);
      const t1 = Math.min(1, (segment + 1 + BUTTRESS_SEGMENT_OVERLAP) / BUTTRESS_SEGMENTS);
      const from = at(t0), to = at(t1);
      const span_m = length(sub(to, from));
      if (!(span_m > 1e-4)) continue;
      const along = normalize(sub(to, from));
      const lateral = normalize(cross(V(0, 1, 0), along));
      const upright = cross(along, lateral);
      const centre = scale(add(from, to), .5);
      const long_m = .56 * span_m;
      const tall_m = tallAt(.5 * (t0 + t1));
      // The section. The thin axis is a quarter of whichever of the other two is
      // larger, which is the eccentricity cap holding — and because the chaining
      // keeps `long_m` under `tall_m` for most of the run, what sets it is the
      // height, so the fin is a blade rather than a log.
      const thin_m = Math.max(tall_m, long_m) / BUTTRESS_MAXIMUM_ECCENTRICITY;
      nodes.push({
        kind: "cluster", field: "seeded-lobes", id: `${key}/root-${index}-${segment}`,
        group: key, tags: partTags("root"),
        place: { position: centre, orientation: basisOrientation(along, upright, lateral) },
        lobe: V(long_m, tall_m, thin_m),
        lobeCount: 5 + ((index + segment) % 4),
        anisotropy: BUTTRESS_LOBE_ANISOTROPY,
        // Against the *shortest* half-axis, which is what the ABI measures a
        // blend by: span plus spread plus this must stay under one or the hard
        // `max` slices the lobes it was meant to contain.
        smoothRadius: BUTTRESS_BLEND_SHARE * thin_m,
        seed: (seed ^ (0x85eb_ca6b * (index + 1)) ^ (0x9e37_79b1 * (segment + 1))) >>> 0,
        material: bark(spec.barkValue - .03 + .04 * hash01(seed + 23 * index + segment)),
      });
      // The envelope contains the field by construction, so its own box bounds it.
      cover(centre, V(Math.max(long_m, tall_m), Math.max(long_m, tall_m), Math.max(long_m, tall_m)));
    }
    const toe = at(1);
    rootFeet_m.push(V(standX + toe.x, groundY + toe.y, standZ + toe.z));
  }
  if (rootFeet_m.length < 3) throw new RangeError("Bonsai has no bank to put its buttresses on");

  // No fillet ring. It used to bridge the notches between neighbouring roots,
  // and with fins that is precisely the wrong thing to do: what has to read is
  // the deep V between one blade and the next, running from high on the trunk
  // down to the soil and opening outward. Seven roots with their notches filled
  // came out as a single lumpy collar. Five with air between them read as roots,
  // and the merge that is wanted happens on its own — every fin leaves the axis
  // inside the base's own flare, so they fuse for the first fifth of their run
  // and are separate objects for the rest.

  // The trunks. Radial offset goes as t^1.8 so each leaves the base nearly
  // upright and is splayed by the time it forks: that curve is what makes the
  // group read as one tree opening out, where a straight splay reads as a wigwam
  // and no splay at all reads as table legs, which is what four of these looked
  // like before the count came down and the taper went up.
  //
  // Each starts *below* the base's top, inside the mass, so the fork is where
  // the trunks separate rather than where four discs are stacked on a stump.
  const TRUNK_STATIONS = 5;
  const trunkTops: {
    readonly at: Vec3;
    readonly radius: number;
    /** The trunk's own centreline, so a limb can leave from inside it. */
    readonly path: (t: number) => Vec3;
    readonly radiusAt: (t: number) => number;
  }[] = [];
  for (let index = 0; index < trunkCount; index += 1) {
    const angle = 2 * Math.PI * (index + .34 * hashSigned(seed + 131 * index)) / trunkCount + .4;
    const outward = bearing(angle);
    const splay = boleRadius * spec.trunkSplay * (.80 + .40 * hash01(seed + 131 * index + 1));
    const at = (t: number): Vec3 => add(
      scale(outward, splay * t ** 1.8),
      add(V(0, stumpTopY + rise * t, 0), scale(lean, .22 * rise * t * t)),
    );
    const radius = boleRadius * spec.trunkWidth * (.90 + .25 * hash01(seed + 131 * index + 2));
    const stations: SweptTubeStation[] = [{ at_m: V(0, .55 * stumpTopY, 0), radius_m: radius }];
    for (let step = 0; step <= TRUNK_STATIONS; step += 1) {
      const t = step / TRUNK_STATIONS;
      stations.push({ at_m: at(t), radius_m: radius * (1 - TRUNK_TAPER * t ** .85) });
    }
    // The tight budget: a trunk leaves the base at ground level, where the set
    // beds plants against it. See `GROUND_ENVELOPE_WASTE_BUDGET`.
    tube(`trunk-${index}`, stations,
      bark(spec.barkValue + .02 * hash01(seed + 137 * index)), "stem", GROUND_ENVELOPE_WASTE_BUDGET);
    trunkTops.push({
      at: at(1),
      radius: radius * (1 - TRUNK_TAPER),
      path: at,
      radiusAt: (t: number) => radius * (1 - TRUNK_TAPER * t ** .85),
    });
  }
  // ---- Canopy lobes -------------------------------------------------------
  // The crown is one plate of clumped heads. Centres go on a sunflower disc,
  // sizes come from the swell, and the two are coupled in the one way that
  // matters: each head is inset by *its own* plan radius rather than the nominal
  // one, so however far the swell takes it, its surface reaches `crownRadius_m`
  // and never past it. Insetting by the nominal radius let the swelliest rim
  // head overrun the authored crown by a fifth, and whether it did depended on
  // which azimuth the seed happened to put it at — a silhouette that is correct
  // by luck is one nobody can re-tune.
  const crownCenter = add(scale(lean, overhang), V(0, height - .5 * crownThickness, 0));
  // A *share* of each crown radius rather than a radius, so a mass has the
  // crown's own aspect and the inset below bites on both axes. It used to be
  // `sqrt(Rx*Rz/n)`, one isotropic number for an elliptical crown, and on a
  // crown as eccentric as this one that let a mass sized off the long axis
  // overrun the short one by a tenth — past the silhouette the form authored,
  // in the one direction nobody was measuring.
  //
  // `1/sqrt(n)` is the share at which n masses pave the crown's plan area, so
  // `lobeFill` at one is a paving and not a covering — equal ellipses always
  // leave the gaps between ellipses. Above one is the overlap that closes them,
  // which is also the overlap that makes the crown one fused mass with clefts in
  // it rather than a heap of separate balls.
  const lobePlanShare = spec.lobeFill / Math.sqrt(lobeCount);
  // `crownThickness_m` is the crown's own surface either way, because that is
  // the number the reference is read for — and the two canopy modes reach it by
  // solving a different equation. Under `florets` a nodule stands proud of the
  // lobe and a floret proud of that, so the lobe underneath is what is left of
  // the plate after both. Under `aggregate` the cluster's envelope *is* the
  // surface and the closure core is inset under it, so the plate is exactly one
  // lobe thick and there is nothing to subtract.
  const lobeHalfThickness = spec.canopy === "aggregate"
    ? .5 * crownThickness
    : Math.max(.18 * crownThickness, (.5 * crownThickness - .38 * floretRadius) / 1.495);
  // A lobe is a squashed ball, so the area to cover is dominated by its two
  // faces: n nodules of radius r cover it once over when n pi r^2 = 2 pi a^2,
  // and the factor above one is the overlap that fuses them. Capped against the
  // plate's own half-thickness for the reason `bumpsPerLobe` states.
  const lobePlanRadius = lobePlanShare * Math.sqrt(spec.crownRadius_m[0] * spec.crownRadius_m[1]);
  const bumpRadius = Math.min(1.1 * lobeHalfThickness, 1.14 * lobePlanRadius * Math.sqrt(2 / bumpsPerLobe));
  // The plane the heads hang from. Not the mid-plane: a head is placed by its
  // *top* and grows downward, so the swell that gives the crown its clumped
  // masses thickens the underside instead of doming the top — and a flat top
  // over a rounded underside is the one thing the reference is unambiguous
  // about. The droop then takes the rim down with the square of the radius.
  const crownTopY = crownCenter.y + .5 * crownThickness;
  const lobes: BonsaiLobe[] = [];
  for (let index = 0; index < lobeCount; index += 1) {
    // Radius along the sunflower; see `CROWN_RIM_CROWDING` for why the exponent
    // is a third rather than the equal-area half. The endpoints are exact: the
    // first mass sits on the crown's own axis, so the plate has full thickness
    // where it is thickest, and the last sits on the rim, so the silhouette
    // lands on the authored width instead of a few per cent inside it.
    const u = (index / (lobeCount - 1)) ** CROWN_RIM_CROWDING;
    const angle = GOLDEN_ANGLE * index + 1.31 * hash01(seed + 5);
    const swell = 1 - .5 * spec.lobeSwell + spec.lobeSwell * hash01(seed + 43 * index + 1);
    // Clamped at the crown's own radius, and the clamp is the bound rather than
    // a tidy-up: the inset below subtracts the mass's radius from the crown's,
    // so a mass the swell had pushed past the crown would inset to zero, sit on
    // the axis and still draw past the authored silhouette. `lobeFill` is set so
    // this is slack at every seed; it is here so it cannot stop being.
    const radius = V(
      Math.min(spec.crownRadius_m[0], spec.crownRadius_m[0] * lobePlanShare * swell * (1 + .10 * hashSigned(seed + 43 * index + 2))),
      lobeHalfThickness * (.80 + .34 * hash01(seed + 43 * index + 3)),
      Math.min(spec.crownRadius_m[1], spec.crownRadius_m[1] * lobePlanShare * swell * (1 + .10 * hashSigned(seed + 43 * index + 4))),
    );
    lobes.push({
      center_m: V(
        crownCenter.x + Math.max(0, spec.crownRadius_m[0] - radius.x) * u * Math.cos(angle),
        crownTopY - radius.y
          - spec.crownDroop * crownThickness * u * u
          + .07 * crownThickness * hashSigned(seed + 43 * index),
        crownCenter.z + Math.max(0, spec.crownRadius_m[1] - radius.z) * u * Math.sin(angle),
      ),
      radius_m: radius,
    });
  }

  // ---- Limbs --------------------------------------------------------------
  // Each limb ends inside the crown mass it holds up, because a canopy resting
  // on nothing is the most obvious way a generated tree reads as generated. It
  // starts *inside* its trunk rather than on it, which is what a fork looks like
  // and what removes the near-end cap a chain used to need a sphere to hide.
  //
  // Targets come off the crown's own plan rather than off the list of masses.
  // They used to be one lobe apiece and capped at the lobe count, which was fine
  // when the crown was eighteen heads and nonsense once it became three plates:
  // a tree with three limbs under a metre of canopy. A sunflower inside the
  // crown's own radius gives as many spread targets as there are limbs, and puts
  // every one of them well inside the mass it holds up.
  const LIMB_STATIONS = 5;
  const limbCount = spec.limbsPerTrunk * trunkCount;
  for (let index = 0; index < limbCount; index += 1) {
    const reach = LIMB_TARGET_SHARE * Math.sqrt((index + .5) / limbCount);
    const spin = GOLDEN_ANGLE * index + 2.7 * hash01(seed + 53);
    const target = add(crownCenter, V(
      spec.crownRadius_m[0] * reach * Math.cos(spin),
      -.10 * crownThickness,
      spec.crownRadius_m[1] * reach * Math.sin(spin),
    ));
    const trunk = trunkTops[index % trunkCount];
    // On the trunk's own centreline, not below its top.
    //
    // This was the open-ended tube in the base: dropping in y from `trunk.at`
    // walks off the centreline, because a trunk splays as it climbs and its axis
    // at a lower height is somewhere else entirely. The limb's first station
    // landed *outside* its parent and a sweep caps both ends, so what showed was
    // the cap — a ring with a dark interior, exactly where the fork should be.
    // Sampling the trunk's own path puts the station inside the trunk by
    // construction, at whatever height the fork is authored at.
    const forkAt = LIMB_FORK_SHARE * (.86 + .14 * hash01(seed + 59 * index));
    const start = trunk.path(forkAt);
    const delta = sub(target, start);
    const outward = normalize(V(delta.x, 0, delta.z));
    const bow = scale(outward, spec.limbBow * Math.hypot(delta.x, delta.z) * (.8 + .4 * hash01(seed + 59 * index + 1)));
    const at = (t: number): Vec3 => add(add(start, scale(delta, t)), scale(bow, Math.sin(Math.PI * t)));
    const baseRadius = trunk.radius * spec.limbWidth * (.88 + .24 * hash01(seed + 59 * index + 2));
    const stations: SweptTubeStation[] = [];
    for (let step = 0; step <= LIMB_STATIONS; step += 1) {
      const t = step / LIMB_STATIONS;
      // The first station takes the *trunk's* radius rather than the limb's, so
      // the cap it is capped with is inside the trunk's own section however the
      // seed sized the limb.
      stations.push({
        at_m: at(t),
        radius_m: step === 0
          ? Math.max(baseRadius, trunk.radiusAt(forkAt))
          : baseRadius * (1 - LIMB_TAPER * t ** .85),
      });
    }
    tube(`limb-${index}`, stations,
      bark(spec.barkValue - .01 + .02 * hash01(seed + 61 * index)), "stem");
  }

  // ---- Canopy ---------------------------------------------------------------
  // Two aggregate records per lobe, or three scales of explicit primitive. See
  // `BonsaiForm.canopy` for why both exist and what each costs.
  let floretCapArea = 0;
  if (spec.canopy === "aggregate") {
    for (let index = 0; index < lobeCount; index += 1) {
      const lobe = lobes[index];
      // The closure core, and it is not an optimisation — it is what lets the
      // packing above it be granular at all.
      //
      // A lattice stops letting rays through only at `r >= P*sqrt(3)/2`, where
      // every cell corner is inside a sphere; the previous build reached that by
      // authoring `r = 0.95 P`, and paid for it with a crown whose every lobe
      // drew as the bare ellipsoid the cluster was published to avoid, because
      // the envelope's hard `max` slices a solid packing off flush. At the ratio
      // the florets are actually distinct at, a cubic lattice has open corridors
      // straight down its three axes and the crown shows sky through them. So
      // closure is a separate, exact record: one ellipsoid a granulation-depth
      // in from the surface, which no ray gets past and which the florets stand
      // proud of. Exactly the job the explicit build spent its lobe cores on,
      // and it also fails safe — if a march ever runs out of iterations on the
      // packing, what shows through is a smooth crown rather than a hole.
      const core = V(
        Math.max(1e-4, lobe.radius_m.x - crownGrain.relief_m),
        Math.max(1e-4, lobe.radius_m.y - crownGrain.relief_m),
        Math.max(1e-4, lobe.radius_m.z - crownGrain.relief_m),
      );
      nodes.push({
        kind: "ellipsoid", id: `${key}/lobe-${index}-core`, tags: partTags("crown"),
        place: { position: lobe.center_m }, radius: core,
        // The crevice value. Only the deepest tenth of the crown is core, and
        // that tenth is what the eye reads the granulation's depth from, so it
        // sits under the florets rather than beside them.
        material: canopy(spec.canopyValue - .10 + .03 * hash01(seed + 73 * index)),
      });
      coverCrown(lobe.center_m, core);
      // The granulation, all three scales of it in one record. The plate *is*
      // the envelope here — the packing is clipped to exactly it — so the mass's
      // own silhouette is the one the layout placed, with no proud term to add
      // back, and the grain wraps the whole surface including the underside
      // rather than stopping where a cast field would have run out of florets.
      nodes.push({
        kind: "cluster", id: `${key}/lobe-${index}`, tags: partTags("crown"),
        place: { position: lobe.center_m },
        lobe: lobe.radius_m,
        floretRadius: crownGrain.floretRadius_m,
        latticePeriod: crownGrain.latticePeriod_m,
        octaves: crownGrain.octaves,
        jitter: CANOPY_JITTER,
        smoothRadius: crownGrain.blendRadius_m,
        seed: (seed ^ (0x9e37_79b1 * (index + 1))) >>> 0,
        material: canopy(spec.canopyValue - .02 + .03 * hash01(seed + 71 * index)),
      });
      coverCrown(lobe.center_m, lobe.radius_m);
      // The same ratio the explicit build reports, with the head standing in for
      // the floret: how many times over the crown's envelope is covered by the
      // surfaces published on it. For an aggregate that is the heads' own
      // overlap, which is the property that decides whether the crown is one
      // fused mass with clefts in it or a heap of separate balls.
      floretCapArea += ellipsoidSurfaceArea(lobe.radius_m);
    }
  } else {
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
      material: canopy(spec.canopyValue - .09 + .03 * hash01(seed + 71 * index)),
    });
    coverCrown(lobe.center_m, lobe.radius_m);
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
      sphere(`bump-${index}-${bump}`, center, radius, canopy(spec.canopyValue - .11 + .09 * rise), "crown");
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
        material: canopy(spec.canopyValue - .04 + .06 * lift + .02 * hashSigned(seed + 109 * index + floret)),
      });
      coverCrown(at, wide);
      floretCapArea += Math.PI * wide * wide;
    }
  }
  }

  const leafCount = nodes.length;
  if (leafCount > spec.maximumLeaves) {
    throw new RangeError(`Bonsai published ${leafCount} primitives, over its ${spec.maximumLeaves}-leaf budget`);
  }

  // The envelope the canopy had to cover, as the ellipsoid of the crown's own
  // measured extent — not the specimen's, which would make the ratio a function
  // of the trunk. Approximate on purpose: it is a ratio to be compared between
  // forms, not a surface integral.
  const half = V(
    .5 * (crownMaximum.x - crownMinimum.x),
    .5 * (crownMaximum.y - crownMinimum.y),
    .5 * (crownMaximum.z - crownMinimum.z),
  );
  const envelopeArea = ellipsoidSurfaceArea(half);

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
    crownBounds_m: {
      min: V(standX + crownMinimum.x, groundY + crownMinimum.y, standZ + crownMinimum.z),
      max: V(standX + crownMaximum.x, groundY + crownMaximum.y, standZ + crownMaximum.z),
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
