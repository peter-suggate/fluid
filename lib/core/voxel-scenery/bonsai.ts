import type { Vec3 } from "../model";
import type { SceneryMaterial, SceneryNode } from "../scenery-graph";
import {
  SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY,
  SVO_CLUSTER_LOBE_MAXIMUM_COUNT,
  SVO_CLUSTER_LOBE_MINIMUM_COUNT,
} from "../../svo/svo-cluster-limits";
import { bonsaiCanopyField, bonsaiCanopyPadProgram } from "./bonsai-canopy-field";
import { bonsaiCanopyPads } from "./bonsai-canopy-pads";
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
 * here means the canopy heads and the bed under them, are free to be triaxial
 * because their axes are the world's — and the same argument is why every canopy
 * `cluster` is placed unrotated: its envelope is the crown's own plate. The runs
 * below the crown are the exception and carry a full orientation, because
 * `sweptTubeNodes` fits a frame to each one; anything measuring a station has to
 * rotate it, which `tests/bonsai.test.ts` does and had to learn to.
 *
 * ## What actually limits the canopy: the leaf, not the record
 *
 * This section used to say the opposite, and the sentence it said was measured:
 * "rendering this canopy at 25 mm, 50 mm and 100 mm cells produced three
 * byte-identical frames", because primary visibility went through
 * `traceScenePrimitives` — a BVH over the authored records solved exactly — and
 * never consulted the voxel payload. **Analytic shading is gone.** The primary
 * ray now returns the voxel cell's surface, so the leaf size is the geometric
 * resolution of every scenery primitive, and the cell size that used to change
 * nothing now changes everything.
 *
 * The law that follows is the whole reason this module was rebuilt, and it is
 * project-wide rather than local to the bonsai:
 *
 * > A feature whose period is under about **two leaves** does not render as that
 * > feature; it renders as aliasing. Above about **three leaves** it renders as
 * > geometry.
 *
 * At the hero garden's `HERO_GARDEN_CELL_M = 0.025` that is a floor of 50 mm to
 * exist at all and 75 mm to be legible, on a crown 720 mm wide. Which is a
 * brutal constraint, and it is what the previous crown broke on every scale it
 * had: the lattice's three octaves stood at 45 mm, 22.5 mm and 11.25 mm — 1.8,
 * 0.9 and 0.45 leaves. Not one of them could render.
 *
 * The frame that came back said so precisely. Casting 3 000 rays at a hero lobe
 * through the ABI's own CPU tracer, and reporting how far under the envelope the
 * drawn surface sits as percentiles rather than a mean:
 *
 *                              p50     p90     p99   core visible
 *   9 heads, P45, r/P 0.38   0.0 mm  16.2 mm  50.4 mm     1.2%
 *
 * Half the crown is *exactly* the envelope — flat, because three octaves at that
 * ratio pack to a solid and the hard `max` slices it off flush — while the
 * deepest hundredth is two leaves down a well whose bottom is the smooth closure
 * core. Flat where it should be lumpy and pitted where it should be smooth,
 * which in the frame is bright ridges around near-black hard-edged pits: the
 * relief inverted. No tuning of the ratio moves both ends of that distribution
 * at once, because they are the same construction seen from two sides.
 *
 * The record ceiling is still real — `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES`
 * is a cliff, not a slope, and above it `buildSvoPrimitiveCandidates` is simply
 * not called and the whole set stops being drawn — but it is no longer the
 * binding constraint on this object. The leaf is.
 *
 * ## Which is why the crown is one seeded-lobes record a head
 *
 * A lattice cluster cannot be both granular and closed. It only stops letting
 * rays through at `r >= period * sqrt(3)/2`, and at that ratio the packing is a
 * solid that the envelope's hard `max` cuts off flush, so the head draws as the
 * bare ellipsoid the cluster existed to avoid. That is why the previous build
 * spent a second record per head on an ellipsoid *closure core* one granulation
 * depth beneath — and that core is what the black pits are the bottom of.
 *
 * `seeded-lobes` has neither problem. It is a smooth union of solid ellipsoids
 * placed inside the envelope by a rule that guarantees the clipping `max` never
 * reaches any of them, so its silhouette is *the lobes' own* and it is closed by
 * construction. Measured the same way, on a 195 x 125 x 168 mm head:
 *
 *   lobes  span  displacement |  p50     p90    rays through
 *      8   0.40      0.85     | 17.8 mm 28.4 mm     0.0%
 *      8   0.48      0.85     | 14.3 mm 21.7 mm     0.0%
 *     12   0.56      0.85     | 12.6 mm 17.9 mm     0.0%
 *
 * Zero, at every setting tried, and a *tight* distribution — the surface sits
 * about a leaf under the envelope and stays there, with no tail. There is no
 * well for a core to sit at the bottom of because there is no core: one record a
 * head instead of two, and the record freed is what pays for the heads the crown
 * was missing.
 *
 * What `seeded-lobes` gives up is the octave stack, so the sub-head scales come
 * from the head *layout* rather than from inside one record. See
 * {@link bonsaiCanopyLadder}.
 *
 * ## The tape this crown wants, and why it is not written here
 *
 * `lib/svo-field-program.ts` now carries `scatter` (op 7, code 13) — domain
 * repetition with per-cell jitter — and its own row says what it is for: "the
 * florets; an occupant that is itself a scatter is self-similarity for one op".
 * That is exactly this object. The whole crown is three ops:
 *
 *   scatter(period = head spacing, jitter)     // the heads
 *     scatter(period = floret spacing, jitter) // the florets in each head
 *       ellipsoid(floret radii)                // one floret
 *
 * clipped to the plate, with recursion depth equal to the number of scatter ops
 * on the chain — so a third scatter is the sub-floret grain and costs one op and
 * no records at all. Thirty records for the crown become **one**, the head
 * layout stops being a sunflower of envelopes and becomes a lattice the shader
 * evaluates, and the bed stops being needed because a scatter has no envelope to
 * leave holes between.
 *
 * It is not written here for one reason, and it is the same wall
 * `voxel-scenery/stone-set.ts` and `voxel-scenery/swept-coping.ts` both document:
 * **`SceneryNode` has no `field-program` member.** The graph runs box, sphere,
 * ellipsoid, cone, cluster and group, and `scenery-expand.ts` has no arm that
 * could publish a tape, so a generator cannot author one however much the record
 * would like to exist. The kind exists in `svo-primitive-kinds.ts` (code 12) and
 * the tape assembles and evaluates; the authoring path is what is missing.
 *
 * **This is now the *whole* wall, and it is the same one for all three
 * species.** The op set is no longer a second blocker anywhere — `scatter` is
 * landed and so is everything the stone grain wanted — and the leaf is no longer
 * a third. That last one is worth stating in numbers, because it is what makes
 * this the next thing to build rather than a note: the crown's ladder converges
 * on the plate's own 30.8 mm floret at a 6.25 mm leaf and then **stops moving**,
 * because a third scale is all a `seeded-lobes` record has. The plate has a
 * fourth — the 5-8 plate-pixel sub-floret grain `canopyGrainAcross` documents
 * having once been misread as the floret, about 7 mm on this crown. At 0.78 mm
 * that grain is nine leaves across and squarely drawable. A third `scatter` on
 * the chain above is one op and no records; there is no other construction in
 * this file that could reach it.
 *
 * The shape of this module is chosen so that arriving is a small change rather
 * than a rewrite. Everything the tape needs is already derived rather than
 * authored and already in one place: {@link bonsaiCanopyLadder} solves the head
 * period, the floret period and the jitter from the crown and the leaf, and the
 * emission below reads nothing else. Adopting `scatter` replaces the two loops
 * that publish the bed and the heads with one node built from the same ladder,
 * and deletes {@link CANOPY_BED_HEADS_PER_RECORD} and its friends. Nothing about
 * the layout, the trunk or the tests moves.
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
   * Canopy heads — the clumped masses the crown reads as, one floret cluster
   * apiece, and **the scale the whole object is judged on**.
   *
   * They are flattened balls laid out in a plane, and the crown is flat because
   * of the arrangement as much as because of the flattening, so this and
   * `crownThickness_m` are not independent: the plate is exactly one mass thick,
   * and a count that disagrees reads as a lumpy dinner plate however the rest is
   * tuned.
   *
   * The count is set by the head *width* it produces, which is
   * `2 * crownRadius_m[0] * lobeFill / sqrt(lobes)` — see
   * {@link bonsaiCanopyLadder} for the two numbers that width is caught between.
   * Nine heads, which is what this was, put the width at 331 mm on the hero
   * crown: 46% of the crown, so the union of them is one disc with no head
   * distinguishable in it, which is the lumpy dinner plate exactly. Twenty-four
   * puts it at 176 mm, 24% of the crown and seven leaves across.
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
   *
   * It costs nothing in *grain* either, and that took a fix. A head's florets
   * are a fraction of its own envelope, so one nominal `lobeSpan` across a
   * 1.6 : 1 spread of head sizes would give the widest head florets 1.6 times the
   * narrowest head's — and a floret is a length the leaf decides rather than a
   * proportion of whichever head it fell in. The span is re-solved per head; see
   * the canopy emission.
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
   * Florets across the crown's full width **at the reference's own proportion**,
   * before the leaf-size floor is applied.
   *
   * The granulation, as the number the plate is actually read for: count the
   * florets along the widest part of the silhouette. Measured off
   * `artifacts/plate-crops/canopy.png` at its native 1672 x 940 resolution, one
   * floret spans 33 to 50 plate pixels against a canopy spanning 890 of them —
   * 0.037 to 0.056 of the width, so twenty-four across. That is the *authored*
   * ambition and it is deliberately finer than the leaf can currently draw:
   * {@link bonsaiCanopyLadder} raises the published floret to the legibility
   * floor and lets this number take over as the leaf shrinks, so the specimen
   * converges on the plate's own proportions rather than being re-authored.
   *
   * This used to be 64, and 64 is where the reading went wrong: it counted the
   * plate's *sub-floret grain* — 5 to 8 plate pixels, about 7 mm on this crown —
   * as the floret, which put every published scale between a quarter of a leaf
   * and two leaves, which is the aliasing band from end to end.
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
   * `aggregate` publishes **one** `seeded-lobes` record per head: a smooth union
   * of solid ellipsoids placed inside the head's envelope, evaluated in the
   * shader, closed by construction and silhouetted on its own lobes rather than
   * on the envelope. Twenty-four records rather than eleven hundred, and a
   * floret whose size follows the leaf. `floretsPerLobe`, `bumpsPerLobe`,
   * `floretRadius_m`, `floretFlatten` and `floretSeat` are all unused under it;
   * `lobes`, `lobeFill` and `canopyGrainAcross` are what set the surface.
   *
   * It published *two* records a head until the leaf became the shading
   * resolution — a lattice cluster plus an ellipsoid closure core under it —
   * and the core is what the frame's black pits were the bottom of. See the
   * header.
   */
  readonly canopy: "florets" | "aggregate" | "field";
  /**
   * Stump radius. Every length below the crown is a multiple of it — the flare
   * at its foot, the buttresses that leave it and the trunks that rise from it —
   * so this is the one number that says how heavy the base reads against the
   * crown over it.
   *
   * Caught between the plate and the leaf, and the two disagree. Measured off
   * `artifacts/plate-crops/canopy.png`, the plate's trunk is 34 mm across at the
   * ground and its limbs 22 to 30 mm, against a canopy 720 mm wide — a trunk
   * 1.4 leaves thick and limbs under 1.2, none of which can be drawn at all once
   * the leaf is the shading resolution. So every run below the crown is authored
   * to the legibility floor of three leaves rather than to the plate, and the
   * coral read is bought back from the *topology* instead: five trunks rather
   * than three, near-zero taper along a run, and the narrowing put at the forks
   * where a coral's actually is. See {@link TRUNK_TAPER}.
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
  /**
   * The leaf the specimen will be drawn at, in metres. Absent takes
   * {@link BONSAI_DEFAULT_LEAF_SIZE_M}.
   *
   * Placement rather than form, exactly like `groundHeightAt`: the species has
   * an opinion about proportions and no opinion at all about what resolution a
   * given scene runs. What the leaf decides is which of the specimen's scales
   * may be published at their authored size and which have to be raised to the
   * legibility floor — see {@link bonsaiCanopyLadder}. Hand it the scene's own
   * cell size and the crown converges on the plate's proportions as the leaf
   * shrinks, with no re-authoring anywhere.
   */
  readonly leafSize_m?: number;
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
 * ## Read off the plate, in plate pixels
 *
 * Every proportion below is a measurement of
 * `artifacts/plate-crops/canopy.png` at its native 1672 x 940, against a canopy
 * spanning 890 of those pixels, converted to this crown's 720 mm width:
 *
 *   what              plate px   of the width   on this crown   leaves (25 mm)
 *   canopy               890         1.000          720 mm          28.8
 *   head                130-175      0.15-0.20     108-144 mm       4.3-5.8
 *   floret                33-50      0.037-0.056    27-40 mm        1.1-1.6
 *   sub-floret grain        5-8      0.006-0.009      4-6 mm         0.2
 *   trunk at the ground     42       0.047           34 mm           1.4
 *   limb                   27-37     0.030-0.042     22-30 mm        0.9-1.2
 *
 * So the plate is a four-scale object of which, at a 25 mm leaf, exactly one
 * scale is renderable. That is the whole shape of the problem: the head is the
 * only thing on the list above three leaves, everything under it aliases, and
 * the trunk and limbs are *below* the floret. The form therefore authors the
 * head slightly coarser than the plate rather than slightly finer — a legible
 * head that is 24% of the crown beats a faithful one that is 17% and renders as
 * noise — and hands the finer scales to {@link bonsaiCanopyLadder}, which
 * publishes them at the plate's own proportion the moment the leaf can carry it.
 *
 * ## What that made this
 *
 * The crown is twenty-four heads rather than nine plates. Nine put the head at
 * 46% of the crown, which is not a head at all: the union of nine of those is
 * one disc, and the frame showed a disc. Twenty-four puts it at 176 mm — seven
 * leaves — and the plate is still exactly one head thick, which is what keeps it
 * a cloud layer at 5.5 : 1 rather than a mound.
 *
 * The trunk is five trunks and ten limbs, at three leaves and near-zero taper,
 * because that is what a coral is and it is also the only thing the leaf will
 * draw. The buttresses are six blades rather than five and half again as thick,
 * for the same reason.
 *
 * The record count is the *worst* seed's rather than this one's. Culling is
 * geometric, so the count swings as the heads move, and a form tuned on its own
 * seed overruns the budget on a re-seed — which is what `tests/bonsai.test.ts`
 * found. Every count here is the value at which four hundred seeds all fit.
 */
export const BONSAI_POND_CANOPY: BonsaiForm = Object.freeze({
  height_m: 0.40,
  crownRadius_m: [0.37, 0.32] as const,
  crownThickness_m: 0.135,
  crownOverhang_m: 0.11,
  crownDroop: 0.24,
  lobes: 24,
  lobeFill: 1.20,
  lobeSwell: 0.42,
  bumpsPerLobe: 16,
  floretsPerLobe: 106,
  floretRadius_m: 0.0293,
  canopyGrainAcross: 24,
  floretFlatten: 0.56,
  floretSeat: 0.55,
  floretGrain: 0.52,
  canopy: "field",
  boleRadius_m: 0.054,
  stumpShare: 0.16,
  forkShare: 0.40,
  // Seven, and slender. The plate's base is a *fan* — you can see sky between
  // the stems for most of their run, and that gap is what separates a
  // multi-trunk from a single bole with a lumpy outline.
  trunks: 7,
  // Wider than five trunks wanted, because seven of them have to clear each
  // other before the gaps read at all.
  trunkSplay: 3.9,
  // The measurement, not the caution. `trunkWidth` was 0.86 of a 50 mm stump —
  // an 86 mm trunk against the 34 mm this module's own header measures off the
  // plate, three times over, and the whole group came back as one dark mass with
  // no light between the stems. The 0.86 was set to stop a trunk "arriving at
  // the canopy as a dark stick", and at the 25 mm leaf it had to be: 35 mm is
  // 1.4 voxels there and a taper on top of that is nothing. At the 6.25 mm leaf
  // the scene actually draws, a 38 mm trunk is 6.1 voxels and its 0.22 taper
  // still leaves 4.8 at the fork, so the caution has expired and the plate's own
  // proportion is affordable.
  trunkWidth: 0.35,
  limbWidth: 0.84,
  limbsPerTrunk: 2,
  limbBow: 0.34,
  roots: 5,
  rootReach: 3.2,
  rootWidth: 0.62,
  rootTaper: 0.36,
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
 *
 * `lobes` is inherited for the same reason and with one caveat. The head width
 * is `2 * crownRadius_m[0] * lobeFill / sqrt(lobes)`, so it scales with the
 * crown too — 118 mm here against the pond form's 176 — and stays over the
 * three-leaf floor at a 25 mm leaf. A variant narrower than about 0.30 m has to
 * drop the count instead, which is what {@link BONSAI_SHELF_MINIATURE} does.
 */
export const BONSAI_COURTYARD_STANDARD: BonsaiForm = Object.freeze({
  ...BONSAI_POND_CANOPY,
  height_m: 0.62,
  crownRadius_m: [0.24, 0.22] as const,
  crownThickness_m: 0.17,
  crownOverhang_m: 0.03,
  crownDroop: 0.18,
  lobes: 24,
  bumpsPerLobe: 14,
  floretsPerLobe: 150,
  floretRadius_m: 0.029,
  boleRadius_m: 0.044,
  stumpShare: 0.05,
  forkShare: 0.46,
  trunks: 4,
  trunkSplay: 2.0,
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
 *
 * What it does buy, and what the head count says, is the *leaf*. This crown is a
 * third of the pond form's width and the leaf is whatever the scene runs, so
 * inheriting twenty-four heads would put each at 61 mm — 2.4 leaves at 25 mm,
 * inside the aliasing band. Ten keeps the head at 95 mm and 3.8 leaves. The
 * proportion is worse than the plate's on purpose; see {@link bonsaiCanopyLadder}.
 */
export const BONSAI_SHELF_MINIATURE: BonsaiForm = Object.freeze({
  ...BONSAI_POND_CANOPY,
  height_m: 0.155,
  crownRadius_m: [0.125, 0.115] as const,
  crownThickness_m: 0.055,
  crownOverhang_m: 0.035,
  lobes: 10,
  bumpsPerLobe: 12,
  floretsPerLobe: 176,
  floretRadius_m: 0.0114,
  boleRadius_m: 0.017,
  trunks: 4,
  limbsPerTrunk: 3,
  roots: 6,
});

/**
 * The leaf a specimen is drawn at when nobody says, in metres.
 *
 * `HERO_GARDEN_CELL_M`, which is the scene every one of these numbers was
 * measured against. It is a default rather than a constant of the species: a
 * caller that knows its own cell size should pass it, and everything below then
 * moves with it.
 */
export const BONSAI_DEFAULT_LEAF_SIZE_M = 0.025;

/**
 * Leaves a feature must span before it renders as that feature.
 *
 * The project-wide band law, and the single fact this module is now organised
 * around. Shading is the voxel cell's surface, not an analytic intersection with
 * the record, so a feature whose period is under about two leaves does not draw
 * small — it draws as aliasing. Three is where it stops being marginal and
 * starts being geometry.
 *
 * The three-leaf figure is the one used for everything published, because two is
 * where the previous crown already was: its coarsest lattice octave stood at
 * 45 mm against a 25 mm leaf, which is 1.8, and that octave is exactly what the
 * frame renders as bright ridges around black pits. A floor set at the edge of
 * the band is not a floor.
 */
const LEGIBLE_FEATURE_LEAVES = 3;

/**
 * The plate's own head and floret, as fractions of the crown's full width.
 *
 * Measured off `artifacts/plate-crops/canopy.png` at its native resolution
 * against a canopy 890 plate pixels wide; see {@link BONSAI_POND_CANOPY} for the
 * whole table. These are the sizes the specimen converges on as the leaf
 * shrinks, and they are *not* what is published at a 25 mm leaf — 0.045 of a
 * 720 mm crown is a 32 mm floret, which is 1.3 leaves.
 *
 * `canopyGrainAcross` authors the floret share as a count rather than a
 * fraction, so only the head share lives here.
 */
const PLATE_HEAD_WIDTH_SHARE = 0.175;

/**
 * The crown's scale ladder: what the layout produces, what the plate asks for,
 * and what the leaf will actually draw.
 *
 * Three self-similar scales — crown, head, floret — each the next one's cluster,
 * which is what makes the object a Romanesco rather than a lumpy ball. On the
 * hero specimen, and the middle column is the only one the renderer has an
 * opinion about:
 *
 *   scale    published   leaves   the plate asks for
 *   crown      720 mm     28.8         720 mm
 *   head       176 mm      7.0         126 mm    (0.175 of the crown)
 *   floret      58 mm      2.3          30 mm    (1/24 of the crown)
 *
 * The head is published *coarser* than the plate on purpose. A head at the
 * plate's own 126 mm is 5.0 leaves and would render; at 176 mm it is 7.0 and
 * renders well, and while the leaf is the shading resolution a slightly-too-large
 * head that reads beats a faithful one that is marginal. It costs nothing to
 * take back later — drop `lobeFill`, raise `lobes` — and the count that would do
 * it is already written down: forty-eight heads at fill 1.20 lands on the plate
 * exactly.
 *
 * The floret is published *finer* than legible and coarser than the plate: it is
 * the plate's 30 mm raised to the three-leaf floor, capped at what a head can
 * contain. So it is 2.3 leaves today — inside the aliasing band, and knowingly —
 * because a head has to be a cluster of something, and the alternative is a head
 * with no floret scale in it at all. As the leaf shrinks the floor stops binding
 * and the floret falls to the plate's own 30 mm with nothing re-authored: at
 * 12.5 mm it is 38 mm and 3.0 leaves, at 6 mm it is 30 mm and 5.0 leaves, and
 * the specimen has become the plate.
 *
 * That convergence is the whole future-proofing mechanism and it is deliberately
 * a *derivation*. Authoring 30 mm florets now and letting them alias is the
 * failure this module was rebuilt out of; authoring 58 mm florets now and
 * re-authoring them later is a number that goes stale silently the first time
 * the leaf moves.
 *
 * `headWidth_m` is read out of `lobes` and `lobeFill` rather than solved,
 * because those are what the layout actually uses. `plateHeadWidth_m` beside it
 * is what they were authored to hit, and `tests/bonsai.test.ts` pins the ratio
 * between them — which is what stops the count and the intent drifting apart.
 */
export interface BonsaiCanopyLadder {
  /** The crown's own full width on its long axis, in metres. */
  readonly crownWidth_m: number;
  /** Leaf this ladder was solved for. */
  readonly leafSize_m: number;
  /** Head width the plate's proportion asks for. */
  readonly plateHeadWidth_m: number;
  /** Floret width the plate's proportion asks for, before the floor. */
  readonly plateFloretWidth_m: number;
  /** Head width `heads` and `lobeFill` actually produce, before the swell. */
  readonly headWidth_m: number;
  /** Floret width published inside a nominal head: the plate's, raised to the floor. */
  readonly floretWidth_m: number;
  /** Floret over head on a nominal head — what `lobeSpan` is set from. */
  readonly floretSpan: number;
  /**
   * Heads the crown actually publishes: enough that a head is a floret-cluster
   * rather than an envelope with a few beads rattling in it. See
   * {@link CANOPY_HEAD_FLORET_SPAN}.
   */
  readonly heads: number;
  /**
   * Head groups the heads are dealt into, which is the crown's *third* scale.
   *
   * `form.lobes` is no longer the record count; it is the count of visible
   * clumps, which is what it was always being read for. See the placement loop.
   */
  readonly clumps: number;
  /** Head width in leaves. Under {@link LEGIBLE_FEATURE_LEAVES} is a bug. */
  readonly headLeaves: number;
  /** Floret width in leaves. A number to read, not a number to assert on. */
  readonly floretLeaves: number;
}

/**
 * The span a head is sized to publish its florets at.
 *
 * This is the number the crown was missing, and its absence is worth recording
 * because the failure was silent and got *worse* as the leaf got finer.
 *
 * A `seeded-lobes` record carries at most {@link SVO_CLUSTER_LOBE_MAXIMUM_COUNT}
 * florets — a march cost, twelve rotations per distance evaluation and four of
 * those per normal — so one record spans exactly one octave of scale. The ladder
 * below correctly shrank the floret toward the plate's own 30 mm as the leaf
 * shrank, but the head stayed at the authored `lobes`, so the *span* collapsed
 * and the florets stopped touching each other. Measured on the hero specimen, as
 * the share of a head's envelope its florets occupy:
 *
 *     leaf      head     floret    span    florets fill
 *     25 mm     134 mm    75 mm    0.460      17 %
 *     6.25 mm   134 mm    30 mm    0.225       2 %
 *
 * At two per cent a head is not a surface at all, and what the frame showed was
 * the smooth {@link CANOPY_BED_FILL} bed underneath it — a crown that went
 * *smoother* the finer it was drawn. The module header's promise that "the
 * specimen has become the plate" could never come true, because nothing grew the
 * count to match.
 *
 * So the head is sized off the floret instead of the other way round: at this
 * span seven to twelve florets close a head, which is the regime
 * {@link SEEDED_FLORET_MAXIMUM_SPAN} was measured in. A little under that
 * ceiling, so the per-head re-solve in the emission has room to move without
 * running into the containment budget.
 */
const CANOPY_HEAD_FLORET_SPAN = 0.42;

/**
 * The most heads a crown may publish.
 *
 * A cost bound and a backstop, not a shape: the count is quadratic in the leaf,
 * so a form asking for a plate-fine floret at a millimetre leaf would run away.
 * The hero specimen lands at 161 and stops there, because its floret floors at
 * the plate's own 30 mm and the count stops moving with it.
 */
const CANOPY_HEAD_MAXIMUM_COUNT = 240;

/**
 * How deep a head is against how wide, once the leaf rather than the plate is
 * sizing it.
 *
 * A little under one, because a cauliflower's clumps are pressed against their
 * neighbours rather than free-standing balls — the same argument
 * {@link BonsaiForm.floretFlatten} makes one scale down, and a much gentler
 * version of it, since a head has neighbours on every side and a floret has a
 * surface to lie against.
 */
const CANOPY_HEAD_FLATTEN = 0.85;

export function bonsaiCanopyLadder(form: BonsaiForm, leafSize_m = BONSAI_DEFAULT_LEAF_SIZE_M): BonsaiCanopyLadder {
  if (!Number.isInteger(form.canopyGrainAcross) || form.canopyGrainAcross < 4) {
    throw new RangeError("Bonsai canopy grain must be an integer of at least four florets across");
  }
  if (!Number.isInteger(form.lobes) || form.lobes < 3) throw new RangeError("Bonsai needs at least three canopy heads");
  if (!(form.lobeFill > 0)) throw new RangeError("Bonsai lobe fill must be positive");
  if (!(leafSize_m > 0)) throw new RangeError("Bonsai leaf size must be positive");
  const crownWidth_m = 2 * form.crownRadius_m[0];
  const floor_m = LEGIBLE_FEATURE_LEAVES * leafSize_m;
  const plateFloretWidth_m = crownWidth_m / form.canopyGrainAcross;
  // The floret is solved first and from the leaf alone, because it is the only
  // scale on the crown that has an outside authority: the plate says how wide it
  // is and the leaf says how wide it may be. Everything above it is then sized
  // to contain it, which is the inversion this ladder was missing.
  const floretTarget_m = Math.max(plateFloretWidth_m, floor_m);
  // The head that holds that floret at a span where the florets close it, and
  // then the count that produces that head. The layout's own arithmetic, run
  // backwards: `n` heads of plan share `lobeFill / sqrt(n)` pave the crown, so
  // `n = (crownWidth * lobeFill / headWidth)^2`.
  //
  // `form.lobes` is the floor rather than the answer. A coarse leaf that wants
  // fewer heads than the form authored is a leaf asking for a crown coarser than
  // the composition was built on, and the clump scale below reads the authored
  // count anyway.
  const headTarget_m = floretTarget_m / CANOPY_HEAD_FLORET_SPAN;
  const heads = Math.min(
    CANOPY_HEAD_MAXIMUM_COUNT,
    Math.max(form.lobes, Math.round((crownWidth_m * form.lobeFill / headTarget_m) ** 2)),
  );
  const headWidth_m = crownWidth_m * form.lobeFill / Math.sqrt(heads);
  // Never wider than the head it is a floret of. A specimen small enough that
  // the floor reaches its own head has stopped having two scales, and a floret
  // at its head's width would put the seeded lobes hard against their envelope
  // rather than inside it — which is exactly where the containment argument in
  // `validateSvoClusterPacking` lives, and it throws rather than draws.
  const floretWidth_m = Math.min(SEEDED_FLORET_MAXIMUM_SPAN * headWidth_m, floretTarget_m);
  return {
    crownWidth_m,
    leafSize_m,
    plateHeadWidth_m: PLATE_HEAD_WIDTH_SHARE * crownWidth_m,
    plateFloretWidth_m,
    headWidth_m,
    floretWidth_m,
    floretSpan: floretWidth_m / headWidth_m,
    heads,
    // The authored count, kept as the *clump* scale it was always read as — but
    // never more clumps than there are heads to deal into them.
    clumps: Math.max(3, Math.min(form.lobes, heads)),
    headLeaves: headWidth_m / leafSize_m,
    floretLeaves: floretWidth_m / leafSize_m,
  };
}

/**
 * Heads per bed record, how far the bed is inset under them, and how hard the
 * bed overlaps itself.
 *
 * Four heads a record is what closes the plate for six records on a
 * twenty-four-head crown; the bed is judged on covering the plan and nothing
 * else, so it takes the count that does that and no more. Its own fill is well
 * above the heads' 1.20 for the same reason — a bed lobe is 405 mm across and
 * three of them already overlap through the middle, and the measurement that
 * matters is that the residue is zero rather than small.
 *
 * The inset has to clear a leaf or the heads do not stand proud of the bed at
 * all and the crown draws as one dome; it has to stay well under half the
 * plate's thickness or the bed stops being able to reach the underside.
 *
 * It is **a bit over half a head**, and that is a correction. It used to be a
 * fifth, which was right while a head was 176 mm and half the plate thick: a
 * fifth of it was 35 mm, and a head standing 35 mm out of a 123 mm-tall mass
 * showed a good third of itself. Once the leaf started sizing the head the same
 * fifth became 11 mm under a 55 mm head, so five sixths of every head was buried
 * and what the frame showed was the bed — a smooth flared skirt under a
 * granulated cap, which read as a mushroom rather than as a crown.
 *
 * A share of the head rather than a length, still, so it tracks whatever the
 * leaf deals; what changed is which share. Half is the number that keeps the
 * proportion the fifth used to produce, and a little over it because the heads
 * now overlap each other far more than twenty-four of them did.
 */
const CANOPY_BED_HEADS_PER_RECORD = 4;
const CANOPY_BED_INSET_SHARE = 0.55;
const CANOPY_BED_FILL = 1.55;

/**
 * The largest a floret may be as a fraction of its head.
 *
 * The ABI's containment budget, spent rather than derived: a seeded-lobes
 * packing must keep `span + spanSpread + blend/shortest` under one or the hard
 * `max` slices the lobes it exists to contain. This takes 0.46 of it, leaving
 * {@link SEEDED_FLORET_SPAN_SPREAD} and {@link SEEDED_FLORET_BLEND_SHARE} to
 * fill 0.32 and 0.10, for a margin of 0.12.
 *
 * It also bounds what a floret can *look* like. At 0.46 a head of eight florets
 * is a cluster; at 0.7 the florets are the head and the record draws as one
 * irregular ball. Measured on a 195 x 125 x 168 mm head, 4 000 rays, as the
 * drawn surface's depth under the envelope:
 *
 *   span   0.40    0.48    0.56
 *   p50   17.8 mm 14.3 mm 12.6 mm
 *   p90   28.4 mm 21.7 mm 17.9 mm
 *
 * The spread between the two percentiles is the modulation the eye reads, and it
 * halves from 0.40 to 0.56 — a wider floret fills its head more evenly and the
 * head goes smooth. Rays through: zero at all three, which is the property that
 * removed the closure core.
 */
const SEEDED_FLORET_MAXIMUM_SPAN = 0.46;

/**
 * How much floret size varies within one head, as a fraction of the envelope.
 *
 * A cauliflower head is not a paving of equal florets; a few large ones with
 * smaller ones packed against them is most of what separates it from a
 * raspberry. Spending a third of the containment budget here is what buys that,
 * and it costs nothing — the ABI places every lobe inside the envelope whatever
 * size it drew.
 */
const SEEDED_FLORET_SPAN_SPREAD = 0.32;

/**
 * Smooth-minimum radius that fuses one head's florets, as a fraction of the
 * head's *shortest* half-axis — which is the axis the ABI measures it against.
 *
 * A tenth. A polynomial smooth minimum rounds away every feature under its own
 * radius, and the shortest half-axis of a hero head is 62.5 mm, so this is a
 * 6 mm fillet against a 75 mm floret: enough to fuse the florets into one solid
 * head with soft valleys between them, nowhere near enough to round the valleys
 * away. It is also the last term of the containment budget above.
 */
const SEEDED_FLORET_BLEND_SHARE = 0.10;

/**
 * How far a floret travels from the head's centre, as a share of its own room.
 *
 * High, because the florets have to reach the head's surface — a head whose
 * florets are concentric draws as the innermost one's envelope, which is the
 * bare ellipsoid again by another route. The remaining 0.15 keeps the outermost
 * floret from sitting exactly on the clip.
 */
const SEEDED_FLORET_DISPLACEMENT = 0.85;

/**
 * How much rounder or flatter than its head a floret may be.
 *
 * Against the ABI's ceiling of four. The florets live in the head's *normalised*
 * space, so at one every floret is a scaled copy of the head — the same 1.4 : 1
 * flattening, which regularises the surface — and this is irregularity on top of
 * that rather than instead of it. Two is enough that no two florets present the
 * same curvature and low enough that none of them is thin enough for the march's
 * own lobe bound to bite.
 */
const SEEDED_FLORET_ANISOTROPY = 2.0;
if (SEEDED_FLORET_ANISOTROPY > SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY) {
  throw new RangeError(`Bonsai canopy florets ask for anisotropy ${SEEDED_FLORET_ANISOTROPY}; the render ABI allows ${SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY}`);
}

/**
 * Florets in one head, as a range dealt from the seed.
 *
 * Against the ABI's `SVO_CLUSTER_LOBE_MINIMUM_COUNT` of four and maximum of
 * twelve. Below the minimum the centred lobe dominates and the head is its own
 * envelope; the range is what stops twenty-four heads in a row being the same
 * head at different sizes.
 *
 * The count and the span are not independent, and the span is the one that
 * decides the look: `floretSpan` sets the size and this sets how many of that
 * size there are, so a head with more florets than its span can pave has them
 * overlapping, which is what a cauliflower head is.
 */
const SEEDED_FLORET_COUNT_LOW = 7;
const SEEDED_FLORET_COUNT_RANGE = 5;
if (SEEDED_FLORET_COUNT_LOW < SVO_CLUSTER_LOBE_MINIMUM_COUNT
  || SEEDED_FLORET_COUNT_LOW + SEEDED_FLORET_COUNT_RANGE - 1 > SVO_CLUSTER_LOBE_MAXIMUM_COUNT) {
  throw new RangeError(
    `Bonsai canopy asks for ${SEEDED_FLORET_COUNT_LOW}..${SEEDED_FLORET_COUNT_LOW + SEEDED_FLORET_COUNT_RANGE - 1} florets`
    + ` a head; the render ABI allows ${SVO_CLUSTER_LOBE_MINIMUM_COUNT}..${SVO_CLUSTER_LOBE_MAXIMUM_COUNT}`,
  );
}

/**
 * The ABI's containment condition for a head, discharged once here at the ratios
 * above rather than trusted per call site. Violating it makes the hard `max`
 * slice the florets it exists to contain, which draws as a head with a flat
 * facet on it:
 *
 *   span + spread + blend/shortest  <  1
 *   0.46 + 0.32   + 0.10            =  0.88     margin 0.12
 *
 * Every term is a fraction of the envelope, so it holds for every head width any
 * form may author and at every leaf.
 */
const SEEDED_FLORET_CONTAINMENT =
  SEEDED_FLORET_MAXIMUM_SPAN + SEEDED_FLORET_SPAN_SPREAD + SEEDED_FLORET_BLEND_SHARE;
if (!(SEEDED_FLORET_CONTAINMENT < 1)) {
  throw new RangeError(
    `Bonsai canopy florets overrun their head by ${(SEEDED_FLORET_CONTAINMENT - 1).toFixed(3)} of the envelope`,
  );
}

/**
 * How much of its parent a trunk, and then a limb, has spent by the time it
 * ends.
 *
 * Both were steep — 0.52 and 0.60, a run arriving at half what it left with —
 * and both were wrong twice over.
 *
 * Wrong about the subject first. This is a *coral*, and the plate is
 * unambiguous: `artifacts/plate-crops/canopy.png` shows limbs of very nearly
 * constant section between forks, with all the narrowing at the forks
 * themselves. That is how a branching structure that grows by bifurcation looks,
 * and a trunk that tapers smoothly along its length reads as a turned table leg,
 * which is exactly what the frame showed.
 *
 * Wrong about the renderer second, and this is the binding one. A hero trunk
 * leaves the stump at 86 mm across, which is 3.4 leaves; tapering it by 0.52
 * lands the far end at 41 mm, which is 1.6, and 1.6 leaves is not a thin branch
 * but a dotted line. Under 0.22 the same trunk arrives at 67 mm and 2.7 leaves,
 * and the fork is where the section is allowed to fall.
 */
const TRUNK_TAPER = 0.22;
const LIMB_TAPER = 0.26;

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

/**
 * The same budget for a buttress root, which is looser and has to be.
 *
 * A root is a *curve* — out on `t^0.62` and down on `(1-t)^2` while following
 * whatever the ground does under it — where a trunk is nearly straight, so its
 * best single envelope wastes far more and the splitter cuts far harder at the
 * same budget. Measured on the hero specimen, as the whole specimen's record
 * count against the test the tight budget exists for, which is
 * `tests/hero-layout.test.ts`: it beds air plants and pebbles against this base
 * and fails when an envelope swallows one.
 *
 *   budget      6     8    10    12
 *   records   100    89    85    80
 *   layout    pass  pass  FAIL  FAIL
 *
 * Eight. Eleven records for an envelope nothing in the set is standing inside is
 * not a trade worth making, and ten is already over the cliff — at ten a root's
 * envelope swallows the `plant/air-1` rosette whole and `hero-layout.test.ts`
 * names it. The cliff is where it is because these envelopes sit exactly where
 * the set beds its plants, which is the same reason the trunks have a budget of
 * their own; both are stated as measurements rather than as a rule about ground
 * level, because the number moved when the roots stopped being fins.
 */
const ROOT_ENVELOPE_WASTE_BUDGET = 8;

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
const CROWN_RIM_CROWDING = 0.50;

/**
 * The head's own eccentricity, which is what the seeded-lobes march is bounded
 * by, discharged once here rather than trusted per call site.
 *
 * A cluster is clipped by `(|p/R| - 1)·min R`, a Lipschitz-1 lower bound whose
 * slack goes as the envelope's own longest-over-shortest, so a head `n` times
 * flatter than it is wide understeps by up to `n` and, with
 * `SVO_PRIMITIVE_MARCH_ITERATIONS` fixed at 48, a grazing ray runs out before it
 * arrives — a hole, reported nowhere. The buttress measurement below found four
 * to be the last eccentricity clean at 100% of rays; a hero head is
 * `176 / 125 = 1.4`, so the crown has almost three times that margin, and it has
 * it because the head count went up: nine heads were 331 x 125, which is 2.6.
 */
const CANOPY_HEAD_MAXIMUM_ECCENTRICITY = 4;

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
 * How many leaves across one floret of the *published* crown is.
 *
 * The number this whole module is now organised around, and the one that used to
 * be documented as not mattering: "primary visibility for a scenery primitive is
 * a BVH over exact records and never consults the voxel payload, so a floret is
 * silhouetted exactly whether it spans four cells or a quarter of one". That was
 * measured and it was true. Analytic shading is gone; the primary returns the
 * voxel cell's surface, so this is the resolution the floret is drawn at, full
 * stop. Under {@link LEGIBLE_FEATURE_LEAVES} it is not a floret, it is noise.
 *
 * Under `aggregate` the leaf size is an input rather than a report, because the
 * published floret is already the larger of the plate's proportion and the
 * floor — so this returns the floor whenever the plate would have gone under it,
 * which is the answer, not a rounding of it.
 */
export function bonsaiFloretVoxelSpan(form: BonsaiForm, cellSize_m: number): number {
  if (!(cellSize_m > 0)) throw new RangeError("Bonsai floret voxel span needs a positive cell size");
  const width_m = form.canopy === "aggregate"
    ? bonsaiCanopyLadder(form, cellSize_m).floretWidth_m
    : 2 * form.floretRadius_m;
  return width_m / cellSize_m;
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
  // Throws on a bad `canopyGrainAcross` or leaf whichever crown is spelled, so a
  // form cannot carry a grain that only fails once someone flips `canopy`.
  const ladder = bonsaiCanopyLadder(spec, spec.leafSize_m ?? BONSAI_DEFAULT_LEAF_SIZE_M);
  /**
   * Head records the crown publishes, which under `aggregate` is the ladder's
   * answer rather than the form's.
   *
   * A head has to be small enough that the florets the leaf can draw actually
   * close it — see {@link CANOPY_HEAD_FLORET_SPAN} — so the count is a
   * consequence of the leaf. `spec.lobes` survives as the number of *clumps*
   * those heads are dealt into, which is the scale it was always read for.
   *
   * Under `florets` a head is a plain ellipsoid with nodules stood on it and
   * there is no containment to satisfy, so the authored count is the count.
   */
  const headCount = spec.canopy === "aggregate" ? ladder.heads : lobeCount;

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
  // Down from 2.15 / 1.42 / 1.10, and the reason is what the frame showed: a
  // foot 2.15 stump radii wide over a stump `stumpShare` tall is *wider than the
  // run is long*, so the flare is not a flare, it is a ball, and the frame's base
  // was a smooth grey ball with the buttresses stuck on it as pods. At 1.45 over
  // a stump raised to 0.16 of the height the run is 106 mm tall against a 73 mm
  // foot radius and it reads as a thrown column that spreads, which is what the
  // plate shows under the pebbles.
  const BASE_FOOT_SHARE = 1.45;
  const BASE_WAIST_SHARE = 1.16;
  const BASE_TOP_SHARE = 1.02;
  tube("base", [
    { at_m: V(0, -.85 * boleRadius, 0), radius_m: BASE_FOOT_SHARE * boleRadius },
    { at_m: V(0, -.10 * boleRadius, 0), radius_m: 1.30 * boleRadius },
    { at_m: V(0, .45 * stumpTopY, 0), radius_m: BASE_WAIST_SHARE * boleRadius },
    { at_m: V(0, stumpTopY, 0), radius_m: BASE_TOP_SHARE * boleRadius },
  ], bark(spec.barkValue + .01), "stem", GROUND_ENVELOPE_WASTE_BUDGET);

  // The buttresses. **Round roots**, and the module has now been round, then a
  // fin, and round again — so it is worth writing down why each way was right
  // when it was and what changed.
  //
  // Round sweeps came first and read as tentacles: a sweep is circular in
  // section at every station by construction, and under an analytic normal a
  // bundle of circular sections is unmistakably a bundle of pipes. The fix was a
  // fin — several short `seeded-lobes` envelopes chained along the run, each
  // inside the march's eccentricity cap, whose union is longer and thinner than
  // any one of them could be. That worked, and it is what shipped.
  //
  // Both halves of its argument have since gone. The *thinness* is unbuildable:
  // a blade a quarter as thick as it is tall works out to 14 mm on this stump,
  // which is 0.6 of a leaf, and under voxel shading a feature under a leaf is
  // not a thin blade but a row of disconnected cell faces. And the *pipe* it was
  // avoiding is no longer a pipe either — a cell-face surface has no smooth
  // circular section to give away.
  //
  // Thickening the fin back to something drawable made it worse rather than
  // better, and the frame said so: at a section the leaf could carry, each
  // chained envelope was 100 mm long, 86 tall and 42 thick, and `seeded-lobes`
  // silhouettes on its own lobes rather than filling its envelope, so what drew
  // was two or three eggs per segment — a ring of pods around the stump.
  //
  // So: one swept run per bearing, round, tapering, following the ground down
  // and out. It is also what the plate shows. `artifacts/plate-crops/canopy.png`
  // has no blades anywhere at the foot; it has pale rounded roots spreading over
  // the pebbles, which is exactly a tapered sweep. Five records instead of
  // fifteen, C1 along the whole run rather than a hard union at every junction,
  // and nothing under a leaf and a half until the toe.
  const rootFeet_m: Vec3[] = [];
  const ROOT_STEPS = 12;
  const ROOT_STATIONS = 5;
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
    const shoulderRadius_m = boleRadius * width;
    const toeRadius_m = shoulderRadius_m * (1 - spec.rootTaper);
    const reach = (() => {
      const fall = ROOT_MAXIMUM_FALL_SHARE * boleRadius;
      for (let step = 1; step <= ROOT_STEPS; step += 1) {
        // The ground under the root's own *outer edge*, not under its
        // centreline. A round root is as wide as its section, and the twenty
        // millimetres of one that ended up over the coping were all on the
        // outboard side of a centreline that was still on soil.
        const radial = wanted * step / ROOT_STEPS + toeRadius_m;
        if (!(-groundLocal(outward.x * radial, outward.z * radial) > fall)) continue;
        // Back off by the root's own *section at the toe*, not by a step of the
        // walk. A sweep is the union of balls about its stations, so the last
        // station reaches past its own centre by its radius there — and by half
        // as much again, because the run has been walked in twelfths and the toe
        // can land anywhere inside the last one. Backing off a step alone left
        // three millimetres of a root hanging over the coping and
        // `tests/bonsai.test.ts` said so, station by station.
        return Math.max(0, wanted * (step - 1) / ROOT_STEPS - 1.5 * toeRadius_m);
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
    // The section along the run, from the shoulder to the toe. The root is at
    // its thickest where it leaves the trunk and tapers the whole way out,
    // rather than being fattest in the middle like a leaf.
    //
    // `rootWidth` is the shoulder radius as a fraction of the stump's, and it is
    // well under one: a root as thick as the bole it leaves is not a root. At
    // 0.62 on the hero specimen the shoulder is 62 mm across — 2.5 leaves — and
    // the toe is 36 mm, which is 1.4 and is where a root is meant to disappear
    // into the ground anyway.
    const radiusAt = (t: number): number => boleRadius * width * (1 - spec.rootTaper * t ** .7);
    const stations: SweptTubeStation[] = [];
    for (let step = 0; step <= ROOT_STATIONS; step += 1) {
      const t = step / ROOT_STATIONS;
      stations.push({
        at_m: step === 0
          // Inside the base, on the axis, so the joint has no cap disc to show
          // and the roots fuse with the flare they leave rather than being
          // stuck onto it. The same treatment every run in this module gets.
          ? V(0, .55 * stumpTopY, 0)
          : at(t),
        radius_m: step === 0 ? Math.max(radiusAt(0), BASE_TOP_SHARE * boleRadius * .75) : radiusAt(t),
      });
    }
    tube(`root-${index}`, stations,
      bark(spec.barkValue - .03 + .04 * hash01(seed + 23 * index)), "root", ROOT_ENVELOPE_WASTE_BUDGET);

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
  const lobePlanShare = spec.lobeFill / Math.sqrt(headCount);
  /**
   * The clump a head belongs to, and how far it may wander inside it.
   *
   * The heads used to be the crown's only sub-scale, so they went straight onto
   * one sunflower. Now that the leaf sets their size, a fine leaf deals a
   * hundred and sixty of them and an even sunflower of a hundred and sixty
   * equal balls is a *skin*, not a cauliflower — the plate's crown reads as a
   * few dozen clumps of florets, and it is the clumping that says which.
   *
   * So `spec.lobes` keeps its old job at its old count and the heads are dealt
   * into it: clump centres on the sunflower the heads used to sit on, and each
   * head placed on a small sunflower of its own inside its clump. Round-robin
   * rather than blocked, so a clump count that does not divide the head count
   * spreads the remainder instead of leaving one clump short.
   */
  const clumpCount = ladder.clumps;
  const clumpPlanShare = spec.lobeFill / Math.sqrt(clumpCount);
  // `crownThickness_m` is the crown's own surface either way, because that is
  // the number the reference is read for — and the two canopy modes reach it by
  // solving a different equation. Under `florets` a nodule stands proud of the
  // lobe and a floret proud of that, so the lobe underneath is what is left of
  // the plate after both. Under `aggregate` the cluster's envelope *is* the
  // surface, so the plate is exactly one head thick and there is nothing to
  // subtract.
  //
  // A lobe is a squashed ball, so the area to cover is dominated by its two
  // faces: n nodules of radius r cover it once over when n pi r^2 = 2 pi a^2,
  // and the factor above one is the overlap that fuses them. Capped against the
  // plate's own half-thickness for the reason `bumpsPerLobe` states.
  const lobePlanRadius = lobePlanShare * Math.sqrt(spec.crownRadius_m[0] * spec.crownRadius_m[1]);
  // Which is why the head count and the plate thickness cannot be tuned apart.
  // At twenty-four heads the hero head comes out 176 mm wide against a 125 mm
  // plate — near enough round, which is what the plate's heads are — where nine
  // put it at 331 by 125 and the crown was a disc with nothing in it. Every head
  // is placed by its own *top*, so however the swell sizes them the crown's
  // upper surface stays one plane and the variation goes into the underside.
  //
  // Once the leaf started sizing the head, "one head thick" stopped being a
  // description and became a defect. At a 6.25 mm leaf the crown deals 146 heads
  // of 54 mm plan, and holding them to half the plate's thickness made each one a
  // 54 x 123 mm *column*: a palisade rather than a granulation, and it stood
  // every head's own eccentricity at 2.5 for the march to pay for. So the head is
  // near enough round, and the crown stops being one head thick — it becomes a
  // granulated shell standing on the bed, which is what the bed was always for
  // and what the plate actually shows. The `min` is what keeps the old shape at
  // the old leaf: at 25 mm the round head would be 139 mm against a 125 mm plate,
  // so the plate still binds and the coarse crown is unchanged.
  const lobeHalfThickness = spec.canopy === "aggregate"
    ? Math.min(.5 * crownThickness, CANOPY_HEAD_FLATTEN * lobePlanRadius)
    : Math.max(.18 * crownThickness, (.5 * crownThickness - .38 * floretRadius) / 1.495);
  const bumpRadius = Math.min(1.1 * lobeHalfThickness, 1.14 * lobePlanRadius * Math.sqrt(2 / bumpsPerLobe));
  // The plane the heads hang from. Not the mid-plane: a head is placed by its
  // *top* and grows downward, so the swell that gives the crown its clumped
  // masses thickens the underside instead of doming the top — and a flat top
  // over a rounded underside is the one thing the reference is unambiguous
  // about. The droop then takes the rim down with the square of the radius.
  const crownTopY = crownCenter.y + .5 * crownThickness;
  const lobes: BonsaiLobe[] = [];
  /** Heads in the fullest clump, which is what a head's place inside one is measured against. */
  const perClump = Math.ceil(headCount / clumpCount);
  for (let index = 0; index < headCount; index += 1) {
    // Round-robin, so a clump count that does not divide the head count spreads
    // its remainder over the first few clumps rather than starving the last one.
    const clump = index % clumpCount;
    const within = Math.floor(index / clumpCount);
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
    // The head's own eccentricity, checked here rather than trusted, because it
    // is what bounds the march that draws it and because the swell and the
    // thickness jitter move it independently of anything a form authors. A hero
    // head is 1.4; the cap is 4. See `CANOPY_HEAD_MAXIMUM_ECCENTRICITY`.
    //
    // Under `florets` there is no march: the head is published as a plain
    // ellipsoid, solved in closed form, and the explicit crown's heads really are
    // flat plates at five to one. So the bound applies to the aggregate only —
    // it is a property of how the shape is *evaluated*, not of the shape.
    const eccentricity = Math.max(radius.x, radius.y, radius.z) / Math.min(radius.x, radius.y, radius.z);
    if (spec.canopy === "aggregate" && eccentricity > CANOPY_HEAD_MAXIMUM_ECCENTRICITY) {
      throw new RangeError(
        `Bonsai canopy head ${index} is ${eccentricity.toFixed(1)} times longer than thick;`
        + ` the march is only sound to ${CANOPY_HEAD_MAXIMUM_ECCENTRICITY}`,
      );
    }
    // Where the clump sits on the crown. See `CROWN_RIM_CROWDING` for why the
    // exponent is a third rather than the equal-area half. The endpoints are
    // exact: the first clump sits on the crown's own axis, so the plate has full
    // thickness where it is thickest, and the last sits at the rim.
    const clumpU = clumpCount > 1 ? (clump / (clumpCount - 1)) ** CROWN_RIM_CROWDING : 0;
    const clumpAngle = GOLDEN_ANGLE * clump + 1.31 * hash01(seed + 5);
    const clumpRadius = V(
      spec.crownRadius_m[0] * clumpPlanShare,
      0,
      spec.crownRadius_m[1] * clumpPlanShare,
    );
    const clumpX = Math.max(0, spec.crownRadius_m[0] - clumpRadius.x) * clumpU * Math.cos(clumpAngle);
    const clumpZ = Math.max(0, spec.crownRadius_m[1] - clumpRadius.z) * clumpU * Math.sin(clumpAngle);
    // And where the head sits inside its clump. The equal-area exponent here
    // rather than the rim crowding: a clump is judged on filling its own plan,
    // and crowding its rim is what leaves a hole in the middle of every clump.
    const withinU = perClump > 1 ? Math.sqrt(within / (perClump - 1)) : 0;
    const withinAngle = GOLDEN_ANGLE * index + 2.13 * hash01(seed + 17 + 7 * clump);
    // Both offsets are inset by the radius of the thing they carry, so their sum
    // is bounded by `crownRadius - radius` in the crown's own normalised metric
    // and the head's far edge lands on the authored silhouette rather than past
    // it — the same guarantee the single sunflower gave, kept through two hops.
    const planX = clumpX + Math.max(0, clumpRadius.x - radius.x) * withinU * Math.cos(withinAngle);
    const planZ = clumpZ + Math.max(0, clumpRadius.z - radius.z) * withinU * Math.sin(withinAngle);
    // The droop follows the head's *own* distance out along the crown rather
    // than its clump's, so a head on the outboard side of an inboard clump
    // hangs where it actually is. Normalised by the reach a head centre has, so
    // a rim head still gets the full authored droop.
    const u = Math.min(1, Math.hypot(planX / spec.crownRadius_m[0], planZ / spec.crownRadius_m[1])
      / Math.max(1e-6, 1 - lobePlanShare));
    lobes.push({
      center_m: V(
        crownCenter.x + planX,
        crownTopY - radius.y
          - spec.crownDroop * crownThickness * u * u
          + .07 * crownThickness * hashSigned(seed + 43 * index),
        crownCenter.z + planZ,
      ),
      radius_m: radius,
    });
  }

  // ---- Canopy bed ---------------------------------------------------------
  // The continuous mass the heads stand on, and it is read off the plate rather
  // than invented to plug a hole. `artifacts/plate-crops/canopy.png` shows the
  // canopy's underside between the limbs as one smooth pale surface; the floret
  // heads are on the top and around the rim, and what is under them is a dome.
  //
  // Which is also the closure the heads cannot provide. A finite set of round
  // heads never paves a plane — measured on the hero crown, twenty-four heads at
  // `lobeFill` 1.20 leave 15.4% of the plate's inner nine tenths open to a
  // vertical ray, and 7.2% of it open by more than a whole leaf, which is sky
  // through the canopy. Raising the fill barely touches it: at 1.48 the heads
  // are 218 mm and 11.8% is still open. It is not a tuning failure, it is what
  // circles do.
  //
  // Six records, on the same sunflower and the same droop as the heads so the
  // bed follows the plate's own curve, and inset in *plan* as well as in height:
  // the rim has to be the heads' own scallop and a bed reaching the authored
  // crown radius would draw a smooth ellipse under it. The heads then stand
  // `CANOPY_BED_INSET_SHARE` of a head proud of the bed everywhere, which at the
  // hero leaf is 35 mm — 1.4 leaves, and the largest single modulation on the
  // whole crown.
  // Off the *clump* count and not the head count. The bed's job is to close the
  // plan under the crown, which is a question about the crown's size and not
  // about how finely the leaf lets its surface be granulated — tying it to the
  // heads would have a fine leaf publish forty smooth ellipsoids where six do
  // the job, and each of them would be a smooth surface competing with the very
  // granulation the finer leaf was spent on.
  const bedCount = Math.max(3, Math.round(ladder.clumps / CANOPY_BED_HEADS_PER_RECORD));
  const bedInset_m = CANOPY_BED_INSET_SHARE * ladder.headWidth_m;
  const bedPlanShare = CANOPY_BED_FILL / Math.sqrt(bedCount);
  const bed: BonsaiLobe[] = [];
  if (spec.canopy === "aggregate") {
    for (let index = 0; index < bedCount; index += 1) {
      // The equal-area exponent rather than the heads' rim crowding: this layer
      // is judged on covering the plan and nothing else, and crowding the rim is
      // what leaves a middle to see through.
      const u = bedCount > 1 ? Math.sqrt(index / (bedCount - 1)) : 0;
      const angle = GOLDEN_ANGLE * index + 2.9 * hash01(seed + 11);
      const reach = V(
        Math.max(1e-4, spec.crownRadius_m[0] - bedInset_m),
        0,
        Math.max(1e-4, spec.crownRadius_m[1] - bedInset_m),
      );
      const radius = V(
        reach.x * bedPlanShare,
        // Off the *plate* and not off the head. These were the same number while
        // a head was half the plate thick; once the leaf started sizing heads,
        // reading the head here shrank the bed with them and the crown lost its
        // body — a 42 mm bed under a 125 mm plate, with nothing between the
        // granulated top and the limbs. The bed is the plate, by definition.
        Math.max(1e-4, .5 * crownThickness - .5 * bedInset_m),
        reach.z * bedPlanShare,
      );
      bed.push({
        center_m: V(
          crownCenter.x + Math.max(0, reach.x - radius.x) * u * Math.cos(angle),
          crownTopY - bedInset_m - radius.y - spec.crownDroop * crownThickness * u * u,
          crownCenter.z + Math.max(0, reach.z - radius.z) * u * Math.sin(angle),
        ),
        radius_m: radius,
      });
    }
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
  // One aggregate record per head, or three scales of explicit primitive. See
  // `BonsaiForm.canopy` for why both exist and what each costs.
  let floretCapArea = 0;
  if (spec.canopy === "field") {
    /**
     * The crown as **cloud pads**: one tape per pad, seven of them.
     *
     * Six ops each against the aggregate's hundred and fifty `seeded-lobes`
     * records, and it is not a saving so much as a different object: a tape
     * carries two scales of *packed* cellular relief where a cluster record
     * carries one scale of separated lobes, so this is the first construction
     * here that is actually self-similar. See `bonsai-canopy-field.ts` for why
     * the nested `scatter` the header proposes cannot do it.
     *
     * **It was one record, and one record was the fault.** A displacement can
     * roughen a mass; it cannot turn one mass into six. The single-ellipsoid
     * version rendered as a smooth flat disc on a stem at every relief setting
     * tried, because the silhouette was the ellipsoid's and there was only ever
     * one ellipsoid — where the plate's canopy is five to seven distinct masses
     * with sky between them. The scale at which the object stops being connected
     * is geometry; everything below it is relief. See `bonsai-canopy-pads.ts`.
     *
     * There is no bed. The bed existed because a finite set of round heads never
     * paves a plane — 15.4 % of the plate stayed open to a vertical ray — and a
     * displaced solid has no such holes: each pad's core *is* closed and the
     * relief only roughens it. Sky *between* pads is wanted and is not a hole.
     */
    // The leaf the set will be voxelized into, so the ladder can stop where the
    // lattice does. At refinement depth 3 — what production runs — this is
    // 0.78 mm and the tape carries a rung the 6.25 mm leaf cannot draw.
    const field = bonsaiCanopyField(spec.leafSize_m ?? BONSAI_DEFAULT_LEAF_SIZE_M);
    const pads = bonsaiCanopyPads({
      crownRadius_m: spec.crownRadius_m,
      crownThickness_m: crownThickness,
      crownDroop: spec.crownDroop,
      center_m: crownCenter,
      seed: seed >>> 0,
    });
    for (const [index, pad] of pads.entries()) {
      nodes.push({
        kind: "field-program",
        id: `${key}/pad-${index}`,
        tags: partTags("crown"),
        // A tape is authored in metres and must resolve at a unit scale; the
        // record is centred on its pad and takes no orientation, exactly as the
        // heads it replaces did.
        place: { units: "metres", position: pad.center_m },
        program: bonsaiCanopyPadProgram(pad.radius_m, field, (seed + 7919 * index) >>> 0),
        // A shade per pad, off the same seed the pad is placed with. The set is
        // one flat plaster closure with the key raised, so a canopy of identical
        // masses loses its separation the moment two pads meet edge-on; this is
        // the little help the form gets.
        material: canopy(spec.canopyValue - .03 + .05 * hash01(seed + 131 * index)),
      });
      coverCrown(pad.center_m, pad.radius_m);
    }
    floretCapArea = pads.reduce((area, pad) => area + ellipsoidSurfaceArea(pad.radius_m), 0);
  } else if (spec.canopy === "aggregate") {
    // The bed first, so a head is always the nearer surface where the two meet
    // and the union's shading normal comes off the head. Its florets are the
    // same field at the same fraction of a much larger envelope, which is the
    // one place in this object the self-similarity is spelled rather than
    // arranged: the crown is a cluster of beds, a bed is a cluster of florets,
    // and a head is a cluster of florets one scale down.
    for (let index = 0; index < bed.length; index += 1) {
      const lobe = bed[index];
      // An ellipsoid, and the exactness is the point: this layer's whole job is
      // to be closed, and an ellipsoid fills its envelope by definition where
      // every procedural field fills some fraction of it. That was measured
      // rather than assumed — the first bed built here was a `seeded-lobes`
      // record like the heads, and a 411 mm envelope with twelve lobes at the
      // maximum span was still 35% open to a vertical ray, because the lobes are
      // that fraction of the envelope in its *normalised* space and the
      // anisotropy then shortens two of their three axes. Raising the
      // displacement from 0.35 to 0.85 moved it to 35.1% from 40.4%. It is the
      // wrong tool: a field that silhouettes on its own lobes is exactly a field
      // that does not reach its envelope.
      //
      // What it costs is that the crown's underside is smooth. That is not a
      // concession — the plate's underside *is* smooth, one pale dome between
      // the limbs, and the granulation there is a scale below anything this leaf
      // can draw.
      nodes.push({
        kind: "ellipsoid", id: `${key}/bed-${index}`, tags: partTags("crown"),
        place: { position: lobe.center_m }, radius: lobe.radius_m,
        // Under the heads and reading as their shadow side. The whole set is one
        // flat plaster closure with the key raised, so what separates the bed
        // from the heads on it is form; this is the little help that form gets.
        material: canopy(spec.canopyValue - .07 + .02 * hash01(seed + 79 * index)),
      });
      coverCrown(lobe.center_m, lobe.radius_m);
    }
    for (let index = 0; index < headCount; index += 1) {
      const lobe = lobes[index];
      // One head, one record, and it carries the two things that used to take
      // two: the floret granulation *and* the closure.
      //
      // A lattice could not do both. It only stops letting rays through at
      // `r >= P*sqrt(3)/2`, where every cell corner is inside a sphere, and at
      // that ratio the packing is a solid the envelope's hard `max` cuts off
      // flush — so it needed an ellipsoid core a granulation-depth beneath to be
      // opaque, and that core is what the frame's black pits were the bottom of.
      // `seeded-lobes` places its florets inside the envelope by a rule that
      // guarantees the clip never reaches them, so the silhouette it draws is
      // the florets' own rather than the ellipsoid's, and it is a smooth union
      // of solid ellipsoids and therefore closed with nothing behind it. Three
      // thousand rays through the head's centre from every direction find it
      // every time; see the header's table.
      //
      // The floret size is the leaf's business rather than the form's — see
      // `bonsaiCanopyLadder` — and it arrives here as a fraction of the head,
      // which is exactly the space the ABI places lobes in, so a head of any
      // width gets florets of the size the leaf can draw without either end
      // being re-authored.
      const shortest_m = Math.min(lobe.radius_m.x, lobe.radius_m.y, lobe.radius_m.z);
      // The span is solved against *this* head rather than the nominal one.
      //
      // `lobeSwell` deals heads over a 1.6:1 range on purpose — a few large ones
      // with smaller ones packed against them is most of what separates a
      // cauliflower from a mound of equal balls — and `lobeSpan` is a fraction of
      // whatever envelope it lands in, so a single nominal span would give the
      // widest head florets 1.6 times the narrowest head's. The floret is a
      // *length* the leaf decides, not a proportion of whichever head it fell
      // in, so the span has to be re-solved per head to keep it one.
      const span = Math.min(SEEDED_FLORET_MAXIMUM_SPAN, ladder.floretWidth_m / (2 * lobe.radius_m.x));
      nodes.push({
        kind: "cluster", field: "seeded-lobes", id: `${key}/lobe-${index}`, tags: partTags("crown"),
        place: { position: lobe.center_m },
        lobe: lobe.radius_m,
        lobeCount: SEEDED_FLORET_COUNT_LOW + (index % SEEDED_FLORET_COUNT_RANGE),
        anisotropy: SEEDED_FLORET_ANISOTROPY,
        lobeSpan: span,
        // Whatever the span left of the containment budget, so a head whose
        // florets were raised to the legibility floor still gets the size
        // variation that separates a cauliflower from a raspberry — it just gets
        // less of it. On the hero specimen the leaf floors the floret at 75 mm
        // and the swell deals heads of 143 to 233 mm, so the span runs 0.46 down
        // to 0.32 and this runs 0.32 up to 0.46.
        lobeSpanSpread: Math.min(
          SEEDED_FLORET_SPAN_SPREAD + (SEEDED_FLORET_MAXIMUM_SPAN - span),
          1 - span - SEEDED_FLORET_BLEND_SHARE - .02,
        ),
        displacement: SEEDED_FLORET_DISPLACEMENT,
        smoothRadius: SEEDED_FLORET_BLEND_SHARE * shortest_m,
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
  for (let index = 0; index < headCount; index += 1) {
    const lobe = lobes[index];
    nodes.push({
      kind: "ellipsoid", id: `${key}/lobe-${index}`, tags: partTags("crown"),
      place: { position: lobe.center_m }, radius: lobe.radius_m,
      material: canopy(spec.canopyValue - .09 + .03 * hash01(seed + 71 * index)),
    });
    coverCrown(lobe.center_m, lobe.radius_m);
  }

  const bumps: { readonly lobe: number; readonly index: number; readonly center: Vec3; readonly radius: number }[] = [];
  for (let index = 0; index < headCount; index += 1) {
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
  for (let index = 0; index < headCount; index += 1) {
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
