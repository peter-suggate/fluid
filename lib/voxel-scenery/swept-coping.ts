import type { Vec3 } from "../model";
import type { SceneryMaterial, SceneryNode } from "../scenery-graph";
import { alongAxis, V } from "./builder";

/**
 * A coping: a bullnose kerb swept along a plan curve and set into the ground.
 *
 * One species — a rounded solid following a rail — and therefore not only the
 * pond's rim. The same generator is a path edging, a planter kerb, a step nosing
 * or a parapet capping; it takes a rail and a ground query, so it can follow a
 * pond, a wall footing or a stream bank without knowing what drew them.
 *
 * ## Why this is not part of the heightfield
 *
 * The pond vessel bakes its coping into the terrain grid, which buys one
 * authority for the surface the water rests in and the surface that renders
 * (`pond-vessel.ts`). It costs three things the reference has, and each is a
 * consequence of the representation rather than of the tuning:
 *
 *  - **No line where the rim meets the plaster.** `pondVesselCrestProfile` is
 *    tangent at its foot by construction — measured, the surface normal is
 *    within a quarter of a degree of vertical at the foot and takes 2.75 mm of
 *    run to reach 42 degrees — so the coping *swells out of* the ground with a
 *    continuous normal. The reference's rim is a carved object dropped into the
 *    plaster and there is a hard shadow line all the way round it. An honest
 *    union of a solid with the ground has that line for free: the two surfaces
 *    meet at whatever angle they meet at, and the normal jumps.
 *  - **No section wider than the plaster it stands on.** A heightfield rim is a
 *    swelling of the ground, so its width and its height are the same profile
 *    read twice and it cannot be broad and low without the plaster around it
 *    becoming broad and low too. A solid section is authored on its own terms —
 *    `widthToHeight` below is the whole point of this generator now — and the
 *    reference's rim is 2.8 times as wide as it is tall.
 *  - **No sharpening without a global tax.** The grid carries one Lipschitz
 *    slope bound for the whole scene (`terrainGridExtent`) and the dry render's
 *    march steps by it everywhere, so steepening any one feature is stepped for
 *    on the flat plaster metres away. Measured on the hero pond: taking the
 *    inner face from 160 mm of run to 35 mm took the bound from 2.85 to 9.51 and
 *    the mean height evaluations per hit ray from 41.4 to 65.6, against a
 *    64-step budget. A primitive costs the frame nothing anywhere it is not.
 *
 * What it costs in return is that **scenery is invisible to the water**
 * (`VoxelSceneSource` is container ∪ terrain ∪ rigid bodies). That is affordable
 * here and it is worth saying exactly why rather than waving at it: a coping is
 * the part of a vessel that stands *above* the still waterline. On the hero pond
 * the crest is 46 mm above the outer ground and the waterline sits 40.9 mm below
 * the lowest plaster outside it, so the water at rest never reaches the coping at all — the wall it
 * presses against is the plaster the coping is set into, which is terrain either
 * way. A coping swept as scenery over a vessel whose *wet* surfaces are still
 * heightfield is not a rim the water pours through; it is a rim the water was
 * never going to touch. Splash is a different question and the answer to it is
 * freeboard in the terrain, not geometry in the renderer.
 *
 * ## The section, and the two numbers that describe it
 *
 * The section is a circle, because a bullnose is an arc. It is authored as the
 * two things you can stand in front of the pond and see: how far the crest stands
 * proud of the ground (`crestHeight_m`), and how wide the rim's footprint on that
 * ground is as a multiple of that height (`widthToHeight`). The radius and the
 * depth the circle's centre is bedded to are derived from those, because neither
 * is a thing anybody can look at and estimate.
 *
 * The one number worth carrying in your head is that **`widthToHeight = 2` is a
 * semicircle**, and it is the hinge of the whole family:
 *
 *  - Below two the circle's centre stands *above* the plaster and its widest
 *    point with it, so between that widest point and the ground the flank turns
 *    back under itself. That is an undercut, and it costs height: for a circle,
 *    `crestHeight >= radius >= groundHalfWidth` whenever the centre is above the
 *    plaster, so **no undercut section can be more than twice as wide as it is
 *    tall**. It is also what makes a rim read as a pipe lying on the floor rather
 *    than as a kerb set into it, and at the 1.15 this form used to carry it read
 *    as one in the frame.
 *  - Above two the centre is bedded *below* the plaster and the visible solid is
 *    a circular cap: a broad gently domed top rolling over into a steep flank
 *    that enters the plaster with no overhang. At the 2.8 authored below, the
 *    flank meets the ground running 71.1 degrees from horizontal, which is still
 *    a hard line with a 71-degree normal jump against the flat plaster — the
 *    meeting this generator exists for — but it is a line the rim stands *behind*
 *    rather than one it hangs over.
 *
 * What is paid for the cap is burial: the radius goes as `groundHalfWidth^2 /
 * (2 * crestHeight)` once the width dominates, so the hero rim carries 90.2 mm of
 * circle under the plaster to show 46 mm above it. That is free in occupancy —
 * the buried part is inside a terrain that is solid to the container floor there,
 * so the union has exactly the same surface — and it is not free in the record's
 * proxy box, which is what the stride table below is about.
 *
 * ## Records
 *
 * One round-cone SDF per station pair — the convex hull of the two end spheres —
 * with consecutive segments sharing their endpoint sphere. Two properties follow
 * from that and both are load-bearing.
 *
 * **It is C1 through every joint.** A round cone's lateral surface is tangent to
 * both of its end spheres, so where two of them meet on a shared sphere the
 * surface has one normal however far the rail turns. That is *not* true of the
 * constructions this replaced, and the difference invalidated the rule this
 * module was built on: a truncated cone is straight and its cap is planar, so a
 * cone chain's shading normal stepped by the whole turn angle at every joint and
 * a convincing rim cost 336 segments at 1.1 degrees apiece. With round cones the
 * joint angle costs nothing, and the only thing rail resolution still buys is
 * chord sag — `c^2 / 8R`. The hero plan's tightest turn has a 186 mm radius, so
 * the 36.4 mm chords the 67 stations used here leave 0.89 mm of sag on a 128.8 mm
 * rim. It was 0.57 mm at 84 stations and 29.1 mm chords.
 *
 * **Its density is the ceiling, not its total.**
 * `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` is 64 primitives per 200 mm brick and
 * the rest of a crowded brick is dropped from the voxelized occupancy — not from
 * primary visibility, which is a BVH over the records — so an overflowing brick
 * keeps its silhouette and silently stops casting shadows and stops occluding
 * indirect light. A rim is the worst possible shape for that ceiling: a long thin
 * run at a fixed segment rate is *uniformly* dense all the way round, so it
 * overflows everywhere at once or nowhere.
 *
 * Both together are why `segmentStride` is 5, and why it used to be 4: **the
 * stride is the price of the section's radius.** A record's proxy box is its
 * round cone's, so a section that went from a 36.7 mm radius to a 68.1 mm one
 * reaches into more bricks in both directions across the rail, and the count in
 * the bricks the rim does occupy goes up with it. Measured on the hero garden's
 * published proxies against a 200 mm brick — `rim/brick` is the most rim records
 * any one brick holds, which is deliberately *not* the busiest brick in the scene
 * (that is the canopy, and it holds no rim at all, so it moves when the bonsai
 * moves and tells you nothing about a coping):
 *
 *   section          stride   rim recs   rim/brick
 *   63.5 x 55 mm        4         84         14
 *   128.8 x 46 mm       3        112         24
 *   128.8 x 46 mm       4         84         18
 *   128.8 x 46 mm       5         67         15
 *   128.8 x 46 mm       6         56         13
 *
 * Widening the section at the old stride would have cost 29 % of the rim's
 * per-brick headroom against a 64-record ceiling whose overflow is a *silent
 * drop*. Stride 5 buys it all back — 15 against the old 14 — for 0.32 mm of
 * extra chord sag and 17 fewer records than the rim used to publish.
 *
 * Coarsening the rail also makes the rim *smoother*, which is the opposite of
 * what the old cone-chain rule predicted and is the clearest evidence that the
 * rule died with the cap: what the extra stations were carrying was not
 * curvature, it was `relief_m` sampled faster than the eye wants it.
 * `reliefLobes` came down with the stride to keep four stations per lobe, which
 * is the band limit its own doc comment asks for.
 *
 * ## The marched sweep was tried here, and does not fit
 *
 * `field: "tapered-sweep"` (`lib/voxel-scenery/swept-tube.ts`) is the multi-
 * segment sweep SDF this record layer was factored apart to wait for, and it
 * carries the *identical solid*: a sweep segment is the same round cone, so the
 * surface does not move by a nanometre. It collapses the ring to 64 records and
 * takes the scene's busiest brick to 67 with one brick over. It was still
 * rejected, and the reason is worth keeping so nobody re-tries it blind.
 *
 * A cluster is clipped to a fitted ellipsoid by a hard `max`, and the fitter
 * minimises envelope volume subject to every *station* clearing a Lipschitz-1
 * lower bound on the ellipsoid distance. On a hose that is fine. On a coping the
 * run is short and fat — it was 73 mm of section over 29 mm of rail when this was
 * measured and the broad section makes it 136 over 36 — so the envelope
 * hugs the solid, and measured, the rim's own surface comes within 0.81 mm of
 * its clipping envelope on 21 of 64 records, always at a record's ends. Nothing
 * is actually cut. What breaks is the *normal*: the shading gradient is a
 * tetrahedron tap at a quarter of the field's feature radius, which here is
 * about 7.5 mm, so at a record end the taps reach outside the envelope and come
 * back with the ellipsoid's normal instead of the rim's. Measured on the
 * published records, the normal step across a record boundary averages 8.57
 * degrees against 1.63 inside one, peaking at 18.2 — and at 1600x920 that is
 * unmistakable transverse banding on the near rim.
 *
 * It is not tunable from here. The blend radius is the only lever this side of
 * the seam that widens the envelope, and raising it makes both problems worse:
 * it beads the section at every station, and measured, 4 mm of blend takes the
 * boundary step from 18.2 to 25.2 degrees while growing the crest by up to 2 mm.
 * The fix belongs in the fitter — an envelope margin scaled to the field's own
 * feature radius rather than to the station radius alone — and until it exists
 * the chain is strictly better here: it reaches the same busiest brick at
 * stride 4 with a fifth of the shading defects.
 */

/**
 * Widest section this generator will grow, as a multiple of the crest height.
 *
 * A typo guard rather than a shape limit. The buried radius goes as
 * `widthToHeight^2 * crestHeight / 8`, so at eight the circle carries sixteen
 * crest heights of itself under the plaster to show one above it, and a proxy
 * box that size stops being a rim as far as the brick binning is concerned.
 */
export const SWEPT_COPING_MAXIMUM_WIDTH_TO_HEIGHT = 8;

/**
 * The rim's surface grain, and why it is a constant here rather than geometry.
 *
 * The plate's rim is cast stone: a dense, even pitting at about two millimetres,
 * *finer* than the water-worn boulders beside it and subtractive — pits, not
 * ridges. That is a shape, not a shading trick, and the destination for it is a
 * `field-program` record (`lib/svo-field-program.ts`) whose tape subtracts a
 * Worley field from the swept section.
 *
 * It is not emitted, and both halves of the reason belong here:
 *
 *  - **There is no `field-program` scenery node kind.** `SceneryNode` runs box,
 *    cylinder, ellipsoid, capsule, torus, cone, cluster and group
 *    (`lib/scenery-graph.ts`), and a generator cannot mint a record kind the
 *    graph cannot carry. **Still the blocker**, and now the only one shared with
 *    `stone-set.ts` and `bonsai.ts` — all three species want the same one node
 *    kind for three different tapes.
 *  - ~~The op set is not there yet.~~ **The modifiers landed.** The table now
 *    carries `worley-subtract` and the variable-k `smooth-union` /
 *    `smooth-subtract` / `smooth-intersect` family the grain asks for.
 *  - **What is still missing is a source op for a swept run.** The tape's
 *    sources are `sphere`, `ellipsoid` and `box`, all centred blobs, so a field
 *    program cannot carry a coping's *body* — only decorate one. The shape that
 *    would is already specified next door as a cluster field rather than a tape
 *    op: `SvoClusterTaperedSweepField`, a round-cone chain through up to eight
 *    control points as **one record**, exact and Lipschitz-1. Sixty-four coping
 *    records are nine of those. Nothing in this file authors one, and doing so
 *    is a much smaller change than a tape — it needs no new node kind at all,
 *    because `kind: "cluster"` already carries a tapered sweep.
 *
 * The numbers are recorded so that whoever wires it does not have to re-derive
 * them from the plate, and so the decision that they are *invisible today* is
 * written down rather than rediscovered: shading is voxel cell faces at
 * `HERO_GARDEN_CELL_M` = 25 mm, and a 2 mm feature is a twelfth of a cell. It
 * resolves when leaves do, which is the point of authoring it as a field rather
 * than as records.
 */
export const SWEPT_COPING_GRAIN = Object.freeze({
  /** Deepest a pit cuts into the section, in metres. Subtractive only. */
  depth_m: 0.002,
  /** Mean spacing of the pits, in metres. Cast aggregate, not river-worn. */
  period_m: 0.002,
  /**
   * How far in from the section's inner ground line the grain is faded out.
   *
   * The inner edge is the sharpest line in the plate and it is what makes the
   * pond read as a vessel. Unmasked erosion crumbles exactly that. Three
   * millimetres is a pit and a half of run, which is enough to bring the
   * amplitude to zero without the fade itself being a visible band.
   */
  lipMaskRun_m: 0.003,
} as const);

/**
 * The grain's amplitude at a point on the section, as a fraction of `depth_m`.
 *
 * `inwardOffset_m` is measured from the section's inner ground line, positive
 * outward across the rim. Zero at the lip, one by `lipMaskRun_m`, smoothstepped
 * so the fade has no edge of its own — a linear ramp puts a crease where it
 * reaches full, which on a 2 mm grain is a bigger defect than the grain.
 *
 * Exported and unit-tested rather than inlined into a tape that does not exist
 * yet, because this is the part of the grain that is a *decision* and the rest
 * is a noise call.
 */
export function sweptCopingGrainAmplitude(inwardOffset_m: number): number {
  const t = Math.min(1, Math.max(0, inwardOffset_m / SWEPT_COPING_GRAIN.lipMaskRun_m));
  return t * t * (3 - 2 * t);
}

/** Shape only: the species, with no rail, no ground and no seed. */
export interface SweptCopingForm {
  /** How far the crest stands above the ground the coping is set into. */
  readonly crestHeight_m: number;
  /**
   * The rim's full footprint on that ground, as a multiple of `crestHeight_m`.
   *
   * Two is a semicircle and is the hinge described in the module header: below
   * it the section undercuts and cannot be more than twice as wide as it is
   * tall, above it the section is a bedded circular cap with no overhang. Must
   * be in (0, 8]; the upper bound is a typo guard, not a shape limit — at eight
   * the buried radius is already sixteen times the crest.
   *
   * This replaced an `undercut` fraction that authored the same family in the
   * other coordinates (`undercut` in [0, 1) is `widthToHeight` in (0, 2]), and
   * the swap is not cosmetic: the reference's rim is at 2.8, which the old
   * parameterization could not spell at all.
   */
  readonly widthToHeight: number;
  /**
   * How much the section swells and narrows along the run, as a fraction of the
   * radius. This is the difference between a formed kerb and an extrusion, and
   * it is the same argument `PondVesselSpec.sectionWidthVariation` makes: the
   * eye reads an unvarying crest line long before it reads a wandering outline.
   *
   * A fraction of the *radius*, so it has to come down when the radius goes up:
   * the eye reads the swell in millimetres of footprint, not in percent. The
   * hero form went from 14 % of a 36.7 mm radius to 7 % of a 68.1 mm one, which
   * is 5.1 mm of swell against 4.8 mm — the same rim, not a calmer one.
   */
  readonly sectionVariation: number;
  /** How much the crest rises and falls along the run, as a fraction of `crestHeight_m`. */
  readonly crestVariation: number;
  /**
   * Control points for those two modulations around a closed run.
   *
   * Deliberately not the rail's own lobe count. A section that swelled exactly
   * where the outline bulged reads as one repeating motif rather than as two
   * independent accidents of the same hand.
   */
  readonly variationLobes: number;
  /**
   * A second, finer modulation of the same two quantities, in metres.
   *
   * This is the hand-formed unevenness, and it has to be *smooth*. The obvious
   * construction — an independent hash per rail node — was tried and is wrong:
   * white noise at the rail's own resolution makes every segment a slightly
   * different cylinder, and because the catalog has no smooth union, each joint
   * is already a crease that the shading finds. The result reads as a
   * caterpillar rather than as a kerb, at an amplitude (1.2 mm on a 37 mm
   * section) far below anything that should have been visible. So the relief is
   * a second spline at `reliefLobes` control points, band-limited well below the
   * segment rate, and the grain below *it* is the material's business.
   */
  readonly relief_m: number;
  /**
   * Control points for that finer modulation. Roughly four times
   * `variationLobes` puts its wavelength at a few segments, which is a hand's
   * unevenness rather than a manufacturing tolerance.
   */
  readonly reliefLobes: number;
  /**
   * Rail points per emitted segment, resampled uniformly in index.
   *
   * One is the rail as given. Two halves the record count and quadruples the
   * outline's faceting.
   */
  readonly segmentStride?: number;
}

/**
 * The pond's rim: a broad flattened bullnose, nearly three times as wide as it
 * is tall.
 *
 * ## What the plate says, in numbers
 *
 * Measured off `artifacts/plate-crops/coping.png` and the plate it is cut from
 * (`output/imagegen/garden-pond-hose-fill-simplified.png`), against the render it
 * is registered to (`artifacts/hero-now/frame.png`, exactly half the plate's
 * 1672 x 941). The camera is shared, so the two are directly comparable in
 * pixels and no camera has to be solved to compare a rim to a rim.
 *
 * On the near arc the rim shows as one band running from its crest — which is
 * what silhouettes against the water, because the inner face behind it faces
 * away — down to where its outer flank meets the floor. Measured as the run from
 * the luminance step at the water's edge to the contact-shadow minimum at the
 * foot, along five columns that cross the near rim cleanly in both images:
 *
 *   plate column   plate span   frame column   frame span   x2    ratio
 *        400          144            200           63        126   1.14
 *        450          144            225           64        128   1.13
 *        500          141            250           62        124   1.14
 *        550          142            275           63        126   1.13
 *        350          145            175           57        114   1.27
 *
 * The last row is dropped: at frame x=175 the pebble course crosses the rim's
 * foot and the shadow minimum being measured is the pebbles', not the rim's. The
 * four that stand give **1.135**, and they agree to a percent, which is what says
 * the two images are registered well enough for this to mean anything.
 *
 * With the camera 24 degrees above horizontal, a section offset of `a` outward
 * and `h` up projects to `a*sin24 + h*cos24 = 0.407a + 0.914h` of screen. The
 * render is `a = 31.75, h = 55`, which is 63.2 mm of screen and measures 126
 * plate-equivalent pixels: 1.99 px per screen-millimetre. So the plate's rim is
 * **`0.407a + 0.914h = 71.7 mm`**, and that is the one hard constraint the plate
 * hands over. It is a line, not a point — every section on it draws the same
 * band, from 88 x 55 mm to 176 x 40 — and the proportion has to come from
 * looking:
 *
 *   - the plate's rim has **no dark band under its lip**. The render's flank
 *     turns under and its last 8 px go to luminance 41 against a 146 floor; the
 *     plate's falls smoothly 217 -> 104 and recovers to 147 over the 35 px of
 *     cast shadow beyond the foot. There is no overhang, which caps the section
 *     at `widthToHeight = 2` before anything else is decided.
 *   - the top is broad: the plate holds within 7 % of its peak luminance for
 *     36 px past the crest where the render holds for 16.
 *
 * At `widthToHeight = 2.8` and `crestHeight_m = 0.046` the constraint reads
 * `0.407*64.4 + 0.914*46 = 68.2 mm` against the measured 71.7 — 4.9 % short,
 * which is inside the error in calling where the plate's foot is against a
 * contact shadow. The section that lands it exactly at the same proportion is
 * 135 x 48 mm, and the two differ by a quarter of a solver cell.
 *
 * ## What it costs and what it moves
 *
 * A 128.8 mm footprint on a 46 mm crest makes a circle of radius 68.1 mm bedded
 * 22.1 mm below the plaster. Against the 63.5 x 55 mm it replaces: the footprint
 * doubles, the crest comes down 9 mm, and the ratio goes from 1.15 to 2.80.
 *
 * `HERO_GARDEN_VESSEL.rimHalfWidth_m` is derived from this section's
 * `groundHalfWidth_m`, so the flat plaster band the rim is bedded into widens
 * with it and the pond's wetted plan comes in about 33 mm per side. Containment
 * does not move at all: `pondVesselWaterline` is `groundHeight - relief - below`
 * and the plaster outside the rail is `groundHeight + terrace + relief` for every
 * width. Measured on the baked grid either way: the lowest plaster outside the rim
 * is 298.42 mm and the still waterline is 257.50 mm, so the clearance is 40.92 mm
 * before the change and 40.92 mm after it.
 *
 * The crest is deliberately **not** at 50 mm. The solver lattice is 25 mm with
 * the plaster at 300 mm, which is a cell boundary exactly; 46 mm puts the crest
 * 4 mm *under* the 350 mm boundary rather than on it or just over it. Just over
 * is the bad side — it lays one extra 25 mm cell layer along the whole rim for
 * the sake of a few millimetres of stone — and 4 mm under, with the crest
 * modulating plus or minus 8 mm, means the high lobes reach that layer and the
 * rest do not. That is the crest undulation reading as a crest undulation
 * instead of as a lattice artifact.
 */
export const SWEPT_COPING_POND_BULLNOSE: SweptCopingForm = Object.freeze({
  crestHeight_m: 0.046,
  widthToHeight: 2.8,
  // Seven per cent of a 68.1 mm radius is 4.8 mm of swell, which is what the old
  // form's 14 % of 36.7 mm was in millimetres. See `sectionVariation`.
  sectionVariation: 0.07,
  crestVariation: 0.11,
  // Eleven lobes on a 2.9 m run is a 264 mm period — a good fraction of the
  // pond's 450 mm radius, which is the wavelength the crest wanders on in the
  // plate. Deliberately nowhere near the 40-80 mm band: a rim that undulated
  // there would read as slumped rather than as hand-formed.
  variationLobes: 11,
  relief_m: 0.0015,
  // Four stations a lobe at the stride below, which is what the relief's own
  // doc comment asks for and what stops it reading as a caterpillar.
  reliefLobes: 17,
  // The rail is walked one point in five. See the module header: with round-cone
  // segments the joint angle is free, so this is bounded by chord sag (0.89 mm
  // at the 36.4 mm chords this leaves) and by the relief's band limit, not by
  // the shading. It went from four to five with the section: a record whose
  // radius nearly doubled reaches into more bricks and takes the busiest rim
  // brick from 14 to 18 against a ceiling of 64 whose overflow is silent. Five
  // puts it back at 15. See the table in the module header.
  segmentStride: 5,
});

export interface SweptCopingSpec extends SweptCopingForm {
  /** Id prefix. Node ids must be unique across the scene. */
  readonly key: string;
  /**
   * The rail, as a closed polyline in world metres. The last point joins the
   * first; do not repeat it.
   *
   * `pondVesselPlanCurve` hands out exactly this, which is the whole reason it
   * is an export — the coping, the pebble beds and the stepping path all follow
   * one outline rather than three that drift.
   */
  readonly rail: readonly (readonly [number, number])[];
  /**
   * The ground the coping is set into, in metres above the container floor.
   *
   * A query, not a heightfield: callers pass `terrainHeightAt(scene.terrain, …)`
   * — the baked grid the renderer draws and the solver collides against — so the
   * rim is bedded into the ground that actually exists rather than into a
   * re-derivation of it.
   */
  readonly groundHeightAt: (x_m: number, z_m: number) => number;
  /**
   * The selectable object.
   *
   * It used to pick the surface closure as well: the name was matched against
   * a regular expression, `coping` landed on the `stone` procedural material,
   * and renaming the group restyled the rim. `material.surface` says it
   * directly now, so this is a name again.
   */
  readonly group?: string;
  readonly material: SceneryMaterial;
  /** The only entropy. Any integer. */
  readonly seed: number;
  /**
   * The finest voxel this rim will be drawn at, in metres.
   *
   * Placement rather than form — see `SceneryGeneratorRequest.detailCellSize_m`.
   * The rim has exactly one feature whose legibility is decided by the leaf and
   * not by the plate, and it is the chord sag of the walked rail; see
   * {@link sweptCopingSegmentStride}. Absent is
   * {@link SWEPT_COPING_DEFAULT_LEAF_SIZE_M} and reproduces the 25 mm rim
   * exactly.
   */
  readonly leafSize_m?: number;
}

/**
 * The leaf a rim is drawn at when nobody says, in metres. `HERO_GARDEN_CELL_M`.
 */
export const SWEPT_COPING_DEFAULT_LEAF_SIZE_M = 0.025;

/**
 * How much of a leaf the walked rail may cut off the outline.
 *
 * Half. A silhouette error under half a leaf cannot move a voxelized outline —
 * the surface the primary returns is the cell's face, so an error that never
 * crosses a cell boundary is not drawable — and above about one leaf it is a
 * visible flat on the near arc. Half is the largest budget that is still
 * provably invisible, which is the right side of the line to be on for a number
 * whose only cost is records.
 */
const SWEPT_COPING_SAG_LEAVES = 0.5;

/**
 * Largest perpendicular distance from the walked chords to the rail they skip,
 * in metres.
 *
 * Measured off the rail rather than modelled from a curvature: a coping follows
 * whatever outline it is handed, and a species that assumed a circle would be
 * wrong on exactly the lobed plans this one exists for. This is the silhouette
 * error the stride actually costs, at the actual pond.
 */
export function sweptCopingChordSag_m(rail: readonly (readonly [number, number])[], stride: number): number {
  if (stride <= 1) return 0;
  const count = Math.floor(rail.length / stride);
  let worst = 0;
  for (let index = 0; index < count; index += 1) {
    const [ax, az] = rail[index * stride];
    const [bx, bz] = rail[((index + 1) % count) * stride % rail.length];
    const dx = bx - ax, dz = bz - az;
    const lengthSquared = dx * dx + dz * dz;
    if (!(lengthSquared > 0)) continue;
    for (let step = 1; step < stride; step += 1) {
      const [px, pz] = rail[(index * stride + step) % rail.length];
      const t = Math.min(1, Math.max(0, ((px - ax) * dx + (pz - az) * dz) / lengthSquared));
      worst = Math.max(worst, Math.hypot(px - ax - t * dx, pz - az - t * dz));
    }
  }
  return worst;
}

/**
 * The stride this rail may be walked at for this leaf.
 *
 * **The one number on this rim that the lattice decides.** The authored stride
 * is a ceiling and this only ever comes down from it, so a scene that says
 * nothing draws exactly the rim it drew before. What moves it is the sag test
 * above: at the hero pond and `segmentStride: 5` the walked chords cut 0.89 mm
 * off the outline, which is a seventh of a 6.25 mm leaf and invisible, and a
 * fifty-seventh of a 0.78 mm one is not — at the bottom of the ladder that same
 * 0.89 mm is 1.1 leaves and draws as a flat on the near arc.
 *
 * Measured on `HERO_GARDEN_VESSEL`'s 336-point rail:
 *
 *   leaf         budget    stride   sag       records
 *   25 mm       12.50 mm     5      0.89 mm      67
 *   6.25 mm      3.13 mm     5      0.89 mm      67
 *   3.125 mm     1.56 mm     5      0.89 mm      67
 *   1.5625 mm    0.78 mm     4      0.57 mm      84
 *   0.78125 mm   0.39 mm     3      0.32 mm     112
 *
 * So it costs nothing at all until the leaf is under two millimetres, which is
 * the property that lets it land without touching the depth-0 ratchets. It is
 * also the *only* leaf-following lever this species has and it is a records
 * lever, which is the finding rather than the fix: the rim's own grain is
 * {@link SWEPT_COPING_GRAIN}, it is a field, and it still has nowhere to live.
 */
export function sweptCopingSegmentStride(
  rail: readonly (readonly [number, number])[],
  authoredStride: number,
  leafSize_m: number = SWEPT_COPING_DEFAULT_LEAF_SIZE_M,
): number {
  if (!(leafSize_m > 0)) throw new RangeError("Swept coping leaf size must be positive");
  const budget_m = SWEPT_COPING_SAG_LEAVES * leafSize_m;
  let stride = Math.max(1, Math.floor(authoredStride));
  while (stride > 1 && sweptCopingChordSag_m(rail, stride) > budget_m) stride -= 1;
  return stride;
}

/** Integer avalanche hash in [0, 1). The same coping on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

const hashSigned = (n: number): number => 2 * hash01(n) - 1;

/**
 * A closed Catmull-Rom through `values`, evaluated at a fraction of the way
 * round.
 *
 * Written out here rather than imported from the pond, because a coping that
 * imported the pond's internals would be a pond part with a general name. Rule
 * one of `lib/voxel-scenery/README.md`: depend on geometry, never on a scene.
 */
function periodicSpline(values: readonly number[], turn: number): number {
  const count = values.length;
  const position = ((turn % 1) + 1) % 1 * count;
  const index = Math.floor(position), t = position - index, t2 = t * t, t3 = t2 * t;
  const at = (offset: number) => values[((index + offset) % count + count) % count];
  const a = at(-1), b = at(0), c = at(1), d = at(2);
  return .5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - 3 * c + d - a) * t3);
}

function modulation(count: number, seed: number): number[] {
  const values: number[] = [];
  for (let index = 0; index < count; index += 1) values.push(hashSigned(seed + 61 * index));
  return values;
}

/**
 * The section a coping form makes, in metres.
 *
 * Exported because it is the answer to "how wide is this rim and how far does it
 * hang over", which the generators that place things against a coping — a pebble
 * course at its foot, a stepping path over its lip — need without re-deriving
 * the trigonometry. `HERO_GARDEN_VESSEL.rimHalfWidth_m` is `groundHalfWidth_m`,
 * so widening a section here widens the flat plaster band the rim is bedded into
 * and moves every pebble course that offsets from it; that is the intent, and it
 * is why the two numbers are one derivation rather than two constants.
 *
 * `axisLift_m` is signed. Positive is a centre standing out of the plaster and a
 * section that undercuts; negative is a centre bedded under it and a section that
 * is a circular cap. Callers that only want "how deep is this thing buried" want
 * `radius_m - crestHeight_m`, which is `-axisLift_m`.
 */
export function sweptCopingSection(form: Pick<SweptCopingForm, "crestHeight_m" | "widthToHeight">): {
  radius_m: number; axisLift_m: number; groundHalfWidth_m: number; rollBack_m: number;
} {
  const { crestHeight_m, widthToHeight } = form;
  if (!(crestHeight_m > 0)) throw new RangeError("Swept coping crest height must be positive");
  if (!(widthToHeight > 0 && widthToHeight <= SWEPT_COPING_MAXIMUM_WIDTH_TO_HEIGHT)) {
    throw new RangeError(`Swept coping width-to-height must be in (0, ${SWEPT_COPING_MAXIMUM_WIDTH_TO_HEIGHT}]`);
  }
  const groundHalfWidth_m = 0.5 * widthToHeight * crestHeight_m;
  // The circle through the crest at (0, crest) and the ground line at
  // (groundHalfWidth, 0), centred on the rail: `(radius - crest)^2 +
  // groundHalfWidth^2 = radius^2`. `axisLift_m` is signed and is the whole shape
  // decision — positive lifts the centre out of the plaster and the section
  // undercuts, negative beds it and the section is a cap. It is exactly zero at
  // `widthToHeight = 2`, where the identity collapses to `radius = crest`.
  const radius_m = 0.5 * (groundHalfWidth_m * groundHalfWidth_m / crestHeight_m + crestHeight_m);
  const axisLift_m = crestHeight_m - radius_m;
  // How far the widest point of the section hangs out past the ground line — a
  // question about the *visible* solid, so a section whose widest point is
  // buried has none rather than having a negative one. `rollBack_m` is what a
  // generator placing something against the rim's foot needs, and a pebble bedded
  // against a buried bulge would be bedded against nothing.
  const rollBack_m = axisLift_m > 0 ? radius_m - groundHalfWidth_m : 0;
  return { radius_m, axisLift_m, groundHalfWidth_m, rollBack_m };
}

/**
 * One station on the rail: where the section's centre sits and how wide it is.
 *
 * The whole of a coping's *shape* is this list. Everything below it is a choice
 * about which records express that shape, and the two are deliberately not the
 * same decision — the emitter has already been a truncated-cone chain, a
 * tapered-sweep cluster and a capsule chain while the centreline underneath
 * stayed the same numbers.
 */
export interface SweptCopingStation {
  /** Centre of the circular section, in world metres. */
  readonly at: Vec3;
  /** Section radius there. */
  readonly radius: number;
  /** Fraction of a turn round the rail, so a caller can stay in step with the run. */
  readonly turn: number;
}

/**
 * The centreline and radius profile: a coping's shape, before anything decides
 * how to draw it.
 *
 * ### The seam
 *
 * This is the half of the generator that is geometry. `sweptCopingNodes` is the
 * half that is records, and it may be replaced wholesale — by a tapered sweep
 * along a control polyline as a single SDF record, which is what this run wants
 * and what would retire the chain — without a line of this function moving.
 * Anything that needs a coping's *form* rather than its primitives (a pebble
 * course banking against its foot, a test asserting the crest clears the water)
 * should ask here rather than reading records back.
 *
 * The rail is resampled by `segmentStride` and the section evaluated at each
 * surviving point. Every station's ground is queried where that station is, so a
 * rim over lumpy plaster follows the lumps rather than floating over them —
 * which is also why the crest is authored *above the ground* and not as a world
 * height.
 */
export function sweptCopingStations(spec: SweptCopingSpec): readonly SweptCopingStation[] {
  const { rail, seed, relief_m } = spec;
  if (rail.length < 3) throw new RangeError("Swept coping needs a rail of at least three points");
  for (const [count, label] of [[spec.variationLobes, "variation"], [spec.reliefLobes, "relief"]] as const) {
    if (!Number.isInteger(count) || count < 3) throw new RangeError(`Swept coping needs at least three ${label} lobes`);
  }
  const authoredStride = spec.segmentStride ?? 1;
  if (!Number.isInteger(authoredStride) || authoredStride < 1) {
    throw new RangeError("Swept coping segment stride must be a positive integer");
  }
  // The authored stride is a ceiling, not the answer: a finer leaf resolves the
  // chord sag it costs. See `sweptCopingSegmentStride`.
  const stride = sweptCopingSegmentStride(rail, authoredStride, spec.leafSize_m);
  const base = sweptCopingSection(spec);

  const radiusModulation = modulation(spec.variationLobes, seed);
  const crestModulation = modulation(spec.variationLobes, seed ^ 0x3d1f_0977);
  const radiusRelief = modulation(spec.reliefLobes, seed ^ 0x51ed_2701);
  const crestRelief = modulation(spec.reliefLobes, seed ^ 0x6a09_e667);

  // Stations first, runs between them second: the section at a joint has to be
  // the same number for the segment arriving and the segment leaving, or the
  // chain steps whatever primitive it is spelled with.
  const count = Math.floor(rail.length / stride);
  if (count < 3) throw new RangeError("Swept coping segment stride leaves fewer than three nodes");
  return Array.from({ length: count }, (_unused, index) => {
    const [x, z] = rail[index * stride];
    // The rail's own fraction of a turn, not the station index's. They agree
    // when the stride is one, and when it is not, this is the one that keeps the
    // modulation a function of *where round the pond* a station is rather than
    // of how coarsely the rail was walked.
    const turn = (index * stride) / rail.length;
    const radius = base.radius_m * (1 + spec.sectionVariation * periodicSpline(radiusModulation, turn))
      + relief_m * periodicSpline(radiusRelief, turn);
    const crest = spec.crestHeight_m * (1 + spec.crestVariation * periodicSpline(crestModulation, turn))
      + relief_m * periodicSpline(crestRelief, turn);
    // The centre is placed off *this station's* radius, so the crest lands
    // exactly where the crest modulation says and the section swell shows as a
    // wider footprint bedded deeper rather than as a taller rim.
    //
    // It used to be placed off the authored radius, which let a swell push the
    // crest up with it — deliberate then, because the section undercut and
    // holding the axis was what kept the widest point above the plaster. With a
    // bedded cap there is no undercut to protect and the coupling is only a
    // defect: at the hero form's 68.1 mm radius a 7 % swell would have moved the
    // crest by 4.8 mm on a 46 mm rim, so a tenth of the height would have been
    // an accident of the width modulation rather than the crest's own.
    const ground = spec.groundHeightAt(x, z);
    return { at: V(x, ground + crest - radius, z), radius: Math.max(1e-4, radius), turn };
  });
}

/**
 * Grow one coping along a rail.
 *
 * Shape from `sweptCopingStations`, records from here: one round-cone SDF per
 * station pair, consecutive ones sharing their endpoint sphere. A round cone is
 * the convex hull of its two end spheres and its lateral surface is tangent to
 * both, so a chain of them is C1 through every joint whatever angle the rail
 * turns through — there is no crease to hide and no planar cap for a voxel owner
 * to leak. That property is the reason the rail can be walked as coarsely as it
 * is; see the module header.
 */
export function sweptCopingNodes(spec: SweptCopingSpec): SceneryNode[] {
  const { key } = spec;
  const group = spec.group ?? "stone-coping";
  const stations = sweptCopingStations(spec);
  const count = stations.length;

  const emitted: SceneryNode[] = [];
  for (let index = 0; index < count; index += 1) {
    const from = stations[index], to = stations[(index + 1) % count];
    const axis = V(to.at.x - from.at.x, to.at.y - from.at.y, to.at.z - from.at.z);
    const halfHeight = .5 * Math.hypot(axis.x, axis.y, axis.z);
    if (!(halfHeight > 0)) throw new Error(`Swept coping run ${index} has zero length`);
    emitted.push({
      kind: "cone",
      id: `${key}/run-${index}`,
      group,
      tags: ["coping", "stone"],
      place: {
        units: "metres",
        position: V(.5 * (from.at.x + to.at.x), .5 * (from.at.y + to.at.y), .5 * (from.at.z + to.at.z)),
        orientation: alongAxis(axis),
      },
      baseRadius: from.radius,
      topRadius: to.radius,
      halfHeight,
      roundedEnds: true,
      material: spec.material,
    });
  }
  return emitted;
}
