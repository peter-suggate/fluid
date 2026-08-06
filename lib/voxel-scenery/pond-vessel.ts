import { MAX_TERRAIN_GRID_SAMPLES, MIN_TERRAIN_GRID_SIZE, type TerrainGrid } from "../terrain";

/**
 * The pond vessel: basin, coping ring and surrounding ground, generated from a
 * plan curve rather than authored as a heightfield.
 *
 * Why this is terrain and not scenery. The scene has three authorities and only
 * one of them is visible to the water: `VoxelSceneSource` is container ∪ terrain
 * ∪ rigid bodies, so a coping spelled as scenery primitives would be a rim the
 * water pours straight through. Spelling the whole vessel as terrain gives one
 * authority for both the solid the water rests in and the surface that renders,
 * and the waterline then cannot disagree with the geometry it meets.
 *
 * What that costs is the overhang. A heightfield is a function of (x, z), so the
 * coping's outer lip cannot roll back under itself; what it can do is a bullnose
 * whose crest is round and whose feet are tangent, which is what the reference's
 * rim reads as from any camera that is not below the lip.
 *
 * The plan curve is the load-bearing export. The pebble beds, the stepping-stone
 * path and the bank planting all want to follow the same outline, and an outline
 * each of them re-derived would drift from this one the moment a control point
 * moved. `pondVesselPlanCurve` hands out the polyline this bake used, so those
 * generators offset a curve instead of guessing one.
 *
 * ## The interior is one curve, not a wall and a floor
 *
 * A vessel whose face drops its whole depth over a short run and then stops is a
 * cylinder with a lid on it, and it renders as one: a hard ring of banding where
 * the face meets a floor that is dead flat for the next third of a metre. What
 * the reference shows is a shallow, wide basin — a soft continuous fall from the
 * kerb into a gently dished floor, deepest in the middle and a finger deep at
 * the sides.
 *
 * So the drop is shared. `innerFace_m` carries most of it over its own run and
 * `floorDish` spreads the rest across the whole floor, as two smoothsteps summed
 * rather than sequenced, and the sum is tangent where it leaves the plaster and
 * tangent again where it arrives at the deepest point. That is a shape decision
 * and also the cheapest performance decision available here: the grid carries
 * one Lipschitz slope bound for the *whole scene* and the dry render's march
 * steps by it everywhere, so a face that is 45 % less steep is paid back on the
 * flat plaster metres away. Measured on the hero pond, 9.41 to 5.32, and to 4.40
 * once the face was allowed a crease at the top (`pondVesselInnerFaceProfile`).
 *
 * Determinism is the same contract the procedural tree keeps: the seed is the
 * only entropy, it enters through an integer avalanche hash, and the same spec
 * bakes the same grid on every rebuild — the sparse publication cache is keyed
 * on the geometry that comes out of here.
 */

/**
 * A formed wall's cross-section: what the eye reads as "this was built".
 *
 * Authored as four lengths in metres rather than as a band width, because each
 * one controls exactly one feature — see {@link PondVesselTerrace.wall} for why
 * that matters and what it replaces.
 */
export interface WallProfile {
  /** Radius of the roll where the plateau turns over into the face. */
  readonly crestRadius_m: number;
  /** Radius of the fillet where the face meets the ground. */
  readonly footRadius_m: number;
  /**
   * How far the face leans out from vertical, in radians.
   *
   * Must be positive, and that is the medium rather than a style choice: a plumb
   * face is multivalued in plan distance and a heightfield cannot carry one. Real
   * formed walls are battered anyway, or the shuttering could not be struck.
   */
  readonly batter_rad: number;
}

/**
 * Height above the outside ground at signed plan distance `d`, where `d = 0` is
 * the **crest line** — the edge of the plateau — and `d` runs positive outward.
 *
 * Same convention as the coping, whose rail is its crest centreline, so a bed
 * outline and a coping rail mean the same thing and can be offset from each
 * other without a conversion.
 *
 * Reading outward from the plateau: a circular roll of `crestRadius_m`, a
 * straight face leaning `batter_rad` off vertical, a circular fillet of
 * `footRadius_m`, then the ground. Both arcs are tangent to the face and to
 * their own flat by construction, so the surface is C¹ everywhere and what the
 * eye reads as an edge is the *curvature* jump at each tangent point. That is
 * what a formed edge is — a casting has no zero-radius arris either.
 *
 * Drawn and measured by `tools/shape-lab.ts wall`, which is where the numbers
 * for the hero garden's beds were chosen: 184 mm of flat top and an 82 degree
 * face, against a smoothstep that has neither at any setting.
 */
export function wallProfile_m(profile: WallProfile, rise_m: number, d: number): number {
  const batter = Math.max(0.02, profile.batter_rad);
  const cos = Math.cos(batter), sin = Math.sin(batter), tan = Math.tan(batter);
  // The two arcs eat into the rise by `r(1 - sin)` each; whatever is left is the
  // straight face. Radii too fat for the rise are shared down rather than
  // rejected, so a short wall degrades to two tangent arcs and no flat face.
  const appetite = (profile.crestRadius_m + profile.footRadius_m) * (1 - sin);
  const fit = appetite > rise_m ? rise_m / appetite : 1;
  const crest = profile.crestRadius_m * fit;
  const foot = profile.footRadius_m * fit;
  const crestCentreY = rise_m - crest;
  const crestTangent = { d: crest * cos, y: crestCentreY + crest * sin };
  const footTangentY = foot * (1 - sin);
  const footTangentD = crestTangent.d + (crestTangent.y - footTangentY) * tan;
  const footCentreD = footTangentD + foot * cos;
  if (d <= 0) return rise_m;
  if (d >= footCentreD) return 0;
  if (d <= crestTangent.d) return crestCentreY + Math.sqrt(Math.max(0, crest * crest - d * d));
  if (d >= footTangentD) {
    const dx = d - footCentreD;
    return foot - Math.sqrt(Math.max(0, foot * foot - dx * dx));
  }
  return crestTangent.y - (d - crestTangent.d) / tan;
}

/** How far out a wall profile reaches from its crest line, in metres. */
export function wallProfileRun_m(profile: WallProfile, rise_m: number): number {
  let run = 0;
  // Walked rather than solved: the tangent-point algebra is already in the
  // profile and a second copy of it here is a second chance to disagree.
  while (run < 4 * rise_m + profile.crestRadius_m + profile.footRadius_m) {
    if (wallProfile_m(profile, rise_m, run) <= 0) return run;
    run += 0.0005;
  }
  return run;
}

export interface PondVesselTerrace {
  /** Footprint centre in world metres, container-centred like every other feature. */
  readonly center_m: readonly [number, number];
  /** Elliptical footprint semi-axes. */
  readonly radius_m: readonly [number, number];
  /** Rise above the surrounding ground. */
  readonly height_m: number;
  readonly rotation_rad?: number;
  /** Fraction of the radius that is a flat plateau, in [0, 1). */
  readonly flat?: number;
  /**
   * Wobble on the footprint, as a fraction of the radius, and how many lobes it
   * runs on. A bed cast as a bare ellipse reads as a machine part; the pond's
   * own outline wanders for the same reason and by the same construction.
   */
  readonly wobble?: number;
  readonly lobes?: number;
  /**
   * The rise taken as a **formed wall** rather than as a mound: a flat top, a
   * rolled crest, a steep battered face and a fillet at its foot.
   *
   * Present is the whole difference between a raised bed and a dune, and the
   * reason it is a profile rather than another parameter of the smoothstep is
   * worth stating, because two art passes were lost to it. `ramp` is
   * `t^2(3-2t)`: its derivative is zero at *both* ends, so it has no flat and no
   * crest and there is exactly one knob — the band width — which moves the
   * crest, the face and the foot together. Narrowing it does not make a wall. It
   * makes a narrower dome. No amount of tuning inside that function could
   * produce the shape the reference shows, and measuring the slope at its
   * steepest point said nothing about whether a flat top existed.
   *
   * {@link wallProfile_m} is a piecewise function of the signed plan distance
   * instead, and each of its lengths controls one visible feature and nothing
   * else. It is the same construction `pondVesselHeightAt` already uses for the
   * pond, which composes a named inner-face profile with a named dish rather
   * than asking one smoothstep to be both.
   */
  readonly wall?: WallProfile;
  /**
   * The rise taken as a *wall* over this much run, in metres, instead of as a
   * mound over everything outside {@link flat}.
   *
   * A terrace was a swelling of the ground, and on the hero garden that is what
   * it looked like: the bonsai's plateau took its 75 mm over 308 mm of run, and
   * because a smoothstep concentrates its slope in the middle, what the frame
   * showed was a soft mound with one steep patch across it — measured 307 to
   * 352 mm over 44 mm of run — which reads as a fault line in the plaster rather
   * than as an edge anything was built to.
   *
   * The reference's raised beds are *formed*: a flat top, a defined crest, and a
   * short steep face down to the ground. That is one number — how much run the
   * whole rise is allowed — and the smoothstep already has the right shape for
   * it: a convex roll at the crest, a near-straight face, a concave fillet at the
   * foot. It only ever needed to be given a hand's breadth instead of a third of
   * a metre.
   *
   * It also voxelizes better, which is not a coincidence. A 20-degree ramp
   * crosses a voxel boundary every few centimetres and each crossing is a face
   * normal flipping, so a soft terrace comes back as a contour map; a face near
   * enough vertical is parallel to the voxel faces it is drawn into and a flat
   * top is parallel to the others.
   *
   * Measured against the ellipse's mean radius, so a terrace keeps one face
   * width all the way round rather than a wider one on its long axis. Omitted
   * keeps the mound, which is what a bank or a dune wants.
   */
  readonly faceRun_m?: number;
}

/**
 * A shallow sector of the inner face: the shore the stepping stones wade in over.
 *
 * The reference is a vessel on three sides and a beach on the fourth — the discs
 * enter the water on a curve, the pebbles run down into it, and the whole left
 * of the pond is a graded shelf while the right is a wall. One inner-face run for
 * the whole ring cannot be both, and choosing either loses something real: a
 * ring-wide wall leaves the discs standing on the basin floor like drums and the
 * shore pebbles nowhere to sit, and a ring-wide ramp turns the pond back into
 * the dish it started as.
 *
 * So the run is a function of where round the pond you are. `width` is the
 * half-width of the sector in turns, and the transition is the same smoothstep
 * everything else here uses, which keeps the plan curve's own wander and this
 * sector from meeting at a hard join.
 */
export interface PondVesselBeach {
  /** Centre of the shore, in turns about the plan's centre. */
  readonly turn: number;
  /** Angular half-width, in turns. Past this the ring is back to its wall. */
  readonly width: number;
  /** The inner-face run over the sector, replacing `innerFace_m` there. */
  readonly innerFace_m: number;
}

export interface PondVesselSpec {
  /** Centre of the pond's plan, in world metres. */
  readonly center_m: readonly [number, number];
  /**
   * Semi-axes of the crest centreline before the wander is applied.
   *
   * Three things move the curve off this ellipse and they are deliberately
   * different in kind: the seeded grain and the long sway, both funded out of
   * `wobble`, and a beach's inward sweep, which is funded out of the beach.
   * See `pondVesselPlanCurve`.
   */
  readonly radius_m: readonly [number, number];
  /** Ground level around the pond, in metres above the container floor. */
  readonly groundHeight_m: number;
  /** The basin's *deepest* point below `groundHeight_m`, reached out in the dish. */
  readonly basinDepth_m: number;
  /** Coping crest above `groundHeight_m`. */
  readonly rimHeight_m: number;
  /** Half-width of the coping, measured across the crest centreline. */
  readonly rimHalfWidth_m: number;
  /**
   * Whether the coping is part of this heightfield at all.
   *
   * `bullnose`, the default, is the vessel as one surface: crest, faces and
   * ground from a single function of the plan distance. `flat` bakes the same
   * vessel with the crest omitted — plaster level to the plan curve, then the
   * inner face — so that a *solid* coping can be swept along `pondVesselPlanCurve`
   * and set into it.
   *
   * The reason to want that is the meeting, not the crest. `pondVesselCrestProfile`
   * is tangent at its foot on purpose, so the coping leaves the plaster with a
   * continuous normal and there is no line where the two meet; the reference has
   * a hard one. A heightfield cannot produce it — an honest union of a swept
   * solid with the ground produces it for free, and produces the reference's
   * slight roll-back under the lip along with it, which no function of (x, z)
   * can. See `lib/voxel-scenery/swept-coping.ts`.
   *
   * What does *not* change is containment: the coping crest stands above the
   * outer ground and the still waterline sits below it, so the wall the water
   * actually rests against is the outer plaster either way. Omitting the crest
   * lowers nothing the water can reach.
   */
  readonly crest?: "bullnose" | "flat" | "wall";
  /**
   * The coping's section when `crest` is `"wall"`: a **flat top with rolled
   * edges**, mirrored across the plan curve.
   *
   * This exists because a swept solid could not produce one. `sweptCopingNodes`
   * emits a chain of **round cones**, so its cross-section is a circular arc by
   * construction, and `widthToHeight` only chooses how much of that circle is
   * above ground. At the hero rim's 2.8 you see **142 degrees of a circle** — a
   * tube laid around the pond, which is exactly what the frame showed — and the
   * parameter cannot reach flat: it buys arc with footprint, and even at 5.2 it
   * is still 84 degrees on a rim 239 mm wide, a quarter of the pond.
   *
   * A coping is a narrow raised bed, so it takes the same {@link wallProfile_m}
   * the beds take, mirrored about the rail: plateau across the crown, a small
   * roll at each edge, steep flanks down to the plaster. The rolls are what stop
   * it being an arris; the flat between them is what stops it being a tube.
   *
   * The reason the crest left the heightfield in the first place was the
   * *meeting* — a tangent heightfield rim leaves the plaster with a continuous
   * normal and there is no line where the two join. This profile has a foot
   * fillet of a few millimetres rather than a tangent approach, so the line is
   * back: at the depth-2 leaf a 3 mm fillet is two voxels, which is as hard an
   * edge as the lattice can carry.
   */
  readonly crestWall?: WallProfile;
  /**
   * Run from the coping's inner foot down to the *edge* of the floor's dish.
   *
   * Not down to the deepest point: `floorDish` takes a share of the depth off
   * this face and spreads it across the floor, so the face carries
   * `(1 - floorDish) * basinDepth_m` over this run and the rest arrives as a
   * bowl. The two together are one continuous fall from the plaster to the
   * middle of the pond.
   */
  readonly innerFace_m: number;
  /**
   * Share of `basinDepth_m` the floor's dish carries rather than the inner face.
   *
   * Defaults to `POND_VESSEL_FLOOR_DISH`. Zero is the old vessel: the whole drop
   * on the face and a dead flat floor under it.
   */
  readonly floorDish?: number;
  /** A shallow sector of that face, if this pond has a shore. */
  readonly beach?: PondVesselBeach;
  /** Control points around the plan. More lobes, more places for it to wander. */
  readonly lobes: number;
  /** Radial wander of the plan, as a fraction of the local radius. */
  readonly wobble: number;
  /**
   * How much the coping's *section* swells and narrows as it runs, as fractions
   * of `rimHeight_m` and `rimHalfWidth_m`.
   *
   * This is the difference between a hand-formed rim and an extrusion. A plan
   * that wanders while the section stays constant still reads as machined,
   * because the eye picks up the unvarying crest line long before it picks up
   * the outline — which is exactly what the first version of this generator got
   * wrong, and what it had a test pinning in place.
   */
  readonly sectionHeightVariation: number;
  readonly sectionWidthVariation: number;
  /**
   * Amplitude of the hand-formed lumpiness carried over every surface here.
   *
   * Two octaves at centimetre scale, which is the finest relief a 7.5 mm lattice
   * can hold. The plaster *grain* below that is a material property and belongs
   * to the shading model, not to this heightfield — asking geometry for it would
   * only alias.
   */
  readonly relief_m: number;
  /** The only entropy. Any integer. */
  readonly seed: number;
  /** Raised ground outside the pond: the plateau a tree or a bench stands on. */
  readonly terraces?: readonly PondVesselTerrace[];
}

/**
 * Polyline samples per control point. Sixteen holds a 1 m pond inside a
 * millimetre, which is all a distance field sampled at 6.25 mm can use.
 *
 * A caller that *sweeps* the curve rather than measures distance to it wants
 * more, and for a different reason: a swept chain's shading normal is
 * discontinuous by the turn at each joint, so the resolution that settles the
 * outline is not the resolution that settles the highlight. Hence the override
 * on `pondVesselPlanCurve` — see `lib/voxel-scenery/swept-coping.ts`.
 */
export const PLAN_SAMPLES_PER_LOBE = 16;

/**
 * How much of the authored wander the long sway carries, against the per-lobe
 * grain.
 *
 * The two are the same amplitude and read completely differently, which is the
 * only reason this constant exists. Wander spent entirely on the grain is white
 * noise at the control rate: seven independent scales round the ring is a wiggle
 * with a wavelength of about two lobes, and at the hero's 8 % it comes back as a
 * *slightly noisy circle* — the outline never commits to being anywhere. The
 * reference commits: it bulges over half a side and narrows over the next, which
 * is energy at two or three cycles per turn and almost none above it.
 *
 * So most of the wander is moved down there and the grain is what is left. Not
 * all of it: a plan built only from two sinusoids is a rounded triangle or a
 * peanut, and reads as a *shape* rather than as something dug. The grain is what
 * stops the sway being recognisable as one.
 */
const PLAN_SWAY_SHARE = 0.62;

/**
 * The sway itself: two cycles and three, weighted, with a seeded phase each.
 *
 * Two is the lowest useful term — one cycle of radial modulation is, to first
 * order, the same ellipse with its centre moved, so it buys a translation rather
 * than a shape. Three is the highest that survives being read at the hero's
 * seven lobes without turning back into grain. The weights sum to one so the
 * sway's amplitude is exactly the share of `wobble` it was given.
 *
 * Evaluated continuously rather than at the control points, and that is
 * load-bearing: three cycles sampled at seven lobes is 2.33 samples a cycle,
 * under the two the Catmull-Rom would need, so a sway laid into the control
 * values would alias into something the seed did not ask for.
 */
const PLAN_SWAY_CYCLES = Object.freeze([
  Object.freeze({ cycles: 2, weight: 0.58 }),
  Object.freeze({ cycles: 3, weight: 0.42 }),
]);

/**
 * How far a beach draws the plan in, as a fraction of the local radius.
 *
 * A shore is not only a shallower inner face. The reference's pond is a vessel
 * on three sides and on the fourth the kerb *sweeps inward* and leaves a bank
 * with stones on it — the outline is narrower exactly where the ground is wider.
 * Deriving that from `beach` rather than authoring it separately is the same
 * argument `PondVesselBeach` already makes: the graded face, the pinch and the
 * bank are three consequences of one decision about where the near shore is.
 *
 * Twelve per cent is the reference's own sweep, and it draws the hero pond's
 * shore in by 60 mm — a coping's full width. It was six for one round, held
 * there by a stepping stone: the near plate was authored at a fixed position
 * 300 mm out and cleared the coping's inner foot by only 31 mm before any pinch
 * at all, so the sweep ate that clearance almost millimetre for millimetre and
 * ran out at eight per cent.
 *
 * What released it is that the wading chain stopped being authored and started
 * being *solved against this curve*: `stone-set.ts` now clamps every plate to a
 * minimum run off the rail, so the shore sweeping inward pushes the plates out
 * into the water with it instead of being overtaken. Measured across the pinch,
 * the whole chain's tightest plate moves from 46.5 mm of clearance at six per
 * cent to 39.4 at twelve — 7 mm spent for 30 mm of sweep, where before it was
 * millimetre for millimetre. It is still the binding constraint and it is no
 * longer a tight one.
 *
 * Gentle at this depth too: over the hero beach's 61-degree sector the rail's
 * tangent never turns more than about ten degrees off the ellipse's, so the
 * sweep is something the eye follows rather than a dent it catches on.
 */
const PLAN_BEACH_PINCH = 0.12;

/**
 * Share of the basin's own half-width the floor's dish spends reaching its
 * deepest.
 *
 * Short of the whole of it on purpose. The dish is defined on the distance in
 * from the plan curve, so it would otherwise reach its minimum only on the
 * plan's medial axis — a ridge, not a point — and every sample would land on the
 * approach to it. Stopping at nine tenths leaves a small true floor in the
 * middle at exactly `basinDepth_m`, which is what makes that number mean
 * something a test can assert.
 */
const POND_VESSEL_FLOOR_DISH_REACH = 0.9;

/**
 * The share of the depth a dished floor carries, when the spec does not say.
 *
 * The reference is a *shallow, wide basin*: the water is a hand deep in the
 * middle and a finger deep at the sides, and the surface under it is one
 * continuous curve from the kerb's inner foot to the centre. The vessel this
 * replaces put the whole 155 mm on 35 mm of face and left a dead flat floor,
 * which renders as a cylinder with a lid — and paid for the steepness twice,
 * because the grid carries one Lipschitz slope bound for the whole scene and the
 * dry march steps by it everywhere (`lib/voxel-scenery/swept-coping.ts`).
 *
 * Moving 45 % of the drop onto the floor buys both at once. Measured on the hero
 * pond: the bound falls from 9.41 to 5.32 and the face still drops 85 mm over
 * its 35 mm of run, so the water still meets a wall — 42.5 mm of it, at about 68
 * degrees, which is all of the face that is ever above the waterline. What
 * changed is the 112 mm that is under it.
 */
export const POND_VESSEL_FLOOR_DISH = 0.45;

/** Integer avalanche hash in [0, 1). The same pond on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

const hashSigned = (n: number): number => 2 * hash01(n) - 1;

/** Smoothstep on [0, 1], clamped outside it. */
function ramp(value: number): number {
  const t = Math.min(1, Math.max(0, value));
  return t * t * (3 - 2 * t);
}

/**
 * The coping's cross-section, as a fraction of `rimHeight_m` at a fraction `u`
 * of the way from the crest centreline to its foot.
 *
 * `(1 - u^2)^{3/2}` is the one profile in this family that is round at the crown
 * *and* tangent at the foot: a circle has the roundness but meets the ground
 * vertically, and a quartic bump has the tangency but flattens the crown into a
 * plateau. Curvature at the crown is 3 * rimHeight / rimHalfWidth^2, so the two
 * authored numbers are the only controls over how fat the rim reads.
 */
export function pondVesselCrestProfile(u: number): number {
  const t = Math.min(1, Math.max(0, u));
  return (1 - t * t) ** 1.5;
}

/**
 * How much of the inner face's fillet is spent easing into the floor, as a
 * fraction of its run.
 *
 * The face is otherwise straight, and both ends of that choice are deliberate.
 * See `pondVesselInnerFaceProfile`.
 */
const POND_VESSEL_INNER_FACE_FILLET = 0.35;

/** Straight run and fillet share the fall, so the straight part is this much steeper than the mean. */
const POND_VESSEL_INNER_FACE_SLOPE = 1 / (1 - POND_VESSEL_INNER_FACE_FILLET / 2);

/**
 * The inner face's fall, as a fraction of its share of the depth at a fraction
 * `u` of the way down its run.
 *
 * A straight face with a fillet at the bottom and **nothing at the top**, which
 * is the one thing here that is a break rather than a blend, and it buys two
 * unrelated things at once.
 *
 * The first is the line. Every other join in this vessel is tangent on purpose,
 * and for the coping's outer flank that is wrong — which is why the coping left
 * the heightfield (`lib/voxel-scenery/swept-coping.ts`). The *inner* foot had
 * the same problem and no such escape: a smoothstep leaves the plaster band with
 * a horizontal normal, so the flat band did not end anywhere, it just gradually
 * stopped being flat, and the whole interior rendered as one unbroken white
 * sheet. The reference has a hard shadow line where the kerb's inner foot meets
 * the floor. Starting the face at its full slope puts the normal 71 degrees over
 * in a single cell, and that line appears for free.
 *
 * The second is the slope, and it is the opposite of what you would guess.
 * Introducing a crease makes the face *shallower*, not steeper: a profile that
 * has to start and finish flat must overshoot the mean in the middle, and
 * smoothstep's peak is 1.5x it. A face that is allowed to begin at full slope
 * only has to ease at one end, and the peak falls to `1 / (1 - fillet / 2)` —
 * 1.21x at the 35 % fillet here. Since the terrain grid carries one Lipschitz
 * bound for the whole scene, that is a fifth off the marching cost of every
 * ray in the frame, paid for by a feature the reference wanted anyway.
 *
 * The fillet is at the bottom rather than the top because that is the end that
 * has to meet something: the floor's dish arrives tangent, and a face landing on
 * it at full slope would put a second crease where the reference has none.
 */
export function pondVesselInnerFaceProfile(u: number): number {
  const t = Math.min(1, Math.max(0, u));
  const fillet = POND_VESSEL_INNER_FACE_FILLET;
  const straight = 1 - fillet;
  if (t <= straight) return POND_VESSEL_INNER_FACE_SLOPE * t;
  // A quadratic ease over the fillet: value and slope match the straight run at
  // its start, and the slope reaches zero exactly at the foot.
  const v = (t - straight) / fillet;
  return POND_VESSEL_INNER_FACE_SLOPE * (straight + fillet * (v - .5 * v * v));
}

/**
 * Weight of a beach's sector at a turn about the plan's centre, in [0, 1].
 *
 * One definition, because two quantities are now functions of the same sector —
 * how far the inner face runs there, and how far the plan is drawn in — and a
 * second copy of the wrap-around arithmetic would be a shore whose slope and
 * whose outline part company the day one of them is edited.
 *
 * The separation is the shortest signed one on the circle, so a sector that
 * straddles the atan2 branch cut behaves like any other.
 */
function beachWeight(beach: PondVesselBeach | undefined, turn: number): number {
  if (!beach || !(beach.width > 0)) return 0;
  const separation = Math.abs((((turn - beach.turn + 0.5) % 1) + 1) % 1 - 0.5);
  return ramp(1 - separation / beach.width);
}

/** The sway, in [-1, 1]: the long bulges, at a phase the seed chooses. */
function planSway(turn: number, seed: number): number {
  let total = 0;
  for (const [index, term] of PLAN_SWAY_CYCLES.entries()) {
    total += term.weight * Math.sin(2 * Math.PI * (term.cycles * turn + hash01(seed + 0x5147_0000 + index)));
  }
  return total;
}

/**
 * The plan, as a closed polyline.
 *
 * Built in *polar* form: a periodic spline interpolates the seeded radial scale
 * around the ring, and the point itself is then evaluated exactly on the
 * ellipse. Splining the Cartesian control points instead — the obvious
 * construction, and the one this started as — makes `radius_m` a lie: a uniform
 * Catmull-Rom through points on a circle undershoots between them, by 3.4 mm on
 * a 400 mm eight-lobe ring and over 5 mm at the seven lobes the hero pond uses.
 * That is a systematic inward scallop nobody authored, on the order of a tenth
 * of the coping's width. In polar form a zero wobble reproduces the ellipse to
 * the last bit, and every millimetre of wander is one the seed asked for.
 *
 * The wobble moves the *curve* rather than the distance field, so the coping
 * keeps its section all the way round a wandering outline. Displacing the field
 * would thin the rim on every turn.
 *
 * ## Three wanders, and why they are not one
 *
 * The scale is a sum of three terms with different wavelengths, because an
 * outline that reads as *dug* has structure at every scale and one term can only
 * have it at one:
 *
 *  - the **grain**, a Catmull-Rom through `lobes` seeded values, which is where
 *    the wander used to live in its entirety;
 *  - the **sway**, two and three cycles a turn at a seeded phase, which is the
 *    bulging and narrowing the reference reads as, and which the grain cannot
 *    produce because white noise at seven control points has almost no energy
 *    that low (`PLAN_SWAY_SHARE`);
 *  - a **beach's pinch**, which is not wander at all — it is the outline
 *    agreeing with the shore that `PondVesselBeach` already declares.
 *
 * The first two share `wobble` between them, so the authored amplitude still
 * bounds the whole wander and a zero wobble is still exactly the ellipse. The
 * pinch is funded by the beach and is absent from a pond without one.
 *
 * The pinch is placed by the point's own **bearing** rather than by the curve
 * parameter, and the difference is not small: on a 0.52 x 0.38 ellipse the two
 * part company by up to 8.8 degrees, a quarter of the hero beach's half-width.
 * `beach.turn` is a bearing — it is compared against `pondVesselTurnAt` when the
 * inner face is graded — so the pinch has to be one too, or the outline would
 * sweep in beside the shore instead of along it. The bearing is available before
 * the scale is: scaling is radial in the ellipse's own frame, so every scale
 * puts the point on the same ray out of the centre.
 */
export function pondVesselPlanCurve(
  spec: PondVesselSpec,
  samplesPerLobe: number = PLAN_SAMPLES_PER_LOBE,
): readonly (readonly [number, number])[] {
  const { center_m: [cx, cz], radius_m: [rx, rz], lobes, wobble, seed } = spec;
  if (!Number.isInteger(samplesPerLobe) || samplesPerLobe < 2) throw new RangeError("Pond vessel plan needs at least two samples per lobe");
  if (!Number.isInteger(lobes) || lobes < 3) throw new RangeError("Pond vessel needs at least three plan lobes");
  if (!(rx > 0) || !(rz > 0)) throw new RangeError("Pond vessel radii must be positive");
  if (!(wobble >= 0 && wobble < 1)) throw new RangeError("Pond vessel wobble must be in [0, 1)");

  const grain: number[] = [];
  for (let index = 0; index < lobes; index += 1) grain.push(hashSigned(seed + 31 * index));
  const grainAt = (index: number) => grain[((index % lobes) + lobes) % lobes];

  const points: (readonly [number, number])[] = [];
  for (let index = 0; index < lobes; index += 1) {
    const a = grainAt(index - 1), b = grainAt(index), c = grainAt(index + 1), d = grainAt(index + 2);
    for (let step = 0; step < samplesPerLobe; step += 1) {
      const t = step / samplesPerLobe, t2 = t * t, t3 = t2 * t;
      // Uniform Catmull-Rom on the scalar. Control samples are exactly evenly
      // spaced in angle, so this is the natural parameterization rather than an
      // approximation of one, and a constant sequence interpolates to a constant.
      const grained = .5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - 3 * c + d - a) * t3);
      const turn = (index + t) / lobes;
      const angle = 2 * Math.PI * turn;
      const cos = Math.cos(angle), sin = Math.sin(angle);
      const bearing = Math.atan2(rz * sin, rx * cos) / (2 * Math.PI);
      const scale = 1
        + wobble * ((1 - PLAN_SWAY_SHARE) * grained + PLAN_SWAY_SHARE * planSway(turn, seed))
        - PLAN_BEACH_PINCH * beachWeight(spec.beach, bearing);
      points.push([cx + rx * scale * cos, cz + rz * scale * sin]);
    }
  }
  return points;
}

/** Squared distance from a point to a segment, and the segment's own crossing test. */
function segmentDistanceSquared(px: number, pz: number, ax: number, az: number, bx: number, bz: number): number {
  const dx = bx - ax, dz = bz - az;
  const lengthSquared = dx * dx + dz * dz;
  const t = lengthSquared > 0 ? Math.min(1, Math.max(0, ((px - ax) * dx + (pz - az) * dz) / lengthSquared)) : 0;
  const ox = px - (ax + t * dx), oz = pz - (az + t * dz);
  return ox * ox + oz * oz;
}

/**
 * Distance to the plan curve, negative inside it.
 *
 * Nearest-segment for the magnitude and a crossing count for the sign, which is
 * exact for the simple closed loop the curve is and costs one pass either way.
 *
 * "One pass" is 112 segments, and this is the single hottest arithmetic in the
 * whole scene build: `bakePondVesselTerrain` asks for it once per heightfield
 * node, which is 222 k nodes at the shipping leaf and **14.16 M** three
 * refinement rungs down. Measured at 1.70 us per call it was 80 % of a
 * derivation that took 38.3 s at that rung — the wait before the first frame,
 * not a frame cost. So above a threshold of queries the polyline earns an index
 * (`pondVesselPlanIndex`) and the scan stops being over all of it.
 *
 * The index changes the *set* each loop runs over and nothing else. Both
 * quantities survive that exactly rather than approximately, and for two
 * different reasons worth stating because they are what makes this safe:
 *
 *   - the magnitude is a **selection**. `nearestSquared` is the minimum of a
 *     set of f64 values, and a min taken over any subset that still contains
 *     the argmin is the identical f64. It is not a sum, so no term is dropped
 *     and no rounding is reordered. The grid's candidate rule is a proof that
 *     the argmin is present (see `buildPondVesselPlanIndex`).
 *   - the sign is a **parity**. Only segments that straddle the query's z can
 *     satisfy `(az > z) !== (bz > z)`; every other segment toggles nothing, so
 *     omitting them cannot change the count's parity, and toggling a boolean is
 *     order-independent.
 *
 * Verified rather than argued: `tests/pond-vessel.test.ts` walks the hero
 * vessel's footprint and requires `Object.is` against the linear scan, which is
 * bit equality including the sign of zero.
 */
export function pondVesselPlanDistance(
  curve: readonly (readonly [number, number])[],
  x: number,
  z: number,
): number {
  const index = pondVesselPlanIndex(curve);
  if (!index) return pondVesselPlanDistanceLinear(curve, x, z);

  const { points, count } = index;
  let nearestSquared = Infinity;
  const cellX = Math.floor((x - index.gridOriginX) * index.gridInversePitch);
  const cellZ = Math.floor((z - index.gridOriginZ) * index.gridInversePitch);
  if (cellX >= 0 && cellX < index.gridNx && cellZ >= 0 && cellZ < index.gridNz) {
    const cell = cellX + index.gridNx * cellZ;
    const end = index.cellStart[cell + 1];
    for (let slot = index.cellStart[cell]; slot < end; slot += 1) {
      const segment = index.cellItems[slot], next = segment + 1 < count ? segment + 1 : 0;
      const squared = segmentDistanceSquared(x, z,
        points[2 * segment], points[2 * segment + 1], points[2 * next], points[2 * next + 1]);
      if (squared < nearestSquared) nearestSquared = squared;
    }
  } else {
    // Outside the indexed box the candidate rule has nothing to say, so the
    // honest answer is the whole polyline. The box is three times the plan's
    // own extent, so no lattice this vessel is baked on reaches here.
    for (let segment = 0; segment < count; segment += 1) {
      const next = segment + 1 < count ? segment + 1 : 0;
      const squared = segmentDistanceSquared(x, z,
        points[2 * segment], points[2 * segment + 1], points[2 * next], points[2 * next + 1]);
      if (squared < nearestSquared) nearestSquared = squared;
    }
  }

  let inside = false;
  const band = Math.floor((z - index.bandOriginZ) * index.bandInversePitch);
  if (band >= 0 && band < index.bandCount) {
    const end = index.bandStart[band + 1];
    for (let slot = index.bandStart[band]; slot < end; slot += 1) {
      const segment = index.bandItems[slot], next = segment + 1 < count ? segment + 1 : 0;
      const az = points[2 * segment + 1], bz = points[2 * next + 1];
      if ((az > z) !== (bz > z)
        && x < points[2 * segment] + ((z - az) / (bz - az)) * (points[2 * next] - points[2 * segment])) {
        inside = !inside;
      }
    }
  }
  return (inside ? -1 : 1) * Math.sqrt(nearestSquared);
}

/** The scan the index replaces, and the oracle every exactness test compares against. */
export function pondVesselPlanDistanceLinear(
  curve: readonly (readonly [number, number])[],
  x: number,
  z: number,
): number {
  let nearestSquared = Infinity;
  let inside = false;
  for (let index = 0; index < curve.length; index += 1) {
    const [ax, az] = curve[index];
    const [bx, bz] = curve[(index + 1) % curve.length];
    const squared = segmentDistanceSquared(x, z, ax, az, bx, bz);
    if (squared < nearestSquared) nearestSquared = squared;
    if ((az > z) !== (bz > z) && x < ax + ((z - az) / (bz - az)) * (bx - ax)) inside = !inside;
  }
  return (inside ? -1 : 1) * Math.sqrt(nearestSquared);
}

/**
 * Segments a query can reach, bucketed twice because the two answers this
 * function computes prune on different things.
 *
 * The magnitude prunes on *proximity*: a uniform grid over the plan's
 * neighbourhood, each cell carrying the segments that could be nearest to any
 * point inside it. The sign prunes on *z alone*: a segment that does not
 * straddle the query's z cannot cross the ray, so a band index over z is both
 * necessary and sufficient, and it is far tighter than the grid — a horizontal
 * line meets a simple closed loop twice, so a band holds about three segments
 * against the grid cell's dozen.
 */
interface PondVesselPlanIndex {
  /** The polyline as flat f64 pairs. Same values, one indirection fewer. */
  readonly points: Float64Array;
  readonly count: number;
  readonly gridOriginX: number;
  readonly gridOriginZ: number;
  readonly gridInversePitch: number;
  readonly gridNx: number;
  readonly gridNz: number;
  /** CSR offsets into `cellItems`, one per cell plus a terminator. */
  readonly cellStart: Int32Array;
  readonly cellItems: Int32Array;
  readonly bandOriginZ: number;
  readonly bandInversePitch: number;
  readonly bandCount: number;
  readonly bandStart: Int32Array;
  readonly bandItems: Int32Array;
}

/**
 * Queries a polyline must take before it is worth indexing.
 *
 * `heroGardenPondPlanDistance` builds a fresh curve per call, so an index built
 * eagerly would be built once per query and never amortized. A bake asks
 * millions of times off one curve and a layout asks a handful off many, and
 * counting is the only thing that tells those apart from inside here. Below the
 * threshold the linear scan is what runs, and 64 scans of 112 segments is 12 us
 * — under the cost of building the index once.
 */
const POND_VESSEL_PLAN_INDEX_QUERIES = 64;

/** Cells the grid aims for. Enough that a cell holds a handful of segments. */
const POND_VESSEL_PLAN_INDEX_CELLS = 4096;

/**
 * Relative slack on the candidate rule.
 *
 * The rule is exact in exact arithmetic — see `buildPondVesselPlanIndex` — and
 * the bound it compares against is itself computed in f64, so it can land a
 * unit in the last place below the true maximum. Widening the acceptance by a
 * relative 1e-9 admits a segment or two more per cell and makes the containment
 * argument independent of that rounding. It cannot change any *result*: a
 * larger candidate set still yields the same minimum.
 */
const POND_VESSEL_PLAN_INDEX_SLACK = 1 + 1e-9;

/**
 * A one-entry inline cache in front of the `WeakMap`.
 *
 * The bake queries the same curve tens of millions of times in a row, so the
 * common case should be a reference comparison rather than a hash lookup.
 */
let lastPlanCurve: readonly (readonly [number, number])[] | undefined;
let lastPlanIndex: PondVesselPlanIndex | undefined;
const planIndexCache = new WeakMap<object, PondVesselPlanIndex | number>();

function pondVesselPlanIndex(
  curve: readonly (readonly [number, number])[],
): PondVesselPlanIndex | undefined {
  if (curve === lastPlanCurve) return lastPlanIndex;
  const entry = planIndexCache.get(curve);
  if (typeof entry === "object") {
    lastPlanCurve = curve;
    return lastPlanIndex = entry;
  }
  const queries = (entry ?? 0) + 1;
  if (queries < POND_VESSEL_PLAN_INDEX_QUERIES || curve.length < 3) {
    planIndexCache.set(curve, queries);
    return undefined;
  }
  const index = buildPondVesselPlanIndex(curve);
  planIndexCache.set(curve, index);
  lastPlanCurve = curve;
  return lastPlanIndex = index;
}

/**
 * The containment argument, which is the whole of why the index is exact.
 *
 * For a cell `C` let `bound = min over segments s of max over p in C of
 * dist(p, s)^2`. Distance to a segment is a convex function of the point, so
 * its maximum over a box is attained at a corner and four evaluations give that
 * maximum exactly. Then for every `p` in `C`:
 *
 *   - `dist(p, s*)^2 = min_s dist(p, s)^2 <= bound`, because the outer min is
 *     over the same segments and each term is no larger than its own maximum;
 *   - any `s` with `minDist(C, s)^2 > bound` has `dist(p, s)^2 > bound >=
 *     dist(p, s*)^2`, so it is not the argmin and dropping it is free.
 *
 * `minDist` is taken as the box-to-segment-bounding-box distance, which is a
 * lower bound on the true one — a looser bound admits more candidates and can
 * only ever be conservative, never wrong. The segments here are ~25 mm chords
 * of a smooth loop, so their boxes are tight and the looseness costs little.
 *
 * The far field prunes too, and that is the part worth saying out loud, because
 * "everything is equidistant far away" is the intuition that says it should
 * not. What matters is not the distance but the *spread*: seen from anywhere,
 * the far side of the loop is about the loop's own diameter further away than
 * the near side, and the bound only has to separate a cell's own slack —
 * roughly one cell diagonal plus one segment — from that diameter. So a distant
 * cell keeps the arc facing it and drops the rest.
 */
function buildPondVesselPlanIndex(
  curve: readonly (readonly [number, number])[],
): PondVesselPlanIndex {
  const count = curve.length;
  const points = new Float64Array(2 * count);
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let index = 0; index < count; index += 1) {
    const [x, z] = curve[index];
    points[2 * index] = x;
    points[2 * index + 1] = z;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  // Three times the plan's extent in each axis. Every lattice this vessel is
  // baked on spans its own container, which is comfortably inside that, and a
  // query beyond it falls back to the scan rather than answering wrongly.
  const spanX = Math.max(maxX - minX, 1e-6), spanZ = Math.max(maxZ - minZ, 1e-6);
  const margin = Math.max(spanX, spanZ);
  const gridOriginX = minX - margin, gridOriginZ = minZ - margin;
  const boxX = spanX + 2 * margin, boxZ = spanZ + 2 * margin;
  const pitch = Math.sqrt(boxX * boxZ / POND_VESSEL_PLAN_INDEX_CELLS);
  const gridNx = Math.max(1, Math.ceil(boxX / pitch)), gridNz = Math.max(1, Math.ceil(boxZ / pitch));
  /**
   * Slack on the cell's own extent, and the reason it is needed at all.
   *
   * A cell's candidates are computed from the box `[origin + k * pitch, origin +
   * (k + 1) * pitch]`, while a query lands in it through `floor((x - origin) *
   * (1 / pitch))`. Those two are the same partition in exact arithmetic and can
   * disagree by an ulp in f64, which would hand a point candidates computed for
   * the box next door. Widening the box a millionth of a cell makes the rule
   * hold for the point either partition assigns, and — like every other widening
   * here — it can only admit candidates, never drop one.
   */
  const cellSlack = pitch * 1e-6;

  const cellStart = new Int32Array(gridNx * gridNz + 1);
  const cellItems: number[] = [];
  // Segment bounding boxes, for the lower bound. Hoisted so the cell loop is
  // one pass over flat arrays rather than a repeated min/max of the polyline.
  const segMinX = new Float64Array(count), segMaxX = new Float64Array(count);
  const segMinZ = new Float64Array(count), segMaxZ = new Float64Array(count);
  for (let segment = 0; segment < count; segment += 1) {
    const next = segment + 1 < count ? segment + 1 : 0;
    const ax = points[2 * segment], az = points[2 * segment + 1];
    const bx = points[2 * next], bz = points[2 * next + 1];
    segMinX[segment] = Math.min(ax, bx); segMaxX[segment] = Math.max(ax, bx);
    segMinZ[segment] = Math.min(az, bz); segMaxZ[segment] = Math.max(az, bz);
  }

  for (let cellZ = 0; cellZ < gridNz; cellZ += 1) {
    const z0 = gridOriginZ + cellZ * pitch - cellSlack, z1 = z0 + pitch + 2 * cellSlack;
    for (let cellX = 0; cellX < gridNx; cellX += 1) {
      const x0 = gridOriginX + cellX * pitch - cellSlack, x1 = x0 + pitch + 2 * cellSlack;
      let bound = Infinity;
      for (let segment = 0; segment < count; segment += 1) {
        const next = segment + 1 < count ? segment + 1 : 0;
        const ax = points[2 * segment], az = points[2 * segment + 1];
        const bx = points[2 * next], bz = points[2 * next + 1];
        const corner = Math.max(
          segmentDistanceSquared(x0, z0, ax, az, bx, bz),
          segmentDistanceSquared(x1, z0, ax, az, bx, bz),
          segmentDistanceSquared(x0, z1, ax, az, bx, bz),
          segmentDistanceSquared(x1, z1, ax, az, bx, bz));
        if (corner < bound) bound = corner;
      }
      const admit = bound * POND_VESSEL_PLAN_INDEX_SLACK;
      cellStart[cellX + gridNx * cellZ] = cellItems.length;
      for (let segment = 0; segment < count; segment += 1) {
        const gapX = Math.max(0, Math.max(segMinX[segment] - x1, x0 - segMaxX[segment]));
        const gapZ = Math.max(0, Math.max(segMinZ[segment] - z1, z0 - segMaxZ[segment]));
        if (gapX * gapX + gapZ * gapZ <= admit) cellItems.push(segment);
      }
    }
  }
  cellStart[gridNx * gridNz] = cellItems.length;

  // One band per two segments: dense enough that a band holds the two or three
  // segments a horizontal line actually meets, cheap enough to build.
  const bandCount = Math.max(8, Math.min(1024, count * 2));
  // The identical expression the query uses, and that identity is the argument:
  // `v -> floor((v - minZ) * inverse)` is monotone in `v` for any fixed
  // `inverse`, so a segment bucketed from its own z extremes covers every band
  // a z between them can land in. Written as `/ pitch` here and `* inverse`
  // there, the two could round apart and the containment would not hold.
  const bandInversePitch = bandCount / Math.max(maxZ - minZ, Number.MIN_VALUE);
  const bandBuckets: number[][] = [];
  for (let band = 0; band < bandCount; band += 1) bandBuckets.push([]);
  for (let segment = 0; segment < count; segment += 1) {
    const low = Math.max(0, Math.min(bandCount - 1, Math.floor((segMinZ[segment] - minZ) * bandInversePitch)));
    const high = Math.max(0, Math.min(bandCount - 1, Math.floor((segMaxZ[segment] - minZ) * bandInversePitch)));
    for (let band = low; band <= high; band += 1) bandBuckets[band].push(segment);
  }
  const bandStart = new Int32Array(bandCount + 1);
  const bandItems: number[] = [];
  for (let band = 0; band < bandCount; band += 1) {
    bandStart[band] = bandItems.length;
    for (const segment of bandBuckets[band]) bandItems.push(segment);
  }
  bandStart[bandCount] = bandItems.length;

  return {
    points, count,
    gridOriginX, gridOriginZ, gridInversePitch: 1 / pitch, gridNx, gridNz,
    cellStart, cellItems: Int32Array.from(cellItems),
    bandOriginZ: minZ, bandInversePitch, bandCount,
    bandStart, bandItems: Int32Array.from(bandItems),
  };
}

/**
 * A closed Catmull-Rom through `values`, evaluated at a fraction of the way
 * round. The same interpolation the plan's radius uses, so a section modulation
 * and a plan wobble authored at the same lobe count stay in step rather than
 * beating against one another.
 */
function periodicSpline(values: readonly number[], turn: number): number {
  const count = values.length;
  const position = ((turn % 1) + 1) % 1 * count;
  const index = Math.floor(position), t = position - index, t2 = t * t, t3 = t2 * t;
  const at = (offset: number) => values[((index + offset) % count + count) % count];
  const a = at(-1), b = at(0), c = at(1), d = at(2);
  return .5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - 3 * c + d - a) * t3);
}

/** Two octaves of seeded value noise on the ground plane, in [-1, 1]. */
function relief(x: number, z: number, seed: number): number {
  let total = 0, amplitude = 1, frequency = 26, weight = 0;
  for (let octave = 0; octave < 2; octave += 1) {
    const px = x * frequency, pz = z * frequency;
    const cx = Math.floor(px), cz = Math.floor(pz);
    const fx = px - cx, fz = pz - cz;
    const sx = fx * fx * (3 - 2 * fx), sz = fz * fz * (3 - 2 * fz);
    const corner = (i: number, j: number) => hashSigned(seed + 0x9e37 * octave + 73_856_093 * (cx + i) + 19_349_663 * (cz + j));
    const lower = corner(0, 0) * (1 - sx) + corner(1, 0) * sx;
    const upper = corner(0, 1) * (1 - sx) + corner(1, 1) * sx;
    total += amplitude * (lower * (1 - sz) + upper * sz);
    weight += amplitude;
    amplitude *= .45;
    frequency *= 2.7;
  }
  return total / weight;
}

/** Smooth elliptical footprint weight, the same form the analytic terrain features use. */
function terraceWeight(terrace: PondVesselTerrace, x: number, z: number): number {
  const rotation = terrace.rotation_rad ?? 0;
  const cos = Math.cos(rotation), sin = Math.sin(rotation);
  const dx = x - terrace.center_m[0], dz = z - terrace.center_m[1];
  const localX = (cos * dx + sin * dz) / terrace.radius_m[0];
  const localZ = (-sin * dx + cos * dz) / terrace.radius_m[1];
  const distance = Math.hypot(localX, localZ);
  // A wall takes the whole rise over its own run at the footprint's edge; a
  // mound spreads it from `flat` outward. See `PondVesselTerrace.faceRun_m`.
  const inner = terrace.faceRun_m === undefined
    ? terrace.flat ?? .45
    // Normalised against the ellipse's mean radius rather than against whichever
    // semi-axis this bearing is nearest, so the face is one width all the way
    // round instead of a wall on the short axis and a ramp on the long one.
    : Math.max(0, 1 - Math.min(.95, Math.max(1e-3,
      terrace.faceRun_m / Math.sqrt(terrace.radius_m[0] * terrace.radius_m[1]))));
  if (distance <= inner) return 1;
  if (distance >= 1) return 0;
  return ramp(1 - (distance - inner) / (1 - inner));
}

/**
 * A terrace's rise at a world point, in metres.
 *
 * Two constructions behind one signature. A terrace carrying a {@link
 * PondVesselTerrace.wall} is a formed bed: its outline is the wall's *crest
 * line*, and the height comes from {@link wallProfile_m} of the signed plan
 * distance to it — the same shape of rule the coping and the inner face already
 * use. A terrace without one is the old mound, unchanged, because a bank or a
 * dune wants exactly that and nothing else in the repository should move.
 */
function terraceLift_m(terrace: PondVesselTerrace, x: number, z: number): number {
  if (!terrace.wall) return terrace.height_m * terraceWeight(terrace, x, z);
  const rotation = terrace.rotation_rad ?? 0;
  const cos = Math.cos(rotation), sin = Math.sin(rotation);
  const dx = x - terrace.center_m[0], dz = z - terrace.center_m[1];
  const localX = cos * dx + sin * dz;
  const localZ = -sin * dx + cos * dz;
  // The outline wanders, on the plan's own lobe construction rather than a
  // second kind of noise: a bed cast as a bare ellipse reads as a machine part
  // beside a pond whose rim is deliberately hand-formed.
  const bearing = Math.atan2(localZ, localX);
  const wander = 1 + (terrace.wobble ?? 0) * Math.sin((terrace.lobes ?? 5) * bearing + 0.7);
  const rx = Math.max(1e-4, terrace.radius_m[0] * wander);
  const rz = Math.max(1e-4, terrace.radius_m[1] * wander);
  const q = Math.hypot(localX / rx, localZ / rz);
  // Signed plan distance in metres, gradient-normalised so a wall is one width
  // all the way round an eccentric footprint. `(q - 1)/|grad q|` is the standard
  // first-order distance to an implicit curve, and it is exact on a circle.
  const gradient = Math.hypot(localX / (rx * rx), localZ / (rz * rz));
  const d = gradient > 1e-9 ? ((q - 1) * q) / gradient : (q - 1) * Math.sqrt(rx * rz);
  return wallProfile_m(terrace.wall, terrace.height_m, d);
}

/**
 * Ground height at a world point, in metres above the container floor.
 *
 * The profile is a function of the signed plan distance alone, which is what
 * keeps the coping a constant section all the way round a wandering outline:
 *
 *   base   the ground outside, falling into the basin once past the coping's
 *          inner foot — the face over `innerFace_m`, then the dish across the
 *          rest of the floor
 *   crest  the bullnose, centred on the plan curve and symmetric across it
 *   lift   terraces, masked out of the pond so a plateau can never swallow the rim
 *
 * The fall is *two* smoothsteps summed rather than one followed by a flat floor,
 * and that is the difference between a cylinder and a basin. Both are tangent at
 * their own start, so the whole surface leaves the plaster band with a horizontal
 * normal, turns down through the face, and flattens again into the middle of the
 * pond with no join anywhere: one continuous curve from the kerb's inner foot to
 * the floor. Sharing the drop between them is also the only lever on the scene's
 * Lipschitz bound that does not cost a millimetre of the rim — see
 * `POND_VESSEL_FLOOR_DISH`.
 */
export function pondVesselHeightAt(
  spec: PondVesselSpec,
  curve: readonly (readonly [number, number])[],
  x: number,
  z: number,
): number {
  const distance = pondVesselPlanDistance(curve, x, z);

  // Where round the pond this point is, so the coping's section can be a
  // function of its own run. Taken about the plan's centre rather than along the
  // curve's arc length: the two differ by the wobble, which is small, and an arc
  // length would have to be integrated per sample for no visible gain.
  const turn = pondVesselTurnAt(spec, x, z);
  const rimHeight = spec.rimHeight_m * (1 + spec.sectionHeightVariation * periodicSpline(sectionModulation(spec, 0), turn));
  const rimHalfWidth = pondVesselRimHalfWidthAt(spec, turn);

  const inward = -distance - rimHalfWidth;
  const dish = spec.floorDish ?? POND_VESSEL_FLOOR_DISH;
  const fall = distance >= 0 ? 0 : spec.basinDepth_m * (
    (1 - dish) * pondVesselInnerFaceProfile(inward / pondVesselInnerFaceRun(spec, turn))
    + dish * ramp(inward / pondVesselFloorDishReach(spec, curve))
  );
  const base = spec.groundHeight_m - fall;
  const crest = spec.crest === "flat" ? 0
    : spec.crest === "wall" && spec.crestWall
      // Mirrored about the rail: the profile's own `d = 0` is the plateau edge,
      // so the crown runs from the centreline out to `rimHalfWidth - run` and the
      // rolled flank takes the rest. Both sides get it from `|distance|`, which
      // is what makes the section symmetric all the way round a wandering plan.
      ? wallProfile_m(
        spec.crestWall,
        rimHeight,
        Math.abs(distance) - Math.max(0, rimHalfWidth - wallProfileRun_m(spec.crestWall, rimHeight)),
      )
      : rimHeight * pondVesselCrestProfile(Math.abs(distance) / rimHalfWidth);
  let lift = 0;
  if (spec.terraces?.length) {
    // One rim-width of fade beyond the crest, so a terrace arrives outside the
    // coping rather than growing out of its shoulder.
    const outside = ramp((distance - rimHalfWidth) / rimHalfWidth);
    for (const terrace of spec.terraces) lift = Math.max(lift, terraceLift_m(terrace, x, z) * outside);
  }
  return Math.max(0, base + crest + lift + spec.relief_m * relief(x, z, spec.seed ^ 0x51ed_2701));
}

/**
 * Where round the plan a world point lies, as a fraction of a turn about the
 * plan's centre. The parameter every per-turn quantity here is a function of.
 */
export function pondVesselTurnAt(spec: PondVesselSpec, x: number, z: number): number {
  return Math.atan2(z - spec.center_m[1], x - spec.center_m[0]) / (2 * Math.PI);
}

/**
 * Half-width of the coping's band at a given turn, in metres.
 *
 * Exported for the same reason `pondVesselInnerFaceRun` is: it is a number two
 * authorities have to agree on, and a caller that re-derived it would be a
 * second copy of this modulation that drifts the moment a seed moves.
 *
 * With `crest: "flat"` it is no longer a *crest* width. It is the half-width of
 * the level plaster band the vessel leaves either side of the plan curve —
 * where the ground stops being flat and the inner face starts falling — and the
 * swept solid set into that band has a ground footprint of its own
 * (`sweptCopingSection(...).groundHalfWidth_m`) that has to land on it. The hero
 * vessel derives `rimHalfWidth_m` from exactly that footprint so the two agree
 * on the mean; asking whether they agree at a *particular* turn is what this is
 * for, and `tests/swept-coping.test.ts` is where it is asked.
 *
 * The salt is 977 rather than 0 because the height modulation already uses 0 —
 * a shared salt would make the band widen exactly where the crest rises, which
 * is one motif repeating rather than two accidents of the same hand.
 */
export function pondVesselRimHalfWidthAt(spec: PondVesselSpec, turn: number): number {
  return spec.rimHalfWidth_m * (1 + spec.sectionWidthVariation * periodicSpline(sectionModulation(spec, 977), turn));
}

/**
 * The inner face's run at a given turn about the plan's centre.
 *
 * Exported because the stones need it: whether a stepping disc wades or stands
 * on a floor, and whether a pebble course has a shore to lie on, are both
 * questions about this number, and a generator that re-derived it from the
 * sector would drift from the surface the water actually meets.
 */
export function pondVesselInnerFaceRun(spec: PondVesselSpec, turn: number): number {
  const beach = spec.beach;
  if (!beach) return spec.innerFace_m;
  return spec.innerFace_m + (beach.innerFace_m - spec.innerFace_m) * beachWeight(beach, turn);
}

/** The plan's inradius, cached per curve so the bake is not quadratic. */
const planInradiusCache = new WeakMap<object, number>();

/**
 * How far in from the coping's inner foot the floor's dish takes to reach its
 * deepest, in metres.
 *
 * Derived from the curve rather than from `radius_m`, and the reason is that
 * three separate things move the outline inward — the grain, the sway and a
 * beach's pinch — so a reach taken off the authored semi-axes would be a
 * different number from the one the pond actually has. What is wanted is the
 * plan's *inradius about its own centre*: the smallest distance from the centre
 * to the outline, which is a lower bound on how far in any point can be and is
 * exact on a circle. `POND_VESSEL_FLOOR_DISH_REACH` then stops the dish a little
 * short of it, so the middle of the pond is a floor at `basinDepth_m` rather
 * than a ridge approaching it.
 *
 * Exported for the same reason `pondVesselInnerFaceRun` is: how deep the water
 * is at a bed is a question about this number, and a stone generator that
 * re-derived it would bed against a floor the vessel does not have.
 */
export function pondVesselFloorDishReach(
  spec: PondVesselSpec,
  curve: readonly (readonly [number, number])[],
): number {
  // Cached on the curve, not on the spec: it is a property of the polyline, and
  // `bakePondVesselTerrain` asks for it once per sample over ~56k of them.
  let inradius = planInradiusCache.get(curve);
  if (inradius === undefined) {
    inradius = Infinity;
    for (const [x, z] of curve) inradius = Math.min(inradius, Math.hypot(x - spec.center_m[0], z - spec.center_m[1]));
    planInradiusCache.set(curve, inradius);
  }
  return Math.max(1e-4, POND_VESSEL_FLOOR_DISH_REACH * (inradius - spec.rimHalfWidth_m));
}

/** Per-lobe section multipliers in [-1, 1], cached per spec so the bake is not quadratic. */
const sectionModulationCache = new WeakMap<PondVesselSpec, Map<number, number[]>>();
function sectionModulation(spec: PondVesselSpec, salt: number): number[] {
  let bySalt = sectionModulationCache.get(spec);
  if (!bySalt) sectionModulationCache.set(spec, bySalt = new Map());
  let values = bySalt.get(salt);
  if (!values) {
    values = [];
    // Deliberately not the plan's own lobe count: a section that swelled exactly
    // where the outline bulged would read as one repeating motif rather than as
    // two independent accidents of the same hand.
    const count = spec.lobes + 4;
    for (let index = 0; index < count; index += 1) values.push(hashSigned(spec.seed + salt + 61 * index));
    bySalt.set(salt, values);
  }
  return values;
}

/**
 * Still-water level, set `below_m` under the ground *outside* the pond.
 *
 * Measured from the outer ground rather than down from the crest, because the
 * property that matters is not how much dry rim shows — it is that a tank fill
 * wets the basin and nothing else. A waterline above the outer ground floods the
 * whole container with a film two cells deep, which is a degenerate topology for
 * the sparse solver (it published no liquid-row frontier at all) as well as
 * being wrong for a pond that is set into its ground.
 *
 * `relief_m` is subtracted too: the lumpiness that makes the plaster look
 * hand-laid also lowers it in places, and the clearance has to survive that.
 */
export function pondVesselWaterline(spec: PondVesselSpec, below_m: number): number {
  return spec.groundHeight_m - spec.relief_m - below_m;
}

/**
 * Bake the vessel onto the lattice every solver already consumes.
 *
 * Layout matches `bakeTerrainGrid` exactly — origin at the container's minimum
 * corner, one sample per spacing with a sample *on* each edge — because the
 * solvers, the renderer and the contact path all read a `TerrainGrid` through
 * one bilinear sampler and none of them may learn that this one was generated.
 */
export function bakePondVesselTerrain(
  spec: PondVesselSpec,
  container: { readonly width_m: number; readonly height_m: number; readonly depth_m: number },
  spacing_m: number,
  /**
   * The budget the caller is spending, defaulting to the document's.
   *
   * A bake that lands in `scene.terrain.grid` is JSON that every clone and
   * every worker hand-off carries, and `MAX_TERRAIN_GRID_SAMPLES` is what keeps
   * that survivable. A bake that is *derived* from `TerrainProcedural` is
   * neither serialized nor transported, so it answers to host memory instead —
   * see `MAX_TERRAIN_DERIVED_GRID_SAMPLES`. The budget is therefore the
   * caller's to name rather than a property of this function.
   */
  maximumSamples: number = MAX_TERRAIN_GRID_SAMPLES,
): TerrainGrid {
  if (!(spacing_m > 0)) throw new RangeError("Pond vessel grid spacing must be positive");
  if (!(spec.rimHalfWidth_m > 0) || !(spec.innerFace_m > 0)) throw new RangeError("Pond vessel rim width and inner face must be positive");
  if (!(spec.basinDepth_m > 0) || !(spec.rimHeight_m >= 0)) throw new RangeError("Pond vessel basin depth must be positive and rim height non-negative");
  if (spec.basinDepth_m > spec.groundHeight_m) throw new RangeError("Pond vessel basin cannot be cut below the container floor");
  const dish = spec.floorDish ?? POND_VESSEL_FLOOR_DISH;
  if (!(dish >= 0 && dish < 1)) throw new RangeError("Pond vessel floor dish must be in [0, 1)");

  const nx = Math.max(MIN_TERRAIN_GRID_SIZE, Math.ceil(container.width_m / spacing_m) + 1);
  const nz = Math.max(MIN_TERRAIN_GRID_SIZE, Math.ceil(container.depth_m / spacing_m) + 1);
  if (nx * nz > maximumSamples) throw new RangeError(`Pond vessel grid ${nx}x${nz} exceeds ${maximumSamples} samples`);

  const curve = pondVesselPlanCurve(spec);
  const origin_m = { x: -.5 * container.width_m, z: -.5 * container.depth_m };
  const heights_m = new Array<number>(nx * nz);
  for (let j = 0; j < nz; j += 1) for (let i = 0; i < nx; i += 1) {
    const height = pondVesselHeightAt(spec, curve, origin_m.x + i * spacing_m, origin_m.z + j * spacing_m);
    heights_m[i + nx * j] = Math.min(container.height_m, Math.max(0, height));
  }
  return { kind: "grid", origin_m, spacing_m, size: { nx, nz }, heights_m };
}
