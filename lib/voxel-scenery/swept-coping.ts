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
 *  - **No roll-back.** A height is a function of (x, z), so nothing can lie
 *    under anything. A circular section whose centre stands *above* the plaster
 *    does: between its widest point and the ground the flank turns back under
 *    itself, which is the undercut the reference shows and the reason its rim
 *    reads as a fat kerb rather than as a swelling.
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
 * the crest is 55 mm above the outer ground and the waterline sits 42.5 mm below
 * it, so the water at rest never reaches the coping at all — the wall it presses
 * against is the plaster the coping is set into, which is terrain either way.
 * A coping swept as scenery over a vessel whose *wet* surfaces are still
 * heightfield is not a rim the water pours through; it is a rim the water was
 * never going to touch. Splash is a different question and the answer to it is
 * freeboard in the terrain, not geometry in the renderer.
 *
 * ## The section, and the two numbers that describe it
 *
 * The section is a circle, because a bullnose is a circle. It is authored as how
 * far the crest stands proud of the ground (`crestHeight_m`) and how far the
 * circle's centre is lifted above the ground as a fraction of its own radius
 * (`undercut`) — both of which are things you can see — and the radius is
 * derived. At `undercut = 0` the centre sits in the plaster and the flank leaves
 * it vertically: a crisp line, no roll-back. Above zero the widest point rises
 * above the ground and the flank turns back under itself by
 * `radius * (1 - sqrt(1 - undercut^2))`. Past about a half the rim starts to
 * read as a pipe lying on the floor rather than as a kerb set into it.
 *
 * ## Records
 *
 * The rail is emitted as a chain of round-cone SDFs: the exact convex hull of
 * the endpoint spheres. Unlike a capsule, each segment preserves both authored
 * radii; unlike a truncated cone, its ends are spherical parts of the same SDF
 * rather than planar caps that separate geometry is expected to hide. That is
 * load-bearing in an SVO. A voxel stores one primitive owner, so the old
 * endpoint sphere could not reliably hide an overlapping cone cap; whichever
 * owner won the cell was the only primitive the ray tested. Buried cap normals
 * consequently leaked through as dark one-pixel dots along the coping.
 *
 * The outline is settled long before the highlight is. A chord's departure from
 * the plan is `c^2 / 8R`, which on this pond is 0.18 mm at the plan curve's own
 * 112 points — invisible at any zoom. But the chain's *shading normal* steps by
 * the whole turn angle at every joint, because a cone is straight and the rim is
 * not, and 112 joints round a closed run is 3.2 degrees apiece. Rendered on a
 * near-white surface that is unmistakable banding, and it is *not* the section
 * modulation: turning every modulation off leaves an identically banded uniform
 * tube. Measured, on the hero rim at 0.15 m:
 *
 *   112 segments   3.2 deg/joint    112 records   banded, obviously
 *   336 segments   1.1 deg/joint    336 records   clean
 *   448 segments   0.8 deg/joint    448 records   clean
 *
 * So a convincing rim on this pond is about 336 records. The round-cone halves
 * the old cone-plus-sphere record cost, but C1 round-a-curve still has to be
 * bought with rail resolution because the ABI has no multi-segment swept-arc
 * field. A future marched sweep SDF can replace this emitter without changing
 * `sweptCopingStations`, which is why the form and record layers are separate.
 *
 * The remaining constraint is density, not total.
 * `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` is 64 primitives per brick and the
 * rest of a crowded brick is dropped from the voxelized occupancy — not from
 * primary visibility, which is a BVH over the records — so an overflowing brick
 * loses shadowing and ambient occlusion rather than silhouette. The coping is
 * therefore kept as one record per rail segment; reintroducing endpoint helper
 * primitives would spend this local density budget as well as reviving the
 * single-owner overlap ambiguity fixed here.
 */

/** Shape only: the species, with no rail, no ground and no seed. */
export interface SweptCopingForm {
  /** How far the crest stands above the ground the coping is set into. */
  readonly crestHeight_m: number;
  /**
   * Lift of the section's centre above that ground, as a fraction of the
   * section radius. Zero is a vertical meeting with no roll-back; a half is a
   * pronounced undercut. Must be in [0, 1).
   */
  readonly undercut: number;
  /**
   * How much the section swells and narrows along the run, as a fraction of the
   * radius. This is the difference between a formed kerb and an extrusion, and
   * it is the same argument `PondVesselSpec.sectionWidthVariation` makes: the
   * eye reads an unvarying crest line long before it reads a wandering outline.
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
 * The pond's rim: a fat bullnose, as tall as it is nearly wide, with a real
 * undercut.
 *
 * 55 mm proud at a half-radius lift makes a section of radius 36.7 mm and a
 * footprint 63.6 mm across — taller and narrower than the heightfield vessel's
 * 110 mm swell, and closer to the reference, whose coping reads about as wide as
 * it is tall. The roll-back is 4.9 mm and the flank leaves the plaster with its
 * normal 30 degrees below horizontal, which is what puts a shadow line under the
 * lip all the way round.
 */
export const SWEPT_COPING_POND_BULLNOSE: SweptCopingForm = Object.freeze({
  crestHeight_m: 0.055,
  undercut: 0.5,
  sectionVariation: 0.14,
  crestVariation: 0.11,
  variationLobes: 11,
  relief_m: 0.0015,
  reliefLobes: 44,
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
 * the trigonometry.
 */
export function sweptCopingSection(form: Pick<SweptCopingForm, "crestHeight_m" | "undercut">): {
  radius_m: number; axisLift_m: number; groundHalfWidth_m: number; rollBack_m: number;
} {
  const { crestHeight_m, undercut } = form;
  if (!(crestHeight_m > 0)) throw new RangeError("Swept coping crest height must be positive");
  if (!(undercut >= 0 && undercut < 1)) throw new RangeError("Swept coping undercut must be in [0, 1)");
  const radius_m = crestHeight_m / (1 + undercut);
  const groundHalfWidth_m = radius_m * Math.sqrt(1 - undercut * undercut);
  return { radius_m, axisLift_m: radius_m * undercut, groundHalfWidth_m, rollBack_m: radius_m - groundHalfWidth_m };
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
  const stride = spec.segmentStride ?? 1;
  if (!Number.isInteger(stride) || stride < 1) throw new RangeError("Swept coping segment stride must be a positive integer");
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
    // The lift stays proportional to the *authored* radius rather than to this
    // station's, so a swell widens the rim without also lifting it out of the
    // ground and undoing its own undercut.
    const ground = spec.groundHeightAt(x, z);
    return { at: V(x, ground + crest - base.radius_m, z), radius: Math.max(1e-4, radius), turn };
  });
}

/**
 * Grow one coping along a rail.
 *
 * Shape from `sweptCopingStations`, records from here. Consecutive round-cone
 * SDFs share the exact endpoint sphere, so either owner gives the same surface
 * at a joint and there is no planar cap for a voxel owner to leak.
 */
export function sweptCopingNodes(spec: SweptCopingSpec): SceneryNode[] {
  const { key } = spec;
  const group = spec.group ?? "stone-coping";
  const nodes = sweptCopingStations(spec);
  const count = nodes.length;

  const emitted: SceneryNode[] = [];
  for (let index = 0; index < count; index += 1) {
    const from = nodes[index], to = nodes[(index + 1) % count];
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
