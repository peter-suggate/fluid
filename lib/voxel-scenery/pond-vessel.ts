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
 * Determinism is the same contract the procedural tree keeps: the seed is the
 * only entropy, it enters through an integer avalanche hash, and the same spec
 * bakes the same grid on every rebuild — the sparse publication cache is keyed
 * on the geometry that comes out of here.
 */

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
  /** Semi-axes of the crest centreline before the wobble is applied. */
  readonly radius_m: readonly [number, number];
  /** Ground level around the pond, in metres above the container floor. */
  readonly groundHeight_m: number;
  /** Basin floor below `groundHeight_m`. */
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
  readonly crest?: "bullnose" | "flat";
  /** Run from the coping's inner foot down to the basin floor. */
  readonly innerFace_m: number;
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

  const scales: number[] = [];
  for (let index = 0; index < lobes; index += 1) scales.push(1 + wobble * hashSigned(seed + 31 * index));
  const scaleAt = (index: number) => scales[((index % lobes) + lobes) % lobes];

  const points: (readonly [number, number])[] = [];
  for (let index = 0; index < lobes; index += 1) {
    const a = scaleAt(index - 1), b = scaleAt(index), c = scaleAt(index + 1), d = scaleAt(index + 2);
    for (let step = 0; step < samplesPerLobe; step += 1) {
      const t = step / samplesPerLobe, t2 = t * t, t3 = t2 * t;
      // Uniform Catmull-Rom on the scalar. Control samples are exactly evenly
      // spaced in angle, so this is the natural parameterization rather than an
      // approximation of one, and a constant sequence interpolates to a constant.
      const scale = .5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - 3 * c + d - a) * t3);
      const angle = 2 * Math.PI * (index + t) / lobes;
      points.push([cx + rx * scale * Math.cos(angle), cz + rz * scale * Math.sin(angle)]);
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
 */
export function pondVesselPlanDistance(
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
  const flat = terrace.flat ?? .45;
  if (distance <= flat) return 1;
  if (distance >= 1) return 0;
  return ramp(1 - (distance - flat) / (1 - flat));
}

/**
 * Ground height at a world point, in metres above the container floor.
 *
 * The profile is a function of the signed plan distance alone, which is what
 * keeps the coping a constant section all the way round a wandering outline:
 *
 *   base   the ground outside, falling to the basin floor over `innerFace_m`
 *          once past the coping's inner foot
 *   crest  the bullnose, centred on the plan curve and symmetric across it
 *   lift   terraces, masked out of the pond so a plateau can never swallow the rim
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
  const turn = Math.atan2(z - spec.center_m[1], x - spec.center_m[0]) / (2 * Math.PI);
  const section = (variation: number, salt: number) => 1 + variation * periodicSpline(
    sectionModulation(spec, salt), turn,
  );
  const rimHeight = spec.rimHeight_m * section(spec.sectionHeightVariation, 0);
  const rimHalfWidth = spec.rimHalfWidth_m * section(spec.sectionWidthVariation, 977);

  const inward = -distance - rimHalfWidth;
  const base = distance >= 0
    ? spec.groundHeight_m
    : spec.groundHeight_m - spec.basinDepth_m * ramp(inward / pondVesselInnerFaceRun(spec, turn));
  const crest = spec.crest === "flat" ? 0 : rimHeight * pondVesselCrestProfile(Math.abs(distance) / rimHalfWidth);
  let lift = 0;
  if (spec.terraces?.length) {
    // One rim-width of fade beyond the crest, so a terrace arrives outside the
    // coping rather than growing out of its shoulder.
    const outside = ramp((distance - rimHalfWidth) / rimHalfWidth);
    for (const terrace of spec.terraces) lift = Math.max(lift, terrace.height_m * terraceWeight(terrace, x, z) * outside);
  }
  return Math.max(0, base + crest + lift + spec.relief_m * relief(x, z, spec.seed ^ 0x51ed_2701));
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
  if (!beach || !(beach.width > 0)) return spec.innerFace_m;
  // Shortest signed separation on the circle, in turns, so a sector that
  // straddles the atan2 branch cut behaves like any other.
  const separation = Math.abs((((turn - beach.turn + 0.5) % 1) + 1) % 1 - 0.5);
  return spec.innerFace_m + (beach.innerFace_m - spec.innerFace_m) * ramp(1 - separation / beach.width);
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
): TerrainGrid {
  if (!(spacing_m > 0)) throw new RangeError("Pond vessel grid spacing must be positive");
  if (!(spec.rimHalfWidth_m > 0) || !(spec.innerFace_m > 0)) throw new RangeError("Pond vessel rim width and inner face must be positive");
  if (!(spec.basinDepth_m > 0) || !(spec.rimHeight_m >= 0)) throw new RangeError("Pond vessel basin depth must be positive and rim height non-negative");
  if (spec.basinDepth_m > spec.groundHeight_m) throw new RangeError("Pond vessel basin cannot be cut below the container floor");

  const nx = Math.max(MIN_TERRAIN_GRID_SIZE, Math.ceil(container.width_m / spacing_m) + 1);
  const nz = Math.max(MIN_TERRAIN_GRID_SIZE, Math.ceil(container.depth_m / spacing_m) + 1);
  if (nx * nz > MAX_TERRAIN_GRID_SAMPLES) throw new RangeError(`Pond vessel grid ${nx}x${nz} exceeds ${MAX_TERRAIN_GRID_SAMPLES} samples`);

  const curve = pondVesselPlanCurve(spec);
  const origin_m = { x: -.5 * container.width_m, z: -.5 * container.depth_m };
  const heights_m = new Array<number>(nx * nz);
  for (let j = 0; j < nz; j += 1) for (let i = 0; i < nx; i += 1) {
    const height = pondVesselHeightAt(spec, curve, origin_m.x + i * spacing_m, origin_m.z + j * spacing_m);
    heights_m[i + nx * j] = Math.min(container.height_m, Math.max(0, height));
  }
  return { kind: "grid", origin_m, spacing_m, size: { nx, nz }, heights_m };
}
