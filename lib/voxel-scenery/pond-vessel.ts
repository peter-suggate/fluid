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
  /** Run from the coping's inner foot down to the basin floor. */
  readonly innerFace_m: number;
  /** Control points around the plan. More lobes, more places for it to wander. */
  readonly lobes: number;
  /** Radial wander of the plan, as a fraction of the local radius. */
  readonly wobble: number;
  /** The only entropy. Any integer. */
  readonly seed: number;
  /** Raised ground outside the pond: the plateau a tree or a bench stands on. */
  readonly terraces?: readonly PondVesselTerrace[];
}

/** Polyline samples per control point. Sixteen holds a 1 m pond inside a millimetre. */
const PLAN_SAMPLES_PER_LOBE = 16;

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
export function pondVesselPlanCurve(spec: PondVesselSpec): readonly (readonly [number, number])[] {
  const { center_m: [cx, cz], radius_m: [rx, rz], lobes, wobble, seed } = spec;
  if (!Number.isInteger(lobes) || lobes < 3) throw new RangeError("Pond vessel needs at least three plan lobes");
  if (!(rx > 0) || !(rz > 0)) throw new RangeError("Pond vessel radii must be positive");
  if (!(wobble >= 0 && wobble < 1)) throw new RangeError("Pond vessel wobble must be in [0, 1)");

  const scales: number[] = [];
  for (let index = 0; index < lobes; index += 1) scales.push(1 + wobble * hashSigned(seed + 31 * index));
  const scaleAt = (index: number) => scales[((index % lobes) + lobes) % lobes];

  const points: (readonly [number, number])[] = [];
  for (let index = 0; index < lobes; index += 1) {
    const a = scaleAt(index - 1), b = scaleAt(index), c = scaleAt(index + 1), d = scaleAt(index + 2);
    for (let step = 0; step < PLAN_SAMPLES_PER_LOBE; step += 1) {
      const t = step / PLAN_SAMPLES_PER_LOBE, t2 = t * t, t3 = t2 * t;
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
  const inward = -distance - spec.rimHalfWidth_m;
  const base = distance >= 0
    ? spec.groundHeight_m
    : spec.groundHeight_m - spec.basinDepth_m * ramp(inward / spec.innerFace_m);
  const crest = spec.rimHeight_m * pondVesselCrestProfile(Math.abs(distance) / spec.rimHalfWidth_m);
  let lift = 0;
  if (spec.terraces?.length) {
    // One rim-width of fade beyond the crest, so a terrace arrives outside the
    // coping rather than growing out of its shoulder.
    const outside = ramp((distance - spec.rimHalfWidth_m) / spec.rimHalfWidth_m);
    for (const terrace of spec.terraces) lift = Math.max(lift, terrace.height_m * terraceWeight(terrace, x, z) * outside);
  }
  return Math.max(0, base + crest + lift);
}

/** Still-water level that leaves `freeboard_m` of dry coping above it. */
export function pondVesselWaterline(spec: PondVesselSpec, freeboard_m: number): number {
  return spec.groundHeight_m + spec.rimHeight_m - freeboard_m;
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
