/**
 * The backdrop's ground: a pure, seeded height function around the set.
 *
 * Sums of seeded directional waves rather than lattice noise, for three reasons
 * that all matter to the passes downstream. The gradient is analytic and cheap,
 * which is how the phase-2 shader lights the hills without storing a normal. The
 * Lipschitz constant is exact (`sum a_i k_i`), which is what lets the octree
 * builder prove that a box cannot reach the surface without sampling it. And
 * the whole field is a small table of numbers a WGSL port reads verbatim —
 * `BackdropField.waves` is laid out for exactly that.
 *
 * The shape, measured by `d`, the horizontal distance to the set's footprint
 * rectangle (zero inside it):
 *
 *   h(x, z) = seam + ramp(d) * (valleyRise * valley(d) + amplitude * n01(x, z))
 *
 * `ramp` is 0 across the flat ring and eases to 1 over `rampWidth_m`, so the set's
 * slab edge meets level ground flush and the first hills rise gently. `valley`
 * lifts the mean ground toward the outer radius, which seats the set in the
 * bottom of a shallow bowl and lets the far rim hide the edge of the world. `n01`
 * is the wave sum remapped into [0, 1], so the hills only ever rise from the seam
 * — a dip just past the ring would read as the set standing on a plateau.
 *
 * No DOM or GPU imports: this runs in the render worker.
 */
import type { BackdropDescription } from "../../../core/backdrop";
import type { SceneDescription } from "../../../core/model";
import { sceneSvoGroundPlane } from "../materials/svo-terrain-material";

/** Where the set's ground ends and the backdrop's begins. */
export interface BackdropSeam {
  /** Height of the set's ground at its footprint edge. */
  height_m: number;
  /** Set footprint `[minX, minZ, maxX, maxZ]`; the backdrop leaves it empty. */
  footprint_m: readonly [number, number, number, number];
}

/** Waves per octave. Three directions per band keep the ridges from reading as corduroy. */
export const BACKDROP_WAVES_PER_OCTAVE = 3;
/** Floats per wave in `BackdropField.waves`: `[dirX, dirZ, k, phase, amplitude]`. */
export const BACKDROP_WAVE_STRIDE = 5;
/** Amplitude ratio between successive octaves. */
const OCTAVE_GAIN = 0.45;

/** A description compiled against one seam; every query below reads this. */
export interface BackdropField {
  readonly description: BackdropDescription;
  readonly seam: BackdropSeam;
  /** `[dirX, dirZ, k (rad/m), phase (rad), amplitude]` per wave, amplitudes summing to 1. */
  readonly waves: Float64Array;
  readonly waveCount: number;
  /** Upper bound on |grad h| anywhere, in m/m. */
  readonly lipschitz: number;
  /** Upper bound on h - seam anywhere. */
  readonly maximumRise_m: number;
}

/**
 * A 32-bit integer hash of up to five words (the murmur3 finaliser folded over
 * the inputs). Every seeded choice in the backdrop is a pure function of its
 * indices, so an item's shape does not depend on the order anything else was
 * generated in.
 */
export function backdropHash(a: number, b = 0, c = 0, d = 0, e = 0): number {
  let h = 0x9e37_79b9 ^ Math.imul(a | 0, 0x85eb_ca6b);
  for (const word of [b, c, d, e]) {
    h = Math.imul(h ^ (h >>> 16), 0x85eb_ca6b);
    h = Math.imul(h ^ (h >>> 13), 0xc2b2_ae35);
    h ^= (h >>> 16) ^ Math.imul(word | 0, 0x27d4_eb2f);
  }
  h = Math.imul(h ^ (h >>> 16), 0x85eb_ca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2_ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

/** Uniform in [0, 1) from `backdropHash`. */
export function backdropRandom(a: number, b = 0, c = 0, d = 0, e = 0): number {
  return backdropHash(a, b, c, d, e) / 0x1_0000_0000;
}

function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** Horizontal distance from (x, z) to the footprint rectangle; zero inside. */
export function backdropFootprintDistance(seam: BackdropSeam, x: number, z: number): number {
  const [minX, minZ, maxX, maxZ] = seam.footprint_m;
  const dx = Math.max(minX - x, 0, x - maxX);
  const dz = Math.max(minZ - z, 0, z - maxZ);
  return Math.sqrt(dx * dx + dz * dz);
}

function validateSeam(seam: BackdropSeam): void {
  const [minX, minZ, maxX, maxZ] = seam.footprint_m;
  if (!Number.isFinite(seam.height_m)) throw new RangeError("Backdrop seam height must be finite");
  if (![minX, minZ, maxX, maxZ].every(Number.isFinite) || !(maxX > minX) || !(maxZ > minZ)) {
    throw new RangeError("Backdrop seam footprint must be a finite, non-empty rectangle");
  }
}

/** Compile a (validated) description against the seam it will meet. */
export function compileBackdropField(description: BackdropDescription, seam: BackdropSeam): BackdropField {
  validateSeam(seam);
  if (description.kind !== "rolling-hills") throw new RangeError(`Unknown backdrop kind ${String(description.kind)}`);
  const { hills, seed, outerRadius_m } = description;
  const waveCount = hills.octaves * BACKDROP_WAVES_PER_OCTAVE;
  const waves = new Float64Array(waveCount * BACKDROP_WAVE_STRIDE);
  let amplitudeSum = 0;
  for (let octave = 0; octave < hills.octaves; octave += 1) {
    for (let index = 0; index < BACKDROP_WAVES_PER_OCTAVE; index += 1) {
      const wave = octave * BACKDROP_WAVES_PER_OCTAVE + index;
      // Directions are spread a third of a turn apart and then jittered, so no
      // octave can collapse onto a single axis by chance.
      const angle = 2 * Math.PI * (index / BACKDROP_WAVES_PER_OCTAVE + 0.3 * (backdropRandom(seed, 1, octave, index) - 0.5));
      const wavelength = hills.wavelength_m / 2 ** octave * (0.85 + 0.3 * backdropRandom(seed, 2, octave, index));
      const amplitude = OCTAVE_GAIN ** octave * (0.75 + 0.5 * backdropRandom(seed, 3, octave, index));
      const base = wave * BACKDROP_WAVE_STRIDE;
      waves[base] = Math.cos(angle);
      waves[base + 1] = Math.sin(angle);
      waves[base + 2] = 2 * Math.PI / wavelength;
      waves[base + 3] = 2 * Math.PI * backdropRandom(seed, 4, octave, index);
      waves[base + 4] = amplitude;
      amplitudeSum += amplitude;
    }
  }
  let waveSlope = 0;
  for (let wave = 0; wave < waveCount; wave += 1) {
    waves[wave * BACKDROP_WAVE_STRIDE + 4] /= amplitudeSum;
    waveSlope += waves[wave * BACKDROP_WAVE_STRIDE + 4] * waves[wave * BACKDROP_WAVE_STRIDE + 2];
  }
  // Each term's bound: |ramp'| <= 1.5/width (smoothstep), |grad d| <= 1, and the
  // factors multiplying it are at most 1. `n01 = 0.5 + 0.5 n` halves the wave slope.
  const valleySpan = outerRadius_m - hills.flatRing_m;
  const lipschitz = 1.5 / hills.rampWidth_m * (hills.valleyRise_m + hills.amplitude_m)
    + hills.valleyRise_m * 1.5 / valleySpan
    + hills.amplitude_m * 0.5 * waveSlope;
  return {
    description,
    seam: { height_m: seam.height_m, footprint_m: [...seam.footprint_m] as unknown as BackdropSeam["footprint_m"] },
    waves,
    waveCount,
    lipschitz,
    maximumRise_m: hills.valleyRise_m + hills.amplitude_m,
  };
}

/** The ramp and valley envelopes at footprint distance `d`. */
function envelopes(field: BackdropField, d: number): readonly [number, number] {
  const { hills, outerRadius_m } = field.description;
  return [
    smoothstep(hills.flatRing_m, hills.flatRing_m + hills.rampWidth_m, d),
    smoothstep(hills.flatRing_m, outerRadius_m, d),
  ];
}

/** Normalised wave sum n in [-1, 1]. */
export function backdropWaveSum(field: BackdropField, x: number, z: number): number {
  const waves = field.waves;
  let sum = 0;
  for (let base = 0; base < waves.length; base += BACKDROP_WAVE_STRIDE) {
    sum += waves[base + 4] * Math.sin(waves[base + 2] * (waves[base] * x + waves[base + 1] * z) + waves[base + 3]);
  }
  return sum;
}

/** Ground height of the compiled field at (x, z). */
export function backdropFieldHeight(field: BackdropField, x: number, z: number): number {
  const d = backdropFootprintDistance(field.seam, x, z);
  const { hills } = field.description;
  // The flat ring skips the wave sum outright.
  if (d <= hills.flatRing_m) return field.seam.height_m;
  return backdropHeightFromWaveSum(field, d, backdropWaveSum(field, x, z));
}

/**
 * Height with the ramp/valley envelopes already evaluated and the wave sum
 * supplied — the voxeliser's inner loop, which forms the wave sum separably.
 */
export function backdropHeightFromWaveSum(field: BackdropField, d: number, waveSum: number): number {
  const { hills, outerRadius_m } = field.description;
  const ramp = smoothstep(hills.flatRing_m, hills.flatRing_m + hills.rampWidth_m, d);
  if (ramp === 0) return field.seam.height_m;
  const valley = smoothstep(hills.flatRing_m, outerRadius_m, d);
  return field.seam.height_m + ramp * (hills.valleyRise_m * valley + hills.amplitude_m * (0.5 + 0.5 * waveSum));
}

const compiledByDescription = new WeakMap<BackdropDescription, Map<string, BackdropField>>();

/**
 * `h(x, z)` for a description and seam, as the plan names it. Compiled fields
 * are memoised per description object and seam, so a caller probing a handful
 * of points pays the wave table once; hot loops should hold the field instead.
 */
export function backdropHeight(description: BackdropDescription, seam: BackdropSeam, x: number, z: number): number {
  let bySeam = compiledByDescription.get(description);
  if (!bySeam) { bySeam = new Map(); compiledByDescription.set(description, bySeam); }
  const key = `${seam.height_m}|${seam.footprint_m.join(",")}`;
  let field = bySeam.get(key);
  if (!field) { field = compileBackdropField(description, seam); bySeam.set(key, field); }
  return backdropFieldHeight(field, x, z);
}

/**
 * Conservative `[min, max]` of the ground height over a horizontal rectangle.
 *
 * Two independent bounds, intersected. The envelope bound is monotone in the
 * footprint distance, so it is tight for large boxes far from any crest; the
 * Lipschitz bound about the centre is tight for small boxes, which are the ones
 * the octree refines most.
 */
export function backdropHeightBounds(
  field: BackdropField,
  minX: number, minZ: number, maxX: number, maxZ: number,
): readonly [number, number] {
  const [fMinX, fMinZ, fMaxX, fMaxZ] = field.seam.footprint_m;
  // Nearest distance between the box and the footprint rectangle, and the
  // farthest (a corner, since distance to a convex set is convex).
  const gapX = Math.max(fMinX - maxX, 0, minX - fMaxX);
  const gapZ = Math.max(fMinZ - maxZ, 0, minZ - fMaxZ);
  const dNear = Math.hypot(gapX, gapZ);
  let dFar = 0;
  for (const x of [minX, maxX]) for (const z of [minZ, maxZ]) dFar = Math.max(dFar, backdropFootprintDistance(field.seam, x, z));
  const [rampNear, valleyNear] = envelopes(field, dNear);
  const [rampFar, valleyFar] = envelopes(field, dFar);
  const { hills } = field.description;
  const seam = field.seam.height_m;
  let low = seam + rampNear * hills.valleyRise_m * valleyNear;
  let high = seam + rampFar * (hills.valleyRise_m * valleyFar + hills.amplitude_m);
  const cx = 0.5 * (minX + maxX), cz = 0.5 * (minZ + maxZ);
  const reach = field.lipschitz * 0.5 * Math.hypot(maxX - minX, maxZ - minZ);
  const centre = backdropFieldHeight(field, cx, cz);
  low = Math.max(low, centre - reach);
  high = Math.min(high, centre + reach);
  return [low, high];
}

/**
 * Upper bound on every second derivative of the ground height, in 1/m, over
 * the ground at least `nearestDistance_m` from the footprint (zero: anywhere).
 *
 * `h = seam + ramp(d) body(d, x, z)`. With `w` the ramp width, `S` the valley
 * span, `A` the amplitude, `V` the valley rise and `n01 = 0.5 + 0.5 sum a_i
 * sin(...)`: `|ramp'| <= 1.5/w`, `|ramp''| <= 6/w^2`, `|body| <= V + A`,
 * `|grad body| <= 1.5 V/S + 0.5 A sum a k`, `|hess body| <= 6 V/S^2 + 1.5 V/(S
 * d) + 0.5 A sum a k^2`, and the footprint distance has `|grad d| = 1` and
 * curvature at most `1/d`. Every `1/d` term multiplies a ramp or valley slope,
 * which is zero inside the flat ring `r`, so `d` is at least `max(r, nearest)`.
 * Past the ramp (`nearest >= r + w`) the ramp is the constant 1 and only the
 * body's curvature is left: about 1.6/m for the hero hills against 14.8/m
 * anywhere, which is what makes a per-leaf Taylor bound tight.
 */
export function backdropFieldCurvatureBound(field: BackdropField, nearestDistance_m = 0): number {
  const { hills, outerRadius_m } = field.description;
  const w = hills.rampWidth_m, S = outerRadius_m - hills.flatRing_m, r = Math.max(hills.flatRing_m, nearestDistance_m);
  if (!(hills.flatRing_m > 0) || !(S > 0) || !(w > 0)) throw new RangeError("Backdrop curvature bound needs a positive flat ring, ramp and valley span");
  const A = hills.amplitude_m, V = hills.valleyRise_m;
  let slope = 0, curvature = 0;
  for (let base = 0; base < field.waves.length; base += BACKDROP_WAVE_STRIDE) {
    const k = field.waves[base + 2]!, a = field.waves[base + 4]!;
    slope += a * k; curvature += a * k * k;
  }
  const body = V + A;
  const bodySlope = 1.5 * V / S + 0.5 * A * slope;
  const bodyCurvature = 6 * V / (S * S) + 1.5 * V / (S * r) + 0.5 * A * curvature;
  if (nearestDistance_m >= hills.flatRing_m + w) return bodyCurvature;
  return 6 / (w * w) * body + 1.5 / (w * r) * body + 2 * 1.5 / w * bodySlope + bodyCurvature;
}

/** Ground height and its analytic gradient at (x, z): `[h, dh/dx, dh/dz]`. */
export function backdropFieldHeightAndGradient(field: BackdropField, x: number, z: number): readonly [number, number, number] {
  const [minX, minZ, maxX, maxZ] = field.seam.footprint_m;
  const ox = x - Math.min(Math.max(x, minX), maxX), oz = z - Math.min(Math.max(z, minZ), maxZ);
  const d = Math.sqrt(ox * ox + oz * oz);
  const { hills, outerRadius_m } = field.description;
  if (d <= hills.flatRing_m) return [field.seam.height_m, 0, 0];
  const gdx = ox / d, gdz = oz / d;
  const tr = Math.min(1, Math.max(0, (d - hills.flatRing_m) / hills.rampWidth_m));
  const ramp = tr * tr * (3 - 2 * tr), rampSlope = 6 * tr * (1 - tr) / hills.rampWidth_m;
  const span = outerRadius_m - hills.flatRing_m;
  const tv = Math.min(1, Math.max(0, (d - hills.flatRing_m) / span));
  const valley = tv * tv * (3 - 2 * tv), valleySlope = 6 * tv * (1 - tv) / span;
  const waves = field.waves;
  let n = 0, gx = 0, gz = 0;
  for (let base = 0; base < waves.length; base += BACKDROP_WAVE_STRIDE) {
    const phase = waves[base + 2]! * (waves[base]! * x + waves[base + 1]! * z) + waves[base + 3]!;
    n += waves[base + 4]! * Math.sin(phase);
    const c = waves[base + 4]! * waves[base + 2]! * Math.cos(phase);
    gx += c * waves[base]!; gz += c * waves[base + 1]!;
  }
  const body = hills.valleyRise_m * valley + hills.amplitude_m * (0.5 + 0.5 * n);
  const radial = rampSlope * body + ramp * hills.valleyRise_m * valleySlope;
  return [field.seam.height_m + ramp * body,
    radial * gdx + ramp * 0.5 * hills.amplitude_m * gx,
    radial * gdz + ramp * 0.5 * hills.amplitude_m * gz];
}

/**
 * Conservative `[min, max, slope, curvature]` of the ground over a horizontal
 * rectangle: the height range, an upper bound on `|grad h|` and the local
 * second-derivative bound `K` the other three were derived with.
 *
 * A second-order Taylor bound about the centre of each of `subdivisions^2`
 * sub-rectangles: `h(p) <= h(c) + g.(p - c) + K |p - c|^2 / 2`, whose linear
 * part peaks at a corner (`(|gx| sx + |gz| sz) / 2`), with `K` the local
 * curvature bound (`backdropFieldCurvatureBound` at the rectangle's nearest
 * footprint distance). The slope is `|g(c)| + K r`. Intersected with the
 * envelope bound of `backdropHeightBounds`. Against the Lipschitz margin this
 * replaces, the slack over a backdrop leaf falls from about 1.4 leaf edges to
 * 0.07 at two subdivisions (hero hills, 6 px leaves).
 */
export function backdropTaylorHeightBounds(
  field: BackdropField,
  minX: number, minZ: number, maxX: number, maxZ: number,
  subdivisions = 2,
): readonly [number, number, number, number] {
  const [fMinX, fMinZ, fMaxX, fMaxZ] = field.seam.footprint_m;
  const nearest = Math.hypot(Math.max(fMinX - maxX, 0, minX - fMaxX), Math.max(fMinZ - maxZ, 0, minZ - fMaxZ));
  const K = backdropFieldCurvatureBound(field, nearest);
  const sx = (maxX - minX) / subdivisions, sz = (maxZ - minZ) / subdivisions;
  const quadratic = 0.125 * K * (sx * sx + sz * sz), reach = 0.5 * Math.hypot(sx, sz) * K;
  let low = Number.POSITIVE_INFINITY, high = Number.NEGATIVE_INFINITY, slope = 0;
  for (let j = 0; j < subdivisions; j += 1) for (let i = 0; i < subdivisions; i += 1) {
    const [h, gx, gz] = backdropFieldHeightAndGradient(field, minX + (i + 0.5) * sx, minZ + (j + 0.5) * sz);
    const margin = 0.5 * (Math.abs(gx) * sx + Math.abs(gz) * sz) + quadratic;
    low = Math.min(low, h - margin); high = Math.max(high, h + margin);
    slope = Math.max(slope, Math.hypot(gx, gz) + reach);
  }
  const [envelopeLow, envelopeHigh] = backdropHeightBounds(field, minX, minZ, maxX, maxZ);
  return [Math.max(low, envelopeLow), Math.min(high, envelopeHigh), Math.min(slope, field.lipschitz), K];
}

/**
 * The seam, derived from the set's own ground rather than stored twice.
 *
 * `sceneSvoGroundPlane` is the one authority for where the garden's slab edge
 * sits; the analytic plane the backdrop replaces is built from the same call.
 * A backdrop on a scene with no such plane has nothing to meet and is refused.
 */
export function backdropSeamForScene(
  scene: Pick<SceneDescription, "scenery" | "terrain" | "container" | "environment">,
): BackdropSeam {
  const plane = sceneSvoGroundPlane(scene);
  if (!plane) {
    throw new Error("Backdrop requires a garden scene with a terrain shell: sceneSvoGroundPlane found no seam to meet");
  }
  return { height_m: plane.height_m, footprint_m: plane.footprint_m };
}
