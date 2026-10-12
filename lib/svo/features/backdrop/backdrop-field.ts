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
  // One fold per word, written out: the scatter planner hashes four times per
  // grid cell, and a word list would be an array per call.
  let h = 0x9e37_79b9 ^ Math.imul(a | 0, 0x85eb_ca6b);
  h = Math.imul(h ^ (h >>> 16), 0x85eb_ca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2_ae35);
  h ^= (h >>> 16) ^ Math.imul(b | 0, 0x27d4_eb2f);
  h = Math.imul(h ^ (h >>> 16), 0x85eb_ca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2_ae35);
  h ^= (h >>> 16) ^ Math.imul(c | 0, 0x27d4_eb2f);
  h = Math.imul(h ^ (h >>> 16), 0x85eb_ca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2_ae35);
  h ^= (h >>> 16) ^ Math.imul(d | 0, 0x27d4_eb2f);
  h = Math.imul(h ^ (h >>> 16), 0x85eb_ca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2_ae35);
  h ^= (h >>> 16) ^ Math.imul(e | 0, 0x27d4_eb2f);
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

/**
 * Everything a compiled field's heights depend on, as one string: what a table
 * derived from the field is memoised on (`content-memo.ts`).
 */
export function backdropFieldContentKey(field: BackdropField): string {
  return JSON.stringify([field.description, field.seam.height_m, field.seam.footprint_m]);
}

/**
 * The field over a grid of columns `(xs[i], zs[j])`, evaluated separably, a
 * run of one row at a time.
 *
 * A wave's phase `k (dx x + dz z) + phi` is a sum of an x term and a z term,
 * so over a grid `sin(A + B) = sin A cos B + cos A sin B` needs the sines and
 * cosines of each axis once per row or column of the grid rather than once per
 * column and wave. The x table carries the wave's amplitude and the z table
 * its phase offset; along a row the z factors are constants, and each wave is
 * one pass over the run.
 *
 * Equal to the direct evaluation to within rounding (a few ulps of the phase).
 * The footprint distance and the envelopes are formed exactly as the direct
 * evaluation forms them, so only the wave sum differs: a caller that rounds
 * the height to a lattice re-evaluates directly when the value is within that
 * of a rounding boundary.
 */
export interface BackdropGridSampler {
  /** `backdropFieldHeight(field, xs[first + i], zs[zIndex])` into `heights[i]`, for `count` columns. */
  heights(zIndex: number, first: number, count: number, heights: Float64Array): void;
  /** `backdropFieldHeightAndGradient` over the same run: `[h, dh/dx, dh/dz]` into the three arrays. */
  heightsAndGradients(zIndex: number, first: number, count: number,
    heights: Float64Array, gradientX: Float64Array, gradientZ: Float64Array): void;
}

export function createBackdropGridSampler(field: BackdropField, xs: Float64Array, zs: Float64Array): BackdropGridSampler {
  const waves = field.waves, count = field.waveCount;
  // Wave-major, so a wave's pass along a row reads one contiguous run.
  const table = (positions: Float64Array, axis: 0 | 1): readonly [Float64Array, Float64Array] => {
    const sin = new Float64Array(positions.length * count), cos = new Float64Array(positions.length * count);
    for (let wave = 0; wave < count; wave += 1) {
      const base = wave * BACKDROP_WAVE_STRIDE;
      const k = waves[base + 2]!, direction = waves[base + axis]!;
      const scale = axis === 0 ? waves[base + 4]! : 1, offset = axis === 0 ? 0 : waves[base + 3]!;
      for (let index = 0; index < positions.length; index += 1) {
        const phase = k * (direction * positions[index]!) + offset;
        sin[wave * positions.length + index] = scale * Math.sin(phase);
        cos[wave * positions.length + index] = scale * Math.cos(phase);
      }
    }
    return [sin, cos];
  };
  const [xSin, xCos] = table(xs, 0), [zSin, zCos] = table(zs, 1);
  const xCount = xs.length, zCount = zs.length;
  const sums = new Float64Array(xCount), slopesX = new Float64Array(xCount), slopesZ = new Float64Array(xCount);
  const minX = field.seam.footprint_m[0], minZ = field.seam.footprint_m[1];
  const maxX = field.seam.footprint_m[2], maxZ = field.seam.footprint_m[3];
  const seam = field.seam.height_m;
  const { hills, outerRadius_m } = field.description;
  const flatRing = hills.flatRing_m, rampWidth = hills.rampWidth_m, span = outerRadius_m - flatRing;
  const valleyRise = hills.valleyRise_m, amplitude = hills.amplitude_m;
  // `smoothstep(edge0, edge1, d)` as `backdropHeightFromWaveSum` forms it.
  const rampEdge = flatRing + rampWidth - flatRing, valleyEdge = outerRadius_m - flatRing;
  return {
    heights(zIndex, first, runLength, heights) {
      sums.fill(0, 0, runLength);
      for (let wave = 0; wave < count; wave += 1) {
        const sb = zSin[wave * zCount + zIndex]!, cb = zCos[wave * zCount + zIndex]!, base = wave * xCount + first;
        for (let i = 0; i < runLength; i += 1) sums[i]! += xSin[base + i]! * cb + xCos[base + i]! * sb;
      }
      const z = zs[zIndex]!, dz = Math.max(minZ - z, 0, z - maxZ), dz2 = dz * dz;
      for (let i = 0; i < runLength; i += 1) {
        const x = xs[first + i]!, dx = Math.max(minX - x, 0, x - maxX);
        const d = Math.sqrt(dx * dx + dz2);
        if (d <= flatRing) { heights[i] = seam; continue; }
        const tr = Math.min(1, Math.max(0, (d - flatRing) / rampEdge)), ramp = tr * tr * (3 - 2 * tr);
        if (ramp === 0) { heights[i] = seam; continue; }
        const tv = Math.min(1, Math.max(0, (d - flatRing) / valleyEdge)), valley = tv * tv * (3 - 2 * tv);
        heights[i] = seam + ramp * (valleyRise * valley + amplitude * (0.5 + 0.5 * sums[i]!));
      }
    },
    heightsAndGradients(zIndex, first, runLength, heights, gradientX, gradientZ) {
      sums.fill(0, 0, runLength); slopesX.fill(0, 0, runLength); slopesZ.fill(0, 0, runLength);
      for (let wave = 0; wave < count; wave += 1) {
        const sb = zSin[wave * zCount + zIndex]!, cb = zCos[wave * zCount + zIndex]!, base = wave * xCount + first;
        const kx = waves[wave * BACKDROP_WAVE_STRIDE + 2]! * waves[wave * BACKDROP_WAVE_STRIDE]!;
        const kz = waves[wave * BACKDROP_WAVE_STRIDE + 2]! * waves[wave * BACKDROP_WAVE_STRIDE + 1]!;
        for (let i = 0; i < runLength; i += 1) {
          const sa = xSin[base + i]!, ca = xCos[base + i]!;
          sums[i]! += sa * cb + ca * sb;
          const c = ca * cb - sa * sb;
          slopesX[i]! += c * kx; slopesZ[i]! += c * kz;
        }
      }
      const z = zs[zIndex]!, dz = Math.max(minZ - z, 0, z - maxZ), dz2 = dz * dz;
      const oz = z - Math.min(Math.max(z, minZ), maxZ);
      for (let i = 0; i < runLength; i += 1) {
        const x = xs[first + i]!, dx = Math.max(minX - x, 0, x - maxX);
        const d = Math.sqrt(dx * dx + dz2);
        if (d <= flatRing) { heights[i] = seam; gradientX[i] = 0; gradientZ[i] = 0; continue; }
        const n = sums[i]!;
        const tr = Math.min(1, Math.max(0, (d - flatRing) / rampWidth));
        const ramp = tr * tr * (3 - 2 * tr), rampSlope = 6 * tr * (1 - tr) / rampWidth;
        const tv = Math.min(1, Math.max(0, (d - flatRing) / span));
        const valley = tv * tv * (3 - 2 * tv), valleySlope = 6 * tv * (1 - tv) / span;
        const body = valleyRise * valley + amplitude * (0.5 + 0.5 * n);
        const radial = rampSlope * body + ramp * valleyRise * valleySlope;
        // The height by `backdropHeightFromWaveSum`'s own expression.
        const th = Math.min(1, Math.max(0, (d - flatRing) / rampEdge)), rampHeight = th * th * (3 - 2 * th);
        const tw = Math.min(1, Math.max(0, (d - flatRing) / valleyEdge)), valleyHeight = tw * tw * (3 - 2 * tw);
        heights[i] = rampHeight === 0 ? seam : seam + rampHeight * (valleyRise * valleyHeight + amplitude * (0.5 + 0.5 * n));
        gradientX[i] = radial * (x - Math.min(Math.max(x, minX), maxX)) / d + ramp * 0.5 * amplitude * slopesX[i]!;
        gradientZ[i] = radial * oz / d + ramp * 0.5 * amplitude * slopesZ[i]!;
      }
    },
  };
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
  const bounds = new Float64Array(2);
  backdropHeightBoundsInto(field, minX, minZ, maxX, maxZ, bounds);
  return [bounds[0]!, bounds[1]!];
}

/** The same bounds written to `bounds[0]` and `bounds[1]`, for a caller that asks per octree node. */
export function backdropHeightBoundsInto(
  field: BackdropField,
  minX: number, minZ: number, maxX: number, maxZ: number,
  bounds: Float64Array,
): void {
  const [fMinX, fMinZ, fMaxX, fMaxZ] = field.seam.footprint_m;
  // Nearest distance between the box and the footprint rectangle, and the
  // farthest (a corner, since distance to a convex set is convex).
  const gapX = Math.max(fMinX - maxX, 0, minX - fMaxX);
  const gapZ = Math.max(fMinZ - maxZ, 0, minZ - fMaxZ);
  const dNear = Math.hypot(gapX, gapZ);
  const dFar = Math.max(0,
    backdropFootprintDistance(field.seam, minX, minZ), backdropFootprintDistance(field.seam, minX, maxZ),
    backdropFootprintDistance(field.seam, maxX, minZ), backdropFootprintDistance(field.seam, maxX, maxZ));
  const { hills, outerRadius_m } = field.description;
  const rampEdge = hills.flatRing_m + hills.rampWidth_m;
  const rampNear = smoothstep(hills.flatRing_m, rampEdge, dNear), valleyNear = smoothstep(hills.flatRing_m, outerRadius_m, dNear);
  const rampFar = smoothstep(hills.flatRing_m, rampEdge, dFar), valleyFar = smoothstep(hills.flatRing_m, outerRadius_m, dFar);
  const seam = field.seam.height_m;
  const low = seam + rampNear * hills.valleyRise_m * valleyNear;
  const high = seam + rampFar * (hills.valleyRise_m * valleyFar + hills.amplitude_m);
  const cx = 0.5 * (minX + maxX), cz = 0.5 * (minZ + maxZ);
  const reach = field.lipschitz * 0.5 * Math.hypot(maxX - minX, maxZ - minZ);
  const centre = backdropFieldHeight(field, cx, cz);
  bounds[0] = Math.max(low, centre - reach);
  bounds[1] = Math.min(high, centre + reach);
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
