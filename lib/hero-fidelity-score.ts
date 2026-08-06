/**
 * The hero garden's fidelity score — H0 of `docs/hero-fidelity-1000x-handoff.md`.
 *
 * The plate is `output/imagegen/garden-pond-hose-fill-simplified.png` — already
 * in the tree, already named as the reference by `lib/hero-garden-scene.ts:18`.
 * It is 1672 x 941 and it is what every number below is measured against.
 *
 * "Pixel perfect" is unfalsifiable without a number, and every change after H0
 * is an opinion without one. This module is that number, and it is deliberately
 * *two* numbers, because the two failures this program can suffer are
 * independent and one metric cannot see both:
 *
 *   - **ΔE₀₀** (CIEDE2000, per region) answers *is the tone and colour right*.
 *     It is what H1's light rig and grading move, and it is close to blind to
 *     whether the surface has any detail at all: a perfectly lit smooth
 *     ellipsoid scores well on it.
 *
 *   - **Gradient-magnitude histogram distance** answers *does the frame have
 *     the plate's spatial-frequency content*. This is the one that measures
 *     "1000x". It is computed on **standardised L\***, so a frame that is
 *     uniformly darker or warmer than the plate scores identically to one that
 *     is not — it sees structure, not tone. A smooth ellipsoid scores badly on
 *     it no matter how it is lit, which is precisely the H2 signal.
 *
 * Keeping them separate is the point. A single blended score would let H1's
 * gains hide H2's absence, and the handoff's §0 table exists because those two
 * gaps have to be tracked apart.
 *
 * ## Why standardise before differencing gradients
 *
 * The naive version differences raw luminance gradients, and then every score
 * moves when the exposure moves. That would make H1 *appear* to buy detail,
 * which is exactly the false positive that would mis-sequence the rest of the
 * program. Standardising L\* per region to zero mean and unit variance removes
 * gain and offset, leaving only how the energy is distributed across spatial
 * frequency.
 *
 * ## Why EMD rather than chi-square
 *
 * The histogram bins are ordered — a frame whose detail sits one bin coarser
 * than the plate's is nearly right, and a frame with no detail at all is not.
 * A bin-wise divergence scores both as "different bins" and cannot tell them
 * apart. The 1-D earth-mover distance is the L1 distance between the CDFs, it
 * respects that ordering, and normalised by the bin count it lands in [0, 1]
 * where 0 is identical structure and 1 is "all the detail is at the opposite
 * end of the scale".
 */

import { decodeRgbPng, encodeRgbPng, type RgbImage } from "./png-codec";

// ---------------------------------------------------------------------------
// Linear-light images
// ---------------------------------------------------------------------------

/** An image in scene-linear (not sRGB-encoded) RGB, row-major, top row first. */
export interface LinearImage {
  readonly width: number;
  readonly height: number;
  /** `width * height * 3` values, linear, nominally in [0, 1]. */
  readonly rgb: Float32Array;
}

/**
 * sRGB EOTF. Resampling and averaging happen in linear light throughout — the
 * classic way to lose a third of a stop is to box-filter sRGB bytes.
 */
export function srgbToLinear(encoded: number): number {
  const value = encoded / 255;
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

/** sRGB inverse EOTF, back to a byte. */
export function linearToSrgb(linear: number): number {
  const clamped = Math.min(1, Math.max(0, linear));
  const encoded = clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055;
  return Math.round(encoded * 255);
}

const SRGB_TO_LINEAR_TABLE = (() => {
  const table = new Float32Array(256);
  for (let index = 0; index < 256; index += 1) table[index] = srgbToLinear(index);
  return table;
})();

export function linearFromSrgbImage(image: RgbImage): LinearImage {
  const rgb = new Float32Array(image.width * image.height * 3);
  for (let index = 0; index < rgb.length; index += 1) rgb[index] = SRGB_TO_LINEAR_TABLE[image.rgb[index]];
  return { width: image.width, height: image.height, rgb };
}

export function srgbImageFromLinear(image: LinearImage): RgbImage {
  const rgb = new Uint8Array(image.rgb.length);
  for (let index = 0; index < rgb.length; index += 1) rgb[index] = linearToSrgb(image.rgb[index]);
  return { width: image.width, height: image.height, rgb };
}

export function decodeLinearPng(bytes: Uint8Array): LinearImage {
  return linearFromSrgbImage(decodeRgbPng(bytes));
}

export function encodeLinearPng(image: LinearImage): Uint8Array {
  return encodeRgbPng(srgbImageFromLinear(image));
}

/**
 * Area-average resample, in linear light, correct in both directions.
 *
 * Each destination pixel integrates the source over its exact footprint, so a
 * 2x downsample is a true box filter and an upsample degrades to nearest-with-
 * partial-coverage rather than to aliasing. The plate is 1672x941 and the frame
 * is smaller; getting this wrong would put the plate's fine detail into the
 * score as noise, which would flatter the frame.
 */
export function resampleLinear(image: LinearImage, width: number, height: number): LinearImage {
  if (width <= 0 || height <= 0) throw new RangeError(`resample target must be positive, received ${width}x${height}`);
  if (width === image.width && height === image.height) return image;
  const out = new Float32Array(width * height * 3);
  const scaleX = image.width / width;
  const scaleY = image.height / height;
  for (let y = 0; y < height; y += 1) {
    const sourceTop = y * scaleY;
    const sourceBottom = (y + 1) * scaleY;
    const firstRow = Math.floor(sourceTop);
    const lastRow = Math.min(image.height - 1, Math.ceil(sourceBottom) - 1);
    for (let x = 0; x < width; x += 1) {
      const sourceLeft = x * scaleX;
      const sourceRight = (x + 1) * scaleX;
      const firstColumn = Math.floor(sourceLeft);
      const lastColumn = Math.min(image.width - 1, Math.ceil(sourceRight) - 1);
      let r = 0;
      let g = 0;
      let b = 0;
      let weightTotal = 0;
      for (let row = firstRow; row <= lastRow; row += 1) {
        const rowWeight = Math.min(row + 1, sourceBottom) - Math.max(row, sourceTop);
        if (rowWeight <= 0) continue;
        for (let column = firstColumn; column <= lastColumn; column += 1) {
          const columnWeight = Math.min(column + 1, sourceRight) - Math.max(column, sourceLeft);
          if (columnWeight <= 0) continue;
          const weight = rowWeight * columnWeight;
          const source = (row * image.width + column) * 3;
          r += image.rgb[source] * weight;
          g += image.rgb[source + 1] * weight;
          b += image.rgb[source + 2] * weight;
          weightTotal += weight;
        }
      }
      const destination = (y * width + x) * 3;
      const inverse = weightTotal > 0 ? 1 / weightTotal : 0;
      out[destination] = r * inverse;
      out[destination + 1] = g * inverse;
      out[destination + 2] = b * inverse;
    }
  }
  return { width, height, rgb: out };
}

/**
 * Centre-crop to an aspect ratio.
 *
 * The plate and the render lane do not share an aspect (1.7768 vs 1.7391), and
 * stretching one to the other would shear every gradient direction — which the
 * gradient score would then read as a structure difference that is really a
 * registration error. Cropping the wider one is the honest reconciliation, and
 * it is why the scoring grid is derived rather than authored.
 */
export function cropToAspect(image: LinearImage, aspect: number): LinearImage {
  const currentAspect = image.width / image.height;
  let cropWidth = image.width;
  let cropHeight = image.height;
  if (currentAspect > aspect) cropWidth = Math.round(image.height * aspect);
  else if (currentAspect < aspect) cropHeight = Math.round(image.width / aspect);
  if (cropWidth === image.width && cropHeight === image.height) return image;
  const originX = Math.floor((image.width - cropWidth) / 2);
  const originY = Math.floor((image.height - cropHeight) / 2);
  const rgb = new Float32Array(cropWidth * cropHeight * 3);
  for (let y = 0; y < cropHeight; y += 1) {
    const source = ((y + originY) * image.width + originX) * 3;
    rgb.set(image.rgb.subarray(source, source + cropWidth * 3), y * cropWidth * 3);
  }
  return { width: cropWidth, height: cropHeight, rgb };
}

// ---------------------------------------------------------------------------
// CIELAB and CIEDE2000
// ---------------------------------------------------------------------------

/**
 * D65 white, derived from the sRGB matrix rather than quoted beside it.
 *
 * The quoted D65 (0.95047, 1, 1.08883) is not *exactly* what these matrix rows
 * sum to — the Y row sums to 1.0000001 — and dividing by the quoted value
 * instead leaves pure white at L* = 100.000004 rather than 100. That is
 * harmless for a ΔE, and it is a needless place for a reader to have to decide
 * whether a number is a bug. Summing the rows makes white exact by
 * construction.
 */
const SRGB_TO_XYZ = Object.freeze([
  Object.freeze([0.4124564, 0.3575761, 0.1804375]),
  Object.freeze([0.2126729, 0.7151522, 0.0721750]),
  Object.freeze([0.0193339, 0.1191920, 0.9503041]),
] as const);
const WHITE_X = SRGB_TO_XYZ[0][0] + SRGB_TO_XYZ[0][1] + SRGB_TO_XYZ[0][2];
const WHITE_Y = SRGB_TO_XYZ[1][0] + SRGB_TO_XYZ[1][1] + SRGB_TO_XYZ[1][2];
const WHITE_Z = SRGB_TO_XYZ[2][0] + SRGB_TO_XYZ[2][1] + SRGB_TO_XYZ[2][2];

function labTransfer(t: number): number {
  return t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27) * t / 116 + 16 / 116;
}

/** Linear sRGB primaries to CIELAB (D65). */
export function labFromLinearRgb(r: number, g: number, b: number): [number, number, number] {
  const x = (SRGB_TO_XYZ[0][0] * r + SRGB_TO_XYZ[0][1] * g + SRGB_TO_XYZ[0][2] * b) / WHITE_X;
  const y = (SRGB_TO_XYZ[1][0] * r + SRGB_TO_XYZ[1][1] * g + SRGB_TO_XYZ[1][2] * b) / WHITE_Y;
  const z = (SRGB_TO_XYZ[2][0] * r + SRGB_TO_XYZ[2][1] * g + SRGB_TO_XYZ[2][2] * b) / WHITE_Z;
  const fx = labTransfer(x);
  const fy = labTransfer(y);
  const fz = labTransfer(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

const DEGREES = Math.PI / 180;

/**
 * CIEDE2000 colour difference.
 *
 * Transcribed from Sharma, Wu & Dalal (2005), including the hue-rotation and
 * the mean-hue wrap conventions that the naive reading of the CIE text gets
 * wrong. `tests/hero-fidelity-score.test.ts` checks it against that paper's
 * published 34-pair table, which is the only way to know an implementation of
 * this is right — the formula has four places where a plausible reading is
 * silently off by a degree or two.
 */
export function deltaE2000(
  lab1: readonly [number, number, number],
  lab2: readonly [number, number, number],
): number {
  const [L1, a1, b1] = lab1;
  const [L2, a2, b2] = lab2;

  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const meanC = (C1 + C2) / 2;
  const meanC7 = meanC ** 7;
  const G = 0.5 * (1 - Math.sqrt(meanC7 / (meanC7 + 25 ** 7)));

  const a1p = (1 + G) * a1;
  const a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);

  const h1p = C1p === 0 ? 0 : ((Math.atan2(b1, a1p) / DEGREES) + 360) % 360;
  const h2p = C2p === 0 ? 0 : ((Math.atan2(b2, a2p) / DEGREES) + 360) % 360;

  const deltaLp = L2 - L1;
  const deltaCp = C2p - C1p;

  let deltahp: number;
  if (C1p * C2p === 0) deltahp = 0;
  else {
    const difference = h2p - h1p;
    if (Math.abs(difference) <= 180) deltahp = difference;
    else if (difference > 180) deltahp = difference - 360;
    else deltahp = difference + 360;
  }
  const deltaHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((deltahp / 2) * DEGREES);

  const meanLp = (L1 + L2) / 2;
  const meanCp = (C1p + C2p) / 2;

  let meanhp: number;
  if (C1p * C2p === 0) meanhp = h1p + h2p;
  else {
    const sum = h1p + h2p;
    const difference = Math.abs(h1p - h2p);
    if (difference <= 180) meanhp = sum / 2;
    else if (sum < 360) meanhp = (sum + 360) / 2;
    else meanhp = (sum - 360) / 2;
  }

  const T =
    1 -
    0.17 * Math.cos((meanhp - 30) * DEGREES) +
    0.24 * Math.cos(2 * meanhp * DEGREES) +
    0.32 * Math.cos((3 * meanhp + 6) * DEGREES) -
    0.20 * Math.cos((4 * meanhp - 63) * DEGREES);

  const deltaTheta = 30 * Math.exp(-(((meanhp - 275) / 25) ** 2));
  const meanCp7 = meanCp ** 7;
  const RC = 2 * Math.sqrt(meanCp7 / (meanCp7 + 25 ** 7));
  const SL = 1 + (0.015 * (meanLp - 50) ** 2) / Math.sqrt(20 + (meanLp - 50) ** 2);
  const SC = 1 + 0.045 * meanCp;
  const SH = 1 + 0.015 * meanCp * T;
  const RT = -Math.sin(2 * deltaTheta * DEGREES) * RC;

  const termL = deltaLp / SL;
  const termC = deltaCp / SC;
  const termH = deltaHp / SH;
  return Math.sqrt(termL * termL + termC * termC + termH * termH + RT * termC * termH);
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/**
 * A scored region, in normalised coordinates of the *registered* grid — the
 * grid both the plate and the frame have been resampled onto, so one rectangle
 * addresses the same subject in both.
 */
export interface FidelityRegion {
  readonly id: string;
  /** What the region is for, in one line — this is read off the contact sheet. */
  readonly subject: string;
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

/**
 * The hero garden's five scored regions, ordered as §0 of the handoff orders
 * the gaps.
 *
 * These are registered against `docs/plates/hero-garden-hose-reference.png` and
 * are deliberately *interior* rectangles rather than tight object bounds: a
 * rectangle that clips an object's silhouette makes the gradient score depend
 * on the object's position to within a pixel, and the camera solve is not that
 * good. Each one is a patch of one subject with margin.
 */
export const HERO_FIDELITY_REGIONS: readonly FidelityRegion[] = Object.freeze([
  Object.freeze({
    id: "pond",
    subject: "open pond surface: refraction, caustic net, ripple rings — the H6 region",
    x0: 0.33, y0: 0.48, x1: 0.62, y1: 0.72,
  }),
  Object.freeze({
    id: "coping",
    subject: "near coping: the plate's brightest plaster, and H1's cleanest tone target",
    x0: 0.22, y0: 0.92, x1: 0.55, y1: 0.995,
  }),
  Object.freeze({
    id: "stones",
    subject: "mushroom stones and their pebble bed — the H2 form target",
    x0: 0.055, y0: 0.20, x1: 0.34, y1: 0.50,
  }),
  Object.freeze({
    id: "canopy",
    subject: "coral canopy: the recursive-scatter target (H2 op 7)",
    x0: 0.46, y0: 0.01, x1: 0.80, y1: 0.15,
  }),
  Object.freeze({
    id: "ground",
    subject: "backdrop and floor sweep — the H2b cove region",
    x0: 0.03, y0: 0.02, x1: 0.22, y1: 0.14,
  }),
]);

interface RegionBounds {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export function regionBounds(region: FidelityRegion, width: number, height: number): RegionBounds {
  const x = Math.max(0, Math.min(width - 1, Math.round(region.x0 * width)));
  const y = Math.max(0, Math.min(height - 1, Math.round(region.y0 * height)));
  const x1 = Math.max(x + 1, Math.min(width, Math.round(region.x1 * width)));
  const y1 = Math.max(y + 1, Math.min(height, Math.round(region.y1 * height)));
  return { x, y, width: x1 - x, height: y1 - y };
}

// ---------------------------------------------------------------------------
// The two scores
// ---------------------------------------------------------------------------

/** Bins of the gradient-magnitude histogram, log2-spaced over standardised L*. */
export const GRADIENT_HISTOGRAM_BINS = 32;
const GRADIENT_LOG2_MINIMUM = -8;
const GRADIENT_LOG2_MAXIMUM = 4;
/** Width of one bin, in octaves of gradient magnitude. */
export const GRADIENT_BIN_OCTAVES = (GRADIENT_LOG2_MAXIMUM - GRADIENT_LOG2_MINIMUM) / GRADIENT_HISTOGRAM_BINS;

/**
 * L\* over a region, standardised to zero mean and unit variance.
 *
 * This is the step that makes the gradient score blind to exposure and to a
 * colour cast, and it is why H1 cannot flatter H2's number.
 */
function standardisedLightness(image: LinearImage, bounds: RegionBounds): Float64Array {
  const field = new Float64Array(bounds.width * bounds.height);
  let total = 0;
  for (let y = 0; y < bounds.height; y += 1) {
    for (let x = 0; x < bounds.width; x += 1) {
      const source = ((y + bounds.y) * image.width + (x + bounds.x)) * 3;
      const [L] = labFromLinearRgb(image.rgb[source], image.rgb[source + 1], image.rgb[source + 2]);
      field[y * bounds.width + x] = L;
      total += L;
    }
  }
  const mean = total / field.length;
  let variance = 0;
  for (const value of field) variance += (value - mean) ** 2;
  variance /= field.length;
  // A perfectly flat region has no structure to compare; unit scale keeps it
  // finite and its histogram collapses into the lowest bin, which is the
  // truthful answer for "this surface has no detail".
  const scale = variance > 1e-12 ? 1 / Math.sqrt(variance) : 1;
  for (let index = 0; index < field.length; index += 1) field[index] = (field[index] - mean) * scale;
  return field;
}

/** Sobel magnitude histogram over a standardised field, normalised to sum 1. */
export function gradientMagnitudeHistogram(
  field: Float64Array,
  width: number,
  height: number,
  bins = GRADIENT_HISTOGRAM_BINS,
): Float64Array {
  const histogram = new Float64Array(bins);
  const step = (GRADIENT_LOG2_MAXIMUM - GRADIENT_LOG2_MINIMUM) / bins;
  let samples = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const at = (dx: number, dy: number) => field[(y + dy) * width + (x + dx)];
      const gx =
        at(-1, -1) + 2 * at(-1, 0) + at(-1, 1) - at(1, -1) - 2 * at(1, 0) - at(1, 1);
      const gy =
        at(-1, -1) + 2 * at(0, -1) + at(1, -1) - at(-1, 1) - 2 * at(0, 1) - at(1, 1);
      const magnitude = Math.hypot(gx, gy) / 8;
      const log2Magnitude = magnitude > 0 ? Math.log2(magnitude) : GRADIENT_LOG2_MINIMUM;
      const bin = Math.max(
        0,
        Math.min(bins - 1, Math.floor((log2Magnitude - GRADIENT_LOG2_MINIMUM) / step)),
      );
      histogram[bin] += 1;
      samples += 1;
    }
  }
  if (samples > 0) for (let bin = 0; bin < bins; bin += 1) histogram[bin] /= samples;
  return histogram;
}

/**
 * 1-D earth-mover distance between two histograms that sum to 1, **in octaves
 * of gradient magnitude**, together with its signed counterpart.
 *
 * The units are the point. A distance expressed as a fraction of full scale is
 * unreadable — every real pair of frames lands in 0.00–0.10 and nobody can say
 * whether 0.044 is a lot. Multiplying the CDF displacement by the bin width
 * instead gives *how far the frame's gradient energy sits from the plate's,
 * measured in octaves*, and half an octave is a sentence a person can act on.
 *
 * `bias` keeps the sign that `distance` throws away, and it is the more useful
 * of the two for this program: **positive means the frame is smoother than the
 * plate** — its gradient mass sits at coarser scales — which is exactly the H2
 * signal. Negative means the frame is noisier than the plate, which is the H5
 * signal (aliasing reads as detail to any gradient metric, so a frame can
 * improve on `distance` by shimmering, and only the sign catches it).
 */
export function histogramEarthMoverOctaves(
  a: Float64Array,
  b: Float64Array,
): { readonly distance: number; readonly bias: number } {
  if (a.length !== b.length) throw new RangeError(`histogram lengths differ: ${a.length} vs ${b.length}`);
  let cumulative = 0;
  let absolute = 0;
  let signed = 0;
  for (let bin = 0; bin < a.length; bin += 1) {
    cumulative += a[bin] - b[bin];
    absolute += Math.abs(cumulative);
    signed += cumulative;
  }
  return { distance: absolute * GRADIENT_BIN_OCTAVES, bias: signed * GRADIENT_BIN_OCTAVES };
}

export interface RegionFidelityScore {
  readonly id: string;
  readonly subject: string;
  readonly bounds: RegionBounds;
  /** Mean CIEDE2000 over the region. The H1 number. */
  readonly deltaE00Mean: number;
  /** 95th percentile CIEDE2000 — catches a region that is right on average and wrong in a corner. */
  readonly deltaE00P95: number;
  /** EMD between the frame's and the plate's gradient histograms, in octaves. The H2 number. */
  readonly gradientDistance: number;
  /** Signed: positive means the frame is smoother than the plate, negative noisier. */
  readonly gradientBias: number;
  /** Mean standardised gradient magnitude, frame then plate — the raw detail density, for diagnosis. */
  readonly gradientMeanFrame: number;
  readonly gradientMeanPlate: number;
}

export interface FidelityReport {
  readonly width: number;
  readonly height: number;
  readonly regions: readonly RegionFidelityScore[];
  /** Unweighted means across regions — the two headline numbers. */
  readonly deltaE00Mean: number;
  readonly gradientDistanceMean: number;
}

function meanGradientMagnitude(field: Float64Array, width: number, height: number): number {
  let total = 0;
  let samples = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const at = (dx: number, dy: number) => field[(y + dy) * width + (x + dx)];
      const gx = at(-1, -1) + 2 * at(-1, 0) + at(-1, 1) - at(1, -1) - 2 * at(1, 0) - at(1, 1);
      const gy = at(-1, -1) + 2 * at(0, -1) + at(1, -1) - at(-1, 1) - 2 * at(0, 1) - at(1, 1);
      total += Math.hypot(gx, gy) / 8;
      samples += 1;
    }
  }
  return samples > 0 ? total / samples : 0;
}

/**
 * Score a frame against a plate. Both must already be on the same grid — use
 * `registerToScoringGrid` to get there, so the resampling decision is made in
 * one place and recorded.
 */
export function scoreFidelity(
  frame: LinearImage,
  plate: LinearImage,
  regions: readonly FidelityRegion[] = HERO_FIDELITY_REGIONS,
): FidelityReport {
  if (frame.width !== plate.width || frame.height !== plate.height) {
    throw new RangeError(
      `frame ${frame.width}x${frame.height} and plate ${plate.width}x${plate.height} are not on the same grid`,
    );
  }
  const scores: RegionFidelityScore[] = [];
  for (const region of regions) {
    const bounds = regionBounds(region, frame.width, frame.height);
    const differences = new Float64Array(bounds.width * bounds.height);
    for (let y = 0; y < bounds.height; y += 1) {
      for (let x = 0; x < bounds.width; x += 1) {
        const source = ((y + bounds.y) * frame.width + (x + bounds.x)) * 3;
        const frameLab = labFromLinearRgb(frame.rgb[source], frame.rgb[source + 1], frame.rgb[source + 2]);
        const plateLab = labFromLinearRgb(plate.rgb[source], plate.rgb[source + 1], plate.rgb[source + 2]);
        differences[y * bounds.width + x] = deltaE2000(frameLab, plateLab);
      }
    }
    const sorted = Float64Array.from(differences).sort();
    const mean = differences.reduce((sum, value) => sum + value, 0) / differences.length;
    const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];

    const frameField = standardisedLightness(frame, bounds);
    const plateField = standardisedLightness(plate, bounds);
    const frameHistogram = gradientMagnitudeHistogram(frameField, bounds.width, bounds.height);
    const plateHistogram = gradientMagnitudeHistogram(plateField, bounds.width, bounds.height);

    const earthMover = histogramEarthMoverOctaves(frameHistogram, plateHistogram);
    scores.push({
      id: region.id,
      subject: region.subject,
      bounds,
      deltaE00Mean: mean,
      deltaE00P95: p95,
      gradientDistance: earthMover.distance,
      gradientBias: earthMover.bias,
      gradientMeanFrame: meanGradientMagnitude(frameField, bounds.width, bounds.height),
      gradientMeanPlate: meanGradientMagnitude(plateField, bounds.width, bounds.height),
    });
  }
  return {
    width: frame.width,
    height: frame.height,
    regions: scores,
    deltaE00Mean: scores.reduce((sum, score) => sum + score.deltaE00Mean, 0) / Math.max(1, scores.length),
    gradientDistanceMean: scores.reduce((sum, score) => sum + score.gradientDistance, 0) / Math.max(1, scores.length),
  };
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/**
 * How well two images' *edges* line up, in [-1, 1]. This is the camera solve's
 * objective, and it is a different question from either score above.
 *
 * Neither ΔE₀₀ nor the gradient histogram can solve a camera. ΔE₀₀ is a
 * per-pixel tone difference between a flat white render and a lit colour plate,
 * so it is dominated by the tone gap and barely moves when the framing does.
 * The gradient histogram is a *distribution* — it deliberately discards where
 * the gradients are, which is the only thing registration cares about.
 *
 * So: standardised L\*, Sobel magnitude, a box blur, and a normalised cross
 * correlation. The blur is load-bearing. Without it the objective is a field of
 * needles — the plate has detail the render cannot have at any framing, so
 * unblurred edge maps correlate near zero everywhere and the search has no
 * gradient to walk. Blurring to a few pixels leaves the silhouettes, which are
 * the features the two images genuinely share.
 */
export function edgeAlignment(frame: LinearImage, plate: LinearImage, blurRadius = 3): number {
  if (frame.width !== plate.width || frame.height !== plate.height) {
    throw new RangeError(`edgeAlignment needs one grid, got ${frame.width}x${frame.height} and ${plate.width}x${plate.height}`);
  }
  const bounds = { x: 0, y: 0, width: frame.width, height: frame.height };
  const frameEdges = blurred(sobelMagnitude(standardisedLightness(frame, bounds), frame.width, frame.height), frame.width, frame.height, blurRadius);
  const plateEdges = blurred(sobelMagnitude(standardisedLightness(plate, bounds), plate.width, plate.height), plate.width, plate.height, blurRadius);
  return normalisedCrossCorrelation(frameEdges, plateEdges);
}

function sobelMagnitude(field: Float64Array, width: number, height: number): Float64Array {
  const out = new Float64Array(width * height);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const at = (dx: number, dy: number) => field[(y + dy) * width + (x + dx)];
      const gx = at(-1, -1) + 2 * at(-1, 0) + at(-1, 1) - at(1, -1) - 2 * at(1, 0) - at(1, 1);
      const gy = at(-1, -1) + 2 * at(0, -1) + at(1, -1) - at(-1, 1) - 2 * at(0, 1) - at(1, 1);
      out[y * width + x] = Math.hypot(gx, gy) / 8;
    }
  }
  return out;
}

/** Separable box blur, clamped at the border. */
function blurred(field: Float64Array, width: number, height: number, radius: number): Float64Array {
  if (radius <= 0) return field;
  const horizontal = new Float64Array(field.length);
  const span = 2 * radius + 1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let total = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        total += field[y * width + Math.min(width - 1, Math.max(0, x + offset))];
      }
      horizontal[y * width + x] = total / span;
    }
  }
  const out = new Float64Array(field.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let total = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        total += horizontal[Math.min(height - 1, Math.max(0, y + offset)) * width + x];
      }
      out[y * width + x] = total / span;
    }
  }
  return out;
}

function normalisedCrossCorrelation(a: Float64Array, b: Float64Array): number {
  let meanA = 0;
  let meanB = 0;
  for (let index = 0; index < a.length; index += 1) {
    meanA += a[index];
    meanB += b[index];
  }
  meanA /= a.length;
  meanB /= b.length;
  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (let index = 0; index < a.length; index += 1) {
    const da = a[index] - meanA;
    const db = b[index] - meanB;
    covariance += da * db;
    varianceA += da * da;
    varianceB += db * db;
  }
  const denominator = Math.sqrt(varianceA * varianceB);
  return denominator > 0 ? covariance / denominator : 0;
}

/**
 * Put a frame and a plate on one grid.
 *
 * The grid is the *frame's* aspect at the frame's size: the render lane's
 * framing is the thing under test, and resampling the frame up to the plate
 * would invent detail it does not have and flatter the gradient score.
 */
export function registerToScoringGrid(
  frame: LinearImage,
  plate: LinearImage,
): { readonly frame: LinearImage; readonly plate: LinearImage } {
  const aspect = frame.width / frame.height;
  const cropped = cropToAspect(plate, aspect);
  return { frame, plate: resampleLinear(cropped, frame.width, frame.height) };
}
