import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { decodeRgbPng, encodeRgbPng } from "../lib/png-codec";
import {
  cropToAspect,
  deltaE2000,
  gradientMagnitudeHistogram,
  GRADIENT_HISTOGRAM_BINS,
  HERO_FIDELITY_REGIONS,
  GRADIENT_BIN_OCTAVES,
  histogramEarthMoverOctaves,
  labFromLinearRgb,
  linearFromSrgbImage,
  linearToSrgb,
  registerToScoringGrid,
  resampleLinear,
  scoreFidelity,
  srgbImageFromLinear,
  srgbToLinear,
  type LinearImage,
} from "../lib/hero-fidelity-score";

// ---------------------------------------------------------------------------
// CIEDE2000
// ---------------------------------------------------------------------------

/**
 * Sharma, Wu & Dalal (2005), Table 1 — the 34 pairs that exist precisely
 * because the CIE's own text admits four readings and only one is right. Pairs
 * 9-16 are the ones that catch a hue-wrap bug; 17-20 catch a missing R_T.
 *
 * An implementation of ΔE₀₀ that has not been run against this table is not
 * known to be an implementation of ΔE₀₀, and every number this program reports
 * for H1 stands on it.
 */
const SHARMA_PAIRS: ReadonlyArray<readonly [number, number, number, number, number, number, number]> = [
  [50.0, 2.6772, -79.7751, 50.0, 0.0, -82.7485, 2.0425],
  [50.0, 3.1571, -77.2803, 50.0, 0.0, -82.7485, 2.8615],
  [50.0, 2.8361, -74.02, 50.0, 0.0, -82.7485, 3.4412],
  [50.0, -1.3802, -84.2814, 50.0, 0.0, -82.7485, 1.0],
  [50.0, -1.1848, -84.8006, 50.0, 0.0, -82.7485, 1.0],
  [50.0, -0.9009, -85.5211, 50.0, 0.0, -82.7485, 1.0],
  [50.0, 0.0, 0.0, 50.0, -1.0, 2.0, 2.3669],
  [50.0, -1.0, 2.0, 50.0, 0.0, 0.0, 2.3669],
  [50.0, 2.49, -0.001, 50.0, -2.49, 0.0009, 7.1792],
  [50.0, 2.49, -0.001, 50.0, -2.49, 0.001, 7.1792],
  [50.0, 2.49, -0.001, 50.0, -2.49, 0.0011, 7.2195],
  [50.0, 2.49, -0.001, 50.0, -2.49, 0.0012, 7.2195],
  [50.0, -0.001, 2.49, 50.0, 0.0009, -2.49, 4.8045],
  [50.0, -0.001, 2.49, 50.0, 0.001, -2.49, 4.8045],
  [50.0, -0.001, 2.49, 50.0, 0.0011, -2.49, 4.7461],
  [50.0, 2.5, 0.0, 50.0, 0.0, -2.5, 4.3065],
  [50.0, 2.5, 0.0, 73.0, 25.0, -18.0, 27.1492],
  [50.0, 2.5, 0.0, 61.0, -5.0, 29.0, 22.8977],
  [50.0, 2.5, 0.0, 56.0, -27.0, -3.0, 31.903],
  [50.0, 2.5, 0.0, 58.0, 24.0, 15.0, 19.4535],
  [50.0, 2.5, 0.0, 50.0, 3.1736, 0.5854, 1.0],
  [50.0, 2.5, 0.0, 50.0, 3.2972, 0.0, 1.0],
  [50.0, 2.5, 0.0, 50.0, 1.8634, 0.5757, 1.0],
  [50.0, 2.5, 0.0, 50.0, 3.2592, 0.335, 1.0],
  [60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644],
  [63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.263],
  [61.2901, 3.7196, -5.3901, 61.4292, 2.248, -4.962, 1.8731],
  [35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645],
  [22.7233, 20.0904, -46.694, 23.0331, 14.973, -42.5619, 2.0373],
  [36.4612, 47.858, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146],
  [90.8027, -2.0831, 1.441, 91.1528, -1.6435, 0.0447, 1.4441],
  [90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381],
  [6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377],
  [2.0776, 0.0795, -1.135, 0.9033, -0.0636, -0.5514, 0.9082],
];

test("deltaE2000 reproduces the Sharma reference table", () => {
  SHARMA_PAIRS.forEach(([L1, a1, b1, L2, a2, b2, expected], index) => {
    const actual = deltaE2000([L1, a1, b1], [L2, a2, b2]);
    assert.ok(
      Math.abs(actual - expected) < 1e-4,
      `pair ${index + 1}: expected ${expected}, got ${actual.toFixed(6)}`,
    );
  });
});

test("deltaE2000 is symmetric and zero on identity", () => {
  for (const [L1, a1, b1, L2, a2, b2] of SHARMA_PAIRS) {
    assert.equal(deltaE2000([L1, a1, b1], [L1, a1, b1]), 0);
    const forward = deltaE2000([L1, a1, b1], [L2, a2, b2]);
    const backward = deltaE2000([L2, a2, b2], [L1, a1, b1]);
    assert.ok(Math.abs(forward - backward) < 1e-9, `asymmetric: ${forward} vs ${backward}`);
  }
});

test("labFromLinearRgb puts the sRGB white at L*=100 and neutral chroma", () => {
  const [L, a, b] = labFromLinearRgb(1, 1, 1);
  assert.ok(Math.abs(L - 100) < 1e-6, `white L* was ${L}`);
  assert.ok(Math.hypot(a, b) < 1e-3, `white chroma was ${Math.hypot(a, b)}`);
  const [black] = labFromLinearRgb(0, 0, 0);
  assert.ok(Math.abs(black) < 1e-9, `black L* was ${black}`);
});

// ---------------------------------------------------------------------------
// sRGB transfer and the PNG container
// ---------------------------------------------------------------------------

test("srgbToLinear and linearToSrgb round-trip every byte", () => {
  for (let byte = 0; byte < 256; byte += 1) {
    assert.equal(linearToSrgb(srgbToLinear(byte)), byte, `byte ${byte} did not round-trip`);
  }
});

test("PNG encode/decode round-trips exact bytes", () => {
  const width = 37;
  const height = 19;
  const rgb = new Uint8Array(width * height * 3);
  for (let index = 0; index < rgb.length; index += 1) rgb[index] = (index * 97 + (index >> 3)) & 0xff;
  const decoded = decodeRgbPng(encodeRgbPng({ width, height, rgb }));
  assert.equal(decoded.width, width);
  assert.equal(decoded.height, height);
  assert.deepEqual(Array.from(decoded.rgb), Array.from(rgb));
});

test("the reference plate decodes at its recorded size", () => {
  const plate = decodeRgbPng(readFileSync("output/imagegen/garden-pond-hose-fill-simplified.png"));
  assert.equal(plate.width, 1672);
  assert.equal(plate.height, 941);
  // A plate that decoded as garbage would still have the right dimensions, so
  // check it is actually the bright porcelain set and not, say, all zeroes.
  let total = 0;
  for (const byte of plate.rgb) total += byte;
  const meanByte = total / plate.rgb.length;
  assert.ok(meanByte > 140 && meanByte < 230, `plate mean byte ${meanByte.toFixed(1)} is not a bright set`);
});

test("decodeRgbPng refuses what it cannot decode rather than guessing", () => {
  const good = encodeRgbPng({ width: 2, height: 2, rgb: new Uint8Array(12) });
  const notPng = Uint8Array.from(good);
  notPng[1] = 0;
  assert.throws(() => decodeRgbPng(notPng), /signature/);

  const sixteenBit = Uint8Array.from(good);
  // IHDR body starts at byte 16; bit depth is its 9th byte.
  sixteenBit[24] = 16;
  assert.throws(() => decodeRgbPng(sixteenBit), /bit depth/);
});

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

function constantImage(width: number, height: number, value: number): LinearImage {
  return { width, height, rgb: new Float32Array(width * height * 3).fill(value) };
}

test("resampleLinear preserves a constant field in both directions", () => {
  const source = constantImage(17, 11, 0.42);
  for (const [width, height] of [[5, 3], [34, 22], [17, 11]] as const) {
    const resampled = resampleLinear(source, width, height);
    assert.equal(resampled.width, width);
    for (const value of resampled.rgb) assert.ok(Math.abs(value - 0.42) < 1e-5, `got ${value}`);
  }
});

test("resampleLinear averages a checkerboard to its mean when halved", () => {
  const width = 16;
  const height = 16;
  const rgb = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = (x + y) % 2 === 0 ? 1 : 0;
      const at = (y * width + x) * 3;
      rgb[at] = value;
      rgb[at + 1] = value;
      rgb[at + 2] = value;
    }
  }
  const halved = resampleLinear({ width, height, rgb }, width / 2, height / 2);
  for (const value of halved.rgb) assert.ok(Math.abs(value - 0.5) < 1e-6, `expected 0.5, got ${value}`);
});

test("cropToAspect takes the centre and never stretches", () => {
  const source = constantImage(200, 100, 0.5);
  const narrower = cropToAspect(source, 1);
  assert.equal(narrower.width, 100);
  assert.equal(narrower.height, 100);
  const wider = cropToAspect(source, 4);
  assert.equal(wider.width, 200);
  assert.equal(wider.height, 50);
  // Already at the target aspect: returned untouched.
  assert.equal(cropToAspect(source, 2), source);
});

test("registerToScoringGrid puts the plate on the frame's grid, not the other way round", () => {
  const frame = constantImage(83, 47, 0.2);
  const plate = constantImage(1672, 941, 0.8);
  const registered = registerToScoringGrid(frame, plate);
  assert.equal(registered.frame.width, 83);
  assert.equal(registered.frame.height, 47);
  assert.equal(registered.plate.width, 83);
  assert.equal(registered.plate.height, 47);
});

// ---------------------------------------------------------------------------
// The gradient score
// ---------------------------------------------------------------------------

test("histogramEarthMoverOctaves measures the CDF displacement in octaves, with a sign", () => {
  const a = new Float64Array(GRADIENT_HISTOGRAM_BINS);
  const b = new Float64Array(GRADIENT_HISTOGRAM_BINS);
  a[0] = 1;
  b[0] = 1;
  assert.deepEqual(histogramEarthMoverOctaves(a, b), { distance: 0, bias: 0 });

  // All the frame's mass in the lowest bin against all the plate's in the
  // highest: the full span of the histogram, and the frame is the smooth one.
  b[0] = 0;
  b[GRADIENT_HISTOGRAM_BINS - 1] = 1;
  const fullSpan = (GRADIENT_HISTOGRAM_BINS - 1) * GRADIENT_BIN_OCTAVES;
  const extreme = histogramEarthMoverOctaves(a, b);
  assert.ok(Math.abs(extreme.distance - fullSpan) < 1e-9, `distance was ${extreme.distance}, expected ${fullSpan}`);
  assert.ok(Math.abs(extreme.bias - fullSpan) < 1e-9, "a frame with all its mass at low gradients reads as smoother");

  // Swapping the arguments flips the sign but not the magnitude.
  const swapped = histogramEarthMoverOctaves(b, a);
  assert.ok(Math.abs(swapped.distance - fullSpan) < 1e-9);
  assert.ok(Math.abs(swapped.bias + fullSpan) < 1e-9, "a frame noisier than the plate reads negative");
});

test("gradientMagnitudeHistogram puts a flat field in the lowest bin", () => {
  const field = new Float64Array(16 * 16);
  const histogram = gradientMagnitudeHistogram(field, 16, 16);
  assert.ok(Math.abs(histogram[0] - 1) < 1e-12, `flat field spread across bins: ${Array.from(histogram)}`);
});

/**
 * The load-bearing claim of the whole gate: the gradient score sees structure,
 * not tone. If this fails, H1's light rig would appear to buy detail and the
 * program's sequencing argument collapses.
 */
test("the gradient score is blind to exposure and colour cast; deltaE00 is not", () => {
  const width = 96;
  const height = 96;
  const rgb = new Float32Array(width * height * 3);
  // A deterministic multi-scale field, so it has content in several bins.
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value =
        0.35 +
        0.18 * Math.sin(x * 0.21) * Math.cos(y * 0.17) +
        0.09 * Math.sin(x * 0.93 + y * 0.71) +
        0.03 * Math.sin(x * 2.7 - y * 3.1);
      const at = (y * width + x) * 3;
      rgb[at] = value;
      rgb[at + 1] = value * 0.98;
      rgb[at + 2] = value * 0.95;
    }
  }
  const original: LinearImage = { width, height, rgb };
  // Half a stop down and warmed: a large tone change with identical structure.
  const graded: LinearImage = {
    width,
    height,
    rgb: Float32Array.from(rgb, (value, index) => value * 0.7 * (index % 3 === 2 ? 0.8 : 1)),
  };
  const wholeFrame = [{ id: "all", subject: "whole frame", x0: 0, y0: 0, x1: 1, y1: 1 }] as const;
  const report = scoreFidelity(graded, original, wholeFrame);
  const [score] = report.regions;
  assert.ok(score.deltaE00Mean > 5, `a half-stop warm grade should move deltaE00, got ${score.deltaE00Mean}`);
  assert.ok(
    score.gradientDistance < 0.02,
    `structure is identical, so the gradient distance should be ~0 octaves, got ${score.gradientDistance}`,
  );
});

test("the gradient score separates a detailed field from a smooth one", () => {
  const width = 96;
  const height = 96;
  const build = (octaves: number): LinearImage => {
    const rgb = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let value = 0.4 + 0.2 * Math.sin(x * 0.08) * Math.cos(y * 0.07);
        for (let octave = 1; octave <= octaves; octave += 1) {
          const frequency = 0.08 * 2 ** octave;
          value += (0.16 / 2 ** octave) * Math.sin(x * frequency + octave) * Math.cos(y * frequency * 1.3 - octave);
        }
        const at = (y * width + x) * 3;
        rgb[at] = value;
        rgb[at + 1] = value;
        rgb[at + 2] = value;
      }
    }
    return { width, height, rgb };
  };
  const wholeFrame = [{ id: "all", subject: "whole frame", x0: 0, y0: 0, x1: 1, y1: 1 }] as const;
  const detailed = build(4);
  const smooth = build(0);
  const against = scoreFidelity(smooth, detailed, wholeFrame).regions[0];
  const matching = scoreFidelity(detailed, detailed, wholeFrame).regions[0];
  assert.equal(matching.gradientDistance, 0);
  assert.equal(matching.gradientBias, 0);
  assert.ok(
    against.gradientDistance > 0.4,
    `a smooth field against a 4-octave one should sit a large fraction of an octave away, got ${against.gradientDistance}`,
  );
  assert.ok(
    against.gradientBias > 0.4,
    `the smooth field is the smoother of the pair, so the bias must be positive, got ${against.gradientBias}`,
  );
  assert.ok(
    against.gradientMeanFrame < against.gradientMeanPlate,
    "the smooth field should carry less standardised gradient than the detailed one",
  );
});

/**
 * Monotonicity is the property that makes the number usable as a gate: adding
 * a detail band to a frame must move it toward a detailed plate, every time.
 * A metric that merely separates two cases could still be non-monotone in
 * between, and a non-monotone gate would reject real progress.
 */
test("the gradient distance falls monotonically as detail bands are added", () => {
  const width = 96;
  const height = 96;
  const build = (octaves: number): LinearImage => {
    const rgb = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let value = 0.4 + 0.2 * Math.sin(x * 0.08) * Math.cos(y * 0.07);
        for (let octave = 1; octave <= octaves; octave += 1) {
          const frequency = 0.08 * 2 ** octave;
          value += (0.16 / 2 ** octave) * Math.sin(x * frequency + octave) * Math.cos(y * frequency * 1.3 - octave);
        }
        const at = (y * width + x) * 3;
        rgb[at] = value;
        rgb[at + 1] = value;
        rgb[at + 2] = value;
      }
    }
    return { width, height, rgb };
  };
  const wholeFrame = [{ id: "all", subject: "whole frame", x0: 0, y0: 0, x1: 1, y1: 1 }] as const;
  const plate = build(5);
  const distances = [0, 1, 2, 3, 4, 5].map(
    (octaves) => scoreFidelity(build(octaves), plate, wholeFrame).regions[0].gradientDistance,
  );
  for (let index = 1; index < distances.length; index += 1) {
    assert.ok(
      distances[index] <= distances[index - 1] + 1e-9,
      `adding band ${index} moved the score away from the plate: ${distances.join(", ")}`,
    );
  }
  assert.equal(distances[distances.length - 1], 0);
});

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

test("every hero region is a non-degenerate rectangle inside the frame", () => {
  const seen = new Set<string>();
  for (const region of HERO_FIDELITY_REGIONS) {
    assert.ok(!seen.has(region.id), `duplicate region id ${region.id}`);
    seen.add(region.id);
    assert.ok(region.x1 > region.x0, `${region.id} has non-positive width`);
    assert.ok(region.y1 > region.y0, `${region.id} has non-positive height`);
    for (const value of [region.x0, region.y0, region.x1, region.y1]) {
      assert.ok(value >= 0 && value <= 1, `${region.id} escapes the frame: ${value}`);
    }
    assert.ok(region.subject.length > 10, `${region.id} needs a subject line that says what it is`);
  }
  assert.equal(HERO_FIDELITY_REGIONS.length, 5);
});

test("scoreFidelity refuses a frame and plate that are not on one grid", () => {
  assert.throws(
    () => scoreFidelity(constantImage(10, 10, 0.5), constantImage(11, 10, 0.5)),
    /not on the same grid/,
  );
});

test("linear and sRGB image conversions round-trip through the byte grid", () => {
  const width = 8;
  const height = 4;
  const rgb = new Uint8Array(width * height * 3);
  for (let index = 0; index < rgb.length; index += 1) rgb[index] = (index * 13) & 0xff;
  const restored = srgbImageFromLinear(linearFromSrgbImage({ width, height, rgb }));
  assert.deepEqual(Array.from(restored.rgb), Array.from(rgb));
});
