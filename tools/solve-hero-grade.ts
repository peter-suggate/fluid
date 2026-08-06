#!/usr/bin/env node
/**
 * Solve the hero garden's exposure and white balance against the plate — H1 of
 * `docs/hero-fidelity-1000x-handoff.md`.
 *
 *   npm run solve:hero-grade
 *
 * H1 is the handoff's "largest delta per day", and the measurement that says
 * why is blunt. Per-region mean CIELAB, frame against plate, at the registered
 * camera:
 *
 *   region     L* frame   L* plate    b* frame   b* plate
 *   coping        86.8       60.5         1.1        5.2
 *   ground        86.9       58.6         1.0        6.0
 *   stones        83.9       69.2         0.4        3.8
 *
 * The frame is roughly 27 L\* too bright everywhere and neutral where the plate
 * is warm. Both are grade knobs, so neither needs a light moved — and both can
 * be solved without the GPU, because `FLUID_SVO_DRY_SMOKE_RAW` dumps the frame
 * scene-linear and the grade is a pure function of that buffer. A candidate
 * costs a pass over 393k pixels rather than a render.
 *
 * ---------------------------------------------------------------------------
 * What it optimises, and what it deliberately ignores
 * ---------------------------------------------------------------------------
 * The objective is mean ΔE₀₀ over the **neutral** regions only — coping,
 * ground, stones. The pond is excluded and that is not a convenience: the plate's
 * pond is teal water at a\* ≈ −13 and the frame has no water at all (gap #1,
 * ~55 % of the plate, H6's whole workstream). Including it would drag the
 * balance cyan to chase a colour that belongs to a fluid nobody has rendered
 * yet, and every other region would be worse for it.
 *
 * The canopy is excluded for the same class of reason: at b\* ≈ 15 it is the
 * warmest thing in the plate by a factor of three, and it is warm because it is
 * a different *material*, not because the frame is mis-balanced.
 *
 * ---------------------------------------------------------------------------
 * Environment
 * ---------------------------------------------------------------------------
 *   FLUID_HERO_GRADE_RAW      packed rgba16float frame (default artifacts/hero-grade/frame.raw)
 *   FLUID_HERO_GRADE_WIDTH / _HEIGHT   its dimensions (default 836 x 470)
 *   FLUID_HERO_GRADE_PLATE    plate path
 *   FLUID_HERO_GRADE_CURVE    `aces` (default) | `reinhard`
 *   FLUID_HERO_GRADE_OUT      report path (default artifacts/hero-grade/solution.json)
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";

import { resolveDisplayGrade, type SceneToneCurve } from "../lib/webgpu-lighting";
import {
  decodeLinearPng,
  deltaE2000,
  HERO_FIDELITY_REGIONS,
  labFromLinearRgb,
  regionBounds,
  registerToScoringGrid,
  srgbToLinear,
  type LinearImage,
} from "../lib/hero-fidelity-score";

const environment = process.env;
const rawPath = environment.FLUID_HERO_GRADE_RAW ?? "artifacts/hero-grade/frame.raw";
const width = Number(environment.FLUID_HERO_GRADE_WIDTH ?? 836);
const height = Number(environment.FLUID_HERO_GRADE_HEIGHT ?? 470);
const platePath = environment.FLUID_HERO_GRADE_PLATE ?? "output/imagegen/garden-pond-hose-fill-simplified.png";
const toneCurve = (environment.FLUID_HERO_GRADE_CURVE ?? "aces") as SceneToneCurve;
const outPath = environment.FLUID_HERO_GRADE_OUT ?? "artifacts/hero-grade/solution.json";

/** Regions the grade is answerable for. See the module note on the two exclusions. */
const NEUTRAL_REGIONS = new Set(["coping", "ground", "stones"]);

// ---------------------------------------------------------------------------
// The scene-linear frame
// ---------------------------------------------------------------------------

function decodeF16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? Number.NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

const rawBytes = readFileSync(rawPath);
const halfWords = new Uint16Array(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength / 2);
if (halfWords.length < width * height * 4) {
  throw new RangeError(`${rawPath} holds ${halfWords.length} halfs, short of ${width * height * 4} for ${width}x${height}`);
}
/** Scene-linear radiance, three channels per pixel — the grade's input. */
const radiance = new Float32Array(width * height * 3);
for (let pixel = 0; pixel < width * height; pixel += 1) {
  for (let channel = 0; channel < 3; channel += 1) {
    const value = decodeF16(halfWords[pixel * 4 + channel]);
    radiance[pixel * 3 + channel] = Number.isFinite(value) ? Math.max(0, value) : 0;
  }
}

// ---------------------------------------------------------------------------
// The grade, applied exactly as the shipped one is
// ---------------------------------------------------------------------------

function applyToneCurve(value: number, curve: SceneToneCurve): number {
  if (curve !== "aces") return value / (value + 1);
  return Math.min(1, Math.max(0, (value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14)));
}

/**
 * Grade the scene-linear frame to a linear-light image on the scoring grid.
 *
 * The path is scene-linear -> balance -> exposure -> tone curve -> gamma 2.2 ->
 * *back to linear*. The last two steps look like a round trip that could be
 * skipped, and skipping them would be wrong: the encode is 8-bit in the real
 * lane, and ΔE₀₀ is being asked about the image a person sees. Quantising here
 * keeps the solve honest about the frame it is actually grading.
 */
function gradeFrame(exposure: number, whiteBalance: readonly [number, number, number]): LinearImage {
  const resolved = resolveDisplayGrade({ exposure, toneCurve, whiteBalance });
  const rgb = new Float32Array(width * height * 3);
  for (let index = 0; index < rgb.length; index += 1) {
    const channel = index % 3;
    const balanced = radiance[index] * resolved.whiteBalance[channel] * resolved.exposure;
    const display = applyToneCurve(balanced, resolved.toneCurve) ** (1 / 2.2);
    rgb[index] = srgbToLinear(Math.round(255 * Math.min(1, Math.max(0, display))));
  }
  return { width, height, rgb };
}

const plate = decodeLinearPng(readFileSync(platePath));
const registeredPlate = registerToScoringGrid({ width, height, rgb: new Float32Array(width * height * 3) }, plate).plate;

const scoredRegions = HERO_FIDELITY_REGIONS.filter((region) => NEUTRAL_REGIONS.has(region.id));

function objective(exposure: number, whiteBalance: readonly [number, number, number]): number {
  const frame = gradeFrame(exposure, whiteBalance);
  let total = 0;
  let regions = 0;
  for (const region of scoredRegions) {
    const bounds = regionBounds(region, width, height);
    let regionTotal = 0;
    let samples = 0;
    // Every fourth pixel: the objective is a mean over tens of thousands of
    // samples and the search runs hundreds of candidates. A quarter of the
    // pixels moves the mean by well under the 0.25 ΔE the gate calls a floor.
    for (let y = 0; y < bounds.height; y += 2) {
      for (let x = 0; x < bounds.width; x += 2) {
        const at = ((y + bounds.y) * width + (x + bounds.x)) * 3;
        regionTotal += deltaE2000(
          labFromLinearRgb(frame.rgb[at], frame.rgb[at + 1], frame.rgb[at + 2]),
          labFromLinearRgb(registeredPlate.rgb[at], registeredPlate.rgb[at + 1], registeredPlate.rgb[at + 2]),
        );
        samples += 1;
      }
    }
    total += regionTotal / Math.max(1, samples);
    regions += 1;
  }
  return total / Math.max(1, regions);
}

// ---------------------------------------------------------------------------
// Coordinate descent over exposure and the two free balance channels
// ---------------------------------------------------------------------------

let exposure = 1;
let balance: [number, number, number] = [1, 1, 1];
let best = objective(exposure, balance);
process.stdout.write(`[hero-grade] start deltaE00 ${best.toFixed(3)} at exposure 1, balance neutral\n`);

const trace: Array<Record<string, unknown>> = [{ pass: -1, exposure, balance: [...balance], deltaE00: best }];

for (let pass = 0; pass < 6; pass += 1) {
  const shrink = 0.6 ** pass;
  // Exposure in stops, so a probe is perceptually even across the range.
  for (const stops of [-1.2, -0.8, -0.5, -0.3, -0.15, 0.15, 0.3, 0.5, 0.8, 1.2]) {
    const candidate = exposure * 2 ** (stops * shrink);
    const score = objective(candidate, balance);
    if (score < best) {
      best = score;
      exposure = candidate;
    }
  }
  // Red and blue only: green is fixed by the luminance normalisation, so these
  // two are the balance's genuine degrees of freedom.
  for (const channel of [0, 2] as const) {
    for (const delta of [-0.16, -0.08, -0.04, -0.02, 0.02, 0.04, 0.08, 0.16]) {
      const candidate: [number, number, number] = [...balance];
      candidate[channel] = Math.max(0.25, Math.min(4, candidate[channel] + delta * shrink));
      const score = objective(exposure, candidate);
      if (score < best) {
        best = score;
        balance = candidate;
      }
    }
  }
  process.stdout.write(
    `[hero-grade] pass ${pass}  deltaE00 ${best.toFixed(3)}  exposure ${exposure.toFixed(4)}` +
      `  balance ${balance.map((value) => value.toFixed(4)).join(", ")}\n`,
  );
  trace.push({ pass, exposure, balance: [...balance], deltaE00: best });
}

const resolved = resolveDisplayGrade({ exposure, toneCurve, whiteBalance: balance });

// Report the per-region before/after so the gain is attributable rather than a
// single number that could have come from anywhere.
const perRegion = HERO_FIDELITY_REGIONS.map((region) => {
  const bounds = regionBounds(region, width, height);
  const measure = (frame: LinearImage) => {
    let total = 0;
    let samples = 0;
    for (let y = 0; y < bounds.height; y += 2) {
      for (let x = 0; x < bounds.width; x += 2) {
        const at = ((y + bounds.y) * width + (x + bounds.x)) * 3;
        total += deltaE2000(
          labFromLinearRgb(frame.rgb[at], frame.rgb[at + 1], frame.rgb[at + 2]),
          labFromLinearRgb(registeredPlate.rgb[at], registeredPlate.rgb[at + 1], registeredPlate.rgb[at + 2]),
        );
        samples += 1;
      }
    }
    return total / Math.max(1, samples);
  };
  return {
    id: region.id,
    scored: NEUTRAL_REGIONS.has(region.id),
    before: measure(gradeFrame(1, [1, 1, 1])),
    after: measure(gradeFrame(exposure, balance)),
  };
});

process.stdout.write("\n[hero-grade] per-region deltaE00\nregion       scored    before     after\n");
for (const entry of perRegion) {
  process.stdout.write(
    `${entry.id.padEnd(12)}${(entry.scored ? "yes" : "no").padEnd(9)}${entry.before.toFixed(2).padStart(8)}${entry.after.toFixed(2).padStart(10)}\n`,
  );
}
process.stdout.write(
  `\n[hero-grade] grade: { toneCurve: "${toneCurve}", exposure: ${exposure.toFixed(4)}, ` +
    `whiteBalance: [${balance.map((value) => value.toFixed(4)).join(", ")}] }\n` +
    `[hero-grade] resolved (luminance-normalised) balance: [${resolved.whiteBalance.map((value) => value.toFixed(4)).join(", ")}]\n`,
);

mkdirSync(path.dirname(outPath), { recursive: true });
writeFileSync(
  outPath,
  `${JSON.stringify({ raw: rawPath, plate: platePath, toneCurve, exposure, whiteBalance: balance, resolved, deltaE00: best, perRegion, trace }, null, 2)}\n`,
);
process.stdout.write(`[hero-grade] solution written to ${outPath}\n`);
