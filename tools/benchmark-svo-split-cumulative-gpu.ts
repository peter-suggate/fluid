#!/usr/bin/env node
/**
 * Controlled hose-scene comparison for the cumulative traversal candidate:
 * canonical-parametric, authored 8^3 bricks, occupancy off, exact primary
 * traversal, reduced cone scale 0.5. The sole experimental variable is the
 * inline versus split visibility/lighting path.
 *
 * Each arm runs in a fresh Dawn process. This wrapper is intentionally not a
 * production default selector; it records the evidence needed to decide one.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

type ShadingPath = "inline" | "split";

interface ArmReport {
  readonly traversalMode: string;
  readonly brickOccupancyMode: string;
  readonly shadingPath: ShadingPath;
  readonly rayCoherenceMode: string;
  readonly screenSpaceTermination: { thresholdPixels: number; mode: string };
  readonly resolution: { width: number; height: number };
  readonly timing: { median_ms: number; p95_ms: number; samples_ms: number[] };
  readonly coneLighting: { scale: number };
  readonly scene: { presetId: string; brickSize: number; authoredBrickSize: number };
  readonly fingerprint: {
    imageHashFnv1a32: string;
    configuredImageHashFnv1a32: string;
    packedRgba16FloatPath?: string;
    configuredPackedRgba16FloatPath?: string;
  };
}

const width = Number(process.env.FLUID_SVO_SPLIT_CUMULATIVE_WIDTH ?? 660);
const height = Number(process.env.FLUID_SVO_SPLIT_CUMULATIVE_HEIGHT ?? 662);
const warmups = Number(process.env.FLUID_SVO_SPLIT_CUMULATIVE_WARMUPS ?? 4);
const cycles = Number(process.env.FLUID_SVO_SPLIT_CUMULATIVE_CYCLES ?? 12);
const repeats = Number(process.env.FLUID_SVO_SPLIT_CUMULATIVE_REPEATS ?? 2);
const outputDirectory = path.resolve(process.env.FLUID_SVO_SPLIT_CUMULATIVE_OUT_DIR
  ?? "/tmp/svo-split-cumulative");
const resultPath = path.join(outputDirectory, "comparison.json");

assert.ok(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0);
assert.ok(Number.isSafeInteger(warmups) && warmups >= 1);
assert.ok(Number.isSafeInteger(cycles) && cycles >= 3);
assert.ok(Number.isSafeInteger(repeats) && repeats >= 2);
mkdirSync(outputDirectory, { recursive: true });

function runArm(shadingPath: ShadingPath, run: number): ArmReport {
  const reportPath = path.join(outputDirectory, `${shadingPath}-run-${run}.json`);
  const rawPath = path.join(outputDirectory, `${shadingPath}-run-${run}.rgba16float.bin`);
  const configuredRawPath = path.join(outputDirectory, `${shadingPath}-run-${run}.configured.rgba16float.bin`);
  const child = spawnSync(process.execPath, ["--import", "tsx", "tools/benchmark-svo-dry-frame-gpu.ts"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      FLUID_SVO_DRY_FRAME_SCENE: "hose-tank",
      FLUID_SVO_DRY_FRAME_WIDTH: String(width),
      FLUID_SVO_DRY_FRAME_HEIGHT: String(height),
      FLUID_SVO_DRY_FRAME_WARMUPS: String(warmups),
      FLUID_SVO_DRY_FRAME_CYCLES: String(cycles),
      FLUID_SVO_DRY_FRAME_CONE_SCALE: "0.5",
      FLUID_SVO_DRY_FRAME_SHADOWS: "1",
      FLUID_SVO_DRY_FRAME_AO: "1",
      FLUID_SVO_DRY_FRAME_TRAVERSAL: "canonical-parametric",
      FLUID_SVO_DRY_FRAME_BRICK_OCCUPANCY: "off",
      FLUID_SVO_DRY_FRAME_BRICK_SIZE: "8",
      FLUID_SVO_DRY_FRAME_SHADING: shadingPath,
      FLUID_SVO_DRY_FRAME_COHERENCE: "off",
      FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS: "0",
      FLUID_SVO_DRY_FRAME_TIMING: "wall",
      FLUID_SVO_DRY_FRAME_OUT: reportPath,
      FLUID_SVO_DRY_FRAME_RAW_OUT: rawPath,
      FLUID_SVO_DRY_FRAME_CONFIGURED_RAW_OUT: configuredRawPath,
    },
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (child.status !== 0) {
    process.stderr.write(child.stdout);
    process.stderr.write(child.stderr);
    throw new Error(`${shadingPath} benchmark exited ${child.status ?? "without a status"}`);
  }
  const report = JSON.parse(readFileSync(reportPath, "utf8")) as ArmReport;
  assert.equal(report.traversalMode, "canonical-parametric");
  assert.equal(report.brickOccupancyMode, "off");
  assert.equal(report.shadingPath, shadingPath);
  assert.equal(report.rayCoherenceMode, "off");
  assert.equal(report.screenSpaceTermination.thresholdPixels, 0);
  assert.equal(report.screenSpaceTermination.mode, "exact");
  assert.equal(report.coneLighting.scale, 0.5);
  assert.deepEqual(report.resolution, { width, height });
  assert.equal(report.scene.presetId, "hose-tank");
  assert.equal(report.scene.brickSize, 8);
  assert.equal(report.scene.authoredBrickSize, 8);
  assert.equal(report.fingerprint.packedRgba16FloatPath, rawPath);
  assert.equal(report.fingerprint.configuredPackedRgba16FloatPath, configuredRawPath);
  return report;
}

function halfToFloat(value: number): number {
  const sign = (value & 0x8000) !== 0 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function toneByte(linear: number): number {
  return Math.max(0, Math.min(255, Math.round(255 * Math.min(1, Math.max(0, linear)) ** (1 / 2.2))));
}

function percentile(sorted: readonly number[], fraction: number): number {
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)] ?? 0;
}

function timing(samples: readonly number[]) {
  const sorted = [...samples].sort((left, right) => left - right);
  return {
    samples_ms: samples,
    median_ms: sorted[Math.floor(sorted.length / 2)],
    p95_ms: percentile(sorted, 0.95),
  };
}

function fullImageError(referencePath: string, candidatePath: string) {
  const referenceBytes = readFileSync(referencePath);
  const candidateBytes = readFileSync(candidatePath);
  assert.equal(referenceBytes.byteLength, width * height * 4 * 2);
  assert.equal(candidateBytes.byteLength, referenceBytes.byteLength);
  const reference = new Uint16Array(referenceBytes.buffer, referenceBytes.byteOffset, referenceBytes.byteLength / 2);
  const candidate = new Uint16Array(candidateBytes.buffer, candidateBytes.byteOffset, candidateBytes.byteLength / 2);
  const luminanceErrors: number[] = [];
  let differingHalfWords = 0;
  let differingPixels = 0;
  let differingRgbPixels = 0;
  let differingDepthPixels = 0;
  let differingTonePixels = 0;
  let maximumToneByteDelta = 0;
  let absoluteRgbSum = 0;
  let toneDifferenceSum = 0;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  const weights = [0.2126, 0.7152, 0.0722] as const;

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    let pixelDiffers = false;
    let rgbDiffers = false;
    let toneDiffers = false;
    let referenceY = 0;
    let candidateY = 0;
    for (let channel = 0; channel < 4; channel += 1) {
      const index = pixel * 4 + channel;
      if (reference[index] !== candidate[index]) {
        differingHalfWords += 1;
        pixelDiffers = true;
        if (channel < 3) rgbDiffers = true;
        else differingDepthPixels += 1;
      }
      if (channel === 3) continue;
      const referenceValue = halfToFloat(reference[index]);
      const candidateValue = halfToFloat(candidate[index]);
      referenceY += weights[channel] * referenceValue;
      candidateY += weights[channel] * candidateValue;
      absoluteRgbSum += Math.abs(candidateValue - referenceValue);
      const toneDelta = Math.abs(toneByte(candidateValue) - toneByte(referenceValue));
      toneDifferenceSum += toneDelta;
      maximumToneByteDelta = Math.max(maximumToneByteDelta, toneDelta);
      toneDiffers ||= toneDelta !== 0;
    }
    if (pixelDiffers) {
      differingPixels += 1;
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
    if (rgbDiffers) differingRgbPixels += 1;
    if (toneDiffers) differingTonePixels += 1;
    if (referenceY > 1e-4) luminanceErrors.push(Math.abs(candidateY - referenceY) / Math.max(0.01, referenceY));
  }
  luminanceErrors.sort((left, right) => left - right);
  const pixelCount = width * height;
  return {
    bitExact: differingHalfWords === 0,
    differingHalfWords,
    differingHalfWordPercent: 100 * differingHalfWords / reference.length,
    differingPixels,
    differingPixelPercent: 100 * differingPixels / pixelCount,
    differingRgbPixels,
    differingDepthPixels,
    differingTonePixels,
    differingTonePixelPercent: 100 * differingTonePixels / pixelCount,
    maximumToneByteDelta,
    meanToneByteDeltaPerChannel: toneDifferenceSum / (pixelCount * 3),
    meanAbsoluteRgb: absoluteRgbSum / (pixelCount * 3),
    changedPixelBounds: differingPixels === 0 ? undefined : { minX, minY, maxX, maxY },
    relativeLuminance: {
      litPixels: luminanceErrors.length,
      mean: luminanceErrors.reduce((sum, value) => sum + value, 0) / Math.max(1, luminanceErrors.length),
      p95: percentile(luminanceErrors, 0.95),
      max: luminanceErrors[luminanceErrors.length - 1] ?? 0,
    },
  };
}

const runs: Record<ShadingPath, ArmReport[]> = { inline: [], split: [] };
const executionOrder: ShadingPath[] = [];
for (let repeat = 0; repeat < repeats; repeat += 1) {
  // A/B then B/A cancels first-process and thermal-order bias without placing
  // both giant shader variants in one Dawn device/process.
  const order: readonly ShadingPath[] = repeat % 2 === 0 ? ["inline", "split"] : ["split", "inline"];
  for (const shadingPath of order) {
    executionOrder.push(shadingPath);
    runs[shadingPath].push(runArm(shadingPath, runs[shadingPath].length + 1));
  }
}
const inlineTiming = timing(runs.inline.flatMap(run => run.timing.samples_ms));
const splitTiming = timing(runs.split.flatMap(run => run.timing.samples_ms));
const inlineRaw = runs.inline[0].fingerprint.configuredPackedRgba16FloatPath!;
const splitRaw = runs.split[0].fingerprint.configuredPackedRgba16FloatPath!;
const inlineHashes = runs.inline.map(run => run.fingerprint.configuredImageHashFnv1a32);
const splitHashes = runs.split.map(run => run.fingerprint.configuredImageHashFnv1a32);

const result = {
  phase: "svo-split-cumulative-comparison",
  controls: {
    scene: "hose-tank",
    resolution: { width, height },
    warmups,
    cycles,
    repeats,
    executionOrder,
    traversal: "canonical-parametric",
    brickSize: 8,
    occupancy: "off",
    screenSpaceTerminationPixels: 0,
    coneScale: 0.5,
    shadows: true,
    ambientOcclusion: true,
    coherence: "off",
    simulation: "absent/frozen",
    soleVariable: "shadingPath",
    timingCaveat: "fresh-process alternating A/B-B/A; compare relative medians because absolute submit-to-fence times remain host-contention sensitive",
  },
  arms: {
    inline: {
      aggregateTiming: inlineTiming,
      deterministicWithinArm: new Set(inlineHashes).size === 1,
      imageHashes: inlineHashes,
      runs: runs.inline,
    },
    split: {
      aggregateTiming: splitTiming,
      deterministicWithinArm: new Set(splitHashes).size === 1,
      imageHashes: splitHashes,
      runs: runs.split,
    },
  },
  comparison: {
    speedup: inlineTiming.median_ms / splitTiming.median_ms,
    improvementPercent: 100 * (1 - splitTiming.median_ms / inlineTiming.median_ms),
    medianDelta_ms: splitTiming.median_ms - inlineTiming.median_ms,
    p95Delta_ms: splitTiming.p95_ms - inlineTiming.p95_ms,
    image: fullImageError(inlineRaw, splitRaw),
  },
};

writeFileSync(resultPath, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result, null, 2));
