#!/usr/bin/env node
/**
 * Render-only 4^3 versus 8^3 SVO brick experiment on one fixed production
 * scene. Each arm is a fresh Dawn process because the static world owns its
 * topology for life. The authored scene is never mutated: the override enters
 * only WebGPUStaticSvoScene construction.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

interface ArmReport {
  timing: { median_ms: number; p95_ms: number; samples_ms: number[] };
  scene: {
    presetId: string;
    brickSize: number;
    authoredBrickSize: number;
    maximumDepth: number;
    structuralCapacities: { nodes: number; leaves: number; voxels: number };
    structuralBytes: { topology: number; geometry: number; velocity: number; materialOwners: number; payload: number };
    allocatedBytes: number;
  };
  fingerprint: { imageHashFnv1a32: string; packedRgba16FloatPath?: string };
}

const width = Number(process.env.FLUID_SVO_BRICK_SIZE_WIDTH ?? 660);
const height = Number(process.env.FLUID_SVO_BRICK_SIZE_HEIGHT ?? 662);
const scene = process.env.FLUID_SVO_BRICK_SIZE_SCENE ?? "hose-tank";
const warmups = Number(process.env.FLUID_SVO_BRICK_SIZE_WARMUPS ?? 5);
const cycles = Number(process.env.FLUID_SVO_BRICK_SIZE_CYCLES ?? 16);
const outputDirectory = path.resolve(process.env.FLUID_SVO_BRICK_SIZE_OUT_DIR ?? "/tmp/svo-brick-size");
const resultPath = path.join(outputDirectory, "comparison.json");
assert.ok(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0);
assert.ok(Number.isSafeInteger(warmups) && warmups >= 1 && Number.isSafeInteger(cycles) && cycles >= 3);
mkdirSync(outputDirectory, { recursive: true });

function runArm(brickSize: 4 | 8): ArmReport {
  const reportPath = path.join(outputDirectory, `brick-${brickSize}.json`);
  const rawPath = path.join(outputDirectory, `brick-${brickSize}.rgba16float.bin`);
  const child = spawnSync(process.execPath, ["--import", "tsx", "tools/benchmark-svo-dry-frame-gpu.ts"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      FLUID_SVO_DRY_FRAME_SCENE: scene,
      FLUID_SVO_DRY_FRAME_WIDTH: String(width),
      FLUID_SVO_DRY_FRAME_HEIGHT: String(height),
      FLUID_SVO_DRY_FRAME_WARMUPS: String(warmups),
      FLUID_SVO_DRY_FRAME_CYCLES: String(cycles),
      FLUID_SVO_DRY_FRAME_CONE_SCALE: process.env.FLUID_SVO_BRICK_SIZE_CONE_SCALE ?? "1",
      FLUID_SVO_DRY_FRAME_SHADOWS: process.env.FLUID_SVO_BRICK_SIZE_SHADOWS ?? "1",
      FLUID_SVO_DRY_FRAME_AO: process.env.FLUID_SVO_BRICK_SIZE_AO ?? "1",
      FLUID_SVO_DRY_FRAME_TRAVERSAL: process.env.FLUID_SVO_BRICK_SIZE_TRAVERSAL ?? "hybrid",
      FLUID_SVO_DRY_FRAME_BRICK_OCCUPANCY: "off",
      FLUID_SVO_DRY_FRAME_BRICK_SIZE: String(brickSize),
      // Dawn on current macOS can advertise timestamp-query while returning
      // no resolved trace. Serialized submit-to-fence remains a real GPU
      // completion measurement and is the dry benchmark's supported fallback.
      FLUID_SVO_DRY_FRAME_TIMING: process.env.FLUID_SVO_BRICK_SIZE_TIMING ?? "wall",
      FLUID_SVO_DRY_FRAME_OUT: reportPath,
      FLUID_SVO_DRY_FRAME_RAW_OUT: rawPath,
    },
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (child.status !== 0) {
    process.stderr.write(child.stdout);
    process.stderr.write(child.stderr);
    throw new Error(`brick-${brickSize} benchmark exited ${child.status ?? "without a status"}`);
  }
  return JSON.parse(readFileSync(reportPath, "utf8")) as ArmReport;
}

function halfToFloat(value: number): number {
  const sign = (value & 0x8000) !== 0 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function imageError(referencePath: string, candidatePath: string) {
  const referenceBytes = readFileSync(referencePath);
  const candidateBytes = readFileSync(candidatePath);
  assert.equal(candidateBytes.byteLength, referenceBytes.byteLength, "brick-size frames must have identical dimensions and formats");
  const reference = new Uint16Array(referenceBytes.buffer, referenceBytes.byteOffset, referenceBytes.byteLength / 2);
  const candidate = new Uint16Array(candidateBytes.buffer, candidateBytes.byteOffset, candidateBytes.byteLength / 2);
  const relativeLuminanceErrors: number[] = [];
  let differingHalfWords = 0;
  let absoluteRgbSum = 0;
  let comparedRgbChannels = 0;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    let referenceY = 0;
    let candidateY = 0;
    for (let channel = 0; channel < 4; channel += 1) {
      const index = pixel * 4 + channel;
      if (reference[index] !== candidate[index]) differingHalfWords += 1;
      if (channel === 3) continue;
      const weight = [0.2126, 0.7152, 0.0722][channel];
      const referenceValue = halfToFloat(reference[index]);
      const candidateValue = halfToFloat(candidate[index]);
      referenceY += weight * referenceValue;
      candidateY += weight * candidateValue;
      absoluteRgbSum += Math.abs(candidateValue - referenceValue);
      comparedRgbChannels += 1;
    }
    if (referenceY > 1e-4) {
      relativeLuminanceErrors.push(Math.abs(candidateY - referenceY) / Math.max(0.01, referenceY));
    }
  }
  relativeLuminanceErrors.sort((left, right) => left - right);
  return {
    bitExact: differingHalfWords === 0,
    differingHalfWords,
    differingHalfWordPercent: 100 * differingHalfWords / reference.length,
    meanAbsoluteRgb: absoluteRgbSum / Math.max(1, comparedRgbChannels),
    relativeLuminance: {
      litPixels: relativeLuminanceErrors.length,
      mean: relativeLuminanceErrors.reduce((sum, value) => sum + value, 0) / Math.max(1, relativeLuminanceErrors.length),
      p95: relativeLuminanceErrors[Math.min(relativeLuminanceErrors.length - 1, Math.ceil(relativeLuminanceErrors.length * 0.95) - 1)] ?? 0,
      max: relativeLuminanceErrors[relativeLuminanceErrors.length - 1] ?? 0,
    },
  };
}

// Run the authored 8^3 representation first, then its render-only 4^3 arm.
const brick8 = runArm(8);
const brick4 = runArm(4);
assert.equal(brick8.scene.authoredBrickSize, 8);
assert.equal(brick4.scene.authoredBrickSize, 8, "the 4^3 arm must not alter the hose scene's simulation contract");
assert.equal(brick8.scene.brickSize, 8);
assert.equal(brick4.scene.brickSize, 4);
assert.ok(brick8.fingerprint.packedRgba16FloatPath && brick4.fingerprint.packedRgba16FloatPath);

const result = {
  phase: "svo-render-brick-size-comparison",
  scene,
  resolution: { width, height },
  controls: {
    warmups,
    cycles,
    traversal: process.env.FLUID_SVO_BRICK_SIZE_TRAVERSAL ?? "hybrid",
    coneScale: Number(process.env.FLUID_SVO_BRICK_SIZE_CONE_SCALE ?? 1),
    shadows: process.env.FLUID_SVO_BRICK_SIZE_SHADOWS !== "0",
    ambientOcclusion: process.env.FLUID_SVO_BRICK_SIZE_AO !== "0",
    simulation: "absent/frozen",
  },
  arms: { brick8, brick4 },
  comparison: {
    speedup4Vs8: brick8.timing.median_ms / brick4.timing.median_ms,
    medianDelta_ms: brick4.timing.median_ms - brick8.timing.median_ms,
    nodeRatio4Vs8: brick4.scene.structuralCapacities.nodes / brick8.scene.structuralCapacities.nodes,
    leafRatio4Vs8: brick4.scene.structuralCapacities.leaves / brick8.scene.structuralCapacities.leaves,
    payloadRatio4Vs8: brick4.scene.structuralBytes.payload / brick8.scene.structuralBytes.payload,
    image: imageError(brick8.fingerprint.packedRgba16FloatPath, brick4.fingerprint.packedRgba16FloatPath),
  },
};
writeFileSync(resultPath, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result, null, 2));
