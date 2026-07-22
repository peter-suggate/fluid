import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { planOctreeCoarsePhi } from "../lib/webgpu-octree-coarse-levelset";
import { FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES } from "../lib/webgpu-octree-fine-levelset-redistance";
import { planFineLevelSetGPUSummaries } from "../lib/webgpu-octree-fine-levelset-summary";
import { fineLevelSetLeafSeedAllocatedBytes, FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES } from
  "../lib/webgpu-octree-fine-levelset-topology";
import { planFineLevelSetGPUTransport } from "../lib/webgpu-octree-fine-levelset-transport";
import { planFineLevelSetGPUVolume } from "../lib/webgpu-octree-fine-levelset-volume";
import { planFineToCoarseLevelSet } from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { FINE_TO_COARSE_LEVELSET_ERROR, unpackFineToCoarseGPUControl } from
  "../lib/webgpu-octree-fine-to-coarse-levelset";
import { OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS, WebGPUOctreePowerCoarseLevelSet,
  planOctreePowerCoarseLevelSet } from "../lib/webgpu-octree-power-coarse-levelset";
import { planOctreePowerDescriptors } from "../lib/webgpu-octree-power-descriptor";
import { planOctreePowerFaceTransfer } from "../lib/webgpu-octree-power-face-transfer";
import { planOctreePowerFaces } from "../lib/webgpu-octree-power-faces";
import { planOctreePowerGPUOperator } from "../lib/webgpu-octree-power-operator";
import { planOctreePowerVelocityPrepass } from "../lib/webgpu-octree-power-velocity-prepass";
import { planOctreePowerVelocity } from "../lib/webgpu-octree-power-velocity";
import { estimateGlobalFineNarrowBandBrickCapacity,
  planGlobalFineNarrowBandBrickCapacity, resolveGlobalFineBrickCapacity,
  sumOctreePowerAllocationBreakdown } from "../lib/webgpu-octree";

function fineArchitectureBytes(factor: 4 | 8, dimensions: readonly [number, number, number],
  brickCapacity: number, rowCapacity: number): number {
  const bricks = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: dimensions,
    finestCellWidth: 1, fineFactor: factor, brickResolution: 4, maximumResidentBricks: brickCapacity });
  const samples = brickCapacity * bricks.samplesPerBrick;
  const velocityChunk = Math.min(4096, samples);
  const coarseDirectoryCapacity = planOctreePowerCoarseLevelSet(rowCapacity).sampleHashCapacity;
  const volumeA = planFineLevelSetGPUVolume(rowCapacity, samples, true, coarseDirectoryCapacity);
  const volumeB = planFineLevelSetGPUVolume(rowCapacity, samples, false, coarseDirectoryCapacity);
  return bricks.allocatedBytes + 2 * 80 + fineLevelSetLeafSeedAllocatedBytes(brickCapacity, bricks.hashCapacity)
    + 2 * FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES + 2 * FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES
    + planOctreePowerVelocityPrepass(velocityChunk).allocatedBytes
    + 2 * planFineLevelSetGPUTransport(samples, velocityChunk).allocatedBytes
    + volumeA.allocatedBytes + volumeB.allocatedBytes
    + planFineToCoarseLevelSet(rowCapacity, samples).allocatedBytes;
}

test("factor-4/factor-8 global fine memory is resident-capacity-scaled, not domain-volume-scaled", () => {
  for (const factor of [4, 8] as const) {
    const smallDomain = fineArchitectureBytes(factor, [16, 16, 16], 32, 128);
    const largeDomain = fineArchitectureBytes(factor, [64, 64, 64], 32, 128);
    assert.equal(largeDomain, smallDomain,
      `factor ${factor} allocation changed with logical domain at fixed sparse capacities`);
    const doubledResidents = fineArchitectureBytes(factor, [64, 64, 64], 64, 128);
    assert.ok(doubledResidents > largeDomain);

    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [64, 64, 64],
      finestCellWidth: 1, fineFactor: factor, brickResolution: 4, maximumResidentBricks: 32 });
    assert.equal(plan.payloadCapacityBytes, 32 * plan.samplesPerBrick * 4 * 4);
    assert.ok(plan.logicalBrickCount > plan.maximumResidentBricks);
    assert.ok(plan.allocatedBytes < plan.logicalBrickCount * plan.payloadBytesPerBrick,
      "sparse fine allocation must not materialize a persistent full-domain phi lattice");
  }
});

test("fine-to-coarse restriction allocation is O(rows), independent of fine sample capacity", () => {
  const small = planFineToCoarseLevelSet(41_728, 64);
  const factor8 = planFineToCoarseLevelSet(41_728, 213_648 * 64);
  assert.equal(factor8.allocatedBytes, small.allocatedBytes);
  assert.equal(factor8.aggregateScratchBytes, 41_728 * 48,
    "each row owns four scalar atomics plus eight deterministic center-corner samples");
  assert.ok(factor8.allocatedBytes < 3_000_000,
    "factor-8 restriction must not allocate per-resident-sample owners or contributions");
});

test("factor-8 transport keeps trajectory storage chunk-bounded", () => {
  const samples = 262_144 * 64, chunk = 4096;
  const plan = planFineLevelSetGPUTransport(samples, chunk);
  assert.equal(plan.positionCapacity, chunk);
  assert.equal(plan.positionBytes, chunk * 16);
  assert.equal(plan.outcomeBytes, chunk * 8);
  assert.equal(plan.chunkCount, 4096);
  assert.ok(plan.allocatedBytes < 1_500_000,
    "transport must not allocate a max-resident-sample trajectory array per A/B generation");
});

test("global fine capacity uses 2D dispatch and never silently shrinks the physical band estimate", () => {
  assert.equal(resolveGlobalFineBrickCapacity(213_648, undefined, 65_535), 213_648);
  assert.equal(resolveGlobalFineBrickCapacity(213_648, 48_000, 65_535), 48_000);
  assert.throws(() => resolveGlobalFineBrickCapacity(400_000, undefined, 65_535, 64, 256 * 1024 * 1024, 64, 8),
    /physical narrow-band estimate is not reduced implicitly/);
  assert.equal(resolveGlobalFineBrickCapacity(337_500, undefined, 65_535, 64,
    256 * 1024 * 1024, 64, 8, 506_697), 337_500,
  "exact merged-summary capacity must replace the legacy bricks-times-levels device gate");
  assert.throws(() => resolveGlobalFineBrickCapacity(337_500, undefined, 65_535, 64,
    32 * 1024 * 1024, 1, 8, 506_697), /sparse summary requires/);
  assert.equal(resolveGlobalFineBrickCapacity(400_000, 262_144, 65_535, 64, 256 * 1024 * 1024, 64, 8), 262_144);
  assert.throws(() => resolveGlobalFineBrickCapacity(400_000, 262_145, 65_535, 64, 256 * 1024 * 1024, 64, 8),
    /exceeds the sparse binding\/dispatch limit/);
  assert.throws(() => resolveGlobalFineBrickCapacity(213_648, 0, 65_535), /positive integer/);
  assert.equal(estimateGlobalFineNarrowBandBrickCapacity([60, 45, 40], 7), 50_625);
  assert.equal(estimateGlobalFineNarrowBandBrickCapacity([120, 90, 80], 12), 337_500);
});

test("global fine capacity is an explicit surface-area times band plan", () => {
  const balanced = planGlobalFineNarrowBandBrickCapacity([60, 45, 40], 7);
  assert.deepEqual(balanced, {
    logicalBrickCount: 108_000,
    maximumInterfaceAreaBricks: 2_700,
    bandLayers: 15,
    bandBrickCount: 40_500,
    surfaceGrowthSafety: 1.25,
    surfaceGrowthHeadroomBricks: 10_125,
    maximumResidentBricks: 50_625,
  });
  const doubled = planGlobalFineNarrowBandBrickCapacity([120, 90, 80], 7);
  assert.equal(doubled.maximumResidentBricks, 4 * balanced.maximumResidentBricks,
    "fixed-width sparse bands must grow with interface area, not logical volume");
  assert.equal(doubled.logicalBrickCount, 8 * balanced.logicalBrickCount);
  const clipped = planGlobalFineNarrowBandBrickCapacity([4, 3, 2], 7);
  assert.equal(clipped.maximumResidentBricks, clipped.logicalBrickCount,
    "a band wider than a tiny domain may conservatively cover that whole domain");
});

test("fine summary hash reserves merged sparse ancestors instead of capacity at every level", () => {
  const fine = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
    finestCellDimensions: [60, 45, 40], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 50_625 });
  const summary = planFineLevelSetGPUSummaries(fine, 45_312);
  assert.equal(summary.fineEntryCapacity, 66_510);
  assert.equal(summary.coarseEntryCapacity, 45_312);
  assert.equal(summary.entryCapacity, 111_822);
  assert.equal(summary.hashCapacity, 262_144);
  assert.equal(summary.allocatedBytes, 8_389_344);
  const legacyHashCapacity = 1_048_576;
  const legacyBytes = 64 + legacyHashCapacity * 32 + summary.parameterBytes;
  assert.equal(legacyBytes - summary.allocatedBytes, 25_165_824,
    "balanced factor-4 must save exactly 24 MiB of over-reserved hierarchy hash");
});

test("parallel total-volume scratch is bounded by compact directory and resident fine samples", () => {
  const a = planFineLevelSetGPUVolume(257, 4097, true);
  assert.equal(a.coarsePartialCount, 5); assert.equal(a.finePartialCount, 65);
  assert.equal(a.coarsePartialBytes, 80); assert.equal(a.finePartialBytes, 2080);
  assert.equal(a.reductionScratchBytes, 2080); assert.equal(a.allocatedBytes, 64 + 2080 + 64);
  const b = planFineLevelSetGPUVolume(257, 4097, false);
  assert.equal(b.allocatedBytes, a.allocatedBytes - 64, "B must share, not double-count, the A/B reference control");
  const snapshot = planFineLevelSetGPUVolume(257, 64, true, 1024);
  assert.equal(snapshot.coarsePartialCount, 16);
  assert.equal(snapshot.coarsePartialBytes, 256);
  assert.equal(snapshot.reductionScratchBytes, 256);
  assert.equal(snapshot.allocatedBytes, 64 + 256 + 64,
    "the accepted coarse-directory snapshot, not only live row capacity, sizes coarse reduction scratch");
});

test("coarse and power allocations scale with compact row/face capacities", () => {
  const compact = (rows: number, faces: number) => planOctreePowerDescriptors(rows).allocatedBytes
    + planOctreeCoarsePhi(rows).allocatedBytes + planOctreePowerCoarseLevelSet(rows).allocatedBytes
    + planOctreePowerVelocity(rows).allocatedBytes + planOctreePowerFaces(rows, faces).allocatedBytes
    + planOctreePowerGPUOperator(rows, faces, rows * 18, 18).allocatedBytes
    + planOctreePowerFaceTransfer(faces).allocatedBytes;
  const low = compact(128, 512), highRows = compact(256, 512), highFaces = compact(128, 1024);
  assert.ok(highRows > low); assert.ok(highFaces > low);
  // None of these planners accepts finest-domain voxel count: compact rows,
  // faces, incidences, and bounded sparse samples are their only scale inputs.
  assert.equal(compact(128, 512), low);
});

test("power allocation accounting charges each face/operator arena once", () => {
  assert.equal(sumOctreePowerAllocationBreakdown({ faces: 52_841_136, operator: 13_520_048,
    topology: 8_383_068 }), 74_744_252);
  assert.throws(() => sumOctreePowerAllocationBreakdown({ faces: -1 }), /non-negative safe bytes/);
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const accounting = source.match(/const powerAllocated = sumOctreePowerAllocationBreakdown\(\{[\s\S]*?\}\);/)?.[0];
  assert.ok(accounting);
  assert.equal(accounting.match(/faces: this\.powerFaces\.plan\.allocatedBytes/g)?.length, 1);
  assert.equal(accounting.match(/operator: this\.powerOperator\.plan\.allocatedBytes/g)?.length, 1);
});

test("coarse phi schedule bootstraps and fine-corrects at dt0 before recurring advection", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const calls = [...source.matchAll(/this\.powerCoarseLevelSetSchedule\.encode\([\s\S]*?\n\s*}\);/g)]
    .map((match) => match[0]);
  assert.equal(calls.length, 3);
  assert.match(calls[0], /dt:\s*0,/); assert.doesNotMatch(calls[0], /dt:\s*dt_s,/);
  assert.match(calls[1], /dt:\s*coarseBootstrappedThisStep\s*\?\s*0\s*:\s*dt_s,/);
  assert.match(calls[2], /dt:\s*dt_s,/);
});

test("coarse schedule parameters use one encoder-local arena across cold bootstrap and 64 substeps", () => {
  const plan = planOctreePowerCoarseLevelSet(32, 6);
  assert.equal(plan.allocatedBytes, 119_832,
    "allocation must include one aligned 65-invocation arena, valid-fine control, and empty correction buffers");
  assert.equal(plan.parameterArenaBytes, 65 * 7 * 256);
  const constructor = WebGPUOctreePowerCoarseLevelSet.toString().replace(/\s+/g, "");
  const encode = WebGPUOctreePowerCoarseLevelSet.prototype.encode.toString().replace(/\s+/g, "");
  const retire = WebGPUOctreePowerCoarseLevelSet.prototype.retireSubmittedEncoder.toString().replace(/\s+/g, "");
  assert.equal(OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS, 65);
  assert.doesNotMatch(constructor, /this\.params=Array\.from|this\.redistanceParams=Array\.from|activeEncoder/);
  assert.match(encode, /this\.encoderArenas\.get\(encoder\)/);
  assert.match(encode, /Powercoarsephiencoderparameterarena/);
  assert.match(encode, /invocationBase=encoderInvocation\*\(this\.plan\.redistancePasses\+1\)\*OCTREE_POWER_COARSE_LEVELSET_PARAM_STRIDE/);
  assert.match(encode,
    /encoderInvocation>=OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS\)\{thrownewRangeError\("Powercoarselevel-setencoderexceedsits65parameter-arenainvocations"\)/,
    "one command encoder must fail before its aligned arena can wrap");
  assert.doesNotMatch(encode, /submittedandretired|activeEncoder/,
    "encoding a second command buffer must not depend on submission or retirement of the first");
  assert.match(retire, /encoderArenas\.delete\(encoder\)/);
  assert.match(retire, /arena\.params\.destroy\(\)/);
});

test("fine-to-coarse capacity diagnostics decode fail-closed control words", () => {
  assert.deepEqual(FINE_TO_COARSE_LEVELSET_ERROR,
    { capacity: 1, unowned: 2, nonfinite: 4, unpublishedSource: 8 });
  assert.deepEqual(unpackFineToCoarseGPUControl([17, 5, 3, 2, 9, 0]), {
    contributionCount: 17, maximumContributionsPerRow: 5, flags: 3,
    unownedSamples: 2, rowCount: 9, valid: false,
  });
});
