import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { planOctreeCoarsePhi } from "../lib/webgpu-octree-coarse-levelset";
import { fineLevelSetRedistanceAllocatedBytes } from "../lib/webgpu-octree-fine-levelset-redistance";
import { FINE_LEVELSET_SUMMARY_ENTRY_WORDS,
  planFineLevelSetGPUSummaries } from "../lib/webgpu-octree-fine-levelset-summary";
import { fineLevelSetLeafSeedAllocatedBytes, FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES } from
  "../lib/webgpu-octree-fine-levelset-topology";
import { planFineLevelSetGPUTransport } from "../lib/webgpu-octree-fine-levelset-transport";
import { planFineLevelSetGPUVolume } from "../lib/webgpu-octree-fine-levelset-volume";
import { planFineToCoarseLevelSet } from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { FINE_TO_COARSE_LEVELSET_ERROR, unpackFineToCoarseGPUControl } from
  "../lib/webgpu-octree-fine-to-coarse-levelset";
import { OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS ,
  planOctreePowerCoarseLevelSet } from "../lib/webgpu-octree-power-coarse-levelset";
import { planOctreePowerDescriptors } from "../lib/webgpu-octree-power-descriptor";
import { planStructuredVelocityGPU } from "../lib/webgpu-octree-structured-velocity-gpu";
import { estimateGlobalFineNarrowBandBrickCapacity,
  planGlobalFineNarrowBandBrickCapacity, resolveGlobalFineBrickCapacity,
  sumOctreePowerAllocationBreakdown } from "../lib/webgpu-octree";

function fineArchitectureBytes(factor: 4 | 8, dimensions: readonly [number, number, number],
  brickCapacity: number, rowCapacity: number): number {
  const bricks = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: dimensions,
    finestCellWidth: 1, fineFactor: factor, brickResolution: 4, maximumResidentBricks: brickCapacity });
  const samples = brickCapacity * bricks.samplesPerBrick;
  const coarseDirectoryCapacity = planOctreePowerCoarseLevelSet(rowCapacity).rowCapacity;
  const volumeA = planFineLevelSetGPUVolume(rowCapacity, samples, true, coarseDirectoryCapacity);
  const volumeB = planFineLevelSetGPUVolume(rowCapacity, samples, false, coarseDirectoryCapacity);
  return bricks.allocatedBytes + 2 * 80 + fineLevelSetLeafSeedAllocatedBytes(brickCapacity)
    + 2 * fineLevelSetRedistanceAllocatedBytes(brickCapacity) + 2 * FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES
    + planStructuredVelocityGPU(rowCapacity).allocatedBytes
    + 2 * planFineLevelSetGPUTransport(samples, Math.min(4096, samples)).allocatedBytes
    + volumeA.allocatedBytes + volumeB.allocatedBytes
    + planFineToCoarseLevelSet(rowCapacity, samples).allocatedBytes;
}

test("factor-4/factor-8 global fine memory keeps payload resident-capacity-scaled", () => {
  for (const factor of [4, 8] as const) {
    const smallDomain = fineArchitectureBytes(factor, [16, 16, 16], 32, 128);
    const largeDomain = fineArchitectureBytes(factor, [64, 64, 64], 32, 128);
    assert.ok(largeDomain > smallDomain,
      `factor ${factor} direct logical directory must be represented explicitly`);
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

test("factor-8 transport keeps direct structured scratch page-bounded", () => {
  const samples = 262_144 * 64, pages = 262_144;
  const plan = planFineLevelSetGPUTransport(samples, 4096, pages);
  assert.equal(plan.chunkCount, 1);
  assert.equal(plan.velocityChunkCapacity, samples);
  assert.equal(plan.pageStatusBytes, pages * 12,
    "classification storage is reused as two packed u16 telemetry pairs plus exact displacement");
  assert.equal(plan.topologyDeltaBytes, (8 + 3 * pages) * 4);
  assert.ok(plan.allocatedBytes < 7_500_000,
    "direct structured transport must allocate only page reductions (including exact displacement) and the topology delta");
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
  assert.equal(estimateGlobalFineNarrowBandBrickCapacity([60, 45, 40], 7), 60_750);
  assert.equal(estimateGlobalFineNarrowBandBrickCapacity([120, 90, 80], 12), 405_000);
});

test("global fine capacity is an explicit surface-area times band plan", () => {
  const balanced = planGlobalFineNarrowBandBrickCapacity([60, 45, 40], 7);
  assert.deepEqual(balanced, {
    logicalBrickCount: 108_000,
    maximumInterfaceAreaBricks: 2_700,
    bandLayers: 15,
    bandBrickCount: 40_500,
    surfaceGrowthSafety: 1.5,
    surfaceGrowthHeadroomBricks: 20_250,
    maximumResidentBricks: 60_750,
  });
  const doubled = planGlobalFineNarrowBandBrickCapacity([120, 90, 80], 7);
  assert.equal(doubled.maximumResidentBricks, 4 * balanced.maximumResidentBricks,
    "fixed-width sparse bands must grow with interface area, not logical volume");
  assert.equal(doubled.logicalBrickCount, 8 * balanced.logicalBrickCount);
  const clipped = planGlobalFineNarrowBandBrickCapacity([4, 3, 2], 7);
  assert.equal(clipped.maximumResidentBricks, clipped.logicalBrickCount,
    "a band wider than a tiny domain may conservatively cover that whole domain");
});

test("the Section 5 fine band retains a rolling ocean surface through advection", () => {
  // Aanjaneya et al. 2017, Section 5
  // (`docs/papers/aanjaneya-2017-power-liquids.txt`) assumes that the fine
  // SPGrid surrounding the free surface is updated at every advection step.
  // It does not assume the surface stays planar. The capacity policy must
  // therefore include deformation headroom instead of reserving only the
  // initial area-times-band estimate. Dawn reaches 679,613 desired pages as
  // the authored ocean slab rolls over; the former 1.25 policy held 652,800
  // and violated that retained-fine-authority assumption.
  const ocean = planGlobalFineNarrowBandBrickCapacity([320, 96, 80], 8);
  assert.equal(ocean.bandBrickCount, 522_240);
  assert.equal(ocean.surfaceGrowthSafety, 1.5);
  assert.equal(ocean.maximumResidentBricks, 783_360);
  assert.ok(ocean.maximumResidentBricks >= 679_613,
    "the evolving Section 5 surface band must remain resident at the reproduced Dawn crest");
});

test("fine summary budgets a bounded sparse paged directory and compact active mip", () => {
  const fine = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
    finestCellDimensions: [60, 45, 40], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 50_625 });
  const summary = planFineLevelSetGPUSummaries(fine, 45_312);
  assert.equal(summary.fineEntryCapacity, 66_510);
  assert.equal(summary.coarseEntryCapacity, 45_312);
  assert.equal(summary.entryCapacity, 111_822);
  assert.equal(summary.directoryPageSize, 32);
  assert.equal(summary.hierarchyTopLevelPages,
    Math.ceil(summary.hierarchyKeyCapacity / summary.directoryPageSize));
  assert.equal(summary.directoryPageCapacity,
    Math.min(summary.hierarchyTopLevelPages, summary.entryCapacity));
  assert.equal(summary.directoryBytes,
    (16 + summary.hierarchyTopLevelPages
      + summary.directoryPageCapacity * summary.directoryPageSize
      + summary.entryCapacity * FINE_LEVELSET_SUMMARY_ENTRY_WORDS) * 4);
  assert.equal(summary.fineEntriesBytes, summary.entryCapacity * FINE_LEVELSET_SUMMARY_ENTRY_WORDS * 4);
  assert.equal(summary.keyStateBytes, summary.entryCapacity * 8,
    "fine references and corrected-coarse rows exist only for compact active ranks");
  assert.equal(summary.rankStateBytes, summary.entryCapacity * 8);
  assert.equal(summary.pageStateBytes, summary.directoryPageCapacity * 8);
  assert.equal(summary.indirectBytes, 4 * 12,
    "fine-summary work has fixed validation, coarse, mutation, and active-rank dispatch records");
  assert.equal(summary.workStateBytes, 256);
  assert.equal(summary.allocatedBytes,
    summary.directoryBytes + summary.fineEntriesBytes + summary.keyStateBytes
      + summary.pageStateBytes
      + summary.rankStateBytes + summary.workStateBytes + summary.indirectBytes + summary.parameterBytes);
  assert.ok(summary.entryCapacity >= summary.fineEntryCapacity,
    "every simultaneously active fine ancestor must have a compact rank");
});

test("fine summary hierarchy metadata scales with active capacity plus top-level pages", () => {
  const fine = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
    finestCellDimensions: [512, 512, 512], finestCellWidth: 1,
    fineFactor: 8, brickResolution: 4, maximumResidentBricks: 65_536 });
  const summary = planFineLevelSetGPUSummaries(fine, 65_536);
  const oldDenseKeyStateBytes = summary.hierarchyKeyCapacity * 3 * 4;
  assert.ok(summary.hierarchyKeyCapacity > 1_000_000_000);
  assert.ok(summary.keyStateBytes + summary.pageStateBytes < oldDenseKeyStateBytes / 100,
    "reference, coarse-row, and page ownership state must not allocate per logical hierarchy key");
  assert.ok(summary.directoryPageCapacity <= summary.entryCapacity);
  assert.equal(summary.directoryWords,
    16 + summary.hierarchyTopLevelPages
      + summary.directoryPageCapacity * summary.directoryPageSize
      + summary.entryCapacity * FINE_LEVELSET_SUMMARY_ENTRY_WORDS);
});

test("parallel total-volume scratch is bounded by compact directory and resident fine samples", () => {
  const a = planFineLevelSetGPUVolume(257, 4097, true);
  assert.equal(a.coarsePartialCount, 5); assert.equal(a.finePartialCount, 65);
  assert.equal(a.coarsePartialBytes, 80); assert.equal(a.finePartialBytes, 2080);
  assert.equal(a.reductionScratchBytes, 2080); assert.equal(a.allocatedBytes, 64 + 16 + 2080 + 16 + 12 + 64,
    "coarse parameters plus fine and exact coarse indirect records are all charged");
  const b = planFineLevelSetGPUVolume(257, 4097, false);
  assert.equal(b.allocatedBytes, a.allocatedBytes - 64, "B must share, not double-count, the A/B reference control");
  const snapshot = planFineLevelSetGPUVolume(257, 64, true, 1024);
  assert.equal(snapshot.coarsePartialCount, 16);
  assert.equal(snapshot.coarsePartialBytes, 256);
  assert.equal(snapshot.reductionScratchBytes, 256);
  assert.equal(snapshot.allocatedBytes, 64 + 16 + 256 + 16 + 12 + 64,
    "the accepted coarse-directory snapshot, not only live row capacity, sizes coarse reduction scratch");
});

test("coarse and structured allocations scale with compact row capacity", () => {
  const compact = (rows: number) => planOctreePowerDescriptors(rows).allocatedBytes
    + planOctreeCoarsePhi(rows).allocatedBytes + planOctreePowerCoarseLevelSet(rows).allocatedBytes
    + planStructuredVelocityGPU(rows).allocatedBytes;
  const low = compact(128), highRows = compact(256);
  assert.ok(highRows > low);
  // None of these planners accepts finest-domain voxel count: compact rows and
  // bounded catalog slots are their only scale inputs.
  assert.equal(compact(128), low);
});

test("power allocation accounting charges each structured arena once", () => {
  assert.equal(sumOctreePowerAllocationBreakdown({ structured: 52_841_136, solver: 13_520_048,
    topology: 8_383_068 }), 74_744_252);
  assert.throws(() => sumOctreePowerAllocationBreakdown({ faces: -1 }), /non-negative safe bytes/);
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const accounting = source.match(/const powerAllocated = sumOctreePowerAllocationBreakdown\(\{[\s\S]*?\}\);/)?.[0];
  assert.ok(accounting);
  assert.equal(accounting.match(/structuredVelocity: structured\.allocatedBytes/g)?.length, 1);
});

test("fine-to-coarse capacity diagnostics decode fail-closed control words", () => {
  assert.deepEqual(FINE_TO_COARSE_LEVELSET_ERROR,
    { capacity: 1, unowned: 2, nonfinite: 4, unpublishedSource: 8 });
  assert.deepEqual(unpackFineToCoarseGPUControl([17, 5, 3, 2, 9, 0]), {
    contributionCount: 17, maximumContributionsPerRow: 5, flags: 3,
    unacceptedRows: 2, rowCount: 9, valid: false,
  });
});
