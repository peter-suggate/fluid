import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { WebGPUOctreeProjection, octreeProjectionShader } from "../lib/webgpu-octree";
import { fineLevelSetSummaryWGSL, WebGPUFineLevelSetSummaries } from "../lib/webgpu-octree-fine-levelset-summary";
import {
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  octreePowerCoarseDirectoryIsAuthoritative,
  
} from "../lib/webgpu-octree-power-coarse-levelset";

const header = {
  state: OCTREE_POWER_COARSE_LEVELSET_VALID, generation: 7, rowCount: 12,
  maximumLeafSize: 8, dimensions: [16, 8, 4] as const, physicalCellSize: 0.25,
  actualRowCapacity: 32,
};

test("coarse directory authority rejects stale, unpublished, malformed, and dimension-mismatched headers", () => {
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative(header, 7, [16, 8, 4], 0.25), true);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative({ ...header, state: 0 }, 7, [16, 8, 4], 0.25), false);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative({ ...header, generation: 6 }, 7, [16, 8, 4], 0.25), false);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative(header, 7, [16, 8, 5], 0.25), false);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative({ ...header, actualRowCapacity: 8 }, 7,
    [16, 8, 4], 0.25), false);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative({ ...header, maximumLeafSize: 6 }, 7,
    [16, 8, 4], 0.25), false);
});

test("binding 15 cutover keeps only compact coarse authority", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  assert.match(source, /ab: this\.createProjectionGroup[\s\S]*this\.pressureA, this\.pressureB, coarseDirectory\)/);
  assert.doesNotMatch(source, /pagedGroups|pagedPhi|SurfacePage/);
  assert.doesNotMatch(source, /extrapolateOut|extrapolateScratch/);
  assert.doesNotMatch(octreeProjectionShader, /bulkResidentCell|extrapolateSeedSparse|extrapolateSparse/,
    "Section 5 compact face extension supersedes texture-space sparse extrapolation");
  assert.match(source, /const flags = 1 \| \(generation << 2\)/,
    "global-fine generation packing always selects the optimized warm-start lane");
  assert.doesNotMatch(source, /pressureWarmStart/,
    "the power-octree warm-start policy must not remain configurable");
});

test("fine-corrected intervals drive refinement while exact centre phi drives wet classification", () => {
  assert.match(octreeProjectionShader,
    /if\(coarse\.authority&&coarse\.leafSize==owner\.size\)\{return coarse\.phi<0\.0;\}/,
    "a coarse interval may classify only its exact pressure leaf, never all newly refined children");
  assert.match(octreeProjectionShader,
    /fn liquidOwner\(owner: Owner\)[\s\S]*if\(analyticInitialPhiEnabled\(\)\)\{return analyticInitialPhi\(centre\)<0\.0;\}[\s\S]*if\(coarse\.authority&&coarse\.leafSize==0u\)\{return false;\}[\s\S]*if\(coarse\.authority&&coarse\.maximumPhi<0\.0\)\{return true;\}/,
    "the authored first solve is exclusive; after cutover, directory misses remain explicit air");
  assert.match(octreeProjectionShader,
    /let fine=fineLeafSummary\(origin,owner\.size\);[\s\S]*if\(fine\.found\)\{[\s\S]*fine\.centerValid[\s\S]*else if\(fine\.complete&&!fine\.coarseAuthority\)/,
    "recurring frontier phase selection consumes a current centre stencil independently of complete sparse coverage");
  const rebuild = WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidate.toString();
  assert.match(rebuild,
    /setBindGroup\(0,\s*active\s*\?\s*this\.fineSummarySizingGroup\s*:\s*this\.groups\.ab\)[\s\S]*classifyFrontierCandidates[\s\S]*emitFrontierCandidates/,
    "frontier filtering and insertion both consume current fine-summary authority");
  assert.match(fineLevelSetSummaryWGSL,
    /fn coarseAuthoritative[\s\S]*coarse\.state==PUBLISHED[\s\S]*coarseControl\[2\]/,
    "coarse live-prefix work is admitted only from the immutable current publication");
  assert.match(fineLevelSetSummaryWGSL,
    /fn coarseEntryAt[\s\S]*value\.minimumPhi=ordered\(e\.minimumPhi\)[\s\S]*COARSE_AUTHORITY/,
    "an exact corrected-coarse leaf marks the unified summary authoritative");
  assert.match(octreeProjectionShader, /result\.coarseAuthority = \(entryFlags & 0x80000000u\) != 0u/);
  assert.match(octreeProjectionShader,
    /else if\(fine\.complete&&!fine\.coarseAuthority\)\{/,
    "coarse-only summaries must preserve the exact coarse cell-centre phase used by power-boundary sampling");
  assert.match(fineLevelSetSummaryWGSL,
    /fn publishFineSummaryDirect[\s\S]*let coarseValue=coarseEntryAt\(key,rank\)[\s\S]*value=combine\(value,coarseValue\)/,
    "public active-mip rows combine current fine values with corrected-coarse authority");
  assert.match(fineLevelSetSummaryWGSL,
    /fn ensureFineSummaryCoarseDirectoryPages[\s\S]*coarseHierarchyKey[\s\S]*fn ensureFineSummaryCoarseRanks/,
    "corrected-coarse rows allocate through the same sparse direct directory as fine ancestors");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /sourceSlot|hashCapacity|maximumHashProbes|recordLowerBound|sortFineSummary|mergeFineSummary/,
    "summary publication must never scan, probe, sort, or merge a capacity-sized record arena");
  const summaryEncodeOrder = WebGPUFineLevelSetSummaries.prototype.encode.toString();
  assert.match(summaryEncodeOrder,
    /removeFineSummaryPages[\s\S]*ensureFineSummaryDirectoryPages[\s\S]*ensureFineSummaryRanks[\s\S]*addFineSummaryPages/,
    "retirement precedes staged sparse-page/rank allocation and exact reference mutation");
  assert.match(summaryEncodeOrder, /indirect\("recomputeFineSummaryBase"/);
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /prepareFineSummaryWork|summarizeFineBricks|mergeCoarsePhiSummaries|scanFineSummarySegments/);
  assert.match(fineLevelSetSummaryWGSL,
    /fn centerSampleAt[\s\S]*let span=\(1u<<p\.level\)\*resolution[\s\S]*index>=arrayLength\(&sampleFlags\)\|\|index>=arrayLength\(&finePhi\)[\s\S]*fn finishCenter[\s\S]*mask==0xffu[\s\S]*CENTER_COMPLETE/,
    "every dyadic node retains an exact finite eight-sample centre phase independently of narrow-band membership");
  assert.match(fineLevelSetSummaryWGSL,
    /fn coarseEntryAt[\s\S]*COARSE_AUTHORITY\|CENTER_COMPLETE/,
    "corrected coarse entries publish complete centre authority without pretending to own fine samples");
  const summaryEncode = WebGPUFineLevelSetSummaries.prototype.encode.toString();
  for (const field of ["source.metadata", "source.worklist", "source.flags", "source.phi"]) {
    assert.ok(summaryEncode.includes(field), `summary update must bind current fine ${field}`);
  }
  assert.doesNotMatch(fineLevelSetSummaryWGSL, /pageHash|finePageHash|hashProbe|while\(low<high\)/,
    "fine summary consumer lookup must use bounded sparse page/rank addressing without probing");
  assert.doesNotMatch(octreeProjectionShader, /legacyPhi|pagedSurface|surfacePagePhi/,
    "missing compact authority must fail closed instead of reviving a deleted page/dense fallback");
  assert.match(octreeProjectionShader, /coarseWord\(0u\)!=0x80000000u[\s\S]*coarseWord\(1u\)&0x3fffffffu\)!=expected/);
});

