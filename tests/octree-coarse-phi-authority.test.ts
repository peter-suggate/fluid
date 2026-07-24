import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { WebGPUOctreeProjection, octreeProjectionShader } from "../lib/webgpu-octree";
import { fineLevelSetSummaryWGSL, WebGPUFineLevelSetSummaries } from "../lib/webgpu-octree-fine-levelset-summary";
import {
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  octreePowerCoarseDirectoryIsAuthoritative,
  octreePowerCoarseLevelSetShader,
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

test("packed coarse generation cannot alter any pre-existing pressure-capacity flag consumer", () => {
  const uses = [...octreeProjectionShader.matchAll(/params\.pressureCapacity\.w[^;\n]*/g)].map((match) => match[0]);
  assert.equal(uses.length, 2);
  assert.ok(uses.some((use) => use.includes(">>2u")));
  assert.ok(uses.some((use) => use.includes("& 1u")));
  assert.ok(uses.every((use) => !use.includes("!= 0u") || use.includes("&")), uses.join("\n"));
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
  const rebuild = WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString();
  assert.match(rebuild,
    /setBindGroup\(0,\s*active\s*\?\s*this\.fineSummarySizingGroup\s*:\s*this\.groups\.ab\)[\s\S]*classifyFrontierCandidates[\s\S]*emitFrontierCandidates/,
    "frontier filtering and insertion both consume current fine-summary authority");
  assert.match(fineLevelSetSummaryWGSL,
    /fn buildFineSummaryDelta[\s\S]*for\(var row=lid;row<coarseCount[\s\S]*coarseHierarchyKey\(coarseDirtyIdentity\(row\)\)/,
    "the persistent exact transaction builder consumes corrected-coarse delta rows directly");
  assert.match(fineLevelSetSummaryWGSL,
    /fn coarseAuthoritative[\s\S]*coarse\.state==PUBLISHED[\s\S]*coarseControl\[2\]/,
    "coarse live-prefix work is admitted only from the immutable current publication");
  assert.match(fineLevelSetSummaryWGSL, /Entry\(key,ordered\(e\.minimumPhi\)[\s\S]*COARSE_AUTHORITY/,
    "an exact corrected-coarse leaf marks the unified summary authoritative");
  assert.match(octreeProjectionShader, /result\.coarseAuthority = \(entryFlags & 0x80000000u\) != 0u/);
  assert.match(octreeProjectionShader,
    /else if\(fine\.complete&&!fine\.coarseAuthority\)\{/,
    "coarse-only summaries must preserve the exact coarse cell-centre phase used by power-boundary sampling");
  assert.match(fineLevelSetSummaryWGSL,
    /fn publicDirtySummary[\s\S]*committedFineAt\(key\)[\s\S]*coarseSummaryAt\(key\)[\s\S]*combineSummary/,
    "public dirty rows combine the exact private fine tree with current corrected-coarse contributions");
  assert.match(fineLevelSetSummaryWGSL,
    /fn stageFineOnlyCarry[\s\S]*fineDirtyContains[\s\S]*fn stageFineOnlyDirty/,
    "the private fine-only tree carries untouched rows and compacts changed ancestors independently");
  assert.match(fineLevelSetSummaryWGSL,
    /fn stageFineOnlyDirty[\s\S]*let fineDirty=first&&key!=INVALID&&fineDirtyContains\(key\)[\s\S]*dirtySummaryAt\(key\)/,
    "duplicate fine/coarse delta keys must query fine authority instead of trusting unstable equal-key sort order");
  assert.match(fineLevelSetSummaryWGSL,
    /fn coarseDirtyIdentity[\s\S]*coarseDelta\.items\[row\][\s\S]*fn buildFineSummaryDelta[\s\S]*records\[output\]=Entry/,
    "recurring corrected coarse work emits only the compact value/phase delta");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /sourceSlot|hashCapacity|maximumHashProbes/,
    "summary merging must consume the exact published row count, never scan unused directory capacity");
  const summaryEncodeOrder = WebGPUFineLevelSetSummaries.prototype.encode.toString();
  assert.ok(
    summaryEncodeOrder.indexOf('run("buildFineSummaryDelta"')
      < summaryEncodeOrder.indexOf("broker.updateIndirectBuffer"),
    "fine and coarse delta records are built persistently before the exact recompute extent is published",
  );
  assert.match(summaryEncodeOrder, /runIndirect\("recomputeFineSummaryBase"/);
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /prepareFineSummaryWork|summarizeFineBricks|mergeCoarsePhiSummaries|scanFineSummarySegments/);
  assert.match(fineLevelSetSummaryWGSL,
    /fn entryPresent[\s\S]*COARSE_AUTHORITY\|CENTER_COMPLETE[\s\S]*fn centerSummary[\s\S]*let span=\(1u<<p\.level\)\*resolution[\s\S]*index>=arrayLength\(&c\)\|\|index>=arrayLength\(&d\)[\s\S]*mask==0xffu[\s\S]*CENTER_COMPLETE/,
    "every dyadic node retains an exact finite eight-sample centre phase independently of narrow-band membership");
  const summaryEncode = WebGPUFineLevelSetSummaries.prototype.encode.toString();
  for (const field of ["source.metadata", "source.worklist", "source.flags", "source.phi"]) {
    assert.ok(summaryEncode.includes(field), `summary update must bind current fine ${field}`);
  }
  assert.doesNotMatch(fineLevelSetSummaryWGSL, /pageHash|finePageHash|hashProbe/,
    "fine summary lookup must use the canonical sorted worklist");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
    "fine-summary construction and publication must be fully atomic-free");
  assert.doesNotMatch(octreeProjectionShader, /legacyPhi|pagedSurface|surfacePagePhi/,
    "missing compact authority must fail closed instead of reviving a deleted page/dense fallback");
  assert.match(octreeProjectionShader, /coarseWord\(0u\)!=0x80000000u[\s\S]*coarseWord\(1u\)&0x3fffffffu\)!=expected/);
});

test("published-directory miss is air only after every requested sorted row publishes successfully", () => {
  assert.match(octreePowerCoarseLevelSetShader,
    /fn publishPowerCoarsePhi[\s\S]*slot>=requested\(\)[\s\S]*descriptor=rowDirectory\[slot\][\s\S]*candidateDirectory\.entries\[slot\]=SampleEntry/);
  assert.match(octreePowerCoarseLevelSetShader,
    /descriptor\.morton==morton\(header\.cell\)[\s\S]*directoryLess\(level\(prior\.size\),prior\.morton,level\(descriptor\.size\),descriptor\.morton\)/);
  assert.match(octreePowerCoarseLevelSetShader,
    /fn finalizePowerCoarsePhi[\s\S]*reduceAdvected\[0\]==count&&reduceDirectoryRows\[0\]==count[\s\S]*candidateDirectory\.state=VALID[\s\S]*candidateDirectory\.state=0u[\s\S]*publishPowerCoarsePhiDeltaAndCommit[\s\S]*sampleDirectory\.state=VALID/,
    "a malformed candidate must leave the prior coarse directory immutable");
  assert.doesNotMatch(octreePowerCoarseLevelSetShader,
    /hash|probe|atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/i,
    "coarse publication must be a fixed-record sorted reduction, not a hash or atomic append");
  assert.match(octreeProjectionShader, /A miss in a valid directory is the[\s\S]*explicit positive-air complement/);
});
