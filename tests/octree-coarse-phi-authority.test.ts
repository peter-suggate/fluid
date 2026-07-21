import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { octreeProjectionShader } from "../lib/webgpu-octree";
import { fineLevelSetSummaryWGSL } from "../lib/webgpu-octree-fine-levelset-summary";
import {
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  octreePowerCoarseDirectoryIsAuthoritative,
  octreePowerCoarseLevelSetShader,
} from "../lib/webgpu-octree-power-coarse-levelset";

const header = {
  state: OCTREE_POWER_COARSE_LEVELSET_VALID, generation: 7, hashCapacity: 32,
  maximumLeafSize: 8, dimensions: [16, 8, 4] as const, physicalCellSize: 0.25,
  actualHashCapacity: 32,
};

test("coarse directory authority rejects stale, unpublished, malformed, and dimension-mismatched headers", () => {
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative(header, 7, [16, 8, 4], 0.25), true);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative({ ...header, state: 0 }, 7, [16, 8, 4], 0.25), false);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative({ ...header, generation: 6 }, 7, [16, 8, 4], 0.25), false);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative(header, 7, [16, 8, 5], 0.25), false);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative({ ...header, actualHashCapacity: 16 }, 7,
    [16, 8, 4], 0.25), false);
  assert.equal(octreePowerCoarseDirectoryIsAuthoritative({ ...header, maximumLeafSize: 6 }, 7,
    [16, 8, 4], 0.25), false);
});

test("binding 15 cutover preserves bulk worklists only in extrapolation and paged rollback paths", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  assert.match(source, /ab: this\.createProjectionGroup[\s\S]*this\.pressureA, this\.pressureB, undefined, coarseDirectory\)/);
  assert.match(source, /extrapolateOut: this\.groups\.extrapolateOut/);
  assert.match(source, /extrapolateScratch: this\.groups\.extrapolateScratch/);
  assert.equal((octreeProjectionShader.match(/bulkResidentCell\(/g) ?? []).length, 4,
    "one definition plus exactly the three sparse extrapolation wrappers");
  assert.match(octreeProjectionShader,
    /fn pagedSurfaceBindings\(\) -> bool \{ return \(params\.pressureCapacity\.w & 2u\) != 0u; \}/,
    "the paged bit selects binding ABI only");
  assert.match(octreeProjectionShader,
    /fn pagedSurfaceAuthority\(\)[\s\S]*atomicLoad\(&solidOrSurface\[3\]\) == 0u[\s\S]*atomicLoad\(&solidOrSurface\[6\]\) > 0u[\s\S]*atomicLoad\(&solidOrSurface\[7\]\) > 0u/,
    "page-backed phi requires a fault-free, non-empty GPU-published generation");
  assert.match(source, /\(this\.pressureWarmStart \? 1 : 0\) \| \(generation << 2\)/,
    "global-fine generation packing cannot enable the paged bit");
});

test("packed coarse generation cannot alter any pre-existing pressure-capacity flag consumer", () => {
  const uses = [...octreeProjectionShader.matchAll(/params\.pressureCapacity\.w[^;\n]*/g)].map((match) => match[0]);
  assert.equal(uses.length, 3);
  assert.ok(uses.some((use) => use.includes(">>2u")));
  assert.ok(uses.some((use) => use.includes("& 2u")));
  assert.ok(uses.some((use) => use.includes("& 1u")));
  assert.ok(uses.every((use) => !use.includes("!= 0u") || use.includes("&")), uses.join("\n"));
});

test("fine-corrected intervals drive wet/refinement classification and invalid directories retain rollback", () => {
  assert.match(octreeProjectionShader, /if\(coarse\.authority\)\{return coarse\.minimumPhi<0\.0;\}/);
  assert.match(fineLevelSetSummaryWGSL, /mergeCoarsePhiSummaries/);
  assert.match(fineLevelSetSummaryWGSL, /coarse\.state!=PUBLISHED[\s\S]*coarse\.generation&0x3fffffffu/);
  assert.match(fineLevelSetSummaryWGSL, /atomicOr\(&directory\[base\+7u\],1u\)/,
    "an exact corrected-coarse leaf marks the unified summary authoritative");
  assert.match(octreeProjectionShader, /result\.coarseAuthority = \(fineSummaryWord\(base \+ 7u\) & 1u\) != 0u/);
  assert.match(octreeProjectionShader, /if \(!fineSummary\.complete\)[\s\S]*legacyPhi/,
    "a missing exact coarse/fine summary remains inconclusive and executes the rollback scan");
  assert.match(octreeProjectionShader, /if\(coarse\.authority\)\{return coarseClassificationPhi\(coarse\);\}[\s\S]*return legacyPhi\(p\);/);
  assert.match(octreeProjectionShader, /coarseWord\(0u\)!=0x80000000u[\s\S]*coarseWord\(1u\)&0x3fffffffu\)!=expected/);
});

test("published-directory miss is air only after every requested live row inserted successfully", () => {
  assert.match(octreePowerCoarseLevelSetShader,
    /fn publishPowerCoarsePhi[\s\S]*row>=requested\(\)[\s\S]*publishSample\(row,output\)/);
  assert.match(octreePowerCoarseLevelSetShader,
    /fn publishSample[\s\S]*fail\(row,128u\)/);
  assert.match(octreePowerCoarseLevelSetShader,
    /fn finalizePowerCoarsePhi\(\)\{if\(rejectedFine\(\)\)\{return;\}let complete=control\.rowCount>0u[\s\S]*atomicLoad\(&control\.advected\)==control\.rowCount[\s\S]*atomicStore\(&sampleDirectory\.state,VALID\)[\s\S]*atomicStore\(&sampleDirectory\.state,0u\)/,
    "an empty or partially advected compact row set must remain an unpublished coarse directory");
  assert.match(octreeProjectionShader, /A miss in a valid directory is the[\s\S]*explicit positive-air complement/);
});
