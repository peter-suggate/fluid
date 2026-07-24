import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  fluidBrickResidencyShader,
  planFluidBrickResidencyAllocation,
  planFineSeedCandidateResidencyPools,
  sparseFineSeedCandidateResidencyShader,
  fineSeedCandidateCommitShader,
  fineSeedCandidateResidencyShader,
} from "../lib/webgpu-fluid-brick-residency";

test("direct-paged identity residency stores one sentinel instead of a box-sized identity map", () => {
  // 320x96x80 cells at the production 8-cell brick size; topology tiles are
  // 4 bricks wide (the 32-cell maximum leaf extent).
  const implicit = planFluidBrickResidencyAllocation([40, 12, 10], [10, 3, 3], 4_800);
  const compatibility = planFluidBrickResidencyAllocation([40, 12, 10], [10, 3, 3], 4_800, true);

  assert.equal(implicit.brickCapacity, 4_800);
  assert.equal(implicit.tileCapacity, 90);
  assert.equal(implicit.identityMapping, "implicit");
  assert.equal(implicit.leafIndexBytes, 4);
  assert.equal(implicit.savedIdentityBytes, 19_196);
  assert.equal(compatibility.identityMapping, "explicit");
  assert.equal(compatibility.leafIndexBytes, 19_200);
  assert.equal(compatibility.allocatedBytes - implicit.allocatedBytes, 19_196);
  assert.equal(implicit.allocatedBytes,
    implicit.stateBytes + implicit.worklistBytes + implicit.tileWorklistBytes
      + implicit.tileStateBytes + implicit.leafIndexBytes + implicit.leafStateBytes + implicit.parameterBytes
      + implicit.transactionalBytes);
});

test("fine-seed-candidate-only mode collapses both unused legacy leaf buffers", () => {
  const candidateOnly = planFluidBrickResidencyAllocation([40, 12, 10], [10, 3, 3], 4_800, false, true);
  const general = planFluidBrickResidencyAllocation([40, 12, 10], [10, 3, 3], 4_800);

  assert.equal(candidateOnly.fineSeedCandidatesOnly, true);
  assert.equal(candidateOnly.identityMapping, "implicit");
  assert.equal(candidateOnly.leafIndexBytes, 4);
  assert.equal(candidateOnly.leafStateBytes, 4);
  assert.equal(candidateOnly.savedLeafStateBytes, 19_196);
  assert.equal(candidateOnly.allocatedBytes, 194_492);
  assert.equal(general.allocatedBytes - candidateOnly.allocatedBytes, 19_196);
  assert.throws(
    () => planFluidBrickResidencyAllocation([1, 1, 1], [1, 1, 1], 1, true, true),
    /requires implicit brick\/leaf identity/,
  );
});

test("direct-page scheduler storage follows producer key pools rather than logical box volume", () => {
  const sparse = planFluidBrickResidencyAllocation(
    [400, 120, 100], [100, 30, 25], 4_800_000, false, true, 120_000, 24_000,
  );
  assert.equal(sparse.sparseKeyPools, true);
  assert.equal(sparse.brickCapacity, 4_800_000);
  assert.equal(sparse.brickStateCapacity, 120_000);
  assert.equal(sparse.tileCapacity, 75_000);
  assert.equal(sparse.tileStateCapacity, 24_000);
  assert.equal(sparse.stateBytes, 120_000 * 8, "sparse state records are key + lifecycle state");
  assert.equal(sparse.tileStateBytes, 24_000 * 8);
  assert.equal(sparse.allocatedBytes, 6_528_328);
  assert.equal(sparse.savedSchedulerBytes, 187_272_000);

  const pools = planFineSeedCandidateResidencyPools(
    [400, 120, 100], [100, 30, 25], 8, 4, 120_000, 100,
  );
  assert.deepEqual(pools, {
    brickCapacity: 120_000, tileCapacity: 18_750,
    logicalBrickCount: 4_800_000, logicalTileCount: 75_000,
    bandBrickLayers: 3, bandTileLayers: 3,
  });
  assert.throws(
    () => planFluidBrickResidencyAllocation([1, 1, 1], [1, 1, 1], 1, false, true, undefined, 1),
    /Sparse tile capacity requires/,
  );
});

test("production-sized sparse scheduler accounting exposes both deep-domain savings and small-grid overhead", () => {
  const oceanPools = planFineSeedCandidateResidencyPools([40, 12, 10], [10, 3, 3], 8, 4, 384_768);
  const ocean = planFluidBrickResidencyAllocation(
    [40, 12, 10], [10, 3, 3], 4_800, false, true,
    oceanPools.brickCapacity, oceanPools.tileCapacity,
  );
  assert.equal(oceanPools.brickCapacity, 3_000);
  assert.equal(ocean.allocatedBytes, 147_208);
  assert.equal(ocean.savedSchedulerBytes, 47_280);
  assert.equal(ocean.schedulerByteDelta, -47_280);

  const damPools = planFineSeedCandidateResidencyPools([8, 6, 5], [2, 2, 2], 8, 4, 41_472);
  const dam = planFluidBrickResidencyAllocation(
    [8, 6, 5], [2, 2, 2], 240, false, true,
    damPools.brickCapacity, damPools.tileCapacity,
  );
  assert.equal(dam.allocatedBytes, 12_104);
  assert.equal(dam.savedSchedulerBytes, 0);
  assert.equal(dam.schedulerByteDelta, 1_984,
    "key records have an explicit small-domain cost when the pool clamps to the logical domain");
});

test("implicit identity is reconstructed by the GPU while explicit mappings retain compatibility", () => {
  assert.match(fluidBrickResidencyShader, /const IMPLICIT_IDENTITY: u32 = 0xffffffffu/);
  assert.match(fluidBrickResidencyShader,
    /arrayLength\(&leafIndices\) == 1u && leafIndices\[0\] == IMPLICIT_IDENTITY/);
  assert.match(fluidBrickResidencyShader, /let leafIndex = leafIndexFor\(brickIndex\)/);
  assert.doesNotMatch(fluidBrickResidencyShader,
    /brickIndex >= arrayLength\(&leafIndices\)/,
    "the sentinel-backed identity map deliberately has only one word");

  const source = readFileSync(new URL("../lib/webgpu-fluid-brick-residency.ts", import.meta.url), "utf8");
  assert.match(source, /explicitMapping \?\? new Uint32Array\(\[0xffff_ffff\]\)/);
  assert.match(source, /explicitMapping !== undefined/);
  assert.match(source, /get allocatedBytes\(\): number \{ return this\.currentAllocationPlan\.allocatedBytes; \}/);
  assert.doesNotMatch(source, /Fine-seed-candidate-only residency cannot classify dense level-set textures/,
    "bootstrap classification is safe because leaf-state publication is bounds checked");
  assert.match(fluidBrickResidencyShader, /if \(leafIndex < arrayLength\(&leafStates\)\)/);
  assert.match(source, /candidateOnly \? "Unused sparse leaf residency fallback"/);
  assert.doesNotMatch(source, /cutoverToFineSeedCandidatesOnly|candidateTransaction/,
    "the immutable candidate-authority plan has no legacy runtime cutover or transaction buffer");
});

test("fine-seed-candidate publication is deterministic, fail-closed, and count-independent", () => {
  assert.match(fineSeedCandidateResidencyShader, /fn producerAccepted\(\)->bool/);
  assert.match(fineSeedCandidateResidencyShader, /candidateControl\[5\]==1u&&candidateControl\[6\]==0u/);
  assert.match(fineSeedCandidateResidencyShader, /candidateControl\[4\]>publishedTileWorklist\[15\]/,
    "stale or unpublished candidate generations must preserve the prior bounded worklist");
  assert.match(fineSeedCandidateResidencyShader, /tileWorklist\[11\]=COMMIT/);
  assert.match(fineSeedCandidateResidencyShader,
    /storageBarrier\(\);workgroupBarrier\(\);commitCandidateGeneration\(lid,256u\)/,
    "the validated dense generation is committed by its final publication workgroup");
  assert.match(fineSeedCandidateCommitShader, /if\(candidateTileWorklist\[11\]!=COMMIT\)\{return;\}/,
    "the sparse compatibility publisher retains its separate GPU commit predicate");
  assert.doesNotMatch(fineSeedCandidateResidencyShader, /candidateControl\[0\]==0u/,
    "zero count is a valid-empty publication, not a failed-empty proxy");
  assert.match(fineSeedCandidateResidencyShader,
    /@compute @workgroup_size\(256\) fn publishFineSeedCandidateResidency/);
  assert.match(fineSeedCandidateResidencyShader, /atomicOr\(&states\[logical\],bits\)/,
    "parallel candidate marks must be idempotent and order independent");
  assert.match(fineSeedCandidateResidencyShader,
    /activePrefix\[lane\]=a;retiredPrefix\[lane\]=r[\s\S]*worklist\[0\]=a;worklist\[4\]=r/,
    "the bounded lane-prefix pass must preserve canonical logical-key ordering");
  assert.deepEqual([...fineSeedCandidateResidencyShader.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s+fn\s+([A-Za-z0-9_]+)/g,
  )].map((match) => match[1]), [
    "prepareFineSeedCandidateResidency",
    "markFineSeedCandidateResidency",
    "resolveFineSeedCandidateResidency",
    "publishFineSeedCandidateResidency",
  ]);
});

test("sparse candidate scheduler fails closed on brick or tile key-pool exhaustion", () => {
  assert.match(sparseFineSeedCandidateResidencyShader, /logical\+1u/);
  assert.match(sparseFineSeedCandidateResidencyShader, /states\[slot\*2u\]=INVALID/,
    "retired keys leave tombstones so collision chains remain searchable");
  assert.match(sparseFineSeedCandidateResidencyShader,
    /brickClaimKeys\[cache\]==encoded[\s\S]*let slot=claimBrick\(logical\)/,
    "duplicate brick marks may bypass probing, while cache misses retain canonical insertion order");
  assert.match(sparseFineSeedCandidateResidencyShader,
    /tileRingKeys\[ringCache\]==encodedCenter[\s\S]*tileRingKeys\[ringCache\]=encodedCenter/,
    "an already completed tile ring may be reused without changing partial-failure behavior");
  assert.match(sparseFineSeedCandidateResidencyShader, /slot==INVALID\)\{sparseError=6u/);
  assert.match(sparseFineSeedCandidateResidencyShader, /slot==INVALID\)\{sparseError=7u/);
  assert.match(sparseFineSeedCandidateResidencyShader,
    /@workgroup_size\(256\)[\s\S]*brickChunk[\s\S]*sparsePrefix0\[lane\]=a[\s\S]*if\(lid==0u\)\{finishHeaders\(\);\}/,
    "pool lifecycle and publication use stable contiguous lane prefixes while hash insertion remains ordered");
  assert.doesNotMatch(sparseFineSeedCandidateResidencyShader,
    /\batomic(?:Add|And|CompareExchangeWeak|Exchange|Load|Max|Min|Or|Store|Sub|Xor)\b/);
  assert.deepEqual([...sparseFineSeedCandidateResidencyShader.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s+fn\s+([A-Za-z0-9_]+)/g,
  )].map((match) => match[1]), ["publishFineSeedCandidateResidency"]);
});

test("candidate residency encoding stages dense work across the device with no recurring host clears or copies", () => {
  const source = readFileSync(new URL("../lib/webgpu-fluid-brick-residency.ts", import.meta.url), "utf8");
  const body = source.match(/encodeFineSeedCandidates\([\s\S]*?async readStats/)?.[0] ?? "";
  assert.match(body, /fineSeedCandidatePipelines\.slice\(0,3\)\.forEach/);
  assert.match(body, /fineSeedCandidatePublishLayout[\s\S]*finalPublishGroup[\s\S]*dispatchWorkgroups\(1\)/,
    "the dense final publisher uses a reachability-minimal layout within the ten-storage-buffer device limit");
  assert.match(body, /Math\.ceil\(prepareWords \/ 256\)/);
  assert.match(body, /Math\.ceil\(markItems \/ 256\)/);
  assert.match(body, /Math\.ceil\(this\.publicationCapacity \/ 256\)/);
  assert.match(body, /if \(this\.currentAllocationPlan\.sparseKeyPools\) \{[\s\S]*const commitGroup=/,
    "only sparse-key publication pays for a separate commit bind group and dispatch");
  assert.doesNotMatch(body, /clearBuffer|copyBufferToBuffer/);
  assert.doesNotMatch(source,
    /beginFineSeedCandidatesPipeline|markPressureTopologyTilesPipeline|markFineSeedCandidatesPipeline|resolveFineSeedCandidatesPipeline|emitFineSeedCandidateTilesPipeline|finalizeFineSeedCandidatesPipeline/);
});

test("allocation planner rejects malformed domains", () => {
  assert.throws(() => planFluidBrickResidencyAllocation([0, 1, 1], [1, 1, 1], 1), /positive integers/);
  assert.throws(() => planFluidBrickResidencyAllocation([1, 1, 1], [1, 1, 1], 0), /Leaf capacity/);
});
