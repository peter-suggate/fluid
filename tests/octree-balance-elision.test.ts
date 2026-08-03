import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  octreeBalancePredicatesWouldSplit,
  octreeBalanceRounds,
  octreeEffectiveLeafSize,
  octreeGradingFixpointEnabled,
  octreeGradingMembershipLoadEnabled,
  octreeGradingPageFillEnabled,
  octreeGradingSplitHelpersEnabled,
  octreeProjectionPipelineRequired,
  octreeProjectionShader,
} from "../lib/webgpu-octree";
import { OCTREE_OWNER_ARENA_CONTROL_WORDS } from "../lib/webgpu-octree-owner-pages";

const octreeSource = readFileSync(
  new URL("../lib/webgpu-octree.ts", import.meta.url),
  "utf8",
);

test("maximum leaf size two cannot satisfy any octree balance predicate", () => {
  const ringWidth = 18;
  const ringConfigurations = 1 << ringWidth;
  for (const anchorSize of [1, 2]) {
    for (let mask = 0; mask < ringConfigurations; mask += 1) {
      const neighborSizes = Array.from(
        { length: ringWidth },
        (_, direction) => 1 + ((mask >>> direction) & 1),
      );
      assert.equal(
        octreeBalancePredicatesWouldSplit(anchorSize, neighborSizes),
        false,
        `anchor ${anchorSize}, ring mask ${mask.toString(16)} must be balance-closed`,
      );
    }
  }
});

test("balance-round planning elides only the maximum-leaf-size-two domain", () => {
  assert.equal(octreeBalanceRounds(2), 0);
  assert.equal(octreeBalanceRounds(4), 4);
  assert.equal(octreeBalanceRounds(8), 6);
  assert.equal(octreeBalanceRounds(16), 8);
  assert.equal(octreeBalanceRounds(32), 10);

  assert.equal(octreeBalancePredicatesWouldSplit(1, [4]), true,
    "the ordinary ratio predicate remains reachable above maximum size two");
  assert.equal(octreeBalancePredicatesWouldSplit(2, [1, 4]), true,
    "the mixed paper-ring predicate remains reachable above maximum size two");
  assert.equal(octreeBalancePredicatesWouldSplit(4, [1]), true,
    "the face-neighbor predicate remains reachable above maximum size two");
});

test("host scheduling guards recurring setup and cold/recurring grading dispatches", () => {
  assert.match(
    octreeSource,
    /this\.balanceRounds = octreeBalanceRounds\(this\.effectiveLeafSize\);/,
    "the largest leaf the domain can hold must determine the common cold/recurring round count",
  );
  assert.match(
    octreeSource,
    /if \(active && !analyticColdBootstrap && this\.balanceRounds > 0\) \{[\s\S]*label: "Octree resident grading closure"/,
    "a size-two recurring generation must not restore grading worklists or open a grading pass",
  );
  assert.match(
    octreeSource,
    /for \(let round = 0; round < this\.balanceRounds; round \+= 1\) \{[\s\S]*dispatchCandidates\(this\.balancePipeline, this\.balanceDeltaPipeline\);/,
    "the zero round count must suppress both cold full-domain and recurring delta balance dispatches",
  );
});

test("the split materializer resolves one owner word per page, not per cell", () => {
  const materializer = octreeSource.slice(
    octreeSource.indexOf("fn splitPageAt"),
    octreeSource.indexOf("fn splitLeafSeeded"),
  );
  assert.match(materializer,
    /fn splitPageAt[\s\S]*return SplitPage\([\s\S]*encodePagedOwner\(brickOrigin, childOrigin, child\)\)/,
    "a page inside one child has a single owner word; resolving it per cell is the defect");
  assert.doesNotMatch(materializer,
    /\/ vec3u\(child\)|firstTrailingBit/,
    "the per-cell floor-to-child division must not reappear inside the page fill");
  assert.match(materializer,
    /fn fillSplitPage[\s\S]*for \(var slot = 1u; slot < 512u; slot \+= 1u\) \{\s*let at = plan\.base \+ slot;\s*atomicMin\(&owners\[at\], splitOwnerWord\(at, plan\.word\)\);/,
    "the fill must be one contiguous atomic per cell over the whole page");
  assert.match(materializer,
    /fn splitOwnerWord[\s\S]*if \(topologyCandidateView == 1u && !gradingMembershipLoad\) \{ return word; \}/,
    "the membership load must be elided only inside the topology candidate view");
});

test("the elided membership load rests on the candidate bank clearing that bit", () => {
  // splitOwnerWord drops the per-cell OWNER_WORD_TOPOLOGY load because the
  // candidate payload bank is rewritten with the bit clear in the pass
  // immediately before every topology dispatch. If that clear ever goes away
  // the materializer silently coarsens marked leaf origins, so the premise is
  // asserted across the module boundary rather than left in a comment.
  const ownerPages = readFileSync(
    new URL("../lib/webgpu-octree-owner-pages.ts", import.meta.url),
    "utf8",
  );
  assert.match(ownerPages, /word&=~OWNER_WORD_TOPOLOGY;/,
    "commitOwnerPageCandidate must keep clearing membership on the inactive payload bank");
  assert.match(octreeSource,
    /markAcceptedOwner\(unpackOrigin\(acceptedOwner\.packedOrigin\)\)/,
    "membership must remain published only by frontier emission, which follows grading");
});

test("grading materializer arms are environment-selected with the fast form on by default", () => {
  assert.equal(octreeGradingPageFillEnabled({}), true);
  assert.equal(octreeGradingPageFillEnabled({ FLUID_OCTREE_GRADING_PAGE_FILL: "0" }), false,
    "the serial per-cell walk stays reachable as the A/B control");
  assert.equal(octreeGradingSplitHelpersEnabled({}), true);
  assert.equal(octreeGradingSplitHelpersEnabled(
    { FLUID_OCTREE_GRADING_SPLIT_HELPERS: "0" }), false);
  assert.equal(octreeGradingMembershipLoadEnabled({}), false,
    "the redundant per-cell membership load is off by default");
  assert.equal(octreeGradingMembershipLoadEnabled(
    { FLUID_OCTREE_GRADING_MEMBERSHIP_LOAD: "1" }), true);
});

test("the grading fixpoint is off by default and leaves the shader byte-identical", () => {
  // A broken default in a shared worktree costs far more than an unfinished
  // experiment, so this one is spliced at the source rather than selected by a
  // pipeline override: with the variable unset there is no extra entry point to
  // compile, no extra arena word to shift the owner tables, and not one byte of
  // difference in the module.
  assert.equal(octreeGradingFixpointEnabled({}), false);
  assert.equal(octreeGradingFixpointEnabled({ FLUID_OCTREE_GRADING_FIXPOINT: "0" }), false);
  assert.equal(octreeGradingFixpointEnabled({ FLUID_OCTREE_GRADING_FIXPOINT: "1" }), true);
  assert.equal(process.env.FLUID_OCTREE_GRADING_FIXPOINT, undefined,
    "this test asserts the unflagged shape, so the variable must not be set for it");
  assert.doesNotMatch(octreeProjectionShader,
    /advanceGradingRound|gradingRoundActive|markGradingSplit|GRADING_SPLIT_FLAG/,
    "the unflagged module must carry no part of the experiment");
  assert.equal(OCTREE_OWNER_ARENA_CONTROL_WORDS, 16,
    "the unflagged owner arena must keep its sixteen-word control block");
  assert.equal(octreeProjectionPipelineRequired("advanceGradingRound", {
    solidRasterization: true,
    localFrontierCandidateSort: true,
    cooperativeRowDeltaRing: true,
  }), false, "the unflagged tree must not compile the carry pipeline");
  // The splices are exact single-match replacements that throw at module
  // evaluation, so a shader that half-applies cannot exist.
  assert.match(octreeSource,
    /Grading-fixpoint shader splice matched \$\{occurrences\} times, expected 1/);
  // Zero must mean "keep grading", so a freshly created arena runs the full
  // closure. The inverse polarity silently disables grading wherever the rounds
  // are reached before the flag is seeded, which publishes no liquid rows.
  assert.match(octreeSource,
    /fn gradingRoundActive\(\) -> bool \{\s*return atomicLoad\(&owners\[GRADING_CONVERGED\]\) == 0u;/);
});

test("the refinement ladder stops at the largest leaf the domain can hold", () => {
  // A leaf is dyadic and size-aligned, so origin + size <= dims must hold on
  // every axis. A domain whose shortest axis is 16 contains no size-32 leaf and
  // can never grow one, because refinement only makes leaves finer. Every rung
  // above that size is a dispatch that provably matches nothing.
  assert.equal(octreeEffectiveLeafSize(32, { nx: 32, ny: 16, nz: 32 }), 16,
    "symmetric-expansion is 16 cells tall, so its ladder tops out at 16");
  assert.equal(octreeEffectiveLeafSize(32, { nx: 16, ny: 16, nz: 16 }), 16);
  assert.equal(octreeEffectiveLeafSize(32, { nx: 24, ny: 24, nz: 24 }), 16,
    "a non-power-of-two extent still cannot hold a leaf larger than itself");
  assert.equal(octreeEffectiveLeafSize(8, { nx: 256, ny: 256, nz: 256 }), 8,
    "the authored maximum is still an upper bound");
  assert.equal(octreeEffectiveLeafSize(32, { nx: 2, ny: 2, nz: 2 }), 2,
    "the ladder never drops below the smallest leaf");
  // Regression safety: every lane whose shortest axis already admits the
  // authored leaf is bit-identical, which is why this cannot move the measured
  // droplet-256 or ocean results.
  for (const dims of [{ nx: 256, ny: 256, nz: 256 }, { nx: 320, ny: 96, nz: 80 },
    { nx: 64, ny: 64, nz: 64 }]) {
    assert.equal(octreeEffectiveLeafSize(32, dims), 32);
    assert.equal(octreeBalanceRounds(octreeEffectiveLeafSize(32, dims)),
      octreeBalanceRounds(32));
  }
  assert.equal(octreeBalanceRounds(octreeEffectiveLeafSize(32, { nx: 32, ny: 16, nz: 32 })), 8,
    "two of the ten fixed-point rounds budget for a tree level that cannot exist");
  // The ladder, the coarse ladder, the grading round budget and the coarse
  // pipeline set must all derive from the same effective size, or a dispatched
  // size would have no compiled pipeline.
  assert.match(octreeSource,
    /this\.effectiveLeafSize = octreeEffectiveLeafSize\(this\.maxLeafSize, dims\);/);
  assert.match(octreeSource,
    /for \(let size = this\.effectiveLeafSize; size >= 2; size >>= 1\) sizes\.push\(size\);/);
  assert.match(octreeSource,
    /this\.balanceRounds = octreeBalanceRounds\(this\.effectiveLeafSize\);/);
  assert.match(octreeSource,
    /for \(let size = this\.effectiveLeafSize; size >= 16; size >>= 1\) \{/,
    "the coarse pipeline set must cover exactly the dispatched coarse sizes");
  assert.match(octreeSource, /effectiveLeafSize: this\.effectiveLeafSize,/,
    "the pipeline cache key must distinguish two domains that share an authored leaf");
});
