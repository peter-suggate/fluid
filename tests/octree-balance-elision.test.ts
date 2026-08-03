import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  octreeBalancePredicatesWouldSplit,
  octreeBalanceRounds,
  octreeGradingMembershipLoadEnabled,
  octreeGradingPageFillEnabled,
  octreeGradingSplitHelpersEnabled,
} from "../lib/webgpu-octree";

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
    /this\.balanceRounds = octreeBalanceRounds\(this\.maxLeafSize\);/,
    "the immutable maximum leaf size must determine the common cold/recurring round count",
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
