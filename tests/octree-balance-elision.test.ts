import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  octreeBalancePredicatesWouldSplit,
  octreeBalanceRounds,
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
