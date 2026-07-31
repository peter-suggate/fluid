import assert from "node:assert/strict";
import test from "node:test";
import { planOctreeCoarseSummary } from "../lib/webgpu-octree-coarse-summary";

test("factor-one coarse summary uses B4 nodes and bounded sparse entries", () => {
  const plan = planOctreeCoarseSummary([16, 16, 16], 4_096);
  assert.deepEqual(plan.baseDimensions, [4, 4, 4]);
  assert.deepEqual(plan.levelDimensions, [[4, 4, 4], [2, 2, 2], [1, 1, 1]]);
  assert.deepEqual(plan.levelOffsets, [0, 64, 72]);
  assert.equal(plan.hierarchyKeyCapacity, 73);
  assert.equal(plan.entryCapacity, 73);
  assert.ok(plan.directoryWords * 4 < 64 * 1_024,
    "the complete coarse lattice remains far smaller than the former 641 KiB fine band");
});

test("coarse summary entry capacity is bounded by live-row ancestry", () => {
  const plan = planOctreeCoarseSummary([256, 128, 64], 128);
  assert.ok(plan.entryCapacity <= 128 * plan.levelDimensions.length);
  assert.ok(plan.entryCapacity < plan.hierarchyKeyCapacity);
  assert.ok(plan.directoryPageCapacity <= plan.entryCapacity);
});
