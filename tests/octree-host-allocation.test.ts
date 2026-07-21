import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { planOctreeHostAllocation } from "../lib/octree-host-allocation";

const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");

test("octree host cutover replaces box-sized velocity transport with exact compatibility bytes", () => {
  const plan = planOctreeHostAllocation(80, 160, 60, "maccormack", true);
  assert.deepEqual(plan.velocityExtent, [1, 1, 1]);
  assert.deepEqual(plan.transportExtent, [1, 1, 1]);
  assert.deepEqual(plan.fluxExtent, [1, 1, 1]);
  assert.deepEqual(plan.pressureExtent, [1, 1, 1]);
  assert.deepEqual(plan.volumeExtent, [1, 1, 1]);
  assert.equal(plan.velocityAllocatedBytes, 4 * 16 + 2 * 8 + 8);
  assert.equal(plan.scalarAllocatedBytes, 4 * 4);
  assert.equal(plan.conditioningBytes, 4);
  assert.equal(plan.allocatedBytes, 4 * 16 + 2 * 8 + 8 + 4 * 4 + 4);
  assert.equal(plan.savedBytes, plan.denseBaselineBytes - plan.allocatedBytes);
});

test("default host plan preserves every dense extent and byte", () => {
  const plan = planOctreeHostAllocation(7, 5, 3, "semi-lagrangian", false);
  assert.deepEqual(plan.velocityExtent, [7, 5, 3]);
  assert.deepEqual(plan.transportExtent, [9, 7, 5]);
  assert.deepEqual(plan.fluxExtent, [7, 5, 3]);
  assert.deepEqual(plan.pressureExtent, [7, 5, 3]);
  assert.deepEqual(plan.volumeExtent, [7, 5, 3]);
  assert.equal(plan.allocatedBytes, 7 * 5 * 3 * (2 * 16 + 8 + 4 * 4 + 4) + 9 * 7 * 5 * 8);
  assert.equal(plan.savedBytes, 0);
});

test("large target reports the exact scalar and total host savings", () => {
  const plan = planOctreeHostAllocation(320, 96, 80, "maccormack", true);
  assert.equal(plan.denseScalarBaselineBytes, 39_321_600);
  assert.equal(plan.scalarAllocatedBytes, 16);
  assert.equal(plan.scalarSavedBytes, 39_321_584);
  assert.equal(plan.denseConditioningBaselineBytes, 9_830_400);
  assert.equal(plan.denseBaselineBytes, 267_500_672);
  assert.equal(plan.allocatedBytes, 108);
  assert.equal(plan.savedBytes, 267_500_564);
});

test("host allocation rejects invalid simulation extents", () => {
  assert.throws(() => planOctreeHostAllocation(0, 4, 4, "maccormack", true), RangeError);
});

test("compact-face host scalars never receive a box-sized upload, capture, or dense pipeline", () => {
  assert.match(uniformSource, /pressureA = scalarTexture\("r32float", this\.hostAllocation\.pressureExtent\)/);
  assert.match(uniformSource, /volumeA = scalarTexture\("r32float", this\.hostAllocation\.volumeExtent\)/);
  assert.match(uniformSource, /const data = this\.adaptiveFaceVelocityCutover \? undefined : new Float32Array/);
  assert.match(uniformSource, /captureTexture\("projection", this\.adaptiveFaceVelocityCutover \? undefined : this\.velocityA\)/);
  assert.match(uniformSource, /if\(!options\.deferPipelineCompilation && !this\.adaptiveFaceVelocityCutover\)this\.createPipelinesSync\(\)/);
  assert.match(uniformSource, /if \(!this\.adaptiveFaceVelocityCutover\) \{\n\s+const cached=uniformPipelineCache/);
});
