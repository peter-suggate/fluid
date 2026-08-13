import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { planUniformHostAllocation } from "../lib/uniform-host-allocation";

const octreeHostSource = readFileSync(new URL("../lib/webgpu-octree-eulerian.ts", import.meta.url), "utf8");
const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
const allocationSource = readFileSync(new URL("../lib/uniform-host-allocation.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("uniform host allocation reports only the fields it actually owns", () => {
  const plan = planUniformHostAllocation(7, 5, 3, "semi-lagrangian");
  assert.deepEqual(plan.velocityExtent, [7, 5, 3]);
  assert.deepEqual(plan.transportExtent, [9, 7, 5]);
  assert.deepEqual(plan.fluxExtent, [7, 5, 3]);
  assert.deepEqual(plan.pressureExtent, [7, 5, 3]);
  assert.deepEqual(plan.volumeExtent, [7, 5, 3]);
  assert.equal(plan.boundaryVelocityBytes, (5 * 3 + 7 * 3 + 7 * 5) * 4);
  assert.equal(plan.velocityBytes, 7 * 5 * 3 * 2 * 16
    + 4 * plan.boundaryVelocityBytes + 9 * 7 * 5 * (1 + 6) * 16);
  assert.equal(plan.scalarBytes, 7 * 5 * 3 * 9 * 4);
  assert.equal(plan.conditioningBytes, 7 * 5 * 3 * 3 * 4);
  assert.equal(plan.allocatedBytes,
    plan.velocityBytes + plan.scalarBytes + plan.conditioningBytes);
});

test("MacCormack allocation retains its two predictor/corrector pairs", () => {
  const plan = planUniformHostAllocation(80, 160, 60, "maccormack");
  assert.equal(plan.velocityBytes,
    80 * 160 * 60 * 4 * 16 + 4 * plan.boundaryVelocityBytes
      + 82 * 162 * 62 * (2 + 6) * 16);
});

test("uniform host allocation rejects invalid simulation extents", () => {
  assert.throws(() => planUniformHostAllocation(0, 4, 4, "maccormack"), RangeError);
});

test("adaptive and uniform methods own separate solver graphs", () => {
  for (const retired of [
    "adaptiveFaceVelocityCutover",
    "planOctreeHostAllocation",
    "OctreeHostAllocationPlan",
    "denseBaselineBytes",
    "savedBytes",
    "velocitySavedBytes",
    "scalarSavedBytes",
  ]) {
    assert.doesNotMatch(`${octreeHostSource}\n${allocationSource}`, new RegExp(retired));
  }
  assert.doesNotMatch(octreeHostSource,
    /planUniformHostAllocation|uniformReferenceComputeShader|hostAllocation|velocityA|pressureA|volumeA/,
    "the octree host must not retain the retired dense compatibility solver");
  assert.match(octreeHostSource, /class WebGPUOctreeEulerianSolver/);
  assert.match(octreeHostSource, /new WebGPUOctreeProjection/);
  assert.match(uniformSource, /class WebGPUUniformReferenceSolver/);
  assert.match(uniformSource, /planUniformHostAllocation/);
  assert.match(uniformSource, /uniformReferenceComputeShader/);
  assert.doesNotMatch(uniformSource, /WebGPUOctreeProjection|options\.octree|coarseDynamics/,
    "the uniform reference must not expose an adaptive compatibility seam");
  const octreeResources = octreeSource.slice(
    octreeSource.indexOf("interface OctreeProjectionResources"),
    octreeSource.indexOf("function octreeLeafSize"),
  );
  assert.doesNotMatch(`${octreeHostSource}\n${octreeResources}\n${octreeSource}`,
    /velocityIn|velocityOut|@binding\(0\) var velocityIn|@binding\(1\) var velocityOut|constrainedFaceVelocity/,
    "the octree projection graph must expose only native compact velocity authority");
});
