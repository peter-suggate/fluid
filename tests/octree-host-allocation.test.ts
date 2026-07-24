import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { planUniformHostAllocation } from "../lib/octree-host-allocation";

const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const allocationSource = readFileSync(new URL("../lib/octree-host-allocation.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("uniform host allocation reports only the fields it actually owns", () => {
  const plan = planUniformHostAllocation(7, 5, 3, "semi-lagrangian");
  assert.deepEqual(plan.velocityExtent, [7, 5, 3]);
  assert.deepEqual(plan.transportExtent, [9, 7, 5]);
  assert.deepEqual(plan.fluxExtent, [7, 5, 3]);
  assert.deepEqual(plan.pressureExtent, [7, 5, 3]);
  assert.deepEqual(plan.volumeExtent, [7, 5, 3]);
  assert.equal(plan.velocityBytes, 7 * 5 * 3 * (2 * 16 + 8) + 9 * 7 * 5 * 8);
  assert.equal(plan.scalarBytes, 7 * 5 * 3 * 4 * 4);
  assert.equal(plan.conditioningBytes, 7 * 5 * 3 * 4);
  assert.equal(plan.allocatedBytes,
    plan.velocityBytes + plan.scalarBytes + plan.conditioningBytes);
});

test("MacCormack allocation retains its two predictor/corrector pairs", () => {
  const plan = planUniformHostAllocation(80, 160, 60, "maccormack");
  assert.equal(plan.velocityBytes,
    80 * 160 * 60 * (4 * 16 + 8)
      + 82 * 162 * 62 * 2 * 8);
});

test("uniform host allocation rejects invalid simulation extents", () => {
  assert.throws(() => planUniformHostAllocation(0, 4, 4, "maccormack"), RangeError);
});

test("octree has no dense allocation switch, compatibility telemetry, or shared shader graph", () => {
  for (const retired of [
    "adaptiveFaceVelocityCutover",
    "planOctreeHostAllocation",
    "OctreeHostAllocationPlan",
    "denseBaselineBytes",
    "savedBytes",
    "velocitySavedBytes",
    "scalarSavedBytes",
  ]) {
    assert.doesNotMatch(`${uniformSource}\n${allocationSource}`, new RegExp(retired));
  }
  assert.match(uniformSource,
    /this\.hostAllocation = options\.octree\s*\?\s*undefined\s*:\s*planUniformHostAllocation/);
  assert.match(uniformSource,
    /if \(this\.hostAllocation\) \{\s*this\.shaderModule = device\.createShaderModule/);
  assert.match(uniformSource,
    /if \(this\.hostAllocation\) \{\s*const surfaceAuthority/);
  assert.match(uniformSource,
    /if \(this\.hostAllocation\) \{\s*this\.reductionBuffer = device\.createBuffer/,
    "the dense diagnostic reduction is allocated only for a dense host");
  assert.doesNotMatch(uniformSource,
    /this\.reductionBuffer = device\.createBuffer[\s\S]{0,200}\n\s*if \(this\.hostAllocation\)/,
    "octree construction must not allocate a dense diagnostic placeholder");
  assert.doesNotMatch(uniformSource,
    /octreeHostAllocation|compatibility VOF texture|compact-face host scalars/);
  assert.doesNotMatch(uniformSource,
    /velocityExtent\s*=\s*this\.hostAllocation\?\.velocityExtent\s*\?\?|\[1,\s*1,\s*1\].*rgba32float|Octree renderer column binding/,
    "compact octree construction must not allocate format-only velocity or column textures");
  const octreeConstruction = uniformSource.slice(
    uniformSource.indexOf("new WebGPUOctreeProjection"),
    uniformSource.indexOf("this.applyOctreeInfo"),
  );
  assert.doesNotMatch(octreeConstruction, /velocityIn|velocityOut|columnBase/);
  const octreeResources = octreeSource.slice(
    octreeSource.indexOf("interface OctreeProjectionResources"),
    octreeSource.indexOf("function octreeLeafSize"),
  );
  assert.doesNotMatch(`${octreeResources}\n${octreeSource}`,
    /velocityIn|velocityOut|@binding\(0\) var velocityIn|@binding\(1\) var velocityOut|constrainedFaceVelocity/,
    "the octree projection graph must expose only native compact velocity authority");
});
