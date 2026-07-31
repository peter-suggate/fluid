import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createDenseCacheFrozenMini,
  DENSE_CACHE_KERNELS,
  DENSE_CACHE_LAYOUTS,
  DENSE_CACHE_MINI_OCCUPIED_COUNTS,
  DENSE_CACHE_WORKGROUP_SIZES,
  denseCachePhysicalIndex,
  denseCachePhysicalPlan,
} from "../tools/benchmark-dense-hierarchy-cache";

test("dense hierarchy cache benchmark freezes the measured mini census with parent closure", () => {
  const hierarchy = createDenseCacheFrozenMini();
  assert.deepEqual(hierarchy.occupiedCounts, DENSE_CACHE_MINI_OCCUPIED_COUNTS);
  assert.equal(hierarchy.totalLogicalSlots, 4681);
  assert.equal(hierarchy.worklist.length, 1733);
  for (let level = 0; level + 1 < hierarchy.dimensions.length; level += 1) {
    const fine = hierarchy.dimensions[level]!;
    const coarse = hierarchy.dimensions[level + 1]!;
    const parents = new Set(hierarchy.occupiedLocals[level + 1]);
    for (const local of hierarchy.occupiedLocals[level]!) {
      const yz = Math.floor(local / fine[0]);
      const q = [local - yz * fine[0], yz % fine[1], Math.floor(yz / fine[1])] as const;
      const parent = (q[0] >> 1) + coarse[0] * ((q[1] >> 1) + coarse[1] * (q[2] >> 1));
      assert.ok(parents.has(parent), `level ${level} occupied slot ${local} must have an occupied parent`);
    }
  }
});

test("dense cache physical plans separate logical 4-cubed scheduling from physical swizzle", () => {
  const hierarchy = createDenseCacheFrozenMini();
  assert.deepEqual(DENSE_CACHE_LAYOUTS,
    ["xfast-full", "xfast-worklist", "tile4-physical", "tile4-logical", "stage8-logical"]);
  assert.deepEqual(DENSE_CACHE_KERNELS, ["apply", "restrict", "prolong"]);
  assert.deepEqual(DENSE_CACHE_WORKGROUP_SIZES, [32, 64, 128, 256]);
  for (const layout of [
    "xfast-full", "xfast-worklist", "tile4-logical", "stage8-logical",
  ] as const) {
    assert.equal(denseCachePhysicalPlan(hierarchy, layout).totalPhysicalSlots, 4681);
    assert.equal(denseCachePhysicalIndex(
      denseCachePhysicalPlan(hierarchy, layout),
      hierarchy.dimensions[0]!,
      0,
      [4, 0, 0],
    ), 4);
  }
  const tiled = denseCachePhysicalPlan(hierarchy, "tile4-physical");
  assert.equal(tiled.totalPhysicalSlots, 4800);
  const level = 0, dimensions = hierarchy.dimensions[level]!;
  assert.equal(denseCachePhysicalIndex(tiled, dimensions, level, [0, 0, 0]), 0);
  assert.equal(denseCachePhysicalIndex(tiled, dimensions, level, [3, 3, 3]), 63);
  assert.equal(denseCachePhysicalIndex(tiled, dimensions, level, [4, 0, 0]), 64);
});

test("dense cache GPU harness subtracts matched empty batches and enforces coordinate hashes", () => {
  const source = readFileSync(
    new URL("../tools/benchmark-dense-hierarchy-cache.ts", import.meta.url), "utf8",
  );
  assert.match(source,
    /DENSE_CACHE_MINI_OCCUPIED_COUNTS = Object\.freeze\(\[1475, 214, 35, 8, 1\]/);
  assert.match(source,
    /const staged8 = layout === "stage8-logical"[\s\S]*var<workgroup> stageValues:array<f32,600>[\s\S]*vec3u\(8u,8u,4u\)/,
    "8x8x4 is a logical staging policy over the unswizzled x-fast address");
  assert.match(source,
    /const scheduled4 = physicalTiled4 \|\| layout === "tile4-logical"[\s\S]*const physical = physicalTiled4/,
    "logical 4x4x4 transfer scheduling must retain the canonical x-fast physical address");
  assert.match(source,
    /kernelSamples[\s\S]*emptySamples[\s\S]*netMicroseconds: \(gross - overhead\)/,
    "every reported kernel time must subtract its matching empty-dispatch batch");
  assert.match(source,
    /actualWords\[index\] !== expectedWords\[index\][\s\S]*coordinateHash/,
    "every layout and workgroup arm must pass coordinate-keyed bit equality before timing");
  assert.match(source,
    /acquireWebGPUExclusiveLock\("dawn-benchmark", "tools\/benchmark-dense-hierarchy-cache\.ts"\)/);
});
