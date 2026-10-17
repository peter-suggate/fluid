import assert from "node:assert/strict";
import test from "node:test";
import { svoVoxelLightCacheShape, svoVoxelLightCacheSlot,
  SVO_VOXEL_LIGHT_CACHE_MAX_ENTRIES, SVO_VOXEL_LIGHT_CACHE_BYTES_PER_ENTRY } from "./svo-voxel-light-cache";

test("directional visibility storage stays within 64 MiB for million-page worlds", () => {
  for (const pages of [0, 1, 3, 8192, 964810, 1168650, 4_000_000]) {
    const shape = svoVoxelLightCacheShape(pages);
    const entries = shape.reduce((a, b) => a * b, 1);
    assert.ok(entries <= SVO_VOXEL_LIGHT_CACHE_MAX_ENTRIES);
    assert.ok(entries * SVO_VOXEL_LIGHT_CACHE_BYTES_PER_ENTRY <= 64 * 1024 ** 2);
    assert.equal(entries & (entries - 1), 0, "mask addressing requires a power of two");
    assert.ok(shape.every(n => Number.isInteger(n) && n >= 1 && n <= 256));
    assert.equal(entries, Math.min(SVO_VOXEL_LIGHT_CACHE_MAX_ENTRIES, 2 ** Math.ceil(Math.log2(Math.max(1, pages * 512)))));
    const mask = entries - 1;
    for (const voxel of [0, 511, 512, 494000000, 0xfffffffe]) {
      const slot = svoVoxelLightCacheSlot(voxel, mask);
      assert.ok(slot >= 0 && slot < entries);
      const x = slot % shape[0], y = Math.floor(slot / shape[0]) % shape[1];
      const z = Math.floor(slot / (shape[0] * shape[1]));
      assert.ok(z < shape[2]);
      assert.equal(x + shape[0] * (y + shape[1] * z), slot);
    }
  }
  assert.throws(() => svoVoxelLightCacheShape(-1), RangeError);
  assert.throws(() => svoVoxelLightCacheShape(NaN), RangeError);
});
