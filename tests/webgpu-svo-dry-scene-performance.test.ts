import assert from "node:assert/strict";
import test from "node:test";

import {
  sparseVoxelDrySceneCullingMode,
  SVO_DRY_SCENE_DIRECT_PRIMITIVE_LIMIT,
} from "../lib/webgpu-svo-dry-scene";

test("small authored catalogs avoid repeated root traversal", () => {
  assert.equal(SVO_DRY_SCENE_DIRECT_PRIMITIVE_LIMIT, 64);
  assert.equal(sparseVoxelDrySceneCullingMode(0), "direct-small-catalog");
  assert.equal(sparseVoxelDrySceneCullingMode(64), "direct-small-catalog");
  assert.equal(sparseVoxelDrySceneCullingMode(65), "svo-payload-dda");
});

test("dry-scene culling mode rejects lossy primitive counts", () => {
  for (const invalid of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(() => sparseVoxelDrySceneCullingMode(invalid), /non-negative integer/);
  }
});
