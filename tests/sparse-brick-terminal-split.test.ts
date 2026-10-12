import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/core/model";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/core/voxel-environments";
import { intersectPlanarBoundaryPatch as intersectPlanarBoundary } from "../lib/core/planar-boundary";
import { buildSvoPlanarBoundaryCatalog, svoPlanarResidualEnvironmentPrimitives } from "../lib/svo/features/scene-publication/svo-planar-boundary";

test("the exact stage floor remains finite when sibling terminals reuse its record", () => {
  const primitives = environmentProxyPrimitives(buildEnvironmentProxyCatalog(defaultScene, "stage"), true);
  const floor = primitives.find(primitive => primitive.key.endsWith("stage/floor"))!;
  assert.ok(floor);
  const catalog = buildSvoPlanarBoundaryCatalog(primitives, primitive => ({ materialId: primitive.ownerIndex + 32, ownerId: primitive.ownerIndex + 64 }));
  const index = catalog.patchIndexByOwner.get(floor.ownerIndex)!;
  const patch = catalog.sources[index]!.patch;
  assert.ok(patch);
  assert.ok(!svoPlanarResidualEnvironmentPrimitives(primitives, catalog).includes(floor),
    "the globally visible analytic floor must never be duplicated in sampled edit children");
  const inside = intersectPlanarBoundary(patch, [floor.aabb_m.max.x - .001, 2, 0], [0, -1, 0], 0, 10);
  assert.ok(inside);
  assert.ok(Math.abs(inside.tHit_m - (2 - floor.aabb_m.max.y)) < 1e-9);
  for (const offset of [.001, .05, .4]) {
    assert.equal(intersectPlanarBoundary(patch, [floor.aabb_m.max.x + offset, 2, 0], [0, -1, 0], 0, 10), null,
      "reusing the finite planar record cannot expand it to its child's AABB");
  }
  // A child interval below the slab cannot turn its inherited record into a
  // new boundary on the child's face. The shader uses this same bounded test.
  assert.equal(intersectPlanarBoundary(patch, [0, 2, 0], [0, -1, 0], 3, 4), null);
});
