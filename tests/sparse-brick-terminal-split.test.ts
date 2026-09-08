import assert from "node:assert/strict";
import test from "node:test";
import { packSparseBrickTopologyMutationWorklist, sparseBrickTopologyMutationLeafReserve,
  sparseBrickTopologyMutationNodeReserve, planSparseBrickTopologyLeafReservation } from "../lib/core/webgpu-sparse-brick-topology-mutation";
import { defaultScene } from "../lib/core/model";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/core/voxel-environments";
import { intersectPlanarBoundaryPatch as intersectPlanarBoundary } from "../lib/core/planar-boundary";
import { buildSvoPlanarBoundaryCatalog, svoPlanarResidualEnvironmentPrimitives } from "../lib/svo/features/scene-publication/svo-planar-boundary";

test("terminal growth reserves account for seven preserved siblings per depth", () => {
  assert.equal(sparseBrickTopologyMutationLeafReserve(0, 3), 3);
  assert.equal(sparseBrickTopologyMutationLeafReserve(2, 3), 45);
  assert.equal(sparseBrickTopologyMutationNodeReserve(2, 3), 49);
  for (const [depth, count] of [[-1, 1], [22, 1], [1, -1], [1, Infinity], [21, Number.MAX_SAFE_INTEGER]]) {
    assert.throws(() => sparseBrickTopologyMutationLeafReserve(depth!, count!));
  }
});

test("clustered edit targets share split capacity, including across strokes", () => {
  const planes = new Set(["0:0,0,0"]);
  const first = planSparseBrickTopologyLeafReservation(2, [{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }], planes);
  assert.equal(first.leaves, 14, "adjacent finest targets reuse both ancestor splits");
  const next = planSparseBrickTopologyLeafReservation(2, [{ x: 3, y: 3, z: 3 }], planes, new Set(first.splits));
  assert.equal(next.leaves, 7, "later stroke only splits its one remaining parent");
  const targets = Array.from({ length: 512 }, (_, i) => ({ x: i % 8, y: Math.floor(i / 8) % 8, z: Math.floor(i / 64) }));
  assert.equal(planSparseBrickTopologyLeafReservation(8, targets, planes).leaves, 546,
    "one clustered page uses hundreds of leaves, not512 times the entire tree depth");
  assert.equal(planSparseBrickTopologyLeafReservation(2, [{ x: 0, y: 0, z: 0 }], new Set()).leaves, 1);
});

test("topology worklists deduplicate shared edit pages and reject malformed coordinates before upload", () => {
  const packed = packSparseBrickTopologyMutationWorklist([{ x: 1, y: 2, z: 3 }, { x: 1, y: 2, z: 3 }], 17);
  assert.equal(packed[0], 1); assert.equal(packed[1], 17);
  assert.deepEqual([...packed.slice(8, 12)], [1, 2, 3, 1]);
  const ordered = packSparseBrickTopologyMutationWorklist([{ x: 0, y: 0, z: 1 }, { x: 1, y: 1, z: 0 }, { x: 0, y: 0, z: 0 }], 17);
  assert.deepEqual([...ordered.slice(8)], [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    "Morton order keeps shared split prefixes adjacent for linear GPU preflight");
  assert.throws(() => packSparseBrickTopologyMutationWorklist([{ x: -1, y: 0, z: 0 }], 17));
  assert.throws(() => packSparseBrickTopologyMutationWorklist([{ x: 0, y: 0, z: 0 }], 17, 0));
});

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
