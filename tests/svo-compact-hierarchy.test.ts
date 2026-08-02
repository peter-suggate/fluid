import assert from "node:assert/strict";
import test from "node:test";

import { planSparseBrickOctree, packSparseBrickPlan, SPARSE_BRICK_INVALID_INDEX } from "../lib/sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE } from "../lib/svo-brick-occupancy";
import {
  decodeSvoCompactNode,
  packSvoCompactHierarchy,
  SVO_COMPACT_NODE_STRIDE_BYTES,
  traverseCompactSvo,
} from "../lib/svo-compact-hierarchy";
import { traversePackedSvo, type SvoRay, type SvoWorldMapping } from "../lib/webgpu-svo-traversal";
import { resolveWebGpuSvoCompactHierarchy } from "../lib/webgpu-svo-compact-hierarchy";

function nextRandom(state: { value: number }): number {
  let value = state.value >>> 0;
  value ^= value << 13; value ^= value >>> 17; value ^= value << 5;
  state.value = value >>> 0;
  return state.value / 0x1_0000_0000;
}

test("compact render hierarchy packs five words without changing canonical storage", () => {
  const plan = planSparseBrickOctree([{ x: 0, y: 0, z: 0 }, { x: 3, y: 2, z: 1 }], { brickSize: 8, maximumDepth: 2 });
  const canonical = packSparseBrickPlan(plan, 17);
  const original = canonical.nodes.slice();
  canonical.nodes[7] = 0xfedcba98;
  const compact = packSvoCompactHierarchy({ nodes: canonical.nodes, leaves: canonical.leaves,
    publishedNodeCount: plan.nodes.length, publishedLeafCount: plan.leaves.length }, 17);

  assert.equal(SVO_COMPACT_NODE_STRIDE_BYTES, 20);
  assert.equal(compact.residentBytes, plan.nodes.length * 20);
  assert.equal(compact.canonicalNodeBytes, plan.nodes.length * 32);
  assert.equal(compact.residentBytes / compact.canonicalNodeBytes, 0.625);
  assert.equal(compact.sourceGeneration, 17);
  assert.equal(canonical.nodes[7], 0xfedcba98, "source remains owned by the canonical publication");
  assert.deepEqual(canonical.nodes.subarray(8), original.subarray(8));

  for (const node of plan.nodes) {
    const decoded = decodeSvoCompactNode(compact.nodes, node.index);
    assert.equal(decoded.level, node.level);
    assert.equal(decoded.childMask, node.childMask);
    assert.equal(decoded.terminal, node.leafIndex !== SPARSE_BRICK_INVALID_INDEX);
    assert.equal(decoded.linkOrLeaf, decoded.terminal ? node.leafIndex : node.firstChild);
    assert.equal(decoded.flags, node.index === 0
      ? 0xfedcba98
      : node.leafIndex !== SPARSE_BRICK_INVALID_INDEX ? SVO_BRICK_LIFECYCLE.activeBit : 0);
  }
});

test("compact packing rejects overflow, partial records, and inconsistent child masks", () => {
  assert.throws(() => packSvoCompactHierarchy({ nodes: new Uint32Array(7), leaves: new Uint32Array() }), /partial/);
  const invalid = new Uint32Array(8);
  invalid[2] = 0; invalid[3] = 3; invalid[4] = 1; invalid[5] = 1; invalid[6] = SPARSE_BRICK_INVALID_INDEX;
  assert.throws(() => packSvoCompactHierarchy({ nodes: invalid, leaves: new Uint32Array() }), /child count/);
  assert.throws(() => packSvoCompactHierarchy({ nodes: new Uint32Array(), leaves: new Uint32Array(), overflowFlags: 1 }), /overflowed/);
});

test("GPU compact capability fails closed on absent, malformed, or stale publication", () => {
  const buffer = {} as GPUBuffer;
  const ready = { nodes: { buffer, size: 160 }, nodeCount: 10, leafCount: 7,
    sourceGeneration: 3, strideBytes: 16 as const, residentBytes: 160 };
  assert.equal(resolveWebGpuSvoCompactHierarchy(undefined,
    { nodeCount: 10, leafCount: 7, sourceGeneration: 3 }).status, "absent");
  assert.equal(resolveWebGpuSvoCompactHierarchy({ ...ready, residentBytes: 159 },
    { nodeCount: 10, leafCount: 7, sourceGeneration: 3 }).status, "invalid");
  assert.equal(resolveWebGpuSvoCompactHierarchy(ready,
    { nodeCount: 11, leafCount: 7, sourceGeneration: 3 }).status, "stale");
  assert.equal(resolveWebGpuSvoCompactHierarchy(ready,
    { nodeCount: 10, leafCount: 7, sourceGeneration: 4 }).status, "stale");
  assert.equal(resolveWebGpuSvoCompactHierarchy(ready,
    { nodeCount: 10, leafCount: 7, sourceGeneration: 3 }).status, "ready");
});

test("random rays preserve canonical hit, interval, and visit behavior", () => {
  const random = { value: 0xc011a57 };
  const coordinates = new Map<string, { x: number; y: number; z: number }>();
  while (coordinates.size < 220) {
    const coordinate = { x: Math.floor(nextRandom(random) * 32), y: Math.floor(nextRandom(random) * 32), z: Math.floor(nextRandom(random) * 32) };
    coordinates.set(`${coordinate.x},${coordinate.y},${coordinate.z}`, coordinate);
  }
  const plan = planSparseBrickOctree([...coordinates.values()], { brickSize: 8, maximumDepth: 5 });
  const packed = packSparseBrickPlan(plan, 9);
  const view = { nodes: packed.nodes, leaves: packed.leaves,
    publishedNodeCount: plan.nodes.length, publishedLeafCount: plan.leaves.length };
  const compact = packSvoCompactHierarchy(view, 9);
  const mapping: SvoWorldMapping = { origin: [-4, 2, 8], cellSize: [0.05, 0.08, 0.06], brickSize: 8, maximumDepth: 5 };

  for (let sample = 0; sample < 4_096; sample += 1) {
    const ray: SvoRay = {
      origin: [-10 + nextRandom(random) * 40, -8 + nextRandom(random) * 40, -10 + nextRandom(random) * 45],
      direction: [nextRandom(random) * 2 - 1, nextRandom(random) * 2 - 1, nextRandom(random) * 2 - 1],
      tMin: nextRandom(random) * 0.02,
      tMax: 120,
    };
    if (ray.direction.every((component) => Math.abs(component) < 1e-8)) ray.direction = [1, 0, 0];
    const canonical = traversePackedSvo(ray, view, mapping);
    const derived = traverseCompactSvo(ray, compact, packed.leaves, mapping);
    assert.equal(derived.status, canonical.status, `status differs for randomized ray ${sample}`);
    assert.equal(derived.visits, canonical.visits, `visit count differs for randomized ray ${sample}`);
    if (canonical.status === "hit" && derived.status === "hit") {
      assert.equal(derived.hit.nodeIndex, canonical.hit.nodeIndex);
      assert.equal(derived.hit.leafIndex, canonical.hit.leafIndex);
      assert.equal(derived.hit.voxelOffset, canonical.hit.voxelOffset);
      assert.equal(derived.hit.level, canonical.hit.level);
      assert.deepEqual(derived.hit.coordinate, canonical.hit.coordinate);
      assert.equal(derived.hit.tEnter, canonical.hit.tEnter);
      assert.equal(derived.hit.tExit, canonical.hit.tExit);
      assert.deepEqual(derived.hit.bounds, canonical.hit.bounds);
    }
  }
});
