import assert from "node:assert/strict";
import test from "node:test";

import { planAdaptiveSparseBrickOctree } from "../lib/adaptive-sparse-brick-plan";
import { packSparseBrickPlan, planSparseBrickOctree } from "../lib/sparse-brick-octree";
import {
  createWebgpuSvoTraversalWGSL,
  intersectSvoRayAabb,
  svoBrickVoxelIndex,
  traversePackedSvo,
  webgpuSvoTraversalWGSL,
  type SvoPackedTopologyView,
  type SvoWorldMapping,
} from "../lib/webgpu-svo-traversal";

const mapping: SvoWorldMapping = {
  origin: [10, 20, 30],
  cellSize: [0.5, 1, 2],
  brickSize: 4,
  maximumDepth: 2,
};

function packedView(plan: ReturnType<typeof planSparseBrickOctree>): SvoPackedTopologyView {
  const packed = packSparseBrickPlan(plan);
  return {
    nodes: packed.nodes,
    leaves: packed.leaves,
    publishedNodeCount: plan.nodes.length,
    publishedLeafCount: plan.leaves.length,
  };
}

test("ray/AABB intersection handles hits, misses, inside starts, and parallel boundary rays", () => {
  const bounds = { minimum: [1, 2, 3] as const, maximum: [5, 6, 7] as const };
  assert.deepEqual(intersectSvoRayAabb({ origin: [0, 4, 5], direction: [1, 0, 0] }, bounds), { tEnter: 1, tExit: 5 });
  assert.deepEqual(intersectSvoRayAabb({ origin: [2, 4, 5], direction: [1, 0, 0] }, bounds), { tEnter: 0, tExit: 3 });
  assert.deepEqual(intersectSvoRayAabb({ origin: [0, 2, 5], direction: [1, 0, 0] }, bounds), { tEnter: 1, tExit: 5 });
  assert.equal(intersectSvoRayAabb({ origin: [0, 1.99, 5], direction: [1, 0, 0] }, bounds), null);
  assert.equal(intersectSvoRayAabb({ origin: [0, 4, 8], direction: [1, 0, 0] }, bounds), null);
});

test("direct traversal returns fine leaves and anisotropic world bounds", () => {
  const plan = planSparseBrickOctree([{ x: 1, y: 2, z: 3 }], { brickSize: 4, maximumDepth: 2 });
  const result = traversePackedSvo(
    { origin: [0, 30, 58], direction: [1, 0, 0] },
    packedView(plan),
    mapping,
  );
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  assert.equal(result.hit.level, 2);
  assert.deepEqual(result.hit.coordinate, [1, 2, 3]);
  assert.deepEqual(result.hit.bounds, { minimum: [12, 28, 54], maximum: [14, 32, 62] });
  assert.equal(result.hit.tEnter, 12);
  assert.equal(result.hit.tExit, 14);
});

test("direct traversal returns adaptively coarse environment leaves", () => {
  const plan = planAdaptiveSparseBrickOctree({
    brickSize: 4,
    solverBricks: [{ x: 0, y: 0, z: 0 }],
    proxyBricks: [{ x: 2, y: 0, z: 0 }],
    maximumDepth: 2,
    maximumEnvironmentCoarseningPower: 1,
  });
  const packed = packSparseBrickPlan(plan);
  const result = traversePackedSvo(
    { origin: [13, 22, 32], direction: [1, 0, 0] },
    { nodes: packed.nodes, leaves: packed.leaves },
    mapping,
  );
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  assert.equal(result.hit.level, 1);
  assert.deepEqual(result.hit.coordinate, [1, 0, 0]);
  assert.deepEqual(result.hit.bounds, { minimum: [14, 20, 30], maximum: [18, 28, 46] });
});

test("children are visited geometrically near-to-far from either ray direction", () => {
  const plan = planSparseBrickOctree([
    { x: 0, y: 0, z: 0 },
    { x: 1, y: 0, z: 0 },
  ], { brickSize: 4, maximumDepth: 1 });
  const localMapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 1 };
  const topology = packedView(plan);
  const forward = traversePackedSvo({ origin: [-1, 2, 2], direction: [1, 0, 0] }, topology, localMapping);
  const backward = traversePackedSvo({ origin: [9, 2, 2], direction: [-1, 0, 0] }, topology, localMapping);
  assert.equal(forward.status, "hit");
  assert.equal(backward.status, "hit");
  if (forward.status === "hit" && backward.status === "hit") {
    assert.deepEqual(forward.hit.coordinate, [0, 0, 0]);
    assert.deepEqual(backward.hit.coordinate, [1, 0, 0]);
  }
});

test("shared-face boundary traversal has stable ascending-octant tie breaking", () => {
  const plan = planSparseBrickOctree([
    { x: 0, y: 0, z: 0 },
    { x: 0, y: 1, z: 0 },
  ], { brickSize: 4, maximumDepth: 1 });
  const localMapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 1 };
  const result = traversePackedSvo({ origin: [-1, 4, 2], direction: [1, 0, 0] }, packedView(plan), localMapping);
  assert.equal(result.status, "hit");
  if (result.status === "hit") assert.deepEqual(result.hit.coordinate, [0, 0, 0]);
});

test("miss, visit exhaustion, stack overflow, and source overflow are distinct", () => {
  const coordinates = Array.from({ length: 8 }, (_, octant) => ({
    x: octant & 1,
    y: (octant >> 1) & 1,
    z: (octant >> 2) & 1,
  }));
  const plan = planSparseBrickOctree(coordinates, { brickSize: 4, maximumDepth: 1 });
  const topology = packedView(plan);
  const localMapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 1 };
  assert.equal(traversePackedSvo({ origin: [-1, 20, 2], direction: [1, 0, 0] }, topology, localMapping).status, "miss");
  assert.equal(traversePackedSvo(
    { origin: [-1, 4, 4], direction: [1, 0, 0] }, topology, localMapping, { maxNodeVisits: 1 },
  ).status, "work-exhausted");
  assert.equal(traversePackedSvo(
    { origin: [-1, 4, 4], direction: [1, 0, 0] }, topology, localMapping, { stackCapacity: 1 },
  ).status, "stack-overflow");
  assert.deepEqual(traversePackedSvo(
    { origin: [-1, 4, 4], direction: [1, 0, 0] }, { ...topology, overflowFlags: 5 }, localMapping,
  ), { status: "source-overflow", visits: 0, overflowFlags: 5 });
});

test("malformed packed topology fails closed", () => {
  const plan = planSparseBrickOctree([{ x: 0, y: 0, z: 0 }], { brickSize: 4, maximumDepth: 0 });
  const topology = packedView(plan);
  topology.leaves[0] = 99;
  const result = traversePackedSvo({ origin: [-1, 1, 1], direction: [1, 0, 0] }, topology, {
    origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 0,
  });
  assert.equal(result.status, "invalid-topology");
});

test("WGSL helper consumes packed topology directly with bounded traversal", () => {
  assert.match(webgpuSvoTraversalWGSL, /var<storage, read> svoNodes: array<SvoNode>/);
  assert.match(webgpuSvoTraversalWGSL, /var<storage, read> svoLeaves: array<SvoLeaf>/);
  assert.match(webgpuSvoTraversalWGSL, /SVO_STACK_CAPACITY: u32 = 32u/);
  assert.match(webgpuSvoTraversalWGSL, /SVO_MAX_VISITS: u32 = 256u/);
  assert.match(webgpuSvoTraversalWGSL, /svoNodeBounds/);
  assert.match(webgpuSvoTraversalWGSL, /countOneBits\(mask/);
  assert.match(webgpuSvoTraversalWGSL, /fn svoBrickVoxelIndex/);
  assert.doesNotMatch(webgpuSvoTraversalWGSL, /DebugRecord|voxelRecords|brickRecords/);
  assert.match(createWebgpuSvoTraversalWGSL({ group: 2, control: 4, nodes: 5, leaves: 6 }), /@group\(2\) @binding\(5\) var<storage, read> svoNodes/);
  assert.equal(svoBrickVoxelIndex(512, [2, 3, 1], 8), 602);
});
