import assert from "node:assert/strict";
import test from "node:test";

import { packSparseBrickPlan, planSparseBrickOctree } from "../lib/sparse-brick-octree";
import {
  createWebgpuSvoTraversalWGSL,
  enumerateSvoChildIntersectionsParametric,
  intersectSvoRayAabb,
  traversePackedSvo,
  type SvoAabb,
  type SvoChildIntersection,
  type SvoRay,
  type SvoWorldMapping,
} from "../lib/webgpu-svo-traversal";

function generator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function childBounds(parent: SvoAabb, octant: number): SvoAabb {
  const middle = parent.minimum.map((value, axis) => (value + parent.maximum[axis]) * 0.5) as [number, number, number];
  return {
    minimum: middle.map((value, axis) => (octant & (1 << axis)) === 0 ? parent.minimum[axis] : value) as [number, number, number],
    maximum: middle.map((value, axis) => (octant & (1 << axis)) === 0 ? value : parent.maximum[axis]) as [number, number, number],
  };
}

function aabbChildren(ray: SvoRay, parent: SvoAabb, mask: number): SvoChildIntersection[] {
  const hits: SvoChildIntersection[] = [];
  for (let octant = 0; octant < 8; octant += 1) {
    if ((mask & (1 << octant)) === 0) continue;
    const interval = intersectSvoRayAabb(ray, childBounds(parent, octant));
    if (interval) hits.push({ octant, ...interval });
  }
  return hits.sort((left, right) => left.tEnter - right.tEnter || left.tExit - right.tExit || left.octant - right.octant);
}

test("parametric midpoint crossings reproduce randomized child AABB order and intervals", () => {
  const random = generator(0x51_70_2026);
  let parametricCases = 0;
  for (let sample = 0; sample < 20_000; sample += 1) {
    const minimum = [random() * 20 - 10, random() * 20 - 10, random() * 20 - 10] as const;
    const parent: SvoAabb = {
      minimum,
      maximum: [minimum[0] + 0.1 + random() * 20, minimum[1] + 0.1 + random() * 20, minimum[2] + 0.1 + random() * 20],
    };
    const center = parent.minimum.map((value, axis) => (value + parent.maximum[axis]) * 0.5) as [number, number, number];
    const origin = center.map((value) => value + (random() * 4 - 2) * 20) as [number, number, number];
    const target = parent.minimum.map((value, axis) => value + random() * (parent.maximum[axis] - value)) as [number, number, number];
    const direction = target.map((value, axis) => value - origin[axis]) as [number, number, number];
    const ray: SvoRay = { origin, direction, tMin: random() * 0.05, tMax: 2 + random() * 2 };
    const parentInterval = intersectSvoRayAabb(ray, parent);
    if (!parentInterval) continue;
    const mask = 1 + Math.floor(random() * 255);
    const actual = enumerateSvoChildIntersectionsParametric(ray, parent, parentInterval, mask);
    assert.deepEqual(actual.intersections, aabbChildren(ray, parent, mask), `child differential ${sample}`);
    if (actual.mode === "parametric") parametricCases += 1;
  }
  assert.ok(parametricCases > 10_000, `only ${parametricCases} randomized rays exercised the parametric path`);
});

test("midpoint-plane degeneracies retain closed-box touching-child semantics", () => {
  const parent: SvoAabb = { minimum: [0, 0, 0], maximum: [8, 8, 8] };
  const cases: SvoRay[] = [
    { origin: [-1, 4, 2], direction: [1, 0, 0], tMax: 20 },
    { origin: [-1, -1, 4], direction: [1, 1, 0], tMax: 20 },
    { origin: [-1, -1, -1], direction: [1, 1, 1], tMax: 20 },
    { origin: [4, 4, 4], direction: [1, 0.25, -0.5], tMax: 20 },
  ];
  for (const ray of cases) {
    const parentInterval = intersectSvoRayAabb(ray, parent);
    assert.ok(parentInterval);
    const result = enumerateSvoChildIntersectionsParametric(ray, parent, parentInterval, 0xff);
    assert.equal(result.mode, "aabb-fallback");
    assert.deepEqual(result.intersections, aabbChildren(ray, parent, 0xff));
  }
});

test("parametric full traversal is randomized-result equivalent to canonical AABB traversal", () => {
  const random = generator(0x0c_70_8ee);
  const occupied = new Set<number>();
  while (occupied.size < 180) occupied.add(Math.floor(random() * 8 ** 3));
  const coordinates = [...occupied].map((index) => ({ x: index & 7, y: (index >> 3) & 7, z: (index >> 6) & 7 }));
  const plan = planSparseBrickOctree(coordinates, { brickSize: 8, maximumDepth: 3 });
  const packed = packSparseBrickPlan(plan);
  const topology = { nodes: packed.nodes, leaves: packed.leaves };
  const mapping: SvoWorldMapping = {
    origin: [-13, 7, 19], cellSize: [0.5, 0.75, 1.25], brickSize: 8, maximumDepth: 3,
  };
  const extent = mapping.cellSize.map((size) => size * mapping.brickSize * 2 ** mapping.maximumDepth) as [number, number, number];
  for (let sample = 0; sample < 4_000; sample += 1) {
    const center = mapping.origin.map((value, axis) => value + extent[axis] * 0.5) as [number, number, number];
    const origin = center.map((value, axis) => value + (random() * 2 - 1) * extent[axis] * 1.5) as [number, number, number];
    const target = mapping.origin.map((value, axis) => value + random() * extent[axis]) as [number, number, number];
    const ray: SvoRay = {
      origin,
      direction: target.map((value, axis) => value - origin[axis]) as [number, number, number],
      tMin: random() * 0.025,
      tMax: 1.25 + random(),
    };
    const reference = traversePackedSvo(ray, topology, mapping, { childEnumeration: "aabb" });
    const parametric = traversePackedSvo(ray, topology, mapping, { childEnumeration: "parametric" });
    assert.deepEqual(parametric, reference, `full traversal differential ${sample}`);
  }
});

test("parametric traversal preserves budgets, overflow, and off-ray topology validation", () => {
  const coordinates = Array.from({ length: 8 }, (_, octant) => ({
    x: octant & 1, y: (octant >> 1) & 1, z: (octant >> 2) & 1,
  }));
  const plan = planSparseBrickOctree(coordinates, { brickSize: 4, maximumDepth: 1 });
  const packed = packSparseBrickPlan(plan);
  const mapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 1 };
  const ray: SvoRay = { origin: [-1, 1.1, 1.3], direction: [1, 0.31, 0.27], tMax: 20 };
  for (const options of [
    { maxNodeVisits: 1 },
    { stackCapacity: 1 },
  ] as const) {
    assert.deepEqual(
      traversePackedSvo(ray, { nodes: packed.nodes, leaves: packed.leaves }, mapping, { ...options, childEnumeration: "parametric" }),
      traversePackedSvo(ray, { nodes: packed.nodes, leaves: packed.leaves }, mapping, { ...options, childEnumeration: "aabb" }),
    );
  }
  assert.deepEqual(
    traversePackedSvo(ray, { nodes: packed.nodes, leaves: packed.leaves, overflowFlags: 7 }, mapping,
      { childEnumeration: "parametric" }),
    traversePackedSvo(ray, { nodes: packed.nodes, leaves: packed.leaves, overflowFlags: 7 }, mapping,
      { childEnumeration: "aabb" }),
  );

  const malformed = packed.nodes.slice();
  // Corrupt the highest-octant child, which this ray does not cross. Both paths
  // must still validate the complete sparse child range before descending.
  const firstChild = malformed[4];
  malformed[(firstChild + 7) * 8 + 2] = 9;
  assert.deepEqual(
    traversePackedSvo(ray, { nodes: malformed, leaves: packed.leaves }, mapping, { childEnumeration: "parametric" }),
    traversePackedSvo(ray, { nodes: malformed, leaves: packed.leaves }, mapping, { childEnumeration: "aabb" }),
  );
});

test("parametric WGSL is opt-in and retains an explicit degeneracy fallback", () => {
  const production = createWebgpuSvoTraversalWGSL();
  const parametric = createWebgpuSvoTraversalWGSL({ childEnumeration: "parametric" });
  assert.notEqual(parametric, production);
  assert.match(parametric, /fn svoParametricSegments\(/);
  assert.match(parametric, /parametricSegments\.valid == 0u/);
  // The sparse child range is still bounds-checked before the descent, but as
  // one comparison over the contiguous run rather than a per-octant loop that
  // loaded every child record only to discard it.
  assert.match(parametric, /node\.links\.x \+ countOneBits\(mask\) > mapping\.nodeCount/);
  assert.doesNotMatch(parametric, /validationOctant/);
  assert.doesNotMatch(parametric, /let child = svoNodeLoad\(childIndex\)/);
  assert.match(parametric, /reverseSegment = segmentCount/);
  assert.match(parametric, /svoRayAabbWithInverse\(ray, \(\*continuation\)\.inverseDirection, childBounds\)/,
    "closed-AABB fallback must remain available for degeneracies");
  assert.doesNotMatch(production, /fn svoParametricSegments\(/);
  assert.throws(() => createWebgpuSvoTraversalWGSL({ childEnumeration: "invalid" as never }), /aabb or parametric/);
});
