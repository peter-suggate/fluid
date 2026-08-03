import assert from "node:assert/strict";
import test from "node:test";
import {
  adaptiveSparseBrickLeafBounds,
  adaptiveSparseBrickLeafContains,
  planAdaptiveSparseBrickOctree,
  reportAdaptiveSparseBrickReduction,
} from "../lib/adaptive-sparse-brick-plan";
import {
  SPARSE_BRICK_INVALID_INDEX,
  mortonChild,
  mortonEncode3D,
  packSparseBrickPlan,
  type SparseBrickCoordinate,
  type SparseBrickPlan,
} from "../lib/sparse-brick-octree";

function roomShell(size: number): SparseBrickCoordinate[] {
  const result: SparseBrickCoordinate[] = [];
  for (let z = 0; z < size; z += 1) for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
    if (x === 0 || y === 0 || z === 0 || x === size - 1 || y === size - 1 || z === size - 1) {
      result.push({ x, y, z });
    }
  }
  return result;
}

function assertCovered(plan: SparseBrickPlan, coordinates: readonly SparseBrickCoordinate[]): void {
  for (const coordinate of coordinates) {
    assert.ok(plan.leaves.some((leaf) => adaptiveSparseBrickLeafContains(plan, leaf.index, coordinate)),
      `expected (${coordinate.x},${coordinate.y},${coordinate.z}) to be covered`);
  }
}

function assertNoLeafOverlap(plan: SparseBrickPlan): void {
  const bounds = plan.leaves.map((leaf) => adaptiveSparseBrickLeafBounds(plan, leaf.index));
  for (let a = 0; a < bounds.length; a += 1) for (let b = a + 1; b < bounds.length; b += 1) {
    const overlap = bounds[a].minimum.x < bounds[b].maximum.x && bounds[b].minimum.x < bounds[a].maximum.x
      && bounds[a].minimum.y < bounds[b].maximum.y && bounds[b].minimum.y < bounds[a].maximum.y
      && bounds[a].minimum.z < bounds[b].maximum.z && bounds[b].minimum.z < bounds[a].maximum.z;
    assert.equal(overlap, false, `leaves ${a} and ${b} overlap`);
  }
}

test("a broad distant room shell collapses to coarse environment bricks", () => {
  const proxyBricks = roomShell(64);
  const options = {
    brickSize: 4 as const,
    solverBricks: [],
    proxyBricks,
    maximumDepth: 6,
    maximumEnvironmentCoarseningPower: 3,
  };
  const plan = planAdaptiveSparseBrickOctree(options);
  const report = reportAdaptiveSparseBrickReduction(plan, options);

  assert.equal(proxyBricks.length, 23_816);
  assert.equal(plan.leaves.length, 8 ** 3 - 6 ** 3);
  assert.ok(plan.leaves.every((leaf) => plan.nodes[leaf.nodeIndex].level === 3));
  assert.equal(report.maximumCoarseningPowerUsed, 3);
  assert.ok(report.reductionFraction > 0.98);
  assert.ok(report.compressionRatio > 80);
  assertCovered(plan, proxyBricks);
});

test("a proxy region intersecting a solver brick splits only along that branch", () => {
  const proxyBricks: SparseBrickCoordinate[] = [];
  for (let z = 0; z < 8; z += 1) for (let y = 0; y < 8; y += 1) for (let x = 0; x < 8; x += 1) {
    proxyBricks.push({ x, y, z });
  }
  const solver = { x: 1, y: 1, z: 1 };
  const plan = planAdaptiveSparseBrickOctree({
    brickSize: 8,
    solverBricks: [solver],
    proxyBricks,
    maximumDepth: 3,
    maximumEnvironmentCoarseningPower: 2,
  });
  const levels = new Map<number, number>();
  for (const leaf of plan.leaves) {
    const level = plan.nodes[leaf.nodeIndex].level;
    levels.set(level, (levels.get(level) ?? 0) + 1);
  }

  assert.deepEqual([...levels], [[1, 7], [2, 7], [3, 8]]);
  const solverLeaves = plan.leaves.filter((leaf) => adaptiveSparseBrickLeafContains(plan, leaf.index, solver));
  assert.equal(solverLeaves.length, 1);
  assert.equal(plan.nodes[solverLeaves[0].nodeIndex].level, 3);
  assert.deepEqual(solverLeaves[0].coordinate, solver);
  assertCovered(plan, proxyBricks);
  assertNoLeafOverlap(plan);
});

test("solver leaves remain exact, fine, and canonical under input ordering", () => {
  const solverBricks = [{ x: 7, y: 2, z: 3 }, { x: 1, y: 5, z: 0 }, { x: 7, y: 2, z: 3 }];
  const proxyBricks = [{ x: 12, y: 12, z: 12 }, { x: 13, y: 12, z: 12 }];
  const options = {
    brickSize: 4 as const,
    solverBricks,
    proxyBricks,
    maximumDepth: 4,
    maximumEnvironmentCoarseningPower: 3,
  };
  const plan = planAdaptiveSparseBrickOctree(options);
  const reverse = planAdaptiveSparseBrickOctree({
    ...options,
    solverBricks: [...solverBricks].reverse(),
    proxyBricks: [...proxyBricks].reverse(),
  });
  for (const solver of solverBricks) {
    const exact = plan.leaves.filter((leaf) => leaf.morton === mortonEncode3D(solver.x, solver.y, solver.z)
      && plan.nodes[leaf.nodeIndex].level === plan.maximumDepth);
    assert.equal(exact.length, 1);
    assert.deepEqual(exact[0].coordinate, solver);
  }
  assert.deepEqual(plan, reverse);
  assertNoLeafOverlap(plan);
});

test("mixed-level plans retain the packed pointerless topology ABI", () => {
  const plan = planAdaptiveSparseBrickOctree({
    brickSize: 8,
    solverBricks: [{ x: 1, y: 1, z: 1 }, { x: 6, y: 6, z: 6 }],
    proxyBricks: roomShell(8),
    maximumDepth: 3,
    maximumEnvironmentCoarseningPower: 2,
  });
  const packed = packSparseBrickPlan(plan, 29);
  assert.deepEqual([...packed.counts], [plan.nodes.length, plan.leaves.length, plan.voxelCount, 29, 0, plan.nodes.length * 8]);

  for (const node of plan.nodes) {
    assert.equal(packed.nodes[node.index * 8 + 2], node.level);
    assert.equal(packed.nodes[node.index * 8 + 3], node.childMask);
    assert.equal(packed.nodes[node.index * 8 + 4], node.firstChild);
    assert.equal(packed.nodes[node.index * 8 + 6], node.leafIndex);
    if (node.childCount === 0) continue;
    assert.notEqual(node.firstChild, SPARSE_BRICK_INVALID_INDEX);
    const children = plan.nodes.filter((candidate) => candidate.level === node.level + 1
      && (candidate.morton >> 3n) === node.morton);
    assert.equal(children.length, node.childCount);
    assert.equal(children[0].index, node.firstChild);
    for (const child of children) {
      assert.equal((node.childMask & (1 << Number(child.morton & 7n))) !== 0, true);
      assert.equal(mortonChild(node.morton, Number(child.morton & 7n)), child.morton);
    }
  }
  for (const leaf of plan.leaves) {
    assert.equal(packed.leaves[leaf.index * 4], leaf.nodeIndex);
    assert.equal(packed.leaves[leaf.index * 4 + 1], leaf.voxelOffset);
    assert.equal(plan.nodes[leaf.nodeIndex].childCount, 0);
  }
  assertNoLeafOverlap(plan);
});

test("adaptive planner validates finite tree bounds and coarsening", () => {
  const base = { brickSize: 4 as const, solverBricks: [], proxyBricks: [], maximumDepth: 3 };
  assert.throws(() => planAdaptiveSparseBrickOctree({ ...base, maximumEnvironmentCoarseningPower: 4 }), /coarsening power/);
  assert.throws(() => planAdaptiveSparseBrickOctree({
    ...base,
    proxyBricks: [{ x: 8, y: 0, z: 0 }],
    maximumEnvironmentCoarseningPower: 0,
  }), /\[0, 8\)/);
});

test("environment leaves refine below the solver level exactly where they are told to", () => {
  // The direction the planner could not go until `solverLevel` existed. Before
  // it, environment leaves started at `maximumDepth - coarseningPower` and
  // descended only where a *solver* brick was in the way, so resolution could
  // be given up but never spent: the sole knob was the domain's finest cell
  // size, and it applied everywhere at once.
  const solverBricks: SparseBrickCoordinate[] = [{ x: 0, y: 0, z: 0 }];
  // Two proxy regions at the fine lattice, in different coarse bricks. One is
  // dense enough to want splitting; the other is not.
  // Fine-lattice coordinates: the tree is one level deeper than the solver, so
  // these live in [0, 16) while the solver's brick lives in [0, 8).
  const proxyBricks: SparseBrickCoordinate[] = [];
  for (let x = 2; x < 4; x += 1) for (let y = 0; y < 2; y += 1) for (let z = 0; z < 2; z += 1) {
    proxyBricks.push({ x, y, z });
  }
  proxyBricks.push({ x: 12, y: 12, z: 12 });

  const dense = { x: 1, y: 0, z: 0 };
  const plan = planAdaptiveSparseBrickOctree({
    brickSize: 8,
    solverBricks,
    proxyBricks,
    maximumDepth: 4,
    solverLevel: 3,
    maximumEnvironmentCoarseningPower: 0,
    // Split only the one solver-level brick, so the assertion below is about the
    // mechanism rather than about a cascade.
    refineEnvironmentLeaf: (level, coordinate) => level === 3
      && coordinate.x === dense.x && coordinate.y === dense.y && coordinate.z === dense.z,
  });

  const levelOf = new Map(plan.leaves.map((leaf) => [leaf.index, plan.nodes[leaf.nodeIndex].level]));
  // The solver keeps its own lattice. This is the whole point of the split: the
  // simulation does not move when the renderer resolves geometry more finely.
  const solverLeaves = plan.leaves.filter((leaf) => plan.nodes[leaf.nodeIndex].level === 3
    && plan.nodes[leaf.nodeIndex].morton === mortonEncode3D(0, 0, 0));
  assert.equal(solverLeaves.length, 1, "the solver brick must remain one leaf at its own level");

  const refined = plan.leaves.filter((leaf) => levelOf.get(leaf.index) === 4);
  assert.ok(refined.length > 0, "the dense region must have descended below the solver level");
  for (const leaf of refined) {
    const bounds = adaptiveSparseBrickLeafBounds(plan, leaf.index);
    assert.equal(bounds.minimum.x >= dense.x * 2 && bounds.maximum.x <= (dense.x + 1) * 2, true,
      "only the region the predicate named may refine");
  }
  // The far single brick is not dense, so it stays whole at the solver level.
  assert.ok(plan.leaves.some((leaf) => levelOf.get(leaf.index) === 3
    && adaptiveSparseBrickLeafContains(plan, leaf.index, { x: 12, y: 12, z: 12 })));

  assertCovered(plan, proxyBricks);
  // The invariant that makes a mixed-depth tree traceable at all: no leaf may
  // be an ancestor of another, or traversal terminates on the coarse one and
  // the fine leaf below it is allocated, voxelised, and unreachable.
  assertNoLeafOverlap(plan);
});

test("omitting the solver level reproduces the uniform plan exactly", () => {
  // The invariance that lets this ship dark. Every scene that does not ask for
  // refinement must get byte-identical topology, because owner indices are the
  // GPU material table's own addresses and a reordered leaf set moves them all.
  const solverBricks = [{ x: 1, y: 1, z: 1 }, { x: 2, y: 1, z: 1 }];
  const proxyBricks = roomShell(8);
  const shared = { brickSize: 8 as const, solverBricks, proxyBricks, maximumDepth: 3, maximumEnvironmentCoarseningPower: 1 };
  const before = planAdaptiveSparseBrickOctree(shared);
  const after = planAdaptiveSparseBrickOctree({ ...shared, solverLevel: 3 });
  assert.deepEqual(after, before);
  // And a refinement predicate that never fires changes nothing either.
  assert.deepEqual(planAdaptiveSparseBrickOctree({ ...shared, refineEnvironmentLeaf: () => false }), before);
});
