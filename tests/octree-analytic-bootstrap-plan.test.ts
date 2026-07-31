import assert from "node:assert/strict";
import test from "node:test";

import {
  OCTREE_ANALYTIC_BOOTSTRAP_GRADING_HALO_TILES,
  OCTREE_ANALYTIC_BOOTSTRAP_OWNER_PROBE_HALO_TILES,
  planOctreeAnalyticBootstrap,
  planOctreeAnalyticBootstrapBounds,
  sampleOctreeAnalyticBootstrapPhi,
  type OctreeAnalyticBootstrapInput,
} from "../lib/octree-analytic-bootstrap";

const tank: OctreeAnalyticBootstrapInput = {
  dimensions: [60, 45, 40],
  containerSize: [1.2, 0.9, 0.8],
  tileSizeCells: 16,
  initialCondition: "tank-fill",
  fillFraction: 0.22,
  interfaceBandCells: 4,
};

test("tank-fill bootstrap is a bounded slab plus its grading and owner-probe rings", () => {
  const plan = planOctreeAnalyticBootstrap(tank);
  assert.deepEqual(plan.tileDimensions, [4, 3, 3]);
  assert.equal(plan.tileCapacity, 36);
  assert.equal(plan.interfaceSupportCells, 4);
  assert.equal(plan.interfaceSupportWorld, 0.08);
  assert.equal(plan.gradingHaloTiles, OCTREE_ANALYTIC_BOOTSTRAP_GRADING_HALO_TILES);
  assert.deepEqual(plan.activeTileLimits, { minimum: [0, 0, 0], maximumExclusive: [4, 2, 3] });
  assert.equal(plan.activeTileCount, 24);
  assert.deepEqual(plan.ownerPageTileLimits,
    { minimum: [0, 0, 0], maximumExclusive: [4, 3, 3] });
  assert.equal(plan.ownerPageTileCount, 36);
  assert.deepEqual([...plan.activeTileIndices.slice(0, 5)], [0, 1, 2, 3, 4]);
  assert.deepEqual([...plan.activeTileIndices.slice(-4)], [28, 29, 30, 31]);
  assert.equal(plan.liquidTileCount, 12);
  assert.equal(plan.interfaceTileCount, 12);
});

test("dam-break bootstrap includes deep liquid in a compact coarse-tile worklist", () => {
  const input: OctreeAnalyticBootstrapInput = {
    dimensions: [128, 128, 128], containerSize: [128, 128, 128], tileSizeCells: 16,
    initialCondition: "dam-break", fillFraction: 0.1, interfaceBandCells: 2,
  };
  const plan = planOctreeAnalyticBootstrap(input);
  assert.deepEqual(plan.tileDimensions, [8, 8, 8]);
  assert.equal(plan.tileCapacity, 512);
  assert.deepEqual(plan.activeTileLimits, { minimum: [0, 0, 0], maximumExclusive: [4, 8, 4] });
  assert.equal(plan.activeTileCount, 128);
  assert.deepEqual(plan.ownerPageTileLimits,
    { minimum: [0, 0, 0], maximumExclusive: [5, 8, 5] });
  assert.equal(plan.ownerPageTileCount, 200);
  assert.equal(plan.liquidTileCount, 72);
  assert.ok(plan.interfaceTileCount < plan.activeTileCount);
  assert.ok(plan.activeTileCount < plan.tileCapacity / 2);
  for (const index of plan.liquidTileIndices) assert.ok(plan.activeTileIndices.includes(index));
});

test("large mini-dam keeps the final owner-probe tile out of topology work", () => {
  const plan = planOctreeAnalyticBootstrapBounds({
    dimensions: [64, 20, 64], containerSize: [3.2, 1, 3.2], tileSizeCells: 8,
    initialCondition: "dam-break", fillFraction: 23 / 64, interfaceBandCells: 3,
  });
  assert.equal(OCTREE_ANALYTIC_BOOTSTRAP_OWNER_PROBE_HALO_TILES, 1);
  assert.deepEqual(plan.tileDimensions, [8, 3, 8]);
  assert.deepEqual(plan.activeTileLimits, { minimum: [0, 0, 0], maximumExclusive: [7, 3, 7] });
  assert.equal(plan.activeTileCount, 147);
  assert.deepEqual(plan.ownerPageTileLimits,
    { minimum: [0, 0, 0], maximumExclusive: [8, 3, 8] });
  assert.equal(plan.ownerPageTileCount, 192);
});

test("large scene's authored mini reservoir uses leaf 32 and band 1", () => {
  const input: OctreeAnalyticBootstrapInput = {
    dimensions: [64, 20, 64], containerSize: [3.2, 1, 3.2], tileSizeCells: 32,
    initialCondition: "dam-break", fillFraction: (0.22 * 0.8 ** 3) / (3.2 * 1 * 3.2),
    damBreakDimensions: [0.5, 0.736, 0.5], interfaceBandCells: 1,
  };
  const compact = planOctreeAnalyticBootstrapBounds(input);
  const oracle = planOctreeAnalyticBootstrap(input);
  assert.deepEqual(compact.damBreak, { width: 0.15625, height: 0.736, depth: 0.15625 });
  assert.deepEqual(compact.tileDimensions, [2, 1, 2]);
  assert.deepEqual(compact.activeTileLimits,
    { minimum: [0, 0, 0], maximumExclusive: [2, 1, 2] });
  assert.equal(compact.activeTileCount, 4);
  assert.deepEqual(compact.ownerPageTileLimits,
    { minimum: [0, 0, 0], maximumExclusive: [2, 1, 2] });
  assert.equal(compact.ownerPageTileCount, 4);
  assert.deepEqual(compact.activeTileLimits, oracle.activeTileLimits);
  assert.ok(sampleOctreeAnalyticBootstrapPhi(input, [-1.55, 0.2, -1.55]) < 0);
  assert.ok(sampleOctreeAnalyticBootstrapPhi(input, [-1, 0.2, -1.55]) > 0);
  assert.ok(sampleOctreeAnalyticBootstrapPhi(input, [-1.55, 0.8, -1.55]) > 0);
});

test("every missing tile is analytically proven non-negative while deep liquid keeps negative authority", () => {
  const input: OctreeAnalyticBootstrapInput = {
    dimensions: [128, 128, 128], containerSize: [128, 128, 128], tileSizeCells: 16,
    initialCondition: "dam-break", fillFraction: 0.1, interfaceBandCells: 2,
  };
  const plan = planOctreeAnalyticBootstrap(input);
  assert.deepEqual(plan.outsideWorklist, {
    sign: "non-negative-air",
    bootstrapAuthority: "analytic-sdf",
    publishedCoarseAuthority: "positive-air",
  });
  assert.ok(sampleOctreeAnalyticBootstrapPhi(input, [-50, 20, -50]) < 0, "deep reservoir sign must remain negative");
  assert.ok(sampleOctreeAnalyticBootstrapPhi(input, [-63.5, 20, -63.5]) < -1,
    "closed-wall contact must extend liquid phi instead of inventing a free surface");
  const active = new Set(plan.activeTileIndices);
  for (let z = 0; z < plan.tileDimensions[2]; z += 1) for (let y = 0; y < plan.tileDimensions[1]; y += 1) {
    for (let x = 0; x < plan.tileDimensions[0]; x += 1) {
      const index = x + plan.tileDimensions[0] * (y + plan.tileDimensions[1] * z);
      if (active.has(index)) continue;
      // Check every tile corner. A box SDF is non-negative throughout a tile
      // once all eight corners lie beyond at least one upper reservoir face.
      for (const dz of [0, 1]) for (const dy of [0, 1]) for (const dx of [0, 1]) {
        const point = [-64 + (x + dx) * 16, (y + dy) * 16, -64 + (z + dz) * 16] as const;
        assert.ok(sampleOctreeAnalyticBootstrapPhi(input, point) >= 0, `missing tile ${x},${y},${z} contains negative phi`);
      }
    }
  }
});

test("constant-time bounds exactly match the enumerated oracle across analytic configurations", () => {
  const grids = [
    { dimensions: [60, 45, 40], containerSize: [1.2, 0.9, 0.8] },
    { dimensions: [73, 31, 19], containerSize: [7.3, 6.2, 7.6] },
  ] as const;
  for (const grid of grids) for (const initialCondition of ["dam-break", "tank-fill"] as const) {
    for (const fillFraction of [0, 0.1, 0.22, 0.5, 1]) for (const interfaceBandCells of [0, 1, 4, 12]) {
      for (const tileSizeCells of [8, 16, 32]) {
        const input = { ...grid, initialCondition, fillFraction, interfaceBandCells, tileSizeCells };
        const compact = planOctreeAnalyticBootstrapBounds(input);
        const oracle = planOctreeAnalyticBootstrap(input);
        assert.deepEqual(compact.activeTileLimits, oracle.activeTileLimits);
        assert.equal(compact.activeTileCount, oracle.activeTileCount);
        assert.deepEqual(compact.tileDimensions, oracle.tileDimensions);
        assert.equal(compact.interfaceSupportWorld, oracle.interfaceSupportWorld);
        assert.equal(oracle.activeTileIndices.length, compact.activeTileCount);
      }
    }
  }
  assert.equal("activeTileIndices" in planOctreeAnalyticBootstrapBounds(tank), false,
    "production bounds must not allocate or expose an enumerated tile array");
});

test("anisotropic support uses the largest physical cell dimension", () => {
  const compact = planOctreeAnalyticBootstrapBounds({
    dimensions: [40, 20, 10], containerSize: [4, 4, 4], tileSizeCells: 8,
    initialCondition: "dam-break", fillFraction: 0.1, interfaceBandCells: 3,
  });
  assert.deepEqual(compact.cellSize, [0.1, 0.2, 0.4]);
  assert.ok(Math.abs(compact.interfaceSupportWorld - 1.2) < 1e-12);
});

test("zero-width interfaces remain discoverable and full/empty signs are explicit", () => {
  const emptyDam = planOctreeAnalyticBootstrap({ ...tank, initialCondition: "dam-break", fillFraction: 0, interfaceBandCells: 0 });
  assert.ok(emptyDam.interfaceTileCount > 0, "the degenerate authored boundary remains a bounded interface seed");
  assert.equal(sampleOctreeAnalyticBootstrapPhi({ ...tank, initialCondition: "dam-break", fillFraction: 0 }, [0, 0.2, 0]), Math.hypot(0.6, 0.2, 0.4));
  assert.ok(sampleOctreeAnalyticBootstrapPhi({ ...tank, fillFraction: 1 }, [0, 0.2, 0]) < 0);
  assert.equal(sampleOctreeAnalyticBootstrapPhi({ ...tank, fillFraction: 0 }, [0, 0.2, 0]), 0.2);
});

test("dam phi has only the three exposed faces", () => {
  const input = { ...tank, initialCondition: "dam-break" as const, fillFraction: 0.22 };
  const plan = planOctreeAnalyticBootstrapBounds(input);
  const maximum = [
    -0.5 * input.containerSize[0] + plan.damBreak.width * input.containerSize[0],
    plan.damBreak.height * input.containerSize[1],
    -0.5 * input.containerSize[2] + plan.damBreak.depth * input.containerSize[2],
  ] as const;
  assert.ok(sampleOctreeAnalyticBootstrapPhi(input,
    [-0.5 * input.containerSize[0], 0.1, -0.5 * input.containerSize[2]]) < 0,
  "the three tank-contact planes are closed-wall extensions");
  for (let axis = 0; axis < 3; axis += 1) {
    const point = [...maximum] as [number, number, number];
    point[axis] += 0.01;
    assert.ok(sampleOctreeAnalyticBootstrapPhi(input, point) > 0,
      `exposed dam face ${axis} must retain positive air`);
  }
});

test("planner rejects values that cannot form a bounded WebGPU worklist", () => {
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, dimensions: [0, 45, 40] }), /dimensions/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, containerSize: [1.2, Number.NaN, 0.8] }), /container size/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, tileSizeCells: 12 }), /power of two/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, fillFraction: 1.1 }), /fill fraction/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, damBreakDimensions: [1.3, 0.2, 0.2] }), /dam dimensions/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, damBreakDimensions: [0.2, Number.NaN, 0.2] }), /dam dimensions/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, interfaceBandCells: -1 }), /interface band/);
  assert.throws(() => planOctreeAnalyticBootstrapBounds({ ...tank, containerSize: [Number.MAX_VALUE, 0.9, 0.8], interfaceBandCells: Number.MAX_VALUE }), /support/);
});
