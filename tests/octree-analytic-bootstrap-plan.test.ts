import assert from "node:assert/strict";
import test from "node:test";

import {
  OCTREE_ANALYTIC_BOOTSTRAP_GRADING_HALO_TILES,
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
  surfaceDetailStrength: 0,
};

test("tank-fill bootstrap is a bounded slab plus the exact grading tile ring", () => {
  const plan = planOctreeAnalyticBootstrap(tank);
  assert.deepEqual(plan.tileDimensions, [4, 3, 3]);
  assert.equal(plan.tileCapacity, 36);
  assert.equal(plan.interfaceSupportCells, 4);
  assert.equal(plan.interfaceSupportWorld, 0.08);
  assert.equal(plan.gradingHaloTiles, OCTREE_ANALYTIC_BOOTSTRAP_GRADING_HALO_TILES);
  assert.deepEqual(plan.activeTileLimits, { minimum: [0, 0, 0], maximumExclusive: [4, 2, 3] });
  assert.equal(plan.activeTileCount, 24);
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
  assert.equal(plan.liquidTileCount, 72);
  assert.ok(plan.interfaceTileCount < plan.activeTileCount);
  assert.ok(plan.activeTileCount < plan.tileCapacity / 2);
  for (const index of plan.liquidTileIndices) assert.ok(plan.activeTileIndices.includes(index));
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

test("surface-detail support widens before the independent grading dilation", () => {
  const baseline = planOctreeAnalyticBootstrap({ ...tank, dimensions: [128, 128, 128], containerSize: [128, 128, 128], fillFraction: 0.3 });
  const detailed = planOctreeAnalyticBootstrap({
    ...tank, dimensions: [128, 128, 128], containerSize: [128, 128, 128], fillFraction: 0.3, surfaceDetailStrength: 1,
  });
  assert.equal(baseline.interfaceSupportCells, 4);
  assert.equal(detailed.interfaceSupportCells, 12);
  assert.ok(detailed.interfaceTileCount > baseline.interfaceTileCount);
  assert.ok(detailed.activeTileCount > baseline.activeTileCount);
});

test("constant-time bounds exactly match the enumerated oracle across analytic configurations", () => {
  const grids = [
    { dimensions: [60, 45, 40], containerSize: [1.2, 0.9, 0.8] },
    { dimensions: [73, 31, 19], containerSize: [7.3, 6.2, 7.6] },
  ] as const;
  for (const grid of grids) for (const initialCondition of ["dam-break", "tank-fill"] as const) {
    for (const fillFraction of [0, 0.1, 0.22, 0.5, 1]) for (const interfaceBandCells of [0, 1, 4, 12]) {
      for (const tileSizeCells of [8, 16, 32]) for (const surfaceDetailStrength of [0, 0.35, 1]) {
        const input = { ...grid, initialCondition, fillFraction, interfaceBandCells, tileSizeCells, surfaceDetailStrength };
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

test("planner rejects values that cannot form a bounded WebGPU worklist", () => {
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, dimensions: [0, 45, 40] }), /dimensions/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, containerSize: [1.2, Number.NaN, 0.8] }), /container size/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, tileSizeCells: 12 }), /power of two/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, fillFraction: 1.1 }), /fill fraction/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, interfaceBandCells: -1 }), /interface band/);
  assert.throws(() => planOctreeAnalyticBootstrapBounds({ ...tank, containerSize: [Number.MAX_VALUE, 0.9, 0.8], interfaceBandCells: Number.MAX_VALUE }), /support/);
  assert.throws(() => planOctreeAnalyticBootstrap({ ...tank, surfaceDetailStrength: 2 }), /surface detail/);
});
