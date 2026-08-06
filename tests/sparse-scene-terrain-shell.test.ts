import assert from "node:assert/strict";
import test from "node:test";
import {
  adaptiveSparseBrickLeafBounds,
  adaptiveSparseBrickLeafContains,
  planAdaptiveSparseBrickOctree,
} from "../lib/adaptive-sparse-brick-plan";
import type { SparseBrickCoordinate } from "../lib/sparse-brick-octree";
import {
  reduceSvoNodeMipChildren,
  type SvoNodeMipRgba8,
} from "../lib/svo-node-mip-pyramid";
import { svoNodeMipCoverageOpacity } from "../lib/svo-node-mip-sampling";
import {
  sparseSceneTerrainBrickCoordinates,
  sparseSceneTerrainBrickCoverage,
  sparseSceneTerrainNodeCoverage,
  type SparseSceneTerrainDomain,
  type SparseSceneTerrainField,
} from "../lib/sparse-scene-terrain-field";

/**
 * Ground on a 64^3 lattice of 10 mm cells, so a finest brick is 80 mm and the
 * tree is four levels deep. Small enough to enumerate exhaustively and still
 * deep enough that a coarse leaf spans several finest bricks, which is the only
 * configuration any of this is about.
 */
const CELL_M = 0.01;
const CELLS = 64;
const BRICK_SIZE = 8;
const MAXIMUM_DEPTH = 3; // 64 cells / 8 per brick = 8 bricks = 2^3

const DOMAIN: SparseSceneTerrainDomain = Object.freeze({
  worldOrigin_m: [0, 0, 0] as const,
  cellSize_m: [CELL_M, CELL_M, CELL_M] as const,
  dimensionsCells: [CELLS, CELLS, CELLS] as const,
});

function terrainField(height: (cellX: number, cellZ: number) => number): SparseSceneTerrainField {
  const heights_m = new Float32Array(CELLS * CELLS);
  for (let z = 0; z < CELLS; z += 1) for (let x = 0; x < CELLS; x += 1) heights_m[x + CELLS * z] = height(x, z);
  let minimumHeight_m = Infinity, maximumHeight_m = -Infinity;
  for (const value of heights_m) {
    if (value < minimumHeight_m) minimumHeight_m = value;
    if (value > maximumHeight_m) maximumHeight_m = value;
  }
  return {
    dimensions: [CELLS, CELLS],
    heights_m,
    minimumHeight_m,
    maximumHeight_m,
    bounds: { minimum: [0, 0, 0], maximum: [CELLS * CELL_M, maximumHeight_m, CELLS * CELL_M] },
  };
}

/** Flat at 300 mm: a plane strictly inside a finest brick, not on its boundary. */
const FLAT = terrainField(() => 0.30);
/**
 * A step, with the riser at cell 20 so it lands inside a coarse node rather than
 * on one of its faces. Every level-1 and level-2 node straddling x = 20 covers
 * both heights, which is what makes the classification's two column extremes
 * distinguishable at all.
 */
const STEP = terrainField((cellX) => (cellX < 20 ? 0.30 : 0.10));

const key = (coordinate: SparseBrickCoordinate) => `${coordinate.x},${coordinate.y},${coordinate.z}`;

test("a node is buried only when its ceiling clears the lowest column it covers", () => {
  const finest = (y: number) => sparseSceneTerrainNodeCoverage(
    FLAT, DOMAIN, BRICK_SIZE, MAXIMUM_DEPTH, MAXIMUM_DEPTH, { x: 0, y, z: 0 });
  // A finest brick is 8 cells = 80 mm, so ground at 300 mm falls inside brick 3.
  assert.equal(finest(0), "buried");
  assert.equal(finest(2), "buried", "ceiling 240 mm clears the 300 mm column");
  assert.equal(finest(3), "surface", "240..320 mm straddles the ground plane");
  assert.equal(finest(4), "outside", "floor 320 mm is above every column");
  assert.equal(finest(7), "outside");
});

test("a node under open air beside a ridge is a surface node, not a buried one", () => {
  // Level 2 halves the finest brick count per axis, so this node covers cells
  // 16..31 in x — both sides of the step — and 0..15 in y, i.e. 0..160 mm.
  const straddling = sparseSceneTerrainNodeCoverage(STEP, DOMAIN, BRICK_SIZE, 2, MAXIMUM_DEPTH, { x: 1, y: 0, z: 0 });
  // Its ceiling (160 mm) clears the *tallest* column it covers (300 mm), which
  // is the test a maximum-only classifier would apply — and it would be wrong,
  // because the low half of the step is ground only up to 100 mm and the node's
  // top 60 mm there is open air.
  assert.equal(straddling, "surface");
  // The mirror image: its floor (0 mm) is at or above the *lowest* column
  // (100 mm) nowhere, but a minimum-only "outside" test at the node above it
  // would discard ground that is still there under the high half.
  const above = sparseSceneTerrainNodeCoverage(STEP, DOMAIN, BRICK_SIZE, 2, MAXIMUM_DEPTH, { x: 1, y: 1, z: 0 });
  assert.equal(above, "surface", "160..320 mm still contains the 300 mm side of the step");
  // Away from the riser both answers are unambiguous.
  assert.equal(sparseSceneTerrainNodeCoverage(STEP, DOMAIN, BRICK_SIZE, 2, MAXIMUM_DEPTH, { x: 0, y: 0, z: 0 }), "buried");
  assert.equal(sparseSceneTerrainNodeCoverage(STEP, DOMAIN, BRICK_SIZE, 2, MAXIMUM_DEPTH, { x: 3, y: 1, z: 0 }), "outside");
});

test("the surface/buried split partitions the claim exactly", () => {
  for (const field of [FLAT, STEP]) {
    const claim = sparseSceneTerrainBrickCoordinates(field, DOMAIN, BRICK_SIZE);
    const { surface, buried } = sparseSceneTerrainBrickCoverage(field, DOMAIN, BRICK_SIZE);
    assert.equal(surface.length + buried.length, claim.length);
    const split = new Set([...surface, ...buried].map(key));
    assert.equal(split.size, claim.length, "surface and buried must be disjoint");
    for (const coordinate of claim) assert.ok(split.has(key(coordinate)), `${key(coordinate)} left the claim`);
    assert.ok(buried.length > 0 && surface.length > 0);
  }
});

test("the surface claim scales with area while the whole claim scales with volume", () => {
  // Four lattices over the same world, each half the cell size of the last. The
  // ratios are what the hero garden's memory table stands or falls on, so they
  // are measured on a heightfield rather than asserted from a formula.
  const ratios: { surface: number[]; total: number[] } = { surface: [], total: [] };
  let previous: { surface: number; total: number } | undefined;
  for (const cells of [16, 32, 64, 128]) {
    const cellSize = (CELLS * CELL_M) / cells;
    const domain: SparseSceneTerrainDomain = {
      worldOrigin_m: [0, 0, 0], cellSize_m: [cellSize, cellSize, cellSize], dimensionsCells: [cells, cells, cells],
    };
    const heights = new Float32Array(cells * cells);
    for (let z = 0; z < cells; z += 1) for (let x = 0; x < cells; x += 1) {
      // A tilted, gently curved ground so the surface is neither axis-aligned
      // nor flat; a plane on a lattice boundary would flatter the shell count.
      heights[x + cells * z] = 0.18 + 0.10 * (x / cells) + 0.04 * Math.sin((6 * z) / cells);
    }
    const field: SparseSceneTerrainField = {
      dimensions: [cells, cells], heights_m: heights,
      minimumHeight_m: Math.min(...heights), maximumHeight_m: Math.max(...heights),
      bounds: { minimum: [0, 0, 0], maximum: [CELLS * CELL_M, 0.32, CELLS * CELL_M] },
    };
    const coverage = sparseSceneTerrainBrickCoverage(field, domain, BRICK_SIZE);
    const counts = { surface: coverage.surface.length, total: coverage.surface.length + coverage.buried.length };
    if (previous) {
      ratios.surface.push(counts.surface / previous.surface);
      ratios.total.push(counts.total / previous.total);
    }
    previous = counts;
  }
  // Halving the cell quadruples an area and octuples a volume. The measured
  // ratios sit inside a band either side of 4 and 8 rather than on them, because
  // a brick is a finite box and the surface is not axis-aligned.
  const finest = { surface: ratios.surface.at(-1)!, total: ratios.total.at(-1)! };
  assert.ok(finest.surface > 3.2 && finest.surface < 4.8,
    `surface growth ${finest.surface} is not quadratic (series ${ratios.surface.join(", ")})`);
  assert.ok(finest.total > 6.5 && finest.total < 8.5,
    `total growth ${finest.total} is not cubic (series ${ratios.total.join(", ")})`);
  // And the gap between them is the whole point: the shell falls behind the
  // volume by very close to a factor of two on every halving.
  assert.ok(finest.total / finest.surface > 1.6,
    `volume outgrows the shell by only ${finest.total / finest.surface}x per halving`);
});

// ---------------------------------------------------------------------------
// The planner, driven by the same classification
// ---------------------------------------------------------------------------

function groundPlan(field: SparseSceneTerrainField, refine: boolean) {
  const proxyBricks = sparseSceneTerrainBrickCoordinates(field, DOMAIN, BRICK_SIZE);
  return {
    proxyBricks,
    plan: planAdaptiveSparseBrickOctree({
      brickSize: BRICK_SIZE,
      solverBricks: [],
      proxyBricks,
      maximumDepth: MAXIMUM_DEPTH,
      // Two levels of room above the finest, exactly as a dry scene at
      // `environmentRefinementDepth` 2 has: the planner enters at the solver's
      // own level and only descends where it is told to, so this is where the
      // coarsening room comes from at all.
      solverLevel: MAXIMUM_DEPTH - 2,
      maximumEnvironmentCoarseningPower: 0,
      refineEnvironmentLeaf: refine
        ? (level, coordinate) => sparseSceneTerrainNodeCoverage(
          field, DOMAIN, BRICK_SIZE, level, MAXIMUM_DEPTH, coordinate) === "surface"
        : () => true,
    }),
  };
}

test("surface-driven descent leaves a fine shell over a coarse interior, with no gap", () => {
  const { proxyBricks, plan } = groundPlan(STEP, true);
  const levels = new Map<number, number>();
  for (const leaf of plan.leaves) {
    const level = plan.nodes[leaf.nodeIndex].level;
    levels.set(level, (levels.get(level) ?? 0) + 1);
  }
  assert.ok((levels.get(MAXIMUM_DEPTH) ?? 0) > 0, "the shell must reach the finest level");
  assert.ok([...levels.keys()].some((level) => level < MAXIMUM_DEPTH), "the interior must collapse to coarse leaves");
  assert.ok(plan.leaves.length < proxyBricks.length,
    `${plan.leaves.length} leaves should be fewer than the ${proxyBricks.length} bricks claimed`);

  // The invariant the whole design rests on: every finest brick of the ground is
  // inside exactly one leaf. A brick inside none is a region with no mip page,
  // and a missing page samples as zero coverage rather than as unknown
  // (`lib/webgpu-svo-dry-scene.ts:1826` returns `valid = 1u`) — a hole the
  // pyramid lights through. A brick inside two is an ambiguous payload address.
  for (const coordinate of proxyBricks) {
    const owners = plan.leaves.filter((leaf) => adaptiveSparseBrickLeafContains(plan, leaf.index, coordinate));
    assert.equal(owners.length, 1, `${key(coordinate)} is covered by ${owners.length} leaves`);
  }
});

test("surface-driven descent costs strictly fewer leaves than one per claimed brick", () => {
  const driven = groundPlan(STEP, true);
  const volumetric = groundPlan(STEP, false);
  assert.equal(volumetric.plan.leaves.length, volumetric.proxyBricks.length,
    "the unconditional planner must produce one finest leaf per claimed brick");
  assert.ok(driven.plan.leaves.length < volumetric.plan.leaves.length);
});

// ---------------------------------------------------------------------------
// The crux: what a cone reads over a coarse leaf, and over a hollow one
// ---------------------------------------------------------------------------

/**
 * The deepest leaf covering one finest cell, and its own local voxel.
 *
 * A CPU mirror of the two GPU functions the derived builder resolves a mip texel
 * through — `deepestLeaf` and `leafLocal` in
 * `lib/webgpu-svo-live-derived-builder.ts:182,240`. `leafLocal` divides the
 * global cell by the leaf's level scale, which is the single fact that makes a
 * coarse leaf legible to the pyramid at all: without it a coarse leaf would be
 * indexed with a fine coordinate and read another brick's cells.
 */
function resolveCell(
  plan: ReturnType<typeof planAdaptiveSparseBrickOctree>,
  cell: readonly [number, number, number],
): { level: number; origin: [number, number, number]; scale: number } | undefined {
  const brick = cell.map((value) => Math.floor(value / BRICK_SIZE)) as [number, number, number];
  let found: { level: number; origin: [number, number, number]; scale: number } | undefined;
  for (const leaf of plan.leaves) {
    if (adaptiveSparseBrickLeafContains(plan, leaf.index, { x: brick[0], y: brick[1], z: brick[2] })) {
      const node = plan.nodes[leaf.nodeIndex];
      const bounds = adaptiveSparseBrickLeafBounds(plan, leaf.index);
      found = {
        level: node.level,
        origin: [bounds.minimum.x * BRICK_SIZE, bounds.minimum.y * BRICK_SIZE, bounds.minimum.z * BRICK_SIZE],
        scale: 2 ** (plan.maximumDepth - node.level),
      };
      break;
    }
  }
  return found;
}

/**
 * The solid fraction the voxeliser writes into one leaf voxel.
 *
 * Mirrors `lib/webgpu-sparse-scene-proxies.ts:1936-1949`: the cell's extent is
 * the leaf's own scaled cell, and the fraction is taken against the *tallest*
 * column under that extent so a ridge crossing a coarse voxel is not lost to a
 * centre sample. Ground fills every column from the domain floor upward, so a
 * voxel wholly below the ridge reads exactly 1.
 */
function voxelSolidFraction(field: SparseSceneTerrainField, cell: readonly [number, number, number], scale: number): number {
  const voxelCell = cell.map((value) => Math.floor(value / scale) * scale) as [number, number, number];
  let tallest = -Infinity;
  for (let z = voxelCell[2]; z < voxelCell[2] + scale; z += 1) for (let x = voxelCell[0]; x < voxelCell[0] + scale; x += 1) {
    tallest = Math.max(tallest, field.heights_m[Math.min(CELLS - 1, x) + CELLS * Math.min(CELLS - 1, z)]);
  }
  const bottom_m = voxelCell[1] * CELL_M;
  return Math.max(0, Math.min(1, (tallest - bottom_m) / (CELL_M * scale)));
}

/** Level-0 mip coverage byte for one finest cell, or 0 where no leaf owns it. */
function baseCoverage(
  plan: ReturnType<typeof planAdaptiveSparseBrickOctree>,
  field: SparseSceneTerrainField,
  cell: readonly [number, number, number],
): number {
  const resolved = resolveCell(plan, cell);
  if (!resolved) return 0;
  return Math.round(255 * voxelSolidFraction(field, cell, resolved.scale));
}

/** Mean coverage over a 2^level cube of finest cells, reduced the way the GPU does. */
function reducedCoverage(
  plan: ReturnType<typeof planAdaptiveSparseBrickOctree>,
  field: SparseSceneTerrainField,
  origin: readonly [number, number, number],
  level: number,
): number {
  if (level === 0) {
    const value = baseCoverage(plan, field, origin);
    return value;
  }
  const half = 2 ** (level - 1);
  const children: SvoNodeMipRgba8[] = [];
  for (let octant = 0; octant < 8; octant += 1) {
    const child = [
      origin[0] + (octant & 1) * half,
      origin[1] + ((octant >> 1) & 1) * half,
      origin[2] + ((octant >> 2) & 1) * half,
    ] as const;
    const mean = reducedCoverage(plan, field, child, level - 1);
    children.push([mean, mean > 0 ? 255 : 0, 0, 0]);
  }
  return reduceSvoNodeMipChildren(children)[0];
}

test("a coarse leaf over solid ground reports full opacity at every mip level", () => {
  const { plan } = groundPlan(FLAT, true);
  // A column of ground well under the 300 mm surface. Every mip level from the
  // finest cell up to a 16-cell (160 mm) block must read fully solid; anything
  // less is the cone seeing daylight through the floor. Level 5 is deliberately
  // excluded — a 320 mm block crosses the ground plane, so it *should* read
  // 30/32 solid, and it is the level the leak test compares against.
  for (const level of [0, 1, 2, 3, 4]) {
    const coverage = reducedCoverage(plan, FLAT, [0, 0, 0], level);
    assert.equal(coverage, 255, `mip level ${level} read ${coverage}/255 inside solid ground`);
  }
  // And the interior really is being served by coarse leaves rather than by a
  // tower of fine ones — otherwise this test proves nothing about the design.
  const resolved = resolveCell(plan, [4, 4, 4]);
  assert.ok(resolved && resolved.scale > 1, "the deep interior should resolve through a coarse leaf");
});

test("a hollow interior leaks light, and by how much", () => {
  // The counterfactual the interior claim exists to prevent: keep only the
  // bricks the surface passes through and let the rest go unclaimed.
  const { surface } = sparseSceneTerrainBrickCoverage(FLAT, DOMAIN, BRICK_SIZE);
  const hollow = planAdaptiveSparseBrickOctree({
    brickSize: BRICK_SIZE,
    solverBricks: [],
    proxyBricks: surface,
    maximumDepth: MAXIMUM_DEPTH,
    solverLevel: MAXIMUM_DEPTH - 2,
    maximumEnvironmentCoarseningPower: 0,
    refineEnvironmentLeaf: () => true,
  });
  const solid = groundPlan(FLAT, true).plan;

  // Ground at 300 mm with 10 mm cells: cells 0..29 are solid, and the surface
  // shell is the single brick row y = 24..31. Reduce a 32-cell block that
  // contains the whole ground column.
  const hollowCoverage = reducedCoverage(hollow, FLAT, [0, 0, 0], 5);
  const solidCoverage = reducedCoverage(solid, FLAT, [0, 0, 0], 5);
  // 30 of the block's 32 cell layers are ground, so a correctly claimed floor
  // reads 30/32 = 239/255 here, give or take the byte rounding the GPU applies
  // at every level of the reduction. The hollow one reads only its shell.
  assert.ok(solidCoverage >= 239 && solidCoverage <= 241, `claimed floor read ${solidCoverage}/255, expected ~239`);
  assert.ok(hollowCoverage < solidCoverage / 2,
    `a hollow interior should read under half the solid one, read ${hollowCoverage} against ${solidCoverage}`);

  // What that costs a cone. The sampler turns mean coverage into opacity over a
  // step measured in voxels (`svoNodeMipCoverageOpacity`), so a single step
  // through this block transmits what the ground should have stopped. The CPU
  // reference takes the byte and normalizes internally; its WGSL twin takes the
  // already-normalized texture sample.
  const leaked = 1 - svoNodeMipCoverageOpacity(hollowCoverage, 1);
  const stopped = 1 - svoNodeMipCoverageOpacity(solidCoverage, 1);
  assert.ok(stopped < 0.07, `a claimed floor transmits ${(stopped * 100).toFixed(1)} % of the cone`);
  assert.ok(leaked > 4 * stopped,
    `a hollow floor transmits ${(leaked * 100).toFixed(1)} % against a claimed floor's ${(stopped * 100).toFixed(1)} %`);
});
