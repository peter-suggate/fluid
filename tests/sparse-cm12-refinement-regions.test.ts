import assert from "node:assert/strict";
import test from "node:test";
import type { FluidRefinementRegion } from "../lib/core/model";
import type { RefinementRegionLattice } from "../lib/core/refinement-regions";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { createDeepPowerHydrostaticScene } from "../lib/core/scenes";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickSpan,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  applySparseCM12RefinementRegionResolutionBounds,
  packSparseCM12RefinementRegions,
  sparseCM12RefinementRegionResolutionBoundsForBrick,
  SPARSE_CM12_REFINEMENT_REGION_BYTES,
  SPARSE_CM12_REFINEMENT_REGION_PARAMETER_OFFSET,
} from "../lib/methods/adaptive-mass/sparse-cm12-refinement-regions";

const lattice: RefinementRegionLattice = {
  dimensions: [32, 32, 32],
  cellSize_m: [1, 1, 1],
  origin_m: { x: 0, y: 0, z: 0 },
};

function region(
  id: string,
  minimumCellSize_cells: number,
  maximumCellSize_cells?: number,
  bounds: {
    readonly min_m: { readonly x: number; readonly y: number; readonly z: number };
    readonly max_m: { readonly x: number; readonly y: number; readonly z: number };
  } = {
    min_m: { x: 8, y: 8, z: 8 },
    max_m: { x: 24, y: 24, z: 24 },
  },
): FluidRefinementRegion {
  return {
    id,
    rule: "minimum-cell-size",
    minimumCellSize_cells,
    ...(maximumCellSize_cells === undefined ? {} : { maximumCellSize_cells }),
    min_m: bounds.min_m,
    max_m: bounds.max_m,
  };
}

test("Sparse CM12 packs authored cell-size bounds into its uniform tail", () => {
  const packed = packSparseCM12RefinementRegions([region("bounded", 4, 8)], lattice);
  assert.equal(packed.byteLength, SPARSE_CM12_REFINEMENT_REGION_BYTES);
  assert.equal(SPARSE_CM12_REFINEMENT_REGION_PARAMETER_OFFSET, 448);
  assert.equal(new Uint32Array(packed, 0, 4)[0], 1);
  assert.deepEqual(Array.from(new Float32Array(packed, 16, 8)),
    [8, 8, 8, 4, 24, 24, 24, 8]);
});

test("a fully contained sparse brick is clamped to the authored cell-size interval", () => {
  const packed = packSparseCM12RefinementRegions([region("bounded", 4, 8)], lattice);
  const bounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
    packed, [8, 8, 8], [8, 8, 8], 8);
  assert.deepEqual(bounds, { minimumResolution: 1, maximumResolution: 2 });
  assert.equal(applySparseCM12RefinementRegionResolutionBounds(8, bounds), 2,
    "4-cell minimum caps an 8-cell brick at 2^3 cells");
  assert.equal(applySparseCM12RefinementRegionResolutionBounds(1, bounds), 1,
    "8-cell maximum still permits the coarsest brick rung");
});

test("a sparse brick crossing the box boundary obeys the hard minimum", () => {
  const packed = packSparseCM12RefinementRegions([region("bounded", 4, 8)], lattice);
  const bounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
    packed, [4, 8, 8], [8, 8, 8], 8);
  assert.deepEqual(bounds, { minimumResolution: 1, maximumResolution: 2 });
  assert.equal(applySparseCM12RefinementRegionResolutionBounds(8, bounds), 2);
});

test("a hard minimum-size floor wins over a conflicting maximum-size ceiling", () => {
  const packed = packSparseCM12RefinementRegions([
    region("coarse", 4),
    region("fine", 1, 2),
  ], lattice);
  const bounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
    packed, [8, 8, 8], [8, 8, 8], 8);
  assert.deepEqual(bounds, { minimumResolution: 4, maximumResolution: 2 });
  assert.equal(applySparseCM12RefinementRegionResolutionBounds(8, bounds), 2);
});

test("multiple disjoint regions independently contribute to the cell-size envelope", () => {
  const left = region("left", 4, undefined, {
    min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 16, y: 32, z: 32 },
  });
  const right = region("right", 1, 2, {
    min_m: { x: 16, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 },
  });
  for (const authored of [[left, right], [right, left]]) {
    const packed = packSparseCM12RefinementRegions(authored, lattice);
    assert.deepEqual(sparseCM12RefinementRegionResolutionBoundsForBrick(
      packed, [0, 8, 8], [8, 8, 8], 8,
    ), { minimumResolution: 1, maximumResolution: 2 });
    assert.deepEqual(sparseCM12RefinementRegionResolutionBoundsForBrick(
      packed, [16, 8, 8], [8, 8, 8], 8,
    ), { minimumResolution: 4, maximumResolution: 8 });
  }
});

test("a nested finer request cannot override an enclosing hard minimum", () => {
  const base = createDeepPowerHydrostaticScene();
  const bounded = structuredClone(base);
  bounded.fluid.refinementRegions = [{
    id: "coarse-domain",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 4,
    min_m: { x: -0.5 * base.container.width_m, y: 0,
      z: -0.5 * base.container.depth_m },
    max_m: { x: 0.5 * base.container.width_m, y: base.container.height_m,
      z: 0.5 * base.container.depth_m },
  }, {
    id: "fine-centre",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 1,
    maximumCellSize_cells: 1,
    min_m: { x: -0.4, y: 0.8, z: -0.4 },
    max_m: { x: 0.4, y: 1.6, z: 0.4 },
  }];
  const atlas = initializeSparseBrickAtlasFromScene(bounded, {
    finestDimensions: sceneLatticeDimensions(bounded),
  });
  const innerBricks = atlas.bricks.filter((brick) => {
    const span = sparseBrickSpan(brick);
    const low = brick.coordinate.map((value) => value * atlas.brickFineResolution);
    const high = low.map((value) => value + span * atlas.brickFineResolution);
    return low[0]! >= 24 && low[1]! >= 16 && low[2]! >= 24
      && high[0]! <= 40 && high[1]! <= 32 && high[2]! <= 40;
  });
  assert.ok(innerBricks.length > 0);
  for (const brick of innerBricks) {
    const cellSize = atlas.brickFineResolution * sparseBrickSpan(brick)
      / brick.resolution;
    assert.equal(cellSize, 4,
      `nested region brick ${brick.key} refined through the enclosing minimum`);
  }
});

test("generation-zero 2:1 grading maps outward from a hard regional minimum", () => {
  const base = createDeepPowerHydrostaticScene();
  const bounded = structuredClone(base);
  bounded.fluid.refinementRegions = [{
    id: "left-half-floor",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 4,
    min_m: { x: -0.5 * base.container.width_m, y: 0,
      z: -0.5 * base.container.depth_m },
    max_m: { x: 0, y: base.container.height_m,
      z: 0.5 * base.container.depth_m },
  }];
  const atlas = initializeSparseBrickAtlasFromScene(bounded, {
    finestDimensions: sceneLatticeDimensions(bounded),
    resolutionForBrick: () => 8,
  });
  const byCoordinate = new Map(atlas.bricks.map((brick) =>
    [brick.coordinate.join("/"), brick] as const));
  const cellSize = (brick: (typeof atlas.bricks)[number]) =>
    atlas.brickFineResolution * sparseBrickSpan(brick) / brick.resolution;
  for (const brick of atlas.bricks) {
    const lowX = brick.coordinate[0] * atlas.brickFineResolution;
    const worldX = -0.5 * base.container.width_m
      + lowX * base.container.width_m / atlas.dimensions[0];
    if (worldX < 0) assert.ok(cellSize(brick) >= 4,
      `regional brick ${brick.key} escaped the four-cell minimum`);
    for (const [dx, dy, dz] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const) {
      const neighbor = byCoordinate.get([
        brick.coordinate[0] + dx, brick.coordinate[1] + dy,
        brick.coordinate[2] + dz,
      ].join("/"));
      if (!neighbor) continue;
      const ratio = Math.max(cellSize(brick), cellSize(neighbor))
        / Math.min(cellSize(brick), cellSize(neighbor));
      assert.ok(ratio <= 2,
        `grading did not map outward at ${brick.key}/${neighbor.key}`);
    }
  }
});

test("an authored minimum cell size bounds Sparse CM12 generation zero", () => {
  const base = createDeepPowerHydrostaticScene();
  const bounded = {
    ...base,
    fluid: {
      ...base.fluid,
      refinementRegions: [{
        id: "whole-domain-floor",
        rule: "minimum-cell-size" as const,
        minimumCellSize_cells: 4,
        min_m: { x: -0.5 * base.container.width_m, y: 0,
          z: -0.5 * base.container.depth_m },
        max_m: { x: 0.5 * base.container.width_m, y: base.container.height_m,
          z: 0.5 * base.container.depth_m },
      }],
    },
  };
  const atlas = initializeSparseBrickAtlasFromScene(bounded, {
    finestDimensions: sceneLatticeDimensions(bounded),
  });
  for (const brick of atlas.bricks) {
    const cellSize = atlas.brickFineResolution * sparseBrickSpan(brick)
      / brick.resolution;
    assert.ok(cellSize >= 4,
      `generation-zero brick ${brick.key} used ${cellSize}-cell leaves below the floor`);
  }
});

test("an authored maximum cell size refines Sparse CM12 generation zero", () => {
  const base = createDeepPowerHydrostaticScene();
  const bounded = {
    ...base,
    fluid: {
      ...base.fluid,
      refinementRegions: [{
        id: "whole-domain-ceiling",
        rule: "minimum-cell-size" as const,
        minimumCellSize_cells: 1,
        maximumCellSize_cells: 2,
        min_m: { x: -0.5 * base.container.width_m, y: 0,
          z: -0.5 * base.container.depth_m },
        max_m: { x: 0.5 * base.container.width_m, y: base.container.height_m,
          z: 0.5 * base.container.depth_m },
      }],
    },
  };
  const atlas = initializeSparseBrickAtlasFromScene(bounded, {
    finestDimensions: sceneLatticeDimensions(bounded),
  });
  for (const brick of atlas.bricks) {
    const cellSize = atlas.brickFineResolution * sparseBrickSpan(brick)
      / brick.resolution;
    assert.ok(cellSize <= 2,
      `generation-zero brick ${brick.key} used ${cellSize}-cell leaves above the ceiling`);
  }
});
