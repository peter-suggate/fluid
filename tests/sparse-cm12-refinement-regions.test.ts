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
import { createWebgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";

const lattice: RefinementRegionLattice = {
  dimensions: [32, 32, 32],
  cellSize_m: [1, 1, 1],
  origin_m: { x: 0, y: 0, z: 0 },
};

function region(
  id: string,
  minimumCellSize_cells: number,
  maximumCellSize_cells?: number,
): FluidRefinementRegion {
  return {
    id,
    rule: "minimum-cell-size",
    minimumCellSize_cells,
    ...(maximumCellSize_cells === undefined ? {} : { maximumCellSize_cells }),
    min_m: { x: 8, y: 8, z: 8 },
    max_m: { x: 24, y: 24, z: 24 },
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

test("a sparse brick crossing the box boundary keeps its ordinary resolution", () => {
  const packed = packSparseCM12RefinementRegions([region("bounded", 4, 8)], lattice);
  const bounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
    packed, [4, 8, 8], [8, 8, 8], 8);
  assert.deepEqual(bounds, { minimumResolution: 1, maximumResolution: 8 });
  assert.equal(applySparseCM12RefinementRegionResolutionBounds(8, bounds), 8);
});

test("overlapping ceilings conservatively win over conflicting minimum-size floors", () => {
  const packed = packSparseCM12RefinementRegions([
    region("coarse", 4),
    region("fine", 1, 2),
  ], lattice);
  const bounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
    packed, [8, 8, 8], [8, 8, 8], 8);
  assert.deepEqual(bounds, { minimumResolution: 4, maximumResolution: 2 });
  assert.equal(applySparseCM12RefinementRegionResolutionBounds(1, bounds), 4);
});

test("resident WGSL applies bounds to planned and newly activated bricks", () => {
  const source = createWebgpuSparseCM12ResidentWGSL(8);
  assert.match(source, /refinementRegionControl:vec4u/);
  assert.match(source, /fn applySparseCM12RefinementRegionBounds/);
  assert.match(source,
    /requested=applySparseCM12RefinementRegionBounds\(brick,requested\)/);
  assert.match(source,
    /applySparseCM12RefinementRegionBounds\(brick,BRICK_FINE_RESOLUTION\)/);
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
