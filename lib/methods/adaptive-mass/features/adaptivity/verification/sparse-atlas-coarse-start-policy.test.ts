import assert from "node:assert/strict";
import test from "node:test";

import {
  buildSparseAtlasCompositeGrid,
} from "../../../sparse-atlas-composite-projection";
import {
  SPARSE_ATLAS_MAX_PROMOTIONS_PER_EPOCH,
  initializeSparseAtlasResolutionPolicy,
  planSparseAtlasResolution,
} from "../sparse-atlas-resolution-policy";
import {
  coarsenLargeQuiescentComponents,
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  sparseBrickSpan,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../../../sparse-brick-atlas";

const filledBrick = (
  coordinate: readonly [number, number, number],
  brickDimensions: readonly [number, number, number],
  resolution: SparseBrickResolution = 8,
  spanBricks = 1,
): SparseAdaptiveMassBrick => ({
  key: sparseBrickKey(coordinate, brickDimensions), coordinate,
  ...(spanBricks === 1 ? {} : { spanBricks }), resolution,
  density: new Float64Array(resolution ** 3).fill(1),
  gamma: new Float64Array(resolution ** 3).fill(1),
});

const physicalMass = (bricks: readonly SparseAdaptiveMassBrick[]) => bricks.reduce(
  (sum, brick) => sum + brick.density.reduce((value, rho) => value + rho, 0)
    * (8 * sparseBrickSpan(brick) / brick.resolution) ** 3,
  0,
);

test("quiescent components select every dyadic rung without losing mass", () => {
  const brickDimensions = [90, 1, 1] as const;
  const groups = [[0, 1], [2, 2], [5, 9], [15, 65]] as const;
  const bricks = groups.flatMap(([start, count]) => Array.from({ length: count }, (_, x) =>
    filledBrick([start + x, 0, 0], brickDimensions)));
  const atlas = createSparseAdaptiveMassAtlas([720, 8, 8], bricks, 1, 8);
  const coarsened = coarsenLargeQuiescentComponents(atlas, 1);

  assert.equal(physicalMass(coarsened.bricks), physicalMass(atlas.bricks));
  assert.deepEqual(groups.map(([start]) =>
    coarsened.directory.get(sparseBrickKey([start, 0, 0], brickDimensions))!.resolution),
  [8, 4, 2, 1]);
});

test("component coarsening connects macro faces and remains strongly graded", () => {
  const brickDimensions = [3, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([24, 8, 8], [
    filledBrick([0, 0, 0], brickDimensions, 8, 2),
    filledBrick([2, 0, 0], brickDimensions),
  ], 1, 8);
  const coarsened = coarsenLargeQuiescentComponents(atlas, 1);
  const [macro, ordinary] = coarsened.bricks;
  const macroWidth = 8 * sparseBrickSpan(macro!) / macro!.resolution;
  const ordinaryWidth = 8 * sparseBrickSpan(ordinary!) / ordinary!.resolution;

  assert.equal(physicalMass(coarsened.bricks), physicalMass(atlas.bricks));
  assert.ok(Math.max(macroWidth, ordinaryWidth) <= 2 * Math.min(macroWidth, ordinaryWidth));
  assert.equal(macro!.resolution, 2);
  assert.equal(ordinary!.resolution, 2);
});

test("ordinary promotion is bounded while quiet demotion has no residence penalty", () => {
  const brickDimensions = [SPARSE_ATLAS_MAX_PROMOTIONS_PER_EPOCH + 1, 1, 1] as const;
  const bricks = Array.from({ length: brickDimensions[0] }, (_, x) =>
    filledBrick([x, 0, 0], brickDimensions, 1));
  const atlas = createSparseAdaptiveMassAtlas(
    [8 * brickDimensions[0], 8, 8], bricks, 1, 8,
  );
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const initialized = initializeSparseAtlasResolutionPolicy(atlas);
  const hotHistory = new Map([...initialized.history].map(([key, value]) => [key, {
    ...value, hotEpochs: 1, meanDensity: 0,
  }]));
  const promoted = planSparseAtlasResolution(grid,
    new Float64Array(grid.cells.length).fill(1),
    new Float64Array(3 * grid.cells.length),
    { ...initialized, acceptedSteps: 7, history: hotHistory }, 1 / 30);
  assert.equal(promoted.receipt.promotedBrickCount, SPARSE_ATLAS_MAX_PROMOTIONS_PER_EPOCH);
  assert.equal(promoted.receipt.deferredPromotionCount, 1);

  const fineAtlas = createSparseAdaptiveMassAtlas([8, 8, 8], [
    filledBrick([0, 0, 0], [1, 1, 1], 8),
  ], 1, 8);
  const fineGrid = buildSparseAtlasCompositeGrid(fineAtlas);
  const fineState = initializeSparseAtlasResolutionPolicy(fineAtlas);
  const quietHistory = new Map([...fineState.history].map(([key, value]) => [key, {
    ...value, quietEpochs: 7, meanDensity: 0, lastTransitionStep: 7,
  }]));
  const demoted = planSparseAtlasResolution(fineGrid,
    new Float64Array(fineGrid.cells.length),
    new Float64Array(3 * fineGrid.cells.length),
    { ...fineState, acceptedSteps: 7, history: quietHistory }, 1 / 30);
  assert.equal(demoted.targetResolutionByBrick.get(0), 4);
});

test("runtime 2:1 closure uses physical macro width and coarse-first polarity", () => {
  const brickDimensions = [3, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([24, 8, 8], [
    filledBrick([0, 0, 0], brickDimensions, 1, 2),
    filledBrick([2, 0, 0], brickDimensions, 8),
  ], 1, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const initialized = initializeSparseAtlasResolutionPolicy(atlas);
  const quietHistory = new Map([...initialized.history].map(([key, value]) => [key, {
    ...value, quietEpochs: 7, meanDensity: 1,
  }]));
  const decision = planSparseAtlasResolution(grid,
    Float64Array.from(grid.cells, () => 1), new Float64Array(3 * grid.cells.length),
    { ...initialized, acceptedSteps: 7, history: quietHistory }, 1 / 30);

  assert.equal(decision.targetResolutionByBrick.get(atlas.bricks[0]!.key), 1);
  assert.equal(decision.targetResolutionByBrick.get(atlas.bricks[1]!.key), 1,
    "closure must restrict the physically finer ordinary leaf");
});
