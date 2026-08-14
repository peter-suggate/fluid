import assert from "node:assert/strict";
import test from "node:test";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
  type SparseAdaptiveMassBrick,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { buildSparseAtlasCompositeGrid, collocateSparseAtlasVelocity } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { initializeSparseAtlasDynamics, stepSparseAtlasDynamics } from
  "../lib/methods/adaptive-mass/sparse-atlas-dynamics";
import { extrapolateSparseAtlasFaceVelocity, transportSparseAtlasCM12 } from
  "../lib/methods/adaptive-mass/sparse-atlas-cm12-transport";

const brick = (
  key: number,
  coordinate: readonly [number, number, number],
  resolution: 4 | 8,
): SparseAdaptiveMassBrick => ({
  key,
  coordinate,
  resolution,
  density: Float64Array.from({ length: resolution ** 3 }, (_, index) => {
    const x = index % resolution;
    const y = Math.floor(index / resolution) % resolution;
    const z = Math.floor(index / resolution ** 2);
    return Math.max(0, 1 - Math.hypot(
      (x + 0.5) / resolution - 0.5,
      (y + 0.5) / resolution - 0.5,
      (z + 0.5) / resolution - 0.5,
    ));
  }),
  gamma: new Float64Array(resolution ** 3).fill(1),
});

test("fixed coarse initialization overrides the usual fine interface floor", () => {
  const atlas = initializeSparseBrickAtlasFromScene(
    createSymmetricExpansionScene(),
    {
      finestDimensions: [32, 16, 32],
      resolutionForBrick: () => 4,
    },
  );
  assert.ok(atlas.bricks.length > 0);
  assert.equal(atlas.bricks.every((candidate) => candidate.resolution === 4), true);
});

test("MAC collocation includes zero-valued domain-wall faces", () => {
  const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [
    { ...brick(0, [0, 0, 0], 4), density: new Float64Array(64).fill(1) },
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const collocated = collocateSparseAtlasVelocity(
    grid, new Float64Array(grid.gradientRows.length).fill(1),
  );
  const negativeBoundary = grid.cells.find((cell) => cell.minimumFine[0] === 0);
  const interior = grid.cells.find((cell) =>
    cell.minimumFine[0] > 0 && cell.maximumFine[0] < 8);
  assert.ok(negativeBoundary && interior);
  assert.equal(collocated[3 * negativeBoundary.id], 0.5);
  assert.equal(collocated[3 * interior.id], 1);
});

test("CM12 characteristic rows conserve mass across an arbitrary 2:1 tiling", () => {
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [
    brick(0, [0, 0, 0], 8),
    brick(1, [1, 0, 0], 4),
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const density = Float64Array.from(grid.cells, (cell) => cell.density);
  const gamma = Float64Array.from(grid.cells, (cell) => cell.gamma);
  const velocity = new Float64Array(3 * grid.cells.length);
  for (const cell of grid.cells) {
    velocity[3 * cell.id] = 2.25;
    velocity[3 * cell.id + 1] = 0.15 * (cell.centerFine[2] - 4);
  }
  const before = grid.cells.reduce((sum, cell) =>
    sum + cell.volume * density[cell.id], 0);
  const result = transportSparseAtlasCM12(
    grid, { density, gamma, velocity }, 1 / 30,
  );
  const after = grid.cells.reduce((sum, cell) =>
    sum + cell.volume * result.fields.density[cell.id], 0);
  assert.ok(Math.abs(after - before) < 1e-10, `${before} -> ${after}`);
  assert.ok(result.finalBetaMaximumAbsoluteError < 1e-12,
    `${result.finalBetaMaximumAbsoluteError}`);
  assert.ok(Array.from(result.fields.density).every((value) =>
    Number.isFinite(value) && value >= -1e-12));
});

test("coarse CM12 has no sub-cell transport dead zone", () => {
  const atlas = createSparseAdaptiveMassAtlas([24, 8, 8], [
    { ...brick(0, [0, 0, 0], 4), density: new Float64Array(64) },
    { ...brick(1, [1, 0, 0], 4), density: new Float64Array(64).fill(1) },
    { ...brick(2, [2, 0, 0], 4), density: new Float64Array(64) },
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const density = Float64Array.from(grid.cells, (cell) => cell.density);
  const gamma = Float64Array.from(grid.cells, (cell) => cell.gamma);
  const velocity = new Float64Array(3 * grid.cells.length);
  for (const cell of grid.cells) velocity[3 * cell.id] = 10;
  const result = transportSparseAtlasCM12(
    grid, { density, gamma, velocity }, 1 / 30,
  );
  assert.ok(Array.from(result.fields.density).some((value, id) =>
    grid.cells[id].brickKey === 2 && value > 0));
  const before = grid.cells.reduce((total, cell) =>
    total + cell.volume * density[cell.id], 0);
  const after = grid.cells.reduce((total, cell) =>
    total + cell.volume * result.fields.density[cell.id], 0);
  assert.ok(Math.abs(after - before) < 1e-10, `${before} -> ${after}`);
});

test("CM12 face extension uses the rho > 0.5 MAC source band", () => {
  const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [
    brick(0, [0, 0, 0], 8),
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const density = new Float64Array(grid.cells.length).fill(0.1);
  const wet = grid.cells.find((cell) =>
    cell.minimumFine[0] === 3 && cell.minimumFine[1] === 3
    && cell.minimumFine[2] === 3);
  assert.ok(wet);
  density[wet.id] = 1;
  const input = new Float64Array(grid.gradientRows.length).fill(3);
  const fallback = new Float64Array(grid.gradientRows.length).fill(9);
  const extended = extrapolateSparseAtlasFaceVelocity(
    grid, density, input, fallback,
  );
  const sources = grid.gradientRows.filter((row) =>
    row.terms.some((term) => term.cellId === wet.id));
  assert.ok(sources.length > 0);
  assert.ok(sources.every((row) => extended[row.id] === 3));
  assert.ok(grid.gradientRows.some((row) =>
    !sources.includes(row) && extended[row.id] === 3),
  "the two-cell narrow band should extend beyond source faces");
  assert.ok(grid.gradientRows.some((row) => extended[row.id] === 9),
    "faces outside the narrow band should retain the sparse far-field fill");
});

test("large-CFL transport allocates reachable tiles without scanning the domain", () => {
  const atlas = createSparseAdaptiveMassAtlas([64, 8, 8], [
    brick(1, [1, 0, 0], 8),
  ]);
  const initial = initializeSparseAtlasDynamics(atlas);
  const cellVelocity = new Float64Array(3 * initial.grid.cells.length);
  for (const cell of initial.grid.cells) cellVelocity[3 * cell.id] = 800;
  const faceNormalVelocity = Float64Array.from(initial.grid.gradientRows, (row) =>
    row.axis === 0 ? 800 : 0);
  const result = stepSparseAtlasDynamics({
    ...initial,
    cellVelocity,
    faceNormalVelocity,
  }, {
    dt_s: 1 / 30,
    resolutionMode: "all-fine",
    project: false,
  });
  // 800 * dt = 26.7 finest cells: one brick cannot contain the trace. The
  // support grows to the reachable sparse band, but remains below all 8
  // authored x tiles because the source begins near the negative boundary.
  assert.ok(result.workGrid.atlas.bricks.length >= 5,
    `${result.workGrid.atlas.bricks.length}`);
  assert.ok(result.workGrid.atlas.bricks.length < 8,
    `${result.workGrid.atlas.bricks.length}`);
  assert.equal(result.workGrid.atlas.bricks.every((candidate) =>
    candidate.resolution === 8), true);
});

test("all-coarse dynamics keeps resident and newly reached tiles at 4 cubed", () => {
  const atlas = createSparseAdaptiveMassAtlas([32, 8, 8], [
    brick(1, [1, 0, 0], 4),
  ]);
  const initial = initializeSparseAtlasDynamics(atlas);
  const cellVelocity = new Float64Array(3 * initial.grid.cells.length);
  for (const cell of initial.grid.cells) cellVelocity[3 * cell.id] = 80;
  const faceNormalVelocity = Float64Array.from(initial.grid.gradientRows, (row) =>
    row.axis === 0 ? 80 : 0);
  const result = stepSparseAtlasDynamics({
    ...initial,
    cellVelocity,
    faceNormalVelocity,
  }, {
    dt_s: 1 / 30,
    resolutionMode: "all-coarse",
    project: false,
  });
  assert.equal(result.workGrid.atlas.bricks.every((candidate) =>
    candidate.resolution === 4), true);
  assert.equal(result.state.atlas.bricks.every((candidate) =>
    candidate.resolution === 4), true);
  assert.equal(result.stats.resolutionPolicy.targetFineBrickCount, 0);
  assert.equal(
    result.stats.resolutionPolicy.targetCoarseBrickCount,
    result.workGrid.atlas.bricks.length,
  );
});
