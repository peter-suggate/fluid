import assert from "node:assert/strict";
import test from "node:test";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import {
  applySparseAtlasDivergence,
  applySparseAtlasGradient,
  buildSparseAtlasCompositeGrid,
} from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { initializeSparseAtlasDynamics, stepSparseAtlasDynamics } from
  "../lib/methods/adaptive-mass/sparse-atlas-dynamics";
import { extrapolateSparseAtlasFaceVelocity, transportSparseAtlasCM12 } from
  "../lib/methods/adaptive-mass/sparse-atlas-cm12-transport";
import { conditionSparseAtlasSurface } from
  "../lib/methods/adaptive-mass/sparse-atlas-surface-conditioning";

const brick = (
  key: number,
  coordinate: readonly [number, number, number],
  resolution: SparseBrickResolution,
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

test("the complete 1/2/4/8 ladder is strongly 2:1 graded", () => {
  const atlas = createSparseAdaptiveMassAtlas([32, 8, 8], [
    brick(0, [0, 0, 0], 8),
    brick(1, [1, 0, 0], 4),
    brick(2, [2, 0, 0], 2),
    brick(3, [3, 0, 0], 1),
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  assert.deepEqual(atlas.bricks.map((candidate) => candidate.resolution), [8, 4, 2, 1]);
  assert.equal(grid.mixedSeamRowCount, 4 * 4 + 2 * 2 + 1,
    "each adjacent rung must emit ports on the coarser face lattice");
  for (const row of grid.gradientRows) {
    if (row.negativeBrickKey === undefined || row.positiveBrickKey === undefined) continue;
    const negative = atlas.directory.get(row.negativeBrickKey)!;
    const positive = atlas.directory.get(row.positiveBrickKey)!;
    assert.ok(Math.max(negative.resolution, positive.resolution)
      / Math.min(negative.resolution, positive.resolution) <= 2);
  }
  const pressure = Float64Array.from(grid.cells, (cell) =>
    Math.sin(0.37 * (cell.id + 1)));
  const faceVelocity = Float64Array.from(grid.gradientRows, (row) =>
    Math.cos(0.23 * (row.id + 1)));
  const gradient = applySparseAtlasGradient(grid, pressure);
  const divergence = applySparseAtlasDivergence(grid, faceVelocity);
  const gradientPairing = grid.gradientRows.reduce((sum, row) =>
    sum + row.dualWeight * gradient[row.id] * faceVelocity[row.id], 0);
  const divergencePairing = grid.cells.reduce((sum, cell) =>
    sum + cell.volume * pressure[cell.id] * divergence[cell.id], 0);
  assert.ok(Math.abs(gradientPairing + divergencePairing) < 1e-11,
    `${gradientPairing} + ${divergencePairing}`);

  assert.throws(() => createSparseAdaptiveMassAtlas([16, 8, 8], [
    brick(0, [0, 0, 0], 8), brick(1, [1, 0, 0], 2),
  ]), /exceeds 2:1 grading/);
});

test("SolidWorld restriction evidence overrides coarse policy only for lost detail", () => {
  const options = {
    finestDimensions: [32, 16, 32] as const,
    resolutionForBrick: () => 4 as const,
  };
  const planar = initializeSparseBrickAtlasFromScene(
    createSymmetricExpansionScene(), options,
  );
  assert.ok(planar.bricks.length > 0);
  assert.equal(planar.bricks.every((candidate) => candidate.resolution === 4), true,
    "restriction-exact tank planes must honor a coarse scene policy");

  const detailedScene = createSymmetricExpansionScene();
  detailedScene.solidVoxels.push({ operation: "fill", minimum: [8, 0, 8],
    maximumExclusive: [9, 1, 9] });
  const detailed = initializeSparseBrickAtlasFromScene(detailedScene, options);
  assert.equal(detailed.bricks.find((candidate) =>
    candidate.coordinate[0] === 1 && candidate.coordinate[1] === 0
      && candidate.coordinate[2] === 1)?.resolution, 8,
  "a sub-macro-cell solid edit must retain its finest representation rung");
});

test("CM12 sharpening dose scales inversely with physical finest-cell size", () => {
  const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [
    { ...brick(0, [0, 0, 0], 4), density: new Float64Array(64) },
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const density = Float64Array.from(grid.cells, (cell) =>
    cell.centerFine[0] < 2 ? 0.35 : cell.centerFine[0] < 4 ? 0.45 : 0.55);
  const gamma = new Float64Array(grid.cells.length).fill(1);
  const run = (finestCellSize_m: number) => conditionSparseAtlasSurface(
    grid,
    { density: density.slice(), gamma: gamma.slice() },
    { gammaDiffusionIterations: 0, timeStep_s: 0.001, finestCellSize_m },
  );
  const metreGrid = run(1);
  const decimetreGrid = run(0.1);
  const l1Change = (values: ArrayLike<number>) => Array.from(values).reduce(
    (sum, value, index) => sum + Math.abs(value - density[index]), 0,
  );
  const weakDose = l1Change(metreGrid.fields.density);
  const physicalDose = l1Change(decimetreGrid.fields.density);
  assert.ok(physicalDose > 9.9 * weakDose && physicalDose < 10.1 * weakDose,
    `${weakDose} -> ${physicalDose}`);
  assert.ok(metreGrid.massAbsoluteError < 1e-12);
  assert.ok(decimetreGrid.massAbsoluteError < 1e-12);
});

test("CM12 Algorithm 2 traces sharpening mass across a 2:1 seam", () => {
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [
    brick(0, [0, 0, 0], 8),
    brick(1, [1, 0, 0], 4),
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const density = Float64Array.from(grid.cells, (cell) => {
    const x = cell.centerFine[0];
    return x < 5 ? 0.2 : x < 6 ? 0.3 : x < 7 ? 0.36 : x < 8 ? 0.42 : 0.6;
  });
  const coarseMass = (values: ArrayLike<number>) => grid.cells.reduce(
    (sum, cell) => sum + (cell.brickKey === 1 ? cell.volume * values[cell.id] : 0),
    0,
  );
  const before = coarseMass(density);
  const run = (distanceCells: number) => conditionSparseAtlasSurface(
    grid,
    { density: density.slice(), gamma: new Float64Array(density.length).fill(1) },
    {
      gammaDiffusionIterations: 0,
      timeStep_s: 0.05,
      finestCellSize_m: 1,
      sharpeningDistanceCells: distanceCells,
      preserveHorizontalD4: false,
    },
  );
  const local = run(1);
  const traced = run(2.1);
  assert.ok(coarseMass(local.fields.density) - before < 1e-3,
    "a one-cell trace should remain effectively on the fine side");
  assert.ok(coarseMass(traced.fields.density) - before > 1e-3,
    "the paper-distance trace must deposit into the coarse brick");
  assert.ok(traced.massAbsoluteError < 1e-10, `${traced.massAbsoluteError}`);
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
