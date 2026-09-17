import assert from "node:assert/strict";
import test from "node:test";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import {
  applySparseAtlasDivergence,
  applySparseAtlasGradient,
  buildSparseAtlasCompositeGrid,
} from
  "../lib/methods/adaptive-volume/sparse-atlas-composite-projection";
import { initializeSparseAtlasDynamics, stepSparseAtlasDynamics } from
  "../lib/methods/adaptive-volume/sparse-atlas-dynamics";
import { extrapolateSparseAtlasFaceVelocity, transportSparseAtlasCM12 } from
  "../lib/methods/adaptive-volume/sparse-atlas-cm12-transport";
import { conditionSparseAtlasSurface } from
  "../lib/methods/adaptive-volume/sparse-atlas-surface-conditioning";

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

test("stationary CM12 transport retains non-unit gamma history", () => {
  const grid = buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas([8, 8, 8], [
    brick(0, [0, 0, 0], 8),
  ]));
  for (const value of [0.5, 0.75, 1.25, 2]) {
    const density = new Float64Array(grid.cells.length).fill(1);
    const gamma = new Float64Array(grid.cells.length).fill(value);
    const velocity = new Float64Array(3 * grid.cells.length);
    const result = transportSparseAtlasCM12(grid, { density, gamma, velocity }, 1 / 30);
    assert.deepEqual(result.fields.density, density);
    assert.deepEqual(result.fields.gamma, gamma,
      `zero velocity must not erase gamma=${value}`);
  }
});

test("CM12 transports cumulative gamma with density through repeated axial flow", () => {
  // A passive constant concentration has rho / gamma = constant. This must
  // survive arbitrary conservative transport, including the sign-changing
  // characteristic stencils at the centre and unequal-volume brick seams.
  for (const resolution of [8, 4] as const) {
    const grid = buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas([16, 8, 8], [
      brick(0, [0, 0, 0], 8), brick(1, [1, 0, 0], resolution),
    ]));
    let gamma: Float64Array = Float64Array.from(grid.cells, (cell) =>
      1 + 0.2 * Math.cos(0.7 * cell.centerFine[0]) * Math.cos(0.4 * cell.centerFine[1]));
    let density: Float64Array = Float64Array.from(gamma, value => 0.7 * value);
    const velocity = Float64Array.from({ length: 3 * grid.cells.length }, (_, index) => {
      const cell = grid.cells[Math.floor(index / 3)];
      return index % 3 === 0 ? 3 * (cell.centerFine[0] - 8)
        : index % 3 === 1 ? -3 * (cell.centerFine[1] - 4) : 0;
    });
    const integral = (values: ArrayLike<number>) => grid.cells.reduce(
      (sum, cell) => sum + cell.volume * values[cell.id], 0);
    const initialGamma = integral(gamma), initialMass = integral(density);
    for (let step = 0; step < 6; step++) {
      const result = transportSparseAtlasCM12(grid, { density, gamma, velocity }, 1 / 30);
      density = result.fields.density;
      gamma = result.fields.gamma;
      assert.ok(Math.abs(integral(density) - initialMass) < 1e-9);
      assert.ok(Math.abs(integral(gamma) - initialGamma) < 1e-9,
        `cumulative gamma volume must survive step ${step}, B${resolution}`);
      for (const cell of grid.cells) {
        assert.ok(Math.abs(density[cell.id] - 0.7 * gamma[cell.id]) < 1e-11,
          `transport must retain concentration at cell ${cell.id}, step ${step}, B${resolution}`);
      }
    }
  }
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

for (const fixture of [
  { name: "B1", dimensions: [16, 8, 8] as const, spanBricks: 1 },
  { name: "macro B1", dimensions: [32, 16, 16] as const, spanBricks: 2 },
] as const) test(`${fixture.name} velocity extension reaches the next parallel face`, () => {
  const source = (x: number, density: number): SparseAdaptiveMassBrick => ({
    key: x,
    coordinate: [x, 0, 0],
    ...(fixture.spanBricks === 1 ? {} : { spanBricks: fixture.spanBricks }),
    resolution: 1,
    density: new Float64Array([density]),
    gamma: new Float64Array([1]),
  });
  const atlas = createSparseAdaptiveMassAtlas(fixture.dimensions, [
    source(0, 1), source(fixture.spanBricks, 0),
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const density = Float64Array.from(grid.cells, (cell) => cell.density);
  const input = Float64Array.from(grid.gradientRows, (row) => row.axis === 0 ? 3 : 0);
  const fallback = new Float64Array(grid.gradientRows.length).fill(9);
  const extended = extrapolateSparseAtlasFaceVelocity(
    grid, density, input, fallback, atlas.brickFineResolution * fixture.spanBricks,
  );
  const farFace = grid.gradientRows.find((row) => row.axis === 0
    && row.centerFine[0] === fixture.dimensions[0]);
  assert.ok(farFace, "fixture must expose the dry brick's positive x face");
  assert.equal(extended[farFace.id], 3,
    "the FIM graph must not lose its normal axis beyond a B2-width search");
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


test("surface conditioning preserves an asymmetric edit after a symmetric frame", () => {
  const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [brick(0, [0, 0, 0], 8)]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const density = new Float64Array(grid.cells.length).fill(0.8);
  const gamma = new Float64Array(grid.cells.length).fill(1);
  const options = { gammaDiffusionIterations: 0, sharpeningCourant: 0 };
  conditionSparseAtlasSurface(grid, { density: density.slice(), gamma: gamma.slice() }, options);
  // A previous symmetric frame must not grant permission to average a later edit.
  density[0] = 0.9;
  gamma[0] = 1.1;
  const result = conditionSparseAtlasSurface(grid, { density, gamma }, options);
  assert.deepEqual(result.fields.density, density);
  assert.deepEqual(result.fields.gamma, gamma);
});
