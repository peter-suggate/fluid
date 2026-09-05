import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import {
  applySparseAtlasPressureOperator,
  buildSparseAtlasCompositeGrid,
  type SparseAtlasCompositeGrid,
} from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { transportSparseAtlasCM12 } from
  "../lib/methods/adaptive-mass/sparse-atlas-cm12-transport";
import { compileSparseCM12BrickTileFaceProgram,
  validateSparseCM12BrickTileFaceProgram } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-program";
import { compileSparseCM12BrickTileImage, validateSparseCM12BrickTileImage } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image";
import {
  sparseBrickKey,
  sparseBrickLadder,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";

/** Deliberately bypass strong grading only inside this adversarial topology fixture. */
function ungradedAtlas(resolutions: readonly SparseBrickResolution[], generation = 1):
SparseAdaptiveMassAtlas {
  const brickDimensions = [resolutions.length, 1, 1] as const;
  const dimensions = [8 * resolutions.length, 8, 8] as const;
  const bricks: SparseAdaptiveMassBrick[] = resolutions.map((resolution, x) => ({
    key: sparseBrickKey([x, 0, 0], brickDimensions),
    coordinate: [x, 0, 0],
    resolution,
    density: Float64Array.from({ length: resolution ** 3 }, (_, local) =>
      0.25 + 0.5 * ((local * 17 + x * 11) % 29) / 28),
    gamma: new Float64Array(resolution ** 3).fill(1),
  }));
  const directory = new Map(bricks.map((brick) => [brick.key, brick] as const));
  return {
    dimensions,
    brickFineResolution: 8,
    brickCellCapacity: 8 ** 3,
    ladder: sparseBrickLadder(8),
    brickDimensions,
    bricks,
    directory,
    directoriesBySpan: new Map([[1, directory]]),
    maximumSpanBricks: 1,
    generation,
  };
}

function dot(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let result = 0;
  for (let index = 0; index < left.length; index += 1) {
    result += left[index] * right[index];
  }
  return result;
}

function validateNumerics(grid: SparseAtlasCompositeGrid): void {
  const first = Float64Array.from(grid.cells, (cell) =>
    Math.sin(0.17 * (cell.id + 1)) + 0.003 * cell.centerFine[0]);
  const second = Float64Array.from(grid.cells, (cell) =>
    Math.cos(0.11 * (cell.id + 3)) - 0.002 * cell.centerFine[2]);
  const appliedFirst = applySparseAtlasPressureOperator(grid, first);
  const appliedSecond = applySparseAtlasPressureOperator(grid, second);
  const left = dot(first, appliedSecond), right = dot(second, appliedFirst);
  const symmetryScale = Math.max(1, Math.abs(left), Math.abs(right));
  assert.ok(Math.abs(left - right) <= 2e-13 * symmetryScale,
    `pressure operator lost symmetry: ${left} != ${right}`);

  let minimumRayleigh = Number.POSITIVE_INFINITY;
  for (let mode = 1; mode <= 5; mode += 1) {
    const sample = Float64Array.from(grid.cells, (cell) =>
      Math.sin(0.07 * mode * (cell.id + 1)) + Math.cos(0.13 * cell.centerFine[1]));
    minimumRayleigh = Math.min(minimumRayleigh,
      dot(sample, applySparseAtlasPressureOperator(grid, sample)) / dot(sample, sample));
  }
  assert.ok(minimumRayleigh >= -1e-12, `negative Rayleigh quotient ${minimumRayleigh}`);

  const fields = {
    density: Float64Array.from(grid.cells, (cell) => cell.density),
    gamma: new Float64Array(grid.cells.length).fill(1),
    velocity: Float64Array.from({ length: 3 * grid.cells.length }, (_, index) =>
      0.35 * Math.sin(0.19 * (index + 1))),
  };
  const transported = transportSparseAtlasCM12(grid, fields, 1 / 30);
  assert.ok(transported.finalBetaMaximumAbsoluteError <= 2e-12,
    `conservative column error ${transported.finalBetaMaximumAbsoluteError}`);
}

function validateCompiledTopology(grid: SparseAtlasCompositeGrid): {
  readonly imageWords: Uint32Array;
  readonly programWords: Uint32Array;
} {
  const image = compileSparseCM12BrickTileImage(grid);
  const imageReceipt = validateSparseCM12BrickTileImage(image, grid);
  assert.equal(imageReceipt.explicitAddressCollisionCount, 0);
  const program = compileSparseCM12BrickTileFaceProgram(image, grid);
  const programReceipt = validateSparseCM12BrickTileFaceProgram(program, image, grid);
  assert.equal(programReceipt.interiorRowCount + programReceipt.seamRowCount,
    grid.gradientRows.length);
  return { imageWords: image.words, programWords: program.words };
}

for (const fixture of [
  { name: "8|2", resolutions: [8, 2] as const, rows: 4, terms: 17 },
  { name: "8|1", resolutions: [8, 1] as const, rows: 1, terms: 65 },
] as const) test(`${fixture.name} seam remains ratio-generic`, () => {
  const grid = buildSparseAtlasCompositeGrid(ungradedAtlas(fixture.resolutions));
  const seams = grid.gradientRows.filter((row) => row.kind === "mixed-seam");
  assert.equal(seams.length, fixture.rows);
  assert.deepEqual([...new Set(seams.map((row) => row.terms.length))], [fixture.terms]);
  validateCompiledTopology(grid);
  validateNumerics(grid);
});

test("the four-rung row and compiled topology remain complete", () => {
  const grid = buildSparseAtlasCompositeGrid(ungradedAtlas([8, 4, 2, 1]));
  const seamHistogram = new Map<number, number>();
  for (const row of grid.gradientRows.filter((candidate) => candidate.kind === "mixed-seam")) {
    seamHistogram.set(row.terms.length, (seamHistogram.get(row.terms.length) ?? 0) + 1);
  }
  assert.deepEqual([...seamHistogram], [[5, 4 * 4 + 2 * 2 + 1]]);
  validateCompiledTopology(grid);
  validateNumerics(grid);
});

test("symmetric 8|1|8 expansion has a raw-bit topology digest", () => {
  const grid = buildSparseAtlasCompositeGrid(ungradedAtlas([8, 1, 8], 7));
  const compiled = validateCompiledTopology(grid);
  const digest = createHash("sha256")
    .update(new Uint8Array(compiled.imageWords.buffer, compiled.imageWords.byteOffset,
      compiled.imageWords.byteLength))
    .update(new Uint8Array(compiled.programWords.buffer, compiled.programWords.byteOffset,
      compiled.programWords.byteLength))
    .digest("hex");
  assert.equal(digest, "a690ae5799d79a325462d3cc959b2c8533ebdb8a23a5fdee033e4ed238b9e8a3");
});
