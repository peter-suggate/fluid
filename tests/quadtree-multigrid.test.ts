import assert from "node:assert/strict";
import test from "node:test";
import {
  buildQuadtreeMultigridHierarchy,
  applyQuadtreeMultigridVcycle,
  galerkinCoarseMatrix,
  prolongatePiecewiseConstant,
  restrictTranspose,
} from "../lib/quadtree-multigrid";

function packedFineMatrix(matrix: number[][]) {
  const count = matrix.length;
  const rows = matrix.map((values, row) => values.flatMap((value, column) => value !== 0 || row === column ? [{ column, value }] : []));
  const entryCount = rows.reduce((sum, row) => sum + row.length, 0);
  const words = new Uint32Array(count + 1 + 4 * entryCount);
  const floats = new Float32Array(words.buffer);
  let cursor = 0;
  rows.forEach((entries, row) => {
    words[row] = cursor;
    for (const entry of entries) {
      const base = count + 1 + 4 * cursor++;
      words[base] = entry.column;
      floats[base + 2] = entry.value;
    }
  });
  words[count] = cursor;
  return words;
}

function sampleAux(columns: number, rows: number) {
  const words = new Uint32Array(4 * columns * rows);
  let node = 0;
  for (let x = 0; x < columns; x += 1) for (let y = 0; y < rows; y += 1) {
    words[4 * node] = x | (y << 20);
    words[4 * node + 1] = 1;
    words[4 * node + 2] = 1;
    node += 1;
  }
  return words;
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>) {
  let result = 0;
  for (let index = 0; index < a.length; index += 1) result += a[index] * b[index];
  return result;
}

test("quadtree geometric transfers preserve constants and use R = P^T", () => {
  // Four vertical two-node columns. The first geometric step aggregates x in
  // pairs but deliberately retains both y modes for the vertical line solve.
  const count = 8;
  const matrix: number[][] = Array.from({ length: count }, (_, row) => Array.from({ length: count }, (_, column) => row === column ? 4 : 0));
  for (let x = 0; x < 4; x += 1) for (let y = 0; y < 2; y += 1) {
    const row = 2 * x + y;
    if (x + 1 < 4) { matrix[row][row + 2] = -1; matrix[row + 2][row] = -1; }
    if (y === 0) { matrix[row][row + 1] = -1; matrix[row + 1][row] = -1; }
  }
  const hierarchy = buildQuadtreeMultigridHierarchy(packedFineMatrix(matrix), count, sampleAux(4, 2), 0, 4, 2, 1, { coarsestNodeLimit: 2 });
  assert.ok(hierarchy.levels.length >= 2);
  const first = hierarchy.levels[0], parent = first.nodeToCoarse!;
  assert.equal(hierarchy.levels[1].nodeCount, 4, "x semicoarsens while the two vertical samples remain distinct");
  assert.deepEqual([...prolongatePiecewiseConstant(new Float64Array(hierarchy.levels[1].nodeCount).fill(1), parent)], new Array(count).fill(1));

  const fine = Float64Array.from({ length: count }, (_, index) => 0.25 + index);
  const coarse = Float64Array.from({ length: hierarchy.levels[1].nodeCount }, (_, index) => 1.5 - 0.2 * index);
  assert.ok(Math.abs(dot(restrictTranspose(fine, parent), coarse) - dot(fine, prolongatePiecewiseConstant(coarse, parent))) < 1e-12);
});

test("Galerkin aggregation preserves energy and positive definiteness", () => {
  const matrix = [
    3, -1, 0, 0,
    -1, 3, -1, 0,
    0, -1, 3, -1,
    0, 0, -1, 2,
  ];
  const parent = new Uint32Array([0, 0, 1, 1]);
  const coarse = galerkinCoarseMatrix(matrix, 4, parent, 2);
  assert.deepEqual([...coarse], [4, -1, -1, 3]);
  for (const vector of [[1, 0], [0, 1], [1, -2], [-0.3, 0.7]]) {
    const prolonged = prolongatePiecewiseConstant(vector, parent);
    assert.ok(Math.abs(dot(vector, Float64Array.from([
      coarse[0] * vector[0] + coarse[1] * vector[1],
      coarse[2] * vector[0] + coarse[3] * vector[1],
    ])) - dot(prolonged, Float64Array.from({ length: 4 }, (_, row) => {
      let value = 0;
      for (let column = 0; column < 4; column += 1) value += matrix[4 * row + column] * prolonged[column];
      return value;
    }))) < 1e-12);
    assert.ok(dot(vector, Float64Array.from([
      coarse[0] * vector[0] + coarse[1] * vector[1],
      coarse[2] * vector[0] + coarse[3] * vector[1],
    ])) > 0);
  }
});

test("symbolic Galerkin entry maps exactly cover the next CSR", () => {
  const count = 8;
  const matrix = Array.from({ length: count }, (_, row) => Array.from({ length: count }, (_, column) => row === column ? 3 : Math.abs(row - column) === 2 ? -0.5 : 0));
  const hierarchy = buildQuadtreeMultigridHierarchy(packedFineMatrix(matrix), count, sampleAux(4, 2), 0, 4, 2, 1, { coarsestNodeLimit: 2 });
  for (let levelIndex = 0; levelIndex + 1 < hierarchy.levels.length; levelIndex += 1) {
    const level = hierarchy.levels[levelIndex], coarse = hierarchy.levels[levelIndex + 1];
    assert.equal(level.nodeToCoarse?.length, level.nodeCount);
    assert.equal(level.entryToCoarse?.length, level.columns.length);
    for (const target of level.entryToCoarse!) assert.ok(target < coarse.columns.length);
    for (let row = 0; row < coarse.nodeCount; row += 1) {
      const columns = [...coarse.columns.slice(coarse.rowOffsets[row], coarse.rowOffsets[row + 1])];
      assert.ok(columns.includes(row), "every coarse row has an explicit diagonal");
      assert.deepEqual(columns, [...columns].sort((a, b) => a - b));
    }
  }
});

test("the symmetric V-cycle is a positive PCG preconditioner", () => {
  const count = 8;
  const rows: number[][] = Array.from({ length: count }, (_, row) => Array.from({ length: count }, (_, column) => row === column ? 4 : 0));
  for (let row = 0; row < count; row += 1) {
    if (row + 1 < count && row % 2 === 0) { rows[row][row + 1] = -1; rows[row + 1][row] = -1; }
    if (row + 2 < count) { rows[row][row + 2] = -0.5; rows[row + 2][row] = -0.5; }
  }
  const dense = rows.flat();
  const hierarchy = buildQuadtreeMultigridHierarchy(packedFineMatrix(rows), count, sampleAux(4, 2), 0, 4, 2, 1, { coarsestNodeLimit: 2 });
  const x = Float64Array.from([1, -0.5, 0.2, 0.7, -0.1, 0.4, 0.3, -0.8]);
  const y = Float64Array.from([-0.2, 0.9, 0.1, -0.4, 0.8, 0.3, -0.6, 0.5]);
  const mx = applyQuadtreeMultigridVcycle(dense, hierarchy, x);
  const my = applyQuadtreeMultigridVcycle(dense, hierarchy, y);
  assert.ok(Math.abs(dot(x, my) - dot(mx, y)) < 1e-11, "pre/post block smoothing and R=P^T make the V-cycle symmetric");
  assert.ok(dot(x, mx) > 0, "the fixed V-cycle has positive energy");
  assert.ok(dot(y, my) > 0, "positive energy is not vector-specific");
});
