import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_SPGRID_ACTIVE,
  OCTREE_SPGRID_FINEST_AT_LEVEL,
  OCTREE_SPGRID_GHOST,
  OCTREE_SPGRID_MULTIGRID_ONLY,
  applyOctreeSPGridVCycle,
  buildOctreeSPGridMultigrid,
  factorOctreeSPGridCoarsest,
  galerkinOctreeSPGridOperator,
  materializeOctreeSPGridTransfer,
  multiplyOctreeSPGridMatrix,
  solveOctreeSPGridCoarsest,
  type OctreeSPGridCellInput,
  type OctreeSPGridHierarchy,
  type OctreeSPGridLevelInput,
} from "../lib/octree-spgrid-multigrid";

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) sum += a[index] * b[index];
  return sum;
};

function makePatchLevels(includeCopy = false): readonly [OctreeSPGridLevelInput, OctreeSPGridLevelInput] {
  const coarseCells: OctreeSPGridCellInput[] = [];
  for (let z = 0; z < 4; z += 1) for (let y = 0; y < 4; y += 1) for (let x = 0; x < 4; x += 1) {
    coarseCells.push({ id: `c:${x}:${y}:${z}`, coordinate: [x, y, z],
      flags: OCTREE_SPGRID_MULTIGRID_ONLY | OCTREE_SPGRID_FINEST_AT_LEVEL });
  }
  const fineCells: OctreeSPGridCellInput[] = [];
  for (let z = 2; z < 7; z += 1) for (let y = 2; y < 7; y += 1) for (let x = 2; x < 7; x += 1) {
    const copy = includeCopy && x === 4 && y === 4 && z === 4;
    fineCells.push({ id: `f:${x}:${y}:${z}`, coordinate: [x, y, z],
      flags: copy ? OCTREE_SPGRID_GHOST : OCTREE_SPGRID_ACTIVE | OCTREE_SPGRID_FINEST_AT_LEVEL,
      persistsAs: copy ? "c:2:2:2" : undefined });
  }
  return [{ cellWidth: 1, cells: fineCells }, { cellWidth: 2, cells: coarseCells }];
}

function dirichletLaplacian(cells: readonly OctreeSPGridCellInput[]): Float64Array {
  const count = cells.length, operator = new Float64Array(count * count);
  const byCoordinate = new Map(cells.map((cell, index) => [cell.coordinate.join(":"), index]));
  const directions = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]] as const;
  cells.forEach((cell, row) => {
    operator[row * count + row] = 7;
    for (const direction of directions) {
      const neighbor = byCoordinate.get(cell.coordinate.map((value, axis) => value + direction[axis]).join(":"));
      if (neighbor !== undefined) operator[row * count + neighbor] = -1;
    }
  });
  return operator;
}

function patchHierarchy(includeCopy = false): OctreeSPGridHierarchy {
  const levels = makePatchLevels(includeCopy);
  return buildOctreeSPGridMultigrid({ finestOperator: dirichletLaplacian(levels[0].cells), levels });
}

test("native levels retain active, ghost, multigrid-only, and finest-cell smoothing flags", () => {
  const hierarchy = patchHierarchy(true), fine = hierarchy.levels[0], coarse = hierarchy.levels[1];
  assert.equal(fine.ids.length, 125); assert.equal(coarse.ids.length, 64);
  assert.equal([...fine.flags].filter((flags) => (flags & OCTREE_SPGRID_GHOST) !== 0).length, 1);
  assert.equal([...fine.smoothingMask].reduce((sum, value) => sum + value, 0), 124,
    "persistent ghosts participate in transfers but not finest-cell smoothing");
  assert.ok([...coarse.flags].every((flags) => (flags & OCTREE_SPGRID_MULTIGRID_ONLY) !== 0));
  assert.deepEqual([...fine.coordinates.slice(0, 3)], [2, 2, 2]);
});

test("restriction copies persistent cells, distributes finest cells trilinearly, and is exactly P transpose", () => {
  const levels = makePatchLevels(true), transfer = materializeOctreeSPGridTransfer(levels[0], levels[1]);
  const copyFine = levels[0].cells.findIndex((cell) => cell.persistsAs !== undefined);
  const copyRecords = transfer.records.filter((record) => record.fine === copyFine);
  assert.deepEqual(copyRecords.map(({ weight, mode }) => ({ weight, mode })), [{ weight: 1, mode: "copy" }]);

  const trilinearFine = levels[0].cells.findIndex((cell) => cell.coordinate.join(":") === "2:2:2");
  const trilinearRecords = transfer.records.filter((record) => record.fine === trilinearFine);
  assert.equal(trilinearRecords.length, 8);
  assert.ok(Math.abs(trilinearRecords.reduce((sum, record) => sum + record.weight, 0) - 1) < 1e-15);
  assert.deepEqual([...new Set(trilinearRecords.map((record) => record.weight))].sort(), [1 / 64, 3 / 64, 9 / 64, 27 / 64],
    "cell-centred factor-two interpolation is the tensor product of 1/4 and 3/4 weights");

  for (let fine = 0; fine < transfer.fineCount; fine += 1) for (let coarse = 0; coarse < transfer.coarseCount; coarse += 1) {
    assert.equal(transfer.restriction[coarse * transfer.fineCount + fine],
      transfer.prolongation[fine * transfer.coarseCount + coarse]);
  }
  const x = Float64Array.from({ length: transfer.coarseCount }, (_, index) => Math.sin(index + 0.25));
  const y = Float64Array.from({ length: transfer.fineCount }, (_, index) => Math.cos(0.3 * index));
  const px = multiplyOctreeSPGridMatrix(transfer.prolongation, transfer.fineCount, transfer.coarseCount, x);
  const ry = multiplyOctreeSPGridMatrix(transfer.restriction, transfer.coarseCount, transfer.fineCount, y);
  assert.ok(Math.abs(dot(px, y) - dot(x, ry)) < 1e-12);
});

test("sparse topology fails closed when a trilinear support cell is absent", () => {
  const levels = makePatchLevels();
  const incomplete = { ...levels[1], cells: levels[1].cells.slice(1) };
  assert.throws(() => materializeOctreeSPGridTransfer(levels[0], incomplete), /Incomplete trilinear support/);
});

test("Galerkin levels are symmetric and retain strictly positive energy", () => {
  const levels = makePatchLevels(), fineOperator = dirichletLaplacian(levels[0].cells);
  const transfer = materializeOctreeSPGridTransfer(levels[0], levels[1]);
  const coarse = galerkinOctreeSPGridOperator(fineOperator, transfer), count = transfer.coarseCount;
  for (let row = 0; row < count; row += 1) for (let column = 0; column < count; column += 1) {
    assert.ok(Math.abs(coarse[row * count + column] - coarse[column * count + row]) < 1e-12);
  }
  for (let sample = 0; sample < 12; sample += 1) {
    const value = Float64Array.from({ length: count }, (_, index) => Math.sin((sample + 1.25) * (index + 0.5)));
    const product = multiplyOctreeSPGridMatrix(coarse, count, count, value);
    assert.ok(dot(value, product) > 1e-8);
    const prolonged = multiplyOctreeSPGridMatrix(transfer.prolongation, transfer.fineCount, count, value);
    const fineProduct = multiplyOctreeSPGridMatrix(fineOperator, transfer.fineCount, transfer.fineCount, prolonged);
    assert.ok(Math.abs(dot(value, product) - dot(prolonged, fineProduct)) < 1e-10,
      "coarse and prolonged vectors have identical Galerkin energy");
  }
});

test("the fixed LDLT bottom solve is deterministic, linear, and exact", () => {
  const operator = new Float64Array([4, -1, 0, -1, 3, -0.5, 0, -0.5, 2]);
  const factor = factorOctreeSPGridCoarsest(operator, 3), a = new Float64Array([1, -2, 0.5]);
  const b = new Float64Array([-0.25, 3, 2]), solveA = solveOctreeSPGridCoarsest(factor, a);
  const solveB = solveOctreeSPGridCoarsest(factor, b), combined = solveOctreeSPGridCoarsest(factor,
    Float64Array.from(a, (value, index) => 2 * value - 0.5 * b[index]));
  assert.deepEqual([...solveOctreeSPGridCoarsest(factor, a)], [...solveA]);
  for (let index = 0; index < 3; index += 1) assert.ok(Math.abs(combined[index] - (2 * solveA[index] - 0.5 * solveB[index])) < 1e-14);
  const product = multiplyOctreeSPGridMatrix(operator, 3, 3, solveA);
  for (let index = 0; index < 3; index += 1) assert.ok(Math.abs(product[index] - a[index]) < 1e-14);
  assert.throws(() => factorOctreeSPGridCoarsest(new Float64Array([1, 2, 2, 1]), 2), /not positive definite/);
});

test("the finest-cell-only V-cycle is symmetric, positive, linear, and contracts a bounded residual", () => {
  const hierarchy = patchHierarchy(), count = hierarchy.levels[0].ids.length;
  const x = Float64Array.from({ length: count }, (_, index) => Math.sin(0.31 * index + 0.2));
  const y = Float64Array.from({ length: count }, (_, index) => Math.cos(0.23 * index - 0.4));
  const mx = applyOctreeSPGridVCycle(hierarchy, x), my = applyOctreeSPGridVCycle(hierarchy, y);
  assert.ok(Math.abs(dot(x, my) - dot(y, mx)) < 2e-10);
  assert.ok(dot(x, mx) > 0 && dot(y, my) > 0);
  const combinedRhs = Float64Array.from(x, (value, index) => 1.7 * value - 0.4 * y[index]);
  const combined = applyOctreeSPGridVCycle(hierarchy, combinedRhs);
  for (let index = 0; index < count; index += 1) {
    assert.ok(Math.abs(combined[index] - (1.7 * mx[index] - 0.4 * my[index])) < 2e-12);
  }

  const operator = hierarchy.levels[0].operator;
  const residual = Float64Array.from(x, (value, row) => value
    - multiplyOctreeSPGridMatrix(operator, count, count, mx)[row]);
  assert.ok(Math.sqrt(dot(residual, residual)) < Math.sqrt(dot(x, x)), "one bounded V-cycle reduces this Poisson residual");
});
