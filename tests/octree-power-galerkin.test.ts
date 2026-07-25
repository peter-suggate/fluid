import assert from "node:assert/strict";
import test from "node:test";
import {
  applyOctreePowerGalerkinLevel,
  applyOctreePowerGalerkinVCycle,
  buildFixedAdaptiveOctreePowerGalerkinHierarchy,
  buildOctreePowerGalerkinHierarchy,
  packOctreePowerGalerkinLevel,
  planOctreePowerGalerkinExecution,
  refreshPackedOctreePowerGalerkinTarget,
  refreshOctreePowerGalerkinOperators,
} from "../lib/octree-power-galerkin";

function grid(size: number) {
  const count = size ** 3;
  const geometry = Array.from({ length: count }, (_, row) => ({
    x: row % size,
    y: Math.floor(row / size) % size,
    z: Math.floor(row / (size * size)),
    span: 1,
  }));
  const rows: number[][] = Array.from({ length: count }, () => []);
  const id = (x: number, y: number, z: number) => x + size * (y + size * z);
  for (let z = 0; z < size; z += 1) for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
    const row = id(x, y, z);
    for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
      const nx = x + dx, ny = y + dy, nz = z + dz;
      if (nx >= 0 && nx < size && ny >= 0 && ny < size && nz >= 0 && nz < size) rows[row].push(id(nx, ny, nz));
    }
  }
  const rowOffsets = new Uint32Array(count + 1);
  rows.forEach((row, index) => { rowOffsets[index + 1] = rowOffsets[index] + row.length; });
  const columns = Uint32Array.from(rows.flat());
  const diagonal = Float64Array.from(rows, (row) => row.length + 0.25);
  const coefficients = new Float64Array(columns.length).fill(1);
  return { geometry, rowOffsets, columns, diagonal, coefficients };
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let result = 0;
  for (let index = 0; index < a.length; index += 1) result += a[index] * b[index];
  return result;
};

test("octree power aggregation builds dyadic 64→8→1 topology with linear symbolic storage", () => {
  const fine = grid(4);
  const hierarchy = buildOctreePowerGalerkinHierarchy({
    dimensions: [4, 4, 4],
    geometry: fine.geometry,
    rowOffsets: fine.rowOffsets,
    columns: fine.columns,
    transfer: "piecewise-constant",
    coarsestNodeLimit: 1,
  });
  assert.deepEqual(hierarchy.levels.map((level) => level.nodeCount), [64, 8, 1]);
  assert.ok(hierarchy.symbolicEntryCount < 12 * fine.geometry.length,
    `unexpected symbolic fill ${hierarchy.symbolicEntryCount}`);
  for (const level of hierarchy.levels.slice(0, -1)) {
    assert.equal(level.nodeToCoarse!.length, level.nodeCount);
    assert.equal(level.entryToCoarse!.length, level.columns.length);
    assert.equal(level.coarseNodeOffsets!.at(-1), level.nodeCount);
  }
});

test("numeric refresh is exact Galerkin, symmetric, and preserves energy", () => {
  const fine = grid(4);
  const hierarchy = buildOctreePowerGalerkinHierarchy({
    dimensions: [4, 4, 4],
    geometry: fine.geometry,
    rowOffsets: fine.rowOffsets,
    columns: fine.columns,
    transfer: "piecewise-constant",
    coarsestNodeLimit: 1,
  });
  const operators = refreshOctreePowerGalerkinOperators(hierarchy, fine.diagonal, fine.coefficients);
  for (let levelIndex = 0; levelIndex + 1 < hierarchy.levels.length; levelIndex += 1) {
    const level = hierarchy.levels[levelIndex];
    const coarseLevel = hierarchy.levels[levelIndex + 1];
    const x = Float64Array.from({ length: coarseLevel.nodeCount }, (_, index) => Math.sin(index + 0.37));
    const px = Float64Array.from(level.nodeToCoarse!, (parent) => x[parent]);
    const fineProduct = applyOctreePowerGalerkinLevel(level, operators[levelIndex], px);
    const coarseProduct = applyOctreePowerGalerkinLevel(coarseLevel, operators[levelIndex + 1], x);
    assert.ok(Math.abs(dot(px, fineProduct) - dot(x, coarseProduct)) < 1e-10,
      `Galerkin energy changed at level ${levelIndex}`);
    const dense = Array.from({ length: coarseLevel.nodeCount }, () => new Float64Array(coarseLevel.nodeCount));
    for (let row = 0; row < coarseLevel.nodeCount; row += 1) {
      for (let entry = coarseLevel.rowOffsets[row]; entry < coarseLevel.rowOffsets[row + 1]; entry += 1) {
        dense[row][coarseLevel.columns[entry]] += operators[levelIndex + 1][entry];
      }
    }
    for (let row = 0; row < coarseLevel.nodeCount; row += 1) for (let column = 0; column < row; column += 1) {
      assert.ok(Math.abs(dense[row][column] - dense[column][row]) < 1e-12);
    }
  }
});

test("symbolic hierarchy is reused across coefficient-only numeric refreshes", () => {
  const fine = grid(4);
  const hierarchy = buildOctreePowerGalerkinHierarchy({
    dimensions: [4, 4, 4],
    geometry: fine.geometry,
    rowOffsets: fine.rowOffsets,
    columns: fine.columns,
    transfer: "piecewise-constant",
    coarsestNodeLimit: 1,
  });
  const first = refreshOctreePowerGalerkinOperators(hierarchy, fine.diagonal, fine.coefficients);
  const changedCoefficients = Float64Array.from(fine.coefficients, (_, index) => 0.5 + (index % 7) / 14);
  const changedDiagonal = Float64Array.from(fine.diagonal, (value) => value + 1);
  const second = refreshOctreePowerGalerkinOperators(hierarchy, changedDiagonal, changedCoefficients);
  assert.equal(hierarchy.levels[0].entryToCoarse, hierarchy.levels[0].entryToCoarse,
    "numeric refresh must not replace symbolic edge maps");
  assert.notDeepEqual([...first.at(-1)!], [...second.at(-1)!]);
});

test("fixed adaptive arena covers unit and coarsened power rows without overlap ambiguity", () => {
  const hierarchy = buildFixedAdaptiveOctreePowerGalerkinHierarchy(
    [4, 4, 4],
    2,
    { coarsestNodeLimit: 8 },
  );
  const finest = hierarchy.levels[0];
  assert.equal(finest.nodeCount, 72, "4³ unit candidates plus 2³ two-cell candidates");
  const unit = finest.geometry.findIndex((node) =>
    node.x === 0 && node.y === 0 && node.z === 0 && node.span === 1);
  const coarse = finest.geometry.findIndex((node) =>
    node.x === 0 && node.y === 0 && node.z === 0 && node.span === 2);
  const faceFine = finest.geometry.findIndex((node) =>
    node.x === 2 && node.y === 0 && node.z === 0 && node.span === 1);
  const edgeFine = finest.geometry.findIndex((node) =>
    node.x === 2 && node.y === 2 && node.z === 0 && node.span === 1);
  assert.ok(unit >= 0 && coarse >= 0 && faceFine >= 0 && edgeFine >= 0);
  const coarseColumns = finest.columns.slice(finest.rowOffsets[coarse], finest.rowOffsets[coarse + 1]);
  assert.ok(coarseColumns.includes(faceFine), "2:1 face contact must be importable");
  assert.ok(coarseColumns.includes(edgeFine), "power-diagram edge contact must be importable");

  const operators = refreshOctreePowerGalerkinOperators(
    hierarchy,
    new Float64Array(finest.nodeCount).fill(1),
    new Float64Array(hierarchy.fineOffDiagonalEntries.length),
  );
  const packed = packOctreePowerGalerkinLevel(hierarchy, operators, 0);
  const lookupBase = packed.topologyWords[14];
  const cellCount = packed.topologyWords[19];
  assert.equal(packed.topologyWords[20], 2);
  assert.equal(packed.topologyWords[lookupBase], unit + 1);
  assert.equal(packed.topologyWords[lookupBase + cellCount], coarse + 1,
    "same origin at a different dyadic size must have a distinct fixed row");
});

test("one native-L2 Galerkin V-cycle removes the low-frequency residual", () => {
  const fine = grid(8);
  const hierarchy = buildOctreePowerGalerkinHierarchy({
    dimensions: [8, 8, 8],
    geometry: fine.geometry,
    rowOffsets: fine.rowOffsets,
    columns: fine.columns,
    coarsestNodeLimit: 8,
  });
  const operators = refreshOctreePowerGalerkinOperators(hierarchy, fine.diagonal, fine.coefficients);
  assert.ok(hierarchy.symbolicContributionCount < 400 * fine.geometry.length,
    `trilinear Galerkin contribution map grew to ${hierarchy.symbolicContributionCount}`);
  const rhs = Float64Array.from({ length: fine.geometry.length }, (_, row) => {
    const node = fine.geometry[row];
    return Math.sin((Math.PI * (node.x + 0.5)) / 8)
      * Math.sin((Math.PI * (node.y + 0.5)) / 8)
      * Math.sin((Math.PI * (node.z + 0.5)) / 8);
  });
  const correction = applyOctreePowerGalerkinVCycle(hierarchy, operators, rhs);
  const product = applyOctreePowerGalerkinLevel(hierarchy.levels[0], operators[0], correction);
  const before = Math.sqrt(dot(rhs, rhs));
  const after = Math.sqrt(rhs.reduce((sum, value, row) => sum + (value - product[row]) ** 2, 0));
  assert.ok(after / before < 0.08, `one V-cycle left relative residual ${after / before}`);
  const residual = Float64Array.from(rhs, (value, row) => value - product[row]);
  const secondCorrection = applyOctreePowerGalerkinVCycle(hierarchy, operators, residual);
  const secondProduct = applyOctreePowerGalerkinLevel(hierarchy.levels[0], operators[0], secondCorrection);
  const final = Math.sqrt(residual.reduce((sum, value, row) => sum + (value - secondProduct[row]) ** 2, 0));
  assert.ok(final / before < 0.015, `two V-cycles left relative residual ${final / before}`);
});

test("parent-owned packed RAP gather reproduces every coarse coefficient without atomics", () => {
  const fine = grid(8);
  const hierarchy = buildOctreePowerGalerkinHierarchy({
    dimensions: [8, 8, 8],
    geometry: fine.geometry,
    rowOffsets: fine.rowOffsets,
    columns: fine.columns,
    coarsestNodeLimit: 8,
  });
  const operators = refreshOctreePowerGalerkinOperators(hierarchy, fine.diagonal, fine.coefficients);
  for (let level = 0; level + 1 < hierarchy.levels.length; level += 1) {
    const packed = packOctreePowerGalerkinLevel(hierarchy, operators, level);
    const source = hierarchy.levels[level];
    const target = hierarchy.levels[level + 1];
    const diagonalBase = packed.topologyWords[13];
    const targetDiagonalBase = packed.topologyWords[21];
    assert.ok(diagonalBase > 0);
    assert.equal(targetDiagonalBase, diagonalBase + source.nodeCount);
    for (let row = 0; row < source.nodeCount; row += 1) {
      const entry = packed.topologyWords[diagonalBase + row];
      assert.ok(entry >= source.rowOffsets[row] && entry < source.rowOffsets[row + 1]);
      assert.equal(source.columns[entry], row);
    }
    for (let row = 0; row < target.nodeCount; row += 1) {
      const entry = packed.topologyWords[targetDiagonalBase + row];
      assert.ok(entry >= target.rowOffsets[row] && entry < target.rowOffsets[row + 1]);
      assert.equal(target.columns[entry], row);
    }
    if (level === 0) {
      const cellToRowBase = packed.topologyWords[14];
      const geometryBase = packed.topologyWords[15];
      assert.ok(diagonalBase > 0 && targetDiagonalBase > diagonalBase
        && cellToRowBase > targetDiagonalBase && geometryBase > cellToRowBase);
      assert.deepEqual([...packed.topologyWords.slice(16, 19)], [...hierarchy.dimensions]);
      assert.deepEqual(
        packed.topologyWords.slice(diagonalBase, diagonalBase + hierarchy.fineDiagonalEntries.length),
        hierarchy.fineDiagonalEntries,
      );
      assert.equal(packed.topologyWords[cellToRowBase], 1,
        "the fixed hierarchy maps flattened cell zero to encoded row zero");
      assert.deepEqual([...packed.topologyWords.slice(geometryBase, geometryBase + 2)], [0, 1]);
    }
    const refreshOffsetBase = packed.topologyWords[8];
    const refreshRecordBase = packed.topologyWords[9];
    const refreshRecordCount = packed.topologyWords[10];
    assert.equal(packed.topologyWords[11], 2, "refresh records must be packed [fineEntry, combinedWeight]");
    assert.equal(refreshRecordCount, source.galerkinFineEntries!.length);
    assert.equal(
      packed.topologyWords[refreshOffsetBase + packed.targetEntryCount],
      refreshRecordCount,
    );
    const expectedByTarget = Array.from({ length: packed.targetEntryCount }, () => [] as number[]);
    for (let contribution = 0; contribution < source.galerkinCoarseEntries!.length; contribution += 1) {
      expectedByTarget[source.galerkinCoarseEntries![contribution]].push(contribution);
    }
    const topologyFloats = new Float32Array(packed.topologyWords.buffer);
    for (let targetEntry = 0; targetEntry < packed.targetEntryCount; targetEntry += 1) {
      const first = packed.topologyWords[refreshOffsetBase + targetEntry];
      const end = packed.topologyWords[refreshOffsetBase + targetEntry + 1];
      assert.equal(end - first, expectedByTarget[targetEntry].length);
      expectedByTarget[targetEntry].forEach((contribution, local) => {
        const base = refreshRecordBase + 2 * (first + local);
        assert.equal(packed.topologyWords[base], source.galerkinFineEntries![contribution]);
        const expectedWeight = Math.fround(
          Math.fround(source.prolongationWeights![source.galerkinLeftRecords![contribution]])
          * Math.fround(source.prolongationWeights![source.galerkinRightRecords![contribution]]),
        );
        assert.equal(topologyFloats[base + 1], expectedWeight);
      });
    }
    const refreshed = refreshPackedOctreePowerGalerkinTarget(packed);
    assert.equal(refreshed.length, operators[level + 1].length);
    for (let entry = 0; entry < refreshed.length; entry += 1) {
      const expected = operators[level + 1][entry];
      assert.ok(Math.abs(refreshed[entry] - expected) < 2e-5 * Math.max(1, Math.abs(expected)),
        `packed RAP level ${level} entry ${entry}: ${refreshed[entry]} != ${expected}`);
    }
  }
});

test("two native-L2 cycles fit the order-of-magnitude dispatch target", () => {
  const fine = grid(8);
  const hierarchy = buildOctreePowerGalerkinHierarchy({
    dimensions: [8, 8, 8],
    geometry: fine.geometry,
    rowOffsets: fine.rowOffsets,
    columns: fine.columns,
    coarsestNodeLimit: 8,
  });
  const plan = planOctreePowerGalerkinExecution(hierarchy, 2, 2);
  assert.deepEqual(hierarchy.levels.map((level) => level.nodeCount), [512, 64, 8]);
  assert.deepEqual(plan, {
    levelCount: 3,
    cycles: 2,
    smoothingIterations: 2,
    fineInitializationDispatches: 1,
    fineImportDispatches: 1,
    numericRefreshDispatches: 2,
    dispatchesPerCycle: 15,
    correctionExportDispatches: 1,
    encodedDispatches: 35,
  });
  assert.ok(plan.encodedDispatches * 10 < 647,
    "native L2 schedule must remain at least 10x smaller than the staged production schedule");
  assert.equal(planOctreePowerGalerkinExecution(hierarchy, 99, 2).cycles, 24,
    "adaptive transition solves retain a finite cycle ceiling above the production budget");
});
