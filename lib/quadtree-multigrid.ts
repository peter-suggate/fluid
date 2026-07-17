/**
 * Symbolic geometric hierarchy for the adaptive tall-cell pressure operator.
 *
 * Pressure samples inherit the dyadic x/z footprint of their quadtree leaf.
 * A multigrid level maps each sample to the ancestor covering the level's
 * horizontal scale.  We deliberately semicoarsen in x/z first: the pressure
 * operator is strongly anisotropic in a tall water column, so vertical modes
 * stay available to the line smoother until the horizontal tree is exhausted.
 * Only then are y bins doubled to keep the one-workgroup coarse solve bounded.
 *
 * P is unsmoothed, piecewise-constant aggregation.  R is exactly P^T and the
 * numeric GPU path assembles every coarse matrix as P^T A P after the live
 * free-surface coefficients have been refreshed.
 */

export interface QuadtreeMultigridGeometry {
  x: number;
  y: number;
  z: number;
  horizontalSpan: number;
  verticalSpan: number;
}

export interface QuadtreeMultigridLevel {
  nodeCount: number;
  /** Symbolic CSR. Level zero refers to the projection's four-word CSR. */
  rowOffsets: Uint32Array;
  columns: Uint32Array;
  geometry: QuadtreeMultigridGeometry[];
  lineOffsets: Uint32Array;
  lineNodes: Uint32Array;
  /** Fine node -> node on the next level. Absent on the coarsest level. */
  nodeToCoarse?: Uint32Array;
  /** Fine CSR entry -> CSR entry on the next level. */
  entryToCoarse?: Uint32Array;
  /** Inverse transfer CSR: coarse node -> its fine aggregate. */
  coarseNodeOffsets?: Uint32Array;
  coarseNodes?: Uint32Array;
  /** Two-word [column, coefficient] CSR uploaded for levels above zero. */
  matrixWords?: Uint32Array;
}

export interface QuadtreeMultigridHierarchy {
  levels: QuadtreeMultigridLevel[];
  coarsestNodeCount: number;
}

export interface QuadtreeMultigridOptions {
  coarsestNodeLimit?: number;
  maximumLevels?: number;
}

const nextPowerOfTwo = (value: number) => {
  let result = 1;
  while (result < Math.max(1, value)) result *= 2;
  return result;
};

function linesFor(geometry: QuadtreeMultigridGeometry[]) {
  const byColumn = new Map<string, number[]>();
  geometry.forEach((node, index) => {
    const key = `${node.x}:${node.z}:${node.horizontalSpan}`;
    const line = byColumn.get(key);
    if (line) line.push(index); else byColumn.set(key, [index]);
  });
  const columns = [...byColumn.values()];
  for (const line of columns) line.sort((a, b) => geometry[a].y - geometry[b].y);
  // A whole-column Thomas solve is an excellent serial preconditioner but a
  // poor GPU smoother: one invocation can walk hundreds of dependent rows.
  // Two-sample blocks are the smallest stable vertical block-Jacobi smoother
  // measured on the dam-break hierarchy: scalar Jacobi leaves a near-null
  // alternating vertical mode and makes PCG stall.
  const lines = columns.flatMap((line) => {
    const blocks: number[][] = [];
    for (let first = 0; first < line.length; first += 2) blocks.push(line.slice(first, first + 2));
    return blocks;
  });
  // Stable geometry order makes the symbolic pack deterministic across the
  // worker and main-thread reference implementations.
  lines.sort((a, b) => {
    const ga = geometry[a[0]], gb = geometry[b[0]];
    return ga.z - gb.z || ga.x - gb.x || ga.horizontalSpan - gb.horizontalSpan;
  });
  const offsets = new Uint32Array(lines.length + 1);
  let count = 0;
  lines.forEach((line, index) => { offsets[index] = count; count += line.length; });
  offsets[lines.length] = count;
  const nodes = new Uint32Array(count);
  let cursor = 0;
  for (const line of lines) { nodes.set(line, cursor); cursor += line.length; }
  return { offsets, nodes };
}

function coarseGeometry(
  fine: QuadtreeMultigridGeometry[],
  horizontalScale: number,
  verticalScale: number,
) {
  const coarse: QuadtreeMultigridGeometry[] = [];
  const nodeToCoarse = new Uint32Array(fine.length);
  const ids = new Map<string, number>();
  fine.forEach((node, index) => {
    const span = Math.max(node.horizontalSpan, horizontalScale);
    const ySpan = Math.max(node.verticalSpan, verticalScale);
    const x = Math.floor(node.x / span) * span;
    const z = Math.floor(node.z / span) * span;
    const y = Math.floor(node.y / ySpan) * ySpan;
    const key = `${x}:${y}:${z}:${span}:${ySpan}`;
    let coarseId = ids.get(key);
    if (coarseId === undefined) {
      coarseId = coarse.length; ids.set(key, coarseId);
      coarse.push({ x, y, z, horizontalSpan: span, verticalSpan: ySpan });
    }
    nodeToCoarse[index] = coarseId;
  });
  return { coarse, nodeToCoarse };
}

function coarseSparsity(rowOffsets: Uint32Array, columns: Uint32Array, nodeToCoarse: Uint32Array, coarseCount: number) {
  const rows = Array.from({ length: coarseCount }, () => new Set<number>());
  for (let row = 0; row < nodeToCoarse.length; row += 1) {
    const coarseRow = nodeToCoarse[row];
    rows[coarseRow].add(coarseRow);
    for (let entry = rowOffsets[row]; entry < rowOffsets[row + 1]; entry += 1) rows[coarseRow].add(nodeToCoarse[columns[entry]]);
  }
  const coarseOffsets = new Uint32Array(coarseCount + 1);
  let entryCount = 0;
  const sortedRows = rows.map((row, index) => {
    row.add(index);
    const result = [...row].sort((a, b) => a - b);
    entryCount += result.length;
    coarseOffsets[index + 1] = entryCount;
    return result;
  });
  const coarseColumns = new Uint32Array(entryCount);
  const coarseEntryByPair = new Map<string, number>();
  let cursor = 0;
  sortedRows.forEach((row, coarseRow) => {
    for (const column of row) {
      coarseColumns[cursor] = column;
      coarseEntryByPair.set(`${coarseRow}:${column}`, cursor++);
    }
  });
  const entryToCoarse = new Uint32Array(columns.length);
  for (let row = 0; row < nodeToCoarse.length; row += 1) {
    const coarseRow = nodeToCoarse[row];
    for (let entry = rowOffsets[row]; entry < rowOffsets[row + 1]; entry += 1) {
      const target = coarseEntryByPair.get(`${coarseRow}:${nodeToCoarse[columns[entry]]}`);
      if (target === undefined) throw new Error("Missing Galerkin coarse entry");
      entryToCoarse[entry] = target;
    }
  }
  return { rowOffsets: coarseOffsets, columns: coarseColumns, entryToCoarse };
}

function inverseTransfer(nodeToCoarse: Uint32Array, coarseCount: number) {
  const offsets = new Uint32Array(coarseCount + 1);
  for (const parent of nodeToCoarse) offsets[parent + 1] += 1;
  for (let index = 0; index < coarseCount; index += 1) offsets[index + 1] += offsets[index];
  const nodes = new Uint32Array(nodeToCoarse.length), cursor = offsets.slice(0, coarseCount);
  nodeToCoarse.forEach((parent, node) => { nodes[cursor[parent]++] = node; });
  return { offsets, nodes };
}

function packedCoarseMatrix(rowOffsets: Uint32Array, columns: Uint32Array) {
  const words = new Uint32Array(rowOffsets.length + 2 * columns.length);
  words.set(rowOffsets);
  const base = rowOffsets.length;
  for (let entry = 0; entry < columns.length; entry += 1) words[base + 2 * entry] = columns[entry];
  return words;
}

/** Build a hierarchy directly from the projection's packed fine CSR/aux data. */
export function buildQuadtreeMultigridHierarchy(
  fineMatrixWords: Uint32Array,
  dofCount: number,
  factorAuxWords: Uint32Array,
  dofSamplesBase: number,
  nx: number,
  ny: number,
  nz: number,
  options: QuadtreeMultigridOptions = {},
): QuadtreeMultigridHierarchy {
  const coarsestNodeLimit = Math.max(1, options.coarsestNodeLimit ?? 96);
  const maximumLevels = Math.max(1, options.maximumLevels ?? 12);
  const fineOffsets = fineMatrixWords.slice(0, dofCount + 1);
  const fineColumns = new Uint32Array(fineOffsets[dofCount] ?? 0);
  const fineEntryBase = dofCount + 1;
  for (let entry = 0; entry < fineColumns.length; entry += 1) fineColumns[entry] = fineMatrixWords[fineEntryBase + 4 * entry];
  const fineGeometry: QuadtreeMultigridGeometry[] = [];
  for (let row = 0; row < dofCount; row += 1) {
    const base = dofSamplesBase + 4 * row;
    const packed = factorAuxWords[base] ?? 0;
    fineGeometry.push({
      x: packed & 1023,
      y: packed >>> 20,
      z: (packed >>> 10) & 1023,
      horizontalSpan: Math.max(1, factorAuxWords[base + 1] ?? 1),
      verticalSpan: 1,
    });
  }
  const fineLines = linesFor(fineGeometry);
  const levels: QuadtreeMultigridLevel[] = [{
    nodeCount: dofCount,
    rowOffsets: fineOffsets,
    columns: fineColumns,
    geometry: fineGeometry,
    lineOffsets: fineLines.offsets,
    lineNodes: fineLines.nodes,
  }];
  if (dofCount <= coarsestNodeLimit) return { levels, coarsestNodeCount: dofCount };

  const horizontalLimit = nextPowerOfTwo(Math.max(nx, nz));
  let horizontalScale = 1;
  let verticalScale = 1;
  while (levels.length < maximumLevels && levels.at(-1)!.nodeCount > coarsestNodeLimit) {
    if (horizontalScale < horizontalLimit) horizontalScale *= 2;
    else verticalScale = Math.min(nextPowerOfTwo(ny), verticalScale * 2);
    const fine = levels.at(-1)!;
    const aggregated = coarseGeometry(fine.geometry, horizontalScale, verticalScale);
    if (aggregated.coarse.length === fine.nodeCount) {
      if (horizontalScale >= horizontalLimit && verticalScale >= nextPowerOfTwo(ny)) break;
      continue;
    }
    const sparsity = coarseSparsity(fine.rowOffsets, fine.columns, aggregated.nodeToCoarse, aggregated.coarse.length);
    const inverse = inverseTransfer(aggregated.nodeToCoarse, aggregated.coarse.length);
    fine.nodeToCoarse = aggregated.nodeToCoarse;
    fine.entryToCoarse = sparsity.entryToCoarse;
    fine.coarseNodeOffsets = inverse.offsets;
    fine.coarseNodes = inverse.nodes;
    const coarseLines = linesFor(aggregated.coarse);
    levels.push({
      nodeCount: aggregated.coarse.length,
      rowOffsets: sparsity.rowOffsets,
      columns: sparsity.columns,
      geometry: aggregated.coarse,
      lineOffsets: coarseLines.offsets,
      lineNodes: coarseLines.nodes,
      matrixWords: packedCoarseMatrix(sparsity.rowOffsets, sparsity.columns),
    });
  }
  if (levels.at(-1)!.nodeCount > 128) throw new Error(`Quadtree multigrid coarsest level has ${levels.at(-1)!.nodeCount} DOFs; one-workgroup coarse solve supports at most 128`);
  return { levels, coarsestNodeCount: levels.at(-1)!.nodeCount };
}

/** Dense reference for A_c = P^T A P, used by tests and diagnostics. */
export function galerkinCoarseMatrix(matrix: ArrayLike<number>, fineCount: number, nodeToCoarse: Uint32Array, coarseCount = 1 + Math.max(...nodeToCoarse)) {
  const coarse = new Float64Array(coarseCount * coarseCount);
  for (let row = 0; row < fineCount; row += 1) for (let column = 0; column < fineCount; column += 1) {
    coarse[nodeToCoarse[row] * coarseCount + nodeToCoarse[column]] += matrix[row * fineCount + column];
  }
  return coarse;
}

export function prolongatePiecewiseConstant(coarse: ArrayLike<number>, nodeToCoarse: Uint32Array) {
  return Float64Array.from(nodeToCoarse, (parent) => coarse[parent]);
}

export function restrictTranspose(fine: ArrayLike<number>, nodeToCoarse: Uint32Array, coarseCount = 1 + Math.max(...nodeToCoarse)) {
  const coarse = new Float64Array(coarseCount);
  for (let row = 0; row < nodeToCoarse.length; row += 1) coarse[nodeToCoarse[row]] += fine[row];
  return coarse;
}

function denseProduct(matrix: ArrayLike<number>, vector: ArrayLike<number>) {
  const count = vector.length, result = new Float64Array(count);
  for (let row = 0; row < count; row += 1) for (let column = 0; column < count; column += 1) result[row] += matrix[row * count + column] * vector[column];
  return result;
}

function blockJacobi(level: QuadtreeMultigridLevel, matrix: ArrayLike<number>, rhs: ArrayLike<number>, omega: number) {
  const result = new Float64Array(level.nodeCount);
  for (let line = 0; line + 1 < level.lineOffsets.length; line += 1) {
    const first = level.lineOffsets[line], end = level.lineOffsets[line + 1], count = end - first;
    if (count === 0) continue;
    const nodes = level.lineNodes.slice(first, end);
    // Runtime blocks contain at most two nodes. Keep the dense reference
    // general so changing the block width cannot silently weaken this oracle.
    const lower = new Float64Array(count * count);
    for (let row = 0; row < count; row += 1) for (let column = 0; column <= row; column += 1) {
      let value = matrix[nodes[row] * level.nodeCount + nodes[column]];
      for (let k = 0; k < column; k += 1) value -= lower[row * count + k] * lower[column * count + k];
      lower[row * count + column] = row === column ? Math.sqrt(Math.max(1e-12, value)) : value / lower[column * count + column];
    }
    const y = new Float64Array(count), x = new Float64Array(count);
    for (let row = 0; row < count; row += 1) {
      let value = rhs[nodes[row]];
      for (let column = 0; column < row; column += 1) value -= lower[row * count + column] * y[column];
      y[row] = value / lower[row * count + row];
    }
    for (let reverse = count; reverse > 0; reverse -= 1) {
      const row = reverse - 1; let value = y[row];
      for (let column = row + 1; column < count; column += 1) value -= lower[column * count + row] * x[column];
      x[row] = value / lower[row * count + row];
    }
    for (let row = 0; row < count; row += 1) result[nodes[row]] = omega * x[row];
  }
  return result;
}

/**
 * Dense oracle for the exact transfer/smoothing order used by the GPU V-cycle.
 * It is intentionally small-matrix code for invariant tests, not a runtime
 * fallback. The coarsest fixed Jacobi polynomial matches the GPU's linear,
 * symmetric zero-start iteration.
 */
export function applyQuadtreeMultigridVcycle(
  fineMatrix: ArrayLike<number>,
  hierarchy: QuadtreeMultigridHierarchy,
  rhs: ArrayLike<number>,
  omega = 0.7,
  coarseIterations = 8,
) {
  const matrices: Float64Array[] = [Float64Array.from(fineMatrix)];
  for (let level = 0; level + 1 < hierarchy.levels.length; level += 1) {
    const source = hierarchy.levels[level], target = hierarchy.levels[level + 1];
    matrices.push(galerkinCoarseMatrix(matrices[level], source.nodeCount, source.nodeToCoarse!, target.nodeCount));
  }
  const visit = (levelIndex: number, rightHandSide: ArrayLike<number>): Float64Array => {
    const level = hierarchy.levels[levelIndex], matrix = matrices[levelIndex];
    if (levelIndex + 1 === hierarchy.levels.length) {
      const correction = new Float64Array(level.nodeCount);
      for (let iteration = 0; iteration < coarseIterations; iteration += 1) {
        const product = denseProduct(matrix, correction);
        for (let row = 0; row < level.nodeCount; row += 1) correction[row] += omega * (rightHandSide[row] - product[row]) / Math.max(1e-12, matrix[row * level.nodeCount + row]);
      }
      return correction;
    }
    const correction = blockJacobi(level, matrix, rightHandSide, omega);
    const product = denseProduct(matrix, correction);
    const defect = Float64Array.from(rightHandSide, (value, row) => value - product[row]);
    const coarseRhs = restrictTranspose(defect, level.nodeToCoarse!, hierarchy.levels[levelIndex + 1].nodeCount);
    const coarseCorrection = visit(levelIndex + 1, coarseRhs);
    const prolonged = prolongatePiecewiseConstant(coarseCorrection, level.nodeToCoarse!);
    for (let row = 0; row < level.nodeCount; row += 1) correction[row] += prolonged[row];
    const correctedProduct = denseProduct(matrix, correction);
    const finalDefect = Float64Array.from(rightHandSide, (value, row) => value - correctedProduct[row]);
    const post = blockJacobi(level, matrix, finalDefect, omega);
    for (let row = 0; row < level.nodeCount; row += 1) correction[row] += post[row];
    return correction;
  };
  return visit(0, rhs);
}
