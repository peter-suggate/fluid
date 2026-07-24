/**
 * Experimental symbolic aggregation hierarchy for the native L2 power graph.
 *
 * Production P is sparse cell-centred trilinear interpolation and R is exactly
 * P^T, so each coarse operator is A_c = P^T A P. A piecewise-constant transfer
 * remains available as a bounded comparison oracle. The trilinear Galerkin
 * fill is expanded once into a deterministic contribution map; that symbolic
 * map can be cached by topology generation while changing coefficients are
 * refreshed in one parent-owned gather per level.
 *
 * Aanjaneya et al. Section 4.3 does not use this hierarchy: it applies CG to
 * L2 with boundary-only L2 Jacobi around the first-order SPGrid V-cycle M1.
 * This module implements the explicitly selected native-L2 alternative. The
 * UI supplies its immutable hierarchy at construction; the retained paper
 * implementation is a separate selection and is never used as a fallback.
 */

export interface OctreePowerGalerkinGeometry {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  /** Power-of-two leaf/aggregate width in finest cells. */
  readonly span: number;
}

export interface OctreePowerGalerkinInput {
  readonly dimensions: readonly [number, number, number];
  readonly geometry: readonly OctreePowerGalerkinGeometry[];
  /** Fine off-diagonal CSR. Diagonals are supplied separately at refresh. */
  readonly rowOffsets: Uint32Array;
  readonly columns: Uint32Array;
  /** Production uses trilinear transfer; piecewise constant remains useful as
   * a bounded comparison/oracle for aggregation-only variants. */
  readonly transfer?: "trilinear" | "piecewise-constant";
  readonly coarsestNodeLimit?: number;
  readonly maximumLevels?: number;
}

export interface OctreePowerGalerkinLevel {
  readonly nodeCount: number;
  /** Full signed-matrix CSR, including one explicit diagonal per row. */
  readonly rowOffsets: Uint32Array;
  readonly columns: Uint32Array;
  readonly geometry: readonly OctreePowerGalerkinGeometry[];
  /** Sparse fine-by-coarse P. R traverses the same records in transpose. */
  readonly prolongationOffsets?: Uint32Array;
  readonly prolongationColumns?: Uint32Array;
  readonly prolongationWeights?: Float64Array;
  /** Fine node -> next-level aggregate. */
  readonly nodeToCoarse?: Uint32Array;
  /** Fine full-CSR entry -> next-level full-CSR entry. */
  readonly entryToCoarse?: Uint32Array;
  /** Inverse P: next-level aggregate -> its fine nodes. */
  readonly coarseNodeOffsets?: Uint32Array;
  readonly coarseNodes?: Uint32Array;
  /** Expanded symbolic products. Each record contributes
   * A[fineEntry] * P[left] * P[right] to A_c[coarseEntry]. */
  readonly galerkinFineEntries?: Uint32Array;
  readonly galerkinLeftRecords?: Uint32Array;
  readonly galerkinRightRecords?: Uint32Array;
  readonly galerkinCoarseEntries?: Uint32Array;
}

export interface OctreePowerGalerkinHierarchy {
  readonly dimensions: readonly [number, number, number];
  readonly levels: readonly OctreePowerGalerkinLevel[];
  /** Fine row -> explicit diagonal entry in level zero. */
  readonly fineDiagonalEntries: Uint32Array;
  /** Input off-diagonal entry -> full-CSR entry in level zero. */
  readonly fineOffDiagonalEntries: Uint32Array;
  readonly symbolicEntryCount: number;
  readonly symbolicContributionCount: number;
}

export interface OctreePowerGalerkinExecutionPlan {
  readonly levelCount: number;
  readonly cycles: number;
  readonly smoothingIterations: number;
  readonly fineInitializationDispatches: 1;
  readonly fineImportDispatches: 1;
  readonly fineOperatorChangeDetectionDispatches: 1;
  readonly numericRefreshDispatches: number;
  readonly dispatchesPerCycle: number;
  readonly correctionExportDispatches: 1;
  readonly encodedDispatches: number;
}

/**
 * Immutable production command target. A complete bounded one-workgroup
 * bottom solve is one dispatch; each non-bottom level owns matching pre/post
 * sweeps plus one restriction and one prolongation. The fixed fine matrix is
 * reset before compact import, and successful correction is gathered back to
 * compact row order after convergence publication.
 */
export function planOctreePowerGalerkinExecution(
  hierarchy: OctreePowerGalerkinHierarchy,
  cycles = 2,
  smoothingIterations = 2,
): OctreePowerGalerkinExecutionPlan {
  const levelCount = positiveInteger(hierarchy.levels.length, "power Galerkin level count");
  const boundedCycles = Math.max(1, Math.min(24, Math.round(cycles)));
  // GPU vector publication is deliberately fixed to channel A. Matching
  // A→B/B→A pairs preserve both that invariant and a symmetric V-cycle.
  const smoothing = Math.max(2, Math.min(4, Math.round(smoothingIterations / 2) * 2));
  const adjacent = levelCount - 1;
  const numericRefreshDispatches = adjacent;
  // Each statically encoded cycle ends with a two-stage residual checkpoint.
  // The bounded partial reduction now publishes convergence in its final
  // lane, so publication no longer costs a third singleton dispatch. Once the
  // checkpoint publishes convergence, later cycle dispatches remain encoded
  // but their kernels skip device work through the shared solve control.
  const dispatchesPerCycle = 2 * adjacent * (smoothing + 1) + 3;
  return Object.freeze({
    levelCount,
    cycles: boundedCycles,
    smoothingIterations: smoothing,
    fineInitializationDispatches: 1,
    fineImportDispatches: 1,
    fineOperatorChangeDetectionDispatches: 1,
    numericRefreshDispatches,
    dispatchesPerCycle,
    correctionExportDispatches: 1,
    // Fine initialization clears the correction channels in the same row
    // dispatch; import, refresh, checkpointed cycles, and accepted-correction
    // export account for the remaining commands.
    encodedDispatches: 3 + numericRefreshDispatches + boundedCycles * dispatchesPerCycle + 1,
  });
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < Math.max(1, value)) result *= 2;
  return result;
}

function validateGeometry(node: OctreePowerGalerkinGeometry, index: number): void {
  positiveInteger(node.span, `power Galerkin node ${index} span`);
  if ((node.span & (node.span - 1)) !== 0) throw new RangeError(`power Galerkin node ${index} span must be a power of two`);
  for (const [axis, value] of [node.x, node.y, node.z].entries()) {
    if (!Number.isSafeInteger(value) || value < 0 || value % node.span !== 0) {
      throw new RangeError(`power Galerkin node ${index} axis ${axis} is not aligned to its span`);
    }
  }
}

function materializeFineSparsity(input: OctreePowerGalerkinInput): {
  rowOffsets: Uint32Array;
  columns: Uint32Array;
  diagonalEntries: Uint32Array;
  offDiagonalEntries: Uint32Array;
} {
  const count = input.geometry.length;
  if (input.rowOffsets.length !== count + 1 || input.rowOffsets[0] !== 0
    || input.rowOffsets[count] !== input.columns.length) {
    throw new RangeError("power Galerkin fine CSR dimensions disagree");
  }
  const rows: number[][] = [];
  const pairEntries = new Map<string, number>();
  const rowOffsets = new Uint32Array(count + 1);
  let entryCount = 0;
  for (let row = 0; row < count; row += 1) {
    const columns = new Set<number>([row]);
    for (let entry = input.rowOffsets[row]; entry < input.rowOffsets[row + 1]; entry += 1) {
      const column = input.columns[entry];
      if (column >= count) throw new RangeError(`power Galerkin fine entry ${entry} has an invalid column`);
      columns.add(column);
    }
    const sorted = [...columns].sort((a, b) => a - b);
    rows.push(sorted); entryCount += sorted.length; rowOffsets[row + 1] = entryCount;
  }
  const packedColumns = new Uint32Array(entryCount);
  let cursor = 0;
  rows.forEach((columns, row) => {
    for (const column of columns) {
      packedColumns[cursor] = column;
      pairEntries.set(`${row}:${column}`, cursor++);
    }
  });
  const diagonalEntries = Uint32Array.from({ length: count }, (_, row) => pairEntries.get(`${row}:${row}`)!);
  const offDiagonalEntries = new Uint32Array(input.columns.length);
  for (let row = 0; row < count; row += 1) {
    for (let entry = input.rowOffsets[row]; entry < input.rowOffsets[row + 1]; entry += 1) {
      offDiagonalEntries[entry] = pairEntries.get(`${row}:${input.columns[entry]}`)!;
    }
  }
  return { rowOffsets, columns: packedColumns, diagonalEntries, offDiagonalEntries };
}

function aggregateGeometry(
  fine: readonly OctreePowerGalerkinGeometry[],
  scale: number,
): { geometry: OctreePowerGalerkinGeometry[]; nodeToCoarse: Uint32Array } {
  const geometry: OctreePowerGalerkinGeometry[] = [];
  const nodeToCoarse = new Uint32Array(fine.length);
  const ids = new Map<string, number>();
  fine.forEach((node, fineNode) => {
    const span = Math.max(node.span, scale);
    const x = Math.floor(node.x / span) * span;
    const y = Math.floor(node.y / span) * span;
    const z = Math.floor(node.z / span) * span;
    const key = `${x}:${y}:${z}:${span}`;
    let coarse = ids.get(key);
    if (coarse === undefined) {
      coarse = geometry.length;
      ids.set(key, coarse);
      geometry.push({ x, y, z, span });
    }
    nodeToCoarse[fineNode] = coarse;
  });
  return { geometry, nodeToCoarse };
}

function materializeTransfer(
  fine: OctreePowerGalerkinLevel,
  coarseGeometry: readonly OctreePowerGalerkinGeometry[],
  nodeToCoarse: Uint32Array,
  scale: number,
  kind: "trilinear" | "piecewise-constant",
): { offsets: Uint32Array; columns: Uint32Array; weights: Float64Array } {
  const offsets = new Uint32Array(fine.nodeCount + 1);
  const columns: number[] = [], weights: number[] = [];
  if (kind === "piecewise-constant") {
    for (let row = 0; row < fine.nodeCount; row += 1) {
      columns.push(nodeToCoarse[row]); weights.push(1); offsets[row + 1] = columns.length;
    }
    return { offsets, columns: Uint32Array.from(columns), weights: Float64Array.from(weights) };
  }
  const coarseByCoordinate = new Map(coarseGeometry.map((node, index) =>
    [`${node.x / node.span}:${node.y / node.span}:${node.z / node.span}:${node.span}`, index]));
  fine.geometry.forEach((node, row) => {
    if (node.span >= scale) {
      columns.push(nodeToCoarse[row]); weights.push(1); offsets[row + 1] = columns.length; return;
    }
    const lower = [0, 0, 0], fraction = [0, 0, 0];
    for (let axis = 0; axis < 3; axis += 1) {
      const origin = [node.x, node.y, node.z][axis];
      const coordinate = (origin + 0.5 * node.span) / scale - 0.5;
      lower[axis] = Math.floor(coordinate); fraction[axis] = coordinate - lower[axis];
    }
    const candidates = new Map<number, number>();
    for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) for (let dx = 0; dx < 2; dx += 1) {
      const key = `${lower[0] + dx}:${lower[1] + dy}:${lower[2] + dz}:${scale}`;
      const coarse = coarseByCoordinate.get(key);
      if (coarse === undefined) continue;
      const weight = (dx ? fraction[0] : 1 - fraction[0])
        * (dy ? fraction[1] : 1 - fraction[1])
        * (dz ? fraction[2] : 1 - fraction[2]);
      if (weight > 0) candidates.set(coarse, (candidates.get(coarse) ?? 0) + weight);
    }
    if (candidates.size === 0) candidates.set(nodeToCoarse[row], 1);
    const sum = [...candidates.values()].reduce((total, value) => total + value, 0);
    for (const [coarse, weight] of [...candidates].sort(([a], [b]) => a - b)) {
      columns.push(coarse); weights.push(weight / sum);
    }
    offsets[row + 1] = columns.length;
  });
  return { offsets, columns: Uint32Array.from(columns), weights: Float64Array.from(weights) };
}

function coarseSparsity(
  fine: OctreePowerGalerkinLevel,
  transfer: { offsets: Uint32Array; columns: Uint32Array; weights: Float64Array },
  coarseCount: number,
): {
  rowOffsets: Uint32Array;
  columns: Uint32Array;
  entryToCoarse?: Uint32Array;
  fineEntries: Uint32Array;
  leftRecords: Uint32Array;
  rightRecords: Uint32Array;
  coarseEntries: Uint32Array;
} {
  const rows = Array.from({ length: coarseCount }, (_, row) => new Set<number>([row]));
  const contributions: Array<readonly [number, number, number, number, number]> = [];
  for (let row = 0; row < fine.nodeCount; row += 1) {
    for (let entry = fine.rowOffsets[row]; entry < fine.rowOffsets[row + 1]; entry += 1) {
      const column = fine.columns[entry];
      for (let left = transfer.offsets[row]; left < transfer.offsets[row + 1]; left += 1) {
        const coarseRow = transfer.columns[left];
        for (let right = transfer.offsets[column]; right < transfer.offsets[column + 1]; right += 1) {
          const coarseColumn = transfer.columns[right];
          rows[coarseRow].add(coarseColumn);
          contributions.push([entry, left, right, coarseRow, coarseColumn]);
        }
      }
    }
  }
  const rowOffsets = new Uint32Array(coarseCount + 1);
  const sortedRows = rows.map((row, index) => {
    const sorted = [...row].sort((a, b) => a - b);
    rowOffsets[index + 1] = rowOffsets[index] + sorted.length;
    return sorted;
  });
  const columns = new Uint32Array(rowOffsets[coarseCount]);
  const entries = new Map<string, number>();
  let cursor = 0;
  sortedRows.forEach((row, coarseRow) => {
    for (const column of row) {
      columns[cursor] = column;
      entries.set(`${coarseRow}:${column}`, cursor++);
    }
  });
  const fineEntries = new Uint32Array(contributions.length);
  const leftRecords = new Uint32Array(contributions.length);
  const rightRecords = new Uint32Array(contributions.length);
  const coarseEntries = new Uint32Array(contributions.length);
  contributions.forEach(([fineEntry, left, right, coarseRow, coarseColumn], index) => {
    fineEntries[index] = fineEntry; leftRecords[index] = left; rightRecords[index] = right;
    coarseEntries[index] = entries.get(`${coarseRow}:${coarseColumn}`)!;
  });
  const piecewise = transfer.columns.length === fine.nodeCount
    && transfer.weights.every((weight) => weight === 1);
  const entryToCoarse = piecewise ? new Uint32Array(fine.columns.length) : undefined;
  if (entryToCoarse) {
    for (let index = 0; index < fineEntries.length; index += 1) {
      entryToCoarse[fineEntries[index]] = coarseEntries[index];
    }
  }
  return { rowOffsets, columns, entryToCoarse, fineEntries, leftRecords, rightRecords, coarseEntries };
}

function inverseTransfer(nodeToCoarse: Uint32Array, coarseCount: number): {
  offsets: Uint32Array;
  nodes: Uint32Array;
} {
  const offsets = new Uint32Array(coarseCount + 1);
  for (const coarse of nodeToCoarse) offsets[coarse + 1] += 1;
  for (let coarse = 0; coarse < coarseCount; coarse += 1) offsets[coarse + 1] += offsets[coarse];
  const nodes = new Uint32Array(nodeToCoarse.length);
  const cursors = offsets.slice(0, coarseCount);
  nodeToCoarse.forEach((coarse, fine) => { nodes[cursors[coarse]++] = fine; });
  return { offsets, nodes };
}

/** Build the reusable symbolic hierarchy from one compact power-row topology. */
export function buildOctreePowerGalerkinHierarchy(input: OctreePowerGalerkinInput): OctreePowerGalerkinHierarchy {
  input.dimensions.forEach((value) => positiveInteger(value, "power Galerkin dimension"));
  if (input.geometry.length < 1) throw new RangeError("power Galerkin hierarchy requires at least one node");
  input.geometry.forEach(validateGeometry);
  const fine = materializeFineSparsity(input);
  const levels: OctreePowerGalerkinLevel[] = [{
    nodeCount: input.geometry.length,
    rowOffsets: fine.rowOffsets,
    columns: fine.columns,
    geometry: Object.freeze(input.geometry.map((node) => ({ ...node }))),
  }];
  const coarsestNodeLimit = Math.max(1, Math.round(input.coarsestNodeLimit ?? 64));
  const maximumLevels = Math.max(1, Math.round(input.maximumLevels ?? 12));
  const transferKind = input.transfer ?? "trilinear";
  const maximumScale = nextPowerOfTwo(Math.max(...input.dimensions));
  let scale = 1;
  while (levels.length < maximumLevels && levels.at(-1)!.nodeCount > coarsestNodeLimit
    && scale < maximumScale) {
    scale *= 2;
    const source = levels.at(-1)!;
    const aggregated = aggregateGeometry(source.geometry, scale);
    if (aggregated.geometry.length === source.nodeCount) continue;
    const transfer = materializeTransfer(source, aggregated.geometry, aggregated.nodeToCoarse, scale, transferKind);
    const sparsity = coarseSparsity(source, transfer, aggregated.geometry.length);
    const inverse = inverseTransfer(aggregated.nodeToCoarse, aggregated.geometry.length);
    Object.assign(source, {
      nodeToCoarse: aggregated.nodeToCoarse,
      entryToCoarse: sparsity.entryToCoarse,
      coarseNodeOffsets: inverse.offsets,
      coarseNodes: inverse.nodes,
      prolongationOffsets: transfer.offsets,
      prolongationColumns: transfer.columns,
      prolongationWeights: transfer.weights,
      galerkinFineEntries: sparsity.fineEntries,
      galerkinLeftRecords: sparsity.leftRecords,
      galerkinRightRecords: sparsity.rightRecords,
      galerkinCoarseEntries: sparsity.coarseEntries,
    });
    levels.push({
      nodeCount: aggregated.geometry.length,
      rowOffsets: sparsity.rowOffsets,
      columns: sparsity.columns,
      geometry: Object.freeze(aggregated.geometry),
    });
  }
  return Object.freeze({
    dimensions: Object.freeze([...input.dimensions]) as unknown as readonly [number, number, number],
    levels: Object.freeze(levels),
    fineDiagonalEntries: fine.diagonalEntries,
    fineOffDiagonalEntries: fine.offDiagonalEntries,
    symbolicEntryCount: levels.reduce((sum, level) => sum + level.columns.length, 0),
    symbolicContributionCount: levels.slice(0, -1)
      .reduce((sum, level) => sum + level.galerkinFineEntries!.length, 0),
  });
}

/**
 * Build the immutable full-domain hierarchy used by the UI Galerkin lane.
 * Rows are indexed by flattened finest-cell index, which gives the GPU a
 * direct, readback-free mapping from each live compact LeafHeader.cell.
 * Liquid membership and coefficients may change; the supplied hierarchy may
 * not.
 */
export function buildFixedUniformOctreePowerGalerkinHierarchy(
  dimensions: readonly [number, number, number],
  options: Pick<OctreePowerGalerkinInput, "transfer" | "coarsestNodeLimit" | "maximumLevels"> = {},
): OctreePowerGalerkinHierarchy {
  dimensions.forEach((value) => positiveInteger(value, "fixed power Galerkin dimension"));
  const [nx, ny, nz] = dimensions;
  const count = nx * ny * nz;
  if (!Number.isSafeInteger(count) || count < 1) {
    throw new RangeError("fixed power Galerkin domain is too large");
  }
  const geometry = Array.from({ length: count }, (_, cell) => ({
    x: cell % nx,
    y: Math.floor(cell / nx) % ny,
    z: Math.floor(cell / (nx * ny)),
    span: 1,
  }));
  const rows: number[][] = Array.from({ length: count }, () => []);
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const row = rows[index(x, y, z)];
        if (x > 0) row.push(index(x - 1, y, z));
        if (x + 1 < nx) row.push(index(x + 1, y, z));
        if (y > 0) row.push(index(x, y - 1, z));
        if (y + 1 < ny) row.push(index(x, y + 1, z));
        if (z > 0) row.push(index(x, y, z - 1));
        if (z + 1 < nz) row.push(index(x, y, z + 1));
      }
    }
  }
  const rowOffsets = new Uint32Array(count + 1);
  for (let row = 0; row < count; row += 1) {
    rowOffsets[row + 1] = rowOffsets[row] + rows[row].length;
  }
  return buildOctreePowerGalerkinHierarchy({
    dimensions,
    geometry,
    rowOffsets,
    columns: Uint32Array.from(rows.flat()),
    transfer: options.transfer,
    coarsestNodeLimit: options.coarsestNodeLimit,
    maximumLevels: options.maximumLevels,
  });
}

/**
 * Build a fixed GPU arena containing every dyadic cell that can occur in an
 * adaptive frontier up to `maximumLeafSize`. Overlapping candidates are never
 * simultaneously live: the fine matrix is the zero extension of the current
 * leaf frontier into this immutable superset.
 *
 * The graph contains every face/edge contact allowed by ordinary 2:1 grading.
 * Runtime import activates only the exact power faces published by the current
 * topology, so harmless superset entries remain zero. This preserves a static
 * command graph without replacing the paper's adaptive pressure variables by
 * a unit-cell grid.
 */
export function buildFixedAdaptiveOctreePowerGalerkinHierarchy(
  dimensions: readonly [number, number, number],
  maximumLeafSize: number,
  options: Pick<OctreePowerGalerkinInput, "transfer" | "coarsestNodeLimit" | "maximumLevels"> = {},
): OctreePowerGalerkinHierarchy {
  dimensions.forEach((value) => positiveInteger(value, "adaptive power Galerkin dimension"));
  positiveInteger(maximumLeafSize, "adaptive power Galerkin maximum leaf size");
  if ((maximumLeafSize & (maximumLeafSize - 1)) !== 0) {
    throw new RangeError("adaptive power Galerkin maximum leaf size must be a power of two");
  }
  const geometry: OctreePowerGalerkinGeometry[] = [];
  for (let span = 1; span <= maximumLeafSize; span *= 2) {
    for (let z = 0; z + span <= dimensions[2]; z += span) {
      for (let y = 0; y + span <= dimensions[1]; y += span) {
        for (let x = 0; x + span <= dimensions[0]; x += span) {
          geometry.push({ x, y, z, span });
        }
      }
    }
  }
  if (geometry.length === 0) throw new RangeError("adaptive power Galerkin arena has no candidate cells");
  const rows: number[][] = Array.from({ length: geometry.length }, () => []);
  const contact = (left: OctreePowerGalerkinGeometry, right: OctreePowerGalerkinGeometry): boolean => {
    let touching = 0, overlapping = 0;
    for (const axis of ["x", "y", "z"] as const) {
      const low = Math.max(left[axis], right[axis]);
      const high = Math.min(left[axis] + left.span, right[axis] + right.span);
      if (high < low) return false;
      if (high === low) touching += 1;
      else overlapping += 1;
    }
    return (touching === 1 && overlapping === 2) || (touching === 2 && overlapping === 1);
  };
  for (let left = 0; left + 1 < geometry.length; left += 1) {
    for (let right = left + 1; right < geometry.length; right += 1) {
      const smaller = Math.min(geometry[left].span, geometry[right].span);
      const larger = Math.max(geometry[left].span, geometry[right].span);
      if (larger > 2 * smaller || !contact(geometry[left], geometry[right])) continue;
      rows[left].push(right);
      rows[right].push(left);
    }
  }
  const rowOffsets = new Uint32Array(geometry.length + 1);
  for (let row = 0; row < geometry.length; row += 1) {
    rows[row].sort((a, b) => a - b);
    rowOffsets[row + 1] = rowOffsets[row] + rows[row].length;
  }
  return buildOctreePowerGalerkinHierarchy({
    dimensions,
    geometry,
    rowOffsets,
    columns: Uint32Array.from(rows.flat()),
    transfer: options.transfer,
    coarsestNodeLimit: options.coarsestNodeLimit,
    maximumLevels: options.maximumLevels,
  });
}

/**
 * Numeric A_c=P^TAP refresh. Fine off-diagonal coefficients use the live
 * power-row convention (positive stored coupling, negative matrix entry).
 */
export function refreshOctreePowerGalerkinOperators(
  hierarchy: OctreePowerGalerkinHierarchy,
  diagonal: ArrayLike<number>,
  offDiagonal: ArrayLike<number>,
): readonly Float64Array[] {
  const finest = hierarchy.levels[0];
  if (diagonal.length !== finest.nodeCount
    || offDiagonal.length !== hierarchy.fineOffDiagonalEntries.length) {
    throw new RangeError("power Galerkin numeric coefficients disagree with the symbolic hierarchy");
  }
  const values: Float64Array[] = [new Float64Array(finest.columns.length)];
  for (let row = 0; row < diagonal.length; row += 1) {
    const value = diagonal[row];
    if (!(value > 0) || !Number.isFinite(value)) throw new RangeError(`power Galerkin diagonal ${row} is invalid`);
    values[0][hierarchy.fineDiagonalEntries[row]] += value;
  }
  for (let entry = 0; entry < offDiagonal.length; entry += 1) {
    const value = offDiagonal[entry];
    if (!(value >= 0) || !Number.isFinite(value)) throw new RangeError(`power Galerkin entry ${entry} is invalid`);
    values[0][hierarchy.fineOffDiagonalEntries[entry]] -= value;
  }
  for (let level = 0; level + 1 < hierarchy.levels.length; level += 1) {
    const source = hierarchy.levels[level];
    const coarse = new Float64Array(hierarchy.levels[level + 1].columns.length);
    const fineEntries = source.galerkinFineEntries!, left = source.galerkinLeftRecords!;
    const right = source.galerkinRightRecords!, targets = source.galerkinCoarseEntries!;
    const weights = source.prolongationWeights!;
    for (let contribution = 0; contribution < fineEntries.length; contribution += 1) {
      coarse[targets[contribution]] += values[level][fineEntries[contribution]]
        * weights[left[contribution]] * weights[right[contribution]];
    }
    values.push(coarse);
  }
  return Object.freeze(values);
}

export function applyOctreePowerGalerkinLevel(
  level: OctreePowerGalerkinLevel,
  values: ArrayLike<number>,
  vector: ArrayLike<number>,
): Float64Array {
  if (values.length !== level.columns.length || vector.length !== level.nodeCount) {
    throw new RangeError("power Galerkin level dimensions disagree");
  }
  const result = new Float64Array(level.nodeCount);
  for (let row = 0; row < level.nodeCount; row += 1) {
    for (let entry = level.rowOffsets[row]; entry < level.rowOffsets[row + 1]; entry += 1) {
      result[row] += values[entry] * vector[level.columns[entry]];
    }
  }
  return result;
}

function diagonalEntry(level: OctreePowerGalerkinLevel, row: number): number {
  for (let entry = level.rowOffsets[row]; entry < level.rowOffsets[row + 1]; entry += 1) {
    if (level.columns[entry] === row) return entry;
  }
  throw new RangeError(`power Galerkin row ${row} has no diagonal`);
}

function jacobiCorrection(
  level: OctreePowerGalerkinLevel,
  values: ArrayLike<number>,
  rhs: ArrayLike<number>,
  initial: ArrayLike<number>,
  iterations: number,
  damping: number,
): Float64Array {
  let correction = Float64Array.from(initial);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const product = applyOctreePowerGalerkinLevel(level, values, correction);
    const next = correction.slice();
    for (let row = 0; row < level.nodeCount; row += 1) {
      const diagonal = values[diagonalEntry(level, row)];
      if (!(diagonal > 0) || !Number.isFinite(diagonal)) throw new RangeError(`power Galerkin row ${row} is not positive`);
      next[row] += damping * (rhs[row] - product[row]) / diagonal;
    }
    correction = next;
  }
  return correction;
}

function solveDenseCoarsest(
  level: OctreePowerGalerkinLevel,
  values: ArrayLike<number>,
  rhs: ArrayLike<number>,
): Float64Array {
  const count = level.nodeCount;
  const lower = new Float64Array(count * count);
  for (let row = 0; row < count; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let value = 0;
      for (let entry = level.rowOffsets[row]; entry < level.rowOffsets[row + 1]; entry += 1) {
        if (level.columns[entry] === column) value += values[entry];
      }
      for (let inner = 0; inner < column; inner += 1) {
        value -= lower[row * count + inner] * lower[column * count + inner];
      }
      if (row === column) {
        if (!(value > 1e-14) || !Number.isFinite(value)) throw new RangeError("power Galerkin coarsest operator is not SPD");
        lower[row * count + column] = Math.sqrt(value);
      } else {
        lower[row * count + column] = value / lower[column * count + column];
      }
    }
  }
  const intermediate = new Float64Array(count);
  for (let row = 0; row < count; row += 1) {
    let value = rhs[row];
    for (let column = 0; column < row; column += 1) value -= lower[row * count + column] * intermediate[column];
    intermediate[row] = value / lower[row * count + row];
  }
  const result = new Float64Array(count);
  for (let reverse = count; reverse > 0; reverse -= 1) {
    const row = reverse - 1;
    let value = intermediate[row];
    for (let column = row + 1; column < count; column += 1) value -= lower[column * count + row] * result[column];
    result[row] = value / lower[row * count + row];
  }
  return result;
}

export interface OctreePowerGalerkinVCycleOptions {
  readonly smoothingIterations?: number;
  readonly damping?: number;
}

/** Symmetric native-L2 V-cycle used as the numerical oracle for GPU parity. */
export function applyOctreePowerGalerkinVCycle(
  hierarchy: OctreePowerGalerkinHierarchy,
  operators: readonly ArrayLike<number>[],
  rhs: ArrayLike<number>,
  options: OctreePowerGalerkinVCycleOptions = {},
): Float64Array {
  if (operators.length !== hierarchy.levels.length || rhs.length !== hierarchy.levels[0].nodeCount) {
    throw new RangeError("power Galerkin V-cycle dimensions disagree");
  }
  const smoothingIterations = Math.max(1, Math.min(8, Math.round(options.smoothingIterations ?? 2)));
  const damping = Math.max(0.05, Math.min(0.95, options.damping ?? 2 / 3));
  const visit = (levelIndex: number, levelRhs: ArrayLike<number>): Float64Array => {
    const level = hierarchy.levels[levelIndex], values = operators[levelIndex];
    if (levelIndex + 1 === hierarchy.levels.length) return solveDenseCoarsest(level, values, levelRhs);
    let correction = jacobiCorrection(level, values, levelRhs, new Float64Array(level.nodeCount),
      smoothingIterations, damping);
    const product = applyOctreePowerGalerkinLevel(level, values, correction);
    const residual = Float64Array.from(levelRhs, (value, row) => value - product[row]);
    const coarseLevel = hierarchy.levels[levelIndex + 1];
    const coarseRhs = new Float64Array(coarseLevel.nodeCount);
    const offsets = level.prolongationOffsets!, columns = level.prolongationColumns!;
    const weights = level.prolongationWeights!;
    for (let fine = 0; fine < level.nodeCount; fine += 1) {
      for (let record = offsets[fine]; record < offsets[fine + 1]; record += 1) {
        coarseRhs[columns[record]] += weights[record] * residual[fine];
      }
    }
    const coarseCorrection = visit(levelIndex + 1, coarseRhs);
    for (let fine = 0; fine < level.nodeCount; fine += 1) {
      for (let record = offsets[fine]; record < offsets[fine + 1]; record += 1) {
        correction[fine] += weights[record] * coarseCorrection[columns[record]];
      }
    }
    correction = jacobiCorrection(level, values, levelRhs, correction, smoothingIterations, damping);
    return correction;
  };
  return visit(0, rhs);
}

export interface PackedOctreePowerGalerkinLevel {
  readonly sourceNodeCount: number;
  readonly targetNodeCount: number;
  readonly sourceEntryCount: number;
  readonly targetEntryCount: number;
  /** Source/target full CSR: offsets, then [column, coefficient] records. */
  readonly sourceMatrixWords: Uint32Array;
  readonly targetMatrixWords: Uint32Array;
  /**
   * Header followed by:
   * - fine-row P offsets and P records [fine, coarse, weight]
   * - R=P^T coarse offsets and P-record indices
   * - coarse-entry refresh offsets
   * - target-major refresh records [fineEntry, P(left)*P(right)]
   */
  readonly topologyWords: Uint32Array;
}

function packMatrix(level: OctreePowerGalerkinLevel, values: ArrayLike<number>): Uint32Array {
  if (values.length !== level.columns.length) throw new RangeError("power Galerkin packed matrix dimensions disagree");
  const words = new Uint32Array(level.nodeCount + 1 + 2 * level.columns.length);
  words.set(level.rowOffsets);
  const floats = new Float32Array(words.buffer);
  const base = level.nodeCount + 1;
  for (let entry = 0; entry < level.columns.length; entry += 1) {
    words[base + 2 * entry] = level.columns[entry];
    floats[base + 2 * entry + 1] = values[entry];
  }
  return words;
}

/**
 * Pack one adjacent-level GPU gather contract. The packed numeric target is an
 * oracle snapshot; production refresh overwrites only its coefficient words.
 */
export function packOctreePowerGalerkinLevel(
  hierarchy: OctreePowerGalerkinHierarchy,
  operators: readonly ArrayLike<number>[],
  levelIndex: number,
): PackedOctreePowerGalerkinLevel {
  if (!Number.isSafeInteger(levelIndex) || levelIndex < 0 || levelIndex + 1 >= hierarchy.levels.length
    || operators.length !== hierarchy.levels.length) {
    throw new RangeError("power Galerkin packed level index is invalid");
  }
  const source = hierarchy.levels[levelIndex], target = hierarchy.levels[levelIndex + 1];
  const pOffsets = source.prolongationOffsets!, pColumns = source.prolongationColumns!;
  const pWeights = source.prolongationWeights!;
  const pRows = new Uint32Array(pColumns.length);
  for (let row = 0; row < source.nodeCount; row += 1) {
    for (let record = pOffsets[row]; record < pOffsets[row + 1]; record += 1) pRows[record] = row;
  }
  const restrictionOffsets = new Uint32Array(target.nodeCount + 1);
  for (const coarse of pColumns) restrictionOffsets[coarse + 1] += 1;
  for (let coarse = 0; coarse < target.nodeCount; coarse += 1) {
    restrictionOffsets[coarse + 1] += restrictionOffsets[coarse];
  }
  const restrictionRecords = new Uint32Array(pColumns.length);
  const restrictionCursors = restrictionOffsets.slice(0, target.nodeCount);
  pColumns.forEach((coarse, record) => { restrictionRecords[restrictionCursors[coarse]++] = record; });

  const contributionTargets = source.galerkinCoarseEntries!;
  const fineEntries = source.galerkinFineEntries!, left = source.galerkinLeftRecords!;
  const right = source.galerkinRightRecords!;
  const contributionOffsets = new Uint32Array(target.columns.length + 1);
  for (const targetEntry of contributionTargets) contributionOffsets[targetEntry + 1] += 1;
  for (let entry = 0; entry < target.columns.length; entry += 1) {
    contributionOffsets[entry + 1] += contributionOffsets[entry];
  }
  const contributionCursors = contributionOffsets.slice(0, target.columns.length);
  const refreshFineEntries = new Uint32Array(contributionTargets.length);
  const refreshWeights = new Float32Array(contributionTargets.length);
  contributionTargets.forEach((targetEntry, contribution) => {
    const refreshRecord = contributionCursors[targetEntry]++;
    refreshFineEntries[refreshRecord] = fineEntries[contribution];
    refreshWeights[refreshRecord] = Math.fround(
      Math.fround(pWeights[left[contribution]]) * Math.fround(pWeights[right[contribution]]),
    );
  });

  const headerWords = 21;
  const pOffsetBase = headerWords;
  const pBase = pOffsetBase + pOffsets.length;
  const restrictionOffsetBase = pBase + 3 * pColumns.length;
  const restrictionRecordBase = restrictionOffsetBase + restrictionOffsets.length;
  const contributionOffsetBase = restrictionRecordBase + restrictionRecords.length;
  const contributionBase = contributionOffsetBase + contributionOffsets.length;
  const contributionEnd = contributionBase + 2 * contributionTargets.length;
  // Every recurring smoother/prolongation lookup needs the source diagonal.
  // Pack its CSR index once per row instead of linearly searching the row in
  // every sweep (coarse Galerkin rows are substantially wider than L0).
  const sourceDiagonalEntries = new Uint32Array(source.nodeCount);
  for (let row = 0; row < source.nodeCount; row += 1) {
    let diagonal = 0xffff_ffff;
    for (let entry = source.rowOffsets[row]; entry < source.rowOffsets[row + 1]; entry += 1) {
      if (source.columns[entry] === row) {
        if (diagonal !== 0xffff_ffff) {
          throw new RangeError(`power Galerkin source row ${row} has duplicate diagonals`);
        }
        diagonal = entry;
      }
    }
    if (diagonal === 0xffff_ffff) {
      throw new RangeError(`power Galerkin source row ${row} has no diagonal`);
    }
    sourceDiagonalEntries[row] = diagonal;
  }
  const sourceDiagonalBase = contributionEnd;
  const fineCellToRowBase = levelIndex === 0 ? sourceDiagonalBase + sourceDiagonalEntries.length : 0;
  const fineCellCount = hierarchy.dimensions[0] * hierarchy.dimensions[1] * hierarchy.dimensions[2];
  const fineLookupLevelCount = levelIndex === 0
    ? Math.max(...source.geometry.map((node) => Math.log2(node.span))) + 1
    : 0;
  const fineCellToRowWords = fineCellCount * fineLookupLevelCount;
  const fineGeometryBase = levelIndex === 0 ? fineCellToRowBase + fineCellToRowWords : 0;
  const topologyWords = new Uint32Array(
    contributionEnd + sourceDiagonalEntries.length
      + (levelIndex === 0 ? fineCellToRowWords + 2 * source.geometry.length : 0),
  );
  topologyWords.set([
    source.nodeCount, target.nodeCount, source.columns.length, target.columns.length,
    pBase, pColumns.length, restrictionOffsetBase, restrictionRecordBase,
    contributionOffsetBase, contributionBase, contributionTargets.length, 2,
    pOffsetBase, sourceDiagonalBase, fineCellToRowBase, fineGeometryBase,
    ...hierarchy.dimensions,
    fineCellCount,
    fineLookupLevelCount,
  ]);
  const topologyFloats = new Float32Array(topologyWords.buffer);
  topologyWords.set(pOffsets, pOffsetBase);
  for (let record = 0; record < pColumns.length; record += 1) {
    topologyWords[pBase + 3 * record] = pRows[record];
    topologyWords[pBase + 3 * record + 1] = pColumns[record];
    topologyFloats[pBase + 3 * record + 2] = pWeights[record];
  }
  topologyWords.set(restrictionOffsets, restrictionOffsetBase);
  topologyWords.set(restrictionRecords, restrictionRecordBase);
  topologyWords.set(contributionOffsets, contributionOffsetBase);
  for (let contribution = 0; contribution < contributionTargets.length; contribution += 1) {
    const base = contributionBase + 2 * contribution;
    topologyWords[base] = refreshFineEntries[contribution];
    topologyFloats[base + 1] = refreshWeights[contribution];
  }
  topologyWords.set(sourceDiagonalEntries, sourceDiagonalBase);
  if (levelIndex === 0) {
    for (let row = 0; row < source.geometry.length; row += 1) {
      const node = source.geometry[row];
      const cell = node.x + hierarchy.dimensions[0]
        * (node.y + hierarchy.dimensions[1] * node.z);
      const level = Math.log2(node.span);
      const lookup = fineCellToRowBase + level * fineCellCount + cell;
      if (!Number.isSafeInteger(level) || level < 0 || level >= fineLookupLevelCount
        || cell >= fineCellCount || topologyWords[lookup] !== 0) {
        throw new RangeError("power Galerkin fixed hierarchy has duplicate or invalid fine-cell origins");
      }
      topologyWords[lookup] = row + 1;
      topologyWords[fineGeometryBase + 2 * row] = cell;
      topologyWords[fineGeometryBase + 2 * row + 1] = node.span;
    }
  }
  return Object.freeze({
    sourceNodeCount: source.nodeCount,
    targetNodeCount: target.nodeCount,
    sourceEntryCount: source.columns.length,
    targetEntryCount: target.columns.length,
    sourceMatrixWords: packMatrix(source, operators[levelIndex]),
    targetMatrixWords: packMatrix(target, operators[levelIndex + 1]),
    topologyWords,
  });
}

/** CPU decoder for the exact parent-owned numeric refresh used by WGSL. */
export function refreshPackedOctreePowerGalerkinTarget(
  packed: PackedOctreePowerGalerkinLevel,
): Float32Array {
  const header = packed.topologyWords;
  const contributionOffsetBase = header[8], contributionBase = header[9];
  const sourceBase = packed.sourceNodeCount + 1;
  const sourceFloats = new Float32Array(packed.sourceMatrixWords.buffer);
  const topologyFloats = new Float32Array(packed.topologyWords.buffer);
  const result = new Float32Array(packed.targetEntryCount);
  for (let targetEntry = 0; targetEntry < packed.targetEntryCount; targetEntry += 1) {
    let sum = 0;
    const first = header[contributionOffsetBase + targetEntry];
    const end = header[contributionOffsetBase + targetEntry + 1];
    for (let cursor = first; cursor < end; cursor += 1) {
      const base = contributionBase + 2 * cursor;
      sum += sourceFloats[sourceBase + 2 * header[base] + 1] * topologyFloats[base + 1];
    }
    result[targetEntry] = sum;
  }
  return result;
}
