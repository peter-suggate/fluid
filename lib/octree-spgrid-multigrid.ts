/**
 * Small, dense oracle for the sparse uniform-grid pressure pyramid described
 * by Setaluri et al. (2014) and reused by Aanjaneya et al. (2017), section 4.3.
 *
 * This is deliberately not a production CPU solver.  It turns a bounded
 * native-level topology into explicit transfer and Galerkin matrices so the
 * compact GPU implementation can be checked without duplicating its kernels.
 */

export const OCTREE_SPGRID_ACTIVE = 1 << 0;
export const OCTREE_SPGRID_GHOST = 1 << 1;
export const OCTREE_SPGRID_MULTIGRID_ONLY = 1 << 2;
export const OCTREE_SPGRID_FINEST_AT_LEVEL = 1 << 3;

export type OctreeSPGridCoordinate = readonly [number, number, number];

export interface OctreeSPGridCellInput {
  /** Stable within the hierarchy; used by explicit copy transfers. */
  readonly id: string;
  /** Integer cell coordinate in this level's native uniform grid. */
  readonly coordinate: OctreeSPGridCoordinate;
  /** Exactly one storage-class flag, optionally plus FINEST_AT_LEVEL. */
  readonly flags: number;
  /**
   * Coarser-level cell receiving an exact copy.  Cells without this field must
   * be finest at this level and use the standard cell-centred trilinear map.
   */
  readonly persistsAs?: string;
}

export interface OctreeSPGridLevelInput {
  /** Cell width measured in finest-grid cell widths. Adjacent levels double. */
  readonly cellWidth: number;
  readonly cells: readonly OctreeSPGridCellInput[];
}

export interface OctreeSPGridHierarchyInput {
  /** Row-major dense captured first-order L1 operator. */
  readonly finestOperator: ArrayLike<number>;
  /** Finest to coarsest native sparse uniform grids. */
  readonly levels: readonly OctreeSPGridLevelInput[];
  readonly symmetryTolerance?: number;
}

export interface OctreeSPGridTransferRecord {
  readonly fine: number;
  readonly coarse: number;
  readonly weight: number;
  readonly mode: "copy" | "trilinear";
}

export interface OctreeSPGridAdjacentTransfer {
  readonly fineCount: number;
  readonly coarseCount: number;
  /** Row-major fine-by-coarse prolongation matrix. */
  readonly prolongation: Float64Array;
  /** Row-major coarse-by-fine restriction matrix; exactly P transpose. */
  readonly restriction: Float64Array;
  readonly records: readonly OctreeSPGridTransferRecord[];
}

export interface OctreeSPGridLevel {
  readonly cellWidth: number;
  readonly ids: readonly string[];
  /** xyz triples in the native uniform grid's integer coordinates. */
  readonly coordinates: Int32Array;
  readonly flags: Uint8Array;
  /** One only for cells on which this level's damped-Jacobi kernel operates. */
  readonly smoothingMask: Uint8Array;
  /** Explicit row-major Galerkin operator for oracle/parity checks. */
  readonly operator: Float64Array;
  readonly transferToCoarser?: OctreeSPGridAdjacentTransfer;
}

export interface OctreeSPGridLDLTFactor {
  readonly size: number;
  /** Unit-lower triangular factor, row-major (upper entries are zero). */
  readonly lower: Float64Array;
  readonly diagonal: Float64Array;
}

export interface OctreeSPGridHierarchy {
  readonly levels: readonly OctreeSPGridLevel[];
  readonly coarsestFactor: OctreeSPGridLDLTFactor;
}

export interface OctreeSPGridVCycleOptions {
  readonly smoothingIterations?: number;
  readonly damping?: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function cellKey(coordinate: OctreeSPGridCoordinate): string {
  return `${coordinate[0]}:${coordinate[1]}:${coordinate[2]}`;
}

function validateCell(cell: OctreeSPGridCellInput, level: number): void {
  if (!cell.id) throw new RangeError(`SPGrid level ${level} contains an empty cell id`);
  for (const coordinate of cell.coordinate) {
    if (!Number.isSafeInteger(coordinate)) throw new RangeError(`SPGrid cell ${cell.id} has a non-integer coordinate`);
  }
  const storage = cell.flags & (OCTREE_SPGRID_ACTIVE | OCTREE_SPGRID_GHOST | OCTREE_SPGRID_MULTIGRID_ONLY);
  if (storage === 0 || (storage & (storage - 1)) !== 0) {
    throw new RangeError(`SPGrid cell ${cell.id} must have exactly one active, ghost, or multigrid-only flag`);
  }
  if (cell.persistsAs === undefined && (cell.flags & OCTREE_SPGRID_FINEST_AT_LEVEL) === 0) {
    throw new RangeError(`SPGrid cell ${cell.id} is neither persistent nor finest at its level`);
  }
}

function transpose(matrix: ArrayLike<number>, rows: number, columns: number): Float64Array {
  const result = new Float64Array(columns * rows);
  for (let row = 0; row < rows; row += 1) for (let column = 0; column < columns; column += 1) {
    result[column * rows + row] = matrix[row * columns + column];
  }
  return result;
}

export function multiplyOctreeSPGridMatrix(
  matrix: ArrayLike<number>,
  rows: number,
  columns: number,
  vector: ArrayLike<number>,
): Float64Array {
  if (matrix.length !== rows * columns || vector.length !== columns) throw new RangeError("SPGrid matrix dimensions do not match");
  const result = new Float64Array(rows);
  for (let row = 0; row < rows; row += 1) {
    let sum = 0;
    for (let column = 0; column < columns; column += 1) sum += matrix[row * columns + column] * vector[column];
    result[row] = sum;
  }
  return result;
}

/** Materialize P and R=P^T between two native sparse uniform-grid levels. */
export function materializeOctreeSPGridTransfer(
  fineLevel: OctreeSPGridLevelInput,
  coarseLevel: OctreeSPGridLevelInput,
): OctreeSPGridAdjacentTransfer {
  const fineWidth = positiveInteger(fineLevel.cellWidth, "fine SPGrid cell width");
  const coarseWidth = positiveInteger(coarseLevel.cellWidth, "coarse SPGrid cell width");
  if (coarseWidth !== 2 * fineWidth) throw new RangeError("Adjacent SPGrid levels must differ by exactly a factor of two");

  const coarseById = new Map<string, number>(), coarseByCoordinate = new Map<string, number>();
  coarseLevel.cells.forEach((cell, index) => {
    validateCell(cell, 1);
    if (coarseById.has(cell.id)) throw new RangeError(`Duplicate SPGrid cell id ${cell.id}`);
    const key = cellKey(cell.coordinate);
    if (coarseByCoordinate.has(key)) throw new RangeError(`Duplicate coarse SPGrid coordinate ${key}`);
    coarseById.set(cell.id, index); coarseByCoordinate.set(key, index);
  });

  const fineCount = fineLevel.cells.length, coarseCount = coarseLevel.cells.length;
  const prolongation = new Float64Array(fineCount * coarseCount);
  const records: OctreeSPGridTransferRecord[] = [];
  const fineIds = new Set<string>(), fineCoordinates = new Set<string>();
  fineLevel.cells.forEach((cell, fine) => {
    validateCell(cell, 0);
    if (fineIds.has(cell.id)) throw new RangeError(`Duplicate SPGrid cell id ${cell.id}`);
    const ownKey = cellKey(cell.coordinate);
    if (fineCoordinates.has(ownKey)) throw new RangeError(`Duplicate fine SPGrid coordinate ${ownKey}`);
    fineIds.add(cell.id); fineCoordinates.add(ownKey);
    if (cell.persistsAs !== undefined) {
      const coarse = coarseById.get(cell.persistsAs);
      if (coarse === undefined) throw new RangeError(`Persistent SPGrid target ${cell.persistsAs} does not exist`);
      prolongation[fine * coarseCount + coarse] = 1;
      records.push({ fine, coarse, weight: 1, mode: "copy" });
      return;
    }

    // Cell-centred trilinear interpolation.  If x is a fine cell index, its
    // centre in coarse index space is (x + 1/2) / 2 - 1/2.
    const lower = [0, 0, 0], fraction = [0, 0, 0];
    for (let axis = 0; axis < 3; axis += 1) {
      const coordinate = ((cell.coordinate[axis] + 0.5) * fineWidth) / coarseWidth - 0.5;
      lower[axis] = Math.floor(coordinate); fraction[axis] = coordinate - lower[axis];
    }
    for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) for (let dx = 0; dx < 2; dx += 1) {
      const coordinate = [lower[0] + dx, lower[1] + dy, lower[2] + dz] as const;
      const weight = (dx ? fraction[0] : 1 - fraction[0])
        * (dy ? fraction[1] : 1 - fraction[1])
        * (dz ? fraction[2] : 1 - fraction[2]);
      if (weight === 0) continue;
      const coarse = coarseByCoordinate.get(cellKey(coordinate));
      if (coarse === undefined) {
        throw new RangeError(`Incomplete trilinear support for SPGrid cell ${cell.id}: missing ${cellKey(coordinate)}`);
      }
      prolongation[fine * coarseCount + coarse] += weight;
      records.push({ fine, coarse, weight, mode: "trilinear" });
    }
  });
  return Object.freeze({
    fineCount,
    coarseCount,
    prolongation,
    restriction: transpose(prolongation, fineCount, coarseCount),
    records: Object.freeze(records),
  });
}

/** Explicit dense R A P, used only on bounded oracle matrices. */
export function galerkinOctreeSPGridOperator(
  fineOperator: ArrayLike<number>,
  transfer: OctreeSPGridAdjacentTransfer,
): Float64Array {
  const { fineCount, coarseCount, prolongation, restriction } = transfer;
  if (fineOperator.length !== fineCount * fineCount) throw new RangeError("Fine SPGrid operator size does not match its level");
  const aTimesP = new Float64Array(fineCount * coarseCount);
  for (let fineRow = 0; fineRow < fineCount; fineRow += 1) for (let coarseColumn = 0; coarseColumn < coarseCount; coarseColumn += 1) {
    let sum = 0;
    for (let fineColumn = 0; fineColumn < fineCount; fineColumn += 1) {
      sum += fineOperator[fineRow * fineCount + fineColumn] * prolongation[fineColumn * coarseCount + coarseColumn];
    }
    aTimesP[fineRow * coarseCount + coarseColumn] = sum;
  }
  const coarse = new Float64Array(coarseCount * coarseCount);
  for (let coarseRow = 0; coarseRow < coarseCount; coarseRow += 1) for (let coarseColumn = 0; coarseColumn < coarseCount; coarseColumn += 1) {
    let sum = 0;
    for (let fine = 0; fine < fineCount; fine += 1) {
      sum += restriction[coarseRow * fineCount + fine] * aTimesP[fine * coarseCount + coarseColumn];
    }
    coarse[coarseRow * coarseCount + coarseColumn] = sum;
  }
  return coarse;
}

/** Deterministic, pivot-free LDLT factorization for a bounded SPD bottom solve. */
export function factorOctreeSPGridCoarsest(
  operator: ArrayLike<number>,
  size: number,
  tolerance = 1e-12,
): OctreeSPGridLDLTFactor {
  positiveInteger(size, "coarsest SPGrid size");
  if (operator.length !== size * size) throw new RangeError("Coarsest SPGrid operator size does not match");
  const lower = new Float64Array(size * size), diagonal = new Float64Array(size);
  for (let row = 0; row < size; row += 1) {
    lower[row * size + row] = 1;
    for (let column = 0; column < row; column += 1) {
      let value = operator[row * size + column];
      for (let k = 0; k < column; k += 1) value -= lower[row * size + k] * diagonal[k] * lower[column * size + k];
      if (!(Math.abs(diagonal[column]) > tolerance)) throw new RangeError("Coarsest SPGrid operator is not positive definite");
      lower[row * size + column] = value / diagonal[column];
    }
    let pivot = operator[row * size + row];
    for (let k = 0; k < row; k += 1) pivot -= lower[row * size + k] * lower[row * size + k] * diagonal[k];
    if (!(pivot > tolerance) || !Number.isFinite(pivot)) throw new RangeError("Coarsest SPGrid operator is not positive definite");
    diagonal[row] = pivot;
  }
  return Object.freeze({ size, lower, diagonal });
}

export function solveOctreeSPGridCoarsest(factor: OctreeSPGridLDLTFactor, rhs: ArrayLike<number>): Float64Array {
  const { size, lower, diagonal } = factor;
  if (rhs.length !== size) throw new RangeError("Coarsest SPGrid right hand side size does not match");
  const intermediate = new Float64Array(size), scaled = new Float64Array(size), result = new Float64Array(size);
  for (let row = 0; row < size; row += 1) {
    let value = rhs[row];
    for (let column = 0; column < row; column += 1) value -= lower[row * size + column] * intermediate[column];
    intermediate[row] = value; scaled[row] = value / diagonal[row];
  }
  for (let reverse = size; reverse > 0; reverse -= 1) {
    const row = reverse - 1; let value = scaled[row];
    for (let column = row + 1; column < size; column += 1) value -= lower[column * size + row] * result[column];
    result[row] = value;
  }
  return result;
}

function assertSymmetric(operator: ArrayLike<number>, size: number, tolerance: number, level: number): void {
  if (operator.length !== size * size) throw new RangeError(`SPGrid level ${level} operator size does not match its cells`);
  for (let row = 0; row < size; row += 1) for (let column = 0; column < row; column += 1) {
    const scale = Math.max(1, Math.abs(operator[row * size + column]), Math.abs(operator[column * size + row]));
    if (Math.abs(operator[row * size + column] - operator[column * size + row]) > tolerance * scale) {
      throw new RangeError(`SPGrid level ${level} operator is not symmetric`);
    }
  }
}

/** Build explicit levels, transfers, Galerkin operators, and fixed bottom factor. */
export function buildOctreeSPGridMultigrid(input: OctreeSPGridHierarchyInput): OctreeSPGridHierarchy {
  if (input.levels.length < 1) throw new RangeError("SPGrid hierarchy requires at least one level");
  const symmetryTolerance = Math.max(0, input.symmetryTolerance ?? 1e-12);
  const operators: Float64Array[] = [Float64Array.from(input.finestOperator)];
  const transfers: OctreeSPGridAdjacentTransfer[] = [];
  for (let index = 0; index < input.levels.length; index += 1) {
    const level = input.levels[index];
    positiveInteger(level.cellWidth, `SPGrid level ${index} cell width`);
    if (level.cells.length < 1) throw new RangeError(`SPGrid level ${index} is empty`);
    const ids = new Set<string>(), coordinates = new Set<string>();
    for (const cell of level.cells) {
      validateCell(cell, index);
      if (ids.has(cell.id)) throw new RangeError(`Duplicate SPGrid cell id ${cell.id}`);
      const key = cellKey(cell.coordinate);
      if (coordinates.has(key)) throw new RangeError(`Duplicate SPGrid coordinate ${key} at level ${index}`);
      ids.add(cell.id); coordinates.add(key);
    }
    assertSymmetric(operators[index], level.cells.length, symmetryTolerance, index);
    if (index + 1 < input.levels.length) {
      const transfer = materializeOctreeSPGridTransfer(level, input.levels[index + 1]);
      transfers.push(transfer);
      operators.push(galerkinOctreeSPGridOperator(operators[index], transfer));
    }
  }
  const levels = input.levels.map((source, levelIndex): OctreeSPGridLevel => {
    const coordinates = new Int32Array(3 * source.cells.length), flags = new Uint8Array(source.cells.length);
    source.cells.forEach((cell, index) => { coordinates.set(cell.coordinate, 3 * index); flags[index] = cell.flags; });
    return Object.freeze({
      cellWidth: source.cellWidth,
      ids: Object.freeze(source.cells.map((cell) => cell.id)),
      coordinates,
      flags,
      smoothingMask: Uint8Array.from(source.cells, (cell) => (cell.flags & OCTREE_SPGRID_FINEST_AT_LEVEL) !== 0 ? 1 : 0),
      operator: operators[levelIndex],
      transferToCoarser: transfers[levelIndex],
    });
  });
  const coarsest = levels[levels.length - 1];
  return Object.freeze({ levels: Object.freeze(levels), coarsestFactor: factorOctreeSPGridCoarsest(coarsest.operator, coarsest.ids.length) });
}

function dampedJacobi(
  operator: ArrayLike<number>,
  rhs: ArrayLike<number>,
  initial: ArrayLike<number>,
  mask: ArrayLike<number>,
  iterations: number,
  damping: number,
): Float64Array {
  const size = rhs.length; let value = Float64Array.from(initial);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const product = multiplyOctreeSPGridMatrix(operator, size, size, value);
    const next = value.slice();
    for (let row = 0; row < size; row += 1) {
      if (mask[row] === 0) continue;
      const diagonal = operator[row * size + row];
      if (!(diagonal > 0)) throw new RangeError("SPGrid Jacobi smoother requires positive diagonals");
      next[row] += damping * (rhs[row] - product[row]) / diagonal;
    }
    value = next;
  }
  return value;
}

/**
 * Apply a symmetric V-cycle: matching finest-cell-only Jacobi sweeps bracket
 * each adjacent-level correction, and the coarsest operation is fixed LDLT.
 */
export function applyOctreeSPGridVCycle(
  hierarchy: OctreeSPGridHierarchy,
  rhs: ArrayLike<number>,
  options: OctreeSPGridVCycleOptions = {},
): Float64Array {
  const smoothingIterations = Math.max(1, Math.min(16, Math.round(options.smoothingIterations ?? 2)));
  const damping = options.damping ?? 2 / 3;
  if (!(damping > 0 && damping < 1) || !Number.isFinite(damping)) throw new RangeError("SPGrid damping must be finite and between zero and one");
  if (rhs.length !== hierarchy.levels[0].ids.length) throw new RangeError("Finest SPGrid right hand side size does not match");
  const visit = (levelIndex: number, levelRhs: ArrayLike<number>): Float64Array => {
    const level = hierarchy.levels[levelIndex], size = level.ids.length;
    if (levelIndex + 1 === hierarchy.levels.length) return solveOctreeSPGridCoarsest(hierarchy.coarsestFactor, levelRhs);
    const correction = dampedJacobi(level.operator, levelRhs, new Float64Array(size), level.smoothingMask,
      smoothingIterations, damping);
    const product = multiplyOctreeSPGridMatrix(level.operator, size, size, correction);
    const residual = Float64Array.from(levelRhs, (value, row) => value - product[row]);
    const transfer = level.transferToCoarser!;
    const coarseRhs = multiplyOctreeSPGridMatrix(transfer.restriction, transfer.coarseCount, transfer.fineCount, residual);
    const coarseCorrection = visit(levelIndex + 1, coarseRhs);
    const prolonged = multiplyOctreeSPGridMatrix(transfer.prolongation, transfer.fineCount, transfer.coarseCount, coarseCorrection);
    for (let row = 0; row < size; row += 1) correction[row] += prolonged[row];
    return dampedJacobi(level.operator, levelRhs, correction, level.smoothingMask, smoothingIterations, damping);
  };
  return visit(0, rhs);
}
