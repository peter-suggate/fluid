import type { OctreeFirstOrderSPDVCycle } from "./webgpu-octree-mgpcg";

/** Native sparse-level cell roles from Setaluri et al., section 5. */
export const SPGRID_CELL_FLAG = Object.freeze({
  active: 1 << 0,
  ghost: 1 << 1,
  multigridOnly: 1 << 2,
} as const);

export interface SPGridLeafOracle {
  readonly origin: readonly [number, number, number];
  /** Power-of-two width in finest cells. */
  readonly size: number;
}

/** One undirected pressure coupling from the captured adaptive L1 graph. */
export interface SPGridContactOracle {
  readonly negative: number;
  readonly positive: number;
  readonly coefficient: number;
}

export interface SPGridTransferRecord {
  readonly fine: number;
  readonly coarse: number;
  readonly weight: number;
}

/** Stable parent-owned traversal of the immutable transfer records. The GPU
 * replacement for restriction must publish this exact layout before it can
 * replace scatter atomics with one deterministic gather per coarse slot. */
export interface SPGridParentGatherCSR {
  readonly coarseCount: number;
  readonly offsets: readonly number[];
  readonly recordIndices: readonly number[];
}

export interface SPGridOracleLevel {
  readonly scale: number;
  readonly coordinates: readonly (readonly [number, number, number])[];
  readonly flags: readonly number[];
  /** Owning adaptive leaf.  Multigrid-only aggregate cells use -1. */
  readonly owners?: readonly number[];
}

export interface SPGridPyramidOracle {
  readonly levels: readonly SPGridOracleLevel[];
  /** P(fine,coarse), stored once and used as P and P^T. */
  readonly transfers: readonly (readonly SPGridTransferRecord[])[];
}

export interface SPGridContactLevelOracle {
  readonly scale: number;
  readonly coordinates: readonly (readonly [number, number, number])[];
  readonly flags: readonly number[];
  readonly owners: readonly number[];
  /** E: exact GhostValuePropagate map, row-major slot-by-leaf. */
  readonly propagate: Float64Array;
  /** E^T: exact GhostValueAccumulate map, row-major leaf-by-slot. */
  readonly accumulate: Float64Array;
  /** B: sparse-level contact operator, row-major slot-by-slot. */
  readonly slotOperator: Float64Array;
  /** E^T B E, row-major leaf-by-leaf. */
  readonly assembledOperator: Float64Array;
}

const coordinateKey = (value: readonly [number, number, number]) => `${value[0]},${value[1]},${value[2]}`;

function contactCoordinate(owner: SPGridLeafOracle, neighbour: SPGridLeafOracle, scale: number): [number, number, number] {
  const result: [number, number, number] = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const begin = owner.origin[axis], end = begin + owner.size;
    const otherBegin = neighbour.origin[axis], otherEnd = otherBegin + neighbour.size;
    let sample: number;
    if (otherBegin >= end) sample = end - 0.5 * scale;
    else if (otherEnd <= begin) sample = begin + 0.5 * scale;
    else sample = Math.max(begin + 0.5 * scale, Math.min(end - 0.5 * scale, 0.5 * (otherBegin + otherEnd)));
    result[axis] = Math.floor(sample / scale);
  }
  return result;
}

function transposeDense(matrix: ArrayLike<number>, rows: number, columns: number): Float64Array {
  const result = new Float64Array(columns * rows);
  for (let row = 0; row < rows; row += 1) for (let column = 0; column < columns; column += 1) {
    result[column * rows + row] = matrix[row * columns + column];
  }
  return result;
}

function multiplyDense(a: ArrayLike<number>, aRows: number, shared: number, b: ArrayLike<number>, bColumns: number): Float64Array {
  if (a.length !== aRows * shared || b.length !== shared * bColumns) throw new RangeError("SPGrid dense oracle dimensions disagree");
  const result = new Float64Array(aRows * bColumns);
  for (let row = 0; row < aRows; row += 1) for (let column = 0; column < bColumns; column += 1) {
    let sum = 0;
    for (let inner = 0; inner < shared; inner += 1) sum += a[row * shared + inner] * b[inner * bColumns + column];
    result[row * bColumns + column] = sum;
  }
  return result;
}

/**
 * Bounded assembled oracle for Aanjaneya et al. (2017), Section 4.2.
 *
 * A coarse adaptive leaf has one ghost alias for every distinct fine-grid
 * contact cell.  Face contacts therefore create a face patch and power-face
 * edge contacts create the corresponding edge aliases.  E copies a leaf
 * value to every alias (`GhostValuePropagate`); accumulation is the same
 * immutable incidence list traversed in reverse, so it is exactly E^T.
 */
export function buildSPGridContactLevelOracle(
  leaves: readonly SPGridLeafOracle[],
  contacts: readonly SPGridContactOracle[],
  level: number,
  anchors: readonly number[] = leaves.map(() => 1),
): SPGridContactLevelOracle {
  if (!Number.isSafeInteger(level) || level < 0) throw new RangeError("SPGrid contact level must be a non-negative integer");
  if (anchors.length !== leaves.length) throw new RangeError("SPGrid contact anchors disagree with the leaf count");
  const scale = 2 ** level, native = leaves.map((leaf, index) => {
    assertPowerOfTwo(leaf.size, `SPGrid leaf ${index} size`);
    if (leaf.origin.some((value) => !Number.isSafeInteger(value) || value < 0 || value % leaf.size !== 0)) {
      throw new RangeError(`SPGrid leaf ${index} origin is not aligned to its size`);
    }
    return Math.round(Math.log2(leaf.size));
  });
  const slots = new Map<string, { coordinate: [number, number, number]; owner: number; flags: number }>();
  const addOwned = (coordinate: [number, number, number], owner: number, flags: number) => {
    const key = coordinateKey(coordinate), old = slots.get(key);
    if (old !== undefined) {
      if (old.owner !== owner) throw new RangeError(`Overlapping SPGrid owners ${old.owner} and ${owner} at ${key}`);
      old.flags = (old.flags & SPGRID_CELL_FLAG.active) !== 0 || (flags & SPGRID_CELL_FLAG.active) !== 0
        ? SPGRID_CELL_FLAG.active : SPGRID_CELL_FLAG.ghost;
      return;
    }
    slots.set(key, { coordinate, owner, flags });
  };
  leaves.forEach((leaf, owner) => {
    if (native[owner] === level) addOwned(leaf.origin.map((value) => value / scale) as [number, number, number], owner, SPGRID_CELL_FLAG.active);
  });
  contacts.forEach((contact, contactIndex) => {
    if (!Number.isSafeInteger(contact.negative) || !Number.isSafeInteger(contact.positive)
      || contact.negative < 0 || contact.positive < 0 || contact.negative >= leaves.length || contact.positive >= leaves.length
      || contact.negative === contact.positive || !(contact.coefficient > 0) || !Number.isFinite(contact.coefficient)) {
      throw new RangeError(`Invalid SPGrid contact ${contactIndex}`);
    }
    for (const [owner, neighbour] of [[contact.negative, contact.positive], [contact.positive, contact.negative]] as const) {
      if (native[owner] > level && native[neighbour] <= level) {
        addOwned(contactCoordinate(leaves[owner], leaves[neighbour], scale), owner, SPGRID_CELL_FLAG.ghost);
      }
    }
  });
  const cells = [...slots.values()], slotCount = cells.length, leafCount = leaves.length;
  const slotByOwnerCoordinate = new Map(cells.map((cell, index) => [`${cell.owner}:${coordinateKey(cell.coordinate)}`, index]));
  const propagate = new Float64Array(slotCount * leafCount);
  cells.forEach((cell, slot) => { propagate[slot * leafCount + cell.owner] = 1; });
  const accumulate = transposeDense(propagate, slotCount, leafCount), slotOperator = new Float64Array(slotCount * slotCount);
  // Split an adaptive leaf's Dirichlet/solid anchor evenly among its aliases;
  // E^T B E therefore retains the original leaf anchor exactly.
  const aliasesPerOwner = new Uint32Array(leafCount);
  cells.forEach((cell) => { aliasesPerOwner[cell.owner] += 1; });
  cells.forEach((cell, slot) => {
    const anchor = anchors[cell.owner];
    if (!(anchor >= 0) || !Number.isFinite(anchor)) throw new RangeError(`Invalid SPGrid anchor for leaf ${cell.owner}`);
    slotOperator[slot * slotCount + slot] += anchor / aliasesPerOwner[cell.owner];
  });
  contacts.forEach((contact) => {
    const a = contactCoordinate(leaves[contact.negative], leaves[contact.positive], scale);
    const b = contactCoordinate(leaves[contact.positive], leaves[contact.negative], scale);
    const negative = slotByOwnerCoordinate.get(`${contact.negative}:${coordinateKey(a)}`);
    const positive = slotByOwnerCoordinate.get(`${contact.positive}:${coordinateKey(b)}`);
    // A contact belongs to this uniform level only when both endpoint storage
    // cells exist here (native active or a spawned contact ghost).
    if (negative === undefined || positive === undefined) return;
    const c = contact.coefficient;
    slotOperator[negative * slotCount + negative] += c;
    slotOperator[positive * slotCount + positive] += c;
    slotOperator[negative * slotCount + positive] -= c;
    slotOperator[positive * slotCount + negative] -= c;
  });
  const bTimesE = multiplyDense(slotOperator, slotCount, slotCount, propagate, leafCount);
  const assembledOperator = multiplyDense(accumulate, leafCount, slotCount, bTimesE, leafCount);
  return Object.freeze({ scale, coordinates: Object.freeze(cells.map((cell) => cell.coordinate)),
    flags: Object.freeze(cells.map((cell) => cell.flags)), owners: Object.freeze(cells.map((cell) => cell.owner)),
    propagate, accumulate, slotOperator, assembledOperator });
}

function assertPowerOfTwo(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1 || (value & (value - 1)) !== 0) {
    throw new RangeError(`${label} must be a positive power of two`);
  }
}

/**
 * Small CPU oracle for topology/transfer tests. Active octree cells live at
 * their native level; coarser leaves appear as ghosts on finer MG levels and
 * generated interpolation targets are explicitly marked multigrid-only.
 */
export function buildSPGridPyramidOracle(leaves: readonly SPGridLeafOracle[], levelCount: number): SPGridPyramidOracle {
  if (!Number.isSafeInteger(levelCount) || levelCount < 2) throw new RangeError("SPGrid level count must be at least two");
  const maps = Array.from({ length: levelCount }, () => new Map<string, { coordinate: [number, number, number]; flags: number }>());
  const add = (level: number, coordinate: [number, number, number], flags: number) => {
    const key = coordinateKey(coordinate), old = maps[level].get(key);
    if (old) old.flags = (old.flags & SPGRID_CELL_FLAG.active) !== 0 || (flags & SPGRID_CELL_FLAG.active) !== 0
      ? SPGRID_CELL_FLAG.active
      : (old.flags & SPGRID_CELL_FLAG.ghost) !== 0 || (flags & SPGRID_CELL_FLAG.ghost) !== 0
        ? SPGRID_CELL_FLAG.ghost : SPGRID_CELL_FLAG.multigridOnly;
    else maps[level].set(key, { coordinate, flags });
  };
  for (const leaf of leaves) {
    assertPowerOfTwo(leaf.size, "SPGrid leaf size");
    if (leaf.origin.some((value) => !Number.isSafeInteger(value) || value < 0)) {
      throw new RangeError("SPGrid leaf origins must be non-negative integers");
    }
    const nativeLevel = Math.min(levelCount - 1, Math.round(Math.log2(leaf.size)));
    for (let level = 0; level < levelCount; level += 1) {
      const scale = 2 ** level;
      const coordinate: [number, number, number] = [
        Math.floor(leaf.origin[0] / scale), Math.floor(leaf.origin[1] / scale), Math.floor(leaf.origin[2] / scale),
      ];
      add(level, coordinate, level === nativeLevel ? SPGRID_CELL_FLAG.active
        : level < nativeLevel ? SPGRID_CELL_FLAG.ghost : SPGRID_CELL_FLAG.multigridOnly);
    }
  }
  const transfers: SPGridTransferRecord[][] = [];
  for (let level = 0; level < levelCount - 1; level += 1) {
    const fine = [...maps[level].values()];
    const records: SPGridTransferRecord[] = [];
    for (let fineIndex = 0; fineIndex < fine.length; fineIndex += 1) {
      const cell = fine[fineIndex];
      const targets: Array<{ coordinate: [number, number, number]; weight: number }> = [];
      if ((cell.flags & SPGRID_CELL_FLAG.ghost) !== 0) {
        targets.push({ coordinate: cell.coordinate.map((v) => Math.floor(v / 2)) as [number, number, number], weight: 1 });
      } else {
        // Cell-centred trilinear interpolation. At a boundary, clamped targets
        // deliberately remain duplicate records: accumulation then retains a
        // unit row sum and prolongation consumes the identical records.
        const axis = cell.coordinate.map((v) => {
          const base = Math.floor(v / 2), neighbour = Math.max(0, base + ((v & 1) === 0 ? -1 : 1));
          return [{ value: base, weight: 0.75 }, { value: neighbour, weight: 0.25 }] as const;
        });
        for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
          targets.push({ coordinate: [axis[0][x].value, axis[1][y].value, axis[2][z].value],
            weight: axis[0][x].weight * axis[1][y].weight * axis[2][z].weight });
        }
      }
      for (const target of targets) add(level + 1, target.coordinate, SPGRID_CELL_FLAG.multigridOnly);
    }
    const coarse = [...maps[level + 1].values()], coarseIndex = new Map(coarse.map((cell, index) => [coordinateKey(cell.coordinate), index]));
    for (let fineIndex = 0; fineIndex < fine.length; fineIndex += 1) {
      const cell = fine[fineIndex];
      if ((cell.flags & SPGRID_CELL_FLAG.ghost) !== 0) {
        const key = coordinateKey(cell.coordinate.map((v) => Math.floor(v / 2)) as [number, number, number]);
        records.push({ fine: fineIndex, coarse: coarseIndex.get(key)!, weight: 1 });
        continue;
      }
      const axis = cell.coordinate.map((v) => {
        const base = Math.floor(v / 2), neighbour = Math.max(0, base + ((v & 1) === 0 ? -1 : 1));
        return [{ value: base, weight: 0.75 }, { value: neighbour, weight: 0.25 }] as const;
      });
      for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
        const key = coordinateKey([axis[0][x].value, axis[1][y].value, axis[2][z].value]);
        records.push({ fine: fineIndex, coarse: coarseIndex.get(key)!,
          weight: axis[0][x].weight * axis[1][y].weight * axis[2][z].weight });
      }
    }
    transfers.push(records);
  }
  return { levels: maps.map((map, level) => ({ scale: 2 ** level,
    coordinates: [...map.values()].map((entry) => entry.coordinate), flags: [...map.values()].map((entry) => entry.flags) })), transfers };
}

export function restrictSPGrid(fine: readonly number[], records: readonly SPGridTransferRecord[], coarseCount: number): number[] {
  const result = new Array<number>(coarseCount).fill(0);
  for (const record of records) result[record.coarse] += record.weight * fine[record.fine];
  return result;
}

/** Stable counting-sort oracle for the future GPU parent-key radix/CSR stage.
 * Repeated boundary records remain repeated indices; coalescing them would
 * change both interpolation weights and floating-point accumulation order. */
export function buildSPGridParentGatherCSR(
  records: readonly SPGridTransferRecord[],
  fineCount: number,
  coarseCount: number,
): SPGridParentGatherCSR {
  if (!Number.isSafeInteger(fineCount) || fineCount < 0 || !Number.isSafeInteger(coarseCount) || coarseCount < 0) {
    throw new RangeError("SPGrid parent-gather dimensions must be non-negative integers");
  }
  const offsets = new Array<number>(coarseCount + 1).fill(0);
  records.forEach((record, index) => {
    if (!Number.isSafeInteger(record.fine) || record.fine < 0 || record.fine >= fineCount
      || !Number.isSafeInteger(record.coarse) || record.coarse < 0 || record.coarse >= coarseCount
      || !(record.weight >= 0) || !Number.isFinite(record.weight)) {
      throw new RangeError(`Invalid SPGrid parent-gather record ${index}`);
    }
    offsets[record.coarse + 1] += 1;
  });
  for (let coarse = 0; coarse < coarseCount; coarse += 1) offsets[coarse + 1] += offsets[coarse];
  const cursors = offsets.slice(0, -1), recordIndices = new Array<number>(records.length);
  records.forEach((record, index) => { recordIndices[cursors[record.coarse]++] = index; });
  return Object.freeze({ coarseCount, offsets: Object.freeze(offsets), recordIndices: Object.freeze(recordIndices) });
}

/** Atomic-free restriction oracle: one parent owns each sum and traverses a
 * stable range. It consumes the original records so duplicate boundary
 * weights and the exact P/P^T relationship remain unchanged. */
export function restrictSPGridParentGather(
  fine: readonly number[],
  records: readonly SPGridTransferRecord[],
  csr: SPGridParentGatherCSR,
): number[] {
  if (csr.offsets.length !== csr.coarseCount + 1 || csr.recordIndices.length !== records.length
    || csr.offsets[0] !== 0 || csr.offsets.at(-1) !== records.length) {
    throw new RangeError("Invalid SPGrid parent-gather CSR");
  }
  const result = new Array<number>(csr.coarseCount).fill(0);
  for (let coarse = 0; coarse < csr.coarseCount; coarse += 1) {
    let sum = 0;
    for (let cursor = csr.offsets[coarse]; cursor < csr.offsets[coarse + 1]; cursor += 1) {
      const record = records[csr.recordIndices[cursor]];
      if (record.coarse !== coarse || record.fine >= fine.length) throw new RangeError("Corrupt SPGrid parent-gather range");
      sum += record.weight * fine[record.fine];
    }
    result[coarse] = sum;
  }
  return result;
}

/** Exact transpose of restrictSPGrid because it consumes the same immutable records. */
export function prolongSPGrid(coarse: readonly number[], records: readonly SPGridTransferRecord[], fineCount: number): number[] {
  const result = new Array<number>(fineCount).fill(0);
  for (const record of records) result[record.fine] += record.weight * coarse[record.coarse];
  return result;
}

/** Deterministic direct bottom oracle. No residual-dependent stopping is used. */
export function solveSPGridBottomLDLT(operator: readonly (readonly number[])[], rhs: readonly number[]): number[] {
  const n = operator.length;
  if (rhs.length !== n || operator.some((row) => row.length !== n)) throw new RangeError("bottom matrix dimensions disagree");
  const l = Array.from({ length: n }, () => new Array<number>(n).fill(0)), d = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < i; j += 1) {
      let value = operator[i][j];
      for (let k = 0; k < j; k += 1) value -= l[i][k] * d[k] * l[j][k];
      l[i][j] = value / d[j];
    }
    let diagonal = operator[i][i];
    for (let k = 0; k < i; k += 1) diagonal -= l[i][k] * l[i][k] * d[k];
    if (!(diagonal > 1e-14) || !Number.isFinite(diagonal)) throw new RangeError("bottom matrix is not SPD");
    d[i] = diagonal; l[i][i] = 1;
  }
  const y = new Array<number>(n), z = new Array<number>(n), x = new Array<number>(n);
  for (let i = 0; i < n; i += 1) { let value = rhs[i]; for (let j = 0; j < i; j += 1) value -= l[i][j] * y[j]; y[i] = value; }
  for (let i = 0; i < n; i += 1) z[i] = y[i] / d[i];
  for (let i = n - 1; i >= 0; i -= 1) { let value = z[i]; for (let j = i + 1; j < n; j += 1) value -= l[j][i] * x[j]; x[i] = value; }
  return x;
}

export interface OctreeSPGridVCyclePlan {
  readonly rowCapacity: number;
  readonly levelCount: number;
  readonly levelStride: number;
  readonly transferStride: number;
  readonly topologyBytes: number;
  readonly stateBytes: number;
  readonly dispatchBytes: number;
  readonly allocatedBytes: number;
  readonly rowDispatch: readonly [number, number, number];
  readonly slotDispatch: readonly [number, number, number];
  readonly transferDispatch: readonly [number, number, number];
}

export interface OctreeSPGridVCycleOptions {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
  readonly finestCellWidth: number;
  readonly maximumLevels?: number;
  readonly preSmoothingIterations?: number;
  readonly postSmoothingIterations?: number;
  readonly bottomIterations?: number;
  readonly damping?: number;
}

export interface OctreeSPGridVCycleSource { readonly leafHeaders: GPUBuffer; readonly leafEntries: GPUBuffer }

// Key/class/diagonal, six Cartesian and twelve octree-edge coefficients, five
// vectors, and the adaptive owner of active/ghost storage.  The three groups
// of four edge directions are the additional offset grids introduced for
// power diagrams in Aanjaneya et al. (2017), Section 4.2.
const STATE_CHANNELS = 27;
const TOPOLOGY_HEADER_WORDS = 16;
const DISPATCH_RECORD_BYTES_PER_LEVEL = 32;
const DISPATCH_LIFECYCLE_BYTES = 32;
/** Explicit row bound; the constructor also fails closed on device limits. */
export const SPGRID_MAXIMUM_ROW_CAPACITY = 16_384;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}
function nextPowerOfTwo(value: number): number { let result = 1; while (result < value) result *= 2; return result; }
function dispatchFor(capacity: number): readonly [number, number, number] {
  const blocks = Math.ceil(capacity / 64), x = Math.min(65_535, Math.max(1, blocks));
  return [x, Math.max(1, Math.ceil(blocks / x)), 1];
}

export function planOctreeSPGridVCycle(options: Pick<OctreeSPGridVCycleOptions, "dimensions" | "rowCapacity" | "maximumLevels">): OctreeSPGridVCyclePlan {
  const rowCapacity = positiveInteger(options.rowCapacity, "SPGrid row capacity");
  if (rowCapacity > SPGRID_MAXIMUM_ROW_CAPACITY) {
    throw new RangeError(`SPGrid row capacity exceeds the bounded ${SPGRID_MAXIMUM_ROW_CAPACITY}-row implementation`);
  }
  const width = Math.max(...options.dimensions.map((value) => positiveInteger(value, "SPGrid dimension")));
  const requiredLevels = Math.ceil(Math.log2(width)) + 1;
  if (requiredLevels > 12) throw new RangeError("SPGrid dimensions exceed the bounded 12-level exact-bottom hierarchy");
  if (options.maximumLevels !== undefined && options.maximumLevels < requiredLevels) {
    throw new RangeError("SPGrid maximumLevels would truncate the exact one-cell bottom level");
  }
  const levelCount = Math.max(2, requiredLevels);
  // A balanced 2:1 leaf can expose all eight children as distinct contact
  // ghosts.  Reserve twice that worst-case population so the open-addressed
  // table stays below 50% occupancy; overflow remains fail-closed.
  const levelStride = nextPowerOfTwo(rowCapacity * 16), transferStride = rowCapacity * 8;
  const rowMapWords = levelCount * rowCapacity, worklistWords = levelCount * levelStride;
  const transferWords = (levelCount - 1) * transferStride * 3;
  const topologyBytes = (TOPOLOGY_HEADER_WORDS + rowMapWords + worklistWords + transferWords) * 4;
  const stateBytes = STATE_CHANNELS * levelCount * levelStride * 4;
  const dispatchBytes = levelCount * DISPATCH_RECORD_BYTES_PER_LEVEL + DISPATCH_LIFECYCLE_BYTES;
  return { rowCapacity, levelCount, levelStride, transferStride, topologyBytes, stateBytes, dispatchBytes,
    // Storage-authored counts and indirect arguments must be separate WebGPU
    // buffers; the final term is the per-level uniform parameter storage.
    allocatedBytes: topologyBytes + stateBytes + 2 * dispatchBytes + levelCount * 64,
    rowDispatch: dispatchFor(rowCapacity), slotDispatch: dispatchFor(levelStride), transferDispatch: dispatchFor(transferStride) };
}

export type OctreeSPGridVCyclePipelineName = "emitCells" | "emitGhostAliases" | "buildTransfers" | "buildStencil" | "ensureDiagonal" | "finalizeIndirect"
  | "resetInvalidBuffers" | "retireSlots" | "retireRows" | "resetSetupMetadata" | "finalizeLifecycle" | "clearCorrection"
  | "zeroVectors" | "seedRhs" | "applyA" | "applyB" | "jacobiAtoB" | "jacobiBtoA"
  | "formResidual" | "restrictAndGhostAccumulate" | "exactBottom"
  | "prolongAndGhostPropagate" | "publish";

export const OCTREE_SPGRID_VCYCLE_BINDINGS: Readonly<Record<OctreeSPGridVCyclePipelineName, readonly number[]>> = Object.freeze({
  resetInvalidBuffers: [4, 5], retireSlots: [0, 4, 5, 6], retireRows: [0, 3, 4, 6],
  resetSetupMetadata: [0, 6], finalizeLifecycle: [0, 3, 4, 5, 6, 7],
  clearCorrection: [0, 3, 7, 9],
  emitCells: [0, 1, 3, 4, 5, 6, 7], emitGhostAliases: [0, 1, 2, 3, 4, 5, 6, 7],
  buildTransfers: [0, 1, 3, 4, 5, 6, 7],
  buildStencil: [0, 1, 2, 3, 4, 5, 7], ensureDiagonal: [0, 4, 5, 6], finalizeIndirect: [0, 6],
  zeroVectors: [0, 4, 5, 6, 7], seedRhs: [0, 1, 3, 4, 5, 7, 8], applyA: [0, 4, 5, 6, 7],
  applyB: [0, 4, 5, 6, 7], jacobiAtoB: [0, 4, 5, 6, 7], jacobiBtoA: [0, 4, 5, 6, 7],
  formResidual: [0, 4, 5, 6, 7], restrictAndGhostAccumulate: [0, 1, 3, 4, 5, 6, 7],
  exactBottom: [0, 4, 5, 6, 7],
  prolongAndGhostPropagate: [0, 1, 3, 4, 5, 6, 7],
  publish: [0, 1, 3, 4, 5, 7, 9],
});

type CachedGroup = { rowCount: GPUBuffer; control: GPUBuffer; rhs?: GPUBuffer; correction?: GPUBuffer; group: GPUBindGroup };

/**
 * A/B implementation of the paper-style native sparse pyramid. Setup is
 * intentionally GPU-idempotent and rebuilds from the captured authoritative
 * L1 rows whenever MGPCG invokes encodeSetup; no readback or fence is needed.
 */
export class WebGPUOctreeSPGridVCycle implements OctreeFirstOrderSPDVCycle {
  readonly operatorOrder = 1 as const;
  readonly isSymmetricPositiveDefinite = true as const;
  readonly plan: OctreeSPGridVCyclePlan;
  readonly allocatedBytes: number;
  /** Legacy MGPCG accounting name: ordered dispatch count, not transitions. */
  readonly encodedCorrectionPassCount: number;
  readonly encodedPassTransitionCount = 1;
  readonly encodedCorrectionPassTransitionCount = 1;
  readonly encodedSetupDispatchCount: number;
  readonly encodedCorrectionDispatchCount: number;
  readonly diagnostics: Readonly<{ levelCount: number; coarsestCapacity: number; maximumTransferRecordsPerLevel: number;
    correctionDispatchCount: number; correctionPassTransitions: number; restrictionScatterDispatchCount: number;
    restrictionAtomicAddUpperBound: number; parentGatherDispatchCount: number; parentGatherAtomicAddCount: 0;
    bottomOperation: "exact-single-cell"; coarsestDegreesOfFreedom: 1 }>;
  private readonly capturedHeaders: GPUBuffer;
  private readonly capturedEntries: GPUBuffer;
  private readonly topology: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly dispatchMeta: GPUBuffer;
  private readonly indirectDispatch: GPUBuffer;
  private readonly params: readonly GPUBuffer[];
  private readonly pipelines: Readonly<Record<OctreeSPGridVCyclePipelineName, GPUComputePipeline>>;
  private readonly groups = new Map<string, CachedGroup>();
  private readonly pre: number;
  private readonly post: number;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly source: OctreeSPGridVCycleSource,
    options: OctreeSPGridVCycleOptions) {
    this.plan = planOctreeSPGridVCycle(options);
    if (!(options.finestCellWidth > 0) || !Number.isFinite(options.finestCellWidth)) throw new RangeError("SPGrid finest cell width must be positive");
    if (source.leafHeaders.size < this.plan.rowCapacity * 48 || source.leafEntries.size < 8) throw new RangeError("SPGrid L1 source capacity is too small");
    if ((source.leafHeaders.usage & GPUBufferUsage.COPY_SRC) === 0 || (source.leafEntries.usage & GPUBufferUsage.COPY_SRC) === 0) {
      throw new RangeError("SPGrid L1 source buffers require COPY_SRC capture usage");
    }
    const limits = device.limits;
    const storageLimit = limits?.maxStorageBufferBindingSize ?? Number.POSITIVE_INFINITY;
    const bufferLimit = limits?.maxBufferSize ?? Number.POSITIVE_INFINITY;
    if (this.plan.stateBytes > storageLimit || this.plan.topologyBytes > storageLimit
      || Math.max(this.plan.stateBytes, this.plan.topologyBytes) > bufferLimit) {
      throw new RangeError("SPGrid pyramid exceeds this device's storage-buffer limits");
    }
    this.pre = Math.max(1, Math.min(8, Math.round(options.preSmoothingIterations ?? 2)));
    this.post = Math.max(1, Math.min(8, Math.round(options.postSmoothingIterations ?? this.pre)));
    if (this.pre !== this.post) throw new RangeError("SPGrid pre/post smoothing must match to retain symmetry");
    if ((this.pre & 1) !== 0) throw new RangeError("SPGrid smoothing count must be even");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.capturedHeaders = device.createBuffer({ label: "SPGrid captured L1 headers", size: this.plan.rowCapacity * 48, usage: storage });
    this.capturedEntries = device.createBuffer({ label: "SPGrid captured L1 entries", size: source.leafEntries.size, usage: storage });
    this.topology = device.createBuffer({ label: "SPGrid native sparse topology/worklists/transfers", size: this.plan.topologyBytes, usage: storage });
    this.state = device.createBuffer({ label: "SPGrid six-face stencils and vectors", size: this.plan.stateBytes, usage: storage });
    this.dispatchMeta = device.createBuffer({ label: "SPGrid worklist counts and published dispatches", size: this.plan.dispatchBytes,
      usage: storage | GPUBufferUsage.COPY_SRC });
    this.indirectDispatch = device.createBuffer({ label: "SPGrid live indirect dispatches", size: this.plan.dispatchBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    const damping = Math.max(0.05, Math.min(0.95, options.damping ?? 2 / 3));
    this.params = Object.freeze(Array.from({ length: this.plan.levelCount }, (_, level) => {
      const buffer = device.createBuffer({ label: `SPGrid level ${level} parameters`, size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(16), floats = new Float32Array(words.buffer);
      words.set([options.dimensions[0], options.dimensions[1], options.dimensions[2], level,
        this.plan.rowCapacity, this.plan.levelCount, this.plan.levelStride, this.plan.transferStride,
        this.plan.rowDispatch[0], this.plan.slotDispatch[0], this.plan.transferDispatch[0], this.pre]);
      words[12] = this.post; words[13] = 1; floats[14] = damping; floats[15] = options.finestCellWidth;
      device.queue.writeBuffer(buffer, 0, words); return buffer;
    }));
    const shaderModule = device.createShaderModule({ label: "Paper native sparse SPGrid V-cycle", code: octreeSPGridVCycleShader });
    const make = (entryPoint: OctreeSPGridVCyclePipelineName) => device.createComputePipeline({ label: `SPGrid V-cycle · ${entryPoint}`,
      layout: "auto", compute: { module: shaderModule, entryPoint } });
    this.pipelines = Object.freeze(Object.fromEntries((Object.keys(OCTREE_SPGRID_VCYCLE_BINDINGS) as OctreeSPGridVCyclePipelineName[]).map((name) => [name, make(name)])) as Record<OctreeSPGridVCyclePipelineName, GPUComputePipeline>);
    const l = this.plan.levelCount;
    // One conditional cold reset, live retirement per level, row-map/reset
    // bookkeeping, finest-cell emission, transfer construction, stencil
    // construction, diagonal/finalization per level, and lifecycle publish.
    this.encodedSetupDispatchCount = 4 * l + 6;
    this.encodedCorrectionDispatchCount = l + 4 + (l - 1) * (2 * this.pre + 2 * this.post + 4);
    this.encodedCorrectionPassCount = this.encodedCorrectionDispatchCount;
    this.diagnostics = Object.freeze({ levelCount: l, coarsestCapacity: this.plan.levelStride,
      maximumTransferRecordsPerLevel: this.plan.transferStride, correctionDispatchCount: this.encodedCorrectionDispatchCount,
      correctionPassTransitions: 1, restrictionScatterDispatchCount: l - 1,
      restrictionAtomicAddUpperBound: 8 * (l - 1) * this.plan.levelStride,
      parentGatherDispatchCount: l - 1, parentGatherAtomicAddCount: 0 as const,
      bottomOperation: "exact-single-cell" as const, coarsestDegreesOfFreedom: 1 as const });
    this.allocatedBytes = this.plan.allocatedBytes + this.capturedHeaders.size + this.capturedEntries.size;
  }

  encodeCapture(encoder: GPUCommandEncoder): void {
    this.assertLive();
    encoder.copyBufferToBuffer(this.source.leafHeaders, 0, this.capturedHeaders, 0, this.capturedHeaders.size);
    encoder.copyBufferToBuffer(this.source.leafEntries, 0, this.capturedEntries, 0, this.capturedEntries.size);
  }

  encodeSetup(encoder: GPUCommandEncoder, input: { solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive();
    const pass = encoder.beginComputePass({ label: "SPGrid V-cycle · rebuild native pyramid" });
    this.bind(pass, "resetInvalidBuffers", 0, input);
    pass.dispatchWorkgroupsIndirect(this.indirectDispatch, this.plan.levelCount * DISPATCH_RECORD_BYTES_PER_LEVEL + 8);
    for (let level = 0; level < this.plan.levelCount; level += 1) {
      this.bind(pass, "retireSlots", level, input);
      pass.dispatchWorkgroupsIndirect(this.indirectDispatch, level * DISPATCH_RECORD_BYTES_PER_LEVEL + 8);
    }
    this.run(pass, "retireRows", 0, input, this.plan.rowDispatch);
    this.run(pass, "resetSetupMetadata", 0, input, [1, 1, 1]);
    this.run(pass, "emitCells", 0, input, this.plan.rowDispatch);
    this.run(pass, "emitGhostAliases", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount - 1; level += 1) this.run(pass, "buildTransfers", level, input, this.plan.slotDispatch);
    this.run(pass, "buildStencil", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount; level += 1) {
      this.run(pass, "ensureDiagonal", level, input, this.plan.slotDispatch);
      this.run(pass, "finalizeIndirect", level, input, [1, 1, 1]);
    }
    this.run(pass, "finalizeLifecycle", 0, input, [1, 1, 1]);
    pass.end();
    // dispatchMeta remains STORAGE-only inside compute passes. Copying its
    // finalized records after the setup boundary gives correction a distinct
    // INDIRECT-only source and avoids a whole-pass storage/indirect conflict.
    encoder.copyBufferToBuffer(this.dispatchMeta, 0, this.indirectDispatch, 0, this.plan.dispatchBytes);
  }

  encodeCorrection(encoder: GPUCommandEncoder, input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer },
    sharedPass?: GPUComputePassEncoder): void {
    this.assertLive();
    const pass = sharedPass ?? encoder.beginComputePass({ label: "SPGrid V-cycle · one-pass symmetric correction" });
    this.run(pass, "clearCorrection", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount; level += 1) this.runIndirect(pass, "zeroVectors", level, input, false);
    this.run(pass, "seedRhs", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount - 1; level += 1) {
      this.smooth(pass, level, this.pre, input); this.runIndirect(pass, "applyA", level, input, false);
      this.runIndirect(pass, "formResidual", level, input, false);
      this.runIndirect(pass, "restrictAndGhostAccumulate", level, input, false);
    }
    this.runIndirect(pass, "exactBottom", this.plan.levelCount - 1, input, false);
    for (let level = this.plan.levelCount - 2; level >= 0; level -= 1) {
      this.runIndirect(pass, "prolongAndGhostPropagate", level, input, false);
      this.smooth(pass, level, this.post, input);
    }
    this.run(pass, "publish", 0, input, this.plan.rowDispatch); if (!sharedPass) pass.end();
  }

  private smooth(pass: GPUComputePassEncoder, level: number, iterations: number,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const b = (iteration & 1) !== 0; this.runIndirect(pass, b ? "applyB" : "applyA", level, input, false);
      this.runIndirect(pass, b ? "jacobiBtoA" : "jacobiAtoB", level, input, false);
    }
  }

  private runIndirect(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }, transfer: boolean): void {
    this.bind(pass, name, level, input);
    pass.dispatchWorkgroupsIndirect(this.indirectDispatch, level * 32 + (transfer ? 20 : 8));
  }
  private run(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer },
    dispatch: readonly [number, number, number]): void {
    this.bind(pass, name, level, input); pass.dispatchWorkgroups(...dispatch);
  }
  private bind(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    const pipeline = this.pipelines[name], key = `${name}:${level}`, cached = this.groups.get(key);
    let group = cached?.group;
    if (!cached || cached.rowCount !== input.rowCount || cached.control !== input.solverControl
      || cached.rhs !== input.rhs || cached.correction !== input.correction) {
      const buffers = [this.params[level], this.capturedHeaders, this.capturedEntries, input.rowCount, this.topology,
        this.state, this.dispatchMeta, input.solverControl, input.rhs, input.correction];
      group = this.device.createBindGroup({ label: `SPGrid V-cycle · ${name} · level ${level}`,
        layout: pipeline.getBindGroupLayout(0), entries: OCTREE_SPGRID_VCYCLE_BINDINGS[name].map((binding) => ({
        binding, resource: { buffer: buffers[binding]! },
      })) });
      this.groups.set(key, { rowCount: input.rowCount, control: input.solverControl, rhs: input.rhs, correction: input.correction, group });
    }
    pass.setPipeline(pipeline); pass.setBindGroup(0, group!);
  }
  private assertLive(): void { if (this.destroyed) throw new Error("SPGrid V-cycle is destroyed"); }
  destroy(): void {
    if (this.destroyed) return; this.destroyed = true; this.groups.clear();
    this.capturedHeaders.destroy(); this.capturedEntries.destroy(); this.topology.destroy(); this.state.destroy(); this.dispatchMeta.destroy();
    this.indirectDispatch.destroy();
    for (const buffer of this.params) buffer.destroy();
  }
}

export const octreeSPGridVCycleShader = /* wgsl */ `
struct Params{dimsLevel:vec4u,capacity:vec4u,dispatchSmooth:vec4u,solve:vec2u,weights:vec2f}
struct LeafHeader{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f}
struct LeafEntry{row:u32,coefficient:f32}
struct TransferTarget{coarse:u32,weight:f32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> entries:array<LeafEntry>;
@group(0) @binding(3) var<storage,read> rowCounts:array<u32>;
@group(0) @binding(4) var<storage,read_write> topology:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> state:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> dispatchMeta:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> inputRhs:array<f32>;
@group(0) @binding(9) var<storage,read_write> outputCorrection:array<f32>;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;const INVALID=0xffffffffu;
const OVERFLOW=2u;const NONFINITE=4u;const NONPOSITIVE=8u;
const KEY=0u;const FLAGS=1u;const DIAG=2u;const XP=3u;const XM=4u;const YP=5u;const YM=6u;const ZP=7u;const ZM=8u;
const XYPP=9u;const XYPM=10u;const XYMP=11u;const XYMM=12u;const XZPP=13u;const XZPM=14u;const XZMP=15u;const XZMM=16u;
const YZPP=17u;const YZPM=18u;const YZMP=19u;const YZMM=20u;
const RHS=21u;const A=22u;const B=23u;const AX=24u;const RESIDUAL=25u;const OWNER=26u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn stopped()->bool{return atomicLoad(&control[0])!=0u||atomicLoad(&control[1])!=0u;}
fn report(flag:u32){atomicOr(&control[0],flag);}fn rows()->u32{return min(rowCounts[0],p.capacity.x);}fn level()->u32{return p.dimsLevel.w;}
fn stride()->u32{return p.capacity.z;}fn levels()->u32{return p.capacity.y;}fn transferStride()->u32{return p.capacity.w;}
fn rowIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.x*64u;}fn slotIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.y*64u;}
fn transferIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.z*64u;}fn at(c:u32,l:u32,s:u32)->u32{return(c*levels()+l)*stride()+s;}
fn loadf(c:u32,l:u32,s:u32)->f32{return bitcast<f32>(atomicLoad(&state[at(c,l,s)]));}fn storef(c:u32,l:u32,s:u32,v:f32){atomicStore(&state[at(c,l,s)],bitcast<u32>(v));}
fn atomicAddF(index:u32,value:f32){if(!finite(value)){report(NONFINITE);return;}var old=atomicLoad(&state[index]);loop{let v=bitcast<f32>(old);
 let claim=atomicCompareExchangeWeak(&state[index],old,bitcast<u32>(v+value));if(claim.exchanged){return;}old=claim.old_value;}}
fn rowMapBase()->u32{return 16u;}fn workBase()->u32{return rowMapBase()+levels()*p.capacity.x;}
fn transferBase()->u32{return workBase()+levels()*stride();}fn rowMap(l:u32,r:u32)->u32{return atomicLoad(&topology[rowMapBase()+l*p.capacity.x+r]);}
fn workSlot(l:u32,i:u32)->u32{return atomicLoad(&topology[workBase()+l*stride()+i]);}
fn transferWord(l:u32,i:u32,w:u32)->u32{return transferBase()+(l*transferStride()+i)*3u+w;}
fn count(l:u32)->u32{return atomicLoad(&dispatchMeta[l*8u]);}fn transferCount(l:u32)->u32{return atomicLoad(&dispatchMeta[l*8u+1u]);}
fn lifecycleBase()->u32{return levels()*8u;}fn previousValid()->bool{return atomicLoad(&dispatchMeta[lifecycleBase()])==1u;}
fn previousRows()->u32{return atomicLoad(&dispatchMeta[lifecycleBase()+1u]);}
fn genericIndex(g:vec3u)->u32{return g.x+g.y*65535u*64u;}
fn dims(l:u32)->vec3u{let s=1u<<l;return (p.dimsLevel.xyz+vec3u(s-1u))/s;}fn coordKey(q:vec3u,l:u32)->u32{let d=dims(l);return q.x+d.x*(q.y+d.y*q.z)+1u;}
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;return vec3u(v%d.x,(v/d.x)%d.y,v/(d.x*d.y));}
fn hash(key:u32)->u32{var h=key*0x9e3779b1u;h=(h^(h>>16u))*0x7feb352du;return(h^(h>>15u))&(stride()-1u);}
fn mergeClass(index:u32,incoming:u32){var old=atomicLoad(&state[index]);loop{var merged=MG_ONLY;if((old&ACTIVE)!=0u||(incoming&ACTIVE)!=0u){merged=ACTIVE;}
 else if((old&GHOST)!=0u||(incoming&GHOST)!=0u){merged=GHOST;}let claim=atomicCompareExchangeWeak(&state[index],old,merged);if(claim.exchanged){return;}old=claim.old_value;}}
fn insert(l:u32,q:vec3u,flags:u32)->u32{let key=coordKey(min(q,dims(l)-vec3u(1u)),l);var slot=hash(key);for(var probe=0u;probe<256u;probe+=1u){
 let index=at(KEY,l,slot);var old=atomicLoad(&state[index]);if(old==key){mergeClass(at(FLAGS,l,slot),flags);return slot;}if(old==0u){let c=atomicCompareExchangeWeak(&state[index],0u,key);
  if(c.exchanged){mergeClass(at(FLAGS,l,slot),flags);let w=atomicAdd(&dispatchMeta[l*8u],1u);if(w>=stride()){report(OVERFLOW);return INVALID;}
   atomicStore(&topology[workBase()+l*stride()+w],slot);return slot;}old=c.old_value;if(old==key){mergeClass(at(FLAGS,l,slot),flags);return slot;}}
 slot=(slot+1u)&(stride()-1u);}report(OVERFLOW);return INVALID;}
fn insertOwned(l:u32,q:vec3u,flags:u32,owner:u32)->u32{let slot=insert(l,q,flags);if(slot==INVALID){return INVALID;}let encoded=owner+1u;
 for(var retry=0u;retry<16u;retry+=1u){let old=atomicLoad(&state[at(OWNER,l,slot)]);if(old==encoded){return slot;}if(old!=0u){report(OVERFLOW);return INVALID;}
  let claim=atomicCompareExchangeWeak(&state[at(OWNER,l,slot)],0u,encoded);if(claim.exchanged||claim.old_value==encoded){return slot;}}
 report(OVERFLOW);return INVALID;}
fn find(l:u32,q:vec3u)->u32{let key=coordKey(q,l);var slot=hash(key);for(var probe=0u;probe<256u;probe+=1u){let old=atomicLoad(&state[at(KEY,l,slot)]);
 if(old==key){return slot;}if(old==0u){return INVALID;}slot=(slot+1u)&(stride()-1u);}return INVALID;}
fn originOf(h:LeafHeader)->vec3u{return vec3u(h.cell%p.dimsLevel.x,(h.cell/p.dimsLevel.x)%p.dimsLevel.y,h.cell/(p.dimsLevel.x*p.dimsLevel.y));}
fn contactCoord(own:LeafHeader,other:LeafHeader,l:u32)->vec3u{let scale=1u<<l;let begin=originOf(own);let finish=begin+vec3u(own.size);
 let otherBegin=originOf(other);let otherFinish=otherBegin+vec3u(other.size);var result=vec3u(0u);
 for(var axis=0u;axis<3u;axis+=1u){if(otherBegin[axis]>=finish[axis]){result[axis]=(finish[axis]-1u)/scale;}
  else if(otherFinish[axis]<=begin[axis]){result[axis]=begin[axis]/scale;}else{let centre=(2u*otherBegin[axis]+other.size)/(2u*scale);
   result[axis]=clamp(centre,begin[axis]/scale,(finish[axis]-1u)/scale);}}return result;}
fn ownedContactSlot(l:u32,row:u32,other:u32)->u32{let h=headers[row];let native=firstTrailingBit(h.size);if(l>=native){return rowMap(l,row);}
 let slot=find(l,contactCoord(h,headers[other],l));if(slot==INVALID||atomicLoad(&state[at(OWNER,l,slot)])!=row+1u){return INVALID;}return slot;}
fn appendTransfer(l:u32,fine:u32,coarse:u32,weight:f32){let i=atomicAdd(&dispatchMeta[l*8u+1u],1u);if(i>=transferStride()){report(OVERFLOW);return;}
 atomicStore(&topology[transferWord(l,i,0u)],fine);atomicStore(&topology[transferWord(l,i,1u)],coarse);atomicStore(&topology[transferWord(l,i,2u)],bitcast<u32>(weight));}
// A successful generation records every occupied hash slot exactly once in
// its worklist. Retiring that live prefix is therefore equivalent to clearing
// the full sparse arena, without writing tens of megabytes of empty capacity.
@compute @workgroup_size(64) fn resetInvalidBuffers(@builtin(global_invocation_id) g:vec3u){let i=genericIndex(g);
 if(i<arrayLength(&topology)){atomicStore(&topology[i],0u);}if(i<arrayLength(&state)){atomicStore(&state[i],0u);}}
@compute @workgroup_size(64) fn retireSlots(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();
 if(!previousValid()||i>=count(l)){return;}let s=workSlot(l,i);for(var c=0u;c<27u;c+=1u){atomicStore(&state[at(c,l,s)],0u);}}
@compute @workgroup_size(64) fn retireRows(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);
 if(!previousValid()||r>=max(rows(),previousRows())){return;}for(var l=0u;l<levels();l+=1u){atomicStore(&topology[rowMapBase()+l*p.capacity.x+r],0u);}}
@compute @workgroup_size(1) fn resetSetupMetadata(){for(var i=0u;i<levels()*8u;i+=1u){atomicStore(&dispatchMeta[i],0u);}
 let base=lifecycleBase();atomicStore(&dispatchMeta[base],0u);atomicStore(&dispatchMeta[base+1u],0u);
 for(var i=2u;i<8u;i+=1u){atomicStore(&dispatchMeta[base+i],0u);}}
@compute @workgroup_size(64) fn clearCorrection(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){outputCorrection[r]=0.0;}}
@compute @workgroup_size(64) fn emitCells(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r>=rows()||stopped()){return;}let h=headers[r];
 let native=firstTrailingBit(h.size);for(var l=native;l<levels();l+=1u){let scale=1u<<l;let flag=select(MG_ONLY,ACTIVE,l==native);
  let q=originOf(h)/scale;var slot=INVALID;if(l==native){slot=insertOwned(l,q,flag,r);}else{slot=insert(l,q,flag);}
  if(slot!=INVALID){atomicStore(&topology[rowMapBase()+l*p.capacity.x+r],slot);}}}
@compute @workgroup_size(64) fn emitGhostAliases(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r>=rows()||stopped()){return;}let h=headers[r];
 if(h.entryStart+h.entryCount>arrayLength(&entries)){report(OVERFLOW);return;}let native=firstTrailingBit(h.size);for(var j=0u;j<h.entryCount;j+=1u){let other=entries[h.entryStart+j].row;
  if(other>=rows()){report(OVERFLOW);continue;}let otherNative=firstTrailingBit(headers[other].size);for(var l=otherNative;l<native;l+=1u){
   let ghostSlot=insertOwned(l,contactCoord(h,headers[other],l),GHOST,r);if(ghostSlot==INVALID){return;}}}}
@compute @workgroup_size(64) fn buildTransfers(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||l+1u>=levels()||stopped()){return;}
 let fine=workSlot(l,i);let q=decode(atomicLoad(&state[at(KEY,l,fine)]),l);let flags=atomicLoad(&state[at(FLAGS,l,fine)]);if((flags&GHOST)!=0u){let encodedOwner=atomicLoad(&state[at(OWNER,l,fine)]);
  if(encodedOwner==0u||encodedOwner>rows()){report(OVERFLOW);return;}let owner=encodedOwner-1u;let native=firstTrailingBit(headers[owner].size);var c=INVALID;
  if(l+1u==native){c=rowMap(l+1u,owner);}else{c=insertOwned(l+1u,q/2u,GHOST,owner);}if(c!=INVALID){appendTransfer(l,fine,c,1.0);}return;}
 let base=q/2u;let side=vec3i(select(-1,1,(q.x&1u)!=0u),select(-1,1,(q.y&1u)!=0u),select(-1,1,(q.z&1u)!=0u));
 for(var corner=0u;corner<8u;corner+=1u){var targetCoord=vec3i(base);var weight=1.0;for(var axis=0u;axis<3u;axis+=1u){if((corner&(1u<<axis))!=0u){targetCoord[axis]+=side[axis];weight*=0.25;}else{weight*=0.75;}}
  let cq=vec3u(max(targetCoord,vec3i(0)));let c=insert(l+1u,cq,MG_ONLY);if(c!=INVALID){appendTransfer(l,fine,c,weight);}}}
fn addFace(l:u32,own:u32,a:vec3u,b:vec3u,c:f32)->bool{let d=vec3i(b)-vec3i(a);var ch=0u;if(all(d==vec3i(1,0,0))){ch=XP;}else if(all(d==vec3i(-1,0,0))){ch=XM;}
 else if(all(d==vec3i(0,1,0))){ch=YP;}else if(all(d==vec3i(0,-1,0))){ch=YM;}else if(all(d==vec3i(0,0,1))){ch=ZP;}else if(all(d==vec3i(0,0,-1))){ch=ZM;}
 else if(all(d==vec3i(1,1,0))){ch=XYPP;}else if(all(d==vec3i(1,-1,0))){ch=XYPM;}else if(all(d==vec3i(-1,1,0))){ch=XYMP;}else if(all(d==vec3i(-1,-1,0))){ch=XYMM;}
 else if(all(d==vec3i(1,0,1))){ch=XZPP;}else if(all(d==vec3i(1,0,-1))){ch=XZPM;}else if(all(d==vec3i(-1,0,1))){ch=XZMP;}else if(all(d==vec3i(-1,0,-1))){ch=XZMM;}
 else if(all(d==vec3i(0,1,1))){ch=YZPP;}else if(all(d==vec3i(0,1,-1))){ch=YZPM;}else if(all(d==vec3i(0,-1,1))){ch=YZMP;}else if(all(d==vec3i(0,-1,-1))){ch=YZMM;}else{return false;}
 atomicAddF(at(ch,l,own),c);return true;}
@compute @workgroup_size(64) fn buildStencil(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r>=rows()||stopped()){return;}let h=headers[r];var sum=0.0;
 if(h.entryStart+h.entryCount>arrayLength(&entries)){report(OVERFLOW);return;}for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=rows()||!finite(e.coefficient)||e.coefficient<0.0){report(OVERFLOW);continue;}sum+=e.coefficient;}
 let anchor=max(0.0,h.diagonal-sum);let native=firstTrailingBit(h.size);for(var l=0u;l<levels();l+=1u){if(l>=native){let canonical=rowMap(l,r);atomicAddF(at(DIAG,l,canonical),anchor);}
  for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=rows()){continue;}let otherNative=firstTrailingBit(headers[e.row].size);
   if(l<native&&l<otherNative){continue;}let own=ownedContactSlot(l,r,e.row);let other=ownedContactSlot(l,e.row,r);
   if(own==INVALID||other==INVALID){report(OVERFLOW);continue;}if(other!=own){atomicAddF(at(DIAG,l,own),e.coefficient);let ownQ=decode(atomicLoad(&state[at(KEY,l,own)]),l);
    if(!addFace(l,own,ownQ,decode(atomicLoad(&state[at(KEY,l,other)]),l),e.coefficient)){report(OVERFLOW);}}}}}
@compute @workgroup_size(64) fn ensureDiagonal(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)){return;}let s=workSlot(l,i);
 if(loadf(DIAG,l,s)<=1e-20){storef(DIAG,l,s,1.0);}}
@compute @workgroup_size(1) fn finalizeIndirect(){let l=level();let n=count(l);let t=transferCount(l);let nb=(n+63u)/64u;let tb=(t+63u)/64u;
 atomicStore(&dispatchMeta[l*8u+2u],min(65535u,max(1u,nb)));atomicStore(&dispatchMeta[l*8u+3u],max(1u,(nb+65534u)/65535u));atomicStore(&dispatchMeta[l*8u+4u],1u);
 atomicStore(&dispatchMeta[l*8u+5u],min(65535u,max(1u,tb)));atomicStore(&dispatchMeta[l*8u+6u],max(1u,(tb+65534u)/65535u));atomicStore(&dispatchMeta[l*8u+7u],1u);}
@compute @workgroup_size(1) fn finalizeLifecycle(){let base=lifecycleBase();if(atomicLoad(&control[0])==0u){
 atomicStore(&dispatchMeta[base],1u);atomicStore(&dispatchMeta[base+1u],rows());return;}
 // A failed insertion may have claimed a hash slot before discovering that
 // its live worklist overflowed. Publish a conditional full reset for the
 // next generation rather than trusting an incomplete retirement list.
 let words=max(arrayLength(&topology),arrayLength(&state));let blocks=(words+63u)/64u;let x=min(65535u,blocks);
 atomicStore(&dispatchMeta[base],0u);atomicStore(&dispatchMeta[base+1u],0u);atomicStore(&dispatchMeta[base+2u],x);
 atomicStore(&dispatchMeta[base+3u],select(0u,(blocks+x-1u)/x,x>0u));atomicStore(&dispatchMeta[base+4u],1u);}
@compute @workgroup_size(64) fn zeroVectors(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||stopped()){return;}let s=workSlot(l,i);
 for(var c=RHS;c<=RESIDUAL;c+=1u){storef(c,l,s,0.0);}}
@compute @workgroup_size(64) fn seedRhs(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let v=inputRhs[r];let native=firstTrailingBit(headers[r].size);
 if(!finite(v)){report(NONFINITE);}else{atomicAddF(at(RHS,native,rowMap(native,r)),v);}}}
fn neighbour(l:u32,q:vec3u,axis:u32,positive:bool)->u32{var v=vec3i(q);v[axis]+=select(-1,1,positive);if(any(v<vec3i(0))||any(vec3u(v)>=dims(l))){return INVALID;}return find(l,vec3u(v));}
fn offsetNeighbour(l:u32,q:vec3u,d:vec3i)->u32{let v=vec3i(q)+d;if(any(v<vec3i(0))||any(vec3u(v)>=dims(l))){return INVALID;}return find(l,vec3u(v));}
fn apply(slot:u32,source:u32){let l=level();let q=decode(atomicLoad(&state[at(KEY,l,slot)]),l);var value=loadf(DIAG,l,slot)*loadf(source,l,slot);
 let channels=array<u32,6>(XP,XM,YP,YM,ZP,ZM);for(var k=0u;k<6u;k+=1u){let c=loadf(channels[k],l,slot);let other=neighbour(l,q,k/2u,(k&1u)==0u);
  if(other!=INVALID){value-=c*loadf(source,l,other);}}
 let edgeChannels=array<u32,12>(XYPP,XYPM,XYMP,XYMM,XZPP,XZPM,XZMP,XZMM,YZPP,YZPM,YZMP,YZMM);
 let edgeOffsets=array<vec3i,12>(vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),
  vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));
 for(var k=0u;k<12u;k+=1u){let c=loadf(edgeChannels[k],l,slot);let other=offsetNeighbour(l,q,edgeOffsets[k]);if(other!=INVALID){value-=c*loadf(source,l,other);}}
 storef(AX,l,slot,value);}
fn smoothable(l:u32,s:u32)->bool{return(atomicLoad(&state[at(FLAGS,l,s)])&GHOST)==0u;}
@compute @workgroup_size(64) fn applyA(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){apply(workSlot(l,i),A);}}
@compute @workgroup_size(64) fn applyB(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){apply(workSlot(l,i),B);}}
fn jacobi(slot:u32,src:u32,dst:u32){let l=level();let d=loadf(DIAG,l,slot);if(!(d>0.0)){report(NONPOSITIVE);return;}let x=loadf(src,l,slot)+p.weights.x*(loadf(RHS,l,slot)-loadf(AX,l,slot))/d;
 if(!finite(x)){report(NONFINITE);}else{storef(dst,l,slot,x);}}
@compute @workgroup_size(64) fn jacobiAtoB(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){let s=workSlot(l,i);
 if(smoothable(l,s)){jacobi(s,A,B);}else{storef(B,l,s,loadf(A,l,s));}}}
@compute @workgroup_size(64) fn jacobiBtoA(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){let s=workSlot(l,i);
 if(smoothable(l,s)){jacobi(s,B,A);}else{storef(A,l,s,loadf(B,l,s));}}}
@compute @workgroup_size(64) fn formResidual(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){let s=workSlot(l,i);
 storef(RESIDUAL,l,s,select(-loadf(AX,l,s),loadf(RHS,l,s)-loadf(AX,l,s),smoothable(l,s)));}}
// Return the unique transfer target owned by one fine slot/corner. Both E^T
// restriction and E prolongation call this function, so their weights cannot
// diverge. Setup has already inserted every returned coarse cell.
fn correctionTransfer(l:u32,fine:u32,corner:u32)->TransferTarget{let flags=atomicLoad(&state[at(FLAGS,l,fine)]);
 let q=decode(atomicLoad(&state[at(KEY,l,fine)]),l);if((flags&GHOST)!=0u){if(corner!=0u){return TransferTarget(INVALID,0.0);}
  let encodedOwner=atomicLoad(&state[at(OWNER,l,fine)]);if(encodedOwner==0u||encodedOwner>rows()){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
  let owner=encodedOwner-1u;let native=firstTrailingBit(headers[owner].size);let coarse=select(find(l+1u,q/2u),rowMap(l+1u,owner),l+1u==native);
  if(coarse==INVALID){report(OVERFLOW);}return TransferTarget(coarse,1.0);}
 if(corner>=8u){return TransferTarget(INVALID,0.0);}let base=q/2u;
 let side=vec3i(select(-1,1,(q.x&1u)!=0u),select(-1,1,(q.y&1u)!=0u),select(-1,1,(q.z&1u)!=0u));var targetCoord=vec3i(base);var weight=1.0;
 for(var axis=0u;axis<3u;axis+=1u){if((corner&(1u<<axis))!=0u){targetCoord[axis]+=side[axis];weight*=0.25;}else{weight*=0.75;}}
 let cq=min(vec3u(max(targetCoord,vec3i(0))),dims(l+1u)-vec3u(1u));let coarse=find(l+1u,cq);if(coarse==INVALID){report(OVERFLOW);}
 return TransferTarget(coarse,weight);}
// Section 4.2 restriction and GhostValueAccumulate share one fine-owned
// dispatch. Coarse destinations still require atomic sums until setup publishes
// the inverse parent CSR; the exact E^T weights are shared with prolongation.
@compute @workgroup_size(64) fn restrictAndGhostAccumulate(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||stopped()){return;}
 let fine=workSlot(l,i);let ghost=(atomicLoad(&state[at(FLAGS,l,fine)])&GHOST)!=0u;let residualValue=loadf(RESIDUAL,l,fine);let targetCount=select(8u,1u,ghost);
 for(var corner=0u;corner<targetCount;corner+=1u){let transfer=correctionTransfer(l,fine,corner);if(transfer.coarse==INVALID){return;}
  atomicAddF(at(RHS,l+1u,transfer.coarse),transfer.weight*residualValue);}}
@compute @workgroup_size(64) fn exactBottom(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>0u||stopped()){return;}if(count(l)!=1u){report(NONPOSITIVE);return;}
 let s=workSlot(l,0u);let d=loadf(DIAG,l,s);if(!(d>0.0)){report(NONPOSITIVE);return;}let x=loadf(RHS,l,s)/d;if(!finite(x)){report(NONFINITE);}else{storef(A,l,s,x);}}
// One fine invocation owns the complete interpolation sum, deleting all
// prolongation atomics. GhostValuePropagate is the unit-copy branch of the
// same E mapping rather than a second dispatch.
@compute @workgroup_size(64) fn prolongAndGhostPropagate(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||stopped()){return;}
 let fine=workSlot(l,i);let ghost=(atomicLoad(&state[at(FLAGS,l,fine)])&GHOST)!=0u;let targetCount=select(8u,1u,ghost);var value=select(loadf(A,l,fine),0.0,ghost);
 for(var corner=0u;corner<targetCount;corner+=1u){let transfer=correctionTransfer(l,fine,corner);if(transfer.coarse==INVALID){return;}
  value+=transfer.weight*loadf(A,l+1u,transfer.coarse);}if(!finite(value)){report(NONFINITE);}else{storef(A,l,fine,value);}}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let native=firstTrailingBit(headers[r].size);
 let v=loadf(A,native,rowMap(native,r));if(!finite(v)){report(NONFINITE);}else{outputCorrection[r]=v;}}}
`;
