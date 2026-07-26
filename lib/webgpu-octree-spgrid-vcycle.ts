import {
  OCTREE_FIRST_ORDER_CHEBYSHEV_DEGREES,
  OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION,
  type OctreeFirstOrderSPDVCycle,
} from "./webgpu-octree-section43-contract";
import { PassBroker } from "./webgpu-pass-broker";
import type { OctreePipelinedWorksetLinearOperator } from "./webgpu-octree-pipelined-mgpcg";
import { octreeAlgorithmDiagnosticsEnabled } from "./octree-algorithm-diagnostics";

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
  /** Maximum exact per-level capacity. Direct dispatch bounds continue to use
   * this maximum; arena addressing uses levelCapacities/levelOffsets. */
  readonly levelStride: number;
  readonly transferStride: number;
  readonly levelCapacities: readonly number[];
  readonly levelOffsets: readonly number[];
  readonly totalLevelSlots: number;
  readonly transferCapacities: readonly number[];
  readonly transferOffsets: readonly number[];
  readonly topologyBytes: number;
  readonly stateBytes: number;
  readonly dispatchBytes: number;
  /** Eight-word exact affected-row/publication record per sparse level. */
  readonly levelDeltaBytes: number;
  readonly brickCount: number;
  readonly directoryBytes: number;
  /** Dense logical-page directory plus immutable key+27-neighbour records. */
  readonly pageDirectoryBytes: number;
  readonly pageRecordWords: 28;
  /** Exact 64-row L1 page transaction: twelve publication words followed by
   * four exclusive words per page (generation/change/validation/copy). */
  readonly capturePageStateBytes: number;
  readonly capturePageCount: number;
  readonly allocatedBytes: number;
  readonly rowDispatch: readonly [number, number, number];
  readonly slotDispatch: readonly [number, number, number];
  readonly transferDispatch: readonly [number, number, number];
  readonly brickDispatch: readonly [number, number, number];
}

export interface SPGridBrickRankDirectory {
  readonly dimensions: readonly [number, number, number];
  readonly masks: readonly (readonly [number, number])[];
  readonly bases: readonly number[];
  readonly slots: readonly number[];
}

export interface SPGridPhysicalPageAdjacency {
  readonly dimensions: readonly [number, number, number];
  readonly pageShape: readonly [8, 8, 4];
  /** Stable physical IDs are assigned in dense logical-page order. */
  readonly origins: readonly (readonly [number, number, number])[];
  /** Twenty-seven physical IDs in z/y/x order, with -1 outside/empty. */
  readonly neighbours: readonly (readonly number[])[];
  readonly logicalToPhysical: readonly number[];
}

/** CPU oracle for the pressure/MG page publication. A page is present iff it
 * owns at least one sparse slot. Its 27-entry record contains physical page
 * IDs, never coordinates or search keys, and is therefore directly usable by
 * halo staging after the epoch is accepted. */
export function buildSPGridPhysicalPageAdjacency(
  dimensions: readonly [number, number, number],
  coordinates: readonly (readonly [number, number, number])[],
): SPGridPhysicalPageAdjacency {
  dimensions.forEach((value) => positiveInteger(value, "SPGrid page dimension"));
  const pageShape = [8, 8, 4] as const;
  const pageDimensions = dimensions.map((value, axis) =>
    Math.ceil(value / pageShape[axis])) as [number, number, number];
  const logicalCount = pageDimensions[0] * pageDimensions[1] * pageDimensions[2];
  const occupied = new Uint8Array(logicalCount);
  for (const coordinate of coordinates) {
    if (coordinate.some((value, axis) => !Number.isSafeInteger(value)
      || value < 0 || value >= dimensions[axis])) {
      throw new RangeError("Malformed SPGrid page coordinate");
    }
    const page = coordinate.map((value, axis) => Math.floor(value / pageShape[axis]));
    occupied[page[0] + pageDimensions[0] * (page[1] + pageDimensions[1] * page[2])] = 1;
  }
  const logicalToPhysical = new Array<number>(logicalCount).fill(-1);
  const origins: Array<readonly [number, number, number]> = [];
  for (let logical = 0; logical < logicalCount; logical += 1) if (occupied[logical] !== 0) {
    const x = logical % pageDimensions[0];
    const yz = Math.floor(logical / pageDimensions[0]);
    const y = yz % pageDimensions[1], z = Math.floor(yz / pageDimensions[1]);
    logicalToPhysical[logical] = origins.length;
    origins.push([x * pageShape[0], y * pageShape[1], z * pageShape[2]]);
  }
  const neighbours = origins.map((origin) => {
    const page = origin.map((value, axis) => value / pageShape[axis]);
    const result: number[] = [];
    for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
      const q = [page[0] + dx, page[1] + dy, page[2] + dz];
      result.push(q.some((value, axis) => value < 0 || value >= pageDimensions[axis])
        ? -1 : logicalToPhysical[q[0] + pageDimensions[0] * (q[1] + pageDimensions[1] * q[2])]);
    }
    return Object.freeze(result);
  });
  return Object.freeze({ dimensions, pageShape, origins: Object.freeze(origins),
    neighbours: Object.freeze(neighbours), logicalToPhysical: Object.freeze(logicalToPhysical) });
}

/** Exhaustive CPU oracle for the authoritative 4^3 mask/rank directory. */
export function buildSPGridBrickRankDirectory(dimensions: readonly [number, number, number],
  cells: readonly { readonly coordinate: readonly [number, number, number]; readonly slot: number }[]): SPGridBrickRankDirectory {
  dimensions.forEach((value) => positiveInteger(value, "SPGrid directory dimension"));
  const brickDims = dimensions.map((value) => Math.ceil(value / 4)) as [number, number, number];
  const brickCount = brickDims[0] * brickDims[1] * brickDims[2];
  const masks = Array.from({ length: brickCount }, () => [0, 0] as [number, number]);
  const seen = new Set<string>(), seenSlots = new Set<number>();
  for (const { coordinate, slot } of cells) {
    if (!Number.isSafeInteger(slot) || slot < 0 || coordinate.some((v, axis) => !Number.isSafeInteger(v) || v < 0 || v >= dimensions[axis])) {
      throw new RangeError("Malformed SPGrid brick-directory cell");
    }
    const key = coordinateKey(coordinate);
    if (seen.has(key) || seenSlots.has(slot)) throw new RangeError("Duplicate SPGrid brick-directory cell or slot");
    seen.add(key); seenSlots.add(slot);
    const brick = Math.floor(coordinate[0] / 4) + brickDims[0]
      * (Math.floor(coordinate[1] / 4) + brickDims[1] * Math.floor(coordinate[2] / 4));
    const bit = coordinate[0] % 4 + 4 * (coordinate[1] % 4) + 16 * (coordinate[2] % 4);
    masks[brick][bit >>> 5] = (masks[brick][bit >>> 5] | (1 << (bit & 31))) >>> 0;
  }
  const bases = new Array<number>(brickCount); let total = 0;
  for (let brick = 0; brick < brickCount; brick += 1) {
    bases[brick] = total; total += popcount32(masks[brick][0]) + popcount32(masks[brick][1]);
  }
  const slots = new Array<number>(total);
  for (const { coordinate, slot } of cells) {
    const brick = Math.floor(coordinate[0] / 4) + brickDims[0]
      * (Math.floor(coordinate[1] / 4) + brickDims[1] * Math.floor(coordinate[2] / 4));
    const bit = coordinate[0] % 4 + 4 * (coordinate[1] % 4) + 16 * (coordinate[2] % 4);
    const low = masks[brick][0], before = bit < 32 ? (low & lowerBitsMask32(bit)) >>> 0 : low;
    const highBefore = bit < 32 ? 0 : (masks[brick][1] & lowerBitsMask32(bit - 32)) >>> 0;
    slots[bases[brick] + popcount32(before) + popcount32(highBefore)] = slot;
  }
  const result = { dimensions, masks, bases, slots };
  validateSPGridBrickRankDirectory(result);
  return result;
}

function popcount32(value: number): number { value >>>= 0; let count = 0; while (value !== 0) { value &= value - 1; count += 1; } return count; }
function lowerBitsMask32(bitCount: number): number {
  if (bitCount <= 0) return 0;
  if (bitCount >= 32) return 0xffff_ffff;
  return (2 ** bitCount - 1) >>> 0;
}

/** Reject a torn or malformed CPU publication rather than returning a slot
 * from unrelated storage. This mirrors the GPU publication gate and key
 * revalidation used by every production lookup. */
export function validateSPGridBrickRankDirectory(directory: SPGridBrickRankDirectory): void {
  directory.dimensions.forEach((value) => positiveInteger(value, "SPGrid directory dimension"));
  const bd = directory.dimensions.map((value) => Math.ceil(value / 4));
  const brickCount = bd[0] * bd[1] * bd[2];
  if (directory.masks.length !== brickCount || directory.bases.length !== brickCount) {
    throw new RangeError("Malformed SPGrid brick-directory publication");
  }
  let expectedBase = 0;
  for (let brick = 0; brick < brickCount; brick += 1) {
    const mask = directory.masks[brick];
    if (!mask || mask.length !== 2 || mask.some((word) => !Number.isSafeInteger(word) || word < 0 || word > 0xffff_ffff)
      || directory.bases[brick] !== expectedBase) {
      throw new RangeError("Malformed SPGrid brick-directory publication");
    }
    const bx = brick % bd[0], by = Math.floor(brick / bd[0]) % bd[1], bz = Math.floor(brick / (bd[0] * bd[1]));
    for (let bit = 0; bit < 64; bit += 1) if (((mask[bit >>> 5] >>> (bit & 31)) & 1) !== 0) {
      const coordinate = [4 * bx + bit % 4, 4 * by + Math.floor(bit / 4) % 4, 4 * bz + Math.floor(bit / 16)];
      if (coordinate.some((value, axis) => value >= directory.dimensions[axis])) {
        throw new RangeError("Malformed SPGrid brick-directory publication");
      }
    }
    expectedBase += popcount32(mask[0]) + popcount32(mask[1]);
  }
  if (expectedBase !== directory.slots.length
    || directory.slots.some((slot) => !Number.isSafeInteger(slot) || slot < 0)
    || new Set(directory.slots).size !== directory.slots.length) {
    throw new RangeError("Malformed SPGrid brick-directory publication");
  }
}

export function lookupSPGridBrickRank(directory: SPGridBrickRankDirectory,
  coordinate: readonly [number, number, number]): number | undefined {
  validateSPGridBrickRankDirectory(directory);
  if (coordinate.some((v, axis) => !Number.isSafeInteger(v) || v < 0 || v >= directory.dimensions[axis])) return undefined;
  const bd = directory.dimensions.map((value) => Math.ceil(value / 4));
  const brick = Math.floor(coordinate[0] / 4) + bd[0] * (Math.floor(coordinate[1] / 4) + bd[1] * Math.floor(coordinate[2] / 4));
  const bit = coordinate[0] % 4 + 4 * (coordinate[1] % 4) + 16 * (coordinate[2] % 4);
  const [low, high] = directory.masks[brick], word = bit < 32 ? low : high, flag = (1 << (bit & 31)) >>> 0;
  if (((word >>> 0) & flag) === 0) return undefined;
  const before = bit < 32 ? (low & lowerBitsMask32(bit)) >>> 0 : low;
  const highBefore = bit < 32 ? 0 : (high & lowerBitsMask32(bit - 32)) >>> 0;
  return directory.slots[directory.bases[brick] + popcount32(before) + popcount32(highBefore)];
}

export interface OctreeSPGridVCycleOptions {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
  readonly finestCellWidth: number;
  readonly maximumLevels?: number;
  readonly preSmoothingIterations?: number;
  readonly postSmoothingIterations?: number;
  readonly bottomIterations?: number;
}

export interface SPGridScaledSpectralBounds {
  readonly lower: number;
  readonly upper: number;
}

/**
 * CPU oracle for the bound published transactionally by the GPU hierarchy.
 * A first-order M-matrix has D^-1 A Gershgorin discs centred at one.  The
 * small outward pad covers f32 accumulation/rounding at publication.
 */
export function computeSPGridScaledSpectralBounds(
  rows: readonly { readonly diagonal: number; readonly offDiagonalSum: number }[],
): SPGridScaledSpectralBounds {
  if (rows.length === 0) throw new RangeError("SPGrid spectral publication requires at least one row");
  let upper = 1;
  for (const [index, row] of rows.entries()) {
    if (!(row.diagonal > 0) || !Number.isFinite(row.diagonal)
      || !(row.offDiagonalSum >= 0) || !Number.isFinite(row.offDiagonalSum)
      || row.offDiagonalSum > row.diagonal * (1 + 1e-4)) {
      throw new RangeError(`SPGrid scaled operator row ${index} is not a valid first-order M-matrix row`);
    }
    upper = Math.max(upper, 1 + row.offDiagonalSum / row.diagonal);
  }
  upper *= 1.0005;
  return Object.freeze({
    lower: upper * OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION,
    upper,
  });
}

/** Fixed roots of the degree-2/4 Chebyshev error polynomial. */
export function spgridChebyshevRelaxationWeight(
  bounds: SPGridScaledSpectralBounds,
  degree: 2 | 4,
  phase: number,
): number {
  if (!OCTREE_FIRST_ORDER_CHEBYSHEV_DEGREES.includes(degree)
    || !Number.isSafeInteger(phase) || phase < 0 || phase >= degree
    || !(bounds.lower > 0) || !(bounds.upper > bounds.lower)
    || !Number.isFinite(bounds.lower) || !Number.isFinite(bounds.upper)) {
    throw new RangeError("Invalid SPGrid Chebyshev schedule");
  }
  const centre = 0.5 * (bounds.upper + bounds.lower);
  const radius = 0.5 * (bounds.upper - bounds.lower);
  return 1 / (centre - radius * Math.cos(Math.PI * (2 * phase + 1) / (2 * degree)));
}

export interface OctreeSPGridL1DeltaSource {
  /** Exact old/new row transaction published by the leaf-frontier producer. */
  readonly rows: GPUBuffer;
  readonly rowCapacity: number;
  readonly controlOffsetWords: number;
  readonly newToOldOffsetWords: number;
  readonly dirtyRowsOffsetWords: number;
}

export type OctreeSPGridWorksetBinding = GPUBuffer | Readonly<{
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}>;

export interface OctreeSPGridVCycleSource {
  readonly rowCapacity: number;
  readonly control: GPUBuffer;
  readonly worksets: Readonly<Record<
    "regularInterior" | "transitionInterior" | "physicalBoundary" | "transitionBoundary",
    OctreeSPGridWorksetBinding
  >>;
  /** Fixed vec4u per row: cell-linear, size-in-finest-cells, physical page, local cell. */
  readonly rowGeometry: GPUBuffer;
  readonly rowDelta: OctreeSPGridL1DeltaSource;
  readonly classDispatch?: GPUBuffer;
  readonly classDispatchOffsetBytes?: number;
  readonly worksetStrideWords?: number;
  readonly worksetBankStrideWords?: number;
  /** Accepted per-row case/transform metrics and immutable §6.3 table. */
  readonly topologyMetrics: GPUBuffer;
  readonly catalogCoefficients: GPUBuffer;
  /** Accepted banked dynamic diagonal + eighteen canonical coefficients. */
  readonly coefficients: GPUBuffer;
  readonly coefficientBankStrideWords: number;
}

// Key/class/diagonal, six Cartesian and twelve octree-edge coefficients, three
// vectors, the adaptive owner of active/ghost storage, and one transactional
// scaled-operator spectral publication per level. Accurate A2 reads the
// accepted case/transform plus the immutable 19-channel catalog directly; it
// is not duplicated in this first-order rediscretized MG state.
const STATE_CHANNELS = 26;
const TOPOLOGY_HEADER_WORDS = 16;
const DISPATCH_RECORD_WORDS_PER_LEVEL = 12;
const DISPATCH_RECORD_BYTES_PER_LEVEL = DISPATCH_RECORD_WORDS_PER_LEVEL * 4;
// Only valid + published-row-count remain. The deleted six-word tail encoded
// the former full-capacity recovery dispatch.
const DISPATCH_LIFECYCLE_BYTES = 8;
const CAPTURE_PAGE_ROWS = 64;
const CAPTURE_PUBLICATION_WORDS = 12;
const LEVEL_DELTA_WORDS = 8;
const PAGE_RECORD_WORDS = 28;
// Per-level uniform parameters plus the memoized level tables. Every kernel
// used to recompute levelCapacity/levelBase/brick/page/transfer offsets with
// nested loops on every address; the tables make each one a single uniform
// read and are the exact values the CPU plan already allocated for.
const PARAMS_LEVEL_TABLE_SLOTS = 16;
const PARAMS_BYTES = 80 + 16 + 5 * PARAMS_LEVEL_TABLE_SLOTS * 4;
/** Per-row ghost-alias detection record: one channel mask plus 18 owners. */
const GHOST_SCRATCH_WORDS_PER_ROW = 20;
/** valid/rows/mismatch-generation header plus the per-row build fingerprint. */
const COMMITTED_INPUT_HEADER_WORDS = 4;
const COMMITTED_INPUT_WORDS_PER_ROW = 4;
/** Explicit row bound; the constructor also fails closed on device limits. */
export const SPGRID_MAXIMUM_ROW_CAPACITY = 16_384;
export const OCTREE_SPGRID_CAPTURE_CONTROL_WORD = Object.freeze({
  readyGeneration: 4,
  error: 8,
  sourceGeneration: 10,
} as const);

type OctreeSPGridSourceMode = "accepted" | "candidate";
type OctreeSPGridSetupSource = {
  readonly solverControl: GPUBuffer;
  readonly rowCount: GPUBuffer;
  readonly sourceControl?: GPUBuffer;
  readonly topologyMetrics?: GPUBuffer;
  readonly sourceMode: OctreeSPGridSourceMode;
};

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
  // ghosts. The former layout reserved that finest-level bound at every
  // level. A level cannot contain more unique coordinates than its domain,
  // so cap each power-of-two hash arena by that exact domain cardinality.
  const maximumSparseSlots = nextPowerOfTwo(rowCapacity * 16);
  const levelCapacities = Array.from({ length: levelCount }, (_, level) => {
    const scale = 2 ** level;
    const domainCells = options.dimensions
      .map((value) => Math.ceil(value / scale))
      .reduce((product, value) => product * value, 1);
    // Preserve the original open-address table's <=50% load guarantee even
    // when a coarse level occupies every coordinate in its dense domain.
    return nextPowerOfTwo(Math.min(maximumSparseSlots, 2 * domainCells));
  });
  const levelOffsets: number[] = [];
  let totalLevelSlots = 0;
  for (const capacity of levelCapacities) {
    levelOffsets.push(totalLevelSlots);
    totalLevelSlots += capacity;
  }
  const levelStride = Math.max(...levelCapacities);
  const transferStride = rowCapacity * 8;
  const transferCapacities = levelCapacities.slice(0, -1)
    .map((capacity) => Math.min(transferStride, capacity * 8));
  const transferOffsets: number[] = [];
  let transferWords = 0;
  for (let level = 0; level < transferCapacities.length; level += 1) {
    transferOffsets.push(transferWords);
    transferWords += transferCapacities[level] * 4 + 4 * levelCapacities[level];
  }
  const rowMapWords = levelCount * rowCapacity, worklistWords = totalLevelSlots;
  const pageWorklistWords = PAGE_RECORD_WORDS * totalLevelSlots;
  let pageDirectoryWords = 0;
  for (let level = 0; level < levelCount; level += 1) {
    const scale = 2 ** level;
    const levelDimensions = options.dimensions.map((value) => Math.ceil(value / scale));
    pageDirectoryWords += Math.ceil(levelDimensions[0] / 8)
      * Math.ceil(levelDimensions[1] / 8) * Math.ceil(levelDimensions[2] / 4);
  }
  // Four immutable words per transfer (fine, coarse, weight, next), followed
  // by parent head/tail slots for deterministic restriction and fine
  // head/count slots for direct indexed prolongation.
  let brickCount = 0;
  for (let level = 0; level < levelCount; level += 1) {
    const scale = 2 ** level;
    brickCount += options.dimensions.map((value) => Math.ceil(Math.ceil(value / scale) / 4))
      .reduce((product, value) => product * value, 1);
  }
  // Sixteen publication words, four words per 4^3 brick (generation, two
  // occupancy masks, ranked base), and one compact ranked slot vector/level.
  const directoryWords = 16 + 4 * brickCount + totalLevelSlots;
  const directoryBytes = directoryWords * 4;
  const topologyBytes = (TOPOLOGY_HEADER_WORDS + rowMapWords + worklistWords
    + pageWorklistWords + pageDirectoryWords
    + transferWords + directoryWords) * 4;
  const stateBytes = STATE_CHANNELS * totalLevelSlots * 4;
  const dispatchBytes = levelCount * DISPATCH_RECORD_BYTES_PER_LEVEL + DISPATCH_LIFECYCLE_BYTES;
  const levelDeltaBytes = levelCount * LEVEL_DELTA_WORDS * 4;
  const capturePageCount = Math.ceil(rowCapacity / CAPTURE_PAGE_ROWS);
  const capturePageStateBytes = (CAPTURE_PUBLICATION_WORDS + 4 * capturePageCount) * 4;
  return { rowCapacity, levelCount, levelStride, transferStride,
    levelCapacities: Object.freeze(levelCapacities),
    levelOffsets: Object.freeze(levelOffsets), totalLevelSlots,
    transferCapacities: Object.freeze(transferCapacities),
    transferOffsets: Object.freeze(transferOffsets),
    topologyBytes, stateBytes, dispatchBytes,
    // Storage-authored counts and indirect arguments must be separate WebGPU
    // buffers; the final term is the per-level uniform parameter storage.
    allocatedBytes: topologyBytes + stateBytes + 2 * dispatchBytes
      + capturePageStateBytes + levelDeltaBytes + levelCount * PARAMS_BYTES,
    levelDeltaBytes, brickCount, directoryBytes,
    pageDirectoryBytes: pageDirectoryWords * 4, pageRecordWords: PAGE_RECORD_WORDS,
    capturePageStateBytes, capturePageCount,
    rowDispatch: dispatchFor(rowCapacity), slotDispatch: dispatchFor(levelStride), transferDispatch: dispatchFor(transferStride),
    brickDispatch: dispatchFor(brickCount) };
}

export type OctreeSPGridVCyclePipelineName = "beginL1CapturePlan"
  | "planL1CaptureDelta" | "commitChangedL1" | "finalizeL1CapturePublication"
  | "probeCandidateSkip" | "applyCandidateSkip" | "publishCommittedInputs"
  | "clearCandidateLevels" | "buildCandidateLevelSets"
  | "detectCandidateGhosts" | "insertCandidateGhosts"
  | "buildCandidateLevelDeltas" | "countCandidateTransfers" | "scanCandidateTransfers"
  | "writeCandidateTransfers" | "linkCandidateParentChains"
  | "markCandidateBrickOccupancy" | "rankCandidateBricks" | "scatterCandidateRankedSlots"
  | "markCandidatePageOccupancy" | "compactCandidatePages" | "linkCandidatePageNeighbours"
  | "buildCandidateStencils" | "publishCandidateSpectralBounds"
  | "validateCandidateHierarchy" | "commitCandidateLevels"
  | "finalizeLifecycle" | "prepareCorrectionDispatches" | "clearCorrection"
  | "zeroVectors" | "seedRhs"
  | "smoothPageChebyshevForward" | "smoothPageChebyshevReverse"
  | "restrictAndGhostAccumulate" | "exactBottom"
  | "prolongAndGhostPropagate" | "publish";

export const OCTREE_SPGRID_VCYCLE_BINDINGS: Readonly<Record<OctreeSPGridVCyclePipelineName, readonly number[]>> = Object.freeze({
  beginL1CapturePlan: [0, 3, 6, 13, 14, 18],
  planL1CaptureDelta: [0, 1, 3, 11, 13, 14, 18, 20, 21],
  commitChangedL1: [0, 1, 3, 11, 13, 18],
  finalizeL1CapturePublication: [0, 3, 13, 18],
  probeCandidateSkip: [0, 3, 11, 13, 20, 23],
  applyCandidateSkip: [0, 13, 14, 23],
  publishCommittedInputs: [0, 1, 3, 7, 13, 20, 23],
  clearCandidateLevels: [0, 14, 15, 16, 17],
  buildCandidateLevelSets: [0, 1, 3, 14, 15, 16, 17],
  detectCandidateGhosts: [0, 1, 3, 14, 16, 20, 21, 22],
  insertCandidateGhosts: [0, 1, 3, 14, 15, 16, 17, 20, 22],
  buildCandidateLevelDeltas: [0, 1, 3, 4, 5, 6, 14, 15, 16, 17],
  countCandidateTransfers: [0, 1, 3, 4, 5, 6, 14, 15, 16, 17],
  scanCandidateTransfers: [0, 4, 6, 14, 15, 17],
  writeCandidateTransfers: [0, 1, 4, 5, 6, 14, 15, 16, 17],
  linkCandidateParentChains: [0, 14, 15, 17],
  markCandidateBrickOccupancy: [0, 14, 15, 16],
  rankCandidateBricks: [0, 14, 15, 17],
  scatterCandidateRankedSlots: [0, 14, 15, 16],
  markCandidatePageOccupancy: [0, 14, 15],
  compactCandidatePages: [0, 14, 15, 17],
  linkCandidatePageNeighbours: [0, 14, 15, 17],
  buildCandidateStencils: [0, 14, 15, 16, 17],
  publishCandidateSpectralBounds: [0, 14, 15, 16, 17],
  validateCandidateHierarchy: [0, 6, 13, 14, 17],
  commitCandidateLevels: [0, 4, 5, 6, 13, 14, 15, 16, 17],
  finalizeLifecycle: [0, 3, 6, 7, 13],
  prepareCorrectionDispatches: [0, 6, 7, 19],
  clearCorrection: [0, 3, 7, 9],
  zeroVectors: [0, 4, 5, 6, 7], seedRhs: [0, 3, 4, 5, 7, 8, 11],
  smoothPageChebyshevForward: [0, 4, 5, 6, 7],
  smoothPageChebyshevReverse: [0, 4, 5, 6, 7],
  restrictAndGhostAccumulate: [0, 4, 5, 6, 7],
  exactBottom: [0, 4, 5, 6, 7],
  prolongAndGhostPropagate: [0, 4, 5, 6, 7],
  publish: [0, 3, 4, 5, 7, 9, 11],
});

type CachedGroup = { rowCount: GPUBuffer; control: GPUBuffer; geometry: GPUBuffer;
  sourceControl: GPUBuffer; topologyMetrics: GPUBuffer;
  rhs?: GPUBuffer; correction?: GPUBuffer; group: GPUBindGroup };
type AccurateClass = "regularInterior" | "transitionInterior"
  | "physicalBoundary" | "transitionBoundary";
type CachedAccurateApply = {
  readonly input: GPUBuffer;
  readonly output: GPUBuffer;
  readonly solverControl: GPUBuffer;
  readonly worksets: GPUBuffer;
  readonly worksetOffset: number;
  readonly worksetSize?: number;
  readonly worksetLayout: GPUBuffer;
  readonly gateGroup: GPUBindGroup;
  readonly classGroups: Readonly<Record<AccurateClass, GPUBindGroup>>;
  mergedBandGroup?: GPUBindGroup;
};

/**
 * Paper-style native sparse pyramid with an authoritative brick-rank lookup.
 * Setup is GPU-resident and generation-transactional: exact captured-L1
 * deltas retain unchanged levels and rebuild only the dependent coarse suffix.
 * No readback, host decision, or whole-hierarchy fallback is used.
 */
export class WebGPUOctreeSPGridVCycle implements OctreeFirstOrderSPDVCycle {
  readonly operatorOrder = 1 as const;
  readonly isSymmetricPositiveDefinite = true as const;
  readonly convergenceTail = "gpu-zero-indirect" as const;
  readonly plan: OctreeSPGridVCyclePlan;
  readonly allocatedBytes: number;
  readonly encodedPassTransitionCount = 1;
  readonly encodedCorrectionPassTransitionCount = 1;
  readonly encodedSetupDispatchCount: number;
  readonly encodedCorrectionDispatchCount: number;
  /** Accurate second-order executor over four disjoint topology classes. */
  readonly accurateOperator: OctreePipelinedWorksetLinearOperator;
  readonly smootherContract;
  readonly diagnostics: Readonly<{ levelCount: number; coarsestCapacity: number; maximumTransferRecordsPerLevel: number;
    correctionDispatchCount: number; correctionPassTransitions: number; restrictionScatterDispatchCount: number;
    restrictionAtomicAddUpperBound: number; parentGatherDispatchCount: number; parentGatherAtomicAddCount: 0;
    bottomOperation: "exact-single-cell"; coarsestDegreesOfFreedom: 1;
    directoryLookup: "brick-mask-rank"; lookupProbeUpperBound: 1; directoryBytes: number;
    directoryBrickCount: number; directoryBuildDispatchCount: number;
    pageAdjacency: "physical-27"; smootherLookup: "adjacent-page-mask-rank" }>;
  private readonly capturedGeometry: GPUBuffer;
  private readonly candidateCapturedGeometry: GPUBuffer;
  private readonly topology: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly dispatchMeta: GPUBuffer;
  private readonly indirectDispatch: GPUBuffer;
  private readonly capturePageState: GPUBuffer;
  private readonly levelDelta: GPUBuffer;
  private readonly candidateTopology: GPUBuffer;
  private readonly candidateState: GPUBuffer;
  private readonly candidateDispatch: GPUBuffer;
  /** Row-parallel ghost-alias detection record consumed by the ordered insert. */
  private readonly candidateGhosts: GPUBuffer;
  /** Exact fingerprint of the inputs the last committed hierarchy consumed. */
  private readonly committedInputs: GPUBuffer;
  private readonly levelDispatch: readonly [number, number, number];
  private readonly clearDispatch: readonly [number, number, number];
  private readonly pageDispatch: readonly [number, number, number];
  private readonly accurateWorksetLayout: GPUBuffer;
  private readonly accurateGatePipeline: GPUComputePipeline;
  private readonly accurateClassDispatch: GPUBuffer;
  private readonly accurateClassPipelines: Readonly<Record<AccurateClass, GPUComputePipeline>>;
  private readonly accurateMergedBandPipeline: GPUComputePipeline;
  private readonly accurateBindings: CachedAccurateApply[] = [];
  private readonly params: readonly GPUBuffer[];
  private readonly candidateParams: readonly GPUBuffer[];
  private readonly pipelines: Readonly<Record<OctreeSPGridVCyclePipelineName, GPUComputePipeline>>;
  private readonly groups = new Map<string, CachedGroup>();
  private readonly pre: number;
  private readonly post: number;
  private lastSetupInput?: { readonly solverControl: GPUBuffer; readonly rowCount: GPUBuffer };
  private preparedCaptureSource?: OctreeSPGridSetupSource;
  private candidateSetupInput?: OctreeSPGridSetupSource;
  private destroyed = false;

  /** GPU-authored controls suitable for the runtime's diagnostics readback. */
  get workAccountingBuffers(): Readonly<{ dispatch: GPUBuffer; capture: GPUBuffer }> {
    return Object.freeze({ dispatch: this.dispatchMeta, capture: this.capturePageState });
  }

  /** Candidate hierarchy/capture report consumed by the coupled epoch reduction. */
  get candidateControl(): GPUBuffer { return this.capturePageState; }

  /** Accepted physical pages and ghost ownership shared with the Section 4.3 shell. */
  get section63Topology(): Readonly<{
    topology: GPUBuffer; state: GPUBuffer; geometry: GPUBuffer; layout: GPUBuffer;
    dispatch: GPUBuffer;
  }> {
    return Object.freeze({ topology: this.topology, state: this.state,
      geometry: this.capturedGeometry, layout: this.accurateWorksetLayout,
      dispatch: this.dispatchMeta });
  }

  get workAccountingPlan(): Readonly<{ levelCount: number; levelCapacities: readonly number[];
    encodedCorrectionDispatches: number; persistentEnabled: boolean;
    persistentMaximumIterations: number }> {
    return Object.freeze({ levelCount: this.plan.levelCount,
      levelCapacities: this.plan.levelCapacities,
      encodedCorrectionDispatches: this.encodedCorrectionDispatchCount,
      persistentEnabled: false,
      persistentMaximumIterations: 12 });
  }

  constructor(private readonly device: GPUDevice, private readonly source: OctreeSPGridVCycleSource,
    options: OctreeSPGridVCycleOptions) {
    this.plan = planOctreeSPGridVCycle(options);
    if (!(options.finestCellWidth > 0) || !Number.isFinite(options.finestCellWidth)) throw new RangeError("SPGrid finest cell width must be positive");
    if (source.rowGeometry.size < this.plan.rowCapacity * 32
      || source.control.size < 24
      || source.coefficients.size < 2 * this.plan.rowCapacity * 19 * 4
      || source.coefficientBankStrideWords < this.plan.rowCapacity * 19
      || source.rowCapacity !== this.plan.rowCapacity) {
      throw new RangeError("SPGrid fixed Section 6.3 source capacity is too small");
    }
    const captureUsage = GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE;
    if ((source.rowGeometry.usage & captureUsage) !== captureUsage
      || (source.topologyMetrics.usage & GPUBufferUsage.STORAGE) === 0
      || (source.catalogCoefficients.usage & GPUBufferUsage.STORAGE) === 0) {
      throw new RangeError("SPGrid Section 6.3 buffers require publication storage usage");
    }
    const delta = source.rowDelta;
    if (!Number.isSafeInteger(delta.rowCapacity) || delta.rowCapacity !== this.plan.rowCapacity
      || ![delta.controlOffsetWords, delta.newToOldOffsetWords, delta.dirtyRowsOffsetWords]
        .every((offset) => Number.isSafeInteger(offset) && offset >= 0)
      || delta.newToOldOffsetWords < delta.controlOffsetWords + 16
      || delta.dirtyRowsOffsetWords < delta.newToOldOffsetWords + delta.rowCapacity
      || delta.rows.size < (delta.dirtyRowsOffsetWords + delta.rowCapacity) * 4
      || (delta.rows.usage & GPUBufferUsage.STORAGE) === 0) {
      throw new RangeError("SPGrid L1 requires the exact compact row-delta producer authority");
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
    if (!OCTREE_FIRST_ORDER_CHEBYSHEV_DEGREES.includes(this.pre as 2 | 4)) {
      throw new RangeError("SPGrid smoothing degree must be exactly 2 or 4");
    }
    this.smootherContract = Object.freeze({
      kind: "chebyshev" as const,
      degree: this.pre as 2 | 4,
      spectralBounds: "transactional-scaled-gershgorin" as const,
      lowerFraction: OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION,
    });
    // COPY_SRC supports bounded lifecycle verification and preserves the
    // existing cold capture path; it does not add storage or bind stages.
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.capturedGeometry = device.createBuffer({ label: "SPGrid captured fixed row geometry", size: this.plan.rowCapacity * 16, usage: storage });
    this.candidateCapturedGeometry = device.createBuffer({ label: "SPGrid candidate fixed row geometry", size: this.plan.rowCapacity * 16, usage: storage });
    this.topology = device.createBuffer({ label: "SPGrid native sparse topology/worklists/transfers", size: this.plan.topologyBytes, usage: storage });
    this.state = device.createBuffer({ label: "SPGrid six-face stencils and vectors", size: this.plan.stateBytes, usage: storage });
    this.dispatchMeta = device.createBuffer({ label: "SPGrid worklist counts and published dispatches", size: this.plan.dispatchBytes,
      usage: storage | GPUBufferUsage.COPY_SRC });
    this.indirectDispatch = device.createBuffer({ label: "SPGrid live indirect dispatches", size: this.plan.dispatchBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.capturePageState = device.createBuffer({ label: "SPGrid exact L1 page generations",
      size: this.plan.capturePageStateBytes, usage: storage });
    this.levelDelta = device.createBuffer({ label: "SPGrid exact per-level setup delta",
      size: this.plan.levelDeltaBytes, usage: storage });
    this.candidateTopology = device.createBuffer({ label: "SPGrid immutable candidate topology",
      size: this.plan.topologyBytes, usage: storage });
    this.candidateState = device.createBuffer({ label: "SPGrid immutable candidate stencil state",
      size: this.plan.stateBytes, usage: storage });
    this.candidateDispatch = device.createBuffer({ label: "SPGrid immutable candidate dispatch records",
      size: this.plan.dispatchBytes, usage: storage });
    this.candidateGhosts = device.createBuffer({ label: "SPGrid candidate ghost-alias detection records",
      size: this.plan.rowCapacity * GHOST_SCRATCH_WORDS_PER_ROW * 4, usage: storage });
    this.committedInputs = device.createBuffer({ label: "SPGrid committed hierarchy input fingerprint",
      size: (COMMITTED_INPUT_HEADER_WORDS
        + this.plan.rowCapacity * COMMITTED_INPUT_WORDS_PER_ROW) * 4, usage: storage });
    this.levelDispatch = [this.plan.levelCount, 1, 1];
    this.clearDispatch = dispatchFor(Math.max(this.plan.levelStride, this.plan.rowCapacity,
      DISPATCH_RECORD_WORDS_PER_LEVEL));
    this.pageDispatch = dispatchFor(Math.max(1, this.plan.pageDirectoryBytes / 4));
    this.accurateWorksetLayout = device.createBuffer({
      label: "SPGrid Section 6.3 page operator layout", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const accurateLayout = new ArrayBuffer(64), accurateWords = new Uint32Array(accurateLayout);
    accurateWords.set([source.worksetStrideWords ?? this.plan.rowCapacity + 7,
      source.worksetBankStrideWords ?? 4 * (this.plan.rowCapacity + 7), 0, 0,
      options.dimensions[0], options.dimensions[1], options.dimensions[2], this.plan.rowCapacity,
      this.plan.levelCount, this.plan.levelStride, this.plan.totalLevelSlots,
      source.coefficientBankStrideWords]);
    device.queue.writeBuffer(this.accurateWorksetLayout, 0, accurateLayout);
    // Memoized level tables. Entries below levelCount are the exact allocation
    // authority (plan.levelCapacities/levelOffsets); the padding entries repeat
    // the same closed form so an out-of-range probe cannot read stale storage.
    const maximumSparseSlots = nextPowerOfTwo(this.plan.rowCapacity * 16);
    const levelCapacityAt = (level: number) => {
      if (level < this.plan.levelCount) return this.plan.levelCapacities[level];
      const scale = 2 ** level;
      const domainCells = options.dimensions
        .map((value) => Math.ceil(value / scale))
        .reduce((product, value) => product * value, 1);
      return nextPowerOfTwo(Math.min(maximumSparseSlots, 2 * domainCells));
    };
    const levelCaps: number[] = [], levelBases: number[] = [], brickOffsets: number[] = [];
    const pageOffsets: number[] = [], transferOffsets: number[] = [];
    let slotBase = 0, brickBase = 0, pageBase = 0, transferBase = 0;
    for (let level = 0; level < PARAMS_LEVEL_TABLE_SLOTS; level += 1) {
      const capacity = levelCapacityAt(level);
      const levelDimensions = options.dimensions.map((value) => Math.ceil(value / 2 ** level));
      levelCaps.push(capacity);
      levelBases.push(slotBase); slotBase += capacity;
      brickOffsets.push(brickBase);
      brickBase += levelDimensions.map((value) => Math.ceil(value / 4))
        .reduce((product, value) => product * value, 1);
      pageOffsets.push(pageBase);
      pageBase += Math.ceil(levelDimensions[0] / 8) * Math.ceil(levelDimensions[1] / 8)
        * Math.ceil(levelDimensions[2] / 4);
      transferOffsets.push(transferBase);
      transferBase += Math.min(this.plan.transferStride, capacity * 8) * 4 + 4 * capacity;
    }
    if (levelBases[this.plan.levelCount] !== this.plan.totalLevelSlots
      || brickOffsets[this.plan.levelCount] !== this.plan.brickCount
      || pageOffsets[this.plan.levelCount] !== this.plan.pageDirectoryBytes / 4) {
      throw new RangeError("SPGrid level tables disagree with the allocated plan");
    }
    const makeParams = (level: number, sourceMode: OctreeSPGridSourceMode) => {
      const buffer = device.createBuffer({
        label: `SPGrid level ${level} ${sourceMode} parameters`, size: PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(PARAMS_BYTES / 4), floats = new Float32Array(words.buffer);
      words.set([options.dimensions[0], options.dimensions[1], options.dimensions[2], level,
        this.plan.rowCapacity, this.plan.levelCount, this.plan.levelStride, this.plan.transferStride,
        this.plan.rowDispatch[0], this.plan.slotDispatch[0], this.plan.transferDispatch[0], this.pre]);
      words[12] = this.post;
      words[13] = sourceMode === "candidate" ? 1 : 0;
      floats[15] = options.finestCellWidth;
      words.set([delta.rowCapacity, delta.controlOffsetWords, delta.newToOldOffsetWords,
        delta.dirtyRowsOffsetWords], 16);
      words.set([this.plan.totalLevelSlots, this.plan.brickCount,
        this.plan.pageDirectoryBytes / 4, 0], 20);
      words.set(levelCaps, 24);
      words.set(levelBases, 24 + PARAMS_LEVEL_TABLE_SLOTS);
      words.set(brickOffsets, 24 + 2 * PARAMS_LEVEL_TABLE_SLOTS);
      words.set(pageOffsets, 24 + 3 * PARAMS_LEVEL_TABLE_SLOTS);
      words.set(transferOffsets, 24 + 4 * PARAMS_LEVEL_TABLE_SLOTS);
      device.queue.writeBuffer(buffer, 0, words); return buffer;
    };
    this.params = Object.freeze(Array.from(
      { length: this.plan.levelCount }, (_, level) => makeParams(level, "accepted")));
    this.candidateParams = Object.freeze(Array.from(
      { length: this.plan.levelCount }, (_, level) => makeParams(level, "candidate")));
    const shaderModule = device.createShaderModule({ label: "Paper native sparse SPGrid V-cycle", code: octreeSPGridVCycleShader });
    const make = (entryPoint: OctreeSPGridVCyclePipelineName) => device.createComputePipeline({ label: `SPGrid V-cycle · ${entryPoint}`,
      layout: "auto", compute: { module: shaderModule, entryPoint } });
    this.pipelines = Object.freeze(Object.fromEntries((Object.keys(OCTREE_SPGRID_VCYCLE_BINDINGS) as OctreeSPGridVCyclePipelineName[]).map((name) => [name, make(name)])) as Record<OctreeSPGridVCyclePipelineName, GPUComputePipeline>);
    const accurateModule = device.createShaderModule({
      label: "SPGrid accurate A2 class-specialized row apply", code: octreeSPGridAccurateOperatorShader,
    });
    const accurateGateModule = device.createShaderModule({
      label: "SPGrid accurate A2 convergence-tail publisher",
      code: octreeSPGridAccurateDispatchGateShader,
    });
    this.accurateGatePipeline = device.createComputePipeline({
      label: "SPGrid accurate A2 · convergence-tail publisher", layout: "auto",
      compute: { module: accurateGateModule, entryPoint: "prepareAccurateDispatches" },
    });
    this.accurateClassDispatch = device.createBuffer({
      label: "SPGrid accurate A2 convergence-gated class records",
      size: 4 * 12,
      usage: storage | GPUBufferUsage.INDIRECT,
    });
    const accurateEntries: Readonly<Record<AccurateClass, string>> = Object.freeze({
      regularInterior: "applyRegularInterior",
      transitionInterior: "applyTransitionInterior",
      physicalBoundary: "applyPhysicalBoundary",
      transitionBoundary: "applyTransitionBoundary",
    });
    this.accurateClassPipelines = Object.freeze(Object.fromEntries(
      (Object.keys(accurateEntries) as AccurateClass[]).map((rowClass) => [rowClass,
        device.createComputePipeline({ label: `SPGrid accurate A2 · ${rowClass}`, layout: "auto",
          compute: { module: accurateModule, entryPoint: accurateEntries[rowClass] } })]),
    ) as Record<AccurateClass, GPUComputePipeline>);
    this.accurateMergedBandPipeline = device.createComputePipeline({
      label: "SPGrid Section 6.3 · destination-owned merged band", layout: "auto",
      compute: { module: accurateModule, entryPoint: "applyMergedBand" },
    });
    this.accurateOperator = Object.freeze({
      convergenceTail: "gpu-zero-indirect" as const,
      encodedDispatchCount: 5,
      encodedMergedBandDispatchCount: 1 as const,
      encode: (broker: PassBroker, input: GPUBuffer, output: GPUBuffer, solverControl: GPUBuffer) => {
        if (!this.source.classDispatch) {
          throw new Error("SPGrid accurate A2 requires accepted class dispatch publication");
        }
        const canonical = this.source.worksets.regularInterior;
        const binding = "buffer" in canonical ? canonical : { buffer: canonical };
        this.encodeAccurateWorksets(broker, input, output, solverControl,
          { buffer: binding.buffer }, this.source.classDispatch, this.accurateWorksetLayout,
          this.source.classDispatchOffsetBytes ?? 0);
      },
      encodeWorksets: (broker: PassBroker, input: GPUBuffer, output: GPUBuffer,
        solverControl: GPUBuffer, worksets: GPUBuffer, classDispatch: GPUBuffer,
        worksetLayout: GPUBuffer, classDispatchOffsetBytes = 0) => {
        this.encodeAccurateWorksets(broker, input, output, solverControl,
          { buffer: worksets }, classDispatch, worksetLayout, classDispatchOffsetBytes);
      },
      encodeMergedBandWorkset: (broker: PassBroker, input: GPUBuffer, output: GPUBuffer,
        solverControl: GPUBuffer, worksets: GPUBuffer, mergedDispatch: GPUBuffer,
        worksetLayout: GPUBuffer, mergedDispatchOffsetBytes: number) => {
        this.encodeAccurateMergedBandWorkset(broker, input, output, solverControl,
          { buffer: worksets }, mergedDispatch, worksetLayout, mergedDispatchOffsetBytes);
      },
    });
    const l = this.plan.levelCount;
    // Two exact L1-capture dispatches, the two-dispatch unchanged-input skip
    // probe, the changed-page commit, thirteen data-parallel candidate phases
    // separated by dispatch order rather than by statement order in one thread,
    // the candidate validator, and five transactional publication dispatches.
    // Keep this exact for command and active/scheduled accounting.
    this.encodedSetupDispatchCount = 2 + 2 + 1 + 17 + 1 + 5;
    // One compact 8x8x4 page dispatch stages the one-cell halo and executes
    // the complete even-degree Chebyshev polynomial in workgroup memory.
    // Post-smoothing consumes the weights in reverse order, retaining the
    // fixed symmetric V-cycle schedule without pointwise dispatches.
    this.encodedCorrectionDispatchCount = l + 5 + (l - 1) * 4;
    this.diagnostics = Object.freeze({ levelCount: l,
      coarsestCapacity: this.plan.levelCapacities[this.plan.levelCount - 1],
      maximumTransferRecordsPerLevel: Math.max(...this.plan.transferCapacities),
      correctionDispatchCount: this.encodedCorrectionDispatchCount,
      correctionPassTransitions: 1, restrictionScatterDispatchCount: 0,
      restrictionAtomicAddUpperBound: 0,
      parentGatherDispatchCount: l - 1, parentGatherAtomicAddCount: 0 as const,
      bottomOperation: "exact-single-cell" as const, coarsestDegreesOfFreedom: 1 as const,
      directoryLookup: "brick-mask-rank" as const, lookupProbeUpperBound: 1 as const,
      directoryBytes: this.plan.directoryBytes, directoryBrickCount: this.plan.brickCount,
      directoryBuildDispatchCount: 1,
      pageAdjacency: "physical-27" as const,
      smootherLookup: "adjacent-page-mask-rank" as const });
    this.allocatedBytes = this.plan.allocatedBytes + this.capturedGeometry.size + this.candidateCapturedGeometry.size
      + this.candidateTopology.size + this.candidateState.size + this.candidateDispatch.size
      + this.candidateGhosts.size + this.committedInputs.size
      + this.accurateWorksetLayout.size + this.accurateClassDispatch.size
      + this.candidateParams.reduce((bytes, buffer) => bytes + buffer.size, 0);
  }

  encodeCapture(broker: PassBroker): void {
    this.assertLive();
    // The first capture is deferred until encodeSetup supplies the authoritative
    // row-count buffer. There is deliberately no capacity-copy bootstrap.
    if (!this.lastSetupInput) return;
    this.encodeCaptureDelta(broker, { ...this.lastSetupInput, sourceMode: "accepted" });
  }

  private encodeCaptureDelta(broker: PassBroker,
    input: OctreeSPGridSetupSource): void {
    const pass = broker.compute({ label: "SPGrid V-cycle · select setup delta and capture changed L1" });
    this.run(pass, "beginL1CapturePlan", 0, input, [1, 1, 1]);
    this.run(pass, "planL1CaptureDelta", 0, input, [1, 1, 1]);
    this.preparedCaptureSource = input;
  }

  encodeCandidateSetup(broker: PassBroker, input: { solverControl: GPUBuffer; rowCount: GPUBuffer;
    sourceControl: GPUBuffer; topologyMetrics: GPUBuffer }): void {
    this.encodeSetupCandidate(broker, { ...input, sourceMode: "candidate" });
  }

  private encodeSetupCandidate(broker: PassBroker, input: OctreeSPGridSetupSource): void {
    this.assertLive();
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic before SPGrid candidate hierarchy rebuild");
    }
    const prepared = this.preparedCaptureSource;
    if (!prepared || prepared.rowCount !== input.rowCount
      || prepared.sourceControl !== input.sourceControl
      || prepared.topologyMetrics !== input.topologyMetrics
      || prepared.sourceMode !== input.sourceMode) this.encodeCaptureDelta(broker, input);
    const pass = broker.compute({ label: "SPGrid V-cycle · build inactive exact level deltas" });
    // Sound unchanged-input skip: the probe stamps a mismatch whenever the row
    // count or any row's fixed geometry/Section 6.3 case differs from the exact
    // fingerprint of the last successfully committed build, and only an
    // unstamped generation is allowed to retire the per-level dirty flags.
    this.run(pass, "probeCandidateSkip", 0, input, this.plan.rowDispatch);
    this.run(pass, "applyCandidateSkip", 0, input, [1, 1, 1]);
    const geometry = this.candidateCapturedGeometry;
    this.run(pass, "commitChangedL1", 0, input, [1, 1, 1], geometry);
    // Ordered data-parallel candidate construction. Each phase is its own
    // dispatch so per-level and inter-phase ordering is carried by dispatch
    // boundaries instead of statement order inside a single invocation.
    this.run(pass, "clearCandidateLevels", 0, input, this.clearDispatch, geometry);
    this.run(pass, "buildCandidateLevelSets", 0, input, this.levelDispatch, geometry);
    this.run(pass, "detectCandidateGhosts", 0, input, this.plan.rowDispatch, geometry);
    this.run(pass, "insertCandidateGhosts", 0, input, this.levelDispatch, geometry);
    this.run(pass, "buildCandidateLevelDeltas", 0, input, [1, 1, 1], geometry);
    this.run(pass, "countCandidateTransfers", 0, input, this.plan.slotDispatch, geometry);
    this.run(pass, "scanCandidateTransfers", 0, input, this.levelDispatch, geometry);
    this.run(pass, "writeCandidateTransfers", 0, input, this.plan.slotDispatch, geometry);
    this.run(pass, "linkCandidateParentChains", 0, input, this.levelDispatch, geometry);
    this.run(pass, "markCandidateBrickOccupancy", 0, input, this.plan.brickDispatch, geometry);
    this.run(pass, "rankCandidateBricks", 0, input, this.levelDispatch, geometry);
    this.run(pass, "scatterCandidateRankedSlots", 0, input, this.plan.brickDispatch, geometry);
    this.run(pass, "markCandidatePageOccupancy", 0, input, this.pageDispatch, geometry);
    this.run(pass, "compactCandidatePages", 0, input, this.levelDispatch, geometry);
    this.run(pass, "linkCandidatePageNeighbours", 0, input, this.pageDispatch, geometry);
    this.run(pass, "buildCandidateStencils", 0, input, this.plan.slotDispatch, geometry);
    this.run(pass, "publishCandidateSpectralBounds", 0, input, this.levelDispatch, geometry);
    this.run(pass, "validateCandidateHierarchy", 0, input, [1, 1, 1]);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after SPGrid candidate hierarchy rebuild");
    }
    this.candidateSetupInput = input;
    this.preparedCaptureSource = undefined;
  }

  encodeReadySetupCommit(broker: PassBroker, input: { solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive();
    if (!this.candidateSetupInput || this.candidateSetupInput.rowCount !== input.rowCount
      || this.candidateSetupInput.solverControl !== input.solverControl) {
      throw new Error("SPGrid ready commit requires the matching inactive hierarchy candidate");
    }
    const candidate = this.candidateSetupInput;
    const pass = broker.compute({ label: "SPGrid V-cycle · publish validated exact level deltas" });
    this.run(pass, "commitChangedL1", 0, input, [1, 1, 1], this.capturedGeometry);
    this.run(pass, "finalizeL1CapturePublication", 0, input, [1, 1, 1]);
    this.run(pass, "commitCandidateLevels", 0, input,
      dispatchFor(Math.max(this.plan.levelStride, this.plan.brickCount,
        ...this.plan.transferCapacities)));
    this.run(pass, "finalizeLifecycle", 0, input, [1, 1, 1]);
    // Fingerprint the exact inputs this hierarchy consumed, and only after the
    // lifecycle gate accepted it. A rejected epoch leaves the previous
    // fingerprint (or none), so the next probe can never skip a stale build.
    this.run(pass, "publishCommittedInputs", 0, candidate,
      this.plan.rowDispatch, this.candidateCapturedGeometry);
    // dispatchMeta remains STORAGE-only inside compute passes. Copying its
    // finalized records after the setup boundary gives correction a distinct
    // INDIRECT-only source and avoids a whole-pass storage/indirect conflict.
    broker.updateIndirectBuffer(this.dispatchMeta, 0, this.indirectDispatch, 0, this.plan.dispatchBytes);
    this.lastSetupInput = input;
    this.candidateSetupInput = undefined;
  }

  encodeSetup(broker: PassBroker, input: { solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    if (!this.candidateSetupInput && this.lastSetupInput?.rowCount === input.rowCount
      && this.lastSetupInput.solverControl === input.solverControl) return;
    if (!this.candidateSetupInput) {
      this.encodeSetupCandidate(broker, { ...input, sourceMode: "accepted" });
    }
    this.encodeReadySetupCommit(broker, input);
  }

  encodeCorrection(broker: PassBroker, input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive();
    let pass = broker.compute({ label: "SPGrid V-cycle · publish convergence-gated level records" });
    this.run(pass, "prepareCorrectionDispatches", 0, input, [1, 1, 1]);
    broker.fence("SPGrid V-cycle convergence-gated indirect publication");
    pass = broker.compute({ label: "SPGrid V-cycle · one-pass symmetric correction" });
    this.runIndirect(pass, "clearCorrection", 0, input, false);
    for (let level = 0; level < this.plan.levelCount; level += 1) this.runIndirect(pass, "zeroVectors", level, input, false);
    this.runIndirect(pass, "seedRhs", 0, input, false);
    for (let level = 0; level < this.plan.levelCount - 1; level += 1) {
      this.smooth(pass, level, false, input);
      this.runIndirect(pass, "restrictAndGhostAccumulate", level, input, true);
    }
    this.runIndirect(pass, "exactBottom", this.plan.levelCount - 1, input, false);
    for (let level = this.plan.levelCount - 2; level >= 0; level -= 1) {
      this.runIndirect(pass, "prolongAndGhostPropagate", level, input, false);
      this.smooth(pass, level, true, input);
    }
    this.runIndirect(pass, "publish", 0, input, false);
  }

  private smooth(pass: GPUComputePassEncoder, level: number, reverse: boolean,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.bind(pass, reverse
      ? "smoothPageChebyshevReverse"
      : "smoothPageChebyshevForward", level, input);
    pass.dispatchWorkgroupsIndirect(
      this.indirectDispatch,
      level * DISPATCH_RECORD_BYTES_PER_LEVEL + 9 * 4,
    );
  }

  private runIndirect(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }, transfer: boolean): void {
    this.bind(pass, name, level, input);
    pass.dispatchWorkgroupsIndirect(this.indirectDispatch,
      level * DISPATCH_RECORD_BYTES_PER_LEVEL + (transfer ? 20 : 8));
  }
  private run(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer;
      sourceControl?: GPUBuffer; topologyMetrics?: GPUBuffer; sourceMode?: OctreeSPGridSourceMode },
    dispatch: readonly [number, number, number], geometry = this.capturedGeometry): void {
    this.bind(pass, name, level, input, geometry); pass.dispatchWorkgroups(...dispatch);
  }
  private bind(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer;
      sourceControl?: GPUBuffer; topologyMetrics?: GPUBuffer; sourceMode?: OctreeSPGridSourceMode },
    geometry = this.capturedGeometry): void {
    const pipeline = this.pipelines[name];
    const sourceMode = input.sourceMode ?? "accepted";
    const key = `${name}:${level}:${geometry === this.candidateCapturedGeometry ? "candidate" : "accepted"}:${sourceMode}`;
    const cached = this.groups.get(key);
    let group = cached?.group;
    if (!cached || cached.rowCount !== input.rowCount || cached.control !== input.solverControl
      || cached.rhs !== input.rhs || cached.correction !== input.correction || cached.geometry !== geometry
      || cached.sourceControl !== (input.sourceControl ?? this.source.control)
      || cached.topologyMetrics !== (input.topologyMetrics ?? this.source.topologyMetrics)) {
      const buffers = new Map<number, GPUBuffer | undefined>([
        [0, sourceMode === "candidate" ? this.candidateParams[level] : this.params[level]],
        [1, geometry], [3, input.sourceControl ?? this.source.control],
        [4, this.topology], [5, this.state], [6, this.dispatchMeta], [7, input.solverControl],
        [8, input.rhs], [9, input.correction],
        [11, this.source.rowGeometry],
        [13, this.capturePageState], [14, this.levelDelta], [15, this.candidateTopology],
        [16, this.candidateState], [17, this.candidateDispatch], [18, this.source.rowDelta.rows],
        [19, this.indirectDispatch],
        [20, input.topologyMetrics ?? this.source.topologyMetrics], [21, this.source.catalogCoefficients],
        [22, this.candidateGhosts], [23, this.committedInputs],
      ]);
      group = this.device.createBindGroup({ label: `SPGrid V-cycle · ${name} · level ${level}`,
        layout: pipeline.getBindGroupLayout(0), entries: OCTREE_SPGRID_VCYCLE_BINDINGS[name].map((binding) => ({
        binding, resource: { buffer: buffers.get(binding)! },
      })) });
      this.groups.set(key, { rowCount: input.rowCount, control: input.solverControl, geometry,
        sourceControl: input.sourceControl ?? this.source.control,
        topologyMetrics: input.topologyMetrics ?? this.source.topologyMetrics,
        rhs: input.rhs, correction: input.correction, group });
    }
    pass.setPipeline(pipeline); pass.setBindGroup(0, group!);
  }

  private encodeAccurateWorksets(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBufferBinding,
    _classDispatch: GPUBuffer,
    worksetLayout: GPUBuffer,
    _classDispatchOffsetBytes: number,
  ): void {
    void _classDispatchOffsetBytes;
    this.assertLive();
    if (input.size < this.plan.rowCapacity * 4 || output.size < this.plan.rowCapacity * 4) {
      throw new RangeError("SPGrid accurate A2 vectors are smaller than row capacity");
    }
    const cached = this.accurateBinding(input, output, solverControl, worksets, worksetLayout);
    let pass = broker.compute({ label: "SPGrid accurate A2 · publish convergence-gated records" });
    pass.setPipeline(this.accurateGatePipeline);
    pass.setBindGroup(0, cached.gateGroup);
    pass.dispatchWorkgroups(1, 1, 1);
    broker.fence("SPGrid accurate A2 convergence-gated indirect publication");
    pass = broker.compute({ label: "SPGrid accurate A2 · four disjoint row classes" });
    const classes = Object.keys(this.accurateClassPipelines) as AccurateClass[];
    for (const [index, rowClass] of classes.entries()) {
      pass.setPipeline(this.accurateClassPipelines[rowClass]);
      pass.setBindGroup(0, cached.classGroups[rowClass]);
      pass.dispatchWorkgroupsIndirect(this.accurateClassDispatch, index * 12);
    }
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after accurate A2 class apply");
    }
  }

  private encodeAccurateMergedBandWorkset(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBufferBinding,
    mergedDispatch: GPUBuffer,
    worksetLayout: GPUBuffer,
    mergedDispatchOffsetBytes: number,
  ): void {
    this.assertLive();
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic before merged-band A2 apply");
    }
    if (input.size < this.plan.rowCapacity * 4 || output.size < this.plan.rowCapacity * 4) {
      throw new RangeError("SPGrid accurate A2 vectors are smaller than row capacity");
    }
    if (!Number.isSafeInteger(mergedDispatchOffsetBytes) || mergedDispatchOffsetBytes < 0
      || (mergedDispatchOffsetBytes & 3) !== 0
      || mergedDispatchOffsetBytes + 12 > mergedDispatch.size) {
      throw new RangeError("SPGrid accurate A2 merged-band dispatch offset is invalid");
    }
    const cached = this.accurateBinding(input, output, solverControl, worksets, worksetLayout);
    if (!cached.mergedBandGroup) {
      const shared = new Map<number, GPUBufferBinding>([
        [0, { buffer: input }], [1, { buffer: output }], [2, { buffer: solverControl }],
        [3, { buffer: this.source.control }],
        [4, { buffer: cached.worksets, offset: cached.worksetOffset,
          ...(cached.worksetSize === undefined ? {} : { size: cached.worksetSize }) }],
        [5, { buffer: worksetLayout }], [6, { buffer: this.topology }],
        [7, { buffer: this.state }], [8, { buffer: this.capturedGeometry }],
        [9, { buffer: this.source.topologyMetrics }], [10, { buffer: this.source.coefficients }],
        [11, { buffer: this.accurateWorksetLayout }],
      ]);
      cached.mergedBandGroup = this.device.createBindGroup({
        label: "SPGrid Section 6.3 · destination-owned merged band bindings",
        layout: this.accurateMergedBandPipeline.getBindGroupLayout(0),
        entries: [...shared].map(([binding, resource]) => ({ binding, resource })),
      });
    }
    const pass = broker.compute({ label: "SPGrid Section 6.3 · destination-owned merged band" });
    pass.setPipeline(this.accurateMergedBandPipeline); pass.setBindGroup(0, cached.mergedBandGroup);
    pass.dispatchWorkgroupsIndirect(mergedDispatch, mergedDispatchOffsetBytes);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after merged-band A2 apply");
    }
  }

  private accurateBinding(
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBufferBinding,
    worksetLayout: GPUBuffer,
  ): CachedAccurateApply {
    const worksetOffset = Number(worksets.offset ?? 0);
    const worksetSize = worksets.size === undefined ? undefined : Number(worksets.size);
    let cached = this.accurateBindings.find((candidate) =>
      candidate.input === input && candidate.output === output
      && candidate.solverControl === solverControl
      && candidate.worksets === worksets.buffer
      && candidate.worksetOffset === worksetOffset
      && candidate.worksetSize === worksetSize
      && candidate.worksetLayout === worksetLayout);
    if (!cached) {
      const shared = new Map<number, GPUBufferBinding>([
        [0, { buffer: input }], [1, { buffer: output }], [2, { buffer: solverControl }],
        [3, { buffer: this.source.control }],
        [4, { buffer: worksets.buffer, offset: worksetOffset,
          ...(worksetSize === undefined ? {} : { size: worksetSize }) }],
        [5, { buffer: worksetLayout }], [6, { buffer: this.topology }],
        [7, { buffer: this.state }], [8, { buffer: this.capturedGeometry }],
        [9, { buffer: this.source.topologyMetrics }],
        [10, { buffer: this.source.coefficients }],
        [11, { buffer: this.accurateWorksetLayout }],
      ]);
      const makeGroup = (pipeline: GPUComputePipeline, bindings: readonly number[], label: string) => this.device.createBindGroup({
        label, layout: pipeline.getBindGroupLayout(0), entries: bindings.map((binding) => ({
          binding, resource: shared.get(binding)!,
        })),
      });
      const classGroups = Object.freeze(Object.fromEntries(
        (Object.keys(this.accurateClassPipelines) as AccurateClass[]).map((rowClass) => [
          rowClass, makeGroup(this.accurateClassPipelines[rowClass],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            `SPGrid Section 6.3 · ${rowClass} bindings`),
        ]),
      ) as Record<AccurateClass, GPUBindGroup>);
      const gateGroup = this.device.createBindGroup({
        label: "SPGrid accurate A2 · convergence-tail bindings",
        layout: this.accurateGatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: solverControl } },
          { binding: 1, resource: { buffer: worksets.buffer, offset: worksetOffset,
            ...(worksetSize === undefined ? {} : { size: worksetSize }) } },
          { binding: 2, resource: { buffer: worksetLayout } },
          { binding: 3, resource: { buffer: this.source.control } },
          { binding: 4, resource: { buffer: this.accurateClassDispatch } },
        ],
      });
      cached = { input, output, solverControl, worksets: worksets.buffer,
        worksetOffset, worksetSize, worksetLayout, gateGroup, classGroups };
      this.accurateBindings.push(cached);
    }
    return cached;
  }

  private assertLive(): void { if (this.destroyed) throw new Error("SPGrid V-cycle is destroyed"); }
  destroy(): void {
    if (this.destroyed) return; this.destroyed = true; this.groups.clear(); this.accurateBindings.length = 0;
    this.capturedGeometry.destroy(); this.candidateCapturedGeometry.destroy();
    this.topology.destroy(); this.state.destroy(); this.dispatchMeta.destroy();
    this.indirectDispatch.destroy();
    this.capturePageState.destroy(); this.levelDelta.destroy();
    this.candidateTopology.destroy(); this.candidateState.destroy(); this.candidateDispatch.destroy();
    this.candidateGhosts.destroy(); this.committedInputs.destroy();
    this.accurateWorksetLayout.destroy(); this.accurateClassDispatch.destroy();
    for (const buffer of [...this.params, ...this.candidateParams]) buffer.destroy();
  }
}

/** Accurate second-order matrix-free apply over four accepted row classes. */
export const octreeSPGridAccurateDispatchGateShader = /* wgsl */ `
@group(0) @binding(0) var<storage,read> solverControl:array<u32>;
@group(0) @binding(1) var<storage,read> worksets:array<u32>;
@group(0) @binding(2) var<uniform> worksetLayout:vec4u;
@group(0) @binding(3) var<storage,read> accepted:array<u32>;
@group(0) @binding(4) var<storage,read_write> classDispatch:array<u32>;
fn activeSolve()->bool{return arrayLength(&solverControl)>=2u
 &&solverControl[0]==0u&&solverControl[1]==0u;}
@compute @workgroup_size(1)
fn prepareAccurateDispatches(){
 let solveLive=activeSolve();
 let bank=select(0u,accepted[4]&1u,arrayLength(&accepted)>4u);
 for(var cls=0u;cls<4u;cls+=1u){
  let source=bank*worksetLayout.y+cls*worksetLayout.x+4u;let destination=cls*3u;
  let valid=source+2u<arrayLength(&worksets);
  classDispatch[destination]=select(0u,worksets[source],solveLive&&valid);
  classDispatch[destination+1u]=select(1u,worksets[source+1u],valid);
  classDispatch[destination+2u]=select(1u,worksets[source+2u],valid);
 }
}
`;

export const octreeSPGridAccurateOperatorShader = /* wgsl */ `
struct Layout{workset:vec4u,dimsCapacity:vec4u,hierarchy:vec4u,numerics:vec4f}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
@group(0) @binding(0) var<storage,read> inputVector:array<f32>;
@group(0) @binding(1) var<storage,read_write> outputVector:array<f32>;
@group(0) @binding(2) var<storage,read_write> solverControl:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> accepted:array<u32>;
@group(0) @binding(4) var<storage,read> worksets:array<u32>;
@group(0) @binding(5) var<uniform> worksetLayout:vec4u;
@group(0) @binding(6) var<storage,read> topology:array<u32>;
@group(0) @binding(7) var<storage,read> state:array<u32>;
@group(0) @binding(8) var<storage,read> geometry:array<vec4u>;
@group(0) @binding(9) var<storage,read> metrics:array<Metric>;
@group(0) @binding(10) var<storage,read> section63Coefficients:array<f32>;
@group(0) @binding(11) var<uniform> p:Layout;
const INVALID=0xffffffffu;const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;
const KEY=0u;const FLAGS=1u;const OWNER=24u;const STATE_CHANNELS=26u;
const WORKSET_HEADER_WORDS=7u;const PAGE_RECORD_WORDS=28u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn stopped()->bool{return arrayLength(&solverControl)<2u||atomicLoad(&solverControl[0])!=0u||atomicLoad(&solverControl[1])!=0u;}
fn reportAt(flag:u32,stage:u32,row:u32){if(arrayLength(&solverControl)>0u){
 atomicOr(&solverControl[0],flag);
 if(arrayLength(&solverControl)>7u){let claim=atomicCompareExchangeWeak(&solverControl[6],0u,stage);
  if(claim.exchanged){atomicStore(&solverControl[7],row);}}
}}
fn acceptedBank()->u32{return accepted[4]&1u;}
fn worksetBase(cls:u32)->u32{return acceptedBank()*worksetLayout.y+cls*worksetLayout.x;}
fn linearLane(wg:vec3u,groups:vec3u,lane:u32)->u32{return((wg.z*groups.y+wg.y)*groups.x+wg.x)*64u+lane;}
fn workRow(item:u32,cls:u32)->u32{let base=worksetBase(cls);if(base+WORKSET_HEADER_WORDS>arrayLength(&worksets)
 ||worksets[base]!=accepted[3]||worksets[base+1u]>worksets[base+2u]||item>=worksets[base+1u]
 ||base+WORKSET_HEADER_WORDS+item>=arrayLength(&worksets)){return INVALID;}return worksets[base+WORKSET_HEADER_WORDS+item];}
fn levels()->u32{return p.hierarchy.x;}fn maxStride()->u32{return p.hierarchy.y;}fn capacity()->u32{return p.dimsCapacity.w;}
fn dims(l:u32)->vec3u{let s=1u<<l;return(p.dimsCapacity.xyz+vec3u(s-1u))/s;}
fn levelCapacity(l:u32)->u32{let d=dims(l);let threshold=maxStride()/2u;var cells=d.x;
 if(cells>=threshold||d.y>threshold/cells){return maxStride();}cells*=d.y;
 if(cells>=threshold||d.z>threshold/cells){return maxStride();}cells*=d.z;
 var result=1u;let limit=2u*cells;while(result<limit){result<<=1u;}return result;}
fn levelBase(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=levelCapacity(k);}return result;}
fn totalLevelSlots()->u32{return p.hierarchy.z;}
fn at(c:u32,l:u32,s:u32)->u32{return c*totalLevelSlots()+levelBase(l)+s;}
fn rowMapBase()->u32{return 16u;}fn workBase()->u32{return rowMapBase()+levels()*capacity();}
fn pageWorkBase()->u32{return workBase()+totalLevelSlots();}
fn logicalPageDims(l:u32)->vec3u{return(dims(l)+vec3u(7u,7u,3u))/vec3u(8u,8u,4u);}
fn logicalPageCount(l:u32)->u32{let d=logicalPageDims(l);return d.x*d.y*d.z;}
fn pageLevelOffset(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=logicalPageCount(k);}return result;}
fn pageDirectoryBase()->u32{return pageWorkBase()+PAGE_RECORD_WORDS*totalLevelSlots();}
fn transferCapacity(l:u32)->u32{return min(capacity()*8u,levelCapacity(l)*8u);}
fn transferLevelOffset(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=transferCapacity(k)*4u+4u*levelCapacity(k);}return result;}
fn transferBase()->u32{return pageDirectoryBase()+pageLevelOffset(levels());}
fn directoryBase()->u32{return transferBase()+transferLevelOffset(levels()-1u);}
fn brickDims(l:u32)->vec3u{return(dims(l)+vec3u(3u))/4u;}
fn brickCount(l:u32)->u32{let d=brickDims(l);return d.x*d.y*d.z;}
fn brickLevelOffset(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=brickCount(k);}return result;}
fn totalBrickCount()->u32{return brickLevelOffset(levels());}
fn rankedSlotsBase()->u32{return directoryBase()+16u+totalBrickCount()*4u;}
fn brickRecord(l:u32,q:vec3u)->u32{let d=brickDims(l);let b=q/4u;let dense=b.x+d.x*(b.y+d.y*b.z);
 return directoryBase()+16u+(brickLevelOffset(l)+dense)*4u;}
fn pageRecord(l:u32,page:u32)->u32{return pageWorkBase()+(levelBase(l)+page)*PAGE_RECORD_WORDS;}
fn pageNeighbour(l:u32,page:u32,ordinal:u32)->u32{return topology[pageRecord(l,page)+1u+ordinal];}
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;return vec3u(v%d.x,(v/d.x)%d.y,v/(d.x*d.y));}
fn localBit(q:vec3u)->u32{let local=q&vec3u(3u);return local.x+4u*local.y+16u*local.z;}
fn pageFor(l:u32,q:vec3u)->u32{let pages=logicalPageDims(l);let v=q/vec3u(8u,8u,4u);return topology[pageDirectoryBase()+pageLevelOffset(l)+v.x+pages.x*(v.y+pages.y*v.z)];}
fn pageSlot(l:u32,page:u32,origin:vec3u,q:vec3u,row:u32)->u32{
 let shape=vec3u(8u,8u,4u);let delta=vec3i(q/shape)-vec3i(origin/shape);
 if(any(delta<vec3i(-1))||any(delta>vec3i(1))){reportAt(2u,21u,row);return INVALID;}
 let ordinal=u32(delta.x+1)+3u*(u32(delta.y+1)+3u*u32(delta.z+1));let physical=pageNeighbour(l,page,ordinal);
 if(physical==INVALID){return INVALID;}let physicalOrigin=decode(topology[pageRecord(l,physical)],l);
 if(any(physicalOrigin/shape!=q/shape)){reportAt(2u,22u,row);return INVALID;}
 let record=brickRecord(l,q);let bit=localBit(q);let low=topology[record+1u];let high=topology[record+2u];
 if(((select(low,high,bit>=32u)>>(bit&31u))&1u)==0u){return INVALID;}
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
 let slot=topology[rankedSlotsBase()+levelBase(l)+topology[record+3u]+rank];
 if(slot>=levelCapacity(l)){reportAt(2u,23u,row);return INVALID;}return slot;}
fn originOf(h:vec4u)->vec3u{return vec3u(h.x%p.dimsCapacity.x,(h.x/p.dimsCapacity.x)%p.dimsCapacity.y,h.x/(p.dimsCapacity.x*p.dimsCapacity.y));}
fn canonicalDirection(channel:u32)->vec3i{let d=array<vec3i,18>(
 vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),
 vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),
 vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[channel];}
fn worldDirection(value:vec3i,code:u32)->vec3i{let signs=vec3i(select(1,-1,(code&1u)!=0u),select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));let q=value*signs;let permutation=(code/8u)%6u;
 if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}if(permutation==2u){return q.yxz;}
 if(permutation==3u){return q.zxy;}if(permutation==4u){return q.yzx;}return q.zyx;}
fn coefficientBase(row:u32)->u32{return acceptedBank()*p.hierarchy.w+row*19u;}
fn coefficientForDirection(row:u32,metric:Metric,direction:vec3i)->f32{let base=coefficientBase(row);for(var channel=0u;channel<18u;channel+=1u){
 if(all(worldDirection(canonicalDirection(channel),metric.transformAndFlags&63u)==direction)){return section63Coefficients[base+1u+channel];}}return 0.0;}
// Destination-owned GhostValueAccumulate. A 2:1 coarse leaf has at most
// eight fine-level aliases. Each alias gathers the (at most eighteen) active
// page neighbours that point to it. This is E^T by construction: it reads the
// same owner incidence used by propagation and performs no scatter atomic.
fn finerAdjoint(row:u32,h:vec4u,q:vec3u,l:u32,x:f32)->f32{if(l==0u){return 0.0;}let fine=l-1u;var result=0.0;
 for(var child=0u;child<8u;child+=1u){let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
  let ghostPage=pageFor(fine,ghostQ);if(ghostPage==INVALID){continue;}if(ghostPage>=levelCapacity(fine)){reportAt(2u,31u,row);continue;}
  let ghost=pageSlot(fine,ghostPage,ghostQ,ghostQ,row);
  if(ghost==INVALID||(state[at(FLAGS,fine,ghost)]&GHOST)==0u||state[at(OWNER,fine,ghost)]!=row+1u){continue;}
  for(var candidateDirection=0u;candidateDirection<18u;candidateDirection+=1u){let delta=canonicalDirection(candidateDirection);let activeQ=vec3i(ghostQ)-delta;
   if(any(activeQ<vec3i(0))||any(activeQ>=vec3i(dims(fine)))){continue;}let activeSlot=pageSlot(fine,ghostPage,ghostQ,vec3u(activeQ),row);
   if(activeSlot==INVALID||(state[at(FLAGS,fine,activeSlot)]&ACTIVE)==0u){continue;}let encoded=state[at(OWNER,fine,activeSlot)];
   if(encoded==0u||encoded>capacity()){reportAt(2u,24u,row);continue;}let other=encoded-1u;let otherMetric=metrics[other];
   let c=coefficientForDirection(other,otherMetric,delta);
   if(c>0.0){result+=c*(x-inputVector[other]);}
  }
 }return result;}
fn applyRow(row:u32){if(row>=capacity()||row>=arrayLength(&geometry)||row>=arrayLength(&metrics)||row>=arrayLength(&inputVector)){reportAt(2u,25u,row);return;}
 let h=geometry[row];let m=metrics[row];let base=coefficientBase(row);if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u||base+19u>arrayLength(&section63Coefficients)){reportAt(1u,26u,row);return;}
 let l=countTrailingZeros(h.y);let q=originOf(h)/(1u<<l);let page=pageFor(l,q);
 if(page==INVALID||page>=levelCapacity(l)){reportAt(2u,31u,row);return;}
 let x=inputVector[row];var sum=0.0;
 for(var channel=0u;channel<18u;channel+=1u){sum+=section63Coefficients[base+1u+channel];}
 var value=max(0.0,section63Coefficients[base]-sum)*x;
 for(var channel=0u;channel<18u;channel+=1u){let c=section63Coefficients[base+1u+channel];if(c==0.0){continue;}
  let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),m.transformAndFlags&63u);if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){reportAt(2u,27u,row);continue;}
  let slot=pageSlot(l,page,q,vec3u(targetQ),row);if(slot==INVALID){reportAt(2u,28u,row);continue;}let flags=state[at(FLAGS,l,slot)];
  if((flags&MG_ONLY)!=0u){continue;}let encoded=state[at(OWNER,l,slot)];if(encoded==0u||encoded>capacity()){reportAt(2u,29u,row);continue;}
  value+=c*(x-inputVector[encoded-1u]);}
 value+=finerAdjoint(row,h,q,l,x);if(!finite(value)){reportAt(4u,30u,row);}else{outputVector[row]=value;}}
@compute @workgroup_size(64) fn applyRegularInterior(@builtin(workgroup_id) wg:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){let row=workRow(linearLane(wg,groups,lane),0u);if(!stopped()&&row!=INVALID){applyRow(row);}}
@compute @workgroup_size(64) fn applyTransitionInterior(@builtin(workgroup_id) wg:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){let row=workRow(linearLane(wg,groups,lane),1u);if(!stopped()&&row!=INVALID){applyRow(row);}}
@compute @workgroup_size(64) fn applyPhysicalBoundary(@builtin(workgroup_id) wg:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){let row=workRow(linearLane(wg,groups,lane),2u);if(!stopped()&&row!=INVALID){applyRow(row);}}
@compute @workgroup_size(64) fn applyTransitionBoundary(@builtin(workgroup_id) wg:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){let row=workRow(linearLane(wg,groups,lane),3u);if(!stopped()&&row!=INVALID){applyRow(row);}}
@compute @workgroup_size(64) fn applyMergedBand(@builtin(workgroup_id) wg:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){let row=workRow(linearLane(wg,groups,lane),4u);if(!stopped()&&row!=INVALID){applyRow(row);}}
`;

export const octreeSPGridVCycleShader = /* wgsl */ `
// The five sixteen-entry tables memoize what every address helper used to
// recompute with nested loops. They are the exact CPU allocation authority, so
// no addressing arithmetic changes; only its cost does.
struct Params{dimsLevel:vec4u,capacity:vec4u,dispatchSmooth:vec4u,solve:vec2u,reserved:vec2f,delta:vec4u,
 totals:vec4u,levelCaps:array<vec4u,4>,levelBases:array<vec4u,4>,brickOffsets:array<vec4u,4>,
 pageOffsets:array<vec4u,4>,transferOffsets:array<vec4u,4>}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
struct TransferTarget{coarse:u32,weight:f32}
struct CapturePageRecord{generation:u32,changeStamp:u32,validationStamp:u32,copyStamp:u32}
struct CapturePageState{generation:u32,expectedPages:u32,validatedPages:u32,changedPages:u32,
 readyGeneration:u32,copiedPages:u32,publishedGeneration:u32,publishedRows:u32,
 error:u32,bootstrap:u32,sourceGeneration:u32,publishedSourceGeneration:u32,pages:array<CapturePageRecord>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> capturedGeometry:array<vec4u>;
@group(0) @binding(3) var<storage,read> acceptedRows:array<u32>;
@group(0) @binding(4) var<storage,read_write> topology:array<u32>;
@group(0) @binding(5) var<storage,read_write> state:array<u32>;
@group(0) @binding(6) var<storage,read_write> dispatchMeta:array<u32>;
@group(0) @binding(7) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> inputRhs:array<f32>;
@group(0) @binding(9) var<storage,read_write> outputCorrection:array<f32>;
@group(0) @binding(11) var<storage,read> sourceGeometry:array<vec4u>;
@group(0) @binding(13) var<storage,read_write> capturePages:CapturePageState;
@group(0) @binding(14) var<storage,read_write> levelDelta:array<u32>;
@group(0) @binding(15) var<storage,read_write> candidateTopology:array<u32>;
@group(0) @binding(16) var<storage,read_write> candidateState:array<u32>;
@group(0) @binding(17) var<storage,read_write> candidateDispatch:array<u32>;
@group(0) @binding(18) var<storage,read> sourceDelta:array<u32>;
@group(0) @binding(19) var<storage,read_write> correctionDispatch:array<u32>;
@group(0) @binding(20) var<storage,read> topologyMetrics:array<Metric>;
@group(0) @binding(21) var<storage,read> catalogCoefficients:array<f32>;
@group(0) @binding(22) var<storage,read_write> ghostScratch:array<u32>;
@group(0) @binding(23) var<storage,read_write> committedInputs:array<u32>;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;const INVALID=0xffffffffu;
const OVERFLOW=2u;const NONFINITE=4u;const NONPOSITIVE=8u;
const KEY=0u;const FLAGS=1u;const DIAG=2u;const XP=3u;const XM=4u;const YP=5u;const YM=6u;const ZP=7u;const ZM=8u;
const XYPP=9u;const XYPM=10u;const XYMP=11u;const XYMM=12u;const XZPP=13u;const XZPM=14u;const XZMP=15u;const XZMM=16u;
const YZPP=17u;const YZPM=18u;const YZMP=19u;const YZMM=20u;
const RHS=21u;const A=22u;const B=23u;const OWNER=24u;const SPECTRAL=25u;
const STATE_CHANNELS=26u;
const PAGE_X=8u;const PAGE_Y=8u;const PAGE_Z=4u;
const HALO_X=10u;const HALO_Y=10u;const HALO_Z=6u;
const PAGE_ELEMENTS=256u;const HALO_ELEMENTS=600u;
var<workgroup> pageSlots:array<u32,600>;
var<workgroup> pageA:array<f32,600>;
var<workgroup> pageB:array<f32,600>;
var<workgroup> pageRhs:array<f32,600>;
var<workgroup> pageDiagonal:array<f32,600>;
const STRUCTURED_CANDIDATE_READY=0x5356454cu;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn stopped()->bool{return atomicLoad(&control[0])!=0u||atomicLoad(&control[1])!=0u;}
fn sourceControlReady()->bool{if(arrayLength(&acceptedRows)<6u){return false;}if(p.solve.y==0u){return acceptedRows[0]==0u&&acceptedRows[3]!=0u;}if(p.solve.y==1u){return acceptedRows[0]==STRUCTURED_CANDIDATE_READY&&acceptedRows[4]!=0u;}return false;}
fn sourceGeneration()->u32{return select(0u,select(acceptedRows[3],acceptedRows[4],p.solve.y==1u),sourceControlReady());}
fn reportAt(flag:u32,stage:u32,index:u32){atomicOr(&control[0],flag);for(var retry=0u;retry<16u;retry+=1u){
 let claim=atomicCompareExchangeWeak(&control[6],0u,stage);if(claim.exchanged){atomicStore(&control[7],index);return;}
 if(claim.old_value!=0u){return;}}}
fn report(flag:u32){reportAt(flag,60u,INVALID);}fn rows()->u32{return min(select(0u,acceptedRows[2],sourceControlReady()),p.capacity.x);}fn level()->u32{return p.dimsLevel.w;}
fn acceptedBank()->u32{return select(acceptedRows[4],acceptedRows[5],p.solve.y==1u)&1u;}
fn geometry(row:u32)->vec4u{return capturedGeometry[row];}
fn sourceRowGeometry(row:u32)->vec4u{return sourceGeometry[acceptedBank()*p.capacity.x+row];}
fn maxStride()->u32{return p.capacity.z;}fn levels()->u32{return p.capacity.y;}fn transferStride()->u32{return p.capacity.w;}
fn dims(l:u32)->vec3u{let s=1u<<l;return (p.dimsLevel.xyz+vec3u(s-1u))/s;}
fn levelTable(l:u32)->vec2u{let clamped=min(l,15u);return vec2u(clamped>>2u,clamped&3u);}
fn levelCapacity(l:u32)->u32{let t=levelTable(l);return p.levelCaps[t.x][t.y];}
fn levelBase(l:u32)->u32{let t=levelTable(l);return p.levelBases[t.x][t.y];}
fn totalLevelSlots()->u32{return p.totals.x;}
fn transferCapacity(l:u32)->u32{return min(transferStride(),levelCapacity(l)*8u);}
fn transferLevelOffset(l:u32)->u32{let t=levelTable(l);return p.transferOffsets[t.x][t.y];}
fn rowIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.x*64u;}fn slotIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.y*64u;}
fn boundedLinearIndex(g:vec3u)->u32{return g.x+g.y*65535u*64u;}
fn transferIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.z*64u;}fn at(c:u32,l:u32,s:u32)->u32{return c*totalLevelSlots()+levelBase(l)+s;}
fn loadf(c:u32,l:u32,s:u32)->f32{return bitcast<f32>(state[at(c,l,s)]);}fn storef(c:u32,l:u32,s:u32,v:f32){state[at(c,l,s)]=bitcast<u32>(v);}
fn rowMapBase()->u32{return 16u;}fn workBase()->u32{return rowMapBase()+levels()*p.capacity.x;}
fn pageWorkBase()->u32{return workBase()+totalLevelSlots();}
fn logicalPageDims(l:u32)->vec3u{return(dims(l)+vec3u(7u,7u,3u))/vec3u(8u,8u,4u);}
fn logicalPageCount(l:u32)->u32{let d=logicalPageDims(l);return d.x*d.y*d.z;}
fn pageLevelOffset(l:u32)->u32{let t=levelTable(l);return p.pageOffsets[t.x][t.y];}
fn pageDirectoryBase()->u32{return pageWorkBase()+28u*totalLevelSlots();}
fn pageRecord(l:u32,i:u32)->u32{return pageWorkBase()+(levelBase(l)+i)*28u;}
fn pageKey(l:u32,i:u32)->u32{return topology[pageRecord(l,i)];}
fn pageNeighbour(l:u32,i:u32,ordinal:u32)->u32{return topology[pageRecord(l,i)+1u+ordinal];}
fn transferBase()->u32{return pageDirectoryBase()+pageLevelOffset(levels());}fn rowMap(l:u32,r:u32)->u32{return topology[rowMapBase()+l*p.capacity.x+r];}
fn workSlot(l:u32,i:u32)->u32{return topology[workBase()+levelBase(l)+i];}
fn transferWord(l:u32,i:u32,w:u32)->u32{return transferBase()+transferLevelOffset(l)+i*4u+w;}
fn parentHeadBase(l:u32)->u32{return transferBase()+transferLevelOffset(l)+transferCapacity(l)*4u;}
fn parentTailBase(l:u32)->u32{return parentHeadBase(l)+levelCapacity(l);}
fn fineHeadBase(l:u32)->u32{return parentTailBase(l)+levelCapacity(l);}
fn fineCountBase(l:u32)->u32{return fineHeadBase(l)+levelCapacity(l);}
fn directoryBase()->u32{return transferBase()+transferLevelOffset(levels()-1u);}
fn brickDims(l:u32)->vec3u{return(dims(l)+vec3u(3u))/4u;}fn brickCount(l:u32)->u32{let d=brickDims(l);return d.x*d.y*d.z;}
fn brickLevelOffset(l:u32)->u32{let t=levelTable(l);return p.brickOffsets[t.x][t.y];}
fn totalBrickCount()->u32{return p.totals.y;}
fn brickRecord(l:u32,q:vec3u)->u32{let d=brickDims(l);let b=q/4u;let dense=b.x+d.x*(b.y+d.y*b.z);
 return directoryBase()+16u+(brickLevelOffset(l)+dense)*4u;}
fn rankedSlotsBase()->u32{return directoryBase()+16u+totalBrickCount()*4u;}
fn localBit(q:vec3u)->u32{let local=q&vec3u(3u);return local.x+4u*local.y+16u*local.z;}
const DISPATCH_WORDS=12u;
fn count(l:u32)->u32{return dispatchMeta[l*DISPATCH_WORDS];}fn transferCount(l:u32)->u32{return dispatchMeta[l*DISPATCH_WORDS+1u];}
fn pageCount(l:u32)->u32{return dispatchMeta[l*DISPATCH_WORDS+8u];}
fn lifecycleBase()->u32{return levels()*DISPATCH_WORDS;}fn previousValid()->bool{return dispatchMeta[lifecycleBase()]==1u;}
fn previousRows()->u32{return dispatchMeta[lifecycleBase()+1u];}
const DELTA_WORDS=8u;const DELTA_TOPOLOGY=1u;const DELTA_STENCIL=2u;const DELTA_ERROR=4u;
fn deltaAt(l:u32,w:u32)->u32{return l*DELTA_WORDS+w;}
fn deltaFlags(l:u32)->u32{return levelDelta[deltaAt(l,1u)];}
fn topologyDirty(l:u32)->bool{return l<levels()&&(deltaFlags(l)&DELTA_TOPOLOGY)!=0u;}
fn stencilDirty(l:u32)->bool{return l<levels()&&(deltaFlags(l)&DELTA_STENCIL)!=0u;}
fn levelDirty(l:u32)->bool{return topologyDirty(l)||stencilDirty(l);}
fn markDirtyFrom(first:u32,flags:u32,firstRow:u32,rowEnd:u32){
 for(var l=first;l<levels();l+=1u){levelDelta[deltaAt(l,0u)]=captureGeneration();
  levelDelta[deltaAt(l,1u)]|=flags;levelDelta[deltaAt(l,2u)]=min(levelDelta[deltaAt(l,2u)],firstRow);
  levelDelta[deltaAt(l,3u)]=max(levelDelta[deltaAt(l,3u)],rowEnd);}
}
const CAPTURE_PAGE_ROWS=64u;
const ROW_DELTA_VALID=0x52444c54u;
fn capturePageCount()->u32{return arrayLength(&capturePages.pages);}
fn captureGeneration()->u32{return capturePages.generation;}
fn captureExpectedPages()->u32{return capturePages.expectedPages;}
fn capturePageGeneration(page:u32)->u32{return capturePages.pages[page].generation;}
fn capturePageStamp(page:u32)->u32{return capturePages.pages[page].changeStamp;}
fn captureFailed()->bool{return capturePages.error!=0u;}
fn captureReport(flag:u32){capturePages.error|=flag;}
fn deltaControl(word:u32)->u32{return sourceDelta[p.delta.y+word];}
fn deltaOldRow(row:u32)->u32{let encoded=sourceDelta[p.delta.z+row]&0x7fffffffu;
 return select(INVALID,encoded-1u,encoded!=0u);}
fn deltaDirtyRow(index:u32)->u32{return sourceDelta[p.delta.w+index];}
fn deltaAccepted(n:u32)->bool{
 if(p.delta.x!=p.capacity.x||p.delta.y+16u>arrayLength(&sourceDelta)
  ||p.delta.z+p.delta.x>arrayLength(&sourceDelta)||p.delta.w+p.delta.x>arrayLength(&sourceDelta)){return false;}
 let previous=deltaControl(1u);let carried=deltaControl(2u);let added=deltaControl(3u);
 let retired=deltaControl(4u);let dirty=deltaControl(5u);
 return deltaControl(0u)==n&&deltaControl(7u)!=0u&&deltaControl(8u)==ROW_DELTA_VALID
  &&carried<=min(n,previous)&&n==carried+added&&n==previous+added-retired&&dirty<=n;
}
fn captureBootstrap()->bool{return capturePages.bootstrap!=0u;}
fn captureWorkCount()->u32{return select(deltaControl(5u),captureExpectedPages(),captureBootstrap());}
fn captureWorkPage(work:u32)->u32{return select(deltaDirtyRow(work)/CAPTURE_PAGE_ROWS,work,captureBootstrap());}
fn captureWorkUnique(work:u32,page:u32)->bool{
 return captureBootstrap()||work==0u||deltaDirtyRow(work-1u)/CAPTURE_PAGE_ROWS!=page;
}
fn sameL1Topology(a:vec4u,b:vec4u)->bool{return a.x==b.x&&a.y==b.y;}
fn validSection63Row(row:u32)->bool{if(row>=rows()||row>=arrayLength(&topologyMetrics)){return false;}let m=topologyMetrics[row];
 if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u||m.caseId>=arrayLength(&catalogCoefficients)/19u){return false;}
 let base=m.caseId*19u;for(var channel=0u;channel<19u;channel+=1u){let c=catalogCoefficients[base+channel];if(!finite(c)||c<0.0){return false;}}
 return catalogCoefficients[base]>0.0;}
@compute @workgroup_size(1)
fn prepareCorrectionDispatches(){
 let noRows=stopped();let words=levels()*DISPATCH_WORDS+2u;
 for(var word=0u;word<words;word+=1u){
  var value=dispatchMeta[word];let local=word%DISPATCH_WORDS;
  if(noRows&&(local==2u||local==5u||local==9u)){value=0u;}
  correctionDispatch[word]=value;
 }
}
var<workgroup> captureLaneFlags:array<u32,64>;
var<workgroup> captureLaneFirst:array<u32,64>;
var<workgroup> captureLaneRange:array<vec2u,64>;
var<workgroup> captureLaneChanged:array<u32,64>;
fn coordKey(q:vec3u,l:u32)->u32{let d=dims(l);return q.x+d.x*(q.y+d.y*q.z)+1u;}
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;return vec3u(v%d.x,(v/d.x)%d.y,v/(d.x*d.y));}
fn insertionHash(key:u32,l:u32)->u32{var h=key*0x9e3779b1u;h=(h^(h>>16u))*0x7feb352du;return(h^(h>>15u))&(levelCapacity(l)-1u);}
fn directoryLookup(l:u32,q:vec3u,requirePublication:bool)->u32{if(l>=levels()||any(q>=dims(l))){return INVALID;}
 let generation=topology[directoryBase()+2u+l];if(generation==0u){reportAt(OVERFLOW,61u,l);return INVALID;}
 let record=brickRecord(l,q);if(topology[record]!=generation){reportAt(OVERFLOW,62u,record);return INVALID;}let bit=localBit(q);let word=topology[record+1u+(bit>>5u)];
 let flag=1u<<(bit&31u);if((word&flag)==0u){return INVALID;}let low=topology[record+1u];let high=topology[record+2u];
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}let ranked=topology[record+3u]+rank;
 if(ranked>=count(l)||ranked>=levelCapacity(l)){reportAt(OVERFLOW,63u,ranked);return INVALID;}let slot=topology[rankedSlotsBase()+levelBase(l)+ranked];
 if(slot>=levelCapacity(l)||state[at(KEY,l,slot)]!=coordKey(q,l)){reportAt(OVERFLOW,64u,slot);return INVALID;}return slot;}
fn find(l:u32,q:vec3u)->u32{return directoryLookup(l,q,true);}
// Page hot loops consume the immutable physical adjacency record. The only
// mask/rank operation is against the selected neighbouring page's four 4^3
// bricks; no global directory lookup or generation read is repeated for the
// 600 staged halo values.
fn pageSlot(l:u32,page:u32,origin:vec3u,q:vec3u)->u32{
 let shape=vec3u(PAGE_X,PAGE_Y,PAGE_Z);let ownPage=origin/shape;let qPage=q/shape;
 let delta=vec3i(qPage)-vec3i(ownPage);if(any(delta<vec3i(-1))||any(delta>vec3i(1))){reportAt(OVERFLOW,65u,page);return INVALID;}
 let ordinal=u32(delta.x+1)+3u*(u32(delta.y+1)+3u*u32(delta.z+1));let physical=pageNeighbour(l,page,ordinal);
 if(physical==INVALID){return INVALID;}if(physical>=pageCount(l)){reportAt(OVERFLOW,66u,physical);return INVALID;}
 let physicalOrigin=decode(pageKey(l,physical),l);if(any(physicalOrigin/shape!=qPage)){reportAt(OVERFLOW,67u,physical);return INVALID;}
 let record=brickRecord(l,q);let bit=localBit(q);let word=topology[record+1u+(bit>>5u)];
 if((word&(1u<<(bit&31u)))==0u){return INVALID;}let low=topology[record+1u];let high=topology[record+2u];
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}let ranked=topology[record+3u]+rank;
 if(ranked>=count(l)||ranked>=levelCapacity(l)){reportAt(OVERFLOW,68u,ranked);return INVALID;}
 let slot=topology[rankedSlotsBase()+levelBase(l)+ranked];
 if(slot>=levelCapacity(l)||state[at(KEY,l,slot)]!=coordKey(q,l)){reportAt(OVERFLOW,69u,slot);return INVALID;}return slot;
}
fn originOf(h:vec4u)->vec3u{return vec3u(h.x%p.dimsLevel.x,(h.x/p.dimsLevel.x)%p.dimsLevel.y,h.x/(p.dimsLevel.x*p.dimsLevel.y));}
fn contactCoord(own:vec4u,other:vec4u,l:u32)->vec3u{let scale=1u<<l;let begin=originOf(own);let finish=begin+vec3u(own.y);
 let otherBegin=originOf(other);let otherFinish=otherBegin+vec3u(other.y);var result=vec3u(0u);
 for(var axis=0u;axis<3u;axis+=1u){if(otherBegin[axis]>=finish[axis]){result[axis]=(finish[axis]-1u)/scale;}
  else if(otherFinish[axis]<=begin[axis]){result[axis]=begin[axis]/scale;}else{let centre=(2u*otherBegin[axis]+other.y)/(2u*scale);
   result[axis]=clamp(centre,begin[axis]/scale,(finish[axis]-1u)/scale);}}return result;}
// Cold bootstrap validates every L1 page once. Recurring generations consume
// only the producer's compact, sorted dirty-row authority; the first dirty row
// in each page exclusively validates and stamps that complete page. No warm
// kernel scans the row publication, entry arena, or page table globally.
@compute @workgroup_size(1) fn beginL1CapturePlan(){
 var generation=capturePages.publishedGeneration+1u;if(generation==0u){generation=1u;}
 let n=rows();let expected=(n+CAPTURE_PAGE_ROWS-1u)/CAPTURE_PAGE_ROWS;
 capturePages.generation=generation;capturePages.expectedPages=expected;
 capturePages.validatedPages=0u;capturePages.changedPages=0u;
 capturePages.readyGeneration=0u;capturePages.copiedPages=0u;
 capturePages.error=0u;
 capturePages.bootstrap=select(0u,1u,capturePages.publishedGeneration==0u);
 capturePages.sourceGeneration=sourceGeneration();
 for(var l=0u;l<levels();l+=1u){for(var w=0u;w<DELTA_WORDS;w+=1u){levelDelta[deltaAt(l,w)]=0u;}
  levelDelta[deltaAt(l,2u)]=0xffffffffu;}
 let publishedGeneration=capturePages.publishedGeneration;let publishedRows=capturePages.publishedRows;
 if(!sourceControlReady()||acceptedRows[2]>p.capacity.x||expected>capturePageCount()
  ||publishedRows>p.capacity.x||(publishedGeneration==0u&&publishedRows!=0u)
  ||capturePages.sourceGeneration==0u||capturePages.sourceGeneration!=deltaControl(7u)
  ||!deltaAccepted(n)){captureReport(OVERFLOW);}
 if(!previousValid()&&publishedGeneration!=0u){captureReport(OVERFLOW);}
}
@compute @workgroup_size(64) fn planL1CaptureDelta(@builtin(local_invocation_index) lane:u32){
 let generation=captureGeneration();let workCount=captureWorkCount();
 var changed=0u;var first=levels();var flags=0u;var firstRow=rows();var rowEnd=0u;
 for(var work=lane;work<workCount;work+=64u){let page=captureWorkPage(work);
  if(page>=captureExpectedPages()||page>=capturePageCount()){flags|=DELTA_ERROR;continue;}
  if(!captureWorkUnique(work,page)){continue;}
  var pageFlags=1u;var pageFirst=levels();
  let begin=page*CAPTURE_PAGE_ROWS;let end=min(rows(),begin+CAPTURE_PAGE_ROWS);
  for(var r=begin;r<end;r+=1u){let source=sourceRowGeometry(r);let volume=p.dimsLevel.x*p.dimsLevel.y*p.dimsLevel.z;
   if(!validSection63Row(r)||source.x>=volume||source.y==0u||(source.y&(source.y-1u))!=0u){pageFlags|=2u;continue;}
   let old=select(deltaOldRow(r),INVALID,captureBootstrap());
   // The direct resolved-row producer's compact dirty list is the sparse-identity
   // authority. A dirty fixed row can change a neighbor handle without moving
   // its cell, so rebuild its dependent sparse suffix instead of comparing a
   // second adjacency representation.
   var topologyChanged=true;
   var stencilChanged=true;
   if(!topologyChanged){let captured=capturedGeometry[old];topologyChanged=!sameL1Topology(source,captured);
    if(topologyChanged){pageFirst=min(pageFirst,min(firstTrailingBit(source.y),firstTrailingBit(captured.y)));}}
   else{pageFirst=0u;}
   if(topologyChanged){pageFlags|=4u;stencilChanged=true;}
   if(stencilChanged){pageFlags|=8u;pageFirst=min(pageFirst,firstTrailingBit(source.y));}
  }
  capturePages.pages[page].changeStamp=generation;
  capturePages.pages[page].validationStamp=generation;
  capturePages.pages[page].copyStamp=(pageFlags&0xffu)|((pageFirst&0xffffu)<<16u);
  if((pageFlags&2u)!=0u){flags|=DELTA_ERROR;}
  first=min(first,pageFirst);
  flags|=select(0u,DELTA_TOPOLOGY,(pageFlags&4u)!=0u);
  flags|=select(0u,DELTA_STENCIL,(pageFlags&8u)!=0u);
  firstRow=min(firstRow,begin);rowEnd=max(rowEnd,end);changed+=1u;
 }
 captureLaneFlags[lane]=flags;captureLaneFirst[lane]=first;captureLaneRange[lane]=vec2u(firstRow,rowEnd);
 captureLaneChanged[lane]=changed;workgroupBarrier();
 if(lane==0u){var totalChanged=0u;var firstLevel=levels();var totalFlags=0u;var affectedBegin=rows();var affectedEnd=0u;
  for(var i=0u;i<64u;i+=1u){totalChanged+=captureLaneChanged[i];firstLevel=min(firstLevel,captureLaneFirst[i]);
   totalFlags|=captureLaneFlags[i];affectedBegin=min(affectedBegin,captureLaneRange[i].x);affectedEnd=max(affectedEnd,captureLaneRange[i].y);}
  if(captureFailed()||generation==0u||captureExpectedPages()!=(rows()+CAPTURE_PAGE_ROWS-1u)/CAPTURE_PAGE_ROWS
   ||captureExpectedPages()>capturePageCount()||(totalFlags&DELTA_ERROR)!=0u){captureReport(OVERFLOW);return;}
  if(captureBootstrap()||deltaControl(3u)!=0u||deltaControl(4u)!=0u){
   firstLevel=0u;totalFlags=DELTA_TOPOLOGY|DELTA_STENCIL;affectedBegin=0u;affectedEnd=rows();}
  if(firstLevel<levels()){markDirtyFrom(firstLevel,totalFlags,affectedBegin,affectedEnd);}
  capturePages.validatedPages=totalChanged;capturePages.changedPages=totalChanged;capturePages.readyGeneration=generation;
 }
}
@compute @workgroup_size(64) fn commitChangedL1(@builtin(local_invocation_index) lane:u32){
 let generation=captureGeneration();let workCount=captureWorkCount();
 for(var work=lane;work<workCount;work+=64u){let page=captureWorkPage(work);
  if(page>=captureExpectedPages()||!captureWorkUnique(work,page)||capturePageStamp(page)!=generation
   ||capturePages.readyGeneration!=generation||captureFailed()){continue;}
  let end=min(rows(),(page+1u)*CAPTURE_PAGE_ROWS);
  for(var r=page*CAPTURE_PAGE_ROWS;r<end;r+=1u){capturedGeometry[r]=sourceRowGeometry(r);}
  capturePages.pages[page].generation=generation;capturePages.pages[page].copyStamp=generation;
 }
}
@compute @workgroup_size(64) fn finalizeL1CapturePublication(@builtin(local_invocation_index) lane:u32){
 let generation=captureGeneration();let workCount=captureWorkCount();
 var copied=0u;var invalid=0u;for(var work=lane;work<workCount;work+=64u){let page=captureWorkPage(work);
  if(page>=captureExpectedPages()||!captureWorkUnique(work,page)){continue;}let record=capturePages.pages[page];
  if(record.validationStamp!=generation||record.changeStamp!=generation
   ||record.copyStamp!=generation||record.generation!=generation){invalid=1u;}copied+=1u;
 }
 captureLaneFlags[lane]=invalid;captureLaneChanged[lane]=copied;workgroupBarrier();
 if(lane==0u){var totalCopied=0u;var anyInvalid=0u;for(var i=0u;i<64u;i+=1u){
   totalCopied+=captureLaneChanged[i];anyInvalid|=captureLaneFlags[i];}
  capturePages.copiedPages=totalCopied;
  if(captureFailed()||generation==0u||capturePages.readyGeneration!=generation||anyInvalid!=0u
   ||totalCopied!=capturePages.changedPages){captureReport(OVERFLOW);return;}
  capturePages.publishedGeneration=generation;capturePages.publishedRows=rows();
  capturePages.publishedSourceGeneration=capturePages.sourceGeneration;
 }
}
// Sound unchanged-input predicate. The hierarchy is a pure function of the row
// count, every row's captured fixed geometry, and every row's Section 6.3
// case/transform (the catalog table, domain, and finest width are immutable for
// the instance). committedInputs holds that exact tuple for the last accepted
// publication, so a generation with no stamped mismatch provably reproduces the
// published hierarchy and may retire its dirty flags. Any difference, any
// bootstrap, any earlier capture fault, and any absent fingerprint stamps the
// generation and leaves the rebuild unconditional.
const COMMITTED_VALID=0u;const COMMITTED_ROWS=1u;const COMMITTED_MISMATCH=2u;
const COMMITTED_HEADER=4u;const COMMITTED_STRIDE=4u;
fn committedRowBase(r:u32)->u32{return COMMITTED_HEADER+r*COMMITTED_STRIDE;}
@compute @workgroup_size(64) fn probeCandidateSkip(@builtin(global_invocation_id) g:vec3u){
 let generation=captureGeneration();let r=rowIndex(g);
 if(r==0u){if(committedInputs[COMMITTED_VALID]!=1u||committedInputs[COMMITTED_ROWS]!=rows()
  ||captureBootstrap()||captureFailed()||!sourceControlReady()||generation==0u){
   committedInputs[COMMITTED_MISMATCH]=generation;}}
 if(r>=rows()){return;}
 let source=sourceRowGeometry(r);let m=topologyMetrics[r];let base=committedRowBase(r);
 if(committedInputs[base]!=source.x||committedInputs[base+1u]!=source.y
  ||committedInputs[base+2u]!=m.caseId
  ||committedInputs[base+3u]!=(m.transformAndFlags&63u)){
  committedInputs[COMMITTED_MISMATCH]=generation;}
}
@compute @workgroup_size(1) fn applyCandidateSkip(){
 let generation=captureGeneration();
 if(generation==0u||committedInputs[COMMITTED_MISMATCH]==generation
  ||committedInputs[COMMITTED_VALID]!=1u||captureBootstrap()||captureFailed()){return;}
 for(var l=0u;l<levels();l+=1u){levelDelta[deltaAt(l,1u)]=0u;}
}
@compute @workgroup_size(64) fn clearCorrection(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){outputCorrection[r]=0.0;}}
fn cAt(c:u32,l:u32,s:u32)->u32{return c*totalLevelSlots()+levelBase(l)+s;}
fn cCount(l:u32)->u32{return candidateDispatch[l*DISPATCH_WORDS];}
fn cWorkSlot(l:u32,i:u32)->u32{return candidateTopology[workBase()+levelBase(l)+i];}
fn cRowMap(l:u32,r:u32)->u32{return candidateTopology[rowMapBase()+l*p.capacity.x+r];}
fn cLoadf(c:u32,l:u32,s:u32)->f32{return bitcast<f32>(candidateState[cAt(c,l,s)]);}
fn cStoref(c:u32,l:u32,s:u32,v:f32){candidateState[cAt(c,l,s)]=bitcast<u32>(v);}
fn candidateReport(l:u32){levelDelta[deltaAt(l,4u)]|=DELTA_ERROR;}
fn cMergeClass(index:u32,incoming:u32){let old=candidateState[index];var merged=MG_ONLY;
 if((old&ACTIVE)!=0u||(incoming&ACTIVE)!=0u){merged=ACTIVE;}else if((old&GHOST)!=0u||(incoming&GHOST)!=0u){merged=GHOST;}
 candidateState[index]=merged;}
fn cInsert(l:u32,q:vec3u,flags:u32)->u32{let key=coordKey(min(q,dims(l)-vec3u(1u)),l);var slot=insertionHash(key,l);
 for(var probe=0u;probe<256u;probe+=1u){let index=cAt(KEY,l,slot);let old=candidateState[index];
  if(old==key){cMergeClass(cAt(FLAGS,l,slot),flags);return slot;}if(old==0u){candidateState[index]=key;
   cMergeClass(cAt(FLAGS,l,slot),flags);let w=cCount(l);if(w>=levelCapacity(l)){candidateReport(l);return INVALID;}
   candidateDispatch[l*DISPATCH_WORDS]=w+1u;candidateTopology[workBase()+levelBase(l)+w]=slot;return slot;}
  slot=(slot+1u)&(levelCapacity(l)-1u);}candidateReport(l);return INVALID;}
fn cInsertOwned(l:u32,q:vec3u,flags:u32,owner:u32)->u32{let slot=cInsert(l,q,flags);if(slot==INVALID){return INVALID;}
 let encoded=owner+1u;let old=candidateState[cAt(OWNER,l,slot)];if(old==encoded){return slot;}
 if(old!=0u){candidateReport(l);return INVALID;}candidateState[cAt(OWNER,l,slot)]=encoded;return slot;}
fn selectedCount(l:u32)->u32{return select(count(l),cCount(l),topologyDirty(l));}
fn selectedWorkSlot(l:u32,i:u32)->u32{return select(workSlot(l,i),cWorkSlot(l,i),topologyDirty(l));}
fn selectedRowMap(l:u32,r:u32)->u32{return select(rowMap(l,r),cRowMap(l,r),topologyDirty(l));}
fn selectedState(c:u32,l:u32,s:u32)->u32{return select(state[at(c,l,s)],candidateState[cAt(c,l,s)],topologyDirty(l));}
// One immutable transfer record. The record index is no longer a running
// counter: it is the exclusive prefix sum of the per-fine fan-out, so the
// contiguous fine range and the ascending parent chain are identical to the
// former append order without any cross-invocation serialization.
fn cAppendTransfer(l:u32,record:u32,fine:u32,coarse:u32,weight:f32){
 candidateTopology[transferWord(l,record,0u)]=fine;candidateTopology[transferWord(l,record,1u)]=coarse;
 candidateTopology[transferWord(l,record,2u)]=bitcast<u32>(weight);candidateTopology[transferWord(l,record,3u)]=INVALID;}
fn cDirectoryLookup(l:u32,q:vec3u)->u32{if(l>=levels()||any(q>=dims(l))){return INVALID;}
 let generation=levelDelta[deltaAt(l,6u)];let record=brickRecord(l,q);if(candidateTopology[record]!=generation){candidateReport(l);return INVALID;}
 let bit=localBit(q);let word=candidateTopology[record+1u+(bit>>5u)];let flag=1u<<(bit&31u);if((word&flag)==0u){return INVALID;}
 let low=candidateTopology[record+1u];let high=candidateTopology[record+2u];let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);
 var rank=countOneBits(low&lower);if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
 let ranked=candidateTopology[record+3u]+rank;if(ranked>=cCount(l)||ranked>=levelCapacity(l)){candidateReport(l);return INVALID;}
 let slot=candidateTopology[rankedSlotsBase()+levelBase(l)+ranked];
 if(slot>=levelCapacity(l)||candidateState[cAt(KEY,l,slot)]!=coordKey(q,l)){candidateReport(l);return INVALID;}return slot;}
fn cLookup(l:u32,q:vec3u)->u32{if(l>=levels()||any(q>=dims(l))){return INVALID;}let wanted=coordKey(q,l);var slot=insertionHash(wanted,l);
 for(var probe=0u;probe<256u;probe+=1u){let old=candidateState[cAt(KEY,l,slot)];if(old==wanted){return slot;}if(old==0u){return INVALID;}slot=(slot+1u)&(levelCapacity(l)-1u);}return INVALID;}
fn section63Direction(k:u32)->vec3i{let d=array<vec3i,18>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[k];}
fn section63WorldDirection(value:vec3i,code:u32)->vec3i{let signs=vec3i(select(1,-1,(code&1u)!=0u),select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));let q=value*signs;let permutation=(code/8u)%6u;
 if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}if(permutation==2u){return q.yxz;}if(permutation==3u){return q.zxy;}if(permutation==4u){return q.yzx;}return q.zyx;}

// Phase 1 (slot/row parallel). Every capacity-wide reset the singleton used to
// walk serially. Pure constant stores, so the published bytes cannot depend on
// scheduling. The transfer-chain reset moves here too: nothing between this
// dispatch and the transfer builder touches the parent/fine chain arena.
@compute @workgroup_size(64) fn clearCandidateLevels(@builtin(global_invocation_id) g:vec3u){
 let i=boundedLinearIndex(g);
 for(var l=0u;l<levels();l+=1u){
  let dirtyTopology=topologyDirty(l);
  let dirtyTransfer=l+1u<levels()&&(dirtyTopology||topologyDirty(l+1u));
  if(dirtyTopology){
   if(i==0u){levelDelta[deltaAt(l,4u)]=0u;}
   if(i<levelCapacity(l)){candidateState[cAt(KEY,l,i)]=0u;
    candidateState[cAt(FLAGS,l,i)]=0u;candidateState[cAt(OWNER,l,i)]=0u;
    candidateTopology[workBase()+levelBase(l)+i]=INVALID;}
   if(i<p.capacity.x){candidateTopology[rowMapBase()+l*p.capacity.x+i]=INVALID;}
   if(i<DISPATCH_WORDS){candidateDispatch[l*DISPATCH_WORDS+i]=0u;}
  }
  if(dirtyTransfer){
   if(i==0u){candidateDispatch[l*DISPATCH_WORDS+1u]=0u;}
   if(i<levelCapacity(l)){candidateTopology[parentHeadBase(l)+i]=INVALID;
    candidateTopology[parentTailBase(l)+i]=INVALID;
    candidateTopology[fineHeadBase(l)+i]=INVALID;candidateTopology[fineCountBase(l)+i]=0u;}
  }
 }
}
// Phase 2 (one invocation per level). Open-address insertion is the only
// order-defining step left, and levels are disjoint: an insert touches one
// level's key arena, its workset, its count and its own delta word. The row
// sequence inside a level is unchanged, so slots and workset order are
// identical to the serial builder.
@compute @workgroup_size(1) fn buildCandidateLevelSets(@builtin(workgroup_id) wg:vec3u){
 let l=wg.x;if(l>=levels()||!topologyDirty(l)){return;}
 let n=rows();let rowBase=rowMapBase()+l*p.capacity.x;
 for(var r=0u;r<n;r+=1u){let h=geometry(r);let native=firstTrailingBit(h.y);
  if(l>=native){let q=originOf(h)/(1u<<l);var slot=INVALID;if(l==native){slot=cInsertOwned(l,q,ACTIVE,r);}else{slot=cInsert(l,q,MG_ONLY);}
   if(slot!=INVALID){candidateTopology[rowBase+r]=slot;}}
 }
}
const GHOST_STRIDE=20u;
// Phase 3 (row parallel, read only). A contact-ghost alias is a pure function
// of its coordinate: presence in the finished level set, the coarse owner at
// half that coordinate, and its ACTIVE flag never depend on which row proposed
// it. Detection therefore parallelizes exactly, and the ordered insert below
// only has to replay the surviving proposals in (row,channel) order.
// One row's catalog-owned proposals. The former per-level sweep tested the
// same predicate for the same (row,channel) pairs; only its ownership changed.
fn rebuildCandidateGhostsFor(r:u32){
 let record=r*GHOST_STRIDE;ghostScratch[record]=0u;
 if(r>=rows()){return;}
 let h=geometry(r);let l=firstTrailingBit(h.y);
 if(l+1u>=levels()||!topologyDirty(l)){return;}
 let m=topologyMetrics[r];let q=originOf(h)/(1u<<l);let base=m.caseId*19u;var mask=0u;
 for(var channel=0u;channel<18u;channel+=1u){if(catalogCoefficients[base+1u+channel]==0.0){continue;}
  let targetQ=vec3i(q)+section63WorldDirection(section63Direction(channel),m.transformAndFlags&63u);
  if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))||cLookup(l,vec3u(targetQ))!=INVALID){continue;}
  let coarse=cLookup(l+1u,vec3u(targetQ)/2u);if(coarse==INVALID||(candidateState[cAt(FLAGS,l+1u,coarse)]&ACTIVE)==0u){continue;}
  let owner=candidateState[cAt(OWNER,l+1u,coarse)];if(owner==0u||owner>rows()){candidateReport(l);continue;}
  ghostScratch[record+2u+channel]=owner;mask|=1u<<channel;
 }
 ghostScratch[record]=mask;
}
@compute @workgroup_size(64) fn detectCandidateGhosts(@builtin(global_invocation_id) g:vec3u){
 let r=rowIndex(g);if(r<p.capacity.x){rebuildCandidateGhostsFor(r);}
}
// Phase 4 (one invocation per level). Replays the surviving proposals in the
// original (row,channel) order. A repeat proposal at an already-aliased
// coordinate merges GHOST into GHOST with the identical owner and appends
// nothing, which is exactly what the live presence test used to skip.
@compute @workgroup_size(1) fn insertCandidateGhosts(@builtin(workgroup_id) wg:vec3u){
 let l=wg.x;if(l+1u>=levels()||!topologyDirty(l)){return;}
 let n=rows();
 for(var r=0u;r<n;r+=1u){let base=r*GHOST_STRIDE;let mask=ghostScratch[base];if(mask==0u){continue;}
  let h=geometry(r);if(firstTrailingBit(h.y)!=l){continue;}
  let m=topologyMetrics[r];let q=originOf(h)/(1u<<l);
  for(var channel=0u;channel<18u;channel+=1u){if((mask&(1u<<channel))==0u){continue;}
   let owner=ghostScratch[base+2u+channel];
   let targetQ=vec3i(q)+section63WorldDirection(section63Direction(channel),m.transformAndFlags&63u);
   _=cInsertOwned(l,vec3u(targetQ),GHOST,owner-1u);}
 }
}
struct CoarseCorner{coordinate:vec3u,weight:f32}
// One shared definition of the cell-centred trilinear corner, so the ordered
// insertion and the parallel record writer cannot disagree on either the target
// coordinate or the exact float multiply order that forms the weight.
fn cornerTarget(q:vec3u,corner:u32)->CoarseCorner{
 let origin=q/2u;let side=vec3i(select(-1,1,(q.x&1u)!=0u),select(-1,1,(q.y&1u)!=0u),select(-1,1,(q.z&1u)!=0u));
 var targetCoord=vec3i(origin);var weight=1.0;
 for(var axis=0u;axis<3u;axis+=1u){if((corner&(1u<<axis))!=0u){targetCoord[axis]+=side[axis];weight*=0.25;}else{weight*=0.75;}}
 return CoarseCorner(vec3u(max(targetCoord,vec3i(0))),weight);}
// Read side of cInsert: the same boundary clamp, so a coordinate resolves to
// the slot the ordered insertion claimed for it.
fn cResolve(l:u32,q:vec3u)->u32{if(l>=levels()){return INVALID;}return cLookup(l,min(q,dims(l)-vec3u(1u)));}
// Exact fan-out of one fine slot: one record for a resolvable ghost alias,
// eight for a trilinear interior cell, none for a rejected owner. This is the
// per-fine count the former append counter produced.
fn transferFanOut(l:u32,fine:u32,flags:u32,q:vec3u)->u32{
 if((flags&GHOST)==0u){return 8u;}
 let encodedOwner=selectedState(OWNER,l,fine);
 if(encodedOwner==0u||encodedOwner>rows()){return 0u;}
 let owner=encodedOwner-1u;var coarse=INVALID;
 if(l+1u==firstTrailingBit(geometry(owner).y)){coarse=selectedRowMap(l+1u,owner);}
 else{coarse=cResolve(l+1u,q/2u);}
 return select(0u,1u,coarse!=INVALID);}
fn transferLive(l:u32)->bool{return l+1u<levels()&&(topologyDirty(l)||topologyDirty(l+1u));}
// Phase 5a. The only genuinely coupled step: the fine sweep of level l inserts
// coarse aliases into level l+1, and that insertion order defines level l+1's
// workset order, which in turn fixes every downstream summation order. It keeps
// one ordered owner, but now performs only the hash insertion - no record
// stores, no chain updates, no capacity counter.
fn rebuildCandidateTransferFor(l:u32){
 if(!transferLive(l)){return;}
 let selected=selectedCount(l);
 for(var i=0u;i<selected;i+=1u){let fine=selectedWorkSlot(l,i);
  let flags=selectedState(FLAGS,l,fine);let q=decode(selectedState(KEY,l,fine),l);
  if((flags&GHOST)!=0u){let encodedOwner=selectedState(OWNER,l,fine);
   if(encodedOwner==0u||encodedOwner>rows()){candidateReport(l);continue;}let owner=encodedOwner-1u;
   if(l+1u!=firstTrailingBit(geometry(owner).y)){_=cInsertOwned(l+1u,q/2u,GHOST,owner);}}
  else{for(var corner=0u;corner<8u;corner+=1u){_=cInsert(l+1u,cornerTarget(q,corner).coordinate,MG_ONLY);}}
 }
}
@compute @workgroup_size(1) fn buildCandidateLevelDeltas(){
 for(var l=0u;l+1u<levels();l+=1u){rebuildCandidateTransferFor(l);}
}
// Phase 5b (slot parallel). Publish the exact per-fine record fan-out.
@compute @workgroup_size(64) fn countCandidateTransfers(@builtin(global_invocation_id) g:vec3u){
 let i=boundedLinearIndex(g);
 for(var l=0u;l+1u<levels();l+=1u){
  if(!transferLive(l)||i>=selectedCount(l)){continue;}
  let fine=selectedWorkSlot(l,i);
  candidateTopology[fineCountBase(l)+fine]=transferFanOut(l,fine,selectedState(FLAGS,l,fine),
   decode(selectedState(KEY,l,fine),l));}
}
// Phase 5c (one invocation per level). Exclusive prefix sum over the fan-out in
// workset order. This reproduces the former append counter exactly, and the
// capacity rejection stays fail closed instead of writing past the arena.
@compute @workgroup_size(1) fn scanCandidateTransfers(@builtin(workgroup_id) wg:vec3u){
 let l=wg.x;if(!transferLive(l)){return;}
 let selected=selectedCount(l);let capacity=transferCapacity(l);var base=0u;
 for(var i=0u;i<selected;i+=1u){let fine=selectedWorkSlot(l,i);
  let owned=candidateTopology[fineCountBase(l)+fine];if(owned==0u){continue;}
  if(base+owned>capacity){candidateReport(l);candidateTopology[fineCountBase(l)+fine]=0u;continue;}
  candidateTopology[fineHeadBase(l)+fine]=base;base+=owned;}
 candidateDispatch[l*DISPATCH_WORDS+1u]=base;
}
// Phase 5d (slot parallel). Each fine slot owns a disjoint contiguous record
// range, so the immutable records are written without contention.
@compute @workgroup_size(64) fn writeCandidateTransfers(@builtin(global_invocation_id) g:vec3u){
 let i=boundedLinearIndex(g);
 for(var l=0u;l+1u<levels();l+=1u){
  if(!transferLive(l)||i>=selectedCount(l)){continue;}
  let fine=selectedWorkSlot(l,i);let owned=candidateTopology[fineCountBase(l)+fine];
  if(owned==0u){continue;}
  let base=candidateTopology[fineHeadBase(l)+fine];
  if(base==INVALID||base+owned>transferCapacity(l)){candidateReport(l);continue;}
  let flags=selectedState(FLAGS,l,fine);let q=decode(selectedState(KEY,l,fine),l);
  if((flags&GHOST)!=0u){let owner=selectedState(OWNER,l,fine)-1u;var coarse=INVALID;
   if(l+1u==firstTrailingBit(geometry(owner).y)){coarse=selectedRowMap(l+1u,owner);}
   else{coarse=cResolve(l+1u,q/2u);}
   if(coarse==INVALID){candidateReport(l);}
   cAppendTransfer(l,base,fine,coarse,1.0);}
  else{if(owned!=8u){candidateReport(l);continue;}
   for(var corner=0u;corner<8u;corner+=1u){let parent=cornerTarget(q,corner);
    let coarse=cResolve(l+1u,parent.coordinate);
    if(coarse==INVALID){candidateReport(l);}
    cAppendTransfer(l,base+corner,fine,coarse,parent.weight);}}
 }
}
// Phase 5e (one invocation per level). The parent chain is the ascending record
// order restricted to one coarse slot; walking the published records in index
// order reproduces it exactly, and the levels run concurrently.
@compute @workgroup_size(1) fn linkCandidateParentChains(@builtin(workgroup_id) wg:vec3u){
 let l=wg.x;if(!transferLive(l)){return;}
 let n=min(candidateDispatch[l*DISPATCH_WORDS+1u],transferCapacity(l));let limit=levelCapacity(l+1u);
 for(var j=0u;j<n;j+=1u){let coarse=candidateTopology[transferWord(l,j,1u)];
  if(coarse>=limit){candidateReport(l);continue;}
  let tail=candidateTopology[parentTailBase(l)+coarse];
  if(tail==INVALID){candidateTopology[parentHeadBase(l)+coarse]=j;}
  else{candidateTopology[transferWord(l,tail,3u)]=j;}
  candidateTopology[parentTailBase(l)+coarse]=j;}
}
fn brickOfIndex(index:u32)->vec2u{
 var l=levels();var local=0u;
 for(var k=0u;k<levels();k+=1u){let begin=brickLevelOffset(k);
  if(index>=begin&&index<begin+brickCount(k)){l=k;local=index-begin;}}
 return vec2u(l,local);
}
fn brickOrigin(l:u32,local:u32)->vec3u{let d=brickDims(l);
 return vec3u(local%d.x,(local/d.x)%d.y,local/(d.x*d.y))*4u;}
fn brickCell(origin:vec3u,bit:u32)->vec3u{return origin+vec3u(bit%4u,(bit/4u)%4u,bit/16u);}
// Phase 6 (brick parallel). The 4^3 occupancy mask is derived from the level's
// finished key arena instead of scattered from the workset, so no atomic OR and
// no scheduling order enters the published mask.
@compute @workgroup_size(64) fn markCandidateBrickOccupancy(@builtin(global_invocation_id) g:vec3u){
 let index=boundedLinearIndex(g);if(index>=totalBrickCount()){return;}
 let located=brickOfIndex(index);let l=located.x;
 if(l>=levels()||!topologyDirty(l)){return;}
 let origin=brickOrigin(l,located.y);let extent=dims(l);var low=0u;var high=0u;
 for(var bit=0u;bit<64u;bit+=1u){let q=brickCell(origin,bit);
  if(any(q>=extent)||cLookup(l,q)==INVALID){continue;}
  if(bit<32u){low|=1u<<bit;}else{high|=1u<<(bit-32u);}}
 let record=directoryBase()+16u+index*4u;
 candidateTopology[record]=levelDelta[deltaAt(l,0u)];
 candidateTopology[record+1u]=low;candidateTopology[record+2u]=high;candidateTopology[record+3u]=0u;
}
// Phase 7 (one invocation per level). The ranked base is a prefix sum over the
// level's bricks in dense order; keeping one owner per level preserves it
// exactly while the levels run concurrently.
@compute @workgroup_size(1) fn rankCandidateBricks(@builtin(workgroup_id) wg:vec3u){
 let l=wg.x;if(l>=levels()||!topologyDirty(l)){return;}
 let generation=levelDelta[deltaAt(l,0u)];levelDelta[deltaAt(l,6u)]=generation;
 let first=directoryBase()+16u+brickLevelOffset(l)*4u;let n=brickCount(l);var base=0u;
 for(var b=0u;b<n;b+=1u){let record=first+b*4u;candidateTopology[record+3u]=base;
  base+=countOneBits(candidateTopology[record+1u])+countOneBits(candidateTopology[record+2u]);}
 if(base!=cCount(l)){candidateReport(l);}
 candidateTopology[directoryBase()+2u+l]=generation;
}
// Phase 8 (brick parallel). Each brick owns a disjoint ranked range, so the
// compact slot vector is written without contention and in mask-rank order.
@compute @workgroup_size(64) fn scatterCandidateRankedSlots(@builtin(global_invocation_id) g:vec3u){
 let index=boundedLinearIndex(g);if(index>=totalBrickCount()){return;}
 let located=brickOfIndex(index);let l=located.x;
 if(l>=levels()||!topologyDirty(l)){return;}
 let origin=brickOrigin(l,located.y);let record=directoryBase()+16u+index*4u;
 let low=candidateTopology[record+1u];let high=candidateTopology[record+2u];
 let base=rankedSlotsBase()+levelBase(l)+candidateTopology[record+3u];var rank=0u;
 for(var bit=0u;bit<64u;bit+=1u){
  let word=select(low,high,bit>=32u);if(((word>>(bit&31u))&1u)==0u){continue;}
  candidateTopology[base+rank]=cLookup(l,brickCell(origin,bit));rank+=1u;}
}
fn pageOfIndex(index:u32)->vec2u{
 var l=levels();var local=0u;
 for(var k=0u;k<levels();k+=1u){let begin=pageLevelOffset(k);
  if(index>=begin&&index<begin+logicalPageCount(k)){l=k;local=index-begin;}}
 return vec2u(l,local);
}
fn logicalPageOrigin(l:u32,dense:u32)->vec3u{let d=logicalPageDims(l);
 return vec3u(dense%d.x,(dense/d.x)%d.y,dense/(d.x*d.y))*vec3u(8u,8u,4u);}
// Phase 9 (logical-page parallel). Marks raw occupancy in the page directory
// word the compaction is about to overwrite, so no scratch storage is needed.
@compute @workgroup_size(64) fn markCandidatePageOccupancy(@builtin(global_invocation_id) g:vec3u){
 let index=boundedLinearIndex(g);if(index>=p.totals.z){return;}
 let located=pageOfIndex(index);let l=located.x;
 if(l>=levels()||!topologyDirty(l)){return;}
 let origin=logicalPageOrigin(l,located.y);let extent=dims(l);var occupied=false;
 for(var by=0u;by<2u;by+=1u){for(var bx=0u;bx<2u;bx+=1u){let q=origin+vec3u(4u*bx,4u*by,0u);
   if(any(q>=extent)){continue;}let record=brickRecord(l,q);
   occupied=occupied||candidateTopology[record+1u]!=0u||candidateTopology[record+2u]!=0u;}}
 candidateTopology[pageDirectoryBase()+pageLevelOffset(l)+located.y]=select(0u,1u,occupied);
}
// Phase 10 (one invocation per level). Physical page identifiers stay assigned
// in dense logical order, which is the identity the halo staging depends on.
@compute @workgroup_size(1) fn compactCandidatePages(@builtin(workgroup_id) wg:vec3u){
 let l=wg.x;if(l>=levels()||!topologyDirty(l)){return;}
 let logicalPages=logicalPageCount(l);let directory=pageDirectoryBase()+pageLevelOffset(l);
 var pageTotal=0u;var overflow=false;
 for(var dense=0u;dense<logicalPages;dense+=1u){
  let occupied=candidateTopology[directory+dense]!=0u;
  if(!occupied||overflow){candidateTopology[directory+dense]=INVALID;continue;}
  if(pageTotal>=levelCapacity(l)){candidateReport(l);overflow=true;
   candidateTopology[directory+dense]=INVALID;continue;}
  candidateTopology[pageRecord(l,pageTotal)]=coordKey(logicalPageOrigin(l,dense),l);
  candidateTopology[directory+dense]=pageTotal;pageTotal+=1u;}
 candidateDispatch[l*DISPATCH_WORDS+8u]=select(pageTotal,0u,overflow);
}
// Phase 11 (page parallel). The immutable 27-entry physical record is a pure
// read of the compacted directory.
@compute @workgroup_size(64) fn linkCandidatePageNeighbours(@builtin(global_invocation_id) g:vec3u){
 let index=boundedLinearIndex(g);if(index>=p.totals.z){return;}
 let located=pageOfIndex(index);let l=located.x;let page=located.y;
 if(l>=levels()||!topologyDirty(l)||page>=candidateDispatch[l*DISPATCH_WORDS+8u]){return;}
 let pageDims=logicalPageDims(l);let record=pageRecord(l,page);
 let origin=decode(candidateTopology[record],l)/vec3u(8u,8u,4u);
 let directory=pageDirectoryBase()+pageLevelOffset(l);
 for(var ordinal=0u;ordinal<27u;ordinal+=1u){let dx=i32(ordinal%3u)-1;let yz=ordinal/3u;
  let neighbour=vec3i(origin)+vec3i(dx,i32(yz%3u)-1,i32(yz/3u)-1);var physical=INVALID;
  if(all(neighbour>=vec3i(0))&&all(neighbour<vec3i(pageDims))){let q=vec3u(neighbour);
   physical=candidateTopology[directory+q.x+pageDims.x*(q.y+pageDims.y*q.z)];}
  candidateTopology[record+1u+ordinal]=physical;}
}
// Phase 12 (slot parallel). Every workset entry is a distinct slot, so the
// former reset sweep and the rediscretization fuse into one owner per slot and
// resolve neighbours through the published mask/rank directory, never a probe.
@compute @workgroup_size(64) fn buildCandidateStencils(@builtin(global_invocation_id) g:vec3u){
 let i=boundedLinearIndex(g);
 for(var l=0u;l<levels();l+=1u){
  if(!stencilDirty(l)||i>=cCount(l)){continue;}
  let s=cWorkSlot(l,i);
  for(var c=DIAG;c<=YZMM;c+=1u){candidateState[cAt(c,l,s)]=0u;}
  let coefficient=f32(1u<<l)*p.reserved.y;
  let q=decode(candidateState[cAt(KEY,l,s)],l);let flags=candidateState[cAt(FLAGS,l,s)];var diagonal=0.0;
  for(var k=0u;k<6u;k+=1u){let targetQ=vec3i(q)+section63Direction(k);if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){continue;}
   let other=cDirectoryLookup(l,vec3u(targetQ));if(other==INVALID){if((flags&GHOST)==0u){diagonal+=coefficient;}continue;}
   diagonal+=coefficient;cStoref(XP+k,l,s,coefficient);}
  if(!(diagonal>1e-20)||!finite(diagonal)){diagonal=1.0;}cStoref(DIAG,l,s,diagonal);
 }
}
var<workgroup> spectralLane:array<f32,64>;
// Phase 13 (one workgroup per level). The published bound is a maximum over the
// level's rows, which is exact and order independent in f32, so the lane-strided
// reduction reproduces the serial scan bit for bit. The M-matrix row validation
// and its report path are retained per row.
@compute @workgroup_size(64) fn publishCandidateSpectralBounds(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let l=wg.x;let live=l<levels()&&stencilDirty(l);var upper=1.0;
 if(live){let n=cCount(l);
  for(var i=lane;i<n;i+=64u){let s=cWorkSlot(l,i);let d=cLoadf(DIAG,l,s);var off=0.0;
   for(var k=0u;k<18u;k+=1u){let c=cLoadf(XP+k,l,s);
    if(!finite(c)||c<0.0){candidateReport(l);}off+=abs(c);}
   if(!finite(d)||!(d>0.0)||!finite(off)||off>d*(1.0+1e-4)){candidateReport(l);}
   else{upper=max(upper,1.0+off/d);}}}
 spectralLane[lane]=upper;workgroupBarrier();
 if(lane==0u&&live){var total=1.0;
  for(var i=0u;i<64u;i+=1u){total=max(total,spectralLane[i]);}
  total*=1.0005;levelDelta[deltaAt(l,7u)]=bitcast<u32>(total);}
}
@compute @workgroup_size(1) fn validateCandidateHierarchy(){
 if(captureFailed()){return;}for(var l=0u;l<levels();l+=1u){
  let spectral=bitcast<f32>(levelDelta[deltaAt(l,7u)]);
  if((levelDirty(l)&&levelDelta[deltaAt(l,4u)]!=0u)
   ||(stencilDirty(l)&&(!(spectral>0.0)||!finite(spectral)))
   ||(topologyDirty(l)&&(levelDelta[deltaAt(l,6u)]!=captureGeneration()
    ||cCount(l)>levelCapacity(l)||candidateDispatch[l*DISPATCH_WORDS+8u]>levelCapacity(l)))){
   captureReport(OVERFLOW);return;}}
 if(selectedCount(levels()-1u)!=1u){captureReport(OVERFLOW);}
}
fn commitCandidateLevelAt(l:u32,i:u32){
 let topologyChanged=topologyDirty(l);
 let transferChanged=topologyChanged||(l+1u<levels()&&topologyDirty(l+1u));
 if(!topologyChanged&&!stencilDirty(l)&&!transferChanged){return;}
 if(captureFailed()||levelDelta[deltaAt(l,4u)]!=0u){return;}
 if(stencilDirty(l)&&i<levelCapacity(l)){for(var c=DIAG;c<=YZMM;c+=1u){state[at(c,l,i)]=candidateState[cAt(c,l,i)];}}
 if(topologyChanged&&i<levelCapacity(l)){for(var c=0u;c<STATE_CHANNELS;c+=1u){state[at(c,l,i)]=candidateState[cAt(c,l,i)];}
  topology[workBase()+levelBase(l)+i]=candidateTopology[workBase()+levelBase(l)+i];
  topology[rankedSlotsBase()+levelBase(l)+i]=candidateTopology[rankedSlotsBase()+levelBase(l)+i];}
 if(topologyChanged&&i<candidateDispatch[l*DISPATCH_WORDS+8u]){let page=pageRecord(l,i);
  for(var word=0u;word<28u;word+=1u){topology[page+word]=candidateTopology[page+word];}}
 if(topologyChanged&&i<logicalPageCount(l)){let pageIndex=pageDirectoryBase()+pageLevelOffset(l)+i;
  topology[pageIndex]=candidateTopology[pageIndex];}
 if(topologyChanged&&i<p.capacity.x){topology[rowMapBase()+l*p.capacity.x+i]=candidateTopology[rowMapBase()+l*p.capacity.x+i];}
 if(transferChanged&&l+1u<levels()&&i<levelCapacity(l)){topology[parentHeadBase(l)+i]=candidateTopology[parentHeadBase(l)+i];
  topology[parentTailBase(l)+i]=candidateTopology[parentTailBase(l)+i];
  topology[fineHeadBase(l)+i]=candidateTopology[fineHeadBase(l)+i];
  topology[fineCountBase(l)+i]=candidateTopology[fineCountBase(l)+i];}
 if(transferChanged&&l+1u<levels()&&i<transferCapacity(l)){for(var w=0u;w<4u;w+=1u){
  topology[transferWord(l,i,w)]=candidateTopology[transferWord(l,i,w)];}}
 if(topologyChanged&&i<brickCount(l)){let record=directoryBase()+16u+(brickLevelOffset(l)+i)*4u;
  for(var w=0u;w<4u;w+=1u){topology[record+w]=candidateTopology[record+w];}}
 if(i==0u){if(stencilDirty(l)){state[at(SPECTRAL,l,0u)]=levelDelta[deltaAt(l,7u)];}
  if(topologyChanged){dispatchMeta[l*DISPATCH_WORDS]=candidateDispatch[l*DISPATCH_WORDS];
   dispatchMeta[l*DISPATCH_WORDS+8u]=candidateDispatch[l*DISPATCH_WORDS+8u];
   topology[directoryBase()+2u+l]=candidateTopology[directoryBase()+2u+l];
   levelDelta[deltaAt(l,5u)]=captureGeneration();}
  if(transferChanged&&l+1u<levels()){dispatchMeta[l*DISPATCH_WORDS+1u]=candidateDispatch[l*DISPATCH_WORDS+1u];}
  let blocks=(selectedCount(l)+63u)/64u;dispatchMeta[l*DISPATCH_WORDS+2u]=min(65535u,blocks);
  dispatchMeta[l*DISPATCH_WORDS+3u]=select(1u,(blocks+65534u)/65535u,blocks>0u);dispatchMeta[l*DISPATCH_WORDS+4u]=1u;
  var parentBlocks=0u;if(l+1u<levels()){parentBlocks=(selectedCount(l+1u)+63u)/64u;}
  dispatchMeta[l*DISPATCH_WORDS+5u]=min(65535u,parentBlocks);
  dispatchMeta[l*DISPATCH_WORDS+6u]=select(1u,(parentBlocks+65534u)/65535u,parentBlocks>0u);
  dispatchMeta[l*DISPATCH_WORDS+7u]=1u;
  let pages=dispatchMeta[l*DISPATCH_WORDS+8u];dispatchMeta[l*DISPATCH_WORDS+9u]=pages;
  dispatchMeta[l*DISPATCH_WORDS+10u]=1u;dispatchMeta[l*DISPATCH_WORDS+11u]=1u;}
}
@compute @workgroup_size(64) fn commitCandidateLevels(@builtin(global_invocation_id) g:vec3u){
 let i=boundedLinearIndex(g);for(var l=0u;l<levels();l+=1u){commitCandidateLevelAt(l,i);}
}
@compute @workgroup_size(1) fn finalizeLifecycle(){let base=lifecycleBase();if(atomicLoad(&control[0])==0u&&!captureFailed()){
 dispatchMeta[base]=1u;dispatchMeta[base+1u]=rows();return;}
 // Candidate failure leaves the previously published hierarchy byte-for-byte
 // authoritative. No recovery or in-place partial publication exists.
}
// Fingerprint the exact inputs the accepted hierarchy consumed, under the same
// gate finalizeLifecycle used. A rejected epoch clears the valid word instead,
// so the next unchanged-input probe can never retire dirty flags against a
// hierarchy that was never published.
@compute @workgroup_size(64) fn publishCommittedInputs(@builtin(global_invocation_id) g:vec3u){
 let accepted=atomicLoad(&control[0])==0u&&!captureFailed();
 let r=rowIndex(g);
 if(r==0u){committedInputs[COMMITTED_VALID]=select(0u,1u,accepted);
  committedInputs[COMMITTED_ROWS]=select(0u,rows(),accepted);}
 if(!accepted||r>=rows()){return;}
 let h=geometry(r);let m=topologyMetrics[r];let base=committedRowBase(r);
 committedInputs[base]=h.x;committedInputs[base+1u]=h.y;
 committedInputs[base+2u]=m.caseId;committedInputs[base+3u]=m.transformAndFlags&63u;
}
@compute @workgroup_size(64) fn zeroVectors(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||stopped()){return;}let s=workSlot(l,i);
 for(var c=RHS;c<=B;c+=1u){storef(c,l,s,0.0);}}
@compute @workgroup_size(64) fn seedRhs(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let v=inputRhs[r];let native=firstTrailingBit(sourceRowGeometry(r).y);
 if(!finite(v)){reportAt(NONFINITE,73u,r);}else{storef(RHS,native,rowMap(native,r),v);}}}
fn stencilDirection(k:u32)->vec3i{let d=array<vec3i,18>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[k];}
fn applied(slot:u32,source:u32)->f32{let l=level();var value=loadf(DIAG,l,slot)*loadf(source,l,slot);let q=decode(state[at(KEY,l,slot)],l);
 for(var k=0u;k<18u;k+=1u){let c=loadf(XP+k,l,slot);if(c==0.0){continue;}let neighborCoord=vec3i(q)+stencilDirection(k);
  if(any(neighborCoord<vec3i(0))||any(neighborCoord>=vec3i(dims(l)))){reportAt(OVERFLOW,74u,slot);continue;}let other=find(l,vec3u(neighborCoord));
  if(other==INVALID){reportAt(OVERFLOW,75u,slot);continue;}value-=c*loadf(source,l,other);}return value;}
fn smoothable(l:u32,s:u32)->bool{return(state[at(FLAGS,l,s)]&GHOST)==0u;}
fn chebyshevWeight(l:u32,phase:u32,degree:u32)->f32{
 let upper=loadf(SPECTRAL,l,0u);let lower=upper/30.0;
 if(!(lower>0.0)||!(upper>lower)||!finite(upper)){reportAt(NONPOSITIVE,76u,l);return 0.0;}
 let centre=0.5*(upper+lower);let radius=0.5*(upper-lower);
 return 1.0/(centre-radius*cos(3.141592653589793*(2.0*f32(phase)+1.0)/(2.0*f32(degree))));
}
fn pageInteriorHaloIndex(local:u32)->u32{let x=local%PAGE_X;let yz=local/PAGE_X;
 return x+1u+HALO_X*((yz%PAGE_Y)+1u+HALO_Y*((yz/PAGE_Y)+1u));}
fn pageAppliedA(l:u32,slot:u32,origin:vec3u,halo:u32)->f32{
 var value=pageDiagonal[halo]*pageA[halo];
 for(var k=0u;k<18u;k+=1u){let c=loadf(XP+k,l,slot);if(c==0.0){continue;}
  let relative=vec3i(decode(state[at(KEY,l,slot)],l))+stencilDirection(k)-vec3i(origin)+vec3i(1);
  if(any(relative<vec3i(0))||any(relative>=vec3i(i32(HALO_X),i32(HALO_Y),i32(HALO_Z)))){reportAt(OVERFLOW,77u,slot);continue;}
  let neighbourHalo=u32(relative.x)+HALO_X*(u32(relative.y)+HALO_Y*u32(relative.z));
  if(pageSlots[neighbourHalo]==INVALID){reportAt(OVERFLOW,78u,slot);continue;}value-=c*pageA[neighbourHalo];}
 return value;
}
fn pageAppliedB(l:u32,slot:u32,origin:vec3u,halo:u32)->f32{
 var value=pageDiagonal[halo]*pageB[halo];
 for(var k=0u;k<18u;k+=1u){let c=loadf(XP+k,l,slot);if(c==0.0){continue;}
  let relative=vec3i(decode(state[at(KEY,l,slot)],l))+stencilDirection(k)-vec3i(origin)+vec3i(1);
  if(any(relative<vec3i(0))||any(relative>=vec3i(i32(HALO_X),i32(HALO_Y),i32(HALO_Z)))){reportAt(OVERFLOW,77u,slot);continue;}
  let neighbourHalo=u32(relative.x)+HALO_X*(u32(relative.y)+HALO_Y*u32(relative.z));
  if(pageSlots[neighbourHalo]==INVALID){reportAt(OVERFLOW,78u,slot);continue;}value-=c*pageB[neighbourHalo];}
 return value;
}
fn pageSweepAtoB(l:u32,origin:vec3u,lid:u32,phase:u32,degree:u32){
 let weight=chebyshevWeight(l,phase,degree);for(var local=lid;local<PAGE_ELEMENTS;local+=128u){let halo=pageInteriorHaloIndex(local);
  let slot=pageSlots[halo];if(slot==INVALID){continue;}let source=pageA[halo];if(!smoothable(l,slot)){pageB[halo]=source;continue;}
  let d=pageDiagonal[halo];if(!(d>0.0)){reportAt(NONPOSITIVE,79u,slot);pageB[halo]=source;continue;}
  let next=source+weight*(pageRhs[halo]-pageAppliedA(l,slot,origin,halo))/d;
  if(!finite(next)){reportAt(NONFINITE,80u,slot);pageB[halo]=source;}else{pageB[halo]=next;}}
}
fn pageSweepBtoA(l:u32,origin:vec3u,lid:u32,phase:u32,degree:u32){
 let weight=chebyshevWeight(l,phase,degree);for(var local=lid;local<PAGE_ELEMENTS;local+=128u){let halo=pageInteriorHaloIndex(local);
  let slot=pageSlots[halo];if(slot==INVALID){continue;}let source=pageB[halo];if(!smoothable(l,slot)){pageA[halo]=source;continue;}
  let d=pageDiagonal[halo];if(!(d>0.0)){reportAt(NONPOSITIVE,79u,slot);pageA[halo]=source;continue;}
  let next=source+weight*(pageRhs[halo]-pageAppliedB(l,slot,origin,halo))/d;
  if(!finite(next)){reportAt(NONFINITE,80u,slot);pageA[halo]=source;}else{pageA[halo]=next;}}
}
fn pagePhase(step:u32,degree:u32,reverse:bool)->u32{return select(step,degree-1u-step,reverse);}
fn smoothPage(reverse:bool,page:u32,lid:u32){let l=level();let pageLive=page<pageCount(l)&&!stopped();var origin=vec3u(0u);
 if(pageLive){origin=decode(pageKey(l,page),l);}let d=dims(l);
 for(var halo=lid;halo<HALO_ELEMENTS;halo+=128u){let x=halo%HALO_X;let yz=halo/HALO_X;
  let relative=vec3i(i32(x)-1,i32(yz%HALO_Y)-1,i32(yz/HALO_Y)-1);let q=vec3i(origin)+relative;var slot=INVALID;
  if(pageLive&&all(q>=vec3i(0))&&all(q<vec3i(d))){slot=pageSlot(l,page,origin,vec3u(q));}pageSlots[halo]=slot;
  var value=0.0;var rhs=0.0;var diagonal=1.0;if(slot!=INVALID){value=loadf(A,l,slot);rhs=loadf(RHS,l,slot);diagonal=loadf(DIAG,l,slot);}
  pageA[halo]=value;pageB[halo]=value;pageRhs[halo]=rhs;pageDiagonal[halo]=diagonal;}
 workgroupBarrier();
 pageSweepAtoB(l,origin,lid,pagePhase(0u,p.solve.x,reverse),p.solve.x);workgroupBarrier();
 pageSweepBtoA(l,origin,lid,pagePhase(1u,p.solve.x,reverse),p.solve.x);workgroupBarrier();
 if(p.solve.x==4u){pageSweepAtoB(l,origin,lid,pagePhase(2u,p.solve.x,reverse),p.solve.x);}workgroupBarrier();
 if(p.solve.x==4u){pageSweepBtoA(l,origin,lid,pagePhase(3u,p.solve.x,reverse),p.solve.x);}workgroupBarrier();
 if(pageLive){for(var local=lid;local<PAGE_ELEMENTS;local+=128u){let halo=pageInteriorHaloIndex(local);let slot=pageSlots[halo];
   if(slot!=INVALID){storef(A,l,slot,pageA[halo]);}}}
}
@compute @workgroup_size(128) fn smoothPageChebyshevForward(@builtin(workgroup_id) page:vec3u,@builtin(local_invocation_index) lid:u32){
 smoothPage(false,page.x,lid);
}
@compute @workgroup_size(128) fn smoothPageChebyshevReverse(@builtin(workgroup_id) page:vec3u,@builtin(local_invocation_index) lid:u32){
 smoothPage(true,page.x,lid);
}
// Return the immutable transfer target owned by one fine slot/corner.
// Restriction consumes the same records through its parent-owned chains, so
// E and E^T cannot diverge.
fn correctionTransfer(l:u32,fine:u32,corner:u32)->TransferTarget{
 if(l+1u>=levels()||fine>=levelCapacity(l)){reportAt(OVERFLOW,81u,fine);return TransferTarget(INVALID,0.0);}
 let first=topology[fineHeadBase(l)+fine];let n=topology[fineCountBase(l)+fine];
 if(first==INVALID||corner>=n||first+corner>=transferCount(l)){reportAt(OVERFLOW,82u,fine);return TransferTarget(INVALID,0.0);}
 let record=first+corner;if(topology[transferWord(l,record,0u)]!=fine){reportAt(OVERFLOW,83u,fine);return TransferTarget(INVALID,0.0);}
 let coarse=topology[transferWord(l,record,1u)];let weight=bitcast<f32>(topology[transferWord(l,record,2u)]);
 if(coarse>=levelCapacity(l+1u)||!finite(weight)){reportAt(OVERFLOW,84u,fine);return TransferTarget(INVALID,0.0);}
 return TransferTarget(coarse,weight);}
// Section 4.2 GhostValueAccumulate is E^T: one coarse owner traverses its
// immutable fine-major chain, so no destination synchronization is required.
@compute @workgroup_size(64) fn restrictAndGhostAccumulate(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();
 if(l+1u>=levels()||i>=count(l+1u)||stopped()){return;}let coarse=workSlot(l+1u,i);var sum=0.0;var record=topology[parentHeadBase(l)+coarse];
 for(var visited=0u;record!=INVALID&&visited<transferCount(l);visited+=1u){if(record>=transferCount(l)){reportAt(OVERFLOW,85u,coarse);return;}
  let fine=topology[transferWord(l,record,0u)];let parent=topology[transferWord(l,record,1u)];if(parent!=coarse){reportAt(OVERFLOW,86u,coarse);return;}
  let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;let product=applied(fine,A);let residual=select(-product,loadf(RHS,l,fine)-product,!ghost);
  sum+=bitcast<f32>(topology[transferWord(l,record,2u)])*residual;record=topology[transferWord(l,record,3u)];}
 if(record!=INVALID||!finite(sum)){reportAt(OVERFLOW,87u,coarse);return;}storef(RHS,l+1u,coarse,sum);}
@compute @workgroup_size(64) fn exactBottom(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>0u||stopped()){return;}if(count(l)!=1u){reportAt(NONPOSITIVE,88u,l);return;}
 let s=workSlot(l,0u);let d=loadf(DIAG,l,s);if(!(d>0.0)){reportAt(NONPOSITIVE,89u,s);return;}let x=loadf(RHS,l,s)/d;if(!finite(x)){reportAt(NONFINITE,90u,s);}else{storef(A,l,s,x);}}
// One fine invocation owns the complete interpolation sum, deleting all
// prolongation atomics. GhostValuePropagate is the unit-copy branch of the
// same E mapping rather than a second dispatch.
@compute @workgroup_size(64) fn prolongAndGhostPropagate(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||stopped()){return;}
 let fine=workSlot(l,i);let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;let targetCount=select(8u,1u,ghost);var value=select(loadf(A,l,fine),0.0,ghost);
 for(var corner=0u;corner<targetCount;corner+=1u){let transfer=correctionTransfer(l,fine,corner);if(transfer.coarse==INVALID){return;}
  value+=transfer.weight*loadf(A,l+1u,transfer.coarse);}if(!finite(value)){reportAt(NONFINITE,91u,fine);}else{storef(A,l,fine,value);}}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let native=firstTrailingBit(sourceRowGeometry(r).y);
 let v=loadf(A,native,rowMap(native,r));if(!finite(v)){reportAt(NONFINITE,92u,r);}else{outputCorrection[r]=v;}}}
`;
