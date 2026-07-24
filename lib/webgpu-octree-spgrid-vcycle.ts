import {
  OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY,
  OCTREE_PERSISTENT_MGPCG_STATE_CHANNELS,
  type OctreeFirstOrderSPDVCycle,
  type OctreePersistentMGPCGExecutor,
} from "./webgpu-octree-mgpcg";
import { PassBroker } from "./webgpu-pass-broker";

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
  /** Eight-word exact affected-row/publication record per sparse level. */
  readonly levelDeltaBytes: number;
  readonly brickCount: number;
  readonly directoryBytes: number;
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
  readonly damping?: number;
}

export interface OctreeSPGridL1DeltaSource {
  /** Exact old/new row transaction published by the leaf-frontier producer. */
  readonly rows: GPUBuffer;
  readonly rowCapacity: number;
  readonly controlOffsetWords: number;
  readonly newToOldOffsetWords: number;
  readonly dirtyRowsOffsetWords: number;
}

export interface OctreeSPGridVCycleSource {
  readonly leafHeaders: GPUBuffer;
  readonly leafEntries: GPUBuffer;
  readonly rowDelta: OctreeSPGridL1DeltaSource;
}

// Key/class/diagonal, six Cartesian and twelve octree-edge coefficients, three
// vectors, and the adaptive owner of active/ghost storage. Resolved stencil
// slots live in the topology arena so neither portable storage binding exceeds
// 128 MiB at the 24x18x16 UI capacity.
const STATE_CHANNELS = 25;
const TOPOLOGY_HEADER_WORDS = 16;
const DISPATCH_RECORD_BYTES_PER_LEVEL = 32;
// Only valid + published-row-count remain. The deleted six-word tail encoded
// the former full-capacity recovery dispatch.
const DISPATCH_LIFECYCLE_BYTES = 8;
const CAPTURE_PAGE_ROWS = 64;
const CAPTURE_PUBLICATION_WORDS = 12;
const LEVEL_DELTA_WORDS = 8;
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
  // Four immutable words per transfer (fine, coarse, weight, next), followed
  // by parent head/tail slots for deterministic restriction and fine
  // head/count slots for direct indexed prolongation.
  const transferWords = (levelCount - 1) * (transferStride * 4 + 4 * levelStride);
  let brickCount = 0;
  for (let level = 0; level < levelCount; level += 1) {
    const scale = 2 ** level;
    brickCount += options.dimensions.map((value) => Math.ceil(Math.ceil(value / scale) / 4))
      .reduce((product, value) => product * value, 1);
  }
  // Sixteen publication words, four words per 4^3 brick (generation, two
  // occupancy masks, ranked base), and one compact ranked slot vector/level.
  const directoryWords = 16 + 4 * brickCount + levelCount * levelStride;
  const stencilNeighbourWords = 18 * levelCount * levelStride;
  const directoryBytes = directoryWords * 4;
  const topologyBytes = (TOPOLOGY_HEADER_WORDS + rowMapWords + worklistWords
    + transferWords + directoryWords + stencilNeighbourWords) * 4;
  const stateBytes = STATE_CHANNELS * levelCount * levelStride * 4;
  const dispatchBytes = levelCount * DISPATCH_RECORD_BYTES_PER_LEVEL + DISPATCH_LIFECYCLE_BYTES;
  const levelDeltaBytes = levelCount * LEVEL_DELTA_WORDS * 4;
  const capturePageCount = Math.ceil(rowCapacity / CAPTURE_PAGE_ROWS);
  const capturePageStateBytes = (CAPTURE_PUBLICATION_WORDS + 4 * capturePageCount) * 4;
  return { rowCapacity, levelCount, levelStride, transferStride, topologyBytes, stateBytes, dispatchBytes,
    // Storage-authored counts and indirect arguments must be separate WebGPU
    // buffers; the final term is the per-level uniform parameter storage.
    allocatedBytes: topologyBytes + stateBytes + 2 * dispatchBytes
      + capturePageStateBytes + levelDeltaBytes + levelCount * 64,
    levelDeltaBytes, brickCount, directoryBytes,
    capturePageStateBytes, capturePageCount,
    rowDispatch: dispatchFor(rowCapacity), slotDispatch: dispatchFor(levelStride), transferDispatch: dispatchFor(transferStride),
    brickDispatch: dispatchFor(brickCount) };
}

export type OctreeSPGridVCyclePipelineName = "beginL1CapturePlan"
  | "planL1CaptureDelta" | "commitChangedL1" | "finalizeL1CapturePublication"
  | "buildCandidateLevelDeltas" | "validateCandidateHierarchy" | "commitCandidateLevels"
  | "finalizeLifecycle" | "clearCorrection"
  | "zeroVectors" | "seedRhs" | "smoothAtoB" | "smoothBtoA"
  | "restrictAndGhostAccumulate" | "exactBottom"
  | "prolongAndGhostPropagate" | "publish";

export const OCTREE_SPGRID_VCYCLE_BINDINGS: Readonly<Record<OctreeSPGridVCyclePipelineName, readonly number[]>> = Object.freeze({
  beginL1CapturePlan: [0, 3, 6, 13, 14, 18],
  planL1CaptureDelta: [0, 1, 2, 3, 11, 12, 13, 14, 18],
  commitChangedL1: [0, 1, 2, 3, 11, 12, 13, 18],
  finalizeL1CapturePublication: [0, 3, 13, 18],
  buildCandidateLevelDeltas: [0, 3, 4, 5, 6, 11, 12, 14, 15, 16, 17],
  validateCandidateHierarchy: [0, 6, 7, 13, 14, 17],
  commitCandidateLevels: [0, 4, 5, 6, 13, 14, 15, 16, 17],
  finalizeLifecycle: [0, 3, 6, 7, 13],
  clearCorrection: [0, 3, 7, 9],
  zeroVectors: [0, 4, 5, 6, 7], seedRhs: [0, 1, 3, 4, 5, 7, 8],
  smoothAtoB: [0, 4, 5, 6, 7], smoothBtoA: [0, 4, 5, 6, 7],
  restrictAndGhostAccumulate: [0, 4, 5, 6, 7],
  exactBottom: [0, 4, 5, 6, 7],
  prolongAndGhostPropagate: [0, 4, 5, 6, 7],
  publish: [0, 1, 3, 4, 5, 7, 9],
});

type CachedGroup = { rowCount: GPUBuffer; control: GPUBuffer; rhs?: GPUBuffer; correction?: GPUBuffer; group: GPUBindGroup };
type PersistentSolveInput = Parameters<OctreePersistentMGPCGExecutor["encodeSolve"]>[1];
type CachedPersistentGroup = {
  readonly input: PersistentSolveInput;
  readonly group: GPUBindGroup;
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
  readonly plan: OctreeSPGridVCyclePlan;
  readonly allocatedBytes: number;
  readonly encodedPassTransitionCount = 1;
  readonly encodedCorrectionPassTransitionCount = 1;
  readonly encodedSetupDispatchCount: number;
  readonly encodedCorrectionDispatchCount: number;
  readonly persistentMGPCG?: OctreePersistentMGPCGExecutor;
  readonly diagnostics: Readonly<{ levelCount: number; coarsestCapacity: number; maximumTransferRecordsPerLevel: number;
    correctionDispatchCount: number; correctionPassTransitions: number; restrictionScatterDispatchCount: number;
    restrictionAtomicAddUpperBound: number; parentGatherDispatchCount: number; parentGatherAtomicAddCount: 0;
    bottomOperation: "exact-single-cell"; coarsestDegreesOfFreedom: 1;
    directoryLookup: "brick-mask-rank"; lookupProbeUpperBound: 1; directoryBytes: number;
    directoryBrickCount: number; directoryBuildDispatchCount: number }>;
  private readonly capturedHeaders: GPUBuffer;
  private readonly capturedEntries: GPUBuffer;
  private readonly topology: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly dispatchMeta: GPUBuffer;
  private readonly indirectDispatch: GPUBuffer;
  private readonly capturePageState: GPUBuffer;
  private readonly levelDelta: GPUBuffer;
  private readonly candidateTopology: GPUBuffer;
  private readonly candidateState: GPUBuffer;
  private readonly candidateDispatch: GPUBuffer;
  private readonly params: readonly GPUBuffer[];
  private readonly pipelines: Readonly<Record<OctreeSPGridVCyclePipelineName, GPUComputePipeline>>;
  private readonly groups = new Map<string, CachedGroup>();
  private readonly pre: number;
  private readonly post: number;
  private readonly dimensions: readonly [number, number, number];
  private readonly persistentState?: GPUBuffer;
  private readonly persistentParams?: GPUBuffer;
  private readonly persistentPipeline?: GPUComputePipeline;
  private persistentGroup?: CachedPersistentGroup;
  private persistentConfiguration?: readonly [number, number];
  private lastSetupInput?: { readonly solverControl: GPUBuffer; readonly rowCount: GPUBuffer };
  private preparedCaptureRowCount?: GPUBuffer;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly source: OctreeSPGridVCycleSource,
    options: OctreeSPGridVCycleOptions) {
    this.plan = planOctreeSPGridVCycle(options);
    this.dimensions = options.dimensions;
    if (!(options.finestCellWidth > 0) || !Number.isFinite(options.finestCellWidth)) throw new RangeError("SPGrid finest cell width must be positive");
    if (source.leafHeaders.size < this.plan.rowCapacity * 48 || source.leafEntries.size < 8) throw new RangeError("SPGrid L1 source capacity is too small");
    const captureUsage = GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE;
    if ((source.leafHeaders.usage & captureUsage) !== captureUsage
      || (source.leafEntries.usage & captureUsage) !== captureUsage) {
      throw new RangeError("SPGrid L1 source buffers require COPY_SRC and STORAGE capture usage");
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
    // COPY_SRC supports bounded lifecycle verification and preserves the
    // existing cold capture path; it does not add storage or bind stages.
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.capturedHeaders = device.createBuffer({ label: "SPGrid captured L1 headers", size: this.plan.rowCapacity * 48, usage: storage });
    this.capturedEntries = device.createBuffer({ label: "SPGrid captured L1 entries", size: source.leafEntries.size, usage: storage });
    this.topology = device.createBuffer({ label: "SPGrid native sparse topology/worklists/transfers", size: this.plan.topologyBytes, usage: storage });
    this.state = device.createBuffer({ label: "SPGrid six-face stencils and vectors", size: this.plan.stateBytes, usage: storage });
    this.dispatchMeta = device.createBuffer({ label: "SPGrid worklist counts and published dispatches", size: this.plan.dispatchBytes,
      usage: storage | GPUBufferUsage.COPY_SRC });
    this.indirectDispatch = device.createBuffer({ label: "SPGrid live indirect dispatches", size: this.plan.dispatchBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
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
    const damping = Math.max(0.05, Math.min(0.95, options.damping ?? 2 / 3));
    this.params = Object.freeze(Array.from({ length: this.plan.levelCount }, (_, level) => {
      const buffer = device.createBuffer({ label: `SPGrid level ${level} parameters`, size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(20), floats = new Float32Array(words.buffer);
      words.set([options.dimensions[0], options.dimensions[1], options.dimensions[2], level,
        this.plan.rowCapacity, this.plan.levelCount, this.plan.levelStride, this.plan.transferStride,
        this.plan.rowDispatch[0], this.plan.slotDispatch[0], this.plan.transferDispatch[0], this.pre]);
      words[12] = this.post; words[13] = 1; floats[14] = damping; floats[15] = options.finestCellWidth;
      words.set([delta.rowCapacity, delta.controlOffsetWords, delta.newToOldOffsetWords,
        delta.dirtyRowsOffsetWords], 16);
      device.queue.writeBuffer(buffer, 0, words); return buffer;
    }));
    const shaderModule = device.createShaderModule({ label: "Paper native sparse SPGrid V-cycle", code: octreeSPGridVCycleShader });
    const make = (entryPoint: OctreeSPGridVCyclePipelineName) => device.createComputePipeline({ label: `SPGrid V-cycle · ${entryPoint}`,
      layout: "auto", compute: { module: shaderModule, entryPoint } });
    this.pipelines = Object.freeze(Object.fromEntries((Object.keys(OCTREE_SPGRID_VCYCLE_BINDINGS) as OctreeSPGridVCyclePipelineName[]).map((name) => [name, make(name)])) as Record<OctreeSPGridVCyclePipelineName, GPUComputePipeline>);
    if (this.plan.rowCapacity <= OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY) {
      this.persistentState = device.createBuffer({ label: "SPGrid persistent MGPCG packed state",
        size: this.plan.rowCapacity * OCTREE_PERSISTENT_MGPCG_STATE_CHANNELS * 4, usage: storage });
      this.persistentParams = device.createBuffer({ label: "SPGrid persistent MGPCG parameters", size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const persistentModule = device.createShaderModule({ label: "SPGrid persistent small-domain MGPCG",
        code: octreeSPGridPersistentMGPCGShader });
      this.persistentPipeline = device.createComputePipeline({ label: "SPGrid persistent small-domain MGPCG",
        layout: "auto", compute: { module: persistentModule, entryPoint: "persistentMGPCG" } });
      this.persistentMGPCG = Object.freeze({
        maximumRowCapacity: this.plan.rowCapacity,
        encodedDispatchCount: 1 as const,
        dispatchShape: [1, 1, 1] as const,
        invariantProof: Object.freeze({ ghostRows: "spgrid-identical" as const,
          transfers: "validated-adjoint-pair" as const,
          invalidRows: "uniform-fail-closed-before-arithmetic" as const }),
        encodeSolve: (broker: PassBroker, input: PersistentSolveInput) => this.encodePersistentSolve(broker, input),
      });
    }
    const l = this.plan.levelCount;
    // Two exact L1-capture dispatches plus one ordered singleton builder and
    // five transactional publication dispatches. This is the complete encoded
    // setup schedule consumed by MGPCG telemetry.
    this.encodedSetupDispatchCount = 8;
    // Each Jacobi sweep evaluates its stencil and publishes the opposite
    // ping/pong vector in one dispatch.  The final pre-smoothing stencil
    // evaluation is likewise consumed directly by restriction.  These are
    // row-local fusions: the dispatch boundary between successive ping/pong
    // sweeps (and therefore the symmetric V-cycle operator) is unchanged.
    this.encodedCorrectionDispatchCount = l + 4 + (l - 1) * (this.pre + this.post + 2);
    this.diagnostics = Object.freeze({ levelCount: l, coarsestCapacity: this.plan.levelStride,
      maximumTransferRecordsPerLevel: this.plan.transferStride, correctionDispatchCount: this.encodedCorrectionDispatchCount,
      correctionPassTransitions: 1, restrictionScatterDispatchCount: 0,
      restrictionAtomicAddUpperBound: 0,
      parentGatherDispatchCount: l - 1, parentGatherAtomicAddCount: 0 as const,
      bottomOperation: "exact-single-cell" as const, coarsestDegreesOfFreedom: 1 as const,
      directoryLookup: "brick-mask-rank" as const, lookupProbeUpperBound: 1 as const,
      directoryBytes: this.plan.directoryBytes, directoryBrickCount: this.plan.brickCount,
      directoryBuildDispatchCount: 1 });
    this.allocatedBytes = this.plan.allocatedBytes + this.capturedHeaders.size + this.capturedEntries.size
      + this.candidateTopology.size + this.candidateState.size + this.candidateDispatch.size
      + (this.persistentState?.size ?? 0) + (this.persistentParams?.size ?? 0);
  }

  encodeCapture(broker: PassBroker): void {
    this.assertLive();
    // The first capture is deferred until encodeSetup supplies the authoritative
    // row-count buffer. There is deliberately no capacity-copy bootstrap.
    if (!this.lastSetupInput) return;
    this.encodeCaptureDelta(broker, this.lastSetupInput);
  }

  private encodeCaptureDelta(broker: PassBroker,
    input: { readonly solverControl: GPUBuffer; readonly rowCount: GPUBuffer }): void {
    const pass = broker.compute({ label: "SPGrid V-cycle · select setup delta and capture changed L1" });
    this.run(pass, "beginL1CapturePlan", 0, input, [1, 1, 1]);
    this.run(pass, "planL1CaptureDelta", 0, input, [1, 1, 1]);
    this.preparedCaptureRowCount = input.rowCount;
  }

  encodeSetup(broker: PassBroker, input: { solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive();
    if (this.preparedCaptureRowCount !== input.rowCount) this.encodeCaptureDelta(broker, input);
    const pass = broker.compute({ label: "SPGrid V-cycle · build and publish exact level deltas" });
    this.run(pass, "buildCandidateLevelDeltas", 0, input, [1, 1, 1]);
    this.run(pass, "validateCandidateHierarchy", 0, input, [1, 1, 1]);
    this.run(pass, "commitChangedL1", 0, input, [1, 1, 1]);
    this.run(pass, "finalizeL1CapturePublication", 0, input, [1, 1, 1]);
    this.run(pass, "commitCandidateLevels", 0, input,
      dispatchFor(Math.max(this.plan.levelStride, this.plan.brickCount)));
    this.run(pass, "finalizeLifecycle", 0, input, [1, 1, 1]);
    // dispatchMeta remains STORAGE-only inside compute passes. Copying its
    // finalized records after the setup boundary gives correction a distinct
    // INDIRECT-only source and avoids a whole-pass storage/indirect conflict.
    broker.updateIndirectBuffer(this.dispatchMeta, 0, this.indirectDispatch, 0, this.plan.dispatchBytes);
    this.lastSetupInput = input;
    this.preparedCaptureRowCount = undefined;
  }

  encodeCorrection(broker: PassBroker, input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive();
    const pass = broker.compute({ label: "SPGrid V-cycle · one-pass symmetric correction" });
    this.run(pass, "clearCorrection", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount; level += 1) this.runIndirect(pass, "zeroVectors", level, input, false);
    this.run(pass, "seedRhs", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount - 1; level += 1) {
      this.smooth(pass, level, this.pre, input);
      this.runIndirect(pass, "restrictAndGhostAccumulate", level, input, true);
    }
    this.runIndirect(pass, "exactBottom", this.plan.levelCount - 1, input, false);
    for (let level = this.plan.levelCount - 2; level >= 0; level -= 1) {
      this.runIndirect(pass, "prolongAndGhostPropagate", level, input, false);
      this.smooth(pass, level, this.post, input);
    }
    this.run(pass, "publish", 0, input, this.plan.rowDispatch);
  }

  private smooth(pass: GPUComputePassEncoder, level: number, iterations: number,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const b = (iteration & 1) !== 0;
      this.runIndirect(pass, b ? "smoothBtoA" : "smoothAtoB", level, input, false);
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
      const buffers = new Map<number, GPUBuffer | undefined>([
        [0, this.params[level]], [1, this.capturedHeaders], [2, this.capturedEntries], [3, input.rowCount],
        [4, this.topology], [5, this.state], [6, this.dispatchMeta], [7, input.solverControl],
        [8, input.rhs], [9, input.correction], [11, this.source.leafHeaders], [12, this.source.leafEntries],
        [13, this.capturePageState], [14, this.levelDelta], [15, this.candidateTopology],
        [16, this.candidateState], [17, this.candidateDispatch], [18, this.source.rowDelta.rows],
      ]);
      group = this.device.createBindGroup({ label: `SPGrid V-cycle · ${name} · level ${level}`,
        layout: pipeline.getBindGroupLayout(0), entries: OCTREE_SPGRID_VCYCLE_BINDINGS[name].map((binding) => ({
        binding, resource: { buffer: buffers.get(binding)! },
      })) });
      this.groups.set(key, { rowCount: input.rowCount, control: input.solverControl, rhs: input.rhs, correction: input.correction, group });
    }
    pass.setPipeline(pipeline); pass.setBindGroup(0, group!);
  }
  private encodePersistentSolve(broker: PassBroker, input: PersistentSolveInput): void {
    this.assertLive();
    if (!this.persistentState || !this.persistentParams || !this.persistentPipeline) {
      throw new Error("persistent MGPCG is unavailable for this immutable row capacity");
    }
    const configuration = [input.boundarySmoothingIterations, input.relativeTolerance] as const;
    if (this.persistentConfiguration === undefined) {
      const words = new Uint32Array(16), floats = new Float32Array(words.buffer);
      words.set([this.dimensions[0], this.dimensions[1], this.dimensions[2], this.plan.rowCapacity,
        this.plan.levelCount, this.plan.levelStride, 0, input.boundarySmoothingIterations]);
      floats[8] = input.relativeTolerance; floats[9] = 1e-30; floats[10] = 2 / 3;
      words[12] = this.pre;
      this.device.queue.writeBuffer(this.persistentParams, 0, words);
      this.persistentConfiguration = configuration;
    } else if (this.persistentConfiguration[0] !== configuration[0]
      || this.persistentConfiguration[1] !== configuration[1]) {
      throw new Error("persistent MGPCG solver configuration is immutable after first encode");
    }
    const previous = this.persistentGroup?.input;
    if (!previous || previous.leafHeaders !== input.leafHeaders || previous.leafEntries !== input.leafEntries
      || previous.rowCount !== input.rowCount || previous.pressureIn !== input.pressureIn
      || previous.pressureOut !== input.pressureOut || previous.solverControl !== input.solverControl) {
      // L2 publication replaces coefficients/RHS in the shared live rows but
      // preserves cell/size topology. The persistent outer solve must bind
      // those live L2 rows; its L1 correction reads coefficients exclusively
      // from the captured SPGrid stencil state built during encodeSetup.
      const buffers = [this.persistentParams, input.leafHeaders, input.leafEntries, input.rowCount,
        this.topology, this.state, this.dispatchMeta, input.solverControl, input.pressureIn,
        input.pressureOut, this.persistentState];
      this.persistentGroup = { input, group: this.device.createBindGroup({
        label: "SPGrid persistent MGPCG bindings", layout: this.persistentPipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      }) };
    }
    const pass = broker.compute({ label: "SPGrid persistent small-domain MGPCG" });
    pass.setPipeline(this.persistentPipeline); pass.setBindGroup(0, this.persistentGroup!.group);
    pass.dispatchWorkgroups(1, 1, 1);
  }
  private assertLive(): void { if (this.destroyed) throw new Error("SPGrid V-cycle is destroyed"); }
  destroy(): void {
    if (this.destroyed) return; this.destroyed = true; this.groups.clear();
    this.capturedHeaders.destroy(); this.capturedEntries.destroy(); this.topology.destroy(); this.state.destroy(); this.dispatchMeta.destroy();
    this.indirectDispatch.destroy();
    this.capturePageState.destroy(); this.levelDelta.destroy();
    this.candidateTopology.destroy(); this.candidateState.destroy(); this.candidateDispatch.destroy();
    this.persistentState?.destroy(); this.persistentParams?.destroy(); this.persistentGroup = undefined;
    for (const buffer of this.params) buffer.destroy();
  }
}

/** One-workgroup implementation of the complete Section 4.3 MGPCG solve.
 * The SPGrid operator helpers below use the validated relaxJacobi/applied and
 * correctionTransfer bodies verbatim. */
export const octreeSPGridPersistentMGPCGShader = /* wgsl */ `
struct PersistentParams{dimsCapacity:vec4u,hierarchySolve:vec4u,numerics:vec4f,vcycle:vec4u}
struct LeafHeader{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f}
struct LeafEntry{row:u32,coefficient:f32}
struct TransferTarget{coarse:u32,weight:f32}
@group(0) @binding(0) var<uniform> p:PersistentParams;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> entries:array<LeafEntry>;
@group(0) @binding(3) var<storage,read_write> rowCounts:array<u32>;
@group(0) @binding(4) var<storage,read> topology:array<u32>;
@group(0) @binding(5) var<storage,read_write> state:array<u32>;
@group(0) @binding(6) var<storage,read> dispatchMeta:array<u32>;
@group(0) @binding(7) var<storage,read_write> control:array<u32>;
@group(0) @binding(8) var<storage,read> pressureSeed:array<f32>;
@group(0) @binding(9) var<storage,read_write> pressureOut:array<f32>;
@group(0) @binding(10) var<storage,read_write> packed:array<f32>;
const ACTIVE=1u;const GHOST=2u;const INVALID=0xffffffffu;const ROW_BOUNDARY=1u;
const INVALID_ROW_ERROR=1u;const OVERFLOW=2u;const NONFINITE=4u;const NONPOSITIVE=8u;const NONCONVERGENCE=16u;
const PERSISTENT_LANES=64u;
const KEY=0u;const FLAGS=1u;const DIAG=2u;const XP=3u;const XM=4u;const YP=5u;const YM=6u;const ZP=7u;const ZM=8u;
const XYPP=9u;const XYPM=10u;const XYMP=11u;const XYMM=12u;const XZPP=13u;const XZPM=14u;const XZMP=15u;const XZMM=16u;
const YZPP=17u;const YZPM=18u;const YZMP=19u;const YZMM=20u;
const RHS=21u;const A=22u;const B=23u;const OWNER=24u;
const STATE_CHANNELS=25u;
const PX=0u;const PR=1u;const PZ=2u;const PD=3u;const PQ=4u;const HA=5u;const HB=6u;const HRHS=7u;const HCORR=8u;const BAND_A=9u;const BAND_B=10u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn capacity()->u32{return p.dimsCapacity.w;}fn rows()->u32{return min(select(0u,rowCounts[0],arrayLength(&rowCounts)>0u),capacity());}
fn levels()->u32{return p.hierarchySolve.x;}fn stride()->u32{return p.hierarchySolve.y;}
fn boundarySweeps()->u32{return p.hierarchySolve.w;}fn tolerance()->f32{return p.numerics.x;}fn epsilon()->f32{return p.numerics.y;}
fn failed()->bool{return control[0]!=0u;}fn stopped()->bool{return failed()||control[1]!=0u;}
var<private> persistentLane:u32;
var<workgroup> persistentErrors:array<vec4u,64>;
var<workgroup> persistentStop:u32;
fn report(flag:u32){if(persistentErrors[persistentLane].x==0u){persistentErrors[persistentLane]=vec4u(flag,0u,INVALID,0u);}
 else{persistentErrors[persistentLane].x|=flag;}}
fn reportAt(flag:u32,stage:u32,row:u32,value:f32){if(persistentErrors[persistentLane].x==0u){
 persistentErrors[persistentLane]=vec4u(flag,stage,row,bitcast<u32>(value));}else{persistentErrors[persistentLane].x|=flag;}}
fn sync(){storageBarrier();workgroupBarrier();if(persistentLane==0u){var flags=0u;var chosen=INVALID;
 for(var lane=0u;lane<PERSISTENT_LANES;lane+=1u){let record=persistentErrors[lane];flags|=record.x;
  if(record.x!=0u&&(chosen==INVALID||record.y<persistentErrors[chosen].y
   ||(record.y==persistentErrors[chosen].y&&record.z<persistentErrors[chosen].z))){chosen=lane;}}
 control[0]|=flags;if(control[10]==0u&&chosen!=INVALID){let record=persistentErrors[chosen];
  control[10]=record.y;control[11]=record.z;control[12]=record.w;}}storageBarrier();workgroupBarrier();}
fn pv(channel:u32,row:u32)->f32{return packed[channel*capacity()+row];}fn storePV(channel:u32,row:u32,value:f32){packed[channel*capacity()+row]=value;}
fn pu(channel:u32,row:u32)->u32{return bitcast<u32>(pv(channel,row));}fn storePU(channel:u32,row:u32,value:u32){storePV(channel,row,bitcast<f32>(value));}
fn at(channel:u32,l:u32,slot:u32)->u32{return(channel*levels()+l)*stride()+slot;}
fn loadf(channel:u32,l:u32,slot:u32)->f32{return bitcast<f32>(state[at(channel,l,slot)]);}
fn storef(channel:u32,l:u32,slot:u32,value:f32){state[at(channel,l,slot)]=bitcast<u32>(value);}
fn rowMapBase()->u32{return 16u;}fn workBase()->u32{return rowMapBase()+levels()*capacity();}
fn rowMap(l:u32,row:u32)->u32{return topology[rowMapBase()+l*capacity()+row];}
fn workSlot(l:u32,index:u32)->u32{return topology[workBase()+l*stride()+index];}
fn count(l:u32)->u32{return dispatchMeta[l*8u];}
fn dims(l:u32)->vec3u{let scale=1u<<l;return(p.dimsCapacity.xyz+vec3u(scale-1u))/scale;}
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;return vec3u(v%d.x,(v/d.x)%d.y,v/(d.x*d.y));}
fn coordKey(q:vec3u,l:u32)->u32{let d=dims(l);return q.x+d.x*(q.y+d.y*q.z)+1u;}
fn transferBase()->u32{return workBase()+levels()*stride();}fn transferStride()->u32{return capacity()*8u;}
fn transferLevelWords()->u32{return transferStride()*4u+4u*stride();}
fn transferCount(l:u32)->u32{return dispatchMeta[l*8u+1u];}
fn transferWord(l:u32,i:u32,w:u32)->u32{return transferBase()+l*transferLevelWords()+i*4u+w;}
fn parentHeadBase(l:u32)->u32{return transferBase()+l*transferLevelWords()+transferStride()*4u;}
fn fineHeadBase(l:u32)->u32{return parentHeadBase(l)+2u*stride();}
fn fineCountBase(l:u32)->u32{return fineHeadBase(l)+stride();}
fn directoryBase()->u32{return transferBase()+(levels()-1u)*transferLevelWords();}
fn brickDims(l:u32)->vec3u{return(dims(l)+vec3u(3u))/4u;}fn brickCount(l:u32)->u32{let d=brickDims(l);return d.x*d.y*d.z;}
fn brickLevelOffset(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=brickCount(k);}return result;}
fn totalBrickCount()->u32{return brickLevelOffset(levels());}fn rankedSlotsBase()->u32{return directoryBase()+16u+totalBrickCount()*4u;}
fn neighbourBase()->u32{return rankedSlotsBase()+levels()*stride();}
fn neighbourAt(k:u32,l:u32,s:u32)->u32{return neighbourBase()+(k*levels()+l)*stride()+s;}
fn find(l:u32,q:vec3u)->u32{if(l>=levels()||any(q>=dims(l))){return INVALID;}
 let generation=topology[directoryBase()+2u+l];if(generation==0u){report(OVERFLOW);return INVALID;}let d=brickDims(l);let b=q/4u;
 let record=directoryBase()+16u+(brickLevelOffset(l)+b.x+d.x*(b.y+d.y*b.z))*4u;if(topology[record]!=generation){report(OVERFLOW);return INVALID;}
 let local=q&vec3u(3u);let bit=local.x+4u*local.y+16u*local.z;let word=topology[record+1u+(bit>>5u)];if((word&(1u<<(bit&31u)))==0u){return INVALID;}
 let low=topology[record+1u];let high=topology[record+2u];let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);
 var rank=countOneBits(low&lower);if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}let ranked=topology[record+3u]+rank;
 if(ranked>=count(l)||ranked>=stride()){report(OVERFLOW);return INVALID;}let slot=topology[rankedSlotsBase()+l*stride()+ranked];
 if(slot>=stride()||state[at(KEY,l,slot)]!=coordKey(q,l)){report(OVERFLOW);return INVALID;}return slot;}
fn smoothable(l:u32,slot:u32)->bool{return(state[at(FLAGS,l,slot)]&GHOST)==0u;}
// Verbatim staged stencil semantics, with level passed explicitly because one
// persistent invocation traverses the entire pyramid.
fn persistentApplied(l:u32,slot:u32,source:u32)->f32{var value=loadf(DIAG,l,slot)*loadf(source,l,slot);
 for(var k=0u;k<18u;k+=1u){let c=loadf(XP+k,l,slot);if(c==0.0){continue;}
  let other=topology[neighbourAt(k,l,slot)];if(other>=stride()){report(OVERFLOW);continue;}
  value-=c*loadf(source,l,other);}return value;}
fn persistentRelaxJacobi(l:u32,slot:u32,src:u32,dst:u32){if(!smoothable(l,slot)){storef(dst,l,slot,loadf(src,l,slot));return;}
 let d=loadf(DIAG,l,slot);if(!(d>0.0)){report(NONPOSITIVE);return;}let x=loadf(src,l,slot)+p.numerics.z*(loadf(RHS,l,slot)-persistentApplied(l,slot,src))/d;
 if(!finite(x)){report(NONFINITE);}else{storef(dst,l,slot,x);}}
fn correctionTransfer(l:u32,fine:u32,corner:u32)->TransferTarget{
 if(l+1u>=levels()||fine>=stride()){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
 let first=topology[fineHeadBase(l)+fine];let n=topology[fineCountBase(l)+fine];
 if(first==INVALID||corner>=n||first+corner>=transferCount(l)){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
 let record=first+corner;if(topology[transferWord(l,record,0u)]!=fine){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
 let coarse=topology[transferWord(l,record,1u)];let weight=bitcast<f32>(topology[transferWord(l,record,2u)]);
 if(coarse>=stride()||!finite(weight)){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
 return TransferTarget(coarse,weight);}
fn l2Value(row:u32,channel:u32)->f32{return pv(channel,row);}
fn applyL2(row:u32,channel:u32)->f32{let h=headers[row];var value=h.diagonal*l2Value(row,channel);for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];value-=e.coefficient*l2Value(e.row,channel);}return value;}
fn smoothHybrid(row:u32,src:u32)->f32{let current=pv(src,row);if(pu(BAND_B,row)==0u){return current;}
 let next=current+p.numerics.z*(pv(PR,row)-applyL2(row,src))/headers[row].diagonal;if(!finite(next)){reportAt(NONFINITE,3u,row,next);return current;}return next;}
fn applyOuter(row:u32,channel:u32)->f32{let h=headers[row];var value=h.diagonal*pv(channel,row);for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];value-=e.coefficient*pv(channel,e.row);}return value;}
fn applyPersistentVCycle(lid:u32){
 for(var l=0u;l<levels();l+=1u){for(var i=lid;i<count(l);i+=PERSISTENT_LANES){let s=workSlot(l,i);for(var c=RHS;c<=B;c+=1u){storef(c,l,s,0.0);}}}sync();
 if(!stopped()){for(var row=lid;row<rows();row+=PERSISTENT_LANES){let value=pv(HRHS,row);let native=firstTrailingBit(headers[row].size);storef(RHS,native,rowMap(native,row),value);}}sync();
 for(var l=0u;l+1u<levels();l+=1u){
  for(var iteration=0u;iteration<p.vcycle.x;iteration+=1u){if(!stopped()){let src=select(A,B,(iteration&1u)!=0u);let dst=select(B,A,(iteration&1u)!=0u);
    for(var i=lid;i<count(l);i+=PERSISTENT_LANES){persistentRelaxJacobi(l,workSlot(l,i),src,dst);}}sync();}
  if(!stopped()){for(var i=lid;i<count(l+1u);i+=PERSISTENT_LANES){let coarse=workSlot(l+1u,i);var sum=0.0;
    var record=topology[parentHeadBase(l)+coarse];for(var visited=0u;record!=INVALID&&visited<transferCount(l);visited+=1u){
     let fine=topology[transferWord(l,record,0u)];let parent=topology[transferWord(l,record,1u)];
     if(parent!=coarse){report(OVERFLOW);break;}let ghost=!smoothable(l,fine);let product=persistentApplied(l,fine,A);
     let residualValue=select(-product,loadf(RHS,l,fine)-product,!ghost);
     sum+=bitcast<f32>(topology[transferWord(l,record,2u)])*residualValue;
     record=topology[transferWord(l,record,3u)];}
    if(record!=INVALID||!finite(sum)){report(OVERFLOW);}else{storef(RHS,l+1u,coarse,sum);}}}sync();
 }
 if(lid==0u&&!stopped()){let bottom=levels()-1u;if(count(bottom)!=1u){report(NONPOSITIVE);}else{let s=workSlot(bottom,0u);let d=loadf(DIAG,bottom,s);
  if(!(d>0.0)){report(NONPOSITIVE);}else{let value=loadf(RHS,bottom,s)/d;if(!finite(value)){report(NONFINITE);}else{storef(A,bottom,s,value);}}}}sync();
 for(var reverse=1u;reverse<levels();reverse+=1u){let l=levels()-1u-reverse;
  if(!stopped()){for(var i=lid;i<count(l);i+=PERSISTENT_LANES){let fine=workSlot(l,i);let ghost=!smoothable(l,fine);let targetCount=select(8u,1u,ghost);var value=select(loadf(A,l,fine),0.0,ghost);
    for(var corner=0u;corner<targetCount;corner+=1u){let transfer=correctionTransfer(l,fine,corner);if(transfer.coarse!=INVALID){value+=transfer.weight*loadf(A,l+1u,transfer.coarse);}}
    if(!finite(value)){report(NONFINITE);}else{storef(A,l,fine,value);}}}sync();
  for(var iteration=0u;iteration<p.vcycle.x;iteration+=1u){if(!stopped()){let src=select(A,B,(iteration&1u)!=0u);let dst=select(B,A,(iteration&1u)!=0u);
    for(var i=lid;i<count(l);i+=PERSISTENT_LANES){persistentRelaxJacobi(l,workSlot(l,i),src,dst);}}sync();}
 }
 if(!stopped()){for(var row=lid;row<rows();row+=PERSISTENT_LANES){let native=firstTrailingBit(headers[row].size);let value=loadf(A,native,rowMap(native,row));
  if(!finite(value)){report(NONFINITE);}else{storePV(HCORR,row,value);}}}sync();
}
fn applyHybridPreconditioner(lid:u32){
 if(!stopped()){for(var row=lid;row<rows();row+=PERSISTENT_LANES){storePV(HA,row,0.0);storePV(HB,row,0.0);storePV(HRHS,row,0.0);storePV(HCORR,row,0.0);}}sync();
 for(var iteration=0u;iteration<boundarySweeps();iteration+=1u){if(!stopped()){let src=select(HA,HB,(iteration&1u)!=0u);let dst=select(HB,HA,(iteration&1u)!=0u);
   for(var row=lid;row<rows();row+=PERSISTENT_LANES){storePV(dst,row,smoothHybrid(row,src));}}sync();}
 if(!stopped()){for(var row=lid;row<rows();row+=PERSISTENT_LANES){let next=pv(PR,row)-applyL2(row,HA);if(!finite(next)){reportAt(NONFINITE,4u,row,next);}else{storePV(HRHS,row,next);}}}sync();
 applyPersistentVCycle(lid);
 if(!stopped()){for(var row=lid;row<rows();row+=PERSISTENT_LANES){let next=pv(HA,row)+pv(HCORR,row);if(!finite(next)){reportAt(NONFINITE,5u,row,next);}else{storePV(HB,row,next);}}}sync();
 for(var iteration=0u;iteration<boundarySweeps();iteration+=1u){if(!stopped()){let src=select(HB,HA,(iteration&1u)!=0u);let dst=select(HA,HB,(iteration&1u)!=0u);
   for(var row=lid;row<rows();row+=PERSISTENT_LANES){storePV(dst,row,smoothHybrid(row,src));}}sync();}
 if(!stopped()){for(var row=lid;row<rows();row+=PERSISTENT_LANES){let value=pv(HB,row);if(!finite(value)){reportAt(NONFINITE,6u,row,value);}else{storePV(PZ,row,value);}}}sync();
}
var<workgroup> sums:array<vec4f,64>;
fn reduce(lid:u32,value:vec4f)->vec4f{sums[lid]=value;for(var width=32u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}workgroupBarrier();return sums[0];}
fn persistentPCGIteration(lid:u32){let n=rows();
 if(!stopped()){for(var row=lid;row<n;row+=PERSISTENT_LANES){storePV(PQ,row,applyOuter(row,PD));}}sync();
 var localDQ=0.0;if(!stopped()){for(var row=lid;row<n;row+=PERSISTENT_LANES){localDQ+=pv(PD,row)*pv(PQ,row);}}let dq=reduce(lid,vec4f(localDQ,0.0,0.0,0.0)).x;
 if(lid==0u&&!stopped()){let rz=bitcast<f32>(control[6]);if(!finite(dq)||!finite(rz)){reportAt(NONFINITE,8u,INVALID,dq);}else if(dq<=epsilon()||rz<0.0){report(NONPOSITIVE);}else{control[7]=bitcast<u32>(rz/dq);control[9]=bitcast<u32>(dq);}}sync();
 if(!stopped()){let alpha=bitcast<f32>(control[7]);for(var row=lid;row<n;row+=PERSISTENT_LANES){let x=pv(PX,row)+alpha*pv(PD,row);let r=pv(PR,row)-alpha*pv(PQ,row);
   if(!finite(x)||!finite(r)){reportAt(NONFINITE,9u,row,r);}else{storePV(PX,row,x);storePV(PR,row,r);}}}sync();
 var localRR=0.0;if(!stopped()){for(var row=lid;row<n;row+=PERSISTENT_LANES){let r=pv(PR,row);localRR+=r*r;}}let rr=reduce(lid,vec4f(localRR,0.0,0.0,0.0)).x;
 if(lid==0u&&!stopped()){let bb=bitcast<f32>(control[5]);if(!finite(rr)){reportAt(NONFINITE,10u,INVALID,rr);}else{control[4]=bitcast<u32>(rr);control[2]+=1u;if(rr<=tolerance()*tolerance()*max(bb,epsilon())){control[1]=1u;}}}sync();
 applyHybridPreconditioner(lid);
 var localRZ=0.0;if(!stopped()){for(var row=lid;row<n;row+=PERSISTENT_LANES){localRZ+=pv(PR,row)*pv(PZ,row);}}let nextRZ=reduce(lid,vec4f(localRZ,0.0,0.0,0.0)).x;
 if(lid==0u&&!stopped()){let previous=bitcast<f32>(control[6]);if(!finite(nextRZ)||nextRZ<0.0||previous<=epsilon()){reportAt(NONFINITE,13u,INVALID,nextRZ);}else{control[8]=bitcast<u32>(nextRZ/previous);control[6]=bitcast<u32>(nextRZ);}}sync();
 if(!stopped()){let beta=bitcast<f32>(control[8]);for(var row=lid;row<n;row+=PERSISTENT_LANES){let next=pv(PZ,row)+beta*pv(PD,row);if(!finite(next)){reportAt(NONFINITE,11u,row,next);}else{storePV(PD,row,next);}}}sync();
}
fn persistentInitialize(lid:u32){let n=rows();
 // Uniform fail-closed gate: every structural row and entry is checked before
 // any pressure, residual, smoother, transfer, or Krylov arithmetic executes.
 for(var row=lid;row<n;row+=PERSISTENT_LANES){let h=headers[row];if(h.entryStart>arrayLength(&entries)||h.entryCount>arrayLength(&entries)-h.entryStart||!finite(h.diagonal)||h.diagonal<=0.0||!finite(h.rhs)){report(INVALID_ROW_ERROR);}else{
  for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=n||!finite(e.coefficient)){report(INVALID_ROW_ERROR);}}}
  if(!finite(pressureSeed[row])){reportAt(NONFINITE,1u,row,pressureSeed[row]);}}
 if(lid==0u){let words=arrayLength(&rowCounts);if(words<8u||rowCounts[words-8u]!=0u){report(INVALID_ROW_ERROR);}}sync();
 if(!failed()){for(var row=lid;row<n;row+=PERSISTENT_LANES){storePV(PX,row,pressureSeed[row]);storePV(PR,row,0.0);storePV(PZ,row,0.0);storePV(PD,row,0.0);storePV(PQ,row,0.0);}}sync();
 if(!failed()){for(var row=lid;row<n;row+=PERSISTENT_LANES){storePV(PD,row,applyOuter(row,PX));}}sync();
 if(!failed()){for(var row=lid;row<n;row+=PERSISTENT_LANES){let value=-headers[row].rhs-pv(PD,row);if(!finite(value)){reportAt(NONFINITE,2u,row,value);}else{storePV(PR,row,value);}}}sync();
 if(!failed()){for(var row=lid;row<n;row+=PERSISTENT_LANES){let h=headers[row];var offDiagonalSum=0.0;var transition=false;for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];offDiagonalSum+=e.coefficient;transition=transition||headers[e.row].size!=h.size;}
   let boundaryGap=h.diagonal-offDiagonalSum;let boundary=(h.pad0&ROW_BOUNDARY)!=0u||boundaryGap>1e-5*max(1.0,h.diagonal);storePU(BAND_A,row,select(0u,1u,boundary||transition));}}sync();
 for(var layer=0u;layer<3u;layer+=1u){if(!failed()){let src=select(BAND_A,BAND_B,(layer&1u)!=0u);let dst=select(BAND_B,BAND_A,(layer&1u)!=0u);
   for(var row=lid;row<n;row+=PERSISTENT_LANES){var band=pu(src,row);let h=headers[row];for(var j=0u;j<h.entryCount&&band==0u;j+=1u){band=max(band,pu(src,entries[h.entryStart+j].row));}storePU(dst,row,band);}}sync();}
 applyHybridPreconditioner(lid);
 var initial=vec4f(0.0);if(!stopped()){for(var row=lid;row<n;row+=PERSISTENT_LANES){let r=pv(PR,row);let z=pv(PZ,row);let b=-headers[row].rhs;initial+=vec4f(r*r,b*b,r*z,0.0);storePV(PD,row,z);}}
 let initialTotal=reduce(lid,initial);if(lid==0u&&!failed()){control[3]=n;control[4]=bitcast<u32>(initialTotal.x);control[5]=bitcast<u32>(initialTotal.y);control[6]=bitcast<u32>(initialTotal.z);
  if(!finite(initialTotal.x)||!finite(initialTotal.y)||!finite(initialTotal.z)||initialTotal.z<0.0){reportAt(NONFINITE,7u,INVALID,initialTotal.z);}else if(initialTotal.x<=tolerance()*tolerance()*max(initialTotal.y,epsilon())){control[1]=1u;}}sync();
}
fn persistentFinalize(lid:u32){let n=rows();
 if(lid==0u){if(control[1]==0u&&control[0]==0u){report(NONCONVERGENCE);}if(arrayLength(&rowCounts)>=2u){let tail=arrayLength(&rowCounts);let success=!failed()&&control[1]!=0u;
   rowCounts[tail-2u]=select(0x7fc00000u,control[4],success);rowCounts[tail-1u]=select(0x7fc00000u,control[5],success);}}sync();
 for(var row=lid;row<n;row+=PERSISTENT_LANES){let success=!failed()&&control[1]!=0u;let seed=select(0.0,pressureSeed[row],finite(pressureSeed[row]));pressureOut[row]=select(seed,pv(PX,row),success&&finite(pv(PX,row)));}
}
@compute @workgroup_size(64) fn persistentMGPCG(@builtin(local_invocation_index) lid:u32){
 persistentLane=lid;persistentErrors[lid]=vec4u(0u);workgroupBarrier();
 persistentInitialize(lid);
 for(var iteration=0u;iteration<12u;iteration+=1u){
  if(lid==0u){persistentStop=select(0u,1u,stopped());}
  let stop=workgroupUniformLoad(&persistentStop);if(stop!=0u){break;}
  persistentPCGIteration(lid);
 }
 persistentFinalize(lid);
}
`;

export const octreeSPGridVCycleShader = /* wgsl */ `
struct Params{dimsLevel:vec4u,capacity:vec4u,dispatchSmooth:vec4u,solve:vec2u,weights:vec2f,delta:vec4u}
struct LeafHeader{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f}
struct LeafEntry{row:u32,coefficient:f32}
struct TransferTarget{coarse:u32,weight:f32}
struct CapturePageRecord{generation:u32,changeStamp:u32,validationStamp:u32,copyStamp:u32}
struct CapturePageState{generation:u32,expectedPages:u32,validatedPages:u32,changedPages:u32,
 readyGeneration:u32,copiedPages:u32,publishedGeneration:u32,publishedRows:u32,
 error:u32,bootstrap:u32,sourceGeneration:u32,publishedSourceGeneration:u32,pages:array<CapturePageRecord>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read_write> entries:array<LeafEntry>;
@group(0) @binding(3) var<storage,read> rowCounts:array<u32>;
@group(0) @binding(4) var<storage,read_write> topology:array<u32>;
@group(0) @binding(5) var<storage,read_write> state:array<u32>;
@group(0) @binding(6) var<storage,read_write> dispatchMeta:array<u32>;
@group(0) @binding(7) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> inputRhs:array<f32>;
@group(0) @binding(9) var<storage,read_write> outputCorrection:array<f32>;
@group(0) @binding(11) var<storage,read> sourceHeaders:array<LeafHeader>;
@group(0) @binding(12) var<storage,read> sourceEntries:array<LeafEntry>;
@group(0) @binding(13) var<storage,read_write> capturePages:CapturePageState;
@group(0) @binding(14) var<storage,read_write> levelDelta:array<u32>;
@group(0) @binding(15) var<storage,read_write> candidateTopology:array<u32>;
@group(0) @binding(16) var<storage,read_write> candidateState:array<u32>;
@group(0) @binding(17) var<storage,read_write> candidateDispatch:array<u32>;
@group(0) @binding(18) var<storage,read> sourceDelta:array<u32>;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;const INVALID=0xffffffffu;
const OVERFLOW=2u;const NONFINITE=4u;const NONPOSITIVE=8u;
const KEY=0u;const FLAGS=1u;const DIAG=2u;const XP=3u;const XM=4u;const YP=5u;const YM=6u;const ZP=7u;const ZM=8u;
const XYPP=9u;const XYPM=10u;const XYMP=11u;const XYMM=12u;const XZPP=13u;const XZPM=14u;const XZMP=15u;const XZMM=16u;
const YZPP=17u;const YZPM=18u;const YZMP=19u;const YZMM=20u;
const RHS=21u;const A=22u;const B=23u;const OWNER=24u;
const STATE_CHANNELS=25u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn stopped()->bool{return atomicLoad(&control[0])!=0u||atomicLoad(&control[1])!=0u;}
fn report(flag:u32){atomicOr(&control[0],flag);}fn rows()->u32{return min(rowCounts[0],p.capacity.x);}fn level()->u32{return p.dimsLevel.w;}
fn stride()->u32{return p.capacity.z;}fn levels()->u32{return p.capacity.y;}fn transferStride()->u32{return p.capacity.w;}
fn rowIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.x*64u;}fn slotIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.y*64u;}
fn boundedLinearIndex(g:vec3u)->u32{return g.x+g.y*65535u*64u;}
fn transferIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.z*64u;}fn at(c:u32,l:u32,s:u32)->u32{return(c*levels()+l)*stride()+s;}
fn loadf(c:u32,l:u32,s:u32)->f32{return bitcast<f32>(state[at(c,l,s)]);}fn storef(c:u32,l:u32,s:u32,v:f32){state[at(c,l,s)]=bitcast<u32>(v);}
fn rowMapBase()->u32{return 16u;}fn workBase()->u32{return rowMapBase()+levels()*p.capacity.x;}
fn transferBase()->u32{return workBase()+levels()*stride();}fn rowMap(l:u32,r:u32)->u32{return topology[rowMapBase()+l*p.capacity.x+r];}
fn workSlot(l:u32,i:u32)->u32{return topology[workBase()+l*stride()+i];}
fn transferLevelWords()->u32{return transferStride()*4u+4u*stride();}
fn transferWord(l:u32,i:u32,w:u32)->u32{return transferBase()+l*transferLevelWords()+i*4u+w;}
fn parentHeadBase(l:u32)->u32{return transferBase()+l*transferLevelWords()+transferStride()*4u;}
fn parentTailBase(l:u32)->u32{return parentHeadBase(l)+stride();}
fn fineHeadBase(l:u32)->u32{return parentTailBase(l)+stride();}
fn fineCountBase(l:u32)->u32{return fineHeadBase(l)+stride();}
fn directoryBase()->u32{return transferBase()+(levels()-1u)*transferLevelWords();}
fn brickDims(l:u32)->vec3u{return(dims(l)+vec3u(3u))/4u;}fn brickCount(l:u32)->u32{let d=brickDims(l);return d.x*d.y*d.z;}
fn brickLevelOffset(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=brickCount(k);}return result;}
fn totalBrickCount()->u32{return brickLevelOffset(levels());}
fn brickRecord(l:u32,q:vec3u)->u32{let d=brickDims(l);let b=q/4u;let dense=b.x+d.x*(b.y+d.y*b.z);
 return directoryBase()+16u+(brickLevelOffset(l)+dense)*4u;}
fn rankedSlotsBase()->u32{return directoryBase()+16u+totalBrickCount()*4u;}
fn neighbourBase()->u32{return rankedSlotsBase()+levels()*stride();}
fn neighbourAt(k:u32,l:u32,s:u32)->u32{return neighbourBase()+(k*levels()+l)*stride()+s;}
fn localBit(q:vec3u)->u32{let local=q&vec3u(3u);return local.x+4u*local.y+16u*local.z;}
fn count(l:u32)->u32{return dispatchMeta[l*8u];}fn transferCount(l:u32)->u32{return dispatchMeta[l*8u+1u];}
fn lifecycleBase()->u32{return levels()*8u;}fn previousValid()->bool{return dispatchMeta[lifecycleBase()]==1u;}
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
fn sameL1Topology(a:LeafHeader,b:LeafHeader)->bool{
 // RHS, diagonal, and diagnostic gradients do not define sparse identity.
 return a.cell==b.cell&&a.entryCount==b.entryCount&&a.size==b.size;
}
fn validEntryRange(h:LeafHeader)->bool{
 let sourceLength=arrayLength(&sourceEntries);let targetLength=arrayLength(&entries);
 return h.entryStart<=sourceLength&&h.entryCount<=sourceLength-h.entryStart
  &&h.entryStart<=targetLength&&h.entryCount<=targetLength-h.entryStart;
}
var<workgroup> captureLaneFlags:array<u32,64>;
var<workgroup> captureLaneFirst:array<u32,64>;
var<workgroup> captureLaneRange:array<vec2u,64>;
var<workgroup> captureLaneChanged:array<u32,64>;
fn dims(l:u32)->vec3u{let s=1u<<l;return (p.dimsLevel.xyz+vec3u(s-1u))/s;}fn coordKey(q:vec3u,l:u32)->u32{let d=dims(l);return q.x+d.x*(q.y+d.y*q.z)+1u;}
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;return vec3u(v%d.x,(v/d.x)%d.y,v/(d.x*d.y));}
fn insertionHash(key:u32)->u32{var h=key*0x9e3779b1u;h=(h^(h>>16u))*0x7feb352du;return(h^(h>>15u))&(stride()-1u);}
fn directoryLookup(l:u32,q:vec3u,requirePublication:bool)->u32{if(l>=levels()||any(q>=dims(l))){return INVALID;}
 let generation=topology[directoryBase()+2u+l];if(generation==0u){report(OVERFLOW);return INVALID;}
 let record=brickRecord(l,q);if(topology[record]!=generation){report(OVERFLOW);return INVALID;}let bit=localBit(q);let word=topology[record+1u+(bit>>5u)];
 let flag=1u<<(bit&31u);if((word&flag)==0u){return INVALID;}let low=topology[record+1u];let high=topology[record+2u];
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}let ranked=topology[record+3u]+rank;
 if(ranked>=count(l)||ranked>=stride()){report(OVERFLOW);return INVALID;}let slot=topology[rankedSlotsBase()+l*stride()+ranked];
 if(slot>=stride()||state[at(KEY,l,slot)]!=coordKey(q,l)){report(OVERFLOW);return INVALID;}return slot;}
fn find(l:u32,q:vec3u)->u32{return directoryLookup(l,q,true);}
fn originOf(h:LeafHeader)->vec3u{return vec3u(h.cell%p.dimsLevel.x,(h.cell/p.dimsLevel.x)%p.dimsLevel.y,h.cell/(p.dimsLevel.x*p.dimsLevel.y));}
fn contactCoord(own:LeafHeader,other:LeafHeader,l:u32)->vec3u{let scale=1u<<l;let begin=originOf(own);let finish=begin+vec3u(own.size);
 let otherBegin=originOf(other);let otherFinish=otherBegin+vec3u(other.size);var result=vec3u(0u);
 for(var axis=0u;axis<3u;axis+=1u){if(otherBegin[axis]>=finish[axis]){result[axis]=(finish[axis]-1u)/scale;}
  else if(otherFinish[axis]<=begin[axis]){result[axis]=begin[axis]/scale;}else{let centre=(2u*otherBegin[axis]+other.size)/(2u*scale);
   result[axis]=clamp(centre,begin[axis]/scale,(finish[axis]-1u)/scale);}}return result;}
// Cold bootstrap validates every L1 page once. Recurring generations consume
// only the producer's compact, sorted dirty-row authority; the first dirty row
// in each page exclusively validates and stamps that complete page. No warm
// kernel scans the row publication, entry arena, or page table globally.
@compute @workgroup_size(1) fn beginL1CapturePlan(){
 var generation=capturePages.generation+1u;if(generation==0u){generation=1u;}
 let n=rows();let expected=(n+CAPTURE_PAGE_ROWS-1u)/CAPTURE_PAGE_ROWS;
 capturePages.generation=generation;capturePages.expectedPages=expected;
 capturePages.validatedPages=0u;capturePages.changedPages=0u;
 capturePages.readyGeneration=0u;capturePages.copiedPages=0u;
 capturePages.error=0u;
 capturePages.bootstrap=select(0u,1u,capturePages.publishedGeneration==0u);
 capturePages.sourceGeneration=select(0u,deltaControl(7u),p.delta.y+8u<arrayLength(&sourceDelta));
 for(var l=0u;l<levels();l+=1u){for(var w=0u;w<DELTA_WORDS;w+=1u){levelDelta[deltaAt(l,w)]=0u;}
  levelDelta[deltaAt(l,2u)]=0xffffffffu;}
 let publishedGeneration=capturePages.publishedGeneration;let publishedRows=capturePages.publishedRows;
 if(arrayLength(&rowCounts)==0u||rowCounts[0]>p.capacity.x||expected>capturePageCount()
  ||publishedRows>p.capacity.x||(publishedGeneration==0u&&publishedRows!=0u)
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
  for(var r=begin;r<end;r+=1u){let source=sourceHeaders[r];let volume=p.dimsLevel.x*p.dimsLevel.y*p.dimsLevel.z;
   if(!validEntryRange(source)||source.cell>=volume||source.size==0u||(source.size&(source.size-1u))!=0u
    ||!finite(source.diagonal)||source.diagonal<=0.0){pageFlags|=2u;continue;}
   let old=select(deltaOldRow(r),INVALID,captureBootstrap());
   var topologyChanged=old==INVALID||old>=capturePages.publishedRows;
   var stencilChanged=topologyChanged;
   if(!topologyChanged){let captured=headers[old];topologyChanged=!sameL1Topology(source,captured);
    stencilChanged=topologyChanged||bitcast<u32>(source.diagonal)!=bitcast<u32>(captured.diagonal);
    if(topologyChanged){pageFirst=min(pageFirst,min(firstTrailingBit(source.size),firstTrailingBit(captured.size)));}
    if(captured.entryStart>arrayLength(&entries)||captured.entryCount>arrayLength(&entries)-captured.entryStart){
     pageFlags|=2u;topologyChanged=true;stencilChanged=true;pageFirst=0u;
    }else if(!topologyChanged){for(var j=0u;j<source.entryCount;j+=1u){let a=sourceEntries[source.entryStart+j];
      let b=entries[captured.entryStart+j];var oldNeighbor=INVALID;
      if(a.row<rows()){oldNeighbor=deltaOldRow(a.row);}
      if(a.row>=rows()||!finite(a.coefficient)||oldNeighbor!=b.row){topologyChanged=true;}
      if(bitcast<u32>(a.coefficient)!=bitcast<u32>(b.coefficient)){stencilChanged=true;}
      if(topologyChanged||stencilChanged){pageFirst=min(pageFirst,firstTrailingBit(source.size));
       if(a.row<rows()){pageFirst=min(pageFirst,firstTrailingBit(sourceHeaders[a.row].size));}
       if(b.row<capturePages.publishedRows){pageFirst=min(pageFirst,firstTrailingBit(headers[b.row].size));}}
    }}
   }else{pageFirst=0u;}
   if(topologyChanged){pageFlags|=4u;stencilChanged=true;}
   if(stencilChanged){pageFlags|=8u;pageFirst=min(pageFirst,firstTrailingBit(source.size));}
   for(var j=0u;j<source.entryCount;j+=1u){let a=sourceEntries[source.entryStart+j];
    if(a.row>=rows()||!finite(a.coefficient)){pageFlags|=2u;}}
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
  for(var r=page*CAPTURE_PAGE_ROWS;r<end;r+=1u){let h=sourceHeaders[r];headers[r]=h;
   for(var j=0u;j<h.entryCount;j+=1u){entries[h.entryStart+j]=sourceEntries[h.entryStart+j];}}
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
@compute @workgroup_size(64) fn clearCorrection(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){outputCorrection[r]=0.0;}}
fn cAt(c:u32,l:u32,s:u32)->u32{return(c*levels()+l)*stride()+s;}
fn cCount(l:u32)->u32{return candidateDispatch[l*8u];}
fn cWorkSlot(l:u32,i:u32)->u32{return candidateTopology[workBase()+l*stride()+i];}
fn cRowMap(l:u32,r:u32)->u32{return candidateTopology[rowMapBase()+l*p.capacity.x+r];}
fn cLoadf(c:u32,l:u32,s:u32)->f32{return bitcast<f32>(candidateState[cAt(c,l,s)]);}
fn cStoref(c:u32,l:u32,s:u32,v:f32){candidateState[cAt(c,l,s)]=bitcast<u32>(v);}
fn candidateReport(l:u32){levelDelta[deltaAt(l,4u)]|=DELTA_ERROR;}
fn cMergeClass(index:u32,incoming:u32){let old=candidateState[index];var merged=MG_ONLY;
 if((old&ACTIVE)!=0u||(incoming&ACTIVE)!=0u){merged=ACTIVE;}else if((old&GHOST)!=0u||(incoming&GHOST)!=0u){merged=GHOST;}
 candidateState[index]=merged;}
fn cInsert(l:u32,q:vec3u,flags:u32)->u32{let key=coordKey(min(q,dims(l)-vec3u(1u)),l);var slot=insertionHash(key);
 for(var probe=0u;probe<256u;probe+=1u){let index=cAt(KEY,l,slot);let old=candidateState[index];
  if(old==key){cMergeClass(cAt(FLAGS,l,slot),flags);return slot;}if(old==0u){candidateState[index]=key;
   cMergeClass(cAt(FLAGS,l,slot),flags);let w=cCount(l);if(w>=stride()){candidateReport(l);return INVALID;}
   candidateDispatch[l*8u]=w+1u;candidateTopology[workBase()+l*stride()+w]=slot;return slot;}
  slot=(slot+1u)&(stride()-1u);}candidateReport(l);return INVALID;}
fn cInsertOwned(l:u32,q:vec3u,flags:u32,owner:u32)->u32{let slot=cInsert(l,q,flags);if(slot==INVALID){return INVALID;}
 let encoded=owner+1u;let old=candidateState[cAt(OWNER,l,slot)];if(old==encoded){return slot;}
 if(old!=0u){candidateReport(l);return INVALID;}candidateState[cAt(OWNER,l,slot)]=encoded;return slot;}
fn selectedCount(l:u32)->u32{return select(count(l),cCount(l),topologyDirty(l));}
fn selectedWorkSlot(l:u32,i:u32)->u32{return select(workSlot(l,i),cWorkSlot(l,i),topologyDirty(l));}
fn selectedRowMap(l:u32,r:u32)->u32{return select(rowMap(l,r),cRowMap(l,r),topologyDirty(l));}
fn selectedState(c:u32,l:u32,s:u32)->u32{return select(state[at(c,l,s)],candidateState[cAt(c,l,s)],topologyDirty(l));}
fn cAppendTransfer(l:u32,fine:u32,coarse:u32,weight:f32){let i=candidateDispatch[l*8u+1u];
 if(i>=transferStride()){candidateReport(l);return;}candidateDispatch[l*8u+1u]=i+1u;
 candidateTopology[transferWord(l,i,0u)]=fine;candidateTopology[transferWord(l,i,1u)]=coarse;
 candidateTopology[transferWord(l,i,2u)]=bitcast<u32>(weight);candidateTopology[transferWord(l,i,3u)]=INVALID;
 let fineHead=fineHeadBase(l)+fine;let fineCount=fineCountBase(l)+fine;let owned=candidateTopology[fineCount];
 if(owned==0u){candidateTopology[fineHead]=i;}else if(candidateTopology[fineHead]+owned!=i){candidateReport(l);return;}
 candidateTopology[fineCount]=owned+1u;
 let tail=candidateTopology[parentTailBase(l)+coarse];if(tail==INVALID){candidateTopology[parentHeadBase(l)+coarse]=i;}
 else{candidateTopology[transferWord(l,tail,3u)]=i;}candidateTopology[parentTailBase(l)+coarse]=i;}
fn cDirectoryLookup(l:u32,q:vec3u)->u32{if(l>=levels()||any(q>=dims(l))){return INVALID;}
 let generation=levelDelta[deltaAt(l,6u)];let record=brickRecord(l,q);if(candidateTopology[record]!=generation){candidateReport(l);return INVALID;}
 let bit=localBit(q);let word=candidateTopology[record+1u+(bit>>5u)];let flag=1u<<(bit&31u);if((word&flag)==0u){return INVALID;}
 let low=candidateTopology[record+1u];let high=candidateTopology[record+2u];let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);
 var rank=countOneBits(low&lower);if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
 let ranked=candidateTopology[record+3u]+rank;if(ranked>=cCount(l)||ranked>=stride()){candidateReport(l);return INVALID;}
 let slot=candidateTopology[rankedSlotsBase()+l*stride()+ranked];
 if(slot>=stride()||candidateState[cAt(KEY,l,slot)]!=coordKey(q,l)){candidateReport(l);return INVALID;}return slot;}
fn cOwnedContactSlot(l:u32,row:u32,other:u32)->u32{let h=sourceHeaders[row];let native=firstTrailingBit(h.size);
 if(l>=native){return cRowMap(l,row);}let slot=cDirectoryLookup(l,contactCoord(h,sourceHeaders[other],l));
 if(slot==INVALID||candidateState[cAt(OWNER,l,slot)]!=row+1u){return INVALID;}return slot;}
fn cAddFace(l:u32,own:u32,other:u32,a:vec3u,b:vec3u,c:f32)->bool{let d=vec3i(b)-vec3i(a);var ch=0u;
 if(all(d==vec3i(1,0,0))){ch=XP;}else if(all(d==vec3i(-1,0,0))){ch=XM;}
 else if(all(d==vec3i(0,1,0))){ch=YP;}else if(all(d==vec3i(0,-1,0))){ch=YM;}
 else if(all(d==vec3i(0,0,1))){ch=ZP;}else if(all(d==vec3i(0,0,-1))){ch=ZM;}
 else if(all(d==vec3i(1,1,0))){ch=XYPP;}else if(all(d==vec3i(1,-1,0))){ch=XYPM;}else if(all(d==vec3i(-1,1,0))){ch=XYMP;}else if(all(d==vec3i(-1,-1,0))){ch=XYMM;}
 else if(all(d==vec3i(1,0,1))){ch=XZPP;}else if(all(d==vec3i(1,0,-1))){ch=XZPM;}else if(all(d==vec3i(-1,0,1))){ch=XZMP;}else if(all(d==vec3i(-1,0,-1))){ch=XZMM;}
 else if(all(d==vec3i(0,1,1))){ch=YZPP;}else if(all(d==vec3i(0,1,-1))){ch=YZPM;}else if(all(d==vec3i(0,-1,1))){ch=YZMP;}else if(all(d==vec3i(0,-1,-1))){ch=YZMM;}else{return false;}
 let neighbourIndex=neighbourAt(ch-XP,l,own);let previous=candidateTopology[neighbourIndex];
 if(previous!=INVALID&&previous!=other){return false;}candidateTopology[neighbourIndex]=other;
 cStoref(ch,l,own,cLoadf(ch,l,own)+c);return true;}

fn rebuildCandidateLevelSetFor(l:u32){
 if(!topologyDirty(l)){return;}levelDelta[deltaAt(l,4u)]=0u;
 for(var s=0u;s<stride();s+=1u){candidateState[cAt(KEY,l,s)]=0u;
  candidateState[cAt(FLAGS,l,s)]=0u;candidateState[cAt(OWNER,l,s)]=0u;
  candidateTopology[workBase()+l*stride()+s]=INVALID;}
 for(var r=0u;r<p.capacity.x;r+=1u){candidateTopology[rowMapBase()+l*p.capacity.x+r]=INVALID;}
 for(var w=0u;w<8u;w+=1u){candidateDispatch[l*8u+w]=0u;}
 for(var r=0u;r<rows();r+=1u){let h=sourceHeaders[r];let native=firstTrailingBit(h.size);
  if(l>=native){let q=originOf(h)/(1u<<l);var slot=INVALID;if(l==native){slot=cInsertOwned(l,q,ACTIVE,r);}else{slot=cInsert(l,q,MG_ONLY);}
   if(slot!=INVALID){candidateTopology[rowMapBase()+l*p.capacity.x+r]=slot;}}
  if(l<native){for(var j=0u;j<h.entryCount;j+=1u){let other=sourceEntries[h.entryStart+j].row;if(other>=rows()){candidateReport(l);continue;}
    if(l>=firstTrailingBit(sourceHeaders[other].size)){_ = cInsertOwned(l,contactCoord(h,sourceHeaders[other],l),GHOST,r);}}}
 }
}
fn rebuildCandidateTransferFor(l:u32){
 if(l+1u>=levels()||!(topologyDirty(l)||topologyDirty(l+1u))){return;}
 candidateDispatch[l*8u+1u]=0u;for(var s=0u;s<stride();s+=1u){
  candidateTopology[parentHeadBase(l)+s]=INVALID;candidateTopology[parentTailBase(l)+s]=INVALID;
  candidateTopology[fineHeadBase(l)+s]=INVALID;candidateTopology[fineCountBase(l)+s]=0u;}
 for(var i=0u;i<selectedCount(l);i+=1u){let fine=selectedWorkSlot(l,i);let q=decode(selectedState(KEY,l,fine),l);
  let flags=selectedState(FLAGS,l,fine);if((flags&GHOST)!=0u){let encodedOwner=selectedState(OWNER,l,fine);
   if(encodedOwner==0u||encodedOwner>rows()){candidateReport(l);continue;}let owner=encodedOwner-1u;
   let native=firstTrailingBit(sourceHeaders[owner].size);var coarse=INVALID;
   if(l+1u==native){coarse=selectedRowMap(l+1u,owner);}else{coarse=cInsertOwned(l+1u,q/2u,GHOST,owner);}
   if(coarse!=INVALID){cAppendTransfer(l,fine,coarse,1.0);}}
  else{let base=q/2u;let side=vec3i(select(-1,1,(q.x&1u)!=0u),select(-1,1,(q.y&1u)!=0u),select(-1,1,(q.z&1u)!=0u));
   for(var corner=0u;corner<8u;corner+=1u){var targetCoord=vec3i(base);var weight=1.0;
    for(var axis=0u;axis<3u;axis+=1u){if((corner&(1u<<axis))!=0u){targetCoord[axis]+=side[axis];weight*=0.25;}else{weight*=0.75;}}
    let coarse=cInsert(l+1u,vec3u(max(targetCoord,vec3i(0))),MG_ONLY);if(coarse!=INVALID){cAppendTransfer(l,fine,coarse,weight);}}
  }
 }
}
fn rebuildCandidateDirectoryFor(l:u32){
 if(!topologyDirty(l)){return;}let generation=levelDelta[deltaAt(l,0u)];levelDelta[deltaAt(l,6u)]=generation;
 for(var b=0u;b<brickCount(l);b+=1u){let record=brickRecord(l,vec3u((b%brickDims(l).x)*4u,
  ((b/brickDims(l).x)%brickDims(l).y)*4u,(b/(brickDims(l).x*brickDims(l).y))*4u));
  candidateTopology[record]=generation;candidateTopology[record+1u]=0u;candidateTopology[record+2u]=0u;candidateTopology[record+3u]=0u;}
 for(var i=0u;i<cCount(l);i+=1u){let slot=cWorkSlot(l,i);let q=decode(candidateState[cAt(KEY,l,slot)],l);
  let record=brickRecord(l,q);let bit=localBit(q);candidateTopology[record+1u+(bit>>5u)]|=1u<<(bit&31u);}
 var base=0u;for(var b=0u;b<brickCount(l);b+=1u){let record=directoryBase()+16u+(brickLevelOffset(l)+b)*4u;
  candidateTopology[record+3u]=base;base+=countOneBits(candidateTopology[record+1u])+countOneBits(candidateTopology[record+2u]);}
 if(base!=cCount(l)){candidateReport(l);}
 for(var i=0u;i<cCount(l);i+=1u){let slot=cWorkSlot(l,i);let q=decode(candidateState[cAt(KEY,l,slot)],l);
  let record=brickRecord(l,q);let bit=localBit(q);let low=candidateTopology[record+1u];let high=candidateTopology[record+2u];
  let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
  if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
  candidateTopology[rankedSlotsBase()+l*stride()+candidateTopology[record+3u]+rank]=slot;}
 candidateTopology[directoryBase()+2u+l]=generation;
}
fn rebuildCandidateStencilFor(l:u32){
 if(!stencilDirty(l)){return;}for(var i=0u;i<cCount(l);i+=1u){let s=cWorkSlot(l,i);
  for(var c=DIAG;c<=YZMM;c+=1u){candidateState[cAt(c,l,s)]=0u;}
  for(var k=0u;k<18u;k+=1u){candidateTopology[neighbourAt(k,l,s)]=INVALID;}}
 for(var r=0u;r<rows();r+=1u){let h=sourceHeaders[r];var sum=0.0;
  for(var j=0u;j<h.entryCount;j+=1u){let e=sourceEntries[h.entryStart+j];if(e.row>=rows()||!finite(e.coefficient)||e.coefficient<0.0){candidateReport(l);continue;}sum+=e.coefficient;}
  let native=firstTrailingBit(h.size);if(l>=native){let canonical=cRowMap(l,r);
   cStoref(DIAG,l,canonical,cLoadf(DIAG,l,canonical)+max(0.0,h.diagonal-sum));}
  for(var j=0u;j<h.entryCount;j+=1u){let e=sourceEntries[h.entryStart+j];if(e.row>=rows()){continue;}
   let otherNative=firstTrailingBit(sourceHeaders[e.row].size);if(l<native&&l<otherNative){continue;}
   let own=cOwnedContactSlot(l,r,e.row);let other=cOwnedContactSlot(l,e.row,r);
   if(own==INVALID||other==INVALID){candidateReport(l);continue;}if(other!=own){cStoref(DIAG,l,own,cLoadf(DIAG,l,own)+e.coefficient);
    let ownQ=decode(candidateState[cAt(KEY,l,own)],l);let otherQ=decode(candidateState[cAt(KEY,l,other)],l);
    if(!cAddFace(l,own,other,ownQ,otherQ,e.coefficient)){candidateReport(l);}}}
 }
 for(var i=0u;i<cCount(l);i+=1u){let s=cWorkSlot(l,i);if(cLoadf(DIAG,l,s)<=1e-20){cStoref(DIAG,l,s,1.0);}}
}
// Exact dependencies are ordered inside one singleton: every row set exists
// before transfers can insert coarse MG-only cells; all transfers exist before
// directories and stencils consume the immutable candidate identity. Dirty
// predicates remain per-level, so unchanged publications are never touched.
@compute @workgroup_size(1) fn buildCandidateLevelDeltas(){
 for(var l=0u;l<levels();l+=1u){rebuildCandidateLevelSetFor(l);}
 for(var l=0u;l+1u<levels();l+=1u){rebuildCandidateTransferFor(l);}
 for(var l=0u;l<levels();l+=1u){rebuildCandidateDirectoryFor(l);}
 for(var l=0u;l<levels();l+=1u){rebuildCandidateStencilFor(l);}
}
@compute @workgroup_size(1) fn validateCandidateHierarchy(){
 if(captureFailed()){report(OVERFLOW);return;}for(var l=0u;l<levels();l+=1u){
  if((levelDirty(l)&&levelDelta[deltaAt(l,4u)]!=0u)
   ||(topologyDirty(l)&&(levelDelta[deltaAt(l,6u)]!=captureGeneration()||cCount(l)>stride()))){
   captureReport(OVERFLOW);report(OVERFLOW);return;}}
 if(selectedCount(levels()-1u)!=1u){captureReport(OVERFLOW);report(OVERFLOW);}
}
fn commitCandidateLevelAt(l:u32,i:u32){
 let topologyChanged=topologyDirty(l);let transferChanged=topologyChanged||(l+1u<levels()&&topologyDirty(l+1u));
 if(!topologyChanged&&!stencilDirty(l)&&!transferChanged){return;}
 if(captureFailed()||levelDelta[deltaAt(l,4u)]!=0u){return;}
 if(stencilDirty(l)&&i<stride()){for(var c=DIAG;c<=YZMM;c+=1u){state[at(c,l,i)]=candidateState[cAt(c,l,i)];}
  for(var k=0u;k<18u;k+=1u){topology[neighbourAt(k,l,i)]=candidateTopology[neighbourAt(k,l,i)];}}
 if(topologyChanged&&i<stride()){for(var c=0u;c<STATE_CHANNELS;c+=1u){state[at(c,l,i)]=candidateState[cAt(c,l,i)];}
  topology[workBase()+l*stride()+i]=candidateTopology[workBase()+l*stride()+i];
  topology[rankedSlotsBase()+l*stride()+i]=candidateTopology[rankedSlotsBase()+l*stride()+i];}
 if(topologyChanged&&i<p.capacity.x){topology[rowMapBase()+l*p.capacity.x+i]=candidateTopology[rowMapBase()+l*p.capacity.x+i];}
 if(transferChanged&&l+1u<levels()&&i<stride()){topology[parentHeadBase(l)+i]=candidateTopology[parentHeadBase(l)+i];
  topology[parentTailBase(l)+i]=candidateTopology[parentTailBase(l)+i];
  topology[fineHeadBase(l)+i]=candidateTopology[fineHeadBase(l)+i];
  topology[fineCountBase(l)+i]=candidateTopology[fineCountBase(l)+i];}
 if(transferChanged&&l+1u<levels()&&i<transferStride()){for(var w=0u;w<4u;w+=1u){
  topology[transferWord(l,i,w)]=candidateTopology[transferWord(l,i,w)];}}
 if(topologyChanged&&i<brickCount(l)){let record=directoryBase()+16u+(brickLevelOffset(l)+i)*4u;
  for(var w=0u;w<4u;w+=1u){topology[record+w]=candidateTopology[record+w];}}
 if(i==0u){if(topologyChanged){dispatchMeta[l*8u]=candidateDispatch[l*8u];topology[directoryBase()+2u+l]=candidateTopology[directoryBase()+2u+l];
   levelDelta[deltaAt(l,5u)]=captureGeneration();}
  if(transferChanged&&l+1u<levels()){dispatchMeta[l*8u+1u]=candidateDispatch[l*8u+1u];}
  let blocks=(selectedCount(l)+63u)/64u;dispatchMeta[l*8u+2u]=min(65535u,blocks);
  dispatchMeta[l*8u+3u]=select(1u,(blocks+65534u)/65535u,blocks>0u);dispatchMeta[l*8u+4u]=1u;
  var parentBlocks=0u;if(l+1u<levels()){parentBlocks=(selectedCount(l+1u)+63u)/64u;}
  dispatchMeta[l*8u+5u]=min(65535u,parentBlocks);dispatchMeta[l*8u+6u]=select(1u,(parentBlocks+65534u)/65535u,parentBlocks>0u);
  dispatchMeta[l*8u+7u]=1u;}
}
@compute @workgroup_size(64) fn commitCandidateLevels(@builtin(global_invocation_id) g:vec3u){
 let i=boundedLinearIndex(g);for(var l=0u;l<levels();l+=1u){commitCandidateLevelAt(l,i);}
}
@compute @workgroup_size(1) fn finalizeLifecycle(){let base=lifecycleBase();if(atomicLoad(&control[0])==0u&&!captureFailed()){
 dispatchMeta[base]=1u;dispatchMeta[base+1u]=rows();return;}
 // Candidate failure leaves the previously published hierarchy byte-for-byte
 // authoritative. No recovery or in-place partial publication exists.
}
@compute @workgroup_size(64) fn zeroVectors(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||stopped()){return;}let s=workSlot(l,i);
 for(var c=RHS;c<=B;c+=1u){storef(c,l,s,0.0);}}
@compute @workgroup_size(64) fn seedRhs(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let v=inputRhs[r];let native=firstTrailingBit(headers[r].size);
 if(!finite(v)){report(NONFINITE);}else{storef(RHS,native,rowMap(native,r),v);}}}
fn applied(slot:u32,source:u32)->f32{let l=level();var value=loadf(DIAG,l,slot)*loadf(source,l,slot);
 for(var k=0u;k<18u;k+=1u){let c=loadf(XP+k,l,slot);if(c==0.0){continue;}
  let other=topology[neighbourAt(k,l,slot)];if(other>=stride()){report(OVERFLOW);continue;}
  value-=c*loadf(source,l,other);}return value;}
fn smoothable(l:u32,s:u32)->bool{return(state[at(FLAGS,l,s)]&GHOST)==0u;}
fn relaxJacobi(slot:u32,src:u32,dst:u32){let l=level();if(!smoothable(l,slot)){storef(dst,l,slot,loadf(src,l,slot));return;}
 let d=loadf(DIAG,l,slot);if(!(d>0.0)){report(NONPOSITIVE);return;}
 let x=loadf(src,l,slot)+p.weights.x*(loadf(RHS,l,slot)-applied(slot,src))/d;
 if(!finite(x)){report(NONFINITE);}else{storef(dst,l,slot,x);}}
@compute @workgroup_size(64) fn smoothAtoB(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){relaxJacobi(workSlot(l,i),A,B);}}
@compute @workgroup_size(64) fn smoothBtoA(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){relaxJacobi(workSlot(l,i),B,A);}}
// Return the immutable transfer target owned by one fine slot/corner.
// Restriction consumes the same records through its parent-owned chains, so
// E and E^T cannot diverge.
fn correctionTransfer(l:u32,fine:u32,corner:u32)->TransferTarget{
 if(l+1u>=levels()||fine>=stride()){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
 let first=topology[fineHeadBase(l)+fine];let n=topology[fineCountBase(l)+fine];
 if(first==INVALID||corner>=n||first+corner>=transferCount(l)){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
 let record=first+corner;if(topology[transferWord(l,record,0u)]!=fine){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
 let coarse=topology[transferWord(l,record,1u)];let weight=bitcast<f32>(topology[transferWord(l,record,2u)]);
 if(coarse>=stride()||!finite(weight)){report(OVERFLOW);return TransferTarget(INVALID,0.0);}
 return TransferTarget(coarse,weight);}
// Section 4.2 GhostValueAccumulate is E^T: one coarse owner traverses its
// immutable fine-major chain, so no destination synchronization is required.
@compute @workgroup_size(64) fn restrictAndGhostAccumulate(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();
 if(l+1u>=levels()||i>=count(l+1u)||stopped()){return;}let coarse=workSlot(l+1u,i);var sum=0.0;var record=topology[parentHeadBase(l)+coarse];
 for(var visited=0u;record!=INVALID&&visited<transferCount(l);visited+=1u){if(record>=transferCount(l)){report(OVERFLOW);return;}
  let fine=topology[transferWord(l,record,0u)];let parent=topology[transferWord(l,record,1u)];if(parent!=coarse){report(OVERFLOW);return;}
  let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;let product=applied(fine,A);let residual=select(-product,loadf(RHS,l,fine)-product,!ghost);
  sum+=bitcast<f32>(topology[transferWord(l,record,2u)])*residual;record=topology[transferWord(l,record,3u)];}
 if(record!=INVALID||!finite(sum)){report(OVERFLOW);return;}storef(RHS,l+1u,coarse,sum);}
@compute @workgroup_size(64) fn exactBottom(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>0u||stopped()){return;}if(count(l)!=1u){report(NONPOSITIVE);return;}
 let s=workSlot(l,0u);let d=loadf(DIAG,l,s);if(!(d>0.0)){report(NONPOSITIVE);return;}let x=loadf(RHS,l,s)/d;if(!finite(x)){report(NONFINITE);}else{storef(A,l,s,x);}}
// One fine invocation owns the complete interpolation sum, deleting all
// prolongation atomics. GhostValuePropagate is the unit-copy branch of the
// same E mapping rather than a second dispatch.
@compute @workgroup_size(64) fn prolongAndGhostPropagate(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||stopped()){return;}
 let fine=workSlot(l,i);let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;let targetCount=select(8u,1u,ghost);var value=select(loadf(A,l,fine),0.0,ghost);
 for(var corner=0u;corner<targetCount;corner+=1u){let transfer=correctionTransfer(l,fine,corner);if(transfer.coarse==INVALID){return;}
  value+=transfer.weight*loadf(A,l+1u,transfer.coarse);}if(!finite(value)){report(NONFINITE);}else{storef(A,l,fine,value);}}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let native=firstTrailingBit(headers[r].size);
 let v=loadf(A,native,rowMap(native,r));if(!finite(v)){report(NONFINITE);}else{outputCorrection[r]=v;}}}
`;
