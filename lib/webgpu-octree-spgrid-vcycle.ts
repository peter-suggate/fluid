import {
  OCTREE_FIRST_ORDER_CHEBYSHEV_DEGREES,
  OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION,
  type OctreeFirstOrderSPDVCycle,
} from "./webgpu-octree-section43-contract";
import { PassBroker } from "./webgpu-pass-broker";
import { planOctreeVCycleParallelLevels } from "./octree-solve-tail-policy";
import type { OctreeWorksetLinearOperator } from "./octree-linear-operator";
import { octreeAlgorithmDiagnosticsEnabled } from "./octree-algorithm-diagnostics";
import {
  RADIX_SORT_INPUT_VALID,
  WebGPURadixSortU32,
  spgridTouchedRadixSortEnabled,
} from "./webgpu-radix-sort-u32";
import {
  OCTREE_CUBE_TRANSFORMS,
  inverseCubeTransform,
  transformPowerVector,
} from "./octree-power-topology";

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
  /** Dense cell cardinality of each level's domain, finest first. The V-cycle
   * launch shape is authored from this immutable geometry. */
  readonly levelDomainCells: readonly number[];
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
  /** Eighteen published stencil-neighbour slot indices per cell. Written by the
   * same builder statement that publishes the coefficient they belong to, so a
   * coefficient can never be paired with a neighbour from another epoch. */
  readonly stencilNeighbourBytes: number;
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

export interface OctreeSPGridHierarchyLevelCensus {
  readonly level: number;
  readonly capacity: number;
  readonly occupied: number;
  readonly active: number;
  readonly ghost: number;
  readonly multigridOnly: number;
  readonly invalidClass: number;
  readonly publishedCount: number;
  readonly publishedTransferCount: number;
  readonly publishedPageCount: number;
  readonly selectedGroups: number;
  readonly restrictionGroups: number;
  readonly pageGroups: number;
  readonly gatedSelectedGroups: number;
  readonly gatedRestrictionGroups: number;
  readonly gatedPageGroups: number;
}

export interface OctreeSPGridHierarchyCensus {
  readonly levels: readonly Readonly<OctreeSPGridHierarchyLevelCensus>[];
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
  /**
   * Compile the page-parallel correction and accurate A2 executor used by the
   * hierarchical outer solve. Persistent MGPCG still requires the hierarchy's
   * transactional setup pipelines and buffers, but transcribes both applies
   * inside its single kernel.
   */
  readonly compileHierarchicalExecutor?: boolean;
  /** Allocate persistent hierarchy state now and compile its setup pipelines later. */
  readonly deferPipelineCompilation?: boolean;
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
  readonly oldToNewOffsetWords: number;
  readonly dirtyRowsOffsetWords: number;
  /** Control word holding the count for dirtyRowsOffsetWords (5=structural,
   * 6=the wider positional/face influence stream). */
  readonly dirtyCountControlWord?: 5 | 6;
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
  /** GPU-authored exact row dispatch. Record zero covers all live rows. */
  readonly liveRowDispatch?: GPUBuffer;
  readonly liveRowDispatchOffsetBytes?: number;
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
// valid/rows followed by a convergence-gated live-row indirect record.
const DISPATCH_LIFECYCLE_BYTES = 20;
// Candidate construction writes these records as STORAGE, then the broker
// copies them into a distinct INDIRECT-only arena.  Keeping them out of the
// accepted dispatch ABI lets the persistent solver continue consuming the
// compact per-level records above byte-for-byte.
const CANDIDATE_SCHEDULE_RECORDS = 11;
const CANDIDATE_SCHEDULE_BYTES = CANDIDATE_SCHEDULE_RECORDS * 12;
const CANDIDATE_SCHEDULE = Object.freeze({
  capture: 0, clear: 12, topologyLevels: 24, transferSlots: 36,
  bricks: 48, logicalPages: 60, physicalPages: 72, commit: 84,
  stencilLevels: 96, stencilSlots: 108, topologyRows: 120,
} as const);
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
// The accurate Section 6.3 operator addresses the same five variable arenas
// from the same plan, so it carries the same five tables after its four
// existing vec4 header rows. Its own helpers were the last per-address prefix
// loops in the file, and pageSlot alone evaluated three of them per channel.
const ACCURATE_LAYOUT_BYTES = 64 + 5 * PARAMS_LEVEL_TABLE_SLOTS * 4;
/** Per-row ghost-alias detection record: one channel mask plus 18 owners. */
const GHOST_SCRATCH_WORDS_PER_ROW = 20;
/** valid/rows/mismatch-generation header plus the per-row build fingerprint. */
const COMMITTED_INPUT_HEADER_WORDS = 4;
const COMMITTED_INPUT_WORDS_PER_ROW = 4;
/**
 * Explicit row bound; the constructor also fails closed on device limits.
 *
 * Nothing in the pyramid is O(rowCapacity) with a large constant: the sparse
 * hash arenas are `nextPowerOfTwo(min(rowCapacity * 16, 2 * domainCells))` per
 * level, so they saturate at the domain's own cardinality long before this
 * bound, and the only strictly per-row terms are the level row map
 * (`levelCount * rowCapacity`) and the transfer stride (`rowCapacity * 8`).
 * Row indices are compared against `CHANNEL_CODE_BASE` (0xffff0000), and every
 * capacity-shaped dispatch is already the two-dimensional `dispatchFor` form,
 * so a capacity of 2^20 rows stays inside both. The former 16,384 was a
 * bring-up guard, and because `WebGPUOctreeProjection` constructs this V-cycle
 * unconditionally as the Section 4.3 preconditioner it was the hard ceiling on
 * the whole power-octree method: any scene planning more than 16,384 pressure
 * rows -- roughly a 25-cubed liquid domain -- failed solver construction with
 * a RangeError before reaching a single dispatch. The device-limit check in
 * the constructor below remains the real fail-closed gate.
 */
export const SPGRID_MAXIMUM_ROW_CAPACITY = 1_048_576;
export const OCTREE_SPGRID_CAPTURE_CONTROL_WORD = Object.freeze({
  readyGeneration: 4,
  error: 8,
  sourceGeneration: 10,
} as const);

/**
 * The ABI of the per-epoch compiled Section 6.3 operator image
 * (`buildAccurateOperatorRows` writes it, `stageDirectTerm` reads it).
 *
 * `rowWords` u32 per row: word 0 is the row's page-resolution status (0, or
 * `pageUnresolved` when the row has no resolvable native page), word
 * `1 + channel` is either the absolute destination row index or a code at or
 * above `codeBase`. A code carries the primary report stage in its low byte and
 * pageSlot's own stage in the next byte, so the consumer replays exactly the
 * reports the inline walk raised, in the same per-lane order. A primary of
 * zero is the multigrid-only skip, which reported nothing.
 *
 * Row indices and codes cannot collide: `SPGRID_MAXIMUM_ROW_CAPACITY` is far
 * below `codeBase`, and the differential harness asserts it.
 */
/**
 * Words of compiled fine-adjoint image per row: eight fine aliases of a 2:1
 * coarse leaf times eighteen candidate directions. This is deliberately the
 * adjoint half of the `accurateTerms` staging layout, so an image index and a
 * staged index are the same arithmetic and the ordered fold is untouched.
 */
const ADJOINT_ROW_WORDS = 144;

export const OCTREE_SPGRID_OPERATOR_IMAGE = Object.freeze({
  rowWords: 19,
  adjointRowWords: ADJOINT_ROW_WORDS,
  adjointRowMask: 0xf_ffff,
  adjointChannelShift: 20,
  codeBase: 0xffff_0000,
  skip: 0,
  outOfDomain: 27,
  unresolvedSlot: 28,
  invalidOwner: 29,
  pageUnresolved: 31,
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
  // Exact dense cardinality of each level's domain. This is the immutable
  // geometry the V-cycle launch shape is authored from; it never depends on
  // how many of those cells a given generation happens to activate.
  const levelDomainCells = Array.from({ length: levelCount }, (_, level) => {
    const scale = 2 ** level;
    return options.dimensions
      .map((value) => Math.ceil(value / scale))
      .reduce((product, value) => product * value, 1);
  });
  const levelCapacities = Array.from({ length: levelCount }, (_, level) => {
    const domainCells = levelDomainCells[level]!;
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
  // Eighteen resolved neighbour slots per cell, channel-major exactly like the
  // eighteen coefficients they index. The setup builder already resolves every
  // one of them to accumulate the coefficient; publishing the slot it resolved
  // is what stops the recurring correction from re-deriving it per spoke.
  const stencilNeighbourWords = 18 * totalLevelSlots;
  const topologyBytes = (TOPOLOGY_HEADER_WORDS + rowMapWords + worklistWords
    + pageWorklistWords + pageDirectoryWords
    + transferWords + directoryWords + stencilNeighbourWords) * 4;
  const stateBytes = STATE_CHANNELS * totalLevelSlots * 4;
  const dispatchBytes = levelCount * DISPATCH_RECORD_BYTES_PER_LEVEL + DISPATCH_LIFECYCLE_BYTES;
  const levelDeltaBytes = levelCount * LEVEL_DELTA_WORDS * 4;
  const capturePageCount = Math.ceil(rowCapacity / CAPTURE_PAGE_ROWS);
  const capturePageStateBytes = (CAPTURE_PUBLICATION_WORDS + 4 * capturePageCount) * 4;
  return { rowCapacity, levelCount, levelStride, transferStride,
    levelDomainCells: Object.freeze(levelDomainCells),
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
    stencilNeighbourBytes: stencilNeighbourWords * 4,
    pageDirectoryBytes: pageDirectoryWords * 4, pageRecordWords: PAGE_RECORD_WORDS,
    capturePageStateBytes, capturePageCount,
    rowDispatch: dispatchFor(rowCapacity), slotDispatch: dispatchFor(levelStride), transferDispatch: dispatchFor(transferStride),
    brickDispatch: dispatchFor(brickCount) };
}

/** Pure decoder shared by the post-submit readback and deterministic tests. */
export function decodeOctreeSPGridHierarchyCensus(
  plan: OctreeSPGridVCyclePlan,
  flags: Uint32Array,
  dispatch: Uint32Array,
  gatedDispatch: Uint32Array,
): Readonly<OctreeSPGridHierarchyCensus> {
  const dispatchWords = plan.dispatchBytes / 4;
  if (flags.length < plan.totalLevelSlots
    || dispatch.length < dispatchWords || gatedDispatch.length < dispatchWords) {
    throw new RangeError("SPGrid hierarchy census buffers are shorter than the plan");
  }
  const levels = plan.levelCapacities.map((capacity, level) => {
    let occupied = 0, active = 0, ghost = 0, multigridOnly = 0, invalidClass = 0;
    const begin = plan.levelOffsets[level]!;
    for (let slot = 0; slot < capacity; slot += 1) {
      const cellFlags = flags[begin + slot]!;
      if (cellFlags === 0) continue;
      occupied += 1;
      const classes = Number((cellFlags & SPGRID_CELL_FLAG.active) !== 0)
        + Number((cellFlags & SPGRID_CELL_FLAG.ghost) !== 0)
        + Number((cellFlags & SPGRID_CELL_FLAG.multigridOnly) !== 0);
      if (classes !== 1) invalidClass += 1;
      if ((cellFlags & SPGRID_CELL_FLAG.active) !== 0) active += 1;
      if ((cellFlags & SPGRID_CELL_FLAG.ghost) !== 0) ghost += 1;
      if ((cellFlags & SPGRID_CELL_FLAG.multigridOnly) !== 0) multigridOnly += 1;
    }
    const record = level * DISPATCH_RECORD_WORDS_PER_LEVEL;
    return Object.freeze({ level, capacity, occupied, active, ghost, multigridOnly,
      invalidClass, publishedCount: dispatch[record]!,
      publishedTransferCount: dispatch[record + 1]!, publishedPageCount: dispatch[record + 8]!,
      selectedGroups: dispatch[record + 2]!, restrictionGroups: dispatch[record + 5]!,
      pageGroups: dispatch[record + 9]!, gatedSelectedGroups: gatedDispatch[record + 2]!,
      gatedRestrictionGroups: gatedDispatch[record + 5]!,
      gatedPageGroups: gatedDispatch[record + 9]! });
  });
  return Object.freeze({ levels: Object.freeze(levels) });
}

/** Largest 256-row SPGrid plan whose two bindable arenas fit the device. */
export function spgridRowCapacityForBindingLimit(
  dimensions: readonly [number, number, number],
  maximumBindingBytesValue: number,
  maximumRowsValue = SPGRID_MAXIMUM_ROW_CAPACITY,
  rowAlignment = 256,
): number {
  const maximumBindingBytes = positiveInteger(maximumBindingBytesValue, "SPGrid storage binding limit");
  const maximumRows = positiveInteger(maximumRowsValue, "SPGrid maximum rows");
  const alignment = positiveInteger(rowAlignment, "SPGrid row alignment");
  let low = 0;
  let high = Math.floor(Math.min(maximumRows, SPGRID_MAXIMUM_ROW_CAPACITY) / alignment);
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    const plan = planOctreeSPGridVCycle({ dimensions, rowCapacity: middle * alignment });
    if (Math.max(plan.stateBytes, plan.topologyBytes) <= maximumBindingBytes) low = middle;
    else high = middle - 1;
  }
  return low * alignment;
}

export type OctreeSPGridVCyclePipelineName = "beginL1CapturePlan"
  | "planL1CaptureDelta" | "reduceL1CaptureDelta"
  | "commitChangedL1" | "finalizeL1CapturePublication"
  | "probeCandidateSkip" | "applyCandidateSkip" | "publishCommittedInputs"
  | "prepareCandidateSchedules"
  | "clearCandidateLevels" | "buildCandidateLevelSets"
  | "detectCandidateGhosts" | "insertCandidateGhosts"
  | "buildCandidateLevelDeltas" | "countCandidateTransfers" | "scanCandidateTransfers"
  | "writeCandidateTransfers" | "linkCandidateParentChains"
  | "markCandidateBrickOccupancy" | "rankCandidateBricks" | "scatterCandidateRankedSlots"
  | "markCandidatePageOccupancy" | "compactCandidatePages" | "linkCandidatePageNeighbours"
  | "appendCandidateDirectoryIdentities" | "markCompactCandidateBrickOccupancy"
  | "rankCompactCandidateBricks" | "buildCompactCandidatePages"
  | "linkCompactCandidatePageNeighbours"
  | "buildCandidateStencils" | "publishCandidateSpectralBounds"
  | "validateCandidateHierarchy" | "commitCandidateLevels"
  | "commitCandidateTouchedBricks"
  | "finalizeLifecycle" | "prepareCorrectionDispatches" | "clearCorrection"
  | "zeroVectors" | "seedRhs" | "seedRhsAndClearCorrection"
  | "smoothChebyshevAtoB0" | "smoothChebyshevBtoA0"
  | "smoothChebyshevAtoB1" | "smoothChebyshevBtoA1"
  | "smoothChebyshevAtoB2" | "smoothChebyshevBtoA2"
  | "smoothChebyshevAtoB3" | "smoothChebyshevBtoA3"
  | "restrictAndGhostAccumulate" | "coarseVcycleTail" | "exactBottom"
  | "prolongAndGhostPropagate" | "publish";

const OCTREE_SPGRID_HIERARCHICAL_ONLY_PIPELINES = new Set<OctreeSPGridVCyclePipelineName>([
  "prepareCorrectionDispatches", "clearCorrection", "zeroVectors", "seedRhs",
  "seedRhsAndClearCorrection", "smoothChebyshevAtoB0", "smoothChebyshevBtoA0",
  "smoothChebyshevAtoB1", "smoothChebyshevBtoA1", "smoothChebyshevAtoB2",
  "smoothChebyshevBtoA2", "smoothChebyshevAtoB3", "smoothChebyshevBtoA3",
  "restrictAndGhostAccumulate", "coarseVcycleTail", "exactBottom",
  "prolongAndGhostPropagate", "publish",
]);

/** Pipeline reachability mirror used by construction and non-GPU tests. */
export function octreeSPGridPipelineNamesForExecutor(
  compileHierarchicalExecutor: boolean,
): readonly OctreeSPGridVCyclePipelineName[] {
  const names = Object.keys(OCTREE_SPGRID_VCYCLE_BINDINGS) as OctreeSPGridVCyclePipelineName[];
  return Object.freeze(compileHierarchicalExecutor
    ? names
    : names.filter((name) => !OCTREE_SPGRID_HIERARCHICAL_ONLY_PIPELINES.has(name)));
}

export const OCTREE_SPGRID_VCYCLE_BINDINGS: Readonly<Record<OctreeSPGridVCyclePipelineName, readonly number[]>> = Object.freeze({
  beginL1CapturePlan: [0, 3, 6, 13, 14, 17, 18],
  // The row scan no longer reaches the level-delta arena (14): markDirtyFrom
  // moved with the reduction below.
  planL1CaptureDelta: [0, 1, 3, 11, 13, 18, 20, 21],
  // The reduction dropped the row scan, so it dropped the four row-scan
  // bindings with it: captured geometry (1), source row geometry (11),
  // topology metrics (20) and the Section 6.3 catalog (21).
  reduceL1CaptureDelta: [0, 3, 13, 14, 18],
  commitChangedL1: [0, 1, 3, 11, 13, 18],
  finalizeL1CapturePublication: [0, 3, 13, 18],
  probeCandidateSkip: [0, 3, 11, 13, 20, 23],
  applyCandidateSkip: [0, 13, 14, 23],
  prepareCandidateSchedules: [0, 3, 6, 13, 14, 17, 26],
  publishCommittedInputs: [0, 1, 3, 7, 13, 20, 23],
  clearCandidateLevels: [0, 3, 6, 14, 15, 16, 17],
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
  appendCandidateDirectoryIdentities: [0, 14, 15, 16, 17, 25, 26],
  markCompactCandidateBrickOccupancy: [0, 13, 14, 15, 16, 27, 28],
  rankCompactCandidateBricks: [0, 13, 14, 15, 16, 17, 26, 27, 28, 29, 30],
  buildCompactCandidatePages: [0, 13, 14, 15, 17, 30, 31, 32],
  linkCompactCandidatePageNeighbours: [0, 13, 14, 15, 31, 32],
  buildCandidateStencils: [0, 3, 4, 5, 6, 7, 14, 15, 16, 17, 24],
  publishCandidateSpectralBounds: [0, 4, 6, 14, 15, 16, 17],
  validateCandidateHierarchy: [0, 6, 13, 14, 17],
  commitCandidateTouchedBricks: [0, 4, 13, 14, 15, 27, 28],
  commitCandidateLevels: [0, 3, 4, 5, 6, 13, 14, 15, 16, 17],
  finalizeLifecycle: [0, 3, 6, 7, 13],
  prepareCorrectionDispatches: [0, 3, 6, 7, 19],
  clearCorrection: [0, 3, 7, 9],
  zeroVectors: [0, 4, 5, 6, 7], seedRhs: [0, 3, 4, 5, 7, 8, 11],
  seedRhsAndClearCorrection: [0, 3, 4, 5, 7, 8, 9, 11],
  smoothChebyshevAtoB0: [0, 4, 5, 6, 7], smoothChebyshevBtoA0: [0, 4, 5, 6, 7],
  smoothChebyshevAtoB1: [0, 4, 5, 6, 7], smoothChebyshevBtoA1: [0, 4, 5, 6, 7],
  smoothChebyshevAtoB2: [0, 4, 5, 6, 7], smoothChebyshevBtoA2: [0, 4, 5, 6, 7],
  smoothChebyshevAtoB3: [0, 4, 5, 6, 7], smoothChebyshevBtoA3: [0, 4, 5, 6, 7],
  restrictAndGhostAccumulate: [0, 4, 5, 6, 7],
  coarseVcycleTail: [0, 4, 5, 6, 7],
  exactBottom: [0, 4, 5, 6, 7],
  prolongAndGhostPropagate: [0, 4, 5, 6, 7],
  publish: [0, 3, 4, 5, 7, 9, 11],
});

type CachedGroup = { rowCount: GPUBuffer; control: GPUBuffer; geometry: GPUBuffer;
  sourceControl: GPUBuffer; topologyMetrics: GPUBuffer;
  rhs?: GPUBuffer; correction?: GPUBuffer; group: GPUBindGroup };
type CachedAccurateApply = {
  readonly input: GPUBuffer;
  readonly output: GPUBuffer;
  readonly solverControl: GPUBuffer;
  readonly worksets: GPUBuffer;
  readonly worksetOffset: number;
  readonly worksetSize?: number;
  readonly worksetLayout: GPUBuffer;
  readonly gateGroup: GPUBindGroup;
  readonly termGroup: GPUBindGroup;
  readonly adjointGroup: GPUBindGroup;
  readonly finalizeGroup: GPUBindGroup;
  readonly residualSource?: GPUBuffer;
  readonly residualFinalizeGroup?: GPUBindGroup;
  mergedTermGroup?: GPUBindGroup;
  mergedAdjointGroup?: GPUBindGroup;
};

/**
 * Byte offset of the fifth accurate indirect record: the union of the five
 * accepted row classes, which the staged row finalizer consumes as one dispatch.
 */
const ACCURATE_UNION_DISPATCH_OFFSET_BYTES = 4 * 12;
const ACCURATE_TERM_DISPATCH_OFFSET_BYTES = 5 * 12;
const ACCURATE_ADJOINT_DISPATCH_OFFSET_BYTES = 6 * 12;
const ACCURATE_SOURCE_IMAGE_ROW_RECORD_BYTES = 5 * 12;
const ACCURATE_SOURCE_IMAGE_TRANSITION_RECORD_BYTES = 6 * 12;

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
  private accurateOperatorInstance?: OctreeWorksetLinearOperator;
  /** Accurate second-order executor over five disjoint row classes. */
  get accurateOperator(): OctreeWorksetLinearOperator {
    if (!this.accurateOperatorInstance) {
      throw new Error("SPGrid accurate A2 executor was not compiled");
    }
    return this.accurateOperatorInstance;
  }
  readonly smootherContract;
  readonly diagnostics: Readonly<{ levelCount: number; coarsestCapacity: number; maximumTransferRecordsPerLevel: number;
    correctionDispatchCount: number; correctionInitializationDispatchCount: 1 | 2;
    correctionPassTransitions: number; restrictionScatterDispatchCount: number;
    restrictionAtomicAddUpperBound: number; parentGatherDispatchCount: number; parentGatherAtomicAddCount: 0;
    bottomOperation: "exact-single-cell"; coarsestDegreesOfFreedom: 1;
    directoryLookup: "brick-mask-rank"; lookupProbeUpperBound: 1; directoryBytes: number;
    directoryBrickCount: number; directoryBuildDispatchCount: number;
    pageAdjacency: "physical-27"; smootherLookup: "published-column-index" }>;
  private readonly capturedGeometry: GPUBuffer;
  private readonly candidateCapturedGeometry: GPUBuffer;
  private readonly topology: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly dispatchMeta: GPUBuffer;
  private readonly indirectDispatch: GPUBuffer;
  private readonly capturePageState: GPUBuffer;
  /** Read by the step-snapshot ring (POWER_LIQUIDS_ULTIMATE_M1MAX A4). Copy
   * source only; no caller may bind or write it. */
  readonly levelDelta: GPUBuffer;
  private readonly candidateTopology: GPUBuffer;
  private readonly candidateState: GPUBuffer;
  private readonly candidateDispatch: GPUBuffer;
  private readonly candidateIndirect: GPUBuffer;
  /** Row-parallel ghost-alias detection record consumed by the ordered insert. */
  private readonly candidateGhosts: GPUBuffer;
  /** Exact fingerprint of the inputs the last committed hierarchy consumed. */
  private readonly committedInputs: GPUBuffer;
  private readonly touchedDirectoryEnabled = spgridTouchedRadixSortEnabled();
  private readonly touchedDirectoryTripwire = this.touchedDirectoryEnabled
    && typeof process !== "undefined"
    && process.env?.FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE === "1";
  private readonly touchedBrickHeader: GPUBuffer;
  private readonly touchedPageHeader: GPUBuffer;
  private readonly touchedDirectoryDummy: GPUBuffer;
  private readonly touchedBrickSort?: WebGPURadixSortU32;
  private readonly touchedPageSort?: WebGPURadixSortU32;
  /**
   * Leading levels the correction runs as wide per-level dispatches; the rest
   * join the one-workgroup tail. Authored from immutable level geometry, so
   * the command graph never depends on a readback or a previous advance.
   */
  private get vcycleParallelLevels(): number {
    return planOctreeVCycleParallelLevels(this.plan.levelDomainCells);
  }
  private readonly accurateWorksetLayout: GPUBuffer;
  private accurateGatePipeline!: GPUComputePipeline;
  private readonly accurateClassDispatch!: GPUBuffer;
  private accurateMergedTermPipeline!: GPUComputePipeline;
  private accurateMergedAdjointPipeline!: GPUComputePipeline;
  private accurateTermPipeline!: GPUComputePipeline;
  private accurateAdjointPipeline!: GPUComputePipeline;
  private accurateFinalizePipeline!: GPUComputePipeline;
  private accurateResidualFinalizePipeline!: GPUComputePipeline;
  private readonly accurateTerms!: GPUBuffer;
  /** Per-epoch compiled operator image: 19 u32 per row, indices only. */
  private readonly accurateOperatorRows!: GPUBuffer;
  private accurateOperatorRowsPipeline!: GPUComputePipeline;
  private accurateOperatorRowsGroup!: GPUBindGroup;
  private readonly accurateImageDeltaParams!: GPUBuffer;
  private readonly accurateImageEpochs!: GPUBuffer;
  /**
   * Measurement-only control arm for the direct half of the operator image.
   * The shader retains the exact pre-image chase for the differential harness;
   * selecting it here prices address compilation without changing the build,
   * topology, coefficients, term staging, or fold order. Production is the
   * compiled image.
   */
  private readonly directByChase = typeof process !== "undefined"
    && process.env?.FLUID_SPGRID_DIRECT_BY_CHASE === "1";
  /**
   * Measurement-only control arm: run the merged-band adjoint stage through
   * the pre-image chase instead of the compiled image. Set
   * `FLUID_SPGRID_ADJOINT_BY_CHASE=1` to select it. Production is the image.
   */
  private readonly adjointByChase = typeof process !== "undefined"
    && process.env?.FLUID_SPGRID_ADJOINT_BY_CHASE === "1";
  /** Per-epoch compiled fine-adjoint image: 144 u32 per row, indices only. */
  private readonly accurateAdjointRows!: GPUBuffer;
  private accurateAdjointRowsPipeline!: GPUComputePipeline;
  private accurateAdjointRowsGroup!: GPUBindGroup;
  private accurateModule?: GPUShaderModule;
  private accurateGateModule?: GPUShaderModule;
  private readonly persistentImageCarry = typeof process !== "undefined"
    && process.env?.FLUID_SPGRID_PERSISTENT_IMAGES === "1";
  private readonly accurateBindings: CachedAccurateApply[] = [];
  private readonly params: readonly GPUBuffer[];
  private readonly candidateParams: readonly GPUBuffer[];
  private setupShaderModule!: GPUShaderModule;
  private pipelines: Readonly<Partial<Record<
    OctreeSPGridVCyclePipelineName, GPUComputePipeline
  >>> = Object.freeze({});
  private pipelineInitialization?: Promise<void>;
  private readonly groups = new Map<string, CachedGroup>();
  private readonly pre: number;
  private readonly post: number;
  private readonly hierarchicalExecutorCompiled: boolean;
  private lastSetupInput?: { readonly solverControl: GPUBuffer; readonly rowCount: GPUBuffer };
  private preparedCaptureSource?: OctreeSPGridSetupSource;
  private candidateSetupInput?: OctreeSPGridSetupSource;
  /** Diagnostic control arm for proving whether a live-row launch changes
   * hierarchy bytes. Production always consumes the GPU-authored schedule. */
  private readonly capacityRowScheduleOracle = typeof process !== "undefined"
    && process.env?.FLUID_SPGRID_ROW_SCHEDULE_ORACLE === "capacity";
  private destroyed = false;

  /** GPU-authored controls suitable for the runtime's diagnostics readback. */
  get workAccountingBuffers(): Readonly<{
    dispatch: GPUBuffer; capture: GPUBuffer; accurateClassDispatch: GPUBuffer;
  }> {
    if (!this.hierarchicalExecutorCompiled) {
      throw new Error("SPGrid hierarchical work accounting is not compiled");
    }
    return Object.freeze({ dispatch: this.dispatchMeta, capture: this.capturePageState,
      accurateClassDispatch: this.accurateClassDispatch });
  }

  /** Candidate hierarchy/capture report consumed by the coupled epoch reduction. */
  get candidateControl(): GPUBuffer { return this.capturePageState; }

  /** Diagnostics-only immutable snapshot of the accepted M1 publication.
   * Callers may inspect exact D4/epoch invariants, but no returned word feeds
   * scheduling or a later GPU submission. */
  async readPublishedHierarchyForDiagnostics(): Promise<Readonly<{
    plan: OctreeSPGridVCyclePlan;
    state: Uint32Array;
    topology: Uint32Array;
    dispatch: Uint32Array;
  }>> {
    this.assertLive();
    const stateBytes = this.state.size, topologyBytes = this.topology.size;
    const dispatchBytes = this.dispatchMeta.size;
    const readback = this.device.createBuffer({
      label: "SPGrid accepted hierarchy diagnostic readback",
      size: stateBytes + topologyBytes + dispatchBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read SPGrid accepted hierarchy diagnostics",
    });
    encoder.copyBufferToBuffer(this.state, 0, readback, 0, stateBytes);
    encoder.copyBufferToBuffer(this.topology, 0, readback, stateBytes, topologyBytes);
    encoder.copyBufferToBuffer(this.dispatchMeta, 0, readback,
      stateBytes + topologyBytes, dispatchBytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange();
      return Object.freeze({
        plan: this.plan,
        state: Uint32Array.from(new Uint32Array(mapped, 0, stateBytes / 4)),
        topology: Uint32Array.from(new Uint32Array(mapped, stateBytes, topologyBytes / 4)),
        dispatch: Uint32Array.from(new Uint32Array(mapped, stateBytes + topologyBytes,
          dispatchBytes / 4)),
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Failure-only exact candidate state. This is never used for scheduling;
   * it identifies the first level whose compact hierarchy transaction
   * rejected without widening the recurring snapshot ring. */
  async readCandidateFailureDiagnostics(): Promise<Readonly<{
    levelDelta: readonly number[]; candidateDispatch: readonly number[];
  }>> {
    this.assertLive();
    const dispatchBytes = this.plan.dispatchBytes + CANDIDATE_SCHEDULE_BYTES;
    const readback = this.device.createBuffer({
      label: "SPGrid candidate failure diagnostics",
      size: this.plan.levelDeltaBytes + dispatchBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read SPGrid candidate failure diagnostics" });
    encoder.copyBufferToBuffer(this.levelDelta, 0, readback, 0, this.plan.levelDeltaBytes);
    encoder.copyBufferToBuffer(this.candidateDispatch, 0, readback,
      this.plan.levelDeltaBytes, dispatchBytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const deltaWords = this.plan.levelDeltaBytes / 4;
      return Object.freeze({
        levelDelta: Object.freeze(Array.from(words.slice(0, deltaWords))),
        candidateDispatch: Object.freeze(Array.from(words.slice(deltaWords))),
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  async readTouchedDirectoryTripwireDiagnostics(): Promise<Readonly<{
    enabled: boolean; active: boolean; brickHeader: readonly number[]; pageHeader: readonly number[];
    brickControl: readonly number[]; pageControl: readonly number[];
  }>> {
    this.assertLive();
    if (!this.touchedDirectoryEnabled) return Object.freeze({ enabled: false, active: false,
      brickHeader: Object.freeze([]), pageHeader: Object.freeze([]),
      brickControl: Object.freeze([]), pageControl: Object.freeze([]) });
    const readback = this.device.createBuffer({ label: "SPGrid touched-directory tripwire readback",
      size: 96, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read SPGrid touched-directory tripwire" });
    encoder.copyBufferToBuffer(this.touchedBrickHeader, 0, readback, 0, 16);
    encoder.copyBufferToBuffer(this.touchedPageHeader, 0, readback, 16, 16);
    encoder.copyBufferToBuffer(this.touchedBrickSort!.control, 0, readback, 32, 32);
    encoder.copyBufferToBuffer(this.touchedPageSort!.control, 0, readback, 64, 32);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return Object.freeze({ enabled: true, active: this.touchedDirectoryTripwire,
        brickHeader: Object.freeze(Array.from(words.slice(0, 4))),
        pageHeader: Object.freeze(Array.from(words.slice(4, 8))),
        brickControl: Object.freeze(Array.from(words.slice(8, 16))),
        pageControl: Object.freeze(Array.from(words.slice(16, 24))) });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

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

  /**
   * Diagnostic-only census of the committed sparse hierarchy. This copies the
   * FLAGS channel rather than the full 26-channel state arena, so a pressure
   * profile can distinguish useful ACTIVE/GHOST/MG_ONLY work from capacity
   * padding without perturbing the recurring command graph.
   */
  async readHierarchyCensus(): Promise<Readonly<OctreeSPGridHierarchyCensus>> {
    this.assertLive();
    const flagBytes = this.plan.totalLevelSlots * 4;
    const dispatchBytes = this.plan.dispatchBytes;
    const readback = this.device.createBuffer({
      label: "SPGrid hierarchy FLAGS/dispatch census readback",
      size: flagBytes + 2 * dispatchBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read SPGrid hierarchy FLAGS/dispatch census",
    });
    // FLAGS is channel one in the channel-major state arena.
    encoder.copyBufferToBuffer(this.state, flagBytes, readback, 0, flagBytes);
    encoder.copyBufferToBuffer(this.dispatchMeta, 0, readback, flagBytes, dispatchBytes);
    encoder.copyBufferToBuffer(
      this.indirectDispatch, 0, readback, flagBytes + dispatchBytes, dispatchBytes,
    );
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const dispatchBase = this.plan.totalLevelSlots, gatedBase = dispatchBase + dispatchBytes / 4;
      return decodeOctreeSPGridHierarchyCensus(
        this.plan,
        words.subarray(0, this.plan.totalLevelSlots),
        words.subarray(dispatchBase, gatedBase),
        words.subarray(gatedBase, gatedBase + dispatchBytes / 4),
      );
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  constructor(private readonly device: GPUDevice, private readonly source: OctreeSPGridVCycleSource,
    options: OctreeSPGridVCycleOptions) {
    this.plan = planOctreeSPGridVCycle(options);
    this.hierarchicalExecutorCompiled = options.compileHierarchicalExecutor !== false;
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
    const deltaCountControlWord = delta.dirtyCountControlWord ?? 5;
    if (!Number.isSafeInteger(delta.rowCapacity) || delta.rowCapacity !== this.plan.rowCapacity
      || (deltaCountControlWord !== 5 && deltaCountControlWord !== 6)
      || ![delta.controlOffsetWords, delta.newToOldOffsetWords, delta.oldToNewOffsetWords,
        delta.dirtyRowsOffsetWords]
        .every((offset) => Number.isSafeInteger(offset) && offset >= 0)
      || delta.newToOldOffsetWords < delta.controlOffsetWords + 16
      || delta.oldToNewOffsetWords < delta.newToOldOffsetWords + delta.rowCapacity
      || delta.dirtyRowsOffsetWords < delta.oldToNewOffsetWords + delta.rowCapacity
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
    this.topology = device.createBuffer({ label: "SPGrid native sparse topology/worklists/transfers",
      size: this.plan.topologyBytes, usage: storage });
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
    this.candidateDispatch = device.createBuffer({ label: "SPGrid candidate counts and live schedules",
      size: this.plan.dispatchBytes + CANDIDATE_SCHEDULE_BYTES,
      usage: storage | GPUBufferUsage.COPY_SRC });
    this.candidateIndirect = device.createBuffer({ label: "SPGrid candidate live indirect schedules",
      size: CANDIDATE_SCHEDULE_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.candidateGhosts = device.createBuffer({ label: "SPGrid candidate ghost-alias detection records",
      size: this.plan.rowCapacity * GHOST_SCRATCH_WORDS_PER_ROW * 4, usage: storage });
    this.committedInputs = device.createBuffer({ label: "SPGrid committed hierarchy input fingerprint",
      size: (COMMITTED_INPUT_HEADER_WORDS
        + this.plan.rowCapacity * COMMITTED_INPUT_WORDS_PER_ROW) * 4, usage: storage });
    const touchedHeaderUsage = storage;
    this.touchedBrickHeader = device.createBuffer({ label: "SPGrid touched-brick sort header",
      size: 16, usage: touchedHeaderUsage });
    this.touchedPageHeader = device.createBuffer({ label: "SPGrid touched-page sort header",
      size: 16, usage: touchedHeaderUsage });
    this.touchedDirectoryDummy = device.createBuffer({ label: "SPGrid disabled touched-directory binding",
      size: 32, usage: storage });
    if (this.touchedDirectoryEnabled) {
      this.touchedBrickSort = new WebGPURadixSortU32(device, 7 * this.plan.totalLevelSlots,
        this.touchedBrickHeader);
      this.touchedPageSort = new WebGPURadixSortU32(device, this.plan.totalLevelSlots,
        this.touchedPageHeader);
    }
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
    this.accurateWorksetLayout = device.createBuffer({
      label: "SPGrid Section 6.3 page operator layout", size: ACCURATE_LAYOUT_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const accurateLayout = new ArrayBuffer(ACCURATE_LAYOUT_BYTES);
    const accurateWords = new Uint32Array(accurateLayout);
    accurateWords.set([source.worksetStrideWords ?? this.plan.rowCapacity + 7,
      source.worksetBankStrideWords ?? 4 * (this.plan.rowCapacity + 7), 0, 0,
      options.dimensions[0], options.dimensions[1], options.dimensions[2], this.plan.rowCapacity,
      this.plan.levelCount, this.plan.levelStride, this.plan.totalLevelSlots,
      source.coefficientBankStrideWords]);
    // Words 12-15 stay the unread `numerics` padding row, so the four header
    // rows every existing accessor reads keep their byte offsets exactly.
    accurateWords.set(levelCaps, 16);
    accurateWords.set(levelBases, 16 + PARAMS_LEVEL_TABLE_SLOTS);
    accurateWords.set(brickOffsets, 16 + 2 * PARAMS_LEVEL_TABLE_SLOTS);
    accurateWords.set(pageOffsets, 16 + 3 * PARAMS_LEVEL_TABLE_SLOTS);
    accurateWords.set(transferOffsets, 16 + 4 * PARAMS_LEVEL_TABLE_SLOTS);
    device.queue.writeBuffer(this.accurateWorksetLayout, 0, accurateLayout);
    const makeParams = (level: number, sourceMode: OctreeSPGridSourceMode) => {
      const buffer = device.createBuffer({
        label: `SPGrid level ${level} ${sourceMode} parameters`, size: PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(PARAMS_BYTES / 4), floats = new Float32Array(words.buffer);
      words.set([options.dimensions[0], options.dimensions[1], options.dimensions[2], level,
        this.plan.rowCapacity, this.plan.levelCount, this.plan.levelStride, this.plan.transferStride,
        this.plan.rowDispatch[0], this.plan.slotDispatch[0], this.plan.transferDispatch[0], this.pre]);
      words[12] = this.post;
      // Low bit selects the accepted or candidate source authority.
      words[13] = sourceMode === "candidate" ? 1 : 0;
      if (this.touchedDirectoryEnabled) words[13] |= 2;
      if (this.touchedDirectoryTripwire) words[13] |= 4;
      floats[14] = 1;
      floats[15] = options.finestCellWidth;
      // The high bit is internal parameter metadata, not a buffer offset. It
      // selects control[6] for an affected-list source while preserving the
      // compact four-word shader ABI.
      const encodedDirtyRowsOffset = delta.dirtyRowsOffsetWords
        | (deltaCountControlWord === 6 ? 0x80000000 : 0);
      words.set([delta.rowCapacity, delta.controlOffsetWords, delta.newToOldOffsetWords,
        encodedDirtyRowsOffset], 16);
      words.set([this.plan.totalLevelSlots, this.plan.brickCount,
        this.plan.pageDirectoryBytes / 4, 0], 20);
      floats[23] = 0;
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
    if (this.hierarchicalExecutorCompiled) {
    this.accurateClassDispatch = device.createBuffer({
      label: "SPGrid accurate A2 convergence-gated class records",
      // Four per-class records, then the union record the single accepted-row
      // dispatch consumes. The class records stay published so the four-way
      // encode remains a one-line A/B against the union encode.
      // Six trailing high-water diagnostics preserve the first live gate even
      // after fail-closed tail gates zero the executable records.
      size: ACCURATE_ADJOINT_DISPATCH_OFFSET_BYTES + 12 + 8 * 4,
      usage: storage | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.accurateClassDispatch, 0, new Uint32Array(29));
    this.accurateTerms = device.createBuffer({
      label: "SPGrid accurate A2 staged direct terms and compact row map",
      // Eighteen direct terms and 8×18 fine-adjoint candidate terms per row,
      // followed by one compact union-row id per row and one count word.
      size: (this.plan.rowCapacity * 163 + 1) * 4,
      usage: storage,
    });
    // Exact cross-generation carry remains an opt-in discovery arm. Its
    // old->new destination remap was slower than rebuilding compact live rows
    // on the large lane, so production keeps one bank and pays no dormant
    // double-image memory/cache cost.
    const imageBanks = this.persistentImageCarry ? 2 : 1;
    this.accurateOperatorRows = device.createBuffer({
      label: "SPGrid accurate A2 compiled operator rows",
      // One page-status word plus eighteen resolved destination rows per row.
      // At the mini lane's live count this is ~113 KB; it is sized off the
      // planned row capacity so a topology growth cannot outrun it.
      size: imageBanks * this.plan.rowCapacity * 19 * 4,
      usage: storage,
    });
    this.accurateAdjointRows = device.createBuffer({
      label: "SPGrid accurate A2 compiled fine-adjoint rows",
      // One word per (child, candidate) lane of the adjoint staging layout, so
      // the image index and the accurateTerms index are the same arithmetic.
      // This is 144 of the 163 words per row accurateTerms already carries, and
      // it is sized off the same planned row capacity for the same reason.
      size: imageBanks * this.plan.rowCapacity * ADJOINT_ROW_WORDS * 4,
      usage: storage,
    });
    this.accurateImageDeltaParams = device.createBuffer({
      label: "SPGrid accurate A2 persistent-image delta layout",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.accurateImageDeltaParams, 0, new Uint32Array([
      delta.controlOffsetWords, delta.newToOldOffsetWords,
      delta.oldToNewOffsetWords, deltaCountControlWord,
    ]));
    this.accurateImageEpochs = device.createBuffer({
      label: "SPGrid accurate A2 persistent-image epoch stamps",
      size: 16,
      usage: GPUBufferUsage.STORAGE,
    });
    // Interleaved A/B arm for the compiled fine-adjoint image. Both entry
    // points stage the same words from the same expression; they differ only in
    // where the destination row comes from, so selecting between them in ONE
    // build is the only way to price the image without rebuilding the tree
    // between samples. Production is the image; the chase is the control.
    this.accurateOperatorInstance = Object.freeze({
      convergenceTail: "gpu-zero-indirect" as const,
      encodedDispatchCount: 4,
      encodedResidualDispatchCount: 4,
      encodedMergedBandDispatchCount: 3,
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
      encodeGate: (pass: GPUComputePassEncoder, input: GPUBuffer, output: GPUBuffer,
        solverControl: GPUBuffer) => {
        const canonical = this.source.worksets.regularInterior;
        const binding = "buffer" in canonical ? canonical : { buffer: canonical };
        this.encodeAccurateGate(pass, input, output, solverControl,
          { buffer: binding.buffer }, this.accurateWorksetLayout);
      },
      encodeBody: (broker: PassBroker, input: GPUBuffer, output: GPUBuffer,
        solverControl: GPUBuffer) => {
        const canonical = this.source.worksets.regularInterior;
        const binding = "buffer" in canonical ? canonical : { buffer: canonical };
        this.encodeAccurateBody(broker, input, output, solverControl,
          { buffer: binding.buffer }, this.accurateWorksetLayout);
      },
      encodeResidualBody: (broker: PassBroker, input: GPUBuffer, residualRhs: GPUBuffer,
        residual: GPUBuffer, solverControl: GPUBuffer) => {
        const canonical = this.source.worksets.regularInterior;
        const binding = "buffer" in canonical ? canonical : { buffer: canonical };
        this.encodeAccurateResidualBody(broker, input, residualRhs, residual, solverControl,
          { buffer: binding.buffer }, this.accurateWorksetLayout);
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
    }
    const l = this.plan.levelCount;
    // Three exact L1-capture dispatches (plan, the page-parallel row scan, and
    // its work-list reduction), the two-dispatch unchanged-input skip probe,
    // the changed-page commit, thirteen data-parallel candidate phases
    // separated by dispatch order rather than by statement order in one thread,
    // the candidate validator, five transactional publication dispatches, and
    // and the two once-per-epoch image compiles (direct, fine-adjoint) that
    // ride the same pass.
    // Keep this exact for command and active/scheduled accounting.
    this.encodedSetupDispatchCount = 3 + 2 + 2
      + 17 - (this.touchedDirectoryEnabled && !this.touchedDirectoryTripwire ? 6 : 0) + 1 + 5
      + (this.touchedDirectoryEnabled ? 36 : 0)
      + (this.hierarchicalExecutorCompiled ? 2 : 0);
    // The widest levels retain fully parallel, globally synchronized Jacobi
    // dispatches. Only the levels that fit in one workgroup join the tail,
    // whose barriers then provide identical phase ordering without serializing
    // a level's restriction parent-chain walk one coarse cell at a time.
    const parallelLevels = this.vcycleParallelLevels;
    const correctionInitializationDispatchCount = 2 as const;
    const sparseCorrectionDispatchCount = l + 3 + correctionInitializationDispatchCount
      + parallelLevels * (this.pre + this.post + 2);
    this.encodedCorrectionDispatchCount = this.hierarchicalExecutorCompiled
      ? sparseCorrectionDispatchCount : 0;
    this.diagnostics = Object.freeze({ levelCount: l,
      coarsestCapacity: this.plan.levelCapacities[this.plan.levelCount - 1],
      maximumTransferRecordsPerLevel: Math.max(...this.plan.transferCapacities),
      correctionDispatchCount: this.encodedCorrectionDispatchCount,
      correctionInitializationDispatchCount,
      correctionPassTransitions: 1, restrictionScatterDispatchCount: 0,
      restrictionAtomicAddUpperBound: 0,
      parentGatherDispatchCount: l - 1, parentGatherAtomicAddCount: 0 as const,
      bottomOperation: "exact-single-cell" as const, coarsestDegreesOfFreedom: 1 as const,
      directoryLookup: "brick-mask-rank" as const, lookupProbeUpperBound: 1 as const,
      directoryBytes: this.plan.directoryBytes, directoryBrickCount: this.plan.brickCount,
      directoryBuildDispatchCount: 1,
      pageAdjacency: "physical-27" as const,
      smootherLookup: "published-column-index" as const });
    this.allocatedBytes = this.plan.allocatedBytes + this.capturedGeometry.size + this.candidateCapturedGeometry.size
      + this.candidateTopology.size + this.candidateState.size + this.candidateDispatch.size
      + this.candidateIndirect.size
      + this.candidateGhosts.size + this.committedInputs.size
      + this.touchedBrickHeader.size + this.touchedPageHeader.size + this.touchedDirectoryDummy.size
      + (this.touchedBrickSort?.allocatedBytes ?? 0) + (this.touchedPageSort?.allocatedBytes ?? 0)
      + this.accurateWorksetLayout.size + (this.accurateClassDispatch?.size ?? 0)
      + (this.accurateOperatorRows?.size ?? 0) + (this.accurateAdjointRows?.size ?? 0)
      + (this.accurateImageDeltaParams?.size ?? 0) + (this.accurateImageEpochs?.size ?? 0)
      + this.candidateParams.reduce((bytes, buffer) => bytes + buffer.size, 0);
  }

  private pipelineDescriptor(
    entryPoint: OctreeSPGridVCyclePipelineName,
  ): GPUComputePipelineDescriptor {
    return {
      label: `SPGrid V-cycle · ${entryPoint}`,
      layout: "auto",
      compute: { module: this.setupShaderModule, entryPoint },
    };
  }

  private async initializeAccuratePipelines(): Promise<void> {
    if (!this.hierarchicalExecutorCompiled || this.accurateGatePipeline) return;
    this.accurateModule = this.device.createShaderModule({
      label: "SPGrid accurate A2 class-specialized row apply", code: octreeSPGridAccurateOperatorShader,
    });
    this.accurateGateModule = this.device.createShaderModule({
      label: "SPGrid accurate A2 convergence-tail publisher",
      code: octreeSPGridAccurateDispatchGateShader,
    });
    const shaderModule = this.accurateModule!;
    const make = (label: string, entryPoint: string, constants?: Record<string, number>) =>
      this.device.createComputePipelineAsync({
        label, layout: "auto", compute: { module: shaderModule, entryPoint, constants },
      });
    this.accurateGatePipeline = await this.device.createComputePipelineAsync({
      label: "SPGrid accurate A2 · convergence-tail publisher", layout: "auto",
      compute: { module: this.accurateGateModule!, entryPoint: "prepareAccurateDispatches" },
    });
    this.accurateMergedTermPipeline = await make(
      "SPGrid Section 6.3 · parallel merged-band direct terms",
      this.directByChase ? "stageMergedBandTermsByChase" : "stageMergedBandTerms");
    this.accurateMergedAdjointPipeline = await make(
      "SPGrid Section 6.3 · parallel merged-band adjoint children",
      this.adjointByChase ? "stageMergedBandAdjointsByChase" : "stageMergedBandAdjoints");
    this.accurateTermPipeline = await make("SPGrid accurate A2 · parallel direct terms",
      this.directByChase ? "stageAcceptedUnionTermsByChase" : "stageAcceptedUnionTerms");
    this.accurateAdjointPipeline = await make(
      "SPGrid accurate A2 · parallel fine-adjoint children", "stageAcceptedUnionAdjoints");
    this.accurateFinalizePipeline = await make(
      "SPGrid accurate A2 · ordered row fold", "finalizeStagedUnionRows");
    this.accurateResidualFinalizePipeline = await make(
      "SPGrid accurate A2 · ordered row fold to residual", "finalizeStagedUnionResidualRows");
    const constants = { persistentImageCarry: this.persistentImageCarry ? 1 : 0 };
    this.accurateOperatorRowsPipeline = await make(
      "SPGrid accurate A2 · compile operator rows", "buildAccurateOperatorRows", constants);
    this.accurateAdjointRowsPipeline = await make(
      "SPGrid accurate A2 · compile fine-adjoint rows", "buildAccurateAdjointRows", constants);
    const common = [
      { binding: 6, resource: { buffer: this.topology } },
      { binding: 3, resource: { buffer: this.source.control } },
      { binding: 4, resource: { buffer: "buffer" in this.source.worksets.regularInterior
        ? this.source.worksets.regularInterior.buffer : this.source.worksets.regularInterior } },
      { binding: 5, resource: { buffer: this.accurateWorksetLayout } },
      { binding: 7, resource: { buffer: this.state } },
      { binding: 8, resource: { buffer: this.capturedGeometry } },
      { binding: 9, resource: { buffer: this.source.topologyMetrics } },
      { binding: 11, resource: { buffer: this.accurateWorksetLayout } },
    ];
    const tail = [
      { binding: 15, resource: { buffer: this.accurateImageDeltaParams } },
      { binding: 16, resource: { buffer: this.source.rowDelta.rows } },
      { binding: 17, resource: { buffer: this.accurateImageEpochs } },
    ];
    this.accurateOperatorRowsGroup = this.device.createBindGroup({
      label: "SPGrid accurate A2 · operator image bindings",
      layout: this.accurateOperatorRowsPipeline.getBindGroupLayout(0),
      entries: [...common, { binding: 13, resource: { buffer: this.accurateOperatorRows } }, ...tail],
    });
    this.accurateAdjointRowsGroup = this.device.createBindGroup({
      label: "SPGrid accurate A2 · fine-adjoint image bindings",
      layout: this.accurateAdjointRowsPipeline.getBindGroupLayout(0),
      entries: [...common, { binding: 14, resource: { buffer: this.accurateAdjointRows } }, ...tail],
    });
  }

  /** Sequential compilation keeps driver pressure bounded during startup. */
  async initializePipelines(
    onProgress: (label: string, completed: number, total: number) => void = () => {},
  ): Promise<void> {
    this.assertLive();
    await this.touchedBrickSort?.initializePipelines();
    await this.touchedPageSort?.initializePipelines();
    const names = octreeSPGridPipelineNamesForExecutor(this.hierarchicalExecutorCompiled);
    if (names.every((name) => this.pipelines[name] !== undefined)) return;
    if (!this.pipelineInitialization) {
      this.pipelineInitialization = (async () => {
        this.setupShaderModule = this.device.createShaderModule({
          label: "Paper native sparse SPGrid V-cycle", code: octreeSPGridVCycleShader,
        });
        const compiled: Partial<Record<OctreeSPGridVCyclePipelineName, GPUComputePipeline>> = {};
        for (let index = 0; index < names.length; index += 1) {
          const entryPoint = names[index];
          const label = `SPGrid V-cycle · ${entryPoint}`;
          onProgress(label, index, names.length);
          compiled[entryPoint] = await this.device.createComputePipelineAsync(
            this.pipelineDescriptor(entryPoint),
          );
          this.assertLive();
          onProgress(label, index + 1, names.length);
        }
        // Setup bind groups are keyed by the caller-supplied solve control,
        // row-count authority and candidate/accepted source mode. Publish the
        // complete pipeline map atomically; `bind` then materializes and caches
        // each dependent group on its first encode with those live resources.
        this.groups.clear();
        this.pipelines = Object.freeze(compiled);
      })();
    }
    await this.pipelineInitialization;
    await this.initializeAccuratePipelines();
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
    broker.updateIndirectBuffer(this.candidateDispatch, this.plan.dispatchBytes,
      this.candidateIndirect, CANDIDATE_SCHEDULE.capture, 12);
    // Label isolation only; see encodeCorrection. Production returns this same
    // open pass and drops the label.
    //
    // One workgroup per work item, sixty-four lanes per page row. Most
    // workgroups exit on the work-count or uniqueness test after two loads;
    // the ones that survive are the distinct dirty pages, and they now run
    // concurrently across the machine instead of down the lanes of a single
    // workgroup. Measured at 6.40 ms/advance over two calls before this split.
    this.runCandidateIndirect(broker.compute({ label: "SPGrid V-cycle · capture plan L1 delta" }),
      "planL1CaptureDelta", 0, input, 0);
    this.run(broker.compute({ label: "SPGrid V-cycle · capture reduce L1 delta" }),
      "reduceL1CaptureDelta", 0, input, [1, 1, 1]);
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
    this.runLiveRows(pass, "probeCandidateSkip", 0, input);
    const geometry = this.candidateCapturedGeometry;
    // Label isolation only; see encodeCorrection. Every `staged()` below returns
    // the pass opened above when `FLUID_GPU_ISOLATE_PASS_LABELS` is off, and the
    // label is discarded, so production encodes the identical single pass.
    // With isolation on, each of the twenty-one candidate phases brackets its
    // own dispatch -- the only way to see which phase holds the rebuild's time.
    const staged = (label: string): GPUComputePassEncoder =>
      broker.compute({ label: `SPGrid V-cycle · candidate ${label}` });
    this.run(staged("apply skip"), "applyCandidateSkip", 0, input, [1, 1, 1]);
    this.run(staged("prepare live schedules"), "prepareCandidateSchedules", 0, input,
      [1, 1, 1], geometry);
    broker.updateIndirectBuffer(this.candidateDispatch, this.plan.dispatchBytes,
      this.candidateIndirect, 0, CANDIDATE_SCHEDULE_BYTES);
    this.runCandidateIndirect(staged("commit changed L1"), "commitChangedL1", 0, input,
      CANDIDATE_SCHEDULE.capture, geometry);
    // Ordered data-parallel candidate construction. Each phase is its own
    // dispatch so per-level and inter-phase ordering is carried by dispatch
    // boundaries instead of statement order inside a single invocation.
    this.runCandidateIndirect(staged("clear levels"), "clearCandidateLevels", 0, input,
      CANDIDATE_SCHEDULE.clear, geometry);
    this.runCandidateIndirect(staged("build level sets"), "buildCandidateLevelSets", 0, input,
      CANDIDATE_SCHEDULE.topologyLevels, geometry);
    this.runCandidateIndirect(staged("detect ghosts"), "detectCandidateGhosts", 0, input,
      CANDIDATE_SCHEDULE.topologyRows, geometry);
    this.runCandidateIndirect(staged("insert ghosts"), "insertCandidateGhosts", 0, input,
      CANDIDATE_SCHEDULE.topologyLevels, geometry);
    this.run(staged("build level deltas"), "buildCandidateLevelDeltas", 0, input, [1, 1, 1], geometry);
    this.runCandidateIndirect(staged("count transfers"), "countCandidateTransfers", 0, input,
      CANDIDATE_SCHEDULE.transferSlots, geometry);
    this.runCandidateIndirect(staged("scan transfers"), "scanCandidateTransfers", 0, input,
      CANDIDATE_SCHEDULE.topologyLevels, geometry);
    this.runCandidateIndirect(staged("write transfers"), "writeCandidateTransfers", 0, input,
      CANDIDATE_SCHEDULE.transferSlots, geometry);
    this.runCandidateIndirect(staged("link parent chains"), "linkCandidateParentChains", 0, input,
      CANDIDATE_SCHEDULE.topologyLevels, geometry);
    if (this.touchedDirectoryEnabled) this.run(staged("append directory identities"),
      "appendCandidateDirectoryIdentities", 0, input, [1, 1, 1], geometry);
    if (!this.touchedDirectoryEnabled || this.touchedDirectoryTripwire) {
      this.runCandidateIndirect(staged("mark brick occupancy"), "markCandidateBrickOccupancy", 0, input,
        CANDIDATE_SCHEDULE.bricks, geometry);
      this.runCandidateIndirect(staged("rank bricks"), "rankCandidateBricks", 0, input,
        CANDIDATE_SCHEDULE.topologyLevels, geometry);
      this.runCandidateIndirect(staged("scatter ranked slots"), "scatterCandidateRankedSlots", 0, input,
        CANDIDATE_SCHEDULE.bricks, geometry);
      this.runCandidateIndirect(staged("mark page occupancy"), "markCandidatePageOccupancy", 0, input,
        CANDIDATE_SCHEDULE.logicalPages, geometry);
      this.runCandidateIndirect(staged("compact pages"), "compactCandidatePages", 0, input,
        CANDIDATE_SCHEDULE.topologyLevels, geometry);
      this.runCandidateIndirect(staged("link page neighbours"), "linkCandidatePageNeighbours", 0, input,
        CANDIDATE_SCHEDULE.physicalPages, geometry);
    }
    if (this.touchedDirectoryEnabled) {
      const brickSort = this.touchedBrickSort!, pageSort = this.touchedPageSort!;
      brickSort.encode(broker);
      this.runExternalIndirect(staged("compact brick masks"), "markCompactCandidateBrickOccupancy",
        input, brickSort.liveDispatch, geometry);
      this.run(staged("compact brick ranks"), "rankCompactCandidateBricks", 0, input,
        [1, 1, 1], geometry);
      pageSort.encode(broker);
      this.run(staged("compact pages"), "buildCompactCandidatePages", 0, input,
        [1, 1, 1], geometry);
      this.runExternalIndirect(staged("link compact page neighbours"),
        "linkCompactCandidatePageNeighbours", input, pageSort.liveDispatch, geometry);
    }
    this.runCandidateIndirect(staged("build stencils"), "buildCandidateStencils", 0, input,
      CANDIDATE_SCHEDULE.stencilSlots, geometry);
    this.runCandidateIndirect(staged("publish spectral bounds"), "publishCandidateSpectralBounds", 0, input,
      CANDIDATE_SCHEDULE.stencilLevels, geometry);
    this.run(staged("validate hierarchy"), "validateCandidateHierarchy", 0, input, [1, 1, 1]);
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
    this.runCandidateIndirect(pass, "commitChangedL1", 0, input,
      CANDIDATE_SCHEDULE.capture, this.capturedGeometry);
    this.run(pass, "finalizeL1CapturePublication", 0, input, [1, 1, 1]);
    if (this.touchedDirectoryEnabled) this.run(pass, "commitCandidateTouchedBricks", 0, input,
      [1, 1, 1]);
    this.runCandidateIndirect(pass, "commitCandidateLevels", 0, input,
      CANDIDATE_SCHEDULE.commit);
    this.run(pass, "finalizeLifecycle", 0, input, [1, 1, 1]);
    // Fingerprint the exact inputs this hierarchy consumed, and only after the
    // lifecycle gate accepted it. A rejected epoch leaves the previous
    // fingerprint (or none), so the next probe can never skip a stale build.
    this.runLiveRows(pass, "publishCommittedInputs", 0, candidate, this.candidateCapturedGeometry);
    if (this.hierarchicalExecutorCompiled) {
      const imageDispatch = this.source.classDispatch;
      if (!imageDispatch) {
        throw new Error("SPGrid image compilation requires structured live-row task records");
      }
      const imageDispatchBase = this.source.classDispatchOffsetBytes ?? 0;
      // Compile the Section 6.3 operator's addressing for the epoch this pass
      // just published. The persistent executor transcribes these applies and
      // therefore neither compiles nor publishes the fallback operator image.
      pass.setPipeline(this.accurateOperatorRowsPipeline);
      pass.setBindGroup(0, this.accurateOperatorRowsGroup);
      pass.dispatchWorkgroupsIndirect(imageDispatch,
        imageDispatchBase + ACCURATE_SOURCE_IMAGE_ROW_RECORD_BYTES);
      pass.setPipeline(this.accurateAdjointRowsPipeline);
      pass.setBindGroup(0, this.accurateAdjointRowsGroup);
      pass.dispatchWorkgroupsIndirect(imageDispatch,
        imageDispatchBase + ACCURATE_SOURCE_IMAGE_TRANSITION_RECORD_BYTES);
    }
    // dispatchMeta remains STORAGE-only inside compute passes. Copying its
    // finalized records after the setup boundary gives correction a distinct
    // INDIRECT-only source and avoids a whole-pass storage/indirect conflict.
    broker.updateIndirectBuffer(this.dispatchMeta, 0, this.indirectDispatch, 0, this.plan.dispatchBytes);
    this.lastSetupInput = input;
    this.candidateSetupInput = undefined;
  }

  encodeSetup(broker: PassBroker, input: { solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    if (!this.candidateSetupInput && !this.preparedCaptureSource
      && this.lastSetupInput?.rowCount === input.rowCount
      && this.lastSetupInput.solverControl === input.solverControl) return;
    if (!this.candidateSetupInput) {
      this.encodeSetupCandidate(broker, { ...input, sourceMode: "accepted" });
    }
    this.encodeReadySetupCommit(broker, input);
  }

  encodeCorrection(broker: PassBroker, input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive();
    this.assertHierarchicalExecutorCompiled();
    const pass = broker.compute({ label: "SPGrid V-cycle · publish convergence-gated level records" });
    this.encodeCorrectionGate(pass, input);
    broker.fence("SPGrid V-cycle convergence-gated indirect publication");
    this.encodeCorrectionBody(broker, input);
  }

  encodeCorrectionGate(pass: GPUComputePassEncoder,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive();
    this.assertHierarchicalExecutorCompiled();
    this.run(pass, "prepareCorrectionDispatches", 0, input, [1, 1, 1]);
  }

  encodeCorrectionBody(broker: PassBroker,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive();
    this.assertHierarchicalExecutorCompiled();
    const pass = broker.compute({ label: "SPGrid V-cycle · one-pass symmetric correction" });
    // Every stage below re-asks the broker for a pass under its own label. With
    // label isolation OFF -- production, and every wall-clock lane -- `compute()`
    // returns the pass already open and DISCARDS the label, so the encoded
    // command stream is byte-for-byte the single pass this reads as. With
    // `FLUID_GPU_ISOLATE_PASS_LABELS=1` each label brackets its own dispatch,
    // which is the only way to attribute individual V-cycle phases.
    const staged = (label: string): GPUComputePassEncoder =>
      broker.compute({ label: `SPGrid V-cycle · ${label}` });
    // Correction clear, RHS seed, and final publication are defined on the
    // accepted pressure-row domain, not on level zero's sparse slot worklist.
    // Aanjaneya et al. §4.3 relies
    // on L1 and L2 having exactly the same pressure variables; dispatching
    // through count(0) silently omitted native coarser rows on an adaptive
    // octree and allowed their correction entries to retain an earlier M1
    // application. The fixed-capacity row dispatch remains convergence-gated
    // in WGSL and covers every live accepted row.
    this.runRowIndirect(pass, "clearCorrection", 0, input);
    for (let level = 0; level < this.plan.levelCount; level += 1) this.runIndirect(pass, "zeroVectors", level, input, false);
    this.runRowIndirect(pass, "seedRhs", 0, input);
    const parallelLevels = this.vcycleParallelLevels;
    for (let level = 0; level < parallelLevels; level += 1) {
      this.smooth(staged(`pre-smooth level ${level}`), level, false, input);
      this.runIndirect(staged(`restrict level ${level}`),
        "restrictAndGhostAccumulate", level, input, true);
    }
    this.run(staged(`coarse V-cycle tail levels ${parallelLevels}-bottom`),
      "coarseVcycleTail", parallelLevels, input, [1, 1, 1]);
    for (let level = parallelLevels - 1; level >= 0; level -= 1) {
      this.runIndirect(staged(`prolong level ${level}`),
        "prolongAndGhostPropagate", level, input, false);
      this.smooth(staged(`post-smooth level ${level}`), level, true, input);
    }
    this.runRowIndirect(staged("publish correction"), "publish", 0, input);
  }

  private smooth(pass: GPUComputePassEncoder, level: number, reverse: boolean,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    for (let step = 0; step < this.pre; step += 1) {
      const phase = reverse ? this.pre - 1 - step : step;
      const direction = (step & 1) === 0 ? "AtoB" : "BtoA";
      const name = `smoothChebyshev${direction}${phase}` as OctreeSPGridVCyclePipelineName;
      this.runIndirect(pass, name, level, input, false);
    }
  }

  private runIndirect(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }, transfer: boolean): void {
    this.bind(pass, name, level, input);
    pass.dispatchWorkgroupsIndirect(this.indirectDispatch,
      level * DISPATCH_RECORD_BYTES_PER_LEVEL + (transfer ? 20 : 8));
  }
  private runRowIndirect(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.bind(pass, name, level, input);
    pass.dispatchWorkgroupsIndirect(this.indirectDispatch,
      this.plan.levelCount * DISPATCH_RECORD_BYTES_PER_LEVEL + 8);
  }
  private runLiveRows(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer;
      sourceControl?: GPUBuffer; topologyMetrics?: GPUBuffer; sourceMode?: OctreeSPGridSourceMode },
    geometry = this.capturedGeometry): void {
    if (this.capacityRowScheduleOracle) {
      this.run(pass, name, level, input, this.plan.rowDispatch, geometry);
      return;
    }
    const dispatch = this.source.liveRowDispatch;
    if (!dispatch) {
      // Unit-test and third-party synthetic sources predating the live-count
      // ABI retain the bounded path. The production structured authority
      // always publishes this record.
      this.run(pass, name, level, input, this.plan.rowDispatch, geometry);
      return;
    }
    this.bind(pass, name, level, input, geometry);
    pass.dispatchWorkgroupsIndirect(dispatch, this.source.liveRowDispatchOffsetBytes ?? 0);
  }
  private run(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer;
      sourceControl?: GPUBuffer; topologyMetrics?: GPUBuffer; sourceMode?: OctreeSPGridSourceMode },
    dispatch: readonly [number, number, number], geometry = this.capturedGeometry): void {
    this.bind(pass, name, level, input, geometry); pass.dispatchWorkgroups(...dispatch);
  }
  private runCandidateIndirect(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName,
    level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer;
      sourceControl?: GPUBuffer; topologyMetrics?: GPUBuffer; sourceMode?: OctreeSPGridSourceMode },
    offset: number, geometry = this.capturedGeometry): void {
    this.bind(pass, name, level, input, geometry);
    pass.dispatchWorkgroupsIndirect(this.candidateIndirect, offset);
  }
  private runExternalIndirect(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName,
    input: { solverControl: GPUBuffer; rowCount: GPUBuffer; sourceControl?: GPUBuffer;
      topologyMetrics?: GPUBuffer; sourceMode?: OctreeSPGridSourceMode }, dispatch: GPUBuffer,
    geometry = this.capturedGeometry): void {
    this.bind(pass, name, 0, input, geometry); pass.dispatchWorkgroupsIndirect(dispatch, 0);
  }
  private bind(pass: GPUComputePassEncoder, name: OctreeSPGridVCyclePipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer;
      sourceControl?: GPUBuffer; topologyMetrics?: GPUBuffer; sourceMode?: OctreeSPGridSourceMode },
    geometry = this.capturedGeometry): void {
    const pipeline = this.pipelines[name];
    if (!pipeline) {
      throw new Error(`SPGrid pipeline ${name} is not initialized`);
    }
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
        [22, this.candidateGhosts], [23, this.committedInputs], [24, this.source.coefficients],
        [25, this.touchedBrickSort?.keys ?? this.touchedDirectoryDummy],
        [26, this.touchedBrickHeader],
        [27, this.touchedBrickSort?.runs ?? this.touchedDirectoryDummy],
        [28, this.touchedBrickSort?.control ?? this.touchedDirectoryDummy],
        [29, this.touchedPageSort?.keys ?? this.touchedDirectoryDummy],
        [30, this.touchedPageHeader],
        [31, this.touchedPageSort?.runs ?? this.touchedDirectoryDummy],
        [32, this.touchedPageSort?.control ?? this.touchedDirectoryDummy],
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
    const pass = broker.compute({ label: "SPGrid accurate A2 - publish convergence-gated records" });
    this.encodeAccurateGate(pass, input, output, solverControl, worksets, worksetLayout);
    broker.fence("SPGrid accurate A2 convergence-gated indirect publication");
    this.encodeAccurateBody(broker, input, output, solverControl, worksets, worksetLayout);
  }

  private encodeAccurateGate(
    pass: GPUComputePassEncoder,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBufferBinding,
    worksetLayout: GPUBuffer,
  ): void {
    this.assertLive();
    if (input.size < this.plan.rowCapacity * 4 || output.size < this.plan.rowCapacity * 4) {
      throw new RangeError("SPGrid accurate A2 vectors are smaller than row capacity");
    }
    const cached = this.accurateBinding(input, output, solverControl, worksets, worksetLayout);
    pass.setPipeline(this.accurateGatePipeline);
    pass.setBindGroup(0, cached.gateGroup);
    pass.dispatchWorkgroups(1, 1, 1);
  }

  private encodeAccurateBody(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBufferBinding,
    worksetLayout: GPUBuffer,
  ): void {
    this.assertLive();
    if (input.size < this.plan.rowCapacity * 4 || output.size < this.plan.rowCapacity * 4) {
      throw new RangeError("SPGrid accurate A2 vectors are smaller than row capacity");
    }
    const cached = this.accurateBinding(input, output, solverControl, worksets, worksetLayout);
    let pass = broker.compute({ label: "SPGrid accurate A2 - parallel direct terms" });
    pass.setPipeline(this.accurateTermPipeline); pass.setBindGroup(0, cached.termGroup);
    pass.dispatchWorkgroupsIndirect(this.accurateClassDispatch, ACCURATE_TERM_DISPATCH_OFFSET_BYTES);
    pass = broker.compute({ label: "SPGrid accurate A2 - parallel fine-adjoint children" });
    pass.setPipeline(this.accurateAdjointPipeline); pass.setBindGroup(0, cached.adjointGroup);
    pass.dispatchWorkgroupsIndirect(this.accurateClassDispatch, ACCURATE_ADJOINT_DISPATCH_OFFSET_BYTES);
    pass = broker.compute({ label: "SPGrid accurate A2 - ordered row fold" });
    pass.setPipeline(this.accurateFinalizePipeline); pass.setBindGroup(0, cached.finalizeGroup);
    pass.dispatchWorkgroupsIndirect(this.accurateClassDispatch, ACCURATE_UNION_DISPATCH_OFFSET_BYTES);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after accurate A2 class apply");
    }
  }

  private encodeAccurateResidualBody(
    broker: PassBroker,
    input: GPUBuffer,
    residualRhs: GPUBuffer,
    residual: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBufferBinding,
    worksetLayout: GPUBuffer,
  ): void {
    this.assertLive();
    if (input.size < this.plan.rowCapacity * 4
      || residualRhs.size < this.plan.rowCapacity * 4
      || residual.size < this.plan.rowCapacity * 4) {
      throw new RangeError("SPGrid accurate A2 residual vectors are smaller than row capacity");
    }
    const cached = this.accurateBinding(
      input, residual, solverControl, worksets, worksetLayout, residualRhs,
    );
    let pass = broker.compute({ label: "SPGrid accurate A2 residual - parallel direct terms" });
    pass.setPipeline(this.accurateTermPipeline); pass.setBindGroup(0, cached.termGroup);
    pass.dispatchWorkgroupsIndirect(this.accurateClassDispatch, ACCURATE_TERM_DISPATCH_OFFSET_BYTES);
    pass = broker.compute({ label: "SPGrid accurate A2 residual - parallel fine-adjoint children" });
    pass.setPipeline(this.accurateAdjointPipeline); pass.setBindGroup(0, cached.adjointGroup);
    pass.dispatchWorkgroupsIndirect(this.accurateClassDispatch, ACCURATE_ADJOINT_DISPATCH_OFFSET_BYTES);
    pass = broker.compute({ label: "SPGrid accurate A2 residual - ordered row fold and subtraction" });
    pass.setPipeline(this.accurateResidualFinalizePipeline);
    pass.setBindGroup(0, cached.residualFinalizeGroup!);
    pass.dispatchWorkgroupsIndirect(this.accurateClassDispatch, ACCURATE_UNION_DISPATCH_OFFSET_BYTES);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after accurate A2 residual apply");
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
      || mergedDispatchOffsetBytes + 36 > mergedDispatch.size) {
      throw new RangeError("SPGrid accurate A2 merged-band dispatch offset is invalid");
    }
    const cached = this.accurateBinding(input, output, solverControl, worksets, worksetLayout);
    if (!cached.mergedTermGroup) {
      const shared = new Map<number, GPUBufferBinding>([
        [0, { buffer: input }], [2, { buffer: solverControl }],
        [3, { buffer: this.source.control }],
        [4, { buffer: cached.worksets, offset: cached.worksetOffset,
          ...(cached.worksetSize === undefined ? {} : { size: cached.worksetSize }) }],
        [5, { buffer: worksetLayout }], [6, { buffer: this.topology }],
        [7, { buffer: this.state }], [8, { buffer: this.capturedGeometry }],
        [9, { buffer: this.source.topologyMetrics }], [10, { buffer: this.source.coefficients }],
        [11, { buffer: this.accurateWorksetLayout }], [12, { buffer: this.accurateTerms }],
        [13, { buffer: this.accurateOperatorRows }],
      ]);
      // Image and chase are deliberately mirror ABIs. Production drops the
      // topology/state arenas in favour of the compiled index image; the
      // measurement arm retains the arenas and drops that image.
      if (this.directByChase) shared.delete(13);
      else { shared.delete(6); shared.delete(7); }
      cached.mergedTermGroup = this.device.createBindGroup({
        label: "SPGrid Section 6.3 · parallel merged-band term bindings",
        layout: this.accurateMergedTermPipeline.getBindGroupLayout(0),
        entries: [...shared].map(([binding, resource]) => ({ binding, resource })),
      });
    }
    let pass = broker.compute({ label: "SPGrid Section 6.3 - parallel merged-band direct terms" });
    pass.setPipeline(this.accurateMergedTermPipeline); pass.setBindGroup(0, cached.mergedTermGroup);
    pass.dispatchWorkgroupsIndirect(mergedDispatch, mergedDispatchOffsetBytes + 12);
    if (!cached.mergedAdjointGroup) {
      const shared = new Map<number, GPUBufferBinding>([
        [0, { buffer: input }], [2, { buffer: solverControl }],
        [3, { buffer: this.source.control }],
        [4, { buffer: cached.worksets, offset: cached.worksetOffset,
          ...(cached.worksetSize === undefined ? {} : { size: cached.worksetSize }) }],
        [5, { buffer: worksetLayout }], [6, { buffer: this.topology }],
        [7, { buffer: this.state }], [8, { buffer: this.capturedGeometry }],
        [9, { buffer: this.source.topologyMetrics }], [10, { buffer: this.source.coefficients }],
        [11, { buffer: this.accurateWorksetLayout }], [12, { buffer: this.accurateTerms }],
        [14, { buffer: this.accurateAdjointRows }],
      ]);
      // Same reachability change the direct-term stage took: the addressing is
      // the compiled fine-adjoint image (14), so the stage no longer reaches
      // the topology (6) or state (7) arenas, and dropping them keeps it inside
      // the ten-storage-buffer ceiling that binding 14 would otherwise break.
      // The control arm is the mirror image of that: it walks the arenas and
      // never reads the compiled image.
      if (this.adjointByChase) shared.delete(14);
      else { shared.delete(6); shared.delete(7); }
      cached.mergedAdjointGroup = this.device.createBindGroup({
        label: "SPGrid Section 6.3 · parallel merged-band adjoint bindings",
        layout: this.accurateMergedAdjointPipeline.getBindGroupLayout(0),
        entries: [...shared].map(([binding, resource]) => ({ binding, resource })),
      });
    }
    pass = broker.compute({ label: "SPGrid Section 6.3 - parallel merged-band adjoint children" });
    pass.setPipeline(this.accurateMergedAdjointPipeline); pass.setBindGroup(0, cached.mergedAdjointGroup);
    pass.dispatchWorkgroupsIndirect(mergedDispatch, mergedDispatchOffsetBytes + 24);
    pass = broker.compute({ label: "SPGrid Section 6.3 - ordered merged-band row fold" });
    pass.setPipeline(this.accurateFinalizePipeline); pass.setBindGroup(0, cached.finalizeGroup);
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
    residualSource?: GPUBuffer,
  ): CachedAccurateApply {
    const worksetOffset = Number(worksets.offset ?? 0);
    const worksetSize = worksets.size === undefined ? undefined : Number(worksets.size);
    let cached = this.accurateBindings.find((candidate) =>
      candidate.input === input && candidate.output === output
      && candidate.solverControl === solverControl
      && candidate.worksets === worksets.buffer
      && candidate.worksetOffset === worksetOffset
      && candidate.worksetSize === worksetSize
      && candidate.worksetLayout === worksetLayout
      && candidate.residualSource === residualSource);
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
      shared.set(12, { buffer: this.accurateTerms });
      shared.set(13, { buffer: this.accurateOperatorRows });
      const termGroup = makeGroup(this.accurateTermPipeline,
        this.directByChase
          ? [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
          : [0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13],
        "SPGrid Section 6.3 · staged direct-term bindings");
      shared.set(14, { buffer: this.accurateAdjointRows });
      // Likewise for the fine-adjoint stage: compiled image (14) instead of
      // topology/state.
      const adjointGroup = makeGroup(this.accurateAdjointPipeline,
        [0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14],
        "SPGrid Section 6.3 · staged fine-adjoint bindings");
      const finalizeGroup = makeGroup(this.accurateFinalizePipeline,
        [0, 1, 2, 3, 8, 9, 10, 11, 12],
        "SPGrid Section 6.3 · ordered row-fold bindings");
      let residualFinalizeGroup: GPUBindGroup | undefined;
      if (residualSource) {
        shared.set(18, { buffer: residualSource });
        residualFinalizeGroup = makeGroup(this.accurateResidualFinalizePipeline,
          [0, 1, 2, 3, 8, 9, 10, 11, 12, 18],
          "SPGrid Section 6.3 · ordered residual row-fold bindings");
      }
      cached = { input, output, solverControl, worksets: worksets.buffer,
        worksetOffset, worksetSize, worksetLayout, gateGroup,
        termGroup, adjointGroup, finalizeGroup,
        residualSource, residualFinalizeGroup };
      this.accurateBindings.push(cached);
    }
    return cached;
  }

  private assertHierarchicalExecutorCompiled(): void {
    if (!this.hierarchicalExecutorCompiled) {
      throw new Error("SPGrid hierarchical correction executor was not compiled");
    }
  }

  private assertLive(): void { if (this.destroyed) throw new Error("SPGrid V-cycle is destroyed"); }
  destroy(): void {
    if (this.destroyed) return; this.destroyed = true; this.groups.clear(); this.accurateBindings.length = 0;
    this.capturedGeometry.destroy(); this.candidateCapturedGeometry.destroy();
    this.topology.destroy(); this.state.destroy(); this.dispatchMeta.destroy();
    this.indirectDispatch.destroy(); this.candidateIndirect.destroy();
    this.capturePageState.destroy(); this.levelDelta.destroy();
    this.candidateTopology.destroy(); this.candidateState.destroy(); this.candidateDispatch.destroy();
    this.touchedBrickSort?.destroy(); this.touchedPageSort?.destroy();
    this.touchedBrickHeader.destroy(); this.touchedPageHeader.destroy(); this.touchedDirectoryDummy.destroy();
    this.candidateGhosts.destroy(); this.committedInputs.destroy();
    this.accurateWorksetLayout.destroy(); this.accurateClassDispatch?.destroy();
    this.accurateTerms?.destroy();
    this.accurateOperatorRows?.destroy(); this.accurateAdjointRows?.destroy();
    this.accurateImageDeltaParams?.destroy(); this.accurateImageEpochs?.destroy();
    for (const buffer of [...this.params, ...this.candidateParams]) buffer.destroy();
  }
}

/** Accurate second-order matrix-free apply over five accepted row classes. */
export const octreeSPGridAccurateDispatchGateShader = /* wgsl */ `
@group(0) @binding(0) var<storage,read> solverControl:array<u32>;
@group(0) @binding(1) var<storage,read> worksets:array<u32>;
@group(0) @binding(2) var<uniform> worksetLayout:vec4u;
@group(0) @binding(3) var<storage,read> accepted:array<u32>;
@group(0) @binding(4) var<storage,read_write> classDispatch:array<u32>;
fn activeSolve()->bool{return arrayLength(&solverControl)>=2u
 &&solverControl[0]==0u&&solverControl[1]==0u;}
fn publishAccurateDispatch(at:u32,blocks:u32,live:bool){
 if(live&&blocks>0u){let x=min(65535u,blocks);classDispatch[at]=x;
  classDispatch[at+1u]=(blocks+x-1u)/x;classDispatch[at+2u]=1u;}
 else{classDispatch[at]=0u;classDispatch[at+1u]=1u;classDispatch[at+2u]=1u;}
}
@compute @workgroup_size(1)
fn prepareAccurateDispatches(){
 let solveLive=activeSolve();
 let bank=select(0u,accepted[4]&1u,arrayLength(&accepted)>4u);
 var unionRows=0u;var transitionRows=0u;var unionValid=true;
 for(var cls=0u;cls<5u;cls+=1u){
  let base=bank*worksetLayout.y+cls*worksetLayout.x;
  let source=base+4u;
  let valid=source+2u<arrayLength(&worksets);
  if(cls<4u){let destination=cls*3u;
   classDispatch[destination]=select(0u,worksets[source],solveLive&&valid);
   classDispatch[destination+1u]=select(1u,worksets[source+1u],valid);
   classDispatch[destination+2u]=select(1u,worksets[source+2u],valid);}
  // The union record covers the same rows as the five class records. Its lane
  // count is the exact sum of the published class counts, so the concatenated
  // staged union walk lands on every accepted row and on no other.
  if(!valid||base+2u>=arrayLength(&worksets)||worksets[base+1u]>worksets[base+2u]){unionValid=false;}
  else{unionRows+=worksets[base+1u];if(cls==1u||cls==3u){transitionRows+=worksets[base+1u];}}
 }
 publishAccurateDispatch(12u,(unionRows+63u)/64u,solveLive&&unionValid);
 publishAccurateDispatch(15u,(unionRows*18u+63u)/64u,solveLive&&unionValid);
 // One lane per (transition row, child, candidate): the eighteen candidates a
 // child's chains cover are independent, so they issue concurrently instead of
 // eighteen deep behind one lane.
 publishAccurateDispatch(18u,(transitionRows*144u+63u)/64u,solveLive&&unionValid);
 classDispatch[21]=max(classDispatch[21],select(0u,1u,solveLive));
 classDispatch[22]=max(classDispatch[22],select(0u,1u,unionValid));
 classDispatch[23]=max(classDispatch[23],unionRows);
 classDispatch[24]=max(classDispatch[24],transitionRows);
 classDispatch[25]=max(classDispatch[25],bank);
 classDispatch[26]=max(classDispatch[26],select(0u,accepted[3],arrayLength(&accepted)>3u));
 classDispatch[27]+=select(0u,1u,solveLive);
 classDispatch[28]+=select(0u,1u,solveLive&&unionValid&&unionRows>0u);
}
`;

/**
 * The eighteen canonical Section 6.3 channel directions. This is the exact
 * table both operator shaders carry as `canonicalDirection`; it exists on the
 * host only so the inverse channel lookup below can be *derived from the
 * shader's own scan* instead of transcribed.
 */
const SECTION63_CANONICAL_DIRECTIONS: readonly (readonly [number, number, number])[]
  = Object.freeze(([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [1, 0, 1], [1, 0, -1],
    [-1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
  ] as [number, number, number][]).map((direction) => Object.freeze(direction)));

/**
 * Literal transliteration of the WGSL `worldDirection`: the transform code's
 * three sign bits scale the *source* components, then `(code / 8) % 6` selects
 * the component permutation. Kept statement-for-statement identical to the
 * shader so the generated table cannot drift from the scan it replaces.
 */
function section63WorldDirectionOnHost(
  value: readonly [number, number, number], code: number,
): [number, number, number] {
  const q: [number, number, number] = [
    ((code & 1) !== 0 ? -1 : 1) * value[0],
    ((code & 2) !== 0 ? -1 : 1) * value[1],
    ((code & 4) !== 0 ? -1 : 1) * value[2],
  ];
  const permutation = Math.floor(code / 8) % 6;
  if (permutation === 0) return [q[0], q[1], q[2]];
  if (permutation === 1) return [q[0], q[2], q[1]];
  if (permutation === 2) return [q[1], q[0], q[2]];
  if (permutation === 3) return [q[2], q[0], q[1]];
  if (permutation === 4) return [q[1], q[2], q[0]];
  return [q[2], q[1], q[0]];
}

/** 64 transform codes x 27 direction slots, one byte each. */
const SECTION63_CHANNEL_TABLE_SLOTS = 27;
const SECTION63_CHANNEL_TABLE_ENTRIES = 64 * SECTION63_CHANNEL_TABLE_SLOTS;
const SECTION63_CHANNEL_TABLE_WORDS = SECTION63_CHANNEL_TABLE_ENTRIES / 4;
/** Sentinel for "no stored channel points this way" — the scan's `return 0.0`. */
const SECTION63_CHANNEL_NONE = 255;

/**
 * Inverts `worldDirection` once, on the host, by *running the shader's own
 * linear scan* over all 64 transform codes and all 27 directions in
 * {-1,0,1}^3. The result is the identical answer the scan would return, so the
 * replacement is a pure lookup with no arithmetic change.
 *
 * Four self-checks fail closed at import rather than on the GPU:
 *  1. an independently authored authority — the cube-symmetry group in
 *     `octree-power-topology`, whose `transformPowerVector`/`inverseCubeTransform`
 *     were written for the catalog generator and share no code with the shader —
 *     must reproduce `worldDirection` on every (code, canonical direction) pair;
 *  2. every canonical direction must resolve to a channel;
 *  3. each code's canonical map must be a bijection over the 18 channels
 *     (no channel claimed twice, so the scan's first-match rule is unambiguous);
 *  4. the nine non-canonical slots must stay unmatched, mirroring the scan's
 *     fall-through to 0.0.
 */
function buildSection63DirectionChannelTable(): Uint8Array {
  const table = new Uint8Array(SECTION63_CHANNEL_TABLE_ENTRIES).fill(SECTION63_CHANNEL_NONE);
  for (let code = 0; code < 64; code += 1) {
    for (let slot = 0; slot < SECTION63_CHANNEL_TABLE_SLOTS; slot += 1) {
      const wanted: [number, number, number] = [
        (slot % 3) - 1, (Math.floor(slot / 3) % 3) - 1, Math.floor(slot / 9) - 1,
      ];
      // This loop *is* `coefficientForDirection`'s scan, first match wins.
      for (let channel = 0; channel < 18; channel += 1) {
        const world = section63WorldDirectionOnHost(SECTION63_CANONICAL_DIRECTIONS[channel]!, code);
        if (world[0] === wanted[0] && world[1] === wanted[1] && world[2] === wanted[2]) {
          table[code * SECTION63_CHANNEL_TABLE_SLOTS + slot] = channel;
          break;
        }
      }
    }
  }
  for (let code = 0; code < 64; code += 1) {
    const transform = OCTREE_CUBE_TRANSFORMS[(Math.floor(code / 8) % 6) * 8 + (code & 7)]!;
    const inverse = inverseCubeTransform(transform);
    const claimed = new Set<number>();
    for (let channel = 0; channel < 18; channel += 1) {
      const direction = SECTION63_CANONICAL_DIRECTIONS[channel]!;
      const mine = section63WorldDirectionOnHost(direction, code);
      const theirs = transformPowerVector([direction[0], direction[1], direction[2]], inverse);
      if (mine[0] !== theirs[0] || mine[1] !== theirs[1] || mine[2] !== theirs[2]) {
        throw new Error("Section 6.3 world direction disagrees with the cube-transform group");
      }
      const slot = (direction[0] + 1) + 3 * (direction[1] + 1) + 9 * (direction[2] + 1);
      const resolved = table[code * SECTION63_CHANNEL_TABLE_SLOTS + slot]!;
      if (resolved >= 18) throw new Error("Section 6.3 channel table left a canonical direction unmatched");
      claimed.add(resolved);
    }
    if (claimed.size !== 18) throw new Error("Section 6.3 channel table is not a per-code bijection");
    for (const slot of [0, 2, 6, 8, 13, 18, 20, 24, 26]) {
      if (table[code * SECTION63_CHANNEL_TABLE_SLOTS + slot] !== SECTION63_CHANNEL_NONE) {
        throw new Error("Section 6.3 channel table matched a non-canonical direction");
      }
    }
  }
  return table;
}

/**
 * The packed table plus its accessor, shared verbatim by the accurate A2
 * operator and the Section 4.3 band dilation. Four one-byte entries per word.
 */
export const octreeSection63DirectionChannelWGSL: string = (() => {
  const table = buildSection63DirectionChannelTable();
  const words = new Uint32Array(SECTION63_CHANNEL_TABLE_WORDS);
  for (let entry = 0; entry < SECTION63_CHANNEL_TABLE_ENTRIES; entry += 1) {
    words[entry >> 2] |= table[entry]! << ((entry & 3) * 8);
  }
  for (let entry = 0; entry < SECTION63_CHANNEL_TABLE_ENTRIES; entry += 1) {
    if (((words[entry >> 2]! >>> ((entry & 3) * 8)) & 0xff) !== table[entry]!) {
      throw new Error("Section 6.3 channel table packing is not lossless");
    }
  }
  const literals: string[] = [];
  for (let word = 0; word < words.length; word += 6) {
    literals.push([...words.slice(word, word + 6)]
      .map((value) => `0x${value.toString(16).padStart(8, "0")}u`).join(","));
  }
  return `
// Generated by buildSection63DirectionChannelTable: for every transform code
// (transformAndFlags & 63) and every direction in {-1,0,1}^3, the stored
// Section 6.3 channel whose worldDirection equals it, or 255 when none does.
// This is the memoized result of the eighteen-step scan it replaces.
const S63_CHANNEL_TABLE:array<u32,${SECTION63_CHANNEL_TABLE_WORDS}>=array<u32,${SECTION63_CHANNEL_TABLE_WORDS}>(
 ${literals.join(",\n ")});
fn section63ChannelForDirection(code:u32,direction:vec3i)->u32{
 if(any(direction<vec3i(-1))||any(direction>vec3i(1))){return ${SECTION63_CHANNEL_NONE}u;}
 let index=code*${SECTION63_CHANNEL_TABLE_SLOTS}u
  +u32(direction.x+1)+3u*u32(direction.y+1)+9u*u32(direction.z+1);
 return(S63_CHANNEL_TABLE[index>>2u]>>((index&3u)*8u))&0xffu;}
`;
})();

export const octreeSPGridAccurateOperatorShader = /* wgsl */ `
override persistentImageCarry:bool=true;
// The five sixteen-entry tables are the same memoized allocation authority the
// V-cycle uniform already carries. They replace this shader's four remaining
// per-address prefix loops; pageSlot alone used to run three of them - two of
// them full depth - on every one of applyRow's eighteen channels.
struct Layout{workset:vec4u,dimsCapacity:vec4u,hierarchy:vec4u,numerics:vec4f,
 levelCaps:array<vec4u,4>,levelBases:array<vec4u,4>,brickOffsets:array<vec4u,4>,
 pageOffsets:array<vec4u,4>,transferOffsets:array<vec4u,4>}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
struct ImageDeltaParams{controlOffset:u32,newToOldOffset:u32,oldToNewOffset:u32,dirtyCountWord:u32}
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
@group(0) @binding(12) var<storage,read_write> accurateTerms:array<f32>;
// Compiled operator image: nineteen u32 per row, published once per accepted
// topology epoch by buildAccurateOperatorRows and consumed by stageDirectTerm.
// Word 0 is the row's page-resolution status; word 1+channel is the resolved
// destination row, or an encoded skip/report code. u32 only: storing a float
// here and reloading it would end the term expression and cost the fused
// multiply-add, which is not a restructuring-only change on this backend
// (POWER_LIQUIDS_ULTIMATE_M1MAX, refuted lever 10).
@group(0) @binding(13) var<storage,read_write> operatorRows:array<u32>;
// Compiled fine-adjoint image: 144 u32 per row, one per (child, candidate)
// lane of the destination-owned GhostValueAccumulate, published by
// buildAccurateAdjointRows in the same commit and consumed by
// stageAdjointCandidate. A word is either an encoded edge -- the destination
// row in the low twenty bits and its Section 6.3 channel in the next five --
// or a code at or above CHANNEL_CODE_BASE carrying the reports the inline walk
// raised. Same discipline as the direct image: u32 only, so the coefficient is
// still a live load from the accepted bank and the term stays one fused
// multiply-add.
@group(0) @binding(14) var<storage,read_write> adjointRows:array<u32>;
@group(0) @binding(15) var<uniform> imageDelta:ImageDeltaParams;
@group(0) @binding(16) var<storage,read> rowDelta:array<u32>;
@group(0) @binding(17) var<storage,read_write> imageEpochs:array<atomic<u32>>;
// Optional source reached only by finalizeStagedUnionResidualRows. Ordinary
// A2 entry points keep their existing auto-layout and never bind this buffer.
@group(0) @binding(18) var<storage,read> residualRhs:array<f32>;
const INVALID=0xffffffffu;const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;
const KEY=0u;const FLAGS=1u;const OWNER=24u;const STATE_CHANNELS=26u;
const WORKSET_HEADER_WORDS=7u;const PAGE_RECORD_WORDS=28u;
const OPERATOR_ROW_WORDS=19u;
// Eight fine aliases of a 2:1 coarse leaf, eighteen candidate directions each.
// This is the staging layout accurateTerms already uses for the adjoint half of
// a row, so image word and staged word are the same index and the ordered fold
// is untouched.
const ADJOINT_ROW_WORDS=144u;
// The edge packing. Twenty bits of row is exactly SPGRID_MAXIMUM_ROW_CAPACITY,
// and five bits carry a channel that is always below eighteen here, so the
// largest edge is far below CHANNEL_CODE_BASE and can never be read as a code.
const ADJOINT_ROW_MASK=0xfffffu;
const ADJOINT_CHANNEL_SHIFT=20u;
// Row capacity is bounded by SPGRID_MAXIMUM_ROW_CAPACITY (1,048,576), so every
// value at or above this base is unambiguously a code and never a row index.
// Keep that headroom in mind if the ceiling is ever raised again: the margin is
// what makes this encoding unambiguous, and it is asserted by the differential
// harness rather than left to this comment.
const CHANNEL_CODE_BASE=0xffff0000u;
// Word-zero bit proving the row used the topology-regular image builder.  The
// row destination words remain ordinary row ids/codes; coefficients are never
// cached here, so GFM theta and cut fractions remain the shared face authority.
const HYBRID_REGULAR_ROW=0x80000000u;
// primary 0 / secondary 0: the direction resolved to a multigrid-only slot.
// The stencil contributes nothing and reports nothing, exactly as the inline
// walk's continue did.
const CHANNEL_SKIP=0xffff0000u;
const ROW_DELTA_VALID=0x52444c54u;const ROW_DELTA_STRUCTURAL=0x40000000u;
var<workgroup> compiledImagePredecessor:u32;
fn channelCode(primary:u32,secondary:u32)->u32{return CHANNEL_CODE_BASE|(secondary<<8u)|primary;}
fn operatorImageBank()->u32{return select(0u,accepted[3]&1u,
 arrayLength(&operatorRows)>=2u*capacity()*OPERATOR_ROW_WORDS);}
fn adjointImageBank()->u32{return select(0u,accepted[3]&1u,
 arrayLength(&adjointRows)>=2u*capacity()*ADJOINT_ROW_WORDS);}
fn operatorRowBase(row:u32)->u32{return(operatorImageBank()*capacity()+row)*OPERATOR_ROW_WORDS;}
fn adjointRowBase(row:u32)->u32{return(adjointImageBank()*capacity()+row)*ADJOINT_ROW_WORDS;}
fn priorOperatorRowBase(row:u32)->u32{return((1u-operatorImageBank())*capacity()+row)*OPERATOR_ROW_WORDS;}
fn priorAdjointRowBase(row:u32)->u32{return((1u-adjointImageBank())*capacity()+row)*ADJOINT_ROW_WORDS;}
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn stopped()->bool{return arrayLength(&solverControl)<2u||atomicLoad(&solverControl[0])!=0u||atomicLoad(&solverControl[1])!=0u;}
fn reportAt(flag:u32,stage:u32,row:u32){if(arrayLength(&solverControl)>0u){
 atomicOr(&solverControl[0],flag);
 if(arrayLength(&solverControl)>7u){let claim=atomicCompareExchangeWeak(&solverControl[6],0u,stage);
  if(claim.exchanged){atomicStore(&solverControl[7],row);}}
}}
fn acceptedBank()->u32{return accepted[4]&1u;}
fn persistentImagePredecessor(row:u32,bank:u32)->u32{
 if(!persistentImageCarry||arrayLength(&imageEpochs)<2u||bank>1u
  ||accepted[3]==0u
  ||atomicLoad(&imageEpochs[1u-bank])!=accepted[3]){return INVALID;}
 let base=imageDelta.controlOffset;
 if(base+15u>=arrayLength(&rowDelta)||row>=rowDelta[base]
  ||rowDelta[base]>capacity()||rowDelta[base+7u]!=accepted[3]
  ||rowDelta[base+8u]!=ROW_DELTA_VALID
  ||imageDelta.newToOldOffset>arrayLength(&rowDelta)
  ||row>=arrayLength(&rowDelta)-imageDelta.newToOldOffset){return INVALID;}
 let encoded=rowDelta[imageDelta.newToOldOffset+row];let value=encoded&0x3fffffffu;
 if((encoded&ROW_DELTA_STRUCTURAL)!=0u||value==0u||value-1u>=capacity()){return INVALID;}
 return value-1u;
}
fn remapPersistentDestination(oldRow:u32)->u32{
 if(imageDelta.oldToNewOffset>arrayLength(&rowDelta)
  ||oldRow>=arrayLength(&rowDelta)-imageDelta.oldToNewOffset){return INVALID;}
 let encoded=rowDelta[imageDelta.oldToNewOffset+oldRow];
 return select(INVALID,encoded-1u,encoded!=0u&&encoded-1u<capacity());
}
fn worksetBase(cls:u32)->u32{return acceptedBank()*worksetLayout.y+cls*worksetLayout.x;}
fn linearLane(wg:vec3u,groups:vec3u,lane:u32)->u32{return((wg.z*groups.y+wg.y)*groups.x+wg.x)*64u+lane;}
fn linearGroup(wg:vec3u,groups:vec3u)->u32{return(wg.z*groups.y+wg.y)*groups.x+wg.x;}
fn workRow(item:u32,cls:u32)->u32{let base=worksetBase(cls);if(base+WORKSET_HEADER_WORDS>arrayLength(&worksets)
 ||worksets[base]!=accepted[3]||worksets[base+1u]>worksets[base+2u]||item>=worksets[base+1u]
 ||base+WORKSET_HEADER_WORDS+item>=arrayLength(&worksets)){return INVALID;}return worksets[base+WORKSET_HEADER_WORDS+item];}
fn levels()->u32{return p.hierarchy.x;}fn maxStride()->u32{return p.hierarchy.y;}fn capacity()->u32{return p.dimsCapacity.w;}
// The level stride is 2^l by construction, so the ceiling division is a shift.
// Apple GPUs emulate integer division; this is the identical u32 result.
fn dims(l:u32)->vec3u{let s=1u<<l;return(p.dimsCapacity.xyz+vec3u(s-1u))>>vec3u(l);}
// Every level index this shader can form is in range. levels() is at most the
// twelve planOctreeSPGridVCycle admits, and applyRow's countTrailingZeros(h.y)
// reads a LeafHeader.size that only decodePagedOwner writes, from a three-bit
// exponent it rejects above 5 - so l is at most 5 whenever h.y is non-zero, and
// h.y==0 already makes dims(l) and 1u<<l shifts of 32, which WGSL leaves
// indeterminate, so the loops defined nothing there to preserve.
fn levelTable(l:u32)->vec2u{let clamped=min(l,15u);return vec2u(clamped>>2u,clamped&3u);}
fn levelCapacity(l:u32)->u32{let t=levelTable(l);return p.levelCaps[t.x][t.y];}
fn levelBase(l:u32)->u32{let t=levelTable(l);return p.levelBases[t.x][t.y];}
fn totalLevelSlots()->u32{return p.hierarchy.z;}
// levelBase(l) was an O(l) prefix loop whose body was itself a loop, which no
// compiler would lift out of a hot channel loop; it is now one uniform read.
// Callers that address many slots at one fixed level evaluate the prefix once
// and use atBase; at() is that same expression with the prefix inlined, so the
// two agree by construction and every existing call site is unchanged.
fn atBase(c:u32,base:u32,s:u32)->u32{return c*totalLevelSlots()+base+s;}
fn at(c:u32,l:u32,s:u32)->u32{return atBase(c,levelBase(l),s);}
fn rowMapBase()->u32{return 16u;}fn workBase()->u32{return rowMapBase()+levels()*capacity();}
fn pageWorkBase()->u32{return workBase()+totalLevelSlots();}
fn logicalPageDims(l:u32)->vec3u{return(dims(l)+vec3u(7u,7u,3u))/vec3u(8u,8u,4u);}
fn logicalPageCount(l:u32)->u32{let d=logicalPageDims(l);return d.x*d.y*d.z;}
fn pageLevelOffset(l:u32)->u32{let t=levelTable(l);return p.pageOffsets[t.x][t.y];}
fn pageDirectoryBase()->u32{return pageWorkBase()+PAGE_RECORD_WORDS*totalLevelSlots();}
fn transferCapacity(l:u32)->u32{return min(capacity()*8u,levelCapacity(l)*8u);}
fn transferLevelOffset(l:u32)->u32{let t=levelTable(l);return p.transferOffsets[t.x][t.y];}
fn transferBase()->u32{return pageDirectoryBase()+pageLevelOffset(levels());}
fn directoryBase()->u32{return transferBase()+transferLevelOffset(levels()-1u);}
fn brickDims(l:u32)->vec3u{return(dims(l)+vec3u(3u))/4u;}
fn brickCount(l:u32)->u32{let d=brickDims(l);return d.x*d.y*d.z;}
fn brickLevelOffset(l:u32)->u32{let t=levelTable(l);return p.brickOffsets[t.x][t.y];}
fn totalBrickCount()->u32{return brickLevelOffset(levels());}
fn rankedSlotsBase()->u32{return directoryBase()+16u+totalBrickCount()*4u;}
fn brickRecord(l:u32,q:vec3u)->u32{let d=brickDims(l);let b=q/4u;let dense=b.x+d.x*(b.y+d.y*b.z);
 return directoryBase()+16u+(brickLevelOffset(l)+dense)*4u;}
fn pageRecord(l:u32,page:u32)->u32{return pageWorkBase()+(levelBase(l)+page)*PAGE_RECORD_WORDS;}
fn pageNeighbour(l:u32,page:u32,ordinal:u32)->u32{return topology[pageRecord(l,page)+1u+ordinal];}
// Two divisions, not three: floor(floor(v/dx)/dy) == floor(v/(dx*dy)) exactly
// over the unsigned integers, and each remainder is recovered by one
// multiply-subtract from the quotient already in hand. Bit-identical to the
// modulo form, and it never forms the dx*dy product.
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;let row=v/d.x;let plane=row/d.y;
 return vec3u(v-row*d.x,row-plane*d.y,plane);}
fn localBit(q:vec3u)->u32{let local=q&vec3u(3u);return local.x+4u*local.y+16u*local.z;}
fn pageFor(l:u32,q:vec3u)->u32{let pages=logicalPageDims(l);let v=q/vec3u(8u,8u,4u);return topology[pageDirectoryBase()+pageLevelOffset(l)+v.x+pages.x*(v.y+pages.y*v.z)];}
// Do not memoize the (l, page, ordinal) page resolution across a stencil walk.
// It is exact to do so - pageNeighbour reads read-only topology - and the
// eighteen directions from one cell do land on ordinal 13 most of the time, but
// it MEASURED as nothing: applyRow's merged band moved 2.796 -> 2.776 ms over
// two interleaved captures while an untouched neighbour kernel moved by the same
// 0.02. The page resolution is two loads of one address that stays hot in L1;
// the cost here is the q-DEPENDENT tail below - brickRecord, the two brick-mask
// loads and the ranked-slot indirection - which is a dependent chain of
// scattered loads that no ordinal memo can remove.
// The resolution itself, with the report split out as a returned stage rather
// than an atomic. One definition serves the inline walk AND the once-per-epoch
// image builder, so the compiled addresses cannot drift from the addresses the
// inline walk would have produced - that identity is what makes the image a
// restructuring-only change rather than a re-derivation.
fn pageSlotCoded(l:u32,page:u32,origin:vec3u,q:vec3u)->vec2u{
 let shape=vec3u(8u,8u,4u);let delta=vec3i(q/shape)-vec3i(origin/shape);
 if(any(delta<vec3i(-1))||any(delta>vec3i(1))){return vec2u(INVALID,21u);}
 let ordinal=u32(delta.x+1)+3u*(u32(delta.y+1)+3u*u32(delta.z+1));let physical=pageNeighbour(l,page,ordinal);
 if(physical==INVALID){return vec2u(INVALID,0u);}let physicalOrigin=decode(topology[pageRecord(l,physical)],l);
 if(any(physicalOrigin/shape!=q/shape)){return vec2u(INVALID,22u);}
 let record=brickRecord(l,q);let bit=localBit(q);let low=topology[record+1u];let high=topology[record+2u];
 if(((select(low,high,bit>=32u)>>(bit&31u))&1u)==0u){return vec2u(INVALID,0u);}
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
 let slot=topology[rankedSlotsBase()+levelBase(l)+topology[record+3u]+rank];
 if(slot>=levelCapacity(l)){return vec2u(INVALID,23u);}return vec2u(slot,0u);}
fn pageSlot(l:u32,page:u32,origin:vec3u,q:vec3u,row:u32)->u32{
 let resolved=pageSlotCoded(l,page,origin,q);
 if(resolved.y!=0u){reportAt(2u,resolved.y,row);}
 return resolved.x;}
fn originOf(h:vec4u)->vec3u{return vec3u(h.x%p.dimsCapacity.x,(h.x/p.dimsCapacity.x)%p.dimsCapacity.y,h.x/(p.dimsCapacity.x*p.dimsCapacity.y));}
fn canonicalDirection(channel:u32)->vec3i{let d=array<vec3i,18>(
 vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),
 vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),
 vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[channel];}
fn worldDirection(value:vec3i,code:u32)->vec3i{let signs=vec3i(select(1,-1,(code&1u)!=0u),select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));let q=value*signs;let permutation=(code/8u)%6u;
 if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}if(permutation==2u){return q.yxz;}
 if(permutation==3u){return q.zxy;}if(permutation==4u){return q.yzx;}return q.zyx;}
fn coefficientBase(row:u32)->u32{return acceptedBank()*p.hierarchy.w+row*19u;}
${octreeSection63DirectionChannelWGSL}
// One indexed load replaces the eighteen-step scan that re-evaluated
// worldDirection per candidate. The table is the memoized scan result.
fn coefficientForDirection(row:u32,metric:Metric,direction:vec3i)->f32{
 let channel=section63ChannelForDirection(metric.transformAndFlags&63u,direction);
 if(channel>=18u){return 0.0;}
 return section63Coefficients[coefficientBase(row)+1u+channel];}
fn d4Four(a:f32,b:f32,c:f32,d:f32)->f32{let ab=min(a,b)+max(a,b);let cd=min(c,d)+max(c,d);return min(ab,cd)+max(ab,cd);}
fn sorted18Sum(values:array<f32,18>)->f32{var sorted=values;for(var i=1u;i<18u;i+=1u){let value=sorted[i];var j=i;loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}var sum=0.;for(var i=0u;i<18u;i+=1u){sum+=sorted[i];}return sum;}
fn sorted8Sum(values:array<f32,8>)->f32{var sorted=values;for(var i=1u;i<8u;i+=1u){let value=sorted[i];var j=i;loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}var sum=0.;for(var i=0u;i<8u;i+=1u){sum+=sorted[i];}return sum;}
fn sortedCoefficientSum(row:u32)->f32{var values:array<f32,18>;let base=coefficientBase(row);for(var channel=0u;channel<18u;channel+=1u){values[channel]=section63Coefficients[base+1u+channel];}return sorted18Sum(values);}
fn d4CoefficientSum(row:u32,m:Metric)->f32{
 let horizontal=d4Four(coefficientForDirection(row,m,vec3i(1,0,0)),coefficientForDirection(row,m,vec3i(-1,0,0)),coefficientForDirection(row,m,vec3i(0,0,1)),coefficientForDirection(row,m,vec3i(0,0,-1)));
 let vertical=coefficientForDirection(row,m,vec3i(0,1,0))+coefficientForDirection(row,m,vec3i(0,-1,0));
 let upper=d4Four(coefficientForDirection(row,m,vec3i(1,1,0)),coefficientForDirection(row,m,vec3i(-1,1,0)),coefficientForDirection(row,m,vec3i(0,1,1)),coefficientForDirection(row,m,vec3i(0,1,-1)));
 let lower=d4Four(coefficientForDirection(row,m,vec3i(1,-1,0)),coefficientForDirection(row,m,vec3i(-1,-1,0)),coefficientForDirection(row,m,vec3i(0,-1,1)),coefficientForDirection(row,m,vec3i(0,-1,-1)));
 let diagonal=d4Four(coefficientForDirection(row,m,vec3i(1,0,1)),coefficientForDirection(row,m,vec3i(-1,0,-1)),coefficientForDirection(row,m,vec3i(1,0,-1)),coefficientForDirection(row,m,vec3i(-1,0,1)));
 return((horizontal+vertical)+upper)+(lower+diagonal);}
fn directWorldTerm(row:u32,m:Metric,q:vec3u,l:u32,page:u32,levelDims:vec3i,slotBase:u32,x:f32,direction:vec3i)->f32{
 let channel=section63ChannelForDirection(m.transformAndFlags&63u,direction);if(channel>=18u){return 0.0;}
 let c=section63Coefficients[coefficientBase(row)+1u+channel];if(c==0.0){return 0.0;}
 let targetQ=vec3i(q)+direction;if(any(targetQ<vec3i(0))||any(targetQ>=levelDims)){reportAt(2u,27u,row);return 0.0;}
 let slot=pageSlot(l,page,q,vec3u(targetQ),row);if(slot==INVALID){reportAt(2u,28u,row);return 0.0;}
 let flags=state[atBase(FLAGS,slotBase,slot)];if((flags&MG_ONLY)!=0u){return 0.0;}
 let encoded=state[atBase(OWNER,slotBase,slot)];if(encoded==0u||encoded>capacity()){reportAt(2u,29u,row);return 0.0;}
 return c*(x-inputVector[encoded-1u]);}
fn d4DirectSum(row:u32,m:Metric,q:vec3u,l:u32,page:u32,levelDims:vec3i,slotBase:u32,x:f32)->f32{
 let horizontal=d4Four(directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(1,0,0)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(-1,0,0)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(0,0,1)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(0,0,-1)));
 let vertical=directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(0,1,0))+directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(0,-1,0));
 let upper=d4Four(directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(1,1,0)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(-1,1,0)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(0,1,1)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(0,1,-1)));
 let lower=d4Four(directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(1,-1,0)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(-1,-1,0)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(0,-1,1)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(0,-1,-1)));
 let diagonal=d4Four(directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(1,0,1)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(-1,0,-1)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(1,0,-1)),directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,vec3i(-1,0,1)));
 return((horizontal+vertical)+upper)+(lower+diagonal);}
fn sortedDirectSum(row:u32,m:Metric,q:vec3u,l:u32,page:u32,levelDims:vec3i,slotBase:u32,x:f32)->f32{var values:array<f32,18>;for(var channel=0u;channel<18u;channel+=1u){values[channel]=directWorldTerm(row,m,q,l,page,levelDims,slotBase,x,worldDirection(canonicalDirection(channel),m.transformAndFlags&63u));}return sorted18Sum(values);}
// "Which row does this stencil direction reach", as the once-per-epoch image
// builder asks it: every guard the inline walk applies, in the inline walk's
// order, with each report returned as a stage code instead of raised, so a
// builder that must not touch the solver control can carry it to the consumer.
//
// The dependent part - pageNeighbour, the origin compare, brickRecord, the two
// brick-mask loads and the ranked-slot indirection - is not transcribed here:
// it is pageSlotCoded, the one definition the inline walk's pageSlot also
// calls. That shared call is what keeps the compiled address equal to the
// address the walk would have produced, and
// tests/webgpu-octree-operator-image-differential.test.ts applies both forms to
// the same vectors on the same published topology to require it.
//
// levelDims and slotBase stay parameters because they are dispatch-invariant
// per row; both are pure functions of l.
fn resolveDirectChannel(l:u32,page:u32,q:vec3u,levelDims:vec3i,slotBase:u32,
 transform:u32,channel:u32)->u32{
 let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),transform);
 if(any(targetQ<vec3i(0))||any(targetQ>=levelDims)){return channelCode(27u,0u);}
 let resolved=pageSlotCoded(l,page,q,vec3u(targetQ));
 if(resolved.x==INVALID){return channelCode(28u,resolved.y);}
 let flags=state[atBase(FLAGS,slotBase,resolved.x)];
 if((flags&MG_ONLY)!=0u){return CHANNEL_SKIP;}
 let encoded=state[atBase(OWNER,slotBase,resolved.x)];
 if(encoded==0u||encoded>capacity()){return channelCode(29u,0u);}
 return encoded-1u;}
// A topology-regular row has only its six Cartesian faces.  The accepted page
// transaction already proved physical adjacency and every brick's mask/rank
// publication, so repeating pageSlotCoded's page-origin and general-direction
// checks on those six faces is redundant.  This direct page-neighbour/rank
// path retains every capacity/owner guard and fails closed with the same codes.
fn resolveRegularChannel(l:u32,page:u32,q:vec3u,slotBase:u32,transform:u32,channel:u32)->u32{
 let direction=worldDirection(canonicalDirection(channel),transform);let absolute=abs(direction);
 if(absolute.x+absolute.y+absolute.z!=1){return CHANNEL_SKIP;}
 let targetQ=vec3i(q)+direction;if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){return channelCode(27u,0u);}
 let targetCell=vec3u(targetQ);let shape=vec3u(8u,8u,4u);let delta=vec3i(targetCell/shape)-vec3i(q/shape);
 if(any(delta<vec3i(-1))||any(delta>vec3i(1))){return channelCode(28u,21u);}
 let ordinal=u32(delta.x+1)+3u*(u32(delta.y+1)+3u*u32(delta.z+1));let physical=pageNeighbour(l,page,ordinal);
 if(physical==INVALID){return channelCode(28u,0u);}
 let record=brickRecord(l,targetCell);let bit=localBit(targetCell);let low=topology[record+1u];let high=topology[record+2u];
 if(((select(low,high,bit>=32u)>>(bit&31u))&1u)==0u){return channelCode(28u,0u);}
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
 let ranked=topology[record+3u]+rank;let slot=topology[rankedSlotsBase()+levelBase(l)+ranked];
 if(slot>=levelCapacity(l)){return channelCode(28u,23u);}
 let flags=state[atBase(FLAGS,slotBase,slot)];if((flags&MG_ONLY)!=0u){return CHANNEL_SKIP;}
 let encoded=state[atBase(OWNER,slotBase,slot)];if(encoded==0u||encoded>capacity()){return channelCode(29u,0u);}
 return encoded-1u;}
// Replay a coded resolution's reports in the order the inline walk raised
// them: pageSlot's own stage first, then the caller's. reportAt claims the
// first error by compare-exchange, so preserving per-lane order preserves the
// claimed stage.
fn reportChannelCode(code:u32,row:u32){
 if(code<CHANNEL_CODE_BASE){return;}
 let secondary=(code>>8u)&0xffu;let primary=code&0xffu;
 if(secondary!=0u){reportAt(2u,secondary,row);}
 if(primary!=0u){reportAt(2u,primary,row);}}
// "Which fine row does this (child, candidate) alias reach, and through which
// Section 6.3 channel", as the once-per-epoch image builder asks it: every
// guard the inline walk applies, in the inline walk's order, with each report
// returned as a code instead of raised, so a builder that must not touch the
// solver control can hand it to the consumer.
//
// The dependent part -- the ghost resolution and the active-neighbour
// resolution -- is not transcribed here: both are pageSlotCoded, the same one
// definition the inline walk's pageSlot calls. That shared call is what keeps
// the compiled edge equal to the edge the walk would have produced.
//
// A candidate whose direction has no channel in this row's transform resolves
// to a zero coefficient, so the walk's c>0.0 test leaves the staged word at
// zero and raises nothing -- which is precisely CHANNEL_SKIP, so it is encoded
// as one rather than given a representation of its own.
fn resolveAdjointCandidate(row:u32,child:u32,candidate:u32)->u32{
 let h=geometry[row];let l=countTrailingZeros(h.y);if(l==0u){return CHANNEL_SKIP;}
 let fine=l-1u;let q=originOf(h)/(1u<<l);
 let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
 let ghostPage=pageFor(fine,ghostQ);if(ghostPage==INVALID){return CHANNEL_SKIP;}
 if(ghostPage>=levelCapacity(fine)){return channelCode(31u,0u);}
 let ghost=pageSlotCoded(fine,ghostPage,ghostQ,ghostQ);
 if(ghost.x==INVALID){return channelCode(0u,ghost.y);}
 let fineBase=levelBase(fine);
 if((state[atBase(FLAGS,fineBase,ghost.x)]&GHOST)==0u
  ||state[atBase(OWNER,fineBase,ghost.x)]!=row+1u){return CHANNEL_SKIP;}
 let delta=canonicalDirection(candidate);let activeQ=vec3i(ghostQ)-delta;
 if(any(activeQ<vec3i(0))||any(activeQ>=vec3i(dims(fine)))){return CHANNEL_SKIP;}
 let resolvedActive=pageSlotCoded(fine,ghostPage,ghostQ,vec3u(activeQ));
 if(resolvedActive.x==INVALID){return channelCode(0u,resolvedActive.y);}
 if((state[atBase(FLAGS,fineBase,resolvedActive.x)]&ACTIVE)==0u){return CHANNEL_SKIP;}
 let encoded=state[atBase(OWNER,fineBase,resolvedActive.x)];
 if(encoded==0u||encoded>capacity()){return channelCode(24u,0u);}
 let other=encoded-1u;
 let channel=section63ChannelForDirection(metrics[other].transformAndFlags&63u,delta);
 if(channel>=18u){return CHANNEL_SKIP;}
 return other|(channel<<ADJOINT_CHANNEL_SHIFT);}
// Destination-owned GhostValueAccumulate. A 2:1 coarse leaf has at most
// eight fine-level aliases. Each alias gathers the (at most eighteen) active
// page neighbours that point to it. This is E^T by construction: it reads the
// same owner incidence used by propagation and performs no scatter atomic.
fn finerAdjointTerm(row:u32,q:vec3u,l:u32,x:f32,child:u32,delta:vec3i)->f32{
 if(l==0u){return 0.0;}let fine=l-1u;let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
 let ghostPage=pageFor(fine,ghostQ);if(ghostPage==INVALID){return 0.0;}if(ghostPage>=levelCapacity(fine)){reportAt(2u,31u,row);return 0.0;}
 let ghost=pageSlot(fine,ghostPage,ghostQ,ghostQ,row);let fineBase=levelBase(fine);
 if(ghost==INVALID||(state[atBase(FLAGS,fineBase,ghost)]&GHOST)==0u||state[atBase(OWNER,fineBase,ghost)]!=row+1u){return 0.0;}
 let activeQ=vec3i(ghostQ)-delta;if(any(activeQ<vec3i(0))||any(activeQ>=vec3i(dims(fine)))){return 0.0;}
 let activeSlot=pageSlot(fine,ghostPage,ghostQ,vec3u(activeQ),row);
 if(activeSlot==INVALID||(state[atBase(FLAGS,fineBase,activeSlot)]&ACTIVE)==0u){return 0.0;}
 let encoded=state[atBase(OWNER,fineBase,activeSlot)];if(encoded==0u||encoded>capacity()){reportAt(2u,24u,row);return 0.0;}
 let other=encoded-1u;let c=coefficientForDirection(other,metrics[other],delta);
 return select(0.0,c*(x-inputVector[other]),c>0.0);}
fn d4FinerChildSum(row:u32,q:vec3u,l:u32,x:f32,child:u32)->f32{
 let horizontal=d4Four(finerAdjointTerm(row,q,l,x,child,vec3i(1,0,0)),finerAdjointTerm(row,q,l,x,child,vec3i(-1,0,0)),finerAdjointTerm(row,q,l,x,child,vec3i(0,0,1)),finerAdjointTerm(row,q,l,x,child,vec3i(0,0,-1)));
 let vertical=finerAdjointTerm(row,q,l,x,child,vec3i(0,1,0))+finerAdjointTerm(row,q,l,x,child,vec3i(0,-1,0));
 let upper=d4Four(finerAdjointTerm(row,q,l,x,child,vec3i(1,1,0)),finerAdjointTerm(row,q,l,x,child,vec3i(-1,1,0)),finerAdjointTerm(row,q,l,x,child,vec3i(0,1,1)),finerAdjointTerm(row,q,l,x,child,vec3i(0,1,-1)));
 let lower=d4Four(finerAdjointTerm(row,q,l,x,child,vec3i(1,-1,0)),finerAdjointTerm(row,q,l,x,child,vec3i(-1,-1,0)),finerAdjointTerm(row,q,l,x,child,vec3i(0,-1,1)),finerAdjointTerm(row,q,l,x,child,vec3i(0,-1,-1)));
 let diagonal=d4Four(finerAdjointTerm(row,q,l,x,child,vec3i(1,0,1)),finerAdjointTerm(row,q,l,x,child,vec3i(-1,0,-1)),finerAdjointTerm(row,q,l,x,child,vec3i(1,0,-1)),finerAdjointTerm(row,q,l,x,child,vec3i(-1,0,1)));
 return((horizontal+vertical)+upper)+(lower+diagonal);}
fn sortedFinerChildSum(row:u32,q:vec3u,l:u32,x:f32,child:u32)->f32{
 var values:array<f32,18>;
 for(var candidate=0u;candidate<18u;candidate+=1u){
  values[candidate]=finerAdjointTerm(row,q,l,x,child,canonicalDirection(candidate));
 }
 return sorted18Sum(values);
}
fn finerAdjoint(row:u32,h:vec4u,q:vec3u,l:u32,x:f32)->f32{
 if(l==0u){return 0.0;}
 var children:array<f32,8>;
 for(var child=0u;child<8u;child+=1u){children[child]=sortedFinerChildSum(row,q,l,x,child);}
 return sorted8Sum(children);}

// One linear, prefetchable load replaces the five-to-seven-deep dependent
// chase this used to run per channel: brickRecord, two brick-mask loads and
// the ranked-slot indirection, once per direction, on ~100 SpMV-equivalents
// per solve. The chase now runs once per accepted topology epoch, in
// buildAccurateOperatorRows, through the same resolveDirectChannel.
//
// Values and evaluation order are untouched: the coefficient is the same load
// from the same bank, the operand is the same inputVector element, and the
// product is the same single expression, so the backend contracts the same
// multiply-add. Only the address arrives differently.
fn stageDirectTerm(row:u32,channel:u32){let destination=row*162u+channel;
 if(destination>=arrayLength(&accurateTerms)||row>=capacity()||row>=arrayLength(&geometry)
  ||row>=arrayLength(&metrics)||row>=arrayLength(&inputVector)){reportAt(2u,25u,row);return;}
 let m=metrics[row];let base=coefficientBase(row);
 if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u||base+19u>arrayLength(&section63Coefficients)){reportAt(1u,26u,row);return;}
 let c=section63Coefficients[base+1u+channel];var term=0.0;
 if(c!=0.0){let image=operatorRowBase(row);
  if(image+OPERATOR_ROW_WORDS>arrayLength(&operatorRows)||(operatorRows[image]&~HYBRID_REGULAR_ROW)!=0u){reportAt(2u,31u,row);return;}
  let code=operatorRows[image+1u+channel];
  if(code>=CHANNEL_CODE_BASE){reportChannelCode(code,row);}
  else{term=c*(inputVector[row]-inputVector[code]);}}
 accurateTerms[destination]=term;
}

// Compile the operator once per accepted topology epoch.
//
// Encoded by encodeReadySetupCommit, in the pass that commits the hierarchy,
// after commitCandidateLevels and commitChangedL1 have published this epoch's
// pages, brick masks, ranked slots, FLAGS and OWNER. Those are the only inputs
// the resolution reads, and the only writer of any of them is that same commit,
// so an image built here is exact for every apply until the next commit - and
// the next commit re-runs this. There is no staleness window and no epoch
// stamp to get wrong.
//
// One lane per (row, word). Word 0 publishes the row's page-resolution status,
// words 1..18 the eighteen channel destinations.
//
// This is a pure function of (topology, state, geometry, metrics) over the
// whole row capacity: it depends on no publication word and takes no view of
// which rows are live, so it cannot mark a live row dead on a step where a
// control has not landed yet. Rows the apply never visits get compiled and
// never read. The h.y guard is the one addition to applyRow's own metric gate:
// countTrailingZeros(0) is 32, and a shift of 32 is indeterminate in WGSL, so a
// row with no size is refused rather than compiled from an indeterminate level.
fn buildOperatorRow(row:u32,word:u32,predecessor:u32){
 let image=operatorRowBase(row);
 if(word>=OPERATOR_ROW_WORDS||image+OPERATOR_ROW_WORDS>arrayLength(&operatorRows)
  ||row>=capacity()||row>=arrayLength(&geometry)||row>=arrayLength(&metrics)){return;}
 if(predecessor!=INVALID){let carried=operatorRows[priorOperatorRowBase(predecessor)+word];
  if(word!=0u&&carried<CHANNEL_CODE_BASE){let remapped=remapPersistentDestination(carried);
   if(remapped!=INVALID){operatorRows[image+word]=remapped;return;}}}
 let h=geometry[row];let m=metrics[row];
 var status=31u;
 var l=0u;var q=vec3u(0u);var page=INVALID;
 if(h.y!=0u&&m.error==0u&&(m.transformAndFlags&0x80000000u)!=0u){
  l=countTrailingZeros(h.y);q=originOf(h)/(1u<<l);page=pageFor(l,q);
  if(page!=INVALID&&page<levelCapacity(l)){status=0u;}}
 let regular=m.caseId==0u&&(m.transformAndFlags&0x3f00u)==0u;
 if(word==0u&&regular&&status==0u){let base=coefficientBase(row);
  if(base+19u>arrayLength(&section63Coefficients)){reportAt(1u,26u,row);status=26u;}
  else{for(var channel=6u;channel<18u;channel+=1u){
   if(section63Coefficients[base+1u+channel]!=0.0){reportAt(1u,32u,row);status=32u;break;}}}}
 if(word==0u){operatorRows[image]=status|select(0u,HYBRID_REGULAR_ROW,regular&&status==0u);return;}
 if(status!=0u){operatorRows[image+word]=CHANNEL_SKIP;return;}
 if(regular){operatorRows[image+word]=resolveRegularChannel(
   l,page,q,levelBase(l),m.transformAndFlags&63u,word-1u);
 }else{operatorRows[image+word]=resolveDirectChannel(
   l,page,q,vec3i(dims(l)),levelBase(l),m.transformAndFlags&63u,word-1u);}
}
@compute @workgroup_size(32) fn buildAccurateOperatorRows(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let item=linearGroup(wg,groups);let row=unionRow(item);
 if(lane==0u){compiledImagePredecessor=select(INVALID,
  persistentImagePredecessor(row,operatorImageBank()),row!=INVALID);}
 workgroupBarrier();
 if(row!=INVALID&&lane<OPERATOR_ROW_WORDS){buildOperatorRow(row,lane,compiledImagePredecessor);}
 if(item==0u&&lane==0u&&arrayLength(&imageEpochs)>=2u){
  atomicStore(&imageEpochs[operatorImageBank()],accepted[3]+1u);}
}

// E^T, one lane per (row, child, candidate) instead of one lane per (row,
// child) walking eighteen candidates in sequence.
//
// The walk each lane used to own was eighteen INDEPENDENT dependent chains run
// back to back: nothing in candidate k+1 needs candidate k, so the sequencing
// bought nothing and the launch was 152 chains deep on
// ceil(transitionRows*8/64) workgroups. Giving each candidate its own lane
// issues them concurrently: the depth per lane drops from nineteen chains to
// three, and the launch widens eighteen-fold onto a machine on which nothing
// in this solve is saturated.
//
// Bit-exact by construction, and deliberately NOT a fold change: each lane
// writes exactly the one accurateTerms word its (child, candidate) wrote
// before, from the identical expression on the identical operands, and
// finalizeStagedRow still sums them in ascending child-then-candidate order in
// one lane's registers. Nothing is staged through workgroup memory - doing that
// to these terms is what turned one fused multiply-accumulate into a bare
// multiply plus a bare add and moved peak speed 7.8269 -> 7.5066
// (POWER_LIQUIDS_ULTIMATE_M1MAX, refuted lever 10).
//
// The per-lane cost used to be the ghost resolution (pageFor + pageSlot)
// repeated by the eighteen lanes of a child rather than shared. Both that
// resolution and the per-candidate active-neighbour resolution are now
// compiled once per accepted topology epoch, so a lane loads one word where it
// used to run two five-to-seven-deep dependent chains. On the mini lane this
// stage is entered by 165 merged-band applies per advance and measured 1.584
// ms/advance -- the largest single pressure line in the isolated capture -- so
// the chase is paid 165 times for a topology that changed once.
//
// Values and evaluation order are untouched: the coefficient is the same load
// from the same accepted bank, the operand is the same inputVector element,
// the product is the same single expression, and the staged word is the same
// index, so finalizeStagedRow folds an unchanged array in unchanged order.
fn stageAdjointCandidate(row:u32,child:u32,candidate:u32){let base=row*162u+18u+child*18u;
 if(base+18u>arrayLength(&accurateTerms)||row>=capacity()||row>=arrayLength(&geometry)
  ||row>=arrayLength(&metrics)||row>=arrayLength(&inputVector)){reportAt(2u,25u,row);return;}
 let destination=base+candidate;
 accurateTerms[destination]=0.0;
 let image=adjointRowBase(row)+child*18u+candidate;
 if(image>=arrayLength(&adjointRows)){reportAt(2u,31u,row);return;}
 let code=adjointRows[image];
 if(code>=CHANNEL_CODE_BASE){reportChannelCode(code,row);return;}
 let other=code&ADJOINT_ROW_MASK;
 let c=section63Coefficients[coefficientBase(other)+1u+(code>>ADJOINT_CHANNEL_SHIFT)];
 if(c>0.0){accurateTerms[destination]=c*(inputVector[row]-inputVector[other]);}
}
// Compile the fine-adjoint half of the operator, in the same commit and on the
// same inputs as buildAccurateOperatorRows: one lane per (row, child,
// candidate), writing the staging layout's own index so image word and staged
// word never need translating.
fn buildAdjointRow(row:u32,word:u32,predecessor:u32){
 let image=adjointRowBase(row);
 if(word>=ADJOINT_ROW_WORDS||image+ADJOINT_ROW_WORDS>arrayLength(&adjointRows)
  ||row>=capacity()||row>=arrayLength(&geometry)||row>=arrayLength(&metrics)){return;}
 if(predecessor!=INVALID){let carried=adjointRows[priorAdjointRowBase(predecessor)+word];
  if(carried<CHANNEL_CODE_BASE){let remapped=remapPersistentDestination(carried&ADJOINT_ROW_MASK);
   if(remapped!=INVALID){adjointRows[image+word]=remapped|(carried&~ADJOINT_ROW_MASK);return;}}}
 // The same liveness test buildOperatorRow uses. An unpublished row carries a
 // zero size word, and countTrailingZeros(0u) is 32, so the level has to be
 // rejected before anything derives a level from it. Such a row is never in a
 // transition workset, so no apply ever reads the word this writes.
 if(geometry[row].y==0u){adjointRows[image+word]=CHANNEL_SKIP;return;}
 adjointRows[image+word]=resolveAdjointCandidate(row,word/18u,word%18u);
}
@compute @workgroup_size(64) fn buildAccurateAdjointRows(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let item=linearGroup(wg,groups);let row=transitionUnionRow(item);
 if(lane==0u){compiledImagePredecessor=select(INVALID,
  persistentImagePredecessor(row,adjointImageBank()),row!=INVALID);}
 workgroupBarrier();
 if(row!=INVALID){for(var word=lane;word<ADJOINT_ROW_WORDS;word+=64u){buildAdjointRow(row,word,compiledImagePredecessor);}}
 if(item==0u&&lane==0u&&arrayLength(&imageEpochs)>=2u){
  atomicStore(&imageEpochs[adjointImageBank()],accepted[3]+1u);}
}
// Reference arm of the operator-image differential. This is
// stageAdjointCandidate's body from before the compiled image: it re-runs both
// page/brick/rank chases on every call. Production never encodes it. Keep it
// byte-faithful to what the image replaced; it is the only executable
// statement of what the image has to agree with.
fn stageAdjointCandidateByChase(row:u32,child:u32,candidate:u32){let base=row*162u+18u+child*18u;
 if(base+18u>arrayLength(&accurateTerms)||row>=capacity()||row>=arrayLength(&geometry)
  ||row>=arrayLength(&metrics)||row>=arrayLength(&inputVector)){reportAt(2u,25u,row);return;}
 let destination=base+candidate;
 accurateTerms[destination]=0.0;
 let h=geometry[row];let l=countTrailingZeros(h.y);if(l==0u){return;}let fine=l-1u;
 let q=originOf(h)/(1u<<l);let x=inputVector[row];let fineDims=vec3i(dims(fine));let fineBase=levelBase(fine);
 let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
 let ghostPage=pageFor(fine,ghostQ);if(ghostPage==INVALID){return;}
 if(ghostPage>=levelCapacity(fine)){reportAt(2u,31u,row);return;}
 let ghost=pageSlot(fine,ghostPage,ghostQ,ghostQ,row);
 if(ghost==INVALID||(state[at(FLAGS,fine,ghost)]&GHOST)==0u||state[at(OWNER,fine,ghost)]!=row+1u){return;}
 let delta=canonicalDirection(candidate);let activeQ=vec3i(ghostQ)-delta;
 if(any(activeQ<vec3i(0))||any(activeQ>=fineDims)){return;}
 let activeSlot=pageSlot(fine,ghostPage,ghostQ,vec3u(activeQ),row);
 if(activeSlot==INVALID||(state[atBase(FLAGS,fineBase,activeSlot)]&ACTIVE)==0u){return;}
 let encoded=state[atBase(OWNER,fineBase,activeSlot)];if(encoded==0u||encoded>capacity()){reportAt(2u,24u,row);return;}
 let other=encoded-1u;let c=coefficientForDirection(other,metrics[other],delta);
 if(c>0.0){accurateTerms[destination]=c*(x-inputVector[other]);}
}

struct StagedRowFold{value:f32,valid:bool}
fn stagedDirectWorldTerm(row:u32,m:Metric,direction:vec3i)->f32{
 let channel=section63ChannelForDirection(m.transformAndFlags&63u,direction);
 if(channel>=18u){return 0.;}return accurateTerms[row*162u+channel];
}
fn stagedD4DirectSum(row:u32,m:Metric)->f32{
 let horizontal=d4Four(stagedDirectWorldTerm(row,m,vec3i(1,0,0)),stagedDirectWorldTerm(row,m,vec3i(-1,0,0)),stagedDirectWorldTerm(row,m,vec3i(0,0,1)),stagedDirectWorldTerm(row,m,vec3i(0,0,-1)));
 let vertical=stagedDirectWorldTerm(row,m,vec3i(0,1,0))+stagedDirectWorldTerm(row,m,vec3i(0,-1,0));
 let upper=d4Four(stagedDirectWorldTerm(row,m,vec3i(1,1,0)),stagedDirectWorldTerm(row,m,vec3i(-1,1,0)),stagedDirectWorldTerm(row,m,vec3i(0,1,1)),stagedDirectWorldTerm(row,m,vec3i(0,1,-1)));
 let lower=d4Four(stagedDirectWorldTerm(row,m,vec3i(1,-1,0)),stagedDirectWorldTerm(row,m,vec3i(-1,-1,0)),stagedDirectWorldTerm(row,m,vec3i(0,-1,1)),stagedDirectWorldTerm(row,m,vec3i(0,-1,-1)));
 let diagonal=d4Four(stagedDirectWorldTerm(row,m,vec3i(1,0,1)),stagedDirectWorldTerm(row,m,vec3i(-1,0,-1)),stagedDirectWorldTerm(row,m,vec3i(1,0,-1)),stagedDirectWorldTerm(row,m,vec3i(-1,0,1)));
 return((horizontal+vertical)+upper)+(lower+diagonal);
}
fn stagedSortedDirectSum(row:u32)->f32{var values:array<f32,18>;for(var channel=0u;channel<18u;channel+=1u){values[channel]=accurateTerms[row*162u+channel];}return sorted18Sum(values);}
fn stagedSortedAdjointSum(row:u32)->f32{
 var children:array<f32,8>;
 for(var child=0u;child<8u;child+=1u){
  var values:array<f32,18>;
  for(var candidate=0u;candidate<18u;candidate+=1u){
   values[candidate]=accurateTerms[row*162u+18u+child*18u+candidate];
  }
  children[child]=sorted18Sum(values);
 }
 return sorted8Sum(children);
}
fn foldStagedRow(row:u32)->StagedRowFold{
 if(row>=capacity()||row>=arrayLength(&geometry)||row>=arrayLength(&metrics)
  ||row>=arrayLength(&inputVector)||row*162u+162u>arrayLength(&accurateTerms)){
  reportAt(2u,25u,row);return StagedRowFold(0.0,false);}
 let h=geometry[row];let m=metrics[row];let base=coefficientBase(row);
 if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u||base+19u>arrayLength(&section63Coefficients)){
  reportAt(1u,26u,row);return StagedRowFold(0.0,false);}
 let x=inputVector[row];let sum=sortedCoefficientSum(row);
 var value=max(0.0,section63Coefficients[base]-sum)*x+stagedSortedDirectSum(row);
 if(m.caseId!=0u){value+=stagedSortedAdjointSum(row);}
 if(!finite(value)){reportAt(4u,30u,row);return StagedRowFold(0.0,false);}
 return StagedRowFold(value,true);
}
fn finalizeStagedRow(row:u32){
 let folded=foldStagedRow(row);if(folded.valid){outputVector[row]=folded.value;}
}
fn finalizeStagedResidualRow(row:u32){
 let folded=foldStagedRow(row);if(!folded.valid){return;}
 let residual=residualRhs[row]-folded.value;
 if(!finite(residual)){reportAt(4u,12u,row);}else{outputVector[row]=residual;}
}

fn stagedRowIdsBase()->u32{return capacity()*162u;}
fn stagedCountIndex()->u32{return capacity()*163u;}
fn validWorkCount(cls:u32)->u32{let base=worksetBase(cls);
 if(base+WORKSET_HEADER_WORDS>arrayLength(&worksets)||worksets[base]!=accepted[3]
  ||worksets[base+1u]>worksets[base+2u]){return 0u;}
 return worksets[base+1u];}
fn acceptedUnionCount()->u32{var count=0u;
 for(var cls=0u;cls<5u;cls+=1u){let base=worksetBase(cls);
  if(base+WORKSET_HEADER_WORDS>arrayLength(&worksets)||worksets[base]!=accepted[3]
   ||worksets[base+1u]>worksets[base+2u]){return 0u;}
  count+=worksets[base+1u];}
 return count;}
fn unionRow(item:u32)->u32{var remaining=item;
 for(var cls=0u;cls<5u;cls+=1u){let base=worksetBase(cls);
  if(base+WORKSET_HEADER_WORDS>arrayLength(&worksets)||worksets[base]!=accepted[3]
   ||worksets[base+1u]>worksets[base+2u]){return INVALID;}
  let count=worksets[base+1u];if(remaining<count){
   if(base+WORKSET_HEADER_WORDS+remaining>=arrayLength(&worksets)){return INVALID;}
   return worksets[base+WORKSET_HEADER_WORDS+remaining];}remaining-=count;}
 return INVALID;}
// A fine ghost can point to a coarse owner only at a coarse/fine interface.
// Section 6.3 classifies exactly those rows as transition classes 1 and 3;
// regular and physical-only rows therefore have an identically zero E^T tail.
fn transitionUnionCount()->u32{var count=0u;
 for(var index=0u;index<2u;index+=1u){let cls=select(1u,3u,index==1u);let base=worksetBase(cls);
  if(base+WORKSET_HEADER_WORDS>arrayLength(&worksets)||worksets[base]!=accepted[3]
   ||worksets[base+1u]>worksets[base+2u]){return 0u;}
  count+=worksets[base+1u];}
 return count;}
fn transitionUnionRow(item:u32)->u32{var remaining=item;
 for(var index=0u;index<2u;index+=1u){let cls=select(1u,3u,index==1u);let base=worksetBase(cls);
  if(base+WORKSET_HEADER_WORDS>arrayLength(&worksets)||worksets[base]!=accepted[3]
   ||worksets[base+1u]>worksets[base+2u]){return INVALID;}
  let count=worksets[base+1u];if(remaining<count){
   if(base+WORKSET_HEADER_WORDS+remaining>=arrayLength(&worksets)){return INVALID;}
   return worksets[base+WORKSET_HEADER_WORDS+remaining];}remaining-=count;}
 return INVALID;}
fn stageUnionItem(item:u32,count:u32,row:u32){let countIndex=stagedCountIndex();
 if(item==0u&&countIndex<arrayLength(&accurateTerms)){accurateTerms[countIndex]=bitcast<f32>(count);}
 let rowItem=item/18u;let channel=item%18u;if(rowItem>=count){return;}
 let rowIndex=stagedRowIdsBase()+rowItem;
 if(channel==0u&&rowIndex<arrayLength(&accurateTerms)){accurateTerms[rowIndex]=bitcast<f32>(row);}
 if(row!=INVALID){stageDirectTerm(row,channel);}}
// Reference arm of the operator-image differential. This is stageDirectTerm's
// body from before the compiled image: it re-runs the page/brick/rank chase on
// every call. Production never encodes it. It exists so
// tests/webgpu-octree-operator-image-differential.test.ts can apply BOTH
// addressings to the same vectors on the same published topology and require
// the staged terms to be bit-identical, which is the harness
// POWER_LIQUIDS_ULTIMATE_M1MAX asks for before the chase is trusted to be gone.
// Keep it byte-faithful to what the image replaced; it is the only executable
// statement of what the image has to agree with.
fn stageDirectTermByChase(row:u32,channel:u32){let destination=row*162u+channel;
 if(destination>=arrayLength(&accurateTerms)||row>=capacity()||row>=arrayLength(&geometry)
  ||row>=arrayLength(&metrics)||row>=arrayLength(&inputVector)){reportAt(2u,25u,row);return;}
 let h=geometry[row];let m=metrics[row];let base=coefficientBase(row);
 if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u||base+19u>arrayLength(&section63Coefficients)){reportAt(1u,26u,row);return;}
 let c=section63Coefficients[base+1u+channel];var term=0.0;
 if(c!=0.0){let l=countTrailingZeros(h.y);let q=originOf(h)/(1u<<l);let page=pageFor(l,q);
  if(page==INVALID||page>=levelCapacity(l)){reportAt(2u,31u,row);return;}
  let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),m.transformAndFlags&63u);
  if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){reportAt(2u,27u,row);}
  else{let slot=pageSlot(l,page,q,vec3u(targetQ),row);if(slot==INVALID){reportAt(2u,28u,row);}
   else{let slotBase=levelBase(l);if((state[atBase(FLAGS,slotBase,slot)]&MG_ONLY)==0u){let encoded=state[atBase(OWNER,slotBase,slot)];
    if(encoded==0u||encoded>capacity()){reportAt(2u,29u,row);}else{term=c*(inputVector[row]-inputVector[encoded-1u]);}}}}}
 accurateTerms[destination]=term;
}
fn stageUnionItemByChase(item:u32,count:u32,row:u32){let countIndex=stagedCountIndex();
 if(item==0u&&countIndex<arrayLength(&accurateTerms)){accurateTerms[countIndex]=bitcast<f32>(count);}
 let rowItem=item/18u;let channel=item%18u;if(rowItem>=count){return;}
 let rowIndex=stagedRowIdsBase()+rowItem;
 if(channel==0u&&rowIndex<arrayLength(&accurateTerms)){accurateTerms[rowIndex]=bitcast<f32>(row);}
 if(row!=INVALID){stageDirectTermByChase(row,channel);}}
@compute @workgroup_size(64) fn stageAcceptedUnionTerms(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let count=acceptedUnionCount();
 stageUnionItem(item,count,unionRow(item/18u));}
// Measurement-only counterpart to stageAcceptedUnionTerms. Production selects
// the image entry point; keeping both union shapes available lets one process
// price the cache without changing any workset or reduction geometry.
@compute @workgroup_size(64) fn stageAcceptedUnionTermsByChase(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let count=acceptedUnionCount();
 stageUnionItemByChase(item,count,unionRow(item/18u));}
@compute @workgroup_size(64) fn stageMergedBandTerms(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let count=validWorkCount(4u);
 stageUnionItem(item,count,workRow(item/18u,4u));}
// Differential reference only; never encoded. See stageDirectTermByChase.
@compute @workgroup_size(64) fn stageMergedBandTermsByChase(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let count=validWorkCount(4u);
 stageUnionItemByChase(item,count,workRow(item/18u,4u));}
// ADJOINT_LANES_PER_ROW = 8 children x 18 candidates. The dispatch records in
// octreeSPGridAccurateDispatchGateShader and in the Section 4.3 shell's
// prepareCorrectionDispatches publish transitionRows*144 lanes to match.
@compute @workgroup_size(64) fn stageAcceptedUnionAdjoints(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let count=transitionUnionCount();let rowItem=item/144u;
 if(rowItem<count){let row=transitionUnionRow(rowItem);if(row!=INVALID){stageAdjointCandidate(row,(item%144u)/18u,item%18u);}}}
@compute @workgroup_size(64) fn stageMergedBandAdjoints(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let count=transitionUnionCount();let rowItem=item/144u;
 if(rowItem<count){let row=transitionUnionRow(rowItem);if(row!=INVALID){stageAdjointCandidate(row,(item%144u)/18u,item%18u);}}}
// Differential reference only; never encoded. See stageAdjointCandidateByChase.
@compute @workgroup_size(64) fn stageMergedBandAdjointsByChase(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let count=transitionUnionCount();let rowItem=item/144u;
 if(rowItem<count){let row=transitionUnionRow(rowItem);if(row!=INVALID){stageAdjointCandidateByChase(row,(item%144u)/18u,item%18u);}}}
@compute @workgroup_size(64) fn finalizeStagedUnionRows(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let countIndex=stagedCountIndex();
 if(countIndex>=arrayLength(&accurateTerms)){reportAt(2u,25u,item);return;}
 let count=bitcast<u32>(accurateTerms[countIndex]);if(item>=count){return;}
 let rowIndex=stagedRowIdsBase()+item;if(rowIndex>=arrayLength(&accurateTerms)){reportAt(2u,25u,item);return;}
 let row=bitcast<u32>(accurateTerms[rowIndex]);if(row!=INVALID){finalizeStagedRow(row);}}
@compute @workgroup_size(64) fn finalizeStagedUnionResidualRows(@builtin(workgroup_id) wg:vec3u,
 @builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(stopped()){return;}let item=linearLane(wg,groups,lane);let countIndex=stagedCountIndex();
 if(countIndex>=arrayLength(&accurateTerms)){reportAt(2u,25u,item);return;}
 let count=bitcast<u32>(accurateTerms[countIndex]);if(item>=count){return;}
 let rowIndex=stagedRowIdsBase()+item;if(rowIndex>=arrayLength(&accurateTerms)){reportAt(2u,25u,item);return;}
 let row=bitcast<u32>(accurateTerms[rowIndex]);if(row!=INVALID){finalizeStagedResidualRow(row);}}

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
@group(0) @binding(24) var<storage,read> acceptedCoefficients:array<f32>;
@group(0) @binding(25) var<storage,read_write> touchedBrickKeys:array<u32>;
@group(0) @binding(26) var<storage,read_write> touchedBrickHeader:array<u32>;
@group(0) @binding(27) var<storage,read> touchedBrickRuns:array<u32>;
@group(0) @binding(28) var<storage,read> touchedBrickControl:array<u32>;
@group(0) @binding(29) var<storage,read_write> touchedPageKeys:array<u32>;
@group(0) @binding(30) var<storage,read_write> touchedPageHeader:array<u32>;
@group(0) @binding(31) var<storage,read_write> touchedPageRuns:array<u32>;
@group(0) @binding(32) var<storage,read> touchedPageControl:array<u32>;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;const INVALID=0xffffffffu;
const OVERFLOW=2u;const NONFINITE=4u;const NONPOSITIVE=8u;
const KEY=0u;const FLAGS=1u;const DIAG=2u;const XP=3u;const XM=4u;const YP=5u;const YM=6u;const ZP=7u;const ZM=8u;
const XYPP=9u;const XYPM=10u;const XYMP=11u;const XYMM=12u;const XZPP=13u;const XZPM=14u;const XZMP=15u;const XZMM=16u;
const YZPP=17u;const YZPM=18u;const YZMP=19u;const YZMM=20u;
const RHS=21u;const A=22u;const B=23u;const OWNER=24u;const SPECTRAL=25u;
const STATE_CHANNELS=26u;
const PAGE_X=8u;const PAGE_Y=8u;const PAGE_Z=4u;
const STRUCTURED_CANDIDATE_READY=0x5356454cu;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn stopped()->bool{return atomicLoad(&control[0])!=0u||atomicLoad(&control[1])!=0u;}
fn candidateSource()->bool{return (p.solve.y&1u)!=0u;}
fn touchedDirectory()->bool{return (p.solve.y&2u)!=0u;}
fn touchedTripwire()->bool{return (p.solve.y&4u)!=0u;}
fn sourceControlReady()->bool{if(arrayLength(&acceptedRows)<6u){return false;}if(!candidateSource()){return acceptedRows[0]==0u&&acceptedRows[3]!=0u;}return acceptedRows[0]==STRUCTURED_CANDIDATE_READY&&acceptedRows[4]!=0u;}
fn sourceGeneration()->u32{return select(0u,select(acceptedRows[3],acceptedRows[4],candidateSource()),sourceControlReady());}
fn reportAt(flag:u32,stage:u32,index:u32){atomicOr(&control[0],flag);for(var retry=0u;retry<16u;retry+=1u){
 let claim=atomicCompareExchangeWeak(&control[6],0u,stage);if(claim.exchanged){atomicStore(&control[7],index);return;}
 if(claim.old_value!=0u){return;}}}
fn report(flag:u32){reportAt(flag,60u,INVALID);}fn rows()->u32{return min(select(0u,acceptedRows[2],sourceControlReady()),p.capacity.x);}fn level()->u32{return p.dimsLevel.w;}
fn acceptedBank()->u32{return select(acceptedRows[4],acceptedRows[5],candidateSource())&1u;}
fn geometry(row:u32)->vec4u{return capturedGeometry[row];}
fn sourceRowGeometry(row:u32)->vec4u{return sourceGeometry[acceptedBank()*p.capacity.x+row];}
fn maxStride()->u32{return p.capacity.z;}fn levels()->u32{return p.capacity.y;}fn transferStride()->u32{return p.capacity.w;}
// The level stride is 2^l by construction, so the ceiling division is a shift.
// Apple GPUs emulate integer division; this is the identical u32 result.
fn dims(l:u32)->vec3u{let s=1u<<l;return (p.dimsLevel.xyz+vec3u(s-1u))>>vec3u(l);}
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
// Published stencil-neighbour column indices: the eighteen slots the setup
// builder resolved while it accumulated the eighteen coefficients, stored
// channel-major so channel k of a level is contiguous across slots exactly like
// the coefficient channel XP+k it belongs to. This is the CSR column-index
// vector for the rediscretized operator; the recurring correction reads it
// instead of re-walking the global brick/rank directory once per spoke.
fn neighbourBase()->u32{return rankedSlotsBase()+totalLevelSlots();}
fn neighbourAt(k:u32,l:u32,s:u32)->u32{return neighbourBase()+k*totalLevelSlots()+levelBase(l)+s;}
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
fn deltaOldRow(row:u32)->u32{let encoded=sourceDelta[p.delta.z+row]&0x3fffffffu;
 return select(INVALID,encoded-1u,encoded!=0u);}
fn deltaDirtyRowsOffset()->u32{return p.delta.w&0x7fffffffu;}
fn deltaDirtyCountWord()->u32{return select(5u,6u,(p.delta.w&0x80000000u)!=0u);}
fn deltaDirtyRow(index:u32)->u32{return sourceDelta[deltaDirtyRowsOffset()+index];}
fn deltaAccepted(n:u32)->bool{
 if(p.delta.x!=p.capacity.x||p.delta.y+16u>arrayLength(&sourceDelta)
  ||p.delta.z+p.delta.x>arrayLength(&sourceDelta)
  ||deltaDirtyRowsOffset()+p.delta.x>arrayLength(&sourceDelta)){return false;}
 let previous=deltaControl(1u);let carried=deltaControl(2u);let added=deltaControl(3u);
 let retired=deltaControl(4u);let dirty=deltaControl(deltaDirtyCountWord());
 return deltaControl(0u)==n&&deltaControl(7u)!=0u&&deltaControl(8u)==ROW_DELTA_VALID
  &&carried<=min(n,previous)&&n==carried+added&&n==previous+added-retired&&dirty<=n;
}
fn captureBootstrap()->bool{return capturePages.bootstrap!=0u;}
fn captureWorkCount()->u32{return select(deltaControl(deltaDirtyCountWord()),captureExpectedPages(),captureBootstrap());}
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
 let noRows=stopped();let base=levels()*DISPATCH_WORDS;let words=base+2u;
 for(var word=0u;word<words;word+=1u){
  var value=dispatchMeta[word];let local=word%DISPATCH_WORDS;
  if(noRows&&(local==2u||local==5u||local==9u)){value=0u;}
  correctionDispatch[word]=value;
 }
 correctionDispatch[base+2u]=select((rows()+63u)/64u,0u,noRows);
 correctionDispatch[base+3u]=1u;
 correctionDispatch[base+4u]=1u;
}
var<workgroup> captureLaneFlags:array<u32,64>;
var<workgroup> captureLaneFirst:array<u32,64>;
var<workgroup> captureLaneRange:array<vec2u,64>;
var<workgroup> captureLaneChanged:array<u32,64>;
fn coordKey(q:vec3u,l:u32)->u32{let d=dims(l);return q.x+d.x*(q.y+d.y*q.z)+1u;}
// Two divisions, not three: floor(floor(v/dx)/dy) == floor(v/(dx*dy)) exactly
// over the unsigned integers, and each remainder is recovered by one
// multiply-subtract from the quotient already in hand. Bit-identical to the
// modulo form, and it never forms the dx*dy product.
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;let row=v/d.x;let plane=row/d.y;
 return vec3u(v-row*d.x,row-plane*d.y,plane);}
fn insertionHash(key:u32,l:u32)->u32{var h=key*0x9e3779b1u;h=(h^(h>>16u))*0x7feb352du;return(h^(h>>15u))&(levelCapacity(l)-1u);}
// The published-arena definition of a coordinate's slot. Since Section 4.6 it
// has no recurring caller: buildCandidateStencils resolves the same relation
// once per epoch through its candidate-side twin cDirectoryLookup and publishes
// the answer, and commitCandidateLevelAt copies every word this reads from that
// same candidate arena, so the two agree term for term on an accepted epoch.
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
 writeCandidateSchedule(0u,select(captureWorkCount(),0u,captureFailed()),1u);
}
// Page-parallel L1 validation. One WORKGROUP owns one work item and its
// sixty-four lanes own that page's sixty-four rows, so the per-row
// validSection63Row scan - nineteen catalog loads each - runs across the
// machine instead of down one lane of one workgroup.
//
// This is the same computation, reassociated. The old single-workgroup kernel
// folded pageFlags with bitwise-or and pageFirst with min over rows in ascending
// order; both are associative and commutative over u32, and no float or atomic
// is involved, so the per-page fold below is bit-identical for any lane order.
// The visited work items, the visited rows, and every early-out are unchanged.
// The page record this writes is exactly what the fold used to write, so the
// work-list reduction that follows reads its own dispatch's output.
@compute @workgroup_size(64) fn planL1CaptureDelta(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 // The GPU-authored record folds exact work items into two dimensions and
 // pins X at 65,535 only when the live count needs a second dimension.
 let generation=captureGeneration();let work=wg.x+wg.y*65535u;
 // Out-of-range pages and non-unique work items contribute nothing to the page
 // records; reduceL1CaptureDelta re-derives their flags from the same work
 // list, so this dispatch owns only the row scan.
 //
 // The live predicate is workgroup-uniform by construction - it reads only wg.x and
 // dispatch-invariant storage - but WGSL's uniformity analysis cannot see that
 // through a storage load. So no lane returns early and the barrier below is
 // reached unconditionally by every lane, which needs no uniformity proof from
 // any implementation. The dispatch is one workgroup per work item, so the
 // lanes that skip here are whole workgroups exiting after two loads.
 let page=captureWorkPage(work);
 let live=work<captureWorkCount()&&page<captureExpectedPages()&&page<capturePageCount()
  &&captureWorkUnique(work,page);
 let begin=page*CAPTURE_PAGE_ROWS;let end=min(rows(),begin+CAPTURE_PAGE_ROWS);
 var laneFlags=0u;var laneFirst=levels();
 let r=begin+lane;
 if(live&&r<end){let source=sourceRowGeometry(r);let volume=p.dimsLevel.x*p.dimsLevel.y*p.dimsLevel.z;
  if(!validSection63Row(r)||source.x>=volume||source.y==0u||(source.y&(source.y-1u))!=0u){laneFlags|=2u;}
  else{
   let old=select(deltaOldRow(r),INVALID,captureBootstrap());
   // The direct resolved-row producer's compact dirty list is the sparse-identity
   // authority. A dirty fixed row can change a neighbor handle without moving
   // its cell, so rebuild its dependent sparse suffix instead of comparing a
   // second adjacency representation.
   var topologyChanged=true;
   var stencilChanged=true;
   if(!topologyChanged){let captured=capturedGeometry[old];topologyChanged=!sameL1Topology(source,captured);
    if(topologyChanged){laneFirst=min(laneFirst,min(firstTrailingBit(source.y),firstTrailingBit(captured.y)));}}
   else{laneFirst=0u;}
   if(topologyChanged){laneFlags|=4u;stencilChanged=true;}
   if(stencilChanged){laneFlags|=8u;laneFirst=min(laneFirst,firstTrailingBit(source.y));}
  }
 }
 captureLaneFlags[lane]=laneFlags;captureLaneFirst[lane]=laneFirst;workgroupBarrier();
 if(live&&lane==0u){var pageFlags=1u;var pageFirst=levels();
  for(var i=0u;i<64u;i+=1u){pageFlags|=captureLaneFlags[i];pageFirst=min(pageFirst,captureLaneFirst[i]);}
  capturePages.pages[page].changeStamp=generation;
  capturePages.pages[page].validationStamp=generation;
  capturePages.pages[page].copyStamp=(pageFlags&0xffu)|((pageFirst&0xffffu)<<16u);
 }}
// Work-list reduction. Identical traversal, identical fold order and identical
// finalization to the kernel this was split out of; the only difference is that
// the per-page flags now come from the record planL1CaptureDelta just wrote
// instead of from an inline row scan. Membership is still decided by the work
// list, never by a page stamp, so a stale record from an earlier capture in the
// same generation can never be mistaken for this one's.
@compute @workgroup_size(64) fn reduceL1CaptureDelta(@builtin(local_invocation_index) lane:u32){
 let generation=captureGeneration();let workCount=captureWorkCount();
 var changed=0u;var first=levels();var flags=0u;var firstRow=rows();var rowEnd=0u;
 for(var work=lane;work<workCount;work+=64u){let page=captureWorkPage(work);
  if(page>=captureExpectedPages()||page>=capturePageCount()){flags|=DELTA_ERROR;continue;}
  if(!captureWorkUnique(work,page)){continue;}
  let record=capturePages.pages[page].copyStamp;
  let pageFlags=record&0xffu;let pageFirst=(record>>16u)&0xffffu;
  let begin=page*CAPTURE_PAGE_ROWS;let end=min(rows(),begin+CAPTURE_PAGE_ROWS);
  if(capturePages.pages[page].validationStamp!=generation){flags|=DELTA_ERROR;continue;}
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
// Page-parallel fixed-geometry commit, matching planL1CaptureDelta: one
// workgroup per work item, one lane per row of the page it owns.
//
// The body is a pure copy - no reduction, no ordering, no shared accumulator -
// so the lane split needs no barrier and no equivalence argument beyond the row
// address. Each page still writes its own two record words exactly once, from
// lane 0, under the identical eligibility test.
@compute @workgroup_size(64) fn commitChangedL1(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 // The matching GPU-authored record uses the same fixed two-dimensional fold.
 let generation=captureGeneration();let work=wg.x+wg.y*65535u;
 let page=captureWorkPage(work);
 if(work>=captureWorkCount()||page>=captureExpectedPages()||!captureWorkUnique(work,page)
  ||capturePageStamp(page)!=generation
  ||capturePages.readyGeneration!=generation||captureFailed()){return;}
 let begin=page*CAPTURE_PAGE_ROWS;let end=min(rows(),begin+CAPTURE_PAGE_ROWS);
 let r=begin+lane;
 if(r<end){capturedGeometry[r]=sourceRowGeometry(r);}
 if(lane==0u){capturePages.pages[page].generation=generation;capturePages.pages[page].copyStamp=generation;}
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
// case/transform. Dynamic boundary fractions are deliberately not part of this
// topology fingerprint: an unchanged topology retires its coarse dirty flags,
// then rebuilds the level-zero stencil from the current accepted coefficients.
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
 // Aanjaneya et al. Section 4.3's first-order M1 shares the pressure unknowns
 // and boundary conditions with L2. Free-surface fractions can change without
 // a topology/case change, so the finest stencil must follow every publication.
 levelDelta[deltaAt(0u,1u)]=DELTA_STENCIL;
}
fn candidateScheduleBase()->u32{return levels()*DISPATCH_WORDS+5u;}
fn writeCandidateSchedule(record:u32,items:u32,lanes:u32){
 let blocks=(items+lanes-1u)/lanes;let x=min(65535u,blocks);let at=candidateScheduleBase()+record*3u;
 candidateDispatch[at]=x;candidateDispatch[at+1u]=select(1u,(blocks+x-1u)/x,x>0u);
 candidateDispatch[at+2u]=1u;
}
@compute @workgroup_size(1) fn prepareCandidateSchedules(){
 var topologyLevelItems=0u;var stencilLevelItems=0u;var transferSlotItems=0u;
 var stencilSlotItems=0u;var topologyRowItems=0u;var clearItems=0u;var brickItems=0u;
 var logicalPageItems=0u;var physicalPageItems=0u;var commitItems=0u;
 if(touchedDirectory()){touchedBrickHeader[0]=${RADIX_SORT_INPUT_VALID}u;
  touchedBrickHeader[1]=0u;touchedBrickHeader[2]=captureGeneration();touchedBrickHeader[3]=0u;}
 for(var l=0u;l<levels();l+=1u){
  let priorTransfers=candidateDispatch[l*DISPATCH_WORDS+1u];
  let priorSlots=candidateDispatch[l*DISPATCH_WORDS];
  let priorPages=candidateDispatch[l*DISPATCH_WORDS+8u];
  candidateDispatch[l*DISPATCH_WORDS+9u]=priorTransfers;
  candidateDispatch[l*DISPATCH_WORDS+10u]=priorSlots;
  candidateDispatch[l*DISPATCH_WORDS+11u]=priorPages;
  let topology=topologyDirty(l);let stencil=stencilDirty(l);
  let transfer=l+1u<levels()&&(topology||topologyDirty(l+1u));
  if(topology||stencil||transfer){commitItems=l+1u;}
  if(topology||transfer){topologyLevelItems=l+1u;}
  if(stencil){stencilLevelItems=l+1u;}
  var selected=count(l);
  if(topology){
   // Sixteen is the proven maximum support expansion per live row. This is a
   // launch bound only; the inserter remains the fail-closed capacity authority.
   let liveBound=16u*rows();
   selected=min(levelCapacity(l),liveBound);
   topologyRowItems=rows();
   clearItems=max(clearItems,max(max(priorSlots,priorPages),max(rows(),previousRows())));}
  if(stencil){stencilSlotItems=max(stencilSlotItems,selected);}
  if(transfer){var coarseSelected=count(l+1u);
   if(topologyDirty(l+1u)){coarseSelected=min(levelCapacity(l+1u),16u*rows());}
   transferSlotItems=max(transferSlotItems,max(selected,coarseSelected));
   clearItems=max(clearItems,max(priorSlots,priorTransfers));}
 }
 // The current directory builders still traverse dense brick/page identity on
 // dirty epochs. Their compact touched-identity replacement is a separate
 // cutover; clean epochs publish zero work here.
 if(topologyLevelItems!=0u){brickItems=p.totals.y;logicalPageItems=p.totals.z;
  physicalPageItems=p.totals.z;}
 let failed=captureFailed();
 writeCandidateSchedule(1u,select(clearItems,0u,failed),64u);
 writeCandidateSchedule(2u,select(topologyLevelItems,0u,failed),1u);
 writeCandidateSchedule(3u,select(transferSlotItems,0u,failed),64u);
 writeCandidateSchedule(4u,select(brickItems,0u,failed),64u);
 writeCandidateSchedule(5u,select(logicalPageItems,0u,failed),64u);
 writeCandidateSchedule(6u,select(physicalPageItems,0u,failed),64u);
 writeCandidateSchedule(7u,select(commitItems,0u,failed),1u);
 writeCandidateSchedule(8u,select(stencilLevelItems,0u,failed),1u);
 writeCandidateSchedule(9u,select(stencilSlotItems,0u,failed),64u);
 writeCandidateSchedule(10u,select(topologyRowItems,0u,failed),64u);
}
@compute @workgroup_size(64) fn clearCorrection(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){outputCorrection[r]=0.0;}}
fn cAt(c:u32,l:u32,s:u32)->u32{return c*totalLevelSlots()+levelBase(l)+s;}
fn cCount(l:u32)->u32{return candidateDispatch[l*DISPATCH_WORDS];}
fn cWorkSlot(l:u32,i:u32)->u32{return candidateTopology[workBase()+levelBase(l)+i];}
fn cRowMap(l:u32,r:u32)->u32{return candidateTopology[rowMapBase()+l*p.capacity.x+r];}
fn cLoadf(c:u32,l:u32,s:u32)->f32{return bitcast<f32>(candidateState[cAt(c,l,s)]);}
fn cStoref(c:u32,l:u32,s:u32,v:f32){candidateState[cAt(c,l,s)]=bitcast<u32>(v);}
fn candidateReport(l:u32){levelDelta[deltaAt(l,4u)]|=DELTA_ERROR;}
fn candidateReportCode(l:u32,code:u32){levelDelta[deltaAt(l,4u)]|=DELTA_ERROR|code;}
fn cMergeClass(index:u32,incoming:u32){let old=candidateState[index];var merged=MG_ONLY;
 if((old&ACTIVE)!=0u||(incoming&ACTIVE)!=0u){merged=ACTIVE;}else if((old&GHOST)!=0u||(incoming&GHOST)!=0u){merged=GHOST;}
 candidateState[index]=merged;}
// Running worklist counter for the ordered insertion owner. The claim path used
// to read candidateDispatch[l*DISPATCH_WORDS] and store back w+1 on every claim.
// That is a global read-after-write carried across iterations of a loop only ONE
// lane runs: the next claim cannot form its worklist index until the previous
// claim's store has landed, so ~1,473 full memory round trips sit end to end on
// the critical path of a single-lane replay with nothing resident to hide them.
// Holding the counter in workgroup storage for the level and publishing it once
// yields the identical worklist indices and the identical final count -- inside
// the dispatch the counter's only reader is that same ordered owner, and every
// consumer (selectedCount, the transfer scan, validateCandidateHierarchy) reads
// it from a LATER dispatch, after endClaims has stored it.
var<workgroup> claimCount:u32;
fn beginClaims(l:u32){claimCount=cCount(l);}
fn endClaims(l:u32){candidateDispatch[l*DISPATCH_WORDS]=claimCount;}
// cMergeClass against a zero word, term for term. A slot whose KEY is zero has
// zero FLAGS and zero OWNER: the candidate arena starts zeroed, the only other
// writers of those two channels are clearCandidateLevels (which zeroes all
// three together) and the claim path below (which sets KEY non-zero in the same
// step), so no slot can carry a class or an owner without a key. That makes the
// read-back on the fresh-claim path a load of a value already known.
fn cFreshClass(incoming:u32)->u32{
 if((incoming&ACTIVE)!=0u){return ACTIVE;}
 if((incoming&GHOST)!=0u){return GHOST;}
 return MG_ONLY;}
// Ordered open-address insertion, addressed by key. Every caller already holds
// the key; the coordinate form recomputed coordKey(decode(key)) per proposal --
// two emulated integer divisions and a multiply chain -- on the owner's critical
// path. encodedOwner is owner+1, or 0 for an insertion that claims no
// ownership; owner+1 is never 0, so the sentinel is unambiguous.
//
// Statement order is preserved exactly, including on the failure paths: the
// overflow check still runs after KEY and FLAGS are stored and before the
// counter advances, so an overflowing claim still leaves its key and class
// behind and still appends nothing, and an owner is still never written for a
// proposal that returned INVALID.
fn cClaimKey(l:u32,key:u32,flags:u32,encodedOwner:u32)->u32{
 var slot=insertionHash(key,l);
 for(var probe=0u;probe<256u;probe+=1u){
  let index=cAt(KEY,l,slot);let old=candidateState[index];
  if(old==key){
   cMergeClass(cAt(FLAGS,l,slot),flags);
   if(encodedOwner!=0u){
    let previous=candidateState[cAt(OWNER,l,slot)];
    if(previous!=encodedOwner){
     if(previous!=0u){candidateReport(l);return INVALID;}
     candidateState[cAt(OWNER,l,slot)]=encodedOwner;}}
   return slot;}
  if(old==0u){
   candidateState[index]=key;
   candidateState[cAt(FLAGS,l,slot)]=cFreshClass(flags);
   let w=claimCount;if(w>=levelCapacity(l)){candidateReport(l);return INVALID;}
   claimCount=w+1u;candidateTopology[workBase()+levelBase(l)+w]=slot;
   if(encodedOwner!=0u){candidateState[cAt(OWNER,l,slot)]=encodedOwner;}
   return slot;}
  slot=(slot+1u)&(levelCapacity(l)-1u);}
 candidateReport(l);return INVALID;}
// Coordinate-addressed wrappers, kept for the one call site whose proposal is a
// computed direction rather than a staged key. The clamp is cInsert's own.
fn cInsertKeyed(l:u32,q:vec3u,flags:u32,encodedOwner:u32)->u32{
 return cClaimKey(l,coordKey(min(q,dims(l)-vec3u(1u)),l),flags,encodedOwner);}
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
fn selectedDirectoryLookup(l:u32,q:vec3u)->u32{
 if(topologyDirty(l)){return cDirectoryLookup(l,q);}
 return directoryLookup(l,q,false);
}
fn cLookup(l:u32,q:vec3u)->u32{if(l>=levels()||any(q>=dims(l))){return INVALID;}let wanted=coordKey(q,l);var slot=insertionHash(wanted,l);
 for(var probe=0u;probe<256u;probe+=1u){let old=candidateState[cAt(KEY,l,slot)];if(old==wanted){return slot;}if(old==0u){return INVALID;}slot=(slot+1u)&(levelCapacity(l)-1u);}return INVALID;}
fn section63Direction(k:u32)->vec3i{let d=array<vec3i,18>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[k];}
fn section63WorldDirection(value:vec3i,code:u32)->vec3i{let signs=vec3i(select(1,-1,(code&1u)!=0u),select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));let q=value*signs;let permutation=(code/8u)%6u;
 if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}if(permutation==2u){return q.yxz;}if(permutation==3u){return q.zxy;}if(permutation==4u){return q.yzx;}return q.zyx;}

// Cooperative 256-lane primitives for the ordered candidate phases. Every
// barrier reached from here sits in unconditional control flow whose governing
// bound is a literal or a value laundered through workgroupUniformLoad, so the
// per-lane barrier count is identical on every path through a workgroup and no
// barrier is ever predicated on a raw storage read.
const CHUNK=256u;
var<workgroup> chunkBound:u32;
var<workgroup> chunkCarry:u32;
var<workgroup> chunkKey:array<u32,256>;
var<workgroup> chunkAux:array<u32,256>;
var<workgroup> chunkFlag:array<u32,256>;
var<workgroup> chunkMask:array<u32,8>;
var<workgroup> chunkScan:array<u32,256>;
// Publish one storage-derived word as a workgroup-uniform value. Lane 0 owns
// the write; the uniform load is what lets every loop below carry a barrier in
// provably uniform control flow.
fn uniformWord(lane:u32,value:u32)->u32{
 if(lane==0u){chunkBound=value;}
 workgroupBarrier();
 let published=workgroupUniformLoad(&chunkBound);
 workgroupBarrier();
 return published;
}
// Inclusive prefix sum across the 256 lanes. Integer addition is associative,
// so a block scan publishes exactly the value the serial running counter did.
fn blockInclusiveSum(lane:u32,value:u32)->u32{
 chunkScan[lane]=value;
 workgroupBarrier();
 for(var span=1u;span<CHUNK;span=span<<1u){
  var addend=0u;
  if(lane>=span){addend=chunkScan[lane-span];}
  workgroupBarrier();
  chunkScan[lane]=chunkScan[lane]+addend;
  workgroupBarrier();
 }
 return chunkScan[lane];
}
// Reduce the per-lane replay flags to eight ascending bit words with no atomic:
// each of the first eight lanes owns a disjoint 32-lane span. The storage
// barrier retires the lanes' parallel probe of the hash arena before the single
// ordered owner starts writing that same arena.
fn packChunkFlags(lane:u32){
 storageBarrier();workgroupBarrier();
 if(lane<8u){var bits=0u;
  for(var b=0u;b<32u;b+=1u){if(chunkFlag[lane*32u+b]!=0u){bits|=1u<<b;}}
  chunkMask[lane]=bits;}
 workgroupBarrier();
}

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
   let priorSlots=candidateDispatch[l*DISPATCH_WORDS+10u];
   let priorPages=candidateDispatch[l*DISPATCH_WORDS+11u];
   if(i==0u){levelDelta[deltaAt(l,4u)]=0u;
    candidateDispatch[l*DISPATCH_WORDS]=0u;candidateDispatch[l*DISPATCH_WORDS+8u]=0u;}
   if(i<priorSlots){let slot=cWorkSlot(l,i);
    if(slot>=levelCapacity(l)){candidateReport(l);}else{
     candidateState[cAt(KEY,l,slot)]=0u;candidateState[cAt(FLAGS,l,slot)]=0u;
     candidateState[cAt(OWNER,l,slot)]=0u;}}
   if(i<max(rows(),previousRows())){candidateTopology[rowMapBase()+l*p.capacity.x+i]=INVALID;}
   if(i<priorPages){let key=candidateTopology[pageRecord(l,i)];
    if(key==0u){candidateReport(l);}else{let q=decode(key,l)/vec3u(8u,8u,4u);let d=logicalPageDims(l);
     if(any(q>=d)){candidateReport(l);}else{
      candidateTopology[pageDirectoryBase()+pageLevelOffset(l)+q.x+d.x*(q.y+d.y*q.z)]=INVALID;}}}
  }
  if(dirtyTransfer){
   if(i==0u){candidateDispatch[l*DISPATCH_WORDS+1u]=0u;}
   let priorFine=candidateDispatch[l*DISPATCH_WORDS+10u];
   if(i<priorFine){let fine=cWorkSlot(l,i);if(fine>=levelCapacity(l)){candidateReport(l);}else{
    candidateTopology[fineHeadBase(l)+fine]=INVALID;candidateTopology[fineCountBase(l)+fine]=0u;}}
   let priorCoarse=candidateDispatch[(l+1u)*DISPATCH_WORDS+10u];
   if(i<priorCoarse){let coarse=cWorkSlot(l+1u,i);if(coarse>=levelCapacity(l+1u)){candidateReport(l);}else{
    candidateTopology[parentHeadBase(l)+coarse]=INVALID;
    candidateTopology[parentTailBase(l)+coarse]=INVALID;}}
  }
 }
}
// Phase 2 (one 256-lane workgroup per level). Open-address insertion is the only
// order-defining step left, and levels are disjoint: an insert touches one
// level's key arena, its workset, its count and its own delta word. The row
// sequence inside a level is unchanged, so slots and workset order are
// identical to the serial builder: the lanes only pre-resolve rows whose
// coordinate is already resident, and the single ordered owner replays the rest
// in the identical ascending-row sequence. Retiring a resident row is a
// provable no-op, not a reordering - cClaimKey returns the slot cLookup just
// found and cMergeClass(existing,MG_ONLY) is the identity on ACTIVE, GHOST and
// MG_ONLY alike, so the only remaining effect is the row-map write the lane
// performs itself. Rows at their native level are never retired because the
// claim also takes ownership.
@compute @workgroup_size(256) fn buildCandidateLevelSets(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let l=wg.x;
 let live=uniformWord(lane,select(0u,1u,l<levels()&&topologyDirty(l)));
 if(live==0u){return;}
 let n=uniformWord(lane,rows());
 let rowBase=rowMapBase()+l*p.capacity.x;
 if(lane==0u){beginClaims(l);}
 for(var base=0u;base<n;base+=CHUNK){
  let r=base+lane;
  var key=0u;var encoded=0u;var flag=0u;
  if(r<n){let h=geometry(r);let native=firstTrailingBit(h.y);
   if(l>=native){let q=min(originOf(h)/(1u<<l),dims(l)-vec3u(1u));key=coordKey(q,l);
    if(l==native){encoded=r+1u;flag=1u;}
    else{let resident=cLookup(l,q);
     if(resident==INVALID){flag=1u;}else{candidateTopology[rowBase+r]=resident;}}}}
  chunkKey[lane]=key;chunkAux[lane]=encoded;chunkFlag[lane]=flag;
  packChunkFlags(lane);
  if(lane==0u){
   for(var word=0u;word<8u;word+=1u){var bits=chunkMask[word];
    loop{if(bits==0u){break;}
     let t=word*32u+firstTrailingBit(bits);bits&=bits-1u;
     // chunkKey[t] is coordKey of an already-clamped coordinate, so the former
     // decode(key) -> coordKey(min(q,dims-1)) round trip returned the same key
     // it started from, at the cost of two emulated integer divisions per
     // proposal on the single-lane critical path.
     let owner=chunkAux[t];var slot=INVALID;
     if(owner!=0u){slot=cClaimKey(l,chunkKey[t],ACTIVE,owner);}
     else{slot=cClaimKey(l,chunkKey[t],MG_ONLY,0u);}
     if(slot!=INVALID){candidateTopology[rowBase+base+t]=slot;}}}}
  storageBarrier();workgroupBarrier();
 }
 if(lane==0u){endClaims(l);}
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
// Phase 4 (one 256-lane workgroup per level). Replays the surviving proposals in
// the original (row,channel) order. A repeat proposal at an already-aliased
// coordinate merges GHOST into GHOST with the identical owner and appends
// nothing, which is exactly what the live presence test used to skip. The lanes
// only evaluate the serial sweep's own two continue predicates - an empty
// catalog mask and a row whose native level is not this one - both pure reads,
// so the ordered owner visits exactly the rows the serial sweep visited, in the
// same ascending order, and performs the identical channel sequence per row.
@compute @workgroup_size(256) fn insertCandidateGhosts(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let l=wg.x;
 let live=uniformWord(lane,select(0u,1u,l+1u<levels()&&topologyDirty(l)));
 if(live==0u){return;}
 let n=uniformWord(lane,rows());
 if(lane==0u){beginClaims(l);}
 for(var base=0u;base<n;base+=CHUNK){
  let r=base+lane;var flag=0u;
  if(r<n&&ghostScratch[r*GHOST_STRIDE]!=0u&&firstTrailingBit(geometry(r).y)==l){flag=1u;}
  chunkFlag[lane]=flag;
  packChunkFlags(lane);
  if(lane==0u){
   for(var word=0u;word<8u;word+=1u){var bits=chunkMask[word];
    loop{if(bits==0u){break;}
     let t=word*32u+firstTrailingBit(bits);bits&=bits-1u;
     let row=base+t;let record=row*GHOST_STRIDE;let mask=ghostScratch[record];
     let m=topologyMetrics[row];let q=originOf(geometry(row))/(1u<<l);
     for(var channel=0u;channel<18u;channel+=1u){if((mask&(1u<<channel))==0u){continue;}
      let owner=ghostScratch[record+2u+channel];
      let targetQ=vec3i(q)+section63WorldDirection(section63Direction(channel),m.transformAndFlags&63u);
      _=cInsertKeyed(l,vec3u(targetQ),GHOST,owner);}}}}
  storageBarrier();workgroupBarrier();
 }
 if(lane==0u){endClaims(l);}
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
// Exact fan-out of one fine slot. Physical cells use the eight cell-centred
// trilinear parents; ghost aliases are unit mappings.
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
// stores, no chain updates, no capacity counter - and 256 lanes pre-resolve
// which proposals the owner can skip.
//
// Determinism. The proposal sequence is addressed by its exact serial rank
// i*8+corner, so the owner still visits proposals in ascending
// (fine workset index, corner) order and the slot each new key claims, the
// workset index it appends at, and every level's count are byte-identical to
// the serial builder. Only proposals whose coarse key is already resident are
// retired, and cInsert of a resident key is a provable no-op: it merges
// MG_ONLY into an entry that already carries ACTIVE, GHOST or MG_ONLY (the
// identity of cMergeClass), appends nothing, advances no counter, and its
// discarded return value is the only other effect. A key that becomes resident
// mid-chunk is still replayed and still no-ops. Ghost proposals are never
// retired: their GHOST merge and owner claim are not idempotent.
fn rebuildCandidateTransferFor(l:u32,lane:u32){
 let live=uniformWord(lane,select(0u,1u,transferLive(l)));
 if(live==0u){return;}
 let selected=uniformWord(lane,selectedCount(l));
 let proposalFanOut=8u;
 let total=selected*proposalFanOut;
 let coarseLimit=dims(l+1u)-vec3u(1u);
 if(lane==0u){beginClaims(l+1u);}
 for(var base=0u;base<total;base+=CHUNK){
  let rank=base+lane;
  var key=0u;var encoded=0u;var flag=0u;
  if(rank<total){
   let i=rank/proposalFanOut;let corner=rank%proposalFanOut;let fine=selectedWorkSlot(l,i);
   let flags=selectedState(FLAGS,l,fine);let q=decode(selectedState(KEY,l,fine),l);
   if((flags&GHOST)!=0u){
    if(corner==0u){let encodedOwner=selectedState(OWNER,l,fine);
     if(encodedOwner==0u||encodedOwner>rows()){encoded=INVALID;flag=1u;}
     else if(l+1u!=firstTrailingBit(geometry(encodedOwner-1u).y)){
      key=coordKey(min(q/2u,coarseLimit),l+1u);encoded=encodedOwner;flag=1u;}}}
   else{var parent=cornerTarget(q,corner).coordinate;
    parent=min(parent,coarseLimit);
    key=coordKey(parent,l+1u);
    if(cLookup(l+1u,parent)==INVALID){flag=1u;}}}
  chunkKey[lane]=key;chunkAux[lane]=encoded;chunkFlag[lane]=flag;
  packChunkFlags(lane);
  if(lane==0u){
   for(var word=0u;word<8u;word+=1u){var bits=chunkMask[word];
    loop{if(bits==0u){break;}
     let t=word*32u+firstTrailingBit(bits);bits&=bits-1u;
     let owner=chunkAux[t];
     if(owner==INVALID){candidateReport(l);continue;}
     // Both writers of chunkKey above emit coordKey of a coarseLimit-clamped
     // coordinate, so the former decode(key) -> coordKey(min(q,dims-1)) round
     // trip was the identity, paid in emulated integer division per proposal.
     if(owner!=0u){_=cClaimKey(l+1u,chunkKey[t],GHOST,owner);}
     else{_=cClaimKey(l+1u,chunkKey[t],MG_ONLY,0u);}}}}
  storageBarrier();workgroupBarrier();
 }
 // Level l+1's count must be published before the next iteration reads it as
 // selectedCount(l+1); the barrier pair is what orders that store against the
 // other lanes' reads, and it sits in uniform control flow.
 if(lane==0u){endClaims(l+1u);}
 storageBarrier();workgroupBarrier();
}
@compute @workgroup_size(256) fn buildCandidateLevelDeltas(@builtin(local_invocation_index) lane:u32){
 for(var l=0u;l+1u<levels();l+=1u){rebuildCandidateTransferFor(l,lane);}
}
// Phase 5b (slot parallel). Publish the exact per-fine record fan-out.
@compute @workgroup_size(64) fn countCandidateTransfers(@builtin(global_invocation_id) g:vec3u){
 let i=boundedLinearIndex(g);
 for(var l=0u;l+1u<levels();l+=1u){
  if(!transferLive(l)){continue;}
  // Initialize the exact live identities that the new epoch will link. The
  // prior-identity retirement pass cannot cover cold-start slots or coarse
  // slots newly introduced by this rebuild; leaving their zero-filled tails
  // intact makes record 0 look like a valid predecessor and splices distinct
  // parent chains together.
  if(i<selectedCount(l+1u)){let coarse=selectedWorkSlot(l+1u,i);
   candidateTopology[parentHeadBase(l)+coarse]=INVALID;
   candidateTopology[parentTailBase(l)+coarse]=INVALID;}
  if(i<selectedCount(l)){let fine=selectedWorkSlot(l,i);
   candidateTopology[fineHeadBase(l)+fine]=INVALID;
   candidateTopology[fineCountBase(l)+fine]=transferFanOut(l,fine,
    selectedState(FLAGS,l,fine),decode(selectedState(KEY,l,fine),l));}}
}
// Phase 5c (one 256-lane workgroup per level). Exclusive prefix sum over the
// fan-out in workset order. This reproduces the former append counter exactly:
// the block scan is integer addition only, which is associative, so every
// published record base is bit-identical to the serial running counter. The
// capacity rejection stays fail closed instead of writing past the arena; it is
// also the only path on which the two running totals can diverge, and it
// rejects the whole candidate epoch.
@compute @workgroup_size(256) fn scanCandidateTransfers(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let l=wg.x;
 let live=uniformWord(lane,select(0u,1u,transferLive(l)));
 if(live==0u){return;}
 let selected=uniformWord(lane,selectedCount(l));
 let capacity=transferCapacity(l);
 if(lane==0u){chunkCarry=0u;}
 workgroupBarrier();
 for(var block=0u;block<selected;block+=CHUNK){
  let i=block+lane;var fine=INVALID;var owned=0u;
  if(i<selected){fine=selectedWorkSlot(l,i);owned=candidateTopology[fineCountBase(l)+fine];}
  let inclusive=blockInclusiveSum(lane,owned);
  let base=chunkCarry+inclusive-owned;
  if(i<selected&&owned!=0u){
   if(base+owned>capacity){candidateReport(l);candidateTopology[fineCountBase(l)+fine]=0u;}
   else{candidateTopology[fineHeadBase(l)+fine]=base;}}
  storageBarrier();workgroupBarrier();
  if(lane==0u){chunkCarry=chunkCarry+chunkScan[CHUNK-1u];}
  workgroupBarrier();
 }
 if(lane==0u){candidateDispatch[l*DISPATCH_WORDS+1u]=chunkCarry;}
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
  else{
   if(owned!=8u){candidateReport(l);continue;}
   for(var corner=0u;corner<8u;corner+=1u){let parent=cornerTarget(q,corner);
    let coarse=cResolve(l+1u,parent.coordinate);
    if(coarse==INVALID){candidateReport(l);}
    cAppendTransfer(l,base+corner,fine,coarse,parent.weight);}}
 }
}
// Phase 5e (one 256-lane workgroup per level). The parent chain is the ascending
// record order restricted to one coarse slot, so it is a grouping problem, not
// an accumulation: each record's predecessor is the largest smaller record
// index carrying the same coarse slot. A 256-record block resolves that
// relation exactly - inside the block from the staged coarse ids, and across
// blocks from the running per-coarse tail the serial owner already maintained -
// so head, next and tail are the identical words the serial walk published.
// Writes are partitioned by coarse slot: at most one lane per block opens a
// coarse chain, at most one closes it, and a link target is the immediate
// predecessor of exactly one record, so no two lanes address the same word.
@compute @workgroup_size(256) fn linkCandidateParentChains(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let l=wg.x;
 let live=uniformWord(lane,select(0u,1u,transferLive(l)));
 if(live==0u){return;}
 let n=uniformWord(lane,min(candidateDispatch[l*DISPATCH_WORDS+1u],transferCapacity(l)));
 let limit=levelCapacity(l+1u);
 for(var block=0u;block<n;block+=CHUNK){
  let j=block+lane;var coarse=INVALID;
  if(j<n){coarse=candidateTopology[transferWord(l,j,1u)];
   if(coarse>=limit){candidateReport(l);coarse=INVALID;}}
  chunkKey[lane]=coarse;
  workgroupBarrier();
  // Keep the uniform sweep. Replacing it with per-lane searches that break at
  // the first hit - scan down from lane-1 for the predecessor, up from lane+1
  // for last - is exactly equivalent (verified over 1,024,000 lane cases) and
  // MEASURED SLOWER: 0.903 -> 1.134 ms/advance on the mini lane, while every
  // neighbouring SPGrid kernel stayed flat in the same capture. The sweep reads
  // chunkKey[t] at an index every lane shares, which is a workgroup-memory
  // broadcast; the per-lane searches give each lane its own address, so the
  // reads stop being a broadcast and the early exits buy nothing because a SIMD
  // group still runs to its slowest lane. Do not "optimize" this again without
  // measuring it.
  var predecessor=INVALID;var last=true;
  for(var t=0u;t<CHUNK;t+=1u){
   if(coarse==INVALID||chunkKey[t]!=coarse){continue;}
   if(t<lane){predecessor=block+t;}
   if(t>lane){last=false;}}
  var link=predecessor;
  if(coarse!=INVALID&&predecessor==INVALID){link=candidateTopology[parentTailBase(l)+coarse];}
  storageBarrier();workgroupBarrier();
  if(coarse!=INVALID){
   if(link==INVALID){candidateTopology[parentHeadBase(l)+coarse]=j;}
   else{candidateTopology[transferWord(l,link,3u)]=j;}
   if(last){candidateTopology[parentTailBase(l)+coarse]=j;}}
  storageBarrier();workgroupBarrier();
 }
}
@compute @workgroup_size(1) fn appendCandidateDirectoryIdentities(){
 let capacity=arrayLength(&touchedBrickKeys)/2u;var count=0u;
 for(var l=0u;l<levels();l+=1u){if(!topologyDirty(l)){continue;}let n=cCount(l);
  for(var i=0u;i<n;i+=1u){let slot=cWorkSlot(l,i);if(slot>=levelCapacity(l)){candidateReport(l);continue;}
   let q=decode(candidateState[cAt(KEY,l,slot)],l);for(var ordinal=0u;ordinal<7u;ordinal+=1u){
    var query=vec3i(q);if(ordinal>0u){query+=section63Direction(ordinal-1u);}
    if(any(query<vec3i(0))||any(query>=vec3i(dims(l)))){continue;}let tq=vec3u(query);let b=tq/4u;let d=brickDims(l);
    let identity=brickLevelOffset(l)+b.x+d.x*(b.y+d.y*b.z);if(count<capacity){touchedBrickKeys[count]=identity;}count+=1u;}}}
 touchedBrickHeader[1]=count;if(count>capacity){touchedBrickHeader[3]=INVALID;}
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
// Phase 7 (one 256-lane workgroup per level). The ranked base is a prefix sum
// over the level's bricks in dense order.
@compute @workgroup_size(256) fn rankCandidateBricks(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let l=wg.x;let live=uniformWord(lane,select(0u,1u,l<levels()&&topologyDirty(l)));
 if(live==0u){return;}let generation=levelDelta[deltaAt(l,0u)];
 let first=directoryBase()+16u+brickLevelOffset(l)*4u;let n=uniformWord(lane,brickCount(l));
 if(lane==0u){levelDelta[deltaAt(l,6u)]=generation;chunkCarry=0u;}workgroupBarrier();
 for(var block=0u;block<n;block+=CHUNK){let b=block+lane;var occupancy=0u;var record=0u;
  if(b<n){record=first+b*4u;occupancy=countOneBits(candidateTopology[record+1u])+countOneBits(candidateTopology[record+2u]);}
  let inclusive=blockInclusiveSum(lane,occupancy);
  if(b<n){candidateTopology[record+3u]=chunkCarry+inclusive-occupancy;}
  storageBarrier();workgroupBarrier();if(lane==0u){chunkCarry+=chunkScan[CHUNK-1u];}workgroupBarrier();}
 if(lane==0u){if(chunkCarry!=cCount(l)){candidateReport(l);}
  candidateTopology[directoryBase()+2u+l]=generation;}
}
// Phase 8 (brick parallel). Each brick owns a disjoint ranked range.
@compute @workgroup_size(64) fn scatterCandidateRankedSlots(@builtin(global_invocation_id) g:vec3u){
 let index=boundedLinearIndex(g);if(index>=totalBrickCount()){return;}
 let located=brickOfIndex(index);let l=located.x;if(l>=levels()||!topologyDirty(l)){return;}
 let origin=brickOrigin(l,located.y);let record=directoryBase()+16u+index*4u;
 let low=candidateTopology[record+1u];let high=candidateTopology[record+2u];
 let base=rankedSlotsBase()+levelBase(l)+candidateTopology[record+3u];var rank=0u;
 for(var bit=0u;bit<64u;bit+=1u){
  let word=select(low,high,bit>=32u);if(((word>>(bit&31u))&1u)==0u){continue;}
  candidateTopology[base+rank]=cLookup(l,brickCell(origin,bit));rank+=1u;}
}
fn logicalPageOrigin(l:u32,dense:u32)->vec3u{let d=logicalPageDims(l);
 return vec3u(dense%d.x,(dense/d.x)%d.y,dense/(d.x*d.y))*vec3u(8u,8u,4u);}
fn pageOfIndex(index:u32)->vec2u{var l=levels();var local=0u;
 for(var k=0u;k<levels();k+=1u){let begin=pageLevelOffset(k);
  if(index>=begin&&index<begin+logicalPageCount(k)){l=k;local=index-begin;}}
 return vec2u(l,local);}
// Phase 9 (logical-page parallel). Marks raw occupancy in the page directory.
@compute @workgroup_size(64) fn markCandidatePageOccupancy(@builtin(global_invocation_id) g:vec3u){
 let index=boundedLinearIndex(g);if(index>=p.totals.z){return;}
 let located=pageOfIndex(index);let l=located.x;if(l>=levels()||!topologyDirty(l)){return;}
 let origin=logicalPageOrigin(l,located.y);let extent=dims(l);var occupied=false;
 for(var by=0u;by<2u;by+=1u){for(var bx=0u;bx<2u;bx+=1u){let q=origin+vec3u(4u*bx,4u*by,0u);
  if(any(q>=extent)){continue;}let record=brickRecord(l,q);
  occupied=occupied||candidateTopology[record+1u]!=0u||candidateTopology[record+2u]!=0u;}}
 candidateTopology[pageDirectoryBase()+pageLevelOffset(l)+located.y]=select(0u,1u,occupied);}
// Phase 10 (one 256-lane workgroup per level). Compact occupied logical pages.
@compute @workgroup_size(256) fn compactCandidatePages(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let l=wg.x;let live=uniformWord(lane,select(0u,1u,l<levels()&&topologyDirty(l)));
 if(live==0u){return;}let logicalPages=uniformWord(lane,logicalPageCount(l));
 let directory=pageDirectoryBase()+pageLevelOffset(l);let limit=levelCapacity(l);
 if(lane==0u){chunkCarry=0u;}workgroupBarrier();
 for(var block=0u;block<logicalPages;block+=CHUNK){let dense=block+lane;var occupied=0u;
  if(dense<logicalPages&&candidateTopology[directory+dense]!=0u){occupied=1u;}
  let inclusive=blockInclusiveSum(lane,occupied);let pageTotal=chunkCarry+inclusive-occupied;
  if(dense<logicalPages){if(occupied==0u||pageTotal>=limit){if(occupied!=0u){candidateReport(l);}
    candidateTopology[directory+dense]=INVALID;}else{
    candidateTopology[pageRecord(l,pageTotal)]=coordKey(logicalPageOrigin(l,dense),l);
    candidateTopology[directory+dense]=pageTotal;}}
  storageBarrier();workgroupBarrier();if(lane==0u){chunkCarry+=chunkScan[CHUNK-1u];}workgroupBarrier();}
 if(lane==0u){candidateDispatch[l*DISPATCH_WORDS+8u]=select(chunkCarry,0u,chunkCarry>limit);}}
// Phase 11 (page parallel). The immutable record is a pure read of the compacted directory.
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
@compute @workgroup_size(64) fn markCompactCandidateBrickOccupancy(@builtin(global_invocation_id) g:vec3u){
 let run=boundedLinearIndex(g);let generation=captureGeneration();
 if(touchedBrickControl[0]!=0u||touchedBrickControl[3]!=generation||run>=touchedBrickControl[4]){return;}
 let identity=touchedBrickRuns[2u*run];if(identity>=totalBrickCount()){captureReport(OVERFLOW);return;}
 let located=brickOfIndex(identity);let l=located.x;if(l>=levels()||!topologyDirty(l)){captureReport(OVERFLOW);return;}
 let origin=brickOrigin(l,located.y);let extent=dims(l);var low=0u;var high=0u;
 for(var bit=0u;bit<64u;bit+=1u){let q=brickCell(origin,bit);
  if(any(q>=extent)||cLookup(l,q)==INVALID){continue;}
  if(bit<32u){low|=1u<<bit;}else{high|=1u<<(bit-32u);}}
 let record=directoryBase()+16u+identity*4u;
 if(touchedTripwire()&&(candidateTopology[record]!=generation||candidateTopology[record+1u]!=low
  ||candidateTopology[record+2u]!=high)){candidateReportCode(l,0x2000u);}
 candidateTopology[record]=generation;candidateTopology[record+1u]=low;candidateTopology[record+2u]=high;
}
@compute @workgroup_size(1) fn rankCompactCandidateBricks(){
 let generation=captureGeneration();if(touchedBrickControl[0]!=0u||touchedBrickControl[3]!=generation){captureReport(OVERFLOW);return;}
 let runs=touchedBrickControl[4];let pageCapacity=arrayLength(&touchedPageKeys)/2u;var pageCount=0u;
 touchedPageHeader[0]=${RADIX_SORT_INPUT_VALID}u;touchedPageHeader[1]=0u;touchedPageHeader[2]=generation;touchedPageHeader[3]=0u;
 for(var l=0u;l<levels();l+=1u){if(!topologyDirty(l)){continue;}candidateDispatch[l*DISPATCH_WORDS+8u]=0u;
  let begin=brickLevelOffset(l);let end=begin+brickCount(l);var ranked=0u;
  for(var run=0u;run<runs;run+=1u){let identity=touchedBrickRuns[2u*run];if(identity<begin||identity>=end){continue;}
   let record=directoryBase()+16u+identity*4u;let low=candidateTopology[record+1u];let high=candidateTopology[record+2u];
   if(touchedTripwire()&&candidateTopology[record+3u]!=ranked){candidateReportCode(l,0x4000u);}
   candidateTopology[record+3u]=ranked;let origin=brickOrigin(l,identity-begin);var local=0u;
   for(var bit=0u;bit<64u;bit+=1u){let word=select(low,high,bit>=32u);
    if(((word>>(bit&31u))&1u)==0u){continue;}let slot=cLookup(l,brickCell(origin,bit));
    if(slot==INVALID){candidateReportCode(l,0x8000u);}else{
     candidateTopology[rankedSlotsBase()+levelBase(l)+ranked+local]=slot;}local+=1u;}
   ranked+=local;if(low!=0u||high!=0u){let d=brickDims(l);let brick=identity-begin;let bx=brick%d.x;let by=(brick/d.x)%d.y;
    let bz=brick/(d.x*d.y);let pd=logicalPageDims(l);let page=(bx/2u)+pd.x*((by/2u)+pd.y*bz);
    if(pageCount<pageCapacity){touchedPageKeys[pageCount]=pageLevelOffset(l)+page;}pageCount+=1u;}}
  if(ranked!=cCount(l)){candidateReport(l);}else{levelDelta[deltaAt(l,6u)]=generation;
   candidateTopology[directoryBase()+2u+l]=generation;}}
 touchedBrickHeader[3]=select(0u,4u*runs,touchedTripwire());touchedPageHeader[1]=pageCount;
 if(pageCount>pageCapacity){touchedPageHeader[3]=INVALID;}
}
@compute @workgroup_size(1) fn buildCompactCandidatePages(){
 let generation=captureGeneration();if(touchedPageControl[0]!=0u||touchedPageControl[3]!=generation){captureReport(OVERFLOW);return;}
 let runs=touchedPageControl[4];for(var l=0u;l<levels();l+=1u){if(!topologyDirty(l)){continue;}
  let begin=pageLevelOffset(l);let end=begin+logicalPageCount(l);var physical=0u;
  for(var run=0u;run<runs;run+=1u){let identity=touchedPageRuns[2u*run];if(identity<begin||identity>=end){continue;}
   let dense=identity-begin;let directory=pageDirectoryBase()+identity;let key=coordKey(logicalPageOrigin(l,dense),l);
   if(touchedTripwire()&&(candidateTopology[directory]!=physical
    ||candidateTopology[pageRecord(l,physical)]!=key)){candidateReportCode(l,0x10000u);}
   candidateTopology[directory]=physical;candidateTopology[pageRecord(l,physical)]=key;
   touchedPageRuns[2u*run+1u]=physical;physical+=1u;}
  if(physical>levelCapacity(l)){candidateReport(l);candidateDispatch[l*DISPATCH_WORDS+8u]=0u;}
  else{candidateDispatch[l*DISPATCH_WORDS+8u]=physical;}}
 touchedPageHeader[3]=select(0u,29u*runs,touchedTripwire());
}
@compute @workgroup_size(64) fn linkCompactCandidatePageNeighbours(@builtin(global_invocation_id) g:vec3u){
 let run=boundedLinearIndex(g);let generation=captureGeneration();
 if(touchedPageControl[0]!=0u||touchedPageControl[3]!=generation||run>=touchedPageControl[4]){return;}
 let identity=touchedPageRuns[2u*run];let located=pageOfIndex(identity);let l=located.x;
 if(l>=levels()||!topologyDirty(l)){captureReport(OVERFLOW);return;}let page=touchedPageRuns[2u*run+1u];
 let pageDims=logicalPageDims(l);let record=pageRecord(l,page);let origin=decode(candidateTopology[record],l)/vec3u(8u,8u,4u);
 let directory=pageDirectoryBase()+pageLevelOffset(l);for(var ordinal=0u;ordinal<27u;ordinal+=1u){
  let dx=i32(ordinal%3u)-1;let yz=ordinal/3u;let neighbour=vec3i(origin)+vec3i(dx,i32(yz%3u)-1,i32(yz/3u)-1);var physical=INVALID;
  if(all(neighbour>=vec3i(0))&&all(neighbour<vec3i(pageDims))){let q=vec3u(neighbour);
   physical=candidateTopology[directory+q.x+pageDims.x*(q.y+pageDims.y*q.z)];}
  if(touchedTripwire()&&candidateTopology[record+1u+ordinal]!=physical){candidateReportCode(l,0x20000u);}
  candidateTopology[record+1u+ordinal]=physical;}
}
// Phase 12 (slot parallel). Every workset entry is a distinct slot, so the
// former reset sweep and the rediscretization fuse into one owner per slot and
// resolve neighbours through the published mask/rank directory, never a probe.
@compute @workgroup_size(64) fn buildCandidateStencils(@builtin(global_invocation_id) g:vec3u){
 let i=boundedLinearIndex(g);
 for(var l=0u;l<levels();l+=1u){
  if(!stencilDirty(l)||i>=selectedCount(l)){continue;}
  let s=selectedWorkSlot(l,i);
  for(var c=DIAG;c<=YZMM;c+=1u){candidateState[cAt(c,l,s)]=0u;}
  // A channel keeps INVALID unless this owner publishes both its coefficient
  // and the slot that coefficient was accumulated against, in the same
  // iteration. The pair is therefore always same-epoch and same-direction, and
  // a zero coefficient is always paired with an unresolvable neighbour.
  for(var k=0u;k<18u;k+=1u){candidateTopology[neighbourAt(k,l,s)]=INVALID;}
  let coefficient=f32(1u<<l)*p.reserved.y;
  let q=decode(selectedState(KEY,l,s),l);let flags=selectedState(FLAGS,l,s);var diagonal=0.0;
  let encodedOwner=selectedState(OWNER,l,s);var acceptedFine=(flags&ACTIVE)!=0u
   &&encodedOwner>0u&&encodedOwner<=p.capacity.x;
  var acceptedBase=0u;
  if(acceptedFine){let row=encodedOwner-1u;acceptedBase=(acceptedBank()*p.capacity.x+row)*19u;
   if(acceptedBase+19u>arrayLength(&acceptedCoefficients)){
    candidateReportCode(l,0x100u);acceptedFine=false;
   }else{let acceptedDiagonal=acceptedCoefficients[acceptedBase];var acceptedTerms:array<f32,18>;
    for(var channel=0u;channel<18u;channel+=1u){let c=acceptedCoefficients[acceptedBase+1u+channel];
     if(!finite(c)||c<0.0){candidateReportCode(l,0x200u);acceptedFine=false;}acceptedTerms[channel]=max(0.0,c);}
    let acceptedOff=canonical18Sum(acceptedTerms);
    if(!finite(acceptedDiagonal)||!(acceptedDiagonal>0.0)||!finite(acceptedOff)){
     candidateReportCode(l,0x400u);acceptedFine=false;
    }else if(acceptedFine){
     // Aanjaneya et al. Section 4.3 uses a first-order M1 with the same
     // boundary conditions as L2. Keep the symmetric first-order grid graph,
     // and import only L2's non-negative Dirichlet/free-surface reaction
     // (diag - sum(offdiag)). Importing directional L2 contacts onto adjacent
     // unit SPGrid slots is not a shared discretization on adaptive rows and
     // makes M1 nonsymmetric.
     diagonal=max(0.0,acceptedDiagonal-acceptedOff);
    }}}
  for(var k=0u;k<6u;k+=1u){let c=coefficient;
   let targetQ=vec3i(q)+section63Direction(k);if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){continue;}
   let other=selectedDirectoryLookup(l,vec3u(targetQ));if(other==INVALID){if((flags&GHOST)==0u){diagonal+=c;}continue;}
   diagonal+=c;candidateTopology[neighbourAt(k,l,s)]=other;cStoref(XP+k,l,s,c);}
  if((flags&MG_ONLY)!=0u){diagonal+=bitcast<f32>(p.totals.w)*coefficient;}
  // The one-cell terminal grid has no in-domain neighbour, but M1 represents
  // the same free-surface pressure problem as L2 rather than a pure Neumann
  // null space. Keep its exact solve on the level's physical first-order
  // scale; a dimensionless unit fallback increasingly over-corrected the
  // constant mode as the hierarchy deepened.
  if(!(diagonal>1e-20)||!finite(diagonal)){diagonal=coefficient;}cStoref(DIAG,l,s,diagonal);
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
 if(live){let n=selectedCount(l);
  for(var i=lane;i<n;i+=64u){let s=selectedWorkSlot(l,i);let d=cLoadf(DIAG,l,s);var off=0.0;
   for(var k=0u;k<18u;k+=1u){let c=cLoadf(XP+k,l,s);
    if(!finite(c)||c<0.0){candidateReportCode(l,0x800u);}off+=abs(c);}
   if(!finite(d)||!(d>0.0)||!finite(off)||off>d*(1.0+1e-4)){candidateReportCode(l,0x1000u);}
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
fn commitCandidateBrickForKey(l:u32,key:u32){
 if(key==0u){return;}let q=decode(key,l);let source=brickRecord(l,q);
 for(var w=0u;w<4u;w+=1u){topology[source+w]=candidateTopology[source+w];}
}
@compute @workgroup_size(1) fn commitCandidateTouchedBricks(){
 let generation=captureGeneration();if(captureFailed()||touchedBrickControl[0]!=0u
  ||touchedBrickControl[3]!=generation){return;}let runs=touchedBrickControl[4];
 for(var run=0u;run<runs;run+=1u){let identity=touchedBrickRuns[2u*run];
  if(identity>=totalBrickCount()){return;}let located=brickOfIndex(identity);let l=located.x;
  if(l>=levels()||!topologyDirty(l)||levelDelta[deltaAt(l,4u)]!=0u){continue;}
  let record=directoryBase()+16u+identity*4u;for(var word=0u;word<4u;word+=1u){
   topology[record+word]=candidateTopology[record+word];}}
}
fn commitCandidateSlot(l:u32,slot:u32,whole:bool,stencil:bool){
 if(slot>=levelCapacity(l)){captureReport(OVERFLOW);return;}
 if(stencil){for(var c=DIAG;c<=YZMM;c+=1u){state[at(c,l,slot)]=candidateState[cAt(c,l,slot)];}
  for(var k=0u;k<18u;k+=1u){topology[neighbourAt(k,l,slot)]=candidateTopology[neighbourAt(k,l,slot)];}}
 if(whole){for(var c=0u;c<STATE_CHANNELS;c+=1u){state[at(c,l,slot)]=candidateState[cAt(c,l,slot)];}}
}
fn commitCandidateLevelAt(l:u32){
 let topologyChanged=topologyDirty(l);
 let transferChanged=topologyChanged||(l+1u<levels()&&topologyDirty(l+1u));
 if(!topologyChanged&&!stencilDirty(l)&&!transferChanged){return;}
 if(captureFailed()||levelDelta[deltaAt(l,4u)]!=0u){return;}
 let oldSlots=count(l);let newSlots=selectedCount(l);let stencil=stencilDirty(l);
 if(topologyChanged){
  // Empty bricks are part of the accepted directory contract too: their
  // current-generation stamp proves that a failed occupancy lookup is an
  // authoritative miss rather than stale topology. Copying only bricks that
  // contain live slots leaves queried neighbouring empties on the prior
  // generation and makes directoryLookup fail closed. The touched-directory
  // cutover may narrow this loop only after it publishes both occupied and
  // queried-empty brick identities.
  if(!touchedDirectory()){for(var brick=0u;brick<brickCount(l);brick+=1u){let record=directoryBase()+16u+
    (brickLevelOffset(l)+brick)*4u;
   for(var word=0u;word<4u;word+=1u){topology[record+word]=candidateTopology[record+word];}}}
  // Retired accepted identities must publish the candidate's cleared words;
  // new identities publish their claimed words. Their union is the complete
  // sparse replacement for the old level-capacity channel sweep.
  for(var i=0u;i<oldSlots;i+=1u){let slot=workSlot(l,i);commitCandidateSlot(l,slot,true,true);}
  for(var i=0u;i<newSlots;i+=1u){let slot=cWorkSlot(l,i);commitCandidateSlot(l,slot,true,true);
   topology[workBase()+levelBase(l)+i]=slot;
   topology[rankedSlotsBase()+levelBase(l)+i]=candidateTopology[rankedSlotsBase()+levelBase(l)+i];}
  for(var r=0u;r<max(rows(),previousRows());r+=1u){topology[rowMapBase()+l*p.capacity.x+r]=candidateTopology[rowMapBase()+l*p.capacity.x+r];}
  let oldPages=pageCount(l);for(var i=0u;i<oldPages;i+=1u){let key=pageKey(l,i);if(key==0u){captureReport(OVERFLOW);continue;}
   let q=decode(key,l)/vec3u(8u,8u,4u);let d=logicalPageDims(l);
   if(any(q>=d)){captureReport(OVERFLOW);}else{let at=pageDirectoryBase()+pageLevelOffset(l)+q.x+d.x*(q.y+d.y*q.z);
    topology[at]=candidateTopology[at];}}
  let newPages=candidateDispatch[l*DISPATCH_WORDS+8u];for(var i=0u;i<newPages;i+=1u){let page=pageRecord(l,i);
   for(var word=0u;word<28u;word+=1u){topology[page+word]=candidateTopology[page+word];}
   let q=decode(candidateTopology[page],l)/vec3u(8u,8u,4u);let d=logicalPageDims(l);
   if(any(q>=d)){captureReport(OVERFLOW);}else{let at=pageDirectoryBase()+pageLevelOffset(l)+q.x+d.x*(q.y+d.y*q.z);
    topology[at]=candidateTopology[at];}}
 }
 if(stencil&&!topologyChanged){for(var i=0u;i<newSlots;i+=1u){commitCandidateSlot(l,selectedWorkSlot(l,i),false,true);}}
 if(transferChanged&&l+1u<levels()){
  for(var i=0u;i<count(l);i+=1u){let fine=workSlot(l,i);topology[fineHeadBase(l)+fine]=candidateTopology[fineHeadBase(l)+fine];
   topology[fineCountBase(l)+fine]=candidateTopology[fineCountBase(l)+fine];}
  for(var i=0u;i<selectedCount(l);i+=1u){let fine=selectedWorkSlot(l,i);topology[fineHeadBase(l)+fine]=candidateTopology[fineHeadBase(l)+fine];
   topology[fineCountBase(l)+fine]=candidateTopology[fineCountBase(l)+fine];}
  for(var i=0u;i<count(l+1u);i+=1u){let coarse=workSlot(l+1u,i);topology[parentHeadBase(l)+coarse]=candidateTopology[parentHeadBase(l)+coarse];
   topology[parentTailBase(l)+coarse]=candidateTopology[parentTailBase(l)+coarse];}
  for(var i=0u;i<selectedCount(l+1u);i+=1u){let coarse=selectedWorkSlot(l+1u,i);topology[parentHeadBase(l)+coarse]=candidateTopology[parentHeadBase(l)+coarse];
   topology[parentTailBase(l)+coarse]=candidateTopology[parentTailBase(l)+coarse];}
  let records=max(transferCount(l),candidateDispatch[l*DISPATCH_WORDS+1u]);for(var i=0u;i<records;i+=1u){
   if(i>=transferCapacity(l)){captureReport(OVERFLOW);break;}for(var w=0u;w<4u;w+=1u){topology[transferWord(l,i,w)]=candidateTopology[transferWord(l,i,w)];}}
 }
 if(stencil){state[at(SPECTRAL,l,0u)]=levelDelta[deltaAt(l,7u)];}
  if(topologyChanged){dispatchMeta[l*DISPATCH_WORDS]=candidateDispatch[l*DISPATCH_WORDS];
   dispatchMeta[l*DISPATCH_WORDS+8u]=candidateDispatch[l*DISPATCH_WORDS+8u];
   topology[directoryBase()+2u+l]=candidateTopology[directoryBase()+2u+l];
   levelDelta[deltaAt(l,5u)]=captureGeneration();}
  if(transferChanged&&l+1u<levels()){dispatchMeta[l*DISPATCH_WORDS+1u]=candidateDispatch[l*DISPATCH_WORDS+1u];}
  let blocks=(selectedCount(l)+63u)/64u;dispatchMeta[l*DISPATCH_WORDS+2u]=min(65535u,blocks);
  dispatchMeta[l*DISPATCH_WORDS+3u]=select(1u,(blocks+65534u)/65535u,blocks>0u);dispatchMeta[l*DISPATCH_WORDS+4u]=1u;
  // The parent record is now one workgroup per coarse slot, not one per
  // sixty-four: the whole workgroup cooperates on that slot's transfer chain.
  var parentSlots=0u;if(l+1u<levels()){parentSlots=selectedCount(l+1u);}
  dispatchMeta[l*DISPATCH_WORDS+5u]=min(65535u,parentSlots);
  dispatchMeta[l*DISPATCH_WORDS+6u]=select(1u,(parentSlots+65534u)/65535u,parentSlots>0u);
  dispatchMeta[l*DISPATCH_WORDS+7u]=1u;
  let pages=dispatchMeta[l*DISPATCH_WORDS+8u];dispatchMeta[l*DISPATCH_WORDS+9u]=pages;
  dispatchMeta[l*DISPATCH_WORDS+10u]=1u;dispatchMeta[l*DISPATCH_WORDS+11u]=1u;
}
@compute @workgroup_size(1) fn commitCandidateLevels(@builtin(global_invocation_id) g:vec3u){
 let l=g.x;if(l<levels()){commitCandidateLevelAt(l);}
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
fn seedNativeRhs(r:u32){let v=inputRhs[r];let native=firstTrailingBit(sourceRowGeometry(r).y);
 if(!finite(v)){reportAt(NONFINITE,73u,r);}else{storef(RHS,native,rowMap(native,r),v);}}
@compute @workgroup_size(64) fn seedRhs(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){seedNativeRhs(r);}}
// Factor-1 uses this exact union of the two accepted-row initializers. The
// intervening sparse zero dispatches touch only state, so correction can be
// cleared here immediately before the same per-row RHS seed without changing
// any read, write, stop-gate, row-domain, or floating-point operation.
@compute @workgroup_size(64) fn seedRhsAndClearCorrection(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){outputCorrection[r]=0.0;seedNativeRhs(r);}}
fn stencilDirection(k:u32)->vec3i{let d=array<vec3i,18>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[k];}
fn canonical18Sum(values:array<f32,18>)->f32{var sorted=values;for(var i=1u;i<18u;i+=1u){let value=sorted[i];var j=i;loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}var sum=0.0;for(var i=0u;i<18u;i+=1u){sum+=sorted[i];}return sum;}
fn canonical8Sum(values:array<f32,8>)->f32{var sorted=values;for(var i=1u;i<8u;i+=1u){let value=sorted[i];var j=i;loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}var sum=0.0;for(var i=0u;i<8u;i+=1u){sum+=sorted[i];}return sum;}
// Section 4.6. The eighteen spokes consume the column indices setup published
// beside the coefficients instead of re-deriving each one through the global
// brick/rank directory. Every retained term is bit-identical: a spoke is taken
// exactly when its coefficient is non-zero, and buildCandidateStencils writes a
// non-zero coefficient only in the iteration where cDirectoryLookup resolved
// that same direction to a slot, publishing that slot into the same k channel.
// An unresolved or out-of-domain direction leaves the coefficient at zero and
// the column at INVALID, so the c==0 early-out still skips exactly the spokes
// the recomputed find() used to skip - and it now skips them without paying for
// the lookup first.
fn applied(slot:u32,source:u32)->f32{let l=level();
 // Section 4.5, independent of the column-index change: every address term the
 // eighteen iterations used to re-derive per spoke is a dispatch invariant.
 // levelCapacity and levelBase are memoized-table reads and neighbourBase is a
 // whole arena-prefix chain. Hoisting is integer identity, because at(c,l,s) is
 // c*totalLevelSlots()+levelBase(l)+s and neighbourAt(k,l,s) is
 // neighbourBase()+k*totalLevelSlots()+levelBase(l)+s, so base+k*span addresses
 // exactly the word the helper addressed. Reverting these four lines to the
 // loadf/neighbourAt/levelCapacity calls restores the unhoisted form.
 let span=totalLevelSlots();let capacity=levelCapacity(l);let base=levelBase(l);
 let coefficientBase=XP*span+base+slot;let columnBase=neighbourBase()+base+slot;
 let sourceBase=source*span+base;
 var terms:array<f32,18>;
 for(var k=0u;k<18u;k+=1u){let c=bitcast<f32>(state[coefficientBase+k*span]);if(c==0.0){continue;}
  let other=topology[columnBase+k*span];if(other>=capacity){reportAt(OVERFLOW,75u,slot);continue;}
  terms[k]=c*bitcast<f32>(state[sourceBase+other]);}
 return loadf(DIAG,l,slot)*bitcast<f32>(state[sourceBase+slot])-canonical18Sum(terms);}
fn smoothable(l:u32,s:u32)->bool{return(state[at(FLAGS,l,s)]&GHOST)==0u;}
fn chebyshevWeight(l:u32,phase:u32,degree:u32)->f32{
 let upper=loadf(SPECTRAL,l,0u);let lower=upper/30.0;
 if(!(lower>0.0)||!(upper>lower)||!finite(upper)){reportAt(NONPOSITIVE,76u,l);return 0.0;}
 let centre=0.5*(upper+lower);let radius=0.5*(upper-lower);
 return 1.0/(centre-radius*cos(3.141592653589793*(2.0*f32(phase)+1.0)/(2.0*f32(degree))));
}
fn relaxChebyshev(slot:u32,src:u32,dst:u32,phase:u32){let l=level();let source=loadf(src,l,slot);
 if(!smoothable(l,slot)){storef(dst,l,slot,source);return;}let d=loadf(DIAG,l,slot);
 if(!(d>0.0)){reportAt(NONPOSITIVE,79u,slot);storef(dst,l,slot,source);return;}
 let next=source+chebyshevWeight(l,phase,p.solve.x)*(loadf(RHS,l,slot)-applied(slot,src))/d;
 if(!finite(next)){reportAt(NONFINITE,80u,slot);storef(dst,l,slot,source);}else{storef(dst,l,slot,next);}}
fn smoothGlobal(g:vec3u,src:u32,dst:u32,phase:u32){let i=slotIndex(g);let l=level();
 if(i<count(l)&&!stopped()){relaxChebyshev(workSlot(l,i),src,dst,phase);}}
@compute @workgroup_size(64) fn smoothChebyshevAtoB0(@builtin(global_invocation_id) g:vec3u){smoothGlobal(g,A,B,0u);}
@compute @workgroup_size(64) fn smoothChebyshevBtoA0(@builtin(global_invocation_id) g:vec3u){smoothGlobal(g,B,A,0u);}
@compute @workgroup_size(64) fn smoothChebyshevAtoB1(@builtin(global_invocation_id) g:vec3u){smoothGlobal(g,A,B,1u);}
@compute @workgroup_size(64) fn smoothChebyshevBtoA1(@builtin(global_invocation_id) g:vec3u){smoothGlobal(g,B,A,1u);}
@compute @workgroup_size(64) fn smoothChebyshevAtoB2(@builtin(global_invocation_id) g:vec3u){smoothGlobal(g,A,B,2u);}
@compute @workgroup_size(64) fn smoothChebyshevBtoA2(@builtin(global_invocation_id) g:vec3u){smoothGlobal(g,B,A,2u);}
@compute @workgroup_size(64) fn smoothChebyshevAtoB3(@builtin(global_invocation_id) g:vec3u){smoothGlobal(g,A,B,3u);}
@compute @workgroup_size(64) fn smoothChebyshevBtoA3(@builtin(global_invocation_id) g:vec3u){smoothGlobal(g,B,A,3u);}
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
// Record-parallel staging for E^T. The parent chain is the ascending record
// order restricted to one coarse slot, so a chunk of it is a contiguous
// ascending run that can be evaluated in any order and folded back in index
// order. Sixty-four staged indices, their eighteen-channel stencil residual and
// their immutable weight are the only workgroup state; the weight and residual
// stay separate so the fold keeps the source-level multiply-add of the
// single-lane walk.
const RESTRICT_LANES=64u;
var<workgroup> restrictRecord:array<u32,64>;
var<workgroup> restrictWeight:array<f32,64>;
var<workgroup> restrictResidual:array<f32,64>;
var<workgroup> restrictStaged:u32;
var<workgroup> restrictNext:u32;
var<workgroup> restrictFailed:u32;
var<workgroup> restrictSum:f32;
// Section 4.2 GhostValueAccumulate is E^T: one coarse owner still owns each sum
// and no destination synchronization exists, but the owner is now a whole
// workgroup instead of a lane. Lane 0 walks the immutable chain and stages up to
// sixty-four record indices - applying exactly the bounds and parent-identity
// checks the serial walk applied, in the same order, so the visited record set
// and every reportAt is unchanged - then all sixty-four lanes evaluate one
// record each, and lane 0 folds the staged terms in ascending record index.
// The per-record expression, the operand values and the accumulation order are
// the serial walk's, so the published sum is bit-identical; only the
// eighteen-channel gather stops being serialized behind the pointer chase.
@compute @workgroup_size(64) fn restrictAndGhostAccumulate(@builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32){
 let l=level();let i=wg.x+wg.y*65535u;
 let live=uniformWord(lane,select(0u,1u,l+1u<levels()&&i<count(l+1u)&&!stopped()));
 if(live==0u){return;}
 let coarse=workSlot(l+1u,i);
 let limit=uniformWord(lane,transferCount(l));
 if(lane==0u){restrictNext=topology[parentHeadBase(l)+coarse];
  restrictSum=0.0;restrictFailed=0u;restrictStaged=0u;}
 workgroupBarrier();
 for(var visited=0u;visited<limit;visited+=RESTRICT_LANES){
  if(lane==0u){var record=restrictNext;var staged=0u;let span=min(RESTRICT_LANES,limit-visited);
   loop{if(staged>=span||record==INVALID){break;}
    if(record>=limit){reportAt(OVERFLOW,85u,coarse);restrictFailed=1u;record=INVALID;break;}
    if(topology[transferWord(l,record,1u)]!=coarse){reportAt(OVERFLOW,86u,coarse);restrictFailed=1u;record=INVALID;break;}
    restrictRecord[staged]=record;staged+=1u;record=topology[transferWord(l,record,3u)];}
   restrictStaged=staged;restrictNext=record;}
  workgroupBarrier();
  let staged=workgroupUniformLoad(&restrictStaged);
  if(staged==0u){break;}
  var weight=0.0;var residual=0.0;
  if(lane<staged){let record=restrictRecord[lane];let fine=topology[transferWord(l,record,0u)];
   let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;let product=applied(fine,A);
   residual=select(-product,loadf(RHS,l,fine)-product,!ghost);
   weight=p.reserved.x*bitcast<f32>(topology[transferWord(l,record,2u)]);}
  restrictWeight[lane]=weight;restrictResidual[lane]=residual;
  workgroupBarrier();
  if(lane==0u){var folded=restrictSum;
   for(var t=0u;t<staged;t+=1u){restrictResidual[t]=restrictWeight[t]*restrictResidual[t];}
   for(var t=1u;t<staged;t+=1u){let value=restrictResidual[t];var j=t;loop{if(j==0u||restrictResidual[j-1u]<=value){break;}restrictResidual[j]=restrictResidual[j-1u];j-=1u;}restrictResidual[j]=value;}
   for(var t=0u;t<staged;t+=1u){folded+=restrictResidual[t];}
   restrictSum=folded;}
  workgroupBarrier();
 }
 if(lane==0u&&restrictFailed==0u){let sum=restrictSum;
  if(restrictNext!=INVALID||!finite(sum)){reportAt(OVERFLOW,87u,coarse);}
  // Accepted pressure rows are injected at their native octree levels before
  // the descent. GhostValueAccumulate contributes the finer residual to that
  // existing native RHS; replacing it would delete the direct coarse pressure
  // variable from M1 while prolongation still returned a correction to it,
  // breaking the E^T/E symmetry required by Aanjaneya et al. §4.3.
  else{storef(RHS,l+1u,coarse,loadf(RHS,l+1u,coarse)+sum);}}}

// The coarse levels contain too little work to amortize a dispatch per
// Chebyshev phase. Unlike the removed page-frozen smoother, this tail has one
// workgroup for the complete level and a storage/workgroup barrier after every
// phase. It therefore evaluates the same global Jacobi polynomial and the same
// E/E^T record chains as the dispatched oracle while level zero stays wide.
fn coarseApplied(l:u32,slot:u32,source:u32)->f32{
 let span=totalLevelSlots();let capacity=levelCapacity(l);let base=levelBase(l);
 let coefficientBase=XP*span+base+slot;let columnBase=neighbourBase()+base+slot;
 let sourceBase=source*span+base;
 var terms:array<f32,18>;
 for(var k=0u;k<18u;k+=1u){
  let c=bitcast<f32>(state[coefficientBase+k*span]);if(c==0.0){continue;}
  let other=topology[columnBase+k*span];
  if(other>=capacity){reportAt(OVERFLOW,75u,slot);continue;}
  terms[k]=c*bitcast<f32>(state[sourceBase+other]);}
 return loadf(DIAG,l,slot)*bitcast<f32>(state[sourceBase+slot])-canonical18Sum(terms);
}
fn coarseSmoothPhase(l:u32,src:u32,dst:u32,phase:u32,lane:u32){
 let n=count(l);
 for(var i=lane;i<n;i+=64u){
  if(stopped()){continue;}
  let slot=workSlot(l,i);let source=loadf(src,l,slot);
  if(!smoothable(l,slot)){storef(dst,l,slot,source);continue;}
  let d=loadf(DIAG,l,slot);
  if(!(d>0.0)){reportAt(NONPOSITIVE,79u,slot);storef(dst,l,slot,source);continue;}
  let next=source+chebyshevWeight(l,phase,p.solve.x)
    *(loadf(RHS,l,slot)-coarseApplied(l,slot,src))/d;
  if(!finite(next)){reportAt(NONFINITE,80u,slot);storef(dst,l,slot,source);}
  else{storef(dst,l,slot,next);}
 }
}
fn coarseSmooth(l:u32,reverse:bool,lane:u32){
 for(var step=0u;step<p.solve.x;step+=1u){
  let phase=select(step,p.solve.x-1u-step,reverse);
  if((step&1u)==0u){coarseSmoothPhase(l,A,B,phase,lane);}
  else{coarseSmoothPhase(l,B,A,phase,lane);}
  storageBarrier();workgroupBarrier();
  }
}
fn coarseRestrict(l:u32,lane:u32){
 if(l+1u>=levels()){return;}
 let n=uniformWord(lane,count(l+1u));let limit=uniformWord(lane,transferCount(l));
 // Tail levels have at most eight coarse owners. Process owners in order but
 // use the whole workgroup for each immutable parent-chain block, matching the
 // wide restriction kernel's evaluation and ordered lane-zero fold.
 for(var i=0u;i<n;i+=1u){
  let coarse=workSlot(l+1u,i);
  if(lane==0u){
   restrictNext=select(INVALID,topology[parentHeadBase(l)+coarse],!stopped());
   restrictSum=0.0;restrictFailed=0u;restrictStaged=0u;
  }
  workgroupBarrier();
  for(var visited=0u;visited<limit;visited+=RESTRICT_LANES){
   if(lane==0u){var record=restrictNext;var staged=0u;
    let span=min(RESTRICT_LANES,limit-visited);
    loop{if(staged>=span||record==INVALID){break;}
     if(record>=limit){reportAt(OVERFLOW,85u,coarse);restrictFailed=1u;record=INVALID;break;}
     if(topology[transferWord(l,record,1u)]!=coarse){
      reportAt(OVERFLOW,86u,coarse);restrictFailed=1u;record=INVALID;break;}
     restrictRecord[staged]=record;staged+=1u;
     record=topology[transferWord(l,record,3u)];}
    restrictStaged=staged;restrictNext=record;
   }
   workgroupBarrier();
   let staged=workgroupUniformLoad(&restrictStaged);
   if(staged==0u){break;}
   var weight=0.0;var residual=0.0;
   if(lane<staged){
    let record=restrictRecord[lane];let fine=topology[transferWord(l,record,0u)];
    let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;
    let product=coarseApplied(l,fine,A);
    residual=select(-product,loadf(RHS,l,fine)-product,!ghost);
    weight=p.reserved.x*bitcast<f32>(topology[transferWord(l,record,2u)]);
   }
   restrictWeight[lane]=weight;restrictResidual[lane]=residual;
   workgroupBarrier();
   if(lane==0u){var folded=restrictSum;
    for(var t=0u;t<staged;t+=1u){restrictResidual[t]=restrictWeight[t]*restrictResidual[t];}
    for(var t=1u;t<staged;t+=1u){let value=restrictResidual[t];var j=t;loop{if(j==0u||restrictResidual[j-1u]<=value){break;}restrictResidual[j]=restrictResidual[j-1u];j-=1u;}restrictResidual[j]=value;}
    for(var t=0u;t<staged;t+=1u){folded+=restrictResidual[t];}
    restrictSum=folded;
   }
   workgroupBarrier();
  }
  if(lane==0u&&restrictFailed==0u){
   let sum=restrictSum;
   if(restrictNext!=INVALID||!finite(sum)){reportAt(OVERFLOW,87u,coarse);}
   else{storef(RHS,l+1u,coarse,loadf(RHS,l+1u,coarse)+sum);}
  }
  storageBarrier();workgroupBarrier();
 }
}
fn coarseProlong(l:u32,lane:u32){
 let n=count(l);
 for(var i=lane;i<n;i+=64u){
  if(stopped()){continue;}
  let fine=workSlot(l,i);let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;
  var value=select(loadf(A,l,fine),0.0,ghost);
  {
   let targetCount=topology[fineCountBase(l)+fine];var failed=false;var values:array<f32,8>;
   for(var corner=0u;corner<targetCount;corner+=1u){
    let transfer=correctionTransfer(l,fine,corner);
    if(transfer.coarse==INVALID){failed=true;break;}
    values[corner]=p.reserved.x*transfer.weight*loadf(A,l+1u,transfer.coarse);
   }
   if(failed){continue;}
   value+=canonical8Sum(values);
  }
  if(!finite(value)){reportAt(NONFINITE,91u,fine);}else{storef(A,l,fine,value);}
 }
}
@compute @workgroup_size(64) fn coarseVcycleTail(
 @builtin(local_invocation_index) lane:u32){
 let first=level();
 // The small tail descends with globally synchronized smoothing.
 for(var l=first;l+1u<levels();l+=1u){
  coarseSmooth(l,false,lane);
  coarseRestrict(l,lane);
  storageBarrier();workgroupBarrier();
 }
 // The hierarchy invariant is an exact one-cell bottom.
 if(lane==0u&&!stopped()){
  let l=levels()-1u;
  if(count(l)!=1u){reportAt(NONPOSITIVE,88u,l);}
  else{let s=workSlot(l,0u);let d=loadf(DIAG,l,s);
   if(!(d>0.0)){reportAt(NONPOSITIVE,89u,s);}
   else{let x=loadf(RHS,l,s)/d;
    if(!finite(x)){reportAt(NONFINITE,90u,s);}else{storef(A,l,s,x);}}}
 }
 storageBarrier();workgroupBarrier();
 // Ascend to the first fused level. The signed loop avoids unsigned wrap.
 for(var signedLevel=i32(levels())-2;signedLevel>=i32(first);signedLevel-=1){
  let l=u32(signedLevel);
  coarseProlong(l,lane);
  storageBarrier();workgroupBarrier();
  coarseSmooth(l,true,lane);
 }
}
@compute @workgroup_size(64) fn exactBottom(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>0u||stopped()){return;}if(count(l)!=1u){reportAt(NONPOSITIVE,88u,l);return;}
 let s=workSlot(l,0u);let d=loadf(DIAG,l,s);if(!(d>0.0)){reportAt(NONPOSITIVE,89u,s);return;}let x=loadf(RHS,l,s)/d;if(!finite(x)){reportAt(NONFINITE,90u,s);}else{storef(A,l,s,x);}}
// One fine invocation owns the complete interpolation sum, deleting all
// prolongation atomics. GhostValuePropagate is the unit-copy branch of the
// same E mapping rather than a second dispatch.
@compute @workgroup_size(64) fn prolongAndGhostPropagate(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||stopped()){return;}
 let fine=workSlot(l,i);let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;var value=select(loadf(A,l,fine),0.0,ghost);
 {let targetCount=topology[fineCountBase(l)+fine];var values:array<f32,8>;
  for(var corner=0u;corner<targetCount;corner+=1u){let transfer=correctionTransfer(l,fine,corner);if(transfer.coarse==INVALID){return;}
   values[corner]=p.reserved.x*transfer.weight*loadf(A,l+1u,transfer.coarse);}
  value+=canonical8Sum(values);}
 if(!finite(value)){reportAt(NONFINITE,91u,fine);}else{storef(A,l,fine,value);}}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let native=firstTrailingBit(sourceRowGeometry(r).y);
 let v=loadf(A,native,rowMap(native,r));if(!finite(v)){reportAt(NONFINITE,92u,r);}else{outputCorrection[r]=v;}}}
`;
