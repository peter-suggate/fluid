/**
 * Setaluri et al. Section 6.3 pressure-operator layout.
 *
 * This module is the bounded numerical oracle for the production GPU layout:
 * one diagonal plus eighteen spatial channels, page-derivable neighbours, and
 * cross-level couplings represented exclusively by the exact E / E^T ghost
 * incidence.  The older resolved-row graph is deliberately accepted only by
 * `applyResolvedPressureOracle`, the differential-test oracle at the bottom.
 */

export const OCTREE_SECTION63_COEFFICIENT_CHANNELS = 19 as const;
export const OCTREE_SECTION63_PAGE_SHAPE = [8, 8, 4] as const;

export const OCTREE_SECTION63_DIRECTIONS = Object.freeze([
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
  [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
  [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
] as const);

export interface Section63Leaf {
  readonly origin: readonly [number, number, number];
  /** Power-of-two width in finest cells. */
  readonly size: number;
  /** Non-negative physical-boundary/Dirichlet contribution. */
  readonly anchor?: number;
}

export interface Section63Contact {
  readonly negative: number;
  readonly positive: number;
  readonly coefficient: number;
}

export interface Section63SparseLevel {
  readonly level: number;
  readonly scale: number;
  readonly coordinates: readonly (readonly [number, number, number])[];
  readonly owners: Uint32Array;
  /** Bit zero active, bit one ghost. */
  readonly flags: Uint8Array;
  /** Slot-major: diagonal followed by the eighteen canonical directions. */
  readonly coefficients: Float64Array;
  /** Compact physical-page ordinal per slot. */
  readonly pages: Uint32Array;
}

export interface Section63Operator {
  readonly leafCount: number;
  readonly levels: readonly Section63SparseLevel[];
  readonly coefficientChannels: typeof OCTREE_SECTION63_COEFFICIENT_CHANNELS;
  readonly storedNeighbourIndices: 0;
}

export interface Section63BandwidthEstimate {
  readonly rows: number;
  readonly averageNeighbours: number;
  readonly resolvedGatherBytes: number;
  readonly section63StreamBytes: number;
  readonly byteReductionRatio: number;
  readonly cachePressureBytes: Readonly<{ resolvedRows: number; catalogTable: number; pageValues: number }>;
}

const ACTIVE = 1;
const GHOST = 2;
const key = (q: readonly number[]) => `${q[0]},${q[1]},${q[2]}`;

function powerOfTwo(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || (value & (value - 1)) !== 0) {
    throw new RangeError(`${label} must be a positive power of two`);
  }
  return Math.round(Math.log2(value));
}

function validOrigin(origin: readonly number[], size: number, label: string): void {
  if (origin.length !== 3 || origin.some((value) => !Number.isSafeInteger(value)
    || value < 0 || value % size !== 0)) {
    throw new RangeError(`${label} origin must be non-negative and aligned`);
  }
}

function contactCoordinate(owner: Section63Leaf, neighbour: Section63Leaf, scale: number): [number, number, number] {
  const result: [number, number, number] = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const begin = owner.origin[axis]!, end = begin + owner.size;
    const otherBegin = neighbour.origin[axis]!, otherEnd = otherBegin + neighbour.size;
    const sample = otherBegin >= end ? end - 0.5 * scale
      : otherEnd <= begin ? begin + 0.5 * scale
        : Math.max(begin + 0.5 * scale,
          Math.min(end - 0.5 * scale, 0.5 * (otherBegin + otherEnd)));
    result[axis] = Math.floor(sample / scale);
  }
  return result;
}

function directionChannel(a: readonly number[], b: readonly number[]): number {
  const delta = [b[0]! - a[0]!, b[1]! - a[1]!, b[2]! - a[2]!];
  const index = OCTREE_SECTION63_DIRECTIONS.findIndex((direction) =>
    direction[0] === delta[0] && direction[1] === delta[1] && direction[2] === delta[2]);
  if (index < 0) throw new RangeError(`Section 6.3 contact offset ${delta.join(",")} exceeds the 19-channel stencil`);
  return index + 1;
}

/** Build the exact native-level B blocks whose sum is E^T B E. */
export function buildSection63Operator(
  leaves: readonly Section63Leaf[], contacts: readonly Section63Contact[],
): Section63Operator {
  if (leaves.length < 1) throw new RangeError("Section 6.3 requires at least one leaf");
  const native = leaves.map((leaf, index) => {
    const level = powerOfTwo(leaf.size, `Section 6.3 leaf ${index} size`);
    validOrigin(leaf.origin, leaf.size, `Section 6.3 leaf ${index}`);
    const anchor = leaf.anchor ?? 0;
    if (!Number.isFinite(anchor) || anchor < 0) throw new RangeError(`Section 6.3 leaf ${index} anchor is invalid`);
    return level;
  });
  const maximumLevel = Math.max(...native);
  const contactsByLevel = Array.from({ length: maximumLevel + 1 }, () => [] as Section63Contact[]);
  contacts.forEach((contact, index) => {
    if (!Number.isSafeInteger(contact.negative) || !Number.isSafeInteger(contact.positive)
      || contact.negative < 0 || contact.positive < 0 || contact.negative >= leaves.length
      || contact.positive >= leaves.length || contact.negative === contact.positive
      || !Number.isFinite(contact.coefficient) || !(contact.coefficient > 0)) {
      throw new RangeError(`Section 6.3 contact ${index} is invalid`);
    }
    if (Math.abs(native[contact.negative]! - native[contact.positive]!) > 1) {
      throw new RangeError(`Section 6.3 contact ${index} violates 2:1 grading`);
    }
    contactsByLevel[Math.min(native[contact.negative]!, native[contact.positive]!)]!.push(contact);
  });

  const levels: Section63SparseLevel[] = [];
  for (let level = 0; level <= maximumLevel; level += 1) {
    const scale = 2 ** level;
    const slots = new Map<string, { coordinate: [number, number, number]; owner: number; flags: number }>();
    const insert = (coordinate: [number, number, number], owner: number, flags: number) => {
      const coordinateKey = key(coordinate), old = slots.get(coordinateKey);
      if (old && old.owner !== owner) throw new RangeError(`Section 6.3 overlapping aliases at ${coordinateKey}`);
      if (old) old.flags |= flags;
      else slots.set(coordinateKey, { coordinate, owner, flags });
    };
    leaves.forEach((leaf, owner) => {
      if (native[owner] === level) insert(leaf.origin.map((value) => value / scale) as [number, number, number], owner, ACTIVE);
    });
    for (const contact of contactsByLevel[level]!) {
      for (const [owner, other] of [[contact.negative, contact.positive], [contact.positive, contact.negative]] as const) {
        if (native[owner]! > level) insert(contactCoordinate(leaves[owner]!, leaves[other]!, scale), owner, GHOST);
      }
    }
    const cells = [...slots.values()];
    const slotByOwnerCoordinate = new Map(cells.map((cell, slot) => [`${cell.owner}:${key(cell.coordinate)}`, slot]));
    const coefficients = new Float64Array(cells.length * OCTREE_SECTION63_COEFFICIENT_CHANNELS);
    leaves.forEach((leaf, owner) => {
      if (native[owner] !== level) return;
      const slot = slotByOwnerCoordinate.get(`${owner}:${key(leaf.origin.map((value) => value / scale))}`)!;
      coefficients[slot * OCTREE_SECTION63_COEFFICIENT_CHANNELS] += leaf.anchor ?? 0;
    });
    for (const contact of contactsByLevel[level]!) {
      const aCoordinate = contactCoordinate(leaves[contact.negative]!, leaves[contact.positive]!, scale);
      const bCoordinate = contactCoordinate(leaves[contact.positive]!, leaves[contact.negative]!, scale);
      const a = slotByOwnerCoordinate.get(`${contact.negative}:${key(aCoordinate)}`);
      const b = slotByOwnerCoordinate.get(`${contact.positive}:${key(bCoordinate)}`);
      if (a === undefined || b === undefined) throw new Error("Section 6.3 ghost propagation is incomplete");
      const ab = directionChannel(aCoordinate, bCoordinate), ba = directionChannel(bCoordinate, aCoordinate);
      const aBase = a * OCTREE_SECTION63_COEFFICIENT_CHANNELS;
      const bBase = b * OCTREE_SECTION63_COEFFICIENT_CHANNELS;
      coefficients[aBase] += contact.coefficient; coefficients[bBase] += contact.coefficient;
      coefficients[aBase + ab] += contact.coefficient; coefficients[bBase + ba] += contact.coefficient;
    }
    const pageByKey = new Map<string, number>(), pages = new Uint32Array(cells.length);
    cells.forEach((cell, slot) => {
      const page = cell.coordinate.map((value, axis) => Math.floor(value / OCTREE_SECTION63_PAGE_SHAPE[axis]!));
      const pageKey = key(page); let ordinal = pageByKey.get(pageKey);
      if (ordinal === undefined) { ordinal = pageByKey.size; pageByKey.set(pageKey, ordinal); }
      pages[slot] = ordinal;
    });
    levels.push(Object.freeze({ level, scale,
      coordinates: Object.freeze(cells.map((cell) => Object.freeze(cell.coordinate))),
      owners: Uint32Array.from(cells.map((cell) => cell.owner)),
      flags: Uint8Array.from(cells.map((cell) => cell.flags)), coefficients, pages }));
  }
  return Object.freeze({ leafCount: leaves.length, levels: Object.freeze(levels),
    coefficientChannels: OCTREE_SECTION63_COEFFICIENT_CHANNELS, storedNeighbourIndices: 0 });
}

/** Production-order application: GhostValuePropagate, page B, GhostValueAccumulate. */
export function applySection63Operator(operator: Section63Operator, input: ArrayLike<number>): Float64Array {
  if (input.length !== operator.leafCount) throw new RangeError("Section 6.3 input length disagrees with leaf count");
  const output = new Float64Array(operator.leafCount);
  for (const level of operator.levels) {
    const slotCount = level.owners.length;
    const propagated = new Float64Array(slotCount);
    for (let slot = 0; slot < slotCount; slot += 1) propagated[slot] = input[level.owners[slot]!]!;
    const slotByCoordinate = new Map(level.coordinates.map((coordinate, slot) => [key(coordinate), slot]));
    const applied = new Float64Array(slotCount);
    for (let slot = 0; slot < slotCount; slot += 1) {
      const base = slot * OCTREE_SECTION63_COEFFICIENT_CHANNELS;
      let value = level.coefficients[base]! * propagated[slot]!;
      const q = level.coordinates[slot]!;
      for (let channel = 1; channel < OCTREE_SECTION63_COEFFICIENT_CHANNELS; channel += 1) {
        const coefficient = level.coefficients[base + channel]!;
        if (coefficient === 0) continue;
        const direction = OCTREE_SECTION63_DIRECTIONS[channel - 1]!;
        const neighbour = slotByCoordinate.get(key([q[0]! + direction[0], q[1]! + direction[1], q[2]! + direction[2]]));
        if (neighbour === undefined) throw new Error("Section 6.3 coefficient has no page-addressable neighbour");
        value -= coefficient * propagated[neighbour]!;
      }
      applied[slot] = value;
    }
    // Exactly E^T: the same owner incidence used by propagation, reversed.
    for (let slot = 0; slot < slotCount; slot += 1) output[level.owners[slot]!] += applied[slot]!;
  }
  return output;
}

/** The retired explicit-neighbour ABI, retained solely as a test oracle. */
export function applyResolvedPressureOracle(
  leaves: readonly Section63Leaf[], contacts: readonly Section63Contact[], input: ArrayLike<number>,
): Float64Array {
  if (input.length !== leaves.length) throw new RangeError("Resolved oracle input length disagrees with leaf count");
  const output = Float64Array.from(leaves.map((leaf, row) => (leaf.anchor ?? 0) * input[row]!));
  for (const contact of contacts) {
    const delta = input[contact.negative]! - input[contact.positive]!;
    output[contact.negative] += contact.coefficient * delta;
    output[contact.positive] -= contact.coefficient * delta;
  }
  return output;
}

export function assembleSection63Dense(operator: Section63Operator): Float64Array {
  const matrix = new Float64Array(operator.leafCount * operator.leafCount);
  for (let column = 0; column < operator.leafCount; column += 1) {
    const basis = new Float64Array(operator.leafCount); basis[column] = 1;
    const applied = applySection63Operator(operator, basis);
    for (let row = 0; row < operator.leafCount; row += 1) matrix[row * operator.leafCount + column] = applied[row]!;
  }
  return matrix;
}

export function estimateSection63Bandwidth(rows: number, averageNeighbours: number): Section63BandwidthEstimate {
  if (!Number.isSafeInteger(rows) || rows < 1 || !Number.isFinite(averageNeighbours)
    || averageNeighbours < 0 || averageNeighbours > 36) throw new RangeError("Invalid Section 6.3 bandwidth dimensions");
  // Resolved rows stream one row id and coefficient per spoke, then chase one
  // input value. Section 6.3 streams the fixed catalog channels plus page-local
  // values; its neighbour address consumes no per-cell storage.
  const resolvedGatherBytes = rows * (4 + averageNeighbours * 12);
  const section63StreamBytes = rows * (OCTREE_SECTION63_COEFFICIENT_CHANNELS * 4 + 4);
  return Object.freeze({ rows, averageNeighbours, resolvedGatherBytes, section63StreamBytes,
    byteReductionRatio: resolvedGatherBytes / section63StreamBytes,
    cachePressureBytes: Object.freeze({ resolvedRows: rows * (6 + 2 * 36 + 12) * 4,
      catalogTable: 1_608 * OCTREE_SECTION63_COEFFICIENT_CHANNELS * 4,
      pageValues: OCTREE_SECTION63_PAGE_SHAPE.reduce((product, value) => product * value, 1) * 4 }) });
}
