/**
 * CPU oracle and packed-arena contract for Section 5 velocity extrapolation.
 *
 * Aanjaneya et al. (2017), Section 5 extrapolates velocity on regular octree
 * faces before interpolating full vectors at cell centers.  Pressure rows are
 * therefore not a sufficient support domain: positive-air cells needed by a
 * cube or transition tetrahedron must remain addressable without acquiring a
 * momentum degree of freedom.  This module plans that sparse cell domain.  It
 * deliberately does not extrapolate row vectors or mutate compact rows.
 */

export const STRUCTURED_AIR_SUPPORT_INVALID = 0xffff_ffff;
export const STRUCTURED_AIR_SUPPORT_ARENA_MAGIC = 0x5355_5050; // "SUPP"
export const STRUCTURED_AIR_SUPPORT_ARENA_VERSION = 1;
export const STRUCTURED_AIR_SUPPORT_CONTROL_WORDS = 16;
export const STRUCTURED_AIR_SUPPORT_RECORD_WORDS = 8;
/** Four 2:1 subfaces on each positive Cartesian axis. */
export const STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS = 12;
export const STRUCTURED_AIR_SUPPORT_VECTOR_WORDS = 4;

export const STRUCTURED_AIR_SUPPORT_ARENA_FLAGS = Object.freeze({
  ready: 1 << 0,
  validated: 1 << 1,
} as const);

export const STRUCTURED_AIR_SUPPORT_RECORD_FLAGS = Object.freeze({
  compactMomentumRow: 1 << 0,
  regularCell: 1 << 1,
  interfaceSource: 1 << 2,
  transitionSelector: 1 << 3,
  fineBandDemand: 1 << 4,
  regularInterpolationAnchor: 1 << 5,
  regularInterpolationStencil: 1 << 6,
  extensionClosure: 1 << 7,
} as const);

export type StructuredAirSupportCoordinate = readonly [number, number, number];

export interface StructuredAirSupportIdentity {
  readonly origin: StructuredAirSupportCoordinate;
  readonly size: number;
}

/** One leaf from the complete, accepted octree topology publication. */
export interface StructuredAirSupportTopologyCell extends StructuredAirSupportIdentity {
  readonly caseId: number;
  readonly transform: number;
  readonly regular: boolean;
  /** Undefined for positive-air support. Air never receives a momentum DOF. */
  readonly compactRow?: number;
}

export interface StructuredAirSupportRequest {
  readonly dimensions: StructuredAirSupportCoordinate;
  /** Literal proof that `topologyCells` is the complete accepted leaf set. */
  readonly topologyComplete: true;
  readonly topologyCells: readonly StructuredAirSupportTopologyCell[];
  /** Liquid/interface compact rows whose incident faces seed extrapolation. */
  readonly interfaceRows: readonly StructuredAirSupportIdentity[];
  /** Cell-center vertices selected by transition tetrahedra. */
  readonly transitionSelectorCells?: readonly StructuredAirSupportIdentity[];
  /** Containing leaves demanded by the independently allocated fine-phi band. */
  readonly fineBandCells?: readonly StructuredAirSupportIdentity[];
  /** Regular cube anchors; every in-domain member of their 3x3x3 support is mandatory. */
  readonly regularInterpolationRows?: readonly StructuredAirSupportIdentity[];
  readonly extensionLayers: number;
  readonly capacity: number;
  readonly generation: number;
}

export interface StructuredAirSupportRecord extends StructuredAirSupportTopologyCell {
  readonly flags: number;
  readonly layer: number;
}

export interface StructuredAirSupportArenaLayout {
  readonly capacity: number;
  readonly controlOffsetWords: 0;
  readonly recordOffsetWords: number;
  readonly vectorOffsetWords: number;
  readonly allocatedWords: number;
  readonly allocatedBytes: number;
}

export interface StructuredAirSupportPlan {
  readonly dimensions: StructuredAirSupportCoordinate;
  readonly generation: number;
  readonly extensionLayers: number;
  readonly records: readonly StructuredAirSupportRecord[];
  readonly layout: StructuredAirSupportArenaLayout;
}

const key = (identity: StructuredAirSupportIdentity) =>
  `${identity.size}:${identity.origin[0]},${identity.origin[1]},${identity.origin[2]}`;

function u32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be a non-negative u32 integer`);
  }
  return value;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value;
}

function powerOfTwo(value: number, label: string): number {
  positiveInteger(value, label);
  if ((value & (value - 1)) !== 0) throw new RangeError(`${label} must be a power of two`);
  return value;
}

function validateIdentity(identity: StructuredAirSupportIdentity,
  dimensions: StructuredAirSupportCoordinate, label: string): void {
  const size = powerOfTwo(identity.size, `${label} size`);
  identity.origin.forEach((coordinate, axis) => {
    if (!Number.isSafeInteger(coordinate) || coordinate < 0 || coordinate % size !== 0
      || coordinate + size > dimensions[axis]) {
      throw new RangeError(`${label} origin must be aligned and inside the physical domain`);
    }
  });
}

function inside(origin: StructuredAirSupportCoordinate, size: number,
  dimensions: StructuredAirSupportCoordinate): boolean {
  return origin.every((coordinate, axis) => coordinate >= 0 && coordinate + size <= dimensions[axis]);
}

/** Face/edge adjacency used by the catalog topology; corner-only contact is excluded. */
function catalogAdjacent(a: StructuredAirSupportIdentity, b: StructuredAirSupportIdentity): boolean {
  let touchingAxes = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const a0 = a.origin[axis], a1 = a0 + a.size;
    const b0 = b.origin[axis], b1 = b0 + b.size;
    if (a1 === b0 || b1 === a0) touchingAxes += 1;
    else if (Math.min(a1, b1) <= Math.max(a0, b0)) return false;
  }
  return touchingAxes === 1 || touchingAxes === 2;
}

function overlapVolume(a: StructuredAirSupportIdentity, b: StructuredAirSupportIdentity): boolean {
  return a.origin.every((origin, axis) =>
    Math.min(origin + a.size, b.origin[axis] + b.size) > Math.max(origin, b.origin[axis]));
}

function deterministicCellOrder(a: StructuredAirSupportTopologyCell, b: StructuredAirSupportTopologyCell): number {
  return a.size - b.size || a.origin[2] - b.origin[2]
    || a.origin[1] - b.origin[1] || a.origin[0] - b.origin[0];
}

export function planStructuredAirSupportArena(capacityValue: number): StructuredAirSupportArenaLayout {
  const capacity = positiveInteger(capacityValue, "Structured air-support capacity");
  const recordOffsetWords = STRUCTURED_AIR_SUPPORT_CONTROL_WORDS;
  const vectorOffsetWords = recordOffsetWords + capacity * STRUCTURED_AIR_SUPPORT_RECORD_WORDS;
  const usedWords = vectorOffsetWords + capacity * STRUCTURED_AIR_SUPPORT_VECTOR_WORDS;
  // The grouped binding is copy-friendly and leaves room for future control words without
  // changing the record ABI.
  const allocatedWords = Math.ceil(usedWords / 64) * 64;
  return { capacity, controlOffsetWords: 0, recordOffsetWords, vectorOffsetWords,
    allocatedWords, allocatedBytes: allocatedWords * 4 };
}

/**
 * Build deterministic support closure from accepted topology identities.
 * Missing demanded identities, incomplete regular cube support, overlapping
 * leaves, and capacity overflow reject the whole publication.
 */
export function buildStructuredAirSupport(request: StructuredAirSupportRequest): StructuredAirSupportPlan {
  if (request.topologyComplete !== true) {
    throw new RangeError("Structured air support requires a complete accepted topology publication");
  }
  const dimensions = request.dimensions.map((value, axis) =>
    positiveInteger(value, `Structured air-support dimension ${axis}`)) as [number, number, number];
  const capacity = positiveInteger(request.capacity, "Structured air-support capacity");
  const extensionLayers = u32(request.extensionLayers, "Structured air-support extension layers");
  const generation = u32(request.generation, "Structured air-support generation");
  if (generation === 0) throw new RangeError("Structured air-support generation zero is unpublished");

  const cells = [...request.topologyCells].sort(deterministicCellOrder);
  const byKey = new Map<string, StructuredAirSupportTopologyCell>();
  const compactRows = new Set<number>();
  cells.forEach((cell, index) => {
    validateIdentity(cell, dimensions, `Structured topology cell ${index}`);
    u32(cell.caseId, `Structured topology cell ${index} case ID`);
    if (!Number.isSafeInteger(cell.transform) || cell.transform < 0 || cell.transform >= 48) {
      throw new RangeError(`Structured topology cell ${index} transform must name a cube symmetry`);
    }
    if (cell.compactRow !== undefined) {
      const row = u32(cell.compactRow, `Structured topology cell ${index} compact row`);
      if (row === STRUCTURED_AIR_SUPPORT_INVALID || compactRows.has(row)) {
        throw new RangeError("Structured topology compact rows must be unique and cannot use INVALID");
      }
      compactRows.add(row);
    }
    const identityKey = key(cell);
    if (byKey.has(identityKey)) throw new RangeError(`Duplicate structured topology cell ${identityKey}`);
    byKey.set(identityKey, cell);
  });
  for (let a = 0; a < cells.length; a += 1) for (let b = a + 1; b < cells.length; b += 1) {
    if (overlapVolume(cells[a]!, cells[b]!)) {
      throw new RangeError(`Structured topology leaves overlap at ${key(cells[a]!)}`);
    }
  }

  const flagsByKey = new Map<string, number>();
  const layerByKey = new Map<string, number>();
  const requireAcceptedCell = (identity: StructuredAirSupportIdentity,
    flags: number, label: string, layer = 0) => {
    validateIdentity(identity, dimensions, label);
    const identityKey = key(identity), cell = byKey.get(identityKey);
    if (!cell) throw new RangeError(`${label} ${identityKey} is absent from accepted topology`);
    flagsByKey.set(identityKey, (flagsByKey.get(identityKey) ?? 0) | flags);
    layerByKey.set(identityKey, Math.min(layerByKey.get(identityKey) ?? layer, layer));
    return cell;
  };

  for (const identity of request.interfaceRows) {
    const cell = requireAcceptedCell(identity, STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.interfaceSource,
      "Structured interface source");
    if (cell.compactRow === undefined) {
      throw new RangeError(`Structured interface source ${key(identity)} has no compact momentum row`);
    }
  }
  for (const identity of request.transitionSelectorCells ?? []) {
    requireAcceptedCell(identity, STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.transitionSelector,
      "Structured transition selector");
  }
  for (const identity of request.fineBandCells ?? []) {
    requireAcceptedCell(identity, STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.fineBandDemand,
      "Structured fine-band demand");
  }
  for (const identity of request.regularInterpolationRows ?? []) {
    const anchor = requireAcceptedCell(identity,
      STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.regularInterpolationAnchor,
      "Structured regular interpolation anchor");
    if (!anchor.regular) {
      throw new RangeError(`Structured regular interpolation anchor ${key(identity)} is not regular`);
    }
    for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0 && dz === 0) continue;
        const origin = [identity.origin[0] + dx * identity.size,
          identity.origin[1] + dy * identity.size,
          identity.origin[2] + dz * identity.size] as const;
        // Physical exterior is a boundary condition, never an air support row.
        if (!inside(origin, identity.size, dimensions)) continue;
        requireAcceptedCell({ origin, size: identity.size },
          STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.regularInterpolationStencil,
          "Structured regular 3x3x3 stencil");
      }
    }
  }

  let frontier = new Set(flagsByKey.keys());
  for (let layer = 1; layer <= extensionLayers && frontier.size > 0; layer += 1) {
    const next = new Set<string>();
    for (const frontierKey of frontier) {
      const source = byKey.get(frontierKey)!;
      for (const candidate of cells) {
        const candidateKey = key(candidate);
        if (flagsByKey.has(candidateKey) || !catalogAdjacent(source, candidate)) continue;
        flagsByKey.set(candidateKey, STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.extensionClosure);
        layerByKey.set(candidateKey, layer);
        next.add(candidateKey);
      }
    }
    frontier = next;
  }

  if (flagsByKey.size > capacity) {
    throw new RangeError(`Structured air-support closure ${flagsByKey.size} exceeds capacity ${capacity}`);
  }
  const records = cells.filter((cell) => flagsByKey.has(key(cell))).map((cell) => {
    let flags = flagsByKey.get(key(cell))!;
    if (cell.compactRow !== undefined) flags |= STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.compactMomentumRow;
    if (cell.regular) flags |= STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.regularCell;
    return Object.freeze({ ...cell, flags, layer: layerByKey.get(key(cell))! });
  });
  return Object.freeze({ dimensions: Object.freeze(dimensions), generation, extensionLayers,
    records: Object.freeze(records), layout: Object.freeze(planStructuredAirSupportArena(capacity)) });
}

/** Exact spatial lookup used by CPU fixtures and future production differential tests. */
export function findStructuredAirSupportRecord(plan: StructuredAirSupportPlan,
  identity: StructuredAirSupportIdentity): StructuredAirSupportRecord | undefined {
  const identityKey = key(identity);
  return plan.records.find((record) => key(record) === identityKey);
}

/** Pack the grouped support-record/final-vector arena; vectors begin invalid (w = 0). */
export function packStructuredAirSupportArena(plan: StructuredAirSupportPlan): Uint32Array<ArrayBuffer> {
  if (plan.records.length > plan.layout.capacity) {
    throw new RangeError("Structured air-support plan exceeds its packed arena capacity");
  }
  const words = new Uint32Array(new ArrayBuffer(plan.layout.allocatedBytes));
  words.fill(STRUCTURED_AIR_SUPPORT_INVALID);
  words.set([
    STRUCTURED_AIR_SUPPORT_ARENA_MAGIC,
    STRUCTURED_AIR_SUPPORT_ARENA_VERSION,
    plan.generation,
    plan.records.length,
    plan.layout.capacity,
    STRUCTURED_AIR_SUPPORT_ARENA_FLAGS.ready | STRUCTURED_AIR_SUPPORT_ARENA_FLAGS.validated,
    plan.extensionLayers,
    plan.layout.recordOffsetWords,
    plan.layout.vectorOffsetWords,
    plan.layout.allocatedWords,
    STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS,
    STRUCTURED_AIR_SUPPORT_RECORD_WORDS,
    STRUCTURED_AIR_SUPPORT_VECTOR_WORDS,
    0, 0, 0,
  ], plan.layout.controlOffsetWords);
  plan.records.forEach((record, index) => {
    const base = plan.layout.recordOffsetWords + index * STRUCTURED_AIR_SUPPORT_RECORD_WORDS;
    words.set([
      ...record.origin,
      record.size,
      record.caseId,
      (record.transform | (record.flags << 6)) >>> 0,
      record.compactRow ?? STRUCTURED_AIR_SUPPORT_INVALID,
      plan.generation,
    ], base);
  });
  words.fill(0, plan.layout.vectorOffsetWords,
    plan.layout.vectorOffsetWords + plan.layout.capacity * STRUCTURED_AIR_SUPPORT_VECTOR_WORDS);
  return words;
}
