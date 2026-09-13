/**
 * One immutable, fully rebuilt connectivity image for an accepted CM12
 * topology generation.  The image deliberately has no dirty-record ABI: a
 * changed source generation invalidates the whole image and every plane is
 * rebuilt before the header can be sealed.
 */

export const SPARSE_CM12_COMPILED_TOPOLOGY_MAGIC = 0x434e5831; // "CNX1"
export const SPARSE_CM12_COMPILED_TOPOLOGY_VERSION = 1;
export const SPARSE_CM12_COMPILED_TOPOLOGY_HEADER_WORDS = 64;
export const SPARSE_CM12_COMPILED_TOPOLOGY_INVALID = 0xffff_ffff;
export const SPARSE_CM12_COMPILED_TOPOLOGY_ROW_RECORD_WORDS = 4;
export const SPARSE_CM12_COMPILED_TOPOLOGY_TERM_WORDS = 2;
export const SPARSE_CM12_COMPILED_TOPOLOGY_INCIDENCE_WORDS = 2;
export const SPARSE_CM12_COMPILED_TOPOLOGY_ROW_RECORD = Object.freeze({
  stableRow: 0, termBegin: 1, termEnd: 2, metadata: 3,
} as const);
export const SPARSE_CM12_COMPILED_TOPOLOGY_TERM = Object.freeze({
  stableCell: 0, coefficient: 1,
} as const);
export const SPARSE_CM12_COMPILED_TOPOLOGY_INCIDENCE = Object.freeze({
  rowOrdinal: 0, ownTerm: 1,
} as const);

export const SPARSE_CM12_COMPILED_TOPOLOGY_PHASE = Object.freeze({
  empty: 0,
  building: 1,
  accepted: 2,
  fault: 3,
} as const);

export const SPARSE_CM12_COMPILED_TOPOLOGY_FAULT = Object.freeze({
  sourceNotAccepted: 1 << 0,
  cellCapacity: 1 << 1,
  rowCapacity: 1 << 2,
  termCapacity: 1 << 3,
  incidenceCapacity: 1 << 4,
  duplicateCell: 1 << 5,
  duplicateRow: 1 << 6,
  malformedRow: 1 << 7,
  malformedIncidence: 1 << 8,
  missingRequiredView: 1 << 9,
  transportCapacity: 1 << 10,
  staleSource: 1 << 11,
  transportMalformed: 1 << 12,
} as const);

export const SPARSE_CM12_COMPILED_TOPOLOGY_VIEW = Object.freeze({
  connectivity: 1 << 0,
  transport: 1 << 1,
  velocityExtension: 1 << 2,
  projection: 1 << 3,
  presentation: 1 << 4,
} as const);

export const SPARSE_CM12_COMPILED_TOPOLOGY_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, totalWords: 3,
  phase: 4, fault: 5, firstFaultId: 6, rebuildRequired: 7,
  generation: 8, sourceTopologyGeneration: 9, sourceAcceptedSlot: 10,
  acceptedCellCount: 11, acceptedRowWorklistCount: 12, acceptedRowCount: 13,
  orderedTermCount: 14, cellIncidenceCount: 15,
  requiredViews: 16, readyViews: 17,
  physicalFaceCount: 18, physicalFaceEntryCount: 19,
  transportGeneration: 20, velocityExtensionGeneration: 21,
  projectionGeneration: 22, presentationGeneration: 23,
  certificate: 24,
  cellCapacity: 25, rowCapacity: 26, termCapacity: 27,
  incidenceCapacity: 28, physicalFaceCapacity: 29,
  acceptedCellIdsBase: 30, cellOrdinalByStableBase: 31,
  acceptedCellRangesBase: 32, acceptedRowRecordsBase: 33,
  rowOrdinalByStableBase: 34, orderedTermsBase: 35,
  cellIncidencesBase: 36,
  clearedGeneration: 37, cellPlaneGeneration: 38, rowPlaneGeneration: 39,
  incidencePlaneGeneration: 40,
} as const);

export interface SparseCM12CompiledTopologyLayout {
  readonly baseWords: number;
  readonly headerBaseWords: number;
  readonly acceptedCellIdsBaseWords: number;
  readonly cellOrdinalByStableBaseWords: number;
  readonly acceptedCellRangesBaseWords: number;
  readonly acceptedRowRecordsBaseWords: number;
  readonly rowOrdinalByStableBaseWords: number;
  readonly orderedTermsBaseWords: number;
  readonly cellIncidencesBaseWords: number;
  readonly totalWords: number;
  readonly cellCapacity: number;
  readonly rowCapacity: number;
  readonly termCapacity: number;
  readonly incidenceCapacity: number;
  readonly physicalFaceCapacity: number;
  readonly requiredViews: number;
}

/**
 * Storage is dense by capacity and rebuilt as one image. Accepted cell and row
 * ordinals are compact. Term records retain their source term slot so row term
 * order is bit-for-bit address-compatible; cell incidences compact accepted
 * rows inside each source cell's disjoint incidence range. No plane carries a
 * per-record generation or dirty stamp.
 */

export interface SparseCM12CompiledTopologyLayoutRequest {
  readonly baseWords?: number;
  readonly cellCapacity: number;
  readonly rowCapacity: number;
  readonly termCapacity: number;
  readonly incidenceCapacity: number;
  readonly physicalFaceCapacity: number;
  readonly requiredViews?: number;
  /** WebGPU's maxStorageBufferBindingSize expressed in u32 words. */
  readonly maximumArenaWords?: number;
}

const integer = (value: number, name: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${name} must be an unsigned safe integer`);
  }
  return value;
};

const add = (at: number, words: number, name: string): number => {
  const next = at + words;
  if (!Number.isSafeInteger(next) || next > 0xffff_ffff) {
    throw new RangeError(`${name} exceeds the u32 word-address ABI`);
  }
  return next;
};

/**
 * Allocates the connectivity planes after an existing topology arena prefix.
 * Four-word alignment makes every plane suitable for future vec4 loads.
 */
export function createSparseCM12CompiledTopologyLayout(
  request: SparseCM12CompiledTopologyLayoutRequest,
): SparseCM12CompiledTopologyLayout {
  const baseWords = integer(request.baseWords ?? 0, "baseWords");
  const cellCapacity = integer(request.cellCapacity, "cellCapacity");
  const rowCapacity = integer(request.rowCapacity, "rowCapacity");
  const termCapacity = integer(request.termCapacity, "termCapacity");
  const incidenceCapacity = integer(request.incidenceCapacity, "incidenceCapacity");
  const physicalFaceCapacity = integer(request.physicalFaceCapacity, "physicalFaceCapacity");
  const allowedViews = Object.values(SPARSE_CM12_COMPILED_TOPOLOGY_VIEW)
    .reduce((sum, value) => sum | value, 0);
  const requiredViews = integer(request.requiredViews
    ?? SPARSE_CM12_COMPILED_TOPOLOGY_VIEW.connectivity, "requiredViews");
  if ((requiredViews & ~allowedViews) !== 0
      || (requiredViews & SPARSE_CM12_COMPILED_TOPOLOGY_VIEW.connectivity) === 0) {
    throw new RangeError("requiredViews must include connectivity and contain only CNX view bits");
  }
  const align4 = (value: number) => Math.ceil(value / 4) * 4;
  // The host binds the 64-word header as a small standalone storage slice for
  // indirect dispatch publication. WebGPU dynamic/storage offsets are
  // commonly 256-byte aligned, so the ABI guarantees that without relying on
  // a caller-specific device limit.
  let at = Math.ceil(baseWords / 64) * 64;
  const headerBaseWords = at;
  at = add(at, SPARSE_CM12_COMPILED_TOPOLOGY_HEADER_WORDS, "compiled topology header");
  const plane = (words: number, name: string) => {
    at = align4(at);
    const result = at;
    at = add(at, words, name);
    return result;
  };
  const acceptedCellIdsBaseWords = plane(cellCapacity, "accepted cell ids");
  const cellOrdinalByStableBaseWords = plane(cellCapacity, "cell ordinal map");
  const acceptedCellRangesBaseWords = plane(2 * cellCapacity, "cell incidence ranges");
  const acceptedRowRecordsBaseWords = plane(
    SPARSE_CM12_COMPILED_TOPOLOGY_ROW_RECORD_WORDS * rowCapacity, "accepted row records");
  const rowOrdinalByStableBaseWords = plane(rowCapacity, "row ordinal map");
  const orderedTermsBaseWords = plane(
    SPARSE_CM12_COMPILED_TOPOLOGY_TERM_WORDS * termCapacity, "ordered row terms");
  const cellIncidencesBaseWords = plane(
    SPARSE_CM12_COMPILED_TOPOLOGY_INCIDENCE_WORDS * incidenceCapacity,
    "ordered cell incidences");
  const totalWords = align4(at);
  if (request.maximumArenaWords !== undefined
      && totalWords > integer(request.maximumArenaWords, "maximumArenaWords")) {
    throw new RangeError(`compiled topology requires ${totalWords} words, exceeding the ${request.maximumArenaWords}-word arena limit`);
  }
  return Object.freeze({
    baseWords, headerBaseWords, acceptedCellIdsBaseWords,
    cellOrdinalByStableBaseWords, acceptedCellRangesBaseWords,
    acceptedRowRecordsBaseWords, rowOrdinalByStableBaseWords,
    orderedTermsBaseWords, cellIncidencesBaseWords, totalWords,
    cellCapacity, rowCapacity, termCapacity, incidenceCapacity,
    physicalFaceCapacity, requiredViews,
  });
}

export function sparseCM12CompiledTopologyAdditionalWords(
  layout: SparseCM12CompiledTopologyLayout,
): number {
  return layout.totalWords - layout.baseWords;
}

export function createSparseCM12CompiledTopologyInitialWords(
  layout: SparseCM12CompiledTopologyLayout,
): Uint32Array {
  // Only this header is uploaded. The first GPU full-build clears the two
  // stable-id maps; allocating and transferring the capacity-sized tail here
  // would turn a 256-byte bootstrap into a roughly 50 MiB mini32 upload.
  const words = new Uint32Array(SPARSE_CM12_COMPILED_TOPOLOGY_HEADER_WORDS);
  const h = SPARSE_CM12_COMPILED_TOPOLOGY_HEADER;
  words[h.magic] = SPARSE_CM12_COMPILED_TOPOLOGY_MAGIC;
  words[h.version] = SPARSE_CM12_COMPILED_TOPOLOGY_VERSION;
  words[h.headerWords] = SPARSE_CM12_COMPILED_TOPOLOGY_HEADER_WORDS;
  words[h.totalWords] = layout.totalWords;
  words[h.firstFaultId] = SPARSE_CM12_COMPILED_TOPOLOGY_INVALID;
  words[h.requiredViews] = layout.requiredViews;
  words[h.cellCapacity] = layout.cellCapacity;
  words[h.rowCapacity] = layout.rowCapacity;
  words[h.termCapacity] = layout.termCapacity;
  words[h.incidenceCapacity] = layout.incidenceCapacity;
  words[h.physicalFaceCapacity] = layout.physicalFaceCapacity;
  words[h.acceptedCellIdsBase] = layout.acceptedCellIdsBaseWords;
  words[h.cellOrdinalByStableBase] = layout.cellOrdinalByStableBaseWords;
  words[h.acceptedCellRangesBase] = layout.acceptedCellRangesBaseWords;
  words[h.acceptedRowRecordsBase] = layout.acceptedRowRecordsBaseWords;
  words[h.rowOrdinalByStableBase] = layout.rowOrdinalByStableBaseWords;
  words[h.orderedTermsBase] = layout.orderedTermsBaseWords;
  words[h.cellIncidencesBase] = layout.cellIncidencesBaseWords;
  return words;
}

const mix = (hash: number, value: number): number => {
  hash = Math.imul((hash ^ value) >>> 0, 0x01000193);
  return (hash ^ (hash >>> 16)) >>> 0;
};

export interface SparseCM12CompiledTopologyManifestInput {
  readonly sourceTopologyGeneration: number;
  readonly sourceAcceptedSlot: number;
  readonly acceptedCellCount: number;
  readonly acceptedRowWorklistCount: number;
  readonly acceptedRowCount: number;
  readonly orderedTermCount: number;
  readonly cellIncidenceCount: number;
  readonly readyViews?: number;
  readonly physicalFaceCount?: number;
  readonly physicalFaceEntryCount?: number;
  readonly transportGeneration?: number;
  readonly velocityExtensionGeneration?: number;
  readonly projectionGeneration?: number;
  readonly presentationGeneration?: number;
}

export function sparseCM12CompiledTopologyCertificate(
  layout: SparseCM12CompiledTopologyLayout,
  input: SparseCM12CompiledTopologyManifestInput,
): number {
  let hash = 0x811c9dc5;
  for (const value of [SPARSE_CM12_COMPILED_TOPOLOGY_MAGIC,
    SPARSE_CM12_COMPILED_TOPOLOGY_VERSION, input.sourceTopologyGeneration,
    input.sourceAcceptedSlot, input.acceptedCellCount,
    input.acceptedRowWorklistCount, input.acceptedRowCount,
    input.orderedTermCount, input.cellIncidenceCount,
    input.readyViews ?? SPARSE_CM12_COMPILED_TOPOLOGY_VIEW.connectivity,
    input.physicalFaceCount ?? 0, input.physicalFaceEntryCount ?? 0,
    layout.cellCapacity, layout.rowCapacity, layout.termCapacity,
    layout.incidenceCapacity, layout.physicalFaceCapacity]) hash = mix(hash, value >>> 0);
  return hash;
}

/** CPU oracle for the GPU seal transaction; it intentionally compiles no deltas. */
export function compileSparseCM12TopologyGenerationManifest(
  layout: SparseCM12CompiledTopologyLayout,
  input: SparseCM12CompiledTopologyManifestInput,
): Uint32Array {
  const checked = { ...input };
  for (const [name, value] of Object.entries(checked)) {
    if (value !== undefined) integer(value, name);
  }
  if (input.sourceAcceptedSlot > 1) throw new RangeError("sourceAcceptedSlot must be 0 or 1");
  if (input.acceptedCellCount > layout.cellCapacity) throw new RangeError("accepted cell capacity exceeded");
  if (input.acceptedRowWorklistCount > layout.rowCapacity
      || input.acceptedRowCount > input.acceptedRowWorklistCount) throw new RangeError("accepted row capacity exceeded");
  if (input.orderedTermCount > layout.termCapacity) throw new RangeError("ordered term capacity exceeded");
  if (input.cellIncidenceCount > layout.incidenceCapacity) throw new RangeError("cell incidence capacity exceeded");
  if ((input.physicalFaceCount ?? 0) > layout.physicalFaceCapacity
      || (input.physicalFaceEntryCount ?? 0) > 2 * layout.physicalFaceCapacity) {
    throw new RangeError("physical transport capacity exceeded");
  }
  const readyViews = input.readyViews ?? SPARSE_CM12_COMPILED_TOPOLOGY_VIEW.connectivity;
  if ((readyViews & layout.requiredViews) !== layout.requiredViews) {
    throw new Error("required compiled topology views are not ready");
  }
  const words = createSparseCM12CompiledTopologyInitialWords(layout);
  const h = SPARSE_CM12_COMPILED_TOPOLOGY_HEADER;
  words[h.phase] = SPARSE_CM12_COMPILED_TOPOLOGY_PHASE.accepted;
  words[h.generation] = input.sourceTopologyGeneration;
  words[h.sourceTopologyGeneration] = input.sourceTopologyGeneration;
  words[h.sourceAcceptedSlot] = input.sourceAcceptedSlot;
  words[h.acceptedCellCount] = input.acceptedCellCount;
  words[h.acceptedRowWorklistCount] = input.acceptedRowWorklistCount;
  words[h.acceptedRowCount] = input.acceptedRowCount;
  words[h.orderedTermCount] = input.orderedTermCount;
  words[h.cellIncidenceCount] = input.cellIncidenceCount;
  words[h.readyViews] = readyViews;
  words[h.physicalFaceCount] = input.physicalFaceCount ?? 0;
  words[h.physicalFaceEntryCount] = input.physicalFaceEntryCount ?? 0;
  words[h.transportGeneration] = input.transportGeneration ?? 0;
  words[h.velocityExtensionGeneration] = input.velocityExtensionGeneration ?? 0;
  words[h.projectionGeneration] = input.projectionGeneration ?? 0;
  words[h.presentationGeneration] = input.presentationGeneration ?? 0;
  words[h.clearedGeneration] = input.sourceTopologyGeneration;
  words[h.cellPlaneGeneration] = input.sourceTopologyGeneration;
  words[h.rowPlaneGeneration] = input.sourceTopologyGeneration;
  words[h.incidencePlaneGeneration] = input.sourceTopologyGeneration;
  words[h.certificate] = sparseCM12CompiledTopologyCertificate(layout, input);
  return words;
}

export function sparseCM12CompiledTopologyAccepted(
  words: Uint32Array,
  layout: SparseCM12CompiledTopologyLayout,
  sourceTopologyGeneration: number,
  sourceAcceptedSlot: number,
): boolean {
  const h = SPARSE_CM12_COMPILED_TOPOLOGY_HEADER;
  const input: SparseCM12CompiledTopologyManifestInput = {
    sourceTopologyGeneration: words[h.sourceTopologyGeneration]!,
    sourceAcceptedSlot: words[h.sourceAcceptedSlot]!,
    acceptedCellCount: words[h.acceptedCellCount]!,
    acceptedRowWorklistCount: words[h.acceptedRowWorklistCount]!,
    acceptedRowCount: words[h.acceptedRowCount]!,
    orderedTermCount: words[h.orderedTermCount]!,
    cellIncidenceCount: words[h.cellIncidenceCount]!,
    readyViews: words[h.readyViews]!,
    physicalFaceCount: words[h.physicalFaceCount]!,
    physicalFaceEntryCount: words[h.physicalFaceEntryCount]!,
  };
  return words[h.magic] === SPARSE_CM12_COMPILED_TOPOLOGY_MAGIC
    && words[h.version] === SPARSE_CM12_COMPILED_TOPOLOGY_VERSION
    && words[h.phase] === SPARSE_CM12_COMPILED_TOPOLOGY_PHASE.accepted
    && words[h.fault] === 0
    && words[h.sourceTopologyGeneration] === sourceTopologyGeneration
    && words[h.sourceAcceptedSlot] === sourceAcceptedSlot
    && (words[h.readyViews]! & layout.requiredViews) === layout.requiredViews
    && words[h.certificate] === sparseCM12CompiledTopologyCertificate(layout, input);
}
