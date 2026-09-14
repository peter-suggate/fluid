/** GPU-resident adaptive vertex level-set storage (LSV1).
 *
 * A slot is a complete topology generation.  Two slots let the resident build
 * and transfer a candidate while the accepted field remains readable.  Cell
 * corner references use compact accepted-cell ordinals, never the much larger
 * stable/template cell capacity.
 */

export const LEVELSET_VOLUME_MAGIC = 0x4c535631; // "LSV1"
export const LEVELSET_VOLUME_VERSION = 1;
export const LEVELSET_VOLUME_GLOBAL_HEADER_WORDS = 32;
export const LEVELSET_VOLUME_SLOT_HEADER_WORDS = 32;
export const LEVELSET_VOLUME_INVALID = 0xffff_ffff;
export const LEVELSET_VOLUME_HASH_EMPTY = 0xffff_ffff;
export const LEVELSET_VOLUME_HASH_LOCK = 0xffff_fffe;
export const LEVELSET_VOLUME_WORKGROUP_SIZE = 64;

export const LEVELSET_VOLUME_PHASE = Object.freeze({
  empty: 0, building: 1, accepted: 2, fault: 3,
} as const);

export const LEVELSET_VOLUME_SUPPORT = Object.freeze({
  absent: 0, deepAir: 1, deepLiquid: 2, metric: 3,
} as const);

export const LEVELSET_VOLUME_FAULT = Object.freeze({
  vertexCapacity: 1 << 0,
  hashCapacity: 1 << 1,
  missingCorner: 1 << 2,
  malformedConstraint: 1 << 3,
  missingSupport: 1 << 4,
  staleGeneration: 1 << 5,
  invalidAdvection: 1 << 6,
} as const);

export const LEVELSET_VOLUME_GLOBAL_HEADER = Object.freeze({
  magic: 0, version: 1, totalWords: 2, acceptedSlot: 3,
  acceptedGeneration: 4, fault: 5, slot0Base: 6, slotStride: 7,
  vertexDispatch: 8,
} as const);

export const LEVELSET_VOLUME_SLOT_HEADER = Object.freeze({
  phase: 0, fault: 1, generation: 2, vertexCount: 3,
  sourceBank: 4, activeCellCount: 5, constraintCount: 6,
  firstFaultOwner: 7, hashProbeLimit: 8,
  validatedVertices: 9, validatedCells: 10,
  maximumCellSpan: 11,
  projectionWidth: 12,
} as const);

export interface LevelSetVolumeSlotLayout {
  readonly baseWords: number;
  readonly headerBaseWords: number;
  readonly cornerRefsBaseWords: number;
  readonly cellRecordsBaseWords: number;
  readonly cellHashBaseWords: number;
  readonly hashBaseWords: number;
  readonly vertexRecordsBaseWords: number;
  readonly constraintSourcesBaseWords: number;
  readonly constraintWeightsBaseWords: number;
  readonly phi0BaseWords: number;
  readonly phi1BaseWords: number;
  readonly support0BaseWords: number;
  readonly support1BaseWords: number;
  readonly totalWords: number;
}

export interface LevelSetVolumeLayout {
  readonly baseWords: number;
  readonly headerBaseWords: number;
  readonly slots: readonly [LevelSetVolumeSlotLayout, LevelSetVolumeSlotLayout];
  readonly slotStrideWords: number;
  readonly redistanceOriginalPhiBaseWords: number;
  readonly redistanceOriginalSupportBaseWords: number;
  readonly redistanceSeedPointBaseWords: number;
  readonly totalWords: number;
  readonly activeCellCapacity: number;
  readonly vertexCapacity: number;
  readonly hashCapacity: number;
  readonly cellHashCapacity: number;
  readonly hashProbeLimit: number;
}

export interface LevelSetVolumeLayoutRequest {
  readonly baseWords?: number;
  /** Maximum compact cells in one accepted/candidate generation. */
  readonly activeCellCapacity: number;
  /** Explicit active-vertex budget, including generation-growth headroom. */
  readonly vertexCapacity: number;
  /** Power-of-two hash slots. Defaults to the next power of two >= 2V. */
  readonly hashCapacity?: number;
  readonly hashProbeLimit?: number;
  readonly maximumArenaWords?: number;
}

const integer = (value: number, name: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${name} must be an unsigned safe integer`);
  }
  return value;
};

const nextPowerOfTwo = (value: number): number => {
  if (value <= 1) return 1;
  return 2 ** Math.ceil(Math.log2(value));
};

const align = (value: number, words = 4): number => Math.ceil(value / words) * words;

export function createLevelSetVolumeLayout(
  request: LevelSetVolumeLayoutRequest,
): LevelSetVolumeLayout {
  const baseWords = integer(request.baseWords ?? 0, "baseWords");
  const activeCellCapacity = integer(request.activeCellCapacity, "activeCellCapacity");
  const vertexCapacity = integer(request.vertexCapacity, "vertexCapacity");
  if (activeCellCapacity === 0 || vertexCapacity === 0) {
    throw new RangeError("adaptive level set requires nonzero cell and vertex capacities");
  }
  const hashCapacity = integer(request.hashCapacity
    ?? nextPowerOfTwo(2 * vertexCapacity), "hashCapacity");
  if ((hashCapacity & (hashCapacity - 1)) !== 0 || hashCapacity < vertexCapacity) {
    throw new RangeError("hashCapacity must be a power of two at least vertexCapacity");
  }
  const hashProbeLimit = integer(request.hashProbeLimit ?? Math.min(128, hashCapacity),
    "hashProbeLimit");
  if (hashProbeLimit === 0 || hashProbeLimit > hashCapacity) {
    throw new RangeError("hashProbeLimit must be in [1, hashCapacity]");
  }
  const cellHashCapacity = nextPowerOfTwo(2 * activeCellCapacity);
  const headerBaseWords = align(baseWords, 64);
  let at = headerBaseWords + LEVELSET_VOLUME_GLOBAL_HEADER_WORDS;
  at = align(at, 64);
  const slot0 = at;
  const makeSlot = (base: number): LevelSetVolumeSlotLayout => {
    let cursor = base;
    const header = cursor; cursor += LEVELSET_VOLUME_SLOT_HEADER_WORDS;
    const plane = (words: number) => { cursor = align(cursor); const result = cursor;
      cursor += words; return result; };
    const cornerRefs = plane(8 * activeCellCapacity);
    // lower.xyz, widths.xyz, nominal dyadic span, stable cell id.
    const cellRecords = plane(8 * activeCellCapacity);
    const cellHash = plane(cellHashCapacity);
    const hash = plane(hashCapacity);
    // xyz are signed fine-lattice coordinates; word 3 is support.
    const vertexRecords = plane(4 * vertexCapacity);
    // Count/controller width are stored in vertex-record support's upper bits;
    // four sources suffice for a face-bilinear 2:1 hanging constraint.
    const constraintSources = plane(4 * vertexCapacity);
    const constraintWeights = plane(4 * vertexCapacity);
    const phi0 = plane(vertexCapacity);
    const phi1 = plane(vertexCapacity);
    const support0 = plane(vertexCapacity);
    const support1 = plane(vertexCapacity);
    return Object.freeze({ baseWords: base, headerBaseWords: header,
      cornerRefsBaseWords: cornerRefs, cellRecordsBaseWords: cellRecords,
      cellHashBaseWords: cellHash, hashBaseWords: hash,
      vertexRecordsBaseWords: vertexRecords,
      constraintSourcesBaseWords: constraintSources,
      constraintWeightsBaseWords: constraintWeights,
      phi0BaseWords: phi0, phi1BaseWords: phi1,
      support0BaseWords: support0, support1BaseWords: support1,
      totalWords: align(cursor, 64) });
  };
  const first = makeSlot(slot0);
  const slotStrideWords = first.totalWords - first.baseWords;
  const second = makeSlot(first.totalWords);
  at = second.totalWords;
  const redistanceOriginalPhiBaseWords = align(at); at += vertexCapacity;
  const redistanceOriginalSupportBaseWords = align(at); at += vertexCapacity;
  const redistanceSeedPointBaseWords = align(at); at += 3 * vertexCapacity;
  at = align(at, 64);
  if (!Number.isSafeInteger(at) || at > 0xffff_ffff) {
    throw new RangeError("adaptive level-set arena exceeds the u32 word-address ABI");
  }
  if (request.maximumArenaWords !== undefined
      && at > integer(request.maximumArenaWords, "maximumArenaWords")) {
    throw new RangeError(`adaptive level set requires ${at} words, exceeding the ${request.maximumArenaWords}-word arena limit`);
  }
  return Object.freeze({ baseWords, headerBaseWords, slots: [first, second] as const,
    slotStrideWords, redistanceOriginalPhiBaseWords,
    redistanceOriginalSupportBaseWords, redistanceSeedPointBaseWords,
    totalWords: at, activeCellCapacity, vertexCapacity,
    hashCapacity, cellHashCapacity, hashProbeLimit });
}

export function createLevelSetVolumeInitialWords(layout: LevelSetVolumeLayout): Uint32Array {
  // The resident uploads this small prefix at headerBaseWords. Slot headers
  // are initialized by lsvBeginTopology, avoiding a capacity-sized bootstrap.
  const words = new Uint32Array(LEVELSET_VOLUME_GLOBAL_HEADER_WORDS);
  const h = LEVELSET_VOLUME_GLOBAL_HEADER;
  words[h.magic] = LEVELSET_VOLUME_MAGIC;
  words[h.version] = LEVELSET_VOLUME_VERSION;
  words[h.totalWords] = layout.totalWords;
  words[h.acceptedSlot] = LEVELSET_VOLUME_INVALID;
  words[h.slot0Base] = layout.slots[0].baseWords;
  words[h.slotStride] = layout.slotStrideWords;
  return words;
}

export const levelSetVolumeDispatchCount = (count: number): number =>
  Math.ceil(count / LEVELSET_VOLUME_WORKGROUP_SIZE);

export function levelSetVolumeClearInvocationCount(layout: LevelSetVolumeLayout): number {
  return Math.max(layout.hashCapacity, 8 * layout.activeCellCapacity,
    layout.cellHashCapacity, layout.vertexCapacity);
}
