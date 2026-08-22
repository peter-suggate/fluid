/**
 * TPM1: packet-local producer masks emitted by conservative scalar transport.
 *
 * One transport workgroup owns one 64-cell rung packet.  Its lanes publish exact
 * surface-feature and density-changed bitsets.  Later dispatches expand the
 * masks into TRA row closure and VEX root/neighbor closure; transport gather
 * never walks either graph.
 */

export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_MAGIC = 0x5450_4d31; // TPM1
export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_VERSION = 1;
export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER_WORDS = 32;
export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_INVALID = 0xffff_ffff;

export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, packetCapacity: 3,
  phase: 4, fault: 5, candidateGeneration: 6, publishedPacketCount: 7,
  surfaceCellCount: 8, densityChangedCellCount: 9, firstFaultPacket: 10,
  packetStampBase: 11, surfaceLowBase: 12, surfaceHighBase: 13,
  densityLowBase: 14, densityHighBase: 15, totalWords: 16,
  reservedBase: 17,
} as const);

export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_PHASE = Object.freeze({
  idle: 0, collecting: 1, published: 2, fault: 3,
} as const);

export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, packetOutOfRange: 2, generationMismatch: 3,
  duplicatePacket: 4, incompletePublication: 5,
} as const);

/** VEX cause retained from the scalar density producer's exact HEAD path. */
export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_VEX_CAUSE = 1 << 2;

export interface SparseCM12TransportProducerMaskLayout {
  readonly baseWords: number;
  readonly packetCapacity: number;
  readonly packetStampBaseWords: number;
  readonly surfaceLowBaseWords: number;
  readonly surfaceHighBaseWords: number;
  readonly densityLowBaseWords: number;
  readonly densityHighBaseWords: number;
  /** Absolute exclusive end in the containing atomic arena. */
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12TransportProducerPacketMask {
  readonly surfaceLow: number;
  readonly surfaceHigh: number;
  readonly densityLow: number;
  readonly densityHigh: number;
}

const integer = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0)
    || value >= SPARSE_CM12_TRANSPORT_PRODUCER_MASK_INVALID) {
    throw new RangeError(`${label} must be a ${positive ? "positive" : "non-negative"} safe integer`);
  }
  return value;
};
const align64 = (words: number, label: string): number => integer(
  Math.ceil(words / 64) * 64, label,
);

/** Creates an appendable SoA layout in an existing `array<atomic<u32>>` arena. */
export function createSparseCM12TransportProducerMaskLayout(options: {
  readonly packetCapacity: number;
  readonly baseWords?: number;
}): SparseCM12TransportProducerMaskLayout {
  const packetCapacity = integer(options.packetCapacity, "TPM1 packetCapacity", true);
  const baseWords = align64(integer(options.baseWords ?? 0, "TPM1 baseWords"),
    "TPM1 aligned baseWords");
  const packetStampBaseWords = align64(baseWords
    + SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER_WORDS, "TPM1 packet stamp base");
  const surfaceLowBaseWords = align64(packetStampBaseWords + packetCapacity,
    "TPM1 surface-low base");
  const surfaceHighBaseWords = align64(surfaceLowBaseWords + packetCapacity,
    "TPM1 surface-high base");
  const densityLowBaseWords = align64(surfaceHighBaseWords + packetCapacity,
    "TPM1 density-low base");
  const densityHighBaseWords = align64(densityLowBaseWords + packetCapacity,
    "TPM1 density-high base");
  const totalWords = align64(densityHighBaseWords + packetCapacity,
    "TPM1 totalWords");
  return Object.freeze({ baseWords, packetCapacity, packetStampBaseWords,
    surfaceLowBaseWords, surfaceHighBaseWords, densityLowBaseWords,
    densityHighBaseWords, totalWords, totalBytes: 4 * (totalWords - baseWords) });
}

/** Initial bytes for the layout's slice, ready to upload at `4 * baseWords`. */
export function createSparseCM12TransportProducerMaskInitialWords(
  layout: SparseCM12TransportProducerMaskLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER;
  words[h.magic] = SPARSE_CM12_TRANSPORT_PRODUCER_MASK_MAGIC;
  words[h.version] = SPARSE_CM12_TRANSPORT_PRODUCER_MASK_VERSION;
  words[h.headerWords] = SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER_WORDS;
  words[h.packetCapacity] = layout.packetCapacity;
  words[h.phase] = SPARSE_CM12_TRANSPORT_PRODUCER_MASK_PHASE.idle;
  words[h.firstFaultPacket] = SPARSE_CM12_TRANSPORT_PRODUCER_MASK_INVALID;
  words[h.packetStampBase] = layout.packetStampBaseWords;
  words[h.surfaceLowBase] = layout.surfaceLowBaseWords;
  words[h.surfaceHighBase] = layout.surfaceHighBaseWords;
  words[h.densityLowBase] = layout.densityLowBaseWords;
  words[h.densityHighBase] = layout.densityHighBaseWords;
  words[h.totalWords] = layout.totalWords;
  return words;
}

/** CPU oracle for the portable two-u32 workgroup ballot. */
export function packSparseCM12TransportProducerMask(
  lanes: readonly boolean[],
): readonly [low: number, high: number] {
  if (lanes.length !== 64) throw new RangeError("TPM1 mask requires exactly 64 lanes");
  let low = 0;
  let high = 0;
  for (let lane = 0; lane < 64; lane += 1) {
    if (!lanes[lane]) continue;
    if (lane < 32) low = (low | (1 << lane)) >>> 0;
    else high = (high | (1 << (lane - 32))) >>> 0;
  }
  return Object.freeze([low, high] as const);
}

export function sparseCM12TransportProducerMaskHasLane(
  low: number, high: number, lane: number,
): boolean {
  if (!Number.isInteger(lane) || lane < 0 || lane >= 64) {
    throw new RangeError("TPM1 lane must be in [0, 63]");
  }
  const word = lane < 32 ? low : high;
  return ((word >>> (lane & 31)) & 1) !== 0;
}

/**
 * Exact CPU reference for the two compiler dispatches.  It deliberately uses
 * arrays (not sets) while traversing so tests can compare legacy visitation;
 * returned authorities are canonicalized only at the publication boundary.
 */
export function compileSparseCM12TransportProducerMaskReference(options: {
  readonly mask: SparseCM12TransportProducerPacketMask;
  readonly packetCells: readonly number[];
  readonly incidenceRows: (cell: number) => readonly number[];
  readonly rowTermCells: (row: number) => readonly number[];
  readonly rowAccepted?: (row: number) => boolean;
  readonly cellActive?: (cell: number) => boolean;
}): Readonly<{ traRows: readonly number[]; vexRoots: readonly number[] }> {
  if (options.packetCells.length !== 64) {
    throw new RangeError("TPM1 reference requires exactly 64 packet cells");
  }
  const accepted = options.rowAccepted ?? (() => true);
  const active = options.cellActive ?? (() => true);
  const traRows = new Set<number>();
  const vexRoots = new Set<number>();
  for (let lane = 0; lane < 64; lane += 1) {
    const cell = options.packetCells[lane]!;
    if (!active(cell)) continue;
    if (sparseCM12TransportProducerMaskHasLane(
      options.mask.surfaceLow, options.mask.surfaceHigh, lane,
    )) {
      for (const row of options.incidenceRows(cell)) if (accepted(row)) traRows.add(row);
    }
    if (sparseCM12TransportProducerMaskHasLane(
      options.mask.densityLow, options.mask.densityHigh, lane,
    )) {
      vexRoots.add(cell);
      for (const row of options.incidenceRows(cell)) {
        if (!accepted(row)) continue;
        for (const neighbor of options.rowTermCells(row)) {
          if (neighbor !== cell && active(neighbor)) vexRoots.add(neighbor);
        }
      }
    }
  }
  return Object.freeze({
    traRows: Object.freeze([...traRows].sort((a, b) => a - b)),
    vexRoots: Object.freeze([...vexRoots].sort((a, b) => a - b)),
  });
}

/** Required GPU ordering; authority begin/finalize calls bracket these hooks. */
export const SPARSE_CM12_TRANSPORT_PRODUCER_MASK_DISPATCH_ORDER = Object.freeze([
  "beginSparseCM12TransportProducerMasks + beginTransportRowAuthority",
  "gatherConservativeDensity (calls cm12TransportProducerMaskPublish once per lane)",
  "sealSparseCM12TransportProducerMasks",
  "compileSparseCM12TransportRowMasks + compileSparseCM12VexRootMasks (VEX candidate is already collecting)",
  "finalizeTransportRowAuthority",
  "late frame: beginVelocityExtensionCandidate + sealVelocityExtensionRoots",
] as const);
