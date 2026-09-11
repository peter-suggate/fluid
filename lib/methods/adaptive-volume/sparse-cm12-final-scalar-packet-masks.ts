/**
 * FSM1: final scalar facts in the accepted TEI2 packet address space.
 *
 * The four persistent masks are the only cross-stage dirty facts authored by
 * scalar finalization. Pressure repair consumes FLIP directly for PCM cells
 * and CHANGED | FLIP for dependent theta rows; no temporal list or dilation
 * carrier survives. Packet ids are stable TEI2 ids (`leaf * 64 + localPacket`).
 */

export const SPARSE_CM12_FINAL_SCALAR_MASK_MAGIC = 0x4653_4d31; // FSM1
export const SPARSE_CM12_FINAL_SCALAR_MASK_VERSION = 1;
export const SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS = 32;
export const SPARSE_CM12_FINAL_SCALAR_MASK_INVALID = 0xffff_ffff;

export const SPARSE_CM12_FINAL_SCALAR_MASK_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, packetCapacity: 3,
  generation: 4, topologyGeneration: 5, topologySlot: 6, phase: 7,
  changedCellCount: 8, nonexactCellCount: 9, bulkCellCount: 10,
  flipCellCount: 11, reservedCount0: 12, reservedCount1: 13,
  fault: 14, firstFaultPacket: 15, stampBase: 16,
  changedLowBase: 17, changedHighBase: 18,
  nonexactLowBase: 19, nonexactHighBase: 20,
  bulkLowBase: 21, bulkHighBase: 22,
  flipLowBase: 23, flipHighBase: 24,
  reservedBase0: 25, reservedBase1: 26,
  reservedBase2: 27, reservedBase3: 28,
  totalWords: 29, reserved0: 30, reserved1: 31,
} as const);

export const SPARSE_CM12_FINAL_SCALAR_MASK_PHASE = Object.freeze({
  idle: 0, collecting: 1, published: 2, fault: 3,
} as const);

export interface SparseCM12FinalScalarPacketMaskLayout {
  readonly baseWords: number;
  readonly packetCapacity: number;
  /** Fixed per-leaf loop bound; keeps workgroup barriers uniform. */
  readonly maximumPacketsPerLeaf: 1 | 8 | 64;
  readonly changedLowBaseWords: number;
  readonly changedHighBaseWords: number;
  readonly nonexactLowBaseWords: number;
  readonly nonexactHighBaseWords: number;
  readonly bulkLowBaseWords: number;
  readonly bulkHighBaseWords: number;
  readonly flipLowBaseWords: number;
  readonly flipHighBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const checked = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0)
    || value >= SPARSE_CM12_FINAL_SCALAR_MASK_INVALID) {
    throw new RangeError(`${label} must be a ${positive ? "positive" : "non-negative"} integer`);
  }
  return value;
};
const align64 = (value: number, label: string): number => checked(
  Math.ceil(value / 64) * 64, label,
);

export function createSparseCM12FinalScalarPacketMaskLayout(options: Readonly<{
  baseWords?: number;
  packetCapacity: number;
  brickFineResolution: 4 | 8 | 16;
}>): SparseCM12FinalScalarPacketMaskLayout {
  const baseWords = align64(checked(options.baseWords ?? 0, "FSM1 baseWords"),
    "FSM1 aligned baseWords");
  const packetCapacity = checked(options.packetCapacity, "FSM1 packetCapacity", true);
  const maximumPacketsPerLeaf = options.brickFineResolution === 4
    ? 1 : options.brickFineResolution === 8 ? 8 : 64;
  let cursor = align64(baseWords + SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS,
    "FSM1 stamp base");
  const take = () => {
    const result = cursor;
    cursor = align64(cursor + packetCapacity, "FSM1 mask plane");
    return result;
  };
  const changedLowBaseWords = take(), changedHighBaseWords = take();
  const nonexactLowBaseWords = take(), nonexactHighBaseWords = take();
  const bulkLowBaseWords = take(), bulkHighBaseWords = take();
  const flipLowBaseWords = take(), flipHighBaseWords = take();
  return Object.freeze({ baseWords, packetCapacity, maximumPacketsPerLeaf,
    changedLowBaseWords, changedHighBaseWords,
    nonexactLowBaseWords, nonexactHighBaseWords,
    bulkLowBaseWords, bulkHighBaseWords,
    flipLowBaseWords, flipHighBaseWords,
    totalWords: cursor, totalBytes: 4 * (cursor - baseWords) });
}

export function createSparseCM12FinalScalarPacketMaskInitialWords(
  layout: SparseCM12FinalScalarPacketMaskLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_FINAL_SCALAR_MASK_HEADER;
  words[h.magic] = SPARSE_CM12_FINAL_SCALAR_MASK_MAGIC;
  words[h.version] = SPARSE_CM12_FINAL_SCALAR_MASK_VERSION;
  words[h.headerWords] = SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS;
  words[h.packetCapacity] = layout.packetCapacity;
  words[h.phase] = SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.idle;
  words[h.firstFaultPacket] = SPARSE_CM12_FINAL_SCALAR_MASK_INVALID;
  words[h.changedLowBase] = layout.changedLowBaseWords;
  words[h.changedHighBase] = layout.changedHighBaseWords;
  words[h.nonexactLowBase] = layout.nonexactLowBaseWords;
  words[h.nonexactHighBase] = layout.nonexactHighBaseWords;
  words[h.bulkLowBase] = layout.bulkLowBaseWords;
  words[h.bulkHighBase] = layout.bulkHighBaseWords;
  words[h.flipLowBase] = layout.flipLowBaseWords;
  words[h.flipHighBase] = layout.flipHighBaseWords;
  words[h.totalWords] = layout.totalWords;
  return words;
}
