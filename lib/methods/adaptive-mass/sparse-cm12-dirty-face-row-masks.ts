/**
 * DFRM1: transient face-row masks in compact direct-packet order.
 *
 * Each direct ordinal owns twelve words: low/high lane masks for the x/y/z
 * positive-cell owner rows followed by the three positive-side sparse-air
 * boundary families. The stable topology packet remains
 * `leaf * 64 + localPacket`; compact B4/B8 profiles simply omit the
 * structurally empty reserved packet slots.
 */

export const SPARSE_CM12_DIRTY_FACE_ROW_MASK_FAMILY_COUNT = 6;
export const SPARSE_CM12_DIRTY_FACE_ROW_MASK_WORDS_PER_PACKET = 12;
export const SPARSE_CM12_DIRTY_FACE_ROW_MASK_DISPATCH_WIDTH = 65_535;

export interface SparseCM12DirtyFaceRowMaskLayout {
  readonly baseWords: number;
  readonly maskBaseWords: number;
  readonly acceptedPressureBitsBaseWords: number;
  readonly cellCapacity: number;
  readonly packetCapacity: number;
  readonly dispatchPacketsPerLeaf: 1 | 8 | 64;
  readonly dispatchPacketCount: number;
  readonly dispatchWidth: number;
  readonly dispatchRows: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const integer = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0)) {
    throw new RangeError(`${label} must be a ${positive ? "positive" : "non-negative"}`
      + " safe integer");
  }
  return value;
};

export function createSparseCM12DirtyFaceRowMaskLayout(options: Readonly<{
  baseWords?: number;
  cellCapacity: number;
  packetCapacity: number;
  dispatchPacketsPerLeaf: 1 | 8 | 64;
  dispatchPacketCount: number;
  alignmentWords?: number;
}>): SparseCM12DirtyFaceRowMaskLayout {
  const alignmentWords = integer(options.alignmentWords ?? 64,
    "DFRM1 alignmentWords", true);
  const unalignedBase = integer(options.baseWords ?? 0, "DFRM1 baseWords");
  const baseWords = Math.ceil(unalignedBase / alignmentWords) * alignmentWords;
  const cellCapacity = integer(options.cellCapacity, "DFRM1 cellCapacity", true);
  const packetCapacity = integer(options.packetCapacity, "DFRM1 packetCapacity", true);
  const dispatchPacketCount = integer(options.dispatchPacketCount,
    "DFRM1 dispatchPacketCount", true);
  if (packetCapacity % 64 !== 0
    || dispatchPacketCount !== packetCapacity / 64 * options.dispatchPacketsPerLeaf) {
    throw new RangeError("DFRM1 compact packet profile does not match stable packet capacity");
  }
  const dispatchWidth = Math.min(
    SPARSE_CM12_DIRTY_FACE_ROW_MASK_DISPATCH_WIDTH, dispatchPacketCount);
  const dispatchRows = Math.ceil(dispatchPacketCount / dispatchWidth);
  const maskBaseWords = baseWords;
  const acceptedPressureBitsBaseWords = maskBaseWords
    + SPARSE_CM12_DIRTY_FACE_ROW_MASK_WORDS_PER_PACKET * dispatchPacketCount;
  const totalWords = acceptedPressureBitsBaseWords + cellCapacity;
  return Object.freeze({ baseWords, maskBaseWords, acceptedPressureBitsBaseWords,
    cellCapacity, packetCapacity,
    dispatchPacketsPerLeaf: options.dispatchPacketsPerLeaf,
    dispatchPacketCount, dispatchWidth, dispatchRows, totalWords,
    totalBytes: 4 * (totalWords - baseWords) });
}

/** Initial accepted pressure is deliberately impossible, forcing bootstrap. */
export function createSparseCM12DirtyFaceRowMaskInitialWords(
  layout: SparseCM12DirtyFaceRowMaskLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  words.fill(0xffff_ffff,
    layout.acceptedPressureBitsBaseWords - layout.baseWords);
  return words;
}
