/** Packet-masked, full-domain eight-sweep velocity extension. */
export interface SparseCM12VelocityExtensionLayout {
  readonly headerBaseWords: number;
  readonly validityABaseWords: number;
  readonly validityBBaseWords: number;
  readonly acceptedDepthBaseWords: number;
  readonly cellCapacity: number;
  readonly packetCapacity: number;
  readonly totalWords: number;
}

export interface SparseCM12VelocityExtensionStateLayout {
  readonly characteristicSupportFloatBase: number;
  readonly floatCount: number;
}

export const SPARSE_CM12_VELOCITY_EXTENSION_MAGIC = 0x5645_5832; // VEX2
export const SPARSE_CM12_VELOCITY_EXTENSION_VERSION = 2;
export const SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS = 16;
export const SPARSE_CM12_VELOCITY_EXTENSION_DEPTH = 8;
/** WebGPU's guaranteed per-dimension workgroup-dispatch limit. */
export const SPARSE_CM12_VELOCITY_EXTENSION_DISPATCH_WIDTH = 65_535;

/** Header fields are receipts only; none schedules work. */
export const SPARSE_CM12_VELOCITY_EXTENSION_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, capacity: 3, packetCapacity: 4,
  completedFrameGeneration: 5, topologyGeneration: 6, executedCellCount: 7,
  validCellCount: 8, emptyPacketCount: 9, neighborLoadCount: 10,
  faultCount: 11, firstFaultCell: 12, firstFaultDepth: 13,
  reserved0: 14, reserved1: 15,
} as const);

export type SparseCM12VelocityExtensionBank = "source" | "destination";

export function sparseCM12VelocityExtensionInputBank(
  depth: number,
): SparseCM12VelocityExtensionBank {
  if (!Number.isInteger(depth) || depth < 1
    || depth > SPARSE_CM12_VELOCITY_EXTENSION_DEPTH) {
    throw new RangeError("velocity-extension depth must be in [1, 8]");
  }
  return (depth & 1) === 1 ? "destination" : "source";
}

export function sparseCM12VelocityExtensionOutputBank(
  depth: number,
): SparseCM12VelocityExtensionBank {
  return sparseCM12VelocityExtensionInputBank(depth) === "destination"
    ? "source" : "destination";
}

const integer = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
  return value;
};

export function createSparseCM12VelocityExtensionStateLayout(options: {
  readonly baseFloats: number;
  readonly cellCapacity: number;
}): SparseCM12VelocityExtensionStateLayout {
  const base = integer(options.baseFloats, "baseFloats");
  const cellCapacity = integer(options.cellCapacity, "cellCapacity");
  const characteristicSupportFloatBase = Math.ceil(base / 4) * 4;
  return Object.freeze({ characteristicSupportFloatBase,
    floatCount: characteristicSupportFloatBase + cellCapacity });
}

export function createSparseCM12VelocityExtensionLayout(options: {
  readonly baseWords?: number;
  readonly cellCapacity: number;
  readonly packetCapacity?: number;
  readonly alignmentWords?: number;
}): SparseCM12VelocityExtensionLayout {
  const alignment = integer(options.alignmentWords ?? 64, "alignmentWords");
  if (alignment === 0) throw new RangeError("alignmentWords must be positive");
  const base = integer(options.baseWords ?? 0, "baseWords");
  const cellCapacity = integer(options.cellCapacity, "cellCapacity");
  const packetCapacity = integer(options.packetCapacity
    ?? Math.ceil(cellCapacity / 64), "packetCapacity");
  const headerBaseWords = Math.ceil(base / alignment) * alignment;
  const validityABaseWords = headerBaseWords + SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
  const validityBBaseWords = validityABaseWords + 2 * packetCapacity;
  const acceptedDepthBaseWords = validityBBaseWords + 2 * packetCapacity;
  return Object.freeze({ headerBaseWords, validityABaseWords, validityBBaseWords,
    acceptedDepthBaseWords,
    cellCapacity, packetCapacity,
    totalWords: acceptedDepthBaseWords + cellCapacity });
}

export function createSparseCM12VelocityExtensionInitialWords(
  layout: SparseCM12VelocityExtensionLayout,
): Uint32Array {
  const result = new Uint32Array(layout.totalWords - layout.headerBaseWords);
  const h = SPARSE_CM12_VELOCITY_EXTENSION_HEADER;
  result[h.magic] = SPARSE_CM12_VELOCITY_EXTENSION_MAGIC;
  result[h.version] = SPARSE_CM12_VELOCITY_EXTENSION_VERSION;
  result[h.headerWords] = SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
  result[h.capacity] = layout.cellCapacity;
  result[h.packetCapacity] = layout.packetCapacity;
  result[h.firstFaultCell] = 0xffff_ffff;
  result[h.firstFaultDepth] = 0xffff_ffff;
  result.fill(0xffff_ffff, layout.acceptedDepthBaseWords - layout.headerBaseWords);
  return result;
}

export function createSparseCM12VelocityExtensionResidentLayouts(options: {
  readonly activityTailWords: number;
  readonly stateTailFloats: number;
  readonly cellCapacity: number;
  readonly packetCapacity?: number;
}): Readonly<{ activity: SparseCM12VelocityExtensionLayout;
  state: SparseCM12VelocityExtensionStateLayout }> {
  return Object.freeze({
    activity: createSparseCM12VelocityExtensionLayout({
      baseWords: options.activityTailWords, cellCapacity: options.cellCapacity,
      packetCapacity: options.packetCapacity,
    }),
    state: createSparseCM12VelocityExtensionStateLayout({
      baseFloats: options.stateTailFloats, cellCapacity: options.cellCapacity,
    }),
  });
}
