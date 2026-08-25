/** Packet-masked, full-domain eight-sweep velocity extension. */
export interface SparseCM12VelocityExtensionLayout {
  readonly headerBaseWords: number;
  readonly validityABaseWords: number;
  readonly validityBBaseWords: number;
  readonly acceptedDepthBaseWords: number;
  readonly cellCapacity: number;
  readonly packetCapacity: number;
  /** Compact direct-domain packets per leaf; masks retain the 64-slot TEI stride. */
  readonly dispatchPacketsPerLeaf: 1 | 8 | 64;
  readonly dispatchPacketCount: number;
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

export function sparseCM12VelocityExtensionDispatchPacketsPerLeaf(
  brickFineResolution: 4 | 8 | 16,
): 1 | 8 | 64 {
  const packets = (brickFineResolution / 4) ** 3;
  if (packets !== 1 && packets !== 8 && packets !== 64) {
    throw new RangeError("velocity-extension topology profile must be B4, B8, or B16");
  }
  return packets;
}

/** Header fields are receipts only; none schedules work. */
export const SPARSE_CM12_VELOCITY_EXTENSION_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, capacity: 3, packetCapacity: 4,
  sourceFrameGeneration: 5, topologyGeneration: 6, reserved0: 7,
  validCellCount: 8, emptyPacketCount: 9, reserved1: 10,
  faultCount: 11, firstFaultCell: 12, firstFaultDepth: 13,
  reserved2: 14, reserved3: 15,
} as const);

export interface SparseCM12VelocityExtensionMaskDensity {
  readonly validCellCount: number;
  readonly emptyPacketCount: number;
}

const popcount32 = (input: number): number => {
  let value = input >>> 0;
  value -= (value >>> 1) & 0x5555_5555;
  value = (value & 0x3333_3333) + ((value >>> 2) & 0x3333_3333);
  return Math.imul((value + (value >>> 4)) & 0x0f0f_0f0f, 0x0101_0101) >>> 24;
};

/** QA-only census over compact direct packets stored at stable leaf*64 addresses. */
export function sparseCM12VelocityExtensionMaskDensity(
  validity: Uint32Array,
  packetCapacity: number,
  dispatchPacketCount = packetCapacity,
): SparseCM12VelocityExtensionMaskDensity {
  integer(packetCapacity, "packetCapacity");
  integer(dispatchPacketCount, "dispatchPacketCount");
  if (validity.length !== 2 * packetCapacity) {
    throw new RangeError(`velocity-extension validity mask has ${validity.length}`
      + ` words; expected ${2 * packetCapacity}`);
  }
  if (dispatchPacketCount > packetCapacity) {
    throw new RangeError("dispatchPacketCount exceeds packetCapacity");
  }
  if (packetCapacity === 0) {
    return Object.freeze({ validCellCount: 0, emptyPacketCount: 0 });
  }
  let validCellCount = 0, emptyPacketCount = 0;
  const census = (packet: number) => {
    const low = validity[2 * packet]!, high = validity[2 * packet + 1]!;
    validCellCount += popcount32(low) + popcount32(high);
    emptyPacketCount += Number((low | high) === 0);
  };
  if (dispatchPacketCount === packetCapacity) {
    for (let packet = 0; packet < packetCapacity; packet += 1) census(packet);
    return Object.freeze({ validCellCount, emptyPacketCount });
  }
  const leafCapacity = packetCapacity / 64;
  const dispatchPacketsPerLeaf = dispatchPacketCount / leafCapacity;
  if (!Number.isInteger(leafCapacity)
    || (dispatchPacketsPerLeaf !== 1 && dispatchPacketsPerLeaf !== 8
      && dispatchPacketsPerLeaf !== 64)) {
    throw new RangeError("compact velocity-extension packet profile is invalid");
  }
  for (let leaf = 0; leaf < leafCapacity; leaf += 1) {
    for (let local = 0; local < dispatchPacketsPerLeaf; local += 1) {
      census(64 * leaf + local);
    }
  }
  return Object.freeze({ validCellCount, emptyPacketCount });
}

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
    // Resident state fields are vec4-aligned even when the logical cell count
    // is not a multiple of four. Preserve that arena tail contract here too.
    floatCount: Math.ceil((characteristicSupportFloatBase + cellCapacity) / 4) * 4 });
}

export function createSparseCM12VelocityExtensionLayout(options: {
  readonly baseWords?: number;
  readonly cellCapacity: number;
  readonly packetCapacity?: number;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly alignmentWords?: number;
}): SparseCM12VelocityExtensionLayout {
  const alignment = integer(options.alignmentWords ?? 64, "alignmentWords");
  if (alignment === 0) throw new RangeError("alignmentWords must be positive");
  const base = integer(options.baseWords ?? 0, "baseWords");
  const cellCapacity = integer(options.cellCapacity, "cellCapacity");
  const packetCapacity = integer(options.packetCapacity
    ?? Math.ceil(cellCapacity / 64), "packetCapacity");
  const dispatchPacketsPerLeaf = options.brickFineResolution === undefined ? 64
    : sparseCM12VelocityExtensionDispatchPacketsPerLeaf(options.brickFineResolution);
  // Packet storage uses TEI's stable leaf*64 address even when the direct domain
  // visits only B4's first packet or B8's first eight packets in each leaf.
  const leafCapacity = Math.ceil(packetCapacity / 64);
  const dispatchPacketCount = options.brickFineResolution === undefined
    ? packetCapacity : integer(leafCapacity * dispatchPacketsPerLeaf,
      "dispatchPacketCount");
  const headerBaseWords = Math.ceil(base / alignment) * alignment;
  const validityABaseWords = headerBaseWords + SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
  const validityBBaseWords = validityABaseWords + 2 * packetCapacity;
  const acceptedDepthBaseWords = validityBBaseWords + 2 * packetCapacity;
  return Object.freeze({ headerBaseWords, validityABaseWords, validityBBaseWords,
    acceptedDepthBaseWords,
    cellCapacity, packetCapacity, dispatchPacketsPerLeaf, dispatchPacketCount,
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
  readonly brickFineResolution?: 4 | 8 | 16;
}): Readonly<{ activity: SparseCM12VelocityExtensionLayout;
  state: SparseCM12VelocityExtensionStateLayout }> {
  return Object.freeze({
    activity: createSparseCM12VelocityExtensionLayout({
      baseWords: options.activityTailWords, cellCapacity: options.cellCapacity,
      packetCapacity: options.packetCapacity,
      brickFineResolution: options.brickFineResolution,
    }),
    state: createSparseCM12VelocityExtensionStateLayout({
      baseFloats: options.stateTailFloats, cellCapacity: options.cellCapacity,
    }),
  });
}
