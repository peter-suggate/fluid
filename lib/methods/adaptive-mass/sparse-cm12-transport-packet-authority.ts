export const SPARSE_CM12_TRANSPORT_PACKET_AUTHORITY_INDIRECT_WORDS = 3;

export const SPARSE_CM12_TRANSPORT_PACKET_INVALID = 0xffff_ffff;

/**
 * CPU oracle for the hot staged packet prologue. The GPU loads one sealed
 * packet descriptor and its family mask on lane zero, then every lane performs
 * exactly this arithmetic from workgroup storage.
 */
export function sparseCM12TransportPacketLaneCell(options: {
  readonly first: number;
  readonly counts: readonly [number, number, number];
  readonly strideY: number;
  readonly strideZ: number;
  readonly maskLow: number;
  readonly maskHigh: number;
  readonly lane: number;
}): number {
  const { lane } = options;
  if (!Number.isInteger(lane) || lane < 0 || lane >= 64
    || options.first === SPARSE_CM12_TRANSPORT_PACKET_INVALID) {
    return SPARSE_CM12_TRANSPORT_PACKET_INVALID;
  }
  const mask = lane < 32 ? options.maskLow : options.maskHigh;
  if (((mask >>> (lane & 31)) & 1) === 0) {
    return SPARSE_CM12_TRANSPORT_PACKET_INVALID;
  }
  const x = lane & 3;
  const y = (lane >>> 2) & 3;
  const z = lane >>> 4;
  if (x >= options.counts[0] || y >= options.counts[1]
    || z >= options.counts[2]) {
    return SPARSE_CM12_TRANSPORT_PACKET_INVALID;
  }
  return (options.first + x + options.strideY * y + options.strideZ * z) >>> 0;
}

export interface SparseCM12TransportPacketAuthorityLayout {
  readonly baseWords: number;
  readonly packetCapacity: number;
  readonly dispatchPacketsPerLeaf: 1 | 8 | 64;
  readonly dispatchPacketCount: number;
  readonly dispatchWidth: number;
  readonly dispatchRows: number;
  readonly compilerWorkgroupCount: number;
  readonly compilerDispatchWidth: number;
  readonly compilerDispatchRows: number;
  readonly indirectBaseWords: number;
  readonly transportMaskLowBaseWords: number;
  readonly transportMaskHighBaseWords: number;
  readonly sharpeningMaskLowBaseWords: number;
  readonly sharpeningMaskHighBaseWords: number;
  readonly packetListBaseWords: number;
  readonly totalWords: number;
}

const checked = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} is outside the u32 range`);
  }
  return value;
};

/**
 * Per-frame compact packet execution image compiled from FSM1 neighborhoods.
 * One canonical list/mask drives all three ordered transport transforms; a
 * separate compact sharpening mask preserves trace-time closure until its
 * later consumer without duplicating the transport catalogue.
 */
export function createSparseCM12TransportPacketAuthorityLayout(options: {
  readonly baseWords: number;
  readonly packetCapacity: number;
  readonly dispatchPacketsPerLeaf: 1 | 8 | 64;
  readonly dispatchPacketCount: number;
}): SparseCM12TransportPacketAuthorityLayout {
  const baseWords = checked(options.baseWords, "baseWords");
  const packetCapacity = checked(options.packetCapacity, "packetCapacity");
  const dispatchPacketsPerLeaf = options.dispatchPacketsPerLeaf;
  if (![1, 8, 64].includes(dispatchPacketsPerLeaf)) {
    throw new RangeError("dispatchPacketsPerLeaf must be 1, 8, or 64");
  }
  const dispatchPacketCount = checked(options.dispatchPacketCount,
    "dispatchPacketCount");
  if (packetCapacity % 64 !== 0
    || dispatchPacketCount !== (packetCapacity / 64) * dispatchPacketsPerLeaf) {
    throw new RangeError("dispatchPacketCount does not match the stable packet profile");
  }
  const dispatchWidth = Math.min(65_535, dispatchPacketCount);
  const dispatchRows = Math.ceil(dispatchPacketCount / dispatchWidth);
  const compilerWorkgroupCount = Math.ceil(dispatchPacketCount / 64);
  const compilerDispatchWidth = Math.min(65_535, compilerWorkgroupCount);
  const compilerDispatchRows = Math.ceil(compilerWorkgroupCount / compilerDispatchWidth);
  const indirectBaseWords = baseWords;
  const transportMaskLowBaseWords = checked(indirectBaseWords
    + SPARSE_CM12_TRANSPORT_PACKET_AUTHORITY_INDIRECT_WORDS,
  "transportMaskLowBaseWords");
  const transportMaskHighBaseWords = checked(transportMaskLowBaseWords
    + dispatchPacketCount, "transportMaskHighBaseWords");
  const sharpeningMaskLowBaseWords = checked(transportMaskHighBaseWords
    + dispatchPacketCount, "sharpeningMaskLowBaseWords");
  const sharpeningMaskHighBaseWords = checked(sharpeningMaskLowBaseWords
    + dispatchPacketCount, "sharpeningMaskHighBaseWords");
  const packetListBaseWords = checked(sharpeningMaskHighBaseWords
    + dispatchPacketCount, "packetListBaseWords");
  const totalWords = checked(packetListBaseWords + dispatchPacketCount, "totalWords");
  return Object.freeze({ baseWords, packetCapacity, dispatchPacketsPerLeaf,
    dispatchPacketCount, dispatchWidth, dispatchRows,
    compilerWorkgroupCount, compilerDispatchWidth, compilerDispatchRows,
    indirectBaseWords,
    transportMaskLowBaseWords, transportMaskHighBaseWords,
    sharpeningMaskLowBaseWords, sharpeningMaskHighBaseWords, packetListBaseWords,
    totalWords });
}
