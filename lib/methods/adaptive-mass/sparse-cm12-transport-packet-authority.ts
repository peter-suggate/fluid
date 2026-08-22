export const SPARSE_CM12_TRANSPORT_PACKET_AUTHORITY_HEADER_WORDS = 8;
export const SPARSE_CM12_TRANSPORT_PACKET_AUTHORITY_FAMILY_COUNT = 3;

export const SPARSE_CM12_TRANSPORT_PACKET_FAMILY = Object.freeze({
  traceGammaAndBeta: 0,
  scatterDensityDeficit: 1,
  gatherConservativeDensity: 2,
} as const);

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
  readonly familyStrideWords: number;
  readonly familyHeaderBaseWords: readonly [number, number, number];
  readonly familyStampBaseWords: readonly [number, number, number];
  readonly familyMaskLowBaseWords: readonly [number, number, number];
  readonly familyMaskHighBaseWords: readonly [number, number, number];
  readonly familyListBaseWords: readonly [number, number, number];
  readonly totalWords: number;
}

const checked = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} is outside the u32 range`);
  }
  return value;
};

/**
 * Per-frame bridge from stable SRR1 spatial tiles to rung-major AEI packets.
 * The arena is deliberately separate from the immutable/plain-u32 TEI so hot
 * descriptors never become atomic storage merely to support scheduling.
 */
export function createSparseCM12TransportPacketAuthorityLayout(options: {
  readonly baseWords: number;
  readonly packetCapacity: number;
}): SparseCM12TransportPacketAuthorityLayout {
  const baseWords = checked(options.baseWords, "baseWords");
  const packetCapacity = checked(options.packetCapacity, "packetCapacity");
  const familyStrideWords = checked(
    SPARSE_CM12_TRANSPORT_PACKET_AUTHORITY_HEADER_WORDS + 4 * packetCapacity,
    "familyStrideWords",
  );
  const familyHeaderBaseWords = [0, 1, 2].map((family) => checked(
    baseWords + family * familyStrideWords, `familyHeaderBaseWords[${family}]`,
  )) as [number, number, number];
  const familyStampBaseWords = familyHeaderBaseWords.map((header, family) => checked(
    header + SPARSE_CM12_TRANSPORT_PACKET_AUTHORITY_HEADER_WORDS,
    `familyStampBaseWords[${family}]`,
  )) as [number, number, number];
  const familyMaskLowBaseWords = familyStampBaseWords.map((stamp, family) => checked(
    stamp + packetCapacity, `familyMaskLowBaseWords[${family}]`,
  )) as [number, number, number];
  const familyMaskHighBaseWords = familyMaskLowBaseWords.map((low, family) => checked(
    low + packetCapacity, `familyMaskHighBaseWords[${family}]`,
  )) as [number, number, number];
  const familyListBaseWords = familyMaskHighBaseWords.map((high, family) => checked(
    high + packetCapacity, `familyListBaseWords[${family}]`,
  )) as [number, number, number];
  const totalWords = checked(baseWords
    + SPARSE_CM12_TRANSPORT_PACKET_AUTHORITY_FAMILY_COUNT * familyStrideWords,
  "totalWords");
  return Object.freeze({ baseWords, packetCapacity, familyStrideWords,
    familyHeaderBaseWords: Object.freeze(familyHeaderBaseWords),
    familyStampBaseWords: Object.freeze(familyStampBaseWords),
    familyMaskLowBaseWords: Object.freeze(familyMaskLowBaseWords),
    familyMaskHighBaseWords: Object.freeze(familyMaskHighBaseWords),
    familyListBaseWords: Object.freeze(familyListBaseWords), totalWords });
}
