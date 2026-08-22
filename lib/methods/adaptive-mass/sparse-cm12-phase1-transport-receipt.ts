/** QA-only raw transport receipt captured at the actual pass boundaries. */
export const SPARSE_CM12_PHASE1_TRANSPORT_QA_MAGIC = 0x5031_5451; // P1TQ
export const SPARSE_CM12_PHASE1_TRANSPORT_QA_VERSION = 1;
export const SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER_WORDS = 24;

export const SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER = Object.freeze({
  magic: 0, version: 1, cellCapacity: 2,
  frameGeneration: 3, topologyGeneration: 4,
  effectiveVelocityFrameGeneration: 5,
  effectiveVelocityTopologyGeneration: 6,
  traceCount: 7, betaCount: 8, deficitCount: 9, massCount: 10,
  packetCellCount: 11, packetFault: 12,
  packetFaultCell: 13, packetFaultActual: 14, packetFaultPacked: 15,
  sharpeningTileClaim: 16, sharpeningTile: 17, sharpeningSlot: 18,
  sharpeningSlotGeneration: 19, sharpeningRecordGeneration: 20,
  sharpeningRecordPacketId: 21, sharpeningRecordMaskLow: 22,
  sharpeningRecordMaskHigh: 23,
} as const);

export interface SparseCM12Phase1TransportQALayout {
  readonly baseWords: number;
  readonly cellCapacity: number;
  readonly departureBaseWords: number;
  readonly stencilCellBaseWords: number;
  readonly stencilWeightBaseWords: number;
  readonly betaBaseWords: number;
  readonly deficitDensityBaseWords: number;
  readonly deficitGammaBaseWords: number;
  readonly massDensityBaseWords: number;
  readonly massGammaBaseWords: number;
  readonly packetIdBaseWords: number;
  readonly packetLaneBaseWords: number;
  readonly totalWords: number;
}

const checked = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} is outside the u32 range`);
  }
  return value;
};

export function createSparseCM12Phase1TransportQALayout(options: {
  readonly baseWords: number;
  readonly cellCapacity: number;
}): SparseCM12Phase1TransportQALayout {
  const baseWords = checked(options.baseWords, "baseWords");
  const cellCapacity = checked(options.cellCapacity, "cellCapacity");
  const departureBaseWords = checked(baseWords
    + SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER_WORDS, "departureBaseWords");
  const stencilCellBaseWords = checked(departureBaseWords + 3 * cellCapacity,
    "stencilCellBaseWords");
  const stencilWeightBaseWords = checked(stencilCellBaseWords + 8 * cellCapacity,
    "stencilWeightBaseWords");
  const betaBaseWords = checked(stencilWeightBaseWords + 8 * cellCapacity,
    "betaBaseWords");
  const deficitDensityBaseWords = checked(betaBaseWords + cellCapacity,
    "deficitDensityBaseWords");
  const deficitGammaBaseWords = checked(deficitDensityBaseWords + cellCapacity,
    "deficitGammaBaseWords");
  const massDensityBaseWords = checked(deficitGammaBaseWords + cellCapacity,
    "massDensityBaseWords");
  const massGammaBaseWords = checked(massDensityBaseWords + cellCapacity,
    "massGammaBaseWords");
  const packetIdBaseWords = checked(massGammaBaseWords + cellCapacity,
    "packetIdBaseWords");
  const packetLaneBaseWords = checked(packetIdBaseWords + cellCapacity,
    "packetLaneBaseWords");
  const totalWords = checked(packetLaneBaseWords + cellCapacity, "totalWords");
  return Object.freeze({ baseWords, cellCapacity, departureBaseWords,
    stencilCellBaseWords, stencilWeightBaseWords, betaBaseWords,
    deficitDensityBaseWords, deficitGammaBaseWords, massDensityBaseWords,
    massGammaBaseWords, packetIdBaseWords, packetLaneBaseWords, totalWords });
}

export interface SparseCM12Phase1TransportReceipt {
  readonly stencilCellsSha256: string;
  readonly stencilWeightBitsSha256: string;
  readonly departureBitsSha256: string;
  readonly betaFixedSha256: string;
  readonly deficitDensityFixedSha256: string;
  readonly deficitGammaFixedSha256: string;
  readonly massReceiptSha256: string;
  readonly packetSha256: string;
  readonly publishedPacketSha256: string;
  readonly packetCount: number;
  readonly packetCellCount: number;
  readonly duplicateScatterCellCount: number;
  readonly omittedAcceptedCellCount: number;
  readonly acceptedTopologyGeneration: number;
  readonly transportTopologyGeneration: number;
  readonly frameGeneration: number;
  readonly packetGeneration: number;
  readonly publishedPacketGeneration: number;
  readonly effectiveVelocityTopologyGeneration: number;
  readonly effectiveVelocityFrameGeneration: number;
}

export async function sparseCM12Phase1Sha256(view: ArrayBufferView): Promise<string> {
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  const copy = Uint8Array.from(bytes);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", copy);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0"))
    .join("");
}
