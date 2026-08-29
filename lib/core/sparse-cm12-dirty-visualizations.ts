/**
 * Sparse CM12 temporal-coherence publications, and the overlay modes that read
 * them.
 *
 * These modes deliberately use the generic grid overlay: the dirty records
 * are indexed by stable logical 4³ work tiles in the same finest-domain frame
 * as Grid structure, so the existing slice/volume controls are the honest way
 * to inspect them. The solver may omit the optional source while bringing the
 * producer up; the shader renders that state as UNKNOWN rather than as clean.
 *
 * **No catalog entries.** Ten coherence views is more than half the field
 * picker spent on one solver's bookkeeping, and none of them answer a question
 * about the water — so they were withdrawn from the catalog and from the
 * method's `supportedFieldModes`, which is the only thing the picker and the
 * quick bar are built from. The overlay codes below and the shader that
 * dispatches on them are untouched: `parseGridOverlayMode` still accepts these
 * mode names, so a link that names one still draws it, and re-listing them is
 * one entry per view whenever a coherence audit wants them back.
 */

export const SPARSE_CM12_DIRTY_GPU_MAGIC = 0x434d_4431;
export const SPARSE_CM12_DIRTY_GPU_VERSION = 1;
export const SPARSE_CM12_DIRTY_GPU_HEADER_WORDS = 16;
export const SPARSE_CM12_DIRTY_GPU_RECORD_WORDS = 16;
export const SPARSE_CM12_DIRTY_TILE_EDGE = 4;

/** Optional CMD1 v1 tail describing which logical stages share physical passes. */
export const SPARSE_CM12_DIRTY_PACKING_GPU_MAGIC = 0x504b_5431; // "PKT1"
export const SPARSE_CM12_DIRTY_PACKING_GPU_VERSION = 1;
export const SPARSE_CM12_DIRTY_PACKING_GPU_HEADER_WORDS = 16;
export const SPARSE_CM12_DIRTY_PACKING_GPU_PACKET_WORDS = 4;
export const SPARSE_CM12_DIRTY_PACKING_GPU_TILE_WORDS = 4;
export const SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS = 32;

export const SPARSE_CM12_DIRTY_GPU_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, recordWords: 3,
  tileEdge: 4, tilesPerLogicalBrickAxis: 5, logicalBrickCount: 6, stageCount: 7,
  acceptedGeneration: 8, candidateGeneration: 9, provenanceGeneration: 10,
  publicationFlags: 11, uncoveredWriteCount: 12,
  firstFaultLogicalBrick: 13, firstFaultLocalTile: 14, firstFaultStage: 15,
} as const);

export const SPARSE_CM12_DIRTY_GPU_RECORD = Object.freeze({
  directStageMask: 0,
  closureStageMask: 1,
  processedStageMask: 2,
  skippedStageMask: 3,
  unknownStageMask: 4,
  uncoveredWriteStageMask: 5,
  originCauseMask: 6,
  inheritedCauseMask: 7,
  packedClosureDepths: 8,
  packedOriginStages: 9,
  stageGenerationPairs: 10,
} as const);

export const SPARSE_CM12_DIRTY_PACKING_GPU_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, packetWords: 3, tileWords: 4,
  packetCount: 5, tileCount: 6, epochCount: 7,
  acceptedGeneration: 8, candidateGeneration: 9, provenanceGeneration: 10,
  publicationFlags: 11, uncoveredPackingFaultCount: 12,
  firstFaultTile: 13, firstFaultPacket: 14, reserved: 15,
} as const);

export const SPARSE_CM12_DIRTY_PACKING_GPU_PACKET = Object.freeze({
  identity: 0, epoch: 1, logicalStageMask: 2, executionGeneration: 3,
} as const);

export const SPARSE_CM12_DIRTY_PACKING_GPU_TILE = Object.freeze({
  assignedPacketMask: 0,
  executedPacketMask: 1,
  /** Five 6-bit lanes for stages 0..4: zero skipped, 1..32 packet index + 1. */
  packedStagePacketIndicesLow: 2,
  /** Stage 5 uses the low 6 bits; the remaining bits are reserved and zero. */
  packedStagePacketIndexHigh: 3,
} as const);

export function sparseCM12DirtyTilesPerLogicalBrickAxis(brickFineResolution: number): number {
  if (!Number.isSafeInteger(brickFineResolution) || brickFineResolution < SPARSE_CM12_DIRTY_TILE_EDGE
    || brickFineResolution % SPARSE_CM12_DIRTY_TILE_EDGE !== 0) {
    throw new RangeError("Sparse CM12 dirty tiles require a brick resolution divisible by 4");
  }
  return brickFineResolution / SPARSE_CM12_DIRTY_TILE_EDGE;
}

export function sparseCM12DirtyGPUBufferWords(
  logicalBrickCount: number,
  brickFineResolution: number,
): number {
  if (!Number.isSafeInteger(logicalBrickCount) || logicalBrickCount < 0) {
    throw new RangeError("Sparse CM12 dirty logical-brick count must be non-negative");
  }
  const tilesPerAxis = sparseCM12DirtyTilesPerLogicalBrickAxis(brickFineResolution);
  return SPARSE_CM12_DIRTY_GPU_HEADER_WORDS
    + logicalBrickCount * tilesPerAxis ** 3 * SPARSE_CM12_DIRTY_GPU_RECORD_WORDS;
}

/** Total words for CMD1 records plus the optional fixed-layout PKT1 tail. */
export function sparseCM12DirtyPackingGPUBufferWords(
  logicalBrickCount: number,
  brickFineResolution: number,
  packetCount: number,
): number {
  if (!Number.isSafeInteger(packetCount) || packetCount < 0
    || packetCount > SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS) {
    throw new RangeError("Sparse CM12 dirty pass packing supports 0..32 packets");
  }
  const tilesPerAxis = sparseCM12DirtyTilesPerLogicalBrickAxis(brickFineResolution);
  const tileCount = logicalBrickCount * tilesPerAxis ** 3;
  return sparseCM12DirtyGPUBufferWords(logicalBrickCount, brickFineResolution)
    + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER_WORDS
    + packetCount * SPARSE_CM12_DIRTY_PACKING_GPU_PACKET_WORDS
    + tileCount * SPARSE_CM12_DIRTY_PACKING_GPU_TILE_WORDS;
}

export const SPARSE_CM12_DIRTY_STAGE = Object.freeze({
  facePreparation: 0,
  massTransport: 1,
  gammaTransport: 2,
  surfaceConditioning: 3,
  pressureCoefficients: 4,
  presentation: 5,
} as const);
export const SPARSE_CM12_DIRTY_STAGE_COUNT = 6;

export const SPARSE_CM12_DIRTY_CAUSE_BIT = Object.freeze({
  topologyCreated: 1 << 0,
  topologyRetired: 1 << 1,
  phaseCrossing: 1 << 2,
  densityChanged: 1 << 3,
  gammaChanged: 1 << 4,
  velocityCharacteristic: 1 << 5,
  cflGrowth: 1 << 6,
  movingSolidSweep: 1 << 7,
  boundarySource: 1 << 8,
  coefficientChanged: 1 << 9,
  dependencyClosure: 1 << 10,
  generationMismatch: 1 << 11,
  capacityOrProvenance: 1 << 12,
  pageActivated: 1 << 13,
  pageRetired: 1 << 14,
} as const);

export const SPARSE_CM12_DIRTY_PUBLICATION_FLAG = Object.freeze({
  complete: 1 << 0,
  accepted: 1 << 1,
  rejected: 1 << 2,
  capacityOrProvenanceFault: 1 << 3,
} as const);

/**
 * Header words followed by one 16-word record for every logical base-brick
 * 4³ tile. Logical identity, not a mutable physical slot, keeps the picture
 * stable through coarsening and refinement.
 *
 * Header:
 *  0 magic, 1 version, 2 header words, 3 record words,
 *  4 tile edge, 5 tiles/base-brick axis, 6 logical brick count, 7 stage count,
 *  8 accepted generation, 9 candidate generation, 10 provenance generation,
 * 11 publication flags, 12 uncovered-write count,
 * 13 first-fault logical brick, 14 first-fault local tile, 15 first-fault stage.
 *
 * Record:
 *  0 direct stages, 1 closure stages, 2 processed/executed stages,
 *  3 skipped/reused stages, 4 unknown stages, 5 uncovered-write stages,
 *  6 origin cause mask, 7 inherited cause mask,
 *  8 closure depths (six 4-bit lanes), 9 origin stages (six 3-bit lanes),
 * 10..15 producer generation in low 16 bits and consumer in high 16 bits.
 */
export interface SparseCM12DirtyGPUHeader {
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly provenanceGeneration: number;
  readonly publicationFlags: number;
  readonly uncoveredWriteCount: number;
}

/** Optional PKT1 header appended after all v1 CMD1 tile records. */
export interface SparseCM12DirtyPackingGPUHeader {
  readonly packetCount: number;
  readonly tileCount: number;
  readonly epochCount: number;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly provenanceGeneration: number;
  readonly publicationFlags: number;
  readonly uncoveredPackingFaultCount: number;
}

export const SPARSE_CM12_DIRTY_OVERLAY_MODES = [
  "dirty-face-preparation",
  "dirty-mass-transport",
  "dirty-gamma-transport",
  "dirty-surface-conditioning",
  "dirty-pressure-coefficients",
  "dirty-presentation",
  "dirty-causes",
  "dirty-closure-depth",
  "dirty-generations",
  "dirty-pass-packing",
] as const;
export type SparseCM12DirtyOverlayMode = typeof SPARSE_CM12_DIRTY_OVERLAY_MODES[number];

export function isSparseCM12DirtyOverlayMode(value: unknown): value is SparseCM12DirtyOverlayMode {
  return typeof value === "string"
    && (SPARSE_CM12_DIRTY_OVERLAY_MODES as readonly string[]).includes(value);
}

/** Generic grid-overlay mode codes; 1..10 are the pre-existing field modes. */
export const SPARSE_CM12_DIRTY_OVERLAY_CODES: Readonly<Record<SparseCM12DirtyOverlayMode, number>> =
  Object.freeze({
    "dirty-face-preparation": 11,
    "dirty-mass-transport": 12,
    "dirty-gamma-transport": 13,
    "dirty-surface-conditioning": 14,
    "dirty-pressure-coefficients": 15,
    "dirty-presentation": 16,
    "dirty-causes": 17,
    "dirty-closure-depth": 18,
    "dirty-generations": 19,
    "dirty-pass-packing": 20,
  });

export function sparseCM12DirtyOverlayCode(value: unknown): number {
  return isSparseCM12DirtyOverlayMode(value) ? SPARSE_CM12_DIRTY_OVERLAY_CODES[value] : 0;
}
