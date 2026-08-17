import {
  SPARSE_CM12_DIRTY_CAUSE_BIT,
  SPARSE_CM12_DIRTY_GPU_HEADER,
  SPARSE_CM12_DIRTY_GPU_HEADER_WORDS,
  SPARSE_CM12_DIRTY_GPU_MAGIC,
  SPARSE_CM12_DIRTY_GPU_RECORD,
  SPARSE_CM12_DIRTY_GPU_RECORD_WORDS,
  SPARSE_CM12_DIRTY_GPU_VERSION,
  SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS,
  SPARSE_CM12_DIRTY_PACKING_GPU_HEADER,
  SPARSE_CM12_DIRTY_PACKING_GPU_HEADER_WORDS,
  SPARSE_CM12_DIRTY_PACKING_GPU_MAGIC,
  SPARSE_CM12_DIRTY_PACKING_GPU_PACKET_WORDS,
  SPARSE_CM12_DIRTY_PACKING_GPU_TILE_WORDS,
  SPARSE_CM12_DIRTY_PACKING_GPU_VERSION,
  SPARSE_CM12_DIRTY_PUBLICATION_FLAG,
  SPARSE_CM12_DIRTY_STAGE_COUNT,
  SPARSE_CM12_DIRTY_TILE_EDGE,
  sparseCM12DirtyGPUBufferWords,
  sparseCM12DirtyPackingGPUBufferWords,
  sparseCM12DirtyTilesPerLogicalBrickAxis,
} from "../../core/sparse-cm12-dirty-visualizations";

/** Stable, method-private scheduler ABI. CMD1 remains the public visualization ABI. */
export const SPARSE_CM12_DIRTY_SCHEDULER_MAGIC = 0x434d_5331; // CMS1
export const SPARSE_CM12_DIRTY_SCHEDULER_VERSION = 1;
export const SPARSE_CM12_DIRTY_SCHEDULER_HEADER_WORDS = 36;
export const SPARSE_CM12_DIRTY_SCHEDULER_PACKET_WORDS = 29;
export const SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL_WORDS = 8;
export const SPARSE_CM12_DIRTY_SCHEDULER_QUEUE_HEADER_WORDS = 16;
export const SPARSE_CM12_DIRTY_SCHEDULER_ALIGNMENT_WORDS = 64; // 256 bytes
export const SPARSE_CM12_DIRTY_INVALID = 0xffff_ffff;

export const SPARSE_CM12_DIRTY_DOMAIN = Object.freeze({
  cellScalars: 0,
  cellVelocity: 1,
  scalarRows: 2,
  faceRows: 3,
  pressureCells: 4,
  pressureRows: 5,
  pressureEdges: 6,
  brickAggregates: 7,
  brickEdges: 8,
  hierarchyGroups: 9,
  hierarchyEdges: 10,
  topologyPages: 11,
  activityTiles: 12,
  velocityExtension: 13,
  presentationTiles: 14,
} as const);
export const SPARSE_CM12_DIRTY_DOMAIN_COUNT = 15;
export type SparseCM12DirtyDomain =
  typeof SPARSE_CM12_DIRTY_DOMAIN[keyof typeof SPARSE_CM12_DIRTY_DOMAIN];

export function sparseCM12DirtyDomainMask(...domains: readonly SparseCM12DirtyDomain[]): number {
  let mask = 0;
  for (const domain of domains) {
    if (!Number.isSafeInteger(domain) || domain < 0 || domain >= SPARSE_CM12_DIRTY_DOMAIN_COUNT) {
      throw new RangeError(`Invalid Sparse CM12 dirty domain ${domain}`);
    }
    mask |= 1 << domain;
  }
  return mask >>> 0;
}

export const SPARSE_CM12_DIRTY_EPOCH = Object.freeze({ pre: 0, post: 1 } as const);
export type SparseCM12DirtyEpoch = typeof SPARSE_CM12_DIRTY_EPOCH[keyof typeof SPARSE_CM12_DIRTY_EPOCH];

export const SPARSE_CM12_DIRTY_SCHEDULER_FLAG = Object.freeze({
  initialized: 1 << 0,
  preOpen: 1 << 1,
  preSealed: 1 << 2,
  postOpen: 1 << 3,
  complete: 1 << 4,
  accepted: 1 << 5,
  rejected: 1 << 6,
  fault: 1 << 7,
} as const);

export const SPARSE_CM12_DIRTY_PACKET_FLAG = Object.freeze({
  preSeen: 1 << 0,
  postSeen: 1 << 1,
  closureRequested: 1 << 2,
  closureResolved: 1 << 3,
  executed: 1 << 4,
  skipped: 1 << 5,
  coverageComplete: 1 << 6,
  fault: 1 << 7,
} as const);

export const SPARSE_CM12_DIRTY_FAULT = Object.freeze({
  none: 0,
  invalidHeader: 1,
  generationMismatch: 2,
  journalCapacity: 3,
  preQueueCapacity: 4,
  postQueueCapacity: 5,
  closureQueueCapacity: 6,
  packetInitializationRace: 7,
  missingCoverage: 8,
  unresolvedClosure: 9,
  invalidStableTile: 10,
  invalidStageOrDomain: 11,
  invalidEpochTransition: 12,
  generationExhausted: 13,
  provenanceGap: 14,
} as const);

export const SPARSE_CM12_DIRTY_SCHEDULER_HEADER = Object.freeze({
  magic: 0,
  version: 1,
  headerWords: 2,
  packetWords: 3,
  journalWords: 4,
  stageCount: 5,
  domainCount: 6,
  stableTileCount: 7,
  acceptedGeneration: 8,
  candidateGeneration: 9,
  preEpochGeneration: 10,
  postEpochGeneration: 11,
  flags: 12,
  faultCode: 13,
  firstFaultTile: 14,
  firstFaultStage: 15,
  firstFaultDomain: 16,
  journalCount: 17,
  journalCapacity: 18,
  expectedPreWrites: 19,
  coveredPreWrites: 20,
  expectedPostWrites: 21,
  coveredPostWrites: 22,
  closureRequestedCount: 23,
  closureResolvedCount: 24,
  maxClosureDepth: 25,
  publicationBase: 26,
  packetBase: 27,
  journalBase: 28,
  frontierABase: 29,
  frontierBBase: 30,
  totalWords: 31,
  initializedPacketCount: 32,
  finalizedPacketCount: 33,
  promotedPostCount: 34,
  reserved: 35,
} as const);

export const SPARSE_CM12_DIRTY_SCHEDULER_PACKET = Object.freeze({
  generation: 0,
  preStageMask: 1,
  postStageMask: 2,
  preDomainMask: 3,
  postDomainMask: 4,
  directStageMask: 5,
  closureStageMask: 6,
  originCauseMask: 7,
  inheritedCauseMask: 8,
  requestedClosureDepths: 9,
  resolvedClosureDepths: 10,
  executedStageMask: 11,
  skippedStageMask: 12,
  expectedCoverageStageMask: 13,
  coveredStageMask: 14,
  uncoveredStageMask: 15,
  preQueuedGeneration: 16,
  postQueuedGeneration: 17,
  firstJournal: 18,
  preEventCount: 19,
  postEventCount: 20,
  preCoverageCount: 21,
  postCoverageCount: 22,
  producerGeneration: 23,
  consumerGeneration: 24,
  flags: 25,
  faultCode: 26,
  packedOriginStages: 27,
  finalizedGeneration: 28,
} as const);

export const SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL = Object.freeze({
  stableTile: 0,
  epochAndFlags: 1,
  stageMask: 2,
  domainMask: 3,
  causeMask: 4,
  closureDepthsAndOrigin: 5,
  producerGeneration: 6,
  /** 0 not required, 1 expected, 2 covered; event slot itself is the receipt token. */
  receiptState: 7,
} as const);

/** Four vec3 indirect records: pre, closure A, closure B, and post. */
export const SPARSE_CM12_DIRTY_SCHEDULER_QUEUE = Object.freeze({
  pre: 0,
  closureA: 4,
  closureB: 8,
  post: 12,
  itemCount: 0,
  dispatchX: 1,
  dispatchY: 2,
  dispatchZ: 3,
} as const);

export interface SparseCM12DirtySchedulerLayoutOptions {
  readonly logicalBrickCount: number;
  readonly brickFineResolution: number;
  /** Must cover the maximum coalesced producer-event count. Overflow rejects the epoch. */
  readonly journalCapacity: number;
  /** Reserves the existing optional PKT1 pass-packing publication before CMS1. */
  readonly packingPacketCount?: number;
  /** Defaults to zero; permits placement inside a larger arena. */
  readonly baseWords?: number;
}

export interface SparseCM12DirtySchedulerLayout {
  readonly baseWords: number;
  readonly publicationBaseWords: number;
  readonly publicationWords: number;
  readonly schedulerBaseWords: number;
  readonly packetBaseWords: number;
  readonly journalBaseWords: number;
  readonly frontierABaseWords: number;
  readonly frontierBBaseWords: number;
  readonly queueHeaderBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
  readonly stableTileCount: number;
  readonly journalCapacity: number;
  readonly packingPacketCount: number;
  readonly logicalBrickCount: number;
  readonly brickFineResolution: number;
  readonly tilesPerLogicalBrickAxis: number;
  readonly dirtyProvenanceBinding: Readonly<{ offset: number; size: number }>;
}

function checkedNonNegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${label} must be a non-negative safe integer`);
  return value;
}

function checkedWords(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value > 0x3fff_ffff) {
    throw new RangeError(`${label} exceeds the addressable u32 GPU arena`);
  }
  return value;
}

function alignWords(value: number): number {
  return Math.ceil(value / SPARSE_CM12_DIRTY_SCHEDULER_ALIGNMENT_WORDS)
    * SPARSE_CM12_DIRTY_SCHEDULER_ALIGNMENT_WORDS;
}

export function createSparseCM12DirtySchedulerLayout(
  options: SparseCM12DirtySchedulerLayoutOptions,
): SparseCM12DirtySchedulerLayout {
  const baseWords = checkedNonNegativeInteger(options.baseWords ?? 0, "baseWords");
  if (baseWords % SPARSE_CM12_DIRTY_SCHEDULER_ALIGNMENT_WORDS !== 0) {
    throw new RangeError("Sparse CM12 dirty scheduler baseWords must be 256-byte aligned");
  }
  const logicalBrickCount = checkedNonNegativeInteger(options.logicalBrickCount, "logicalBrickCount");
  const journalCapacity = checkedNonNegativeInteger(options.journalCapacity, "journalCapacity");
  const packingPacketCount = checkedNonNegativeInteger(options.packingPacketCount ?? 0, "packingPacketCount");
  if (packingPacketCount > SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS) {
    throw new RangeError(`packingPacketCount must not exceed ${SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS}`);
  }
  const tilesPerLogicalBrickAxis = sparseCM12DirtyTilesPerLogicalBrickAxis(options.brickFineResolution);
  const stableTileCount = checkedWords(
    logicalBrickCount * tilesPerLogicalBrickAxis ** 3,
    "stableTileCount",
  );
  const publicationBaseWords = baseWords;
  const publicationWords = packingPacketCount === 0
    ? sparseCM12DirtyGPUBufferWords(logicalBrickCount, options.brickFineResolution)
    : sparseCM12DirtyPackingGPUBufferWords(
      logicalBrickCount,
      options.brickFineResolution,
      packingPacketCount,
    );
  const schedulerBaseWords = alignWords(publicationBaseWords + publicationWords);
  const packetBaseWords = schedulerBaseWords + SPARSE_CM12_DIRTY_SCHEDULER_HEADER_WORDS;
  const journalBaseWords = checkedWords(
    packetBaseWords + stableTileCount * SPARSE_CM12_DIRTY_SCHEDULER_PACKET_WORDS,
    "journalBaseWords",
  );
  const frontierABaseWords = checkedWords(
    journalBaseWords + journalCapacity * SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL_WORDS,
    "frontierABaseWords",
  );
  const frontierBBaseWords = checkedWords(frontierABaseWords + stableTileCount, "frontierBBaseWords");
  const queueHeaderBaseWords = checkedWords(frontierBBaseWords + stableTileCount, "queueHeaderBaseWords");
  const totalWords = checkedWords(
    queueHeaderBaseWords + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE_HEADER_WORDS,
    "totalWords",
  );
  return Object.freeze({
    baseWords,
    publicationBaseWords,
    publicationWords,
    schedulerBaseWords,
    packetBaseWords,
    journalBaseWords,
    frontierABaseWords,
    frontierBBaseWords,
    queueHeaderBaseWords,
    totalWords,
    totalBytes: totalWords * Uint32Array.BYTES_PER_ELEMENT,
    stableTileCount,
    journalCapacity,
    packingPacketCount,
    logicalBrickCount,
    brickFineResolution: options.brickFineResolution,
    tilesPerLogicalBrickAxis,
    dirtyProvenanceBinding: Object.freeze({
      offset: publicationBaseWords * Uint32Array.BYTES_PER_ELEMENT,
      size: publicationWords * Uint32Array.BYTES_PER_ELEMENT,
    }),
  });
}

export interface SparseCM12DirtySchedulerInitialization {
  readonly acceptedGeneration?: number;
  /** Only valid after the caller has independently certified the resident state. */
  readonly acceptedCleanBootstrap?: boolean;
}

export function createSparseCM12DirtySchedulerInitialWords(
  layout: SparseCM12DirtySchedulerLayout,
  initialization: SparseCM12DirtySchedulerInitialization = {},
): Uint32Array {
  const acceptedGeneration = checkedNonNegativeInteger(
    initialization.acceptedGeneration ?? 0,
    "acceptedGeneration",
  );
  if (acceptedGeneration >= 0x8000_0000) throw new RangeError("acceptedGeneration reserves bit 31 for packet initialization");
  const words = new Uint32Array(layout.totalWords);
  const publication = layout.publicationBaseWords;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.magic] = SPARSE_CM12_DIRTY_GPU_MAGIC;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.version] = SPARSE_CM12_DIRTY_GPU_VERSION;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.headerWords] = SPARSE_CM12_DIRTY_GPU_HEADER_WORDS;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.recordWords] = SPARSE_CM12_DIRTY_GPU_RECORD_WORDS;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.tileEdge] = SPARSE_CM12_DIRTY_TILE_EDGE;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.tilesPerLogicalBrickAxis] = layout.tilesPerLogicalBrickAxis;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.logicalBrickCount] = layout.logicalBrickCount;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.stageCount] = SPARSE_CM12_DIRTY_STAGE_COUNT;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.acceptedGeneration] = acceptedGeneration;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.candidateGeneration] = acceptedGeneration;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.provenanceGeneration] = acceptedGeneration;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.publicationFlags] = initialization.acceptedCleanBootstrap
    ? SPARSE_CM12_DIRTY_PUBLICATION_FLAG.complete | SPARSE_CM12_DIRTY_PUBLICATION_FLAG.accepted
    : SPARSE_CM12_DIRTY_PUBLICATION_FLAG.complete | SPARSE_CM12_DIRTY_PUBLICATION_FLAG.rejected;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.firstFaultLogicalBrick] = SPARSE_CM12_DIRTY_INVALID;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.firstFaultLocalTile] = SPARSE_CM12_DIRTY_INVALID;
  words[publication + SPARSE_CM12_DIRTY_GPU_HEADER.firstFaultStage] = SPARSE_CM12_DIRTY_INVALID;

  const allStages = (1 << SPARSE_CM12_DIRTY_STAGE_COUNT) - 1;
  const generationPair = sparseCM12DirtyGenerationPair(acceptedGeneration, acceptedGeneration);
  for (let tile = 0; tile < layout.stableTileCount; tile++) {
    const record = publication + SPARSE_CM12_DIRTY_GPU_HEADER_WORDS
      + tile * SPARSE_CM12_DIRTY_GPU_RECORD_WORDS;
    words[record + SPARSE_CM12_DIRTY_GPU_RECORD.skippedStageMask] =
      initialization.acceptedCleanBootstrap ? allStages : 0;
    words[record + SPARSE_CM12_DIRTY_GPU_RECORD.unknownStageMask] =
      initialization.acceptedCleanBootstrap ? 0 : allStages;
    words[record + SPARSE_CM12_DIRTY_GPU_RECORD.originCauseMask] =
      initialization.acceptedCleanBootstrap ? 0 : SPARSE_CM12_DIRTY_CAUSE_BIT.generationMismatch;
    for (let stage = 0; stage < SPARSE_CM12_DIRTY_STAGE_COUNT; stage++) {
      words[record + SPARSE_CM12_DIRTY_GPU_RECORD.stageGenerationPairs + stage] = generationPair;
    }
  }

  if (layout.packingPacketCount > 0) {
    const packing = publication + SPARSE_CM12_DIRTY_GPU_HEADER_WORDS
      + layout.stableTileCount * SPARSE_CM12_DIRTY_GPU_RECORD_WORDS;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.magic] = SPARSE_CM12_DIRTY_PACKING_GPU_MAGIC;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.version] = SPARSE_CM12_DIRTY_PACKING_GPU_VERSION;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.headerWords] =
      SPARSE_CM12_DIRTY_PACKING_GPU_HEADER_WORDS;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.packetWords] =
      SPARSE_CM12_DIRTY_PACKING_GPU_PACKET_WORDS;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.tileWords] =
      SPARSE_CM12_DIRTY_PACKING_GPU_TILE_WORDS;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.packetCount] = layout.packingPacketCount;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.tileCount] = layout.stableTileCount;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.acceptedGeneration] = acceptedGeneration;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.candidateGeneration] = acceptedGeneration;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.provenanceGeneration] = acceptedGeneration;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.publicationFlags] =
      initialization.acceptedCleanBootstrap
        ? SPARSE_CM12_DIRTY_PUBLICATION_FLAG.complete | SPARSE_CM12_DIRTY_PUBLICATION_FLAG.accepted
        : SPARSE_CM12_DIRTY_PUBLICATION_FLAG.complete | SPARSE_CM12_DIRTY_PUBLICATION_FLAG.rejected;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.firstFaultTile] = SPARSE_CM12_DIRTY_INVALID;
    words[packing + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER.firstFaultPacket] = SPARSE_CM12_DIRTY_INVALID;
  }

  const header = layout.schedulerBaseWords;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.magic] = SPARSE_CM12_DIRTY_SCHEDULER_MAGIC;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.version] = SPARSE_CM12_DIRTY_SCHEDULER_VERSION;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.headerWords] = SPARSE_CM12_DIRTY_SCHEDULER_HEADER_WORDS;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.packetWords] = SPARSE_CM12_DIRTY_SCHEDULER_PACKET_WORDS;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.journalWords] = SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL_WORDS;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.stageCount] = SPARSE_CM12_DIRTY_STAGE_COUNT;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.domainCount] = SPARSE_CM12_DIRTY_DOMAIN_COUNT;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.stableTileCount] = layout.stableTileCount;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.acceptedGeneration] = acceptedGeneration;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration] = acceptedGeneration;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.preEpochGeneration] = acceptedGeneration;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.postEpochGeneration] = acceptedGeneration;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags] = initialization.acceptedCleanBootstrap
    ? SPARSE_CM12_DIRTY_SCHEDULER_FLAG.initialized
      | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.complete
      | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.accepted
    : SPARSE_CM12_DIRTY_SCHEDULER_FLAG.initialized
      | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.complete
      | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.rejected;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultTile] = SPARSE_CM12_DIRTY_INVALID;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultStage] = SPARSE_CM12_DIRTY_INVALID;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultDomain] = SPARSE_CM12_DIRTY_INVALID;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.journalCapacity] = layout.journalCapacity;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.publicationBase] = layout.publicationBaseWords;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.packetBase] = layout.packetBaseWords;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.journalBase] = layout.journalBaseWords;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.frontierABase] = layout.frontierABaseWords;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.frontierBBase] = layout.frontierBBaseWords;
  words[header + SPARSE_CM12_DIRTY_SCHEDULER_HEADER.totalWords] = layout.totalWords;
  for (let tile = 0; tile < layout.stableTileCount; tile++) {
    const packet = layout.packetBaseWords + tile * SPARSE_CM12_DIRTY_SCHEDULER_PACKET_WORDS;
    words[packet + SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation] = acceptedGeneration;
    words[packet + SPARSE_CM12_DIRTY_SCHEDULER_PACKET.preQueuedGeneration] = SPARSE_CM12_DIRTY_INVALID;
    words[packet + SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postQueuedGeneration] = SPARSE_CM12_DIRTY_INVALID;
    words[packet + SPARSE_CM12_DIRTY_SCHEDULER_PACKET.firstJournal] = SPARSE_CM12_DIRTY_INVALID;
  }
  for (const queue of [
    SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.pre,
    SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureA,
    SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureB,
    SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.post,
  ]) {
    words[layout.queueHeaderBaseWords + queue + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchY] = 1;
    words[layout.queueHeaderBaseWords + queue + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchZ] = 1;
  }
  return words;
}

export function sparseCM12DirtyGenerationPair(producer: number, consumer: number): number {
  return ((producer & 0xffff) | ((consumer & 0xffff) << 16)) >>> 0;
}

export function sparseCM12DirtyPackClosureDepth(
  packed: number,
  stage: number,
  depth: number,
): number {
  if (!Number.isSafeInteger(stage) || stage < 0 || stage >= SPARSE_CM12_DIRTY_STAGE_COUNT) {
    throw new RangeError("Invalid Sparse CM12 dirty stage");
  }
  if (!Number.isSafeInteger(depth) || depth < 0 || depth > 15) {
    throw new RangeError("Sparse CM12 closure depth must fit in four bits");
  }
  const shift = stage * 4;
  return ((packed & ~(0xf << shift)) | (depth << shift)) >>> 0;
}

export function sparseCM12DirtyPackOriginStage(
  packed: number,
  stage: number,
  originStage: number,
): number {
  if (!Number.isSafeInteger(stage) || stage < 0 || stage >= SPARSE_CM12_DIRTY_STAGE_COUNT) {
    throw new RangeError("Invalid Sparse CM12 dirty stage");
  }
  if (!Number.isSafeInteger(originStage) || originStage < 0 || originStage > SPARSE_CM12_DIRTY_STAGE_COUNT) {
    throw new RangeError("Sparse CM12 origin stage must be external (0) or one-based stage (1..6)");
  }
  const shift = stage * 3;
  return ((packed & ~(0x7 << shift)) | (originStage << shift)) >>> 0;
}

export interface SparseCM12DirtyClosureReceipt {
  readonly stableTile: number;
  readonly stageMask: number;
  readonly requestedDepths: number;
  readonly resolvedDepths: number;
  readonly producerGeneration: number;
  readonly consumerGeneration: number;
  readonly expectedCoverageStageMask: number;
  readonly coveredStageMask: number;
  readonly uncoveredStageMask: number;
  readonly faultCode: number;
}
