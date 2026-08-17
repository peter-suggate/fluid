/** Deterministic, tile-resident Sparse CM12 activity authority (A4D2). */
export const SPARSE_CM12_PRODUCTION_ACTIVITY_MAGIC = 0x4134_4432; // A4D2
export const SPARSE_CM12_PRODUCTION_ACTIVITY_VERSION = 2;
export const SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER_WORDS = 64;
export const SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_WORDS = 16;
export const SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER_WORDS = 4;
export const SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_WORDS = 16;
export const SPARSE_CM12_PRODUCTION_ACTIVITY_CENSUS_BINS = 256;
export const SPARSE_CM12_PRODUCTION_ACTIVITY_COMPACTION_BLOCK = 256;
export const SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_EDGE = 4;
export const SPARSE_CM12_PRODUCTION_ACTIVITY_INVALID = 0xffff_ffff;

export const SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, flags: 3,
  acceptedGeneration: 4, candidateGeneration: 5, topologyGeneration: 6,
  framePlanGeneration: 7, brickCapacity: 8, tilesPerBrick: 9,
  compactionBlockCount: 10, dirtyBrickCount: 11, dirtyTileCount: 12,
  rebuiltBrickCount: 13, rebuiltTileCount: 14, reusedActiveBrickCount: 15,
  fplReceiptCount: 16, producerGeneration: 17, consumerGeneration: 18,
  faultCount: 19, firstFaultCode: 20, firstFaultBrick: 21, firstFaultTile: 22,
  maximumClosureDepth: 23, activeBrickCount: 24, surfaceBrickCount: 25,
  hotBrickCount: 26, quietBrickCount: 27, maximumScore: 28,
  triggerBase: 29, tileSummaryBase: 30, brickBase: 31,
  dirtyFlagBase: 32, dirtyPrefixBase: 33, dirtyListBase: 34,
  blockSumBase: 35, blockPrefixBase: 36, censusBlockBase: 37,
  histogramBase: 38, totalWords: 39,
  rebuildIndirectX: 40, rebuildIndirectY: 41, rebuildIndirectZ: 42,
  censusIndirectX: 43, censusIndirectY: 44, censusIndirectZ: 45,
  expectedTriggerCount: 46, classifiedTriggerCount: 47,
  expectedFplReceiptCount: 48, uncoveredWriteCount: 49,
  blockTileSumBase: 50,
  candidateListBase: 51, candidateBrickCount: 52,
  candidateListGeneration: 53,
  classifyIndirectX: 54, classifyIndirectY: 55, classifyIndirectZ: 56,
  scanIndirectX: 57, scanIndirectY: 58, scanIndirectZ: 59,
} as const);

export const SPARSE_CM12_PRODUCTION_ACTIVITY_TILE = Object.freeze({
  generation: 0, topologySignature: 1,
  densityFixed: 2, momentXFixed: 3, momentYFixed: 4, momentZFixed: 5,
  maximumDeformation: 6, maximumPredictedMotion: 7,
  maximumDetailError: 8, maximumVelocityTravel: 9,
  flags: 10, supportMask: 11, sweptSupportMask: 12,
  contributionCount: 13, packedCauseDepth: 14, check: 15,
} as const);

export const SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER = Object.freeze({
  generation: 0, causeMask: 1,
  /** Stages/depth in low 16 bits; exact decision certificate in high 16. */
  packedStagesAndDepth: 2, topologySignature: 3,
} as const);

export const SPARSE_CM12_PRODUCTION_ACTIVITY_CERTIFICATE = Object.freeze({
  momentsExact: 1 << 16, metricScoreClassExact: 1 << 17,
  featureThresholdsExact: 1 << 18, reasonPredicatesExact: 1 << 19,
  supportExact: 1 << 20, topologyExact: 1 << 21,
  policyGenerationExact: 1 << 22, valid: 0x8000_0000,
} as const);
export const SPARSE_CM12_PRODUCTION_ACTIVITY_REQUIRED_CERTIFICATE = (
  Object.values(SPARSE_CM12_PRODUCTION_ACTIVITY_CERTIFICATE)
    .reduce((mask, bit) => mask | bit, 0) >>> 0
);

export const SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK = Object.freeze({
  generation: 0, topologySignature: 1, dirtyMaskLow: 2, dirtyMaskHigh: 3,
  score: 4, reasons: 5, history: 6, flags: 7,
  previousScore: 8, previousReasons: 9, previousFlags: 10,
  causeMask: 11, maximumClosureDepth: 12, fplReceiptMask: 13,
  producerGeneration: 14, consumerGeneration: 15,
} as const);

export const SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG = Object.freeze({
  initialized: 1 << 0, collecting: 1 << 1, compacted: 1 << 2,
  rebuilt: 1 << 3, censusCommitted: 1 << 4, accepted: 1 << 5,
  fault: 1 << 6,
} as const);

export const SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_FLAG = Object.freeze({
  active: 1 << 0, surface: 1 << 1, hot: 1 << 2, quiet: 1 << 3,
  topologyChanged: 1 << 4, rebuilt: 1 << 5, coverageComplete: 1 << 6,
} as const);

export const SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, generationMismatch: 2, triggerMissing: 3,
  topologyMismatch: 4, compactionOverflow: 5, prefixMismatch: 6,
  staleTile: 7, contributionMismatch: 8, censusUnderflow: 9,
  censusMismatch: 10, fplCoverage: 11, generationExhausted: 12,
} as const);

export interface SparseCM12ProductionActivityLayout {
  readonly baseWords: number;
  readonly triggerBaseWords: number;
  readonly candidateListBaseWords: number;
  readonly tileSummaryBaseWords: number;
  readonly brickBaseWords: number;
  readonly dirtyFlagBaseWords: number;
  readonly dirtyPrefixBaseWords: number;
  readonly dirtyListBaseWords: number;
  readonly blockSumBaseWords: number;
  readonly blockTileSumBaseWords: number;
  readonly blockPrefixBaseWords: number;
  readonly censusBlockBaseWords: number;
  readonly histogramBaseWords: number;
  readonly totalWords: number;
  readonly brickCapacity: number;
  readonly brickFineResolution: 4 | 8 | 16;
  readonly tilesPerBrick: 1 | 8 | 64;
  readonly compactionBlockCount: number;
}

const checked = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0)) {
    throw new RangeError(`${label} must be ${positive ? "positive" : "non-negative"}`);
  }
  return value;
};
const align = (value: number): number => Math.ceil(value / 64) * 64;

export function sparseCM12ProductionActivityValidTileMask(
  brickFineResolution: 4 | 8 | 16,
): readonly [number, number] {
  const count = (brickFineResolution / SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_EDGE) ** 3;
  if (count < 32) return [(2 ** count - 1) >>> 0, 0] as const;
  return [0xffff_ffff, count === 64 ? 0xffff_ffff : 0] as const;
}

export function createSparseCM12ProductionActivityLayout(options: {
  readonly baseWords?: number;
  readonly brickCapacity: number;
  readonly brickFineResolution?: 4 | 8 | 16;
}): SparseCM12ProductionActivityLayout {
  const baseWords = checked(options.baseWords ?? 0, "baseWords");
  if (baseWords % 64 !== 0) throw new RangeError("A4D2 baseWords must be 256-byte aligned");
  const brickCapacity = checked(options.brickCapacity, "brickCapacity", true);
  if (brickCapacity > 65_536) throw new RangeError("A4D2 supports at most 65536 bricks");
  const brickFineResolution = options.brickFineResolution ?? 16;
  const tilesPerBrick = ((brickFineResolution / 4) ** 3) as 1 | 8 | 64;
  const tileCapacity = brickCapacity * tilesPerBrick;
  const compactionBlockCount = Math.ceil(brickCapacity
    / SPARSE_CM12_PRODUCTION_ACTIVITY_COMPACTION_BLOCK);
  const candidateListBaseWords = align(baseWords
    + SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER_WORDS);
  const triggerBaseWords = align(candidateListBaseWords + brickCapacity);
  const tileSummaryBaseWords = align(triggerBaseWords
    + tileCapacity * SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER_WORDS);
  const brickBaseWords = align(tileSummaryBaseWords
    + tileCapacity * SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_WORDS);
  const dirtyFlagBaseWords = align(brickBaseWords
    + brickCapacity * SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_WORDS);
  const dirtyPrefixBaseWords = align(dirtyFlagBaseWords + brickCapacity);
  const dirtyListBaseWords = align(dirtyPrefixBaseWords + brickCapacity);
  const blockSumBaseWords = align(dirtyListBaseWords + brickCapacity);
  const blockTileSumBaseWords = align(blockSumBaseWords + compactionBlockCount);
  const blockPrefixBaseWords = align(blockTileSumBaseWords + compactionBlockCount);
  // 256 signed histogram deltas plus surface/hot/quiet/active deltas per block.
  const censusBlockBaseWords = align(blockPrefixBaseWords + compactionBlockCount);
  const histogramBaseWords = align(censusBlockBaseWords
    + compactionBlockCount * 260);
  const totalWords = align(histogramBaseWords
    + SPARSE_CM12_PRODUCTION_ACTIVITY_CENSUS_BINS);
  return Object.freeze({ baseWords, candidateListBaseWords, triggerBaseWords, tileSummaryBaseWords,
    brickBaseWords, dirtyFlagBaseWords, dirtyPrefixBaseWords, dirtyListBaseWords,
    blockSumBaseWords, blockTileSumBaseWords, blockPrefixBaseWords, censusBlockBaseWords,
    histogramBaseWords, totalWords, brickCapacity, brickFineResolution,
    tilesPerBrick, compactionBlockCount });
}

export function createSparseCM12ProductionActivityInitialWords(
  layout: SparseCM12ProductionActivityLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER;
  words[h.magic] = SPARSE_CM12_PRODUCTION_ACTIVITY_MAGIC;
  words[h.version] = SPARSE_CM12_PRODUCTION_ACTIVITY_VERSION;
  words[h.headerWords] = SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER_WORDS;
  words[h.flags] = SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.initialized;
  words[h.brickCapacity] = layout.brickCapacity;
  words[h.tilesPerBrick] = layout.tilesPerBrick;
  words[h.compactionBlockCount] = layout.compactionBlockCount;
  words[h.firstFaultBrick] = SPARSE_CM12_PRODUCTION_ACTIVITY_INVALID;
  words[h.firstFaultTile] = SPARSE_CM12_PRODUCTION_ACTIVITY_INVALID;
  words[h.candidateListBase] = layout.candidateListBaseWords;
  words[h.triggerBase] = layout.triggerBaseWords;
  words[h.tileSummaryBase] = layout.tileSummaryBaseWords;
  words[h.brickBase] = layout.brickBaseWords;
  words[h.dirtyFlagBase] = layout.dirtyFlagBaseWords;
  words[h.dirtyPrefixBase] = layout.dirtyPrefixBaseWords;
  words[h.dirtyListBase] = layout.dirtyListBaseWords;
  words[h.blockSumBase] = layout.blockSumBaseWords;
  words[h.blockTileSumBase] = layout.blockTileSumBaseWords;
  words[h.blockPrefixBase] = layout.blockPrefixBaseWords;
  words[h.censusBlockBase] = layout.censusBlockBaseWords;
  words[h.histogramBase] = layout.histogramBaseWords;
  words[h.totalWords] = layout.totalWords;
  words[h.rebuildIndirectY] = 1;
  words[h.rebuildIndirectZ] = 1;
  words[h.censusIndirectY] = 1;
  words[h.censusIndirectZ] = 1;
  words[h.classifyIndirectY] = 1;
  words[h.classifyIndirectZ] = 1;
  words[h.scanIndirectY] = 1;
  words[h.scanIndirectZ] = 1;
  return words;
}

export function sparseCM12ProductionActivityCompact(
  dirty: readonly boolean[],
): { readonly prefix: readonly number[]; readonly list: readonly number[] } {
  const prefix: number[] = []; const list: number[] = []; let count = 0;
  dirty.forEach((value, brick) => {
    prefix.push(count);
    if (value) { list.push(brick); count += 1; }
  });
  return Object.freeze({ prefix: Object.freeze(prefix), list: Object.freeze(list) });
}

export function sparseCM12ProductionActivityCompactCandidates(
  candidates: readonly number[], dirty: readonly boolean[],
): { readonly prefix: readonly number[]; readonly list: readonly number[] } {
  if (candidates.length !== dirty.length) throw new RangeError("candidate/dirty length mismatch");
  candidates.forEach((brick, index) => {
    checked(brick, "candidate brick");
    if (index > 0 && brick <= candidates[index - 1]!) {
      throw new RangeError("candidate bricks must be strictly increasing");
    }
  });
  const compact = sparseCM12ProductionActivityCompact(dirty);
  return Object.freeze({ prefix: compact.prefix,
    list: Object.freeze(compact.list.map((index) => candidates[index]!)) });
}
