/** GPU-resident, generation-stamped activity summary worklist. */
export interface SparseCM12IncrementalActivityLayout {
  readonly headerBaseWords: number;
  readonly stableTileStampBaseWords: number;
  readonly stableTileCauseBaseWords: number;
  readonly brickStampBaseWords: number;
  readonly brickListBaseWords: number;
  readonly brickTopologyStateBaseWords: number;
  readonly brickCensusStateBaseWords: number;
  readonly brickTileMaskLowBaseWords: number;
  readonly brickTileMaskHighBaseWords: number;
  readonly scoreHistogramBaseWords: number;
  readonly stableTileCount: number;
  readonly brickCount: number;
  readonly totalWords: number;
}

export const SPARSE_CM12_INCREMENTAL_ACTIVITY_MAGIC = 0x41435431; // ACT1
export const SPARSE_CM12_INCREMENTAL_ACTIVITY_VERSION = 1;
export const SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER_WORDS = 16;

export const SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER = Object.freeze({
  magic: 0,
  version: 1,
  headerWords: 2,
  flags: 3,
  generation: 4,
  dirtyTileCount: 5,
  dirtyBrickCount: 6,
  uncoveredWriteFaultCount: 7,
  dispatchX: 8,
  dispatchY: 9,
  dispatchZ: 10,
  stableTileCount: 11,
  brickCount: 12,
  measuredBrickCount: 13,
  reusedBrickCount: 14,
  reserved: 15,
} as const);

const checked = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
  return value;
};

export function createSparseCM12IncrementalActivityLayout(options: {
  readonly baseWords: number;
  readonly stableTileCount: number;
  readonly brickCount: number;
  readonly alignmentWords?: number;
}): SparseCM12IncrementalActivityLayout {
  const alignment = checked(options.alignmentWords ?? 64, "alignmentWords");
  if (alignment === 0) throw new RangeError("alignmentWords must be positive");
  const base = checked(options.baseWords, "baseWords");
  const stableTileCount = checked(options.stableTileCount, "stableTileCount");
  const brickCount = checked(options.brickCount, "brickCount");
  const headerBaseWords = Math.ceil(base / alignment) * alignment;
  const stableTileStampBaseWords = headerBaseWords
    + SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER_WORDS;
  const stableTileCauseBaseWords = stableTileStampBaseWords + stableTileCount;
  const brickStampBaseWords = stableTileCauseBaseWords + stableTileCount;
  const brickListBaseWords = brickStampBaseWords + brickCount;
  const brickTopologyStateBaseWords = brickListBaseWords + brickCount;
  const brickCensusStateBaseWords = brickTopologyStateBaseWords + brickCount;
  const brickTileMaskLowBaseWords = brickCensusStateBaseWords + brickCount;
  const brickTileMaskHighBaseWords = brickTileMaskLowBaseWords + brickCount;
  const scoreHistogramBaseWords = brickTileMaskHighBaseWords + brickCount;
  return Object.freeze({
    headerBaseWords,
    stableTileStampBaseWords,
    stableTileCauseBaseWords,
    brickStampBaseWords,
    brickListBaseWords,
    brickTopologyStateBaseWords,
    brickCensusStateBaseWords,
    brickTileMaskLowBaseWords,
    brickTileMaskHighBaseWords,
    scoreHistogramBaseWords,
    stableTileCount,
    brickCount,
    totalWords: scoreHistogramBaseWords + 256,
  });
}

export function createSparseCM12IncrementalActivityInitialWords(
  layout: SparseCM12IncrementalActivityLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.headerBaseWords);
  const h = SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER;
  words[h.magic] = SPARSE_CM12_INCREMENTAL_ACTIVITY_MAGIC;
  words[h.version] = SPARSE_CM12_INCREMENTAL_ACTIVITY_VERSION;
  words[h.headerWords] = SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER_WORDS;
  words[h.dispatchY] = 1;
  words[h.dispatchZ] = 1;
  words[h.stableTileCount] = layout.stableTileCount;
  words[h.brickCount] = layout.brickCount;
  // Topology state 0xffffffff guarantees a bounded first-frame bootstrap.
  words.fill(0xffff_ffff,
    layout.brickTopologyStateBaseWords - layout.headerBaseWords,
    layout.brickTopologyStateBaseWords - layout.headerBaseWords + layout.brickCount);
  return words;
}
