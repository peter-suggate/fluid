/** Construction-only census of actual FPA xyz reads from the accepted VEX accessor. */
export const SPARSE_CM12_FPA_VEX_READ_CENSUS_MAGIC = 0x4656_5231; // FVR1
export const SPARSE_CM12_FPA_VEX_READ_CENSUS_VERSION = 1;
export const SPARSE_CM12_FPA_VEX_READ_CENSUS_MAX_TILES_PER_ROW = 32;
export const SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, generation: 3, fault: 4,
  rowCapacity: 5, cellCapacity: 6, tileCapacity: 7, maximumTilesPerRow: 8,
  changedEffectiveCellCount: 9, changedEffectiveTileCount: 10,
  uniqueDonorTileCount: 11, actualXyzReadCount: 12, donorEdgeCount: 13,
  maximumRowTileCount: 14, firstOverflowRow: 15,
  oracleChangedRowCount: 16, oracleMismatchRowCount: 17,
  firstOracleMismatchRow: 18, predictedScheduledRowCount: 19,
  acceptedParity: 20, acceptedGeneration: 21, candidateGeneration: 22,
  omittedChangedRowCount: 23, firstOmittedChangedRow: 24,
  acceptedTopologyGeneration: 25, candidateTopologyGeneration: 26,
  recording: 27,
  pairHash0: 28, pairHash1: 29, pairHash2: 30, pairHash3: 31,
  constructionBootstrapPublished: 32,
  liveRowCount: 33,
  topologyRebuildPublished: 34,
} as const);
export const SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER_WORDS = 40;

export interface SparseCM12FpaVexReadCensusLayout {
  readonly baseWords: number; readonly headerBaseWords: number;
  readonly rowTileHistogramBaseWords: number; readonly tileFanoutBaseWords: number;
  readonly compactSummaryWords: number; readonly compactSummaryBytes: number;
  readonly changedCellBitsBaseWords: number; readonly changedTileBitsBaseWords: number;
  readonly donorTileBitsBaseWords: number; readonly predictedRowBitsBaseWords: number;
  readonly directChangedCellBitsBaseWords: number;
  readonly sourceFaceChangedRowBitsBaseWords: number;
  readonly oracleChangedRowBitsBaseWords: number;
  readonly xyzDonorCellBitsBaseWords: readonly [number, number];
  readonly rowTileCountBaseWords: readonly [number, number];
  readonly rowTilesBaseWords: readonly [number, number];
  readonly velocityBitsBaseWords: readonly [number, number]; readonly priorFaceBitsBaseWords: number;
  readonly densityClassBaseWords: readonly [number, number];
  readonly validityClassBaseWords: readonly [number, number];
  readonly sourceFaceBitsBaseWords: readonly [number, number];
  readonly priorAcceptedAuthorityBitsBaseWords: number;
  readonly oracleFaceBitsBaseWords: number;
  readonly cellBitWords: number; readonly tileBitWords: number; readonly rowBitWords: number;
  readonly cellCapacity: number; readonly rowCapacity: number; readonly tileCapacity: number;
  readonly maximumTilesPerRow: number; readonly totalWords: number; readonly totalBytes: number;
}

const checked = (value: number, name: string) => {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${name} is invalid`);
  return value;
};
export function createSparseCM12FpaVexReadCensusLayout(options: {
  readonly baseWords: number; readonly cellCapacity: number;
  readonly rowCapacity: number; readonly tileCapacity: number;
  readonly maximumTilesPerRow?: number; readonly alignmentWords?: number;
}): SparseCM12FpaVexReadCensusLayout {
  const alignment = checked(options.alignmentWords ?? 64, "alignmentWords");
  if (alignment === 0) throw new RangeError("alignmentWords must be positive");
  const baseWords = Math.ceil(checked(options.baseWords, "baseWords") / alignment) * alignment;
  const cellCapacity = checked(options.cellCapacity, "cellCapacity");
  const rowCapacity = checked(options.rowCapacity, "rowCapacity");
  const tileCapacity = checked(options.tileCapacity, "tileCapacity");
  const maximumTilesPerRow = checked(options.maximumTilesPerRow
    ?? SPARSE_CM12_FPA_VEX_READ_CENSUS_MAX_TILES_PER_ROW, "maximumTilesPerRow");
  if (maximumTilesPerRow === 0) throw new RangeError("maximumTilesPerRow must be positive");
  const cellBitWords = Math.ceil(cellCapacity / 32);
  const tileBitWords = Math.ceil(tileCapacity / 32);
  const rowBitWords = Math.ceil(rowCapacity / 32);
  const headerBaseWords = baseWords;
  const rowTileHistogramBaseWords = headerBaseWords
    + SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER_WORDS;
  const tileFanoutBaseWords = rowTileHistogramBaseWords + maximumTilesPerRow + 1;
  const changedCellBitsBaseWords = tileFanoutBaseWords + tileCapacity;
  const changedTileBitsBaseWords = changedCellBitsBaseWords + cellBitWords;
  const donorTileBitsBaseWords = changedTileBitsBaseWords + tileBitWords;
  const predictedRowBitsBaseWords = donorTileBitsBaseWords + tileBitWords;
  const directChangedCellBitsBaseWords = predictedRowBitsBaseWords + rowBitWords;
  const sourceFaceChangedRowBitsBaseWords = directChangedCellBitsBaseWords + cellBitWords;
  const oracleChangedRowBitsBaseWords = sourceFaceChangedRowBitsBaseWords + rowBitWords;
  const xyzDonorCellBitsA = oracleChangedRowBitsBaseWords + rowBitWords;
  const xyzDonorCellBitsB = xyzDonorCellBitsA + cellBitWords;
  const rowTileCountA = xyzDonorCellBitsB + cellBitWords;
  const rowTileCountB = rowTileCountA + rowCapacity;
  const rowTilesA = rowTileCountB + rowCapacity;
  const rowTilesB = rowTilesA + rowCapacity * maximumTilesPerRow;
  const velocityA = rowTilesB
    + rowCapacity * maximumTilesPerRow;
  const velocityB = velocityA + 3 * cellCapacity;
  const densityA = velocityB + 3 * cellCapacity;
  const densityB = densityA + cellCapacity;
  const validityA = densityB + cellCapacity;
  const validityB = validityA + cellCapacity;
  const sourceFaceA = validityB + cellCapacity;
  const sourceFaceB = sourceFaceA + rowCapacity;
  const priorAcceptedAuthorityBitsBaseWords = sourceFaceB + rowCapacity;
  const priorFaceBitsBaseWords = priorAcceptedAuthorityBitsBaseWords + rowCapacity;
  const oracleFaceBitsBaseWords = priorFaceBitsBaseWords + rowCapacity;
  const totalWords = oracleFaceBitsBaseWords + rowCapacity;
  return Object.freeze({ baseWords, headerBaseWords, changedCellBitsBaseWords,
    rowTileHistogramBaseWords, tileFanoutBaseWords,
    compactSummaryWords: changedCellBitsBaseWords - baseWords,
    compactSummaryBytes: 4 * (changedCellBitsBaseWords - baseWords),
    changedTileBitsBaseWords, donorTileBitsBaseWords, predictedRowBitsBaseWords,
    directChangedCellBitsBaseWords, sourceFaceChangedRowBitsBaseWords,
    oracleChangedRowBitsBaseWords,
    xyzDonorCellBitsBaseWords: [xyzDonorCellBitsA, xyzDonorCellBitsB] as const,
    rowTileCountBaseWords: [rowTileCountA, rowTileCountB] as const,
    rowTilesBaseWords: [rowTilesA, rowTilesB] as const,
    velocityBitsBaseWords: [velocityA, velocityB] as const,
    densityClassBaseWords: [densityA, densityB] as const,
    validityClassBaseWords: [validityA, validityB] as const,
    sourceFaceBitsBaseWords: [sourceFaceA, sourceFaceB] as const,
    priorAcceptedAuthorityBitsBaseWords, priorFaceBitsBaseWords,
    oracleFaceBitsBaseWords, cellBitWords, tileBitWords,
    rowBitWords, cellCapacity, rowCapacity, tileCapacity, maximumTilesPerRow,
    totalWords, totalBytes: 4 * totalWords });
}

export function createSparseCM12FpaVexReadCensusInitialWords(
  layout: SparseCM12FpaVexReadCensusLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER;
  words[h.magic] = SPARSE_CM12_FPA_VEX_READ_CENSUS_MAGIC;
  words[h.version] = SPARSE_CM12_FPA_VEX_READ_CENSUS_VERSION;
  words[h.headerWords] = SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER_WORDS;
  words[h.rowCapacity] = layout.rowCapacity; words[h.cellCapacity] = layout.cellCapacity;
  words[h.tileCapacity] = layout.tileCapacity;
  words[h.maximumTilesPerRow] = layout.maximumTilesPerRow;
  words[h.firstOverflowRow] = 0xffff_ffff;
  words[h.firstOracleMismatchRow] = 0xffff_ffff;
  words[h.firstOmittedChangedRow] = 0xffff_ffff;
  words.fill(0x7fc0_0001,
    layout.velocityBitsBaseWords[0] - layout.baseWords,
    layout.densityClassBaseWords[0] - layout.baseWords);
  words.fill(0xffff_ffff,
    layout.densityClassBaseWords[0] - layout.baseWords,
    layout.priorFaceBitsBaseWords - layout.baseWords);
  return words;
}

export interface SparseCM12FpaVexReadCensusQA {
  readonly generation: number; readonly fault: number;
  readonly changedEffectiveCellCount: number; readonly changedEffectiveTiles: readonly number[];
  readonly donorTiles: readonly number[]; readonly actualXyzReadCount: number;
  readonly donorEdgeCount: number; readonly maximumRowTileCount: number;
  readonly firstOverflowRow?: number; readonly oracleChangedRows: readonly number[];
  readonly oracleMismatchRows: readonly number[]; readonly predictedScheduledRows: readonly number[];
  readonly omittedChangedRowCount: number; readonly firstOmittedChangedRow?: number;
  readonly acceptedParity: number; readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly rowTiles: readonly (readonly number[])[];
}
export interface SparseCM12FpaVexReadCensusSummaryQA {
  readonly generation: number; readonly fault: number;
  readonly changedEffectiveCellCount: number; readonly changedEffectiveTileCount: number;
  readonly uniqueDonorTileCount: number; readonly actualXyzReadCount: number;
  readonly donorEdgeCount: number; readonly maximumRowTileCount: number;
  readonly rowTileHistogram: readonly number[]; readonly tileFanoutHistogram: readonly number[];
  readonly rowTileQuantiles: Readonly<{ p50: number; p90: number; p99: number; maximum: number }>;
  readonly tileFanoutQuantiles: Readonly<{ p50: number; p90: number; p99: number; maximum: number }>;
  readonly pairHash: readonly [number, number, number, number];
  readonly oracleChangedRowCount: number; readonly oracleMismatchRowCount: number;
  readonly predictedScheduledRowCount: number; readonly omittedChangedRowCount: number;
  readonly acceptedGeneration: number; readonly candidateGeneration: number;
  readonly constructionBootstrapPublished: boolean;
  readonly topologyRebuildPublished: boolean;
  readonly liveRowCount: number; readonly compactSummaryBytes: number;
  readonly totalQAArenaBytes: number;
}
const quantile = (sorted: readonly number[], fraction: number) => sorted.length === 0
  ? 0 : sorted[Math.floor((sorted.length - 1) * fraction)]!;
const histogramQuantile = (histogram: readonly number[], fraction: number) => {
  const total = histogram.reduce((sum, count) => sum + count, 0);
  if (total === 0) return 0;
  const target = Math.floor((total - 1) * fraction); let cumulative = 0;
  for (let value = 0; value < histogram.length; value += 1) {
    cumulative += histogram[value]!; if (cumulative > target) return value;
  }
  return histogram.length - 1;
};
export function inspectSparseCM12FpaVexReadCensusSummaryQA(words: Uint32Array,
  layout: SparseCM12FpaVexReadCensusLayout): SparseCM12FpaVexReadCensusSummaryQA {
  if (words.length < layout.compactSummaryWords) throw new RangeError("FVR1 summary truncated");
  const h = SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER;
  if (words[h.magic] !== SPARSE_CM12_FPA_VEX_READ_CENSUS_MAGIC
    || words[h.version] !== SPARSE_CM12_FPA_VEX_READ_CENSUS_VERSION
    || words[h.headerWords] !== SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER_WORDS) {
    throw new Error("FVR1 summary header is invalid");
  }
  const histogramAt = layout.rowTileHistogramBaseWords - layout.baseWords;
  const rowTileHistogram = Object.freeze(Array.from(words.slice(histogramAt,
    histogramAt + layout.maximumTilesPerRow + 1)));
  const fanoutAt = layout.tileFanoutBaseWords - layout.baseWords;
  const fanouts = Array.from(words.slice(fanoutAt, fanoutAt + layout.tileCapacity))
    .filter((count) => count > 0).sort((a, b) => a - b);
  const fanoutMaximum = fanouts.at(-1) ?? 0;
  const tileFanoutHistogram = new Array(fanoutMaximum + 1).fill(0) as number[];
  for (const fanout of fanouts) tileFanoutHistogram[fanout] += 1;
  const rowMaximum = words[h.maximumRowTileCount]!;
  const liveRowCount = words[h.liveRowCount]!;
  const histogramSum = rowTileHistogram.reduce((sum, count) => sum + count, 0);
  if (histogramSum !== liveRowCount) throw new Error("FVR1 row histogram is incomplete");
  const fanoutSum = fanouts.reduce((sum, count) => sum + count, 0);
  if (fanoutSum !== words[h.donorEdgeCount]) throw new Error("FVR1 tile fanout is incomplete");
  return Object.freeze({ generation: words[h.generation]!, fault: words[h.fault]!,
    changedEffectiveCellCount: words[h.changedEffectiveCellCount]!,
    changedEffectiveTileCount: words[h.changedEffectiveTileCount]!,
    uniqueDonorTileCount: words[h.uniqueDonorTileCount]!,
    actualXyzReadCount: words[h.actualXyzReadCount]!, donorEdgeCount: words[h.donorEdgeCount]!,
    maximumRowTileCount: rowMaximum, rowTileHistogram,
    tileFanoutHistogram: Object.freeze(tileFanoutHistogram),
    rowTileQuantiles: Object.freeze({ p50: histogramQuantile(rowTileHistogram, 0.5),
      p90: histogramQuantile(rowTileHistogram, 0.9),
      p99: histogramQuantile(rowTileHistogram, 0.99), maximum: rowMaximum }),
    tileFanoutQuantiles: Object.freeze({ p50: quantile(fanouts, 0.5),
      p90: quantile(fanouts, 0.9), p99: quantile(fanouts, 0.99), maximum: fanoutMaximum }),
    pairHash: [words[h.pairHash0]!, words[h.pairHash1]!, words[h.pairHash2]!,
      words[h.pairHash3]!] as const,
    oracleChangedRowCount: words[h.oracleChangedRowCount]!,
    oracleMismatchRowCount: words[h.oracleMismatchRowCount]!,
    predictedScheduledRowCount: words[h.predictedScheduledRowCount]!,
    omittedChangedRowCount: words[h.omittedChangedRowCount]!,
    acceptedGeneration: words[h.acceptedGeneration]!,
    candidateGeneration: words[h.candidateGeneration]!,
    constructionBootstrapPublished: words[h.constructionBootstrapPublished] !== 0,
    topologyRebuildPublished: words[h.topologyRebuildPublished] !== 0,
    liveRowCount, compactSummaryBytes: layout.compactSummaryBytes,
    totalQAArenaBytes: layout.totalBytes - 4 * layout.baseWords,
  });
}
const bitIds = (words: Uint32Array, base: number, count: number) => {
  const ids: number[] = [];
  for (let id = 0; id < count; id += 1) {
    if ((words[base + (id >>> 5)]! & (1 << (id & 31))) !== 0) ids.push(id);
  }
  return Object.freeze(ids);
};
export function inspectSparseCM12FpaVexReadCensusQA(words: Uint32Array,
  layout: SparseCM12FpaVexReadCensusLayout): SparseCM12FpaVexReadCensusQA {
  if (words.length < layout.totalWords - layout.baseWords) throw new RangeError("FVR1 truncated");
  const h = SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER;
  if (words[h.magic] !== SPARSE_CM12_FPA_VEX_READ_CENSUS_MAGIC
    || words[h.version] !== SPARSE_CM12_FPA_VEX_READ_CENSUS_VERSION
    || words[h.headerWords] !== SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER_WORDS) {
    throw new Error("FVR1 header is invalid");
  }
  const relative = (absolute: number) => absolute - layout.baseWords;
  const acceptedParity = words[h.acceptedParity]! & 1;
  const rowTiles = Array.from({ length: layout.rowCapacity }, (_, row) => {
    const count = Math.min(words[relative(layout.rowTileCountBaseWords[acceptedParity]!) + row]!,
      layout.maximumTilesPerRow);
    return Object.freeze(Array.from(words.subarray(
      relative(layout.rowTilesBaseWords[acceptedParity]!) + row * layout.maximumTilesPerRow,
      relative(layout.rowTilesBaseWords[acceptedParity]!)
        + row * layout.maximumTilesPerRow + count))
      .sort((a, b) => a - b));
  });
  const oracleChangedRows: number[] = [], oracleMismatchRows: number[] = [];
  oracleChangedRows.push(...bitIds(words,
    relative(layout.oracleChangedRowBitsBaseWords), layout.rowCapacity));
  const oracle = relative(layout.oracleFaceBitsBaseWords);
  // The shader publishes exact mismatch ids only as a first witness/count;
  // expose that bounded witness without manufacturing an incomplete list.
  const firstMismatch = words[h.firstOracleMismatchRow]!;
  if (firstMismatch !== 0xffff_ffff) oracleMismatchRows.push(firstMismatch);
  const overflow = words[h.firstOverflowRow]!;
  const omitted = words[h.firstOmittedChangedRow]!;
  return Object.freeze({ generation: words[h.generation]!, fault: words[h.fault]!,
    changedEffectiveCellCount: words[h.changedEffectiveCellCount]!,
    changedEffectiveTiles: bitIds(words, relative(layout.changedTileBitsBaseWords),
      layout.tileCapacity),
    donorTiles: bitIds(words, relative(layout.donorTileBitsBaseWords), layout.tileCapacity),
    actualXyzReadCount: words[h.actualXyzReadCount]!,
    donorEdgeCount: words[h.donorEdgeCount]!,
    maximumRowTileCount: words[h.maximumRowTileCount]!,
    ...(overflow === 0xffff_ffff ? {} : { firstOverflowRow: overflow }),
    oracleChangedRows: Object.freeze(oracleChangedRows),
    oracleMismatchRows: Object.freeze(oracleMismatchRows),
    predictedScheduledRows: bitIds(words, relative(layout.predictedRowBitsBaseWords),
      layout.rowCapacity), omittedChangedRowCount: words[h.omittedChangedRowCount]!,
    ...(omitted === 0xffff_ffff ? {} : { firstOmittedChangedRow: omitted }),
    acceptedParity, acceptedGeneration: words[h.acceptedGeneration]!,
    candidateGeneration: words[h.candidateGeneration]!,
    rowTiles: Object.freeze(rowTiles) });
}
