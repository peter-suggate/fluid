/** Construction-only shadow census for the FPA preparation tile cut. */
export const SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_MAGIC = 0x46544331; // FTC1
export const SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_VERSION = 1;
export const SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HISTOGRAM_WORDS = 25;

export const SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, generation: 3, fault: 4,
  dirtyBricks: 5, maskPopcount: 6, fullCells: 7, selectedCells: 8,
  fullIncidenceVisits: 9, selectedIncidenceVisits: 10,
  uniqueFullRows: 11, uniqueTileRows: 12, uniqueOverlapRows: 13,
  uniqueFullOnlyRows: 14, omittedChangedRowCount: 15,
  firstWitnessRow: 16, firstWitnessCell: 17, firstWitnessTile: 18,
  firstWitnessRung: 19, firstWitnessSpan: 20, firstWitnessCause: 21,
  macroFallbackCount: 22, partialFallbackCount: 23,
  topologyFallbackCount: 24, histogramBase: 32,
} as const);

export const SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS =
  SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER.histogramBase
  + SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HISTOGRAM_WORDS;

export interface SparseCM12FacePreparationTileCensusLayout {
  readonly baseWords: number;
  readonly headerBaseWords: number;
  readonly fullRowBitsBaseWords: number;
  readonly tileRowBitsBaseWords: number;
  readonly fullSourceCellBaseWords: number;
  readonly priorAuthorityBaseWords: number;
  readonly rowCapacity: number;
  readonly rowBitWordCount: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const checked = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
  return value;
};

export function createSparseCM12FacePreparationTileCensusLayout(options: {
  readonly baseWords: number;
  readonly rowCapacity: number;
  readonly alignmentWords?: number;
}): SparseCM12FacePreparationTileCensusLayout {
  const alignment = checked(options.alignmentWords ?? 64, "alignmentWords");
  if (alignment === 0) throw new RangeError("alignmentWords must be positive");
  const baseWords = Math.ceil(checked(options.baseWords, "baseWords") / alignment)
    * alignment;
  const rowCapacity = checked(options.rowCapacity, "rowCapacity");
  const rowBitWordCount = Math.ceil(rowCapacity / 32);
  const headerBaseWords = baseWords;
  const fullRowBitsBaseWords = headerBaseWords
    + SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS;
  const tileRowBitsBaseWords = fullRowBitsBaseWords + rowBitWordCount;
  const fullSourceCellBaseWords = tileRowBitsBaseWords + rowBitWordCount;
  const priorAuthorityBaseWords = fullSourceCellBaseWords + rowCapacity;
  const totalWords = priorAuthorityBaseWords + rowCapacity;
  return Object.freeze({ baseWords, headerBaseWords, fullRowBitsBaseWords,
    tileRowBitsBaseWords, fullSourceCellBaseWords, priorAuthorityBaseWords,
    rowCapacity, rowBitWordCount, totalWords, totalBytes: 4 * totalWords });
}

export function createSparseCM12FacePreparationTileCensusInitialWords(
  layout: SparseCM12FacePreparationTileCensusLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER;
  words[h.magic] = SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_MAGIC;
  words[h.version] = SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_VERSION;
  words[h.headerWords] = SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS;
  words[h.firstWitnessRow] = 0xffff_ffff;
  words[h.firstWitnessCell] = 0xffff_ffff;
  return words;
}

export interface SparseCM12FacePreparationTileCensusQA {
  readonly generation: number;
  readonly fault: number;
  readonly dirtyBricks: number;
  readonly maskPopcount: number;
  readonly fullCells: number;
  readonly selectedCells: number;
  readonly fullIncidenceVisits: number;
  readonly selectedIncidenceVisits: number;
  readonly uniqueFullRows: number;
  readonly uniqueTileRows: number;
  readonly uniqueOverlapRows: number;
  readonly uniqueFullOnlyRows: number;
  readonly omittedChangedRowCount: number;
  readonly firstWitness: undefined | Readonly<{
    row: number; cell: number; tile: number; rung: number;
    span: number; cause: number;
  }>;
  readonly macroFallbackCount: number;
  readonly partialFallbackCount: number;
  readonly topologyFallbackCount: number;
  /** Row-major [spanLog2 0..4][rungLog2 0..4] mask-bit totals. */
  readonly maskPopcountBySpanRung: readonly number[];
}

export function inspectSparseCM12FacePreparationTileCensusQA(
  words: Uint32Array,
): SparseCM12FacePreparationTileCensusQA {
  if (words.length < SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS) {
    throw new RangeError("face preparation tile census header is truncated");
  }
  const h = SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER;
  if (words[h.magic] !== SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_MAGIC
    || words[h.version] !== SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_VERSION
    || words[h.headerWords] !== SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS) {
    throw new Error("face preparation tile census header is invalid");
  }
  const first = words[h.firstWitnessRow]!;
  return Object.freeze({
    generation: words[h.generation]!, fault: words[h.fault]!,
    dirtyBricks: words[h.dirtyBricks]!, maskPopcount: words[h.maskPopcount]!,
    fullCells: words[h.fullCells]!, selectedCells: words[h.selectedCells]!,
    fullIncidenceVisits: words[h.fullIncidenceVisits]!,
    selectedIncidenceVisits: words[h.selectedIncidenceVisits]!,
    uniqueFullRows: words[h.uniqueFullRows]!,
    uniqueTileRows: words[h.uniqueTileRows]!,
    uniqueOverlapRows: words[h.uniqueOverlapRows]!,
    uniqueFullOnlyRows: words[h.uniqueFullOnlyRows]!,
    omittedChangedRowCount: words[h.omittedChangedRowCount]!,
    firstWitness: first === 0xffff_ffff ? undefined : Object.freeze({
      row: first, cell: words[h.firstWitnessCell]!, tile: words[h.firstWitnessTile]!,
      rung: words[h.firstWitnessRung]!, span: words[h.firstWitnessSpan]!,
      cause: words[h.firstWitnessCause]!,
    }),
    macroFallbackCount: words[h.macroFallbackCount]!,
    partialFallbackCount: words[h.partialFallbackCount]!,
    topologyFallbackCount: words[h.topologyFallbackCount]!,
    maskPopcountBySpanRung: Object.freeze(Array.from(words.slice(h.histogramBase,
      h.histogramBase + SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HISTOGRAM_WORDS))),
  });
}
