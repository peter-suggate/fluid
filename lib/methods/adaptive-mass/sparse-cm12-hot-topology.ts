import type {
  SparseAtlasCompositeGrid,
  SparseAtlasGradientRow,
  SparseAtlasGradientRowKind,
} from "./sparse-atlas-composite-projection";
import {
  SPARSE_CM12_LOGICAL_OWNER_FLAG,
  SPARSE_CM12_LOGICAL_OWNER_HEADER,
  createSparseCM12LogicalOwnerDirectory,
  sparseCM12LogicalOwnerHeaderValid,
  validateSparseCM12LogicalOwnerDirectory,
  type SparseCM12LogicalOwnerDirectoryLayout,
} from "./sparse-cm12-logical-owner-directory";
import type { SparseBrickFineResolution } from "./sparse-brick-atlas";

/** Immutable compiled hot-topology arena, version HTP1. */
export const SPARSE_CM12_HOT_TOPOLOGY_MAGIC = 0x4854_5031;
export const SPARSE_CM12_HOT_TOPOLOGY_VERSION = 1;
export const SPARSE_CM12_HOT_TOPOLOGY_HEADER_WORDS = 32;
export const SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS = 8;
export const SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS = 16;
export const SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS = 2;
export const SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS = 2;
export const SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS = 4;
export const SPARSE_CM12_HOT_TOPOLOGY_INVALID = 0xffff_ffff;
export const SPARSE_CM12_HOT_TOPOLOGY_BRICK_BITS = 27;
export const SPARSE_CM12_HOT_TOPOLOGY_COMPLETE = 1 << 0;
export const SPARSE_CM12_HOT_TOPOLOGY_VALIDATED = 1 << 1;
export const SPARSE_CM12_HOT_TOPOLOGY_COMMON_TWO_TERM = 1 << 20;

export const SPARSE_CM12_HOT_TOPOLOGY_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, flags: 3,
  brickFineResolution: 4, presentationPageResolution: 5,
  atlasGeneration: 6, logicalOwnerWords: 7,
  cellCount: 8, rowCount: 9, variableTermCount: 10,
  incidenceCount: 11, directedEdgeCount: 12, requirementCount: 13,
  cellBase: 14, rowBase: 15, variableTermBase: 16,
  incidenceOffsetBase: 17, incidenceBase: 18,
  directedEdgeOffsetBase: 19, directedEdgeBase: 20,
  requirementBase: 21, totalWords: 22,
  cellWords: 23, rowWords: 24, termWords: 25,
  incidenceWords: 26, edgeWords: 27,
  reserved0: 28, reserved1: 29, reserved2: 30, reserved3: 31,
} as const);

export const SPARSE_CM12_HOT_TOPOLOGY_CELL = Object.freeze({
  centerX: 0, centerY: 1, centerZ: 2, volume: 3,
  widthX: 4, widthY: 5, widthZ: 6, brickAndResolution: 7,
} as const);

export const SPARSE_CM12_HOT_TOPOLOGY_ROW = Object.freeze({
  tag: 0,
  /** Inline cell 0 for a common row, otherwise first variable-term index. */
  cell0OrFirstTerm: 1,
  coefficient0: 2, cell1: 3, coefficient1: 4,
  dualWeight: 5, area: 6, distance: 7, exteriorPhi: 8,
  centerX: 9, centerY: 10, centerZ: 11,
  firstRequirement: 12, requirementCount: 13, reserved: 14, identity: 15,
} as const);

export const SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE = Object.freeze({
  row: 0, termOrdinal: 1,
} as const);

export const SPARSE_CM12_HOT_TOPOLOGY_EDGE = Object.freeze({
  neighbor: 0, row: 1, pressureBaseWeight: 2, extrapolationWeight: 3,
} as const);

const ROW_KIND: Readonly<Record<SparseAtlasGradientRowKind, number>> = Object.freeze({
  "intra-brick": 0, "brick-face": 1, "mixed-seam": 2, "sparse-air": 3,
});

export interface SparseCM12HotTopologyLayout {
  readonly brickFineResolution: SparseBrickFineResolution;
  readonly presentationPageResolution: SparseBrickFineResolution;
  readonly atlasGeneration: number;
  readonly logicalOwner: SparseCM12LogicalOwnerDirectoryLayout;
  readonly logicalOwnerWords: number;
  readonly headerBaseWords: number;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly variableTermCount: number;
  readonly incidenceCount: number;
  readonly directedEdgeCount: number;
  readonly requirementCount: number;
  readonly cellBaseWords: number;
  readonly rowBaseWords: number;
  readonly variableTermBaseWords: number;
  readonly incidenceOffsetBaseWords: number;
  readonly incidenceBaseWords: number;
  readonly directedEdgeOffsetBaseWords: number;
  readonly directedEdgeBaseWords: number;
  readonly requirementBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12HotTopology {
  readonly layout: SparseCM12HotTopologyLayout;
  readonly words: Uint32Array;
}

export interface SparseCM12HotTopologyByteMapEntry {
  readonly name: string;
  readonly offsetBytes: number;
  readonly bytesPerRecord: number;
  readonly recordCount: number;
  readonly totalBytes: number;
  readonly ordering: string;
}

const alignWords = (value: number, alignment: number): number =>
  Math.ceil(value / alignment) * alignment;

const F32_BITS_BUFFER = new ArrayBuffer(4);
const F32_BITS_FLOAT = new Float32Array(F32_BITS_BUFFER);
const F32_BITS_WORD = new Uint32Array(F32_BITS_BUFFER);
const f32Bits = (value: number): number => {
  if (!Number.isFinite(value)) throw new RangeError(`non-finite topology scalar ${value}`);
  F32_BITS_FLOAT[0] = value;
  return F32_BITS_WORD[0]!;
};

const packBrickResolution = (brick: number, resolution: number): number => {
  if (!Number.isSafeInteger(brick) || brick < 0 || brick >= 2 ** SPARSE_CM12_HOT_TOPOLOGY_BRICK_BITS
    || !Number.isSafeInteger(resolution) || resolution < 1 || resolution > 31) {
    throw new RangeError(`unrepresentable brick/resolution ${brick}/${resolution}`);
  }
  return ((brick << 5) | resolution) >>> 0;
};

const rowTag = (row: SparseAtlasGradientRow): number => {
  if (row.terms.length < 1 || row.terms.length > 0xffff) {
    throw new RangeError(`row ${row.id} has unrepresentable term count ${row.terms.length}`);
  }
  const common = row.terms.length === 2;
  return (row.terms.length | (row.axis << 16) | (ROW_KIND[row.kind] << 18)
    | (common ? SPARSE_CM12_HOT_TOPOLOGY_COMMON_TWO_TERM : 0)) >>> 0;
};

const requireContiguousGrid = (grid: SparseAtlasCompositeGrid): Map<number, number> => {
  const brickByKey = new Map<number, number>();
  grid.atlas.bricks.forEach((brick, index) => brickByKey.set(brick.key, index));
  grid.cells.forEach((cell, index) => {
    if (cell.id !== index) throw new Error(`cell ${index} has non-canonical id ${cell.id}`);
    if (!brickByKey.has(cell.brickKey)) throw new Error(`cell ${index} has no resident brick`);
    if (cell.openFraction !== 1 || cell.openVolume !== cell.volume
      || cell.separatingPressureMinimum) {
      throw new Error(`cell ${index} requires dynamic boundary geometry; HTP1 is literal open-domain topology`);
    }
  });
  grid.gradientRows.forEach((row, index) => {
    if (row.id !== index) throw new Error(`row ${index} has non-canonical id ${row.id}`);
    row.terms.forEach((term) => {
      if (!Number.isSafeInteger(term.cellId) || term.cellId < 0
        || term.cellId >= grid.cells.length) throw new Error(`row ${index} has invalid cell ${term.cellId}`);
    });
  });
  return brickByKey;
};

const rowRequirements = (
  row: SparseAtlasGradientRow,
  grid: SparseAtlasCompositeGrid,
  brickByKey: ReadonlyMap<number, number>,
): number[] => {
  const seenBricks = new Set<number>();
  const result: number[] = [];
  for (const term of row.terms) {
    const cell = grid.cells[term.cellId]!;
    const brick = brickByKey.get(cell.brickKey);
    if (brick === undefined) throw new Error(`row ${row.id} cell ${term.cellId} has no brick`);
    const packed = packBrickResolution(brick, cell.brickResolution);
    if (seenBricks.has(brick)) continue;
    seenBricks.add(brick);
    result.push(packed);
  }
  return result;
};

export function createSparseCM12HotTopology(
  grid: SparseAtlasCompositeGrid,
  options: {
    readonly brickFineResolution?: SparseBrickFineResolution;
    readonly presentationPageResolution?: SparseBrickFineResolution;
  } = {},
): SparseCM12HotTopology {
  const brickFineResolution = options.brickFineResolution ?? 16;
  const presentationPageResolution = options.presentationPageResolution ?? 16;
  if (brickFineResolution !== 16 || presentationPageResolution !== 16) {
    throw new Error("HTP1 is intentionally the B16/P16 physical ABI");
  }
  const brickByKey = requireContiguousGrid(grid);
  const logicalOwner = createSparseCM12LogicalOwnerDirectory(grid.atlas, {
    brickFineResolution, presentationPageResolution,
  });
  const cellCount = grid.cells.length;
  const rowCount = grid.gradientRows.length;
  let variableTermCount = 0, incidenceCount = 0, directedEdgeCount = 0;
  let requirementCount = 0;
  const requirementsByRow: number[][] = [];
  for (const row of grid.gradientRows) {
    if (row.terms.length !== 2) variableTermCount += row.terms.length;
    incidenceCount += row.terms.length;
    for (const own of row.terms) for (const other of row.terms) {
      if (other.cellId !== own.cellId) directedEdgeCount += 1;
    }
    const requirements = rowRequirements(row, grid, brickByKey);
    requirementsByRow.push(requirements);
    requirementCount += requirements.length;
  }
  const counts = [cellCount, rowCount, variableTermCount, incidenceCount,
    directedEdgeCount, requirementCount];
  if (counts.some((value) => !Number.isSafeInteger(value) || value > 0xffff_fffe)) {
    throw new RangeError("HTP1 record count exceeds u32 address space");
  }

  const headerBaseWords = alignWords(logicalOwner.layout.totalWords, 64);
  const cellBaseWords = headerBaseWords + SPARSE_CM12_HOT_TOPOLOGY_HEADER_WORDS;
  const rowBaseWords = alignWords(cellBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS * cellCount, 4);
  const variableTermBaseWords = alignWords(rowBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS * rowCount, 4);
  const incidenceOffsetBaseWords = alignWords(variableTermBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS * variableTermCount, 4);
  const incidenceBaseWords = alignWords(incidenceOffsetBaseWords + cellCount + 1, 4);
  const directedEdgeOffsetBaseWords = alignWords(incidenceBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS * incidenceCount, 4);
  const directedEdgeBaseWords = alignWords(directedEdgeOffsetBaseWords + cellCount + 1, 4);
  const requirementBaseWords = alignWords(directedEdgeBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS * directedEdgeCount, 4);
  const totalWords = alignWords(requirementBaseWords + requirementCount, 4);
  if (totalWords > 0x3fff_ffff) throw new RangeError("HTP1 arena exceeds 4 GiB");
  const layout: SparseCM12HotTopologyLayout = Object.freeze({
    brickFineResolution, presentationPageResolution,
    atlasGeneration: grid.atlas.generation, logicalOwner: logicalOwner.layout,
    logicalOwnerWords: logicalOwner.layout.totalWords, headerBaseWords,
    cellCount, rowCount, variableTermCount, incidenceCount, directedEdgeCount,
    requirementCount, cellBaseWords, rowBaseWords, variableTermBaseWords,
    incidenceOffsetBaseWords, incidenceBaseWords, directedEdgeOffsetBaseWords,
    directedEdgeBaseWords, requirementBaseWords, totalWords, totalBytes: totalWords * 4,
  });
  const words = new Uint32Array(totalWords);
  words.set(logicalOwner.words);
  words.set([
    SPARSE_CM12_HOT_TOPOLOGY_MAGIC, SPARSE_CM12_HOT_TOPOLOGY_VERSION,
    SPARSE_CM12_HOT_TOPOLOGY_HEADER_WORDS, 0,
    brickFineResolution, presentationPageResolution, grid.atlas.generation,
    layout.logicalOwnerWords, cellCount, rowCount, variableTermCount,
    incidenceCount, directedEdgeCount, requirementCount,
    cellBaseWords, rowBaseWords, variableTermBaseWords,
    incidenceOffsetBaseWords, incidenceBaseWords,
    directedEdgeOffsetBaseWords, directedEdgeBaseWords,
    requirementBaseWords, totalWords, SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS,
    SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS, SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS,
    SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS, SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS,
    0, 0, 0, 0,
  ], headerBaseWords);

  grid.cells.forEach((cell, id) => {
    const brick = brickByKey.get(cell.brickKey)!;
    const at = cellBaseWords + id * SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS;
    words.set([
      f32Bits(cell.centerFine[0]), f32Bits(cell.centerFine[1]), f32Bits(cell.centerFine[2]),
      f32Bits(cell.volume), f32Bits(cell.widthsFine[0]), f32Bits(cell.widthsFine[1]),
      f32Bits(cell.widthsFine[2]), packBrickResolution(brick, cell.brickResolution),
    ], at);
  });

  const incidenceCounts = new Uint32Array(cellCount);
  const edgeCounts = new Uint32Array(cellCount);
  for (const row of grid.gradientRows) for (const term of row.terms) {
    incidenceCounts[term.cellId]! += 1;
    edgeCounts[term.cellId]! += row.terms.reduce((count, other) =>
      count + Number(other.cellId !== term.cellId), 0);
  }
  let incidenceCursor = 0, edgeCursor = 0;
  for (let cell = 0; cell < cellCount; cell += 1) {
    words[incidenceOffsetBaseWords + cell] = incidenceCursor;
    words[directedEdgeOffsetBaseWords + cell] = edgeCursor;
    incidenceCursor += incidenceCounts[cell]!;
    edgeCursor += edgeCounts[cell]!;
  }
  words[incidenceOffsetBaseWords + cellCount] = incidenceCursor;
  words[directedEdgeOffsetBaseWords + cellCount] = edgeCursor;
  const incidenceWrite = Uint32Array.from(words.subarray(
    incidenceOffsetBaseWords, incidenceOffsetBaseWords + cellCount));
  const edgeWrite = Uint32Array.from(words.subarray(
    directedEdgeOffsetBaseWords, directedEdgeOffsetBaseWords + cellCount));
  let variableTermCursor = 0, requirementCursor = 0;
  grid.gradientRows.forEach((row, rowId) => {
    const tag = rowTag(row);
    const common = row.terms.length === 2;
    const at = rowBaseWords + rowId * SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS;
    const firstRequirement = requirementCursor;
    for (const requirement of requirementsByRow[rowId]!) {
      words[requirementBaseWords + requirementCursor++] = requirement;
    }
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.tag] = tag;
    if (common) {
      words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm] = row.terms[0]!.cellId;
      words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.coefficient0] = f32Bits(row.terms[0]!.coefficient);
      words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell1] = row.terms[1]!.cellId;
      words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.coefficient1] = f32Bits(row.terms[1]!.coefficient);
    } else {
      words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm] = variableTermCursor;
      words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.coefficient0] = SPARSE_CM12_HOT_TOPOLOGY_INVALID;
      words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell1] = SPARSE_CM12_HOT_TOPOLOGY_INVALID;
      words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.coefficient1] = SPARSE_CM12_HOT_TOPOLOGY_INVALID;
      for (const term of row.terms) {
        const termAt = variableTermBaseWords
          + variableTermCursor * SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS;
        words[termAt] = term.cellId;
        words[termAt + 1] = f32Bits(term.coefficient);
        variableTermCursor += 1;
      }
    }
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.dualWeight] = f32Bits(row.dualWeight);
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.area] = f32Bits(row.area);
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.distance] = f32Bits(row.distance);
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.exteriorPhi] = f32Bits(row.exteriorPhi ?? 0.5);
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.centerX] = f32Bits(row.centerFine[0]);
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.centerY] = f32Bits(row.centerFine[1]);
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.centerZ] = f32Bits(row.centerFine[2]);
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.firstRequirement] = firstRequirement;
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.requirementCount] =
      requirementsByRow[rowId]!.length;
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.reserved] = 0;
    words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.identity] = rowId;

    row.terms.forEach((own, ordinal) => {
      const incidence = incidenceWrite[own.cellId]!++;
      const incidenceAt = incidenceBaseWords
        + incidence * SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS;
      words.set([rowId, ordinal], incidenceAt);
      row.terms.forEach((other) => {
        if (other.cellId === own.cellId) return;
        const edge = edgeWrite[own.cellId]!++;
        const edgeAt = directedEdgeBaseWords + edge * SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS;
        const extrapolationTerm = row.terms.find((term) => term.cellId === other.cellId)!;
        words.set([
          other.cellId, rowId,
          f32Bits(own.coefficient * row.dualWeight * other.coefficient),
          f32Bits(extrapolationTerm.coefficient) & 0x7fff_ffff,
        ], edgeAt);
      });
    });
  });
  if (variableTermCursor !== variableTermCount || requirementCursor !== requirementCount
    || incidenceCursor !== incidenceCount || edgeCursor !== directedEdgeCount) {
    throw new Error("HTP1 construction cursor mismatch");
  }
  words[headerBaseWords + SPARSE_CM12_HOT_TOPOLOGY_HEADER.flags] =
    SPARSE_CM12_HOT_TOPOLOGY_COMPLETE | SPARSE_CM12_HOT_TOPOLOGY_VALIDATED;
  const result = { layout, words } as const;
  validateSparseCM12HotTopology(result, grid);
  return result;
}

export function sparseCM12HotTopologyHeaderValid(topology: SparseCM12HotTopology): boolean {
  const { layout: l, words } = topology;
  const h = l.headerBaseWords;
  const expected = [
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.magic, SPARSE_CM12_HOT_TOPOLOGY_MAGIC],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.version, SPARSE_CM12_HOT_TOPOLOGY_VERSION],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.headerWords, SPARSE_CM12_HOT_TOPOLOGY_HEADER_WORDS],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.brickFineResolution, l.brickFineResolution],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.presentationPageResolution, l.presentationPageResolution],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.atlasGeneration, l.atlasGeneration],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.logicalOwnerWords, l.logicalOwnerWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.cellCount, l.cellCount],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.rowCount, l.rowCount],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.variableTermCount, l.variableTermCount],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.incidenceCount, l.incidenceCount],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.directedEdgeCount, l.directedEdgeCount],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.requirementCount, l.requirementCount],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.cellBase, l.cellBaseWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.rowBase, l.rowBaseWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.variableTermBase, l.variableTermBaseWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.incidenceOffsetBase, l.incidenceOffsetBaseWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.incidenceBase, l.incidenceBaseWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.directedEdgeOffsetBase, l.directedEdgeOffsetBaseWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.directedEdgeBase, l.directedEdgeBaseWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.requirementBase, l.requirementBaseWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.totalWords, l.totalWords],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.cellWords, SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.rowWords, SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.termWords, SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.incidenceWords, SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS],
    [SPARSE_CM12_HOT_TOPOLOGY_HEADER.edgeWords, SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS],
  ] as const;
  const flags = words[h + SPARSE_CM12_HOT_TOPOLOGY_HEADER.flags] ?? 0;
  const directory = { layout: l.logicalOwner, words };
  const expectedHeaderBase = alignWords(l.logicalOwnerWords, 64);
  const expectedCellBase = expectedHeaderBase + SPARSE_CM12_HOT_TOPOLOGY_HEADER_WORDS;
  const expectedRowBase = alignWords(expectedCellBase
    + SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS * l.cellCount, 4);
  const expectedVariableTermBase = alignWords(expectedRowBase
    + SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS * l.rowCount, 4);
  const expectedIncidenceOffsetBase = alignWords(expectedVariableTermBase
    + SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS * l.variableTermCount, 4);
  const expectedIncidenceBase = alignWords(expectedIncidenceOffsetBase + l.cellCount + 1, 4);
  const expectedDirectedEdgeOffsetBase = alignWords(expectedIncidenceBase
    + SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS * l.incidenceCount, 4);
  const expectedDirectedEdgeBase = alignWords(expectedDirectedEdgeOffsetBase
    + l.cellCount + 1, 4);
  const expectedRequirementBase = alignWords(expectedDirectedEdgeBase
    + SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS * l.directedEdgeCount, 4);
  const expectedTotal = alignWords(expectedRequirementBase + l.requirementCount, 4);
  return l.brickFineResolution === 16 && l.presentationPageResolution === 16
    && l.logicalOwnerWords === l.logicalOwner.totalWords
    && l.headerBaseWords === expectedHeaderBase
    && l.cellBaseWords === expectedCellBase && l.rowBaseWords === expectedRowBase
    && l.variableTermBaseWords === expectedVariableTermBase
    && l.incidenceOffsetBaseWords === expectedIncidenceOffsetBase
    && l.incidenceBaseWords === expectedIncidenceBase
    && l.directedEdgeOffsetBaseWords === expectedDirectedEdgeOffsetBase
    && l.directedEdgeBaseWords === expectedDirectedEdgeBase
    && l.requirementBaseWords === expectedRequirementBase && l.totalWords === expectedTotal
    && words.length >= l.totalWords && l.headerBaseWords >= l.logicalOwnerWords
    && expected.every(([at, value]) => words[h + at] === value)
    && words[h + SPARSE_CM12_HOT_TOPOLOGY_HEADER.reserved0] === 0
    && words[h + SPARSE_CM12_HOT_TOPOLOGY_HEADER.reserved1] === 0
    && words[h + SPARSE_CM12_HOT_TOPOLOGY_HEADER.reserved2] === 0
    && words[h + SPARSE_CM12_HOT_TOPOLOGY_HEADER.reserved3] === 0
    && (flags & SPARSE_CM12_HOT_TOPOLOGY_COMPLETE) !== 0
    && (flags & SPARSE_CM12_HOT_TOPOLOGY_VALIDATED) !== 0
    && sparseCM12LogicalOwnerHeaderValid(directory);
}

/** Exhaustive word-exact comparison against the authoritative composite-grid templates. */
export function validateSparseCM12HotTopology(
  topology: SparseCM12HotTopology,
  grid: SparseAtlasCompositeGrid,
): void {
  if (!sparseCM12HotTopologyHeaderValid(topology)) throw new Error("invalid HTP1 header");
  const brickByKey = requireContiguousGrid(grid);
  const { layout: l, words } = topology;
  validateSparseCM12LogicalOwnerDirectory({ layout: l.logicalOwner, words }, grid.atlas);
  const expect = (at: number, value: number, label: string): void => {
    if (words[at] !== (value >>> 0)) throw new Error(`${label}: ${words[at]} != ${value >>> 0}`);
  };
  grid.cells.forEach((cell, id) => {
    const at = l.cellBaseWords + id * SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS;
    const expected = [f32Bits(cell.centerFine[0]), f32Bits(cell.centerFine[1]),
      f32Bits(cell.centerFine[2]), f32Bits(cell.volume), f32Bits(cell.widthsFine[0]),
      f32Bits(cell.widthsFine[1]), f32Bits(cell.widthsFine[2]),
      packBrickResolution(brickByKey.get(cell.brickKey)!, cell.brickResolution)];
    expected.forEach((value, word) => expect(at + word, value, `cell ${id}/${word}`));
  });
  const expectedIncidence: number[][][] = Array.from({ length: l.cellCount }, () => []);
  const expectedEdges: number[][][] = Array.from({ length: l.cellCount }, () => []);
  let variableCursor = 0, requirementCursor = 0;
  grid.gradientRows.forEach((row, id) => {
    const at = l.rowBaseWords + id * SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS;
    const tag = rowTag(row), requirements = rowRequirements(row, grid, brickByKey);
    const expected = new Uint32Array(SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS);
    expected[0] = tag;
    if (row.terms.length === 2) {
      expected[1] = row.terms[0]!.cellId; expected[2] = f32Bits(row.terms[0]!.coefficient);
      expected[3] = row.terms[1]!.cellId; expected[4] = f32Bits(row.terms[1]!.coefficient);
    } else {
      expected[1] = variableCursor; expected[2] = SPARSE_CM12_HOT_TOPOLOGY_INVALID;
      expected[3] = SPARSE_CM12_HOT_TOPOLOGY_INVALID;
      expected[4] = SPARSE_CM12_HOT_TOPOLOGY_INVALID;
      row.terms.forEach((term) => {
        const termAt = l.variableTermBaseWords
          + variableCursor * SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS;
        expect(termAt, term.cellId, `row ${id} variable cell`);
        expect(termAt + 1, f32Bits(term.coefficient), `row ${id} variable coefficient`);
        variableCursor += 1;
      });
    }
    expected[5] = f32Bits(row.dualWeight); expected[6] = f32Bits(row.area);
    expected[7] = f32Bits(row.distance); expected[8] = f32Bits(row.exteriorPhi ?? 0.5);
    expected[9] = f32Bits(row.centerFine[0]); expected[10] = f32Bits(row.centerFine[1]);
    expected[11] = f32Bits(row.centerFine[2]); expected[12] = requirementCursor;
    expected[13] = requirements.length; expected[14] = 0; expected[15] = id;
    expected.forEach((value, word) => expect(at + word, value, `row ${id}/${word}`));
    requirements.forEach((value) => expect(l.requirementBaseWords + requirementCursor++,
      value, `row ${id} requirement`));
    row.terms.forEach((own, ordinal) => {
      expectedIncidence[own.cellId]!.push([id, ordinal]);
      row.terms.forEach((other) => {
        if (own.cellId === other.cellId) return;
        const extrapolationTerm = row.terms.find((term) => term.cellId === other.cellId)!;
        expectedEdges[own.cellId]!.push([other.cellId, id,
          f32Bits(own.coefficient * row.dualWeight * other.coefficient),
          f32Bits(extrapolationTerm.coefficient) & 0x7fff_ffff]);
      });
    });
  });
  let incidenceCursor = 0, edgeCursor = 0;
  for (let cell = 0; cell < l.cellCount; cell += 1) {
    expect(l.incidenceOffsetBaseWords + cell, incidenceCursor, `cell ${cell} incidence offset`);
    for (const record of expectedIncidence[cell]!) {
      record.forEach((value, word) => expect(l.incidenceBaseWords
        + incidenceCursor * SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS + word,
      value, `cell ${cell} incidence ${incidenceCursor}/${word}`));
      incidenceCursor += 1;
    }
    expect(l.directedEdgeOffsetBaseWords + cell, edgeCursor, `cell ${cell} edge offset`);
    for (const record of expectedEdges[cell]!) {
      record.forEach((value, word) => expect(l.directedEdgeBaseWords
        + edgeCursor * SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS + word,
      value, `cell ${cell} edge ${edgeCursor}/${word}`));
      edgeCursor += 1;
    }
  }
  expect(l.incidenceOffsetBaseWords + l.cellCount, incidenceCursor, "incidence sentinel");
  expect(l.directedEdgeOffsetBaseWords + l.cellCount, edgeCursor, "edge sentinel");
  if (variableCursor !== l.variableTermCount || requirementCursor !== l.requirementCount
    || incidenceCursor !== l.incidenceCount || edgeCursor !== l.directedEdgeCount) {
    throw new Error("HTP1 exhaustive validation count mismatch");
  }
}

export function sparseCM12HotTopologyByteMap(
  layout: SparseCM12HotTopologyLayout,
): readonly SparseCM12HotTopologyByteMapEntry[] {
  const entry = (name: string, base: number, bytesPerRecord: number, recordCount: number,
    ordering: string): SparseCM12HotTopologyByteMapEntry => ({
    name, offsetBytes: base * 4, bytesPerRecord, recordCount,
    totalBytes: bytesPerRecord * recordCount, ordering,
  });
  return Object.freeze([
    entry("LOD1 direct logical owners", 0, 8, layout.logicalOwner.logicalBrickCount,
      "dense logical key (x-major)"),
    entry("HTP1 header", layout.headerBaseWords, 128, 1, "single fail-closed ABI header"),
    entry("literal cells", layout.cellBaseWords, 32, layout.cellCount,
      "canonical cell id; brick/key then z/y/x source order"),
    entry("tagged rows", layout.rowBaseWords, 64, layout.rowCount,
      "canonical gradient-row id"),
    entry("variable row terms", layout.variableTermBaseWords, 8, layout.variableTermCount,
      "row id then authoritative term ordinal; common two-term rows are inline"),
    entry("cell incidence offsets", layout.incidenceOffsetBaseWords, 4, layout.cellCount + 1,
      "canonical cell id CSR"),
    entry("cell incidences", layout.incidenceBaseWords, 8, layout.incidenceCount,
      "cell CSR; row id then term ordinal"),
    entry("directed edge offsets", layout.directedEdgeOffsetBaseWords, 4,
      layout.cellCount + 1, "canonical source cell id CSR"),
    entry("pressure/extrapolation edges", layout.directedEdgeBaseWords, 16,
      layout.directedEdgeCount, "source cell CSR; row id then neighbor term ordinal"),
    entry("row rung requirements", layout.requirementBaseWords, 4, layout.requirementCount,
      "row id then first occurrence in authoritative term order"),
  ]);
}

/** Clone-and-corrupt helper used by checkers; never mutates a published arena. */
export function corruptSparseCM12HotTopologyWord(
  topology: SparseCM12HotTopology,
  absoluteWord: number,
  value: number,
): SparseCM12HotTopology {
  const words = Uint32Array.from(topology.words);
  words[absoluteWord] = value >>> 0;
  return { layout: topology.layout, words };
}

// Retain these imports in the ABI module so a combined arena's LOD1 publication
// bits are part of the explicit immutable contract rather than incidental data.
export const SPARSE_CM12_HOT_TOPOLOGY_REQUIRED_LOD_FLAGS =
  SPARSE_CM12_LOGICAL_OWNER_FLAG.complete | SPARSE_CM12_LOGICAL_OWNER_FLAG.validated;
export const SPARSE_CM12_HOT_TOPOLOGY_LOD_FLAGS_WORD =
  SPARSE_CM12_LOGICAL_OWNER_HEADER.flags;
