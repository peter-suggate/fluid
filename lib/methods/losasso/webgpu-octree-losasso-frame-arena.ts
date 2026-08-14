/**
 * Versioned physical arena ABI shared by the recurring LoSASSO frame graph.
 *
 * Region offsets are planned once from construction-time capacities and baked
 * into generated WGSL.  The directory header exists for fail-closed epoch and
 * ABI validation; hot row/edge loops never chase an offset table.
 */

export const OCTREE_LOSASSO_FRAME_ARENA_MAGIC = 0x4c534152; // "LSAR"
export const OCTREE_LOSASSO_FRAME_ARENA_VERSION = 1;
export const OCTREE_LOSASSO_ARENA_ALIGNMENT_BYTES = 256;
const ALIGNMENT_WORDS = OCTREE_LOSASSO_ARENA_ALIGNMENT_BYTES / 4;

export interface OctreeLosassoArenaRegion {
  readonly wordOffset: number;
  readonly wordLength: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface OctreeLosassoBufferView {
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
}

export function octreeLosassoArenaView(buffer: GPUBuffer,
  region: OctreeLosassoArenaRegion): OctreeLosassoBufferView {
  if (buffer.size < region.byteOffset + region.byteLength) {
    throw new RangeError("LoSASSO arena view exceeds its physical buffer");
  }
  return Object.freeze({ buffer, offset: region.byteOffset, size: region.byteLength });
}

export interface OctreeLosassoOperatorArenaLayout {
  readonly header: OctreeLosassoArenaRegion;
  readonly banks: readonly [OctreeLosassoArenaRegion, OctreeLosassoArenaRegion];
  readonly bankStrideWords: number;
  readonly levelBasesWords: readonly number[];
  readonly candidateLevelBasesWords: readonly number[];
  readonly levels: readonly OctreeLosassoOperatorLevelLayout[];
  readonly bufferBytes: number;
}

export interface OctreeLosassoOperatorLevelLayout {
  /** Offsets below are relative to this level's bank-relative base. */
  readonly bankRelativeBaseWords: number;
  readonly controlOffsetWords: number;
  readonly rowOffsetsOffsetWords: number;
  /** Dense directed `{neighbour:u32, coefficient:f32}` records. */
  readonly directedEdgesOffsetWords: number;
  readonly parentsOffsetWords: number;
  readonly childOffsetsOffsetWords: number;
  readonly childListOffsetWords: number;
  readonly rowCapacity: number;
  readonly directedEdgeCapacity: number;
  readonly blockWords: number;
}

export interface OctreeLosassoFrameArenaLayout {
  readonly header: OctreeLosassoArenaRegion;
  readonly pressureA: OctreeLosassoArenaRegion;
  readonly pressureB: OctreeLosassoArenaRegion;
  readonly rightHandSide: OctreeLosassoArenaRegion;
  readonly diagonal: OctreeLosassoArenaRegion;
  readonly residual: OctreeLosassoArenaRegion;
  readonly direction: OctreeLosassoArenaRegion;
  /** Four multilevel arrays: RHS/XA/XB/residual. */
  readonly vcycle: readonly [OctreeLosassoArenaRegion, OctreeLosassoArenaRegion,
    OctreeLosassoArenaRegion, OctreeLosassoArenaRegion];
  readonly vectorLevelBasesWords: readonly number[];
  readonly vectorArenaWords: number;
  readonly bufferBytes: number;
}

export interface OctreeLosassoControlArenaLayout {
  readonly header: OctreeLosassoArenaRegion;
  readonly solve: OctreeLosassoArenaRegion;
  readonly dispatch: OctreeLosassoArenaRegion;
  readonly tuningWordOffset: number;
  readonly rowDispatchByteOffset: number;
  readonly faceDispatchByteOffset: number;
  readonly bufferBytes: number;
}

export interface OctreeLosassoFrameArenaPlan {
  readonly magic: typeof OCTREE_LOSASSO_FRAME_ARENA_MAGIC;
  readonly version: typeof OCTREE_LOSASSO_FRAME_ARENA_VERSION;
  readonly rowCapacity: number;
  readonly levelRowCapacities: readonly number[];
  readonly operator: OctreeLosassoOperatorArenaLayout;
  readonly frame: OctreeLosassoFrameArenaLayout;
  readonly control: OctreeLosassoControlArenaLayout;
}

function positive(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`LoSASSO arena ${label} must be a positive safe integer`);
  }
  return value;
}

function alignWords(words: number): number {
  return Math.ceil(words / ALIGNMENT_WORDS) * ALIGNMENT_WORDS;
}

function region(wordOffset: number, wordLength: number): OctreeLosassoArenaRegion {
  return Object.freeze({ wordOffset, wordLength,
    byteOffset: wordOffset * 4, byteLength: wordLength * 4 });
}

/** 256-byte aligned multilevel vector bases, finest first. */
export function planOctreeLosassoVectorBases(levelRowCapacities: readonly number[]): {
  readonly basesWords: readonly number[]; readonly arenaWords: number;
} {
  if (levelRowCapacities.length < 1) throw new RangeError("LoSASSO arena needs one level");
  const bases: number[] = [];
  let cursor = 0;
  for (const [level, raw] of levelRowCapacities.entries()) {
    const capacity = positive(raw, `level ${level} row capacity`);
    if (level > 0 && capacity > levelRowCapacities[level - 1]!) {
      throw new RangeError("LoSASSO arena row capacities must not grow toward the bottom");
    }
    cursor = alignWords(cursor);
    bases.push(cursor);
    cursor += capacity;
  }
  return Object.freeze({ basesWords: Object.freeze(bases), arenaWords: alignWords(cursor) });
}

export function planOctreeLosassoFrameArenas(options: {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly incidenceCapacity?: number;
  readonly levelRowCapacities: readonly number[];
  readonly levelDirectedEdgeCapacities?: readonly number[];
}): OctreeLosassoFrameArenaPlan {
  const rows = positive(options.rowCapacity, "row capacity");
  const faces = positive(options.faceCapacity, "face capacity");
  const incidences = positive(options.incidenceCapacity ?? 2 * faces, "incidence capacity");
  if (incidences > 2 * faces) {
    throw new RangeError("LoSASSO arena incidence capacity exceeds the unique-face bound");
  }
  const levels = Object.freeze([...options.levelRowCapacities]);
  if (levels[0] !== rows) throw new RangeError("LoSASSO arena L0 capacity must equal row capacity");
  const vectors = planOctreeLosassoVectorBases(levels);

  const directedCapacities = options.levelDirectedEdgeCapacities === undefined
    ? levels.map((levelRows, level) => level === 0 ? incidences
      : Math.min(2 * faces, levelRows * levelRows))
    : [...options.levelDirectedEdgeCapacities];
  if (directedCapacities.length !== levels.length) {
    throw new RangeError("LoSASSO arena needs one directed-edge capacity per level");
  }
  let bankCursor = 0;
  const operatorLevels: OctreeLosassoOperatorLevelLayout[] = [];
  for (let level = 0; level < levels.length; level += 1) {
    const rowCapacity = positive(levels[level]!, `level ${level} row capacity`);
    const directedEdgeCapacity = positive(directedCapacities[level]!,
      `level ${level} directed-edge capacity`);
    if (directedEdgeCapacity > (level === 0 ? incidences : 2 * faces)) {
      throw new RangeError(`LoSASSO arena level ${level} directed-edge capacity exceeds its source bound`);
    }
    bankCursor = alignWords(bankCursor);
    const bankRelativeBaseWords = bankCursor;
    const controlOffsetWords = 0;
    const rowOffsetsOffsetWords = 8;
    const directedEdgesOffsetWords = alignWords(rowOffsetsOffsetWords + rowCapacity + 1);
    const previousRows = level === 0 ? 0 : levels[level - 1]!;
    const parentsOffsetWords = alignWords(directedEdgesOffsetWords + 2 * directedEdgeCapacity);
    const childOffsetsOffsetWords = alignWords(parentsOffsetWords + previousRows);
    const childListOffsetWords = alignWords(childOffsetsOffsetWords + rowCapacity + 1);
    const blockWords = alignWords(childListOffsetWords + previousRows);
    operatorLevels.push(Object.freeze({ bankRelativeBaseWords, controlOffsetWords,
      rowOffsetsOffsetWords, directedEdgesOffsetWords, parentsOffsetWords,
      childOffsetsOffsetWords, childListOffsetWords, rowCapacity,
      directedEdgeCapacity, blockWords }));
    bankCursor += blockWords;
  }
  const operatorHeaderWords = ALIGNMENT_WORDS;
  const bankStrideWords = alignWords(bankCursor);
  const acceptedBank = region(operatorHeaderWords, bankStrideWords);
  const candidateBank = region(operatorHeaderWords + bankStrideWords, bankStrideWords);
  const levelBasesWords = Object.freeze(operatorLevels.map(level =>
    acceptedBank.wordOffset + level.bankRelativeBaseWords));
  const candidateLevelBasesWords = Object.freeze(operatorLevels.map(level =>
    candidateBank.wordOffset + level.bankRelativeBaseWords));
  const operatorWords = alignWords(candidateBank.wordOffset + candidateBank.wordLength);

  let frameCursor = ALIGNMENT_WORDS;
  const allocateFrame = (words: number): OctreeLosassoArenaRegion => {
    frameCursor = alignWords(frameCursor);
    const result = region(frameCursor, words);
    frameCursor += words;
    return result;
  };
  const pressureA = allocateFrame(rows);
  const pressureB = allocateFrame(rows);
  const rightHandSide = allocateFrame(rows);
  const diagonal = allocateFrame(rows);
  const residual = allocateFrame(rows);
  const direction = allocateFrame(rows);
  const vcycle = [allocateFrame(vectors.arenaWords), allocateFrame(vectors.arenaWords),
    allocateFrame(vectors.arenaWords), allocateFrame(vectors.arenaWords)] as const;

  // Solve ABI remains at byte zero so existing diagnostics/readback consume
  // the shared buffer without a staging copy. Dispatch and header are disjoint
  // pages and therefore survive the per-solve range clear.
  const solve = region(0, ALIGNMENT_WORDS);
  const dispatch = region(ALIGNMENT_WORDS, ALIGNMENT_WORDS);
  const header = region(2 * ALIGNMENT_WORDS, ALIGNMENT_WORDS);
  // Runtime dials live in the versioned header page, outside the per-solve
  // clear range, so queue writes need no staging buffer or copy command.
  const tuningWordOffset = header.wordOffset + 8;
  const rowDispatchByteOffset = dispatch.byteOffset;
  const faceDispatchByteOffset = dispatch.byteOffset + 16;
  const controlWords = alignWords(header.wordOffset + header.wordLength);

  return Object.freeze({
    magic: OCTREE_LOSASSO_FRAME_ARENA_MAGIC,
    version: OCTREE_LOSASSO_FRAME_ARENA_VERSION,
    rowCapacity: rows,
    levelRowCapacities: levels,
    operator: Object.freeze({ header: region(0, operatorHeaderWords),
      banks: Object.freeze([acceptedBank, candidateBank]) as readonly [
        OctreeLosassoArenaRegion, OctreeLosassoArenaRegion],
      bankStrideWords, levelBasesWords, candidateLevelBasesWords,
      levels: Object.freeze(operatorLevels), bufferBytes: operatorWords * 4 }),
    frame: Object.freeze({ header: region(0, ALIGNMENT_WORDS), pressureA, pressureB,
      rightHandSide, diagonal, residual, direction, vcycle,
      vectorLevelBasesWords: vectors.basesWords, vectorArenaWords: vectors.arenaWords,
      bufferBytes: alignWords(frameCursor) * 4 }),
    control: Object.freeze({ header, solve, dispatch,
      tuningWordOffset, rowDispatchByteOffset, faceDispatchByteOffset,
      bufferBytes: controlWords * 4 }),
  });
}
