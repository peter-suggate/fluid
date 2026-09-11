/**
 * PEI1 is the production B8/P8 coarse pressure execution image. It occupies
 * binding 15's existing ordinary-u32 pressure worklist arena and contains only
 * data consumed by the iterative solve: dense pressure-cell IDs, one canonical
 * global membership bitset, dense wet-brick IDs and dense hierarchy tokens. PCF's finalized
 * brick-range cache is the sole wet-brick authority used while compiling this
 * ordinary read-only image.
 */
export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_MAGIC = 0x5045_4931; // PEI1
export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_VERSION = 1;
export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER_WORDS = 32;
export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_ALIGNMENT_WORDS = 16;
export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID = 0xffff_ffff;

export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE = Object.freeze({
  uninitialized: 0, compiling: 1, accepted: 2, fault: 3,
} as const);

export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, invalidPhase: 2, brickCapacity: 3,
  hierarchyCapacity: 4, invalidSource: 5, cellCapacity: 6,
} as const);

export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, totalWords: 3,
  phase: 4, fault: 5, firstFaultId: 6, generation: 7,
  topologyGeneration: 8, pcmGeneration: 9, coefficientGeneration: 10,
  brickCount: 11, wetBrickCount: 12, hierarchyCount: 13, pressureCellCount: 14,
  cellIndirectX: 15, cellIndirectY: 16, cellIndirectZ: 17,
  // The four coarse triplets deliberately match the resident indirect buffer
  // order, so their publication requires one contiguous 48-byte copy.
  brickReductionIndirectX: 18, brickReductionIndirectY: 19,
  brickReductionIndirectZ: 20,
  brickIndirectX: 21, brickIndirectY: 22, brickIndirectZ: 23,
  hierarchyReductionIndirectX: 24, hierarchyReductionIndirectY: 25,
  hierarchyReductionIndirectZ: 26,
  hierarchyIndirectX: 27, hierarchyIndirectY: 28, hierarchyIndirectZ: 29,
  acceptedReceipts: 30, pcmRowGeneration: 31,
} as const);

export interface SparseCM12PressureExecutionImageLayout {
  readonly baseWords: number;
  readonly brickFineResolution: 8;
  readonly presentationPageResolution: 8;
  readonly cellCapacity: number;
  readonly brickCapacity: number;
  readonly hierarchyCapacity: number;
  readonly pressureCellBaseWords: number;
  readonly pressureMembershipBaseWords: number;
  readonly pressureMembershipWordCount: number;
  readonly wetBrickBaseWords: number;
  readonly hierarchyTokenBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const alignWords = (value: number): number => Math.ceil(value
  / SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_ALIGNMENT_WORDS)
  * SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_ALIGNMENT_WORDS;

const checkedCapacity = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1 || value >= 0x4000_0000) {
    throw new RangeError(`${label} must be an integer in [1, 2^30)`);
  }
  return value;
};

export function createSparseCM12PressureExecutionImageLayout(options: {
  readonly baseWords: number;
  readonly cellCapacity: number;
  readonly brickCapacity: number;
  readonly hierarchyCapacity: number;
  readonly brickFineResolution: 8;
  readonly presentationPageResolution: 8;
}): SparseCM12PressureExecutionImageLayout {
  if (options.brickFineResolution !== 8 || options.presentationPageResolution !== 8) {
    throw new Error("PEI1 is the B8/P8 pressure ABI");
  }
  if (!Number.isSafeInteger(options.baseWords) || options.baseWords < 0
    || options.baseWords >= 0x4000_0000) {
    throw new RangeError("PEI1 baseWords must be a non-negative u30 integer");
  }
  const baseWords = alignWords(options.baseWords);
  const cellCapacity = checkedCapacity(options.cellCapacity, "PEI1 cellCapacity");
  const brickCapacity = checkedCapacity(options.brickCapacity, "PEI1 brickCapacity");
  const hierarchyCapacity = checkedCapacity(options.hierarchyCapacity,
    "PEI1 hierarchyCapacity");
  let at = alignWords(baseWords + SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER_WORDS);
  const plane = (words: number) => {
    const start = at;
    at = alignWords(at + words);
    return start;
  };
  const pressureCellBaseWords = plane(cellCapacity);
  const pressureMembershipWordCount = Math.ceil(cellCapacity / 32);
  const pressureMembershipBaseWords = plane(pressureMembershipWordCount);
  const wetBrickBaseWords = plane(brickCapacity);
  const hierarchyTokenBaseWords = plane(hierarchyCapacity);
  const totalWords = alignWords(at);
  return Object.freeze({ baseWords, brickFineResolution: 8,
    presentationPageResolution: 8, cellCapacity, brickCapacity, hierarchyCapacity,
    pressureCellBaseWords, pressureMembershipBaseWords, pressureMembershipWordCount,
    wetBrickBaseWords,
    hierarchyTokenBaseWords,
    totalWords, totalBytes: 4 * totalWords });
}

export function createSparseCM12PressureExecutionImageInitialWords(
  layout: SparseCM12PressureExecutionImageLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER;
  words[h.magic] = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_MAGIC;
  words[h.version] = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_VERSION;
  words[h.headerWords] = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER_WORDS;
  words[h.totalWords] = layout.totalWords;
  words[h.phase] = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE.uninitialized;
  words[h.firstFaultId] = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID;
  words[h.cellIndirectY] = 1; words[h.cellIndirectZ] = 1;
  words[h.brickIndirectY] = 1; words[h.brickIndirectZ] = 1;
  words[h.brickReductionIndirectY] = 1; words[h.brickReductionIndirectZ] = 1;
  words[h.hierarchyIndirectY] = 1; words[h.hierarchyIndirectZ] = 1;
  words[h.hierarchyReductionIndirectY] = 1;
  words[h.hierarchyReductionIndirectZ] = 1;
  const local = (absolute: number) => absolute - layout.baseWords;
  words.fill(SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID,
    local(layout.pressureCellBaseWords),
    local(layout.pressureCellBaseWords) + layout.cellCapacity);
  words.fill(SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID,
    local(layout.wetBrickBaseWords),
    local(layout.wetBrickBaseWords) + layout.brickCapacity);
  words.fill(SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID,
    local(layout.hierarchyTokenBaseWords),
    local(layout.hierarchyTokenBaseWords) + layout.hierarchyCapacity);
  return words;
}

export function sparseCM12PressureExecutionImageIndirectByteOffset(
  layout: SparseCM12PressureExecutionImageLayout,
): number {
  const h = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER;
  return 4 * (layout.baseWords + h.brickReductionIndirectX);
}

export function sparseCM12PressureExecutionImageCellIndirectByteOffset(
  layout: SparseCM12PressureExecutionImageLayout,
): number {
  const h = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER;
  return 4 * (layout.baseWords + h.cellIndirectX);
}

export const SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_ENTRY_POINTS = Object.freeze([
  "finalizeSparseCM12PressureExecutionImage",
] as const);
