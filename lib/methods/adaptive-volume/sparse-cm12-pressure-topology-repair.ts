/** PTR1: compact pressure-topology transaction journal. */
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC = 0x5054_5231; // PTR1
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION = 3;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS = 21;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS = 256;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID = 0xffff_ffff;

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE = Object.freeze({
  uninitialized: 0, accepted: 1, collecting: 2, executingCells: 4, fault: 8,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, invalidPhase: 2, generationExhausted: 3,
  invalidCellRange: 6, producerCoverageGap: 10,
  cellExecutionGap: 12, coefficientGenerationGap: 17,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE = Object.freeze({
  topologyCreated: 1 << 0, topologyRetired: 1 << 1, resolutionChanged: 1 << 2,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, totalWords: 3,
  phase: 4, candidateGeneration: 5, acceptedGeneration: 6,
  topologyGeneration: 7,
  fault: 8, firstFaultFamily: 9, firstFaultId: 10,
  expectedProducerReceipts: 11, coveredProducerReceipts: 12,
  cellExecutionCount: 13, changedBrickCount: 14, causeMask: 15,
  candidateWriteCount: 16, dirtyLeafCount: 17,
  acceptedChangedBrickCount: 18, acceptedCellExecutionCount: 19,
  acceptedBrickDirtyLeafCount: 20,
} as const);

export interface SparseCM12PressureTopologyRepairFamilyLayout {
  readonly capacity: number;
  readonly candidateGenerationBaseWords: number;
  /** Compact, generation-local IDs appended directly by topology publishers. */
  readonly changedBrickListBaseWords: number;
  /** One generation stamp per 256-brick leaf, used only for bounded dedup/census. */
  readonly dirtyLeafStampBaseWords: number;
  readonly leafCount: number;
}

export interface SparseCM12PressureTopologyRepairLayout {
  readonly baseWords: number;
  readonly brickCapacity: number;
  readonly brick: SparseCM12PressureTopologyRepairFamilyLayout;
  readonly brickOldStateBaseWords: number;
  readonly brickNewStateBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const alignWords = (value: number): number => Math.ceil(value
  / SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_ALIGNMENT_WORDS)
  * SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_ALIGNMENT_WORDS;

const checkedCapacity = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0x3fff_ffff) {
    throw new RangeError(`${label} must be an integer in [1, 2^30)`);
  }
  return value;
};

export function createSparseCM12PressureTopologyRepairLayout(options: {
  readonly brickCapacity: number;
  readonly baseWords?: number;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly presentationPageResolution?: 4 | 8 | 16;
}): SparseCM12PressureTopologyRepairLayout {
  const brickFineResolution = options.brickFineResolution ?? 8;
  const presentationPageResolution = options.presentationPageResolution ?? brickFineResolution;
  if ((brickFineResolution !== 4 && brickFineResolution !== 8 && brickFineResolution !== 16)
    || presentationPageResolution !== brickFineResolution) {
    throw new Error("PTR1 requires a matched B4/P4, B8/P8, or B16/P16 physical ABI");
  }
  const brickCapacity = checkedCapacity(options.brickCapacity, "PTR1 brickCapacity");
  const baseWords = alignWords(options.baseWords ?? 0);
  let at = alignWords(baseWords + SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS);
  const leafCount = Math.ceil(brickCapacity / SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS);
  const candidateGenerationBaseWords = at; at = alignWords(at + brickCapacity);
  const changedBrickListBaseWords = at; at = alignWords(at + brickCapacity);
  const dirtyLeafStampBaseWords = at; at = alignWords(at + leafCount);
  const brickOldStateBaseWords = at; at = alignWords(at + brickCapacity);
  const brickNewStateBaseWords = at; at = alignWords(at + brickCapacity);
  const totalWords = alignWords(at);
  const brick = Object.freeze({ capacity: brickCapacity,
    candidateGenerationBaseWords, changedBrickListBaseWords,
    dirtyLeafStampBaseWords, leafCount });
  return Object.freeze({ baseWords, brickCapacity, brick,
    brickOldStateBaseWords, brickNewStateBaseWords,
    totalWords, totalBytes: 4 * totalWords });
}

export function initializeSparseCM12PressureTopologyRepairWords(
  words: Uint32Array,
  layout: SparseCM12PressureTopologyRepairLayout,
): void {
  if (words.length < layout.totalWords) throw new RangeError("PTR1 target is smaller than layout");
  const h = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER;
  const base = layout.baseWords;
  words[base + h.magic] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC;
  words[base + h.version] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION;
  words[base + h.headerWords] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS;
  words[base + h.totalWords] = layout.totalWords;
  words[base + h.phase] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.uninitialized;
  words[base + h.firstFaultFamily] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID;
  words[base + h.firstFaultId] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID;
}

export function createSparseCM12PressureTopologyRepairInitialWords(
  layout: SparseCM12PressureTopologyRepairLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords);
  initializeSparseCM12PressureTopologyRepairWords(words, layout);
  return words;
}

export function sparseCM12PressureTopologyRepairEntryPoints(
  _layout: SparseCM12PressureTopologyRepairLayout,
): readonly string[] {
  return Object.freeze([
    "beginSparseCM12PressureTopologyRepair",
    "finalizeSparseCM12PressureTopologyBrickFrontier",
    "finalizeSparseCM12BoundedPressureTopologyRepair",
  ]);
}
