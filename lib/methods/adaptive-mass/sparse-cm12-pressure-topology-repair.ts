/** PTR1: bounded parallel pressure topology repair authority. */
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC = 0x5054_5231; // PTR1
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION = 1;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS = 39;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER_WORDS = 19;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS = 256;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH = 32;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID = 0xffff_ffff;

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE = Object.freeze({
  uninitialized: 0, accepted: 1, collecting: 2, repairingBricks: 3,
  executingCells: 4,
  awaitingAcceptance: 7, fault: 8,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, invalidPhase: 2, generationExhausted: 3,
  invalidCellRange: 6,
  dirtyLeafCapacity: 9,
  producerCoverageGap: 10, brickRepairGap: 11, cellExecutionGap: 12,
  topologyGenerationGap: 15,
  pcmGenerationGap: 16, coefficientGenerationGap: 17,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE = Object.freeze({
  topologyCreated: 1 << 0, topologyRetired: 1 << 1, resolutionChanged: 1 << 2,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, familyHeaderWords: 3,
  familyCount: 4, leafBits: 5, branch: 6, brickCapacity: 7,
  brickHeaderBase: 8, totalWords: 9,
  phase: 10, candidateGeneration: 11, acceptedGeneration: 12,
  frameGeneration: 13, topologyGeneration: 14,
  pcmCellGeneration: 15, pcmRowGeneration: 16, coefficientGeneration: 17,
  fault: 18, firstFaultFamily: 19, firstFaultId: 20,
  expectedProducerReceipts: 21, coveredProducerReceipts: 22,
  cellExecutionCount: 23, changedBrickCount: 24, causeMask: 25,
  brickOldStateBase: 26, brickNewStateBase: 27, brickCauseBase: 28,
  commitIndirectX: 29, commitIndirectY: 30, commitIndirectZ: 31,
  brickStateCommitCount: 32,
  brickSeedIndirectX: 33, brickSeedIndirectY: 34, brickSeedIndirectZ: 35,
  acceptedChangedBrickCount: 36,
  acceptedCellExecutionCount: 37,
  acceptedBrickDirtyLeafCount: 38,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER = Object.freeze({
  capacity: 0, candidateWriteCount: 1,
  dirtyLeafCount: 2, previousActiveLeafCount: 3, activeLeafCount: 4,
  workCount: 5, repairedLeafCount: 6,
  repairIndirectX: 7, repairIndirectY: 8, repairIndirectZ: 9,
  workIndirectX: 10, workIndirectY: 11, workIndirectZ: 12,
  candidateGenerationBase: 13, bitsBase: 14, dirtyLeafStampBase: 15,
  dirtyLeafListBase: 16, activeLeafListBase: 17, treeLevelCount: 18,
} as const);

export interface SparseCM12PressureTopologyRepairFamilyLayout {
  readonly headerBaseWords: number;
  readonly capacity: number;
  readonly candidateGenerationBaseWords: number;
  readonly activeBitsBaseWords: number;
  readonly activeBitWordCount: number;
  readonly dirtyLeafStampBaseWords: number;
  readonly dirtyLeafListBaseWords: number;
  readonly activeLeafListBaseWords: number;
  readonly leafCount: number;
  readonly treeLevelBaseWords: readonly number[];
  readonly treeLevelCounts: readonly number[];
}

export interface SparseCM12PressureTopologyRepairLayout {
  readonly baseWords: number;
  readonly brickCapacity: number;
  readonly brick: SparseCM12PressureTopologyRepairFamilyLayout;
  readonly brickOldStateBaseWords: number;
  readonly brickNewStateBaseWords: number;
  readonly brickCauseBaseWords: number;
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
  const brickHeaderBaseWords = baseWords + SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS;
  let at = alignWords(brickHeaderBaseWords
    + SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER_WORDS);
  const makeFamily = (capacity: number): SparseCM12PressureTopologyRepairFamilyLayout => {
    const leafCount = Math.ceil(capacity / SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS);
    const candidateGenerationBaseWords = at; at = alignWords(at + capacity);
    const activeBitWordCount = Math.ceil(capacity / 32);
    const activeBitsBaseWords = at; at = alignWords(at + activeBitWordCount);
    const dirtyLeafStampBaseWords = at; at = alignWords(at + leafCount);
    const dirtyLeafListBaseWords = at; at = alignWords(at + leafCount);
    const activeLeafListBaseWords = at; at = alignWords(at + leafCount);
    const treeLevelBaseWords: number[] = [], treeLevelCounts: number[] = [];
    for (let count = leafCount;; count = Math.ceil(count
      / SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH)) {
      treeLevelBaseWords.push(at); treeLevelCounts.push(count); at = alignWords(at + count);
      if (count === 1) break;
    }
    return Object.freeze({ headerBaseWords: brickHeaderBaseWords, capacity,
      candidateGenerationBaseWords,
      activeBitsBaseWords, activeBitWordCount, dirtyLeafStampBaseWords,
      dirtyLeafListBaseWords, activeLeafListBaseWords,
      leafCount, treeLevelBaseWords: Object.freeze(treeLevelBaseWords),
      treeLevelCounts: Object.freeze(treeLevelCounts) });
  };
  const brick = makeFamily(brickCapacity);
  const brickOldStateBaseWords = at; at = alignWords(at + brickCapacity);
  const brickNewStateBaseWords = at; at = alignWords(at + brickCapacity);
  const brickCauseBaseWords = at; at = alignWords(at + brickCapacity);
  const totalWords = alignWords(at);
  return Object.freeze({ baseWords, brickCapacity, brick,
    brickOldStateBaseWords, brickNewStateBaseWords, brickCauseBaseWords,
    totalWords, totalBytes: 4 * totalWords });
}

export function initializeSparseCM12PressureTopologyRepairWords(
  words: Uint32Array,
  layout: SparseCM12PressureTopologyRepairLayout,
): void {
  if (words.length < layout.totalWords) throw new RangeError("PTR1 target is smaller than layout");
  const h = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER;
  const f = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER;
  const base = layout.baseWords;
  words[base + h.magic] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC;
  words[base + h.version] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION;
  words[base + h.headerWords] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS;
  words[base + h.familyHeaderWords] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER_WORDS;
  words[base + h.familyCount] = 1;
  words[base + h.leafBits] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS;
  words[base + h.branch] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH;
  words[base + h.brickCapacity] = layout.brickCapacity;
  words[base + h.brickHeaderBase] = layout.brick.headerBaseWords;
  words[base + h.totalWords] = layout.totalWords;
  words[base + h.phase] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.uninitialized;
  words[base + h.firstFaultFamily] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID;
  words[base + h.firstFaultId] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID;
  words[base + h.brickOldStateBase] = layout.brickOldStateBaseWords;
  words[base + h.brickNewStateBase] = layout.brickNewStateBaseWords;
  words[base + h.brickCauseBase] = layout.brickCauseBaseWords;
  words[base + h.commitIndirectY] = 1; words[base + h.commitIndirectZ] = 1;
  words[base + h.brickSeedIndirectY] = 1; words[base + h.brickSeedIndirectZ] = 1;
  for (const family of [layout.brick]) {
    const header = family.headerBaseWords;
    words[header + f.capacity] = family.capacity;
    words[header + f.repairIndirectY] = 1; words[header + f.repairIndirectZ] = 1;
    words[header + f.workIndirectY] = 1; words[header + f.workIndirectZ] = 1;
    words[header + f.candidateGenerationBase] = family.candidateGenerationBaseWords;
    words[header + f.bitsBase] = family.activeBitsBaseWords;
    words[header + f.dirtyLeafStampBase] = family.dirtyLeafStampBaseWords;
    words[header + f.dirtyLeafListBase] = family.dirtyLeafListBaseWords;
    words[header + f.activeLeafListBase] = family.activeLeafListBaseWords;
    words[header + f.treeLevelCount] = family.treeLevelCounts.length;
  }
}

export function createSparseCM12PressureTopologyRepairInitialWords(
  layout: SparseCM12PressureTopologyRepairLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords);
  initializeSparseCM12PressureTopologyRepairWords(words, layout);
  return words;
}

export function sparseCM12PressureTopologyRepairIndirectByteOffset(
  layout: SparseCM12PressureTopologyRepairLayout,
  kind: "repair" | "work",
): number {
  const f = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER;
  return 4 * (layout.brick.headerBaseWords
    + (kind === "repair" ? f.repairIndirectX : f.workIndirectX));
}

export function sparseCM12PressureTopologyRepairHeaderIndirectByteOffset(
  layout: SparseCM12PressureTopologyRepairLayout,
  kind: "commit" | "brickSeed",
): number {
  const h = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER;
  const field = kind === "commit" ? h.commitIndirectX
    : h.brickSeedIndirectX;
  return 4 * (layout.baseWords + field);
}

export function sparseCM12PressureTopologyRepairEntryPoints(
  layout: SparseCM12PressureTopologyRepairLayout,
): readonly string[] {
  const reductions = layout.brick.treeLevelCounts.slice(1).map((_, level) =>
    `reduceSparseCM12PressureTopologyBrickLevel${level + 1}`);
  return Object.freeze([
    "beginSparseCM12PressureTopologyRepair",
    "captureSparseCM12PressureTopologyConsumerGenerations",
    "seedPreviousSparseCM12PressureTopologyBrickLeaves",
    "finalizeSparseCM12PressureTopologyBrickFrontier",
    "repairSparseCM12PressureTopologyBrickLeaves",
    ...reductions,
    "finalizeSparseCM12PressureTopologyBrickPlan",
    "repairSparseCM12PressureTopologyChangedBricks",
    "finalizeSparseCM12PressureTopologyCellExecution",
    "sealSparseCM12PressureTopologyRowImage",
    "commitSparseCM12PressureTopologyBrickStates",
    "finalizeSparseCM12BoundedPressureTopologyRepair",
  ]);
}
