/** PTR1: bounded parallel pressure topology repair authority. */
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC = 0x5054_5231; // PTR1
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION = 1;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS = 64;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER_WORDS = 24;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS = 256;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH = 32;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID = 0xffff_ffff;

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY = Object.freeze({
  brick: 0, row: 1,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE = Object.freeze({
  uninitialized: 0, accepted: 1, collecting: 2, repairingBricks: 3,
  executingCells: 4, repairingRows: 5, executingRows: 6,
  awaitingAcceptance: 7, fault: 8,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, invalidPhase: 2, generationExhausted: 3,
  invalidBrick: 4, conflictingBrickRecord: 5, invalidCellRange: 6,
  invalidCell: 7, invalidRow: 8, dirtyLeafCapacity: 9,
  producerCoverageGap: 10, brickRepairGap: 11, cellExecutionGap: 12,
  rowRepairGap: 13, rowExecutionGap: 14, topologyGenerationGap: 15,
  pcmGenerationGap: 16, pcfGenerationGap: 17,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE = Object.freeze({
  topologyCreated: 1 << 0, topologyRetired: 1 << 1, resolutionChanged: 1 << 2,
  identityChanged: 1 << 3, incidenceClosure: 1 << 4,
  pcmMembership: 1 << 5, pcfCoefficient: 1 << 6,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, familyHeaderWords: 3,
  familyCount: 4, leafBits: 5, branch: 6, brickCapacity: 7, rowCapacity: 8,
  brickHeaderBase: 9, rowHeaderBase: 10, totalWords: 11, flags: 12,
  phase: 13, candidateGeneration: 14, acceptedGeneration: 15,
  frameGeneration: 16, topologyGeneration: 17,
  pcmCellGeneration: 18, pcmRowGeneration: 19, pcfGeneration: 20,
  fault: 21, firstFaultFamily: 22, firstFaultId: 23,
  expectedProducerReceipts: 24, coveredProducerReceipts: 25,
  cellExecutionCount: 26, rowExecutionCount: 27,
  changedBrickCount: 28, changedRowCount: 29, causeMask: 30,
  brickOldStateBase: 31, brickNewStateBase: 32, brickCauseBase: 33,
  brickAcceptedStateBase: 34,
  commitIndirectX: 35, commitIndirectY: 36, commitIndirectZ: 37,
  brickStateCommitCount: 38,
  brickSeedIndirectX: 39, brickSeedIndirectY: 40, brickSeedIndirectZ: 41,
  rowSeedIndirectX: 42, rowSeedIndirectY: 43, rowSeedIndirectZ: 44,
  acceptedChangedBrickCount: 45, acceptedChangedRowCount: 46,
  acceptedCellExecutionCount: 47, acceptedRowExecutionCount: 48,
  acceptedBrickDirtyLeafCount: 49, acceptedRowDirtyLeafCount: 50,
} as const);

export const SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER = Object.freeze({
  capacity: 0, candidateWriteCount: 1, closureWriteCount: 2,
  dirtyLeafCount: 3, previousActiveLeafCount: 4, activeLeafCount: 5,
  workCount: 6, repairedLeafCount: 7, executedCount: 8, causeMask: 9,
  repairIndirectX: 10, repairIndirectY: 11, repairIndirectZ: 12,
  workIndirectX: 13, workIndirectY: 14, workIndirectZ: 15,
  candidateGenerationBase: 16, bitsBase: 17, dirtyLeafStampBase: 18,
  dirtyLeafListBase: 19, activeLeafListBase: 20, treeLevelCount: 21,
  executionGenerationBase: 22, reserved: 23,
} as const);

export type SparseCM12PressureTopologyRepairFamilyName = "brick" | "row";

export interface SparseCM12PressureTopologyRepairFamilyLayout {
  readonly headerBaseWords: number;
  readonly capacity: number;
  readonly candidateGenerationBaseWords: number;
  readonly activeBitsBaseWords: number;
  readonly activeBitWordCount: number;
  readonly dirtyLeafStampBaseWords: number;
  readonly dirtyLeafListBaseWords: number;
  readonly activeLeafListBaseWords: number;
  readonly executionGenerationBaseWords: number;
  readonly leafCount: number;
  readonly treeLevelBaseWords: readonly number[];
  readonly treeLevelCounts: readonly number[];
}

export interface SparseCM12PressureTopologyRepairLayout {
  readonly baseWords: number;
  readonly brickCapacity: number;
  readonly rowCapacity: number;
  readonly brick: SparseCM12PressureTopologyRepairFamilyLayout;
  readonly row: SparseCM12PressureTopologyRepairFamilyLayout;
  readonly brickOldStateBaseWords: number;
  readonly brickNewStateBaseWords: number;
  readonly brickCauseBaseWords: number;
  readonly brickAcceptedStateBaseWords: number;
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
  readonly rowCapacity: number;
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
  const rowCapacity = checkedCapacity(options.rowCapacity, "PTR1 rowCapacity");
  const baseWords = alignWords(options.baseWords ?? 0);
  const brickHeaderBaseWords = baseWords + SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS;
  const rowHeaderBaseWords = brickHeaderBaseWords
    + SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER_WORDS;
  let at = alignWords(rowHeaderBaseWords
    + SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER_WORDS);
  const makeFamily = (headerBaseWords: number,
    capacity: number): SparseCM12PressureTopologyRepairFamilyLayout => {
    const leafCount = Math.ceil(capacity / SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS);
    const candidateGenerationBaseWords = at; at = alignWords(at + capacity);
    const activeBitWordCount = Math.ceil(capacity / 32);
    const activeBitsBaseWords = at; at = alignWords(at + activeBitWordCount);
    const dirtyLeafStampBaseWords = at; at = alignWords(at + leafCount);
    const dirtyLeafListBaseWords = at; at = alignWords(at + leafCount);
    const activeLeafListBaseWords = at; at = alignWords(at + leafCount);
    const executionGenerationBaseWords = at; at = alignWords(at + capacity);
    const treeLevelBaseWords: number[] = [], treeLevelCounts: number[] = [];
    for (let count = leafCount;; count = Math.ceil(count
      / SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH)) {
      treeLevelBaseWords.push(at); treeLevelCounts.push(count); at = alignWords(at + count);
      if (count === 1) break;
    }
    return Object.freeze({ headerBaseWords, capacity, candidateGenerationBaseWords,
      activeBitsBaseWords, activeBitWordCount, dirtyLeafStampBaseWords,
      dirtyLeafListBaseWords, activeLeafListBaseWords, executionGenerationBaseWords,
      leafCount, treeLevelBaseWords: Object.freeze(treeLevelBaseWords),
      treeLevelCounts: Object.freeze(treeLevelCounts) });
  };
  const brick = makeFamily(brickHeaderBaseWords, brickCapacity);
  const row = makeFamily(rowHeaderBaseWords, rowCapacity);
  const brickOldStateBaseWords = at; at = alignWords(at + brickCapacity);
  const brickNewStateBaseWords = at; at = alignWords(at + brickCapacity);
  const brickCauseBaseWords = at; at = alignWords(at + brickCapacity);
  const brickAcceptedStateBaseWords = at; at = alignWords(at + brickCapacity);
  const totalWords = alignWords(at);
  return Object.freeze({ baseWords, brickCapacity, rowCapacity, brick, row,
    brickOldStateBaseWords, brickNewStateBaseWords, brickCauseBaseWords,
    brickAcceptedStateBaseWords, totalWords, totalBytes: 4 * totalWords });
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
  words[base + h.familyCount] = 2;
  words[base + h.leafBits] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS;
  words[base + h.branch] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH;
  words[base + h.brickCapacity] = layout.brickCapacity;
  words[base + h.rowCapacity] = layout.rowCapacity;
  words[base + h.brickHeaderBase] = layout.brick.headerBaseWords;
  words[base + h.rowHeaderBase] = layout.row.headerBaseWords;
  words[base + h.totalWords] = layout.totalWords;
  words[base + h.phase] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.uninitialized;
  words[base + h.firstFaultFamily] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID;
  words[base + h.firstFaultId] = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID;
  words[base + h.brickOldStateBase] = layout.brickOldStateBaseWords;
  words[base + h.brickNewStateBase] = layout.brickNewStateBaseWords;
  words[base + h.brickCauseBase] = layout.brickCauseBaseWords;
  words[base + h.brickAcceptedStateBase] = layout.brickAcceptedStateBaseWords;
  words[base + h.commitIndirectY] = 1; words[base + h.commitIndirectZ] = 1;
  words[base + h.brickSeedIndirectY] = 1; words[base + h.brickSeedIndirectZ] = 1;
  words[base + h.rowSeedIndirectY] = 1; words[base + h.rowSeedIndirectZ] = 1;
  for (const family of [layout.brick, layout.row]) {
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
    words[header + f.executionGenerationBase] = family.executionGenerationBaseWords;
  }
  words.fill(SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID,
    layout.brickAcceptedStateBaseWords,
    layout.brickAcceptedStateBaseWords + layout.brickCapacity);
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
  family: SparseCM12PressureTopologyRepairFamilyName,
  kind: "repair" | "work",
): number {
  const f = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER;
  return 4 * (layout[family].headerBaseWords
    + (kind === "repair" ? f.repairIndirectX : f.workIndirectX));
}

export function sparseCM12PressureTopologyRepairHeaderIndirectByteOffset(
  layout: SparseCM12PressureTopologyRepairLayout,
  kind: "commit" | "brickSeed" | "rowSeed",
): number {
  const h = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER;
  const field = kind === "commit" ? h.commitIndirectX
    : kind === "brickSeed" ? h.brickSeedIndirectX : h.rowSeedIndirectX;
  return 4 * (layout.baseWords + field);
}

export function sparseCM12PressureTopologyRepairEntryPoints(
  layout: SparseCM12PressureTopologyRepairLayout,
): readonly string[] {
  const reductions = (family: "Brick" | "Row") => layout[
    family === "Brick" ? "brick" : "row"
  ].treeLevelCounts.slice(1).map((_, level) =>
    `reduceSparseCM12PressureTopology${family}Level${level + 1}`);
  return Object.freeze([
    "beginSparseCM12PressureTopologyRepair",
    "captureSparseCM12PressureTopologyConsumerGenerations",
    "importSparseCM12PressureTopologyChangedBricks",
    "seedPreviousSparseCM12PressureTopologyBrickLeaves",
    "seedPreviousSparseCM12PressureTopologyRowLeaves",
    "finalizeSparseCM12PressureTopologyBrickFrontier",
    "repairSparseCM12PressureTopologyBrickLeaves",
    ...reductions("Brick"),
    "finalizeSparseCM12PressureTopologyBrickPlan",
    "repairSparseCM12PressureTopologyChangedBricks",
    "finalizeSparseCM12PressureTopologyCellExecution",
    "repairSparseCM12PressureTopologyRowLeaves",
    ...reductions("Row"),
    "finalizeSparseCM12PressureTopologyRowPlan",
    "repairSparseCM12PressureTopologyRows",
    "finalizeSparseCM12PressureTopologyRowExecution",
    "commitSparseCM12PressureTopologyBrickStates",
    "finalizeSparseCM12BoundedPressureTopologyRepair",
  ]);
}

export interface SparseCM12PressureTopologyRepairQARecord {
  readonly brick: number;
  readonly oldState: number;
  readonly newState: number;
  readonly cause: number;
}

export interface SparseCM12PressureTopologyRepairQAInput {
  readonly brickCapacity: number;
  readonly rowCapacity: number;
  readonly records: readonly SparseCM12PressureTopologyRepairQARecord[];
  readonly cellRange: (brick: number, state: number) => readonly [number, number];
  readonly incidences: readonly (readonly number[])[];
  readonly membership: (cell: number, newState: boolean) => boolean;
}

export interface SparseCM12PressureTopologyRepairQAReceipt {
  readonly changedBricks: readonly number[];
  readonly retiredCells: readonly number[];
  readonly currentCells: readonly number[];
  readonly dirtyRows: readonly number[];
  readonly pcfTopologyCells: readonly number[];
  readonly pcfMembershipCells: readonly number[];
}

/** CPU construction oracle for stable changed-brick/cell/incidence closure. */
export function evaluateSparseCM12PressureTopologyRepairQA(
  input: SparseCM12PressureTopologyRepairQAInput,
): SparseCM12PressureTopologyRepairQAReceipt {
  const byBrick = new Map<number, SparseCM12PressureTopologyRepairQARecord>();
  for (const record of input.records) {
    if (!Number.isSafeInteger(record.brick) || record.brick < 0
      || record.brick >= input.brickCapacity) throw new RangeError("PTR1 QA brick out of range");
    const prior = byBrick.get(record.brick);
    if (prior && (prior.oldState !== record.oldState || prior.newState !== record.newState)) {
      throw new Error("PTR1 QA conflicting changed-brick provenance");
    }
    byBrick.set(record.brick, prior ? { ...record, cause: prior.cause | record.cause } : record);
  }
  const changedBricks = [...byBrick.keys()].sort((a, b) => a - b);
  const retired = new Set<number>(), current = new Set<number>(), rows = new Set<number>();
  const pcfTopology = new Set<number>(), pcfMembership = new Set<number>();
  const addRange = (brick: number, state: number, isCurrent: boolean) => {
    if (state === SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID) return;
    const [first, count] = input.cellRange(brick, state);
    if (!Number.isSafeInteger(first) || !Number.isSafeInteger(count) || first < 0 || count < 0
      || first + count > input.incidences.length) throw new RangeError("PTR1 QA invalid cell range");
    for (let cell = first; cell < first + count; cell += 1) {
      (isCurrent ? current : retired).add(cell); pcfTopology.add(cell);
      if (input.membership(cell, isCurrent) !== input.membership(cell, !isCurrent)) {
        pcfMembership.add(cell);
      }
      for (const row of input.incidences[cell]!) {
        if (!Number.isSafeInteger(row) || row < 0 || row >= input.rowCapacity) {
          throw new RangeError("PTR1 QA incidence row out of range");
        }
        rows.add(row);
      }
    }
  };
  for (const brick of changedBricks) {
    const record = byBrick.get(brick)!;
    addRange(brick, record.oldState, false); addRange(brick, record.newState, true);
  }
  const sorted = (values: Set<number>) => Object.freeze([...values].sort((a, b) => a - b));
  return Object.freeze({ changedBricks: Object.freeze(changedBricks),
    retiredCells: sorted(retired), currentCells: sorted(current), dirtyRows: sorted(rows),
    pcfTopologyCells: sorted(pcfTopology), pcfMembershipCells: sorted(pcfMembership) });
}
