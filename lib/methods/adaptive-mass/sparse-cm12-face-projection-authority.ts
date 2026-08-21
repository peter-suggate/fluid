/** Exact persistent face-projection work authority. Face preparation is owned
 * independently by contiguous brick row intervals and has no FPA state. */
export const SPARSE_CM12_FACE_PROJECTION_MAGIC = 0x4650_4131; // FPA1
export const SPARSE_CM12_FACE_PROJECTION_VERSION = 3;
export const SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS = 32;
export const SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS = 32;
export const SPARSE_CM12_FACE_PROJECTION_STAGE_COUNT = 1;
export const SPARSE_CM12_FACE_PROJECTION_LEAF_BITS = 256;
export const SPARSE_CM12_FACE_PROJECTION_BRANCH = 32;
export const SPARSE_CM12_FACE_PROJECTION_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_FACE_PROJECTION_INVALID = 0xffff_ffff;

export const SPARSE_CM12_FACE_PROJECTION_STAGE = Object.freeze({
  projection: 0,
} as const);

export const SPARSE_CM12_FACE_PROJECTION_PHASE = Object.freeze({
  uninitialized: 0,
  accepted: 1,
  collecting: 2,
  repairing: 3,
  executing: 4,
  fault: 5,
} as const);

export const SPARSE_CM12_FACE_PROJECTION_FAULT = Object.freeze({
  none: 0,
  invalidHeader: 1,
  invalidPhase: 2,
  generationExhausted: 3,
  invalidRow: 4,
  invalidCell: 5,
  dirtyCapacity: 6,
  activeLeafCapacity: 7,
  producerCoverageGap: 8,
  executionCoverageGap: 9,
  dependencyGenerationGap: 10,
  topologyGap: 11,
  pcmGenerationGap: 12,
  atomicContention: 13,
} as const);

export const SPARSE_CM12_FACE_PROJECTION_CAUSE = Object.freeze({
  velocityBits: 1 << 0,
  densityPhaseBits: 1 << 1,
  sourceFaceBits: 1 << 2,
  characteristicPolicy: 1 << 3,
  cflSpan: 1 << 4,
  topology: 1 << 5,
  movingSolid: 1 << 6,
  boundary: 1 << 7,
  force: 1 << 8,
  pressureBits: 1 << 9,
  thetaOrCoefficient: 1 << 10,
  pcmMembership: 1 << 11,
  preparedFaceBits: 1 << 12,
  closure: 1 << 13,
  bootstrap: 1 << 14,
  qaOracle: 1 << 15,
} as const);

export const SPARSE_CM12_FACE_PROJECTION_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, stageHeaderWords: 3,
  stageCount: 4, leafBits: 5, branch: 6, rowCapacity: 7,
  cellCapacity: 8, preparationHeaderBase: 9, projectionHeaderBase: 10,
  totalWords: 11, flags: 12, firstFaultStage: 13,
  brickFineResolution: 14, presentationPageResolution: 15,
  preparedAuthorityBase: 16,
  reserved0: 17, reserved1: 18, reserved2: 19, reserved3: 20,
} as const);

export const SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER = Object.freeze({
  phase: 0, acceptedGeneration: 1, candidateGeneration: 2,
  frameGeneration: 3, topologyGeneration: 4, pcmGeneration: 5,
  sourceParity: 6, policyBits: 7, fault: 8, firstFaultRow: 9,
  expectedProducerReceipts: 10, coveredProducerReceipts: 11,
  directWriteCount: 12, closureWriteCount: 13, causeMask: 14,
  dirtyLeafCount: 15, previousActiveLeafCount: 16, activeLeafCount: 17,
  workCount: 18, reusedCount: 19, executedCount: 20,
  repairIndirectX: 21, repairIndirectY: 22, repairIndirectZ: 23,
  workIndirectX: 24, workIndirectY: 25, workIndirectZ: 26,
  verifiedLeafCount: 27, reserved0: 28, reserved1: 29,
  reserved2: 30, reserved3: 31,
} as const);

export interface SparseCM12FaceProjectionStageLayout {
  readonly headerBaseWords: number;
  readonly activeBitsBaseWords: number;
  readonly activeBitWordCount: number;
  readonly candidateGenerationBaseWords: number;
  readonly candidateCauseBaseWords: number;
  readonly candidateDepthBaseWords: number;
  readonly candidateDependencyGenerationBaseWords: number;
  readonly acceptedDependencyGenerationBaseWords: number;
  readonly executionGenerationBaseWords: number;
  readonly dirtyLeafStampBaseWords: number;
  readonly dirtyLeafListBaseWords: number;
  readonly activeLeafListBaseWords: number;
  readonly leafCount: number;
  readonly treeLevelBaseWords: readonly number[];
  readonly treeLevelCounts: readonly number[];
}

export interface SparseCM12FaceProjectionAuthorityLayout {
  readonly baseWords: number;
  readonly brickFineResolution: 4 | 8 | 16;
  readonly presentationPageResolution: 4 | 8 | 16;
  readonly rowCapacity: number;
  readonly cellCapacity: number;
  /** Exact accepted pressure bits used to root projection incidence changes. */
  readonly acceptedPressureBitsBaseWords: number;
  readonly projection: SparseCM12FaceProjectionStageLayout;
  readonly totalWords: number;
  readonly totalBytes: number;
  readonly qaFullOracle: boolean;
}

const alignWords = (value: number): number => Math.ceil(value
  / SPARSE_CM12_FACE_PROJECTION_ALIGNMENT_WORDS)
  * SPARSE_CM12_FACE_PROJECTION_ALIGNMENT_WORDS;

const capacity = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0x3fff_ffff) {
    throw new RangeError(`${label} must be an integer in [1, 2^30)`);
  }
  return value;
};

export function createSparseCM12FaceProjectionAuthorityLayout(options: {
  readonly rowCapacity: number;
  readonly cellCapacity: number;
  readonly baseWords?: number;
  /** Construction-only test specialization. It is compiled into WGSL too. */
  readonly qaFullOracle?: boolean;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly presentationPageResolution?: 4 | 8 | 16;
}): SparseCM12FaceProjectionAuthorityLayout {
  const rowCapacity = capacity(options.rowCapacity, "FPA1 rowCapacity");
  const cellCapacity = capacity(options.cellCapacity, "FPA1 cellCapacity");
  const brickFineResolution = options.brickFineResolution ?? 8;
  const presentationPageResolution = options.presentationPageResolution ?? brickFineResolution;
  if ((brickFineResolution !== 4 && brickFineResolution !== 8 && brickFineResolution !== 16)
    || presentationPageResolution !== brickFineResolution) {
    throw new Error("FPA1 requires a matched B4/P4, B8/P8, or B16/P16 physical ABI");
  }
  const baseWords = alignWords(options.baseWords ?? 0);
  const projectionHeaderBaseWords = baseWords + SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS;
  let at = alignWords(projectionHeaderBaseWords
    + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS);
  const acceptedPressureBitsBaseWords = at;
  at = alignWords(at + cellCapacity);
  const makeStage = (headerBaseWords: number): SparseCM12FaceProjectionStageLayout => {
    const activeBitWordCount = Math.ceil(rowCapacity / 32);
    const leafCount = Math.ceil(rowCapacity / SPARSE_CM12_FACE_PROJECTION_LEAF_BITS);
    const activeBitsBaseWords = at; at = alignWords(at + activeBitWordCount);
    const candidateGenerationBaseWords = at; at = alignWords(at + rowCapacity);
    const candidateCauseBaseWords = at; at = alignWords(at + rowCapacity);
    const candidateDepthBaseWords = at; at = alignWords(at + rowCapacity);
    const candidateDependencyGenerationBaseWords = at; at = alignWords(at + rowCapacity);
    const acceptedDependencyGenerationBaseWords = at; at = alignWords(at + rowCapacity);
    const executionGenerationBaseWords = at; at = alignWords(at + rowCapacity);
    const dirtyLeafStampBaseWords = at; at = alignWords(at + leafCount);
    const dirtyLeafListBaseWords = at; at = alignWords(at + leafCount);
    const activeLeafListBaseWords = at; at = alignWords(at + leafCount);
    const treeLevelBaseWords: number[] = [];
    const treeLevelCounts: number[] = [];
    for (let count = leafCount;; count = Math.ceil(count
      / SPARSE_CM12_FACE_PROJECTION_BRANCH)) {
      treeLevelBaseWords.push(at); treeLevelCounts.push(count);
      at = alignWords(at + count);
      if (count === 1) break;
    }
    return Object.freeze({ headerBaseWords, activeBitsBaseWords, activeBitWordCount,
      candidateGenerationBaseWords, candidateCauseBaseWords, candidateDepthBaseWords,
      candidateDependencyGenerationBaseWords,
      acceptedDependencyGenerationBaseWords, executionGenerationBaseWords,
      dirtyLeafStampBaseWords, dirtyLeafListBaseWords, activeLeafListBaseWords,
      leafCount, treeLevelBaseWords: Object.freeze(treeLevelBaseWords),
      treeLevelCounts: Object.freeze(treeLevelCounts) });
  };
  const projection = makeStage(projectionHeaderBaseWords);
  const totalWords = alignWords(at);
  return Object.freeze({ baseWords, brickFineResolution, presentationPageResolution,
    rowCapacity, cellCapacity, acceptedPressureBitsBaseWords, projection,
    totalWords, totalBytes: 4 * totalWords, qaFullOracle: options.qaFullOracle ?? false });
}

export function initializeSparseCM12FaceProjectionAuthorityWords(
  words: Uint32Array,
  layout: SparseCM12FaceProjectionAuthorityLayout,
): void {
  if (words.length < layout.totalWords) throw new RangeError("FPA1 target is smaller than layout");
  const h = SPARSE_CM12_FACE_PROJECTION_HEADER;
  const base = layout.baseWords;
  words[base + h.magic] = SPARSE_CM12_FACE_PROJECTION_MAGIC;
  words[base + h.version] = SPARSE_CM12_FACE_PROJECTION_VERSION;
  words[base + h.headerWords] = SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS;
  words[base + h.stageHeaderWords] = SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS;
  words[base + h.stageCount] = SPARSE_CM12_FACE_PROJECTION_STAGE_COUNT;
  words[base + h.leafBits] = SPARSE_CM12_FACE_PROJECTION_LEAF_BITS;
  words[base + h.branch] = SPARSE_CM12_FACE_PROJECTION_BRANCH;
  words[base + h.rowCapacity] = layout.rowCapacity;
  words[base + h.cellCapacity] = layout.cellCapacity;
  words[base + h.preparationHeaderBase] = 0;
  words[base + h.projectionHeaderBase] = layout.projection.headerBaseWords;
  words[base + h.totalWords] = layout.totalWords;
  words[base + h.flags] = layout.qaFullOracle ? 1 : 0;
  words[base + h.firstFaultStage] = SPARSE_CM12_FACE_PROJECTION_INVALID;
  words[base + h.brickFineResolution] = layout.brickFineResolution;
  words[base + h.presentationPageResolution] = layout.presentationPageResolution;
  words[base + h.preparedAuthorityBase] = 0;
  words[base + h.reserved0] = layout.acceptedPressureBitsBaseWords;
  words[base + h.reserved1] = 0;
  const d = SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER;
  for (const stage of [layout.projection]) {
    words[stage.headerBaseWords + d.phase] = SPARSE_CM12_FACE_PROJECTION_PHASE.uninitialized;
    words[stage.headerBaseWords + d.firstFaultRow] = SPARSE_CM12_FACE_PROJECTION_INVALID;
    words[stage.headerBaseWords + d.repairIndirectY] = 1;
    words[stage.headerBaseWords + d.repairIndirectZ] = 1;
    words[stage.headerBaseWords + d.workIndirectY] = 1;
    words[stage.headerBaseWords + d.workIndirectZ] = 1;
  }
}

export function createSparseCM12FaceProjectionAuthorityInitialWords(
  layout: SparseCM12FaceProjectionAuthorityLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords);
  initializeSparseCM12FaceProjectionAuthorityWords(words, layout);
  return words;
}

export type SparseCM12FaceProjectionStageName = "projection";

export function sparseCM12FaceProjectionIndirectByteOffset(
  layout: SparseCM12FaceProjectionAuthorityLayout,
  stage: SparseCM12FaceProjectionStageName,
  kind: "repair" | "work",
): number {
  const d = SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER;
  return 4 * (layout[stage].headerBaseWords
    + (kind === "repair" ? d.repairIndirectX : d.workIndirectX));
}

/** GPU-authored zero/full triplet used only by bootstrap or the compiled QA oracle. */
export function sparseCM12FaceProjectionBootstrapIndirectByteOffset(
  layout: SparseCM12FaceProjectionAuthorityLayout,
  stage: SparseCM12FaceProjectionStageName,
): number {
  return 4 * (layout[stage].headerBaseWords
    + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER.reserved0);
}

export interface SparseCM12FaceProjectionAuthoritySource {
  readonly kind: "sparse-cm12-face-projection-authority";
  readonly arena: GPUBufferBinding;
  readonly projectionBootstrapIndirect: GPUBufferBinding;
  readonly projectionRepairIndirect: GPUBufferBinding;
  readonly projectionWorkIndirect: GPUBufferBinding;
  readonly layout: SparseCM12FaceProjectionAuthorityLayout;
}
