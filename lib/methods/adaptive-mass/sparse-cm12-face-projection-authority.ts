/**
 * FPA1: exact, persistent face-preparation/projection work authority.
 *
 * Stable row ids are HTP1 row ids.  The arena stores no floating-point
 * approximation and no scene predicate: a producer advances a row's local
 * dependency generation iff any bit read by the unchanged HEAD invocation
 * changes.  Absent such an event the previously mirrored final face word is
 * the exact authority for both ping-pong banks.
 */
export const SPARSE_CM12_FACE_PROJECTION_MAGIC = 0x4650_4131; // FPA1
export const SPARSE_CM12_FACE_PROJECTION_VERSION = 2;
export const SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS = 32;
export const SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS = 32;
export const SPARSE_CM12_FACE_PROJECTION_STAGE_COUNT = 2;
export const SPARSE_CM12_FACE_PROJECTION_LEAF_BITS = 256;
export const SPARSE_CM12_FACE_PROJECTION_BRANCH = 32;
export const SPARSE_CM12_FACE_PROJECTION_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_FACE_PROJECTION_INVALID = 0xffff_ffff;

export const SPARSE_CM12_FACE_PROJECTION_STAGE = Object.freeze({
  preparation: 0,
  projection: 1,
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
  preparationDependency: 1 << 12,
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
  /** Exact pre-pressure face word, separate from the mirrored final face banks. */
  readonly preparedAuthorityBaseWords: number;
  /** Persistent deep-row preparation certificate and transient support token. */
  readonly preparationCertificateBaseWords: number;
  /** Exact accepted pressure bits used to root projection incidence changes. */
  readonly acceptedPressureBitsBaseWords: number;
  readonly preparation: SparseCM12FaceProjectionStageLayout;
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
  const preparationHeaderBaseWords = baseWords + SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS;
  const projectionHeaderBaseWords = preparationHeaderBaseWords
    + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS;
  const preparedAuthorityBaseWords = alignWords(projectionHeaderBaseWords
    + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS);
  const preparationCertificateBaseWords = alignWords(
    preparedAuthorityBaseWords + rowCapacity,
  );
  const acceptedPressureBitsBaseWords = alignWords(
    preparationCertificateBaseWords + rowCapacity,
  );
  let at = alignWords(acceptedPressureBitsBaseWords + cellCapacity);
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
  const preparation = makeStage(preparationHeaderBaseWords);
  const projection = makeStage(projectionHeaderBaseWords);
  const totalWords = alignWords(at);
  return Object.freeze({ baseWords, brickFineResolution, presentationPageResolution,
    rowCapacity, cellCapacity, preparedAuthorityBaseWords, preparationCertificateBaseWords,
    acceptedPressureBitsBaseWords,
    preparation, projection,
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
  words[base + h.preparationHeaderBase] = layout.preparation.headerBaseWords;
  words[base + h.projectionHeaderBase] = layout.projection.headerBaseWords;
  words[base + h.totalWords] = layout.totalWords;
  words[base + h.flags] = layout.qaFullOracle ? 1 : 0;
  words[base + h.firstFaultStage] = SPARSE_CM12_FACE_PROJECTION_INVALID;
  words[base + h.brickFineResolution] = layout.brickFineResolution;
  words[base + h.presentationPageResolution] = layout.presentationPageResolution;
  words[base + h.preparedAuthorityBase] = layout.preparedAuthorityBaseWords;
  words[base + h.reserved0] = layout.acceptedPressureBitsBaseWords;
  words[base + h.reserved1] = layout.preparationCertificateBaseWords;
  const d = SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER;
  for (const stage of [layout.preparation, layout.projection]) {
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

export type SparseCM12FaceProjectionStageName = "preparation" | "projection";

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
  readonly preparationBootstrapIndirect: GPUBufferBinding;
  readonly projectionBootstrapIndirect: GPUBufferBinding;
  readonly preparationRepairIndirect: GPUBufferBinding;
  readonly projectionRepairIndirect: GPUBufferBinding;
  readonly preparationWorkIndirect: GPUBufferBinding;
  readonly projectionWorkIndirect: GPUBufferBinding;
  readonly layout: SparseCM12FaceProjectionAuthorityLayout;
}

export interface SparseCM12FaceProjectionQAInput {
  readonly preparationLive: Uint8Array;
  readonly projectionLive: Uint8Array;
  readonly preparationDirtyRows: readonly number[];
  readonly projectionDirtyRows: readonly number[];
  readonly acceptedFaceBits: Uint32Array;
  readonly acceptedPreparedFaceBits: Uint32Array;
  readonly prepareHead: (row: number) => number;
  readonly projectHead: (row: number, preparedBits: number) => number;
}

export interface SparseCM12FaceProjectionQAResult {
  readonly localBits: Uint32Array;
  readonly oracleBits: Uint32Array;
  readonly firstMismatchRow: number;
}

/** Construction/test-only byte oracle; never participates in frame scheduling. */
export function compareSparseCM12FaceProjectionAuthorityQA(
  input: SparseCM12FaceProjectionQAInput,
): SparseCM12FaceProjectionQAResult {
  const count = input.acceptedFaceBits.length;
  if (input.acceptedPreparedFaceBits.length !== count
    || input.preparationLive.length !== count || input.projectionLive.length !== count) {
    throw new RangeError("FPA1 QA live masks must match acceptedFaceBits");
  }
  const canonical = (rows: readonly number[], label: string): number[] => {
    const sorted = [...new Set(rows)].sort((a, b) => a - b);
    if (sorted.some((row) => !Number.isSafeInteger(row) || row < 0 || row >= count)) {
      throw new RangeError(`FPA1 QA ${label} row is out of range`);
    }
    return sorted;
  };
  const preparation = canonical(input.preparationDirtyRows, "preparation");
  const projection = canonical([...input.projectionDirtyRows, ...preparation.filter(
    (row) => input.projectionLive[row] !== 0)], "projection");
  const localBits = input.acceptedFaceBits.slice();
  const prepared = input.acceptedPreparedFaceBits.slice();
  for (const row of preparation) if (input.preparationLive[row] !== 0) {
    prepared[row] = input.prepareHead(row) >>> 0;
    if (input.projectionLive[row] === 0) localBits[row] = prepared[row]!;
  }
  for (const row of projection) if (input.projectionLive[row] !== 0) {
    localBits[row] = input.projectHead(row, prepared[row]!) >>> 0;
  }
  const oracleBits = input.acceptedFaceBits.slice();
  for (let row = 0; row < count; row += 1) {
    if (input.preparationLive[row] === 0) continue;
    const base = input.prepareHead(row) >>> 0;
    oracleBits[row] = input.projectionLive[row] !== 0
      ? input.projectHead(row, base) >>> 0 : base;
  }
  let firstMismatchRow = SPARSE_CM12_FACE_PROJECTION_INVALID;
  for (let row = 0; row < count; row += 1) if (localBits[row] !== oracleBits[row]) {
    firstMismatchRow = row; break;
  }
  return { localBits, oracleBits, firstMismatchRow };
}
