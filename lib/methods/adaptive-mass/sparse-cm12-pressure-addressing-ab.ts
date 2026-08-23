/**
 * PAB1: construction-only pressure-addressing A/B receipt.
 *
 * This ABI is intentionally absent from method options and URL state.  A QA
 * constructor may append it to an already-bound atomic arena and compile the
 * same pressure kernels twice with different override constants.  Production
 * construction does neither and there is no run-time fallback between arms.
 */
export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_MAGIC = 0x5041_4231; // PAB1
export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_VERSION = 1;
export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS = 32;
export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_INVALID = 0xffff_ffff;

export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_MODE = Object.freeze({
  canonicalRankSelect: 0,
  materializedList: 1,
} as const);
export type SparseCM12PressureAddressingABModeName = keyof
  typeof SPARSE_CM12_PRESSURE_ADDRESSING_AB_MODE;

export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE = Object.freeze({
  uninitialized: 0,
  collecting: 1,
  accepted: 2,
  fault: 3,
} as const);

export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT = Object.freeze({
  none: 0,
  invalidHeader: 1,
  invalidPhase: 2,
  generationMismatch: 3,
  countMismatch: 4,
  capacity: 5,
  invalidRankSelect: 6,
  listMismatch: 7,
  nonCanonicalOrder: 8,
} as const);

export const SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, flags: 3,
  phase: 4, fault: 5, firstFaultRank: 6,
  expectedPCMGeneration: 7, materializedPCMGeneration: 8,
  expectedCount: 9, materializedCount: 10,
  materializedExecutions: 11, verifiedExecutions: 12,
  mismatchCount: 13, firstMismatchRank: 14,
  firstExpectedCell: 15, firstActualCell: 16,
  materializedHash: 17, verifiedHash: 18,
  listCapacity: 19, listBaseWords: 20,
  materializeIndirectX: 21, materializeIndirectY: 22, materializeIndirectZ: 23,
  acceptedReceipts: 24,
  solveIndirectX: 25, solveIndirectY: 26, solveIndirectZ: 27,
} as const);

export interface SparseCM12PressureAddressingABLayout {
  readonly baseWords: number;
  readonly listBaseWords: number;
  readonly cellCapacity: number;
  readonly totalWords: number;
  readonly totalBytes: number;
  readonly constructionOnly: boolean;
  readonly productionAddressAuthority: boolean;
  readonly runtimeSelectable: false;
  readonly outputSelectable: false;
}

const align = (value: number): number => Math.ceil(value
  / SPARSE_CM12_PRESSURE_ADDRESSING_AB_ALIGNMENT_WORDS)
  * SPARSE_CM12_PRESSURE_ADDRESSING_AB_ALIGNMENT_WORDS;

export function createSparseCM12PressureAddressingABLayout(options: {
  readonly baseWords?: number;
  readonly cellCapacity: number;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly presentationPageResolution?: 4 | 8 | 16;
  /** Required literal makes accidental production allocation conspicuous. */
  readonly constructionMode: "qa-pressure-addressing-ab";
}): SparseCM12PressureAddressingABLayout {
  if (options.constructionMode !== "qa-pressure-addressing-ab") {
    throw new Error("PAB1 is available only to its explicit QA construction");
  }
  const brickFineResolution = options.brickFineResolution ?? 8;
  const presentationPageResolution = options.presentationPageResolution ?? brickFineResolution;
  if ((brickFineResolution !== 4 && brickFineResolution !== 8 && brickFineResolution !== 16)
    || presentationPageResolution !== brickFineResolution) {
    throw new Error("PAB1 requires a matched B4/P4, B8/P8, or B16/P16 QA ABI");
  }
  if (!Number.isSafeInteger(options.cellCapacity)
    || options.cellCapacity < 1 || options.cellCapacity >= 0x4000_0000) {
    throw new RangeError("PAB1 cellCapacity must be an integer in [1, 2^30)");
  }
  const baseWords = align(options.baseWords ?? 0);
  const listBaseWords = align(baseWords
    + SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS);
  const totalWords = align(listBaseWords + options.cellCapacity);
  return Object.freeze({ baseWords, listBaseWords,
    cellCapacity: options.cellCapacity, totalWords, totalBytes: 4 * totalWords,
    constructionOnly: true, productionAddressAuthority: false,
    runtimeSelectable: false, outputSelectable: false });
}

/** Immutable production PCM address list. It has no runtime selector/fallback. */
export function createSparseCM12ProductionPressureAddressingLayout(options: {
  readonly baseWords: number;
  readonly cellCapacity: number;
  readonly brickFineResolution: 4 | 8 | 16;
  readonly presentationPageResolution: 4 | 8 | 16;
}): SparseCM12PressureAddressingABLayout {
  const qa = createSparseCM12PressureAddressingABLayout({
    ...options, constructionMode: "qa-pressure-addressing-ab",
  });
  return Object.freeze({ ...qa, constructionOnly: false,
    productionAddressAuthority: true });
}

export function createSparseCM12PressureAddressingABInitialWords(
  layout: SparseCM12PressureAddressingABLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER;
  words[h.magic] = SPARSE_CM12_PRESSURE_ADDRESSING_AB_MAGIC;
  words[h.version] = SPARSE_CM12_PRESSURE_ADDRESSING_AB_VERSION;
  words[h.headerWords] = SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS;
  words[h.phase] = SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE.uninitialized;
  words[h.firstFaultRank] = SPARSE_CM12_PRESSURE_ADDRESSING_AB_INVALID;
  words[h.firstMismatchRank] = SPARSE_CM12_PRESSURE_ADDRESSING_AB_INVALID;
  words[h.firstExpectedCell] = SPARSE_CM12_PRESSURE_ADDRESSING_AB_INVALID;
  words[h.firstActualCell] = SPARSE_CM12_PRESSURE_ADDRESSING_AB_INVALID;
  words[h.listCapacity] = layout.cellCapacity;
  words[h.listBaseWords] = layout.listBaseWords;
  words[h.materializeIndirectY] = 1;
  words[h.materializeIndirectZ] = 1;
  words[h.solveIndirectY] = 1;
  words[h.solveIndirectZ] = 1;
  words.fill(SPARSE_CM12_PRESSURE_ADDRESSING_AB_INVALID,
    layout.listBaseWords - layout.baseWords,
    layout.listBaseWords - layout.baseWords + layout.cellCapacity);
  return words;
}

/** Compile constants for two separately constructed QA solvers. */
export function sparseCM12PressureAddressingABPipelineConstants(
  mode: SparseCM12PressureAddressingABModeName,
): Readonly<Record<"CM12_PRESSURE_ADDRESS_MODE", number>> {
  return Object.freeze({
    CM12_PRESSURE_ADDRESS_MODE: SPARSE_CM12_PRESSURE_ADDRESSING_AB_MODE[mode],
  });
}

export interface SparseCM12PressureAddressingABReceipt {
  readonly phase: number;
  readonly fault: number;
  readonly firstFaultRank: number;
  readonly expectedPCMGeneration: number;
  readonly materializedPCMGeneration: number;
  readonly expectedCount: number;
  readonly materializedCount: number;
  readonly materializedExecutions: number;
  readonly verifiedExecutions: number;
  readonly mismatchCount: number;
  readonly firstMismatchRank: number;
  readonly firstExpectedCell: number;
  readonly firstActualCell: number;
  readonly materializedHash: number;
  readonly verifiedHash: number;
  readonly acceptedReceipts: number;
}

export function inspectSparseCM12PressureAddressingABReceipt(
  words: Uint32Array,
  layout: SparseCM12PressureAddressingABLayout,
  /** Arena word represented by words[0]; QA header readbacks pass baseWords. */
  wordsBase = 0,
): SparseCM12PressureAddressingABReceipt {
  const localHeader = layout.baseWords - wordsBase;
  if (localHeader < 0
    || words.length < localHeader + SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS) {
    throw new RangeError("PAB1 receipt arena is truncated");
  }
  const h = SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER;
  const at = (word: number) => words[localHeader + word]!;
  if (at(h.magic) !== SPARSE_CM12_PRESSURE_ADDRESSING_AB_MAGIC
    || at(h.version) !== SPARSE_CM12_PRESSURE_ADDRESSING_AB_VERSION
    || at(h.headerWords) !== SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS) {
    throw new Error("PAB1 receipt header is invalid");
  }
  return Object.freeze({
    phase: at(h.phase), fault: at(h.fault), firstFaultRank: at(h.firstFaultRank),
    expectedPCMGeneration: at(h.expectedPCMGeneration),
    materializedPCMGeneration: at(h.materializedPCMGeneration),
    expectedCount: at(h.expectedCount), materializedCount: at(h.materializedCount),
    materializedExecutions: at(h.materializedExecutions),
    verifiedExecutions: at(h.verifiedExecutions), mismatchCount: at(h.mismatchCount),
    firstMismatchRank: at(h.firstMismatchRank), firstExpectedCell: at(h.firstExpectedCell),
    firstActualCell: at(h.firstActualCell), materializedHash: at(h.materializedHash),
    verifiedHash: at(h.verifiedHash), acceptedReceipts: at(h.acceptedReceipts),
  });
}

export function sparseCM12PressureAddressingABReceiptAccepted(
  receipt: SparseCM12PressureAddressingABReceipt,
): boolean {
  return receipt.phase === SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE.accepted
    && receipt.fault === SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.none
    && receipt.expectedPCMGeneration === receipt.materializedPCMGeneration
    && receipt.expectedCount === receipt.materializedCount
    && receipt.materializedExecutions === receipt.expectedCount
    && receipt.verifiedExecutions === receipt.expectedCount
    && receipt.mismatchCount === 0
    && receipt.materializedHash === receipt.verifiedHash;
}

/** Copy this sealed triplet into the production pressure-cell indirect buffer.
 * Finalization publishes X last and every fault path forces it to zero. */
export function sparseCM12PressureAddressingSolveIndirectByteOffset(
  layout: SparseCM12PressureAddressingABLayout,
): number {
  return 4 * (layout.baseWords
    + SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER.solveIndirectX);
}

export interface SparseCM12PressureAddressingABPipelineDescriptor {
  readonly key: string;
  readonly entryPoint: string;
  readonly constants?: Readonly<Record<string, number>>;
  readonly constructionOnly: true;
  readonly runtimeSelectable: false;
  readonly outputSelectable: false;
}

/** Descriptors are intentionally data-only for GPUCompilationManager. */
export function createSparseCM12PressureAddressingABPipelineDescriptors(
  mode: SparseCM12PressureAddressingABModeName,
): readonly SparseCM12PressureAddressingABPipelineDescriptor[] {
  const sealed = (key: string, entryPoint: string,
    constants?: Readonly<Record<string, number>>) => Object.freeze({
      key, entryPoint, ...(constants ? { constants } : {}),
      constructionOnly: true as const, runtimeSelectable: false as const,
      outputSelectable: false as const,
    });
  return Object.freeze([
    sealed("beginPressureAddressMaterialization",
      "beginSparseCM12PressureAddressMaterialization"),
    sealed("materializePressureCellAddresses", "materializeSparseCM12PressureCellAddresses"),
    sealed("verifyPressureCellAddresses", "verifySparseCM12PressureCellAddresses"),
    sealed("finalizePressureAddressMaterialization",
      "finalizeSparseCM12PressureAddressMaterialization"),
    sealed(`pressureAddressMode-${mode}`, "exerciseSparseCM12PressureAddressing",
      sparseCM12PressureAddressingABPipelineConstants(mode)),
  ]);
}
