/**
 * VDA1: bounded authorization for topology/injection VEX roots and retired
 * cache invalidations.  The authority is deliberately delta-sized.  It is
 * absent from the baseline allocation and never owns an alternate world scan.
 */
export const SPARSE_CM12_VEX_DELTA_AUTHORITY_MAGIC = 0x5644_4131; // VDA1
export const SPARSE_CM12_VEX_DELTA_AUTHORITY_VERSION = 1;
export const SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER_WORDS = 32;
export const SPARSE_CM12_VEX_DELTA_AUTHORITY_INVALID = 0xffff_ffff;

export const SPARSE_CM12_VEX_DELTA_AUTHORITY_PHASE = Object.freeze({
  idle: 0, collecting: 1, preflighted: 2, authorized: 3, published: 4,
  fault: 0xffff_ffff,
} as const);

export const SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT = Object.freeze({
  none: 0, invalidPhase: 1, staleTopologyGeneration: 2,
  staleVexGeneration: 3, inputCapacity: 4, duplicateOrUnorderedRoot: 5,
  duplicateOrUnorderedRetirement: 6, invalidRoot: 7, invalidRetirement: 8,
  rootCapacity: 9, missingCoverage: 10,
} as const);

export const SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, phase: 3,
  transactionGeneration: 4, topologyGeneration: 5, vexGeneration: 6,
  rootInputCount: 7, retiredInputCount: 8,
  rootInputCapacity: 9, retiredInputCapacity: 10, cellCapacity: 11,
  existingRootCount: 12, newRootCount: 13, expectedFrameGeneration: 14,
  rootPublishedCount: 15, retiredPublishedCount: 16,
  rootCoverageCount: 17, retiredCoverageCount: 18,
  authorizationGeneration: 19, successGeneration: 20,
  injectionRequired: 21, injectionPublished: 22,
  d4Required: 23, d4Published: 24,
  fault: 25, firstFaultRank: 26, firstFaultCell: 27,
  expectedRootHash: 28, observedRootHash: 29,
  expectedRetiredHash: 30, observedRetiredHash: 31,
} as const);

export interface SparseCM12VexDeltaAuthorityLayout {
  readonly baseWords: number;
  readonly cellCapacity: number;
  readonly authorityWords: number;
  readonly totalWords: number;
}

const integer = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_fffe) {
    throw new RangeError(`${label} must be a non-negative u32`);
  }
  return value;
};

/** Optional candidate allocation. Calling code omits this layout entirely for
 * the baseline arm, so the baseline byte layout is unchanged. */
export function createSparseCM12VexDeltaAuthorityLayout(options: Readonly<{
  baseWords?: number;
  cellCapacity: number;
}>): SparseCM12VexDeltaAuthorityLayout {
  const baseWords = integer(options.baseWords ?? 0, "baseWords");
  const cellCapacity = integer(options.cellCapacity, "cellCapacity");
  const totalWords = baseWords + SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER_WORDS;
  return Object.freeze({ baseWords, cellCapacity,
    authorityWords: SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER_WORDS, totalWords });
}

export function createSparseCM12VexDeltaAuthorityInitialWords(
  layout: SparseCM12VexDeltaAuthorityLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER;
  words[h.magic] = SPARSE_CM12_VEX_DELTA_AUTHORITY_MAGIC;
  words[h.version] = SPARSE_CM12_VEX_DELTA_AUTHORITY_VERSION;
  words[h.headerWords] = SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER_WORDS;
  words[h.rootInputCapacity] = 0;
  words[h.retiredInputCapacity] = 0;
  words[h.cellCapacity] = layout.cellCapacity;
  words[h.firstFaultRank] = SPARSE_CM12_VEX_DELTA_AUTHORITY_INVALID;
  words[h.firstFaultCell] = SPARSE_CM12_VEX_DELTA_AUTHORITY_INVALID;
  return words;
}

export interface SparseCM12VexDeltaRoot { readonly cell: number; readonly cause: number }

export const sparseCM12VexDeltaRootReceipt = (cell: number, cause: number): number =>
  (Math.imul(cell, 0x9e37_79b9) ^ Math.imul(cause, 0x85eb_ca6b)
    ^ 0xc2b2_ae35) >>> 0;
export const sparseCM12VexDeltaRetiredReceipt = (cell: number): number =>
  (Math.imul(cell, 0x27d4_eb2d) ^ 0x1656_67b1) >>> 0;
export const sparseCM12VexDeltaRootCoverageHash = (
  roots: readonly SparseCM12VexDeltaRoot[],
): number => roots.reduce((sum, root) =>
  (sum + sparseCM12VexDeltaRootReceipt(root.cell, root.cause)) >>> 0, 0);
export const sparseCM12VexDeltaRetiredCoverageHash = (
  cells: readonly number[],
): number => cells.reduce((sum, cell) =>
  (sum + sparseCM12VexDeltaRetiredReceipt(cell)) >>> 0, 0);

export interface SparseCM12VexDeltaPreflightResult {
  readonly authorized: boolean;
  readonly fault: number;
  readonly rootSlots: Uint32Array;
  readonly existingRootCount: number;
  readonly newRootCount: number;
  readonly reservedRootBase: number;
}

/**
 * Uniqueness proof used by the no-fail GPU seam:
 *
 * - the exact producer traversal may request one endpoint more than once;
 * - `atomicExchange(rootStamp[cell], generation)` has exactly one invocation
 *   observe a different generation for each cell;
 * - only that invocation appends, hence the root list has one entry per cell;
 * - later ordered cause batches only OR cause bits at the same depth-zero root;
 * - retired-cache stores are idempotent, so repeated retirement requests have
 *   the same unique effect as their set.
 *
 * Therefore at most `cellCapacity` root slots can be won, and a valid VEX
 * unique-root count cannot overflow. No post-authorization capacity test is
 * needed. The CPU oracle below independently computes this set cardinality.
 */

/** CPU oracle for the GPU seal. Requests need not be sorted or unique: exact
 * repeated incidences are intentionally admitted and publication merges them
 * through the VEX generation stamp. */
export function preflightSparseCM12VexDelta(options: Readonly<{
  transactionGeneration: number;
  topologyGeneration: number;
  expectedTopologyGeneration: number;
  vexGeneration: number;
  expectedVexGeneration: number;
  vexCollecting: boolean;
  cellCapacity: number;
  vexRootCount: number;
  vexRootCapacity: number;
  rootStamp: ArrayLike<number>;
  roots: readonly SparseCM12VexDeltaRoot[];
  /** Split between two ordered producer batches (normally topology/injection). */
  rootBatch0Count?: number;
  retiredCells: readonly number[];
  rootCoverageCount?: number;
  retiredCoverageCount?: number;
  finalCellActive: (cell: number) => boolean;
  finalCellRetired: (cell: number) => boolean;
  d4Required?: boolean;
  framePhaseValid?: boolean;
  frameGeneration?: number;
  expectedFrameGeneration?: number;
}>): SparseCM12VexDeltaPreflightResult {
  const invalid = (fault: number) => Object.freeze({ authorized: false, fault,
    rootSlots: new Uint32Array(options.roots.length).fill(
      SPARSE_CM12_VEX_DELTA_AUTHORITY_INVALID), existingRootCount: 0,
    newRootCount: 0, reservedRootBase: SPARSE_CM12_VEX_DELTA_AUTHORITY_INVALID });
  if (options.topologyGeneration !== options.expectedTopologyGeneration) {
    return invalid(SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.staleTopologyGeneration);
  }
  if (!options.vexCollecting || options.vexGeneration !== options.expectedVexGeneration) {
    return invalid(SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.staleVexGeneration);
  }
  if (options.d4Required && (!options.framePhaseValid
    || options.frameGeneration !== options.expectedFrameGeneration)) {
    return invalid(SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.invalidPhase);
  }
  if ((options.rootCoverageCount ?? options.roots.length) !== options.roots.length
    || (options.retiredCoverageCount ?? options.retiredCells.length)
      !== options.retiredCells.length) {
    return invalid(SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.missingCoverage);
  }
  const batch0Count = options.rootBatch0Count ?? options.roots.length;
  if (!Number.isInteger(batch0Count) || batch0Count < 0
    || batch0Count > options.roots.length) {
    return invalid(SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.missingCoverage);
  }
  const batchCause = [options.roots[0]?.cause,
    options.roots[batch0Count]?.cause] as const;
  for (let rank = 0; rank < options.roots.length; rank += 1) {
    const root = options.roots[rank]!;
    if (root.cell >= options.cellCapacity || root.cause === 0
      || root.cause !== batchCause[rank < batch0Count ? 0 : 1]
      || !options.finalCellActive(root.cell)) {
      return invalid(SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.invalidRoot);
    }
  }
  for (const cell of options.retiredCells) {
    if (cell >= options.cellCapacity || !options.finalCellRetired(cell)) {
      return invalid(SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.invalidRetirement);
    }
  }
  const rootSlots = new Uint32Array(options.roots.length).fill(
    SPARSE_CM12_VEX_DELTA_AUTHORITY_INVALID);
  const unique = new Set<number>();
  for (const root of options.roots) unique.add(root.cell);
  let existingRootCount = 0;
  for (const cell of unique) {
    if (options.rootStamp[cell] === options.vexGeneration) existingRootCount += 1;
  }
  const newRootCount = unique.size - existingRootCount;
  let ordinal = 0;
  const planned = new Set<number>();
  for (let rank = 0; rank < options.roots.length; rank += 1) {
    const cell = options.roots[rank]!.cell;
    if (options.rootStamp[cell] !== options.vexGeneration && !planned.has(cell)) {
      rootSlots[rank] = ordinal++; planned.add(cell);
    }
  }
  if (options.vexRootCount + newRootCount > options.vexRootCapacity) {
    return invalid(SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.rootCapacity);
  }
  return Object.freeze({ authorized: true,
    fault: SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT.none, rootSlots,
    existingRootCount, newRootCount, reservedRootBase: options.vexRootCount });
}
