/** Deterministic GPU-resident membership and rank-select ABI (PCM1). */
export const SPARSE_CM12_CANONICAL_MEMBERSHIP_MAGIC = 0x5043_4d31; // PCM1
export const SPARSE_CM12_CANONICAL_MEMBERSHIP_VERSION = 1;
export const SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER_WORDS = 16;
export const SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER_WORDS = 20;
export const SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS = 256;
export const SPARSE_CM12_CANONICAL_MEMBERSHIP_BRANCH = 32;
export const SPARSE_CM12_CANONICAL_MEMBERSHIP_ALIGNMENT_WORDS = 64;

export const SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE = Object.freeze({
  uninitialized: 0,
  accepted: 1,
  collecting: 2,
  repairing: 3,
  fault: 4,
} as const);

export const SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT = Object.freeze({
  none: 0,
  invalidHeader: 1,
  generationExhausted: 2,
  invalidStableId: 3,
  conflictingCandidate: 4,
  dirtyCapacity: 5,
  invalidPhase: 6,
  countUnderflow: 7,
  countOverflow: 8,
  publicationGap: 9,
  atomicContention: 10,
} as const);

export const SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER = Object.freeze({
  magic: 0,
  version: 1,
  headerWords: 2,
  domainHeaderWords: 3,
  domainCount: 4,
  leafBits: 5,
  branch: 6,
  cellHeaderBase: 7,
  rowHeaderBase: 8,
  totalWords: 9,
  flags: 10,
} as const);

export const SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER = Object.freeze({
  capacity: 0,
  phase: 1,
  candidateGeneration: 2,
  acceptedGeneration: 3,
  fault: 4,
  firstFaultId: 5,
  dirtyCount: 6,
  directWriteCount: 7,
  closureWriteCount: 8,
  directCauseMask: 9,
  closureCauseMask: 10,
  totalCount: 11,
  repairIndirectX: 12,
  repairIndirectY: 13,
  repairIndirectZ: 14,
  dirtyCapacity: 15,
  treeLevelCount: 16,
  expectedClosureCount: 17,
  coveredClosureCount: 18,
  flags: 19,
} as const);

export interface SparseCM12CanonicalMembershipDomainLayout {
  readonly capacity: number;
  readonly headerBaseWords: number;
  readonly activeBitsBaseWords: number;
  readonly activeBitWordCount: number;
  readonly candidateTokenBaseWords: number;
  readonly dirtyStampBaseWords: number;
  readonly dirtyListBaseWords: number;
  readonly leafCount: number;
  /** Level zero stores one count per 256-ID leaf; the last level is the root. */
  readonly treeLevelBaseWords: readonly number[];
  readonly treeLevelCounts: readonly number[];
}

export interface SparseCM12CanonicalMembershipLayout {
  readonly baseWords: number;
  readonly cell: SparseCM12CanonicalMembershipDomainLayout;
  readonly row: SparseCM12CanonicalMembershipDomainLayout;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export type SparseCM12CanonicalMembershipDomain = "cell" | "row";

export function sparseCM12CanonicalMembershipRepairIndirectByteOffset(
  layout: SparseCM12CanonicalMembershipLayout,
  domain: SparseCM12CanonicalMembershipDomain,
): number {
  return 4 * (layout[domain].headerBaseWords
    + SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER.repairIndirectX);
}

export function sparseCM12CanonicalMembershipAcceptedCountWord(
  layout: SparseCM12CanonicalMembershipLayout,
  domain: SparseCM12CanonicalMembershipDomain,
): number {
  return layout[domain].headerBaseWords
    + SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER.totalCount;
}

const alignWords = (value: number): number => {
  const alignment = SPARSE_CM12_CANONICAL_MEMBERSHIP_ALIGNMENT_WORDS;
  return Math.ceil(value / alignment) * alignment;
};

function checkedCapacity(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0x3fff_ffff) {
    throw new RangeError(`${label} must be an integer in [1, 2^30)`);
  }
  return value;
}

export function createSparseCM12CanonicalMembershipLayout(request: {
  readonly cellCapacity: number;
  readonly rowCapacity: number;
  readonly baseWords?: number;
}): SparseCM12CanonicalMembershipLayout {
  const baseWords = alignWords(request.baseWords ?? 0);
  let at = alignWords(baseWords + SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER_WORDS
    + 2 * SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER_WORDS);
  const domain = (capacityInput: number, headerBaseWords: number):
  SparseCM12CanonicalMembershipDomainLayout => {
    const capacity = checkedCapacity(capacityInput, "canonical membership capacity");
    const activeBitWordCount = Math.ceil(capacity / 32);
    const leafCount = Math.ceil(capacity / SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS);
    const activeBitsBaseWords = at; at = alignWords(at + activeBitWordCount);
    const candidateTokenBaseWords = at; at = alignWords(at + capacity);
    const dirtyStampBaseWords = at; at = alignWords(at + leafCount);
    const dirtyListBaseWords = at; at = alignWords(at + leafCount);
    const treeLevelBaseWords: number[] = [];
    const treeLevelCounts: number[] = [];
    let count = leafCount;
    while (true) {
      treeLevelBaseWords.push(at);
      treeLevelCounts.push(count);
      at = alignWords(at + count);
      if (count === 1) break;
      count = Math.ceil(count / SPARSE_CM12_CANONICAL_MEMBERSHIP_BRANCH);
    }
    return Object.freeze({
      capacity, headerBaseWords, activeBitsBaseWords, activeBitWordCount,
      candidateTokenBaseWords, dirtyStampBaseWords, dirtyListBaseWords, leafCount,
      treeLevelBaseWords: Object.freeze(treeLevelBaseWords),
      treeLevelCounts: Object.freeze(treeLevelCounts),
    });
  };
  const cellHeaderBaseWords = baseWords + SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER_WORDS;
  const rowHeaderBaseWords = cellHeaderBaseWords
    + SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER_WORDS;
  const cell = domain(request.cellCapacity, cellHeaderBaseWords);
  const row = domain(request.rowCapacity, rowHeaderBaseWords);
  const totalWords = alignWords(at);
  return Object.freeze({ baseWords, cell, row, totalWords, totalBytes: 4 * totalWords });
}

export function initializeSparseCM12CanonicalMembershipWords(
  words: Uint32Array,
  layout: SparseCM12CanonicalMembershipLayout,
): void {
  if (words.length < layout.totalWords) throw new RangeError("PCM1 target is smaller than layout");
  const h = SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER;
  const root = layout.baseWords;
  words[root + h.magic] = SPARSE_CM12_CANONICAL_MEMBERSHIP_MAGIC;
  words[root + h.version] = SPARSE_CM12_CANONICAL_MEMBERSHIP_VERSION;
  words[root + h.headerWords] = SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER_WORDS;
  words[root + h.domainHeaderWords] = SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER_WORDS;
  words[root + h.domainCount] = 2;
  words[root + h.leafBits] = SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS;
  words[root + h.branch] = SPARSE_CM12_CANONICAL_MEMBERSHIP_BRANCH;
  words[root + h.cellHeaderBase] = layout.cell.headerBaseWords;
  words[root + h.rowHeaderBase] = layout.row.headerBaseWords;
  words[root + h.totalWords] = layout.totalWords - layout.baseWords;
  const initializeDomain = (domain: SparseCM12CanonicalMembershipDomainLayout) => {
    const d = SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER;
    const base = domain.headerBaseWords;
    words[base + d.capacity] = domain.capacity;
    words[base + d.phase] = SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.uninitialized;
    words[base + d.firstFaultId] = 0xffff_ffff;
    words[base + d.repairIndirectY] = 1;
    words[base + d.repairIndirectZ] = 1;
    words[base + d.dirtyCapacity] = domain.leafCount;
    words[base + d.treeLevelCount] = domain.treeLevelCounts.length;
  };
  initializeDomain(layout.cell);
  initializeDomain(layout.row);
}

export function createSparseCM12CanonicalMembershipInitialWords(
  layout: SparseCM12CanonicalMembershipLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords);
  initializeSparseCM12CanonicalMembershipWords(words, layout);
  return words;
}
