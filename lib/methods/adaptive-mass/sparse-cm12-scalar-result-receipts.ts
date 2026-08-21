/**
 * SRR1: producer-authored exact scalar-result authority for matched Sparse CM12 profiles.
 *
 * This authority never reads or writes scalar physics.  A scalar producer compares
 * every physical word in a tile with the exact HEAD result and publishes the
 * resulting mismatch counts.  Persistent work/clean trees are repaired only for
 * generation-stamped candidate leaves or by a producer that just executed a tile.
 */

export const SPARSE_CM12_SCALAR_RESULT_MAGIC = 0x5352_5231; // SRR1
export const SPARSE_CM12_SCALAR_RESULT_VERSION = 1;
export const SPARSE_CM12_SCALAR_RESULT_HEADER_WORDS = 32;
export const SPARSE_CM12_SCALAR_RESULT_TILE_WORDS = 16;
export const SPARSE_CM12_SCALAR_RESULT_CANDIDATE_WORDS = 16;
export const SPARSE_CM12_SCALAR_RESULT_LEAF_WORDS = 4;
export const SPARSE_CM12_SCALAR_RESULT_INVALID = 0xffff_ffff;

export const SPARSE_CM12_SCALAR_RESULT_HEADER = Object.freeze({
  magic: 0, version: 1, brickFineResolution: 2, presentationPageResolution: 3,
  tileCapacity: 4, leafCapacity: 5, phase: 6, fault: 7,
  acceptedGeneration: 8, candidateGeneration: 9, topologyGeneration: 10,
  sourceParity: 11, leafCount: 12, workCount: 13, cleanCount: 14,
  firstFaultTile: 15, tileBase: 16, candidateReceiptBase: 17, leafBase: 18,
  workListBase: 19, workTreeBase: 20, cleanTreeBase: 21, treeWords: 22,
  totalWords: 23, bootstrapFrames: 24, committedFrames: 25,
  scheduledWorkCount: 26, executedWorkCount: 27, reservedBase: 28,
} as const);

export const SPARSE_CM12_SCALAR_RESULT_TILE = Object.freeze({
  resultGeneration: 0, topologyGeneration: 1, bank0Generation: 2,
  bank1Generation: 3, headResultGeneration: 4, coveredWords: 5,
  expectedWords: 6, sourceMismatchCount: 7, destinationMismatchCount: 8,
  dependencyGeneration: 9, dependencyCertifiedGeneration: 10,
  candidateGeneration: 11, causeMask: 12, flags: 13, resultKind: 14,
  receiptGeneration: 15,
} as const);

export const SPARSE_CM12_SCALAR_RESULT_LEAF = Object.freeze({
  tile: 0, generation: 1, causeMask: 2, reserved: 3,
} as const);

export const SPARSE_CM12_SCALAR_RESULT_CANDIDATE = Object.freeze({
  topologyGeneration: 0, bank0Generation: 1, bank1Generation: 2,
  headResultGeneration: 3, coveredWords: 4, expectedWords: 5,
  sourceMismatchCount: 6, destinationMismatchCount: 7,
  dependencyGeneration: 8, dependencyCertifiedGeneration: 9,
  resultKind: 10, generation: 11, flags: 12, reservedBase: 13,
} as const);

export const SPARSE_CM12_SCALAR_RESULT_PHASE = Object.freeze({
  accepted: 1, collecting: 2, sealed: 3, fault: 4, rejected: 5,
} as const);

export const SPARSE_CM12_SCALAR_RESULT_FAULT = Object.freeze({
  none: 0, header: 1, phase: 2, generation: 3, tile: 4,
  leafOverflow: 5, missingExecution: 6, duplicateExecution: 7,
  bootstrapRequired: 8, atomicContention: 9,
} as const);

export const SPARSE_CM12_SCALAR_RESULT_FLAG = Object.freeze({
  producerExecuted: 1 << 0, fullWordCoverage: 1 << 1,
  dualBankExact: 1 << 2, dependencyExact: 1 << 3,
  classified: 1 << 4, work: 1 << 5, clean: 1 << 6,
} as const);

export const SPARSE_CM12_SCALAR_RESULT_CAUSE = Object.freeze({
  scalarWrite: 1 << 0, velocityWrite: 1 << 1, topologyWrite: 1 << 2,
  gammaWrite: 1 << 3, surfaceWrite: 1 << 4, solidWrite: 1 << 5,
  dependencyClosure: 1 << 6, bootstrap: 1 << 7,
} as const);

export interface SparseCM12ScalarResultLayout {
  readonly brickFineResolution: 4 | 8 | 16;
  readonly presentationPageResolution: 4 | 8 | 16;
  readonly tileCapacity: number;
  readonly leafCapacity: number;
  readonly treeLeafCapacity: number;
  readonly treeWords: number;
  readonly tileBaseWords: number;
  readonly candidateReceiptBaseWords: number;
  readonly leafBaseWords: number;
  readonly workListBaseWords: number;
  readonly workTreeBaseWords: number;
  readonly cleanTreeBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12ScalarResultAuthority {
  readonly layout: SparseCM12ScalarResultLayout;
  readonly words: Uint32Array;
}

export interface SparseCM12ScalarResultByteMapEntry {
  readonly name: "control" | "tileReceipts" | "candidateReceipts" | "candidateLeaves"
    | "workList" | "workCountTree" | "cleanCountTree";
  readonly offsetBytes: number;
  readonly sizeBytes: number;
  readonly bytesPerRecord: number;
  readonly ordering: string;
  readonly writers: string;
  readonly readers: string;
}

export interface SparseCM12ScalarResultExpectation {
  readonly topologyGeneration: number;
  /** Physical-bank generations; their meaning does not change with parity. */
  readonly bank0Generation: number;
  readonly bank1Generation: number;
  readonly headResultGeneration: number;
  readonly expectedWords: number;
  readonly dependencyGeneration: number;
}

export interface SparseCM12ScalarResultReceipt
  extends SparseCM12ScalarResultExpectation {
  readonly coveredWords: number;
  /** Count from a bit-exact comparison of source words with HEAD output words. */
  readonly sourceMismatchCount: number;
  /** Count from a bit-exact comparison of destination words with HEAD output words. */
  readonly destinationMismatchCount: number;
  readonly dependencyCertifiedGeneration: number;
  readonly resultKind: number;
}

const integer = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0)
    || value >= SPARSE_CM12_SCALAR_RESULT_INVALID) throw new RangeError(label);
  return value;
};
const align64 = (words: number): number => Math.ceil(words / 64) * 64;
const nextPowerOfTwo = (value: number): number => {
  let result = 1;
  while (result < value) result *= 2;
  return result;
};

export function scalarResultTileWord(
  layout: SparseCM12ScalarResultLayout, tile: number,
): number {
  if (!Number.isInteger(tile) || tile < 0 || tile >= layout.tileCapacity) {
    throw new RangeError("SRR1 tile");
  }
  return layout.tileBaseWords + SPARSE_CM12_SCALAR_RESULT_TILE_WORDS * tile;
}

export function scalarResultCandidateWord(
  layout: SparseCM12ScalarResultLayout, tile: number,
): number {
  if (!Number.isInteger(tile) || tile < 0 || tile >= layout.tileCapacity) {
    throw new RangeError("SRR1 candidate tile");
  }
  return layout.candidateReceiptBaseWords
    + SPARSE_CM12_SCALAR_RESULT_CANDIDATE_WORDS * tile;
}

export function createSparseCM12ScalarResultAuthority(options: {
  readonly tileCapacity: number;
  readonly leafCapacity?: number;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly presentationPageResolution?: 4 | 8 | 16;
}): SparseCM12ScalarResultAuthority {
  const tileCapacity = integer(options.tileCapacity, "SRR1 tile capacity", true);
  const leafCapacity = integer(options.leafCapacity ?? tileCapacity,
    "SRR1 leaf capacity", true);
  if (leafCapacity < tileCapacity) throw new RangeError(
    "SRR1 leaf capacity must admit one deduplicated leaf per tile");
  const brickFineResolution = options.brickFineResolution ?? 8;
  const presentationPageResolution = options.presentationPageResolution ?? brickFineResolution;
  if ((brickFineResolution !== 4 && brickFineResolution !== 8 && brickFineResolution !== 16)
    || presentationPageResolution !== brickFineResolution) {
    throw new Error("SRR1 requires a matched B4/P4, B8/P8, or B16/P16 ABI");
  }
  const treeLeafCapacity = nextPowerOfTwo(tileCapacity);
  const treeWords = 2 * treeLeafCapacity - 1;
  const tileBaseWords = align64(SPARSE_CM12_SCALAR_RESULT_HEADER_WORDS);
  const leafBaseWords = align64(tileBaseWords
    + SPARSE_CM12_SCALAR_RESULT_TILE_WORDS * tileCapacity);
  const candidateReceiptBaseWords = leafBaseWords;
  const candidateLeafBaseWords = align64(candidateReceiptBaseWords
    + SPARSE_CM12_SCALAR_RESULT_CANDIDATE_WORDS * tileCapacity);
  const workListBaseWords = align64(candidateLeafBaseWords
    + SPARSE_CM12_SCALAR_RESULT_LEAF_WORDS * leafCapacity);
  const workTreeBaseWords = align64(workListBaseWords + tileCapacity);
  const cleanTreeBaseWords = align64(workTreeBaseWords + treeWords);
  const totalWords = align64(cleanTreeBaseWords + treeWords);
  const layout = Object.freeze({ brickFineResolution,
    presentationPageResolution, tileCapacity, leafCapacity,
    treeLeafCapacity, treeWords, tileBaseWords,
    candidateReceiptBaseWords, leafBaseWords: candidateLeafBaseWords,
    workListBaseWords, workTreeBaseWords, cleanTreeBaseWords,
    totalWords, totalBytes: 4 * totalWords });
  const words = new Uint32Array(totalWords);
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  words[h.magic] = SPARSE_CM12_SCALAR_RESULT_MAGIC;
  words[h.version] = SPARSE_CM12_SCALAR_RESULT_VERSION;
  words[h.brickFineResolution] = brickFineResolution;
  words[h.presentationPageResolution] = presentationPageResolution;
  words[h.tileCapacity] = tileCapacity;
  words[h.leafCapacity] = leafCapacity;
  words[h.phase] = SPARSE_CM12_SCALAR_RESULT_PHASE.accepted;
  words[h.firstFaultTile] = SPARSE_CM12_SCALAR_RESULT_INVALID;
  words[h.tileBase] = tileBaseWords;
  words[h.candidateReceiptBase] = candidateReceiptBaseWords;
  words[h.leafBase] = candidateLeafBaseWords;
  words[h.workListBase] = workListBaseWords;
  words[h.workTreeBase] = workTreeBaseWords;
  words[h.cleanTreeBase] = cleanTreeBaseWords;
  words[h.treeWords] = treeWords;
  words[h.totalWords] = totalWords;
  return { layout, words };
}

/** Exact, contiguous arena partition including alignment padding. */
export function sparseCM12ScalarResultByteMap(
  layout: SparseCM12ScalarResultLayout,
): readonly SparseCM12ScalarResultByteMapEntry[] {
  return Object.freeze([
    { name: "control" as const, offsetBytes: 0,
      sizeBytes: 4 * layout.tileBaseWords, bytesPerRecord: 128,
      ordering: "single header followed by 256-byte alignment padding",
      writers: "FCA-gated begin/seal/commit", readers: "all SRR1 phases" },
    { name: "tileReceipts" as const, offsetBytes: 4 * layout.tileBaseWords,
      sizeBytes: 4 * (layout.candidateReceiptBaseWords - layout.tileBaseWords),
      bytesPerRecord: 4 * SPARSE_CM12_SCALAR_RESULT_TILE_WORDS,
      ordering: "canonical logical tile id; aligned tail padding",
      writers: "candidate repair and scalar HEAD comparator producer",
      readers: "candidate repair and dirty overlay" },
    { name: "candidateReceipts" as const,
      offsetBytes: 4 * layout.candidateReceiptBaseWords,
      sizeBytes: 4 * (layout.leafBaseWords - layout.candidateReceiptBaseWords),
      bytesPerRecord: 4 * SPARSE_CM12_SCALAR_RESULT_CANDIDATE_WORDS,
      ordering: "canonical logical tile id; unpublished until frame commit",
      writers: "scalar HEAD comparator producer", readers: "commit promotion" },
    { name: "candidateLeaves" as const, offsetBytes: 4 * layout.leafBaseWords,
      sizeBytes: 4 * (layout.workListBaseWords - layout.leafBaseWords),
      bytesPerRecord: 4 * SPARSE_CM12_SCALAR_RESULT_LEAF_WORDS,
      ordering: "generation-stamped append evidence; aligned tail padding",
      writers: "dependency/topology/scalar producers", readers: "candidate repair" },
    { name: "workList" as const, offsetBytes: 4 * layout.workListBaseWords,
      sizeBytes: 4 * (layout.workTreeBaseWords - layout.workListBaseWords),
      bytesPerRecord: 4, ordering: "canonical tile id by deterministic work rank",
      writers: "seal rank-select", readers: "scalar producer and commit promotion" },
    { name: "workCountTree" as const, offsetBytes: 4 * layout.workTreeBaseWords,
      sizeBytes: 4 * (layout.cleanTreeBaseWords - layout.workTreeBaseWords),
      bytesPerRecord: 4, ordering: "binary heap; canonical tile leaves",
      writers: "candidate repair and scalar result producer",
      readers: "fixed indirect rank-select" },
    { name: "cleanCountTree" as const, offsetBytes: 4 * layout.cleanTreeBaseWords,
      sizeBytes: 4 * (layout.totalWords - layout.cleanTreeBaseWords),
      bytesPerRecord: 4, ordering: "binary heap; canonical tile leaves",
      writers: "candidate repair and scalar result producer",
      readers: "diagnostics and dirty overlay" },
  ]);
}

function fail(
  authority: SparseCM12ScalarResultAuthority, fault: number, tile: number,
): false {
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  authority.words[h.phase] = SPARSE_CM12_SCALAR_RESULT_PHASE.fault;
  authority.words[h.fault] = fault;
  authority.words[h.firstFaultTile] = tile >>> 0;
  authority.words[h.scheduledWorkCount] = 0;
  return false;
}

function reject(
  authority: SparseCM12ScalarResultAuthority, fault: number, tile: number,
): false {
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  authority.words[h.phase] = SPARSE_CM12_SCALAR_RESULT_PHASE.rejected;
  authority.words[h.fault] = fault;
  authority.words[h.firstFaultTile] = tile >>> 0;
  authority.words[h.scheduledWorkCount] = 0;
  return false;
}

function treeSet(
  authority: SparseCM12ScalarResultAuthority, base: number, tile: number,
  value: 0 | 1,
): void {
  const { words, layout } = authority;
  let node = layout.treeLeafCapacity - 1 + tile;
  if (words[base + node] === value) return;
  words[base + node] = value;
  while (node > 0) {
    node = Math.floor((node - 1) / 2);
    words[base + node] = words[base + 2 * node + 1]!
      + words[base + 2 * node + 2]!;
  }
}

function setClassification(
  authority: SparseCM12ScalarResultAuthority, tile: number, clean: boolean,
): void {
  const { words, layout } = authority;
  const at = scalarResultTileWord(layout, tile);
  const t = SPARSE_CM12_SCALAR_RESULT_TILE;
  const receiptMask = SPARSE_CM12_SCALAR_RESULT_FLAG.producerExecuted
    | SPARSE_CM12_SCALAR_RESULT_FLAG.fullWordCoverage
    | SPARSE_CM12_SCALAR_RESULT_FLAG.dualBankExact
    | SPARSE_CM12_SCALAR_RESULT_FLAG.dependencyExact;
  words[at + t.flags] = (words[at + t.flags]! & receiptMask)
    | SPARSE_CM12_SCALAR_RESULT_FLAG.classified
    | (clean ? SPARSE_CM12_SCALAR_RESULT_FLAG.clean
      : SPARSE_CM12_SCALAR_RESULT_FLAG.work);
  treeSet(authority, layout.workTreeBaseWords, tile, clean ? 0 : 1);
  treeSet(authority, layout.cleanTreeBaseWords, tile, clean ? 1 : 0);
}

export function beginSparseCM12ScalarResultFrame(
  authority: SparseCM12ScalarResultAuthority,
  input: { readonly generation: number; readonly topologyGeneration: number;
    readonly sourceParity: 0 | 1; readonly bootstrap?: boolean },
): boolean {
  const { words, layout } = authority;
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  if (words[h.magic] !== SPARSE_CM12_SCALAR_RESULT_MAGIC
    || words[h.version] !== SPARSE_CM12_SCALAR_RESULT_VERSION
    || words[h.tileCapacity] !== layout.tileCapacity
    || words[h.treeWords] !== layout.treeWords) {
    return fail(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.header,
      SPARSE_CM12_SCALAR_RESULT_INVALID);
  }
  if (words[h.phase] !== SPARSE_CM12_SCALAR_RESULT_PHASE.accepted
    && words[h.phase] !== SPARSE_CM12_SCALAR_RESULT_PHASE.rejected) {
    return fail(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.phase,
      SPARSE_CM12_SCALAR_RESULT_INVALID);
  }
  const generation = integer(input.generation, "SRR1 generation", true);
  const bootstrap = input.bootstrap === true;
  const constructionBootstrap = bootstrap
    && words[h.acceptedGeneration] === 0 && words[h.bootstrapFrames] === 0;
  if (!constructionBootstrap && generation !== words[h.acceptedGeneration]! + 1) {
    return fail(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.generation,
      SPARSE_CM12_SCALAR_RESULT_INVALID);
  }
  if (words[h.acceptedGeneration] === 0 && !bootstrap) {
    return fail(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.bootstrapRequired,
      SPARSE_CM12_SCALAR_RESULT_INVALID);
  }
  if (bootstrap && words[h.acceptedGeneration] !== 0) {
    return fail(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.phase,
      SPARSE_CM12_SCALAR_RESULT_INVALID);
  }
  if (words[h.phase] === SPARSE_CM12_SCALAR_RESULT_PHASE.rejected) {
    const workCount = words[layout.workTreeBaseWords]!;
    for (let rank = 0; rank < workCount; rank += 1) {
      const tile = selectSparseCM12ScalarResultTile(authority, "work", rank);
      if (tile !== SPARSE_CM12_SCALAR_RESULT_INVALID) {
        words[scalarResultCandidateWord(layout, tile)
          + SPARSE_CM12_SCALAR_RESULT_CANDIDATE.generation] = 0;
      }
    }
  }
  words[h.phase] = SPARSE_CM12_SCALAR_RESULT_PHASE.collecting;
  words[h.candidateGeneration] = generation;
  words[h.topologyGeneration] = integer(input.topologyGeneration,
    "SRR1 topology generation", true);
  words[h.sourceParity] = input.sourceParity;
  words[h.fault] = 0;
  words[h.firstFaultTile] = SPARSE_CM12_SCALAR_RESULT_INVALID;
  words[h.leafCount] = 0;
  words[h.scheduledWorkCount] = 0;
  words[h.executedWorkCount] = 0;
  if (bootstrap && words[h.bootstrapFrames] === 0) {
    words[h.bootstrapFrames]! += 1;
    for (let tile = 0; tile < layout.tileCapacity; tile += 1) {
      appendSparseCM12ScalarResultCandidate(authority, tile,
        SPARSE_CM12_SCALAR_RESULT_CAUSE.bootstrap);
    }
  }
  return true;
}

/** Append once per changed tile. Duplicate causes are generation-stamped and deduped. */
export function appendSparseCM12ScalarResultCandidate(
  authority: SparseCM12ScalarResultAuthority, tile: number, causeMask: number,
): boolean {
  const { words, layout } = authority;
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  if (words[h.phase] !== SPARSE_CM12_SCALAR_RESULT_PHASE.collecting) return false;
  let at: number;
  try { at = scalarResultTileWord(layout, tile); } catch {
    return fail(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.tile, tile);
  }
  const t = SPARSE_CM12_SCALAR_RESULT_TILE;
  const generation = words[h.candidateGeneration]!;
  if (words[at + t.candidateGeneration] === generation) {
    words[at + t.causeMask]! |= causeMask >>> 0;
    return true;
  }
  words[at + t.candidateGeneration] = generation;
  words[at + t.causeMask] = causeMask >>> 0;
  // A candidate is dirty until its exact receipt is revalidated locally.
  setClassification(authority, tile, false);
  const leaf = words[h.leafCount]!;
  if (leaf >= layout.leafCapacity) {
    return fail(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.leafOverflow, tile);
  }
  const leafAt = layout.leafBaseWords + SPARSE_CM12_SCALAR_RESULT_LEAF_WORDS * leaf;
  words[leafAt + SPARSE_CM12_SCALAR_RESULT_LEAF.tile] = tile;
  words[leafAt + SPARSE_CM12_SCALAR_RESULT_LEAF.generation] = generation;
  words[leafAt + SPARSE_CM12_SCALAR_RESULT_LEAF.causeMask] = causeMask >>> 0;
  words[h.leafCount] = leaf + 1;
  return true;
}

function receiptIsExact(receipt: SparseCM12ScalarResultReceipt): boolean {
  return receipt.coveredWords === receipt.expectedWords
    && receipt.sourceMismatchCount === 0
    && receipt.destinationMismatchCount === 0
    && receipt.dependencyGeneration === receipt.dependencyCertifiedGeneration;
}

/**
 * Called only by a physical producer after it executed the sealed work tile and
 * compared both scalar banks with the exact HEAD output, word for word.
 */
export function publishSparseCM12ScalarExactResult(
  authority: SparseCM12ScalarResultAuthority, tile: number,
  receipt: SparseCM12ScalarResultReceipt,
): boolean {
  const { words, layout } = authority;
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  if (words[h.phase] !== SPARSE_CM12_SCALAR_RESULT_PHASE.sealed) return false;
  let at: number;
  try { at = scalarResultTileWord(layout, tile); } catch {
    return fail(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.tile, tile);
  }
  const c = SPARSE_CM12_SCALAR_RESULT_CANDIDATE;
  const candidateAt = scalarResultCandidateWord(layout, tile);
  const generation = words[h.candidateGeneration]!;
  if (words[candidateAt + c.generation] === generation) {
    return reject(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.duplicateExecution, tile);
  }
  const exact = receiptIsExact(receipt)
    && receipt.topologyGeneration === words[h.topologyGeneration];
  words[candidateAt + c.topologyGeneration] = receipt.topologyGeneration;
  words[candidateAt + c.bank0Generation] = receipt.bank0Generation;
  words[candidateAt + c.bank1Generation] = receipt.bank1Generation;
  words[candidateAt + c.headResultGeneration] = receipt.headResultGeneration;
  words[candidateAt + c.coveredWords] = receipt.coveredWords;
  words[candidateAt + c.expectedWords] = receipt.expectedWords;
  words[candidateAt + c.sourceMismatchCount] = receipt.sourceMismatchCount;
  words[candidateAt + c.destinationMismatchCount] = receipt.destinationMismatchCount;
  words[candidateAt + c.dependencyGeneration] = receipt.dependencyGeneration;
  words[candidateAt + c.dependencyCertifiedGeneration]
    = receipt.dependencyCertifiedGeneration;
  words[candidateAt + c.resultKind] = receipt.resultKind;
  words[candidateAt + c.generation] = generation;
  words[candidateAt + c.flags] = SPARSE_CM12_SCALAR_RESULT_FLAG.producerExecuted
    | (receipt.coveredWords === receipt.expectedWords
      ? SPARSE_CM12_SCALAR_RESULT_FLAG.fullWordCoverage : 0)
    | (receipt.sourceMismatchCount === 0 && receipt.destinationMismatchCount === 0
      ? SPARSE_CM12_SCALAR_RESULT_FLAG.dualBankExact : 0)
    | (receipt.dependencyGeneration === receipt.dependencyCertifiedGeneration
      ? SPARSE_CM12_SCALAR_RESULT_FLAG.dependencyExact : 0);
  words[h.executedWorkCount]! += 1;
  return exact;
}

export function sparseCM12ScalarResultTileCanSkip(
  authority: SparseCM12ScalarResultAuthority, tile: number,
  expected: SparseCM12ScalarResultExpectation,
): boolean {
  const { words, layout } = authority;
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  const at = scalarResultTileWord(layout, tile);
  const t = SPARSE_CM12_SCALAR_RESULT_TILE;
  const required = SPARSE_CM12_SCALAR_RESULT_FLAG.producerExecuted
    | SPARSE_CM12_SCALAR_RESULT_FLAG.fullWordCoverage
    | SPARSE_CM12_SCALAR_RESULT_FLAG.dualBankExact
    | SPARSE_CM12_SCALAR_RESULT_FLAG.dependencyExact;
  return words[h.phase] === SPARSE_CM12_SCALAR_RESULT_PHASE.collecting
    && words[at + t.topologyGeneration] === expected.topologyGeneration
    && words[at + t.bank0Generation] === expected.bank0Generation
    && words[at + t.bank1Generation] === expected.bank1Generation
    && words[at + t.headResultGeneration] === expected.headResultGeneration
    && words[at + t.coveredWords] === expected.expectedWords
    && words[at + t.expectedWords] === expected.expectedWords
    && words[at + t.dependencyGeneration] === expected.dependencyGeneration
    && words[at + t.dependencyCertifiedGeneration] === expected.dependencyGeneration
    && (words[at + t.flags]! & required) === required;
}

/** Candidate-only repair; unchanged tile classifications and tree paths persist. */
export function sealSparseCM12ScalarResultFrame(
  authority: SparseCM12ScalarResultAuthority,
): boolean {
  const { words, layout } = authority;
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  if (words[h.phase] !== SPARSE_CM12_SCALAR_RESULT_PHASE.collecting) return false;
  // appendCandidate already made every changed leaf work. No candidate may be
  // recertified clean from speculative topology/dependency state; only the
  // physical producer may do that, and promotion waits for commit.
  words[h.workCount] = words[layout.workTreeBaseWords]!;
  words[h.cleanCount] = words[layout.cleanTreeBaseWords]!;
  words[h.scheduledWorkCount] = words[h.workCount]!;
  for (let rank = 0; rank < words[h.scheduledWorkCount]!; rank += 1) {
    words[layout.workListBaseWords + rank] = selectSparseCM12ScalarResultTile(
      authority, "work", rank);
  }
  words[h.executedWorkCount] = 0;
  words[h.phase] = SPARSE_CM12_SCALAR_RESULT_PHASE.sealed;
  return true;
}

/** Deterministic rank-select over the persistent work or clean count tree. */
export function selectSparseCM12ScalarResultTile(
  authority: SparseCM12ScalarResultAuthority, kind: "work" | "clean", rank: number,
): number {
  const { words, layout } = authority;
  const base = kind === "work" ? layout.workTreeBaseWords : layout.cleanTreeBaseWords;
  if (!Number.isInteger(rank) || rank < 0 || rank >= words[base]!) {
    return SPARSE_CM12_SCALAR_RESULT_INVALID;
  }
  let node = 0;
  let remaining = rank;
  while (node < layout.treeLeafCapacity - 1) {
    const left = 2 * node + 1;
    const leftCount = words[base + left]!;
    if (remaining < leftCount) node = left;
    else { remaining -= leftCount; node = left + 1; }
  }
  const tile = node - (layout.treeLeafCapacity - 1);
  return tile < layout.tileCapacity ? tile : SPARSE_CM12_SCALAR_RESULT_INVALID;
}

/**
 * Publication is rollback-free: the candidate bank is accepted only after every
 * tile in the sealed work tree authored exactly one receipt.  Failure keeps the
 * previous accepted generation authoritative and produces zero future work.
 */
export function commitSparseCM12ScalarResultFrame(
  authority: SparseCM12ScalarResultAuthority,
): boolean {
  const { words } = authority;
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  if (words[h.phase] !== SPARSE_CM12_SCALAR_RESULT_PHASE.sealed) return false;
  const scheduled = words[h.scheduledWorkCount]!;
  if (words[h.executedWorkCount] !== scheduled) {
    return reject(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.missingExecution,
      SPARSE_CM12_SCALAR_RESULT_INVALID);
  }
  const scheduledTiles = Array.from({ length: scheduled }, (_, rank) =>
    words[authority.layout.workListBaseWords + rank]!);
  const t = SPARSE_CM12_SCALAR_RESULT_TILE;
  const c = SPARSE_CM12_SCALAR_RESULT_CANDIDATE;
  for (const tile of scheduledTiles) {
    if (tile === SPARSE_CM12_SCALAR_RESULT_INVALID) {
      return reject(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.missingExecution, tile);
    }
    const candidateAt = scalarResultCandidateWord(authority.layout, tile);
    if (words[candidateAt + c.generation] !== words[h.candidateGeneration]) {
      return reject(authority, SPARSE_CM12_SCALAR_RESULT_FAULT.missingExecution, tile);
    }
  }
  for (const tile of scheduledTiles) {
    const candidateAt = scalarResultCandidateWord(authority.layout, tile);
    const at = scalarResultTileWord(authority.layout, tile);
    words[at + t.resultGeneration] = words[h.candidateGeneration]!;
    words[at + t.topologyGeneration] = words[candidateAt + c.topologyGeneration]!;
    words[at + t.bank0Generation] = words[candidateAt + c.bank0Generation]!;
    words[at + t.bank1Generation] = words[candidateAt + c.bank1Generation]!;
    words[at + t.headResultGeneration] = words[candidateAt + c.headResultGeneration]!;
    words[at + t.coveredWords] = words[candidateAt + c.coveredWords]!;
    words[at + t.expectedWords] = words[candidateAt + c.expectedWords]!;
    words[at + t.sourceMismatchCount] = words[candidateAt + c.sourceMismatchCount]!;
    words[at + t.destinationMismatchCount]
      = words[candidateAt + c.destinationMismatchCount]!;
    words[at + t.dependencyGeneration] = words[candidateAt + c.dependencyGeneration]!;
    words[at + t.dependencyCertifiedGeneration]
      = words[candidateAt + c.dependencyCertifiedGeneration]!;
    words[at + t.resultKind] = words[candidateAt + c.resultKind]!;
    words[at + t.receiptGeneration] = words[h.candidateGeneration]!;
    words[at + t.flags] = words[candidateAt + c.flags]!;
    const exact = (words[candidateAt + c.flags]!
      & (SPARSE_CM12_SCALAR_RESULT_FLAG.producerExecuted
        | SPARSE_CM12_SCALAR_RESULT_FLAG.fullWordCoverage
        | SPARSE_CM12_SCALAR_RESULT_FLAG.dualBankExact
        | SPARSE_CM12_SCALAR_RESULT_FLAG.dependencyExact))
      === (SPARSE_CM12_SCALAR_RESULT_FLAG.producerExecuted
        | SPARSE_CM12_SCALAR_RESULT_FLAG.fullWordCoverage
        | SPARSE_CM12_SCALAR_RESULT_FLAG.dualBankExact
        | SPARSE_CM12_SCALAR_RESULT_FLAG.dependencyExact)
      && words[candidateAt + c.coveredWords]
        === words[candidateAt + c.expectedWords]
      && words[candidateAt + c.topologyGeneration] === words[h.topologyGeneration];
    setClassification(authority, tile, exact);
  }
  words[h.acceptedGeneration] = words[h.candidateGeneration]!;
  words[h.workCount] = words[authority.layout.workTreeBaseWords]!;
  words[h.cleanCount] = words[authority.layout.cleanTreeBaseWords]!;
  words[h.committedFrames]! += 1;
  words[h.phase] = SPARSE_CM12_SCALAR_RESULT_PHASE.accepted;
  return true;
}
