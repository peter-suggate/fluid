/**
 * Accepted-read reverse dependency compiler for VEX -> FPA preparation.
 *
 * The characteristic path depends on velocity values, so ordinary cell/row
 * incidence cannot prove its read set. VRD1 therefore inverts a forward read
 * journal emitted by the prior result execution. Runtime lookup consumes an
 * immutable accepted CSR while the next candidate journal is built separately.
 */

export const SPARSE_CM12_VEX_REVERSE_DEPENDENCY_MAGIC = 0x5652_4431; // VRD1
export const SPARSE_CM12_VEX_REVERSE_DEPENDENCY_VERSION = 1;
export const SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER_WORDS = 32;
export const SPARSE_CM12_VEX_REVERSE_DEPENDENCY_INVALID = 0xffff_ffff;

export const SPARSE_CM12_VEX_REVERSE_DEPENDENCY_FLAG = Object.freeze({
  complete: 1 << 0, validated: 1 << 1, supportedPolicy: 1 << 2,
} as const);

export const SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, flags: 3,
  cellCount: 4, rowCount: 5, edgeCount: 6, edgeCapacity: 7,
  offsetBaseWords: 8, rowBaseWords: 9, totalWords: 10,
  topologyGeneration: 11,
  topologyHash0: 12, topologyHash1: 13, topologyHash2: 14, topologyHash3: 15,
  policyHash0: 16, policyHash1: 17, policyHash2: 18, policyHash3: 19,
  contentHash0: 20, contentHash1: 21, contentHash2: 22, contentHash3: 23,
  acceptedPolicyEpoch: 24, maximumSubsteps: 25, traceQueryCountPerSubstep: 26,
  interpolationCornerCount: 27, compilerVersion: 28,
  maximumRowsPerCell: 29, reserved0: 30, reserved1: 31,
} as const);

export const SPARSE_CM12_VEX_REVERSE_POLICY = Object.freeze({
  traceIntegrator: "rk2-midpoint",
  traceDirection: "departure",
  interpolation: "adaptive-trilinear-owner-cell",
  boundaryPolicy: "clamp-and-embedded-segment-clip",
  maximumSubsteps: 16,
  traceQueryCountPerSubstep: 2,
  interpolationCornerCount: 8,
  compiler: "cm12-actual-vex-accessor-read-journal",
  compilerVersion: 1,
} as const);

export type SparseCM12Hash128 = readonly [number, number, number, number];

export interface SparseCM12VexReverseDependencyPolicy {
  readonly traceIntegrator: typeof SPARSE_CM12_VEX_REVERSE_POLICY.traceIntegrator;
  readonly traceDirection: typeof SPARSE_CM12_VEX_REVERSE_POLICY.traceDirection;
  readonly interpolation: typeof SPARSE_CM12_VEX_REVERSE_POLICY.interpolation;
  readonly boundaryPolicy: typeof SPARSE_CM12_VEX_REVERSE_POLICY.boundaryPolicy;
  readonly maximumSubsteps: typeof SPARSE_CM12_VEX_REVERSE_POLICY.maximumSubsteps;
  readonly traceQueryCountPerSubstep:
    typeof SPARSE_CM12_VEX_REVERSE_POLICY.traceQueryCountPerSubstep;
  readonly interpolationCornerCount:
    typeof SPARSE_CM12_VEX_REVERSE_POLICY.interpolationCornerCount;
  readonly compiler: typeof SPARSE_CM12_VEX_REVERSE_POLICY.compiler;
  readonly compilerVersion: typeof SPARSE_CM12_VEX_REVERSE_POLICY.compilerVersion;
}

export type SparseCM12VexActualReadKind =
  | "incident-validity"
  | "probe-owner-span"
  | "interpolation-corner"
  | "final-interpolation-corner";

export interface SparseCM12VexActualRead {
  readonly cell: number;
  readonly kind: SparseCM12VexActualReadKind;
}

/** Actual reads captured by the accepted prior FPA execution. */
export interface SparseCM12VexForwardReadJournal {
  readonly topologyGeneration: number;
  readonly topologyHash: SparseCM12Hash128;
  readonly policyHash: SparseCM12Hash128;
  readonly policyEpoch: number;
  readonly generation: number;
  /** One record for every stable preparation row, in any input order. */
  readonly rows: readonly Readonly<{
    row: number;
    resultGeneration: number;
    dependencyGeneration: number;
    actualReadsComplete: true;
    /** Exact accessor reads made by this row's paired result execution. */
    reads: readonly SparseCM12VexActualRead[];
  }>[];
}

export interface SparseCM12VexReverseDependencyLayout {
  readonly baseWords: number;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly edgeCount: number;
  readonly edgeCapacity: number;
  readonly offsetBaseWords: number;
  readonly rowBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
  readonly headerBytes: number;
  readonly offsetBytes: number;
  readonly rowBytes: number;
  readonly unusedCapacityBytes: number;
  readonly maximumRowsPerCell: number;
}

export interface SparseCM12VexReverseDependency {
  readonly layout: SparseCM12VexReverseDependencyLayout;
  readonly words: Uint32Array;
  readonly topologyHash: SparseCM12Hash128;
  readonly policyHash: SparseCM12Hash128;
  readonly contentHash: SparseCM12Hash128;
  readonly policyEpoch: number;
  /** Canonical accepted forward journal retained for transactional inheritance. */
  readonly forwardRows: readonly (readonly SparseCM12VexActualRead[])[];
}

const u32 = (value: number, label: string, allowZero = true): number => {
  if (!Number.isSafeInteger(value) || value < (allowZero ? 0 : 1)
    || value > 0xffff_fffe) throw new RangeError(`${label} is outside stable u32 range`);
  return value;
};
const align64 = (words: number): number => Math.ceil(words / 64) * 64;
const hashString = (text: string): number[] => Array.from(text, (character) =>
  character.codePointAt(0)!);
const hash128 = (words: Iterable<number>): SparseCM12Hash128 => {
  const state = new Uint32Array([0x811c9dc5, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35]);
  for (const word of words) for (let lane = 0; lane < 4; lane += 1) {
    state[lane] = Math.imul((state[lane]! ^ (word >>> 0)
      ^ Math.imul(lane + 1, 0x27d4eb2d)) >>> 0, 0x01000193) >>> 0;
  }
  return [state[0]!, state[1]!, state[2]!, state[3]!];
};
const sameHash = (left: SparseCM12Hash128, right: SparseCM12Hash128): boolean =>
  left.every((word, index) => word === right[index]);

export function sparseCM12VexReverseDependencyPolicyHash(
  policy: SparseCM12VexReverseDependencyPolicy,
): SparseCM12Hash128 {
  requireSparseCM12VexReverseDependencyPolicy(policy);
  return hash128([
    ...hashString(policy.traceIntegrator), ...hashString(policy.traceDirection),
    ...hashString(policy.interpolation), ...hashString(policy.boundaryPolicy),
    policy.maximumSubsteps, policy.traceQueryCountPerSubstep,
    policy.interpolationCornerCount, ...hashString(policy.compiler),
    policy.compilerVersion,
  ]);
}

export function requireSparseCM12VexReverseDependencyPolicy(
  policy: SparseCM12VexReverseDependencyPolicy,
): void {
  const fixed = SPARSE_CM12_VEX_REVERSE_POLICY;
  for (const field of ["traceIntegrator", "traceDirection", "interpolation",
    "boundaryPolicy", "maximumSubsteps", "traceQueryCountPerSubstep",
    "interpolationCornerCount", "compiler", "compilerVersion"] as const) {
    if (policy[field] !== fixed[field]) {
      throw new Error(`VRD1 unsupported trace policy: ${field}`);
    }
  }
}

/**
 * Invert a construction-certified forward read set into stable cell -> row CSR.
 * Input order and duplicate cell mentions cannot affect the output.
 */
export function buildSparseCM12VexReverseDependency(options: {
  readonly baseWords?: number;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly edgeCapacity?: number;
  readonly policy: SparseCM12VexReverseDependencyPolicy;
  readonly journal: SparseCM12VexForwardReadJournal;
}): SparseCM12VexReverseDependency {
  const cellCount = u32(options.cellCount, "VRD1 cell count", false);
  const rowCount = u32(options.rowCount, "VRD1 row count", false);
  const baseWords = align64(u32(options.baseWords ?? 0, "VRD1 base words"));
  requireSparseCM12VexReverseDependencyPolicy(options.policy);
  const policyHash = sparseCM12VexReverseDependencyPolicyHash(options.policy);
  if (!sameHash(policyHash, options.journal.policyHash)) {
    throw new Error("VRD1 actual-read journal policy hash mismatch");
  }
  const topologyGeneration = u32(options.journal.topologyGeneration,
    "VRD1 topology generation", false);
  const generation = u32(options.journal.generation, "VRD1 journal generation");
  const policyEpoch = u32(options.journal.policyEpoch, "VRD1 policy epoch");
  options.journal.topologyHash.forEach((word) => u32(word, "VRD1 topology hash"));
  if (options.journal.rows.length !== rowCount) {
    throw new Error("VRD1 actual-read journal does not cover every stable row");
  }
  const seenRows = new Uint8Array(rowCount);
  const reverse: number[][] = Array.from({ length: cellCount }, () => []);
  const forwardRows: SparseCM12VexActualRead[][] = Array.from(
    { length: rowCount }, () => []);
  const kindOrder: Readonly<Record<SparseCM12VexActualReadKind, number>> = {
    "incident-validity": 0, "probe-owner-span": 1,
    "interpolation-corner": 2, "final-interpolation-corner": 3,
  };
  for (const record of options.journal.rows) {
    const row = u32(record.row, "VRD1 journal row");
    if (row >= rowCount || seenRows[row] !== 0) {
      throw new Error("VRD1 actual-read journal has invalid or duplicate stable row");
    }
    if (!record.actualReadsComplete || record.resultGeneration !== generation
      || record.dependencyGeneration !== generation) {
      throw new Error("VRD1 result/dependency generation pair is incomplete");
    }
    seenRows[row] = 1;
    const readsByKey = new Map<string, SparseCM12VexActualRead>();
    for (const read of record.reads) {
      if (!(read.kind in kindOrder)) throw new Error("VRD1 actual read kind is unsupported");
      const cell = u32(read.cell, "VRD1 actual-read VEX cell");
      if (cell >= cellCount) throw new Error("VRD1 actual-read VEX cell exceeds topology");
      readsByKey.set(`${cell}/${read.kind}`, Object.freeze({ cell, kind: read.kind }));
    }
    const reads = [...readsByKey.values()].sort((left, right) =>
      left.cell - right.cell || kindOrder[left.kind] - kindOrder[right.kind]);
    forwardRows[row] = reads;
    const cells = [...new Set(reads.map((read) => read.cell))];
    for (const cell of cells) {
      reverse[cell]!.push(row);
    }
  }
  if (seenRows.some((seen) => seen === 0)) {
    throw new Error("VRD1 actual-read journal has a stable-row coverage gap");
  }
  for (const rows of reverse) rows.sort((a, b) => a - b);
  const edgeCount = reverse.reduce((sum, rows) => sum + rows.length, 0);
  u32(edgeCount, "VRD1 edge count");
  const edgeCapacity = u32(options.edgeCapacity ?? edgeCount, "VRD1 edge capacity");
  if (edgeCapacity < edgeCount) throw new RangeError("VRD1 reverse edge capacity overflow");
  const offsetBaseWords = baseWords + SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER_WORDS;
  const rowBaseWords = align64(offsetBaseWords + cellCount + 1);
  const totalWords = align64(rowBaseWords + edgeCapacity);
  if (totalWords > 0x3fff_ffff) throw new RangeError("VRD1 immutable CSR exceeds 4 GiB");
  const maximumRowsPerCell = reverse.reduce((maximum, rows) =>
    Math.max(maximum, rows.length), 0);
  const layout: SparseCM12VexReverseDependencyLayout = Object.freeze({
    baseWords, cellCount, rowCount, edgeCount, edgeCapacity, offsetBaseWords,
    rowBaseWords, totalWords, totalBytes: 4 * (totalWords - baseWords),
    headerBytes: 4 * SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER_WORDS,
    offsetBytes: 4 * (cellCount + 1), rowBytes: 4 * edgeCount,
    unusedCapacityBytes: 4 * (edgeCapacity - edgeCount), maximumRowsPerCell,
  });
  const words = new Uint32Array(totalWords - baseWords);
  words.fill(SPARSE_CM12_VEX_REVERSE_DEPENDENCY_INVALID);
  const relativeOffset = offsetBaseWords - baseWords;
  const relativeRows = rowBaseWords - baseWords;
  let cursor = 0;
  for (let cell = 0; cell < cellCount; cell += 1) {
    words[relativeOffset + cell] = cursor;
    for (const row of reverse[cell]!) words[relativeRows + cursor++] = row;
  }
  words[relativeOffset + cellCount] = cursor;
  if (cursor !== edgeCount) throw new Error("VRD1 construction cursor mismatch");
  const contentHash = hash128([
    topologyGeneration, generation, policyEpoch,
    ...options.journal.topologyHash, ...policyHash,
    cellCount, rowCount, edgeCount, ...words.subarray(relativeOffset,
      relativeOffset + cellCount + 1), ...words.subarray(relativeRows, relativeRows + edgeCount),
  ]);
  const h = SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER;
  words[h.magic] = SPARSE_CM12_VEX_REVERSE_DEPENDENCY_MAGIC;
  words[h.version] = SPARSE_CM12_VEX_REVERSE_DEPENDENCY_VERSION;
  words[h.headerWords] = SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER_WORDS;
  words[h.flags] = SPARSE_CM12_VEX_REVERSE_DEPENDENCY_FLAG.complete
    | SPARSE_CM12_VEX_REVERSE_DEPENDENCY_FLAG.validated
    | SPARSE_CM12_VEX_REVERSE_DEPENDENCY_FLAG.supportedPolicy;
  words[h.cellCount] = cellCount; words[h.rowCount] = rowCount;
  words[h.edgeCount] = edgeCount; words[h.edgeCapacity] = edgeCapacity;
  words[h.offsetBaseWords] = offsetBaseWords; words[h.rowBaseWords] = rowBaseWords;
  words[h.totalWords] = totalWords; words[h.topologyGeneration] = topologyGeneration;
  words.set(options.journal.topologyHash, h.topologyHash0);
  words.set(policyHash, h.policyHash0); words.set(contentHash, h.contentHash0);
  // A dt/CFL or other policy-epoch change is an explicit FPE producer that
  // schedules affected rows and replaces their journal; it is not a geometric
  // radius baked into the accepted-read graph.
  words[h.acceptedPolicyEpoch] = policyEpoch;
  words[h.maximumSubsteps] = options.policy.maximumSubsteps;
  words[h.traceQueryCountPerSubstep] = options.policy.traceQueryCountPerSubstep;
  words[h.interpolationCornerCount] = options.policy.interpolationCornerCount;
  words[h.compilerVersion] = options.policy.compilerVersion;
  words[h.maximumRowsPerCell] = maximumRowsPerCell;
  return Object.freeze({ layout, words,
    topologyHash: options.journal.topologyHash, policyHash, contentHash,
    policyEpoch,
    forwardRows: Object.freeze(forwardRows.map((reads) => Object.freeze(reads))),
  });
}

/**
 * Publish candidate result dependencies atomically with the matching result
 * generation. Unexecuted rows inherit the prior accepted journal exactly.
 */
export function advanceSparseCM12VexReverseDependency(options: {
  readonly accepted: SparseCM12VexReverseDependency;
  readonly candidateGeneration: number;
  readonly candidatePolicyEpoch?: number;
  readonly edgeCapacity?: number;
  readonly policy: SparseCM12VexReverseDependencyPolicy;
  readonly executedRows: readonly Readonly<{
    row: number;
    resultGeneration: number;
    dependencyGeneration: number;
    actualReadsComplete: true;
    reads: readonly SparseCM12VexActualRead[];
  }>[];
}): SparseCM12VexReverseDependency {
  const { accepted } = options;
  const generation = u32(options.candidateGeneration, "VRD1 candidate generation", false);
  const policyEpoch = u32(options.candidatePolicyEpoch ?? accepted.policyEpoch,
    "VRD1 candidate policy epoch");
  const seen = new Set<number>();
  const replacements = new Map(options.executedRows.map((record) => {
    const row = u32(record.row, "VRD1 executed row");
    if (row >= accepted.layout.rowCount || seen.has(row)) {
      throw new Error("VRD1 candidate has invalid or duplicate executed row");
    }
    seen.add(row); return [row, record] as const;
  }));
  const rows = Array.from({ length: accepted.layout.rowCount }, (_, row) => {
    const replacement = replacements.get(row);
    return replacement ?? { row, resultGeneration: generation,
      dependencyGeneration: generation, actualReadsComplete: true as const,
      reads: accepted.forwardRows[row]!,
    };
  });
  if (policyEpoch !== accepted.policyEpoch && replacements.size !== accepted.layout.rowCount) {
    throw new Error("VRD1 policy epoch change requires every affected row journal");
  }
  return buildSparseCM12VexReverseDependency({
    baseWords: accepted.layout.baseWords, cellCount: accepted.layout.cellCount,
    rowCount: accepted.layout.rowCount,
    edgeCapacity: options.edgeCapacity ?? accepted.layout.edgeCapacity,
    policy: options.policy,
    journal: { topologyGeneration: accepted.words[
      SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER.topologyGeneration]!,
    topologyHash: accepted.topologyHash, policyHash: accepted.policyHash,
    policyEpoch, generation, rows },
  });
}

export function sparseCM12VexRowsScheduledByChangedCells(
  dependency: SparseCM12VexReverseDependency, changedCells: readonly number[],
): readonly number[] {
  const rows = new Set<number>();
  for (const cell of changedCells) for (const row of sparseCM12VexReverseRows(
    dependency, cell)) rows.add(row);
  return Object.freeze([...rows].sort((left, right) => left - right));
}

export function sparseCM12VexReverseRows(
  dependency: SparseCM12VexReverseDependency, cell: number,
): readonly number[] {
  const { layout, words } = dependency;
  if (!Number.isSafeInteger(cell) || cell < 0 || cell >= layout.cellCount) {
    throw new RangeError("VRD1 lookup cell is invalid");
  }
  const offsets = layout.offsetBaseWords - layout.baseWords;
  const rows = layout.rowBaseWords - layout.baseWords;
  return Object.freeze(Array.from(words.subarray(rows + words[offsets + cell]!,
    rows + words[offsets + cell + 1]!)));
}

export interface SparseCM12VexDonorTileSummary {
  readonly rowCount: number;
  readonly tileCount: number;
  readonly edgeCount: number;
  readonly rowTiles: readonly (readonly number[])[];
  readonly tileRows: readonly (readonly number[])[];
  readonly rowTileCount: Readonly<{
    p50: number; p90: number; p99: number; maximum: number;
  }>;
  readonly tileSubscriberCount: Readonly<{
    p50: number; p90: number; p99: number; maximum: number;
  }>;
  /** Canonical hash of ascending `(row,tile)` pairs. */
  readonly canonicalPairHash: SparseCM12Hash128;
}

const quantile = (sorted: readonly number[], fraction: number): number =>
  sorted.length === 0 ? 0 : sorted[Math.floor((sorted.length - 1) * fraction)]!;

/**
 * Coarsen the accepted xyz accessor read journal to stable 4^3 donor tiles.
 * This is dependency-complete: a changed cell marks its canonical tile, and
 * every row that previously read any cell in that tile is a subscriber. The
 * coarsening can add same-tile work but cannot omit an affected row.
 *
 * Incident `.w` reads remain on immutable HTP incidence and probe/span reads
 * belong to the topology epoch, so neither enters this value-dependency graph.
 */
export function summarizeSparseCM12VexDonorTiles(options: {
  readonly dependency: SparseCM12VexReverseDependency;
  readonly tileCount: number;
  readonly tileForCell: (cell: number) => number;
}): SparseCM12VexDonorTileSummary {
  const tileCount = u32(options.tileCount, "VRD1 donor tile count", false);
  const { dependency } = options;
  const rowTiles = dependency.forwardRows.map((reads, row) => {
    const tiles = new Set<number>();
    for (const read of reads) {
      if (read.kind !== "interpolation-corner"
        && read.kind !== "final-interpolation-corner") continue;
      const tile = u32(options.tileForCell(read.cell),
        `VRD1 donor tile for cell ${read.cell}`);
      if (tile >= tileCount) {
        throw new RangeError(`VRD1 donor tile ${tile} for row ${row} exceeds capacity`);
      }
      tiles.add(tile);
    }
    return Object.freeze([...tiles].sort((left, right) => left - right));
  });
  const tileRows: number[][] = Array.from({ length: tileCount }, () => []);
  const pairWords: number[] = [];
  for (let row = 0; row < rowTiles.length; row += 1) {
    for (const tile of rowTiles[row]!) {
      tileRows[tile]!.push(row); pairWords.push(row, tile);
    }
  }
  const frozenTileRows = tileRows.map((rows) => Object.freeze(rows));
  const rowCounts = rowTiles.map((tiles) => tiles.length).sort((a, b) => a - b);
  const tileCounts = tileRows.map((rows) => rows.length).sort((a, b) => a - b);
  const distribution = (values: readonly number[]) => Object.freeze({
    p50: quantile(values, 0.50), p90: quantile(values, 0.90),
    p99: quantile(values, 0.99), maximum: values.at(-1) ?? 0,
  });
  return Object.freeze({ rowCount: rowTiles.length, tileCount,
    edgeCount: pairWords.length / 2,
    rowTiles: Object.freeze(rowTiles), tileRows: Object.freeze(frozenTileRows),
    rowTileCount: distribution(rowCounts),
    tileSubscriberCount: distribution(tileCounts),
    canonicalPairHash: hash128(pairWords),
  });
}

export function sparseCM12VexRowsScheduledByChangedTiles(
  summary: SparseCM12VexDonorTileSummary, changedTiles: readonly number[],
): readonly number[] {
  const rows = new Set<number>();
  for (const rawTile of changedTiles) {
    const tile = u32(rawTile, "VRD1 changed donor tile");
    if (tile >= summary.tileCount) {
      throw new RangeError("VRD1 changed donor tile exceeds capacity");
    }
    for (const row of summary.tileRows[tile]!) rows.add(row);
  }
  return Object.freeze([...rows].sort((left, right) => left - right));
}

export const SPARSE_CM12_VEX_REVERSE_DEPENDENCY_SOURCE_MANIFEST = Object.freeze({
  authority: "FPA1", role: "accepted-index-plus-candidate-read-journal-scaffold",
  initialCSRConstructionOnly: true, runtimeMutationRequired: true,
  productionIntegrated: false,
  consumes: Object.freeze([
    "stable template cell ids", "stable preparation row ids",
    "accepted prior execution actual VEX accessor reads",
    "fixed trace/interpolation/boundary policy provenance",
  ]),
  provides: Object.freeze([
    "fpeaVexReverseBegin", "fpeaVexReverseEnd", "fpeaVexReverseRow",
    "fpeaVexReverseProvenanceValid",
  ]),
  invariants: Object.freeze([
    "generation-paired result and dependency acceptance",
    "unchanged rows inherit accepted dependencies", "stable ascending rows per cell",
    "duplicate-free CSR",
    "no runtime accepted-row scan", "no fallback", "unsupported policy fails closed",
    "FPA1 remains sole preparation work and result authority",
  ]),
  liveMissing: Object.freeze([
    "GPU canonical candidate-journal to accepted-index replacement",
    "bounded paged reverse adjacency mutation and capacity preflight",
    "result plus dependency atomic bank acceptance",
    "ocean donor-cell/donor-tile read-cardinality census",
  ]),
});

export const SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_HEADER_WORDS = 16;
export const SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_ROW_WORDS = 5;
export const SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_CHUNK_HEADER_WORDS = 4;
export const SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_READ_WORDS = 2;
export const SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_READS_PER_CHUNK = 32;
export const SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_HEADER = Object.freeze({
  phase: 0, acceptedGeneration: 1, candidateGeneration: 2,
  expectedRows: 3, coveredRows: 4, allocatedChunks: 5, rawReadCount: 6,
  fault: 7, firstFaultRow: 8, firstFaultCell: 9,
  rowCapacity: 10, chunkCapacity: 11, rowBaseWords: 12, chunkBaseWords: 13,
  chunkWords: 14, reserved: 15,
} as const);
export const SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_PHASE = Object.freeze({
  accepted: 1, collecting: 2, sealed: 3, fault: 4,
} as const);
export const SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_FAULT = Object.freeze({
  none: 0, invalidPhase: 1, generation: 2, invalidRow: 3, invalidCell: 4,
  chunkOverflow: 5, incompleteRow: 6, coverage: 7,
} as const);
export interface SparseCM12VexDependencyJournalLayout {
  readonly baseWords: number; readonly rowCapacity: number;
  readonly cellCapacity: number; readonly chunkCapacity: number;
  readonly readsPerChunk: number; readonly chunkWords: number;
  readonly rowBaseWords: number; readonly chunkBaseWords: number;
  readonly totalWords: number; readonly totalBytes: number;
  readonly maximumRawReads: number;
}
export function createSparseCM12VexDependencyJournalLayout(options: {
  readonly baseWords?: number; readonly rowCapacity: number;
  readonly cellCapacity: number; readonly chunkCapacity: number;
}): SparseCM12VexDependencyJournalLayout {
  const baseWords = align64(u32(options.baseWords ?? 0, "VRDJ base words"));
  const rowCapacity = u32(options.rowCapacity, "VRDJ row capacity", false);
  const cellCapacity = u32(options.cellCapacity, "VRDJ cell capacity", false);
  const chunkCapacity = u32(options.chunkCapacity, "VRDJ chunk capacity", false);
  const readsPerChunk = SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_READS_PER_CHUNK;
  const chunkWords = SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_CHUNK_HEADER_WORDS
    + SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_READ_WORDS * readsPerChunk;
  const rowBaseWords = baseWords + SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_HEADER_WORDS;
  const chunkBaseWords = align64(rowBaseWords
    + rowCapacity * SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_ROW_WORDS);
  const totalWords = align64(chunkBaseWords + chunkCapacity * chunkWords);
  if (totalWords > 0x3fff_ffff) throw new RangeError("VRDJ arena exceeds 4 GiB");
  return Object.freeze({ baseWords, rowCapacity, cellCapacity, chunkCapacity,
    readsPerChunk, chunkWords, rowBaseWords, chunkBaseWords, totalWords,
    totalBytes: 4 * (totalWords - baseWords),
    maximumRawReads: chunkCapacity * readsPerChunk });
}

export function createSparseCM12VexDependencyJournalInitialWords(
  layout: SparseCM12VexDependencyJournalLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_HEADER;
  words[h.phase] = SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_PHASE.accepted;
  words[h.firstFaultRow] = SPARSE_CM12_VEX_REVERSE_DEPENDENCY_INVALID;
  words[h.firstFaultCell] = SPARSE_CM12_VEX_REVERSE_DEPENDENCY_INVALID;
  words[h.rowCapacity] = layout.rowCapacity;
  words[h.chunkCapacity] = layout.chunkCapacity;
  words[h.rowBaseWords] = layout.rowBaseWords;
  words[h.chunkBaseWords] = layout.chunkBaseWords;
  words[h.chunkWords] = layout.chunkWords;
  return words;
}
