/**
 * Construction oracle and capacity proof for the footprint-sized Section 5
 * cell authority. This module deliberately has no domain-dimensions input:
 * only accepted rows and proven-reach support identities may consume space.
 *
 * The production GPU table may claim collision-chain slots concurrently. Its
 * physical slot order is not semantic: all consumers reach records through
 * the cell authority, and support indices are opaque tags. This oracle inserts
 * canonical cells in ascending order and emits candidates in authored winner
 * order, giving differential tests one exact dense-path comparison order.
 */

export const OCTREE_AIR_SUPPORT_COMPACT_HASH_RECORD_WORDS = 5;
export const OCTREE_AIR_SUPPORT_COMPACT_MAXIMUM_PROBES = 64;

export interface OctreeAirSupportCompactAuthorityInput {
  readonly cell: number;
  /** Zero admits a query cell whose exact owner size is not known yet. */
  readonly size: number;
  readonly flags: number;
  /** Canonical authored candidate item, or undefined for query-only demand. */
  readonly winner?: number;
  /** Accepted direct row, or undefined for a support identity. */
  readonly directRow?: number;
}

export interface OctreeAirSupportCompactAuthorityRecord {
  readonly cell: number;
  readonly size: number;
  readonly flags: number;
  readonly winner?: number;
  readonly directRow?: number;
  readonly slot: number;
}

export interface OctreeAirSupportCompactAuthorityFailure {
  readonly ok: false;
  readonly reason: "capacity" | "probe-overflow" | "size-mismatch" | "row-mismatch" | "invalid";
  readonly cell: number;
}

export interface OctreeAirSupportCompactAuthorityPublication {
  readonly ok: true;
  readonly hashCapacity: number;
  readonly records: readonly OctreeAirSupportCompactAuthorityRecord[];
  readonly touchedSlots: readonly number[];
  readonly candidates: readonly OctreeAirSupportCompactAuthorityRecord[];
}

export type OctreeAirSupportCompactAuthorityResult =
  OctreeAirSupportCompactAuthorityPublication | OctreeAirSupportCompactAuthorityFailure;

export interface OctreeAirSupportCompactAuthorityPlan {
  readonly rowCapacity: number;
  readonly supportCapacity: number;
  readonly candidateStride: number;
  readonly identityCapacity: number;
  readonly hashCapacity: number;
  readonly rowCandidateCapacity: number;
  readonly fineCandidateCapacity: number;
  readonly candidateCapacity: number;
  readonly candidateBlockCapacity: number;
  readonly radixBlockCapacity: number;
  readonly radixHistogramWords: number;
  readonly offsets: Readonly<{
    control: 0;
    candidates: number;
    ranks: number;
    hashRecords: number;
    touchedSlots: number;
    ownerTouchedSlots: number;
    radixScratch: number;
    radixHistograms: number;
    blockCounts: number;
    blockOffsets: number;
  }>;
  readonly scratchWords: number;
  readonly scratchBytes: number;
  /** A cold zero-initialized table has no live slots to clear. */
  readonly coldClearItems: 0;
  /** All recurring clearing is bounded by this footprint count. */
  readonly maximumLiveClearItems: number;
  /** Production allocation/launch records whose extent is domain volume. */
  readonly domainShapedRecords: 0;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function checkedAdd(label: string, ...values: number[]): number {
  const value = values.reduce((sum, part) => sum + part, 0);
  if (!Number.isSafeInteger(value)) throw new RangeError(`${label} exceeds exact host addressing`);
  return value;
}

function checkedProduct(label: string, ...values: number[]): number {
  const value = values.reduce((product, part) => product * part, 1);
  if (!Number.isSafeInteger(value)) throw new RangeError(`${label} exceeds exact host addressing`);
  return value;
}

/** Exact production-target census. No dimensions or domain volume is accepted. */
export function planOctreeAirSupportCompactAuthority(
  rowCapacityValue: number,
  supportCapacityValue: number,
  candidateStrideValue = 63,
  controlWordsValue = 73,
  workgroupSizeValue = 256,
): OctreeAirSupportCompactAuthorityPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Compact authority row capacity");
  const supportCapacity = positiveInteger(supportCapacityValue, "Compact authority support capacity");
  const candidateStride = positiveInteger(candidateStrideValue, "Compact authority candidate stride");
  const controlWords = positiveInteger(controlWordsValue, "Compact authority control words");
  const workgroupSize = positiveInteger(workgroupSizeValue, "Compact authority workgroup size");
  const identityCapacity = checkedAdd("Compact authority identity capacity", rowCapacity, supportCapacity);
  // At most one direct or support record exists per identity. Two slots per
  // possible live identity is the fixed <= 0.5 load-factor allocation.
  const hashCapacity = checkedProduct("Compact authority hash capacity", 2, identityCapacity);
  const rowCandidateCapacity = checkedProduct("Compact authority row candidate capacity",
    rowCapacity, candidateStride);
  const fineCandidateCapacity = supportCapacity;
  const candidateCapacity = checkedAdd("Compact authority candidate capacity",
    rowCandidateCapacity, fineCandidateCapacity);
  const candidateBlockCapacity = Math.ceil(candidateCapacity / workgroupSize);
  const radixBlockCapacity = Math.ceil(identityCapacity / workgroupSize);
  // Four stable byte passes: one 256-bin histogram per live-list block. The
  // fixed allocation is footprint-shaped even though cold dispatch is zero.
  const radixHistogramWords = checkedProduct("Compact authority radix histograms",
    radixBlockCapacity, 256);
  const offsets = {
    control: 0 as const,
    candidates: controlWords,
    ranks: 0,
    hashRecords: 0,
    touchedSlots: 0,
    ownerTouchedSlots: 0,
    radixScratch: 0,
    radixHistograms: 0,
    blockCounts: 0,
    blockOffsets: 0,
  };
  offsets.ranks = checkedAdd("Compact authority rank offset", offsets.candidates,
    checkedProduct("Compact authority candidate words", 3, candidateCapacity));
  offsets.hashRecords = checkedAdd("Compact authority hash offset", offsets.ranks, candidateCapacity);
  offsets.touchedSlots = checkedAdd("Compact authority touched-list offset", offsets.hashRecords,
    checkedProduct("Compact authority hash words", OCTREE_AIR_SUPPORT_COMPACT_HASH_RECORD_WORDS,
      hashCapacity));
  offsets.ownerTouchedSlots = checkedAdd("Compact authority owner touched-list offset",
    offsets.touchedSlots, identityCapacity);
  offsets.radixScratch = checkedAdd("Compact authority radix scratch offset",
    offsets.ownerTouchedSlots, identityCapacity);
  offsets.radixHistograms = checkedAdd("Compact authority histogram offset",
    offsets.radixScratch, identityCapacity);
  offsets.blockCounts = checkedAdd("Compact authority block-count offset",
    offsets.radixHistograms, radixHistogramWords);
  offsets.blockOffsets = checkedAdd("Compact authority block-offset offset",
    offsets.blockCounts, candidateBlockCapacity);
  const scratchWords = checkedAdd("Compact authority scratch words",
    offsets.blockOffsets, candidateBlockCapacity);
  const scratchBytes = checkedProduct("Compact authority scratch bytes", scratchWords, 4);
  return Object.freeze({
    rowCapacity, supportCapacity, candidateStride, identityCapacity, hashCapacity,
    rowCandidateCapacity, fineCandidateCapacity, candidateCapacity, candidateBlockCapacity,
    radixBlockCapacity, radixHistogramWords, offsets: Object.freeze(offsets),
    scratchWords, scratchBytes, coldClearItems: 0 as const,
    maximumLiveClearItems: identityCapacity, domainShapedRecords: 0 as const,
  });
}

function u32(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0 && value <= 0xffff_fffd;
}

function hashStart(cell: number, capacity: number): number {
  return (Math.imul(cell, 0x9e37_79b1) >>> 0) % capacity;
}

/** Four stable byte passes used by the production-target GPU path. */
export function stableRadixOrderCompactAuthorityRecords(
  recordsValue: readonly OctreeAirSupportCompactAuthorityRecord[],
): readonly OctreeAirSupportCompactAuthorityRecord[] {
  let records = [...recordsValue];
  for (let shift = 0; shift < 32; shift += 8) {
    const bins = Array.from({ length: 256 }, () => [] as OctreeAirSupportCompactAuthorityRecord[]);
    for (const record of records) bins[(record.cell >>> shift) & 0xff]!.push(record);
    records = bins.flat();
  }
  return Object.freeze(records);
}

/**
 * Deterministic construction oracle for the five-word GPU hash. Duplicate
 * query/exact entries are merged before insertion; conflicting size/direct-row
 * claims fail closed. The winner is the lowest authored candidate item.
 */
export function buildOctreeAirSupportCompactAuthority(
  inputs: readonly OctreeAirSupportCompactAuthorityInput[],
  identityCapacityValue: number,
  hashCapacityValue = 2 * identityCapacityValue,
  maximumProbesValue = OCTREE_AIR_SUPPORT_COMPACT_MAXIMUM_PROBES,
): OctreeAirSupportCompactAuthorityResult {
  const identityCapacity = positiveInteger(identityCapacityValue, "Compact authority identity capacity");
  const hashCapacity = positiveInteger(hashCapacityValue, "Compact authority hash capacity");
  const maximumProbes = positiveInteger(maximumProbesValue, "Compact authority maximum probes");
  const canonical = [...inputs].sort((a, b) => a.cell - b.cell
    || a.size - b.size
    || (a.directRow ?? 0xffff_ffff) - (b.directRow ?? 0xffff_ffff)
    || (a.winner ?? 0xffff_ffff) - (b.winner ?? 0xffff_ffff));
  const merged: Array<Omit<OctreeAirSupportCompactAuthorityRecord, "slot">> = [];
  for (const input of canonical) {
    if (!u32(input.cell) || !u32(input.size) || !u32(input.flags)
      || (input.winner !== undefined && !u32(input.winner))
      || (input.directRow !== undefined && !u32(input.directRow))) {
      return Object.freeze({ ok: false, reason: "invalid", cell: input.cell });
    }
    const prior = merged.at(-1);
    if (!prior || prior.cell !== input.cell) {
      merged.push({ ...input });
      continue;
    }
    if (prior.size !== 0 && input.size !== 0 && prior.size !== input.size) {
      return Object.freeze({ ok: false, reason: "size-mismatch", cell: input.cell });
    }
    if (prior.directRow !== undefined && input.directRow !== undefined
      && prior.directRow !== input.directRow) {
      return Object.freeze({ ok: false, reason: "row-mismatch", cell: input.cell });
    }
    merged[merged.length - 1] = {
      cell: prior.cell,
      size: prior.size || input.size,
      flags: (prior.flags | input.flags) >>> 0,
      winner: prior.winner === undefined ? input.winner
        : input.winner === undefined ? prior.winner : Math.min(prior.winner, input.winner),
      directRow: prior.directRow ?? input.directRow,
    };
  }
  if (merged.length > identityCapacity) {
    return Object.freeze({ ok: false, reason: "capacity", cell: merged[identityCapacity]!.cell });
  }
  const slots = new Int32Array(hashCapacity).fill(-1);
  const records: OctreeAirSupportCompactAuthorityRecord[] = [];
  const touchedSlots: number[] = [];
  for (const value of merged) {
    const start = hashStart(value.cell, hashCapacity);
    let slot = -1;
    for (let probe = 0; probe < Math.min(hashCapacity, maximumProbes); probe += 1) {
      const candidate = (start + probe) % hashCapacity;
      if (slots[candidate] === -1) { slot = candidate; break; }
    }
    if (slot < 0) return Object.freeze({ ok: false, reason: "probe-overflow", cell: value.cell });
    slots[slot] = records.length;
    records.push(Object.freeze({ ...value, slot }));
    touchedSlots.push(slot);
  }
  // Fixed row occurrences retain authored winner order. Fine-only identities
  // follow them in canonical cell order, matching the former dense-cell sweep.
  const candidates = records.filter((record) => record.directRow === undefined
    && record.size !== 0 && record.flags !== 0)
    .sort((a, b) => Number(a.winner === undefined) - Number(b.winner === undefined)
      || (a.winner ?? a.cell) - (b.winner ?? b.cell) || a.cell - b.cell);
  return Object.freeze({ ok: true, hashCapacity,
    records: Object.freeze(records), touchedSlots: Object.freeze(touchedSlots),
    candidates: Object.freeze(candidates) });
}

/** Lookup oracle matching bounded linear probing and its empty-slot terminator. */
export function lookupOctreeAirSupportCompactAuthority(
  publication: OctreeAirSupportCompactAuthorityPublication,
  cell: number,
  maximumProbes = OCTREE_AIR_SUPPORT_COMPACT_MAXIMUM_PROBES,
): OctreeAirSupportCompactAuthorityRecord | undefined {
  if (!u32(cell)) return undefined;
  const bySlot = new Map(publication.records.map((record) => [record.slot, record]));
  const start = hashStart(cell, publication.hashCapacity);
  for (let probe = 0; probe < Math.min(publication.hashCapacity, maximumProbes); probe += 1) {
    const slot = (start + probe) % publication.hashCapacity;
    const record = bySlot.get(slot);
    if (!record) return undefined;
    if (record.cell === cell) return record;
  }
  return undefined;
}
