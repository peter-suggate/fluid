/**
 * Canonical compact-work ABI shared by octree pages and rows.
 *
 * The seven-word prefix is deliberately also the indirect-dispatch record:
 * `dispatchWorkgroupsIndirect` consumes words 4..6 while shaders consume the
 * complete header. Payload IDs begin immediately at word 7.
 */

export const OCTREE_WORKSET_HEADER_WORDS = 7 as const;
export const OCTREE_WORKSET_HEADER_BYTES = OCTREE_WORKSET_HEADER_WORDS * 4;
export const OCTREE_WORKSET_PAYLOAD_OFFSET_WORDS = OCTREE_WORKSET_HEADER_WORDS;
export const OCTREE_WORKSET_INDIRECT_OFFSET_WORDS = 4 as const;
export const OCTREE_WORKSET_INDIRECT_OFFSET_BYTES = OCTREE_WORKSET_INDIRECT_OFFSET_WORDS * 4;
export const OCTREE_WORKSET_INVALID_ID = 0xffff_ffff;

export const OCTREE_WORKSET_WORDS = Object.freeze({
  epoch: 0,
  count: 1,
  capacity: 2,
  flags: 3,
  dispatchX: 4,
  dispatchY: 5,
  dispatchZ: 6,
} as const);

export const OCTREE_WORKSET_FLAGS = Object.freeze({
  ready: 1 << 0,
  validated: 1 << 1,
  diagnostic: 1 << 2,
} as const);

export type OctreeDispatchDimensions = readonly [number, number, number];

export interface OctreeWorksetHeader {
  readonly epoch: number;
  readonly count: number;
  readonly capacity: number;
  readonly flags: number;
  readonly dispatch: OctreeDispatchDimensions;
}

export interface OctreeWorksetBuildOptions {
  readonly epoch: number;
  readonly capacity: number;
  readonly workgroupSize: number;
  readonly flags?: number;
  /** Reject duplicate IDs. Enabled by default because worksets are sets. */
  readonly requireUnique?: boolean;
  /** Portable WebGPU limit unless the selected adapter proves a lower limit. */
  readonly maximumWorkgroupsPerDimension?: number;
}

export interface OctreeMarkRankResult {
  /** Exclusive rank for a marked entry, INVALID for an unmarked entry. */
  readonly ranks: Uint32Array;
  readonly count: number;
}

function uint32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must fit uint32`);
  }
  return value >>> 0;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value;
}

/** Advance a published epoch while reserving zero for unpublished storage. */
export function nextOctreeEpoch(epoch: number): number {
  const current = uint32(epoch, "Octree epoch");
  return ((current + 1) >>> 0) || 1;
}

/**
 * RFC-1982-style uint32 serial comparison. This remains correct across wrap
 * provided fewer than 2^31 publications separate the two observations.
 */
export function isOctreeEpochNewer(candidate: number, current: number): boolean {
  const next = uint32(candidate, "Candidate octree epoch");
  const previous = uint32(current, "Current octree epoch");
  if (next === 0 || next === previous) return false;
  const distance = (next - previous) >>> 0;
  return distance < 0x8000_0000;
}

/**
 * Convert element count to a bounded 3-D dispatch. Empty work is always the
 * canonical `(0, 1, 1)` record required by the runtime contract.
 */
export function octreeDispatchForCount(
  countValue: number,
  workgroupSizeValue: number,
  maximumWorkgroupsPerDimensionValue = 65_535,
): OctreeDispatchDimensions {
  const count = uint32(countValue, "Octree workset count");
  const workgroupSize = positiveInteger(workgroupSizeValue, "Octree workgroup size");
  const maximum = positiveInteger(
    maximumWorkgroupsPerDimensionValue,
    "Octree maximum workgroups per dimension",
  );
  if (maximum > 0xffff_ffff) throw new RangeError("Octree workgroup dimension must fit uint32");
  if (count === 0) return [0, 1, 1];
  const groups = Math.ceil(count / workgroupSize);
  if (groups > maximum ** 3) {
    throw new RangeError("Octree workset dispatch exceeds the configured 3-D limit");
  }
  const x = Math.min(groups, maximum);
  const yz = Math.ceil(groups / x);
  const y = Math.min(yz, maximum);
  const z = Math.ceil(yz / y);
  return [x, y, z];
}

/** Deterministic exclusive prefix rank over strict binary marks. */
export function rankOctreeMarks(marks: ArrayLike<number>): OctreeMarkRankResult {
  const ranks = new Uint32Array(marks.length).fill(OCTREE_WORKSET_INVALID_ID);
  let count = 0;
  for (let index = 0; index < marks.length; index += 1) {
    const mark = Number(marks[index]);
    if (mark !== 0 && mark !== 1) {
      throw new RangeError(`Octree mark ${index} must be zero or one`);
    }
    if (mark === 1) {
      ranks[index] = count;
      count += 1;
    }
  }
  return { ranks, count };
}

/**
 * Stable scatter paired with `rankOctreeMarks`. Destination slots are written
 * once in input order, matching the GPU mark/rank/scatter construction.
 */
export function scatterOctreeMarked<T>(
  values: ArrayLike<T>,
  marks: ArrayLike<number>,
  ranking: OctreeMarkRankResult = rankOctreeMarks(marks),
): T[] {
  if (values.length !== marks.length || ranking.ranks.length !== marks.length) {
    throw new RangeError("Octree mark, rank, and source lengths must match");
  }
  const output = new Array<T>(ranking.count);
  const written = new Uint8Array(ranking.count);
  for (let index = 0; index < marks.length; index += 1) {
    const mark = Number(marks[index]);
    const rank = ranking.ranks[index]!;
    if (mark === 0) {
      if (rank !== OCTREE_WORKSET_INVALID_ID) {
        throw new RangeError("Unmarked octree entry has a non-empty rank");
      }
      continue;
    }
    if (mark !== 1 || rank >= ranking.count || written[rank] !== 0) {
      throw new RangeError("Octree rank stream is not a valid exclusive prefix");
    }
    output[rank] = values[index]!;
    written[rank] = 1;
  }
  if (written.some((value) => value === 0)) {
    throw new RangeError("Octree rank stream leaves an unwritten destination");
  }
  return output;
}

/** Mark/rank/scatter convenience helper; source order is preserved exactly. */
export function compactOctreeMarked<T>(
  values: ArrayLike<T>,
  marks: ArrayLike<number>,
): { readonly values: readonly T[]; readonly ranks: Uint32Array } {
  const ranking = rankOctreeMarks(marks);
  return { values: scatterOctreeMarked(values, marks, ranking), ranks: ranking.ranks };
}

/**
 * Build a fixed-capacity workset packet. Unused payload words are INVALID so a
 * malformed count cannot silently consume page zero.
 */
export function buildOctreeWorkset(
  idsValue: Iterable<number>,
  options: OctreeWorksetBuildOptions,
): Uint32Array {
  const epoch = uint32(options.epoch, "Octree workset epoch");
  if (epoch === 0) throw new RangeError("Octree workset epoch zero is reserved");
  const capacity = uint32(options.capacity, "Octree workset capacity");
  const flags = uint32(
    options.flags ?? (OCTREE_WORKSET_FLAGS.ready | OCTREE_WORKSET_FLAGS.validated),
    "Octree workset flags",
  );
  const requiredFlags = OCTREE_WORKSET_FLAGS.ready | OCTREE_WORKSET_FLAGS.validated;
  const knownFlags = requiredFlags | OCTREE_WORKSET_FLAGS.diagnostic;
  if ((flags & requiredFlags) !== requiredFlags || (flags & ~knownFlags) !== 0) {
    throw new RangeError("Octree workset flags must be ready, validated, and otherwise known");
  }
  const ids = [...idsValue].map((id, index) => uint32(id, `Octree workset ID ${index}`));
  if (ids.some((id) => id === OCTREE_WORKSET_INVALID_ID)) {
    throw new RangeError("Octree workset payload cannot contain the reserved invalid ID");
  }
  if (ids.length > capacity) throw new RangeError("Octree workset exceeds fixed capacity");
  if (options.requireUnique !== false && new Set(ids).size !== ids.length) {
    throw new RangeError("Octree workset payload contains duplicate IDs");
  }
  const dispatch = octreeDispatchForCount(
    ids.length,
    options.workgroupSize,
    options.maximumWorkgroupsPerDimension ?? 65_535,
  );
  const words = new Uint32Array(OCTREE_WORKSET_HEADER_WORDS + capacity)
    .fill(OCTREE_WORKSET_INVALID_ID);
  words.set([epoch, ids.length, capacity, flags, ...dispatch], 0);
  words.set(ids, OCTREE_WORKSET_PAYLOAD_OFFSET_WORDS);
  return words;
}

/** Build a workset directly from a deterministic mark/rank/scatter stream. */
export function buildMarkedOctreeWorkset(
  ids: ArrayLike<number>,
  marks: ArrayLike<number>,
  options: OctreeWorksetBuildOptions,
): Uint32Array {
  const compact = compactOctreeMarked(ids, marks);
  return buildOctreeWorkset(compact.values as readonly number[], options);
}

export function unpackOctreeWorksetHeader(words: ArrayLike<number>): OctreeWorksetHeader {
  if (words.length < OCTREE_WORKSET_HEADER_WORDS) {
    throw new RangeError(`Octree workset needs at least ${OCTREE_WORKSET_HEADER_WORDS} words`);
  }
  return {
    epoch: uint32(Number(words[OCTREE_WORKSET_WORDS.epoch]), "Octree workset epoch"),
    count: uint32(Number(words[OCTREE_WORKSET_WORDS.count]), "Octree workset count"),
    capacity: uint32(Number(words[OCTREE_WORKSET_WORDS.capacity]), "Octree workset capacity"),
    flags: uint32(Number(words[OCTREE_WORKSET_WORDS.flags]), "Octree workset flags"),
    dispatch: [
      uint32(Number(words[OCTREE_WORKSET_WORDS.dispatchX]), "Octree dispatch X"),
      uint32(Number(words[OCTREE_WORKSET_WORDS.dispatchY]), "Octree dispatch Y"),
      uint32(Number(words[OCTREE_WORKSET_WORDS.dispatchZ]), "Octree dispatch Z"),
    ],
  };
}

/** Return every ABI violation instead of hiding a later issue behind the first. */
export function validateOctreeWorkset(
  words: ArrayLike<number>,
  expectedEpoch: number,
  workgroupSize: number,
  maximumWorkgroupsPerDimension = 65_535,
): readonly string[] {
  const issues: string[] = [];
  let header: OctreeWorksetHeader;
  try {
    header = unpackOctreeWorksetHeader(words);
  } catch (error) {
    return [error instanceof Error ? error.message : String(error)];
  }
  const epoch = uint32(expectedEpoch, "Expected octree workset epoch");
  if (header.epoch === 0) issues.push("epoch zero is unpublished");
  if (header.epoch !== epoch) issues.push(`epoch ${header.epoch} does not match ${epoch}`);
  if (header.count > header.capacity) issues.push("count exceeds capacity");
  if (words.length !== OCTREE_WORKSET_HEADER_WORDS + header.capacity) {
    issues.push("packet length does not match fixed capacity");
  }
  let expectedDispatch: OctreeDispatchDimensions | undefined;
  try {
    expectedDispatch = octreeDispatchForCount(
      header.count,
      workgroupSize,
      maximumWorkgroupsPerDimension,
    );
  } catch (error) {
    issues.push(error instanceof Error ? error.message : String(error));
  }
  if (expectedDispatch && expectedDispatch.some((value, axis) => value !== header.dispatch[axis])) {
    issues.push("indirect dispatch does not match count");
  }
  const requiredFlags = OCTREE_WORKSET_FLAGS.ready | OCTREE_WORKSET_FLAGS.validated;
  const knownFlags = requiredFlags | OCTREE_WORKSET_FLAGS.diagnostic;
  if ((header.flags & requiredFlags) !== requiredFlags) issues.push("workset is not ready and validated");
  if ((header.flags & ~knownFlags) !== 0) issues.push("workset has unknown flags");
  const seen = new Set<number>();
  const available = Math.max(0, words.length - OCTREE_WORKSET_PAYLOAD_OFFSET_WORDS);
  for (let rank = 0; rank < Math.min(header.count, available); rank += 1) {
    const id = Number(words[OCTREE_WORKSET_PAYLOAD_OFFSET_WORDS + rank]) >>> 0;
    if (id === OCTREE_WORKSET_INVALID_ID) issues.push(`payload ${rank} is invalid`);
    if (seen.has(id)) issues.push(`payload ${rank} duplicates ID ${id}`);
    seen.add(id);
  }
  for (let rank = header.count; rank < Math.min(header.capacity, available); rank += 1) {
    if ((Number(words[OCTREE_WORKSET_PAYLOAD_OFFSET_WORDS + rank]) >>> 0)
      !== OCTREE_WORKSET_INVALID_ID) {
      issues.push(`unused payload ${rank} is not invalid`);
    }
  }
  return issues;
}

export function assertOctreeWorksetValid(
  words: ArrayLike<number>,
  expectedEpoch: number,
  workgroupSize: number,
  maximumWorkgroupsPerDimension = 65_535,
): void {
  const issues = validateOctreeWorkset(
    words,
    expectedEpoch,
    workgroupSize,
    maximumWorkgroupsPerDimension,
  );
  if (issues.length > 0) throw new RangeError(`Invalid octree workset: ${issues.join("; ")}`);
}
