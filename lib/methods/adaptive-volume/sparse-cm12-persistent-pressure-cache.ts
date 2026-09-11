/** Persistent coefficient-frontier ABI, version PCF1. */
export const SPARSE_CM12_PRESSURE_CACHE_MAGIC = 0x5043_4631;
export const SPARSE_CM12_PRESSURE_CACHE_VERSION = 1;
export const SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS = 15;
export const SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS = 256;
export const SPARSE_CM12_PRESSURE_CACHE_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_MAGIC = 0x5043_4131; // PCA1
export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS = 19;
export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS = 16;
export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_BRANCH = 32;

export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER = Object.freeze({
  magic: 0, headerWords: 1, familyHeaderWords: 2, familyCount: 3,
  brickCount: 4, aggregateEdgeCount: 5, hierarchyLevelCount: 6,
  hierarchyNodeCount: 7, hierarchyEdgeCount: 8,
  brickHeaderBase: 9, aggregateEdgeHeaderBase: 10,
  hierarchyNodeHeaderBase: 11, hierarchyEdgeHeaderBase: 12,
  firstFaultFamily: 13, firstFaultId: 14,
  topologyGeneration: 15, pcmGeneration: 16, aggregateTopologyGeneration: 17,
  acceptedAggregateTopologyGeneration: 18,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER = Object.freeze({
  dirtyLeafCount: 0, repairedLeafCount: 1, workCount: 2, executedCount: 3,
  repairIndirectX: 4, repairIndirectY: 5, repairIndirectZ: 6,
  workIndirectX: 7, workIndirectY: 8, workIndirectZ: 9,
  causeMask: 10, previousActiveLeafCount: 11, activeLeafCount: 12,
  seedIndirectX: 13, seedIndirectY: 14, seedIndirectZ: 15,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_FLAG = Object.freeze({
  complete: 1 << 0,
  validated: 1 << 1,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_PHASE = Object.freeze({
  uninitialized: 0,
  accepted: 1,
  collecting: 2,
  fault: 4,
  aggregateRepairing: 5,
  aggregateExecuting: 6,
  hierarchyRepairing: 7,
  hierarchyExecuting: 8,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_FAULT = Object.freeze({
  none: 0,
  invalidHeader: 1,
  generationExhausted: 2,
  invalidPhase: 3,
  invalidTopology: 6,
  dirtyCapacity: 7,
  repairGap: 9,
  nonFiniteCoefficient: 10,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, flags: 3,
  cellCount: 4, rowCount: 5, directedEdgeCount: 6, leafBits: 7,
  phase: 8, candidateGeneration: 9, acceptedGeneration: 10,
  fault: 11, firstFaultId: 12,
  changedEdgeCount: 13, changedDiagonalCount: 14,
} as const);

export interface SparseCM12PersistentPressureCacheLayout {
  readonly version: 1;
  readonly bufferId: "cm12.pressure-cache";
  readonly baseWords: number;
  readonly bufferSizeWords: number;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly directedEdgeCount: number;
  readonly headerBaseWords: number;
  readonly brickCount: number;
  readonly aggregateEdgeCount: number;
  readonly hierarchyLevelCounts: readonly number[];
  readonly hierarchyLevelOffsets: readonly number[];
  readonly hierarchyEdgeLevelCounts: readonly number[];
  readonly hierarchyEdgeLevelOffsets: readonly number[];
  readonly hierarchyNodeCount: number;
  readonly hierarchyEdgeCount: number;
  readonly brickAggregateEdgeBaseWords: number;
  readonly brickAggregateDiagonalBaseWords: number;
  readonly brickAggregateRangeBaseWords: number;
  readonly hierarchyEdgeBaseWords: readonly number[];
  readonly hierarchyDiagonalBaseWords: readonly number[];
  readonly aggregateHeaderBaseWords: number;
  readonly aggregateFamilies: Readonly<Record<SparseCM12PressureCacheAggregateFamilyName,
    SparseCM12PressureCacheAggregateFamilyLayout>>;
  readonly controlEndWords: number;
}

export type SparseCM12PressureCacheAggregateFamilyName =
  "brick" | "aggregateEdge" | "hierarchyNode" | "hierarchyEdge";

export interface SparseCM12PressureCacheAggregateFamilyLayout {
  readonly headerBaseWords: number;
  readonly capacity: number;
  readonly candidateGenerationBaseWords: number;
  readonly bitsBaseWords: number;
  readonly dirtyLeafStampBaseWords: number;
  readonly dirtyLeafListBaseWords: number;
  readonly activeLeafListBaseWords: number;
  readonly leafCount: number;
  readonly treeLevelBaseWords: readonly number[];
  readonly treeLevelCounts: readonly number[];
}

const alignWords = (value: number): number => Math.ceil(value
  / SPARSE_CM12_PRESSURE_CACHE_ALIGNMENT_WORDS) * SPARSE_CM12_PRESSURE_CACHE_ALIGNMENT_WORDS;

/**
 * Places the same PCF1/PCA1 ABI in an existing atomic resident arena.  This is
 * the production migration seam for the legacy resident, whose immutable
 * topology and mutable command state already share one storage binding.  The
 * returned word addresses are absolute in that arena; no shader-side base
 * translation or aliased candidate storage is permitted.
 */
export function createSparseCM12ResidentPersistentPressureCacheLayout(request: {
  readonly baseWords: number;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly directedEdgeCount: number;
  readonly brickCount: number;
  readonly aggregateEdgeCount: number;
  readonly hierarchyLevelCounts: readonly number[];
  readonly hierarchyEdgeLevelCounts: readonly number[];
}): SparseCM12PersistentPressureCacheLayout {
  const checked = (value: number, name: string): number => {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_fffe) {
      throw new RangeError(`PCF1 ${name} must be an addressable u32 count`);
    }
    return value;
  };
  let at = alignWords(checked(request.baseWords, "baseWords"));
  const cellCount = checked(request.cellCount, "cellCount");
  const rowCount = checked(request.rowCount, "rowCount");
  const directedEdgeCount = checked(request.directedEdgeCount, "directedEdgeCount");
  const brickCount = checked(request.brickCount, "brickCount");
  const aggregateEdgeCount = checked(request.aggregateEdgeCount, "aggregateEdgeCount");
  if (request.hierarchyLevelCounts.length !== request.hierarchyEdgeLevelCounts.length
    || request.hierarchyLevelCounts.length === 0) {
    throw new RangeError("PCF1 resident hierarchy count vectors must be non-empty/equal");
  }
  const hierarchyLevelCounts = request.hierarchyLevelCounts.map((count, level) =>
    checked(count, `hierarchyLevelCounts[${level}]`));
  const hierarchyEdgeLevelCounts = request.hierarchyEdgeLevelCounts.map((count, level) =>
    checked(count, `hierarchyEdgeLevelCounts[${level}]`));
  const prefix = (counts: readonly number[]) => {
    const offsets: number[] = []; let total = 0;
    for (const count of counts) { offsets.push(total); total += count; }
    return { offsets, total: checked(total, "hierarchy total") };
  };
  const hierarchyNodes = prefix(hierarchyLevelCounts);
  const hierarchyEdges = prefix(hierarchyEdgeLevelCounts);

  const baseWords = at;
  const brickAggregateEdgeBaseWords = at; at = alignWords(at + aggregateEdgeCount);
  const brickAggregateDiagonalBaseWords = at; at = alignWords(at + brickCount);
  const brickAggregateRangeBaseWords = at; at = alignWords(at + brickCount);
  const hierarchyEdgeBaseWords = hierarchyEdgeLevelCounts.map((count) => {
    const base = at; at = alignWords(at + count); return base;
  });
  const hierarchyDiagonalBaseWords = hierarchyLevelCounts.map((count) => {
    const base = at; at = alignWords(at + count); return base;
  });
  const headerBaseWords = at;
  at = alignWords(at + SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS);
  const aggregateHeaderBaseWords = at;
  at = alignWords(at + SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS
    + 4 * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS);
  const familyNames: readonly SparseCM12PressureCacheAggregateFamilyName[] =
    ["brick", "aggregateEdge", "hierarchyNode", "hierarchyEdge"];
  const familyHeaderBase = aggregateHeaderBaseWords
    + SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS;
  const capacities: Readonly<Record<SparseCM12PressureCacheAggregateFamilyName, number>> = {
    brick: brickCount, aggregateEdge: aggregateEdgeCount,
    hierarchyNode: hierarchyNodes.total, hierarchyEdge: hierarchyEdges.total,
  };
  const aggregateFamilies = {} as Record<SparseCM12PressureCacheAggregateFamilyName,
  SparseCM12PressureCacheAggregateFamilyLayout>;
  familyNames.forEach((name, familyIndex) => {
    const capacity = capacities[name];
    const storageCapacity = Math.max(1, capacity);
    const leafCount = Math.max(1,
      Math.ceil(capacity / SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS));
    const candidateGenerationBaseWords = at; at = alignWords(at + storageCapacity);
    const bitsBaseWords = at; at = alignWords(at + Math.ceil(storageCapacity / 32));
    const dirtyLeafStampBaseWords = at; at = alignWords(at + leafCount);
    const dirtyLeafListBaseWords = at; at = alignWords(at + leafCount);
    const activeLeafListBaseWords = at; at = alignWords(at + leafCount);
    const treeLevelBaseWords: number[] = [], treeLevelCounts: number[] = [];
    for (let count = leafCount;; count = Math.ceil(count
      / SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_BRANCH)) {
      treeLevelBaseWords.push(at); treeLevelCounts.push(count);
      at = alignWords(at + count);
      if (count === 1) break;
    }
    aggregateFamilies[name] = Object.freeze({
      headerBaseWords: familyHeaderBase
        + familyIndex * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS,
      capacity, candidateGenerationBaseWords, bitsBaseWords,
      dirtyLeafStampBaseWords, dirtyLeafListBaseWords, activeLeafListBaseWords,
      leafCount, treeLevelBaseWords: Object.freeze(treeLevelBaseWords),
      treeLevelCounts: Object.freeze(treeLevelCounts),
    });
  });
  const controlEndWords = at;
  return Object.freeze({
    version: 1, bufferId: "cm12.pressure-cache" as const,
    baseWords,
    bufferSizeWords: controlEndWords, cellCount, rowCount, directedEdgeCount,
    headerBaseWords,
    brickCount, aggregateEdgeCount,
    hierarchyLevelCounts: Object.freeze(hierarchyLevelCounts),
    hierarchyLevelOffsets: Object.freeze(hierarchyNodes.offsets),
    hierarchyEdgeLevelCounts: Object.freeze(hierarchyEdgeLevelCounts),
    hierarchyEdgeLevelOffsets: Object.freeze(hierarchyEdges.offsets),
    hierarchyNodeCount: hierarchyNodes.total, hierarchyEdgeCount: hierarchyEdges.total,
    brickAggregateEdgeBaseWords, brickAggregateDiagonalBaseWords,
    brickAggregateRangeBaseWords,
    hierarchyEdgeBaseWords: Object.freeze(hierarchyEdgeBaseWords),
    hierarchyDiagonalBaseWords: Object.freeze(hierarchyDiagonalBaseWords),
    aggregateHeaderBaseWords, aggregateFamilies: Object.freeze(aggregateFamilies),
    controlEndWords,
  });
}

export function initializeSparseCM12PersistentPressureCacheWords(
  words: Uint32Array,
  layout: SparseCM12PersistentPressureCacheLayout,
): void {
  if (words.length < layout.bufferSizeWords) {
    throw new RangeError("PCF1 target is smaller than the PressureCache phase arena");
  }
  const h = SPARSE_CM12_PRESSURE_CACHE_HEADER;
  const base = layout.headerBaseWords;
  words[base + h.magic] = SPARSE_CM12_PRESSURE_CACHE_MAGIC;
  words[base + h.version] = SPARSE_CM12_PRESSURE_CACHE_VERSION;
  words[base + h.headerWords] = SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS;
  words[base + h.flags] = SPARSE_CM12_PRESSURE_CACHE_FLAG.complete
    | SPARSE_CM12_PRESSURE_CACHE_FLAG.validated;
  words[base + h.cellCount] = layout.cellCount;
  words[base + h.rowCount] = layout.rowCount;
  words[base + h.directedEdgeCount] = layout.directedEdgeCount;
  words[base + h.leafBits] = SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS;
  words[base + h.phase] = SPARSE_CM12_PRESSURE_CACHE_PHASE.uninitialized;
  words[base + h.firstFaultId] = 0xffff_ffff;
  const ah = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER;
  const aggregate = layout.aggregateHeaderBaseWords;
  words[aggregate + ah.magic] = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_MAGIC;
  words[aggregate + ah.headerWords] = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS;
  words[aggregate + ah.familyHeaderWords] =
    SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS;
  words[aggregate + ah.familyCount] = 4;
  words[aggregate + ah.brickCount] = layout.brickCount;
  words[aggregate + ah.aggregateEdgeCount] = layout.aggregateEdgeCount;
  words[aggregate + ah.hierarchyLevelCount] = layout.hierarchyLevelCounts.length;
  words[aggregate + ah.hierarchyNodeCount] = layout.hierarchyNodeCount;
  words[aggregate + ah.hierarchyEdgeCount] = layout.hierarchyEdgeCount;
  words[aggregate + ah.brickHeaderBase] = layout.aggregateFamilies.brick.headerBaseWords;
  words[aggregate + ah.aggregateEdgeHeaderBase] =
    layout.aggregateFamilies.aggregateEdge.headerBaseWords;
  words[aggregate + ah.hierarchyNodeHeaderBase] =
    layout.aggregateFamilies.hierarchyNode.headerBaseWords;
  words[aggregate + ah.hierarchyEdgeHeaderBase] =
    layout.aggregateFamilies.hierarchyEdge.headerBaseWords;
  words[aggregate + ah.firstFaultFamily] = 0xffff_ffff;
  words[aggregate + ah.firstFaultId] = 0xffff_ffff;
  // Active hierarchy diagonals sum all child brick diagonals. Initialize the
  // persistent inactive values to the same positive floor used by repair.
  const diagonalFloorBits = new Uint32Array(new Float32Array([1e-12]).buffer)[0]!;
  words.fill(diagonalFloorBits, layout.brickAggregateDiagonalBaseWords,
    layout.brickAggregateDiagonalBaseWords + layout.brickCount);
  for (let level = 0; level < layout.hierarchyDiagonalBaseWords.length; level += 1) {
    const first = layout.hierarchyDiagonalBaseWords[level]!;
    words.fill(diagonalFloorBits, first, first + layout.hierarchyLevelCounts[level]!);
  }
  const fh = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER;
  for (const family of Object.values(layout.aggregateFamilies)) {
    words[family.headerBaseWords + fh.repairIndirectY] = 1;
    words[family.headerBaseWords + fh.repairIndirectZ] = 1;
    words[family.headerBaseWords + fh.workIndirectY] = 1;
    words[family.headerBaseWords + fh.workIndirectZ] = 1;
    words[family.headerBaseWords + fh.seedIndirectY] = 1;
    words[family.headerBaseWords + fh.seedIndirectZ] = 1;
  }
}

export function sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
  layout: SparseCM12PersistentPressureCacheLayout,
  family: SparseCM12PressureCacheAggregateFamilyName,
  kind: "repair" | "work" | "seed",
): number {
  const h = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER;
  return 4 * (layout.aggregateFamilies[family].headerBaseWords
    + (kind === "repair" ? h.repairIndirectX
      : kind === "work" ? h.workIndirectX : h.seedIndirectX));
}
