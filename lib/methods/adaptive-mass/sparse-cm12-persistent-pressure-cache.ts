import {
  SPARSE_CM12_HOT_TOPOLOGY_EDGE,
  SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_CELL,
  SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_ROW,
  SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS,
  type SparseCM12HotTopology,
  type SparseCM12HotTopologyLayout,
} from "./sparse-cm12-hot-topology";
import type {
  SparseCM12PhaseArenaBuffer,
  SparseCM12PhaseArenaPlan,
  SparseCM12PhaseArenaRegion,
} from "./sparse-cm12-phase-arenas";

/** Persistent coefficient-frontier ABI, version PCF1. */
export const SPARSE_CM12_PRESSURE_CACHE_MAGIC = 0x5043_4631;
export const SPARSE_CM12_PRESSURE_CACHE_VERSION = 1;
export const SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS = 32;
export const SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS = 256;
export const SPARSE_CM12_PRESSURE_CACHE_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_MAGIC = 0x5043_4131; // PCA1
export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS = 32;
export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS = 16;
export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_BRANCH = 32;

export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY = Object.freeze({
  brick: 0, aggregateEdge: 1, hierarchyNode: 2, hierarchyEdge: 3,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER = Object.freeze({
  magic: 0, headerWords: 1, familyHeaderWords: 2, familyCount: 3,
  brickCount: 4, aggregateEdgeCount: 5, hierarchyLevelCount: 6,
  hierarchyNodeCount: 7, hierarchyEdgeCount: 8,
  brickHeaderBase: 9, aggregateEdgeHeaderBase: 10,
  hierarchyNodeHeaderBase: 11, hierarchyEdgeHeaderBase: 12,
  qaFullOracle: 13, firstFaultFamily: 14, firstFaultId: 15,
  topologyGeneration: 16, pcmGeneration: 17, aggregateTopologyGeneration: 18,
  acceptedAggregateTopologyGeneration: 19,
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
  rigidScaleEnabled: 1 << 2,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_PHASE = Object.freeze({
  uninitialized: 0,
  accepted: 1,
  collecting: 2,
  repairing: 3,
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
  invalidCell: 4,
  invalidRow: 5,
  invalidTopology: 6,
  dirtyCapacity: 7,
  coverageGap: 8,
  repairGap: 9,
  nonFiniteCoefficient: 10,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_CAUSE = Object.freeze({
  cellMembership: 1 << 0,
  rowMembership: 1 << 1,
  theta: 1 << 2,
  topology: 1 << 3,
  solidCoefficient: 1 << 4,
  bootstrap: 1 << 5,
} as const);

export const SPARSE_CM12_PRESSURE_CACHE_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, flags: 3,
  cellCount: 4, rowCount: 5, directedEdgeCount: 6, leafBits: 7,
  phase: 8, candidateGeneration: 9, acceptedGeneration: 10,
  fault: 11, firstFaultId: 12,
  dirtyLeafCount: 13, repairIndirectX: 14, repairIndirectY: 15,
  repairIndirectZ: 16, directEventCount: 17, closureWriteCount: 18,
  causeMask: 19, expectedEventCount: 20, coveredEventCount: 21,
  repairedLeafCount: 22, changedEdgeCount: 23, changedDiagonalCount: 24,
  thetaBaseWords: 25, effectiveEdgeBaseWords: 26, diagonalBaseWords: 27,
  dirtyTokenBaseWords: 28, dirtyLeafStampBaseWords: 29,
  dirtyLeafListBaseWords: 30, controlWords: 31,
} as const);

export interface SparseCM12PersistentPressureCacheLayout {
  readonly version: 1;
  readonly bufferId: "cm12.pressure-cache";
  readonly bufferSizeWords: number;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly directedEdgeCount: number;
  readonly thetaBaseWords: number;
  readonly effectiveEdgeBaseWords: number;
  readonly membershipRegionBaseWords: number;
  readonly membershipRegionWords: number;
  readonly controlOffsetWords: number;
  readonly headerBaseWords: number;
  readonly diagonalBaseWords: number;
  readonly dirtyTokenBaseWords: number;
  readonly dirtyLeafStampBaseWords: number;
  readonly dirtyLeafListBaseWords: number;
  readonly dirtyLeafCount: number;
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
  readonly controlWords: number;
  readonly controlEndWords: number;
  readonly rigidScaleEnabled: boolean;
  readonly qaFullOracle: boolean;
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

const bufferByOwner = (plan: SparseCM12PhaseArenaPlan): SparseCM12PhaseArenaBuffer => {
  const matches = plan.buffers.filter((entry) => entry.owner === "PressureCache");
  if (matches.length !== 1 || matches[0]!.id !== "cm12.pressure-cache") {
    throw new Error("PCF1 requires exactly one non-aliased cm12.pressure-cache allocation");
  }
  return matches[0]!;
};

const region = (buffer: SparseCM12PhaseArenaBuffer, name: string):
SparseCM12PhaseArenaRegion => {
  const matches = buffer.regions.filter((entry) => entry.name === name);
  if (matches.length !== 1) throw new Error(`PCF1 requires ${buffer.id}.${name}`);
  const value = matches[0]!;
  if ((value.offsetBytes & 3) !== 0 || (value.sizeBytes & 3) !== 0) {
    throw new Error(`PCF1 region ${name} is not word aligned`);
  }
  return value;
};

/**
 * Maps PCF1 into the phase-arena PressureCache owner. `controlOffsetWords`
 * is relative to the opaque membership region, allowing resident integration
 * to place PCF1 after PCM1 without either ABI knowing the other's size.
 */
export function createSparseCM12PersistentPressureCacheLayout(request: {
  readonly phaseArena: SparseCM12PhaseArenaPlan;
  readonly htp1: SparseCM12HotTopologyLayout;
  readonly controlOffsetWords?: number;
  readonly rigidScaleEnabled?: boolean;
  /** Construction-only full-bake oracle specialization. */
  readonly qaFullOracle?: boolean;
}): SparseCM12PersistentPressureCacheLayout {
  const { phaseArena, htp1 } = request;
  if (phaseArena.input.brickFineResolution !== 16
    || phaseArena.input.presentationPageResolution !== 16
    || htp1.brickFineResolution !== 16 || htp1.presentationPageResolution !== 16) {
    throw new Error("PCF1 is intentionally the B16/P16 pressure-cache ABI");
  }
  if (htp1.cellCount !== phaseArena.input.cellCount
    || htp1.rowCount !== phaseArena.input.rowCount
    || htp1.directedEdgeCount !== phaseArena.input.pressureEdgeCount) {
    throw new Error("PCF1 phase-arena capacities disagree with HTP1");
  }
  const buffer = bufferByOwner(phaseArena);
  if (buffer.immutable || !buffer.usage.includes("STORAGE")) {
    throw new Error("PCF1 requires a mutable storage PressureCache owner");
  }
  const theta = region(buffer, "theta");
  const edges = region(buffer, "effectiveEdgeWeights");
  const membership = region(buffer, "membership");
  const aggregateEdges = region(buffer, "brickAggregateEdgeWeights");
  const aggregateDiagonal = region(buffer, "brickAggregateDiagonal");
  if (theta.element !== "f32" || theta.count !== htp1.rowCount
    || edges.element !== "f32" || edges.count !== htp1.directedEdgeCount
    || membership.element !== "u32") {
    throw new Error("PCF1 phase-arena region types/counts disagree with HTP1");
  }
  const controlOffsetWords = request.controlOffsetWords ?? 0;
  if (!Number.isSafeInteger(controlOffsetWords) || controlOffsetWords < 0) {
    throw new RangeError("PCF1 controlOffsetWords must be a non-negative integer");
  }
  const membershipBase = membership.offsetBytes / 4;
  const membershipWords = membership.sizeBytes / 4;
  const headerBaseWords = alignWords(membershipBase + controlOffsetWords);
  const dirtyLeafCount = Math.ceil(htp1.cellCount / SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS);
  const diagonalBaseWords = alignWords(headerBaseWords
    + SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS);
  const dirtyTokenBaseWords = alignWords(diagonalBaseWords + htp1.cellCount);
  const dirtyLeafStampBaseWords = alignWords(dirtyTokenBaseWords + htp1.cellCount);
  const dirtyLeafListBaseWords = alignWords(dirtyLeafStampBaseWords + dirtyLeafCount);
  let controlEndWords = alignWords(dirtyLeafListBaseWords + dirtyLeafCount);
  const brickAggregateRangeBaseWords = controlEndWords;
  controlEndWords = alignWords(controlEndWords + phaseArena.input.brickCount);
  const aggregateHeaderBaseWords = controlEndWords;
  const aggregateHeaderEnd = aggregateHeaderBaseWords
    + SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS;
  const aggregateFamilyNames: readonly SparseCM12PressureCacheAggregateFamilyName[] =
    ["brick", "aggregateEdge", "hierarchyNode", "hierarchyEdge"];
  const familyHeaderBases = Object.fromEntries(aggregateFamilyNames.map((name, index) =>
    [name, aggregateHeaderEnd
      + index * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS])) as
    Record<SparseCM12PressureCacheAggregateFamilyName, number>;
  controlEndWords = alignWords(aggregateHeaderEnd + aggregateFamilyNames.length
    * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS);
  const hierarchyLevelCounts = phaseArena.input.pressureHierarchy.map((level) =>
    level.groupCount);
  const hierarchyEdgeLevelCounts = phaseArena.input.pressureHierarchy.map((level) =>
    level.edgeCount);
  const prefix = (counts: readonly number[]) => {
    const offsets: number[] = []; let total = 0;
    for (const count of counts) { offsets.push(total); total += count; }
    return { offsets, total };
  };
  const hierarchyNodes = prefix(hierarchyLevelCounts);
  const hierarchyEdges = prefix(hierarchyEdgeLevelCounts);
  const capacities: Record<SparseCM12PressureCacheAggregateFamilyName, number> = {
    brick: phaseArena.input.brickCount,
    aggregateEdge: phaseArena.input.pressureCoarseEdgeCount,
    hierarchyNode: hierarchyNodes.total,
    hierarchyEdge: hierarchyEdges.total,
  };
  const aggregateFamilies = {} as Record<SparseCM12PressureCacheAggregateFamilyName,
  SparseCM12PressureCacheAggregateFamilyLayout>;
  for (const name of aggregateFamilyNames) {
    const capacity = capacities[name];
    const storageCapacity = Math.max(1, capacity);
    const leafCount = Math.max(1,
      Math.ceil(capacity / SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS));
    const candidateGenerationBaseWords = controlEndWords;
    controlEndWords = alignWords(controlEndWords + storageCapacity);
    const bitsBaseWords = controlEndWords;
    controlEndWords = alignWords(controlEndWords + Math.ceil(storageCapacity / 32));
    const dirtyLeafStampBaseWords = controlEndWords;
    controlEndWords = alignWords(controlEndWords + leafCount);
    const dirtyLeafListBaseWords = controlEndWords;
    controlEndWords = alignWords(controlEndWords + leafCount);
    const activeLeafListBaseWords = controlEndWords;
    controlEndWords = alignWords(controlEndWords + leafCount);
    const treeLevelBaseWords: number[] = [], treeLevelCounts: number[] = [];
    for (let count = leafCount;; count = Math.ceil(count
      / SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_BRANCH)) {
      treeLevelBaseWords.push(controlEndWords); treeLevelCounts.push(count);
      controlEndWords = alignWords(controlEndWords + count);
      if (count === 1) break;
    }
    aggregateFamilies[name] = Object.freeze({ headerBaseWords: familyHeaderBases[name],
      capacity, candidateGenerationBaseWords, bitsBaseWords,
      dirtyLeafStampBaseWords, dirtyLeafListBaseWords, activeLeafListBaseWords, leafCount,
      treeLevelBaseWords: Object.freeze(treeLevelBaseWords),
      treeLevelCounts: Object.freeze(treeLevelCounts) });
  }
  const controlWords = controlEndWords - headerBaseWords;
  if (headerBaseWords < membershipBase
    || controlEndWords > membershipBase + membershipWords) {
    throw new RangeError(`PCF1 requires ${controlWords} control words at membership offset `
      + `${headerBaseWords - membershipBase}, but only ${membershipWords} are allocated`);
  }
  const hierarchyEdgeBaseWords = phaseArena.input.pressureHierarchy.map((_, level) =>
    region(buffer, `hierarchy${level}.edgeWeights`).offsetBytes / 4);
  const hierarchyDiagonalBaseWords = phaseArena.input.pressureHierarchy.map((_, level) =>
    region(buffer, `hierarchy${level}.diagonal`).offsetBytes / 4);
  return Object.freeze({
    version: 1, bufferId: "cm12.pressure-cache" as const,
    bufferSizeWords: buffer.sizeBytes / 4,
    cellCount: htp1.cellCount, rowCount: htp1.rowCount,
    directedEdgeCount: htp1.directedEdgeCount,
    thetaBaseWords: theta.offsetBytes / 4,
    effectiveEdgeBaseWords: edges.offsetBytes / 4,
    membershipRegionBaseWords: membershipBase,
    membershipRegionWords: membershipWords,
    controlOffsetWords: headerBaseWords - membershipBase,
    headerBaseWords, diagonalBaseWords, dirtyTokenBaseWords, dirtyLeafStampBaseWords,
    dirtyLeafListBaseWords, dirtyLeafCount, controlWords, controlEndWords,
    brickCount: phaseArena.input.brickCount,
    aggregateEdgeCount: phaseArena.input.pressureCoarseEdgeCount,
    hierarchyLevelCounts: Object.freeze(hierarchyLevelCounts),
    hierarchyLevelOffsets: Object.freeze(hierarchyNodes.offsets),
    hierarchyEdgeLevelCounts: Object.freeze(hierarchyEdgeLevelCounts),
    hierarchyEdgeLevelOffsets: Object.freeze(hierarchyEdges.offsets),
    hierarchyNodeCount: hierarchyNodes.total, hierarchyEdgeCount: hierarchyEdges.total,
    brickAggregateEdgeBaseWords: aggregateEdges.offsetBytes / 4,
    brickAggregateDiagonalBaseWords: aggregateDiagonal.offsetBytes / 4,
    brickAggregateRangeBaseWords,
    hierarchyEdgeBaseWords: Object.freeze(hierarchyEdgeBaseWords),
    hierarchyDiagonalBaseWords: Object.freeze(hierarchyDiagonalBaseWords),
    aggregateHeaderBaseWords, aggregateFamilies: Object.freeze(aggregateFamilies),
    rigidScaleEnabled: request.rigidScaleEnabled ?? false,
    qaFullOracle: request.qaFullOracle ?? false,
  });
}

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
  readonly rigidScaleEnabled?: boolean;
  /** Construction-only full-bake oracle specialization. */
  readonly qaFullOracle?: boolean;
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

  const thetaBaseWords = at; at = alignWords(at + rowCount);
  const effectiveEdgeBaseWords = at; at = alignWords(at + directedEdgeCount);
  const brickAggregateEdgeBaseWords = at; at = alignWords(at + aggregateEdgeCount);
  const brickAggregateDiagonalBaseWords = at; at = alignWords(at + brickCount);
  const brickAggregateRangeBaseWords = at; at = alignWords(at + brickCount);
  const hierarchyEdgeBaseWords = hierarchyEdgeLevelCounts.map((count) => {
    const base = at; at = alignWords(at + count); return base;
  });
  const hierarchyDiagonalBaseWords = hierarchyLevelCounts.map((count) => {
    const base = at; at = alignWords(at + count); return base;
  });
  const membershipRegionBaseWords = at;
  const headerBaseWords = at;
  at = alignWords(at + SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS);
  const diagonalBaseWords = at; at = alignWords(at + cellCount);
  const dirtyTokenBaseWords = at; at = alignWords(at + cellCount);
  const dirtyLeafCount = Math.max(1,
    Math.ceil(cellCount / SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS));
  const dirtyLeafStampBaseWords = at; at = alignWords(at + dirtyLeafCount);
  const dirtyLeafListBaseWords = at; at = alignWords(at + dirtyLeafCount);
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
  const controlWords = controlEndWords - headerBaseWords;
  return Object.freeze({
    version: 1, bufferId: "cm12.pressure-cache" as const,
    bufferSizeWords: controlEndWords, cellCount, rowCount, directedEdgeCount,
    thetaBaseWords, effectiveEdgeBaseWords,
    membershipRegionBaseWords, membershipRegionWords: controlWords,
    controlOffsetWords: 0, headerBaseWords, diagonalBaseWords,
    dirtyTokenBaseWords, dirtyLeafStampBaseWords, dirtyLeafListBaseWords,
    dirtyLeafCount, brickCount, aggregateEdgeCount,
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
    controlWords, controlEndWords,
    rigidScaleEnabled: request.rigidScaleEnabled ?? false,
    qaFullOracle: request.qaFullOracle ?? false,
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
    | SPARSE_CM12_PRESSURE_CACHE_FLAG.validated
    | (layout.rigidScaleEnabled ? SPARSE_CM12_PRESSURE_CACHE_FLAG.rigidScaleEnabled : 0);
  words[base + h.cellCount] = layout.cellCount;
  words[base + h.rowCount] = layout.rowCount;
  words[base + h.directedEdgeCount] = layout.directedEdgeCount;
  words[base + h.leafBits] = SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS;
  words[base + h.phase] = SPARSE_CM12_PRESSURE_CACHE_PHASE.uninitialized;
  words[base + h.firstFaultId] = 0xffff_ffff;
  words[base + h.repairIndirectY] = 1;
  words[base + h.repairIndirectZ] = 1;
  words[base + h.thetaBaseWords] = layout.thetaBaseWords;
  words[base + h.effectiveEdgeBaseWords] = layout.effectiveEdgeBaseWords;
  words[base + h.diagonalBaseWords] = layout.diagonalBaseWords;
  words[base + h.dirtyTokenBaseWords] = layout.dirtyTokenBaseWords;
  words[base + h.dirtyLeafStampBaseWords] = layout.dirtyLeafStampBaseWords;
  words[base + h.dirtyLeafListBaseWords] = layout.dirtyLeafListBaseWords;
  words[base + h.controlWords] = layout.controlWords;
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
  words[aggregate + ah.qaFullOracle] = layout.qaFullOracle ? 1 : 0;
  words[aggregate + ah.firstFaultFamily] = 0xffff_ffff;
  words[aggregate + ah.firstFaultId] = 0xffff_ffff;
  // The legacy full bake authors max(sum, 1e-12) for every brick and every
  // hierarchy node, including inactive ones. Active hierarchy diagonals sum
  // all child brick diagonals, so construction zero is not interchangeable
  // with that floor even though both values are numerically tiny.
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

export interface SparseCM12PressureAuthorityQAInput {
  readonly activeCellBits: Uint32Array;
  readonly activeRowBits: Uint32Array;
  readonly thetaBits: Uint32Array;
  readonly solidScaleBits: Uint32Array;
  readonly fluxWeightBits: Uint32Array;
  readonly faceVelocityBits: Uint32Array;
  readonly rhsCorrectionBits: Uint32Array;
}

export interface SparseCM12PressureAuthorityQAOutput {
  readonly edgeBits: Uint32Array;
  readonly diagonalBits: Uint32Array;
  readonly rhsBits: Uint32Array;
}

const FLOAT_BITS_BUFFER = new ArrayBuffer(4);
const FLOAT_BITS_F32 = new Float32Array(FLOAT_BITS_BUFFER);
const FLOAT_BITS_U32 = new Uint32Array(FLOAT_BITS_BUFFER);
const bitsToF32 = (bits: number): number => {
  FLOAT_BITS_U32[0] = bits >>> 0;
  return FLOAT_BITS_F32[0]!;
};
const f32Bits = (value: number): number => {
  FLOAT_BITS_F32[0] = value;
  return FLOAT_BITS_U32[0]!;
};
const fadd = (a: number, b: number) => Math.fround(Math.fround(a) + Math.fround(b));
const fmul = (a: number, b: number) => Math.fround(Math.fround(a) * Math.fround(b));
const fdiv = (a: number, b: number) => Math.fround(Math.fround(a) / Math.fround(b));
const contains = (bits: Uint32Array, id: number): boolean =>
  (bits[id >>> 5]! & (1 << (id & 31))) !== 0;

const rowTermCount = (topology: SparseCM12HotTopology, row: number): number =>
  topology.words[topology.layout.rowBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS * row + SPARSE_CM12_HOT_TOPOLOGY_ROW.tag]! & 0xffff;

const rowTerm = (topology: SparseCM12HotTopology, row: number, ordinal: number):
readonly [number, number] => {
  const { layout: l, words } = topology;
  const at = l.rowBaseWords + SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS * row;
  const tag = words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.tag]!;
  if ((tag & (1 << 20)) !== 0) {
    return ordinal === 0
      ? [words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm]!,
        words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.coefficient0]!]
      : [words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell1]!,
        words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.coefficient1]!];
  }
  const first = words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm]!;
  return [words[l.variableTermBaseWords + 2 * (first + ordinal)]!,
    words[l.variableTermBaseWords + 2 * (first + ordinal) + 1]!];
};

const checkQAInput = (topology: SparseCM12HotTopology,
  input: SparseCM12PressureAuthorityQAInput): void => {
  const { cellCount, rowCount } = topology.layout;
  if (input.activeCellBits.length !== Math.ceil(cellCount / 32)
    || input.activeRowBits.length !== Math.ceil(rowCount / 32)
    || input.thetaBits.length !== rowCount || input.solidScaleBits.length !== rowCount
    || input.fluxWeightBits.length !== rowCount || input.faceVelocityBits.length !== rowCount
    || input.rhsCorrectionBits.length !== cellCount) {
    throw new RangeError("pressure authority QA input capacities disagree with HTP1");
  }
};

/** Exact f32 reference using HTP1 edge/incidence order and HEAD expressions. */
export function bakeSparseCM12PressureAuthorityQA(
  topology: SparseCM12HotTopology,
  input: SparseCM12PressureAuthorityQAInput,
  target?: SparseCM12PressureAuthorityQAOutput,
  dirtyCells?: ReadonlySet<number>,
): SparseCM12PressureAuthorityQAOutput {
  checkQAInput(topology, input);
  const { layout: l, words } = topology;
  const output = target ?? {
    edgeBits: new Uint32Array(l.directedEdgeCount),
    diagonalBits: new Uint32Array(l.cellCount),
    rhsBits: new Uint32Array(l.cellCount),
  };
  if (output.edgeBits.length !== l.directedEdgeCount
    || output.diagonalBits.length !== l.cellCount || output.rhsBits.length !== l.cellCount) {
    throw new RangeError("pressure authority QA output capacities disagree with HTP1");
  }
  for (let cell = 0; cell < l.cellCount; cell += 1) {
    const repairCoefficient = !dirtyCells || dirtyCells.has(cell);
    const active = contains(input.activeCellBits, cell);
    const edgeFirst = words[l.directedEdgeOffsetBaseWords + cell]!;
    const edgeEnd = words[l.directedEdgeOffsetBaseWords + cell + 1]!;
    for (let edge = edgeFirst; repairCoefficient && edge < edgeEnd; edge += 1) {
      const edgeAt = l.directedEdgeBaseWords + SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS * edge;
      const neighbor = words[edgeAt + SPARSE_CM12_HOT_TOPOLOGY_EDGE.neighbor]!;
      const row = words[edgeAt + SPARSE_CM12_HOT_TOPOLOGY_EDGE.row]!;
      const theta = bitsToF32(input.thetaBits[row]!);
      let weight = 0;
      if (active && contains(input.activeCellBits, neighbor)
        && contains(input.activeRowBits, row) && theta > 0) {
        weight = fdiv(bitsToF32(words[
          edgeAt + SPARSE_CM12_HOT_TOPOLOGY_EDGE.pressureBaseWeight]!), theta);
        weight = fmul(weight, bitsToF32(input.solidScaleBits[row]!));
      }
      output.edgeBits[edge] = f32Bits(weight);
    }
    if (!active) {
      if (repairCoefficient) output.diagonalBits[cell] = 0;
      output.rhsBits[cell] = 0;
      continue;
    }
    let diagonal = repairCoefficient ? 0 : bitsToF32(output.diagonalBits[cell]!);
    let rhs = 0;
    const incidenceFirst = words[l.incidenceOffsetBaseWords + cell]!;
    const incidenceEnd = words[l.incidenceOffsetBaseWords + cell + 1]!;
    for (let incidence = incidenceFirst; incidence < incidenceEnd; incidence += 1) {
      const incidenceAt = l.incidenceBaseWords
        + SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS * incidence;
      const row = words[incidenceAt]!, ordinal = words[incidenceAt + 1]!;
      const theta = bitsToF32(input.thetaBits[row]!);
      if (!contains(input.activeRowBits, row) || theta <= 0) continue;
      const coefficient = bitsToF32(rowTerm(topology, row, ordinal)[1]);
      const rowAt = l.rowBaseWords + SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS * row;
      const dual = bitsToF32(words[rowAt + SPARSE_CM12_HOT_TOPOLOGY_ROW.dualWeight]!);
      if (repairCoefficient) {
        let diagonalTerm = fmul(dual, coefficient);
        diagonalTerm = fmul(diagonalTerm, coefficient);
        diagonal = fadd(diagonal, fdiv(diagonalTerm, theta));
      }
      let rhsTerm = fmul(coefficient, bitsToF32(input.fluxWeightBits[row]!));
      rhsTerm = fmul(rhsTerm, bitsToF32(input.faceVelocityBits[row]!));
      rhs = fadd(rhs, rhsTerm);
    }
    if (repairCoefficient) output.diagonalBits[cell] = f32Bits(diagonal);
    output.rhsBits[cell] = f32Bits(fadd(rhs, bitsToF32(input.rhsCorrectionBits[cell]!)));
  }
  return output;
}

/** Exact cell closure for PCM transitions and row/topology/solid events. */
export function sparseCM12PressureCacheDirtyCellClosure(
  topology: SparseCM12HotTopology,
  events: {
    readonly membershipCells?: readonly number[];
    readonly rows?: readonly number[];
    readonly topologyCells?: readonly number[];
    readonly solidRows?: readonly number[];
  },
): ReadonlySet<number> {
  const { layout: l, words } = topology;
  const dirty = new Set<number>();
  const markRow = (row: number) => {
    if (!Number.isSafeInteger(row) || row < 0 || row >= l.rowCount) {
      throw new RangeError(`invalid PCF1 row event ${row}`);
    }
    for (let ordinal = 0; ordinal < rowTermCount(topology, row); ordinal += 1) {
      dirty.add(rowTerm(topology, row, ordinal)[0]);
    }
  };
  const markCellClosure = (cell: number) => {
    if (!Number.isSafeInteger(cell) || cell < 0 || cell >= l.cellCount) {
      throw new RangeError(`invalid PCF1 cell event ${cell}`);
    }
    dirty.add(cell);
    const first = words[l.incidenceOffsetBaseWords + cell]!;
    const end = words[l.incidenceOffsetBaseWords + cell + 1]!;
    for (let incidence = first; incidence < end; incidence += 1) {
      markRow(words[l.incidenceBaseWords
        + SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS * incidence]!);
    }
  };
  for (const cell of events.membershipCells ?? []) markCellClosure(cell);
  for (const cell of events.topologyCells ?? []) markCellClosure(cell);
  for (const row of events.rows ?? []) markRow(row);
  for (const row of events.solidRows ?? []) markRow(row);
  return dirty;
}

export interface SparseCM12PressureAuthorityQAReceipt {
  readonly exact: boolean;
  readonly firstEdgeMismatch: number;
  readonly firstDiagonalMismatch: number;
  readonly firstRhsMismatch: number;
}

/** Active-domain byte comparison used by the local-vs-full-bake QA oracle. */
export function compareSparseCM12PressureAuthorityQA(
  topology: SparseCM12HotTopology,
  input: SparseCM12PressureAuthorityQAInput,
  local: SparseCM12PressureAuthorityQAOutput,
  full: SparseCM12PressureAuthorityQAOutput,
): SparseCM12PressureAuthorityQAReceipt {
  let firstEdgeMismatch = -1, firstDiagonalMismatch = -1, firstRhsMismatch = -1;
  const { layout: l, words } = topology;
  for (let cell = 0; cell < l.cellCount; cell += 1) {
    if (!contains(input.activeCellBits, cell)) continue;
    const first = words[l.directedEdgeOffsetBaseWords + cell]!;
    const end = words[l.directedEdgeOffsetBaseWords + cell + 1]!;
    for (let edge = first; edge < end; edge += 1) {
      if (firstEdgeMismatch < 0 && local.edgeBits[edge] !== full.edgeBits[edge]) {
        firstEdgeMismatch = edge;
      }
    }
    if (firstDiagonalMismatch < 0
      && local.diagonalBits[cell] !== full.diagonalBits[cell]) firstDiagonalMismatch = cell;
    if (firstRhsMismatch < 0
      && local.rhsBits[cell] !== full.rhsBits[cell]) firstRhsMismatch = cell;
  }
  return Object.freeze({ exact: firstEdgeMismatch < 0 && firstDiagonalMismatch < 0
    && firstRhsMismatch < 0, firstEdgeMismatch, firstDiagonalMismatch, firstRhsMismatch });
}

export interface SparseCM12PressureAggregateEdgeTopology {
  readonly sourceBrick: number;
  readonly targetBrick: number;
  /** HTP1 directed-edge ids, already in canonical packed contribution order. */
  readonly contributionFineEdges: readonly number[];
}

export interface SparseCM12PressureHierarchyEdgeTopology {
  readonly sourceGroup: number;
  readonly targetGroup: number;
  /** Aggregate-edge ids in canonical packed contribution order. */
  readonly contributionAggregateEdges: readonly number[];
}

export interface SparseCM12PressureHierarchyLevelTopology {
  readonly parentsByBrick: readonly number[];
  readonly childrenByGroup: readonly (readonly number[])[];
  readonly internalAggregateEdgesByGroup: readonly (readonly number[])[];
  readonly edges: readonly SparseCM12PressureHierarchyEdgeTopology[];
}

/** Construction representation of the immutable packed aggregate topology. */
export interface SparseCM12PressureAggregateTopology {
  readonly brickCount: number;
  readonly aggregateEdges: readonly SparseCM12PressureAggregateEdgeTopology[];
  readonly hierarchy: readonly SparseCM12PressureHierarchyLevelTopology[];
}

export interface SparseCM12PressureAggregateAuthorityQAOutput {
  readonly aggregateEdgeBits: Uint32Array;
  readonly brickDiagonalBits: Uint32Array;
  readonly hierarchyEdgeBits: readonly Uint32Array[];
  readonly hierarchyDiagonalBits: readonly Uint32Array[];
}

export interface SparseCM12PressureAggregateDirtyClosure {
  readonly bricks: ReadonlySet<number>;
  readonly aggregateEdges: ReadonlySet<number>;
  readonly hierarchyNodes: readonly ReadonlySet<number>[];
  readonly hierarchyEdges: readonly ReadonlySet<number>[];
}

const cellBrick = (topology: SparseCM12HotTopology, cell: number): number =>
  topology.words[topology.layout.cellBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS * cell
    + SPARSE_CM12_HOT_TOPOLOGY_CELL.brickAndResolution]! >>> 5;

const exactLaneReduction = (lanes: Float32Array): number => {
  for (let width = 32; width >= 1; width /= 2) {
    for (let lane = 0; lane < width; lane += 1) {
      lanes[lane] = Math.fround(lanes[lane]! + lanes[lane + width]!);
    }
  }
  return lanes[0]!;
};

const validateAggregateTopology = (topology: SparseCM12HotTopology,
  aggregate: SparseCM12PressureAggregateTopology): void => {
  if (!Number.isSafeInteger(aggregate.brickCount) || aggregate.brickCount < 1) {
    throw new RangeError("PCF1 aggregate QA requires at least one brick");
  }
  for (let cell = 0; cell < topology.layout.cellCount; cell += 1) {
    if (cellBrick(topology, cell) >= aggregate.brickCount) {
      throw new RangeError(`PCF1 cell ${cell} has invalid aggregate brick`);
    }
  }
  aggregate.aggregateEdges.forEach((edge, edgeId) => {
    if (edge.sourceBrick < 0 || edge.sourceBrick >= aggregate.brickCount
      || edge.targetBrick < 0 || edge.targetBrick >= aggregate.brickCount
      || edge.sourceBrick === edge.targetBrick || edge.contributionFineEdges.length < 1) {
      throw new RangeError(`PCF1 aggregate edge ${edgeId} is invalid`);
    }
    for (const fine of edge.contributionFineEdges) {
      if (!Number.isSafeInteger(fine) || fine < 0
        || fine >= topology.layout.directedEdgeCount) {
        throw new RangeError(`PCF1 aggregate edge ${edgeId} has invalid fine edge ${fine}`);
      }
    }
  });
  aggregate.hierarchy.forEach((level, levelIndex) => {
    if (level.parentsByBrick.length !== aggregate.brickCount
      || level.childrenByGroup.length !== level.internalAggregateEdgesByGroup.length) {
      throw new RangeError(`PCF1 hierarchy level ${levelIndex} capacity mismatch`);
    }
    const childOwners = new Uint8Array(aggregate.brickCount);
    level.childrenByGroup.forEach((children, group) => {
      for (const brick of children) {
        if (brick < 0 || brick >= aggregate.brickCount
          || level.parentsByBrick[brick] !== group || childOwners[brick] !== 0) {
          throw new RangeError(`PCF1 hierarchy level ${levelIndex} child ${brick} mismatch`);
        }
        childOwners[brick] = 1;
      }
    });
    if (childOwners.some((owner) => owner === 0)) {
      throw new RangeError(`PCF1 hierarchy level ${levelIndex} misses a brick child`);
    }
    const edgeOwners = new Uint8Array(aggregate.aggregateEdges.length);
    level.internalAggregateEdgesByGroup.forEach((edges, group) => {
      for (const edgeId of edges) {
        const edge = aggregate.aggregateEdges[edgeId];
        if (!edge || level.parentsByBrick[edge.sourceBrick] !== group
          || level.parentsByBrick[edge.targetBrick] !== group
          || edgeOwners[edgeId] !== 0) {
          throw new RangeError(`PCF1 hierarchy level ${levelIndex} internal edge mismatch`);
        }
        edgeOwners[edgeId] = 1;
      }
    });
    level.edges.forEach((edge, hierarchyEdge) => {
      for (const edgeId of edge.contributionAggregateEdges) {
        const aggregateEdge = aggregate.aggregateEdges[edgeId];
        if (!aggregateEdge
          || level.parentsByBrick[aggregateEdge.sourceBrick] !== edge.sourceGroup
          || level.parentsByBrick[aggregateEdge.targetBrick] !== edge.targetGroup
          || edge.sourceGroup === edge.targetGroup || edgeOwners[edgeId] !== 0) {
          throw new RangeError(`PCF1 hierarchy level ${levelIndex} edge ${hierarchyEdge} mismatch`);
        }
        edgeOwners[edgeId] = 1;
      }
    });
    if (edgeOwners.some((owner) => owner !== 1)) {
      throw new RangeError(`PCF1 hierarchy level ${levelIndex} misses an aggregate edge`);
    }
  });
};

/** Exact local owner/ancestor closure for fine coefficient changes. */
export function sparseCM12PressureAggregateDirtyClosure(
  topology: SparseCM12HotTopology,
  aggregate: SparseCM12PressureAggregateTopology,
  dirtyCells: ReadonlySet<number>,
): SparseCM12PressureAggregateDirtyClosure {
  validateAggregateTopology(topology, aggregate);
  const bricks = new Set<number>(), aggregateEdges = new Set<number>();
  const fineToAggregate = new Int32Array(topology.layout.directedEdgeCount).fill(-1);
  aggregate.aggregateEdges.forEach((edge, id) => {
    for (const fine of edge.contributionFineEdges) fineToAggregate[fine] = id;
  });
  for (const cell of dirtyCells) {
    if (!Number.isSafeInteger(cell) || cell < 0 || cell >= topology.layout.cellCount) {
      throw new RangeError(`invalid PCF1 aggregate dirty cell ${cell}`);
    }
    bricks.add(cellBrick(topology, cell));
    const begin = topology.words[topology.layout.directedEdgeOffsetBaseWords + cell]!;
    const end = topology.words[topology.layout.directedEdgeOffsetBaseWords + cell + 1]!;
    for (let edge = begin; edge < end; edge += 1) {
      const aggregateEdge = fineToAggregate[edge]!;
      if (aggregateEdge >= 0) aggregateEdges.add(aggregateEdge);
    }
  }
  const hierarchyNodes = aggregate.hierarchy.map(() => new Set<number>());
  const hierarchyEdges = aggregate.hierarchy.map(() => new Set<number>());
  aggregate.hierarchy.forEach((level, levelIndex) => {
    for (const brick of bricks) hierarchyNodes[levelIndex]!.add(level.parentsByBrick[brick]!);
    const aggregateToHierarchy = new Int32Array(aggregate.aggregateEdges.length).fill(-1);
    level.edges.forEach((edge, id) => {
      for (const coarse of edge.contributionAggregateEdges) aggregateToHierarchy[coarse] = id;
    });
    for (const edge of aggregateEdges) {
      const hierarchyEdge = aggregateToHierarchy[edge]!;
      if (hierarchyEdge >= 0) hierarchyEdges[levelIndex]!.add(hierarchyEdge);
      else {
        const source = aggregate.aggregateEdges[edge]!.sourceBrick;
        hierarchyNodes[levelIndex]!.add(level.parentsByBrick[source]!);
      }
    }
  });
  return Object.freeze({ bricks, aggregateEdges,
    hierarchyNodes: Object.freeze(hierarchyNodes),
    hierarchyEdges: Object.freeze(hierarchyEdges) });
}

/**
 * Exact f32 construction oracle for the existing aggregate/hierarchy bakes.
 * Lane ownership, inner iteration order, and 64-lane reduction order match the
 * resident kernels. Passing `dirty` mutates only the local closure in `target`.
 */
export function bakeSparseCM12PressureAggregateAuthorityQA(
  topology: SparseCM12HotTopology,
  input: SparseCM12PressureAuthorityQAInput,
  fine: SparseCM12PressureAuthorityQAOutput,
  aggregate: SparseCM12PressureAggregateTopology,
  target?: SparseCM12PressureAggregateAuthorityQAOutput,
  dirty?: SparseCM12PressureAggregateDirtyClosure,
): SparseCM12PressureAggregateAuthorityQAOutput {
  checkQAInput(topology, input); validateAggregateTopology(topology, aggregate);
  const output = target ?? {
    aggregateEdgeBits: new Uint32Array(aggregate.aggregateEdges.length),
    brickDiagonalBits: new Uint32Array(aggregate.brickCount),
    hierarchyEdgeBits: aggregate.hierarchy.map((level) => new Uint32Array(level.edges.length)),
    hierarchyDiagonalBits: aggregate.hierarchy.map((level) =>
      new Uint32Array(level.childrenByGroup.length)),
  };
  const cellsByBrick = Array.from({ length: aggregate.brickCount }, () => [] as number[]);
  for (let cell = 0; cell < topology.layout.cellCount; cell += 1) {
    cellsByBrick[cellBrick(topology, cell)]!.push(cell);
  }
  aggregate.aggregateEdges.forEach((edge, id) => {
    if (dirty && !dirty.aggregateEdges.has(id)) return;
    let weight = 0;
    for (const fineEdge of edge.contributionFineEdges) {
      weight = fadd(weight, bitsToF32(fine.edgeBits[fineEdge]!));
    }
    output.aggregateEdgeBits[id] = f32Bits(weight);
  });
  for (let brick = 0; brick < aggregate.brickCount; brick += 1) {
    if (dirty && !dirty.bricks.has(brick)) continue;
    const cells = cellsByBrick[brick]!; const lanes = new Float32Array(64);
    for (let lane = 0; lane < 64; lane += 1) {
      let diagonal = 0;
      for (let local = lane; local < cells.length; local += 64) {
        const cell = cells[local]!;
        if (!contains(input.activeCellBits, cell)) continue;
        diagonal = fadd(diagonal, bitsToF32(fine.diagonalBits[cell]!));
        const begin = topology.words[topology.layout.directedEdgeOffsetBaseWords + cell]!;
        const end = topology.words[topology.layout.directedEdgeOffsetBaseWords + cell + 1]!;
        for (let edge = begin; edge < end; edge += 1) {
          const edgeAt = topology.layout.directedEdgeBaseWords
            + SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS * edge;
          const other = topology.words[edgeAt + SPARSE_CM12_HOT_TOPOLOGY_EDGE.neighbor]!;
          if (cellBrick(topology, other) === brick) {
            diagonal = fadd(diagonal, bitsToF32(fine.edgeBits[edge]!));
          }
        }
      }
      lanes[lane] = diagonal;
    }
    output.brickDiagonalBits[brick] = f32Bits(Math.max(exactLaneReduction(lanes), 1e-12));
  }
  aggregate.hierarchy.forEach((level, levelIndex) => {
    level.edges.forEach((edge, id) => {
      if (dirty && !dirty.hierarchyEdges[levelIndex]!.has(id)) return;
      let weight = 0;
      for (const coarse of edge.contributionAggregateEdges) {
        weight = fadd(weight, bitsToF32(output.aggregateEdgeBits[coarse]!));
      }
      output.hierarchyEdgeBits[levelIndex]![id] = f32Bits(weight);
    });
    for (let group = 0; group < level.childrenByGroup.length; group += 1) {
      if (dirty && !dirty.hierarchyNodes[levelIndex]!.has(group)) continue;
      const children = level.childrenByGroup[group]!;
      const internal = level.internalAggregateEdgesByGroup[group]!;
      const lanes = new Float32Array(64);
      for (let lane = 0; lane < 64; lane += 1) {
        let diagonal = 0;
        for (let at = lane; at < children.length; at += 64) {
          diagonal = fadd(diagonal,
            bitsToF32(output.brickDiagonalBits[children[at]!]!));
        }
        for (let at = lane; at < internal.length; at += 64) {
          diagonal = fadd(diagonal,
            bitsToF32(output.aggregateEdgeBits[internal[at]!]!));
        }
        lanes[lane] = diagonal;
      }
      output.hierarchyDiagonalBits[levelIndex]![group] =
        f32Bits(Math.max(exactLaneReduction(lanes), 1e-12));
    }
  });
  return output;
}

export interface SparseCM12PressureAggregateAuthorityQAReceipt {
  readonly exact: boolean;
  readonly firstAggregateEdgeMismatch: number;
  readonly firstBrickDiagonalMismatch: number;
  readonly firstHierarchyEdgeMismatch: readonly [number, number] | null;
  readonly firstHierarchyDiagonalMismatch: readonly [number, number] | null;
}

export function compareSparseCM12PressureAggregateAuthorityQA(
  local: SparseCM12PressureAggregateAuthorityQAOutput,
  full: SparseCM12PressureAggregateAuthorityQAOutput,
): SparseCM12PressureAggregateAuthorityQAReceipt {
  const first = (left: Uint32Array, right: Uint32Array): number => {
    if (left.length !== right.length) return 0;
    for (let id = 0; id < left.length; id += 1) if (left[id] !== right[id]) return id;
    return -1;
  };
  const firstAggregateEdgeMismatch = first(local.aggregateEdgeBits, full.aggregateEdgeBits);
  const firstBrickDiagonalMismatch = first(local.brickDiagonalBits, full.brickDiagonalBits);
  let firstHierarchyEdgeMismatch: readonly [number, number] | null = null;
  let firstHierarchyDiagonalMismatch: readonly [number, number] | null = null;
  for (let level = 0; level < full.hierarchyEdgeBits.length; level += 1) {
    const edge = first(local.hierarchyEdgeBits[level]!, full.hierarchyEdgeBits[level]!);
    if (edge >= 0 && firstHierarchyEdgeMismatch === null) {
      firstHierarchyEdgeMismatch = [level, edge];
    }
    const node = first(local.hierarchyDiagonalBits[level]!,
      full.hierarchyDiagonalBits[level]!);
    if (node >= 0 && firstHierarchyDiagonalMismatch === null) {
      firstHierarchyDiagonalMismatch = [level, node];
    }
  }
  return Object.freeze({ exact: firstAggregateEdgeMismatch < 0
    && firstBrickDiagonalMismatch < 0 && firstHierarchyEdgeMismatch === null
    && firstHierarchyDiagonalMismatch === null, firstAggregateEdgeMismatch,
  firstBrickDiagonalMismatch, firstHierarchyEdgeMismatch,
  firstHierarchyDiagonalMismatch });
}
