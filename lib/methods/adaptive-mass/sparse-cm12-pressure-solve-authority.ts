/**
 * PSA1: GPU-owned sparse pressure-execution authority.
 *
 * PCM remains the canonical liquid-cell/active-row authority. PSA1 derives two
 * additional canonical domains from it: wet physical bricks and live pressure
 * hierarchy nodes. Both domains use stable ids and count-tree rank select, so
 * sparse dispatch changes invocation count without changing the order of any
 * retained HEAD invocation.
 */
export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_MAGIC = 0x5053_4131; // PSA1
export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_VERSION = 1;
export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER_WORDS = 48;
export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER_WORDS = 16;
export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_LEAF_BITS = 256;
export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_BRANCH = 32;
export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_ALIGNMENT_WORDS = 64;
export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_INVALID = 0xffff_ffff;

export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY = Object.freeze({
  brick: 0,
  hierarchyNode: 1,
} as const);

export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE = Object.freeze({
  uninitialized: 0,
  accepted: 1,
  collecting: 2,
  repairingBricks: 3,
  repairingNodes: 4,
  fault: 5,
} as const);

export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT = Object.freeze({
  none: 0,
  invalidHeader: 1,
  invalidPhase: 2,
  generationExhausted: 3,
  invalidBrick: 4,
  invalidHierarchyNode: 5,
  dirtyCapacity: 6,
  producerCoverageGap: 7,
  topologyGenerationGap: 8,
  pcmGenerationGap: 9,
  pcfGenerationGap: 10,
  countTreeUnderflow: 11,
  countTreeOverflow: 12,
  inactiveHierarchyParent: 13,
  pressureAddressingGap: 15,
} as const);

export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_CAUSE = Object.freeze({
  pcmCellCreated: 1 << 0,
  pcmCellRetired: 1 << 1,
  topologyBrick: 1 << 2,
  solidCoefficient: 1 << 3,
  pcfCoefficient: 1 << 4,
  hierarchyClosure: 1 << 5,
  bootstrap: 1 << 6,
  qaOracle: 1 << 7,
} as const);

export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, familyHeaderWords: 3,
  familyCount: 4, leafBits: 5, branch: 6, brickCapacity: 7,
  hierarchyLevelCount: 8, hierarchyNodeCapacity: 9,
  brickHeaderBase: 10, nodeHeaderBase: 11, totalWords: 12, flags: 13,
  phase: 14, acceptedGeneration: 15, candidateGeneration: 16,
  frameGeneration: 17, topologyGeneration: 18, pcmGeneration: 19,
  pcfGeneration: 20, fault: 21, firstFaultFamily: 22, firstFaultId: 23,
  expectedProducerReceipts: 24, coveredProducerReceipts: 25,
  causeMask: 26, reserved0: 27, reserved1: 28,
  // Construction-only bootstrap/oracle dispatch. Runtime incremental frames
  // receive x=0 from the GPU planner; the host always encodes both paths.
  bootstrapIndirectX: 29, bootstrapIndirectY: 30, bootstrapIndirectZ: 31,
} as const);

export const SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER = Object.freeze({
  activeCount: 0, dirtyLeafCount: 1, directWriteCount: 2,
  closureWriteCount: 3,
  repairIndirectX: 4, repairIndirectY: 5, repairIndirectZ: 6,
  workIndirectX: 7, workIndirectY: 8, workIndirectZ: 9,
  repairedLeafCount: 10, priorActiveCount: 11,
  reserved0: 12, reserved1: 13, reserved2: 14, reserved3: 15,
} as const);

export interface SparseCM12PressureSolveAuthorityFamilyLayout {
  readonly headerBaseWords: number;
  readonly capacity: number;
  readonly activeBitsBaseWords: number;
  readonly activeBitWordCount: number;
  readonly dirtyLeafStampBaseWords: number;
  readonly dirtyLeafListBaseWords: number;
  readonly leafCount: number;
  readonly treeLevelBaseWords: readonly number[];
  readonly treeLevelCounts: readonly number[];
}

export interface SparseCM12PressureSolveAuthorityLayout {
  readonly baseWords: number;
  readonly brickCapacity: number;
  readonly hierarchyLevelCounts: readonly number[];
  /** Prefix sum; level N occupies [offset[N], offset[N] + count[N]). */
  readonly hierarchyLevelOffsets: readonly number[];
  readonly hierarchyNodeCapacity: number;
  readonly brick: SparseCM12PressureSolveAuthorityFamilyLayout;
  readonly hierarchyNode: SparseCM12PressureSolveAuthorityFamilyLayout;
  readonly totalWords: number;
  readonly totalBytes: number;
  /** Construction-specialized and unavailable to runtime policy. */
  readonly qaFullOracle: boolean;
}

const alignWords = (value: number): number => Math.ceil(value
  / SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_ALIGNMENT_WORDS)
  * SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_ALIGNMENT_WORDS;

const checkedCapacity = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0x3fff_ffff) {
    throw new RangeError(`${label} must be an integer in [1, 2^30)`);
  }
  return value;
};

export function createSparseCM12PressureSolveAuthorityLayout(options: {
  readonly brickCapacity: number;
  readonly hierarchyLevelCounts: readonly number[];
  readonly baseWords?: number;
  readonly qaFullOracle?: boolean;
  readonly brickFineResolution?: 16;
  readonly presentationPageResolution?: 16;
}): SparseCM12PressureSolveAuthorityLayout {
  if ((options.brickFineResolution ?? 16) !== 16
    || (options.presentationPageResolution ?? 16) !== 16) {
    throw new Error("PSA1 is intentionally the B16/P16 physical ABI");
  }
  const brickCapacity = checkedCapacity(options.brickCapacity, "PSA1 brickCapacity");
  if (options.hierarchyLevelCounts.length < 1) {
    throw new RangeError("PSA1 requires at least one hierarchy level");
  }
  const hierarchyLevelCounts = options.hierarchyLevelCounts.map((value, level) =>
    checkedCapacity(value, `PSA1 hierarchyLevelCounts[${level}]`));
  const hierarchyLevelOffsets: number[] = [];
  let hierarchyNodeCapacity = 0;
  for (const count of hierarchyLevelCounts) {
    hierarchyLevelOffsets.push(hierarchyNodeCapacity);
    hierarchyNodeCapacity += count;
    if (!Number.isSafeInteger(hierarchyNodeCapacity) || hierarchyNodeCapacity > 0x3fff_ffff) {
      throw new RangeError("PSA1 hierarchy node capacity overflow");
    }
  }
  const baseWords = alignWords(options.baseWords ?? 0);
  const brickHeaderBaseWords = baseWords + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER_WORDS;
  const nodeHeaderBaseWords = brickHeaderBaseWords
    + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER_WORDS;
  let at = alignWords(nodeHeaderBaseWords
    + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER_WORDS);
  const makeFamily = (headerBaseWords: number,
    entityCapacity: number): SparseCM12PressureSolveAuthorityFamilyLayout => {
    const activeBitWordCount = Math.ceil(entityCapacity / 32);
    const leafCount = Math.ceil(entityCapacity
      / SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_LEAF_BITS);
    const activeBitsBaseWords = at; at = alignWords(at + activeBitWordCount);
    const dirtyLeafStampBaseWords = at; at = alignWords(at + leafCount);
    const dirtyLeafListBaseWords = at; at = alignWords(at + leafCount);
    const treeLevelBaseWords: number[] = [];
    const treeLevelCounts: number[] = [];
    for (let count = leafCount;; count = Math.ceil(count
      / SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_BRANCH)) {
      treeLevelBaseWords.push(at); treeLevelCounts.push(count);
      at = alignWords(at + count);
      if (count === 1) break;
    }
    return Object.freeze({ headerBaseWords, capacity: entityCapacity,
      activeBitsBaseWords, activeBitWordCount, dirtyLeafStampBaseWords,
      dirtyLeafListBaseWords, leafCount,
      treeLevelBaseWords: Object.freeze(treeLevelBaseWords),
      treeLevelCounts: Object.freeze(treeLevelCounts) });
  };
  const brick = makeFamily(brickHeaderBaseWords, brickCapacity);
  const hierarchyNode = makeFamily(nodeHeaderBaseWords, hierarchyNodeCapacity);
  return Object.freeze({ baseWords, brickCapacity,
    hierarchyLevelCounts: Object.freeze(hierarchyLevelCounts),
    hierarchyLevelOffsets: Object.freeze(hierarchyLevelOffsets),
    hierarchyNodeCapacity, brick, hierarchyNode,
    totalWords: at, totalBytes: 4 * at, qaFullOracle: options.qaFullOracle ?? false });
}

export function initializeSparseCM12PressureSolveAuthorityWords(
  words: Uint32Array,
  layout: SparseCM12PressureSolveAuthorityLayout,
): void {
  if (words.length < layout.totalWords) throw new RangeError("PSA1 target is smaller than layout");
  const h = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER;
  const base = layout.baseWords;
  words[base + h.magic] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_MAGIC;
  words[base + h.version] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_VERSION;
  words[base + h.headerWords] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER_WORDS;
  words[base + h.familyHeaderWords] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER_WORDS;
  words[base + h.familyCount] = 2;
  words[base + h.leafBits] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_LEAF_BITS;
  words[base + h.branch] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_BRANCH;
  words[base + h.brickCapacity] = layout.brickCapacity;
  words[base + h.hierarchyLevelCount] = layout.hierarchyLevelCounts.length;
  words[base + h.hierarchyNodeCapacity] = layout.hierarchyNodeCapacity;
  words[base + h.brickHeaderBase] = layout.brick.headerBaseWords;
  words[base + h.nodeHeaderBase] = layout.hierarchyNode.headerBaseWords;
  words[base + h.totalWords] = layout.totalWords;
  words[base + h.flags] = layout.qaFullOracle ? 1 : 0;
  words[base + h.phase] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE.uninitialized;
  words[base + h.firstFaultFamily] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_INVALID;
  words[base + h.firstFaultId] = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_INVALID;
  words[base + h.bootstrapIndirectY] = 1;
  words[base + h.bootstrapIndirectZ] = 1;
  const f = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER;
  for (const family of [layout.brick, layout.hierarchyNode]) {
    words[family.headerBaseWords + f.repairIndirectY] = 1;
    words[family.headerBaseWords + f.repairIndirectZ] = 1;
    words[family.headerBaseWords + f.workIndirectY] = 1;
    words[family.headerBaseWords + f.workIndirectZ] = 1;
  }
}

export function createSparseCM12PressureSolveAuthorityInitialWords(
  layout: SparseCM12PressureSolveAuthorityLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords);
  initializeSparseCM12PressureSolveAuthorityWords(words, layout);
  return words;
}

export type SparseCM12PressureSolveAuthorityFamilyName = "brick" | "hierarchyNode";

export function sparseCM12PressureSolveAuthorityIndirectByteOffset(
  layout: SparseCM12PressureSolveAuthorityLayout,
  family: SparseCM12PressureSolveAuthorityFamilyName,
  kind: "repair" | "work",
): number {
  const h = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER;
  return 4 * (layout[family].headerBaseWords
    + (kind === "repair" ? h.repairIndirectX : h.workIndirectX));
}

export interface SparseCM12PressureSolveAuthoritySource {
  readonly kind: "sparse-cm12-pressure-solve-authority";
  readonly arena: GPUBufferBinding;
}

const bit = (words: Uint32Array, id: number): boolean =>
  ((words[id >>> 5] ?? 0) & (1 << (id & 31))) !== 0;

export interface SparseCM12PressureSolveAuthorityQAInput {
  readonly brickWet: readonly boolean[];
  /** One stable parent id per brick for each hierarchy level. */
  readonly hierarchyParentByLevel: readonly (readonly number[])[];
  readonly localBrickBits: Uint32Array;
  readonly localHierarchyNodeBits: Uint32Array;
  /** Optional exact u32 outputs of unchanged HEAD arithmetic, by stable id. */
  readonly fullBrickWords?: readonly (readonly number[])[];
  readonly localBrickWords?: readonly (readonly number[])[];
  readonly fullHierarchyNodeWords?: readonly (readonly number[])[];
  readonly localHierarchyNodeWords?: readonly (readonly number[])[];
}

export interface SparseCM12PressureSolveAuthorityQAReceipt {
  readonly exact: boolean;
  readonly wetBrickCount: number;
  readonly activeHierarchyNodeCount: number;
  readonly firstMismatchFamily: "brick-membership" | "node-membership"
    | "brick-arithmetic" | "node-arithmetic" | null;
  readonly firstMismatchId: number | null;
  readonly firstMismatchWord: number | null;
}

/** Construction/test oracle. It is never consulted by runtime scheduling. */
export function compareSparseCM12PressureSolveAuthorityQA(
  input: SparseCM12PressureSolveAuthorityQAInput,
): SparseCM12PressureSolveAuthorityQAReceipt {
  const levelOffsets: number[] = [];
  let nodes = 0;
  for (const parents of input.hierarchyParentByLevel) {
    if (parents.length !== input.brickWet.length) {
      throw new RangeError("PSA1 QA parent planes must match brick count");
    }
    levelOffsets.push(nodes);
    let maximum = -1;
    for (const parent of parents) {
      if (!Number.isSafeInteger(parent) || parent < 0) {
        throw new RangeError("PSA1 QA parent ids must be non-negative integers");
      }
      maximum = Math.max(maximum, parent);
    }
    nodes += maximum + 1;
  }
  const fullNodes = new Uint8Array(nodes);
  let wetBrickCount = 0;
  input.brickWet.forEach((wet, brick) => {
    if (!wet) return;
    wetBrickCount += 1;
    input.hierarchyParentByLevel.forEach((parents, level) => {
      fullNodes[levelOffsets[level]! + parents[brick]!] = 1;
    });
  });
  const activeHierarchyNodeCount = fullNodes.reduce((sum, value) => sum + value, 0);
  for (let brick = 0; brick < input.brickWet.length; brick += 1) {
    if (bit(input.localBrickBits, brick) !== input.brickWet[brick]) {
      return { exact: false, wetBrickCount, activeHierarchyNodeCount,
        firstMismatchFamily: "brick-membership", firstMismatchId: brick,
        firstMismatchWord: null };
    }
  }
  for (let node = 0; node < nodes; node += 1) {
    if (bit(input.localHierarchyNodeBits, node) !== (fullNodes[node] !== 0)) {
      return { exact: false, wetBrickCount, activeHierarchyNodeCount,
        firstMismatchFamily: "node-membership", firstMismatchId: node,
        firstMismatchWord: null };
    }
  }
  const compareWords = (full: readonly (readonly number[])[] | undefined,
    local: readonly (readonly number[])[] | undefined,
    family: "brick-arithmetic" | "node-arithmetic") => {
    if (full === undefined && local === undefined) return null;
    if (!full || !local || full.length !== local.length) {
      return { family, id: 0, word: 0 } as const;
    }
    for (let id = 0; id < full.length; id += 1) {
      const expected = full[id]!, accepted = local[id]!;
      if (expected.length !== accepted.length) return { family, id, word: 0 } as const;
      for (let word = 0; word < expected.length; word += 1) {
        if ((expected[word]! >>> 0) !== (accepted[word]! >>> 0)) {
          return { family, id, word } as const;
        }
      }
    }
    return null;
  };
  const mismatch = compareWords(input.fullBrickWords, input.localBrickWords,
    "brick-arithmetic") ?? compareWords(input.fullHierarchyNodeWords,
    input.localHierarchyNodeWords, "node-arithmetic");
  return { exact: mismatch === null, wetBrickCount, activeHierarchyNodeCount,
    firstMismatchFamily: mismatch?.family ?? null,
    firstMismatchId: mismatch?.id ?? null, firstMismatchWord: mismatch?.word ?? null };
}
