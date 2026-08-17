/** TWA1: generation-stamped local authority for Sparse CM12 topology rebuilds. */
export const SPARSE_CM12_TWA_MAGIC = 0x5457_4131; // TWA1
export const SPARSE_CM12_TWA_VERSION = 1;
export const SPARSE_CM12_TWA_HEADER_WORDS = 64;
export const SPARSE_CM12_TWA_FAMILY_WORDS = 8;
export const SPARSE_CM12_TWA_CHANGE_WORDS = 12;

export const SPARSE_CM12_TWA_PHASE = Object.freeze({
  accepted: 1, collecting: 2, sealed: 3, fault: 4, rejected: 5,
} as const);
export const SPARSE_CM12_TWA_FAULT = Object.freeze({
  none: 0, header: 1, phase: 2, generation: 3, brick: 4,
  capacity: 5, missingExecution: 6, uncoveredWrite: 7,
  treeRoot: 8, runtimeOracle: 9,
} as const);
export const SPARSE_CM12_TWA_CAUSE = Object.freeze({
  rungChanged: 1 << 0, activated: 1 << 1, retired: 1 << 2,
  pageIdentity: 1 << 3, incidenceClosure: 1 << 4, bootstrap: 1 << 5,
} as const);
export const SPARSE_CM12_TWA_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, constructionMode: 3,
  brickCapacity: 4, cellCapacity: 5, rowCapacity: 6,
  phase: 7, fault: 8, firstFaultId: 9, acceptedGeneration: 10,
  candidateGeneration: 11, topologyGeneration: 12, changeCount: 13,
  brickCount: 14, cellCount: 15, rowCount: 16, familyCount: 17,
  familyBase: 18, changeBase: 19, brickStampBase: 20,
  brickTreeBase: 21, brickListBase: 22, cellStampBase: 23,
  cellTreeBase: 24, cellListBase: 25, rowStampBase: 26,
  rowTreeBase: 27, rowListBase: 28, acceptedFrames: 29,
  rejectedFrames: 30, uncoveredWriteCount: 31, firstUncoveredOwner: 32,
  expectedReceipts: 33, coveredReceipts: 34, causeMask: 35,
  maximumClosureDepth: 36, reservedBase: 37,
} as const);
export const SPARSE_CM12_TWA_CHANGE = Object.freeze({
  brick: 0, generation: 1, causeMask: 2,
  oldCellFirst: 3, oldCellCount: 4, newCellFirst: 5, newCellCount: 6,
  oldRowFirst: 7, oldRowCount: 8, newRowFirst: 9, newRowCount: 10,
  reserved: 11,
} as const);

export const SPARSE_CM12_TWA_FAMILY = Object.freeze({
  allocatePagesBricks: 0, synthesizeCellPagesBricks: 1,
  buildShadowCells: 2, buildShadowRows: 3, transferCellsBricks: 4,
  prepareFaceReceiptsBricks: 5, transferFaces: 6,
  writeShadowCellsBricks: 7, reconstructShadowFacesRows: 8,
  validateChangedBricks: 9, postTopologyBricks: 10,
} as const);
export type SparseCM12TWAFamily = keyof typeof SPARSE_CM12_TWA_FAMILY;
export const SPARSE_CM12_TWA_FAMILY_COUNT = 11;
export const SPARSE_CM12_TWA_PLANNER_FAMILY = Object.freeze({
  clearPriorBricks: 11, clearPriorCells: 12, clearPriorRows: 13,
  expandChangedBricks: 14, writeBrickList: 15, writeCellList: 16, writeRowList: 17,
} as const);
export const SPARSE_CM12_TWA_INDIRECT_FAMILY_COUNT = 18;
export type SparseCM12TWAConstructionMode = "temporal" | "immutable-full-oracle";

export interface SparseCM12TWALayout {
  readonly brickCapacity: number; readonly cellCapacity: number; readonly rowCapacity: number;
  readonly headerBaseWords: number; readonly familyBaseWords: number;
  readonly changeBaseWords: number; readonly brickStampBaseWords: number;
  readonly brickTreeBaseWords: number; readonly brickTreeLeafCapacity: number;
  readonly brickListBaseWords: number; readonly cellStampBaseWords: number;
  readonly cellTreeBaseWords: number; readonly cellTreeLeafCapacity: number;
  readonly cellListBaseWords: number; readonly rowStampBaseWords: number;
  readonly rowTreeBaseWords: number; readonly rowTreeLeafCapacity: number;
  readonly rowListBaseWords: number; readonly totalWords: number; readonly totalBytes: number;
  readonly indirectWords: number; readonly indirectBytes: number;
}

export interface SparseCM12TWAChange {
  readonly brick: number; readonly causeMask: number;
  readonly oldCellFirst: number; readonly oldCellCount: number;
  readonly newCellFirst: number; readonly newCellCount: number;
  readonly oldRowFirst: number; readonly oldRowCount: number;
  readonly newRowFirst: number; readonly newRowCount: number;
}
export interface SparseCM12TWATopology {
  readonly brickCapacity: number; readonly cellCapacity: number; readonly rowCapacity: number;
  /** Immutable accepted-template incidence CSR used for exact local row closure. */
  readonly cellRows: readonly (readonly number[])[];
}
export interface SparseCM12TWAPlan {
  readonly generation: number; readonly topologyGeneration: number;
  readonly causeMask: number; readonly maximumClosureDepth: number;
  readonly changedBricks: readonly number[]; readonly closureCells: readonly number[];
  readonly closureRows: readonly number[];
  readonly families: Readonly<Record<SparseCM12TWAFamily, readonly number[]>>;
  readonly indirect: Readonly<Record<SparseCM12TWAFamily,
    readonly [number, number, number]>>;
}
export interface SparseCM12TWA {
  readonly topology: SparseCM12TWATopology;
  readonly constructionMode: SparseCM12TWAConstructionMode;
  phase: number; fault: number; firstFaultId: number;
  acceptedGeneration: number; candidateGeneration: number; topologyGeneration: number;
  readonly changes: Map<number, SparseCM12TWAChange>;
  plan?: SparseCM12TWAPlan; readonly executed: Set<SparseCM12TWAFamily>;
  expectedReceipts: number; coveredReceipts: number;
  uncoveredWriteCount: number; firstUncoveredOwner: number;
}

const integer = (value: number, label: string, upper?: number): number => {
  if (!Number.isSafeInteger(value) || value < 0 || (upper !== undefined && value >= upper)) {
    throw new RangeError(label);
  } return value;
};
const stable = (values: Iterable<number>): readonly number[] =>
  Object.freeze([...new Set(values)].sort((a, b) => a - b));
const align64 = (words: number): number => Math.ceil(words / 64) * 64;
const power2 = (value: number): number => {
  let result = 1; while (result < value) result *= 2; return result;
};
const range = (first: number, count: number, capacity: number): readonly number[] => {
  integer(first, "TWA range first"); integer(count, "TWA range count");
  if (first + count > capacity) throw new RangeError("TWA range capacity");
  return Array.from({ length: count }, (_, offset) => first + offset);
};

export function createSparseCM12TWALayout(options: {
  readonly brickCapacity: number; readonly cellCapacity: number; readonly rowCapacity: number;
  readonly baseWords?: number;
}): SparseCM12TWALayout {
  const brickCapacity = integer(options.brickCapacity, "TWA brick capacity");
  const cellCapacity = integer(options.cellCapacity, "TWA cell capacity");
  const rowCapacity = integer(options.rowCapacity, "TWA row capacity");
  if (Math.min(brickCapacity, cellCapacity, rowCapacity) < 1) {
    throw new RangeError("TWA capacity");
  }
  const headerBaseWords = align64(options.baseWords ?? 0);
  const familyBaseWords = headerBaseWords + SPARSE_CM12_TWA_HEADER_WORDS;
  const changeBaseWords = align64(familyBaseWords
    + SPARSE_CM12_TWA_FAMILY_WORDS * SPARSE_CM12_TWA_FAMILY_COUNT);
  const brickStampBaseWords = align64(changeBaseWords
    + SPARSE_CM12_TWA_CHANGE_WORDS * brickCapacity);
  const brickTreeLeafCapacity = power2(brickCapacity);
  const brickTreeBaseWords = align64(brickStampBaseWords + 2 * brickCapacity);
  const brickListBaseWords = align64(brickTreeBaseWords + 2 * brickTreeLeafCapacity - 1);
  const cellStampBaseWords = align64(brickListBaseWords + brickCapacity);
  const cellTreeLeafCapacity = power2(cellCapacity);
  const cellTreeBaseWords = align64(cellStampBaseWords + cellCapacity);
  const cellListBaseWords = align64(cellTreeBaseWords + 2 * cellTreeLeafCapacity - 1);
  const rowStampBaseWords = align64(cellListBaseWords + cellCapacity);
  const rowTreeLeafCapacity = power2(rowCapacity);
  const rowTreeBaseWords = align64(rowStampBaseWords + rowCapacity);
  const rowListBaseWords = align64(rowTreeBaseWords + 2 * rowTreeLeafCapacity - 1);
  const totalWords = align64(rowListBaseWords + rowCapacity);
  const indirectWords = 3 * SPARSE_CM12_TWA_INDIRECT_FAMILY_COUNT;
  return Object.freeze({ brickCapacity, cellCapacity, rowCapacity, headerBaseWords,
    familyBaseWords, changeBaseWords, brickStampBaseWords, brickTreeBaseWords,
    brickTreeLeafCapacity, brickListBaseWords, cellStampBaseWords, cellTreeBaseWords,
    cellTreeLeafCapacity, cellListBaseWords, rowStampBaseWords, rowTreeBaseWords,
    rowTreeLeafCapacity, rowListBaseWords, totalWords, totalBytes: 4 * totalWords,
    indirectWords, indirectBytes: 4 * indirectWords });
}

export function createSparseCM12TWAInitialWords(options: {
  readonly layout: SparseCM12TWALayout;
  readonly constructionMode?: SparseCM12TWAConstructionMode;
}): Uint32Array {
  const { layout } = options; const words = new Uint32Array(layout.totalWords);
  const h = SPARSE_CM12_TWA_HEADER; const base = layout.headerBaseWords;
  words[base + h.magic] = SPARSE_CM12_TWA_MAGIC;
  words[base + h.version] = SPARSE_CM12_TWA_VERSION;
  words[base + h.headerWords] = SPARSE_CM12_TWA_HEADER_WORDS;
  words[base + h.constructionMode]
    = options.constructionMode === "immutable-full-oracle" ? 1 : 0;
  words[base + h.brickCapacity] = layout.brickCapacity;
  words[base + h.cellCapacity] = layout.cellCapacity;
  words[base + h.rowCapacity] = layout.rowCapacity;
  words[base + h.phase] = SPARSE_CM12_TWA_PHASE.accepted;
  words[base + h.firstFaultId] = 0xffff_ffff;
  words[base + h.firstUncoveredOwner] = 0xffff_ffff;
  words[base + h.familyCount] = SPARSE_CM12_TWA_FAMILY_COUNT;
  words[base + h.familyBase] = layout.familyBaseWords;
  words[base + h.changeBase] = layout.changeBaseWords;
  words[base + h.brickStampBase] = layout.brickStampBaseWords;
  words[base + h.brickTreeBase] = layout.brickTreeBaseWords;
  words[base + h.brickListBase] = layout.brickListBaseWords;
  words[base + h.cellStampBase] = layout.cellStampBaseWords;
  words[base + h.cellTreeBase] = layout.cellTreeBaseWords;
  words[base + h.cellListBase] = layout.cellListBaseWords;
  words[base + h.rowStampBase] = layout.rowStampBaseWords;
  words[base + h.rowTreeBase] = layout.rowTreeBaseWords;
  words[base + h.rowListBase] = layout.rowListBaseWords;
  return words;
}

export function createSparseCM12TWA(options: {
  readonly topology: SparseCM12TWATopology;
  readonly constructionMode?: SparseCM12TWAConstructionMode;
}): SparseCM12TWA {
  const t = options.topology;
  integer(t.brickCapacity, "TWA brick capacity");
  integer(t.cellCapacity, "TWA cell capacity"); integer(t.rowCapacity, "TWA row capacity");
  if (Math.min(t.brickCapacity, t.cellCapacity, t.rowCapacity) < 1
    || t.cellRows.length !== t.cellCapacity) throw new RangeError("TWA topology");
  t.cellRows.forEach((rows) => rows.forEach((row) => integer(row, "TWA incidence row",
    t.rowCapacity)));
  return { topology: t, constructionMode: options.constructionMode ?? "temporal",
    phase: SPARSE_CM12_TWA_PHASE.accepted, fault: 0, firstFaultId: 0xffff_ffff,
    acceptedGeneration: 0, candidateGeneration: 0, topologyGeneration: 0,
    changes: new Map(), executed: new Set(), expectedReceipts: 0,
    coveredReceipts: 0, uncoveredWriteCount: 0, firstUncoveredOwner: 0xffff_ffff };
}
function fail(authority: SparseCM12TWA, fault: number, owner: number): false {
  authority.phase = SPARSE_CM12_TWA_PHASE.fault; authority.fault = fault;
  authority.firstFaultId = owner; authority.plan = undefined; return false;
}
export function beginSparseCM12TWA(authority: SparseCM12TWA, input: {
  readonly generation: number; readonly topologyGeneration: number;
  readonly runtime?: boolean;
}): boolean {
  if (authority.phase !== SPARSE_CM12_TWA_PHASE.accepted
    && authority.phase !== SPARSE_CM12_TWA_PHASE.rejected) {
    return fail(authority, SPARSE_CM12_TWA_FAULT.phase, 0xffff_ffff);
  }
  if (input.runtime && authority.constructionMode === "immutable-full-oracle") {
    return fail(authority, SPARSE_CM12_TWA_FAULT.runtimeOracle, 0xffff_ffff);
  }
  if (authority.constructionMode === "immutable-full-oracle"
    && authority.acceptedGeneration !== 0) {
    return fail(authority, SPARSE_CM12_TWA_FAULT.runtimeOracle, 0xffff_ffff);
  }
  const generation = integer(input.generation, "TWA generation");
  const topologyGeneration = integer(input.topologyGeneration, "TWA topology generation");
  if (generation < 1 || topologyGeneration < 1
    || (authority.acceptedGeneration !== 0
      && generation !== authority.acceptedGeneration + 1)) {
    return fail(authority, SPARSE_CM12_TWA_FAULT.generation, 0xffff_ffff);
  }
  authority.phase = SPARSE_CM12_TWA_PHASE.collecting; authority.fault = 0;
  authority.firstFaultId = 0xffff_ffff; authority.candidateGeneration = generation;
  authority.topologyGeneration = topologyGeneration; authority.changes.clear();
  authority.executed.clear(); authority.expectedReceipts = 0; authority.coveredReceipts = 0;
  authority.uncoveredWriteCount = 0; authority.firstUncoveredOwner = 0xffff_ffff;
  authority.plan = undefined;
  if (authority.constructionMode === "immutable-full-oracle") {
    for (let brick = 0; brick < authority.topology.brickCapacity; brick += 1) {
      authority.changes.set(brick, { brick, causeMask: SPARSE_CM12_TWA_CAUSE.bootstrap,
        oldCellFirst: 0, oldCellCount: authority.topology.cellCapacity,
        newCellFirst: 0, newCellCount: authority.topology.cellCapacity,
        oldRowFirst: 0, oldRowCount: authority.topology.rowCapacity,
        newRowFirst: 0, newRowCount: authority.topology.rowCapacity });
    }
  }
  return true;
}
export function appendSparseCM12TWAChange(authority: SparseCM12TWA,
  change: SparseCM12TWAChange): boolean {
  if (authority.phase !== SPARSE_CM12_TWA_PHASE.collecting) return false;
  integer(change.brick, "TWA changed brick", authority.topology.brickCapacity);
  range(change.oldCellFirst, change.oldCellCount, authority.topology.cellCapacity);
  range(change.newCellFirst, change.newCellCount, authority.topology.cellCapacity);
  range(change.oldRowFirst, change.oldRowCount, authority.topology.rowCapacity);
  range(change.newRowFirst, change.newRowCount, authority.topology.rowCapacity);
  const old = authority.changes.get(change.brick);
  if (old) {
    authority.changes.set(change.brick, Object.freeze({ ...change,
      causeMask: old.causeMask | change.causeMask,
      oldCellFirst: Math.min(old.oldCellFirst, change.oldCellFirst),
      oldCellCount: Math.max(old.oldCellFirst + old.oldCellCount,
        change.oldCellFirst + change.oldCellCount)
        - Math.min(old.oldCellFirst, change.oldCellFirst),
      oldRowFirst: Math.min(old.oldRowFirst, change.oldRowFirst),
      oldRowCount: Math.max(old.oldRowFirst + old.oldRowCount,
        change.oldRowFirst + change.oldRowCount)
        - Math.min(old.oldRowFirst, change.oldRowFirst),
      newCellFirst: Math.min(old.newCellFirst, change.newCellFirst),
      newCellCount: Math.max(old.newCellFirst + old.newCellCount,
        change.newCellFirst + change.newCellCount)
        - Math.min(old.newCellFirst, change.newCellFirst),
      newRowFirst: Math.min(old.newRowFirst, change.newRowFirst),
      newRowCount: Math.max(old.newRowFirst + old.newRowCount,
        change.newRowFirst + change.newRowCount)
        - Math.min(old.newRowFirst, change.newRowFirst),
    }));
  } else authority.changes.set(change.brick, Object.freeze({ ...change }));
  return true;
}

export function sealSparseCM12TWA(authority: SparseCM12TWA): SparseCM12TWAPlan | undefined {
  if (authority.phase !== SPARSE_CM12_TWA_PHASE.collecting) return undefined;
  const topology = authority.topology;
  const changes = [...authority.changes.values()].sort((a, b) => a.brick - b.brick);
  const bricks = stable(changes.map(({ brick }) => brick));
  const directCells = stable(changes.flatMap((change) => [
    ...range(change.oldCellFirst, change.oldCellCount, topology.cellCapacity),
    ...range(change.newCellFirst, change.newCellCount, topology.cellCapacity),
  ]));
  const explicitRows = stable(changes.flatMap((change) => [
    ...range(change.oldRowFirst, change.oldRowCount, topology.rowCapacity),
    ...range(change.newRowFirst, change.newRowCount, topology.rowCapacity),
  ]));
  const incidenceRows = stable(directCells.flatMap((cell) => topology.cellRows[cell]!));
  const rows = stable([...explicitRows, ...incidenceRows]);
  const faces = stable(bricks.flatMap((brick) =>
    Array.from({ length: 6 }, (_, face) => 6 * brick + face)));
  const family = <T extends readonly number[]>(values: T): T => values;
  const families = Object.freeze({
    allocatePagesBricks: family(bricks), synthesizeCellPagesBricks: family(bricks),
    buildShadowCells: family(directCells), buildShadowRows: family(rows),
    transferCellsBricks: family(bricks), prepareFaceReceiptsBricks: family(bricks),
    transferFaces: family(faces), writeShadowCellsBricks: family(bricks),
    reconstructShadowFacesRows: family(rows), validateChangedBricks: family(bricks),
    postTopologyBricks: family(bricks),
  });
  const indirect = Object.freeze(Object.fromEntries(Object.entries(families).map(
    ([name, packets]) => [name, Object.freeze([
      Math.ceil((packets as readonly number[]).length / 64), 1, 1,
    ])])) as unknown as Readonly<Record<SparseCM12TWAFamily,
      readonly [number, number, number]>>);
  authority.expectedReceipts = Object.values(families)
    .filter((packets) => packets.length > 0).length;
  const causeMask = changes.reduce((mask, change) => mask | change.causeMask, 0)
    | (incidenceRows.length > 0 ? SPARSE_CM12_TWA_CAUSE.incidenceClosure : 0);
  authority.plan = Object.freeze({ generation: authority.candidateGeneration,
    topologyGeneration: authority.topologyGeneration, causeMask,
    maximumClosureDepth: incidenceRows.length > 0 ? 1 : 0,
    changedBricks: bricks, closureCells: directCells, closureRows: rows,
    families, indirect });
  authority.phase = SPARSE_CM12_TWA_PHASE.sealed; return authority.plan;
}
export function publishSparseCM12TWAFamilyExecution(authority: SparseCM12TWA,
  family: SparseCM12TWAFamily, executedCount?: number): boolean {
  if (authority.phase !== SPARSE_CM12_TWA_PHASE.sealed || !authority.plan) return false;
  if (!(family in SPARSE_CM12_TWA_FAMILY) || authority.executed.has(family)) {
    return fail(authority, SPARSE_CM12_TWA_FAULT.missingExecution,
      SPARSE_CM12_TWA_FAMILY[family]);
  }
  if ((executedCount ?? authority.plan.families[family].length)
    !== authority.plan.families[family].length) {
    return fail(authority, SPARSE_CM12_TWA_FAULT.missingExecution,
      SPARSE_CM12_TWA_FAMILY[family]);
  }
  authority.executed.add(family); if (authority.plan.families[family].length > 0) {
    authority.coveredReceipts += 1;
  } return true;
}
export function recordSparseCM12TWAUncoveredWrite(authority: SparseCM12TWA,
  owner: number): void {
  authority.uncoveredWriteCount += 1;
  if (authority.firstUncoveredOwner === 0xffff_ffff) authority.firstUncoveredOwner = owner;
}
export function commitSparseCM12TWA(authority: SparseCM12TWA): boolean {
  if (authority.phase !== SPARSE_CM12_TWA_PHASE.sealed || !authority.plan) return false;
  for (const family of Object.keys(SPARSE_CM12_TWA_FAMILY) as SparseCM12TWAFamily[]) {
    if (authority.plan.families[family].length > 0 && !authority.executed.has(family)) {
      return fail(authority, SPARSE_CM12_TWA_FAULT.missingExecution,
        SPARSE_CM12_TWA_FAMILY[family]);
    }
  }
  if (authority.expectedReceipts !== authority.coveredReceipts) {
    return fail(authority, SPARSE_CM12_TWA_FAULT.missingExecution, 0xffff_ffff);
  }
  if (authority.uncoveredWriteCount !== 0) {
    return fail(authority, SPARSE_CM12_TWA_FAULT.uncoveredWrite,
      authority.firstUncoveredOwner);
  }
  authority.acceptedGeneration = authority.candidateGeneration;
  authority.phase = SPARSE_CM12_TWA_PHASE.accepted; return true;
}

export const SPARSE_CM12_TWA_LIVE_GLOBAL_TOKENS = Object.freeze([
  "planBrickResolution", "activateSweptReceivers", "closePlannedResolution",
  "validateCandidateResolution", "allocateCandidateTopologyPages",
  "synthesizeCandidateCellPages", "buildShadowCellWorklist", "buildShadowRowWorklist",
  "prepareCandidateFaceReceipts",
  "transferCandidateCells", "transferCandidateFaces", "writeCandidateCellsToShadow",
  "reconstructShadowFaces", "validateAndCommitShadowTopology",
  "publishSparseCM12TopologyVelocityRoots", "retireUnsupportedEmptyBricks",
  "markIncrementalActivityPostTopology",
] as const);

/** Audited legacy domains that TWA1 replaces; these are gap tokens, not an adapter. */
export const SPARSE_CM12_TWA_GLOBAL_PASS_AUDIT = Object.freeze([
  { pass: "planBrickResolution", domain: "all brick workgroups" },
  { pass: "activateSweptReceivers", domain: "all brick workgroups" },
  { pass: "closePlannedResolution", domain: "all brick workgroups x B16 ladder depth (4)" },
  { pass: "validateCandidateResolution", domain: "all brick workgroups" },
  { pass: "allocateCandidateTopologyPages", domain: "all brick workgroups" },
  { pass: "synthesizeCandidateCellPages", domain: "packed.brickCount" },
  { pass: "buildShadowCellWorklist", domain: "templateCellCount" },
  { pass: "buildShadowRowWorklist", domain: "templateRowCount" },
  { pass: "transferCandidateCells", domain: "packed.brickCount" },
  { pass: "prepareCandidateFaceReceipts", domain: "all brick workgroups" },
  { pass: "transferCandidateFaces", domain: "packed.brickCount x 6" },
  { pass: "writeCandidateCellsToShadow", domain: "packed.brickCount" },
  { pass: "reconstructShadowFaces", domain: "templateRowCount" },
  { pass: "validateAndCommitShadowTopology", domain: "singleton dispatch, internal all-brick validation" },
  { pass: "publishSparseCM12TopologyVelocityRoots", domain: "packed.brickCount" },
  { pass: "retireUnsupportedEmptyBricks", domain: "all brick workgroups" },
  { pass: "markIncrementalActivityPostTopology", domain: "packed.brickCount" },
] as const);
