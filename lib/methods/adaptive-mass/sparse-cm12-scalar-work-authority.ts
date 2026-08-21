/** Exact, non-physical scalar work authority for matched Sparse CM12 profiles. */
import { SPARSE_CM12_FRAME_PLAN_STAGE, SPARSE_CM12_FRAME_PLAN_STAGE_COUNT } from
  "../../core/sparse-cm12-frame-plan";

export const SPARSE_CM12_SCALAR_AUTHORITY_MAGIC = 0x5341_5731; // SAW1
export const SPARSE_CM12_SCALAR_AUTHORITY_VERSION = 1;
export const SPARSE_CM12_SCALAR_AUTHORITY_HEADER_WORDS = 32;
export const SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER_WORDS = 32;
export const SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT = 3;
export const SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT = 8;
export const SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_WORDS = 2;
export const SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT_WORDS = 4;
export const SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT_WORDS = 4;
export const SPARSE_CM12_SCALAR_AUTHORITY_TILE_WORDS = 4;
export const SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES = 256;
export const SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH = 32;
export const SPARSE_CM12_SCALAR_AUTHORITY_INVALID = 0xffff_ffff;

export const SPARSE_CM12_SCALAR_STAGE = Object.freeze({
  massTransport: 0,
  gammaTransport: 1,
  surfaceConditioning: 2,
} as const);

export const SPARSE_CM12_SCALAR_FPL_STAGE = Object.freeze([
  SPARSE_CM12_FRAME_PLAN_STAGE.massTransport,
  SPARSE_CM12_FRAME_PLAN_STAGE.gammaTransport,
  SPARSE_CM12_FRAME_PLAN_STAGE.surfaceConditioning,
] as const);

export const SPARSE_CM12_SCALAR_DEPENDENCY = Object.freeze({
  densitySource: 0,
  gammaSource: 1,
  faceVelocity: 2,
  topology: 3,
  solid: 4,
  characteristic: 5,
  incidence: 6,
  scatterTarget: 7,
} as const);

const dependencyBit = (dependency: number): number => 1 << dependency;
export const SPARSE_CM12_SCALAR_REQUIRED_DEPENDENCIES = Object.freeze([
  dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.densitySource)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.faceVelocity)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.topology)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.solid)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.characteristic),
  dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.densitySource)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.gammaSource)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.faceVelocity)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.topology)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.solid)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.incidence),
  dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.densitySource)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.gammaSource)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.topology)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.solid)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.incidence)
    | dependencyBit(SPARSE_CM12_SCALAR_DEPENDENCY.scatterTarget),
] as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_PHASE = Object.freeze({
  accepted: 1, collecting: 2, classified: 3, sealed: 4, fault: 5,
} as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_FAULT = Object.freeze({
  none: 0, invalidHeader: 1, invalidPhase: 2, generation: 3,
  invalidTile: 4, invalidStage: 5, treeMismatch: 6,
  missingExecution: 7, generationExhausted: 8,
} as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_CAUSE = Object.freeze({
  bank0NotExact: 1 << 0, bank1NotExact: 1 << 1,
  dependency: 1 << 2, fplReceipt: 1 << 3,
  topology: 1 << 4, generation: 1 << 5,
} as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG = Object.freeze({
  classified: 1 << 0, exactCleanSkip: 1 << 1, work: 1 << 2,
  executed: 1 << 3, fplReceipt: 1 << 4,
} as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, stageHeaderWords: 3,
  stageCount: 4, dependencyCount: 5, brickFineResolution: 6,
  presentationPageResolution: 7, tileCapacity: 8, phase: 9,
  acceptedGeneration: 10, candidateGeneration: 11,
  topologyGeneration: 12, sourceParity: 13, fault: 14, firstFaultTile: 15,
  firstFaultStage: 16, stageHeadersBase: 17, dependencyBase: 18,
  bankReceiptBase: 19, fplReceiptBase: 20, tileBase: 21,
  treeBase: 22, workListBase: 23, closureOffsetBase: 24,
  closureIdBase: 25, totalWords: 26, leafTiles: 27, treeBranch: 28,
  flags: 29, reserved0: 30, reserved1: 31,
} as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER = Object.freeze({
  phase: 0, frameGeneration: 1, topologyGeneration: 2, sourceParity: 3,
  requiredDependencyMask: 4, headResultGeneration: 5, headWordsPerTile: 6,
  fplGeneration: 7, fplPacketEpoch: 8, workCount: 9, cleanCount: 10,
  classifiedCount: 11, receiptCount: 12, fault: 13, firstFaultTile: 14,
  workIndirectX: 15, workIndirectY: 16, workIndirectZ: 17,
  cleanIndirectX: 18, cleanIndirectY: 19, cleanIndirectZ: 20,
  treeRootCount: 21, reservedBase: 22, // ten words
  reserved0: 30, reserved1: 31,
} as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT = Object.freeze({
  resultGeneration: 0, topologyGeneration: 1, coveredWords: 2, mismatchCount: 3,
} as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT = Object.freeze({
  frameGeneration: 0, topologyGeneration: 1, packetEpoch: 2, packed: 3,
} as const);

export const SPARSE_CM12_SCALAR_AUTHORITY_TILE = Object.freeze({
  generation: 0, flags: 1, causeMask: 2, fplPacket: 3,
} as const);

export interface SparseCM12ScalarStageRequest {
  readonly headResultGeneration: number;
  readonly headWordsPerTile: number;
  readonly fplGeneration: number;
  readonly fplPacketEpoch: number;
}

export interface SparseCM12ScalarWorkAuthorityLayout {
  readonly brickFineResolution: 4 | 8 | 16;
  readonly presentationPageResolution: 4 | 8 | 16;
  readonly tileCapacity: number;
  readonly stageHeadersBaseWords: number;
  readonly dependencyBaseWords: number;
  readonly bankReceiptBaseWords: number;
  readonly fplReceiptBaseWords: number;
  readonly tileBaseWords: number;
  readonly stageTreeBaseWords: readonly number[];
  readonly treeLevelBaseWords: readonly (readonly number[])[];
  readonly treeLevelCounts: readonly number[];
  readonly workListBaseWords: number;
  readonly closureOffsetBaseWords: number;
  readonly closureIdBaseWords: number;
  readonly closureOffsets: readonly (readonly number[])[];
  readonly closureIds: readonly number[];
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12ScalarWorkAuthority {
  readonly layout: SparseCM12ScalarWorkAuthorityLayout;
  readonly words: Uint32Array;
}

/** Algebraic constant-field certificate for the conservative mass operator.
 * Velocity is intentionally absent: a fully supported constant capacity field
 * is an exact fixed point for every characteristic admitted by the stencil. */
export function sparseCM12FullyFloodedCertificate(input: {
  readonly sourceDensityBits: number;
  readonly destinationDensityBits: number;
  readonly sourceGammaBits: number;
  readonly destinationGammaBits: number;
  readonly openFractionBits: number;
  readonly sixSided: boolean;
  readonly fullSupport: boolean;
}): boolean {
  const one = 0x3f80_0000;
  return input.sourceDensityBits === one && input.destinationDensityBits === one
    && input.sourceGammaBits === one && input.destinationGammaBits === one
    && input.openFractionBits === one && input.sixSided && input.fullSupport;
}

/** Exact dry counterpart. Missing-side sparse-band cells intentionally cannot
 * mint this receipt because sixSided/fullSupport must both be true. */
export function sparseCM12FullyDryCertificate(input: {
  readonly sourceDensityBits: number;
  readonly destinationDensityBits: number;
  readonly sourceGammaBits: number;
  readonly destinationGammaBits: number;
  readonly openFractionBits: number;
  readonly sixSided: boolean;
  readonly fullSupport: boolean;
}): boolean {
  const one = 0x3f80_0000;
  return (input.sourceDensityBits >>> 0) === 0
    && (input.destinationDensityBits >>> 0) === 0
    && input.sourceGammaBits === one && input.destinationGammaBits === one
    && input.openFractionBits === one && input.sixSided && input.fullSupport;
}

/** Accepted-topology projection of the immutable variant-row superset. */
export function sparseCM12AcceptedConstantSupport(rows: readonly {
  readonly accepted: boolean;
  readonly side: 0 | 1 | 2 | 3 | 4 | 5;
  readonly fullOpen: boolean;
  readonly samePhaseNeighbor: boolean;
}[]): boolean {
  let sideMask = 0;
  for (const row of rows) {
    if (!row.accepted) continue;
    if (!row.fullOpen || !row.samePhaseNeighbor) return false;
    sideMask |= 1 << row.side;
  }
  return sideMask === 0x3f;
}

const integer = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0) || value > 0x7fff_fffe) {
    throw new RangeError(`${label} must be ${positive ? "a positive" : "a non-negative"} u31`);
  }
  return value;
};
const alignWords = (value: number): number => Math.ceil(value / 64) * 64;

function defaultClosure(tileCapacity: number): readonly (readonly number[])[] {
  return Object.freeze(Array.from({ length: SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT },
    () => Object.freeze(Array.from({ length: tileCapacity }, (_, tile) => tile))));
}

function compileClosure(
  tileCapacity: number,
  closureInput?: readonly (readonly (readonly number[])[])[],
): { readonly offsets: readonly (readonly number[])[]; readonly ids: readonly number[] } {
  const input = closureInput ?? defaultClosure(tileCapacity).map((ids) =>
    Array.from({ length: tileCapacity }, (_, tile) => [ids[tile]!]));
  if (input.length !== SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT) {
    throw new RangeError("SAW1 requires one closure graph per scalar stage");
  }
  const offsets: number[][] = [];
  const ids: number[] = [];
  input.forEach((stageClosure, stage) => {
    if (stageClosure.length !== tileCapacity) {
      throw new RangeError(`scalar closure stage ${stage} must cover every tile`);
    }
    const stageOffsets = [ids.length];
    stageClosure.forEach((neighbors, tile) => {
      const canonical = [...new Set(neighbors)].sort((a, b) => a - b);
      if (!canonical.includes(tile)) canonical.push(tile);
      canonical.sort((a, b) => a - b);
      for (const neighbor of canonical) {
        integer(neighbor, "closure tile");
        if (neighbor >= tileCapacity) throw new RangeError("closure tile exceeds capacity");
        ids.push(neighbor);
      }
      stageOffsets.push(ids.length);
    });
    offsets.push(stageOffsets);
  });
  return { offsets: Object.freeze(offsets.map((value) => Object.freeze(value))),
    ids: Object.freeze(ids) };
}

export function createSparseCM12ScalarWorkAuthority(options: {
  readonly tileCapacity: number;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly presentationPageResolution?: 4 | 8 | 16;
  readonly stageClosure?: readonly (readonly (readonly number[])[])[];
}): SparseCM12ScalarWorkAuthority {
  const tileCapacity = integer(options.tileCapacity, "tileCapacity", true);
  const brickFineResolution = options.brickFineResolution ?? 8;
  const presentationPageResolution = options.presentationPageResolution ?? brickFineResolution;
  if ((brickFineResolution !== 4 && brickFineResolution !== 8 && brickFineResolution !== 16)
    || presentationPageResolution !== brickFineResolution) {
    throw new Error("SAW1 requires a matched B4/P4, B8/P8, or B16/P16 physical ABI");
  }
  const closure = compileClosure(tileCapacity, options.stageClosure);
  const stageHeadersBaseWords = SPARSE_CM12_SCALAR_AUTHORITY_HEADER_WORDS;
  const dependencyBaseWords = alignWords(stageHeadersBaseWords
    + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT
      * SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER_WORDS);
  const bankReceiptBaseWords = alignWords(dependencyBaseWords
    + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT * tileCapacity
    * SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT
    * SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_WORDS);
  const fplReceiptBaseWords = alignWords(bankReceiptBaseWords
    + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT * tileCapacity * 2
      * SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT_WORDS);
  const tileBaseWords = alignWords(fplReceiptBaseWords
    + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT * tileCapacity
      * SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT_WORDS);
  let at = alignWords(tileBaseWords + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT
    * tileCapacity * SPARSE_CM12_SCALAR_AUTHORITY_TILE_WORDS);
  const treeLevelCounts: number[] = [];
  for (let count = Math.ceil(tileCapacity / SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES);;) {
    treeLevelCounts.push(count);
    if (count === 1) break;
    count = Math.ceil(count / SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH);
  }
  const stageTreeBaseWords: number[] = [];
  const treeLevelBaseWords: number[][] = [];
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    stageTreeBaseWords.push(at);
    const levels: number[] = [];
    for (const count of treeLevelCounts) { levels.push(at); at = alignWords(at + count); }
    treeLevelBaseWords.push(levels);
  }
  const workListBaseWords = at;
  at = alignWords(at + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT * tileCapacity);
  const closureOffsetBaseWords = at;
  at = alignWords(at + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT * (tileCapacity + 1));
  const closureIdBaseWords = at;
  const totalWords = alignWords(at + closure.ids.length);
  const layout: SparseCM12ScalarWorkAuthorityLayout = Object.freeze({
    brickFineResolution, presentationPageResolution, tileCapacity,
    stageHeadersBaseWords, dependencyBaseWords, bankReceiptBaseWords,
    fplReceiptBaseWords, tileBaseWords,
    stageTreeBaseWords: Object.freeze(stageTreeBaseWords),
    treeLevelBaseWords: Object.freeze(treeLevelBaseWords.map((value) => Object.freeze(value))),
    treeLevelCounts: Object.freeze(treeLevelCounts), workListBaseWords,
    closureOffsetBaseWords, closureIdBaseWords, closureOffsets: closure.offsets,
    closureIds: closure.ids, totalWords, totalBytes: 4 * totalWords,
  });
  const words = new Uint32Array(totalWords);
  const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  words[h.magic] = SPARSE_CM12_SCALAR_AUTHORITY_MAGIC;
  words[h.version] = SPARSE_CM12_SCALAR_AUTHORITY_VERSION;
  words[h.headerWords] = SPARSE_CM12_SCALAR_AUTHORITY_HEADER_WORDS;
  words[h.stageHeaderWords] = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER_WORDS;
  words[h.stageCount] = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT;
  words[h.dependencyCount] = SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT;
  words[h.brickFineResolution] = brickFineResolution;
  words[h.presentationPageResolution] = presentationPageResolution;
  words[h.tileCapacity] = tileCapacity;
  words[h.phase] = SPARSE_CM12_SCALAR_AUTHORITY_PHASE.accepted;
  words[h.firstFaultTile] = SPARSE_CM12_SCALAR_AUTHORITY_INVALID;
  words[h.firstFaultStage] = SPARSE_CM12_SCALAR_AUTHORITY_INVALID;
  words[h.stageHeadersBase] = stageHeadersBaseWords;
  words[h.dependencyBase] = dependencyBaseWords; words[h.bankReceiptBase] = bankReceiptBaseWords;
  words[h.fplReceiptBase] = fplReceiptBaseWords; words[h.tileBase] = tileBaseWords;
  words[h.treeBase] = stageTreeBaseWords[0]!; words[h.workListBase] = workListBaseWords;
  words[h.closureOffsetBase] = closureOffsetBaseWords;
  words[h.closureIdBase] = closureIdBaseWords; words[h.totalWords] = totalWords;
  words[h.leafTiles] = SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES;
  words[h.treeBranch] = SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH;
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    const base = scalarStageHeaderWord(layout, stage);
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.phase]
      = SPARSE_CM12_SCALAR_AUTHORITY_PHASE.accepted;
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.requiredDependencyMask]
      = SPARSE_CM12_SCALAR_REQUIRED_DEPENDENCIES[stage]!;
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.firstFaultTile]
      = SPARSE_CM12_SCALAR_AUTHORITY_INVALID;
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.workIndirectY] = 1;
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.workIndirectZ] = 1;
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.cleanIndirectY] = 1;
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.cleanIndirectZ] = 1;
    const offsets = closure.offsets[stage]!;
    for (let tile = 0; tile <= tileCapacity; tile += 1) {
      words[closureOffsetBaseWords + stage * (tileCapacity + 1) + tile] = offsets[tile]!;
    }
  }
  words.set(closure.ids, closureIdBaseWords);
  return { layout, words };
}

export function scalarStageHeaderWord(layout: SparseCM12ScalarWorkAuthorityLayout, stage: number): number {
  if (!Number.isInteger(stage) || stage < 0 || stage >= SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT) {
    throw new RangeError("invalid scalar stage");
  }
  return layout.stageHeadersBaseWords + stage * SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER_WORDS;
}

export function scalarDependencyWord(
  layout: SparseCM12ScalarWorkAuthorityLayout, stage: number, tile: number, dependency: number,
): number {
  scalarStageHeaderWord(layout, stage);
  if (tile < 0 || tile >= layout.tileCapacity) throw new RangeError("invalid scalar tile");
  if (dependency < 0 || dependency >= SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT) {
    throw new RangeError("invalid scalar dependency");
  }
  return layout.dependencyBaseWords + SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_WORDS
    * ((stage * layout.tileCapacity + tile)
      * SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT + dependency);
}

export function scalarBankReceiptWord(
  layout: SparseCM12ScalarWorkAuthorityLayout, stage: number, tile: number, bank: number,
): number {
  scalarStageHeaderWord(layout, stage);
  if (tile < 0 || tile >= layout.tileCapacity || (bank !== 0 && bank !== 1)) {
    throw new RangeError("invalid scalar bank receipt address");
  }
  return layout.bankReceiptBaseWords + SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT_WORDS
    * ((stage * layout.tileCapacity + tile) * 2 + bank);
}

export function scalarFPLReceiptWord(
  layout: SparseCM12ScalarWorkAuthorityLayout, stage: number, tile: number,
): number {
  scalarStageHeaderWord(layout, stage);
  if (tile < 0 || tile >= layout.tileCapacity) throw new RangeError("invalid scalar tile");
  return layout.fplReceiptBaseWords + SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT_WORDS
    * (stage * layout.tileCapacity + tile);
}

export function scalarTileWord(
  layout: SparseCM12ScalarWorkAuthorityLayout, stage: number, tile: number,
): number {
  scalarStageHeaderWord(layout, stage);
  if (tile < 0 || tile >= layout.tileCapacity) throw new RangeError("invalid scalar tile");
  return layout.tileBaseWords + SPARSE_CM12_SCALAR_AUTHORITY_TILE_WORDS
    * (stage * layout.tileCapacity + tile);
}

export function sparseCM12ScalarAuthorityHeaderValid(authority: SparseCM12ScalarWorkAuthority): boolean {
  const { layout: l, words } = authority; const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  return words.length === l.totalWords && l.totalBytes === 4 * l.totalWords
    && (l.brickFineResolution === 4 || l.brickFineResolution === 8
      || l.brickFineResolution === 16)
    && l.presentationPageResolution === l.brickFineResolution
    && words[h.magic] === SPARSE_CM12_SCALAR_AUTHORITY_MAGIC
    && words[h.version] === SPARSE_CM12_SCALAR_AUTHORITY_VERSION
    && words[h.headerWords] === SPARSE_CM12_SCALAR_AUTHORITY_HEADER_WORDS
    && words[h.stageHeaderWords] === SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER_WORDS
    && words[h.stageCount] === SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT
    && words[h.dependencyCount] === SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT
    && words[h.brickFineResolution] === l.brickFineResolution
    && words[h.presentationPageResolution] === l.presentationPageResolution
    && words[h.tileCapacity] === l.tileCapacity
    && words[h.dependencyBase] === l.dependencyBaseWords
    && words[h.bankReceiptBase] === l.bankReceiptBaseWords
    && words[h.fplReceiptBase] === l.fplReceiptBaseWords
    && words[h.tileBase] === l.tileBaseWords && words[h.workListBase] === l.workListBaseWords
    && words[h.closureOffsetBase] === l.closureOffsetBaseWords
    && words[h.closureIdBase] === l.closureIdBaseWords && words[h.totalWords] === l.totalWords;
}

function fail(authority: SparseCM12ScalarWorkAuthority, code: number, stage: number, tile: number): false {
  const { words, layout } = authority; const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  if (words[h.fault] === SPARSE_CM12_SCALAR_AUTHORITY_FAULT.none) {
    words[h.fault] = code; words[h.firstFaultStage] = stage >>> 0; words[h.firstFaultTile] = tile >>> 0;
  }
  words[h.phase] = SPARSE_CM12_SCALAR_AUTHORITY_PHASE.fault;
  for (let s = 0; s < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; s += 1) {
    const base = scalarStageHeaderWord(layout, s);
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.workIndirectX] = 0;
    words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.cleanIndirectX] = 0;
  }
  return false;
}

export function beginSparseCM12ScalarAuthority(
  authority: SparseCM12ScalarWorkAuthority,
  request: { readonly frameGeneration: number; readonly topologyGeneration: number;
    readonly sourceParity: 0 | 1; readonly stages: readonly SparseCM12ScalarStageRequest[] },
): boolean {
  const { words, layout } = authority; const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  if (!sparseCM12ScalarAuthorityHeaderValid(authority)) return fail(authority,
    SPARSE_CM12_SCALAR_AUTHORITY_FAULT.invalidHeader, SPARSE_CM12_SCALAR_AUTHORITY_INVALID,
    SPARSE_CM12_SCALAR_AUTHORITY_INVALID);
  if (words[h.phase] !== SPARSE_CM12_SCALAR_AUTHORITY_PHASE.accepted) return fail(authority,
    SPARSE_CM12_SCALAR_AUTHORITY_FAULT.invalidPhase, SPARSE_CM12_SCALAR_AUTHORITY_INVALID,
    SPARSE_CM12_SCALAR_AUTHORITY_INVALID);
  const frameGeneration = integer(request.frameGeneration, "frameGeneration", true);
  const topologyGeneration = integer(request.topologyGeneration, "topologyGeneration", true);
  if (request.stages.length !== SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT) {
    throw new RangeError("SAW1 begin requires all scalar stages");
  }
  words[h.phase] = SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting;
  words[h.candidateGeneration] = frameGeneration; words[h.topologyGeneration] = topologyGeneration;
  words[h.sourceParity] = request.sourceParity; words[h.fault] = 0;
  words[h.firstFaultTile] = SPARSE_CM12_SCALAR_AUTHORITY_INVALID;
  words[h.firstFaultStage] = SPARSE_CM12_SCALAR_AUTHORITY_INVALID;
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    const config = request.stages[stage]!;
    if (config.fplGeneration !== frameGeneration) {
      throw new RangeError("FPL receipt generation must equal the candidate frame generation");
    }
    const base = scalarStageHeaderWord(layout, stage); const sh = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER;
    words[base + sh.phase] = SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting;
    words[base + sh.frameGeneration] = frameGeneration;
    words[base + sh.topologyGeneration] = topologyGeneration;
    words[base + sh.sourceParity] = request.sourceParity;
    words[base + sh.headResultGeneration] = integer(config.headResultGeneration, "head result", true);
    words[base + sh.headWordsPerTile] = integer(config.headWordsPerTile, "head words", true);
    words[base + sh.fplGeneration] = integer(config.fplGeneration, "FPL generation", true);
    words[base + sh.fplPacketEpoch] = integer(config.fplPacketEpoch, "FPL packet epoch", true);
    words[base + sh.workCount] = 0; words[base + sh.cleanCount] = 0;
    words[base + sh.classifiedCount] = 0; words[base + sh.receiptCount] = 0;
    words[base + sh.fault] = 0; words[base + sh.firstFaultTile] = SPARSE_CM12_SCALAR_AUTHORITY_INVALID;
    words[base + sh.workIndirectX] = 0; words[base + sh.workIndirectY] = 1;
    words[base + sh.workIndirectZ] = 1; words[base + sh.cleanIndirectX] = 0;
    words[base + sh.cleanIndirectY] = 1; words[base + sh.cleanIndirectZ] = 1;
    words[base + sh.treeRootCount] = 0;
  }
  words.fill(0, layout.tileBaseWords, layout.treeLevelBaseWords[0]![0]);
  words.fill(SPARSE_CM12_SCALAR_AUTHORITY_INVALID, layout.workListBaseWords,
    layout.workListBaseWords + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT * layout.tileCapacity);
  return true;
}

export function publishSparseCM12ScalarDependency(
  authority: SparseCM12ScalarWorkAuthority, stage: number, tile: number, dependency: number,
  producerGeneration: number, headCertifiedGeneration: number,
): void {
  const at = scalarDependencyWord(authority.layout, stage, tile, dependency);
  authority.words[at] = integer(producerGeneration, "producer generation", true);
  authority.words[at + 1] = integer(headCertifiedGeneration,
    "HEAD-certified dependency generation", true);
}

export function publishSparseCM12ScalarBankReceipt(
  authority: SparseCM12ScalarWorkAuthority, stage: number, tile: number, bank: 0 | 1,
  receipt: { readonly resultGeneration: number; readonly topologyGeneration: number;
    readonly coveredWords: number; readonly mismatchCount: number },
): void {
  const at = scalarBankReceiptWord(authority.layout, stage, tile, bank);
  authority.words.set([receipt.resultGeneration, receipt.topologyGeneration,
    receipt.coveredWords, receipt.mismatchCount], at);
}

export function publishSparseCM12ScalarFPLReceipt(
  authority: SparseCM12ScalarWorkAuthority, stage: number, tile: number,
  receipt: { readonly frameGeneration: number; readonly topologyGeneration: number;
    readonly packetEpoch: number; readonly packet: number; readonly coverageComplete: boolean },
): void {
  if (receipt.packet < 0 || receipt.packet >= SPARSE_CM12_FRAME_PLAN_STAGE_COUNT) {
    throw new RangeError("invalid FPL physical packet");
  }
  const at = scalarFPLReceiptWord(authority.layout, stage, tile);
  const packed = (receipt.packet & 0xffff) | (SPARSE_CM12_SCALAR_FPL_STAGE[stage]! << 16)
    | (receipt.coverageComplete ? 1 << 24 : 0);
  authority.words.set([receipt.frameGeneration, receipt.topologyGeneration,
    receipt.packetEpoch, packed >>> 0], at);
}

function exactBank(authority: SparseCM12ScalarWorkAuthority, stage: number, tile: number, bank: 0 | 1): boolean {
  const { words, layout } = authority; const base = scalarStageHeaderWord(layout, stage);
  const at = scalarBankReceiptWord(layout, stage, tile, bank);
  const sh = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER;
  return words[at + SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT.resultGeneration]
      === words[base + sh.headResultGeneration]
    && words[at + SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT.topologyGeneration]
      === words[scalarDependencyWord(layout, stage, tile,
        SPARSE_CM12_SCALAR_DEPENDENCY.topology)]
    && words[at + SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT.coveredWords]
      === words[base + sh.headWordsPerTile]
    && words[at + SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT.mismatchCount] === 0;
}

function fplValid(authority: SparseCM12ScalarWorkAuthority, stage: number, tile: number): boolean {
  const { words, layout } = authority; const base = scalarStageHeaderWord(layout, stage);
  const at = scalarFPLReceiptWord(layout, stage, tile); const sh = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER;
  const packed = words[at + SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT.packed]!;
  return words[at] === words[base + sh.fplGeneration]
    && words[at + 1] === words[base + sh.topologyGeneration]
    && words[at + 2] === words[base + sh.fplPacketEpoch]
    && ((packed >>> 16) & 0xff) === SPARSE_CM12_SCALAR_FPL_STAGE[stage]
    && (packed & (1 << 24)) !== 0;
}

function dependenciesValid(authority: SparseCM12ScalarWorkAuthority, stage: number, tile: number): boolean {
  const { words, layout } = authority; const sh = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER;
  const base = scalarStageHeaderWord(layout, stage);
  const required = words[base + sh.requiredDependencyMask]!;
  const offsets = layout.closureOffsets[stage]!;
  for (let at = offsets[tile]!; at < offsets[tile + 1]!; at += 1) {
    const dependencyTile = layout.closureIds[at]!;
    for (let dependency = 0; dependency < SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT;
      dependency += 1) {
      if ((required & (1 << dependency)) === 0) continue;
      const stamp = scalarDependencyWord(layout, stage, dependencyTile, dependency);
      // The producer generation is current physical input authority. The
      // second word is written only by the exact HEAD comparator/producer.
      // Equality is an exact receipt; no numeric tolerance or value snapping
      // exists in SAW1.
      if (words[stamp] === 0 || words[stamp] !== words[stamp + 1]) return false;
    }
  }
  return true;
}

export function sparseCM12ScalarTileCanSkipExactly(
  authority: SparseCM12ScalarWorkAuthority, stage: number, tile: number,
): boolean {
  return exactBank(authority, stage, tile, 0) && exactBank(authority, stage, tile, 1)
    && dependenciesValid(authority, stage, tile) && fplValid(authority, stage, tile);
}

function buildTreeAndList(authority: SparseCM12ScalarWorkAuthority, stage: number): void {
  const { layout, words } = authority;
  const levels = layout.treeLevelBaseWords[stage]!;
  const leafCount = layout.treeLevelCounts[0]!;
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    let count = 0;
    const end = Math.min(layout.tileCapacity, (leaf + 1) * SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES);
    for (let tile = leaf * SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES; tile < end; tile += 1) {
      const record = scalarTileWord(layout, stage, tile);
      if ((words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags]!
        & SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work) !== 0) count += 1;
    }
    words[levels[0]! + leaf] = count;
  }
  for (let level = 1; level < levels.length; level += 1) {
    for (let node = 0; node < layout.treeLevelCounts[level]!; node += 1) {
      let sum = 0;
      const childStart = node * SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH;
      const childEnd = Math.min(layout.treeLevelCounts[level - 1]!,
        childStart + SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH);
      for (let child = childStart; child < childEnd; child += 1) sum += words[levels[level - 1]! + child]!;
      words[levels[level]! + node] = sum;
    }
  }
  let rank = 0;
  for (let tile = 0; tile < layout.tileCapacity; tile += 1) {
    const record = scalarTileWord(layout, stage, tile);
    if ((words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags]!
      & SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work) !== 0) {
      words[layout.workListBaseWords + stage * layout.tileCapacity + rank] = tile; rank += 1;
    }
  }
}

export function classifySparseCM12ScalarAuthority(authority: SparseCM12ScalarWorkAuthority): boolean {
  const { words, layout } = authority; const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  if (words[h.phase] !== SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting) return fail(authority,
    SPARSE_CM12_SCALAR_AUTHORITY_FAULT.invalidPhase, SPARSE_CM12_SCALAR_AUTHORITY_INVALID,
    SPARSE_CM12_SCALAR_AUTHORITY_INVALID);
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    let work = 0; let clean = 0;
    for (let tile = 0; tile < layout.tileCapacity; tile += 1) {
      let causes = 0;
      if (!exactBank(authority, stage, tile, 0)) causes |= SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.bank0NotExact;
      if (!exactBank(authority, stage, tile, 1)) causes |= SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.bank1NotExact;
      if (!dependenciesValid(authority, stage, tile)) causes |= SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.dependency;
      const hasFPLReceipt = fplValid(authority, stage, tile);
      if (!hasFPLReceipt) causes |= SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.fplReceipt;
      const record = scalarTileWord(layout, stage, tile);
      const cleanSkip = causes === 0;
      words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.generation] = words[h.candidateGeneration]!;
      words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags]
        = SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.classified
          | (hasFPLReceipt ? SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.fplReceipt : 0)
          | (cleanSkip ? SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.exactCleanSkip
            : SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work);
      words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.causeMask] = causes;
      words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.fplPacket]
        = words[scalarFPLReceiptWord(layout, stage, tile)
          + SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT.packed]! & 0xffff;
      if (cleanSkip) clean += 1; else work += 1;
    }
    buildTreeAndList(authority, stage);
    const base = scalarStageHeaderWord(layout, stage); const sh = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER;
    const root = words[layout.treeLevelBaseWords[stage]!.at(-1)!]!;
    if (root !== work) return fail(authority, SPARSE_CM12_SCALAR_AUTHORITY_FAULT.treeMismatch, stage,
      SPARSE_CM12_SCALAR_AUTHORITY_INVALID);
    words[base + sh.workCount] = work; words[base + sh.cleanCount] = clean;
    words[base + sh.classifiedCount] = layout.tileCapacity; words[base + sh.treeRootCount] = root;
    words[base + sh.workIndirectX] = work; words[base + sh.cleanIndirectX] = clean;
    words[base + sh.phase] = SPARSE_CM12_SCALAR_AUTHORITY_PHASE.classified;
  }
  words[h.phase] = SPARSE_CM12_SCALAR_AUTHORITY_PHASE.sealed;
  return true;
}

export function sparseCM12ScalarRankSelect(
  authority: SparseCM12ScalarWorkAuthority, stage: number, rank: number,
): number | undefined {
  const { layout, words } = authority; const base = scalarStageHeaderWord(layout, stage);
  const workCount = words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.workCount]!;
  if (!Number.isInteger(rank) || rank < 0 || rank >= workCount) return undefined;
  let node = 0; let remaining = rank;
  const levels = layout.treeLevelBaseWords[stage]!;
  for (let level = levels.length - 1; level > 0; level -= 1) {
    const childStart = node * SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH;
    const childEnd = Math.min(layout.treeLevelCounts[level - 1]!,
      childStart + SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH);
    for (let child = childStart; child < childEnd; child += 1) {
      const count = words[levels[level - 1]! + child]!;
      if (remaining < count) { node = child; break; }
      remaining -= count;
    }
  }
  const start = node * SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES;
  const end = Math.min(layout.tileCapacity, start + SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES);
  for (let tile = start; tile < end; tile += 1) {
    const record = scalarTileWord(layout, stage, tile);
    if ((words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags]!
      & SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work) === 0) continue;
    if (remaining === 0) return tile;
    remaining -= 1;
  }
  return undefined;
}

/** Records HEAD execution coverage only; this module never receives a physics buffer. */
export function publishSparseCM12ScalarExecution(
  authority: SparseCM12ScalarWorkAuthority, stage: number, tile: number,
): boolean {
  const { layout, words } = authority; const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  if (words[h.phase] !== SPARSE_CM12_SCALAR_AUTHORITY_PHASE.sealed) return false;
  const record = scalarTileWord(layout, stage, tile);
  if ((words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags]!
    & SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work) === 0) return false;
  words[record + SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags]!
    |= SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.executed;
  const base = scalarStageHeaderWord(layout, stage);
  words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.receiptCount]! += 1;
  return true;
}

export function commitSparseCM12ScalarAuthority(authority: SparseCM12ScalarWorkAuthority): boolean {
  const { layout, words } = authority; const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  if (words[h.phase] !== SPARSE_CM12_SCALAR_AUTHORITY_PHASE.sealed) return fail(authority,
    SPARSE_CM12_SCALAR_AUTHORITY_FAULT.invalidPhase, SPARSE_CM12_SCALAR_AUTHORITY_INVALID,
    SPARSE_CM12_SCALAR_AUTHORITY_INVALID);
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    for (let tile = 0; tile < layout.tileCapacity; tile += 1) {
      const record = scalarTileWord(layout, stage, tile); const flags = words[record + 1]!;
      if ((flags & SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work) !== 0
        && (flags & SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.executed) === 0) {
        const base = scalarStageHeaderWord(layout, stage);
        words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.fault]
          = SPARSE_CM12_SCALAR_AUTHORITY_FAULT.missingExecution;
        words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.firstFaultTile] = tile;
        return fail(authority, SPARSE_CM12_SCALAR_AUTHORITY_FAULT.missingExecution, stage, tile);
      }
    }
  }
  words[h.acceptedGeneration] = words[h.candidateGeneration]!;
  words[h.phase] = SPARSE_CM12_SCALAR_AUTHORITY_PHASE.accepted;
  return true;
}
