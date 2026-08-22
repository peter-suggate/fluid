/** TFX1: candidate-only SCA/PTR effects coupled to one topology transaction. */
export const SPARSE_CM12_TOPOLOGY_EFFECTS_MAGIC = 0x5446_5831;
export const SPARSE_CM12_TOPOLOGY_EFFECTS_VERSION = 1;
export const SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS = 64;
export const SPARSE_CM12_TOPOLOGY_EFFECTS_INVALID = 0xffff_ffff;

export const SPARSE_CM12_TOPOLOGY_EFFECTS_PHASE = Object.freeze({
  idle: 0, recording: 1, preflighted: 2, authorized: 3, published: 4, fault: 5,
} as const);

export const SPARSE_CM12_TOPOLOGY_EFFECTS_DISPATCH_ORDER = Object.freeze([
  "begin PTR/SIR candidate generations",
  "beginSparseCM12TopologyEffectsPreflight (snapshot distinct SCA/PTR target generations)",
  "candidate producers call tfxRecordSCATile/tfxRecordPTRBrick",
  "finalizeSparseCM12TopologyEffectsPreflight (exact count/hash/capacity/compatibility)",
  "aggregate authorization calls tfxPreflightReady then tfxAuthorize; no accepted write",
  "publishSparseCM12TopologySCAEffects + publishSparseCM12TopologyPTREffects (indirect O(delta))",
  "finishSparseCM12TopologyEffectsPublication (infallible ordered singleton)",
  "publish remaining authorized fields/membership/SCA/VEX/counters",
  "one final singleton flips the shared accepted selector",
] as const);

export const SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER = Object.freeze({
  magic: 0, version: 1, totalWords: 2, phase: 3, generation: 4,
  fault: 5, firstFaultId: 6, scaCount: 7, ptrCount: 8, ptrLeafCount: 9,
  scaNewCount: 10, ptrNewCount: 11, ptrNewLeafCount: 12,
  scaHash: 13, ptrHash: 14, expectedEffects: 15, coveredEffects: 16,
  scaDispatchX: 17, scaDispatchY: 18, scaDispatchZ: 19,
  ptrDispatchX: 20, ptrDispatchY: 21, ptrDispatchZ: 22,
  scaCapacity: 23, ptrCapacity: 24, ptrLeafCapacity: 25,
  scaStampBase: 26, scaCauseBase: 27, scaListBase: 28,
  ptrStampBase: 29, ptrListBase: 30, scaTargetGeneration: 31,
  ptrTargetGeneration: 32, reservedBase: 33,
} as const);

export interface SparseCM12TopologyEffectsAuthorityLayout {
  readonly baseWords: number;
  readonly scaCapacity: number;
  readonly ptrCapacity: number;
  readonly ptrLeafCapacity: number;
  readonly scaStampBaseWords: number;
  readonly scaCauseBaseWords: number;
  readonly scaListBaseWords: number;
  readonly ptrStampBaseWords: number;
  readonly ptrOldStateBaseWords: number;
  readonly ptrNewStateBaseWords: number;
  readonly ptrCauseBaseWords: number;
  readonly ptrOwnsLeafBaseWords: number;
  readonly ptrListBaseWords: number;
  readonly ptrLeafStampBaseWords: number;
  readonly ptrLeafListBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const align64 = (words: number): number => Math.ceil(words / 64) * 64;
const capacity = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1 || value >= 0x8000_0000) {
    throw new RangeError(`${label} must be a positive u31`);
  }
  return value;
};

export function createSparseCM12TopologyEffectsAuthorityLayout(options: Readonly<{
  baseWords: number;
  scaCapacity: number;
  ptrCapacity: number;
  ptrLeafCapacity: number;
}>): SparseCM12TopologyEffectsAuthorityLayout {
  if (!Number.isSafeInteger(options.baseWords) || options.baseWords < 0) {
    throw new RangeError("TFX1 baseWords must be a non-negative safe integer");
  }
  const scaCapacity = capacity(options.scaCapacity, "TFX1 scaCapacity");
  const ptrCapacity = capacity(options.ptrCapacity, "TFX1 ptrCapacity");
  const ptrLeafCapacity = capacity(options.ptrLeafCapacity, "TFX1 ptrLeafCapacity");
  const requiredPtrLeafCapacity = Math.ceil(ptrCapacity / 256);
  if (ptrLeafCapacity < requiredPtrLeafCapacity) {
    throw new RangeError(`TFX1 ptrLeafCapacity must cover every PTR brick leaf (${requiredPtrLeafCapacity})`);
  }
  const baseWords = align64(options.baseWords);
  let at = baseWords + SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS;
  const take = (count: number) => { const result = align64(at); at = result + count; return result; };
  const scaStampBaseWords = take(scaCapacity);
  const scaCauseBaseWords = take(scaCapacity);
  const scaListBaseWords = take(scaCapacity);
  const ptrStampBaseWords = take(ptrCapacity);
  const ptrOldStateBaseWords = take(ptrCapacity);
  const ptrNewStateBaseWords = take(ptrCapacity);
  const ptrCauseBaseWords = take(ptrCapacity);
  const ptrOwnsLeafBaseWords = take(ptrCapacity);
  const ptrListBaseWords = take(ptrCapacity);
  const ptrLeafStampBaseWords = take(ptrLeafCapacity);
  const ptrLeafListBaseWords = take(ptrLeafCapacity);
  const totalWords = align64(at);
  if (totalWords > 0xffff_ffff) {
    throw new RangeError("TFX1 layout offsets must fit u32 WGSL addressing");
  }
  return Object.freeze({ baseWords, scaCapacity, ptrCapacity, ptrLeafCapacity,
    scaStampBaseWords, scaCauseBaseWords, scaListBaseWords, ptrStampBaseWords,
    ptrOldStateBaseWords, ptrNewStateBaseWords, ptrCauseBaseWords,
    ptrOwnsLeafBaseWords, ptrListBaseWords, ptrLeafStampBaseWords,
    ptrLeafListBaseWords, totalWords, totalBytes: 4 * (totalWords - baseWords) });
}

export function createSparseCM12TopologyEffectsAuthorityInitialWords(
  layout: SparseCM12TopologyEffectsAuthorityLayout,
): Uint32Array {
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER;
  words[h.magic] = SPARSE_CM12_TOPOLOGY_EFFECTS_MAGIC;
  words[h.version] = SPARSE_CM12_TOPOLOGY_EFFECTS_VERSION;
  words[h.totalWords] = layout.totalWords;
  words[h.scaCapacity] = layout.scaCapacity;
  words[h.ptrCapacity] = layout.ptrCapacity;
  words[h.ptrLeafCapacity] = layout.ptrLeafCapacity;
  words[h.scaStampBase] = layout.scaStampBaseWords;
  words[h.scaCauseBase] = layout.scaCauseBaseWords;
  words[h.scaListBase] = layout.scaListBaseWords;
  words[h.ptrStampBase] = layout.ptrStampBaseWords;
  words[h.ptrListBase] = layout.ptrListBaseWords;
  words[h.scaDispatchY] = words[h.scaDispatchZ] = 1;
  words[h.ptrDispatchY] = words[h.ptrDispatchZ] = 1;
  words[h.firstFaultId] = SPARSE_CM12_TOPOLOGY_EFFECTS_INVALID;
  return words;
}

export function sparseCM12TopologyEffectsIndirectByteOffset(
  layout: SparseCM12TopologyEffectsAuthorityLayout, family: "sca" | "ptr",
): number {
  const h = SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER;
  return 4 * (layout.baseWords + (family === "sca" ? h.scaDispatchX : h.ptrDispatchX));
}

export interface SparseCM12TopologySCAEffect {
  readonly tile: number;
  readonly cause: number;
}

export interface SparseCM12TopologyPTREffect {
  readonly brick: number;
  readonly oldState: number;
  readonly newState: number;
  readonly cause: number;
}

const u32 = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be a u32`);
  }
  return value >>> 0;
};
const hashWord = (value: number, hash: number): number =>
  Math.imul((hash ^ value) >>> 0, 0x0100_0193) >>> 0;
const scaEffectHash = (effect: SparseCM12TopologySCAEffect): number =>
  hashWord(effect.cause, hashWord(effect.tile, 0x811c_9dc5));
const ptrEffectHash = (effect: SparseCM12TopologyPTREffect): number =>
  hashWord(effect.cause, hashWord(effect.newState, hashWord(effect.oldState,
    hashWord(effect.brick, 0x811c_9dc5))));

/** CPU oracle for the exact, order-independent candidate-effects receipt. */
export function compileSparseCM12TopologyEffectsReference(options: Readonly<{
  scaCapacity: number;
  ptrCapacity: number;
  sca: Iterable<SparseCM12TopologySCAEffect>;
  ptr: Iterable<SparseCM12TopologyPTREffect>;
}>): Readonly<{
  sca: readonly SparseCM12TopologySCAEffect[];
  ptr: readonly SparseCM12TopologyPTREffect[];
  ptrLeaves: readonly number[];
  scaHash: number;
  ptrHash: number;
}> {
  const scaCapacity = capacity(options.scaCapacity, "TFX1 reference scaCapacity");
  const ptrCapacity = capacity(options.ptrCapacity, "TFX1 reference ptrCapacity");
  const sca = new Map<number, number>();
  for (const effect of options.sca) {
    const tile = u32(effect.tile, "TFX1 SCA tile");
    const cause = u32(effect.cause, "TFX1 SCA cause");
    if (tile >= scaCapacity || cause === 0) throw new RangeError("TFX1 invalid SCA effect");
    sca.set(tile, ((sca.get(tile) ?? 0) | cause) >>> 0);
  }
  const ptr = new Map<number, SparseCM12TopologyPTREffect>();
  for (const input of options.ptr) {
    const effect = Object.freeze({ brick: u32(input.brick, "TFX1 PTR brick"),
      oldState: u32(input.oldState, "TFX1 PTR oldState"),
      newState: u32(input.newState, "TFX1 PTR newState"),
      cause: u32(input.cause, "TFX1 PTR cause") });
    if (effect.brick >= ptrCapacity || effect.cause === 0) {
      throw new RangeError("TFX1 invalid PTR effect");
    }
    const prior = ptr.get(effect.brick);
    if (prior && (prior.oldState !== effect.oldState || prior.newState !== effect.newState)) {
      throw new Error(`TFX1 conflicting PTR effect for brick ${effect.brick}`);
    }
    ptr.set(effect.brick, prior ? Object.freeze({ ...prior,
      cause: (prior.cause | effect.cause) >>> 0 }) : effect);
  }
  const scaEffects = [...sca].sort(([a], [b]) => a - b)
    .map(([tile, cause]) => Object.freeze({ tile, cause }));
  const ptrEffects = [...ptr.values()].sort((a, b) => a.brick - b.brick);
  const leaves = [...new Set(ptrEffects.map((effect) => Math.floor(effect.brick / 256)))];
  return Object.freeze({ sca: Object.freeze(scaEffects), ptr: Object.freeze(ptrEffects),
    ptrLeaves: Object.freeze(leaves),
    scaHash: scaEffects.reduce((hash, effect) => (hash ^ scaEffectHash(effect)) >>> 0, 0),
    ptrHash: ptrEffects.reduce((hash, effect) => (hash ^ ptrEffectHash(effect)) >>> 0, 0) });
}
