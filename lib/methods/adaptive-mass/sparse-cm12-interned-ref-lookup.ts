import { SPARSE_CM12_FACTORED_AEI_INVALID } from
  "./sparse-cm12-factored-aei-topology";
import type { SparseCM12InternedBoundaryCompilation } from
  "./sparse-cm12-interned-boundary-operators";

export const SPARSE_CM12_INTERNED_REF_LOOKUP_MAGIC = 0x4952_4c31; // IRL1
export const SPARSE_CM12_INTERNED_REF_LOOKUP_HEADER_WORDS = 16;
export const SPARSE_CM12_INTERNED_REF_LOOKUP_ENTRY_WORDS = 2;
export const SPARSE_CM12_INTERNED_REF_LOOKUP_INVALID16 = 0xffff;
export const SPARSE_CM12_INTERNED_REF_LOOKUP_EXTERIOR_DELTA16 = 0x8000;

export interface SparseCM12InternedRefLookupLayout {
  readonly baseWords: number;
  readonly canonicalCapacity: number;
  readonly sideDirectoryCount: number;
  readonly directoryBaseWords: number;
  readonly templateDirectoryBaseWords: number;
  readonly entryBaseWords: number;
  readonly templateCount: number;
  readonly templateEntryCount: number;
  readonly entryCount: number;
  readonly fallbackAnchorBaseWords: number;
  readonly fallbackAnchorCount: number;
  readonly levelsPerLeaf: number;
  readonly maximumEntriesPerSide: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12InternedRefLookup {
  readonly compilation: SparseCM12InternedBoundaryCompilation;
  readonly layout: SparseCM12InternedRefLookupLayout;
  readonly words: Uint32Array;
}

const align64 = (value: number) => Math.ceil(value / 64) * 64;
const putU16 = (words: Uint32Array, baseWords: number, index: number, value: number) => {
  const at = Math.floor(index / 2) - baseWords;
  const shift = 16 * (index & 1);
  words[at] = ((words[at]! & ~(0xffff << shift)) | (value << shift)) >>> 0;
};
const getU16 = (words: Uint32Array, baseWords: number, index: number) => {
  const at = Math.floor(index / 2) - baseWords;
  return (words[at]! >>> (16 * (index & 1))) & 0xffff;
};

/**
 * Pack the exact physical instantiation mapping omitted from content-addressed
 * IBO templates. Source-canonical/side lists are interned after replacing the
 * physical target canonical and stable row base with signed target and row
 * deltas. The only canonical descriptors without an axis row anchor are the
 * resolution-one variants; their six per-leaf anchors are retained explicitly.
 */
export function createSparseCM12InternedRefLookup(options: Readonly<{
  ibo: SparseCM12InternedBoundaryCompilation;
  baseWords?: number;
  maximumEntriesPerSide?: number;
}>): SparseCM12InternedRefLookup {
  const { ibo } = options;
  const baseWords = align64(options.baseWords ?? 0);
  const maximumAllowed = options.maximumEntriesPerSide ?? 16;
  const canonicalCapacity = ibo.catalog.layout.canonicalCapacity;
  const leafCapacity = ibo.catalog.layout.leafCapacity;
  // The empty-world rung has no canonical records, but its ABI still carries
  // the ladder width. Avoid manufacturing a leaf merely to make 0/0 defined.
  const levelsPerLeaf = leafCapacity === 0
    ? ibo.catalog.layout.levelCount
    : canonicalCapacity / leafCapacity;
  if (!Number.isSafeInteger(levelsPerLeaf) || levelsPerLeaf < 1) {
    throw new Error("IRL1 canonical capacity is not leaf-major");
  }
  const sideDirectoryCount = canonicalCapacity * 6;
  const groups: number[][] = Array.from({ length: canonicalCapacity * 6 }, () => []);
  for (const patch of ibo.catalog.patches) {
    groups[patch.sourceCanonicalId * 6 + patch.sourceSide]!.push(patch.id);
  }
  let maximumEntriesPerSide = 0;
  for (let group = 0; group < groups.length; group += 1) {
    const seen = new Set<number>();
    for (const patchId of groups[group]!) {
      const target = ibo.catalog.patches[patchId]!.targetCanonicalId;
      if (seen.has(target)) {
        throw new Error(`IRL1 source canonical/side ${group} has duplicate target ${target}`);
      }
      seen.add(target);
    }
    maximumEntriesPerSide = Math.max(maximumEntriesPerSide, groups[group]!.length);
  }
  if (maximumEntriesPerSide > maximumAllowed) {
    throw new Error(`IRL1 side fanout ${maximumEntriesPerSide} exceeds ${maximumAllowed}`);
  }
  if (ibo.catalog.patches.length >= SPARSE_CM12_INTERNED_REF_LOOKUP_INVALID16
    || canonicalCapacity >= SPARSE_CM12_INTERNED_REF_LOOKUP_INVALID16
    || ibo.templates.length >= SPARSE_CM12_INTERNED_REF_LOOKUP_INVALID16) {
    throw new Error("IRL1 u16 identity capacity exceeded");
  }
  const fallbackAnchors = new Uint32Array(canonicalCapacity)
    .fill(SPARSE_CM12_FACTORED_AEI_INVALID);
  for (const patch of ibo.catalog.patches) fallbackAnchors[patch.sourceCanonicalId] = Math.min(
    fallbackAnchors[patch.sourceCanonicalId]!, ibo.rowBaseByPatch[patch.id]!);
  for (let canonical = 0; canonical < canonicalCapacity; canonical += 1) {
    if (fallbackAnchors[canonical] === SPARSE_CM12_FACTORED_AEI_INVALID) {
      fallbackAnchors[canonical] = 0;
    }
  }
  const normalizedGroups: Array<readonly (readonly [number, number])[]> = [];
  const groupTemplateIds: number[] = [];
  const templateByKey = new Map<string, number>();
  for (let group = 0; group < groups.length; group += 1) {
    const sourceCanonical = Math.floor(group / 6), side = group % 6;
    const descriptor = ibo.catalog.canonical[sourceCanonical]!;
    if (descriptor.id !== sourceCanonical
      || descriptor.leafId !== Math.floor(sourceCanonical / levelsPerLeaf)) {
      throw new Error("IRL1 canonical descriptors are not stable leaf-major identities");
    }
    let anchor = descriptor.rowBase[side >>> 1]!;
    if (anchor === SPARSE_CM12_FACTORED_AEI_INVALID) {
      anchor = descriptor.rowBase.find((value) =>
        value !== SPARSE_CM12_FACTORED_AEI_INVALID) ?? SPARSE_CM12_FACTORED_AEI_INVALID;
      if (anchor === SPARSE_CM12_FACTORED_AEI_INVALID) {
        anchor = fallbackAnchors[sourceCanonical]!;
      }
    }
    const entries = groups[group]!.map((patchId) => {
      const patch = ibo.catalog.patches[patchId]!;
      const delta = patch.targetCanonicalId === SPARSE_CM12_FACTORED_AEI_INVALID
        ? SPARSE_CM12_INTERNED_REF_LOOKUP_EXTERIOR_DELTA16
        : patch.targetCanonicalId - sourceCanonical;
      if (delta !== SPARSE_CM12_INTERNED_REF_LOOKUP_EXTERIOR_DELTA16
        && (delta <= -0x8000 || delta >= 0x8000)) {
        throw new Error(`IRL1 target canonical delta ${delta} exceeds i16`);
      }
      const encodedDelta = delta === SPARSE_CM12_INTERNED_REF_LOOKUP_EXTERIOR_DELTA16
        ? delta : delta & 0xffff;
      const rowDelta = ibo.rowBaseByPatch[patchId]! - anchor;
      if (!Number.isSafeInteger(rowDelta) || rowDelta < -0x8000_0000
        || rowDelta >= 0x8000_0000) {
        throw new Error(`IRL1 row delta ${rowDelta} is outside i32`);
      }
      return Object.freeze([(ibo.templateIdByPatch[patchId]! << 16) | encodedDelta,
        rowDelta >>> 0] as const);
    });
    const key = JSON.stringify(entries);
    let templateId = templateByKey.get(key);
    if (templateId === undefined) {
      templateId = normalizedGroups.length;
      if (templateId >= SPARSE_CM12_INTERNED_REF_LOOKUP_INVALID16) {
        throw new Error("IRL1 normalized group template capacity exceeded");
      }
      templateByKey.set(key, templateId);
      normalizedGroups.push(Object.freeze(entries));
    }
    groupTemplateIds.push(templateId);
  }
  const templateEntryCount = normalizedGroups.reduce((sum, entries) => sum + entries.length, 0);
  if (templateEntryCount >= SPARSE_CM12_INTERNED_REF_LOOKUP_INVALID16) {
    throw new Error("IRL1 normalized entry capacity exceeded");
  }
  const directoryBaseWords = baseWords + SPARSE_CM12_INTERNED_REF_LOOKUP_HEADER_WORDS;
  const templateDirectoryBaseWords = directoryBaseWords
    + Math.ceil(sideDirectoryCount / 2);
  const entryBaseWords = templateDirectoryBaseWords
    + Math.ceil((normalizedGroups.length + 1) / 2);
  const fallbackAnchorBaseWords = entryBaseWords
    + SPARSE_CM12_INTERNED_REF_LOOKUP_ENTRY_WORDS * templateEntryCount;
  const totalWords = align64(fallbackAnchorBaseWords + fallbackAnchors.length);
  const words = new Uint32Array(totalWords - baseWords);
  words.set([SPARSE_CM12_INTERNED_REF_LOOKUP_MAGIC, 2, canonicalCapacity,
    ibo.catalog.patches.length, directoryBaseWords, templateDirectoryBaseWords,
    entryBaseWords, normalizedGroups.length, templateEntryCount,
    maximumEntriesPerSide, fallbackAnchorBaseWords, fallbackAnchors.length,
    levelsPerLeaf, totalWords], 0);
  groupTemplateIds.forEach((templateId, group) =>
    putU16(words, baseWords, 2 * directoryBaseWords + group, templateId));
  let entry = 0;
  normalizedGroups.forEach((entries, templateId) => {
    putU16(words, baseWords, 2 * templateDirectoryBaseWords + templateId, entry);
    for (const value of entries) {
      const at = entryBaseWords - baseWords
        + SPARSE_CM12_INTERNED_REF_LOOKUP_ENTRY_WORDS * entry++;
      words.set(value, at);
    }
  });
  putU16(words, baseWords, 2 * templateDirectoryBaseWords
    + normalizedGroups.length, entry);
  words.set(fallbackAnchors, fallbackAnchorBaseWords - baseWords);
  if (entry !== templateEntryCount) throw new Error("IRL1 normalized entry count drifted");
  const layout = Object.freeze({ baseWords, canonicalCapacity, sideDirectoryCount,
    directoryBaseWords, templateDirectoryBaseWords, entryBaseWords,
    templateCount: normalizedGroups.length, templateEntryCount,
    entryCount: ibo.catalog.patches.length, fallbackAnchorBaseWords,
    fallbackAnchorCount: fallbackAnchors.length, levelsPerLeaf, maximumEntriesPerSide,
    totalWords, totalBytes: 4 * (totalWords - baseWords) });
  return Object.freeze({ compilation: ibo, layout, words });
}

/** CPU oracle for the exact WGSL scheduled-ref lookup. */
export function sparseCM12InternedRefLookup(options: Readonly<{
  lookup: SparseCM12InternedRefLookup;
  sourceCanonicalId: number;
  side: number;
  targetCanonicalId: number;
}>): readonly [number, number, number] | undefined {
  const { lookup } = options, { layout, words } = lookup;
  if (options.sourceCanonicalId < 0
    || options.sourceCanonicalId >= layout.canonicalCapacity
    || options.side < 0 || options.side >= 6) return undefined;
  const group = options.sourceCanonicalId * 6 + options.side;
  const templateId = getU16(words, layout.baseWords,
    2 * layout.directoryBaseWords + group);
  const begin = getU16(words, layout.baseWords,
    2 * layout.templateDirectoryBaseWords + templateId);
  const end = getU16(words, layout.baseWords,
    2 * layout.templateDirectoryBaseWords + templateId + 1);
  const descriptor = lookup.compilation.catalog.canonical[options.sourceCanonicalId]!;
  const canonicalAnchor = descriptor.rowBase[options.side >>> 1]!;
  const alternateAnchor = descriptor.rowBase.find((value) =>
    value !== SPARSE_CM12_FACTORED_AEI_INVALID);
  const anchor = canonicalAnchor !== SPARSE_CM12_FACTORED_AEI_INVALID
    ? canonicalAnchor : alternateAnchor ?? words[layout.fallbackAnchorBaseWords
      - layout.baseWords + options.sourceCanonicalId]!;
  for (let entry = begin; entry < end; entry += 1) {
    const at = layout.entryBaseWords - layout.baseWords
      + SPARSE_CM12_INTERNED_REF_LOOKUP_ENTRY_WORDS * entry;
    const packed = words[at]!;
    const encoded = packed & 0xffff;
    const targetCanonical = encoded === SPARSE_CM12_INTERNED_REF_LOOKUP_EXTERIOR_DELTA16
      ? SPARSE_CM12_FACTORED_AEI_INVALID
      : options.sourceCanonicalId + (encoded << 16 >> 16);
    if (targetCanonical !== options.targetCanonicalId) continue;
    const targetLeaf = targetCanonical === SPARSE_CM12_FACTORED_AEI_INVALID
      ? SPARSE_CM12_FACTORED_AEI_INVALID
      : lookup.compilation.catalog.canonical[targetCanonical]!.leafId;
    return Object.freeze([packed >>> 16, targetLeaf,
      (anchor + words[at + 1]!) >>> 0] as const);
  }
  return undefined;
}
