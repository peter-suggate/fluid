import {
  SPARSE_CM12_FACTORED_AEI_INVALID,
  type SparseCM12FactoredAEICanonicalDescriptor,
  type SparseCM12FactoredAEICatalog,
  type SparseCM12FactoredAEIPatchDescriptor,
} from "./sparse-cm12-factored-aei-topology";
import {
  SPARSE_CM12_PACKED_TEMPLATE_CELL_WORDS,
  SPARSE_CM12_PACKED_TEMPLATE_HEADER,
  SPARSE_CM12_PACKED_TEMPLATE_MAGIC,
} from "./sparse-cm12-factored-aei-packed-template";

export const SPARSE_CM12_INTERNED_BOUNDARY_MAGIC = 0x4942_4f31; // IBO1
export const SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS = 32;
export const SPARSE_CM12_INTERNED_BOUNDARY_DIRECTORY_WORDS = 4;
export const SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS = 8;
export const SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS = 7;
export const SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS = 2;
export const SPARSE_CM12_INTERNED_BOUNDARY_SLOT_HEADER_WORDS = 32;
export const SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS = 8;
// A selected reference stores `[template:12|targetLeaf:20,rowBase]`. The
// all-ones target encodes an exterior face and the all-ones identity encodes
// an unused record. This removes one permanently redundant word per one of the
// 24 bounded face records without changing the logical vec3 reference ABI.
export const SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS = 2;
export const SPARSE_CM12_INTERNED_BOUNDARY_LOGICAL_REF_WORDS = 3;
export const SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_BITS = 20;
export const SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_MASK = 0x000f_ffff;
export const SPARSE_CM12_INTERNED_BOUNDARY_REF_TEMPLATE_MAXIMUM = 0x0000_0fff;
export const SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF = 24;

export const packSparseCM12InternedBoundaryRefIdentity = (
  templateId: number, targetLeaf: number,
): number => {
  if (!Number.isSafeInteger(templateId) || templateId < 0
    || templateId >= SPARSE_CM12_INTERNED_BOUNDARY_REF_TEMPLATE_MAXIMUM) {
    throw new RangeError("IBO reference template exceeds its 12-bit identity");
  }
  const target = targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID
    ? SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_MASK : targetLeaf;
  if (!Number.isSafeInteger(targetLeaf) || targetLeaf < 0
    || (targetLeaf !== SPARSE_CM12_FACTORED_AEI_INVALID
      && targetLeaf >= SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_MASK)) {
    throw new RangeError("IBO reference target exceeds its 20-bit identity");
  }
  return ((templateId << SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_BITS)
    | target) >>> 0;
};

export const unpackSparseCM12InternedBoundaryRefIdentity = (
  packed: number,
): readonly [number, number] => packed === SPARSE_CM12_FACTORED_AEI_INVALID
  ? Object.freeze([SPARSE_CM12_FACTORED_AEI_INVALID,
    SPARSE_CM12_FACTORED_AEI_INVALID] as const)
  : Object.freeze([packed >>> SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_BITS,
    (packed & SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_MASK)
      === SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_MASK
      ? SPARSE_CM12_FACTORED_AEI_INVALID
      : packed & SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_MASK] as const);

export interface SparseCM12InternedBoundaryTemplate {
  readonly id: number;
  readonly relation: number;
  readonly sourceResolution: number;
  readonly targetResolution: number;
  readonly side: number;
  readonly sourceDimensions: readonly [number, number, number];
  readonly targetDimensions: readonly [number, number, number];
  readonly rowCount: number;
  readonly termCount: number;
  /** Stable-ID-free exact local operator payload. */
  readonly words: Uint32Array;
}

export interface SparseCM12InternedBoundaryInstance {
  readonly sourceLeaf: number;
  readonly sourceCanonicalId: number;
  readonly targetLeaf: number;
  readonly targetCanonicalId: number;
  readonly side: number;
  readonly localRef: number;
  readonly templateId: number;
  readonly authoritativeRowBase: number;
}

export interface SparseCM12InternedBoundaryLayout {
  readonly leafCapacity: number;
  readonly canonicalCapacity: number;
  readonly templateCount: number;
  readonly templatePayloadWords: number;
  readonly canonicalBaseWords: number;
  readonly templateDirectoryBaseWords: number;
  readonly templatePayloadBaseWords: number;
  readonly immutableWords: number;
  readonly immutableBytes: number;
  readonly slotBaseWords: readonly [number, number];
  readonly slotLeafBaseWords: readonly [number, number];
  readonly slotRefBaseWords: readonly [number, number];
  readonly wordsPerSlot: number;
  readonly bytesPerSlot: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12InternedBoundaryCompilation {
  readonly catalog: SparseCM12FactoredAEICatalog;
  readonly layout: SparseCM12InternedBoundaryLayout;
  readonly templates: readonly SparseCM12InternedBoundaryTemplate[];
  readonly instances: readonly SparseCM12InternedBoundaryInstance[];
  readonly templateIdByPatch: Uint32Array;
  readonly rowBaseByPatch: Uint32Array;
  /** Fixed `[templateId,targetLeaf,rowBase]` records; INVALID means unused. */
  readonly slotRefs: Uint32Array;
  readonly representedStableRows: Uint32Array;
  readonly expectedStableRows: Uint32Array;
  readonly exactStableRowSet: boolean;
  readonly firstMissing: number | null;
  readonly firstExtra: number | null;
}

const align = (value: number, words = 64) => Math.ceil(value / words) * words;
const packDimensions = (value: readonly [number, number, number]) =>
  (value[0] | (value[1] << 10) | (value[2] << 20)) >>> 0;
const rowIds = (catalog: SparseCM12FactoredAEICatalog,
  facePatch: SparseCM12FactoredAEIPatchDescriptor): readonly number[] =>
  facePatch.exceptionCount === 0
    ? Array.from({ length: facePatch.rowCount }, (_, local) => facePatch.rowFirst + local)
    : [...catalog.exceptionRows.subarray(facePatch.exceptionFirst,
      facePatch.exceptionFirst + facePatch.exceptionCount)];

interface Reader {
  readonly words: Uint32Array;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly cellBase: number;
  readonly rowBase: number;
  readonly termBase: number;
}
const reader = (words: Uint32Array): Reader => {
  const h = SPARSE_CM12_PACKED_TEMPLATE_HEADER;
  if (words[h.magic] !== SPARSE_CM12_PACKED_TEMPLATE_MAGIC) {
    throw new Error("IBO packed authority header is invalid");
  }
  return { words, cellCount: words[h.cellCount]!, rowCount: words[h.rowCount]!,
    cellBase: words[h.cellBase]!, rowBase: words[h.rowBase]!, termBase: words[h.termBase]! };
};
const rowWord = (source: Reader, plane: number, row: number) =>
  source.words[source.rowBase + plane * source.rowCount + row]!;
const rowTerms = (source: Reader, row: number) => {
  const packed = rowWord(source, 0, row);
  return { first: packed & 0x007f_ffff, count: packed >>> 23 };
};
const termCell = (source: Reader, term: number) => source.words[source.termBase + 2 * term]!;
const termCoefficient = (source: Reader, term: number) =>
  source.words[source.termBase + 2 * term + 1]!;
const cellMetadata = (source: Reader, cell: number) =>
  source.words[source.cellBase + SPARSE_CM12_PACKED_TEMPLATE_CELL_WORDS * cell + 7]!;

const normalizedCell = (options: Readonly<{
  source: Reader;
  cell: number;
  sourceDescriptor: SparseCM12FactoredAEICanonicalDescriptor;
  targetDescriptor?: SparseCM12FactoredAEICanonicalDescriptor;
}>): number => {
  const { source, cell, sourceDescriptor, targetDescriptor } = options;
  const metadata = cellMetadata(source, cell);
  const leaf = metadata >>> 5, resolution = metadata & 31;
  if (leaf === sourceDescriptor.leafId && resolution === sourceDescriptor.resolution
    && cell >= sourceDescriptor.cellFirst) {
    return (cell - sourceDescriptor.cellFirst) >>> 0;
  }
  if (targetDescriptor && leaf === targetDescriptor.leafId
    && resolution === targetDescriptor.resolution && cell >= targetDescriptor.cellFirst) {
    return (0x8000_0000 | (cell - targetDescriptor.cellFirst)) >>> 0;
  }
  throw new Error(`IBO term cell ${cell} is outside its source/target local operator`);
};

const compileTemplateWords = (options: Readonly<{
  packed: Reader;
  catalog: SparseCM12FactoredAEICatalog;
  facePatch: SparseCM12FactoredAEIPatchDescriptor;
}>): Readonly<{ words: Uint32Array; rowCount: number; termCount: number;
  rowBase: number }> => {
  const { packed, catalog, facePatch } = options;
  const sourceDescriptor = catalog.canonical[facePatch.sourceCanonicalId]!;
  const targetDescriptor = facePatch.targetCanonicalId === SPARSE_CM12_FACTORED_AEI_INVALID
    ? undefined : catalog.canonical[facePatch.targetCanonicalId];
  const rows = rowIds(catalog, facePatch);
  if (rows.length === 0) throw new Error(`IBO patch ${facePatch.id} has no rows`);
  const authoritativeRowBase = Math.min(...rows);
  let termCount = 0;
  for (const row of rows) termCount += rowTerms(packed, row).count;
  const words = new Uint32Array(SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
    + SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS * rows.length
    + SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS * termCount);
  words.set([facePatch.relation, sourceDescriptor.resolution,
    targetDescriptor?.resolution ?? SPARSE_CM12_FACTORED_AEI_INVALID,
    facePatch.sourceSide, packDimensions(sourceDescriptor.validDimensions),
    packDimensions(targetDescriptor?.validDimensions ?? [0, 0, 0]),
    rows.length, termCount]);
  let nextTerm = 0;
  for (let localRow = 0; localRow < rows.length; localRow += 1) {
    const row = rows[localRow]!, terms = rowTerms(packed, row);
    const rowAt = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
      + SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS * localRow;
    words.set([row - authoritativeRowBase, nextTerm | (terms.count << 23),
      rowWord(packed, 1, row) & 0xf000_0000,
      rowWord(packed, 2, row), rowWord(packed, 3, row),
      rowWord(packed, 4, row), rowWord(packed, 5, row)], rowAt);
    for (let localTerm = 0; localTerm < terms.count; localTerm += 1) {
      const term = terms.first + localTerm;
      const termAt = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
        + SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS * rows.length
        + SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS * nextTerm++;
      words[termAt] = normalizedCell({ source: packed, cell: termCell(packed, term),
        sourceDescriptor, targetDescriptor });
      words[termAt + 1] = termCoefficient(packed, term);
    }
  }
  return { words, rowCount: rows.length, termCount, rowBase: authoritativeRowBase };
};

const acceptedRows = (options: Readonly<{
  packed: Reader;
  active: ReadonlySet<number>;
  selectedDescriptorIdByLeaf: readonly number[];
  canonical: readonly SparseCM12FactoredAEICanonicalDescriptor[];
}>): Uint32Array => {
  const result: number[] = [];
  for (let row = 0; row < options.packed.rowCount; row += 1) {
    const requirementAt = rowWord(options.packed, 1, row) & 0x0fff_ffff;
    const count = options.packed.words[requirementAt]!;
    let accepted = count > 0;
    for (let local = 0; local < count && accepted; local += 1) {
      const metadata = options.packed.words[requirementAt + 1 + local]!;
      const leaf = metadata >>> 5, resolution = metadata & 31;
      const selected = options.canonical[options.selectedDescriptorIdByLeaf[leaf]!];
      accepted = options.active.has(leaf) && selected?.resolution === resolution;
    }
    if (accepted) result.push(row);
  }
  return Uint32Array.from(result);
};

/** Compile content-addressed immutable operators and selected slot instances. */
export function compileSparseCM12InternedBoundaryOperators(options: Readonly<{
  catalog: SparseCM12FactoredAEICatalog;
  packedWords: Uint32Array;
  activeLeaves: Iterable<number>;
}>): SparseCM12InternedBoundaryCompilation {
  const { catalog } = options, packed = reader(options.packedWords);
  const active = new Set(options.activeLeaves);
  const templates: SparseCM12InternedBoundaryTemplate[] = [];
  const templateByKey = new Map<string, number>();
  const templateIdByPatch = new Uint32Array(catalog.patches.length);
  const rowBaseByPatch = new Uint32Array(catalog.patches.length);
  let templatePayloadWords = 0;
  for (const facePatch of catalog.patches) {
    const compiled = compileTemplateWords({ packed, catalog, facePatch });
    const key = [...compiled.words].join(",");
    let templateId = templateByKey.get(key);
    if (templateId === undefined) {
      templateId = templates.length;templateByKey.set(key, templateId);
      const source = catalog.canonical[facePatch.sourceCanonicalId]!;
      const target = facePatch.targetCanonicalId === SPARSE_CM12_FACTORED_AEI_INVALID
        ? undefined : catalog.canonical[facePatch.targetCanonicalId];
      templates.push(Object.freeze({ id: templateId, relation: facePatch.relation,
        sourceResolution: source.resolution,
        targetResolution: target?.resolution ?? SPARSE_CM12_FACTORED_AEI_INVALID,
        side: facePatch.sourceSide, sourceDimensions: source.validDimensions,
        targetDimensions: target?.validDimensions ?? [0, 0, 0] as const,
        rowCount: compiled.rowCount, termCount: compiled.termCount,
        words: compiled.words }));
      templatePayloadWords += compiled.words.length;
    }
    templateIdByPatch[facePatch.id] = templateId;
    rowBaseByPatch[facePatch.id] = compiled.rowBase;
  }

  // CPU-facing logical triples remain unpacked; only the two GPU slots use
  // the compact physical encoding.
  const slotRefs = new Uint32Array(catalog.layout.leafCapacity
    * SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF
    * SPARSE_CM12_INTERNED_BOUNDARY_LOGICAL_REF_WORDS).fill(SPARSE_CM12_FACTORED_AEI_INVALID);
  const instances: SparseCM12InternedBoundaryInstance[] = [];
  const represented = new Set<number>();
  for (let leaf = 0; leaf < catalog.layout.leafCapacity; leaf += 1) {
    if (!active.has(leaf)) continue;
    const selectedId = catalog.descriptorIdByLeaf[leaf]!;
    const descriptor = catalog.canonical[selectedId]!;
    for (const axis of [0, 1, 2] as const) for (let local = 0;
      local < descriptor.rowCount[axis]; local += 1) {
      represented.add(descriptor.rowBase[axis]! + local);
    }
    for (let side = 0; side < 6; side += 1) {
      const compatible = catalog.patchIdsByCanonicalSide[selectedId]![side]!
        .map((id) => catalog.patches[id]!).filter((facePatch) =>
          facePatch.targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID
          || active.has(facePatch.targetLeaf)
            && catalog.descriptorIdByLeaf[facePatch.targetLeaf]
              === facePatch.targetCanonicalId);
      if (compatible.length > 4) {
        throw new Error(`IBO selected leaf ${leaf} side ${side} exceeds four instances`);
      }
      for (let localRef = 0; localRef < compatible.length; localRef += 1) {
        const facePatch = compatible[localRef]!, templateId = templateIdByPatch[facePatch.id]!;
        const authoritativeRowBase = rowBaseByPatch[facePatch.id]!;
        const ref = (leaf * SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF
          + side * 4 + localRef) * SPARSE_CM12_INTERNED_BOUNDARY_LOGICAL_REF_WORDS;
        slotRefs[ref] = templateId;
        slotRefs[ref + 1] = facePatch.targetLeaf;
        slotRefs[ref + 2] = authoritativeRowBase;
        instances.push(Object.freeze({ sourceLeaf: leaf, sourceCanonicalId: selectedId,
          targetLeaf: facePatch.targetLeaf, targetCanonicalId: facePatch.targetCanonicalId,
          side, localRef, templateId, authoritativeRowBase }));
        const template = templates[templateId]!;
        for (let row = 0; row < template.rowCount; row += 1) {
          const rowAt = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
            + row * SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS;
          represented.add(authoritativeRowBase + template.words[rowAt]!);
        }
      }
    }
  }
  const expected = acceptedRows({ packed, active,
    selectedDescriptorIdByLeaf: catalog.descriptorIdByLeaf,
    canonical: catalog.canonical });
  const representedRows = Uint32Array.from([...represented].sort((a, b) => a - b));
  const expectedSet = new Set(expected), representedSet = new Set(representedRows);
  const missing = [...expectedSet].filter((row) => !representedSet.has(row));
  const extra = [...representedSet].filter((row) => !expectedSet.has(row));
  const canonicalBaseWords = SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS;
  const templateDirectoryBaseWords = canonicalBaseWords
    + catalog.layout.canonicalCapacity * 16;
  const templatePayloadBaseWords = templateDirectoryBaseWords
    + templates.length * SPARSE_CM12_INTERNED_BOUNDARY_DIRECTORY_WORDS;
  const immutableWords = align(templatePayloadBaseWords + templatePayloadWords);
  const wordsPerSlot = align(SPARSE_CM12_INTERNED_BOUNDARY_SLOT_HEADER_WORDS
    + catalog.layout.leafCapacity * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS
    + catalog.layout.leafCapacity * SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF
      * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS);
  const slotBaseWords = [immutableWords, immutableWords + wordsPerSlot] as const;
  const slotLeafBaseWords = [slotBaseWords[0]
    + SPARSE_CM12_INTERNED_BOUNDARY_SLOT_HEADER_WORDS,
  slotBaseWords[1] + SPARSE_CM12_INTERNED_BOUNDARY_SLOT_HEADER_WORDS] as const;
  const slotRefBaseWords = [slotLeafBaseWords[0]
    + catalog.layout.leafCapacity * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS,
  slotLeafBaseWords[1]
    + catalog.layout.leafCapacity * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS] as const;
  const totalWords = immutableWords + 2 * wordsPerSlot;
  const layout = Object.freeze({ leafCapacity: catalog.layout.leafCapacity,
    canonicalCapacity: catalog.layout.canonicalCapacity,
    templateCount: templates.length, templatePayloadWords, canonicalBaseWords,
    templateDirectoryBaseWords, templatePayloadBaseWords, immutableWords,
    immutableBytes: 4 * immutableWords, wordsPerSlot, bytesPerSlot: 4 * wordsPerSlot,
    slotBaseWords, slotLeafBaseWords, slotRefBaseWords, totalWords,
    totalBytes: 4 * totalWords });
  return Object.freeze({ catalog, layout, templates: Object.freeze(templates),
    instances: Object.freeze(instances), templateIdByPatch, rowBaseByPatch, slotRefs,
    representedStableRows: representedRows, expectedStableRows: expected,
    exactStableRowSet: missing.length === 0 && extra.length === 0,
    firstMissing: missing[0] ?? null, firstExtra: extra[0] ?? null });
}
