import {
  SPARSE_CM12_FACTORED_AEI_INVALID,
  type SparseCM12FactoredAEICanonicalDescriptor,
  type SparseCM12FactoredAEIPatchDescriptor,
} from "./sparse-cm12-factored-aei-topology";
import {
  SPARSE_CM12_INTERNED_BOUNDARY_DIRECTORY_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_MAGIC,
  SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF,
  SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS,
  packSparseCM12InternedBoundaryRefIdentity,
  unpackSparseCM12InternedBoundaryRefIdentity,
  type SparseCM12InternedBoundaryCompilation,
  type SparseCM12InternedBoundaryLayout,
} from "./sparse-cm12-interned-boundary-operators";

export const SPARSE_CM12_INTERNED_BOUNDARY_VERSION = 1;
export const SPARSE_CM12_INTERNED_BOUNDARY_SLOT_STATE = Object.freeze({
  mirror: 0, building: 1, ready: 2, fault: 3,
});
export const SPARSE_CM12_INTERNED_BOUNDARY_FAULT = Object.freeze({
  none: 0,
  generation: 1,
  deltaCoverage: 2,
  topologyMismatch: 3,
  canonicalCertificate: 4,
  reference: 5,
  immutableImage: 6,
  selectorChanged: 7,
});

export interface SparseCM12InternedBoundaryImage {
  readonly compilation: SparseCM12InternedBoundaryCompilation;
  readonly layout: SparseCM12InternedBoundaryLayout;
  readonly immutableSupplements: readonly SparseCM12InternedBoundarySupplement[];
  readonly immutableCertificate: SparseCM12InternedBoundaryImmutableCertificate;
  readonly words: Uint32Array;
}

export interface SparseCM12InternedBoundaryImmutableCertificate {
  readonly layoutHash: number;
  readonly canonicalTemplateHash: number;
  readonly supplementHash: number;
  readonly contentHash: number;
  readonly certificateHash: number;
}

export interface SparseCM12InternedBoundarySupplement {
  readonly label: string;
  readonly baseWords: number;
  readonly words: Uint32Array;
}

export interface SparseCM12InternedBoundaryPreflipReceipt {
  readonly passed: boolean;
  readonly acceptedSlot: 0 | 1;
  readonly shadowSlot: 0 | 1;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly fault: number;
  readonly firstFaultRecord: number;
  readonly activeLeafCount: number;
  readonly referenceCount: number;
  readonly canonicalLeafCount: number;
  readonly changedLeafCount: number;
  readonly deltaClosureCount: number;
  readonly leafHash: number;
  readonly referenceHash: number;
  readonly selectorUnchanged: boolean;
}

const fnv = (hash: number, value: number): number => {
  hash ^= value >>> 0;
  return Math.imul(hash, 0x0100_0193) >>> 0;
};
const hashWords = (words: Uint32Array, first: number, end: number,
  hash = 0x811c_9dc5): number => {
  for (let at = first; at < end; at += 1) hash = fnv(hash, words[at]!);
  return hash;
};
const packDimensions = (value: readonly [number, number, number]): number =>
  (value[0] | (value[1] << 10) | (value[2] << 20)) >>> 0;
const align = (value: number, words = 64): number => Math.ceil(value / words) * words;

export function createSparseCM12InternedBoundaryComposedLayout(
  base: SparseCM12InternedBoundaryLayout,
  immutableEndWords: number,
): SparseCM12InternedBoundaryLayout {
  if (!Number.isSafeInteger(immutableEndWords)
    || immutableEndWords < base.immutableWords) {
    throw new RangeError("IBO composed immutable end precedes the operator image");
  }
  const immutableWords = align(immutableEndWords);
  const slotBaseWords = [immutableWords, immutableWords + base.wordsPerSlot] as const;
  const leafOffset = base.slotLeafBaseWords[0] - base.slotBaseWords[0];
  const refOffset = base.slotRefBaseWords[0] - base.slotBaseWords[0];
  const slotLeafBaseWords = [slotBaseWords[0] + leafOffset,
    slotBaseWords[1] + leafOffset] as const;
  const slotRefBaseWords = [slotBaseWords[0] + refOffset,
    slotBaseWords[1] + refOffset] as const;
  const totalWords = immutableWords + 2 * base.wordsPerSlot;
  return Object.freeze({ ...base, immutableWords, immutableBytes: 4 * immutableWords,
    slotBaseWords, slotLeafBaseWords, slotRefBaseWords, totalWords,
    totalBytes: 4 * totalWords });
}

const immutableCertificate = (options: Readonly<{
  compilation: SparseCM12InternedBoundaryCompilation;
  layout: SparseCM12InternedBoundaryLayout;
  supplements: readonly SparseCM12InternedBoundarySupplement[];
  words: Uint32Array;
}>): SparseCM12InternedBoundaryImmutableCertificate => {
  const { compilation, layout, supplements, words } = options;
  let layoutHash = 0x811c_9dc5;
  for (const value of [layout.leafCapacity, layout.canonicalCapacity,
    layout.templateCount, layout.canonicalBaseWords,
    layout.templateDirectoryBaseWords, layout.templatePayloadBaseWords,
    layout.immutableWords, layout.wordsPerSlot, layout.slotBaseWords[0],
    layout.slotBaseWords[1], layout.totalWords]) layoutHash = fnv(layoutHash, value);
  const canonicalTemplateHash = hashWords(words,
    SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS, compilation.layout.immutableWords);
  let supplementHash = 0x811c_9dc5;
  for (const supplement of supplements) {
    supplementHash = fnv(fnv(supplementHash, supplement.baseWords), supplement.words.length);
    supplementHash = hashWords(words, supplement.baseWords,
      supplement.baseWords + supplement.words.length, supplementHash);
  }
  // Covers canonical words, template directory/payload, every supplement and
  // alignment padding. Certificate words live in the excluded fixed header.
  const contentHash = hashWords(words, SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS,
    layout.immutableWords);
  const certificateHash = fnv(fnv(fnv(fnv(fnv(0x811c_9dc5,
    SPARSE_CM12_INTERNED_BOUNDARY_MAGIC), SPARSE_CM12_INTERNED_BOUNDARY_VERSION),
  layoutHash), canonicalTemplateHash), fnv(supplementHash, contentHash));
  return Object.freeze({ layoutHash, canonicalTemplateHash, supplementHash,
    contentHash, certificateHash });
};

const activeSet = (values: Iterable<number>, capacity: number): Set<number> => {
  const result = new Set<number>();
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0 || value >= capacity) {
      throw new RangeError(`IBO active leaf ${value} is outside capacity`);
    }
    result.add(value);
  }
  return result;
};

const canonicalWords = (
  descriptor: SparseCM12FactoredAEICanonicalDescriptor,
): readonly number[] => [descriptor.certified ? 1 : 0, descriptor.leafId,
  descriptor.resolution, descriptor.cellFirst, packDimensions(descriptor.validDimensions),
  descriptor.scaleLog2, descriptor.rowBase[0], descriptor.rowBase[1],
  descriptor.rowBase[2], descriptor.rowCount[0], descriptor.rowCount[1],
  descriptor.rowCount[2], descriptor.canonicalRowCount, descriptor.rowIdHash,
  descriptor.termHash, descriptor.geometryHash];

const descriptorMap = (image: SparseCM12InternedBoundaryImage,
  values: readonly number[]): readonly number[] => {
  const { catalog } = image.compilation;
  if (values.length !== catalog.layout.leafCapacity) {
    throw new Error("IBO descriptor map differs from leaf capacity");
  }
  for (let leaf = 0; leaf < values.length; leaf += 1) {
    const descriptor = catalog.canonical[values[leaf]!];
    if (!descriptor || descriptor.leafId !== leaf) {
      throw new Error(`IBO descriptor ${values[leaf]} is not owned by leaf ${leaf}`);
    }
  }
  return values;
};

const selectedReferences = (image: SparseCM12InternedBoundaryImage, leaf: number,
  active: ReadonlySet<number>, descriptors: readonly number[]):
readonly (readonly SparseCM12FactoredAEIPatchDescriptor[])[] => {
  const { catalog } = image.compilation;
  const descriptorId = descriptors[leaf]!;
  return Array.from({ length: 6 }, (_, side) => !active.has(leaf) ? []
    : catalog.patchIdsByCanonicalSide[descriptorId]![side]!
      .map((id) => catalog.patches[id]!)
      .filter((patch) => patch.targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID
        || active.has(patch.targetLeaf)
          && descriptors[patch.targetLeaf] === patch.targetCanonicalId));
};

const refWord = (image: SparseCM12InternedBoundaryImage, slot: 0 | 1,
  leaf: number, side: number, local: number): number =>
  image.layout.slotRefBaseWords[slot]
    + (leaf * SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF + side * 4 + local)
      * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS;

const writeLeaf = (image: SparseCM12InternedBoundaryImage, slot: 0 | 1,
  leaf: number, generation: number, active: ReadonlySet<number>,
  descriptors: readonly number[]): void => {
  const { compilation, words } = image;
  const descriptorId = descriptors[leaf]!;
  const descriptor = compilation.catalog.canonical[descriptorId]!;
  const faces = selectedReferences(image, leaf, active, descriptors);
  if (faces.some((patches) => patches.length > 4)) {
    throw new Error(`IBO selected leaf ${leaf} exceeds four references on one face`);
  }
  let packedFaceCounts = 0, referenceHash = 0x811c_9dc5;
  for (let side = 0; side < 6; side += 1) {
    const patches = faces[side]!;
    packedFaceCounts |= patches.length << (3 * side);
    for (let local = 0; local < 4; local += 1) {
      const at = refWord(image, slot, leaf, side, local);
      const patch = patches[local];
      if (!patch) {
        words.set([SPARSE_CM12_FACTORED_AEI_INVALID,
          SPARSE_CM12_FACTORED_AEI_INVALID], at);
        continue;
      }
      const template = compilation.templateIdByPatch[patch.id]!;
      const rowBase = compilation.rowBaseByPatch[patch.id]!;
      words.set([packSparseCM12InternedBoundaryRefIdentity(template,
        patch.targetLeaf), rowBase], at);
      referenceHash = fnv(fnv(fnv(fnv(referenceHash,
        side * 4 + local), template), patch.targetLeaf), rowBase);
    }
  }
  const at = image.layout.slotLeafBaseWords[slot]
    + leaf * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS;
  words.set([generation,
    (active.has(leaf) ? 1 : 0) | (descriptor.certified ? 2 : 0),
    descriptorId, descriptor.cellFirst, packDimensions(descriptor.validDimensions),
    packedFaceCounts, referenceHash, descriptor.canonicalRowCount], at);
};

const slotActiveLeaves = (image: SparseCM12InternedBoundaryImage,
  slot: 0 | 1): Set<number> => {
  const result = new Set<number>();
  for (let leaf = 0; leaf < image.layout.leafCapacity; leaf += 1) {
    const at = image.layout.slotLeafBaseWords[slot]
      + leaf * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS;
    if ((image.words[at + 1]! & 1) !== 0) result.add(leaf);
  }
  return result;
};

const slotDescriptors = (image: SparseCM12InternedBoundaryImage,
  slot: 0 | 1): readonly number[] => Array.from(
    { length: image.layout.leafCapacity }, (_, leaf) =>
      image.words[image.layout.slotLeafBaseWords[slot]
        + leaf * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS + 2]!,
  );

const summarizeSlot = (image: SparseCM12InternedBoundaryImage, slot: 0 | 1,
  generation: number, baseGeneration: number, state: number,
  changedLeafCount = 0, deltaHash = 0): void => {
  let activeLeafCount = 0, referenceCount = 0, canonicalLeafCount = 0;
  let leafHash = 0x811c_9dc5, referenceHash = 0x811c_9dc5;
  for (let leaf = 0; leaf < image.layout.leafCapacity; leaf += 1) {
    const leafAt = image.layout.slotLeafBaseWords[slot]
      + leaf * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS;
    if ((image.words[leafAt + 1]! & 1) === 0) continue;
    activeLeafCount += 1;
    if ((image.words[leafAt + 1]! & 2) !== 0) canonicalLeafCount += 1;
    leafHash = fnv(fnv(leafHash, leaf), image.words[leafAt + 2]!);
    for (let side = 0; side < 6; side += 1) for (let local = 0;
      local < (image.words[leafAt + 5]! >>> (3 * side) & 7); local += 1) {
      const at = refWord(image, slot, leaf, side, local);
      const [template, targetLeaf] = unpackSparseCM12InternedBoundaryRefIdentity(
        image.words[at]!);
      referenceCount += 1;
      referenceHash = fnv(fnv(fnv(fnv(fnv(referenceHash,
        leaf), side * 4 + local), template), targetLeaf), image.words[at + 1]!);
    }
  }
  image.words.set([generation, state, baseGeneration, activeLeafCount,
    referenceCount, SPARSE_CM12_INTERNED_BOUNDARY_FAULT.none,
    SPARSE_CM12_FACTORED_AEI_INVALID, leafHash, referenceHash,
    changedLeafCount, deltaHash, canonicalLeafCount],
  image.layout.slotBaseWords[slot]);
};

/** Serialize the exact canonical/template authority and two fixed selected slots. */
export function createSparseCM12InternedBoundaryImage(
  compilation: SparseCM12InternedBoundaryCompilation,
  activeLeaves: Iterable<number>, generation = 1,
  selectedDescriptorIdByLeaf: readonly number[] = compilation.catalog.descriptorIdByLeaf,
  immutableSupplements: readonly SparseCM12InternedBoundarySupplement[] = [],
): SparseCM12InternedBoundaryImage {
  if (!Number.isSafeInteger(generation) || generation < 1 || generation > 0xffff_ffff) {
    throw new RangeError("IBO generation must be a positive u32");
  }
  const { catalog } = compilation;
  const supplements = [...immutableSupplements].sort((a, b) => a.baseWords - b.baseWords)
    .map((supplement) => Object.freeze({ ...supplement, words: supplement.words.slice() }));
  let immutableEndWords = compilation.layout.immutableWords;
  for (const supplement of supplements) {
    if (!Number.isSafeInteger(supplement.baseWords) || supplement.baseWords % 64 !== 0
      || supplement.baseWords < immutableEndWords) {
      throw new Error(`IBO immutable supplement ${supplement.label} overlaps or is unaligned`);
    }
    immutableEndWords = supplement.baseWords + supplement.words.length;
  }
  const layout = createSparseCM12InternedBoundaryComposedLayout(
    compilation.layout, immutableEndWords);
  const words = new Uint32Array(layout.totalWords);
  words.set([SPARSE_CM12_INTERNED_BOUNDARY_MAGIC,
    SPARSE_CM12_INTERNED_BOUNDARY_VERSION, 0, generation,
    SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS, layout.leafCapacity,
    layout.canonicalCapacity, layout.templateCount, layout.canonicalBaseWords,
    layout.templateDirectoryBaseWords, layout.templatePayloadBaseWords,
    layout.slotBaseWords[0], layout.slotBaseWords[1], layout.wordsPerSlot,
    layout.slotLeafBaseWords[0] - layout.slotBaseWords[0],
    layout.slotRefBaseWords[0] - layout.slotBaseWords[0], 16,
    SPARSE_CM12_INTERNED_BOUNDARY_DIRECTORY_WORDS,
    SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS,
    SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS,
    SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS,
    SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS,
    SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS,
    SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF, layout.totalWords], 0);
  for (const descriptor of catalog.canonical) {
    words.set(canonicalWords(descriptor),
      layout.canonicalBaseWords + 16 * descriptor.id);
  }
  let payload = layout.templatePayloadBaseWords;
  for (const template of compilation.templates) {
    words.set([payload, template.words.length, template.rowCount, template.termCount],
      layout.templateDirectoryBaseWords
        + SPARSE_CM12_INTERNED_BOUNDARY_DIRECTORY_WORDS * template.id);
    words.set(template.words, payload);
    payload += template.words.length;
  }
  if (payload !== layout.templatePayloadBaseWords + layout.templatePayloadWords) {
    throw new Error("IBO template serialization differs from its measured layout");
  }
  for (const supplement of supplements) words.set(supplement.words, supplement.baseWords);
  const certificate = immutableCertificate({ compilation, layout, supplements, words });
  words.set([certificate.layoutHash, certificate.canonicalTemplateHash,
    certificate.supplementHash, certificate.contentHash,
    certificate.certificateHash], 25);
  const image: SparseCM12InternedBoundaryImage = { compilation, layout,
    immutableSupplements: Object.freeze(supplements),
    immutableCertificate: certificate, words };
  const active = activeSet(activeLeaves, layout.leafCapacity);
  const descriptors = descriptorMap(image, selectedDescriptorIdByLeaf);
  for (const slot of [0, 1] as const) {
    for (let leaf = 0; leaf < layout.leafCapacity; leaf += 1) {
      writeLeaf(image, slot, leaf, generation, active, descriptors);
    }
    summarizeSlot(image, slot, generation, generation,
      SPARSE_CM12_INTERNED_BOUNDARY_SLOT_STATE.mirror);
  }
  return image;
}

const deltaClosure = (image: SparseCM12InternedBoundaryImage,
  changed: ReadonlySet<number>): Set<number> => {
  const closure = new Set(changed);
  for (const leaf of changed) for (const neighbor of
    image.compilation.catalog.neighborLeavesByLeaf[leaf]!) closure.add(neighbor);
  return closure;
};

/** Rewrite only changed leaves and their immutable face-neighbor closure. */
export function prepareSparseCM12InternedBoundaryShadow(options: Readonly<{
  image: SparseCM12InternedBoundaryImage;
  targetActiveLeaves: Iterable<number>;
  targetDescriptorIdByLeaf: readonly number[];
  changedLeaves: Iterable<number>;
  candidateGeneration: number;
}>): Readonly<{ shadowSlot: 0 | 1; deltaClosure: readonly number[] }> {
  const { image } = options, { layout } = image;
  const acceptedSlot = (image.words[2]! & 1) as 0 | 1;
  const shadowSlot = (1 - acceptedSlot) as 0 | 1;
  const acceptedGeneration = image.words[3]!;
  if (options.candidateGeneration !== acceptedGeneration + 1) {
    throw new Error("IBO candidate generation is not accepted+1");
  }
  const accepted = slotActiveLeaves(image, acceptedSlot);
  const acceptedDescriptors = slotDescriptors(image, acceptedSlot);
  const target = activeSet(options.targetActiveLeaves, layout.leafCapacity);
  const targetDescriptors = descriptorMap(image, options.targetDescriptorIdByLeaf);
  const changed = activeSet(options.changedLeaves, layout.leafCapacity);
  const actualChanged = new Set<number>();
  for (let leaf = 0; leaf < layout.leafCapacity; leaf += 1) {
    if (accepted.has(leaf) !== target.has(leaf)
      || acceptedDescriptors[leaf] !== targetDescriptors[leaf]) actualChanged.add(leaf);
  }
  if (actualChanged.size !== changed.size
    || [...actualChanged].some((leaf) => !changed.has(leaf))) {
    throw new Error("IBO changed leaves do not equal the accepted/target delta");
  }
  const closure = deltaClosure(image, changed);
  let deltaHash = 0x811c_9dc5;
  for (const leaf of [...closure].sort((a, b) => a - b)) {
    writeLeaf(image, shadowSlot, leaf, options.candidateGeneration, target,
      targetDescriptors);
    deltaHash = fnv(deltaHash, leaf);
  }
  summarizeSlot(image, shadowSlot, options.candidateGeneration, acceptedGeneration,
    SPARSE_CM12_INTERNED_BOUNDARY_SLOT_STATE.building, changed.size, deltaHash);
  return Object.freeze({ shadowSlot,
    deltaClosure: Object.freeze([...closure].sort((a, b) => a - b)) });
}

const validateImmutable = (image: SparseCM12InternedBoundaryImage): number | null => {
  const { compilation, words, layout } = image, { catalog } = compilation;
  if (words[0] !== SPARSE_CM12_INTERNED_BOUNDARY_MAGIC
    || words[1] !== SPARSE_CM12_INTERNED_BOUNDARY_VERSION) return 0;
  const expectedCertificate = image.immutableCertificate;
  if (words[25] !== expectedCertificate.layoutHash
    || words[26] !== expectedCertificate.canonicalTemplateHash
    || words[27] !== expectedCertificate.supplementHash
    || words[28] !== expectedCertificate.contentHash
    || words[29] !== expectedCertificate.certificateHash) return 25;
  for (const descriptor of catalog.canonical) {
    const expected = canonicalWords(descriptor);
    const at = layout.canonicalBaseWords + 16 * descriptor.id;
    for (let local = 0; local < expected.length; local += 1) {
      if (words[at + local] !== expected[local]) return descriptor.id;
    }
  }
  for (const template of compilation.templates) {
    const directory = layout.templateDirectoryBaseWords
      + SPARSE_CM12_INTERNED_BOUNDARY_DIRECTORY_WORDS * template.id;
    const payload = words[directory]!;
    if (words[directory + 1] !== template.words.length
      || words[directory + 2] !== template.rowCount
      || words[directory + 3] !== template.termCount) return template.id;
    for (let local = 0; local < template.words.length; local += 1) {
      if (words[payload + local] !== template.words[local]) return template.id;
    }
  }
  for (const supplement of image.immutableSupplements) {
    for (let local = 0; local < supplement.words.length; local += 1) {
      if (words[supplement.baseWords + local] !== supplement.words[local]) {
        return supplement.baseWords + local;
      }
    }
  }
  return null;
};

export function validateSparseCM12InternedBoundaryPreflip(options: Readonly<{
  image: SparseCM12InternedBoundaryImage;
  targetActiveLeaves: Iterable<number>;
  targetDescriptorIdByLeaf: readonly number[];
  changedLeaves: Iterable<number>;
}>): SparseCM12InternedBoundaryPreflipReceipt {
  const { image } = options, { compilation, words } = image;
  const { layout } = image, { catalog } = compilation;
  const acceptedSlot = (words[2]! & 1) as 0 | 1;
  const shadowSlot = (1 - acceptedSlot) as 0 | 1;
  const acceptedGeneration = words[3]!;
  const slotAt = layout.slotBaseWords[shadowSlot];
  const candidateGeneration = words[slotAt]!;
  const accepted = slotActiveLeaves(image, acceptedSlot);
  const acceptedDescriptors = slotDescriptors(image, acceptedSlot);
  const target = activeSet(options.targetActiveLeaves, layout.leafCapacity);
  const targetDescriptors = descriptorMap(image, options.targetDescriptorIdByLeaf);
  const changed = activeSet(options.changedLeaves, layout.leafCapacity);
  const actualChanged = new Set<number>();
  for (let leaf = 0; leaf < layout.leafCapacity; leaf += 1) {
    if (accepted.has(leaf) !== target.has(leaf)
      || acceptedDescriptors[leaf] !== targetDescriptors[leaf]) actualChanged.add(leaf);
  }
  const closure = deltaClosure(image, changed);
  let expectedDeltaHash = 0x811c_9dc5;
  for (const leaf of [...closure].sort((a, b) => a - b)) {
    expectedDeltaHash = fnv(expectedDeltaHash, leaf);
  }
  let fault: number = SPARSE_CM12_INTERNED_BOUNDARY_FAULT.none;
  let firstFaultRecord = SPARSE_CM12_FACTORED_AEI_INVALID;
  const fail = (code: number, record: number) => {
    if (fault === SPARSE_CM12_INTERNED_BOUNDARY_FAULT.none) {
      fault = code; firstFaultRecord = record;
    }
  };
  const immutableFault = validateImmutable(image);
  if (immutableFault !== null) {
    fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.immutableImage, immutableFault);
  }
  if (candidateGeneration !== acceptedGeneration + 1
    || words[slotAt + 2] !== acceptedGeneration) {
    fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.generation, 0);
  }
  if (actualChanged.size !== changed.size
    || [...actualChanged].some((leaf) => !changed.has(leaf))) {
    fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.deltaCoverage, 0);
  }
  let activeLeafCount = 0, referenceCount = 0, canonicalLeafCount = 0;
  let leafHash = 0x811c_9dc5, referenceHash = 0x811c_9dc5;
  for (let leaf = 0; leaf < layout.leafCapacity; leaf += 1) {
    const leafAt = layout.slotLeafBaseWords[shadowSlot]
      + leaf * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS;
    const isActive = (words[leafAt + 1]! & 1) !== 0;
    const descriptorId = targetDescriptors[leaf]!;
    const descriptor = catalog.canonical[descriptorId]!;
    if (isActive !== target.has(leaf) || words[leafAt + 2] !== descriptorId
      || words[leafAt + 3] !== descriptor.cellFirst
      || words[leafAt + 4] !== packDimensions(descriptor.validDimensions)) {
      fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.topologyMismatch, leaf);
    }
    if (closure.has(leaf) && words[leafAt] !== candidateGeneration) {
      fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.generation, leaf);
    }
    if (isActive) {
      activeLeafCount += 1;
      leafHash = fnv(fnv(leafHash, leaf), descriptorId);
      if (!descriptor.certified) {
        fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.canonicalCertificate, leaf);
      } else canonicalLeafCount += 1;
    }
    const expectedFaces = selectedReferences(image, leaf, target, targetDescriptors);
    let packedCounts = 0;
    for (let side = 0; side < 6; side += 1) {
      const expected = expectedFaces[side]!;
      packedCounts |= expected.length << (3 * side);
      for (let local = 0; local < 4; local += 1) {
        const refAt = refWord(image, shadowSlot, leaf, side, local);
        const patch = expected[local];
        const wanted = patch ? [compilation.templateIdByPatch[patch.id]!,
          patch.targetLeaf, compilation.rowBaseByPatch[patch.id]!]
          : [SPARSE_CM12_FACTORED_AEI_INVALID, SPARSE_CM12_FACTORED_AEI_INVALID,
            SPARSE_CM12_FACTORED_AEI_INVALID];
        const [template, targetLeaf] = unpackSparseCM12InternedBoundaryRefIdentity(
          words[refAt]!);
        if (template !== wanted[0] || targetLeaf !== wanted[1]
          || words[refAt + 1] !== wanted[2]) {
          fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.reference,
            leaf * SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF + side * 4 + local);
        }
        if (patch) {
          referenceCount += 1;
          referenceHash = fnv(fnv(fnv(fnv(fnv(referenceHash,
            leaf), side * 4 + local), wanted[0]!), wanted[1]!), wanted[2]!);
        }
      }
    }
    if (words[leafAt + 5] !== packedCounts) {
      fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.reference, leaf);
    }
  }
  const selectorUnchanged = words[2] === acceptedSlot
    && words[3] === acceptedGeneration;
  if (!selectorUnchanged) {
    fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.selectorChanged, 2);
  }
  if (words[slotAt + 9] !== changed.size
    || words[slotAt + 10] !== expectedDeltaHash) {
    fail(SPARSE_CM12_INTERNED_BOUNDARY_FAULT.deltaCoverage, 9);
  }
  words[slotAt + 5] = fault;
  words[slotAt + 6] = firstFaultRecord;
  words[slotAt + 1] = fault === 0
    ? SPARSE_CM12_INTERNED_BOUNDARY_SLOT_STATE.ready
    : SPARSE_CM12_INTERNED_BOUNDARY_SLOT_STATE.fault;
  return Object.freeze({ passed: fault === 0, acceptedSlot, shadowSlot,
    acceptedGeneration, candidateGeneration, fault, firstFaultRecord,
    activeLeafCount, referenceCount, canonicalLeafCount,
    changedLeafCount: changed.size, deltaClosureCount: closure.size,
    leafHash, referenceHash, selectorUnchanged });
}

/** Publish one selector flip, then replay the fixed delta closure into the retired slot. */
export function commitSparseCM12InternedBoundaryShadow(options: Readonly<{
  image: SparseCM12InternedBoundaryImage;
  receipt: SparseCM12InternedBoundaryPreflipReceipt;
  targetActiveLeaves: Iterable<number>;
  targetDescriptorIdByLeaf: readonly number[];
  changedLeaves: Iterable<number>;
}>): void {
  const { image, receipt } = options;
  if (!receipt.passed || image.words[2] !== receipt.acceptedSlot
    || image.words[3] !== receipt.acceptedGeneration) {
    throw new Error("IBO cannot commit an invalid or stale receipt");
  }
  const target = activeSet(options.targetActiveLeaves,
    image.layout.leafCapacity);
  const targetDescriptors = descriptorMap(image, options.targetDescriptorIdByLeaf);
  const changed = activeSet(options.changedLeaves,
    image.layout.leafCapacity);
  const closure = deltaClosure(image, changed);
  image.words[3] = receipt.candidateGeneration;
  image.words[2] = receipt.shadowSlot;
  for (const leaf of closure) {
    writeLeaf(image, receipt.acceptedSlot, leaf, receipt.candidateGeneration, target,
      targetDescriptors);
  }
  summarizeSlot(image, receipt.acceptedSlot, receipt.candidateGeneration,
    receipt.candidateGeneration, SPARSE_CM12_INTERNED_BOUNDARY_SLOT_STATE.mirror);
}
