import type {
  SparseAtlasCompositeCell,
  SparseAtlasCompositeGrid,
  SparseAtlasGradientRow,
} from "./sparse-atlas-composite-projection";
import { sparseBrickSpan } from "./sparse-brick-atlas";

/** Standalone topology-only ABI for the factored shared accepted execution image. */
export const SPARSE_CM12_FACTORED_AEI_MAGIC = 0x4145_4933; // AEI3
export const SPARSE_CM12_FACTORED_AEI_VERSION = 1;
export const SPARSE_CM12_FACTORED_AEI_HEADER_WORDS = 32;
export const SPARSE_CM12_FACTORED_AEI_CANONICAL_WORDS = 16;
export const SPARSE_CM12_FACTORED_AEI_PATCH_WORDS = 14;
export const SPARSE_CM12_FACTORED_AEI_SLOT_HEADER_WORDS = 32;
export const SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS = 8;
export const SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS = 2;
export const SPARSE_CM12_FACTORED_AEI_PATCHES_PER_FACE = 4;
export const SPARSE_CM12_FACTORED_AEI_PATCHES_PER_LEAF = 24;
export const SPARSE_CM12_FACTORED_AEI_INVALID = 0xffff_ffff;

export const SPARSE_CM12_FACTORED_AEI_RELATION = Object.freeze({
  equalRungCanonical: 1,
  explicitMixed: 2,
  explicitSparseAir: 3,
  explicitOther: 4,
} as const);

export const SPARSE_CM12_FACTORED_AEI_SLOT_STATE = Object.freeze({
  mirror: 1,
  building: 2,
  ready: 3,
  fault: 4,
} as const);

export const SPARSE_CM12_FACTORED_AEI_FAULT = Object.freeze({
  none: 0,
  generation: 1,
  deltaCoverage: 2,
  canonicalCertificate: 3,
  patchCapacity: 4,
  patchReference: 5,
  topologyMismatch: 6,
  selectorChanged: 7,
} as const);

export interface SparseCM12FactoredAEILayout {
  readonly leafCapacity: number;
  readonly levelCount: number;
  readonly canonicalCapacity: number;
  readonly patchCapacity: number;
  readonly exceptionRowCapacity: number;
  readonly canonicalBaseWords: number;
  readonly patchBaseWords: number;
  readonly exceptionRowBaseWords: number;
  readonly slotBaseWords: readonly [number, number];
  readonly slotStrideWords: number;
  readonly slotLeafBaseWords: readonly [number, number];
  readonly slotPatchRefBaseWords: readonly [number, number];
  readonly totalWords: number;
  readonly totalBytes: number;
  readonly immutableBytes: number;
  readonly bytesPerSlot: number;
}

export interface SparseCM12FactoredAEICanonicalDescriptor {
  readonly id: number;
  readonly leafId: number;
  readonly resolution: number;
  readonly cellFirst: number;
  readonly validDimensions: readonly [number, number, number];
  readonly scaleLog2: number;
  readonly rowBase: readonly [number, number, number];
  readonly rowCount: readonly [number, number, number];
  readonly canonicalRowCount: number;
  readonly rowIdHash: number;
  readonly termHash: number;
  readonly geometryHash: number;
  readonly certified: boolean;
  readonly firstFailure?: string;
}

export interface SparseCM12FactoredAEIPatchDescriptor {
  readonly id: number;
  readonly sourceLeaf: number;
  readonly targetLeaf: number;
  readonly sourceSide: number;
  readonly relation: number;
  readonly rowFirst: number;
  readonly rowCount: number;
  readonly exceptionFirst: number;
  readonly exceptionCount: number;
  readonly rowHash: number;
  readonly termHash: number;
  readonly sourceCanonicalId: number;
  readonly targetCanonicalId: number;
  /** Certified source/target tangential rectangles for equal-rung arithmetic. */
  readonly sourceFaceOrigin: readonly [number, number];
  readonly targetFaceOrigin: readonly [number, number];
  readonly faceDimensions: readonly [number, number];
  readonly mappingCertified: boolean;
}

export interface SparseCM12FactoredAEICatalog {
  readonly layout: SparseCM12FactoredAEILayout;
  readonly words: Uint32Array;
  readonly canonical: readonly SparseCM12FactoredAEICanonicalDescriptor[];
  readonly patches: readonly SparseCM12FactoredAEIPatchDescriptor[];
  readonly exceptionRows: Uint32Array;
  readonly patchIdsByLeafSide: readonly (readonly (readonly number[])[])[];
  /** All immutable rung variants, indexed by canonical descriptor then side. */
  readonly patchIdsByCanonicalSide: readonly (readonly (readonly number[])[])[];
  readonly neighborLeavesByLeaf: readonly (readonly number[])[];
  readonly leafIdByBrickKey: ReadonlyMap<number, number>;
  readonly brickKeyByLeafId: readonly number[];
  readonly descriptorIdByLeaf: readonly number[];
}

/**
 * Stable-leaf face adjacency compiled independently of SCMT/operator rows.
 * Logical-brick occupancy is expanded only at construction; runtime consumes
 * the bounded per-leaf lists and remains O(changed-surface).
 */
export function compileSparseCM12StableLeafFaceNeighbors(options: Readonly<{
  coordinates: readonly (readonly [number, number, number])[];
  spans: readonly number[];
}>): readonly (readonly number[])[] {
  if (options.coordinates.length !== options.spans.length) {
    throw new Error("AEI stable geometry coordinate/span capacities differ");
  }
  const owner = new Map<string, number>();
  const key = (x: number, y: number, z: number) => `${x}/${y}/${z}`;
  for (let leaf = 0; leaf < options.coordinates.length; leaf += 1) {
    const origin = options.coordinates[leaf]!, span = options.spans[leaf]!;
    if (!Number.isSafeInteger(span) || span < 1
      || origin.some((value) => !Number.isSafeInteger(value))) {
      throw new Error(`AEI stable geometry leaf ${leaf} is invalid`);
    }
    for (let z = 0; z < span; z += 1) for (let y = 0; y < span; y += 1) {
      for (let x = 0; x < span; x += 1) {
        const address = key(origin[0] + x, origin[1] + y, origin[2] + z);
        if (owner.has(address)) throw new Error("AEI stable geometry leaves overlap");
        owner.set(address, leaf);
      }
    }
  }
  const result = options.coordinates.map(() => new Set<number>());
  for (let leaf = 0; leaf < options.coordinates.length; leaf += 1) {
    const origin = options.coordinates[leaf]!, span = options.spans[leaf]!;
    for (let axis = 0; axis < 3; axis += 1) for (const sign of [-1, 1]) {
      const tangents = [0, 1, 2].filter((value) => value !== axis);
      for (let v = 0; v < span; v += 1) for (let u = 0; u < span; u += 1) {
        const coordinate = [...origin] as [number, number, number];
        coordinate[axis] += sign < 0 ? -1 : span;
        coordinate[tangents[0]!] += u; coordinate[tangents[1]!] += v;
        const neighbor = owner.get(key(coordinate[0], coordinate[1], coordinate[2]));
        if (neighbor !== undefined && neighbor !== leaf) result[leaf]!.add(neighbor);
      }
    }
  }
  return Object.freeze(result.map((neighbors) =>
    Object.freeze([...neighbors].sort((a, b) => a - b))));
}

export interface SparseCM12FactoredAEIImage {
  readonly catalog: SparseCM12FactoredAEICatalog;
  readonly words: Uint32Array;
}

export interface SparseCM12FactoredAEIPreflipReceipt {
  readonly passed: boolean;
  readonly acceptedSlot: 0 | 1;
  readonly shadowSlot: 0 | 1;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly fault: number;
  readonly firstFaultRecord: number;
  readonly activeLeafCount: number;
  readonly patchReferenceCount: number;
  readonly canonicalLeafCount: number;
  readonly deltaLeafCount: number;
  readonly deltaClosureCount: number;
  readonly leafHash: number;
  readonly patchHash: number;
  readonly selectorUnchanged: boolean;
}

const align = (value: number, words = 64): number =>
  Math.ceil(value / words) * words;

const u32 = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be a u32`);
  }
  return value >>> 0;
};

const fnv = (hash: number, value: number): number => {
  let result = (hash ^ (value >>> 0)) >>> 0;
  return Math.imul(result, 0x0100_0193) >>> 0;
};

const F32_BUFFER = new ArrayBuffer(4);
const F32_FLOAT = new Float32Array(F32_BUFFER);
const F32_WORD = new Uint32Array(F32_BUFFER);
const f32Bits = (value: number): number => {
  F32_FLOAT[0] = value;
  return F32_WORD[0]!;
};

const packDimensions = (value: readonly number[]): number =>
  (value[0]! | (value[1]! << 10) | (value[2]! << 20)) >>> 0;

const packCounts = (value: readonly number[]): number =>
  (value[0]! | (value[1]! << 10) | (value[2]! << 20)) >>> 0;

const rowCounts = (dimensions: readonly number[]): readonly [number, number, number] => [
  Math.max(0, dimensions[0]! - 1) * dimensions[1]! * dimensions[2]!,
  dimensions[0]! * Math.max(0, dimensions[1]! - 1) * dimensions[2]!,
  dimensions[0]! * dimensions[1]! * Math.max(0, dimensions[2]! - 1),
];

const canonicalOrdinal = (
  axis: number,
  local: readonly number[],
  dimensions: readonly number[],
): number => axis === 0
  ? local[0]! - 1 + (dimensions[0]! - 1)
    * (local[1]! + dimensions[1]! * local[2]!)
  : axis === 1
    ? local[0]! + dimensions[0]!
      * (local[1]! - 1 + (dimensions[1]! - 1) * local[2]!)
    : local[0]! + dimensions[0]!
      * (local[1]! + dimensions[1]! * (local[2]! - 1));

const patchRefIndex = (leaf: number, side: number, local: number): number =>
  leaf * SPARSE_CM12_FACTORED_AEI_PATCHES_PER_LEAF
    + side * SPARSE_CM12_FACTORED_AEI_PATCHES_PER_FACE + local;

export function createSparseCM12FactoredAEILayout(options: Readonly<{
  leafCapacity: number;
  levelCount: number;
  patchCapacity: number;
  exceptionRowCapacity: number;
}>): SparseCM12FactoredAEILayout {
  const leafCapacity = u32(options.leafCapacity, "AEI leaf capacity");
  const levelCount = u32(options.levelCount, "AEI level count");
  const patchCapacity = u32(options.patchCapacity, "AEI patch capacity");
  const exceptionRowCapacity = u32(options.exceptionRowCapacity,
    "AEI exception row capacity");
  const canonicalCapacity = leafCapacity * levelCount;
  const canonicalBaseWords = SPARSE_CM12_FACTORED_AEI_HEADER_WORDS;
  const patchBaseWords = align(canonicalBaseWords
    + canonicalCapacity * SPARSE_CM12_FACTORED_AEI_CANONICAL_WORDS);
  const exceptionRowBaseWords = align(patchBaseWords
    + patchCapacity * SPARSE_CM12_FACTORED_AEI_PATCH_WORDS);
  const immutableEnd = align(exceptionRowBaseWords + exceptionRowCapacity);
  const slotStrideWords = align(SPARSE_CM12_FACTORED_AEI_SLOT_HEADER_WORDS
    + leafCapacity * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS
    + leafCapacity * SPARSE_CM12_FACTORED_AEI_PATCHES_PER_LEAF
      * SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS);
  const slot0 = immutableEnd, slot1 = slot0 + slotStrideWords;
  const slotLeaf0 = slot0 + SPARSE_CM12_FACTORED_AEI_SLOT_HEADER_WORDS;
  const slotLeaf1 = slot1 + SPARSE_CM12_FACTORED_AEI_SLOT_HEADER_WORDS;
  const slotPatch0 = slotLeaf0
    + leafCapacity * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
  const slotPatch1 = slotLeaf1
    + leafCapacity * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
  const totalWords = slot1 + slotStrideWords;
  return Object.freeze({ leafCapacity, levelCount, canonicalCapacity,
    patchCapacity, exceptionRowCapacity, canonicalBaseWords, patchBaseWords,
    exceptionRowBaseWords, slotBaseWords: [slot0, slot1] as const, slotStrideWords,
    slotLeafBaseWords: [slotLeaf0, slotLeaf1] as const,
    slotPatchRefBaseWords: [slotPatch0, slotPatch1] as const, totalWords,
    totalBytes: 4 * totalWords, immutableBytes: 4 * immutableEnd,
    bytesPerSlot: 4 * slotStrideWords });
}

const leafDimensions = (
  cells: readonly SparseAtlasCompositeCell[],
): readonly [number, number, number] => {
  let x = 0, y = 0, z = 0;
  for (const cell of cells) {
    x = Math.max(x, cell.local[0]! + 1);
    y = Math.max(y, cell.local[1]! + 1);
    z = Math.max(z, cell.local[2]! + 1);
  }
  return [x, y, z];
};

function certifyCanonicalLeaf(
  grid: SparseAtlasCompositeGrid,
  cells: readonly SparseAtlasCompositeCell[],
  intra: readonly (readonly SparseAtlasGradientRow[])[],
  leafId: number,
  levelIndex: number,
  levelCount: number,
): SparseCM12FactoredAEICanonicalDescriptor {
  const resolution = cells[0]?.brickResolution ?? 0;
  const id = leafId * levelCount + levelIndex;
  const dimensions = leafDimensions(cells);
  const counts = rowCounts(dimensions);
  const cellFirst = cells[0]?.id ?? SPARSE_CM12_FACTORED_AEI_INVALID;
  let failure: string | undefined;
  if (cells.length !== dimensions[0] * dimensions[1] * dimensions[2]) {
    failure = "cell range is not a dense clipped brick";
  }
  const bases = intra.map((rows) => rows[0]?.id ?? SPARSE_CM12_FACTORED_AEI_INVALID);
  let rowIdHash = 0x811c_9dc5, termHash = 0x811c_9dc5, geometryHash = 0x811c_9dc5;
  for (const axis of [0, 1, 2] as const) {
    const rows = intra[axis]!;
    if (!failure && rows.length !== counts[axis]) {
      failure = `axis ${axis} canonical row count differs`;
    }
    for (const row of rows) {
      const positive = row.terms.find((term) => term.coefficient > 0);
      const negative = row.terms.find((term) => term.coefficient < 0);
      if (!positive || !negative || row.terms.length !== 2
        || row.terms[0] !== negative || row.terms[1] !== positive) {
        failure ??= `row ${row.id} term order is not negative,positive`;
        continue;
      }
      const positiveCell = grid.cells[positive.cellId]!;
      const expectedOrdinal = canonicalOrdinal(axis, positiveCell.local, dimensions);
      const expectedRow = bases[axis] === SPARSE_CM12_FACTORED_AEI_INVALID
        ? SPARSE_CM12_FACTORED_AEI_INVALID : bases[axis]! + expectedOrdinal;
      const stride = axis === 0 ? 1 : axis === 1 ? dimensions[0]!
        : dimensions[0]! * dimensions[1]!;
      if (row.id !== expectedRow || negative.cellId !== positive.cellId - stride) {
        failure ??= `row ${row.id} disagrees with the canonical ID/cell formula`;
      }
      rowIdHash = fnv(rowIdHash, row.id);
      termHash = fnv(fnv(fnv(fnv(termHash, negative.cellId),
        f32Bits(negative.coefficient)), positive.cellId), f32Bits(positive.coefficient));
      geometryHash = fnv(fnv(fnv(fnv(geometryHash, row.axis),
        f32Bits(row.area)), f32Bits(row.distance)), f32Bits(row.dualWeight));
    }
  }
  const width = cells[0]?.widthsFine[0] ?? 1;
  const scaleLog2 = Number.isInteger(Math.log2(width)) ? Math.log2(width) : 31;
  return Object.freeze({ id, leafId, resolution, cellFirst,
    validDimensions: dimensions, scaleLog2, rowBase: bases as [number, number, number],
    rowCount: counts, canonicalRowCount: counts[0] + counts[1] + counts[2],
    rowIdHash, termHash, geometryHash, certified: failure === undefined,
    ...(failure ? { firstFailure: failure } : {}) });
}

const sideFor = (row: SparseAtlasGradientRow, sourceKey: number): number => {
  if (row.negativeBrickKey === sourceKey) return 2 * row.axis + 1;
  if (row.positiveBrickKey === sourceKey) return 2 * row.axis;
  return -1;
};

const hashRows = (rows: readonly SparseAtlasGradientRow[]): readonly [number, number] => {
  let rowHash = 0x811c_9dc5, termHash = 0x811c_9dc5;
  for (const row of rows) {
    rowHash = fnv(rowHash, row.id);
    for (const term of row.terms) {
      termHash = fnv(fnv(termHash, term.cellId), f32Bits(term.coefficient));
    }
  }
  return [rowHash, termHash];
};

const equalFaceMapping = (options: Readonly<{
  rows: readonly SparseAtlasGradientRow[];
  sourceKey: number;
  targetKey: number;
  sourceSide: number;
  grid: SparseAtlasCompositeGrid;
}>): Readonly<{
  sourceOrigin: readonly [number, number];
  targetOrigin: readonly [number, number];
  dimensions: readonly [number, number];
  certified: boolean;
}> => {
  const axis = Math.floor(options.sourceSide / 2);
  const tangents = ([0, 1, 2] as const).filter((value) => value !== axis);
  const pairs = options.rows.map((row) => {
    const source = row.terms.filter((term) =>
      options.grid.cells[term.cellId]!.brickKey === options.sourceKey);
    const target = row.terms.filter((term) =>
      options.grid.cells[term.cellId]!.brickKey === options.targetKey);
    return { source: source.length === 1 ? options.grid.cells[source[0]!.cellId] : undefined,
      target: target.length === 1 ? options.grid.cells[target[0]!.cellId] : undefined };
  });
  if (pairs.some((pair) => !pair.source || !pair.target) || pairs.length === 0) {
    return { sourceOrigin: [0, 0], targetOrigin: [0, 0], dimensions: [0, 0],
      certified: false };
  }
  const sourceU = pairs.map((pair) => pair.source!.local[tangents[0]!]!);
  const sourceV = pairs.map((pair) => pair.source!.local[tangents[1]!]!);
  const targetU = pairs.map((pair) => pair.target!.local[tangents[0]!]!);
  const targetV = pairs.map((pair) => pair.target!.local[tangents[1]!]!);
  const sourceOrigin = [Math.min(...sourceU), Math.min(...sourceV)] as const;
  const targetOrigin = [Math.min(...targetU), Math.min(...targetV)] as const;
  const dimensions = [Math.max(...sourceU) - sourceOrigin[0] + 1,
    Math.max(...sourceV) - sourceOrigin[1] + 1] as const;
  let certified = dimensions[0] * dimensions[1] === pairs.length;
  for (let index = 0; index < pairs.length && certified; index += 1) {
    const u = index % dimensions[0], v = Math.floor(index / dimensions[0]);
    const source = pairs[index]!.source!, target = pairs[index]!.target!;
    certified = source.local[tangents[0]!] === sourceOrigin[0] + u
      && source.local[tangents[1]!] === sourceOrigin[1] + v
      && target.local[tangents[0]!] === targetOrigin[0] + u
      && target.local[tangents[1]!] === targetOrigin[1] + v
      && source.local[axis] === (options.sourceSide % 2 === 1
        ? source.brickResolution - 1 : 0)
      && target.local[axis] === (options.sourceSide % 2 === 1
        ? 0 : target.brickResolution - 1);
  }
  return { sourceOrigin, targetOrigin, dimensions, certified };
};

/** Serialize descriptors compiled from an authoritative external topology catalog. */
export function createSparseCM12FactoredAEICatalogFromAuthority(options: Readonly<{
  leafCapacity: number;
  levelCount: number;
  canonical: readonly SparseCM12FactoredAEICanonicalDescriptor[];
  patches: readonly SparseCM12FactoredAEIPatchDescriptor[];
  exceptionRows: ArrayLike<number>;
  brickKeyByLeafId: readonly number[];
  descriptorIdByLeaf: readonly number[];
  /** Stable-domain geometry closure, independent of accepted SCMT patch rows. */
  neighborLeavesByLeaf?: readonly (readonly number[])[];
}>): SparseCM12FactoredAEICatalog {
  const canonicalCapacity = options.leafCapacity * options.levelCount;
  if (options.canonical.length !== canonicalCapacity
    || options.brickKeyByLeafId.length !== options.leafCapacity
    || options.descriptorIdByLeaf.length !== options.leafCapacity) {
    throw new Error("AEI authoritative catalog capacity differs from its descriptors");
  }
  options.canonical.forEach((descriptor, id) => {
    if (descriptor.id !== id || descriptor.leafId !== Math.floor(id / options.levelCount)) {
      throw new Error(`AEI canonical descriptor ${id} has unstable identity`);
    }
  });
  options.patches.forEach((facePatch, id) => {
    if (facePatch.id !== id || facePatch.sourceCanonicalId >= canonicalCapacity) {
      throw new Error(`AEI patch descriptor ${id} has unstable identity`);
    }
  });
  const exceptionRows = Uint32Array.from(options.exceptionRows);
  const layout = createSparseCM12FactoredAEILayout({ leafCapacity: options.leafCapacity,
    levelCount: options.levelCount, patchCapacity: options.patches.length,
    exceptionRowCapacity: exceptionRows.length });
  const words = new Uint32Array(layout.totalWords);
  words.set([SPARSE_CM12_FACTORED_AEI_MAGIC, SPARSE_CM12_FACTORED_AEI_VERSION,
    SPARSE_CM12_FACTORED_AEI_HEADER_WORDS, 0, 1, layout.leafCapacity,
    layout.levelCount, layout.canonicalBaseWords, layout.patchBaseWords,
    layout.exceptionRowBaseWords, layout.exceptionRowCapacity,
    layout.slotBaseWords[0], layout.slotBaseWords[1], layout.slotStrideWords,
    layout.totalWords, SPARSE_CM12_FACTORED_AEI_CANONICAL_WORDS,
    SPARSE_CM12_FACTORED_AEI_PATCH_WORDS, SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS,
    SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS,
    SPARSE_CM12_FACTORED_AEI_PATCHES_PER_FACE], 0);
  for (const descriptor of options.canonical) {
    const at = layout.canonicalBaseWords
      + descriptor.id * SPARSE_CM12_FACTORED_AEI_CANONICAL_WORDS;
    words.set([descriptor.certified ? 1 : 0, descriptor.leafId,
      descriptor.resolution, descriptor.cellFirst,
      packDimensions(descriptor.validDimensions), descriptor.scaleLog2,
      descriptor.rowBase[0], descriptor.rowBase[1], descriptor.rowBase[2],
      descriptor.rowCount[0], descriptor.rowCount[1], descriptor.rowCount[2],
      descriptor.canonicalRowCount, descriptor.rowIdHash,
      descriptor.termHash, descriptor.geometryHash], at);
  }
  const patchIdsByCanonicalSide: number[][][] = Array.from(
    { length: canonicalCapacity }, () => Array.from({ length: 6 }, () => []));
  const patchIdsByLeafSide: number[][][] = Array.from(
    { length: options.leafCapacity }, () => Array.from({ length: 6 }, () => []));
  const neighborSets = Array.from({ length: options.leafCapacity }, () => new Set<number>());
  for (const facePatch of options.patches) {
    const at = layout.patchBaseWords
      + facePatch.id * SPARSE_CM12_FACTORED_AEI_PATCH_WORDS;
    words.set([facePatch.sourceLeaf, facePatch.targetLeaf, facePatch.sourceSide,
      facePatch.relation, facePatch.rowFirst, facePatch.rowCount,
      facePatch.exceptionFirst, facePatch.exceptionCount, facePatch.rowHash,
      facePatch.termHash, facePatch.sourceCanonicalId, facePatch.targetCanonicalId,
      facePatch.sourceFaceOrigin[0] | (facePatch.sourceFaceOrigin[1] << 5)
        | (facePatch.faceDimensions[0] << 10) | (facePatch.faceDimensions[1] << 15),
      facePatch.targetFaceOrigin[0] | (facePatch.targetFaceOrigin[1] << 5)
        | (facePatch.mappingCertified ? 0x8000_0000 : 0)], at);
    patchIdsByCanonicalSide[facePatch.sourceCanonicalId]![facePatch.sourceSide]!
      .push(facePatch.id);
    if (options.descriptorIdByLeaf[facePatch.sourceLeaf]
      === facePatch.sourceCanonicalId) {
      patchIdsByLeafSide[facePatch.sourceLeaf]![facePatch.sourceSide]!.push(facePatch.id);
    }
    if (facePatch.targetLeaf !== SPARSE_CM12_FACTORED_AEI_INVALID) {
      neighborSets[facePatch.sourceLeaf]!.add(facePatch.targetLeaf);
    }
  }
  if (options.neighborLeavesByLeaf) {
    if (options.neighborLeavesByLeaf.length !== options.leafCapacity) {
      throw new Error("AEI stable geometry closure differs from leaf capacity");
    }
    options.neighborLeavesByLeaf.forEach((neighbors, leaf) => {
      let prior = -1;
      for (const neighbor of neighbors) {
        if (!Number.isSafeInteger(neighbor) || neighbor < 0
          || neighbor >= options.leafCapacity || neighbor === leaf || neighbor <= prior) {
          throw new Error(`AEI stable geometry closure ${leaf} is not sorted unique`);
        }
        prior = neighbor;
      }
    });
    options.neighborLeavesByLeaf.forEach((neighbors, leaf) => {
      for (const neighbor of neighbors) if (!options.neighborLeavesByLeaf![neighbor]!
        .includes(leaf)) throw new Error(`AEI stable geometry closure ${leaf}/${neighbor} is not reciprocal`);
    });
    for (const patch of options.patches) if (patch.targetLeaf
      !== SPARSE_CM12_FACTORED_AEI_INVALID
      && !options.neighborLeavesByLeaf[patch.sourceLeaf]!.includes(patch.targetLeaf)) {
      throw new Error(`AEI patch ${patch.id} is outside stable geometry closure`);
    }
  }
  words.set(exceptionRows, layout.exceptionRowBaseWords);
  const leafIdByBrickKey = new Map(options.brickKeyByLeafId.map((key, leaf) => [key, leaf]));
  return Object.freeze({ layout, words,
    canonical: Object.freeze([...options.canonical]),
    patches: Object.freeze([...options.patches]), exceptionRows,
    patchIdsByLeafSide, patchIdsByCanonicalSide,
    neighborLeavesByLeaf: options.neighborLeavesByLeaf
      ? Object.freeze(options.neighborLeavesByLeaf.map((neighbors) =>
        Object.freeze([...neighbors])))
      : neighborSets.map((set) => Object.freeze([...set].sort((a, b) => a - b))),
    leafIdByBrickKey,
    brickKeyByLeafId: Object.freeze([...options.brickKeyByLeafId]),
    descriptorIdByLeaf: Object.freeze([...options.descriptorIdByLeaf]) });
}

/** Build immutable certificates and face-patch metadata from authoritative IDs. */
export function compileSparseCM12FactoredAEICatalog(
  grid: SparseAtlasCompositeGrid,
): SparseCM12FactoredAEICatalog {
  // Atlas array order is the stable TEI leaf/packet authority. Keys are lookup
  // identities only; sorting here would silently change leaf*64 packet IDs.
  const bricks = [...grid.atlas.bricks];
  const leafIdByBrickKey = new Map(bricks.map((brick, index) => [brick.key, index]));
  const levels = Array.from({ length: Math.log2(grid.atlas.brickFineResolution) + 1 },
    (_, index) => 2 ** index);
  const descriptorIdByLeaf = bricks.map((brick, leaf) =>
    leaf * levels.length + levels.indexOf(brick.resolution));
  const cellsByBrick = new Map<number, SparseAtlasCompositeCell[]>();
  for (const cell of grid.cells) {
    const cells = cellsByBrick.get(cell.brickKey) ?? [];
    cells.push(cell); cellsByBrick.set(cell.brickKey, cells);
  }
  const intraByBrick = new Map<number, SparseAtlasGradientRow[][]>();
  for (const row of grid.gradientRows) if (row.kind === "intra-brick") {
    const brickKey = grid.cells[row.terms[0]!.cellId]!.brickKey;
    const axes = intraByBrick.get(brickKey) ?? [[], [], []];
    axes[row.axis]!.push(row); intraByBrick.set(brickKey, axes);
  }
  const canonical: SparseCM12FactoredAEICanonicalDescriptor[] = [];
  for (const [leaf, brick] of bricks.entries()) for (let level = 0;
    level < levels.length; level += 1) {
    canonical.push(levels[level] === brick.resolution
      ? certifyCanonicalLeaf(grid, cellsByBrick.get(brick.key) ?? [],
        intraByBrick.get(brick.key) ?? [[], [], []], leaf, level, levels.length)
      : Object.freeze({ id: leaf * levels.length + level, leafId: leaf,
        resolution: levels[level]!, cellFirst: SPARSE_CM12_FACTORED_AEI_INVALID,
        validDimensions: [0, 0, 0] as const, scaleLog2: 0,
        rowBase: [SPARSE_CM12_FACTORED_AEI_INVALID,
          SPARSE_CM12_FACTORED_AEI_INVALID,
          SPARSE_CM12_FACTORED_AEI_INVALID] as const,
        rowCount: [0, 0, 0] as const, canonicalRowCount: 0,
        rowIdHash: 0, termHash: 0, geometryHash: 0, certified: false,
        firstFailure: "rung is absent from this construction catalog" }));
  }

  const groups = new Map<string, SparseAtlasGradientRow[]>();
  for (const row of grid.gradientRows) {
    if (row.kind === "intra-brick") continue;
    const keys = new Set<number>();
    if (row.negativeBrickKey !== undefined) keys.add(row.negativeBrickKey);
    if (row.positiveBrickKey !== undefined) keys.add(row.positiveBrickKey);
    for (const sourceKey of keys) {
      const sourceLeaf = leafIdByBrickKey.get(sourceKey);
      if (sourceLeaf === undefined) continue;
      const side = sideFor(row, sourceKey);
      const targetKey = row.negativeBrickKey === sourceKey
        ? row.positiveBrickKey : row.negativeBrickKey;
      const targetLeaf = targetKey === undefined
        ? SPARSE_CM12_FACTORED_AEI_INVALID : leafIdByBrickKey.get(targetKey)
          ?? SPARSE_CM12_FACTORED_AEI_INVALID;
      const key = `${sourceLeaf}/${side}/${targetLeaf}`;
      const list = groups.get(key) ?? [];
      list.push(row); groups.set(key, list);
    }
  }
  const patches: SparseCM12FactoredAEIPatchDescriptor[] = [];
  const exceptionRows: number[] = [];
  for (const [key, rawRows] of groups) {
    const [sourceLeaf, sourceSide, targetLeaf] = key.split("/").map(Number);
    const rows = [...rawRows].sort((a, b) => a.id - b.id);
    const allEqual = rows.every((row) => row.kind === "brick-face"
      && row.terms.length === 2);
    const contiguous = rows.every((row, index) => row.id === rows[0]!.id + index);
    const sourceKey = bricks[sourceLeaf]!.key;
    const targetKey = targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID
      ? undefined : bricks[targetLeaf]!.key;
    const mapping = allEqual && contiguous && targetKey !== undefined
      ? equalFaceMapping({ rows, sourceKey, targetKey, sourceSide, grid })
      : { sourceOrigin: [0, 0] as const, targetOrigin: [0, 0] as const,
        dimensions: [0, 0] as const, certified: false };
    const relation = allEqual && contiguous && mapping.certified
      ? SPARSE_CM12_FACTORED_AEI_RELATION.equalRungCanonical
      : rows.every((row) => row.kind === "mixed-seam")
        ? SPARSE_CM12_FACTORED_AEI_RELATION.explicitMixed
        : rows.every((row) => row.kind === "sparse-air")
          ? SPARSE_CM12_FACTORED_AEI_RELATION.explicitSparseAir
          : SPARSE_CM12_FACTORED_AEI_RELATION.explicitOther;
    const exceptionFirst = exceptionRows.length;
    if (relation !== SPARSE_CM12_FACTORED_AEI_RELATION.equalRungCanonical) {
      exceptionRows.push(...rows.map((row) => row.id));
    }
    const [rowHash, termHash] = hashRows(rows);
    const id = patches.length;
    const patch = Object.freeze({ id, sourceLeaf, targetLeaf, sourceSide, relation,
      rowFirst: contiguous ? rows[0]!.id : SPARSE_CM12_FACTORED_AEI_INVALID,
      rowCount: rows.length, exceptionFirst,
      exceptionCount: relation === SPARSE_CM12_FACTORED_AEI_RELATION.equalRungCanonical
        ? 0 : rows.length, rowHash, termHash,
      sourceCanonicalId: descriptorIdByLeaf[sourceLeaf]!,
      targetCanonicalId: targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID
        ? SPARSE_CM12_FACTORED_AEI_INVALID : descriptorIdByLeaf[targetLeaf]!,
      sourceFaceOrigin: mapping.sourceOrigin,
      targetFaceOrigin: mapping.targetOrigin,
      faceDimensions: mapping.dimensions,
      mappingCertified: mapping.certified,
    });
    patches.push(patch);
  }
  return createSparseCM12FactoredAEICatalogFromAuthority({
    leafCapacity: bricks.length, levelCount: levels.length, canonical, patches,
    exceptionRows, brickKeyByLeafId: bricks.map((brick) => brick.key),
    descriptorIdByLeaf,
    neighborLeavesByLeaf: compileSparseCM12StableLeafFaceNeighbors({
      coordinates: bricks.map((brick) => brick.coordinate),
      spans: bricks.map((brick) => sparseBrickSpan(brick)),
    }) });
}

const activeSet = (values: Iterable<number>, capacity: number): Set<number> => {
  const result = new Set<number>();
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0 || value >= capacity) {
      throw new RangeError(`AEI active leaf ${value} is outside capacity`);
    }
    result.add(value);
  }
  return result;
};

const slotActiveLeaves = (
  image: SparseCM12FactoredAEIImage,
  slot: 0 | 1,
): Set<number> => {
  const result = new Set<number>();
  const { layout } = image.catalog;
  for (let leaf = 0; leaf < layout.leafCapacity; leaf += 1) {
    const at = layout.slotLeafBaseWords[slot]
      + leaf * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
    if ((image.words[at + 1]! & 1) !== 0) result.add(leaf);
  }
  return result;
};

function writeLeafAndPatches(
  image: SparseCM12FactoredAEIImage,
  slot: 0 | 1,
  leaf: number,
  generation: number,
  active: ReadonlySet<number>,
): void {
  const { catalog, words } = image;
  const descriptor = catalog.canonical[catalog.descriptorIdByLeaf[leaf]!]!;
  const leafAt = catalog.layout.slotLeafBaseWords[slot]
    + leaf * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
  const faceCounts = Array.from({ length: 6 }, () => 0);
  let patchHash = 0x811c_9dc5;
  for (let side = 0; side < 6; side += 1) {
    const candidates = catalog.patchIdsByCanonicalSide[descriptor.id]![side]!
      .filter((patchId) => {
      const patch = catalog.patches[patchId]!;
      return active.has(leaf) && (patch.targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID
        || active.has(patch.targetLeaf)
          && catalog.descriptorIdByLeaf[patch.targetLeaf] === patch.targetCanonicalId);
    });
    if (candidates.length > SPARSE_CM12_FACTORED_AEI_PATCHES_PER_FACE) {
      throw new Error(`AEI selected leaf ${leaf} side ${side} exceeds patch capacity`);
    }
    faceCounts[side] = candidates.length;
    for (let local = 0; local < SPARSE_CM12_FACTORED_AEI_PATCHES_PER_FACE; local += 1) {
      const ref = patchRefIndex(leaf, side, local);
      const at = catalog.layout.slotPatchRefBaseWords[slot]
        + ref * SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS;
      words[at] = generation;
      words[at + 1] = candidates[local] ?? SPARSE_CM12_FACTORED_AEI_INVALID;
      if (local < candidates.length) patchHash = fnv(patchHash, candidates[local]!);
    }
  }
  words.set([generation,
    (active.has(leaf) ? 1 : 0) | (descriptor.certified ? 2 : 0),
    descriptor.id, descriptor.cellFirst, packDimensions(descriptor.validDimensions),
    faceCounts.reduce((packed, count, side) => packed | (count << (3 * side)), 0),
    patchHash, descriptor.canonicalRowCount], leafAt);
}

function summarizeSlot(
  image: SparseCM12FactoredAEIImage,
  slot: 0 | 1,
  generation: number,
  baseGeneration: number,
  state: number,
  deltaLeafCount = 0,
  deltaHash = 0,
): void {
  const active = slotActiveLeaves(image, slot);
  let patchCount = 0, canonicalCount = 0;
  let leafHash = 0x811c_9dc5, patchHash = 0x811c_9dc5;
  for (const leaf of active) {
    const at = image.catalog.layout.slotLeafBaseWords[slot]
      + leaf * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
    leafHash = fnv(fnv(leafHash, leaf), image.words[at + 2]!);
    if ((image.words[at + 1]! & 2) !== 0) canonicalCount += 1;
  }
  const refs = image.catalog.layout.leafCapacity
    * SPARSE_CM12_FACTORED_AEI_PATCHES_PER_LEAF;
  for (let ref = 0; ref < refs; ref += 1) {
    const at = image.catalog.layout.slotPatchRefBaseWords[slot]
      + ref * SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS;
    const patch = image.words[at + 1]!;
    if (patch === SPARSE_CM12_FACTORED_AEI_INVALID) continue;
    patchCount += 1; patchHash = fnv(fnv(patchHash, ref), patch);
  }
  const at = image.catalog.layout.slotBaseWords[slot];
  image.words.set([generation, state, baseGeneration, active.size, patchCount,
    SPARSE_CM12_FACTORED_AEI_FAULT.none, SPARSE_CM12_FACTORED_AEI_INVALID,
    leafHash, patchHash, deltaLeafCount, deltaHash, canonicalCount], at);
}

export function createSparseCM12FactoredAEIImage(
  catalog: SparseCM12FactoredAEICatalog,
  activeLeaves: Iterable<number>,
  generation = 1,
): SparseCM12FactoredAEIImage {
  u32(generation, "AEI generation");
  const words = catalog.words.slice();
  const image: SparseCM12FactoredAEIImage = { catalog, words };
  const active = activeSet(activeLeaves, catalog.layout.leafCapacity);
  for (const slot of [0, 1] as const) {
    for (let leaf = 0; leaf < catalog.layout.leafCapacity; leaf += 1) {
      writeLeafAndPatches(image, slot, leaf, generation, active);
    }
    summarizeSlot(image, slot, generation, generation,
      SPARSE_CM12_FACTORED_AEI_SLOT_STATE.mirror);
  }
  words[3] = 0; words[4] = generation;
  return image;
}

const deltaClosure = (
  catalog: SparseCM12FactoredAEICatalog,
  changed: ReadonlySet<number>,
): Set<number> => {
  const result = new Set(changed);
  for (const leaf of changed) for (const neighbor of catalog.neighborLeavesByLeaf[leaf]!) {
    result.add(neighbor);
  }
  return result;
};

/** Patch only fixed local records in the retired mirror; selector remains unchanged. */
export function prepareSparseCM12FactoredAEIShadow(options: Readonly<{
  image: SparseCM12FactoredAEIImage;
  targetActiveLeaves: Iterable<number>;
  changedLeaves: Iterable<number>;
  candidateGeneration: number;
}>): Readonly<{ shadowSlot: 0 | 1; deltaClosure: readonly number[] }> {
  const { image } = options;
  const acceptedSlot = (image.words[3]! & 1) as 0 | 1;
  const shadowSlot = (1 - acceptedSlot) as 0 | 1;
  const acceptedGeneration = image.words[4]!;
  if (options.candidateGeneration !== acceptedGeneration + 1) {
    throw new Error("AEI candidate generation is not accepted+1");
  }
  const accepted = slotActiveLeaves(image, acceptedSlot);
  const target = activeSet(options.targetActiveLeaves, image.catalog.layout.leafCapacity);
  const changed = activeSet(options.changedLeaves, image.catalog.layout.leafCapacity);
  const actualChanged = new Set<number>();
  for (let leaf = 0; leaf < image.catalog.layout.leafCapacity; leaf += 1) {
    if (accepted.has(leaf) !== target.has(leaf)) actualChanged.add(leaf);
  }
  if (actualChanged.size !== changed.size
    || [...actualChanged].some((leaf) => !changed.has(leaf))) {
    throw new Error("AEI changed leaf list does not equal the accepted/target delta");
  }
  const closure = deltaClosure(image.catalog, changed);
  let deltaHash = 0x811c_9dc5;
  for (const leaf of [...closure].sort((a, b) => a - b)) {
    writeLeafAndPatches(image, shadowSlot, leaf, options.candidateGeneration, target);
    deltaHash = fnv(deltaHash, leaf);
  }
  summarizeSlot(image, shadowSlot, options.candidateGeneration,
    acceptedGeneration, SPARSE_CM12_FACTORED_AEI_SLOT_STATE.building,
    changed.size, deltaHash);
  return Object.freeze({ shadowSlot,
    deltaClosure: Object.freeze([...closure].sort((a, b) => a - b)) });
}

export function validateSparseCM12FactoredAEIPreflip(options: Readonly<{
  image: SparseCM12FactoredAEIImage;
  targetActiveLeaves: Iterable<number>;
  changedLeaves: Iterable<number>;
}>): SparseCM12FactoredAEIPreflipReceipt {
  const { image } = options;
  const acceptedSlot = (image.words[3]! & 1) as 0 | 1;
  const shadowSlot = (1 - acceptedSlot) as 0 | 1;
  const acceptedGeneration = image.words[4]!;
  const slotAt = image.catalog.layout.slotBaseWords[shadowSlot];
  const candidateGeneration = image.words[slotAt]!;
  const target = activeSet(options.targetActiveLeaves, image.catalog.layout.leafCapacity);
  const changed = activeSet(options.changedLeaves, image.catalog.layout.leafCapacity);
  const expectedClosure = deltaClosure(image.catalog, changed);
  const accepted = slotActiveLeaves(image, acceptedSlot);
  const actualChanged = new Set<number>();
  for (let leaf = 0; leaf < image.catalog.layout.leafCapacity; leaf += 1) {
    if (accepted.has(leaf) !== target.has(leaf)) actualChanged.add(leaf);
  }
  let expectedDeltaHash = 0x811c_9dc5;
  for (const leaf of [...expectedClosure].sort((a, b) => a - b)) {
    expectedDeltaHash = fnv(expectedDeltaHash, leaf);
  }
  let fault: number = SPARSE_CM12_FACTORED_AEI_FAULT.none;
  let firstFaultRecord = SPARSE_CM12_FACTORED_AEI_INVALID;
  const fail = (code: number, record: number) => {
    if (fault === SPARSE_CM12_FACTORED_AEI_FAULT.none) {
      fault = code; firstFaultRecord = record;
    }
  };
  if (candidateGeneration !== acceptedGeneration + 1
    || image.words[slotAt + 2] !== acceptedGeneration) {
    fail(SPARSE_CM12_FACTORED_AEI_FAULT.generation, 0);
  }
  if (actualChanged.size !== changed.size
    || [...actualChanged].some((leaf) => !changed.has(leaf))) {
    fail(SPARSE_CM12_FACTORED_AEI_FAULT.deltaCoverage, 9);
  }
  const actual = slotActiveLeaves(image, shadowSlot);
  if (actual.size !== target.size || [...target].some((leaf) => !actual.has(leaf))) {
    fail(SPARSE_CM12_FACTORED_AEI_FAULT.topologyMismatch, 0);
  }
  let activeLeafCount = 0, patchReferenceCount = 0, canonicalLeafCount = 0;
  let leafHash = 0x811c_9dc5, patchHash = 0x811c_9dc5;
  for (let leaf = 0; leaf < image.catalog.layout.leafCapacity; leaf += 1) {
    const leafAt = image.catalog.layout.slotLeafBaseWords[shadowSlot]
      + leaf * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
    const shouldBeActive = target.has(leaf);
    const descriptorId = image.words[leafAt + 2]!;
    const authoritativeDescriptorId = image.catalog.descriptorIdByLeaf[leaf]!;
    if (((image.words[leafAt + 1]! & 1) !== 0) !== shouldBeActive
      || descriptorId !== authoritativeDescriptorId) {
      fail(SPARSE_CM12_FACTORED_AEI_FAULT.topologyMismatch, leaf);
    }
    if (expectedClosure.has(leaf) && image.words[leafAt] !== candidateGeneration) {
      fail(SPARSE_CM12_FACTORED_AEI_FAULT.generation, leaf);
    }
    if (!shouldBeActive) {
      for (let localRef = 0; localRef < SPARSE_CM12_FACTORED_AEI_PATCHES_PER_LEAF;
        localRef += 1) {
        const ref = leaf * SPARSE_CM12_FACTORED_AEI_PATCHES_PER_LEAF + localRef;
        const refAt = image.catalog.layout.slotPatchRefBaseWords[shadowSlot]
          + ref * SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS;
        if (expectedClosure.has(leaf)
          && image.words[refAt] !== candidateGeneration) {
          fail(SPARSE_CM12_FACTORED_AEI_FAULT.generation, ref);
        }
        if (image.words[refAt + 1] !== SPARSE_CM12_FACTORED_AEI_INVALID) {
          fail(SPARSE_CM12_FACTORED_AEI_FAULT.patchReference, ref);
        }
      }
      continue;
    }
    activeLeafCount += 1;
    const descriptor = image.catalog.canonical[descriptorId];
    if (!descriptor?.certified) {
      fail(SPARSE_CM12_FACTORED_AEI_FAULT.canonicalCertificate, leaf);
    } else canonicalLeafCount += 1;
    leafHash = fnv(fnv(leafHash, leaf), descriptorId);
    for (let side = 0; side < 6; side += 1) {
      const expected = image.catalog.patchIdsByCanonicalSide[authoritativeDescriptorId]![side]!
        .filter((id) => {
        const patch = image.catalog.patches[id]!;
        return patch.targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID
          || target.has(patch.targetLeaf)
            && image.catalog.descriptorIdByLeaf[patch.targetLeaf]
              === patch.targetCanonicalId;
      });
      for (let local = 0; local < SPARSE_CM12_FACTORED_AEI_PATCHES_PER_FACE;
        local += 1) {
        const ref = patchRefIndex(leaf, side, local);
        const refAt = image.catalog.layout.slotPatchRefBaseWords[shadowSlot]
          + ref * SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS;
        const observed = image.words[refAt + 1]!;
        const wanted = expected[local] ?? SPARSE_CM12_FACTORED_AEI_INVALID;
        if (expectedClosure.has(leaf)
          && image.words[refAt] !== candidateGeneration) {
          fail(SPARSE_CM12_FACTORED_AEI_FAULT.generation, ref);
        }
        if (observed !== wanted) fail(SPARSE_CM12_FACTORED_AEI_FAULT.patchReference, ref);
        if (observed !== SPARSE_CM12_FACTORED_AEI_INVALID) {
          patchReferenceCount += 1; patchHash = fnv(fnv(patchHash, ref), observed);
        }
      }
    }
  }
  const selectorUnchanged = (image.words[3]! & 1) === acceptedSlot
    && image.words[4] === acceptedGeneration;
  if (!selectorUnchanged) fail(SPARSE_CM12_FACTORED_AEI_FAULT.selectorChanged, 3);
  if (image.words[slotAt + 9] !== changed.size
    || image.words[slotAt + 10] !== expectedDeltaHash) {
    fail(SPARSE_CM12_FACTORED_AEI_FAULT.deltaCoverage, 9);
  }
  image.words[slotAt + 5] = fault;
  image.words[slotAt + 6] = firstFaultRecord;
  image.words[slotAt + 1] = fault === 0
    ? SPARSE_CM12_FACTORED_AEI_SLOT_STATE.ready
    : SPARSE_CM12_FACTORED_AEI_SLOT_STATE.fault;
  return Object.freeze({ passed: fault === 0, acceptedSlot, shadowSlot,
    acceptedGeneration, candidateGeneration, fault, firstFaultRecord,
    activeLeafCount, patchReferenceCount, canonicalLeafCount,
    deltaLeafCount: changed.size, deltaClosureCount: expectedClosure.size,
    leafHash, patchHash, selectorUnchanged });
}

/** Single selector publication followed by exact fixed-record replay. */
export function commitSparseCM12FactoredAEIShadow(options: Readonly<{
  image: SparseCM12FactoredAEIImage;
  receipt: SparseCM12FactoredAEIPreflipReceipt;
  targetActiveLeaves: Iterable<number>;
  changedLeaves: Iterable<number>;
}>): void {
  const { image, receipt } = options;
  if (!receipt.passed || image.words[3] !== receipt.acceptedSlot
    || image.words[4] !== receipt.acceptedGeneration) {
    throw new Error("AEI cannot commit an invalid or stale preflip receipt");
  }
  image.words[4] = receipt.candidateGeneration;
  image.words[3] = receipt.shadowSlot;
  const target = activeSet(options.targetActiveLeaves, image.catalog.layout.leafCapacity);
  const changed = activeSet(options.changedLeaves, image.catalog.layout.leafCapacity);
  const closure = deltaClosure(image.catalog, changed);
  const retired = receipt.acceptedSlot;
  for (const leaf of closure) {
    writeLeafAndPatches(image, retired, leaf, receipt.candidateGeneration, target);
  }
  summarizeSlot(image, retired, receipt.candidateGeneration,
    receipt.candidateGeneration, SPARSE_CM12_FACTORED_AEI_SLOT_STATE.mirror);
  const acceptedAt = image.catalog.layout.slotBaseWords[receipt.shadowSlot];
  image.words[acceptedAt + 1] = SPARSE_CM12_FACTORED_AEI_SLOT_STATE.mirror;
}
