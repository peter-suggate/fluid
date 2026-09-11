import { SPARSE_CM12_FACTORED_AEI_INVALID } from
  "./sparse-cm12-factored-aei-topology";
import {
  SPARSE_CM12_PACKED_TEMPLATE_CELL_WORDS,
  SPARSE_CM12_PACKED_TEMPLATE_HEADER,
  SPARSE_CM12_PACKED_TEMPLATE_MAGIC,
} from "./sparse-cm12-factored-aei-packed-template";
import {
  SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF,
  SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS,
  unpackSparseCM12InternedBoundaryRefIdentity,
} from "./sparse-cm12-interned-boundary-operators";
import type { SparseCM12InternedBoundaryImage } from
  "./sparse-cm12-interned-boundary-image";

export interface SparseCM12IBOSemanticLeafReceipt {
  readonly leaf: number;
  readonly descriptorId: number;
  readonly rowCount: number;
  readonly digest: number;
  readonly digestSum: number;
  readonly rows: readonly number[];
  readonly duplicateCandidateRows: number;
}

export interface SparseCM12IBOSemanticAuthority {
  readonly receipts: readonly SparseCM12IBOSemanticLeafReceipt[];
  readonly exact: boolean;
  readonly firstMismatchLeaf: number | null;
  readonly totalRows: number;
  readonly duplicateCandidateRows: number;
}

const fnv = (hash: number, value: number): number =>
  Math.imul((hash ^ value) >>> 0, 0x0100_0193) >>> 0;

interface Reader { readonly words: Uint32Array; readonly rowCount: number;
  readonly cellBase: number; readonly rowBase: number; readonly termBase: number;
  readonly configurationBase: number }
const reader = (words: Uint32Array): Reader => {
  const h = SPARSE_CM12_PACKED_TEMPLATE_HEADER;
  if (words[h.magic] !== SPARSE_CM12_PACKED_TEMPLATE_MAGIC) {
    throw new Error("ISA1 requires a packed SCMT authority");
  }
  return { words, rowCount: words[h.rowCount]!, cellBase: words[h.cellBase]!,
    rowBase: words[h.rowBase]!, termBase: words[h.termBase]!,
    configurationBase: words[h.candidateConfigurationBase]! };
};
const rowWord = (r: Reader, plane: number, row: number) =>
  r.words[r.rowBase + plane * r.rowCount + row]!;
const rowTerms = (r: Reader, row: number) => {
  const packed = rowWord(r, 0, row);
  return { first: packed & 0x007f_ffff, count: packed >>> 23 };
};
const stableRowSemanticWords = (r: Reader, row: number): readonly number[] => {
  const terms = rowTerms(r, row), result = [row, rowWord(r, 1, row) & 0xf000_0000,
    rowWord(r, 2, row), rowWord(r, 3, row), rowWord(r, 4, row), rowWord(r, 5, row),
    terms.count];
  for (let local = 0; local < terms.count; local += 1) {
    const at = r.termBase + 2 * (terms.first + local);
    result.push(r.words[at]!, r.words[at + 1]!);
  }
  return result;
};
const semanticCommutativeDigest = (rows: ReadonlyMap<number, readonly number[]>):
readonly [number, number] => {
  let xor = 0, sum = 0;
  for (const words of rows.values()) {
    let rowHash = 0x811c_9dc5; for (const word of words) rowHash = fnv(rowHash, word);
    const mixed = fnv(fnv(0x9e37_79b9, rowHash), words[0]!);
    xor = (xor ^ mixed) >>> 0; sum = (sum + Math.imul(mixed, 0x85eb_ca6b)) >>> 0;
  }
  return [xor, sum];
};

const selected = (active: ReadonlySet<number>, descriptors: readonly number[],
  image: SparseCM12InternedBoundaryImage, metadata: number): boolean => {
  const leaf = metadata >>> 5, resolution = metadata & 31;
  if (!active.has(leaf)) return false;
  return image.compilation.catalog.canonical[descriptors[leaf]!]!.resolution === resolution;
};

/** SCMT candidate-face semantic authority; no IBO/IRL word is read. */
export function compileSparseCM12IBOSCMTLeafSemantics(options: Readonly<{
  image: SparseCM12InternedBoundaryImage;
  packedWords: Uint32Array;
  activeLeaves: Iterable<number>;
  descriptorIdByLeaf: readonly number[];
  leaves: Iterable<number>;
}>): readonly SparseCM12IBOSemanticLeafReceipt[] {
  const { image } = options, r = reader(options.packedWords);
  const active = new Set(options.activeLeaves), result: SparseCM12IBOSemanticLeafReceipt[] = [];
  for (const leaf of [...new Set(options.leaves)].sort((a, b) => a - b)) {
    const descriptorId = options.descriptorIdByLeaf[leaf]!;
    const descriptor = image.compilation.catalog.canonical[descriptorId];
    if (!descriptor || descriptor.leafId !== leaf) throw new Error("ISA1 invalid descriptor");
    const rows = new Map<number, readonly number[]>(); let duplicateCandidateRows = 0;
    if (active.has(leaf)) for (let side = 0; side < 6; side += 1) {
      const offsets = r.words[r.configurationBase + descriptorId * 6 + side]!;
      for (let boundary = 0; boundary < descriptor.resolution ** 2; boundary += 1) {
        const begin = r.words[offsets + boundary]!, end = r.words[offsets + boundary + 1]!;
        for (let at = begin; at < end; at += 1) {
          const row = r.words[at]!, requirementAt = rowWord(r, 1, row) & 0x0fff_ffff;
          const count = r.words[requirementAt]!;
          let accepted = count > 0;
          for (let local = 0; local < count && accepted; local += 1) {
            accepted = selected(active, options.descriptorIdByLeaf, image,
              r.words[requirementAt + 1 + local]!);
          }
          if (!accepted) continue;
          if (rows.has(row)) duplicateCandidateRows += 1;
          else rows.set(row, stableRowSemanticWords(r, row));
        }
      }
    }
    const [digest, digestSum] = semanticCommutativeDigest(rows);
    result.push(Object.freeze({ leaf, descriptorId, rowCount: rows.size,
      digest, digestSum, rows: Object.freeze([...rows.keys()].sort((a, b) => a - b)),
      duplicateCandidateRows }));
  }
  return Object.freeze(result);
}

/** Expand selected IBO slot refs into stable row/cell semantics; no IRL read. */
export function compileSparseCM12IBOSlotLeafSemantics(options: Readonly<{
  image: SparseCM12InternedBoundaryImage; slot: 0 | 1; leaves: Iterable<number>;
}>): readonly SparseCM12IBOSemanticLeafReceipt[] {
  const { image, slot } = options, result: SparseCM12IBOSemanticLeafReceipt[] = [];
  for (const leaf of [...new Set(options.leaves)].sort((a, b) => a - b)) {
    const leafAt = image.layout.slotLeafBaseWords[slot]
      + leaf * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS;
    const active = (image.words[leafAt + 1]! & 1) !== 0;
    const descriptorId = image.words[leafAt + 2]!;
    const source = image.compilation.catalog.canonical[descriptorId]!;
    const rows = new Map<number, readonly number[]>();
    if (active) for (let side = 0; side < 6; side += 1) {
      const count = image.words[leafAt + 5]! >>> (3 * side) & 7;
      for (let local = 0; local < count; local += 1) {
        const refAt = image.layout.slotRefBaseWords[slot]
          + (leaf * SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF + side * 4 + local)
            * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS;
        const [templateId, targetLeaf] = unpackSparseCM12InternedBoundaryRefIdentity(
          image.words[refAt]!);
        const rowBase = image.words[refAt + 1]!;
        const template = image.compilation.templates[templateId]!;
        const target = targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID ? undefined
          : image.compilation.catalog.canonical[image.words[
            image.layout.slotLeafBaseWords[slot]
              + targetLeaf * SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS + 2]!]!;
        for (let rowLocal = 0; rowLocal < template.rowCount; rowLocal += 1) {
          const at = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
            + rowLocal * SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS;
          const row = rowBase + template.words[at]!, packedTerms = template.words[at + 1]!;
          const first = packedTerms & 0x007f_ffff, termCount = packedTerms >>> 23;
          const words = [row, template.words[at + 2]!, template.words[at + 3]!,
            template.words[at + 4]!, template.words[at + 5]!, template.words[at + 6]!,
            termCount];
          for (let term = 0; term < termCount; term += 1) {
            const termAt = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
              + template.rowCount * SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS
              + (first + term) * SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS;
            const normalized = template.words[termAt]!;
            const stableCell = (normalized & 0x8000_0000) !== 0
              ? (target?.cellFirst ?? SPARSE_CM12_FACTORED_AEI_INVALID)
                + (normalized & 0x7fff_ffff)
              : source.cellFirst + normalized;
            words.push(stableCell >>> 0, template.words[termAt + 1]!);
          }
          if (rows.has(row)) throw new Error(`ISA1 duplicate IBO stable row ${row}`);
          rows.set(row, words);
        }
      }
    }
    const [digest, digestSum] = semanticCommutativeDigest(rows);
    result.push(Object.freeze({ leaf, descriptorId, rowCount: rows.size,
      digest, digestSum, rows: Object.freeze([...rows.keys()].sort((a, b) => a - b)),
      duplicateCandidateRows: 0 }));
  }
  return Object.freeze(result);
}

export function compareSparseCM12IBOSemanticAuthority(options: Readonly<{
  image: SparseCM12InternedBoundaryImage; packedWords: Uint32Array;
  activeLeaves: Iterable<number>; descriptorIdByLeaf: readonly number[];
  leaves: Iterable<number>; slot: 0 | 1;
}>): SparseCM12IBOSemanticAuthority {
  const expected = compileSparseCM12IBOSCMTLeafSemantics(options);
  const observed = compileSparseCM12IBOSlotLeafSemantics(options);
  let firstMismatchLeaf: number | null = null, totalRows = 0, duplicates = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const a = expected[index]!, b = observed[index]!; totalRows += a.rowCount;
    duplicates += a.duplicateCandidateRows;
    if (firstMismatchLeaf === null && (a.leaf !== b.leaf || a.descriptorId !== b.descriptorId
      || a.rowCount !== b.rowCount || a.digest !== b.digest || a.digestSum !== b.digestSum
      || a.rows.length !== b.rows.length || a.rows.some((row, at) => row !== b.rows[at]))) {
      firstMismatchLeaf = a.leaf;
    }
  }
  return Object.freeze({ receipts: expected, exact: firstMismatchLeaf === null,
    firstMismatchLeaf, totalRows, duplicateCandidateRows: duplicates });
}

export interface SparseCM12GeometryFaceNeighbors {
  readonly offsets: Uint32Array;
  readonly neighbors: Uint32Array;
  readonly words: Uint32Array;
  readonly offsetBaseWords: number;
  readonly neighborBaseWords: number;
  readonly bytes: number;
  readonly maximumFanout: number;
}

/** Construction-only geometry compiler; runtime consumes the bounded CSR. */
export function compileSparseCM12GeometryFaceNeighbors(options: Readonly<{
  coordinates: readonly (readonly [number, number, number])[];
  spans: readonly number[];
}>): SparseCM12GeometryFaceNeighbors {
  if (options.coordinates.length !== options.spans.length) {
    throw new Error("ISA1 geometry coordinate/span capacities differ");
  }
  const lists = options.coordinates.map(() => [] as number[]);
  const overlaps = (a0: number, a1: number, b0: number, b1: number) =>
    Math.max(a0, b0) < Math.min(a1, b1);
  for (let a = 0; a < options.coordinates.length; a += 1) {
    const ca = options.coordinates[a]!, sa = options.spans[a]!;
    if (!Number.isSafeInteger(sa) || sa < 1) throw new Error("ISA1 invalid brick span");
    for (let b = a + 1; b < options.coordinates.length; b += 1) {
      const cb = options.coordinates[b]!, sb = options.spans[b]!;
      let face = false;
      for (let axis = 0; axis < 3 && !face; axis += 1) {
        const t0 = (axis + 1) % 3, t1 = (axis + 2) % 3;
        face = (ca[axis]! + sa === cb[axis]! || cb[axis]! + sb === ca[axis]!)
          && overlaps(ca[t0]!, ca[t0]! + sa, cb[t0]!, cb[t0]! + sb)
          && overlaps(ca[t1]!, ca[t1]! + sa, cb[t1]!, cb[t1]! + sb);
      }
      if (face) { lists[a]!.push(b); lists[b]!.push(a); }
    }
  }
  const offsets = new Uint32Array(lists.length + 1);
  for (let leaf = 0; leaf < lists.length; leaf += 1) {
    lists[leaf]!.sort((a, b) => a - b);
    offsets[leaf + 1] = offsets[leaf]! + lists[leaf]!.length;
  }
  const neighbors = Uint32Array.from(lists.flat());
  const offsetBaseWords = 8, neighborBaseWords = offsetBaseWords + offsets.length;
  const words = new Uint32Array(neighborBaseWords + neighbors.length);
  words.set([0x4947_4e31, 1, lists.length, offsetBaseWords, neighborBaseWords,
    neighbors.length, words.length], 0); words.set(offsets, offsetBaseWords);
  words.set(neighbors, neighborBaseWords);
  return Object.freeze({ offsets, neighbors, words, offsetBaseWords, neighborBaseWords,
    bytes: 4 * words.length,
    maximumFanout: lists.reduce((maximum, list) => Math.max(maximum, list.length), 0) });
}

export function compileSparseCM12GeometryDeltaClosure(options: Readonly<{
  geometry: SparseCM12GeometryFaceNeighbors; changedLeaves: Iterable<number>;
}>): Readonly<{ leaves: readonly number[]; count: number; hash: number }> {
  const closure = new Set<number>();
  for (const leaf of options.changedLeaves) {
    if (leaf < 0 || leaf + 1 >= options.geometry.offsets.length) {
      throw new RangeError("ISA1 changed leaf outside geometry capacity");
    }
    closure.add(leaf);
    for (let at = options.geometry.offsets[leaf]!;
      at < options.geometry.offsets[leaf + 1]!; at += 1) {
      closure.add(options.geometry.neighbors[at]!);
    }
  }
  const leaves = [...closure].sort((a, b) => a - b);
  let hash = 0x811c_9dc5; for (const leaf of leaves) hash = fnv(hash, leaf);
  return Object.freeze({ leaves: Object.freeze(leaves), count: leaves.length, hash });
}
