import {
  SPARSE_CM12_FACTORED_AEI_INVALID,
  SPARSE_CM12_FACTORED_AEI_RELATION,
  createSparseCM12FactoredAEICatalogFromAuthority,
  type SparseCM12FactoredAEICanonicalDescriptor,
  type SparseCM12FactoredAEICatalog,
  type SparseCM12FactoredAEIPatchDescriptor,
} from "./sparse-cm12-factored-aei-topology";

export const SPARSE_CM12_PACKED_TEMPLATE_MAGIC = 0x5343_4d54;
export const SPARSE_CM12_PACKED_TEMPLATE_HEADER_WORDS = 27;
export const SPARSE_CM12_PACKED_TEMPLATE_CELL_WORDS = 8;
export const SPARSE_CM12_PACKED_TEMPLATE_ROW_PLANES = 9;
export const SPARSE_CM12_PACKED_TEMPLATE_HEADER = Object.freeze({
  magic: 0, version: 1, cellCount: 2, rowCount: 3, termCount: 4,
  cellBase: 6, rowBase: 7, termBase: 8, cellRangeBase: 11,
  leafCapacity: 13, rowOwnerRangeBase: 16,
  candidateConfigurationBase: 24, candidatePatchBase: 25,
  candidateRowBase: 26,
} as const);

export interface SparseCM12FactoredAEIPackedTemplateAuthority {
  readonly words: Uint32Array;
  readonly brickFineResolution: 8 | 16;
  /** Atlas-array-order stable keys; leaf identity is always the array index. */
  readonly brickKeyByLeafId: readonly number[];
  readonly validDimensions: (
    leaf: number, resolution: number, cellFirst: number, cellCount: number,
  ) => readonly [number, number, number];
  readonly scaleLog2: (leaf: number, resolution: number) => number;
  /** Initial selected rung only; immutable compilation still covers every rung. */
  readonly selectedResolution: (leaf: number) => number;
  /** Stable-domain face adjacency, independent of accepted SCMT row coverage. */
  readonly neighborLeavesByLeaf?: readonly (readonly number[])[];
  /** Optional hard stop for the performance-plan immutable budget. */
  readonly immutableMaximumBytes?: number;
}

const fnv = (hash: number, value: number): number =>
  Math.imul((hash ^ (value >>> 0)) >>> 0, 0x0100_0193) >>> 0;
const F32_BUFFER = new ArrayBuffer(4);
const F32_WORD = new Uint32Array(F32_BUFFER);
const F32_FLOAT = new Float32Array(F32_BUFFER);
const f32 = (bits: number): number => {
  F32_WORD[0] = bits; return F32_FLOAT[0]!;
};

interface PackedReader {
  readonly words: Uint32Array;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly cellBase: number;
  readonly rowBase: number;
  readonly termBase: number;
  readonly cellRangeBase: number;
  readonly rowOwnerRangeBase: number;
  readonly candidateConfigurationBase: number;
  readonly candidatePatchBase: number;
  readonly candidateRowBase: number;
}

const packedReader = (words: Uint32Array): PackedReader => {
  const h = SPARSE_CM12_PACKED_TEMPLATE_HEADER;
  if (words.length < SPARSE_CM12_PACKED_TEMPLATE_HEADER_WORDS
    || words[h.magic] !== SPARSE_CM12_PACKED_TEMPLATE_MAGIC
    || words[h.version] !== 1) {
    throw new Error("AEI packed-template authority has an invalid SCMT header");
  }
  const readOffset = (word: number, label: string) => {
    const value = words[word]!;
    if (value < SPARSE_CM12_PACKED_TEMPLATE_HEADER_WORDS || value >= words.length) {
      throw new Error(`AEI packed-template ${label} offset is out of range`);
    }
    return value;
  };
  return { words, cellCount: words[h.cellCount]!, rowCount: words[h.rowCount]!,
    cellBase: readOffset(h.cellBase, "cell"), rowBase: readOffset(h.rowBase, "row"),
    termBase: readOffset(h.termBase, "term"),
    cellRangeBase: readOffset(h.cellRangeBase, "cell range"),
    rowOwnerRangeBase: readOffset(h.rowOwnerRangeBase, "row owner"),
    candidateConfigurationBase: readOffset(h.candidateConfigurationBase,
      "candidate configuration"),
    candidatePatchBase: readOffset(h.candidatePatchBase, "candidate patch"),
    candidateRowBase: readOffset(h.candidateRowBase, "candidate row") };
};
const rowWord = (reader: PackedReader, plane: number, row: number) =>
  reader.words[reader.rowBase + plane * reader.rowCount + row]!;
const rowMetadata = (reader: PackedReader, row: number) => rowWord(reader, 1, row);
const rowKind = (reader: PackedReader, row: number) => (rowMetadata(reader, row) >> 28) & 3;
const rowAxis = (reader: PackedReader, row: number) => rowMetadata(reader, row) >>> 30;
const rowTerms = (reader: PackedReader, row: number): readonly number[] => {
  const packed = rowWord(reader, 0, row);const first = packed & 0x007f_ffff;
  const count = packed >>> 23;
  return Array.from({ length: count }, (_, local) => first + local);
};
const termCell = (reader: PackedReader, term: number) =>
  reader.words[reader.termBase + 2 * term]!;
const termBits = (reader: PackedReader, term: number) =>
  reader.words[reader.termBase + 2 * term + 1]!;
const cellMetadata = (reader: PackedReader, cell: number) =>
  reader.words[reader.cellBase + SPARSE_CM12_PACKED_TEMPLATE_CELL_WORDS * cell + 7]!;
const cellLeaf = (reader: PackedReader, cell: number) => cellMetadata(reader, cell) >>> 5;
const cellResolution = (reader: PackedReader, cell: number) => cellMetadata(reader, cell) & 31;

const localFor = (cell: number, descriptor: SparseCM12FactoredAEICanonicalDescriptor):
readonly [number, number, number] | undefined => {
  if (cell < descriptor.cellFirst) return undefined;
  const ordinal = cell - descriptor.cellFirst;
  const [nx, ny, nz] = descriptor.validDimensions;
  if (ordinal >= nx * ny * nz) return undefined;
  const z = Math.floor(ordinal / (nx * ny));
  const remain = ordinal - z * nx * ny;
  const y = Math.floor(remain / nx), x = remain - y * nx;
  return [x, y, z];
};

const compileCanonical = (options: Readonly<{
  authority: SparseCM12FactoredAEIPackedTemplateAuthority;
  reader: PackedReader;
  levels: readonly number[];
}>): SparseCM12FactoredAEICanonicalDescriptor[] => {
  const { authority, reader, levels } = options;
  const result: SparseCM12FactoredAEICanonicalDescriptor[] = [];
  for (let leaf = 0; leaf < authority.brickKeyByLeafId.length; leaf += 1) {
    for (let level = 0; level < levels.length; level += 1) {
      const id = leaf * levels.length + level, resolution = levels[level]!;
      const rangeAt = reader.cellRangeBase + 2 * id;
      const cellFirst = reader.words[rangeAt]!, cellCount = reader.words[rangeAt + 1]!;
      const dimensions = authority.validDimensions(leaf, resolution, cellFirst, cellCount);
      const ownerFirst = reader.words[reader.rowOwnerRangeBase + id]!;
      const ownerEnd = reader.words[reader.rowOwnerRangeBase + id + 1]!;
      let failure: string | undefined;
      let unavailable = false;
      if (cellFirst + cellCount > reader.cellCount
        || dimensions[0] * dimensions[1] * dimensions[2] !== cellCount) {
        failure = "packed leaf/rung range is unavailable";
        unavailable = true;
      }
      for (let cell = cellFirst; cell < cellFirst + cellCount && !failure; cell += 1) {
        if (cellLeaf(reader, cell) !== leaf || cellResolution(reader, cell) !== resolution) {
          failure = "packed leaf/rung range aliases another authoritative rung";
          unavailable = true;
        }
      }
      if (unavailable) {
        result.push(Object.freeze({ id, leafId: leaf, resolution,
          cellFirst: SPARSE_CM12_FACTORED_AEI_INVALID,
          validDimensions: [0, 0, 0] as const,
          scaleLog2: authority.scaleLog2(leaf, resolution),
          rowBase: [SPARSE_CM12_FACTORED_AEI_INVALID,
            SPARSE_CM12_FACTORED_AEI_INVALID,
            SPARSE_CM12_FACTORED_AEI_INVALID] as const,
          rowCount: [0, 0, 0] as const, canonicalRowCount: 0,
          rowIdHash: 0, termHash: 0, geometryHash: 0, certified: false,
          firstFailure: failure }));
        continue;
      }
      const rowsByAxis = ([0, 1, 2] as const).map((axis) => Array.from(
        { length: Math.max(0, ownerEnd - ownerFirst) }, (_, offset) => ownerFirst + offset,
      ).filter((row) => rowKind(reader, row) === 0 && rowAxis(reader, row) === axis));
      const expectedCounts = [Math.max(0, dimensions[0] - 1) * dimensions[1] * dimensions[2],
        dimensions[0] * Math.max(0, dimensions[1] - 1) * dimensions[2],
        dimensions[0] * dimensions[1] * Math.max(0, dimensions[2] - 1)] as const;
      const bases = rowsByAxis.map((rows) =>
        rows[0] ?? SPARSE_CM12_FACTORED_AEI_INVALID) as [number, number, number];
      let rowIdHash = 0x811c_9dc5, termHash = 0x811c_9dc5, geometryHash = 0x811c_9dc5;
      for (const axis of [0, 1, 2] as const) {
        if (!failure && rowsByAxis[axis]!.length !== expectedCounts[axis]) {
          failure = `packed axis ${axis} row count differs`;
        }
        for (let ordinal = 0; ordinal < rowsByAxis[axis]!.length; ordinal += 1) {
          const row = rowsByAxis[axis]![ordinal]!;
          if (!failure && row !== bases[axis]! + ordinal) {
            failure = `packed axis ${axis} rows are not contiguous`;
          }
          const terms = rowTerms(reader, row);
          if (terms.length !== 2 || !(f32(termBits(reader, terms[0]!)) < 0)
            || !(f32(termBits(reader, terms[1]!)) > 0)) {
            failure ??= `packed row ${row} term identity/order differs`;
          } else {
            const negative = termCell(reader, terms[0]!);
            const positive = termCell(reader, terms[1]!);
            const local = localFor(positive, { cellFirst,
              validDimensions: dimensions } as SparseCM12FactoredAEICanonicalDescriptor);
            const stride = axis === 0 ? 1 : axis === 1 ? dimensions[0]
              : dimensions[0] * dimensions[1];
            const expectedOrdinal = local === undefined ? -1 : axis === 0
              ? local[0] - 1 + (dimensions[0] - 1) * (local[1] + dimensions[1] * local[2])
              : axis === 1 ? local[0] + dimensions[0]
                * (local[1] - 1 + (dimensions[1] - 1) * local[2])
                : local[0] + dimensions[0] * (local[1]
                  + dimensions[1] * (local[2] - 1));
            if (negative + stride !== positive || expectedOrdinal !== ordinal) {
              failure ??= `packed row ${row} endpoint formula differs`;
            }
          }
          rowIdHash = fnv(rowIdHash, row);
          for (const term of terms) {
            termHash = fnv(fnv(termHash, termCell(reader, term)), termBits(reader, term));
          }
          geometryHash = fnv(fnv(fnv(fnv(geometryHash, axis), rowWord(reader, 3, row)),
            rowWord(reader, 4, row)), rowWord(reader, 2, row));
        }
      }
      result.push(Object.freeze({ id, leafId: leaf, resolution, cellFirst,
        validDimensions: dimensions, scaleLog2: authority.scaleLog2(leaf, resolution),
        rowBase: bases, rowCount: expectedCounts,
        canonicalRowCount: expectedCounts[0] + expectedCounts[1] + expectedCounts[2],
        rowIdHash, termHash, geometryHash, certified: failure === undefined,
        ...(failure ? { firstFailure: failure } : {}) }));
    }
  }
  return result;
};

const compilePatches = (options: Readonly<{
  reader: PackedReader;
  canonical: readonly SparseCM12FactoredAEICanonicalDescriptor[];
  levels: readonly number[];
}>): Readonly<{ patches: SparseCM12FactoredAEIPatchDescriptor[];
  exceptionRows: number[] }> => {
  const { reader, canonical, levels } = options;
  const patches: SparseCM12FactoredAEIPatchDescriptor[] = [], exceptionRows: number[] = [];
  const canonicalForCell = (cell: number) => {
    const leaf = cellLeaf(reader, cell), resolution = cellResolution(reader, cell);
    const level = levels.indexOf(resolution);return level < 0 ? undefined
      : canonical[leaf * levels.length + level];
  };
  for (const source of canonical) if (source.certified) for (let side = 0;
    side < 6; side += 1) {
    const configuration = reader.candidateConfigurationBase
      + (source.id * 6 + side);
    const patchOffsets = reader.words[configuration]!;
    const rows = new Set<number>();
    for (let boundary = 0; boundary < source.resolution ** 2; boundary += 1) {
      const begin = reader.words[patchOffsets + boundary]!;
      const end = reader.words[patchOffsets + boundary + 1]!;
      for (let at = begin; at < end; at += 1) rows.add(reader.words[at]!);
    }
    const groups = new Map<number, number[]>();
    for (const row of rows) {
      const targets = new Set<number>();
      for (const term of rowTerms(reader, row)) {
        const descriptor = canonicalForCell(termCell(reader, term));
        if (descriptor?.certified && descriptor.id !== source.id) targets.add(descriptor.id);
      }
      if (targets.size > 1) {
        throw new Error(`AEI packed row ${row} spans more than one target leaf/rung`);
      }
      const target = [...targets][0] ?? SPARSE_CM12_FACTORED_AEI_INVALID;
      const group = groups.get(target) ?? []; group.push(row); groups.set(target, group);
    }
    for (const [targetId, rawRows] of groups) {
      const selectedRows = [...new Set(rawRows)].sort((a, b) => a - b);
      const target = targetId === SPARSE_CM12_FACTORED_AEI_INVALID
        ? undefined : canonical[targetId];
      const contiguous = selectedRows.every((row, index) => row === selectedRows[0]! + index);
      const allEqual = target !== undefined && selectedRows.every((row) =>
        rowKind(reader, row) === 1 && rowTerms(reader, row).length === 2);
      const axis = side >> 1, tangents = ([0, 1, 2] as const)
        .filter((value) => value !== axis);
      const pairs = selectedRows.map((row) => {
        let sourceLocal: readonly [number, number, number] | undefined;
        let targetLocal: readonly [number, number, number] | undefined;
        for (const term of rowTerms(reader, row)) {
          const cell = termCell(reader, term), descriptor = canonicalForCell(cell);
          if (descriptor?.id === source.id) sourceLocal = localFor(cell, source);
          if (descriptor?.id === targetId && target) targetLocal = localFor(cell, target);
        }
        return { sourceLocal, targetLocal };
      });
      const sourceU = pairs.map((pair) => pair.sourceLocal?.[tangents[0]!] ?? 0);
      const sourceV = pairs.map((pair) => pair.sourceLocal?.[tangents[1]!] ?? 0);
      const targetU = pairs.map((pair) => pair.targetLocal?.[tangents[0]!] ?? 0);
      const targetV = pairs.map((pair) => pair.targetLocal?.[tangents[1]!] ?? 0);
      const sourceOrigin = [Math.min(...sourceU), Math.min(...sourceV)] as const;
      const targetOrigin = [Math.min(...targetU), Math.min(...targetV)] as const;
      const dimensions = [Math.max(...sourceU) - sourceOrigin[0] + 1,
        Math.max(...sourceV) - sourceOrigin[1] + 1] as const;
      let mappingCertified = allEqual && contiguous
        && dimensions[0] * dimensions[1] === selectedRows.length;
      for (let index = 0; index < pairs.length && mappingCertified; index += 1) {
        const pair = pairs[index]!, u = index % dimensions[0], v = Math.floor(index / dimensions[0]);
        mappingCertified = pair.sourceLocal !== undefined && pair.targetLocal !== undefined
          && pair.sourceLocal[tangents[0]!] === sourceOrigin[0] + u
          && pair.sourceLocal[tangents[1]!] === sourceOrigin[1] + v
          && pair.targetLocal[tangents[0]!] === targetOrigin[0] + u
          && pair.targetLocal[tangents[1]!] === targetOrigin[1] + v
          && pair.sourceLocal[axis] === ((side & 1) !== 0
            ? source.validDimensions[axis]! - 1 : 0)
          && pair.targetLocal[axis] === ((side & 1) !== 0
            ? 0 : target!.validDimensions[axis]! - 1);
      }
      const relation = mappingCertified
        ? SPARSE_CM12_FACTORED_AEI_RELATION.equalRungCanonical
        : selectedRows.every((row) => rowKind(reader, row) === 2)
          ? SPARSE_CM12_FACTORED_AEI_RELATION.explicitMixed
          : selectedRows.every((row) => rowKind(reader, row) === 3)
            ? SPARSE_CM12_FACTORED_AEI_RELATION.explicitSparseAir
            : SPARSE_CM12_FACTORED_AEI_RELATION.explicitOther;
      let rowHash = 0x811c_9dc5, termHash = 0x811c_9dc5;
      for (const row of selectedRows) {
        rowHash = fnv(rowHash, row);
        for (const term of rowTerms(reader, row)) {
          termHash = fnv(fnv(termHash, termCell(reader, term)), termBits(reader, term));
        }
      }
      const exceptionFirst = exceptionRows.length;
      if (relation !== SPARSE_CM12_FACTORED_AEI_RELATION.equalRungCanonical) {
        exceptionRows.push(...selectedRows);
      }
      patches.push(Object.freeze({ id: patches.length, sourceLeaf: source.leafId,
        targetLeaf: target?.leafId ?? SPARSE_CM12_FACTORED_AEI_INVALID,
        sourceSide: side, relation,
        rowFirst: contiguous ? selectedRows[0]! : SPARSE_CM12_FACTORED_AEI_INVALID,
        rowCount: selectedRows.length, exceptionFirst,
        exceptionCount: relation === SPARSE_CM12_FACTORED_AEI_RELATION.equalRungCanonical
          ? 0 : selectedRows.length, rowHash, termHash,
        sourceCanonicalId: source.id,
        targetCanonicalId: target?.id ?? SPARSE_CM12_FACTORED_AEI_INVALID,
        sourceFaceOrigin: sourceOrigin, targetFaceOrigin: targetOrigin,
        faceDimensions: dimensions, mappingCertified }));
    }
  }
  return { patches, exceptionRows };
};

/** Compile AEI3 only from packed stable IDs/bits; no coefficient is synthesized. */
export function compileSparseCM12FactoredAEIPackedTemplateCatalog(
  authority: SparseCM12FactoredAEIPackedTemplateAuthority,
): SparseCM12FactoredAEICatalog {
  const reader = packedReader(authority.words);
  if (authority.brickKeyByLeafId.length
    !== authority.words[SPARSE_CM12_PACKED_TEMPLATE_HEADER.leafCapacity]) {
    throw new Error("AEI packed-template leaf capacity differs from atlas order");
  }
  const levels = Array.from({ length: Math.log2(authority.brickFineResolution) + 1 },
    (_, level) => 2 ** level);
  const canonical = compileCanonical({ authority, reader, levels });
  const descriptorIdByLeaf = authority.brickKeyByLeafId.map((_, leaf) => {
    const resolution = authority.selectedResolution(leaf), level = levels.indexOf(resolution);
    if (level < 0) throw new Error(`AEI selected rung ${leaf}/${resolution} is invalid`);
    const id = leaf * levels.length + level;
    if (!canonical[id]?.certified) {
      throw new Error(`AEI selected packed canonical ${leaf}/${resolution} failed: ${
        canonical[id]?.firstFailure ?? "missing descriptor"}`);
    }
    return id;
  });
  const { patches, exceptionRows } = compilePatches({ reader, canonical, levels });
  const catalog = createSparseCM12FactoredAEICatalogFromAuthority({
    leafCapacity: authority.brickKeyByLeafId.length, levelCount: levels.length,
    canonical, patches, exceptionRows, brickKeyByLeafId: authority.brickKeyByLeafId,
    descriptorIdByLeaf, neighborLeavesByLeaf: authority.neighborLeavesByLeaf });
  if (authority.immutableMaximumBytes !== undefined
    && catalog.layout.immutableBytes > authority.immutableMaximumBytes) {
    throw new Error(`AEI packed immutable topology needs ${catalog.layout.immutableBytes} bytes `
      + `(patches=${catalog.patches.length},exceptions=${catalog.exceptionRows.length},`
      + `slot=${catalog.layout.bytesPerSlot}); budget is ${authority.immutableMaximumBytes}`);
  }
  return catalog;
}
