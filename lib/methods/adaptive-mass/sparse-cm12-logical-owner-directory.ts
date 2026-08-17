import {
  sparseBrickContainingCoordinate,
  sparseBrickKey,
  sparseBrickSpan,
  type SparseAdaptiveMassAtlas,
  type SparseBrickFineResolution,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";

/** Immutable dense logical-owner directory, version "LOD1". */
export const SPARSE_CM12_LOGICAL_OWNER_MAGIC = 0x4c4f_4431;
export const SPARSE_CM12_LOGICAL_OWNER_VERSION = 1;
export const SPARSE_CM12_LOGICAL_OWNER_HEADER_WORDS = 16;
export const SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS = 2;
export const SPARSE_CM12_LOGICAL_OWNER_INVALID = 0xffff_ffff;
export const SPARSE_CM12_LOGICAL_OWNER_BRICK_BITS = 27;
export const SPARSE_CM12_LOGICAL_OWNER_BRICK_LIMIT =
  2 ** SPARSE_CM12_LOGICAL_OWNER_BRICK_BITS;
export const SPARSE_CM12_LOGICAL_OWNER_MAXIMUM_SPAN_LOG = 30;

export const SPARSE_CM12_LOGICAL_OWNER_FLAG = Object.freeze({
  complete: 1 << 0,
  validated: 1 << 1,
} as const);

export const SPARSE_CM12_LOGICAL_OWNER_HEADER = Object.freeze({
  magic: 0,
  version: 1,
  headerWords: 2,
  recordWords: 3,
  brickFineResolution: 4,
  presentationPageResolution: 5,
  logicalBricksX: 6,
  logicalBricksY: 7,
  logicalBricksZ: 8,
  logicalBrickCount: 9,
  residentBrickCount: 10,
  maximumSpanLog: 11,
  atlasGeneration: 12,
  flags: 13,
  recordBase: 14,
  totalWords: 15,
} as const);

export const SPARSE_CM12_LOGICAL_OWNER_RECORD = Object.freeze({
  /** `[owner brick:27 | span log2:5]`, or INVALID for an empty logical brick. */
  ownerAndSpan: 0,
  /** Logical key of the physical owner's aligned lower corner. */
  originKey: 1,
} as const);

export interface SparseCM12LogicalOwnerDirectoryLayout {
  readonly brickFineResolution: SparseBrickFineResolution;
  readonly presentationPageResolution: SparseBrickFineResolution;
  readonly logicalBrickDimensions: SparseBrickVec3;
  readonly logicalBrickCount: number;
  readonly residentBrickCount: number;
  readonly maximumSpanLog: number;
  readonly atlasGeneration: number;
  readonly recordBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12LogicalOwnerDirectory {
  readonly layout: SparseCM12LogicalOwnerDirectoryLayout;
  readonly words: Uint32Array;
}

export interface SparseCM12LogicalOwnerRecord {
  readonly brick: number;
  readonly spanBricks: number;
  readonly originKey: number;
  readonly origin: SparseBrickVec3;
}

export interface SparseCM12LogicalOwnerRuntime {
  readonly brickActive: (brick: number) => boolean;
  readonly acceptedBrickResolution: (brick: number) => SparseBrickResolution;
  readonly templateBrickCellRange: (
    brick: number,
    resolution: SparseBrickResolution,
  ) => readonly [first: number, count: number];
  readonly cellResolution?: (cell: number) => SparseBrickResolution;
  readonly cellOpenVolume?: (cell: number) => number;
}

export interface SparseCM12LogicalOwnerCell {
  readonly cell: number;
  readonly brick: number;
  readonly resolution: SparseBrickResolution;
}

const checkedU32 = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0) || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be ${positive ? "a positive" : "a non-negative"} u32`);
  }
  return value;
};

const checkedFineResolution = (
  value: number,
  label: string,
): SparseBrickFineResolution => {
  if (value !== 4 && value !== 8 && value !== 16) {
    throw new RangeError(`${label} must be 4, 8, or 16`);
  }
  return value;
};

const checkedProduct = (values: readonly number[], label: string): number => {
  const product = values.reduce((result, value) => result * value, 1);
  if (!Number.isSafeInteger(product) || product > 0xffff_fffe) {
    throw new RangeError(`${label} exceeds the addressable logical-key range`);
  }
  return product;
};

const coordinateForKey = (
  key: number,
  dimensions: SparseBrickVec3,
): [number, number, number] => {
  const xy = dimensions[0] * dimensions[1];
  const z = Math.floor(key / xy);
  const remainder = key - z * xy;
  const y = Math.floor(remainder / dimensions[0]);
  return [remainder - y * dimensions[0], y, z];
};

const recordWord = (
  layout: SparseCM12LogicalOwnerDirectoryLayout,
  key: number,
): number => layout.recordBaseWords + SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS * key;

/**
 * Build the immutable direct directory used by every finest-coordinate owner
 * lookup. Defaults are deliberately B16/P16; other supported shapes must opt
 * in explicitly so diagnostic/direct callers cannot silently retarget work.
 *
 * Bricks are applied in ascending span order. An ordinary or smaller macro
 * therefore wins inside an overlapping larger macro, exactly matching
 * `sparseBrickContainingCoordinate` and the resident's former span loop.
 */
export function createSparseCM12LogicalOwnerDirectory(
  atlas: SparseAdaptiveMassAtlas,
  options: {
    readonly brickFineResolution?: SparseBrickFineResolution;
    readonly presentationPageResolution?: SparseBrickFineResolution;
  } = {},
): SparseCM12LogicalOwnerDirectory {
  const brickFineResolution = checkedFineResolution(
    options.brickFineResolution ?? 16,
    "brickFineResolution",
  );
  const presentationPageResolution = checkedFineResolution(
    options.presentationPageResolution ?? 16,
    "presentationPageResolution",
  );
  if (atlas.brickFineResolution !== brickFineResolution) {
    throw new Error(`logical-owner B${brickFineResolution} does not match atlas B${atlas.brickFineResolution}`);
  }
  if (presentationPageResolution > brickFineResolution
    || brickFineResolution % presentationPageResolution !== 0) {
    throw new RangeError(`presentation page ${presentationPageResolution} does not divide brick ${brickFineResolution}`);
  }
  const dimensions = atlas.brickDimensions;
  dimensions.forEach((value, axis) => checkedU32(value, `logicalBrickDimensions[${axis}]`, true));
  const logicalBrickCount = checkedProduct(dimensions, "logicalBrickCount");
  if (atlas.bricks.length >= SPARSE_CM12_LOGICAL_OWNER_BRICK_LIMIT) {
    throw new RangeError("logical-owner resident brick count exhausts its 27-bit record");
  }
  const maximumSpanLog = Math.log2(atlas.maximumSpanBricks);
  if (!Number.isInteger(maximumSpanLog) || maximumSpanLog < 0
    || maximumSpanLog > SPARSE_CM12_LOGICAL_OWNER_MAXIMUM_SPAN_LOG) {
    throw new RangeError(`maximum span ${atlas.maximumSpanBricks} is not representable`);
  }
  const recordBaseWords = SPARSE_CM12_LOGICAL_OWNER_HEADER_WORDS;
  const totalWords = recordBaseWords
    + SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS * logicalBrickCount;
  if (!Number.isSafeInteger(totalWords) || totalWords > 0x3fff_ffff) {
    throw new RangeError("logical-owner directory exceeds the addressable GPU arena");
  }
  const layout: SparseCM12LogicalOwnerDirectoryLayout = Object.freeze({
    brickFineResolution,
    presentationPageResolution,
    logicalBrickDimensions: [...dimensions] as [number, number, number],
    logicalBrickCount,
    residentBrickCount: atlas.bricks.length,
    maximumSpanLog,
    atlasGeneration: checkedU32(atlas.generation, "atlas generation"),
    recordBaseWords,
    totalWords,
    totalBytes: Uint32Array.BYTES_PER_ELEMENT * totalWords,
  });
  const words = new Uint32Array(totalWords);
  words.fill(SPARSE_CM12_LOGICAL_OWNER_INVALID, recordBaseWords);
  words.set([
    SPARSE_CM12_LOGICAL_OWNER_MAGIC,
    SPARSE_CM12_LOGICAL_OWNER_VERSION,
    SPARSE_CM12_LOGICAL_OWNER_HEADER_WORDS,
    SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS,
    brickFineResolution,
    presentationPageResolution,
    dimensions[0], dimensions[1], dimensions[2],
    logicalBrickCount,
    atlas.bricks.length,
    maximumSpanLog,
    layout.atlasGeneration,
    0,
    recordBaseWords,
    totalWords,
  ]);

  const byAscendingSpan = atlas.bricks.map((source, brick) => ({ brick, source }))
    .sort((left, right) => sparseBrickSpan(left.source) - sparseBrickSpan(right.source)
      || left.brick - right.brick);
  for (const { brick, source } of byAscendingSpan) {
    const span = sparseBrickSpan(source);
    const spanLog = Math.log2(span);
    if (!Number.isInteger(spanLog) || spanLog > maximumSpanLog
      || source.coordinate.some((value) => value % span !== 0)) {
      throw new Error(`brick ${brick} has invalid aligned span ${span}`);
    }
    const originKey = sparseBrickKey(source.coordinate, dimensions);
    if (originKey !== source.key || originKey >= logicalBrickCount) {
      throw new Error(`brick ${brick} has invalid logical origin ${source.key}`);
    }
    const packed = ((brick << 5) | spanLog) >>> 0;
    const upper = source.coordinate.map((value, axis) =>
      Math.min(dimensions[axis]!, value + span)) as [number, number, number];
    for (let z = source.coordinate[2]; z < upper[2]; z += 1)
      for (let y = source.coordinate[1]; y < upper[1]; y += 1)
        for (let x = source.coordinate[0]; x < upper[0]; x += 1) {
          const key = sparseBrickKey([x, y, z], dimensions);
          const at = recordWord(layout, key);
          // Ascending spans reproduce the old lookup's smallest-span winner.
          if (words[at] !== SPARSE_CM12_LOGICAL_OWNER_INVALID) continue;
          words[at + SPARSE_CM12_LOGICAL_OWNER_RECORD.ownerAndSpan] = packed;
          words[at + SPARSE_CM12_LOGICAL_OWNER_RECORD.originKey] = originKey;
        }
  }

  words[SPARSE_CM12_LOGICAL_OWNER_HEADER.flags] =
    SPARSE_CM12_LOGICAL_OWNER_FLAG.complete
    | SPARSE_CM12_LOGICAL_OWNER_FLAG.validated;
  const directory = { layout, words } as const;
  validateSparseCM12LogicalOwnerDirectory(directory, atlas);
  return directory;
}

/** Decode one logical key. Any malformed header or record fails closed. */
export function sparseCM12LogicalOwnerAtKey(
  directory: SparseCM12LogicalOwnerDirectory,
  key: number,
): SparseCM12LogicalOwnerRecord | undefined {
  const { layout, words } = directory;
  if (!sparseCM12LogicalOwnerHeaderValid(directory)
    || !Number.isSafeInteger(key) || key < 0 || key >= layout.logicalBrickCount) {
    return undefined;
  }
  const at = recordWord(layout, key);
  const packed = words[at + SPARSE_CM12_LOGICAL_OWNER_RECORD.ownerAndSpan]!;
  const originKey = words[at + SPARSE_CM12_LOGICAL_OWNER_RECORD.originKey]!;
  if (packed === SPARSE_CM12_LOGICAL_OWNER_INVALID
    || originKey === SPARSE_CM12_LOGICAL_OWNER_INVALID) return undefined;
  const brick = packed >>> 5;
  const spanLog = packed & 31;
  if (brick >= layout.residentBrickCount || spanLog > layout.maximumSpanLog
    || originKey >= layout.logicalBrickCount) return undefined;
  const spanBricks = 2 ** spanLog;
  const origin = coordinateForKey(originKey, layout.logicalBrickDimensions);
  const query = coordinateForKey(key, layout.logicalBrickDimensions);
  if (origin.some((value) => value % spanBricks !== 0)
    || query.some((value, axis) => value < origin[axis]!
      || value - origin[axis]! >= spanBricks)) return undefined;
  return { brick, spanBricks, originKey, origin };
}

export function sparseCM12LogicalOwnerAtCoordinate(
  directory: SparseCM12LogicalOwnerDirectory,
  coordinate: SparseBrickVec3,
): SparseCM12LogicalOwnerRecord | undefined {
  if (coordinate.some((value, axis) => !Number.isSafeInteger(value) || value < 0
    || value >= directory.layout.logicalBrickDimensions[axis])) return undefined;
  return sparseCM12LogicalOwnerAtKey(directory,
    sparseBrickKey(coordinate, directory.layout.logicalBrickDimensions));
}

export function sparseCM12LogicalOwnerHeaderValid(
  directory: SparseCM12LogicalOwnerDirectory,
): boolean {
  const { layout, words } = directory;
  const h = SPARSE_CM12_LOGICAL_OWNER_HEADER;
  const requiredFlags = SPARSE_CM12_LOGICAL_OWNER_FLAG.complete
    | SPARSE_CM12_LOGICAL_OWNER_FLAG.validated;
  return words.length >= layout.totalWords
    && words[h.magic] === SPARSE_CM12_LOGICAL_OWNER_MAGIC
    && words[h.version] === SPARSE_CM12_LOGICAL_OWNER_VERSION
    && words[h.headerWords] === SPARSE_CM12_LOGICAL_OWNER_HEADER_WORDS
    && words[h.recordWords] === SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS
    && words[h.brickFineResolution] === layout.brickFineResolution
    && words[h.presentationPageResolution] === layout.presentationPageResolution
    && words[h.logicalBricksX] === layout.logicalBrickDimensions[0]
    && words[h.logicalBricksY] === layout.logicalBrickDimensions[1]
    && words[h.logicalBricksZ] === layout.logicalBrickDimensions[2]
    && words[h.logicalBrickCount] === layout.logicalBrickCount
    && words[h.residentBrickCount] === layout.residentBrickCount
    && words[h.maximumSpanLog] === layout.maximumSpanLog
    && words[h.atlasGeneration] === layout.atlasGeneration
    && (words[h.flags]! & requiredFlags) === requiredFlags
    && words[h.recordBase] === layout.recordBaseWords
    && words[h.totalWords] === layout.totalWords;
}

/**
 * Construction-time equivalence receipt. It exhaustively compares every
 * logical coordinate against the old span-directory lookup, including partial
 * macro coverage and a smaller leaf overriding a larger macro.
 */
export function validateSparseCM12LogicalOwnerDirectory(
  directory: SparseCM12LogicalOwnerDirectory,
  atlas: SparseAdaptiveMassAtlas,
): void {
  if (!sparseCM12LogicalOwnerHeaderValid(directory)) {
    throw new Error("logical-owner directory header is incomplete or invalid");
  }
  if (atlas.bricks.length !== directory.layout.residentBrickCount
    || atlas.brickDimensions.some((value, axis) =>
      value !== directory.layout.logicalBrickDimensions[axis])) {
    throw new Error("logical-owner directory does not describe this atlas");
  }
  for (let key = 0; key < directory.layout.logicalBrickCount; key += 1) {
    const coordinate = coordinateForKey(key, directory.layout.logicalBrickDimensions);
    const expected = sparseBrickContainingCoordinate(atlas, coordinate);
    const actual = sparseCM12LogicalOwnerAtKey(directory, key);
    const at = recordWord(directory.layout, key);
    const packed = directory.words[at + SPARSE_CM12_LOGICAL_OWNER_RECORD.ownerAndSpan]!;
    const originKey = directory.words[at + SPARSE_CM12_LOGICAL_OWNER_RECORD.originKey]!;
    if (!expected) {
      if (actual || packed !== SPARSE_CM12_LOGICAL_OWNER_INVALID
        || originKey !== SPARSE_CM12_LOGICAL_OWNER_INVALID) {
        throw new Error(`logical-owner key ${key} invents or corrupts an empty owner`);
      }
      continue;
    }
    const brick = atlas.bricks.indexOf(expected);
    if (!actual || actual.brick !== brick || actual.spanBricks !== sparseBrickSpan(expected)
      || actual.originKey !== expected.key) {
      throw new Error(`logical-owner key ${key} does not preserve span lookup semantics`);
    }
  }
}

/** CPU mirror of the WGSL compact-owner lookup. */
export function sparseCM12LogicalOwnerCellAtFine(
  directory: SparseCM12LogicalOwnerDirectory,
  position: SparseBrickVec3,
  finestDimensions: SparseBrickVec3,
  runtime: SparseCM12LogicalOwnerRuntime,
  requireActiveAndOpen = false,
): SparseCM12LogicalOwnerCell | undefined {
  if (position.some((value, axis) => !Number.isSafeInteger(value) || value < 0
    || value >= finestDimensions[axis])) return undefined;
  const brickFine = directory.layout.brickFineResolution;
  const logical = position.map((value) => Math.floor(value / brickFine)) as
    [number, number, number];
  const owner = sparseCM12LogicalOwnerAtCoordinate(directory, logical);
  if (!owner || (requireActiveAndOpen && !runtime.brickActive(owner.brick))) return undefined;
  const resolution = runtime.acceptedBrickResolution(owner.brick);
  if (!Number.isSafeInteger(resolution) || resolution <= 0
    || brickFine % resolution !== 0) return undefined;
  const range = runtime.templateBrickCellRange(owner.brick, resolution);
  const scale = brickFine * owner.spanBricks / resolution;
  if (!Number.isSafeInteger(scale) || scale <= 0) return undefined;
  const originFine = owner.origin.map((value) => value * brickFine) as
    [number, number, number];
  const local = position.map((value, axis) =>
    Math.floor((value - originFine[axis]!) / scale)) as [number, number, number];
  const valid = finestDimensions.map((value, axis) => Math.floor(Math.min(
    value - originFine[axis]! + scale - 1,
    brickFine * owner.spanBricks,
  ) / scale)) as [number, number, number];
  const offset = local[0] + valid[0] * (local[1] + valid[1] * local[2]);
  if (!Number.isSafeInteger(range[0]) || !Number.isSafeInteger(range[1])
    || offset < 0 || offset >= range[1]) return undefined;
  const cell = range[0] + offset;
  if (requireActiveAndOpen
    && (runtime.cellResolution?.(cell) !== resolution
      || !(runtime.cellOpenVolume?.(cell) ?? 0 > 1e-8))) return undefined;
  return { cell, brick: owner.brick, resolution };
}
