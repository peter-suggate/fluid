/**
 * BTI1: executable proof of a unified B8 brick/tile topology image.
 *
 * The hot domain is one stable 4^3 cell tile (`leaf * 8 + localTile`). Uniform
 * brick interiors are arithmetic. Cross-brick, mixed-rung, and sparse-air
 * faces live in a compact per-tile exception stream. A finest-lattice 4^3
 * directory provides one-load point ownership before brick-local arithmetic.
 *
 * BTI1 is deliberately a single accepted-slot proof. Production adoption must
 * add the existing shadow/validate/flip transaction; it must not weaken this
 * image's exact cell, face, and point-owner invariants.
 */

import type {
  SparseAtlasCompositeCell,
  SparseAtlasCompositeGrid,
  SparseAtlasGradientRow,
} from "./sparse-atlas-composite-projection";
import {
  sparseBrickSpan,
  type SparseAdaptiveMassAtlas,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";

export const SPARSE_CM12_BRICK_TILE_IMAGE_MAGIC = 0x4254_4931; // BTI1
export const SPARSE_CM12_BRICK_TILE_IMAGE_VERSION = 1;
export const SPARSE_CM12_BRICK_TILE_IMAGE_INVALID = 0xffff_ffff;
export const SPARSE_CM12_BRICK_TILE_IMAGE_HEADER_WORDS = 32;
export const SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS = 16;
export const SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS = 10;
export const SPARSE_CM12_BRICK_TILE_IMAGE_TILE_EDGE = 4;
export const SPARSE_CM12_BRICK_TILE_IMAGE_TILES_PER_LEAF = 8;
export const SPARSE_CM12_BRICK_TILE_IMAGE_FACE_FAMILIES = 6;
export const SPARSE_CM12_BRICK_TILE_IMAGE_FACE_MASK_WORDS = 12;
export const SPARSE_CM12_BRICK_TILE_IMAGE_EXCEPTION_WORDS = 2;

export const SPARSE_CM12_BRICK_TILE_IMAGE_FLAG = Object.freeze({
  complete: 1 << 0,
  validated: 1 << 1,
  active: 0x8000_0000,
} as const);

export const SPARSE_CM12_BRICK_TILE_IMAGE_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, brickWords: 3, tileWords: 4,
  faceFamilies: 5, leafCount: 6, tileCapacity: 7,
  spatialTilesX: 8, spatialTilesY: 9, spatialTilesZ: 10, generation: 11,
  brickBase: 12, tileBase: 13, faceMaskBase: 14, exceptionBase: 15,
  spatialOwnerBase: 16, totalWords: 17, cellCount: 18, rowCount: 19,
  exceptionCount: 20, flags: 21, finestX: 22, finestY: 23, finestZ: 24,
} as const);

export const SPARSE_CM12_BRICK_TILE_IMAGE_BRICK = Object.freeze({
  generation: 0, flags: 1, originX: 2, originY: 3, originZ: 4,
  cellFirst: 5, cellCount: 6, validDimensions: 7, scale: 8,
  tileBase: 9, tileCount: 10, rowBaseX: 11, rowBaseY: 12, rowBaseZ: 13,
  brickKey: 14, reserved: 15,
} as const);

export const SPARSE_CM12_BRICK_TILE_IMAGE_TILE = Object.freeze({
  generation: 0, flags: 1, leaf: 2, cellFirst: 3, counts: 4,
  strides: 5, validMaskLow: 6, validMaskHigh: 7,
  exceptionFirst: 8, exceptionCount: 9,
} as const);

export interface SparseCM12BrickTileImageLayout {
  readonly leafCount: number;
  readonly tileCapacity: number;
  readonly finestDimensions: SparseBrickVec3;
  readonly spatialTileDimensions: SparseBrickVec3;
  readonly spatialTileCapacity: number;
  readonly exceptionCount: number;
  readonly brickBaseWords: number;
  readonly tileBaseWords: number;
  readonly faceMaskBaseWords: number;
  readonly exceptionBaseWords: number;
  readonly spatialOwnerBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12BrickTileImage {
  readonly layout: SparseCM12BrickTileImageLayout;
  readonly words: Uint32Array;
}

export interface SparseCM12BrickTileMemoryReport {
  readonly headerBytes: number;
  readonly brickBytes: number;
  readonly tileBytes: number;
  readonly faceMaskBytes: number;
  readonly explicitFaceBytes: number;
  readonly spatialOwnerBytes: number;
  readonly totalBytes: number;
  readonly bytesPerCell: number;
}

export interface SparseCM12BrickTileValidationReceipt {
  readonly cellCount: number;
  readonly rowCount: number;
  readonly activeTileCount: number;
  readonly implicitInteriorRowCount: number;
  readonly explicitFaceRowCount: number;
  readonly explicitAddressCount: number;
  readonly explicitAddressCollisionCount: number;
  readonly finestPointCount: number;
  readonly memory: SparseCM12BrickTileMemoryReport;
}

interface TileDraft {
  readonly leaf: number;
  readonly first: number;
  readonly counts: SparseBrickVec3;
  readonly strides: readonly [number, number];
  readonly validLow: number;
  readonly validHigh: number;
  readonly exceptions: Array<readonly [address: number, row: number]>;
}

const align64 = (value: number) => Math.ceil(value / 64) * 64;
const checkedU32 = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} is outside the u32 range`);
  }
  return value;
};
const pack3x8 = (value: readonly number[]) =>
  checkedU32(value[0]! | (value[1]! << 8) | (value[2]! << 16), "packed vec3");
const unpack3x8 = (value: number): [number, number, number] =>
  [value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff];
const packStrides = (y: number, z: number) =>
  checkedU32(y | (z << 16), "packed strides");
const laneLocal = (lane: number): [number, number, number] =>
  [lane & 3, (lane >>> 2) & 3, lane >>> 4];
const tileLocal = (localTile: number): [number, number, number] =>
  [localTile & 1, (localTile >>> 1) & 1, localTile >>> 2];
const bitSet = (low: number, high: number, lane: number): boolean =>
  lane < 32 ? ((low >>> lane) & 1) !== 0 : ((high >>> (lane - 32)) & 1) !== 0;
const setBit = (pair: [number, number], lane: number): void => {
  if (lane < 32) pair[0] = (pair[0] | (1 << lane)) >>> 0;
  else pair[1] = (pair[1] | (1 << (lane - 32))) >>> 0;
};
const product = (value: readonly number[]) => value.reduce((a, b) => a * b, 1);

function cellAddress(
  cell: SparseAtlasCompositeCell,
  leafByKey: ReadonlyMap<number, number>,
): readonly [tile: number, lane: number] {
  const leaf = leafByKey.get(cell.brickKey);
  if (leaf === undefined) throw new Error(`BTI1 cell ${cell.id} has no leaf`);
  const tx = Math.floor(cell.local[0] / 4);
  const ty = Math.floor(cell.local[1] / 4);
  const tz = Math.floor(cell.local[2] / 4);
  const localTile = tx + 2 * (ty + 2 * tz);
  const lane = (cell.local[0] & 3) + 4 * ((cell.local[1] & 3)
    + 4 * (cell.local[2] & 3));
  return [leaf * SPARSE_CM12_BRICK_TILE_IMAGE_TILES_PER_LEAF + localTile, lane];
}

function rowOwner(row: SparseAtlasGradientRow, grid: SparseAtlasCompositeGrid):
readonly [cell: SparseAtlasCompositeCell, family: number] {
  const positive = row.terms.find((term) => term.coefficient > 0);
  if (positive) return [grid.cells[positive.cellId]!, row.axis];
  if (row.kind === "sparse-air" && row.terms.length === 1
    && row.terms[0]!.coefficient < 0) {
    return [grid.cells[row.terms[0]!.cellId]!, row.axis + 3];
  }
  throw new Error(`BTI1 row ${row.id} has no BFA-compatible owner`);
}

function validDimensions(cells: readonly SparseAtlasCompositeCell[]): SparseBrickVec3 {
  const result: [number, number, number] = [0, 0, 0];
  for (const cell of cells) for (let axis = 0; axis < 3; axis += 1) {
    result[axis] = Math.max(result[axis], cell.local[axis]! + 1);
  }
  return result;
}

function canonicalRowId(
  axis: number,
  local: SparseBrickVec3,
  dimensions: SparseBrickVec3,
  bases: readonly number[],
): number {
  const base = bases[axis]!;
  if (base === SPARSE_CM12_BRICK_TILE_IMAGE_INVALID || local[axis]! === 0) {
    return SPARSE_CM12_BRICK_TILE_IMAGE_INVALID;
  }
  if (axis === 0) return base + local[0] - 1
    + (dimensions[0] - 1) * (local[1] + dimensions[1] * local[2]);
  if (axis === 1) return base + local[0]
    + dimensions[0] * (local[1] - 1 + (dimensions[1] - 1) * local[2]);
  return base + local[0] + dimensions[0]
    * (local[1] + dimensions[1] * (local[2] - 1));
}

function createLayout(
  atlas: SparseAdaptiveMassAtlas,
  exceptionCount: number,
): SparseCM12BrickTileImageLayout {
  const leafCount = atlas.bricks.length;
  const tileCapacity = checkedU32(leafCount
    * SPARSE_CM12_BRICK_TILE_IMAGE_TILES_PER_LEAF, "BTI1 tile capacity");
  const spatialTileDimensions = atlas.dimensions.map((value) =>
    Math.ceil(value / SPARSE_CM12_BRICK_TILE_IMAGE_TILE_EDGE)) as
    [number, number, number];
  const spatialTileCapacity = checkedU32(product(spatialTileDimensions),
    "BTI1 spatial tile capacity");
  const brickBaseWords = align64(SPARSE_CM12_BRICK_TILE_IMAGE_HEADER_WORDS);
  const tileBaseWords = align64(brickBaseWords
    + leafCount * SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS);
  const faceMaskBaseWords = align64(tileBaseWords
    + tileCapacity * SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS);
  const exceptionBaseWords = align64(faceMaskBaseWords
    + tileCapacity * SPARSE_CM12_BRICK_TILE_IMAGE_FACE_MASK_WORDS);
  const spatialOwnerBaseWords = align64(exceptionBaseWords
    + exceptionCount * SPARSE_CM12_BRICK_TILE_IMAGE_EXCEPTION_WORDS);
  const totalWords = align64(spatialOwnerBaseWords + spatialTileCapacity);
  checkedU32(totalWords, "BTI1 total words");
  return Object.freeze({ leafCount, tileCapacity,
    finestDimensions: [...atlas.dimensions] as [number, number, number],
    spatialTileDimensions, spatialTileCapacity, exceptionCount,
    brickBaseWords, tileBaseWords, faceMaskBaseWords, exceptionBaseWords,
    spatialOwnerBaseWords, totalWords, totalBytes: 4 * totalWords });
}

/** Compile the exact accepted B8 composite topology into BTI1. */
export function compileSparseCM12BrickTileImage(
  grid: SparseAtlasCompositeGrid,
  generation = grid.atlas.generation,
): SparseCM12BrickTileImage {
  const atlas = grid.atlas;
  if (atlas.brickFineResolution !== 8) {
    throw new Error("BTI1 executable gate is intentionally restricted to production B8");
  }
  checkedU32(generation, "BTI1 generation");
  const leafByKey = new Map<number, number>();
  atlas.bricks.forEach((brick, leaf) => leafByKey.set(brick.key, leaf));
  const cellsByLeaf = Array.from({ length: atlas.bricks.length }, () =>
    [] as SparseAtlasCompositeCell[]);
  for (const cell of grid.cells) {
    const leaf = leafByKey.get(cell.brickKey);
    if (leaf === undefined) throw new Error(`BTI1 cell ${cell.id} references absent brick`);
    cellsByLeaf[leaf]!.push(cell);
  }

  const dimensionsByLeaf: SparseBrickVec3[] = [];
  const rowBasesByLeaf: Array<readonly [number, number, number]> = [];
  for (let leaf = 0; leaf < atlas.bricks.length; leaf += 1) {
    const cells = cellsByLeaf[leaf]!;
    const dimensions = validDimensions(cells);
    if (cells.length !== product(dimensions)) {
      throw new Error(`BTI1 leaf ${leaf} is not a dense clipped brick`);
    }
    dimensionsByLeaf.push(dimensions);
    const key = atlas.bricks[leaf]!.key;
    const bases: [number, number, number] = [
      SPARSE_CM12_BRICK_TILE_IMAGE_INVALID,
      SPARSE_CM12_BRICK_TILE_IMAGE_INVALID,
      SPARSE_CM12_BRICK_TILE_IMAGE_INVALID,
    ];
    for (const row of grid.gradientRows) {
      if (row.kind !== "intra-brick") continue;
      const [owner] = rowOwner(row, grid);
      if (owner.brickKey === key) bases[row.axis] = Math.min(bases[row.axis], row.id);
    }
    rowBasesByLeaf.push(bases);
  }

  const drafts: Array<TileDraft | undefined> = new Array(
    atlas.bricks.length * SPARSE_CM12_BRICK_TILE_IMAGE_TILES_PER_LEAF);
  for (let leaf = 0; leaf < atlas.bricks.length; leaf += 1) {
    const dimensions = dimensionsByLeaf[leaf]!;
    const cellFirst = cellsByLeaf[leaf]![0]?.id ?? SPARSE_CM12_BRICK_TILE_IMAGE_INVALID;
    for (let localTile = 0; localTile < 8; localTile += 1) {
      const tile = leaf * 8 + localTile;
      const tc = tileLocal(localTile);
      const origin = tc.map((value) => 4 * value) as [number, number, number];
      const counts = origin.map((value, axis) =>
        Math.max(0, Math.min(4, dimensions[axis]! - value))) as [number, number, number];
      if (counts.some((value) => value === 0)) continue;
      const first = cellFirst + origin[0] + dimensions[0]
        * (origin[1] + dimensions[1] * origin[2]);
      const valid: [number, number] = [0, 0];
      for (let lane = 0; lane < 64; lane += 1) {
        const local = laneLocal(lane);
        if (local.every((value, axis) => value < counts[axis]!)) setBit(valid, lane);
      }
      drafts[tile] = { leaf, first, counts,
        strides: [dimensions[0], dimensions[0] * dimensions[1]],
        validLow: valid[0], validHigh: valid[1], exceptions: [] };
    }
  }

  const faceMasks = Array.from({ length: drafts.length }, () =>
    new Uint32Array(SPARSE_CM12_BRICK_TILE_IMAGE_FACE_MASK_WORDS));
  let implicitInteriorRowCount = 0;
  for (const row of grid.gradientRows) {
    const [owner, family] = rowOwner(row, grid);
    const [tile, lane] = cellAddress(owner, leafByKey);
    const draft = drafts[tile];
    if (!draft || !bitSet(draft.validLow, draft.validHigh, lane)) {
      throw new Error(`BTI1 row ${row.id} owner is outside its active tile`);
    }
    const word = 2 * family + (lane >= 32 ? 1 : 0);
    faceMasks[tile]![word] = (faceMasks[tile]![word]!
      | (1 << (lane & 31))) >>> 0;
    if (row.kind === "intra-brick") {
      const leaf = draft.leaf;
      const expected = canonicalRowId(row.axis, owner.local,
        dimensionsByLeaf[leaf]!, rowBasesByLeaf[leaf]!);
      if (expected !== row.id) {
        throw new Error(`BTI1 intra row ${row.id} is not canonical (expected ${expected})`);
      }
      implicitInteriorRowCount += 1;
    } else {
      if (family < 3 && owner.local[family]! !== 0) {
        throw new Error(`BTI1 non-interior row ${row.id} is not owned on a brick boundary`);
      }
      draft.exceptions.push([family * 64 + lane, row.id]);
    }
  }
  for (const draft of drafts) draft?.exceptions.sort((a, b) => a[0] - b[0]
    || a[1] - b[1]);
  const exceptions = drafts.flatMap((draft) => draft?.exceptions ?? []);
  const layout = createLayout(atlas, exceptions.length);
  const words = new Uint32Array(layout.totalWords);
  words.fill(SPARSE_CM12_BRICK_TILE_IMAGE_INVALID, layout.spatialOwnerBaseWords,
    layout.spatialOwnerBaseWords + layout.spatialTileCapacity);
  words.set([
    SPARSE_CM12_BRICK_TILE_IMAGE_MAGIC, SPARSE_CM12_BRICK_TILE_IMAGE_VERSION,
    SPARSE_CM12_BRICK_TILE_IMAGE_HEADER_WORDS,
    SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS,
    SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS,
    SPARSE_CM12_BRICK_TILE_IMAGE_FACE_FAMILIES,
    layout.leafCount, layout.tileCapacity,
    ...layout.spatialTileDimensions, generation,
    layout.brickBaseWords, layout.tileBaseWords, layout.faceMaskBaseWords,
    layout.exceptionBaseWords, layout.spatialOwnerBaseWords, layout.totalWords,
    grid.cells.length, grid.gradientRows.length, layout.exceptionCount, 0,
    ...atlas.dimensions,
  ], 0);

  for (let leaf = 0; leaf < atlas.bricks.length; leaf += 1) {
    const brick = atlas.bricks[leaf]!;
    const cells = cellsByLeaf[leaf]!;
    const dimensions = dimensionsByLeaf[leaf]!;
    const spanFine = 8 * sparseBrickSpan(brick);
    if (spanFine % brick.resolution !== 0) {
      throw new Error(`BTI1 leaf ${leaf} has non-integral cell scale`);
    }
    const at = layout.brickBaseWords + leaf * SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS;
    const tileCount = drafts.slice(8 * leaf, 8 * leaf + 8)
      .reduce((count, value) => count + (value ? 1 : 0), 0);
    words.set([generation,
      (SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.active | brick.resolution) >>> 0,
      brick.coordinate[0] * 8, brick.coordinate[1] * 8, brick.coordinate[2] * 8,
      cells[0]?.id ?? SPARSE_CM12_BRICK_TILE_IMAGE_INVALID, cells.length,
      pack3x8(dimensions), spanFine / brick.resolution, 8 * leaf, tileCount,
      ...rowBasesByLeaf[leaf]!, brick.key, 0], at);
  }

  let exceptionFirst = 0;
  for (let tile = 0; tile < drafts.length; tile += 1) {
    const draft = drafts[tile];
    if (!draft) continue;
    const at = layout.tileBaseWords + tile * SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS;
    words.set([generation, SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.active, draft.leaf,
      draft.first, pack3x8(draft.counts), packStrides(...draft.strides),
      draft.validLow, draft.validHigh, exceptionFirst, draft.exceptions.length], at);
    words.set(faceMasks[tile]!, layout.faceMaskBaseWords
      + tile * SPARSE_CM12_BRICK_TILE_IMAGE_FACE_MASK_WORDS);
    for (const [address, row] of draft.exceptions) {
      words.set([address, row], layout.exceptionBaseWords
        + SPARSE_CM12_BRICK_TILE_IMAGE_EXCEPTION_WORDS * exceptionFirst++);
    }
  }

  const [sx, sy] = layout.spatialTileDimensions;
  for (let leaf = 0; leaf < atlas.bricks.length; leaf += 1) {
    const brick = atlas.bricks[leaf]!;
    const origin = brick.coordinate.map((value) => value * 8) as [number, number, number];
    const upper = origin.map((value, axis) => Math.min(atlas.dimensions[axis]!,
      value + 8 * sparseBrickSpan(brick))) as [number, number, number];
    for (let z = Math.floor(origin[2] / 4); z < Math.ceil(upper[2] / 4); z += 1)
      for (let y = Math.floor(origin[1] / 4); y < Math.ceil(upper[1] / 4); y += 1)
        for (let x = Math.floor(origin[0] / 4); x < Math.ceil(upper[0] / 4); x += 1) {
          const at = layout.spatialOwnerBaseWords + x + sx * (y + sy * z);
          if (words[at] !== SPARSE_CM12_BRICK_TILE_IMAGE_INVALID) {
            throw new Error(`BTI1 spatial tile ${x}/${y}/${z} has overlapping leaves`);
          }
          words[at] = leaf;
        }
  }
  words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.flags] =
    SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.complete;
  const image = Object.freeze({ layout, words });
  validateSparseCM12BrickTileImage(image, grid, { markValidated: true });
  return image;
}

function tileWord(image: SparseCM12BrickTileImage, tile: number): number {
  return image.layout.tileBaseWords + tile * SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS;
}

/** Resolve one stable tile lane to its compact cell, with no topology search. */
export function sparseCM12BrickTileCell(
  image: SparseCM12BrickTileImage,
  tile: number,
  lane: number,
): number | undefined {
  if (!Number.isSafeInteger(tile) || tile < 0 || tile >= image.layout.tileCapacity
    || !Number.isSafeInteger(lane) || lane < 0 || lane >= 64) return undefined;
  const at = tileWord(image, tile), words = image.words;
  const local = laneLocal(lane);
  const counts = unpack3x8(words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.counts]!);
  if (local.some((value, axis) => value >= counts[axis]!)) return undefined;
  const strides = words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.strides]!;
  const first = words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.cellFirst]!;
  if (first === SPARSE_CM12_BRICK_TILE_IMAGE_INVALID) return undefined;
  return first
    + local[0] + (strides & 0xffff) * local[1] + (strides >>> 16) * local[2];
}

/** Resolve an integer finest-lattice point through one 4^3 owner load. */
export function sparseCM12BrickTileCellAtFine(
  image: SparseCM12BrickTileImage,
  position: SparseBrickVec3,
): number | undefined {
  const [sx, sy] = image.layout.spatialTileDimensions;
  const finest = [
    image.words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.finestX]!,
    image.words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.finestY]!,
    image.words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.finestZ]!,
  ];
  if (position.some((value) => !Number.isSafeInteger(value) || value < 0)
    || position.some((value, axis) => value >= finest[axis]!)) {
    return undefined;
  }
  const tileCoordinate = position.map((value) => Math.floor(value / 4)) as
    [number, number, number];
  const ownerAt = image.layout.spatialOwnerBaseWords + tileCoordinate[0]
    + sx * (tileCoordinate[1] + sy * tileCoordinate[2]);
  const leaf = image.words[ownerAt]!;
  if (leaf === SPARSE_CM12_BRICK_TILE_IMAGE_INVALID || leaf >= image.layout.leafCount) {
    return undefined;
  }
  const at = image.layout.brickBaseWords
    + leaf * SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS;
  const words = image.words;
  const scale = words[at + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.scale]!;
  const origin = [words[at + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.originX]!,
    words[at + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.originY]!,
    words[at + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.originZ]!] as const;
  const local = position.map((value, axis) =>
    Math.floor((value - origin[axis]!) / scale)) as [number, number, number];
  const dimensions = unpack3x8(
    words[at + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.validDimensions]!,
  );
  if (local.some((value, axis) => value < 0 || value >= dimensions[axis]!)) {
    return undefined;
  }
  return words[at + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.cellFirst]!
    + local[0] + dimensions[0] * (local[1] + dimensions[1] * local[2]);
}

/**
 * Resolve every row owned by a face address. The common interior path is pure
 * arithmetic; only a brick boundary scans its compact per-tile exception run.
 */
export function sparseCM12BrickTileRows(
  image: SparseCM12BrickTileImage,
  tile: number,
  family: number,
  lane: number,
): readonly number[] {
  const cell = sparseCM12BrickTileCell(image, tile, lane);
  if (cell === undefined || family < 0 || family >= 6) return [];
  const words = image.words, tileAt = tileWord(image, tile);
  const maskAt = image.layout.faceMaskBaseWords
    + tile * SPARSE_CM12_BRICK_TILE_IMAGE_FACE_MASK_WORDS
    + 2 * family + (lane >= 32 ? 1 : 0);
  if (((words[maskAt]! >>> (lane & 31)) & 1) === 0) return [];
  const result: number[] = [];
  const leaf = words[tileAt + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.leaf]!;
  const brickAt = image.layout.brickBaseWords
    + leaf * SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS;
  const dimensions = unpack3x8(
    words[brickAt + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.validDimensions]!,
  );
  const localTile = tile - leaf * SPARSE_CM12_BRICK_TILE_IMAGE_TILES_PER_LEAF;
  const tc = tileLocal(localTile), ll = laneLocal(lane);
  const local = ll.map((value, axis) => value + 4 * tc[axis]!) as
    [number, number, number];
  if (family < 3 && local[family]! > 0) {
    const bases = [words[brickAt + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.rowBaseX]!,
      words[brickAt + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.rowBaseY]!,
      words[brickAt + SPARSE_CM12_BRICK_TILE_IMAGE_BRICK.rowBaseZ]!] as const;
    const row = canonicalRowId(family, local, dimensions, bases);
    if (row !== SPARSE_CM12_BRICK_TILE_IMAGE_INVALID) return [row];
  }
  const address = family * 64 + lane;
  const first = words[tileAt + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.exceptionFirst]!;
  const count = words[tileAt + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.exceptionCount]!;
  for (let index = 0; index < count; index += 1) {
    const at = image.layout.exceptionBaseWords
      + SPARSE_CM12_BRICK_TILE_IMAGE_EXCEPTION_WORDS * (first + index);
    const candidate = words[at]!;
    if (candidate > address) break;
    if (candidate === address) result.push(words[at + 1]!);
  }
  return result;
}

export function sparseCM12BrickTileMemoryReport(
  image: SparseCM12BrickTileImage,
  cellCount: number,
): SparseCM12BrickTileMemoryReport {
  const l = image.layout;
  const report = {
    headerBytes: 4 * l.brickBaseWords,
    brickBytes: 4 * l.leafCount * SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS,
    tileBytes: 4 * l.tileCapacity * SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS,
    faceMaskBytes: 4 * l.tileCapacity * SPARSE_CM12_BRICK_TILE_IMAGE_FACE_MASK_WORDS,
    explicitFaceBytes: 4 * l.exceptionCount * SPARSE_CM12_BRICK_TILE_IMAGE_EXCEPTION_WORDS,
    spatialOwnerBytes: 4 * l.spatialTileCapacity,
    totalBytes: l.totalBytes,
    bytesPerCell: cellCount > 0 ? l.totalBytes / cellCount : 0,
  };
  return Object.freeze(report);
}

/** Adversarial exhaustive validator used as the architecture's first gate. */
export function validateSparseCM12BrickTileImage(
  image: SparseCM12BrickTileImage,
  grid: SparseAtlasCompositeGrid,
  options: { readonly markValidated?: boolean; readonly maximumDensePoints?: number } = {},
): SparseCM12BrickTileValidationReceipt {
  const { words, layout } = image;
  if (words[0] !== SPARSE_CM12_BRICK_TILE_IMAGE_MAGIC
    || words[1] !== SPARSE_CM12_BRICK_TILE_IMAGE_VERSION
    || words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.totalWords] !== layout.totalWords
    || words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.cellCount] !== grid.cells.length
    || words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.rowCount] !== grid.gradientRows.length) {
    throw new Error("BTI1 header or source cardinality mismatch");
  }
  const seenCells = new Uint8Array(grid.cells.length);
  let activeTileCount = 0;
  for (let tile = 0; tile < layout.tileCapacity; tile += 1) {
    const at = tileWord(image, tile);
    if ((words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.flags]!
      & SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.active) === 0) continue;
    activeTileCount += 1;
    for (let lane = 0; lane < 64; lane += 1) {
      const cell = sparseCM12BrickTileCell(image, tile, lane);
      if (cell === undefined) continue;
      if (cell >= grid.cells.length || seenCells[cell] !== 0) {
        throw new Error(`BTI1 cell enumeration duplicates or invents cell ${cell}`);
      }
      seenCells[cell] = 1;
    }
  }
  if (seenCells.some((value) => value !== 1)) {
    throw new Error("BTI1 tile enumeration does not cover every compact cell exactly once");
  }

  const seenRows = new Uint8Array(grid.gradientRows.length);
  const explicitAddresses = new Set<number>();
  let explicitFaceRowCount = 0;
  for (let tile = 0; tile < layout.tileCapacity; tile += 1) {
    for (let family = 0; family < 6; family += 1) for (let lane = 0; lane < 64; lane += 1) {
      const rows = sparseCM12BrickTileRows(image, tile, family, lane);
      for (const row of rows) {
        if (row >= grid.gradientRows.length || seenRows[row] !== 0) {
          throw new Error(`BTI1 face enumeration duplicates or invents row ${row}`);
        }
        seenRows[row] = 1;
        if (grid.gradientRows[row]!.kind !== "intra-brick") {
          explicitFaceRowCount += 1;
          explicitAddresses.add(tile * 384 + family * 64 + lane);
        }
      }
    }
  }
  if (seenRows.some((value) => value !== 1)) {
    throw new Error("BTI1 face enumeration does not cover every row exactly once");
  }

  const finestPointCount = product(grid.atlas.dimensions);
  if (finestPointCount > (options.maximumDensePoints ?? 32 * 1024 * 1024)) {
    throw new RangeError(`BTI1 dense point oracle requires ${finestPointCount} points`);
  }
  const dense = new Uint32Array(finestPointCount);
  dense.fill(SPARSE_CM12_BRICK_TILE_IMAGE_INVALID);
  const [dx, dy] = grid.atlas.dimensions;
  for (const cell of grid.cells) {
    for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1)
      for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1)
        for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
          const at = x + dx * (y + dy * z);
          if (dense[at] !== SPARSE_CM12_BRICK_TILE_IMAGE_INVALID) {
            throw new Error(`BTI1 dense oracle detects overlapping cell ${cell.id}`);
          }
          dense[at] = cell.id;
        }
  }
  for (let z = 0; z < grid.atlas.dimensions[2]; z += 1)
    for (let y = 0; y < grid.atlas.dimensions[1]; y += 1)
      for (let x = 0; x < grid.atlas.dimensions[0]; x += 1) {
        const expected = dense[x + dx * (y + dy * z)]!;
        const actual = sparseCM12BrickTileCellAtFine(image, [x, y, z])
          ?? SPARSE_CM12_BRICK_TILE_IMAGE_INVALID;
        if (actual !== expected) {
          throw new Error(`BTI1 point owner differs at ${x}/${y}/${z}: ${actual} != ${expected}`);
        }
      }
  if (options.markValidated) {
    words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.flags] =
      SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.complete
      | SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.validated;
  }
  return Object.freeze({ cellCount: grid.cells.length,
    rowCount: grid.gradientRows.length, activeTileCount,
    implicitInteriorRowCount: grid.gradientRows.length - explicitFaceRowCount,
    explicitFaceRowCount, explicitAddressCount: explicitAddresses.size,
    explicitAddressCollisionCount: explicitFaceRowCount - explicitAddresses.size,
    finestPointCount, memory: sparseCM12BrickTileMemoryReport(image, grid.cells.length) });
}
