/**
 * BFP1: compact physical face-execution view for one accepted BTI1 epoch.
 *
 * Stable tile identity remains in BTI1. BFP1 publishes a compact immutable
 * interior-tile order plus a compact explicit port stream for every non-local
 * face. Dynamic row selection stays in mask planes keyed by the stable tile
 * address; compiling this view does not create another topology authority.
 */

import type { SparseAtlasCompositeGrid } from "./sparse-atlas-composite-projection";
import {
  SPARSE_CM12_BRICK_TILE_IMAGE_INVALID,
  sparseCM12BrickTileRows,
  type SparseCM12BrickTileImage,
} from "./sparse-cm12-brick-tile-image";

export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_MAGIC = 0x4246_5031; // BFP1
export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_VERSION = 1;
export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER_WORDS = 16;
export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_INTERIOR_WORDS = 1;
export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM_WORDS = 3;
export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_PORTS_PER_PACKET = 64;

export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_FLAG = Object.freeze({
  complete: 1 << 0,
  validated: 1 << 1,
} as const);

export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, interiorWords: 3, seamWords: 4,
  portsPerPacket: 5, generation: 6, interiorTileCount: 7, seamPortCount: 8,
  seamPacketCount: 9, interiorBase: 10, seamBase: 11, totalWords: 12,
  flags: 13,
} as const);

export const SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM = Object.freeze({
  stableTile: 0,
  /** `family * 64 + lane`; family is in [0, 6). */
  address: 1,
  row: 2,
} as const);

export interface SparseCM12BrickTileFaceProgramLayout {
  readonly generation: number;
  readonly interiorTileCount: number;
  readonly seamPortCount: number;
  readonly seamPacketCount: number;
  readonly interiorBaseWords: number;
  readonly seamBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12BrickTileFaceProgram {
  readonly layout: SparseCM12BrickTileFaceProgramLayout;
  readonly words: Uint32Array;
}

export interface SparseCM12BrickTileFaceProgramReceipt {
  readonly rowCount: number;
  readonly interiorRowCount: number;
  readonly seamRowCount: number;
  readonly interiorTileCount: number;
  readonly seamPacketCount: number;
  readonly invokedInteriorLanes: number;
  readonly invokedSeamLanes: number;
  readonly invokedLaneCount: number;
  readonly rowsPerInvokedLane: number;
  readonly totalBytes: number;
  readonly bytesPerRow: number;
}

const align64 = (value: number) => Math.ceil(value / 64) * 64;
const checkedU32 = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} is outside the u32 range`);
  }
  return value;
};

/** Compile compact physical execution order from the exact BTI1 row service. */
export function compileSparseCM12BrickTileFaceProgram(
  image: SparseCM12BrickTileImage,
  grid: SparseAtlasCompositeGrid,
  generation = grid.atlas.generation,
): SparseCM12BrickTileFaceProgram {
  checkedU32(generation, "BFP1 generation");
  const interiorTiles: number[] = [];
  const seams: Array<readonly [stableTile: number, address: number, row: number]> = [];
  for (let tile = 0; tile < image.layout.tileCapacity; tile += 1) {
    let hasInterior = false;
    for (let family = 0; family < 6; family += 1) for (let lane = 0; lane < 64; lane += 1) {
      for (const row of sparseCM12BrickTileRows(image, tile, family, lane)) {
        if (grid.gradientRows[row]!.kind === "intra-brick") hasInterior = true;
        else seams.push([tile, family * 64 + lane, row]);
      }
    }
    if (hasInterior) interiorTiles.push(tile);
  }
  seams.sort((left, right) => left[0] - right[0] || left[1] - right[1]
    || left[2] - right[2]);
  const interiorBaseWords = align64(SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER_WORDS);
  const seamBaseWords = align64(interiorBaseWords
    + SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_INTERIOR_WORDS * interiorTiles.length);
  const totalWords = align64(seamBaseWords
    + SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM_WORDS * seams.length);
  checkedU32(totalWords, "BFP1 total words");
  const layout = Object.freeze({ generation,
    interiorTileCount: interiorTiles.length, seamPortCount: seams.length,
    seamPacketCount: Math.ceil(seams.length
      / SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_PORTS_PER_PACKET),
    interiorBaseWords, seamBaseWords, totalWords, totalBytes: 4 * totalWords });
  const words = new Uint32Array(totalWords);
  words.set([SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_MAGIC,
    SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_VERSION,
    SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER_WORDS,
    SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_INTERIOR_WORDS,
    SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM_WORDS,
    SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_PORTS_PER_PACKET,
    generation, interiorTiles.length, seams.length, layout.seamPacketCount,
    interiorBaseWords, seamBaseWords, totalWords, 0], 0);
  words.set(interiorTiles, interiorBaseWords);
  seams.forEach((seam, index) => words.set(seam,
    seamBaseWords + SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM_WORDS * index));
  words[SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER.flags] =
    SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_FLAG.complete;
  const program = Object.freeze({ layout, words });
  validateSparseCM12BrickTileFaceProgram(program, image, grid, true);
  return program;
}

export function sparseCM12BrickTileFaceProgramInteriorTile(
  program: SparseCM12BrickTileFaceProgram,
  ordinal: number,
): number | undefined {
  if (!Number.isSafeInteger(ordinal) || ordinal < 0
    || ordinal >= program.layout.interiorTileCount) return undefined;
  const tile = program.words[program.layout.interiorBaseWords + ordinal]!;
  return tile === SPARSE_CM12_BRICK_TILE_IMAGE_INVALID ? undefined : tile;
}

export interface SparseCM12BrickTileSeamPort {
  readonly stableTile: number;
  readonly family: number;
  readonly lane: number;
  readonly row: number;
}

export function sparseCM12BrickTileFaceProgramSeamPort(
  program: SparseCM12BrickTileFaceProgram,
  ordinal: number,
): SparseCM12BrickTileSeamPort | undefined {
  if (!Number.isSafeInteger(ordinal) || ordinal < 0
    || ordinal >= program.layout.seamPortCount) return undefined;
  const at = program.layout.seamBaseWords
    + SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM_WORDS * ordinal;
  const address = program.words[at + SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM.address]!;
  return Object.freeze({
    stableTile: program.words[at + SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM.stableTile]!,
    family: Math.floor(address / 64), lane: address & 63,
    row: program.words[at + SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_SEAM.row]!,
  });
}

export function validateSparseCM12BrickTileFaceProgram(
  program: SparseCM12BrickTileFaceProgram,
  image: SparseCM12BrickTileImage,
  grid: SparseAtlasCompositeGrid,
  markValidated = false,
): SparseCM12BrickTileFaceProgramReceipt {
  const { words, layout } = program;
  if (words[0] !== SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_MAGIC
    || words[1] !== SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_VERSION
    || words[SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER.totalWords] !== layout.totalWords) {
    throw new Error("BFP1 header mismatch");
  }
  const seenRows = new Uint8Array(grid.gradientRows.length);
  let interiorRowCount = 0;
  const seenTiles = new Set<number>();
  for (let ordinal = 0; ordinal < layout.interiorTileCount; ordinal += 1) {
    const tile = sparseCM12BrickTileFaceProgramInteriorTile(program, ordinal)!;
    if (tile >= image.layout.tileCapacity || seenTiles.has(tile)) {
      throw new Error(`BFP1 interior ordinal ${ordinal} duplicates or invents tile ${tile}`);
    }
    seenTiles.add(tile);
    let tileHasInterior = false;
    for (let family = 0; family < 3; family += 1) for (let lane = 0; lane < 64; lane += 1) {
      for (const row of sparseCM12BrickTileRows(image, tile, family, lane)) {
        if (grid.gradientRows[row]!.kind !== "intra-brick") continue;
        tileHasInterior = true;
        if (seenRows[row] !== 0) throw new Error(`BFP1 duplicates interior row ${row}`);
        seenRows[row] = 1;interiorRowCount += 1;
      }
    }
    if (!tileHasInterior) throw new Error(`BFP1 interior tile ${tile} is empty`);
  }
  for (let ordinal = 0; ordinal < layout.seamPortCount; ordinal += 1) {
    const port = sparseCM12BrickTileFaceProgramSeamPort(program, ordinal)!;
    if (port.stableTile >= image.layout.tileCapacity || port.family >= 6
      || port.lane >= 64 || port.row >= grid.gradientRows.length
      || grid.gradientRows[port.row]!.kind === "intra-brick") {
      throw new Error(`BFP1 seam port ${ordinal} is malformed`);
    }
    const exact = sparseCM12BrickTileRows(image, port.stableTile, port.family, port.lane);
    if (!exact.includes(port.row) || seenRows[port.row] !== 0) {
      throw new Error(`BFP1 seam port ${ordinal} duplicates or loses row ${port.row}`);
    }
    seenRows[port.row] = 1;
  }
  if (seenRows.some((value) => value !== 1)) {
    throw new Error("BFP1 interior/seam streams do not partition every row exactly once");
  }
  if (markValidated) words[SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER.flags] =
    SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_FLAG.complete
      | SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_FLAG.validated;
  const invokedInteriorLanes = 64 * layout.interiorTileCount;
  const invokedSeamLanes = 64 * layout.seamPacketCount;
  const invokedLaneCount = invokedInteriorLanes + invokedSeamLanes;
  return Object.freeze({ rowCount: grid.gradientRows.length, interiorRowCount,
    seamRowCount: layout.seamPortCount, interiorTileCount: layout.interiorTileCount,
    seamPacketCount: layout.seamPacketCount, invokedInteriorLanes, invokedSeamLanes,
    invokedLaneCount, rowsPerInvokedLane: grid.gradientRows.length / invokedLaneCount,
    totalBytes: layout.totalBytes,
    bytesPerRow: grid.gradientRows.length > 0
      ? layout.totalBytes / grid.gradientRows.length : 0 });
}
