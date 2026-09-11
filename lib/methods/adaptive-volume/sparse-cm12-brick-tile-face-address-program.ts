/**
 * BFA1: immutable all-rung face execution addresses for production B8.
 *
 * Rows remain selected by the accepted IBO slot and resolved through ITR1.
 * BFA1 only separates the physical execution order into interior tiles and
 * explicit boundary addresses. It is therefore valid across rerungs and
 * activation changes and does not duplicate accepted row authority.
 */

import {
  SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS,
  type SparseCM12InternedBoundaryCompilation,
} from "./sparse-cm12-interned-boundary-operators";
import { SPARSE_CM12_FACTORED_AEI_INVALID } from
  "./sparse-cm12-factored-aei-topology";
import type { SparseCM12BrickTileFaceProgram } from
  "./sparse-cm12-brick-tile-face-program";

export const SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_MAGIC = 0x4246_4131; // BFA1
export const SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_VERSION = 2;
export const SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_HEADER_WORDS = 16;
export const SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_INTERIOR_WORDS = 1;
export const SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_SEAM_WORDS = 1;
export const SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_PORTS_PER_PACKET = 64;

export interface SparseCM12BrickTileFaceAddressLayout {
  readonly baseWords: number;
  readonly leafCapacity: number;
  readonly interiorTileCount: number;
  readonly seamAddressCount: number;
  readonly seamPacketCount: number;
  readonly interiorBaseWords: number;
  readonly seamBaseWords: number;
  readonly dispatchWidth: number;
  readonly interiorDispatchRows: number;
  readonly seamDispatchRows: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12BrickTileFaceAddressProgram {
  readonly layout: SparseCM12BrickTileFaceAddressLayout;
  /** Region starting at layout.baseWords; embedded offsets remain absolute. */
  readonly words: Uint32Array;
}

const align64 = (value: number) => Math.ceil(value / 64) * 64;
const f32 = (bits: number): number => {
  const buffer = new ArrayBuffer(4);new Uint32Array(buffer)[0] = bits;
  return new Float32Array(buffer)[0]!;
};
const localFromOrdinal = (ordinal: number, dimensions: readonly number[]) => {
  const z = Math.floor(ordinal / (dimensions[0]! * dimensions[1]!));
  const remainder = ordinal - z * dimensions[0]! * dimensions[1]!;
  const y = Math.floor(remainder / dimensions[0]!);
  return [remainder - y * dimensions[0]!, y, z] as const;
};
const addressFor = (leaf: number, local: readonly number[], family: number) => {
  if (local.some((value) => value < 0 || value >= 8)) {
    throw new Error(`BFA1 leaf ${leaf} local address is outside B8`);
  }
  const localTile = Math.floor(local[0]! / 4)
    + 2 * (Math.floor(local[1]! / 4) + 2 * Math.floor(local[2]! / 4));
  const lane = (local[0]! & 3) + 4 * ((local[1]! & 3) + 4 * (local[2]! & 3));
  return [8 * leaf + localTile, 64 * family + lane] as const;
};
const packSeamAddress = (address: readonly [number, number]): number => {
  if (address[0] >= 0x0080_0000 || address[1] >= 0x200) {
    throw new RangeError(`BFA1 seam address ${address.join("/")} exceeds packed ABI`);
  }
  return (address[0] * 0x200 + address[1]) >>> 0;
};

export function compileSparseCM12BrickTileFaceAddressProgram(options: Readonly<{
  ibo: SparseCM12InternedBoundaryCompilation;
  baseWords?: number;
}>): SparseCM12BrickTileFaceAddressProgram {
  const { ibo } = options, baseWords = align64(options.baseWords ?? 0);
  const interior = new Set<number>();
  for (const descriptor of ibo.catalog.canonical) {
    if (!descriptor.certified) continue;
    const axis = descriptor.validDimensions.map((value) => Math.ceil(value / 4));
    for (let z = 0; z < axis[2]!; z += 1) for (let y = 0; y < axis[1]!; y += 1)
      for (let x = 0; x < axis[0]!; x += 1) {
        const origin = [4 * x, 4 * y, 4 * z];
        const hasInterior = [0, 1, 2].some((faceAxis) =>
          Math.max(0, Math.min(4, descriptor.validDimensions[faceAxis]! - origin[faceAxis]!))
            > (origin[faceAxis] === 0 ? 1 : 0));
        if (hasInterior) interior.add(8 * descriptor.leafId + x + 2 * (y + 2 * z));
      }
  }

  const seams = new Map<string, readonly [number, number]>();
  for (const patch of ibo.catalog.patches) {
    const template = ibo.templates[ibo.templateIdByPatch[patch.id]!]!;
    const termBase = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
      + SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS * template.rowCount;
    for (let localRow = 0; localRow < template.rowCount; localRow += 1) {
      const rowAt = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
        + SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS * localRow;
      const packed = template.words[rowAt + 1]!;
      const first = packed & 0x007f_ffff, count = packed >>> 23;
      const metadata = template.words[rowAt + 2]!;
      const axis = metadata >>> 30, kind = (metadata >>> 28) & 3;
      let ownerTerm = -1;
      for (let term = 0; term < count; term += 1) {
        if (f32(template.words[termBase + 2 * (first + term) + 1]!) > 0) {
          ownerTerm = term;break;
        }
      }
      let family = axis;
      if (ownerTerm < 0) {
        if (kind !== 3 || count !== 1
          || f32(template.words[termBase + 2 * first + 1]!) >= 0) {
          throw new Error(`BFA1 template ${template.id} row ${localRow} has no owner`);
        }
        ownerTerm = 0;family += 3;
      }
      const normalized = template.words[termBase + 2 * (first + ownerTerm)]!;
      const target = (normalized & 0x8000_0000) !== 0;
      const leaf = target ? patch.targetLeaf : patch.sourceLeaf;
      if (leaf === SPARSE_CM12_FACTORED_AEI_INVALID) {
        throw new Error(`BFA1 template ${template.id} assigns a row to sparse air`);
      }
      const dimensions = target ? template.targetDimensions : template.sourceDimensions;
      const local = localFromOrdinal(normalized & 0x7fff_ffff, dimensions);
      const address = addressFor(leaf, local, family);
      seams.set(`${address[0]}/${address[1]}`, address);
    }
  }
  const interiorTiles = [...interior].sort((a, b) => a - b);
  const seamAddresses = [...seams.values()].sort((left, right) =>
    left[0] - right[0] || left[1] - right[1]);
  const interiorBaseWords = baseWords + SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_HEADER_WORDS;
  const seamBaseWords = align64(interiorBaseWords + interiorTiles.length);
  const totalWords = align64(seamBaseWords
    + SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_SEAM_WORDS * seamAddresses.length);
  const dispatchWidth = 65_535;
  const layout = Object.freeze({ baseWords, leafCapacity: ibo.layout.leafCapacity,
    interiorTileCount: interiorTiles.length, seamAddressCount: seamAddresses.length,
    seamPacketCount: Math.ceil(seamAddresses.length
      / SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_PORTS_PER_PACKET),
    interiorBaseWords, seamBaseWords, dispatchWidth,
    interiorDispatchRows: Math.ceil(interiorTiles.length / dispatchWidth),
    seamDispatchRows: Math.ceil(Math.ceil(seamAddresses.length / 64) / dispatchWidth),
    totalWords, totalBytes: 4 * (totalWords - baseWords) });
  const words = new Uint32Array(totalWords - baseWords);
  const put = (absolute: number, values: readonly number[]) =>
    words.set(values, absolute - baseWords);
  put(baseWords, [SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_MAGIC,
    SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_VERSION,
    SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_HEADER_WORDS,
    SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_INTERIOR_WORDS,
    SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_SEAM_WORDS,
    SPARSE_CM12_BRICK_TILE_FACE_ADDRESS_PORTS_PER_PACKET,
    ibo.layout.leafCapacity, interiorTiles.length, seamAddresses.length,
    layout.seamPacketCount, interiorBaseWords, seamBaseWords, totalWords, 3]);
  put(interiorBaseWords, interiorTiles);
  seamAddresses.forEach((address, index) =>
    put(seamBaseWords + index, [packSeamAddress(address)]));
  return Object.freeze({ layout, words });
}

/** Prove that a concrete accepted BFP1 partition is covered by the all-rung view. */
export function validateSparseCM12BrickTileFaceAddressCoverage(
  program: SparseCM12BrickTileFaceAddressProgram,
  accepted: SparseCM12BrickTileFaceProgram,
): Readonly<{ interiorCovered: number; seamCovered: number }> {
  const interior = new Set<number>();
  for (let index = 0; index < program.layout.interiorTileCount; index += 1) {
    interior.add(program.words[program.layout.interiorBaseWords
      - program.layout.baseWords + index]!);
  }
  const seams = new Set<string>();
  for (let index = 0; index < program.layout.seamAddressCount; index += 1) {
    const at = program.layout.seamBaseWords - program.layout.baseWords + index;
    const packed = program.words[at]!;
    seams.add(`${packed >>> 9}/${packed & 0x1ff}`);
  }
  for (let index = 0; index < accepted.layout.interiorTileCount; index += 1) {
    const tile = accepted.words[accepted.layout.interiorBaseWords + index]!;
    if (!interior.has(tile)) throw new Error(`BFA1 omits accepted interior tile ${tile}`);
  }
  for (let index = 0; index < accepted.layout.seamPortCount; index += 1) {
    const at = accepted.layout.seamBaseWords + 3 * index;
    const key = `${accepted.words[at]!}/${accepted.words[at + 1]!}`;
    if (!seams.has(key)) throw new Error(`BFA1 omits accepted seam address ${key}`);
  }
  return Object.freeze({ interiorCovered: accepted.layout.interiorTileCount,
    seamCovered: accepted.layout.seamPortCount });
}
