/**
 * Per-brick identity and occupancy for a renderer-only sparse-brick world.
 *
 * The full payload spends four bytes on a packed `material | owner << 16` word
 * for every one of a brick's 512 voxels. It does not need to. Censused on the
 * hero garden at a 6.25 mm lattice (`tmp/owner-census.ts`, reproduced 2026-08):
 *
 *     occupied bricks        10 625
 *     single-owner            9 740   91.67 %
 *     two owners                798
 *     three owners               85
 *     four owners                 2
 *     mean owners/brick        1.092
 *
 * No brick in that scene holds more than four distinct identities, and nine in
 * ten hold exactly one. So the lane is not per-voxel data at all — it is a tiny
 * per-brick palette plus, for the 8.3 % that need one, a two-bit index.
 *
 * Four layouts, chosen per brick and tagged in the header word:
 *
 *   `empty`    no occupied voxel. Header only, 8 B.
 *   `full`     every voxel occupied by one identity — an interior. Header only,
 *              8 B: the mask is all-ones and is implied by the kind, so the 64 B
 *              a constant bitmask would cost is not spent.
 *   `uniform`  one identity over an occupancy bitmask. 8 B header + 64 B mask.
 *   `palette`  2..4 identities, 2-bit index per voxel. 8 B header + 64 B mask
 *              + 16 B palette + 128 B index = 216 B.
 *
 * `full` is the interior of every solid in the scene, and it is the reason a
 * buried brick does not have to be deleted to stop being expensive: it still
 * answers "solid" to every reader, for the price of the header. `empty` and
 * `full` are the two ends of the same statement and cost the same.
 *
 * The exception path matters even though this scene never takes it: palette
 * width is a property of the scene and the lattice, not of the format, and a
 * denser scene or a coarser brick will exceed four. `classify` reports overflow
 * rather than silently losing an identity, and the caller decides — today by
 * retaining a full per-voxel lane for those bricks, which at this measured
 * distribution costs nothing because the set is empty.
 *
 * Occupancy is one bit per voxel and is derived, not authored: the voxeliser
 * writes an identity only where the cell's solid fraction is positive
 * (`lib/webgpu-sparse-scene-proxies.ts:1974`), so `material != 0` — the
 * solidity predicate the dry primary already uses at
 * `lib/webgpu-svo-dry-scene.ts:4944` — and `fraction > 0` are the same test.
 * That equivalence is what lets a bitmask replace the lane without changing
 * what any consumer decides.
 *
 * Everything here is device-free so the packing is unit-testable on CPU.
 */

import { SPARSE_BRICK_NO_OWNER, packMaterialOwner, unpackMaterialOwner } from "./sparse-brick-octree";

/** Air. Written wherever the voxeliser found no coverage. */
export const SPARSE_BRICK_DRY_EMPTY_IDENTITY = packMaterialOwner(0, SPARSE_BRICK_NO_OWNER);

export const SPARSE_BRICK_DRY_PAYLOAD = Object.freeze({
  brickSize: 8,
  voxelsPerBrick: 512,
  /** 512 bits. */
  occupancyWords: 16,
  occupancyBytes: 64,
  /** kind + count, then the dominant identity. */
  headerWords: 2,
  headerBytes: 8,
  /** Two index bits per voxel caps the palette at four entries. */
  paletteIndexBits: 2,
  maximumPaletteEntries: 4,
  paletteBytes: 16,
  indexWords: 32,
  indexBytes: 128,
  /**
   * Two header bits, and the three shipped codes keep the values they had.
   * `full` takes the one remaining code rather than widening the field, so an
   * unpacker written against the old header still reads the old three.
   */
  kinds: { empty: 0, uniform: 1, palette: 2, full: 3 } as const,
  /** Set in the header when `classify` had to fall back to a full lane. */
  overflowBit: 0x8000_0000,
});

export type SparseBrickDryLayoutKind = "empty" | "full" | "uniform" | "palette" | "overflow";

export interface SparseBrickDryClassification {
  kind: SparseBrickDryLayoutKind;
  /** Distinct non-air identities, in first-seen scan order. */
  palette: readonly number[];
  occupiedVoxelCount: number;
  /** Bytes this brick costs in the chosen layout, header included. */
  byteLength: number;
}

function checkBrick(identities: ArrayLike<number>, voxelOffset: number): void {
  if (!Number.isSafeInteger(voxelOffset) || voxelOffset < 0) {
    throw new RangeError("Voxel offset must be a non-negative safe integer");
  }
  if (voxelOffset + SPARSE_BRICK_DRY_PAYLOAD.voxelsPerBrick > identities.length) {
    throw new RangeError("Identity payload does not contain a complete 8^3 brick");
  }
}

/**
 * A voxel is solid exactly when its material is non-zero.
 *
 * Not `identity != 0`: air is `packMaterialOwner(0, SPARSE_BRICK_NO_OWNER)` =
 * 0xffff0000, and a brick that exists but was never voxelized is all zeroes.
 * Both must read empty, which is what masking to the low half-word does.
 */
export function sparseBrickDryVoxelSolid(identity: number): boolean {
  return ((identity >>> 0) & 0xffff) !== 0;
}

/** Decide which of the four layouts represents this brick losslessly. */
export function classifySparseBrickDryBrick(
  identities: ArrayLike<number>,
  voxelOffset = 0,
): SparseBrickDryClassification {
  checkBrick(identities, voxelOffset);
  const palette: number[] = [];
  let occupiedVoxelCount = 0;
  let overflowed = false;
  for (let local = 0; local < SPARSE_BRICK_DRY_PAYLOAD.voxelsPerBrick; local += 1) {
    const identity = (identities[voxelOffset + local] ?? 0) >>> 0;
    if (!sparseBrickDryVoxelSolid(identity)) continue;
    occupiedVoxelCount += 1;
    if (palette.includes(identity)) continue;
    if (palette.length >= SPARSE_BRICK_DRY_PAYLOAD.maximumPaletteEntries) { overflowed = true; continue; }
    palette.push(identity);
  }
  const { headerBytes, occupancyBytes, paletteBytes, indexBytes, voxelsPerBrick } = SPARSE_BRICK_DRY_PAYLOAD;
  if (overflowed) {
    return { kind: "overflow", palette, occupiedVoxelCount, byteLength: headerBytes + voxelsPerBrick * 4 };
  }
  if (occupiedVoxelCount === 0) return { kind: "empty", palette, occupiedVoxelCount, byteLength: headerBytes };
  if (palette.length === 1) {
    // An all-ones mask carries no information the kind does not already carry,
    // and the interior of a solid is where every one of these bricks lives.
    if (occupiedVoxelCount === voxelsPerBrick) {
      return { kind: "full", palette, occupiedVoxelCount, byteLength: headerBytes };
    }
    return { kind: "uniform", palette, occupiedVoxelCount, byteLength: headerBytes + occupancyBytes };
  }
  return {
    kind: "palette", palette, occupiedVoxelCount,
    byteLength: headerBytes + occupancyBytes + paletteBytes + indexBytes,
  };
}

export interface SparseBrickDryPackedBrick {
  classification: SparseBrickDryClassification;
  /** Header, occupancy, palette, and index words in that order. */
  words: Uint32Array<ArrayBuffer>;
}

/**
 * Pack one brick. The word stream is self-describing: `words[0]` carries the
 * kind in its low two bits and the palette length in bits 2..5, so an unpacker
 * needs no side channel and a reader can skip a brick without decoding it.
 */
export function packSparseBrickDryBrick(
  identities: ArrayLike<number>,
  voxelOffset = 0,
): SparseBrickDryPackedBrick {
  const classification = classifySparseBrickDryBrick(identities, voxelOffset);
  const { occupancyWords, headerWords, indexWords, voxelsPerBrick, kinds, maximumPaletteEntries } = SPARSE_BRICK_DRY_PAYLOAD;
  const { kind, palette } = classification;
  if (kind === "overflow") {
    // The exception keeps the original per-voxel lane verbatim behind the same
    // header, so an over-wide brick degrades to today's cost and nothing else.
    const words = new Uint32Array(headerWords + voxelsPerBrick);
    words[0] = (SPARSE_BRICK_DRY_PAYLOAD.overflowBit | kinds.palette) >>> 0;
    words[1] = SPARSE_BRICK_DRY_EMPTY_IDENTITY;
    for (let local = 0; local < voxelsPerBrick; local += 1) {
      words[headerWords + local] = (identities[voxelOffset + local] ?? 0) >>> 0;
    }
    return { classification, words };
  }
  if (kind === "empty") {
    const words = new Uint32Array(headerWords);
    words[0] = kinds.empty;
    words[1] = SPARSE_BRICK_DRY_EMPTY_IDENTITY;
    return { classification, words };
  }
  if (kind === "full") {
    // Header only. `words[1]` is the identity every voxel holds, and the
    // absence of an occupancy word is the statement that all 512 are set.
    const words = new Uint32Array(headerWords);
    words[0] = (kinds.full | (1 << 2)) >>> 0;
    words[1] = palette[0] ?? SPARSE_BRICK_DRY_EMPTY_IDENTITY;
    return { classification, words };
  }
  const paletteWords = kind === "palette" ? maximumPaletteEntries : 0;
  const indexLength = kind === "palette" ? indexWords : 0;
  const words = new Uint32Array(headerWords + occupancyWords + paletteWords + indexLength);
  words[0] = ((kind === "palette" ? kinds.palette : kinds.uniform) | (palette.length << 2)) >>> 0;
  words[1] = palette[0] ?? SPARSE_BRICK_DRY_EMPTY_IDENTITY;
  for (let local = 0; local < voxelsPerBrick; local += 1) {
    const identity = (identities[voxelOffset + local] ?? 0) >>> 0;
    if (!sparseBrickDryVoxelSolid(identity)) continue;
    words[headerWords + (local >>> 5)] |= (1 << (local & 31)) >>> 0;
    if (kind !== "palette") continue;
    const slot = palette.indexOf(identity);
    words[headerWords + occupancyWords + maximumPaletteEntries + (local >>> 4)] |= (slot << ((local & 15) * 2)) >>> 0;
  }
  for (const [slot, identity] of palette.entries()) {
    words[headerWords + occupancyWords + slot] = identity >>> 0;
  }
  return { classification, words };
}

/**
 * Recover one voxel's identity. Round-trips `packSparseBrickDryBrick` exactly
 * for every layout, including the overflow fallback.
 */
export function unpackSparseBrickDryVoxel(words: ArrayLike<number>, local: number): number {
  if (!Number.isSafeInteger(local) || local < 0 || local >= SPARSE_BRICK_DRY_PAYLOAD.voxelsPerBrick) {
    throw new RangeError("Brick-local voxel index must be an integer in [0, 512)");
  }
  const { headerWords, occupancyWords, kinds, maximumPaletteEntries, overflowBit } = SPARSE_BRICK_DRY_PAYLOAD;
  const header = (words[0] ?? 0) >>> 0;
  if ((header & overflowBit) !== 0) return ((words[headerWords + local] ?? 0) >>> 0);
  const kind = header & 3;
  if (kind === kinds.empty) return SPARSE_BRICK_DRY_EMPTY_IDENTITY;
  if (kind === kinds.full) return ((words[1] ?? 0) >>> 0);
  const occupied = (((words[headerWords + (local >>> 5)] ?? 0) >>> (local & 31)) & 1) !== 0;
  if (!occupied) return SPARSE_BRICK_DRY_EMPTY_IDENTITY;
  if (kind === kinds.uniform) return ((words[1] ?? 0) >>> 0);
  const indexWord = (words[headerWords + occupancyWords + maximumPaletteEntries + (local >>> 4)] ?? 0) >>> 0;
  const slot = (indexWord >>> ((local & 15) * 2)) & 3;
  return ((words[headerWords + occupancyWords + slot] ?? 0) >>> 0);
}

/** True when the voxel is solid, read straight off the bitmask with no identity decode. */
export function sparseBrickDryOccupied(words: ArrayLike<number>, local: number): boolean {
  const { headerWords, kinds, overflowBit } = SPARSE_BRICK_DRY_PAYLOAD;
  const header = (words[0] ?? 0) >>> 0;
  if ((header & overflowBit) !== 0) return sparseBrickDryVoxelSolid((words[headerWords + local] ?? 0) >>> 0);
  if ((header & 3) === kinds.empty) return false;
  // The interior answers without a fetch, which is the point of the kind: a
  // buried voxel reports solid to every cone for the cost of the header word.
  if ((header & 3) === kinds.full) return true;
  return ((((words[headerWords + (local >>> 5)] ?? 0) >>> (local & 31)) & 1) !== 0);
}

export interface SparseBrickDryPayloadCensus {
  brickCount: number;
  emptyBricks: number;
  /** Interiors: all 512 voxels one identity, header-only like `empty`. */
  fullBricks: number;
  uniformBricks: number;
  paletteBricks: number;
  overflowBricks: number;
  /** Distinct-identity histogram over occupied bricks, keyed by palette size. */
  paletteHistogram: Record<number, number>;
  packedBytes: number;
  /** What the same bricks cost as a per-voxel `material | owner` lane. */
  perVoxelBytes: number;
  compressionRatio: number;
  meanBytesPerBrick: number;
}

/**
 * Census a whole identity lane. This is the measurement that decides whether
 * the palette width is right for a scene, and it is the shape a GPU census
 * pass must reproduce.
 */
export function censusSparseBrickDryPayload(
  identities: ArrayLike<number>,
  brickCount: number,
): SparseBrickDryPayloadCensus {
  if (!Number.isSafeInteger(brickCount) || brickCount < 0) {
    throw new RangeError("Brick count must be a non-negative safe integer");
  }
  const { voxelsPerBrick } = SPARSE_BRICK_DRY_PAYLOAD;
  const paletteHistogram: Record<number, number> = {};
  let emptyBricks = 0, fullBricks = 0, uniformBricks = 0, paletteBricks = 0, overflowBricks = 0, packedBytes = 0;
  for (let brick = 0; brick < brickCount; brick += 1) {
    const classification = classifySparseBrickDryBrick(identities, brick * voxelsPerBrick);
    packedBytes += classification.byteLength;
    if (classification.kind === "empty") { emptyBricks += 1; continue; }
    if (classification.kind === "full") fullBricks += 1;
    else if (classification.kind === "uniform") uniformBricks += 1;
    else if (classification.kind === "palette") paletteBricks += 1;
    else overflowBricks += 1;
    const size = classification.palette.length;
    paletteHistogram[size] = (paletteHistogram[size] ?? 0) + 1;
  }
  const perVoxelBytes = brickCount * voxelsPerBrick * 4;
  return {
    brickCount, emptyBricks, fullBricks, uniformBricks, paletteBricks, overflowBricks, paletteHistogram,
    packedBytes, perVoxelBytes,
    compressionRatio: packedBytes === 0 ? 0 : perVoxelBytes / packedBytes,
    meanBytesPerBrick: brickCount === 0 ? 0 : packedBytes / brickCount,
  };
}

/** Convenience for readers that want the identity split rather than the word. */
export function unpackSparseBrickDryIdentity(identity: number): { materialId: number; ownerId: number } {
  return unpackMaterialOwner(identity);
}
