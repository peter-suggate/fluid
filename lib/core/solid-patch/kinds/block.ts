import type { SolidWorldCoordinate, SolidWorldVoxelPatch } from "../../solid-world";
import { decodeVoxelBlock, encodeVoxelBlock, resampleVoxelBlock } from "../block";
import { SOLID_PATCH_BLOCK_KIND, type SolidPatchKind, type VoxelBlock } from "../kind";

const blockSize = (patch: SolidWorldVoxelPatch): SolidWorldCoordinate => [
  patch.maximumExclusive[0] - patch.minimum[0],
  patch.maximumExclusive[1] - patch.minimum[1],
  patch.maximumExclusive[2] - patch.minimum[2],
];

/**
 * The block last decoded. A bake validates a patch and then rasterizes it, and
 * this is what makes that one decode; one slot, so no document's worth of
 * decoded cells is ever held.
 */
let decoded: { readonly payload: string; readonly block: VoxelBlock } | undefined;

function blockOf(patch: SolidWorldVoxelPatch): VoxelBlock {
  if (typeof patch.payload !== "string") throw new RangeError("Solid block patch must carry a payload");
  const size = blockSize(patch);
  if (decoded?.payload === patch.payload && decoded.block.size.every((value, axis) => value === size[axis])) {
    return decoded.block;
  }
  const block = decodeVoxelBlock(patch.payload, size);
  decoded = { payload: patch.payload, block };
  return block;
}

/** A payload's hash, per patch object: a stamp is taken far more often than a patch is made. */
const payloadHashes = new WeakMap<SolidWorldVoxelPatch, number>();

/**
 * Dense cells carried in the patch: whatever shape, coverage and per-cell
 * materials were captured. A cell the block leaves empty is left alone, so the
 * block is a stamp of its solid cells, and a cell whose material is 0 takes
 * the patch's.
 */
export const blockSolidPatchKind: SolidPatchKind = {
  id: SOLID_PATCH_BLOCK_KIND,
  validate(patch) { blockOf(patch); },
  fold(patch, word) {
    let hash = payloadHashes.get(patch);
    if (hash === undefined) {
      const payload = patch.payload ?? "";
      hash = 0x811c_9dc5;
      for (let index = 0; index < payload.length; index += 1) {
        hash = Math.imul(hash ^ payload.charCodeAt(index), 0x0100_0193) >>> 0;
      }
      payloadHashes.set(patch, hash);
    }
    word(patch.payload?.length ?? 0);
    word(hash);
  },
  rasterizer(patch) {
    const block = blockOf(patch), [sx, sy] = block.size, [x, y, z] = patch.minimum;
    const ownMaterial = patch.materialId ?? 1;
    return {
      row(rowY, rowZ, x0, x1, out) {
        const from = x0 - x + sx * (rowY - y + sy * (rowZ - z)), count = x1 - x0;
        let covered = 0, whole = 0;
        for (let index = 0; index < count; index += 1) {
          const coverage = block.solidFraction[from + index]!, distance = block.signedDistanceQ8[from + index]!;
          const material = block.materialId[from + index]!;
          out.coverage[index] = coverage;
          out.signedDistanceQ8[index] = distance;
          out.materialId[index] = material;
          if (coverage > 0) covered += 1;
          if (coverage === 255 && distance === -128 && (material === 0 || material === ownMaterial)) whole += 1;
        }
        return covered === 0 ? "empty" : whole === count ? "full" : "cells";
      },
    };
  },
  resized(patch, minimum, maximumExclusive) {
    const size = maximumExclusive.map((value, axis) => value - minimum[axis]!) as unknown as SolidWorldCoordinate;
    return { ...patch, minimum, maximumExclusive,
      payload: encodeVoxelBlock(resampleVoxelBlock(blockOf(patch), size)) };
  },
};
