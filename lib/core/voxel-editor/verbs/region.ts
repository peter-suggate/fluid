import { voxelToolUnavailable } from "../geometry";
import { sceneLatticeDimensions } from "../../scene-lattice";
import { VOXEL_BLOCK_MAXIMUM_CELLS } from "../../solid-patch/block";
import { solidFractionOccupies, type VoxelBlock } from "../../solid-patch/kind";
import type { VoxelRegionVerbTarget } from "../verb";
import type { VoxelRegion } from "../voxel-access";
import { croppedBlock, type PlacedVoxelBlock } from "../voxel-grid";

/** What the verbs that act on a whole selection have in common. */

export const plural = (count: number, noun: string): string => `${count} ${noun}${count === 1 ? "" : "s"}`;

/**
 * Why a verb that changes solids cannot run here, when the selection itself is
 * not the reason: the same gate the tools stand behind, so a verb is offered
 * exactly where a stroke would be taken.
 */
export const solidEditUnavailable = (target: VoxelRegionVerbTarget): string | undefined =>
  voxelToolUnavailable({ scene: target.scene, methodId: target.methodId });

const extent = (region: VoxelRegion): readonly [number, number, number] =>
  [region.maximumExclusive[0] - region.minimum[0], region.maximumExclusive[1] - region.minimum[1],
    region.maximumExclusive[2] - region.minimum[2]];

/**
 * Whether a selection's box is past what may be read into memory at once.
 *
 * Counting the solids of such a box is the same read, so a verb that only
 * needs the count to word its caption or to refuse an empty selection does
 * without it there: a box patch over a whole lattice is still one cheap edit.
 */
export function regionIsVast(region: VoxelRegion): boolean {
  const [i, j, k] = extent(region);
  return i * j * k > VOXEL_BLOCK_MAXIMUM_CELLS;
}

/** The selection's solids as a caption names them: counted, or just named where counting is the cost. */
export const solidsOf = (target: VoxelRegionVerbTarget): string =>
  regionIsVast(target.region) ? "the solid voxels" : plural(target.solidCount, "solid voxel");

/** Why a verb that edits the selection's solid cells cannot run: the solid-edit gate, or nothing solid to act on. */
export const contentsUnavailable = (target: VoxelRegionVerbTarget): string | undefined =>
  solidEditUnavailable(target)
    ?? (!regionIsVast(target.region) && target.solidCount === 0 ? "No solid voxels in this selection." : undefined);

/**
 * Why the selection's contents cannot be lifted off the lattice as a block: a
 * copy and every transform read the whole box, and one cell beyond it, at once.
 */
export function contentsUnreadable(target: VoxelRegionVerbTarget): string | undefined {
  const [i, j, k] = extent(target.region);
  if ((i + 2) * (j + 2) * (k + 2) > VOXEL_BLOCK_MAXIMUM_CELLS) {
    return `A selection of ${i} × ${j} × ${k} voxels is too large to copy or transform; `
      + `the limit is ${VOXEL_BLOCK_MAXIMUM_CELLS.toLocaleString("en-US")} cells.`;
  }
  return target.solidCount === 0 ? "No solid voxels in this selection." : undefined;
}

/** Why a transform — a verb that lifts the contents and sets them down changed — cannot run. */
export const transformUnavailable = (target: VoxelRegionVerbTarget): string | undefined =>
  solidEditUnavailable(target) ?? contentsUnreadable(target);

/** How many cells of a block are solid. */
export function blockSolidCount(block: VoxelBlock): number {
  let count = 0;
  for (let cell = 0; cell < block.solidFraction.length; cell += 1) {
    if (solidFractionOccupies(block.solidFraction[cell]!)) count += 1;
  }
  return count;
}

/**
 * Refuse to set solid cells down outside the lattice.
 *
 * The cells beyond it are the vessel's own shell, and past that nothing the
 * solver or the renderer holds. A piece moved onto the shell would overwrite
 * the wall, and moved on again would leave a hole in it — so a transform that
 * would cross the edge is refused whole rather than clipped into a different shape.
 */
export function assertInsideLattice(target: Pick<VoxelRegionVerbTarget, "scene">, placed: PlacedVoxelBlock): void {
  const solid = croppedBlock(placed.block);
  if (!solid) return;
  const dimensions = sceneLatticeDimensions(target.scene);
  for (let axis = 0; axis < 3; axis += 1) {
    const first = placed.minimum[axis]! + solid.offset[axis]!;
    if (first < 0 || first + solid.block.size[axis]! > dimensions[axis]!) {
      throw new Error("That would put voxels past the edge of the scene's lattice.");
    }
  }
}
