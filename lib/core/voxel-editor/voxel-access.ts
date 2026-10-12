import type { EditorRay } from "../editor-entity";
import type { PickedSolidVoxel, SolidVoxelPickOptions } from "../editor-solid-voxel";
import type { SceneDescription } from "../model";
import type { VoxelBlock } from "../solid-patch/kind";
import type { SolidWorldCoordinate } from "../solid-world";

/**
 * How the editor reads the solids it is about to edit.
 *
 * Tools and selection verbs never replay the patch list and never reach for
 * the solid world themselves: they are handed a reader for the immutable scene
 * a gesture began on, and ask it. That is what lets a flood fill, a repaint or
 * a rotate be written against voxels rather than against however the document
 * happens to record them.
 */
export interface VoxelReader {
  /** The scene this reader answers for. A later document needs its own reader. */
  readonly scene: SceneDescription;
  /** Covered share of a cell, 0..255. */
  fraction(coordinate: SolidWorldCoordinate): number;
  /** True when any of the cell is solid — the same rule the solver's occupancy uses. */
  solid(coordinate: SolidWorldCoordinate): boolean;
  /** The cell's material, or 0 where it is empty. */
  material(coordinate: SolidWorldCoordinate): number;
  /** The first solid cell along a ray, and the face the ray met. */
  pick(ray: EditorRay, options?: SolidVoxelPickOptions): PickedSolidVoxel | undefined;
  /** A dense copy of every channel over a box, empty cells included. */
  read(minimum: SolidWorldCoordinate, maximumExclusive: SolidWorldCoordinate): VoxelBlock;
}

/**
 * A set of cells: a box, and optionally which cells of it.
 *
 * With no mask the region is the whole box. A mask is one byte per cell of the
 * box, x fastest, non-zero where the cell belongs — what a connected-piece
 * selection needs so that acting on it does not also act on its bounding box.
 */
export interface VoxelRegion {
  readonly minimum: SolidWorldCoordinate;
  readonly maximumExclusive: SolidWorldCoordinate;
  readonly mask?: Uint8Array;
}
