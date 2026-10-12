import { VOXEL_MATERIALS, type LinearRgb } from "./voxel-scene";

/**
 * The materials an author can give a voxel.
 *
 * One list, so the tool rail's swatches, a paint patch and a newly attached
 * voxel all mean the same ids the renderer already shades.
 */
export interface VoxelPaletteEntry {
  /** The solid world material id written into a patch. */
  readonly id: number;
  readonly label: string;
  /** A CSS colour for the swatch, matching how the renderer shades the id. */
  readonly color: string;
}

/** The sRGB hex of a scene-linear albedo: what a CSS swatch has to be given to show it. */
function cssColor(linear: LinearRgb): string {
  return `#${linear.map((channel) => {
    const value = Math.min(1, Math.max(0, channel));
    const encoded = value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
    return Math.round(255 * encoded).toString(16).padStart(2, "0");
  }).join("")}`;
}

/**
 * Every opaque entry of the renderer's own material table, in its order, so a
 * swatch is the albedo the id is shaded with and a material added there
 * appears here. Water is the table's one dielectric and is not a solid.
 *
 * Terrain's swatch is its flat albedo; the renderer may lay the lawn or the
 * porcelain closure over it, by scene.
 */
export const VOXEL_PALETTE: readonly VoxelPaletteEntry[] = Object.freeze(VOXEL_MATERIALS
  .filter((material) => material.closure === "opaque")
  .map((material) => Object.freeze({ id: material.id, label: material.name, color: cssColor(material.baseColorLinear) })));

/** The material a new voxel takes when the author has not chosen one. */
export const DEFAULT_VOXEL_MATERIAL_ID = 2;
