import type { SceneDescription } from "./model";

/**
 * The one place the container's metre extents become a cell count.
 *
 * `createTallCellLayout`, the editor's paint lattice, the configuration
 * readout, and the solver rebuild key all have to agree about this rounding:
 * the moment two of them disagree, an edit either rebuilds a solver that did
 * not need rebuilding or — far worse — keeps one whose arenas no longer match
 * the scene. Scaling the world makes that agreement load-bearing, because the
 * whole point of a world scale is that extent and cell size move together and
 * the lattice does not move at all.
 */

/** Matches `createTallCellLayout`'s default device cap. */
export const DEFAULT_MAXIMUM_LATTICE_DIMENSION = 2048;
/** Below this a container axis stops resolving anything, so the layout floors. */
export const MINIMUM_LATTICE_DIMENSION = 8;

export function latticeAxisDimension(
  extent_m: number,
  cellSize_m: number,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): number {
  if (!(extent_m > 0) || !(cellSize_m > 0)) return MINIMUM_LATTICE_DIMENSION;
  return Math.min(maximumDimension,
    Math.max(MINIMUM_LATTICE_DIMENSION, Math.round(extent_m / cellSize_m)));
}

/** Finest-lattice cell counts the container resolves to, in x, y, z. */
export function sceneLatticeDimensions(
  scene: SceneDescription,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): readonly [number, number, number] {
  const cellSize_m = scene.voxelDomain.finestCellSize_m;
  const c = scene.container;
  return [
    latticeAxisDimension(c.width_m, cellSize_m, maximumDimension),
    latticeAxisDimension(c.height_m, cellSize_m, maximumDimension),
    latticeAxisDimension(c.depth_m, cellSize_m, maximumDimension),
  ];
}

/** Finest cell size actually realized on each axis after the rounding above. */
export function sceneCellSizes_m(
  scene: SceneDescription,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): readonly [number, number, number] {
  const [nx, ny, nz] = sceneLatticeDimensions(scene, maximumDimension);
  const c = scene.container;
  return [c.width_m / nx, c.height_m / ny, c.depth_m / nz];
}

export function sceneLatticeCellCount(
  scene: SceneDescription,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): number {
  const [nx, ny, nz] = sceneLatticeDimensions(scene, maximumDimension);
  return nx * ny * nz;
}
