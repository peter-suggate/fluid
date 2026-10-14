import type { SceneDescription } from "./model";

/** Default WebGPU texture extent cap used by scene planning. */
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

/** Finest-lattice cell counts occupied by the physical tank interior. */
export function sceneLatticeDimensions(
  scene: Pick<SceneDescription, "container" | "voxelDomain">,
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

/** A fluid solver stores its fields in pages this many cells on an edge. */
export const LATTICE_PAGE_CELLS = 32;

/** Cells of air a fluid lattice holds above an open top, before rounding to pages. */
export const LATTICE_OPEN_TOP_HEADROOM_CELLS = 64;

/**
 * Cells of the lattice a fluid solver allocates and addresses for the scene:
 * the container's cells rounded up to whole pages on every axis, and under an
 * open top the headroom above it (LATTICE_OPEN_TOP_HEADROOM_CELLS).
 *
 * The container stays what was authored. The lattice's cell (0,0,0) is the
 * container's, and so are its cell size and its origin; the cells it adds lie
 * past the container's high faces, solid across a closed wall and air above an
 * open top (`solidVoxelPaddingForLattice`). The lattice itself is a closed
 * box: liquid that leaves an open top rises between walls to a ceiling at the
 * lattice's top. A closed container that is already whole pages is its own
 * lattice.
 */
export function sceneFluidLatticeDimensions(
  scene: Pick<SceneDescription, "container" | "voxelDomain">,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): readonly [number, number, number] {
  const whole = (cells: number) => Math.ceil(cells / LATTICE_PAGE_CELLS) * LATTICE_PAGE_CELLS;
  const [nx, ny, nz] = sceneLatticeDimensions(scene, maximumDimension);
  return [whole(nx), whole(ny + (scene.container.top === "open" ? LATTICE_OPEN_TOP_HEADROOM_CELLS : 0)), whole(nz)];
}

/** Finest cell size actually realized on each axis after lattice rounding. */
export function sceneCellSizes_m(
  scene: Pick<SceneDescription, "container" | "voxelDomain">,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): readonly [number, number, number] {
  const [nx, ny, nz] = sceneLatticeDimensions(scene, maximumDimension);
  const c = scene.container;
  return [c.width_m / nx, c.height_m / ny, c.depth_m / nz];
}

export function sceneLatticeCellCount(
  scene: Pick<SceneDescription, "container" | "voxelDomain">,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): number {
  const [nx, ny, nz] = sceneLatticeDimensions(scene, maximumDimension);
  return nx * ny * nz;
}
