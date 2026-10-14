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

/**
 * Cells of the lattice a fluid solver allocates and addresses for the scene:
 * the container's cells rounded up to whole pages on every axis.
 *
 * The container stays what was authored. The lattice's cell (0,0,0) is the
 * container's, and so are its cell size and its origin; the cells it adds lie
 * past the container's high faces, solid across a closed wall and air above an
 * open top (`solidVoxelPaddingForLattice`). Outside the lattice the page table
 * holds the rest of the world: solid ground below an open top's brim and air
 * above it, so liquid that leaves an open top claims pages above the box. A
 * closed container that is already whole pages is its own lattice.
 */
export function sceneFluidLatticeDimensions(
  scene: Pick<SceneDescription, "container" | "voxelDomain">,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): readonly [number, number, number] {
  const whole = (cells: number) => Math.ceil(cells / LATTICE_PAGE_CELLS) * LATTICE_PAGE_CELLS;
  const [nx, ny, nz] = sceneLatticeDimensions(scene, maximumDimension);
  return [whole(nx), whole(ny), whole(nz)];
}

/** Pages of the fluid solver's window on each axis: lattice coordinates run 0..64 pages. */
export const FLUID_WINDOW_PAGES = 64;

/**
 * Where the container's lattice (the box) sits in the fluid solver's window,
 * in pages: its low corner. A container with walls sits at the window's
 * origin; one without (`container.walls: "none"`) sits on the floor in the
 * middle of the window, so liquid that leaves it has the window's width to
 * spread in on every side.
 */
export function sceneFluidBoxPage(scene: Pick<SceneDescription, "container" | "voxelDomain">): readonly [number, number, number] {
  if (scene.container.walls !== "none") return [0, 0, 0];
  const [nx, , nz] = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const centre = (cells: number) => Math.floor((FLUID_WINDOW_PAGES - cells / LATTICE_PAGE_CELLS) / 2);
  return [centre(nx), 0, centre(nz)];
}

/**
 * Where a fluid solver's lattice sits in the world: its frame. A lattice
 * coordinate in cells is `(world - origin_m) * cells / size_m` on each axis,
 * and a world position `origin_m + cell * size_m / cells`. The container
 * fixes the cell size (its extent over its cells). The origin is the world
 * position of lattice cell (0,0,0)'s low corner: the container's low corner
 * less the box's place in the window (`boxCell`, the container's low corner
 * in lattice cells, zero for a container with walls).
 */
export interface SceneFluidFrame {
  readonly origin_m: readonly [number, number, number];
  /** The extent and the cells a cell's size is measured over: size_m / cells. */
  readonly size_m: readonly [number, number, number];
  readonly cells: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  /** The container's low corner: in the world, and in lattice cells. */
  readonly container_m: readonly [number, number, number];
  readonly boxCell: readonly [number, number, number];
}

export function sceneFluidFrame(scene: Pick<SceneDescription, "container" | "voxelDomain">): SceneFluidFrame {
  const c = scene.container, cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const size_m = [c.width_m, c.height_m, c.depth_m] as const;
  const cellSize_m = [size_m[0] / cells[0], size_m[1] / cells[1], size_m[2] / cells[2]] as const;
  const container_m = [-0.5 * c.width_m, 0, -0.5 * c.depth_m] as const;
  const page = sceneFluidBoxPage(scene);
  const boxCell = [LATTICE_PAGE_CELLS * page[0], LATTICE_PAGE_CELLS * page[1], LATTICE_PAGE_CELLS * page[2]] as const;
  return { origin_m: [container_m[0] - boxCell[0] * cellSize_m[0], container_m[1] - boxCell[1] * cellSize_m[1], container_m[2] - boxCell[2] * cellSize_m[2]],
    size_m, cells, cellSize_m, container_m, boxCell };
}

/** A world coordinate on one axis in lattice cells (unfloored). */
export function fluidFrameCell(frame: SceneFluidFrame, axis: number, world_m: number): number {
  return (world_m - frame.origin_m[axis]!) * frame.cells[axis]! / frame.size_m[axis]!;
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
