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

/** Cells of open ground kept between an open world's initial liquid and its box's sides and top. */
export const OPEN_WORLD_LIQUID_MARGIN_CELLS = 8;

type FluidBoxScene = Pick<SceneDescription, "container" | "voxelDomain" | "fluid">;

/**
 * The box a fluid solver builds its start of run on, in the container's
 * cells: `cells` from the container's cell `low` (whole pages on every axis).
 *
 * A container with walls is its own box: its cells rounded up to whole pages,
 * from cell 0. A container without walls (`container.walls: "none"`) holds
 * nothing; it only places the world. Its box is where the liquid starts: the
 * bounds of `fluid.initialLiquidVolumes`, `OPEN_WORLD_LIQUID_MARGIN_CELLS`
 * out on every side but the ground's and rounded out to the fewest whole pages
 * centred on them, from the ground (y = 0). Liquid laid against the container's faces is no different
 * from liquid in its middle: the box reaches past those faces as far as the
 * liquid's band does, so every page the band reaches at t = 0 is seeded.
 */
export function sceneFluidBox(scene: FluidBoxScene, maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION):
  { readonly cells: readonly [number, number, number]; readonly low: readonly [number, number, number] } {
  const whole = (cells: number) => Math.ceil(cells / LATTICE_PAGE_CELLS) * LATTICE_PAGE_CELLS;
  const container = sceneLatticeDimensions(scene, maximumDimension);
  if (scene.container.walls !== "none") return { cells: [whole(container[0]), whole(container[1]), whole(container[2])], low: [0, 0, 0] };
  const c = scene.container, volumes = scene.fluid.initialLiquidVolumes ?? [];
  if (volumes.length === 0) throw new Error("An open world (container.walls \"none\") needs initial liquid volumes to place its box");
  const corner = [-0.5 * c.width_m, 0, -0.5 * c.depth_m], size = [c.width_m, c.height_m, c.depth_m];
  const lowest = [Infinity, Infinity, Infinity], highest = [-Infinity, -Infinity, -Infinity];
  const grow = (low: readonly number[], high: readonly number[]) => {
    for (let a = 0; a < 3; a++) { lowest[a] = Math.min(lowest[a]!, low[a]!); highest[a] = Math.max(highest[a]!, high[a]!); }
  };
  for (const v of volumes) {
    if (v.shape === "box") { grow([v.min_m.x, v.min_m.y, v.min_m.z], [v.max_m.x, v.max_m.y, v.max_m.z]); continue; }
    const p = v.center_m;
    const r = v.shape === "torus" ? [v.radius_m + v.tubeRadius_m, v.tubeRadius_m, v.radius_m + v.tubeRadius_m]
      : v.shape === "cylinder" ? [v.radius_m, v.radius_m, v.halfHeight_m] : [v.radius_m, v.radius_m, v.radius_m];
    grow([p.x - r[0]!, p.y - r[1]!, p.z - r[2]!], [p.x + r[0]!, p.y + r[1]!, p.z + r[2]!]);
  }
  const P = LATTICE_PAGE_CELLS, m = OPEN_WORLD_LIQUID_MARGIN_CELLS, cells: number[] = [], low: number[] = [];
  for (let a = 0; a < 3; a++) {
    const h = size[a]! / container[a]!;
    const first = (lowest[a]! - corner[a]!) / h, last = (highest[a]! - corner[a]!) / h;
    if (!Number.isFinite(first) || !Number.isFinite(last) || last <= first) throw new Error("An open world's initial liquid has no extent");
    const lo = Math.floor(first + 1e-6) - m, hi = Math.ceil(last - 1e-6) + m;
    if (a === 1) { low.push(0); cells.push(P * Math.ceil(hi / P)); continue; }
    // The fewest whole pages over the liquid and its margin, centred on them:
    // the box's low corner is any cell of the container's, not a page of it.
    const span = P * Math.ceil((hi - lo) / P);
    low.push(lo - Math.floor((span - (hi - lo)) / 2)); cells.push(span);
  }
  return { cells: cells as unknown as readonly [number, number, number], low: low as unknown as readonly [number, number, number] };
}

/**
 * Cells of the lattice a fluid solver allocates and addresses for the scene:
 * its box (`sceneFluidBox`).
 *
 * For a container with walls this is the container's cells rounded up to
 * whole pages. The lattice's cell (0,0,0) is the container's, and so are its
 * cell size and its origin; the cells it adds lie past the container's high
 * faces, solid across a closed wall and air above an open top
 * (`solidVoxelPaddingForLattice`). Outside the lattice the page table holds
 * the rest of the world: solid ground below an open top's brim and air above
 * it, so liquid that leaves an open top claims pages above the box. A closed
 * container that is already whole pages is its own lattice. An open world's
 * box is its initial liquid's, wherever that lies against the container.
 */
export function sceneFluidLatticeDimensions(
  scene: FluidBoxScene,
  maximumDimension = DEFAULT_MAXIMUM_LATTICE_DIMENSION,
): readonly [number, number, number] {
  return sceneFluidBox(scene, maximumDimension).cells;
}

/** Pages of the fluid solver's window on each axis: lattice coordinates run 0..64 pages. */
export const FLUID_WINDOW_PAGES = 64;

/**
 * Where the box (`sceneFluidBox`) sits in the fluid solver's window, in
 * pages: its low corner. A container with walls sits at the window's origin;
 * one without (`container.walls: "none"`) sits on the floor in the middle of
 * the window, so liquid that leaves it has the window's width to spread in on
 * every side.
 */
export function sceneFluidBoxPage(scene: FluidBoxScene): readonly [number, number, number] {
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
 * position of lattice cell (0,0,0)'s low corner: the box's low corner, the
 * container's cell `boxLow`, less the box's place in the window (`boxCell`,
 * the box's low corner in lattice cells, zero for a container with walls).
 */
export interface SceneFluidFrame {
  readonly origin_m: readonly [number, number, number];
  /** The extent and the cells a cell's size is measured over: size_m / cells. */
  readonly size_m: readonly [number, number, number];
  readonly cells: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  /** The container's low corner in the world. */
  readonly container_m: readonly [number, number, number];
  /** The box's low corner in lattice cells. */
  readonly boxCell: readonly [number, number, number];
  /** The box's low corner in the container's cells (sceneFluidBox().low): zero for a container with walls. */
  readonly boxLow: readonly [number, number, number];
}

export function sceneFluidFrame(scene: FluidBoxScene): SceneFluidFrame {
  const c = scene.container, cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const size_m = [c.width_m, c.height_m, c.depth_m] as const;
  const cellSize_m = [size_m[0] / cells[0], size_m[1] / cells[1], size_m[2] / cells[2]] as const;
  const container_m = [-0.5 * c.width_m, 0, -0.5 * c.depth_m] as const;
  const page = sceneFluidBoxPage(scene), boxLow = sceneFluidBox(scene, Number.MAX_SAFE_INTEGER).low;
  const boxCell = [LATTICE_PAGE_CELLS * page[0], LATTICE_PAGE_CELLS * page[1], LATTICE_PAGE_CELLS * page[2]] as const;
  const origin = (a: 0 | 1 | 2) => container_m[a] + (boxLow[a] - boxCell[a]) * cellSize_m[a];
  return { origin_m: [origin(0), origin(1), origin(2)], size_m, cells, cellSize_m, container_m, boxCell, boxLow };
}

/**
 * The scene a fluid solver's start of run is built from: one whose container
 * is its box (`sceneFluidBox`). A container with walls is its own box, and
 * this is the scene itself. An open world's is the scene moved so its box is
 * the container: the box's extent, its liquid and its terrain features
 * translated by the box's centre, so the initial liquid, the vertex phi, the
 * solid world and the terrain columns are built over the box's cells alone.
 * One object a scene object, so a cache keyed by scene sees one scene.
 */
export function sceneFluidBoxScene<T extends SceneDescription>(scene: T): T {
  if (scene.container.walls !== "none") return scene;
  const held = boxScenes.get(scene);
  if (held) return held as T;
  // Voxels are authored in the container's cells: an open world has none (validateScene).
  if (scene.solidVoxels.length > 0) throw new Error("An open world (container.walls \"none\") takes no solid voxels");
  const frame = sceneFluidFrame(scene), box = sceneFluidBox(scene, Number.MAX_SAFE_INTEGER), c = scene.container;
  const extent = [0, 1, 2].map(a => box.cells[a]! * frame.cellSize_m[a]!);
  const dx = frame.container_m[0] + box.low[0] * frame.cellSize_m[0] + 0.5 * extent[0]!;
  const dz = frame.container_m[2] + box.low[2] * frame.cellSize_m[2] + 0.5 * extent[2]!;
  const move = (p: { x: number; y: number; z: number }) => ({ ...p, x: p.x - dx, z: p.z - dz });
  const volumes = (scene.fluid.initialLiquidVolumes ?? []).map(v => v.shape === "box"
    ? { ...v, min_m: move(v.min_m), max_m: move(v.max_m) } : { ...v, center_m: move(v.center_m) });
  const terrain = scene.terrain && { ...scene.terrain,
    features: scene.terrain.features.map(f => ({ ...f, center_m: { x: f.center_m.x - dx, z: f.center_m.z - dz } })) };
  const moved = { ...scene, container: { ...c, width_m: extent[0]!, height_m: extent[1]!, depth_m: extent[2]! },
    fluid: { ...scene.fluid, initialLiquidVolumes: volumes }, ...(terrain ? { terrain } : {}) } as T;
  boxScenes.set(scene, moved);
  return moved;
}
const boxScenes = new WeakMap<SceneDescription, SceneDescription>();

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
