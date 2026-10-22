import type { SceneDescription } from "./model";
import { sceneCellSizes_m, sceneFluidLatticeDimensions, sceneLatticeDimensions } from "./scene-lattice-dimensions";
import {
  PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO,
  type PlanarBoundaryPatch,
  type PlanarBoundaryVec3,
} from "./planar-boundary";
import {
  sceneHasTerrain,
  terrainCellSolidFraction,
  terrainContentStamp,
  terrainColumnHeightAt,
  terrainGridExtent,
  terrainWorldContainer,
  type TerrainDescription,
} from "./terrain";
import {
  buildEnvironmentProxyCatalog,
  environmentProxyPrimitives,
} from "./voxel-environments";
import { composeSolidPatch, SOLID_PATCH_PAGE_CELLS, validateSolidPatch,
  type SolidPatchPageStore } from "./solid-patch/compose";
import { SOLID_PATCH_BOX_KIND, solidPatchKindId } from "./solid-patch/kind";
import { solidPatchKinds } from "./solid-patch/registry";

export const SOLID_WORLD_VERSION = 1;
export const SOLID_WORLD_BRICK_CELLS = 8;
// The patch compositor addresses pages without importing this module.
if (SOLID_PATCH_PAGE_CELLS !== SOLID_WORLD_BRICK_CELLS) {
  throw new Error(`Solid patch pages are ${SOLID_PATCH_PAGE_CELLS} cells wide; SolidWorld pages are ${SOLID_WORLD_BRICK_CELLS}`);
}
export const SOLID_WORLD_VOXELS_PER_PAGE = SOLID_WORLD_BRICK_CELLS ** 3;
export const SOLID_WORLD_EMPTY_PAGE = 0xffff_ffff;
export const SOLID_WORLD_STATIC_PAGE_BYTES = 2560;
export const SOLID_WORLD_DYNAMIC_PAGE_BYTES = 5120;
export const SOLID_WORLD_DIRECTORY_ENTRY_BYTES = 7 * 4;
/** Stable material-table slot shared with the voxel scene renderer. */
export const SOLID_WORLD_TERRAIN_MATERIAL_ID = 2;

export type SolidWorldCoordinate = readonly [number, number, number];

/**
 * What a patch does with the cells its kind covers.
 *
 * `paint` changes the material of cells that are already solid and nothing
 * else: it never fills air and never alters coverage or distance.
 */
export type SolidPatchOperation = "fill" | "clear" | "paint";

/**
 * One ordered edit of the solid world: an integer box, what fills it, and what
 * that does to the cells underneath.
 *
 * The box is always the patch's bounds. `kind` names the registered
 * `SolidPatchKind` (lib/core/solid-patch) that says how much of each cell in
 * the box the patch covers; absent means `"box"`, every cell whole, which is
 * every document written before kinds existed. The record stays flat JSON so
 * it survives `cloneScene`, the worker hand-off and a saved file unchanged.
 */
export interface SolidWorldVoxelPatch {
  readonly operation: SolidPatchOperation;
  readonly minimum: SolidWorldCoordinate;
  readonly maximumExclusive: SolidWorldCoordinate;
  readonly materialId?: number;
  /** Clear this volume from authored scenery as well as voxel solids. Absent/false keeps legacy voxel-only clears. */
  readonly affectsScenery?: boolean;
  /** Registered patch kind id. Absent means `"box"`. */
  readonly kind?: string;
  /** The axis a kind with one is aligned to (a cylinder's). Absent means 1, up. */
  readonly axis?: 0 | 1 | 2;
  /** A kind's own encoded payload (a voxel block's cells). Opaque outside its kind. */
  readonly payload?: string;
}

export interface SolidWorldVoxelPatchBounds {
  readonly minimum: PlanarBoundaryVec3;
  readonly maximum: PlanarBoundaryVec3;
}

/** Exact metre-space bounds of one signed-lattice voxel edit box. */
export function solidWorldVoxelPatchBounds_m(
  scene: Pick<SceneDescription, "container" | "voxelDomain">,
  patch: SolidWorldVoxelPatch,
): SolidWorldVoxelPatchBounds {
  const cell = sceneCellSizes_m(scene);
  const origin = [-0.5 * scene.container.width_m, 0,
    -0.5 * scene.container.depth_m] as const;
  return {
    minimum: [
      origin[0] + patch.minimum[0] * cell[0],
      origin[1] + patch.minimum[1] * cell[1],
      origin[2] + patch.minimum[2] * cell[2],
    ],
    maximum: [
      origin[0] + patch.maximumExclusive[0] * cell[0],
      origin[1] + patch.maximumExclusive[1] * cell[1],
      origin[2] + patch.maximumExclusive[2] * cell[2],
    ],
  };
}

/**
 * Compile a thin filled voxel box into the same exact slab used by SVO rays.
 *
 * Only a whole box is a slab: a round kind's box is not its surface, and a
 * paint or a clear is not a surface at all.
 */
export function planarBoundaryForSolidWorldVoxelPatch(
  scene: SceneDescription,
  patch: SolidWorldVoxelPatch,
  ownerId = 0xffff,
): PlanarBoundaryPatch | null {
  if (patch.operation !== "fill" || solidPatchKindId(patch) !== SOLID_PATCH_BOX_KIND) return null;
  const bounds = solidWorldVoxelPatchBounds_m(scene, patch);
  const dimensions: [number, number, number] = [
    bounds.maximum[0] - bounds.minimum[0],
    bounds.maximum[1] - bounds.minimum[1],
    bounds.maximum[2] - bounds.minimum[2],
  ];
  const ordered = dimensions.map((value, axis) => ({ value, axis }))
    .sort((left, right) => left.value - right.value);
  const thinnest = ordered[0]!;
  const next = ordered[1]!;
  if (!(thinnest.value > 0)
    || next.value < thinnest.value * PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO) {
    return null;
  }
  const normalAxis = thinnest.axis;
  const tangentAxes = [0, 1, 2].filter((axis) => axis !== normalAxis);
  const basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const;
  const tangentUAxis = tangentAxes[0]!;
  const tangentVAxis = tangentAxes[1]!;
  return {
    center_m: [
      0.5 * (bounds.minimum[0] + bounds.maximum[0]),
      0.5 * (bounds.minimum[1] + bounds.maximum[1]),
      0.5 * (bounds.minimum[2] + bounds.maximum[2]),
    ],
    normal: basis[normalAxis]!,
    tangentU: basis[tangentUAxis]!,
    tangentV: basis[tangentVAxis]!,
    halfExtentU_m: 0.5 * dimensions[tangentUAxis]!,
    halfExtentV_m: 0.5 * dimensions[tangentVAxis]!,
    halfThickness_m: 0.5 * dimensions[normalAxis],
    materialId: patch.materialId ?? 1,
    ownerId,
  };
}

export interface SolidWorldPage {
  readonly coordinate: SolidWorldCoordinate;
  /** 0 is empty and 255 is a fully covered solid voxel. */
  readonly solidFraction: Uint8Array;
  /** Signed distance in 1/256-cell units; negative is inside solid. */
  readonly signedDistanceQ8: Int16Array;
  readonly materialId: Uint16Array;
  readonly revision: number;
  /**
   * One bit per cell (bit `x + 8 (y + 8 z)`, low bit first), set once a fill
   * or a clear has written the cell. Absent on a page no patch has shaped.
   *
   * It is what tells an authored cell from the ground a page was seeded with:
   * a renderer that draws the ground analytically takes only these cells from
   * the page. A clear sets it on every cell it covers, solid or not, so a dig
   * stays dug under a ground drawn from a different sampling of the same
   * heights. A paint does not set it; it moves no surface.
   */
  authored?: Uint8Array;
}

export interface SolidWorldMemoryPlan {
  readonly staticPageCapacity: number;
  readonly dynamicPageCapacity: number;
  readonly directoryCapacity: number;
  readonly directoryBytes: number;
  readonly staticPayloadBytes: number;
  readonly dynamicPayloadBytes: number;
  readonly totalBytes: number;
}

const checkedProduct = (left: number, right: number, label: string): number => {
  const value = left * right;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} exceeds safe byte arithmetic`);
  }
  return value;
};

const nextPowerOfTwo = (value: number): number => {
  let result = 1;
  while (result < value) result *= 2;
  return result;
};

export function planSolidWorldMemory(options: {
  readonly staticPageCapacity: number;
  readonly dynamicPageCapacity?: number;
  readonly maximumBufferBytes?: number;
}): SolidWorldMemoryPlan {
  const staticPageCapacity = options.staticPageCapacity;
  const dynamicPageCapacity = options.dynamicPageCapacity ?? 0;
  if (![staticPageCapacity, dynamicPageCapacity].every((value) =>
    Number.isSafeInteger(value) && value >= 0)) {
    throw new RangeError("SolidWorld page capacities must be non-negative integers");
  }
  const pageCapacity = staticPageCapacity + dynamicPageCapacity;
  const directoryCapacity = nextPowerOfTwo(Math.max(2, 2 * pageCapacity));
  const directoryBytes = checkedProduct(directoryCapacity,
    SOLID_WORLD_DIRECTORY_ENTRY_BYTES, "SolidWorld directory");
  const staticPayloadBytes = checkedProduct(staticPageCapacity,
    SOLID_WORLD_STATIC_PAGE_BYTES, "SolidWorld static payload");
  const dynamicPayloadBytes = checkedProduct(dynamicPageCapacity,
    SOLID_WORLD_DYNAMIC_PAGE_BYTES, "SolidWorld dynamic payload");
  const totalBytes = directoryBytes + staticPayloadBytes + dynamicPayloadBytes;
  if (!Number.isSafeInteger(totalBytes)
    || (options.maximumBufferBytes !== undefined
      && totalBytes > options.maximumBufferBytes)) {
    throw new RangeError(`SolidWorld requires ${totalBytes} bytes, exceeding its buffer budget`);
  }
  return Object.freeze({ staticPageCapacity, dynamicPageCapacity,
    directoryCapacity, directoryBytes, staticPayloadBytes, dynamicPayloadBytes,
    totalBytes });
}

const floorDiv = (value: number, divisor: number): number => Math.floor(value / divisor);
const floorMod = (value: number, divisor: number): number =>
  value - floorDiv(value, divisor) * divisor;

export function solidWorldPageAddress(coordinate: SolidWorldCoordinate): {
  readonly page: SolidWorldCoordinate;
  readonly local: SolidWorldCoordinate;
  readonly localIndex: number;
} {
  if (!coordinate.every((value) => Number.isSafeInteger(value)
    && value >= -0x8000_0000 && value <= 0x7fff_ffff)) {
    throw new RangeError("SolidWorld voxel coordinate must fit signed i32");
  }
  const page = coordinate.map((value) => floorDiv(value,
    SOLID_WORLD_BRICK_CELLS)) as unknown as SolidWorldCoordinate;
  const local = coordinate.map((value) => floorMod(value,
    SOLID_WORLD_BRICK_CELLS)) as unknown as SolidWorldCoordinate;
  return { page, local, localIndex: local[0] + SOLID_WORLD_BRICK_CELLS
    * (local[1] + SOLID_WORLD_BRICK_CELLS * local[2]) };
}

/** The order every SolidWorld lists its pages in: ascending z, then y, then x. */
export const solidWorldPageOrder = (left: SolidWorldCoordinate, right: SolidWorldCoordinate): number =>
  left[2] - right[2] || left[1] - right[1] || left[0] - right[0];

/** First index of an ascending page list whose page is not before page coordinate (x, y, z). */
function pageLowerBound(pages: readonly SolidWorldPage[], x: number, y: number, z: number): number {
  let low = 0, high = pages.length;
  while (low < high) {
    const middle = (low + high) >>> 1, q = pages[middle]!.coordinate;
    if ((q[2] - z || q[1] - y || q[0] - x) < 0) low = middle + 1;
    else high = middle;
  }
  return low;
}

/**
 * CPU oracle form of the exact signed sparse page directory.
 *
 * A world lists the pages its patches wrote ascending in (z, y, x), so the
 * list is its own directory and a lookup is a search of it. Nothing is built per world: a
 * stroke that copies three pages of a sixty-thousand-page world must not pay
 * for a table of all of them, which is what a hashed directory cost it.
 */
export class SolidWorldDirectory {
  // A lattice walk asks for the same page eight cells running; the last answer is kept.
  private lastX = Number.NaN;
  private lastY = Number.NaN;
  private lastZ = Number.NaN;
  private lastIndex: number | undefined;

  /** `pages` must ascend in (z, y, x) and never change: it is the world's own list. */
  constructor(private readonly pages: readonly SolidWorldPage[]) {}

  /** Index into the world's `pages` of the page at a page coordinate. */
  lookup(coordinate: SolidWorldCoordinate): number | undefined {
    const x = coordinate[0], y = coordinate[1], z = coordinate[2];
    if (x === this.lastX && y === this.lastY && z === this.lastZ) return this.lastIndex;
    const index = pageLowerBound(this.pages, x, y, z), q = this.pages[index]?.coordinate;
    this.lastX = x; this.lastY = y; this.lastZ = z;
    return this.lastIndex = q !== undefined && q[0] === x && q[1] === y && q[2] === z ? index : undefined;
  }

  get size(): number { return this.pages.length; }
}

const emptyPage = (coordinate: SolidWorldCoordinate, revision: number): SolidWorldPage => ({
  coordinate: [...coordinate],
  solidFraction: new Uint8Array(SOLID_WORLD_VOXELS_PER_PAGE),
  signedDistanceQ8: new Int16Array(SOLID_WORLD_VOXELS_PER_PAGE).fill(0x7fff),
  materialId: new Uint16Array(SOLID_WORLD_VOXELS_PER_PAGE),
  revision,
});

/**
 * The scene's terrain as cells, answered a column at a time and never baked.
 *
 * Terrain is a function of a height per column, so nothing about it is stored
 * until something asks: one f32 height per column, kept in 8 x 8 tiles that
 * exist only where a consumer has looked, and a page of cells only for a
 * consumer that needs the bits of that page (`page`). A world over this ground
 * holds the pages patches wrote and no others, so a document's world costs its
 * patches and not its landscape: the largest garden is seconds and a hundred
 * megabytes as pages, and nothing as this.
 *
 * The lattice placement, the f32 column height, and the fraction, distance and
 * material of every cell are one definition, here. A page seeded from this and
 * composed over is exactly the cells any other reader of the ground sees.
 */
export class SolidGround {
  /** The lattice box the ground can occupy; nothing outside it is ever terrain. */
  readonly minimum: SolidWorldCoordinate;
  readonly maximumExclusive: SolidWorldCoordinate;
  private readonly originX_m: number;
  private readonly originZ_m: number;
  private readonly hx: number;
  private readonly hy: number;
  private readonly hz: number;
  private readonly nx: number;
  private readonly nz: number;
  private readonly maximumHeight_m: number;
  private readonly tilesX: number;
  private readonly tiles: (Float32Array | undefined)[];
  /** Tiles `sample` has filled, and how many it has not. */
  private readonly sampledTiles: Uint8Array;
  private unsampledTiles: number;
  /** Solid cells of the tallest column under each lattice page, -1 until its 64 columns have been sampled. */
  private readonly pageTops: Int32Array;
  private readonly pagesX: number;
  private readonly pagesZ: number;
  /** Pages some consumer needed as bits, by `pageSlot`. They are never written. */
  private readonly baked = new Map<number, SolidWorldPage>();

  constructor(readonly key: string, private readonly terrain: TerrainDescription,
    scene: Pick<SceneDescription, "terrain" | "container" | "voxelDomain">) {
    const [hx, hy, hz] = sceneCellSizes_m(scene);
    const extent = terrainWorldContainer(scene);
    // Signed coordinates let a garden's banks reach past the solver tank, so
    // the ground starts at a negative column. Physics samples its own domain.
    const x0 = Math.round((scene.container.width_m - extent.width_m) / (2 * hx));
    const z0 = Math.round((scene.container.depth_m - extent.depth_m) / (2 * hz));
    this.nx = Math.round(extent.width_m / hx);
    this.nz = Math.round(extent.depth_m / hz);
    const ny = Math.round(extent.height_m / hy);
    this.hx = hx; this.hy = hy; this.hz = hz;
    this.originX_m = -scene.container.width_m / 2 + x0 * hx;
    this.originZ_m = -scene.container.depth_m / 2 + z0 * hz;
    this.maximumHeight_m = extent.height_m;
    // The tallest the ground can be without sampling it. A described ground
    // would have to be derived to know, and a point query must never do that,
    // so it is bounded by the height every column is clamped to instead.
    const tallest_m = terrain.grid ? terrainGridExtent(terrain.grid).maximum_m
      : terrain.procedural ? extent.height_m
        : terrain.baseHeight_m + terrain.features.reduce((sum, feature) =>
          sum + (feature.kind === "mound" ? feature.amount_m : 0), 0);
    const ceiling = Math.max(0, Math.min(ny, Math.ceil(Math.min(extent.height_m, tallest_m) / hy)));
    const B = SOLID_WORLD_BRICK_CELLS;
    this.minimum = [x0, 0, z0];
    this.maximumExclusive = [x0 + this.nx, ceiling, z0 + this.nz];
    this.tilesX = Math.ceil(this.nx / B);
    this.tiles = new Array(this.tilesX * Math.ceil(this.nz / B));
    this.sampledTiles = new Uint8Array(this.tiles.length);
    this.unsampledTiles = this.tiles.length;
    this.pagesX = floorDiv(x0 + this.nx - 1, B) - floorDiv(x0, B) + 1;
    this.pagesZ = floorDiv(z0 + this.nz - 1, B) - floorDiv(z0, B) + 1;
    this.pageTops = new Int32Array(this.pagesX * this.pagesZ).fill(-1);
  }

  /** Whether any cell of the cube of `size` cells at `(x, y, z)` could be terrain. */
  reaches(x: number, y: number, z: number, size: number): boolean {
    const lo = this.minimum, hi = this.maximumExclusive;
    return x < hi[0] && x + size > lo[0] && y < hi[1] && y + size > lo[1] && z < hi[2] && z + size > lo[2];
  }

  /** The column's ground height in metres, as an f32; 0 outside the ground. */
  height_m(x: number, z: number): number {
    const cx = x - this.minimum[0], cz = z - this.minimum[2];
    if (cx < 0 || cx >= this.nx || cz < 0 || cz >= this.nz) return 0;
    const tileIndex = (cz >> 3) * this.tilesX + (cx >> 3);
    let tile = this.tiles[tileIndex];
    if (!tile) tile = this.tiles[tileIndex] = new Float32Array(SOLID_WORLD_BRICK_CELLS ** 2).fill(Number.NaN);
    const local = (cx & 7) + SOLID_WORLD_BRICK_CELLS * (cz & 7);
    let height = tile[local]!;
    if (height !== height) {
      tile[local] = terrainColumnHeightAt(this.terrain, this.originX_m + (cx + 0.5) * this.hx,
        this.originZ_m + (cz + 0.5) * this.hz, this.maximumHeight_m);
      height = tile[local]!;
    }
    return height;
  }

  /**
   * Sample the height of every column under the pages a lattice box touches,
   * a tile at a time, until `expired` says to stop. True once all are held.
   *
   * A height is the one cost of this ground that is the generator's, and it is
   * paid by whoever looks at a column first: a patch composed over the ground
   * pays for 64 columns a page, a pick ray for every column it crosses. This
   * pays ahead of them in turns as short as a tile, and changes no answer.
   */
  sample(minimum: SolidWorldCoordinate, maximumExclusive: SolidWorldCoordinate, expired?: () => boolean): boolean {
    if (this.unsampledTiles === 0) return true;
    const B = SOLID_WORLD_BRICK_CELLS, lo = this.minimum;
    // No page above the ground's ceiling or below its floor is asked for a column.
    if (maximumExclusive[1] <= 0 || B * floorDiv(minimum[1], B) >= this.maximumExclusive[1]) return true;
    const x0 = Math.max(0, B * floorDiv(minimum[0], B) - lo[0]);
    const x1 = Math.min(this.nx, B * (floorDiv(maximumExclusive[0] - 1, B) + 1) - lo[0]);
    const z0 = Math.max(0, B * floorDiv(minimum[2], B) - lo[2]);
    const z1 = Math.min(this.nz, B * (floorDiv(maximumExclusive[2] - 1, B) + 1) - lo[2]);
    if (x0 >= x1 || z0 >= z1) return true;
    for (let tz = z0 >> 3; tz <= (z1 - 1) >> 3; tz += 1) for (let tx = x0 >> 3; tx <= (x1 - 1) >> 3; tx += 1) {
      const tile = tz * this.tilesX + tx;
      if (this.sampledTiles[tile]) continue;
      for (let z = B * tz; z < B * tz + B; z += 1) for (let x = B * tx; x < B * tx + B; x += 1) {
        this.height_m(lo[0] + x, lo[2] + z);
      }
      this.sampledTiles[tile] = 1;
      this.unsampledTiles -= 1;
      if (expired?.()) return false;
    }
    return true;
  }

  /**
   * How many cells of the column, counted up from y = 0, hold any ground.
   *
   * A column's fraction never rises with y, so its solid cells are exactly
   * `0 <= y < solidCells(x, z)`: the whole of the ground's occupancy, and all
   * a yes-or-no consumer (the solver's mask, a pick) needs of a column.
   */
  solidCells(x: number, z: number): number {
    const height = this.height_m(x, z);
    let cells = Math.min(this.maximumExclusive[1], Math.ceil(height / this.hy));
    while (cells > 0 && Math.round(255 * terrainCellSolidFraction(height, (cells - 1) * this.hy, this.hy)) === 0) cells -= 1;
    return cells;
  }

  /** `solidCells` of the tallest column under the page column at page coordinate (px, pz). */
  pageColumnSolidCells(px: number, pz: number): number {
    const B = SOLID_WORLD_BRICK_CELLS;
    const sx = px - floorDiv(this.minimum[0], B), sz = pz - floorDiv(this.minimum[2], B);
    if (sx < 0 || sx >= this.pagesX || sz < 0 || sz >= this.pagesZ) return 0;
    const slot = sx + this.pagesX * sz;
    let top = this.pageTops[slot]!;
    if (top < 0) {
      top = 0;
      for (let z = B * pz; z < B * pz + B; z += 1) for (let x = B * px; x < B * px + B; x += 1) {
        top = Math.max(top, this.solidCells(x, z));
      }
      this.pageTops[slot] = top;
    }
    return top;
  }

  /**
   * Whether the page at a page coordinate holds any ground.
   *
   * Tighter than `reaches`, which knows only the tallest the ground can be,
   * and paid for by sampling the page's 64 columns the first time any page of
   * that column is asked about. For whoever is about to write the page; a ray
   * passing through asks `reaches` and samples only the columns it crosses.
   */
  stands(px: number, py: number, pz: number): boolean {
    const B = SOLID_WORLD_BRICK_CELLS;
    return py >= 0 && B * py < this.maximumExclusive[1] && this.pageColumnSolidCells(px, pz) > B * py;
  }

  /** Covered share of a cell, 0..255. */
  fraction(x: number, y: number, z: number): number {
    if (y < 0 || y >= this.maximumExclusive[1]) return 0;
    return Math.round(255 * terrainCellSolidFraction(this.height_m(x, z), y * this.hy, this.hy));
  }

  /** Signed distance of the cell at height `y` to the surface of a column `height_m` tall, in 1/256-cell units. */
  distanceQ8(height_m: number, y: number): number {
    return Math.max(-0x8000, Math.min(0x7fff, Math.round((((y + 0.5) * this.hy - height_m) / this.hy) * 256)));
  }

  /**
   * Write the ground into channels laid out x fastest over the box at
   * `minimum`, leaving every cell it does not occupy as it was.
   *
   * Returns whether it wrote any cell, which is how a seeded page learns it
   * has nothing to carry.
   */
  write(minimum: SolidWorldCoordinate, size: SolidWorldCoordinate,
    solidFraction: Uint8Array, signedDistanceQ8: Int16Array, materialId: Uint16Array): boolean {
    const lo = this.minimum, hi = this.maximumExclusive;
    const x0 = Math.max(minimum[0], lo[0]), x1 = Math.min(minimum[0] + size[0], hi[0]);
    const y0 = Math.max(minimum[1], lo[1]), y1 = Math.min(minimum[1] + size[1], hi[1]);
    const z0 = Math.max(minimum[2], lo[2]), z1 = Math.min(minimum[2] + size[2], hi[2]);
    let wrote = false;
    if (y0 >= y1) return wrote;
    for (let z = z0; z < z1; z += 1) for (let x = x0; x < x1; x += 1) {
      const height = this.height_m(x, z);
      const top = Math.min(y1, Math.ceil(height / this.hy));
      for (let y = y0; y < top; y += 1) {
        const fraction = Math.round(255 * terrainCellSolidFraction(height, y * this.hy, this.hy));
        if (fraction === 0) continue;
        const cell = x - minimum[0] + size[0] * (y - minimum[1] + size[1] * (z - minimum[2]));
        solidFraction[cell] = fraction;
        signedDistanceQ8[cell] = this.distanceQ8(height, y);
        materialId[cell] = SOLID_WORLD_TERRAIN_MATERIAL_ID;
        wrote = true;
      }
    }
    return wrote;
  }

  /** Write this page's terrain cells into a fresh, empty page. False when it has none. */
  seed(page: SolidWorldPage): boolean {
    const B = SOLID_WORLD_BRICK_CELLS, q = page.coordinate;
    return this.write([B * q[0], B * q[1], B * q[2]], [B, B, B],
      page.solidFraction, page.signedDistanceQ8, page.materialId);
  }

  /**
   * The ground's own cells of one page as a page, or nothing where it has
   * none there. Baked on first request and the same object ever after, so a
   * consumer that tracks pages by identity sees an unwritten page stay put.
   * Shared and read-only: a patch composes over a copy (`seed`), never this.
   */
  page(coordinate: SolidWorldCoordinate): SolidWorldPage | undefined {
    const B = SOLID_WORLD_BRICK_CELLS, px = coordinate[0], py = coordinate[1], pz = coordinate[2];
    if (!this.stands(px, py, pz)) return undefined;
    const slot = px - floorDiv(this.minimum[0], B) + this.pagesX * (pz - floorDiv(this.minimum[2], B) + this.pagesZ * py);
    let page = this.baked.get(slot);
    if (!page) {
      page = emptyPage([px, py, pz], 1);
      this.seed(page);
      this.baked.set(slot, page);
    }
    return page;
  }

  /**
   * Every page coordinate that holds ground, ascending in (z, y, x), without
   * baking one of them. This samples every column of the ground: it is for a
   * consumer that claims space for the whole terrain, once, at a build.
   */
  pageCoordinates(): SolidWorldCoordinate[] {
    const B = SOLID_WORLD_BRICK_CELLS, coordinates: SolidWorldCoordinate[] = [];
    const px0 = floorDiv(this.minimum[0], B), pz0 = floorDiv(this.minimum[2], B);
    const pagesY = Math.ceil(this.maximumExclusive[1] / B);
    for (let sz = 0; sz < this.pagesZ; sz += 1) for (let py = 0; py < pagesY; py += 1) {
      for (let sx = 0; sx < this.pagesX; sx += 1) {
        if (this.pageColumnSolidCells(px0 + sx, pz0 + sz) > B * py) coordinates.push([px0 + sx, py, pz0 + sz]);
      }
    }
    return coordinates;
  }

  /** Bytes this ground holds: the heights sampled so far and the pages baked so far. */
  get heldBytes(): number {
    let tiles = 0;
    for (const tile of this.tiles) if (tile) tiles += 1;
    return tiles * 4 * SOLID_WORLD_BRICK_CELLS ** 2 + this.pageTops.byteLength
      + this.baked.size * 5 * SOLID_WORLD_VOXELS_PER_PAGE;
  }
}

const groundsByTerrain = new WeakMap<TerrainDescription, SolidGround>();
/**
 * Grounds still in use, newest first, found again by content.
 *
 * `cloneScene` is a JSON round trip and the worker hand-off a structured
 * clone, so undo, redo, a loaded document and every published scene arrive
 * with a terrain object nobody has seen. Its content stamp names the ground it
 * equals, and taking that one keeps the sampled columns, the baked pages, and
 * everything built over them. Weak: a ground lives as long as a world, an
 * index or a reader holds it and no longer.
 */
const recentGrounds: WeakRef<SolidGround>[] = [];
const RECENT_GROUNDS = 4;

/**
 * The ground for a scene, or `undefined` when it has no terrain to stand on.
 *
 * Two scenes whose terrain, container and lattice are equal get the same
 * object, so identity is the test for "the same ground".
 */
export function solidGroundForScene(
  scene: Pick<SceneDescription, "terrain" | "container" | "voxelDomain">,
): SolidGround | undefined {
  const terrain = scene.terrain;
  if (!terrain || !sceneHasTerrain(scene)) return undefined;
  const extent = terrainWorldContainer(scene), c = scene.container;
  const lattice = `${c.width_m}:${c.height_m}:${c.depth_m}:${sceneCellSizes_m(scene).join(":")}`
    + `:${extent.width_m}:${extent.height_m}:${extent.depth_m}`;
  const known = groundsByTerrain.get(terrain);
  // The stamp is only paid for a terrain object not seen before on this lattice.
  if (known?.key.endsWith(`|${lattice}`)) return known;
  const key = `${terrainContentStamp(terrain)}|${lattice}`;
  let ground: SolidGround | undefined;
  for (let index = recentGrounds.length - 1; index >= 0; index -= 1) {
    const candidate = recentGrounds[index]!.deref();
    if (!candidate) recentGrounds.splice(index, 1);
    else if (candidate.key === key) ground = candidate;
  }
  if (!ground) {
    ground = new SolidGround(key, terrain, scene);
    recentGrounds.unshift(new WeakRef(ground));
    recentGrounds.length = Math.min(recentGrounds.length, RECENT_GROUNDS);
  }
  groundsByTerrain.set(terrain, ground);
  return ground;
}

/**
 * The scene's solids: an analytic ground, and the pages patches have written.
 *
 * `pages` holds a page for every page coordinate some patch has written, and
 * nothing else. **Where a page stands it is the whole answer for its 512
 * cells; where none stands the ground answers**, from its heights, and where
 * there is no ground the cell is empty. A page a clear emptied therefore stays
 * in the list for as long as there is ground under it to keep dug.
 *
 * Nothing here is proportional to the terrain: a document's world costs its
 * patches. Ask `solidWorldPageAt`, `sampleSolidWorld` or the ground itself for
 * what lies outside the pages; `materializedSolidWorldPages` is for the
 * consumer that really does need every cell of the scene as bits.
 */
export interface SolidWorld {
  /** Every page a patch has written, ascending in (z, y, x). The ground is not in it. */
  readonly pages: readonly SolidWorldPage[];
  /** Lookup into `pages`. A miss is unwritten space, which over a ground is not empty space. */
  readonly directory: SolidWorldDirectory;
  /** The terrain under the pages. Absent when the scene has none. */
  readonly ground?: SolidGround;
  readonly patches: readonly SolidWorldVoxelPatch[];
  /** Compact ordered boxes evaluated beside sparse voxel pages. Large planar
   * colliders stay exact without expanding their area into page payload. */
  readonly regions?: readonly SolidWorldVoxelPatch[];
}

/**
 * Refuse a region that is not a whole box.
 *
 * A region is evaluated beside the pages by testing a cell against its box and
 * nothing else. A round kind's box is not its shape, so it has to be baked
 * into pages; read as a region it would be solid out to its corners.
 */
export function assertSolidWorldRegionIsBox(region: SolidWorldVoxelPatch): void {
  if (solidPatchKindId(region) !== SOLID_PATCH_BOX_KIND) {
    throw new RangeError(`A SolidWorld region must be a whole box; a "${solidPatchKindId(region)}" patch is baked into pages`);
  }
}

/**
 * Cheap canonical identity for every authored value that changes SolidWorld.
 *
 * Terrain samples are represented by their existing streaming content stamp;
 * voxel edits are folded in document order because patch composition is
 * ordered. Physical extents and realized lattice dimensions are included so a
 * scale change cannot reuse pages under a stale metre mapping.
 */
interface PatchFold { readonly length: number; readonly low: number; readonly high: number }
const patchFolds = new WeakMap<readonly SolidWorldVoxelPatch[], PatchFold>();
/** Stable words for the three operations. Fill and clear keep the values every stamp to date was folded with. */
const OPERATION_FOLD_WORDS = { fill: 1, clear: 2, paint: 3 } as const;
/**
 * Order-sensitive fold of an authored patch list, kept per array.
 *
 * A stroke appends to an immutable list, so the fold of the longer list is the
 * fold of the shorter one continued (`from`): stamping an edited scene costs
 * the stroke, not the scene's whole editing history.
 *
 * A box folds exactly as it did before patches had kinds, so the stamp of a
 * document made only of boxes has not moved. Any other kind adds its id and
 * whatever else it says identifies it.
 */
function solidVoxelPatchFold(patches: readonly SolidWorldVoxelPatch[], from?: PatchFold): PatchFold {
  const cached = patchFolds.get(patches);
  if (cached && cached.length === patches.length) return cached;
  const resume = cached && cached.length < patches.length ? cached
    : from && from.length <= patches.length ? from : undefined;
  let low = resume?.low ?? 0x811c_9dc5, high = resume?.high ?? 0x9e37_79b9;
  const foldWord = (word: number) => {
    low = Math.imul(low ^ word, 0x0100_0193) >>> 0;
    high = Math.imul(high ^ (word >>> 16 | word << 16), 0x85eb_ca6b) >>> 0;
  };
  for (let index = resume?.length ?? 0; index < patches.length; index += 1) {
    const patch = patches[index]!;
    const operationWord = OPERATION_FOLD_WORDS[patch.operation];
    if (operationWord === undefined) throw new RangeError(`Unsupported solid patch operation "${String(patch.operation)}"`);
    foldWord(operationWord);
    for (const value of patch.minimum) foldWord(value | 0);
    for (const value of patch.maximumExclusive) foldWord(value | 0);
    foldWord(patch.materialId ?? 1);
    // Preserve every legacy stamp; only explicit scenery cuts add this tag.
    if (patch.affectsScenery === true) foldWord(0x5343_5554);
    const kindId = solidPatchKindId(patch);
    if (kindId === SOLID_PATCH_BOX_KIND) continue;
    for (let character = 0; character < kindId.length; character += 1) foldWord(kindId.charCodeAt(character));
    foldWord(0xff);
    solidPatchKinds.get(patch).fold(patch, foldWord);
  }
  const result = { length: patches.length, low, high };
  patchFolds.set(patches, result);
  return result;
}

/** Compact identity of an authored patch list, for keys that are compared every frame. */
export function solidVoxelPatchesKey(patches: readonly SolidWorldVoxelPatch[]): string {
  const fold = solidVoxelPatchFold(patches);
  return `${fold.length}:${fold.high.toString(16)}:${fold.low.toString(16)}`;
}

export function solidWorldContentStamp(scene: SceneDescription): string {
  let low = 0x811c_9dc5, high = 0x9e37_79b9;
  const foldWord = (word: number) => {
    low = Math.imul(low ^ word, 0x0100_0193) >>> 0;
    high = Math.imul(high ^ (word >>> 16 | word << 16), 0x85eb_ca6b) >>> 0;
  };
  const foldString = (value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      foldWord(value.charCodeAt(index));
    }
    foldWord(0xff);
  };
  const dimensions = sceneLatticeDimensions(scene);
  foldString(terrainContentStamp(scene.terrain));
  foldString(`${scene.container.width_m}:${scene.container.height_m}:`
    + `${scene.container.depth_m}:${dimensions.join(":")}`);
  const patches = solidVoxelPatchFold(scene.solidVoxels);
  foldWord(patches.length); foldWord(patches.low); foldWord(patches.high);
  return `solid-world-v1:${high.toString(16).padStart(8, "0")}`
    + low.toString(16).padStart(8, "0");
}

/** Page coordinates either side of the origin that pack exactly into one number. */
const NEAR_PAGES = 1 << 16;
const writtenPageKey = (x: number, y: number, z: number): number | string =>
  x >= -NEAR_PAGES && x < NEAR_PAGES && y >= -NEAR_PAGES && y < NEAR_PAGES && z >= -NEAR_PAGES && z < NEAR_PAGES
    ? ((z + NEAR_PAGES) * (2 * NEAR_PAGES) + (y + NEAR_PAGES)) * (2 * NEAR_PAGES) + (x + NEAR_PAGES)
    // A patch may stand anywhere in i32; past half a million cells out the key is spelled instead.
    : `${x},${y},${z}`;

const pageHoldsSolid = (page: SolidWorldPage): boolean => {
  const fractions = page.solidFraction;
  for (let cell = 0; cell < fractions.length; cell += 1) if (fractions[cell]! > 0) return true;
  return false;
};

/**
 * Whether a written page stays in a world's list. One that holds solid does.
 * One a clear emptied does only over ground: there it is what keeps the dig
 * dug, and anywhere else an empty page says nothing an absent one does not.
 */
const pageIsKept = (page: SolidWorldPage, ground: SolidGround | undefined): boolean =>
  pageHoldsSolid(page) || (ground?.stands(page.coordinate[0], page.coordinate[1], page.coordinate[2]) ?? false);

/** A page a bake wrote, with where it stands in the base's list: its own index there, or the one it would take. */
interface WrittenPage { readonly page: SolidWorldPage; readonly baseIndex: number; readonly inBase: boolean }

/**
 * What one bake changed, kept per page list (a wrapper that adds regions shares
 * its source's list): the list it was baked over, and every page coordinate at
 * which the two lists hold different pages, ascending in (z, y, x).
 *
 * The base is held weakly. A document is a chain of thousands of strokes, and
 * a strong reference would keep every world of it alive behind the last.
 */
interface SolidWorldPageDelta {
  readonly base: WeakRef<readonly SolidWorldPage[]>;
  readonly changed: readonly SolidWorldCoordinate[];
  /** Some patch of the bake was not a paint. A paint writes material where a cell is already solid and nothing else. */
  readonly movesSolid: boolean;
}
const pageDeltas = new WeakMap<readonly SolidWorldPage[], SolidWorldPageDelta>();
/** Strokes walked back before a whole-world comparison is the better answer. */
const MAX_PAGE_DELTA_STROKES = 256;

/**
 * Every page coordinate at which `next` may hold a different page than
 * `previous`, ascending in (z, y, x), when `next` was composed from `previous`
 * by appending strokes; undefined when it was not (an undo, a branch, a
 * document composed again, another document) or when that history is no
 * longer held.
 *
 * This is the composition's own record of the pages it copied, created and dropped,
 * so a consumer that folds a world in by its pages pays for the stroke and
 * never for the pages the world holds.
 */
export function solidWorldChangedPages(previous: SolidWorld, next: SolidWorld):
  readonly SolidWorldCoordinate[] | undefined {
  if (previous.pages === next.pages) return [];
  const budget = previous.pages.length + next.pages.length;
  const strokes: (readonly SolidWorldCoordinate[])[] = [];
  let pages = next.pages, visited = 0;
  while (strokes.length < MAX_PAGE_DELTA_STROKES) {
    const delta = pageDeltas.get(pages), base = delta?.base.deref();
    if (!delta || !base) return undefined;
    // Past the pages the two worlds hold, comparing those is the cheaper walk.
    if ((visited += delta.changed.length) > budget) return undefined;
    strokes.push(delta.changed);
    if (base !== previous.pages) { pages = base; continue; }
    if (strokes.length === 1) return delta.changed;
    // Several strokes: one ascending list with each coordinate once.
    const merged = strokes.flat().sort(solidWorldPageOrder);
    let kept = 0;
    for (const coordinate of merged) {
      if (kept === 0 || solidWorldPageOrder(merged[kept - 1]!, coordinate) !== 0) merged[kept++] = coordinate;
    }
    merged.length = kept;
    return merged;
  }
  return undefined;
}

/**
 * Whether `next` is `previous` repainted: baked from it by strokes that held
 * nothing but paint. Coverage and distance are then the same at every cell, a
 * page neither came nor went, and a consumer of occupancy alone has nothing to
 * read. Answered from the strokes, not from the pages they copied.
 *
 * False for anything else, including a pair whose history is no longer held:
 * that is "not known", and the pages say.
 */
export function solidWorldOnlyRepainted(previous: SolidWorld, next: SolidWorld): boolean {
  let pages = next.pages;
  for (let strokes = 0; pages !== previous.pages; strokes += 1) {
    const delta = pageDeltas.get(pages), base = delta?.base.deref();
    if (!delta || !base || delta.movesSolid || strokes >= MAX_PAGE_DELTA_STROKES) return false;
    pages = base;
  }
  return true;
}

/**
 * Compose patches over a world. The cost is the pages the patches touch: the
 * base's list is searched for each, the result's list is the base's with those
 * entries replaced, and only a stroke that adds or drops a page copies the
 * list's pointers into a new order.
 *
 * Over a ground, a page no patch has written yet starts as the ground's own
 * cells, so a fractional fill, a paint or a clear composes against what was
 * really there, and a cell the patch leaves alone still reads as ground.
 */
function applySolidWorldPatches(
  base: SolidWorld,
  patches: readonly SolidWorldVoxelPatch[],
  cloneBase = true,
): SolidWorld {
  const ground = base.ground;
  // Pages this call has written. Every other page of the result is the base's own.
  const written = new Map<number | string, WrittenPage>();
  // Copy on write: a page of the base is copied the first time a patch asks to write it.
  const store: SolidPatchPageStore = {
    page(coordinate, create) {
      const x = coordinate[0], y = coordinate[1], z = coordinate[2], pageKey = writtenPageKey(x, y, z);
      const held = written.get(pageKey);
      if (held) return held.page;
      const baseIndex = pageLowerBound(base.pages, x, y, z), source = base.pages[baseIndex];
      const inBase = source !== undefined && source.coordinate[0] === x && source.coordinate[1] === y
        && source.coordinate[2] === z;
      let page: SolidWorldPage;
      if (!inBase) {
        const grounded = ground?.stands(x, y, z) ?? false;
        // Nothing is solid here, and only a fill makes it so.
        if (!create && !grounded) return undefined;
        // A page born inside a copy-on-write edit carries the edit's revision, as a cloned one does.
        page = emptyPage(coordinate, cloneBase ? 2 : 1);
        if (grounded) ground!.seed(page);
      } else if (cloneBase) {
        page = { ...source, authored: source.authored?.slice(), solidFraction: source.solidFraction.slice(),
          signedDistanceQ8: source.signedDistanceQ8.slice(),
          materialId: source.materialId.slice(), revision: source.revision + 1 };
      } else {
        page = source;
      }
      written.set(pageKey, { page, baseIndex, inBase });
      return page;
    },
  };
  for (const patch of patches) {
    validateSolidPatch(patch);
    composeSolidPatch(patch, store);
  }
  const ordered = [...written.values()].sort((left, right) =>
    solidWorldPageOrder(left.page.coordinate, right.page.coordinate));
  // Only a page this call wrote can have stopped being worth keeping.
  const solid = ordered.map((entry) => pageIsKept(entry.page, ground));
  const changed: SolidWorldCoordinate[] = [];
  let pages: SolidWorldPage[];
  if (ordered.every((entry, index) => entry.inBase && solid[index])) {
    // No page came or went: the base's order stands, with the written pages in their places.
    pages = base.pages.slice();
    for (const entry of ordered) {
      pages[entry.baseIndex] = entry.page;
      changed.push(entry.page.coordinate);
    }
  } else {
    pages = [];
    let cursor = 0;
    ordered.forEach((entry, index) => {
      for (; cursor < entry.baseIndex; cursor += 1) pages.push(base.pages[cursor]!);
      if (entry.inBase) cursor += 1;
      if (solid[index]) pages.push(entry.page);
      if (entry.inBase || solid[index]) changed.push(entry.page.coordinate);
    });
    for (; cursor < base.pages.length; cursor += 1) pages.push(base.pages[cursor]!);
  }
  // A base written in place is not a world anyone else holds, so nothing was changed *from* it.
  if (cloneBase) {
    pageDeltas.set(pages, { base: new WeakRef(base.pages), changed,
      movesSolid: patches.some((patch) => patch.operation !== "paint") });
  }
  return { pages, directory: new SolidWorldDirectory(pages), ground, patches: base.patches.concat(patches) };
}

/** Compose authored voxel patches over an already migrated uniform solid world. */
export function withSolidWorldPatches(
  base: SolidWorld,
  patches: readonly SolidWorldVoxelPatch[],
): SolidWorld {
  return patches.length === 0 ? base : applySolidWorldPatches(base, patches);
}

/** Compile generic voxel edits into the sole static solid authority. */
export function createSolidWorld(
  patches: readonly SolidWorldVoxelPatch[] = [],
): SolidWorld {
  const pages: SolidWorldPage[] = [];
  return withSolidWorldPatches({ pages, directory: new SolidWorldDirectory(pages), patches: [] }, patches);
}

/** The scene's ground with no patch over it: no page, and nothing sampled. */
function groundSolidWorldForScene(scene: SceneDescription): SolidWorld {
  const pages: SolidWorldPage[] = [];
  return { pages, directory: new SolidWorldDirectory(pages), ground: solidGroundForScene(scene), patches: [] };
}

const sceneSolidWorldCache = new WeakMap<SceneDescription, {
  readonly stamp: string;
  readonly world: SolidWorld;
}>();
/** The world last built, by content stamp. An edit that leaves the solids
 * alone (a refinement region, a fluid parameter) arrives as a new scene
 * object, always so across the worker hand-off, and takes the world its stamp
 * names: no patch replay, and a consumer that folds a later world in by page
 * identity (SolidOccupancyMask) sees the same world. Weak, so the memo never
 * outlives the world's own holders. */
let latestSolidWorld: { readonly stamp: string; readonly world: WeakRef<SolidWorld> } | undefined;

/**
 * Solid over the cells a fluid lattice holds past its container's high faces.
 *
 * Ordinary filled slabs across the lattice and its one-cell halo, which is
 * where the box shell authors a wall: past the +x face, past the +z face, and
 * above a closed top. An open top gets no slab above it, so those cells are
 * air between walls that rise to the lattice's top, and no slab reaches above
 * that top. None when the lattice is the container.
 */
export function solidVoxelPaddingForLattice(
  cells: SolidWorldCoordinate,
  lattice: SolidWorldCoordinate,
  top: "open" | "closed",
): readonly SolidWorldVoxelPatch[] {
  const [cx, cy, cz] = cells, [nx, ny, nz] = lattice;
  if (cells.some((n, axis) => !Number.isSafeInteger(n) || n <= 0 || n > lattice[axis]!)) {
    throw new RangeError(`A fluid lattice of ${lattice.join("x")} cells does not hold a container of ${cells.join("x")}`);
  }
  // Under an open top the slabs stop at the lattice's top: above it is the world's air, outside the lattice.
  const fill = (minimum: SolidWorldCoordinate): SolidWorldVoxelPatch =>
    ({ operation: "fill", minimum, maximumExclusive: [nx + 1, top === "open" ? ny : ny + 1, nz + 1], materialId: 1 });
  return [
    ...(nx > cx ? [fill([cx, -1, -1])] : []),
    ...(nz > cz ? [fill([-1, -1, cz])] : []),
    ...(ny > cy && top === "closed" ? [fill([-1, cy, -1])] : []),
  ];
}

/** The padded world of a scene's canonical world, by lattice: one composition a world. */
const fluidLatticeSolidWorlds = new WeakMap<SolidWorld, { readonly key: string; readonly world: SolidWorld }>();

/**
 * The solid world a fluid solver reads: the scene's own, with the padding of
 * its lattice filled (`solidVoxelPaddingForLattice`). It is the scene's world
 * itself when the container is whole pages, and otherwise shares every page
 * the padding does not write, so a consumer that folds worlds in by page
 * identity pays for an edit what it paid before.
 */
export function solidWorldForFluidLattice(scene: SceneDescription): SolidWorld {
  const world = solidWorldForScene(scene);
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const lattice = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  // A container without walls pads nothing: past its high faces is the open world's air.
  const padding = scene.container.walls === "none" ? [] : solidVoxelPaddingForLattice(cells, lattice, scene.container.top);
  if (padding.length === 0) return world;
  const key = `${cells.join(":")}:${lattice.join(":")}:${scene.container.top}`, held = fluidLatticeSolidWorlds.get(world);
  if (held?.key === key) return held.world;
  const padded = applySolidWorldPatches(world, padding);
  fluidLatticeSolidWorlds.set(world, { key, world: padded });
  return padded;
}

export function solidWorldForScene(scene: SceneDescription): SolidWorld {
  const stamp = solidWorldContentStamp(scene);
  const cached = sceneSolidWorldCache.get(scene);
  if (cached?.stamp === stamp) return cached.world;
  // The ground is taken as it is described and only the document's patches
  // are composed, in place over pages nobody else holds. The returned snapshot
  // is canonical for this scene content stamp.
  const world = (latestSolidWorld?.stamp === stamp ? latestSolidWorld.world.deref() : undefined)
    ?? applySolidWorldPatches(groundSolidWorldForScene(scene), scene.solidVoxels, false);
  sceneSolidWorldCache.set(scene, { stamp, world });
  latestSolidWorld = { stamp, world: new WeakRef(world) };
  return world;
}

/**
 * The scene's terrain under a different ordered patch list than its document's.
 *
 * The ground is the world's base rather than a patch, so a consumer that wants
 * a subset of `scene.solidVoxels` — presentation, which leaves the canonical
 * tank shell out — cannot filter `world.patches` and compose over nothing:
 * that loses the ground. This is the same composition `solidWorldForScene`
 * performs, over the list it is handed.
 */
export function terrainSolidWorldWithPatches(
  scene: SceneDescription,
  patches: readonly SolidWorldVoxelPatch[],
): SolidWorld {
  return applySolidWorldPatches(groundSolidWorldForScene(scene), patches, false);
}

/**
 * Static scenery surfaces explicitly authored as fluid colliders.
 *
 * Scenery is normally presentation-only and must not silently become physics.
 * A `fluid-collider` tag opts an axis-aligned box into the same signed voxel
 * authority as terrain and document voxel edits. This is what makes the finite
 * studio slab support an outside-tank drop without inventing an infinite y=0
 * floor. Non-box and rotated colliders need a proper voxelizer before they can
 * opt in; using their AABB here would create solid space the scene does not own.
 */
export function fluidColliderVoxelPatchesForScene(
  scene: SceneDescription,
): readonly SolidWorldVoxelPatch[] {
  const environmentId = scene.environment ?? "default";
  const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
  const cell = sceneCellSizes_m(scene);
  const origin = [-0.5 * scene.container.width_m, 0,
    -0.5 * scene.container.depth_m] as const;
  const epsilon = 1e-9;
  return environmentProxyPrimitives(catalog).flatMap((primitive) => {
    if (!primitive.tags.includes("fluid-collider") || primitive.kind !== "box"
      || primitive.orientation) return [];
    const minimum = [primitive.aabb_m.min.x, primitive.aabb_m.min.y,
      primitive.aabb_m.min.z].map((value, axis) =>
      Math.floor((value - origin[axis]!) / cell[axis]! + epsilon)) as
      [number, number, number];
    const maximumExclusive = [primitive.aabb_m.max.x, primitive.aabb_m.max.y,
      primitive.aabb_m.max.z].map((value, axis) =>
      Math.ceil((value - origin[axis]!) / cell[axis]! - epsilon) || 0) as
      [number, number, number];
    if (minimum.some((value, axis) => value >= maximumExclusive[axis]!)) return [];
    return [{ operation: "fill" as const, minimum, maximumExclusive }];
  });
}

const fluidSceneWorldCache = new WeakMap<SceneDescription, {
  readonly source: SolidWorld; readonly regionStamp: string; readonly world: SolidWorld;
}>();

/** SolidWorld used by fluid dynamics, including explicitly opted-in scenery.
 * Preserve its identity between preflight and commit, including collider
 * wrappers. The structural stamp also detects callers mutating scene graphs. */
export function fluidSolidWorldForScene(scene: SceneDescription): SolidWorld {
  const world = solidWorldForScene(scene);
  const regions = fluidColliderVoxelPatchesForScene(scene);
  if (regions.length === 0) return world;
  const regionStamp = JSON.stringify(regions), cached = fluidSceneWorldCache.get(scene);
  if (cached?.source === world && cached.regionStamp === regionStamp) return cached.world;
  const wrapped = { ...world, regions: [...(world.regions ?? []), ...regions] };
  fluidSceneWorldCache.set(scene, { source: world, regionStamp, world: wrapped });
  return wrapped;
}

/** Generic voxel-box authoring helper; runtime consumers never infer this shell. */
export function boxSolidVoxelShell(
  dimensions: SolidWorldCoordinate,
  options: { readonly top?: "open" | "closed"; readonly materialId?: number } = {},
): readonly SolidWorldVoxelPatch[] {
  const [nx, ny, nz] = dimensions;
  if (!dimensions.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("Voxel shell dimensions must be positive integers");
  }
  const materialId = options.materialId ?? 1;
  const fill = (minimum: SolidWorldCoordinate,
    maximumExclusive: SolidWorldCoordinate): SolidWorldVoxelPatch =>
    ({ operation: "fill", minimum, maximumExclusive, materialId });
  return [
    fill([0, -1, 0], [nx, 0, nz]),
    fill([-1, 0, 0], [0, ny, nz]),
    fill([nx, 0, 0], [nx + 1, ny, nz]),
    fill([0, 0, -1], [nx, ny, 0]),
    fill([0, 0, nz], [nx, ny, nz + 1]),
    ...(options.top === "closed" ? [fill([0, ny, 0], [nx, ny + 1, nz])] : []),
  ];
}

/** Compile a closed spherical vessel into ordinary filled voxel runs. */
export function sphericalSolidVoxelShell(
  dimensions: SolidWorldCoordinate,
  materialId = 1,
): readonly SolidWorldVoxelPatch[] {
  const [nx, ny, nz] = dimensions;
  if (!dimensions.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("Spherical voxel shell dimensions must be positive integers");
  }
  const radius = 0.5 * Math.min(nx, ny, nz);
  const center = [0.5 * nx, 0.5 * ny, 0.5 * nz] as const;
  const patches: SolidWorldVoxelPatch[] = [];
  for (let z = -1; z <= nz; z += 1) for (let y = -1; y <= ny; y += 1) {
    const solid = (x: number): boolean => Math.hypot(x + 0.5 - center[0],
      y + 0.5 - center[1], z + 0.5 - center[2]) >= radius;
    for (let x = -1; x <= nx;) {
      while (x <= nx && !solid(x)) x += 1;
      const start = x;
      while (x <= nx && solid(x)) x += 1;
      if (x > start) patches.push({ operation: "fill", minimum: [start, y, z],
        maximumExclusive: [x, y + 1, z + 1], materialId });
    }
  }
  return patches;
}

/**
 * One cell of the world: its page's where a patch has written the page, the
 * ground's otherwise, read from the column's height without baking anything.
 *
 * The distance of an empty cell no page holds is +Infinity: nothing is near.
 */
export function sampleSolidWorld(
  world: SolidWorld,
  coordinate: SolidWorldCoordinate,
  separateTerrain = false,
): { readonly solidFraction: number; readonly signedDistance_cells: number;
  readonly materialId: number } {
  const address = solidWorldPageAddress(coordinate);
  const pageIndex = world.directory.lookup(address.page);
  let page = pageIndex === undefined ? undefined : world.pages[pageIndex]!;
  if (separateTerrain && world.ground && page
    && (!page.authored || !(page.authored[address.localIndex >>> 3]! & (1 << (address.localIndex & 7))))) page = undefined;
  let solidFraction = page?.solidFraction[address.localIndex] ?? 0;
  let signedDistanceQ8 = page?.signedDistanceQ8[address.localIndex] ?? 0x7fff;
  let materialId = page?.materialId[address.localIndex] ?? 0;
  let held = page !== undefined;
  const ground = held || separateTerrain ? undefined : world.ground;
  if (ground) {
    const x = coordinate[0], y = coordinate[1], z = coordinate[2];
    solidFraction = ground.fraction(x, y, z);
    if (solidFraction > 0) {
      signedDistanceQ8 = ground.distanceQ8(ground.height_m(x, z), y);
      materialId = SOLID_WORLD_TERRAIN_MATERIAL_ID;
      held = true;
    }
  }
  let regionMatched = false;
  for (const region of world.regions ?? []) {
    if (!coordinate.every((value, axis) => value >= region.minimum[axis]!
      && value < region.maximumExclusive[axis]!)) continue;
    assertSolidWorldRegionIsBox(region);
    // A paint recolours what is solid and makes nothing so.
    if (region.operation === "paint") {
      if (solidFraction > 0) materialId = region.materialId ?? 1;
      continue;
    }
    regionMatched = true;
    const fill = region.operation === "fill";
    solidFraction = fill ? 255 : 0;
    signedDistanceQ8 = fill ? -128 : 0x7fff;
    materialId = fill ? (region.materialId ?? 1) : 0;
  }
  return { solidFraction: solidFraction / 255,
    signedDistance_cells: !held && !regionMatched
      ? Number.POSITIVE_INFINITY : signedDistanceQ8 / 256, materialId };
}

/**
 * The page that answers for a page coordinate: the one a patch wrote, else the
 * ground's own cells there, else nothing, and the space is empty.
 *
 * A ground page is baked the first time it is asked for and is the same object
 * ever after (`SolidGround.page`), so this is for a consumer that needs the
 * bits of the pages it names — the pages under a stroke, an edit's neighbours
 * — and never for a walk of the scene.
 */
export function solidWorldPageAt(world: SolidWorld, coordinate: SolidWorldCoordinate): SolidWorldPage | undefined {
  const index = world.directory.lookup(coordinate);
  return index !== undefined ? world.pages[index] : world.ground?.page(coordinate);
}

/** Walk the written pages that hold solid and the ground's page coordinates no patch has written, ascending in (z, y, x). */
function forEachSolidPage(world: SolidWorld, written: (page: SolidWorldPage) => void,
  unwritten: (coordinate: SolidWorldCoordinate) => void): void {
  const pages = world.pages;
  let cursor = 0;
  for (const coordinate of world.ground?.pageCoordinates() ?? []) {
    let order = -1;
    while (cursor < pages.length && (order = solidWorldPageOrder(pages[cursor]!.coordinate, coordinate)) < 0) {
      if (pageHoldsSolid(pages[cursor]!)) written(pages[cursor]!);
      cursor += 1;
    }
    if (cursor < pages.length && order === 0) {
      if (pageHoldsSolid(pages[cursor]!)) written(pages[cursor]!);
      cursor += 1;
    } else unwritten(coordinate);
  }
  for (; cursor < pages.length; cursor += 1) if (pageHoldsSolid(pages[cursor]!)) written(pages[cursor]!);
}

/**
 * Every page coordinate at which the world holds any solid, the ground's
 * included, ascending in (z, y, x). No page is baked for it, but over a ground
 * it samples every column: it is what a build claims space from, once.
 */
export function solidWorldPageCoordinates(world: SolidWorld): SolidWorldCoordinate[] {
  const coordinates: SolidWorldCoordinate[] = [];
  forEachSolidPage(world, (page) => coordinates.push(page.coordinate), (coordinate) => coordinates.push(coordinate));
  return coordinates;
}

const materializedPages = new WeakMap<readonly SolidWorldPage[], readonly SolidWorldPage[]>();
/**
 * Every page that holds any solid, the ground's included, as pages of cells,
 * ascending in (z, y, x).
 *
 * This is the scene-sized image the world exists to avoid: megabytes of ground
 * that is a function of its heights. It is for the consumer that has no other
 * way to read the ground than as bits — a display with no heightfield of its
 * own to sample, a CPU oracle — and it is what such a consumer pays, once per
 * ground: a ground page is the same object in every world over that ground, so
 * a later world costs the merge of its written pages into the list and an
 * identity-tracking uploader sends only what an edit changed.
 */
export function materializedSolidWorldPages(world: SolidWorld): readonly SolidWorldPage[] {
  const ground = world.ground;
  if (!ground) return world.pages;
  let pages = materializedPages.get(world.pages);
  if (!pages) {
    const list: SolidWorldPage[] = [];
    forEachSolidPage(world, (page) => list.push(page), (coordinate) => list.push(ground.page(coordinate)!));
    materializedPages.set(world.pages, pages = list);
  }
  return pages;
}

/** Compile only this stroke over an immutable base; the ground and prior edits are reused. */
export function sceneWithSolidStroke(base: SceneDescription,
  patches: readonly SolidWorldVoxelPatch[]): SceneDescription {
  const scene = { ...base, solidVoxels: base.solidVoxels.concat(patches) };
  solidVoxelPatchFold(scene.solidVoxels, solidVoxelPatchFold(base.solidVoxels));
  // A browser input transaction may hold no image of its document; authoring
  // a descriptor must not synchronously compose the document's whole history
  // on that thread. Runtime owners already holding the base image retain the
  // incremental path.
  const cached = sceneSolidWorldCache.get(base);
  if (cached && cached.stamp === solidWorldContentStamp(base)) {
    const world = withSolidWorldPatches(cached.world, patches);
    sceneSolidWorldCache.set(scene, { stamp: solidWorldContentStamp(scene), world });
  }
  return scene;
}

/** Preserve a compiled immutable image across worker structured-clone boundaries. */
export function reuseSolidWorld(source: SceneDescription, target: SceneDescription): void {
  const cached = sceneSolidWorldCache.get(source);
  if (cached && cached.stamp === solidWorldContentStamp(target)) sceneSolidWorldCache.set(target, cached);
}

/**
 * Whether this document object already holds the image of its own content, so
 * that asking for its world composes nothing. True of a document composed in this
 * realm by `sceneWithSolidStroke` over a base that held one, and of any that
 * `solidWorldForScene` or `reuseSolidWorld` has answered for.
 */
export function sceneHoldsSolidWorld(scene: SceneDescription): boolean {
  return sceneSolidWorldCache.get(scene)?.stamp === solidWorldContentStamp(scene);
}
