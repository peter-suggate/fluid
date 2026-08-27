import { signedSpatialCoordinateHash } from "./signed-spatial-hash";
import type { SceneDescription } from "./model";
import { sceneCellSizes_m, sceneLatticeDimensions } from "./scene-lattice-dimensions";
import {
  PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO,
  type PlanarBoundaryPatch,
  type PlanarBoundaryVec3,
} from "./planar-boundary";
import {
  terrainCellSolidFraction,
  terrainColumnHeights,
  terrainContentStamp,
} from "./terrain";
import {
  buildEnvironmentProxyCatalog,
  environmentProxyPrimitives,
} from "./voxel-environments";

export const SOLID_WORLD_VERSION = 1;
export const SOLID_WORLD_BRICK_CELLS = 8;
export const SOLID_WORLD_VOXELS_PER_PAGE = SOLID_WORLD_BRICK_CELLS ** 3;
export const SOLID_WORLD_EMPTY_PAGE = 0xffff_ffff;
export const SOLID_WORLD_STATIC_PAGE_BYTES = 2560;
export const SOLID_WORLD_DYNAMIC_PAGE_BYTES = 5120;
export const SOLID_WORLD_DIRECTORY_ENTRY_BYTES = 7 * 4;
/** Stable material-table slot shared with the voxel scene renderer. */
export const SOLID_WORLD_TERRAIN_MATERIAL_ID = 2;

export type SolidWorldCoordinate = readonly [number, number, number];

export interface SolidWorldVoxelPatch {
  readonly operation: "fill" | "clear";
  readonly minimum: SolidWorldCoordinate;
  readonly maximumExclusive: SolidWorldCoordinate;
  readonly materialId?: number;
}

export interface SolidWorldVoxelPatchBounds {
  readonly minimum: PlanarBoundaryVec3;
  readonly maximum: PlanarBoundaryVec3;
}

/** Exact metre-space bounds of one signed-lattice voxel edit box. */
export function solidWorldVoxelPatchBounds_m(
  scene: SceneDescription,
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

/** Compile a thin filled voxel box into the same exact slab used by SVO rays. */
export function planarBoundaryForSolidWorldVoxelPatch(
  scene: SceneDescription,
  patch: SolidWorldVoxelPatch,
  ownerId = 0xffff,
): PlanarBoundaryPatch | null {
  if (patch.operation !== "fill") return null;
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

interface DirectoryEntry {
  readonly hash: number;
  readonly coordinate: SolidWorldCoordinate;
  readonly page: number;
}

/** CPU construction/oracle form of the exact signed sparse page directory. */
export class SolidWorldDirectory {
  readonly capacity: number;
  readonly mask: number;
  private readonly entries: Array<DirectoryEntry | undefined>;
  private liveCount = 0;

  constructor(readonly pageCapacity: number) {
    if (!Number.isSafeInteger(pageCapacity) || pageCapacity < 0) {
      throw new RangeError("SolidWorld directory page capacity is invalid");
    }
    this.capacity = nextPowerOfTwo(Math.max(2, 2 * pageCapacity));
    this.mask = this.capacity - 1;
    this.entries = new Array(this.capacity);
  }

  insert(coordinate: SolidWorldCoordinate, page: number): void {
    if (!Number.isSafeInteger(page) || page < 0 || page >= this.pageCapacity) {
      throw new RangeError("SolidWorld page index exceeds capacity");
    }
    const hash = signedSpatialCoordinateHash(coordinate);
    let slot = hash & this.mask;
    for (let probe = 0; probe < this.capacity; probe += 1) {
      const entry = this.entries[slot];
      if (!entry) {
        this.entries[slot] = { hash, coordinate: [...coordinate], page };
        this.liveCount += 1;
        return;
      }
      if (entry.hash === hash && entry.coordinate.every((value, axis) =>
        value === coordinate[axis])) {
        if (entry.page !== page) throw new Error("SolidWorld coordinate has two page owners");
        return;
      }
      slot = (slot + 1) & this.mask;
    }
    throw new Error("SolidWorld directory probe exhausted");
  }

  lookup(coordinate: SolidWorldCoordinate): number | undefined {
    const hash = signedSpatialCoordinateHash(coordinate);
    let slot = hash & this.mask;
    for (let probe = 0; probe < this.capacity; probe += 1) {
      const entry = this.entries[slot];
      if (!entry) return undefined;
      if (entry.hash === hash && entry.coordinate.every((value, axis) =>
        value === coordinate[axis])) return entry.page;
      slot = (slot + 1) & this.mask;
    }
    return undefined;
  }

  get size(): number { return this.liveCount; }
}

const emptyPage = (coordinate: SolidWorldCoordinate, revision: number): SolidWorldPage => ({
  coordinate: [...coordinate],
  solidFraction: new Uint8Array(SOLID_WORLD_VOXELS_PER_PAGE),
  signedDistanceQ8: new Int16Array(SOLID_WORLD_VOXELS_PER_PAGE).fill(0x7fff),
  materialId: new Uint16Array(SOLID_WORLD_VOXELS_PER_PAGE),
  revision,
});

export interface SolidWorld {
  readonly pages: readonly SolidWorldPage[];
  readonly directory: SolidWorldDirectory;
  readonly patches: readonly SolidWorldVoxelPatch[];
  /** Compact ordered boxes evaluated beside sparse voxel pages. Large planar
   * colliders stay exact without expanding their area into page payload. */
  readonly regions?: readonly SolidWorldVoxelPatch[];
}

/**
 * Cheap canonical identity for every authored value that changes SolidWorld.
 *
 * Terrain samples are represented by their existing streaming content stamp;
 * voxel edits are folded in document order because patch composition is
 * ordered. Physical extents and realized lattice dimensions are included so a
 * scale change cannot reuse pages under a stale metre mapping.
 */
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
  for (const patch of scene.solidVoxels) {
    foldString(`${patch.operation}:${patch.minimum.join(":")}:`
      + `${patch.maximumExclusive.join(":")}:${patch.materialId ?? 1}`);
  }
  return `solid-world-v1:${high.toString(16).padStart(8, "0")}`
    + low.toString(16).padStart(8, "0");
}

function applySolidWorldPatches(
  base: SolidWorld,
  patches: readonly SolidWorldVoxelPatch[],
  cloneBase = true,
): SolidWorld {
  const pages = new Map<string, SolidWorldPage>();
  const key = (q: SolidWorldCoordinate) => q.join(",");
  for (const source of base.pages) pages.set(key(source.coordinate), cloneBase ? {
      coordinate: [...source.coordinate],
      solidFraction: source.solidFraction.slice(),
      signedDistanceQ8: source.signedDistanceQ8.slice(),
      materialId: source.materialId.slice(),
      revision: source.revision + 1,
    } : source);
  for (const patch of patches) {
    if (!patch.minimum.every(Number.isSafeInteger)
      || !patch.maximumExclusive.every(Number.isSafeInteger)
      || patch.minimum.some((value, axis) => value >= patch.maximumExclusive[axis]!)) {
      throw new RangeError("SolidWorld voxel patch must be a non-empty integer box");
    }
    const materialId = patch.materialId ?? 1;
    if (patch.operation === "fill"
      && (!Number.isSafeInteger(materialId) || materialId < 1 || materialId > 0xffff)) {
      throw new RangeError("SolidWorld material ID must fit non-zero u16");
    }
    for (let z = patch.minimum[2]; z < patch.maximumExclusive[2]; z += 1)
      for (let y = patch.minimum[1]; y < patch.maximumExclusive[1]; y += 1)
        for (let x = patch.minimum[0]; x < patch.maximumExclusive[0]; x += 1) {
          const address = solidWorldPageAddress([x, y, z]);
          let page = pages.get(key(address.page));
          if (!page && patch.operation === "clear") continue;
          if (!page) {
            page = emptyPage(address.page, 1);
            pages.set(key(address.page), page);
          }
          const fill = patch.operation === "fill";
          page.solidFraction[address.localIndex] = fill ? 255 : 0;
          page.signedDistanceQ8[address.localIndex] = fill ? -128 : 0x7fff;
          page.materialId[address.localIndex] = fill ? materialId : 0;
        }
  }
  const ordered = [...pages.values()].filter((page) =>
    page.solidFraction.some((fraction) => fraction > 0)).sort((left, right) =>
    left.coordinate[2] - right.coordinate[2]
    || left.coordinate[1] - right.coordinate[1]
    || left.coordinate[0] - right.coordinate[0]);
  const directory = new SolidWorldDirectory(ordered.length);
  ordered.forEach((page, index) => directory.insert(page.coordinate, index));
  return { pages: ordered, directory, patches: [...base.patches, ...patches] };
}

/** Compose authored voxel boxes over an already migrated uniform solid world. */
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
  const directory = new SolidWorldDirectory(0);
  return withSolidWorldPatches({ pages: [], directory, patches: [] }, patches);
}

function terrainSolidWorldForScene(scene: SceneDescription): SolidWorld {
  if (!scene.terrain) return createSolidWorld();
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  const cellHeight_m = scene.container.height_m / ny;
  const heights = terrainColumnHeights(scene, nx, nz);
  const pages = new Map<string, SolidWorldPage>();
  const key = (coordinate: SolidWorldCoordinate): string => coordinate.join(",");
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const height_m = heights[x + nx * z]!;
    const coveredCells = Math.min(ny, Math.ceil(height_m / cellHeight_m));
    for (let y = 0; y < coveredCells; y += 1) {
      const fraction = terrainCellSolidFraction(height_m, y * cellHeight_m,
        cellHeight_m);
      const quantizedFraction = Math.round(255 * fraction);
      if (quantizedFraction === 0) continue;
      const address = solidWorldPageAddress([x, y, z]);
      let page = pages.get(key(address.page));
      if (!page) {
        page = emptyPage(address.page, 1);
        pages.set(key(address.page), page);
      }
      page.solidFraction[address.localIndex] = quantizedFraction;
      const distanceQ8 = Math.round((((y + 0.5) * cellHeight_m - height_m)
        / cellHeight_m) * 256);
      page.signedDistanceQ8[address.localIndex] = Math.max(-0x8000,
        Math.min(0x7fff, distanceQ8));
      page.materialId[address.localIndex] = SOLID_WORLD_TERRAIN_MATERIAL_ID;
    }
  }
  const ordered = [...pages.values()].sort((left, right) =>
    left.coordinate[2] - right.coordinate[2]
    || left.coordinate[1] - right.coordinate[1]
    || left.coordinate[0] - right.coordinate[0]);
  const directory = new SolidWorldDirectory(ordered.length);
  ordered.forEach((page, index) => directory.insert(page.coordinate, index));
  return { pages: ordered, directory, patches: [] };
}

export function solidWorldForScene(scene: SceneDescription): SolidWorld {
  // The terrain bake is already page-native. Apply generic scene edits in
  // place before publishing the new world so construction never holds a second
  // terrain-sized page image or expands the columns into voxel-box descriptors.
  return applySolidWorldPatches(terrainSolidWorldForScene(scene),
    scene.solidVoxels, false);
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

/** SolidWorld used by fluid dynamics, including explicitly opted-in scenery. */
export function fluidSolidWorldForScene(scene: SceneDescription): SolidWorld {
  const world = solidWorldForScene(scene);
  const regions = fluidColliderVoxelPatchesForScene(scene);
  return regions.length === 0 ? world : { ...world,
    regions: [...(world.regions ?? []), ...regions] };
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

export function sampleSolidWorld(
  world: SolidWorld,
  coordinate: SolidWorldCoordinate,
): { readonly solidFraction: number; readonly signedDistance_cells: number;
  readonly materialId: number } {
  const address = solidWorldPageAddress(coordinate);
  const pageIndex = world.directory.lookup(address.page);
  const page = pageIndex === undefined ? undefined : world.pages[pageIndex]!;
  let solidFraction = page?.solidFraction[address.localIndex] ?? 0;
  let signedDistanceQ8 = page?.signedDistanceQ8[address.localIndex] ?? 0x7fff;
  let materialId = page?.materialId[address.localIndex] ?? 0;
  let regionMatched = false;
  for (const region of world.regions ?? []) {
    if (!coordinate.every((value, axis) => value >= region.minimum[axis]!
      && value < region.maximumExclusive[axis]!)) continue;
    regionMatched = true;
    const fill = region.operation === "fill";
    solidFraction = fill ? 255 : 0;
    signedDistanceQ8 = fill ? -128 : 0x7fff;
    materialId = fill ? (region.materialId ?? 1) : 0;
  }
  return { solidFraction: solidFraction / 255,
    signedDistance_cells: page === undefined && !regionMatched
      ? Number.POSITIVE_INFINITY : signedDistanceQ8 / 256, materialId };
}
