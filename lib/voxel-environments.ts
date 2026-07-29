import type { EnvironmentId } from "./environments";
import { environmentIndex } from "./environments";
import type { SceneDescription, Vec3 } from "./model";
import {
  aabb,
  C,
  environmentScenery,
  ProxyBuilder,
  V,
  type EnvironmentProxyPrimitive,
  type EnvironmentProxyShell,
  type EnvironmentSceneryContext,
} from "./voxel-scenery";

export type {
  EnvironmentBoxProxy,
  EnvironmentCylinderProxy,
  EnvironmentEllipsoidProxy,
  EnvironmentLinearColor,
  EnvironmentProxyAabb,
  EnvironmentProxyMaterial,
  EnvironmentProxyPrimitive,
  EnvironmentProxyShell,
  EnvironmentSceneryContext,
  EnvironmentSceneryModule,
} from "./voxel-scenery";
export { ENVIRONMENT_SCENERY, environmentScenery } from "./voxel-scenery";

export interface EnvironmentProxyCatalog {
  readonly environmentId: EnvironmentId;
  readonly environmentIndex: number;
  readonly scale_m: number;
  readonly floorY_m: number;
  readonly shell: EnvironmentProxyShell;
  /** Authored props only; use environmentProxyPrimitives() when shell faces are also wanted. */
  readonly primitives: readonly EnvironmentProxyPrimitive[];
}

export interface EnvironmentProxyCatalogOptions {
  /** Physical thickness of the finite room shell faces. Defaults to the scene nominal resolution. */
  readonly shellThickness_m?: number;
}

/**
 * Convert authored `scene.props` into proxies on the shared builder, so they
 * receive owner indices from the same sequence as procedural scenery and are
 * indistinguishable to every downstream consumer.
 */
function addScenePropProxies(builder: ProxyBuilder, scene: SceneDescription): void {
  for (const prop of scene.props ?? []) {
    const color = C(prop.colorLinear[0], prop.colorLinear[1], prop.colorLinear[2]);
    const key = `prop/${prop.id}`;
    const emission = prop.emission ?? 0;
    if (prop.shape === "box") {
      builder.box(key, "prop", prop.position_m, prop.halfSize_m, color, emission, ["prop", "authored"]);
    } else if (prop.shape === "cylinder") {
      builder.cylinder(key, "prop", prop.position_m, prop.halfSize_m.x, prop.halfSize_m.y, color, emission, ["prop", "authored"]);
    } else {
      builder.ellipsoid(key, "prop", prop.position_m, prop.halfSize_m, color, emission, ["prop", "authored"]);
    }
  }
}

/**
 * Assembles one environment's scenery module into the catalog every downstream
 * consumer reads. The geometry itself lives in lib/voxel-scenery/<id>.ts; this
 * function only resolves the world frame it is placed in and appends the
 * scene's own authored props.
 */
export function buildEnvironmentProxyCatalog(scene: SceneDescription, environmentId: EnvironmentId, options: EnvironmentProxyCatalogOptions = {}): EnvironmentProxyCatalog {
  const s = Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
  const roomHalf = V(Math.max(scene.container.width_m * 2.8, s * 2.25), Math.max(scene.container.height_m * 1.85, s * 1.8), Math.max(scene.container.depth_m * 2.8, s * 2.25));
  const floorY = environmentId === "night-lab" ? -.72 * s : environmentId === "garden" ? (scene.terrain?.baseHeight_m ?? 0) : -.025;
  const thickness = options.shellThickness_m ?? scene.voxelDomain.finestCellSize_m;
  if (!(thickness > 0) || !Number.isFinite(thickness)) throw new Error("Environment shell thickness must be positive and finite");
  const b = new ProxyBuilder(environmentId);
  const context: EnvironmentSceneryContext = {
    scene, s, floorY_m: floorY, roomHalf_m: roomHalf, shellThickness_m: thickness,
  };
  const shell = environmentScenery(environmentId).build(b, context);

  // Authored props append to the procedural catalog rather than replacing it,
  // so presets keep their hand-built scenery while the editor can add to it.
  // They enter every downstream consumer (SVO records, static mips, coverage,
  // sparse domain) through this one list, and never enter the solve.
  addScenePropProxies(b, scene);

  return { environmentId, environmentIndex: environmentIndex(environmentId), scale_m: s, floorY_m: floorY, shell, primitives: b.props };
}

export function environmentProxyPrimitives(catalog: EnvironmentProxyCatalog, includeShell = true): readonly EnvironmentProxyPrimitive[] {
  return includeShell ? [...catalog.shell.primitives, ...catalog.primitives] : catalog.primitives;
}

export interface EnvironmentProxyMaterialEntry {
  /** Matches the primitive ownerIndex and is therefore directly GPU-indexable. */
  readonly index: number;
  readonly key: string;
  readonly material: EnvironmentProxyPrimitive["material"];
}

/** Stable owner-aligned material table; repeated authored materials intentionally retain separate named slots. */
export function environmentProxyMaterialTable(catalog: EnvironmentProxyCatalog, includeShell = true): readonly EnvironmentProxyMaterialEntry[] {
  return environmentProxyPrimitives(catalog, includeShell).map((primitive) => ({
    index: primitive.ownerIndex, key: primitive.key, material: primitive.material
  }));
}

export interface SparseBrickCoordinateOptions {
  readonly cellSize_m: number | Vec3;
  readonly worldOrigin_m: Vec3;
  readonly brickSize_cells: number;
}

export interface Integer3 { readonly x: number; readonly y: number; readonly z: number }
export interface VoxelCellRange { readonly minInclusive: Integer3; readonly maxInclusive: Integer3 }

function positiveCellSize(value: number | Vec3): Vec3 {
  const h = typeof value === "number" ? V(value, value, value) : value;
  if (!(h.x > 0) || !(h.y > 0) || !(h.z > 0) || !Number.isFinite(h.x + h.y + h.z)) throw new Error("Voxel cell size must be positive and finite");
  return h;
}

/** Conservative cell bounds; a face exactly on a cell boundary retains both touching cells. */
export function voxelCellRangeForAabb(bounds_m: EnvironmentProxyPrimitive["aabb_m"], cellSize_m: number | Vec3, worldOrigin_m: Vec3): VoxelCellRange {
  const h = positiveCellSize(cellSize_m);
  return {
    minInclusive: {
      x: Math.floor((bounds_m.min.x - worldOrigin_m.x) / h.x),
      y: Math.floor((bounds_m.min.y - worldOrigin_m.y) / h.y),
      z: Math.floor((bounds_m.min.z - worldOrigin_m.z) / h.z)
    },
    maxInclusive: {
      x: Math.floor((bounds_m.max.x - worldOrigin_m.x) / h.x),
      y: Math.floor((bounds_m.max.y - worldOrigin_m.y) / h.y),
      z: Math.floor((bounds_m.max.z - worldOrigin_m.z) / h.z)
    }
  };
}

/** Unique lexicographically sorted sparse-brick coordinates covering the supplied AABBs. */
export function sparseBrickCoordinatesForAabbs(bounds: readonly EnvironmentProxyPrimitive["aabb_m"][], options: SparseBrickCoordinateOptions): readonly Integer3[] {
  if (!Number.isInteger(options.brickSize_cells) || options.brickSize_cells <= 0) throw new Error("Sparse brick size must be a positive integer");
  const keys = new Map<string, Integer3>();
  for (const aabb_m of bounds) {
    const range = voxelCellRangeForAabb(aabb_m, options.cellSize_m, options.worldOrigin_m);
    const min = V(Math.floor(range.minInclusive.x / options.brickSize_cells), Math.floor(range.minInclusive.y / options.brickSize_cells), Math.floor(range.minInclusive.z / options.brickSize_cells));
    const max = V(Math.floor(range.maxInclusive.x / options.brickSize_cells), Math.floor(range.maxInclusive.y / options.brickSize_cells), Math.floor(range.maxInclusive.z / options.brickSize_cells));
    for (let z = min.z; z <= max.z; z++) for (let y = min.y; y <= max.y; y++) for (let x = min.x; x <= max.x; x++) keys.set(`${x},${y},${z}`, { x, y, z });
  }
  return [...keys.values()].sort((a, b) => a.z - b.z || a.y - b.y || a.x - b.x);
}

export function sparseBrickCoordinatesForEnvironment(catalog: EnvironmentProxyCatalog, options: SparseBrickCoordinateOptions, includeShell = false): readonly Integer3[] {
  return sparseBrickCoordinatesForAabbs(environmentProxyPrimitives(catalog, includeShell).map((primitive) => primitive.aabb_m), options);
}

export { aabb };
