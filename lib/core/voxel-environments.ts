import type { EnvironmentId } from "./environments";
import { environmentIndex } from "./environments";
import type { SceneDescription, Vec3 } from "./model";
import { expandSceneryGraph, type SceneryPane, type ScenerySpan } from "./scenery-expand";
import { sceneryGraphForEnvironment } from "./scenery-presets";
import {
  aabb,
  ProxyBuilder,
  V,
  type EnvironmentProxyPrimitive,
  type EnvironmentProxyShell,
  type EnvironmentSceneryContext,
} from "./voxel-scenery/index";

export type {
  EnvironmentBoxProxy,
  EnvironmentCylinderProxy,
  EnvironmentEllipsoidProxy,
  EnvironmentLinearColor,
  EnvironmentProxyAabb,
  EnvironmentProxyMaterial,
  EnvironmentProxyPrimitive,
  EnvironmentProxyShell,
  EnvironmentProxySway,
  EnvironmentSceneryContext,
} from "./voxel-scenery/index";

export interface EnvironmentProxyCatalog {
  readonly environmentId: EnvironmentId;
  readonly environmentIndex: number;
  readonly scale_m: number;
  readonly floorY_m: number;
  readonly shell: EnvironmentProxyShell;
  /** Authored props only; use environmentProxyPrimitives() when shell faces are also wanted. */
  readonly primitives: readonly EnvironmentProxyPrimitive[];
  /**
   * Which described object each primitive came from, indexed into `primitives`.
   * Selection and the hover outline both resolve a click to one of these.
   */
  readonly spans: readonly ScenerySpan[];
  /** Declared dielectric panes, keyed like primitives: `<environment>/<node id>`. */
  readonly panes: readonly SceneryPane[];
}

export interface EnvironmentProxyCatalogOptions {
  /** Physical thickness of the finite room shell faces. Defaults to the scene nominal resolution. */
  readonly shellThickness_m?: number;
  /**
   * Override the finest voxel the set is drawn at, in metres.
   *
   * The document's `voxelDomain.detailCellSize_m` is the answer for every
   * ordinary build and this exists for the lane that knows better than the
   * document does — a bench that renders one scene at four resolutions without
   * rebuilding it. Prefer the document: a catalog built at one detail size and a
   * heightfield baked at another disagree about the ground, and only the
   * document can carry both to the render worker.
   */
  readonly detailCellSize_m?: number;
}

/** The world frame a scene's scenery is placed in. */
export function environmentSceneryContext(
  scene: SceneDescription,
  environmentId: EnvironmentId,
  options: EnvironmentProxyCatalogOptions = {},
): EnvironmentSceneryContext {
  const s = Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
  const thickness = options.shellThickness_m ?? scene.voxelDomain.finestCellSize_m;
  if (!(thickness > 0) || !Number.isFinite(thickness)) throw new Error("Environment shell thickness must be positive and finite");
  // Absent means the two lattices are the same, which is every scene that has
  // not asked for extra octree levels under the solver's own.
  const detailCellSize_m = options.detailCellSize_m
    ?? scene.voxelDomain.detailCellSize_m
    ?? scene.voxelDomain.finestCellSize_m;
  if (!(detailCellSize_m > 0) || !Number.isFinite(detailCellSize_m)) {
    throw new Error("Environment detail cell size must be positive and finite");
  }
  return {
    scene, s,
    floorY_m: environmentId === "night-lab" ? -.72 * s
      : environmentId === "garden" ? (scene.terrain?.baseHeight_m ?? 0) : -.025,
    roomHalf_m: V(
      Math.max(scene.container.width_m * 2.8, s * 2.25),
      Math.max(scene.container.height_m * 1.85, s * 1.8),
      Math.max(scene.container.depth_m * 2.8, s * 2.25),
    ),
    shellThickness_m: thickness,
    detailCellSize_m,
  };
}

/**
 * Expand a scene's scenery into the catalog every downstream consumer reads.
 *
 * The geometry is the scene's own `scenery` graph — see lib/scenery-graph.ts —
 * and this only resolves the world frame it is placed in. A scene that has not
 * been given one yet falls back to its environment's seed graph, which is the
 * same value `sceneWithEnvironment` would have written; the fallback exists so a
 * hand-built scene fixture still renders, not as a second authority.
 */
export function buildEnvironmentProxyCatalog(scene: SceneDescription, environmentId: EnvironmentId, options: EnvironmentProxyCatalogOptions = {}): EnvironmentProxyCatalog {
  const cached = cachedCatalog(scene, environmentId, options);
  if (cached) return cached;
  const context = environmentSceneryContext(scene, environmentId, options);
  const graph = scene.scenery ?? sceneryGraphForEnvironment(scene, environmentId);
  const b = new ProxyBuilder(environmentId);
  const { shell, spans, panes } = expandSceneryGraph(b, graph, context);
  return retainCatalog(scene, environmentId, options, {
    environmentId, environmentIndex: environmentIndex(environmentId),
    scale_m: context.s, floorY_m: context.floorY_m, shell, primitives: b.props, spans,
    panes: panes.map((p) => ({ ...p, id: `${environmentId}/${p.id}` })),
  });
}

/**
 * One expansion per document.
 *
 * Expansion is not free — a garden grows its trees bead by bead — and the same
 * catalog is asked for several times over: the cursor's pick, the rim's owner
 * range, the gizmo around the selected object, and every publication the
 * renderer assembles when it attaches a scene. Uncached, a pointer-move paid for
 * three full expansions of a set that had not changed between them.
 *
 * Scene documents are replaced rather than edited, so identity is a sound key.
 * The fingerprint beside it covers the one thing identity misses — a fixture
 * that resizes a container in place — because a stale catalog would put the
 * cursor's idea of the room somewhere the render does not agree with.
 */
const catalogCache = new WeakMap<SceneDescription, Map<string, EnvironmentProxyCatalog>>();

function catalogFingerprint(scene: SceneDescription, environmentId: EnvironmentId, options: EnvironmentProxyCatalogOptions): string {
  const c = scene.container;
  // The detail cell belongs here for the same reason the finest cell does: it is
  // an *input to expansion*, not just a size the tree happens to have. A
  // generator sizes its features in detail voxels — `bonsaiCanopyLadder` raises
  // its floret to a legibility floor of leaves — so two catalogs built from one
  // document at two detail cells are two different sets. It was missing, which
  // made the option's own documented use ("a bench that renders one scene at
  // four resolutions without rebuilding it") return the first resolution four
  // times.
  return `${environmentId}|${options.shellThickness_m ?? "scene"}|${c.width_m}:${c.height_m}:${c.depth_m}`
    + `|${scene.voxelDomain.finestCellSize_m}|${options.detailCellSize_m ?? scene.voxelDomain.detailCellSize_m ?? "cell"}`
    + `|${JSON.stringify(scene.terrain ?? null)}`;
}

function cachedCatalog(scene: SceneDescription, environmentId: EnvironmentId, options: EnvironmentProxyCatalogOptions): EnvironmentProxyCatalog | undefined {
  return catalogCache.get(scene)?.get(catalogFingerprint(scene, environmentId, options));
}

function retainCatalog(scene: SceneDescription, environmentId: EnvironmentId, options: EnvironmentProxyCatalogOptions, catalog: EnvironmentProxyCatalog): EnvironmentProxyCatalog {
  let entries = catalogCache.get(scene);
  if (!entries) { entries = new Map(); catalogCache.set(scene, entries); }
  entries.set(catalogFingerprint(scene, environmentId, options), catalog);
  return catalog;
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
