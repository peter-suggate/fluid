import { terrainContentStamp, terrainWorldContainer } from "./terrain";
import type { EnvironmentId } from "./environments";
import { environmentIndex } from "./environments";
import type { SceneDescription, Vec3 } from "./model";
import { classifySceneryChange } from "./scenery-construction-key";
import { expandSceneryGraph, type ScenerySpan } from "./scenery-expand";
import type { SceneryGraph, SceneryPalette } from "./scenery-graph";
import { sceneryGraphForEnvironment } from "./scenery-presets";
import { meshAssetHeader } from "./imported-assets/asset-library";
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

export const ENVIRONMENT_MATERIAL_BASE = 32;

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
  const extent = terrainWorldContainer(scene);
  const s = Math.max(extent.width_m, extent.height_m, extent.depth_m);
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
      Math.max(extent.width_m * 2.8, s * 2.25),
      Math.max(extent.height_m * 1.85, s * 1.8),
      Math.max(extent.depth_m * 2.8, s * 2.25),
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
  const names = Object.keys(graph.palettes);
  // One expansion says both what the set looks like and which palette entry
  // each colour is: it runs under the probe tints, and the document's own are
  // multiplied in afterwards.
  let b = new ProxyBuilder(environmentId);
  let expansion = expandSceneryGraph(b, names.length > 0 ? { ...graph, palettes: environmentPaletteProbes(names) } : graph, context);
  const resolved = names.length > 0 ? resolvePaletteProbes(expansion.shell.primitives, b.props, graph, names) : undefined;
  let shell = expansion.shell, primitives: readonly EnvironmentProxyPrimitive[] = b.props;
  if (resolved) {
    shell = { ...shell, primitives: resolved.shell };
    primitives = resolved.primitives;
  } else if (names.length > 0) {
    // A colour no palette entry describes: the set is expanded as it is drawn.
    b = new ProxyBuilder(environmentId);
    expansion = expandSceneryGraph(b, graph, context);
    shell = expansion.shell; primitives = b.props;
  }
  // Source palettes follow ordinary owner materials; instances of the same
  // immutable asset share a palette. Owner IDs remain the editor's original IDs.
  let materialBase = ENVIRONMENT_MATERIAL_BASE + Math.max(-1, ...shell.primitives.map(p => p.ownerIndex), ...primitives.map(p => p.ownerIndex)) + 1;
  const assetBases = new Map<string, number>();
  primitives = primitives.map(p => {
    if (p.kind !== "field-program" || !p.program.sampledVolume) return p;
    const asset = p.program.sampledVolume.asset;
    let base = assetBases.get(asset);
    if (base === undefined) { base = materialBase; assetBases.set(asset, base); materialBase += meshAssetHeader(asset).materials.length; }
    if (materialBase > 8192) throw new RangeError("Imported asset palettes exceed the scene material capacity");
    return { ...p, program: { ...p.program, sampledVolume: { asset, materialBase: base } } };
  });
  const catalog = retainCatalog(scene, environmentId, options, {
    environmentId, environmentIndex: environmentIndex(environmentId),
    scale_m: context.s, floorY_m: context.floorY_m, shell, primitives, spans: expansion.spans,
  });
  if (resolved) catalogPaletteSources.set(catalog, resolved.sources);
  return catalog;
}

/**
 * A catalog's colours, as the palette entries they were resolved from.
 *
 * Indexed by publication position, shell faces first. `palette` is an index
 * into `names`, or -1 for a colour the document wrote out; `value` is the
 * position on that palette's ramp. A tint edit recolours the positions naming
 * the palette and reads nothing else.
 */
export interface EnvironmentCatalogPaletteSources {
  readonly names: string[];
  readonly palette: Int16Array;
  readonly value: Float64Array;
}

const catalogPaletteSources = new WeakMap<EnvironmentProxyCatalog, EnvironmentCatalogPaletteSources>();

/** The palette entries a catalog's colours came from. Absent when one of them is nothing a palette entry describes. */
export function environmentCatalogPaletteSources(catalog: EnvironmentProxyCatalog): EnvironmentCatalogPaletteSources | undefined {
  return catalogPaletteSources.get(catalog);
}

/** Say where the colours of a catalog derived from another by recolouring came from. */
export function holdEnvironmentCatalogPaletteSources(catalog: EnvironmentProxyCatalog, sources: EnvironmentCatalogPaletteSources): void {
  catalogPaletteSources.set(catalog, sources);
}

/**
 * Tints no authored colour can be mistaken for, in exact powers of two.
 *
 * Expanding a node under these leaves every palette colour as
 * `[v·2^40, v·2^(60+k), v·2^-40]`: `v` and `k` come back exactly, and a
 * literal passes through untouched. A colour that is neither — one the
 * expansion clamped or mixed after resolving it — is caught by its magnitude.
 */
const PROBE_RED = 2 ** 40, PROBE_BLUE = 2 ** -40;
const authoredMagnitude = (value: number): boolean => value === 0 || (Math.abs(value) > 2 ** -30 && Math.abs(value) < 2 ** 30);

/** The probe tint of each named palette, in the order that numbers them. */
export function environmentPaletteProbes(names: readonly string[]): Record<string, SceneryPalette> {
  const probes: Record<string, SceneryPalette> = {};
  names.forEach((name, index) => { probes[name] = { tint: [PROBE_RED, 2 ** (60 + index), PROBE_BLUE] }; });
  return probes;
}

/** The palette index a probe colour names, -1 for a literal, undefined when it is neither. */
export function environmentPaletteProbeSource(color: readonly [number, number, number], paletteCount: number): number | undefined {
  if (authoredMagnitude(color[0]) && authoredMagnitude(color[1]) && authoredMagnitude(color[2])) return -1;
  const index = Math.log2(color[1] / color[0]) - 20;
  return Number.isInteger(index) && index >= 0 && index < paletteCount
    && color[1] === color[0] * 2 ** (20 + index) && color[2] === color[0] * 2 ** -80 ? index : undefined;
}

/** Where on its palette's ramp a probe colour sits. */
export function environmentPaletteProbeValue(color: readonly [number, number, number]): number {
  return color[0] / PROBE_RED;
}

/**
 * The primitives of a probe expansion under the graph's own tints, and the
 * palette entry each colour was. Undefined when a colour is neither a literal
 * nor a palette entry.
 */
function resolvePaletteProbes(
  shellProbes: readonly EnvironmentProxyShell["primitives"][number][],
  propProbes: readonly EnvironmentProxyPrimitive[],
  graph: SceneryGraph,
  names: string[],
): { shell: EnvironmentProxyShell["primitives"]; primitives: readonly EnvironmentProxyPrimitive[]; sources: EnvironmentCatalogPaletteSources } | undefined {
  const shellCount = shellProbes.length;
  const palette = new Int16Array(shellCount + propProbes.length);
  const value = new Float64Array(palette.length);
  const resolve = <T extends EnvironmentProxyPrimitive>(probes: readonly T[], first: number): T[] | undefined => {
    const out = new Array<T>(probes.length);
    for (let index = 0; index < probes.length; index += 1) {
      const probe = probes[index]!;
      const source = environmentPaletteProbeSource(probe.material.colorLinear, names.length);
      if (source === undefined) return undefined;
      palette[first + index] = source;
      if (source < 0) { out[index] = probe; continue; }
      const v = environmentPaletteProbeValue(probe.material.colorLinear), tint = graph.palettes[names[source]!]!.tint;
      value[first + index] = v;
      out[index] = { ...probe, material: { ...probe.material, colorLinear: [v * tint[0], v * tint[1], v * tint[2]] } };
    }
    return out;
  };
  const shell = resolve(shellProbes, 0), primitives = shell && resolve(propProbes, shellCount);
  return shell && primitives ? { shell, primitives, sources: { names, palette, value } } : undefined;
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
const sceneryCatalogCache = new WeakMap<object, Map<string, EnvironmentProxyCatalog>>();
const remoteCatalogScenes = new WeakSet<SceneDescription>();
let remoteCatalogConsumer = false;
/** The worker client enables this only in the UI realm, before any new document renders. */
export function useRemoteEnvironmentCatalogs(): void { remoteCatalogConsumer = true; }
/** Worker-owned documents must never expand procedural scenery in an input event. */
export function markEnvironmentCatalogRemote(scene: SceneDescription): void { remoteCatalogScenes.add(scene); }
export function environmentCatalogPending(scene: SceneDescription): boolean {
  return (remoteCatalogConsumer || remoteCatalogScenes.has(scene)) && !cachedEnvironmentProxyCatalog(scene);
}
export function cachedEnvironmentProxyCatalog(scene: SceneDescription): EnvironmentProxyCatalog | undefined {
  return cachedCatalog(scene, scene.environment ?? "default", {});
}
export function adoptEnvironmentProxyCatalog(scene: SceneDescription, catalog: EnvironmentProxyCatalog): void {
  retainCatalog(scene, catalog.environmentId, {}, catalog);
}
export function reuseEnvironmentProxyCatalog(source: SceneDescription, target: SceneDescription): void {
  const catalog = cachedEnvironmentProxyCatalog(source);
  if (!catalog || source.environment !== target.environment
    || catalogFingerprint(source, catalog.environmentId, {}) !== catalogFingerprint(target, catalog.environmentId, {})
    || classifySceneryChange(source.scenery, target.scenery).kind !== "same") return;
  adoptEnvironmentProxyCatalog(target, catalog);
}



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
  //
  // The ground is named by its content stamp, which is kept per terrain object.
  // It was the terrain serialized whole: every sample of a sculpted grid, built
  // and hashed again on each lookup, which an accepted voxel edit makes several
  // of while leaving the ground alone.
  return `${environmentId}|${options.shellThickness_m ?? "scene"}|${c.width_m}:${c.height_m}:${c.depth_m}`
    + `|${scene.voxelDomain.finestCellSize_m}|${options.detailCellSize_m ?? scene.voxelDomain.detailCellSize_m ?? "cell"}`
    + `|${terrainContentStamp(scene.terrain)}`;
}

function cachedCatalog(scene: SceneDescription, environmentId: EnvironmentId, options: EnvironmentProxyCatalogOptions): EnvironmentProxyCatalog | undefined {
  const key = catalogFingerprint(scene, environmentId, options);
  return catalogCache.get(scene)?.get(key) ?? (scene.scenery ? sceneryCatalogCache.get(scene.scenery)?.get(key) : undefined);
}

function retainCatalog(scene: SceneDescription, environmentId: EnvironmentId, options: EnvironmentProxyCatalogOptions, catalog: EnvironmentProxyCatalog): EnvironmentProxyCatalog {
  let entries = catalogCache.get(scene);
  if (!entries) { entries = new Map(); catalogCache.set(scene, entries); }
  const key = catalogFingerprint(scene, environmentId, options);
  entries.set(key, catalog);
  if (scene.scenery) {
    let graphEntries = sceneryCatalogCache.get(scene.scenery);
    if (!graphEntries) { graphEntries = new Map(); sceneryCatalogCache.set(scene.scenery, graphEntries); }
    graphEntries.set(key, catalog);
  }
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
