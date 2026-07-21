import { environmentIds, type EnvironmentId } from "./environments";
import type { Quaternion, SceneDescription } from "./model";
import {
  canonicalSvoThinGlassPane,
  packSvoThinGlassPanes,
  svoThinGlassBounds,
  type SvoThinGlassBounds,
  type SvoThinGlassPane,
} from "./svo-thin-glass";
import { SPARSE_BRICK_NO_OWNER } from "./sparse-brick-octree";
import {
  cachedSvoStaticPublication,
  hashSvoStaticPublication,
  internSvoStaticPublication,
} from "./svo-static-publication-cache";
import { GLASS_OPTICS } from "./webgpu-lighting";
import {
  buildEnvironmentProxyCatalog,
  type EnvironmentProxyCatalog,
  type EnvironmentProxyPrimitive,
} from "./voxel-environments";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import type { SvoVec3 } from "./webgpu-svo-traversal";

export const SVO_SCENE_GLASS_VERSION = "1" as const;
export const SVO_SCENE_GLASS_MAXIMUM_PANES = 256;

const sceneGlassCache = new Map<string, SvoSceneGlassBuild>();

export type SvoSceneGlassRole = "container-pane" | "container-top" | "environment-glazing";
export type SvoSceneContainerGlassSide = "floor" | "left" | "right" | "front" | "back" | "ceiling";

export interface SvoSceneGlassBuildOptions {
  environmentId?: EnvironmentId;
  /** Finest SVO cell size used for conservative candidate padding. */
  cellSize_m?: number | SvoVec3;
  maximumPanes?: number;
}

export interface SvoSceneGlassMetadata {
  recordIndex: number;
  key: string;
  role: SvoSceneGlassRole;
  side?: SvoSceneContainerGlassSide;
  paneId: number;
  materialId: number;
  ownerId: number;
  bounds: SvoThinGlassBounds;
  /** Opaque proxy that production must cut out before the pane can reveal the exterior. */
  opaqueCutoutKey?: string;
}

export interface SvoSceneGlassUnsupportedEntry {
  key: string;
  source: "catalog" | "procedural-shell";
  reason: "curved-emissive-volume" | "emissive-display" | "procedural-circular-glazing";
  fallback: "existing-opaque-primitive" | "existing-procedural-shell";
}

export interface SvoSceneGlassBuild {
  environmentId: EnvironmentId;
  descriptors: readonly Required<SvoThinGlassPane>[];
  packedRecords: Uint32Array<ArrayBuffer>;
  metadata: readonly SvoSceneGlassMetadata[];
  unsupportedEntries: readonly SvoSceneGlassUnsupportedEntry[];
  containerPolicy: "thin-glass-vessel" | "absent-open-environment";
  containerPaneIndices: readonly number[];
  containerTopPaneIndex?: number;
  environmentPaneIndices: readonly number[];
  /** Content hash over records, bounds policy, metadata, and diagnostics. */
  staticRevision: string;
  /** Versioned key suitable for renderer upload caches. */
  cacheKey: string;
}

const CONTAINER_PANE_ID_BASE = 0x1000;
const ENVIRONMENT_PANE_ID_BASE = 0x2000;
const DEFAULT_ABSORPTION_M_INV = [0.18, 0.04, 0.03] as const;
const SQRT_HALF = Math.SQRT1_2;

const Q_IDENTITY: Quaternion = { w: 1, x: 0, y: 0, z: 0 };
const Q_NORMAL_NEGATIVE_Z: Quaternion = { w: 0, x: 0, y: 1, z: 0 };
const Q_NORMAL_POSITIVE_X: Quaternion = { w: SQRT_HALF, x: 0, y: SQRT_HALF, z: 0 };
const Q_NORMAL_NEGATIVE_X: Quaternion = { w: SQRT_HALF, x: 0, y: -SQRT_HALF, z: 0 };
const Q_NORMAL_POSITIVE_Y: Quaternion = { w: SQRT_HALF, x: -SQRT_HALF, y: 0, z: 0 };
const Q_NORMAL_NEGATIVE_Y: Quaternion = { w: SQRT_HALF, x: SQRT_HALF, y: 0, z: 0 };

interface AuthoredSceneGlassPane {
  key: string;
  role: SvoSceneGlassRole;
  descriptor: SvoThinGlassPane;
  side?: SvoSceneContainerGlassSide;
  opaqueCutoutKey?: string;
}

function positiveInteger(value: number, maximum: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new RangeError(`${label} must be an integer from 1 to ${maximum}`);
  }
  return value;
}

function cellSize(scene: SceneDescription, input: number | SvoVec3 | undefined): SvoVec3 {
  const value: SvoVec3 = typeof input === "number"
    ? [input, input, input]
    : input ?? [scene.voxelDomain.finestCellSize_m, scene.voxelDomain.finestCellSize_m, scene.voxelDomain.finestCellSize_m];
  if (value.some((component) => !Number.isFinite(component) || component <= 0)) {
    throw new RangeError("SVO scene glass cell size must contain finite positive components");
  }
  return [...value];
}

function glassThickness(scene: SceneDescription): number {
  return Math.max(0.002, Math.min(0.012, scene.voxelDomain.finestCellSize_m * 0.25));
}

function pane(
  paneId: number,
  center_m: SvoVec3,
  halfExtent_m: readonly [number, number],
  thickness_m: number,
  orientation: Quaternion,
): SvoThinGlassPane {
  return {
    paneId,
    materialId: VOXEL_MATERIAL_IDS.containerGlass,
    ownerId: SPARSE_BRICK_NO_OWNER,
    center_m,
    halfExtent_m,
    thickness_m,
    orientation,
    indexOfRefraction: GLASS_OPTICS.indexOfRefraction,
    absorption_mInv: DEFAULT_ABSORPTION_M_INV,
    edgeEpsilon_m: Math.max(1e-6, thickness_m * 1e-3),
    maximumOpticalPath_m: thickness_m * 64,
  };
}

function containerPanes(scene: SceneDescription, environmentId: EnvironmentId, thickness_m: number): AuthoredSceneGlassPane[] {
  if (environmentId === "garden") return [];
  const halfWidth = 0.5 * scene.container.width_m;
  const halfDepth = 0.5 * scene.container.depth_m;
  const halfHeight = 0.5 * scene.container.height_m;
  const result: AuthoredSceneGlassPane[] = [
    { key: "container/floor", side: "floor", role: "container-pane", descriptor: pane(CONTAINER_PANE_ID_BASE, [0, 0, 0], [halfWidth, halfDepth], thickness_m, Q_NORMAL_NEGATIVE_Y) },
    { key: "container/left", side: "left", role: "container-pane", descriptor: pane(CONTAINER_PANE_ID_BASE + 1, [-halfWidth, halfHeight, 0], [halfDepth, halfHeight], thickness_m, Q_NORMAL_NEGATIVE_X) },
    { key: "container/right", side: "right", role: "container-pane", descriptor: pane(CONTAINER_PANE_ID_BASE + 2, [halfWidth, halfHeight, 0], [halfDepth, halfHeight], thickness_m, Q_NORMAL_POSITIVE_X) },
    { key: "container/front", side: "front", role: "container-pane", descriptor: pane(CONTAINER_PANE_ID_BASE + 3, [0, halfHeight, -halfDepth], [halfWidth, halfHeight], thickness_m, Q_NORMAL_NEGATIVE_Z) },
    { key: "container/back", side: "back", role: "container-pane", descriptor: pane(CONTAINER_PANE_ID_BASE + 4, [0, halfHeight, halfDepth], [halfWidth, halfHeight], thickness_m, Q_IDENTITY) },
  ];
  if (scene.container.top === "closed") result.push({
    key: "container/ceiling", side: "ceiling", role: "container-top",
    descriptor: pane(CONTAINER_PANE_ID_BASE + 5, [0, scene.container.height_m, 0], [halfWidth, halfDepth], thickness_m, Q_NORMAL_POSITIVE_Y),
  });
  return result;
}

function environmentPanes(catalog: EnvironmentProxyCatalog, thickness_m: number): AuthoredSceneGlassPane[] {
  const result: AuthoredSceneGlassPane[] = [];
  const environmentBase = ENVIRONMENT_PANE_ID_BASE + environmentIds.indexOf(catalog.environmentId) * 0x100;
  if (catalog.environmentId === "conservatory") {
    const s = catalog.scale_m;
    const xCenters = [-0.56 * s, 0.56 * s];
    const yBands = [
      { name: "low", center: 0.2975 * s, half: 0.2975 * s },
      { name: "middle", center: 0.94 * s, half: 0.295 * s },
      { name: "high", center: 1.5625 * s, half: 0.2775 * s },
    ];
    let localIndex = 0;
    for (let column = 0; column < xCenters.length; column += 1) for (const band of yBands) {
      result.push({
        key: `conservatory/glazing/pane-${column === 0 ? "left" : "right"}-${band.name}`,
        role: "environment-glazing",
        descriptor: pane(environmentBase + localIndex, [xCenters[column], band.center, -1.48 * s], [0.53 * s, band.half], thickness_m, Q_IDENTITY),
      });
      localIndex += 1;
    }
  } else if (catalog.environmentId === "night-lab") {
    const s = catalog.scale_m;
    result.push({
      key: "night-lab/window/city-glazing",
      role: "environment-glazing",
      descriptor: pane(environmentBase, [0, catalog.floorY_m + 1.60 * s, catalog.shell.bounds_m.min.z], [1.62 * s, 0.55 * s], thickness_m, Q_IDENTITY),
    });
  } else if (catalog.environmentId === "research-station") {
    const s = catalog.scale_m;
    result.push({
      key: "research-station/observation-port/glazing",
      role: "environment-glazing",
      descriptor: pane(environmentBase, [0, catalog.floorY_m + 1.55 * s, catalog.shell.bounds_m.min.z + .018 * s], [.66 * s, .39 * s], thickness_m, Q_IDENTITY),
    });
  }
  return result;
}

function unsupportedCatalogGlass(catalog: EnvironmentProxyCatalog): SvoSceneGlassUnsupportedEntry[] {
  const result: SvoSceneGlassUnsupportedEntry[] = [];
  for (const primitive of catalog.primitives) {
    if (primitive.group === "emissive-glass") result.push({
      key: primitive.key,
      source: "catalog",
      reason: "curved-emissive-volume",
      fallback: "existing-opaque-primitive",
    });
    else if (primitive.group === "monitor-glass") result.push({
      key: primitive.key,
      source: "catalog",
      reason: "emissive-display",
      fallback: "existing-opaque-primitive",
    });
  }
  if (catalog.environmentId === "research-station") result.push({
    key: "research-station/shell/procedural-portholes",
    source: "procedural-shell",
    reason: "procedural-circular-glazing",
    fallback: "existing-procedural-shell",
  });
  return result;
}

/** Build pane records from an already-authored deterministic environment catalog. */
export function svoSceneGlassFromEnvironmentCatalog(
  scene: SceneDescription,
  catalog: EnvironmentProxyCatalog,
  options: Omit<SvoSceneGlassBuildOptions, "environmentId"> = {},
): SvoSceneGlassBuild {
  const maximumPanes = positiveInteger(options.maximumPanes ?? SVO_SCENE_GLASS_MAXIMUM_PANES, SVO_SCENE_GLASS_MAXIMUM_PANES, "SVO scene glass pane limit");
  const cell = cellSize(scene, options.cellSize_m);
  const thickness_m = glassThickness(scene);
  const authored: AuthoredSceneGlassPane[] = [
    ...containerPanes(scene, catalog.environmentId, thickness_m),
    ...environmentPanes(catalog, thickness_m),
  ];
  if (authored.length > maximumPanes) {
    throw new RangeError(`Environment ${catalog.environmentId} needs ${authored.length} glass panes, exceeding the ${maximumPanes} record limit`);
  }
  const unsupportedEntries = unsupportedCatalogGlass(catalog);
  const staticRevision = hashSvoStaticPublication(new Uint32Array(), JSON.stringify({
    environmentId: catalog.environmentId,
    authored,
    unsupportedEntries,
    cell,
  }));
  const cacheKey = `svo-scene-glass-v${SVO_SCENE_GLASS_VERSION}:${catalog.environmentId}:${staticRevision}`;
  const cached = cachedSvoStaticPublication(sceneGlassCache, cacheKey);
  if (cached) return cached;
  const descriptors = authored.map(({ descriptor }) => canonicalSvoThinGlassPane(descriptor));
  const packedRecords = packSvoThinGlassPanes(descriptors);
  const metadata: SvoSceneGlassMetadata[] = authored.map((entry, recordIndex) => ({
    recordIndex,
    key: entry.key,
    role: entry.role,
    ...(entry.side ? { side: entry.side } : {}),
    paneId: descriptors[recordIndex].paneId,
    materialId: descriptors[recordIndex].materialId,
    ownerId: descriptors[recordIndex].ownerId,
    bounds: svoThinGlassBounds(descriptors[recordIndex], cell),
    ...(entry.opaqueCutoutKey ? { opaqueCutoutKey: entry.opaqueCutoutKey } : {}),
  }));
  const containerPaneIndices = metadata.filter(({ role }) => role === "container-pane" || role === "container-top").map(({ recordIndex }) => recordIndex);
  const environmentPaneIndices = metadata.filter(({ role }) => role === "environment-glazing").map(({ recordIndex }) => recordIndex);
  return internSvoStaticPublication(sceneGlassCache, cacheKey, {
    environmentId: catalog.environmentId,
    descriptors,
    packedRecords,
    metadata,
    unsupportedEntries,
    containerPolicy: catalog.environmentId === "garden" ? "absent-open-environment" : "thin-glass-vessel",
    containerPaneIndices,
    containerTopPaneIndex: metadata.find(({ role }) => role === "container-top")?.recordIndex,
    environmentPaneIndices,
    staticRevision,
    cacheKey,
  });
}

/** Build the selected scene environment catalog and its glass records in one call. */
export function buildSvoSceneGlass(
  scene: SceneDescription,
  options: SvoSceneGlassBuildOptions = {},
): SvoSceneGlassBuild {
  const environmentId = options.environmentId ?? scene.environment ?? "default";
  return svoSceneGlassFromEnvironmentCatalog(scene, buildEnvironmentProxyCatalog(scene, environmentId), options);
}

/** Narrow helper for coverage tooling that classifies proxy entries consistently. */
export function isUnsupportedThinGlassProxy(primitive: EnvironmentProxyPrimitive): boolean {
  return primitive.group === "emissive-glass" || primitive.group === "monitor-glass";
}
