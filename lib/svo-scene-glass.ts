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
  cachedSvoPublication,
  hashSvoPublication,
  internSvoPublication,
} from "./svo-publication-cache";
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
  /** Always a catalog primitive: every optical gap is a described object. */
  source: "catalog";
  reason: "curved-emissive-volume" | "emissive-display" | "procedural-circular-glazing";
  fallback: "existing-opaque-primitive";
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
  contentRevision: string;
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

/**
 * Whether this document's container is a visible vessel.
 *
 * The garden has always been the open case: its water sits in the ground, and a
 * pane hanging in the air around it would read as a bug. `container.vessel`
 * generalises that from an environment the renderer happens to know the name of
 * to a property of the scene, which is what lets a fresh room start without a
 * tank in it and gain one when its author asks for one.
 */
function containerIsGlass(scene: SceneDescription, environmentId: EnvironmentId): boolean {
  return scene.container.vessel !== "none" && environmentId !== "garden";
}

function containerPanes(scene: SceneDescription, environmentId: EnvironmentId, thickness_m: number): AuthoredSceneGlassPane[] {
  if (!containerIsGlass(scene, environmentId)) return [];
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

/**
 * The panes an environment declares.
 *
 * These used to be a per-environment branch of hand-placed constants that had to
 * agree with a wall authored in a different file — a window and its glass held
 * apart, so resizing the room moved one and not the other. Glazing is now a
 * scenery node like any other and arrives already positioned; all this does is
 * assign pane ids, which stay stable because expansion order is document order.
 */
function environmentPanes(catalog: EnvironmentProxyCatalog, thickness_m: number): AuthoredSceneGlassPane[] {
  const environmentBase = ENVIRONMENT_PANE_ID_BASE + environmentIds.indexOf(catalog.environmentId) * 0x100;
  return catalog.panes.map((declared, index) => ({
    key: declared.id,
    role: "environment-glazing" as const,
    descriptor: pane(environmentBase + index,
      [declared.center_m.x, declared.center_m.y, declared.center_m.z],
      [declared.half_m[0], declared.half_m[1]], thickness_m, declared.orientation ?? Q_IDENTITY),
  }));
}

/**
 * Glass-like surfaces that are not finite dielectric panes, named rather than
 * quietly dropped.
 *
 * Keyed off the group a scenery node declares, so an environment that grows a
 * new curved lamp or round port publishes the exception by describing itself —
 * this used to carry a branch per environment id, which meant a set could gain
 * one and nobody would notice it had gone unaccounted.
 */
const UNSUPPORTED_GLASS_GROUPS: Readonly<Record<string, SvoSceneGlassUnsupportedEntry["reason"]>> =
  Object.freeze({
    "emissive-glass": "curved-emissive-volume",
    "monitor-glass": "emissive-display",
    "porthole-glass": "procedural-circular-glazing",
  });

function unsupportedCatalogGlass(catalog: EnvironmentProxyCatalog): SvoSceneGlassUnsupportedEntry[] {
  const result: SvoSceneGlassUnsupportedEntry[] = [];
  for (const primitive of catalog.primitives) {
    const reason = UNSUPPORTED_GLASS_GROUPS[primitive.group];
    if (reason) result.push({
      key: primitive.key,
      source: "catalog",
      reason,
      fallback: "existing-opaque-primitive",
    });
  }
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
  const contentRevision = hashSvoPublication(new Uint32Array(), JSON.stringify({
    environmentId: catalog.environmentId,
    authored,
    unsupportedEntries,
    cell,
  }));
  const cacheKey = `svo-scene-glass-v${SVO_SCENE_GLASS_VERSION}:${catalog.environmentId}:${contentRevision}`;
  const cached = cachedSvoPublication(sceneGlassCache, cacheKey);
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
  return internSvoPublication(sceneGlassCache, cacheKey, {
    environmentId: catalog.environmentId,
    descriptors,
    packedRecords,
    metadata,
    unsupportedEntries,
    containerPolicy: containerIsGlass(scene, catalog.environmentId) ? "thin-glass-vessel" : "absent-open-environment",
    containerPaneIndices,
    containerTopPaneIndex: metadata.find(({ role }) => role === "container-top")?.recordIndex,
    environmentPaneIndices,
    contentRevision,
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
