import type { EnvironmentId } from "./environments";
import type { SceneDescription } from "./model";
import {
  packSvoPrimitiveRecords,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveDescriptor,
} from "./svo-primitive-abi";
import {
  buildSvoPrimitiveCandidates,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
  type SvoPrimitiveCandidatePublication,
} from "./svo-primitive-candidates";
import {
  cachedSvoStaticPublication,
  hashSvoStaticPublication,
  internSvoStaticPublication,
} from "./svo-static-publication-cache";
import {
  buildEnvironmentProxyCatalog,
  environmentProxyPrimitives,
  type EnvironmentProxyCatalog,
  type EnvironmentProxyMaterial,
  type EnvironmentProxyPrimitive,
} from "./voxel-environments";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE } from "./webgpu-octree-sparse-bricks";

/** Defensive ceiling: current authored catalogs contain fewer than 64 entries. */
export const SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES = 4_096;
export const SVO_SCENE_PRIMITIVE_VERSION = "1" as const;

const scenePrimitiveCache = new Map<string, SvoScenePrimitiveBuild>();

export interface SvoScenePrimitiveBuildOptions {
  environmentId?: EnvironmentId;
  includeShell?: boolean;
  maximumPrimitives?: number;
  /** Nominal coverage width used only for conservative subcell audit bounds. */
  coverageCellSize_m?: number;
}

export interface SvoPrimitiveCoverageBounds {
  exact_m: EnvironmentProxyPrimitive["aabb_m"];
  conservative_m: EnvironmentProxyPrimitive["aabb_m"];
  subcellAxes: readonly ("x" | "y" | "z")[];
  policy: "exact" | "conservative-subcell";
}

export interface SvoEnvironmentPrimitiveMetadata {
  /** Dense index into descriptors and packedRecords. */
  primitiveIndex: number;
  /** Original dense owner index from EnvironmentProxyCatalog. */
  environmentOwnerIndex: number;
  /** Sparse voxel convention: rigid body count plus environment owner index. */
  ownerId: number;
  /** Sparse voxel convention: ENVIRONMENT_VOXEL_MATERIAL_BASE plus owner index. */
  materialId: number;
  key: string;
  group: string;
  tags: readonly string[];
  sourceKind: EnvironmentProxyPrimitive["kind"];
  material: EnvironmentProxyMaterial;
  shell: boolean;
  /** Front room shell is retained in the model but skipped for interior presentation. */
  openShell: boolean;
  /** Audit-only bounds; they do not alter rigid-body or solver collision physics. */
  coverageBounds: SvoPrimitiveCoverageBounds;
}

export interface SvoUnsupportedStaticSource {
  kind: "terrain-heightfield";
  fallback: "raster-terrain";
  materialId: number;
  reason: string;
}

export interface SvoAnalyticTerrainSource {
  kind: "terrain-heightfield";
  materialId: number;
  /** Matches the central-difference normal policy in lib/terrain.ts. */
  normalEpsilon_m: number;
}

export interface SvoScenePrimitiveBuild {
  environmentId: EnvironmentId;
  descriptors: readonly SvoPrimitiveDescriptor[];
  packedRecords: Uint32Array<ArrayBuffer>;
  /** Bounded static BVH sharing the primitive-record stride and GPU binding. */
  primitiveCandidates?: SvoPrimitiveCandidatePublication;
  metadata: readonly SvoEnvironmentPrimitiveMetadata[];
  primitiveIndexByOwnerId: ReadonlyMap<number, number>;
  primitiveIndexByMaterialId: ReadonlyMap<number, number>;
  /** Owner IDs the SVO dry-scene hit loop should ignore for an interior view. */
  skipOwnerIds: readonly number[];
  openShellOwnerId?: number;
  unsupportedSources: readonly SvoUnsupportedStaticSource[];
  requiresRasterTerrainFallback: boolean;
  /** Analytic heightfield consumed directly from the packed scene uniforms. */
  analyticTerrain?: SvoAnalyticTerrainSource;
  /** Content hash over packed records, identities, bounds policy, and terrain. */
  staticRevision: string;
  /** Versioned identity used to reuse immutable packed publication records. */
  cacheKey: string;
}

function positiveSafeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function environmentIdentity(scene: Pick<SceneDescription, "rigidBodies">, primitive: EnvironmentProxyPrimitive) {
  const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
  const ownerId = scene.rigidBodies.length + primitive.ownerIndex;
  if (materialId > 0xffff) throw new RangeError(`Environment material ID for ${primitive.key} does not fit uint16`);
  // 0xffff is the established no-owner sentinel, so authored owners stop at 0xfffe.
  if (ownerId >= 0xffff) throw new RangeError(`Environment owner ID for ${primitive.key} collides with the no-owner sentinel`);
  return { materialId, ownerId };
}

function descriptorForProxy(
  scene: Pick<SceneDescription, "rigidBodies">,
  primitive: EnvironmentProxyPrimitive,
): SvoPrimitiveDescriptor {
  const identity = environmentIdentity(scene, primitive);
  // Primitive ID follows the scene-global owner ID. It is stable for the same
  // scene body list and catalog, and remains distinct from the packed index.
  const base = {
    primitiveId: identity.ownerId,
    materialId: identity.materialId,
    ownerId: identity.ownerId,
    center_m: { ...primitive.center_m },
  };
  if (primitive.kind === "box") {
    return { ...base, kind: "box", halfExtents_m: { ...primitive.halfSize_m } };
  }
  if (primitive.kind === "cylinder") {
    return { ...base, kind: "cylinder", radius_m: primitive.radius_m, halfHeight_m: primitive.halfHeight_m };
  }
  return { ...base, kind: "ellipsoid", radii_m: { ...primitive.radius_m } };
}

function frontOpenShell(primitive: EnvironmentProxyPrimitive): boolean {
  return primitive.tags.includes("shell")
    && primitive.tags.includes("wall")
    && primitive.key.endsWith("/shell/wall-front");
}

function coverageBounds(primitive: EnvironmentProxyPrimitive, minimumWidth_m: number): SvoPrimitiveCoverageBounds {
  const axes = ["x", "y", "z"] as const;
  const exact_m = {
    min: { ...primitive.aabb_m.min },
    max: { ...primitive.aabb_m.max },
  };
  if (!(minimumWidth_m > 0)) return { exact_m, conservative_m: exact_m, subcellAxes: [], policy: "exact" };
  const center = primitive.center_m;
  const halfMinimum = 0.5 * minimumWidth_m;
  const subcellAxes = axes.filter((axis) => primitive.aabb_m.max[axis] - primitive.aabb_m.min[axis] < minimumWidth_m);
  const conservative_m = {
    min: {
      x: Math.min(primitive.aabb_m.min.x, center.x - halfMinimum),
      y: Math.min(primitive.aabb_m.min.y, center.y - halfMinimum),
      z: Math.min(primitive.aabb_m.min.z, center.z - halfMinimum),
    },
    max: {
      x: Math.max(primitive.aabb_m.max.x, center.x + halfMinimum),
      y: Math.max(primitive.aabb_m.max.y, center.y + halfMinimum),
      z: Math.max(primitive.aabb_m.max.z, center.z + halfMinimum),
    },
  };
  return { exact_m, conservative_m, subcellAxes, policy: subcellAxes.length > 0 ? "conservative-subcell" : "exact" };
}

/**
 * Convert one existing deterministic environment catalog into bounded SVO
 * primitive records without changing sparse-voxel material or owner identity.
 */
export function svoScenePrimitivesFromEnvironmentCatalog(
  scene: Pick<SceneDescription, "rigidBodies">,
  catalog: EnvironmentProxyCatalog,
  options: Omit<SvoScenePrimitiveBuildOptions, "environmentId"> = {},
): SvoScenePrimitiveBuild {
  const includeShell = options.includeShell ?? true;
  const maximumPrimitives = positiveSafeInteger(
    options.maximumPrimitives ?? SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES,
    "Maximum SVO scene primitives",
  );
  const primitives = environmentProxyPrimitives(catalog, includeShell);
  if (primitives.length > maximumPrimitives) {
    throw new RangeError(`Environment ${catalog.environmentId} needs ${primitives.length} SVO primitives, exceeding the ${maximumPrimitives} record limit`);
  }
  const coverageCellSize_m = options.coverageCellSize_m ?? 0;
  if (!Number.isFinite(coverageCellSize_m) || coverageCellSize_m < 0) {
    throw new RangeError("SVO scene primitive coverage cell size must be finite and non-negative");
  }
  const analyticTerrain: SvoAnalyticTerrainSource | undefined = catalog.shell.kind === "terrain-heightfield" ? {
    kind: "terrain-heightfield",
    materialId: VOXEL_MATERIAL_IDS.terrain,
    normalEpsilon_m: 0.02,
  } : undefined;
  const staticRevision = hashSvoStaticPublication(new Uint32Array(), JSON.stringify({
    environmentId: catalog.environmentId,
    rigidBodyCount: scene.rigidBodies.length,
    coverageCellSize_m,
    primitives,
    analyticTerrain,
  }));
  const cacheKey = `svo-scene-primitives-v${SVO_SCENE_PRIMITIVE_VERSION}:${catalog.environmentId}:${staticRevision}`;
  const cached = cachedSvoStaticPublication(scenePrimitiveCache, cacheKey);
  if (cached) return cached;

  const descriptors: SvoPrimitiveDescriptor[] = [];
  const metadata: SvoEnvironmentPrimitiveMetadata[] = [];
  const primitiveIndexByOwnerId = new Map<number, number>();
  const primitiveIndexByMaterialId = new Map<number, number>();
  let openShellOwnerId: number | undefined;

  for (const primitive of primitives) {
    const primitiveIndex = descriptors.length;
    const { materialId, ownerId } = environmentIdentity(scene, primitive);
    if (primitiveIndexByOwnerId.has(ownerId)) throw new Error(`Duplicate environment owner ID ${ownerId}`);
    if (primitiveIndexByMaterialId.has(materialId)) throw new Error(`Duplicate environment material ID ${materialId}`);
    const openShell = frontOpenShell(primitive);
    if (openShell && openShellOwnerId !== undefined) throw new Error("Environment catalog contains multiple front/open shell owners");
    if (openShell) openShellOwnerId = ownerId;

    descriptors.push(descriptorForProxy(scene, primitive));
    metadata.push({
      primitiveIndex,
      environmentOwnerIndex: primitive.ownerIndex,
      ownerId,
      materialId,
      key: primitive.key,
      group: primitive.group,
      tags: [...primitive.tags],
      sourceKind: primitive.kind,
      material: {
        colorLinear: [...primitive.material.colorLinear],
        emission: primitive.material.emission,
        roughness: primitive.material.roughness,
      },
      shell: primitive.tags.includes("shell"),
      openShell,
      coverageBounds: coverageBounds(primitive, coverageCellSize_m),
    });
    primitiveIndexByOwnerId.set(ownerId, primitiveIndex);
    primitiveIndexByMaterialId.set(materialId, primitiveIndex);
  }

  const unsupportedSources: SvoUnsupportedStaticSource[] = [];
  const primitiveCandidates = descriptors.length <= SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES
    && descriptors.every(({ kind }) => kind !== "terrain-heightfield")
    ? buildSvoPrimitiveCandidates(descriptors as SvoFinitePrimitiveDescriptor[], { skippedOwnerId: openShellOwnerId })
    : undefined;

  const packedRecords = packSvoPrimitiveRecords(descriptors);
  return internSvoStaticPublication(scenePrimitiveCache, cacheKey, {
    environmentId: catalog.environmentId,
    descriptors,
    packedRecords,
    primitiveCandidates,
    metadata,
    primitiveIndexByOwnerId,
    primitiveIndexByMaterialId,
    skipOwnerIds: openShellOwnerId === undefined ? [] : [openShellOwnerId],
    openShellOwnerId,
    unsupportedSources,
    requiresRasterTerrainFallback: unsupportedSources.length > 0,
    analyticTerrain,
    staticRevision,
    cacheKey,
  });
}

/** Build the selected scene environment catalog and convert it in one call. */
export function buildSvoScenePrimitives(
  scene: SceneDescription,
  options: SvoScenePrimitiveBuildOptions = {},
): SvoScenePrimitiveBuild {
  const environmentId = options.environmentId ?? scene.environment ?? "default";
  const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
  return svoScenePrimitivesFromEnvironmentCatalog(scene, catalog, {
    ...options,
    // Match the default-camera acceptance audit: features below 1.5 nominal
    // cells retain conservative coverage even when cell-centre sampling moves.
    coverageCellSize_m: options.coverageCellSize_m ?? 1.5 * scene.nominalResolution.length_m,
  });
}
