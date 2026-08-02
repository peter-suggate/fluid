import type { EnvironmentId } from "./environments";
import type { SceneDescription } from "./model";
import {
  cachedSvoStaticPublication,
  hashSvoStaticPublication,
  internSvoStaticPublication,
} from "./svo-static-publication-cache";
import {
  packSvoThickGlassVolumes,
  svoThickGlassBounds,
  type SvoThickGlassShape,
  type SvoThickGlassVolume,
} from "./svo-thick-glass";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "./voxel-environments";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE } from "./webgpu-octree-sparse-bricks";

export const SVO_SCENE_THICK_GLASS_VERSION = "1" as const;
export const SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES = 32;
export const SVO_SCENE_THICK_GLASS_DEFAULT_ABSORPTION_M_INV = [0.18, 0.04, 0.03] as const;

export type SvoSceneThickGlassRole = "emissive-globe" | "station-observation-lens";

export interface SvoSceneThickGlassMetadata {
  recordIndex: number;
  key: string;
  sourceKey: string;
  role: SvoSceneThickGlassRole;
  glassId: number;
  materialId: number;
  ownerId: number;
  shape: SvoThickGlassShape;
  bounds_m: ReturnType<typeof svoThickGlassBounds>;
  /** Curved volume is consumed by the renderer-owned bounded uniform binder. */
  productionBinding: "renderer-uniform-binder";
  replacesThinPaneKey?: string;
  replacesUnsupportedKey?: string;
}

export interface SvoSceneThickGlassBuild {
  environmentId: EnvironmentId;
  revision: number;
  descriptors: readonly SvoThickGlassVolume[];
  packedRecords: Uint32Array<ArrayBuffer>;
  metadata: readonly SvoSceneThickGlassMetadata[];
  staticRevision: string;
  cacheKey: string;
}

export interface SvoSceneThickGlassBuildOptions {
  environmentId?: EnvironmentId;
  revision?: number;
  maximumVolumes?: number;
}

interface AuthoredThickGlass {
  key: string;
  sourceKey: string;
  role: SvoSceneThickGlassRole;
  descriptor: SvoThickGlassVolume;
  replacesThinPaneKey?: string;
  replacesUnsupportedKey?: string;
}

const sceneThickGlassCache = new Map<string, SvoSceneThickGlassBuild>();

function positiveUint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 1 || value > 0xffff_ffff) throw new RangeError(`${label} must be a positive uint32`);
  return value >>> 0;
}

function boundedCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES) {
    throw new RangeError(`SVO scene thick-glass capacity must be from 1 to ${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}`);
  }
  return value;
}

function equalRadii(radii: readonly [number, number, number]): boolean {
  return Math.max(...radii) - Math.min(...radii) <= Math.max(...radii) * 1e-6;
}

export function buildSvoSceneThickGlass(
  scene: SceneDescription,
  options: SvoSceneThickGlassBuildOptions = {},
): SvoSceneThickGlassBuild {
  const environmentId = options.environmentId ?? scene.environment ?? "default";
  const revision = positiveUint32(options.revision ?? 1, "SVO scene thick-glass revision");
  const maximumVolumes = boundedCount(options.maximumVolumes ?? SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES);
  const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
  const proxies = environmentProxyPrimitives(catalog, true);
  const glassIdBase = 0x4000 + catalog.environmentIndex * 0x100;
  const authored: AuthoredThickGlass[] = [];

  // Curved glass, from the group a scenery node declares. Both roles used to be
  // reached differently — globes by group, the station lens by a branch on the
  // environment id pointing at a key no primitive published — which meant the
  // lens was authored twice, here and in the wall it sits in, and only one of
  // the two would have moved if the room were resized.
  for (const proxy of proxies) {
    const globe = proxy.group === "emissive-glass" && proxy.kind === "ellipsoid";
    const port = proxy.group === "porthole-glass";
    if (!globe && !port) continue;
    // A round port is authored as a disc and may be turned into a wall, so its
    // lens takes the world extent it actually occupies rather than its local
    // radii. A globe is a sphere either way.
    const radii = globe
      ? [proxy.radius_m.x, proxy.radius_m.y, proxy.radius_m.z] as const
      : [
        .5 * (proxy.aabb_m.max.x - proxy.aabb_m.min.x),
        .5 * (proxy.aabb_m.max.y - proxy.aabb_m.min.y),
        .5 * (proxy.aabb_m.max.z - proxy.aabb_m.min.z),
      ] as const;
    authored.push({
      key: globe ? `${proxy.key}/thick-glass` : `${proxy.key}/thick-lens`,
      sourceKey: proxy.key,
      role: globe ? "emissive-globe" : "station-observation-lens",
      descriptor: {
        glassId: glassIdBase + authored.length,
        materialId: globe ? ENVIRONMENT_VOXEL_MATERIAL_BASE + proxy.ownerIndex : VOXEL_MATERIAL_IDS.containerGlass,
        ownerId: scene.rigidBodies.length + proxy.ownerIndex,
        revision,
        shape: equalRadii(radii) ? "sphere" : "ellipsoid",
        center_m: [proxy.center_m.x, proxy.center_m.y, proxy.center_m.z],
        radii_m: radii,
        absorption_mInv: SVO_SCENE_THICK_GLASS_DEFAULT_ABSORPTION_M_INV,
      },
      replacesUnsupportedKey: proxy.key,
    });
  }

  if (authored.length > maximumVolumes) {
    throw new RangeError(`Environment ${environmentId} needs ${authored.length} thick-glass volumes, exceeding the ${maximumVolumes} record limit`);
  }
  const staticRevision = hashSvoStaticPublication(new Uint32Array(), JSON.stringify({ environmentId, revision, authored }));
  const cacheKey = `svo-scene-thick-glass-v${SVO_SCENE_THICK_GLASS_VERSION}:${environmentId}:${staticRevision}`;
  const cached = cachedSvoStaticPublication(sceneThickGlassCache, cacheKey);
  if (cached) return cached;
  const descriptors = authored.map(({ descriptor }) => descriptor);
  const metadata = authored.map((entry, recordIndex): SvoSceneThickGlassMetadata => ({
    recordIndex,
    key: entry.key,
    sourceKey: entry.sourceKey,
    role: entry.role,
    glassId: entry.descriptor.glassId,
    materialId: entry.descriptor.materialId,
    ownerId: entry.descriptor.ownerId!,
    shape: entry.descriptor.shape,
    bounds_m: svoThickGlassBounds(entry.descriptor),
    productionBinding: "renderer-uniform-binder",
    ...(entry.replacesThinPaneKey ? { replacesThinPaneKey: entry.replacesThinPaneKey } : {}),
    ...(entry.replacesUnsupportedKey ? { replacesUnsupportedKey: entry.replacesUnsupportedKey } : {}),
  }));
  return internSvoStaticPublication(sceneThickGlassCache, cacheKey, {
    environmentId,
    revision,
    descriptors,
    packedRecords: packSvoThickGlassVolumes(descriptors),
    metadata,
    staticRevision,
    cacheKey,
  });
}
