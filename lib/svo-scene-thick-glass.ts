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

  for (const proxy of proxies) {
    if (proxy.group !== "emissive-glass" || proxy.kind !== "ellipsoid") continue;
    const radii = [proxy.radius_m.x, proxy.radius_m.y, proxy.radius_m.z] as const;
    authored.push({
      key: `${proxy.key}/thick-glass`,
      sourceKey: proxy.key,
      role: "emissive-globe",
      descriptor: {
        glassId: glassIdBase + authored.length,
        materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + proxy.ownerIndex,
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

  if (environmentId === "research-station") {
    const s = catalog.scale_m;
    const nextOwner = Math.max(...proxies.map(({ ownerIndex }) => ownerIndex), -1) + 1;
    authored.push({
      key: "research-station/observation-port/thick-lens",
      sourceKey: "research-station/shell/procedural-portholes",
      role: "station-observation-lens",
      descriptor: {
        glassId: glassIdBase + authored.length,
        materialId: VOXEL_MATERIAL_IDS.containerGlass,
        ownerId: scene.rigidBodies.length + nextOwner,
        revision,
        shape: "ellipsoid",
        center_m: [0, catalog.floorY_m + 1.55 * s, catalog.shell.bounds_m.min.z + .018 * s],
        radii_m: [.66 * s, .39 * s, .018 * s],
        absorption_mInv: SVO_SCENE_THICK_GLASS_DEFAULT_ABSORPTION_M_INV,
      },
      replacesThinPaneKey: "research-station/observation-port/glazing",
      replacesUnsupportedKey: "research-station/shell/procedural-portholes",
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
