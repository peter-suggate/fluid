import { environmentIds, type EnvironmentId } from "../core/environments";
import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "../core/model";
import { cameraForPreset, scenePresets } from "../core/scenes";
import { buildSvoSceneGlass, type SvoSceneGlassUnsupportedEntry } from "./svo-scene-glass";
import { buildSvoSceneThickGlass, type SvoSceneThickGlassMetadata } from "./svo-scene-thick-glass";
import { buildSvoSceneLights } from "./svo-light-abi";
import { svoFieldProgramFeatureRadius_m } from "./svo-field-program";
import { svoClusterFeatureRadius_m } from "./svo-primitive-abi";
import { buildSvoScenePrimitives } from "./svo-scene-primitives";
import {
  cachedSvoPublication,
  hashSvoPublication,
  internSvoPublication,
} from "./svo-publication-cache";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives, type EnvironmentProxyPrimitive } from "../core/voxel-environments";
import { materialIdForRigidShape } from "../core/voxel-scene";

export const SVO_SCENE_COVERAGE_VERSION = 2;

const environmentCoverageCache = new Map<string, SvoEnvironmentCoverageReport>();
const shippedCoverageCache = new Map<string, SvoShippedSceneCoverageReport>();

export type SvoSceneVisibleOwnership =
  | "analytic-primitive"
  | "analytic-rigid-body"
  | "analytic-terrain"
  | "thin-glass"
  | "thick-glass"
  | "opaque-proxy-fallback"
  | "not-visible";

export type SvoSceneCollisionOwnership =
  | "solver-rigid-body"
  | "solver-environment-proxy"
  | "solver-terrain-heightfield"
  | "solver-container-boundary"
  | "none-presentation-only";

export type SvoSceneLightingOwnership =
  | "svo-area-light"
  | "svo-directional-environment"
  | "emissive-surface-only"
  | "omitted-light-capacity"
  | "none";

export type SvoSceneCoverageStatus = "complete" | "degraded" | "unsupported";

export interface SvoSceneCoverageEntry {
  key: string;
  category: "shell" | "prop" | "rigid-body" | "terrain" | "container-glass" | "environment-glazing" | "environment-light";
  status: SvoSceneCoverageStatus;
  visibleOwnership: SvoSceneVisibleOwnership;
  collisionOwnership: SvoSceneCollisionOwnership;
  lightingOwnership: SvoSceneLightingOwnership;
  /** Stable identities are present for every analytic/opaque source. */
  materialId?: number;
  ownerId?: number;
  sourceKind?: string;
  defaultCameraPriority: boolean;
  reason?: string;
  fallback?: string;
  /** Conservative audit bounds for subcell analytic/collision proxy coverage. */
  collisionProxyBounds_m?: EnvironmentProxyPrimitive["aabb_m"];
  boundsPolicy?: "exact" | "conservative-subcell";
  subcellAxes?: readonly ("x" | "y" | "z")[];
  plannedThickGlassId?: number;
  plannedThickGlassContract?: "analytic-thick-glass-bound";
}

export interface SvoEnvironmentCoverageReport {
  environmentId: EnvironmentId;
  entries: readonly SvoSceneCoverageEntry[];
  summary: Readonly<{
    complete: number;
    degraded: number;
    unsupported: number;
    defaultCameraPriority: number;
    analyticPrimitives: number;
    thinGlassPanes: number;
    thickGlassVolumes: number;
    lights: number;
  }>;
  unsupportedEntries: readonly SvoSceneCoverageEntry[];
  contentRevision: string;
  cacheKey: string;
}

export interface SvoPresetCoverageReport {
  presetId: string;
  environmentId: EnvironmentId;
  sceneId: string;
  camera: Readonly<CameraState>;
  environment: SvoEnvironmentCoverageReport;
  rigidBodies: readonly SvoSceneCoverageEntry[];
  unsupportedEntries: readonly SvoSceneCoverageEntry[];
}

export interface SvoShippedSceneCoverageReport {
  version: typeof SVO_SCENE_COVERAGE_VERSION;
  environments: readonly SvoEnvironmentCoverageReport[];
  presets: readonly SvoPresetCoverageReport[];
  contentRevision: string;
  cacheKey: string;
}

function dimensions(proxy: EnvironmentProxyPrimitive): readonly [number, number, number] {
  if (proxy.kind === "box") return [2 * proxy.halfSize_m.x, 2 * proxy.halfSize_m.y, 2 * proxy.halfSize_m.z];
  if (proxy.kind === "cylinder") return [2 * proxy.radius_m, 2 * proxy.halfHeight_m, 2 * proxy.radius_m];
  // The thin-feature test below wants the smallest solid width, so a swept
  // shape reports its tube, not the extent of the run it is swept along.
  if (proxy.kind === "capsule") return [2 * proxy.radius_m, 2 * proxy.radius_m, 2 * proxy.radius_m];
  if (proxy.kind === "torus") return [2 * proxy.minorRadius_m, 2 * proxy.minorRadius_m, 2 * proxy.minorRadius_m];
  if (proxy.kind === "cone") {
    const widest = 2 * Math.max(proxy.baseRadius_m, proxy.topRadius_m);
    return [widest, 2 * proxy.halfHeight_m, widest];
  }
  // An aggregate's *envelope* is large and its smallest solid width is one
  // feature of its field — a floret, a lobe, a tube section — which is the
  // number this test wants: a cluster of two-centimetre spheres in a half-metre
  // canopy is thin-featured however big the canopy is, and reading the envelope
  // here would let cell-centre voxelization drop it.
  if (proxy.kind === "cluster") {
    const feature = 2 * svoClusterFeatureRadius_m(proxy.radius_m, proxy.packing);
    return [feature, feature, feature];
  }
  // A tape's smallest solid width is the same question one level up, and the
  // tape answers it directly rather than through an envelope: `worley-subtract`
  // knows its pit period, a `scatter` knows its cell. Reading the conservative
  // box here would call a pitted stone thick because the stone is, which is the
  // failure the cluster arm above exists to avoid.
  if (proxy.kind === "field-program") {
    const feature = 2 * svoFieldProgramFeatureRadius_m(proxy.program);
    return [feature, feature, feature];
  }
  return [2 * proxy.radius_m.x, 2 * proxy.radius_m.y, 2 * proxy.radius_m.z];
}

function priorityProxy(scene: SceneDescription, proxy: EnvironmentProxyPrimitive): boolean {
  // Anything at or below 1.5 nominal cells is vulnerable to camera-dependent
  // disappearance in cell-centre voxelization and therefore remains a direct
  // analytic/default-camera acceptance priority.
  const thinThreshold = Math.max(1.5 * scene.voxelDomain.finestCellSize_m, .025 * Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m));
  return Math.min(...dimensions(proxy)) <= thinThreshold
    || proxy.tags.some((tag) => ["fixture", "monitor", "instrument", "fruit", "flower", "watering-can"].includes(tag));
}

export function buildSvoEnvironmentCoverage(scene: SceneDescription, environmentId: EnvironmentId): SvoEnvironmentCoverageReport {
  const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
  const primitiveBuild = buildSvoScenePrimitives(scene, { environmentId });
  const glass = buildSvoSceneGlass(scene, { environmentId });
  const thickGlass = buildSvoSceneThickGlass(scene, { environmentId });
  const lights = buildSvoSceneLights(scene, { environmentId });
  const contentRevision = hashSvoPublication(new Uint32Array(), JSON.stringify({
    environmentId,
    primitiveRevision: primitiveBuild.contentRevision,
    glassRevision: glass.contentRevision,
    thickGlassRevision: thickGlass.contentRevision,
    lightRevision: lights.contentRevision,
  }));
  const cacheKey = `svo-scene-coverage-v${SVO_SCENE_COVERAGE_VERSION}:${environmentId}:${contentRevision}`;
  const cached = cachedSvoPublication(environmentCoverageCache, cacheKey);
  if (cached) return cached;
  const selectedLights = new Set(lights.records.map(({ sourceKey }) => sourceKey));
  const omittedLights = new Set(lights.omittedFixtureKeys);
  const proxies = new Map(environmentProxyPrimitives(catalog).map((proxy) => [proxy.key, proxy]));
  const unsupportedGlass = new Map(glass.unsupportedEntries.map((entry) => [entry.key, entry]));
  const thickGlassBySource = new Map(thickGlass.metadata.map((entry) => [entry.sourceKey, entry]));
  const entries: SvoSceneCoverageEntry[] = primitiveBuild.metadata.map((metadata) => {
    const proxy = proxies.get(metadata.key)!;
    const opticalGap = unsupportedGlass.get(metadata.key);
    const plannedThickGlass = thickGlassBySource.get(metadata.key);
    const lightingOwnership: SvoSceneLightingOwnership = selectedLights.has(metadata.key) ? "svo-area-light"
      : omittedLights.has(metadata.key) ? "omitted-light-capacity"
        : metadata.material.emission > 0 ? "emissive-surface-only" : "none";
    const documentedSurfaceOnly = lightingOwnership === "emissive-surface-only" && proxy.tags.includes("emissive-surface-only");
    const missingEmitterLight = lightingOwnership === "emissive-surface-only" && !documentedSurfaceOnly;
    const degradedReason = (plannedThickGlass ? undefined : opticalGap?.reason)
      ?? (lightingOwnership === "omitted-light-capacity" ? "light-record-capacity" : undefined)
      ?? (missingEmitterLight ? "emissive-owner-missing-light" : undefined);
    return {
      key: metadata.key,
      category: metadata.shell ? "shell" : "prop",
      status: degradedReason ? "degraded" : "complete",
      visibleOwnership: plannedThickGlass ? "thick-glass" : opticalGap ? "opaque-proxy-fallback" : "analytic-primitive",
      collisionOwnership: "solver-environment-proxy",
      lightingOwnership,
      materialId: metadata.materialId,
      ownerId: metadata.ownerId,
      sourceKind: metadata.sourceKind,
      defaultCameraPriority: priorityProxy(scene, proxy),
      collisionProxyBounds_m: metadata.coverageBounds.conservative_m,
      boundsPolicy: metadata.coverageBounds.policy,
      subcellAxes: metadata.coverageBounds.subcellAxes,
      ...(plannedThickGlass ? {
        plannedThickGlassId: plannedThickGlass.glassId,
        plannedThickGlassContract: "analytic-thick-glass-bound" as const,
        // Why this primitive is on the thick-glass path at all. A bound volume
        // makes the gap complete rather than making it disappear: the audit
        // still has to say that a round port is not a finite pane.
        ...(opticalGap ? { reason: opticalGap.reason } : {}),
      } : {}),
      ...(documentedSurfaceOnly ? { reason: "documented-low-power-emissive-surface" } : {}),
      ...(degradedReason ? {
        reason: degradedReason,
        fallback: opticalGap?.fallback ?? "emissive-surface-only",
      } : {}),
    };
  });
  if (primitiveBuild.analyticTerrain) entries.push({
    key: `${environmentId}/terrain-heightfield`,
    category: "terrain",
    status: "complete",
    visibleOwnership: "analytic-terrain",
    collisionOwnership: "solver-terrain-heightfield",
    lightingOwnership: "none",
    materialId: primitiveBuild.analyticTerrain.materialId,
    sourceKind: primitiveBuild.analyticTerrain.kind,
    defaultCameraPriority: true,
  });
  for (const metadata of glass.metadata) { const replacingVolume = thickGlass.metadata.find(({ replacesThinPaneKey }) => replacesThinPaneKey === metadata.key); entries.push({
    key: metadata.key,
    category: metadata.role === "environment-glazing" ? "environment-glazing" : "container-glass",
    status: metadata.opaqueCutoutKey && !replacingVolume ? "unsupported" : "complete",
    visibleOwnership: replacingVolume ? "thick-glass" : "thin-glass",
    collisionOwnership: metadata.role === "environment-glazing" ? "none-presentation-only" : "solver-container-boundary",
    lightingOwnership: "none",
    materialId: metadata.materialId,
    ownerId: metadata.ownerId,
    sourceKind: metadata.role,
    defaultCameraPriority: true,
    ...(replacingVolume ? { plannedThickGlassId: replacingVolume.glassId, plannedThickGlassContract: "analytic-thick-glass-bound" as const }
      : metadata.opaqueCutoutKey ? { reason: "opaque-cutout-required", fallback: metadata.opaqueCutoutKey } : {}),
  }); }
  entries.push({
    key: "authored/directional",
    category: "environment-light",
    status: "complete",
    visibleOwnership: "not-visible",
    collisionOwnership: "none-presentation-only",
    lightingOwnership: "svo-directional-environment",
    defaultCameraPriority: false,
  });
  const count = (status: SvoSceneCoverageStatus) => entries.filter((entry) => entry.status === status).length;
  const unsupportedEntries = entries.filter(({ status }) => status !== "complete");
  const summary = Object.freeze({
    complete: count("complete"), degraded: count("degraded"), unsupported: count("unsupported"),
    defaultCameraPriority: entries.filter(({ defaultCameraPriority }) => defaultCameraPriority).length,
    analyticPrimitives: primitiveBuild.metadata.length,
    thinGlassPanes: glass.metadata.length,
    thickGlassVolumes: thickGlass.metadata.length,
    lights: lights.records.length,
  });
  return internSvoPublication(environmentCoverageCache, cacheKey, Object.freeze({
    environmentId,
    entries: Object.freeze(entries),
    summary,
    unsupportedEntries: Object.freeze(unsupportedEntries),
    contentRevision,
    cacheKey,
  }));
}

export function buildSvoShippedSceneCoverage(): SvoShippedSceneCoverageReport {
  const environments = environmentIds.map((environmentId) => {
    const scene = cloneScene(defaultScene);
    scene.environment = environmentId;
    return buildSvoEnvironmentCoverage(scene, environmentId);
  });
  const presets = scenePresets.map((preset): SvoPresetCoverageReport => {
    const scene = preset.create();
    const environmentId = scene.environment ?? preset.background;
    const environment = buildSvoEnvironmentCoverage(scene, environmentId);
    const rigidBodies: SvoSceneCoverageEntry[] = scene.rigidBodies.map((body, ownerId) => ({
      key: `rigid-body/${body.id}`,
      category: "rigid-body",
      status: "complete",
      visibleOwnership: "analytic-rigid-body",
      collisionOwnership: "solver-rigid-body",
      lightingOwnership: "none",
      materialId: materialIdForRigidShape(body.shape),
      ownerId,
      sourceKind: body.shape,
      defaultCameraPriority: true,
    }));
    return Object.freeze({
      presetId: preset.id,
      environmentId,
      sceneId: scene.sceneId,
      camera: Object.freeze(cameraForPreset(preset)),
      environment,
      rigidBodies: Object.freeze(rigidBodies),
      unsupportedEntries: Object.freeze(environment.unsupportedEntries),
    });
  });
  const contentRevision = hashSvoPublication(new Uint32Array(), JSON.stringify({
    environments: environments.map(({ cacheKey }) => cacheKey),
    presets,
  }));
  const cacheKey = `svo-shipped-scene-coverage-v${SVO_SCENE_COVERAGE_VERSION}:${contentRevision}`;
  return internSvoPublication(shippedCoverageCache, cacheKey, Object.freeze({
    version: SVO_SCENE_COVERAGE_VERSION,
    environments: Object.freeze(environments),
    presets: Object.freeze(presets),
    contentRevision,
    cacheKey,
  }));
}

/** Stable text suitable for checked-in reports and content hashing. */
export function canonicalSvoSceneCoverage(report: SvoShippedSceneCoverageReport): string {
  return JSON.stringify(report);
}
