import { svoPrimitiveWGSL, SVO_PRIMITIVE_RECORD_STRIDE_BYTES } from "./svo-primitive-abi";
import {
  SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK,
  SVO_PRIMITIVE_CANDIDATE_VERSION,
  packSvoPrimitiveCandidateArena,
  type SvoPrimitiveCandidateArena,
  type SvoPrimitiveCandidatePublication,
} from "./svo-primitive-candidates";
import { SVO_PRIMITIVE_MOTION_STRIDE_BYTES, svoPrimitiveMotionWGSL } from "./svo-primitive-motion";
import { svoGBufferWGSL } from "./svo-gbuffer";
import {
  buildSvoEnvironmentLighting,
  SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
  SVO_ENVIRONMENT_LIGHTING_VERSION,
  svoEnvironmentLightingWGSL,
} from "./svo-environment-lighting";
import {
  SVO_LIGHT_MAXIMUM_RECORDS,
  SVO_LIGHT_RECORD_STRIDE_BYTES,
  SVO_LIGHT_RECORD_WORDS,
  buildSvoSceneLights,
  svoLightWGSL,
} from "./svo-light-abi";
import type { SceneDescription } from "./model";
import { svoMaterialWGSL, SVO_MATERIAL_RECORD_STRIDE_BYTES } from "./svo-material-abi";
import { svoProceduralMaterialWGSL } from "./svo-procedural-material";
import { SVO_SCENE_GLASS_MAXIMUM_PANES } from "./svo-scene-glass";
import { SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES } from "./svo-scene-thick-glass";
import { SVO_CONTACT_VISIBILITY_CONTRACT } from "./svo-contact-visibility";
import { SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES, svoTerrainMaterialWGSL } from "./svo-terrain-material";
import { svoThinGlassWGSL, SVO_THIN_GLASS_RECORD_STRIDE_BYTES } from "./svo-thin-glass";
import {
  SVO_THICK_GLASS_RECORD_STRIDE_BYTES,
  svoThickGlassWGSL,
  unpackSvoThickGlassVolumes,
} from "./svo-thick-glass";
import { SVO_VISIBILITY_LIMITS, svoVisibilityRaysWGSL } from "./svo-visibility-rays";
import {
  resolveSvoFluidRenderOwnership,
  svoFluidMediaPathWGSL,
  type SvoFluidRenderOwnership,
} from "./svo-fluid-media-path";
import { terrainHeightAt, terrainNormalAt, type TerrainDescription } from "./terrain";
import { unifiedLightingShaderLibrary, WATER_OPTICS } from "./webgpu-lighting";
import {
  canConsumeSparseVoxelCoarseFluidPrimary,
  createSvoStructuralFluidPrimaryWGSL,
  DEFAULT_SVO_FLUID_PRIMARY_MODE,
  SVO_STRUCTURAL_FLUID_PRIMARY_LIMITS,
  SVO_STRUCTURAL_FLUID_PRIMARY_STORAGE_BINDINGS,
  svoFluidPrimaryModeWord,
  type SvoFluidPrimaryMode,
} from "./webgpu-svo-fluid-primary";
import {
  svoFinePhiPackedSamplingWGSL,
  svoFinePhiTypesWGSL,
  type SvoFineFluidGpuCapability,
} from "./webgpu-svo-fine-phi-stager";
import { createWebgpuSvoTraversalWGSL } from "./webgpu-svo-traversal";
import {
  SparseVoxelGBufferTargetArena,
  SVO_GBUFFER_RENDER_TARGET_CONTRACT,
  type SparseVoxelGBufferTextures,
} from "./webgpu-svo-gbuffer-targets";
import {
  SparseVoxelGpuPickingReadbackRing,
  svoPickingPixelFromNormalized,
  type SvoGpuPickingReadbackResult,
} from "./webgpu-svo-picking-readback";
import type { SparseVoxelSceneRenderSource } from "./webgpu-voxel-debug";
import { DEFAULT_SVO_LIGHTING_MODE, type SvoLightingMode } from "./svo-render-mode";
import type { DrySceneReplacementResult, TimestampRange } from "./webgpu-water-pipeline";
import { SparseVoxelTemporalAccumulator, type SparseVoxelTemporalFrameState } from "./webgpu-svo-temporal-accumulator";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { svoNodeMipSamplingWGSL } from "./svo-node-mip-sampling";

export interface SparseVoxelDrySceneData {
  /** Packed `SvoPrimitiveRecord` values in dense environment-owner order. */
  primitiveRecords: Uint32Array<ArrayBuffer>;
  /** Static conservative BVH appended to binding 7 after the primitive records. */
  primitiveCandidates?: SvoPrimitiveCandidatePublication;
  /** First owner ID belonging to primitive zero (rigid bodies occupy lower IDs). */
  ownerBase: number;
  /** Interior-facing shell pane omitted so the camera can see into the room. */
  skippedOwnerId?: number;
  /** Stable sparse material-table identity for an analytic terrain hit. */
  terrainMaterialId?: number;
  /** Packed 16-byte garden terrain material metadata; absent preserves table shading. */
  terrainMaterialMetadata?: Uint32Array<ArrayBuffer>;
  /** Stable identity of the packed terrain material policy for diagnostics/caches. */
  terrainMaterialCacheKey?: string;
  /** Packed 80-byte finite-pane records. Empty means this scene has no glass. */
  glassRecords?: Uint32Array<ArrayBuffer>;
  /** Versioned static content key used to avoid redundant pane uploads. */
  glassCacheKey?: string;
  /** Packed analytic sphere/ellipsoid glass records mirrored into a renderer-owned uniform arena. */
  thickGlassRecords?: Uint32Array<ArrayBuffer>;
  thickGlassRevision?: number;
  thickGlassCacheKey?: string;
  /** Thin pane replaced by a curved volume only while the thick binder is valid. */
  thickGlassReplacedThinPaneId?: number;
  /** First vessel-pane ID owned by the existing post-dry-scene glass compositor. */
  primaryCompositeOwnedGlassPaneIdBase?: number;
  /** Contiguous vessel-pane count beginning at `primaryCompositeOwnedGlassPaneIdBase`. */
  primaryCompositeOwnedGlassPaneCount?: number;
  /** CPU-built mirror of the producer's bounded 112-byte light publication. */
  lightRecords?: Uint32Array<ArrayBuffer>;
  /** CPU-built mirror revision; must equal the authoritative source publication. */
  lightRevision?: number;
  /** CPU-built mirror of the selected 96-byte environment-lighting record. */
  environmentLightingRecord?: Uint32Array<ArrayBuffer>;
  /** Content identity; must equal the authoritative source publication. */
  environmentLightingCacheKey?: string;
  /**
   * Explicit handoff contract. The diagnostic mode owns the primary fluid
   * surface, so its caller must suppress the legacy water-interface compositor.
   * Omission retains that compositor and never evaluates fluid in this pass.
   */
  fluidPrimaryMode?: SvoFluidPrimaryMode;
  /** Explicit experimental gate; omission forces the complete legacy-water path. */
  directFluidMediaEndToEndValidated?: boolean;
  /** Experimental bounded indirect-diffuse contact visibility; default is intentionally off pending timing acceptance. */
  contactVisibilityEnabled?: boolean;
  /** Diagnostic primary/shadow split; omission keeps production hard visibility enabled. */
  shadowVisibilityEnabled?: boolean;
  lightDirection?: readonly [number, number, number];
  lightColor?: readonly [number, number, number];
}

export const SVO_DRY_RIGID_MOTION_CAPACITY = 12;
export const SVO_DRY_RIGID_MOTION_UNIFORM_BYTES = SVO_DRY_RIGID_MOTION_CAPACITY * SVO_PRIMITIVE_MOTION_STRIDE_BYTES;
export const SVO_DRY_THICK_GLASS_BINDER_VERSION = 1;
export const SVO_DRY_THICK_GLASS_ARENA_LAYOUT = Object.freeze({
  metadataWordOffset: 0,
  recordWordOffset: 4,
  sizeBytes: 16 + SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES * SVO_THICK_GLASS_RECORD_STRIDE_BYTES,
} as const);

/** Single source of truth for every group-0 declaration and production layout entry. */
export const SVO_DRY_SCENE_BINDING_CONTRACT = Object.freeze([
  ...[0, 1].map((binding) => ({ binding, type: "uniform" as const })),
  ...[2, 3, 4, 5, 6, 7, 8].map((binding) => ({ binding, type: "read-only-storage" as const })),
  { binding: 9, type: "uniform" as const },
  ...[10, 11, 12].map((binding) => ({ binding, type: "read-only-storage" as const })),
  ...[13, 14, 15].map((binding) => ({ binding, type: "uniform" as const })),
  { binding: 16, type: "texture-3d-float" as const },
  { binding: 17, type: "filtering-sampler" as const },
  { binding: 18, type: "texture-2d-uint" as const },
] as const);

export function sparseVoxelDrySceneBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return SVO_DRY_SCENE_BINDING_CONTRACT.map(({ binding, type }): GPUBindGroupLayoutEntry => {
    if (type === "texture-3d-float") return { binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "3d" } };
    if (type === "texture-2d-uint") return { binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint", viewDimension: "2d" } };
    if (type === "filtering-sampler") return { binding, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } };
    return { binding, visibility: GPUShaderStage.FRAGMENT, buffer: { type } };
  });
}

export type SparseVoxelThickGlassBinderStatus =
  | "disabled-empty"
  | "bound"
  | "fallback-malformed"
  | "fallback-overflow"
  | "fallback-stale";

/** Typed optional-binder gate. Any failure retains the existing opaque/thin fallback path. */
export function resolveSparseVoxelThickGlassBinderStatus(
  scene: SparseVoxelDrySceneData | undefined,
): SparseVoxelThickGlassBinderStatus {
  const records = scene?.thickGlassRecords;
  if (!records?.byteLength) return "disabled-empty";
  if (records.byteLength % SVO_THICK_GLASS_RECORD_STRIDE_BYTES !== 0) return "fallback-malformed";
  const count = records.byteLength / SVO_THICK_GLASS_RECORD_STRIDE_BYTES;
  if (count > SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES) return "fallback-overflow";
  const revision = scene?.thickGlassRevision;
  if (!Number.isSafeInteger(revision) || revision! < 1 || revision! > 0xffff_ffff || !scene?.thickGlassCacheKey) {
    return "fallback-malformed";
  }
  if (scene.thickGlassReplacedThinPaneId !== undefined
    && (!Number.isSafeInteger(scene.thickGlassReplacedThinPaneId)
      || scene.thickGlassReplacedThinPaneId < 1 || scene.thickGlassReplacedThinPaneId > 0xffff_ffff)) {
    return "fallback-malformed";
  }
  try {
    const volumes = unpackSvoThickGlassVolumes(records);
    if (volumes.some((volume) => volume.revision !== revision)) return "fallback-stale";
    if (new Set(volumes.map(({ glassId }) => glassId)).size !== volumes.length) return "fallback-malformed";
  } catch {
    return "fallback-malformed";
  }
  return "bound";
}

export function packSparseVoxelDrySceneThickGlassArena(
  scene: SparseVoxelDrySceneData | undefined,
): Uint32Array<ArrayBuffer> {
  const arena = new Uint32Array(new ArrayBuffer(SVO_DRY_THICK_GLASS_ARENA_LAYOUT.sizeBytes));
  if (resolveSparseVoxelThickGlassBinderStatus(scene) !== "bound") return arena;
  const records = scene!.thickGlassRecords!;
  arena.set([
    records.byteLength / SVO_THICK_GLASS_RECORD_STRIDE_BYTES,
    scene!.thickGlassRevision!,
    scene!.thickGlassReplacedThinPaneId ?? 0xffff_ffff,
    SVO_DRY_THICK_GLASS_BINDER_VERSION,
  ], SVO_DRY_THICK_GLASS_ARENA_LAYOUT.metadataWordOffset);
  arena.set(records, SVO_DRY_THICK_GLASS_ARENA_LAYOUT.recordWordOffset);
  return arena;
}

/** DryParams plus static-candidate and renderer-owned fine-phi lanes. */
export const SVO_DRY_SCENE_PARAMS_LAYOUT = Object.freeze({
  sizeBytes: 368,
  terrainWordOffset: 24,
  terrainMaterialWordOffset: 28,
  materialPublicationWordOffset: 32,
  fluidDomainWordOffset: 36,
  primitiveCandidateWordOffset: 40,
  finePhiWordOffset: 44,
  nodeMipWordOffset: 84,
  nodeMipAtlasWordOffset: 88,
} as const);

/** materialPublication.w flags shared by the direct and derived-lighting paths. */
export const SVO_DRY_VISIBILITY_FLAGS = Object.freeze({
  exactContact: 1 << 0,
  exactShadow: 1 << 1,
  coneLightingRequested: 1 << 2,
} as const);

export const SVO_TERRAIN_FAST_MIN_VERTICAL = 0.35;
export const SVO_TERRAIN_FAST_BRACKET_STEPS = 2;
export const SVO_TERRAIN_FAST_REFINEMENTS = 5;
export const SVO_TERRAIN_FALLBACK_STEPS = 20;
export const SVO_TERRAIN_FALLBACK_REFINEMENTS = 8;
/** Includes four terrain-height evaluations used by the central-difference normal. */
export const SVO_TERRAIN_FAST_MAX_HEIGHT_EVALUATIONS = 12;
/** Normal-projected sparse-cell widths used to offset the hard shadow ray. */
export const SVO_DRY_SCENE_SHADOW_BIAS_CELLS = 0.02;
/** Bound direct-light work independently from the producer's 32-record capacity. */
export const SVO_DRY_SCENE_MAX_SHADED_LIGHTS = 8;
/** Two fixed shape samples are stable across frames and keep total visibility work bounded. */
export const SVO_DRY_SCENE_AREA_LIGHT_SAMPLES = 2;
export const SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT = Object.freeze({
  metadataWordOffset: 0,
  lightWordOffset: 4,
  environmentWordOffset: 4 + SVO_LIGHT_MAXIMUM_RECORDS * SVO_LIGHT_RECORD_WORDS,
  sizeBytes: 16 + SVO_LIGHT_MAXIMUM_RECORDS * SVO_LIGHT_RECORD_STRIDE_BYTES + SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
} as const);

// Chrome's WGSL frontend requires the renderer-supplied adapter declaration
// before the shared trace body which calls it. Naga accepts the forward call,
// so keep the composition order explicit here and covered by integration tests.
const SVO_VISIBILITY_TRACE_MARKER = "fn svoTraceVisibility(";
const svoVisibilityTraceOffset = svoVisibilityRaysWGSL.indexOf(SVO_VISIBILITY_TRACE_MARKER);
if (svoVisibilityTraceOffset < 0) throw new Error("SVO visibility WGSL trace marker is missing");
const svoVisibilityPreludeWGSL = svoVisibilityRaysWGSL.slice(0, svoVisibilityTraceOffset);
const svoVisibilityTraceWGSL = svoVisibilityRaysWGSL.slice(svoVisibilityTraceOffset);
const SVO_FLUID_MEDIA_TRACE_MARKER = "fn svoTraceStructuralFluidMedia(";
const svoFluidMediaTraceOffset = svoFluidMediaPathWGSL.indexOf(SVO_FLUID_MEDIA_TRACE_MARKER);
if (svoFluidMediaTraceOffset < 0) throw new Error("SVO fluid media WGSL trace marker is missing");
const svoFluidMediaPreludeWGSL = svoFluidMediaPathWGSL.slice(0, svoFluidMediaTraceOffset);
const svoFluidMediaTraceWGSL = svoFluidMediaPathWGSL.slice(svoFluidMediaTraceOffset);

export interface SvoDirectionalSceneBounds {
  width_m: number;
  height_m: number;
  depth_m: number;
}

/**
 * Finite distance from a point to the directional-light exit of the authored
 * container domain. This CPU mirror keeps secondary-ray clipping testable.
 */
export function directionalLightSceneExitDistance(
  position_m: { x: number; y: number; z: number },
  directionToLight: { x: number; y: number; z: number },
  bounds: SvoDirectionalSceneBounds,
): number {
  const dimensions = [bounds.width_m, bounds.height_m, bounds.depth_m];
  if ([position_m.x, position_m.y, position_m.z, directionToLight.x, directionToLight.y, directionToLight.z, ...dimensions]
    .some((value) => !Number.isFinite(value)) || dimensions.some((value) => !(value > 0))) return 0;
  const magnitude = Math.hypot(directionToLight.x, directionToLight.y, directionToLight.z);
  if (!(magnitude > 1e-12)) return 0;
  const origin = [position_m.x, position_m.y, position_m.z];
  const direction = [directionToLight.x / magnitude, directionToLight.y / magnitude, directionToLight.z / magnitude];
  const minimum = [-0.5 * bounds.width_m, 0, -0.5 * bounds.depth_m];
  const maximum = [0.5 * bounds.width_m, bounds.height_m, 0.5 * bounds.depth_m];
  let enter = 0;
  let exit = Number.POSITIVE_INFINITY;
  for (let axis = 0; axis < 3; axis += 1) {
    if (Math.abs(direction[axis]) <= 1e-12) {
      if (origin[axis] < minimum[axis] || origin[axis] > maximum[axis]) return 0;
      continue;
    }
    const first = (minimum[axis] - origin[axis]) / direction[axis];
    const second = (maximum[axis] - origin[axis]) / direction[axis];
    enter = Math.max(enter, Math.min(first, second));
    exit = Math.min(exit, Math.max(first, second));
    if (exit < enter) return 0;
  }
  return Number.isFinite(exit) ? Math.max(0, exit) : 0;
}

export interface SvoTerrainRayHit {
  t_m: number;
  position_m: { x: number; y: number; z: number };
  normal: { x: number; y: number; z: number };
  solver: "fast" | "fallback";
  heightEvaluations: number;
}

/** CPU mirror of the bounded WGSL terrain bracket/refinement path. */
export function intersectSvoTerrainHeightfield(
  terrain: TerrainDescription | undefined,
  origin_m: { x: number; y: number; z: number },
  direction: { x: number; y: number; z: number },
  sceneScale_m: number,
  normalEpsilon_m = 0.02,
): SvoTerrainRayHit | undefined {
  if (!terrain) return undefined;
  const directionLength = Math.hypot(direction.x, direction.y, direction.z);
  if (!(directionLength > 1e-9) || !(sceneScale_m > 0) || !Number.isFinite(sceneScale_m)) return undefined;
  const rd = { x: direction.x / directionLength, y: direction.y / directionLength, z: direction.z / directionLength };
  const ceiling = terrain.baseHeight_m + terrain.features.reduce((sum, feature) => sum + (feature.kind === "mound" ? feature.amount_m : 0), 0) + 0.05;
  let t0 = 0.005;
  if (origin_m.y > ceiling) {
    if (rd.y >= -0.0005) return undefined;
    t0 = (ceiling - origin_m.y) / rd.y;
  }
  let t1 = t0 + 10 * sceneScale_m;
  if (rd.y < -0.0005) t1 = Math.min(t1, (-0.02 - origin_m.y) / rd.y);
  else if (rd.y > 0.0005) t1 = Math.min(t1, Math.max(t0, (ceiling - origin_m.y) / rd.y));
  if (!(t1 > t0)) return undefined;
  const pointAt = (t: number) => ({ x: origin_m.x + rd.x * t, y: origin_m.y + rd.y * t, z: origin_m.z + rd.z * t });
  let heightEvaluations = 0;
  const fieldAt = (t: number) => {
    const point = pointAt(t);
    heightEvaluations += 1;
    return point.y - terrainHeightAt(terrain, point.x, point.z);
  };
  const surfaceHit = (t_m: number, solver: SvoTerrainRayHit["solver"]): SvoTerrainRayHit => {
    const position_m = pointAt(t_m);
    heightEvaluations += 4;
    return { t_m, position_m, normal: terrainNormalAt(terrain, position_m.x, position_m.z, normalEpsilon_m), solver, heightEvaluations };
  };
  const initialField = fieldAt(t0);
  const ordinaryRay = Math.abs(rd.y) >= SVO_TERRAIN_FAST_MIN_VERTICAL;
  if (Math.abs(initialField) <= 1e-4) return surfaceHit(t0, ordinaryRay ? "fast" : "fallback");

  if (ordinaryRay) {
    let previousT = t0, previousField = initialField;
    for (let bracket = 1; bracket <= SVO_TERRAIN_FAST_BRACKET_STEPS; bracket += 1) {
      const candidateT = t0 + (t1 - t0) * bracket / SVO_TERRAIN_FAST_BRACKET_STEPS;
      const candidateField = fieldAt(candidateT);
      if (Math.abs(candidateField) <= 1e-4) return surfaceHit(candidateT, "fast");
      if ((previousField < 0) !== (candidateField < 0)) {
        let a = previousT, b = candidateT, fieldA = previousField, fieldB = candidateField;
        let bestT = Math.abs(fieldA) < Math.abs(fieldB) ? a : b;
        let bestAbsoluteField = Math.min(Math.abs(fieldA), Math.abs(fieldB));
        for (let refinement = 0; refinement < SVO_TERRAIN_FAST_REFINEMENTS; refinement += 1) {
          const span = b - a;
          const secant = b - fieldB * span / (fieldB - fieldA);
          const t = Math.max(a + span * 0.05, Math.min(b - span * 0.05, Number.isFinite(secant) ? secant : 0.5 * (a + b)));
          const field = fieldAt(t), absoluteField = Math.abs(field);
          if (absoluteField < bestAbsoluteField) { bestAbsoluteField = absoluteField; bestT = t; }
          if (absoluteField <= 1e-4) return surfaceHit(t, "fast");
          if ((fieldA < 0) === (field < 0)) { a = t; fieldA = field; }
          else { b = t; fieldB = field; }
        }
        if (bestAbsoluteField <= 1e-4) return surfaceHit(bestT, "fast");
        break;
      }
      previousT = candidateT;
      previousField = candidateField;
    }
  }

  let previousT = t0;
  let previousField = initialField;
  let closestT = t0;
  let closestAbsoluteField = Math.abs(initialField);
  for (let iteration = 1; iteration <= SVO_TERRAIN_FALLBACK_STEPS; iteration += 1) {
    const t = t0 + (t1 - t0) * (iteration / SVO_TERRAIN_FALLBACK_STEPS) ** 1.4;
    const field = fieldAt(t);
    const absoluteField = Math.abs(field);
    if (absoluteField < closestAbsoluteField) { closestAbsoluteField = absoluteField; closestT = t; }
    if ((previousField < 0) !== (field < 0)) {
      let a = previousT, b = t, fieldA = previousField;
      for (let refinement = 0; refinement < SVO_TERRAIN_FALLBACK_REFINEMENTS; refinement += 1) {
        const middle = 0.5 * (a + b), middleField = fieldAt(middle);
        if ((fieldA < 0) === (middleField < 0)) { a = middle; fieldA = middleField; }
        else b = middle;
      }
      return surfaceHit(0.5 * (a + b), "fallback");
    }
    if (absoluteField <= 1e-4) return surfaceHit(t, "fallback");
    previousT = t;
    previousField = field;
  }
  // Tangent rays do not change sign. Accept only a tightly bounded near-zero
  // sample so near-grazing misses cannot turn into floating terrain specks.
  return closestAbsoluteField <= 5e-4 ? surfaceHit(closestT, "fallback") : undefined;
}

/**
 * Small authored catalogs are cheaper to intersect once than to revisit the
 * root for every empty fluid leaf. Generated catalogs retain SVO payload DDA.
 */
export const SVO_DRY_SCENE_DIRECT_PRIMITIVE_LIMIT = 64;
export type SparseVoxelDrySceneCullingMode = "direct-small-catalog" | "svo-payload-dda";

export function sparseVoxelDrySceneCullingMode(primitiveCount: number): SparseVoxelDrySceneCullingMode {
  if (!Number.isInteger(primitiveCount) || primitiveCount < 0) throw new RangeError("SVO dry-scene primitive count must be a non-negative integer");
  return primitiveCount <= SVO_DRY_SCENE_DIRECT_PRIMITIVE_LIMIT ? "direct-small-catalog" : "svo-payload-dda";
}

/** Metadata-level validation for the producer-owned direct-index PBR table. */
export function canConsumeSparseVoxelPbrMaterials(source: SparseVoxelSceneRenderSource | undefined): boolean {
  const publication = source?.pbrMaterials;
  if (!publication
    || publication.strideBytes !== SVO_MATERIAL_RECORD_STRIDE_BYTES
    || !Number.isSafeInteger(publication.count) || publication.count < 2 || publication.count > 0xffff_ffff
    || !Number.isSafeInteger(publication.revision) || publication.revision < 1 || publication.revision > 0xffff_ffff
    || !publication.binding?.buffer) return false;
  const requiredBytes = publication.count * SVO_MATERIAL_RECORD_STRIDE_BYTES;
  return publication.binding.size === undefined || publication.binding.size >= requiredBytes;
}

/** Validate source metadata and its renderer-owned CPU mirror without reading GPU state back. */
export function canConsumeSparseVoxelLighting(
  source: SparseVoxelSceneRenderSource | undefined,
  scene: SparseVoxelDrySceneData | undefined,
): boolean {
  const lights = source?.lights, environment = source?.environmentLighting;
  const legacyPublication = !lights && !environment
    && !scene?.lightRecords && scene?.lightRevision === undefined
    && !scene?.environmentLightingRecord && scene?.environmentLightingCacheKey === undefined;
  if (legacyPublication) return true;
  if (!lights || !environment || !scene?.lightRecords || !scene.environmentLightingRecord
    || lights.strideBytes !== SVO_LIGHT_RECORD_STRIDE_BYTES
    || !Number.isSafeInteger(lights.count) || lights.count < 1 || lights.count > SVO_LIGHT_MAXIMUM_RECORDS
    || !Number.isSafeInteger(lights.revision) || lights.revision < 1 || lights.revision > 0xffff_ffff
    || !lights.binding?.buffer
    || scene.lightRevision !== lights.revision
    || scene.lightRecords.byteLength !== lights.count * SVO_LIGHT_RECORD_STRIDE_BYTES
    || (lights.binding.size !== undefined && lights.binding.size < lights.count * SVO_LIGHT_RECORD_STRIDE_BYTES)
    || environment.count !== 1
    || environment.strideBytes !== SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES
    || !Number.isSafeInteger(environment.revision) || environment.revision < 1 || environment.revision > 0xffff_ffff
    || !environment.binding?.buffer
    || (environment.binding.size !== undefined && environment.binding.size < SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES)
    || scene.environmentLightingRecord.byteLength !== SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES
    || !scene.environmentLightingCacheKey || scene.environmentLightingCacheKey !== environment.cacheKey) return false;
  const lightWords = scene.lightRecords;
  const lightIds = new Set<number>();
  for (let index = 0; index < lights.count; index += 1) {
    const identity = index * SVO_LIGHT_RECORD_WORDS + 24;
    const kind = lightWords[identity], lightId = lightWords[identity + 1], revision = lightWords[identity + 3];
    if (kind < 1 || kind > 4 || lightId === 0 || lightIds.has(lightId) || revision !== lights.revision) return false;
    lightIds.add(lightId);
  }
  const environmentWords = scene.environmentLightingRecord;
  return environmentWords[21] === environment.revision && environmentWords[22] === SVO_ENVIRONMENT_LIGHTING_VERSION;
}

/** Rebuild static CPU mirrors from canonical scene data; malformed publication metadata never throws into solver setup. */
export function buildSparseVoxelDrySceneLightingMirrors(
  scene: SceneDescription,
  source: SparseVoxelSceneRenderSource | undefined,
): Pick<SparseVoxelDrySceneData, "lightRecords" | "lightRevision" | "environmentLightingRecord" | "environmentLightingCacheKey"> | undefined {
  const lights = source?.lights, environment = source?.environmentLighting;
  if (!lights || !environment
    || lights.strideBytes !== SVO_LIGHT_RECORD_STRIDE_BYTES
    || !Number.isSafeInteger(lights.count) || lights.count < 1 || lights.count > SVO_LIGHT_MAXIMUM_RECORDS
    || !Number.isSafeInteger(lights.revision) || lights.revision < 1 || lights.revision > 0xffff_ffff
    || environment.count !== 1 || environment.strideBytes !== SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES
    || !Number.isSafeInteger(environment.revision) || environment.revision < 1 || environment.revision > 0xffff_ffff) return undefined;
  try {
    const sceneLights = buildSvoSceneLights(scene, { revision: lights.revision, maximumRecords: lights.count });
    const environmentLighting = buildSvoEnvironmentLighting(scene.environment ?? "default", environment.revision);
    if (sceneLights.records.length !== lights.count || environmentLighting.cacheKey !== environment.cacheKey) return undefined;
    return {
      lightRecords: sceneLights.packedRecords,
      lightRevision: sceneLights.revision,
      environmentLightingRecord: environmentLighting.packedRecord,
      environmentLightingCacheKey: environmentLighting.cacheKey,
    };
  } catch {
    return undefined;
  }
}

/** Pack validated CPU mirrors into one uniform arena, preserving the ten-storage-buffer ceiling. */
export function packSparseVoxelDrySceneLightingArena(
  source: SparseVoxelSceneRenderSource | undefined,
  scene: SparseVoxelDrySceneData | undefined,
): Uint32Array<ArrayBuffer> | undefined {
  if (!canConsumeSparseVoxelLighting(source, scene)) return undefined;
  if (!source?.lights && !source?.environmentLighting) {
    const packed = new Uint32Array(new ArrayBuffer(SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.sizeBytes));
    const floats = new Float32Array(packed.buffer);
    packed.set([1, 1, 1, SVO_ENVIRONMENT_LIGHTING_VERSION], SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.metadataWordOffset);
    const lightOffset = SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.lightWordOffset;
    const direction = scene?.lightDirection ?? [-0.45, 0.86, 0.28];
    const directionLength = Math.hypot(...direction) || 1;
    floats.set([direction[0] / directionLength, direction[1] / directionLength, direction[2] / directionLength, 0], lightOffset + 4);
    floats.set([...(scene?.lightColor ?? [1.04, 1, 0.91]), 1], lightOffset + 8);
    floats.set([1, 0, 0, 0], lightOffset + 12);
    floats.set([0, 0, 1, 0], lightOffset + 16);
    packed.set([1, 1, 0xffff_ffff, 1], lightOffset + 24);
    packed.set(buildSvoEnvironmentLighting("default", 1).packedRecord, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.environmentWordOffset);
    return packed;
  }
  const lights = source!.lights!, environment = source!.environmentLighting!;
  const packed = new Uint32Array(new ArrayBuffer(SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.sizeBytes));
  packed.set([lights.count, lights.revision, environment.revision, SVO_ENVIRONMENT_LIGHTING_VERSION], SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.metadataWordOffset);
  packed.set(scene!.lightRecords!, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.lightWordOffset);
  packed.set(scene!.environmentLightingRecord!, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.environmentWordOffset);
  return packed;
}

/** Validate the conservative static candidate publication independently for typed fallback reporting. */
export function canConsumeSparseVoxelPrimitiveCandidates(scene: SparseVoxelDrySceneData | undefined): boolean {
  const primitiveCount = scene?.primitiveRecords.byteLength
    ? scene.primitiveRecords.byteLength / SVO_PRIMITIVE_RECORD_STRIDE_BYTES
    : 0;
  let candidatesValid = primitiveCount > SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES;
  if (!candidatesValid && scene?.primitiveCandidates) {
    try {
      packSvoPrimitiveCandidateArena(scene.primitiveRecords, scene.primitiveCandidates);
      candidatesValid = true;
    } catch {
      candidatesValid = false;
    }
  }
  return candidatesValid;
}

export function canEncodeSparseVoxelDryScene(
  source: SparseVoxelSceneRenderSource | undefined,
  scene: SparseVoxelDrySceneData | undefined
): boolean {
  return Boolean(
    source?.structural
    && scene
    && canConsumeSparseVoxelPbrMaterials(source)
    && canConsumeSparseVoxelLighting(source, scene)
    && scene.primitiveRecords.byteLength >= SVO_PRIMITIVE_RECORD_STRIDE_BYTES
    && scene.primitiveRecords.byteLength % SVO_PRIMITIVE_RECORD_STRIDE_BYTES === 0
    && canConsumeSparseVoxelPrimitiveCandidates(scene)
    && (scene.glassRecords?.byteLength ?? 0) % SVO_THIN_GLASS_RECORD_STRIDE_BYTES === 0
    && (scene.glassRecords?.byteLength ?? 0) / SVO_THIN_GLASS_RECORD_STRIDE_BYTES <= SVO_SCENE_GLASS_MAXIMUM_PANES
    && (scene.terrainMaterialMetadata === undefined || scene.terrainMaterialMetadata.byteLength === SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES)
    && source.structural.fields.topology.residency !== "unavailable"
    && source.structural.fields.staticGeometry.residency !== "unavailable"
    && source.structural.fields.materialOwner.residency !== "unavailable"
    && (svoFluidPrimaryModeWord(scene.fluidPrimaryMode, scene.directFluidMediaEndToEndValidated) === 0
      || canConsumeSparseVoxelCoarseFluidPrimary(source))
  );
}

/** Validate the renderer-owned combined publication/fine arena before binding it. */
export function canConsumeSvoFineFluidCapability(capability: SvoFineFluidGpuCapability | undefined): capability is SvoFineFluidGpuCapability {
  const size = capability?.arena.size ?? capability?.arena.buffer?.size ?? 0;
  return Boolean(capability
    && capability.coarseFallbackRequired === true
    && capability.directWaterOwnership === false
    && capability.publicationMirrorWords === 8
    && capability.statusWord === 16
    && capability.acceptedStructuralGenerationWord === 19
    && capability.acceptedFineGenerationWord === 20
    && capability.paramsWords.length === 40
    && capability.pageGenerationOffsetWords >= 64
    && capability.ownerPageTableOffsetWords > capability.pageGenerationOffsetWords
    && capability.payloadOffsetWords > capability.ownerPageTableOffsetWords
    && size >= capability.payloadOffsetWords * 4);
}

const drySceneShader = /* wgsl */ `
${svoTerrainMaterialWGSL}
${svoMaterialWGSL}
${svoProceduralMaterialWGSL}
${svoThickGlassWGSL}
${svoGBufferWGSL}
${svoPrimitiveMotionWGSL}
${svoLightWGSL}
${svoEnvironmentLightingWGSL}
${svoFinePhiTypesWGSL}
${svoNodeMipSamplingWGSL}
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f, terrainMeta:vec4f, terrainFeatures:array<vec4f,16> }
struct BodyGPU { positionRadius:vec4f, halfSizeShape:vec4f, orientation:vec4f, colorSelected:vec4f }
struct DryParams {
  mapping:SvoMapping,
  metadata:vec4u,
  lightDirection:vec4f,
  lightColor:vec4f,
  // x: terrain material ID; y: pane count; zw: post-compositor-owned pane ID range.
  terrain:vec4u,
  terrainMaterial:SvoTerrainMaterialMetadata,
  // x: dense slot count; y: table revision; z: 96-byte stride; w: bounded contact-visibility gate.
  materialPublication:vec4u,
  // xyz: structural finest-cell dimensions; w: 0 legacy, 1 opaque diagnostic, 2 direct media.
  fluidDomainMode:vec4u,
  // x: record offset after primitives; y: node count; z: root; w: candidate ABI version.
  candidatePublication:vec4u,
  finePhi:SvoFinePhiParams,
  // x: complete static generation; y: directory pages; z: levels; w: available.
  nodeMip:vec4u,
  nodeMipAtlas:vec4u,
}
struct DryLightingArena {
  // x: light count; y: light revision; z: environment revision; w: environment ABI version.
  metadata:vec4u,
  lights:array<SvoLightRecord,${SVO_LIGHT_MAXIMUM_RECORDS}>,
  environment:SvoEnvironmentLightingRecord,
}
struct DryThickGlassArena {
  // x: count; y: revision; z: replaced thin-pane ID; w: binder ABI version.
  metadata:vec4u,
  records:array<SvoThickGlassRecord,${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}>,
}
struct DryHit {
  t:f32,
  normal:vec3f,
  materialId:u32,
  ownerId:u32,
  featureId:u32,
  fieldSource:u32,
  motionKind:u32,
  motionValid:u32,
  selected:f32,
  _padding:vec3u,
}

@group(0) @binding(0) var<uniform> uniforms:Uniforms;
@group(0) @binding(1) var<uniform> bodies:array<BodyGPU,12>;
@group(0) @binding(5) var<storage,read> materialOwners:array<u32>;
@group(0) @binding(6) var<storage,read> materials:array<SvoMaterialRecord>;
@group(0) @binding(7) var<storage,read> primitives:array<SvoPrimitiveRecord>;
@group(0) @binding(8) var<storage,read> publicationState:array<u32>;
@group(0) @binding(9) var<uniform> dry:DryParams;
@group(0) @binding(10) var<storage,read> glassPanes:array<SvoThinGlassRecord>;
@group(0) @binding(11) var<storage,read> svoStructuralGeometry:array<vec4f>;
@group(0) @binding(12) var<storage,read> svoStructuralLeafStates:array<u32>;
@group(0) @binding(13) var<uniform> dryLighting:DryLightingArena;
@group(0) @binding(14) var<uniform> rigidMotion:array<SvoPrimitiveMotionRecord,12>;
@group(0) @binding(15) var<uniform> thickGlass:DryThickGlassArena;
@group(0) @binding(16) var nodeMipAtlas:texture_3d<f32>;
@group(0) @binding(17) var nodeMipSampler:sampler;
@group(0) @binding(18) var nodeMipDirectory:texture_2d<u32>;

${createWebgpuSvoTraversalWGSL({ control: 2, nodes: 3, leaves: 4 })}
fn dryStructuralNodeWordLength()->u32{return arrayLength(&svoNodes)*8u;}
fn dryStructuralNodeWord(index:u32)->u32{let record=svoNodes[index/8u];let lane=index%8u;if(lane<4u){return record.address[lane];}return record.links[lane-4u];}
fn dryStructuralLeafWordLength()->u32{return arrayLength(&svoLeaves)*4u;}
fn dryStructuralLeafWord(index:u32)->u32{return svoLeaves[index/4u].topology[index%4u];}
${svoFinePhiPackedSamplingWGSL("publicationState", "dry.finePhi")}
fn dryFineFluidGridPosition(position_m:vec3f)->vec3f{return (position_m-dry.mapping.worldOrigin)/dry.finePhi.fineCell.xyz-vec3f(0.5);}
fn dryFineFluidSample(position_m:vec3f,expectedStructural:u32)->SvoFinePhiSample{if(dry.finePhi.counts.x==0u){return SvoFinePhiSample(0.0,0u);}return svoFinePackedPhi(dryFineFluidGridPosition(position_m),expectedStructural);}
fn dryFineFluidGradient(position_m:vec3f,expectedStructural:u32)->SvoFinePhiGradient{if(dry.finePhi.counts.x==0u){return SvoFinePhiGradient(vec3f(0.0),0u);}return svoFinePackedGradient(dryFineFluidGridPosition(position_m),expectedStructural);}
struct DryNodeMipLookup{sample:SvoNodeMipSample,valid:u32}
fn dryNodeMipMorton(coordinate:vec3u)->vec2u{
  var result=vec2u(0u);for(var bit=0u;bit<21u;bit+=1u){for(var axis=0u;axis<3u;axis+=1u){let outputBit=bit*3u+axis;let value=(coordinate[axis]>>bit)&1u;if(outputBit<32u){result.x|=value<<outputBit;}else{result.y|=value<<(outputBit-32u);}}}return result;
}
fn dryNodeMipCompare(entry:SvoNodeMipDirectoryEntry,level:u32,morton:vec2u)->i32{
  if(entry.level<level){return -1;}if(entry.level>level){return 1;}if(entry.mortonHigh<morton.y){return -1;}if(entry.mortonHigh>morton.y){return 1;}if(entry.mortonLow<morton.x){return -1;}if(entry.mortonLow>morton.x){return 1;}return 0;
}
fn dryNodeMipFind(level:u32,coordinate:vec3u)->u32{
  if(level>=dry.nodeMip.z||dry.nodeMip.y==0u){return 0xffffffffu;}let morton=dryNodeMipMorton(coordinate);var low=0u;var high=dry.nodeMip.y;
  for(var iteration=0u;iteration<24u&&low<high;iteration+=1u){let middle=low+(high-low)/2u;let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,middle);let comparison=dryNodeMipCompare(entry,level,morton);if(comparison<0){low=middle+1u;}else{high=middle;}}
  if(low>=dry.nodeMip.y){return 0xffffffffu;}let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,low);if(entry.generation!=dry.nodeMip.x||dryNodeMipCompare(entry,level,morton)!=0){return 0xffffffffu;}return low;
}
fn dryNodeMipReady()->bool{return dry.nodeMip.w!=0u&&dry.nodeMip.x!=0u&&dry.nodeMip.x==publicationState[2]&&dry.nodeMip.y>0u&&dry.nodeMip.z>0u;}
fn dryNodeMipAt(position_m:vec3f,lodIn:f32)->DryNodeMipLookup{
  let level=min(u32(max(floor(lodIn),0.0)),dry.nodeMip.z-1u);let levelScale=exp2(f32(level));let virtualVoxel=(position_m-dry.mapping.worldOrigin)/(dry.mapping.cellSize*levelScale);let pageFloor=floor(virtualVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);}let pageCoordinate=vec3u(pageFloor);let pageIndex=dryNodeMipFind(level,pageCoordinate);
  if(pageIndex==0xffffffffu){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);}let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,pageIndex);let local=virtualVoxel-vec3f(pageCoordinate)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);return DryNodeMipLookup(svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,entry.pageOrigin,local),1u);
}
struct DryConeVisibility{transmittance:f32,valid:u32}
fn dryConeVisibility(origin_m:vec3f,direction:vec3f,aperture:f32,maximumDistance_m:f32)->DryConeVisibility{
  if(!dryNodeMipReady()){return DryConeVisibility(1.0,0u);}let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let tangent=tan(aperture*.5);var distance=minimumVoxel*.75;var transmittance=1.0;
  for(var stepIndex=0u;stepIndex<48u&&distance<maximumDistance_m&&transmittance>.005;stepIndex+=1u){let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);let voxelWidth=minimumVoxel*exp2(floor(lod));let stepWidth=min(max(voxelWidth,diameter*.5),maximumDistance_m-distance);let lookup=dryNodeMipAt(origin_m+direction*distance,lod);if(lookup.valid==0u){return DryConeVisibility(1.0,0u);}let conservativeCoverage=max(lookup.sample.solidMean,lookup.sample.solidMaximum*.15);let alpha=svoNodeMipCoverageOpacity(conservativeCoverage,stepWidth/voxelWidth);transmittance*=1.0-alpha;distance+=max(stepWidth,minimumVoxel*.25);}
  return DryConeVisibility(clamp(transmittance,0.0,1.0),1u);
}
fn dryDiagnosticControl()->u32{return u32(round(max(uniforms.options.x,0.0)));}
fn dryDiagnosticMaximumNodeVisits()->u32{return clamp(dryDiagnosticControl()&511u,1u,256u);}
fn dryDiagnosticMaximumDepth()->u32{return clamp(dryDiagnosticControl()>>9u,1u,21u);}
fn dryTraverse(ray:SvoRay,mapping:SvoMapping)->SvoTraversalHit{return svoTraverseWithDepthLimit(ray,mapping,dryDiagnosticMaximumDepth());}
${createSvoStructuralFluidPrimaryWGSL({
  control: "svoControl",
  geometry: "svoStructuralGeometry",
  leafStates: "svoStructuralLeafStates",
  publication: "publicationState",
  nodeWordFunction: "dryStructuralNodeWord",
  nodeWordLengthFunction: "dryStructuralNodeWordLength",
  leafWordFunction: "dryStructuralLeafWord",
  leafWordLengthFunction: "dryStructuralLeafWordLength",
  domainFunction: "dryStructuralFluidDomain",
  domainFunctionBody: "return SvoStructuralSamplingDomain(vec4f(dry.mapping.worldOrigin,0.0),vec4f(dry.mapping.cellSize,0.0),vec4u(dry.fluidDomainMode.xyz,dry.mapping.brickSize),vec4u(dry.mapping.maximumDepth,svoStructuralFluidPrimaryExpectedGeneration,0u,0u));",
  fineSampleFunction: "dryFineFluidSample",
  fineGradientFunction: "dryFineFluidGradient",
  traversalFunction: "dryTraverse",
})}
${svoPrimitiveWGSL}
${unifiedLightingShaderLibrary}
${svoThinGlassWGSL}
${svoVisibilityPreludeWGSL}
${svoFluidMediaPreludeWGSL}

const DRY_MISS:f32 = 3.402823e38;
const REQUIRED_FIELDS:u32 = 67u; // topology | static geometry | material owner
const DRY_OWNER_NONE:u32=0xffffu;
const DRY_MEDIUM_GLASS:u32=2u;const DRY_MEDIUM_OPAQUE:u32=3u;
const DRY_GBUFFER_FIELD_ANALYTIC:u32=4u;const DRY_GBUFFER_FIELD_TERRAIN:u32=5u;
const DRY_GBUFFER_MOTION_STATIC:u32=0u;const DRY_GBUFFER_MOTION_RIGID:u32=1u;const DRY_GBUFFER_MOTION_FLUID:u32=2u;
const DRY_GBUFFER_HARD_FEATURE:u32=256u;const DRY_GBUFFER_NO_INTERSECTION:u32=1u;
const DRY_GBUFFER_WORK_EXHAUSTED:u32=2u;const DRY_GBUFFER_INVALID_FIELD:u32=3u;
const DRY_GBUFFER_SHADOW_DEFERRED:u32=8u;
const DRY_GBUFFER_FLUID_SURFACE:u32=128u;const DRY_GBUFFER_INSIDE_FLUID:u32=512u;
const DRY_REVERSED_Z_NEAR_M:f32=0.01;
var<private> dryFluidPrimaryFailure:u32;var<private> dryFluidInsideAtStart:u32;var<private> dryPrimitiveCandidateFailure:u32;var<private> dryVisibilityIgnoredOwner:u32;var<private> dryThickGlassEnabled:u32;var<private> dryThickGlassFailure:u32;var<private> dryShadowTracingEnabled:u32;
var<private> dryPrimaryNodeVisits:u32;var<private> dryPrimaryLeafVisits:u32;var<private> dryPrimaryMaximumDepth:u32;var<private> dryPrimaryFieldSteps:u32;var<private> dryCandidateWorkItems:u32;var<private> dryShadowNodeVisits:u32;var<private> dryShadowLeafVisits:u32;var<private> dryShadowWorkItems:u32;var<private> dryTraversalFailure:u32;
fn dryConfiguredMapping()->SvoMapping{
  var mapping=dry.mapping;
  mapping.maxVisits=min(mapping.maxVisits,dryDiagnosticMaximumNodeVisits());
  return mapping;
}
fn dryCostRamp(valueIn:f32)->vec3f{
  let value=clamp(valueIn,0.0,1.0);
  if(value<0.35){return mix(vec3f(.035,.08,.34),vec3f(.02,.64,.72),value/.35);}
  if(value<0.65){return mix(vec3f(.02,.64,.72),vec3f(.42,.82,.28),(value-.35)/.30);}
  if(value<0.84){return mix(vec3f(.42,.82,.28),vec3f(.97,.82,.16),(value-.65)/.19);}
  return mix(vec3f(.97,.82,.16),vec3f(.94,.08,.05),(value-.84)/.16);
}
fn dryCostOverlay(radianceDepth:vec4f)->vec4f{
  let mode=u32(round(max(uniforms.cameraPosition.w,0.0)));if(mode==0u){return radianceDepth;}
  let depthCost=f32(dryPrimaryMaximumDepth)/f32(dryDiagnosticMaximumDepth());
  let nodeCost=f32(dryPrimaryNodeVisits)/f32(dryDiagnosticMaximumNodeVisits());
  let fieldCost=f32(dryPrimaryFieldSteps)/f32(${SVO_STRUCTURAL_FLUID_PRIMARY_LIMITS.fieldSteps});
  let candidateCost=f32(dryCandidateWorkItems)/f32(${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES * 2});
  let shadowCost=f32(dryShadowNodeVisits+dryShadowLeafVisits+dryShadowWorkItems)/f32(${(SVO_VISIBILITY_LIMITS.nodeVisits + SVO_VISIBILITY_LIMITS.leafVisits + SVO_VISIBILITY_LIMITS.workItems) * SVO_DRY_SCENE_MAX_SHADED_LIGHTS});
  var cost=depthCost;if(mode==2u){cost=nodeCost;}else if(mode==3u){cost=fieldCost;}else if(mode==4u){cost=candidateCost;}else if(mode==5u){cost=shadowCost;}else if(mode==6u){cost=max(max(depthCost,nodeCost),max(max(fieldCost,candidateCost),shadowCost));}
  var overlayColor=dryCostRamp(cost);
  if(mode==7u){
    var failure=dryTraversalFailure;
    if(dryPrimitiveCandidateFailure==DRY_GBUFFER_WORK_EXHAUSTED||dryFluidPrimaryFailure==DRY_GBUFFER_WORK_EXHAUSTED){failure=max(failure,1u);}
    if(dryPrimitiveCandidateFailure==DRY_GBUFFER_INVALID_FIELD||dryFluidPrimaryFailure==DRY_GBUFFER_INVALID_FIELD){failure=2u;}
    overlayColor=select(select(vec3f(.04,.28,.19),vec3f(.98,.72,.12),failure==1u),vec3f(.95,.08,.05),failure>=2u);
  }
  return vec4f(mix(radianceDepth.rgb,overlayColor,clamp(uniforms.cameraTarget.w,0.0,1.0)),radianceDepth.a);
}
fn dryBoundThickGlassOwner(owner:u32)->bool{
  if(dryThickGlassEnabled==0u){return false;}let count=min(thickGlass.metadata.x,${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u);
  for(var index=0u;index<${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u;index+=1u){if(index>=count){break;}if(svoThickGlassOwnerId(thickGlass.records[index])==owner){return true;}}
  return false;
}
fn dryOpaqueOwnerSuppressed(owner:u32)->bool{return owner==dry.metadata.z||owner==dryVisibilityIgnoredOwner||dryBoundThickGlassOwner(owner);}

fn missHit()->DryHit { return DryHit(DRY_MISS,vec3f(0.0,1.0,0.0),0u,DRY_OWNER_NONE,SVO_FEATURE_SMOOTH,0u,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u)); }
fn dryFluidFieldSource(owner:u32)->u32{return select(SVO_GBUFFER_FIELD_FLUID_COARSE,SVO_GBUFFER_FIELD_FLUID_FINE,owner==SVO_FLUID_OWNER_FINE);}
fn dryIsFluidFieldSource(source:u32)->bool{return source==SVO_GBUFFER_FIELD_FLUID_COARSE||source==SVO_GBUFFER_FIELD_FLUID_FINE;}
fn qrotWxyz(q:vec4f,v:vec3f)->vec3f { let a=cross(q.yzw,v); return v+2.0*(q.x*a+cross(q.yzw,a)); }
fn qinvWxyz(q:vec4f,v:vec3f)->vec3f { return qrotWxyz(vec4f(q.x,-q.yzw),v); }

fn slabHit(ro:vec3f,rd:vec3f,extent:vec3f)->vec2f {
  let inverse=1.0/rd; let first=(-extent-ro)*inverse; let second=(extent-ro)*inverse;
  let near=min(first,second); let far=max(first,second);
  return vec2f(max(max(near.x,near.y),near.z),min(min(far.x,far.y),far.z));
}

struct DryBoundsInterval{nearT:f32,farT:f32,valid:u32}
fn dryBoundsInterval(minimum:vec3f,maximum:vec3f,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryBoundsInterval{
  var nearT=tMin;var farT=tMax;
  for(var axis=0u;axis<3u;axis+=1u){
    if(abs(rd[axis])<=1e-9){if(ro[axis]<minimum[axis]||ro[axis]>maximum[axis]){return DryBoundsInterval(nearT,farT,0u);}}
    else{let first=(minimum[axis]-ro[axis])/rd[axis];let second=(maximum[axis]-ro[axis])/rd[axis];nearT=max(nearT,min(first,second));farT=min(farT,max(first,second));if(nearT>farT){return DryBoundsInterval(nearT,farT,0u);}}
  }
  return DryBoundsInterval(nearT,farT,1u);
}

fn directionalLightSceneExitDistance(position:vec3f,directionToLightIn:vec3f)->f32 {
  // dryLightSample returns a unit direction for every valid light sample.
  let directionToLight=directionToLightIn;
  let minimum=vec3f(-0.5*uniforms.container.x,0.0,-0.5*uniforms.container.z);
  let maximum=vec3f(0.5*uniforms.container.x,uniforms.container.y,0.5*uniforms.container.z);
  var enter=0.0;var exit=DRY_MISS;
  for(var axis=0u;axis<3u;axis+=1u){
    if(abs(directionToLight[axis])<=1e-9){if(position[axis]<minimum[axis]||position[axis]>maximum[axis]){return 0.0;}}
    else{let first=(minimum[axis]-position[axis])/directionToLight[axis];let second=(maximum[axis]-position[axis])/directionToLight[axis];enter=max(enter,min(first,second));exit=min(exit,max(first,second));if(exit<enter){return 0.0;}}
  }
  return max(exit,0.0);
}

fn terrainEnabled()->bool{return uniforms.terrainMeta.x>0.5&&dry.terrain.x!=0xffffffffu;}
fn terrainHeightAt(x:f32,z:f32)->f32{
  if(!terrainEnabled()){return 0.0;}
  var mounds=0.0;var carvePower=0.0;let exponent=max(uniforms.terrainMeta.w,1.0);
  let count=min(i32(round(uniforms.terrainMeta.z)),8);
  for(var i=0;i<count;i+=1){
    let a=uniforms.terrainFeatures[2*i];let b=uniforms.terrainFeatures[2*i+1];
    let cs=cos(b.y);let sn=sin(b.y);let dx=x-a.x;let dz=z-a.y;
    let localX=(cs*dx+sn*dz)/a.z;let localZ=(-sn*dx+cs*dz)/a.w;
    let distance=length(vec2f(localX,localZ));var weight=0.0;
    if(distance<=b.z){weight=1.0;}
    else if(distance<1.0){let s=1.0-(distance-b.z)/(1.0-b.z);weight=s*s*(3.0-2.0*s);}
    if(b.x>=0.0){mounds+=b.x*weight;}else{carvePower+=pow(-b.x*weight,exponent);}
  }
  var carve=0.0;if(carvePower>0.0){carve=pow(carvePower,1.0/exponent);}
  return max(0.0,uniforms.terrainMeta.y+mounds-carve);
}
fn terrainCeiling()->f32{
  var top=uniforms.terrainMeta.y+0.05;let count=min(i32(round(uniforms.terrainMeta.z)),8);
  for(var i=0;i<count;i+=1){let amount=uniforms.terrainFeatures[2*i+1].x;if(amount>0.0){top+=amount;}}
  return top;
}
fn terrainNormalAt(point:vec2f)->vec3f{
  let epsilon=0.02;
  let dx=(terrainHeightAt(point.x+epsilon,point.y)-terrainHeightAt(point.x-epsilon,point.y))/(2.0*epsilon);
  let dz=(terrainHeightAt(point.x,point.y+epsilon)-terrainHeightAt(point.x,point.y-epsilon))/(2.0*epsilon);
  return normalize(vec3f(-dx,1.0,-dz));
}
fn terrainField(ro:vec3f,rd:vec3f,t:f32)->f32{let point=ro+rd*t;return point.y-terrainHeightAt(point.x,point.z);}
fn terrainHitAt(ro:vec3f,rd:vec3f,t:f32)->DryHit{let point=ro+rd*t;return DryHit(t,terrainNormalAt(point.xz),dry.terrain.x,DRY_OWNER_NONE,SVO_FEATURE_TERRAIN,DRY_GBUFFER_FIELD_TERRAIN,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));}

// Ordinary camera rays use at most 8 intersection height evaluations plus the
// four central-difference normal samples. Only unresolved shallow/grazing rays
// pay for the smaller graded fallback. Both paths return the first bracket.
fn traceTerrain(ro:vec3f,rd:vec3f)->DryHit{
  if(!terrainEnabled()){return missHit();}
  let sceneScale=max(max(uniforms.container.x,uniforms.container.y),uniforms.container.z);
  let ceiling=terrainCeiling();var t0=0.005;
  if(ro.y>ceiling){if(rd.y>=-0.0005){return missHit();}t0=(ceiling-ro.y)/rd.y;}
  var t1=t0+10.0*sceneScale;
  if(rd.y<-0.0005){t1=min(t1,(-0.02-ro.y)/rd.y);}
  else if(rd.y>0.0005){t1=min(t1,max(t0,(ceiling-ro.y)/rd.y));}
  if(t1<=t0){return missHit();}
  let initialField=terrainField(ro,rd,t0);
  if(abs(initialField)<=0.0001){return terrainHitAt(ro,rd,t0);}
  if(abs(rd.y)>=${SVO_TERRAIN_FAST_MIN_VERTICAL}){
    var previousFastT=t0;var previousFastField=initialField;
    for(var bracket=1;bracket<=${SVO_TERRAIN_FAST_BRACKET_STEPS};bracket+=1){
      let candidateT=t0+(t1-t0)*f32(bracket)/f32(${SVO_TERRAIN_FAST_BRACKET_STEPS});let candidateField=terrainField(ro,rd,candidateT);
      if(abs(candidateField)<=0.0001){return terrainHitAt(ro,rd,candidateT);}
      if((previousFastField<0.0)!=(candidateField<0.0)){
        var a=previousFastT;var b=candidateT;var fieldA=previousFastField;var fieldB=candidateField;
        var bestT=select(b,a,abs(fieldA)<abs(fieldB));var bestField=min(abs(fieldA),abs(fieldB));
        for(var refinement=0;refinement<${SVO_TERRAIN_FAST_REFINEMENTS};refinement+=1){
          let span=b-a;let secant=b-fieldB*span/(fieldB-fieldA);let t=clamp(secant,a+span*0.05,b-span*0.05);
          let field=terrainField(ro,rd,t);if(abs(field)<bestField){bestField=abs(field);bestT=t;}
          if(abs(field)<=0.0001){return terrainHitAt(ro,rd,t);}
          if((fieldA<0.0)==(field<0.0)){a=t;fieldA=field;}else{b=t;fieldB=field;}
        }
        if(bestField<=0.0001){return terrainHitAt(ro,rd,bestT);}
        break;
      }
      previousFastT=candidateT;previousFastField=candidateField;
    }
  }
  var previousT=t0;var previousField=initialField;var closestT=t0;var closestField=abs(initialField);
  for(var iteration=1;iteration<=${SVO_TERRAIN_FALLBACK_STEPS};iteration+=1){
    let t=t0+(t1-t0)*pow(f32(iteration)/f32(${SVO_TERRAIN_FALLBACK_STEPS}),1.4);let field=terrainField(ro,rd,t);
    if(abs(field)<closestField){closestField=abs(field);closestT=t;}
    if((previousField<0.0)!=(field<0.0)){
      var a=previousT;var b=t;var fieldA=previousField;
      for(var refinement=0;refinement<${SVO_TERRAIN_FALLBACK_REFINEMENTS};refinement+=1){let middle=0.5*(a+b);let middleField=terrainField(ro,rd,middle);if((fieldA<0.0)==(middleField<0.0)){a=middle;fieldA=middleField;}else{b=middle;}}
      return terrainHitAt(ro,rd,0.5*(a+b));
    }
    if(abs(field)<=0.0001){return terrainHitAt(ro,rd,t);}
    previousT=t;previousField=field;
  }
  if(closestField<=0.0005){return terrainHitAt(ro,rd,closestT);}
  return missHit();
}

fn bodyHit(ro:vec3f,rd:vec3f,body:BodyGPU)->DryHit {
  let localOrigin=qinvWxyz(body.orientation,ro-body.positionRadius.xyz);
  let localDirection=qinvWxyz(body.orientation,rd);
  let shape=i32(round(body.halfSizeShape.w)); var t=DRY_MISS; var normal=vec3f(0.0,1.0,0.0);var featureId=SVO_FEATURE_SMOOTH;
  if (shape==0) {
    let radius=body.halfSizeShape.x; let b=dot(localOrigin,localDirection); let discriminant=b*b-dot(localOrigin,localOrigin)+radius*radius;
    if (discriminant>=0.0) { let root=sqrt(discriminant); t=-b-root; if(t<=1e-4){t=-b+root;} if(t>1e-4){normal=normalize(localOrigin+localDirection*t);}else{t=DRY_MISS;} }
  } else if (shape==1) {
    let interval=slabHit(localOrigin,localDirection,body.halfSizeShape.xyz); t=select(interval.x,interval.y,interval.x<=1e-4);
    if(t>1e-4&&interval.x<=interval.y){let point=localOrigin+localDirection*t;let q=abs(point/max(body.halfSizeShape.xyz,vec3f(1e-6)));if(q.x>=q.y&&q.x>=q.z){normal=vec3f(sign(point.x),0,0);featureId=SVO_FEATURE_BOX_X;}else if(q.y>=q.z){normal=vec3f(0,sign(point.y),0);featureId=SVO_FEATURE_BOX_Y;}else{normal=vec3f(0,0,sign(point.z));featureId=SVO_FEATURE_BOX_Z;}}else{t=DRY_MISS;}
  } else {
    let radius=body.halfSizeShape.x; let halfHeight=body.halfSizeShape.y; let a=dot(localDirection.xz,localDirection.xz); let b=dot(localOrigin.xz,localDirection.xz); let c=dot(localOrigin.xz,localOrigin.xz)-radius*radius;
    if(a>1e-7&&b*b-a*c>=0.0){let candidate=(-b-sqrt(b*b-a*c))/a;let y=localOrigin.y+localDirection.y*candidate;if(candidate>1e-4&&abs(y)<=halfHeight){t=candidate;let p=localOrigin+localDirection*t;normal=normalize(vec3f(p.x,0,p.z));featureId=select(SVO_FEATURE_CYLINDER_SIDE,SVO_FEATURE_SMOOTH,shape==2);}}
    if(shape==2){for(var side=-1.0;side<=1.0;side+=2.0){let center=vec3f(0.0,side*halfHeight,0.0);let offset=localOrigin-center;let hb=dot(offset,localDirection);let disc=hb*hb-dot(offset,offset)+radius*radius;if(disc>=0.0){let candidate=-hb-sqrt(disc);if(candidate>1e-4&&candidate<t){t=candidate;normal=normalize(offset+localDirection*t);}}}}
    else if(abs(localDirection.y)>1e-7){for(var side=-1.0;side<=1.0;side+=2.0){let candidate=(side*halfHeight-localOrigin.y)/localDirection.y;let p=localOrigin+localDirection*candidate;if(candidate>1e-4&&candidate<t&&dot(p.xz,p.xz)<=radius*radius){t=candidate;normal=vec3f(0,side,0);featureId=SVO_FEATURE_CYLINDER_CAP;}}}
  }
  return DryHit(t,qrotWxyz(body.orientation,normal),0u,DRY_OWNER_NONE,featureId,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_RIGID,0u,body.colorSelected.w,vec3u(0u));
}

fn bodyCandidateVisible(ro:vec3f,rd:vec3f,body:BodyGPU,tMin:f32,tMax:f32)->bool{
  let localOrigin=qinvWxyz(body.orientation,ro-body.positionRadius.xyz);let localDirection=qinvWxyz(body.orientation,rd);let shape=i32(round(body.halfSizeShape.w));
  let radius=body.halfSizeShape.x;var extent=body.halfSizeShape.xyz;
  if(shape==0){extent=vec3f(radius);}else if(shape==2){extent=vec3f(radius,body.halfSizeShape.y+radius,radius);}else if(shape==3){extent=vec3f(radius,body.halfSizeShape.y,radius);}
  return dryBoundsInterval(-extent,extent,localOrigin,localDirection,tMin,tMax).valid!=0u;
}

fn bodyBoundingSphereVisible(ro:vec3f,rd:vec3f,body:BodyGPU,tMin:f32,tMax:f32)->bool{
  let offset=body.positionRadius.xyz-ro;let projected=clamp(dot(offset,rd),tMin,tMax);let closest=ro+rd*projected;let radius=max(body.positionRadius.w,0.0)+1e-5;
  return dot(closest-body.positionRadius.xyz,closest-body.positionRadius.xyz)<=radius*radius;
}

fn nearestBodyIgnoring(ro:vec3f,rd:vec3f,ignoredOwner:u32)->DryHit {
  var best=missHit(); for(var index=0u;index<12u;index+=1u){if(index>=u32(round(uniforms.options.z))){break;}if(index==ignoredOwner){continue;}let body=bodies[index];if(!bodyBoundingSphereVisible(ro,rd,body,0.0,best.t)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ro,rd,body,0.0,best.t)){continue;}let hit=bodyHit(ro,rd,body);if(hit.t<best.t){best=hit;best.materialId=0x80000000u|index;best.ownerId=index;}} return best;
}
fn nearestBody(ro:vec3f,rd:vec3f)->DryHit{return nearestBodyIgnoring(ro,rd,DRY_OWNER_NONE);}

fn primitiveHit(record:SvoPrimitiveRecord,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryHit {
  // Use the shared analytic ray contract directly. In particular, do not call
  // the bounded closest-point distance evaluator merely to recover an
  // ellipsoid normal after the ray quadratic has already found the surface.
  let exact=svoIntersectPrimitiveExact(record,ro,rd,max(tMin,1e-4),tMax);
  if(exact.status!=SVO_PRIMITIVE_RAY_HIT){return missHit();}
  return DryHit(exact.t_m,exact.normal.xyz,svoPrimitiveMaterialId(record),svoPrimitiveOwnerId(record),exact.featureId,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
}

const DRY_CANDIDATE_COMPLETE:u32=0u;const DRY_CANDIDATE_INVALID:u32=1u;const DRY_CANDIDATE_EXHAUSTED:u32=2u;
struct DryCandidateTrace{hit:DryHit,primitiveIndex:u32,status:u32,workItems:u32}
fn dryCandidateNode(nodeIndex:u32)->SvoPrimitiveRecord{return primitives[dry.candidatePublication.x+nodeIndex];}
fn dryCandidateInterval(node:SvoPrimitiveRecord,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryBoundsInterval{
  return dryBoundsInterval(bitcast<vec3f>(node.centerKind.xyz),bitcast<vec3f>(node.dimensionsIdentity.xyz),ro,rd,tMin,tMax);
}
fn tracePrimitiveCandidates(ro:vec3f,rd:vec3f,tMin:f32,tMax:f32,workLimit:u32,opaqueAnyHit:bool)->DryCandidateTrace{
  var best=missHit();best.t=tMax;var bestIndex=0xffffffffu;var workItems=0u;
  let recordOffset=dry.candidatePublication.x;let nodeCount=dry.candidatePublication.y;let root=dry.candidatePublication.z;
  if(dry.candidatePublication.w!=${SVO_PRIMITIVE_CANDIDATE_VERSION}u||recordOffset!=dry.metadata.x||nodeCount==0u||nodeCount>${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES}u||root>=nodeCount||recordOffset+nodeCount>arrayLength(&primitives)){
    return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_INVALID,workItems);
  }
  if(nodeCount==1u){
    if(workLimit<2u){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_EXHAUSTED,workItems);}let node=dryCandidateNode(root);workItems+=1u;let primitiveIndex=node.centerKind.w;
    if(node.dimensionsIdentity.w!=${SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL}u||primitiveIndex>=recordOffset){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_INVALID,workItems);}
    if(dryCandidateInterval(node,ro,rd,tMin,tMax).valid==0u){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_COMPLETE,workItems);}workItems+=1u;let record=primitives[primitiveIndex];
    let owner=svoPrimitiveOwnerId(record);if(dryOpaqueOwnerSuppressed(owner)){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_COMPLETE,workItems);}let hit=primitiveHit(record,ro,rd,tMin,tMax);return DryCandidateTrace(hit,select(0xffffffffu,primitiveIndex,hit.t<DRY_MISS),DRY_CANDIDATE_COMPLETE,workItems);
  }
  var stack:array<u32,${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK}>;var stackCount=1u;stack[0]=root;
  for(var visit=0u;visit<${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES}u;visit+=1u){
    if(stackCount==0u){break;}if(workItems>=workLimit){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_EXHAUSTED,workItems);}workItems+=1u;
    stackCount-=1u;let nodeIndex=stack[stackCount];if(nodeIndex>=nodeCount){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_INVALID,workItems);}let node=dryCandidateNode(nodeIndex);
    if(dryCandidateInterval(node,ro,rd,tMin,best.t).valid==0u){continue;}
    let leftOrPrimitive=node.centerKind.w;let rightChild=node.dimensionsIdentity.w;
    if(rightChild==${SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL}u){
      if(leftOrPrimitive>=dry.metadata.x||leftOrPrimitive>=recordOffset||workItems>=workLimit){return DryCandidateTrace(best,bestIndex,select(DRY_CANDIDATE_EXHAUSTED,DRY_CANDIDATE_INVALID,leftOrPrimitive>=dry.metadata.x||leftOrPrimitive>=recordOffset),workItems);}workItems+=1u;
      let record=primitives[leftOrPrimitive];let owner=svoPrimitiveOwnerId(record);if(dryOpaqueOwnerSuppressed(owner)){continue;}let candidate=primitiveHit(record,ro,rd,tMin,best.t);if(opaqueAnyHit&&candidate.t<DRY_MISS&&candidate.t>=tMin){return DryCandidateTrace(candidate,leftOrPrimitive,DRY_CANDIDATE_COMPLETE,workItems);}let tolerance=1e-6*max(1.0,max(candidate.t,best.t));
      if(candidate.t<DRY_MISS&&candidate.t>=tMin&&(bestIndex==0xffffffffu||candidate.t<best.t-tolerance||(abs(candidate.t-best.t)<=tolerance&&leftOrPrimitive<bestIndex))){best=candidate;bestIndex=leftOrPrimitive;}
      continue;
    }
    if(leftOrPrimitive>=nodeCount||rightChild>=nodeCount){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_INVALID,workItems);}
    let leftInterval=dryCandidateInterval(dryCandidateNode(leftOrPrimitive),ro,rd,tMin,best.t);let rightInterval=dryCandidateInterval(dryCandidateNode(rightChild),ro,rd,tMin,best.t);
    let childCount=select(0u,1u,leftInterval.valid!=0u)+select(0u,1u,rightInterval.valid!=0u);if(stackCount+childCount>${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK}u){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_EXHAUSTED,workItems);}
    if(leftInterval.valid!=0u&&rightInterval.valid!=0u){if(leftInterval.nearT<=rightInterval.nearT){stack[stackCount]=rightChild;stack[stackCount+1u]=leftOrPrimitive;}else{stack[stackCount]=leftOrPrimitive;stack[stackCount+1u]=rightChild;}stackCount+=2u;}
    else if(leftInterval.valid!=0u){stack[stackCount]=leftOrPrimitive;stackCount+=1u;}else if(rightInterval.valid!=0u){stack[stackCount]=rightChild;stackCount+=1u;}
  }
  if(stackCount!=0u){return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_EXHAUSTED,workItems);}return DryCandidateTrace(best,bestIndex,DRY_CANDIDATE_COMPLETE,workItems);
}

fn traceLeafPayload(ro:vec3f,rd:vec3f,hit:SvoTraversalHit)->DryHit {
  let bounds=svoNodeBounds(svoNodes[hit.nodeIndex],dry.mapping); let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  var entry=max(hit.tEnter,0.0); let point=ro+rd*(entry+1e-5); var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0)); let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9)); let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));
  let tolerance=length(extent)*1.05;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(i32(dry.mapping.brickSize)))||entry>hit.tExit){break;}
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex<arrayLength(&materialOwners)){
      let identity=materialOwners[payloadIndex];let owner=identity>>16u;
      if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){let primitiveIndex=owner-dry.metadata.y;if(primitiveIndex<dry.metadata.x&&primitiveIndex<arrayLength(&primitives)){let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,hit.tExit));let candidate=primitiveHit(primitives[primitiveIndex],ro,rd,max(0.0,entry-tolerance),cellExit+tolerance);if(candidate.t<DRY_MISS){return candidate;}}}
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z)); if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return missHit();
}

fn traceStatic(ro:vec3f,rd:vec3f)->DryHit {
  if(publicationState[0]==0u||(publicationState[1]&REQUIRED_FIELDS)!=REQUIRED_FIELDS){return missHit();}
  // Small authored catalogs use the producer-published conservative BVH in
  // the same binding. Exact intersections occur only at retained leaves.
  if(dry.metadata.x<=${SVO_DRY_SCENE_DIRECT_PRIMITIVE_LIMIT}u){
    let candidate=tracePrimitiveCandidates(ro,rd,0.0,DRY_MISS,${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES * 2}u,false);dryCandidateWorkItems+=candidate.workItems;if(candidate.status!=DRY_CANDIDATE_COMPLETE){dryPrimitiveCandidateFailure=select(DRY_GBUFFER_INVALID_FIELD,DRY_GBUFFER_WORK_EXHAUSTED,candidate.status==DRY_CANDIDATE_EXHAUSTED);}return candidate.hit;
  }
  var minimum=0.0;let mapping=dryConfiguredMapping();for(var leafVisit=0u;leafVisit<48u;leafVisit+=1u){let ray=SvoRay(ro,minimum,rd,DRY_MISS);let leaf=dryTraverse(ray,mapping);dryPrimaryNodeVisits+=leaf.visits;if(leaf.status!=SVO_STATUS_HIT){if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){dryTraversalFailure=max(dryTraversalFailure,1u);}else if(leaf.status!=SVO_STATUS_MISS){dryTraversalFailure=2u;}break;}dryPrimaryLeafVisits+=1u;dryPrimaryMaximumDepth=max(dryPrimaryMaximumDepth,leaf.level);let payloadHit=traceLeafPayload(ro,rd,leaf);if(payloadHit.t<DRY_MISS){return payloadHit;}minimum=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);} return missHit();
}

struct DryGlassHit{hit:SvoThinGlassHit,recordIndex:u32}
fn dryGlassMiss()->DryGlassHit{return DryGlassHit(svoThinGlassMiss(),0u);}
fn dryGlassBoundingSphereVisible(record:SvoThinGlassRecord,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->bool{
  let offset=record.centerThickness.xyz-ro;let projected=clamp(dot(offset,rd),tMin,tMax);let closest=ro+rd*projected;let radius=length(vec3f(record.extentIorEpsilon.xy,.5*record.centerThickness.w))+record.extentIorEpsilon.w+1e-5;
  return dot(closest-record.centerThickness.xyz,closest-record.centerThickness.xyz)<=radius*radius;
}
fn traceGlass(ro:vec3f,rd:vec3f,tMin_m:f32,tMax_m:f32,skipCompositeOwned:bool)->DryGlassHit {
  var best=dryGlassMiss();var bestT=tMax_m;
  let paneCount=min(dry.terrain.y,min(arrayLength(&glassPanes),${SVO_SCENE_GLASS_MAXIMUM_PANES}u));
  for(var paneIndex=0u;paneIndex<${SVO_SCENE_GLASS_MAXIMUM_PANES}u;paneIndex+=1u){
    if(paneIndex>=paneCount){break;}let record=glassPanes[paneIndex];let paneId=svoThinGlassPaneId(record);let compositeOwned=skipCompositeOwned&&dry.terrain.w>0u&&paneId>=dry.terrain.z&&paneId-dry.terrain.z<dry.terrain.w;let thickReplaced=dryThickGlassEnabled!=0u&&paneId==thickGlass.metadata.z;if(compositeOwned||thickReplaced||!dryGlassBoundingSphereVisible(record,ro,rd,tMin_m,bestT)){continue;}let candidate=svoThinGlassIntersect(record,ro,rd,tMin_m,bestT,1e-6,record.extentIorEpsilon.w);
    if(candidate.valid!=0u&&candidate.t_m<bestT){best=DryGlassHit(candidate,paneIndex);bestT=candidate.t_m;}
  }
  return best;
}

struct DryThickGlassHit{interval:SvoThickGlassInterval,recordIndex:u32}
fn dryThickGlassMiss()->DryThickGlassHit{return DryThickGlassHit(svoThickGlassEmpty(SVO_THICK_GLASS_MISS),0u);}
fn dryThickGlassFirst(interval:SvoThickGlassInterval)->SvoThickGlassSurface{var first=interval.exit;if(interval.hasEntry!=0u){first=interval.entry;}return first;}
fn traceThickGlass(ro:vec3f,rd:vec3f,tMin_m:f32,tMax_m:f32)->DryThickGlassHit{
  var best=dryThickGlassMiss();var bestT=tMax_m;if(dryThickGlassEnabled==0u){return best;}
  let count=min(thickGlass.metadata.x,${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u);
  for(var recordIndex=0u;recordIndex<${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u;recordIndex+=1u){
    if(recordIndex>=count){break;}let candidate=svoThickGlassIntersect(thickGlass.records[recordIndex],ro,rd,tMin_m,bestT,thickGlass.metadata.y);
    if(candidate.status==SVO_THICK_GLASS_INVALID||candidate.status==SVO_THICK_GLASS_STALE){dryThickGlassFailure=candidate.status;return dryThickGlassMiss();}
    if(candidate.status==SVO_THICK_GLASS_HIT){let first=dryThickGlassFirst(candidate);if(first.t_m<bestT){best=DryThickGlassHit(candidate,recordIndex);bestT=first.t_m;}}
  }
  return best;
}

fn dryVisibilityStep(status:u32,nodeVisits:u32,leafVisits:u32,workItems:u32,t:f32)->SvoVisibilityStep {
  return SvoVisibilityStep(status,nodeVisits,leafVisits,workItems,t,1u,vec3f(0.0),0u);
}
fn dryVisibilityTransmissionStep(nodeVisits:u32,leafVisits:u32,workItems:u32,t:f32,transmittance:vec3f)->SvoVisibilityStep {
  return SvoVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,t,0u,clamp(transmittance,vec3f(0.0),vec3f(1.0)),0u);
}

// Renderer-local unit-vector variant of the shared bias contract. Surface
// normals and light/contact directions are normalized at their construction
// sites, so repeating both inverse-square-roots per visibility ray is waste.
fn dryBiasedVisibilityRayUnit(surfacePosition_m:vec3f,geometricNormal:vec3f,directionToLight:vec3f,maximumLightDistance_m:f32,cellSize_m:vec3f,biasCells:f32)->SvoVisibilityRay {
  let projectedCellWidth=dot(abs(geometricNormal),cellSize_m);let originBias_m=max(biasCells,0.0)*projectedCellWidth;
  let side=select(-1.0,1.0,dot(geometricNormal,directionToLight)>=0.0);let offset=side*geometricNormal*originBias_m;
  return SvoVisibilityRay(surfacePosition_m+offset,max(0.0,maximumLightDistance_m-dot(offset,directionToLight)),directionToLight,originBias_m);
}

// Shadow payload lookup mirrors the production leaf DDA, but reports invalid
// data and bounded-work exhaustion explicitly so direct light fails closed.
fn traceLeafPayloadVisibility(ray:SvoVisibilityRay,tMin_m:f32,hit:SvoTraversalHit,workLimit:u32)->SvoVisibilityStep {
  let bounds=svoNodeBounds(svoNodes[hit.nodeIndex],dry.mapping);let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  var entry=max(max(hit.tEnter,tMin_m),0.0);let point=ray.origin_m+ray.direction*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));
  let step=select(vec3i(-1),vec3i(1),ray.direction>=vec3f(0.0));let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));let deltaT=select(vec3f(DRY_MISS),abs(extent/ray.direction),abs(ray.direction)>vec3f(1e-9));
  let tolerance=length(extent)*1.05;var workItems=0u;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(i32(dry.mapping.brickSize)))||entry>hit.tExit||entry>ray.tMax_m){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);}
    if(workItems>=workLimit){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);}workItems+=1u;
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex>=arrayLength(&materialOwners)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}
    let identity=materialOwners[payloadIndex];let owner=identity>>16u;
    if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){
      let primitiveIndex=owner-dry.metadata.y;
      if(primitiveIndex>=dry.metadata.x||primitiveIndex>=arrayLength(&primitives)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}
      let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,min(hit.tExit,ray.tMax_m)));let candidate=primitiveHit(primitives[primitiveIndex],ray.origin_m,ray.direction,max(entry-tolerance,tMin_m),cellExit+tolerance);
      if(candidate.t<DRY_MISS){return dryVisibilityStep(SVO_VIS_STEP_HIT,0u,0u,workItems,candidate.t);}
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);
}

const DRY_MEDIUM_AIR:u32=0u;const DRY_MEDIUM_WATER:u32=1u;
// Adapter point for M9's atomic water-exit/thin-wall coincidence resolver. The
// dry target is air today; direct fluid integration will pass WATER only after
// coincident volume exits have been resolved by the shared media contract.
fn dryThinGlassIncidentIor(medium:u32)->f32{return select(1.0,${WATER_OPTICS.indexOfRefraction},medium==DRY_MEDIUM_WATER);}

// Adapter required by svoTraceVisibility. It returns the nearest opaque or
// transmissive candidate and never calls the lighting closure recursively.
fn svoVisibilityNext(ray:SvoVisibilityRay,tMin_m:f32,remaining:SvoVisibilityBudget)->SvoVisibilityStep {
  if(arrayLength(&publicationState)<2u||publicationState[0]==0u||(publicationState[1]&REQUIRED_FIELDS)!=REQUIRED_FIELDS){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,0u,DRY_MISS);}
  if(dry.metadata.x>arrayLength(&primitives)||dry.terrain.y>arrayLength(&glassPanes)||dry.terrain.y>${SVO_SCENE_GLASS_MAXIMUM_PANES}u){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,0u,DRY_MISS);}
  var nodeVisits=0u;var leafVisits=0u;var workItems=0u;var bestT=ray.tMax_m;var found=false;var opaque=true;var glassTransmission=vec3f(0.0);

  let bodyCount=min(u32(round(max(uniforms.options.z,0.0))),12u);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}if(bodyIndex==dryVisibilityIgnoredOwner){continue;}if(workItems>=remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=1u;
    let body=bodies[bodyIndex];if(!bodyBoundingSphereVisible(ray.origin_m,ray.direction,body,tMin_m,bestT)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ray.origin_m,ray.direction,body,tMin_m,bestT)){continue;}let candidate=bodyHit(ray.origin_m,ray.direction,body);if(candidate.t>=tMin_m&&candidate.t<bestT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,candidate.t);}
  }

  if(dry.metadata.x<=${SVO_DRY_SCENE_DIRECT_PRIMITIVE_LIMIT}u){
    let onlyIgnoredReceiver=dry.metadata.x==1u&&svoPrimitiveOwnerId(primitives[0])==dryVisibilityIgnoredOwner;
    if(!onlyIgnoredReceiver){let candidate=tracePrimitiveCandidates(ray.origin_m,ray.direction,tMin_m,bestT,remaining.workItems-workItems,true);workItems+=candidate.workItems;
      if(candidate.status==DRY_CANDIDATE_INVALID){return dryVisibilityStep(SVO_VIS_STEP_INVALID,nodeVisits,leafVisits,workItems,DRY_MISS);}if(candidate.status==DRY_CANDIDATE_EXHAUSTED){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
      if(candidate.hit.t<bestT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,candidate.hit.t);}}
  }else{
    var cursor=max(tMin_m,0.0);
    for(var leafAttempt=0u;leafAttempt<${SVO_VISIBILITY_LIMITS.leafVisits}u;leafAttempt+=1u){
      if(cursor>=bestT){break;}if(leafVisits>=remaining.leafVisits||nodeVisits>=remaining.nodeVisits){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
      var shadowMapping=dryConfiguredMapping();shadowMapping.maxVisits=min(shadowMapping.maxVisits,remaining.nodeVisits-nodeVisits);
      let leaf=dryTraverse(SvoRay(ray.origin_m,cursor,ray.direction,bestT),shadowMapping);nodeVisits+=leaf.visits;
      if(leaf.status==SVO_STATUS_MISS){break;}
      if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
      if(leaf.status!=SVO_STATUS_HIT){return dryVisibilityStep(SVO_VIS_STEP_INVALID,nodeVisits,leafVisits,workItems,DRY_MISS);}leafVisits+=1u;
      let payloadRay=SvoVisibilityRay(ray.origin_m,bestT,ray.direction,ray.originBias_m);let payload=traceLeafPayloadVisibility(payloadRay,tMin_m,leaf,remaining.workItems-workItems);workItems+=payload.workItems;
      if(payload.status==SVO_VIS_STEP_HIT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,payload.t_m);}if(payload.status!=SVO_VIS_STEP_MISS){return dryVisibilityStep(payload.status,nodeVisits,leafVisits,workItems,payload.t_m);}
      cursor=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);
    }
    if(cursor<bestT&&leafVisits>=remaining.leafVisits){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
  }

  if(terrainEnabled()){
    let terrainWork=${SVO_TERRAIN_FALLBACK_STEPS + SVO_TERRAIN_FALLBACK_REFINEMENTS + 6}u;
    if(workItems+terrainWork>remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=terrainWork;
    let terrain=traceTerrain(ray.origin_m,ray.direction);if(terrain.t>=tMin_m&&terrain.t<bestT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,terrain.t);}
  }
  let paneCount=dry.terrain.y;if(workItems+paneCount>remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=paneCount;
  let glass=traceGlass(ray.origin_m,ray.direction,tMin_m,bestT,false);if(glass.hit.valid!=0u&&glass.hit.t_m<bestT){let optics=svoThinGlassOptics(glassPanes[glass.recordIndex],glass.hit,dryThinGlassIncidentIor(DRY_MEDIUM_AIR));bestT=glass.hit.t_m;found=true;opaque=false;glassTransmission=optics.netTransmittance;}
  if(!found){return dryVisibilityStep(SVO_VIS_STEP_MISS,nodeVisits,leafVisits,workItems,DRY_MISS);}if(opaque){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,bestT);}return dryVisibilityTransmissionStep(nodeVisits,leafVisits,workItems,bestT,glassTransmission);
}

${svoVisibilityTraceWGSL}

fn dryLightVisibility(position:vec3f,geometricNormal:vec3f,ownerId:u32,towardLight:vec3f,finiteDistance_m:f32)->vec3f {
  if(dot(geometricNormal,towardLight)<=0.0){return vec3f(0.0);}
  if((dry.materialPublication.w&2u)==0u){return vec3f(1.0);}
  if(dryShadowTracingEnabled==0u){return vec3f(1.0);}
  let maximumDistance=select(directionalLightSceneExitDistance(position,towardLight),finiteDistance_m,finiteDistance_m>0.0);if(maximumDistance<=0.0){return vec3f(0.0);}
  let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,towardLight,maximumDistance,dry.mapping.cellSize,${SVO_DRY_SCENE_SHADOW_BIAS_CELLS});
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u){let cone=dryConeVisibility(ray.origin_m,towardLight,.065,maximumDistance);if(cone.valid!=0u){let rigidBlocker=nearestBodyIgnoring(ray.origin_m,towardLight,ownerId);if(rigidBlocker.t<maximumDistance){return vec3f(0.0);}return vec3f(cone.transmittance);}}
  dryVisibilityIgnoredOwner=ownerId;
  let result=svoTraceVisibility(ray,SvoVisibilityBudget(${SVO_VISIBILITY_LIMITS.nodeVisits}u,${SVO_VISIBILITY_LIMITS.leafVisits}u,${SVO_VISIBILITY_LIMITS.workItems}u,4u),true,0.001,max(ray.originBias_m,1e-6));dryShadowNodeVisits+=result.nodeVisits;dryShadowLeafVisits+=result.leafVisits;dryShadowWorkItems+=result.workItems;if(result.status==SVO_VIS_STATUS_EXHAUSTED){dryTraversalFailure=max(dryTraversalFailure,1u);}else if(result.status==SVO_VIS_STATUS_INVALID){dryTraversalFailure=2u;}
  dryVisibilityIgnoredOwner=DRY_OWNER_NONE;
  return result.transmittance;
}

fn dryContactVisibilityRadius()->f32 {
  let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let sceneScale=max(uniforms.container.x,max(uniforms.container.y,uniforms.container.z));
  return min(sceneScale*${SVO_CONTACT_VISIBILITY_CONTRACT.maximumSceneRadiusFraction},max(cellScale*${SVO_CONTACT_VISIBILITY_CONTRACT.radiusCells}.0,sceneScale*${SVO_CONTACT_VISIBILITY_CONTRACT.minimumSceneRadiusFraction}));
}
fn dryContactVisibilityDirection(geometricNormalIn:vec3f,featureId:u32,sampleIndex:u32)->vec3f {
  let geometricNormal=normalize(geometricNormalIn);let helper=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(geometricNormal.y)>.9);var tangent=normalize(cross(helper,geometricNormal));var bitangent=cross(geometricNormal,tangent);
  if((featureId&1u)!=0u){let previous=tangent;tangent=bitangent;bitangent=-previous;}
  let signValue=select(1.0,-1.0,sampleIndex!=0u);return normalize(geometricNormal+signValue*(.55*tangent+.2*bitangent));
}
fn dryContactVisibility(position:vec3f,geometricNormal:vec3f,featureId:u32,ownerId:u32)->vec3f {
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u&&dryNodeMipReady()){
    let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(1.0);}var visibility=0.0;let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let origin=position+normalize(geometricNormal)*cellScale*.2;
    for(var sampleIndex=0u;sampleIndex<4u;sampleIndex+=1u){let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);let rotated=select(direction,normalize(direction+cross(normalize(geometricNormal),direction)*.7),sampleIndex>=2u);let cone=dryConeVisibility(origin,rotated,.62,radius);if(cone.valid==0u){return vec3f(1.0);}let rigidBlocker=nearestBodyIgnoring(origin,rotated,ownerId);visibility+=select(cone.transmittance,0.0,rigidBlocker.t<radius);}return vec3f(clamp(visibility*.25,0.0,1.0));
  }
  if((dry.materialPublication.w&1u)==0u){return vec3f(1.0);}
  let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(0.0);}let biasCells=select(${SVO_CONTACT_VISIBILITY_CONTRACT.smoothBiasCells},${SVO_CONTACT_VISIBILITY_CONTRACT.hardFeatureBiasCells},featureId!=SVO_FEATURE_SMOOTH);var visibility=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}u;sampleIndex+=1u){let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex);let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,direction,radius,dry.mapping.cellSize,biasCells);let result=svoTraceVisibility(ray,SvoVisibilityBudget(${SVO_CONTACT_VISIBILITY_CONTRACT.maximumNodeVisitsPerSample}u,${SVO_CONTACT_VISIBILITY_CONTRACT.maximumLeafVisitsPerSample}u,${SVO_CONTACT_VISIBILITY_CONTRACT.maximumWorkItemsPerSample}u,${SVO_CONTACT_VISIBILITY_CONTRACT.maximumIntersectionsPerSample}u),true,0.001,max(ray.originBias_m,1e-6));if(result.status==SVO_VIS_STATUS_INVALID||result.status==SVO_VIS_STATUS_EXHAUSTED){return vec3f(0.0);}visibility+=result.transmittance;}
  return clamp(visibility/f32(${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}),vec3f(0.0),vec3f(1.0));
}

fn dryEnvironment(rd:vec3f,roughness:f32)->vec3f{return svoEnvironmentPrefilteredSpecular(dryLighting.environment,rd,roughness);}
struct DryLightSample{towardLight:vec3f,finiteDistance_m:f32,radiance:vec3f,valid:u32}
fn dryInvalidLightSample()->DryLightSample{return DryLightSample(vec3f(0.0,1.0,0.0),0.0,vec3f(0.0),0u);}
fn dryLightSample(light:SvoLightRecord,sampleIndex:u32,position:vec3f)->DryLightSample {
  let baseRadiance=svoLightRadiance(light);if(max(max(baseRadiance.x,baseRadiance.y),baseRadiance.z)<=0.0){return dryInvalidLightSample();}
  if(light.identity.x==SVO_LIGHT_DIRECTIONAL){let lengthSquared=dot(light.directionCone.xyz,light.directionCone.xyz);if(lengthSquared<=1e-12){return dryInvalidLightSample();}return DryLightSample(light.directionCone.xyz*inverseSqrt(lengthSquared),0.0,baseRadiance,1u);}
  var samplePosition=light.positionRange.xyz;
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA){
    let towardCenter=normalize(light.positionRange.xyz-position);let helper=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(towardCenter.y)>.9);let tangent=normalize(cross(towardCenter,helper));let signValue=select(-1.0,1.0,sampleIndex!=0u);samplePosition+=tangent*(signValue*.45*light.shape.x);
  }else if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){
    let signValue=select(-1.0,1.0,sampleIndex!=0u);samplePosition+=light.axisUWidth.xyz*(signValue*.45*light.axisUWidth.w)+light.axisVHeight.xyz*(signValue*.2*light.axisVHeight.w);
  }
  let offset=samplePosition-position;let distanceSquared=dot(offset,offset);if(distanceSquared<=1e-10){return dryInvalidLightSample();}if(light.positionRange.w>0.0&&distanceSquared>=light.positionRange.w*light.positionRange.w){return dryInvalidLightSample();}let distance=sqrt(distanceSquared);let towardLight=offset/distance;
  let rangeFade=select(1.0,pow(clamp(1.0-distance/max(light.positionRange.w,1e-6),0.0,1.0),2.0),light.positionRange.w>0.0);
  var shapeScale=1.0/max(1.0,distanceSquared);
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA){let area=4.0*UNIFIED_PI*light.shape.x*light.shape.x;shapeScale=area/max(area,distanceSquared);}
  if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){let area=4.0*light.axisUWidth.w*light.axisVHeight.w;let emitterFacing=max(dot(normalize(light.directionCone.xyz),-towardLight),0.0);shapeScale=emitterFacing*area/max(area,distanceSquared);}
  let radiance=baseRadiance*(rangeFade*shapeScale);if(max(max(radiance.x,radiance.y),radiance.z)<=0.0){return dryInvalidLightSample();}return DryLightSample(towardLight,distance,radiance,1u);
}
fn traceDrySolidScene(ro:vec3f,rd:vec3f)->DryHit {
  var hit=traceStatic(ro,rd);let terrain=traceTerrain(ro,rd);if(terrain.t<hit.t){hit=terrain;}let rigid=nearestBody(ro,rd);if(rigid.t<hit.t){hit=rigid;}
  return hit;
}
fn traceOpaqueScene(ro:vec3f,rd:vec3f)->DryHit {
  dryFluidPrimaryFailure=0u;dryFluidInsideAtStart=0u;dryPrimitiveCandidateFailure=0u;
  var hit=traceDrySolidScene(ro,rd);
  if(dry.fluidDomainMode.w==1u){
    let fluid=svoTraceStructuralFluidPrimary(ro,rd,hit.t,dryConfiguredMapping());dryPrimaryNodeVisits+=fluid.nodeVisits;dryPrimaryFieldSteps+=fluid.fieldSteps;dryPrimaryMaximumDepth=max(dryPrimaryMaximumDepth,svoStructuralFluidPrimaryTraversalDepth());
    if(fluid.status==SVO_FLUID_PRIMARY_HIT){hit=DryHit(fluid.t_m,fluid.normal,${VOXEL_MATERIAL_IDS.fluid}u,DRY_OWNER_NONE,SVO_FEATURE_SMOOTH,dryFluidFieldSource(fluid.fieldOwner),DRY_GBUFFER_MOTION_FLUID,0u,0.0,vec3u(0u));dryFluidInsideAtStart=fluid.insideFluidAtStart;}
    else if(fluid.status==SVO_FLUID_PRIMARY_EXHAUSTED){dryFluidPrimaryFailure=DRY_GBUFFER_WORK_EXHAUSTED;}
    else if(fluid.status==SVO_FLUID_PRIMARY_INVALID){dryFluidPrimaryFailure=DRY_GBUFFER_INVALID_FIELD;}
  }
  return hit;
}

fn svoFluidMediaQueryWater(ray:SvoFluidMediaRay,generation:u32)->SvoFluidMediaWaterStep {
  let fluid=svoTraceStructuralFluidPrimary(ray.origin_m,ray.direction,ray.maximumDistance_m,dryConfiguredMapping());dryPrimaryNodeVisits+=fluid.nodeVisits;dryPrimaryFieldSteps+=fluid.fieldSteps;dryPrimaryMaximumDepth=max(dryPrimaryMaximumDepth,svoStructuralFluidPrimaryTraversalDepth());
  let publishedGeneration=select(publicationState[0],generation,generation!=0u);let revision=publicationState[5];
  var status=SVO_FLUID_MEDIA_MISS;
  if(fluid.status==SVO_FLUID_PRIMARY_HIT){status=SVO_FLUID_MEDIA_HIT;}
  else if(fluid.status==SVO_FLUID_PRIMARY_INVALID){status=SVO_FLUID_MEDIA_INVALID;}
  else if(fluid.status==SVO_FLUID_PRIMARY_EXHAUSTED){status=SVO_FLUID_MEDIA_EXHAUSTED;}
  return SvoFluidMediaWaterStep(status,fluid.insideFluidAtStart,publishedGeneration,revision,fluid.t_m,fluid.fieldSteps,fluid.nodeVisits,${VOXEL_MATERIAL_IDS.fluid}u,fluid.normal,0.0);
}
fn svoFluidMediaQueryScene(ray:SvoFluidMediaRay)->SvoFluidMediaSceneStep {
  let solid=traceDrySolidScene(ray.origin_m,ray.direction);let glass=traceGlass(ray.origin_m,ray.direction,0.0,ray.maximumDistance_m,false);
  if(solid.t<=ray.maximumDistance_m&&(glass.hit.valid==0u||solid.t<=glass.hit.t_m)){
    return SvoFluidMediaSceneStep(SVO_FLUID_MEDIA_HIT,DRY_MEDIUM_OPAQUE,0u,solid.ownerId,solid.t,vec3f(0.0),solid.normal,1.0,vec3f(1.0),0.0);
  }
  if(glass.hit.valid!=0u){let pane=glassPanes[glass.recordIndex];let optics=svoThinGlassOptics(pane,glass.hit,dryThinGlassIncidentIor(DRY_MEDIUM_WATER));return SvoFluidMediaSceneStep(SVO_FLUID_MEDIA_HIT,DRY_MEDIUM_GLASS,1u,svoThinGlassPaneId(pane),glass.hit.t_m,vec3f(0.0),glass.hit.geometricNormal,pane.extentIorEpsilon.z,optics.absorptionTint,0.0);}
  return SvoFluidMediaSceneStep(SVO_FLUID_MEDIA_MISS,DRY_MEDIUM_AIR,0u,0u,0.0,vec3f(0.0),vec3f(0.0,1.0,0.0),1.0,vec3f(1.0),0.0);
}
${svoFluidMediaTraceWGSL}
const DRY_SURFACE_REGION_NONE:u32=0xffffffffu;
struct DrySurfaceMaterial{baseColor:vec3f,roughness:f32,emissive:vec3f,metallic:f32,specularF0:vec3f,specularWeight:f32,regionId:u32,variationFlags:u32,valid:u32,_padding:u32}
fn dryInvalidSurfaceMaterial()->DrySurfaceMaterial{return DrySurfaceMaterial(vec3f(0.0),1.0,vec3f(0.0),0.0,vec3f(0.04),0.0,DRY_SURFACE_REGION_NONE,0u,0u,0u);}
fn dryBodyPbrMaterialId(body:BodyGPU)->u32{
  let shape=i32(round(body.halfSizeShape.w));if(shape==0){return ${VOXEL_MATERIAL_IDS.sphere}u;}if(shape==1){return ${VOXEL_MATERIAL_IDS.box}u;}if(shape==2){return ${VOXEL_MATERIAL_IDS.capsule}u;}return ${VOXEL_MATERIAL_IDS.cylinder}u;
}
fn dryResolvedMaterialId(hit:DryHit)->u32{
  if((hit.materialId&0x80000000u)!=0u){return dryBodyPbrMaterialId(bodies[hit.materialId&0x7fffffffu]);}
  return hit.materialId;
}
fn dryPublishedMaterialValid(material:SvoMaterialRecord,index:u32)->bool{
  return index<dry.materialPublication.x&&index<arrayLength(&materials)&&svoMaterialValid(material,index)&&material.identity.y==dry.materialPublication.y&&(material.identity.w&SVO_MATERIAL_FLAG_OPAQUE)!=0u;
}
// Stable adapter point for M7's pending G-buffer: material identity remains on
// DryHit, while procedural region/variation identity is evaluated exactly once
// from the same world-space hit used for the PBR closure.
fn dryEvaluateSurfaceMaterial(hit:DryHit,position:vec3f)->DrySurfaceMaterial {
  // Temporary bounded primary-visibility material. It is deliberately opaque
  // until the shared media tracer replaces the legacy water compositor.
  if(dryIsFluidFieldSource(hit.fieldSource)){return DrySurfaceMaterial(vec3f(.035,.22,.25),.12,vec3f(0.0),0.0,vec3f(.0204),1.0,DRY_SURFACE_REGION_NONE,0u,1u,0u);}
  var materialId=dryResolvedMaterialId(hit);var baseOverride=vec3f(0.0);var useBaseOverride=false;var selectedEmission=vec3f(0.0);
  if((hit.materialId&0x80000000u)!=0u){let body=bodies[hit.materialId&0x7fffffffu];baseOverride=body.colorSelected.xyz;useBaseOverride=true;selectedEmission=body.colorSelected.w*vec3f(.12,.42,.32);}
  if(materialId>=dry.materialPublication.x||materialId>=arrayLength(&materials)){return dryInvalidSurfaceMaterial();}let material=materials[materialId];if(!dryPublishedMaterialValid(material,materialId)){return dryInvalidSurfaceMaterial();}
  var base=select(material.baseColorOpacity.xyz,baseOverride,useBaseOverride);var roughness=material.emissiveRoughness.w;var regionId=DRY_SURFACE_REGION_NONE;var variationFlags=0u;
  let terrainPolicyValid=material.identity.z==SVO_MATERIAL_FUNCTION_GARDEN_TERRAIN&&dry.terrainMaterial.policyVersion==1u&&dry.terrainMaterial.materialId==materialId&&materialId==dry.terrain.x;
  if(terrainPolicyValid){let terrainSample=svoTerrainMaterial(dry.terrainMaterial,position,hit.normal);base=terrainSample.colorLinear;regionId=terrainSample.regionId;variationFlags=terrainSample.variationFlags;}
  else{let procedural=svoProceduralMaterial(material.identity.z,base,roughness,position);base=procedural.baseColorLinear;roughness=procedural.roughness;variationFlags=procedural.variationFlags;}
  return DrySurfaceMaterial(base,roughness,material.emissiveRoughness.xyz+selectedEmission,material.surface.x,vec3f(svoMaterialDielectricF0(material)),material.surface.y,regionId,variationFlags,1u,0u);
}
fn shadeDryOpaque(hit:DryHit,ro:vec3f,rd:vec3f)->vec3f {
  if(hit.t>=DRY_MISS){return dryEnvironment(rd,0.0);}let position=ro+rd*hit.t;let surface=dryEvaluateSurfaceMaterial(hit,position);
  if(surface.valid==0u){return vec3f(0.0);}
  let directClosure=unifiedPbrMaterial(surface.baseColor,surface.metallic,surface.roughness,vec3f(0.0),0.0,surface.specularF0,surface.specularWeight,vec3f(0.0),0.0);var direct=vec3f(0.0);var sampleBudget=0u;
  let lightCount=min(dryLighting.metadata.x,min(${SVO_DRY_SCENE_MAX_SHADED_LIGHTS}u,${SVO_LIGHT_MAXIMUM_RECORDS}u));
  for(var lightIndex=0u;lightIndex<${SVO_DRY_SCENE_MAX_SHADED_LIGHTS}u;lightIndex+=1u){
    if(lightIndex>=lightCount||sampleBudget>=${SVO_DRY_SCENE_MAX_SHADED_LIGHTS}u){break;}let light=dryLighting.lights[lightIndex];if(light.identity.w!=dryLighting.metadata.y){continue;}let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA;let sampleCount=select(1u,${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u,area);
    for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=sampleCount||sampleBudget>=${SVO_DRY_SCENE_MAX_SHADED_LIGHTS}u){break;}sampleBudget+=1u;let sample=dryLightSample(light,sampleIndex,position);if(sample.valid==0u||dot(hit.normal,sample.towardLight)<=0.0){continue;}let visibility=dryLightVisibility(position,hit.normal,hit.ownerId,sample.towardLight,sample.finiteDistance_m);let lighting=unifiedLightingInputWithGeometry(hit.normal,hit.normal,-rd,sample.towardLight,sample.radiance*visibility/f32(sampleCount));direct+=shadeUnifiedSurface(directClosure,lighting);}
  }
  let viewDirection=normalize(-rd);let reflected=reflect(rd,hit.normal);let diffuseColor=surface.baseColor*(1.0-surface.metallic);let f0=mix(surface.specularF0*surface.specularWeight,surface.baseColor,surface.metallic);let fresnel=unifiedSchlick(max(dot(hit.normal,viewDirection),0.0),f0);let contactVisibility=dryContactVisibility(position,hit.normal,hit.featureId,hit.ownerId);let diffuseEnvironment=diffuseColor*svoEnvironmentDiffuseIrradiance(dryLighting.environment,hit.normal)*contactVisibility/UNIFIED_PI;let specularEnvironment=dryEnvironment(reflected,surface.roughness)*fresnel;
  return max(surface.emissive+diffuseEnvironment+specularEnvironment+direct,vec3f(0.0));
}

struct DryGlassSurface{color:vec3f,depth:f32,materialId:u32,ownerId:u32,paneId:u32,_padding:u32}
fn shadeThinGlass(glass:DryGlassHit,opaque:DryHit,ro:vec3f,rd:vec3f)->DryGlassSurface {
  let record=glassPanes[glass.recordIndex];let incidentIor=dryThinGlassIncidentIor(DRY_MEDIUM_AIR);let optics=svoThinGlassOptics(record,glass.hit,incidentIor);
  // A collapsed sheet has no net Snell bend, so the already-resolved collinear
  // opaque hit is exactly the transmitted scene query; never traverse it twice.
  let reflected=dryEnvironment(reflect(rd,glass.hit.geometricNormal),.04);let transmitted=shadeDryOpaque(opaque,ro,rd);
  let color=reflected*optics.fresnel+transmitted*optics.netTransmittance;
  return DryGlassSurface(color,glass.hit.t_m,svoThinGlassMaterialId(record),svoThinGlassOwnerId(record),svoThinGlassPaneId(record),0u);
}

fn dryThickGlassEmission(materialId:u32)->vec3f{
  if(materialId>=dry.materialPublication.x||materialId>=arrayLength(&materials)){return vec3f(0.0);}let material=materials[materialId];
  if(!svoMaterialValid(material,materialId)||material.identity.y!=dry.materialPublication.y){return vec3f(0.0);}return material.emissiveRoughness.xyz;
}
fn shadeThickGlass(glass:DryThickGlassHit,ro:vec3f,rd:vec3f)->DryGlassSurface{
  let record=thickGlass.records[glass.recordIndex];let first=dryThickGlassFirst(glass.interval);let ior=record.radiiYzIorEpsilon.z;
  let fromIor=select(1.0,ior,glass.interval.insideAtStart!=0u);let toIor=select(ior,1.0,glass.interval.insideAtStart!=0u);
  let firstOptics=svoThickGlassInterface(record,first,rd,fromIor,toIor,0.0);let reflected=dryEnvironment(firstOptics.reflectedDirection,.04);var transmitted=vec3f(0.0);var transmission=vec3f(0.0);
  if(firstOptics.totalInternalReflection==0u){
    if(glass.interval.insideAtStart!=0u){let origin=first.position_m+firstOptics.refractedDirection*record.radiiYzIorEpsilon.w;let opaque=traceOpaqueScene(origin,firstOptics.refractedDirection);transmitted=shadeDryOpaque(opaque,origin,firstOptics.refractedDirection);transmission=vec3f(1.0-firstOptics.fresnel);}
    else if(glass.interval.tangent!=0u){let origin=first.position_m+rd*record.radiiYzIorEpsilon.w;let opaque=traceOpaqueScene(origin,rd);transmitted=shadeDryOpaque(opaque,origin,rd);transmission=vec3f(1.0-firstOptics.fresnel);}
    else{
      let insideOrigin=first.position_m+firstOptics.refractedDirection*record.radiiYzIorEpsilon.w;let inside=svoThickGlassIntersect(record,insideOrigin,firstOptics.refractedDirection,0.0,record.absorptionPath.w,thickGlass.metadata.y);
      if(inside.status==SVO_THICK_GLASS_HIT){let exitSurface=inside.exit;let exitOptics=svoThickGlassInterface(record,exitSurface,firstOptics.refractedDirection,ior,1.0,inside.opticalPath_m);
        if(exitOptics.totalInternalReflection==0u){let outsideOrigin=exitSurface.position_m+exitOptics.refractedDirection*record.radiiYzIorEpsilon.w;let opaque=traceOpaqueScene(outsideOrigin,exitOptics.refractedDirection);transmitted=shadeDryOpaque(opaque,outsideOrigin,exitOptics.refractedDirection);transmission=exitOptics.absorptionTint*(1.0-firstOptics.fresnel)*(1.0-exitOptics.fresnel);}
      }
    }
  }
  let materialId=svoThickGlassMaterialId(record);let color=reflected*firstOptics.fresnel+transmitted*transmission+dryThickGlassEmission(materialId);
  return DryGlassSurface(max(color,vec3f(0.0)),first.t_m,materialId,svoThickGlassOwnerId(record),svoThickGlassId(record),0u);
}

struct VertexOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index) index:u32)->VertexOut {var points=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var output:VertexOut;output.position=vec4f(points[index],0,1);output.uv=points[index]*.5+.5;return output;}

struct DryFragmentOut{
  @location(0) radianceDepth:vec4f,
  @location(1) packedSurface:vec4u,
  @location(2) identityMedia:vec4u,
  @builtin(frag_depth) hardwareDepth:f32,
}
// Analytic/static surface identity follows the static-geometry revision, not
// the per-frame completion fence. Fluid and rigid paths publish their own
// local generations below.
fn dryPublicationGeneration()->u32{return select(0u,publicationState[3],arrayLength(&publicationState)>3u);}
struct DryRigidMotionSurface{velocity_m_s:vec3f,generation:u32,valid:u32}
fn dryRigidMotionSurface(hit:DryHit,worldSurfacePosition_m:vec3f)->DryRigidMotionSurface{
  if(hit.motionKind!=DRY_GBUFFER_MOTION_RIGID||hit.ownerId>=12u||hit.ownerId>=u32(round(uniforms.options.z))){return DryRigidMotionSurface(vec3f(0.0),dryPublicationGeneration(),0u);}
  let record=rigidMotion[hit.ownerId];let generation=svoPrimitiveMotionGeneration(record);let identityValid=record.identityRevision.x==hit.ownerId&&svoPrimitiveMotionOwnerId(record)==hit.ownerId&&svoPrimitiveMotionMaterialId(record)==dryResolvedMaterialId(hit)&&generation!=0u;let transformValid=distance(record.currentPositionDt.xyz,bodies[hit.ownerId].positionRadius.xyz)<=1e-5;let velocity=svoPrimitiveMotionVelocityAt(record,worldSurfacePosition_m);let valid=identityValid&&transformValid&&velocity.valid!=0u;
  return DryRigidMotionSurface(select(vec3f(0.0),velocity.velocity_m_s,valid),select(dryPublicationGeneration(),generation,generation!=0u),select(0u,1u,valid));
}
fn dryMediumPair(rd:vec3f,geometricNormal:vec3f,surfaceMedium:u32)->vec2u{
  return select(vec2u(surfaceMedium,DRY_MEDIUM_AIR),vec2u(DRY_MEDIUM_AIR,surfaceMedium),dot(rd,geometricNormal)<0.0);
}
fn dryHardwareDepth(t_m:f32,rd:vec3f,forward:vec3f)->f32{
  if(!(t_m<DRY_MISS)){return 0.0;}let viewDepth_m=t_m*max(dot(rd,forward),1e-6);return clamp(DRY_REVERSED_Z_NEAR_M/viewDepth_m,0.0,1.0);
}
fn dryFragmentOut(targetsIn:SvoGBufferTargets,hardwareDepth:f32)->DryFragmentOut{
  var targets=targetsIn;let flags=(targets.packedSurface.w>>4u)&0xffffu;
  if(dryShadowTracingEnabled==0u&&(flags&SVO_GBUFFER_VALID_SURFACE)!=0u){targets.packedSurface.w|=DRY_GBUFFER_SHADOW_DEFERRED<<20u;}
  targets.radianceDepth=dryCostOverlay(targets.radianceDepth);return DryFragmentOut(targets.radianceDepth,targets.packedSurface,targets.identityMedia,hardwareDepth);
}

@fragment fn fragmentMain(input:VertexOut)->DryFragmentOut {
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);dryPrimitiveCandidateFailure=0u;dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryPrimaryNodeVisits=0u;dryPrimaryLeafVisits=0u;dryPrimaryMaximumDepth=0u;dryPrimaryFieldSteps=0u;dryCandidateWorkItems=0u;dryShadowNodeVisits=0u;dryShadowLeafVisits=0u;dryShadowWorkItems=0u;dryTraversalFailure=0u;let temporalShadowSampling=uniforms.viewport.w>=0.0&&(dry.materialPublication.w&2u)!=0u;let shadowParity=(u32(input.position.x)+u32(input.position.y)+u32(uniforms.viewport.w))&1u;dryShadowTracingEnabled=select(1u,select(0u,1u,shadowParity==0u),temporalShadowSampling);let thickBinderValid=thickGlass.metadata.x>0u&&thickGlass.metadata.x<=${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u&&thickGlass.metadata.y!=0u&&thickGlass.metadata.w==${SVO_DRY_THICK_GLASS_BINDER_VERSION}u;dryThickGlassEnabled=select(0u,1u,thickBinderValid&&dry.fluidDomainMode.w!=2u);
  if(dry.fluidDomainMode.w==2u){
    let solid=traceDrySolidScene(ro,rd);let sceneDistance=max(max(uniforms.container.x,uniforms.container.y),uniforms.container.z)*8.0;
    if(dryPrimitiveCandidateFailure!=0u){return dryFragmentOut(svoGBufferMiss(vec3f(.22,.005,.02),DRY_GBUFFER_FIELD_ANALYTIC,dryPublicationGeneration(),dryPrimitiveCandidateFailure,4096u),0.0);}
    let first=svoTraceStructuralFluidPrimary(ro,rd,sceneDistance,dryConfiguredMapping());dryPrimaryNodeVisits+=first.nodeVisits;dryPrimaryFieldSteps+=first.fieldSteps;dryPrimaryMaximumDepth=max(dryPrimaryMaximumDepth,svoStructuralFluidPrimaryTraversalDepth());
    if(first.status==SVO_FLUID_PRIMARY_INVALID||first.status==SVO_FLUID_PRIMARY_EXHAUSTED){dryTraversalFailure=select(2u,1u,first.status==SVO_FLUID_PRIMARY_EXHAUSTED);let failure=select(DRY_GBUFFER_INVALID_FIELD,DRY_GBUFFER_WORK_EXHAUSTED,first.status==SVO_FLUID_PRIMARY_EXHAUSTED);return dryFragmentOut(svoGBufferMiss(vec3f(.22,.005,.02),SVO_GBUFFER_FIELD_FLUID_COARSE,dryPublicationGeneration(),failure,select(8192u,4096u,failure==DRY_GBUFFER_WORK_EXHAUSTED)),0.0);}
    if(first.status==SVO_FLUID_PRIMARY_HIT&&(first.insideFluidAtStart!=0u||first.t_m<solid.t)){
      let initialMedium=select(DRY_MEDIUM_AIR,DRY_MEDIUM_WATER,first.insideFluidAtStart!=0u);let media=svoTraceStructuralFluidMedia(SvoFluidMediaRay(ro,sceneDistance,rd,0.0),initialMedium,publicationState[0],max(1e-5,min(dry.mapping.cellSize.x,min(dry.mapping.cellSize.y,dry.mapping.cellSize.z))*1e-3),max(1e-5,min(dry.mapping.cellSize.x,min(dry.mapping.cellSize.y,dry.mapping.cellSize.z))*1e-3),vec3f(.16,.32,.38));
      if(media.status==SVO_FLUID_MEDIA_RESULT_INVALID||media.status==SVO_FLUID_MEDIA_RESULT_EXHAUSTED){dryTraversalFailure=select(2u,1u,media.status==SVO_FLUID_MEDIA_RESULT_EXHAUSTED);let failure=select(DRY_GBUFFER_INVALID_FIELD,DRY_GBUFFER_WORK_EXHAUSTED,media.status==SVO_FLUID_MEDIA_RESULT_EXHAUSTED);return dryFragmentOut(svoGBufferMiss(vec3f(.22,.005,.02),SVO_GBUFFER_FIELD_FLUID_COARSE,media.generation,failure,select(8192u,4096u,failure==DRY_GBUFFER_WORK_EXHAUSTED)),0.0);}
      let normal=first.normal;let cosine=clamp(abs(dot(rd,normal)),0.0,1.0);let fresnel=${WATER_OPTICS.fresnelF0}+(1.0-${WATER_OPTICS.fresnelF0})*pow(1.0-cosine,5.0);let reflected=dryEnvironment(reflect(rd,normal),.04);let transmitted=dryEnvironment(media.direction,.08);let radiance=max(media.inscatter+media.throughput*transmitted+reflected*fresnel,vec3f(0.0));let mediaBefore=select(DRY_MEDIUM_AIR,DRY_MEDIUM_WATER,first.insideFluidAtStart!=0u);let mediaAfter=select(DRY_MEDIUM_WATER,media.currentMedium,first.insideFluidAtStart!=0u);var flags=DRY_GBUFFER_FLUID_SURFACE;if(first.insideFluidAtStart!=0u){flags|=DRY_GBUFFER_INSIDE_FLUID;}
      let targets=svoGBufferSurface(radiance,first.t_m,normal,normal,vec4u(${VOXEL_MATERIAL_IDS.fluid}u,DRY_OWNER_NONE,mediaBefore,mediaAfter),vec3f(0.0),DRY_GBUFFER_MOTION_FLUID,dryFluidFieldSource(first.fieldOwner),media.generation,flags,SVO_FEATURE_SMOOTH);return dryFragmentOut(targets,dryHardwareDepth(first.t_m,rd,forward));
    }
  }
  var thick=traceThickGlass(ro,rd,0.0,DRY_MISS);if(dryThickGlassFailure!=0u){dryThickGlassEnabled=0u;thick=dryThickGlassMiss();}
  let opaque=traceOpaqueScene(ro,rd);if(dryPrimitiveCandidateFailure!=0u){return dryFragmentOut(svoGBufferMiss(vec3f(.22,.005,.02),DRY_GBUFFER_FIELD_ANALYTIC,dryPublicationGeneration(),dryPrimitiveCandidateFailure,4096u),0.0);}let glass=traceGlass(ro,rd,0.0,opaque.t,true);var color=shadeDryOpaque(opaque,ro,rd);var depth=opaque.t;
  let thickFirst=dryThickGlassFirst(thick.interval);let thickVisible=thick.interval.status==SVO_THICK_GLASS_HIT&&thickFirst.t_m<opaque.t;let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t&&(!thickVisible||glass.hit.t_m<thickFirst.t_m);var glassSurface=DryGlassSurface(vec3f(0.0),DRY_MISS,0u,DRY_OWNER_NONE,0u,0u);
  if(glassVisible){glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  else if(thickVisible){glassSurface=shadeThickGlass(thick,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);let radiance=max(color*vignette,vec3f(0.0));let generation=dryPublicationGeneration();
  if(dryFluidPrimaryFailure!=0u){let flags=select(8192u,4096u,dryFluidPrimaryFailure==DRY_GBUFFER_WORK_EXHAUSTED);return dryFragmentOut(svoGBufferMiss(vec3f(.22,.005,.02),SVO_GBUFFER_FIELD_FLUID_COARSE,generation,dryFluidPrimaryFailure,flags),0.0);}
  if(glassVisible){
    let media=dryMediumPair(rd,glass.hit.geometricNormal,DRY_MEDIUM_GLASS);
    let targets=svoGBufferSurface(radiance,depth,glass.hit.geometricNormal,glass.hit.geometricNormal,vec4u(glassSurface.materialId,glassSurface.ownerId,media.x,media.y),vec3f(0.0),DRY_GBUFFER_MOTION_STATIC,DRY_GBUFFER_FIELD_ANALYTIC,generation,SVO_GBUFFER_MOTION_VALID,SVO_FEATURE_SMOOTH);
    return dryFragmentOut(targets,dryHardwareDepth(depth,rd,forward));
  }
  if(thickVisible){
    let media=dryMediumPair(rd,thickFirst.normal,DRY_MEDIUM_GLASS);
    let targets=svoGBufferSurface(radiance,depth,thickFirst.normal,thickFirst.normal,vec4u(glassSurface.materialId,glassSurface.ownerId,media.x,media.y),vec3f(0.0),DRY_GBUFFER_MOTION_STATIC,DRY_GBUFFER_FIELD_ANALYTIC,generation,SVO_GBUFFER_MOTION_VALID,SVO_FEATURE_SMOOTH);
    return dryFragmentOut(targets,dryHardwareDepth(depth,rd,forward));
  }
  if(opaque.t<DRY_MISS){
    let surfaceMedium=select(DRY_MEDIUM_OPAQUE,DRY_MEDIUM_WATER,dryIsFluidFieldSource(opaque.fieldSource));let media=dryMediumPair(rd,opaque.normal,surfaceMedium);let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}if(dryIsFluidFieldSource(opaque.fieldSource)){flags|=DRY_GBUFFER_FLUID_SURFACE;if(dryFluidInsideAtStart!=0u){flags|=DRY_GBUFFER_INSIDE_FLUID;}}
    let targets=svoGBufferSurface(radiance,opaque.t,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);
    return dryFragmentOut(targets,dryHardwareDepth(opaque.t,rd,forward));
  }
  return dryFragmentOut(svoGBufferMiss(radiance,0u,generation,DRY_GBUFFER_NO_INTERSECTION,0u),0.0);
}
`;

async function checkedModule(device: GPUDevice, code: string): Promise<GPUShaderModule> {
  const shaderModule = device.createShaderModule({ label: "Sparse voxel dry scene", code });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(`Sparse voxel dry scene:\n${errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n")}`);
  return shaderModule;
}

export class SparseVoxelDrySceneRenderer {
  private pipeline?: GPURenderPipeline;
  private layout?: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private primitiveBuffer?: GPUBuffer;
  private primitiveCandidateArena?: SvoPrimitiveCandidateArena;
  private glassBuffer?: GPUBuffer;
  private glassCacheKey?: string;
  private readonly paramsBuffer: GPUBuffer;
  private readonly lightingBuffer: GPUBuffer;
  private readonly rigidMotionUniformBuffer: GPUBuffer;
  private readonly thickGlassUniformBuffer: GPUBuffer;
  private readonly nodeMipFallbackAtlas: GPUTexture;
  private readonly nodeMipFallbackAtlasView: GPUTextureView;
  private readonly nodeMipFallbackDirectory: GPUTexture;
  private readonly nodeMipFallbackDirectoryView: GPUTextureView;
  private readonly nodeMipFallbackSampler: GPUSampler;
  private rigidMotionSource?: GPUBuffer;
  private readonly gBufferTargets: SparseVoxelGBufferTargetArena;
  private readonly pickingReadback: SparseVoxelGpuPickingReadbackRing;
  private lastPickingTarget?: GPUTexture;
  private readonly targetViews = new WeakMap<GPUTexture, GPUTextureView>();
  private reusableKey?: string;
  private reusableStableFrames = 0;
  private reusableTarget?: GPUTexture;
  private reusableResult?: DrySceneReplacementResult;
  /** Resource/source epoch. Later compatible frames do not invalidate a copy already ordered on the queue. */
  private pickingFrameToken = 1;
  private readonly temporalAccumulator: SparseVoxelTemporalAccumulator;
  private paramsWords?: Uint32Array<ArrayBuffer>;
  private source?: SparseVoxelSceneRenderSource;
  private scene?: SparseVoxelDrySceneData;
  private fineFluidCapability?: SvoFineFluidGpuCapability;
  private lightingMode: SvoLightingMode = DEFAULT_SVO_LIGHTING_MODE;

  constructor(
    private readonly device: GPUDevice,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bodyBuffer: GPUBuffer,
    private readonly targetFormat: GPUTextureFormat = "rgba16float"
  ) {
    if (targetFormat !== SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat) {
      throw new Error(`Sparse voxel dry scene location 0 must use ${SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat}`);
    }
    this.paramsBuffer = device.createBuffer({ label: "Sparse voxel dry scene parameters", size: SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lightingBuffer = device.createBuffer({ label: "Sparse voxel dry scene lighting arena", size: SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rigidMotionUniformBuffer = device.createBuffer({ label: "Sparse voxel rigid motion uniform mirror", size: SVO_DRY_RIGID_MOTION_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.thickGlassUniformBuffer = device.createBuffer({ label: "Sparse voxel thick-glass uniform binder", size: SVO_DRY_THICK_GLASS_ARENA_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.nodeMipFallbackAtlas = device.createTexture({ label: "Sparse voxel node-mip fallback atlas", size: [1, 1, 1], dimension: "3d", format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.nodeMipFallbackAtlasView = this.nodeMipFallbackAtlas.createView({ dimension: "3d" });
    this.nodeMipFallbackDirectory = device.createTexture({ label: "Sparse voxel node-mip fallback directory", size: [2, 1], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.nodeMipFallbackDirectoryView = this.nodeMipFallbackDirectory.createView();
    this.nodeMipFallbackSampler = device.createSampler({ label: "Sparse voxel node-mip fallback sampler", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge", magFilter: "linear", minFilter: "linear" });
    this.gBufferTargets = new SparseVoxelGBufferTargetArena(device);
    this.pickingReadback = new SparseVoxelGpuPickingReadbackRing(device);
    this.temporalAccumulator = new SparseVoxelTemporalAccumulator(device);
  }

  async initialize(): Promise<void> {
    const fragmentStorageLimit = Number(this.device.limits?.maxStorageBuffersPerShaderStage);
    if (Number.isFinite(fragmentStorageLimit) && fragmentStorageLimit < SVO_STRUCTURAL_FLUID_PRIMARY_STORAGE_BINDINGS) {
      throw new Error(`Sparse voxel dry scene requires ${SVO_STRUCTURAL_FLUID_PRIMARY_STORAGE_BINDINGS} fragment storage buffers; adapter exposes ${fragmentStorageLimit}`);
    }
    const shaderModule = await checkedModule(this.device, drySceneShader);
    this.layout = this.device.createBindGroupLayout({ label: "Sparse voxel dry scene bindings", entries: sparseVoxelDrySceneBindGroupLayoutEntries() });
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: "Sparse voxel dry scene", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module: shaderModule, entryPoint: "vertexMain" }, fragment: { module: shaderModule, entryPoint: "fragmentMain", targets: [
        { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat },
        { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.packedSurfaceFormat },
        { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.identityMediaFormat },
      ] },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
        depthWriteEnabled: true,
        depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
      },
    });
    await this.temporalAccumulator.initialize();
    this.rebuild();
  }

  setSource(source: SparseVoxelSceneRenderSource | undefined, scene: SparseVoxelDrySceneData | undefined): void {
    this.pickingFrameToken += 1;
    this.lastPickingTarget = undefined;
    this.clearReusableFrame();
    this.temporalAccumulator.invalidate();
    this.source = source;
    this.scene = scene;
    this.primitiveBuffer?.destroy();
    this.primitiveBuffer = undefined;
    this.primitiveCandidateArena = undefined;
    const reuseGlassBuffer = Boolean(
      this.glassBuffer && scene?.glassCacheKey && scene.glassCacheKey === this.glassCacheKey,
    );
    if (!reuseGlassBuffer) {
      this.glassBuffer?.destroy();
      this.glassBuffer = undefined;
      this.glassCacheKey = undefined;
    }
    if (canEncodeSparseVoxelDryScene(source, scene)) {
      const primitiveCount = scene!.primitiveRecords.byteLength / SVO_PRIMITIVE_RECORD_STRIDE_BYTES;
      const candidateArena = primitiveCount <= SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES
        ? packSvoPrimitiveCandidateArena(scene!.primitiveRecords, scene!.primitiveCandidates!)
        : undefined;
      const primitiveArenaRecords = candidateArena?.packedRecords ?? scene!.primitiveRecords;
      this.primitiveCandidateArena = candidateArena;
      this.primitiveBuffer = this.device.createBuffer({ label: "Sparse voxel analytic primitives and static candidate BVH", size: primitiveArenaRecords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.primitiveBuffer, 0, primitiveArenaRecords);
      if (!this.glassBuffer) {
        const records = scene!.glassRecords;
        this.glassBuffer = this.device.createBuffer({ label: "Sparse voxel thin-glass panes", size: Math.max(SVO_THIN_GLASS_RECORD_STRIDE_BYTES, records?.byteLength ?? 0), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        if (records?.byteLength) this.device.queue.writeBuffer(this.glassBuffer, 0, records);
        this.glassCacheKey = scene!.glassCacheKey;
      }
      this.writeParams(source!, scene!);
      const lightingArena = packSparseVoxelDrySceneLightingArena(source!, scene!);
      if (lightingArena) this.device.queue.writeBuffer(this.lightingBuffer, 0, lightingArena);
      this.device.queue.writeBuffer(this.thickGlassUniformBuffer, 0, packSparseVoxelDrySceneThickGlassArena(scene));
    }
    this.rebuild();
  }

  /** Bind renderer-owned fine visibility without changing water presentation ownership. */
  setFineFluidCapability(capability?: SvoFineFluidGpuCapability): void {
    const next = canConsumeSvoFineFluidCapability(capability) ? capability : undefined;
    if (this.fineFluidCapability?.arena.buffer === next?.arena.buffer) return;
    this.fineFluidCapability = next;
    if (this.source && this.scene && canEncodeSparseVoxelDryScene(this.source, this.scene)) this.writeParams(this.source, this.scene);
    this.rebuild();
  }

  /** Select lighting without rebuilding source-owned scene resources. */
  setLightingMode(mode: SvoLightingMode): void {
    if (mode === this.lightingMode) return;
    this.lightingMode = mode;
    this.clearReusableFrame();
    this.temporalAccumulator.invalidate();
    if (this.source && this.scene && canEncodeSparseVoxelDryScene(this.source, this.scene)) {
      this.writeParams(this.source, this.scene);
    }
  }

  private writeParams(source: SparseVoxelSceneRenderSource, scene: SparseVoxelDrySceneData): void {
    const structural = source.structural!;
    const pbrMaterials = source.pbrMaterials!;
    const buffer = new ArrayBuffer(SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes), floats = new Float32Array(buffer), words = new Uint32Array(buffer);
    floats.set(structural.domain.worldOrigin_m, 0); words[3] = structural.domain.brickSize;
    floats.set(structural.domain.cellSize_m, 4); words[7] = structural.domain.maximumDepth;
    words.set([structural.capacities.nodes, structural.capacities.leaves, 256, 0], 8);
    words.set([scene.primitiveRecords.byteLength / SVO_PRIMITIVE_RECORD_STRIDE_BYTES, scene.ownerBase, scene.skippedOwnerId ?? 0xffff_ffff, pbrMaterials.count], 12);
    floats.set(scene.lightDirection ?? [-0.45, 0.86, 0.28], 16);
    floats.set(scene.lightColor ?? [1.04, 1.0, 0.91], 20);
    words.set([scene.terrainMaterialId ?? 0xffff_ffff, (scene.glassRecords?.byteLength ?? 0) / SVO_THIN_GLASS_RECORD_STRIDE_BYTES, scene.primaryCompositeOwnedGlassPaneIdBase ?? 0xffff_ffff, scene.primaryCompositeOwnedGlassPaneCount ?? 0], SVO_DRY_SCENE_PARAMS_LAYOUT.terrainWordOffset);
    if (scene.terrainMaterialMetadata) words.set(scene.terrainMaterialMetadata, SVO_DRY_SCENE_PARAMS_LAYOUT.terrainMaterialWordOffset);
    const visibilityFlags = (scene.contactVisibilityEnabled ? SVO_DRY_VISIBILITY_FLAGS.exactContact : 0)
      | (scene.shadowVisibilityEnabled === false ? 0 : SVO_DRY_VISIBILITY_FLAGS.exactShadow)
      | (this.lightingMode === "cone" ? SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested : 0);
    words.set([pbrMaterials.count, pbrMaterials.revision, pbrMaterials.strideBytes, visibilityFlags], SVO_DRY_SCENE_PARAMS_LAYOUT.materialPublicationWordOffset);
    const fluidDimensions = structural.domain.dimensionsCells ?? [0, 0, 0];
    words.set([...fluidDimensions, svoFluidPrimaryModeWord(scene.fluidPrimaryMode, scene.directFluidMediaEndToEndValidated)], SVO_DRY_SCENE_PARAMS_LAYOUT.fluidDomainWordOffset);
    const candidates = this.primitiveCandidateArena;
    words.set(candidates ? [
      candidates.candidateRecordOffset,
      candidates.candidateNodeCount,
      candidates.candidateRootNodeIndex,
      candidates.candidateVersion,
    ] : [0, 0, 0, 0], SVO_DRY_SCENE_PARAMS_LAYOUT.primitiveCandidateWordOffset);
    if (this.fineFluidCapability) words.set(this.fineFluidCapability.paramsWords, SVO_DRY_SCENE_PARAMS_LAYOUT.finePhiWordOffset);
    const nodeMip = source.nodeMipPyramid;
    if (nodeMip && nodeMip.generation > 0 && nodeMip.plan.complete) {
      words.set([nodeMip.generation, nodeMip.plan.pages.length, Math.max(1, ...nodeMip.plan.pages.map((page) => page.key.level + 1)), 1], SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipWordOffset);
      words.set([...nodeMip.plan.atlas.texels, 0], SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipAtlasWordOffset);
    }
    if (this.paramsWords?.length === words.length && words.every((word, index) => word === this.paramsWords![index])) return;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    this.paramsWords = Uint32Array.from(words);
  }

  private rebuild(): void {
    const source = this.source, structural = source?.structural;
    if (!this.layout || !this.pipeline || !source || !structural || !this.primitiveBuffer || !this.glassBuffer || !this.scene) { this.bindGroup = undefined; return; }
    const nodeMip = source.nodeMipPyramid;
    this.bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.bodyBuffer } },
      { binding: 2, resource: structural.control }, { binding: 3, resource: structural.nodes }, { binding: 4, resource: structural.leaves },
      { binding: 5, resource: structural.materialOwners }, { binding: 6, resource: source.pbrMaterials!.binding }, { binding: 7, resource: { buffer: this.primitiveBuffer } },
      { binding: 8, resource: this.fineFluidCapability?.arena ?? structural.publication.state }, { binding: 9, resource: { buffer: this.paramsBuffer } },
      { binding: 10, resource: { buffer: this.glassBuffer } },
      { binding: 11, resource: structural.geometry }, { binding: 12, resource: structural.fluidLeafStates },
      { binding: 13, resource: { buffer: this.lightingBuffer } },
      { binding: 14, resource: { buffer: this.rigidMotionUniformBuffer } },
      { binding: 15, resource: { buffer: this.thickGlassUniformBuffer } },
      { binding: 16, resource: nodeMip?.view ?? this.nodeMipFallbackAtlasView },
      { binding: 17, resource: nodeMip?.sampler ?? this.nodeMipFallbackSampler },
      { binding: 18, resource: nodeMip?.directoryView ?? this.nodeMipFallbackDirectoryView },
    ] });
  }

  /** GPU-authored storage is copied into this pass's uniform mirror to preserve the ten-storage adapter budget. */
  setRigidMotionSource(source: GPUBuffer | undefined): void {
    if (!source && this.rigidMotionSource) this.device.queue.writeBuffer(this.rigidMotionUniformBuffer, 0, new Uint32Array(SVO_DRY_RIGID_MOTION_UNIFORM_BYTES / 4));
    this.rigidMotionSource = source;
  }

  /** The owner uses this to suppress legacy fluid interfaces before enabling the diagnostic path. */
  get fluidPrimaryMode(): SvoFluidPrimaryMode {
    return this.scene?.fluidPrimaryMode ?? DEFAULT_SVO_FLUID_PRIMARY_MODE;
  }

  get temporalCellSize_m(): number {
    const cellSize = this.source?.structural?.domain.cellSize_m;
    return cellSize ? Math.min(...cellSize) : 0;
  }

  /** Atomic presentation ownership; direct media can never coexist with legacy water stages. */
  get fluidRenderOwnership(): SvoFluidRenderOwnership {
    return resolveSvoFluidRenderOwnership(
      this.scene?.fluidPrimaryMode === "direct-structural-media" ? "direct-structural-media" : "legacy-water",
      this.scene?.directFluidMediaEndToEndValidated === true,
    );
  }

  ensureSize(width: number, height: number): void {
    if (this.gBufferTargets.ensureSize(width, height)) { this.pickingFrameToken += 1; this.lastPickingTarget = undefined; this.clearReusableFrame(); }
    this.temporalAccumulator.ensureSize(width, height);
  }

  invalidateTemporalHistory(): void { this.temporalAccumulator.invalidate(); }

  private lastTemporalEncoded = false;
  get temporalEncodedLastFrame(): boolean { return this.lastTemporalEncoded; }

  /** Auxiliary MRTs and reversed-Z depth for future temporal/picking consumers. */
  get gBufferTextures(): SparseVoxelGBufferTextures | undefined {
    return this.gBufferTargets.textures;
  }

  async pickGBuffer(
    normalizedX: number,
    normalizedY: number,
    rayOrigin_m: readonly [number, number, number],
    rayDirection: readonly [number, number, number],
    rigidBodyCount: number,
  ): Promise<SvoGpuPickingReadbackResult> {
    const gBuffer = this.gBufferTargets.textures, radianceDepth = this.lastPickingTarget;
    if (!gBuffer || !radianceDepth || !this.source?.pbrMaterials) return { status: "invalid", reason: "generation" };
    const pixel = svoPickingPixelFromNormalized(normalizedX, normalizedY, gBuffer.width, gBuffer.height);
    if (!pixel) return { status: "invalid", reason: "coordinates" };
    const frameToken = this.pickingFrameToken;
    return this.pickingReadback.pick(radianceDepth, gBuffer, {
      pixelX: pixel[0], pixelY: pixel[1], rayOrigin_m, rayDirection,
      rigidBodyCount, materialCount: this.source.pbrMaterials.count, frameToken,
    }, () => this.pickingFrameToken === frameToken && this.lastPickingTarget === radianceDepth);
  }

  encode(encoder: GPUCommandEncoder, target: GPUTexture | GPUTextureView, timestampWrites?: TimestampRange, temporalFrame?: SparseVoxelTemporalFrameState, temporalTimestampWrites?: TimestampRange, reuseKey?: string): DrySceneReplacementResult | false {
    this.lastTemporalEncoded = false;
    if (!this.pipeline || !this.bindGroup) return false;
    const gBufferViews = this.gBufferTargets.views;
    if (!gBufferViews) return false;
    const targetTexture = "width" in target ? target as GPUTexture : undefined;
    const requiredStableFrames = temporalFrame ? 2 : 1;
    if (reuseKey && targetTexture && reuseKey === this.reusableKey && targetTexture === this.reusableTarget
      && this.reusableStableFrames >= requiredStableFrames && this.reusableResult) {
      if (timestampWrites) { const pass = encoder.beginComputePass({ label: "Sparse voxel dry scene reuse timestamp", timestampWrites }); pass.end(); }
      if (temporalFrame && temporalTimestampWrites) { const pass = encoder.beginComputePass({ label: "Sparse voxel temporal reuse timestamp", timestampWrites: temporalTimestampWrites }); pass.end(); }
      this.lastTemporalEncoded = Boolean(temporalFrame);
      return this.reusableResult;
    }
    let targetView = target as GPUTextureView;
    if (targetTexture) {
      targetView = this.targetViews.get(targetTexture) ?? targetTexture.createView();
      this.targetViews.set(targetTexture, targetView);
    }
    if (this.rigidMotionSource) encoder.copyBufferToBuffer(this.rigidMotionSource, 0, this.rigidMotionUniformBuffer, 0, SVO_DRY_RIGID_MOTION_UNIFORM_BYTES);
    const pass = encoder.beginRenderPass({
      label: "Sparse voxel dry scene",
      colorAttachments: [
        { view: targetView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        { view: gBufferViews.packedSurface, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        { view: gBufferViews.identityMedia, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      ],
      depthStencilAttachment: {
        view: gBufferViews.hardwareDepth,
        depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
      ...(timestampWrites ? { timestampWrites } : {}),
    });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.bindGroup); pass.draw(3); pass.end();
    this.lastPickingTarget = targetTexture;
    const gBuffer = this.gBufferTargets.textures;
    const temporalResolve = temporalFrame && targetTexture && gBuffer
      ? this.temporalAccumulator.encode(encoder, targetTexture, gBuffer, temporalFrame, temporalTimestampWrites)
      : false;
    this.lastTemporalEncoded = Boolean(temporalResolve);
    const result = { encoded: true, sampledTargetView: temporalResolve ? temporalResolve.resolvedView : targetView } as const;
    if (reuseKey && targetTexture) {
      this.reusableStableFrames = reuseKey === this.reusableKey && targetTexture === this.reusableTarget ? this.reusableStableFrames + 1 : 1;
      this.reusableKey = reuseKey; this.reusableTarget = targetTexture; this.reusableResult = result;
    } else this.clearReusableFrame();
    return result;
  }

  destroy(): void {
    this.primitiveBuffer?.destroy();
    this.glassBuffer?.destroy();
    this.paramsBuffer.destroy();
    this.lightingBuffer.destroy();
    this.rigidMotionUniformBuffer.destroy();
    this.thickGlassUniformBuffer.destroy();
    this.nodeMipFallbackAtlas.destroy();
    this.nodeMipFallbackDirectory.destroy();
    this.gBufferTargets.destroy();
    this.temporalAccumulator.destroy();
    this.pickingReadback.destroy();
    this.lastPickingTarget = undefined;
    this.clearReusableFrame();
    this.pickingFrameToken += 1;
    this.bindGroup = undefined;
    this.glassBuffer = undefined;
    this.glassCacheKey = undefined;
    this.paramsWords = undefined;
  }

  private clearReusableFrame(): void {
    this.reusableKey = undefined; this.reusableStableFrames = 0; this.reusableTarget = undefined; this.reusableResult = undefined;
  }
}

export const svoDrySceneShader = drySceneShader;
