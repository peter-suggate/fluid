import { svoPrimitiveWGSL, SVO_PRIMITIVE_RECORD_STRIDE_BYTES } from "./svo-primitive-abi";
import {
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
  packSvoPrimitiveCandidateArena,
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
import { svoThinGlassWGSL, SVO_THIN_GLASS_RECORD_STRIDE_BYTES, SVO_THIN_GLASS_RECORD_WORDS } from "./svo-thin-glass";
import {
  SVO_RIGID_RASTER_CONTRACT,
  svoRigidRasterCoverageBridgeBindGroupLayoutEntries,
  svoRigidRasterInputBindGroupLayoutEntries,
  svoRigidRasterShader,
} from "./webgpu-svo-rigid-raster";
import {
  SVO_CONE_FANOUT_CONTRACT,
  createSvoConeFanoutWorkerWGSL,
  packSvoConeFanoutFrame,
  svoConeFanoutReducerBindGroupLayoutEntries,
  svoConeFanoutReducerWGSL,
  svoConeFanoutSceneBindGroupLayoutEntries,
  svoConeFanoutWorkerBindGroupLayoutEntries,
} from "./webgpu-svo-cone-fanout";
import {
  SVO_THICK_GLASS_RECORD_STRIDE_BYTES,
  svoThickGlassWGSL,
  unpackSvoThickGlassVolumes,
} from "./svo-thick-glass";
import { SVO_VISIBILITY_LIMITS, svoVisibilityRaysWGSL } from "./svo-visibility-rays";
import { terrainHeightAt, terrainNormalAt, type TerrainDescription } from "./terrain";
import { unifiedLightingShaderLibrary, WATER_OPTICS } from "./webgpu-lighting";
import { createWebgpuSvoTraversalWGSL } from "./webgpu-svo-traversal";
import { createWebgpuSvoCompactTraversalWGSL, resolveWebGpuSvoCompactHierarchy } from "./webgpu-svo-compact-hierarchy";
import {
  createWebgpuSvoWideFanoutTraversalWGSL,
  resolveSvoWideTraversalCapability,
} from "./webgpu-svo-wide-fanout";
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
import { SPARSE_VOXEL_VALID_FIELDS, type SparseVoxelSceneRenderSource } from "./webgpu-voxel-debug";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoLightingOptions } from "./svo-render-options";
import type { DrySceneReplacementResult, RenderPathTracePhase } from "./webgpu-water-pipeline";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { svoFluidCoverageWGSL } from "./svo-fluid-coverage";
import type { WebGpuSvoFluidCoverage } from "./webgpu-svo-fluid-coverage";
import { svoNodeMipSamplingWGSL } from "./svo-node-mip-sampling";
import { svoTetrahedralRadianceWGSL } from "./svo-tetrahedral-radiance";
import { svoTetrahedralRadianceConeCoreWGSL } from "./svo-tetrahedral-radiance-cone";
import { svoCostOverlayCode } from "./svo-render-diagnostics";
import { svoBrickOccupancyWGSL } from "./svo-brick-occupancy";
import { createSvoScreenSpaceTraversalWGSL } from "./svo-screen-space-termination";
import {
  DEFAULT_SVO_RENDER_TUNING,
  SVO_CONE_RADIANCE_RECONSTRUCTION_CODES,
  SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
  normalizeSvoRenderTuning,
  type SvoRenderTuning,
} from "./svo-render-tuning";
import {
  SparseVoxelPixelTraceBuffers,
  createSvoPixelTraceProbeWGSL,
  type SvoPixelTraceProbeOptions,
} from "./webgpu-svo-pixel-trace";
import type { SvoPixelTrace } from "./svo-pixel-trace";

export interface SparseVoxelDrySceneData {
  /** Packed `SvoPrimitiveRecord` values in dense environment-owner order. */
  primitiveRecords: Uint32Array<ArrayBuffer>;
  /** @deprecated Offline audit index; it is never uploaded or consumed by rendering. */
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
  /** Scene capability gate for bounded indirect-diffuse contact visibility. */
  contactVisibilityEnabled?: boolean;
  /** Scene capability gate for shadow visibility; omission keeps shadows available. */
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
  // Evolving fluid coverage. Sampled, like the node-mip atlas, so water shadows
  // cost the fragment stage a texture unit rather than another storage buffer.
  { binding: 19, type: "texture-3d-float" as const },
  // Direct node-mip page table. A sampled r32uint texture preserves the
  // fragment-stage storage-buffer ceiling while replacing directory searches.
  { binding: 20, type: "texture-3d-uint" as const },
  ...[21, 22, 23, 24].map((binding) => ({ binding, type: "texture-3d-float" as const })),
  // Exact zero-radiance certificate by physical page slot. Keeping this in a
  // sampled uint texture avoids another fragment-stage storage buffer.
  { binding: 25, type: "texture-2d-uint" as const },
] as const);

export function sparseVoxelDrySceneBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  // The compact 2x2 kernels need traversal, node-mip, lighting, and rigid-body
  // inputs, but not material shading, glass, or dormant traversal variants.
  // Keeping those fragment-only also stays below WebGPU's per-stage storage
  // binding limit on Apple GPUs.
  const computeBindings = new Set([0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
  // Raster analytic impostors consume only the camera uniform and their
  // source record arena in the vertex stage.
  const vertexBindings = new Set([0, 1, 10]);
  return SVO_DRY_SCENE_BINDING_CONTRACT.map(({ binding, type }): GPUBindGroupLayoutEntry => {
    const visibility = GPUShaderStage.FRAGMENT
      | (computeBindings.has(binding) ? GPUShaderStage.COMPUTE : 0)
      | (vertexBindings.has(binding) ? GPUShaderStage.VERTEX : 0);
    if (type === "texture-3d-float") return { binding, visibility, texture: { sampleType: "float", viewDimension: "3d" } };
    if (type === "texture-3d-uint") return { binding, visibility, texture: { sampleType: "uint", viewDimension: "3d" } };
    if (type === "texture-2d-uint") return { binding, visibility, texture: { sampleType: "uint", viewDimension: "2d" } };
    if (type === "filtering-sampler") return { binding, visibility, sampler: { type: "filtering" } };
    return { binding, visibility, buffer: { type } };
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

/** Packed dry-scene parameters. */
export const SVO_DRY_SCENE_PARAMS_LAYOUT = Object.freeze({
  sizeBytes: 512,
  terrainWordOffset: 24,
  terrainMaterialWordOffset: 28,
  materialPublicationWordOffset: 32,
  nodeMipWordOffset: 36,
  nodeMipAtlasWordOffset: 40,
  wideFanoutWordOffset: 44,
  nodeMipLevelStartWordOffset: 48,
  nodeMipOriginWordOffset: 60,
  /** Packed SvoFluidCoverageFrame; the 256-byte block above is fully spoken for. */
  fluidCoverageWordOffset: 64,
  /** Five vec4 lanes of bounded runtime rendering controls. */
  tuningWordOffset: 76,
  /** xyz: packed direct-table extent; w: published/usable. */
  nodeMipDirectWordOffset: 96,
  /** Twelve constant-indexed Z-slab offsets, one per supported node-mip level. */
  nodeMipDirectLevelZWordOffset: 100,
  /** x: matching radiance generation; y: complete and usable; zw: reserved. */
  tetrahedralRadianceWordOffset: 112,
  /** xyz: complete sparse-lighting world extent in metres. */
  nodeMipExtentWordOffset: 116,
  /** xyzw: GI bounce, broad occlusion, diffuse environment, direct key. */
  giLightingWordOffset: 120,
  /** xy: GI aperture and cone count; zw reserved. */
  giConesWordOffset: 124,
} as const);

/** materialPublication.w flags shared by the direct and derived-lighting paths. */
export const SVO_DRY_VISIBILITY_FLAGS = Object.freeze({
  exactContact: 1 << 0,
  exactShadow: 1 << 1,
  coneLightingRequested: 1 << 2,
  ambientOcclusion: 1 << 3,
  globalIllumination: 1 << 4,
  globalIlluminationOcclusion: 1 << 5,
  globalIlluminationRequested: 1 << 6,
} as const);

export const SVO_TERRAIN_FAST_MIN_VERTICAL = 0.35;
export const SVO_TERRAIN_FAST_BRACKET_STEPS = 2;
export const SVO_TERRAIN_FAST_REFINEMENTS = 5;
export const SVO_TERRAIN_FALLBACK_STEPS = 20;
export const SVO_TERRAIN_FALLBACK_REFINEMENTS = 8;
/** Includes four terrain-height evaluations used by the central-difference normal. */
export const SVO_TERRAIN_FAST_MAX_HEIGHT_EVALUATIONS = 12;
/** Normal-projected sparse-cell widths used to offset the hard shadow ray. */
/** Reversed-Z near plane the dry pass writes device depth against. */
export const SVO_DRY_SCENE_REVERSED_Z_NEAR_M = 0.01;
export const SVO_DRY_SCENE_SHADOW_BIAS_CELLS = 0.02;
/**
 * Sparse-cell widths a shadow cone's origin escapes along the geometric normal
 * before marching. The node-mip march samples trilinear coverage, so a cone
 * that starts on the receiving surface (the hard-ray 0.02-cell bias) reads the
 * receiver's own solid coverage for its first steps; the accumulated
 * self-occlusion changes in visible bands with grazing angle and light
 * distance. Half a voxel clears the origin's own trilinear support without
 * visibly detaching contact shadows (0.25 still banded on grazing terrain,
 * 0.75 measurably lifted mushroom-stem contact shadows).
 */
export const SVO_DRY_SCENE_CONE_SHADOW_NORMAL_ESCAPE_CELLS = 0.5;
/**
 * Fine cells cleared between a FINITE emitter's near surface and a shadow
 * cone's march endpoint. The march used to end exactly at the emitter surface,
 * so the last samples' trilinear/mip support read the emitter's own voxelized
 * solid coverage, and the accumulated amount aliased with the receiver's
 * distance modulo the step size (concentric rings around point lights, plus a
 * hard-edged bright disc where the march was skipped entirely). The clearance
 * is a FIXED cell count so the endpoint - and with it the light-anchored
 * ladder the marcher walks over the far half of the cone - stays world-locked
 * around the emitter for every receiver.
 */
export const SVO_DRY_SCENE_CONE_EMITTER_CLEARANCE_CELLS = 3;
/**
 * fract(lod) width of the transition band in which the cone marcher blends the
 * two bracketing mip levels; below the band a single fine-level fetch suffices.
 * The concentric-ring artifact came from C0 discontinuity at integer LOD
 * switches, not from lack of full-range blending, so the band's blend weight
 * ramps 0 at the band start to 1 at fract==1 (where it equals the next level's
 * band-start value): coverage stays continuous at both band edges while ~70%
 * of steps skip the second atlas fetch and its directory/page-cache work.
 * Measured (M1 Max, garden 1280x720): full-range blending cost scale-1
 * 40.6 ms / scale-0.5 16.0 ms; width 0.3 recovers most of the two-fetch
 * regression with no visible banding (0.5 measured within noise of 0.3).
 */
export const SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH = 0.3;
/** Bound direct-light work independently from the producer's 32-record capacity. */
export const SVO_DRY_SCENE_MAX_SHADED_LIGHTS = 8;
/** Two fixed shape samples are stable across frames and keep total visibility work bounded. */
export const SVO_DRY_SCENE_AREA_LIGHT_SAMPLES = 2;
/**
 * Ambient-occlusion cones traced per receiver while the camera is moving
 * (the SVO_CAMERA_CHANGING_FRAME sentinel in uniforms.viewport.w), against
 * SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES once the view settles.
 *
 * One cone rather than zero: AO stays PRESENT while moving, so settling
 * changes only the estimate's noise, not whether the ambient term exists at
 * all. Measured on the garden scene (M1 Max, 1280x720, cone scale 0.5, via
 * FLUID_SVO_DRY_FRAME_CAMERA_MOVING=1), as relative luminance of the moving
 * frame against the settled frame:
 *   - one cone:    mean 0.0015, p95 0.0072, 0.01% of lit pixels past 10%;
 *   - AO disabled: mean 0.0095, p95 0.0645, 3.0%  of lit pixels past 10%.
 * Disabling AO is ~1.1 ms cheaper again but its error is not diffuse noise: it
 * lands in contiguous patches on cap undersides, stem/cap junctions, and
 * object-to-ground contacts — exactly the shading that reads as objects
 * resting on the terrain — so every settle would pop those regions darker.
 * One cone keeps that error at the 0.01% level, which is invisible.
 */
export const SVO_DRY_SCENE_MOVING_AO_CONE_SAMPLES = 1;
export const SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES = 4;
/**
 * Area-light shape samples while the camera is moving. Shadows stay present
 * at every tier — losing them during motion is far more visible than a
 * slightly harder penumbra — so motion only collapses the area light's two
 * fixed shape samples to its centre sample, which softens the penumbra edge
 * without moving the shadow body. Worth 0.26 ms of the moving tier's 1.44 ms
 * saving on the garden scene at cone scale 0.5.
 *
 * Reducing the cone marchers' step budget was considered and rejected: an
 * exhausted budget returns the partially accumulated transmittance, so long
 * shadow cones would lighten mid-march and the shadow body itself would shift
 * on every settle rather than only its penumbra.
 */
export const SVO_DRY_SCENE_MOVING_AREA_LIGHT_SAMPLES = 1;
/**
 * WGSL predicate for "the camera has settled": the renderer publishes
 * SVO_CAMERA_CHANGING_FRAME (-2) into uniforms.viewport.w while the camera is
 * moving and -1 when settled. Kept as one shared expression so every quality
 * tier switches on the identical test.
 */
export const SVO_DRY_SCENE_CAMERA_SETTLED_WGSL = "uniforms.viewport.w>=-1.0";
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
 * Structural fields a static primary ray needs before it may leave the camera.
 * `traceStatic` and `svoVisibilityNext` both refuse to trace until the producer
 * has published these, so a producer that allocates the structural source but
 * never finalizes a publication renders every static surface as a miss —
 * analytic glass and rigid bodies keep drawing, and nothing else does.
 */
export const SVO_DRY_SCENE_REQUIRED_VALID_FIELDS =
  SPARSE_VOXEL_VALID_FIELDS.topology
  | SPARSE_VOXEL_VALID_FIELDS.staticGeometry
  | SPARSE_VOXEL_VALID_FIELDS.materialOwner;

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
    const environmentLighting = buildSvoEnvironmentLighting(scene.environment ?? "default", environment.revision, scene.lighting?.environment);
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

/** @deprecated Offline validation helper; this result never gates the SVO renderer. */
export function canConsumeSparseVoxelPrimitiveCandidates(scene: SparseVoxelDrySceneData | undefined): boolean {
  const primitiveCount = scene?.primitiveRecords.byteLength
    ? scene.primitiveRecords.byteLength / SVO_PRIMITIVE_RECORD_STRIDE_BYTES
    : 0;
  if (primitiveCount > SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES) return true;
  if (!scene?.primitiveCandidates) return false;
  try {
    packSvoPrimitiveCandidateArena(scene.primitiveRecords, scene.primitiveCandidates);
    return true;
  } catch {
    return false;
  }
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
    && (scene.glassRecords?.byteLength ?? 0) % SVO_THIN_GLASS_RECORD_STRIDE_BYTES === 0
    && (scene.glassRecords?.byteLength ?? 0) / SVO_THIN_GLASS_RECORD_STRIDE_BYTES <= SVO_SCENE_GLASS_MAXIMUM_PANES
    && (scene.terrainMaterialMetadata === undefined || scene.terrainMaterialMetadata.byteLength === SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES)
    && source.structural.fields.topology.residency !== "unavailable"
    && source.structural.fields.staticGeometry.residency !== "unavailable"
    && source.structural.fields.materialOwner.residency !== "unavailable"
  );
}

/** Feature toggles for the cone-traced node-mip marcher. Production enables every optimization. */
export interface SvoDryConeMarcherOptions {
  /** Branchless Morton bit-spread encode instead of the 21-iteration reference loop. */
  branchlessMorton?: boolean;
  /** Restrict the directory binary search to the queried level's contiguous row range. */
  rangedDirectorySearch?: boolean;
  /** Substitute provably zero coverage without fetching when inside a certified empty region. */
  emptySpaceElision?: boolean;
  /**
   * Accumulate an optical path length through evolving fluid alongside solid
   * coverage. Water is dielectric, not an occluder: turning its coverage into
   * opacity would render a pond as a hole. The cone therefore reports metres of
   * water traversed and leaves the wavelength-dependent attenuation to the
   * caller, which is the same Beer-Lambert term the raster composite already
   * applies along the view ray.
   *
   * Only shadow cones pay for the fetch. Ambient-occlusion cones pass a zero
   * surface normal and skip it, so contact darkening keeps its exact cost.
   */
  fluidCoverage?: boolean;
  /** Resolve a virtual page with one r32uint texture load before directory fallback. */
  directPageTable?: boolean;
}

/**
 * Morton/find/`dryNodeMipAt`/`dryConeVisibility` marcher block shared by the
 * production dry shader and the A/B cone benchmark. Every variant is bit-exact:
 * optimizations may only change how a value is computed, never the value.
 * Requires bindings/declarations named `dry` (DryParams), `publicationState`,
 * `nodeMipAtlas`, `nodeMipSampler`, `nodeMipDirectory`, `nodeMipPageTable`, the private counter
 * `dryMipSteps`, and the `svoNodeMipSamplingWGSL` library.
 */
export function createSvoDryConeMarcherWGSL(options: SvoDryConeMarcherOptions = {}): string {
  const morton = options.branchlessMorton
    ? /* wgsl */ `fn dryNodeMipSpreadMortonBits(value:vec3u)->vec3u{
  var spread=value;
  spread=(spread|(spread<<vec3u(16u)))&vec3u(0xff0000ffu);
  spread=(spread|(spread<<vec3u(8u)))&vec3u(0x0f00f00fu);
  spread=(spread|(spread<<vec3u(4u)))&vec3u(0xc30c30c3u);
  spread=(spread|(spread<<vec3u(2u)))&vec3u(0x49249249u);
  return spread;
}
fn dryNodeMipMorton(coordinate:vec3u)->vec2u{
  let masked=coordinate&vec3u(0x1fffffu);
  let low=dryNodeMipSpreadMortonBits(vec3u(masked.x&0x7ffu,masked.y&0x7ffu,masked.z&0x3ffu));
  let high=dryNodeMipSpreadMortonBits(vec3u(masked.x>>11u,masked.y>>11u,masked.z>>10u));
  return vec2u(low.x|(low.y<<1u)|(low.z<<2u),(high.x<<1u)|(high.y<<2u)|high.z);
}`
    : /* wgsl */ `fn dryNodeMipMorton(coordinate:vec3u)->vec2u{
  var result=vec2u(0u);for(var bit=0u;bit<21u;bit+=1u){for(var axis=0u;axis<3u;axis+=1u){let outputBit=bit*3u+axis;let value=(coordinate[axis]>>bit)&1u;if(outputBit<32u){result.x|=value<<outputBit;}else{result.y|=value<<(outputBit-32u);}}}return result;
}`;
  // Directory rows are sorted by (level, morton), so each level occupies one
  // contiguous run; a lower_bound over that run equals the full-range result.
  // Constant vector indexing only: a dynamically indexed uniform array trips a
  // slow-path Tint/Metal transform that taxes the whole fragment shader.
  const levelStart = options.rangedDirectorySearch
    ? /* wgsl */ `fn dryNodeMipLevelStart(level:u32)->u32{
  let clamped=min(level,11u);
  let word=select(select(dry.nodeMipLevelStart[0],dry.nodeMipLevelStart[1],clamped>=4u),dry.nodeMipLevelStart[2],clamped>=8u);
  let lane=clamped&3u;
  return select(select(select(word.x,word.y,lane==1u),word.z,lane==2u),word.w,lane==3u);
}
`
    : "";
  const searchRange = options.rangedDirectorySearch
    ? /* wgsl */ `var low=dryNodeMipLevelStart(level);var high=select(dry.nodeMip.y,dryNodeMipLevelStart(level+1u),level<11u);`
    : /* wgsl */ `var low=0u;var high=dry.nodeMip.y;`;
  const directPageTable = options.directPageTable ? /* wgsl */ `fn dryNodeMipDirectLevelZ(level:u32)->u32{
  let clamped=min(level,11u);
  let word=select(select(dry.nodeMipDirectLevelZ[0],dry.nodeMipDirectLevelZ[1],clamped>=4u),dry.nodeMipDirectLevelZ[2],clamped>=8u);
  let lane=clamped&3u;
  return select(select(select(word.x,word.y,lane==1u),word.z,lane==2u),word.w,lane==3u);
}
fn dryNodeMipDirectFind(level:u32,coordinate:vec3u)->u32{
  if(dry.nodeMipDirect.w==0u||level>=dry.nodeMip.z){return 0xffffffffu;}
  let zStart=dryNodeMipDirectLevelZ(level);
  let zEnd=select(dry.nodeMipDirect.z,dryNodeMipDirectLevelZ(level+1u),level+1u<dry.nodeMip.z);
  if(coordinate.x>=dry.nodeMipDirect.x||coordinate.y>=dry.nodeMipDirect.y||coordinate.z>=zEnd-zStart){return 0xffffffffu;}
  let encoded=textureLoad(nodeMipPageTable,vec3i(vec3u(coordinate.x,coordinate.y,zStart+coordinate.z)),0).x;
  return select(0xffffffffu,encoded-1u,encoded!=0u);
}
` : "";
  const directFind = options.directPageTable
    ? /* wgsl */ `if(dry.nodeMipDirect.w!=0u){return dryNodeMipDirectFind(level,coordinate);}`
    : "";
  const pageOrigin = options.directPageTable
    ? /* wgsl */ `if(dry.nodeMipDirect.w!=0u){let physical=u32(SVO_NODE_MIP_PHYSICAL_SIZE);let atlasPages=max(dry.nodeMipAtlas.xyz/vec3u(physical),vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));*pageCache=DryNodeMipPageCache(pageCoordinate,level,atlasPage*physical,dry.nodeMip.x,1u,pageIndex,0u);}else{let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,pageIndex);*pageCache=DryNodeMipPageCache(pageCoordinate,level,entry.pageOrigin,entry.generation,1u,pageIndex,0u);}`
    : /* wgsl */ `let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,pageIndex);*pageCache=DryNodeMipPageCache(pageCoordinate,level,entry.pageOrigin,entry.generation,1u,pageIndex,0u);`;
  const zeroRegion = options.emptySpaceElision
    ? /* wgsl */ `struct DryConeZeroRegion{minimum:vec3f,maximum:vec3f,valid:u32}
fn dryConeZeroRegionAt(position_m:vec3f,level:u32,pageCache:ptr<function,DryNodeMipPageCache>)->DryConeZeroRegion{
  // A cached page key is only trustworthy once its coordinate is recomputed in
  // range: the dryNodeMipAt out-of-range early return leaves the cache stale.
  let levelWidth=dry.mapping.cellSize*exp2(f32(level));
  let levelVoxel=(position_m-dry.nodeMipOrigin.xyz)/levelWidth;
  let levelPageFloor=floor(levelVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(levelPageFloor<vec3f(0.0))||any(levelPageFloor>=vec3f(2097152.0))
    ||(*pageCache).generation!=dry.nodeMip.x||(*pageCache).level!=level||any((*pageCache).coordinate!=vec3u(levelPageFloor))||(*pageCache).resident!=0u){
    return DryConeZeroRegion(vec3f(0.0),vec3f(0.0),0u);
  }
  // Non-resident page: no page means no atlas content, so every sample whose
  // trilinear support sits inside this page extent is exactly zero.
  var region=DryConeZeroRegion(
    dry.nodeMipOrigin.xyz+levelPageFloor*f32(SVO_NODE_MIP_INTERIOR_SIZE)*levelWidth,
    dry.nodeMipOrigin.xyz+(levelPageFloor+vec3f(1.0))*f32(SVO_NODE_MIP_INTERIOR_SIZE)*levelWidth,1u);
  let coarseLevel=min(level+2u,dry.nodeMip.z-1u);
  if(coarseLevel>level){
    // Directory-only coarse upgrade (no texture fetch): a non-resident coarse
    // page has no resident descendants via the ancestor-residency chain, so the
    // whole coarse page extent is zero.
    let coarseWidth=dry.mapping.cellSize*exp2(f32(coarseLevel));
    let coarsePageFloor=floor((position_m-dry.nodeMipOrigin.xyz)/(coarseWidth*f32(SVO_NODE_MIP_INTERIOR_SIZE)));
    if(dryNodeMipFind(coarseLevel,vec3u(coarsePageFloor))==0xffffffffu){
      region=DryConeZeroRegion(
        dry.nodeMipOrigin.xyz+coarsePageFloor*f32(SVO_NODE_MIP_INTERIOR_SIZE)*coarseWidth,
        dry.nodeMipOrigin.xyz+(coarsePageFloor+vec3f(1.0))*f32(SVO_NODE_MIP_INTERIOR_SIZE)*coarseWidth,1u);
    }
  }
  return region;
}
`
    : "";
  // Both variants march the identical continuous-LOD cone: the step width
  // follows the continuous cone diameter, and coverage is C0-continuous in
  // lod. A single floor(lod) fetch with a floored step width made accumulated
  // opacity jump wherever floor(lod) incremented along the cone, which
  // rendered as concentric isodistance rings around point lights. Continuity
  // is restored by blending the two bracketing mip levels — but only inside
  // the trailing fract(lod) transition band (SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH
  // wide): the blend weight ramps from 0 at the band start to 1 at
  // fract(lod)==1, where the blended value equals the next level's band-start
  // value, so coverage is continuous at both band edges. Below the band a
  // single fine-level fetch suffices, which skips the second atlas fetch and
  // its directory/page-cache work on most steps (full-range blending doubled
  // per-step fetches everywhere and cost +35% frame time).
  //
  // surfaceNormal (zero to disable, as AO and the standalone benchmark do)
  // marks the receiver's tangent plane at the march origin: coverage whose
  // trilinear support still straddles that plane is the receiver's own
  // voxelized surface, and accumulating it self-shadows in bands that track
  // the sub-voxel phase of the analytic surface (terrain height isolines
  // rendered as concentric rings, latitude bands on mushroom caps). Each
  // sample's coverage is therefore scaled by its plane clearance over the
  // sample's own support width, ramping back to full occlusion by 24 fine
  // voxels of marched distance so genuine distant blockers keep their shadows.
  // Shadow cones (surfaceNormal set) refine their step width geometrically as
  // the march approaches its endpoint and fade the trailing 1.5 diameters:
  // every cone toward a light converges on the same emitter neighbourhood, so
  // with diameter-sized steps the number of samples landing inside geometry
  // near the endpoint (the receiver's distance modulo the step size) is
  // quantized, which rendered as concentric rings around point lights. Zeno
  // steps shrink the per-sample opacity near the endpoint until the banding
  // amplitude vanishes while the .25-voxel step floor bounds the extra work.
  // Water is measured, not composited: `stepWidth * coverage` is the length of
  // this step that lies inside liquid, so summing it over the march yields the
  // path length a light ray travels through water. No selfWeight here — a
  // receiver standing in a pond really is under water, and suppressing the
  // first samples would erase exactly the shading that makes it read as wet.
  const fluidAccumulation = options.fluidCoverage ? /* wgsl */ `if(shadowCone){fluidDepth_m+=stepWidth*svoFluidCoverageAt(fluidCoverageVolume,nodeMipSampler,dry.fluidCoverage,position,svoFluidCoverageLod(diameter,fluidTexel_m));}` : "";
  const fluidPrologue = options.fluidCoverage ? /* wgsl */ `var fluidDepth_m=0.0;let fluidTexel_m=max(dry.fluidCoverage.texelSize_m.x,max(dry.fluidCoverage.texelSize_m.y,dry.fluidCoverage.texelSize_m.z));` : "";
  const coneMiss = options.fluidCoverage ? "DryConeVisibility(1.0,0u,0.0)" : "DryConeVisibility(1.0,0u)";
  const coneResult = options.fluidCoverage
    ? "DryConeVisibility(clamp(transmittance,0.0,1.0),1u,max(fluidDepth_m,0.0))"
    : "DryConeVisibility(clamp(transmittance,0.0,1.0),1u)";
  const coneStruct = options.fluidCoverage
    ? "struct DryConeVisibility{transmittance:f32,valid:u32,fluidDepth_m:f32}"
    : "struct DryConeVisibility{transmittance:f32,valid:u32}";
  const stepWidthExpression = /* wgsl */ `let remaining=maximumDistance_m-distance;let stepWidth=min(diameter,remaining);`;
  const selfCoverageWeight = /* wgsl */ `var selfWeight=1.0;if(shadowCone){selfWeight=max(clamp(dot(position-origin_m,surfaceNormal)/(1.5*diameter)-1.0,0.0,1.0),clamp((distance-12.0*minimumVoxel)/(12.0*minimumVoxel),0.0,1.0))*clamp(remaining/(1.5*diameter),0.0,1.0);}`;
  const bandStart = 1 - SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH;
  const blendWeightExpression = /* wgsl */ `let blendWeight=clamp((fract(lod)-${bandStart})*${(1 / SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH).toFixed(8)},0.0,1.0);`;
  const coarseBlendedCoverage = /* wgsl */ `var coverage=max(lookup.sample.solidMean,lookup.sample.solidMaximum*.15);if(blendWeight>0.0){let lookupCoarse=dryNodeMipAt(position,lod+1.0,&pageCacheCoarse);if(lookupCoarse.valid==0u){return ${coneMiss};}coverage=mix(coverage,max(lookupCoarse.sample.solidMean,lookupCoarse.sample.solidMaximum*.15),blendWeight);}`;
  const blendedCoverage = /* wgsl */ `let conservativeCoverage=selfWeight*coverage;let alpha=svoNodeMipCoverageOpacity(conservativeCoverage,stepWidth/diameter);transmittance*=1.0-alpha;${fluidAccumulation}`;
  // Phase B: light-anchored geometric ladder over the far half of the march.
  // Phase A's sample grid is anchored at the receiver, so how many samples
  // land inside the mip-smeared coverage around the emitter (lamp globe, head,
  // pole) aliases with the receiver's distance modulo the local step width,
  // which rendered as concentric rings around point lights and latitude bands
  // on nearby caps. The ladder offsets are measured FROM the march endpoint
  // (a fixed clearance off the emitter surface), so its sample positions are
  // world-locked around the light for every receiver: coverage near the light
  // then varies only smoothly with direction and the rings vanish. Ordering is
  // nearest-to-light first so a shared budget exhaustion drops mid-air rungs.
  const emitterLadderWGSL = /* wgsl */ `
  if(anchored){var emitterOffset=minimumVoxel*3.0;
  for(var rung=0u;rung<48u&&budget>0u&&emitterOffset<maximumDistance_m-phaseSplit&&transmittance>.005;rung+=1u){budget-=1u;dryMipSteps+=1u;
    let distance=maximumDistance_m-emitterOffset;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);let remaining=emitterOffset;let stepWidth=emitterOffset*.5;let position=origin_m+direction*distance;
    let lookup=dryNodeMipAt(position,lod,&pageCache);if(lookup.valid==0u){return ${coneMiss};}
    ${selfCoverageWeight}${blendWeightExpression}${coarseBlendedCoverage}${blendedCoverage}emitterOffset*=1.5;}}
`;
  // The elision variant keeps the identical march (step distances, stepIndex
  // sequence, dryMipSteps accounting, termination) and only replaces the
  // fine-level fetch whose trilinear support is provably inside a zero region
  // with the arithmetically identical zero sample: max(0,0*.15)=0 contributes
  // nothing to the blend.
  const visibility = options.emptySpaceElision
    ? /* wgsl */ `fn dryConeVisibility(origin_m:vec3f,direction:vec3f,aperture:f32,maximumDistance_m:f32,surfaceNormal:vec3f,anchored:bool)->DryConeVisibility{
  if(!dryNodeMipReady()){return ${coneMiss};}let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let tangent=tan(aperture*.5);var distance=minimumVoxel*.75;var transmittance=1.0;${fluidPrologue}var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);var pageCacheCoarse=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);let shadowCone=dot(surfaceNormal,surfaceNormal)>.25;var budget=clamp(dry.tuningCounts0.y,1u,48u);let phaseSplit=select(maximumDistance_m,maximumDistance_m*.5,anchored);
  var zeroRegion=DryConeZeroRegion(vec3f(0.0),vec3f(0.0),0u);
  for(var stepIndex=0u;stepIndex<48u&&budget>0u&&distance<phaseSplit&&transmittance>.005;stepIndex+=1u){budget-=1u;dryMipSteps+=1u;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);${stepWidthExpression}
    let position=origin_m+direction*distance;let level=min(u32(max(floor(lod),0.0)),dry.nodeMip.z-1u);
    // Tap texels lie within 1.5 level-voxels of the sample position, so the
    // whole trilinear support footprint sits inside this conservative box.
    let supportRadius=1.5*dry.mapping.cellSize*exp2(f32(level));
    var lookup=DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);
    if(zeroRegion.valid==0u||any(position-supportRadius<zeroRegion.minimum)||any(position+supportRadius>zeroRegion.maximum)){
      lookup=dryNodeMipAt(position,lod,&pageCache);
      // Establish (or replace) a region only from a non-resident page and only
      // once the march has left the current region entirely: re-deriving the
      // same box would repeat its directory probe for nothing.
      if(pageCache.resident==0u&&(zeroRegion.valid==0u||any(position<zeroRegion.minimum)||any(position>zeroRegion.maximum))){
        zeroRegion=dryConeZeroRegionAt(position,level,&pageCache);
      }
    }
    if(lookup.valid==0u){return ${coneMiss};}
    // The zero region certifies levels at or below its establishment level
    // only, so the in-band coarse bracketing fetch always misses the region
    // and goes through the coarse page cache.
    ${selfCoverageWeight}${blendWeightExpression}${coarseBlendedCoverage}${blendedCoverage}distance+=max(stepWidth,minimumVoxel*.25);}${emitterLadderWGSL}
  return ${coneResult};
}`
    : /* wgsl */ `fn dryConeVisibility(origin_m:vec3f,direction:vec3f,aperture:f32,maximumDistance_m:f32,surfaceNormal:vec3f,anchored:bool)->DryConeVisibility{
  if(!dryNodeMipReady()){return ${coneMiss};}let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let tangent=tan(aperture*.5);var distance=minimumVoxel*.75;var transmittance=1.0;${fluidPrologue}var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);var pageCacheCoarse=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);let shadowCone=dot(surfaceNormal,surfaceNormal)>.25;var budget=clamp(dry.tuningCounts0.y,1u,48u);let phaseSplit=select(maximumDistance_m,maximumDistance_m*.5,anchored);
  for(var stepIndex=0u;stepIndex<48u&&budget>0u&&distance<phaseSplit&&transmittance>.005;stepIndex+=1u){budget-=1u;dryMipSteps+=1u;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);${stepWidthExpression}let position=origin_m+direction*distance;let lookup=dryNodeMipAt(position,lod,&pageCache);if(lookup.valid==0u){return ${coneMiss};}${selfCoverageWeight}${blendWeightExpression}${coarseBlendedCoverage}${blendedCoverage}distance+=max(stepWidth,minimumVoxel*.25);}${emitterLadderWGSL}
  return ${coneResult};
}`;
  return /* wgsl */ `struct DryNodeMipLookup{sample:SvoNodeMipSample,valid:u32}
struct DryNodeMipPageCache{coordinate:vec3u,level:u32,pageOrigin:vec3u,generation:u32,resident:u32,pageIndex:u32,blackRadiance:u32}
${morton}
fn dryNodeMipCompare(entry:SvoNodeMipDirectoryEntry,level:u32,morton:vec2u)->i32{
  if(entry.level<level){return -1;}if(entry.level>level){return 1;}if(entry.mortonHigh<morton.y){return -1;}if(entry.mortonHigh>morton.y){return 1;}if(entry.mortonLow<morton.x){return -1;}if(entry.mortonLow>morton.x){return 1;}return 0;
}
${directPageTable}${levelStart}fn dryNodeMipFind(level:u32,coordinate:vec3u)->u32{
  if(level>=dry.nodeMip.z||dry.nodeMip.y==0u){return 0xffffffffu;}${directFind}let morton=dryNodeMipMorton(coordinate);${searchRange}
  for(var iteration=0u;iteration<24u&&low<high;iteration+=1u){let middle=low+(high-low)/2u;let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,middle);let comparison=dryNodeMipCompare(entry,level,morton);if(comparison<0){low=middle+1u;}else{high=middle;}}
  if(low>=dry.nodeMip.y){return 0xffffffffu;}let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,low);if(entry.generation!=dry.nodeMip.x||dryNodeMipCompare(entry,level,morton)!=0){return 0xffffffffu;}return low;
}
fn dryNodeMipReady()->bool{return dry.nodeMip.w!=0u&&dry.nodeMip.x!=0u&&dry.nodeMip.x==publicationState[2]&&dry.nodeMip.y>0u&&dry.nodeMip.z>0u;}
fn dryNodeMipAt(position_m:vec3f,lodIn:f32,pageCache:ptr<function,DryNodeMipPageCache>)->DryNodeMipLookup{
  let level=min(u32(max(floor(lodIn),0.0)),dry.nodeMip.z-1u);let levelScale=exp2(f32(level));let virtualVoxel=(position_m-dry.nodeMipOrigin.xyz)/(dry.mapping.cellSize*levelScale);let pageFloor=floor(virtualVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);}let pageCoordinate=vec3u(pageFloor);
  if((*pageCache).generation!=dry.nodeMip.x||(*pageCache).level!=level||any((*pageCache).coordinate!=pageCoordinate)){
    *pageCache=DryNodeMipPageCache(pageCoordinate,level,vec3u(0u),dry.nodeMip.x,0u,0xffffffffu,0u);let pageIndex=dryNodeMipFind(level,pageCoordinate);
    if(pageIndex!=0xffffffffu){${pageOrigin}}
  }
  if((*pageCache).resident==0u){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);}let local=virtualVoxel-vec3f(pageCoordinate)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);return DryNodeMipLookup(svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,(*pageCache).pageOrigin,local),1u);
}
${zeroRegion}${coneStruct}
${visibility}`;
}

/**
 * Per-axis resolution scale for the cone-lighting prepass; 1 keeps the inline
 * path.
 *
 * Camera-state budgets remain compile-time-free. The renderer retains and
 * prewarms both production rates (0.25 balanced/moving and 0.5 quality/settled),
 * so a camera-state tier can switch without compiling a Metal shader when
 * motion begins.
 */
export type SvoConeLightingScale = 1 | 0.5 | 0.25 | 0.125;

/** Compile-time static traversal experiment. Hybrid preserves the shipping fallback semantics. */
export type SvoDryTraversalMode = "hybrid" | "canonical" | "canonical-parametric" | "compact" | "wide";

/** Compile-time 8^3 leaf acceleration experiment; off preserves the baseline shader. */
export type SvoBrickOccupancyMode = "off" | "bounds" | "macro" | "macro-hdda";

/** Split policy: production isolates traversal from relighting only when that path is selected. */
export type SvoDryShadingPath = "inline" | "split" | "auto-relight";

/**
 * Exact primary-ray reuse experiment for render-only/static-camera frames.
 * The caller-owned key must change whenever camera, geometry, or rigid bodies
 * change. No key means no reuse, so production callers cannot opt in by
 * accident.
 */
export type SvoDryRayCoherenceMode = "off" | "static-primary";

export type SvoDryPrimaryCoherenceDecision = "trace" | "reuse";

/**
 * Dawn occupancy experiments. Every arm remains independently selectable for
 * controlled A/Bs; only the safe reduced-split diagnostic diet is enabled by
 * the renderer default.
 */
export interface SvoDryOptimizationExperiments {
  /** Remove the production cost-overlay counters from reduced split shaders. */
  readonly stripDiagnostics?: boolean;
  /** Trace incoherent 2x2 receivers in the coherent kernel instead of queueing them. */
  readonly inlineConeBoundaries?: boolean;
  /** Replace the one-thread queue-reset dispatch with a Dawn/Metal blit clear. */
  readonly clearConeQueueWithBlit?: boolean;
  /** Keep guided-upsample accumulators in native f16 registers. */
  readonly halfPrecisionLighting?: boolean;
  /** Drop the invocation-private GI page cache and re-fetch the direct table. */
  readonly dropGiPageCache?: boolean;
  /** Halve the canonical traversal stack for bounded-depth occupancy experiments. */
  readonly shortTraversalStack?: boolean;
  /** Quarter the canonical traversal stack as an overflow/fallback probe. */
  readonly tinyTraversalStack?: boolean;
}

/** Pure policy seam used by the renderer and by fail-closed contract tests. */
export function svoDryPrimaryCoherenceDecision(
  mode: SvoDryRayCoherenceMode,
  splitActive: boolean,
  frameKey: string | undefined,
  cachedKey: string | undefined,
): SvoDryPrimaryCoherenceDecision {
  return mode === "static-primary" && splitActive && frameKey !== undefined && frameKey === cachedKey
    ? "reuse" : "trace";
}

/** Exact float normal + primary hit distance crossing the split pass boundary. */
export const SVO_DRY_SPLIT_GEOMETRY_FORMAT = "rgba32float" as GPUTextureFormat;
export const SVO_DRY_SPLIT_IDENTITY_FORMAT = "rg32uint" as GPUTextureFormat;
/** 24-byte write plus 24-byte read across the pass boundary. */
export const SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL = 48;
/** Resident exact-primary cache: rgba32float geometry + rg32uint identity. */
export const SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL = 24;

/** Reduced-rate cone-lighting prepass target contract. */
export const SVO_DRY_CONE_PREPASS_CONTRACT = Object.freeze({
  /** rg32uint packs 8-bit AO and eight 7-bit light visibilities exactly into 64 bits. */
  visibilityFormat: "rg32uint" as GPUTextureFormat,
  visibilityTargetCount: 1,
  /** rgba16float packing [distance, oct-normal xy, 11-bit feature/field/motion metadata]. */
  geometryFormat: "rgba16float" as GPUTextureFormat,
  /** Exact uint16 material + uint16 owner identity used by the isolated shading pass. */
  identityFormat: "r32uint" as GPUTextureFormat,
  /** HDR opaque radiance evaluated by a separate reduced-rate shading pass. */
  radianceFormat: "rgba16float" as GPUTextureFormat,
  /** Every user-shadable light slot is cached by the reduced-rate prepass. */
  maximumPrepassLights: SVO_DRY_SCENE_MAX_SHADED_LIGHTS,
  /** Guided-upsample weight below this threshold re-traces exact inline cones (silhouettes). */
  fallbackWeightThreshold: 0.05,
} as const);

/** Persistent camera-independent cache used by the reduced split GI pass. */
export const SVO_DRY_WORLD_GI_CACHE_CONTRACT = Object.freeze({
  entryCount: 1 << 18,
  entryBytes: 16,
  probeBytes: 8,
  payloadBytes: 8,
  probeCount: 4,
  allocatedBytes: (1 << 18) * 16,
  frameBytes: 128,
  dynamicInfluenceCells: 12,
  dynamicInfluenceBodyRadii: 3,
} as const);

/** Prepass target dimensions derived from the presentation size, never below 1x1. */
export function svoConePrepassSize(width: number, height: number, scale: SvoConeLightingScale): readonly [number, number] {
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

/** Optional reduced-rate cone lighting layered over the shared user-facing options. */
export type SparseVoxelDrySceneLightingOptions = SvoLightingOptions & {
  readonly coneLightingScale?: SvoConeLightingScale;
};

/**
 * Composes the production dry fragment shader. Scale 1 must return the exact
 * historical string byte-for-byte: the bit-exact frame fingerprint gates on it,
 * and every insertion below is the empty string in that configuration.
 * Reduced scales add isolated geometry (`dryPrepassGeometryMain`), cone
 * visibility (`dryPrepassVisibilityMain`), and opaque-shading
 * (`dryPrepassShadeMain`) entries plus a depth/normal/identity-guided consumer.
 * Keeping primary traversal, cone marching, and shading in separate entries
 * prevents their register lifetimes from collapsing Metal occupancy; every
 * phase is encoded afresh each frame, including for dynamic scenes.
 */
export const SVO_DRY_SCENE_PIXEL_PROBE_GROUP = 1;

/** Constants the pixel-trace probe mirrors, passed instead of imported so the probe module stays free of this one. */
export function svoDryScenePixelProbeOptions(): SvoPixelTraceProbeOptions {
  return {
    group: SVO_DRY_SCENE_PIXEL_PROBE_GROUP,
    coneLodBlendBandWidth: SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH,
    maximumShadedLights: SVO_DRY_SCENE_MAX_SHADED_LIGHTS,
    areaLightSamples: SVO_DRY_SCENE_AREA_LIGHT_SAMPLES,
    stableOcclusionConeSamples: SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES,
    primaryLeafVisitHardLimit: SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
    visibilityFlags: SVO_DRY_VISIBILITY_FLAGS,
    cameraSettledExpression: SVO_DRY_SCENE_CAMERA_SETTLED_WGSL,
  };
}

export function createSvoDrySceneFragmentWGSL(
  coneLightingScale: SvoConeLightingScale = 1,
  traversalMode: SvoDryTraversalMode = "hybrid",
  brickOccupancyMode: SvoBrickOccupancyMode = "off",
  shadingPath: SvoDryShadingPath = "inline",
  screenSpaceTerminationPixels = 0,
  /**
   * Appends the live pixel-trace probe entry point. Off by every production
   * call site, which is what keeps scale-1 output byte-identical to the string
   * the frame fingerprint gates on.
  */
  pixelProbe = false,
  /** Split-only path: pane discovery is rasterized over projected coverage. */
  rasterGlassDiscovery = false,
  /** Split-only path: moving rigid primary hits are rasterized from live BodyGPU records. */
  rasterRigidDiscovery = false,
  /** Reduced split-only experiment: execute one deterministic cone per compute lane. */
  coneFanout = false,
  experiments: SvoDryOptimizationExperiments = {},
): string {
  if (pixelProbe && (coneLightingScale !== 1 || shadingPath !== "inline")) {
    throw new RangeError("The pixel-trace probe requires the inline, full-rate dry-scene composition");
  }
  if (rasterGlassDiscovery && shadingPath !== "split") {
    throw new RangeError("Raster glass discovery requires split shading");
  }
  if (rasterRigidDiscovery && shadingPath !== "split") {
    throw new RangeError("Raster rigid discovery requires split shading");
  }
  if (coneFanout && (shadingPath !== "split" || coneLightingScale === 1)) {
    throw new RangeError("Cone fan-out requires reduced split shading");
  }
  if (traversalMode !== "hybrid" && traversalMode !== "canonical"
    && traversalMode !== "canonical-parametric" && traversalMode !== "compact" && traversalMode !== "wide") {
    throw new RangeError(`Unsupported dry-scene traversal mode: ${traversalMode}`);
  }
  if (brickOccupancyMode !== "off" && brickOccupancyMode !== "bounds"
    && brickOccupancyMode !== "macro" && brickOccupancyMode !== "macro-hdda") {
    throw new RangeError(`Unsupported dry-scene brick occupancy mode: ${brickOccupancyMode}`);
  }
  if (shadingPath !== "inline" && shadingPath !== "split" && shadingPath !== "auto-relight") {
    throw new RangeError(`Unsupported dry-scene shading path: ${shadingPath}`);
  }
  if (!Number.isFinite(screenSpaceTerminationPixels) || screenSpaceTerminationPixels < 0) {
    throw new RangeError("Dry-scene screen-space termination must be a non-negative finite pixel count");
  }
  if (screenSpaceTerminationPixels > 0 && (traversalMode !== "canonical" || shadingPath !== "inline")) {
    throw new RangeError("Diagnostic screen-space termination currently requires canonical inline traversal");
  }
  const reduced = coneLightingScale !== 1;
  const split = shadingPath !== "inline";
  // Full-rate/inline shaders still own the diagnostic overlay. The reduced
  // split shaders are never selected while that overlay is active, so their
  // counters are dead production state and can be removed safely.
  const stripDiagnostics = experiments.stripDiagnostics === true && reduced && split;
  if (experiments.halfPrecisionLighting && !reduced) {
    throw new RangeError("Half-precision lighting is restricted to reduced-rate shaders");
  }
  const inlineBoundaryWGSL = experiments.inlineConeBoundaries
    ? "let opaque=traceOpaqueScene(ray[0],ray[1]);dryPrepassStore(coordinate,opaque,ray[0],ray[1]);return;"
    : "let queueIndex=atomicAdd(&dryPrepassBoundaryQueue.count,1u);dryPrepassBoundaryQueue.coordinates[queueIndex]=globalId.y*dimensions.x+globalId.x;return;";
  const canonicalTraversal = traversalMode === "canonical" || traversalMode === "canonical-parametric";
  const wideTraversalWGSL = canonicalTraversal || traversalMode === "compact"
    ? ""
    : createWebgpuSvoWideFanoutTraversalWGSL({ pages: 11, descriptors: 12 });
  const compactTraversalWGSL = traversalMode === "compact" ? createWebgpuSvoCompactTraversalWGSL(11) : "";
  const compactTraversal = traversalMode === "compact";
  const canonicalTraversalWGSL = createWebgpuSvoTraversalWGSL({ control: 2, nodes: 3, leaves: 4,
    childEnumeration: traversalMode === "canonical-parametric" ? "parametric" : "aabb",
    stackCapacity: experiments.tinyTraversalStack ? 8 : experiments.shortTraversalStack ? 16 : 32 });
  const screenSpaceTraversalWGSL = screenSpaceTerminationPixels > 0
    ? createSvoScreenSpaceTraversalWGSL(canonicalTraversalWGSL) : "";
  const leafAccessWGSL = compactTraversal ? /* wgsl */ `
fn dryLeafBounds(nodeIndex:u32)->mat2x3f{return svoCompactNodeBounds(svoCompactNodes[nodeIndex],dry.mapping);}
fn dryLeafFlags(nodeIndex:u32)->u32{return svoNodes[nodeIndex].links.w;}
` : /* wgsl */ `
fn dryLeafBounds(nodeIndex:u32)->mat2x3f{return svoNodeBounds(svoNodes[nodeIndex],dry.mapping);}
fn dryLeafFlags(nodeIndex:u32)->u32{return svoNodes[nodeIndex].links.w;}
`;
  const primaryTraversalCursorWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
fn drySvoShouldTerminateNodeScreenSpace(bounds:mat2x3f,level:u32)->bool{
  return svoShouldTerminateNodeScreenSpace(bounds,uniforms.cameraPosition.xyz,uniforms.viewport.y,.72,${screenSpaceTerminationPixels},level,0u);
}
fn dryTraversalCursorNextPrimary(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{
  return svoTraversalContinuationNextScreenSpace(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);
}
` : /* wgsl */ `
fn dryTraversalCursorNextPrimary(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return dryTraversalCursorNext(ray,mapping,cursor);}
`;
  const screenSpaceProxyWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
const DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY:u32=15u;
fn dryScreenSpaceProxyHit(ro:vec3f,rd:vec3f,hit:SvoTraversalHit)->DryHit{
  let bounds=svoNodeBounds(svoNodes[hit.nodeIndex],dry.mapping);let point=ro+rd*hit.tEnter;
  let faceDistance=min(abs(point-bounds[0]),abs(bounds[1]-point));var axis=0u;
  if(faceDistance.y<faceDistance.x){axis=1u;}if(faceDistance.z<faceDistance[axis]){axis=2u;}
  var normal=vec3f(0.0);normal[axis]=select(-1.0,1.0,abs(point[axis]-bounds[1][axis])<abs(point[axis]-bounds[0][axis]));
  return DryHit(hit.tEnter,normal,0u,DRY_OWNER_NONE,SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(hit.level,0u,0u));
}
` : "";
  const screenSpaceProxyTraceWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `if(leaf.status==SVO_STATUS_SCREEN_SPACE_PROXY){dryPrimaryMaximumDepth=max(dryPrimaryMaximumDepth,leaf.level);return dryScreenSpaceProxyHit(ro,rd,leaf);}` : "";
  const screenSpaceProxyShadeWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `if(hit.fieldSource==DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY){let depthBand=clamp(f32(hit._padding.x)/21.0,0.0,1.0);return mix(vec3f(.02,.06,.18),vec3f(1.0,.04,.72),depthBand);}` : "";
  const traversalCursorWGSL = canonicalTraversal ? /* wgsl */ `
struct DryTraversalCursor{canonical:SvoTraversalContinuation}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){svoTraversalContinuationBegin(ray,mapping,&(*cursor).canonical);}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoTraversalContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);}
` : traversalMode === "compact" ? /* wgsl */ `
struct DryTraversalCursor{compact:SvoCompactTraversalContinuation}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){svoCompactContinuationBegin(ray,mapping,&(*cursor).compact);}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoCompactContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).compact);}
` : traversalMode === "wide" ? /* wgsl */ `
struct DryTraversalCursor{wide:SvoWideTraversalCursor}
fn dryWidePublication()->SvoWidePublication{return SvoWidePublication(dry.wideFanout.x,dry.wideFanout.y,dry.wideFanout.z,dry.wideFanout.w);}
fn dryCanonicalPublicationGeneration()->u32{return select(0u,publicationState[2],arrayLength(&publicationState)>2u);}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){let initialized=svoWideCursorInitialize(&(*cursor).wide,ray,mapping,dryWidePublication(),dryCanonicalPublicationGeneration());if(!initialized){(*cursor).wide.state=SVO_WIDE_CURSOR_INVALID;}}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoWideCursorNext(&(*cursor).wide,ray,mapping,dryDiagnosticMaximumDepth(),dryWidePublication(),dryCanonicalPublicationGeneration());}
` : /* wgsl */ `
struct DryTraversalCursor{canonical:SvoTraversalContinuation,wide:SvoWideTraversalCursor,useWide:u32}
fn dryWidePublication()->SvoWidePublication{return SvoWidePublication(dry.wideFanout.x,dry.wideFanout.y,dry.wideFanout.z,dry.wideFanout.w);}
fn dryCanonicalPublicationGeneration()->u32{return select(0u,publicationState[2],arrayLength(&publicationState)>2u);}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){
  (*cursor).useWide=select(0u,1u,svoWideCursorInitialize(&(*cursor).wide,ray,mapping,dryWidePublication(),dryCanonicalPublicationGeneration()));
  if((*cursor).useWide==0u){svoTraversalContinuationBegin(ray,mapping,&(*cursor).canonical);}
}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{
  if((*cursor).useWide!=0u){let wideHit=svoWideCursorNext(&(*cursor).wide,ray,mapping,dryDiagnosticMaximumDepth(),dryWidePublication(),dryCanonicalPublicationGeneration());if(wideHit.status==SVO_STATUS_HIT||wideHit.status==SVO_STATUS_MISS||wideHit.status==SVO_STATUS_WORK_EXHAUSTED){return wideHit;}(*cursor).useWide=0u;if(wideHit.visits>=mapping.maxVisits){return svoMiss(SVO_STATUS_WORK_EXHAUSTED,wideHit.visits);}var fallbackMapping=mapping;fallbackMapping.maxVisits-=wideHit.visits;svoTraversalContinuationBegin(ray,fallbackMapping,&(*cursor).canonical);var fallback=svoTraversalContinuationNext(ray,fallbackMapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);fallback.visits+=wideHit.visits;return fallback;}
  return svoTraversalContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);
}
`;
  const brickOccupancyHelpersWGSL = brickOccupancyMode === "off" ? "" : /* wgsl */ `
${svoBrickOccupancyWGSL}
const DRY_BRICK_OCCUPANCY_MACRO:u32=${brickOccupancyMode === "macro" ? 1 : 0}u;
fn dryBrickMacroSkip(summary:SvoBrickOccupancy,local:vec3u,bounds:mat2x3f,extent:vec3f,ro:vec3f,rd:vec3f,entry:f32)->vec2f{
  if(DRY_BRICK_OCCUPANCY_MACRO==0u||svoBrickMacroOccupied(summary,local)){return vec2f(0.0,entry);}
  let macroCoord=local>>vec3u(2u);let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let boundaryCell=vec3i(macroCoord*4u)+select(vec3i(0),vec3i(4),step>vec3i(0));
  let boundary=bounds[0]+vec3f(boundaryCell)*extent;
  let next=select(vec3f(3.402823e38),(boundary-ro)/rd,abs(rd)>vec3f(1e-9));
  return vec2f(1.0,max(entry,min(next.x,min(next.y,next.z))));
}
`;
  const primaryBrickSetupWGSL = brickOccupancyMode === "off"
    ? /* wgsl */ `let bounds=dryLeafBounds(hit.nodeIndex); let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  var entry=max(hit.tEnter,0.0);let point=ro+rd*(entry+1e-5); var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));`
    : /* wgsl */ `let bounds=dryLeafBounds(hit.nodeIndex);let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  let brickSummary=svoBrickOccupancyDecode(dryLeafFlags(hit.nodeIndex));var brickExit=hit.tExit;var entry=max(hit.tEnter,0.0);
  if(brickSummary.ready!=0u){if(brickSummary.occupied==0u){return missHit();}let occupiedInterval=svoRayAabbWithInverse(SvoRay(ro,entry,rd,brickExit),1.0/rd,svoBrickOccupiedBounds(brickSummary,bounds[0],extent));if(occupiedInterval.x==0.0){return missHit();}entry=max(entry,occupiedInterval.y);brickExit=min(brickExit,occupiedInterval.z);}
  let point=ro+rd*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));`;
  const primaryBrickExitWGSL = brickOccupancyMode === "off" ? "hit.tExit" : "brickExit";
  const primaryMacroSkipWGSL = brickOccupancyMode === "macro" ? /* wgsl */ `let macroSkip=dryBrickMacroSkip(brickSummary,vec3u(cell),bounds,extent,ro,rd,entry);if(macroSkip.x!=0.0){if(macroSkip.y>=brickExit||macroSkip.y>=DRY_MISS){break;}entry=macroSkip.y;let skipPoint=ro+rd*(entry+max(1e-5,length(extent)*1e-4));cell=vec3i(clamp(floor((skipPoint-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));let skipBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;nextT=select(vec3f(DRY_MISS),(skipBoundary-ro)/rd,abs(rd)>vec3f(1e-9));continue;}
    ` : "";
  const shadowBrickSetupWGSL = brickOccupancyMode === "off"
    ? /* wgsl */ `let bounds=dryLeafBounds(hit.nodeIndex);let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  var entry=max(max(hit.tEnter,tMin_m),0.0);let point=ray.origin_m+ray.direction*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));`
    : /* wgsl */ `let bounds=dryLeafBounds(hit.nodeIndex);let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  let brickSummary=svoBrickOccupancyDecode(dryLeafFlags(hit.nodeIndex));var brickExit=min(hit.tExit,ray.tMax_m);var entry=max(max(hit.tEnter,tMin_m),0.0);
  if(brickSummary.ready!=0u){if(brickSummary.occupied==0u){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS);}let occupiedInterval=svoRayAabbWithInverse(SvoRay(ray.origin_m,entry,ray.direction,brickExit),1.0/ray.direction,svoBrickOccupiedBounds(brickSummary,bounds[0],extent));if(occupiedInterval.x==0.0){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS);}entry=max(entry,occupiedInterval.y);brickExit=min(brickExit,occupiedInterval.z);}
  let point=ray.origin_m+ray.direction*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));`;
  const shadowBrickExitWGSL = brickOccupancyMode === "off" ? "hit.tExit" : "brickExit";
  const shadowMacroSkipWGSL = brickOccupancyMode === "macro" ? /* wgsl */ `let macroSkip=dryBrickMacroSkip(brickSummary,vec3u(cell),bounds,extent,ray.origin_m,ray.direction,entry);if(macroSkip.x!=0.0){if(macroSkip.y>=brickExit||macroSkip.y>=DRY_MISS){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);}entry=macroSkip.y;let skipPoint=ray.origin_m+ray.direction*(entry+max(1e-5,length(extent)*1e-4));cell=vec3i(clamp(floor((skipPoint-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));let skipBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;nextT=select(vec3f(DRY_MISS),(skipBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));continue;}
    ` : "";
  const primaryLeafTraceCallWGSL = brickOccupancyMode === "macro-hdda" ? "traceLeafPayloadMacroHdda" : "traceLeafPayload";
  const shadowLeafTraceCallWGSL = brickOccupancyMode === "macro-hdda" ? "traceLeafPayloadVisibilityMacroHdda" : "traceLeafPayloadVisibility";
  const macroHddaPrimaryWGSL = brickOccupancyMode === "macro-hdda" ? /* wgsl */ `
fn traceLeafPayloadFineInterval(ro:vec3f,rd:vec3f,hit:SvoTraversalHit,bounds:mat2x3f,extent:vec3f,intervalEnter:f32,intervalExit:f32,cellMinimum:vec3u,cellMaximum:vec3u)->DryHit{
  var entry=max(intervalEnter,0.0);let point=ro+rd*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum-vec3u(1u))));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));let tolerance=length(extent)*1.05;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(cellMinimum))||any(cell>=vec3i(cellMaximum))||entry>intervalExit){break;}
    dryPrimaryVoxelWorkItems+=1u;let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex<arrayLength(&materialOwners)){let identity=materialOwners[payloadIndex];let owner=identity>>16u;if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){let primitiveIndex=owner-dry.metadata.y;if(primitiveIndex<dry.metadata.x&&primitiveIndex<arrayLength(&primitives)){dryPrimaryExactTests+=1u;let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,intervalExit));let candidate=primitiveHit(primitives[primitiveIndex],ro,rd,max(0.0,entry-tolerance),cellExit+tolerance);if(candidate.t<DRY_MISS){return candidate;}}}}
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return missHit();
}
fn traceLeafPayloadMacroHdda(ro:vec3f,rd:vec3f,hit:SvoTraversalHit)->DryHit{
  let bounds=dryLeafBounds(hit.nodeIndex);let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);let summary=svoBrickOccupancyDecode(dryLeafFlags(hit.nodeIndex));
  if(summary.ready==0u){return traceLeafPayloadFineInterval(ro,rd,hit,bounds,extent,hit.tEnter,hit.tExit,vec3u(0u),vec3u(dry.mapping.brickSize));}
  if(summary.occupied==0u){return missHit();}
  let interval=svoRayAabbWithInverse(SvoRay(ro,max(hit.tEnter,0.0),rd,hit.tExit),1.0/rd,svoBrickOccupiedBounds(summary,bounds[0],extent));if(interval.x==0.0){return missHit();}
  var macroEntry=max(max(hit.tEnter,interval.y),0.0);let brickExit=min(hit.tExit,interval.z);let macroExtent=extent*4.0;let point=ro+rd*(macroEntry+1e-5);var macroCell=vec3i(clamp(floor((point-bounds[0])/macroExtent),vec3f(0.0),vec3f(1.0)));
  let macroStep=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));let macroBoundary=bounds[0]+(vec3f(macroCell)+select(vec3f(0.0),vec3f(1.0),macroStep>vec3i(0)))*macroExtent;
  var macroNextT=select(vec3f(DRY_MISS),(macroBoundary-ro)/rd,abs(rd)>vec3f(1e-9));let macroDeltaT=select(vec3f(DRY_MISS),abs(macroExtent/rd),abs(rd)>vec3f(1e-9));
  for(var macroIteration=0u;macroIteration<8u;macroIteration+=1u){
    if(any(macroCell<vec3i(0))||any(macroCell>=vec3i(2))||macroEntry>brickExit){break;}let macroExit=min(min(macroNextT.x,macroNextT.y),min(macroNextT.z,brickExit));let macroCoord=vec3u(macroCell);let macroBit=macroCoord.x|(macroCoord.y<<1u)|(macroCoord.z<<2u);
    if((summary.macroMask&(1u<<macroBit))!=0u){let cellMinimum=macroCoord*4u;let candidate=traceLeafPayloadFineInterval(ro,rd,hit,bounds,extent,macroEntry,macroExit,cellMinimum,cellMinimum+vec3u(4u));if(candidate.t<DRY_MISS){return candidate;}}
    let advance=min(macroNextT.x,min(macroNextT.y,macroNextT.z));if(macroNextT.x<=advance+1e-6){macroCell.x+=macroStep.x;macroNextT.x+=macroDeltaT.x;}if(macroNextT.y<=advance+1e-6){macroCell.y+=macroStep.y;macroNextT.y+=macroDeltaT.y;}if(macroNextT.z<=advance+1e-6){macroCell.z+=macroStep.z;macroNextT.z+=macroDeltaT.z;}macroEntry=advance;
  }
  return missHit();
}
` : "";
  const macroHddaShadowWGSL = brickOccupancyMode === "macro-hdda" ? /* wgsl */ `
fn traceLeafPayloadVisibilityFineInterval(ray:SvoVisibilityRay,tMin_m:f32,hit:SvoTraversalHit,bounds:mat2x3f,extent:vec3f,intervalEnter:f32,intervalExit:f32,cellMinimum:vec3u,cellMaximum:vec3u,workLimit:u32)->SvoVisibilityStep{
  var entry=max(max(intervalEnter,tMin_m),0.0);let point=ray.origin_m+ray.direction*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum-vec3u(1u))));
  let step=select(vec3i(-1),vec3i(1),ray.direction>=vec3f(0.0));let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));let deltaT=select(vec3f(DRY_MISS),abs(extent/ray.direction),abs(ray.direction)>vec3f(1e-9));let tolerance=length(extent)*1.05;var workItems=0u;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(cellMinimum))||any(cell>=vec3i(cellMaximum))||entry>intervalExit||entry>ray.tMax_m){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);}if(workItems>=workLimit){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);}workItems+=1u;
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);if(payloadIndex>=arrayLength(&materialOwners)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}let identity=materialOwners[payloadIndex];let owner=identity>>16u;
    if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){let primitiveIndex=owner-dry.metadata.y;if(primitiveIndex>=dry.metadata.x||primitiveIndex>=arrayLength(&primitives)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,min(intervalExit,ray.tMax_m)));let candidate=primitiveHit(primitives[primitiveIndex],ray.origin_m,ray.direction,max(entry-tolerance,tMin_m),cellExit+tolerance);if(candidate.t<DRY_MISS){return dryVisibilityStep(SVO_VIS_STEP_HIT,0u,0u,workItems,candidate.t);}}
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);
}
fn traceLeafPayloadVisibilityMacroHdda(ray:SvoVisibilityRay,tMin_m:f32,hit:SvoTraversalHit,workLimit:u32)->SvoVisibilityStep{
  let bounds=dryLeafBounds(hit.nodeIndex);let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);let summary=svoBrickOccupancyDecode(dryLeafFlags(hit.nodeIndex));
  if(summary.ready==0u){return traceLeafPayloadVisibilityFineInterval(ray,tMin_m,hit,bounds,extent,hit.tEnter,min(hit.tExit,ray.tMax_m),vec3u(0u),vec3u(dry.mapping.brickSize),workLimit);}if(summary.occupied==0u){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS);}
  let interval=svoRayAabbWithInverse(SvoRay(ray.origin_m,max(max(hit.tEnter,tMin_m),0.0),ray.direction,min(hit.tExit,ray.tMax_m)),1.0/ray.direction,svoBrickOccupiedBounds(summary,bounds[0],extent));if(interval.x==0.0){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS);}
  var macroEntry=max(max(hit.tEnter,interval.y),tMin_m);let brickExit=min(min(hit.tExit,interval.z),ray.tMax_m);let macroExtent=extent*4.0;let point=ray.origin_m+ray.direction*(macroEntry+1e-5);var macroCell=vec3i(clamp(floor((point-bounds[0])/macroExtent),vec3f(0.0),vec3f(1.0)));
  let macroStep=select(vec3i(-1),vec3i(1),ray.direction>=vec3f(0.0));let macroBoundary=bounds[0]+(vec3f(macroCell)+select(vec3f(0.0),vec3f(1.0),macroStep>vec3i(0)))*macroExtent;
  var macroNextT=select(vec3f(DRY_MISS),(macroBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));let macroDeltaT=select(vec3f(DRY_MISS),abs(macroExtent/ray.direction),abs(ray.direction)>vec3f(1e-9));var workItems=0u;
  for(var macroIteration=0u;macroIteration<8u;macroIteration+=1u){
    if(any(macroCell<vec3i(0))||any(macroCell>=vec3i(2))||macroEntry>brickExit){break;}let macroExit=min(min(macroNextT.x,macroNextT.y),min(macroNextT.z,brickExit));let macroCoord=vec3u(macroCell);let macroBit=macroCoord.x|(macroCoord.y<<1u)|(macroCoord.z<<2u);
    if((summary.macroMask&(1u<<macroBit))!=0u){if(workItems>=workLimit){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);}let cellMinimum=macroCoord*4u;var fine=traceLeafPayloadVisibilityFineInterval(ray,tMin_m,hit,bounds,extent,macroEntry,macroExit,cellMinimum,cellMinimum+vec3u(4u),workLimit-workItems);workItems+=fine.workItems;if(fine.status!=SVO_VIS_STEP_MISS){fine.workItems=workItems;return fine;}}
    let advance=min(macroNextT.x,min(macroNextT.y,macroNextT.z));if(macroNextT.x<=advance+1e-6){macroCell.x+=macroStep.x;macroNextT.x+=macroDeltaT.x;}if(macroNextT.y<=advance+1e-6){macroCell.y+=macroStep.y;macroNextT.y+=macroDeltaT.y;}if(macroNextT.z<=advance+1e-6){macroCell.z+=macroStep.z;macroNextT.z+=macroDeltaT.z;}macroEntry=advance;
  }
  return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);
}
` : "";
  const prepassDeclarationsWGSL = reduced ? /* wgsl */ `// Reduced-rate cone-lighting prepass consumption. One integer plane packs
// 8-bit AO plus eight 7-bit light visibilities. Geometry packs distance, an
// octahedral normal, and 11-bit metadata; identity packs uint16 material+owner.
@group(1) @binding(0) var dryPrepassVisibilityKeyTexture:texture_2d<u32>;
@group(1) @binding(1) var dryPrepassGeometryTexture:texture_2d<f32>;
@group(1) @binding(2) var dryPrepassIdentityTexture:texture_2d<u32>;
@group(1) @binding(3) var dryPrepassRadianceTexture:texture_2d<f32>;
var<private> dryPrepassData0:vec4f;
var<private> dryPrepassData1:vec4f;
var<private> dryPrepassData2:vec4f;
var<private> dryPrepassRadiance:vec4f;
var<private> dryPrepassState:u32;
var<private> dryPrepassRadianceState:u32;
var<private> dryPrepassGi:vec4f;
var<private> dryPrepassGiState:u32;
var<private> dryConeFallback:u32;
var<private> dryCurrentLightSlot:u32;
fn dryPrepassQuantize7(value:f32)->u32{return u32(round(clamp(value,0.0,1.0)*127.0));}
fn dryPrepassPack(data0:vec4f,data1:vec4f,data2:vec4f)->vec2u{
  let light3=dryPrepassQuantize7(data1.x);
  let word0=u32(round(clamp(data0.x,0.0,1.0)*255.0))|(dryPrepassQuantize7(data0.y)<<8u)|(dryPrepassQuantize7(data0.z)<<15u)|(dryPrepassQuantize7(data0.w)<<22u)|((light3&7u)<<29u);
  let word1=(light3>>3u)|(dryPrepassQuantize7(data1.y)<<4u)|(dryPrepassQuantize7(data1.z)<<11u)|(dryPrepassQuantize7(data1.w)<<18u)|(dryPrepassQuantize7(data2.x)<<25u);return vec2u(word0,word1);
}
fn dryPrepassUnpack0(packed:vec4u)->vec4f{return vec4f(f32(packed.x&255u)/255.0,f32((packed.x>>8u)&127u)/127.0,f32((packed.x>>15u)&127u)/127.0,f32((packed.x>>22u)&127u)/127.0);}
fn dryPrepassUnpack1(packed:vec4u)->vec4f{let light3=((packed.x>>29u)&7u)|((packed.y&15u)<<3u);return vec4f(f32(light3)/127.0,f32((packed.y>>4u)&127u)/127.0,f32((packed.y>>11u)&127u)/127.0,f32((packed.y>>18u)&127u)/127.0);}
fn dryPrepassUnpack2(packed:vec4u)->vec4f{return vec4f(f32((packed.y>>25u)&127u)/127.0,1.0,1.0,1.0);}
fn dryPrepassEncodeNormal(normalIn:vec3f)->vec2f{let normal=normalize(normalIn);var oct=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));if(normal.z<0.0){oct=(vec2f(1.0)-abs(oct.yx))*select(vec2f(-1.0),vec2f(1.0),oct>=vec2f(0.0));}return oct;}
fn dryPrepassDecodeNormal(octIn:vec2f)->vec3f{var normal=vec3f(octIn,1.0-abs(octIn.x)-abs(octIn.y));if(normal.z<0.0){let folded=(vec2f(1.0)-abs(normal.yx))*select(vec2f(-1.0),vec2f(1.0),normal.xy>=vec2f(0.0));normal=vec3f(folded,normal.z);}return normalize(normal);}
fn dryPrepassHitMetadata(hit:DryHit)->u32{return (hit.featureId&15u)|((hit.fieldSource&15u)<<4u)|((hit.motionKind&3u)<<8u)|((hit.motionValid&1u)<<10u);}
fn dryPrepassPackIdentity(hit:DryHit)->u32{return (hit.materialId&0xffffu)|((hit.ownerId&0xffffu)<<16u);}
fn dryPrepassChannel(index:u32)->f32{
  if(index<4u){return dryPrepassData0[index];}
  if(index<8u){return dryPrepassData1[index-4u];}
  return dryPrepassData2[min(index-8u,3u)];
}
fn dryPrepassIdentityMatches(identity:u32,metadata:u32,hit:DryHit)->bool{return identity==dryPrepassPackIdentity(hit)&&metadata==dryPrepassHitMetadata(hit);}
fn dryPrepassResolve(pixel:vec2f,depth:f32,normalIn:vec3f,hit:DryHit){
  let dims=textureDimensions(dryPrepassGeometryTexture);
  let normal=normalize(normalIn);
  let coordinate=pixel*(vec2f(dims)/max(uniforms.viewport.xy,vec2f(1.0)))-vec2f(.5);
  let base=floor(coordinate);let fraction=coordinate-base;
  var accumulated0=vec4f(0.0);var accumulated1=vec4f(0.0);var accumulated2=vec4f(0.0);var weightSum=0.0;
  var accumulatedRadiance=vec4f(0.0);var radianceWeightSum=0.0;var accumulatedGi=vec4f(0.0);var giWeightSum=0.0;var bestRadianceWeight=0.0;var bestRadianceTexel=vec2i(0);var linearSafe=1u;
  for(var j=0u;j<2u;j+=1u){for(var i=0u;i<2u;i+=1u){
    let texel=vec2i(clamp(base+vec2f(f32(i),f32(j)),vec2f(0.0),vec2f(dims)-vec2f(1.0)));
    let geometry=textureLoad(dryPrepassGeometryTexture,texel,0);
    if(geometry.x<=0.0){linearSafe=0u;continue;}
    let packed=textureLoad(dryPrepassVisibilityKeyTexture,texel,0);
    let bilinear=select(1.0-fraction.x,fraction.x,i==1u)*select(1.0-fraction.y,fraction.y,j==1u);
    let depthWeight=exp(-24.0*abs(geometry.x-depth)/max(depth,1e-3));
    let normalWeight=pow(max(dot(normal,dryPrepassDecodeNormal(geometry.yz)),0.0),8.0);
    let identityMatches=dryPrepassIdentityMatches(textureLoad(dryPrepassIdentityTexture,texel,0).x,u32(round(geometry.w)),hit);
    if(bilinear>1e-6&&!identityMatches){linearSafe=0u;}
    if(bilinear>1e-6&&(depthWeight<0.25||normalWeight<0.25)){linearSafe=0u;}
    let guidedWeight=depthWeight*normalWeight;
    let weight=bilinear*select(guidedWeight,1.0,dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u);
    if(weight<=1e-6){continue;}
    accumulated0+=dryPrepassUnpack0(packed)*weight;
    accumulated1+=dryPrepassUnpack1(packed)*weight;
    accumulated2+=dryPrepassUnpack2(packed)*weight;
    if(identityMatches){
      if(weight>bestRadianceWeight){bestRadianceWeight=weight;bestRadianceTexel=texel;}
      if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u){accumulatedGi+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;giWeightSum+=weight;}
      if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["joint-bilateral"]}u){accumulatedRadiance+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;radianceWeightSum+=weight;}
    }
    weightSum+=weight;
  }}
  if(weightSum<${SVO_DRY_CONE_PREPASS_CONTRACT.fallbackWeightThreshold}){dryConeFallback=1u;return;}
  dryPrepassData0=accumulated0/weightSum;dryPrepassData1=accumulated1/weightSum;dryPrepassData2=accumulated2/weightSum;dryPrepassState=1u;
  if(giWeightSum>=${SVO_DRY_CONE_PREPASS_CONTRACT.fallbackWeightThreshold}){dryPrepassGi=accumulatedGi/giWeightSum;dryPrepassGiState=1u;}
  if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["gated-linear"]}u&&linearSafe!=0u&&bestRadianceWeight>0.0){
    dryPrepassRadiance=textureSampleLevel(dryPrepassRadianceTexture,nodeMipSampler,pixel/max(uniforms.viewport.xy,vec2f(1.0)),0.0);dryPrepassRadianceState=1u;
  }else if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["joint-bilateral"]}u&&radianceWeightSum>1e-6){
    dryPrepassRadiance=accumulatedRadiance/radianceWeightSum;dryPrepassRadianceState=1u;
  }else if(dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u&&dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u&&bestRadianceWeight>0.0){dryPrepassRadiance=textureLoad(dryPrepassRadianceTexture,bestRadianceTexel,0);dryPrepassRadianceState=1u;}
}
` : "";
  const splitGroup = reduced ? 2 : 1;
  const splitGlassKeyDeclarationWGSL = rasterGlassDiscovery
    ? /* wgsl */ `@group(${splitGroup}) @binding(6) var drySplitGlassKeyRead:texture_2d<u32>;
`
    : "";
  const splitRigidReadWGSL = /* wgsl */ `
fn drySplitGeometryAt(coordinate:vec2i)->vec4f{return textureLoad(drySplitGeometryRead,coordinate,0);}
fn drySplitIdentityAt(coordinate:vec2i)->vec4u{return textureLoad(drySplitOpaqueIdentityRead,coordinate,0);}
`;
  const splitDeclarationsWGSL = split ? /* wgsl */ `// Split visibility/lighting bridge. The visibility entry writes exact primary
// geometry while the lighting entry reads it together with the final G-buffer.
// Separate pipelines expose only the bindings reachable from their entry point.
@group(${splitGroup}) @binding(0) var drySplitGeometryWrite:texture_storage_2d<rgba32float,write>;
@group(${splitGroup}) @binding(1) var drySplitGeometryRead:texture_2d<f32>;
@group(${splitGroup}) @binding(4) var drySplitOpaqueIdentityWrite:texture_storage_2d<rg32uint,write>;
@group(${splitGroup}) @binding(5) var drySplitOpaqueIdentityRead:texture_2d<u32>;
${splitGlassKeyDeclarationWGSL}
${splitRigidReadWGSL}
` : "";
  const prepassResolveCallWGSL = reduced
    ? /* wgsl */ `dryPrepassData0=vec4f(1.0);dryPrepassData1=vec4f(1.0);dryPrepassData2=vec4f(1.0);dryPrepassRadiance=vec4f(0.0);dryPrepassGi=vec4f(0.0,0.0,0.0,1.0);dryPrepassState=0u;dryPrepassRadianceState=0u;dryPrepassGiState=0u;dryConeFallback=0u;dryCurrentLightSlot=0xffffffffu;if(opaque.t<DRY_MISS&&(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u&&dryNodeMipReady()){dryPrepassResolve(input.position.xy,opaque.t,opaque.normal,opaque);}`
    : "";
  const prepassShadowShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassState==1u&&dryCurrentLightSlot<${SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights}u){let prepassRigidBlocked=anyBodyBlockerIgnoring(ray.origin_m,towardLight,ownerId,ray.tMax_m);let raw=select(dryPrepassChannel(1u+dryCurrentLightSlot),0.0,prepassRigidBlocked);return vec3f(mix(1.0,raw,dry.tuningRays0.y));}`
    : "";
  const prepassContactShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassState==1u){let prepassRadius=dryContactVisibilityRadius();if(prepassRadius<=0.0){return vec3f(1.0);}let prepassCell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let prepassOrigin=position+normalize(geometricNormal)*prepassCell*.2;let prepassSamples=select(dry.tuningCounts1.z,dry.tuningCounts1.y,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL});var prepassUnblocked=0.0;for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=prepassSamples){break;}let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);let rotated=select(direction,normalize(direction+cross(normalize(geometricNormal),direction)*.7),sampleIndex>=2u);let prepassRigidBlocked=anyBodyBlockerIgnoring(prepassOrigin,rotated,ownerId,prepassRadius);prepassUnblocked+=select(1.0,0.0,prepassRigidBlocked);}let raw=clamp(dryPrepassData0.x*(prepassUnblocked/f32(prepassSamples)),0.0,1.0);return vec3f(mix(1.0,raw,dry.tuningRays0.w));}`
    : "";
  const prepassBodyBlockerWGSL = reduced ? /* wgsl */ `fn anyBodyBlockerIgnoring(ro:vec3f,rd:vec3f,ignoredOwner:u32,tMax:f32)->bool {
  for(var index=0u;index<12u;index+=1u){if(index>=u32(round(uniforms.options.z))){break;}if(index==ignoredOwner){continue;}let body=bodies[index];if(!bodyBoundingSphereVisible(ro,rd,body,0.0,tMax)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ro,rd,body,0.0,tMax)){continue;}if(bodyHit(ro,rd,body).t<tMax){return true;}}
  return false;
}
` : "";
  const prepassLightSlotWGSL = reduced ? /* wgsl */ `dryCurrentLightSlot=lightIndex;` : "";
  const prepassRadianceShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassRadianceState==1u&&hit.motionKind==DRY_GBUFFER_MOTION_STATIC){return max(dryPrepassRadiance.rgb,vec3f(0.0));}`
    : "";
  const prepassGiShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassGiState==1u){return DryGlobalIllumination(max(dryPrepassGi.rgb,vec3f(0.0)),clamp(dryPrepassGi.a,0.0,1.0));}`
    : "";
  const prepassOverlayWGSL = reduced ? /* wgsl */ `if(mode==${svoCostOverlayCode("prepass-fallback")}u){overlayColor=vec3f(.16,.14,.24);if(dryPrepassState==1u){overlayColor=vec3f(0.0,.9,1.0);}if(dryConeFallback==1u){overlayColor=vec3f(1.0,.09,.30);}}
  ` : /* wgsl */ `if(mode==${svoCostOverlayCode("prepass-fallback")}u){overlayColor=vec3f(.16,.14,.24);}
  `;
  const splitVisibilityGlassDiscoveryWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let coordinate=vec2i(input.position.xy);textureStore(drySplitGeometryWrite,coordinate,vec4f(opaque.normal,opaque.t));let opaqueMetadata=(opaque.ownerId&0xffffu)|((opaque.featureId&15u)<<16u)|((opaque.fieldSource&15u)<<20u)|((opaque.motionKind&3u)<<24u)|((opaque.motionValid&1u)<<26u);textureStore(drySplitOpaqueIdentityWrite,coordinate,vec4u(opaque.materialId,opaqueMetadata,0u,0u));let generation=dryPublicationGeneration();`
    : /* wgsl */ `let glass=traceGlass(ro,rd,0.0,opaque.t,true);let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;let coordinate=vec2i(input.position.xy);textureStore(drySplitGeometryWrite,coordinate,vec4f(opaque.normal,opaque.t));let opaqueMetadata=(opaque.ownerId&0xffffu)|((opaque.featureId&15u)<<16u)|((opaque.fieldSource&15u)<<20u)|((opaque.motionKind&3u)<<24u)|((opaque.motionValid&1u)<<26u);let glassKey=select(0u,glass.recordIndex+1u,glassVisible);let packedOpaqueMaterial=(opaque.materialId&0x8000ffffu)|((glassKey&0x1ffu)<<16u);textureStore(drySplitOpaqueIdentityWrite,coordinate,vec4u(packedOpaqueMaterial,opaqueMetadata,0u,0u));let generation=dryPublicationGeneration();`;
  const splitVisibilityGlassReturnWGSL = rasterGlassDiscovery
    ? ""
    : /* wgsl */ `if(glassVisible){let record=glassPanes[glass.recordIndex];let media=dryMediumPair(rd,glass.hit.geometricNormal,DRY_MEDIUM_GLASS);let targets=svoGBufferSurface(vec3f(0.0),glass.hit.t_m,glass.hit.geometricNormal,glass.hit.geometricNormal,vec4u(svoThinGlassMaterialId(record),svoThinGlassOwnerId(record),media.x,media.y),vec3f(0.0),DRY_GBUFFER_MOTION_STATIC,DRY_GBUFFER_FIELD_ANALYTIC,generation,SVO_GBUFFER_MOTION_VALID,SVO_FEATURE_SMOOTH);return drySplitVisibilityOut(targets,dryHardwareDepth(glass.hit.t_m,rd,forward));}`;
  const splitOpaqueMaterialDecodeWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let opaqueMaterial=packedOpaqueMaterial;`
    : /* wgsl */ `let opaqueMaterial=select(packedOpaqueMaterial&0xffffu,0x80000000u|(packedOpaqueMaterial&0xffffu),(packedOpaqueMaterial&0x80000000u)!=0u);`;
  const splitGlassKeyLoadWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let glassKey=textureLoad(drySplitGlassKeyRead,coordinate,0).x;`
    : /* wgsl */ `let glassKey=(packedOpaqueMaterial>>16u)&0x1ffu;`;
  const splitPrimaryTraceWGSL = rasterRigidDiscovery ? "traceStaticSolidScene(ro,rd)" : "traceOpaqueScene(ro,rd)";
  const prepassEntryWGSL = reduced ? /* wgsl */ `struct DryPrepassGeometryOut{@location(0) geometry:vec4f,@location(1) identity:u32}
@fragment fn dryPrepassGeometryMain(input:VertexOut)->DryPrepassGeometryOut{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);
  dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  var output=DryPrepassGeometryOut(vec4f(0.0),0xffffffffu);
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)==0u||!dryNodeMipReady()){return output;}
  let opaque=traceOpaqueScene(ro,rd);
  if(!(opaque.t<DRY_MISS)){return output;}
  let geometricNormal=normalize(opaque.normal);
  output.geometry=vec4f(opaque.t,dryPrepassEncodeNormal(geometricNormal),f32(dryPrepassHitMetadata(opaque)));
  output.identity=dryPrepassPackIdentity(opaque);
  return output;
}
fn dryPrepassTraceVisibility(opaque:DryHit,ro:vec3f,rd:vec3f)->vec2u{
  dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  let position=ro+rd*opaque.t;let geometricNormal=normalize(opaque.normal);
  var visibility0=vec4f(1.0);var visibility1=vec4f(1.0);var visibility2=vec4f(1.0);
  // AO cones exclude rigid blockers; those stay exact at full resolution.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion}u)!=0u){
    let radius=dryContactVisibilityRadius();
    if(radius>0.0){
      let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
      let origin=position+geometricNormal*cellScale*.2;
      let coneSampleCount=select(dry.tuningCounts1.z,dry.tuningCounts1.y,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL});
      var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u;sampleIndex+=1u){
        if(sampleIndex>=coneSampleCount){break;}
        let direction=dryContactVisibilityDirection(geometricNormal,opaque.featureId,sampleIndex&1u);
        let rotated=select(direction,normalize(direction+cross(geometricNormal,direction)*.7),sampleIndex>=2u);
        // AO keeps near-surface self-occlusion by design: zero normal disables
        // the shadow cones' receiver-plane coverage suppression.
        let cone=dryConeVisibility(origin,rotated,dry.tuningRays1.x,radius,vec3f(0.0),false);
        visibility+=cone.transmittance;
      }
      visibility0.x=clamp(visibility/f32(coneSampleCount),0.0,1.0);
    }
  }
  // Per-light cone shadow terms for every shaded slot; area lights average two fixed samples.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.exactShadow}u)!=0u){
    let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_LIGHT_MAXIMUM_RECORDS}u));
    for(var lightIndex=0u;lightIndex<${SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights}u;lightIndex+=1u){
      if(lightIndex>=lightCount){break;}
      let light=dryLighting.lights[lightIndex];
      if(light.identity.w!=dryLighting.metadata.y){continue;}
      let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA;
      let globalIllumination=(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u;
      let sampleCount=select(select(1u,select(dry.tuningCounts1.x,dry.tuningCounts0.w,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL}),area),1u,globalIllumination);
      var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){
        if(sampleIndex>=sampleCount){continue;}
        let sample=dryLightSample(light,sampleIndex,position);
        if(sample.valid==0u||dot(geometricNormal,sample.towardLight)<=0.0){continue;}
        let maximumDistance=select(directionalLightSceneExitDistance(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);
        if(maximumDistance<=0.0){continue;}
        let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,sample.towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);
        // Mirror of the inline path's normal escape and finite-emitter
        // clearance: the reduced-rate texel must hold the same visibility the
        // fallback band computes inline.
        let coneCell_m=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
        let coneEscape_m=coneCell_m*dry.tuningRays1.z;
        let coneMaxRaw_m=max(0.0,ray.tMax_m-coneEscape_m*dot(geometricNormal,sample.towardLight));
        let coneMax_m=coneMaxRaw_m-select(0.0,dry.tuningRays1.w*coneCell_m,sample.finiteDistance_m>0.0);
        let cone=dryConeVisibility(ray.origin_m+geometricNormal*coneEscape_m,sample.towardLight,dry.tuningRays1.y,coneMax_m,geometricNormal,sample.finiteDistance_m>0.0);
        visibility+=mix(1.0,cone.transmittance,dry.tuningRays0.y);
      }
      let packedVisibility=clamp(visibility/f32(sampleCount),0.0,1.0);
      if(lightIndex<3u){visibility0[1u+lightIndex]=packedVisibility;}
      else if(lightIndex<7u){visibility1[lightIndex-3u]=packedVisibility;}
      else{visibility2.x=packedVisibility;}
    }
  }
  return dryPrepassPack(visibility0,visibility1,visibility2);
}
@fragment fn dryPrepassVisibilityMain(input:VertexOut)->@location(0) vec2u{
  let coordinate=vec2i(input.position.xy);let geometry=textureLoad(dryPrepassGeometryTexture,coordinate,0);
  if(geometry.x<=0.0){return vec2u(0xffffffffu);}
  let identity=textureLoad(dryPrepassIdentityTexture,coordinate,0).x;let metadata=u32(round(geometry.w));
  let opaque=DryHit(geometry.x,dryPrepassDecodeNormal(geometry.yz),identity&0xffffu,identity>>16u,metadata&15u,(metadata>>4u)&15u,(metadata>>8u)&3u,(metadata>>10u)&1u,0.0,vec3u(0u));
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);
  return dryPrepassTraceVisibility(opaque,ro,rd);
}
@fragment fn dryPrepassShadeMain(input:VertexOut)->@location(0) vec4f{
  let coordinate=vec2i(input.position.xy);let geometry=textureLoad(dryPrepassGeometryTexture,coordinate,0);
  if(geometry.x<=0.0){return vec4f(0.0);}
  let identity=textureLoad(dryPrepassIdentityTexture,coordinate,0).x;let metadata=u32(round(geometry.w));
  let opaque=DryHit(geometry.x,dryPrepassDecodeNormal(geometry.yz),identity&0xffffu,identity>>16u,metadata&15u,(metadata>>4u)&15u,(metadata>>8u)&3u,(metadata>>10u)&1u,0.0,vec3u(0u));
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);
  // GLOBAL stores a current-frame, world-space surface GI result. The full-rate
  // consumer reconstructs it only across matching identity/depth/normal and
  // retains exact inline cones at discontinuities.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u){dryPrepassGiState=0u;let ignoredBodyOwner=select(DRY_OWNER_NONE,opaque.ownerId,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let gi=dryGlobalIllumination(ro+rd*opaque.t,opaque.normal,ignoredBodyOwner);return vec4f(gi.radiance,gi.visibility);}
  // Until GLOBAL data is ready, rigid opaque radiance remains exact at full
  // rate, so avoid doing an unusable complete material evaluation here.
  if(opaque.motionKind!=DRY_GBUFFER_MOTION_STATIC){return vec4f(0.0);}
  let packed=textureLoad(dryPrepassVisibilityKeyTexture,coordinate,0);dryPrepassData0=dryPrepassUnpack0(packed);dryPrepassData1=dryPrepassUnpack1(packed);dryPrepassData2=dryPrepassUnpack2(packed);
  dryPrepassState=1u;dryPrepassRadianceState=0u;dryConeFallback=0u;dryCurrentLightSlot=0xffffffffu;dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  return vec4f(shadeDryOpaque(opaque,ro,rd),opaque.t);
}
` : "";
  const splitEntryWGSL = split ? /* wgsl */ `struct DryVisibilityOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @builtin(frag_depth) hardwareDepth:f32,
}
fn drySplitVisibilityOut(targetsIn:SvoGBufferTargets,hardwareDepth:f32)->DryVisibilityOut{
  return DryVisibilityOut(targetsIn.packedSurface,targetsIn.identityMedia,hardwareDepth);
}
@fragment fn dryVisibilityMain(input:VertexOut)->DryVisibilityOut{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryPrimaryNodeVisits=0u;dryPrimaryLeafVisits=0u;dryPrimaryEmptyBrickSkips=0u;dryPrimaryVoxelWorkItems=0u;dryPrimaryExactTests=0u;dryPrimaryMaximumDepth=0u;dryShadowNodeVisits=0u;dryShadowLeafVisits=0u;dryShadowWorkItems=0u;dryMipSteps=0u;dryTraversalFailure=0u;dryThickGlassEnabled=0u;
  let opaque=${splitPrimaryTraceWGSL};${splitVisibilityGlassDiscoveryWGSL}
  ${splitVisibilityGlassReturnWGSL}
  if(opaque.t<DRY_MISS){let media=dryMediumPair(rd,opaque.normal,DRY_MEDIUM_OPAQUE);let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}let targets=svoGBufferSurface(vec3f(0.0),opaque.t,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);return drySplitVisibilityOut(targets,dryHardwareDepth(opaque.t,rd,forward));}
  return drySplitVisibilityOut(svoGBufferMiss(vec3f(0.0),0u,generation,DRY_GBUFFER_NO_INTERSECTION,0u),0.0);
}
@fragment fn dryLightingMain(input:VertexOut)->@location(0) vec4f{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryPrimaryNodeVisits=0u;dryPrimaryLeafVisits=0u;dryPrimaryEmptyBrickSkips=0u;dryPrimaryVoxelWorkItems=0u;dryPrimaryExactTests=0u;dryPrimaryMaximumDepth=0u;dryShadowNodeVisits=0u;dryShadowLeafVisits=0u;dryShadowWorkItems=0u;dryMipSteps=0u;dryTraversalFailure=0u;dryThickGlassEnabled=0u;
  let coordinate=vec2i(input.position.xy);let geometry=drySplitGeometryAt(coordinate);let opaqueIdentity=drySplitIdentityAt(coordinate);var opaque=missHit();
  let packedOpaqueMaterial=opaqueIdentity.x;${splitOpaqueMaterialDecodeWGSL}if(geometry.w<DRY_MISS){let metadata=opaqueIdentity.y;opaque=DryHit(geometry.w,geometry.xyz,opaqueMaterial,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));}
  ${prepassResolveCallWGSL}var glass=dryGlassMiss();${splitGlassKeyLoadWGSL}if(glassKey>0u){let recordIndex=glassKey-1u;if(recordIndex<dry.terrain.y&&recordIndex<arrayLength(&glassPanes)){let record=glassPanes[recordIndex];let candidate=svoThinGlassIntersect(record,ro,rd,0.0,opaque.t,1e-6,record.extentIorEpsilon.w);if(candidate.valid!=0u){glass=DryGlassHit(candidate,recordIndex);}}}var color=shadeDryOpaque(opaque,ro,rd);var depth=opaque.t;let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;if(glassVisible){let glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);return dryCostOverlay(vec4f(max(color*vignette,vec3f(0.0)),select(0.0,depth,depth<DRY_MISS)));
}
` : "";
  const prepassVisibilityStoreWGSL = coneFanout
    ? "textureStore(dryPrepassVisibilityWrite,coordinate,vec4u(0xffffffffu));"
    : "textureStore(dryPrepassVisibilityWrite,coordinate,vec4u(dryPrepassTraceVisibility(opaque,ro,rd),0u,0u));";
  const prepassFanoutDeclarationWGSL = coneFanout
    ? "@group(1) @binding(8) var dryPrepassFanoutReceiverWrite:texture_storage_2d<rgba32float,write>;"
    : "";
  const prepassFanoutMissStoreWGSL = coneFanout
    ? "textureStore(dryPrepassFanoutReceiverWrite,coordinate,vec4f(0.0));"
    : "";
  const prepassFanoutHitStoreWGSL = coneFanout
    ? "textureStore(dryPrepassFanoutReceiverWrite,coordinate,vec4f(opaque.t,normalize(opaque.normal)));"
    : "";
  const prepassFromPrimaryEntryWGSL = reduced && split ? /* wgsl */ `struct DryPrepassBoundaryQueue{count:atomic<u32>,coordinates:array<u32>}
@group(1) @binding(4) var dryPrepassVisibilityWrite:texture_storage_2d<rg32uint,write>;
@group(1) @binding(5) var dryPrepassGeometryWrite:texture_storage_2d<rgba16float,write>;
@group(1) @binding(6) var dryPrepassIdentityWrite:texture_storage_2d<r32uint,write>;
@group(1) @binding(7) var<storage,read_write> dryPrepassBoundaryQueue:DryPrepassBoundaryQueue;
${prepassFanoutDeclarationWGSL}
@compute @workgroup_size(1) fn dryPrepassResetMain(){atomicStore(&dryPrepassBoundaryQueue.count,0u);}
fn dryPrepassRay(coordinate:vec2u,dimensions:vec2u)->mat2x3f{let uv=vec2f((f32(coordinate.x)+.5)/f32(dimensions.x),1.0-(f32(coordinate.y)+.5)/f32(dimensions.y));let ndc=uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);return mat2x3f(ro,rd);}
fn dryPrepassStore(coordinate:vec2i,opaque:DryHit,ro:vec3f,rd:vec3f){if(!(opaque.t<DRY_MISS)){textureStore(dryPrepassVisibilityWrite,coordinate,vec4u(0xffffffffu));textureStore(dryPrepassGeometryWrite,coordinate,vec4f(0.0));textureStore(dryPrepassIdentityWrite,coordinate,vec4u(0xffffffffu));${prepassFanoutMissStoreWGSL}return;}textureStore(dryPrepassGeometryWrite,coordinate,vec4f(opaque.t,dryPrepassEncodeNormal(opaque.normal),f32(dryPrepassHitMetadata(opaque))));textureStore(dryPrepassIdentityWrite,coordinate,vec4u(dryPrepassPackIdentity(opaque),0u,0u,0u));${prepassFanoutHitStoreWGSL}${prepassVisibilityStoreWGSL}}
@compute @workgroup_size(8,8) fn dryPrepassCoherentMain(@builtin(global_invocation_id) globalId:vec3u){
  let dimensions=textureDimensions(dryPrepassGeometryWrite);if(any(globalId.xy>=dimensions)){return;}let coordinate=vec2i(globalId.xy);let ray=dryPrepassRay(globalId.xy,dimensions);
  let fullDimensions=textureDimensions(drySplitGeometryRead);let maximumCoordinate=vec2i(fullDimensions)-vec2i(1);let sampleBase=clamp(vec2i(floor((vec2f(globalId.xy)+vec2f(.5))*vec2f(fullDimensions)/vec2f(dimensions)-vec2f(.5))),vec2i(0),maximumCoordinate);
  var primaryGeometry:array<vec4f,4>;var primaryIdentity:array<vec4u,4>;var allMiss=true;
  for(var sample=0u;sample<4u;sample+=1u){let sourceCoordinate=clamp(sampleBase+vec2i(i32(sample&1u),i32(sample>>1u)),vec2i(0),maximumCoordinate);primaryGeometry[sample]=drySplitGeometryAt(sourceCoordinate);primaryIdentity[sample]=drySplitIdentityAt(sourceCoordinate);allMiss=allMiss&&!(primaryGeometry[sample].w<DRY_MISS);}
  if(allMiss){dryPrepassStore(coordinate,missHit(),ray[0],ray[1]);return;}
  let referenceGeometry=primaryGeometry[3];let referenceIdentity=primaryIdentity[3];var homogeneous=referenceGeometry.w<DRY_MISS;
  for(var sample=0u;sample<3u;sample+=1u){let geometry=primaryGeometry[sample];let hit=geometry.w<DRY_MISS;let sameIdentity=(primaryIdentity[sample].x&0x8000ffffu)==(referenceIdentity.x&0x8000ffffu)&&primaryIdentity[sample].y==referenceIdentity.y;let depthClose=abs(geometry.w-referenceGeometry.w)<=max(.0001,.01*referenceGeometry.w);let normalClose=dot(normalize(geometry.xyz),normalize(referenceGeometry.xyz))>=.9999;homogeneous=homogeneous&&hit&&sameIdentity&&depthClose&&normalClose;}
  if(!homogeneous){${inlineBoundaryWGSL}}
  let metadata=referenceIdentity.y;let packedMaterial=referenceIdentity.x;let material=select(packedMaterial&0xffffu,0x80000000u|(packedMaterial&0xffffu),(packedMaterial&0x80000000u)!=0u);let opaque=DryHit(referenceGeometry.w,normalize(referenceGeometry.xyz),material,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));dryPrepassStore(coordinate,opaque,ray[0],ray[1]);
}
@compute @workgroup_size(64) fn dryPrepassBoundaryMain(@builtin(global_invocation_id) globalId:vec3u){
  let queueCount=atomicLoad(&dryPrepassBoundaryQueue.count);if(globalId.x>=queueCount){return;}let dimensions=textureDimensions(dryPrepassGeometryWrite);let packedCoordinate=dryPrepassBoundaryQueue.coordinates[globalId.x];let coordinate=vec2u(packedCoordinate%dimensions.x,packedCoordinate/dimensions.x);let ray=dryPrepassRay(coordinate,dimensions);dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;let opaque=traceOpaqueScene(ray[0],ray[1]);dryPrepassStore(vec2i(coordinate),opaque,ray[0],ray[1]);
}
` : "";
  const worldGiCacheHelpersWGSL = reduced && split ? /* wgsl */ `
struct DryWorldGiCacheMetadata{state:atomic<u32>,signature:u32}
struct DryWorldGiCachePayload{radianceRg:u32,radianceBv:u32}
struct DryWorldGiCache{
  metadata:array<DryWorldGiCacheMetadata,${SVO_DRY_WORLD_GI_CACHE_CONTRACT.entryCount}>,
  payload:array<DryWorldGiCachePayload,${SVO_DRY_WORLD_GI_CACHE_CONTRACT.entryCount}>,
}
struct DryWorldGiCacheKey{readyState:u32,signature:u32,start:u32}
struct DryWorldGiCacheLookup{value:DryGlobalIllumination,hit:u32,claimSlot:u32,claimState:u32}
struct DryWorldGiBodyInfluence{bodyMask:u32,movingMask:u32,signature:u32}
struct DryWorldGiFrame{
  bodySignature:u32,
  movingBodyCount:u32,
  bodyCount:u32,
  reserved:u32,
  cameraPosition:vec4f,
  cameraForwardAspect:vec4f,
  cameraRight:vec4f,
  cameraUp:vec4f,
  bodySignatures:array<u32,12>,
}
@group(2) @binding(7) var<storage,read_write> dryWorldGiCache:DryWorldGiCache;
@group(2) @binding(8) var dryWorldGiOutput:texture_storage_2d<rgba16float,write>;
@group(2) @binding(9) var<storage,read_write> dryWorldGiFrame:DryWorldGiFrame;
fn dryWorldGiHash(valueIn:u32)->u32{
  var value=valueIn;value^=value>>16u;value*=0x7feb352du;value^=value>>15u;value*=0x846ca68bu;return value^(value>>16u);
}
fn dryWorldGiHashAdd(hash:u32,value:u32)->u32{return dryWorldGiHash(hash^(value+0x9e3779b9u+(hash<<6u)+(hash>>2u)));}
fn dryWorldGiMorton4(value:vec3u)->u32{
  var morton=0u;
  for(var bit=0u;bit<4u;bit+=1u){
    morton|=((value.x>>bit)&1u)<<(bit*3u);
    morton|=((value.y>>bit)&1u)<<(bit*3u+1u);
    morton|=((value.z>>bit)&1u)<<(bit*3u+2u);
  }
  return morton;
}
fn dryWorldGiSpatialStart(quantized:vec3i)->u32{
  let coordinate=vec3u(max(quantized,vec3i(0)));
  let tileHash=dryWorldGiHashAdd(dryWorldGiHashAdd(coordinate.x>>4u,coordinate.y>>4u),coordinate.z>>4u);
  // Morton-local low bits keep neighbouring shader lanes in the same cache
  // lines; six hashed high bits distribute repeating world tiles.
  return ((tileHash&63u)<<12u)|dryWorldGiMorton4(coordinate&vec3u(15u));
}
fn dryWorldGiKey(position:vec3f,normalIn:vec3f,bodyNamespace:u32)->DryWorldGiCacheKey{
  let cell=max(dry.mapping.cellSize,vec3f(1e-6));let quantized=vec3i(floor((position-dry.nodeMipOrigin.xyz)/(cell*.25)));
  // Prepass normals are decoded normalized; preserving that contract avoids a
  // normalize in every cache query.
  let normal=normalIn;let normalByte=vec3u(round(clamp(normal*.5+vec3f(.5),vec3f(0.0),vec3f(1.0))*255.0));
  let packedNormal=normalByte.x|(normalByte.y<<8u)|(normalByte.z<<16u);
  var first=dryWorldGiHashAdd(0x811c9dc5u,bitcast<u32>(quantized.x));
  first=dryWorldGiHashAdd(first,bitcast<u32>(quantized.y));first=dryWorldGiHashAdd(first,bitcast<u32>(quantized.z));
  first=dryWorldGiHashAdd(first,packedNormal);first=dryWorldGiHashAdd(first,dry.nodeMip.x);
  first=dryWorldGiHashAdd(first,bitcast<u32>(dry.giCones.x));first=dryWorldGiHashAdd(first,u32(round(dry.giCones.y)));
  first=dryWorldGiHashAdd(first,dry.tuningCounts0.y);first=dryWorldGiHashAdd(first,bitcast<u32>(dry.giLighting.x));
  first=dryWorldGiHashAdd(first,bitcast<u32>(dry.giLighting.y));first=dryWorldGiHashAdd(first,dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion}u);
  let second=dryWorldGiHashAdd(dryWorldGiHash(first^0x68bc21ebu),bodyNamespace);
  let ready=(first&0x3fffffffu)|0x80000000u;return DryWorldGiCacheKey(ready,second,dryWorldGiSpatialStart(quantized));
}
fn dryWorldGiFind(key:DryWorldGiCacheKey)->DryWorldGiCacheLookup{
  var claimSlot=0xffffffffu;var claimState=0u;
  for(var probe=0u;probe<${SVO_DRY_WORLD_GI_CACHE_CONTRACT.probeCount}u;probe+=1u){
    let slot=(key.start+probe*probe)&${SVO_DRY_WORLD_GI_CACHE_CONTRACT.entryCount - 1}u;let state=atomicLoad(&dryWorldGiCache.metadata[slot].state);
    if(state==key.readyState&&dryWorldGiCache.metadata[slot].signature==key.signature){
      let payload=dryWorldGiCache.payload[slot];
      let rg=unpack2x16float(payload.radianceRg);let bv=unpack2x16float(payload.radianceBv);
      let verifiedState=atomicLoad(&dryWorldGiCache.metadata[slot].state);
      if(verifiedState==key.readyState&&dryWorldGiCache.metadata[slot].signature==key.signature){
        return DryWorldGiCacheLookup(DryGlobalIllumination(vec3f(rg,bv.x),bv.y),1u,claimSlot,claimState);
      }
    }
    if(state==0u){claimSlot=slot;claimState=0u;break;}
    // A ready entry is safe to replace after a compare-exchange claim. Never
    // select state 1, which denotes a writer currently publishing its payload.
    if(state!=1u){claimSlot=slot;claimState=state;}
  }
  return DryWorldGiCacheLookup(DryGlobalIllumination(vec3f(0.0),1.0),0u,claimSlot,claimState);
}
fn dryWorldGiInsert(key:DryWorldGiCacheKey,slot:u32,claimState:u32,value:DryGlobalIllumination){
  if(slot==0xffffffffu){return;}let claimed=atomicCompareExchangeWeak(&dryWorldGiCache.metadata[slot].state,claimState,1u);
  if(!claimed.exchanged){return;}dryWorldGiCache.metadata[slot].signature=key.signature;
  dryWorldGiCache.payload[slot].radianceRg=pack2x16float(value.radiance.xy);
  dryWorldGiCache.payload[slot].radianceBv=pack2x16float(vec2f(value.radiance.z,value.visibility));
  atomicStore(&dryWorldGiCache.metadata[slot].state,key.readyState);
}
fn dryWorldGiBodyInfluence(position:vec3f,ignoredBodyOwner:u32)->DryWorldGiBodyInfluence{
  let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let bodyCount=min(u32(round(max(uniforms.options.z,0.0))),12u);
  var bodyMask=0u;var movingMask=0u;var signature=0x4f1bbcdcu;
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}if(bodyIndex==ignoredBodyOwner){continue;}let body=bodies[bodyIndex];
    let influence=body.positionRadius.w+max(
      minimumVoxel*${SVO_DRY_WORLD_GI_CACHE_CONTRACT.dynamicInfluenceCells}.0,
      body.positionRadius.w*${SVO_DRY_WORLD_GI_CACHE_CONTRACT.dynamicInfluenceBodyRadii}.0);
    let delta=position-body.positionRadius.xyz;
    if(dot(delta,delta)<=influence*influence){
      let bodyBit=1u<<bodyIndex;bodyMask|=bodyBit;
      signature=dryWorldGiHashAdd(signature,dryWorldGiFrame.bodySignatures[bodyIndex]);
      let motion=rigidMotion[bodyIndex];
      if(motion.linearVelocityDisplacement.w>1e-7||motion.angularVelocityAngle.w>1e-7){movingMask|=bodyBit;}
    }
  }
  return DryWorldGiBodyInfluence(bodyMask,movingMask,signature);
}
` : "";
  const worldGiCacheEntryWGSL = reduced && split ? /* wgsl */ `
@compute @workgroup_size(1) fn dryWorldGiFrameMain(){
  let bodyCount=min(u32(round(max(uniforms.options.z,0.0))),12u);
  var signature=dryWorldGiHashAdd(0x27d4eb2du,bodyCount);var movingBodyCount=0u;
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}let body=bodies[bodyIndex];let motion=rigidMotion[bodyIndex];
    var bodySignature=dryWorldGiHashAdd(0x85ebca6bu,bodyIndex);
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.positionRadius.x));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.positionRadius.y));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.positionRadius.z));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.positionRadius.w));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.halfSizeShape.x));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.halfSizeShape.y));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.halfSizeShape.z));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.halfSizeShape.w));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.orientation.x));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.orientation.y));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.orientation.z));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.orientation.w));
    dryWorldGiFrame.bodySignatures[bodyIndex]=bodySignature;signature=dryWorldGiHashAdd(signature,bodySignature);
    if(motion.linearVelocityDisplacement.w>1e-7||motion.angularVelocityAngle.w>1e-7){movingBodyCount+=1u;}
  }
  let origin=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-origin);
  let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  dryWorldGiFrame.bodySignature=signature;dryWorldGiFrame.movingBodyCount=movingBodyCount;dryWorldGiFrame.bodyCount=bodyCount;
  dryWorldGiFrame.cameraPosition=vec4f(origin,0.0);
  dryWorldGiFrame.cameraForwardAspect=vec4f(forward,uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72);
  dryWorldGiFrame.cameraRight=vec4f(right,0.0);dryWorldGiFrame.cameraUp=vec4f(up,0.0);
}
fn dryWorldGiFrameRay(coordinate:vec2u,dimensions:vec2u)->mat2x3f{
  let uv=vec2f((f32(coordinate.x)+.5)/f32(dimensions.x),1.0-(f32(coordinate.y)+.5)/f32(dimensions.y));
  let ndc=uv*2.0-1.0;let rd=normalize(dryWorldGiFrame.cameraForwardAspect.xyz
    +dryWorldGiFrame.cameraRight.xyz*ndc.x*dryWorldGiFrame.cameraForwardAspect.w
    +dryWorldGiFrame.cameraUp.xyz*ndc.y*.72);
  return mat2x3f(dryWorldGiFrame.cameraPosition.xyz,rd);
}
@compute @workgroup_size(8,8) fn dryWorldGiCacheMain(@builtin(global_invocation_id) globalId:vec3u){
  let dimensions=textureDimensions(dryWorldGiOutput);if(any(globalId.xy>=dimensions)){return;}let coordinate=vec2i(globalId.xy);
  let geometry=textureLoad(dryPrepassGeometryTexture,coordinate,0);
  if(geometry.x<=0.0){textureStore(dryWorldGiOutput,coordinate,vec4f(0.0,0.0,0.0,1.0));return;}
  let identity=textureLoad(dryPrepassIdentityTexture,coordinate,0).x;let metadata=u32(round(geometry.w));
  let opaque=DryHit(geometry.x,dryPrepassDecodeNormal(geometry.yz),identity&0xffffu,identity>>16u,metadata&15u,(metadata>>4u)&15u,(metadata>>8u)&3u,(metadata>>10u)&1u,0.0,vec3u(0u));
  let ray=dryWorldGiFrameRay(globalId.xy,dimensions);let position=ray[0]+ray[1]*opaque.t;
  let ignoredBodyOwner=select(DRY_OWNER_NONE,opaque.ownerId,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);
  let influence=dryWorldGiBodyInfluence(position,ignoredBodyOwner);
  // Only a moving body's own bounded neighbourhood is recomputed. A cached
  // static-body neighbourhood is keyed solely by the bodies local to it, so an
  // unrelated moving body cannot invalidate that cache line.
  if(influence.movingMask!=0u){
    dryWorldGiIgnoreRigidBodies=0u;dryWorldGiBodyMask=influence.bodyMask;dryPrepassGiState=0u;
    let dynamicValue=dryGlobalIllumination(position,opaque.normal,ignoredBodyOwner);
    textureStore(dryWorldGiOutput,coordinate,vec4f(dynamicValue.radiance,dynamicValue.visibility));return;
  }
  let bodyAware=influence.bodyMask!=0u;let bodyNamespace=select(0x4f1bbcdcu,influence.signature,bodyAware);
  let key=dryWorldGiKey(position,opaque.normal,bodyNamespace);let cached=dryWorldGiFind(key);
  if(cached.hit!=0u){textureStore(dryWorldGiOutput,coordinate,vec4f(cached.value.radiance,cached.value.visibility));return;}
  dryWorldGiIgnoreRigidBodies=select(1u,0u,bodyAware);dryWorldGiBodyMask=influence.bodyMask;dryPrepassGiState=0u;
  let value=dryGlobalIllumination(position,opaque.normal,select(DRY_OWNER_NONE,ignoredBodyOwner,bodyAware));
  dryWorldGiInsert(key,cached.claimSlot,cached.claimState,value);
  textureStore(dryWorldGiOutput,coordinate,vec4f(value.radiance,value.visibility));
}
` : "";
  let shader = /* wgsl */ `
${svoTerrainMaterialWGSL}
${svoMaterialWGSL}
${svoProceduralMaterialWGSL}
${svoThickGlassWGSL}
${svoGBufferWGSL}
${svoPrimitiveMotionWGSL}
${svoLightWGSL}
${svoEnvironmentLightingWGSL}
${svoNodeMipSamplingWGSL}
${svoTetrahedralRadianceWGSL}
${svoTetrahedralRadianceConeCoreWGSL}
${svoFluidCoverageWGSL}
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
  // x: complete static generation; y: directory pages; z: levels; w: available.
  nodeMip:vec4u,
  nodeMipAtlas:vec4u,
  // x: derived generation; y: canonical source generation; z: pages; w: descriptors.
  wideFanout:vec4u,
  // Twelve per-level directory row starts (count of pages with level < i) as three vec4u.
  nodeMipLevelStart:array<vec4u,3>,
  // xyz: world-space origin of the node-mip lattice.
  nodeMipOrigin:vec4f,
  // Dense evolving-fluid coverage volume; valid=0 skips every fluid sample site.
  fluidCoverage:SvoFluidCoverageFrame,
  // Runtime work caps: primary leaves, cone steps, lights, stable area samples.
  tuningCounts0:vec4u,
  // Moving area, stable/moving AO, exact visibility node visits.
  tuningCounts1:vec4u,
  // Exact visibility leaves, work items, intersections, radiance reconstruction mode.
  tuningCounts2:vec4u,
  // Shadow bias/strength and AO radius/strength.
  tuningRays0:vec4f,
  // AO aperture, shadow aperture, normal escape, emitter clearance.
  tuningRays1:vec4f,
  // Packed Z slabs in nodeMipPageTable. w=0 retains sorted-directory fallback.
  nodeMipDirect:vec4u,
  nodeMipDirectLevelZ:array<vec4u,3>,
  // x: tetrahedral-radiance generation; y: complete and generation-matched.
  tetrahedralRadiance:vec4u,
  nodeMipExtent:vec4f,
  // Bounce exposure, broad occlusion, diffuse environment, direct key.
  giLighting:vec4f,
  // Aperture, cone count, reserved, reserved.
  giCones:vec4f,
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
@group(0) @binding(13) var<uniform> dryLighting:DryLightingArena;
@group(0) @binding(14) var<uniform> rigidMotion:array<SvoPrimitiveMotionRecord,12>;
@group(0) @binding(15) var<uniform> thickGlass:DryThickGlassArena;
@group(0) @binding(16) var nodeMipAtlas:texture_3d<f32>;
@group(0) @binding(17) var nodeMipSampler:sampler;
@group(0) @binding(18) var nodeMipDirectory:texture_2d<u32>;
@group(0) @binding(19) var fluidCoverageVolume:texture_3d<f32>;
@group(0) @binding(20) var nodeMipPageTable:texture_3d<u32>;
@group(0) @binding(21) var tetraRadianceLobe0:texture_3d<f32>;
@group(0) @binding(22) var tetraRadianceLobe1:texture_3d<f32>;
@group(0) @binding(23) var tetraRadianceLobe2:texture_3d<f32>;
@group(0) @binding(24) var tetraRadianceLobe3:texture_3d<f32>;
@group(0) @binding(25) var tetraRadianceBlackPages:texture_2d<u32>;

${canonicalTraversalWGSL}${screenSpaceTraversalWGSL}
${wideTraversalWGSL}${compactTraversalWGSL}${brickOccupancyHelpersWGSL}
${createSvoDryConeMarcherWGSL({ branchlessMorton: true, rangedDirectorySearch: true, fluidCoverage: true, directPageTable: true })}
var<private> dryGiPageCache:DryNodeMipPageCache;
fn dryTetraRadianceReady()->bool{return dry.tetrahedralRadiance.y!=0u&&dry.tetrahedralRadiance.x==dry.nodeMip.x&&dryNodeMipReady();}
fn dryNodeMipSceneExitDistance(position:vec3f,direction:vec3f)->f32{
  let minimum=dry.nodeMipOrigin.xyz;let maximum=minimum+dry.nodeMipExtent.xyz;var enter=0.0;var exit=DRY_MISS;
  for(var axis=0u;axis<3u;axis+=1u){if(abs(direction[axis])<=1e-9){if(position[axis]<minimum[axis]||position[axis]>maximum[axis]){return 0.0;}}
    else{let first=(minimum[axis]-position[axis])/direction[axis];let second=(maximum[axis]-position[axis])/direction[axis];enter=max(enter,min(first,second));exit=min(exit,max(first,second));if(exit<enter){return 0.0;}}}
  return max(exit,0.0);
}
fn svoTetraRadianceConeLoad(query:SvoTetraRadianceConeQuery)->SvoTetraRadianceConeSourceSample{
  if(!dryNodeMipReady()){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),0u,0u);}
  let level=min(u32(max(floor(query.lod),0.0)),dry.nodeMip.z-1u);let levelScale=exp2(f32(level));
  let virtualVoxel=(query.position_m-dry.nodeMipOrigin.xyz)/(dry.mapping.cellSize*levelScale);let pageFloor=floor(virtualVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  let pageCoordinate=vec3u(pageFloor);
  if(dryGiPageCache.generation!=dry.nodeMip.x||dryGiPageCache.level!=level||any(dryGiPageCache.coordinate!=pageCoordinate)){
    dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,vec3u(0u),dry.nodeMip.x,0u,0xffffffffu,0u);let pageIndex=dryNodeMipFind(level,pageCoordinate);
    if(pageIndex!=0xffffffffu){
      let black=textureLoad(tetraRadianceBlackPages,vec2u(pageIndex,0u),0).x;
      if(dry.nodeMipDirect.w!=0u){let physical=u32(SVO_NODE_MIP_PHYSICAL_SIZE);let atlasPages=max(dry.nodeMipAtlas.xyz/vec3u(physical),vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,atlasPage*physical,dry.nodeMip.x,1u,pageIndex,black);}
      else{let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,pageIndex);dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,entry.pageOrigin,entry.generation,1u,pageIndex,black);}
    }
  }
  if(dryGiPageCache.resident==0u){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  let local=virtualVoxel-vec3f(pageCoordinate)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);
  let opacity=svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,dryGiPageCache.pageOrigin,local);
  if(!dryTetraRadianceReady()){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,0u);}
  if(dryGiPageCache.blackRadiance!=0u){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  let uv=svoNodeMipAtlasUv(dryGiPageCache.pageOrigin,local,textureDimensions(tetraRadianceLobe0));
  return SvoTetraRadianceConeSourceSample(opacity.solidMean,svoTetraSample(tetraRadianceLobe0,tetraRadianceLobe1,tetraRadianceLobe2,tetraRadianceLobe3,nodeMipSampler,uv),1u,1u);
}
struct DryGlobalIllumination{radiance:vec3f,visibility:f32}
${worldGiCacheHelpersWGSL}
fn dryGlobalIllumination(position:vec3f,normal:vec3f,ignoredBodyOwner:u32)->DryGlobalIllumination{
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)==0u||!dryTetraRadianceReady()){return DryGlobalIllumination(vec3f(0.0),1.0);}
  ${prepassGiShortcutWGSL}
  let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let origin=position+normalize(normal)*minimumVoxel*max(dry.tuningRays1.z,1.0);var indirect=vec3f(0.0);var visibility=0.0;
  let coneCount=clamp(u32(round(dry.giCones.y)),3u,4u);let perConeBudget=max(1u,min(64u,dry.tuningCounts0.y)/coneCount);
  for(var coneIndex=0u;coneIndex<4u;coneIndex+=1u){
    if(coneIndex>=coneCount){break;}let direction=svoTetraRadianceHemisphereDirection(normal,coneIndex,coneCount,0.0);
    dryGiPageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);
    let sceneExit=dryNodeMipSceneExitDistance(origin,direction);var rigidHit=missHit();
    if(dryWorldGiIgnoreRigidBodies==0u){rigidHit=nearestBodyMaskIgnoring(origin,direction,ignoredBodyOwner,dryWorldGiBodyMask);}
    let rigidBlocked=rigidHit.t<sceneExit;
    let result=svoTetraRadianceConeTrace(SvoTetraRadianceConeConfig(origin,direction,dry.giCones.x,minimumVoxel,min(sceneExit,rigidHit.t),perConeBudget,.995,.0039215686,1u));
    let weight=svoTetraRadianceHemisphereWeight(coneIndex,coneCount);dryMipSteps+=result.coneTaps;
    // Sparse GI is fail-soft. A non-finite texture/filter result must not
    // poison the entire lighting closure (max/clamp preserve NaN), turning an
    // otherwise valid directly-lit surface black. Retain every valid cone and
    // treat an invalid cone as zero bounce with unobstructed visibility.
    let finiteRadiance=all(result.radiance==result.radiance)&&all(abs(result.radiance)<vec3f(65504.0));
    let finiteVisibility=result.transmittance==result.transmittance&&abs(result.transmittance)<65504.0;
    indirect+=select(vec3f(0.0),result.radiance,finiteRadiance)*weight;
    let visibleThroughStatic=select(1.0,result.transmittance,finiteVisibility);
    visibility+=select(visibleThroughStatic,0.0,rigidBlocked)*weight;
  }
  let occlusionStrength=select(0.0,dry.giLighting.y,(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion}u)!=0u);
  return DryGlobalIllumination(max(indirect,vec3f(0.0))*dry.giLighting.x,mix(1.0,clamp(visibility,0.0,1.0),occlusionStrength));
}
${prepassDeclarationsWGSL}${splitDeclarationsWGSL}${worldGiCacheEntryWGSL}fn dryDiagnosticControl()->u32{return u32(round(max(uniforms.options.x,0.0)));}
fn dryDiagnosticMaximumNodeVisits()->u32{return clamp(dryDiagnosticControl()&511u,1u,256u);}
fn dryDiagnosticMaximumDepth()->u32{return clamp(dryDiagnosticControl()>>9u,1u,21u);}
fn dryTraverse(ray:SvoRay,mapping:SvoMapping)->SvoTraversalHit{return svoTraverseWithDepthLimit(ray,mapping,dryDiagnosticMaximumDepth());}
${traversalCursorWGSL}
${primaryTraversalCursorWGSL}
${leafAccessWGSL}
${svoPrimitiveWGSL}
${unifiedLightingShaderLibrary}
${svoThinGlassWGSL}
${svoVisibilityPreludeWGSL}

const DRY_MISS:f32 = 3.402823e38;
const REQUIRED_FIELDS:u32 = ${SVO_DRY_SCENE_REQUIRED_VALID_FIELDS}u; // topology | static geometry | material owner
const DRY_OWNER_NONE:u32=0xffffu;
const DRY_MEDIUM_GLASS:u32=2u;const DRY_MEDIUM_OPAQUE:u32=3u;
const DRY_GBUFFER_FIELD_ANALYTIC:u32=4u;const DRY_GBUFFER_FIELD_TERRAIN:u32=5u;
const DRY_GBUFFER_MOTION_STATIC:u32=0u;const DRY_GBUFFER_MOTION_RIGID:u32=1u;
const DRY_GBUFFER_HARD_FEATURE:u32=256u;const DRY_GBUFFER_NO_INTERSECTION:u32=1u;
const DRY_GBUFFER_WORK_EXHAUSTED:u32=2u;const DRY_GBUFFER_INVALID_FIELD:u32=3u;
const DRY_REVERSED_Z_NEAR_M:f32=${SVO_DRY_SCENE_REVERSED_Z_NEAR_M};
var<private> dryVisibilityIgnoredOwner:u32;var<private> dryThickGlassEnabled:u32;var<private> dryThickGlassFailure:u32;
var<private> dryWorldGiIgnoreRigidBodies:u32;
var<private> dryWorldGiBodyMask:u32=0xffffffffu;
var<private> dryPrimaryNodeVisits:u32;var<private> dryPrimaryLeafVisits:u32;var<private> dryPrimaryEmptyBrickSkips:u32;var<private> dryPrimaryVoxelWorkItems:u32;var<private> dryPrimaryExactTests:u32;var<private> dryPrimaryMaximumDepth:u32;var<private> dryShadowNodeVisits:u32;var<private> dryShadowLeafVisits:u32;var<private> dryShadowWorkItems:u32;var<private> dryMipSteps:u32;var<private> dryTraversalFailure:u32;
fn dryConfiguredMapping()->SvoMapping{
  var mapping=dry.mapping;
  mapping.maxVisits=min(mapping.maxVisits,dryDiagnosticMaximumNodeVisits());
  return mapping;
}
fn dryNormalizedCost(value:f32,reference:f32)->f32{
  if(!(value>0.0)){return 0.0;}
  // A reference is the live configured work envelope, not the much larger
  // compile-time hard ceiling. The concave lift keeps ordinary nonzero work
  // out of the visually indistinguishable purple floor while preserving zero.
  let logarithmic=clamp(log2(1.0+value)/log2(1.0+max(reference,1.0)),0.0,1.0);
  return clamp(.14+.86*pow(logarithmic,.55),0.0,1.0);
}
fn dryCostRamp(valueIn:f32)->vec3f{
  let value=clamp(valueIn,0.0,1.0);
  if(value<.16){return mix(vec3f(.031,.020,.122),vec3f(.141,.278,1.0),value/.16);}
  if(value<.34){return mix(vec3f(.141,.278,1.0),vec3f(0.0,.851,1.0),(value-.16)/.18);}
  if(value<.52){return mix(vec3f(0.0,.851,1.0),vec3f(0.0,1.0,.522),(value-.34)/.18);}
  if(value<.69){return mix(vec3f(0.0,1.0,.522),vec3f(.918,1.0,0.0),(value-.52)/.17);}
  if(value<.84){return mix(vec3f(.918,1.0,0.0),vec3f(1.0,.522,0.0),(value-.69)/.15);}
  if(value<.95){return mix(vec3f(1.0,.522,0.0),vec3f(1.0,.09,.302),(value-.84)/.11);}
  return mix(vec3f(1.0,.09,.302),vec3f(1.0),(value-.95)/.05);
}
fn dryCostOverlay(radianceDepth:vec4f)->vec4f{
  let mode=u32(round(max(uniforms.cameraPosition.w,0.0)));if(mode==0u){return radianceDepth;}
  let depthReference=f32(max(1u,min(dryDiagnosticMaximumDepth(),dry.mapping.maximumDepth)));
  let depthCost=dryNormalizedCost(f32(dryPrimaryMaximumDepth),depthReference);
  let nodeCost=dryNormalizedCost(f32(dryPrimaryNodeVisits),f32(dryDiagnosticMaximumNodeVisits()));
  let primaryLeafBudget=f32(max(dry.tuningCounts0.x,1u));
  let brickCost=dryNormalizedCost(f32(dryPrimaryLeafVisits),primaryLeafBudget);
  let emptyBrickCost=dryNormalizedCost(f32(dryPrimaryEmptyBrickSkips),primaryLeafBudget);
  let voxelReference=primaryLeafBudget*f32(max(dry.mapping.brickSize,1u));
  let voxelCost=dryNormalizedCost(f32(dryPrimaryVoxelWorkItems),voxelReference);
  let exactTestCost=dryNormalizedCost(f32(dryPrimaryExactTests),primaryLeafBudget);
  let visibilityRayBudget=f32(max(dry.tuningCounts0.z+dry.tuningCounts1.y,1u));
  let shadowNodeCost=dryNormalizedCost(f32(dryShadowNodeVisits),f32(dry.tuningCounts1.w)*visibilityRayBudget);
  let shadowBrickCost=dryNormalizedCost(f32(dryShadowLeafVisits),f32(dry.tuningCounts2.x)*visibilityRayBudget);
  let shadowVoxelCost=dryNormalizedCost(f32(dryShadowWorkItems),f32(dry.tuningCounts2.y)*visibilityRayBudget);
  let shadowCost=max(shadowNodeCost,max(shadowBrickCost,shadowVoxelCost));
  let mipCost=dryNormalizedCost(f32(dryMipSteps),f32(max(dry.tuningCounts0.y,1u))*2.0);
  let primaryCost=max(max(depthCost,nodeCost),max(max(brickCost,emptyBrickCost),max(voxelCost,exactTestCost)));
  let compositeCost=1.0-(1.0-primaryCost)*(1.0-shadowCost)*(1.0-mipCost);
  var cost=depthCost;
  if(mode==${svoCostOverlayCode("node-visits")}u){cost=nodeCost;}
  else if(mode==${svoCostOverlayCode("brick-tests")}u){cost=brickCost;}
  else if(mode==${svoCostOverlayCode("empty-brick-skips")}u){cost=emptyBrickCost;}
  else if(mode==${svoCostOverlayCode("voxel-work")}u){cost=voxelCost;}
  else if(mode==${svoCostOverlayCode("exact-tests")}u){cost=exactTestCost;}
  else if(mode==${svoCostOverlayCode("shadow-node-visits")}u){cost=shadowNodeCost;}
  else if(mode==${svoCostOverlayCode("shadow-brick-visits")}u){cost=shadowBrickCost;}
  else if(mode==${svoCostOverlayCode("shadow-voxel-work")}u){cost=shadowVoxelCost;}
  else if(mode==${svoCostOverlayCode("shadow-work")}u){cost=shadowCost;}
  else if(mode==${svoCostOverlayCode("mip-steps")}u){cost=mipCost;}
  else if(mode==${svoCostOverlayCode("total-cost")}u){cost=compositeCost;}
  var overlayColor=dryCostRamp(cost);
  if(mode==${svoCostOverlayCode("work-composition")}u){
    let workSum=primaryCost+shadowCost+mipCost;
    let workHue=(primaryCost*vec3f(0.0,.851,1.0)+shadowCost*vec3f(1.0,.078,.576)+mipCost*vec3f(1.0,.902,0.0))/max(workSum,1e-5);
    overlayColor=select(vec3f(.031,.020,.122),mix(vec3f(.031,.020,.122),workHue,.28+.72*compositeCost),workSum>0.0);
  }
  ${prepassOverlayWGSL}if(mode==${svoCostOverlayCode("exhaustion")}u){
    var failure=dryTraversalFailure;
    overlayColor=select(select(vec3f(.024,.235,.204),vec3f(1.0,.69,0.0),failure==1u),vec3f(1.0,0.0,.72),failure>=2u);
  }
  // Diagnostic modes are measurements, not art-direction overlays. Returning
  // the palette directly keeps scene luminance and material colour from
  // contaminating the signal; mode zero above remains the sole radiance path.
  return vec4f(overlayColor,radianceDepth.a);
}
fn dryBoundThickGlassOwner(owner:u32)->bool{
  if(dryThickGlassEnabled==0u){return false;}let count=min(thickGlass.metadata.x,${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u);
  for(var index=0u;index<${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u;index+=1u){if(index>=count){break;}if(svoThickGlassOwnerId(thickGlass.records[index])==owner){return true;}}
  return false;
}
fn dryOpaqueOwnerSuppressed(owner:u32)->bool{return owner==dry.metadata.z||owner==dryVisibilityIgnoredOwner||dryBoundThickGlassOwner(owner);}

fn missHit()->DryHit { return DryHit(DRY_MISS,vec3f(0.0,1.0,0.0),0u,DRY_OWNER_NONE,SVO_FEATURE_SMOOTH,0u,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u)); }
${screenSpaceProxyWGSL}
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
    if(a>1e-7&&b*b-a*c>=0.0){let root=sqrt(b*b-a*c);for(var rootIndex=0u;rootIndex<2u;rootIndex+=1u){let candidate=(-b+select(-root,root,rootIndex!=0u))/a;let y=localOrigin.y+localDirection.y*candidate;if(candidate>1e-4&&candidate<t&&abs(y)<=halfHeight){t=candidate;let p=localOrigin+localDirection*t;normal=normalize(vec3f(p.x,0,p.z));featureId=select(SVO_FEATURE_CYLINDER_SIDE,SVO_FEATURE_SMOOTH,shape==2);}}}
    if(shape==2){for(var side=-1.0;side<=1.0;side+=2.0){let center=vec3f(0.0,side*halfHeight,0.0);let offset=localOrigin-center;let hb=dot(offset,localDirection);let disc=hb*hb-dot(offset,offset)+radius*radius;if(disc>=0.0){let root=sqrt(disc);for(var rootIndex=0u;rootIndex<2u;rootIndex+=1u){let candidate=-hb+select(-root,root,rootIndex!=0u);if(candidate>1e-4&&candidate<t){t=candidate;normal=normalize(offset+localDirection*t);}}}}}
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

fn nearestBodyMaskIgnoring(ro:vec3f,rd:vec3f,ignoredOwner:u32,bodyMask:u32)->DryHit {
  var best=missHit(); for(var index=0u;index<12u;index+=1u){if(index>=u32(round(uniforms.options.z))){break;}if(index==ignoredOwner||(bodyMask&(1u<<index))==0u){continue;}let body=bodies[index];if(!bodyBoundingSphereVisible(ro,rd,body,0.0,best.t)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ro,rd,body,0.0,best.t)){continue;}let hit=bodyHit(ro,rd,body);if(hit.t<best.t){best=hit;best.materialId=0x80000000u|index;best.ownerId=index;}} return best;
}
fn nearestBodyIgnoring(ro:vec3f,rd:vec3f,ignoredOwner:u32)->DryHit{return nearestBodyMaskIgnoring(ro,rd,ignoredOwner,0xffffffffu);}
${prepassBodyBlockerWGSL}
fn nearestBody(ro:vec3f,rd:vec3f)->DryHit{return nearestBodyIgnoring(ro,rd,DRY_OWNER_NONE);}

fn primitiveHit(record:SvoPrimitiveRecord,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryHit {
  // Use the shared analytic ray contract directly. In particular, do not call
  // the bounded closest-point distance evaluator merely to recover an
  // ellipsoid normal after the ray quadratic has already found the surface.
  let exact=svoIntersectPrimitiveExact(record,ro,rd,max(tMin,1e-4),tMax);
  if(exact.status!=SVO_PRIMITIVE_RAY_HIT){return missHit();}
  return DryHit(exact.t_m,exact.normal.xyz,svoPrimitiveMaterialId(record),svoPrimitiveOwnerId(record),exact.featureId,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
}

fn traceLeafPayload(ro:vec3f,rd:vec3f,hit:SvoTraversalHit)->DryHit {
  ${primaryBrickSetupWGSL}
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0)); let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9)); let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));
  let tolerance=length(extent)*1.05;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(i32(dry.mapping.brickSize)))||entry>${primaryBrickExitWGSL}){break;}
    ${primaryMacroSkipWGSL}dryPrimaryVoxelWorkItems+=1u;
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex<arrayLength(&materialOwners)){
      let identity=materialOwners[payloadIndex];let owner=identity>>16u;
      if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){let primitiveIndex=owner-dry.metadata.y;if(primitiveIndex<dry.metadata.x&&primitiveIndex<arrayLength(&primitives)){dryPrimaryExactTests+=1u;let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,${primaryBrickExitWGSL}));let candidate=primitiveHit(primitives[primitiveIndex],ro,rd,max(0.0,entry-tolerance),cellExit+tolerance);if(candidate.t<DRY_MISS){return candidate;}}}
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z)); if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return missHit();
}
${macroHddaPrimaryWGSL}

fn traceStatic(ro:vec3f,rd:vec3f)->DryHit {
  if(publicationState[0]==0u||(publicationState[1]&REQUIRED_FIELDS)!=REQUIRED_FIELDS){return missHit();}
  var minimum=0.0;
  let mapping=dryConfiguredMapping();
  let leafBudget=clamp(dry.tuningCounts0.x,1u,${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u);
  var continuation:DryTraversalCursor;
  var traversalFinished=false;
  dryTraversalCursorBegin(SvoRay(ro,minimum,rd,DRY_MISS),mapping,&continuation);
  for(var leafVisit=0u;leafVisit<${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u&&leafVisit<leafBudget;leafVisit+=1u){
    let ray=SvoRay(ro,minimum,rd,DRY_MISS);
    let leaf=dryTraversalCursorNextPrimary(ray,mapping,&continuation);
    dryPrimaryNodeVisits+=leaf.visits;
    ${screenSpaceProxyTraceWGSL}
    if(leaf.status!=SVO_STATUS_HIT){
      traversalFinished=true;
      if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){dryTraversalFailure=max(dryTraversalFailure,1u);}
      else if(leaf.status!=SVO_STATUS_MISS){dryTraversalFailure=2u;}
      break;
    }
    dryPrimaryLeafVisits+=1u;
    dryPrimaryMaximumDepth=max(dryPrimaryMaximumDepth,leaf.level);
    let payloadHit=${primaryLeafTraceCallWGSL}(ro,rd,leaf);
    if(payloadHit.t<DRY_MISS){return payloadHit;}
    dryPrimaryEmptyBrickSkips+=1u;
    minimum=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);
  }
  // Reaching the uniform budget without an authoritative hierarchy miss is a
  // traversal exhaustion, not an empty scene. Keep that visible in the
  // existing failure heatmap instead of silently returning black.
  if(!traversalFinished){dryTraversalFailure=max(dryTraversalFailure,1u);}
  return missHit();
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
  ${shadowBrickSetupWGSL}
  let step=select(vec3i(-1),vec3i(1),ray.direction>=vec3f(0.0));let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));let deltaT=select(vec3f(DRY_MISS),abs(extent/ray.direction),abs(ray.direction)>vec3f(1e-9));
  let tolerance=length(extent)*1.05;var workItems=0u;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(i32(dry.mapping.brickSize)))||entry>${shadowBrickExitWGSL}||entry>ray.tMax_m){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);}
    ${shadowMacroSkipWGSL}if(workItems>=workLimit){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);}workItems+=1u;
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex>=arrayLength(&materialOwners)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}
    let identity=materialOwners[payloadIndex];let owner=identity>>16u;
    if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){
      let primitiveIndex=owner-dry.metadata.y;
      if(primitiveIndex>=dry.metadata.x||primitiveIndex>=arrayLength(&primitives)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}
      let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,min(${shadowBrickExitWGSL},ray.tMax_m)));let candidate=primitiveHit(primitives[primitiveIndex],ray.origin_m,ray.direction,max(entry-tolerance,tMin_m),cellExit+tolerance);
      if(candidate.t<DRY_MISS){return dryVisibilityStep(SVO_VIS_STEP_HIT,0u,0u,workItems,candidate.t);}
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);
}
${macroHddaShadowWGSL}

const DRY_MEDIUM_AIR:u32=0u;
fn dryThinGlassIncidentIor()->f32{return 1.0;}

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

  var cursor=max(tMin_m,0.0);var shadowContinuation:DryTraversalCursor;let initialShadowMapping=dryConfiguredMapping();dryTraversalCursorBegin(SvoRay(ray.origin_m,cursor,ray.direction,bestT),initialShadowMapping,&shadowContinuation);
  for(var leafAttempt=0u;leafAttempt<${SVO_VISIBILITY_LIMITS.leafVisits}u;leafAttempt+=1u){
    if(cursor>=bestT){break;}if(leafVisits>=remaining.leafVisits||nodeVisits>=remaining.nodeVisits){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
    var shadowMapping=dryConfiguredMapping();shadowMapping.maxVisits=min(shadowMapping.maxVisits,remaining.nodeVisits-nodeVisits);
    let leaf=dryTraversalCursorNext(SvoRay(ray.origin_m,cursor,ray.direction,bestT),shadowMapping,&shadowContinuation);nodeVisits+=leaf.visits;
    if(leaf.status==SVO_STATUS_MISS){break;}
    if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
    if(leaf.status!=SVO_STATUS_HIT){return dryVisibilityStep(SVO_VIS_STEP_INVALID,nodeVisits,leafVisits,workItems,DRY_MISS);}leafVisits+=1u;
    let payloadRay=SvoVisibilityRay(ray.origin_m,bestT,ray.direction,ray.originBias_m);let payload=${shadowLeafTraceCallWGSL}(payloadRay,tMin_m,leaf,remaining.workItems-workItems);workItems+=payload.workItems;
    if(payload.status==SVO_VIS_STEP_HIT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,payload.t_m);}if(payload.status!=SVO_VIS_STEP_MISS){return dryVisibilityStep(payload.status,nodeVisits,leafVisits,workItems,payload.t_m);}
    cursor=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);
  }
  if(cursor<bestT&&leafVisits>=remaining.leafVisits){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}

  if(terrainEnabled()){
    let terrainWork=${SVO_TERRAIN_FALLBACK_STEPS + SVO_TERRAIN_FALLBACK_REFINEMENTS + 6}u;
    if(workItems+terrainWork>remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=terrainWork;
    let terrain=traceTerrain(ray.origin_m,ray.direction);if(terrain.t>=tMin_m&&terrain.t<bestT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,terrain.t);}
  }
  let paneCount=dry.terrain.y;if(workItems+paneCount>remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=paneCount;
  let glass=traceGlass(ray.origin_m,ray.direction,tMin_m,bestT,false);if(glass.hit.valid!=0u&&glass.hit.t_m<bestT){let optics=svoThinGlassOptics(glassPanes[glass.recordIndex],glass.hit,dryThinGlassIncidentIor());bestT=glass.hit.t_m;found=true;opaque=false;glassTransmission=optics.netTransmittance;}
  if(!found){return dryVisibilityStep(SVO_VIS_STEP_MISS,nodeVisits,leafVisits,workItems,DRY_MISS);}if(opaque){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,bestT);}return dryVisibilityTransmissionStep(nodeVisits,leafVisits,workItems,bestT,glassTransmission);
}

${svoVisibilityTraceWGSL}

// Water attenuates by wavelength over distance instead of blocking: red is gone
// long before blue-green is. This is the identical Beer-Lambert term the raster
// composite applies along the view ray, so a floor seen through water and the
// same floor shaded under it agree about how much light reached it.
fn dryFluidTransmittance(depth_m:f32)->vec3f {
  if(!(depth_m>0.0)){return vec3f(1.0);}
  return unifiedBeerLambert(vec3f(${WATER_OPTICS.absorption.join(",")}),depth_m);
}

fn dryLightVisibility(position:vec3f,geometricNormal:vec3f,ownerId:u32,towardLight:vec3f,finiteDistance_m:f32)->vec3f {
  if(dot(geometricNormal,towardLight)<=0.0){return vec3f(0.0);}
  if((dry.materialPublication.w&2u)==0u){return vec3f(1.0);}
  let maximumDistance=select(directionalLightSceneExitDistance(position,towardLight),finiteDistance_m,finiteDistance_m>0.0);if(maximumDistance<=0.0){return vec3f(0.0);}
  let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);
  // GLOBAL already carries shadowed direct exitance in its radiance atlas.
  // Reuse hierarchical cone visibility for the analytic direct term instead of
  // recasting a full exact SVO ray for every receiver and every authored light.
  // Reduced shading retains its full-rate analytic rigid-body correction in
  // prepassShadowShortcutWGSL; invalid/missing cone data still falls through to
  // the bounded exact traversal below.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u){${prepassShadowShortcutWGSL}
    // The cone origin escapes the receiver's own trilinear coverage support
    // along the geometric normal: the 0.02-cell hard-ray bias alone leaves the
    // first cone samples inside the surface, whose accumulated self-occlusion
    // renders as banding. Finite emitters additionally clear the march end by
    // one cone-support width: a march ending exactly at the emitter surface
    // reads the emitter's own voxelized coverage through the last samples'
    // trilinear/mip support, and the amount aliases with the receiver's
    // distance modulo the step size as concentric rings around the light.
    let coneCell_m=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
    let coneEscape_m=coneCell_m*dry.tuningRays1.z;
    let coneMaxRaw_m=max(0.0,ray.tMax_m-coneEscape_m*dot(geometricNormal,towardLight));
    let coneMax_m=coneMaxRaw_m-select(0.0,dry.tuningRays1.w*coneCell_m,finiteDistance_m>0.0);
    let cone=dryConeVisibility(ray.origin_m+geometricNormal*coneEscape_m,towardLight,dry.tuningRays1.y,coneMax_m,geometricNormal,finiteDistance_m>0.0);
    if(cone.valid!=0u){let rigidBlocker=nearestBodyIgnoring(ray.origin_m,towardLight,ownerId);if(rigidBlocker.t<ray.tMax_m){return vec3f(1.0-dry.tuningRays0.y);}let raw=vec3f(cone.transmittance)*dryFluidTransmittance(cone.fluidDepth_m);return mix(vec3f(1.0),raw,dry.tuningRays0.y);}}
  dryVisibilityIgnoredOwner=ownerId;
  let result=svoTraceVisibility(ray,SvoVisibilityBudget(dry.tuningCounts1.w,dry.tuningCounts2.x,dry.tuningCounts2.y,dry.tuningCounts2.z),true,0.001,max(ray.originBias_m,1e-6));dryShadowNodeVisits+=result.nodeVisits;dryShadowLeafVisits+=result.leafVisits;dryShadowWorkItems+=result.workItems;if(result.status==SVO_VIS_STATUS_EXHAUSTED){dryTraversalFailure=max(dryTraversalFailure,1u);}else if(result.status==SVO_VIS_STATUS_INVALID){dryTraversalFailure=2u;}
  dryVisibilityIgnoredOwner=DRY_OWNER_NONE;
  return mix(vec3f(1.0),result.transmittance,dry.tuningRays0.y);
}

fn dryContactVisibilityRadius()->f32 {
  let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let sceneScale=max(uniforms.container.x,max(uniforms.container.y,uniforms.container.z));
  return dry.tuningRays0.z*min(sceneScale*${SVO_CONTACT_VISIBILITY_CONTRACT.maximumSceneRadiusFraction},max(cellScale*${SVO_CONTACT_VISIBILITY_CONTRACT.radiusCells}.0,sceneScale*${SVO_CONTACT_VISIBILITY_CONTRACT.minimumSceneRadiusFraction}));
}
fn dryContactVisibilityDirection(geometricNormalIn:vec3f,featureId:u32,sampleIndex:u32)->vec3f {
  let geometricNormal=normalize(geometricNormalIn);let helper=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(geometricNormal.y)>.9);var tangent=normalize(cross(helper,geometricNormal));var bitangent=cross(geometricNormal,tangent);
  if((featureId&1u)!=0u){let previous=tangent;tangent=bitangent;bitangent=-previous;}
  let signValue=select(1.0,-1.0,sampleIndex!=0u);return normalize(geometricNormal+signValue*(.55*tangent+.2*bitangent));
}
fn dryContactVisibility(position:vec3f,geometricNormal:vec3f,featureId:u32,ownerId:u32)->vec3f {
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion}u)==0u){return vec3f(1.0);}
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u&&dryNodeMipReady()){${prepassContactShortcutWGSL}
    let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(1.0);}var visibility=0.0;var coneValid=true;let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let origin=position+normalize(geometricNormal)*cellScale*.2;let coneSampleCount=select(dry.tuningCounts1.z,dry.tuningCounts1.y,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL});
    for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=coneSampleCount){break;}let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);let rotated=select(direction,normalize(direction+cross(normalize(geometricNormal),direction)*.7),sampleIndex>=2u);let cone=dryConeVisibility(origin,rotated,dry.tuningRays1.x,radius,vec3f(0.0),false);if(cone.valid==0u){coneValid=false;break;}let rigidBlocker=nearestBodyIgnoring(origin,rotated,ownerId);visibility+=select(cone.transmittance,0.0,rigidBlocker.t<radius);}if(coneValid){let raw=clamp(visibility/f32(coneSampleCount),0.0,1.0);return vec3f(mix(1.0,raw,dry.tuningRays0.w));}
  }
  if((dry.materialPublication.w&1u)==0u){return vec3f(1.0);}
  let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(0.0);}let biasCells=select(${SVO_CONTACT_VISIBILITY_CONTRACT.smoothBiasCells},${SVO_CONTACT_VISIBILITY_CONTRACT.hardFeatureBiasCells},featureId!=SVO_FEATURE_SMOOTH);var visibility=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}u;sampleIndex+=1u){let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex);let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,direction,radius,dry.mapping.cellSize,biasCells);let result=svoTraceVisibility(ray,SvoVisibilityBudget(dry.tuningCounts1.w,dry.tuningCounts2.x,dry.tuningCounts2.y,dry.tuningCounts2.z),true,0.001,max(ray.originBias_m,1e-6));dryShadowNodeVisits+=result.nodeVisits;dryShadowLeafVisits+=result.leafVisits;dryShadowWorkItems+=result.workItems;if(result.status==SVO_VIS_STATUS_INVALID||result.status==SVO_VIS_STATUS_EXHAUSTED){dryTraversalFailure=select(2u,max(dryTraversalFailure,1u),result.status==SVO_VIS_STATUS_EXHAUSTED);return vec3f(0.0);}visibility+=result.transmittance;}
  return mix(vec3f(1.0),clamp(visibility/f32(${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}),vec3f(0.0),vec3f(1.0)),dry.tuningRays0.w);
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
  let radiance=baseRadiance*(rangeFade*shapeScale);if(max(max(radiance.x,radiance.y),radiance.z)<=0.0){return dryInvalidLightSample();}
  // Point fixtures retain the finite radius of their visible emissive proxy.
  // Attenuation uses center distance, while visibility ends at the globe's
  // near surface so the source geometry cannot occlude its own light.
  let visibilityDistance=select(distance,max(0.0,distance-light.shape.x),light.identity.x==SVO_LIGHT_POINT);
  return DryLightSample(towardLight,visibilityDistance,radiance,1u);
}
fn traceStaticSolidScene(ro:vec3f,rd:vec3f)->DryHit {
  var hit=traceStatic(ro,rd);let terrain=traceTerrain(ro,rd);if(terrain.t<hit.t){hit=terrain;}return hit;
}
fn traceDrySolidScene(ro:vec3f,rd:vec3f)->DryHit {
  var hit=traceStaticSolidScene(ro,rd);let rigid=nearestBody(ro,rd);if(rigid.t<hit.t){hit=rigid;}
  return hit;
}
fn traceOpaqueScene(ro:vec3f,rd:vec3f)->DryHit {
  return traceDrySolidScene(ro,rd);
}
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
  if(hit.t>=DRY_MISS){return dryEnvironment(rd,0.0);}${screenSpaceProxyShadeWGSL}${prepassRadianceShortcutWGSL}let position=ro+rd*hit.t;let surface=dryEvaluateSurfaceMaterial(hit,position);
  if(surface.valid==0u){return vec3f(0.0);}
  let directClosure=unifiedPbrMaterial(surface.baseColor,surface.metallic,surface.roughness,vec3f(0.0),0.0,surface.specularF0,surface.specularWeight,vec3f(0.0),0.0);var direct=vec3f(0.0);var sampleBudget=0u;
  let globalIllumination=(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u;
  // GI supplements the authored lighting design; it must not replace every
  // local fixture with whichever record happens to be first (the garden's
  // first record is an intentionally weak dusk directional light). Keep all
  // configured emitters, while the sample-count selection below still limits
  // GLOBAL shading to one exact visibility sample per light.
  let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_LIGHT_MAXIMUM_RECORDS}u));
  for(var lightIndex=0u;lightIndex<${SVO_DRY_SCENE_MAX_SHADED_LIGHTS}u;lightIndex+=1u){
    if(lightIndex>=lightCount||sampleBudget>=dry.tuningCounts0.z){break;}${prepassLightSlotWGSL}let light=dryLighting.lights[lightIndex];if(light.identity.w!=dryLighting.metadata.y){continue;}let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA;let sampleCount=select(select(1u,select(dry.tuningCounts1.x,dry.tuningCounts0.w,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL}),area),1u,globalIllumination);
    for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=sampleCount||sampleBudget>=dry.tuningCounts0.z){break;}sampleBudget+=1u;let sample=dryLightSample(light,sampleIndex,position);if(sample.valid==0u||dot(hit.normal,sample.towardLight)<=0.0){continue;}let visibility=dryLightVisibility(position,hit.normal,hit.ownerId,sample.towardLight,sample.finiteDistance_m);let lighting=unifiedLightingInputWithGeometry(hit.normal,hit.normal,-rd,sample.towardLight,sample.radiance*visibility/f32(sampleCount));direct+=shadeUnifiedSurface(directClosure,lighting);}
  }
  let viewDirection=normalize(-rd);let reflected=reflect(rd,hit.normal);let diffuseColor=surface.baseColor*(1.0-surface.metallic);let f0=mix(surface.specularF0*surface.specularWeight,surface.baseColor,surface.metallic);let fresnel=unifiedSchlick(max(dot(hit.normal,viewDirection),0.0),f0);let contactVisibility=dryContactVisibility(position,hit.normal,hit.featureId,hit.ownerId);let ignoredBodyOwner=select(DRY_OWNER_NONE,hit.ownerId,hit.motionKind==DRY_GBUFFER_MOTION_RIGID);let gi=dryGlobalIllumination(position,hit.normal,ignoredBodyOwner);let diffuseEnvironmentScale=select(1.0,dry.giLighting.z,globalIllumination);let directScale=select(1.0,dry.giLighting.w,globalIllumination);let diffuseEnvironment=diffuseColor*svoEnvironmentDiffuseIrradiance(dryLighting.environment,hit.normal)*contactVisibility*gi.visibility*diffuseEnvironmentScale/UNIFIED_PI;let specularEnvironment=dryEnvironment(reflected,surface.roughness)*fresnel;let indirectDiffuse=diffuseColor*gi.radiance;
  return max(surface.emissive+diffuseEnvironment+specularEnvironment+direct*directScale+indirectDiffuse,vec3f(0.0));
}

struct DryGlassSurface{color:vec3f,depth:f32,materialId:u32,ownerId:u32,paneId:u32,_padding:u32}
fn shadeThinGlass(glass:DryGlassHit,opaque:DryHit,ro:vec3f,rd:vec3f)->DryGlassSurface {
  let record=glassPanes[glass.recordIndex];let incidentIor=dryThinGlassIncidentIor();let optics=svoThinGlassOptics(record,glass.hit,incidentIor);
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
  var targets=targetsIn;
  targets.radianceDepth=dryCostOverlay(targets.radianceDepth);return DryFragmentOut(targets.radianceDepth,targets.packedSurface,targets.identityMedia,hardwareDepth);
}

@fragment fn fragmentMain(input:VertexOut)->DryFragmentOut {
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryPrimaryNodeVisits=0u;dryPrimaryLeafVisits=0u;dryPrimaryEmptyBrickSkips=0u;dryPrimaryVoxelWorkItems=0u;dryPrimaryExactTests=0u;dryPrimaryMaximumDepth=0u;dryShadowNodeVisits=0u;dryShadowLeafVisits=0u;dryShadowWorkItems=0u;dryMipSteps=0u;dryTraversalFailure=0u;
  // Curved thick glass is compiled separately from this Metal-sensitive pass.
  // Its authored pane therefore remains visible through the exact thin fallback.
  dryThickGlassEnabled=0u;
  let opaque=traceOpaqueScene(ro,rd);${prepassResolveCallWGSL}let glass=traceGlass(ro,rd,0.0,opaque.t,true);var color=shadeDryOpaque(opaque,ro,rd);var depth=opaque.t;
  let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;var glassSurface=DryGlassSurface(vec3f(0.0),DRY_MISS,0u,DRY_OWNER_NONE,0u,0u);
  if(glassVisible){glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);let radiance=max(color*vignette,vec3f(0.0));let generation=dryPublicationGeneration();
  if(glassVisible){
    let media=dryMediumPair(rd,glass.hit.geometricNormal,DRY_MEDIUM_GLASS);
    let targets=svoGBufferSurface(radiance,depth,glass.hit.geometricNormal,glass.hit.geometricNormal,vec4u(glassSurface.materialId,glassSurface.ownerId,media.x,media.y),vec3f(0.0),DRY_GBUFFER_MOTION_STATIC,DRY_GBUFFER_FIELD_ANALYTIC,generation,SVO_GBUFFER_MOTION_VALID,SVO_FEATURE_SMOOTH);
    return dryFragmentOut(targets,dryHardwareDepth(depth,rd,forward));
  }
  if(opaque.t<DRY_MISS){
    let media=dryMediumPair(rd,opaque.normal,DRY_MEDIUM_OPAQUE);let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}
    let targets=svoGBufferSurface(radiance,opaque.t,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);
    return dryFragmentOut(targets,dryHardwareDepth(opaque.t,rd,forward));
  }
  return dryFragmentOut(svoGBufferMiss(radiance,0u,generation,DRY_GBUFFER_NO_INTERSECTION,0u),0.0);
}
${splitEntryWGSL}${prepassEntryWGSL}${prepassFromPrimaryEntryWGSL}${pixelProbe ? createSvoPixelTraceProbeWGSL(svoDryScenePixelProbeOptions()) : ""}`;
  if (stripDiagnostics) {
    const counterNames = [
      "dryPrimaryNodeVisits", "dryPrimaryLeafVisits", "dryPrimaryEmptyBrickSkips",
      "dryPrimaryVoxelWorkItems", "dryPrimaryExactTests", "dryPrimaryMaximumDepth",
      "dryShadowNodeVisits", "dryShadowLeafVisits", "dryShadowWorkItems",
      "dryMipSteps", "dryTraversalFailure",
    ];
    // The overlay is the only observer of these invocation-private values.
    // Make it a passthrough first, then erase every counter write and declaration
    // so Tint cannot conservatively keep their live ranges.
    shader = shader.replace(
      /fn dryNormalizedCost[\s\S]*?(?=fn dryBoundThickGlassOwner)/,
      "fn dryCostOverlay(radianceDepth:vec4f)->vec4f{return radianceDepth;}\n",
    );
    for (const name of counterNames) {
      shader = shader
        .replace(new RegExp(`var<private> ${name}:u32;`, "g"), "")
        .replace(new RegExp(`${name}=(?:max\\([^;]+\\)|select\\([^;]+\\)|[^;]+);`, "g"), "")
        .replace(new RegExp(`${name}\\+=([^;]+);`, "g"), "");
    }
  }
  if (experiments.dropGiPageCache) {
    shader = shader.replace("var<private> dryGiPageCache:DryNodeMipPageCache;", "");
    const loadStart = shader.indexOf("fn svoTetraRadianceConeLoad(query:SvoTetraRadianceConeQuery)->SvoTetraRadianceConeSourceSample{");
    const loadEnd = shader.indexOf("struct DryGlobalIllumination", loadStart);
    if (loadStart < 0 || loadEnd < 0) throw new Error("GI page-cache source drifted");
    const prefix = shader.slice(0, loadStart);
    const load = shader.slice(loadStart, loadEnd)
      .replace("{\n  if(!dryNodeMipReady())", "{\n  var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);\n  if(!dryNodeMipReady())")
      .replaceAll("dryGiPageCache", "pageCache");
    const suffix = shader.slice(loadEnd)
      .replaceAll("dryGiPageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);", "");
    shader = prefix + load + suffix;
  }
  if (experiments.halfPrecisionLighting) {
    const original = "var accumulated0=vec4f(0.0);var accumulated1=vec4f(0.0);var accumulated2=vec4f(0.0);var weightSum=0.0;"
      + "\n  var accumulatedRadiance=vec4f(0.0);var radianceWeightSum=0.0;var accumulatedGi=vec4f(0.0);var giWeightSum=0.0;";
    const half = "var accumulated0=vec4h(0.0h);var accumulated1=vec4h(0.0h);var accumulated2=vec4h(0.0h);var weightSum=0.0h;"
      + "\n  var accumulatedRadiance=vec4h(0.0h);var radianceWeightSum=0.0h;var accumulatedGi=vec4h(0.0h);var giWeightSum=0.0h;";
    if (!shader.includes(original)) throw new Error("Half-precision accumulator source drifted");
    shader = `enable f16;\n${shader.replace(original, half)
      .replace("accumulated0+=dryPrepassUnpack0(packed)*weight;", "accumulated0+=vec4h(dryPrepassUnpack0(packed))*f16(weight);")
      .replace("accumulated1+=dryPrepassUnpack1(packed)*weight;", "accumulated1+=vec4h(dryPrepassUnpack1(packed))*f16(weight);")
      .replace("accumulated2+=dryPrepassUnpack2(packed)*weight;", "accumulated2+=vec4h(dryPrepassUnpack2(packed))*f16(weight);")
      .replace("accumulatedGi+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;giWeightSum+=weight;",
        "accumulatedGi+=vec4h(textureLoad(dryPrepassRadianceTexture,texel,0))*f16(weight);giWeightSum+=f16(weight);")
      .replace("accumulatedRadiance+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;radianceWeightSum+=weight;",
        "accumulatedRadiance+=vec4h(textureLoad(dryPrepassRadianceTexture,texel,0))*f16(weight);radianceWeightSum+=f16(weight);")
      .replace("weightSum+=weight;", "weightSum+=f16(weight);")
      .replace("if(weightSum<", "if(f32(weightSum)<")
      .replace("dryPrepassData0=accumulated0/weightSum;dryPrepassData1=accumulated1/weightSum;dryPrepassData2=accumulated2/weightSum;",
        "dryPrepassData0=accumulated0/weightSum;dryPrepassData1=accumulated1/weightSum;dryPrepassData2=accumulated2/weightSum;")
      .replace("if(giWeightSum>=", "if(f32(giWeightSum)>=")
      .replace("dryPrepassGi=accumulatedGi/giWeightSum;", "dryPrepassGi=accumulatedGi/giWeightSum;")
      .replace("&&radianceWeightSum>1e-6", "&&f32(radianceWeightSum)>1e-6")
      .replace("dryPrepassRadiance=accumulatedRadiance/radianceWeightSum;",
        "dryPrepassRadiance=accumulatedRadiance/radianceWeightSum;")
      .replaceAll("var<private> dryPrepassData0:vec4f;", "var<private> dryPrepassData0:vec4h;")
      .replaceAll("var<private> dryPrepassData1:vec4f;", "var<private> dryPrepassData1:vec4h;")
      .replaceAll("var<private> dryPrepassData2:vec4f;", "var<private> dryPrepassData2:vec4h;")
      .replaceAll("var<private> dryPrepassRadiance:vec4f;", "var<private> dryPrepassRadiance:vec4h;")
      .replaceAll("var<private> dryPrepassGi:vec4f;", "var<private> dryPrepassGi:vec4h;")
      .replace("return dryPrepassData0[index];", "return f32(dryPrepassData0[index]);")
      .replace("return dryPrepassData1[index-4u];", "return f32(dryPrepassData1[index-4u]);")
      .replace("return dryPrepassData2[min(index-8u,3u)];", "return f32(dryPrepassData2[min(index-8u,3u)]);")
      .replaceAll("dryPrepassData0=vec4f(1.0);dryPrepassData1=vec4f(1.0);dryPrepassData2=vec4f(1.0);dryPrepassRadiance=vec4f(0.0);dryPrepassGi=vec4f(0.0,0.0,0.0,1.0);",
        "dryPrepassData0=vec4h(1.0h);dryPrepassData1=vec4h(1.0h);dryPrepassData2=vec4h(1.0h);dryPrepassRadiance=vec4h(0.0h);dryPrepassGi=vec4h(0.0h,0.0h,0.0h,1.0h);")
      .replace("dryPrepassData0.x*(prepassUnblocked/f32(prepassSamples))",
        "f32(dryPrepassData0.x)*(prepassUnblocked/f32(prepassSamples))")
      .replace("return max(dryPrepassRadiance.rgb,vec3f(0.0));",
        "return max(vec3f(dryPrepassRadiance.rgb),vec3f(0.0));")
      .replace("return DryGlobalIllumination(max(dryPrepassGi.rgb,vec3f(0.0)),clamp(dryPrepassGi.a,0.0,1.0));",
        "return DryGlobalIllumination(max(vec3f(dryPrepassGi.rgb),vec3f(0.0)),clamp(f32(dryPrepassGi.a),0.0,1.0));")
      .replaceAll("dryPrepassData0=dryPrepassUnpack0(packed);dryPrepassData1=dryPrepassUnpack1(packed);dryPrepassData2=dryPrepassUnpack2(packed);",
        "dryPrepassData0=vec4h(dryPrepassUnpack0(packed));dryPrepassData1=vec4h(dryPrepassUnpack1(packed));dryPrepassData2=vec4h(dryPrepassUnpack2(packed));")
      .replace("dryPrepassRadiance=textureSampleLevel(dryPrepassRadianceTexture,nodeMipSampler,pixel/max(uniforms.viewport.xy,vec2f(1.0)),0.0);",
        "dryPrepassRadiance=vec4h(textureSampleLevel(dryPrepassRadianceTexture,nodeMipSampler,pixel/max(uniforms.viewport.xy,vec2f(1.0)),0.0));")
      .replace("dryPrepassRadiance=textureLoad(dryPrepassRadianceTexture,bestRadianceTexel,0);",
        "dryPrepassRadiance=vec4h(textureLoad(dryPrepassRadianceTexture,bestRadianceTexel,0));")}`;
  }
  return shader;
}

const drySceneShader = createSvoDrySceneFragmentWGSL(1);

const drySceneVertexShader = /* wgsl */ `
struct VertexOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index) index:u32)->VertexOut {
  var points=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var output:VertexOut;output.position=vec4f(points[index],0,1);output.uv=points[index]*.5+.5;return output;
}
`;

/**
 * Coverage-scaled thin-glass discovery for split rendering. Quads are only
 * conservative raster candidates: the fragment stage repeats the canonical
 * analytic finite-pane intersection before writing exact depth and identity.
 */
export const svoDryRasterGlassShader = /* wgsl */ `
${svoThinGlassWGSL}
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f, terrainMeta:vec4f, terrainFeatures:array<vec4f,16> }
struct GlassRasterParams { paneCount:u32, compositePaneIdBase:u32, compositePaneCount:u32, _padding:u32 }
struct GlassRasterVertexOut {
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) recordIndex:u32,
}
struct GlassRasterFragmentOut {
  @location(0) glassKey:u32,
  @builtin(frag_depth) hardwareDepth:f32,
}
@group(0) @binding(0) var<uniform> uniforms:Uniforms;
@group(0) @binding(10) var<storage,read> glassPanes:array<SvoThinGlassRecord>;
@group(1) @binding(0) var dryRasterOpaqueGeometry:texture_2d<f32>;
@group(1) @binding(1) var<uniform> glassRaster:GlassRasterParams;
const DRY_RASTER_GLASS_NEAR_M:f32=${SVO_DRY_SCENE_REVERSED_Z_NEAR_M};
fn dryRasterGlassCorner(index:u32)->vec2f{
  var corners=array<vec2f,6>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0));
  return corners[index];
}
fn dryRasterGlassRay(pixel:vec2f)->mat2x3f{
  let uv=vec2f(pixel.x/max(uniforms.viewport.x,1.0),1.0-pixel.y/max(uniforms.viewport.y,1.0));let ndc=uv*2.0-1.0;
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));
  let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*.72+up*ndc.y*.72);return mat2x3f(ro,rd);
}
@vertex fn glassRasterVertex(@builtin(vertex_index) vertexIndex:u32,@builtin(instance_index) recordIndex:u32)->GlassRasterVertexOut{
  let record=glassPanes[recordIndex];let corner=dryRasterGlassCorner(vertexIndex);let padding=max(2.0*record.extentIorEpsilon.w,1e-5);let local=vec3f(corner*(record.extentIorEpsilon.xy+vec2f(padding)),0.0);let world=record.centerThickness.xyz+svoThinGlassRotate(record.orientation,local);
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));let relative=world-ro;let viewDepth=dot(relative,forward);let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  return GlassRasterVertexOut(vec4f(dot(relative,right)/(aspect*.72),dot(relative,up)/.72,.5*viewDepth,viewDepth),recordIndex);
}
@fragment fn glassRasterFragment(input:GlassRasterVertexOut)->GlassRasterFragmentOut{
  if(input.recordIndex>=glassRaster.paneCount||input.recordIndex>=arrayLength(&glassPanes)){discard;}
  let record=glassPanes[input.recordIndex];let paneId=svoThinGlassPaneId(record);let compositeOwned=glassRaster.compositePaneCount>0u&&paneId>=glassRaster.compositePaneIdBase&&paneId-glassRaster.compositePaneIdBase<glassRaster.compositePaneCount;if(compositeOwned){discard;}
  let coordinate=vec2i(input.position.xy);let opaqueDepth=textureLoad(dryRasterOpaqueGeometry,coordinate,0).w;let ray=dryRasterGlassRay(input.position.xy);let hit=svoThinGlassIntersect(record,ray[0],ray[1],0.0,opaqueDepth,1e-6,record.extentIorEpsilon.w);if(hit.valid==0u||!(hit.t_m<opaqueDepth)){discard;}
  let forward=normalize(uniforms.cameraTarget.xyz-uniforms.cameraPosition.xyz);let viewDepth=hit.t_m*max(dot(ray[1],forward),1e-6);let hardwareDepth=clamp(DRY_RASTER_GLASS_NEAR_M/viewDepth,0.0,1.0);return GlassRasterFragmentOut(input.recordIndex+1u,hardwareDepth);
}
`;

export interface SvoDryRasterGlassRecordRange {
  readonly firstRecord: number;
  readonly recordCount: number;
}

/**
 * Omit a contiguous compositor-owned prefix from the raster discovery draw.
 * Interleaved ownership deliberately falls back to drawing every record; the
 * fragment shader's pane-ID check remains the authoritative correctness gate.
 */
export function svoDryRasterGlassRecordRange(
  records: Uint32Array | undefined,
  compositePaneIdBase: number,
  compositePaneCount: number,
): SvoDryRasterGlassRecordRange {
  if (!records || records.length % SVO_THIN_GLASS_RECORD_WORDS !== 0) {
    return { firstRecord: 0, recordCount: 0 };
  }
  const paneCount = records.length / SVO_THIN_GLASS_RECORD_WORDS;
  if (compositePaneCount <= 0) return { firstRecord: 0, recordCount: paneCount };
  const isCompositeOwned = (recordIndex: number): boolean => {
    const paneId = records[recordIndex * SVO_THIN_GLASS_RECORD_WORDS + 16];
    return paneId >= compositePaneIdBase && paneId - compositePaneIdBase < compositePaneCount;
  };
  let firstRecord = 0;
  while (firstRecord < paneCount && isCompositeOwned(firstRecord)) firstRecord += 1;
  for (let recordIndex = firstRecord; recordIndex < paneCount; recordIndex += 1) {
    if (isCompositeOwned(recordIndex)) return { firstRecord: 0, recordCount: paneCount };
  }
  return { firstRecord, recordCount: paneCount - firstRecord };
}

async function checkedModule(device: GPUDevice, label: string, code: string): Promise<GPUShaderModule> {
  const shaderModule = device.createShaderModule({ label, code });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(`${label}:\n${errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n")}`);
  return shaderModule;
}

interface SvoDrySplitPipelineBundle {
  readonly visibility: GPURenderPipeline;
  readonly rasterRigidVisibility?: GPURenderPipeline;
  readonly lighting: GPURenderPipeline;
  readonly prepassReset?: GPUComputePipeline;
  readonly prepassCoherent?: GPUComputePipeline;
  readonly prepassBoundary?: GPUComputePipeline;
  readonly worldGiFrame?: GPUComputePipeline;
  readonly worldGiCache?: GPUComputePipeline;
}

interface SvoDryConePipelineBundle {
  readonly geometry: GPURenderPipeline;
  readonly visibility: GPURenderPipeline;
  readonly shade: GPURenderPipeline;
  readonly reduced: GPURenderPipeline;
}

export const SVO_DRY_RASTER_RIGID_BODY_THRESHOLD = 4;
export function svoDryRigidPrimaryStrategy(bodyCount: number, rasterCapability: boolean): "analytic" | "raster" {
  return rasterCapability && Number.isInteger(bodyCount) && bodyCount >= SVO_DRY_RASTER_RIGID_BODY_THRESHOLD
    ? "raster"
    : "analytic";
}

export class SparseVoxelDrySceneRenderer {
  private pipeline?: GPURenderPipeline;
  private splitVisibilityPipeline?: GPURenderPipeline;
  private splitRasterRigidVisibilityPipeline?: GPURenderPipeline;
  private splitLightingPipeline?: GPURenderPipeline;
  private rasterGlassPipeline?: GPURenderPipeline;
  private rasterRigidPipeline?: GPURenderPipeline;
  private rasterRigidBridgePipeline?: GPURenderPipeline;
  private conePrepassResetPipeline?: GPUComputePipeline;
  private conePrepassCoherentPipeline?: GPUComputePipeline;
  private conePrepassBoundaryPipeline?: GPUComputePipeline;
  private worldGiFramePipeline?: GPUComputePipeline;
  private worldGiCachePipeline?: GPUComputePipeline;
  private worldGiCacheLayout?: GPUBindGroupLayout;
  private worldGiCacheBindGroup?: GPUBindGroup;
  private worldGiCacheBuffer?: GPUBuffer;
  private worldGiFrameBuffer?: GPUBuffer;
  private worldGiCacheDirty = true;
  private coneFanoutWorkerPipeline?: GPUComputePipeline;
  private coneFanoutReducerPipeline?: GPUComputePipeline;
  private coneFanoutSceneLayout?: GPUBindGroupLayout;
  private coneFanoutWorkerLayout?: GPUBindGroupLayout;
  private coneFanoutReducerLayout?: GPUBindGroupLayout;
  private coneFanoutSceneBindGroup?: GPUBindGroup;
  private coneFanoutWorkerBindGroup?: GPUBindGroup;
  private coneFanoutReducerBindGroup?: GPUBindGroup;
  private coneFanoutTemporary?: GPUTexture;
  private coneFanoutTemporaryView?: GPUTextureView;
  private coneFanoutReceiver?: GPUTexture;
  private coneFanoutReceiverView?: GPUTextureView;
  private coneFanoutFrameBuffer?: GPUBuffer;
  private coneFanoutLightCount: number = SVO_CONE_FANOUT_CONTRACT.maximumLights;
  private splitPipelineScale?: SvoConeLightingScale;
  private readonly splitPipelineBundles = new Map<SvoConeLightingScale, SvoDrySplitPipelineBundle>();
  private readonly splitPipelineCompiles = new Map<SvoConeLightingScale, Promise<SvoDrySplitPipelineBundle>>();
  private splitVisibilityLayout?: GPUBindGroupLayout;
  private splitLightingLayout?: GPUBindGroupLayout;
  private rasterGlassLayout?: GPUBindGroupLayout;
  private rasterRigidInputLayout?: GPUBindGroupLayout;
  private rasterRigidLayout?: GPUBindGroupLayout;
  private splitVisibilityBindGroup?: GPUBindGroup;
  private splitLightingBindGroup?: GPUBindGroup;
  private rasterGlassBindGroup?: GPUBindGroup;
  private rasterRigidInputBindGroup?: GPUBindGroup;
  private rasterRigidBindGroup?: GPUBindGroup;
  private splitGeometry?: GPUTexture;
  private splitGeometryView?: GPUTextureView;
  private splitOpaqueIdentity?: GPUTexture;
  private splitOpaqueIdentityView?: GPUTextureView;
  private splitGlassKey?: GPUTexture;
  private splitGlassKeyView?: GPUTextureView;
  private splitGlassDepth?: GPUTexture;
  private splitGlassDepthView?: GPUTextureView;
  private rasterRigidPrimaryGeometry?: GPUTexture;
  private rasterRigidPrimaryGeometryView?: GPUTextureView;
  private splitWidth = 0;
  private splitHeight = 0;
  private splitDiagnosticsActive = false;
  private rasterGlassFirstRecord = 0;
  private rasterGlassRecordCount = 0;
  private rasterRigidActive: boolean;
  private layout?: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private vertexModule?: GPUShaderModule;
  /** Reduced-rate cone-lighting prepass state; absent at scale 1. */
  private coneScale: SvoConeLightingScale = 1;
  private conePipelineScale?: SvoConeLightingScale;
  private readonly conePipelineBundles = new Map<SvoConeLightingScale, SvoDryConePipelineBundle>();
  private readonly conePipelineCompiles = new Map<SvoConeLightingScale, Promise<SvoDryConePipelineBundle>>();
  private coneScalePrewarmStarted = false;
  private conePrepassGeometryPipeline?: GPURenderPipeline;
  private conePrepassVisibilityPipeline?: GPURenderPipeline;
  private conePrepassShadePipeline?: GPURenderPipeline;
  private coneReducedPipeline?: GPURenderPipeline;
  private conePrepassLayout?: GPUBindGroupLayout;
  private conePrepassComputeLayout?: GPUBindGroupLayout;
  private conePrepassComputeBindGroup?: GPUBindGroup;
  private conePrepassBoundaryQueue?: GPUBuffer;
  private conePrepassVisibilityLayout?: GPUBindGroupLayout;
  private conePrepassShadeLayout?: GPUBindGroupLayout;
  private conePrepassBindGroup?: GPUBindGroup;
  private conePrepassVisibilityBindGroup?: GPUBindGroup;
  private conePrepassShadeBindGroup?: GPUBindGroup;
  private conePrepassVisibility?: GPUTexture;
  private conePrepassVisibilityView?: GPUTextureView;
  private conePrepassGeometry?: GPUTexture;
  private conePrepassGeometryView?: GPUTextureView;
  private conePrepassIdentity?: GPUTexture;
  private conePrepassIdentityView?: GPUTextureView;
  private conePrepassRadiance?: GPUTexture;
  private conePrepassRadianceView?: GPUTextureView;
  private conePrepassWidth = 0;
  private conePrepassHeight = 0;
  private targetWidth = 0;
  private targetHeight = 0;
  private primitiveBuffer?: GPUBuffer;
  private glassBuffer?: GPUBuffer;
  private glassCacheKey?: string;
  private readonly paramsBuffer: GPUBuffer;
  private readonly lightingBuffer: GPUBuffer;
  private readonly rigidMotionUniformBuffer: GPUBuffer;
  private readonly thickGlassUniformBuffer: GPUBuffer;
  private readonly rasterGlassParamsBuffer: GPUBuffer;
  private readonly nodeMipFallbackAtlas: GPUTexture;
  private readonly nodeMipFallbackAtlasView: GPUTextureView;
  private readonly nodeMipFallbackDirectory: GPUTexture;
  private readonly nodeMipFallbackDirectoryView: GPUTextureView;
  private readonly nodeMipFallbackDirectPageTable: GPUTexture;
  private readonly nodeMipFallbackDirectPageTableView: GPUTextureView;
  private readonly nodeMipFallbackSampler: GPUSampler;
  private readonly tetrahedralRadianceFallback: readonly [GPUTexture, GPUTexture, GPUTexture, GPUTexture];
  private readonly tetrahedralRadianceFallbackViews: readonly [GPUTextureView, GPUTextureView, GPUTextureView, GPUTextureView];
  private readonly tetrahedralRadianceBlackFallback: GPUTexture;
  private readonly tetrahedralRadianceBlackFallbackView: GPUTextureView;
  private tetrahedralRadianceBlackPages?: GPUTexture;
  private tetrahedralRadianceBlackPagesView?: GPUTextureView;
  private readonly fluidCoverageFallback: GPUTexture;
  private readonly fluidCoverageFallbackView: GPUTextureView;
  private fluidCoverage?: WebGpuSvoFluidCoverage;
  private rigidMotionSource?: GPUBuffer;
  private readonly gBufferTargets: SparseVoxelGBufferTargetArena;
  private readonly pickingReadback: SparseVoxelGpuPickingReadbackRing;
  private lastPickingTarget?: GPUTexture;
  private readonly targetViews = new WeakMap<GPUTexture, GPUTextureView>();
  private reusableKey?: string;
  private reusableStableFrames = 0;
  private reusableTarget?: GPUTexture;
  private reusableResult?: DrySceneReplacementResult;
  /** Last exact split-visibility publication authorized by a caller-owned static-frame key. */
  private primaryVisibilityCacheKey?: string;
  /** Resource/source epoch. Later compatible frames do not invalidate a copy already ordered on the queue. */
  private pickingFrameToken = 1;
  private paramsWords?: Uint32Array<ArrayBuffer>;
  private source?: SparseVoxelSceneRenderSource;
  private scene?: SparseVoxelDrySceneData;
  private lightingOptions: SvoLightingOptions = DEFAULT_SVO_LIGHTING_OPTIONS;
  private renderTuning: SvoRenderTuning = DEFAULT_SVO_RENDER_TUNING;
  /** Pixel-trace probe: compiled on first request, never during normal startup. */
  private probePipeline?: GPURenderPipeline;
  private probeLayout?: GPUBindGroupLayout;
  private probeBindGroup?: GPUBindGroup;
  private probeBuffers?: SparseVoxelPixelTraceBuffers;
  private probeTarget?: GPUTexture;
  private probeTargetView?: GPUTextureView;
  private probeCompilation?: Promise<void>;
  private probeCompilationFailed = false;
  private probeRequest?: { pixelX: number; pixelY: number; token: number };
  private probeEncodedToken = 0;
  private probeReadPending = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bodyBuffer: GPUBuffer,
    private readonly targetFormat: GPUTextureFormat = "rgba16float",
    private readonly traversalMode: SvoDryTraversalMode = "hybrid",
    private readonly brickOccupancyMode: SvoBrickOccupancyMode = "off",
    private readonly shadingPath: SvoDryShadingPath = "inline",
    private readonly screenSpaceTerminationPixels = 0,
    private readonly rayCoherenceMode: SvoDryRayCoherenceMode = "off",
    private readonly rasterGlassDiscovery = false,
    private readonly rasterRigidDiscovery = false,
    private readonly coneFanout = false,
    private readonly experiments: SvoDryOptimizationExperiments = { stripDiagnostics: true },
  ) {
    if (targetFormat !== SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat) {
      throw new Error(`Sparse voxel dry scene location 0 must use ${SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat}`);
    }
    if (shadingPath !== "inline" && shadingPath !== "split" && shadingPath !== "auto-relight") throw new RangeError(`Unsupported dry-scene shading path: ${shadingPath}`);
    if (rayCoherenceMode !== "off" && rayCoherenceMode !== "static-primary") throw new RangeError(`Unsupported dry-scene ray coherence mode: ${rayCoherenceMode}`);
    if (rayCoherenceMode === "static-primary" && shadingPath !== "split") throw new RangeError("Static-primary ray coherence requires split shading");
    if (screenSpaceTerminationPixels > 0 && (traversalMode !== "canonical" || shadingPath !== "inline")) throw new RangeError("Diagnostic screen-space termination currently requires canonical inline traversal");
    this.rasterRigidActive = rasterRigidDiscovery;
    this.paramsBuffer = device.createBuffer({ label: "Sparse voxel dry scene parameters", size: SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lightingBuffer = device.createBuffer({ label: "Sparse voxel dry scene lighting arena", size: SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rigidMotionUniformBuffer = device.createBuffer({ label: "Sparse voxel rigid motion uniform mirror", size: SVO_DRY_RIGID_MOTION_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.thickGlassUniformBuffer = device.createBuffer({ label: "Sparse voxel thick-glass uniform binder", size: SVO_DRY_THICK_GLASS_ARENA_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rasterGlassParamsBuffer = device.createBuffer({ label: "Sparse voxel raster-glass parameters", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (coneFanout) {
      this.coneFanoutFrameBuffer = device.createBuffer({
        label: "Sparse voxel cone fan-out frame",
        size: SVO_CONE_FANOUT_CONTRACT.frameBytes,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this.nodeMipFallbackAtlas = device.createTexture({ label: "Sparse voxel node-mip fallback atlas", size: [1, 1, 1], dimension: "3d", format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.nodeMipFallbackAtlasView = this.nodeMipFallbackAtlas.createView({ dimension: "3d" });
    this.nodeMipFallbackDirectory = device.createTexture({ label: "Sparse voxel node-mip fallback directory", size: [2, 1], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.nodeMipFallbackDirectoryView = this.nodeMipFallbackDirectory.createView();
    this.nodeMipFallbackDirectPageTable = device.createTexture({ label: "Sparse voxel node-mip fallback direct page table", size: [1, 1, 1], dimension: "3d", format: "r32uint", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.nodeMipFallbackDirectPageTableView = this.nodeMipFallbackDirectPageTable.createView({ dimension: "3d" });
    this.nodeMipFallbackSampler = device.createSampler({ label: "Sparse voxel node-mip fallback sampler", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge", magFilter: "linear", minFilter: "linear" });
    this.tetrahedralRadianceFallback = [0, 1, 2, 3].map((lobe) => device.createTexture({
      label: `Sparse voxel tetrahedral-radiance fallback lobe ${lobe}`,
      size: [1, 1, 1], dimension: "3d", format: "rgb9e5ufloat", usage: GPUTextureUsage.TEXTURE_BINDING,
    })) as unknown as readonly [GPUTexture, GPUTexture, GPUTexture, GPUTexture];
    this.tetrahedralRadianceFallbackViews = this.tetrahedralRadianceFallback.map((texture) => texture.createView({ dimension: "3d" })) as unknown as readonly [GPUTextureView, GPUTextureView, GPUTextureView, GPUTextureView];
    this.tetrahedralRadianceBlackFallback = device.createTexture({
      label: "Sparse voxel tetrahedral-radiance black-page fallback",
      size: [1, 1],
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this.tetrahedralRadianceBlackFallbackView = this.tetrahedralRadianceBlackFallback.createView();
    // Zero-initialized, and the packed frame reports invalid alongside it, so a
    // scene with no solver never samples this and never shows a water shadow.
    this.fluidCoverageFallback = device.createTexture({ label: "Sparse voxel fluid coverage fallback", size: [1, 1, 1], dimension: "3d", format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.fluidCoverageFallbackView = this.fluidCoverageFallback.createView({ dimension: "3d" });
    this.gBufferTargets = new SparseVoxelGBufferTargetArena(device);
    this.pickingReadback = new SparseVoxelGpuPickingReadbackRing(device);
  }

  async initialize(progress?: (label: string, completed: number, total: number) => void): Promise<void> {
    progress?.("Compiling sparse dry-scene pipeline", 0, 1);
    const [vertexModule, fragmentModule] = await Promise.all([
      checkedModule(this.device, "Sparse voxel dry scene vertex", drySceneVertexShader),
      checkedModule(this.device, `Sparse voxel dry scene fragment (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
        this.traversalMode === "hybrid" && this.brickOccupancyMode === "off" && this.screenSpaceTerminationPixels === 0
          ? drySceneShader : createSvoDrySceneFragmentWGSL(1, this.traversalMode, this.brickOccupancyMode, this.shadingPath, this.screenSpaceTerminationPixels)),
    ]);
    this.layout = this.device.createBindGroupLayout({ label: "Sparse voxel dry scene bindings", entries: sparseVoxelDrySceneBindGroupLayoutEntries() });
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: `Sparse voxel dry scene (${this.traversalMode}, brick-${this.brickOccupancyMode})`, layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module: vertexModule, entryPoint: "vertexMain" }, fragment: { module: fragmentModule, entryPoint: "fragmentMain", targets: [
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
    this.vertexModule = vertexModule;
    if (this.shadingPath === "split") {
      await Promise.all([
        this.ensureSplitPipelines(1),
        this.rasterGlassDiscovery ? this.ensureRasterGlassPipeline() : Promise.resolve(),
        this.rasterRigidDiscovery ? this.ensureRasterRigidPipeline() : Promise.resolve(),
        this.coneFanout ? this.ensureConeFanoutPipelines() : Promise.resolve(),
      ]);
    }
    progress?.("Sparse presentation pipeline compiled", 1, 1);
    this.rebuild();
  }

  private async ensureConeFanoutPipelines(): Promise<void> {
    if (!this.coneFanout || (this.coneFanoutWorkerPipeline && this.coneFanoutReducerPipeline)) return;
    this.coneFanoutSceneLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel cone fan-out scene",
      entries: svoConeFanoutSceneBindGroupLayoutEntries(),
    });
    this.coneFanoutWorkerLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel cone fan-out worker",
      entries: svoConeFanoutWorkerBindGroupLayoutEntries(),
    });
    this.coneFanoutReducerLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel cone fan-out reducer",
      entries: svoConeFanoutReducerBindGroupLayoutEntries(),
    });
    const [workerModule, reducerModule] = await Promise.all([
      checkedModule(this.device, "Sparse voxel cone fan-out worker", createSvoConeFanoutWorkerWGSL({
        coneMarcherWGSL: createSvoDryConeMarcherWGSL({
          branchlessMorton: true,
          rangedDirectorySearch: true,
          fluidCoverage: true,
          directPageTable: true,
        }),
        visibilityFlags: {
          ambientOcclusion: SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion,
          exactShadow: SVO_DRY_VISIBILITY_FLAGS.exactShadow,
          globalIllumination: SVO_DRY_VISIBILITY_FLAGS.globalIllumination,
        },
      })),
      checkedModule(this.device, "Sparse voxel cone fan-out reducer", svoConeFanoutReducerWGSL),
    ]);
    [this.coneFanoutWorkerPipeline, this.coneFanoutReducerPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({
        label: "Sparse voxel cone fan-out worker",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.coneFanoutSceneLayout, this.coneFanoutWorkerLayout] }),
        compute: { module: workerModule, entryPoint: "svoConeFanoutWorker" },
      }),
      this.device.createComputePipelineAsync({
        label: "Sparse voxel cone fan-out reducer",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.coneFanoutReducerLayout] }),
        compute: { module: reducerModule, entryPoint: "svoConeFanoutReduce" },
      }),
    ]);
  }

  /**
   * Compile the opt-in visibility/G-buffer then lighting pair. Entry-point
   * reachability keeps primary traversal out of the lighting pass for opaque
   * pixels even though both entries share one source module.
   */
  private activateSplitPipelineBundle(scale: SvoConeLightingScale, bundle: SvoDrySplitPipelineBundle): void {
    this.splitVisibilityPipeline = bundle.visibility;
    this.splitRasterRigidVisibilityPipeline = bundle.rasterRigidVisibility;
    this.splitLightingPipeline = bundle.lighting;
    this.conePrepassResetPipeline = bundle.prepassReset;
    this.conePrepassCoherentPipeline = bundle.prepassCoherent;
    this.conePrepassBoundaryPipeline = bundle.prepassBoundary;
    this.worldGiFramePipeline = bundle.worldGiFrame;
    this.worldGiCachePipeline = bundle.worldGiCache;
    this.splitPipelineScale = scale;
    this.ensureSplitTargets();
    this.ensureConePrepassTargets();
  }

  /**
   * Selects between two already-compiled exact primary-discovery strategies.
   * The measured Metal crossover is above one body and below six; four keeps
   * small scenes on the cheaper analytic loop while amortizing raster passes
   * for body stacks. Body motion never changes this choice or recompiles WGSL.
   */
  setRigidBodyCount(bodyCount: number): void {
    const active = svoDryRigidPrimaryStrategy(bodyCount, this.rasterRigidDiscovery) === "raster";
    if (active === this.rasterRigidActive) return;
    this.rasterRigidActive = active;
    this.clearReusableFrame();
    this.clearPrimaryVisibilityCache();
  }

  private async ensureRasterGlassPipeline(): Promise<void> {
    if (!this.rasterGlassDiscovery || this.rasterGlassPipeline || !this.layout) return;
    this.rasterGlassLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel raster-glass discovery inputs",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    const module = await checkedModule(this.device, "Sparse voxel raster thin-glass discovery", svoDryRasterGlassShader);
    this.rasterGlassPipeline = await this.device.createRenderPipelineAsync({
      label: "Sparse voxel raster thin-glass discovery",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.rasterGlassLayout] }),
      vertex: { module, entryPoint: "glassRasterVertex" },
      fragment: { module, entryPoint: "glassRasterFragment", targets: [
        { format: "r32uint" },
      ] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
        depthWriteEnabled: true,
        depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
      },
    });
    this.ensureSplitTargets();
  }

  private async ensureRasterRigidPipeline(): Promise<void> {
    if (!this.rasterRigidDiscovery || (this.rasterRigidPipeline && this.rasterRigidBridgePipeline) || !this.layout) return;
    // The main dry-scene shader deliberately exposes BodyGPU as a uniform so
    // its already-wide fragment interface stays below Metal's storage-buffer
    // limit. The small rigid pipelines bind the same live GPUBuffer through a
    // dedicated storage layout instead; no upload, stale cache, or shader
    // recompilation is required when a body moves.
    this.rasterRigidInputLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel raster-rigid live inputs",
      entries: svoRigidRasterInputBindGroupLayoutEntries(),
    });
    this.rasterRigidLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel raster-rigid certificate bridge input",
      entries: svoRigidRasterCoverageBridgeBindGroupLayoutEntries(),
    });
    this.rasterRigidInputBindGroup ??= this.device.createBindGroup({
      label: "Sparse voxel raster-rigid live input binding",
      layout: this.rasterRigidInputLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bodyBuffer } },
        { binding: 14, resource: { buffer: this.rigidMotionUniformBuffer } },
      ],
    });
    const module = await checkedModule(this.device, "Sparse voxel analytic rigid raster", svoRigidRasterShader);
    const renderPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.rasterRigidInputLayout] });
    const bridgePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.rasterRigidInputLayout, this.rasterRigidLayout] });
    [this.rasterRigidPipeline, this.rasterRigidBridgePipeline] = await Promise.all([this.device.createRenderPipelineAsync({
        label: "Sparse voxel analytic rigid primary discovery",
        layout: renderPipelineLayout,
        vertex: { module, entryPoint: "rigidRasterVertex" },
        fragment: { module, entryPoint: "rigidRasterFragment", targets: [
          { format: SVO_RIGID_RASTER_CONTRACT.packedSurfaceFormat },
          { format: SVO_RIGID_RASTER_CONTRACT.identityMediaFormat },
          { format: SVO_RIGID_RASTER_CONTRACT.primaryGeometryFormat },
        ] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: {
          format: SVO_RIGID_RASTER_CONTRACT.depthFormat,
          depthWriteEnabled: SVO_RIGID_RASTER_CONTRACT.depthWriteEnabled,
          depthCompare: SVO_RIGID_RASTER_CONTRACT.depthCompare,
        },
      }), this.device.createRenderPipelineAsync({
        label: "Sparse voxel raster-rigid certificate bridge",
        layout: bridgePipelineLayout,
        vertex: { module, entryPoint: "rigidRasterVertex" },
        fragment: { module, entryPoint: SVO_RIGID_RASTER_CONTRACT.splitBridgeEntryPoint, targets: [
          { format: SVO_RIGID_RASTER_CONTRACT.geometryFormat },
          { format: SVO_RIGID_RASTER_CONTRACT.identityFormat },
        ] },
        primitive: { topology: "triangle-list", cullMode: "none" },
      })]);
    this.ensureSplitTargets();
  }

  private async ensureSplitPipelines(scale: SvoConeLightingScale): Promise<void> {
    const relight = this.renderTuning.coneRadianceReconstruction === "wide-relight"
      || this.renderTuning.coneRadianceReconstruction === "full-res-relight";
    if (this.shadingPath === "inline"
      || (this.shadingPath === "auto-relight" && (scale === 1 || !relight))
      || !this.layout || !this.vertexModule) return;
    const cached = this.splitPipelineBundles.get(scale);
    if (cached) {
      if (scale === this.coneScale) this.activateSplitPipelineBundle(scale, cached);
      return;
    }
    const pending = this.splitPipelineCompiles.get(scale);
    if (pending) {
      const bundle = await pending;
      if (scale === this.coneScale) this.activateSplitPipelineBundle(scale, bundle);
      return;
    }
    if (scale !== 1 && !this.conePrepassLayout) return;
    if (scale !== 1 && !this.conePrepassComputeLayout) {
      this.conePrepassComputeLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel compact cone-prepass outputs",
        entries: [
          { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_DRY_CONE_PREPASS_CONTRACT.visibilityFormat } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_DRY_CONE_PREPASS_CONTRACT.geometryFormat } },
          { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_DRY_CONE_PREPASS_CONTRACT.identityFormat } },
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          ...(this.coneFanout ? [{ binding: 8, visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: "write-only" as const, format: SVO_CONE_FANOUT_CONTRACT.receiverFormat } }] : []),
        ],
      });
    }
    if (scale !== 1 && !this.worldGiCacheLayout) {
      this.worldGiCacheLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel persistent world GI cache",
        entries: [
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 8, visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: "write-only", format: SVO_DRY_CONE_PREPASS_CONTRACT.radianceFormat } },
          { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
      });
      this.worldGiCacheBuffer = this.device.createBuffer({
        label: "Sparse voxel persistent world GI cache entries",
        size: SVO_DRY_WORLD_GI_CACHE_CONTRACT.allocatedBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.worldGiFrameBuffer = this.device.createBuffer({
        label: "Sparse voxel persistent world GI frame prelude",
        size: SVO_DRY_WORLD_GI_CACHE_CONTRACT.frameBytes,
        usage: GPUBufferUsage.STORAGE,
      });
      this.worldGiCacheDirty = true;
    }
    this.splitVisibilityLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel split visibility outputs",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, storageTexture: { access: "write-only", format: SVO_DRY_SPLIT_GEOMETRY_FORMAT } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, storageTexture: { access: "write-only", format: SVO_DRY_SPLIT_IDENTITY_FORMAT } },
      ],
    });
    this.splitLightingLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel split lighting inputs",
      entries: [
        { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: { sampleType: "uint" } },
        ...(this.rasterGlassDiscovery
          ? [{ binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" as const } }]
          : []),
      ],
    });
    const layout = this.layout;
    const vertexModule = this.vertexModule;
    const shaderExperiments = scale === 1 && this.experiments.halfPrecisionLighting
      ? { ...this.experiments, halfPrecisionLighting: false }
      : this.experiments;
    const compile = (async (): Promise<SvoDrySplitPipelineBundle> => {
      const [module, rasterRigidModule] = await Promise.all([
        checkedModule(this.device, `Sparse voxel dry scene split x${scale} (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
          createSvoDrySceneFragmentWGSL(scale, this.traversalMode, this.brickOccupancyMode, "split", 0, false,
            this.rasterGlassDiscovery, false, this.coneFanout && scale !== 1, shaderExperiments)),
        this.rasterRigidDiscovery
          ? checkedModule(this.device, `Sparse voxel dry scene raster-rigid split x${scale} (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
            createSvoDrySceneFragmentWGSL(scale, this.traversalMode, this.brickOccupancyMode, "split", 0, false,
              this.rasterGlassDiscovery, true, this.coneFanout && scale !== 1, shaderExperiments))
          : Promise.resolve(undefined),
      ]);
      const middleLayouts = scale === 1 ? [] : [this.conePrepassLayout!];
      const visibilityLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitVisibilityLayout] });
      const [visibility, rasterRigidVisibility, lighting, prepassReset, prepassCoherent, prepassBoundary, worldGiFrame, worldGiCache] = await Promise.all([
        this.device.createRenderPipelineAsync({
        label: `Sparse voxel primary visibility (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
        layout: visibilityLayout,
        vertex: { module: vertexModule, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "dryVisibilityMain", targets: [
          { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.packedSurfaceFormat },
          { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.identityMediaFormat },
        ] },
        primitive: { topology: "triangle-list" },
        depthStencil: {
          format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          depthWriteEnabled: true,
          depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
        },
      }), rasterRigidModule ? this.device.createRenderPipelineAsync({
        label: `Sparse voxel raster-rigid primary visibility (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
        layout: visibilityLayout,
        vertex: { module: vertexModule, entryPoint: "vertexMain" },
        fragment: { module: rasterRigidModule, entryPoint: "dryVisibilityMain", targets: [
          { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.packedSurfaceFormat },
          { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.identityMediaFormat },
        ] },
        primitive: { topology: "triangle-list" },
        depthStencil: {
          format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          depthWriteEnabled: true,
          depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
        },
      }) : Promise.resolve(undefined),
        this.device.createRenderPipelineAsync({
        label: `Sparse voxel deferred dry lighting x${scale}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitLightingLayout] }),
        vertex: { module: vertexModule, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "dryLightingMain", targets: [{ format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat }] },
        primitive: { topology: "triangle-list" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel compact cone queue reset",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassComputeLayout!, this.splitLightingLayout!] }),
        compute: { module, entryPoint: "dryPrepassResetMain" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel coherent cone visibility from primary hits",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassComputeLayout!, this.splitLightingLayout!] }),
        compute: { module, entryPoint: "dryPrepassCoherentMain" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel compact boundary cone visibility",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassComputeLayout!, this.splitLightingLayout!] }),
        compute: { module, entryPoint: "dryPrepassBoundaryMain" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel world GI frame prelude",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassShadeLayout!, this.worldGiCacheLayout!] }),
        compute: { module, entryPoint: "dryWorldGiFrameMain" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel persistent world GI cache",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassShadeLayout!, this.worldGiCacheLayout!] }),
        compute: { module, entryPoint: "dryWorldGiCacheMain" },
      }),
      ]);
      const bundle = { visibility, rasterRigidVisibility, lighting, prepassReset, prepassCoherent, prepassBoundary, worldGiFrame, worldGiCache };
      this.splitPipelineBundles.set(scale, bundle);
      return bundle;
    })();
    this.splitPipelineCompiles.set(scale, compile);
    try {
      const bundle = await compile;
      if (scale === this.coneScale) {
        this.activateSplitPipelineBundle(scale, bundle);
        this.clearReusableFrame();
      }
    } finally {
      if (this.splitPipelineCompiles.get(scale) === compile) this.splitPipelineCompiles.delete(scale);
    }
  }

  private ensureSplitTargets(): void {
    const relight = this.renderTuning.coneRadianceReconstruction === "wide-relight"
      || this.renderTuning.coneRadianceReconstruction === "full-res-relight";
    if (this.shadingPath === "inline" || (this.shadingPath === "auto-relight" && !relight)
      || !this.targetWidth || !this.targetHeight
      || !this.splitVisibilityLayout || !this.splitLightingLayout) return;
    if (!this.splitGeometry || !this.splitOpaqueIdentity || (this.rasterGlassDiscovery && (!this.splitGlassKey || !this.splitGlassDepth))
      || (this.rasterRigidDiscovery && !this.rasterRigidPrimaryGeometry)
      || this.splitWidth !== this.targetWidth || this.splitHeight !== this.targetHeight) {
      this.clearPrimaryVisibilityCache();
      this.splitGeometry?.destroy();
      this.splitOpaqueIdentity?.destroy();
      this.splitGlassKey?.destroy();
      this.splitGlassDepth?.destroy();
      this.rasterRigidPrimaryGeometry?.destroy();
      this.splitGeometry = this.device.createTexture({
        label: "Sparse voxel split exact primary geometry",
        size: [this.targetWidth, this.targetHeight],
        format: SVO_DRY_SPLIT_GEOMETRY_FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
          | (this.rasterRigidDiscovery ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
      });
      this.splitGeometryView = this.splitGeometry.createView();
      this.splitOpaqueIdentity = this.device.createTexture({
        label: "Sparse voxel split exact primary identity",
        size: [this.targetWidth, this.targetHeight],
        format: SVO_DRY_SPLIT_IDENTITY_FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
          | (this.rasterRigidDiscovery ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
      });
      this.splitOpaqueIdentityView = this.splitOpaqueIdentity.createView();
      if (this.rasterRigidDiscovery) {
        this.rasterRigidPrimaryGeometry = this.device.createTexture({
          label: "Sparse voxel raster-rigid packed primary geometry",
          size: [this.targetWidth, this.targetHeight],
          format: SVO_RIGID_RASTER_CONTRACT.primaryGeometryFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.rasterRigidPrimaryGeometryView = this.rasterRigidPrimaryGeometry.createView();
      }
      if (this.rasterGlassDiscovery) {
        this.splitGlassKey = this.device.createTexture({
          label: "Sparse voxel nearest raster-glass record",
          size: [this.targetWidth, this.targetHeight],
          format: "r32uint",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.splitGlassKeyView = this.splitGlassKey.createView();
        this.splitGlassDepth = this.device.createTexture({
          label: "Sparse voxel nearest raster-glass depth",
          size: [this.targetWidth, this.targetHeight],
          format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.splitGlassDepthView = this.splitGlassDepth.createView();
      }
      this.splitWidth = this.targetWidth;
      this.splitHeight = this.targetHeight;
    }
    this.splitVisibilityBindGroup = this.device.createBindGroup({
      label: "Sparse voxel split visibility output binding",
      layout: this.splitVisibilityLayout,
      entries: [
        { binding: 0, resource: this.splitGeometryView! },
        { binding: 4, resource: this.splitOpaqueIdentityView! },
      ],
    });
    this.splitLightingBindGroup = this.device.createBindGroup({
      label: "Sparse voxel split lighting input bindings",
      layout: this.splitLightingLayout,
      entries: [
        { binding: 1, resource: this.splitGeometryView! },
        { binding: 5, resource: this.splitOpaqueIdentityView! },
        ...(this.rasterGlassDiscovery ? [{ binding: 6, resource: this.splitGlassKeyView! }] : []),
      ],
    });
    if (this.rasterGlassDiscovery && this.rasterGlassLayout) {
      this.rasterGlassBindGroup = this.device.createBindGroup({
        label: "Sparse voxel raster-glass discovery binding",
        layout: this.rasterGlassLayout,
        entries: [
          { binding: 0, resource: this.splitGeometryView! },
          { binding: 1, resource: { buffer: this.rasterGlassParamsBuffer } },
        ],
      });
    }
    if (this.rasterRigidDiscovery && this.rasterRigidLayout && this.rasterRigidPrimaryGeometryView) {
      this.rasterRigidBindGroup = this.device.createBindGroup({
        label: "Sparse voxel raster-rigid certificate bridge binding",
        layout: this.rasterRigidLayout,
        entries: [{ binding: SVO_RIGID_RASTER_CONTRACT.primaryGeometryReadBinding, resource: this.rasterRigidPrimaryGeometryView }],
      });
    }
  }

  /** Active per-axis cone-lighting rate; 1 keeps the historical inline path. */
  get coneLightingScale(): SvoConeLightingScale {
    return this.coneScale;
  }

  /**
   * Compiles and caches the reduced-rate prepass and consuming pipelines for
   * the current scale. Fail-soft: until this resolves, encode keeps the
   * bit-exact inline path. No-op at scale 1 or before initialize().
   */
  private activateConePipelineBundle(scale: SvoConeLightingScale, bundle: SvoDryConePipelineBundle): void {
    this.conePrepassGeometryPipeline = bundle.geometry;
    this.conePrepassVisibilityPipeline = bundle.visibility;
    this.conePrepassShadePipeline = bundle.shade;
    this.coneReducedPipeline = bundle.reduced;
    this.conePipelineScale = scale;
    this.ensureConePrepassTargets();
  }

  private async ensureConeLightingScale(scale: Exclude<SvoConeLightingScale, 1>): Promise<void> {
    if (!this.layout || !this.pipeline || !this.vertexModule) return;
    const cached = this.conePipelineBundles.get(scale);
    if (cached) {
      if (scale === this.coneScale) this.activateConePipelineBundle(scale, cached);
      await this.ensureSplitPipelines(scale);
      return;
    }
    const pending = this.conePipelineCompiles.get(scale);
    if (pending) {
      const bundle = await pending;
      if (scale === this.coneScale) this.activateConePipelineBundle(scale, bundle);
      await this.ensureSplitPipelines(scale);
      return;
    }
    const compile = (async (): Promise<SvoDryConePipelineBundle> => {
      const module = await checkedModule(this.device, `Sparse voxel dry scene cone prepass (x${scale}, ${this.traversalMode}, brick-${this.brickOccupancyMode})`,
        createSvoDrySceneFragmentWGSL(scale, this.traversalMode, this.brickOccupancyMode, "inline", 0, false, false, false, false, this.experiments));
      this.conePrepassLayout ??= this.device.createBindGroupLayout({
        label: "Sparse voxel cone-prepass outputs",
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        ],
      });
      this.conePrepassVisibilityLayout ??= this.device.createBindGroupLayout({
        label: "Sparse voxel cone-prepass visibility inputs",
        entries: [
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
        ],
      });
      this.conePrepassShadeLayout ??= this.device.createBindGroupLayout({
        label: "Sparse voxel cone-prepass shading inputs",
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: { sampleType: "uint" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: { sampleType: "uint" } },
        ],
      });
      const [prepassGeometryPipeline, prepassVisibilityPipeline, prepassShadePipeline, reducedPipeline] = await Promise.all([
        this.device.createRenderPipelineAsync({
          label: "Sparse voxel cone-prepass geometry",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout!] }),
          vertex: { module: this.vertexModule!, entryPoint: "vertexMain" },
          fragment: { module, entryPoint: "dryPrepassGeometryMain", targets: [
            { format: SVO_DRY_CONE_PREPASS_CONTRACT.geometryFormat },
            { format: SVO_DRY_CONE_PREPASS_CONTRACT.identityFormat },
          ] },
          primitive: { topology: "triangle-list" },
        }),
        this.device.createRenderPipelineAsync({
          label: "Sparse voxel cone-prepass visibility",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout!, this.conePrepassVisibilityLayout!] }),
          vertex: { module: this.vertexModule!, entryPoint: "vertexMain" },
          fragment: { module, entryPoint: "dryPrepassVisibilityMain", targets: [
            { format: SVO_DRY_CONE_PREPASS_CONTRACT.visibilityFormat },
          ] },
          primitive: { topology: "triangle-list" },
        }),
        this.device.createRenderPipelineAsync({
          label: "Sparse voxel reduced-rate opaque shading",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout!, this.conePrepassShadeLayout] }),
          vertex: { module: this.vertexModule!, entryPoint: "vertexMain" },
          fragment: { module, entryPoint: "dryPrepassShadeMain", targets: [
            { format: SVO_DRY_CONE_PREPASS_CONTRACT.radianceFormat },
          ] },
          primitive: { topology: "triangle-list" },
        }),
        this.device.createRenderPipelineAsync({
          label: `Sparse voxel dry scene (cone prepass x${scale})`,
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout!, this.conePrepassLayout] }),
          vertex: { module: this.vertexModule!, entryPoint: "vertexMain" },
          fragment: { module, entryPoint: "fragmentMain", targets: [
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
        }),
      ]);
      const bundle = {
        geometry: prepassGeometryPipeline,
        visibility: prepassVisibilityPipeline,
        shade: prepassShadePipeline,
        reduced: reducedPipeline,
      };
      this.conePipelineBundles.set(scale, bundle);
      return bundle;
    })();
    this.conePipelineCompiles.set(scale, compile);
    try {
      const bundle = await compile;
      if (scale === this.coneScale) {
        this.activateConePipelineBundle(scale, bundle);
        this.clearReusableFrame();
      }
      await this.ensureSplitPipelines(scale);
    } finally {
      if (this.conePipelineCompiles.get(scale) === compile) this.conePipelineCompiles.delete(scale);
    }
  }

  async ensureConeLightingPrepass(): Promise<void> {
    if (this.coneScale === 1) return;
    const requestedScale = this.coneScale;
    await this.ensureConeLightingScale(requestedScale);
    if (!this.coneScalePrewarmStarted) {
      this.coneScalePrewarmStarted = true;
      // These are the production moving/settled tiers. Keeping both bundles
      // resident makes a camera-state transition a pointer swap plus target
      // resize, never a Metal shader compilation at the moment of motion.
      try {
        await Promise.all(([0.25, 0.5] as const)
          .filter((scale) => scale !== requestedScale)
          .map((scale) => this.ensureConeLightingScale(scale)));
      } catch (error) {
        // A transient device/compiler failure must not permanently suppress a
        // later explicit warmup attempt.
        this.coneScalePrewarmStarted = false;
        throw error;
      }
    }
  }

  private ensureConePrepassTargets(): void {
    if (this.coneScale === 1 || !this.conePrepassLayout || !this.conePrepassVisibilityLayout
      || !this.conePrepassShadeLayout || !this.targetWidth || !this.targetHeight) return;
    const [width, height] = svoConePrepassSize(this.targetWidth, this.targetHeight, this.coneScale);
    if (this.conePrepassVisibility && this.conePrepassGeometry && this.conePrepassIdentity && this.conePrepassRadiance
      && this.conePrepassWidth === width && this.conePrepassHeight === height
      && (!this.conePrepassComputeLayout || (this.conePrepassBoundaryQueue && this.conePrepassComputeBindGroup))
      && (!this.worldGiCacheLayout || this.worldGiCacheBindGroup)
      && (!this.coneFanout || (this.coneFanoutReceiver && this.coneFanoutTemporary
        && this.coneFanoutWorkerBindGroup && this.coneFanoutReducerBindGroup))) return;
    this.releaseConePrepassTargets();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    this.conePrepassVisibility = this.device.createTexture({
      label: "Sparse voxel cone-prepass packed visibility",
      size: [width, height],
      format: SVO_DRY_CONE_PREPASS_CONTRACT.visibilityFormat,
      usage,
    });
    this.conePrepassGeometry = this.device.createTexture({
      label: "Sparse voxel cone-prepass geometry",
      size: [width, height],
      format: SVO_DRY_CONE_PREPASS_CONTRACT.geometryFormat,
      usage,
    });
    this.conePrepassIdentity = this.device.createTexture({
      label: "Sparse voxel cone-prepass exact identity",
      size: [width, height],
      format: SVO_DRY_CONE_PREPASS_CONTRACT.identityFormat,
      usage,
    });
    this.conePrepassRadiance = this.device.createTexture({
      label: "Sparse voxel cone-prepass opaque radiance",
      size: [width, height],
      format: SVO_DRY_CONE_PREPASS_CONTRACT.radianceFormat,
      usage,
    });
    this.conePrepassVisibilityView = this.conePrepassVisibility.createView();
    this.conePrepassGeometryView = this.conePrepassGeometry.createView();
    this.conePrepassIdentityView = this.conePrepassIdentity.createView();
    this.conePrepassRadianceView = this.conePrepassRadiance.createView();
    if (this.coneFanout && this.coneFanoutFrameBuffer && this.coneFanoutWorkerLayout && this.coneFanoutReducerLayout) {
      this.coneFanoutReceiver = this.device.createTexture({
        label: "Sparse voxel cone fan-out full-precision receiver",
        size: [width, height],
        format: SVO_CONE_FANOUT_CONTRACT.receiverFormat,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
      this.coneFanoutReceiverView = this.coneFanoutReceiver.createView();
      this.coneFanoutTemporary = this.device.createTexture({
        label: "Sparse voxel cone fan-out samples",
        size: [width, height, SVO_CONE_FANOUT_CONTRACT.layerCount],
        format: SVO_CONE_FANOUT_CONTRACT.temporaryFormat,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
      this.coneFanoutTemporaryView = this.coneFanoutTemporary.createView({
        dimension: "2d-array",
        arrayLayerCount: SVO_CONE_FANOUT_CONTRACT.layerCount,
      });
      this.device.queue.writeBuffer(this.coneFanoutFrameBuffer, 0, packSvoConeFanoutFrame({
        width,
        height,
        lightCount: this.coneFanoutLightCount,
        secondaryLightSamples: false,
      }));
      this.coneFanoutWorkerBindGroup = this.device.createBindGroup({
        label: "Sparse voxel cone fan-out worker resources",
        layout: this.coneFanoutWorkerLayout,
        entries: [
          { binding: 0, resource: { buffer: this.coneFanoutFrameBuffer } },
          { binding: 1, resource: this.coneFanoutReceiverView },
          { binding: 2, resource: this.coneFanoutTemporaryView },
          { binding: 3, resource: this.conePrepassGeometryView },
        ],
      });
      this.coneFanoutReducerBindGroup = this.device.createBindGroup({
        label: "Sparse voxel cone fan-out reducer resources",
        layout: this.coneFanoutReducerLayout,
        entries: [
          { binding: 0, resource: { buffer: this.coneFanoutFrameBuffer } },
          { binding: 1, resource: this.coneFanoutTemporaryView },
          { binding: 2, resource: this.conePrepassVisibilityView },
        ],
      });
    }
    this.conePrepassBindGroup = this.device.createBindGroup({
      label: "Sparse voxel cone-prepass consumption",
      layout: this.conePrepassLayout,
      entries: [
        { binding: 0, resource: this.conePrepassVisibilityView },
        { binding: 1, resource: this.conePrepassGeometryView },
        { binding: 2, resource: this.conePrepassIdentityView },
        { binding: 3, resource: this.conePrepassRadianceView },
      ],
    });
    this.conePrepassVisibilityBindGroup = this.device.createBindGroup({
      label: "Sparse voxel cone-prepass visibility input",
      layout: this.conePrepassVisibilityLayout,
      entries: [
        { binding: 1, resource: this.conePrepassGeometryView },
        { binding: 2, resource: this.conePrepassIdentityView },
      ],
    });
    this.conePrepassShadeBindGroup = this.device.createBindGroup({
      label: "Sparse voxel cone-prepass shading input",
      layout: this.conePrepassShadeLayout,
      entries: [
        { binding: 0, resource: this.conePrepassVisibilityView },
        { binding: 1, resource: this.conePrepassGeometryView },
        { binding: 2, resource: this.conePrepassIdentityView },
      ],
    });
    if (this.worldGiCacheLayout && this.worldGiCacheBuffer && this.worldGiFrameBuffer) {
      this.worldGiCacheBindGroup = this.device.createBindGroup({
        label: "Sparse voxel persistent world GI cache resources",
        layout: this.worldGiCacheLayout,
        entries: [
          { binding: 7, resource: { buffer: this.worldGiCacheBuffer } },
          { binding: 8, resource: this.conePrepassRadianceView },
          { binding: 9, resource: { buffer: this.worldGiFrameBuffer } },
        ],
      });
    }
    if (this.conePrepassComputeLayout) {
      this.conePrepassBoundaryQueue = this.device.createBuffer({
        label: "Sparse voxel compact cone-boundary queue",
        size: 4 * (1 + width * height),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.conePrepassComputeBindGroup = this.device.createBindGroup({
        label: "Sparse voxel compact cone-prepass output binding",
        layout: this.conePrepassComputeLayout,
        entries: [
          { binding: 4, resource: this.conePrepassVisibilityView },
          { binding: 5, resource: this.conePrepassGeometryView },
          { binding: 6, resource: this.conePrepassIdentityView },
          { binding: 7, resource: { buffer: this.conePrepassBoundaryQueue } },
          ...(this.coneFanout && this.coneFanoutReceiverView
            ? [{ binding: 8, resource: this.coneFanoutReceiverView }]
            : []),
        ],
      });
    }
    this.conePrepassWidth = width;
    this.conePrepassHeight = height;
    this.clearReusableFrame();
  }

  private releaseConePrepassTargets(): void {
    this.conePrepassBoundaryQueue?.destroy();
    this.coneFanoutReceiver?.destroy();
    this.coneFanoutTemporary?.destroy();
    this.conePrepassVisibility?.destroy();
    this.conePrepassGeometry?.destroy();
    this.conePrepassIdentity?.destroy();
    this.conePrepassRadiance?.destroy();
    this.conePrepassVisibility = undefined;
    this.conePrepassGeometry = undefined;
    this.conePrepassIdentity = undefined;
    this.conePrepassRadiance = undefined;
    this.conePrepassBoundaryQueue = undefined;
    this.coneFanoutReceiver = undefined;
    this.coneFanoutReceiverView = undefined;
    this.coneFanoutTemporary = undefined;
    this.coneFanoutTemporaryView = undefined;
    this.coneFanoutWorkerBindGroup = undefined;
    this.coneFanoutReducerBindGroup = undefined;
    this.conePrepassVisibilityView = undefined;
    this.conePrepassGeometryView = undefined;
    this.conePrepassIdentityView = undefined;
    this.conePrepassRadianceView = undefined;
    this.conePrepassBindGroup = undefined;
    this.conePrepassComputeBindGroup = undefined;
    this.conePrepassVisibilityBindGroup = undefined;
    this.conePrepassShadeBindGroup = undefined;
    this.worldGiCacheBindGroup = undefined;
    this.conePrepassWidth = 0;
    this.conePrepassHeight = 0;
  }

  setSource(source: SparseVoxelSceneRenderSource | undefined, scene: SparseVoxelDrySceneData | undefined): void {
    this.pickingFrameToken += 1;
    this.lastPickingTarget = undefined;
    this.clearReusableFrame();
    this.clearPrimaryVisibilityCache();
    this.worldGiCacheDirty = true;
    this.source = source;
    this.scene = scene;
    this.updateTetrahedralRadianceBlackPages(source?.tetrahedralRadiance);
    this.coneFanoutLightCount = Math.min(source?.lights?.count ?? 1, SVO_CONE_FANOUT_CONTRACT.maximumLights);
    if (this.coneFanoutFrameBuffer && this.conePrepassWidth && this.conePrepassHeight) {
      this.device.queue.writeBuffer(this.coneFanoutFrameBuffer, 0, packSvoConeFanoutFrame({
        width: this.conePrepassWidth,
        height: this.conePrepassHeight,
        lightCount: this.coneFanoutLightCount,
        secondaryLightSamples: false,
      }));
    }
    const paneCount = (scene?.glassRecords?.byteLength ?? 0) / SVO_THIN_GLASS_RECORD_STRIDE_BYTES;
    const paneIdBase = scene?.primaryCompositeOwnedGlassPaneIdBase ?? 0xffff_ffff;
    const compositePaneCount = scene?.primaryCompositeOwnedGlassPaneCount ?? 0;
    const records = scene?.glassRecords;
    const rasterRange = svoDryRasterGlassRecordRange(records, paneIdBase, compositePaneCount);
    this.rasterGlassFirstRecord = rasterRange.firstRecord;
    this.rasterGlassRecordCount = rasterRange.recordCount;
    this.device.queue.writeBuffer(this.rasterGlassParamsBuffer, 0, new Uint32Array([
      paneCount,
      paneIdBase,
      compositePaneCount,
      0,
    ]));
    this.primitiveBuffer?.destroy();
    this.primitiveBuffer = undefined;
    const reuseGlassBuffer = Boolean(
      this.glassBuffer && scene?.glassCacheKey && scene.glassCacheKey === this.glassCacheKey,
    );
    if (!reuseGlassBuffer) {
      this.glassBuffer?.destroy();
      this.glassBuffer = undefined;
      this.glassCacheKey = undefined;
    }
    if (canEncodeSparseVoxelDryScene(source, scene)) {
      const primitiveRecords = scene!.primitiveRecords;
      this.primitiveBuffer = this.device.createBuffer({ label: "Sparse voxel analytic primitive records", size: primitiveRecords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.primitiveBuffer, 0, primitiveRecords);
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

  /**
   * Materializes the source owner's exact black-slot set as a sampled bit
   * plane. It changes only with the immutable radiance publication, never with
   * the camera, so normal frames do no uploads or allocation work.
   */
  private updateTetrahedralRadianceBlackPages(
    radiance: SparseVoxelSceneRenderSource["tetrahedralRadiance"],
  ): void {
    this.tetrahedralRadianceBlackPages?.destroy();
    this.tetrahedralRadianceBlackPages = undefined;
    this.tetrahedralRadianceBlackPagesView = undefined;
    if (!radiance?.blackSlots?.size || radiance.plan.pages.length === 0) return;
    const flags = new Uint32Array(radiance.plan.pages.length);
    for (const slot of radiance.blackSlots) {
      if (slot >= 0 && slot < flags.length) flags[slot] = 1;
    }
    const texture = this.device.createTexture({
      label: `Sparse voxel tetrahedral-radiance black pages generation ${radiance.generation}`,
      size: [flags.length, 1],
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture },
      flags,
      { bytesPerRow: flags.byteLength },
      [flags.length, 1],
    );
    this.tetrahedralRadianceBlackPages = texture;
    this.tetrahedralRadianceBlackPagesView = texture.createView();
  }

  /**
   * Re-pose a contiguous run of analytic primitive records in place.
   *
   * This is the whole cost of authored scenery motion. Identity, material and
   * dimensions are unchanged, so the published sparse world — its bricks, its
   * material owners, its occupancy mips — stays exactly as valid as it was, and
   * the caller has already bounded the motion so a re-posed surface cannot
   * leave the cell that owns it (lib/scenery-sway.ts). The reusable-frame and
   * primary-visibility caches are the one thing that would go stale, so a
   * moving scene simply never populates them.
   */
  updatePrimitiveRecords(records: Uint32Array<ArrayBuffer>, firstPrimitiveIndex: number): boolean {
    if (!this.primitiveBuffer || !records.byteLength) return false;
    if (records.byteLength % SVO_PRIMITIVE_RECORD_STRIDE_BYTES !== 0) throw new RangeError("Animated primitive records must be whole records");
    const offset = firstPrimitiveIndex * SVO_PRIMITIVE_RECORD_STRIDE_BYTES;
    if (!Number.isSafeInteger(firstPrimitiveIndex) || firstPrimitiveIndex < 0
      || offset + records.byteLength > this.primitiveBuffer.size) {
      throw new RangeError("Animated primitive span falls outside the published primitive records");
    }
    this.device.queue.writeBuffer(this.primitiveBuffer, offset, records);
    this.clearReusableFrame();
    this.clearPrimaryVisibilityCache();
    this.worldGiCacheDirty = true;
    return true;
  }

  /** Enable finished-image visibility effects without rebuilding scene-owned resources. */
  setLightingOptions(options: SparseVoxelDrySceneLightingOptions): void {
    const coneLightingScale = options.coneLightingScale ?? 1;
    if (options.shadowsEnabled === this.lightingOptions.shadowsEnabled
      && options.ambientOcclusionEnabled === this.lightingOptions.ambientOcclusionEnabled
      && coneLightingScale === this.coneScale) return;
    const invalidateWorldGi = options.ambientOcclusionEnabled !== this.lightingOptions.ambientOcclusionEnabled;
    this.lightingOptions = { shadowsEnabled: options.shadowsEnabled, ambientOcclusionEnabled: options.ambientOcclusionEnabled };
    this.coneScale = coneLightingScale;
    if (invalidateWorldGi) this.worldGiCacheDirty = true;
    if (coneLightingScale !== 1) {
      const coneBundle = this.conePipelineBundles.get(coneLightingScale);
      if (coneBundle) this.activateConePipelineBundle(coneLightingScale, coneBundle);
      const splitBundle = this.splitPipelineBundles.get(coneLightingScale);
      if (splitBundle) this.activateSplitPipelineBundle(coneLightingScale, splitBundle);
    }
    this.clearReusableFrame();
    if (this.source && this.scene && canEncodeSparseVoxelDryScene(this.source, this.scene)) {
      this.writeParams(this.source, this.scene);
    }
    // Fail-soft: frames stay on the inline path until the variant pipelines resolve.
    if (coneLightingScale !== 1) void this.ensureConeLightingPrepass().catch(() => {});
    else if (this.shadingPath === "split") void this.ensureSplitPipelines(1).catch(() => {});
  }

  /** Update bounded shader work budgets without rebuilding scene resources. */
  setRenderTuning(tuning: SvoRenderTuning): void {
    const normalized = normalizeSvoRenderTuning(tuning);
    if (Object.keys(normalized).every((key) => normalized[key as keyof SvoRenderTuning] === this.renderTuning[key as keyof SvoRenderTuning])) return;
    this.renderTuning = normalized;
    this.clearReusableFrame();
    this.worldGiCacheDirty = true;
    if (this.source && this.scene && canEncodeSparseVoxelDryScene(this.source, this.scene)) this.writeParams(this.source, this.scene);
    const relight = normalized.coneRadianceReconstruction === "wide-relight"
      || normalized.coneRadianceReconstruction === "full-res-relight";
    if (this.shadingPath === "auto-relight" && relight && this.coneScale !== 1) {
      // Fail-soft while the one-time relight split variant compiles. The
      // existing inline reduced pipeline remains valid in the meantime.
      void this.ensureConeLightingPrepass().catch(() => {});
    }
  }

  /**
   * Refresh only the fluid-coverage block.
   *
   * It is deliberately excluded from the memoized whole-params write: the
   * volume's generation advances every simulation frame, and folding that into
   * the comparison would defeat the early-out for every other field.
   */
  private refreshFluidCoverageFrame(): void {
    if (!this.fluidCoverage) return;
    this.device.queue.writeBuffer(this.paramsBuffer, SVO_DRY_SCENE_PARAMS_LAYOUT.fluidCoverageWordOffset * 4, this.fluidCoverage.frame());
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
    const shadowsEnabled = this.lightingOptions.shadowsEnabled && scene.shadowVisibilityEnabled !== false;
    const ambientOcclusionEnabled = this.lightingOptions.ambientOcclusionEnabled && scene.contactVisibilityEnabled !== false;
    const nodeMip = source.nodeMipPyramid;
    const tetrahedralRadiance = source.tetrahedralRadiance;
    const giReady = Boolean(nodeMip && tetrahedralRadiance
      && nodeMip.generation === tetrahedralRadiance.generation
      && nodeMip.plan.complete && tetrahedralRadiance.plan.complete);
    const coneFallback = !giReady;
    const visibilityFlags = (!giReady && ambientOcclusionEnabled ? SVO_DRY_VISIBILITY_FLAGS.exactContact | SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion : 0)
      | (shadowsEnabled ? SVO_DRY_VISIBILITY_FLAGS.exactShadow : 0)
      | (coneFallback && (shadowsEnabled || ambientOcclusionEnabled) || giReady ? SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested : 0)
      | (giReady ? SVO_DRY_VISIBILITY_FLAGS.globalIllumination : 0)
      | (giReady && ambientOcclusionEnabled ? SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion : 0)
      | SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested;
    words.set([pbrMaterials.count, pbrMaterials.revision, pbrMaterials.strideBytes, visibilityFlags], SVO_DRY_SCENE_PARAMS_LAYOUT.materialPublicationWordOffset);
    const tuning = this.renderTuning;
    words.set([
      tuning.primaryLeafVisits, tuning.coneStepBudget, tuning.maximumShadedLights, tuning.stableAreaLightSamples,
      tuning.movingAreaLightSamples, tuning.stableAoSamples, tuning.movingAoSamples, tuning.visibilityNodeVisits,
      tuning.visibilityLeafVisits, tuning.visibilityWorkItems, tuning.visibilityIntersections,
      SVO_CONE_RADIANCE_RECONSTRUCTION_CODES[tuning.coneRadianceReconstruction],
    ], SVO_DRY_SCENE_PARAMS_LAYOUT.tuningWordOffset);
    floats.set([
      tuning.shadowBiasCells, tuning.shadowStrength, tuning.aoRadiusScale, tuning.aoStrength,
      tuning.aoConeAperture, tuning.shadowConeAperture, tuning.coneNormalEscapeCells, tuning.coneEmitterClearanceCells,
    ], SVO_DRY_SCENE_PARAMS_LAYOUT.tuningWordOffset + 12);
    floats.set(nodeMip?.worldOrigin_m ?? structural.domain.worldOrigin_m, SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipOriginWordOffset);
    floats.set(nodeMip?.worldExtent_m ?? structural.domain.dimensionsCells.map((cells, axis) => cells * structural.domain.cellSize_m[axis]), SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipExtentWordOffset);
    floats.set([
      tuning.giBounceStrength, tuning.giOcclusionStrength, tuning.giEnvironmentStrength, tuning.giDirectStrength,
    ], SVO_DRY_SCENE_PARAMS_LAYOUT.giLightingWordOffset);
    floats.set([tuning.giConeAperture, tuning.giConeCount, 0, 0], SVO_DRY_SCENE_PARAMS_LAYOUT.giConesWordOffset);
    if (nodeMip && nodeMip.generation > 0 && nodeMip.plan.complete) {
      words.set([nodeMip.generation, nodeMip.plan.pages.length, Math.max(1, ...nodeMip.plan.pages.map((page) => page.key.level + 1)), 1], SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipWordOffset);
      words.set([...nodeMip.plan.atlas.texels, 0], SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipAtlasWordOffset);
      // Directory rows are level-major; boundary i counts pages with level < i so
      // the WGSL binary search can restrict itself to one level's contiguous run.
      const levelStart = new Uint32Array(12);
      for (const page of nodeMip.plan.pages) if (page.key.level < 11) levelStart[page.key.level + 1] += 1;
      for (let boundary = 1; boundary < levelStart.length; boundary += 1) levelStart[boundary] += levelStart[boundary - 1];
      words.set(levelStart, SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipLevelStartWordOffset);
      if (nodeMip.directPageTableReady) {
        words.set([...nodeMip.directPageTableDimensions, 1], SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipDirectWordOffset);
        words.set(nodeMip.directPageTableLevelZOffsets, SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipDirectLevelZWordOffset);
      }
    }
    if (giReady) words.set([tetrahedralRadiance!.generation, 1, 0, 0], SVO_DRY_SCENE_PARAMS_LAYOUT.tetrahedralRadianceWordOffset);
    const wide = resolveSvoWideTraversalCapability(source.wideFanout, source.revision, structural.domain.maximumDepth);
    if (wide.status === "ready") {
      const publication = wide.publication;
      words.set([publication.generation, publication.sourceGeneration, publication.pageCount, publication.descriptorCount],
        SVO_DRY_SCENE_PARAMS_LAYOUT.wideFanoutWordOffset);
    }
    if (this.paramsWords?.length === words.length && words.every((word, index) => word === this.paramsWords![index])) return;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    this.paramsWords = Uint32Array.from(words);
  }

  private rebuild(): void {
    const source = this.source, structural = source?.structural;
    if (!this.layout || !this.pipeline || !source || !structural || !this.primitiveBuffer || !this.glassBuffer || !this.scene) {
      this.bindGroup = undefined;
      this.coneFanoutSceneBindGroup = undefined;
      return;
    }
    const nodeMip = source.nodeMipPyramid;
    const tetrahedralRadiance = source.tetrahedralRadiance;
    const wide = resolveSvoWideTraversalCapability(source.wideFanout, source.revision, structural.domain.maximumDepth);
    const compact = resolveWebGpuSvoCompactHierarchy(source.compactHierarchy, {
      nodeCount: structural.capacities.nodes,
      leafCount: structural.capacities.leaves,
      sourceGeneration: source.revision,
    });
    if (this.traversalMode === "compact" && compact.status !== "ready") {
      this.bindGroup = undefined;
      this.coneFanoutSceneBindGroup = undefined;
      return;
    }
    this.bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.bodyBuffer } },
      { binding: 2, resource: structural.control }, { binding: 3, resource: structural.nodes }, { binding: 4, resource: structural.leaves },
      { binding: 5, resource: structural.materialOwners }, { binding: 6, resource: source.pbrMaterials!.binding }, { binding: 7, resource: { buffer: this.primitiveBuffer } },
      { binding: 8, resource: structural.publication.state }, { binding: 9, resource: { buffer: this.paramsBuffer } },
      { binding: 10, resource: { buffer: this.glassBuffer } },
      { binding: 11, resource: this.traversalMode === "compact" && compact.status === "ready"
        ? compact.source.nodes : wide.status === "ready" ? wide.source.pages : structural.nodes },
      { binding: 12, resource: wide.status === "ready" ? wide.source.descriptors : structural.leaves },
      { binding: 13, resource: { buffer: this.lightingBuffer } },
      { binding: 14, resource: { buffer: this.rigidMotionUniformBuffer } },
      { binding: 15, resource: { buffer: this.thickGlassUniformBuffer } },
      { binding: 16, resource: nodeMip?.view ?? this.nodeMipFallbackAtlasView },
      { binding: 17, resource: nodeMip?.sampler ?? this.nodeMipFallbackSampler },
      { binding: 18, resource: nodeMip?.directoryView ?? this.nodeMipFallbackDirectoryView },
      { binding: 19, resource: this.fluidCoverage?.visibleGeneration()?.view ?? this.fluidCoverageFallbackView },
      { binding: 20, resource: nodeMip?.directPageTableView ?? this.nodeMipFallbackDirectPageTableView },
      { binding: 21, resource: tetrahedralRadiance?.views[0] ?? this.tetrahedralRadianceFallbackViews[0] },
      { binding: 22, resource: tetrahedralRadiance?.views[1] ?? this.tetrahedralRadianceFallbackViews[1] },
      { binding: 23, resource: tetrahedralRadiance?.views[2] ?? this.tetrahedralRadianceFallbackViews[2] },
      { binding: 24, resource: tetrahedralRadiance?.views[3] ?? this.tetrahedralRadianceFallbackViews[3] },
      { binding: 25, resource: this.tetrahedralRadianceBlackPagesView ?? this.tetrahedralRadianceBlackFallbackView },
    ] });
    this.coneFanoutSceneBindGroup = this.coneFanout && this.coneFanoutSceneLayout
      ? this.device.createBindGroup({
        label: "Sparse voxel cone fan-out scene resources",
        layout: this.coneFanoutSceneLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.paramsBuffer } },
          { binding: 2, resource: { buffer: this.lightingBuffer } },
          { binding: 3, resource: structural.publication.state },
          { binding: 4, resource: nodeMip?.view ?? this.nodeMipFallbackAtlasView },
          { binding: 5, resource: nodeMip?.sampler ?? this.nodeMipFallbackSampler },
          { binding: 6, resource: nodeMip?.directoryView ?? this.nodeMipFallbackDirectoryView },
          { binding: 7, resource: this.fluidCoverage?.visibleGeneration()?.view ?? this.fluidCoverageFallbackView },
          { binding: 8, resource: nodeMip?.directPageTableView ?? this.nodeMipFallbackDirectPageTableView },
        ],
      })
      : undefined;
  }

  /** GPU-authored storage is copied into this pass's uniform mirror to preserve the ten-storage adapter budget. */
  /**
   * Attach the frame's fluid coverage volume.
   *
   * The volume is owned by the presentation layer, not the structural source:
   * it is resampled from the solver's coarse level set, which reaches the
   * renderer as a dense field rather than through the sparse publication. An
   * absent volume rebinds the zeroed fallback and reports an invalid frame, so
   * the shadow cone skips every fluid fetch.
   */
  setFluidCoverage(coverage: WebGpuSvoFluidCoverage | undefined): void {
    if (this.fluidCoverage === coverage) return;
    this.fluidCoverage = coverage;
    this.rebuild();
  }

  setRigidMotionSource(source: GPUBuffer | undefined): void {
    if (!source && this.rigidMotionSource) this.device.queue.writeBuffer(this.rigidMotionUniformBuffer, 0, new Uint32Array(SVO_DRY_RIGID_MOTION_UNIFORM_BYTES / 4));
    this.rigidMotionSource = source;
  }

  ensureSize(width: number, height: number): void {
    if (this.gBufferTargets.ensureSize(width, height)) { this.pickingFrameToken += 1; this.lastPickingTarget = undefined; this.clearReusableFrame(); this.clearPrimaryVisibilityCache(); }
    this.targetWidth = width;
    this.targetHeight = height;
    this.ensureConePrepassTargets();
    this.ensureSplitTargets();
  }

  /** Heatmap counters are invocation-private, so diagnostics intentionally retain the inline path. */
  setDiagnosticOverlayActive(active: boolean): void {
    if (active === this.splitDiagnosticsActive) return;
    this.splitDiagnosticsActive = active;
    this.clearReusableFrame();
    this.clearPrimaryVisibilityCache();
  }

  /** Copies the compacted boundary count for offline Dawn experiment diagnosis. */
  copyConeBoundaryCount(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    if (!this.conePrepassBoundaryQueue) return false;
    encoder.copyBufferToBuffer(this.conePrepassBoundaryQueue, 0, target, 0, 4);
    return true;
  }

  /** Auxiliary MRTs and reversed-Z depth for picking and split shading. */
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

  /* ----------------------------------------------------------------------- */
  /* Live pixel trace                                                        */
  /* ----------------------------------------------------------------------- */

  /**
   * Ask for the next frame's probe to trace this pixel.
   *
   * Requests supersede rather than queue: a pointer moving across the viewport
   * should produce the newest ray, not a backlog of stale ones. The probe's own
   * module and pipeline compile on the first request, so a session that never
   * opens the diagnostic never pays for it.
   */
  requestPixelTrace(pixelX: number, pixelY: number): void {
    if (this.probeCompilationFailed) return;
    if (!Number.isSafeInteger(pixelX) || !Number.isSafeInteger(pixelY)) return;
    const width = this.targetWidth, height = this.targetHeight;
    if (width < 1 || height < 1) return;
    this.probeRequest = {
      pixelX: Math.max(0, Math.min(width - 1, pixelX)),
      pixelY: Math.max(0, Math.min(height - 1, pixelY)),
      token: (this.probeRequest?.token ?? 0) + 1,
    };
    void this.ensurePixelProbe();
  }

  clearPixelTraceRequest(): void { this.probeRequest = undefined; }

  /**
   * Resource/source epoch. Every republication of the scene — geometry, analytic
   * primitives, materials, the light arena — lands through `setSource` and bumps
   * this, so a caller holding an answer about the old scene can tell.
   */
  get sceneEpoch(): number { return this.pickingFrameToken; }

  /**
   * The pixel the next probe will trace, in presentation-target pixels.
   *
   * Exposed so a caller can tell whether the trace it is holding answers the
   * pixel it last asked about. Resolution scaling means only this class knows
   * how a viewport fraction became a pixel index.
   */
  get pixelTraceRequestedPixel(): readonly [number, number] | undefined {
    return this.probeRequest ? [this.probeRequest.pixelX, this.probeRequest.pixelY] : undefined;
  }

  get pixelTraceReady(): boolean { return Boolean(this.probePipeline && this.probeBuffers); }

  /** True once the probe has been refused for this device; never retried. */
  get pixelTraceUnsupported(): boolean { return this.probeCompilationFailed; }

  /** True while a request exists but its pipelines are still being built. */
  get pixelTraceCompiling(): boolean {
    return Boolean(this.probeRequest) && !this.probeCompilationFailed && !this.probePipeline;
  }

  private async ensurePixelProbe(): Promise<void> {
    if (this.probePipeline || this.probeCompilationFailed) return;
    this.probeCompilation ??= (async () => {
      this.device.pushErrorScope("validation");
      let errorScopeOpen = true;
      try {
        if (!this.layout || !this.vertexModule) throw new Error("Dry-scene pipelines are not initialized");
        // Records go to a storage texture precisely because the dry pass already
        // spends the whole ten-storage-buffer budget browsers report here. The
        // inline path binds no storage texture, so one slot is always free; the
        // gate stays anyway so an unusual device refuses cleanly.
        if (this.device.limits.maxStorageTexturesPerShaderStage < 1) {
          throw new Error("Pixel-trace probe needs one storage-texture binding; this device allows none");
        }
        const module = await checkedModule(
          this.device,
          `Sparse voxel pixel-trace probe (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
          createSvoDrySceneFragmentWGSL(1, this.traversalMode, this.brickOccupancyMode, "inline", 0, true),
        );
        this.probeLayout = this.device.createBindGroupLayout({
          label: "Sparse voxel pixel-trace probe records",
          entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, storageTexture: { access: "write-only", format: "r32uint" } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
          ],
        });
        this.probeBuffers = new SparseVoxelPixelTraceBuffers(this.device);
        this.probeBindGroup = this.device.createBindGroup({
          label: "Sparse voxel pixel-trace probe bind group",
          layout: this.probeLayout,
          entries: [
            { binding: 0, resource: this.probeBuffers.recordsView },
            { binding: 1, resource: { buffer: this.probeBuffers.request } },
          ],
        });
        // A one-pixel target: the probe's output colour is unused, and the guard
        // in the entry point relies on there being exactly one covered pixel.
        this.probeTarget = this.device.createTexture({
          label: "Sparse voxel pixel-trace probe target",
          size: [1, 1],
          format: "rgba8unorm",
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.probeTargetView = this.probeTarget.createView();
        this.probePipeline = await this.device.createRenderPipelineAsync({
          label: "Sparse voxel pixel-trace probe",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.probeLayout] }),
          vertex: { module: this.vertexModule, entryPoint: "vertexMain" },
          fragment: { module, entryPoint: "dryProbeMain", targets: [{ format: "rgba8unorm" }] },
          primitive: { topology: "triangle-list" },
        });
        errorScopeOpen = false;
        const validation = await this.device.popErrorScope();
        if (validation) throw new Error(validation.message);
      } catch (error) {
        if (errorScopeOpen) void this.device.popErrorScope().catch(() => undefined);
        this.probeCompilationFailed = true;
        this.probePipeline = undefined;
        this.probeBuffers?.destroy();
        this.probeBuffers = undefined;
        this.probeTarget?.destroy();
        this.probeTarget = undefined;
        this.probeTargetView = undefined;
        this.probeBindGroup = undefined;
        console.warn("Sparse voxel pixel-trace probe unavailable", error);
      }
    })();
    await this.probeCompilation;
  }

  /**
   * Encode the pending probe and its readback. Called after the frame's own
   * passes so the probe reads the same published topology the frame drew from.
   */
  encodePixelTrace(encoder: GPUCommandEncoder): boolean {
    const request = this.probeRequest;
    if (!request || !this.probePipeline || !this.probeBindGroup || !this.probeBuffers
      || !this.probeTargetView || !this.bindGroup || this.probeReadPending) return false;
    this.probeBuffers.writeRequest(request);
    const pass = encoder.beginRenderPass({
      label: "Sparse voxel pixel-trace probe",
      colorAttachments: [{ view: this.probeTargetView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(this.probePipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setBindGroup(SVO_DRY_SCENE_PIXEL_PROBE_GROUP, this.probeBindGroup);
    pass.draw(3);
    pass.end();
    if (!this.probeBuffers.encodeReadback(encoder)) return false;
    this.probeEncodedToken = request.token;
    this.probeReadPending = true;
    return true;
  }

  /**
   * Resolve the encoded trace. Resolves to `undefined` when the request was
   * superseded before the map completed, which is the common case while the
   * pointer is moving.
   */
  async readPixelTrace(): Promise<SvoPixelTrace | undefined> {
    if (!this.probeBuffers || !this.probeReadPending) return undefined;
    // A newer pointer position does not invalidate this trace: it answers the
    // pixel it was asked about and the next frame supersedes it. Only a resource
    // or source epoch change makes the recorded world-space boxes meaningless.
    const generation = this.pickingFrameToken;
    try {
      return await this.probeBuffers.read(() => this.pickingFrameToken === generation);
    } finally {
      this.probeReadPending = false;
    }
  }

  encode(encoder: GPUCommandEncoder, target: GPUTexture | GPUTextureView, reuseKey?: string, tracePhase?: RenderPathTracePhase): DrySceneReplacementResult | false {
    if (!this.pipeline || !this.bindGroup) return false;
    // The coverage volume allocates lazily and only reports itself once a fill
    // has been encoded, so its validity flips mid-session. Refresh the frame
    // every encode rather than relying on a source change to carry it.
    this.refreshFluidCoverageFrame();
    const gBufferViews = this.gBufferTargets.views;
    if (!gBufferViews) return false;
    // Reduced-rate cone lighting activates only once its pipelines and targets
    // are resolved; until then frames stay on the bit-exact inline path. The
    // effective frame key carries the active scale so toggling invalidates reuse.
    const usePrepass = this.coneScale !== 1 && this.conePipelineScale === this.coneScale
      && Boolean(this.conePrepassGeometryPipeline && this.conePrepassVisibilityPipeline
        && this.conePrepassShadePipeline && this.coneReducedPipeline
        && this.conePrepassBindGroup && this.conePrepassVisibilityBindGroup && this.conePrepassShadeBindGroup && this.conePrepassVisibilityView
        && this.conePrepassGeometryView && this.conePrepassIdentityView && this.conePrepassRadianceView);
    const effectiveScale: SvoConeLightingScale = usePrepass ? this.coneScale : 1;
    const splitRequested = this.shadingPath === "split";
    const activeSplitVisibilityPipeline = this.rasterRigidActive
      ? this.splitRasterRigidVisibilityPipeline
      : this.splitVisibilityPipeline;
    const useSplit = splitRequested && !this.splitDiagnosticsActive
      && this.splitPipelineScale === effectiveScale
      && Boolean(activeSplitVisibilityPipeline && this.splitLightingPipeline
        && (!usePrepass || (this.conePrepassResetPipeline && this.conePrepassCoherentPipeline && this.conePrepassBoundaryPipeline
          && this.conePrepassComputeBindGroup && this.conePrepassBoundaryQueue
          && this.worldGiFramePipeline && this.worldGiCachePipeline && this.worldGiCacheBindGroup
          && this.worldGiCacheBuffer && this.worldGiFrameBuffer
          && (!this.coneFanout || (this.coneFanoutWorkerPipeline && this.coneFanoutReducerPipeline
            && this.coneFanoutSceneBindGroup && this.coneFanoutWorkerBindGroup && this.coneFanoutReducerBindGroup))))
        && (!this.rasterGlassDiscovery || (this.rasterGlassPipeline && this.rasterGlassBindGroup && this.splitGlassKeyView && this.splitGlassDepthView))
        && (!this.rasterRigidActive || (this.rasterRigidPipeline && this.rasterRigidBridgePipeline
          && this.rasterRigidInputBindGroup && this.rasterRigidBindGroup && this.rasterRigidPrimaryGeometryView))
        && this.splitVisibilityBindGroup && this.splitLightingBindGroup && this.splitGeometryView);
    const frameKey = reuseKey === undefined ? undefined : `${reuseKey}|cone=${effectiveScale}|shading=${useSplit ? "split" : "inline"}|rasterRigid=${this.rasterRigidActive}`;
    const primaryFrameKey = reuseKey === undefined ? undefined : `${reuseKey}|primary=${useSplit ? "split" : "inline"}|rasterRigid=${this.rasterRigidActive}`;
    const reusePrimaryVisibility = !this.rasterRigidActive && svoDryPrimaryCoherenceDecision(
      // Reduced split shading computes a complete cone-visibility plane every
      // frame, so its primary G-buffer is parity-invariant. Scale 1 still owns
      // checkerboard shadow-deferred flags and must always retrace.
      this.rayCoherenceMode, useSplit && usePrepass, primaryFrameKey, this.primaryVisibilityCacheKey,
    ) === "reuse";
    const targetTexture = "width" in target ? target as GPUTexture : undefined;
    if (this.rayCoherenceMode === "off" && frameKey && targetTexture && frameKey === this.reusableKey && targetTexture === this.reusableTarget
      && this.reusableStableFrames >= 1 && this.reusableResult) return this.reusableResult;
    let targetView = target as GPUTextureView;
    if (targetTexture) {
      targetView = this.targetViews.get(targetTexture) ?? targetTexture.createView();
      this.targetViews.set(targetTexture, targetView);
    }
    if (this.rigidMotionSource) encoder.copyBufferToBuffer(this.rigidMotionSource, 0, this.rigidMotionUniformBuffer, 0, SVO_DRY_RIGID_MOTION_UNIFORM_BYTES);
    if (usePrepass && !useSplit) {
      const geometry = encoder.beginRenderPass({
        label: "Sparse voxel cone-prepass geometry",
        colorAttachments: [
          { view: this.conePrepassGeometryView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          { view: this.conePrepassIdentityView!, clearValue: { r: 4294967295, g: 4294967295, b: 4294967295, a: 4294967295 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      geometry.setPipeline(this.conePrepassGeometryPipeline!);
      geometry.setBindGroup(0, this.bindGroup);
      geometry.draw(3);
      geometry.end();
      const visibility = encoder.beginRenderPass({
        label: "Sparse voxel cone-prepass visibility",
        colorAttachments: [
          { view: this.conePrepassVisibilityView!, clearValue: { r: 4294967295, g: 4294967295, b: 4294967295, a: 4294967295 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      visibility.setPipeline(this.conePrepassVisibilityPipeline!);
      visibility.setBindGroup(0, this.bindGroup);
      visibility.setBindGroup(1, this.conePrepassVisibilityBindGroup!);
      visibility.draw(3);
      visibility.end();
      // Relight deliberately consumes only the reduced visibility cache. Its
      // full-rate material/BRDF work never reads radiance, so omit this pass.
      const shade = encoder.beginRenderPass({
        label: "Sparse voxel reduced-rate opaque shading",
        colorAttachments: [
          { view: this.conePrepassRadianceView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      shade.setPipeline(this.conePrepassShadePipeline!);
      shade.setBindGroup(0, this.bindGroup);
      shade.setBindGroup(1, this.conePrepassShadeBindGroup!);
      shade.draw(3);
      shade.end();
      tracePhase?.({ id: "svo-cone-lighting", label: "SVO cone-lighting prepass" });
    }
    if (useSplit) {
      const splitGroup = usePrepass ? 2 : 1;
      if (!reusePrimaryVisibility) {
        const visibility = encoder.beginRenderPass({
          label: "Sparse voxel primary visibility",
          colorAttachments: [
            { view: gBufferViews.packedSurface, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
            { view: gBufferViews.identityMedia, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          ],
          depthStencilAttachment: {
            view: gBufferViews.hardwareDepth,
            depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        visibility.setPipeline(activeSplitVisibilityPipeline!);
        visibility.setBindGroup(0, this.bindGroup);
        if (usePrepass) visibility.setBindGroup(1, this.conePrepassBindGroup!);
        visibility.setBindGroup(splitGroup, this.splitVisibilityBindGroup!);
        visibility.draw(3);
        visibility.end();
        tracePhase?.({ id: "svo-primary", label: "SVO primary visibility" });
        if (this.rasterRigidActive) {
          const rigid = encoder.beginRenderPass({
            label: "Sparse voxel analytic rigid primary discovery",
            colorAttachments: [
              { view: gBufferViews.packedSurface, loadOp: "load", storeOp: "store" },
              { view: gBufferViews.identityMedia, loadOp: "load", storeOp: "store" },
              { view: this.rasterRigidPrimaryGeometryView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
            ],
            depthStencilAttachment: {
              view: gBufferViews.hardwareDepth,
              depthLoadOp: "load",
              depthStoreOp: "store",
            },
          });
          rigid.setPipeline(this.rasterRigidPipeline!);
          rigid.setBindGroup(0, this.rasterRigidInputBindGroup!);
          rigid.draw(SVO_RIGID_RASTER_CONTRACT.verticesPerProxy, SVO_RIGID_RASTER_CONTRACT.maximumBodies);
          rigid.end();
          const bridge = encoder.beginRenderPass({
            label: "Sparse voxel raster-rigid certificate bridge",
            colorAttachments: [
              { view: this.splitGeometryView!, loadOp: "load", storeOp: "store" },
              { view: this.splitOpaqueIdentityView!, loadOp: "load", storeOp: "store" },
            ],
          });
          bridge.setPipeline(this.rasterRigidBridgePipeline!);
          bridge.setBindGroup(0, this.rasterRigidInputBindGroup!);
          bridge.setBindGroup(1, this.rasterRigidBindGroup!);
          bridge.draw(SVO_RIGID_RASTER_CONTRACT.verticesPerProxy, SVO_RIGID_RASTER_CONTRACT.maximumBodies);
          bridge.end();
          tracePhase?.({ id: "svo-rigid", label: "SVO analytic rigid discovery" });
        }
        if (this.rasterGlassDiscovery) {
          const glass = encoder.beginRenderPass({
            label: "Sparse voxel raster thin-glass discovery",
            colorAttachments: [
              { view: this.splitGlassKeyView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
            ],
            depthStencilAttachment: {
              view: this.splitGlassDepthView!,
              depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
              depthLoadOp: "clear",
              depthStoreOp: "store",
            },
          });
          glass.setPipeline(this.rasterGlassPipeline!);
          glass.setBindGroup(0, this.bindGroup);
          glass.setBindGroup(1, this.rasterGlassBindGroup!);
          glass.draw(6, this.rasterGlassRecordCount, 0, this.rasterGlassFirstRecord);
          glass.end();
          tracePhase?.({ id: "svo-glass", label: "SVO raster thin-glass discovery" });
        }
        this.primaryVisibilityCacheKey = this.rayCoherenceMode === "static-primary" && usePrepass
          ? primaryFrameKey
          : undefined;
      }

      if (usePrepass) {
        if (this.experiments.clearConeQueueWithBlit) encoder.clearBuffer(this.conePrepassBoundaryQueue!, 0, 4);
        const coherent = encoder.beginComputePass({ label: "Sparse voxel compact cone visibility" });
        if (!this.experiments.clearConeQueueWithBlit && !this.experiments.inlineConeBoundaries) {
          coherent.setPipeline(this.conePrepassResetPipeline!);
          coherent.setBindGroup(0, this.bindGroup);
          coherent.setBindGroup(1, this.conePrepassComputeBindGroup!);
          coherent.setBindGroup(splitGroup, this.splitLightingBindGroup!);
          coherent.dispatchWorkgroups(1);
        }
        coherent.setPipeline(this.conePrepassCoherentPipeline!);
        coherent.setBindGroup(0, this.bindGroup);
        coherent.setBindGroup(1, this.conePrepassComputeBindGroup!);
        coherent.setBindGroup(splitGroup, this.splitLightingBindGroup!);
        coherent.dispatchWorkgroups(Math.ceil(this.conePrepassWidth / 8), Math.ceil(this.conePrepassHeight / 8));
        if (!this.experiments.inlineConeBoundaries) {
          coherent.setPipeline(this.conePrepassBoundaryPipeline!);
          coherent.dispatchWorkgroups(Math.ceil(this.conePrepassWidth * this.conePrepassHeight / 64));
        }
        coherent.end();
        if (this.coneFanout) {
          const fanout = encoder.beginComputePass({ label: "Sparse voxel cone sample fan-out" });
          fanout.setPipeline(this.coneFanoutWorkerPipeline!);
          fanout.setBindGroup(0, this.coneFanoutSceneBindGroup!);
          fanout.setBindGroup(1, this.coneFanoutWorkerBindGroup!);
          fanout.dispatchWorkgroups(
            Math.ceil(this.conePrepassWidth / SVO_CONE_FANOUT_CONTRACT.workgroupSize[0]),
            Math.ceil(this.conePrepassHeight / SVO_CONE_FANOUT_CONTRACT.workgroupSize[1]),
            SVO_CONE_FANOUT_CONTRACT.lightLayerBase + this.coneFanoutLightCount,
          );
          fanout.end();
          const reduce = encoder.beginComputePass({ label: "Sparse voxel cone sample reduction" });
          reduce.setPipeline(this.coneFanoutReducerPipeline!);
          reduce.setBindGroup(0, this.coneFanoutReducerBindGroup!);
          reduce.dispatchWorkgroups(
            Math.ceil(this.conePrepassWidth / SVO_CONE_FANOUT_CONTRACT.workgroupSize[0]),
            Math.ceil(this.conePrepassHeight / SVO_CONE_FANOUT_CONTRACT.workgroupSize[1]),
          );
          reduce.end();
        }
        tracePhase?.({ id: "svo-cone-lighting", label: "SVO compacted cone lighting" });

        // The cache is world-space and source-owned: camera motion changes
        // which keys are queried but never invalidates entries. Only source,
        // authored-scene, or lighting-contract changes clear it.
        if (this.worldGiCacheDirty) {
          encoder.clearBuffer(this.worldGiCacheBuffer!);
          this.worldGiCacheDirty = false;
        }
        const gi = encoder.beginComputePass({ label: "Sparse voxel persistent world GI cache" });
        gi.setPipeline(this.worldGiFramePipeline!);
        gi.setBindGroup(0, this.bindGroup);
        gi.setBindGroup(1, this.conePrepassShadeBindGroup!);
        gi.setBindGroup(2, this.worldGiCacheBindGroup!);
        gi.dispatchWorkgroups(1);
        gi.setPipeline(this.worldGiCachePipeline!);
        gi.setBindGroup(0, this.bindGroup);
        gi.setBindGroup(1, this.conePrepassShadeBindGroup!);
        gi.setBindGroup(2, this.worldGiCacheBindGroup!);
        gi.dispatchWorkgroups(Math.ceil(this.conePrepassWidth / 8), Math.ceil(this.conePrepassHeight / 8));
        gi.end();
        tracePhase?.({ id: "svo-environment-gi", label: "SVO persistent world-space environmental GI" });
      }

      const lighting = encoder.beginRenderPass({
        label: "Sparse voxel deferred dry lighting",
        colorAttachments: [{ view: targetView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
      });
      lighting.setPipeline(this.splitLightingPipeline!);
      lighting.setBindGroup(0, this.bindGroup);
      if (usePrepass) lighting.setBindGroup(1, this.conePrepassBindGroup!);
      lighting.setBindGroup(splitGroup, this.splitLightingBindGroup!);
      lighting.draw(3);
      lighting.end();
      tracePhase?.({ id: "dry-scene", label: "SVO deferred dry lighting" });
    } else {
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
      });
      pass.setPipeline(usePrepass ? this.coneReducedPipeline! : this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      if (usePrepass) pass.setBindGroup(1, this.conePrepassBindGroup!);
      pass.draw(3); pass.end();
      tracePhase?.({ id: "svo-primary", label: "SVO traversal + dry shading" });
    }
    this.lastPickingTarget = targetTexture;
    const result = { encoded: true, sampledTargetView: targetView } as const;
    if (this.rayCoherenceMode === "off" && frameKey && targetTexture) {
      this.reusableStableFrames = frameKey === this.reusableKey && targetTexture === this.reusableTarget ? this.reusableStableFrames + 1 : 1;
      this.reusableKey = frameKey; this.reusableTarget = targetTexture; this.reusableResult = result;
    } else this.clearReusableFrame();
    return result;
  }

  destroy(): void {
    this.probeBuffers?.destroy();
    this.probeTarget?.destroy();
    this.probeBuffers = undefined;
    this.probeTarget = undefined;
    this.probeTargetView = undefined;
    this.probeBindGroup = undefined;
    this.probePipeline = undefined;
    this.probeRequest = undefined;
    this.probeReadPending = false;
    this.splitGeometry?.destroy();
    this.splitOpaqueIdentity?.destroy();
    this.splitGlassKey?.destroy();
    this.splitGlassDepth?.destroy();
    this.rasterRigidPrimaryGeometry?.destroy();
    this.splitGeometry = undefined;
    this.splitGeometryView = undefined;
    this.splitOpaqueIdentity = undefined;
    this.splitOpaqueIdentityView = undefined;
    this.splitGlassKey = undefined;
    this.splitGlassKeyView = undefined;
    this.splitGlassDepth = undefined;
    this.splitGlassDepthView = undefined;
    this.rasterRigidPrimaryGeometry = undefined;
    this.rasterRigidPrimaryGeometryView = undefined;
    this.splitVisibilityBindGroup = undefined;
    this.splitLightingBindGroup = undefined;
    this.rasterGlassBindGroup = undefined;
    this.rasterRigidInputBindGroup = undefined;
    this.rasterRigidBindGroup = undefined;
    this.splitVisibilityPipeline = undefined;
    this.splitLightingPipeline = undefined;
    this.rasterGlassPipeline = undefined;
    this.rasterRigidPipeline = undefined;
    this.rasterRigidBridgePipeline = undefined;
    this.conePrepassResetPipeline = undefined;
    this.conePrepassCoherentPipeline = undefined;
    this.conePrepassBoundaryPipeline = undefined;
    this.worldGiFramePipeline = undefined;
    this.worldGiCachePipeline = undefined;
    this.coneFanoutWorkerPipeline = undefined;
    this.coneFanoutReducerPipeline = undefined;
    this.coneFanoutSceneBindGroup = undefined;
    this.splitPipelineScale = undefined;
    this.releaseConePrepassTargets();
    this.worldGiCacheBuffer?.destroy();
    this.worldGiFrameBuffer?.destroy();
    this.worldGiCacheBuffer = undefined;
    this.worldGiFrameBuffer = undefined;
    this.worldGiCacheLayout = undefined;
    this.worldGiCacheDirty = true;
    this.conePrepassGeometryPipeline = undefined;
    this.conePrepassVisibilityPipeline = undefined;
    this.conePrepassShadePipeline = undefined;
    this.coneReducedPipeline = undefined;
    this.conePipelineScale = undefined;
    this.splitPipelineBundles.clear();
    this.splitPipelineCompiles.clear();
    this.conePipelineBundles.clear();
    this.conePipelineCompiles.clear();
    this.primitiveBuffer?.destroy();
    this.glassBuffer?.destroy();
    this.paramsBuffer.destroy();
    this.lightingBuffer.destroy();
    this.rigidMotionUniformBuffer.destroy();
    this.thickGlassUniformBuffer.destroy();
    this.rasterGlassParamsBuffer.destroy();
    this.coneFanoutFrameBuffer?.destroy();
    this.coneFanoutFrameBuffer = undefined;
    this.nodeMipFallbackAtlas.destroy();
    this.nodeMipFallbackDirectory.destroy();
    this.nodeMipFallbackDirectPageTable.destroy();
    this.tetrahedralRadianceFallback.forEach((texture) => texture.destroy());
    this.tetrahedralRadianceBlackFallback.destroy();
    this.tetrahedralRadianceBlackPages?.destroy();
    this.tetrahedralRadianceBlackPages = undefined;
    this.tetrahedralRadianceBlackPagesView = undefined;
    this.gBufferTargets.destroy();
    this.pickingReadback.destroy();
    this.lastPickingTarget = undefined;
    this.clearReusableFrame();
    this.clearPrimaryVisibilityCache();
    this.pickingFrameToken += 1;
    this.bindGroup = undefined;
    this.glassBuffer = undefined;
    this.glassCacheKey = undefined;
    this.paramsWords = undefined;
  }

  private clearReusableFrame(): void {
    this.reusableKey = undefined; this.reusableStableFrames = 0; this.reusableTarget = undefined; this.reusableResult = undefined;
  }

  private clearPrimaryVisibilityCache(): void {
    this.primaryVisibilityCacheKey = undefined;
  }
}

export const svoDrySceneShader = drySceneShader;
export const svoDrySceneVertexShader = drySceneVertexShader;
