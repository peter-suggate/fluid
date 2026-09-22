import { canUseOpaqueDirectionalCones } from "../features/shading/deferred-specialization";
import type { SceneDescription } from "../../core/model";
import {
PLANAR_BOUNDARY_PATCH_BYTES
} from "../../core/planar-boundary";
import type { RenderFrameSeam } from "../../core/render-frame-stages";
import type { ResourcePluginDefinition } from "../../core/resource-readiness";
import type { FrameBandPartitioner } from "../../core/webgpu-frame-band-sampler";
import { type SparseVoxelSceneRenderSource } from "../../core/webgpu-voxel-debug";
import type { DrySceneReplacementResult } from "../../core/webgpu-water-pipeline";
import type { SparseVoxelDrySceneData } from "../contracts/scene-publication";
import {
buildSvoSceneLights,
SVO_LIGHT_KIND_CODES,
SVO_LIGHT_KINDS,
SVO_LIGHT_MAXIMUM_RECORDS,
SVO_LIGHT_RECORD_STRIDE_BYTES,
SVO_LIGHT_RECORD_WORDS
} from "../contracts/svo-light-abi";
import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../contracts/svo-material-abi";
import {
SVO_CLUSTER_FIELD_TABLE,
SVO_CLUSTER_LOBE_DEFAULT_DISPLACEMENT,
SVO_CLUSTER_LOBE_DEFAULT_SPAN,
SVO_CLUSTER_LOBE_DEFAULT_SPAN_SPREAD,
SVO_CLUSTER_SWEEP_MAXIMUM_POINTS,
SVO_PRIMITIVE_KINDS,
SVO_PRIMITIVE_RECORD_STRIDE_BYTES,
SVO_PRIMITIVE_RECORD_WORDS,
SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS,
svoClusterFieldByCode,
svoClusterFieldName,
type SvoClusterResolver,
type SvoFieldProgramResolver,
type SvoSmoothUnionClusterPacking
} from "../contracts/svo-primitive-abi";
import {
type SparseBrickLeafPayloadMode
} from "../features/construction/sparse-brick-octree";
import { resolveWebGpuSvoCompactHierarchy } from "../features/construction/webgpu-svo-compact-hierarchy";
import {
resolveSvoWideTraversalCapability
} from "../features/construction/webgpu-svo-wide-fanout";
import {
createSvoBrickRasterProbeWGSL,
SparseVoxelBrickRasterProbeBuffers,
SVO_BRICK_RASTER_PROBE_CONTRACT,
svoBrickRasterProbeBindGroupLayoutEntries,
} from "../features/diagnostics/webgpu-svo-brick-raster-probe";
import {
SparseVoxelGpuPickingReadbackRing,
svoPickingPixelFromNormalized,
type SvoGpuPickingReadbackResult,
} from "../features/diagnostics/webgpu-svo-picking-readback";
import type { SvoRenderStagePlanes } from "../features/diagnostics/webgpu-svo-stage-overlay";
import { svoSceneLighting } from "../features/lighting-visibility/svo-dry-scene-lighting";
import {
createSvoConeFanoutWorkerWGSL,
packSvoConeFanoutFrame,
SVO_CONE_FANOUT_CONTRACT,
svoConeFanoutReducerBindGroupLayoutEntries,
svoConeFanoutReducerWGSL,
svoConeFanoutSceneBindGroupLayoutEntries,
svoConeFanoutWorkerBindGroupLayoutEntries,
} from "../features/lighting-visibility/webgpu-svo-cone-fanout";
import {
SVO_SCENE_GLASS_MAXIMUM_PANES,
} from "../features/materials/svo-scene-glass";
import { SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES } from "../features/materials/svo-scene-thick-glass";
import {
SVO_THICK_GLASS_RECORD_STRIDE_BYTES,
unpackSvoThickGlassVolumes
} from "../features/materials/svo-thick-glass";
import { SVO_THIN_GLASS_RECORD_STRIDE_BYTES } from "../features/materials/svo-thin-glass";
import { SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL,SVO_SURFACE_MESH_BYTES,SVO_SURFACE_MESH_HEADER_BYTES,SVO_SURFACE_MESH_QUAD_BYTES,SVO_SURFACE_MESH_STATE,SVO_SURFACE_MESH_STATE_BYTES,type SvoSurfaceMeshStatus,interpretSurfaceMeshState,surfaceMeshBuildBricks,surfaceMeshWorkBytes } from "../features/primary-visibility/svo-surface-mesh";
import {
createSvoBrickRasterCullWGSL,
createSvoRasterCoverageOverflowArgsWGSL,
SVO_BRICK_RASTER_CONTRACT,
SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT,
svoBrickRasterCoverageBindGroupLayoutEntries,
svoBrickRasterCoverageCountBytes,
svoBrickRasterCullBindGroupLayoutEntries,
svoBrickRasterDrawBindGroupLayoutEntries,
svoBrickRasterInstanceBytes,
svoBrickRasterPublicationInstanceOffsetBytes,
svoBrickRasterSortStateBytes,
svoRasterCoverageArenaBytes,
svoRasterCoverageCountAllocationBytes,
svoRasterCoverageOverflowArgsBindGroupLayoutEntries,
svoRasterCoverageOverflowDrawArgsOffsetBytes,
svoRasterCoverageOverflowStatus
} from "../features/primary-visibility/webgpu-svo-brick-raster";
import {
SparseVoxelGBufferTargetArena,
SVO_GBUFFER_RENDER_TARGET_CONTRACT,
type SparseVoxelGBufferTextures,
type SparseVoxelGBufferViews,
} from "../features/primary-visibility/webgpu-svo-gbuffer-targets";
import {
createSvoPrimaryEntryPrepassWGSL,
SVO_PRIMARY_ENTRY_PREPASS_CONTRACT,
svoPrimaryEntryCullBindGroupLayoutEntries,
svoPrimaryEntryDrawBindGroupLayoutEntries,
svoPrimaryEntryInstanceOffsetBytes,
svoPrimaryEntryPublicationBytes,
} from "../features/primary-visibility/webgpu-svo-primary-entry-prepass";
import {
SVO_RIGID_RASTER_CONTRACT,
svoRigidRasterCoverageBridgeBindGroupLayoutEntries,
svoRigidRasterInputBindGroupLayoutEntries,
svoRigidRasterShader,
} from "../features/primary-visibility/webgpu-svo-rigid-raster";
import { SVO_CONE_RADIANCE_RECONSTRUCTION_CODES } from "../features/radiance/definition";
import {
buildSvoEnvironmentLighting,
SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
SVO_ENVIRONMENT_LIGHTING_VERSION
} from "../features/radiance/svo-environment-lighting";
import { SVO_NODE_MIP_LAYOUT } from "../features/radiance/svo-node-mip-pyramid";
import {
packSvoFieldProgramArena,
SVO_FIELD_PROGRAM_BLOCK_WORDS,
unpackSvoFieldProgram,
type SvoFieldProgram
} from "../features/scene-publication/svo-field-program";
import {
packSvoPrimitiveCandidateArena,
SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES,
SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
type SvoPrimitiveCandidateArena,
type SvoPrimitiveCandidatePublication
} from "../features/scene-publication/svo-primitive-candidates";
import { SVO_PRIMITIVE_MOTION_STRIDE_BYTES } from "../features/scene-publication/svo-primitive-motion";
import {
createSvoScenePrimitiveBandWGSL,
packSvoScenePrimitiveBandParams,
readSvoScenePrimitiveCoverageAudit,
SVO_SCENE_PRIMITIVE_BAND_CONTRACT,
SVO_SCENE_PRIMITIVE_BAND_COUNTER_BYTES,
SVO_SCENE_PRIMITIVE_BAND_PARAMS_BYTES,
SVO_SCENE_PRIMITIVE_BAND_STATE_BYTES,
SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_BYTES,
svoScenePrimitiveBandComputeBindGroupLayoutEntries,
svoScenePrimitiveBandReadBindGroupLayoutEntries,
type SvoScenePrimitiveCoverageAudit
} from "../features/scene-publication/svo-scene-primitive-band";
import type { WebGpuSvoFluidCoverage } from "../features/scene-publication/webgpu-svo-fluid-coverage";
import { CLUSTER_BLOCK_COUNT_WORD,CLUSTER_BLOCK_FIELD_WORD,CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD,CLUSTER_BLOCK_LOBE_SPAN_WORD,CLUSTER_BLOCK_POINTS_WORD,CLUSTER_BLOCK_SEED_WORD,CLUSTER_BLOCK_SMOOTH_RADIUS_WORD,createSvoDryConeMarcherWGSL,createSvoDrySceneFragmentWGSL,drySceneShader,drySceneVertexShader,SVO_DRY_CONE_PREPASS_CONTRACT,SVO_DRY_NODE_MIP_PUBLICATION_MODE,SVO_DRY_SCENE_ARENA_LAYOUT,SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES,SVO_DRY_SCENE_CLUSTER_CAPACITY,SVO_DRY_SCENE_FIELD_PROGRAM_ARENA_SIZE_BYTES,SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY,SVO_DRY_SCENE_GLASS_ARENA_SIZE_BYTES,SVO_DRY_SCENE_MATERIAL_ARENA_SIZE_BYTES,SVO_DRY_SCENE_PIXEL_PROBE_GROUP,SVO_DRY_SCENE_REVERSED_Z_NEAR_M,SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT,SVO_DRY_VISIBILITY_FLAGS,SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT,SVO_DRY_WORLD_GI_CACHE_CONTRACT,SVO_SCENE_PRIMITIVE_COVERAGE_MAXIMUM_RECORDS,SVO_SCENE_PRIMITIVE_RASTER_CONTRACT,svoDryRasterGlassShader,svoVisibilityTraceOffset,type SvoBrickOccupancyMode,type SvoConeLightingScale,type SvoDryOptimizationExperiments,type SvoDryShadingPath,type SvoDryTraversalMode } from "../features/shading/program";
import { resolveSvoPipelineComposition } from "./composition";
import {
disabledRenderStagesEqual,
NO_DISABLED_RENDER_STAGES,
type DisabledRenderStages,
} from "./render-stage-switches";
import {
DEFAULT_SVO_LIGHTING_OPTIONS,
type SvoLightingOptions,
type SvoLightingVisibilityStatus,
type SvoSilhouetteRefinementStatus,
} from "./svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING,normalizeSvoRenderTuning,SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,type SvoLodMode,type SvoRenderTuning } from "./svo-render-tuning";

/** Lifecycle metadata lives with the sparse presentation programs it describes. */
export const svoPresentationResourcePlugin: ResourcePluginDefinition = Object.freeze({
  id: "presentation.svo-global",
  lane: "svo",
  label: "GLOBAL sparse voxel presentation",
  provides: ["sparse-voxel-presentation"] as const,
  blocks: "viewport",
  progressPhases: [{ id: "allocation", label: "Allocate" }, { id: "presentation", label: "Compile" }, { id: "warmup", label: "First frame" }],
  phaseCopy: {
    presentation: "Compiling and attaching the complete sparse presentation. Scene interaction resumes after its first fenced frame.",
    allocation: "Allocating sparse presentation targets while the last complete generation remains visible, when one exists.",
    warmup: "Publishing the first sparse frame behind a GPU completion fence.",
  },
});

/** Startup milestones for the sparse presentation plugin. Compilation,
 * attachment, and first presentation share one colocated task vocabulary so
 * the UI never has to invent a denominator. */
export const SVO_PRESENTATION_STARTUP_STAGES = Object.freeze([
  "Build sparse presentation shader sources",
  "Validate sparse presentation shader modules",
  "Compile sparse primary visibility pipeline",
  "Compile sparse brick culling programs",
  "Compile split visibility and lighting programs",
  "Compile raster glass and rigid discovery programs",
  "Compile sparse cone fan-out programs",
  "Finalize sparse presentation resources",
  "Attach sparse renderer",
  "Submit first sparse frame",
] as const);

/** `{vertexCount, instanceCount, firstVertex, firstInstance}`. */
const SVO_DRAW_INDIRECT_ARGS_BYTES = 16;
/** How often the arena-pressure tripwire samples; roughly once a second at 60 Hz. */
const SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_PERIOD_FRAMES = 60;

import {
mergeSvoPixelTrace,
withSvoPixelTraceConePrepass,
type SvoPixelTrace
} from "../features/diagnostics/svo-pixel-trace";
import {
SparseVoxelPixelTraceBuffers
} from "../features/diagnostics/webgpu-svo-pixel-trace";



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
  // Structure, scene-owner payload, authored scene arena, optional derived
  // traversal, and the accepted exact planar-terminal catalogue.
  ...[2, 3, 4, 5, 6].map((binding) => ({ binding, type: "read-only-storage" as const })),
  { binding: 9, type: "uniform" as const },
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
  // Live derived pages publish validity independently from their stable atlas
  // slots. Zero means dirty/unavailable and must be rejected before sampling.
  { binding: 26, type: "texture-2d-uint" as const },
  { binding: 27, type: "texture-2d-uint" as const },
] as const);

export function sparseVoxelDrySceneBindGroupLayoutEntries(
  traversalMode: SvoDryTraversalMode = "hybrid",
): GPUBindGroupLayoutEntry[] {
  // The compact 2x2 kernels need traversal, node-mip, lighting, and rigid-body
  // inputs, but not material shading, glass, or dormant traversal variants.
  // Keeping those fragment-only also stays below WebGPU's per-stage storage
  // binding limit on Apple GPUs.
  const computeBindings = new Set([0, 1, 2, 3, 4, 5, 6, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]);
  // Raster analytic impostors consume the camera/body uniforms, their scene
  // record arena, and the live primitive-count/structural-offset parameters.
  const vertexBindings = new Set([0, 1, 2, 3, 4, 9]);
  const usesDerivedTraversal = traversalMode === "compact" || traversalMode === "wide" || traversalMode === "hybrid";
  return SVO_DRY_SCENE_BINDING_CONTRACT
    .filter(({ binding }) => binding !== 5 || usesDerivedTraversal)
    .map(({ binding, type }): GPUBindGroupLayoutEntry => {
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
const CLUSTER_BLOCK_LATTICE_PERIOD_WORD = 5;
const CLUSTER_BLOCK_JITTER_WORD = 6;
const CLUSTER_BLOCK_ANISOTROPY_WORD = 7;
const CLUSTER_BLOCK_LOBE_SPAN_SPREAD_WORD = 9;
const CLUSTER_BLOCK_DISPLACEMENT_WORD = 10;

export function svoDrySceneClusterReference(index: number): number {
  if (!Number.isInteger(index) || index < 0 || index >= SVO_DRY_SCENE_CLUSTER_CAPACITY) {
    throw new RangeError(`Cluster index ${index} is outside the arena's ${SVO_DRY_SCENE_CLUSTER_CAPACITY}-block capacity`);
  }
  return SVO_DRY_SCENE_ARENA_LAYOUT.clusterOffsetBytes / Uint32Array.BYTES_PER_ELEMENT
    + index * SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS;
}

/**
 * Word offset of field-program tape block `index` in the scene arena.
 *
 * A word offset rather than a slot, for the reason the terrain and cluster
 * references are: a shader handed this number reads the block with no further
 * indirection. It is also what makes a stale reference *checkable* — the block
 * stride is fixed, so an offset that is not a block start is a corrupt word and
 * not a tape read from the middle of its neighbour.
 */
export function svoDrySceneFieldProgramReference(index: number): number {
  if (!Number.isInteger(index) || index < 0 || index >= SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY) {
    throw new RangeError(
      `Field-program index ${index} is outside the arena's ${SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY}-block capacity`,
    );
  }
  return SVO_DRY_SCENE_ARENA_LAYOUT.fieldProgramOffsetBytes / Uint32Array.BYTES_PER_ELEMENT
    + index * SVO_FIELD_PROGRAM_BLOCK_WORDS;
}

/**
 * Pack every tape into the arena's field-program region, in publication order,
 * so block `i` is what `svoDrySceneFieldProgramReference(i)` names.
 *
 * An unfilled block stays zeroed, and a zeroed block decodes as a tape with no
 * ops — which evaluates to the "not resolved" distance both the CPU resolver and
 * the shader reject. A record that names one therefore renders as nothing and
 * reports itself invalid, rather than as the smooth source solid a plausible
 * default would produce.
 */
export function packSvoDrySceneFieldPrograms(
  programs: readonly SvoFieldProgram[],
): Uint32Array<ArrayBuffer> {
  if (programs.length > SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY) {
    throw new RangeError(
      `Scene publishes ${programs.length} field programs, above the arena's ${SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY}`,
    );
  }
  return packSvoFieldProgramArena(programs, SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY);
}

/**
 * CPU mirror of the shader's tape decode, for the analytic oracles and captures.
 *
 * The same rejection discipline as the cluster resolver beside it, and for the
 * same reason: an oracle that resolves a tape the GPU treats as unresolved
 * reports a surface the frame does not contain, which is the one failure a
 * parity test cannot tell apart from a shading bug.
 */
export function svoDrySceneFieldProgramResolver(packed: Uint32Array | undefined): SvoFieldProgramResolver {
  const base = SVO_DRY_SCENE_ARENA_LAYOUT.fieldProgramOffsetBytes / Uint32Array.BYTES_PER_ELEMENT;
  return (fieldProgramReference: number) => {
    if (!packed) return undefined;
    const offset = fieldProgramReference - base;
    if (offset < 0 || offset % SVO_FIELD_PROGRAM_BLOCK_WORDS !== 0) return undefined;
    if (offset + SVO_FIELD_PROGRAM_BLOCK_WORDS > packed.length) return undefined;
    // A never-filled block claims zero ops, and a tape with no ops has no result
    // register anything wrote. Unpacking it would produce a program the
    // validator refuses, so it is refused here instead — the "not resolved"
    // answer, which is what the shader's own probe reports for the same block.
    if (packed[offset] === 0) return undefined;
    try {
      return unpackSvoFieldProgram(packed, offset / SVO_FIELD_PROGRAM_BLOCK_WORDS);
    } catch {
      return undefined;
    }
  };
}

/**
 * Pack every cluster's packing into the arena's cluster region, in publication
 * order, so block `i` is what `svoDrySceneClusterReference(i)` names.
 *
 * An unfilled block stays zeroed, and a zeroed block reads as a lattice with a
 * zero lobe radius — the "not resolved" encoding both the CPU resolver and the
 * shader reject. A cluster record that names one therefore renders as nothing
 * and reports itself invalid, rather than as the smooth ellipsoid a plausible
 * default would produce.
 */
export function packSvoDrySceneClusters(
  packings: readonly SvoSmoothUnionClusterPacking[],
): Uint32Array<ArrayBuffer> {
  if (packings.length > SVO_DRY_SCENE_CLUSTER_CAPACITY) {
    throw new RangeError(`Scene publishes ${packings.length} clusters, above the arena's ${SVO_DRY_SCENE_CLUSTER_CAPACITY}`);
  }
  const buffer = new ArrayBuffer(SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  packings.forEach((packing, index) => {
    const base = index * SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS;
    const field = svoClusterFieldName(packing);
    // The header every field shares, and the one word that says which field
    // the rest of the block is. `count` is deliberately one slot for three
    // meanings: it is the only per-field cardinal, and giving each its own
    // word would leave two of them zero in every block ever published.
    words[base + CLUSTER_BLOCK_FIELD_WORD] = SVO_CLUSTER_FIELD_TABLE[field].code;
    words[base + CLUSTER_BLOCK_SEED_WORD] = packing.seed >>> 0;
    floats[base + CLUSTER_BLOCK_SMOOTH_RADIUS_WORD] = packing.smoothRadius_m;
    if (packing.field === "noise-foliage") {
      words[base + CLUSTER_BLOCK_COUNT_WORD] = 2;
      floats[base + CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD] = packing.detailPeriod_m;
      floats[base + CLUSTER_BLOCK_LATTICE_PERIOD_WORD] = packing.clusterPeriod_m;
      floats[base + CLUSTER_BLOCK_JITTER_WORD] = packing.threshold;
      floats[base + CLUSTER_BLOCK_ANISOTROPY_WORD] = packing.clusterWeight;
      floats[base + CLUSTER_BLOCK_LOBE_SPAN_WORD] = packing.detailWeight;
      floats[base + CLUSTER_BLOCK_LOBE_SPAN_SPREAD_WORD] = packing.interiorBias;
      return;
    }
    if (packing.field === "seeded-lobes") {
      words[base + CLUSTER_BLOCK_COUNT_WORD] = packing.lobeCount >>> 0;
      floats[base + CLUSTER_BLOCK_ANISOTROPY_WORD] = packing.anisotropy;
      // Defaulted here rather than in the shader. The block is the ABI, and a
      // shader that substituted its own default for an absent parameter would
      // be a second place the default lives — which is the arrangement this
      // whole file exists to avoid.
      floats[base + CLUSTER_BLOCK_LOBE_SPAN_WORD] = packing.lobeSpan ?? SVO_CLUSTER_LOBE_DEFAULT_SPAN;
      floats[base + CLUSTER_BLOCK_LOBE_SPAN_SPREAD_WORD] = packing.lobeSpanSpread ?? SVO_CLUSTER_LOBE_DEFAULT_SPAN_SPREAD;
      floats[base + CLUSTER_BLOCK_DISPLACEMENT_WORD] = packing.displacement ?? SVO_CLUSTER_LOBE_DEFAULT_DISPLACEMENT;
      return;
    }
    if (packing.field === "tapered-sweep") {
      words[base + CLUSTER_BLOCK_COUNT_WORD] = packing.points.length >>> 0;
      packing.points.forEach((point, pointIndex) => {
        const pointBase = base + CLUSTER_BLOCK_POINTS_WORD + pointIndex * 4;
        floats.set([point.position_m.x, point.position_m.y, point.position_m.z, point.radius_m], pointBase);
      });
      return;
    }
    words[base + CLUSTER_BLOCK_COUNT_WORD] = (packing.octaves ?? 1) >>> 0;
    floats[base + CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD] = packing.latticeLobeRadius_m;
    floats[base + CLUSTER_BLOCK_LATTICE_PERIOD_WORD] = packing.latticePeriod_m;
    floats[base + CLUSTER_BLOCK_JITTER_WORD] = packing.jitter;
  });
  return words;
}

/**
 * Say so, loudly, when a scene publishes an aggregate whose block never arrived.
 *
 * An aggregate that cannot resolve its packing draws *nothing* — the shader
 * rejects it as invalid rather than falling back to its lobe, which is the
 * right call because a smooth ellipsoid is a shape the scene never asked for.
 * The failure is therefore silent by construction, and it is one assembly site
 * away at all times: `SparseVoxelDrySceneData` is built by hand in the renderer
 * and again in the headless harness, and the first headless frame after this
 * kind landed drew every aggregate in the scene as nothing at all because
 * one of them had not been taught the field.
 *
 * A console warning is the right shape for it because the GPU smoke lane
 * already fails on any `[svo]` warning, so a forgotten upload fails CI rather
 * than being noticed by eye in a frame nobody happened to look at.
 */
function warnOnUnresolvedClusters(scene: SparseVoxelDrySceneData): void {
  const records = scene.primitiveRecords;
  if (!records?.length) return;
  const resolve = svoDrySceneClusterResolver(scene.clusterBlocks);
  let unresolved = 0;
  let clusters = 0;
  for (let base = 0; base + SVO_PRIMITIVE_RECORD_WORDS <= records.length; base += SVO_PRIMITIVE_RECORD_WORDS) {
    if (records[base + 3] !== SVO_PRIMITIVE_KINDS.smoothUnionCluster) continue;
    clusters += 1;
    if (!resolve(records[base + 13])) unresolved += 1;
  }
  if (unresolved === 0) return;
  // The two causes read very differently at the call site and the fix is not
  // the same: an assembly site that never carried `clusterBlocks` at all is a
  // forgotten field on a hand-built `SparseVoxelDrySceneData`, while a block
  // that arrived and was rejected is a field the packer and the resolver
  // disagree about — a layout drift rather than a plumbing one.
  const cause = scene.clusterBlocks
    ? "their blocks arrived and were rejected: check the field code and the per-field validity rule"
    : "this scene published no cluster blocks at all: check that the assembly site carries `clusterBlocks`";
  console.warn(`[svo] ${unresolved} of ${clusters} aggregate records name an unresolved cluster block; ${cause}. They will draw as nothing`);
}

/**
 * CPU mirror of the shader's block decode, for the analytic oracles and captures.
 *
 * Every rejection here has a shader counterpart in `svoClusterPackingValid`,
 * and both must reject the same blocks: an oracle that resolves a packing the
 * GPU treats as invalid reports a surface the frame does not contain, which is
 * the one failure a parity test cannot tell apart from a shading bug.
 */
export function svoDrySceneClusterResolver(packed: Uint32Array | undefined): SvoClusterResolver {
  const clusterBase = SVO_DRY_SCENE_ARENA_LAYOUT.clusterOffsetBytes / Uint32Array.BYTES_PER_ELEMENT;
  return (clusterReference: number) => {
    if (!packed) return undefined;
    const offset = clusterReference - clusterBase;
    // The slot stride is what makes this check possible at all: an offset that
    // is not a block start is a stale or corrupt reference, not a packing read
    // from the middle of its neighbour.
    if (offset < 0 || offset % SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS !== 0) return undefined;
    if (offset + SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS > packed.length) return undefined;
    const floats = new Float32Array(packed.buffer, packed.byteOffset, packed.length);
    const entry = svoClusterFieldByCode(packed[offset + CLUSTER_BLOCK_FIELD_WORD]);
    if (!entry) return undefined;
    const seed = packed[offset + CLUSTER_BLOCK_SEED_WORD] >>> 0;
    const count = packed[offset + CLUSTER_BLOCK_COUNT_WORD];
    const smoothRadius_m = floats[offset + CLUSTER_BLOCK_SMOOTH_RADIUS_WORD];
    if (entry.name === "noise-foliage") {
      const detailPeriod_m = floats[offset + CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD];
      const clusterPeriod_m = floats[offset + CLUSTER_BLOCK_LATTICE_PERIOD_WORD];
      const clusterWeight = floats[offset + CLUSTER_BLOCK_ANISOTROPY_WORD];
      const detailWeight = floats[offset + CLUSTER_BLOCK_LOBE_SPAN_WORD];
      if (!(detailPeriod_m > 0) || !(clusterPeriod_m > 0) || !(clusterWeight + detailWeight > 0)) return undefined;
      return {
        field: "noise-foliage", seed, smoothRadius_m, detailPeriod_m, clusterPeriod_m,
        threshold: floats[offset + CLUSTER_BLOCK_JITTER_WORD], clusterWeight, detailWeight,
        interiorBias: floats[offset + CLUSTER_BLOCK_LOBE_SPAN_SPREAD_WORD],
      };
    }
    if (entry.name === "seeded-lobes") {
      const anisotropy = floats[offset + CLUSTER_BLOCK_ANISOTROPY_WORD];
      const lobeSpan = floats[offset + CLUSTER_BLOCK_LOBE_SPAN_WORD];
      if (!(count >= 1) || !(anisotropy >= 1) || !(lobeSpan > 0)) return undefined;
      return {
        field: "seeded-lobes", seed, smoothRadius_m, lobeCount: count, anisotropy, lobeSpan,
        lobeSpanSpread: floats[offset + CLUSTER_BLOCK_LOBE_SPAN_SPREAD_WORD],
        displacement: floats[offset + CLUSTER_BLOCK_DISPLACEMENT_WORD],
      };
    }
    if (entry.name === "tapered-sweep") {
      if (count < 2 || count > SVO_CLUSTER_SWEEP_MAXIMUM_POINTS) return undefined;
      const points = Array.from({ length: count }, (_unused, index) => {
        const base = offset + CLUSTER_BLOCK_POINTS_WORD + index * 4;
        return {
          position_m: { x: floats[base], y: floats[base + 1], z: floats[base + 2] },
          radius_m: floats[base + 3],
        };
      });
      if (!points.every((point) => point.radius_m > 0)) return undefined;
      return { field: "tapered-sweep", seed, smoothRadius_m, points };
    }
    // The zeroed block lands here — field code 0 is the lattice — and is
    // rejected by its own radii, which is what the "never filled" case has
    // always relied on.
    const latticeLobeRadius_m = floats[offset + CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD];
    const latticePeriod_m = floats[offset + CLUSTER_BLOCK_LATTICE_PERIOD_WORD];
    if (!(latticeLobeRadius_m > 0) || !(latticePeriod_m > 0) || !(count >= 1)) return undefined;
    return {
      latticeLobeRadius_m,
      latticePeriod_m,
      jitter: floats[offset + CLUSTER_BLOCK_JITTER_WORD],
      smoothRadius_m,
      seed,
      octaves: count,
    };
  };
}

/**
 * Liquid coverage along the continuation ray above which an opaque voxel is
 * drawn see-through. Coverage is a fraction per coarse texel, so the water's
 * own boundary texel reads about 0.5 and a texel one step into the vessel wall
 * reads well under this: the ghost begins at the water, not half a texel out.
 */
export const SVO_OCCLUDER_GHOST_COVERAGE_THRESHOLD = 0.3;

/** Packed dry-scene parameters. */
export const SVO_DRY_SCENE_PARAMS_LAYOUT = Object.freeze({
  sizeBytes: 688,
  meshFilterWordOffset: 160,
  /** enabled, ghost opacity, coverage threshold, reserved. */
  occluderGhostWordOffset: 168,
  glassWordOffset: 24,
  /** count, generation, stride bytes, reserved for accepted planar terminals. */
  planarBoundaryWordOffset: 28,
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
  /**
   * x: matching radiance generation; y: complete and usable;
   * z: finest level with a radiance page; w: the slot its atlas begins at.
   */
  tetrahedralRadianceWordOffset: 112,
  /** xyz: complete sparse-lighting world extent in metres. */
  nodeMipExtentWordOffset: 116,
  /** xyzw: GI bounce, broad occlusion, diffuse environment, direct key. */
  giLightingWordOffset: 120,
  /** xy: GI aperture and cone count; zw reserved. */
  giConesWordOffset: 124,
  /**
   * xyz: centre of one sphere enclosing every rigid body; w: its radius, or a
   * negative radius when the scene has no bodies at all. Shadow and contact
   * rays test this before they touch the body array, which is why it is worth a
   * uniform lane of its own.
   */
  rigidBoundsWordOffset: 128,
  /** primitive offset, BVH node count, root node, complete render revision. */
  primitiveCandidatesWordOffset: 132,
  /** u32 word offsets of control, publication, nodes, and leaves in the structural arena. */
  structureOffsetsWordOffset: 136,
  /** Optional derived traversal word offsets (wide pages, wide descriptors). */
  derivedTraversalWordOffset: 140,
  /**
   * Live level-of-detail controls, deliberately a uniform rather than a shader
   * constant: the panel's mode switch and both sliders must move without a
   * pipeline rebuild, and an A/B over the threshold has to be interleavable.
   *
   * x: bitcast f32 screen-space threshold at the contract's reference height;
   * y: `SVO_LOD_MODE_*`; z: fixed level; w: filtered voxel-mesh detail
   * threshold in reference pixels, zero for the exact boundary mesh.
   *
   * `w` used to carry the surface-reconstruction arm. There is one arm now — the
   * normal is baked into the voxel — so nothing selects between them.
   */
  lodWordOffset: 144,
  /**
   * Where scene identity lives inside the payload arena bound at binding 3.
   *
   * xyzw: occupancy mask, record mask, per-leaf header, blob arena — the four
   * banded lane bases, as u32 word offsets, exactly as
   * `tree.bandedLaneWordOffsets` publishes them to the voxeliser. Zero on a
   * `dense` world, where the codec that would read them is not compiled at all.
   *
   * A uniform rather than a shader constant because the offsets are a property of
   * the *arena*, and one renderer outlives several: a world rebuilt at a
   * different capacity moves every lane, and a baked constant would then address
   * the previous one — a plausible read in the wrong lane, which is a frame with
   * holes rather than a validation failure.
   */
  payloadLaneWordOffset: 148,
  /**
   * x: dense scene-geometry lane base; y: the flat owner lane's base, read by
   * the `dense` arm; z: voxel capacity. w packs the payload mode in bits 0..7,
   * geometry stride words in 8..15, fraction word in 16..23, and packed-format
   * presence in bit 24.
   */
  payloadLane1WordOffset: 152,
  /**
   * x: is the primary entry-seed plane live this frame?
   *
   * A uniform rather than a shader constant for the same reason `lod` is one:
   * the RENDER panel's `primary-entry-prepass` switch has to withhold the pass
   * without rebuilding a pipeline, and an A/B over it has to be interleavable.
   * It cannot be inferred from the plane itself — an unwritten texel *means*
   * "no voxel leaf on this ray", which is a proof of absence the prepass earns
   * by drawing every leaf, so a stale or cleared plane would read as a legal
   * answer and resolve the frame to sky. Zero here is the only signal that
   * says "unknown": the fragment then descends from the root, exactly as a
   * build compiled without the prepass does.
   *
   * yzw reserved.
   */
  primaryEntryWordOffset: 156,
} as const);

/** Low-byte `dry.payloadLanes1.w` codes. Mirrors `SparseBrickLeafPayloadMode`. */
export const SVO_DRY_LEAF_PAYLOAD_MODES = Object.freeze({
  dense: 0, occupancy: 1, banded: 2,
} satisfies Record<SparseBrickLeafPayloadMode, number>);

/** `dry.lod.y` codes. Mirrors `SvoLodMode`; the shader constants are in `svoLodDescentWGSL`. */
export const SVO_LOD_MODES = Object.freeze({
  "screen-space": 0,
  "fixed-level": 1,
} satisfies Record<SvoLodMode, number>);
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
export const SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT = Object.freeze({
  metadataWordOffset: 0,
  lightWordOffset: 4,
  environmentWordOffset: 4 + SVO_LIGHT_MAXIMUM_RECORDS * SVO_LIGHT_RECORD_WORDS,
  sizeBytes: 16 + SVO_LIGHT_MAXIMUM_RECORDS * SVO_LIGHT_RECORD_STRIDE_BYTES + SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
} as const);
if (svoVisibilityTraceOffset < 0) throw new Error("SVO visibility WGSL trace marker is missing");

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
  scene: SparseVoxelDrySceneData | undefined,
): boolean {
  if (!scene?.lightRecords || !scene.environmentLightingRecord
    || !Number.isSafeInteger(scene.lightRevision) || scene.lightRevision! < 1 || scene.lightRevision! > 0xffff_ffff
    || scene.lightRecords.byteLength < SVO_LIGHT_RECORD_STRIDE_BYTES
    || scene.lightRecords.byteLength > SVO_LIGHT_MAXIMUM_RECORDS * SVO_LIGHT_RECORD_STRIDE_BYTES
    || scene.lightRecords.byteLength % SVO_LIGHT_RECORD_STRIDE_BYTES !== 0
    || scene.environmentLightingRecord.byteLength !== SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES
    || !scene.environmentLightingCacheKey) return false;
  const lightWords = scene.lightRecords;
  const lightIds = new Set<number>();
  const lightCount = lightWords.byteLength / SVO_LIGHT_RECORD_STRIDE_BYTES;
  for (let index = 0; index < lightCount; index += 1) {
    const identity = index * SVO_LIGHT_RECORD_WORDS + 24;
    const kind = lightWords[identity], lightId = lightWords[identity + 1], revision = lightWords[identity + 3];
    if (!SVO_LIGHT_KIND_CODES.has(kind) || lightId === 0 || lightIds.has(lightId) || revision !== scene.lightRevision) return false;
    lightIds.add(lightId);
  }
  const environmentWords = scene.environmentLightingRecord;
  return environmentWords[21] === scene.lightRevision && environmentWords[22] === SVO_ENVIRONMENT_LIGHTING_VERSION;
}

/** Build one renderer-owned live lighting publication from canonical scene data. */
export function buildSparseVoxelDrySceneLightingMirrors(
  scene: SceneDescription,
  revision: number,
): Pick<SparseVoxelDrySceneData, "lightRecords" | "lightRevision" | "environmentLightingRecord" | "environmentLightingCacheKey"> | undefined {
  if (!Number.isSafeInteger(revision) || revision < 1 || revision > 0xffff_ffff) return undefined;
  try {
    const sceneLights = buildSvoSceneLights(scene, { revision, maximumRecords: SVO_LIGHT_MAXIMUM_RECORDS });
    const environmentLighting = buildSvoEnvironmentLighting(
      scene.environment ?? "default", revision, svoSceneLighting(scene)?.environment);
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

/** Pack validated CPU mirrors into one uniform arena outside the four-storage-buffer contract. */
export function packSparseVoxelDrySceneLightingArena(
  scene: SparseVoxelDrySceneData | undefined,
): Uint32Array<ArrayBuffer> | undefined {
  if (!canConsumeSparseVoxelLighting(scene)) return undefined;
  const lightCount = scene!.lightRecords!.byteLength / SVO_LIGHT_RECORD_STRIDE_BYTES;
  const packed = new Uint32Array(new ArrayBuffer(SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.sizeBytes));
  packed.set([lightCount, scene!.lightRevision!, scene!.lightRevision!, SVO_ENVIRONMENT_LIGHTING_VERSION], SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.metadataWordOffset);
  packed.set(scene!.lightRecords!, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.lightWordOffset);
  packed.set(scene!.environmentLightingRecord!, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.environmentWordOffset);
  return packed;
}

export function canConsumeSparseVoxelPrimitiveCandidates(scene: SparseVoxelDrySceneData | undefined): boolean {
  const primitiveCount = scene?.primitiveRecords.byteLength
    ? scene.primitiveRecords.byteLength / SVO_PRIMITIVE_RECORD_STRIDE_BYTES
    : 0;
  if (primitiveCount > SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES) return false;
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
  return sparseVoxelDrySceneContractFailure(source, scene) === undefined;
}

/** Exact fail-closed reason for the live sparse presentation contract. */
export function sparseVoxelDrySceneContractFailure(
  source: SparseVoxelSceneRenderSource | undefined,
  scene: SparseVoxelDrySceneData | undefined,
): string | undefined {
  if (!source) return "live sparse source is not attached";
  if (!source.structural) return "structural arena is not attached";
  if (!scene) return "scene arena publication is not attached";
  if (!Number.isSafeInteger(scene.renderRevision) || scene.renderRevision <= 0) return "scene revision is invalid";
  if (!Number.isSafeInteger(scene.materialRevision) || scene.materialRevision <= 0) return "material revision is invalid";
  if (scene.materialRecords.byteLength < 2 * SVO_MATERIAL_RECORD_STRIDE_BYTES) return "material arena is incomplete";
  if (scene.materialRecords.byteLength % SVO_MATERIAL_RECORD_STRIDE_BYTES !== 0) return "material arena stride is invalid";
  if (scene.materialRecords.byteLength > SVO_DRY_SCENE_MATERIAL_ARENA_SIZE_BYTES) return "material arena capacity is exceeded";
  if (!canConsumeSparseVoxelPrimitiveCandidates(scene)) return "primitive candidate arena is invalid";
  if (!canConsumeSparseVoxelLighting(scene)) return "lighting publication is invalid";
  if (scene.primitiveRecords.byteLength % SVO_PRIMITIVE_RECORD_STRIDE_BYTES !== 0) return "scene primitive arena stride is invalid";
  const glassBytes = scene.glassRecords?.byteLength ?? 0;
  if (glassBytes % SVO_THIN_GLASS_RECORD_STRIDE_BYTES !== 0) return "thin-glass arena stride is invalid";
  if (glassBytes / SVO_THIN_GLASS_RECORD_STRIDE_BYTES > SVO_SCENE_GLASS_MAXIMUM_PANES) return "thin-glass arena capacity is exceeded";
  if (source.structural.fields.topology.residency === "unavailable") return "topology field is unavailable";
  if (source.structural.fields.sceneGeometry.residency === "unavailable") return "scene geometry field is unavailable";
  if (source.structural.fields.materialOwner.residency === "unavailable") return "material-owner payload field is unavailable";
  const planar = source.structural.planarBoundaries;
  if (!Number.isSafeInteger(planar.count) || planar.count < 0
    || planar.count * PLANAR_BOUNDARY_PATCH_BYTES > (planar.records.size ?? planar.records.buffer.size)) {
    return "planar boundary catalogue count is invalid";
  }
  if (planar.strideBytes !== PLANAR_BOUNDARY_PATCH_BYTES) return "planar boundary catalogue stride is invalid";
  if (!Number.isSafeInteger(planar.generation) || planar.generation <= 0) return "planar boundary catalogue generation is invalid";
  return undefined;
}

export type SvoDryPresentationBundleStatus =
  | { readonly state: "ready" }
  | { readonly state: "compiling"; readonly detail: string }
  | { readonly state: "failed"; readonly detail: string };

/**
 * Readback ABI for {@link SvoDryOptimizationExperiments.primaryLeafVisitHistogram}.
 *
 * Two exact histograms — every pixel, and only the pixels that resolved a voxel
 * surface — indexed by leaf-visit count, so percentiles are computed on the host
 * without the shader choosing buckets. `SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT` is
 * the compiled ceiling of the loop, so a bucket per count covers it exactly.
 */
export const SVO_DRY_PRIMARY_VISIT_HISTOGRAM_CONTRACT = Object.freeze({
  buckets: SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT + 1,
  allPixelsOffset: 0,
  hitPixelsOffset: SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT + 1,
  terminalOffset: 2 * (SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT + 1),
  terminalCount: 8,
  totalPixelsWord: 2 * (SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT + 1) + 8,
  visitSumWord: 2 * (SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT + 1) + 9,
  words: 2 * (SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT + 1) + 10,
  /** Binding added to the split visibility group only while the diagnostic is on. */
  binding: 7,
} as const);

/** How a primary ray left the leaf loop, in the order the terminal words are packed. */
export const SVO_DRY_PRIMARY_VISIT_TERMINALS = Object.freeze([
  "voxel-hit", "tree-miss", "leaf-budget-exhausted", "node-work-exhausted",
  "stack-overflow", "source-overflow", "invalid-topology", "no-traversal",
] as const);

/** Exact float normal + primary hit distance crossing the split pass boundary. */
export const SVO_DRY_SPLIT_GEOMETRY_FORMAT = "rgba32float" as GPUTextureFormat;
export const SVO_DRY_SPLIT_IDENTITY_FORMAT = "rg32uint" as GPUTextureFormat;
/** 24-byte write plus 24-byte read across the pass boundary. */
export const SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL = 48;
/** Resident exact-primary cache: rgba32float geometry + rg32uint identity. */
export const SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL = 24;

/** Compact readback ABI. Counters are reset at the start of every encoded cone frame. */
export const SVO_DRY_DERIVED_FAILURE_COUNTERS = Object.freeze({
  ambientOcclusionPageWord: 0,
  directVisibilityPageWord: 1,
  globalIlluminationPageWord: 2,
  wordCount: 3,
  sizeBytes: 3 * Uint32Array.BYTES_PER_ELEMENT,
} as const);

export interface SvoDrySceneDirtyBounds {
  readonly minimum: readonly [number, number, number];
  readonly maximum: readonly [number, number, number];
}

/**
 * Dependency declaration for a hot analytic-arena publication.
 *
 * Analytic transforms do not mutate the node-mip/radiance/light generations
 * that persistent cone caches store, so their exact dependency action is to
 * retain those caches. Bounds stay attached to the publication for the
 * unified sparse-page updater; no per-pixel AABB scan is introduced here.
 */
export interface SvoDryPrimitiveArenaChange {
  readonly dirtyBounds: readonly SvoDrySceneDirtyBounds[];
  readonly derivedLighting: "unchanged" | "global";
  /** Present for animation: only these fixed-stride records are uploaded. */
  readonly dirtyPrimitiveIndices?: readonly number[];
  /** Candidate-node ancestor closure updated by the incremental refit. */
  readonly dirtyCandidateNodeIndices?: readonly number[];
}

export function svoDryPrimitiveArenaCacheInvalidation(change: SvoDryPrimitiveArenaChange): {
  readonly worldGi: boolean;
  readonly directionalVisibility: boolean;
} {
  const global = change.derivedLighting === "global";
  return { worldGi: global, directionalVisibility: global };
}

/** Prepass target dimensions derived from the presentation size, never below 1x1. */
export function svoConePrepassSize(width: number, height: number, scale: SvoConeLightingScale): readonly [number, number] {
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

/** Optional reduced-rate cone lighting layered over the shared user-facing options. */
export type SparseVoxelDrySceneLightingOptions = SvoLightingOptions & {
  readonly coneLightingScale?: SvoConeLightingScale;
};

async function checkedModule(device: GPUDevice, label: string, code: string): Promise<GPUShaderModule> {
  const shaderModule = device.createShaderModule({ label, code });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(`${label}:\n${errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n")}`);
  return shaderModule;
}

/** One sphere enclosing every rigid body, in world metres. */
export interface SvoDryRigidBounds {
  readonly centre_m: readonly [number, number, number];
  readonly radius_m: number;
}

/** Four depth-tested primary planes; the raster-primary passes share them. */
const rasterPrimaryTargets: GPUColorTargetState[] = [
  { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.packedSurfaceFormat },
  { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.identityMediaFormat },
  { format: SVO_DRY_SPLIT_GEOMETRY_FORMAT },
  { format: SVO_DRY_SPLIT_IDENTITY_FORMAT },
];

/** Below this pixel count, Metal's TBDR fragment resolve remains faster. */
export const SVO_SCENE_PRIMITIVE_COMPUTE_MINIMUM_PIXELS = 1 << 20;

interface SvoDrySplitPipelineBundle {
  readonly visibility: GPURenderPipeline;
  readonly rasterRigidVisibility?: GPURenderPipeline;
  /** Optional one-pixel primary coverage closure before the sky/surface partition. */
  readonly primarySeamClosure: GPURenderPipeline;
  readonly lighting: GPURenderPipeline;
  readonly optimizedLighting?: GPURenderPipeline;
  readonly reconstructedLighting?: GPURenderPipeline;
  /** Complement of `lighting`: the pixels primary visibility left as a miss. */
  readonly skyLighting: GPURenderPipeline;
  readonly surfaceMesh?: {
    prepare: GPUComputePipeline; boxes: GPUComputePipeline; mark: GPUComputePipeline; schedule: GPUComputePipeline;
    count: GPUComputePipeline; allocate: GPUComputePipeline; emit: GPUComputePipeline; publish: GPUComputePipeline;
    select: GPUComputePipeline; cull: GPUComputePipeline; draw: GPURenderPipeline; background: GPURenderPipeline;
  };
  readonly brickBackground?: GPURenderPipeline;
  readonly brickRaster?: GPURenderPipeline;
  readonly brickCoverage?: GPURenderPipeline;
  readonly brickCoverageResolve?: GPURenderPipeline;
  readonly brickLodResolve?: GPURenderPipeline;
  readonly brickExactResolve?: GPURenderPipeline;
  readonly brickCoverageOverflow?: GPURenderPipeline;
  readonly scenePrimitiveRaster?: GPURenderPipeline;
  readonly scenePrimitiveCoverage?: GPURenderPipeline;
  readonly scenePrimitiveLodResolve?: GPURenderPipeline;
  readonly scenePrimitiveComputeArgs?: GPUComputePipeline;
  readonly scenePrimitiveComputeResolve?: GPUComputePipeline;
  readonly scenePrimitiveDepthBridge?: GPURenderPipeline;
  readonly scenePrimitiveCoverageResolve?: GPURenderPipeline;
  readonly scenePrimitiveCoverageOverflow?: GPURenderPipeline;
  readonly prepassReset?: GPUComputePipeline;
  readonly prepassCoherent?: GPUComputePipeline;
  readonly prepassBoundary?: GPUComputePipeline;
  readonly worldGiFrame?: GPUComputePipeline;
  readonly worldGiCache?: GPUComputePipeline;
  readonly voxelLightDemand?: GPUComputePipeline;
  readonly voxelLightPopulate?: GPUComputePipeline;
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
  private primarySeamClosurePipeline?: GPURenderPipeline;
  private splitOptimizedLightingPipeline?: GPURenderPipeline;
  private splitLightingPipeline?: GPURenderPipeline;
  private splitReconstructedLightingPipeline?: GPURenderPipeline;
  private splitSkyLightingPipeline?: GPURenderPipeline;
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
  /** Frame-graph stages this frame must not encode. See `render-stage-switches`. */
  private disabledStages: DisabledRenderStages = NO_DISABLED_RENDER_STAGES;
  private voxelLightDemandPipeline?: GPUComputePipeline;
  private voxelLightPopulatePipeline?: GPUComputePipeline;
  private voxelLightConsumerLayout?: GPUBindGroupLayout;
  private voxelLightDemandLayout?: GPUBindGroupLayout;
  private voxelLightPopulateLayout?: GPUBindGroupLayout;
  private voxelLightConsumerBindGroup?: GPUBindGroup;
  private voxelLightDemandBindGroup?: GPUBindGroup;
  private voxelLightPopulateBindGroup?: GPUBindGroup;
  private voxelLightTexture?: GPUTexture;
  private voxelLightTextureView?: GPUTextureView;
  private voxelLightParamsBuffer?: GPUBuffer;
  private voxelLightRequestBuffer?: GPUBuffer;
  private voxelLightQueueBuffer?: GPUBuffer;
  private voxelLightPageBuffer?: GPUBuffer;
  private voxelLightPageCount = 0;
  private voxelLightEpoch = 1;
  private voxelLightActive = false;
  private voxelLightExclusive = false;
  private voxelLightUserEnabled = true;
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
  /**
   * Keyed by scale AND global-illumination capability: the GI-off kernel is a
   * feature-specialised variant with the gather compiled out, not a uniform
   * branch. A GI flip keeps rendering the stale-but-correct variant until the
   * specialised one activates, exactly like a scale change keeps its bundle.
   */
  private readonly splitPipelineBundles = new Map<string, SvoDrySplitPipelineBundle>();
  private readonly splitPipelineCompiles = new Map<string, Promise<SvoDrySplitPipelineBundle>>();
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
  private primaryWorkMap?: GPUTexture;
  private primaryWorkMapView?: GPUTextureView;
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
  private rasterGlassPaneCount = 0;
  private rasterRigidActive: boolean;
  /** Raster-assisted primary visibility (traversal mode `raster-primary`). */
  private surfaceMeshDisposed = false;
  private surfaceMeshMaximumBytes = 0;
  private surfaceMeshReadback?: GPUBuffer;
  private surfaceMeshReadbackPending = false;
  private surfaceMeshReadbackCopied = false;
  /** Arena sizes bound when the pending receipt's copy was encoded: what its counters were measured against. */
  private surfaceMeshReceiptArenaBytes: [number, number] = [0, 0];
  /** Consecutive presentations the current build has spent pending; paces the brick ramp. */
  private surfaceMeshBuildPresentations = 0;
  surfaceMeshStatus?: SvoSurfaceMeshStatus;
  private surfaceMeshDispatch?: GPUBuffer;
  private surfaceMeshState?: GPUBuffer;
  /** The two quad arenas; the GPU state says which one is drawn. */
  private surfaceMeshArenas: [GPUBuffer | undefined, GPUBuffer | undefined] = [undefined, undefined];
  /** Bound in an arena slot that holds no arena, so the shader always has two. */
  private surfaceMeshEmptyArena?: GPUBuffer;
  private surfaceMeshVisible?: GPUBuffer;
  /** Dirty boxes, per-leaf quad ranges and the brick worklist; sized by the source's leaf capacity. */
  private surfaceMeshWork?: GPUBuffer;
  private surfaceMeshWorkLeafCapacity = 0;
  /** Bound as the maintenance dirty list when the source has none. */
  private surfaceMeshEmptyMaintenance?: GPUBuffer;
  private surfaceMeshMaintenanceBuffer?: GPUBuffer;
  /** The arena slot the host last saw drawn; a receipt showing the other slot is a flip. */
  private surfaceMeshHostFront: 0 | 1 = 0;
  /** Generation stamped into the state when a back arena is bound; the GPU consumes it at a flip. */
  private surfaceMeshBackGeneration = 0;
  private surfaceMeshBackPending = false;
  private surfaceMeshComputeLayout?: GPUBindGroupLayout;
  private surfaceMeshDrawLayout?: GPUBindGroupLayout;
  private surfaceMeshComputeGroup?: GPUBindGroup;
  private surfaceMeshDrawGroup?: GPUBindGroup;
  private surfaceMeshPipelines?: SvoDrySplitPipelineBundle["surfaceMesh"];
  private readonly rasterPrimary: boolean;
  /**
   * Rasterized conservative entry depth in front of the primary megakernel.
   *
   * Everything below is withdrawn together when it is off — pipelines, buffers,
   * texture, bind-group entry and the fragment's binding — so the disabled arm
   * is the program that existed before the pass, not that program plus a
   * never-taken branch.
   */
  private readonly primaryEntryPrepassEnabled: boolean;
  private primaryEntryCullLayout?: GPUBindGroupLayout;
  private primaryEntryDrawLayout?: GPUBindGroupLayout;
  private primaryEntryCullPipeline?: GPUComputePipeline;
  private primaryEntryDrawPipeline?: GPURenderPipeline;
  private primaryEntryCullBindGroup?: GPUBindGroup;
  private primaryEntryDrawBindGroup?: GPUBindGroup;
  private primaryEntryPublicationBuffer?: GPUBuffer;
  private primaryEntryLeafCapacity = 0;
  private primaryEntrySeed?: GPUTexture;
  private primaryEntrySeedView?: GPUTextureView;
  private primaryEntryDepth?: GPUTexture;
  private primaryEntryDepthView?: GPUTextureView;
  private primaryEntryCompilation?: Promise<void>;
  /** Exact historical direct-fragment arm retained as a benchmark control. */
  private readonly rasterPrimaryDirect: boolean;
  /** The same control for the scene-primitive arm, selectable on its own. */
  private readonly scenePrimitiveDirect: boolean;
  private brickCullLayout?: GPUBindGroupLayout;
  private brickDrawLayout?: GPUBindGroupLayout;
  private brickCoverageLayout?: GPUBindGroupLayout;
  private brickCoverageResolveLayout?: GPUBindGroupLayout;
  private scenePrimitiveCoverageLayout?: GPUBindGroupLayout;
  private scenePrimitiveComputeOutputLayout?: GPUBindGroupLayout;
  private scenePrimitiveDepthBridgeLayout?: GPUBindGroupLayout;
  private brickResolveSceneLayout?: GPUBindGroupLayout;
  private brickCullBindGroup?: GPUBindGroup;
  private brickDrawBindGroup?: GPUBindGroup;
  private brickCoverageBindGroup?: GPUBindGroup;
  private brickCoverageResolveBindGroup?: GPUBindGroup;
  private scenePrimitiveCoverageBindGroup?: GPUBindGroup;
  private scenePrimitiveComputeOutputBindGroup?: GPUBindGroup;
  private scenePrimitiveDepthBridgeBindGroup?: GPUBindGroup;
  private brickResolveSceneBindGroup?: GPUBindGroup;
  private scenePrimitiveComputeDepth?: GPUTexture;
  private scenePrimitiveComputeDepthView?: GPUTextureView;
  private scenePrimitiveComputeQueue?: GPUBuffer;
  private scenePrimitiveComputeIndirect?: GPUBuffer;
  private brickEmitPipeline?: GPUComputePipeline;
  private brickScanPipeline?: GPUComputePipeline;
  private brickScatterPipeline?: GPUComputePipeline;
  private brickRasterPipeline?: GPURenderPipeline;
  private brickCoveragePipeline?: GPURenderPipeline;
  private brickCoverageResolvePipeline?: GPURenderPipeline;
  private brickLodResolvePipeline?: GPURenderPipeline;
  private brickExactResolvePipeline?: GPURenderPipeline;
  private brickCoverageOverflowPipeline?: GPURenderPipeline;
  private brickBackgroundPipeline?: GPURenderPipeline;
  private scenePrimitiveRasterPipeline?: GPURenderPipeline;
  private scenePrimitiveCoveragePipeline?: GPURenderPipeline;
  private scenePrimitiveLodResolvePipeline?: GPURenderPipeline;
  private scenePrimitiveComputeArgsPipeline?: GPUComputePipeline;
  private scenePrimitiveComputeResolvePipeline?: GPUComputePipeline;
  private scenePrimitiveDepthBridgePipeline?: GPURenderPipeline;
  private scenePrimitiveCoverageResolvePipeline?: GPURenderPipeline;
  private scenePrimitiveCoverageOverflowPipeline?: GPURenderPipeline;
  /** One-shot: an authored set past the candidate key's index width. */
  private scenePrimitiveCoverageCapacityReported = false;
  /**
   * The near-field analytic band (`lib/svo-scene-primitive-band.ts`).
   *
   * Two compute dispatches per frame over the authored record set — never over
   * pixels — plus one 64 KB state buffer that survives the frame so the
   * hysteresis has a previous membership to be sticky about.
   */
  private bandComputeLayout?: GPUBindGroupLayout;
  private bandReadLayout?: GPUBindGroupLayout;
  private bandComputeBindGroup?: GPUBindGroup;
  private bandReadBindGroup?: GPUBindGroup;
  private bandClassifyPipeline?: GPUComputePipeline;
  private bandResolvePipeline?: GPUComputePipeline;
  private bandStateBuffer?: GPUBuffer;
  private bandParamsBuffer?: GPUBuffer;
  /**
   * The overflow arm's indirect count.
   *
   * `scenePrimitiveOverflowPublicationBuffer` exists because the publisher reads
   * its instance count from a storage buffer, and the authored-SDF arm's count
   * is a host number that changes with every publication — baking it into the
   * shader would mean recompiling the module per scene.
   */
  private coverageOverflowArgsLayout?: GPUBindGroupLayout;
  private coverageOverflowIndirectBuffer?: GPUBuffer;
  private coverageOverflowArgsPipeline?: GPUComputePipeline;
  private brickOverflowArgsBindGroup?: GPUBindGroup;
  private scenePrimitiveOverflowArgsBindGroup?: GPUBindGroup;
  private scenePrimitiveOverflowPublicationBuffer?: GPUBuffer;
  private coverageOverflowReported = false;
  /** Sampled per-pixel arena pressure, and the one-frame-late read that reports it. */
  private coverageAuditBuffer?: GPUBuffer;
  private coverageAuditStaging?: GPUBuffer;
  private coverageAuditCopied = false;
  private coverageAuditReading = false;
  private coverageAuditFrame = 0;
  private brickCandidateBuffer?: GPUBuffer;
  private brickInstanceBuffer?: GPUBuffer;
  private brickSortStateBuffer?: GPUBuffer;
  private brickRasterPublicationBuffer?: GPUBuffer;
  private brickSortStateOffsetBytes = 0;
  private brickInstanceOffsetBytes = 0;
  private brickCoverageCountBuffer?: GPUBuffer;
  private brickCoverageCandidateBuffer?: GPUBuffer;
  private brickCoverageWidth = 0;
  private brickCoverageHeight = 0;
  private brickLeafCapacity = 0;
  private brickCullCompilation?: Promise<void>;
  /**
   * Raster-primary pixel probe: reads this frame's own instance list and cull
   * counters to explain how the depth test found the pixel. Compiled on first
   * trace request and only while the raster primary is the active mode.
   */
  private brickProbeLayout?: GPUBindGroupLayout;
  private brickProbeBindGroup?: GPUBindGroup;
  private brickProbePipeline?: GPUComputePipeline;
  private brickProbeBuffers?: SparseVoxelBrickRasterProbeBuffers;
  private brickProbeCompilation?: Promise<void>;
  private brickProbeReadPending = false;
  private layout?: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private vertexModule?: GPUShaderModule;
  /** Reduced-rate cone-lighting prepass state; absent at scale 1. */
  private coneScale: SvoConeLightingScale = 1;
  private conePipelineScale?: SvoConeLightingScale;
  private readonly conePipelineBundles = new Map<SvoConeLightingScale, SvoDryConePipelineBundle>();
  private readonly conePipelineCompiles = new Map<SvoConeLightingScale, Promise<SvoDryConePipelineBundle>>();
  private requestedBundleFailure?: { readonly scale: SvoConeLightingScale; readonly detail: string };
  private requestedBundleResourceFailure?: string;
  private coneScalePrewarmStarted = false;
  private conePrepassGeometryPipeline?: GPURenderPipeline;
  private conePrepassVisibilityPipeline?: GPURenderPipeline;
  private conePrepassShadePipeline?: GPURenderPipeline;
  private coneReducedPipeline?: GPURenderPipeline;
  private conePrepassLayout?: GPUBindGroupLayout;
  private conePrepassComputeLayout?: GPUBindGroupLayout;
  private conePrepassComputeBindGroup?: GPUBindGroup;
  private conePrepassBoundaryQueue?: GPUBuffer;
  private coneBoundaryCountSnapshot?: GPUBuffer;
  private coneDerivedFailureSnapshot?: GPUBuffer;
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
  private sceneArenaBuffer: GPUBuffer;
  private primitiveCount = 0;
  private primitiveCandidateArena?: SvoPrimitiveCandidateArena;
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
  private readonly nodeMipPageValidityFallback: GPUTexture;
  private readonly nodeMipPageValidityFallbackView: GPUTextureView;
  private readonly nodeMipFallbackSampler: GPUSampler;
  private readonly tetrahedralRadianceFallback: readonly [GPUTexture, GPUTexture, GPUTexture, GPUTexture];
  private readonly tetrahedralRadianceFallbackViews: readonly [GPUTextureView, GPUTextureView, GPUTextureView, GPUTextureView];
  private readonly tetrahedralRadianceBlackFallback: GPUTexture;
  private readonly tetrahedralRadianceBlackFallbackView: GPUTextureView;
  private readonly tetrahedralRadiancePageValidityFallback: GPUTexture;
  private readonly tetrahedralRadiancePageValidityFallbackView: GPUTextureView;
  private tetrahedralRadianceBlackPages?: GPUTexture;
  private tetrahedralRadianceBlackPagesView?: GPUTextureView;
  private readonly fluidCoverageFallback: GPUTexture;
  private readonly fluidCoverageFallbackView: GPUTextureView;
  private fluidCoverage?: WebGpuSvoFluidCoverage;
  /** The coverage view the live bind groups were built with; see refreshFluidCoverageFrame. */
  private boundFluidCoverageView?: GPUTextureView;
  private rigidMotionSource?: GPUBuffer;
  /** xyz centre, w radius of one sphere over every body; negative radius = none. */
  private rigidBounds: [number, number, number, number] = [0, 0, 0, -1];
  private readonly gBufferTargets: SparseVoxelGBufferTargetArena;
  private readonly pickingReadback: SparseVoxelGpuPickingReadbackRing;
  private lastPickingTarget?: GPUTexture;
  private readonly targetViews = new WeakMap<GPUTexture, GPUTextureView>();
  /** Resource/source epoch. Later compatible frames do not invalidate a copy already ordered on the queue. */
  private pickingFrameToken = 1;
  private paramsWords?: Uint32Array<ArrayBuffer>;
  private source?: SparseVoxelSceneRenderSource;
  private scene?: SparseVoxelDrySceneData;
  private primitiveDirtyBounds: readonly SvoDrySceneDirtyBounds[] = [];
  private lightingOptions: SvoLightingOptions = DEFAULT_SVO_LIGHTING_OPTIONS;
  private silhouetteRefinementEnabled = false;
  private renderTuning: SvoRenderTuning = DEFAULT_SVO_RENDER_TUNING;
  /** The scene document's own answer, taken when the tuning says `auto`. */
  private sceneSeeThroughSolids = false;
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
  /** Whether the frame the pending probe was encoded beside ran the cone prepass. */
  private probeEncodedConePrepass = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bodyBuffer: GPUBuffer,
    private readonly targetFormat: GPUTextureFormat = "rgba16float",
    private readonly traversalMode: SvoDryTraversalMode = "hybrid",
    private readonly brickOccupancyMode: SvoBrickOccupancyMode = "bounds",
    private readonly shadingPath: SvoDryShadingPath = "inline",
    private readonly screenSpaceTerminationPixels = 0,
    private readonly rasterGlassDiscovery = false,
    private readonly rasterRigidDiscovery = false,
    private readonly coneFanout = false,
    private readonly experiments: SvoDryOptimizationExperiments = {},
  ) {
    if (targetFormat !== SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat) {
      throw new Error(`Sparse voxel dry scene location 0 must use ${SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat}`);
    }
    if (shadingPath !== "inline" && shadingPath !== "split") throw new RangeError(`Unsupported dry-scene shading path: ${shadingPath}`);
    if (screenSpaceTerminationPixels > 0
      && !((traversalMode === "canonical" && shadingPath === "inline")
        || (traversalMode === "raster-primary" && shadingPath === "split"))) {
      throw new RangeError("Screen-space termination requires canonical inline or raster-primary split traversal");
    }
    this.rasterPrimary = traversalMode === "raster-primary";
    if (experiments.surfaceMesh && (!this.rasterPrimary || shadingPath !== "split" || screenSpaceTerminationPixels !== 0)) {
      throw new RangeError("Surface mesh requires raster-primary, split shading and a zero screen-space threshold");
    }
    // Matches `primaryEntrySeed` in the shader builder exactly: the pass and the
    // fragment binding that reads it are one decision, and a disagreement would
    // be a pipeline whose layout carries a plane nothing writes.
    this.primaryEntryPrepassEnabled = shadingPath === "split" && traversalMode !== "raster-primary"
      && experiments.primaryEntryPrepass !== false;
    this.rasterPrimaryDirect = experiments.surfaceMesh === true || experiments.rasterPrimaryDirect === true
      || experiments.rasterPrimaryNoFragmentDepth === true
      || experiments.rasterPrimaryHsrProbe === true;
    // The scene-primitive arm follows the brick arm's control switch unless it
    // is selected on its own, so the existing `rasterPrimaryDirect` lane stays
    // the fully historical raster-primary graph it has always been.
    this.scenePrimitiveDirect = this.rasterPrimaryDirect || experiments.scenePrimitiveDirect === true;
    if (this.rasterPrimary) {
      if (!(shadingPath === "split" && rasterGlassDiscovery && rasterRigidDiscovery)) {
        throw new RangeError("Raster-primary traversal requires split shading with raster glass and rigid discovery");
      }
      // Four depth-tested colour planes replace the split path's untested
      // storage-texture writes, so the device must have granted the wider
      // per-sample budget requested by requiredFluidDeviceLimits.
      if (device.limits.maxColorAttachmentBytesPerSample < SVO_BRICK_RASTER_CONTRACT.colorAttachmentBytesPerSample) {
        throw new RangeError(`Raster-primary traversal needs maxColorAttachmentBytesPerSample >= ${SVO_BRICK_RASTER_CONTRACT.colorAttachmentBytesPerSample}`);
      }
    }
    // The thresholded graph deliberately removes bodies from its exact-only
    // brick shader, so it keeps the independently depth-tested rigid arm. The
    // threshold-zero coverage resolve still folds the small analytic loop.
    this.rasterRigidActive = rasterRigidDiscovery
      && (!(this.rasterPrimary && !this.rasterPrimaryDirect) || this.screenSpaceTerminationPixels > 0);
    this.paramsBuffer = device.createBuffer({ label: "Sparse voxel dry scene parameters", size: SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sceneArenaBuffer = device.createBuffer({
      label: "Live authored scene arena (materials, primitives/BVH, thin glass)",
      size: SVO_DRY_SCENE_ARENA_LAYOUT.sizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.lightingBuffer = device.createBuffer({ label: "Sparse voxel dry scene lighting arena", size: SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rigidMotionUniformBuffer = device.createBuffer({ label: "Sparse voxel rigid motion uniform mirror", size: SVO_DRY_RIGID_MOTION_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.thickGlassUniformBuffer = device.createBuffer({ label: "Sparse voxel thick-glass uniform binder", size: SVO_DRY_THICK_GLASS_ARENA_LAYOUT.sizeBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rasterGlassParamsBuffer = device.createBuffer({ label: "Sparse voxel raster-glass parameters", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (this.rasterPrimary) {
      // Viewport-independent by construction: the band is per authored record,
      // not per pixel, so it survives every resize the coverage arena does not.
      this.bandStateBuffer = device.createBuffer({
        label: "Sparse voxel near-field analytic band state",
        size: SVO_SCENE_PRIMITIVE_BAND_STATE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.bandParamsBuffer = device.createBuffer({
        label: "Sparse voxel near-field analytic band controls",
        size: SVO_SCENE_PRIMITIVE_BAND_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.coverageAuditBuffer = device.createBuffer({
        label: "Sparse voxel scene-primitive coverage arena audit",
        size: SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.coverageAuditStaging = device.createBuffer({
        label: "Sparse voxel scene-primitive coverage audit readback",
        size: SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.scenePrimitiveOverflowPublicationBuffer = device.createBuffer({
        label: "Sparse voxel scene-primitive overflow instance publication",
        size: 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.coverageOverflowIndirectBuffer = device.createBuffer({
        label: "Sparse voxel coverage overflow draw arguments",
        size: SVO_DRAW_INDIRECT_ARGS_BYTES,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      });
      this.writeBandParams();
    }
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
    this.nodeMipPageValidityFallback = device.createTexture({
      label: "Sparse voxel node-mip page-validity fallback",
      size: [1, 1],
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this.nodeMipPageValidityFallbackView = this.nodeMipPageValidityFallback.createView();
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
    this.tetrahedralRadiancePageValidityFallback = device.createTexture({
      label: "Sparse voxel tetrahedral-radiance page-validity fallback",
      size: [1, 1],
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this.tetrahedralRadiancePageValidityFallbackView = this.tetrahedralRadiancePageValidityFallback.createView();
    // Zero-initialized, and the packed frame reports invalid alongside it, so a
    // scene with no solver never samples this and never shows a water shadow.
    this.fluidCoverageFallback = device.createTexture({ label: "Sparse voxel fluid coverage fallback", size: [1, 1, 1], dimension: "3d", format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.fluidCoverageFallbackView = this.fluidCoverageFallback.createView({ dimension: "3d" });
    this.gBufferTargets = new SparseVoxelGBufferTargetArena(device);
    this.pickingReadback = new SparseVoxelGpuPickingReadbackRing(device);
  }

  async initialize(progress?: (label: string, completed: number, total: number) => void): Promise<void> {
    const report = (completed: number) => progress?.(
      SVO_PRESENTATION_STARTUP_STAGES[completed]!, completed, SVO_PRESENTATION_STARTUP_STAGES.length,
    );
    report(0);
    // Mesh mode fails closed and never dispatches the monolithic primary.
    // Do not specialize that unused ray-marching pipeline during raster startup.
    const meshOnly = this.experiments.surfaceMesh === true;
    const fragmentShader = meshOnly ? undefined : this.traversalMode === "hybrid" && this.brickOccupancyMode === "off" && this.screenSpaceTerminationPixels === 0
      ? drySceneShader : createSvoDrySceneFragmentWGSL(1, this.traversalMode, this.brickOccupancyMode, this.shadingPath, this.screenSpaceTerminationPixels,
        false, this.rasterPrimary && this.rasterGlassDiscovery, this.rasterPrimary && this.rasterRigidDiscovery, false,
        { ...this.experiments, voxelLightCache: false });
    report(1);
    const [vertexModule, fragmentModule] = await Promise.all([
      checkedModule(this.device, "Sparse voxel dry scene vertex", drySceneVertexShader),
      fragmentShader === undefined ? Promise.resolve(undefined) : checkedModule(this.device, `Sparse voxel dry scene fragment (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
        fragmentShader),
    ]);
    report(2);
    this.layout = this.device.createBindGroupLayout({
      label: `Sparse voxel dry scene bindings (${this.traversalMode})`,
      entries: sparseVoxelDrySceneBindGroupLayoutEntries(this.traversalMode),
    });
    if (fragmentModule) this.pipeline = await this.device.createRenderPipelineAsync({
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
    // The brick draw layout is a pipeline-layout input of the split bundle, so
    // instance emission compiles first.
    report(3);
    this.ensureSurfaceMeshBuffers();
    await this.ensureBrickCullPipelines();
    await this.ensurePrimaryEntryPrepassPipelines();
    report(4);
    // These bundles are independent. Track each completion while retaining the
    // parallel compile that keeps overall startup bounded by the slowest
    // browser/driver job rather than the sum of all three.
    let completedFamilies = 4;
    const trackFamily = async (label: string, work: Promise<void>) => {
      await work;
      completedFamilies += 1;
      progress?.(`${label} ready`, completedFamilies, SVO_PRESENTATION_STARTUP_STAGES.length);
    };
    await Promise.all([
      trackFamily(SVO_PRESENTATION_STARTUP_STAGES[4],
        this.shadingPath === "split" ? this.ensureSplitPipelines(1) : Promise.resolve()),
      trackFamily(SVO_PRESENTATION_STARTUP_STAGES[5], Promise.all([
        this.rasterGlassDiscovery ? this.ensureRasterGlassPipeline() : Promise.resolve(),
        this.rasterRigidDiscovery ? this.ensureRasterRigidPipeline() : Promise.resolve(),
      ]).then(() => {})),
      trackFamily(SVO_PRESENTATION_STARTUP_STAGES[6],
        this.coneFanout ? this.ensureConeFanoutPipelines() : Promise.resolve()),
    ]);
    report(7);
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
    this.primarySeamClosurePipeline = bundle.primarySeamClosure;
    this.splitLightingPipeline = bundle.lighting;
    this.splitOptimizedLightingPipeline = bundle.optimizedLighting;
    this.splitReconstructedLightingPipeline = bundle.reconstructedLighting;
    this.splitSkyLightingPipeline = bundle.skyLighting;
    this.conePrepassResetPipeline = bundle.prepassReset;
    this.conePrepassCoherentPipeline = bundle.prepassCoherent;
    this.conePrepassBoundaryPipeline = bundle.prepassBoundary;
    this.worldGiFramePipeline = bundle.worldGiFrame;
    this.worldGiCachePipeline = bundle.worldGiCache;
    this.voxelLightDemandPipeline = bundle.voxelLightDemand;
    this.voxelLightPopulatePipeline = bundle.voxelLightPopulate;
    this.surfaceMeshPipelines = bundle.surfaceMesh;
    this.brickBackgroundPipeline = bundle.brickBackground;
    this.brickRasterPipeline = bundle.brickRaster;
    this.brickCoveragePipeline = bundle.brickCoverage;
    this.brickCoverageResolvePipeline = bundle.brickCoverageResolve;
    this.brickLodResolvePipeline = bundle.brickLodResolve;
    this.brickExactResolvePipeline = bundle.brickExactResolve;
    this.brickCoverageOverflowPipeline = bundle.brickCoverageOverflow;
    this.scenePrimitiveRasterPipeline = bundle.scenePrimitiveRaster;
    this.scenePrimitiveCoveragePipeline = bundle.scenePrimitiveCoverage;
    this.scenePrimitiveLodResolvePipeline = bundle.scenePrimitiveLodResolve;
    this.scenePrimitiveComputeArgsPipeline = bundle.scenePrimitiveComputeArgs;
    this.scenePrimitiveComputeResolvePipeline = bundle.scenePrimitiveComputeResolve;
    this.scenePrimitiveDepthBridgePipeline = bundle.scenePrimitiveDepthBridge;
    this.scenePrimitiveCoverageResolvePipeline = bundle.scenePrimitiveCoverageResolve;
    this.scenePrimitiveCoverageOverflowPipeline = bundle.scenePrimitiveCoverageOverflow;
    this.splitPipelineScale = scale;
    if (this.requestedBundleFailure?.scale === scale) this.requestedBundleFailure = undefined;
    this.requestedBundleResourceFailure = undefined;
    this.ensureSplitTargets();
    this.ensureConePrepassTargets();
  }

  /**
   * Selects between two already-compiled exact primary-discovery strategies.
   * The measured Metal crossover is above one body and below six; four keeps
   * small scenes on the cheaper analytic loop while amortizing raster passes
   * for body stacks. Body motion never changes this choice or recompiles WGSL.
   */
  setRigidBodyCount(bodyCount: number, bounds?: SvoDryRigidBounds): void {
    // Shadow and contact rays consult one sphere around the whole set before
    // they read a body, so it is republished whenever bodies move even though
    // the count has not changed. An empty scene publishes a negative radius,
    // which retires the body loop outright rather than shrinking it.
    const published: [number, number, number, number] = bodyCount > 0 && bounds
      ? [bounds.centre_m[0], bounds.centre_m[1], bounds.centre_m[2], Math.max(bounds.radius_m, 0)]
      : [0, 0, 0, -1];
    // writeParams rebuilds the entire parameter block, so a still scene must not
    // pay for it every frame just to restate a sphere that has not moved.
    if (published.some((value, axis) => value !== this.rigidBounds[axis])) {
      this.rigidBounds = published;
      if (this.source && this.scene && canEncodeSparseVoxelDryScene(this.source, this.scene)) {
        this.writeParams(this.source, this.scene);
      }
    }
    // The coverage resolve carries an analytic body loop of its own (W1's
    // impostor fold), so raster-primary no longer needs the twelve-instance
    // impostor pass to see a body at all — and dropping it is what releases
    // stationary primary reuse, which is worth far more than the pass it
    // replaces. The direct arm keeps the impostors: its background pass traces
    // terrain only, so there a body can come from nowhere else.
    const foldedIntoCoverageResolve = this.rasterPrimary && !this.rasterPrimaryDirect
      && this.screenSpaceTerminationPixels === 0;
    const active = !foldedIntoCoverageResolve
      && (svoDryRigidPrimaryStrategy(bodyCount, this.rasterRigidDiscovery) === "raster"
        || (this.rasterPrimary && bodyCount > 0));
    if (active === this.rasterRigidActive) return;
    this.rasterRigidActive = active;
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

  /**
   * Emission, prefix scan and scatter for the brick instance list. The module
   * is standalone — camera uniform, published topology and the `SvoMapping`
   * prefix of `DryParams` — so it is independent of cone-lighting scale and of
   * the renderer's fragment-only shading bindings.
   */
  private ensureSurfaceMeshBuffers(): void {
    if (!this.experiments.surfaceMesh || this.surfaceMeshState) return;
    this.surfaceMeshStatus = { state: "pending" };
    this.surfaceMeshReadback = this.device.createBuffer({ label: "Voxel mesh status", size: SVO_SURFACE_MESH_STATE_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // Four indirect triples: mark, extract, allocate, cull.
    this.surfaceMeshDispatch = this.device.createBuffer({ label: "Voxel mesh build dispatch", size: 64, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.surfaceMeshState = this.device.createBuffer({ label: "Voxel surface mesh publication",
      size: SVO_SURFACE_MESH_STATE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const requestedBytes = this.experiments.surfaceMeshMaxBytes ?? this.device.limits.maxStorageBufferBindingSize;
    if (!Number.isSafeInteger(requestedBytes) || requestedBytes < 32) throw new RangeError("Surface mesh budget must be an integer of at least 32 bytes");
    this.surfaceMeshMaximumBytes = Math.floor(Math.min(requestedBytes, this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize) / 32) * 32;
    this.surfaceMeshArenas = [this.createSurfaceMeshArena(Math.min(SVO_SURFACE_MESH_BYTES, this.surfaceMeshMaximumBytes)), undefined];
    this.surfaceMeshEmptyArena = this.device.createBuffer({ label: "Absent voxel quad arena", size: 2 * SVO_SURFACE_MESH_QUAD_BYTES, usage: GPUBufferUsage.STORAGE });
    this.surfaceMeshEmptyMaintenance = this.device.createBuffer({ label: "Absent voxel maintenance list", size: 256, usage: GPUBufferUsage.STORAGE });
    this.surfaceMeshVisible = this.createSurfaceMeshVisible(this.surfaceMeshArenas[0]!.size);
    this.surfaceMeshComputeLayout = this.device.createBindGroupLayout({ entries: [
      { binding: 30, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 32, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 38, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 34, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 40, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 41, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    this.surfaceMeshDrawLayout = this.device.createBindGroupLayout({ entries: [
      { binding: 31, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 37, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 35, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 33, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 42, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ] });
    this.ensureSurfaceMeshWork();
    this.bindSurfaceMeshBuffers();
  }

  private createSurfaceMeshArena(bytes: number): GPUBuffer {
    return this.device.createBuffer({ label: "Cached voxel boundary quads", size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  private createSurfaceMeshVisible(arenaBytes: number): GPUBuffer {
    return this.device.createBuffer({ label: "Visible voxel quad indices", size: arenaBytes / 8, usage: GPUBufferUsage.STORAGE });
  }

  private surfaceMeshArenaBytes(slot: 0 | 1): number {
    return this.surfaceMeshArenas[slot]?.size ?? this.surfaceMeshEmptyArena!.size;
  }

  /**
   * The per-leaf range table follows the source's leaf capacity. A new
   * capacity invalidates every range, so the table is re-created zeroed and
   * the GPU state is reset to a first build.
   */
  private ensureSurfaceMeshWork(): void {
    if (!this.surfaceMeshState) return;
    const leafCapacity = Math.max(1, this.source?.structural?.capacities.leaves ?? 1);
    const maintenance = this.source?.structural?.sceneMaintenance;
    const maintenanceBuffer = maintenance?.buffer ?? this.surfaceMeshEmptyMaintenance!;
    const rebind = this.surfaceMeshMaintenanceBuffer !== maintenanceBuffer;
    if (this.surfaceMeshWork && this.surfaceMeshWorkLeafCapacity === leafCapacity) {
      if (rebind) { this.surfaceMeshMaintenanceBuffer = maintenanceBuffer; this.writeSurfaceMeshHostWords(); this.bindSurfaceMeshBuffers(); }
      return;
    }
    const previous = this.surfaceMeshWork;
    this.surfaceMeshWork = this.device.createBuffer({ label: "Voxel mesh brick ranges and worklist", size: surfaceMeshWorkBytes(leafCapacity),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.surfaceMeshWorkLeafCapacity = leafCapacity;
    this.surfaceMeshMaintenanceBuffer = maintenanceBuffer;
    if (previous) { this.resetSurfaceMeshState(); const retire = () => previous.destroy(); void this.device.queue.onSubmittedWorkDone().then(retire, retire); }
    else this.writeSurfaceMeshHostWords();
    if (this.surfaceMeshComputeLayout) this.bindSurfaceMeshBuffers();
  }

  /** Zero the GPU builder state to a first build, keeping the arenas bound as they are. */
  private resetSurfaceMeshState(): void {
    if (!this.surfaceMeshState) return;
    this.device.queue.writeBuffer(this.surfaceMeshState, 0, new Uint32Array(SVO_SURFACE_MESH_STATE_BYTES / 4));
    this.device.queue.writeBuffer(this.surfaceMeshState, SVO_SURFACE_MESH_STATE.front * 4, new Uint32Array([this.surfaceMeshHostFront]));
    // A zeroed state has consumed no back arena; the generation it is told
    // matches so that a replacement build asks for one afresh.
    this.device.queue.writeBuffer(this.surfaceMeshState, SVO_SURFACE_MESH_STATE.backGenerationConsumed * 4, new Uint32Array([this.surfaceMeshBackGeneration]));
    this.surfaceMeshBackPending = false;
    this.writeSurfaceMeshHostWords();
    this.surfaceMeshBuildPresentations = 0;
    if (this.surfaceMeshStatus) this.surfaceMeshStatus = { ...this.surfaceMeshStatus, state: "pending", drawn: false, buildPhase: "extracting" };
  }

  /** Words only the host writes: maintenance layout, table capacity and the back arena generation. */
  private writeSurfaceMeshHostWords(): void {
    if (!this.surfaceMeshState) return;
    const W = SVO_SURFACE_MESH_STATE;
    const maintenance = this.source?.structural?.sceneMaintenance;
    const bound = maintenance !== undefined && this.surfaceMeshMaintenanceBuffer === maintenance.buffer;
    this.device.queue.writeBuffer(this.surfaceMeshState, W.hostMaintenance * 4,
      new Uint32Array([bound ? 1 : 0, this.surfaceMeshWorkLeafCapacity, this.surfaceMeshBackGeneration]));
    this.device.queue.writeBuffer(this.surfaceMeshState, W.hostSurfaceVertices * 4, new Uint32Array([bound ? ((maintenance?.surfaceVertexOffsetBytes ?? 0) / 4) | (maintenance?.surfaceVertexKind === "dual-marching-cubes" ? 0x80000000 : 0) : 0]));
    this.device.queue.writeBuffer(this.surfaceMeshState, W.hostMaintenanceStateWords * 4,
      new Uint32Array([(maintenance?.stateOffsetBytes ?? 0) / 4, (maintenance?.dirtyBrickOffsetBytes ?? 0) / 4, maintenance?.dirtyBrickCapacity ?? 0]));
  }

  private bindSurfaceMeshBuffers(): void {
    const arena = (slot: 0 | 1) => this.surfaceMeshArenas[slot] ?? this.surfaceMeshEmptyArena!;
    this.surfaceMeshComputeGroup = this.device.createBindGroup({ layout: this.surfaceMeshComputeLayout!, entries: [
      { binding: 30, resource: { buffer: this.surfaceMeshState! } },
      { binding: 32, resource: { buffer: arena(0) } },
      { binding: 38, resource: { buffer: arena(1) } },
      { binding: 34, resource: { buffer: this.surfaceMeshVisible! } },
      { binding: 40, resource: { buffer: this.surfaceMeshWork! } },
      { binding: 41, resource: { buffer: this.surfaceMeshMaintenanceBuffer ?? this.surfaceMeshEmptyMaintenance! } },
    ] });
    this.surfaceMeshDrawGroup = this.device.createBindGroup({ layout: this.surfaceMeshDrawLayout!, entries: [
      { binding: 31, resource: { buffer: arena(0) } },
      { binding: 37, resource: { buffer: arena(1) } },
      { binding: 35, resource: { buffer: this.surfaceMeshVisible! } },
      { binding: 33, resource: { buffer: this.surfaceMeshState! } },
      { binding: 42, resource: { buffer: this.surfaceMeshWork! } },
    ] });
  }

  /**
   * Act on one state receipt: adopt a flip, provide a back arena a
   * replacement build waits for, or grow the arena a batch overflowed. Every
   * decision here is idempotent against stale receipts because the GPU only
   * resumes when it sees the binding it asked for.
   */
  private applySurfaceMeshReceipt(words: Uint32Array, arenaBytes: readonly [number, number]): void {
    const receipt = interpretSurfaceMeshState(words, { arenaBytes, maximumBytes: this.surfaceMeshMaximumBytes });
    const previous = this.surfaceMeshStatus;
    this.surfaceMeshStatus = receipt.status;
    if (previous?.state === "ready" && receipt.status.state === "pending") this.surfaceMeshBuildPresentations = 0;
    const retireLater = (...buffers: (GPUBuffer | undefined)[]) => {
      const retire = () => { for (const buffer of buffers) buffer?.destroy(); };
      void this.device.queue.onSubmittedWorkDone().then(retire, retire);
    };
    if (receipt.front !== this.surfaceMeshHostFront) {
      // The replacement build flipped: the arena it replaced is retired and
      // its slot waits empty for the next replacement's request.
      const retired = this.surfaceMeshArenas[this.surfaceMeshHostFront];
      this.surfaceMeshArenas[this.surfaceMeshHostFront] = undefined;
      this.surfaceMeshHostFront = receipt.front;
      this.surfaceMeshBackPending = false;
      this.bindSurfaceMeshBuffers();
      retireLater(retired);
    }
    if (receipt.needBackBytes !== undefined && !this.surfaceMeshBackPending) {
      const slot = (1 - this.surfaceMeshHostFront) as 0 | 1;
      const bytes = Math.max(Math.min(SVO_SURFACE_MESH_BYTES, this.surfaceMeshMaximumBytes),
        Math.min(this.surfaceMeshMaximumBytes, Math.ceil(receipt.needBackBytes / SVO_SURFACE_MESH_QUAD_BYTES) * SVO_SURFACE_MESH_QUAD_BYTES));
      const stale = this.surfaceMeshArenas[slot];
      this.surfaceMeshArenas[slot] = this.createSurfaceMeshArena(bytes);
      const previousVisible = this.surfaceMeshVisible!;
      if (bytes / 8 > previousVisible.size) this.surfaceMeshVisible = this.createSurfaceMeshVisible(bytes);
      this.surfaceMeshBackGeneration += 1;
      this.surfaceMeshBackPending = true;
      this.device.queue.writeBuffer(this.surfaceMeshState!, SVO_SURFACE_MESH_STATE.hostBackGeneration * 4, new Uint32Array([this.surfaceMeshBackGeneration]));
      this.bindSurfaceMeshBuffers();
      retireLater(stale, this.surfaceMeshVisible !== previousVisible ? previousVisible : undefined);
      this.surfaceMeshStatus = { ...this.surfaceMeshStatus, state: "pending", detail: "Mesh storage provided; rebuilding beside the drawn mesh." };
    }
    if (receipt.grow) {
      // The GPU rolled the overflowing batch back to its checkpoint and
      // paused. Copy the whole arena in queue order: unlike a readback-derived
      // prefix length, this remains correct if a newer publication restarted
      // extraction while the receipt was in flight. GPU prepare recognizes
      // the larger binding and resumes on its own.
      const slot = receipt.grow.slot;
      const current = this.surfaceMeshArenas[slot];
      if (current && current.size / SVO_SURFACE_MESH_QUAD_BYTES <= receipt.grow.overflowQuads && current.size < this.surfaceMeshMaximumBytes) {
        const bytes = Math.min(this.surfaceMeshMaximumBytes, current.size * 2);
        const grown = this.createSurfaceMeshArena(bytes);
        const copy = this.device.createCommandEncoder({ label: "Retain completed voxel mesh bricks during growth" });
        copy.copyBufferToBuffer(current, 0, grown, 0, current.size);
        this.device.queue.submit([copy.finish()]);
        this.surfaceMeshArenas[slot] = grown;
        const previousVisible = this.surfaceMeshVisible!;
        if (bytes / 8 > previousVisible.size) this.surfaceMeshVisible = this.createSurfaceMeshVisible(bytes);
        this.bindSurfaceMeshBuffers();
        retireLater(current, this.surfaceMeshVisible !== previousVisible ? previousVisible : undefined);
        this.surfaceMeshStatus = { ...this.surfaceMeshStatus, state: "pending", buildPhase: "extracting",
          ...(slot === this.surfaceMeshHostFront ? { allocatedBytes: bytes, capacityQuads: bytes / SVO_SURFACE_MESH_QUAD_BYTES } : {}),
          detail: "Mesh storage enlarged; resuming from completed bricks." };
      }
    }
  }

  /** Includes face count, overflow, source revisions, build count and raster readiness. */
  copySurfaceMeshDiagnostics(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    if (!this.surfaceMeshState || target.size < SVO_SURFACE_MESH_HEADER_BYTES) return false;
    encoder.copyBufferToBuffer(this.surfaceMeshState, 0, target, 0, SVO_SURFACE_MESH_HEADER_BYTES);
    return true;
  }

  private encodeSurfaceMesh(encoder: GPUCommandEncoder, views: SparseVoxelGBufferViews,
    usePrepass: boolean, group: number, tracePhase?: RenderFrameSeam<"svo">): void {
    this.ensureSurfaceMeshWork();
    // Poll only a copy encoded in the preceding submitted frame.
    if (this.surfaceMeshReadbackCopied && !this.surfaceMeshReadbackPending) {
      this.surfaceMeshReadbackCopied = false; this.surfaceMeshReadbackPending = true;
      const staging = this.surfaceMeshReadback!;
      const arenaBytes = this.surfaceMeshReceiptArenaBytes;
      void staging.mapAsync(GPUMapMode.READ).then(() => {
        const words = new Uint32Array(staging.getMappedRange().slice(0)); staging.unmap();
        if (this.surfaceMeshDisposed) return;
        this.applySurfaceMeshReceipt(words, arenaBytes);
      }).catch(() => { /* Destruction or device loss cancels the diagnostic. */ })
        .finally(() => { this.surfaceMeshReadbackPending = false; });
    }
    const pipelines = this.surfaceMeshPipelines!;
    const state = this.surfaceMeshState!;
    const dispatch = this.surfaceMeshDispatch!;
    const compute = (label: string, pipeline: GPUComputePipeline, indirectOffset?: number, groups = 1) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindGroup);
      if (usePrepass) pass.setBindGroup(1, this.conePrepassBindGroup!);
      pass.setBindGroup(group, this.surfaceMeshComputeGroup!);
      if (indirectOffset === undefined) pass.dispatchWorkgroups(Math.min(groups, 65535), Math.max(1, Math.ceil(groups / 65535))); else pass.dispatchWorkgroupsIndirect(dispatch, indirectOffset);
      pass.end();
    };
    const W = SVO_SURFACE_MESH_STATE;
    // GPU prepare checks revisions, completion and capacity before every
    // batch; the host only decides how many bricks one presentation extracts.
    // The ramp restarts with each build so a small edit's re-extraction is a
    // cheap presentation, and climbs while a build stays pending. While the
    // drawn mesh is complete the build passes are left out: a publication the
    // host made itself is known at once, and one it only learns of from a
    // receipt (a compaction) waits the two frames that receipt takes.
    const pending = this.surfaceMeshStatus?.state !== "ready";
    const drawn = this.surfaceMeshStatus?.drawn === true;
    const bricks = pending ? surfaceMeshBuildBricks(this.surfaceMeshBuildPresentations, drawn) : SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL;
    this.surfaceMeshBuildPresentations = pending ? this.surfaceMeshBuildPresentations + 1 : 0;
    this.device.queue.writeBuffer(state, W.bricksPerBatch * 4, new Uint32Array([bricks]));
    compute("Voxel surface mesh revision check", pipelines.prepare);
    if (pending) {
      compute("Voxel surface mesh dirty boxes", pipelines.boxes);
      encoder.copyBufferToBuffer(state, W.markDispatch * 4, dispatch, 0, 12);
      compute("Voxel surface mesh brick marking", pipelines.mark, 0);
      compute("Voxel surface mesh batch schedule", pipelines.schedule);
      encoder.copyBufferToBuffer(state, W.extractDispatch * 4, dispatch, 16, 12);
      encoder.copyBufferToBuffer(state, W.allocateDispatch * 4, dispatch, 32, 12);
      compute("Voxel surface mesh quad count", pipelines.count, 16);
      compute("Voxel surface mesh range allocation", pipelines.allocate, 32);
      compute("Voxel surface mesh quad emission", pipelines.emit, 16);
    }
    compute("Voxel surface mesh publication", pipelines.publish);
    tracePhase?.("surface-mesh-update");
    const background = encoder.beginRenderPass({ label: "Voxel surface mesh background and exact planes",
      colorAttachments: this.rasterPrimaryAttachments(views, "clear"),
      depthStencilAttachment: { view: views.hardwareDepth, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 0 } });
    background.setPipeline(pipelines.background); background.setBindGroup(0, this.bindGroup);
    if (usePrepass) background.setBindGroup(1, this.conePrepassBindGroup!);
    background.setBindGroup(group, this.surfaceMeshDrawGroup!); background.draw(3); background.end();
    tracePhase?.("surface-mesh-background");
    compute("Voxel mesh detail selection", pipelines.select, undefined, Math.ceil(this.surfaceMeshWorkLeafCapacity / 64));
    if (this.experiments.surfaceMeshCulling !== false) {
      encoder.copyBufferToBuffer(state, W.extractDispatch * 4, dispatch, 48, 12);
      compute("Voxel mesh back-face, frustum and dirty-box culling", pipelines.cull, 48);
    }
    tracePhase?.("surface-mesh-cull");
    const draw = encoder.beginRenderPass({ label: "Voxel surface mesh rasterization",
      colorAttachments: this.rasterPrimaryAttachments(views, "load"),
      depthStencilAttachment: { view: views.hardwareDepth, depthLoadOp: "load", depthStoreOp: "store" } });
    draw.setPipeline(pipelines.draw); draw.setBindGroup(0, this.bindGroup);
    if (usePrepass) draw.setBindGroup(1, this.conePrepassBindGroup!);
    draw.setBindGroup(group, this.surfaceMeshDrawGroup!); draw.drawIndirect(state, 0); draw.end();
    // Every presentation carries a receipt: a build the GPU started on its own
    // (a compaction, or a publication the host did not make) is noticed as
    // soon as the copy returns rather than at a periodic poll.
    if (!this.surfaceMeshReadbackPending && !this.surfaceMeshReadbackCopied) {
      encoder.copyBufferToBuffer(state, 0, this.surfaceMeshReadback!, 0, SVO_SURFACE_MESH_STATE_BYTES);
      this.surfaceMeshReceiptArenaBytes = [this.surfaceMeshArenaBytes(0), this.surfaceMeshArenaBytes(1)];
      this.surfaceMeshReadbackCopied = true;
    }
    tracePhase?.("surface-mesh-draw");
  }

  private async ensureBrickCullPipelines(): Promise<void> {
    if (!this.rasterPrimary || this.brickEmitPipeline) return;
    this.brickCullCompilation ??= (async () => {
      this.brickCullLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel brick instance emission bindings",
        entries: svoBrickRasterCullBindGroupLayoutEntries(),
      });
      this.brickDrawLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel brick instance draw bindings",
        entries: svoBrickRasterDrawBindGroupLayoutEntries(),
      });
      this.brickCoverageLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel conservative brick coverage bindings",
        entries: svoBrickRasterCoverageBindGroupLayoutEntries(),
      });
      this.brickCoverageResolveLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel conservative brick resolve bindings",
        entries: [...svoBrickRasterCoverageBindGroupLayoutEntries(), {
          binding: SVO_BRICK_RASTER_CONTRACT.lodKeyBinding,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "uint" },
        }],
      });
      this.scenePrimitiveCoverageLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel scene-primitive occlusion coverage bindings",
        entries: [...svoBrickRasterCoverageBindGroupLayoutEntries(), {
          binding: SVO_BRICK_RASTER_CONTRACT.primaryGeometryBinding,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float" },
        }, {
          binding: SVO_BRICK_RASTER_CONTRACT.primaryIdentityBinding,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "uint" },
        }],
      });
      // Binding 6 joins the resolve set because `traceLeafPayload` resolves
      // accepted planar terminal records, including from raster-generated hits.
      const resolveBindings = new Set([0, 1, 2, 3, 4, 6, 9, 14, 15]);
      this.brickResolveSceneLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel conservative brick resolve scene bindings",
          entries: sparseVoxelDrySceneBindGroupLayoutEntries(this.traversalMode).filter((entry) => resolveBindings.has(entry.binding)),
      });
      this.bandComputeLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel near-field band selection bindings",
        entries: svoScenePrimitiveBandComputeBindGroupLayoutEntries(),
      });
      this.bandReadLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel near-field band membership binding",
        entries: [
          ...svoScenePrimitiveBandReadBindGroupLayoutEntries(),
          ...(this.screenSpaceTerminationPixels > 0
            ? [{ binding: 8, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
              buffer: { type: "storage" as const } }]
            : []),
        ],
      });
      this.coverageOverflowArgsLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel coverage overflow draw-args bindings",
        entries: svoRasterCoverageOverflowArgsBindGroupLayoutEntries(),
      });
      // One module for both arms: the brick proxy and the authored-SDF proxy are
      // the same thirty-six vertices, and the instance count each arm publishes
      // comes from its own publication buffer rather than from the shader.
      const overflowArgsModule = await checkedModule(this.device, "Sparse voxel coverage overflow draw args",
        createSvoRasterCoverageOverflowArgsWGSL({
          verticesPerInstance: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.verticesPerProxy,
        }));
      this.coverageOverflowArgsPipeline = await this.device.createComputePipelineAsync({
        label: "Sparse voxel coverage overflow draw args",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.coverageOverflowArgsLayout] }),
        compute: { module: overflowArgsModule, entryPoint: SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.entryPoints.publishArgs },
      });
      const bandModule = await checkedModule(this.device, "Sparse voxel near-field analytic band",
        createSvoScenePrimitiveBandWGSL({
          primitiveWordOffset: SVO_DRY_SCENE_ARENA_LAYOUT.primitiveOffsetBytes / 4,
        }));
      const bandLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bandComputeLayout] });
      [this.bandClassifyPipeline, this.bandResolvePipeline] = await Promise.all(
        ([SVO_SCENE_PRIMITIVE_BAND_CONTRACT.entryPoints.classify,
          SVO_SCENE_PRIMITIVE_BAND_CONTRACT.entryPoints.resolve] as const).map((entryPoint) =>
          this.device.createComputePipelineAsync({
            label: `Sparse voxel near-field band ${entryPoint}`,
            layout: bandLayout,
            compute: { module: bandModule, entryPoint },
          })));
      const module = await checkedModule(this.device, "Sparse voxel brick instance emission",
        createSvoBrickRasterCullWGSL({ reversedZNear_m: SVO_DRY_SCENE_REVERSED_Z_NEAR_M }));
      const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.brickCullLayout] });
      const [emit, scan, scatter] = await Promise.all(
        ([SVO_BRICK_RASTER_CONTRACT.entryPoints.emit, SVO_BRICK_RASTER_CONTRACT.entryPoints.scan,
          SVO_BRICK_RASTER_CONTRACT.entryPoints.scatter] as const).map((entryPoint) =>
          this.device.createComputePipelineAsync({
            label: `Sparse voxel brick instance ${entryPoint}`, layout, compute: { module, entryPoint },
          })));
      this.brickEmitPipeline = emit;
      this.brickScanPipeline = scan;
      this.brickScatterPipeline = scatter;
      this.ensureBrickRasterBuffers();
    })();
    await this.brickCullCompilation;
  }

  /** Instance arenas are sized by published leaf capacity, not by leaf count. */
  private ensureBrickRasterBuffers(): void {
    const leafCapacity = this.source?.structural?.capacities.leaves ?? 0;
    if (!this.rasterPrimary || leafCapacity < 1 || this.brickLeafCapacity === leafCapacity) return;
    this.brickCandidateBuffer?.destroy();
    this.brickRasterPublicationBuffer?.destroy();
    const size = svoBrickRasterInstanceBytes(leafCapacity);
    this.brickCandidateBuffer = this.device.createBuffer({
      label: "Sparse voxel brick instance candidates", size, usage: GPUBufferUsage.STORAGE,
    });
    this.brickSortStateOffsetBytes = 0;
    this.brickInstanceOffsetBytes = svoBrickRasterPublicationInstanceOffsetBytes();
    this.brickRasterPublicationBuffer = this.device.createBuffer({
      label: "Sparse voxel brick raster publication (instances and sort state)",
      size: this.brickInstanceOffsetBytes + size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.brickInstanceBuffer = this.brickRasterPublicationBuffer;
    this.brickSortStateBuffer = this.brickRasterPublicationBuffer;
    // Word zero is the indirect vertex count. Every frame clears from word one
    // onward, so this is written exactly once per allocation.
    this.device.queue.writeBuffer(this.brickSortStateBuffer, this.brickSortStateOffsetBytes,
      Uint32Array.of(SVO_BRICK_RASTER_CONTRACT.verticesPerInstance));
    this.brickLeafCapacity = leafCapacity;
    this.brickCullBindGroup = undefined;
    this.brickDrawBindGroup = undefined;
    this.brickCoverageBindGroup = undefined;
    this.brickCoverageResolveBindGroup = undefined;
    this.scenePrimitiveCoverageBindGroup = undefined;
  }

  /** Per-pixel conservative candidate storage; controls bind one dummy lane. */
  private ensureBrickCoverageBuffers(): void {
    if (!this.rasterPrimary || !this.targetWidth || !this.targetHeight) return;
    const width = this.rasterPrimaryDirect ? 1 : this.targetWidth;
    const height = this.rasterPrimaryDirect ? 1 : this.targetHeight;
    const tieredCompute = this.screenSpaceTerminationPixels > 0
      && this.device.limits.maxStorageTexturesPerShaderStage >= 5;
    if (this.brickCoverageCountBuffer && this.brickCoverageCandidateBuffer
      && (!tieredCompute || (this.scenePrimitiveComputeQueue && this.scenePrimitiveComputeIndirect))
      && this.brickCoverageWidth === width && this.brickCoverageHeight === height) return;
    this.brickCoverageCountBuffer?.destroy();
    this.brickCoverageCandidateBuffer?.destroy();
    this.scenePrimitiveComputeQueue?.destroy();
    this.scenePrimitiveComputeIndirect?.destroy();
    // One word per pixel, plus the eight-word tail: the overflow flag the
    // coverage fragments raise and the draw args the overflow pass draws from.
    // The draw args are staged out of here into their own indirect-only buffer
    // before each overflow pass: the overflow fragment binds this buffer as
    // writable storage to read its own pixel's count, and WebGPU forbids a
    // buffer being writable storage and the indirect source inside one pass.
    this.brickCoverageCountBuffer = this.device.createBuffer({
      label: "Sparse voxel conservative coverage counts",
      size: svoRasterCoverageCountAllocationBytes(width, height),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // One arena for both coverage passes plus the seed tail between them; the
    // brick pass strides by its own 24 inside the wider scene-primitive
    // allocation. At 1600x1240 that is 1,984,000 px x (40+1) x 4 B = 325.4 MB,
    // against 190.5 MB for the brick arena alone — a second arena for the SDF
    // set would have cost 317 MB on top of it instead.
    const arenaBytes = svoRasterCoverageArenaBytes(width, height,
      SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel);
    // This allocation is the renderer's largest and it grows with the square of
    // the viewport. Left alone it fails as a Dawn validation error inside a bind
    // group several calls later, which is the "silent memory pressure" row of
    // the capacity audit; name it here instead, with the number.
    const arenaLimit = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
    if (arenaBytes > arenaLimit) {
      throw new RangeError(`Conservative coverage arena needs ${arenaBytes} B at ${width}x${height} `
        + `(${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel} candidates per pixel plus the primary `
        + `depth seed) and the device grants ${arenaLimit} B`);
    }
    this.brickCoverageCandidateBuffer = this.device.createBuffer({
      label: "Sparse voxel conservative coverage candidate arena",
      size: arenaBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    if (tieredCompute) {
      // Four atomic header words followed by one packed coordinate per pixel.
      // Dispatch arguments are copied to a distinct buffer because the queue is
      // writable storage in the resolve pass and WebGPU forbids aliasing that
      // same buffer as an indirect source in the pass.
      this.scenePrimitiveComputeQueue = this.device.createBuffer({
        label: "Sparse voxel tiered scene-primitive exact-pixel queue",
        size: 4 * Uint32Array.BYTES_PER_ELEMENT + width * height * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.scenePrimitiveComputeIndirect = this.device.createBuffer({
        label: "Sparse voxel tiered scene-primitive indirect dispatch",
        size: 3 * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      });
    } else {
      this.scenePrimitiveComputeQueue = undefined;
      this.scenePrimitiveComputeIndirect = undefined;
    }
    this.brickCoverageWidth = width;
    this.brickCoverageHeight = height;
    this.brickCoverageBindGroup = undefined;
    this.brickCoverageResolveBindGroup = undefined;
    this.scenePrimitiveCoverageBindGroup = undefined;
    this.scenePrimitiveComputeOutputBindGroup = undefined;
  }

  /**
   * The raster-primary probe's module and pipeline.
   *
   * Deliberately compiled beside the ray probe rather than with the frame's
   * pipelines: a session that never opens the diagnostic never pays for it. It
   * is also the only pipeline here that binds the instance list read-only,
   * which is what makes it an observer of the frame rather than a participant.
   */
  private async ensureBrickRasterProbe(): Promise<void> {
    if (!this.rasterPrimary || this.experiments.surfaceMesh || this.brickProbePipeline) return;
    this.brickProbeCompilation ??= (async () => {
      try {
        this.brickProbeLayout = this.device.createBindGroupLayout({
          label: "Sparse voxel raster-primary probe bindings",
          entries: svoBrickRasterProbeBindGroupLayoutEntries(),
        });
        const module = await checkedModule(this.device, "Sparse voxel raster-primary probe",
          createSvoBrickRasterProbeWGSL({
            // The shipping brick fragment writes its own depth unless the
            // experiment removes it, and that is exactly what decides whether
            // the covering-proxy count is exact or an upper bound.
            fragmentDepthWritten: !this.experiments.rasterPrimaryHsrProbe
              && !this.experiments.rasterPrimaryNoFragmentDepth,
            primitiveWordOffset: SVO_DRY_SCENE_ARENA_LAYOUT.primitiveOffsetBytes / 4,
            sortStateWordOffset: this.brickSortStateOffsetBytes / 4,
            instanceWordOffset: this.brickInstanceOffsetBytes / 4,
            paramsWordCount: SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes / 4,
            payloadLaneWordOffset: SVO_DRY_SCENE_PARAMS_LAYOUT.payloadLaneWordOffset,
            // The ordinary coverage primary has already published the exact
            // candidate list for every pixel. Direct and screen-space LOD arms
            // do not, so they keep the diagnostic's whole-list fallback.
            coverageAccelerated: !this.rasterPrimaryDirect
              && this.screenSpaceTerminationPixels === 0,
            // The world's own block, not this renderer's opinion of it. The probe
            // reads the lane *addresses* from the uniform for staleness, but which
            // decode to compile is a property of the layout and cannot change
            // under a live module.
            scenePayload: this.source?.structural?.scenePayloadLanes,
          }));
        this.brickProbeBuffers = new SparseVoxelBrickRasterProbeBuffers(this.device);
        this.brickProbePipeline = await this.device.createComputePipelineAsync({
          label: "Sparse voxel raster-primary probe",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.brickProbeLayout] }),
          compute: { module, entryPoint: SVO_BRICK_RASTER_PROBE_CONTRACT.entryPoint },
        });
        this.rebuildBrickRasterBindGroups();
      } catch (error) {
        // A missing primary probe degrades the diagnostic to its lighting half
        // rather than failing it: the ray probe still explains the shading.
        this.brickProbePipeline = undefined;
        this.brickProbeBuffers?.destroy();
        this.brickProbeBuffers = undefined;
        console.warn("Sparse voxel raster-primary probe unavailable", error);
      }
    })();
    await this.brickProbeCompilation;
  }

  /**
   * The entry prepass module: one emission kernel and one depth draw.
   *
   * Compiled beside the brick cull and for the same reason it is separate from
   * the megakernel — it reads the camera uniform, the published topology and the
   * `SvoMapping` prefix of `DryParams`, and nothing the fragment stage owns.
   */
  private async ensurePrimaryEntryPrepassPipelines(): Promise<void> {
    if (!this.primaryEntryPrepassEnabled || this.primaryEntryCullPipeline) return;
    this.primaryEntryCompilation ??= (async () => {
      this.primaryEntryCullLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel primary entry proxy emission bindings",
        entries: svoPrimaryEntryCullBindGroupLayoutEntries(),
      });
      this.primaryEntryDrawLayout = this.device.createBindGroupLayout({
        label: "Sparse voxel primary entry depth draw bindings",
        entries: svoPrimaryEntryDrawBindGroupLayoutEntries(),
      });
      const module = await checkedModule(this.device, "Sparse voxel primary entry prepass",
        createSvoPrimaryEntryPrepassWGSL({ reversedZNear_m: SVO_DRY_SCENE_REVERSED_Z_NEAR_M }));
      const [cull, draw] = await Promise.all([
        this.device.createComputePipelineAsync({
          label: "Sparse voxel primary entry proxy emission",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.primaryEntryCullLayout] }),
          compute: { module, entryPoint: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.entryPoints.emit },
        }),
        this.device.createRenderPipelineAsync({
          label: "Sparse voxel primary entry depth",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.primaryEntryDrawLayout] }),
          vertex: { module, entryPoint: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.entryPoints.vertex },
          fragment: {
            module, entryPoint: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.entryPoints.fragment,
            targets: [{ format: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.seedFormat }],
          },
          primitive: { topology: "triangle-list", cullMode: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.cullMode },
          // The same reversed-Z convention the G-buffer uses, on a private
          // depth plane: nearest entry wins under `greater`, and the primary
          // pass must be free to write its own farther surface depth after.
          depthStencil: {
            format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
            depthWriteEnabled: true,
            depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
          },
        }),
      ]);
      this.primaryEntryCullPipeline = cull;
      this.primaryEntryDrawPipeline = draw;
      this.ensurePrimaryEntryBuffers();
    })();
    await this.primaryEntryCompilation;
  }

  /** Instance arena sized by published leaf capacity, not by leaf count. */
  private ensurePrimaryEntryBuffers(): void {
    const leafCapacity = this.source?.structural?.capacities.leaves ?? 0;
    if (!this.primaryEntryPrepassEnabled || leafCapacity < 1
      || this.primaryEntryLeafCapacity === leafCapacity) return;
    this.primaryEntryPublicationBuffer?.destroy();
    this.primaryEntryPublicationBuffer = this.device.createBuffer({
      label: "Sparse voxel primary entry proxy publication (draw args and instances)",
      size: svoPrimaryEntryPublicationBytes(leafCapacity),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    // Word zero is the constant indirect vertex count; the frame clears word one
    // alone, so this is written exactly once per allocation.
    this.device.queue.writeBuffer(this.primaryEntryPublicationBuffer, 0,
      Uint32Array.of(SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.verticesPerInstance));
    this.primaryEntryLeafCapacity = leafCapacity;
    this.primaryEntryCullBindGroup = undefined;
    this.primaryEntryDrawBindGroup = undefined;
  }

  private rebuildPrimaryEntryBindGroups(): void {
    const structural = this.source?.structural;
    if (!this.primaryEntryPrepassEnabled || !structural || !this.primaryEntryCullLayout
      || !this.primaryEntryDrawLayout || !this.primaryEntryPublicationBuffer) return;
    const { bindings } = SVO_PRIMARY_ENTRY_PREPASS_CONTRACT;
    this.primaryEntryCullBindGroup = this.device.createBindGroup({
      label: "Sparse voxel primary entry proxy emission binding",
      layout: this.primaryEntryCullLayout,
      entries: [
        { binding: bindings.uniforms, resource: { buffer: this.uniformBuffer } },
        { binding: bindings.mapping, resource: { buffer: this.paramsBuffer, offset: 0, size: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.mappingBindingBytes } },
        { binding: bindings.structure, resource: structural.structure },
        { binding: bindings.publication, resource: { buffer: this.primaryEntryPublicationBuffer } },
      ],
    });
    this.primaryEntryDrawBindGroup = this.device.createBindGroup({
      label: "Sparse voxel primary entry depth draw binding",
      layout: this.primaryEntryDrawLayout,
      entries: [
        { binding: bindings.uniforms, resource: { buffer: this.uniformBuffer } },
        { binding: bindings.mapping, resource: { buffer: this.paramsBuffer, offset: 0, size: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.mappingBindingBytes } },
        { binding: bindings.instances, resource: { buffer: this.primaryEntryPublicationBuffer, offset: svoPrimaryEntryInstanceOffsetBytes() } },
      ],
    });
  }

  /** Every resource the seeded primary needs before the megakernel may read the plane. */
  private get primaryEntryPrepassReady(): boolean {
    return Boolean(this.primaryEntryCullPipeline && this.primaryEntryDrawPipeline
      && this.primaryEntryCullBindGroup && this.primaryEntryDrawBindGroup
      && this.primaryEntryPublicationBuffer && this.primaryEntrySeedView && this.primaryEntryDepthView);
  }

  /**
   * Emit one padded proxy per voxel leaf, then resolve the nearest entry per
   * pixel with the depth test.
   *
   * The clear is the pass's other half and is never skipped: an unwritten texel
   * is read as "no voxel leaf on this ray", which is only true because this pass
   * wrote the whole plane this frame.
   */
  private encodePrimaryEntryPrepass(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.primaryEntryPublicationBuffer!, Uint32Array.BYTES_PER_ELEMENT,
      Uint32Array.BYTES_PER_ELEMENT);
    const cull = encoder.beginComputePass({ label: "Sparse voxel primary entry proxy emission" });
    cull.setPipeline(this.primaryEntryCullPipeline!);
    cull.setBindGroup(0, this.primaryEntryCullBindGroup!);
    cull.dispatchWorkgroups(Math.ceil(this.primaryEntryLeafCapacity
      / SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.emitWorkgroupSize));
    cull.end();
    const depth = encoder.beginRenderPass({
      label: "Sparse voxel primary entry depth",
      colorAttachments: [{
        view: this.primaryEntrySeedView!,
        clearValue: { r: 0, g: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.emptyKey, b: 0, a: 0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.primaryEntryDepthView!,
        depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
        depthLoadOp: "clear",
        // Nothing downstream reads this plane; only the colour it resolved.
        depthStoreOp: "discard",
      },
    });
    depth.setPipeline(this.primaryEntryDrawPipeline!);
    depth.setBindGroup(0, this.primaryEntryDrawBindGroup!);
    depth.drawIndirect(this.primaryEntryPublicationBuffer!, 0);
    depth.end();
  }

  private rebuildBrickRasterBindGroups(): void {
    const structural = this.source?.structural;
    if (!this.rasterPrimary || !structural || !this.brickCullLayout || !this.brickDrawLayout
      || !this.brickCoverageLayout || !this.brickCoverageResolveLayout || !this.scenePrimitiveCoverageLayout
      || !this.brickResolveSceneLayout
      || !this.brickCandidateBuffer || !this.brickInstanceBuffer || !this.brickSortStateBuffer
      || !this.brickCoverageCountBuffer || !this.brickCoverageCandidateBuffer
      || !this.splitGlassKeyView || !this.splitGeometryView || !this.splitOpaqueIdentityView) return;
    const { bindings } = SVO_BRICK_RASTER_CONTRACT;
    this.brickCullBindGroup = this.device.createBindGroup({
      label: "Sparse voxel brick instance emission binding",
      layout: this.brickCullLayout,
      entries: [
        { binding: bindings.uniforms, resource: { buffer: this.uniformBuffer } },
        { binding: bindings.mapping, resource: { buffer: this.paramsBuffer, offset: 0, size: SVO_BRICK_RASTER_CONTRACT.mappingBindingBytes } },
        { binding: bindings.structure, resource: structural.structure },
        { binding: bindings.candidates, resource: { buffer: this.brickCandidateBuffer } },
        { binding: bindings.rasterPublication, resource: { buffer: this.brickRasterPublicationBuffer! } },
      ],
    });
    this.brickDrawBindGroup = this.device.createBindGroup({
      label: "Sparse voxel brick instance draw binding",
      layout: this.brickDrawLayout,
      entries: [
        { binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding, resource: { buffer: this.brickInstanceBuffer, offset: this.brickInstanceOffsetBytes } },
      ],
    });
    this.brickCoverageBindGroup = this.device.createBindGroup({
      label: "Sparse voxel conservative brick coverage binding",
      layout: this.brickCoverageLayout,
      entries: [
        { binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding, resource: { buffer: this.brickInstanceBuffer, offset: this.brickInstanceOffsetBytes } },
        { binding: SVO_BRICK_RASTER_CONTRACT.coverageCountBinding, resource: { buffer: this.brickCoverageCountBuffer } },
        { binding: SVO_BRICK_RASTER_CONTRACT.coverageCandidateBinding, resource: { buffer: this.brickCoverageCandidateBuffer } },
      ],
    });
    this.brickCoverageResolveBindGroup = this.device.createBindGroup({
      label: "Sparse voxel conservative brick resolve binding",
      layout: this.brickCoverageResolveLayout,
      entries: [
        { binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding, resource: { buffer: this.brickInstanceBuffer, offset: this.brickInstanceOffsetBytes } },
        { binding: SVO_BRICK_RASTER_CONTRACT.coverageCountBinding, resource: { buffer: this.brickCoverageCountBuffer } },
        { binding: SVO_BRICK_RASTER_CONTRACT.coverageCandidateBinding, resource: { buffer: this.brickCoverageCandidateBuffer } },
        { binding: SVO_BRICK_RASTER_CONTRACT.lodKeyBinding, resource: this.splitGlassKeyView },
      ],
    });
    this.scenePrimitiveCoverageBindGroup = this.device.createBindGroup({
      label: "Sparse voxel scene-primitive occlusion coverage binding",
      layout: this.scenePrimitiveCoverageLayout,
      entries: [
        { binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding, resource: { buffer: this.brickInstanceBuffer, offset: this.brickInstanceOffsetBytes } },
        { binding: SVO_BRICK_RASTER_CONTRACT.coverageCountBinding, resource: { buffer: this.brickCoverageCountBuffer } },
        { binding: SVO_BRICK_RASTER_CONTRACT.coverageCandidateBinding, resource: { buffer: this.brickCoverageCandidateBuffer } },
        { binding: SVO_BRICK_RASTER_CONTRACT.primaryGeometryBinding, resource: this.splitGeometryView! },
        { binding: SVO_BRICK_RASTER_CONTRACT.primaryIdentityBinding, resource: this.splitOpaqueIdentityView! },
      ],
    });
    if (this.bandComputeLayout && this.bandReadLayout && this.bandStateBuffer && this.bandParamsBuffer
      && this.coverageAuditBuffer && (this.screenSpaceTerminationPixels === 0 || this.scenePrimitiveComputeQueue)) {
      const band = SVO_SCENE_PRIMITIVE_BAND_CONTRACT.bindings;
      this.bandComputeBindGroup = this.device.createBindGroup({
        label: "Sparse voxel near-field band selection binding",
        layout: this.bandComputeLayout,
        entries: [
          { binding: band.uniforms, resource: { buffer: this.uniformBuffer } },
          // Mapping plus metadata: the projected voxel size needs the lattice's
          // cell size, and the record loop needs the published record count.
          { binding: band.params, resource: { buffer: this.paramsBuffer, offset: 0, size: SVO_BRICK_RASTER_CONTRACT.mappingBindingBytes + 16 } },
          { binding: band.scene, resource: { buffer: this.sceneArenaBuffer } },
          { binding: band.state, resource: { buffer: this.bandStateBuffer } },
          { binding: band.band, resource: { buffer: this.bandParamsBuffer } },
        ],
      });
      if (this.coverageOverflowArgsLayout && this.scenePrimitiveOverflowPublicationBuffer) {
        const overflow = SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.bindings;
        this.brickOverflowArgsBindGroup = this.device.createBindGroup({
          label: "Sparse voxel brick overflow draw-args binding",
          layout: this.coverageOverflowArgsLayout,
          entries: [
            { binding: overflow.publication, resource: { buffer: this.brickSortStateBuffer, offset: this.brickSortStateOffsetBytes } },
            { binding: overflow.coverageCounts, resource: { buffer: this.brickCoverageCountBuffer } },
          ],
        });
        this.scenePrimitiveOverflowArgsBindGroup = this.device.createBindGroup({
          label: "Sparse voxel scene-primitive overflow draw-args binding",
          layout: this.coverageOverflowArgsLayout,
          entries: [
            { binding: overflow.publication, resource: { buffer: this.scenePrimitiveOverflowPublicationBuffer } },
            { binding: overflow.coverageCounts, resource: { buffer: this.brickCoverageCountBuffer } },
          ],
        });
      }
      this.bandReadBindGroup = this.device.createBindGroup({
        label: "Sparse voxel near-field band membership binding",
        layout: this.bandReadLayout,
        entries: [
          { binding: SVO_SCENE_PRIMITIVE_BAND_CONTRACT.readGroupBinding, resource: { buffer: this.bandStateBuffer } },
          { binding: SVO_SCENE_PRIMITIVE_BAND_CONTRACT.auditGroupBinding, resource: { buffer: this.coverageAuditBuffer! } },
          ...(this.screenSpaceTerminationPixels > 0
            ? [{ binding: 8, resource: { buffer: this.scenePrimitiveComputeQueue! } }]
            : []),
        ],
      });
    }
    this.brickResolveSceneBindGroup = this.device.createBindGroup({
      label: "Sparse voxel conservative brick resolve scene binding",
      layout: this.brickResolveSceneLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bodyBuffer } },
        { binding: 2, resource: structural.structure },
        { binding: 3, resource: structural.scenePayload },
        { binding: 4, resource: { buffer: this.sceneArenaBuffer } },
        { binding: 6, resource: structural.planarBoundaries.records },
        { binding: 9, resource: { buffer: this.paramsBuffer } },
        { binding: 14, resource: { buffer: this.rigidMotionUniformBuffer } },
        { binding: 15, resource: { buffer: this.thickGlassUniformBuffer } },
      ],
    });
    // The primitive arena is republished with the scene, so the probe's group is
    // rebuilt here rather than at compile time: without it there is nothing for
    // the in-brick DDA to resolve an owner tag against.
    if (this.brickProbeLayout && this.brickProbeBuffers && this.probeBuffers) {
      const probe = SVO_BRICK_RASTER_PROBE_CONTRACT.bindings;
      this.brickProbeBindGroup = this.device.createBindGroup({
        label: "Sparse voxel raster-primary probe binding",
        layout: this.brickProbeLayout,
        entries: [
          { binding: probe.uniforms, resource: { buffer: this.uniformBuffer } },
          // The whole parameter block. The DDA needs the primitive base owner and
          // count immediately after the mapping prefix, and the payload lane bases
          // at its far end; the span between is reserved in the probe's struct.
          { binding: probe.params, resource: { buffer: this.paramsBuffer, offset: 0, size: SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes } },
          // One request buffer shared with the ray probe, so the two cannot
          // answer different pixels for the same frame.
          { binding: probe.request, resource: { buffer: this.probeBuffers.request } },
          { binding: probe.structure, resource: structural.structure },
          { binding: probe.scenePayload, resource: structural.scenePayload },
          { binding: probe.scene, resource: { buffer: this.sceneArenaBuffer } },
          { binding: probe.rasterPublication, resource: { buffer: this.brickRasterPublicationBuffer! } },
          { binding: probe.coverageCounts, resource: { buffer: this.brickCoverageCountBuffer! } },
          { binding: probe.coverageCandidates, resource: { buffer: this.brickCoverageCandidateBuffer! } },
          { binding: probe.records, resource: this.brickProbeBuffers.recordsView },
        ],
      });
    }
  }

  /**
   * Raster-assisted primary visibility.
   *
   * Background and terrain come first as one full-screen pass — it owns the
   * G-buffer clear and the exact miss encoding — and the brick instances are
   * then rasterized over it with the ordinary reversed-Z depth test. Because
   * octree leaves partition space, a ray meets each brick proxy over one
   * interval and those intervals are totally ordered, so the depth test alone
   * resolves visibility exactly regardless of submission order.
   */
  /** The four depth-tested primary planes every raster-primary pass writes. */
  private rasterPrimaryAttachments(
    gBufferViews: SparseVoxelGBufferViews, loadOp: GPULoadOp,
  ): GPURenderPassColorAttachment[] {
    return [
      { view: gBufferViews.packedSurface, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp, storeOp: "store" },
      { view: gBufferViews.identityMedia, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp, storeOp: "store" },
      { view: this.splitGeometryView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp, storeOp: "store" },
      { view: this.splitOpaqueIdentityView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp, storeOp: "store" },
    ];
  }

  /**
   * The band's authored controls, mirrored into their own tiny uniform.
   *
   * Deliberately not folded into `SVO_DRY_SCENE_PARAMS_LAYOUT`: that uniform is
   * exactly full at 576 bytes and two live tests pin its shape word for word,
   * and the band's consumer is a standalone compute module that binds nothing
   * else of it anyway.
   */
  /**
   * The instance count the authored-SDF overflow arm draws when it draws at all.
   *
   * Word one, because that is where the brick arm's scan publishes its own
   * culled count and one publisher module serves both.
   */
  /**
   * Publish the overflow pass's draw args between an arm's resolve and its
   * overflow draw. One workgroup, one lane, four words.
   */
  private encodeCoverageOverflowArgs(
    encoder: GPUCommandEncoder, bindGroup: GPUBindGroup | undefined, label: string,
  ): void {
    if (!this.coverageOverflowArgsPipeline || !bindGroup || !this.coverageOverflowIndirectBuffer) return;
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(this.coverageOverflowArgsPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    // Sixteen bytes out to a buffer whose only usage is INDIRECT. The overflow
    // pass binds the count buffer as writable storage — it reads its own pixel's
    // count — and a buffer may not be both writable storage and the indirect
    // source within one render pass's usage scope.
    encoder.copyBufferToBuffer(this.brickCoverageCountBuffer!,
      svoRasterCoverageOverflowDrawArgsOffsetBytes(this.brickCoverageWidth, this.brickCoverageHeight),
      this.coverageOverflowIndirectBuffer, 0, SVO_DRAW_INDIRECT_ARGS_BYTES);
  }

  private writeScenePrimitiveOverflowPublication(): void {
    if (!this.scenePrimitiveOverflowPublicationBuffer) return;
    const words = new Uint32Array(8);
    words[SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.publicationInstanceCountWord] = this.primitiveCount;
    this.device.queue.writeBuffer(this.scenePrimitiveOverflowPublicationBuffer, 0, words);
  }

  private writeBandParams(): void {
    if (!this.bandParamsBuffer) return;
    const tuning = this.renderTuning;
    this.device.queue.writeBuffer(this.bandParamsBuffer, 0, packSvoScenePrimitiveBandParams({
      enterPixels: tuning.nearFieldBandPixels,
      exitPixels: tuning.nearFieldBandPixels * tuning.nearFieldBandHysteresis,
      budgetRecords: tuning.nearFieldBandBudget,
    }));
    // With the band disabled every authored record is a member. Publish that
    // fail-open state once when tuning changes, then omit the per-frame
    // classify/resolve pair entirely. The record words deliberately survive
    // frame clears because positive thresholds use them as hysteresis state.
    if (tuning.nearFieldBandPixels === 0 && this.bandStateBuffer) {
      const membership = new Uint32Array(SVO_SCENE_PRIMITIVE_BAND_CONTRACT.maximumRecords);
      membership.fill(SVO_SCENE_PRIMITIVE_BAND_CONTRACT.memberBit);
      const recordOffsetBytes = (SVO_SCENE_PRIMITIVE_BAND_CONTRACT.headerWords
        + SVO_SCENE_PRIMITIVE_BAND_CONTRACT.buckets) * Uint32Array.BYTES_PER_ELEMENT;
      this.device.queue.writeBuffer(this.bandStateBuffer, recordOffsetBytes, membership);
    }
  }

  /** Whether the authored threshold currently removes any record from the analytic set. */
  get nearFieldBandActive(): boolean {
    return this.rasterPrimary && this.renderTuning.nearFieldBandPixels > 0
      && Boolean(this.bandClassifyPipeline && this.bandResolvePipeline && this.bandComputeBindGroup);
  }

  /**
   * Copies the band's header — cutoff bucket, candidates, admitted, highest
   * occupied bucket — for a lane that needs to report the band's actual size.
   *
   * The band never changes what a pixel is allowed to be, only which pass
   * resolves it, so its size is invisible in the image and this is the only way
   * to see it move.
   */
  copyScenePrimitiveBandHeader(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    const bytes = SVO_SCENE_PRIMITIVE_BAND_CONTRACT.headerWords * Uint32Array.BYTES_PER_ELEMENT;
    if (!this.bandStateBuffer || target.size < bytes) return false;
    encoder.copyBufferToBuffer(this.bandStateBuffer, 0, target, 0, bytes);
    return true;
  }

  /**
   * Authored SDF records, on the brick raster's coverage/resolve/overflow shape.
   *
   * The historical direct arm survives here for two reasons and no others: it is
   * the overflow fallback, and it is the control the coverage arm is measured
   * against (`rasterPrimaryDirect`, and the `scenePrimitiveHsrProbe` experiment
   * that drops its frag_depth to bracket what conservative depth would be worth
   * if WGSL had it).
   */
  private encodeScenePrimitivePrimary(
    encoder: GPUCommandEncoder,
    gBufferViews: SparseVoxelGBufferViews,
    usePrepass: boolean,
    splitGroup: number,
    tracePhase?: RenderFrameSeam<"svo">,
  ): void {
    const tieredComputeActive = this.screenSpaceTerminationPixels > 0
      && this.targetWidth * this.targetHeight >= SVO_SCENE_PRIMITIVE_COMPUTE_MINIMUM_PIXELS;
    const coverageReady = Boolean(this.scenePrimitiveCoveragePipeline
      && this.scenePrimitiveCoverageResolvePipeline && this.scenePrimitiveCoverageOverflowPipeline
      && (this.screenSpaceTerminationPixels === 0
        || (this.scenePrimitiveLodResolvePipeline && this.scenePrimitiveCoverageBindGroup
          && this.scenePrimitiveComputeArgsPipeline && this.scenePrimitiveComputeResolvePipeline
          && this.scenePrimitiveDepthBridgePipeline && this.scenePrimitiveComputeQueue
          && this.scenePrimitiveComputeIndirect
          && this.scenePrimitiveComputeOutputBindGroup && this.scenePrimitiveDepthBridgeBindGroup))
      && this.brickResolveSceneBindGroup && this.brickCoverageBindGroup
      && this.brickCoverageCountBuffer && this.splitGlassKeyView
      // The band's membership word is what the coverage vertex stage reads, so
      // an unbuilt band is a reason to fall back to the exact direct pass rather
      // than to draw a set nothing decided.
      && this.bandReadBindGroup && this.bandComputeBindGroup
      && this.bandClassifyPipeline && this.bandResolvePipeline);
    // Sixteen bits of the candidate key address the record. Past that the key
    // would alias one record onto another, so the arm degrades to the exact
    // direct pass rather than drawing the wrong surface.
    const addressable = this.primitiveCount <= SVO_SCENE_PRIMITIVE_COVERAGE_MAXIMUM_RECORDS;
    if (!addressable && !this.scenePrimitiveCoverageCapacityReported) {
      this.scenePrimitiveCoverageCapacityReported = true;
      console.warn(`Sparse voxel scene-primitive coverage disabled: ${this.primitiveCount} records exceed the `
        + `${SVO_SCENE_PRIMITIVE_COVERAGE_MAXIMUM_RECORDS}-record candidate key; the direct pass is drawing instead`);
    }
    if (this.scenePrimitiveDirect || !coverageReady || !addressable) {
      const exactScene = encoder.beginRenderPass({
        label: "Sparse voxel exact live-scene primitive visibility",
        colorAttachments: this.rasterPrimaryAttachments(gBufferViews, "load"),
        depthStencilAttachment: {
          view: gBufferViews.hardwareDepth,
          depthLoadOp: "load",
          depthStoreOp: "store",
        },
      });
      exactScene.setPipeline(this.scenePrimitiveRasterPipeline!);
      exactScene.setBindGroup(0, this.bindGroup);
      exactScene.draw(SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.verticesPerProxy, this.primitiveCount);
      exactScene.end();
      tracePhase?.("scene-primitive-visibility");
      return;
    }
    if (this.primitiveCount < 1) return;
    // The near-field band, before anything is drawn from it. Two dispatches over
    // records — 79 workgroups at the acceptance scene's 5 039 — decide which
    // records the two proxy passes below will even emit vertices for.
    //
    // The counters are cleared rather than accumulated, but the per-record state
    // words are not: last frame's membership is this frame's hysteresis input,
    // which is the whole reason the buffer outlives the frame.
    const bandBindGroup = this.bandReadBindGroup;
    if (this.nearFieldBandActive && this.bandClassifyPipeline && this.bandResolvePipeline
      && this.bandComputeBindGroup && bandBindGroup) {
      encoder.clearBuffer(this.bandStateBuffer!, 0, SVO_SCENE_PRIMITIVE_BAND_COUNTER_BYTES);
      const band = encoder.beginComputePass({ label: "Sparse voxel near-field analytic band selection" });
      band.setBindGroup(0, this.bandComputeBindGroup);
      band.setPipeline(this.bandClassifyPipeline);
      band.dispatchWorkgroups(Math.ceil(
        Math.min(this.primitiveCount, SVO_SCENE_PRIMITIVE_BAND_CONTRACT.maximumRecords)
        / SVO_SCENE_PRIMITIVE_BAND_CONTRACT.workgroupSize));
      // One workgroup: the histogram is 256 words and the budget is a scan over
      // it, so the resolve is a single group that then marks every record.
      band.setPipeline(this.bandResolvePipeline);
      band.dispatchWorkgroups(1);
      band.end();
      tracePhase?.("near-field-band");
    }
    // The brick arm is finished with the arena — its overflow pass has already
    // read the counters — so both coverage passes share one allocation.
    // Pixel range and the overflow flag, not the draw-args block: the publisher
    // below rewrites the args every frame, and clearing them here would only
    // hand the overflow pass a zero vertex count if the publisher ever failed.
    encoder.clearBuffer(this.brickCoverageCountBuffer!, 0,
      svoRasterCoverageCountAllocationBytes(this.brickCoverageWidth, this.brickCoverageHeight)
      - SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.drawArgsWord * Uint32Array.BYTES_PER_ELEMENT);
    encoder.clearBuffer(this.coverageAuditBuffer!);
    if (tieredComputeActive) {
      // Reset only the four-word queue header. Pixel storage is count-delimited.
      encoder.clearBuffer(this.scenePrimitiveComputeQueue!, 0, 4 * Uint32Array.BYTES_PER_ELEMENT);
      // The exact kernel writes only queued pixels, so clear last frame's depth
      // before the full-screen depth-only bridge samples this scratch plane.
      const clearTieredDepth = encoder.beginRenderPass({
        label: "Sparse voxel tiered scene-primitive depth clear",
        colorAttachments: [{
          view: this.scenePrimitiveComputeDepthView!,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      clearTieredDepth.end();
    }
    this.pollCoverageAudit();
    const coverage = encoder.beginRenderPass({
      label: "Sparse voxel conservative live-scene primitive coverage",
      colorAttachments: [{
        view: this.splitGlassKeyView!, clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear", storeOp: this.screenSpaceTerminationPixels > 0 ? "store" : "discard",
      }],
      ...(this.screenSpaceTerminationPixels > 0 ? { depthStencilAttachment: {
        view: this.splitGlassDepthView!,
        depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
        depthLoadOp: "clear" as const,
        depthStoreOp: "store" as const,
      } } : {}),
    });
    coverage.setPipeline(this.scenePrimitiveCoveragePipeline!);
    coverage.setBindGroup(0, this.brickResolveSceneBindGroup!);
    if (usePrepass) coverage.setBindGroup(1, this.conePrepassBindGroup!);
    coverage.setBindGroup(splitGroup, this.screenSpaceTerminationPixels > 0
      ? this.scenePrimitiveCoverageBindGroup! : this.brickCoverageBindGroup!);
    coverage.setBindGroup(splitGroup + 1, bandBindGroup!);
    coverage.draw(SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.verticesPerProxy, this.primitiveCount);
    coverage.end();

    if (this.screenSpaceTerminationPixels > 0) {
      const lod = encoder.beginRenderPass({
        label: "Sparse voxel resident-record LOD resolve",
        colorAttachments: this.rasterPrimaryAttachments(gBufferViews, "load"),
        depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthLoadOp: "load", depthStoreOp: "store" },
      });
      lod.setPipeline(this.scenePrimitiveLodResolvePipeline!);
      lod.setBindGroup(0, this.brickResolveSceneBindGroup!);
      if (usePrepass) lod.setBindGroup(1, this.conePrepassBindGroup!);
      lod.setBindGroup(splitGroup, this.brickCoverageResolveBindGroup!);
      lod.setBindGroup(splitGroup + 1, bandBindGroup!);
      lod.draw(3);
      lod.end();
    }

    if (tieredComputeActive) {
      const args = encoder.beginComputePass({ label: "Sparse voxel tiered scene-primitive dispatch args" });
      args.setPipeline(this.scenePrimitiveComputeArgsPipeline!);
      args.setBindGroup(0, this.brickResolveSceneBindGroup!);
      if (usePrepass) args.setBindGroup(1, this.conePrepassBindGroup!);
      args.setBindGroup(splitGroup, this.brickCoverageBindGroup!);
      args.setBindGroup(splitGroup + 1, this.scenePrimitiveComputeOutputBindGroup!);
      args.dispatchWorkgroups(1);
      args.end();
      encoder.copyBufferToBuffer(this.scenePrimitiveComputeQueue!, 0,
        this.scenePrimitiveComputeIndirect!, 0, 3 * Uint32Array.BYTES_PER_ELEMENT);
      const resolve = encoder.beginComputePass({ label: "Sparse voxel tiered scene-primitive compute resolve" });
      resolve.setPipeline(this.scenePrimitiveComputeResolvePipeline!);
      resolve.setBindGroup(0, this.brickResolveSceneBindGroup!);
      if (usePrepass) resolve.setBindGroup(1, this.conePrepassBindGroup!);
      resolve.setBindGroup(splitGroup, this.brickCoverageBindGroup!);
      resolve.setBindGroup(splitGroup + 1, this.scenePrimitiveComputeOutputBindGroup!);
      resolve.dispatchWorkgroupsIndirect(this.scenePrimitiveComputeIndirect!, 0);
      resolve.end();
      const depthBridge = encoder.beginRenderPass({
        label: "Sparse voxel tiered scene-primitive depth bridge",
        colorAttachments: [],
        depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthLoadOp: "load", depthStoreOp: "store" },
      });
      depthBridge.setPipeline(this.scenePrimitiveDepthBridgePipeline!);
      depthBridge.setBindGroup(0, this.scenePrimitiveDepthBridgeBindGroup!);
      depthBridge.draw(3);
      depthBridge.end();
    } else {
      const resolve = encoder.beginRenderPass({
        label: "Sparse voxel conservative live-scene primitive resolve",
        colorAttachments: this.rasterPrimaryAttachments(gBufferViews, "load"),
        depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthLoadOp: "load", depthStoreOp: "store" },
      });
      resolve.setPipeline(this.scenePrimitiveCoverageResolvePipeline!);
      resolve.setBindGroup(0, this.brickResolveSceneBindGroup!);
      if (usePrepass) resolve.setBindGroup(1, this.conePrepassBindGroup!);
      resolve.setBindGroup(splitGroup, this.brickCoverageBindGroup!);
      resolve.setBindGroup(splitGroup + 1, bandBindGroup!);
      resolve.draw(3);
      resolve.end();
    }
    if (this.screenSpaceTerminationPixels === 0) {
      this.encodeCoverageOverflowArgs(encoder, this.scenePrimitiveOverflowArgsBindGroup,
        "Sparse voxel scene-primitive overflow draw args");

      const overflow = encoder.beginRenderPass({
        label: "Sparse voxel conservative live-scene primitive overflow",
        colorAttachments: this.rasterPrimaryAttachments(gBufferViews, "load"),
        depthStencilAttachment: {
          view: gBufferViews.hardwareDepth,
          depthLoadOp: "load",
          depthStoreOp: "store",
        },
      });
      overflow.setPipeline(this.scenePrimitiveCoverageOverflowPipeline!);
      overflow.setBindGroup(0, this.brickResolveSceneBindGroup!);
      if (usePrepass) overflow.setBindGroup(1, this.conePrepassBindGroup!);
      overflow.setBindGroup(splitGroup, this.brickCoverageBindGroup!);
      overflow.setBindGroup(splitGroup + 1, bandBindGroup!);
      overflow.drawIndirect(this.coverageOverflowIndirectBuffer!, 0);
      overflow.end();
    }
    // One sampled snapshot a second, staged for the *next* encode to read: by
    // then the caller has submitted this frame, so the map resolves against real
    // numbers rather than an unsubmitted copy.
    this.coverageAuditFrame += 1;
    if (!this.coverageAuditCopied && !this.coverageAuditReading
      && this.coverageAuditFrame % SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_PERIOD_FRAMES === 0
      && this.coverageAuditStaging) {
      encoder.copyBufferToBuffer(this.coverageAuditBuffer!, 0, this.coverageAuditStaging, 0,
        SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_BYTES);
      this.coverageAuditCopied = true;
    }
    tracePhase?.("scene-primitive-visibility");
  }

  /**
   * The arena-pressure tripwire.
   *
   * W1 gated at "overflow rate < 1 % of covered pixels" and measured 0.0 % at
   * the hero's 501 records. It is still 0 % at 5 039 — but the busiest pixel is
   * 37 of 40 at 8x, so a wider viewport or a closer camera crosses it, and the
   * degrade is a second full re-march of every covering proxy on those pixels.
   * Capacity remaining a performance parameter is the guarantee; nothing was
   * reporting when the performance stopped being good.
   *
   * The rate is estimated from the sampled counters rather than from the exact
   * tail word, because covered pixels are only counted on the sample and mixing
   * an exact numerator with a sampled denominator would inflate it by the
   * stride. `copyCoverageOverflowCount` exposes the exact number beside it.
   */
  private pollCoverageAudit(): void {
    if (!this.coverageAuditCopied || this.coverageAuditReading || !this.coverageAuditStaging) return;
    this.coverageAuditCopied = false;
    this.coverageAuditReading = true;
    const staging = this.coverageAuditStaging;
    void staging.mapAsync(GPUMapMode.READ).then(() => {
      const words = new Uint32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      this.coverageAuditReading = false;
      const audit = readSvoScenePrimitiveCoverageAudit(words);
      this.lastCoverageAudit = audit;
      const status = svoRasterCoverageOverflowStatus({
        coveredPixels: audit.coveredPixels,
        overflowPixels: audit.overflowedPixels,
        candidatesPerPixel: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel,
        arm: "Sparse voxel scene-primitive",
      });
      if (status.state !== "over-budget") {
        this.coverageOverflowReported = false;
        return;
      }
      if (this.coverageOverflowReported) return;
      this.coverageOverflowReported = true;
      console.warn(`${status.message} (mean ${audit.meanCandidates.toFixed(1)} candidates, `
        + `max ${audit.maximumCandidates}, 1-in-${audit.samplingStride} pixel sample of `
        + `${this.primitiveCount} records)`);
    }).catch(() => { this.coverageAuditReading = false; });
  }

  /**
   * Exact count of pixels whose candidate list exceeded capacity last frame.
   *
   * One word rather than a width-by-height scan, which is what makes the rate
   * cheap enough to report every frame. It is also the word the overflow pass's
   * indirect instance count is derived from, so a non-zero reading is exactly
   * the condition under which that pass draws at all.
   */
  copyCoverageOverflowCount(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    if (!this.brickCoverageCountBuffer || !this.brickCoverageWidth || !this.brickCoverageHeight) return false;
    if (target.size < Uint32Array.BYTES_PER_ELEMENT) return false;
    const offset = svoBrickRasterCoverageCountBytes(this.brickCoverageWidth, this.brickCoverageHeight)
      + SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.overflowPixelWord * Uint32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(this.brickCoverageCountBuffer, offset, target, 0, Uint32Array.BYTES_PER_ELEMENT);
    return true;
  }

  /** The brick cull's published sort state: draw args, candidate, culled, empty, resident. */
  copyBrickSortState(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    if (!this.brickSortStateBuffer) return false;
    const bytes = SVO_BRICK_RASTER_CONTRACT.sortStateHeaderWords * Uint32Array.BYTES_PER_ELEMENT;
    if (target.size < bytes) return false;
    encoder.copyBufferToBuffer(this.brickSortStateBuffer, this.brickSortStateOffsetBytes, target, 0, bytes);
    return true;
  }

  /** The most recent sampled arena-pressure reading, if one has completed. */
  lastCoverageAudit?: SvoScenePrimitiveCoverageAudit;

  /** Copies the sampled arena-pressure counters for an offline lane. */
  copyScenePrimitiveCoverageAudit(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    if (!this.coverageAuditBuffer || target.size < SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_BYTES) return false;
    encoder.copyBufferToBuffer(this.coverageAuditBuffer, 0, target, 0, SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_BYTES);
    return true;
  }

  private encodeRasterPrimary(
    encoder: GPUCommandEncoder,
    gBufferViews: SparseVoxelGBufferViews,
    usePrepass: boolean,
    splitGroup: number,
    tracePhase?: RenderFrameSeam<"svo">,
  ): void {
    if (this.experiments.surfaceMesh) {
      this.encodeSurfaceMesh(encoder, gBufferViews, usePrepass, splitGroup, tracePhase);
      return;
    }
    // Word zero is the constant indirect vertex count; everything after it is
    // per-frame state.
    encoder.clearBuffer(this.brickSortStateBuffer!, this.brickSortStateOffsetBytes + Uint32Array.BYTES_PER_ELEMENT,
      svoBrickRasterSortStateBytes() - Uint32Array.BYTES_PER_ELEMENT);
    const cull = encoder.beginComputePass({ label: "Sparse voxel brick instance cull" });
    cull.setBindGroup(0, this.brickCullBindGroup!);
    cull.setPipeline(this.brickEmitPipeline!);
    cull.dispatchWorkgroups(Math.ceil(this.brickLeafCapacity / SVO_BRICK_RASTER_CONTRACT.emitWorkgroupSize));
    cull.setPipeline(this.brickScanPipeline!);
    cull.dispatchWorkgroups(1);
    cull.setPipeline(this.brickScatterPipeline!);
    cull.dispatchWorkgroups(Math.ceil(this.brickLeafCapacity / SVO_BRICK_RASTER_CONTRACT.scatterWorkgroupSize));
    cull.end();
    tracePhase?.("brick-cull");

    const attachments = (loadOp: GPULoadOp): GPURenderPassColorAttachment[] =>
      this.rasterPrimaryAttachments(gBufferViews, loadOp);
    if (!this.rasterPrimaryDirect) {
      // Coverage is conservative and cheap: clear only counters, append the
      // sorted instance index for each proxy/pixel overlap, and stop before any
      // payload trace. Discard here only suppresses the overflow-mask colour
      // write; it follows the candidate append and precedes all expensive work.
      encoder.clearBuffer(this.brickCoverageCountBuffer!, 0,
        svoRasterCoverageCountAllocationBytes(this.brickCoverageWidth, this.brickCoverageHeight)
        - SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.drawArgsWord * Uint32Array.BYTES_PER_ELEMENT);
      const coverage = encoder.beginRenderPass({
        label: "Sparse voxel primary conservative brick coverage",
        colorAttachments: [{
          view: this.splitGlassKeyView!, clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear", storeOp: this.screenSpaceTerminationPixels > 0 ? "store" : "discard",
        }],
        ...(this.screenSpaceTerminationPixels > 0 ? { depthStencilAttachment: {
          view: this.splitGlassDepthView!,
          depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
          depthLoadOp: "clear" as const,
          depthStoreOp: "store" as const,
        } } : {}),
      });
      coverage.setPipeline(this.brickCoveragePipeline!);
      coverage.setBindGroup(0, this.brickResolveSceneBindGroup!);
      if (usePrepass) coverage.setBindGroup(1, this.conePrepassBindGroup!);
      coverage.setBindGroup(splitGroup, this.brickCoverageBindGroup!);
      coverage.drawIndirect(this.brickSortStateBuffer!, this.brickSortStateOffsetBytes);
      coverage.end();

      if (this.screenSpaceTerminationPixels > 0) {
        // Establish the plane with the small terrain/sky entry point first.
        // The next two draws load it and only touch their own classified pixels;
        // in particular, the exact marcher is unreachable from the far LOD arm.
        const background = encoder.beginRenderPass({
          label: "Sparse voxel primary background and terrain",
          colorAttachments: attachments("clear"),
          depthStencilAttachment: {
            view: gBufferViews.hardwareDepth,
            depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        background.setPipeline(this.brickBackgroundPipeline!);
        background.setBindGroup(0, this.bindGroup);
        if (usePrepass) background.setBindGroup(1, this.conePrepassBindGroup!);
        background.draw(3);
        background.end();

        const lod = encoder.beginRenderPass({
          label: "Sparse voxel primary resident-cell LOD resolve",
          colorAttachments: attachments("load"),
          depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthLoadOp: "load", depthStoreOp: "store" },
        });
        lod.setPipeline(this.brickLodResolvePipeline!);
        lod.setBindGroup(0, this.brickResolveSceneBindGroup!);
        if (usePrepass) lod.setBindGroup(1, this.conePrepassBindGroup!);
        lod.setBindGroup(splitGroup, this.brickCoverageResolveBindGroup!);
        lod.draw(3);
        lod.end();

        const exact = encoder.beginRenderPass({
          label: "Sparse voxel primary exact-only coverage resolve",
          colorAttachments: attachments("load"),
          depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthLoadOp: "load", depthStoreOp: "store" },
        });
        exact.setPipeline(this.brickExactResolvePipeline!);
        exact.setBindGroup(0, this.brickResolveSceneBindGroup!);
        if (usePrepass) exact.setBindGroup(1, this.conePrepassBindGroup!);
        exact.setBindGroup(splitGroup, this.brickCoverageBindGroup!);
        exact.draw(3);
        exact.end();
      } else {
        // Threshold-zero reference: one exact fragment per pixel resolves
        // front-to-back candidates and publishes the scene-primitive seed.
        const resolve = encoder.beginRenderPass({
          label: "Sparse voxel primary conservative coverage resolve",
          colorAttachments: attachments("clear"),
          depthStencilAttachment: {
            view: gBufferViews.hardwareDepth,
            depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        resolve.setPipeline(this.brickCoverageResolvePipeline!);
        resolve.setBindGroup(0, this.brickResolveSceneBindGroup!);
        if (usePrepass) resolve.setBindGroup(1, this.conePrepassBindGroup!);
        resolve.setBindGroup(splitGroup, this.brickCoverageResolveBindGroup!);
        resolve.draw(3);
        resolve.end();
      }
      this.encodeCoverageOverflowArgs(encoder, this.brickOverflowArgsBindGroup,
        "Sparse voxel brick overflow draw args");

      // Capacity never changes the image. Only pixels whose conservative list
      // overflowed re-run the historical direct brick fragment in this
      // isolated pipeline; garden's 24-entry arena exceeds the measured max=18.
      // The draw is overflow-driven: with no overflowing pixel the published
      // instance count is zero and the pass rasterizes nothing.
      const overflow = encoder.beginRenderPass({
        label: "Sparse voxel primary conservative coverage overflow",
        colorAttachments: attachments("load"),
        depthStencilAttachment: {
          view: gBufferViews.hardwareDepth,
          depthLoadOp: "load",
          depthStoreOp: "store",
        },
      });
      overflow.setPipeline(this.brickCoverageOverflowPipeline!);
      overflow.setBindGroup(0, this.brickResolveSceneBindGroup!);
      if (usePrepass) overflow.setBindGroup(1, this.conePrepassBindGroup!);
      overflow.setBindGroup(splitGroup, this.brickCoverageBindGroup!);
      overflow.drawIndirect(this.coverageOverflowIndirectBuffer!, 0);
      overflow.end();
      // Consume the selected pixel's candidate list before the authored-scene
      // coverage pass reuses this arena. This turns the common diagnostic path
      // from O(all visible bricks) into O(overlap at one pixel), bounded by 24.
      this.encodeBrickRasterProbe(encoder);
      return;
    }
    // Terrain and bricks stay in separate render passes even though they share
    // every attachment, which looks like a wasted tile flush of the 48-byte
    // G-buffer and is not: merging them measured ~4 ms/frame slower at
    // 1500x1500. Apple's tiler overlaps one pass's fragment work with the next
    // pass's binning, and a single pass serialises the full-screen terrain
    // triangle against the indirect brick draw instead.
    const background = encoder.beginRenderPass({
      label: "Sparse voxel primary background and terrain",
      colorAttachments: attachments("clear"),
      depthStencilAttachment: {
        view: gBufferViews.hardwareDepth,
        depthClearValue: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    background.setPipeline(this.brickBackgroundPipeline!);
    background.setBindGroup(0, this.bindGroup);
    if (usePrepass) background.setBindGroup(1, this.conePrepassBindGroup!);
    background.draw(3);
    background.end();

    const bricks = encoder.beginRenderPass({
      label: "Sparse voxel primary brick raster",
      colorAttachments: attachments("load"),
      depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthLoadOp: "load", depthStoreOp: "store" },
    });
    bricks.setPipeline(this.brickRasterPipeline!);
    bricks.setBindGroup(0, this.bindGroup);
    if (usePrepass) bricks.setBindGroup(1, this.conePrepassBindGroup!);
    bricks.setBindGroup(splitGroup, this.brickDrawBindGroup!);
    bricks.drawIndirect(this.brickSortStateBuffer!, this.brickSortStateOffsetBytes);
    bricks.end();
    // The direct experiment has no candidate arena to reuse; its probe retains
    // the global scan, but still runs beside the primary whose list it reads.
    this.encodeBrickRasterProbe(encoder);
    // The caller closes "svo-primary" straight after this returns, so the brick
    // raster lands under the same phase id the traced primary reports.
  }

  private async ensureSplitPipelines(scale: SvoConeLightingScale): Promise<void> {
    if (this.shadingPath === "inline" || !this.layout || !this.vertexModule) return;
    // The variant this call is for, captured now: lighting options can flip
    // while the compile is in flight, and a bundle must only activate if it is
    // still the variant the frame wants.
    const globalIlluminationCapable = this.lightingOptions.globalIlluminationEnabled === true;
    const variantKey = this.splitVariantKey(scale, globalIlluminationCapable);
    const variantCurrent = () => scale === this.coneScale
      && globalIlluminationCapable === (this.lightingOptions.globalIlluminationEnabled === true);
    const cached = this.splitPipelineBundles.get(variantKey);
    if (cached) {
      if (variantCurrent()) this.activateSplitPipelineBundle(scale, cached);
      return;
    }
    const pending = this.splitPipelineCompiles.get(variantKey);
    if (pending) {
      const bundle = await pending;
      if (variantCurrent()) this.activateSplitPipelineBundle(scale, bundle);
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
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.worldGiCacheDirty = true;
    }
    this.splitVisibilityLayout ??= this.device.createBindGroupLayout({
      label: "Sparse voxel split visibility outputs",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, storageTexture: { access: "write-only", format: SVO_DRY_SPLIT_GEOMETRY_FORMAT } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, storageTexture: { access: "write-only", format: SVO_DRY_SPLIT_IDENTITY_FORMAT } },
        ...(this.experiments.primaryWorkMap
          ? [{ binding: 7, visibility: GPUShaderStage.FRAGMENT,
            storageTexture: { access: "write-only" as const, format: "rgba32uint" as const } }]
          : []),
        // The entry-depth seed rides the visibility group because only the
        // visibility entry point reaches it: no lighting or compute-resolve
        // pipeline carries this layout, so no secondary ray can consult it.
        ...(this.primaryEntryPrepassEnabled
          ? [{ binding: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.seedBinding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" as const } }]
          : []),
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
    if (this.rasterPrimary && this.screenSpaceTerminationPixels > 0
      && this.device.limits.maxStorageTexturesPerShaderStage >= 5) {
      this.scenePrimitiveComputeOutputLayout ??= this.device.createBindGroupLayout({
        label: "Sparse voxel tiered compute resolve outputs",
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.packedSurfaceFormat } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.identityMediaFormat } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_DRY_SPLIT_GEOMETRY_FORMAT } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_DRY_SPLIT_IDENTITY_FORMAT } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float" } },
          { binding: SVO_SCENE_PRIMITIVE_BAND_CONTRACT.readGroupBinding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: SVO_SCENE_PRIMITIVE_BAND_CONTRACT.auditGroupBinding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
      });
      this.scenePrimitiveDepthBridgeLayout ??= this.device.createBindGroupLayout({
        label: "Sparse voxel tiered compute depth bridge",
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }],
      });
    }
    const voxelLightCacheEnabled = this.experiments.voxelLightCache !== false
      && this.device.limits.maxSampledTexturesPerShaderStage >= 17;
    if (voxelLightCacheEnabled && !this.voxelLightConsumerLayout) {
      this.voxelLightConsumerLayout = this.device.createBindGroupLayout({ label: "Sparse voxel directional-light cache consumer", entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ] });
      this.voxelLightDemandLayout = this.device.createBindGroupLayout({ label: "Sparse voxel directional-light cache demand", entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
      this.voxelLightPopulateLayout = this.device.createBindGroupLayout({ label: "Sparse voxel directional-light cache population", entries: [
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.format, viewDimension: "3d" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ] });
    }
    const layout = this.layout;
    const vertexModule = this.vertexModule;
    const shaderExperimentsBase = scale === 1 && this.experiments.halfPrecisionLighting
      ? { ...this.experiments, halfPrecisionLighting: false }
      : this.experiments;
    const shaderExperimentsPruned = voxelLightCacheEnabled
      ? shaderExperimentsBase
      : { ...shaderExperimentsBase, voxelLightCache: false };
    // The variant this bundle IS: GI disabled compiles the gather out rather
    // than leaving a never-taken branch priced into every deferred pixel.
    const shaderExperiments = globalIlluminationCapable
      ? shaderExperimentsPruned
      : { ...shaderExperimentsPruned, globalIlluminationAbsent: true };
    const compile = (async (): Promise<SvoDrySplitPipelineBundle> => {
      const [module, rasterRigidModule] = await Promise.all([
        checkedModule(this.device, `Sparse voxel dry scene split x${scale} (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
          createSvoDrySceneFragmentWGSL(scale, this.traversalMode, this.brickOccupancyMode, "split", this.screenSpaceTerminationPixels, false,
            this.rasterGlassDiscovery, false, this.coneFanout && scale !== 1, shaderExperiments)),
        this.rasterRigidDiscovery
          ? checkedModule(this.device, `Sparse voxel dry scene raster-rigid split x${scale} (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
            createSvoDrySceneFragmentWGSL(scale, this.traversalMode, this.brickOccupancyMode, "split", this.screenSpaceTerminationPixels, false,
              this.rasterGlassDiscovery, true, this.coneFanout && scale !== 1, shaderExperiments))
          : Promise.resolve(undefined),
      ]);
      const middleLayouts = scale === 1 ? [] : [this.conePrepassLayout!];
      const cacheConsumerLayouts = voxelLightCacheEnabled ? [this.voxelLightConsumerLayout!] : [];
      const visibilityLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitVisibilityLayout] });
      const [visibility, rasterRigidVisibility, primarySeamClosure, lighting, reconstructedLighting, skyLighting, prepassReset, prepassCoherent, prepassBoundary,
        worldGiFrame, worldGiCache, voxelLightDemand, voxelLightPopulate] = await Promise.all([
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
        label: `Sparse voxel primary seam closure x${scale}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitLightingLayout, ...cacheConsumerLayouts] }),
        vertex: { module: vertexModule, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "dryPrimarySeamMain", targets: [
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
      this.device.createRenderPipelineAsync({
        label: `Sparse voxel deferred dry lighting x${scale}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitLightingLayout, ...cacheConsumerLayouts] }),
        vertex: { module: vertexModule, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "dryLightingMain", targets: [{ format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat }] },
        primitive: { topology: "triangle-list" },
        // Shade only the pixels primary visibility actually resolved to a
        // surface. The full-screen triangle sits at device depth zero, which
        // under reversed-Z is the same value a miss writes, so `less` passes
        // exactly where the depth buffer holds a surface. Nothing here writes
        // frag_depth or discards, so the sky is rejected before the fragment
        // shader runs rather than after it — this shader is the frame's least
        // occupied pass, and every lane it does not spend on sky is one it can
        // spend hiding the G-buffer latency that limits it.
        depthStencil: {
          format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          depthWriteEnabled: false,
          depthCompare: "less",
        },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createRenderPipelineAsync({
        label: `Sparse voxel reconstructed dry lighting x${scale}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitLightingLayout, ...cacheConsumerLayouts] }),
        vertex: { module: vertexModule, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "dryReconstructedLightingMain", targets: [{ format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat }] },
        primitive: { topology: "triangle-list" },
        depthStencil: {
          format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          depthWriteEnabled: false,
          depthCompare: "less",
        },
      }),
      this.device.createRenderPipelineAsync({
        label: `Sparse voxel deferred sky lighting x${scale}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitLightingLayout, ...cacheConsumerLayouts] }),
        vertex: { module: vertexModule, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "drySkyLightingMain", targets: [{ format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat }] },
        primitive: { topology: "triangle-list" },
        // The exact complement of the surface test, so the two draws partition
        // the frame and every pixel is written once. Glass still has to be
        // resolved here: a thin pane in front of open sky carries no primary
        // depth of its own, so sky pixels are not unconditionally background.
        depthStencil: {
          format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          depthWriteEnabled: false,
          depthCompare: "greater-equal",
        },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel compact cone queue reset",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassComputeLayout!, this.splitLightingLayout!, ...cacheConsumerLayouts] }),
        compute: { module, entryPoint: "dryPrepassResetMain" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel coherent cone visibility from primary hits",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassComputeLayout!, this.splitLightingLayout!, ...cacheConsumerLayouts] }),
        compute: { module, entryPoint: "dryPrepassCoherentMain" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel compact boundary cone visibility",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassComputeLayout!, this.splitLightingLayout!, ...cacheConsumerLayouts] }),
        compute: { module, entryPoint: "dryPrepassBoundaryMain" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel world GI frame prelude",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassShadeLayout!, this.worldGiCacheLayout!, ...cacheConsumerLayouts] }),
        compute: { module, entryPoint: "dryWorldGiFrameMain" },
      }),
      scale === 1 ? Promise.resolve(undefined) : this.device.createComputePipelineAsync({
        label: "Sparse voxel persistent world GI cache",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.conePrepassShadeLayout!, this.worldGiCacheLayout!, ...cacheConsumerLayouts] }),
        compute: { module, entryPoint: "dryWorldGiCacheMain" },
      }),
      voxelLightCacheEnabled ? this.device.createComputePipelineAsync({
        label: "Sparse voxel directional-light cache demand",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitLightingLayout!, this.voxelLightDemandLayout!] }),
        compute: { module, entryPoint: "dryVoxelLightDemandMain" },
      }) : Promise.resolve(undefined),
      voxelLightCacheEnabled ? this.device.createComputePipelineAsync({
        label: "Sparse voxel directional-light cache population",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitLightingLayout!, this.voxelLightPopulateLayout!] }),
        compute: { module, entryPoint: "dryVoxelLightPopulateMain" },
      }) : Promise.resolve(undefined),
      ]);
      const coverageLayouts = () => this.device.createPipelineLayout({
        bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageLayout!],
      });
      // The scene-primitive arm carries one group more than the brick arm: the
      // band's membership words, which its vertex stage reads and the brick
      // proxies have no use for.
      const bandedCoverageLayouts = () => this.device.createPipelineLayout({
        bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageLayout!, this.bandReadLayout!],
      });
      const bandedCoverageResolveLayouts = () => this.device.createPipelineLayout({
        bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageResolveLayout!, this.bandReadLayout!],
      });
      const bandedOcclusionCoverageLayouts = () => this.device.createPipelineLayout({
        bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.scenePrimitiveCoverageLayout!, this.bandReadLayout!],
      });
      const [brickBackground, brickRaster, brickCoverage, brickCoverageResolve,
        brickLodResolve, brickExactResolve, brickCoverageOverflow, scenePrimitiveRaster,
        scenePrimitiveCoverage, scenePrimitiveLodResolve, scenePrimitiveCoverageResolve, scenePrimitiveCoverageOverflow] = this.rasterPrimary
        ? await Promise.all([
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel primary background and terrain",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts] }),
            vertex: { module: vertexModule, entryPoint: "vertexMain" },
            fragment: { module, entryPoint: SVO_BRICK_RASTER_CONTRACT.entryPoints.background, targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list" },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              // This pass establishes the G-buffer, including the miss encoding
              // the deferred lighting pass turns into sky. A miss writes device
              // depth zero, which would fail the reversed-Z greater test against
              // its own clear and leave the cleared planes behind.
              depthCompare: "always",
            },
          }),
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel primary brick raster",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.brickDrawLayout!] }),
            vertex: { module, entryPoint: SVO_BRICK_RASTER_CONTRACT.entryPoints.vertex },
            fragment: { module, entryPoint: SVO_BRICK_RASTER_CONTRACT.entryPoints.fragment, targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list", cullMode: SVO_BRICK_RASTER_CONTRACT.cullMode },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            },
          }),
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel primary conservative brick coverage",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageLayout!] }),
            vertex: { module, entryPoint: SVO_BRICK_RASTER_CONTRACT.entryPoints.vertex },
            fragment: { module, entryPoint: SVO_BRICK_RASTER_CONTRACT.entryPoints.coverage, targets: [{ format: "r32uint" }] },
            primitive: { topology: "triangle-list", cullMode: SVO_BRICK_RASTER_CONTRACT.cullMode },
            ...(this.screenSpaceTerminationPixels > 0 ? { depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            } } : {}),
          }),
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel primary conservative coverage resolve",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageResolveLayout!] }),
            vertex: { module: vertexModule, entryPoint: "vertexMain" },
            fragment: { module, entryPoint: SVO_BRICK_RASTER_CONTRACT.entryPoints.resolve, targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list" },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: "always",
            },
          }),
          this.screenSpaceTerminationPixels > 0 ? this.device.createRenderPipelineAsync({
            label: "Sparse voxel primary resident-cell LOD resolve",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageResolveLayout!] }),
            vertex: { module: vertexModule, entryPoint: "vertexMain" },
            fragment: { module, entryPoint: "svoBrickLodResolveFragment", targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list" },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            },
          }) : Promise.resolve(undefined),
          this.screenSpaceTerminationPixels > 0 ? this.device.createRenderPipelineAsync({
            label: "Sparse voxel primary exact-only coverage resolve",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageLayout!] }),
            vertex: { module: vertexModule, entryPoint: "vertexMain" },
            fragment: { module, entryPoint: "svoBrickExactResolveFragment", targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list" },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            },
          }) : Promise.resolve(undefined),
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel primary conservative coverage overflow",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageLayout!] }),
            vertex: { module, entryPoint: SVO_BRICK_RASTER_CONTRACT.entryPoints.vertex },
            fragment: { module, entryPoint: SVO_BRICK_RASTER_CONTRACT.entryPoints.overflowResolve, targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list", cullMode: SVO_BRICK_RASTER_CONTRACT.cullMode },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              // Re-run the direct arm as the authority on marked pixels. Equal
              // depth must replace the provisional candidate resolve too.
              depthCompare: "greater-equal",
            },
          }),
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel exact live-scene primitive visibility",
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            vertex: { module, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.vertex },
            fragment: { module, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.fragment, targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list", cullMode: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.cullMode },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            },
          }),
          // Coverage-only: the same proxy instances, no depth state at all, and
          // a throwaway one-channel target whose only reader is the overflow
          // gate. Nothing here evaluates a field.
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel conservative live-scene primitive coverage",
            layout: this.screenSpaceTerminationPixels > 0 ? bandedOcclusionCoverageLayouts() : bandedCoverageLayouts(),
            vertex: { module, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.bandVertex },
            fragment: { module, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.coverage, targets: [{ format: "r32uint" }] },
            primitive: { topology: "triangle-list", cullMode: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.cullMode },
            ...(this.screenSpaceTerminationPixels > 0 ? { depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            } } : {}),
          }),
          this.screenSpaceTerminationPixels > 0 ? this.device.createRenderPipelineAsync({
            label: "Sparse voxel resident-record LOD resolve",
            layout: bandedCoverageResolveLayouts(),
            vertex: { module: vertexModule, entryPoint: "vertexMain" },
            fragment: { module, entryPoint: "svoScenePrimitiveLodResolveFragment", targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list" },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            },
          }) : Promise.resolve(undefined),
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel conservative live-scene primitive resolve",
            layout: bandedCoverageLayouts(),
            vertex: { module: vertexModule, entryPoint: "vertexMain" },
            fragment: { module, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.resolve, targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list" },
            // Unlike the brick resolve this one does not establish the plane —
            // it competes with the surface already on it, so it keeps the
            // production reversed-Z test the direct fragment used.
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            },
          }),
          this.device.createRenderPipelineAsync({
            label: "Sparse voxel conservative live-scene primitive overflow",
            layout: bandedCoverageLayouts(),
            vertex: { module, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.bandVertex },
            fragment: { module, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.overflowResolve, targets: rasterPrimaryTargets },
            primitive: { topology: "triangle-list", cullMode: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.cullMode },
            depthStencil: {
              format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
              depthWriteEnabled: true,
              depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
            },
          }),
        ])
        : [undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined];
      const tieredComputeEnabled = this.rasterPrimary && this.screenSpaceTerminationPixels > 0
        && this.scenePrimitiveComputeOutputLayout && this.scenePrimitiveDepthBridgeLayout;
      const depthBridgeModule = tieredComputeEnabled
        ? await checkedModule(this.device, "Sparse voxel tiered compute depth bridge", /* wgsl */ `
@group(0) @binding(0) var tieredDepth:texture_2d<f32>;
@fragment fn tieredDepthBridge(@builtin(position) position:vec4f)->@builtin(frag_depth) f32{
  let depth=textureLoad(tieredDepth,vec2i(position.xy),0).x;if(!(depth>0.0)){discard;}return depth;
}`) : undefined;
      const [scenePrimitiveComputeArgs, scenePrimitiveComputeResolve, scenePrimitiveDepthBridge] = tieredComputeEnabled ? await Promise.all([
        this.device.createComputePipelineAsync({
          label: "Sparse voxel tiered scene-primitive dispatch args",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageLayout!, this.scenePrimitiveComputeOutputLayout!] }),
          compute: { module, entryPoint: "svoScenePrimitiveTieredComputeArgs" },
        }),
        this.device.createComputePipelineAsync({
          label: "Sparse voxel tiered scene-primitive compute resolve",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.brickResolveSceneLayout!, ...middleLayouts, this.brickCoverageLayout!, this.scenePrimitiveComputeOutputLayout!] }),
          compute: { module, entryPoint: "svoScenePrimitiveTieredComputeResolve" },
        }),
        this.device.createRenderPipelineAsync({
          label: "Sparse voxel tiered scene-primitive depth bridge",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.scenePrimitiveDepthBridgeLayout!] }),
          vertex: { module: vertexModule, entryPoint: "vertexMain" },
          fragment: { module: depthBridgeModule!, entryPoint: "tieredDepthBridge", targets: [] },
          primitive: { topology: "triangle-list" },
          depthStencil: {
            format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
            depthWriteEnabled: true,
            depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
          },
        }),
      ]) : [undefined, undefined, undefined];
      let surfaceMesh: SvoDrySplitPipelineBundle["surfaceMesh"];
      if (this.experiments.surfaceMesh) {
        const computeLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.surfaceMeshComputeLayout!] });
        const drawLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.surfaceMeshDrawLayout!] });
        const [prepare, boxes, mark, schedule, count, allocate, emit, publish, select, cull] = await Promise.all([
          "surfaceMeshPrepare", "surfaceMeshBoxes", "surfaceMeshMark", "surfaceMeshSchedule", "surfaceMeshCount",
          "surfaceMeshAllocate", "surfaceMeshEmit", "surfaceMeshPublish", "surfaceMeshSelect", "surfaceMeshCull"].map((entryPoint) =>
          this.device.createComputePipelineAsync({ label: entryPoint, layout: computeLayout, compute: { module, entryPoint } })));
        const depthStencil: GPUDepthStencilState = { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          depthWriteEnabled: true, depthCompare: "greater" };
        const draw = await this.device.createRenderPipelineAsync({ label: "Opaque voxel surface triangles", layout: drawLayout,
          vertex: { module, entryPoint: "surfaceMeshVertex" }, fragment: { module, entryPoint: "surfaceMeshFragment", targets: rasterPrimaryTargets },
          primitive: { topology: "triangle-strip", cullMode: "none" }, depthStencil });
        const background = await this.device.createRenderPipelineAsync({ label: "Voxel mesh exact planes", layout: drawLayout,
          vertex: { module: vertexModule, entryPoint: "vertexMain" }, fragment: { module, entryPoint: "surfaceMeshBackground", targets: rasterPrimaryTargets },
          primitive: { topology: "triangle-list" }, depthStencil: { ...depthStencil, depthCompare: "always" } });
        surfaceMesh = { prepare, boxes, mark, schedule, count, allocate, emit, publish, select, cull, draw, background };
      }
      // Compile only the expensive closure again. All visibility and cone work
      // shares the generic bundle; unsupported live publications switch back in
      // the same draw, with no asynchronous capability transition.
      let optimizedLighting: GPURenderPipeline | undefined;
      if (scale !== 1 && !globalIlluminationCapable && this.experiments.specializedDeferredLighting) {
        const optimizedModule = await checkedModule(this.device, "Opaque directional cone deferred lighting",
          createSvoDrySceneFragmentWGSL(scale, this.traversalMode, this.brickOccupancyMode, "split",
            this.screenSpaceTerminationPixels, false, this.rasterGlassDiscovery, false,
            this.coneFanout, { ...shaderExperiments, opaqueDirectionalCones: true }));
        optimizedLighting = await this.device.createRenderPipelineAsync({
          label: "Opaque directional cone deferred lighting",
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, ...middleLayouts, this.splitLightingLayout!, ...cacheConsumerLayouts] }),
          vertex: { module: vertexModule, entryPoint: "vertexMain" },
          fragment: { module: optimizedModule, entryPoint: "dryLightingMain", targets: [{ format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat }] },
          primitive: { topology: "triangle-list" },
          depthStencil: { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat, depthWriteEnabled: false, depthCompare: "less" },
        });
      }
      const bundle = { optimizedLighting, surfaceMesh, visibility, rasterRigidVisibility, primarySeamClosure, lighting, reconstructedLighting, skyLighting, prepassReset, prepassCoherent, prepassBoundary,
        worldGiFrame, worldGiCache, voxelLightDemand, voxelLightPopulate,
        brickBackground, brickRaster, brickCoverage, brickCoverageResolve, brickLodResolve, brickExactResolve,
        brickCoverageOverflow, scenePrimitiveRaster,
        scenePrimitiveCoverage, scenePrimitiveLodResolve, scenePrimitiveComputeArgs, scenePrimitiveComputeResolve, scenePrimitiveDepthBridge,
        scenePrimitiveCoverageResolve, scenePrimitiveCoverageOverflow };
      this.splitPipelineBundles.set(variantKey, bundle);
      return bundle;
    })();
    this.splitPipelineCompiles.set(variantKey, compile);
    try {
      const bundle = await compile;
      if (variantCurrent()) {
        this.activateSplitPipelineBundle(scale, bundle);
      }
    } finally {
      if (this.splitPipelineCompiles.get(variantKey) === compile) this.splitPipelineCompiles.delete(variantKey);
    }
  }

  /** The split-bundle cache key: cone scale plus the GI capability the kernel was compiled with. */
  private splitVariantKey(scale: SvoConeLightingScale, globalIlluminationCapable: boolean): string {
    return `${scale}|${globalIlluminationCapable ? "gi" : "no-gi"}`;
  }

  private currentSplitVariantKey(scale: SvoConeLightingScale): string {
    return this.splitVariantKey(scale, this.lightingOptions.globalIlluminationEnabled === true);
  }

  private ensureSplitTargets(): void {
    if (this.shadingPath === "inline" || !this.targetWidth || !this.targetHeight
      || !this.splitVisibilityLayout || !this.splitLightingLayout) return;
    this.ensureBrickCoverageBuffers();
    if (!this.splitGeometry || !this.splitOpaqueIdentity || (this.experiments.primaryWorkMap && !this.primaryWorkMap)
      || (this.rasterGlassDiscovery && (!this.splitGlassKey || !this.splitGlassDepth))
      || (this.screenSpaceTerminationPixels > 0 && !this.scenePrimitiveComputeDepth)
      || (this.rasterRigidDiscovery && !this.rasterRigidPrimaryGeometry)
      || (this.primaryEntryPrepassEnabled && (!this.primaryEntrySeed || !this.primaryEntryDepth))
      || this.splitWidth !== this.targetWidth || this.splitHeight !== this.targetHeight) {
      this.splitGeometry?.destroy();
      this.splitOpaqueIdentity?.destroy();
      this.primaryWorkMap?.destroy();
      this.splitGlassKey?.destroy();
      this.splitGlassDepth?.destroy();
      this.rasterRigidPrimaryGeometry?.destroy();
      this.scenePrimitiveComputeDepth?.destroy();
      this.primaryEntrySeed?.destroy();
      this.primaryEntryDepth?.destroy();
      this.splitGeometry = this.device.createTexture({
        label: "Sparse voxel split exact primary geometry",
        size: [this.targetWidth, this.targetHeight],
        format: SVO_DRY_SPLIT_GEOMETRY_FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
          | (this.rasterRigidDiscovery || this.rasterPrimary ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
      });
      this.splitGeometryView = this.splitGeometry.createView();
      this.splitOpaqueIdentity = this.device.createTexture({
        label: "Sparse voxel split exact primary identity",
        size: [this.targetWidth, this.targetHeight],
        format: SVO_DRY_SPLIT_IDENTITY_FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
          | (this.rasterRigidDiscovery || this.rasterPrimary ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
      });
      this.splitOpaqueIdentityView = this.splitOpaqueIdentity.createView();
      if (this.experiments.primaryWorkMap) {
        this.primaryWorkMap = this.device.createTexture({
          label: "Sparse voxel exact primary work map",
          size: [this.targetWidth, this.targetHeight],
          format: "rgba32uint",
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.primaryWorkMapView = this.primaryWorkMap.createView();
      }
      if (this.screenSpaceTerminationPixels > 0) {
        this.scenePrimitiveComputeDepth = this.device.createTexture({
          label: "Sparse voxel tiered scene-primitive compute depth",
          size: [this.targetWidth, this.targetHeight],
          format: "r32float",
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.scenePrimitiveComputeDepthView = this.scenePrimitiveComputeDepth.createView();
      }
      if (this.rasterRigidDiscovery) {
        this.rasterRigidPrimaryGeometry = this.device.createTexture({
          label: "Sparse voxel raster-rigid packed primary geometry",
          size: [this.targetWidth, this.targetHeight],
          format: SVO_RIGID_RASTER_CONTRACT.primaryGeometryFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.rasterRigidPrimaryGeometryView = this.rasterRigidPrimaryGeometry.createView();
      }
      if (this.primaryEntryPrepassEnabled) {
        // Sized with the primary G-buffer, because the seed is read at the
        // fragment's own pixel and nowhere else.
        this.primaryEntrySeed = this.device.createTexture({
          label: "Sparse voxel primary entry depth seed",
          size: [this.targetWidth, this.targetHeight],
          format: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.seedFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.primaryEntrySeedView = this.primaryEntrySeed.createView();
        this.primaryEntryDepth = this.device.createTexture({
          label: "Sparse voxel primary entry depth resolve",
          size: [this.targetWidth, this.targetHeight],
          format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.primaryEntryDepthView = this.primaryEntryDepth.createView();
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
    this.rebuildBrickRasterBindGroups();
    this.ensurePrimaryEntryBuffers();
    this.rebuildPrimaryEntryBindGroups();
    this.splitVisibilityBindGroup = this.device.createBindGroup({
      label: "Sparse voxel split visibility output binding",
      layout: this.splitVisibilityLayout,
      entries: [
        { binding: 0, resource: this.splitGeometryView! },
        { binding: 4, resource: this.splitOpaqueIdentityView! },
        ...(this.experiments.primaryWorkMap
          ? [{ binding: 7, resource: this.primaryWorkMapView! }]
          : []),
        ...(this.primaryEntryPrepassEnabled
          ? [{ binding: SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.seedBinding, resource: this.primaryEntrySeedView! }]
          : []),
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
    const gBufferViews = this.gBufferTargets.views;
    if (this.scenePrimitiveComputeOutputLayout && this.scenePrimitiveDepthBridgeLayout
      && this.scenePrimitiveComputeDepthView && this.scenePrimitiveComputeQueue
      && gBufferViews && this.bandStateBuffer && this.coverageAuditBuffer) {
      this.scenePrimitiveComputeOutputBindGroup = this.device.createBindGroup({
        label: "Sparse voxel tiered compute resolve output binding",
        layout: this.scenePrimitiveComputeOutputLayout,
        entries: [
          { binding: 0, resource: gBufferViews.hardwareDepth },
          { binding: 1, resource: gBufferViews.packedSurface },
          { binding: 2, resource: gBufferViews.identityMedia },
          { binding: 3, resource: this.splitGeometryView! },
          { binding: 4, resource: this.splitOpaqueIdentityView! },
          { binding: 5, resource: this.scenePrimitiveComputeDepthView },
          { binding: SVO_SCENE_PRIMITIVE_BAND_CONTRACT.readGroupBinding, resource: { buffer: this.bandStateBuffer } },
          { binding: SVO_SCENE_PRIMITIVE_BAND_CONTRACT.auditGroupBinding, resource: { buffer: this.coverageAuditBuffer } },
          { binding: 8, resource: { buffer: this.scenePrimitiveComputeQueue } },
        ],
      });
      this.scenePrimitiveDepthBridgeBindGroup = this.device.createBindGroup({
        label: "Sparse voxel tiered compute depth bridge binding",
        layout: this.scenePrimitiveDepthBridgeLayout,
        entries: [{ binding: 0, resource: this.scenePrimitiveComputeDepthView }],
      });
    }
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

  private derivedLightingReady(): boolean {
    const source = this.source;
    const nodeMip = source?.nodeMipPyramid;
    const radiance = source?.tetrahedralRadiance;
    return source?.derivedLighting?.state !== "unavailable"
      && Boolean(nodeMip && radiance
        && nodeMip.generation === radiance.generation
        && nodeMip.plan.complete && radiance.plan.complete);
  }

  get lightingVisibilityStatus(): SvoLightingVisibilityStatus {
    const requested = this.lightingOptions.coneTracingMode ?? "cones";
    if (requested === "off") return { state: "off" };
    if (requested === "exact") return { state: "exact" };
    if (this.derivedLightingReady()) return { state: "cones" };
    return {
      state: "exact",
      fallback: true,
      detail: this.source?.derivedLighting?.detail
        ?? "Complete cone-lighting hierarchy is unavailable; exact SVO shadows and AO are active",
    };
  }

  /** Exact readiness of the presentation bundle requested by current options. */
  get presentationBundleStatus(): SvoDryPresentationBundleStatus {
    if (this.requestedBundleFailure?.scale === this.coneScale) {
      return { state: "failed", detail: this.requestedBundleFailure.detail };
    }
    if (this.requestedBundleResourceFailure) {
      return { state: "failed", detail: this.requestedBundleResourceFailure };
    }
    if (this.coneScale !== 1 && this.conePipelineScale !== this.coneScale) {
      return { state: "compiling", detail: `Compiling requested SVO cone bundle at scale ${this.coneScale}` };
    }
    if (this.shadingPath === "split" && this.splitPipelineScale !== this.coneScale) {
      return { state: "compiling", detail: `Compiling requested SVO split bundle at scale ${this.coneScale}` };
    }
    return { state: "ready" };
  }

  /**
   * Compiles and caches the reduced-rate prepass and consuming pipelines for
   * the current scale. Until this resolves, encode rejects the requested frame
   * and exposes an inspectable compiling state. No-op at scale 1 or before initialize().
   */
  private activateConePipelineBundle(scale: SvoConeLightingScale, bundle: SvoDryConePipelineBundle): void {
    this.conePrepassGeometryPipeline = bundle.geometry;
    this.conePrepassVisibilityPipeline = bundle.visibility;
    this.conePrepassShadePipeline = bundle.shade;
    this.coneReducedPipeline = bundle.reduced;
    this.conePipelineScale = scale;
    if (this.requestedBundleFailure?.scale === scale) this.requestedBundleFailure = undefined;
    this.requestedBundleResourceFailure = undefined;
    this.ensureConePrepassTargets();
  }

  private async ensureConeLightingScale(scale: Exclude<SvoConeLightingScale, 1>): Promise<void> {
    if (!this.layout || (!this.pipeline && !this.experiments.surfaceMesh) || !this.vertexModule) return;
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
      && (!this.conePrepassComputeLayout || (this.conePrepassBoundaryQueue && this.coneBoundaryCountSnapshot && this.coneDerivedFailureSnapshot
        && this.conePrepassComputeBindGroup))
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
        size: 4 * (SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.queueHeaderWords + this.targetWidth * this.targetHeight),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.coneBoundaryCountSnapshot = this.device.createBuffer({
        label: "Sparse voxel compact cone-boundary count snapshot",
        size: Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.coneDerivedFailureSnapshot = this.device.createBuffer({
        label: "Sparse voxel compact derived-failure snapshot",
        size: 2 * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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
  }

  private releaseConePrepassTargets(): void {
    this.conePrepassBoundaryQueue?.destroy();
    this.coneBoundaryCountSnapshot?.destroy();
    this.coneDerivedFailureSnapshot?.destroy();
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
    this.coneBoundaryCountSnapshot = undefined;
    this.coneDerivedFailureSnapshot = undefined;
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

  private releaseVoxelLightCache(): void {
    this.voxelLightTexture?.destroy();
    this.voxelLightParamsBuffer?.destroy();
    this.voxelLightRequestBuffer?.destroy();
    this.voxelLightQueueBuffer?.destroy();
    this.voxelLightPageBuffer?.destroy();
    this.voxelLightTexture = undefined;
    this.voxelLightTextureView = undefined;
    this.voxelLightParamsBuffer = undefined;
    this.voxelLightRequestBuffer = undefined;
    this.voxelLightQueueBuffer = undefined;
    this.voxelLightPageBuffer = undefined;
    this.voxelLightConsumerBindGroup = undefined;
    this.voxelLightDemandBindGroup = undefined;
    this.voxelLightPopulateBindGroup = undefined;
    this.voxelLightPageCount = 0;
    this.voxelLightActive = false;
    this.voxelLightExclusive = false;
  }

  private writeVoxelLightCacheParams(): void {
    if (!this.voxelLightParamsBuffer || !this.source?.nodeMipPyramid) return;
    const nodeMip = this.source.nodeMipPyramid;
    const active = this.voxelLightUserEnabled && this.voxelLightPageCount > 0 && !this.fluidCoverage
      && (this.lightingOptions.coneTracingMode ?? "cones") === "cones"
      && this.lightingOptions.shadowsEnabled;
    const exclusive = active && this.renderTuning.maximumShadedLights === 1
      && !this.lightingOptions.ambientOcclusionEnabled
      && !this.source.tetrahedralRadiance;
    this.voxelLightActive = active;
    this.voxelLightExclusive = exclusive;
    this.device.queue.writeBuffer(this.voxelLightParamsBuffer, 0, new Uint32Array([
      this.voxelLightPageCount,
      SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.populationBudget,
      this.voxelLightEpoch,
      (active ? 1 : 0) | (exclusive ? 2 : 0),
      ...nodeMip.plan.atlas.pages,
      0,
    ]));
  }

  private invalidateVoxelLightCache(): void {
    if (!this.voxelLightParamsBuffer) return;
    this.voxelLightEpoch = this.voxelLightEpoch >= 0xffff ? 1 : this.voxelLightEpoch + 1;
    this.writeVoxelLightCacheParams();
  }

  private ensureVoxelLightCache(source: SparseVoxelSceneRenderSource | undefined, scene: SparseVoxelDrySceneData | undefined): void {
    this.releaseVoxelLightCache();
    const nodeMip = source?.nodeMipPyramid;
    const firstKind = scene?.lightRecords?.[24];
    if (this.experiments.voxelLightCache === false
      || !this.voxelLightConsumerLayout || !this.voxelLightDemandLayout || !this.voxelLightPopulateLayout) return;
    // A cache the user has switched off still allocated the full node-mip atlas
    // extent — 386 MB floored and around 2 GB unfloored at depth 3 — because
    // `setVoxelLightCacheEnabled(false)` only flipped a shader control word.
    // The smoke lane disables it before its fingerprint capture and was paying
    // for a texture it had just told the shader to ignore.
    //
    // Released through the ineligible arm rather than by skipping the
    // allocation: the bind groups have to survive, since the compiled pipeline
    // layout still declares group `splitGroup + 1` and `useSplit` requires it
    // (`voxelLightBindingsRequired` tracks the *build-time* experiment, not this
    // switch). Ineligible already means a 1x1x1 texture, four minimum buffers,
    // a zero page count, and `voxelLightActive == false` — the whole cost, gone,
    // with every binding still valid and no pipeline change.
    const eligible = Boolean(this.voxelLightUserEnabled && nodeMip?.plan.complete && nodeMip.directPageTableReady
      && nodeMip.plan.pages.length > 0 && firstKind === SVO_LIGHT_KINDS.directional);
    this.voxelLightEpoch = this.voxelLightEpoch >= 0xffff ? 1 : this.voxelLightEpoch + 1;
    this.voxelLightPageCount = eligible ? nodeMip!.plan.pages.length : 0;
    this.voxelLightTexture = this.device.createTexture({
      label: `Sparse voxel directional-light visibility generation ${nodeMip?.generation ?? 0}`,
      size: eligible ? nodeMip!.plan.atlas.texels : [1, 1, 1],
      dimension: "3d",
      format: SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    this.voxelLightTextureView = this.voxelLightTexture.createView({ dimension: "3d" });
    this.voxelLightParamsBuffer = this.device.createBuffer({ label: "Sparse voxel directional-light cache parameters", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.voxelLightRequestBuffer = this.device.createBuffer({
      label: "Sparse voxel directional-light request bitset",
      size: Math.max(4, this.voxelLightPageCount * SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.requestWordsPerPage * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.voxelLightQueueBuffer = this.device.createBuffer({
      label: "Sparse voxel directional-light population queue",
      size: SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.queueHeaderWords * 4
        + SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.populationBudget * SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.queueEntryWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.voxelLightPageBuffer = this.device.createBuffer({
      label: "Sparse voxel directional-light virtual pages",
      size: Math.max(16, this.voxelLightPageCount * 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const pageWords = new Uint32Array(this.voxelLightPageCount * 4);
    if (eligible) {
      for (const page of nodeMip!.plan.pages) pageWords.set([...page.key.coordinate, page.key.level], page.slot * 4);
      this.device.queue.writeBuffer(this.voxelLightPageBuffer, 0, pageWords);
    }
    this.writeVoxelLightCacheParams();
    this.voxelLightConsumerBindGroup = this.device.createBindGroup({ label: "Sparse voxel directional-light cache consumer", layout: this.voxelLightConsumerLayout, entries: [
      { binding: 0, resource: this.voxelLightTextureView },
      { binding: 1, resource: { buffer: this.voxelLightParamsBuffer } },
    ] });
    this.voxelLightDemandBindGroup = this.device.createBindGroup({ label: "Sparse voxel directional-light cache demand", layout: this.voxelLightDemandLayout, entries: [
      { binding: 0, resource: this.voxelLightTextureView },
      { binding: 1, resource: { buffer: this.voxelLightParamsBuffer } },
      { binding: 3, resource: { buffer: this.voxelLightRequestBuffer } },
      { binding: 4, resource: { buffer: this.voxelLightQueueBuffer } },
    ] });
    this.voxelLightPopulateBindGroup = this.device.createBindGroup({ label: "Sparse voxel directional-light cache population", layout: this.voxelLightPopulateLayout, entries: [
      { binding: 1, resource: { buffer: this.voxelLightParamsBuffer } },
      { binding: 2, resource: this.voxelLightTextureView },
      { binding: 4, resource: { buffer: this.voxelLightQueueBuffer } },
      { binding: 5, resource: { buffer: this.voxelLightPageBuffer } },
    ] });
  }

  /** Attach only the mutable SVO acceleration source. Scene content publishes independently. */
  setSource(source: SparseVoxelSceneRenderSource | undefined): void {
    if (source === this.source) {
      // A warm world-scale re-seed retains the sparse buffers but republishes
      // their metre mapping on the same source object. Refresh the uniforms
      // rather than treating object identity as proof that nothing changed.
      if (source && this.scene && canEncodeSparseVoxelDryScene(source, this.scene)) {
        this.worldGiCacheDirty = true;
        this.writeParams(source, this.scene);
      }
      return;
    }
    this.pickingFrameToken += 1;
    this.lastPickingTarget = undefined;
    this.worldGiCacheDirty = true;
    const oldStructural = this.source?.structural;
    const newStructural = source?.structural;
    const structuralChanged = oldStructural?.structure.buffer !== newStructural?.structure.buffer
      || oldStructural?.structure.offset !== newStructural?.structure.offset
      || oldStructural?.scenePayload.buffer !== newStructural?.scenePayload.buffer
      || oldStructural?.scenePayload.offset !== newStructural?.scenePayload.offset;
    this.source = source;
    if (this.surfaceMeshState && structuralChanged) {
      // A different structural arena means different leaf slots: every cached
      // brick range is void, so the mesh starts over from a first build.
      this.resetSurfaceMeshState();
      this.ensureSurfaceMeshWork();
    }
    this.ensureVoxelLightCache(source, this.scene);
    this.updateTetrahedralRadianceBlackPages(source?.tetrahedralRadiance);
    this.rebuild();
  }

  /**
   * Publish one complete renderer generation into fixed-capacity arenas.
   * Validation happens before the first queue write, so overflow or malformed
   * input retains the preceding complete scene instead of exposing a partial
   * update.
   */
  publishScene(scene: SparseVoxelDrySceneData): boolean {
    const source = this.source;
    if (!canEncodeSparseVoxelDryScene(source, scene)) return false;
    const primitiveArena = packSvoPrimitiveCandidateArena(scene.primitiveRecords, scene.primitiveCandidates);
    if (primitiveArena.packedRecords.byteLength > SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES) throw new RangeError("Live scene primitive arena capacity exceeded");
    if (scene.materialRecords.byteLength > SVO_DRY_SCENE_MATERIAL_ARENA_SIZE_BYTES) throw new RangeError("Live scene material arena capacity exceeded");
    if ((scene.glassRecords?.byteLength ?? 0) > SVO_DRY_SCENE_GLASS_ARENA_SIZE_BYTES) throw new RangeError("Live scene thin-glass arena capacity exceeded");

    // Host publication is known now; waiting for the periodic GPU diagnostic
    // leaves a formerly-ready mesh rebuilding at one batch for up to 30 frames.
    // GPU revision checks remain authoritative and completed builds do no work.
    if (this.surfaceMeshStatus) {
      this.surfaceMeshStatus = { ...this.surfaceMeshStatus, state: "pending", buildPhase: "extracting" };
      this.surfaceMeshBuildPresentations = 0;
    }
    this.pickingFrameToken += 1;
    this.lastPickingTarget = undefined;
    this.worldGiCacheDirty = true;
    this.invalidateVoxelLightCache();
    this.primitiveDirtyBounds = [];
    this.scene = scene;
    this.primitiveCount = primitiveArena.primitiveCount;
    this.writeScenePrimitiveOverflowPublication();
    this.primitiveCandidateArena = primitiveArena;
    this.ensureVoxelLightCache(source, scene);
    this.coneFanoutLightCount = Math.min(
      scene.lightRecords ? scene.lightRecords.byteLength / SVO_LIGHT_RECORD_STRIDE_BYTES : 1,
      SVO_CONE_FANOUT_CONTRACT.maximumLights,
    );
    if (this.coneFanoutFrameBuffer && this.conePrepassWidth && this.conePrepassHeight) {
      this.device.queue.writeBuffer(this.coneFanoutFrameBuffer, 0, packSvoConeFanoutFrame({
        width: this.conePrepassWidth,
        height: this.conePrepassHeight,
        lightCount: this.coneFanoutLightCount,
        secondaryLightSamples: false,
      }));
    }
    const paneCount = (scene.glassRecords?.byteLength ?? 0) / SVO_THIN_GLASS_RECORD_STRIDE_BYTES;
    const records = scene.glassRecords;
    this.rasterGlassPaneCount = paneCount;
    this.rasterGlassFirstRecord = 0;
    this.rasterGlassRecordCount = paneCount;
    this.device.queue.writeBuffer(this.rasterGlassParamsBuffer, 0, new Uint32Array([
      paneCount,
      0,
      0,
      0,
    ]));
    this.device.queue.writeBuffer(this.sceneArenaBuffer, SVO_DRY_SCENE_ARENA_LAYOUT.primitiveOffsetBytes, primitiveArena.packedRecords);
    this.device.queue.writeBuffer(this.sceneArenaBuffer, SVO_DRY_SCENE_ARENA_LAYOUT.materialOffsetBytes, scene.materialRecords);
    if (records?.byteLength) this.device.queue.writeBuffer(this.sceneArenaBuffer, SVO_DRY_SCENE_ARENA_LAYOUT.glassOffsetBytes, records);
    // Written on every publication: a stale block left behind by the previous
    // scene would be resolved by a new scene's cluster record and grow the wrong
    // packing inside its lobe. Zeroes are the "not resolved" encoding.
    this.device.queue.writeBuffer(
      this.sceneArenaBuffer,
      SVO_DRY_SCENE_ARENA_LAYOUT.clusterOffsetBytes,
      scene.clusterBlocks ?? new Uint32Array(SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES / Uint32Array.BYTES_PER_ELEMENT),
    );
    // Unconditional for the same reason the cluster region above is: a tape left
    // behind by the previous scene is a block a new scene's record can resolve,
    // and it would draw that scene's shape at this one's transform. Zeroes are
    // the "not resolved" encoding, which draws nothing and reports itself.
    this.device.queue.writeBuffer(
      this.sceneArenaBuffer,
      SVO_DRY_SCENE_ARENA_LAYOUT.fieldProgramOffsetBytes,
      scene.fieldProgramBlocks
        ?? new Uint32Array(SVO_DRY_SCENE_FIELD_PROGRAM_ARENA_SIZE_BYTES / Uint32Array.BYTES_PER_ELEMENT),
    );
    warnOnUnresolvedClusters(scene);
    this.writeParams(source!, scene);
    const lightingArena = packSparseVoxelDrySceneLightingArena(scene);
    if (lightingArena) this.device.queue.writeBuffer(this.lightingBuffer, 0, lightingArena);
    this.device.queue.writeBuffer(this.thickGlassUniformBuffer, 0, packSparseVoxelDrySceneThickGlassArena(scene));
    if (!this.bindGroup) this.rebuild();
    return true;
  }

  /**
   * Materializes the source owner's exact black-slot set as a sampled bit
   * plane. It changes with a completed radiance generation, never with the
   * camera, so ordinary frames do no uploads or allocation work.
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

  /** Hot-publish a complete primitive generation and its exact BVH in place. */
  publishPrimitiveArena(
    records: Uint32Array<ArrayBuffer>,
    candidates: SvoPrimitiveCandidatePublication,
    renderRevision: number,
    change: SvoDryPrimitiveArenaChange,
  ): boolean {
    if (!this.scene || !this.source || !records.byteLength) return false;
    if (!Number.isSafeInteger(renderRevision) || renderRevision < 1 || renderRevision > 0xffff_ffff) throw new RangeError("Live scene render revision must be a positive uint32");
    for (const bounds of change.dirtyBounds) {
      if (bounds.minimum.some((value) => !Number.isFinite(value)) || bounds.maximum.some((value) => !Number.isFinite(value))
        || bounds.minimum.some((value, axis) => value > bounds.maximum[axis])) {
        throw new RangeError("Live scene dirty bounds must be finite ordered AABBs");
      }
    }
    const incremental = change.dirtyPrimitiveIndices !== undefined
      && change.dirtyCandidateNodeIndices !== undefined
      && this.primitiveCandidateArena?.primitiveCount === candidates.primitiveCount
      && this.primitiveCandidateArena.candidateNodeCount === candidates.nodes.length
      && records.byteLength === candidates.primitiveCount * SVO_PRIMITIVE_RECORD_STRIDE_BYTES;
    let arena = this.primitiveCandidateArena;
    if (incremental && arena) {
      const uploadRecords = (indices: readonly number[], source: Uint32Array<ArrayBuffer>, arenaRecordBase: number): void => {
        const ordered = [...new Set(indices)].sort((left, right) => left - right);
        for (const index of ordered) {
          if (!Number.isSafeInteger(index) || index < 0 || index >= source.length / SVO_PRIMITIVE_RECORD_WORDS) {
            throw new RangeError("Live scene dirty primitive-arena index is invalid");
          }
          const sourceWord = index * SVO_PRIMITIVE_RECORD_WORDS;
          const arenaWord = (arenaRecordBase + index) * SVO_PRIMITIVE_RECORD_WORDS;
          const record = source.subarray(sourceWord, sourceWord + SVO_PRIMITIVE_RECORD_WORDS);
          arena!.packedRecords.set(record, arenaWord);
          this.device.queue.writeBuffer(this.sceneArenaBuffer,
            SVO_DRY_SCENE_ARENA_LAYOUT.primitiveOffsetBytes + arenaWord * Uint32Array.BYTES_PER_ELEMENT,
            record);
        }
      };
      uploadRecords(change.dirtyPrimitiveIndices!, records, 0);
      uploadRecords(change.dirtyCandidateNodeIndices!, candidates.packedRecords, candidates.primitiveCount);
    } else {
      arena = packSvoPrimitiveCandidateArena(records, candidates);
      if (arena.packedRecords.byteLength > SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES) throw new RangeError("Live scene primitive arena capacity exceeded");
      this.device.queue.writeBuffer(this.sceneArenaBuffer, SVO_DRY_SCENE_ARENA_LAYOUT.primitiveOffsetBytes, arena.packedRecords);
    }
    this.primitiveCount = arena.primitiveCount;
    this.writeScenePrimitiveOverflowPublication();
    this.primitiveCandidateArena = arena;
    this.scene = { ...this.scene, renderRevision, primitiveRecords: records, primitiveCandidates: candidates };
    this.writeParams(this.source, this.scene);
    this.pickingFrameToken += 1;
    this.lastPickingTarget = undefined;
    this.primitiveDirtyBounds = change.dirtyBounds;
    const invalidation = svoDryPrimitiveArenaCacheInvalidation(change);
    if (invalidation.worldGi) this.worldGiCacheDirty = true;
    if (invalidation.directionalVisibility) this.invalidateVoxelLightCache();
    return true;
  }

  /** Enable finished-image visibility effects without rebuilding scene-owned resources. */
  /**
   * Which stages this pipeline must not encode.
   *
   * Deliberately not folded into `setLightingOptions`: nothing here changes a
   * shader, a bind group or a bundle, so it must not reach the code that
   * rebuilds them. It does invalidate every cached frame — a reused G-buffer
   * was traced under the previous set and would silently answer for it.
   */
  setDisabledStages(disabled: DisabledRenderStages): void {
    if (disabledRenderStagesEqual(this.disabledStages, disabled)) return;
    // The world-GI cache is persistent and world-keyed, so entries survive
    // everything except an explicit clear. Withholding the pass that fills it,
    // or the primary that decides which keys get queried, leaves entries that
    // describe a frame this pipeline is no longer drawing.
    const staleWorldGi = this.disabledStages.has("primary-traversal") !== disabled.has("primary-traversal");
    const entrySeedMoved = this.disabledStages.has("primary-entry-prepass") !== disabled.has("primary-entry-prepass");
    this.disabledStages = new Set(disabled);
    if (staleWorldGi) this.worldGiCacheDirty = true;
    // The seed plane's "no voxel leaf on this ray" is a proof of absence the
    // prepass earns by drawing every leaf, so a plane left behind by the last
    // frame that ran it is not a stale answer — it is a wrong one, read as
    // legal. The fragment has to learn the pass is gone before it reads it.
    if (entrySeedMoved) this.writePrimaryEntryParams();
  }

  setLightingOptions(options: SparseVoxelDrySceneLightingOptions): void {
    const { coneTracingMode } = resolveSvoPipelineComposition({ coneTracingMode: options.coneTracingMode });
    // Leaving `cones` collapses to the full-resolution inline/split path: with
    // the effective scale forced to 1, encode() never runs the reduced
    // prepass, compact cone visibility, sample fan-out, or world-GI cache
    // passes, and writeParams withholds every cone-dependent flag.
    const coneLightingScale = coneTracingMode === "cones" ? (options.coneLightingScale ?? 1) : 1;
    const silhouetteRefinementEnabled = options.silhouetteRefinementEnabled === true;
    const globalIlluminationEnabled = options.globalIlluminationEnabled === true;
    // Opt-in, unlike every other lighting flag here: the cache is off unless
    // this frame asked for it by name.
    const worldGiCacheEnabled = options.worldGiCacheEnabled === true;
    const previousConeTracingMode = this.lightingOptions.coneTracingMode ?? "cones";
    const previousGlobalIllumination = this.lightingOptions.globalIlluminationEnabled === true;
    const previousWorldGiCache = this.lightingOptions.worldGiCacheEnabled === true;
    if (options.shadowsEnabled === this.lightingOptions.shadowsEnabled
      && options.ambientOcclusionEnabled === this.lightingOptions.ambientOcclusionEnabled
      && silhouetteRefinementEnabled === this.silhouetteRefinementEnabled
      && coneTracingMode === previousConeTracingMode
      && globalIlluminationEnabled === previousGlobalIllumination
      && worldGiCacheEnabled === previousWorldGiCache
      && coneLightingScale === this.coneScale) return;
    // Turning the cache back on must not resume against entries gathered under
    // the lighting of whichever frame last filled it.
    const invalidateWorldGi = options.ambientOcclusionEnabled !== this.lightingOptions.ambientOcclusionEnabled
      || coneTracingMode !== previousConeTracingMode
      || globalIlluminationEnabled !== previousGlobalIllumination
      || worldGiCacheEnabled !== previousWorldGiCache;
    this.lightingOptions = { shadowsEnabled: options.shadowsEnabled, ambientOcclusionEnabled: options.ambientOcclusionEnabled,
      silhouetteRefinementEnabled, coneTracingMode, globalIlluminationEnabled, worldGiCacheEnabled };
    this.silhouetteRefinementEnabled = silhouetteRefinementEnabled;
    this.coneScale = coneLightingScale;
    this.requestedBundleFailure = undefined;
    this.requestedBundleResourceFailure = undefined;
    this.writeVoxelLightCacheParams();
    if (invalidateWorldGi) this.worldGiCacheDirty = true;
    if (coneLightingScale !== 1) {
      const coneBundle = this.conePipelineBundles.get(coneLightingScale);
      if (coneBundle) this.activateConePipelineBundle(coneLightingScale, coneBundle);
      const splitBundle = this.splitPipelineBundles.get(this.currentSplitVariantKey(coneLightingScale));
      if (splitBundle) this.activateSplitPipelineBundle(coneLightingScale, splitBundle);
    }
    if (this.source && this.scene && canEncodeSparseVoxelDryScene(this.source, this.scene)) {
      this.writeParams(this.source, this.scene);
    }
    const retainFailure = (error: unknown) => {
      if (this.coneScale !== coneLightingScale) return;
      const reason = error instanceof Error ? error.message : String(error);
      this.requestedBundleFailure = {
        scale: coneLightingScale,
        detail: `Requested SVO presentation bundle at scale ${coneLightingScale} failed: ${reason}`,
      };
    };
    // Both arms also re-request the split bundle, which is keyed by scale AND
    // GI capability: a GI flip compiles the specialised variant while the
    // stale-but-correct bundle keeps rendering until it activates.
    if (coneLightingScale !== 1) void this.ensureConeLightingPrepass().catch(retainFailure);
    else if (this.shadingPath === "split") void this.ensureSplitPipelines(1).catch(retainFailure);
  }

  get silhouetteRefinementStatus(): SvoSilhouetteRefinementStatus {
    if (this.shadingPath === "inline") {
      return { state: "not-applicable", detail: "Primary seam closure requires split shading" };
    }
    if (!this.silhouetteRefinementEnabled) return { state: "disabled" };
    if (this.requestedBundleFailure?.scale === this.coneScale) return { state: "failed", detail: this.requestedBundleFailure.detail };
    if (this.requestedBundleResourceFailure) return { state: "failed", detail: this.requestedBundleResourceFailure };
    if (this.splitPipelineCompiles.has(this.currentSplitVariantKey(this.coneScale)) || this.splitPipelineScale !== this.coneScale) {
      return { state: "compiling", detail: `Preparing split presentation bundle at scale ${this.coneScale}` };
    }
    if (!this.primarySeamClosurePipeline) {
      return { state: "compiling", detail: "Preparing primary seam-closure pipeline" };
    }
    return { state: "enabled" };
  }

  /** Update bounded shader work budgets without rebuilding scene resources. */
  setRenderTuning(tuning: SvoRenderTuning, retainSurfaceMeshing = false): void {
    resolveSvoPipelineComposition({ coneRadianceReconstruction: tuning.coneRadianceReconstruction, coneTracingMode: this.lightingOptions.coneTracingMode });
    // During a producer replacement the current attachment still belongs to
    // the old mesher. Changing the extraction mode early invalidates its ready
    // raster mesh and reports a misleading completed-but-unavailable build.
    const normalized = normalizeSvoRenderTuning(retainSurfaceMeshing ? {
      ...tuning,
      surfaceMeshing: this.renderTuning.surfaceMeshing,
      surfaceMeshContours: this.renderTuning.surfaceMeshContours,
      surfaceMeshContourInflation: this.renderTuning.surfaceMeshContourInflation,
    } : tuning);
    if (Object.keys(normalized).every((key) => normalized[key as keyof SvoRenderTuning] === this.renderTuning[key as keyof SvoRenderTuning])) return;
    // The band is a *visibility* control and nothing else: it decides which pass
    // resolves a pixel, never what a light sees. Lighting reads the node-mip
    // pyramid and the analytic BVH, neither of which the coverage arena touches,
    // so retuning it must not drop the persistent world-GI cache — otherwise an
    // A/B over the band measures a GI rebuild in both arms and nothing else.
    const bandKeys = ["nearFieldBandPixels", "nearFieldBandHysteresis", "nearFieldBandBudget"] as const;
    const bandOnly = (Object.keys(normalized) as (keyof SvoRenderTuning)[])
      .every((key) => normalized[key] === this.renderTuning[key] || (bandKeys as readonly string[]).includes(key));
    if (bandOnly) {
      this.renderTuning = normalized;
      this.writeBandParams();
      return;
    }
    // Same argument as the band, one pass earlier: level of detail decides how
    // deep the primary descends and nothing else. It adds no pass, moves no
    // march shape, and lighting never reads it — so a slider drag must cost one
    // 16-byte write, not a world-GI rebuild and a discarded primary.
    const lodKeys = ["lodMode", "lodScreenSpacePixels", "lodFixedLevel", "surfaceMeshLodPixels", "surfaceMeshFilteringEnabled", "surfaceMeshNormalSmoothing", "surfaceMeshNormalStrength", "surfaceMeshMaxCoarsening", "surfaceMeshLodHysteresis", "surfaceMeshNormalAgreement", "surfaceMeshPreserveCloseNormals", "occluderGhosting", "occluderGhostOpacity"] as const;
    const lodOnly = (Object.keys(normalized) as (keyof SvoRenderTuning)[])
      .every((key) => normalized[key] === this.renderTuning[key] || (lodKeys as readonly string[]).includes(key));
    if (lodOnly) {
      this.renderTuning = normalized;
      this.writeLodParams();
      return;
    }
    const invalidateVoxelVisibility = normalized.coneStepBudget !== this.renderTuning.coneStepBudget
      || normalized.shadowBiasCells !== this.renderTuning.shadowBiasCells
      || normalized.shadowConeAperture !== this.renderTuning.shadowConeAperture
      || normalized.coneNormalEscapeCells !== this.renderTuning.coneNormalEscapeCells;
    this.renderTuning = normalized;
    // Screen rate, reconstruction, light-loop count, AO and GI controls do
    // not change the cached slot-zero visibility. Keeping its epoch across
    // those presentation-only changes is what makes camera/view-tier A/Bs a
    // valid warm-cache measurement. March-shape changes still invalidate.
    if (invalidateVoxelVisibility) this.invalidateVoxelLightCache();
    this.worldGiCacheDirty = true;
    this.writeBandParams();
    if (this.source && this.scene && canEncodeSparseVoxelDryScene(this.source, this.scene)) this.writeParams(this.source, this.scene);
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
    // The renderer attaches the owner before it encodes the first fill, so the
    // bind groups built at attach time hold the 1x1x1 fallback view; the
    // volume's own view only exists once that fill has run. A sparse scene
    // rebuilt them anyway on its next topology change. A scene whose world
    // never changes after load — the uniform troughs — kept the fallback for
    // the whole session: a frame that says valid over a texture that says
    // nothing, so no water shadow and nothing ever see-through.
    const view = this.fluidCoverage.visibleGeneration()?.view ?? this.fluidCoverageFallbackView;
    if (this.bindGroup && view !== this.boundFluidCoverageView) this.rebuild();
    this.device.queue.writeBuffer(this.paramsBuffer, SVO_DRY_SCENE_PARAMS_LAYOUT.fluidCoverageWordOffset * 4, this.fluidCoverage.frame());
  }

  private writeParams(source: SparseVoxelSceneRenderSource, scene: SparseVoxelDrySceneData): void {
    const structural = source.structural!;
    const materialCount = scene.materialRecords.byteLength / SVO_MATERIAL_RECORD_STRIDE_BYTES;
    const buffer = new ArrayBuffer(SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes), floats = new Float32Array(buffer), words = new Uint32Array(buffer);
    floats.set(structural.domain.worldOrigin_m, 0); words[3] = structural.domain.brickSize;
    floats.set(structural.domain.cellSize_m, 4); words[7] = structural.domain.maximumDepth;
    words.set([structural.capacities.nodes, structural.capacities.leaves, 256, 0], 8);
    words.set([this.primitiveCount, scene.ownerBase, scene.skippedOwnerId ?? 0xffff_ffff, materialCount], 12);
    floats.set(scene.lightDirection ?? [-0.45, 0.86, 0.28], 16);
    floats.set(scene.lightColor ?? [1.04, 1.0, 0.91], 20);
    words.set([0,
      (scene.glassRecords?.byteLength ?? 0) / SVO_THIN_GLASS_RECORD_STRIDE_BYTES,
      0, 0], SVO_DRY_SCENE_PARAMS_LAYOUT.glassWordOffset);
    words.set([
      structural.planarBoundaries.count,
      structural.planarBoundaries.generation,
      structural.planarBoundaries.strideBytes,
      0,
    ], SVO_DRY_SCENE_PARAMS_LAYOUT.planarBoundaryWordOffset);
    const coneTracingMode = this.lightingOptions.coneTracingMode ?? "cones";
    // `off` strictly removes lighting-visibility work: with shadows and AO
    // held false no exact-ray flag is written either, and every visibility
    // entry point returns its unoccluded constant. `exact` keeps the bounded
    // reference traversals while withholding all cone stages.
    const shadowsEnabled = coneTracingMode !== "off" && this.lightingOptions.shadowsEnabled && scene.shadowVisibilityEnabled !== false;
    const ambientOcclusionEnabled = coneTracingMode !== "off" && this.lightingOptions.ambientOcclusionEnabled && scene.contactVisibilityEnabled !== false;
    const coneTracingEnabled = coneTracingMode === "cones" && this.derivedLightingReady();
    // Withheld, not scaled to zero: a gather that runs and is then multiplied by
    // nothing costs exactly what it cost before.
    const globalIlluminationEnabled = this.lightingOptions.globalIlluminationEnabled === true;
    const nodeMip = source.nodeMipPyramid;
    const nodeMipUsesPageValidity = Boolean((nodeMip as typeof nodeMip & {
      pageValidity?: { view: GPUTextureView };
    } | undefined)?.pageValidity?.view);
    const tetrahedralRadiance = source.tetrahedralRadiance;
    const giReady = coneTracingEnabled && globalIlluminationEnabled && Boolean(nodeMip && tetrahedralRadiance
      && nodeMip.generation === tetrahedralRadiance.generation
      && nodeMip.plan.complete && tetrahedralRadiance.plan.complete);
    const silhouetteRefinementActive = this.silhouetteRefinementEnabled && this.shadingPath === "split";
    const visibilityFlags = (!giReady && ambientOcclusionEnabled ? SVO_DRY_VISIBILITY_FLAGS.exactContact | SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion : 0)
      | (shadowsEnabled ? SVO_DRY_VISIBILITY_FLAGS.exactShadow : 0)
      | (coneTracingEnabled && (shadowsEnabled || ambientOcclusionEnabled || giReady) ? SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested : 0)
      | (giReady ? SVO_DRY_VISIBILITY_FLAGS.globalIllumination : 0)
      | (giReady && ambientOcclusionEnabled ? SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion : 0)
      | (coneTracingEnabled && globalIlluminationEnabled ? SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested : 0)
      | (silhouetteRefinementActive ? SVO_DRY_VISIBILITY_FLAGS.silhouetteRefinement : 0)
      | (scene.flatVoxelNormals ? SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals : 0);
    words.set([materialCount, scene.materialRevision, SVO_MATERIAL_RECORD_STRIDE_BYTES, visibilityFlags], SVO_DRY_SCENE_PARAMS_LAYOUT.materialPublicationWordOffset);
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
    floats.set(this.rigidBounds, SVO_DRY_SCENE_PARAMS_LAYOUT.rigidBoundsWordOffset);
    const candidates = this.primitiveCandidateArena;
    words.set(candidates
      ? [candidates.candidateRecordOffset, candidates.candidateNodeCount, candidates.candidateRootNodeIndex, scene.renderRevision]
      : [0, 0, 0, 0], SVO_DRY_SCENE_PARAMS_LAYOUT.primitiveCandidatesWordOffset);
    words.set([
      structural.structureOffsetsWords.control,
      structural.structureOffsetsWords.publication,
      structural.structureOffsetsWords.nodes,
      structural.structureOffsetsWords.leaves,
    ], SVO_DRY_SCENE_PARAMS_LAYOUT.structureOffsetsWordOffset);
    // Scene identity's own addresses inside the arena bound at binding 3. Written
    // from the producer's published block rather than re-derived here, so the
    // decode this shader compiled and the arena it is handed cannot disagree
    // about a lane base — the same reason the banded codec is emitted once.
    const payloadLanes = structural.scenePayloadLanes;
    words.set([
      payloadLanes.occupancyWords, payloadLanes.recordMaskWords,
      payloadLanes.headerWords, payloadLanes.blobWords,
    ], SVO_DRY_SCENE_PARAMS_LAYOUT.payloadLaneWordOffset);
    words.set([
      payloadLanes.geometryWords, payloadLanes.materialOwnerWords,
      structural.capacities.voxels,
      SVO_DRY_LEAF_PAYLOAD_MODES[payloadLanes.mode]
        | (payloadLanes.geometryStrideWords << 8)
        | (payloadLanes.geometryFractionWord << 16)
        | (payloadLanes.geometryPacked ? 1 << 24 : 0),
    ], SVO_DRY_SCENE_PARAMS_LAYOUT.payloadLane1WordOffset);
    words[SVO_DRY_SCENE_PARAMS_LAYOUT.primaryEntryWordOffset] = this.primaryEntrySeedLive() ? 1 : 0;
    if (nodeMip && nodeMip.generation > 0 && nodeMip.plan.complete) {
      // Folded rather than spread. `Math.max(1, ...pages.map(...))` passes one
      // argument per page, and the hero garden reaches 28 232 bricks at a
      // 3.125 mm lattice — far past the engine's argument limit, where this
      // throws `Maximum call stack size exceeded` from a line that reads like
      // arithmetic. The failure scales with scene refinement, so it stayed
      // invisible until the lattice moved and then blocked it outright.
      let nodeMipLevels = 1;
      // The opacity floor is *read off the plan* rather than plumbed alongside
      // it: a floored plan seeds every page at or above the floor, and the
      // ancestor walk only ever goes up, so the finest level present is the
      // floor. Deriving it here makes a CPU/GPU disagreement impossible, and
      // an unfloored plan answers 0 — the identity clamp.
      let opacityFloorLevel = Number.MAX_SAFE_INTEGER;
      for (const page of nodeMip.plan.pages) {
        nodeMipLevels = Math.max(nodeMipLevels, page.key.level + 1);
        opacityFloorLevel = Math.min(opacityFloorLevel, page.key.level);
      }
      words.set([nodeMip.generation, nodeMip.plan.pages.length, nodeMipLevels,
        nodeMipUsesPageValidity ? SVO_DRY_NODE_MIP_PUBLICATION_MODE.pageValidity : SVO_DRY_NODE_MIP_PUBLICATION_MODE.matchingStructuralGeneration],
      SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipWordOffset);
      words.set([...nodeMip.plan.atlas.texels, Math.min(opacityFloorLevel, nodeMipLevels - 1)],
        SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipAtlasWordOffset);
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
    if (giReady) {
      words.set([tetrahedralRadiance!.generation, 1,
        tetrahedralRadiance!.radianceFloorLevel ?? 0, tetrahedralRadiance!.slotOffset ?? 0],
      SVO_DRY_SCENE_PARAMS_LAYOUT.tetrahedralRadianceWordOffset);
    }
    const wide = resolveSvoWideTraversalCapability(source.wideFanout, source.revision, structural.domain.maximumDepth);
    if (wide.status === "ready") {
      const publication = wide.publication;
      words.set([publication.generation, publication.sourceGeneration, publication.pageCount, publication.descriptorCount],
        SVO_DRY_SCENE_PARAMS_LAYOUT.wideFanoutWordOffset);
      words.set([wide.source.traversalOffsetsWords.pages, wide.source.traversalOffsetsWords.descriptors, 0, 0],
        SVO_DRY_SCENE_PARAMS_LAYOUT.derivedTraversalWordOffset);
    }
    this.packLodParams(floats, words, SVO_DRY_SCENE_PARAMS_LAYOUT.lodWordOffset);
    this.packMeshFilterParams(floats, SVO_DRY_SCENE_PARAMS_LAYOUT.meshFilterWordOffset);
    this.packOccluderGhostParams(floats, SVO_DRY_SCENE_PARAMS_LAYOUT.occluderGhostWordOffset);
    if (this.paramsWords?.length === words.length && words.every((word, index) => word === this.paramsWords![index])) return;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    this.paramsWords = Uint32Array.from(words);
  }

  /**
   * The live level-of-detail lane.
   *
   * A threshold that lived in the shader source made every sweep a pipeline
   * rebuild, which is the one thing an interleaved A/B cannot afford, and made
   * the panel's slider impossible. The constructor flag still decides whether
   * the machinery is compiled at all — that build is the bit-exact reference —
   * and a zero threshold reproduces its image from the LOD build.
   */
  private packLodParams(floats: Float32Array, words: Uint32Array, offset: number): void {
    const fixed = this.renderTuning.lodMode === "fixed-level";
    // Zeroing the threshold in fixed-level mode is what keeps the two controls
    // from compounding: the slider is then the only thing deciding depth, which
    // is the whole point of a debug view of the hierarchy.
    floats[offset] = fixed ? 0 : this.renderTuning.lodScreenSpacePixels;
    words[offset + 1] = fixed ? SVO_LOD_MODES["fixed-level"] : SVO_LOD_MODES["screen-space"];
    words[offset + 2] = this.renderTuning.lodFixedLevel;
    // A float lane, so the code goes through `floats`, not `words`.
    //
    // `lod` is a vec4f and the shader reads this as `u32(dry.lod.w)` — a value
    // conversion, not a bitcast. Writing the integer 1 through the u32 view puts
    // the bit pattern 0x00000001 in a float lane, which is a denormal around
    // 1.4e-45, and converting *that* to u32 gives zero. The arm therefore always
    // read as `voxel-face` no matter what the panel said, and it failed silently:
    // no validation error, a frame that renders, and two arms whose hashes match
    // to the byte.
    //
    // The two lines above have the same defect and are left alone here because
    // fixing them changes what an existing control does. `screen-space` is 0, so
    // it survives the round trip by accident; `fixed-level` is 1 and does not, so
    // `dryLodMode()` never returns it and `dryLodFixedLevel()` is always 0. The
    // panel's FIXED button is consequently a no-op today.
    // The mesh threshold shares the lane block so the Frame panel's toggle
    // takes the same 16-byte write and never invalidates a lighting cache.
    floats[offset + 3] = this.renderTuning.surfaceMeshLodPixels;
  }

  /**
   * Rewrite only the level-of-detail lane.
   *
   * Mode, threshold and level change nothing the caches key on — no pass is
   * added or removed, no march shape moves, and lighting never reads them — so
   * dragging the panel's slider must not drop the world-GI cache or the
   * stationary primary. Anything less makes the slider unusable at interactive
   * rates and makes an A/B over it measure a cache rebuild in both arms.
   */
  private packMeshFilterParams(floats: Float32Array, offset: number): void {
    const t = this.renderTuning;
    floats.set([Number(t.surfaceMeshFilteringEnabled), t.surfaceMeshNormalStrength,
      t.surfaceMeshMaxCoarsening, t.surfaceMeshLodHysteresis,
      Number(t.surfaceMeshNormalSmoothing), t.surfaceMeshNormalAgreement,
      Number(t.surfaceMeshPreserveCloseNormals), t.surfaceMeshing === "dual-marching-cubes" ? 3 : t.surfaceMeshing === "dual-contouring" ? 2 : t.surfaceMeshContours ? 1 + t.surfaceMeshContourInflation : 0], offset);
  }

  private writeLodParams(): void {
    const buffer = new ArrayBuffer(16);
    const floats = new Float32Array(buffer), words = new Uint32Array(buffer);
    this.packLodParams(floats, words, 0);
    this.device.queue.writeBuffer(this.paramsBuffer, SVO_DRY_SCENE_PARAMS_LAYOUT.lodWordOffset * 4, buffer);
    // Keep the memoized snapshot in step, or the next whole-params write sees a
    // difference that is already on the device and rewrites 592 bytes for it.
    this.paramsWords?.set(words, SVO_DRY_SCENE_PARAMS_LAYOUT.lodWordOffset);
    const filtering = new Float32Array(8);
    this.packMeshFilterParams(filtering, 0);
    this.device.queue.writeBuffer(this.paramsBuffer, SVO_DRY_SCENE_PARAMS_LAYOUT.meshFilterWordOffset * 4, filtering);
    this.paramsWords?.set(new Uint32Array(filtering.buffer), SVO_DRY_SCENE_PARAMS_LAYOUT.meshFilterWordOffset);
    this.writeOccluderGhostParams();
  }

  /**
   * See-through occluders are a shading decision read from one uniform lane,
   * so the scene flag and the tuning both land as a 16-byte write and never a
   * bundle rebuild: `packOccluderGhostParams` is the only reader of either.
   */
  setSceneSeeThroughSolids(enabled: boolean): void {
    if (this.sceneSeeThroughSolids === enabled) return;
    this.sceneSeeThroughSolids = enabled;
    this.writeOccluderGhostParams();
  }

  private packOccluderGhostParams(floats: Float32Array, offset: number): void {
    const t = this.renderTuning;
    const enabled = t.occluderGhosting === "on" || (t.occluderGhosting === "auto" && this.sceneSeeThroughSolids);
    // Coverage above this along the continuation ray is "water behind". A
    // coarse texel bleeds half a texel past the surface, so the threshold sits
    // above the boundary value rather than at zero.
    floats.set([Number(enabled), t.occluderGhostOpacity, SVO_OCCLUDER_GHOST_COVERAGE_THRESHOLD, 0], offset);
  }

  private writeOccluderGhostParams(): void {
    if (!this.paramsWords) return;
    const ghost = new Float32Array(4);
    this.packOccluderGhostParams(ghost, 0);
    this.device.queue.writeBuffer(this.paramsBuffer, SVO_DRY_SCENE_PARAMS_LAYOUT.occluderGhostWordOffset * 4, ghost);
    this.paramsWords.set(new Uint32Array(ghost.buffer), SVO_DRY_SCENE_PARAMS_LAYOUT.occluderGhostWordOffset);
  }

  /**
   * Whether this frame's fragment may believe the entry-seed plane.
   *
   * Two conditions, and they are different in kind. `primaryEntryPrepassEnabled`
   * is a property of the *build* — the pass and the shader's seed path are
   * compiled together or not at all. The switch is a property of this *frame*:
   * the pipeline is unchanged and only the encode is withheld, so the fragment
   * has to be told, and this is what tells it.
   */
  private primaryEntrySeedLive(): boolean {
    return this.primaryEntryPrepassEnabled && !this.disabledStages.has("primary-entry-prepass");
  }

  /**
   * Rewrite only the entry-seed lane.
   *
   * The panel's switch withholds a pass; it does not change a shader, a bind
   * group or a bundle, and the ablation is only honest if the two arms differ
   * by the pass and nothing else. A full `writeParams` would be correct but
   * needs a live source and scene, which a switch thrown between frames has no
   * business requiring.
   */
  private writePrimaryEntryParams(): void {
    const words = new Uint32Array([this.primaryEntrySeedLive() ? 1 : 0, 0, 0, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer,
      SVO_DRY_SCENE_PARAMS_LAYOUT.primaryEntryWordOffset * 4, words);
    // Keep the memoized snapshot in step, exactly as writeLodParams does.
    this.paramsWords?.set(words, SVO_DRY_SCENE_PARAMS_LAYOUT.primaryEntryWordOffset);
  }

  private rebuild(): void {
    const source = this.source, structural = source?.structural;
    if (!this.layout || (!this.pipeline && !this.experiments.surfaceMesh) || !source || !structural || !this.scene) {
      this.bindGroup = undefined;
      this.coneFanoutSceneBindGroup = undefined;
      return;
    }
    const nodeMip = source.nodeMipPyramid;
    const tetrahedralRadiance = source.tetrahedralRadiance;
    const nodeMipPageValidity = (nodeMip as typeof nodeMip & {
      pageValidity?: { view: GPUTextureView };
    } | undefined)?.pageValidity?.view;
    const tetrahedralRadiancePageValidity = (tetrahedralRadiance as typeof tetrahedralRadiance & {
      pageValidity?: { view: GPUTextureView };
    } | undefined)?.pageValidity?.view;
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
    const derivedTraversal = this.traversalMode === "compact"
      ? compact.status === "ready" ? compact.source.nodes : undefined
      : this.traversalMode === "wide" || this.traversalMode === "hybrid"
        ? wide.status === "ready" ? wide.source.traversal : undefined
        : undefined;
    if ((this.traversalMode === "wide" || this.traversalMode === "hybrid") && !derivedTraversal) {
      this.bindGroup = undefined;
      this.coneFanoutSceneBindGroup = undefined;
      return;
    }
    const fluidCoverageView = this.fluidCoverage?.visibleGeneration()?.view ?? this.fluidCoverageFallbackView;
    this.boundFluidCoverageView = fluidCoverageView;
    this.bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.bodyBuffer } },
      { binding: 2, resource: structural.structure },
      { binding: 3, resource: structural.scenePayload },
      { binding: 4, resource: { buffer: this.sceneArenaBuffer } },
      ...(derivedTraversal ? [{ binding: 5, resource: derivedTraversal }] : []),
      { binding: 6, resource: structural.planarBoundaries.records },
      { binding: 9, resource: { buffer: this.paramsBuffer } },
      { binding: 13, resource: { buffer: this.lightingBuffer } },
      { binding: 14, resource: { buffer: this.rigidMotionUniformBuffer } },
      { binding: 15, resource: { buffer: this.thickGlassUniformBuffer } },
      { binding: 16, resource: nodeMip?.view ?? this.nodeMipFallbackAtlasView },
      { binding: 17, resource: nodeMip?.sampler ?? this.nodeMipFallbackSampler },
      { binding: 18, resource: nodeMip?.directoryView ?? this.nodeMipFallbackDirectoryView },
      { binding: 19, resource: fluidCoverageView },
      { binding: 20, resource: nodeMip?.directPageTableView ?? this.nodeMipFallbackDirectPageTableView },
      { binding: 21, resource: tetrahedralRadiance?.views[0] ?? this.tetrahedralRadianceFallbackViews[0] },
      { binding: 22, resource: tetrahedralRadiance?.views[1] ?? this.tetrahedralRadianceFallbackViews[1] },
      { binding: 23, resource: tetrahedralRadiance?.views[2] ?? this.tetrahedralRadianceFallbackViews[2] },
      { binding: 24, resource: tetrahedralRadiance?.views[3] ?? this.tetrahedralRadianceFallbackViews[3] },
      { binding: 25, resource: this.tetrahedralRadianceBlackPagesView ?? this.tetrahedralRadianceBlackFallbackView },
      { binding: 26, resource: nodeMipPageValidity ?? this.nodeMipPageValidityFallbackView },
      { binding: 27, resource: tetrahedralRadiancePageValidity ?? this.tetrahedralRadiancePageValidityFallbackView },
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
          { binding: 7, resource: fluidCoverageView },
          { binding: 8, resource: nodeMip?.directPageTableView ?? this.nodeMipFallbackDirectPageTableView },
          { binding: 9, resource: nodeMipPageValidity ?? this.nodeMipPageValidityFallbackView },
        ],
      })
      : undefined;
    this.ensureBrickRasterBuffers();
    this.rebuildBrickRasterBindGroups();
    this.ensurePrimaryEntryBuffers();
    this.rebuildPrimaryEntryBindGroups();
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
    this.writeVoxelLightCacheParams();
    this.rebuild();
  }

  /**
   * Runtime A/B and emergency fallback switch.
   *
   * Disabling releases the atlas-sized volume and its four buffers rather than
   * leaving them resident behind a cleared control word; re-enabling rebuilds
   * them cold. Cached *data* is never what survived a disable — the epoch bump
   * inside `ensureVoxelLightCache` already invalidated it — so the switch keeps
   * its contract and stops holding hundreds of megabytes for a shader that has
   * been told to ignore them. The build-time `experiments.voxelLightCache`
   * lever is untouched: that one compiles a different shader.
   */
  setVoxelLightCacheEnabled(enabled: boolean): void {
    if (this.voxelLightUserEnabled === enabled) return;
    this.voxelLightUserEnabled = enabled;
    this.ensureVoxelLightCache(this.source, this.scene);
  }

  setRigidMotionSource(source: GPUBuffer | undefined): void {
    if (!source && this.rigidMotionSource) this.device.queue.writeBuffer(this.rigidMotionUniformBuffer, 0, new Uint32Array(SVO_DRY_RIGID_MOTION_UNIFORM_BYTES / 4));
    this.rigidMotionSource = source;
  }

  ensureSize(width: number, height: number): void {
    if (this.gBufferTargets.ensureSize(width, height)) { this.pickingFrameToken += 1; this.lastPickingTarget = undefined; }
    this.targetWidth = width;
    this.targetHeight = height;
    this.ensureConePrepassTargets();
    this.ensureSplitTargets();
  }

  /** Heatmap counters are invocation-private, so diagnostics intentionally retain the inline path. */
  setDiagnosticOverlayActive(active: boolean): void {
    if (active === this.splitDiagnosticsActive) return;
    this.splitDiagnosticsActive = active;
  }

  /** Copies the compacted boundary count for offline Dawn experiment diagnosis. */
  /**
   * Per-pixel conservative candidate counts left by the last coverage pass of
   * the encoded frame — the scene-primitive one on the production arm.
   *
   * The audit this exists for is the arena's: what fraction of covered pixels
   * exceeded capacity and had to be re-marched by the overflow arm. Capacity
   * never changes the image, so this is the only way to see it move.
   */
  copyCoverageCounts(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    if (!this.brickCoverageCountBuffer || !this.brickCoverageWidth || !this.brickCoverageHeight) return false;
    const bytes = svoBrickRasterCoverageCountBytes(this.brickCoverageWidth, this.brickCoverageHeight);
    if (target.size < bytes) return false;
    encoder.copyBufferToBuffer(this.brickCoverageCountBuffer, 0, target, 0, bytes);
    return true;
  }

  /** Capacity the coverage counts above are compared against. */
  get scenePrimitiveCoverageCapacity(): number {
    return SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel;
  }

  copyConeBoundaryCount(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    if (!this.coneBoundaryCountSnapshot) return false;
    encoder.copyBufferToBuffer(this.coneBoundaryCountSnapshot, 0, target, 0, Uint32Array.BYTES_PER_ELEMENT);
    return true;
  }

  /** Copies the eight Phase-0/1 demand counters from the most recently encoded frame. */
  copyVoxelLightCacheCounters(encoder: GPUCommandEncoder, target: GPUBuffer): boolean {
    if (!this.voxelLightQueueBuffer) return false;
    encoder.copyBufferToBuffer(
      this.voxelLightQueueBuffer,
      0,
      target,
      0,
      SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.queueHeaderWords * 4,
    );
    return true;
  }

  /**
   * Copies this frame's strict live-derived failure counts as AO/direct/GI u32s.
   * The method is the renderer/UI seam: normal frames pay no map/readback cost,
   * while diagnostics can attach their existing asynchronous readback ring.
   */
  copyDerivedPageFailureCounters(encoder: GPUCommandEncoder, target: GPUBuffer, targetOffsetBytes = 0): boolean {
    if (!this.coneDerivedFailureSnapshot || !this.worldGiFrameBuffer) return false;
    if (!Number.isSafeInteger(targetOffsetBytes) || targetOffsetBytes < 0 || targetOffsetBytes % 4 !== 0
      || targetOffsetBytes + SVO_DRY_DERIVED_FAILURE_COUNTERS.sizeBytes > target.size) {
      throw new RangeError("Derived-page failure counter target exceeds its readback buffer");
    }
    encoder.copyBufferToBuffer(this.coneDerivedFailureSnapshot, 0, target, targetOffsetBytes, 8);
    encoder.copyBufferToBuffer(this.worldGiFrameBuffer, 16, target, targetOffsetBytes + 8, 4);
    return true;
  }

  /**
   * Benchmark seam for the documented warm-cache gate. The caller first
   * encodes ordinary settled frames, then this method repeats only the exact
   * production compact cone-visibility pass over the retained split G-buffer.
   * It never substitutes for encode() in the renderer's shipping path.
   */
  encodeWarmConeVisibilityProbe(encoder: GPUCommandEncoder): boolean {
    if (this.coneScale === 1 || this.conePipelineScale !== this.coneScale
      || this.splitPipelineScale !== this.coneScale || !this.conePrepassResetPipeline
      || !this.conePrepassCoherentPipeline || !this.conePrepassBoundaryPipeline
      || !this.conePrepassComputeBindGroup || !this.splitLightingBindGroup
      || !this.conePrepassBoundaryQueue || !this.targetWidth || !this.targetHeight) return false;
    // This is the principal Phase-1 win: once the single-light cache owns the
    // complete direct-visibility tier, production does not encode the
    // screen-space cone pass. Missing/rejected voxels fall through to the live
    // per-pixel chain in dryLightVisibility.
    if (this.voxelLightExclusive) return true;
    const splitGroup = 2;
    const cacheBindings = this.experiments.voxelLightCache !== false
      && this.device.limits.maxSampledTexturesPerShaderStage >= 17;
    if (cacheBindings && !this.voxelLightConsumerBindGroup) return false;
    if (this.experiments.clearConeQueueWithBlit) encoder.clearBuffer(this.conePrepassBoundaryQueue, 0, 4);
    const pass = encoder.beginComputePass({ label: "Sparse voxel compact cone visibility (warm probe)" });
    if (!this.experiments.clearConeQueueWithBlit && !this.experiments.inlineConeBoundaries) {
      pass.setPipeline(this.conePrepassResetPipeline);
      pass.setBindGroup(0, this.bindGroup!);
      pass.setBindGroup(1, this.conePrepassComputeBindGroup);
      pass.setBindGroup(splitGroup, this.splitLightingBindGroup);
      if (cacheBindings) pass.setBindGroup(splitGroup + 1, this.voxelLightConsumerBindGroup!);
      pass.dispatchWorkgroups(1);
    }
    pass.setPipeline(this.conePrepassCoherentPipeline);
    pass.setBindGroup(0, this.bindGroup!);
    pass.setBindGroup(1, this.conePrepassComputeBindGroup);
    pass.setBindGroup(splitGroup, this.splitLightingBindGroup);
    if (cacheBindings) pass.setBindGroup(splitGroup + 1, this.voxelLightConsumerBindGroup!);
    pass.dispatchWorkgroups(Math.ceil(this.conePrepassWidth / 8), Math.ceil(this.conePrepassHeight / 8));
    if (this.conePrepassBoundaryQueueActive) {
      pass.setPipeline(this.conePrepassBoundaryPipeline);
      pass.dispatchWorkgroups(Math.ceil(this.conePrepassWidth * this.conePrepassHeight / 64));
    }
    pass.end();
    return true;
  }

  /**
   * Whether the coherent kernel still queues incoherent receivers for a second,
   * scene-tracing dispatch. The default resolves them from the primary G-buffer
   * inside the coherent kernel, so the queue stays empty and its consumer is not
   * encoded at all — the entry point remains compiled as the analytic control.
   */
  private get conePrepassBoundaryQueueActive(): boolean {
    return this.experiments.analyticConeBoundaries === true
      && this.experiments.inlineConeBoundaries !== true;
  }

  get voxelLightCacheAllocatedBytes(): number {
    const nodeMip = this.source?.nodeMipPyramid;
    return this.voxelLightPageCount > 0 && nodeMip
      ? nodeMip.plan.atlas.capacity * SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.bytesPerTexel
      : 0;
  }

  /** Auxiliary MRTs and reversed-Z depth for picking and split shading. */
  get gBufferTextures(): SparseVoxelGBufferTextures | undefined {
    return this.gBufferTargets.textures;
  }

  /** Exact per-pixel work counters; present only for the benchmark diagnostic arm. */
  get primaryWorkMapTexture(): GPUTexture | undefined { return this.primaryWorkMap; }

  /**
   * Read-only views of every plane this pipeline published for the frame just
   * encoded. Absent entries are configurations that allocate no such plane —
   * full-rate cone lighting, a scene without glass, a scene without bodies —
   * and the stage overlay reports them as absent rather than inventing one.
   */
  get stagePlanes(): SvoRenderStagePlanes {
    const gBuffer = this.gBufferTargets.views;
    return {
      packedSurface: gBuffer?.packedSurface,
      identityMedia: gBuffer?.identityMedia,
      hardwareDepth: gBuffer?.hardwareDepth,
      splitGeometry: this.splitGeometryView,
      splitOpaqueIdentity: this.splitOpaqueIdentityView,
      primaryWork: this.primaryWorkMapView,
      splitGlassKey: this.splitGlassKeyView,
      rigidPrimaryGeometry: this.rasterRigidPrimaryGeometryView,
      conePrepassVisibility: this.conePrepassVisibilityView,
      conePrepassGeometry: this.conePrepassGeometryView,
      conePrepassIdentity: this.conePrepassIdentityView,
      conePrepassRadiance: this.conePrepassRadianceView,
      conePrepassWidth: this.conePrepassWidth,
      conePrepassHeight: this.conePrepassHeight,
    };
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
    // The raster half explains how the depth test found the pixel; it only
    // exists when the raster primary is the active mode, and it compiles on the
    // same first request rather than at startup. It is sequenced after the ray
    // probe because it binds that probe's request buffer — one request, so the
    // two halves cannot answer different pixels for the same frame.
    void this.ensurePixelProbe().then(() => this.ensureBrickRasterProbe());
  }

  clearPixelTraceRequest(): void { this.probeRequest = undefined; }

  /**
   * Resource/source epoch. Every republication of the scene — geometry, analytic
   * primitives, materials, the light arena — lands through `setSource` and bumps
   * this, so a caller holding an answer about the old scene can tell.
   */
  get sceneEpoch(): number { return this.pickingFrameToken; }

  /** Exact old/new transform bounds retained for the next unified sparse-page publication. */
  get latestPrimitiveDirtyBounds(): readonly SvoDrySceneDirtyBounds[] { return this.primitiveDirtyBounds; }

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
          createSvoDrySceneFragmentWGSL(1, this.experiments.surfaceMesh ? "canonical-parametric" : this.traversalMode, this.brickOccupancyMode, "inline", 0, true),
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

  private encodeBrickRasterProbe(encoder: GPUCommandEncoder): void {
    const request = this.probeRequest;
    if (!request || !this.rasterPrimary || !this.brickProbePipeline || !this.brickProbeBindGroup
      || !this.brickProbeBuffers || !this.probeBuffers || this.brickProbeReadPending) return;
    // This pass is deliberately encoded next to the raster primary, before the
    // candidate arena is reused. The queue write still precedes the eventual
    // command-buffer submit, and both probe halves therefore read one request.
    this.probeBuffers.writeRequest(request);
    const pass = encoder.beginComputePass({ label: "Sparse voxel raster-primary probe" });
    pass.setPipeline(this.brickProbePipeline);
    pass.setBindGroup(0, this.brickProbeBindGroup);
    // One workgroup: ordinary frames consume at most the 24 candidates already
    // published for this pixel; overflow/direct experiments retain the strided
    // instance-list fallback. Ordering and election remain a lane-zero epilogue.
    pass.dispatchWorkgroups(1);
    pass.end();
    this.brickProbeReadPending = this.brickProbeBuffers.encodeReadback(encoder);
  }

  /**
   * Resolve the encoded trace. Resolves to `undefined` when the request was
   * superseded before the map completed, which is the common case while the
   * pointer is moving.
   */
  async readPixelTrace(): Promise<SvoPixelTrace | undefined> {
    if (!this.probeBuffers || !this.probeReadPending) return undefined;
    // Queue order makes this a valid snapshot of the scene that was encoded.
    // In-place animation may advance the scene epoch before mapAsync resolves;
    // that makes the answer stale, not invalid. Only replacement of the actual
    // staging buffers invalidates the pending decode.
    const probeBuffers = this.probeBuffers;
    const brickProbeBuffers = this.brickProbeBuffers;
    const current = () => this.probeBuffers === probeBuffers
      && this.brickProbeBuffers === brickProbeBuffers;
    try {
      // Both halves are mapped together and folded into one account of the
      // pixel. The raster half is optional throughout: in traced mode it never
      // runs, and if its pipeline failed the lighting half still stands alone.
      const [lighting, primary] = await Promise.all([
        probeBuffers.read(current),
        this.brickProbeReadPending && brickProbeBuffers
          ? brickProbeBuffers.read(current)
          : Promise.resolve(undefined),
      ]);
      // The prepass flag is the host's to add: the probe is composed inline at
      // full rate, so from inside it the reduced pass beside it is invisible.
      return withSvoPixelTraceConePrepass(mergeSvoPixelTrace(lighting, primary), this.probeEncodedConePrepass);
    } finally {
      this.probeReadPending = false;
      this.brickProbeReadPending = false;
    }
  }

  encode(encoder: GPUCommandEncoder, target: GPUTexture | GPUTextureView, tracePhase?: RenderFrameSeam<"svo">, bandPartitioner?: FrameBandPartitioner): DrySceneReplacementResult | false {
    if ((!this.pipeline && !this.surfaceMeshPipelines) || !this.bindGroup) return false;
    // The coverage volume allocates lazily and only reports itself once a fill
    // has been encoded, so its validity flips mid-session. Refresh the frame
    // every encode rather than relying on a source change to carry it.
    this.refreshFluidCoverageFrame();
    const gBufferViews = this.gBufferTargets.views;
    if (!gBufferViews) return false;
    this.requestedBundleResourceFailure = undefined;
    if (this.presentationBundleStatus.state !== "ready") return false;
    const usePrepass = this.coneScale !== 1 && this.conePipelineScale === this.coneScale
      && Boolean(this.conePrepassGeometryPipeline && this.conePrepassVisibilityPipeline
        && this.conePrepassShadePipeline && this.coneReducedPipeline
        && this.conePrepassBindGroup && this.conePrepassVisibilityBindGroup && this.conePrepassShadeBindGroup && this.conePrepassVisibilityView
        && this.conePrepassGeometryView && this.conePrepassIdentityView && this.conePrepassRadianceView);
    const reconstructReducedRadiance = this.renderTuning.coneRadianceReconstruction !== "wide-relight"
      && this.renderTuning.coneRadianceReconstruction !== "full-res-relight";
    const effectiveScale: SvoConeLightingScale = usePrepass ? this.coneScale : 1;
    // Recorded for the probe, which is encoded after this frame's passes and
    // cannot observe the reduced pass from inside its own full-rate composition.
    this.probeEncodedConePrepass = usePrepass;
    const splitRequested = this.shadingPath === "split";
    // A withheld rigid tier must also withdraw the rigid-reading visibility
    // variant: the dearer shader would otherwise keep running against a plane
    // this frame never wrote, pricing the "off" arm with stale-data work.
    const rasterRigidEncoded = this.rasterRigidActive && !this.disabledStages.has("rigid-impostor");
    const activeSplitVisibilityPipeline = rasterRigidEncoded
      ? this.splitRasterRigidVisibilityPipeline
      : this.splitVisibilityPipeline;
    const brickRasterReady = Boolean(this.brickBackgroundPipeline && this.brickRasterPipeline
      && this.brickCoveragePipeline && this.brickCoverageResolvePipeline && this.brickCoverageOverflowPipeline
      && (this.screenSpaceTerminationPixels === 0 || (this.brickLodResolvePipeline && this.brickExactResolvePipeline))
      && this.scenePrimitiveRasterPipeline
      && (this.scenePrimitiveDirect || (this.scenePrimitiveCoveragePipeline
        && this.scenePrimitiveCoverageResolvePipeline && this.scenePrimitiveCoverageOverflowPipeline
        && (this.screenSpaceTerminationPixels === 0 || this.scenePrimitiveLodResolvePipeline)))
      && this.brickEmitPipeline && this.brickScanPipeline && this.brickScatterPipeline
      && this.brickCullBindGroup && this.brickDrawBindGroup && this.brickCoverageBindGroup
      && this.brickCoverageResolveBindGroup
      && (this.screenSpaceTerminationPixels === 0 || this.scenePrimitiveCoverageBindGroup)
      && this.brickResolveSceneBindGroup && this.brickSortStateBuffer
      && this.brickCoverageCountBuffer && this.brickCoverageCandidateBuffer
      && this.splitOpaqueIdentityView);
    const voxelLightBindingsRequired = this.experiments.voxelLightCache !== false
      && this.device.limits.maxSampledTexturesPerShaderStage >= 17;
    const voxelLightBindingsReady = Boolean(this.voxelLightConsumerBindGroup);
    const useSplit = splitRequested && !this.splitDiagnosticsActive
      && this.splitPipelineScale === effectiveScale
        && (!this.rasterPrimary || brickRasterReady)
        && (!this.silhouetteRefinementEnabled || Boolean(this.primarySeamClosurePipeline))
        && Boolean(activeSplitVisibilityPipeline && this.splitLightingPipeline && this.splitSkyLightingPipeline
        && (!usePrepass || !reconstructReducedRadiance || this.splitReconstructedLightingPipeline)
        && (!usePrepass || (this.conePrepassResetPipeline && this.conePrepassCoherentPipeline && this.conePrepassBoundaryPipeline
          && this.conePrepassComputeBindGroup && this.conePrepassBoundaryQueue && this.coneBoundaryCountSnapshot
          && this.coneDerivedFailureSnapshot
          && this.worldGiFramePipeline && this.worldGiCachePipeline && this.worldGiCacheBindGroup
          && this.worldGiCacheBuffer && this.worldGiFrameBuffer
          && (!this.coneFanout || (this.coneFanoutWorkerPipeline && this.coneFanoutReducerPipeline
            && this.coneFanoutSceneBindGroup && this.coneFanoutWorkerBindGroup && this.coneFanoutReducerBindGroup))))
        && (!this.rasterGlassDiscovery || (this.rasterGlassPipeline && this.rasterGlassBindGroup && this.splitGlassKeyView && this.splitGlassDepthView))
        && (!this.rasterRigidActive || (this.rasterRigidPipeline && this.rasterRigidBridgePipeline
          && this.rasterRigidInputBindGroup && this.rasterRigidBindGroup && this.rasterRigidPrimaryGeometryView))
        && (!voxelLightBindingsRequired || voxelLightBindingsReady)
        // The seed plane's cleared value is read as a proof that no voxel leaf
        // covers the pixel, which is only true if this pass wrote the whole
        // plane this frame. Half-built prepass resources must fall back rather
        // than let the megakernel read a plane nobody filled.
        && (!this.primaryEntryPrepassEnabled || this.primaryEntryPrepassReady)
        && this.splitVisibilityBindGroup && this.splitLightingBindGroup && this.splitGeometryView);
    if (this.coneScale !== 1 && !usePrepass) {
      this.requestedBundleResourceFailure = `Requested SVO cone bundle at scale ${this.coneScale} has incomplete frame resources`;
      return false;
    }
    if (splitRequested && !this.splitDiagnosticsActive && !useSplit) {
      this.requestedBundleResourceFailure = `Requested SVO split bundle at scale ${this.coneScale} has incomplete frame resources`;
      return false;
    }
    const targetTexture = "width" in target ? target as GPUTexture : undefined;
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
      if (!this.disabledStages.has("reduced-shade")) {
        shade.setPipeline(this.conePrepassShadePipeline!);
        shade.setBindGroup(0, this.bindGroup);
        shade.setBindGroup(1, this.conePrepassShadeBindGroup!);
        shade.draw(3);
      }
      shade.end();
      tracePhase?.("cone-prepass");
    }
    if (useSplit) {
      const splitGroup = usePrepass ? 2 : 1;
        // A withheld primary still clears. The G-buffer has to be a defined
        // miss everywhere so the frame resolves to sky and the delta is exactly
        // what the traversal was worth; presenting whatever the last traced
        // frame left in the attachment would measure nothing and look right.
        const primaryWithheld = this.disabledStages.has("primary-traversal");
        if (primaryWithheld) {
          encoder.beginRenderPass({
            label: "Sparse voxel primary visibility · withheld",
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
          }).end();
        } else if (this.rasterPrimary) {
          this.encodeRasterPrimary(encoder, gBufferViews, usePrepass, splitGroup, tracePhase);
        } else {
          if (this.primaryEntryPrepassEnabled) {
            // The seam closes either way: a withheld stage that encoded no pass
            // is a true zero, and that is what the panel has to be able to say.
            if (this.primaryEntrySeedLive()) this.encodePrimaryEntryPrepass(encoder);
            tracePhase?.("primary-entry-prepass");
          }
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
        }
        if (!this.experiments.surfaceMesh) tracePhase?.("primary-traversal");
        else if (primaryWithheld) tracePhase?.("surface-mesh-draw");
        if (this.rasterPrimary && !this.experiments.surfaceMesh && !primaryWithheld && !this.disabledStages.has("scene-primitive")) {
          this.encodeScenePrimitivePrimary(encoder, gBufferViews, usePrepass, splitGroup, tracePhase);
        }
        if (rasterRigidEncoded && !primaryWithheld) {
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
          tracePhase?.("rigid-discovery");
        }
        if (this.rasterGlassDiscovery && this.rasterGlassPaneCount > 0 && !primaryWithheld) {
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
          // Withheld keeps the clears — the lighting pass samples the glass key
          // plane unconditionally, and a skipped clear left it replaying last
          // frame's panes, the one deviation from the keep-clears contract.
          if (!this.disabledStages.has("thin-glass")) {
            glass.setPipeline(this.rasterGlassPipeline!);
            glass.setBindGroup(0, this.bindGroup);
            glass.setBindGroup(1, this.rasterGlassBindGroup!);
            glass.draw(6, this.rasterGlassRecordCount, 0, this.rasterGlassFirstRecord);
          }
          glass.end();
          tracePhase?.("thin-glass-discovery");
        }
        // Nothing to close a seam around once the primary is withheld, and the
        // pass would otherwise charge the seam node for reading an empty
        // G-buffer.
        if (this.silhouetteRefinementEnabled && !primaryWithheld) {
          const seam = encoder.beginRenderPass({
            label: "Sparse voxel primary seam closure",
            colorAttachments: [
              { view: gBufferViews.packedSurface, loadOp: "load", storeOp: "store" },
              { view: gBufferViews.identityMedia, loadOp: "load", storeOp: "store" },
            ],
            depthStencilAttachment: {
              view: gBufferViews.hardwareDepth,
              depthLoadOp: "load",
              depthStoreOp: "store",
            },
          });
          seam.setPipeline(this.primarySeamClosurePipeline!);
          seam.setBindGroup(0, this.bindGroup);
          if (usePrepass) seam.setBindGroup(1, this.conePrepassBindGroup!);
          seam.setBindGroup(splitGroup, this.splitLightingBindGroup!);
          if (voxelLightBindingsRequired) seam.setBindGroup(splitGroup + 1, this.voxelLightConsumerBindGroup!);
          seam.draw(3);
          seam.end();
          tracePhase?.("seam-closure");
        }
      // Band seam on fence-partitioned sampling frames: everything above is
      // primary visibility, everything until the next seam is lighting
      // visibility. The reassignment is load-bearing — the old encoder was
      // submitted by the boundary and must not be encoded into again.
      if (bandPartitioner) encoder = bandPartitioner.boundary("svo-primary");

      if (this.voxelLightActive && !this.disabledStages.has("voxel-light-cache")
        && this.voxelLightDemandPipeline && this.voxelLightPopulatePipeline
        && this.voxelLightDemandBindGroup && this.voxelLightPopulateBindGroup
        && this.voxelLightRequestBuffer && this.voxelLightQueueBuffer) {
        encoder.clearBuffer(this.voxelLightRequestBuffer);
        encoder.clearBuffer(this.voxelLightQueueBuffer, 0, SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.queueHeaderWords * 4);
        const cacheGroup = splitGroup + 1;
        const demand = encoder.beginComputePass({ label: "Sparse voxel directional-light cache demand" });
        demand.setPipeline(this.voxelLightDemandPipeline);
        demand.setBindGroup(0, this.bindGroup);
        if (usePrepass) demand.setBindGroup(1, this.conePrepassBindGroup!);
        demand.setBindGroup(splitGroup, this.splitLightingBindGroup!);
        demand.setBindGroup(cacheGroup, this.voxelLightDemandBindGroup);
        demand.dispatchWorkgroups(Math.ceil(this.targetWidth / 8), Math.ceil(this.targetHeight / 8));
        demand.end();
        const populate = encoder.beginComputePass({ label: "Sparse voxel directional-light cache population" });
        populate.setPipeline(this.voxelLightPopulatePipeline);
        populate.setBindGroup(0, this.bindGroup);
        if (usePrepass) populate.setBindGroup(1, this.conePrepassBindGroup!);
        populate.setBindGroup(splitGroup, this.splitLightingBindGroup!);
        populate.setBindGroup(cacheGroup, this.voxelLightPopulateBindGroup);
        populate.dispatchWorkgroups(Math.ceil(SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.populationBudget / 64));
        populate.end();
        // Its own boundary: both dispatches used to fall inside whichever phase
        // closed next, which priced a bounded per-frame drain as part of the
        // cone stage and left the cache node with nothing to report.
        tracePhase?.("voxel-light-cache");
      }

      if (usePrepass && !this.voxelLightExclusive) {
        if (this.experiments.clearConeQueueWithBlit) encoder.clearBuffer(this.conePrepassBoundaryQueue!, 0, 16);
        const coherent = encoder.beginComputePass({ label: "Sparse voxel compact cone visibility" });
        if (!this.experiments.clearConeQueueWithBlit && !this.experiments.inlineConeBoundaries) {
          coherent.setPipeline(this.conePrepassResetPipeline!);
          coherent.setBindGroup(0, this.bindGroup);
          coherent.setBindGroup(1, this.conePrepassComputeBindGroup!);
          coherent.setBindGroup(splitGroup, this.splitLightingBindGroup!);
          if (voxelLightBindingsRequired) coherent.setBindGroup(splitGroup + 1, this.voxelLightConsumerBindGroup!);
          coherent.dispatchWorkgroups(1);
        }
        coherent.setPipeline(this.conePrepassCoherentPipeline!);
        coherent.setBindGroup(0, this.bindGroup);
        coherent.setBindGroup(1, this.conePrepassComputeBindGroup!);
        coherent.setBindGroup(splitGroup, this.splitLightingBindGroup!);
        if (voxelLightBindingsRequired) coherent.setBindGroup(splitGroup + 1, this.voxelLightConsumerBindGroup!);
        coherent.dispatchWorkgroups(Math.ceil(this.conePrepassWidth / 8), Math.ceil(this.conePrepassHeight / 8));
        if (this.conePrepassBoundaryQueueActive) {
          coherent.setPipeline(this.conePrepassBoundaryPipeline!);
          coherent.dispatchWorkgroups(Math.ceil(this.conePrepassWidth * this.conePrepassHeight / 64));
        }
        coherent.end();
        // The same bounded queue is recycled below for silhouette work. Keep
        // the compact-pass metric stable for diagnostics before resetting it.
        encoder.copyBufferToBuffer(this.conePrepassBoundaryQueue!, 0,
          this.coneBoundaryCountSnapshot!, 0, Uint32Array.BYTES_PER_ELEMENT);
        encoder.copyBufferToBuffer(this.conePrepassBoundaryQueue!, Uint32Array.BYTES_PER_ELEMENT,
          this.coneDerivedFailureSnapshot!, 0, 2 * Uint32Array.BYTES_PER_ELEMENT);
        // Closed before the optional fan-out so the compact march is priced on
        // its own; the fan-out then reports as itself below rather than being
        // folded into a phase named for a pass it is not part of.
        tracePhase?.("compact-cone-lighting");
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
          tracePhase?.("cone-fanout");
        }

        if (!reconstructReducedRadiance && this.lightingOptions.globalIlluminationEnabled === true
          && this.lightingOptions.worldGiCacheEnabled === true) {
          // The cache is world-space and source-owned: camera motion changes
          // which keys are queried but never invalidates entries. Only source,
          // authored-scene, or lighting-contract changes clear it. Its output is
          // a GI closure, not final surface radiance, so only the full-rate
          // relight modes are allowed to consume it as such.
          if (this.worldGiCacheDirty) {
            encoder.clearBuffer(this.worldGiCacheBuffer!);
            this.worldGiCacheDirty = false;
          }
          const gi = encoder.beginComputePass({ label: "Sparse voxel persistent world GI cache" });
          gi.setPipeline(this.worldGiFramePipeline!);
          gi.setBindGroup(0, this.bindGroup);
          gi.setBindGroup(1, this.conePrepassShadeBindGroup!);
          gi.setBindGroup(2, this.worldGiCacheBindGroup!);
          if (voxelLightBindingsRequired) gi.setBindGroup(3, this.voxelLightConsumerBindGroup!);
          gi.dispatchWorkgroups(1);
          gi.setPipeline(this.worldGiCachePipeline!);
          gi.setBindGroup(0, this.bindGroup);
          gi.setBindGroup(1, this.conePrepassShadeBindGroup!);
          gi.setBindGroup(2, this.worldGiCacheBindGroup!);
          if (voxelLightBindingsRequired) gi.setBindGroup(3, this.voxelLightConsumerBindGroup!);
          gi.dispatchWorkgroups(Math.ceil(this.conePrepassWidth / 8), Math.ceil(this.conePrepassHeight / 8));
          gi.end();
          tracePhase?.("world-gi-cache");
        }
      }
      // Band seam: lighting visibility (voxel light cache, cone stage, world
      // GI) ends here; the reduced-rate and deferred shading follow.
      if (bandPartitioner) encoder = bandPartitioner.boundary("svo-lighting");

      if (usePrepass && reconstructReducedRadiance && !this.voxelLightExclusive) {
        // Reconstruction modes interpolate a complete reduced-rate material
        // result. The world-GI cache above deliberately stores only
        // {indirect radiance, visibility}; using that texture as final radiance
        // made directly lit terrain nearly black. Evaluate the full material at
        // reduced rate after visibility is ready, then let the deferred pass
        // reconstruct this colour with its depth/normal/identity guide.
        const shade = encoder.beginRenderPass({
          label: "Sparse voxel reduced-rate opaque shading",
          colorAttachments: [{
            view: this.conePrepassRadianceView!,
            clearValue: { r: 0, g: 0, b: 0, a: -1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        // Withholding the draw keeps the clear, so the reconstruction below
        // guides off a defined (empty) reduced result instead of last frame's.
        if (!this.disabledStages.has("reduced-shade")) {
          shade.setPipeline(this.conePrepassShadePipeline!);
          shade.setBindGroup(0, this.bindGroup);
          shade.setBindGroup(1, this.conePrepassShadeBindGroup!);
          shade.draw(3);
        }
        shade.end();
        tracePhase?.("reduced-shade");
      }

      // Two passes over complementary depth tests: sky takes the miss pixels
      // and the deferred shader takes the rest, so together they write every
      // pixel exactly once. Splitting them costs one extra load/store cycle of
      // the HDR attachment and buys each half its own label, seam, and switch —
      // sky was a known ~0.85 ms the panel could never show.
      const sky = encoder.beginRenderPass({
        label: "Sparse voxel deferred sky lighting",
        colorAttachments: [{ view: targetView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
        // Read-only: the pass classifies pixels by the depth primary visibility
        // already established and never contributes to it, so the attachment
        // costs a tile load and no store.
        depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthReadOnly: true },
      });
      // Withheld, this is the pass's clear and nothing else: the miss pixels go
      // black rather than keeping the previous sky, and the delta is exactly
      // what the sky resolve was worth.
      if (!this.disabledStages.has("sky-lighting")) {
        sky.setBindGroup(0, this.bindGroup);
        if (usePrepass) sky.setBindGroup(1, this.conePrepassBindGroup!);
        sky.setBindGroup(splitGroup, this.splitLightingBindGroup!);
        if (voxelLightBindingsRequired) sky.setBindGroup(splitGroup + 1, this.voxelLightConsumerBindGroup!);
        sky.setPipeline(this.splitSkyLightingPipeline!);
        sky.draw(3);
      }
      sky.end();
      tracePhase?.("sky-lighting");
      const lighting = encoder.beginRenderPass({
        label: "Sparse voxel deferred dry lighting",
        // Load, not clear: the sky pass owns the miss pixels and already wrote
        // them; this pass shades only where the depth buffer holds a surface.
        colorAttachments: [{ view: targetView, loadOp: "load", storeOp: "store" }],
        depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthReadOnly: true },
      });
      // Withheld, the surface pixels keep the sky pass's clear — the frame's
      // geometry goes black while the sky stays, so the delta is the whole
      // cost of deferred shading and the image says which half is missing.
      if (!this.disabledStages.has("deferred-lighting")) {
        lighting.setBindGroup(0, this.bindGroup);
        if (usePrepass) lighting.setBindGroup(1, this.conePrepassBindGroup!);
        lighting.setBindGroup(splitGroup, this.splitLightingBindGroup!);
        if (voxelLightBindingsRequired) lighting.setBindGroup(splitGroup + 1, this.voxelLightConsumerBindGroup!);
        if (usePrepass && reconstructReducedRadiance && this.experiments.singlePassReconstruction === false) {
          lighting.setPipeline(this.splitReconstructedLightingPipeline!);
          lighting.draw(3);
        }
        lighting.setPipeline(this.splitOptimizedLightingPipeline && usePrepass
          && canUseOpaqueDirectionalCones(this.scene, {
            coneMode: this.lightingOptions.coneTracingMode ?? "cones",
            hierarchyReady: this.derivedLightingReady(),
            globalIllumination: this.lightingOptions.globalIlluminationEnabled === true,
            reconstruction: this.renderTuning.coneRadianceReconstruction,
          }) ? this.splitOptimizedLightingPipeline : this.splitLightingPipeline!);
        lighting.draw(3);
      }
      lighting.end();
      tracePhase?.("deferred-lighting");
      // Band seam: deferred shading ends here; interfaces, composite, overlays
      // and present accumulate on the fresh encoder as the final band.
      if (bandPartitioner) encoder = bandPartitioner.boundary("svo-shading");
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
      // The inline arm marches and shades in one draw, so it is the primary and
      // the deferred lighting at once; either switch withholds the whole thing,
      // and its row is priced as the pair it actually is.
      if (!this.disabledStages.has("primary-traversal") && !this.disabledStages.has("deferred-lighting")) {
        pass.setPipeline(usePrepass ? this.coneReducedPipeline! : this.pipeline!);
        pass.setBindGroup(0, this.bindGroup);
        if (usePrepass) pass.setBindGroup(1, this.conePrepassBindGroup!);
        pass.draw(3);
      }
      pass.end();
      tracePhase?.("inline-traversal-shading");
      // The inline arm marches and shades in one draw, and the panel already
      // prices it as that pair under the primary row; the band seam follows
      // the same attribution so the megakernel never hides in the composite.
      if (bandPartitioner) encoder = bandPartitioner.boundary("svo-primary");
    }
    this.lastPickingTarget = targetTexture;
    const result = { encoded: true, sampledTargetView: targetView } as const;
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
    this.brickProbeBuffers?.destroy();
    this.brickProbeBuffers = undefined;
    this.brickProbeBindGroup = undefined;
    this.brickProbePipeline = undefined;
    this.brickProbeReadPending = false;
    this.brickCandidateBuffer?.destroy();
    this.brickRasterPublicationBuffer?.destroy();
    this.brickCoverageCountBuffer?.destroy();
    this.brickCoverageCandidateBuffer?.destroy();
    this.bandStateBuffer?.destroy();
    this.bandParamsBuffer?.destroy();
    this.coverageAuditBuffer?.destroy();
    if (!this.coverageAuditReading) this.coverageAuditStaging?.destroy();
    this.scenePrimitiveOverflowPublicationBuffer?.destroy();
    this.coverageOverflowIndirectBuffer?.destroy();
    this.scenePrimitiveOverflowPublicationBuffer = undefined;
    this.coverageOverflowIndirectBuffer = undefined;
    this.brickOverflowArgsBindGroup = undefined;
    this.scenePrimitiveOverflowArgsBindGroup = undefined;
    this.coverageOverflowArgsPipeline = undefined;
    this.bandStateBuffer = undefined;
    this.bandParamsBuffer = undefined;
    this.coverageAuditBuffer = undefined;
    this.coverageAuditStaging = undefined;
    this.bandComputeBindGroup = undefined;
    this.bandReadBindGroup = undefined;
    this.bandClassifyPipeline = undefined;
    this.bandResolvePipeline = undefined;
    this.brickCandidateBuffer = undefined;
    this.brickInstanceBuffer = undefined;
    this.brickSortStateBuffer = undefined;
    this.brickRasterPublicationBuffer = undefined;
    this.brickSortStateOffsetBytes = 0;
    this.brickInstanceOffsetBytes = 0;
    this.brickCoverageCountBuffer = undefined;
    this.brickCoverageCandidateBuffer = undefined;
    this.brickCoveragePipeline = undefined;
    this.brickCoverageResolvePipeline = undefined;
    this.brickLodResolvePipeline = undefined;
    this.brickExactResolvePipeline = undefined;
    this.brickCoverageOverflowPipeline = undefined;
    this.brickCoverageWidth = 0;
    this.brickCoverageHeight = 0;
    this.scenePrimitiveRasterPipeline = undefined;
    this.scenePrimitiveCoveragePipeline = undefined;
    this.scenePrimitiveLodResolvePipeline = undefined;
    this.scenePrimitiveComputeArgsPipeline = undefined;
    this.scenePrimitiveComputeResolvePipeline = undefined;
    this.scenePrimitiveDepthBridgePipeline = undefined;
    this.scenePrimitiveCoverageResolvePipeline = undefined;
    this.scenePrimitiveCoverageOverflowPipeline = undefined;
    this.brickLeafCapacity = 0;
    this.primaryEntryPublicationBuffer?.destroy();
    this.primaryEntrySeed?.destroy();
    this.primaryEntryDepth?.destroy();
    this.primaryEntryPublicationBuffer = undefined;
    this.primaryEntrySeed = undefined;
    this.primaryEntrySeedView = undefined;
    this.primaryEntryDepth = undefined;
    this.primaryEntryDepthView = undefined;
    this.primaryEntryCullBindGroup = undefined;
    this.primaryEntryDrawBindGroup = undefined;
    this.primaryEntryCullPipeline = undefined;
    this.primaryEntryDrawPipeline = undefined;
    this.primaryEntryCompilation = undefined;
    this.primaryEntryLeafCapacity = 0;
    this.splitGeometry?.destroy();
    this.splitOpaqueIdentity?.destroy();
    this.primaryWorkMap?.destroy();
    this.splitGlassKey?.destroy();
    this.splitGlassDepth?.destroy();
    this.rasterRigidPrimaryGeometry?.destroy();
    this.splitGeometry = undefined;
    this.splitGeometryView = undefined;
    this.splitOpaqueIdentity = undefined;
    this.splitOpaqueIdentityView = undefined;
    this.primaryWorkMap = undefined;
    this.primaryWorkMapView = undefined;
    this.scenePrimitiveComputeDepth?.destroy();
    this.scenePrimitiveComputeQueue?.destroy();
    this.scenePrimitiveComputeIndirect?.destroy();
    this.scenePrimitiveComputeDepth = undefined;
    this.scenePrimitiveComputeDepthView = undefined;
    this.scenePrimitiveComputeQueue = undefined;
    this.scenePrimitiveComputeIndirect = undefined;
    this.scenePrimitiveComputeOutputBindGroup = undefined;
    this.scenePrimitiveDepthBridgeBindGroup = undefined;
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
    this.primarySeamClosurePipeline = undefined;
    this.splitLightingPipeline = undefined;
    this.splitOptimizedLightingPipeline = undefined;
    this.splitReconstructedLightingPipeline = undefined;
    this.splitSkyLightingPipeline = undefined;
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
    this.releaseVoxelLightCache();
    this.voxelLightDemandPipeline = undefined;
    this.voxelLightPopulatePipeline = undefined;
    this.voxelLightConsumerLayout = undefined;
    this.voxelLightDemandLayout = undefined;
    this.voxelLightPopulateLayout = undefined;
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
    this.sceneArenaBuffer.destroy();
    this.paramsBuffer.destroy();
    this.surfaceMeshDisposed = true;
    this.surfaceMeshReadback?.destroy();
    this.surfaceMeshDispatch?.destroy();
    this.surfaceMeshState?.destroy();
    this.surfaceMeshArenas[0]?.destroy();
    this.surfaceMeshArenas[1]?.destroy();
    this.surfaceMeshEmptyArena?.destroy();
    this.surfaceMeshEmptyMaintenance?.destroy();
    this.surfaceMeshWork?.destroy();
    this.surfaceMeshVisible?.destroy();
    this.lightingBuffer.destroy();
    this.rigidMotionUniformBuffer.destroy();
    this.thickGlassUniformBuffer.destroy();
    this.rasterGlassParamsBuffer.destroy();
    this.coneFanoutFrameBuffer?.destroy();
    this.coneFanoutFrameBuffer = undefined;
    this.nodeMipFallbackAtlas.destroy();
    this.nodeMipFallbackDirectory.destroy();
    this.nodeMipFallbackDirectPageTable.destroy();
    this.nodeMipPageValidityFallback.destroy();
    this.tetrahedralRadianceFallback.forEach((texture) => texture.destroy());
    this.tetrahedralRadianceBlackFallback.destroy();
    this.tetrahedralRadiancePageValidityFallback.destroy();
    this.tetrahedralRadianceBlackPages?.destroy();
    this.tetrahedralRadianceBlackPages = undefined;
    this.tetrahedralRadianceBlackPagesView = undefined;
    this.gBufferTargets.destroy();
    this.pickingReadback.destroy();
    this.lastPickingTarget = undefined;
    this.pickingFrameToken += 1;
    this.bindGroup = undefined;
    this.primitiveCandidateArena = undefined;
    this.paramsWords = undefined;
  }

}
