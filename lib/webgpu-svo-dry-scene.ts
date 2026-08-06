import {
  svoClusterFieldByCode,
  svoClusterFieldName,
  svoPrimitiveWGSL,
  SVO_CLUSTER_FIELD_TABLE,
  SVO_CLUSTER_LOBE_DEFAULT_DISPLACEMENT,
  SVO_CLUSTER_LOBE_DEFAULT_SPAN,
  SVO_CLUSTER_LOBE_DEFAULT_SPAN_SPREAD,
  SVO_CLUSTER_SWEEP_MAXIMUM_POINTS,
  SVO_PRIMITIVE_KINDS,
  SVO_PRIMITIVE_RECORD_STRIDE_BYTES,
  SVO_PRIMITIVE_RECORD_WORDS,
  SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS,
  type SvoClusterResolver,
  type SvoFieldProgramResolver,
  type SvoSmoothUnionClusterPacking,
  type SvoTerrainHeightfieldPrimitive,
  type SvoTerrainResolver,
} from "./svo-primitive-abi";
import {
  packSvoFieldProgramArena,
  svoFieldProgramWGSL,
  unpackSvoFieldProgram,
  SVO_FIELD_PROGRAM_BLOCK_WORDS,
  type SvoFieldProgram,
} from "./svo-field-program";
import type { ResourcePluginDefinition } from "./resource-readiness";

/** Lifecycle metadata lives with the sparse presentation programs it describes. */
export const svoPresentationResourcePlugin: ResourcePluginDefinition = Object.freeze({
  id: "presentation.svo-global",
  lane: "svo",
  label: "GLOBAL sparse voxel presentation",
  provides: ["sparse-voxel-presentation"] as const,
  blocks: "viewport",
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
import {
  SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK,
  packSvoPrimitiveCandidateArena,
  type SvoPrimitiveCandidateArena,
  type SvoPrimitiveCandidatePublication,
} from "./svo-primitive-candidates";
import { SVO_PRIMITIVE_MOTION_STRIDE_BYTES, svoPrimitiveMotionWGSL } from "./svo-primitive-motion";
import { svoGBufferWGSL } from "./svo-gbuffer";
import type { SvoRenderStagePlanes } from "./webgpu-svo-stage-overlay";
import {
  buildSvoEnvironmentLighting,
  SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
  SVO_ENVIRONMENT_LIGHTING_VERSION,
  svoEnvironmentLightingWGSL,
} from "./svo-environment-lighting";
import {
  SVO_LIGHT_MAXIMUM_RECORDS,
  SVO_LIGHT_KINDS,
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
import {
  MAX_TERRAIN_DERIVED_GRID_SAMPLES,
  MIN_TERRAIN_GRID_SIZE,
  TERRAIN_CEILING_MARGIN_M,
  terrainCeiling,
  terrainGridExtent,
  terrainHeightAt,
  terrainNormalAt,
  terrainSampleGrid,
  type TerrainDescription,
  type TerrainGrid,
} from "./terrain";
import { cameraApertureShaderLibrary } from "./webgpu-camera";
import { unifiedLightingShaderLibrary, WATER_OPTICS } from "./webgpu-lighting";
import { liveSvoDerivedPageValidityWGSL } from "./webgpu-svo-live-derived-cache";
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
  type SparseVoxelGBufferViews,
} from "./webgpu-svo-gbuffer-targets";
import {
  SparseVoxelGpuPickingReadbackRing,
  svoPickingPixelFromNormalized,
  type SvoGpuPickingReadbackResult,
} from "./webgpu-svo-picking-readback";
import { SPARSE_VOXEL_VALID_FIELDS, type SparseVoxelSceneRenderSource } from "./webgpu-voxel-debug";
import {
  DEFAULT_SVO_LIGHTING_OPTIONS,
  type SvoLightingOptions,
  type SvoLightingVisibilityStatus,
  type SvoSilhouetteRefinementStatus,
} from "./svo-render-options";
import type { DrySceneReplacementResult, RenderPathTracePhase } from "./webgpu-water-pipeline";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { svoFluidCoverageWGSL } from "./svo-fluid-coverage";
import type { WebGpuSvoFluidCoverage } from "./webgpu-svo-fluid-coverage";
import { svoNodeMipSamplingWGSL } from "./svo-node-mip-sampling";
import { SVO_NODE_MIP_LAYOUT } from "./svo-node-mip-pyramid";
import { svoTetrahedralRadianceWGSL } from "./svo-tetrahedral-radiance";
import { svoTetrahedralRadianceConeCoreWGSL } from "./svo-tetrahedral-radiance-cone";
import { svoBrickOccupancyWGSL } from "./svo-brick-occupancy";
import {
  createSvoBrickRasterCullWGSL,
  createSvoRasterCoverageOverflowArgsWGSL,
  svoBrickRasterCullBindGroupLayoutEntries,
  svoBrickRasterCoverageBindGroupLayoutEntries,
  svoBrickRasterCoverageCountBytes,
  svoRasterCoverageArenaBytes,
  svoRasterCoverageCountAllocationBytes,
  svoRasterCoverageOverflowArgsBindGroupLayoutEntries,
  svoRasterCoverageOverflowDrawArgsOffsetBytes,
  svoRasterCoverageOverflowSignalWGSL,
  svoRasterCoverageOverflowStatus,
  svoBrickRasterDrawBindGroupLayoutEntries,
  svoBrickRasterInstanceBytes,
  svoBrickRasterPublicationInstanceOffsetBytes,
  svoBrickRasterSharedWGSL,
  svoBrickRasterSortStateBytes,
  SVO_BRICK_RASTER_CONTRACT,
  SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT,
} from "./webgpu-svo-brick-raster";
import {
  createSvoBrickRasterProbeWGSL,
  svoBrickRasterProbeBindGroupLayoutEntries,
  SparseVoxelBrickRasterProbeBuffers,
  SVO_BRICK_RASTER_PROBE_CONTRACT,
} from "./webgpu-svo-brick-raster-probe";
import {
  SPARSE_BRICK_NO_OWNER, octreeLiveSceneSceneGeometryFormat, sparseBrickSceneGeometryCodecWGSL,
  type SparseBrickSceneGeometryFormat,
} from "./sparse-brick-octree";
import {
  createSvoScreenSpaceTraversalWGSL, SVO_SCREEN_SPACE_TERMINATION_CONTRACT, svoLodDescentWGSL,
} from "./svo-screen-space-termination";
import {
  createSvoScenePrimitiveBandWGSL,
  packSvoScenePrimitiveBandParams,
  readSvoScenePrimitiveCoverageAudit,
  svoScenePrimitiveBandComputeBindGroupLayoutEntries,
  svoScenePrimitiveBandReadBindGroupLayoutEntries,
  svoScenePrimitiveBandReadWGSL,
  SVO_SCENE_PRIMITIVE_BAND_CONTRACT,
  SVO_SCENE_PRIMITIVE_BAND_COUNTER_BYTES,
  SVO_SCENE_PRIMITIVE_BAND_PARAMS_BYTES,
  SVO_SCENE_PRIMITIVE_BAND_STATE_BYTES,
  SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_BYTES,
  type SvoScenePrimitiveCoverageAudit,
} from "./svo-scene-primitive-band";

/**
 * Exact owner upgrades one leaf's payload walk may attempt.
 *
 * See `traceLeafPayload`. Historically this counted *voxels*, and at that
 * granularity it was the quality/cost knob of the whole promotion — measured on
 * the hero at 800x460 with cone lighting off:
 *
 *   unbounded  138.2 ms band-off / 104.7 band-on, depth vs analytic p99 7.8 mm
 *   2          75.0 ms / 55.4 ms,                 depth vs analytic p99 169 mm
 *
 * because a bounded walk abandoned cells whose exact test would have succeeded.
 * A budget that has to buy accuracy with frame time is not a shipping knob.
 *
 * It now counts *runs of consecutive cells naming one owner*, which is what
 * unhooks the two: a ray crossing an aggregate meets one owner, so one upgrade
 * covers the whole crossing rather than one per 25 mm of it. Swept on the hero
 * at 800x460 with the production cone rate, against the analytic reference over
 * the bonsai crop (serialized submit-to-fence, all arms interleaved):
 *
 *   budget    2      8     32
 *   runs   113.0  115.4  113.1 ms,  242 / 259 / 259 differing pixels
 *   cells      -   152.7  167.8 ms,        309 / 325 differing pixels
 *
 * The run curve is flat in frame time and converged in image by eight, while
 * the per-cell curve still buys its accuracy with 10 % of the frame per
 * doubling. Sixteen is the authored point: above where the image converges, so
 * the budget is a backstop rather than a quality dial, and the override exists
 * so the trade can still be swept without an edit.
 */
const SVO_DRY_VOXEL_OWNER_UPGRADES_PER_LEAF = Math.max(1, Math.min(32,
  Number((typeof process !== "undefined" ? process.env?.FLUID_SVO_VOXEL_OWNER_UPGRADES : undefined) ?? 16)));
/** Sentinel owner meaning "no run is open" in the payload walk. */
const DRY_UPGRADE_RUN_NONE = 0xffffffff;
/** The audited bound the shader is compiled against, wherever the value came from. */
function normalizeSvoVoxelOwnerUpgradeBudget(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError("Voxel owner upgrade budget must be finite");
  return Math.max(1, Math.min(32, Math.round(value)));
}
/** `{vertexCount, instanceCount, firstVertex, firstInstance}`. */
const SVO_DRAW_INDIRECT_ARGS_BYTES = 16;
/** How often the arena-pressure tripwire samples; roughly once a second at 60 Hz. */
const SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_PERIOD_FRAMES = 60;
import {
  DEFAULT_SVO_RENDER_TUNING,
  SVO_CONE_RADIANCE_RECONSTRUCTION_CODES,
  SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
  SVO_SURFACE_ANALYTIC_EIKONAL_TOLERANCE,
  SVO_SURFACE_TERRAIN_GRADIENT_CELLS,
  normalizeSvoRenderTuning,
  type SvoLodMode,
  type SvoRenderTuning,
  type SvoSurfaceReconstruction,
} from "./svo-render-tuning";

/**
 * Whether a scene-geometry arm stores the distance in metres.
 *
 * The eikonal certificate in `dryCertifiedSurfaceNormal` compares the stencil's
 * gradient magnitude against 1, which is only a statement about the field when
 * the lane's unit is the same as the lattice's. Both shipping arms store metres;
 * a band-relative arm would not, and this map is exhaustive so that adding one
 * is a decision rather than a silently disabled check.
 */
const sceneDistanceIsMetres: Readonly<Record<SparseBrickSceneGeometryFormat, boolean>> = Object.freeze({
  f32x2: true,
  "f16-unorm8": true,
});
import {
  SparseVoxelPixelTraceBuffers,
  createSvoPixelTraceProbeWGSL,
  type SvoPixelTraceProbeOptions,
} from "./webgpu-svo-pixel-trace";
import {
  mergeSvoPixelTrace,
  withSvoPixelTraceConePrepass,
  type SvoPixelTrace,
  type SvoPixelTracePrimaryMode,
} from "./svo-pixel-trace";

export interface SparseVoxelDrySceneData {
  /** Monotonic renderer publication, independent of the solver generation. */
  renderRevision: number;
  /** Packed `SvoPrimitiveRecord` values in dense live-scene owner order. */
  primitiveRecords: Uint32Array<ArrayBuffer>;
  /** Required exact secondary-ray acceleration for the same primitive records. */
  primitiveCandidates: SvoPrimitiveCandidatePublication;
  /** Complete live material table. Binding 6 is renderer-owned and capacity-stable. */
  materialRecords: Uint32Array<ArrayBuffer>;
  materialRevision: number;
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
  /**
   * Packed sculpted-terrain heightfield region: header plus row-major samples.
   * Absent means the ground is the analytic base-plus-features form, which the
   * shader evaluates in closed form from the uniform mirror as it always has.
   */
  terrainHeightfield?: Uint32Array<ArrayBuffer>;
  /**
   * Packed aggregate parameter blocks, in publication order, as
   * `packSvoDrySceneClusters` produces them. Absent means this scene publishes
   * no aggregates, and the region is uploaded as zeroes rather than left alone.
   */
  clusterBlocks?: Uint32Array<ArrayBuffer>;
  /**
   * Packed field-program tape blocks, in publication order, as
   * `packSvoDrySceneFieldPrograms` produces them. Absent means this scene
   * publishes no tapes, and the region is uploaded as zeroes rather than left
   * alone — a stale block from the previous scene would otherwise be resolved by
   * a new scene's record and grow somebody else's shape.
   */
  fieldProgramBlocks?: Uint32Array<ArrayBuffer>;
  /** Packed 80-byte finite-pane records. Empty means this scene has no glass. */
  glassRecords?: Uint32Array<ArrayBuffer>;
  /** Versioned live content key used to avoid redundant pane uploads. */
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
/**
 * Authored PBR materials one scene may publish.
 *
 * One material per record plus the 32 environment slots, so the acceptance
 * scene's 5 039 records ask for 5 071 — inside this ceiling with 38 % headroom,
 * which is why W3 raises the cluster arena beside it and leaves this alone. The
 * next rung that would cross it is roughly 16x the hero, and the authored record
 * ceiling (16 384) fires first.
 */
export const SVO_DRY_SCENE_MATERIAL_CAPACITY = 8_192;
export const SVO_DRY_SCENE_MATERIAL_ARENA_SIZE_BYTES =
  SVO_DRY_SCENE_MATERIAL_CAPACITY * SVO_MATERIAL_RECORD_STRIDE_BYTES;
export const SVO_DRY_SCENE_GLASS_ARENA_SIZE_BYTES =
  SVO_SCENE_GLASS_MAXIMUM_PANES * SVO_THIN_GLASS_RECORD_STRIDE_BYTES;
const alignDrySceneArenaBytes = (value: number): number => Math.ceil(value / 256) * 256;
/**
 * Header words in front of a sculpted terrain's samples: origin x/z, spacing and
 * slope bound as f32, then nx, nz and the lowest/highest sample.
 *
 * The region is *self-describing* on purpose. A grid's presence and extent could
 * equally have gone in `SVO_DRY_SCENE_PARAMS_LAYOUT`, but that uniform is
 * exactly full at 576 bytes, two live tests pin its shape word for word, and
 * every number the sampler wants is already being uploaded next to the samples
 * it describes. A zeroed header therefore means "analytic terrain" and the
 * shader needs nothing from anywhere else — which is also why the header is
 * reserved even when no scene has a grid: a storage read past the end of the
 * binding is not required to return zero.
 */
export const SVO_DRY_SCENE_TERRAIN_HEADER_WORDS = 8;
export const SVO_DRY_SCENE_TERRAIN_HEADER_BYTES =
  SVO_DRY_SCENE_TERRAIN_HEADER_WORDS * Uint32Array.BYTES_PER_ELEMENT;
/** Bytes a heightfield of `sampleCount` samples occupies, header included. */
export const svoDrySceneTerrainRegionBytes = (sampleCount: number): number =>
  SVO_DRY_SCENE_TERRAIN_HEADER_BYTES + sampleCount * Float32Array.BYTES_PER_ELEMENT;
/**
 * Ceiling on the region, not a reservation: `ensureSceneArenaCapacity` grows the
 * arena to whatever the publication actually carries, and this is the sanity
 * bound above it. Sized on the *derived* sample budget rather than the document
 * one because a described ground (`TerrainProcedural`) is built at the lattice
 * being drawn — 1153x769 at the first refinement rung, which the document
 * budget would have rejected as a heightfield it was never being asked to hold.
 */
export const SVO_DRY_SCENE_TERRAIN_ARENA_MAXIMUM_BYTES =
  svoDrySceneTerrainRegionBytes(MAX_TERRAIN_DERIVED_GRID_SAMPLES);
/**
 * How many aggregate clusters one scene may publish.
 *
 * A block is a 192-byte slot (the tapered sweep carries an eight-point
 * polyline), written once per publication rather than per frame.
 *
 * Raised from 1 024 with the record ceilings W0 took from 4 096 to 16 384. The
 * old value was chosen when it was "well above anything the record budget can
 * reach", and that stopped being true: `hero-garden-hose-x10` spends **948** of
 * 1 024 blocks at its acceptance multiplier, so the 10x rung only fit at all
 * because that scene is authored deliberately cluster-light — one miniature
 * bonsai per nine stands. Any species mix nearer the hero's own 41.5 % aggregate
 * share could not reach 10x, which would have made the ceiling, rather than the
 * renderer, the thing the acceptance gate measured. Four thousand and ninety-six
 * blocks is 786 KB reserved once, against a scene arena that already carries
 * three quarters of a megabyte of material and candidate space and a coverage
 * arena three orders of magnitude larger.
 */
export const SVO_DRY_SCENE_CLUSTER_CAPACITY = 4_096;
export const SVO_DRY_SCENE_CLUSTER_BLOCK_BYTES =
  SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS * Uint32Array.BYTES_PER_ELEMENT;
export const SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES =
  SVO_DRY_SCENE_CLUSTER_CAPACITY * SVO_DRY_SCENE_CLUSTER_BLOCK_BYTES;

/**
 * How many field-program tapes one scene may publish.
 *
 * Two hundred and fifty-six, and deliberately an order of magnitude under the
 * aggregate arena beside it, because the two kinds are used at opposite
 * densities. An aggregate is authored once per *record* — the hero garden
 * publishes 208 of them against 501 records — while a field program is authored
 * once per *species*: one boulder tape is instanced across every stone in a set,
 * and the whole point of the kind is that adding detail to it does not add
 * records. A scene that genuinely wanted a distinct tape per record would be
 * saying it has no shared shape language at all, which is the case this kind
 * exists to replace.
 *
 * A block is 528 bytes, so the region is 132 KB reserved once, against a scene
 * arena that already carries three quarters of a megabyte of material and
 * candidate space.
 */
export const SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY = 256;
export const SVO_DRY_SCENE_FIELD_PROGRAM_BLOCK_BYTES =
  SVO_FIELD_PROGRAM_BLOCK_WORDS * Uint32Array.BYTES_PER_ELEMENT;
export const SVO_DRY_SCENE_FIELD_PROGRAM_ARENA_SIZE_BYTES =
  SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY * SVO_DRY_SCENE_FIELD_PROGRAM_BLOCK_BYTES;

/**
 * One stable renderer-owned storage allocation for every authored scene record.
 *
 * Written as a running offset rather than as nested alignment calls: the terrain
 * region's offset used to be four `alignDrySceneArenaBytes` calls deep and the
 * total five, which meant inserting a region between two of them was an edit to
 * every line below it.
 */
const drySceneArenaRegions = (() => {
  let offsetBytes = 0;
  const region = (sizeBytes: number): number => {
    const start = offsetBytes;
    offsetBytes = alignDrySceneArenaBytes(start + sizeBytes);
    return start;
  };
  const materialOffsetBytes = region(SVO_DRY_SCENE_MATERIAL_ARENA_SIZE_BYTES);
  const primitiveOffsetBytes = region(SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES);
  const glassOffsetBytes = region(SVO_DRY_SCENE_GLASS_ARENA_SIZE_BYTES);
  const clusterOffsetBytes = region(SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES);
  const fieldProgramOffsetBytes = region(SVO_DRY_SCENE_FIELD_PROGRAM_ARENA_SIZE_BYTES);
  const terrainOffsetBytes = region(SVO_DRY_SCENE_TERRAIN_HEADER_BYTES);
  return {
    materialOffsetBytes, primitiveOffsetBytes, glassOffsetBytes, clusterOffsetBytes, fieldProgramOffsetBytes,
    terrainOffsetBytes, sizeBytes: offsetBytes,
  };
})();

export const SVO_DRY_SCENE_ARENA_LAYOUT = Object.freeze({
  materialOffsetBytes: drySceneArenaRegions.materialOffsetBytes,
  primitiveOffsetBytes: drySceneArenaRegions.primitiveOffsetBytes,
  glassOffsetBytes: drySceneArenaRegions.glassOffsetBytes,
  /** Fixed-capacity aggregate parameter blocks, addressed by a record's word-13 reference. */
  clusterOffsetBytes: drySceneArenaRegions.clusterOffsetBytes,
  /** Fixed-capacity field-program tape blocks, addressed by a record's word-13 reference. */
  fieldProgramOffsetBytes: drySceneArenaRegions.fieldProgramOffsetBytes,
  terrainOffsetBytes: drySceneArenaRegions.terrainOffsetBytes,
  /**
   * The allocation a scene with no sculpted terrain needs. Every fixed-capacity
   * region plus a zeroed heightfield header; the samples themselves are sized
   * per scene by `svoDrySceneArenaSizeBytes`, because reserving the 262 144-sample
   * ceiling would cost a megabyte that no authored scene has ever wanted and
   * would be uploaded as zeroes on every publication that has no grid at all.
   */
  sizeBytes: drySceneArenaRegions.sizeBytes,
} as const);
/** Total arena bytes that hold a terrain region of `terrainBytes`, header included. */
export const svoDrySceneArenaSizeBytes = (terrainBytes: number): number => alignDrySceneArenaBytes(
  SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes + Math.max(SVO_DRY_SCENE_TERRAIN_HEADER_BYTES, terrainBytes),
);
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
  // structure, scene-owner payload, authored scene arena, optional derived traversal.
  ...[2, 3, 4, 5].map((binding) => ({ binding, type: "read-only-storage" as const })),
  // The scene-geometry lane: signed distance at the voxel centre, plus coverage.
  //
  // The one smooth-normal source the world holds, and until this binding existed
  // the primary could not read it — it shaded from `dryVoxelFaceNormal`, one of
  // six axis directions, which is why every surface terraced at every lattice.
  // The voxeliser keeps the field's own value at the voxel centre precisely so
  // this can be differentiated (lib/webgpu-sparse-scene-proxies.ts).
  //
  // Measured against the ceiling before it was added: the worst fragment entry
  // point of the shipping raster-primary composition reaches six storage buffers
  // (three here, three in the split group), so this is the seventh against a
  // spec floor of eight. Binding 5 is filtered out in both shipping traversals,
  // which is where the headroom comes from.
  { binding: 6, type: "read-only-storage" as const },
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
  const vertexBindings = new Set([0, 1, 4, 9]);
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

/** Everything the WGSL sampler recovers from a packed heightfield region. */
export interface SvoDrySceneTerrainHeightfield {
  origin_m: { x: number; z: number };
  spacing_m: number;
  size: { nx: number; nz: number };
  minimum_m: number;
  maximum_m: number;
  slopeBound: number;
  heights_m: Float32Array;
}

/**
 * Pack a sculpted grid into the arena's terrain region.
 *
 * Memoized on the grid object because publication runs on every scene revision
 * while the vessel is baked once: the hero pond is 289 x 193 samples, and
 * copying a quarter of a megabyte per republication to produce a byte-identical
 * result is exactly the sort of cost that only shows up as jank.
 */
const terrainHeightfieldRegionCache = new WeakMap<TerrainGrid, Uint32Array<ArrayBuffer>>();
export function packSvoDrySceneTerrainHeightfield(grid: TerrainGrid | undefined): Uint32Array<ArrayBuffer> | undefined {
  if (!grid) return undefined;
  const cached = terrainHeightfieldRegionCache.get(grid);
  if (cached) return cached;
  const { nx, nz } = grid.size;
  if (!Number.isInteger(nx) || !Number.isInteger(nz) || nx < MIN_TERRAIN_GRID_SIZE || nz < MIN_TERRAIN_GRID_SIZE) {
    throw new RangeError(`Sculpted terrain region needs at least ${MIN_TERRAIN_GRID_SIZE} samples per axis`);
  }
  // The *derived* budget, not the document one. `MAX_TERRAIN_GRID_SAMPLES` is
  // what a scene may carry as JSON; a ground the document only describes is
  // built at the lattice in use (`terrainSampleGrid`) and can legitimately be
  // finer than any document would be allowed to spell out — 1153x769 at the
  // first refinement rung, which is where the coverage holes were.
  if (nx * nz > MAX_TERRAIN_DERIVED_GRID_SAMPLES) throw new RangeError(`Sculpted terrain region exceeds ${MAX_TERRAIN_DERIVED_GRID_SAMPLES} samples`);
  if (grid.heights_m.length !== nx * nz) throw new RangeError("Sculpted terrain region heights must be row-major nx * nz samples");
  if (!(grid.spacing_m > 0) || !Number.isFinite(grid.spacing_m)) throw new RangeError("Sculpted terrain region spacing must be positive and finite");
  const extent = terrainGridExtent(grid);
  const buffer = new ArrayBuffer(svoDrySceneTerrainRegionBytes(nx * nz));
  const words = new Uint32Array(buffer), floats = new Float32Array(buffer);
  floats.set([grid.origin_m.x, grid.origin_m.z, grid.spacing_m, extent.slopeBound], 0);
  words.set([nx, nz], 4);
  floats.set([extent.minimum_m, extent.maximum_m], 6);
  floats.set(grid.heights_m, SVO_DRY_SCENE_TERRAIN_HEADER_WORDS);
  terrainHeightfieldRegionCache.set(grid, words);
  return words;
}

/**
 * CPU mirror of the WGSL header decode. A zeroed header is the "no sculpted
 * terrain" encoding, so this returns undefined for it rather than throwing.
 */
export function unpackSvoDrySceneTerrainHeightfield(
  packed: Uint32Array,
): SvoDrySceneTerrainHeightfield | undefined {
  if (packed.length < SVO_DRY_SCENE_TERRAIN_HEADER_WORDS) return undefined;
  const floats = new Float32Array(packed.buffer, packed.byteOffset, packed.length);
  const nx = packed[4], nz = packed[5];
  if (nx < MIN_TERRAIN_GRID_SIZE || nz < MIN_TERRAIN_GRID_SIZE) return undefined;
  if (packed.length !== SVO_DRY_SCENE_TERRAIN_HEADER_WORDS + nx * nz) {
    throw new RangeError(`Sculpted terrain region declares ${nx}x${nz} samples but carries ${packed.length - SVO_DRY_SCENE_TERRAIN_HEADER_WORDS}`);
  }
  return {
    origin_m: { x: floats[0], z: floats[1] },
    spacing_m: floats[2],
    slopeBound: floats[3],
    size: { nx, nz },
    minimum_m: floats[6],
    maximum_m: floats[7],
    heights_m: floats.slice(SVO_DRY_SCENE_TERRAIN_HEADER_WORDS),
  };
}

/**
 * What a `terrainReference` denotes: the u32 word offset of the heightfield
 * region inside the scene arena.
 *
 * The SVO ABI has carried a `terrainHeightfield` primitive kind, an
 * `externalTerrain` flag and a `terrainReference` word since it was written,
 * with nothing to point the reference at and therefore no producer. The arena
 * region is that thing — a shader handed this number can read the header and
 * the samples with no further indirection, exactly as the sampler above does —
 * so the reference is the offset itself rather than an index into a table that
 * would have to exist first.
 */
export const SVO_DRY_SCENE_TERRAIN_REFERENCE = SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes / Uint32Array.BYTES_PER_ELEMENT;

/**
 * The scene's sculpted ground as an ordinary SVO primitive descriptor.
 *
 * This is the first producer of the kind. It does not yet reach the sparse
 * scene voxelizer — that consumer only knows the six finite proxy shapes, and
 * `sparseScenePrimitiveForSvoDescriptor` still refuses terrain outright — so
 * what this closes today is the ABI half: a heightfield can be named, packed,
 * unpacked and evaluated through the same record every other primitive uses,
 * against the same arena region the renderer already uploads.
 */
/**
 * Word offset of aggregate cluster block `index` in the scene arena.
 *
 * Same convention as the terrain reference and for the same reason: a shader
 * handed this number reads the block with no further indirection, so the
 * reference is the offset rather than an index into a table that would have to
 * exist first.
 */
/**
 * Word offsets inside a cluster's arena block.
 *
 * Declared once and read by the packer, the CPU resolver and the shader's own
 * decode below, because those are three transcriptions of the same layout and
 * the failure mode when one of them drifts is a field that renders as a
 * different field with no error anywhere. The order matches
 * `struct SvoClusterPacking` in the shared ABI so the decode is a straight run
 * down the block.
 */
const CLUSTER_BLOCK_FIELD_WORD = 0;
const CLUSTER_BLOCK_SEED_WORD = 1;
const CLUSTER_BLOCK_COUNT_WORD = 2;
const CLUSTER_BLOCK_SMOOTH_RADIUS_WORD = 3;
const CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD = 4;
const CLUSTER_BLOCK_LATTICE_PERIOD_WORD = 5;
const CLUSTER_BLOCK_JITTER_WORD = 6;
const CLUSTER_BLOCK_ANISOTROPY_WORD = 7;
const CLUSTER_BLOCK_LOBE_SPAN_WORD = 8;
const CLUSTER_BLOCK_LOBE_SPAN_SPREAD_WORD = 9;
const CLUSTER_BLOCK_DISPLACEMENT_WORD = 10;
/**
 * Eight `vec4f`, so the shader reads the polyline as vectors rather than
 * scalars. Sixteen-word aligned, which is why the three words above it stop at
 * eleven and the block leaves four unused.
 */
const CLUSTER_BLOCK_POINTS_WORD = 16;

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

export function svoDrySceneTerrainPrimitive(identity: {
  primitiveId: number;
  materialId: number;
  ownerId?: number;
  normalEpsilon_m?: number;
}): SvoTerrainHeightfieldPrimitive {
  return {
    kind: "terrain-heightfield",
    primitiveId: identity.primitiveId,
    materialId: identity.materialId,
    ...(identity.ownerId === undefined ? {} : { ownerId: identity.ownerId }),
    terrainReference: SVO_DRY_SCENE_TERRAIN_REFERENCE,
    normalEpsilon_m: identity.normalEpsilon_m ?? 0.02,
  };
}

/**
 * Resolve a packed region back into something `sampleSvoPrimitive` can query,
 * which is the CPU half of what the GPU does from the same bytes.
 *
 * The reconstruction carries no analytic fields because it cannot and must not:
 * `terrainHeightAt` short-circuits to the grid whenever one is present, so a
 * base height recovered from somewhere else could only ever disagree with the
 * samples. A reference that does not name this region resolves to nothing
 * rather than to a plausible-looking flat plane.
 */
export function svoDrySceneTerrainResolver(packed: Uint32Array | undefined): SvoTerrainResolver {
  const region = packed && unpackSvoDrySceneTerrainHeightfield(packed);
  const terrain: TerrainDescription | undefined = region && {
    baseHeight_m: 0,
    features: [],
    grid: {
      kind: "grid",
      origin_m: region.origin_m,
      spacing_m: region.spacing_m,
      size: region.size,
      heights_m: [...region.heights_m],
    },
  };
  return (terrainReference: number) => terrainReference === SVO_DRY_SCENE_TERRAIN_REFERENCE ? terrain : undefined;
}

/** Packed dry-scene parameters. */
export const SVO_DRY_SCENE_PARAMS_LAYOUT = Object.freeze({
  sizeBytes: 592,
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
   * y: `SVO_LOD_MODE_*`; z: fixed level; w: `SVO_SURFACE_RECONSTRUCTION_CODES`.
   *
   * The surface mode shares this lane rather than earning its own because it
   * wants exactly the properties the lane was built for: no pass is added or
   * removed, no march shape moves, and no cache keys on it, so the panel can
   * toggle it without dropping the world-GI cache or the stationary primary.
   */
  lodWordOffset: 144,
} as const);

/** `dry.lod.y` codes. Mirrors `SvoLodMode`; the shader constants are in `svoLodDescentWGSL`. */
export const SVO_LOD_MODES = Object.freeze({
  "screen-space": 0,
  "fixed-level": 1,
} satisfies Record<SvoLodMode, number>);

/** `dry.lod.w` codes. Mirrors `SvoSurfaceReconstruction`. */
export const SVO_SURFACE_RECONSTRUCTION_CODES = Object.freeze({
  "voxel-face": 0,
  trilinear: 1,
  analytic: 2,
} satisfies Record<SvoSurfaceReconstruction, number>);

/** materialPublication.w flags shared by the direct and derived-lighting paths. */
export const SVO_DRY_VISIBILITY_FLAGS = Object.freeze({
  exactContact: 1 << 0,
  exactShadow: 1 << 1,
  coneLightingRequested: 1 << 2,
  ambientOcclusion: 1 << 3,
  globalIllumination: 1 << 4,
  globalIlluminationOcclusion: 1 << 5,
  globalIlluminationRequested: 1 << 6,
  silhouetteRefinement: 1 << 7,
} as const);

/** How the stable node-mip address plan proves that an atlas sample is current. */
export const SVO_DRY_NODE_MIP_PUBLICATION_MODE = Object.freeze({
  unavailable: 0,
  matchingStructuralGeneration: 1,
  pageValidity: 2,
} as const);

export const SVO_TERRAIN_FAST_MIN_VERTICAL = 0.35;
export const SVO_TERRAIN_FAST_BRACKET_STEPS = 2;
export const SVO_TERRAIN_FAST_REFINEMENTS = 5;
export const SVO_TERRAIN_FALLBACK_STEPS = 20;
export const SVO_TERRAIN_FALLBACK_REFINEMENTS = 8;
/** Includes four terrain-height evaluations used by the central-difference normal. */
export const SVO_TERRAIN_FAST_MAX_HEIGHT_EVALUATIONS = 12;
/**
 * Steps a sculpted heightfield march may take before it gives up on converging.
 *
 * The bracket above cannot be reused for a sculpted grid and the reason is not
 * tuning. It samples the field at *three* points across the whole entry-to-floor
 * interval and refines the first sign change it finds, which is sound only when
 * the ground varies slowly enough that no feature fits between two of those
 * samples. The hero pond's coping is a 5.5 cm bullnose about 11 cm across on a
 * 6.25 mm lattice: over a metre-long interval it lives inside a tenth of one
 * bracket step, so the rim is not aliased, it is *invisible* — the ray samples
 * above it and below it and concludes the ground is where the basin is.
 *
 * What replaces it is a march that provably cannot skip anything. A grid carries
 * a Lipschitz bound on its own slope (`terrainGridExtent`), so the field
 * y - h(p(t)) changes by at most `|rd.y| + slopeBound * |rd.xz|` per metre of
 * ray, and advancing by the current |field| divided by that quantity can never
 * cross a root. The step is large over the flat plaster and collapses to
 * millimetres against the rim, without either being tuned: it is the Nyquist
 * limit `spacing_m` expressed as a bound on the surface rather than as a fixed
 * step, which is both cheaper and stricter than marching at `spacing_m`.
 *
 * The count is what a nearly-vertical ray never approaches and a 6-degree
 * grazing ray still exhausts. It is a budget, not a correctness parameter:
 * exhaustion falls through to bisection of the interval the march has already
 * proven the surface lies in, so the failure mode is a hit resolved late rather
 * than a hole in the ground.
 */
export const SVO_TERRAIN_GRID_MARCH_STEPS = 64;
export const SVO_TERRAIN_GRID_REFINEMENTS = 8;
/**
 * Convergence window as a fraction of the grid spacing. A quarter of a sample
 * is below anything the bake can express, so tightening it buys accuracy the
 * heightfield does not have while costing steps that are geometric in it.
 */
export const SVO_TERRAIN_GRID_EPSILON_SAMPLES = 0.25;
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
  /**
   * Which of the three solvers resolved the hit. `sculpted` is the Lipschitz
   * march a `TerrainGrid` takes; the other two are the analytic bracket and its
   * grazing-ray fallback.
   */
  solver: "fast" | "fallback" | "sculpted";
  heightEvaluations: number;
}

/**
 * Lipschitz-bounded march over a sculpted grid, and the CPU oracle for
 * `traceTerrainHeightfield` in the WGSL below.
 *
 * See `SVO_TERRAIN_GRID_MARCH_STEPS` for why a sculpted vessel cannot be traced
 * by the analytic bracket. The two properties this relies on are that the grid's
 * slope bound makes `|field| / rate` a step that provably cannot cross a root,
 * and that below the grid's lowest sample the ray is under every column the
 * heightfield can express — so the interval ends there, and ends with the field
 * non-positive. That second fact is what turns exhausting the step budget into
 * a bisection rather than a dropped pixel.
 */
function marchSvoSculptedTerrain(
  terrain: TerrainDescription,
  grid: TerrainGrid,
  origin_m: { x: number; y: number; z: number },
  rd: { x: number; y: number; z: number },
  ceiling: number,
  sceneScale_m: number,
  normalEpsilon_m: number,
): SvoTerrainRayHit | undefined {
  const extent = terrainGridExtent(grid);
  let t0 = 0.005;
  if (origin_m.y > ceiling) {
    if (rd.y >= -0.0005) return undefined;
    t0 = (ceiling - origin_m.y) / rd.y;
  }
  const span = t0 + 10 * sceneScale_m;
  let t1 = span;
  let bracketed = false;
  if (rd.y < -0.0005) {
    const lowestT = (extent.minimum_m - origin_m.y) / rd.y;
    if (lowestT > t0 && lowestT <= span) { t1 = lowestT; bracketed = true; }
  } else if (rd.y > 0.0005) {
    t1 = Math.min(t1, Math.max(t0, (ceiling - origin_m.y) / rd.y));
  }
  if (!(t1 > t0)) return undefined;

  const pointAt = (t: number) => ({ x: origin_m.x + rd.x * t, y: origin_m.y + rd.y * t, z: origin_m.z + rd.z * t });
  let heightEvaluations = 0;
  const fieldAt = (t: number) => {
    const point = pointAt(t);
    heightEvaluations += 1;
    return point.y - terrainHeightAt(terrain, point.x, point.z);
  };
  const surfaceHit = (t_m: number): SvoTerrainRayHit => {
    const position_m = pointAt(t_m);
    heightEvaluations += 4;
    return { t_m, position_m, normal: terrainNormalAt(terrain, position_m.x, position_m.z, normalEpsilon_m), solver: "sculpted", heightEvaluations };
  };

  const rate = Math.max(1e-4, Math.abs(rd.y) + extent.slopeBound * Math.hypot(rd.x, rd.z));
  const epsilon = Math.max(1e-6, SVO_TERRAIN_GRID_EPSILON_SAMPLES * grid.spacing_m);
  let t = t0, lastOutside = t0, lastOutsideField = 0;
  for (let step = 0; step < SVO_TERRAIN_GRID_MARCH_STEPS && t <= t1; step += 1) {
    const field = fieldAt(t);
    // The acceptance window is vertical, and a vertical tolerance is a long one
    // along a shallow ray: a quarter-sample of clearance is 3 cm of ray at six
    // degrees. One secant step on the two samples that closed the window costs a
    // single evaluation, removes that amplification wherever the surface is
    // locally well conditioned, and is discarded rather than trusted where it is
    // not — which is why it is guarded on the residual instead of iterated.
    if (Math.abs(field) <= epsilon) {
      if (!(t > lastOutside)) return surfaceHit(t);
      const slope = (field - lastOutsideField) / (t - lastOutside);
      if (!(Math.abs(slope) > 1e-6)) return surfaceHit(t);
      const corrected = Math.min(t1, Math.max(t0, t - field / slope));
      return Math.abs(fieldAt(corrected)) < Math.abs(field) ? surfaceHit(corrected) : surfaceHit(t);
    }
    lastOutside = t;
    lastOutsideField = field;
    t += Math.abs(field) / rate;
  }
  // Bisection is only sound where the march left the surface strictly ahead of
  // it and the interval is known to end inside solid, so a ray that started
  // under the ground reports the miss the analytic paths report as well.
  if (!bracketed || !(lastOutsideField > 0)) return undefined;
  let a = lastOutside, b = t1;
  for (let refinement = 0; refinement < SVO_TERRAIN_GRID_REFINEMENTS; refinement += 1) {
    const middle = 0.5 * (a + b);
    if (fieldAt(middle) > 0) a = middle; else b = middle;
  }
  return surfaceHit(0.5 * (a + b));
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
  const ceiling = terrainCeiling(terrain);
  // `terrainSampleGrid` rather than `terrain.grid`: a described ground is a
  // heightfield too, and this mirror must take the sculpted bracket for it or it
  // would trace the eight-feature closed form — a flat plane at `baseHeight_m`.
  const grid = terrainSampleGrid(terrain);
  if (grid) return marchSvoSculptedTerrain(terrain, grid, origin_m, rd, ceiling, sceneScale_m, normalEpsilon_m);
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
 * Structural fields an SVO-accelerated primary ray needs before it may leave the camera.
 * Primary and secondary traversal both refuse to consume the SVO until the
 * producer has published this live-scene field set. A producer that allocates
 * the structural source but never finalizes its current revision renders every
 * accelerated surface as a miss; analytic glass and rigid bodies keep drawing.
 */
export const SVO_DRY_SCENE_REQUIRED_VALID_FIELDS =
  SPARSE_VOXEL_VALID_FIELDS.topology
  | SPARSE_VOXEL_VALID_FIELDS.sceneGeometry
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
    if (kind < 1 || kind > 4 || lightId === 0 || lightIds.has(lightId) || revision !== scene.lightRevision) return false;
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
    const environmentLighting = buildSvoEnvironmentLighting(scene.environment ?? "default", revision, scene.lighting?.environment);
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
  if (scene.primitiveRecords.byteLength < SVO_PRIMITIVE_RECORD_STRIDE_BYTES) return "scene primitive arena is empty";
  if (scene.primitiveRecords.byteLength % SVO_PRIMITIVE_RECORD_STRIDE_BYTES !== 0) return "scene primitive arena stride is invalid";
  const glassBytes = scene.glassRecords?.byteLength ?? 0;
  if (glassBytes % SVO_THIN_GLASS_RECORD_STRIDE_BYTES !== 0) return "thin-glass arena stride is invalid";
  if (glassBytes / SVO_THIN_GLASS_RECORD_STRIDE_BYTES > SVO_SCENE_GLASS_MAXIMUM_PANES) return "thin-glass arena capacity is exceeded";
  if (scene.terrainMaterialMetadata !== undefined
    && scene.terrainMaterialMetadata.byteLength !== SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES) {
    return "terrain material record is invalid";
  }
  if (scene.terrainHeightfield !== undefined) {
    if (scene.terrainHeightfield.byteLength > SVO_DRY_SCENE_TERRAIN_ARENA_MAXIMUM_BYTES) return "terrain heightfield capacity is exceeded";
    try {
      if (!unpackSvoDrySceneTerrainHeightfield(scene.terrainHeightfield)) return "terrain heightfield header is invalid";
    } catch {
      return "terrain heightfield header is invalid";
    }
  }
  if (source.structural.fields.topology.residency === "unavailable") return "topology field is unavailable";
  if (source.structural.fields.sceneGeometry.residency === "unavailable") return "scene geometry field is unavailable";
  if (source.structural.fields.materialOwner.residency === "unavailable") return "material-owner payload field is unavailable";
  return undefined;
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
 * `nodeMipAtlas`, `nodeMipSampler`, `nodeMipDirectory`, `nodeMipPageTable`,
 * `nodeMipPageValidity`, and the `svoNodeMipSamplingWGSL` library.
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
  // surfaceNormal (zero only for free-space/standalone benchmark cones)
  // marks the receiver's tangent plane at the march origin: coverage whose
  // trilinear support still straddles that plane is the receiver's own
  // voxelized surface, and accumulating it self-shadows in bands that track
  // the sub-voxel phase of the analytic surface (terrain height isolines
  // rendered as concentric rings, latitude bands on mushroom caps). Each
  // sample's coverage is therefore scaled by its plane clearance over the
  // sample's own support width, ramping back to full occlusion by 24 fine
  // voxels of marched distance so genuine distant blockers keep their shadows.
  // Receiver cones (surfaceNormal set) suppress their own coverage. Shadow
  // cones additionally refine their step width geometrically as
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
  for(var rung=0u;rung<48u&&budget>0u&&emitterOffset<maximumDistance_m-phaseSplit&&transmittance>.005;rung+=1u){budget-=1u;
    let distance=maximumDistance_m-emitterOffset;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);let remaining=emitterOffset;let stepWidth=emitterOffset*.5;let position=origin_m+direction*distance;
    let lookup=dryNodeMipAt(position,lod,&pageCache);if(lookup.valid==0u){return ${coneMiss};}
    ${selfCoverageWeight}${blendWeightExpression}${coarseBlendedCoverage}${blendedCoverage}emitterOffset*=1.5;}}
`;
  // The elision variant keeps the identical march (step distances, stepIndex
  // sequence, termination) and only replaces the
  // fine-level fetch whose trilinear support is provably inside a zero region
  // with the arithmetically identical zero sample: max(0,0*.15)=0 contributes
  // nothing to the blend.
  const visibility = options.emptySpaceElision
    ? /* wgsl */ `fn dryConeVisibility(origin_m:vec3f,direction:vec3f,aperture:f32,maximumDistance_m:f32,surfaceNormal:vec3f,anchored:bool)->DryConeVisibility{
  if(!dryNodeMipReady()){return ${coneMiss};}let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let tangent=tan(aperture*.5);var distance=minimumVoxel*.75;var transmittance=1.0;${fluidPrologue}var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);var pageCacheCoarse=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);let shadowCone=dot(surfaceNormal,surfaceNormal)>.25;var budget=clamp(dry.tuningCounts0.y,1u,48u);let phaseSplit=select(maximumDistance_m,maximumDistance_m*.5,anchored);
  var zeroRegion=DryConeZeroRegion(vec3f(0.0),vec3f(0.0),0u);
  for(var stepIndex=0u;stepIndex<48u&&budget>0u&&distance<phaseSplit&&transmittance>.005;stepIndex+=1u){budget-=1u;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);${stepWidthExpression}
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
  for(var stepIndex=0u;stepIndex<48u&&budget>0u&&distance<phaseSplit&&transmittance>.005;stepIndex+=1u){budget-=1u;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);${stepWidthExpression}let position=origin_m+direction*distance;let lookup=dryNodeMipAt(position,lod,&pageCache);if(lookup.valid==0u){return ${coneMiss};}${selfCoverageWeight}${blendWeightExpression}${coarseBlendedCoverage}${blendedCoverage}distance+=max(stepWidth,minimumVoxel*.25);}${emitterLadderWGSL}
  return ${coneResult};
}`;
  return /* wgsl */ `${liveSvoDerivedPageValidityWGSL}
struct DryNodeMipLookup{sample:SvoNodeMipSample,valid:u32}
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
fn dryNodeMipReady()->bool{let generationReady=dry.nodeMip.w==${SVO_DRY_NODE_MIP_PUBLICATION_MODE.pageValidity}u||dry.nodeMip.x==dryPublicationWord(2u);return dry.nodeMip.w!=0u&&dry.nodeMip.x!=0u&&generationReady&&dry.nodeMip.y>0u&&dry.nodeMip.z>0u;}
fn dryNodeMipPageValid(pageIndex:u32)->bool{
  let dimensions=textureDimensions(nodeMipPageValidity);return svoDerivedPageValidityResident(dimensions,pageIndex)&&textureLoad(nodeMipPageValidity,svoDerivedPageValidityTexel(dimensions,pageIndex),0).x!=0u;
}
// Finest level that owns an opacity page (SVO_OPACITY_LEVEL_FLOOR). The base
// level is 81 % of the pyramid's pages at refinement depth 3 and is one texel
// per finest voxel, so a world fine enough to need it stores a resolution no
// cone ever asks for. Zero is the pyramid that shipped, and there this clamp is
// the identity — which is why a reference-leaf frame is bit-exact across it.
fn dryNodeMipOpacityLevelFloor()->u32{return min(dry.nodeMipAtlas.w,dry.nodeMip.z-1u);}
fn dryNodeMipAt(position_m:vec3f,lodIn:f32,pageCache:ptr<function,DryNodeMipPageCache>)->DryNodeMipLookup{
  let level=min(max(u32(max(floor(lodIn),0.0)),dryNodeMipOpacityLevelFloor()),dry.nodeMip.z-1u);let levelScale=exp2(f32(level));let virtualVoxel=(position_m-dry.nodeMipOrigin.xyz)/(dry.mapping.cellSize*levelScale);let pageFloor=floor(virtualVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);}let pageCoordinate=vec3u(pageFloor);
  if((*pageCache).generation!=dry.nodeMip.x||(*pageCache).level!=level||any((*pageCache).coordinate!=pageCoordinate)){
    *pageCache=DryNodeMipPageCache(pageCoordinate,level,vec3u(0u),dry.nodeMip.x,0u,0xffffffffu,0u);let pageIndex=dryNodeMipFind(level,pageCoordinate);
    if(pageIndex!=0xffffffffu){${pageOrigin}}
  }
  if((*pageCache).resident==0u){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);}
  if(!dryNodeMipPageValid((*pageCache).pageIndex)){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),0u);}
  let local=virtualVoxel-vec3f(pageCoordinate)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);return DryNodeMipLookup(svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,(*pageCache).pageOrigin,local),1u);
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

export type SvoDryPresentationBundleStatus =
  | { readonly state: "ready" }
  | { readonly state: "compiling"; readonly detail: string }
  | { readonly state: "failed"; readonly detail: string };

/** Compile-time static traversal experiment. Hybrid preserves the shipping fallback semantics. */
/**
 * `raster-primary` keeps canonical-parametric traversal for every secondary
 * ray and replaces only the full-screen primary megakernel with a hardware
 * rasterization of resident brick proxies plus a bounded in-brick DDA
 * (docs/SVO_RASTER_PRIMARY_HANDOFF.md).
 */
export type SvoDryTraversalMode = "hybrid" | "canonical" | "canonical-parametric" | "compact" | "wide" | "raster-primary";

export const SVO_DRY_TRAVERSAL_MODES: readonly SvoDryTraversalMode[] = Object.freeze([
  "hybrid", "canonical", "canonical-parametric", "compact", "wide", "raster-primary",
]);

/** Compile-time 8^3 leaf acceleration experiment; off preserves the baseline shader. */
export type SvoBrickOccupancyMode = "off" | "bounds" | "macro" | "macro-hdda";

/** Explicit shading topology: production never changes topology from tuning state. */
export type SvoDryShadingPath = "inline" | "split";

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
  /** Persistent level-0 voxel visibility for directional light slot zero. */
  readonly voxelLightCache?: boolean;
  /** Bounded exact-identity receiver search for sub-prepass-pixel surfaces. */
  readonly edgeReceiverRecovery?: boolean;
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
  /**
   * Retain the original one-DDA-per-proxy brick fragment as the exact control
   * for the conservative coverage/resolve production arm.
   */
  readonly rasterPrimaryDirect?: boolean;
  /**
   * Retain the historical no-op leaf payload: voxel owners accelerate nothing
   * and every authored surface comes from the analytic passes.
   *
   * The control for the voxel-resolved primary, which is otherwise the default.
   * Keeping it selectable is what makes "the voxel is the accelerator" a
   * measurable claim rather than an assertion.
   */
  readonly voxelOwnerPrimaryDisabled?: boolean;
  /**
   * Retain the one-fragment-per-covering-proxy scene-primitive raster as the
   * exact control for its conservative coverage/resolve arm.
   *
   * Deliberately separate from `rasterPrimaryDirect`, which switches the *brick*
   * arm at the same time: an image comparison that moves both cannot say which
   * one moved the pixels.
   */
  readonly scenePrimitiveDirect?: boolean;
  /**
   * Drop only the brick raster's frag_depth write, leaving the empty-brick
   * discard in place. Brick proxies are disjoint along any ray and a brick
   * with no hit still discards, so the interpolated proxy exit depth picks the
   * same winner as the hit depth does; only the depth buffer's stored value
   * changes. That makes this arm the one with a sound redesign behind it, so
   * its delta is the recoverable part of {@link rasterPrimaryHsrProbe}.
   */
  readonly rasterPrimaryNoFragmentDepth?: boolean;
  /**
   * Upper-bound probe for tile-based hidden-surface removal in the brick
   * raster. Writing frag_depth and calling discard both leave a fragment's
   * depth and coverage unknown until it has been shaded, so either one alone
   * forces this GPU to shade every overlapping brick proxy rather than keeping
   * only the winner. The arm drops both, which is the most HSR the hardware
   * could ever recover; the delta against the default bounds what any sound
   * redesign is worth. The image is wrong here — missed bricks still shade —
   * so this is a timing probe only, never a rendering mode.
   */
  readonly rasterPrimaryHsrProbe?: boolean;
  /**
   * The same upper-bound probe, for the *scene primitive* raster rather than
   * the brick raster.
   *
   * {@link rasterPrimaryHsrProbe} bounds hidden-surface removal on proxies that
   * are subsets of disjoint SVO leaf cells, where overlap along a ray is rare.
   * Authored primitive proxies are not disjoint and overlap heavily — on the
   * hero pond a pixel is covered by roughly two dozen of them, and 84 % of the
   * scene's kinds resolve their hit by a 48-iteration sphere trace rather than
   * a closed-form root. Writing frag_depth and calling discard both leave a
   * fragment's depth and coverage unknown until it has been shaded, so this
   * pass currently marches every covering proxy instead of keeping the winner.
   *
   * The arm drops both. The image is wrong here — occluded primitives still
   * shade and the stored depth is the proxy's rather than the surface's — so
   * this is a timing probe only, never a rendering mode. Its delta bounds what
   * a sound front-to-back resolve can recover.
   */
  readonly scenePrimitiveHsrProbe?: boolean;
  /**
   * Retain the unbounded `0 .. DRY_MISS` march interval in the scene-primitive
   * raster as the exact control for the box-bounded one. See
   * `dryScenePrimitiveMarchSpan` for why the bounded interval is the contract
   * the marched kinds were written against.
   */
  readonly scenePrimitiveUnboundedMarch?: boolean;
  /**
   * Override the compiled owner-run upgrade budget of `traceLeafPayload`.
   *
   * Present so the frame/accuracy curve over the budget can be swept with both
   * arms alive in one process and interleaved, which a module-load environment
   * override cannot do. Omitted means the authored/environment value.
   */
  readonly voxelOwnerUpgradesPerLeaf?: number;
  /**
   * Retain the per-voxel exact upgrade of `traceLeafPayload` instead of the
   * per-owner-run one. The control for the run merge, and the shape W3 shipped:
   * one exact march per solid cell, each over its own padded cell interval.
   */
  readonly voxelOwnerPerCellUpgrades?: boolean;
  /**
   * Retain the analytic surface in the primary: the per-owner exact upgrade
   * inside `traceLeafPayload`, and the `traceScenePrimitives` tier that
   * `traceStatic` runs bounded by the voxel hit.
   *
   * **Off is the shipping path and is not a tuning choice.** The renderer shades
   * from voxels only; an authored SDF exists to be voxelized, and nothing
   * analytic reaches the screen. What that costs is exact: a hit is a cell face
   * at `HERO_GARDEN_CELL_M`, so silhouettes are blocky and the depth error
   * against the analytic surface is the half-cell the census above measures
   * rather than the 7.8 mm p99 the upgrade bought. That is the pipeline being
   * honest about its resolution, and the answer to it is smaller leaves.
   *
   * This arm survives only as the measurement control — the same role every
   * other `*Direct`/`*Disabled` flag here plays. It is the reference an
   * "is the voxel path converging?" comparison needs, and keeping it selectable
   * is what makes that a measurable claim rather than an assertion. It is not a
   * fallback and no shipping configuration may turn it on.
   */
  readonly analyticPrimaryRetained?: boolean;
  /**
   * Shade every opaque surface as neutral white — a clay render.
   *
   * Not a shortcut and not a stand-in for unauthored materials: while geometry
   * and lattice resolution are the work, albedo is the one channel that
   * *competes* with the thing being judged. A colour difference and a form
   * difference land in the same pixel, and the eye attributes both to whichever
   * it noticed first. Neutralising albedo leaves shape, shadow, occlusion and
   * silhouette carrying the whole image, which is what the reference plate is
   * being compared on right now.
   *
   * Material *resolution* is untouched — the id lookup, the publication check
   * and `dryInvalidSurfaceMaterial` all still run, and an unpublished or
   * malformed record still reports as invalid. Only the appearance is replaced.
   * Returning white before those checks would mean a scene whose material arena
   * never published rendered perfectly, which is precisely the class of failure
   * that hid the voxel path behind analytic shading for this whole program.
   *
   * Solidity is unaffected: a cell is solid when `identity & 0xffff` is nonzero,
   * so the material channel is still load-bearing occupancy data no matter what
   * is done with its colour.
   */
  readonly neutralSurfaceAlbedo?: boolean;
  /**
   * Re-derive an incoherent cone-prepass receiver by tracing the analytic scene
   * again, instead of taking it from the full-resolution primary G-buffer.
   *
   * The control for the G-buffer receiver, which is otherwise the default. See
   * `dryPrepassCoherentMain`: the analytic arm is `O(authored records)` per
   * boundary texel and is what made the cone prepass the one lighting stage that
   * scaled with scene size.
   */
  readonly analyticConeBoundaries?: boolean;
  /**
   * Run the analytic candidate-BVH walk of `traceStatic` first and unbounded,
   * as it historically did, instead of seeding it with the voxel-resolved hit.
   *
   * The control for the bounded order. Both produce the same surface; the
   * unbounded one descends every candidate box the ray meets anywhere along its
   * length, which is where the primary's record-count scaling lived.
   */
  readonly unboundedAnalyticPrimary?: boolean;
  /**
   * Retain the pre-bounded visibility order: the candidate-BVH walk runs first
   * for every shadow/AO/GI ray, bounded only by the ray's own tMax.
   *
   * The control for {@link unboundedAnalyticPrimary}'s counterpart on the
   * lighting path. The primary shed this term by seeding the analytic walk with
   * the voxel hit; visibility carried it unchanged, which is why it was the
   * largest remaining record-count residual after the primary was fixed.
   */
  readonly unboundedAnalyticVisibility?: boolean;
  /**
   * Skip the visibility candidate-BVH walk entirely.
   *
   * Deliberately image-wrong and never a shipping arm: authored records that
   * are not voxel-resolved stop casting shadows. It exists to bound the prize —
   * the reordering can recover at most what deleting the term recovers — so a
   * disappointing reorder can be told apart from a term that was never
   * expensive in the first place.
   */
  readonly visibilityAnalyticDisabledProbe?: boolean;
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
  /** Full-resolution visibility written only for compacted silhouette pixels. */
  silhouetteRefinementFormat: "rg32uint" as GPUTextureFormat,
  /** Core-WebGPU storage-capable state plane: 0 untouched, 1 valid, 2 failed. */
  silhouetteRefinementStateFormat: "r32uint" as GPUTextureFormat,
  /** Every user-shadable light slot is cached by the reduced-rate prepass. */
  maximumPrepassLights: SVO_DRY_SCENE_MAX_SHADED_LIGHTS,
  /** Guided-upsample weight below this threshold publishes an explicit reconstruction failure. */
  minimumReconstructionWeight: 0.05,
} as const);

/** Indirect worklist/readback ABI for full-resolution silhouette refinement. */
export const SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT = Object.freeze({
  queueHeaderWords: 4,
  queuedWord: 0,
  exhaustedWord: 1,
  exactInvalidWord: 2,
  failureTotalWord: 3,
  /** Compatibility alias: the two-counter seam reports all failed refinements. */
  invalidWord: 3,
  indirectOffsetBytes: 0,
  counterWords: 2,
  counterSizeBytes: 2 * Uint32Array.BYTES_PER_ELEMENT,
  diagnosticCounterWords: 4,
  diagnosticCounterSizeBytes: 4 * Uint32Array.BYTES_PER_ELEMENT,
  workgroupSize: 64,
} as const);

/** `[exhausted, publication, sceneLimit, traversal, traceContract]` diagnostics. */
export const SVO_DRY_SILHOUETTE_FAILURE_REASON_CONTRACT = Object.freeze({
  exhaustedWord: 0,
  publicationWord: 1,
  sceneLimitWord: 2,
  traversalWord: 3,
  traceContractWord: 4,
  words: 5,
  sizeBytes: 5 * Uint32Array.BYTES_PER_ELEMENT,
} as const);

/** GPU-visible reasons a requested live-derived lighting sample failed closed. */
export const SVO_DRY_DERIVED_FAILURE = Object.freeze({
  ambientOcclusionPage: 1 << 0,
  directVisibilityPage: 1 << 1,
  globalIlluminationPage: 1 << 2,
  reducedReconstruction: 1 << 3,
} as const);

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

/** Persistent camera-independent cache used by the reduced split GI pass. */
export const SVO_DRY_WORLD_GI_CACHE_CONTRACT = Object.freeze({
  entryCount: 1 << 18,
  entryBytes: 16,
  probeBytes: 8,
  payloadBytes: 8,
  probeCount: 4,
  allocatedBytes: (1 << 18) * 16,
  frameBytes: 144,
  dynamicInfluenceCells: 12,
  dynamicInfluenceBodyRadii: 3,
} as const);

/** Phase-1 static directional-light cache. One bounded queue drains cold demand over several frames. */
export const SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT = Object.freeze({
  format: "rg32uint" as GPUTextureFormat,
  populationBudget: 16_384,
  queueHeaderWords: 8,
  queueEntryWords: 2,
  voxelsPerPage: SVO_NODE_MIP_LAYOUT.interiorSize ** 3,
  requestWordsPerPage: SVO_NODE_MIP_LAYOUT.interiorSize ** 3 / 32,
  bytesPerTexel: 8,
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

/**
 * Constants the pixel-trace probe mirrors, passed instead of imported so the
 * probe module stays free of this one.
 *
 * `primaryMode` is not a constant: it decides whether the probe instruments a
 * hierarchy walk at all, and instrumenting one the frame did not perform is the
 * defect this argument exists to prevent.
 */
export function svoDryScenePixelProbeOptions(
  primaryMode: SvoPixelTracePrimaryMode = "traced",
): SvoPixelTraceProbeOptions {
  return {
    primaryMode,
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
  if (!SVO_DRY_TRAVERSAL_MODES.includes(traversalMode)) {
    throw new RangeError(`Unsupported dry-scene traversal mode: ${traversalMode}`);
  }
  // The mode only replaces the split path's primary entry. Inline variants of
  // the same module — the reduced cone prepass, the diagnostic megakernel —
  // legitimately keep traversing, so they simply omit the raster entries.
  const rasterPrimary = traversalMode === "raster-primary" && shadingPath === "split";
  // The point of the mode is to unfuse the megakernel: panes reach the brick
  // fragment only as an already-rasterized key, never as a loop. Bodies are
  // likewise a renderer-level requirement, checked where the passes are wired.
  if (rasterPrimary && !rasterGlassDiscovery) {
    throw new RangeError("Raster-primary traversal requires raster glass discovery");
  }
  if (brickOccupancyMode !== "off" && brickOccupancyMode !== "bounds"
    && brickOccupancyMode !== "macro" && brickOccupancyMode !== "macro-hdda") {
    throw new RangeError(`Unsupported dry-scene brick occupancy mode: ${brickOccupancyMode}`);
  }
  if (shadingPath !== "inline" && shadingPath !== "split") {
    throw new RangeError(`Unsupported dry-scene shading path: ${shadingPath}`);
  }
  if (!Number.isFinite(screenSpaceTerminationPixels) || screenSpaceTerminationPixels < 0) {
    throw new RangeError("Dry-scene screen-space termination must be a non-negative finite pixel count");
  }
  const screenSpaceTerminationSupported = (traversalMode === "canonical" && shadingPath === "inline")
    || (traversalMode === "raster-primary" && shadingPath === "split");
  if (screenSpaceTerminationPixels > 0 && !screenSpaceTerminationSupported) {
    throw new RangeError("Screen-space termination requires canonical inline or raster-primary split traversal");
  }
  const reduced = coneLightingScale !== 1;
  const split = shadingPath === "split";
  const splitGroup = reduced ? 2 : 1;
  // Compile-time because the codec is: producer and consumer must agree about a
  // shift, and the selector is the same pure function of the environment that
  // the voxeliser resolves its own arm from. A mismatch would render a wrong
  // scene rather than fail, which is exactly what emitting the shared codec —
  // instead of hand-matching the bits here — is for.
  const sceneGeometryFormat = octreeLiveSceneSceneGeometryFormat();
  const voxelLightCache = split && experiments.voxelLightCache !== false;
  const edgeReceiverRecovery = reduced && experiments.edgeReceiverRecovery !== false;
  // Full-rate/inline shaders still own the diagnostic overlay. The reduced
  // split shaders are never selected while that overlay is active, so their
  // counters are dead production state and can be removed safely.
  if (experiments.halfPrecisionLighting && !reduced) {
    throw new RangeError("Half-precision lighting is restricted to reduced-rate shaders");
  }
  // What a prepass texel whose 2x2 full-resolution neighbourhood disagrees does.
  //
  // Both analytic arms re-derive the receiver by tracing the scene again from
  // the prepass ray, and that trace walks the candidate BVH with an exact
  // primitive intersection per leaf — `O(authored records)` per boundary texel,
  // on roughly half of them. Measured on the hero: 43 137 of 92 000 texels at
  // 501 records and 52 497 at 5 039, with the cost *per* re-trace growing 4.05x
  // for 10.06x records. The queue depth barely moves; the walk is the scaling.
  //
  // The default takes the receiver from the full-resolution primary G-buffer
  // instead, which already holds the exact primary hit for every one of those
  // four samples — the same plane `drySilhouetteRefineMain` consumes rather than
  // retraces. It is a quality decision and not a free one: no full-resolution
  // pixel centre coincides with the prepass ray, which is why the homogeneity
  // test exists at all, so the nearest of the four is chosen rather than a
  // fixed corner. At a silhouette that is the foreground surface, which is the
  // receiver a cone from this texel would actually have found, and it is the
  // one the reduced-rate reconstruction can recover the other side of.
  //
  // Measured on the hero at 800x460, cone scale 0.5, all arms interleaved in one
  // process (serialized submit-to-fence): 292.5 -> 227.4 ms at 501 records and
  // 1564.4 -> 1190.7 ms at 5 039, with the boundary queue reading zero and the
  // depth plane unchanged against the analytic reference at both rungs.
  const inlineBoundaryWGSL = experiments.inlineConeBoundaries
    ? "let opaque=traceOpaqueScene(ray[0],ray[1]);dryPrepassStore(coordinate,opaque,ray[0],ray[1]);return;"
    : experiments.analyticConeBoundaries
      ? "let queueIndex=atomicAdd(&dryPrepassBoundaryQueue.count,1u);dryPrepassBoundaryQueue.coordinates[queueIndex]=globalId.y*dimensions.x+globalId.x;return;"
      : "var nearest=3u;var nearestDepth=DRY_MISS;for(var sample=0u;sample<4u;sample+=1u){let candidate=primaryGeometry[sample].w;if(candidate<nearestDepth){nearestDepth=candidate;nearest=sample;}}referenceGeometry=primaryGeometry[nearest];referenceIdentity=primaryIdentity[nearest];";
  // Secondary rays keep the measured production traversal; only the primary
  // changes shape in raster-primary mode.
  //
  // Keyed on the traversal mode alone, not on whether *this* composition emits
  // the raster entries. The inline compositions of this module — the diagnostic
  // megakernel, the pixel probe — do not rasterize, but they run beside a frame
  // that does, and their secondaries must be the traversal that frame's
  // secondaries use. Gating this on the split path instead left "raster-primary"
  // matching neither the canonical nor the compact branch below, so those
  // compositions silently fell through to the wide-fanout cursor.
  const secondaryTraversalMode = traversalMode === "raster-primary" ? "canonical-parametric" : traversalMode;
  const hsrProbe = experiments.rasterPrimaryHsrProbe === true;
  const scenePrimitiveHsrProbe = experiments.scenePrimitiveHsrProbe === true;
  const voxelOwnerPrimary = experiments.voxelOwnerPrimaryDisabled !== true;
  // Voxels only. See `analyticPrimaryRetained` for what the control arm is for
  // and why no shipping configuration turns it on.
  const voxelsOnlyPrimary = experiments.analyticPrimaryRetained !== true;
  const neutralSurfaceAlbedo = experiments.neutralSurfaceAlbedo === true;
  const analyticPrimaryUnbounded = experiments.unboundedAnalyticPrimary === true;
  const analyticVisibilityUnbounded = experiments.unboundedAnalyticVisibility === true;
  const analyticVisibilityProbeOff = experiments.visibilityAnalyticDisabledProbe === true;
  // The bounded order has to reach the analytic walk *after* the octree loop, so
  // every early exit inside that loop becomes a flagged break instead. The
  // control arm keeps the returns exactly where they were.
  const visibilityExhaustedWGSL = analyticVisibilityUnbounded
    ? "return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);"
    : "walkExhausted=true;break;";
  const visibilityPayloadHitWGSL = analyticVisibilityUnbounded
    ? "return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,payload.t_m);"
    : "voxelT=payload.t_m;break;";
  // Owners along the ray, not voxels along the ray. See `traceLeafPayload`.
  const primaryUpgradeBudget = normalizeSvoVoxelOwnerUpgradeBudget(
    experiments.voxelOwnerUpgradesPerLeaf ?? SVO_DRY_VOXEL_OWNER_UPGRADES_PER_LEAF);
  // One flush of the pending same-owner run. Emitted at every place the run can
  // end — an owner change, a macro-occupancy skip over empty space, and the exit
  // of the DDA — so that the run is always closed by whatever ended it.
  // Voxels only closes the run without ever solving it: the walk has already
  // returned the first solid cell's face, so a run can only still be open when
  // the DDA left the brick without meeting one, and there is nothing to flush.
  const primaryUpgradeFlushWGSL = voxelsOnlyPrimary
    ? /* wgsl */ `runOwner=${DRY_UPGRADE_RUN_NONE}u;`
    : /* wgsl */ `if(runOwner!=${DRY_UPGRADE_RUN_NONE}u&&upgrades<${primaryUpgradeBudget}u){upgrades+=1u;let candidate=primitiveHit(dryPrimitive(runOwner),ro,rd,max(0.0,runStart-tolerance),runEnd+tolerance);if(candidate.t<DRY_MISS){return candidate;}}runOwner=${DRY_UPGRADE_RUN_NONE}u;`;
  // The first solid cell along the DDA is the surface. `entry` is the ray's
  // distance to the face it crossed to reach this cell, so the hit sits exactly
  // on the cell boundary and its normal is that face — read back off the cell
  // bounds, where the entry point's zero-distance axis is the one it came in
  // through. No march, no owner solve, no upgrade budget.
  //
  // What the *shading* normal is then allowed to be is a separate question, and
  // the uniform answers it. `t`, hit-versus-miss and the cell face are identical
  // under every arm: the reconstruction replaces the shaded direction only, which
  // is what keeps `voxel-face` a bit-exact reference rather than an approximate
  // one, and what keeps this off the depth-test and coverage paths entirely.
  //
  // `analytic` ships. It asks the owner, and the owner is already in hand: the
  // walk above resolved `cellOwner` for the upgrade path before this line ran,
  // so the exact normal costs one primitive-record fetch and no payload taps at
  // all — against the stencil's eight, up to eight of which pay a root-to-leaf
  // descent when the dual cell straddles a brick face. The cheaper arm is also
  // the exact one, which is unusual enough to say out loud.
  //
  // A cell whose owner names nothing analytic — the ground, and any scenery the
  // voxeliser filled without an in-range owner — falls through to the stencil,
  // and the ground then takes `terrainNormalAt`, which differentiates the same
  // heightfield the writer sampled instead of the vertical pseudo-distance it
  // stored. The remaining fall-through is the stencil, now required to certify
  // itself: see `dryReconstructedSurfaceNormal`.
  const primaryVoxelSurfaceWGSL = voxelsOnlyPrimary
    ? /* wgsl */ `if(cellSolid){
      let cellBounds=mat2x3f(bounds[0]+vec3f(cell)*extent,bounds[0]+(vec3f(cell)+vec3f(1.0))*extent);
      let surfacePoint=ro+rd*entry;
      let faceNormal=dryVoxelFaceNormal(cellBounds,surfacePoint);
      let shaded=dryShadingNormal(hit,bounds,extent,surfacePoint,faceNormal,cellOwner,cellIdentity);
      return DryHit(entry,shaded.normal,cellIdentity&0xffffu,cellIdentity>>16u,
        shaded.featureId,DRY_GBUFFER_FIELD_VOXEL,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
    }`
    : "";
  // The control arm never extends an open run past its first cell, so every
  // owned cell flushes on its own interval exactly as the per-cell walk did —
  // same intervals, same order, same budget accounting, one iteration later.
  const primaryUpgradeRunMergeWGSL = experiments.voxelOwnerPerCellUpgrades === true
    ? `cellOwner==${DRY_UPGRADE_RUN_NONE}u&&runOwner==${DRY_UPGRADE_RUN_NONE}u`
    : "cellOwner==runOwner";
  // The exact hit is unchanged either way — this only decides how much empty
  // interval the sphere trace is asked to walk before it reaches the solid.
  const scenePrimitiveMarchSpanWGSL = experiments.scenePrimitiveUnboundedMarch === true
    ? "let exact=primitiveHit(record,ro,rd,0.0,DRY_MISS);"
    : "let span=dryScenePrimitiveMarchSpan(record,ro,rd);let exact=primitiveHit(record,ro,rd,span.x,span.y);";
  if (scenePrimitiveHsrProbe && traversalMode !== "raster-primary") {
    throw new RangeError("The scene primitive raster depth experiment only applies to raster-primary traversal");
  }
  const noFragmentDepth = hsrProbe || experiments.rasterPrimaryNoFragmentDepth === true;
  const directRasterPrimary = experiments.rasterPrimaryDirect === true || noFragmentDepth;
  if (directRasterPrimary && traversalMode !== "raster-primary") {
    throw new RangeError("The brick raster depth experiments only apply to raster-primary traversal");
  }
  const canonicalTraversal = secondaryTraversalMode === "canonical" || secondaryTraversalMode === "canonical-parametric";
  const wideTraversalWGSL = canonicalTraversal || secondaryTraversalMode === "compact"
    ? ""
    : createWebgpuSvoWideFanoutTraversalWGSL({ arena: {
      binding: 5,
      pageOffset: "dry.derivedTraversal.x",
      descriptorOffset: "dry.derivedTraversal.y",
    } });
  const compactTraversalWGSL = secondaryTraversalMode === "compact" ? createWebgpuSvoCompactTraversalWGSL(5) : "";
  const compactTraversal = secondaryTraversalMode === "compact";
  const canonicalTraversalWGSL = createWebgpuSvoTraversalWGSL({ arena: {
      binding: 2,
      controlOffset: "dry.structureOffsets.x",
      nodeOffset: "dry.structureOffsets.z",
      leafOffset: "dry.structureOffsets.w",
    },
    childEnumeration: secondaryTraversalMode === "canonical-parametric" ? "parametric" : "aabb",
    stackCapacity: experiments.tinyTraversalStack ? 8 : experiments.shortTraversalStack ? 16 : 32 });
  const screenSpaceTraversalWGSL = screenSpaceTerminationPixels > 0
    ? createSvoScreenSpaceTraversalWGSL(canonicalTraversalWGSL) : "";
  const lodDescentWGSL = screenSpaceTerminationPixels > 0 ? svoLodDescentWGSL : "";
  const leafAccessWGSL = compactTraversal ? /* wgsl */ `
fn dryLeafBounds(nodeIndex:u32)->mat2x3f{return svoCompactNodeBounds(svoCompactNodes[nodeIndex],dry.mapping);}
fn dryLeafFlags(nodeIndex:u32)->u32{return svoNodeLoad(nodeIndex).links.w;}
fn dryLeafLevel(nodeIndex:u32)->u32{return svoNodeLoad(nodeIndex).address.z;}
` : /* wgsl */ `
fn dryLeafBounds(nodeIndex:u32)->mat2x3f{return svoNodeBounds(svoNodeLoad(nodeIndex),dry.mapping);}
fn dryLeafFlags(nodeIndex:u32)->u32{return svoNodeLoad(nodeIndex).links.w;}
fn dryLeafLevel(nodeIndex:u32)->u32{return svoNodeLoad(nodeIndex).address.z;}
`;
  const liveLeafLifecycleWGSL = /* wgsl */ `
fn dryLeafCurrent(hit:SvoTraversalHit)->bool{return svoBrickLifecycleCurrent(svoBrickLifecycleDecode(dryLeafFlags(hit.nodeIndex)));}
`;
  // The scene-geometry lane, decoded by the producer's own emitted codec rather
  // than by a hand-matched shift here. `f32x2` emits no codec and addresses the
  // lane directly, exactly as the derived builder's `f32x2` arm does.
  const sceneDistanceLoadWGSL = sceneGeometryFormat === "f32x2"
    ? /* wgsl */ `
fn drySceneDistanceWord(voxel:u32)->u32{return voxel*2u;}
fn drySceneDistanceDecode(word:u32,voxel:u32)->f32{return bitcast<f32>(sceneGeometry[word]);}`
    : /* wgsl */ `
fn drySceneDistanceWord(voxel:u32)->u32{return sceneGeometryWord(0u,voxel);}
fn drySceneDistanceDecode(word:u32,voxel:u32)->f32{return sceneDistanceOf(sceneGeometry[word],voxel);}`;
  // The smooth surface, reconstructed from the field the voxeliser already stored.
  //
  // The primary's normal used to be `dryVoxelFaceNormal` — one of six axis
  // directions — and that six-way quantisation is what terraced every surface at
  // every lattice and banded the interior of flat faces. The fix is not more
  // voxels: it is to stop throwing away the signed distance the voxeliser writes
  // at each voxel centre, in metres, and which it deliberately keeps as the
  // field's own value there rather than the nearer corner sample it uses for
  // coverage (lib/webgpu-sparse-scene-proxies.ts).
  //
  // Trilinear rather than a contour plane, and that ordering is specific to this
  // codebase rather than the usual advice. A normal-oriented slab needs a normal
  // it does not have — nothing per-voxel is stored but the packed identity word,
  // which is full — so it would have to central-difference the same lane anyway,
  // at six taps plus a centre, and then intersect a plane that is discontinuous
  // across cell boundaries. Eight taps of the same lane buy a watertight C0
  // surface whose gradient is analytic from the values already fetched. One extra
  // fetch for continuity and no cracks.
  //
  // Once per ray, never per DDA cell: this runs only where the walk has already
  // stopped on a solid cell, so the loop keeps its single payload fetch per step.
  //
  // The stencil crosses leaves, and that is the whole difference between a
  // smooth surface and a banded one.
  //
  // Clamping the eight taps inside the hit leaf — which is what `safeNormal`
  // does, and what this did first — collapses a difference to zero whenever the
  // dual cell straddles a brick face. Measured over a sub-voxel walk across the
  // pond bowl, that clamp produced 372 normal creases above one degree with a
  // mean of 17.31 degrees and full 180-degree flips where two axes collapsed at
  // once; resolving the neighbour leaf instead leaves 133 creases averaging
  // 3.01 degrees, which is the inherent C0-not-C1 property of trilinear at a
  // dual-cell boundary and not an artifact.
  //
  // Both of the artifacts this removes are the same bug seen on different axes,
  // which is why they looked like two problems. The x and z brick faces draw a
  // rectangular lattice over the surface. The y faces cut a *sloped* surface
  // along lines of constant elevation, so they read as topographic contour
  // bands, one per brick height. Halving the lattice halved the banding
  // (7 549 to 4 003 detected edges, mean band width 10.4 to 16.8 px), which is
  // the signature of a lattice artifact rather than of anything in the field:
  // the stored terrain distance is exactly \`y - h(column)\`, so the in-leaf
  // trilinear normal is analytically correct and only the clamp can corrupt it.
  //
  // The cost is a root-to-leaf descent per out-of-range tap. Interior taps —
  // the common case — keep the single fetch they had.
  const surfaceReconstructionWGSL = /* wgsl */ `
${sparseBrickSceneGeometryCodecWGSL(sceneGeometryFormat)}
${sceneDistanceLoadWGSL}
const DRY_LEAF_NONE:u32=0xffffffffu;
fn drySurfaceReconstruction()->u32{return u32(dry.lod.w);}
/** One voxel of a known leaf, by that leaf's own local cell. */
fn drySceneDistanceOfVoxel(voxel:u32)->f32{
  let word=drySceneDistanceWord(voxel);
  // An unpublished or short lane reads as zero, which the caller sees as a
  // degenerate gradient and answers with the face normal — never as a surface.
  if(word>=arrayLength(&sceneGeometry)){return 0.0;}
  return drySceneDistanceDecode(word,voxel);
}
/**
 * The deepest published leaf whose brick covers one finest-lattice cell.
 *
 * The same descent \`deepestLeaf\` performs in the derived builder
 * (lib/webgpu-svo-live-derived-builder.ts), over the same topology arena, so the
 * two cannot disagree about which leaf owns a cell. It keeps the *last* active
 * leaf seen on the way down rather than requiring a terminal one, because an
 * interior node may carry a leaf that a missing child mask stops the walk above.
 */
fn dryLeafForGlobalCell(globalCell:vec3u)->u32{
  let brickSize=max(dry.mapping.brickSize,1u);
  let finest=dry.mapping.maximumDepth;
  let brick=globalCell/brickSize;
  var node=0u;var selected=DRY_LEAF_NONE;
  for(var level=0u;level<=finest;level+=1u){
    if(node>=dry.mapping.nodeCount){break;}
    let record=svoNodeLoad(node);
    if(record.links.z<dry.mapping.leafCount
      &&svoBrickLifecycleCurrent(svoBrickLifecycleDecode(record.links.w))){selected=record.links.z;}
    if(level==finest){break;}
    let bit=finest-level-1u;
    let octant=((brick.x>>bit)&1u)|(((brick.y>>bit)&1u)<<1u)|(((brick.z>>bit)&1u)<<2u);
    let mask=record.address.w&0xffu;
    if((mask&(1u<<octant))==0u){break;}
    node=record.links.x+countOneBits(mask&((1u<<octant)-1u));
  }
  return selected;
}
/** That leaf's payload voxel for the same cell, honouring its own level. */
fn drySceneDistanceAtGlobalCell(globalCell:vec3u)->f32{
  let leafIndex=dryLeafForGlobalCell(globalCell);
  if(leafIndex==DRY_LEAF_NONE){return 0.0;}
  let leaf=svoLeafLoad(leafIndex);
  let record=svoNodeLoad(leaf.topology.x);
  let brickSize=max(dry.mapping.brickSize,1u);
  // A neighbour may sit at a coarser level, where one of its cells spans
  // \`scale\` finest cells. The stored distance is in metres rather than in cell
  // units, so a coarser sample is still a valid distance at that point — only
  // its centre is further from the tap, which is a smaller error than the zero
  // gradient a clamp would have produced.
  let scale=1u<<(dry.mapping.maximumDepth-record.address.z);
  let origin=svoDecodeMorton(record.address.x,record.address.y,record.address.z)*scale*brickSize;
  let local=min((globalCell-origin)/scale,vec3u(brickSize-1u));
  return drySceneDistanceOfVoxel(svoBrickVoxelIndex(leaf.topology.y,local,brickSize));
}
/**
 * One stencil tap.
 *
 * Taps inside the hit leaf — the common case, and all eight of them whenever the
 * dual cell does not straddle a face — read that leaf's own payload directly and
 * cost one fetch, exactly as the clamped version did. Only a tap that has left
 * the brick pays the descent, which is what keeps the neighbour resolution off
 * the price of every shaded pixel.
 *
 * A tap below the lattice origin is outside the tree rather than inside a
 * neighbour, so it answers zero the same way an unpublished lane does: the
 * caller reads the degenerate gradient and keeps the face normal.
 */
fn drySceneDistanceAtTap(voxelOffset:u32,brickSize:u32,origin:vec3i,cell:vec3i,step:i32,offset:vec3i)->f32{
  let local=cell+offset;
  if(all(local>=vec3i(0))&&all(local<vec3i(i32(brickSize)))){
    return drySceneDistanceOfVoxel(svoBrickVoxelIndex(voxelOffset,vec3u(local),brickSize));
  }
  let global=origin+local*step;
  if(any(global<vec3i(0))){return 0.0;}
  return drySceneDistanceAtGlobalCell(vec3u(global));
}
/**
 * Outward normal of the trilinear reconstruction of the distance field.
 *
 * Samples live at voxel *centres*, so the dual cell containing \`point\` is offset
 * half a voxel from the storage lattice: brick-local coordinate \`c + 0.5\` holds
 * cell \`c\`'s value, and the interpolation cell is therefore based at
 * \`floor(local - 0.5)\`. Getting that half-voxel wrong reads as a normal that
 * lags the surface by half a cell, which looks like a lighting bias rather than
 * like an addressing bug.
 *
 * The gradient is the analytic derivative of the same eight values, so it costs
 * no further fetches, and it is continuous wherever the eight are. Distance is
 * negative inside, so the gradient already points outward and needs no flip.
 *
 * Taps are addressed on the *finest global lattice* rather than in leaf-local
 * cells, because a dual cell may straddle a brick face and the whole point is
 * that it may. \`hit\` supplies the leaf whose bounds set the frame; the taps then
 * resolve their own leaves, which is what keeps the field continuous across one.
 */
fn dryTrilinearSurfaceNormal(hit:SvoTraversalHit,bounds:mat2x3f,extent:vec3f,point:vec3f,brickSize:u32)->vec3f{
  let local=(point-bounds[0])/extent-vec3f(0.5);
  let base=floor(local);
  let f=clamp(local-base,vec3f(0.0),vec3f(1.0));
  let c=vec3i(base);
  // The hit leaf's own origin and cell span on the finest lattice. A tap at
  // leaf-local cell k covers finest cells origin + k*scale up to +scale, and
  // the half-scale offset picks that span's centre so a tap on a brick face
  // resolves to the cell it is inside rather than to the one it borders.
  let node=svoNodeLoad(hit.nodeIndex);
  let scale=1u<<(dry.mapping.maximumDepth-node.address.z);
  let origin=vec3i(svoDecodeMorton(node.address.x,node.address.y,node.address.z)*scale*brickSize)
    +vec3i(i32(scale>>1u));
  let step=i32(scale);
  let vo=hit.voxelOffset;
  let d000=drySceneDistanceAtTap(vo,brickSize,origin,c,step,vec3i(0,0,0));
  let d100=drySceneDistanceAtTap(vo,brickSize,origin,c,step,vec3i(1,0,0));
  let d010=drySceneDistanceAtTap(vo,brickSize,origin,c,step,vec3i(0,1,0));
  let d110=drySceneDistanceAtTap(vo,brickSize,origin,c,step,vec3i(1,1,0));
  let d001=drySceneDistanceAtTap(vo,brickSize,origin,c,step,vec3i(0,0,1));
  let d101=drySceneDistanceAtTap(vo,brickSize,origin,c,step,vec3i(1,0,1));
  let d011=drySceneDistanceAtTap(vo,brickSize,origin,c,step,vec3i(0,1,1));
  let d111=drySceneDistanceAtTap(vo,brickSize,origin,c,step,vec3i(1,1,1));
  let gx=mix(mix(d100-d000,d110-d010,f.y),mix(d101-d001,d111-d011,f.y),f.z);
  let gy=mix(mix(d010-d000,d110-d100,f.x),mix(d011-d001,d111-d101,f.x),f.z);
  let gz=mix(mix(d001-d000,d101-d100,f.x),mix(d011-d010,d111-d110,f.x),f.y);
  // The lane's units cancel under normalize — which is what lets the narrowed
  // \`snorm8\` arm, stored in cell-band units, share this code — but the per-axis
  // cell size does not, so a non-cubic voxel needs the division.
  return vec3f(gx,gy,gz)/max(extent,vec3f(1e-20));
}

struct DryShadingNormal{normal:vec3f,featureId:u32}

/**
 * The ground's normal, differentiated from the heightfield and limited at its
 * own edges.
 *
 * Differentiating \`h\` and not what was stored of it. The writer folds terrain
 * into the scene-geometry lane as \`y - h(x,z)\`
 * (\`webgpu-sparse-scene-proxies.ts\`), a *vertical* pseudo-distance whose
 * gradient is the surface normal on a gentle slope and collapses toward +Y on a
 * steep one — which is what drew the pond coping's vertical wall as columns
 * pointing at the sky. \`(-dh/dx, 1, -dh/dz)\` has no such failure at any slope
 * the graph can express.
 *
 * ## Why the difference is limited rather than centred
 *
 * A centred difference at the coping's arris straddles the drop, so the flat top
 * within one cell of the edge is handed a normal tilted out over the wall, and
 * the edge grows a dark serrated fringe along every upper silhouette in the
 * scene. Widening the window makes that worse, not better; narrowing it below a
 * cell aliases.
 *
 * The minmod limiter takes the smaller of the two one-sided slopes, and zero
 * where they disagree in sign. It is not a tuned threshold — there is nothing to
 * choose — and it can only ever reduce a slope, never invent one. At the arris
 * the along-the-top difference is the smaller and the top keeps its own normal,
 * sharp and unfringed; over ordinary ground the two sides agree and it is the
 * centred difference it replaces; on the wall face itself it declines to guess,
 * which leaves that face flat-lit rather than striped.
 *
 * A vertical face is a surface a heightfield does not have — the column is
 * multi-valued there — and no sampling of \`h\` recovers it. Three widths and a
 * height-aware regime split were measured against the coping and none of them
 * did; the fix for that face is geometry, not a gradient.
 */
fn dryTerrainSurfaceNormal(point:vec3f)->vec3f{
  let epsilon=max(max(dry.mapping.cellSize.x,dry.mapping.cellSize.z),1e-5)
    *${SVO_SURFACE_TERRAIN_GRADIENT_CELLS}.0;
  let centre=terrainHeightAt(point.x,point.z);
  let forward=vec2f(terrainHeightAt(point.x+epsilon,point.z),terrainHeightAt(point.x,point.z+epsilon));
  let backward=vec2f(terrainHeightAt(point.x-epsilon,point.z),terrainHeightAt(point.x,point.z-epsilon));
  let ahead=(forward-vec2f(centre))/epsilon;
  let behind=(vec2f(centre)-backward)/epsilon;
  let slope=select(vec2f(0.0),select(ahead,behind,abs(behind)<abs(ahead)),ahead*behind>vec2f(0.0));
  return normalize(vec3f(-slope.x,1.0,-slope.y));
}

/**
 * The exact outward normal of whatever filled this cell, or a zero vector.
 *
 * The DDA resolved \`owner\` on its way past this cell for the analytic upgrade
 * path, and every authored primitive's normal is bound in this same shader, so
 * the answer is one record fetch and no payload taps at all — against the
 * stencil's eight, any of which pays a root-to-leaf descent when the dual cell
 * straddles a brick face.
 *
 * ## The normal only, not the sample
 *
 * \`svoEvaluatePrimitive\` is the obvious call and it is the wrong one: it
 * evaluates \`svoPrimitiveDistance_m\` beside the normal, and for a marched kind —
 * the smooth-union clusters and field programs the garden's canopies are made of
 * — that distance is a whole field evaluation thrown away unread. Measured on
 * \`garden-svo-lighting\`, three interleaved rounds, it cost more than the stencil
 * it replaced. Rotating \`svoPrimitiveLocalNormal\` by hand is the same arithmetic
 * as that function's tail with the discarded half deleted.
 *
 * Terrain is ownerless by construction (\`SPARSE_SCENE_TERRAIN_MATERIAL_OWNER\`),
 * so it is recognised by material and answered by the heightfield — which is
 * also why nothing here reads the ground: an authored record is never of terrain
 * kind, and sampling \`h\` eagerly to serve a case that cannot arise would put
 * five ground samples on every shaded pixel in the scene.
 */
fn dryAnalyticSurfaceNormal(owner:u32,materialId:u32,point:vec3f)->DryShadingNormal{
  if(owner!=${DRY_UPGRADE_RUN_NONE}u){
    let record=dryPrimitive(owner);
    let local=svoPrimitiveLocalNormal(record,svoPrimitiveLocalPoint(record,point),dryClusterPacking(record));
    let magnitude=length(local.xyz);
    if(magnitude>1e-8){
      return DryShadingNormal(svoQuaternionRotate(record.orientation,local.xyz/magnitude),u32(local.w));
    }
  }
  if(terrainEnabled()&&materialId==dry.terrain.x){
    return DryShadingNormal(dryTerrainSurfaceNormal(point),SVO_FEATURE_TERRAIN);
  }
  return DryShadingNormal(vec3f(0.0),SVO_FEATURE_SMOOTH);
}

/**
 * The stencil, required to certify itself before it is believed.
 *
 * ${sceneDistanceIsMetres[sceneGeometryFormat]
    ? `A signed distance field satisfies \`|grad d| = 1\`, and this lane stores
 * metres, so the magnitude \`length\` already computes is a validity certificate
 * that costs nothing to read. Where the lane is a real distance it is 1 to
 * within quantisation; where it is \`y - h\`, or a \`min\` over a neighbour brick's
 * different candidate set, it is not — and those are exactly the cells whose
 * gradients drew rectangular debris across flat walls.`
    : `This lane does not store metres, so \`|grad d| = 1\` says nothing about it
 * and only the degeneracy test applies.`}
 *
 * The second test is cheaper and blunter: a surface passing through this cell
 * cannot face more than a right angle away from the face the ray entered
 * through. Both failures answer with that face, which is the value the arm
 * without any reconstruction would have used.
 */
fn dryCertifiedSurfaceNormal(hit:SvoTraversalHit,bounds:mat2x3f,extent:vec3f,point:vec3f,faceNormal:vec3f)->vec3f{
  let gradient=dryTrilinearSurfaceNormal(hit,bounds,extent,point,dry.mapping.brickSize);
  let magnitude=length(gradient);
  if(!(magnitude>1e-12)){return faceNormal;}
${sceneDistanceIsMetres[sceneGeometryFormat]
    ? `  if(abs(magnitude-1.0)>${SVO_SURFACE_ANALYTIC_EIKONAL_TOLERANCE}){return faceNormal;}\n`
    : ""}\
  let normal=gradient/magnitude;
  if(dot(normal,faceNormal)<=0.0){return faceNormal;}
  return normal;
}

/** The uniform's arm, resolved once per shaded voxel hit. */
fn dryShadingNormal(hit:SvoTraversalHit,bounds:mat2x3f,extent:vec3f,point:vec3f,faceNormal:vec3f,owner:u32,identity:u32)->DryShadingNormal{
  let mode=drySurfaceReconstruction();
  if(mode==${SVO_SURFACE_RECONSTRUCTION_CODES.trilinear}u){
    // The evidence arm, unchanged and uncertified: it is what the analytic arm
    // is measured against, so it has to keep drawing what it drew.
    let gradient=dryTrilinearSurfaceNormal(hit,bounds,extent,point,dry.mapping.brickSize);
    let magnitude=length(gradient);
    if(magnitude>1e-12){return DryShadingNormal(gradient/magnitude,SVO_FEATURE_SMOOTH);}
    return DryShadingNormal(faceNormal,SVO_FEATURE_SMOOTH);
  }
  if(mode!=${SVO_SURFACE_RECONSTRUCTION_CODES.analytic}u){return DryShadingNormal(faceNormal,SVO_FEATURE_SMOOTH);}
  let exact=dryAnalyticSurfaceNormal(owner,identity&0xffffu,point);
  // A normal more than a right angle from the entered face is not this cell's
  // surface however exactly it was computed — an edge feature resolved from a
  // point up to a voxel off the primitive can land there — and the face is the
  // better answer for the same reason it is in the certificate above.
  if(dot(exact.normal,exact.normal)>0.0&&dot(exact.normal,faceNormal)>0.0){return exact;}
  return DryShadingNormal(dryCertifiedSurfaceNormal(hit,bounds,extent,point,faceNormal),SVO_FEATURE_SMOOTH);
}
`;
  // The compile flag decides whether the LOD machinery exists; the uniform
  // decides what it does. A build without it is the bit-exact reference image,
  // and a runtime threshold of zero reproduces that image from the LOD build —
  // which is what lets the panel drag the slider to zero and get the reference
  // rather than "very nearly" it.
  const lodUniformWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
fn dryLodMode()->u32{return u32(dry.lod.y);}
fn dryLodFixedLevel()->u32{return u32(dry.lod.z);}
/** Authored at the contract's reference height, rescaled here so it stays angular. */
fn dryLodThresholdPixels()->f32{
  if(dryLodMode()==SVO_LOD_MODE_FIXED_LEVEL){return 0.0;}
  return dry.lod.x*uniforms.viewport.y/${SVO_SCREEN_SPACE_TERMINATION_CONTRACT.referenceViewportHeightPixels};
}
/** Cells per aggregate step for one brick. One is the exact per-cell walk. */
fn dryLodCellStride(brickBounds:mat2x3f,leafLevel:u32)->u32{
  let brickSize=max(dry.mapping.brickSize,1u);
  if(dryLodMode()==SVO_LOD_MODE_FIXED_LEVEL){return svoLodCellStrideForLevel(leafLevel,dryLodFixedLevel(),brickSize);}
  return svoLodScreenSpaceCellStride(brickBounds,brickSize,uniforms.cameraPosition.xyz,
    uniforms.viewport.y,cameraTanHalfFov(),dryLodThresholdPixels());
}
` : "";
  const primaryTraversalCursorWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
fn drySvoShouldTerminateNodeScreenSpace(bounds:mat2x3f,level:u32)->bool{
  if(dryLodMode()==SVO_LOD_MODE_FIXED_LEVEL){return level>=dryLodFixedLevel();}
  return svoShouldTerminateNodeScreenSpace(bounds,uniforms.cameraPosition.xyz,uniforms.viewport.y,cameraTanHalfFov(),dryLodThresholdPixels(),level,0u);
}
fn dryTraversalCursorNextPrimary(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{
  return svoTraversalContinuationNextScreenSpace(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);
}
` : /* wgsl */ `
fn dryTraversalCursorNextPrimary(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return dryTraversalCursorNext(ray,mapping,cursor);}
`;
  const screenSpaceProxyWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
const DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY:u32=15u;
const DRY_GBUFFER_FIELD_RESIDENT_CELL_PROXY:u32=14u;
fn dryScreenSpaceProxyHit(ro:vec3f,rd:vec3f,hit:SvoTraversalHit)->DryHit{
  let bounds=svoNodeBounds(svoNodeLoad(hit.nodeIndex),dry.mapping);let point=ro+rd*hit.tEnter;
  let faceDistance=min(abs(point-bounds[0]),abs(bounds[1]-point));var axis=0u;
  if(faceDistance.y<faceDistance.x){axis=1u;}if(faceDistance.z<faceDistance[axis]){axis=2u;}
  var normal=vec3f(0.0);normal[axis]=select(-1.0,1.0,abs(point[axis]-bounds[1][axis])<abs(point[axis]-bounds[0][axis]));
  return DryHit(hit.tEnter,normal,0u,DRY_OWNER_NONE,SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(hit.level,0u,0u));
}
` : "";
  const screenSpaceProxyTraceWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `if(leaf.status==SVO_STATUS_SCREEN_SPACE_PROXY){return dryScreenSpaceProxyHit(ro,rd,leaf);}` : "";
  const screenSpaceProxyShadeWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `if(hit.fieldSource==DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY){let depthBand=clamp(f32(hit._padding.x)/21.0,0.0,1.0);return mix(vec3f(.02,.06,.18),vec3f(1.0,.04,.72),depthBand);}` : "";
  const screenSpacePrimaryProxyWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
fn dryPrimaryBoundsSubPixel(bounds:mat2x3f)->bool{
  return svoShouldTerminateNodeScreenSpace(bounds,uniforms.cameraPosition.xyz,uniforms.viewport.y,
    cameraTanHalfFov(),dryLodThresholdPixels(),0u,0u);
}
// Whole-brick collapse: everything this instance can draw, measured at once.
//
// It used to measure the brick's camera-nearest *cell* and collapse all eight
// cells across on that answer — an eight-fold over-coarsening that stayed
// invisible while 25 mm cells put its onset near 14 m, and became structural at
// 6.25 mm, where it starts around 3.5 m and so claims most of the garden.
// Compounding it, the bounds handed in are the occupied sub-AABB the emit
// kernel publishes (webgpu-svo-brick-raster.ts:727), not the brick box, so
// span/brickSize understated the cell by as much as another eight per axis on
// exactly the sparse foliage bricks where the error is most visible.
//
// The sub-AABB is the right thing to measure *as a whole*: it is the tight
// bound on every surface the brick can produce, so a sub-threshold sub-AABB
// genuinely cannot express more than one proxy. Intermediate granularity is now
// the aggregate stride's job (dryLodCellStride), not this predicate's.
fn dryPrimaryBrickProxySubPixel(bounds:mat2x3f,leafLevel:u32)->bool{
  if(dryLodMode()==SVO_LOD_MODE_FIXED_LEVEL){return dryLodFixedLevel()<=leafLevel;}
  return dryPrimaryBoundsSubPixel(bounds);
}
fn dryPrimaryProxyNormal(bounds:mat2x3f,point:vec3f)->vec3f{
  let faceDistance=min(abs(point-bounds[0]),abs(bounds[1]-point));var axis=0u;
  if(faceDistance.y<faceDistance.x){axis=1u;}if(faceDistance.z<faceDistance[axis]){axis=2u;}
  var normal=vec3f(0.0);
  normal[axis]=select(-1.0,1.0,abs(point[axis]-bounds[1][axis])<abs(point[axis]-bounds[0][axis]));
  return normal;
}
// The entry face, not the interval midpoint.
//
// The midpoint pushed the proxy half an aggregate behind the surface it stands
// for — 25 mm at a 50 mm brick — and this hit is depth-tested against the exact
// tier's, so the bias was not cosmetic: it let an exact neighbouring fragment
// win a silhouette pixel the proxy actually owned. The entry face is the
// nearest point of the aggregate the ray touches, which is what a solid
// aggregate presents, and it makes a stride-one proxy agree with the exact
// voxel hit exactly rather than approximately.
fn dryPrimaryVoxelProxyHit(ro:vec3f,rd:vec3f,bounds:mat2x3f,tEnter:f32,identity:u32)->DryHit{
  let owner=identity>>16u;let material=identity&0xffffu;
  return DryHit(tEnter,dryPrimaryProxyNormal(bounds,ro+rd*tEnter),material,owner,SVO_FEATURE_SMOOTH,
    DRY_GBUFFER_FIELD_RESIDENT_CELL_PROXY,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
}
// A material is what makes a cell solid; an owner is what makes it *solvable*.
// The full argument is at traceLeafPayload, and this is the same test — it
// has to be. This function is the only resolve a pixel classified into the LOD
// tier ever gets ("the exact marcher is unreachable from the far LOD arm"), so
// an owner-range test here is not a missed upgrade, it is a hole: every cell
// the voxeliser filled without an in-range owner — ground is ownerless by
// construction, see SPARSE_SCENE_TERRAIN_MATERIAL_OWNER, and on the hero garden
// so is most of the scenery — discarded the fragment to background.
fn dryLodCellSolid(identity:u32)->bool{
  let owner=identity>>16u;
  return (identity&0xffffu)!=0u&&!(owner!=${SPARSE_BRICK_NO_OWNER}u&&dryOpaqueOwnerSuppressed(owner));
}
// One identity standing for the whole brick, for the ray that crosses it
// without crossing a solid cell.
//
// The macrocell mask makes this bounded and exact rather than a search: a set
// bit promises at least one solid cell inside that 4^3 block, so the first set
// bit's block is normally the only one scanned, and a typical surface brick
// answers within a few fetches of entering it. A block whose cells all belong
// to a suppressed owner is the one case that falls through to the next bit,
// and a brick that is suppressed outright returns zero — which is a genuine
// absence of surface, not a missing answer.
fn dryLodBrickRepresentativeIdentity(voxelOffset:u32,occupancy:SvoBrickOccupancy)->u32{
  let brickSize=max(dry.mapping.brickSize,1u);
  for(var macroBit=0u;macroBit<8u;macroBit+=1u){
    if(occupancy.ready!=0u&&(occupancy.macroMask&(1u<<macroBit))==0u){continue;}
    let origin=vec3u(macroBit&1u,(macroBit>>1u)&1u,(macroBit>>2u)&1u)*4u;
    for(var index=0u;index<64u;index+=1u){
      let local=origin+vec3u(index&3u,(index>>2u)&3u,(index>>4u)&3u);
      if(any(local>=vec3u(brickSize))){continue;}
      let address=svoBrickVoxelIndex(voxelOffset,local,brickSize);
      if(address<arrayLength(&materialOwners)){
        let identity=materialOwners[address];
        if(dryLodCellSolid(identity)){return identity;}
      }
    }
  }
  return 0u;
}
/** First solid cell of one aggregate, scanned along the ray. Aggregate-local DDA. */
fn dryLodAggregateIdentity(ro:vec3f,rd:vec3f,brickMinimum:vec3f,extent:vec3f,voxelOffset:u32,
  cellMinimum:vec3u,stride:u32,tEnter:f32,tExit:f32)->u32{
  let brickSize=max(dry.mapping.brickSize,1u);
  let limit=vec3i(cellMinimum+vec3u(stride));
  var entry=max(tEnter,0.0);let point=ro+rd*(entry+1e-5);
  var cell=clamp(vec3i(floor((point-brickMinimum)/extent)),vec3i(cellMinimum),limit-vec3i(1));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let nextBoundary=brickMinimum+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));
  let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));
  // A DDA through an s^3 subgrid crosses at most 3s-2 cells; 24 covers s=8.
  for(var iteration=0u;iteration<24u;iteration+=1u){
    if(any(cell<vec3i(cellMinimum))||any(cell>=limit)||entry>tExit){break;}
    let index=svoBrickVoxelIndex(voxelOffset,vec3u(cell),brickSize);
    if(index<arrayLength(&materialOwners)){
      let identity=materialOwners[index];
      if(dryLodCellSolid(identity)){return identity;}
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return 0u;
}
// The intermediate rung: step the brick at stride cells and shade the
// aggregate that stops the ray.
//
// The bounds are the brick's own box and never the instance's occupied sub-AABB.
// The distinction is load-bearing and was the second half of the LOD tier's
// damage: cell coordinates derived from the sub-AABB were fed straight to
// svoBrickVoxelIndex as full-brick indices, so a brick whose occupancy filled
// cells [4..7]^3 rendered a squeezed copy of the whole brick's contents into
// that corner. Callers clip the *interval* with the proxy and pass the box.
//
// What the aggregate cannot do yet is answer without descending: occupancy is
// aggregated (macroMask, free in the node's flags word) but identity is not,
// so an occupied aggregate still scans its own cells for the first solid one.
// The saving is therefore empty space — the mask rejects a 4^3 region in one
// bit test where the fine walk paid a payload fetch per cell — and the shading
// is exact-material-at-aggregate-resolution rather than an invented average.
// One representative identity per macrocell (8 u32 per brick, 134 KB at the
// hero's 4 187) is what would make an occupied aggregate free as well.
fn dryPrimaryLeafAggregateHit(ro:vec3f,rd:vec3f,bounds:mat2x3f,tEnter:f32,tExit:f32,
  voxelOffset:u32,occupancy:SvoBrickOccupancy,stride:u32)->DryHit{
  let brickSize=max(dry.mapping.brickSize,1u);
  let extent=(bounds[1]-bounds[0])/f32(brickSize);
  let aggregates=max(brickSize/max(stride,1u),1u);let aggregateExtent=extent*f32(stride);
  var entry=max(tEnter,0.0);let point=ro+rd*(entry+1e-5);
  var block=clamp(vec3i(floor((point-bounds[0])/aggregateExtent)),vec3i(0),vec3i(i32(aggregates)-1));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let nextBoundary=bounds[0]+(vec3f(block)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*aggregateExtent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));
  let deltaT=select(vec3f(DRY_MISS),abs(aggregateExtent/rd),abs(rd)>vec3f(1e-9));
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(block<vec3i(0))||any(block>=vec3i(i32(aggregates)))||entry>tExit){break;}
    let blockExit=min(min(nextT.x,nextT.y),min(nextT.z,tExit));
    let cellMinimum=vec3u(block)*stride;
    // Free rejection: the producer publishes a 4^3 macrocell mask in the
    // terminal node's spare flags word, and an aggregate of four cells or fewer
    // lies wholly inside one macrocell because the grids are aligned and the
    // strides divide. A whole-brick aggregate spans all eight macrocells, so it
    // must consult the occupied bit instead — testing macrocell zero there
    // would reject any brick whose contents sit in another octant.
    let aggregateOccupied=select(svoBrickMacroOccupied(occupancy,cellMinimum),
      occupancy.ready==0u||occupancy.occupied!=0u,stride>4u);
    if(aggregateOccupied){
      let identity=dryLodAggregateIdentity(ro,rd,bounds[0],extent,voxelOffset,cellMinimum,stride,entry,blockExit);
      if(identity!=0u){
        let blockBounds=mat2x3f(bounds[0]+vec3f(cellMinimum)*extent,
          bounds[0]+vec3f(cellMinimum+vec3u(stride))*extent);
        return dryPrimaryVoxelProxyHit(ro,rd,blockBounds,entry,identity);
      }
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){block.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){block.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){block.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return missHit();
}
// LOD-tier entry: brick box from the node, interval already clipped by the proxy.
//
// A sub-threshold brick is drawn as one voxel, and the emphasis is on *drawn*:
// this arm may not answer "nothing". It keeps exactly one candidate per pixel —
// the coverage pass depth-tests the proxy boxes and the winner is the only
// instance the resolve is ever handed — and every brick behind that winner was
// routed away from the exact arena for being sub-threshold as well. So a miss
// here is not "try the next brick", it is a hole straight through to terrain or
// sky, which is what the LOD slider was producing at every threshold high
// enough to collapse overlapping geometry.
//
// Over-covering the proxy box instead is the error the budget already names:
// the box only exists because it measured at or under the threshold, so drawing
// it solid can grow a silhouette by at most that many pixels, and at the
// contract's default of three it is invisible. Descending still runs first and
// still picks the material, because a proxy box is tens of pixels across at a
// high threshold and the along-ray cell beats one representative for all of it.
fn dryPrimaryLeafProxyHit(ro:vec3f,rd:vec3f,nodeIndex:u32,proxyBounds:mat2x3f,tEnter:f32,tExit:f32,voxelOffset:u32)->DryHit{
  let node=svoNodeLoad(nodeIndex);let bounds=svoNodeBounds(node,dry.mapping);
  let occupancy=svoBrickOccupancyDecode(node.links.w);
  let stride=dryLodCellStride(bounds,node.address.z);
  let descended=dryPrimaryLeafAggregateHit(ro,rd,bounds,tEnter,tExit,voxelOffset,occupancy,stride);
  if(descended.t<DRY_MISS){return descended;}
  let identity=dryLodBrickRepresentativeIdentity(voxelOffset,occupancy);
  if(identity==0u){return missHit();}
  return dryPrimaryVoxelProxyHit(ro,rd,proxyBounds,max(tEnter,0.0),identity);
}
fn dryPrimaryPrimitiveProxyHit(record:SvoPrimitiveRecord,ro:vec3f,rd:vec3f,span:vec2f)->DryHit{
  let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let extent=dryScenePrimitiveWorldExtent(localExtent,record.orientation);let centre=svoPrimitiveCenter_m(record);
  let bounds=mat2x3f(centre-extent,centre+extent);let t=mix(span.x,span.y,0.5);
  return DryHit(t,dryPrimaryProxyNormal(bounds,ro+rd*t),svoPrimitiveMaterialId(record),
    svoPrimitiveOwnerId(record),SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_ANALYTIC,
    DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
}
` : "";
  const tieredComputeResolveDeclarationsWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
// W2 scene-primitive compute tier. The four color planes are storage-written
// only for winning pixels; a tiny depth-only bridge commits reversed-Z after
// the compute pass, avoiding another full 48-byte MRT load/store cycle.
@group(${splitGroup + 1}) @binding(0) var dryTierCurrentDepth:texture_depth_2d;
@group(${splitGroup + 1}) @binding(1) var dryTierPackedSurfaceWrite:texture_storage_2d<rgba32uint,write>;
@group(${splitGroup + 1}) @binding(2) var dryTierIdentityMediaWrite:texture_storage_2d<rgba16uint,write>;
@group(${splitGroup + 1}) @binding(3) var dryTierGeometryWrite:texture_storage_2d<rgba32float,write>;
@group(${splitGroup + 1}) @binding(4) var dryTierOpaqueIdentityWrite:texture_storage_2d<rg32uint,write>;
@group(${splitGroup + 1}) @binding(5) var dryTierDepthWrite:texture_storage_2d<r32float,write>;
struct DryTieredResolveQueue{
  dispatchX:atomic<u32>,dispatchY:atomic<u32>,dispatchZ:atomic<u32>,pixelCount:atomic<u32>,pixels:array<u32>,
}
@group(${splitGroup + 1}) @binding(8) var<storage,read_write> dryTieredResolveQueue:DryTieredResolveQueue;
` : "";
  const sceneCoverageReturnTypeWGSL = screenSpaceTerminationPixels > 0
    ? "->DryBrickCoverageOut" : "->@location(0) u32";
  const sceneCoverageMissWGSL = screenSpaceTerminationPixels > 0
    ? "return DryBrickCoverageOut(0u,0.0);" : "return 0u;";
  const sceneCoverageOverflowWGSL = screenSpaceTerminationPixels > 0
    ? "return DryBrickCoverageOut(0u,0.0);" : "return 1u;";
  const screenSpaceSceneCoverageWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let worldExtent=dryScenePrimitiveWorldExtent(localExtent,record.orientation);let centre=svoPrimitiveCenter_m(record);
  if(dryPrimaryBoundsSubPixel(mat2x3f(centre-worldExtent,centre+worldExtent))){
    return DryBrickCoverageOut(input.primitiveIndex+1u,dryHardwareDepth(span.x,rd,camera[1]));
  }
  let resolvedDepth=textureLoad(dryRasterPrimaryGeometryRead,vec2i(coordinate),0).w;
  let resolvedMetadata=textureLoad(dryRasterPrimaryIdentityRead,vec2i(coordinate),0).y;
  if(((resolvedMetadata>>20u)&15u)==DRY_GBUFFER_FIELD_RESIDENT_CELL_PROXY
    &&(resolvedMetadata&0xffffu)==svoPrimitiveOwnerId(record)){${sceneCoverageMissWGSL}}
  if(!(span.x<resolvedDepth)){${sceneCoverageMissWGSL}}
` : "";
  // Retired under voxels-only, and the reason is worth keeping.
  //
  // This per-cell sub-pixel test sat *inside* the owner-range branch, so once
  // the walk accepted solidity as `material != 0` every ownerless cell reached
  // `primaryVoxelSurfaceWGSL` without ever being asked whether it was
  // sub-pixel. Hoisting it out would have been the obvious repair, and it buys
  // nothing: what it saved was the analytic `primitiveHit` march, and under
  // voxels-only there is no march — both arms return at the first solid cell,
  // and the proxy arm returns the *same* cell with a coarser depth. A test that
  // costs a footprint evaluation per cell to choose between two identical
  // answers is a cost, not a level of detail.
  //
  // Real intermediate detail is the aggregate stride (`primaryLodStrideWGSL`),
  // decided once per brick instead of once per cell. The control arm that still
  // marches records keeps the original test, where it still earns its place.
  const screenSpaceVoxelResolveWGSL = screenSpaceTerminationPixels > 0 && !voxelsOnlyPrimary ? /* wgsl */ `
        let cellBounds=mat2x3f(bounds[0]+vec3f(cell)*extent,bounds[0]+(vec3f(cell)+vec3f(1.0))*extent);
        if(dryPrimaryBoundsSubPixel(cellBounds)){
          ${primaryUpgradeFlushWGSL}
          return dryPrimaryVoxelProxyHit(ro,rd,cellBounds,entry,identity);
        }` : "";
  const screenSpacePrimitiveResolveWGSL = screenSpaceTerminationPixels > 0
    ? "let proxyLocalExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));let proxyExtent=dryScenePrimitiveWorldExtent(proxyLocalExtent,record.orientation);let proxyCentre=svoPrimitiveCenter_m(record);if(dryPrimaryBoundsSubPixel(mat2x3f(proxyCentre-proxyExtent,proxyCentre+proxyExtent))){let proxySpan=vec2f(span.x,min(span.y,limit));return dryPrimaryPrimitiveProxyHit(record,ro,rd,proxySpan);}"
    : "";
  const brickCoverageOutputWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
struct DryBrickCoverageOut{@location(0) key:u32,@builtin(frag_depth) hardwareDepth:f32}
` : "";
  const brickCoverageReturnTypeWGSL = screenSpaceTerminationPixels > 0
    ? "->DryBrickCoverageOut" : "->@location(0) u32";
  const brickCoverageMissWGSL = screenSpaceTerminationPixels > 0
    ? "return DryBrickCoverageOut(0u,0.0);" : "return 0u;";
  const brickCoverageOverflowWGSL = screenSpaceTerminationPixels > 0
    ? "return DryBrickCoverageOut(0u,0.0);" : "return 1u;";
  const screenSpaceBrickCoverageWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  let proxyBounds=mat2x3f(input.proxyMinimum,input.proxyMaximum);
  if(dryPrimaryBrickProxySubPixel(proxyBounds,dryLeafLevel(input.nodeIndex))){
    let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
    let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,proxyBounds);
    if(interval.x!=0.0){let entry=max(interval.y,0.0);return DryBrickCoverageOut(input.instanceIndex+1u,dryHardwareDepth(entry,rd,camera[1]));}
  }
` : "";
  const screenSpaceBrickResolveWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  let lodKey=textureLoad(drySplitGlassKeyRead,vec2i(position),0).x;
  if(lodKey>0u){let instanceIndex=lodKey-1u;if(instanceIndex<arrayLength(&svoBrickInstances)){
    let record=svoBrickInstances[instanceIndex];let proxyBounds=mat2x3f(record.proxyMinimum,record.proxyMaximum);
    let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,proxyBounds);
    if(interval.x!=0.0){let proxy=dryPrimaryLeafProxyHit(ro,rd,record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK,proxyBounds,max(interval.y,0.0),interval.z,record.voxelOffset);
      if(proxy.t<opaque.t){opaque=proxy;producer=SVO_GBUFFER_PRODUCER_BRICK;}}
  }}
` : "";
  // The thresholded production graph resolves background, LOD and exact
  // surfaces in separate passes, so it has no single full-screen writer for
  // the old candidate-arena depth seed. Hardware depth remains authoritative;
  // omit this optional early-out instead of consulting stale arena words.
  const sceneCoverageSeedCullWGSL = screenSpaceTerminationPixels > 0 ? "" : /* wgsl */ `
  let seedIndex=dryPrimaryCoverageSeedIndex(pixel);
  if(seedIndex<arrayLength(&svoBrickCoverageCandidates)
    &&!(span.x<bitcast<f32>(svoBrickCoverageCandidates[seedIndex]))){return 0u;}
`;
  const sceneCoverageSeedLimitWGSL = screenSpaceTerminationPixels > 0 ? "" : /* wgsl */ `
  let seedIndex=dryPrimaryCoverageSeedIndex(pixel);
  if(seedIndex<arrayLength(&svoBrickCoverageCandidates)){limit=bitcast<f32>(svoBrickCoverageCandidates[seedIndex]);}
`;
  const traversalCursorWGSL = canonicalTraversal ? /* wgsl */ `
struct DryTraversalCursor{canonical:SvoTraversalContinuation}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){svoTraversalContinuationBegin(ray,mapping,&(*cursor).canonical);}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoTraversalContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);}
` : secondaryTraversalMode === "compact" ? /* wgsl */ `
struct DryTraversalCursor{compact:SvoCompactTraversalContinuation}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){svoCompactContinuationBegin(ray,mapping,&(*cursor).compact);}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoCompactContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).compact);}
` : secondaryTraversalMode === "wide" ? /* wgsl */ `
struct DryTraversalCursor{wide:SvoWideTraversalCursor}
fn dryWidePublication()->SvoWidePublication{return SvoWidePublication(dry.wideFanout.x,dry.wideFanout.y,dry.wideFanout.z,dry.wideFanout.w);}
fn dryCanonicalPublicationGeneration()->u32{return dryPublicationWord(2u);}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){let initialized=svoWideCursorInitialize(&(*cursor).wide,ray,mapping,dryWidePublication(),dryCanonicalPublicationGeneration());if(!initialized){(*cursor).wide.state=SVO_WIDE_CURSOR_INVALID;}}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoWideCursorNext(&(*cursor).wide,ray,mapping,dryDiagnosticMaximumDepth(),dryWidePublication(),dryCanonicalPublicationGeneration());}
` : /* wgsl */ `
struct DryTraversalCursor{canonical:SvoTraversalContinuation,wide:SvoWideTraversalCursor,useWide:u32}
fn dryWidePublication()->SvoWidePublication{return SvoWidePublication(dry.wideFanout.x,dry.wideFanout.y,dry.wideFanout.z,dry.wideFanout.w);}
fn dryCanonicalPublicationGeneration()->u32{return dryPublicationWord(2u);}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){
  (*cursor).useWide=select(0u,1u,svoWideCursorInitialize(&(*cursor).wide,ray,mapping,dryWidePublication(),dryCanonicalPublicationGeneration()));
  if((*cursor).useWide==0u){svoTraversalContinuationBegin(ray,mapping,&(*cursor).canonical);}
}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{
  if((*cursor).useWide!=0u){let wideHit=svoWideCursorNext(&(*cursor).wide,ray,mapping,dryDiagnosticMaximumDepth(),dryWidePublication(),dryCanonicalPublicationGeneration());if(wideHit.status==SVO_STATUS_HIT||wideHit.status==SVO_STATUS_MISS||wideHit.status==SVO_STATUS_WORK_EXHAUSTED){return wideHit;}(*cursor).useWide=0u;if(wideHit.visits>=mapping.maxVisits){return svoMiss(SVO_STATUS_WORK_EXHAUSTED,wideHit.visits);}var fallbackMapping=mapping;fallbackMapping.maxVisits-=wideHit.visits;svoTraversalContinuationBegin(ray,fallbackMapping,&(*cursor).canonical);var fallback=svoTraversalContinuationNext(ray,fallbackMapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);fallback.visits+=wideHit.visits;return fallback;}
  return svoTraversalContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);
}
`;
  const brickOccupancyHelpersWGSL = /* wgsl */ `${svoBrickOccupancyWGSL}
${brickOccupancyMode === "off" ? "" : /* wgsl */ `
const DRY_BRICK_OCCUPANCY_MACRO:u32=${brickOccupancyMode === "macro" ? 1 : 0}u;
fn dryBrickMacroSkip(summary:SvoBrickOccupancy,local:vec3u,bounds:mat2x3f,extent:vec3f,ro:vec3f,rd:vec3f,entry:f32)->vec2f{
  if(DRY_BRICK_OCCUPANCY_MACRO==0u||svoBrickMacroOccupied(summary,local)){return vec2f(0.0,entry);}
  let macroCoord=local>>vec3u(2u);let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let boundaryCell=vec3i(macroCoord*4u)+select(vec3i(0),vec3i(4),step>vec3i(0));
  let boundary=bounds[0]+vec3f(boundaryCell)*extent;
  let next=select(vec3f(3.402823e38),(boundary-ro)/rd,abs(rd)>vec3f(1e-9));
  return vec2f(1.0,max(entry,min(next.x,min(next.y,next.z))));
}
`}`;
  const primaryBrickSetupWGSL = brickOccupancyMode === "off"
    ? /* wgsl */ `let bounds=dryLeafBounds(hit.nodeIndex); let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  var entry=max(hit.tEnter,0.0);let point=ro+rd*(entry+1e-5); var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));`
    : /* wgsl */ `let bounds=dryLeafBounds(hit.nodeIndex);let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  let brickSummary=svoBrickOccupancyDecode(dryLeafFlags(hit.nodeIndex));var brickExit=hit.tExit;var entry=max(hit.tEnter,0.0);
  if(brickSummary.ready!=0u){if(brickSummary.occupied==0u){return missHit();}let occupiedInterval=svoRayAabbWithInverse(SvoRay(ro,entry,rd,brickExit),1.0/rd,svoBrickOccupiedBounds(brickSummary,bounds[0],extent));if(occupiedInterval.x==0.0){return missHit();}entry=max(entry,occupiedInterval.y);brickExit=min(brickExit,occupiedInterval.z);}
  let point=ro+rd*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));`;
  const primaryBrickExitWGSL = brickOccupancyMode === "off" ? "hit.tExit" : "brickExit";
  const primaryMacroSkipWGSL = brickOccupancyMode === "macro" ? /* wgsl */ `let macroSkip=dryBrickMacroSkip(brickSummary,vec3u(cell),bounds,extent,ro,rd,entry);if(macroSkip.x!=0.0){${primaryUpgradeFlushWGSL}if(macroSkip.y>=brickExit||macroSkip.y>=DRY_MISS){break;}entry=macroSkip.y;let skipPoint=ro+rd*(entry+max(1e-5,length(extent)*1e-4));cell=vec3i(clamp(floor((skipPoint-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));let skipBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;nextT=select(vec3f(DRY_MISS),(skipBoundary-ro)/rd,abs(rd)>vec3f(1e-9));continue;}
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
  // Where the exact tier becomes adaptive.
  //
  // Routing the LOD by *pass* — sub-threshold pixels to the proxy fragment,
  // the rest to the exact one — only ever offered two granularities, and once
  // the whole-brick predicate was corrected to measure the whole brick it
  // stopped firing on anything but far background. The intermediate rungs
  // therefore have to live inside the exact walk, decided once per brick
  // fragment from the projected footprint. That is also the cheapest place for
  // it: `bounds` and the occupancy word are already in hand.
  //
  // Stride one leaves the loop below untouched, byte for byte, which is what
  // keeps a zero threshold the reference image rather than a near-miss of it.
  const primaryLodStrideWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  // Capped one rung above the whole brick, and that cap is not a taste call: an
  // aggregate the size of the brick still scans every cell the ray crosses to
  // find a material, so it pays the exact walk's cost and then discards the
  // precision it bought. The whole-brick answer is cheap only where it can be
  // given without touching the payload — the coverage pass, which returns the
  // proxy face and a hardware depth and stops. Here it is a pure loss.
  let lodStride=min(dryLodCellStride(bounds,dryLeafLevel(hit.nodeIndex)),max(dry.mapping.brickSize>>1u,1u));
  if(lodStride>1u){
    return dryPrimaryLeafAggregateHit(ro,rd,bounds,entry,${primaryBrickExitWGSL},hit.voxelOffset,
      svoBrickOccupancyDecode(dryLeafFlags(hit.nodeIndex)),lodStride);
  }` : "";
  const primaryLeafTraceCallWGSL = brickOccupancyMode === "macro-hdda" ? "traceLeafPayloadMacroHdda" : "traceLeafPayload";
  const shadowLeafTraceCallWGSL = brickOccupancyMode === "macro-hdda" ? "traceLeafPayloadVisibilityMacroHdda" : "traceLeafPayloadVisibility";
  const macroHddaPrimaryWGSL = brickOccupancyMode === "macro-hdda" ? /* wgsl */ `
fn traceLeafPayloadFineInterval(ro:vec3f,rd:vec3f,hit:SvoTraversalHit,bounds:mat2x3f,extent:vec3f,intervalEnter:f32,intervalExit:f32,cellMinimum:vec3u,cellMaximum:vec3u)->DryHit{
  var entry=max(intervalEnter,0.0);let point=ro+rd*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum-vec3u(1u))));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));let tolerance=length(extent)*1.05;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(cellMinimum))||any(cell>=vec3i(cellMaximum))||entry>intervalExit){break;}
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex<arrayLength(&materialOwners)){let identity=materialOwners[payloadIndex];let owner=identity>>16u;if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){let primitiveIndex=owner-dry.metadata.y;if(primitiveIndex<dry.metadata.x){let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,intervalExit));let candidate=primitiveHit(dryPrimitive(primitiveIndex),ro,rd,max(0.0,entry-tolerance),cellExit+tolerance);if(candidate.t<DRY_MISS){return candidate;}}}}
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
    if(owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){let primitiveIndex=owner-dry.metadata.y;if(primitiveIndex>=dry.metadata.x){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,min(intervalExit,ray.tMax_m)));let candidate=primitiveHit(dryPrimitive(primitiveIndex),ray.origin_m,ray.direction,max(entry-tolerance,tMin_m),cellExit+tolerance);if(candidate.t<DRY_MISS){return dryVisibilityStep(SVO_VIS_STEP_HIT,0u,0u,workItems,candidate.t);}}
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
var<private> dryPrepassExactEdgeState:u32;
var<private> dryCurrentLightSlot:u32;
const DRY_PREPASS_INVALID_PACKED:vec2u=vec2u(0xffffffffu,0xfffffffeu);
// Retained only by the retired, unreachable exact-refinement shader entries
// until their source block is removed; no production pipeline compiles them.
const DRY_SILHOUETTE_STATE_UNTOUCHED:u32=0u;
const DRY_SILHOUETTE_STATE_VALID:u32=1u;
const DRY_SILHOUETTE_STATE_FAILED:u32=2u;
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
fn dryPrepassReceiverCompatible(identity:u32,metadata:u32,hit:DryHit)->bool{
  let materialMatches=(identity&0xffffu)==(hit.materialId&0xffffu);
  let ownerMatches=(identity>>16u)==(hit.ownerId&0xffffu);
  // Static authored surfaces with the same complete shading classification may
  // share a nearby receiver across object seams. Motion keeps exact ownership:
  // its current-frame rigid blocker correction and GI neighbourhood are owned.
  return materialMatches&&metadata==dryPrepassHitMetadata(hit)&&(hit.motionKind==DRY_GBUFFER_MOTION_STATIC||ownerMatches);
}
${edgeReceiverRecovery ? /* wgsl */ `fn dryPrepassUseExactReceiver(texel:vec2i,depth:f32,normal:vec3f,hit:DryHit)->bool{
  let geometry=textureLoad(dryPrepassGeometryTexture,texel,0);if(geometry.x<=0.0){return false;}
  if(!dryPrepassReceiverCompatible(textureLoad(dryPrepassIdentityTexture,texel,0).x,u32(round(geometry.w)),hit)){return false;}
  let depthWeight=exp(-24.0*abs(geometry.x-depth)/max(depth,1e-3));
  let normalWeight=pow(max(dot(normal,dryPrepassDecodeNormal(geometry.yz)),0.0),8.0);
  if(depthWeight<0.25||normalWeight<0.25){return false;}
  let packed=textureLoad(dryPrepassVisibilityKeyTexture,texel,0);if(all(packed.xy==DRY_PREPASS_INVALID_PACKED)){return false;}
  dryPrepassData0=dryPrepassUnpack0(packed);dryPrepassData1=dryPrepassUnpack1(packed);dryPrepassData2=dryPrepassUnpack2(packed);dryPrepassState=1u;
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u){dryPrepassGi=textureLoad(dryPrepassRadianceTexture,texel,0);dryPrepassGiState=1u;}
  if(dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u&&dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u){dryPrepassRadiance=textureLoad(dryPrepassRadianceTexture,texel,0);dryPrepassRadianceState=1u;}
  return true;
}
fn dryPrepassRecoverExactReceiver(coordinate:vec2f,dims:vec2u,depth:f32,normal:vec3f,hit:DryHit)->bool{
  let nearest=vec2i(clamp(round(coordinate),vec2f(0.0),vec2f(dims)-vec2f(1.0)));
  // Two compact Chebyshev rings cover a one-full-pixel feature at every
  // supported reduced rate. The exceptional path has a hard 24-candidate cap.
  for(var radius=1;radius<=2;radius+=1){for(var y=-radius;y<=radius;y+=1){for(var x=-radius;x<=radius;x+=1){
    if(max(abs(x),abs(y))!=radius){continue;}
    let texel=clamp(nearest+vec2i(x,y),vec2i(0),vec2i(dims)-vec2i(1));
    if(dryPrepassUseExactReceiver(texel,depth,normal,hit)){return true;}
  }}}
  return false;
}
` : ""}
fn dryPrepassResolve(pixel:vec2f,depth:f32,normalIn:vec3f,hit:DryHit){
  ${voxelLightCache ? `if((dryVoxelLight.control.w&2u)!=0u){return;}` : ""}
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
    let bilinear=select(1.0-fraction.x,fraction.x,i==1u)*select(1.0-fraction.y,fraction.y,j==1u);
    let depthWeight=exp(-24.0*abs(geometry.x-depth)/max(depth,1e-3));
    let normalWeight=pow(max(dot(normal,dryPrepassDecodeNormal(geometry.yz)),0.0),8.0);
    let identityMatches=dryPrepassReceiverCompatible(textureLoad(dryPrepassIdentityTexture,texel,0).x,u32(round(geometry.w)),hit);
    if(!identityMatches){if(bilinear>1e-6){linearSafe=0u;}continue;}
    if(bilinear>1e-6&&(depthWeight<0.25||normalWeight<0.25)){linearSafe=0u;}
    let packed=textureLoad(dryPrepassVisibilityKeyTexture,texel,0);
    if(all(packed.xy==DRY_PREPASS_INVALID_PACKED)){linearSafe=0u;continue;}
    let guidedWeight=depthWeight*normalWeight;
    let weight=bilinear*select(guidedWeight,1.0,dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u);
    if(weight<=1e-6){continue;}
    accumulated0+=dryPrepassUnpack0(packed)*weight;
    accumulated1+=dryPrepassUnpack1(packed)*weight;
    accumulated2+=dryPrepassUnpack2(packed)*weight;
    if(weight>bestRadianceWeight){bestRadianceWeight=weight;bestRadianceTexel=texel;}
    if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u
      &&(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u
        ||dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u)){accumulatedGi+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;giWeightSum+=weight;}
    if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["joint-bilateral"]}u){accumulatedRadiance+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;radianceWeightSum+=weight;}
    weightSum+=weight;
  }}
  if(weightSum<${SVO_DRY_CONE_PREPASS_CONTRACT.minimumReconstructionWeight}){
    ${edgeReceiverRecovery ? `if(bestRadianceWeight>0.0&&dryPrepassUseExactReceiver(bestRadianceTexel,depth,normal,hit)){return;}
    if(dryPrepassRecoverExactReceiver(coordinate,dims,depth,normal,hit)){return;}` : ""}
    // A sub-prepass-pixel surface has no lawful screen-space receiver. Mark the
    // explicit exact edge tier and let the existing live cone closures run for
    // this pixel. Any unavailable live page still publishes its typed failure.
    dryPrepassExactEdgeState=1u;return;
  }
  dryPrepassData0=accumulated0/weightSum;dryPrepassData1=accumulated1/weightSum;dryPrepassData2=accumulated2/weightSum;dryPrepassState=1u;
  if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u
    ||dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u){
    if(giWeightSum>=${SVO_DRY_CONE_PREPASS_CONTRACT.minimumReconstructionWeight}){dryPrepassGi=accumulatedGi/giWeightSum;dryPrepassGiState=1u;}
    else if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u;}
  }
  if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["gated-linear"]}u&&linearSafe!=0u&&bestRadianceWeight>0.0){
    let reconstructed=textureSampleLevel(dryPrepassRadianceTexture,nodeMipSampler,pixel/max(uniforms.viewport.xy,vec2f(1.0)),0.0);if(reconstructed.a>=0.0){dryPrepassRadiance=reconstructed;dryPrepassRadianceState=1u;}
  }else if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["joint-bilateral"]}u&&radianceWeightSum>1e-6){
    let reconstructed=accumulatedRadiance/radianceWeightSum;if(reconstructed.a>=0.0){dryPrepassRadiance=reconstructed;dryPrepassRadianceState=1u;}
  }else if(dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u&&dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u&&bestRadianceWeight>0.0){let reconstructed=textureLoad(dryPrepassRadianceTexture,bestRadianceTexel,0);if(reconstructed.a>=0.0){dryPrepassRadiance=reconstructed;dryPrepassRadianceState=1u;}}
}
` : "";
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
  const voxelLightCacheGroup = splitGroup + 1;
  const voxelLightCacheWGSL = voxelLightCache ? /* wgsl */ `
// Persistent world-space visibility for directional light slot zero. The
// request bitset is frame-local; the atlas texels survive camera motion and
// carry a 16-bit lighting epoch plus the representative surface normal.
struct DryVoxelLightCacheParams{control:vec4u,atlasPages:vec4u}
struct DryVoxelLightAddress{atlas:vec3u,voxelIndex:u32,valid:u32}
struct DryVoxelLightQueue{
  distinct:atomic<u32>,misses:atomic<u32>,hits:atomic<u32>,queued:atomic<u32>,
  populated:atomic<u32>,rejected:atomic<u32>,overflow:atomic<u32>,reserved:atomic<u32>,
  entries:array<vec2u>,
}
@group(${voxelLightCacheGroup}) @binding(0) var dryVoxelLightCacheRead:texture_3d<u32>;
@group(${voxelLightCacheGroup}) @binding(1) var<uniform> dryVoxelLight:DryVoxelLightCacheParams;
@group(${voxelLightCacheGroup}) @binding(2) var dryVoxelLightCacheWrite:texture_storage_3d<rg32uint,write>;
@group(${voxelLightCacheGroup}) @binding(3) var<storage,read_write> dryVoxelLightRequests:array<atomic<u32>>;
@group(${voxelLightCacheGroup}) @binding(4) var<storage,read_write> dryVoxelLightQueue:DryVoxelLightQueue;
@group(${voxelLightCacheGroup}) @binding(5) var<storage,read> dryVoxelLightPages:array<vec4u>;
${reduced ? "" : "var<private> dryCurrentLightSlot:u32;"}
var<private> dryVoxelLightConsumerEligible:u32;
fn dryVoxelLightPackNormal(normalIn:vec3f)->u32{
  let normal=normalize(normalIn);var oct=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));
  if(normal.z<0.0){oct=(vec2f(1.0)-abs(oct.yx))*select(vec2f(-1.0),vec2f(1.0),oct>=vec2f(0.0));}
  let encoded=vec2u(round(clamp(oct*.5+.5,vec2f(0.0),vec2f(1.0))*255.0));return encoded.x|(encoded.y<<8u);
}
fn dryVoxelLightUnpackNormal(packed:u32)->vec3f{
  let oct=vec2f(f32(packed&255u),f32((packed>>8u)&255u))/255.0*2.0-1.0;
  var normal=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));if(normal.z<0.0){let folded=(vec2f(1.0)-abs(normal.yx))*select(vec2f(-1.0),vec2f(1.0),normal.xy>=vec2f(0.0));normal=vec3f(folded,normal.z);}return normalize(normal);
}
// The cache is one entry per texel of a *base* page, and the base is the
// opacity floor — the pyramid's finest resident level, not level zero. Under a
// floor of one its entries cover 2^3 finest voxels each, which is the same
// coarsening the opacity field it caches visibility against already took. At
// floor zero every line below is the arithmetic that shipped.
fn dryVoxelLightAddress(position:vec3f)->DryVoxelLightAddress{
  if((dryVoxelLight.control.w&1u)==0u||dry.nodeMipDirect.w==0u){return DryVoxelLightAddress(vec3u(0u),0u,0u);}
  let baseLevel=dryNodeMipOpacityLevelFloor();
  let virtualVoxelFloor=floor((position-dry.nodeMipOrigin.xyz)/max(dry.mapping.cellSize*exp2(f32(baseLevel)),vec3f(1e-6)));
  if(any(virtualVoxelFloor<vec3f(0.0))){return DryVoxelLightAddress(vec3u(0u),0u,0u);}
  let voxel=vec3u(virtualVoxelFloor);let pageCoordinate=voxel/${SVO_NODE_MIP_LAYOUT.interiorSize}u;let pageIndex=dryNodeMipFind(baseLevel,pageCoordinate);
  if(pageIndex==0xffffffffu||pageIndex>=dryVoxelLight.control.x||!dryNodeMipPageValid(pageIndex)){return DryVoxelLightAddress(vec3u(0u),0u,0u);}
  let local=voxel-pageCoordinate*${SVO_NODE_MIP_LAYOUT.interiorSize}u;
  let localIndex=local.x+local.y*${SVO_NODE_MIP_LAYOUT.interiorSize}u+local.z*${SVO_NODE_MIP_LAYOUT.interiorSize ** 2}u;
  let atlasPages=max(dryVoxelLight.atlasPages.xyz,vec3u(1u));
  let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));
  return DryVoxelLightAddress(atlasPage*${SVO_NODE_MIP_LAYOUT.physicalSize}u+vec3u(${SVO_NODE_MIP_LAYOUT.apron}u)+local,pageIndex*${SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.voxelsPerPage}u+localIndex,1u);
}
fn dryVoxelLightVisibility(position:vec3f,normal:vec3f)->vec2f{
  if(dryCurrentLightSlot!=0u||dryVoxelLightConsumerEligible==0u){return vec2f(0.0);}
  let address=dryVoxelLightAddress(position);if(address.valid==0u){return vec2f(0.0);}
  let encoded=textureLoad(dryVoxelLightCacheRead,vec3i(address.atlas),0).xy;
  if(encoded.x==0u||(encoded.y>>16u)!=(dryVoxelLight.control.z&0xffffu)){return vec2f(0.0);}
  if(dot(normalize(normal),dryVoxelLightUnpackNormal(encoded.y&0xffffu))<.9){return vec2f(0.0);}
  return vec2f(f32(encoded.x-1u)/65534.0,1.0);
}
@compute @workgroup_size(8,8) fn dryVoxelLightDemandMain(@builtin(global_invocation_id) id:vec3u){
  let dimensions=textureDimensions(drySplitGeometryRead);if(any(id.xy>=dimensions)){return;}
  let coordinate=vec2i(id.xy);let geometry=drySplitGeometryAt(coordinate);if(!(geometry.w>0.0&&geometry.w<DRY_MISS)){return;}
  let metadata=drySplitIdentityAt(coordinate).y;let motionKind=(metadata>>24u)&3u;let feature=(metadata>>16u)&15u;
  if(motionKind!=DRY_GBUFFER_MOTION_STATIC){atomicAdd(&dryVoxelLightQueue.rejected,1u);return;}
  let uv=vec2f((f32(id.x)+.5)/f32(dimensions.x),1.0-(f32(id.y)+.5)/f32(dimensions.y));let ndc=uv*2.0-1.0;
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());let position=ro+rd*geometry.w;
  let address=dryVoxelLightAddress(position);if(address.valid==0u){atomicAdd(&dryVoxelLightQueue.rejected,1u);return;}
  let word=address.voxelIndex>>5u;let bit=1u<<(address.voxelIndex&31u);let previous=atomicOr(&dryVoxelLightRequests[word],bit);if((previous&bit)!=0u){return;}
  atomicAdd(&dryVoxelLightQueue.distinct,1u);let packedNormal=dryVoxelLightPackNormal(geometry.xyz);
  let encoded=textureLoad(dryVoxelLightCacheRead,vec3i(address.atlas),0).xy;
  if(encoded.x==0u&&(encoded.y>>16u)==(dryVoxelLight.control.z&0xffffu)&&(encoded.y&0xffffu)==0xffffu){atomicAdd(&dryVoxelLightQueue.rejected,1u);return;}
  if(encoded.x!=0u&&(encoded.y>>16u)==(dryVoxelLight.control.z&0xffffu)){
    if(dot(normalize(geometry.xyz),dryVoxelLightUnpackNormal(encoded.y&0xffffu))>=.9){atomicAdd(&dryVoxelLightQueue.hits,1u);}else{atomicAdd(&dryVoxelLightQueue.rejected,1u);}return;
  }
  atomicAdd(&dryVoxelLightQueue.misses,1u);let queueIndex=atomicAdd(&dryVoxelLightQueue.queued,1u);
  if(queueIndex<dryVoxelLight.control.y){dryVoxelLightQueue.entries[queueIndex]=vec2u(address.voxelIndex,packedNormal);}else{atomicAdd(&dryVoxelLightQueue.overflow,1u);}
}
fn dryVoxelLightReject(pageIndex:u32,local:vec3u){
  let atlasPages=max(dryVoxelLight.atlasPages.xyz,vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));
  let atlas=atlasPage*${SVO_NODE_MIP_LAYOUT.physicalSize}u+vec3u(${SVO_NODE_MIP_LAYOUT.apron}u)+local;
  textureStore(dryVoxelLightCacheWrite,vec3i(atlas),vec4u(0u,((dryVoxelLight.control.z&0xffffu)<<16u)|0xffffu,0u,0u));atomicAdd(&dryVoxelLightQueue.rejected,1u);
}
@compute @workgroup_size(64) fn dryVoxelLightPopulateMain(@builtin(global_invocation_id) id:vec3u){
  let count=min(atomicLoad(&dryVoxelLightQueue.queued),dryVoxelLight.control.y);if(id.x>=count){return;}
  let entry=dryVoxelLightQueue.entries[id.x];let pageIndex=entry.x/${SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.voxelsPerPage}u;
  let localIndex=entry.x-pageIndex*${SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.voxelsPerPage}u;if(pageIndex>=dryVoxelLight.control.x){return;}
  let local=vec3u(localIndex%${SVO_NODE_MIP_LAYOUT.interiorSize}u,(localIndex/${SVO_NODE_MIP_LAYOUT.interiorSize}u)%${SVO_NODE_MIP_LAYOUT.interiorSize}u,localIndex/${SVO_NODE_MIP_LAYOUT.interiorSize ** 2}u);
  // Only base-level pages carry cache entries, and the base is the opacity
  // floor; a coarser page's texels are not the lattice the addresses above use.
  let baseLevel=dryNodeMipOpacityLevelFloor();
  let page=dryVoxelLightPages[pageIndex];if(page.w!=baseLevel){return;}let worldVoxel=page.xyz*${SVO_NODE_MIP_LAYOUT.interiorSize}u+local;
  let position=dry.nodeMipOrigin.xyz+(vec3f(worldVoxel)+vec3f(.5))*dry.mapping.cellSize*exp2(f32(baseLevel));let normal=dryVoxelLightUnpackNormal(entry.y);
  var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);let coverage=dryNodeMipAt(position,0.0,&pageCache);
  if(coverage.valid==0u||(coverage.sample.solidMaximum>.85&&coverage.sample.solidMean<.08)){dryVoxelLightReject(pageIndex,local);return;}
  let light=dryLighting.lights[0];if(dryLighting.metadata.x==0u||light.identity.x!=SVO_LIGHT_DIRECTIONAL||light.identity.w!=dryLighting.metadata.y){return;}
  let sample=dryLightSample(light,0u,position);var visibility=0.0;
  if(sample.valid!=0u&&dot(normal,sample.towardLight)>0.0){let maximumDistance=directionalLightSceneExitDistance(position,sample.towardLight);let ray=dryBiasedVisibilityRayUnit(position,normal,sample.towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);let cell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let escape=cell*dry.tuningRays1.z;let coneMaximum=max(0.0,ray.tMax_m-escape*dot(normal,sample.towardLight));let cone=dryConeVisibility(ray.origin_m+normal*escape,sample.towardLight,dry.tuningRays1.y,coneMaximum,normal,false);if(cone.valid==0u){dryVoxelLightReject(pageIndex,local);return;}visibility=cone.transmittance;}
  let atlasPages=max(dryVoxelLight.atlasPages.xyz,vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));
  let atlas=atlasPage*${SVO_NODE_MIP_LAYOUT.physicalSize}u+vec3u(${SVO_NODE_MIP_LAYOUT.apron}u)+local;let packed=u32(round(clamp(visibility,0.0,1.0)*65534.0))+1u;
  textureStore(dryVoxelLightCacheWrite,vec3i(atlas),vec4u(packed,((dryVoxelLight.control.z&0xffffu)<<16u)|(entry.y&0xffffu),0u,0u));atomicAdd(&dryVoxelLightQueue.populated,1u);
}
` : "";
  const voxelLightCacheShortcutWGSL = voxelLightCache ? /* wgsl */ `let cachedVoxelVisibility=dryVoxelLightVisibility(position,geometricNormal);if(cachedVoxelVisibility.y>0.0){let rigidBlocker=nearestBodyIgnoring(ray.origin_m,towardLight,ownerId);let raw=select(cachedVoxelVisibility.x,0.0,rigidBlocker.t<ray.tMax_m);return vec3f(mix(1.0,raw,dry.tuningRays0.y));}if(dryCurrentLightSlot==0u&&(dryVoxelLight.control.w&2u)!=0u){dryCurrentLightSlot=0xffffffffu;}` : "";
  const prepassResolveCallWGSL = reduced
    ? /* wgsl */ `dryPrepassData0=vec4f(1.0);dryPrepassData1=vec4f(1.0);dryPrepassData2=vec4f(1.0);dryPrepassRadiance=vec4f(0.0);dryPrepassGi=vec4f(0.0,0.0,0.0,1.0);dryPrepassState=0u;dryPrepassRadianceState=0u;dryPrepassGiState=0u;dryPrepassExactEdgeState=0u;dryCurrentLightSlot=0xffffffffu;if(opaque.t<DRY_MISS&&(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u){if(dryNodeMipReady()){dryPrepassResolve(input.position.xy,opaque.t,opaque.normal,opaque);}else{dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u;}}${voxelLightCache ? "if((dryVoxelLight.control.w&2u)!=0u){dryPrepassRadianceState=0u;dryPrepassGiState=0u;}" : ""}`
    : "";
  const prepassShadowShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassState==1u&&dryCurrentLightSlot<${SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights}u){let prepassRigidBlocked=anyBodyBlockerIgnoring(ray.origin_m,towardLight,ownerId,ray.tMax_m);let raw=select(dryPrepassChannel(1u+dryCurrentLightSlot),0.0,prepassRigidBlocked);return vec3f(mix(1.0,raw,dry.tuningRays0.y));}`
    : "";
  const prepassContactShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassState==1u){let prepassRadius=dryContactVisibilityRadius();if(prepassRadius<=0.0){return vec3f(1.0);}let prepassCell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let prepassOrigin=position+normalize(geometricNormal)*prepassCell*.2;let prepassSamples=select(dry.tuningCounts1.z,dry.tuningCounts1.y,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL});var prepassUnblocked=0.0;for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=prepassSamples){break;}let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);let rotated=select(direction,normalize(direction+cross(normalize(geometricNormal),direction)*.7),sampleIndex>=2u);let prepassRigidBlocked=anyBodyBlockerIgnoring(prepassOrigin,rotated,ownerId,prepassRadius);prepassUnblocked+=select(1.0,0.0,prepassRigidBlocked);}let raw=clamp(dryPrepassData0.x*(prepassUnblocked/f32(prepassSamples)),0.0,1.0);return vec3f(mix(1.0,raw,dry.tuningRays0.w));}`
    : "";
  const prepassBodyBlockerWGSL = reduced ? /* wgsl */ `fn anyBodyBlockerIgnoring(ro:vec3f,rd:vec3f,ignoredOwner:u32,tMax:f32)->bool {
  // Every shaded pixel calls this once per light and once per contact sample, so
  // at 1500x1500 the body loop runs tens of millions of times a frame and its
  // per-iteration read of bodies[] dominates. One sphere enclosing the whole set
  // rejects the overwhelming majority of those rays before a single body is
  // fetched: rigid bodies occupy a small part of a scene, and contact rays are
  // short. The bound is published per frame, so it costs nothing to consult.
  if(!svoRigidBoundsIntersect(ro,rd,tMax)){return false;}
  for(var index=0u;index<12u;index+=1u){if(index>=u32(round(uniforms.options.z))){break;}if(index==ignoredOwner){continue;}let body=bodies[index];if(!bodyBoundingSphereVisible(ro,rd,body,0.0,tMax)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ro,rd,body,0.0,tMax)){continue;}if(bodyHit(ro,rd,body).t<tMax){return true;}}
  return false;
}
` : "";
  const prepassLightSlotWGSL = reduced || voxelLightCache ? /* wgsl */ `dryCurrentLightSlot=lightIndex;` : "";
  // The rim is re-applied on the cached path too: reduced-rate radiance is a
  // cache of how the room lights this surface, and the cursor is not part of
  // how the room is lit. Skipping it here would make the outline flicker at
  // whatever rate the prepass refreshes.
  const prepassRadianceShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassRadianceState==1u&&hit.motionKind==DRY_GBUFFER_MOTION_STATIC){return dryHoverRim(max(dryPrepassRadiance.rgb,vec3f(0.0)),hit,normalize(-rd));}`
    : "";
  const prepassGiShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassGiState==1u){if(dryPrepassGi.a<0.0){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}return DryGlobalIllumination(max(dryPrepassGi.rgb,vec3f(0.0)),clamp(dryPrepassGi.a,0.0,1.0),1u);}`
    : "";
  const splitVisibilityGlassDiscoveryWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let coordinate=vec2i(input.position.xy);textureStore(drySplitGeometryWrite,coordinate,vec4f(opaque.normal,opaque.t));let opaqueMetadata=(opaque.ownerId&0xffffu)|((opaque.featureId&15u)<<16u)|((opaque.fieldSource&15u)<<20u)|((opaque.motionKind&3u)<<24u)|((opaque.motionValid&1u)<<26u);textureStore(drySplitOpaqueIdentityWrite,coordinate,vec4u(opaque.materialId,opaqueMetadata,0u,0u));let generation=dryPublicationGeneration();`
    : /* wgsl */ `let glass=traceGlass(ro,rd,0.0,opaque.t,true);let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;let coordinate=vec2i(input.position.xy);textureStore(drySplitGeometryWrite,coordinate,vec4f(opaque.normal,opaque.t));let opaqueMetadata=(opaque.ownerId&0xffffu)|((opaque.featureId&15u)<<16u)|((opaque.fieldSource&15u)<<20u)|((opaque.motionKind&3u)<<24u)|((opaque.motionValid&1u)<<26u);let glassKey=select(0u,glass.recordIndex+1u,glassVisible);let packedOpaqueMaterial=(opaque.materialId&0x8000ffffu)|((glassKey&0x1ffu)<<16u);textureStore(drySplitOpaqueIdentityWrite,coordinate,vec4u(packedOpaqueMaterial,opaqueMetadata,0u,0u));let generation=dryPublicationGeneration();`;
  const splitVisibilityGlassReturnWGSL = rasterGlassDiscovery
    ? ""
    : /* wgsl */ `if(glassVisible){let record=dryGlassPane(glass.recordIndex);let media=dryMediumPair(rd,glass.hit.geometricNormal,DRY_MEDIUM_GLASS);let targets=svoGBufferSurface(vec3f(0.0),glass.hit.t_m,glass.hit.geometricNormal,glass.hit.geometricNormal,vec4u(svoThinGlassMaterialId(record),svoThinGlassOwnerId(record),media.x,media.y),vec3f(0.0),DRY_GBUFFER_MOTION_STATIC,DRY_GBUFFER_FIELD_ANALYTIC,generation,SVO_GBUFFER_MOTION_VALID|svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_GLASS),SVO_FEATURE_SMOOTH);return drySplitVisibilityOut(targets,dryHardwareDepth(glass.hit.t_m,rd,forward));}`;
  const splitOpaqueMaterialDecodeWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let opaqueMaterial=packedOpaqueMaterial;`
    : /* wgsl */ `let opaqueMaterial=select(packedOpaqueMaterial&0xffffu,0x80000000u|(packedOpaqueMaterial&0xffffu),(packedOpaqueMaterial&0x80000000u)!=0u);`;
  const splitGlassKeyLoadWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let glassKey=textureLoad(drySplitGlassKeyRead,coordinate,0).x;`
    : /* wgsl */ `let glassKey=(packedOpaqueMaterial>>16u)&0x1ffu;`;
  const splitPrimaryTraceWGSL = rasterRigidDiscovery ? "traceStaticSolidScene(ro,rd)" : "traceOpaqueScene(ro,rd)";
  const rasterPrimaryEntryWGSL = rasterPrimary ? /* wgsl */ `
// Raster-assisted primary visibility. Every plane the deferred lighting pass
// consumes is a depth-tested colour attachment here, so no pass in this graph
// writes primary geometry through an untested storage texture.
@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding}) var<storage,read> svoBrickInstances:array<SvoBrickInstance>;
@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.coverageCountBinding}) var<storage,read_write> svoBrickCoverageCounts:array<atomic<u32>>;
@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.coverageCandidateBinding}) var<storage,read_write> svoBrickCoverageCandidates:array<u32>;
${screenSpaceTerminationPixels > 0 ? `@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.primaryGeometryBinding}) var dryRasterPrimaryGeometryRead:texture_2d<f32>;` : ""}
${screenSpaceTerminationPixels > 0 ? `@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.primaryIdentityBinding}) var dryRasterPrimaryIdentityRead:texture_2d<u32>;` : ""}
${svoScenePrimitiveBandReadWGSL(splitGroup + 1, SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel)}
${svoRasterCoverageOverflowSignalWGSL}
// Pixels the count buffer actually addresses.
//
// The allocation now carries an eight-word tail past the pixel range — the
// overflow flag and the indirect draw args — so arrayLength stopped being a
// pixel bound. It is still a *safe* one (every guard using it is permissive by
// exactly the tail), but a fragment that read the tail as a pixel would corrupt
// the draw the overflow pass is about to issue, so the bound is explicit.
fn dryCoveragePixelLimit()->u32{
  let words=arrayLength(&svoBrickCoverageCounts);
  if(words<=SVO_RASTER_COVERAGE_TAIL_WORDS){return 0u;}
  return min(words-SVO_RASTER_COVERAGE_TAIL_WORDS,
    max(u32(uniforms.viewport.x),1u)*max(u32(uniforms.viewport.y),1u));
}
${svoBrickRasterSharedWGSL}
struct DryRasterPrimaryOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,
  @location(3) opaqueIdentity:vec2u,
  @builtin(frag_depth) hardwareDepth:f32,
}
struct SvoBrickRasterVertexOut{
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) proxyMinimum:vec3f,
  @location(1) @interpolate(flat) proxyMaximum:vec3f,
  @location(2) @interpolate(flat) nodeIndex:u32,
  @location(3) @interpolate(flat) voxelOffset:u32,
  @location(4) @interpolate(flat) instanceIndex:u32,
}
fn dryRasterPrimaryReset(){dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;dryThickGlassFailure=0u;}
fn dryRasterPrimaryCamera()->mat4x3f{
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));return mat4x3f(ro,forward,right,normalize(cross(right,forward)));
}
fn dryRasterPrimaryRay(pixel:vec2f,camera:mat4x3f)->vec3f{
  let uv=vec2f(pixel.x/max(uniforms.viewport.x,1.0),1.0-pixel.y/max(uniforms.viewport.y,1.0));let ndc=uv*2.0-1.0;
  return normalize(camera[1]+camera[2]*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+camera[3]*ndc.y*cameraTanHalfFov());
}
fn dryRasterPrimarySurface(opaque:DryHit,ro:vec3f,rd:vec3f,forward:vec3f,producer:u32)->DryRasterPrimaryOut{
  let generation=dryPublicationGeneration();
  let media=dryMediumPair(rd,opaque.normal,DRY_MEDIUM_OPAQUE);
  let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);
  let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);
  let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);
  let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);
  var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u)|svoGBufferProducerFlags(producer);
  if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}
  let targets=svoGBufferSurface(vec3f(0.0),opaque.t,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);
  let opaqueMetadata=(opaque.ownerId&0xffffu)|((opaque.featureId&15u)<<16u)|((opaque.fieldSource&15u)<<20u)|((opaque.motionKind&3u)<<24u)|((opaque.motionValid&1u)<<26u);
  return DryRasterPrimaryOut(targets.packedSurface,targets.identityMedia,vec4f(opaque.normal,opaque.t),vec2u(opaque.materialId,opaqueMetadata),dryHardwareDepth(opaque.t,rd,forward));
}
fn dryRasterPrimaryMiss()->DryRasterPrimaryOut{
  let targets=svoGBufferMiss(vec3f(0.0),0u,dryPublicationGeneration(),DRY_GBUFFER_NO_INTERSECTION,
    svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND));
  return DryRasterPrimaryOut(targets.packedSurface,targets.identityMedia,vec4f(0.0,1.0,0.0,DRY_MISS),vec2u(0u,0u),0.0);
}
// The brick payload stores one owner per voxel. That is sufficient to find a
// nearby surface, but it cannot represent a sub-voxel visibility boundary
// between two projected primitives. Raster one conservative box per authored
// finite primitive and repeat the shared exact intersection in the fragment;
// reversed-Z then chooses the true nearest surface across all owners.
struct DryScenePrimitiveVertexOut{
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) primitiveIndex:u32,
}
// The ray's span through the primitive's own oriented box, as the march bracket.
//
// primitiveHit brackets a marched kind by its *bounding sphere*, and the ABI
// states the assumption that makes 48 iterations sufficient: "callers hand in a
// bounded interval - one traversed voxel in the renderer - so the march starts
// within a cell of the surface". This pass was handing in 0 .. DRY_MISS, so
// that assumption did not hold on the path carrying 78% of the frame. For a
// round cone the bounding sphere is halfLength + radius, many times the
// thickness the ray actually crosses, and every step of that excess is a full
// SDF evaluation.
//
// The local box is the same containment the raster proxy already relies on, so
// a hit outside this span does not exist. Returns x > y when the ray misses the
// box, which is a hit the exact test cannot produce either.
//
// The span is padded because containment is exact in real arithmetic and not in
// floating point: the exact solve puts a grazing cone hit up to 1.3e-5 *before*
// its own box entry (tests/svo-scene-primitive-march-span.test.ts finds it on
// the authored hero set), and an unpadded bracket would drop that sliver of
// silhouette on the one path carrying most of the frame. A grazing ray converts
// a tiny spatial error into a much larger one in t, so the pad is relative to
// the span rather than absolute. Widening the bracket cannot introduce a hit —
// the exact test still decides — it only declines to save a step or two.
fn dryScenePrimitiveMarchSpan(record:SvoPrimitiveRecord,ro:vec3f,rd:vec3f)->vec2f{
  let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let orientationLength=length(record.orientation);
  if(any(localExtent<vec3f(0.0))||!(orientationLength>1e-8)){return vec2f(0.0,DRY_MISS);}
  let inverse=vec4f(-record.orientation.xyz,record.orientation.w)/orientationLength;
  let localOrigin=svoQuaternionRotate(inverse,ro-svoPrimitiveCenter_m(record));
  let localDirection=svoQuaternionRotate(inverse,rd);
  let span=svoRayAabbWithInverse(SvoRay(localOrigin,0.0,localDirection,DRY_MISS),
    1.0/localDirection,mat2x3f(-localExtent,localExtent));
  if(span.x==0.0){return vec2f(DRY_MISS,0.0);}
  let pad=max(${SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_M},${SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_RELATIVE}*span.z);
  return vec2f(max(0.0,span.y-pad),span.z+pad);
}
fn dryScenePrimitiveWorldExtent(localExtent:vec3f,orientation:vec4f)->vec3f{
  let q=orientation/length(orientation);
  return abs(svoQuaternionRotate(q,vec3f(localExtent.x,0.0,0.0)))
    +abs(svoQuaternionRotate(q,vec3f(0.0,localExtent.y,0.0)))
    +abs(svoQuaternionRotate(q,vec3f(0.0,0.0,localExtent.z)))+vec3f(1e-5);
}
// Tightening this proxy to the primitive's own oriented box was measured and
// does not pay, which is worth stating because the coverage argument for it is
// correct and still tempting: a hose spelled as 336 diagonal round cones has an
// axis-aligned box several times the shape, and the oriented box halves mean
// covering proxies per pixel over the hero framing (16.1 to 7.7, median 15 to
// 3). Interleaved and paired at 1600x1240 that bought -5.2, +3.7 and -24.1 ms
// against a control arm whose own spread was 356.7 to 385.8 — no effect, and
// the sign is not even stable. Fragment *count* is not what this pass is bound
// by; per-fragment march cost is, and the proxies the oriented box shrinks are
// the cheap round cones rather than the tapered-sweep clusters beside them.
fn dryScenePrimitiveProxyVertex(vertexIndex:u32,primitiveIndex:u32)->DryScenePrimitiveVertexOut{
  var position=vec4f(2.0,2.0,0.0,1.0);
  if(primitiveIndex>=dry.metadata.x){return DryScenePrimitiveVertexOut(position,primitiveIndex);}
  let record=dryPrimitive(primitiveIndex);let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let orientationLength=length(record.orientation);
  if(any(localExtent<vec3f(0.0))||!(orientationLength>1e-8)){return DryScenePrimitiveVertexOut(position,primitiveIndex);}
  let extent=dryScenePrimitiveWorldExtent(localExtent,record.orientation);let centre=svoPrimitiveCenter_m(record);let minimum=centre-extent;let maximum=centre+extent;
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let forward=camera[1];let right=camera[2];let up=camera[3];let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  let margin=vec3f(${4 * SVO_DRY_SCENE_REVERSED_Z_NEAR_M});
  if(all(ro>=minimum-margin)&&all(ro<=maximum+margin)){
    var screen=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(-1.0,3.0),vec2f(3.0,-1.0));
    if(vertexIndex<3u){position=vec4f(screen[vertexIndex],1.0,1.0);}
  }else{
    let world=mix(minimum,maximum,svoBrickBoxCorner(vertexIndex));let relative=world-ro;let viewDepth=dot(relative,forward);
    position=vec4f(dot(relative,right)/(aspect*cameraTanHalfFov()),dot(relative,up)/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,viewDepth);
  }
  return DryScenePrimitiveVertexOut(position,primitiveIndex);
}
// The exact historical set: every authored record gets its proxy. This entry
// point stays band-free on purpose — it feeds the direct fragment, which is the
// overflow fallback and the control the coverage arm is measured against, and a
// control that had already had records removed from it would measure nothing.
@vertex fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.vertex}(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) primitiveIndex:u32,
)->DryScenePrimitiveVertexOut{
  return dryScenePrimitiveProxyVertex(vertexIndex,primitiveIndex);
}
// The banded set, for the coverage and overflow passes.
//
// Rejection happens here rather than in the fragment because this is where it is
// free: all thirty-six vertices collapse onto the same clip-space point, the
// triangles are degenerate, and the rasterizer produces nothing at all — so a
// record the band dropped costs no fragment, no arena slot, and no march. The
// two passes must agree on the set (overflow re-runs the pixels whose coverage
// list was full, and a list built from a different set would be re-marched
// against the wrong geometry), which is exactly why they share this entry point.
@vertex fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.bandVertex}(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) primitiveIndex:u32,
)->DryScenePrimitiveVertexOut{
  if(!dryScenePrimitiveBandMember(primitiveIndex)){
    return DryScenePrimitiveVertexOut(vec4f(2.0,2.0,0.0,1.0),primitiveIndex);
  }
  return dryScenePrimitiveProxyVertex(vertexIndex,primitiveIndex);
}
// Scene-primitive fragments carry their own output struct for the same reason
// the brick ones do: the hidden-surface-removal probe drops frag_depth here
// without disturbing any other pass sharing DryRasterPrimaryOut.
struct DrySceneRasterOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,
  @location(3) opaqueIdentity:vec2u,${scenePrimitiveHsrProbe ? "" : `
  @builtin(frag_depth) hardwareDepth:f32,`}
}
fn drySceneRasterOut(surface:DryRasterPrimaryOut)->DrySceneRasterOut{
  return DrySceneRasterOut(surface.packedSurface,surface.identityMedia,surface.geometry,surface.opaqueIdentity${scenePrimitiveHsrProbe ? "" : ",surface.hardwareDepth"});
}
@fragment fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.fragment}(input:DryScenePrimitiveVertexOut)->DrySceneRasterOut{
  dryRasterPrimaryReset();
${scenePrimitiveHsrProbe
    ? `  var probe=dryRasterPrimaryMiss();
  if(input.primitiveIndex<dry.metadata.x){
    let record=dryPrimitive(input.primitiveIndex);
    if(!dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){
      let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
      ${scenePrimitiveMarchSpanWGSL}
      if(exact.t<DRY_MISS){probe=dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);}
    }
  }
  return drySceneRasterOut(probe);`
    : `  if(input.primitiveIndex>=dry.metadata.x){discard;}
  let record=dryPrimitive(input.primitiveIndex);
  if(dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){discard;}
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  ${scenePrimitiveMarchSpanWGSL}
  if(!(exact.t<DRY_MISS)){discard;}
  return drySceneRasterOut(dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE));`}
}
// Background and terrain. The megakernel's octree stack, rigid loop and pane
// loop are all absent here, which is the register budget this mode buys back.
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.background}(input:VertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let terrain=traceTerrain(camera[0],rd);
  if(terrain.t<DRY_MISS){return dryRasterPrimarySurface(terrain,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND);}
  return dryRasterPrimaryMiss();
}
@vertex fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.vertex}(@builtin(vertex_index) vertexIndex:u32,@builtin(instance_index) instanceIndex:u32)->SvoBrickRasterVertexOut{
  let record=svoBrickInstances[instanceIndex];
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));
  let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  var position=vec4f(0.0,0.0,0.0,1.0);
  // Back faces let a camera inside a brick still shade it, but they clip once
  // the box reaches the near plane. Those few instances cover the screen at the
  // nearest depth instead; the fragment's own box intersection rejects the
  // pixels their proxy does not actually contain.
  let margin=vec3f(${4 * SVO_DRY_SCENE_REVERSED_Z_NEAR_M});
  if(all(ro>=record.proxyMinimum-margin)&&all(ro<=record.proxyMaximum+margin)){
    var screen=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(-1.0,3.0),vec2f(3.0,-1.0));
    if(vertexIndex<3u){position=vec4f(screen[vertexIndex],1.0,1.0);}
  }else{
    let world=mix(record.proxyMinimum,record.proxyMaximum,svoBrickBoxCorner(vertexIndex));
    let relative=world-ro;let viewDepth=dot(relative,forward);
    // Constant clip-space z with w = view depth is exactly the reversed-Z
    // infinite-far projection: the interpolated depth is near/viewDepth.
    position=vec4f(dot(relative,right)/(aspect*cameraTanHalfFov()),dot(relative,up)/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,viewDepth);
  }
  return SvoBrickRasterVertexOut(position,record.proxyMinimum,record.proxyMaximum,record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK,record.voxelOffset,instanceIndex);
}
// Stage one is deliberately coverage-only. Ordinary fragments append their
// candidate and stop immediately; only overlaps beyond the fixed arena write
// the throwaway colour. No path traces a payload or writes fragment depth. The
// expensive resolve below consequently runs exactly once per pixel rather than
// once per overlapping brick.
${brickCoverageOutputWGSL}
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.coverage}(
  input:SvoBrickRasterVertexOut,
 )${brickCoverageReturnTypeWGSL}{
  let width=max(u32(uniforms.viewport.x),1u);let coordinate=vec2u(input.position.xy);
  let pixel=coordinate.y*width+coordinate.x;
  if(pixel>=dryCoveragePixelLimit()){${brickCoverageMissWGSL}}
  ${screenSpaceBrickCoverageWGSL}
  let slot=atomicAdd(&svoBrickCoverageCounts[pixel],1u);
  if(slot<${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u){
    let address=pixel*${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u+slot;
    if(address<arrayLength(&svoBrickCoverageCandidates)){svoBrickCoverageCandidates[address]=input.instanceIndex;}
    discard;
  }
  // The fragment that takes the *first* slot past capacity raises the flag, so
  // the counter counts overflowing pixels rather than surplus fragments — and
  // so a frame with none of them can skip the overflow draw entirely.
  if(slot==${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u){svoRasterCoverageOverflowSignal();}
  ${brickCoverageOverflowWGSL}
}
fn dryBrickCoveragePixel(position:vec2f)->u32{
  return u32(position.y)*max(u32(uniforms.viewport.x),1u)+u32(position.x);
}
// One word per pixel past every candidate slot of the shared arena. Both
// coverage passes address the arena from zero with their own stride, and the
// wider of the two (the scene-primitive capacity) sizes the allocation, so this
// tail is unreachable from either — see svoRasterCoverageArenaBytes.
const DRY_SCENE_COVERAGE_CAPACITY:u32=${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel}u;
const DRY_SCENE_COVERAGE_INDEX_MASK:u32=${SVO_SCENE_PRIMITIVE_COVERAGE_MAXIMUM_RECORDS - 1}u;
fn dryPrimaryCoverageSeedIndex(pixel:u32)->u32{
  return max(u32(uniforms.viewport.x),1u)*max(u32(uniforms.viewport.y),1u)*DRY_SCENE_COVERAGE_CAPACITY+pixel;
}
fn dryBrickCoverageExactHit(position:vec2f,ro:vec3f,rd:vec3f,tLimit:f32)->DryHit{
  let pixel=dryBrickCoveragePixel(position);var opaque=missHit();opaque.t=tLimit;
  if(pixel>=dryCoveragePixelLimit()){return opaque;}
  let count=min(atomicLoad(&svoBrickCoverageCounts[pixel]),${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u);
  let base=pixel*${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u;var visited=0u;var iteration=0u;
  while(iteration<count){
    var bestSlot=0xffffffffu;var bestEntry=DRY_MISS;var bestExit=DRY_MISS;var slot=0u;
    while(slot<count){
      if((visited&(1u<<slot))==0u){
        let address=base+slot;
        if(address<arrayLength(&svoBrickCoverageCandidates)){
          let instanceIndex=svoBrickCoverageCandidates[address];
          if(instanceIndex<arrayLength(&svoBrickInstances)){
            let record=svoBrickInstances[instanceIndex];
            let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,mat2x3f(record.proxyMinimum,record.proxyMaximum));
            let entry=select(DRY_MISS,max(interval.y,0.0),interval.x!=0.0);
            if(entry<bestEntry){bestEntry=entry;bestExit=interval.z;bestSlot=slot;}
          }
        }
      }
      slot+=1u;
    }
    if(bestSlot==0xffffffffu||bestEntry>=opaque.t){break;}visited|=1u<<bestSlot;
    let instanceIndex=svoBrickCoverageCandidates[base+bestSlot];let record=svoBrickInstances[instanceIndex];
    let leaf=SvoTraversalHit(SVO_STATUS_HIT,0u,record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK,0u,record.voxelOffset,0u,bestEntry,bestExit);
    let payload=traceLeafPayload(ro,rd,leaf);
    if(payload.t<DRY_MISS){if(payload.t<opaque.t){opaque=payload;}break;}
    iteration+=1u;
  }
  return opaque;
}
fn dryBrickCoverageResolve(position:vec2f)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  var opaque=traceTerrain(ro,rd);var producer=SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND;
  ${screenSpaceBrickResolveWGSL}
  let pixel=dryBrickCoveragePixel(position);let exact=dryBrickCoverageExactHit(position,ro,rd,opaque.t);
  if(exact.t<opaque.t){opaque=exact;producer=SVO_GBUFFER_PRODUCER_BRICK;}
  // The rigid impostor fold. Raster-primary's background pass traces terrain
  // only, which is the whole reason a scene with any body at all had to keep the
  // twelve-instance impostor pass switched on — and that pass, not the bodies,
  // is what forfeited stationary primary reuse. One analytic body loop inside
  // the resolve that already owns this pixel costs a bounding-sphere reject on
  // every ray that misses the set, and it publishes bodies through exactly the
  // same G-buffer encoder as every other producer here.
  if(svoRigidBoundsIntersect(ro,rd,opaque.t)){
    let rigid=nearestBody(ro,rd);
    if(rigid.t<opaque.t){opaque=rigid;producer=SVO_GBUFFER_PRODUCER_RIGID;}
  }
  // Publish this pixel's nearest opaque for the scene-primitive resolve. One
  // full-screen invocation per pixel writes it exactly once, so the store needs
  // no atomic and races with nothing. The brick overflow arm can still find a
  // nearer brick afterwards, which only makes the seed conservative: it is an
  // upper bound on the true nearest, so it can never cull a winning candidate.
  let seedIndex=dryPrimaryCoverageSeedIndex(pixel);
  if(seedIndex<arrayLength(&svoBrickCoverageCandidates)){svoBrickCoverageCandidates[seedIndex]=bitcast<u32>(opaque.t);}
  if(opaque.t<DRY_MISS){return dryRasterPrimarySurface(opaque,ro,rd,camera[1],producer);}
  return dryRasterPrimaryMiss();
}
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.resolve}(input:VertexOut)->DryRasterPrimaryOut{
  return dryBrickCoverageResolve(input.position.xy);
}
${screenSpaceTerminationPixels > 0 ? /* wgsl */ `
// The production threshold keeps the heavyweight exact marcher out of sky and
// sub-pixel tiles entirely. Entry-point reachability gives each tier its own
// Metal register budget even though the source module remains shared.
@fragment fn svoBrickLodResolveFragment(input:VertexOut)->DryRasterPrimaryOut{
  let position=input.position.xy;let lodKey=textureLoad(drySplitGlassKeyRead,vec2i(position),0).x;
  if(lodKey==0u){discard;}let instanceIndex=lodKey-1u;if(instanceIndex>=arrayLength(&svoBrickInstances)){discard;}
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  let record=svoBrickInstances[instanceIndex];let bounds=mat2x3f(record.proxyMinimum,record.proxyMaximum);
  let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,bounds);if(interval.x==0.0){discard;}
  let proxy=dryPrimaryLeafProxyHit(ro,rd,record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK,bounds,max(interval.y,0.0),interval.z,record.voxelOffset);
  if(!(proxy.t<DRY_MISS)){discard;}return dryRasterPrimarySurface(proxy,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
}
@fragment fn svoBrickExactResolveFragment(input:VertexOut)->DryRasterPrimaryOut{
  let position=input.position.xy;let pixel=dryBrickCoveragePixel(position);
  if(pixel>=dryCoveragePixelLimit()||atomicLoad(&svoBrickCoverageCounts[pixel])==0u){discard;}
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  let exact=dryBrickCoverageExactHit(position,ro,rd,DRY_MISS);if(!(exact.t<DRY_MISS)){discard;}
  return dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
}
` : ""}
// Overflow is correctness-only and isolated in its own proxy entry point. It
// repeats the historical direct fragment only on marked pixels, which makes
// capacity performance-only without importing canonical brick-boundary tie
// arithmetic into the raster arm.
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.overflowResolve}(input:SvoBrickRasterVertexOut)->DryRasterPrimaryOut{
  let pixel=dryBrickCoveragePixel(input.position.xy);
  if(pixel>=dryCoveragePixelLimit()||atomicLoad(&svoBrickCoverageCounts[pixel])<=${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u){discard;}
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,mat2x3f(input.proxyMinimum,input.proxyMaximum));
  if(interval.x==0.0){discard;}
  let leaf=SvoTraversalHit(SVO_STATUS_HIT,0u,input.nodeIndex,0u,input.voxelOffset,0u,max(interval.y,0.0),interval.z);
  let payload=traceLeafPayload(ro,rd,leaf);if(!(payload.t<DRY_MISS)){discard;}
  return dryRasterPrimarySurface(payload,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
}
// ---------------------------------------------------------------------------
// The authored SDF set on the same coverage/resolve/overflow shape as bricks.
//
// The direct fragment above it marches the record's field once per covering
// proxy, and it writes frag_depth *and* discards, so a tiler can reject nothing:
// a covered pixel under a median of seven proxies marches all seven even after
// the nearest one has resolved it. WGSL has no conservative depth to promise
// monotonicity with (gpuweb#5342), so the substitute is the arena the brick
// raster already proves out: append candidates without evaluating any field,
// then walk one per-pixel list front to back exactly once.
// ---------------------------------------------------------------------------
fn dryScenePrimitiveCoverageKey(entry_m:f32,primitiveIndex:u32)->u32{
  return ((bitcast<u32>(max(entry_m,0.0))>>16u)<<16u)|(primitiveIndex&DRY_SCENE_COVERAGE_INDEX_MASK);
}
/** The key's floored entry distance: a lower bound, which is what makes the early-out sound. */
fn dryScenePrimitiveCoverageEntry(key:u32)->f32{return bitcast<f32>(key&0xffff0000u);}
// Coverage evaluates no field. It repeats only the oriented-box bracket the
// resolve will march inside, which is a quaternion rotate and a slab test, and
// that bracket is also the sort key. The three rejections here (retired index,
// suppressed owner, ray misses the local box) are exactly the ones the direct
// fragment discards on, so a candidate that never enters the arena is one the
// historical pass would have thrown away after paying for it.
@fragment fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.coverage}(
  input:DryScenePrimitiveVertexOut,
 )${sceneCoverageReturnTypeWGSL}{
  if(input.primitiveIndex>=dry.metadata.x){${sceneCoverageMissWGSL}}
  let width=max(u32(uniforms.viewport.x),1u);let coordinate=vec2u(input.position.xy);
  let pixel=coordinate.y*width+coordinate.x;
  if(pixel>=dryCoveragePixelLimit()){${sceneCoverageMissWGSL}}
  dryRasterPrimaryReset();
  let record=dryPrimitive(input.primitiveIndex);
  if(dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){${sceneCoverageMissWGSL}}
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let span=dryScenePrimitiveMarchSpan(record,ro,rd);
  if(!(span.x<=span.y)){${sceneCoverageMissWGSL}}
  ${screenSpaceSceneCoverageWGSL}
  // Occlusion, one pass earlier than the resolve applies it.
  //
  // The brick/terrain/body resolve has already published this pixel's nearest
  // opaque, and after the voxel primary that surface is usually an authored
  // record's own exact hit rather than a miss. A proxy whose entry is at or
  // beyond it cannot hold a nearer surface — the identical front-to-back
  // argument the resolve's walk makes — so it is not a candidate at all.
  //
  // Rejecting it here rather than there is what keeps the arena from filling:
  // capacity is per pixel and unconditional, so a candidate the resolve would
  // have skipped in one comparison still consumed a slot, and slots are what the
  // overflow pass re-marches. This is image-exact — a rejected candidate is one
  // whose conservative lower bound is already behind a resolved surface.
  ${sceneCoverageSeedCullWGSL}
  let slot=atomicAdd(&svoBrickCoverageCounts[pixel],1u);
  ${screenSpaceTerminationPixels > 0 ? `if(dryCoveragePixelLimit()>=${1 << 20}u&&slot==0u){let queueIndex=atomicAdd(&dryTieredResolveQueue.pixelCount,1u);if(queueIndex<dryCoveragePixelLimit()){dryTieredResolveQueue.pixels[queueIndex]=pixel;}}` : ""}
  if(slot<DRY_SCENE_COVERAGE_CAPACITY){
    let address=pixel*DRY_SCENE_COVERAGE_CAPACITY+slot;
    if(address<arrayLength(&svoBrickCoverageCandidates)){
      svoBrickCoverageCandidates[address]=dryScenePrimitiveCoverageKey(span.x,input.primitiveIndex);
    }
    discard;
  }
  if(slot==DRY_SCENE_COVERAGE_CAPACITY){svoRasterCoverageOverflowSignal();}
  ${sceneCoverageOverflowWGSL}
}
// One expensive fragment per pixel. Candidates are extracted in increasing key
// order — unique, because back-face culling gives a primitive at most one
// fragment per pixel — so no visited set is needed and the walk is not bounded
// by a 32-bit mask the way the brick resolve's is.
fn dryScenePrimitiveCoverageResolve(position:vec2f,ro:vec3f,rd:vec3f)->DryHit{
  var best=missHit();
  let pixel=dryBrickCoveragePixel(position);
  if(pixel>=dryCoveragePixelLimit()){return best;}
  let appended=atomicLoad(&svoBrickCoverageCounts[pixel]);
  dryScenePrimitiveAuditCoverage(pixel,appended);
  let count=min(appended,DRY_SCENE_COVERAGE_CAPACITY);
  if(count==0u){return best;}
  // Seeded with the surface the brick/terrain/body resolve already published:
  // this is the depth authority for the pixel at this point in the frame, and
  // marching past it is the work the direct pass could never avoid.
  var limit=DRY_MISS;
  ${sceneCoverageSeedLimitWGSL}
  ${screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  // Overflow is the deep tier: it walks the authored set at this pixel instead
  // of retaining a full-MRT overflow render pass in every frame.
  if(appended>DRY_SCENE_COVERAGE_CAPACITY){
    for(var recordIndex=0u;recordIndex<dry.metadata.x;recordIndex+=1u){
      if(!dryScenePrimitiveBandMember(recordIndex)){continue;}
      let record=dryPrimitive(recordIndex);if(dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){continue;}
      let span=dryScenePrimitiveMarchSpan(record,ro,rd);if(!(span.x<=span.y)||!(span.x<limit)){continue;}
      let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
      let worldExtent=dryScenePrimitiveWorldExtent(localExtent,record.orientation);let centre=svoPrimitiveCenter_m(record);
      if(dryPrimaryBoundsSubPixel(mat2x3f(centre-worldExtent,centre+worldExtent))){continue;}
      let exact=primitiveHit(record,ro,rd,span.x,min(span.y,limit));if(exact.t<limit){limit=exact.t;best=exact;}
    }
    return best;
  }` : ""}
  let base=pixel*DRY_SCENE_COVERAGE_CAPACITY;
  var previousKey=0u;var started=false;
  for(var iteration=0u;iteration<DRY_SCENE_COVERAGE_CAPACITY;iteration+=1u){
    if(iteration>=count){break;}
    var nextKey=0xffffffffu;var found=false;
    for(var slot=0u;slot<DRY_SCENE_COVERAGE_CAPACITY;slot+=1u){
      if(slot>=count){break;}
      let key=svoBrickCoverageCandidates[base+slot];
      if((!started||key>previousKey)&&key<nextKey){nextKey=key;found=true;}
    }
    if(!found){break;}
    previousKey=nextKey;started=true;
    // Front-to-back termination. Every remaining candidate enters at or beyond
    // this one, so none of them can hold a nearer surface than the best hit.
    if(!(dryScenePrimitiveCoverageEntry(nextKey)<limit)){break;}
    let record=dryPrimitive(nextKey&DRY_SCENE_COVERAGE_INDEX_MASK);
    let span=dryScenePrimitiveMarchSpan(record,ro,rd);
    ${screenSpacePrimitiveResolveWGSL}
    // Marched only over what is still visible: the far end is clamped to the
    // best hit so far, so a sphere trace never walks behind a resolved surface.
    let exact=primitiveHit(record,ro,rd,span.x,min(span.y,limit));
    if(exact.t<limit){limit=exact.t;best=exact;}
  }
  return best;
}
@fragment fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.resolve}(input:VertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let exact=dryScenePrimitiveCoverageResolve(input.position.xy,ro,rd);
  if(!(exact.t<DRY_MISS)){discard;}
  return dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
}
${screenSpaceTerminationPixels > 0 ? /* wgsl */ `
@compute @workgroup_size(1) fn svoScenePrimitiveTieredComputeArgs(){
  let count=min(atomicLoad(&dryTieredResolveQueue.pixelCount),dryCoveragePixelLimit());
  atomicStore(&dryTieredResolveQueue.dispatchX,(count+63u)/64u);
  atomicStore(&dryTieredResolveQueue.dispatchY,1u);atomicStore(&dryTieredResolveQueue.dispatchZ,1u);
}
@compute @workgroup_size(64) fn svoScenePrimitiveTieredComputeResolve(
  @builtin(global_invocation_id) id:vec3u,
){
  let dimensions=vec2u(max(u32(uniforms.viewport.x),1u),max(u32(uniforms.viewport.y),1u));
  let count=min(atomicLoad(&dryTieredResolveQueue.pixelCount),dryCoveragePixelLimit());if(id.x>=count){return;}
  let pixel=dryTieredResolveQueue.pixels[id.x];let pixelCoordinate=vec2u(pixel%dimensions.x,pixel/dimensions.x);
  let coordinate=vec2i(pixelCoordinate);let position=vec2f(pixelCoordinate)+vec2f(.5);dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  let exact=dryScenePrimitiveCoverageResolve(position,ro,rd);if(!(exact.t<DRY_MISS)){return;}
  let hardwareDepth=dryHardwareDepth(exact.t,rd,camera[1]);
  if(!(hardwareDepth>textureLoad(dryTierCurrentDepth,coordinate,0))){return;}
  let surface=dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
  textureStore(dryTierPackedSurfaceWrite,coordinate,surface.packedSurface);
  textureStore(dryTierIdentityMediaWrite,coordinate,surface.identityMedia);
  textureStore(dryTierGeometryWrite,coordinate,surface.geometry);
  textureStore(dryTierOpaqueIdentityWrite,coordinate,vec4u(surface.opaqueIdentity,0u,0u));
  textureStore(dryTierDepthWrite,coordinate,vec4f(hardwareDepth,0.0,0.0,0.0));
}
@fragment fn svoScenePrimitiveLodResolveFragment(input:VertexOut)->DryRasterPrimaryOut{
  let position=input.position.xy;let key=textureLoad(drySplitGlassKeyRead,vec2i(position),0).x;
  if(key==0u){discard;}let recordIndex=key-1u;if(recordIndex>=dry.metadata.x){discard;}
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  let record=dryPrimitive(recordIndex);let span=dryScenePrimitiveMarchSpan(record,ro,rd);if(!(span.x<=span.y)){discard;}
  let proxy=dryPrimaryPrimitiveProxyHit(record,ro,rd,span);return dryRasterPrimarySurface(proxy,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
}
` : ""}
// Correctness-only, exactly as the brick arm: the historical direct fragment,
// verbatim, on the pixels whose conservative list overflowed. Capacity is
// therefore a performance parameter and never an image change.
@fragment fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.overflowResolve}(
  input:DryScenePrimitiveVertexOut,
)->DryRasterPrimaryOut{
  let pixel=dryBrickCoveragePixel(input.position.xy);
  if(pixel>=dryCoveragePixelLimit()||atomicLoad(&svoBrickCoverageCounts[pixel])<=DRY_SCENE_COVERAGE_CAPACITY){discard;}
  dryRasterPrimaryReset();
  if(input.primitiveIndex>=dry.metadata.x){discard;}
  let record=dryPrimitive(input.primitiveIndex);
  if(dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){discard;}
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  ${scenePrimitiveMarchSpanWGSL}
  if(!(exact.t<DRY_MISS)){discard;}
  return dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
}
// Brick fragments carry their own output struct so the hidden-surface-removal
// probe can drop frag_depth here without disturbing the background pass, whose
// single full-screen fragment gains nothing from HSR either way.
struct DryBrickRasterOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,
  @location(3) opaqueIdentity:vec2u,${noFragmentDepth ? "" : `
  @builtin(frag_depth) hardwareDepth:f32,`}
}
fn dryBrickRasterOut(surface:DryRasterPrimaryOut)->DryBrickRasterOut{
  return DryBrickRasterOut(surface.packedSurface,surface.identityMedia,surface.geometry,surface.opaqueIdentity${noFragmentDepth ? "" : ",surface.hardwareDepth"});
}
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.fragment}(input:SvoBrickRasterVertexOut)->DryBrickRasterOut{
  dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,mat2x3f(input.proxyMinimum,input.proxyMaximum));
  ${hsrProbe ? "" : "if(interval.x==0.0){discard;}"}
  // The payload DDA is the production leaf tracer, unmodified: the raster stage
  // only replaces the search that found this leaf.
  let leaf=SvoTraversalHit(SVO_STATUS_HIT,0u,input.nodeIndex,0u,input.voxelOffset,0u,max(interval.y,0.0),interval.z);
  let payload=traceLeafPayload(ro,rd,leaf);
  ${hsrProbe
    ? `var probe=dryRasterPrimaryMiss();
  if(payload.t<DRY_MISS){probe=dryRasterPrimarySurface(payload,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);}
  return dryBrickRasterOut(probe);`
    : `if(!(payload.t<DRY_MISS)){discard;}
  return dryBrickRasterOut(dryRasterPrimarySurface(payload,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK));`}
}
` : "";
  const recordInvalidAoPageWGSL = reduced && split
    ? "atomicAdd(&dryPrepassBoundaryQueue.invalidAoPages,1u);"
    : "";
  const recordInvalidDirectPageWGSL = reduced && split
    ? "atomicAdd(&dryPrepassBoundaryQueue.invalidDirectPages,1u);"
    : "";
  const prepassEntryWGSL = reduced ? /* wgsl */ `struct DryPrepassGeometryOut{@location(0) geometry:vec4f,@location(1) identity:u32}
@fragment fn dryPrepassGeometryMain(input:VertexOut)->DryPrepassGeometryOut{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
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
  ${voxelLightCache ? "dryVoxelLightConsumerEligible=select(0u,1u,opaque.motionKind==DRY_GBUFFER_MOTION_STATIC);" : ""}
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
        if(cone.valid==0u){${recordInvalidAoPageWGSL}return DRY_PREPASS_INVALID_PACKED;}
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
      ${voxelLightCache ? "dryCurrentLightSlot=lightIndex;let cachedVoxel=dryVoxelLightVisibility(position,geometricNormal);if(cachedVoxel.y>0.0){let packedVisibility=mix(1.0,cachedVoxel.x,dry.tuningRays0.y);if(lightIndex<3u){visibility0[1u+lightIndex]=packedVisibility;}else if(lightIndex<7u){visibility1[lightIndex-3u]=packedVisibility;}else{visibility2.x=packedVisibility;}continue;}" : ""}
      var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){
        if(sampleIndex>=sampleCount){continue;}
        let sample=dryLightSample(light,sampleIndex,position);
        if(sample.valid==0u||dot(geometricNormal,sample.towardLight)<=0.0){continue;}
        let maximumDistance=select(directionalLightSceneExitDistance(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);
        if(dryDirectionalRayLeavesDomain(maximumDistance)){visibility+=1.0;continue;}
        let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,sample.towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);
        // Mirror of the inline path's normal escape and finite-emitter
        // clearance: the reduced-rate texel must hold the same visibility the
        // full-rate edge band computes inline.
        let coneCell_m=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
        let coneEscape_m=coneCell_m*dry.tuningRays1.z;
        let coneMaxRaw_m=max(0.0,ray.tMax_m-coneEscape_m*dot(geometricNormal,sample.towardLight));
        let coneMax_m=coneMaxRaw_m-select(0.0,dry.tuningRays1.w*coneCell_m,sample.finiteDistance_m>0.0);
        let cone=dryConeVisibility(ray.origin_m+geometricNormal*coneEscape_m,sample.towardLight,dry.tuningRays1.y,coneMax_m,geometricNormal,sample.finiteDistance_m>0.0);
        if(cone.valid==0u){${recordInvalidDirectPageWGSL}return DRY_PREPASS_INVALID_PACKED;}
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
const DRY_SILHOUETTE_EXACT_VALID:u32=0u;
const DRY_SILHOUETTE_EXACT_EXHAUSTED:u32=1u;
const DRY_SILHOUETTE_EXACT_INVALID:u32=2u;
const DRY_SILHOUETTE_REASON_NONE:u32=0u;
const DRY_SILHOUETTE_REASON_PUBLICATION:u32=1u;
const DRY_SILHOUETTE_REASON_SCENE_LIMIT:u32=2u;
const DRY_SILHOUETTE_REASON_TRAVERSAL:u32=3u;
const DRY_SILHOUETTE_REASON_TRACE_CONTRACT:u32=4u;
struct DrySilhouetteVisibility{packed:vec2u,status:u32,reason:u32}
fn drySilhouetteTraceVisibilityExact(opaque:DryHit,ro:vec3f,rd:vec3f)->DrySilhouetteVisibility{
  dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  let position=ro+rd*opaque.t;let geometricNormal=normalize(opaque.normal);
  // This explicit sparse edge tier is authoritative, so it uses the existing
  // hard visibility caps rather than the lower ordinary-frame quality budget.
  // Work remains strictly bounded to the compacted silhouette queue.
  let budget=SvoVisibilityBudget(${SVO_VISIBILITY_LIMITS.nodeVisits}u,${SVO_VISIBILITY_LIMITS.leafVisits}u,${SVO_VISIBILITY_LIMITS.workItems}u,${SVO_VISIBILITY_LIMITS.intersections}u);
  var visibility0=vec4f(1.0);var visibility1=vec4f(1.0);var visibility2=vec4f(1.0);
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion}u)!=0u){
    let radius=dryContactVisibilityRadius();if(radius>0.0){let biasCells=select(${SVO_CONTACT_VISIBILITY_CONTRACT.smoothBiasCells},${SVO_CONTACT_VISIBILITY_CONTRACT.hardFeatureBiasCells},opaque.featureId!=SVO_FEATURE_SMOOTH);var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}u;sampleIndex+=1u){let direction=dryContactVisibilityDirection(geometricNormal,opaque.featureId,sampleIndex);let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,direction,radius,dry.mapping.cellSize,biasCells);dryVisibilityStepInvalidReason=DRY_SILHOUETTE_REASON_NONE;let result=svoTraceVisibility(ray,budget,true,0.001,max(ray.originBias_m,1e-6));if(result.status==SVO_VIS_STATUS_EXHAUSTED){return DrySilhouetteVisibility(DRY_PREPASS_INVALID_PACKED,DRY_SILHOUETTE_EXACT_EXHAUSTED,DRY_SILHOUETTE_REASON_NONE);}if(result.status==SVO_VIS_STATUS_INVALID){return DrySilhouetteVisibility(DRY_PREPASS_INVALID_PACKED,DRY_SILHOUETTE_EXACT_INVALID,select(dryVisibilityStepInvalidReason,DRY_SILHOUETTE_REASON_TRACE_CONTRACT,dryVisibilityStepInvalidReason==DRY_SILHOUETTE_REASON_NONE));}visibility+=dot(result.transmittance,vec3f(1.0/3.0));}
      visibility0.x=clamp(visibility/f32(${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}),0.0,1.0);
    }
  }
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.exactShadow}u)!=0u){let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_LIGHT_MAXIMUM_RECORDS}u));
    for(var lightIndex=0u;lightIndex<${SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights}u;lightIndex+=1u){if(lightIndex>=lightCount){break;}let light=dryLighting.lights[lightIndex];if(light.identity.w!=dryLighting.metadata.y){continue;}let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA;let globalIllumination=(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u;let sampleCount=select(select(1u,select(dry.tuningCounts1.x,dry.tuningCounts0.w,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL}),area),1u,globalIllumination);var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=sampleCount){continue;}let sample=dryLightSample(light,sampleIndex,position);if(sample.valid==0u||dot(geometricNormal,sample.towardLight)<=0.0){continue;}let maximumDistance=select(directionalLightSceneExitDistance(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);if(dryDirectionalRayLeavesDomain(maximumDistance)){visibility+=1.0;continue;}let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,sample.towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);dryVisibilityIgnoredOwner=opaque.ownerId;dryVisibilityStepInvalidReason=DRY_SILHOUETTE_REASON_NONE;let result=svoTraceVisibility(ray,budget,true,0.001,max(ray.originBias_m,1e-6));dryVisibilityIgnoredOwner=DRY_OWNER_NONE;if(result.status==SVO_VIS_STATUS_EXHAUSTED){return DrySilhouetteVisibility(DRY_PREPASS_INVALID_PACKED,DRY_SILHOUETTE_EXACT_EXHAUSTED,DRY_SILHOUETTE_REASON_NONE);}if(result.status==SVO_VIS_STATUS_INVALID){return DrySilhouetteVisibility(DRY_PREPASS_INVALID_PACKED,DRY_SILHOUETTE_EXACT_INVALID,select(dryVisibilityStepInvalidReason,DRY_SILHOUETTE_REASON_TRACE_CONTRACT,dryVisibilityStepInvalidReason==DRY_SILHOUETTE_REASON_NONE));}visibility+=dot(result.transmittance,vec3f(1.0/3.0));}
      let packedVisibility=clamp(visibility/f32(sampleCount),0.0,1.0);if(lightIndex<3u){visibility0[1u+lightIndex]=packedVisibility;}else if(lightIndex<7u){visibility1[lightIndex-3u]=packedVisibility;}else{visibility2.x=packedVisibility;}
    }
  }
  return DrySilhouetteVisibility(dryPrepassPack(visibility0,visibility1,visibility2),DRY_SILHOUETTE_EXACT_VALID,DRY_SILHOUETTE_REASON_NONE);
}
@fragment fn dryPrepassVisibilityMain(input:VertexOut)->@location(0) vec2u{
  let coordinate=vec2i(input.position.xy);let geometry=textureLoad(dryPrepassGeometryTexture,coordinate,0);
  if(geometry.x<=0.0){return vec2u(0xffffffffu);}
  let identity=textureLoad(dryPrepassIdentityTexture,coordinate,0).x;let metadata=u32(round(geometry.w));
  let opaque=DryHit(geometry.x,dryPrepassDecodeNormal(geometry.yz),identity&0xffffu,identity>>16u,metadata&15u,(metadata>>4u)&15u,(metadata>>8u)&3u,(metadata>>10u)&1u,0.0,vec3u(0u));
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  return dryPrepassTraceVisibility(opaque,ro,rd);
}
fn dryPrepassShadeNoGi(opaque:DryHit,ro:vec3f,rd:vec3f)->vec3f{
  let position=ro+rd*opaque.t;let surface=dryEvaluateSurfaceMaterial(opaque,position);
  if(surface.valid==0u){return vec3f(0.0);}
  let closure=unifiedPbrMaterial(surface.baseColor,surface.metallic,surface.roughness,vec3f(0.0),0.0,surface.specularF0,surface.specularWeight,vec3f(0.0),0.0);
  var direct=vec3f(0.0);var sampleBudget=0u;let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_LIGHT_MAXIMUM_RECORDS}u));
  for(var lightIndex=0u;lightIndex<${SVO_DRY_SCENE_MAX_SHADED_LIGHTS}u;lightIndex+=1u){
    if(lightIndex>=lightCount||sampleBudget>=dry.tuningCounts0.z){break;}let light=dryLighting.lights[lightIndex];if(light.identity.w!=dryLighting.metadata.y){continue;}
    let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA;let sampleCount=select(1u,select(dry.tuningCounts1.x,dry.tuningCounts0.w,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL}),area);
    for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){
      if(sampleIndex>=sampleCount||sampleBudget>=dry.tuningCounts0.z){break;}sampleBudget+=1u;let sample=dryLightSample(light,sampleIndex,position);if(sample.valid==0u||dot(opaque.normal,sample.towardLight)<=0.0){continue;}
      let maximumDistance=select(directionalLightSceneExitDistance(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);
      let rigidBlocked=anyBodyBlockerIgnoring(position,sample.towardLight,opaque.ownerId,maximumDistance);let raw=select(dryPrepassChannel(1u+lightIndex),0.0,rigidBlocked);let visibility=vec3f(mix(1.0,raw,dry.tuningRays0.y));
      let lighting=unifiedLightingInputWithGeometry(opaque.normal,opaque.normal,-rd,sample.towardLight,sample.radiance*visibility/f32(sampleCount));direct+=shadeUnifiedSurface(closure,lighting);
    }
  }
  let viewDirection=normalize(-rd);let reflected=reflect(rd,opaque.normal);let diffuseColor=surface.baseColor*(1.0-surface.metallic);let f0=mix(surface.specularF0*surface.specularWeight,surface.baseColor,surface.metallic);let environmentBrdf=unifiedEnvironmentBrdf(max(dot(opaque.normal,viewDirection),0.0),surface.roughness,f0);let diffuseEnergy=max(vec3f(0.0),vec3f(1.0)-environmentBrdf);
  let diffuseVisibility=dryDiffuseMultiBounceVisibility(dryPrepassData0.x,diffuseColor);let diffuseEnvironment=diffuseColor*diffuseEnergy*svoEnvironmentDiffuseIrradiance(dryLighting.environment,opaque.normal)*diffuseVisibility/UNIFIED_PI;let specularEnvironment=dryEnvironment(reflected,surface.roughness)*environmentBrdf;
  return dryHoverRim(max(surface.emissive+diffuseEnvironment+specularEnvironment+direct*dry.giLighting.w,vec3f(0.0)),opaque,viewDirection);
}
@fragment fn dryPrepassShadeMain(input:VertexOut)->@location(0) vec4f{
  let coordinate=vec2i(input.position.xy);let geometry=textureLoad(dryPrepassGeometryTexture,coordinate,0);
  if(geometry.x<=0.0){return vec4f(0.0);}
  let identity=textureLoad(dryPrepassIdentityTexture,coordinate,0).x;let metadata=u32(round(geometry.w));
  let opaque=DryHit(geometry.x,dryPrepassDecodeNormal(geometry.yz),identity&0xffffu,identity>>16u,metadata&15u,(metadata>>4u)&15u,(metadata>>8u)&3u,(metadata>>10u)&1u,0.0,vec3u(0u));
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  // Until GLOBAL data is ready, rigid opaque radiance remains exact at full
  // rate, so avoid doing an unusable complete material evaluation here.
  if(opaque.motionKind!=DRY_GBUFFER_MOTION_STATIC){return vec4f(0.0);}
  let packed=textureLoad(dryPrepassVisibilityKeyTexture,coordinate,0);let packedValid=!all(packed.xy==DRY_PREPASS_INVALID_PACKED);dryPrepassData0=dryPrepassUnpack0(packed);dryPrepassData1=dryPrepassUnpack1(packed);dryPrepassData2=dryPrepassUnpack2(packed);
  dryPrepassState=select(0u,1u,packedValid);dryPrepassRadianceState=0u;dryPrepassGiState=0u;if(!packedValid){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u;return vec4f(0.0,0.0,0.0,-1.0);}dryCurrentLightSlot=0xffffffffu;dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  // Full-res relight stores only the expensive GI closure and evaluates the
  // material/direct/environment terms at every receiver. Reconstruction modes
  // instead cache the complete reduced-rate radiance: seed the private GI
  // closure first so shadeDryOpaque consumes this exact sample without tracing
  // it a second time. Invalid samples carry a negative alpha and are rejected
  // by the full-rate reconstruction, which then falls through to exact relight.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)==0u){return vec4f(dryPrepassShadeNoGi(opaque,ro,rd),opaque.t);}
  {let ignoredBodyOwner=select(DRY_OWNER_NONE,opaque.ownerId,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let gi=dryGlobalIllumination(ro+rd*opaque.t,opaque.normal,ignoredBodyOwner);
    if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u
      ||dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u){return vec4f(gi.radiance,select(-1.0,gi.visibility,gi.valid!=0u));}
    if(gi.valid==0u){return vec4f(0.0,0.0,0.0,-1.0);}dryPrepassGi=vec4f(gi.radiance,gi.visibility);dryPrepassGiState=1u;
  }
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
struct DryPrimarySeamSample{geometry:vec4f,identity:vec2u,valid:u32}
fn dryPrimarySeamForeground(depth:f32,centreDepth:f32)->bool{
  if(!(depth>0.0&&depth<DRY_MISS)){return false;}
  if(!(centreDepth>0.0&&centreDepth<DRY_MISS)){return true;}
  return depth+max(.001,.003*depth)<centreDepth;
}
fn dryPrimarySeamSample(coordinate:vec2i)->DryPrimarySeamSample{
  let dimensions=vec2i(textureDimensions(drySplitGeometryRead));
  let centre=drySplitGeometryAt(coordinate);let centreDepth=centre.w;
  let pairs=array<vec4i,4>(vec4i(-1,0,1,0),vec4i(0,-1,0,1),vec4i(-1,-1,1,1),vec4i(-1,1,1,-1));
  var bestCoordinate=vec2i(0);var bestDepth=DRY_MISS;var valid=0u;
  for(var pairIndex=0u;pairIndex<4u;pairIndex+=1u){
    let pair=pairs[pairIndex];let firstCoordinate=coordinate+pair.xy;let secondCoordinate=coordinate+pair.zw;
    if(any(firstCoordinate<vec2i(0))||any(firstCoordinate>=dimensions)||any(secondCoordinate<vec2i(0))||any(secondCoordinate>=dimensions)){continue;}
    let first=drySplitGeometryAt(firstCoordinate);let second=drySplitGeometryAt(secondCoordinate);
    if(!dryPrimarySeamForeground(first.w,centreDepth)||!dryPrimarySeamForeground(second.w,centreDepth)){continue;}
    let firstIdentity=drySplitIdentityAt(firstCoordinate);let secondIdentity=drySplitIdentityAt(secondCoordinate);let differentSurface=(firstIdentity.x&0x8000ffffu)!=(secondIdentity.x&0x8000ffffu)||(firstIdentity.y&0xffffu)!=(secondIdentity.y&0xffffu);if(!differentSurface){continue;}
    // Extend the rear of the two bracketing surfaces. This closes the exposed
    // background without growing the nearer silhouette over its neighbour.
    let candidateCoordinate=select(secondCoordinate,firstCoordinate,first.w>=second.w);let candidateDepth=max(first.w,second.w);
    if(candidateDepth<bestDepth){bestDepth=candidateDepth;bestCoordinate=candidateCoordinate;valid=1u;}
  }
  if(valid==0u){return DryPrimarySeamSample(vec4f(0.0,1.0,0.0,DRY_MISS),vec2u(0u),0u);}
  return DryPrimarySeamSample(drySplitGeometryAt(bestCoordinate),drySplitIdentityAt(bestCoordinate).xy,1u);
}
fn dryPrimarySeamHit(sample:DryPrimarySeamSample)->DryHit{
  let packedOpaqueMaterial=sample.identity.x;${splitOpaqueMaterialDecodeWGSL}
  let metadata=sample.identity.y;
  return DryHit(sample.geometry.w,normalize(sample.geometry.xyz),opaqueMaterial,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));
}
@fragment fn dryPrimarySeamMain(input:VertexOut)->DryVisibilityOut{
  let coordinate=vec2i(input.position.xy);let seam=dryPrimarySeamSample(coordinate);if(seam.valid==0u){discard;}
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  let opaque=dryPrimarySeamHit(seam);let generation=dryPublicationGeneration();let media=dryMediumPair(rd,opaque.normal,DRY_MEDIUM_OPAQUE);let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u)|svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_TRACED);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}let targets=svoGBufferSurface(vec3f(0.0),opaque.t,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);
  return drySplitVisibilityOut(targets,dryHardwareDepth(opaque.t,rd,forward));
}
@fragment fn dryVisibilityMain(input:VertexOut)->DryVisibilityOut{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryThickGlassEnabled=0u;
  let opaque=${splitPrimaryTraceWGSL};${splitVisibilityGlassDiscoveryWGSL}
  ${splitVisibilityGlassReturnWGSL}
  if(opaque.t<DRY_MISS){let media=dryMediumPair(rd,opaque.normal,DRY_MEDIUM_OPAQUE);let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u)|svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_TRACED);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}let targets=svoGBufferSurface(vec3f(0.0),opaque.t,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);return drySplitVisibilityOut(targets,dryHardwareDepth(opaque.t,rd,forward));}
  return drySplitVisibilityOut(svoGBufferMiss(vec3f(0.0),0u,generation,DRY_GBUFFER_NO_INTERSECTION,svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_TRACED)),0.0);
}
${reduced ? `@fragment fn dryReconstructedLightingMain(input:VertexOut)->@location(0) vec4f{
  let coordinate=vec2i(input.position.xy);let geometry=drySplitGeometryAt(coordinate);if(!(geometry.w<DRY_MISS)){discard;}
  let opaqueIdentity=drySplitIdentityAt(coordinate);let packedOpaqueMaterial=opaqueIdentity.x;${splitOpaqueMaterialDecodeWGSL}let metadata=opaqueIdentity.y;let opaque=DryHit(geometry.w,geometry.xyz,opaqueMaterial,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));
  dryPrepassData0=vec4f(1.0);dryPrepassData1=vec4f(1.0);dryPrepassData2=vec4f(1.0);dryPrepassRadiance=vec4f(0.0);dryPrepassGi=vec4f(0.0,0.0,0.0,1.0);dryPrepassState=0u;dryPrepassRadianceState=0u;dryPrepassGiState=0u;dryPrepassExactEdgeState=0u;dryCurrentLightSlot=0xffffffffu;
  if(dryNodeMipReady()){dryPrepassResolve(input.position.xy,opaque.t,opaque.normal,opaque);}if(dryPrepassRadianceState!=1u){discard;}
  ${rasterGlassDiscovery ? "let glassKey=textureLoad(drySplitGlassKeyRead,coordinate,0).x;" : "let glassKey=(packedOpaqueMaterial>>16u)&0x1ffu;"}if(glassKey>0u){discard;}
  let ndc=input.uv*2.0-1.0;let vignette=1.0-.14*dot(ndc*.58,ndc*.58);return vec4f(max(dryPrepassRadiance.rgb,vec3f(0.0))*vignette,opaque.t);
}
` : ""}@fragment fn dryLightingMain(input:VertexOut)->@location(0) vec4f{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryThickGlassEnabled=0u;
  let coordinate=vec2i(input.position.xy);var geometry=drySplitGeometryAt(coordinate);var opaqueIdentity=drySplitIdentityAt(coordinate);if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.silhouetteRefinement}u)!=0u){let seam=dryPrimarySeamSample(coordinate);if(seam.valid!=0u){geometry=seam.geometry;opaqueIdentity=vec4u(seam.identity,0u,0u);}}var opaque=missHit();
  let packedOpaqueMaterial=opaqueIdentity.x;${splitOpaqueMaterialDecodeWGSL}if(geometry.w<DRY_MISS){let metadata=opaqueIdentity.y;opaque=DryHit(geometry.w,geometry.xyz,opaqueMaterial,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));}
  ${prepassResolveCallWGSL}var glass=dryGlassMiss();${splitGlassKeyLoadWGSL}${reduced ? `if(dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u&&dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u&&dryPrepassRadianceState==1u&&glassKey==0u){discard;}` : ""}if(glassKey>0u){let recordIndex=glassKey-1u;if(recordIndex<dry.terrain.y){let record=dryGlassPane(recordIndex);let candidate=svoThinGlassIntersect(record,ro,rd,0.0,opaque.t,1e-6,record.extentIorEpsilon.w);if(candidate.valid!=0u){glass=DryGlassHit(candidate,recordIndex);}}}var color=shadeDryOpaque(opaque,ro,rd);var depth=opaque.t;let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;if(glassVisible){let glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);return vec4f(max(color*vignette,vec3f(0.0)),select(0.0,depth,depth<DRY_MISS));
}
@fragment fn drySkyLightingMain(input:VertexOut)->@location(0) vec4f{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryThickGlassEnabled=0u;
  let coordinate=vec2i(input.position.xy);var opaque=missHit();
  // Skipping the G-buffer identity load is most of what makes this entry cheap,
  // but without raster glass discovery the glass key is packed into that very
  // plane, so there it has to be read after all.
  ${rasterGlassDiscovery ? "" : "let packedOpaqueMaterial=drySplitIdentityAt(coordinate).x;"}
  var glass=dryGlassMiss();${splitGlassKeyLoadWGSL}if(glassKey>0u){let recordIndex=glassKey-1u;if(recordIndex<dry.terrain.y){let record=dryGlassPane(recordIndex);let candidate=svoThinGlassIntersect(record,ro,rd,0.0,opaque.t,1e-6,record.extentIorEpsilon.w);if(candidate.valid!=0u){glass=DryGlassHit(candidate,recordIndex);}}}var color=shadeDryOpaque(opaque,ro,rd);var depth=opaque.t;
  if(glass.hit.valid!=0u&&glass.hit.t_m<opaque.t){let glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);return vec4f(max(color*vignette,vec3f(0.0)),select(0.0,depth,depth<DRY_MISS));
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
  const prepassFromPrimaryEntryWGSL = reduced && split ? /* wgsl */ `struct DryPrepassBoundaryQueue{
  count:atomic<u32>,invalidAoPages:atomic<u32>,invalidDirectPages:atomic<u32>,failedRefinements:atomic<u32>,coordinates:array<u32>
}
@group(1) @binding(4) var dryPrepassVisibilityWrite:texture_storage_2d<rg32uint,write>;
@group(1) @binding(5) var dryPrepassGeometryWrite:texture_storage_2d<rgba16float,write>;
@group(1) @binding(6) var dryPrepassIdentityWrite:texture_storage_2d<r32uint,write>;
@group(1) @binding(7) var<storage,read_write> dryPrepassBoundaryQueue:DryPrepassBoundaryQueue;
${prepassFanoutDeclarationWGSL}
@group(1) @binding(9) var drySilhouetteRefinementWrite:texture_storage_2d<rg32uint,write>;
@group(1) @binding(11) var<storage,read_write> drySilhouetteDispatch:array<atomic<u32>>;
@group(1) @binding(12) var drySilhouetteRefinementStateWrite:texture_storage_2d<r32uint,write>;
@group(1) @binding(13) var<storage,read_write> drySilhouetteFailureReasons:array<atomic<u32>>;
@compute @workgroup_size(1) fn dryPrepassResetMain(){
  atomicStore(&dryPrepassBoundaryQueue.count,0u);atomicStore(&dryPrepassBoundaryQueue.invalidAoPages,0u);atomicStore(&dryPrepassBoundaryQueue.invalidDirectPages,0u);atomicStore(&dryPrepassBoundaryQueue.failedRefinements,0u);
}
fn dryPrepassRay(coordinate:vec2u,dimensions:vec2u)->mat2x3f{let uv=vec2f((f32(coordinate.x)+.5)/f32(dimensions.x),1.0-(f32(coordinate.y)+.5)/f32(dimensions.y));let ndc=uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());return mat2x3f(ro,rd);}
fn dryPrepassStore(coordinate:vec2i,opaque:DryHit,ro:vec3f,rd:vec3f){if(!(opaque.t<DRY_MISS)){textureStore(dryPrepassVisibilityWrite,coordinate,vec4u(0xffffffffu));textureStore(dryPrepassGeometryWrite,coordinate,vec4f(0.0));textureStore(dryPrepassIdentityWrite,coordinate,vec4u(0xffffffffu));${prepassFanoutMissStoreWGSL}return;}textureStore(dryPrepassGeometryWrite,coordinate,vec4f(opaque.t,dryPrepassEncodeNormal(opaque.normal),f32(dryPrepassHitMetadata(opaque))));textureStore(dryPrepassIdentityWrite,coordinate,vec4u(dryPrepassPackIdentity(opaque),0u,0u,0u));${prepassFanoutHitStoreWGSL}${prepassVisibilityStoreWGSL}}
@compute @workgroup_size(8,8) fn dryPrepassCoherentMain(@builtin(global_invocation_id) globalId:vec3u){
  let dimensions=textureDimensions(dryPrepassGeometryWrite);if(any(globalId.xy>=dimensions)){return;}let coordinate=vec2i(globalId.xy);let ray=dryPrepassRay(globalId.xy,dimensions);
  let fullDimensions=textureDimensions(drySplitGeometryRead);let maximumCoordinate=vec2i(fullDimensions)-vec2i(1);let sampleBase=clamp(vec2i(floor((vec2f(globalId.xy)+vec2f(.5))*vec2f(fullDimensions)/vec2f(dimensions)-vec2f(.5))),vec2i(0),maximumCoordinate);
  var primaryGeometry:array<vec4f,4>;var primaryIdentity:array<vec4u,4>;var allMiss=true;
  for(var sample=0u;sample<4u;sample+=1u){let sourceCoordinate=clamp(sampleBase+vec2i(i32(sample&1u),i32(sample>>1u)),vec2i(0),maximumCoordinate);primaryGeometry[sample]=drySplitGeometryAt(sourceCoordinate);primaryIdentity[sample]=drySplitIdentityAt(sourceCoordinate);allMiss=allMiss&&!(primaryGeometry[sample].w<DRY_MISS);}
  if(allMiss){dryPrepassStore(coordinate,missHit(),ray[0],ray[1]);return;}
  var referenceGeometry=primaryGeometry[3];var referenceIdentity=primaryIdentity[3];var homogeneous=referenceGeometry.w<DRY_MISS;
  for(var sample=0u;sample<3u;sample+=1u){let geometry=primaryGeometry[sample];let hit=geometry.w<DRY_MISS;let sameIdentity=(primaryIdentity[sample].x&0x8000ffffu)==(referenceIdentity.x&0x8000ffffu)&&primaryIdentity[sample].y==referenceIdentity.y;let depthClose=abs(geometry.w-referenceGeometry.w)<=max(.0001,.01*referenceGeometry.w);let normalClose=dot(normalize(geometry.xyz),normalize(referenceGeometry.xyz))>=.9999;homogeneous=homogeneous&&hit&&sameIdentity&&depthClose&&normalClose;}
  if(!homogeneous){${inlineBoundaryWGSL}}
  let metadata=referenceIdentity.y;let packedMaterial=referenceIdentity.x;let material=select(packedMaterial&0xffffu,0x80000000u|(packedMaterial&0xffffu),(packedMaterial&0x80000000u)!=0u);let opaque=DryHit(referenceGeometry.w,normalize(referenceGeometry.xyz),material,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));dryPrepassStore(coordinate,opaque,ray[0],ray[1]);
}
@compute @workgroup_size(64) fn dryPrepassBoundaryMain(@builtin(global_invocation_id) globalId:vec3u){
  let queueCount=atomicLoad(&dryPrepassBoundaryQueue.count);if(globalId.x>=queueCount){return;}let dimensions=textureDimensions(dryPrepassGeometryWrite);let packedCoordinate=dryPrepassBoundaryQueue.coordinates[globalId.x];let coordinate=vec2u(packedCoordinate%dimensions.x,packedCoordinate/dimensions.x);let ray=dryPrepassRay(coordinate,dimensions);dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassEnabled=0u;let opaque=traceOpaqueScene(ray[0],ray[1]);dryPrepassStore(vec2i(coordinate),opaque,ray[0],ray[1]);
}
@compute @workgroup_size(1) fn drySilhouetteResetMain(){
  atomicStore(&dryPrepassBoundaryQueue.count,0u);atomicStore(&dryPrepassBoundaryQueue.invalidAoPages,0u);atomicStore(&dryPrepassBoundaryQueue.invalidDirectPages,0u);atomicStore(&dryPrepassBoundaryQueue.failedRefinements,0u);for(var reason=0u;reason<${SVO_DRY_SILHOUETTE_FAILURE_REASON_CONTRACT.words}u;reason+=1u){atomicStore(&drySilhouetteFailureReasons[reason],0u);}atomicStore(&drySilhouetteDispatch[0],0u);atomicStore(&drySilhouetteDispatch[1],0u);atomicStore(&drySilhouetteDispatch[2],0u);
}
fn drySilhouetteAmbiguous(coordinate:vec2i,dimensions:vec2u)->bool{
  let geometry=drySplitGeometryAt(coordinate);if(!(geometry.w>0.0&&geometry.w<DRY_MISS)){return false;}let identity=drySplitIdentityAt(coordinate);let offsets=array<vec2i,4>(vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
  for(var index=0u;index<4u;index+=1u){let neighbourCoordinate=coordinate+offsets[index];if(any(neighbourCoordinate<vec2i(0))||any(neighbourCoordinate>=vec2i(dimensions))){continue;}let neighbour=drySplitGeometryAt(neighbourCoordinate);if(!(neighbour.w>0.0&&neighbour.w<DRY_MISS)){return true;}let neighbourIdentity=drySplitIdentityAt(neighbourCoordinate);let sameSurface=(identity.x&0x8000ffffu)==(neighbourIdentity.x&0x8000ffffu)&&(identity.y&0xffffu)==(neighbourIdentity.y&0xffffu);let depthClose=abs(geometry.w-neighbour.w)<=max(.001,.02*geometry.w);if(!sameSurface||!depthClose){return true;}}
  return false;
}
@compute @workgroup_size(8,8) fn drySilhouetteClassifyMain(@builtin(global_invocation_id) globalId:vec3u){
  let dimensions=textureDimensions(drySilhouetteRefinementWrite);if(any(globalId.xy>=dimensions)){return;}let coordinate=vec2i(globalId.xy);textureStore(drySilhouetteRefinementStateWrite,coordinate,vec4u(DRY_SILHOUETTE_STATE_UNTOUCHED));if(!drySilhouetteAmbiguous(coordinate,dimensions)){return;}let queueIndex=atomicAdd(&dryPrepassBoundaryQueue.count,1u);if(queueIndex<dimensions.x*dimensions.y){dryPrepassBoundaryQueue.coordinates[queueIndex]=globalId.y*dimensions.x+globalId.x;}
}
@compute @workgroup_size(1) fn drySilhouetteFinalizeMain(){
  let dimensions=textureDimensions(drySilhouetteRefinementWrite);let count=min(atomicLoad(&dryPrepassBoundaryQueue.count),dimensions.x*dimensions.y);let groupCount=(count+${SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize - 1}u)/${SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize}u;let x=min(groupCount,65535u);var y=0u;if(x>0u){y=(groupCount+x-1u)/x;}atomicStore(&drySilhouetteDispatch[0],x);atomicStore(&drySilhouetteDispatch[1],y);atomicStore(&drySilhouetteDispatch[2],1u);
}
@compute @workgroup_size(${SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize}) fn drySilhouetteRefineMain(@builtin(global_invocation_id) globalId:vec3u,@builtin(num_workgroups) groups:vec3u){
  let queueIndex=globalId.x+globalId.y*groups.x*${SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize}u;let queueCount=atomicLoad(&dryPrepassBoundaryQueue.count);if(queueIndex>=queueCount){return;}let dimensions=textureDimensions(drySilhouetteRefinementWrite);let packedCoordinate=dryPrepassBoundaryQueue.coordinates[queueIndex];let coordinate=vec2u(packedCoordinate%dimensions.x,packedCoordinate/dimensions.x);let geometry=drySplitGeometryAt(vec2i(coordinate));if(!(geometry.w>0.0&&geometry.w<DRY_MISS)){return;}let identity=drySplitIdentityAt(vec2i(coordinate));let metadata=identity.y;let packedMaterial=identity.x;let material=select(packedMaterial&0xffffu,0x80000000u|(packedMaterial&0xffffu),(packedMaterial&0x80000000u)!=0u);let opaque=DryHit(geometry.w,normalize(geometry.xyz),material,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));let ray=dryPrepassRay(coordinate,dimensions);let exact=drySilhouetteTraceVisibilityExact(opaque,ray[0],ray[1]);if(exact.status==DRY_SILHOUETTE_EXACT_EXHAUSTED){atomicAdd(&dryPrepassBoundaryQueue.invalidAoPages,1u);atomicAdd(&dryPrepassBoundaryQueue.failedRefinements,1u);atomicAdd(&drySilhouetteFailureReasons[${SVO_DRY_SILHOUETTE_FAILURE_REASON_CONTRACT.exhaustedWord}u],1u);}else if(exact.status==DRY_SILHOUETTE_EXACT_INVALID){atomicAdd(&dryPrepassBoundaryQueue.invalidDirectPages,1u);atomicAdd(&dryPrepassBoundaryQueue.failedRefinements,1u);atomicAdd(&drySilhouetteFailureReasons[min(exact.reason,${SVO_DRY_SILHOUETTE_FAILURE_REASON_CONTRACT.traceContractWord}u)],1u);}textureStore(drySilhouetteRefinementWrite,vec2i(coordinate),vec4u(exact.packed,0u,0u));textureStore(drySilhouetteRefinementStateWrite,vec2i(coordinate),vec4u(select(DRY_SILHOUETTE_STATE_FAILED,DRY_SILHOUETTE_STATE_VALID,exact.status==DRY_SILHOUETTE_EXACT_VALID)));
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
  invalidGiPages:atomic<u32>,
  reserved1:u32,
  reserved2:u32,
  reserved3:u32,
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
        return DryWorldGiCacheLookup(DryGlobalIllumination(vec3f(rg,bv.x),bv.y,1u),1u,claimSlot,claimState);
      }
    }
    if(state==0u){claimSlot=slot;claimState=0u;break;}
    // A ready entry is safe to replace after a compare-exchange claim. Never
    // select state 1, which denotes a writer currently publishing its payload.
    if(state!=1u){claimSlot=slot;claimState=state;}
  }
  return DryWorldGiCacheLookup(DryGlobalIllumination(vec3f(0.0),1.0,1u),0u,claimSlot,claimState);
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
  dryWorldGiFrame.bodySignature=signature;dryWorldGiFrame.movingBodyCount=movingBodyCount;dryWorldGiFrame.bodyCount=bodyCount;atomicStore(&dryWorldGiFrame.invalidGiPages,0u);
  dryWorldGiFrame.cameraPosition=vec4f(origin,0.0);
  dryWorldGiFrame.cameraForwardAspect=vec4f(forward,uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov());
  dryWorldGiFrame.cameraRight=vec4f(right,0.0);dryWorldGiFrame.cameraUp=vec4f(up,0.0);
}
fn dryWorldGiFrameRay(coordinate:vec2u,dimensions:vec2u)->mat2x3f{
  let uv=vec2f((f32(coordinate.x)+.5)/f32(dimensions.x),1.0-(f32(coordinate.y)+.5)/f32(dimensions.y));
  let ndc=uv*2.0-1.0;let rd=normalize(dryWorldGiFrame.cameraForwardAspect.xyz
    +dryWorldGiFrame.cameraRight.xyz*ndc.x*dryWorldGiFrame.cameraForwardAspect.w
    +dryWorldGiFrame.cameraUp.xyz*ndc.y*cameraTanHalfFov());
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
    if(dynamicValue.valid==0u){atomicAdd(&dryWorldGiFrame.invalidGiPages,1u);}
    textureStore(dryWorldGiOutput,coordinate,vec4f(dynamicValue.radiance,select(-1.0,dynamicValue.visibility,dynamicValue.valid!=0u)));return;
  }
  let bodyAware=influence.bodyMask!=0u;let bodyNamespace=select(0x4f1bbcdcu,influence.signature,bodyAware);
  let key=dryWorldGiKey(position,opaque.normal,bodyNamespace);let cached=dryWorldGiFind(key);
  if(cached.hit!=0u){textureStore(dryWorldGiOutput,coordinate,vec4f(cached.value.radiance,cached.value.visibility));return;}
  dryWorldGiIgnoreRigidBodies=select(1u,0u,bodyAware);dryWorldGiBodyMask=influence.bodyMask;dryPrepassGiState=0u;
  let value=dryGlobalIllumination(position,opaque.normal,select(DRY_OWNER_NONE,ignoredBodyOwner,bodyAware));
  if(value.valid==0u){atomicAdd(&dryWorldGiFrame.invalidGiPages,1u);}else{dryWorldGiInsert(key,cached.claimSlot,cached.claimState,value);}
  textureStore(dryWorldGiOutput,coordinate,vec4f(value.radiance,select(-1.0,value.visibility,value.valid!=0u)));
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
// highlight is (firstOwner, lastOwner, strength, falloff) for the object under
// the editor cursor, appended after the terrain mirror so no other shader's view
// of this buffer moves. A range rather than one id because a described object is
// several primitives — a lantern is three, a grown tree is thirty — and they are
// contiguous in owner order by construction. See lib/scenery-expand.ts.
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f, terrainMeta:vec4f, terrainFeatures:array<vec4f,16>, highlight:vec4f }
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
  // x: stable address-plan generation; y: directory pages; z: levels; w: publication mode.
  nodeMip:vec4u,
  // xyz: opacity atlas texels; w: finest level that owns an opacity page.
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
  // Centre and radius of one sphere enclosing every rigid body; radius < 0 when
  // the scene has none.
  rigidBounds:vec4f,
  // x: first BVH record; y: node count; z: root node; w: complete render revision.
  primitiveCandidates:vec4u,
  // u32 word offsets of control, publication, nodes, and leaves in the structural arena.
  structureOffsets:vec4u,
  // Optional derived traversal offsets; zero for canonical and compact modes.
  derivedTraversal:vec4u,
  // x: screen-space threshold in reference pixels; y: mode; z: fixed level.
  lod:vec4f,
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
${cameraApertureShaderLibrary()}
@group(0) @binding(1) var<uniform> bodies:array<BodyGPU,12>;
@group(0) @binding(3) var<storage,read> materialOwners:array<u32>;
@group(0) @binding(4) var<storage,read> drySceneArena:array<u32>;
// Bound to the scene-geometry lane itself, not to the whole payload arena, so
// the lane's own base is zero and the codec's base argument is a literal.
@group(0) @binding(6) var<storage,read> sceneGeometry:array<u32>;
@group(0) @binding(9) var<uniform> dry:DryParams;
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
@group(0) @binding(26) var nodeMipPageValidity:texture_2d<u32>;
@group(0) @binding(27) var tetraRadiancePageValidity:texture_2d<u32>;

const DRY_SCENE_MATERIAL_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.materialOffsetBytes / 4}u;
const DRY_SCENE_PRIMITIVE_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.primitiveOffsetBytes / 4}u;
const DRY_SCENE_GLASS_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.glassOffsetBytes / 4}u;
const DRY_SCENE_CLUSTER_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.clusterOffsetBytes / 4}u;
const DRY_SCENE_CLUSTER_WORD_LIMIT:u32=DRY_SCENE_CLUSTER_WORD_OFFSET+${SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES / 4}u;
const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.fieldProgramOffsetBytes / 4}u;
const DRY_SCENE_FIELD_PROGRAM_CAPACITY:u32=${SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY}u;
const DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS:u32=${SVO_FIELD_PROGRAM_BLOCK_WORDS}u;
const DRY_SCENE_TERRAIN_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes / 4}u;
const DRY_SCENE_TERRAIN_SAMPLE_WORD_OFFSET:u32=DRY_SCENE_TERRAIN_WORD_OFFSET+${SVO_DRY_SCENE_TERRAIN_HEADER_WORDS}u;
fn drySceneWords4(offset:u32)->vec4u{return vec4u(drySceneArena[offset],drySceneArena[offset+1u],drySceneArena[offset+2u],drySceneArena[offset+3u]);}
fn dryMaterial(index:u32)->SvoMaterialRecord{
  let base=DRY_SCENE_MATERIAL_WORD_OFFSET+index*${SVO_MATERIAL_RECORD_STRIDE_BYTES / 4}u;
  return SvoMaterialRecord(bitcast<vec4f>(drySceneWords4(base)),bitcast<vec4f>(drySceneWords4(base+4u)),
    bitcast<vec4f>(drySceneWords4(base+8u)),bitcast<vec4f>(drySceneWords4(base+12u)),
    bitcast<vec4f>(drySceneWords4(base+16u)),drySceneWords4(base+20u));
}
fn dryPrimitive(index:u32)->SvoPrimitiveRecord{
  let base=DRY_SCENE_PRIMITIVE_WORD_OFFSET+index*${SVO_PRIMITIVE_RECORD_STRIDE_BYTES / 4}u;
  return SvoPrimitiveRecord(drySceneWords4(base),drySceneWords4(base+4u),bitcast<vec4f>(drySceneWords4(base+8u)),drySceneWords4(base+12u));
}
// An aggregate's packing, resolved from the word-13 reference exactly as a
// heightfield's samples are. A record of any other kind carries the invalid
// sentinel there, so it reads the "not resolved" block and the ABI ignores it.
fn dryClusterPacking(record:SvoPrimitiveRecord)->SvoClusterPacking{
  if(svoPrimitiveKind(record)!=SVO_KIND_SMOOTH_UNION_CLUSTER){return svoInvalidClusterPacking();}
  let base=svoPrimitiveClusterReference(record);
  if(base<DRY_SCENE_CLUSTER_WORD_OFFSET||base+${SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS}u>DRY_SCENE_CLUSTER_WORD_LIMIT){return svoInvalidClusterPacking();}
  let head=drySceneWords4(base);
  let shape=bitcast<vec4f>(drySceneWords4(base+${CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD}u));
  let placement=bitcast<vec4f>(drySceneWords4(base+${CLUSTER_BLOCK_LOBE_SPAN_WORD}u));
  // The polyline is copied whatever the field is: a uniform read costs the same
  // for every cluster and branching on the field here would diverge the wave
  // for the one arm that needs it. Fields that do not use it never look.
  var points=array<vec4f,${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}>();
  for(var index=0u;index<${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}u;index+=1u){
    points[index]=bitcast<vec4f>(drySceneWords4(base+${CLUSTER_BLOCK_POINTS_WORD}u+index*4u));
  }
  return SvoClusterPacking(head[${CLUSTER_BLOCK_FIELD_WORD}],head[${CLUSTER_BLOCK_SEED_WORD}],head[${CLUSTER_BLOCK_COUNT_WORD}],
    bitcast<f32>(head[${CLUSTER_BLOCK_SMOOTH_RADIUS_WORD}]),shape.x,shape.y,shape.z,shape.w,
    placement.x,placement.y,placement.z,points);
}
fn drySceneWord(offset:u32)->u32{return drySceneArena[offset];}
${svoFieldProgramWGSL({
  functionName: "dryFieldProgramBlock",
  loadWord: "drySceneWord",
  baseWordExpression: "DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET",
  capacityExpression: "DRY_SCENE_FIELD_PROGRAM_CAPACITY",
})}
// The shared ABI's host hook. The generated evaluator above addresses tapes by
// slot; a record addresses its own by word offset, exactly as a heightfield and
// an aggregate do. Translating here is what keeps the reference *checkable*: an
// offset that is not a block start is a stale or corrupt word rather than a tape
// read from the middle of its neighbour, and it answers as unresolved — which
// the ABI reports as an invalid record instead of drawing the conservative box.
fn svoFieldProgramReferenceSample(reference:u32,localPoint:vec3f)->SvoFieldValue{
  if(reference<DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET){return SvoFieldValue(1e20,1.0);}
  let offset=reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET;
  if(offset%DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS!=0u){return SvoFieldValue(1e20,1.0);}
  return dryFieldProgramBlock(offset/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,localPoint);
}
fn dryGlassPane(index:u32)->SvoThinGlassRecord{
  let base=DRY_SCENE_GLASS_WORD_OFFSET+index*${SVO_THIN_GLASS_RECORD_WORDS}u;
  return SvoThinGlassRecord(bitcast<vec4f>(drySceneWords4(base)),bitcast<vec4f>(drySceneWords4(base+4u)),
    bitcast<vec4f>(drySceneWords4(base+8u)),bitcast<vec4f>(drySceneWords4(base+12u)),drySceneWords4(base+16u));
}
fn dryPublicationWord(index:u32)->u32{return svoStructure[dry.structureOffsets.y+index];}

// Page failures remain typed inside the invocation so dependent cone/GI work
// can fail closed. They are diagnostics, not scene colour output.
var<private> dryDerivedPageFailure:u32=0u;

${canonicalTraversalWGSL}${screenSpaceTraversalWGSL}${lodDescentWGSL}${lodUniformWGSL}${screenSpacePrimaryProxyWGSL}${tieredComputeResolveDeclarationsWGSL}
${wideTraversalWGSL}${compactTraversalWGSL}${brickOccupancyHelpersWGSL}
${liveLeafLifecycleWGSL}
${createSvoDryConeMarcherWGSL({ branchlessMorton: true, rangedDirectorySearch: true, fluidCoverage: true, directPageTable: true })}
var<private> dryGiPageCache:DryNodeMipPageCache;
/** The ancestor page a sub-floor radiance sample redirects to; coarser, so it changes far less often. */
var<private> dryGiRadiancePageCache:DryNodeMipPageCache;
fn dryTetraRadianceReady()->bool{return dry.tetrahedralRadiance.y!=0u&&dry.tetrahedralRadiance.x==dry.nodeMip.x&&dryNodeMipReady();}
fn dryTetraRadiancePageValid(pageIndex:u32)->bool{
  let dimensions=textureDimensions(tetraRadiancePageValidity);return svoDerivedPageValidityResident(dimensions,pageIndex)&&textureLoad(tetraRadiancePageValidity,svoDerivedPageValidityTexel(dimensions,pageIndex),0).x!=0u;
}
fn dryNodeMipSceneExitDistance(position:vec3f,direction:vec3f)->f32{
  let minimum=dry.nodeMipOrigin.xyz;let maximum=minimum+dry.nodeMipExtent.xyz;var enter=0.0;var exit=DRY_MISS;
  for(var axis=0u;axis<3u;axis+=1u){if(abs(direction[axis])<=1e-9){if(position[axis]<minimum[axis]||position[axis]>maximum[axis]){return 0.0;}}
    else{let first=(minimum[axis]-position[axis])/direction[axis];let second=(maximum[axis]-position[axis])/direction[axis];enter=max(enter,min(first,second));exit=min(exit,max(first,second));if(exit<enter){return 0.0;}}}
  return max(exit,0.0);
}
/**
 * The radiance atlas's own page, which is not the opacity page.
 *
 * dry.tetrahedralRadiance.z is the finest level that owns a radiance page and
 * .w is the slot its atlas begins at. A cone sampling finer than the floor
 * reads the ancestor at the floor instead — legitimate, not a fallback:
 * planSvoNodeMipPyramid inserts an ancestor at every level above every occupied
 * base page, so residency is ancestor-closed and the ancestor is always there.
 * The lookup is recomputed from the query position at the floor level rather
 * than shifted from the fine page, so the fractional texel is exact instead of
 * quantised by the fine page's rounding.
 */
fn dryTetraRadianceLevelFloor()->u32{return min(dry.tetrahedralRadiance.z,dry.nodeMip.z-1u);}
/** 2: page unusable, 1: certified black, 0: sample it. Resolved once per page, never per tap. */
fn dryTetraRadianceState(pageIndex:u32)->u32{
  if(!dryTetraRadiancePageValid(pageIndex)){return 2u;}
  if(pageIndex<dry.tetrahedralRadiance.w){return 1u;}
  return textureLoad(tetraRadianceBlackPages,vec2u(pageIndex,0u),0).x;
}
fn dryTetraRadiancePageOrigin(pageIndex:u32)->vec3u{
  let physical=u32(SVO_NODE_MIP_PHYSICAL_SIZE);
  let atlasPages=max(textureDimensions(tetraRadianceLobe0)/vec3u(physical),vec3u(1u));
  let slot=pageIndex-dry.tetrahedralRadiance.w;
  return vec3u(slot%atlasPages.x,(slot/atlasPages.x)%atlasPages.y,slot/(atlasPages.x*atlasPages.y))*physical;
}
fn svoTetraRadianceConeLoad(query:SvoTetraRadianceConeQuery)->SvoTetraRadianceConeSourceSample{
  if(!dryNodeMipReady()){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),0u,0u);}
  let level=min(max(u32(max(floor(query.lod),0.0)),dryNodeMipOpacityLevelFloor()),dry.nodeMip.z-1u);let levelScale=exp2(f32(level));
  let virtualVoxel=(query.position_m-dry.nodeMipOrigin.xyz)/(dry.mapping.cellSize*levelScale);let pageFloor=floor(virtualVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  let pageCoordinate=vec3u(pageFloor);
  if(dryGiPageCache.generation!=dry.nodeMip.x||dryGiPageCache.level!=level||any(dryGiPageCache.coordinate!=pageCoordinate)){
    dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,vec3u(0u),dry.nodeMip.x,0u,0xffffffffu,0u);let pageIndex=dryNodeMipFind(level,pageCoordinate);
    if(pageIndex!=0xffffffffu){
      if(!dryNodeMipPageValid(pageIndex)){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),0u,0u);}
      var black=2u;if(dryTetraRadiancePageValid(pageIndex)){black=textureLoad(tetraRadianceBlackPages,vec2u(pageIndex,0u),0).x;}
      if(dry.nodeMipDirect.w!=0u){let physical=u32(SVO_NODE_MIP_PHYSICAL_SIZE);let atlasPages=max(dry.nodeMipAtlas.xyz/vec3u(physical),vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,atlasPage*physical,dry.nodeMip.x,1u,pageIndex,black);}
      else{let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,pageIndex);dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,entry.pageOrigin,entry.generation,1u,pageIndex,black);}
    }
  }
  if(dryGiPageCache.resident==0u){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  if(!dryNodeMipPageValid(dryGiPageCache.pageIndex)){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),0u,0u);}
  let local=virtualVoxel-vec3f(pageCoordinate)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);
  let opacity=svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,dryGiPageCache.pageOrigin,local);
  if(!dryTetraRadianceReady()){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,0u);}
  let radianceLevel=max(level,dryTetraRadianceLevelFloor());
  var radianceIndex=dryGiPageCache.pageIndex;var radianceLocal=local;var radianceState=dryGiPageCache.blackRadiance;
  if(radianceLevel!=level){
    let radianceScale=exp2(f32(radianceLevel));
    let radianceVoxel=(query.position_m-dry.nodeMipOrigin.xyz)/(dry.mapping.cellSize*radianceScale);
    let radiancePageFloor=floor(radianceVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
    if(any(radiancePageFloor<vec3f(0.0))){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
    let radiancePage=vec3u(radiancePageFloor);
    if(dryGiRadiancePageCache.generation!=dry.nodeMip.x||dryGiRadiancePageCache.level!=radianceLevel||any(dryGiRadiancePageCache.coordinate!=radiancePage)){
      dryGiRadiancePageCache=DryNodeMipPageCache(radiancePage,radianceLevel,vec3u(0u),dry.nodeMip.x,0u,0xffffffffu,1u);
      let ancestor=dryNodeMipFind(radianceLevel,radiancePage);
      if(ancestor!=0xffffffffu){dryGiRadiancePageCache=DryNodeMipPageCache(radiancePage,radianceLevel,dryTetraRadiancePageOrigin(ancestor),dry.nodeMip.x,1u,ancestor,dryTetraRadianceState(ancestor));}
    }
    if(dryGiRadiancePageCache.resident==0u){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
    radianceIndex=dryGiRadiancePageCache.pageIndex;radianceState=dryGiRadiancePageCache.blackRadiance;
    radianceLocal=radianceVoxel-vec3f(radiancePage)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);
  }
  if(radianceState==2u){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,0u);}
  if(radianceState==1u){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  let radianceOrigin=select(dryTetraRadiancePageOrigin(radianceIndex),dryGiRadiancePageCache.pageOrigin,radianceLevel!=level);
  let uv=svoNodeMipAtlasUv(radianceOrigin,radianceLocal,textureDimensions(tetraRadianceLobe0));
  return SvoTetraRadianceConeSourceSample(opacity.solidMean,svoTetraSample(tetraRadianceLobe0,tetraRadianceLobe1,tetraRadianceLobe2,tetraRadianceLobe3,nodeMipSampler,uv),1u,1u);
}
struct DryGlobalIllumination{radiance:vec3f,visibility:f32,valid:u32}
${worldGiCacheHelpersWGSL}
fn dryGlobalIllumination(position:vec3f,normal:vec3f,ignoredBodyOwner:u32)->DryGlobalIllumination{
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)==0u){return DryGlobalIllumination(vec3f(0.0),1.0,1u);}
  if((dryDerivedPageFailure&${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u)!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}
  if(!dryTetraRadianceReady()){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}
  ${prepassGiShortcutWGSL}
  let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let origin=position+normalize(normal)*minimumVoxel*max(dry.tuningRays1.z,1.0);var indirect=vec3f(0.0);var visibility=0.0;
  let coneCount=clamp(u32(round(dry.giCones.y)),3u,4u);let perConeBudget=max(1u,min(64u,dry.tuningCounts0.y)/coneCount);
  for(var coneIndex=0u;coneIndex<4u;coneIndex+=1u){
    if(coneIndex>=coneCount){break;}let direction=svoTetraRadianceHemisphereDirection(normal,coneIndex,coneCount,0.0);
    dryGiPageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);
    dryGiRadiancePageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);
    let sceneExit=dryNodeMipSceneExitDistance(origin,direction);
    // The same reading the direct term needed: a zero-length interval means this
    // cone never enters the pyramid, so there is nothing along it to occlude
    // with and nothing to gather. Full visibility, no bounce — not the zero
    // transmittance an empty trace would otherwise report, which lands as
    // mix(1, 0, occlusionStrength) and dims every surface outside the domain to
    // a fifth of its ambient. On an infinite ground plane that draws the
    // container's footprint onto the floor as a bright rectangle.
    if(sceneExit<=0.0){visibility+=svoTetraRadianceHemisphereWeight(coneIndex,coneCount);continue;}
    var rigidHit=missHit();
    if(dryWorldGiIgnoreRigidBodies==0u){rigidHit=nearestBodyMaskIgnoring(origin,direction,ignoredBodyOwner,dryWorldGiBodyMask);}
    let rigidBlocked=rigidHit.t<sceneExit;
    let result=svoTetraRadianceConeTrace(SvoTetraRadianceConeConfig(origin,direction,dry.giCones.x,minimumVoxel,min(sceneExit,rigidHit.t),perConeBudget,.995,.0039215686,1u));
    let weight=svoTetraRadianceHemisphereWeight(coneIndex,coneCount);
    if(result.valid==0u||result.missingRadianceSamples!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}
    let finiteRadiance=all(result.radiance==result.radiance)&&all(abs(result.radiance)<vec3f(65504.0));
    let finiteVisibility=result.transmittance==result.transmittance&&abs(result.transmittance)<65504.0;
    if(!finiteRadiance||!finiteVisibility){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}
    indirect+=result.radiance*weight;
    let visibleThroughStatic=result.transmittance;
    visibility+=select(visibleThroughStatic,0.0,rigidBlocked)*weight;
  }
  let occlusionStrength=select(0.0,dry.giLighting.y,(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion}u)!=0u);
  return DryGlobalIllumination(max(indirect,vec3f(0.0))*dry.giLighting.x,mix(1.0,clamp(visibility,0.0,1.0),occlusionStrength),1u);
}
fn dryDiffuseMultiBounceVisibility(visibilityIn:f32,albedoIn:vec3f)->vec3f{
  let visibility=clamp(visibilityIn,0.0,1.0);let albedo=clamp(albedoIn,vec3f(0.0),vec3f(1.0));
  let a=2.0404*albedo-vec3f(0.3324);let b=-4.7951*albedo+vec3f(0.6417);let c=2.7552*albedo+vec3f(0.6903);
  return clamp(max(vec3f(visibility),((visibility*a+b)*visibility+c)*visibility),vec3f(0.0),vec3f(1.0));
}
${prepassDeclarationsWGSL}${splitDeclarationsWGSL}${voxelLightCacheWGSL}${worldGiCacheEntryWGSL}fn dryDiagnosticControl()->u32{return u32(round(max(uniforms.options.x,0.0)));}
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
const REQUIRED_FIELDS:u32 = ${SVO_DRY_SCENE_REQUIRED_VALID_FIELDS}u; // topology | scene geometry | material owner
const DRY_OWNER_NONE:u32=0xffffu;
const DRY_MEDIUM_GLASS:u32=2u;const DRY_MEDIUM_OPAQUE:u32=3u;
const DRY_GBUFFER_FIELD_ANALYTIC:u32=4u;const DRY_GBUFFER_FIELD_TERRAIN:u32=5u;
// SVO_GBUFFER_FIELD_SOURCES.structuralDiscrete — a cell face, not a solved
// surface. The voxels-only primary reports this for every opaque hit it makes.
const DRY_GBUFFER_FIELD_VOXEL:u32=1u;
const DRY_GBUFFER_MOTION_STATIC:u32=0u;const DRY_GBUFFER_MOTION_RIGID:u32=1u;
const DRY_GBUFFER_HARD_FEATURE:u32=256u;const DRY_GBUFFER_NO_INTERSECTION:u32=1u;
const DRY_GBUFFER_WORK_EXHAUSTED:u32=2u;const DRY_GBUFFER_INVALID_FIELD:u32=3u;
const DRY_REVERSED_Z_NEAR_M:f32=${SVO_DRY_SCENE_REVERSED_Z_NEAR_M};
var<private> dryVisibilityIgnoredOwner:u32;var<private> dryVisibilityStepInvalidReason:u32;var<private> dryThickGlassEnabled:u32;var<private> dryThickGlassFailure:u32;
var<private> dryWorldGiIgnoreRigidBodies:u32;
var<private> dryWorldGiBodyMask:u32=0xffffffffu;

fn dryConfiguredMapping()->SvoMapping{
  var mapping=dry.mapping;
  mapping.maxVisits=min(mapping.maxVisits,dryDiagnosticMaximumNodeVisits());
  return mapping;
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
/**
 * Whether a shadow ray has any sparse domain to cross at all.
 *
 * A zero exit distance says the segment from this receiver to this light never
 * enters the authored container — which is only ever true of a receiver outside
 * it, because a point inside always exits somewhere. The octree holds nothing
 * out there, so nothing can stand in the light's way, and the reading is *full
 * visibility*.
 *
 * Read as full shadow instead, which is what every call site here did, it
 * switched the sun off for every surface beyond the container footprint. On the
 * garden scenes the ground is an infinite plane and the container is 1.8 m
 * across, so the frame came back with a hard-edged lit rectangle sitting in a
 * dark field — legible as a floating slab, and not as the lighting bug it was.
 */
fn dryDirectionalRayLeavesDomain(maximumDistance:f32)->bool{return !(maximumDistance>0.0);}

fn terrainEnabled()->bool{return uniforms.terrainMeta.x>0.5&&dry.terrain.x!=0xffffffffu;}
// Sculpted ground arrives in the scene arena rather than in the uniform mirror:
// eight analytic features cannot express a brushed vessel, and the region is
// self-describing so nothing about it has to be threaded through the params
// uniform. A zeroed header is the encoding for "this scene is analytic", which
// is why the header is always allocated even when it is never written.
struct DryTerrainGrid{origin:vec2f,spacing:f32,slopeBound:f32,size:vec2i,minimum:f32,maximum:f32}
fn terrainGrid()->DryTerrainGrid{
  let header=drySceneWords4(DRY_SCENE_TERRAIN_WORD_OFFSET);let tail=drySceneWords4(DRY_SCENE_TERRAIN_WORD_OFFSET+4u);
  let origin=bitcast<vec4f>(header);let bounds=bitcast<vec4f>(tail);
  return DryTerrainGrid(origin.xy,origin.z,origin.w,vec2i(i32(tail.x),i32(tail.y)),bounds.z,bounds.w);
}
fn terrainSculpted(grid:DryTerrainGrid)->bool{return grid.size.x>=${MIN_TERRAIN_GRID_SIZE}&&grid.size.y>=${MIN_TERRAIN_GRID_SIZE};}
fn terrainGridSample(grid:DryTerrainGrid,i:i32,j:i32)->f32{
  return bitcast<f32>(drySceneArena[DRY_SCENE_TERRAIN_SAMPLE_WORD_OFFSET+u32(i+grid.size.x*j)]);
}
// Deliberately the same expression, in the same order, as sampleTerrainGrid in
// lib/terrain.ts -- including clamping the sample coordinate into [0, n-1]
// before the cell index is clamped into [0, n-2], and the final clamp to zero.
// The solver bakes its column heights through that function and the renderer
// samples this one, so the ground the water rests on and the ground that is
// drawn agree because they are the same arithmetic, not because both were
// tuned until they looked alike.
fn terrainGridHeightAt(grid:DryTerrainGrid,x:f32,z:f32)->f32{
  let fx=clamp((x-grid.origin.x)/grid.spacing,0.0,f32(grid.size.x-1));
  let fz=clamp((z-grid.origin.y)/grid.spacing,0.0,f32(grid.size.y-1));
  let x0=min(grid.size.x-2,i32(floor(fx)));let z0=min(grid.size.y-2,i32(floor(fz)));
  let tx=fx-f32(x0);let tz=fz-f32(z0);
  let lower=terrainGridSample(grid,x0,z0)*(1.0-tx)+terrainGridSample(grid,x0+1,z0)*tx;
  let upper=terrainGridSample(grid,x0,z0+1)*(1.0-tx)+terrainGridSample(grid,x0+1,z0+1)*tx;
  return max(0.0,lower*(1.0-tz)+upper*tz);
}
fn terrainHeightAt(x:f32,z:f32)->f32{
  if(!terrainEnabled()){return 0.0;}
  let grid=terrainGrid();
  if(terrainSculpted(grid)){return terrainGridHeightAt(grid,x,z);}
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
// Mirrors terrainCeiling in lib/terrain.ts. The analytic sum below is zero for
// a sculpted terrain -- its features survive only as provenance for the bake --
// so a grid must contribute its own tallest sample or the bracket starts 5 cm
// above the *base* ground and clips the whole coping away.
fn terrainCeiling()->f32{
  let grid=terrainGrid();
  if(terrainSculpted(grid)){return grid.maximum+${TERRAIN_CEILING_MARGIN_M};}
  var top=uniforms.terrainMeta.y+${TERRAIN_CEILING_MARGIN_M};let count=min(i32(round(uniforms.terrainMeta.z)),8);
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

// Sculpted ground is marched, not bracketed. The analytic path below samples the
// field at three points across the whole entry-to-floor interval and refines the
// first sign change; that is sound for a smooth mound and blind to a 5.5 cm
// coping 11 cm across, which fits inside a tenth of one of those steps. See
// SVO_TERRAIN_GRID_MARCH_STEPS for why the replacement provably cannot skip it,
// and marchSvoSculptedTerrain above for the CPU mirror this must agree with.
fn traceTerrainHeightfield(ro:vec3f,rd:vec3f,grid:DryTerrainGrid)->DryHit{
  let sceneScale=max(max(uniforms.container.x,uniforms.container.y),uniforms.container.z);
  let ceiling=grid.maximum+${TERRAIN_CEILING_MARGIN_M};var t0=0.005;
  if(ro.y>ceiling){if(rd.y>=-0.0005){return missHit();}t0=(ceiling-ro.y)/rd.y;}
  let span=t0+10.0*sceneScale;var t1=span;var bracketed=false;
  // Under the lowest sample the ray is beneath every column the grid can hold,
  // so the interval ends there rather than at the container floor -- and it ends
  // with the field non-positive, which is what makes the exhaustion path below a
  // bisection of a known bracket instead of a dropped pixel.
  if(rd.y<-0.0005){let lowestT=(grid.minimum-ro.y)/rd.y;if(lowestT>t0&&lowestT<=span){t1=lowestT;bracketed=true;}}
  else if(rd.y>0.0005){t1=min(t1,max(t0,(ceiling-ro.y)/rd.y));}
  if(t1<=t0){return missHit();}
  let rate=max(1e-4,abs(rd.y)+grid.slopeBound*length(rd.xz));
  let epsilon=max(1e-6,${SVO_TERRAIN_GRID_EPSILON_SAMPLES}*grid.spacing);
  var t=t0;var lastOutside=t0;var lastOutsideField=0.0;
  for(var step=0;step<${SVO_TERRAIN_GRID_MARCH_STEPS};step+=1){
    if(t>t1){break;}
    let field=terrainField(ro,rd,t);
    // The acceptance window is vertical, and a vertical tolerance is a long one
    // along a shallow ray. One guarded secant step on the two samples that
    // closed it removes that amplification wherever the surface is locally well
    // conditioned, for one extra evaluation.
    if(abs(field)<=epsilon){
      if(t<=lastOutside){return terrainHitAt(ro,rd,t);}
      let slope=(field-lastOutsideField)/(t-lastOutside);
      if(abs(slope)<=1e-6){return terrainHitAt(ro,rd,t);}
      let corrected=clamp(t-field/slope,t0,t1);
      if(abs(terrainField(ro,rd,corrected))<abs(field)){return terrainHitAt(ro,rd,corrected);}
      return terrainHitAt(ro,rd,t);
    }
    lastOutside=t;lastOutsideField=field;
    t=t+abs(field)/rate;
  }
  if(!bracketed||lastOutsideField<=0.0){return missHit();}
  var a=lastOutside;var b=t1;
  for(var refinement=0;refinement<${SVO_TERRAIN_GRID_REFINEMENTS};refinement+=1){
    let middle=0.5*(a+b);if(terrainField(ro,rd,middle)>0.0){a=middle;}else{b=middle;}
  }
  return terrainHitAt(ro,rd,0.5*(a+b));
}

// Ordinary camera rays use at most 8 intersection height evaluations plus the
// four central-difference normal samples. Only unresolved shallow/grazing rays
// pay for the smaller graded fallback. Both paths return the first bracket.
fn traceTerrain(ro:vec3f,rd:vec3f)->DryHit{
  ${voxelsOnlyPrimary ? /* wgsl */ `
  // Terrain is voxels like everything else. The heightfield is still authored
  // and still voxelized — 73 667 cells of material 2 on the hero garden, the
  // largest single population in the lane — so killing the analytic surface
  // removes a duplicate, not the ground.
  //
  // It was a duplicate that actively hid its twin: the shell has no owner, so
  // the old owner-gated payload walk dropped every terrain cell, and what
  // reached the screen was this function alone. Fixing the solidity test made
  // both draw at once and the ground turned to floating cubes. One of them had
  // to go, and the analytic one is the one that cannot get better.
  //
  // Returning a miss here rather than editing the call sites is deliberate:
  // shading, the raster background pass and shadow visibility all route through
  // this function, and a terrain that is invisible but still occludes would be
  // a worse bug than either surface alone.
  return missHit();` : ""}
  if(!terrainEnabled()){return missHit();}
  let sculpted=terrainGrid();
  if(terrainSculpted(sculpted)){return traceTerrainHeightfield(ro,rd,sculpted);}
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

// The same segment/sphere test bodyBoundingSphereVisible performs, against one
// published sphere enclosing every body rather than against a body. It is the
// guard that keeps shadow and contact rays from reading the body array at all,
// so it must stay conservative: a negative radius means the scene has no bodies,
// and the epsilon matches the per-body test so the two can never disagree about
// a grazing ray.
fn svoRigidBoundsIntersect(ro:vec3f,rd:vec3f,tMax:f32)->bool{
  let radius=dry.rigidBounds.w;
  if(radius<0.0){return false;}
  let centre=dry.rigidBounds.xyz;let offset=centre-ro;let projected=clamp(dot(offset,rd),0.0,tMax);
  let closest=ro+rd*projected;let bound=radius+1e-5;
  return dot(closest-centre,closest-centre)<=bound*bound;
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
  let exact=svoIntersectPrimitiveExact(record,ro,rd,max(tMin,1e-4),tMax,dryClusterPacking(record));
  if(exact.status!=SVO_PRIMITIVE_RAY_HIT){return missHit();}
  return DryHit(exact.t_m,exact.normal.xyz,svoPrimitiveMaterialId(record),svoPrimitiveOwnerId(record),exact.featureId,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
}

// Exact live-scene acceleration. Candidate nodes occupy the same fixed arena
// as primitives, after the primitive span, so this adds no storage binding.
fn traceScenePrimitives(ro:vec3f,rd:vec3f,tMin:f32,tMax:f32,ignoredOwner:u32)->DryHit{
  var best=missHit();best.t=tMax;
  if(dry.primitiveCandidates.w==0u||dry.primitiveCandidates.y==0u){return best;}
  var stack:array<u32,${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK}>;
  var stackSize=1u;stack[0]=dry.primitiveCandidates.z;
  for(var visit=0u;visit<${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES}u&&stackSize>0u;visit+=1u){
    stackSize-=1u;let nodeIndex=stack[stackSize];
    if(nodeIndex>=dry.primitiveCandidates.y){continue;}
    let node=dryPrimitive(dry.primitiveCandidates.x+nodeIndex);
    let interval=dryBoundsInterval(bitcast<vec3f>(node.centerKind.xyz),bitcast<vec3f>(node.dimensionsIdentity.xyz),ro,rd,tMin,best.t);
    if(interval.valid==0u){continue;}
    let leftOrPrimitive=node.centerKind.w;let right=node.dimensionsIdentity.w;
    if(right==0xffffffffu){
      if(leftOrPrimitive>=dry.metadata.x){continue;}let record=dryPrimitive(leftOrPrimitive);let owner=svoPrimitiveOwnerId(record);
      if(owner==ignoredOwner||dryOpaqueOwnerSuppressed(owner)){continue;}
      let candidate=primitiveHit(record,ro,rd,tMin,best.t);if(candidate.t<best.t){best=candidate;}
    }else if(stackSize+2u<=${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK}u){
      stack[stackSize]=right;stack[stackSize+1u]=leftOrPrimitive;stackSize+=2u;
    }
  }
  return best;
}

// The outward normal of the face a point sits on, for an axis-aligned cell.
//
// Called with the DDA's entry point, which lies exactly on one face, so the
// axis whose distance to a bound is zero is the face the ray came in through
// and wins the minimum outright. A point at a cell corner ties between two
// legal faces and either answer is a real face of a real cell.
//
// Distinct from dryPrimaryProxyNormal, which answers the same way but is only
// compiled when screen-space termination is on; this one is always here because
// the voxels-only primary is always here.
fn dryVoxelFaceNormal(bounds:mat2x3f,point:vec3f)->vec3f{
  let faceDistance=min(abs(point-bounds[0]),abs(bounds[1]-point));var axis=0u;
  if(faceDistance.y<faceDistance.x){axis=1u;}if(faceDistance.z<faceDistance[axis]){axis=2u;}
  var normal=vec3f(0.0);
  normal[axis]=select(-1.0,1.0,abs(point[axis]-bounds[1][axis])<abs(point[axis]-bounds[0][axis]));
  return normal;
}
${surfaceReconstructionWGSL}
fn traceLeafPayload(ro:vec3f,rd:vec3f,hit:SvoTraversalHit)->DryHit {
  ${primaryBrickSetupWGSL}
  ${primaryLodStrideWGSL}
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0)); let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9)); let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));
  let tolerance=length(extent)*1.05;var upgrades=0u;
  // The voxel is the accelerator and the authored surface is the authority.
  //
  // A solid voxel names its owning record, and the record's own exact
  // intersection is then taken over just that interval, so what this returns is
  // a point on the real analytic surface rather than a cell face. It is an upper
  // bound on the true nearest hit and never a wrong one: every candidate it
  // produces is a genuine surface of a genuine record, so a resolve seeded with
  // it can only terminate earlier, never wrongly. That is the whole promotion —
  // resolution independence is retained exactly where it is visible, and the
  // search that finds the surface gets cheaper.
  //
  // What is upgraded is a *run* of consecutive cells naming the same owner, not
  // a cell. The exact test is a first-hit march, so marching one owner once over
  // the union of its cells' intervals returns exactly what marching each cell in
  // DDA order returns — the intervals are contiguous and their padded union is
  // the padded run — while paying one bounding-sphere clamp and one march ramp
  // instead of k of them. Per-cell upgrades re-marched the same three-octave
  // lattice field once per 25 mm of the same aggregate, and each one that
  // *missed* paid the full 48-iteration ceiling to discover it: the miss is the
  // expensive case, and a run collapses a run-length's worth of them into one.
  //
  // The budget therefore counts owners along the ray rather than voxels, which
  // is what decouples it from crown accuracy — see
  // SVO_DRY_VOXEL_OWNER_UPGRADES_PER_LEAF.
  //
  // Not the published voxel distance lane, and that is a measured choice rather
  // than an omission. The field is 1-Lipschitz, so a surface point satisfies
  // |p - c| >= d(c) and the only *sound* rejection is "the whole interval this
  // call is given lies inside the ball of radius d(c)" — which, against an
  // interval padded by 1.05 |extent| at both ends, needs d(c) above roughly two
  // and a third cells. Censused over the hero's 82 975 solid scene voxels
  // (tmp/w6-distance-census.ts): median -4.93 cells, p99 +1.11, max +1.70. Zero
  // voxels clear the sound threshold; 2.1 % clear an unsound cell-local variant.
  // A predicate that rejects nothing while adding a payload fetch per cell is a
  // cost, and a cell-local one would be the quality dial again under a new name.
  var runOwner=${DRY_UPGRADE_RUN_NONE}u;var runStart=0.0;var runEnd=0.0;
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(i32(dry.mapping.brickSize)))||entry>${primaryBrickExitWGSL}){break;}
    ${primaryMacroSkipWGSL}
    let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,${primaryBrickExitWGSL}));
    var cellOwner=${DRY_UPGRADE_RUN_NONE}u;var cellIdentity=0u;var cellSolid=false;
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex<arrayLength(&materialOwners)${voxelOwnerPrimary ? "" : "&&false"}){
      let identity=materialOwners[payloadIndex];let owner=identity>>16u;
      // A material is what makes a cell solid; an owner is what makes it
      // *solvable*. Those are not the same test, and conflating them is only
      // invisible while something else draws the surface.
      //
      // The owner-range test below exists to name an authored record for the
      // analytic march. Under voxels-only there is no march, so requiring it
      // discards every cell the voxeliser filled without an in-range owner —
      // which on the hero garden is most of the scenery. It cost nothing before
      // because a rejected cell merely skipped an upgrade and the analytic tier
      // drew the prop anyway; with that tier gone the same rejection is a hole.
      // The raw inspection lane's own solidity test is exactly material != 0
      // (see resolveMaterial in webgpu-octree-sparse-bricks.ts), and it draws
      // 86 % of this frame from the same buffer this walk was missing.
      //
      // Suppression still applies, but only for an owner that names something:
      // SPARSE_BRICK_NO_OWNER collides with DRY_OWNER_NONE and would otherwise
      // suppress every ownerless cell in the scene.
      let suppressed=owner!=${SPARSE_BRICK_NO_OWNER}u&&dryOpaqueOwnerSuppressed(owner);
      if((identity&0xffffu)!=0u&&!suppressed){cellSolid=true;cellIdentity=identity;}
      // Air is packMaterialOwner(0, SPARSE_BRICK_NO_OWNER); a brick that exists
      // but was never voxelized is all zeroes. Both must miss, and the zero one
      // would otherwise resolve to record zero in every empty cell of the scene.
      if(identity!=0u&&owner!=${SPARSE_BRICK_NO_OWNER}u&&owner>=dry.metadata.y&&!dryOpaqueOwnerSuppressed(owner)){
        let primitiveIndex=owner-dry.metadata.y;
        if(primitiveIndex<dry.metadata.x){${screenSpaceVoxelResolveWGSL}cellOwner=primitiveIndex;}
      }
    }
    ${primaryVoxelSurfaceWGSL}
    if(${primaryUpgradeRunMergeWGSL}){runEnd=cellExit;}
    else{
      ${primaryUpgradeFlushWGSL}
      // Nothing after an exhausted budget can produce a hit, and the DDA is not
      // free: stop stepping rather than walking the rest of the brick for a
      // decision that has already been made.
      if(upgrades>=${primaryUpgradeBudget}u){break;}
      runOwner=cellOwner;runStart=entry;runEnd=cellExit;
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z)); if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  ${primaryUpgradeFlushWGSL}
  return missHit();
}
${macroHddaPrimaryWGSL}

// The voxel-resolved surface first, the analytic set bounded by it second.
//
// Both tiers answer the same question and the frame keeps whichever is nearer,
// so the order between them is free to choose — and it is not free to get
// wrong. The analytic tier is one candidate-BVH walk whose node rejection is
// dryBoundsInterval(..., tMin, best.t): seeded with DRY_MISS it descends every
// box the ray touches anywhere along its whole length, which costs one interval
// test per record standing behind the first surface and is exactly the term
// that made this path scale with authored record count. Seeded with the voxel
// hit it descends only the boxes in front of a surface already found.
//
// The image is unchanged by construction: the walk still reports the nearest
// analytic hit that is nearer than the voxel one, and a walk that finds nothing
// returns its own tMax sentinel, which the comparison below reads as "no nearer
// analytic surface" rather than as a hit.
//
// Measured on the hero at 800x460, cone scale 0.5, all arms interleaved in one
// process (serialized submit-to-fence): 292.5 -> 222.0 ms at 501 records and
// 1564.4 -> 546.3 ms at 5 039. It is the single largest term in the 10x gap.
fn traceStatic(ro:vec3f,rd:vec3f)->DryHit {
  // An unpublished scene has no voxels, and voxels are the only surface. Drawing
  // it analytically here would hide exactly the failure worth seeing: a frame
  // that looks perfect because the voxel path never ran. Miss instead and let
  // the existing publication tripwire say why.
  if(dryPublicationWord(0u)==0u||(dryPublicationWord(1u)&REQUIRED_FIELDS)!=REQUIRED_FIELDS){${voxelsOnlyPrimary ? "return missHit();" : "return traceScenePrimitives(ro,rd,0.0,DRY_MISS,DRY_OWNER_NONE);"}}
  ${analyticPrimaryUnbounded ? "let seeded=traceScenePrimitives(ro,rd,0.0,DRY_MISS,DRY_OWNER_NONE);" : "let seeded=missHit();"}
  var voxel=missHit();
  var minimum=0.0;
  let mapping=dryConfiguredMapping();
  let leafBudget=clamp(dry.tuningCounts0.x,1u,${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u);
  var continuation:DryTraversalCursor;
  var traversalFinished=false;
  dryTraversalCursorBegin(SvoRay(ro,minimum,rd,DRY_MISS),mapping,&continuation);
  for(var leafVisit=0u;leafVisit<${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u&&leafVisit<leafBudget;leafVisit+=1u){
    let ray=SvoRay(ro,minimum,rd,DRY_MISS);
    let leaf=dryTraversalCursorNextPrimary(ray,mapping,&continuation);

    ${screenSpaceProxyTraceWGSL}
    if(leaf.status!=SVO_STATUS_HIT){
      traversalFinished=true;
      if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){}
      else if(leaf.status!=SVO_STATUS_MISS){}
      break;
    }
    if(!dryLeafCurrent(leaf)){minimum=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);continue;}


    let payloadHit=${primaryLeafTraceCallWGSL}(ro,rd,leaf);
    // Leaves partition space and are visited front to back, so the first payload
    // hit is the nearest voxel-resolved surface and no later leaf can beat it.
    if(payloadHit.t<seeded.t){voxel=payloadHit;break;}

    minimum=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);
  }
  // Reaching the uniform budget without an authoritative hierarchy miss is a
  // traversal exhaustion, not an empty scene. Keep that visible in the
  // existing failure heatmap instead of silently returning black.
  if(!traversalFinished){}
  ${analyticPrimaryUnbounded
    ? "if(voxel.t<seeded.t){return voxel;}return seeded;"
    : voxelsOnlyPrimary
      ? "return voxel;"
      : "let live=traceScenePrimitives(ro,rd,0.0,voxel.t,DRY_OWNER_NONE);if(live.t<voxel.t){return live;}return voxel;"}
}

struct DryGlassHit{hit:SvoThinGlassHit,recordIndex:u32}
fn dryGlassMiss()->DryGlassHit{return DryGlassHit(svoThinGlassMiss(),0u);}
fn dryGlassBoundingSphereVisible(record:SvoThinGlassRecord,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->bool{
  let offset=record.centerThickness.xyz-ro;let projected=clamp(dot(offset,rd),tMin,tMax);let closest=ro+rd*projected;let radius=length(vec3f(record.extentIorEpsilon.xy,.5*record.centerThickness.w))+record.extentIorEpsilon.w+1e-5;
  return dot(closest-record.centerThickness.xyz,closest-record.centerThickness.xyz)<=radius*radius;
}
fn traceGlass(ro:vec3f,rd:vec3f,tMin_m:f32,tMax_m:f32,skipCompositeOwned:bool)->DryGlassHit {
  var best=dryGlassMiss();var bestT=tMax_m;
  let paneCount=min(dry.terrain.y,${SVO_SCENE_GLASS_MAXIMUM_PANES}u);
  for(var paneIndex=0u;paneIndex<${SVO_SCENE_GLASS_MAXIMUM_PANES}u;paneIndex+=1u){
    if(paneIndex>=paneCount){break;}let record=dryGlassPane(paneIndex);let paneId=svoThinGlassPaneId(record);let compositeOwned=skipCompositeOwned&&dry.terrain.w>0u&&paneId>=dry.terrain.z&&paneId-dry.terrain.z<dry.terrain.w;let thickReplaced=dryThickGlassEnabled!=0u&&paneId==thickGlass.metadata.z;if(compositeOwned||thickReplaced||!dryGlassBoundingSphereVisible(record,ro,rd,tMin_m,bestT)){continue;}let candidate=svoThinGlassIntersect(record,ro,rd,tMin_m,bestT,1e-6,record.extentIorEpsilon.w);
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
    // Live analytic owners are resolved once through the exact BVH below;
    // stale SVO ownership can therefore never resurrect an old transform.
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
  if(dryPublicationWord(0u)==0u||(dryPublicationWord(1u)&REQUIRED_FIELDS)!=REQUIRED_FIELDS){dryVisibilityStepInvalidReason=1u;return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,0u,DRY_MISS);}
  if(dry.terrain.y>${SVO_SCENE_GLASS_MAXIMUM_PANES}u){dryVisibilityStepInvalidReason=2u;return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,0u,DRY_MISS);}
  var nodeVisits=0u;var leafVisits=0u;var workItems=0u;var bestT=ray.tMax_m;var found=false;var opaque=true;var glassTransmission=vec3f(0.0);

  let bodyCount=min(u32(round(max(uniforms.options.z,0.0))),12u);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}if(bodyIndex==dryVisibilityIgnoredOwner){continue;}if(workItems>=remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=1u;
    let body=bodies[bodyIndex];if(!bodyBoundingSphereVisible(ray.origin_m,ray.direction,body,tMin_m,bestT)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ray.origin_m,ray.direction,body,tMin_m,bestT)){continue;}let candidate=bodyHit(ray.origin_m,ray.direction,body);if(candidate.t>=tMin_m&&candidate.t<bestT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,candidate.t);}
  }

  // The voxel-resolved tier first, the analytic set bounded by it second.
  //
  // The same reordering traceStatic took, on the path that kept the old order.
  // Every shadow, AO and GI ray used to open with a candidate-BVH walk bounded
  // only by its own tMax — for a shadow ray that is the whole distance to the
  // light — so each one paid an interval test per record standing anywhere
  // along it. That is the identical term that made the primary scale with
  // authored record count, and the primary shedding it is what promoted this
  // one to the largest remaining residual.
  //
  // Semantics are preserved rather than approximated. Both orders return the
  // nearer of {analytic, voxel}; both let an analytic win fall through to the
  // terrain and glass tail and both let a voxel win short-circuit it. What
  // changes is only which tier supplies the other's bound. A walk that now runs
  // past where the analytic hit used to stop it can only find a farther voxel,
  // which the bounded analytic call then still beats.
  //
  // Exhaustion is the one place the reorder could have lost an answer, and it
  // gains one instead. The old order discarded the analytic hit on every
  // exhausted ray anyway (each exhausted return carries DRY_MISS), but a ray
  // bounded by a near analytic surface often terminated before it could exhaust
  // at all. So the walk reports how far it got, the analytic call is bounded by
  // that reach, and a hit nearer than the reach is returned as a genuine HIT:
  // nothing the unfinished walk could still find lies in front of it.
${analyticVisibilityUnbounded ? "" : "  var voxelT=DRY_MISS;var walkExhausted=false;\n"}${analyticVisibilityUnbounded ? `  let livePrimitive=traceScenePrimitives(ray.origin_m,ray.direction,tMin_m,bestT,dryVisibilityIgnoredOwner);
  if(livePrimitive.t<bestT){bestT=livePrimitive.t;found=true;opaque=true;}
` : ""}
  var cursor=max(tMin_m,0.0);var shadowContinuation:DryTraversalCursor;let initialShadowMapping=dryConfiguredMapping();dryTraversalCursorBegin(SvoRay(ray.origin_m,cursor,ray.direction,bestT),initialShadowMapping,&shadowContinuation);
  for(var leafAttempt=0u;leafAttempt<${SVO_VISIBILITY_LIMITS.leafVisits}u;leafAttempt+=1u){
    if(cursor>=bestT){break;}if(leafVisits>=remaining.leafVisits||nodeVisits>=remaining.nodeVisits){${visibilityExhaustedWGSL}}
    var shadowMapping=dryConfiguredMapping();shadowMapping.maxVisits=min(shadowMapping.maxVisits,remaining.nodeVisits-nodeVisits);
    let leaf=dryTraversalCursorNext(SvoRay(ray.origin_m,cursor,ray.direction,bestT),shadowMapping,&shadowContinuation);nodeVisits+=leaf.visits;
    if(leaf.status==SVO_STATUS_MISS){break;}
    if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){${visibilityExhaustedWGSL}}
    if(leaf.status!=SVO_STATUS_HIT){dryVisibilityStepInvalidReason=3u;return dryVisibilityStep(SVO_VIS_STEP_INVALID,nodeVisits,leafVisits,workItems,DRY_MISS);}leafVisits+=1u;
    if(!dryLeafCurrent(leaf)){cursor=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);continue;}
    let payloadRay=SvoVisibilityRay(ray.origin_m,bestT,ray.direction,ray.originBias_m);let payload=${shadowLeafTraceCallWGSL}(payloadRay,tMin_m,leaf,remaining.workItems-workItems);workItems+=payload.workItems;
    if(payload.status==SVO_VIS_STEP_HIT){${visibilityPayloadHitWGSL}}if(payload.status!=SVO_VIS_STEP_MISS){if(payload.status==SVO_VIS_STEP_INVALID){dryVisibilityStepInvalidReason=3u;}return dryVisibilityStep(payload.status,nodeVisits,leafVisits,workItems,payload.t_m);}
    cursor=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);
  }
${analyticVisibilityUnbounded ? `  if(cursor<bestT&&leafVisits>=remaining.leafVisits){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
` : `  // Only when the walk produced no hit of its own: a payload hit leaves the
  // cursor short of bestT by construction and is not an exhaustion.
  if(voxelT>=DRY_MISS&&cursor<bestT&&leafVisits>=remaining.leafVisits){walkExhausted=true;}
  // Anything an unfinished walk could still find lies at or beyond the cursor.
  let analyticReach=select(min(bestT,voxelT),min(bestT,cursor),walkExhausted);
${analyticVisibilityProbeOff
  ? "  var livePrimitive=missHit();livePrimitive.t=analyticReach;"
  : "  let livePrimitive=traceScenePrimitives(ray.origin_m,ray.direction,tMin_m,analyticReach,dryVisibilityIgnoredOwner);"}
  if(livePrimitive.t<analyticReach){bestT=livePrimitive.t;found=true;opaque=true;}
  else if(walkExhausted){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
  else if(voxelT<bestT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,voxelT);}
`}

  if(terrainEnabled()){
    let terrainWork=${SVO_TERRAIN_FALLBACK_STEPS + SVO_TERRAIN_FALLBACK_REFINEMENTS + 6}u;
    if(workItems+terrainWork>remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=terrainWork;
    let terrain=traceTerrain(ray.origin_m,ray.direction);if(terrain.t>=tMin_m&&terrain.t<bestT){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,terrain.t);}
  }
  let paneCount=dry.terrain.y;if(workItems+paneCount>remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=paneCount;
  // The water compositor already owns the vessel panes' visible dielectric
  // treatment. Skip those same panes in lighting visibility so they cannot
  // project a bright/dark tank-shaped cutout onto nearby dry surfaces. Other
  // authored glazing remains in the query and keeps bounded transmission.
  let glass=traceGlass(ray.origin_m,ray.direction,tMin_m,bestT,true);if(glass.hit.valid!=0u&&glass.hit.t_m<bestT){let optics=svoThinGlassOptics(dryGlassPane(glass.recordIndex),glass.hit,dryThinGlassIncidentIor());bestT=glass.hit.t_m;found=true;opaque=false;glassTransmission=optics.netTransmittance;}
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
  if((dryDerivedPageFailure&${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u)!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.directVisibilityPage}u;return vec3f(0.0);}
  let maximumDistance=select(directionalLightSceneExitDistance(position,towardLight),finiteDistance_m,finiteDistance_m>0.0);if(dryDirectionalRayLeavesDomain(maximumDistance)){return vec3f(1.0);}
  let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);
  ${voxelLightCacheShortcutWGSL}
  // Direct light remains an analytic contribution from the current live light
  // arena. The derived radiance atlas contains emitted energy and supplements
  // this term; it is never a baked replacement for current scene lighting.
  // Reuse hierarchical cone visibility for the analytic direct term instead of
  // recasting a full exact SVO ray for every receiver and every authored light.
  // Reduced shading retains its full-rate analytic rigid-body correction in
  // prepassShadowShortcutWGSL. Cone mode has no undeclared exact escape: an
  // unavailable requested page publishes a typed fail-closed diagnostic.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u){
    if(!dryNodeMipReady()){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.directVisibilityPage}u;return vec3f(0.0);}${prepassShadowShortcutWGSL}
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
    if(cone.valid==0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.directVisibilityPage}u;return vec3f(0.0);}
    let rigidBlocker=nearestBodyIgnoring(ray.origin_m,towardLight,ownerId);if(rigidBlocker.t<ray.tMax_m){return vec3f(1.0-dry.tuningRays0.y);}let raw=vec3f(cone.transmittance)*dryFluidTransmittance(cone.fluidDepth_m);return mix(vec3f(1.0),raw,dry.tuningRays0.y);
  }
  dryVisibilityIgnoredOwner=ownerId;
  let result=svoTraceVisibility(ray,SvoVisibilityBudget(dry.tuningCounts1.w,dry.tuningCounts2.x,dry.tuningCounts2.y,dry.tuningCounts2.z),true,0.001,max(ray.originBias_m,1e-6));if(result.status==SVO_VIS_STATUS_EXHAUSTED){}else if(result.status==SVO_VIS_STATUS_INVALID){}
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
  if((dryDerivedPageFailure&${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u)!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.ambientOcclusionPage}u;return vec3f(0.0);}
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u){
    if(!dryNodeMipReady()){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.ambientOcclusionPage}u;return vec3f(0.0);}${prepassContactShortcutWGSL}
    let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(1.0);}var visibility=0.0;let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let origin=position+normalize(geometricNormal)*cellScale*.2;let coneSampleCount=select(dry.tuningCounts1.z,dry.tuningCounts1.y,${SVO_DRY_SCENE_CAMERA_SETTLED_WGSL});
    for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=coneSampleCount){break;}let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);let rotated=select(direction,normalize(direction+cross(normalize(geometricNormal),direction)*.7),sampleIndex>=2u);let cone=dryConeVisibility(origin,rotated,dry.tuningRays1.x,radius,vec3f(0.0),false);if(cone.valid==0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.ambientOcclusionPage}u;return vec3f(0.0);}let rigidBlocker=nearestBodyIgnoring(origin,rotated,ownerId);visibility+=select(cone.transmittance,0.0,rigidBlocker.t<radius);}let raw=clamp(visibility/f32(coneSampleCount),0.0,1.0);return vec3f(mix(1.0,raw,dry.tuningRays0.w));
  }
  if((dry.materialPublication.w&1u)==0u){return vec3f(1.0);}
  let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(0.0);}let biasCells=select(${SVO_CONTACT_VISIBILITY_CONTRACT.smoothBiasCells},${SVO_CONTACT_VISIBILITY_CONTRACT.hardFeatureBiasCells},featureId!=SVO_FEATURE_SMOOTH);var visibility=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}u;sampleIndex+=1u){let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex);let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,direction,radius,dry.mapping.cellSize,biasCells);let result=svoTraceVisibility(ray,SvoVisibilityBudget(dry.tuningCounts1.w,dry.tuningCounts2.x,dry.tuningCounts2.y,dry.tuningCounts2.z),true,0.001,max(ray.originBias_m,1e-6));if(result.status==SVO_VIS_STATUS_INVALID||result.status==SVO_VIS_STATUS_EXHAUSTED){return vec3f(0.0);}visibility+=result.transmittance;}
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
  return index<dry.materialPublication.x&&svoMaterialValid(material,index)&&material.identity.y==dry.materialPublication.y&&(material.identity.w&SVO_MATERIAL_FLAG_OPAQUE)!=0u;
}
// Stable adapter point for M7's pending G-buffer: material identity remains on
// DryHit, while procedural region/variation identity is evaluated exactly once
// from the same world-space hit used for the PBR closure.
fn dryEvaluateSurfaceMaterial(hit:DryHit,position:vec3f)->DrySurfaceMaterial {
  var materialId=dryResolvedMaterialId(hit);var baseOverride=vec3f(0.0);var useBaseOverride=false;var selectedEmission=vec3f(0.0);
  if((hit.materialId&0x80000000u)!=0u){let body=bodies[hit.materialId&0x7fffffffu];baseOverride=body.colorSelected.xyz;useBaseOverride=true;selectedEmission=body.colorSelected.w*vec3f(.12,.42,.32);}
  if(materialId>=dry.materialPublication.x){return dryInvalidSurfaceMaterial();}let material=dryMaterial(materialId);if(!dryPublishedMaterialValid(material,materialId)){return dryInvalidSurfaceMaterial();}
  var base=select(material.baseColorOpacity.xyz,baseOverride,useBaseOverride);var roughness=material.emissiveRoughness.w;var regionId=DRY_SURFACE_REGION_NONE;var variationFlags=0u;
  ${neutralSurfaceAlbedo ? /* wgsl */ `
  // Clay at 0.8, not 1.0. A unit-albedo surface clips to flat white under the
  // key light and loses precisely the shading this mode exists to show; 0.8 is
  // where plaster and porcelain actually sit, and it keeps a highlight's
  // rolloff on the curve instead of against the ceiling.
  //
  // The selection tint survives: it is a cursor affordance, not a material.
  // So does emission below, because an emitter is a light source and removing
  // it would change how the room is lit rather than what colour things are.
  base=select(vec3f(0.8),baseOverride,useBaseOverride);roughness=0.65;` : /* wgsl */ `
  let terrainPolicyValid=material.identity.z==SVO_MATERIAL_FUNCTION_GARDEN_TERRAIN&&dry.terrainMaterial.policyVersion==1u&&dry.terrainMaterial.materialId==materialId&&materialId==dry.terrain.x;
  if(terrainPolicyValid){let terrainSample=svoTerrainMaterial(dry.terrainMaterial,position,hit.normal);base=terrainSample.colorLinear;regionId=terrainSample.regionId;variationFlags=terrainSample.variationFlags;}
  else{let procedural=svoProceduralMaterial(material.identity.z,base,roughness,position);base=procedural.baseColorLinear;roughness=procedural.roughness;variationFlags=procedural.variationFlags;}`}
  return DrySurfaceMaterial(base,roughness,material.emissiveRoughness.xyz+selectedEmission,material.surface.x,vec3f(svoMaterialDielectricF0(material)),material.surface.y,regionId,variationFlags,1u,0u);
}
/**
 * The hover outline.
 *
 * A rim rather than a tint or a wireframe: it reads on a white porcelain
 * mushroom and on a near-black lab bench alike, it does not lie about the
 * object's colour, and — unlike a screen-space outline — it is occluded by
 * whatever is genuinely in front, so a half-hidden object looks half hidden.
 *
 * Additive, so an object already at white does not saturate into a silhouette,
 * and applied after shading so nothing feeding global illumination sees it: a
 * cursor must not change how the room is lit.
 */
fn dryHoverRim(color:vec3f,hit:DryHit,viewDirection:vec3f)->vec3f {
  let first=uniforms.highlight.x;let last=uniforms.highlight.y;let strength=uniforms.highlight.z;
  // An empty range is the resting state; nothing is hovered far more often
  // than something is.
  if(!(strength>0.0)||!(last>=first)){return color;}
  if(hit.ownerId==DRY_OWNER_NONE){return color;}
  let owner=f32(hit.ownerId);
  if(owner<first-0.5||owner>last+0.5){return color;}
  let facing=1.0-clamp(abs(dot(hit.normal,viewDirection)),0.0,1.0);
  return color+pow(facing,max(uniforms.highlight.w,1.0))*strength*vec3f(.32,.86,.72);
}

fn shadeDryOpaque(hit:DryHit,ro:vec3f,rd:vec3f)->vec3f {
  if(hit.t>=DRY_MISS){return dryEnvironment(rd,0.0);}${screenSpaceProxyShadeWGSL}${prepassRadianceShortcutWGSL}${voxelLightCache ? "dryVoxelLightConsumerEligible=select(0u,1u,hit.motionKind==DRY_GBUFFER_MOTION_STATIC);" : ""}let position=ro+rd*hit.t;let surface=dryEvaluateSurfaceMaterial(hit,position);
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
  let viewDirection=normalize(-rd);let reflected=reflect(rd,hit.normal);let diffuseColor=surface.baseColor*(1.0-surface.metallic);let f0=mix(surface.specularF0*surface.specularWeight,surface.baseColor,surface.metallic);let environmentBrdf=unifiedEnvironmentBrdf(max(dot(hit.normal,viewDirection),0.0),surface.roughness,f0);let diffuseEnergy=max(vec3f(0.0),vec3f(1.0)-environmentBrdf);let contactVisibility=dryContactVisibility(position,hit.normal,hit.featureId,hit.ownerId);let ignoredBodyOwner=select(DRY_OWNER_NONE,hit.ownerId,hit.motionKind==DRY_GBUFFER_MOTION_RIGID);let gi=dryGlobalIllumination(position,hit.normal,ignoredBodyOwner);let diffuseVisibility=dryDiffuseMultiBounceVisibility(gi.visibility,diffuseColor);let diffuseEnvironmentScale=select(1.0,dry.giLighting.z,globalIllumination);let directScale=dry.giLighting.w;let diffuseEnvironment=diffuseColor*diffuseEnergy*svoEnvironmentDiffuseIrradiance(dryLighting.environment,hit.normal)*contactVisibility*diffuseVisibility*diffuseEnvironmentScale/UNIFIED_PI;let specularEnvironment=dryEnvironment(reflected,surface.roughness)*environmentBrdf;let indirectDiffuse=diffuseColor*gi.radiance;
  var shaded=max(surface.emissive+diffuseEnvironment+specularEnvironment+direct*directScale+indirectDiffuse,vec3f(0.0));
  return dryHoverRim(shaded,hit,viewDirection);
}
struct DryGlassSurface{color:vec3f,depth:f32,materialId:u32,ownerId:u32,paneId:u32,_padding:u32}
fn shadeThinGlass(glass:DryGlassHit,opaque:DryHit,ro:vec3f,rd:vec3f)->DryGlassSurface {
  let record=dryGlassPane(glass.recordIndex);let incidentIor=dryThinGlassIncidentIor();let optics=svoThinGlassOptics(record,glass.hit,incidentIor);
  // A collapsed sheet has no net Snell bend, so the already-resolved collinear
  // opaque hit is exactly the transmitted scene query; never traverse it twice.
  let reflected=dryEnvironment(reflect(rd,glass.hit.geometricNormal),.04);let transmitted=shadeDryOpaque(opaque,ro,rd);
  let color=reflected*optics.fresnel+transmitted*optics.netTransmittance;
  return DryGlassSurface(color,glass.hit.t_m,svoThinGlassMaterialId(record),svoThinGlassOwnerId(record),svoThinGlassPaneId(record),0u);
}

fn dryThickGlassEmission(materialId:u32)->vec3f{
  if(materialId>=dry.materialPublication.x){return vec3f(0.0);}let material=dryMaterial(materialId);
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
// Scene surfaces follow the complete renderer publication. Fluid and rigid
// paths publish their own local generations below.
fn dryPublicationGeneration()->u32{return dry.primitiveCandidates.w;}
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
fn dryFragmentOut(targets:SvoGBufferTargets,hardwareDepth:f32)->DryFragmentOut{
  return DryFragmentOut(targets.radianceDepth,targets.packedSurface,targets.identityMedia,hardwareDepth);
}

@fragment fn fragmentMain(input:VertexOut)->DryFragmentOut {
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());dryVisibilityIgnoredOwner=DRY_OWNER_NONE;dryThickGlassFailure=0u;
  // Curved thick glass is compiled separately from this Metal-sensitive pass.
  // Its authored pane therefore remains visible through the exact thin fallback.
  dryThickGlassEnabled=0u;
  let opaque=traceOpaqueScene(ro,rd);${prepassResolveCallWGSL}let glass=traceGlass(ro,rd,0.0,opaque.t,true);var color=shadeDryOpaque(opaque,ro,rd);var depth=opaque.t;
  let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;var glassSurface=DryGlassSurface(vec3f(0.0),DRY_MISS,0u,DRY_OWNER_NONE,0u,0u);
  if(glassVisible){glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);let radiance=max(color*vignette,vec3f(0.0));let generation=dryPublicationGeneration();
  if(glassVisible){
    let media=dryMediumPair(rd,glass.hit.geometricNormal,DRY_MEDIUM_GLASS);
    let targets=svoGBufferSurface(radiance,depth,glass.hit.geometricNormal,glass.hit.geometricNormal,vec4u(glassSurface.materialId,glassSurface.ownerId,media.x,media.y),vec3f(0.0),DRY_GBUFFER_MOTION_STATIC,DRY_GBUFFER_FIELD_ANALYTIC,generation,SVO_GBUFFER_MOTION_VALID|svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_GLASS),SVO_FEATURE_SMOOTH);
    return dryFragmentOut(targets,dryHardwareDepth(depth,rd,forward));
  }
  if(opaque.t<DRY_MISS){
    let media=dryMediumPair(rd,opaque.normal,DRY_MEDIUM_OPAQUE);let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u)|svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_TRACED);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}
    let targets=svoGBufferSurface(radiance,opaque.t,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);
    return dryFragmentOut(targets,dryHardwareDepth(opaque.t,rd,forward));
  }
  return dryFragmentOut(svoGBufferMiss(radiance,0u,generation,DRY_GBUFFER_NO_INTERSECTION,svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_TRACED)),0.0);
}
${splitEntryWGSL}${rasterPrimaryEntryWGSL}${prepassEntryWGSL}${prepassFromPrimaryEntryWGSL}${pixelProbe ? createSvoPixelTraceProbeWGSL(svoDryScenePixelProbeOptions(traversalMode === "raster-primary" ? "raster" : "traced")) : ""}`;
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
      .replace("return DryGlobalIllumination(max(dryPrepassGi.rgb,vec3f(0.0)),clamp(dryPrepassGi.a,0.0,1.0),1u);",
        "return DryGlobalIllumination(max(vec3f(dryPrepassGi.rgb),vec3f(0.0)),clamp(f32(dryPrepassGi.a),0.0,1.0),1u);")
      .replaceAll("dryPrepassData0=dryPrepassUnpack0(packed);dryPrepassData1=dryPrepassUnpack1(packed);dryPrepassData2=dryPrepassUnpack2(packed);",
        "dryPrepassData0=vec4h(dryPrepassUnpack0(packed));dryPrepassData1=vec4h(dryPrepassUnpack1(packed));dryPrepassData2=vec4h(dryPrepassUnpack2(packed));")
      .replace("dryPrepassGi=textureLoad(dryPrepassRadianceTexture,texel,0);",
        "dryPrepassGi=vec4h(textureLoad(dryPrepassRadianceTexture,texel,0));")
      .replace("dryPrepassRadiance=textureLoad(dryPrepassRadianceTexture,texel,0);",
        "dryPrepassRadiance=vec4h(textureLoad(dryPrepassRadianceTexture,texel,0));")
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
${cameraApertureShaderLibrary()}
@group(0) @binding(4) var<storage,read> dryRasterSceneArena:array<u32>;
@group(1) @binding(0) var dryRasterOpaqueGeometry:texture_2d<f32>;
@group(1) @binding(1) var<uniform> glassRaster:GlassRasterParams;
const DRY_RASTER_GLASS_NEAR_M:f32=${SVO_DRY_SCENE_REVERSED_Z_NEAR_M};
const DRY_RASTER_GLASS_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.glassOffsetBytes / 4}u;
fn dryRasterGlassWords4(offset:u32)->vec4u{return vec4u(dryRasterSceneArena[offset],dryRasterSceneArena[offset+1u],dryRasterSceneArena[offset+2u],dryRasterSceneArena[offset+3u]);}
fn dryRasterGlassPane(index:u32)->SvoThinGlassRecord{let base=DRY_RASTER_GLASS_WORD_OFFSET+index*${SVO_THIN_GLASS_RECORD_WORDS}u;return SvoThinGlassRecord(
  bitcast<vec4f>(dryRasterGlassWords4(base)),bitcast<vec4f>(dryRasterGlassWords4(base+4u)),bitcast<vec4f>(dryRasterGlassWords4(base+8u)),
  bitcast<vec4f>(dryRasterGlassWords4(base+12u)),dryRasterGlassWords4(base+16u));}
fn dryRasterGlassCorner(index:u32)->vec2f{
  var corners=array<vec2f,6>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0));
  return corners[index];
}
fn dryRasterGlassRay(pixel:vec2f)->mat2x3f{
  let uv=vec2f(pixel.x/max(uniforms.viewport.x,1.0),1.0-pixel.y/max(uniforms.viewport.y,1.0));let ndc=uv*2.0-1.0;
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));
  let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());return mat2x3f(ro,rd);
}
@vertex fn glassRasterVertex(@builtin(vertex_index) vertexIndex:u32,@builtin(instance_index) recordIndex:u32)->GlassRasterVertexOut{
  let record=dryRasterGlassPane(recordIndex);let corner=dryRasterGlassCorner(vertexIndex);let padding=max(2.0*record.extentIorEpsilon.w,1e-5);let local=vec3f(corner*(record.extentIorEpsilon.xy+vec2f(padding)),0.0);let world=record.centerThickness.xyz+svoThinGlassRotate(record.orientation,local);
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));let relative=world-ro;let viewDepth=dot(relative,forward);let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  return GlassRasterVertexOut(vec4f(dot(relative,right)/(aspect*cameraTanHalfFov()),dot(relative,up)/cameraTanHalfFov(),.5*viewDepth,viewDepth),recordIndex);
}
@fragment fn glassRasterFragment(input:GlassRasterVertexOut)->GlassRasterFragmentOut{
  if(input.recordIndex>=glassRaster.paneCount){discard;}
  let record=dryRasterGlassPane(input.recordIndex);let paneId=svoThinGlassPaneId(record);let compositeOwned=glassRaster.compositePaneCount>0u&&paneId>=glassRaster.compositePaneIdBase&&paneId-glassRaster.compositePaneIdBase<glassRaster.compositePaneCount;if(compositeOwned){discard;}
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

/**
 * Slack on the scene-primitive march bracket, absolute and relative to the span.
 *
 * Exported so the CPU containment test brackets with exactly the shipped
 * numbers rather than a tolerance of its own, which is the only way that test
 * can fail for a real reason.
 */
export const SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_M = 1e-4;
export const SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_RELATIVE = 1e-3;

export const SVO_SCENE_PRIMITIVE_RASTER_CONTRACT = Object.freeze({
  verticesPerProxy: 36,
  /**
   * One face of the proxy box, not both.
   *
   * The fragment repeats the shared exact intersection, so the two faces of a
   * box produce the same hit and the second one is pure duplicated work — and
   * this is the pass every marched primitive lands in, where that work is a
   * sphere trace rather than a quadratic. Drawing the far side (and not the
   * near) is what keeps a camera *inside* a primitive's proxy shading it, which
   * is the same reason the brick raster culls this way: the corner table winds
   * outward-CCW in world space and projection flips the near-facing triangles
   * to clockwise, which WebGPU's default `ccw` front face calls back faces.
   *
   * The near-plane case is unaffected: a proxy that contains the eye emits a
   * full-screen triangle instead, and that one is wound front-facing.
   */
  cullMode: SVO_BRICK_RASTER_CONTRACT.cullMode,
  /**
   * Fixed per-pixel conservative candidate arena for the authored SDF set.
   *
   * WGSL has no conservative depth (gpuweb#5342), so the direct fragment — which
   * writes `frag_depth` *and* discards — lets the tiler reject nothing and every
   * covering proxy marches. The hero framing puts a median of 7 proxies over a
   * covered pixel, p99 33 and max 59; forty entries keep the overflow arm below
   * one percent of covered pixels while capacity stays a performance parameter
   * and never an image change. Deliberately larger than the brick arena's 24 and
   * *not* built on its 32-entry u32 `visited` mask, which p99=33 already exceeds:
   * candidates here carry their own ordering key and are walked by monotone
   * extraction instead.
   */
  coverageCandidatesPerPixel: 40,
  /**
   * Candidate key = floored 16-bit entry distance over a 16-bit record index.
   *
   * Truncating the low half of a non-negative float's bit pattern is monotone
   * and rounds *down*, so ordering by the packed word orders by a conservative
   * lower bound on the ray's entry into the primitive's oriented box. That is
   * exactly what front-to-back early termination needs: a candidate whose lower
   * bound is already behind the best hit cannot produce a nearer one. Sixteen
   * bits of index cap the arena at 65,536 authored records; past that the
   * encode falls back to the historical direct pass rather than aliasing.
   */
  coverageIndexBits: 16,
  entryPoints: Object.freeze({
    vertex: "dryScenePrimitiveRasterVertex",
    /** The same proxy, behind the near-field band's membership bit. */
    bandVertex: "dryScenePrimitiveBandVertex",
    fragment: "dryScenePrimitiveRasterFragment",
    coverage: "dryScenePrimitiveCoverageFragment",
    resolve: "dryScenePrimitiveCoverageResolveFragment",
    overflowResolve: "dryScenePrimitiveCoverageOverflowFragment",
  }),
} as const);

/** Records addressable by a coverage candidate key; beyond it the direct pass runs. */
export const SVO_SCENE_PRIMITIVE_COVERAGE_MAXIMUM_RECORDS =
  1 << SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageIndexBits;

/** Below this pixel count, Metal's TBDR fragment resolve remains faster. */
export const SVO_SCENE_PRIMITIVE_COMPUTE_MINIMUM_PIXELS = 1 << 20;

interface SvoDrySplitPipelineBundle {
  readonly visibility: GPURenderPipeline;
  readonly rasterRigidVisibility?: GPURenderPipeline;
  /** Optional one-pixel primary coverage closure before the sky/surface partition. */
  readonly primarySeamClosure: GPURenderPipeline;
  readonly lighting: GPURenderPipeline;
  readonly reconstructedLighting?: GPURenderPipeline;
  /** Complement of `lighting`: the pixels primary visibility left as a miss. */
  readonly skyLighting: GPURenderPipeline;
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
  private rasterGlassPaneCount = 0;
  private rasterRigidActive: boolean;
  /** Raster-assisted primary visibility (traversal mode `raster-primary`). */
  private readonly rasterPrimary: boolean;
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
  /**
   * Not readonly, because the terrain region at its tail is the one part of the
   * arena whose size the scene chooses. See `ensureSceneArenaCapacity`.
   */
  private sceneArenaBuffer: GPUBuffer;
  private sceneArenaTerrainCapacityBytes = SVO_DRY_SCENE_TERRAIN_HEADER_BYTES;
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
  private rigidMotionSource?: GPUBuffer;
  /** xyz centre, w radius of one sphere over every body; negative radius = none. */
  private rigidBounds: [number, number, number, number] = [0, 0, 0, -1];
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
  private primitiveDirtyBounds: readonly SvoDrySceneDirtyBounds[] = [];
  private lightingOptions: SvoLightingOptions = DEFAULT_SVO_LIGHTING_OPTIONS;
  private silhouetteRefinementEnabled = false;
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
  /** Whether the frame the pending probe was encoded beside ran the cone prepass. */
  private probeEncodedConePrepass = false;

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
    private readonly experiments: SvoDryOptimizationExperiments = {},
  ) {
    if (targetFormat !== SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat) {
      throw new Error(`Sparse voxel dry scene location 0 must use ${SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat}`);
    }
    if (shadingPath !== "inline" && shadingPath !== "split") throw new RangeError(`Unsupported dry-scene shading path: ${shadingPath}`);
    if (rayCoherenceMode !== "off" && rayCoherenceMode !== "static-primary") throw new RangeError(`Unsupported dry-scene ray coherence mode: ${rayCoherenceMode}`);
    if (rayCoherenceMode === "static-primary" && shadingPath !== "split") throw new RangeError("Static-primary ray coherence requires split shading");
    if (screenSpaceTerminationPixels > 0
      && !((traversalMode === "canonical" && shadingPath === "inline")
        || (traversalMode === "raster-primary" && shadingPath === "split"))) {
      throw new RangeError("Screen-space termination requires canonical inline or raster-primary split traversal");
    }
    this.rasterPrimary = traversalMode === "raster-primary";
    this.rasterPrimaryDirect = experiments.rasterPrimaryDirect === true
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
      label: "Live authored scene arena (materials, primitives/BVH, thin glass, terrain)",
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
    const fragmentShader = this.traversalMode === "hybrid" && this.brickOccupancyMode === "off" && this.screenSpaceTerminationPixels === 0
      ? drySceneShader : createSvoDrySceneFragmentWGSL(1, this.traversalMode, this.brickOccupancyMode, this.shadingPath, this.screenSpaceTerminationPixels,
        false, this.rasterPrimary && this.rasterGlassDiscovery, this.rasterPrimary && this.rasterRigidDiscovery, false,
        { ...this.experiments, voxelLightCache: false });
    report(1);
    const [vertexModule, fragmentModule] = await Promise.all([
      checkedModule(this.device, "Sparse voxel dry scene vertex", drySceneVertexShader),
      checkedModule(this.device, `Sparse voxel dry scene fragment (${this.traversalMode}, brick-${this.brickOccupancyMode})`,
        fragmentShader),
    ]);
    report(2);
    this.layout = this.device.createBindGroupLayout({
      label: `Sparse voxel dry scene bindings (${this.traversalMode})`,
      entries: sparseVoxelDrySceneBindGroupLayoutEntries(this.traversalMode),
    });
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
    // The brick draw layout is a pipeline-layout input of the split bundle, so
    // instance emission compiles first.
    report(3);
    await this.ensureBrickCullPipelines();
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
    this.primarySeamClosurePipeline = bundle.primarySeamClosure;
    this.splitLightingPipeline = bundle.lighting;
    this.splitReconstructedLightingPipeline = bundle.reconstructedLighting;
    this.splitSkyLightingPipeline = bundle.skyLighting;
    this.conePrepassResetPipeline = bundle.prepassReset;
    this.conePrepassCoherentPipeline = bundle.prepassCoherent;
    this.conePrepassBoundaryPipeline = bundle.prepassBoundary;
    this.worldGiFramePipeline = bundle.worldGiFrame;
    this.worldGiCachePipeline = bundle.worldGiCache;
    this.voxelLightDemandPipeline = bundle.voxelLightDemand;
    this.voxelLightPopulatePipeline = bundle.voxelLightPopulate;
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

  /**
   * Emission, prefix scan and scatter for the brick instance list. The module
   * is standalone — camera uniform, published topology and the `SvoMapping`
   * prefix of `DryParams` — so it is independent of cone-lighting scale and of
   * the renderer's fragment-only shading bindings.
   */
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
      // Binding 6 joins the resolve set because `traceLeafPayload` reads the
      // scene-geometry lane for its shading normal, and the brick resolve
      // fragment is the entry point that calls it on the shipping path.
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
    if (!this.rasterPrimary || this.brickProbePipeline) return;
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
        { binding: 3, resource: structural.sceneMaterialOwners },
        { binding: 4, resource: { buffer: this.sceneArenaBuffer } },
        { binding: 6, resource: structural.sceneGeometry },
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
          // Mapping plus metadata: the DDA needs the primitive base owner and
          // count, which sit immediately after the mapping prefix.
          { binding: probe.params, resource: { buffer: this.paramsBuffer, offset: 0, size: SVO_BRICK_RASTER_PROBE_CONTRACT.paramsBindingBytes } },
          // One request buffer shared with the ray probe, so the two cannot
          // answer different pixels for the same frame.
          { binding: probe.request, resource: { buffer: this.probeBuffers.request } },
          { binding: probe.structure, resource: structural.structure },
          { binding: probe.materialOwners, resource: structural.sceneMaterialOwners },
          { binding: probe.scene, resource: { buffer: this.sceneArenaBuffer } },
          { binding: probe.rasterPublication, resource: { buffer: this.brickRasterPublicationBuffer! } },
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
    tracePhase?: RenderPathTracePhase,
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
      tracePhase?.({ id: "svo-scene-primitive", label: "SVO exact live-scene primitive visibility" });
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
      tracePhase?.({ id: "svo-band", label: "SVO near-field analytic band selection" });
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
    tracePhase?.({ id: "svo-scene-primitive", label: "SVO exact live-scene primitive visibility" });
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
    tracePhase?: RenderPathTracePhase,
  ): void {
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
    tracePhase?.({ id: "svo-brick-cull", label: "SVO brick instance cull" });

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
    // The caller closes "svo-primary" straight after this returns, so the brick
    // raster lands under the same phase id the traced primary reports.
  }

  private async ensureSplitPipelines(scale: SvoConeLightingScale): Promise<void> {
    if (this.shadingPath === "inline" || !this.layout || !this.vertexModule) return;
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
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
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
    const shaderExperiments = voxelLightCacheEnabled
      ? shaderExperimentsBase
      : { ...shaderExperimentsBase, voxelLightCache: false };
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
      const bundle = { visibility, rasterRigidVisibility, primarySeamClosure, lighting, reconstructedLighting, skyLighting, prepassReset, prepassCoherent, prepassBoundary,
        worldGiFrame, worldGiCache, voxelLightDemand, voxelLightPopulate,
        brickBackground, brickRaster, brickCoverage, brickCoverageResolve, brickLodResolve, brickExactResolve,
        brickCoverageOverflow, scenePrimitiveRaster,
        scenePrimitiveCoverage, scenePrimitiveLodResolve, scenePrimitiveComputeArgs, scenePrimitiveComputeResolve, scenePrimitiveDepthBridge,
        scenePrimitiveCoverageResolve, scenePrimitiveCoverageOverflow };
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
    if (this.shadingPath === "inline" || !this.targetWidth || !this.targetHeight
      || !this.splitVisibilityLayout || !this.splitLightingLayout) return;
    this.ensureBrickCoverageBuffers();
    if (!this.splitGeometry || !this.splitOpaqueIdentity || (this.rasterGlassDiscovery && (!this.splitGlassKey || !this.splitGlassDepth))
      || (this.screenSpaceTerminationPixels > 0 && !this.scenePrimitiveComputeDepth)
      || (this.rasterRigidDiscovery && !this.rasterRigidPrimaryGeometry)
      || this.splitWidth !== this.targetWidth || this.splitHeight !== this.targetHeight) {
      this.clearPrimaryVisibilityCache();
      this.splitGeometry?.destroy();
      this.splitOpaqueIdentity?.destroy();
      this.splitGlassKey?.destroy();
      this.splitGlassDepth?.destroy();
      this.rasterRigidPrimaryGeometry?.destroy();
      this.scenePrimitiveComputeDepth?.destroy();
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
    this.clearReusableFrame();
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
    const eligible = Boolean(nodeMip?.plan.complete && nodeMip.directPageTableReady
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
    if (source === this.source) return;
    this.pickingFrameToken += 1;
    this.lastPickingTarget = undefined;
    this.clearReusableFrame();
    this.clearPrimaryVisibilityCache();
    this.worldGiCacheDirty = true;
    this.source = source;
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
  /**
   * Grow the arena so a scene's sculpted terrain fits behind the fixed regions.
   *
   * The heightfield is the only variable-length thing in here, and it is
   * variable by three orders of magnitude: the hero pond's 289 x 193 lattice is
   * 224 KB while the authored ceiling is a megabyte and most scenes want
   * nothing at all. Reserving the ceiling would cost every scene the worst
   * case, so the allocation follows the scene instead — which the arena can
   * afford precisely because it is republished whole on a scene change.
   *
   * Capacity only ever grows. Shrinking would reclaim a few hundred kilobytes
   * at the price of reallocating and re-uploading every region each time the
   * user stepped between a sculpted scene and a flat one, and the bind groups
   * that name this buffer would churn with it.
   */
  private ensureSceneArenaCapacity(terrainBytes: number): boolean {
    if (terrainBytes <= this.sceneArenaTerrainCapacityBytes) return false;
    const sizeBytes = svoDrySceneArenaSizeBytes(terrainBytes);
    this.sceneArenaBuffer.destroy();
    this.sceneArenaBuffer = this.device.createBuffer({
      label: "Live authored scene arena (materials, primitives/BVH, thin glass, terrain)",
      size: sizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.sceneArenaTerrainCapacityBytes = sizeBytes - SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes;
    return true;
  }

  publishScene(scene: SparseVoxelDrySceneData): boolean {
    const source = this.source;
    if (!canEncodeSparseVoxelDryScene(source, scene)) return false;
    const primitiveArena = packSvoPrimitiveCandidateArena(scene.primitiveRecords, scene.primitiveCandidates);
    if (primitiveArena.packedRecords.byteLength > SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES) throw new RangeError("Live scene primitive arena capacity exceeded");
    if (scene.materialRecords.byteLength > SVO_DRY_SCENE_MATERIAL_ARENA_SIZE_BYTES) throw new RangeError("Live scene material arena capacity exceeded");
    if ((scene.glassRecords?.byteLength ?? 0) > SVO_DRY_SCENE_GLASS_ARENA_SIZE_BYTES) throw new RangeError("Live scene thin-glass arena capacity exceeded");
    if ((scene.terrainHeightfield?.byteLength ?? 0) > SVO_DRY_SCENE_TERRAIN_ARENA_MAXIMUM_BYTES) throw new RangeError("Live scene terrain heightfield capacity exceeded");
    const arenaReallocated = this.ensureSceneArenaCapacity(scene.terrainHeightfield?.byteLength ?? 0);

    this.pickingFrameToken += 1;
    this.lastPickingTarget = undefined;
    this.clearReusableFrame();
    this.clearPrimaryVisibilityCache();
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
    this.rasterGlassPaneCount = paneCount;
    const paneIdBase = scene.primaryCompositeOwnedGlassPaneIdBase ?? 0xffff_ffff;
    const compositePaneCount = scene.primaryCompositeOwnedGlassPaneCount ?? 0;
    const records = scene.glassRecords;
    const rasterRange = svoDryRasterGlassRecordRange(records, paneIdBase, compositePaneCount);
    this.rasterGlassFirstRecord = rasterRange.firstRecord;
    this.rasterGlassRecordCount = rasterRange.recordCount;
    this.device.queue.writeBuffer(this.rasterGlassParamsBuffer, 0, new Uint32Array([
      paneCount,
      paneIdBase,
      compositePaneCount,
      0,
    ]));
    this.device.queue.writeBuffer(this.sceneArenaBuffer, SVO_DRY_SCENE_ARENA_LAYOUT.primitiveOffsetBytes, primitiveArena.packedRecords);
    this.device.queue.writeBuffer(this.sceneArenaBuffer, SVO_DRY_SCENE_ARENA_LAYOUT.materialOffsetBytes, scene.materialRecords);
    if (records?.byteLength) this.device.queue.writeBuffer(this.sceneArenaBuffer, SVO_DRY_SCENE_ARENA_LAYOUT.glassOffsetBytes, records);
    // Written on every publication for the same reason the terrain header is:
    // a stale block left behind by the previous scene would be resolved by a
    // new scene's cluster record and grow the wrong packing inside its lobe.
    // Zeroes are the "not resolved" encoding, which draws nothing.
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
    // The header is written on every publication, never conditionally: a zeroed
    // header is what tells the shader this scene is analytic, and skipping it
    // would leave the previous scene's vessel standing under the new ground.
    this.device.queue.writeBuffer(
      this.sceneArenaBuffer,
      SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes,
      scene.terrainHeightfield ?? new Uint32Array(SVO_DRY_SCENE_TERRAIN_HEADER_WORDS),
    );
    this.writeParams(source!, scene);
    const lightingArena = packSparseVoxelDrySceneLightingArena(scene);
    if (lightingArena) this.device.queue.writeBuffer(this.lightingBuffer, 0, lightingArena);
    this.device.queue.writeBuffer(this.thickGlassUniformBuffer, 0, packSparseVoxelDrySceneThickGlassArena(scene));
    if (!this.bindGroup || arenaReallocated) this.rebuild();
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
    this.clearReusableFrame();
    // sceneEpoch (pickingFrameToken above) is part of the primary reuse key, so
    // an incremental pose cannot reuse the previous G-buffer. Avoid separately
    // destroying the cache: stationary visibility remains reusable again as
    // soon as the caller's generation stops changing.
    if (!incremental) this.clearPrimaryVisibilityCache();
    this.primitiveDirtyBounds = change.dirtyBounds;
    const invalidation = svoDryPrimitiveArenaCacheInvalidation(change);
    if (invalidation.worldGi) this.worldGiCacheDirty = true;
    if (invalidation.directionalVisibility) this.invalidateVoxelLightCache();
    return true;
  }

  /** Enable finished-image visibility effects without rebuilding scene-owned resources. */
  setLightingOptions(options: SparseVoxelDrySceneLightingOptions): void {
    const coneTracingMode = options.coneTracingMode ?? "cones";
    // Leaving `cones` collapses to the full-resolution inline/split path: with
    // the effective scale forced to 1, encode() never runs the reduced
    // prepass, compact cone visibility, sample fan-out, or world-GI cache
    // passes, and writeParams withholds every cone-dependent flag.
    const coneLightingScale = coneTracingMode === "cones" ? (options.coneLightingScale ?? 1) : 1;
    const silhouetteRefinementEnabled = options.silhouetteRefinementEnabled === true;
    const previousConeTracingMode = this.lightingOptions.coneTracingMode ?? "cones";
    if (options.shadowsEnabled === this.lightingOptions.shadowsEnabled
      && options.ambientOcclusionEnabled === this.lightingOptions.ambientOcclusionEnabled
      && silhouetteRefinementEnabled === this.silhouetteRefinementEnabled
      && coneTracingMode === previousConeTracingMode
      && coneLightingScale === this.coneScale) return;
    const invalidateWorldGi = options.ambientOcclusionEnabled !== this.lightingOptions.ambientOcclusionEnabled
      || coneTracingMode !== previousConeTracingMode;
    this.lightingOptions = { shadowsEnabled: options.shadowsEnabled, ambientOcclusionEnabled: options.ambientOcclusionEnabled,
      silhouetteRefinementEnabled, coneTracingMode };
    this.silhouetteRefinementEnabled = silhouetteRefinementEnabled;
    this.coneScale = coneLightingScale;
    this.requestedBundleFailure = undefined;
    this.requestedBundleResourceFailure = undefined;
    this.writeVoxelLightCacheParams();
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
    const retainFailure = (error: unknown) => {
      if (this.coneScale !== coneLightingScale) return;
      const reason = error instanceof Error ? error.message : String(error);
      this.requestedBundleFailure = {
        scale: coneLightingScale,
        detail: `Requested SVO presentation bundle at scale ${coneLightingScale} failed: ${reason}`,
      };
    };
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
    if (this.splitPipelineCompiles.has(this.coneScale) || this.splitPipelineScale !== this.coneScale) {
      return { state: "compiling", detail: `Preparing split presentation bundle at scale ${this.coneScale}` };
    }
    if (!this.primarySeamClosurePipeline) {
      return { state: "compiling", detail: "Preparing primary seam-closure pipeline" };
    }
    return { state: "enabled" };
  }

  /** Update bounded shader work budgets without rebuilding scene resources. */
  setRenderTuning(tuning: SvoRenderTuning): void {
    const normalized = normalizeSvoRenderTuning(tuning);
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
    // `surfaceReconstruction` rides the same lane and earns the same treatment
    // for the same reason: it changes the shaded direction of an already-found
    // hit, not `t`, not hit-versus-miss, and not what any light reads. Toggling
    // the panel between RAW and SHADED is therefore one 16-byte write plus a
    // reshade, which is also what makes the two arms interleavable in an A/B.
    const lodKeys = ["lodMode", "lodScreenSpacePixels", "lodFixedLevel", "surfaceReconstruction"] as const;
    const lodOnly = (Object.keys(normalized) as (keyof SvoRenderTuning)[])
      .every((key) => normalized[key] === this.renderTuning[key] || (lodKeys as readonly string[]).includes(key));
    if (lodOnly) {
      this.renderTuning = normalized;
      this.writeLodParams();
      this.clearReusableFrame();
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
    this.clearReusableFrame();
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
    words.set([scene.terrainMaterialId ?? 0xffff_ffff, (scene.glassRecords?.byteLength ?? 0) / SVO_THIN_GLASS_RECORD_STRIDE_BYTES, scene.primaryCompositeOwnedGlassPaneIdBase ?? 0xffff_ffff, scene.primaryCompositeOwnedGlassPaneCount ?? 0], SVO_DRY_SCENE_PARAMS_LAYOUT.terrainWordOffset);
    if (scene.terrainMaterialMetadata) words.set(scene.terrainMaterialMetadata, SVO_DRY_SCENE_PARAMS_LAYOUT.terrainMaterialWordOffset);
    const coneTracingMode = this.lightingOptions.coneTracingMode ?? "cones";
    // `off` strictly removes lighting-visibility work: with shadows and AO
    // held false no exact-ray flag is written either, and every visibility
    // entry point returns its unoccluded constant. `exact` keeps the bounded
    // reference traversals while withholding all cone stages.
    const shadowsEnabled = coneTracingMode !== "off" && this.lightingOptions.shadowsEnabled && scene.shadowVisibilityEnabled !== false;
    const ambientOcclusionEnabled = coneTracingMode !== "off" && this.lightingOptions.ambientOcclusionEnabled && scene.contactVisibilityEnabled !== false;
    const coneTracingEnabled = coneTracingMode === "cones" && this.derivedLightingReady();
    const nodeMip = source.nodeMipPyramid;
    const nodeMipUsesPageValidity = Boolean((nodeMip as typeof nodeMip & {
      pageValidity?: { view: GPUTextureView };
    } | undefined)?.pageValidity?.view);
    const tetrahedralRadiance = source.tetrahedralRadiance;
    const giReady = coneTracingEnabled && Boolean(nodeMip && tetrahedralRadiance
      && nodeMip.generation === tetrahedralRadiance.generation
      && nodeMip.plan.complete && tetrahedralRadiance.plan.complete);
    const silhouetteRefinementActive = this.silhouetteRefinementEnabled && this.shadingPath === "split";
    const visibilityFlags = (!giReady && ambientOcclusionEnabled ? SVO_DRY_VISIBILITY_FLAGS.exactContact | SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion : 0)
      | (shadowsEnabled ? SVO_DRY_VISIBILITY_FLAGS.exactShadow : 0)
      | (coneTracingEnabled && (shadowsEnabled || ambientOcclusionEnabled || giReady) ? SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested : 0)
      | (giReady ? SVO_DRY_VISIBILITY_FLAGS.globalIllumination : 0)
      | (giReady && ambientOcclusionEnabled ? SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion : 0)
      | (coneTracingEnabled ? SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested : 0)
      | (silhouetteRefinementActive ? SVO_DRY_VISIBILITY_FLAGS.silhouetteRefinement : 0);
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
    floats[offset + 3] = SVO_SURFACE_RECONSTRUCTION_CODES[this.renderTuning.surfaceReconstruction];
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
  private writeLodParams(): void {
    const buffer = new ArrayBuffer(16);
    const floats = new Float32Array(buffer), words = new Uint32Array(buffer);
    this.packLodParams(floats, words, 0);
    this.device.queue.writeBuffer(this.paramsBuffer, SVO_DRY_SCENE_PARAMS_LAYOUT.lodWordOffset * 4, buffer);
    // Keep the memoized snapshot in step, or the next whole-params write sees a
    // difference that is already on the device and rewrites 592 bytes for it.
    this.paramsWords?.set(words, SVO_DRY_SCENE_PARAMS_LAYOUT.lodWordOffset);
  }

  private rebuild(): void {
    const source = this.source, structural = source?.structural;
    if (!this.layout || !this.pipeline || !source || !structural || !this.scene) {
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
    this.bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.bodyBuffer } },
      { binding: 2, resource: structural.structure },
      { binding: 3, resource: structural.sceneMaterialOwners },
      { binding: 4, resource: { buffer: this.sceneArenaBuffer } },
      ...(derivedTraversal ? [{ binding: 5, resource: derivedTraversal }] : []),
      { binding: 6, resource: structural.sceneGeometry },
      { binding: 9, resource: { buffer: this.paramsBuffer } },
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
          { binding: 7, resource: this.fluidCoverage?.visibleGeneration()?.view ?? this.fluidCoverageFallbackView },
          { binding: 8, resource: nodeMip?.directPageTableView ?? this.nodeMipFallbackDirectPageTableView },
          { binding: 9, resource: nodeMipPageValidity ?? this.nodeMipPageValidityFallbackView },
        ],
      })
      : undefined;
    this.ensureBrickRasterBuffers();
    this.rebuildBrickRasterBindGroups();
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

  /** Runtime A/B and emergency fallback switch; disabling never destroys cached data. */
  setVoxelLightCacheEnabled(enabled: boolean): void {
    if (this.voxelLightUserEnabled === enabled) return;
    this.voxelLightUserEnabled = enabled;
    this.writeVoxelLightCacheParams();
    this.clearReusableFrame();
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
    // The raster-primary half runs against the instance list this frame's cull
    // already published, which is why it is encoded here rather than beside the
    // cull: by now the buffer holds the very set the draw consumed.
    this.encodeBrickRasterProbe(encoder);
    this.probeEncodedToken = request.token;
    this.probeReadPending = true;
    return true;
  }

  private encodeBrickRasterProbe(encoder: GPUCommandEncoder): void {
    if (!this.rasterPrimary || !this.brickProbePipeline || !this.brickProbeBindGroup
      || !this.brickProbeBuffers || this.brickProbeReadPending) return;
    const pass = encoder.beginComputePass({ label: "Sparse voxel raster-primary probe" });
    pass.setPipeline(this.brickProbePipeline);
    pass.setBindGroup(0, this.brickProbeBindGroup);
    // One workgroup: the lanes stride the instance list between them, and the
    // ordering and election that follow are a single-lane epilogue.
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
    // A newer pointer position does not invalidate this trace: it answers the
    // pixel it was asked about and the next frame supersedes it. Only a resource
    // or source epoch change makes the recorded world-space boxes meaningless.
    const generation = this.pickingFrameToken;
    const current = () => this.pickingFrameToken === generation;
    try {
      // Both halves are mapped together and folded into one account of the
      // pixel. The raster half is optional throughout: in traced mode it never
      // runs, and if its pipeline failed the lighting half still stands alone.
      const [lighting, primary] = await Promise.all([
        this.probeBuffers.read(current),
        this.brickProbeReadPending && this.brickProbeBuffers
          ? this.brickProbeBuffers.read(current)
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

  encode(encoder: GPUCommandEncoder, target: GPUTexture | GPUTextureView, reuseKey?: string, tracePhase?: RenderPathTracePhase): DrySceneReplacementResult | false {
    if (!this.pipeline || !this.bindGroup) return false;
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
    const activeSplitVisibilityPipeline = this.rasterRigidActive
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
        && this.splitVisibilityBindGroup && this.splitLightingBindGroup && this.splitGeometryView);
    if (this.coneScale !== 1 && !usePrepass) {
      this.requestedBundleResourceFailure = `Requested SVO cone bundle at scale ${this.coneScale} has incomplete frame resources`;
      return false;
    }
    if (splitRequested && !this.splitDiagnosticsActive && !useSplit) {
      this.requestedBundleResourceFailure = `Requested SVO split bundle at scale ${this.coneScale} has incomplete frame resources`;
      return false;
    }
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
        if (this.rasterPrimary) {
          this.encodeRasterPrimary(encoder, gBufferViews, usePrepass, splitGroup, tracePhase);
        } else {
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
        tracePhase?.({ id: "svo-primary", label: "SVO primary visibility" });
        if (this.rasterPrimary) {
          this.encodeScenePrimitivePrimary(encoder, gBufferViews, usePrepass, splitGroup, tracePhase);
        }
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
        if (this.rasterGlassDiscovery && this.rasterGlassPaneCount > 0) {
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
        if (this.silhouetteRefinementEnabled) {
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
          tracePhase?.({ id: "svo-primary", label: "SVO primary seam closure" });
        }
        this.primaryVisibilityCacheKey = this.rayCoherenceMode === "static-primary" && usePrepass
          ? primaryFrameKey
          : undefined;
      }

      if (this.voxelLightActive && this.voxelLightDemandPipeline && this.voxelLightPopulatePipeline
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

        if (!reconstructReducedRadiance) {
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
          tracePhase?.({ id: "svo-environment-gi", label: "SVO persistent world-space environmental GI" });
        }
      }

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
        shade.setPipeline(this.conePrepassShadePipeline!);
        shade.setBindGroup(0, this.bindGroup);
        shade.setBindGroup(1, this.conePrepassShadeBindGroup!);
        shade.draw(3);
        shade.end();
        tracePhase?.({ id: "svo-cone-lighting", label: "SVO reduced-rate opaque shading" });
      }

      const lighting = encoder.beginRenderPass({
        label: "Sparse voxel deferred dry lighting",
        colorAttachments: [{ view: targetView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
        // Read-only: the pass classifies pixels by the depth primary visibility
        // already established and never contributes to it, so the attachment
        // costs a tile load and no store.
        depthStencilAttachment: { view: gBufferViews.hardwareDepth, depthReadOnly: true },
      });
      lighting.setBindGroup(0, this.bindGroup);
      if (usePrepass) lighting.setBindGroup(1, this.conePrepassBindGroup!);
      lighting.setBindGroup(splitGroup, this.splitLightingBindGroup!);
      if (voxelLightBindingsRequired) lighting.setBindGroup(splitGroup + 1, this.voxelLightConsumerBindGroup!);
      // Two complementary depth tests partition the frame: the sky shader takes
      // the miss pixels and the full deferred shader takes the rest. Sky first
      // so the expensive draw is the one still in flight when the pass ends.
      lighting.setPipeline(this.splitSkyLightingPipeline!);
      lighting.draw(3);
      if (usePrepass && reconstructReducedRadiance) {
        lighting.setPipeline(this.splitReconstructedLightingPipeline!);
        lighting.draw(3);
      }
      lighting.setPipeline(this.splitLightingPipeline!);
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
    this.splitGeometry?.destroy();
    this.splitOpaqueIdentity?.destroy();
    this.splitGlassKey?.destroy();
    this.splitGlassDepth?.destroy();
    this.rasterRigidPrimaryGeometry?.destroy();
    this.splitGeometry = undefined;
    this.splitGeometryView = undefined;
    this.splitOpaqueIdentity = undefined;
    this.splitOpaqueIdentityView = undefined;
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
    this.clearReusableFrame();
    this.clearPrimaryVisibilityCache();
    this.pickingFrameToken += 1;
    this.bindGroup = undefined;
    this.primitiveCandidateArena = undefined;
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
