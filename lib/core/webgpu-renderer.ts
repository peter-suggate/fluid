import { planSvoFluidCoverageRatio, type SvoFluidCoverageTriple } from "../svo/svo-fluid-coverage";
import {
  WebGpuSvoFluidCoverage,
  type WebGpuSvoFluidCoverageOptions,
  type WebGpuSvoFluidCoverageSource,
} from "../svo/webgpu-svo-fluid-coverage";
import { cameraBasis, dot } from "./math";
import { sceneLatticeDimensions } from "./scene-lattice";
import { canonicalScene, sceneRevision, sceneUsesFlatVoxelNormals, type CameraState, type SceneDescription } from "./model";
import { SCENE_SHAPE_PALETTE_LINEAR, sceneShapeCode, sceneShapeRenderHalfExtent_m } from "./scene-shape";
import { svoSceneLighting } from "../svo/svo-dry-scene-lighting";
import {
  buildSvoSceneLights,
  waterKeyDirectionalFromSceneLights,
} from "../svo/svo-light-abi";
import { boundingRadius, type RigidBodyState } from "./rigid-body";
import { decodeGPURigidBodyPoses, GPU_RIGID_RENDER_BYTES, SCENE_ENVIRONMENT_OWNER_BASE, type DrawnRigidBodyPose } from "./webgpu-rigid-body";
import type { GPUEulerianInfo, GPURigidLoad } from "./webgpu-eulerian";
import type { GPUQuality } from "./gpu-quality";
import { getMethod } from "./method-registry";
import type { GPUSolverInstance, InjectedLiquidBall, MethodParamValues, OverlayPipeline } from "./method-contract";
import { GridOverlayPipeline } from "./webgpu-grid-overlay";
import { FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE, requiredFluidDeviceLimits } from "./webgpu-device-limits";
import { RasterWaterPipeline, type WaterRenderDiagnostics, type WaterSurfacePresentationDiagnostics } from "./webgpu-water-pipeline";
import { environmentIndex, type EnvironmentId, defaultEnvironmentId } from "./environments";
import type { ScenePresentationMode } from "./scene-definition";
import { sceneHasTerrain } from "./terrain";
import { SecondaryParticleRenderPipeline } from "./webgpu-secondary-particles";
import type { SparseVoxelSceneRenderSource } from "./webgpu-voxel-debug";
import { cameraTanHalfFov, viewportAspect, viewportRayForPixel } from "./webgpu-camera";
import {
  type SvoPixelTrace,
  type SvoPixelTraceLayer,
} from "../svo/svo-pixel-trace";
import { DecorationOverlay } from "./webgpu-decoration-overlay";
import { FaceVelocityOverlay } from "./webgpu-face-velocity-overlay";
import type { PressureJournal } from "./pressure-journal";
import {
  isPressureJournalOverlayMode,
  pressureJournalOverlayChannel,
  PressureJournalOverlay,
  type PressureJournalOverlayMode,
} from "./webgpu-pressure-journal-overlay";
import {
  isStageLensOverlayMode,
  type StageLensOverlayMode,
  type StageLensReceipt,
} from "./stage-lens";
import { StageLensOverlay, type StageLensLayerReport } from "./webgpu-stage-lens-overlay";
import { TracerOverlay } from "./webgpu-tracer-overlay";
import { VISUALIZATION_CATALOG } from "./visualization-catalog";
import { assembleDecorations } from "./visualization-registry";
import { containerDecorationSpace, DecorationBuilder } from "./visualization-decorations";
import { buildVesselOutlineGeometry, sceneVesselPresentation } from "./vessel-outline";
import { WebGPUFluidCellTrace } from "./webgpu-fluid-cell-trace";
import type { FluidCellTrace } from "./fluid-cell-trace";
import type { FineBandCellContext } from "./fine-band-cell-model";
import { buildSparseVoxelDrySceneLightingMirrors, canConsumeSparseVoxelLighting, resolveSparseVoxelThickGlassBinderStatus, sparseVoxelDrySceneContractFailure, SparseVoxelDrySceneRenderer, SVO_DRY_SCENE_REVERSED_Z_NEAR_M, SVO_PRESENTATION_STARTUP_STAGES, svoPresentationResourcePlugin, type SparseVoxelDrySceneData, type SvoDryRigidBounds, type SvoDrySceneDirtyBounds } from "../svo/webgpu-svo-dry-scene";
import {
  buildSvoScenePrimitives,
} from "../svo/svo-scene-primitives";
import {
  buildSvoPrimitiveCandidates,
  createSvoPrimitiveCandidateRefitPlan,
  refitSvoPrimitiveCandidatesIncremental,
  svoPrimitiveCandidateBounds,
  type SvoPrimitiveCandidateRefitPlan,
} from "../svo/svo-primitive-candidates";
import { packSvoPrimitiveRecords, SVO_PRIMITIVE_RECORD_WORDS, type SvoPrimitiveDescriptor } from "../svo/svo-primitive-abi";
import { swayedPrimitiveDescriptor, type EnvironmentProxySway } from "./scenery-sway";
import {
  buildDefaultSvoMaterialRecords,
  packSvoMaterialTable,
  svoMaterialFromEnvironmentProxyMaterial,
  svoMaterialFunctionIdForEnvironmentProxy,
} from "../svo/svo-material-abi";
import { buildSvoSceneGlass } from "../svo/svo-scene-glass";
import { buildSvoSceneThickGlass } from "../svo/svo-scene-thick-glass";
import { sceneTerrainSurfaceModel } from "../svo/svo-terrain-material";
import {
  DEFAULT_SVO_LIGHTING_OPTIONS,
  resolveSvoPrimaryTraversal,
  type SvoLightingOptions,
  type SvoPrimaryTraversalMode,
  type SvoPrimaryTraversalScale,
} from "../svo/svo-render-options";
import { webGPUPlatformResourcePlugin } from "./webgpu-platform-resource";
import { disabledRenderStagesFrom, disabledRenderStagesKey } from "./render-stage-switches";
import { DEFAULT_SVO_RENDER_DIAGNOSTICS, normalizeSvoRenderDiagnostics, type SvoRenderDiagnostics } from "../svo/svo-render-diagnostics";
import { SparseVoxelRenderStageOverlay } from "../svo/webgpu-svo-stage-overlay";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, svoEnvironmentTreeRefinementDepth, svoRenderTuningKey, type SvoRenderTuning } from "../svo/svo-render-tuning";
import { SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../svo/svo-screen-space-termination";
import { isGPUInitializationAbort } from "./gpu-initialization";
import {
  createGlobalFineLevelSetConsumerSource,
  type GlobalFineLevelSetConsumerSource,
} from "./octree-consumer-sampling";
import { OCTREE_TECHNIQUE_OVERLAY_CODES, isOctreeTechniqueOverlayMode, type OctreeTechniqueOverlayMode } from "./octree-technique-debug";
import {
  sparseCM12DirtyOverlayCode,
  type SparseCM12DirtyOverlayMode,
} from "./sparse-cm12-dirty-visualizations";
import {
  automaticGPURecoveryEnabled,
  fluidExecutionDeviceFeatures,
} from "./gpu-startup";
import { initialRasterPresentationReadiness, requiresFencedInitialRasterPresentation } from "./gpu-t0-presentation";
import { liveSvoSceneResourcePlugin, WebGPULiveSvoScene } from "../svo/webgpu-live-svo-scene";
import { planSceneRuntime } from "./scene-runtime";
import type { GPUStatus } from "./gpu-status";
import {
  resolveEffectiveRendererStatus,
  type EffectiveRendererStatus,
} from "./renderer-status";
import {
  CPUPerformanceTrace,
  GPUPassTimestampRecorder,
  GPUQueueWallPerformanceTraceRecorder,
  type GPUTimestampPhase,
  type PerformanceTrace,
} from "./performance-trace";
import { usePerformanceInstrumentationStore } from "./stores/performance-instrumentation-store";
import { FencePartitionedFrameSampler } from "./webgpu-frame-band-sampler";
import {
  RENDER_FRAME_STAGE_TRACE,
  RenderFrameSeamRecorder,
  type RenderFrameManifest,
  type RenderFrameSeam,
  type RenderFrameStageId,
} from "./render-frame-stages";
import type { ResourcePluginDefinition } from "./resource-readiness";
import {
  invalidateGPUCompilationManager,
  managedGPUDevice,
} from "./gpu-compilation-manager";
import type {
  SparseWorldDevice,
  SparseWorldFault,
  SparseWorldPresentation,
  SparseWorldStatus,
} from "../sparse-world";

/** One item executing and one queued keeps the GPU busy without visible FIFO bursts. */
export const BROWSER_GPU_THROUGHPUT_DEPTH = 2;

interface SparseWorldSnapshot {
  readonly deviceStatus: SparseWorldDevice["status"];
  readonly deviceFault: SparseWorldFault | undefined;
  readonly status: SparseWorldStatus;
  readonly presentation: SparseWorldPresentation;
}

/**
 * Solver advances encoded per presented frame.
 *
 * `planGPUAdvance` clamps one `advanceTo` to a single `maxDt_s`, and the
 * renderer called it once per draw — so the browser bought 8 ms of simulation
 * per presented frame and paid the frame's whole fixed cost for it. Raising
 * this is the A/B that attributes that cost:
 *
 *  - frame rate holds and the transport bar's `ACTUAL ×` doubles → the cost is
 *    per PRESENTATION, and amortising it is the real-time-factor lever;
 *  - frame rate halves and `ACTUAL ×` holds → the cost is per ADVANCE, the
 *    solver's own GPU work, and amortising buys nothing.
 *
 * Set back to 1 to restore the previous cadence. Above 1, a presented frame
 * shows a state that jumped this many steps, so interactivity granularity
 * coarsens even when throughput improves.
 */
export const GPU_ADVANCES_PER_PRESENTATION = 1;
export const SVO_CAMERA_CHANGING_FRAME = -2;

/**
 * Exact old/new coverage for each procedurally moving analytic primitive.
 *
 * The list stays per primitive rather than becoming one scene-sized union.
 * Bounded sway retains the sparse reference pose, but the render arena still
 * records the exact old/new dependency bounds for diagnostics and any future
 * localized render cache that consumes them.
 */
export function svoSwayDirtyBounds(
  previous: readonly SvoPrimitiveDescriptor[],
  current: readonly SvoPrimitiveDescriptor[],
  sway: readonly (EnvironmentProxySway | undefined)[],
): readonly SvoDrySceneDirtyBounds[] {
  if (previous.length !== current.length || previous.length !== sway.length) {
    throw new RangeError("Sway dirty-bound inputs must describe the same primitive arena");
  }
  const dirty: SvoDrySceneDirtyBounds[] = [];
  for (let index = 0; index < current.length; index += 1) {
    if (!sway[index]) continue;
    const before = previous[index], after = current[index];
    const oldBounds = svoPrimitiveCandidateBounds(before);
    const newBounds = svoPrimitiveCandidateBounds(after);
    dirty.push({
      minimum: [
        Math.min(oldBounds.minimum_m.x, newBounds.minimum_m.x),
        Math.min(oldBounds.minimum_m.y, newBounds.minimum_m.y),
        Math.min(oldBounds.minimum_m.z, newBounds.minimum_m.z),
      ],
      maximum: [
        Math.max(oldBounds.maximum_m.x, newBounds.maximum_m.x),
        Math.max(oldBounds.maximum_m.y, newBounds.maximum_m.y),
        Math.max(oldBounds.maximum_m.z, newBounds.maximum_m.z),
      ],
    });
  }
  return dirty;
}

/** Submit one solver advance toward the prepared simulation clock. */
export function submitNextPreparedGPUAdvance(fluid: GPUSolverInstance, time_s: number, bodies: RigidBodyState[]) {
  const previousSubmittedTime = fluid.info.submittedTime_s ?? 0;
  if (previousSubmittedTime + 1e-9 < time_s) fluid.advanceTo(time_s, bodies);
  const submittedTime = fluid.info.submittedTime_s ?? previousSubmittedTime;
  return { previousSubmittedTime, submittedTime };
}

/** Bound physics queue depth to the fixed throughput window. */
export function canQueuePreparedGPUAdvance(pendingAdvances: number, maximumPendingAdvances: number) {
  return pendingAdvances < Math.max(1, maximumPendingAdvances);
}

/** Column-major right-handed world-to-WebGPU-clip transform for voxel raster passes. */
export function voxelViewProjectionMatrix(camera: CameraState, aspect: number, near = 0.01, far = 100): Float32Array {
  const basis = cameraBasis(camera), position = basis.position;
  const view = new Float32Array([
    basis.right.x, basis.up.x, -basis.forward.x, 0,
    basis.right.y, basis.up.y, -basis.forward.y, 0,
    basis.right.z, basis.up.z, -basis.forward.z, 0,
    -dot(basis.right, position), -dot(basis.up, position), dot(basis.forward, position), 1
  ]);
  const safeNear = Math.max(1e-4, near), safeFar = Math.max(safeNear + 1, far);
  const focal = 1 / cameraTanHalfFov(camera);
  const projection = new Float32Array([
    focal / Math.max(1e-4, aspect), 0, 0, 0,
    0, focal, 0, 0,
    0, 0, safeFar / (safeNear - safeFar), -1,
    0, 0, safeNear * safeFar / (safeNear - safeFar), 0
  ]);
  const result = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) for (let row = 0; row < 4; row += 1) {
    let value = 0;
    for (let index = 0; index < 4; index += 1) value += projection[index * 4 + row] * view[column * 4 + index];
    result[column * 4 + row] = value;
  }
  return result;
}

/**
 * Debug cross-section of the solver grid, after Chentanez & Mueller Fig. 2:
 * teal tall cells, outlined regular cells with centre sample dots, and the
 * tall cell's top/bottom subcell samples. Uniform grids render as an
 * all-regular band. `position` selects the slice layer as a 0..1 fraction.
 *
 * `mode` recolors represented cells from GPU-resident fields: "cfl" shows the
 * per-cell component CFL at the solver's substep dt (the quantity whose
 * maximum picks the adaptive substep count), "speed" the velocity magnitude
 * normalized by the last reported liquid maximum. Both sample live solver
 * textures in the overlay shader — no readback is involved.
 */
export type GridOverlayMode = "structure" | "resolution" | "optical" | "cfl" | "speed" | "phi" | "divergence" | "pressure" | "projection" | "representation" | "density" | "tracers" | "face-velocity"
  | PressureJournalOverlayMode | OctreeTechniqueOverlayMode | SparseCM12DirtyOverlayMode
  | StageLensOverlayMode;

export interface GridOverlayConfig {
  /** Slice axes, or a ray-integrated diagnostic through the complete volume. */
  axis: "off" | "z" | "x" | "y" | "volume";
  position: number;
  mode?: GridOverlayMode;
  /** Scrubber position within the selected lens's phases. */
  lensPhase?: number;
}

export type OptionalRendererPipeline =
  | "grid-overlay"
  | "technique-overlay"
  | "technique-audit-overlay"
  | "svo-dry-scene"
  | "secondary-particles"
  | "decoration-overlay"
  | "fluid-cell-trace"
  | "tracer-overlay"
  | "face-velocity-overlay"
  | "pressure-journal-overlay"
  | "stage-lens"
  | "svo-stage-overlay";

/**
 * Live pixel-trace diagnostic state.
 *
 * The probe traces the pixel under `normalized`, and the overlay draws the
 * previous frame's decoded result: the readback is one frame behind by
 * construction, which is invisible at pointer rates and is what keeps the
 * diagnostic off the critical path.
 */
export type PixelTraceStatus =
  /** The sparse dry scene is not the active presentation, so there is nothing to trace. */
  | "path-inactive"
  /** The probe's shader and pipeline are still being built. */
  | "compiling"
  /** This device cannot host the probe; the reason was logged once. */
  | "unsupported"
  /** Armed and able, but no trace has been read back yet. */
  | "waiting"
  | "live";

/**
 * Request for the per-cell fluid work diagnostic.
 *
 * Selection is a pixel, exactly as for the ray probe: the gather shader marches
 * that camera ray to decide which pressure cell the pointer means, so the two
 * diagnostics share one gesture and one camera transform.
 */
export interface FluidCellTraceConfig {
  /** Viewport fractions, 0..1 from the top-left. */
  readonly normalizedX: number;
  readonly normalizedY: number;
  /**
   * A pinned cell stops re-aiming, so orbiting cannot silently reselect it.
   *
   * It does not change what is drawn or reported: a hovered cell shows exactly
   * what a pinned one shows, because needing to commit before you can read
   * anything is what makes a picker tedious to explore with. Pinning is for
   * holding a cell still while the camera moves around it, nothing more.
   */
  readonly pinned: boolean;
  /**
   * Which leaf along the ray to describe, nearest first. The pointer alone can
   * only ever name the first, which on a liquid is a surface cell; stepping this
   * is how an interior unknown gets selected. Clamped to the run in the shader.
   */
  readonly hitIndex?: number;
  /** Visualization ids to draw. Absent draws nothing for this selection. */
  readonly layers?: readonly string[];
  readonly widthScale?: number;
  /**
   * Solve policy the dependency cone is grown against. Supplied by the caller
   * because the tail policy lives with the solver, not with the picked cell;
   * absent simply means no cone.
   */
  readonly solvePolicy?: {
    readonly outerIterations: number;
    readonly levels: number;
    readonly smoothsPerLevel: number;
  };
}

export interface PixelTraceConfig {
  /** Viewport fractions, 0..1 from the top-left. */
  readonly normalizedX: number;
  readonly normalizedY: number;
  readonly layers: readonly SvoPixelTraceLayer[];
  /** 0 replays the ray's work from the camera outward; 1 shows all of it. */
  readonly reveal: number;
  readonly occludedAlpha?: number;
  readonly widthScale?: number;
  /**
   * A pinned trace keeps its recorded geometry while the camera orbits it, and
   * stops probing: re-tracing the pixel from a moved camera would quietly answer
   * a different ray. `normalizedX`/`normalizedY` are then the pinned position,
   * not the pointer's.
   */
  readonly pinned?: boolean;
  /**
   * Re-trace the pinned pixel anyway, because the scene the frozen answer
   * describes has moved on. Only the caller knows whether that is honest: it is
   * the same ray only while the camera has not moved since the pin.
   */
  readonly refresh?: boolean;
}

/**
 * Pipeline compilation requested by the current presentation. The sparse
 * dry-scene renderer owns the finished view alongside the authoritative water
 * path; structural diagnostics are optional overlays on that same frame.
 */
export function optionalRendererPipelineRequests(
  gridOverlay: GridOverlayConfig | undefined,
  simulationRunning: boolean,
  secondaryParticlesAvailable: boolean,
  pixelTraceActive = false,
  fluidCellTraceActive = false,
  stageViewActive = false,
  sparsePresentationRequired = true,
  vesselOutlineActive = false,
): OptionalRendererPipeline[] {
  const requested: OptionalRendererPipeline[] = [];
  if (gridOverlay && gridOverlay.axis !== "off") {
    const technique = Boolean(gridOverlay.mode && isOctreeTechniqueOverlayMode(gridOverlay.mode));
    // Markers are their own instanced draw over the finished frame, so they
    // compile neither the generic slice raymarch nor the technique programs.
    if (gridOverlay.mode === "tracers") requested.push("tracer-overlay");
    else if (gridOverlay.mode === "face-velocity") requested.push("face-velocity-overlay");
    else if (isPressureJournalOverlayMode(gridOverlay.mode)) {
      requested.push("pressure-journal-overlay");
    }
    else if (gridOverlay.mode && isStageLensOverlayMode(gridOverlay.mode)) {
      requested.push("stage-lens");
    }
    else if (!technique) requested.push("grid-overlay");
    else requested.push("technique-overlay", "technique-audit-overlay");
  }
  if (sparsePresentationRequired) requested.push("svo-dry-scene");
  if (simulationRunning && secondaryParticlesAvailable) requested.push("secondary-particles");
  // One generic line compositor serves diagnostics and the cheap vessel cue.
  if (pixelTraceActive || vesselOutlineActive) requested.push("decoration-overlay");
  if (fluidCellTraceActive) requested.push("fluid-cell-trace", "decoration-overlay");
  // A session that never opens a render stage view never compiles its pass.
  if (stageViewActive) requested.push("svo-stage-overlay");
  return requested;
}

/** Everything the renderer needs to know about the selected method. */
export interface SimulationRunConfig {
  methodId: string;
  quality: GPUQuality;
  values: MethodParamValues;
  /** Controller-owned identity for a fresh t=0 simulation. */
  simulationEpoch?: number;
}

export function structuralMethodValues(config: SimulationRunConfig): MethodParamValues {
  const runtime = new Set(getMethod(config.methodId).runtimeParamKeys ?? []);
  return Object.fromEntries(Object.entries(config.values).filter(([key]) => !runtime.has(key)));
}

/** Renderer-only worlds are method-independent; fluid worlds require a GPU solver factory. */
export function canInitializeGPUSceneSource(scene: SceneDescription, methodId: string): boolean {
  const method = getMethod(methodId);
  return !planSceneRuntime(scene).fluidSolver || Boolean(method.createSolver || method.createSolverAsync);
}

/**
 * Construction-time identity, split into the three tiers of
 * docs/WYSIWYG_EDITOR_PLAN.md Phase 1b.
 *
 * `structuralKey` fixes the lattice shape and arena capacities: a mismatch can
 * only be answered by building a new solver. `seedKey` covers scene-derived
 * GPU inputs — the phi seed, solids, terrain, inflow — which a warm re-seed
 * will eventually refresh in place against the existing allocations.
 * `uniformKey` is pure scalars that a hot uniform write could carry with no
 * reset at all.
 *
 * Only the tier boundaries exist today; `gpuSceneSolverKey` still concatenates
 * all three, so behaviour is unchanged apart from the terrain fix below. The
 * warm-reset path replaces the seed tier's response, and the uniform tier's
 * response, without moving these boundaries again.
 */
/**
 * The document's own half of the structural tier: what the lattice is, and the
 * wall conditions compiled into the pipelines that walk it.
 *
 * Split out from `gpuSceneStructuralKey` because two documents can be compared
 * without a run config — which is what an editor commit needs in order to ask
 * "does this edit require a reset?" before it takes one.
 */
export function sceneStructuralKey(scene: SceneDescription): string {
  // The lattice is keyed in cells, not metres. A world scale retains these
  // allocations; the warm re-seed republishes the sparse presentation domain's
  // origin and cell size separately.
  const cellSize_m = scene.voxelDomain.finestCellSize_m;
  const bounds = scene.voxelDomain.bounds_m;
  const boundsCells = bounds
    ? [bounds.min.x, bounds.min.y, bounds.min.z, bounds.max.x, bounds.max.y, bounds.max.z]
      .map((value) => Math.round(value / cellSize_m)).join(",")
    : "none";
  const lattice = `${sceneLatticeDimensions(scene).join("x")}:${scene.voxelDomain.brickSize_cells}:${boundsCells}`;
  return `fluid-${planSceneRuntime(scene).fluidSolver}:${lattice}:${scene.container.shape ?? "box"}:${scene.container.top}:${scene.container.fluidWallMode}:${scene.container.depthBoundary ?? "closed"}`;
}

export function gpuSceneStructuralKey(scene: SceneDescription, config: SimulationRunConfig): string {
  return `${config.methodId}:${config.quality}:${JSON.stringify(structuralMethodValues(config))}:${sceneStructuralKey(scene)}`;
}

/**
 * Scene-derived solver inputs. `scene.terrain` belongs here and was previously
 * absent from the key entirely, so a terrain edit never reached the solver —
 * the editor's terrain handles depend on this being fixed.
 *
 * The container extents belong here rather than in the structural tier. A cell
 * measures `extent / dimension` metres: the dimension is structural, so the
 * extent is what is left, and the re-seed already recomputes the cell size from
 * the incoming scene (`WebGPUOctreeProjection.reseed`). It has to stay in some
 * key — a world scale that changed neither the lattice nor the fill would
 * otherwise leave the solver running at the old scale.
 *
 * `voxelDomain.finestCellSize_m` is deliberately absent: it is a *request*, and
 * the lattice rounds it. Two cell sizes that round to the same dimensions
 * describe the same solver, and the size actually simulated is the extent over
 * that dimension — both already keyed.
 */
export function gpuSceneSeedKey(scene: SceneDescription): string {
  const c = scene.container;
  return `${c.width_m}:${c.height_m}:${c.depth_m}:${c.shape ?? "box"}:${c.fillFraction}:${scene.fluid.initialCondition}:${JSON.stringify(scene.fluid.initialDamBreakDimensions_m ?? null)}:${JSON.stringify(scene.fluid.initialDamBreakOrigin_m ?? null)}:${JSON.stringify(scene.fluid.initialBrickSeeds_m ?? null)}:${scene.fluid.initialBrickSeedsAdditive ?? false}:${JSON.stringify(scene.fluid.initialLiquidVolumes ?? null)}:${JSON.stringify(scene.terrain ?? null)}:${inflowBudgetKey(scene.fluid.inflow)}`;
}

/**
 * The rigid-body facts a solver's *allocations* are shaped by.
 *
 * *Which* bodies, where they are, and what shape they take are absent on
 * purpose: every step already writes the whole roster into the rigid state
 * buffer through `syncBodies`, and the solid vertex SDF is re-encoded from that
 * buffer, so moving, reshaping or re-materialing a body is adopted live. That
 * is what keeps `rigidBodyRosterKey` in the uniform tier.
 *
 * What remains is two facts, both of them thresholds rather than counts:
 *
 *  - Are there *any* solids? `octreeSparseWorldRequired`,
 *    `planOctreeSolidCellAllocation` and `planOctreeSurfaceStateAllocation`
 *    each allocate a dense solid field for the whole lattice or skip it.
 *  - Does any of them move? `planOctreeSolveTail` scores one extra iteration
 *    for a moving immersed boundary, regardless of how many there are.
 *
 * The *count* used to be here too, and it is the reason dropping a body into
 * running water restarted the clock: the environment proxies were numbered
 * `rigidBodies.length + ownerIndex`, so the roster length was part of the
 * render source's ABI and adding one renumbered every scenery object. It is
 * now `SCENE_ENVIRONMENT_OWNER_BASE + ownerIndex` — pinned at the fixed rigid
 * capacity every rigid arena is already sized to — so the count names nothing.
 *
 * A method that allocates nothing from either remaining fact declares
 * `adoptsRigidRosterShape` and gets a constant, which is what makes the first
 * body dropped into a running uniform scene warm as well.
 */
function rigidAllocationKey(scene: SceneDescription, methodId: string): string {
  if (getMethod(methodId).capabilities?.adoptsRigidRosterShape) return "live";
  const bodies = scene.rigidBodies;
  // Terrain already forces the sparse world, so on a terrain scene the presence
  // bit answers a question nothing asks. Reading it anyway is what made the
  // first body dropped onto a landscape restart an octree solve that would have
  // allocated exactly the same thing either way.
  const presence = sceneHasTerrain(scene) ? "terrain" : `${bodies.length > 0}`;
  return `${presence}:${bodies.some((body) => body.motion !== "static")}`;
}

/** The roster itself, which a live solver re-uploads rather than rebuilds for. */
function rigidBodyRosterKey(bodies: SceneDescription["rigidBodies"]): string {
  return JSON.stringify(bodies);
}

/**
 * The inflow inputs an allocation is sized from.
 *
 * `fluidFootprint` budgets the fine narrow band from the volume the nozzle will
 * deliver over the whole run — `integratedInflowVolume`, which is the radius,
 * the speed and the timing window — and `includeWholeDomainPressureSupport`
 * reads presence alone. Nothing here is *where* the nozzle is or *which way* it
 * points, because neither changes what has to be allocated. Those live in the
 * uniform tier, where `writeParams` already packs them, which is what makes
 * dragging the hose a buffer write instead of a restart.
 *
 * Speed rather than the velocity vector, deliberately: the integrated volume is
 * flow rate times time, and flow rate is area times speed. Turning the hose
 * without slowing it delivers the same water into the same budget.
 */
function inflowBudgetKey(inflow: SceneDescription["fluid"]["inflow"]): string {
  if (!inflow) return "none";
  const v = inflow.velocity_m_s;
  return `${inflow.radius_m}:${Math.hypot(v.x, v.y, v.z)}:${inflow.start_s}:${inflow.end_s}:${inflow.ramp_s}`;
}

/**
 * Whether landing this edit costs the simulation its timeline.
 *
 * An editor commit used to reset unconditionally, so every gesture — including
 * ones the live solver adopts through a single buffer write — zeroed the clock,
 * cleared the diagnostics store and paused transport. That is the difference
 * between an editor with a simulation in it and a simulation you can edit.
 *
 * The rule is the tier boundary and nothing else: a structural or seed-tier
 * difference has to be answered by a rebuild or a re-seed, both of which start
 * again from t=0. Everything left over is a uniform-tier difference, which
 * `applySceneUniforms` adopts on the running solver. Both keys are compared,
 * not just the seed one, because a lattice change is the case where continuing
 * would be worst: the solver would keep running at the old shape.
 */
export function sceneEditRequiresReset(before: SceneDescription, after: SceneDescription, methodId: string): boolean {
  return sceneStructuralKey(before) !== sceneStructuralKey(after)
    || gpuSceneSeedKey(before) !== gpuSceneSeedKey(after)
    || rigidAllocationKey(before, methodId) !== rigidAllocationKey(after, methodId);
}

/** Where the nozzle is, which way it points, and how far it reaches. */
function inflowAimKey(inflow: SceneDescription["fluid"]["inflow"]): string {
  if (!inflow) return "none";
  const { center_m: c, velocity_m_s: v } = inflow;
  const speed = Math.hypot(v.x, v.y, v.z) || 1;
  return `${c.x},${c.y},${c.z}:${v.x / speed},${v.y / speed},${v.z / speed}:${inflow.length_m}`;
}

/**
 * Inputs a live solver adopts through `applySceneUniforms` — the scalars no
 * lattice or seed depends on, plus the nozzle's placement and aim, which reach
 * the GPU as params rather than as geometry.
 */
export function gpuSceneUniformKey(scene: SceneDescription): string {
  return `${scene.fluid.density_kg_m3}:${scene.fluid.dynamicViscosity_Pa_s}:${scene.fluid.surfaceTension_N_m}:${scene.fluid.gravity_m_s2.y}:${scene.numerics.fixedDt_s}:${scene.numerics.maxDt_s}:${inflowAimKey(scene.fluid.inflow)}:${rigidBodyRosterKey(scene.rigidBodies)}:${refinementRegionKey(scene)}:${JSON.stringify(scene.solidVoxels)}`;
}

/**
 * The authored refinement regions.
 *
 * Uniform tier by construction rather than by concession: a region only changes
 * the topology gate's split decision, and the topology is re-derived from the
 * reset size every epoch, so a new box is adopted by the next candidate epoch
 * of the running solver. Nothing it can say changes an allocation, a seed or
 * the lattice — which is exactly what makes drawing one an experiment you can
 * watch rather than a restart you wait for.
 */
function refinementRegionKey(scene: SceneDescription): string {
  const regions = scene.fluid.refinementRegions;
  if (!regions || regions.length === 0) return "none";
  return regions
    .map((region) => `${region.rule}@${region.minimumCellSize_cells}-${region.maximumCellSize_cells ?? "auto"}:${region.min_m.x},${region.min_m.y},${region.min_m.z}>${region.max_m.x},${region.max_m.y},${region.max_m.z}`)
    .join("|");
}

/**
 * Content identity for every input that requires a new solver.
 *
 * The uniform tier is deliberately absent: those scalars are adopted by the
 * live solver through `applySceneUniforms`, so editing density or gravity is a
 * uniform write rather than a ~350-pipeline rebuild. A method that does not
 * implement `applySceneUniforms` gets the uniform tier folded back in by
 * `solverKey` below, so it keeps the old rebuild behaviour instead of silently
 * ignoring the edit.
 */
export function gpuSceneSolverKey(scene: SceneDescription, config: SimulationRunConfig): string {
  return `${config.simulationEpoch ?? 0}:${gpuSceneStructuralKey(scene, config)}:${gpuSceneSeedKey(scene)}:${rigidAllocationKey(scene, config.methodId)}`;
}


/**
 * One sphere enclosing every rigid body, for the dry scene's shadow and contact
 * rays to reject against before they read the body array.
 *
 * The centre is the midpoint of the axis-aligned bounds rather than a centroid
 * of positions: a centroid is pulled toward clusters and leaves an outlier body
 * near the rim, which inflates the radius for every other ray in the frame.
 * Returns undefined for an empty scene so the caller publishes "no bodies"
 * instead of a degenerate sphere at the origin.
 */
export function svoDryRigidBounds(bodies: readonly RigidBodyState[]): SvoDryRigidBounds | undefined {
  if (bodies.length === 0) return undefined;
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  for (const body of bodies) {
    const radius = boundingRadius(body);
    const position = [body.position_m.x, body.position_m.y, body.position_m.z];
    for (let axis = 0; axis < 3; axis += 1) {
      low[axis] = Math.min(low[axis], position[axis] - radius);
      high[axis] = Math.max(high[axis], position[axis] + radius);
    }
  }
  const centre = [0, 1, 2].map((axis) => (low[axis] + high[axis]) / 2) as [number, number, number];
  let radius = 0;
  for (const body of bodies) {
    const position = [body.position_m.x, body.position_m.y, body.position_m.z];
    radius = Math.max(radius, Math.hypot(...[0, 1, 2].map((axis) => position[axis] - centre[axis]))
      + boundingRadius(body));
  }
  return { centre_m: centre, radius_m: radius };
}

export interface RendererFrameMetrics {
  cpu?: PerformanceTrace;
  presentation?: PerformanceTrace;
  /** Trustworthy per-stage GPU pass costs; its total is a pass sum, not frame time. */
  presentationStages?: PerformanceTrace;
  /**
   * Fence-partitioned band walls from the most recent sampling frame: real
   * queue-wall cost per encode band, including the render passes the
   * hardware timestamps cannot price. Sampled one frame in sixteen while
   * instrumentation is on; that frame publishes no queue-wall frame trace, so
   * the rolling frame mean never contains a partitioned frame.
   */
  presentationBands?: PerformanceTrace;
  /**
   * What the newest recorded frame encoded, stage by stage.
   *
   * Timestamps say how long a stage took; this says whether it ran. Without
   * it, a stage that encoded nothing and a stage whose render passes cannot be
   * timed both arrive as silence, and the band-wall attribution charges the
   * band's whole wall to whichever one is silent — which is how a settled
   * world came to report 27.9 ms of maintenance.
   */
  presentationStageManifest?: RenderFrameManifest;
  context: string;
  methodId: string;
  /** True only when this draw encoded and submitted a presentation command buffer. */
  presentationSubmitted: boolean;
  /** Latest presentation evidence; independent of solver publication authority. */
  waterSurfacePresentation?: WaterSurfacePresentationDiagnostics;
}

interface PendingInitialRasterPresentation {
  readonly solver: GPUSolverInstance;
  readonly solverGeneration: number;
  readonly requestGeneration: number;
  readonly resource: ResourcePluginDefinition;
  submitted: boolean;
}

interface PendingGPUAdvanceCompletion {
  readonly solver: GPUSolverInstance;
  readonly solverGeneration: number;
  readonly submittedTime_s: number;
}

class PendingLiveSvoPresentation {
  private phase: "awaiting-attachment" | "attached" | "submitted" = "awaiting-attachment";

  constructor(
    readonly solver: GPUSolverInstance,
    readonly source: NonNullable<GPUSolverInstance["sparseVoxelSceneSource"]>,
    readonly solverGeneration: number,
    readonly requestGeneration: number,
    readonly startedAt_ms: number,
  ) {}

  attach(): boolean {
    if (this.phase !== "awaiting-attachment") return false;
    this.phase = "attached";
    return true;
  }

  submit(): boolean {
    if (this.phase !== "attached") return false;
    this.phase = "submitted";
    return true;
  }

  get attached(): boolean { return this.phase !== "awaiting-attachment"; }
  get submitted(): boolean { return this.phase === "submitted"; }
}

const upscaleShader = /* wgsl */ `
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
struct Out { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> Out {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: Out; out.position = vec4f(positions[index], 0.0, 1.0); out.uv = positions[index] * 0.5 + 0.5; return out;
}
@fragment fn fragmentMain(input: Out) -> @location(0) vec4f { return textureSample(source, sourceSampler, vec2f(input.uv.x, 1.0-input.uv.y)); }
`;

/**
 * Construct the exact sparse dry-scene pipeline selected by the product.
 *
 * Kept outside `FluidLabRenderer` so the default Dawn startup receipt can
 * compile the same program bundle instead of substituting the water renderer.
 */
export function createProductionSparseVoxelDrySceneRenderer(
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  bodyBuffer: GPUBuffer,
  primaryTraversal: SvoPrimaryTraversalMode,
): SparseVoxelDrySceneRenderer {
  if (primaryTraversal === "raster"
    && device.limits.maxColorAttachmentBytesPerSample < FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE) {
    throw new RangeError(
      `Requested SVO raster primary needs maxColorAttachmentBytesPerSample >= ${FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE}; device exposes ${device.limits.maxColorAttachmentBytesPerSample}`,
    );
  }
  const traversal = primaryTraversal === "raster"
    ? "raster-primary" as const : "canonical-parametric" as const;
  const rasterArms = traversal === "raster-primary";
  return new SparseVoxelDrySceneRenderer(
    device,
    uniformBuffer,
    bodyBuffer,
    "rgba16float",
    traversal,
    "off",
    "split",
    rasterArms ? SVO_SCREEN_SPACE_TERMINATION_CONTRACT.defaultThresholdPixels : 0,
    rasterArms,
    rasterArms,
    true,
  );
}

export class FluidLabRenderer {
  private device?: GPUDevice;
  private disposed = false;
  private initializationPromise?: Promise<void>;
  private shutdownPromise?: Promise<void>;
  private deviceLost = false;
  private context?: GPUCanvasContext;
  private upscalePipeline?: GPURenderPipeline;
  private upscaleSampler?: GPUSampler;
  private upscaleBindGroup?: GPUBindGroup;
  private waterPipeline?: RasterWaterPipeline;
  private secondaryParticlePipeline?: SecondaryParticleRenderPipeline;
  private svoDryScenePipeline?: SparseVoxelDrySceneRenderer;
  private svoDrySceneSource?: SparseVoxelSceneRenderSource;
  private svoDrySceneData?: SparseVoxelDrySceneData;
  /** Live render publication identity; deliberately independent of solver identity. */
  private renderSceneKey = "";
  private renderSceneStamp = 0;
  private renderSceneRevision = 0;
  private renderSceneTerrainContentStamp?: string;
  private liveSceneAnimation?: {
    readonly rest: readonly SvoPrimitiveDescriptor[];
    readonly sway: readonly (EnvironmentProxySway | undefined)[];
    readonly swayIndices: readonly number[];
    readonly current: SvoPrimitiveDescriptor[];
    readonly records: Uint32Array<ArrayBuffer>;
    readonly refitPlan: SvoPrimitiveCandidateRefitPlan;
    origin_ms?: number;
  };
  /** Deduplicates a visible failed-closed diagnostic while the previous pose remains authoritative. */
  private liveSceneAnimationFailure?: string;
  private svoPublicationFailure?: string;
  private gridOverlayPipeline?: GridOverlayPipeline;
  /** Registered by the running method; core never names an overlay module. */
  private techniqueOverlayPipeline?: OverlayPipeline;
  private techniqueAuditOverlayPipeline?: OverlayPipeline;
  private decorationOverlayPipeline?: DecorationOverlay;
  private tracerOverlayPipeline?: TracerOverlay;
  private faceVelocityOverlayPipeline?: FaceVelocityOverlay;
  private pressureJournalOverlayPipeline?: PressureJournalOverlay;
  private stageLensOverlayPipeline?: StageLensOverlay;
  private fluidCellTracePipeline?: WebGPUFluidCellTrace;
  private svoStageOverlay?: SparseVoxelRenderStageOverlay;
  private latestFluidCellTraceValue?: FluidCellTrace;
  private fluidCellTraceRevisionValue = 0;
  private fluidCellTraceReadInFlight = false;
  /** Latest decoded trace, and a revision the UI polls instead of a callback. */
  private latestPixelTraceValue?: SvoPixelTrace;
  private pixelTraceRevisionValue = 0;
  /** Identity of the assembled decorations currently uploaded. */
  private decorationGeometryKey = "";
  private pixelTraceReadInFlight = false;
  /**
   * Everything a trace's answer depends on except the camera and the pixel: the
   * scene epoch, the presentation, and the traversal tuning. The revision is what
   * lets a held trace be recognized as describing a frame that no longer exists.
   */
  private pixelTraceSceneKey = "";
  private pixelTraceSceneRevisionValue = 0;
  /** Revision the in-flight probe was encoded under, carried to the trace it produces. */
  private pixelTraceEncodedSceneRevision = 0;
  private pixelTraceSceneRevisionOfTrace = -1;
  /** Optional programs compile only after their explicit presentation mode is used. */
  private readonly optionalPipelineTasks = new Map<OptionalRendererPipeline, Promise<void>>();
  /** A compile failure is sticky for this device; do not hammer a fragile driver every frame. */
  private readonly failedOptionalPipelines = new Set<OptionalRendererPipeline>();
  /** The sticky failure remains inspectable; a failed pipeline is never silent. */
  private readonly optionalPipelineFailures = new Map<OptionalRendererPipeline, string>();
  /**
   * Primary traversal is fixed when the dry-scene pipeline is built, so the
   * *resolved* mode is recorded here and takes effect by retiring the pipeline:
   * the next `ensureOptionalPipeline` sweep rebuilds against it.
   *
   * Resolved, not requested: `resolveSvoPrimaryTraversal` upgrades the toggle's
   * `raster` to `traced` once the scene emits more proxies than the target has
   * pixels. The resolution is a pure function of the toggle and the scene's
   * scale, so re-running it every frame is stable and only a scale change moves
   * it.
   */
  private requestedPrimaryTraversal: SvoPrimaryTraversalMode =
    DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal ?? "raster";
  private presentationTexture?: GPUTexture;
  private presentationTextureKey = "";
  private activeRenderScale = 1;
  private uniformBuffer?: GPUBuffer;
  private bodyBuffer?: GPUBuffer;
  private fluidTexture?: GPUTexture;
  /** Dense surface currently bound into presentation pipelines. */
  private attachedSurfaceTexture?: GPUTexture;
  private columnBaseTexture?: GPUTexture;
  private gridCellTexture?: GPUTexture;
  private velocityFallbackTexture?: GPUTexture;
  private pressureSamplesFallbackTexture?: GPUTexture;
  private scalarFallbackTexture?: GPUTexture;
  private gpuFluid?: GPUSolverInstance;
  /** Renderer-owned sparse source for fluid methods that do not publish one. */
  private svoSceneSidecar?: WebGPULiveSvoScene;
  private readonly retiredGPUFluids = new Set<GPUSolverInstance>();
  private gpuFluidKey = "";
  private gpuFluidPendingKey = "";
  private gpuFluidPending?: Promise<void>;
  private gpuFluidInitializationAbort?: AbortController;
  private gpuFluidRequestGeneration = 0;
  private adapterName = "WebGPU adapter";
  private gpuInfoCallback?: (info: GPUEulerianInfo) => void;
  private gpuPressureJournalCallback?: (journal: PressureJournal | undefined) => void;
  private gpuStageLensCallback?: (
    receipt: StageLensReceipt | undefined,
    layers: readonly StageLensLayerReport[],
  ) => void;
  /** Lens the last published receipt described, so the clearing post fires once. */
  private publishedStageLens?: string;
  private gpuRigidLoadCallback?: (loads: GPURigidLoad[]) => void;
  private gpuAdvanceCompletedCallback?: (time_s: number) => void;
  private effectiveRendererStatusCallback?: (status: EffectiveRendererStatus) => void;
  private lastEffectiveRendererStatus?: EffectiveRendererStatus;
  private svoPickingAvailable = false;
  private lastSvoPickingBodies: RigidBodyState[] = [];
  /**
   * Rigid poses on their way back to the host, and the two staging buffers they
   * travel in.
   *
   * Two slots and no allocation per frame: the copy rides the presentation
   * encoder that already copies these records into `bodyBuffer`, so publishing
   * them costs the map, not a submit. When both slots are busy the frame simply
   * does not publish — the host's copy is one frame older, which nothing here
   * can tell apart from the frame it is already behind.
   */
  private rigidPoseStaging: { buffer: GPUBuffer; pending: boolean }[] = [];
  private latestRigidBodyPoses: readonly DrawnRigidBodyPose[] = [];
  private latestRigidBodyPoseRevision = 0;
  private svoSourceAvailable = false;
  private svoGlassSupported = true;
  private svoMaterialsSupported = true;
  private svoLightingSupported = true;
  private svoPipelineAvailable = false;
  private svoCameraStabilityKey = "";
  /**
   * Owner range of the object under the editor cursor, for the hover rim.
   *
   * A side channel rather than a `draw` argument for the same reason the
   * selected body is one: hover changes on every pointer move, at a rate that
   * has nothing to do with the frame loop, and it is not part of what the frame
   * is *of*. See dryHoverRim in lib/webgpu-svo-dry-scene.ts.
   */
  private hoverHighlight?: { readonly first: number; readonly last: number };
  private svoRenderDiagnosticsKey = "";
  private gpuPendingBatches = 0;
  /** Advances retired by the single presentation fence that follows them. */
  private pendingGPUAdvanceCompletions: PendingGPUAdvanceCompletion[] = [];
  private presentationsInFlight = 0;
  private completedPresentations = 0;
  private simulationRunning = true;
  private gpuFluidGeneration = 0;
  private reportedSimulationPipelineState = "";
  private reportedSparseWorldStatus = "";
  /** True only while both compact t=0 raster sources are attached. */
  private globalFineWaterAttached = false;
  /**
   * Water's shadow term. Owned here rather than by the sparse publication: its
   * source is the solver's dense coarse level set, which reaches presentation
   * as a texture and only means signed distance while the global-fine lane is
   * attached. Rebuilt whenever that field or its geometry changes.
   */
  private svoFluidCoverage?: WebGpuSvoFluidCoverage;
  private svoFluidCoverageKey?: string;
  private pendingInitialRasterPresentation?: PendingInitialRasterPresentation;
  /** A sparse presentation becomes ready only after its first source-matched frame completes. */
  private pendingLiveSvoPresentation?: PendingLiveSvoPresentation;
  private svoPipelineProgress?: { label: string; completed: number; total: number };
  private svoPipelineStartedAt_ms?: number;
  private format?: GPUTextureFormat;
  private presentationContext = "";
  private cpuTraceSampleId = 0;
  private presentationTraceSampleId = 0;
  private presentationTracePending = false;
  private reportedMissingPresentationTimestamps = false;
  private latestPresentationTrace?: PerformanceTrace;
  private latestPresentationStageTrace?: PerformanceTrace;
  /** Fence-partitioned band walls from the last sampling frame (WP3.1). */
  private latestPresentationBandTrace?: PerformanceTrace;
  private latestPresentationManifest?: RenderFrameManifest;
  private presentationBandSamplePending = false;
  private presentationBandFrameCounter = 0;
  /** Polled by the paused viewport; each successful transactional source attach requests one repaint. */
  private pausedPresentationRevision = 0;
  private deviceRecoveryAttempts = 0;
  private lastDeviceRecoveryAt_ms = -Infinity;
  /**
   * Device loss may be a deterministic driver/watchdog fault rather than a
   * transient reset. Recreating the device automatically can immediately
   * submit the same workload again and turn one loss into a machine-wide
   * failure loop. Keep recovery as an explicit diagnostic opt-in.
   */
  private readonly automaticDeviceRecoveryEnabled = typeof location !== "undefined"
    && automaticGPURecoveryEnabled(location.search);
  /** A t=0 rebuild must not overlap the old solver's queue or allocation. */
  private timelineResetPending = false;
  /** CSS viewport supplied by the main thread when this renderer owns an
   * OffscreenCanvas. HTML canvases continue to derive it locally. */
  private workerViewport?: { width: number; height: number; devicePixelRatio: number };

  get presentationRevision(): number { return this.pausedPresentationRevision; }

  constructor(private readonly canvas: HTMLCanvasElement | OffscreenCanvas, private readonly onStatus: (status: GPUStatus) => void, onGPUInfo?: (info: GPUEulerianInfo) => void, onGPURigidLoads?: (loads: GPURigidLoad[]) => void, onGPUAdvanceCompleted?: (time_s: number) => void, onEffectiveRendererStatus?: (status: EffectiveRendererStatus) => void, onGPUPressureJournal?: (journal: PressureJournal | undefined) => void, onGPUStageLens?: (receipt: StageLensReceipt | undefined, layers: readonly StageLensLayerReport[]) => void) { this.gpuInfoCallback = onGPUInfo; this.gpuPressureJournalCallback = onGPUPressureJournal; this.gpuStageLensCallback = onGPUStageLens; this.gpuRigidLoadCallback = onGPURigidLoads; this.gpuAdvanceCompletedCallback = onGPUAdvanceCompleted; this.effectiveRendererStatusCallback = onEffectiveRendererStatus; }

  setViewportSize(width: number, height: number, devicePixelRatio = 1): void {
    this.workerViewport = {
      width: Math.max(1, width),
      height: Math.max(1, height),
      devicePixelRatio: Math.max(0.25, Math.min(2, devicePixelRatio)),
    };
  }

  private publishEffectiveRendererStatus(status: EffectiveRendererStatus) {
    const previous = this.lastEffectiveRendererStatus;
    if (previous?.state === status.state && previous.failureReason === status.failureReason
      && previous.detail === status.detail
      && previous.silhouetteRefinement?.state === status.silhouetteRefinement?.state
      && previous.silhouetteRefinement?.detail === status.silhouetteRefinement?.detail
      && previous.lightingVisibility?.state === status.lightingVisibility?.state
      && previous.lightingVisibility?.fallback === status.lightingVisibility?.fallback
      && previous.lightingVisibility?.detail === status.lightingVisibility?.detail
      && previous.terminalCounts?.voxel === status.terminalCounts?.voxel
      && previous.terminalCounts?.planarBoundary === status.terminalCounts?.planarBoundary) return;
    this.lastEffectiveRendererStatus = status;
    this.effectiveRendererStatusCallback?.(status);
  }

  /**
   * Point the dry scene at a different primary traversal. The mode is baked
   * into the pipeline's shader variants and render-pass shape, so switching it
   * retires the current pipeline rather than reconfiguring it; the viewport
   * shows the compile progress it already shows on first attach. A sticky
   * earlier failure is cleared too, because the mode that failed is not the
   * mode being asked for now.
   *
   * The scale is resolved here rather than in the pipeline factory so there is
   * one seam: the factory reads the settled mode, and a scene whose brick count
   * crosses the ceiling retires through the same path the toggle uses.
   */
  private applyPrimaryTraversalRequest(
    requested: SvoPrimaryTraversalMode,
    scale: SvoPrimaryTraversalScale,
  ): void {
    const resolved = resolveSvoPrimaryTraversal(requested, scale);
    if (resolved === this.requestedPrimaryTraversal) return;
    this.requestedPrimaryTraversal = resolved;
    this.failedOptionalPipelines.delete("svo-dry-scene");
    this.optionalPipelineFailures.delete("svo-dry-scene");
    const retired = this.svoDryScenePipeline;
    if (!retired) return;
    this.svoDryScenePipeline = undefined;
    this.svoPipelineAvailable = false;
    this.svoPipelineProgress = undefined;
    try { retired.destroy(); } catch { /* Device loss may have invalidated it already. */ }
  }

  private ensureOptionalPipeline<T>(
    key: OptionalRendererPipeline,
    current: T | undefined,
    create: (device: GPUDevice) => T,
    initialize: (pipeline: T) => Promise<void>,
    publish: (pipeline: T) => void,
    destroy: (pipeline: T) => void = () => {},
  ): T | undefined {
    const device = this.device;
    if (current || !device || this.disposed || this.deviceLost || this.failedOptionalPipelines.has(key) || this.optionalPipelineTasks.has(key)) return current;
    let candidate: T;
    try {
      candidate = create(device);
    } catch (error) {
      this.failedOptionalPipelines.add(key);
      this.optionalPipelineFailures.set(key, error instanceof Error ? error.message : String(error));
      console.warn(`Optional ${key} pipeline unavailable`, error);
      if (key === "svo-dry-scene") this.failPendingLiveSvoPresentation(error);
      return undefined;
    }
    const task = initialize(candidate).then(() => {
      if (this.disposed || this.deviceLost || this.device !== device) {
        try { destroy(candidate); } catch { /* Device loss may invalidate resources first. */ }
        return;
      }
      publish(candidate);
      this.optionalPipelineFailures.delete(key);
      this.pausedPresentationRevision += 1;
    }).catch((error: unknown) => {
      try { destroy(candidate); } catch { /* Best-effort cleanup after compile failure. */ }
      if (this.device === device && !this.disposed && !this.deviceLost) {
        this.failedOptionalPipelines.add(key);
        this.optionalPipelineFailures.set(key, error instanceof Error ? error.message : String(error));
        console.warn(`Optional ${key} pipeline unavailable`, error);
        if (key === "svo-dry-scene") this.failPendingLiveSvoPresentation(error);
      }
    }).finally(() => {
      if (this.optionalPipelineTasks.get(key) === task) this.optionalPipelineTasks.delete(key);
    });
    this.optionalPipelineTasks.set(key, task);
    return undefined;
  }

  private reportSvoPipelineProgress(label: string, completed: number, total: number) {
    this.svoPipelineStartedAt_ms ??= performance.now();
    this.svoPipelineProgress = { label, completed, total };
    const pending = this.pendingLiveSvoPresentation;
    this.onStatus({
      state: "initializing", label, phase: "presentation", completed, total,
      startedAt_ms: pending?.startedAt_ms ?? this.svoPipelineStartedAt_ms, kind: "startup", retainingPrevious: false,
      resource: svoPresentationResourcePlugin,
    });
  }

  private reportLiveSvoAttachment() {
    const pending = this.pendingLiveSvoPresentation;
    if (!pending || !this.svoDryScenePipeline || !this.svoSourceAvailable || !pending.attach()) return;
    this.onStatus({
      state: "initializing", label: SVO_PRESENTATION_STARTUP_STAGES[8], phase: "presentation",
      completed: 8, total: SVO_PRESENTATION_STARTUP_STAGES.length, startedAt_ms: pending.startedAt_ms, kind: "startup", retainingPrevious: false,
      resource: svoPresentationResourcePlugin,
    });
  }

  /**
   * The sole solver-to-presentation attachment seam. Publishing a sparse
   * source necessarily opens its source-matched fenced generation; callers
   * cannot attach the source while forgetting the completion lifecycle.
   */
  private attachSparsePresentationSource(
    solver: GPUSolverInstance,
    requestGeneration: number,
    startedAt_ms: number,
    source = solver.sparseVoxelSceneSource,
  ): void {
    this.svoDrySceneSource = source;
    this.svoDrySceneData = undefined;
    this.renderSceneKey = "";
    this.renderSceneStamp = 0;
    this.svoDryScenePipeline?.setSource(source);
    this.pendingLiveSvoPresentation = source
      ? new PendingLiveSvoPresentation(
        solver, source, this.gpuFluidGeneration, requestGeneration, startedAt_ms)
      : undefined;
    if (!source) return;
    if (this.failedOptionalPipelines.has("svo-dry-scene")) this.failPendingLiveSvoPresentation();
    else if (this.svoDryScenePipeline) this.reportLiveSvoAttachment();
    else {
      const pipelineProgress = this.svoPipelineProgress ?? {
        label: SVO_PRESENTATION_STARTUP_STAGES[0],
        completed: 0,
        total: SVO_PRESENTATION_STARTUP_STAGES.length,
      };
      this.onStatus({
        state: "initializing", ...pipelineProgress, phase: "presentation", startedAt_ms,
        kind: "startup", retainingPrevious: false, resource: svoPresentationResourcePlugin,
      });
    }
  }

  private failPendingLiveSvoPresentation(error?: unknown) {
    const pending = this.pendingLiveSvoPresentation;
    if (!pending) return;
    this.pendingLiveSvoPresentation = undefined;
    const detail = error instanceof Error && error.message ? `: ${error.message}` : "";
    this.onStatus({ state: "blocked", label: `Sparse garden renderer unavailable${detail}`, resource: svoPresentationResourcePlugin });
  }

  private ensureRequestedOptionalPipelines(
    requested: readonly OptionalRendererPipeline[],
    methodId: string,
  ) {
    const wants = new Set(requested);
    // The two technique overlays are the running method's own passes, resolved
    // by registration key. A method that registers nothing under a key compiles
    // nothing for it: the pass could only ever read a compact debug source that
    // method does not publish, so the absence is the answer rather than a gap.
    const overlays = getMethod(methodId).overlayPipelines;
    if (wants.has("grid-overlay")) this.ensureOptionalPipeline(
      "grid-overlay", this.gridOverlayPipeline,
      (device) => new GridOverlayPipeline(device, this.format!, this.uniformBuffer!, this.bodyBuffer!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => { this.gridOverlayPipeline = pipeline; this.updateRenderSources(); },
      (pipeline) => pipeline.destroy(),
    );
    const techniqueOverlay = overlays?.["technique-overlay"];
    if (wants.has("technique-overlay") && techniqueOverlay) this.ensureOptionalPipeline(
      "technique-overlay", this.techniqueOverlayPipeline,
      (device) => techniqueOverlay(device, this.format!, this.uniformBuffer!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.techniqueOverlayPipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
        pipeline.setOwnerRows(this.gpuFluid?.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture!);
      },
    );
    if (wants.has("svo-stage-overlay")) this.ensureOptionalPipeline(
      "svo-stage-overlay", this.svoStageOverlay,
      (device) => new SparseVoxelRenderStageOverlay(device, this.format!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => { this.svoStageOverlay = pipeline; },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("fluid-cell-trace")) this.ensureOptionalPipeline(
      "fluid-cell-trace", this.fluidCellTracePipeline,
      (device) => new WebGPUFluidCellTrace(device, this.uniformBuffer!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.fluidCellTracePipeline = pipeline;
        pipeline.setSource(
          this.gpuFluid?.octreeTechniqueDebugSource,
          this.gpuFluid?.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture!,
        );
      },
      (pipeline) => pipeline.destroy(),
    );
    const techniqueAuditOverlay = overlays?.["technique-audit-overlay"];
    if (wants.has("technique-audit-overlay") && techniqueAuditOverlay) this.ensureOptionalPipeline(
      "technique-audit-overlay", this.techniqueAuditOverlayPipeline,
      (device) => techniqueAuditOverlay(device, this.format!, this.uniformBuffer!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.techniqueAuditOverlayPipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
        pipeline.setOwnerRows(this.gpuFluid?.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture!);
      },
    );
    if (wants.has("svo-dry-scene")) this.ensureOptionalPipeline(
      "svo-dry-scene", this.svoDryScenePipeline,
      // Relight already pays a full-resolution material/BRDF pass. Primary
      // traversal is isolated and encoded on every frame so its cost remains
      // visible under paused and stationary cameras.
      // Canonical-parametric removes the wide/canonical hybrid cursor and was
      // revalidated against split full-res relighting: 12.55 ms versus
      // 22.10-23.60 ms hybrid at 660x662 with identical output hashes.
      // Raster-primary replaces the full-screen traversal megakernel with a
      // rasterized brick-proxy pass: 4.89 ms of GPU against the megakernel's
      // 26.95 ms for primary visibility at 1500x1500 garden, whole frame
      // 49.62 -> 29.01 ms. It needs four depth-tested colour planes; a request
      // that the device cannot honor fails through the optional-pipeline
      // diagnostic instead of silently selecting another traversal.
      //
      // Which of the two this frame gets is `resolveSvoPrimaryTraversal`'s
      // answer, already settled into `requestedPrimaryTraversal`: the proxy
      // raster is an area law and inverts against the megakernel once the scene
      // emits more proxies than the target has pixels.
      (device) => createProductionSparseVoxelDrySceneRenderer(
        device, this.uniformBuffer!, this.bodyBuffer!, this.requestedPrimaryTraversal,
      ),
      (pipeline) => pipeline.initialize((label, completed, total) => this.reportSvoPipelineProgress(label, completed, total)),
      (pipeline) => {
        this.svoDryScenePipeline = pipeline;
        this.svoPipelineAvailable = true;
        pipeline.setSource(this.svoDrySceneSource);
        if (this.svoDrySceneData) pipeline.publishScene(this.svoDrySceneData);
        if (this.presentationTexture) pipeline.ensureSize(this.presentationTexture.width, this.presentationTexture.height);
        this.reportLiveSvoAttachment();
      },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("decoration-overlay")) this.ensureOptionalPipeline(
      "decoration-overlay", this.decorationOverlayPipeline,
      (device) => new DecorationOverlay(device, this.format!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => { this.decorationOverlayPipeline = pipeline; },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("tracer-overlay")) this.ensureOptionalPipeline(
      "tracer-overlay", this.tracerOverlayPipeline,
      (device) => new TracerOverlay(device, this.format!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.tracerOverlayPipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.sparseWorldUI?.overlays.tracers
          ?? this.gpuFluid?.tracerSource);
      },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("face-velocity-overlay")) this.ensureOptionalPipeline(
      "face-velocity-overlay", this.faceVelocityOverlayPipeline,
      (device) => new FaceVelocityOverlay(device, this.format!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.faceVelocityOverlayPipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.sparseWorldUI?.overlays.faceVelocity
          ?? this.gpuFluid?.faceVelocitySource);
      },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("pressure-journal-overlay")) this.ensureOptionalPipeline(
      "pressure-journal-overlay", this.pressureJournalOverlayPipeline,
      (device) => new PressureJournalOverlay(device, this.format!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.pressureJournalOverlayPipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.sparseWorldUI?.overlays.pressureFilm
          ?? this.gpuFluid?.pressureJournalSource);
      },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("stage-lens")) this.ensureOptionalPipeline(
      "stage-lens", this.stageLensOverlayPipeline,
      (device) => new StageLensOverlay(device, this.format!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.stageLensOverlayPipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.sparseWorldUI?.overlays.stageLenses
          ?? this.gpuFluid?.stageLensSource);
      },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("secondary-particles")) this.ensureOptionalPipeline(
      "secondary-particles", this.secondaryParticlePipeline,
      (device) => new SecondaryParticleRenderPipeline(device, this.uniformBuffer!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.secondaryParticlePipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.secondaryParticles);
        this.waterPipeline?.setSecondaryParticles(pipeline);
      },
    );
  }

  /**
   * The poses the last published frame was drawn with, in roster order.
   *
   * The host roster the simulation hands to `draw` is a command channel: the
   * solver owns rigid motion once a run starts and never writes back, so a body
   * that has fallen, floated or been shoved by the water is *drawn* from GPU
   * state while the host copy still says where it was last told to be. Anything
   * the user points at — the hover chip, the selection's gizmo, the throw that a
   * press opens — has to agree with the image, so it reads this instead.
   */
  get rigidBodyPoses(): readonly DrawnRigidBodyPose[] { return this.latestRigidBodyPoses; }
  /** Bumped once per published readback, so consumers can skip unchanged frames. */
  get rigidBodyPoseRevision(): number { return this.latestRigidBodyPoseRevision; }

  /** Queue this frame's pose copy into a free staging slot, if there is one. */
  private encodeRigidBodyPoseReadback(encoder: GPUCommandEncoder, source: GPUBuffer) {
    if (this.rigidPoseStaging.length === 0 && this.device) {
      this.rigidPoseStaging = [0, 1].map((index) => ({
        buffer: this.device!.createBuffer({
          label: `Rigid pose readback ${index + 1}/2`,
          size: GPU_RIGID_RENDER_BYTES,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        pending: false,
      }));
    }
    const slot = this.rigidPoseStaging.find((candidate) => !candidate.pending);
    if (!slot) return undefined;
    slot.pending = true;
    encoder.copyBufferToBuffer(source, 0, slot.buffer, 0, GPU_RIGID_RENDER_BYTES);
    return slot;
  }

  /**
   * Publish the copy once the queue has run, named by the ids of the roster the
   * frame was drawn with.
   *
   * Names, not slots: the readback lands a frame or two later, and by then a
   * body may have been added or removed. Index `2` would then be a different
   * object than the one whose pose was copied, which is a gizmo drawn around the
   * wrong crate for as long as it takes the next readback to land.
   */
  private publishRigidBodyPoses(slot: { buffer: GPUBuffer; pending: boolean }, ids: readonly string[]) {
    void slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const poses = decodeGPURigidBodyPoses(new Float32Array(slot.buffer.getMappedRange()), ids.length);
      // A stale device's frame must not overwrite the live one's poses.
      if (!this.disposed && !this.deviceLost) {
        this.latestRigidBodyPoses = poses.map((pose, index) => ({ id: ids[index], ...pose }));
        this.latestRigidBodyPoseRevision += 1;
      }
    }).catch(() => { /* Device loss or a destroyed buffer: the host keeps the last poses. */ })
      .finally(() => {
        try { if (slot.buffer.mapState === "mapped") slot.buffer.unmap(); } catch { /* Destroyed. */ }
        slot.pending = false;
      });
  }

  /** Resolve a click against live GPU poses without restoring a CPU pose mirror. */
  async pickRigidBody(
    origin: RigidBodyState["position_m"],
    direction: RigidBodyState["position_m"],
    screen?: { normalizedX: number; normalizedY: number },
  ) {
    if (this.svoPickingAvailable && screen && this.svoDryScenePipeline) {
      const bodies = this.lastSvoPickingBodies, pipeline = this.svoDryScenePipeline;
      // The surface point comes from the drawn frame, so the centre it is
      // measured against has to come from the same place. `bodies` is the host
      // roster the frame was submitted with, and the solver has been moving
      // those bodies on its own ever since — a settled crate is metres from
      // where that roster says. Both readbacks are issued together so asking
      // costs no extra latency on the press.
      const [picked, poses] = await Promise.all([
        pipeline.pickGBuffer(
          screen.normalizedX, screen.normalizedY,
          [origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z], bodies.length,
        ),
        this.gpuFluid?.readRigidBodyPoses?.(),
      ]);
      if (!this.svoPickingAvailable || this.svoDryScenePipeline !== pipeline || picked.status !== "hit") return undefined;
      const body = bodies[picked.bodyIndex];
      if (!body) return undefined;
      const live = poses?.[picked.bodyIndex];
      return {
        bodyIndex: picked.bodyIndex,
        distance_m: picked.depth_m,
        position_m: live?.position_m ?? body.position_m,
        orientation: live?.orientation ?? body.orientation,
        surfacePosition_m: { x: picked.position_m[0], y: picked.position_m[1], z: picked.position_m[2] },
        materialId: picked.materialId,
        localTopologyGeneration: picked.localTopologyGeneration,
      };
    }
    const fluid=this.gpuFluid,generation=this.gpuFluidGeneration;
    if(!fluid?.pickRigidBody||this.disposed||this.deviceLost)return undefined;
    const picked=await fluid.pickRigidBody(origin,direction);
    return this.gpuFluid===fluid&&this.gpuFluidGeneration===generation?picked:undefined;
  }

  /* ----------------------------------------------------------------------- */
  /* Live pixel trace                                                        */
  /* ----------------------------------------------------------------------- */

  /**
   * Latest decoded trace and a revision counter. The UI polls the revision from
   * its animation frame rather than being called back, so a trace arriving while
   * React is mid-render can never tear a frame.
   */
  get latestPixelTrace(): SvoPixelTrace | undefined { return this.latestPixelTraceValue; }
  get pixelTraceRevision(): number { return this.pixelTraceRevisionValue; }
  get pixelTraceAvailable(): boolean { return Boolean(this.svoDryScenePipeline?.pixelTraceReady); }

  /**
   * Why the diagnostic is or is not showing anything.
   *
   * Without this the HUD cannot tell "you have not moved the pointer yet" from
   * "the live sparse frame failed closed, so there is no traversal to trace" —
   * and both look like a broken pointer.
   */
  get pixelTraceStatus(): PixelTraceStatus {
    if (this.svoDryScenePipeline?.pixelTraceUnsupported) return "unsupported";
    // svoPickingAvailable is exactly "the sparse dry scene encoded this frame",
    // which is the same condition the probe needs to have anything to walk.
    if (!this.svoPickingAvailable) return "path-inactive";
    if (this.svoDryScenePipeline?.pixelTraceCompiling) return "compiling";
    if (!this.latestPixelTraceValue) return "waiting";
    return "live";
  }

  /**
   * True when the newest decoded trace answers the pixel currently requested.
   *
   * Click-to-pin waits for this instead of freezing whatever is already drawn:
   * the readback runs a frame behind its request, so the trace on screen at the
   * instant of a click generally belongs to a neighbouring pixel.
   */
  get pixelTraceAnswersRequest(): boolean {
    const trace = this.latestPixelTraceValue;
    const requested = this.svoDryScenePipeline?.pixelTraceRequestedPixel;
    if (!trace || !requested) return false;
    return trace.pixel[0] === requested[0] && trace.pixel[1] === requested[1];
  }

  /**
   * True when the held trace describes a scene the renderer has moved on from —
   * a republished topology, another light, a different shadow or cone budget.
   *
   * A live trace clears this by itself on the next frame. A pinned one cannot:
   * it has stopped probing, so its counters would go on describing a frame that
   * no longer exists until someone asks for a refresh.
   */
  get pixelTraceStale(): boolean {
    if (!this.latestPixelTraceValue) return false;
    return this.pixelTraceSceneRevisionOfTrace !== this.pixelTraceSceneRevisionValue;
  }

  get latestFluidCellTrace(): FluidCellTrace | undefined { return this.latestFluidCellTraceValue; }
  get fluidCellTraceRevision(): number { return this.fluidCellTraceRevisionValue; }
  /** Ready once the gather pipeline exists and has a published topology to read. */
  get fluidCellTraceReady(): boolean { return this.fluidCellTracePipeline?.ready === true; }

  /**
   * The band widths and redistance ladder the HUD reads a cell against.
   *
   * Frame-wide rather than per-cell, so it stays out of the trace ABI: copying
   * these into every record would invite the two to disagree, and the whole
   * point of reading the planner's own output is that it cannot. Undefined on a
   * scene with no fine band, which is what makes the HUD's fine-band panels
   * conditional rather than empty.
   */
  get fluidCellTraceFineBand(): FineBandCellContext | undefined {
    const bands = this.gpuFluid?.octreeTechniqueDebugSource?.fineBandLifecycle?.bands;
    if (!bands) return undefined;
    return {
      widths: {
        pressureBandCells: bands.pressureBandCells,
        surfaceBandCells: bands.surfaceBandCells,
        transportBandFineCells: bands.transportBandFineCells,
        redistanceBandFineCells: bands.redistanceBandFineCells,
      },
      ladderStrides: bands.ladderStrides,
    };
  }

  private pumpFluidCellTraceReadback(): void {
    const pipeline = this.fluidCellTracePipeline;
    if (!pipeline || this.fluidCellTraceReadInFlight) return;
    this.fluidCellTraceReadInFlight = true;
    void pipeline.read().then((trace) => {
      if (this.disposed || this.deviceLost || this.fluidCellTracePipeline !== pipeline || !trace) return;
      this.latestFluidCellTraceValue = trace;
      this.fluidCellTraceRevisionValue += 1;
    }).catch(() => { /* A superseded or unmapped readback is not a frame error. */ })
      .finally(() => { this.fluidCellTraceReadInFlight = false; });
  }

  private pumpPixelTraceReadback(): void {
    const pipeline = this.svoDryScenePipeline;
    if (!pipeline || this.pixelTraceReadInFlight) return;
    this.pixelTraceReadInFlight = true;
    const encodedSceneRevision = this.pixelTraceEncodedSceneRevision;
    void pipeline.readPixelTrace().then((trace) => {
      if (this.disposed || this.deviceLost || this.svoDryScenePipeline !== pipeline || !trace) return;
      this.latestPixelTraceValue = trace;
      // The trace answers the scene it was encoded against, not the one that
      // happens to be current when its readback resolves.
      this.pixelTraceSceneRevisionOfTrace = encodedSceneRevision;
      this.pixelTraceRevisionValue += 1;
    }).catch(() => { /* A superseded or unmapped readback is not a frame error. */ })
      .finally(() => { this.pixelTraceReadInFlight = false; });
  }

  /**
   * Draw every decoration this frame's selections produce, in one call.
   *
   * The renderer does not know what any of them are. It supplies the lattice,
   * the selections and which ids are on; `assembleDecorations` asks each pass's
   * declaration whether it can draw any of them and merges what comes back. A
   * frame holding both a traced ray and a picked cell therefore costs the same
   * one upload and one draw as either alone.
   *
   * The assembled key is the drawn content, not a revision counter, so orbiting
   * a frozen selection or holding the pointer still re-uploads nothing — which
   * matters because the cell gather re-runs every frame regardless.
   */
  private encodeDecorationOverlay(
    encoder: GPUCommandEncoder,
    basis: ReturnType<typeof cameraBasis>,
    tanHalfFov: number,
    scene: SceneDescription,
    pixelTrace: PixelTraceConfig | undefined,
    fluidCellTrace: FluidCellTraceConfig | undefined,
  ): boolean {
    const overlay = this.decorationOverlayPipeline;
    if (!overlay?.ready || !this.presentationTexture) return false;
    const vesselOutline = buildVesselOutlineGeometry(scene);
    const cellTrace = fluidCellTrace ? this.latestFluidCellTraceValue : undefined;
    const subjects: unknown[] = [];
    if (pixelTrace && this.latestPixelTraceValue) subjects.push(this.latestPixelTraceValue);
    if (cellTrace) {
      // The picked cell, plus what the derived decorators need to grow their own
      // views of it. Each declaration narrows this itself, so adding a field for
      // one of them cannot disturb the others.
      subjects.push(fluidCellTrace?.solvePolicy
        ? { ...cellTrace, solvePolicy: fluidCellTrace.solvePolicy }
        : cellTrace);
    }
    if (subjects.length === 0 && !vesselOutline) {
      if (this.decorationGeometryKey !== "") { overlay.clear(); this.decorationGeometryKey = ""; }
      return false;
    }
    const dimensions = cellTrace?.dimensions
      ?? (this.gpuFluid?.octreeTechniqueDebugSource?.pressureRows.dimensions);
    const enabledIds = new Set<string>([
      ...(pixelTrace?.layers ?? []).map((layer) => `svo-traversal/${layer}`),
      ...(fluidCellTrace?.layers ?? []),
    ]);
    const assembled = assembleDecorations({
      definitions: VISUALIZATION_CATALOG,
      subjects,
      // Both producers place work through this lattice; the ray probe's records
      // are already world metres and bypass it by appending directly.
      space: containerDecorationSpace(
        (dimensions ?? [1, 1, 1]) as readonly [number, number, number],
        [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
      ),
      emphasis: fluidCellTrace && !fluidCellTrace.pinned ? "hover" : "selected",
      widthScale: pixelTrace?.widthScale ?? fluidCellTrace?.widthScale,
      enabled: (definition) => enabledIds.has(definition.id),
    });
    const builder = new DecorationBuilder(containerDecorationSpace([1, 1, 1],
      [scene.container.width_m, scene.container.height_m, scene.container.depth_m]));
    if (vesselOutline) {
      builder.append(vesselOutline.geometry.segments,
        vesselOutline.geometry.segmentCount);
    }
    builder.append(assembled.geometry.segments, assembled.geometry.segmentCount);
    const geometry = builder.finish();
    const geometryKey = `${vesselOutline?.key ?? "no-vessel"}|${assembled.key}`;
    if (geometryKey !== this.decorationGeometryKey) {
      overlay.setGeometry(geometry);
      this.decorationGeometryKey = geometryKey;
    }
    const width = this.presentationTexture.width, height = this.presentationTexture.height;
    overlay.encode(encoder, this.cachedTextureView(this.presentationTexture), this.pixelTraceSceneDepthView(), {
      camera: {
        position_m: [basis.position.x, basis.position.y, basis.position.z],
        forward: [basis.forward.x, basis.forward.y, basis.forward.z],
        right: [basis.right.x, basis.right.y, basis.right.z],
        up: [basis.up.x, basis.up.y, basis.up.z],
        tanHalfFov,
        aspect: viewportAspect(width, height),
      },
      viewportWidth: width,
      viewportHeight: height,
      // Only the ray probe's work is a sequence, so only it has a sweep to
      // reveal; everything else is emitted at order zero and always drawn.
      reveal: pixelTrace?.reveal ?? 1,
      occludedAlpha: pixelTrace?.occludedAlpha ?? (vesselOutline ? 0.18 : undefined),
      depthNear_m: SVO_DRY_SCENE_REVERSED_Z_NEAR_M,
    });
    return true;
  }

  /** Stable views: a fresh view object every frame would rebuild bind groups. */
  private readonly cachedTextureViews = new WeakMap<GPUTexture, GPUTextureView>();

  private cachedTextureView(texture: GPUTexture): GPUTextureView {
    const existing = this.cachedTextureViews.get(texture);
    if (existing) return existing;
    const view = texture.createView();
    this.cachedTextureViews.set(texture, view);
    return view;
  }

  /**
   * Scene depth for the overlay's occlusion ghosting. Absent while the sparse
   * G-buffer has not been published, which simply draws the trace unghosted.
   */
  private pixelTraceSceneDepthView(): GPUTextureView | undefined {
    const depth = this.svoDryScenePipeline?.gBufferTextures?.hardwareDepth;
    return depth ? this.cachedTextureView(depth) : undefined;
  }

  initialize(): Promise<void> {
    if (this.disposed) return Promise.resolve();
    if (this.initializationPromise) return this.initializationPromise;
    const task = this.initializeInternal();
    const pending = task.finally(() => {
      if (this.initializationPromise === pending) this.initializationPromise = undefined;
    });
    this.initializationPromise = pending;
    return this.initializationPromise;
  }

  private async initializeInternal(): Promise<void> {
    const startedAt_ms=performance.now();
    const progress=(label:string,completed:number,total=4,phase="renderer")=>this.onStatus({state:"initializing",label,phase,completed,total,startedAt_ms,resource:webGPUPlatformResourcePlugin});
    // UI-only browser automation must be safe even if a caller accidentally
    // invokes initialize(): return before navigator.gpu or solver creation.
    if (typeof location !== "undefined" && new URLSearchParams(location.search).get("gpu") === "off") {
      this.onStatus({ state: "unavailable", label: "WebGPU disabled by gpu=off (UI-only mode)", resource: webGPUPlatformResourcePlugin });
      return;
    }
    progress("Requesting WebGPU adapter",0);
    if (!("gpu" in navigator)) {
      this.onStatus({ state: "unavailable", label: "WebGPU is not available in this browser", resource: webGPUPlatformResourcePlugin });
      return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (this.disposed) return;
    if (!adapter) {
      this.onStatus({ state: "unavailable", label: "No compatible GPU adapter was found", resource: webGPUPlatformResourcePlugin });
      return;
    }
    progress("Requesting GPU device",1);
    const requiredFeatures = fluidExecutionDeviceFeatures(adapter.features);
    const requiredLimits = requiredFluidDeviceLimits(adapter.limits);
    const rawDevice = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    if (this.disposed) { rawDevice.destroy(); return; }
    const context = this.canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) {
      rawDevice.destroy();
      this.onStatus({ state: "unavailable", label: "WebGPU canvas context could not be created", resource: webGPUPlatformResourcePlugin });
      return;
    }
    const device = managedGPUDevice(rawDevice);
    this.device = device;
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    device.addEventListener("uncapturederror", (event) => console.error(`WebGPU validation: ${event.error.message}`));
    void device.lost.then((info) => {
      if (this.disposed || this.device !== device || this.deviceLost) return;
      invalidateGPUCompilationManager(device, info.message || info.reason);
      this.deviceLost = true;
      const fluid = this.gpuFluid;
      const sidecar = this.svoSceneSidecar;
      this.gpuFluid = undefined;
      this.svoSceneSidecar = undefined;
      this.pendingInitialRasterPresentation = undefined;
      this.pendingLiveSvoPresentation = undefined;
      this.gpuFluidKey = "";
      this.attachedStructuralKey = "";
      this.gpuFluidPendingKey = "";
      this.gpuFluidInitializationAbort?.abort();
      this.gpuFluidInitializationAbort = undefined;
      this.gpuFluidPending = undefined;
      this.resetGPUQueueTracking();
      this.gpuFluidGeneration += 1;
      // A solver initialization pending on the lost device must never attach
      // after recovery: its resources belong to the dead device and any bind
      // group mixing them with the replacement device fails validation.
      this.gpuFluidRequestGeneration += 1;
      try { fluid?.destroy(); } catch { /* Resources may already be invalid after device loss. */ }
      try { sidecar?.destroy(); } catch { /* Resources may already be invalid after device loss. */ }
      // Breadcrumbs for hang diagnosis: the last known solver state narrows a
      // watchdog reset down to a stage without needing a reproduction.
      if (fluid) console.error("GPU device lost mid-simulation", { reason: info.reason, message: info.message, submittedTime_s: fluid.info.submittedTime_s, completedTime_s: fluid.info.completedTime_s, pendingBatches: this.gpuPendingBatches, encodedSteps: fluid.info.encodedSteps, physicsTrace: fluid.info.physicsTrace });
      this.onStatus({ state: "lost", label: `GPU device lost: ${info.message || info.reason}`, resource: webGPUPlatformResourcePlugin });
      this.scheduleDeviceRecovery(info.reason);
    }).catch((error: unknown) => {
      if (!this.disposed) console.error("Unable to observe WebGPU device loss", error);
    });
    // GPUCanvasContext performs an implementation-level GPUDevice brand check;
    // the managed capability is intentionally a Proxy and cannot cross this
    // native boundary. Only the raw device is exposed here. Resource owners
    // continue to receive the compilation-gated device above.
    context.configure({ device: rawDevice, format: this.format, alphaMode: "opaque" });

    progress("Compiling presentation upscale",2);
    const upscaleModule=device.createShaderModule({label:"Presentation upscale shader",code:upscaleShader});
    const upscalePipeline=await device.createRenderPipelineAsync({label:"Presentation upscale",layout:"auto",vertex:{module:upscaleModule,entryPoint:"vertexMain"},fragment:{module:upscaleModule,entryPoint:"fragmentMain",targets:[{format:this.format}]},primitive:{topology:"triangle-list"}});
    if (this.disposed || this.device !== device || this.deviceLost) return;
    this.upscalePipeline=upscalePipeline;
    this.upscaleSampler=device.createSampler({magFilter:"linear",minFilter:"linear"});
    this.uniformBuffer = device.createBuffer({ label: "Fluid Lab view uniforms", size: 416, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bodyBuffer = device.createBuffer({ label: "Fluid Lab rigid bodies", size: 12 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fluidTexture = device.createTexture({ size: [1, 1, 1], dimension: "3d", format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.columnBaseTexture = device.createTexture({ label: "Uniform-grid tall-cell fallback", size: [1, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.gridCellTexture = device.createTexture({ label: "Uniform-grid adaptive-cell fallback", size: [1, 1, 1], dimension: "3d", format: "rg32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.velocityFallbackTexture = device.createTexture({ label: "Overlay velocity fallback", size: [1, 1, 1], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.pressureSamplesFallbackTexture = device.createTexture({ label: "Overlay pressure-sample fallback", size: [1, 1, 1], dimension: "3d", format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.scalarFallbackTexture = device.createTexture({ label: "Overlay scalar fallback", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const waterPipeline = new RasterWaterPipeline(device, this.format, this.uniformBuffer, this.bodyBuffer);
    try {
      progress("Compiling raster water pipelines",3);
      await waterPipeline.initialize((label,completed,total)=>progress(label,completed,total,"water-renderer"));
    } catch (error) {
      waterPipeline.destroy();
      throw error;
    }
    if (this.disposed || this.device !== device || this.deviceLost) { waterPipeline.destroy(); return; }
    this.waterPipeline = waterPipeline;
    this.updateRenderSources();

    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    this.adapterName = info ? [info.vendor, info.architecture].filter(Boolean).join(" · ") || "WebGPU adapter" : "WebGPU adapter";
    progress("Renderer ready; preparing solver",4);
    this.onStatus({ state: "ready", label: "WebGPU renderer ready", adapter: this.adapterName, resource: webGPUPlatformResourcePlugin });
  }

  /**
   * A lost device leaves the app permanently dead without intervention: every
   * frame-loop entry point guards on deviceLost, so a transient TDR would
   * otherwise present as a hard crash until reload. The `gpuRecovery=1`
   * diagnostic may recover by re-running initialize() on a fresh device; the
   * solver then rebuilds from the scene (simulation state does not survive).
   * Normal UI sessions stop after one loss so a deterministic driver fault is
   * never resubmitted automatically. Opt-in attempts remain bounded.
   */
  private scheduleDeviceRecovery(reason: string) {
    if (this.disposed || reason === "destroyed" || !this.automaticDeviceRecoveryEnabled) return;
    if (performance.now() - this.lastDeviceRecoveryAt_ms > 60_000) this.deviceRecoveryAttempts = 0;
    if (this.deviceRecoveryAttempts >= 3) return;
    this.deviceRecoveryAttempts += 1;
    this.lastDeviceRecoveryAt_ms = performance.now();
    setTimeout(() => { void this.recoverDevice(); }, 500 * this.deviceRecoveryAttempts);
  }

  private async recoverDevice() {
    if (this.disposed || !this.deviceLost) return;
    // Resources on a lost device are already invalid and need no destroy;
    // drop every device-scoped reference so the frame loop's !this.device
    // guards hold until initialize() completes on the replacement device.
    this.device = undefined; this.context = undefined;
    this.upscalePipeline = undefined; this.upscaleSampler = undefined; this.upscaleBindGroup = undefined;
    this.waterPipeline = undefined; this.gridOverlayPipeline = undefined; this.techniqueOverlayPipeline = undefined; this.techniqueAuditOverlayPipeline = undefined; this.svoDryScenePipeline = undefined; this.secondaryParticlePipeline = undefined; this.tracerOverlayPipeline = undefined; this.faceVelocityOverlayPipeline = undefined; this.pressureJournalOverlayPipeline = undefined; this.stageLensOverlayPipeline = undefined; this.svoStageOverlay = undefined;
    this.optionalPipelineTasks.clear(); this.failedOptionalPipelines.clear(); this.optionalPipelineFailures.clear(); this.svoDrySceneSource = undefined; this.svoSceneSidecar = undefined; this.svoDrySceneData = undefined; this.liveSceneAnimation = undefined; this.liveSceneAnimationFailure = undefined; this.renderSceneKey = ""; this.renderSceneStamp = 0; this.svoPipelineProgress = undefined; this.svoPipelineStartedAt_ms = undefined; this.pendingLiveSvoPresentation = undefined;
    this.svoPipelineAvailable = false; this.svoSourceAvailable = false; this.svoPublicationFailure = undefined; this.svoGlassSupported = true; this.svoMaterialsSupported = true; this.svoLightingSupported = true;
    this.uniformBuffer = undefined; this.bodyBuffer = undefined;
    // The staging buffers belong to the device that is going away; the poses
    // they carried stay, because the scene they describe has not changed.
    this.rigidPoseStaging = [];
    this.presentationTexture = undefined; this.presentationTextureKey = "";
    this.fluidTexture = undefined; this.attachedSurfaceTexture = undefined;
    this.columnBaseTexture = undefined; this.gridCellTexture = undefined;
    this.velocityFallbackTexture = undefined; this.pressureSamplesFallbackTexture = undefined; this.scalarFallbackTexture = undefined;
    this.presentationTracePending = false; this.latestPresentationTrace = undefined; this.latestPresentationStageTrace = undefined;
    this.presentationBandSamplePending = false; this.latestPresentationBandTrace = undefined;
    this.retiredGPUFluids.clear();
    this.deviceLost = false;
    try {
      await this.initialize();
    } catch (error) {
      this.onStatus({ state: "unavailable", label: error instanceof Error ? `GPU recovery failed: ${error.message}` : "GPU recovery failed", resource: webGPUPlatformResourcePlugin });
    }
  }


  /**
   * Build or reuse the fluid coverage volume for this frame.
   *
   * Gated on the global-fine lane, because that is the only configuration whose
   * surface is a signed distance; the tall-cell and quadtree lanes pack
   * occupancy into the same texture, and resampling that as a distance would put
   * a shadow wherever the field happened to be positive. Outside the lane the
   * volume is released and water casts no shadow, which is the behaviour every
   * lane had before.
   *
   * Inside the lane the source is whichever the solver actually publishes.
   * Sparse CM12 is buffer-native: it allocates 1x1x1 stand-ins for every texture
   * field on `GPUSolverInstance` and keeps the real surface in its page set, so
   * the dense arm is admissible only when the texture genuinely covers the
   * lattice it claims to. That check is the point — a fill over the stand-in
   * reads out of bounds and quantizes to half coverage across the entire domain,
   * which is not "no shadow", it is a uniform fog nobody asked for.
   */
  private ensureFluidCoverage(
    solver: GPUSolverInstance | undefined,
    scene: SceneDescription,
  ): WebGpuSvoFluidCoverage | undefined {
    const device = this.device;
    const sparsePresentation = this.sparseWorldPresentation(solver);
    const brickSource = device && solver && this.globalFineWaterAttached
      ? sparsePresentation?.fineLevelSet ?? solver.globalFineLevelSetSource : undefined;
    // The factory validates, so a solver mid-rebuild can throw here. A shadow is
    // never worth taking the frame down with it: an unusable publication is the
    // same answer as no publication.
    let publication: GlobalFineLevelSetConsumerSource | undefined;
    try {
      publication = brickSource ? createGlobalFineLevelSetConsumerSource(brickSource) : undefined;
    } catch {
      publication = undefined;
    }
    const info = solver?.info;
    if (!publication || !info || !device) {
      this.svoFluidCoverage?.destroy();
      this.svoFluidCoverage = undefined;
      this.svoFluidCoverageKey = undefined;
      return undefined;
    }
    // surfaceFieldTexture is the contoured level set when the solver keeps one
    // apart from volumeTexture; it is the field globalCoarsePhi already reads.
    const dense = solver?.surfaceFieldTexture ?? solver?.volumeTexture;
    const spansLattice = Boolean(dense) && dense!.width >= info.nx
      && dense!.height >= info.ny && dense!.depthOrArrayLayers >= info.nz;
    const container = scene.container;
    const fluidDomain = solver.fluidDomain;
    // Both arms span the same physical world box. Usually that is the authored
    // lattice, centred in x/z with y from its origin; sparse flow may publish a
    // larger fluid world so newly created frontier pages render in the same frame.
    // Only the lattice differs: the dense field is the solver grid, the compact
    // publication is the fine one.
    //
    // Deliberately NOT `publication.domainOrigin` / `publication.fineCellWidth`.
    // Those are the publication's *own* frame, and Sparse CM12 publishes it as
    // index space — `domainOrigin: [0,0,0]`, `fineCellWidth: 1`
    // (`webgpu-sparse-cm12-resident.ts`). `createGlobalFineLevelSetConsumerSource`
    // even refuses a non-zero origin outright. Taking them for metres puts the
    // volume at the world origin with one-metre texels: a box in the wrong
    // place, at the wrong scale, that still samples and still reports valid.
    const lattice: SvoFluidCoverageTriple = spansLattice
      ? [info.nx, info.ny, info.nz] : publication.sampleDimensions;
    const field: WebGpuSvoFluidCoverageOptions = {
      fieldDimensions: lattice,
      worldOrigin_m: fluidDomain?.origin_m
        ?? [-container.width_m / 2, 0, -container.depth_m / 2],
      cellSize_m: fluidDomain?.cellSize_m
        ?? [container.width_m / lattice[0], container.height_m / lattice[1], container.depth_m / lattice[2]],
    };
    // A compact publication's lattice is the fine one, so the dense arm's "one
    // texel per two cells" would ask for gigabytes on a large scene. Coarsen
    // until the volume fits, and decline rather than build one that does not.
    const cellsPerTexel = planSvoFluidCoverageRatio(
      field.fieldDimensions, spansLattice ? undefined : publication.brickResolution,
    );
    if (cellsPerTexel === undefined) {
      this.svoFluidCoverage?.destroy();
      this.svoFluidCoverage = undefined;
      this.svoFluidCoverageKey = undefined;
      return undefined;
    }
    const options: WebGpuSvoFluidCoverageOptions = { ...field, cellsPerTexel };
    const key = `${spansLattice ? "dense" : "compact"}:${options.fieldDimensions.join(",")}`
      + `:${options.worldOrigin_m.join(",")}:${options.cellSize_m.join(",")}:${cellsPerTexel}`;
    // The page set is republished every step, so a live owner still has to take
    // this frame's generation; only a shape or buffer change forces a rebuild.
    if (this.svoFluidCoverage && this.svoFluidCoverageKey === key
      && (spansLattice || this.svoFluidCoverage.adoptPublication({ ...publication, kind: "compact" }))) {
      return this.svoFluidCoverage;
    }
    this.svoFluidCoverage?.destroy();
    this.svoFluidCoverageKey = key;
    try {
      const source: WebGpuSvoFluidCoverageSource = spansLattice
        ? { kind: "dense", coarsePhi: dense!.createView({ dimension: "3d" }) }
        : { ...publication, kind: "compact" };
      this.svoFluidCoverage = new WebGpuSvoFluidCoverage(device, options, source);
      void this.svoFluidCoverage.initializePipelines();
    } catch {
      this.svoFluidCoverage = undefined;
      this.svoFluidCoverageKey = undefined;
    }
    return this.svoFluidCoverage;
  }

  private updateRenderSources(texture = this.fluidTexture, columnSource?: GPUTexture, gridCells = this.gridCellTexture, velocity = this.velocityFallbackTexture, pressureSamples = this.pressureSamplesFallbackTexture, divergence = this.scalarFallbackTexture, pressure = this.scalarFallbackTexture, density = this.scalarFallbackTexture) {
    const columnBases = columnSource ?? this.columnBaseTexture;
    if (!this.device || this.disposed || this.deviceLost || !texture || !columnBases || !gridCells || !velocity || !pressureSamples || !divergence || !pressure || !density) return;
    this.attachedSurfaceTexture = texture;
    this.waterPipeline?.setVolume(texture, columnBases);
    this.waterPipeline?.setFluidDomain(this.gpuFluid?.fluidDomain);
    const sparsePresentation = this.sparseWorldPresentation(this.gpuFluid);
    const globalFineLevelSet = sparsePresentation?.fineLevelSet
      ?? this.gpuFluid?.globalFineLevelSetSource;
    this.waterPipeline?.setGlobalFineLevelSet(globalFineLevelSet
      ? createGlobalFineLevelSetConsumerSource(globalFineLevelSet)
      : undefined);
    this.waterPipeline?.setCoarseLevelSet(this.gpuFluid?.coarseLevelSetSource);
    this.gridOverlayPipeline?.setVolume(texture, columnBases, gridCells, velocity, pressureSamples, divergence, pressure, density);
    this.gridOverlayPipeline?.setSparseSource(sparsePresentation?.adaptiveGrid
      ?? this.gpuFluid?.sparseAdaptiveGridSource);
    this.techniqueOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    this.techniqueOverlayPipeline?.setOwnerRows(pressureSamples);
    this.techniqueAuditOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    this.techniqueAuditOverlayPipeline?.setOwnerRows(pressureSamples);
    this.fluidCellTracePipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource, pressureSamples);
  }

  private solverKey(scene:SceneDescription,config:SimulationRunConfig,presentationMode:ScenePresentationMode){
    return `${gpuSceneSolverKey(scene,config)}:presentation-${presentationMode}`;
  }
  /** Presentation policy used to construct the attached solver/sidecar pair. */
  private attachedPresentationMode: ScenePresentationMode = "full-scene";
  /** Scalars already adopted by the live solver; empty until one is attached. */
  private appliedSceneUniformKey="";
  private reseedInFlight=false;

  /**
   * Attempt a warm re-seed of the live solver. Only legal when the structural
   * tier is unchanged — a different lattice or method has to be built, not
   * re-seeded. Returns true when an attempt is under way, in which case the
   * caller must skip this frame; the attempt either promotes the solver to the
   * new key or leaves the ordinary rebuild to run on a later frame.
   */
  private tryReseedGPUFluid(scene:SceneDescription,config:SimulationRunConfig,key:string,presentationMode:ScenePresentationMode):boolean{
    const solver=this.gpuFluid;
    if(!solver?.reseed||this.reseedInFlight)return false;
    if(gpuSceneStructuralKey(scene,config)!==this.attachedStructuralKey)return false;
    const generation=this.gpuFluidGeneration,requestGeneration=this.gpuFluidRequestGeneration;
    this.reseedInFlight=true;
    const resource=getMethod(config.methodId).resource;
    this.onStatus({state:"initializing",label:"Re-seeding fenced t=0 solver authority",phase:"warmup",completed:0,total:1,startedAt_ms:performance.now(),kind:"rebuild",retainingPrevious:false,resource});
    void solver.reseed(scene).then((reseeded)=>{
      // Anything that replaced or invalidated the solver mid-flight wins.
      if(this.disposed||this.gpuFluid!==solver||this.gpuFluidGeneration!==generation
        ||this.gpuFluidRequestGeneration!==requestGeneration)return;
      if(!reseeded){this.beginGPUFluidInitialization(scene,config,key,presentationMode);return;}
      this.gpuFluidKey=key;this.appliedSceneUniformKey=gpuSceneUniformKey(scene);this.resetGPUQueueTracking();
      if(presentationMode === "full-scene")this.attachSparsePresentationSource(
        solver,requestGeneration,performance.now(),solver.sparseVoxelSceneSource);
      // reset() intentionally clears the diagnostics store. A warm re-seed
      // must therefore republish its authority just like a replacement attach,
      // then earn a fresh raster fence before transport unlocks. Merely moving
      // gpuFluidKey leaves the renderer ready while the UI waits forever for
      // initialSparseAuthorityReady/initialRasterSurfaceReady to reappear.
      solver.info.initialRasterSurfaceReady=false;
      solver.info.initialRasterSurfaceState="pending";
      solver.info.initialRasterSurfaceDiagnostic="Waiting for the first fenced t=0 raster publication after re-seed";
      this.globalFineWaterAttached=false;
      this.pendingInitialRasterPresentation={solver,solverGeneration:generation,requestGeneration,submitted:false,resource};
      this.gpuInfoCallback?.({...solver.info});
      this.onStatus({state:"initializing",label:"Re-seeded solver ready; publishing fenced t=0 raster surface",phase:"presentation",completed:0,total:1,startedAt_ms:performance.now(),kind:"rebuild",retainingPrevious:false,resource});
    }).catch(()=>{
      if(!this.disposed&&this.gpuFluid===solver&&this.gpuFluidGeneration===generation
        &&this.gpuFluidRequestGeneration===requestGeneration)this.beginGPUFluidInitialization(scene,config,key,presentationMode);
    }).finally(()=>{
      this.reseedInFlight=false;
      // Both success and failure need another paused draw: success submits the
      // t=0 raster fence; failure falls through to the full rebuild path.
      if(!this.disposed)this.pausedPresentationRevision+=1;
    });
    return true;
  }
  /** Structural identity of the attached solver, gating warm re-seeds. */
  private attachedStructuralKey="";

  private resetGPUQueueTracking() {
    this.gpuPendingBatches = 0;
    this.pendingGPUAdvanceCompletions.length = 0;
    this.reportedSimulationPipelineState = "";
    this.reportedSparseWorldStatus = "";
  }

  /**
   * The renderer's only view of sparse simulation state. Both values come
   * from the public sparse-world module; adapter compilation details stay on
   * the other side of that boundary.
   */
  private sparseWorldSnapshot(
    solver: GPUSolverInstance | undefined,
  ): SparseWorldSnapshot | undefined {
    if (!solver?.sparseWorld) return undefined;
    return {
      // A sparse solver that has not supplied its device owner is incomplete.
      // Fail closed while adapters finish migrating to the public contract.
      deviceStatus: solver.sparseWorldDevice?.status ?? "loading",
      deviceFault: solver.sparseWorldDevice?.fault,
      status: solver.sparseWorld.status(),
      presentation: solver.sparseWorld.presentation(),
    };
  }

  /** Project the public sparse receipt across the renderer worker seam. */
  private refreshSparseWorldState(
    solver: GPUSolverInstance,
    publish = false,
  ): SparseWorldSnapshot | undefined {
    const snapshot = this.sparseWorldSnapshot(solver);
    if (!snapshot) return undefined;
    const { deviceStatus, deviceFault, status } = snapshot;
    solver.info.sparseWorldDeviceStatus = deviceStatus;
    solver.info.sparseWorldDeviceFault = deviceFault;
    solver.info.sparseWorldStatus = status;
    const key = `${deviceStatus}:${deviceFault?.code ?? ""}:${status.state}`
      + `:${status.acceptedGeneration}:${status.residentTiles}`
      + `:${status.capacityTiles}:${status.lastAcceptedTime}:${status.fault?.code ?? ""}`;
    if (publish && key !== this.reportedSparseWorldStatus) {
      this.reportedSparseWorldStatus = key;
      this.gpuInfoCallback?.({ ...solver.info });
    }
    return snapshot;
  }

  /** One accepted presentation snapshot supplies every sparse renderer binding. */
  private sparseWorldPresentation(
    solver: GPUSolverInstance | undefined,
  ): SparseWorldPresentation | undefined {
    return this.sparseWorldSnapshot(solver)?.presentation;
  }

  private sparseAuthorityReady(solver: GPUSolverInstance | undefined): boolean {
    const status = this.sparseWorldSnapshot(solver)?.status;
    return status ? status.state !== "fault" : solver?.initialSparseAuthorityReady === true;
  }

  private sparseDeviceReady(solver: GPUSolverInstance | undefined): boolean {
    const snapshot = this.sparseWorldSnapshot(solver);
    return snapshot ? snapshot.deviceStatus === "ready" && snapshot.status.state !== "fault"
      : solver?.simulationReady !== false;
  }

  /** Begin a new controller timeline before any old GPU completion can commit. */
  /**
   * Add a ball of liquid to the attached solve.
   *
   * Returns whether a solver took it. False means there is nothing running to
   * add water to, or the attached method has no injection — either way the
   * caller has to fall back to re-seeding the document, which is why this
   * answers rather than silently doing nothing.
   */
  injectLiquidBall(ball: InjectedLiquidBall): boolean {
    if (this.disposed || this.deviceLost
      || !this.sparseDeviceReady(this.gpuFluid)
      || !this.gpuFluid?.injectLiquidBall) return false;
    this.gpuFluid.injectLiquidBall(ball);
    return true;
  }

  resetSimulationTimeline(): void {
    if (this.disposed || this.deviceLost) return;
    this.simulationRunning = false;
    this.timelineResetPending = true;
    this.pendingInitialRasterPresentation = undefined;
    // Every admitted completion captures this generation. Advancing it here,
    // rather than at replacement attachment, makes old callbacks stale at the
    // same synchronous instant that the controller publishes t=0.
    this.gpuFluidGeneration += 1;
    this.resetGPUQueueTracking();
    this.resetPresentationTrace();
  }

  /** Change simulation admission while preserving already-submitted queue work. */
  setSimulationRunning(running: boolean): number | undefined {
    const changed = running !== this.simulationRunning;
    if (changed) this.resetPresentationTrace();
    this.simulationRunning = running;
    // Live frames never map solver state. A pause is the explicit ownership
    // boundary where the UI may refresh its human-rate diagnostics.
    if (changed && !running && this.gpuFluid) {
      const fluid = this.gpuFluid;
      void fluid.readStats().then((info) => {
        if (!this.disposed && !this.deviceLost && this.gpuFluid === fluid && !this.simulationRunning) {
          this.gpuInfoCallback?.({ ...info });
        }
      }).catch(() => { /* Device loss is reported by device.lost. */ });
      // The film's iteration records ride the same boundary, and for the same
      // reason: reading them maps a buffer the solver owns. Pausing is also
      // when the curve is wanted — it is what one does to study a solve — so
      // the restriction and the interaction want the same moment.
      const readPressureFilm = fluid.sparseWorldUI
        ? () => fluid.sparseWorldUI!.diagnostics.readPressureFilm()
        : fluid.readPressureJournal?.bind(fluid);
      if (this.gpuPressureJournalCallback && readPressureFilm) {
        void readPressureFilm().then((journal) => {
          if (!this.disposed && !this.deviceLost && this.gpuFluid === fluid
            && !this.simulationRunning) {
            this.gpuPressureJournalCallback?.(journal);
          }
        }).catch(() => { /* Device loss is reported by device.lost. */ });
      }
    }
    const submittedTime_s = this.gpuFluid?.info.submittedTime_s;
    return submittedTime_s;
  }

  /**
   * Hand the panel this frame's lens receipt, or empty it.
   *
   * Every armed frame rather than at a pause, unlike the film: these counters
   * describe the advance that just ran and a lens is watched while the run
   * moves, so the pause boundary the film needs would be the one moment the
   * numbers are least interesting. Nothing here maps a buffer — the receipt is
   * whatever the source already read back.
   *
   * The clearing post is what the tracking field exists for. Without it the
   * panel would go on showing the last armed frame after the view closed, and a
   * frozen receipt beside an empty viewport is the one reading that is never
   * true of anything.
   */
  private publishStageLensReceipt(lensId: string | undefined): void {
    const callback = this.gpuStageLensCallback;
    if (!callback) return;
    if (!lensId) {
      if (this.publishedStageLens === undefined) return;
      this.publishedStageLens = undefined;
      callback(undefined, []);
      return;
    }
    this.publishedStageLens = lensId;
    callback(
      (this.gpuFluid?.sparseWorldUI?.overlays.stageLenses
        ?? this.gpuFluid?.stageLensSource)?.receipt(lensId),
      this.stageLensOverlayPipeline?.report ?? [],
    );
  }

  /** Clear presentation samples whenever their semantic identity changes. */
  private resetPresentationTrace() {
    this.latestPresentationTrace = undefined;
    this.latestPresentationStageTrace = undefined;
    this.latestPresentationBandTrace = undefined;
    this.latestPresentationManifest = undefined;
  }

  private currentFrameMetrics(
    methodId: string,
    context: string,
    presentationSubmitted: boolean,
    cpu?: PerformanceTrace,
  ): RendererFrameMetrics {
    const water = this.waterPipeline?.surfaceRenderDiagnostics;
    const instrumentation = usePerformanceInstrumentationStore.getState();
    const presentation = instrumentation.enabled
      && this.latestPresentationTrace
      && this.latestPresentationTrace.capturedAt_ms >= instrumentation.enabledAt_ms
      ? this.latestPresentationTrace
      : undefined;
    const presentationStages = instrumentation.enabled
      && this.latestPresentationStageTrace
      && this.latestPresentationStageTrace.capturedAt_ms >= instrumentation.enabledAt_ms
      ? this.latestPresentationStageTrace
      : undefined;
    const presentationBands = instrumentation.enabled
      && this.latestPresentationBandTrace
      && this.latestPresentationBandTrace.capturedAt_ms >= instrumentation.enabledAt_ms
      ? this.latestPresentationBandTrace
      : undefined;
    return {
      cpu,
      presentation,
      presentationStages,
      presentationBands,
      presentationStageManifest: instrumentation.enabled ? this.latestPresentationManifest : undefined,
      context,
      methodId,
      presentationSubmitted,
      ...(water ? { waterSurfacePresentation: {
        surfaceGeometrySource: water.surfaceGeometrySource,
        globalFineAttached: water.globalFineAttached,
        globalFineAttachedGeneration: water.globalFineAttachedGeneration,
        meshPublicationGeneration: water.meshPublicationGeneration,
        globalFineCrossingPublished: water.globalFineCrossingPublished,
        presentationFallbackActive: water.presentationFallbackActive,
        sourceFrameCounts: water.sourceFrameCounts,
      } } : {}),
    };
  }

  private retireGPUFluid(fluid: GPUSolverInstance) {
    const device = this.device;
    if (!device || this.deviceLost) { fluid.destroy(); return; }
    this.retiredGPUFluids.add(fluid);
    // A method switch can occur after a frame encoded the old solver's
    // textures but before that frame submits. Defer the queue fence to the
    // next animation frame so it covers that final submission.
    requestAnimationFrame(() => {
      void device.queue.onSubmittedWorkDone().catch(() => { /* Device loss invalidates the resources. */ }).finally(() => {
        if (this.retiredGPUFluids.delete(fluid)) fluid.destroy();
      });
    });
  }

  private beginGPUFluidInitialization(scene:SceneDescription,config:SimulationRunConfig,key:string,presentationMode:ScenePresentationMode){
    if(!this.device||this.disposed||this.deviceLost)return;
    const method=getMethod(config.methodId);if(!canInitializeGPUSceneSource(scene,config.methodId))return;
    const sparseWorldMethod=method.capabilities?.sparseWorld===true;
    const rendererOnlyScene=!planSceneRuntime(scene).fluidSolver;
    const initializationResource=rendererOnlyScene?liveSvoSceneResourcePlugin:method.resource;
    this.gpuFluidInitializationAbort?.abort();
    /**
     * The build this one replaces, so this one can wait for it to let go.
     *
     * A live-SVO build is now interruptible *and* interleaved: it returns to
     * the event loop between slices, which is what lets the worker see the
     * message that supersedes it at all. The cost of that is an overlap window
     * — the abort above only takes effect at the superseded build's next slice,
     * and until then it still holds every arena it has allocated. Starting the
     * replacement inside that window means two refined worlds resident on one
     * device, which at environment refinement depth 3 is the difference between
     * an allocation that fits and one that steps the refinement ladder down.
     *
     * So the replacement waits. The wait is bounded by the superseded build's
     * slice, not by its remaining work, and it is a `catch(() => {})` because
     * the thing being waited for has just been told to fail.
     */
    const supersededBuild=this.gpuFluidPending;
    const abort=new AbortController();this.gpuFluidInitializationAbort=abort;
    const device=this.device,generation=++this.gpuFluidRequestGeneration,startedAt_ms=performance.now();
    const previous=this.gpuFluid;
    const previousSidecar=this.svoSceneSidecar;
    const drainPreviousForReset=this.timelineResetPending&&Boolean(previous);
    this.timelineResetPending=false;
    this.pendingLiveSvoPresentation=undefined;
    // The active solver remains attached for presentation throughout the
    // transaction. Only the warmed candidate is allowed to replace it.
    this.gpuFluidPendingKey=key;
    let reportedCompleted=0,reportedTotal=1;
    const report=(progress:{phase:string;taskId?:string;label:string;completed:number;total:number})=>{if(this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration)return;reportedCompleted=progress.completed;reportedTotal=progress.total;this.onStatus(sparseWorldMethod
      ? {state:"initializing",label:"Loading sparse world",startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:Boolean(previous),resource:initializationResource}
      : {state:"initializing",...progress,startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:Boolean(previous),resource:initializationResource});};
    let previousDestroyedForReset=false;
    let previousSidecarDestroyedForReset=false;
    const prepare=async()=>{
      if(supersededBuild)await supersededBuild.catch(()=>{});
      if(abort.signal.aborted||this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration)throw new DOMException("GPU initialization superseded","AbortError");
      if(!drainPreviousForReset||!previous)return;
      report({phase:"drain",taskId:"solver.drain",label:"Drain previous GPU work",completed:0,total:1});
      await device.queue.onSubmittedWorkDone();
      if(abort.signal.aborted||this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration)throw new DOMException("GPU initialization superseded","AbortError");
      // Reset does not need the previous frame as a live fallback. Detach every
      // presentation binding before destroying the old solver so the new
      // allocation never overlaps its large field set on the same device.
      if(this.gpuFluid===previous){
        this.gpuFluid=undefined;this.gpuFluidKey="";
        this.updateRenderSources();
        this.secondaryParticlePipeline?.setSource(undefined);
        this.svoDrySceneSource=undefined;this.svoDrySceneData=undefined;this.liveSceneAnimation=undefined;this.liveSceneAnimationFailure=undefined;this.renderSceneKey="";this.renderSceneStamp=0;this.svoDryScenePipeline?.setSource(undefined);
        previous.destroy();previousDestroyedForReset=true;
      }
      if(this.svoSceneSidecar===previousSidecar&&previousSidecar){
        this.svoSceneSidecar=undefined;
        previousSidecar.destroy();previousSidecarDestroyedForReset=true;
      }
      this.resetGPUQueueTracking();
      report({phase:"drain",taskId:"solver.drain",label:"Previous GPU work drained",completed:1,total:1});
    };
    const create:Promise<{solver:GPUSolverInstance;sidecar?:WebGPULiveSvoScene}>=prepare().then(async ()=>{
      if(abort.signal.aborted||this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration)throw new DOMException("GPU initialization superseded","AbortError");
      let solver:GPUSolverInstance;
      if (!planSceneRuntime(scene).fluidSolver) {
        const refinement = config.values.svoEnvironmentBrickRefinementLevels;
        const depth = config.values.svoEnvironmentRefinementDepth;
        solver=await WebGPULiveSvoScene.create(device, scene, config.quality, report, abort.signal, {
          environmentBrickRefinementLevels: typeof refinement === "number" ? refinement : undefined,
          environmentRefinementDepth: typeof depth === "number" ? depth : undefined,
          environmentPlanarRefinementExemption: config.values.svoEnvironmentPlanarRefinementExemption === true,
        });
      } else {
        solver=await (method.createSolverAsync
          ? method.createSolverAsync(device,scene,config.quality,config.values,this.gpuRigidLoadCallback,report,abort.signal)
          : new Promise<GPUSolverInstance>((resolve,reject)=>setTimeout(()=>{try{resolve(method.createSolver!(device,scene,config.quality,config.values,this.gpuRigidLoadCallback));}catch(error){reject(error);}},0)));
      }
      if (presentationMode === "fluid-only" || solver.sparseVoxelSceneSource) return {solver};
      try {
        const refinement=config.values.svoEnvironmentBrickRefinementLevels;
        const depth=config.values.svoEnvironmentRefinementDepth;
        const sidecar=await WebGPULiveSvoScene.create(device,scene,config.quality,report,abort.signal,{
          environmentBrickRefinementLevels:typeof refinement==="number"?refinement:undefined,
          environmentRefinementDepth:typeof depth==="number"?depth:undefined,
          environmentPlanarRefinementExemption:config.values.svoEnvironmentPlanarRefinementExemption===true,
        });
        return {solver,sidecar};
      } catch(error) {
        solver.destroy();
        throw error;
      }
    });
    this.gpuFluidPending=create.then(({solver,sidecar})=>{
      if(this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration){solver.destroy();sidecar?.destroy();return;}
      if(requiresFencedInitialRasterPresentation(config.methodId)&&!this.sparseAuthorityReady(solver)){solver.destroy();sidecar?.destroy();throw new Error(`${method.label} solver returned before fenced sparse t=0 authority`);}
      report({phase:"attach",taskId:"solver.attach",label:"Attach warmed solver",completed:reportedCompleted,total:reportedTotal+1});
      solver.applyRuntimeValues?.(config.values);
      this.gpuFluid=solver;this.svoSceneSidecar=sidecar;this.gpuFluidKey=key;this.attachedPresentationMode=presentationMode;this.attachedStructuralKey=gpuSceneStructuralKey(scene,config);this.gpuFluidPendingKey="";this.resetGPUQueueTracking();this.gpuFluidGeneration+=1;this.globalFineWaterAttached=false;
      const sparseWorldState=this.refreshSparseWorldState(solver);
      const fencedInitialRaster=requiresFencedInitialRasterPresentation(config.methodId);
      if(rendererOnlyScene){solver.info.initialRasterSurfaceReady=true;solver.info.initialRasterSurfaceState="gpu-authoritative";solver.info.initialRasterSurfaceDiagnostic="Live scene source ready; fluid authority intentionally absent";this.pendingInitialRasterPresentation=undefined;}
      else if(fencedInitialRaster){solver.info.initialRasterSurfaceReady=false;solver.info.initialRasterSurfaceState="pending";solver.info.initialRasterSurfaceDiagnostic="Waiting for the first fenced t=0 raster publication";this.pendingInitialRasterPresentation={solver,solverGeneration:this.gpuFluidGeneration,requestGeneration:generation,submitted:false,resource:method.resource};}
      else{solver.info.initialRasterSurfaceReady=true;solver.info.initialRasterSurfaceState="gpu-authoritative";solver.info.initialRasterSurfaceDiagnostic="Direct solver field attached; sparse raster fence not required";this.pendingInitialRasterPresentation=undefined;}
      this.updateRenderSources(solver.surfaceFieldTexture??solver.volumeTexture,solver.columnBaseTexture,solver.gridCellTexture??this.gridCellTexture,solver.velocityTexture??this.velocityFallbackTexture,solver.gridPressureSamplesTexture??this.pressureSamplesFallbackTexture,solver.gridDivergenceTexture??this.scalarFallbackTexture,solver.gridPressureTexture??this.scalarFallbackTexture,solver.volumeTexture);this.secondaryParticlePipeline?.setSource(solver.secondaryParticles);
      if(presentationMode === "full-scene")this.attachSparsePresentationSource(solver,generation,startedAt_ms,sidecar?.sparseVoxelSceneSource??solver.sparseVoxelSceneSource);
      else{this.svoDrySceneSource=undefined;this.svoDrySceneData=undefined;this.liveSceneAnimation=undefined;this.liveSceneAnimationFailure=undefined;this.svoDryScenePipeline?.setSource(undefined);}
      // A world that had to step its refinement ladder down says so in the one
      // status a user reads to the end, because "the leaf is 3.125 mm, not the
      // 1.5625 mm you asked for" is otherwise indistinguishable from a depth
      // control that did nothing. The unchanged case keeps the plain label.
      const degradedDepth=solver instanceof WebGPULiveSvoScene&&solver.builtRefinementDepth!==solver.requestedRefinementDepth?solver:undefined;
      if(rendererOnlyScene&&degradedDepth)
        this.onStatus({state:"ready",label:`Live sparse scene source ready at refinement depth ${degradedDepth.builtRefinementDepth}; depth ${degradedDepth.requestedRefinementDepth} could not be allocated`,adapter:this.adapterName,resource:liveSvoSceneResourcePlugin});
      else if(rendererOnlyScene)
        this.onStatus({state:"ready",label:"Live sparse scene source ready",adapter:this.adapterName,resource:liveSvoSceneResourcePlugin});
      this.pausedPresentationRevision+=1;
      if(previous&&previous!==solver&&!previousDestroyedForReset)this.retireGPUFluid(previous);
      if(previousSidecar&&previousSidecar!==sidecar&&!previousSidecarDestroyedForReset)this.retireGPUFluid(previousSidecar);
      this.gpuInfoCallback?.(solver.info);
      const sparseFault=sparseWorldState?.status.fault??sparseWorldState?.deviceFault;
      if(!rendererOnlyScene&&solver.sparseWorld&&(sparseWorldState?.status.state==="fault"||sparseWorldState?.deviceStatus==="fault"))this.onStatus({state:"blocked",label:`Sparse world fault · ${sparseFault?.code ?? "internal"}`,resource:method.resource});
      else if(!rendererOnlyScene&&solver.sparseWorld&&sparseWorldState?.deviceStatus!=="ready")this.onStatus({state:"initializing",label:"Loading sparse world",startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:false,resource:method.resource});
      else if(!rendererOnlyScene&&solver.sparseWorld&&fencedInitialRaster)this.onStatus({state:"initializing",label:"Loading sparse world",startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:false,resource:method.resource});
      else if(!rendererOnlyScene&&solver.sparseWorld)this.onStatus({state:"ready",label:sparseWorldState?.status.state==="saturated"?"Sparse world capacity reached":"Sparse world ready",adapter:this.adapterName,resource:method.resource});
      else if(!rendererOnlyScene&&fencedInitialRaster)this.onStatus({state:"initializing",label:"Warmed solver attached; publishing fenced t=0 raster surface",phase:"presentation",completed:reportedCompleted,total:reportedTotal+1,startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:false,resource:method.resource});
      else if(!rendererOnlyScene)this.onStatus({state:"ready",label:"WebGPU direct-field solver ready",adapter:this.adapterName,resource:method.resource});
    }).catch((error:unknown)=>{if(this.disposed||generation!==this.gpuFluidRequestGeneration)return;this.gpuFluidPendingKey="";this.pendingInitialRasterPresentation=undefined;this.pendingLiveSvoPresentation=undefined;if(isGPUInitializationAbort(error))return;if(sparseWorldMethod){const label="Sparse world fault · initialization";if(previous)this.onStatus({state:"ready",label:`${label}; previous world retained`,adapter:this.adapterName,resource:initializationResource});else this.onStatus({state:"unavailable",label,resource:initializationResource});}else if(previous)this.onStatus({state:"ready",label:error instanceof Error?`Solver rebuild failed; previous solver retained: ${error.message}`:"Solver rebuild failed; previous solver retained",adapter:this.adapterName,resource:initializationResource});else this.onStatus({state:"unavailable",label:error instanceof Error?`GPU initialization failed: ${error.message}`:"GPU initialization failed",resource:initializationResource});}).finally(()=>{if(generation===this.gpuFluidRequestGeneration){this.gpuFluidPending=undefined;if(this.gpuFluidInitializationAbort===abort)this.gpuFluidInitializationAbort=undefined;}});
  }

  /**
   * Scene the solver is keyed against, when it differs from the one being
   * drawn.
   *
   * Direct manipulation proposes a scene before it commits one (see
   * `scene-draft-store`). A proposal that only changes what the *ground* looks
   * like should redraw immediately, but must not reach the rebuild key — the
   * solver would re-seed on every pointer-move. Setting this pins the solver to
   * the committed scene while `draw` presents the proposed one.
   *
   * Only safe for proposals that cannot change the lattice: the renderer draws
   * the fluid from solver-owned textures, so presenting different container
   * extents than the solver allocated would tear. The viewport is what enforces
   * that, by pinning only for drafts it knows are geometry-preserving.
   */
  setSimulationScene(scene: SceneDescription | undefined) {
    this.simulationScene = scene;
  }
  private simulationScene?: SceneDescription;

  /** Adopt the terrain stamp computed before a scene crosses into a worker. */
  setRenderSceneTerrainContentStamp(stamp: string | undefined) {
    this.renderSceneTerrainContentStamp = stamp;
  }

  /**
   * Publish the presentation scene independently from the solver identity.
   * The solver receives the same revision through its live-maintenance seam,
   * while renderer-owned analytic/material/light arenas become visible on the
   * next submitted frame without allocation or bind-group churn.
   */
  private publishRenderScene(scene: SceneDescription, solver: GPUSolverInstance | undefined): void {
    const stampedRevision = sceneRevision(scene);
    // Worker publications are retained and stamped when they arrive, so this
    // fallback is now reserved for direct/headless callers that supply an
    // external document without entering through `markSceneRevision`.
    const sceneKey = stampedRevision === undefined ? canonicalScene(scene) : "";
    if (stampedRevision === undefined
      ? sceneKey === this.renderSceneKey
      : stampedRevision === this.renderSceneStamp) return;
    solver?.stageSceneUpdate?.(scene);
    const source = solver?.sparseVoxelSceneSource ?? this.svoDrySceneSource;
    if (!source) {
      this.svoSourceAvailable = false;
      this.svoPublicationFailure = sparseVoxelDrySceneContractFailure(undefined, undefined);
      this.svoDrySceneData = undefined;
      this.svoDryScenePipeline?.setSource(undefined);
      return;
    }
    try {
    const revision = this.renderSceneRevision >= 0xffff_fffe ? 1 : this.renderSceneRevision + 1;
    const scenePrimitives = buildSvoScenePrimitives(scene);
    const primitiveCandidates = scenePrimitives.primitiveCandidates
      ?? buildSvoPrimitiveCandidates(scenePrimitives.descriptors as Parameters<typeof buildSvoPrimitiveCandidates>[0], {
        skippedOwnerId: scenePrimitives.openShellOwnerId,
      });
    const terrainSurface = sceneTerrainSurfaceModel(scene);
    const materialRecords = packSvoMaterialTable([
      ...buildDefaultSvoMaterialRecords(revision, { terrainSurface }),
      ...scenePrimitives.metadata.map((primitive) => svoMaterialFromEnvironmentProxyMaterial(
        primitive.materialId,
        primitive.material,
        revision,
        // The same word that chose the ground's closure chooses every prop's:
        // a porcelain scene is one fired material throughout, not a porcelain
        // floor with granite on it.
        svoMaterialFunctionIdForEnvironmentProxy(primitive, terrainSurface),
      )),
    ]);
    const sceneGlass = buildSvoSceneGlass(scene, { cellSize_m: source.structural?.domain.cellSize_m });
    const sceneThickGlass = buildSvoSceneThickGlass(scene, { revision });
    const thickReplacedPaneKey = sceneThickGlass.metadata.find(({ replacesThinPaneKey }) => Boolean(replacesThinPaneKey))?.replacesThinPaneKey;
    const thickReplacedPaneId = sceneGlass.metadata.find(({ key }) => key === thickReplacedPaneKey)?.paneId;
    const lightingMirrors = buildSparseVoxelDrySceneLightingMirrors(scene, revision);
    if (!lightingMirrors) {
      this.svoSourceAvailable = false;
      this.svoPublicationFailure = "lighting mirror construction failed";
      this.svoDrySceneData = undefined;
      this.svoDryScenePipeline?.setSource(undefined);
      return;
    }
    const publication: SparseVoxelDrySceneData = {
      renderRevision: revision,
      primitiveRecords: scenePrimitives.packedRecords,
      primitiveCandidates,
      materialRecords,
      materialRevision: revision,
      ownerBase: SCENE_ENVIRONMENT_OWNER_BASE,
      skippedOwnerId: scenePrimitives.openShellOwnerId,
      // An aggregate's packing does not fit in a 64-byte record, so the record
      // carries a reference and the numbers live in the arena. Publication
      // order assigns the references, so these arrive already in agreement
      // with the records beside them.
      clusterBlocks: scenePrimitives.clusterBlocks,
      // The same door again, for the tape kind. A `field-program` record whose
      // block never arrives resolves to zeroes, which the shader reads as "not
      // resolved" and draws as nothing — the aggregate's failure one level up.
      fieldProgramBlocks: scenePrimitives.fieldProgramBlocks,
      glassRecords: sceneGlass.packedRecords,
      glassCacheKey: sceneGlass.cacheKey,
      thickGlassRecords: sceneThickGlass.packedRecords,
      thickGlassRevision: sceneThickGlass.revision,
      thickGlassCacheKey: sceneThickGlass.cacheKey,
      thickGlassReplacedThinPaneId: thickReplacedPaneId,
      ...lightingMirrors,
      flatVoxelNormals: sceneUsesFlatVoxelNormals(scene),
    };
    const thickGlassBound = resolveSparseVoxelThickGlassBinderStatus(publication) === "bound";
    const replacedPaneKeys = new Set(sceneThickGlass.metadata.flatMap(({ replacesThinPaneKey }) => replacesThinPaneKey ? [replacesThinPaneKey] : []));
    this.svoGlassSupported = !sceneGlass.metadata.some(({ key, opaqueCutoutKey }) => Boolean(opaqueCutoutKey) && (!thickGlassBound || !replacedPaneKeys.has(key)));
    this.svoMaterialsSupported = materialRecords.byteLength > 0;
    this.svoLightingSupported = canConsumeSparseVoxelLighting(publication);
    const supported = this.svoGlassSupported && this.svoMaterialsSupported && this.svoLightingSupported;
    const contractFailure = sparseVoxelDrySceneContractFailure(source, publication);
    this.svoSourceAvailable = supported && !contractFailure;
    if (!this.svoSourceAvailable) {
      const reason = contractFailure ?? "authored scene capabilities are unsupported";
      if (reason !== this.svoPublicationFailure) {
        this.svoPublicationFailure = reason;
        console.error(`Live sparse presentation rejected: ${reason}`);
        this.onStatus({ state: "blocked", label: `Live sparse presentation rejected: ${reason}`, resource: svoPresentationResourcePlugin });
      }
      this.svoDrySceneData = undefined;
      this.svoDryScenePipeline?.setSource(undefined);
      return;
    }
    this.svoPublicationFailure = undefined;
    this.svoDrySceneSource = source;
    this.svoDrySceneData = publication;
    this.svoDryScenePipeline?.setSource(source);
    if (this.svoDryScenePipeline && !this.svoDryScenePipeline.publishScene(publication)) {
      this.svoSourceAvailable = false;
      this.svoPublicationFailure = sparseVoxelDrySceneContractFailure(source, publication)
        ?? "live SVO renderer rejected the scene arena publication";
      this.svoDrySceneData = undefined;
      this.svoDryScenePipeline.setSource(undefined);
      return;
    }
    const sway = scenePrimitives.metadata.map(({ sway: authoredSway }) => authoredSway);
    const swayIndices = sway.flatMap((value, index) => value ? [index] : []);
    this.liveSceneAnimation = swayIndices.length ? {
      rest: scenePrimitives.descriptors,
      sway,
      swayIndices,
      current: [...scenePrimitives.descriptors],
      records: new Uint32Array(scenePrimitives.packedRecords),
      refitPlan: createSvoPrimitiveCandidateRefitPlan(primitiveCandidates),
    } : undefined;
    this.renderSceneRevision = revision;
    this.renderSceneKey = sceneKey;
    this.renderSceneStamp = stampedRevision ?? 0;
    this.pausedPresentationRevision += 1;
    } catch (error) {
      // The authored live scene is the sole authority. Retaining the prior
      // complete image would silently present a different scene.
      this.svoSourceAvailable = false;
      const reason = error instanceof Error ? error.message : "scene publication threw an unknown error";
      if (reason !== this.svoPublicationFailure) {
        this.svoPublicationFailure = reason;
        console.error("Live sparse presentation publication failed", error);
        this.onStatus({ state: "blocked", label: `Live sparse presentation publication failed: ${reason}`, resource: svoPresentationResourcePlugin });
      }
      this.svoDrySceneData = undefined;
      this.svoDryScenePipeline?.setSource(undefined);
    }
  }

  /** Refit bounded authored motion through the exact render arena only. */
  private advanceLiveSceneAnimation(): void {
    const animation = this.liveSceneAnimation;
    const pipeline = this.svoDryScenePipeline;
    if (!animation || !pipeline || !this.svoDrySceneData) return;
    const now_ms = performance.now();
    animation.origin_ms ??= now_ms;
    const time_s = Math.max(0, now_ms - animation.origin_ms) / 1000;
    const dirtyBounds: SvoDrySceneDirtyBounds[] = [];
    for (const index of animation.swayIndices) {
      const before = animation.current[index];
      const sway = animation.sway[index]!;
      const after = swayedPrimitiveDescriptor(animation.rest[index], sway, time_s);
      animation.current[index] = after;
      animation.records.set(packSvoPrimitiveRecords([after]), index * SVO_PRIMITIVE_RECORD_WORDS);
      const oldBounds = svoPrimitiveCandidateBounds(before);
      const newBounds = svoPrimitiveCandidateBounds(after);
      dirtyBounds.push({
        minimum: [
          Math.min(oldBounds.minimum_m.x, newBounds.minimum_m.x),
          Math.min(oldBounds.minimum_m.y, newBounds.minimum_m.y),
          Math.min(oldBounds.minimum_m.z, newBounds.minimum_m.z),
        ],
        maximum: [
          Math.max(oldBounds.maximum_m.x, newBounds.maximum_m.x),
          Math.max(oldBounds.maximum_m.y, newBounds.maximum_m.y),
          Math.max(oldBounds.maximum_m.z, newBounds.maximum_m.z),
        ],
      });
    }
    const refit = refitSvoPrimitiveCandidatesIncremental(
      animation.current as Parameters<typeof refitSvoPrimitiveCandidatesIncremental>[0],
      animation.refitPlan,
      animation.swayIndices,
    );
    const candidates = refit.publication;
    const records = animation.records;
    const revision = this.renderSceneRevision >= 0xffff_fffe ? 1 : this.renderSceneRevision + 1;
    if (!pipeline.publishPrimitiveArena(records, candidates, revision, {
      dirtyBounds,
      dirtyPrimitiveIndices: animation.swayIndices,
      dirtyCandidateNodeIndices: refit.dirtyNodeIndices,
      // Authored sway is constrained to the finest-cell ownership margin. The
      // exact analytic surface moves, while its sparse owner and baked lighting
      // remain valid at the reference pose. Restaging the sparse world here
      // needlessly re-voxelized the tree and rebuilt derived lighting every
      // frame, producing the garden's repeating FPS drain-and-recovery cycle.
      derivedLighting: "unchanged",
    })) {
      const reason = "the analytic primitive arena is not ready for the live render generation";
      this.liveSceneAnimationFailure = reason;
      this.onStatus({ state: "blocked", label: `Live scene update failed closed: ${reason}`, resource: svoPresentationResourcePlugin });
      return;
    }
    this.liveSceneAnimationFailure = undefined;
    this.renderSceneRevision = revision;
    this.svoDrySceneData = { ...this.svoDrySceneData, renderRevision: revision, primitiveRecords: records, primitiveCandidates: candidates };
  }

  /** The solver's sparse world when it has one, otherwise the renderer sidecar. */
  private sparseSceneProducer(fluid: GPUSolverInstance | undefined): GPUSolverInstance | undefined {
    return fluid?.sparseVoxelSceneSource ? fluid : this.svoSceneSidecar;
  }

  private currentGPUFluid(scene: SceneDescription, config: SimulationRunConfig, presentationMode: ScenePresentationMode) {
    if (!this.device || this.disposed || this.deviceLost) return undefined;
    if (!canInitializeGPUSceneSource(scene, config.methodId)) return undefined;
    const key=this.solverKey(scene,config,presentationMode);
    if(!this.gpuFluid||key!==this.gpuFluidKey){
      // A change confined to the seed tier can re-seed the live solver instead
      // of rebuilding it. The attempt is fire-and-forget against a generation
      // guard; if it declines or the solver moved on, the ordinary rebuild
      // below still runs, so this can only make the path faster, never wrong.
      if(this.gpuFluid&&this.attachedPresentationMode===presentationMode&&this.gpuFluidPendingKey!==key&&this.tryReseedGPUFluid(scene,config,key,presentationMode))return undefined;
      if(this.gpuFluidPendingKey!==key)this.beginGPUFluidInitialization(scene,config,key,presentationMode);
      return undefined;
    }
    // A timeline reset is represented by simulationEpoch in the key above.
    // Presentation time can briefly trail submitted GPU time while a worker
    // control acknowledgement crosses realms. That is not source invalidity:
    // retain the attached solver and let advanceTo() reject backward work.
    // Scene scalars are absent from the rebuild key, so they are adopted here
    // instead. A method without applySceneUniforms would otherwise ignore the
    // edit outright, so it falls back to the rebuild it used to take.
    const sceneUniformKey = gpuSceneUniformKey(scene);
    if (sceneUniformKey !== this.appliedSceneUniformKey) {
      if (this.gpuFluid.applySceneUniforms) {
        this.gpuFluid.applySceneUniforms(scene);
        this.appliedSceneUniformKey = sceneUniformKey;
      } else if (this.appliedSceneUniformKey) {
        const rebuildKey = `${key}:${sceneUniformKey}`;
        if (this.gpuFluidPendingKey !== rebuildKey) this.beginGPUFluidInitialization(scene, config, rebuildKey, presentationMode);
        return undefined;
      } else this.appliedSceneUniformKey = sceneUniformKey;
    }
    this.gpuFluid.applyRuntimeValues?.(config.values);
    this.secondaryParticlePipeline?.setSource(this.gpuFluid.secondaryParticles);
    return this.gpuFluid;
  }

  private settleInitialRasterPresentation(
    pending: PendingInitialRasterPresentation,
    diagnosticsRequired: boolean,
    diagnostics: WaterRenderDiagnostics | undefined,
  ) {
    if (this.disposed || this.deviceLost || this.pendingInitialRasterPresentation !== pending
      || this.gpuFluid !== pending.solver || this.gpuFluidGeneration !== pending.solverGeneration
      || this.gpuFluidRequestGeneration !== pending.requestGeneration) return;
    const sparsePresentation = this.sparseWorldPresentation(pending.solver);
    const outcome = diagnosticsRequired && !diagnostics
      ? { ready: false, state: "failed-closed" as const,
          label: "t=0 raster publication failed closed: bounded diagnostics readback was unavailable" }
      : initialRasterPresentationReadiness({
          solverAttached: true,
          initialSparseAuthorityReady: this.sparseAuthorityReady(pending.solver),
          globalFineAttached: Boolean(sparsePresentation?.fineLevelSet
            || pending.solver.globalFineLevelSetSource
            || pending.solver.coarseLevelSetSource),
          surfaceSourceAttached: this.globalFineWaterAttached,
          surfaceExtractionSubmitted: pending.submitted,
          presentationFenceCompleted: true,
          diagnosticsRequired,
          diagnostics,
        });
    if (!outcome.ready && outcome.state !== "failed-closed") return;
    pending.solver.info.initialRasterSurfaceReady = outcome.ready;
    pending.solver.info.initialRasterSurfaceState = outcome.state;
    pending.solver.info.initialRasterSurfaceDiagnostic = outcome.label;
    this.pendingInitialRasterPresentation = undefined;
    const sparseWorldState = this.refreshSparseWorldState(pending.solver);
    this.gpuInfoCallback?.(pending.solver.info);
    this.pausedPresentationRevision += 1;
    if (pending.solver.sparseWorld) {
      const sparseFault = sparseWorldState?.status.fault ?? sparseWorldState?.deviceFault;
      if (!outcome.ready || sparseWorldState?.status.state === "fault"
        || sparseWorldState?.deviceStatus === "fault") {
        this.onStatus({ state: "blocked", label: `Sparse world fault · ${sparseFault?.code ?? "presentation"}`, resource: pending.resource });
      } else if (sparseWorldState?.deviceStatus !== "ready") {
        this.onStatus({ state: "initializing", label: "Loading sparse world", resource: pending.resource });
      } else this.onStatus({ state: "ready", label: sparseWorldState.status.state === "saturated"
        ? "Sparse world capacity reached" : "Sparse world ready", adapter: this.adapterName, resource: pending.resource });
    } else if (outcome.ready) this.onStatus({ state: "ready", label: outcome.label, adapter: this.adapterName, resource: pending.resource });
    else this.onStatus({ state: "blocked", label: outcome.label, resource: pending.resource });
  }

  private settleLiveSvoPresentation(pending: PendingLiveSvoPresentation) {
    if (this.disposed || this.deviceLost || this.pendingLiveSvoPresentation !== pending
      || this.gpuFluid !== pending.solver || this.gpuFluidGeneration !== pending.solverGeneration
      || this.gpuFluidRequestGeneration !== pending.requestGeneration || !pending.attached || !pending.submitted) return;
    this.pendingLiveSvoPresentation = undefined;
    this.pausedPresentationRevision += 1;
    this.onStatus({ state: "ready", label: "Live SVO renderer ready", adapter: this.adapterName, resource: svoPresentationResourcePlugin });
  }

  private retireGPUAdvances(completions: readonly PendingGPUAdvanceCompletion[]) {
    for (const completion of completions) {
      const { solver: fluid, solverGeneration: generation, submittedTime_s: submittedTime } = completion;
      if (this.disposed || this.deviceLost || this.gpuFluid !== fluid || this.gpuFluidGeneration !== generation) continue;
      this.gpuPendingBatches = Math.max(0, this.gpuPendingBatches - 1);
      fluid.info.completedTime_s = Math.max(fluid.info.completedTime_s ?? 0, submittedTime);
      fluid.info.gpuPendingBatches = this.gpuPendingBatches;
      fluid.info.gpuInFlightSimulation_s = Math.max(0,
        (fluid.info.submittedTime_s ?? submittedTime) - fluid.info.completedTime_s);
      this.gpuInfoCallback?.({ ...fluid.info });
      this.gpuAdvanceCompletedCallback?.(submittedTime);
    }
  }

  private submitPreparedGPUFluid(fluid: GPUSolverInstance, time_s: number, bodies: RigidBodyState[], maximumPendingAdvances = 1, resource?: ResourcePluginDefinition) {
    if (!this.device) return fluid.info;
    const sparseWorldState = this.refreshSparseWorldState(fluid, true);
    if (sparseWorldState?.status.state === "fault"
      || sparseWorldState?.deviceStatus === "fault") return fluid.info;
    const sparseWorldLoading = Boolean(fluid.sparseWorld
      && sparseWorldState?.deviceStatus !== "ready");
    if (sparseWorldLoading || (!fluid.sparseWorld && fluid.simulationReady === false)) {
      if (!fluid.sparseWorld) fluid.info.simulationPipelinesReady = false;
      const state = sparseWorldLoading ? "sparse-world-loading" : fluid.info.simulationPipelineError
        ? `failed:${fluid.info.simulationPipelineError}`
        : "compiling";
      if (state !== this.reportedSimulationPipelineState) {
        this.reportedSimulationPipelineState = state;
        this.gpuInfoCallback?.({ ...fluid.info });
      }
      return fluid.info;
    }
    if (!fluid.sparseWorld && fluid.info.simulationPipelinesReady === false) {
      fluid.info.simulationPipelinesReady = true;
      this.reportedSimulationPipelineState = "ready";
      this.gpuInfoCallback?.({ ...fluid.info });
    } else if (fluid.sparseWorld && this.reportedSimulationPipelineState !== "ready") {
      this.reportedSimulationPipelineState = "ready";
      this.gpuInfoCallback?.({ ...fluid.info });
      if (resource) this.onStatus({ state: "ready", label:
        sparseWorldState?.status.state === "saturated" ? "Sparse world capacity reached"
          : "Sparse world ready", adapter: this.adapterName, resource });
    }
    // The presentation that follows carries whatever state these advances end
    // on, so no later simulation work can overtake the frame that visualizes
    // them. The queue-depth ceiling is expressed in FRAMES, so it scales with
    // the advances a frame now carries; otherwise the second advance of every
    // frame would be refused by a ceiling meant to bound presentation latency.
    for (let advance = 0; advance < GPU_ADVANCES_PER_PRESENTATION; advance += 1) {
      if (!canQueuePreparedGPUAdvance(this.gpuPendingBatches,
        maximumPendingAdvances * GPU_ADVANCES_PER_PRESENTATION)) break;
      const { previousSubmittedTime, submittedTime } = submitNextPreparedGPUAdvance(fluid, time_s, bodies);
      // The target clock owed no further whole step; stop rather than spin.
      if (submittedTime <= previousSubmittedTime) break;
      const generation = this.gpuFluidGeneration;
      this.gpuPendingBatches += 1;
      fluid.info.gpuPendingBatches = this.gpuPendingBatches;
      fluid.info.gpuInFlightSimulation_s = Math.max(0, submittedTime - (fluid.info.completedTime_s ?? 0));
      this.pendingGPUAdvanceCompletions.push({
        solver: fluid, solverGeneration: generation, submittedTime_s: submittedTime,
      });
    }
    this.refreshSparseWorldState(fluid, true);
    return fluid.info;
  }

  resize(renderScale = 1): void {
    if (this.disposed || this.deviceLost) return;
    const htmlCanvas = "clientWidth" in this.canvas ? this.canvas : undefined;
    const ratio = this.workerViewport?.devicePixelRatio
      ?? Math.min(globalThis.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor((this.workerViewport?.width ?? htmlCanvas?.clientWidth ?? this.canvas.width) * ratio));
    const height = Math.max(1, Math.floor((this.workerViewport?.height ?? htmlCanvas?.clientHeight ?? this.canvas.height) * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (!this.device || !this.format || !this.upscalePipeline || !this.upscaleSampler) return;
    this.activeRenderScale = renderScale;
    const renderWidth = Math.max(1, Math.floor(width * renderScale));
    const renderHeight = Math.max(1, Math.floor(height * renderScale));
    const key = `${renderWidth}x${renderHeight}`;
    if (key === this.presentationTextureKey) return;
    this.presentationTexture?.destroy();
    this.presentationTexture = this.device.createTexture({label:"Water presentation target",size:[renderWidth,renderHeight],format:this.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
    this.upscaleBindGroup=this.device.createBindGroup({layout:this.upscalePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this.presentationTexture.createView()},{binding:1,resource:this.upscaleSampler}]});
    this.presentationTextureKey=key;
    this.waterPipeline?.ensureSize(renderWidth, renderHeight);
    this.svoDryScenePipeline?.ensureSize(renderWidth, renderHeight);
  }

  get presentationResolution(): string {
    if (!this.presentationTexture) return `${this.canvas.width} × ${this.canvas.height}`;
    return `${this.presentationTexture.width} × ${this.presentationTexture.height} (${Math.round(this.activeRenderScale * 100)}%)`;
  }

  /** Frames whose presentation submission has actually finished on the GPU. */
  get completedPresentationCount(): number { return this.completedPresentations; }

  /** The described object the cursor is over, or undefined to clear the rim. */
  setHoverHighlight(range: { readonly first: number; readonly last: number } | undefined): void {
    if (range?.first === this.hoverHighlight?.first && range?.last === this.hoverHighlight?.last) return;
    this.hoverHighlight = range && range.last >= range.first ? range : undefined;
  }

  draw(time_s: number, scene: SceneDescription, camera: CameraState, bodies: RigidBodyState[], selectedBodyId: string | undefined, config: SimulationRunConfig, gridOverlay?: GridOverlayConfig, environmentId: EnvironmentId = defaultEnvironmentId, presentationMode: ScenePresentationMode = "full-scene", svoLightingOptions: SvoLightingOptions = DEFAULT_SVO_LIGHTING_OPTIONS, svoDiagnostics: SvoRenderDiagnostics = DEFAULT_SVO_RENDER_DIAGNOSTICS, svoTuning: SvoRenderTuning = DEFAULT_SVO_RENDER_TUNING, pixelTrace?: PixelTraceConfig, fluidCellTrace?: FluidCellTraceConfig): RendererFrameMetrics {
    const measurementInstrumentationEnabled = usePerformanceInstrumentationStore.getState().enabled;
    const cpuTrace = measurementInstrumentationEnabled
      ? new CPUPerformanceTrace(
        ++this.cpuTraceSampleId,
        `${config.methodId}:${config.quality}`,
        { id: "frame-control", label: "Frame control + physics submission" },
      )
      : undefined;
    if (!this.device || this.disposed || this.deviceLost || !this.context || !this.uniformBuffer || !this.bodyBuffer || !this.waterPipeline) return this.currentFrameMetrics(config.methodId, config.methodId, false, cpuTrace?.finish());
    const activeSvoTuning = normalizeSvoRenderTuning(svoTuning);
    this.resize(activeSvoTuning.resolutionScale);
    if (!this.presentationTexture || !this.upscalePipeline || !this.upscaleBindGroup) return this.currentFrameMetrics(config.methodId, config.methodId, false, cpuTrace?.finish());
    const requestedSvoDiagnostics = normalizeSvoRenderDiagnostics(svoDiagnostics);
    const activeSvoDiagnostics = requestedSvoDiagnostics;
    const tuningKey = svoRenderTuningKey(activeSvoTuning);
    // The stage view is a presentation choice, not a render choice: it changes
    // which plane is displayed and nothing that produced one. It still keys the
    // trace so a captured partition is never labelled with the wrong view.
    const diagnosticsKey = `${activeSvoDiagnostics.stageView}:${activeSvoDiagnostics.lightSlot}:${activeSvoDiagnostics.maximumTraversalDepth}:${activeSvoDiagnostics.maximumNodeVisits}:${tuningKey}`;
    if (diagnosticsKey !== this.svoRenderDiagnosticsKey) {
      this.svoRenderDiagnosticsKey = diagnosticsKey;
      this.resetPresentationTrace();
    }
    // Ablation belongs in the trace context, not just in the encode. Averaging
    // is per context, so without it the frames before and after a stage was
    // withheld would pool into one mean and the panel would report the cost of
    // neither pipeline.
    const disabledStages = disabledRenderStagesFrom(svoLightingOptions.disabledStages);
    const presentationContext = `${config.methodId}:${config.quality}:${presentationMode}:shadow-${svoLightingOptions.shadowsEnabled ? "on" : "off"}:ao-${svoLightingOptions.ambientOcclusionEnabled ? "on" : "off"}:cones-${svoLightingOptions.coneTracingMode ?? "cones"}:primary-${svoLightingOptions.primaryTraversal ?? "raster"}:tuning-${tuningKey}:without-${disabledRenderStagesKey(disabledStages) || "nothing"}:${this.simulationRunning ? "running" : "paused"}`;
    if (presentationContext !== this.presentationContext) {
      this.presentationContext = presentationContext;
      this.resetPresentationTrace();
    }
    const basis = cameraBasis(camera), position = basis.position;
    // The owner map is allocated lazily and materialized only once something
    // asks for it. The cell picker reads that same map, so gating the request
    // on a grid overlay left the picker resolving every pixel against a
    // zero-filled texture: row 0 everywhere, whose header decodes as a 1³ leaf
    // at the origin with no couplings. Asking for it whenever the picker is on
    // is what makes a pick land on the cell under the pointer.
    if (gridOverlay?.axis !== "off" || fluidCellTrace) {
      this.gpuFluid?.ensureGridDiagnosticTextures?.();
    }
    const sceneRuntime = planSceneRuntime(scene);
    const sparsePresentationRequired = presentationMode === "full-scene";
    // Read off the document, never off the tuning. The set this tree voxelizes
    // was expanded against `voxelDomain.detailCellSize_m`, so that is the only
    // number entitled to say how many levels the tree may spend under it — see
    // `svoSceneryRefinementDepth`. Hoisted because the primary-traversal
    // decision needs it too, for the frames before the brick count exists.
    const environmentRefinementDepth = svoEnvironmentTreeRefinementDepth(scene.voxelDomain, {
      fluid: sceneRuntime.fluidSolver,
    });
    const sceneConfig: SimulationRunConfig = sparsePresentationRequired ? {
      ...config,
      values: {
        ...config.values,
        svoEnvironmentBrickRefinementLevels: activeSvoTuning.environmentBrickRefinementLevels,
        svoEnvironmentRefinementDepth: environmentRefinementDepth,
        // Topology, so it belongs in the structural key alongside the depth:
        // toggling the exemption changes which nodes are leaves, and only a
        // rebuilt world can answer that.
        svoEnvironmentPlanarRefinementExemption: activeSvoTuning.environmentPlanarRefinementExemption,
      },
    } : config;
    const gpuSceneSourceRequired = sceneRuntime.fluidSolver || sparsePresentationRequired;
    const readyGPUFluid = gpuSceneSourceRequired
      ? this.currentGPUFluid(this.simulationScene ?? scene, sceneConfig, presentationMode)
      : undefined;
    const fluidSource = readyGPUFluid ?? this.gpuFluid;
    const sparseSceneProducer = sparsePresentationRequired ? this.sparseSceneProducer(fluidSource) : undefined;
    // Full scenes publish their room/terrain shell. Fluid-only scenes do not
    // even build the publication inputs: their raster water consumes a retained
    // clear attachment instead.
    if (sparsePresentationRequired) this.publishRenderScene(scene, sparseSceneProducer);
    const pixelTraceRequested = sparsePresentationRequired && Boolean(pixelTrace);
    // Ahead of the sweep, so a toggled traversal retires the old pipeline and is
    // rebuilt in the same frame rather than one frame later.
    //
    // `capacities.leaves` is the planned leaf set plus the bounded topology
    // mutation reserve (`webgpu-octree-sparse-bricks.ts:1737`), so it is the
    // proxy count the raster primary would emit, known on the CPU without a
    // readback. The presentation texture is the primary's own target and is
    // guaranteed non-undefined this far into the frame (see the early return
    // above), so the ratio is the live term and the document's refinement depth
    // only decides the frames before the world has published.
    if (sparsePresentationRequired) {
      this.applyPrimaryTraversalRequest(svoLightingOptions.primaryTraversal ?? "raster", {
        leafBricks: this.svoDrySceneSource?.structural?.capacities.leaves,
        targetPixels: this.presentationTexture.width * this.presentationTexture.height,
        environmentRefinementDepth,
      });
    }
    this.ensureRequestedOptionalPipelines(optionalRendererPipelineRequests(
      gridOverlay, this.simulationRunning,
      Boolean((readyGPUFluid ?? this.gpuFluid)?.secondaryParticles),
      pixelTraceRequested, Boolean(fluidCellTrace),
      sparsePresentationRequired && activeSvoDiagnostics.stageView !== "off",
      sparsePresentationRequired,
      sceneVesselPresentation(scene) === "outline",
    ), config.methodId);
    // The probe's answer depends on the scene epoch, the presentation, and the
    // traversal tuning as much as on the pixel. Tracking them as one revision is
    // what lets a pinned ray notice that its numbers describe a frame that has
    // since been replaced — another light, a republished topology, a new budget.
    const traceSceneKey = `${this.svoDryScenePipeline?.sceneEpoch ?? 0}|${presentationContext}|${diagnosticsKey}`;
    if (traceSceneKey !== this.pixelTraceSceneKey) {
      this.pixelTraceSceneKey = traceSceneKey;
      this.pixelTraceSceneRevisionValue += 1;
    }
    // A pinned ray stops probing, except when the caller asks for a refresh
    // because the scene moved on under it.
    const pixelTraceProbing = pixelTraceRequested && Boolean(pixelTrace)
      && (!pixelTrace!.pinned || pixelTrace!.refresh === true);
    if (pixelTraceProbing) {
      const width = this.presentationTexture.width, height = this.presentationTexture.height;
      this.svoDryScenePipeline?.requestPixelTrace(
        Math.floor(Math.max(0, Math.min(1, pixelTrace!.normalizedX)) * width),
        Math.floor(Math.max(0, Math.min(1, pixelTrace!.normalizedY)) * height),
      );
    }
    // The step along the ray applies to a pinned selection too: freezing the ray
    // is exactly when walking into the interior along it becomes useful.
    if (fluidCellTrace) this.fluidCellTracePipeline?.setHitIndex(fluidCellTrace.hitIndex ?? 0);
    // A pinned cell stops re-aiming, so orbiting cannot silently reselect it.
    // Freezing the pixel was not enough to do that: the gather still marched a
    // ray built from that pixel and the *current* camera, so panning swung the
    // ray and the pin slid onto whatever moved under it. The aim is recorded in
    // world space instead, from the same pixel centre the shader would have
    // used, and simply stops being updated once pinned.
    if (fluidCellTrace && !fluidCellTrace.pinned) {
      const width = this.presentationTexture.width, height = this.presentationTexture.height;
      const pixelX = Math.floor(Math.max(0, Math.min(1, fluidCellTrace.normalizedX)) * width);
      const pixelY = Math.floor(Math.max(0, Math.min(1, fluidCellTrace.normalizedY)) * height);
      this.fluidCellTracePipeline?.requestPixel(pixelX, pixelY);
      const aim = viewportRayForPixel(camera, pixelX, pixelY, width, height);
      this.fluidCellTracePipeline?.setAim(aim.origin, aim.direction);
    } else if (!fluidCellTrace) {
      // Leaving pick mode drops the frozen aim, so re-entering starts from the
      // pointer rather than answering the cell picked a scene ago.
      this.fluidCellTracePipeline?.clearAim();
      if (this.latestFluidCellTraceValue) {
        this.latestFluidCellTraceValue = undefined;
        this.fluidCellTraceRevisionValue += 1;
      }
    }
    if (!pixelTraceRequested) {
      this.svoDryScenePipeline?.clearPixelTraceRequest();
      if (this.latestPixelTraceValue) { this.latestPixelTraceValue = undefined; this.pixelTraceRevisionValue += 1; }
    }
    if (this.presentationsInFlight >= BROWSER_GPU_THROUGHPUT_DEPTH) {
      // The next draw uses the newest camera and solver state. Do not append a
      // stale presentation to a saturated FIFO: double buffering is sufficient
      // to keep execution dense without turning throughput into input latency.
      return this.currentFrameMetrics(config.methodId, presentationContext, false, cpuTrace?.finish());
    }
    let gpuInfo = readyGPUFluid?.info;
    if (readyGPUFluid) {
      gpuInfo = this.submitPreparedGPUFluid(
        readyGPUFluid, time_s, bodies,
        this.simulationRunning ? BROWSER_GPU_THROUGHPUT_DEPTH : 1,
        getMethod(config.methodId).resource,
      );
    }
    // The global fine narrow band double-buffers generations. Refresh its
    // tagged renderer binding after each admitted solver encode so extraction
    // follows the newly published generation without any CPU field copy.
    const sparseWorldPresentation = this.sparseWorldPresentation(readyGPUFluid);
    this.waterPipeline.setFluidDomain(readyGPUFluid?.fluidDomain);
    const globalFineLevelSet = sparseWorldPresentation?.fineLevelSet
      ?? readyGPUFluid?.globalFineLevelSetSource;
    if (globalFineLevelSet) {
      this.waterPipeline.setGlobalFineLevelSet(createGlobalFineLevelSetConsumerSource(globalFineLevelSet));
    } else {
      this.waterPipeline.setGlobalFineLevelSet(undefined);
    }
    this.waterPipeline.setCoarseLevelSet(readyGPUFluid?.coarseLevelSetSource);
    const globalFineWaterReady = Boolean(readyGPUFluid
      && this.sparseAuthorityReady(readyGPUFluid)
      && (globalFineLevelSet || readyGPUFluid.coarseLevelSetSource));
    const requestedSurface = readyGPUFluid
      ? globalFineWaterReady ? this.scalarFallbackTexture : readyGPUFluid.surfaceFieldTexture ?? readyGPUFluid.volumeTexture
      : undefined;
    if (readyGPUFluid && (globalFineWaterReady !== this.globalFineWaterAttached
      || requestedSurface !== this.attachedSurfaceTexture)) {
      this.globalFineWaterAttached = globalFineWaterReady;
      this.updateRenderSources(
        requestedSurface,
        readyGPUFluid.columnBaseTexture,
        readyGPUFluid.gridCellTexture ?? this.gridCellTexture,
        readyGPUFluid.velocityTexture ?? this.velocityFallbackTexture,
        readyGPUFluid.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture,
        readyGPUFluid.gridDivergenceTexture ?? this.scalarFallbackTexture,
        readyGPUFluid.gridPressureTexture ?? this.scalarFallbackTexture,
        readyGPUFluid.volumeTexture,
      );
    }
    if (gpuInfo && this.gpuFluid && this.columnBaseTexture && this.gridCellTexture && this.velocityFallbackTexture && this.pressureSamplesFallbackTexture && this.scalarFallbackTexture) {const activeSparsePresentation=this.sparseWorldPresentation(this.gpuFluid);const compactSurface=Boolean(activeSparsePresentation?.fineLevelSet||this.gpuFluid.globalFineLevelSetSource||this.gpuFluid.coarseLevelSetSource);this.gridOverlayPipeline?.setVolume(compactSurface?this.scalarFallbackTexture:this.gpuFluid.surfaceFieldTexture??this.gpuFluid.volumeTexture, this.gpuFluid.columnBaseTexture ?? this.columnBaseTexture, this.gpuFluid.gridCellTexture ?? this.gridCellTexture, this.gpuFluid.velocityTexture ?? this.velocityFallbackTexture, this.gpuFluid.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture, this.gpuFluid.gridDivergenceTexture ?? this.scalarFallbackTexture, this.gpuFluid.gridPressureTexture ?? this.scalarFallbackTexture, this.gpuFluid.volumeTexture);this.gridOverlayPipeline?.setSparseSource(activeSparsePresentation?.adaptiveGrid??this.gpuFluid.sparseAdaptiveGridSource);}
    cpuTrace?.transition({ id: "scene-upload", label: "Scene and field uploads" });
    const cameraStabilityKey = [
      basis.position.x, basis.position.y, basis.position.z,
      basis.forward.x, basis.forward.y, basis.forward.z,
      basis.right.x, basis.right.y, basis.right.z,
      basis.up.x, basis.up.y, basis.up.z,
    ].join("|");
    const cameraChanging = cameraStabilityKey !== this.svoCameraStabilityKey;
    this.svoCameraStabilityKey = cameraStabilityKey;
    const presentationCoherenceKey = [
      this.gpuFluidGeneration, scene.sceneId, scene.randomSeed, environmentId, diagnosticsKey, selectedBodyId ?? "",
      // The rim is part of what a presented frame looks like, so moving the
      // cursor onto an object has to invalidate a reused one.
      this.hoverHighlight ? `${this.hoverHighlight.first}:${this.hoverHighlight.last}` : "",
      cameraStabilityKey,
      ...bodies.flatMap((body) => [
        body.description.id, body.description.shape,
        body.description.dimensions_m.x, body.description.dimensions_m.y, body.description.dimensions_m.z,
        body.position_m.x, body.position_m.y, body.position_m.z,
        body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z,
      ]),
    ].join("|");
    const techniqueModeCode = gridOverlay?.mode && isOctreeTechniqueOverlayMode(gridOverlay.mode)
      ? OCTREE_TECHNIQUE_OVERLAY_CODES[gridOverlay.mode]
      : 0;
    // Keep the diagnostic selected even when the active solver has not
    // published CMD1. The grid overlay binds its one-word dummy storage in
    // that case; the version/extent checks deliberately render UNKNOWN
    // magenta instead of silently falling back to Grid structure.
    const dirtyModeCode = sparseCM12DirtyOverlayCode(gridOverlay?.mode);
    // The compact pressure solve ping-pongs its row buffer. Refresh the
    // diagnostic bundle only while a technique view is visible so pressure
    // updates never keep a bind group pointed at the preceding solve bank.
    if (techniqueModeCode) {
      this.techniqueOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    }
    // The view drives the marker advection, because the view is the only thing
    // that knows whether anyone is looking. Off is free rather than cheap: the
    // solver encodes no marker dispatch at all, and the off-to-on transition
    // re-seeds so switching the view on colours the liquid as it is now.
    const tracersVisible = gridOverlay?.axis !== "off" && gridOverlay?.mode === "tracers";
    const sparseUI = this.gpuFluid?.sparseWorldUI;
    if (sparseUI) sparseUI.control.tracers?.setEnabled(tracersVisible);
    else this.gpuFluid?.setTracersEnabled?.(tracersVisible);
    if (tracersVisible) this.tracerOverlayPipeline?.setSource(
      sparseUI?.overlays.tracers ?? this.gpuFluid?.tracerSource);
    // Face arrows need no counterpart to setTracersEnabled: they read numbers
    // the solve already produced, so an unwatched frame is charged nothing.
    // The source is re-read every frame because its bank swaps with parity.
    const faceVelocityVisible = gridOverlay?.axis !== "off"
      && gridOverlay?.mode === "face-velocity";
    if (faceVelocityVisible) {
      this.faceVelocityOverlayPipeline?.setSource(
        sparseUI?.overlays.faceVelocity ?? this.gpuFluid?.faceVelocitySource);
    }
    // The film follows the marker rule rather than the face-arrow one, because
    // unlike face velocities the numbers it draws do not otherwise exist: an
    // unarmed advance encodes no snapshot dispatch at all. So the view arms the
    // capture, and a session that never opens it never pays for one.
    const journalMode = isPressureJournalOverlayMode(gridOverlay?.mode)
      ? gridOverlay.mode : undefined;
    const journalVisible = gridOverlay?.axis !== "off" && journalMode !== undefined;
    if (sparseUI) sparseUI.control.pressureFilm?.setCaptureEnabled(journalVisible);
    else this.gpuFluid?.armPressureJournal?.(journalVisible);
    if (journalVisible) {
      this.pressureJournalOverlayPipeline?.setSource(
        sparseUI?.overlays.pressureFilm ?? this.gpuFluid?.pressureJournalSource);
    }
    // A lens is the film's bargain again — its taps encode nothing until one is
    // armed — but the arming belongs to `select`. The overlay is the only thing
    // that knows which lens is current, so a second caller arming the source
    // behind it would leave the two disagreeing about whose snapshots are in
    // the buffers. The source is offered every frame because a rebuilt solver
    // publishes a new one and the overlay ignores a repeat.
    const lensMode = gridOverlay?.mode && isStageLensOverlayMode(gridOverlay.mode)
      ? gridOverlay.mode : undefined;
    const lensVisible = gridOverlay?.axis !== "off" && lensMode !== undefined;
    this.stageLensOverlayPipeline?.setSource(
      sparseUI?.overlays.stageLenses ?? this.gpuFluid?.stageLensSource);
    this.stageLensOverlayPipeline?.select(lensVisible ? lensMode : undefined);
    const uniform = new Float32Array([
      this.presentationTexture.width, this.presentationTexture.height, time_s, cameraChanging ? SVO_CAMERA_CHANGING_FRAME : -1,
      // cameraPosition.w is the aperture, tan(fov/2). It was padding until the
      // lens became a scene property; see CAMERA_APERTURE_UNIFORM_LANE for why
      // it rides here rather than in a field of its own.
      position.x, position.y, position.z, cameraTanHalfFov(camera),
      // cameraTarget.w is padding. Geometry comes from SolidWorld rather than
      // an analytic container-shape/top mode encoded in the camera uniform.
      camera.target_m.x, camera.target_m.y, camera.target_m.z, 0,
      scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction,
      // options.w carries the largest represented adaptive pressure-cell
      // width. The grid overlay uses it to normalize its categorical scale
      // palette to the hierarchy that can actually exist in this solver.
      activeSvoDiagnostics.maximumTraversalDepth * 512 + activeSvoDiagnostics.maximumNodeVisits, scene.voxelDomain.finestCellSize_m, Math.min(bodies.length, 12), gpuInfo?.quadtreeMaximumFluidScale ?? 1,
      // Field mode: 1 = raw occupancy, 3 = uniform-layout level set.
      gpuInfo?.nx ?? 1, gpuInfo?.ny ?? 1, gpuInfo?.nz ?? 1, gpuInfo ? (gpuInfo.gridKind === "octree" ? 3 : 1) : 0,
      gridOverlay?.axis === "z" ? 1 : gridOverlay?.axis === "x" ? 2 : gridOverlay?.axis === "y" ? 3 : gridOverlay?.axis === "volume" ? 4 : 0, gridOverlay?.position ?? 0.5, gpuInfo?.gridKind === "octree" ? 1 : 0,
      techniqueModeCode || dirtyModeCode || (gridOverlay?.mode === "cfl" ? 1 : gridOverlay?.mode === "speed" ? 2 : gridOverlay?.mode === "phi" ? 3 : gridOverlay?.mode === "divergence" ? 4 : gridOverlay?.mode === "pressure" ? 5 : gridOverlay?.mode === "representation" ? 6 : gridOverlay?.mode === "optical" ? 7 : gridOverlay?.mode === "projection" && gpuInfo?.gridKind === "octree" ? 8 : gridOverlay?.mode === "resolution" && gpuInfo?.gridKind === "octree" ? 9 : gridOverlay?.mode === "density" ? 10 : 0),
      environmentIndex(environmentId), gpuInfo?.lastDt_s ?? 0, gpuInfo?.maxSpeed_m_s ?? 0,
      0
    ]);
    const packed = new Float32Array(104);
    packed.set(uniform, 0);
    // Lanes 100..103 are the hover rim: the owner range under the cursor, its
    // strength, and the falloff exponent. Appended at the end so every other
    // shader's view of this buffer is byte-identical.
    if (this.hoverHighlight) packed.set([this.hoverHighlight.first, this.hoverHighlight.last, .45, 2.6], 100);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, packed);
    const residentRigidBuffer = this.gpuFluid?.rigidRenderBuffer;
    if (residentRigidBuffer) {
      this.gpuFluid?.setSelectedRigidBody?.(bodies.findIndex((body) => body.description.id === selectedBodyId));
    } else {
      const bodyData = new Float32Array(12 * 16);
      bodies.slice(0, 12).forEach((body, index) => {
        const offset = index * 16;
        const shape = body.description.shape;
        const half = sceneShapeRenderHalfExtent_m(shape, body.description.dimensions_m);
        const color = SCENE_SHAPE_PALETTE_LINEAR[sceneShapeCode(shape)];
        bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, boundingRadius(body)], offset);
        bodyData.set([half[0], half[1], half[2], sceneShapeCode(shape)], offset + 4);
        bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
        bodyData.set([color[0], color[1], color[2], body.description.id === selectedBodyId ? 1 : 0], offset + 12);
      });
      this.device.queue.writeBuffer(this.bodyBuffer, 0, bodyData);
    }
    if (sparsePresentationRequired) this.advanceLiveSceneAnimation();
    if (sparsePresentationRequired) this.svoDryScenePipeline?.setRigidBodyCount(bodies.length, svoDryRigidBounds(bodies));
    // Reduced-rate cone lighting is the production default: quarter-axis-rate
    // prepass + full-resolution relight, with 0.5 retained by the quality tier.
    if (sparsePresentationRequired) {
      this.svoDryScenePipeline?.setLightingOptions({ ...svoLightingOptions, coneLightingScale: activeSvoTuning.coneLightingScale });
      this.svoDryScenePipeline?.setRenderTuning(activeSvoTuning);
    }
    // Its own channel, taken every frame: a withheld stage is an encode-time
    // decision and must never reach the code that rebuilds shaders or bundles.
    this.svoDryScenePipeline?.setDisabledStages(disabledStages);
    this.waterPipeline.setDisabledStages(disabledStages);
    cpuTrace?.transition({ id: "command-encoding", label: "Presentation command encoding" });
    const traceRequestedAt_ms = measurementInstrumentationEnabled ? performance.now() : 0;
    // One frame in sixteen while instrumentation is on, the encode is split at
    // band boundaries into fence-partitioned submits (WP3.1). That frame runs
    // neither the pass-timestamp recorder (its proxy would span dead encoders)
    // nor the queue-wall recorder — a partitioned frame prices its bands, not
    // the frame, and publishing its wall would fold submit-boundary overhead
    // into the rolling frame mean.
    const bandSamplingFrame = measurementInstrumentationEnabled
      && sparsePresentationRequired
      && !this.presentationBandSamplePending
      && ++this.presentationBandFrameCounter % 16 === 0;
    const shouldTracePresentation = measurementInstrumentationEnabled
      && !this.presentationTracePending
      && !bandSamplingFrame;
    const presentationTraceSampleId = shouldTracePresentation
      ? ++this.presentationTraceSampleId
      : 0;
    const presentationTrace = shouldTracePresentation
      && GPUPassTimestampRecorder.supported(this.device)
      ? new GPUPassTimestampRecorder(this.device, 512, "presentation pass timestamps")
      : undefined;
    if (shouldTracePresentation && !presentationTrace && !this.reportedMissingPresentationTimestamps) {
      this.reportedMissingPresentationTimestamps = true;
      console.warn("GPU stage timestamps are unavailable: device lacks timestamp-query");
    }
    const presentationQueueTrace = shouldTracePresentation
      ? new GPUQueueWallPerformanceTraceRecorder(
        presentationTraceSampleId,
        "presentation",
        presentationContext,
      )
      : undefined;
    // Boundaries ride the presentation's own passes, so the traced frame and
    // the untraced frame submit the same command graph.
    const rawEncoder = this.device.createCommandEncoder({ label: "Fluid Lab frame" });
    // Two questions, two instruments over the same encoder. The timestamp
    // recorder answers "how long did these passes take"; the seam recorder
    // answers "were there any" — the question that separates a stage which
    // spent nothing from a stage whose render passes cannot be timed, and
    // therefore the question that decides which row a band's wall residual may
    // land on. It is a pair of integer increments per pass and exists only
    // while the panel is recording.
    const seamRecorder = measurementInstrumentationEnabled ? new RenderFrameSeamRecorder() : undefined;
    const instrumentEncoder = (target: GPUCommandEncoder) =>
      seamRecorder?.instrument(presentationTrace?.instrument(target) ?? target)
        ?? presentationTrace?.instrument(target) ?? target;
    let encoder = instrumentEncoder(rawEncoder);
    const bandSampler = bandSamplingFrame
      ? new FencePartitionedFrameSampler(
        this.device, encoder, ++this.presentationTraceSampleId,
        `${presentationContext}:band-wall`,
        // A sampling frame finishes one encoder per band and opens the next.
        // Uninstrumented successors would leave every stage after the first
        // boundary reporting zero passes, which is the exact false zero this
        // recorder exists to prevent.
        seamRecorder ? (target) => seamRecorder.instrument(target) : undefined)
      : undefined;
    if (bandSampler) this.presentationBandSamplePending = true;
    // Every path names its own stages now. The six-phase partition that used to
    // stand in for the non-SVO frame named stages *by position*, so a frame
    // that skipped one silently relabelled every stage after it; the water
    // pipeline's own seams are a strictly finer and strictly honest partition
    // of the same encode, and they close on both arms.
    const detailedPresentationTrace = presentationTrace;
    /**
     * Close one stage of the frame.
     *
     * Two ledgers, one call: the timestamp recorder closes a semantic group
     * over the passes since the last seam, and the seam recorder counts them.
     * The phase's label is looked up from the stage id rather than written
     * here, which is what makes the panel's row and this seam the same fact.
     */
    const closeStage = (stage: RenderFrameStageId) => {
      seamRecorder?.close(stage);
      detailedPresentationTrace?.completePhase(RENDER_FRAME_STAGE_TRACE[stage].phase);
    };
    // Incremental voxelization: the only work in the frame that changes the
    // structure everything below marches. Withholding it freezes the world at
    // whatever was already built rather than emptying it, which is why this is
    // the one ablation whose image stays correct for a stationary scene.
    if (sparsePresentationRequired && !disabledStages.has("sparse-world-build")) {
      // The producers close the four world stages themselves, from inside the
      // maintenance they encode: topology, voxelization, derived lighting and
      // the radiance feedback window are four different schedules, and one seam
      // over all of them could report none of them. Two producers can run, and
      // the second one's seams merge into the first's entries.
      fluidSource?.encodeSceneMaintenance?.(encoder, closeStage);
      if (sparseSceneProducer !== fluidSource) sparseSceneProducer?.encodeSceneMaintenance?.(encoder, closeStage);
    }
    if (residentRigidBuffer) encoder.copyBufferToBuffer(residentRigidBuffer, 0, this.bodyBuffer, 0, 12 * 16 * 4);
    // The same records, on their way to the host. See `publishRigidBodyPoses`.
    //
    // Issued while the simulation runs, which is the whole point: the solver
    // owns rigid motion from t=0 onward, so a gizmo fed only by paused frames
    // stays pinned where the body was authored while the body itself falls out
    // from under it. "Live frames never map solver state" is the rule for
    // `readStats`, which maps a buffer the solver owns; this maps a
    // renderer-owned staging copy, double-buffered and skipped whenever both
    // slots are still in flight, so it neither blocks the queue nor paces it.
    const poseStaging = residentRigidBuffer && bodies.length > 0
      ? this.encodeRigidBodyPoseReadback(encoder, residentRigidBuffer) : undefined;
    // Copies rather than passes, but they are commands between two seams, and
    // a seam that does not own them charges them to a neighbour.
    closeStage("rigid-pose-mirror");
    if (sparsePresentationRequired) {
      this.svoDryScenePipeline?.setRigidMotionSource(this.gpuFluid?.rigidMotionBuffer);
      this.svoDryScenePipeline?.setFluidCoverage(this.ensureFluidCoverage(readyGPUFluid, scene));
    } else {
      this.ensureFluidCoverage(undefined, scene);
    }
    this.secondaryParticlePipeline?.setSource(this.gpuFluid?.secondaryParticles);
    let svoEncoded = false;
    let requestedBundleStatus = this.svoDryScenePipeline?.presentationBundleStatus;
    const drySceneReplacement = (
      replacementEncoder: GPUCommandEncoder,
      target: GPUTexture | GPUTextureView,
    ) => {
      // Primary visibility is deliberately recomputed every frame. A paused
      // or stationary frame is still a real traversal measurement; no
      // coherence key is published to the dry-scene renderer.
      const replacementResult = this.svoDryScenePipeline?.encode(replacementEncoder, target, closeStage, bandSampler) ?? false;
      svoEncoded = Boolean(replacementResult);
      requestedBundleStatus = this.svoDryScenePipeline?.presentationBundleStatus;
      if (requestedBundleStatus?.state === "failed") {
        this.optionalPipelineFailures.set("svo-dry-scene", requestedBundleStatus.detail);
      } else if (replacementResult && this.svoDryScenePipeline) {
        this.optionalPipelineFailures.delete("svo-dry-scene");
      }
      return replacementResult;
    };
    this.waterPipeline.setSceneHasFluid(Boolean(sceneRuntime.fluidSolver));
    // Who draws the bodies. A full-scene presentation rasters them into the SVO
    // dry G-buffer; a fluid-only one runs no dry scene at all, so the water
    // pipeline authors the same attachment itself. Zero here is not "no bodies"
    // but "not yours" — the roster is unchanged, the renderer of it is not.
    this.waterPipeline.setRigidBodyCount(sparsePresentationRequired ? 0 : bodies.length);
    // The medium, the key and the caustic receiver are all document facts the
    // composite used to have inlined into its WGSL at build time. The pipeline
    // ignores a call that changes nothing, so this stays a per-frame statement
    // of what the scene is rather than a per-frame upload.
    // The shared rig belongs to the full-scene SVO presentation path, whether
    // or not fluid physics is also running. A fluid-only raster presentation
    // remains entitled to its authored lighting.
    const sceneLighting = sparsePresentationRequired ? svoSceneLighting(scene) : scene.lighting;
    const waterDirectional = sparsePresentationRequired
      ? waterKeyDirectionalFromSceneLights(
        buildSvoSceneLights(scene).records,
        [0, 0.5 * scene.container.height_m, 0],
      ) ?? sceneLighting?.directional
      : sceneLighting?.directional;
    this.waterPipeline.setSceneOptics({
      optics: scene.fluid.optics,
      directional: waterDirectional,
      grade: sceneLighting?.grade,
      terrain: scene.terrain,
      terrainContentStamp: this.renderSceneTerrainContentStamp,
      container: { width_m: scene.container.width_m, depth_m: scene.container.depth_m },
    });
    // Before the dry pass samples it, and outside the water pipeline's own
    // passes so the volume is complete for the whole frame. Withheld, the
    // volume freezes at its last fill — consumers keep reading the retained
    // texture — so the delta is the fill + mip chain and nothing downstream.
    const fluidCoverageEncoded = !disabledStages.has("fluid-coverage")
      && (this.svoFluidCoverage?.encode(encoder) ?? false);
    closeStage("fluid-coverage");
    const pendingInitialRaster = this.pendingInitialRasterPresentation;
    const initialRasterSourceReady = Boolean(pendingInitialRaster
      && !pendingInitialRaster.submitted
      && pendingInitialRaster.solver === readyGPUFluid
      && this.sparseAuthorityReady(readyGPUFluid)
      && (globalFineLevelSet || readyGPUFluid.coarseLevelSetSource)
      && this.globalFineWaterAttached);
    const surfaceDiagnosticsRequired = true;
    // Everything encoded so far — world maintenance, rigid copies, coverage —
    // is the source band; the water/SVO pipelines cross their own boundaries.
    if (bandSampler) encoder = bandSampler.boundary("source");
    const rasterResult = this.waterPipeline.encode(
      encoder, this.presentationTexture,
      gpuInfo?.nx ?? 1, gpuInfo?.ny ?? 1, gpuInfo?.nz ?? 1,
      false, gpuInfo?.maximumNeighborDelta ?? 0,
      gpuInfo?.encodedSteps ?? 0,
      sparsePresentationRequired ? drySceneReplacement : undefined,
      closeStage,
      surfaceDiagnosticsRequired && initialRasterSourceReady,
      sparsePresentationRequired ? "require-dry-scene" : "clear",
      !this.simulationRunning || initialRasterSourceReady,
      bandSampler,
      // A paused manual step gets one repaint. Its newly published surface
      // must therefore displace the retained mesh even if that repaint lands
      // inside the ordinary 60 Hz extraction cadence window.
      !this.simulationRunning,
    );
    // The pipelines may have crossed submit boundaries; everything below —
    // overlays, upscale, the final submit — must ride the live encoder.
    if (bandSampler) encoder = bandSampler.current;
    if (!rasterResult) throw new Error("Water optics pipeline is not ready");
    const initialRasterSubmission = pendingInitialRaster
      && initialRasterSourceReady
      && rasterResult.surfaceUpdated
      && (!surfaceDiagnosticsRequired || rasterResult.surfaceDiagnosticsCaptured)
      ? pendingInitialRaster
      : undefined;
    if (initialRasterSubmission) initialRasterSubmission.submitted = true;
    const pendingLiveSvo = this.pendingLiveSvoPresentation;
    const initialLiveSvoSubmission = pendingLiveSvo
      && !pendingLiveSvo.submitted
      && pendingLiveSvo.attached
      && pendingLiveSvo.solver === readyGPUFluid
      && pendingLiveSvo.source === this.svoDrySceneSource
      && svoEncoded
      ? pendingLiveSvo
      : undefined;
    if (initialLiveSvoSubmission?.submit()) {
      this.onStatus({
        state: "initializing", label: SVO_PRESENTATION_STARTUP_STAGES[9], phase: "presentation",
        completed: 9, total: SVO_PRESENTATION_STARTUP_STAGES.length, startedAt_ms: initialLiveSvoSubmission.startedAt_ms, kind: "startup", retainingPrevious: false,
        resource: svoPresentationResourcePlugin,
      });
    }
    this.svoPickingAvailable = svoEncoded;
    this.lastSvoPickingBodies = this.svoPickingAvailable ? bodies.slice(0, 12) : [];
    const silhouetteRefinementStatus = this.svoDryScenePipeline?.silhouetteRefinementStatus;
    const lightingVisibilityStatus = this.svoDryScenePipeline?.lightingVisibilityStatus;
    this.publishEffectiveRendererStatus(resolveEffectiveRendererStatus({
      required: sparsePresentationRequired,
      pipelineAvailable: this.svoPipelineAvailable,
      pipelineCompiling: this.optionalPipelineTasks.has("svo-dry-scene")
        || requestedBundleStatus?.state === "compiling"
        || silhouetteRefinementStatus?.state === "compiling",
      pipelineFailure: silhouetteRefinementStatus?.state === "failed"
        ? silhouetteRefinementStatus.detail
        : this.optionalPipelineFailures.get("svo-dry-scene"),
      pipelinePending: requestedBundleStatus?.state === "compiling"
        ? requestedBundleStatus.detail
        : silhouetteRefinementStatus?.state === "compiling" ? silhouetteRefinementStatus.detail : undefined,
      sourceAvailable: this.svoSourceAvailable,
      terrainSupported: true,
      glassSupported: this.svoGlassSupported,
      materialsSupported: this.svoMaterialsSupported,
      lightingSupported: this.svoLightingSupported,
      svoEncoded,
      contractFailure: this.svoPublicationFailure,
      silhouetteRefinement: silhouetteRefinementStatus,
      lightingVisibility: lightingVisibilityStatus,
      terminalCounts: this.svoDrySceneSource?.structural?.terminalCounts,
    }));
    let inspectionOverlayEncoded = false;
    // Render stage views replace the composited image with a decode of a plane
    // an earlier pass published. Encoded first among the inspection overlays so
    // structural overlays and the ray-trace decoration still draw over it, and
    // strictly read-only, so the frame it explains is the frame that shipped.
    const inspectionWithheld = disabledStages.has("inspection-overlays");
    if (activeSvoDiagnostics.stageView !== "off" && !inspectionWithheld && this.svoStageOverlay?.ready && svoEncoded) {
      const sceneExtent = Math.hypot(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
      inspectionOverlayEncoded = this.svoStageOverlay.encode(
        encoder, this.presentationTexture.createView(),
        activeSvoDiagnostics.stageView, activeSvoDiagnostics.lightSlot,
        this.presentationTexture.width, this.presentationTexture.height,
        camera.distance_m + sceneExtent,
        { ...(this.svoDryScenePipeline?.stagePlanes ?? {}), sceneRadiance: this.waterPipeline.drySceneRadianceView },
      ) || inspectionOverlayEncoded;
    }
    // The gather is a compute pass over published topology, so it is encoded
    // whether or not any overlay drew this frame — but it is inspection work,
    // and the inspection switch withholds it with the rest so the frame being
    // measured is not paying for its own measurement.
    // The readback is pumped after the submit, not here: `mapAsync` puts the
    // buffer into a pending map state immediately, and a buffer with a map
    // pending may not appear in a submit — including the very submit that
    // carries this copy. Mapping before submitting rejects the whole command
    // buffer, so the frame that was gathering the cell also fails to present.
    if (fluidCellTrace && !inspectionWithheld) {
      // Lazy allocation means the owner map may have arrived after the last
      // source refresh; `setSource` is a no-op once it stops changing.
      this.fluidCellTracePipeline?.setSource(
        this.gpuFluid?.octreeTechniqueDebugSource,
        this.gpuFluid?.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture);
      inspectionOverlayEncoded = (this.fluidCellTracePipeline?.encode(encoder) ?? false) || inspectionOverlayEncoded;
    }
    if (gridOverlay && gridOverlay.axis !== "off" && !inspectionWithheld) {
      const overlayView=this.presentationTexture.createView();
      // Generic texture fields and compact paper publications each own both
      // their slice and ray-integrated volume presentation.
      if(tracersVisible){
        this.tracerOverlayPipeline?.encode(encoder,overlayView,this.pixelTraceSceneDepthView(),{
          camera: {
            position_m: [basis.position.x, basis.position.y, basis.position.z],
            forward: [basis.forward.x, basis.forward.y, basis.forward.z],
            right: [basis.right.x, basis.right.y, basis.right.z],
            up: [basis.up.x, basis.up.y, basis.up.z],
            tanHalfFov: cameraTanHalfFov(camera),
            aspect: viewportAspect(this.presentationTexture.width, this.presentationTexture.height),
          },
          viewportWidth: this.presentationTexture.width,
          viewportHeight: this.presentationTexture.height,
          container_m: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
          depthNear_m: SVO_DRY_SCENE_REVERSED_Z_NEAR_M,
          // The view is planeless, so its slice control is a cloud opacity —
          // the same meaning the slider already carries in a volume view.
          globalAlpha: gridOverlay.position,
        });
      }
      else if(faceVelocityVisible){
        this.faceVelocityOverlayPipeline?.encode(encoder,overlayView,this.pixelTraceSceneDepthView(),{
          camera: {
            position_m: [basis.position.x, basis.position.y, basis.position.z],
            forward: [basis.forward.x, basis.forward.y, basis.forward.z],
            right: [basis.right.x, basis.right.y, basis.right.z],
            up: [basis.up.x, basis.up.y, basis.up.z],
            tanHalfFov: cameraTanHalfFov(camera),
            aspect: viewportAspect(this.presentationTexture.width, this.presentationTexture.height),
          },
          viewportWidth: this.presentationTexture.width,
          viewportHeight: this.presentationTexture.height,
          container_m: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
          depthNear_m: SVO_DRY_SCENE_REVERSED_Z_NEAR_M,
          globalAlpha: gridOverlay.position,
        });
      }
      else if(journalVisible&&journalMode){
        // `position` is the film's scrub rather than a slice depth: the view is
        // planeless, so the slider has no plane to place, and stepping through
        // the captured iterations is the one control the film cannot do without.
        const source=this.gpuFluid?.sparseWorldUI?.overlays.pressureFilm
          ??this.gpuFluid?.pressureJournalSource;
        const captured=Math.max(1,source?.snapshotCount??1);
        this.pressureJournalOverlayPipeline?.encode(encoder,overlayView,this.pixelTraceSceneDepthView(),{
          camera: {
            position_m: [basis.position.x, basis.position.y, basis.position.z],
            forward: [basis.forward.x, basis.forward.y, basis.forward.z],
            right: [basis.right.x, basis.right.y, basis.right.z],
            up: [basis.up.x, basis.up.y, basis.up.z],
            tanHalfFov: cameraTanHalfFov(camera),
            aspect: viewportAspect(this.presentationTexture.width, this.presentationTexture.height),
          },
          viewportWidth: this.presentationTexture.width,
          viewportHeight: this.presentationTexture.height,
          container_m: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
          depthNear_m: SVO_DRY_SCENE_REVERSED_Z_NEAR_M,
          snapshot: Math.round(Math.max(0,Math.min(1,gridOverlay.position))*(captured-1)),
          // The residual family is normalised on the device from the capture's
          // own seed record, so this reference is only the fallback; pressure is
          // the channel it actually sets.
          channel: pressureJournalOverlayChannel(journalMode, 1),
        });
      }
      else if(lensVisible&&lensMode){
        // The one inspection view whose slider keeps its plain meaning: a lens
        // is authored for the slice plane, so `position` places the plane
        // rather than dimming a cloud, and the scrub it needs instead is the
        // phase. No scene depth either — glyphs read over the frame, because a
        // row occluded by the water is the row a projection view is about.
        this.stageLensOverlayPipeline?.encode(encoder,overlayView,{
          camera: {
            position_m: [basis.position.x, basis.position.y, basis.position.z],
            forward: [basis.forward.x, basis.forward.y, basis.forward.z],
            right: [basis.right.x, basis.right.y, basis.right.z],
            up: [basis.up.x, basis.up.y, basis.up.z],
            tanHalfFov: cameraTanHalfFov(camera),
            aspect: viewportAspect(this.presentationTexture.width, this.presentationTexture.height),
          },
          viewportWidth: this.presentationTexture.width,
          viewportHeight: this.presentationTexture.height,
          container_m: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
          axis: gridOverlay.axis,
          position: gridOverlay.position,
          phase: gridOverlay.lensPhase ?? 0,
        });
      }
      else if(!techniqueModeCode)this.gridOverlayPipeline?.encode(encoder,overlayView);
      else{
        this.techniqueOverlayPipeline?.encode(encoder,overlayView,techniqueModeCode);
        this.techniqueAuditOverlayPipeline?.encode(encoder,overlayView,techniqueModeCode);
      }
      inspectionOverlayEncoded = true;
    }
    this.publishStageLensReceipt(lensVisible ? lensMode : undefined);
    if (pixelTraceRequested && pixelTrace && !inspectionWithheld) {
      // The probe re-traces the requested pixel against the topology this frame
      // just drew from, and the overlay draws the trace decoded from the last
      // readback. One frame of latency, no stall.
      //
      // A pinned trace stops probing altogether. Re-tracing the same pixel from
      // a moved camera would quietly answer a different ray, which is the exact
      // opposite of freezing one; the recorded world-space work is what the
      // overlay keeps drawing while the camera orbits around it. A refresh is the
      // one exception, and only the caller can tell that it is the same ray.
      if (pixelTraceProbing && this.svoDryScenePipeline?.encodePixelTrace(encoder)) {
        this.pixelTraceEncodedSceneRevision = this.pixelTraceSceneRevisionValue;
        inspectionOverlayEncoded = true;
      }
    }
    // One assembled draw for every decoration any pass contributed this frame.
    const vesselOutlineActive = sceneVesselPresentation(scene) === "outline";
    if (!inspectionWithheld || vesselOutlineActive) {
      inspectionOverlayEncoded = this.encodeDecorationOverlay(
        encoder, basis, cameraTanHalfFov(camera), scene,
        !inspectionWithheld && pixelTraceRequested ? pixelTrace : undefined,
        !inspectionWithheld ? fluidCellTrace : undefined)
        || inspectionOverlayEncoded;
    }
    // Closed after the probes and the decoration draw, not before them: the
    // seam is what attributes their passes, and closing early charged every
    // probe to the upscale row.
    // Closed whether or not anything drew: a stage that encoded nothing is a
    // zero the panel can state, where a missing seam is a silence some other
    // row ends up answering for.
    closeStage("inspection-overlays");
    // The swap-chain texture is acquired and cleared either way: a frame that
    // never wrote it would leave the compositor showing an older one, so a
    // withheld blit would read as "free" while the image stopped updating for a
    // reason nothing on screen could explain.
    const upscalePass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0.0194,g:0.0145,b:0.0097,a:1},loadOp:"clear",storeOp:"store"}]});
    if (!disabledStages.has("present")) {
      upscalePass.setPipeline(this.upscalePipeline);upscalePass.setBindGroup(0,this.upscaleBindGroup);upscalePass.draw(3);
    }
    upscalePass.end();
    closeStage("present");
    // What this frame encoded, stage by stage. Published every recorded frame,
    // including the fence-partitioned sampling frames the pass recorder skips —
    // it is the sampling frame's manifest that decides which of its bands' rows
    // may carry that band's wall.
    if (seamRecorder) this.latestPresentationManifest = seamRecorder.manifest();
    // Every stage names itself now, so there is no mislabelled partition to
    // guard against: a stage the frame skipped is simply absent, and the
    // manifest says it encoded nothing.
    const hardwarePresentationTraceResolved = presentationTrace?.resolve(encoder) ?? false;
    if (!hardwarePresentationTraceResolved) presentationTrace?.destroy();
    presentationQueueTrace?.begin();
    // Every admitted advance precedes this presentation in the same WebGPU
    // queue. One end-of-presentation completion therefore retires both without
    // an extra queue-wide promise between simulation and rendering.
    const completedGPUAdvances = this.pendingGPUAdvanceCompletions.splice(0);
    this.device.queue.submit([encoder.finish()]);
    this.presentationsInFlight+=1;
    const completedPresentationDevice=this.device;
    let presentationRetired=false;
    const retirePresentation=()=>{
      if(presentationRetired)return;
      presentationRetired=true;
      this.presentationsInFlight=Math.max(0,this.presentationsInFlight-1);
    };
    const presentationCompletion = completedPresentationDevice.queue.onSubmittedWorkDone();
    void presentationCompletion.then(()=>{
      retirePresentation();
      if(!this.disposed&&!this.deviceLost&&this.device===completedPresentationDevice){
        this.completedPresentations+=1;
        this.retireGPUAdvances(completedGPUAdvances);
      }
    }).catch(retirePresentation);
    if (poseStaging) this.publishRigidBodyPoses(poseStaging, bodies.slice(0, 12).map((body) => body.description.id));
    if (pixelTraceProbing) this.pumpPixelTraceReadback();
    if (fluidCellTrace) this.pumpFluidCellTraceReadback();
    if (bandSampler) {
      const sampledContext = presentationContext;
      void bandSampler.finish("composite-present", presentationCompletion).then((bandTrace) => {
        if (bandTrace && !this.disposed && !this.deviceLost
          && this.device === completedPresentationDevice && this.presentationContext === sampledContext) {
          this.latestPresentationBandTrace = bandTrace;
        }
      }).finally(() => {
        this.presentationBandSamplePending = false;
      });
    }
    const presentationQueueTraceRead = presentationQueueTrace?.read(this.device.queue);
    const presentationStageTraceRead = hardwarePresentationTraceResolved && presentationTrace
      ? presentationTrace.readSemanticTrace({
        sampleId: presentationTraceSampleId,
        lane: "presentation",
        context: presentationContext,
        capturedAt_ms: traceRequestedAt_ms,
      }).catch(() => undefined)
      : undefined;
    if (presentationQueueTraceRead) {
      this.presentationTracePending = true;
      const sampledContext = presentationContext;
      void Promise.all([presentationQueueTraceRead, presentationStageTraceRead]).then(([trace, stages]) => {
        const instrumentation = usePerformanceInstrumentationStore.getState();
        if (!trace || this.disposed || this.deviceLost || this.presentationContext !== sampledContext
          || !instrumentation.enabled || instrumentation.enabledAt_ms > traceRequestedAt_ms) return;
        this.latestPresentationTrace = trace;
        this.latestPresentationStageTrace = stages;
        if (!this.simulationRunning) this.pausedPresentationRevision += 1;
      }).catch(() => {
        presentationTrace?.destroy();
      }).finally(() => {
        this.presentationTracePending = false;
      });
    }
    const surfaceDiagnosticsCompletion = this.waterPipeline.completeSurfaceDiagnostics(presentationCompletion);
    // The ordinary completion callback only retires a throughput slot and feeds
    // FPS evidence. First-frame startup additionally needs proof that its exact
    // submission completed before publishing renderer authority.
    if(initialLiveSvoSubmission||initialRasterSubmission){
      const presentationDevice=this.device;
      void presentationCompletion.then(async()=>{
        if(this.disposed||this.deviceLost||this.device!==presentationDevice)return;
        if(initialLiveSvoSubmission)this.settleLiveSvoPresentation(initialLiveSvoSubmission);
        if(initialRasterSubmission){
          const initialDiagnostics=await surfaceDiagnosticsCompletion;
          this.settleInitialRasterPresentation(initialRasterSubmission,surfaceDiagnosticsRequired,initialDiagnostics);
        }
      }).catch(()=>{ /* Device loss is reported by device.lost. */ });
    }
    return this.currentFrameMetrics(
      config.methodId,
      presentationContext,
      true,
      cpuTrace?.finish(),
    );
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    const fluid = this.gpuFluid;
    const sidecar = this.svoSceneSidecar;
    this.gpuFluid = undefined;
    this.svoSceneSidecar = undefined;
    this.pendingInitialRasterPresentation = undefined;
    this.pendingLiveSvoPresentation = undefined;
    this.svoPickingAvailable = false;
    this.lastSvoPickingBodies = [];
    this.gpuFluidRequestGeneration += 1;
    this.gpuFluidInitializationAbort?.abort();
    this.gpuFluidInitializationAbort = undefined;
    this.gpuFluidPendingKey = "";
    this.resetGPUQueueTracking();
    this.gpuFluidGeneration += 1;
    try { fluid?.destroy(); } catch { /* Device loss can invalidate solver resources first. */ }
    try { sidecar?.destroy(); } catch { /* Device loss can invalidate sparse scene resources first. */ }
    for (const retired of this.retiredGPUFluids) { try { retired.destroy(); } catch { /* Best-effort cleanup after device loss. */ } }
    this.retiredGPUFluids.clear();
    try { this.waterPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.gridOverlayPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.tracerOverlayPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.faceVelocityOverlayPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.pressureJournalOverlayPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.stageLensOverlayPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.svoDryScenePipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    for (const resource of [this.presentationTexture, this.fluidTexture, this.columnBaseTexture, this.gridCellTexture, this.velocityFallbackTexture, this.pressureSamplesFallbackTexture, this.scalarFallbackTexture, this.uniformBuffer, this.bodyBuffer, ...this.rigidPoseStaging.map((slot) => slot.buffer)]) {
      try { resource?.destroy(); } catch { /* Best-effort cleanup during hot reload. */ }
    }
    this.rigidPoseStaging = [];
    if (this.device) invalidateGPUCompilationManager(this.device, "renderer destroyed");
    try { this.device?.destroy(); } catch { /* The device may already be lost. */ }
  }

  /**
   * Abort new work immediately, then wait for every host-side initialization
   * transaction that can still publish a resource. Callers may release their
   * external exclusivity lease only after this promise resolves.
   */
  shutdown(): Promise<void> {
    if (this.shutdownPromise) return this.shutdownPromise;
    const initialization = this.initializationPromise;
    const solverInitialization = this.gpuFluidPending;
    const optionalInitializations = [...this.optionalPipelineTasks.values()];
    this.destroy();
    this.shutdownPromise = Promise.allSettled([
      ...(initialization ? [initialization] : []),
      ...(solverInitialization ? [solverInitialization] : []),
      ...optionalInitializations,
    ]).then(() => {});
    return this.shutdownPromise;
  }
}
