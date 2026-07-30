import { WebGpuSvoFluidCoverage } from "./webgpu-svo-fluid-coverage";
import { cameraBasis, dot } from "./math";
import type { CameraState, SceneDescription } from "./model";
import { boundingRadius, type RigidBodyState } from "./rigid-body";
import type { EulerianRenderState } from "./eulerian-solver";
import type { GPUEulerianInfo, GPURigidLoad, GPUQuality } from "./webgpu-eulerian";
import { getMethod, type GPUSolverInstance, type MethodParamValues } from "./methods";
import { GridOverlayPipeline } from "./webgpu-grid-overlay";
import { requiredFluidDeviceLimits } from "./webgpu-device-limits";
import { RasterWaterPipeline, type WaterRenderDiagnostics, type WaterSurfacePresentationDiagnostics } from "./webgpu-water-pipeline";
import { environmentIndex, type EnvironmentId, defaultEnvironmentId } from "./environments";
import { MAX_TERRAIN_FEATURES, TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT, sceneHasTerrain } from "./terrain";
import { SecondaryParticleRenderPipeline } from "./webgpu-secondary-particles";
import { SparseVoxelDebugRenderer, type SparseVoxelRenderSource, type SparseVoxelSceneRenderSource, type VoxelRenderMode } from "./webgpu-voxel-debug";
import { CAMERA_TAN_HALF_FOV, viewportAspect } from "./webgpu-camera";
import {
  buildSvoPixelTraceGeometry,
  type SvoPixelTrace,
  type SvoPixelTraceLayer,
} from "./svo-pixel-trace";
import { SparseVoxelPixelTraceOverlay } from "./webgpu-svo-pixel-trace-overlay";
import { buildSparseVoxelDrySceneLightingMirrors, canConsumeSparseVoxelLighting, canConsumeSparseVoxelPbrMaterials, canEncodeSparseVoxelDryScene, resolveSparseVoxelThickGlassBinderStatus, SparseVoxelDrySceneRenderer, SVO_DRY_SCENE_REVERSED_Z_NEAR_M, type SparseVoxelDrySceneData } from "./webgpu-svo-dry-scene";
import {
  buildSvoScenePrimitives,
  packSvoScenePrimitiveAnimation,
  svoScenePrimitiveAnimation,
  type SvoScenePrimitiveAnimation,
} from "./svo-scene-primitives";
import { buildSvoSceneGlass } from "./svo-scene-glass";
import { buildSvoSceneThickGlass } from "./svo-scene-thick-glass";
import { buildSvoTerrainMaterial } from "./svo-terrain-material";
import { svoEnvironmentAmbientBackgroundLinear } from "./svo-environment-lighting";
import {
  DEFAULT_SVO_LIGHTING_MODE,
  DEFAULT_SVO_LIGHTING_OPTIONS,
  DEFAULT_SVO_RENDER_MODE,
  type SvoLightingMode,
  type SvoLightingOptions,
  type SvoRenderMode,
} from "./svo-render-mode";
import type { SparseVoxelTemporalFrameState } from "./webgpu-svo-temporal-accumulator";
import { DEFAULT_SVO_RENDER_DIAGNOSTICS, normalizeSvoRenderDiagnostics, svoCostOverlayCode, type SvoRenderDiagnostics } from "./svo-render-diagnostics";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, svoRenderTuningKey, type SvoRenderTuning } from "./svo-render-tuning";
import { isGPUInitializationAbort } from "./gpu-initialization";
import { GPUAdvanceWallEstimator } from "./gpu-advance-pacing";
import { createGlobalFineLevelSetConsumerSource } from "./octree-consumer-sampling";
import { OCTREE_TECHNIQUE_OVERLAY_CODES, isOctreeTechniqueOverlayMode, type OctreeTechniqueOverlayMode } from "./octree-technique-debug";
import { OctreeTechniqueOverlayPipeline } from "./webgpu-octree-technique-overlay";
import {
  automaticGPURecoveryEnabled,
  fluidExecutionDeviceFeatures,
} from "./gpu-startup";
import { OctreeTechniqueAuditOverlayPipeline } from "./webgpu-octree-technique-audit-overlay";
import { initialRasterPresentationReadiness, requiresFencedInitialRasterPresentation } from "./gpu-t0-presentation";
import { WebGPUStaticSvoScene } from "./webgpu-static-svo-scene";
import { planSceneRuntime } from "./scene-runtime";
import type { GPUFailureReproduction } from "./webgpu-failure-reproduction";
import {
  CPUPerformanceTrace,
  GPUQueueWallPerformanceTraceRecorder,
  GPUStageTimestampRecorder,
  type GPUTimestampPhase,
  type PerformanceTrace,
} from "./performance-trace";
import { usePerformanceInstrumentationStore } from "./stores/performance-instrumentation-store";

export type SimulationBackend = "webgpu" | "cpu-reference";
export const MAX_PRESENTATION_GAP_MS = 1000 / 60;
export const SVO_SHADOW_HISTORY_WARMUP_FRAMES = 2;
export const SVO_CAMERA_CHANGING_FRAME = -2;

export function svoShadowTemporalFrame(enabled: boolean, stableFrames: number, presentationFrameIndex: number): number {
  return enabled && stableFrames >= SVO_SHADOW_HISTORY_WARMUP_FRAMES
    ? Math.max(0, Math.floor(presentationFrameIndex)) % 16_777_216
    : -1;
}

/** Preserve shadow-temporal eligibility while also publishing camera motion to dry-scene shading. */
export function svoDrySceneTemporalFrame(shadowTemporalFrame: number, cameraStableFrames: number): number {
  return cameraStableFrames >= SVO_SHADOW_HISTORY_WARMUP_FRAMES
    ? shadowTemporalFrame
    : SVO_CAMERA_CHANGING_FRAME;
}

export function presentationPriorityDue(lastFrameAt_ms: number, now_ms: number) {
  return !Number.isFinite(lastFrameAt_ms) || now_ms - lastFrameAt_ms + 0.5 >= MAX_PRESENTATION_GAP_MS;
}

/** Only start physics when its measured cost still fits before the next presentation. */
export function presentationHasPhysicsSlack(lastFrameAt_ms: number, now_ms: number, physics_ms: number | undefined, presentation_ms = 0) {
  if (!Number.isFinite(lastFrameAt_ms) || !physics_ms || !Number.isFinite(physics_ms) || physics_ms <= 0) return false;
  const elapsed_ms = Math.max(0, now_ms - lastFrameAt_ms);
  return elapsed_ms + physics_ms + Math.max(0, presentation_ms) + 0.5 < MAX_PRESENTATION_GAP_MS;
}

/** Telemetry-only physics partition total. Scheduling deliberately does NOT
 * consume this: it exists only while instrumentation is enabled, and admission
 * must be identical with the profiler on or off (see GPUAdvanceWallEstimator). */
export function observedGPUAdvanceTime_ms(trace: PerformanceTrace | undefined) {
  return trace?.lane === "physics" && Number.isFinite(trace.total_ms) && trace.total_ms > 0
    ? trace.total_ms
    : undefined;
}

/** Submit one solver advance toward the prepared simulation clock. */
export function submitNextPreparedGPUAdvance(fluid: GPUSolverInstance, time_s: number, bodies: RigidBodyState[]) {
  const previousSubmittedTime = fluid.info.submittedTime_s ?? 0;
  if (previousSubmittedTime + 1e-9 < time_s) fluid.advanceTo(time_s, bodies);
  const submittedTime = fluid.info.submittedTime_s ?? previousSubmittedTime;
  return { previousSubmittedTime, submittedTime };
}

/** Estimate a dense post-presentation queue that fits inside one 60 Hz interval. */
export function presentationPhysicsQueueDepth(physics_ms: number | undefined, presentation_ms = 0) {
  // One 8 ms solver step per 60 Hz presentation can never reach real time.
  // Start with two advances until the first completion fence supplies a wall
  // estimate, then let the measured budget take over.
  if (!physics_ms || !Number.isFinite(physics_ms) || physics_ms <= 0) return 2;
  const physicsBudget_ms = Math.max(physics_ms, MAX_PRESENTATION_GAP_MS - Math.max(0, presentation_ms));
  // Prefer one whole extra advance over leaving an unusable tail in the frame
  // budget. This can miss 60 Hz by at most one measured physics step, but
  // produces the highest simulation throughput for indivisible advances.
  return Math.max(1, Math.min(8, Math.ceil(physicsBudget_ms / physics_ms)));
}

/** Bound physics queue depth to the explicitly calculated rolling window. */
export function canQueuePreparedGPUAdvance(pendingAdvances: number, maximumPendingAdvances: number) {
  return pendingAdvances < Math.max(1, maximumPendingAdvances);
}

/** A paused clock can still carry one controller-authorized single step. */
export function pausedTargetRequiresGPUAdvance(simulationRunning: boolean, targetTime_s: number, submittedTime_s: number) {
  return !simulationRunning && targetTime_s > submittedTime_s + 1e-9;
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
  const focal = 1 / CAMERA_TAN_HALF_FOV;
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
export type GridOverlayMode = "structure" | "resolution" | "optical" | "cfl" | "speed" | "phi" | "divergence" | "pressure" | "projection" | "representation" | OctreeTechniqueOverlayMode;

export interface GridOverlayConfig {
  /** Slice axes, or a ray-integrated diagnostic through the complete volume. */
  axis: "off" | "z" | "x" | "y" | "volume";
  position: number;
  mode?: GridOverlayMode;
}

export type OptionalRendererPipeline =
  | "grid-overlay"
  | "technique-overlay"
  | "technique-audit-overlay"
  | "voxel-debug"
  | "svo-dry-scene"
  | "secondary-particles"
  | "pixel-trace-overlay";

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
 * Pipeline compilation requested by the current presentation mode. Explicit
 * raster mode requests none of these; the WebGPU default requests the sparse
 * dry-scene renderer alongside the authoritative water path.
 */
export function optionalRendererPipelineRequests(
  gridOverlay: GridOverlayConfig | undefined,
  voxelRenderMode: VoxelRenderMode,
  svoRenderMode: SvoRenderMode,
  simulationRunning: boolean,
  secondaryParticlesAvailable: boolean,
  pixelTraceActive = false,
): OptionalRendererPipeline[] {
  const requested: OptionalRendererPipeline[] = [];
  if (gridOverlay && gridOverlay.axis !== "off") {
    const technique = Boolean(gridOverlay.mode && isOctreeTechniqueOverlayMode(gridOverlay.mode));
    if (!technique) requested.push("grid-overlay");
    if (technique) {
      requested.push("technique-overlay", "technique-audit-overlay");
    }
  }
  if (voxelRenderMode !== "smooth") requested.push("voxel-debug");
  if (svoRenderMode === "svo" && voxelRenderMode === "smooth") requested.push("svo-dry-scene");
  if (simulationRunning && secondaryParticlesAvailable) requested.push("secondary-particles");
  // The trace overlay is only meaningful over the sparse path it explains.
  if (pixelTraceActive && svoRenderMode === "svo" && voxelRenderMode === "smooth") requested.push("pixel-trace-overlay");
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

/** Static renderer worlds are method-independent; fluid worlds require a GPU solver factory. */
export function canInitializeGPUSceneSource(scene: SceneDescription, methodId: string): boolean {
  const method = getMethod(methodId);
  return !planSceneRuntime(scene, { methodId }).fluidSolver || Boolean(method.createSolver || method.createSolverAsync);
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
export function gpuSceneStructuralKey(scene: SceneDescription, config: SimulationRunConfig): string {
  return `fluid-${planSceneRuntime(scene, { methodId: config.methodId }).fluidSolver}:${config.methodId}:${config.quality}:${JSON.stringify(structuralMethodValues(config))}:${scene.environment ?? "default"}:${JSON.stringify(scene.lighting ?? null)}:${JSON.stringify(scene.voxelDomain)}:${scene.container.width_m}:${scene.container.height_m}:${scene.container.depth_m}:${scene.container.top}:${scene.container.fluidWallMode}`;
}

/**
 * Scene-derived solver inputs. `scene.terrain` belongs here and was previously
 * absent from the key entirely, so a terrain edit never reached the solver —
 * the editor's terrain handles depend on this being fixed.
 */
export function gpuSceneSeedKey(scene: SceneDescription): string {
  return `${scene.container.fillFraction}:${JSON.stringify(scene.rigidBodies)}:${scene.fluid.initialCondition}:${JSON.stringify(scene.fluid.initialDamBreakDimensions_m ?? null)}:${JSON.stringify(scene.fluid.initialBrickSeeds_m ?? null)}:${scene.fluid.initialBrickSeedsAdditive ?? false}:${JSON.stringify(scene.terrain ?? null)}:${JSON.stringify(scene.fluid.inflow ?? null)}`;
}

/** Pure scalars; no lattice or seed depends on them. */
export function gpuSceneUniformKey(scene: SceneDescription): string {
  return `${scene.fluid.density_kg_m3}:${scene.fluid.dynamicViscosity_Pa_s}:${scene.fluid.surfaceTension_N_m}:${scene.fluid.gravity_m_s2.y}`;
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
  return `${config.simulationEpoch ?? 0}:${gpuSceneStructuralKey(scene, config)}:${gpuSceneSeedKey(scene)}`;
}

export type GPUStatus =
  | {
      state: "initializing";
      label: string;
      phase?: string;
      completed?: number;
      total?: number;
      startedAt_ms?: number;
      /** Startup has no previous frame; rebuild keeps the prior GPU state visible. */
      kind?: "startup" | "rebuild";
      /** User-facing description captured synchronously when a setting changes. */
      operation?: string;
      retainingPrevious?: boolean;
    }
  | { state: "ready"; label: string; adapter: string }
  | { state: "blocked"; label: string }
  | { state: "manual"; label: string }
  | { state: "stopping"; label: string }
  | { state: "unavailable"; label: string; reproduction?: GPUFailureReproduction }
  | { state: "lost"; label: string };

export type SvoRendererFallbackReason =
  | "missing-source"
  | "unsupported-terrain"
  | "unsupported-glass-cutout"
  | "missing-pbr-materials"
  | "missing-lighting-publications"
  | "pipeline-compile-failure"
  | "inspection-mode";

export interface EffectiveRendererStatus {
  requestedMode: SvoRenderMode;
  effectiveMode: SvoRenderMode;
  fallbackReason?: SvoRendererFallbackReason;
}

export interface EffectiveRendererConditions {
  pipelineAvailable: boolean;
  sourceAvailable: boolean;
  terrainSupported: boolean;
  glassSupported?: boolean;
  materialsSupported?: boolean;
  lightingSupported?: boolean;
  inspectionMode: boolean;
  svoEncoded: boolean;
}

/** Resolve one frame's production renderer without changing simulation state. */
export function resolveEffectiveRendererStatus(
  requestedMode: SvoRenderMode,
  conditions: EffectiveRendererConditions,
): EffectiveRendererStatus {
  if (requestedMode === "raster") return { requestedMode, effectiveMode: "raster" };
  if (conditions.inspectionMode) return { requestedMode, effectiveMode: "raster", fallbackReason: "inspection-mode" };
  if (!conditions.pipelineAvailable) return { requestedMode, effectiveMode: "raster", fallbackReason: "pipeline-compile-failure" };
  if (!conditions.terrainSupported) return { requestedMode, effectiveMode: "raster", fallbackReason: "unsupported-terrain" };
  if (conditions.glassSupported === false) return { requestedMode, effectiveMode: "raster", fallbackReason: "unsupported-glass-cutout" };
  if (conditions.materialsSupported === false) return { requestedMode, effectiveMode: "raster", fallbackReason: "missing-pbr-materials" };
  if (conditions.lightingSupported === false) return { requestedMode, effectiveMode: "raster", fallbackReason: "missing-lighting-publications" };
  if (!conditions.sourceAvailable || !conditions.svoEncoded) return { requestedMode, effectiveMode: "raster", fallbackReason: "missing-source" };
  return { requestedMode, effectiveMode: "svo" };
}

export interface RendererFrameMetrics {
  cpu?: PerformanceTrace;
  presentation?: PerformanceTrace;
  context: string;
  methodId: string;
  /** True only when this draw encoded and submitted a presentation command buffer. */
  presentationSubmitted: boolean;
  /** Latest presentation evidence; independent of solver publication authority. */
  waterSurfacePresentation?: WaterSurfacePresentationDiagnostics;
}

const PRESENTATION_TRACE_PHASES: readonly GPUTimestampPhase[] = [
  { id: "surface-extraction", label: "Surface extraction + caustics" },
  { id: "dry-scene", label: "Dry scene + temporal lighting" },
  { id: "water-interfaces", label: "Front/back water interfaces" },
  { id: "optical-composite", label: "Optical composite" },
  { id: "inspection-overlay", label: "Inspection overlays" },
  { id: "present", label: "Final presentation" },
] as const;

interface PendingInitialRasterPresentation {
  readonly solver: GPUSolverInstance;
  readonly solverGeneration: number;
  readonly requestGeneration: number;
  submitted: boolean;
}

interface PendingStaticSvoPresentation {
  readonly solver: GPUSolverInstance;
  readonly solverGeneration: number;
  readonly requestGeneration: number;
  readonly startedAt_ms: number;
  attached: boolean;
  submitted: boolean;
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
  private voxelDebugPipeline?: SparseVoxelDebugRenderer;
  private svoDryScenePipeline?: SparseVoxelDrySceneRenderer;
  private svoDrySceneSource?: SparseVoxelSceneRenderSource;
  private svoDrySceneData?: SparseVoxelDrySceneData;
  /** Authored scenery motion for the attached scene, absent when nothing moves. */
  private sceneryAnimation?: SvoScenePrimitiveAnimation;
  /**
   * Presentation clock for authored scenery, in seconds since the first frame
   * that drew it. Deliberately not the simulation clock: a garden the user has
   * paused to inspect its lighting should still be alive, and the gust is not a
   * physical quantity anyone reads off the timeline.
   */
  private sceneryAnimationOrigin_ms?: number;
  private gridOverlayPipeline?: GridOverlayPipeline;
  private techniqueOverlayPipeline?: OctreeTechniqueOverlayPipeline;
  private techniqueAuditOverlayPipeline?: OctreeTechniqueAuditOverlayPipeline;
  private pixelTraceOverlayPipeline?: SparseVoxelPixelTraceOverlay;
  /** Latest decoded trace, and a revision the UI polls instead of a callback. */
  private latestPixelTraceValue?: SvoPixelTrace;
  private pixelTraceRevisionValue = 0;
  private pixelTraceGeometryKey = "";
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
  private presentationTexture?: GPUTexture;
  private voxelDebugDepth?: GPUTexture;
  private presentationTextureKey = "";
  private activeRenderScale = 1;
  private uniformBuffer?: GPUBuffer;
  private bodyBuffer?: GPUBuffer;
  private fluidTexture?: GPUTexture;
  private columnBaseTexture?: GPUTexture;
  private gridCellTexture?: GPUTexture;
  private velocityFallbackTexture?: GPUTexture;
  private pressureSamplesFallbackTexture?: GPUTexture;
  private scalarFallbackTexture?: GPUTexture;
  private fluidTextureKey = "";
  private fluidRevision = -1;
  private gpuFluid?: GPUSolverInstance;
  private readonly retiredGPUFluids = new Set<GPUSolverInstance>();
  private gpuFluidKey = "";
  private gpuFluidPendingKey = "";
  private gpuFluidPending?: Promise<void>;
  private gpuFluidInitializationAbort?: AbortController;
  private gpuFluidRequestGeneration = 0;
  private adapterName = "WebGPU adapter";
  private gpuInfoCallback?: (info: GPUEulerianInfo) => void;
  private gpuRigidLoadCallback?: (loads: GPURigidLoad[]) => void;
  private gpuAdvanceCompletedCallback?: (time_s: number) => void;
  private effectiveRendererStatusCallback?: (status: EffectiveRendererStatus) => void;
  private lastEffectiveRendererStatus?: EffectiveRendererStatus;
  private svoPickingAvailable = false;
  private lastSvoPickingBodies: RigidBodyState[] = [];
  private svoSourceAvailable = false;
  private svoTerrainSupported = true;
  private svoGlassSupported = true;
  private svoMaterialsSupported = true;
  private svoLightingSupported = true;
  private svoPipelineAvailable = false;
  /** Internal A/B: temporal-off also restores full-rate shadow visibility. */
  private svoTemporalAccumulationEnabled = DEFAULT_SVO_RENDER_TUNING.temporalEnabled;
  private presentationFrameIndex = 0;
  private svoCameraStabilityKey = "";
  private svoCameraStableFrames = 0;
  private svoShadowStabilityKey = "";
  private svoShadowStableFrames = 0;
  private svoRenderDiagnosticsKey = "";
  private gpuPendingBatches = 0;
  /** Instrumentation-independent per-advance wall cost; the only scheduling
   * input. The hardware physics trace is telemetry and must never feed
   * admission, or toggling the profiler changes the simulation's driver. */
  private readonly advanceWallEstimator = new GPUAdvanceWallEstimator();
  /** Same contract for presentation cost: the traced presentation total is
   * instrumentation-gated, so scheduling reads this fence-derived estimate. */
  private readonly presentationWallEstimator = new GPUAdvanceWallEstimator();
  /** Monotone queue epochs used to reject wall samples that contain work from
   * the other lane. These are submission-order evidence, not telemetry. */
  private physicsQueueWorkSequence = 0;
  private nonPhysicsQueueWorkSequence = 0;
  /** Stats readbacks are queue work too; keep idle evidence conservative until
   * their promise settles. */
  private diagnosticQueueWorkPending = 0;
  private lastPresentationCompletedAt_ms = -Infinity;
  private presentationPending = false;
  private simulationRunning = true;
  private preparedGPUTime_s = 0;
  private preparedGPUBodies: RigidBodyState[] = [];
  private gpuFluidGeneration = 0;
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
  /** Static worlds become ready only after their first dry-SVO frame completes. */
  private pendingStaticSvoPresentation?: PendingStaticSvoPresentation;
  private svoPipelineProgress?: { label: string; completed: number };
  private svoPipelineStartedAt_ms?: number;
  /** Debug compaction owns capacity-sized instance buffers only in inspection modes. */
  private voxelDebugSourceGeneration = -1;
  private voxelInspectionSource?: SparseVoxelRenderSource;
  private lastGPUInfoPollAt_ms = -Infinity;
  private format?: GPUTextureFormat;
  private presentationContext = "";
  private cpuTraceSampleId = 0;
  private presentationTraceSampleId = 0;
  private lastPresentationTraceAt_ms = -Infinity;
  private presentationTracePending = false;
  /** One unusable hardware sample retires the stage recorder for this device;
   * the non-invasive queue-wall observation takes over. */
  private hardwarePresentationTraceInvalid = false;
  private latestPresentationTrace?: PerformanceTrace;
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

  get presentationRevision(): number { return this.pausedPresentationRevision; }

  constructor(private readonly canvas: HTMLCanvasElement, private readonly onStatus: (status: GPUStatus) => void, onGPUInfo?: (info: GPUEulerianInfo) => void, onGPURigidLoads?: (loads: GPURigidLoad[]) => void, onGPUAdvanceCompleted?: (time_s: number) => void, onEffectiveRendererStatus?: (status: EffectiveRendererStatus) => void) { this.gpuInfoCallback = onGPUInfo; this.gpuRigidLoadCallback = onGPURigidLoads; this.gpuAdvanceCompletedCallback = onGPUAdvanceCompleted; this.effectiveRendererStatusCallback = onEffectiveRendererStatus; }

  private publishEffectiveRendererStatus(status: EffectiveRendererStatus) {
    const previous = this.lastEffectiveRendererStatus;
    if (previous?.requestedMode === status.requestedMode && previous.effectiveMode === status.effectiveMode && previous.fallbackReason === status.fallbackReason) return;
    this.lastEffectiveRendererStatus = status;
    this.effectiveRendererStatusCallback?.(status);
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
      console.warn(`Optional ${key} pipeline unavailable`, error);
      if (key === "svo-dry-scene") this.failPendingStaticSvoPresentation(error);
      return undefined;
    }
    const task = initialize(candidate).then(() => {
      if (this.disposed || this.deviceLost || this.device !== device) {
        try { destroy(candidate); } catch { /* Device loss may invalidate resources first. */ }
        return;
      }
      publish(candidate);
      this.pausedPresentationRevision += 1;
    }).catch((error: unknown) => {
      try { destroy(candidate); } catch { /* Best-effort cleanup after compile failure. */ }
      if (this.device === device && !this.disposed && !this.deviceLost) {
        this.failedOptionalPipelines.add(key);
        console.warn(`Optional ${key} pipeline unavailable`, error);
        if (key === "svo-dry-scene") this.failPendingStaticSvoPresentation(error);
      }
    }).finally(() => {
      if (this.optionalPipelineTasks.get(key) === task) this.optionalPipelineTasks.delete(key);
    });
    this.optionalPipelineTasks.set(key, task);
    return undefined;
  }

  private reportSvoPipelineProgress(label: string, completed: number) {
    this.svoPipelineStartedAt_ms ??= performance.now();
    this.svoPipelineProgress = { label, completed };
    const pending = this.pendingStaticSvoPresentation;
    this.onStatus({
      state: "initializing", label, phase: "presentation", completed, total: 4,
      startedAt_ms: pending?.startedAt_ms ?? this.svoPipelineStartedAt_ms, kind: "startup", retainingPrevious: false,
    });
  }

  private reportStaticSvoAttachment() {
    const pending = this.pendingStaticSvoPresentation;
    if (!pending || pending.attached || !this.svoDryScenePipeline || !this.svoSourceAvailable) return;
    pending.attached = true;
    this.onStatus({
      state: "initializing", label: "Sparse garden renderer attached", phase: "presentation",
      completed: 3, total: 4, startedAt_ms: pending.startedAt_ms, kind: "startup", retainingPrevious: false,
    });
  }

  private failPendingStaticSvoPresentation(error?: unknown) {
    const pending = this.pendingStaticSvoPresentation;
    if (!pending) return;
    this.pendingStaticSvoPresentation = undefined;
    const detail = error instanceof Error && error.message ? `: ${error.message}` : "";
    this.onStatus({ state: "blocked", label: `Sparse garden renderer unavailable${detail}` });
  }

  private ensureRequestedOptionalPipelines(requested: readonly OptionalRendererPipeline[]) {
    const wants = new Set(requested);
    if (wants.has("grid-overlay")) this.ensureOptionalPipeline(
      "grid-overlay", this.gridOverlayPipeline,
      (device) => new GridOverlayPipeline(device, this.format!, this.uniformBuffer!, this.bodyBuffer!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => { this.gridOverlayPipeline = pipeline; this.updateRenderSources(); },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("technique-overlay")) this.ensureOptionalPipeline(
      "technique-overlay", this.techniqueOverlayPipeline,
      (device) => new OctreeTechniqueOverlayPipeline(device, this.format!, this.uniformBuffer!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.techniqueOverlayPipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
        pipeline.setOwnerRows(this.gpuFluid?.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture!);
      },
    );
    if (wants.has("technique-audit-overlay")) this.ensureOptionalPipeline(
      "technique-audit-overlay", this.techniqueAuditOverlayPipeline,
      (device) => new OctreeTechniqueAuditOverlayPipeline(device, this.format!, this.uniformBuffer!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => {
        this.techniqueAuditOverlayPipeline = pipeline;
        pipeline.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
        pipeline.setOwnerRows(this.gpuFluid?.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture!);
      },
    );
    if (wants.has("voxel-debug")) this.ensureOptionalPipeline(
      "voxel-debug", this.voxelDebugPipeline,
      (device) => new SparseVoxelDebugRenderer(device, { colorFormat: this.format! }),
      (pipeline) => pipeline.initialize(),
      (pipeline) => { this.voxelDebugPipeline = pipeline; pipeline.setSource(this.voxelInspectionSource); },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("svo-dry-scene")) this.ensureOptionalPipeline(
      "svo-dry-scene", this.svoDryScenePipeline,
      (device) => new SparseVoxelDrySceneRenderer(device, this.uniformBuffer!, this.bodyBuffer!),
      (pipeline) => pipeline.initialize((label, completed) => this.reportSvoPipelineProgress(label, completed)),
      (pipeline) => {
        this.svoDryScenePipeline = pipeline;
        this.svoPipelineAvailable = true;
        pipeline.setSource(this.svoDrySceneSource, this.svoDrySceneData);
        if (this.presentationTexture) pipeline.ensureSize(this.presentationTexture.width, this.presentationTexture.height);
        this.reportStaticSvoAttachment();
      },
      (pipeline) => pipeline.destroy(),
    );
    if (wants.has("pixel-trace-overlay")) this.ensureOptionalPipeline(
      "pixel-trace-overlay", this.pixelTraceOverlayPipeline,
      (device) => new SparseVoxelPixelTraceOverlay(device, this.format!),
      (pipeline) => pipeline.initialize(),
      (pipeline) => { this.pixelTraceOverlayPipeline = pipeline; },
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

  /** Resolve a click against live GPU poses without restoring a CPU pose mirror. */
  async pickRigidBody(
    origin: RigidBodyState["position_m"],
    direction: RigidBodyState["position_m"],
    screen?: { normalizedX: number; normalizedY: number },
  ) {
    if (this.svoPickingAvailable && screen && this.svoDryScenePipeline) {
      const bodies = this.lastSvoPickingBodies, pipeline = this.svoDryScenePipeline;
      const picked = await pipeline.pickGBuffer(
        screen.normalizedX, screen.normalizedY,
        [origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z], bodies.length,
      );
      if (!this.svoPickingAvailable || this.svoDryScenePipeline !== pipeline || picked.status !== "hit") return undefined;
      const body = bodies[picked.bodyIndex];
      if (!body) return undefined;
      return {
        bodyIndex: picked.bodyIndex,
        distance_m: picked.depth_m,
        position_m: body.position_m,
        orientation: body.orientation,
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
   * "this scene is on the raster fallback, so there is no sparse traversal to
   * trace" — and both look like a broken pointer.
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

  private encodePixelTraceOverlay(
    encoder: GPUCommandEncoder,
    basis: ReturnType<typeof cameraBasis>,
    config: PixelTraceConfig,
  ): void {
    const overlay = this.pixelTraceOverlayPipeline, trace = this.latestPixelTraceValue;
    if (!overlay?.ready || !trace || !this.presentationTexture) return;
    // Rebuild the segment buffer only when the trace or the requested layers
    // change: orbiting a pinned trace is a uniform update, not a re-upload.
    const key = `${this.pixelTraceRevisionValue}|${config.layers.join(",")}|${config.widthScale ?? 1}`;
    if (key !== this.pixelTraceGeometryKey) {
      const geometry = buildSvoPixelTraceGeometry(trace, { layers: config.layers, widthScale: config.widthScale });
      overlay.setGeometry(geometry);
      this.pixelTraceGeometryKey = key;
    }
    const width = this.presentationTexture.width, height = this.presentationTexture.height;
    overlay.encode(encoder, this.cachedTextureView(this.presentationTexture), this.pixelTraceSceneDepthView(), {
      camera: {
        position_m: [basis.position.x, basis.position.y, basis.position.z],
        forward: [basis.forward.x, basis.forward.y, basis.forward.z],
        right: [basis.right.x, basis.right.y, basis.right.z],
        up: [basis.up.x, basis.up.y, basis.up.z],
        tanHalfFov: CAMERA_TAN_HALF_FOV,
        aspect: viewportAspect(width, height),
      },
      viewportWidth: width,
      viewportHeight: height,
      reveal: config.reveal,
      occludedAlpha: config.occludedAlpha,
      depthNear_m: SVO_DRY_SCENE_REVERSED_Z_NEAR_M,
    });
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
    const progress=(label:string,completed:number,total=4,phase="renderer")=>this.onStatus({state:"initializing",label,phase,completed,total,startedAt_ms});
    // UI-only browser automation must be safe even if a caller accidentally
    // invokes initialize(): return before navigator.gpu or solver creation.
    if (typeof location !== "undefined" && new URLSearchParams(location.search).get("gpu") === "off") {
      this.onStatus({ state: "unavailable", label: "WebGPU disabled by gpu=off (UI-only mode)" });
      return;
    }
    progress("Requesting WebGPU adapter",0);
    if (!("gpu" in navigator)) {
      this.onStatus({ state: "unavailable", label: "WebGPU is not available in this browser" });
      return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (this.disposed) return;
    if (!adapter) {
      this.onStatus({ state: "unavailable", label: "No compatible GPU adapter was found" });
      return;
    }
    progress("Requesting GPU device",1);
    const requiredFeatures = fluidExecutionDeviceFeatures(adapter.features);
    const requiredLimits = requiredFluidDeviceLimits(adapter.limits);
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    if (this.disposed) { device.destroy(); return; }
    const context = this.canvas.getContext("webgpu");
    if (!context) {
      device.destroy();
      this.onStatus({ state: "unavailable", label: "WebGPU canvas context could not be created" });
      return;
    }
    this.device = device;
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    device.addEventListener("uncapturederror", (event) => console.error(`WebGPU validation: ${event.error.message}`));
    void device.lost.then((info) => {
      if (this.disposed || this.device !== device || this.deviceLost) return;
      this.deviceLost = true;
      const fluid = this.gpuFluid;
      this.gpuFluid = undefined;
      this.pendingInitialRasterPresentation = undefined;
      this.pendingStaticSvoPresentation = undefined;
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
      // Breadcrumbs for hang diagnosis: the last known solver state narrows a
      // watchdog reset down to a stage without needing a reproduction.
      if (fluid) console.error("GPU device lost mid-simulation", { reason: info.reason, message: info.message, submittedTime_s: fluid.info.submittedTime_s, completedTime_s: fluid.info.completedTime_s, pendingBatches: this.gpuPendingBatches, encodedSteps: fluid.info.encodedSteps, physicsTrace: fluid.info.physicsTrace });
      this.onStatus({ state: "lost", label: `GPU device lost: ${info.message || info.reason}` });
      this.scheduleDeviceRecovery(info.reason);
    }).catch((error: unknown) => {
      if (!this.disposed) console.error("Unable to observe WebGPU device loss", error);
    });
    context.configure({ device, format: this.format, alphaMode: "opaque" });

    progress("Compiling presentation upscale",2);
    const upscaleModule=device.createShaderModule({label:"Presentation upscale shader",code:upscaleShader});
    const upscalePipeline=await device.createRenderPipelineAsync({label:"Presentation upscale",layout:"auto",vertex:{module:upscaleModule,entryPoint:"vertexMain"},fragment:{module:upscaleModule,entryPoint:"fragmentMain",targets:[{format:this.format}]},primitive:{topology:"triangle-list"}});
    if (this.disposed || this.device !== device || this.deviceLost) return;
    this.upscalePipeline=upscalePipeline;
    this.upscaleSampler=device.createSampler({magFilter:"linear",minFilter:"linear"});
    this.uniformBuffer = device.createBuffer({ label: "Fluid Lab view uniforms", size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
    this.onStatus({ state: "ready", label: "WebGPU renderer ready", adapter: this.adapterName });
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
    this.waterPipeline = undefined; this.gridOverlayPipeline = undefined; this.techniqueOverlayPipeline = undefined; this.techniqueAuditOverlayPipeline = undefined; this.voxelDebugPipeline = undefined; this.svoDryScenePipeline = undefined; this.secondaryParticlePipeline = undefined;
    this.optionalPipelineTasks.clear(); this.failedOptionalPipelines.clear(); this.svoDrySceneSource = undefined; this.svoDrySceneData = undefined; this.sceneryAnimation = undefined; this.svoPipelineProgress = undefined; this.svoPipelineStartedAt_ms = undefined; this.pendingStaticSvoPresentation = undefined;
    this.svoPipelineAvailable = false; this.svoSourceAvailable = false; this.svoTerrainSupported = true; this.svoGlassSupported = true; this.svoMaterialsSupported = true; this.svoLightingSupported = true;
    this.uniformBuffer = undefined; this.bodyBuffer = undefined;
    this.presentationTexture = undefined; this.voxelDebugDepth = undefined; this.presentationTextureKey = "";
    this.fluidTexture = undefined; this.columnBaseTexture = undefined; this.gridCellTexture = undefined;
    this.velocityFallbackTexture = undefined; this.pressureSamplesFallbackTexture = undefined; this.scalarFallbackTexture = undefined;
    this.fluidTextureKey = ""; this.fluidRevision = -1;
    this.presentationTracePending = false; this.latestPresentationTrace = undefined;
    this.retiredGPUFluids.clear();
    this.deviceLost = false;
    try {
      await this.initialize();
    } catch (error) {
      this.onStatus({ state: "unavailable", label: error instanceof Error ? `GPU recovery failed: ${error.message}` : "GPU recovery failed" });
    }
  }


  /**
   * Build or reuse the fluid coverage volume for this frame.
   *
   * Gated on the global-fine lane, because that is the only configuration in
   * which the solver's dense field carries a signed distance; the tall-cell and
   * quadtree lanes pack occupancy into the same texture, and resampling that as
   * a distance would put a shadow wherever the field happened to be positive.
   * Outside the lane the volume is released and water casts no shadow, which is
   * the behaviour every lane had before.
   */
  private ensureFluidCoverage(
    solver: GPUSolverInstance | undefined,
    scene: SceneDescription,
  ): WebGpuSvoFluidCoverage | undefined {
    const device = this.device;
    // surfaceFieldTexture is the contoured level set when the solver keeps one
    // apart from volumeTexture; it is the field globalCoarsePhi already reads.
    const field = device && solver && this.globalFineWaterAttached && solver.globalFineLevelSetSource
      ? solver.surfaceFieldTexture ?? solver.volumeTexture
      : undefined;
    const info = solver?.info;
    if (!field || !info || !device) {
      this.svoFluidCoverage?.destroy();
      this.svoFluidCoverage = undefined;
      this.svoFluidCoverageKey = undefined;
      return undefined;
    }
    const dimensions = [info.nx, info.ny, info.nz] as const;
    const container = scene.container;
    const key = `${field.label}:${dimensions.join(",")}:${container.width_m},${container.height_m},${container.depth_m}`;
    if (this.svoFluidCoverage && this.svoFluidCoverageKey === key) return this.svoFluidCoverage;
    this.svoFluidCoverage?.destroy();
    this.svoFluidCoverageKey = key;
    try {
      // The dense field spans the container box exactly, which is the same
      // mapping the raster composite uses to sample it.
      this.svoFluidCoverage = new WebGpuSvoFluidCoverage(device, {
        fieldDimensions: dimensions,
        worldOrigin_m: [-container.width_m / 2, 0, -container.depth_m / 2],
        cellSize_m: [container.width_m / info.nx, container.height_m / info.ny, container.depth_m / info.nz],
      }, { coarsePhi: field.createView({ dimension: "3d" }) });
    } catch {
      this.svoFluidCoverage = undefined;
      this.svoFluidCoverageKey = undefined;
    }
    return this.svoFluidCoverage;
  }

  private updateRenderSources(texture = this.fluidTexture, columnSource?: GPUTexture, gridCells = this.gridCellTexture, velocity = this.velocityFallbackTexture, pressureSamples = this.pressureSamplesFallbackTexture, divergence = this.scalarFallbackTexture, pressure = this.scalarFallbackTexture) {
    const columnBases = columnSource ?? this.columnBaseTexture;
    if (!this.device || this.disposed || this.deviceLost || !texture || !columnBases || !gridCells || !velocity || !pressureSamples || !divergence || !pressure) return;
    this.waterPipeline?.setVolume(texture, columnBases);
    const globalFineLevelSet = this.gpuFluid?.globalFineLevelSetSource;
    this.waterPipeline?.setGlobalFineLevelSet(globalFineLevelSet
      ? createGlobalFineLevelSetConsumerSource(globalFineLevelSet)
      : undefined);
    this.gridOverlayPipeline?.setVolume(texture, columnBases, gridCells, velocity, pressureSamples, divergence, pressure);
    this.techniqueOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    this.techniqueOverlayPipeline?.setOwnerRows(pressureSamples);
    this.techniqueAuditOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    this.techniqueAuditOverlayPipeline?.setOwnerRows(pressureSamples);
  }

  private solverKey(scene:SceneDescription,config:SimulationRunConfig){return gpuSceneSolverKey(scene,config);}
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
  private tryReseedGPUFluid(scene:SceneDescription,config:SimulationRunConfig,key:string):boolean{
    const solver=this.gpuFluid;
    if(!solver?.reseed||this.reseedInFlight)return false;
    if(gpuSceneStructuralKey(scene,config)!==this.attachedStructuralKey)return false;
    const generation=this.gpuFluidGeneration,requestGeneration=this.gpuFluidRequestGeneration;
    this.reseedInFlight=true;
    this.onStatus({state:"initializing",label:"Re-seeding fenced t=0 solver authority",phase:"warmup",completed:0,total:1,startedAt_ms:performance.now(),kind:"rebuild",retainingPrevious:false});
    void solver.reseed(scene).then((reseeded)=>{
      // Anything that replaced or invalidated the solver mid-flight wins.
      if(this.disposed||this.gpuFluid!==solver||this.gpuFluidGeneration!==generation
        ||this.gpuFluidRequestGeneration!==requestGeneration)return;
      if(!reseeded){this.beginGPUFluidInitialization(scene,config,key);return;}
      this.gpuFluidKey=key;this.appliedSceneUniformKey=gpuSceneUniformKey(scene);this.resetGPUQueueTracking();this.lastGPUInfoPollAt_ms=-Infinity;
      // reset() intentionally clears the diagnostics store. A warm re-seed
      // must therefore republish its authority just like a replacement attach,
      // then earn a fresh raster fence before transport unlocks. Merely moving
      // gpuFluidKey leaves the renderer ready while the UI waits forever for
      // initialSparseAuthorityReady/initialRasterSurfaceReady to reappear.
      solver.info.initialRasterSurfaceReady=false;
      solver.info.initialRasterSurfaceState="pending";
      solver.info.initialRasterSurfaceDiagnostic="Waiting for the first fenced t=0 raster publication after re-seed";
      this.globalFineWaterAttached=false;
      this.pendingInitialRasterPresentation={solver,solverGeneration:generation,requestGeneration,submitted:false};
      this.gpuInfoCallback?.({...solver.info});
      this.onStatus({state:"initializing",label:"Re-seeded solver ready; publishing fenced t=0 raster surface",phase:"presentation",completed:0,total:1,startedAt_ms:performance.now(),kind:"rebuild",retainingPrevious:false});
    }).catch(()=>{
      if(!this.disposed&&this.gpuFluid===solver&&this.gpuFluidGeneration===generation
        &&this.gpuFluidRequestGeneration===requestGeneration)this.beginGPUFluidInitialization(scene,config,key);
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
    this.preparedGPUTime_s = 0;
    this.preparedGPUBodies = [];
    this.advanceWallEstimator.reset();
  }

  /**
   * In-flight advances the queue should hold. Without a measured step this is
   * the documented bootstrap of two, which is exactly enough to keep one
   * advance executing while the next waits — so completion-callback latency
   * never reaches the device, with or without instrumentation.
   */
  private preparedGPUQueueDepth(_fluid: GPUSolverInstance) {
    return presentationPhysicsQueueDepth(
      this.advanceWallEstimator.estimate_ms,
      this.presentationWallEstimator.estimate_ms ?? 0,
    );
  }

  /** Begin a new controller timeline before any old GPU completion can commit. */
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

  /** Stop refilling physics immediately while preserving already-submitted queue work. */
  setSimulationRunning(running: boolean): number | undefined {
    if (running !== this.simulationRunning) this.resetPresentationTrace();
    this.simulationRunning = running;
    const submittedTime_s = this.gpuFluid?.info.submittedTime_s;
    if (!running) {
      this.preparedGPUTime_s = submittedTime_s ?? this.gpuFluid?.info.completedTime_s ?? 0;
      this.preparedGPUBodies = [];
    }
    return submittedTime_s;
  }

  /** Clear presentation samples whenever their semantic identity changes. */
  private resetPresentationTrace() {
    this.latestPresentationTrace = undefined;
    this.lastPresentationTraceAt_ms = -Infinity;
    this.presentationWallEstimator.reset();
  }

  private currentFrameMetrics(methodId: string, context: string, presentationSubmitted: boolean, cpu?: PerformanceTrace): RendererFrameMetrics {
    const water = this.waterPipeline?.surfaceRenderDiagnostics;
    const instrumentation = usePerformanceInstrumentationStore.getState();
    const presentation = instrumentation.enabled
      && this.latestPresentationTrace
      && this.latestPresentationTrace.capturedAt_ms >= instrumentation.enabledAt_ms
      ? this.latestPresentationTrace
      : undefined;
    return {
      cpu,
      presentation,
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

  private beginGPUFluidInitialization(scene:SceneDescription,config:SimulationRunConfig,key:string){
    if(!this.device||this.disposed||this.deviceLost)return;
    const method=getMethod(config.methodId);if(!canInitializeGPUSceneSource(scene,config.methodId))return;
    this.gpuFluidInitializationAbort?.abort();
    const abort=new AbortController();this.gpuFluidInitializationAbort=abort;
    const device=this.device,generation=++this.gpuFluidRequestGeneration,startedAt_ms=performance.now();
    const previous=this.gpuFluid;
    const drainPreviousForReset=this.timelineResetPending&&Boolean(previous);
    this.timelineResetPending=false;
    this.pendingStaticSvoPresentation=undefined;
    // The active solver remains attached for presentation throughout the
    // transaction. Only the warmed candidate is allowed to replace it.
    this.gpuFluidPendingKey=key;
    let reportedCompleted=0,reportedTotal=1;
    const report=(progress:{phase:string;taskId?:string;label:string;completed:number;total:number})=>{if(this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration)return;reportedCompleted=progress.completed;reportedTotal=progress.total;this.onStatus({state:"initializing",...progress,startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:Boolean(previous)});};
    let previousDestroyedForReset=false;
    const prepare=async()=>{
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
        this.voxelInspectionSource?.inspectionPublication?.setEnabled(false);this.voxelInspectionSource=undefined;
        this.voxelDebugPipeline?.setSource(undefined);this.voxelDebugSourceGeneration=-1;
        this.svoDrySceneSource=undefined;this.svoDrySceneData=undefined;this.sceneryAnimation=undefined;this.svoDryScenePipeline?.setSource(undefined,undefined);
        previous.destroy();previousDestroyedForReset=true;
      }
      this.resetGPUQueueTracking();
      report({phase:"drain",taskId:"solver.drain",label:"Previous GPU work drained",completed:1,total:1});
    };
    // Annotated because the two branches return different solver classes: without
    // it the inferred TResult1 pins to WebGPUStaticSvoScene and rejects the
    // GPUSolverInstance branch.
    const create:Promise<GPUSolverInstance>=prepare().then(async ():Promise<GPUSolverInstance>=>{
      if(abort.signal.aborted||this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration)throw new DOMException("GPU initialization superseded","AbortError");
      if (!planSceneRuntime(scene,{methodId:config.methodId}).fluidSolver) {
        return WebGPUStaticSvoScene.create(device, scene, config.quality, report, abort.signal);
      }
      return method.createSolverAsync
        ? method.createSolverAsync(device,scene,config.quality,config.values,this.gpuRigidLoadCallback,report,abort.signal)
        : new Promise<GPUSolverInstance>((resolve,reject)=>setTimeout(()=>{try{resolve(method.createSolver!(device,scene,config.quality,config.values,this.gpuRigidLoadCallback));}catch(error){reject(error);}},0));
    });
    this.gpuFluidPending=create.then((solver)=>{
      if(this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration){solver.destroy();return;}
      if(config.methodId==="octree"&&solver.initialSparseAuthorityReady!==true){solver.destroy();throw new Error("Octree solver returned before fenced sparse t=0 authority");}
      report({phase:"attach",taskId:"solver.attach",label:"Attach warmed solver",completed:reportedCompleted,total:reportedTotal+1});
      solver.applyRuntimeValues?.(config.values);
      this.gpuFluid=solver;this.gpuFluidKey=key;this.attachedStructuralKey=gpuSceneStructuralKey(scene,config);this.gpuFluidPendingKey="";this.resetGPUQueueTracking();this.gpuFluidGeneration+=1;this.lastGPUInfoPollAt_ms=-Infinity;this.globalFineWaterAttached=false;
      const staticRenderScene=!planSceneRuntime(scene,{methodId:config.methodId}).fluidSolver;
      const fencedInitialRaster=requiresFencedInitialRasterPresentation(config.methodId);
      if(staticRenderScene){solver.info.initialRasterSurfaceReady=true;solver.info.initialRasterSurfaceState="gpu-authoritative";solver.info.initialRasterSurfaceDiagnostic="Static SVO scene ready; fluid authority intentionally bypassed";this.pendingInitialRasterPresentation=undefined;this.pendingStaticSvoPresentation={solver,solverGeneration:this.gpuFluidGeneration,requestGeneration:generation,startedAt_ms,attached:false,submitted:false};}
      else if(fencedInitialRaster){solver.info.initialRasterSurfaceReady=false;solver.info.initialRasterSurfaceState="pending";solver.info.initialRasterSurfaceDiagnostic="Waiting for the first fenced t=0 raster publication";this.pendingInitialRasterPresentation={solver,solverGeneration:this.gpuFluidGeneration,requestGeneration:generation,submitted:false};}
      else{solver.info.initialRasterSurfaceReady=true;solver.info.initialRasterSurfaceState="gpu-authoritative";solver.info.initialRasterSurfaceDiagnostic="Direct solver field attached; sparse raster fence not required";this.pendingInitialRasterPresentation=undefined;}
      this.updateRenderSources(solver.surfaceFieldTexture??solver.volumeTexture,solver.columnBaseTexture,solver.gridCellTexture??this.gridCellTexture,solver.velocityTexture??this.velocityFallbackTexture,solver.gridPressureSamplesTexture??this.pressureSamplesFallbackTexture,solver.gridDivergenceTexture??this.scalarFallbackTexture,solver.gridPressureTexture??this.scalarFallbackTexture);this.secondaryParticlePipeline?.setSource(solver.secondaryParticles);this.voxelInspectionSource?.inspectionPublication?.setEnabled(false);this.voxelInspectionSource=undefined;this.voxelDebugPipeline?.setSource(undefined);this.voxelDebugSourceGeneration=-1;
      const sparseSceneSource=solver.sparseVoxelSceneSource;
      const scenePrimitives=buildSvoScenePrimitives(scene);
      const sceneGlass=buildSvoSceneGlass(scene,{cellSize_m:sparseSceneSource?.structural?.domain.cellSize_m});
      const sceneThickGlass=buildSvoSceneThickGlass(scene);
      const thickReplacedPaneKey=sceneThickGlass.metadata.find(({replacesThinPaneKey})=>Boolean(replacesThinPaneKey))?.replacesThinPaneKey;
      const thickReplacedPaneId=sceneGlass.metadata.find(({key})=>key===thickReplacedPaneKey)?.paneId;
      const terrainMaterial=scenePrimitives.analyticTerrain?buildSvoTerrainMaterial(scene):undefined;
      const compositorOwnedGlass=sceneGlass.metadata.filter(({role})=>role==="container-pane"||role==="container-top");
      const lightingMirrors=buildSparseVoxelDrySceneLightingMirrors(scene,sparseSceneSource);
      this.svoTerrainSupported=!scenePrimitives.requiresRasterTerrainFallback&&(!sceneHasTerrain(scene)||Boolean(scenePrimitives.analyticTerrain));
      const thickReplacedPaneKeys=new Set(sceneThickGlass.metadata.flatMap(({replacesThinPaneKey})=>replacesThinPaneKey?[replacesThinPaneKey]:[]));
      this.svoMaterialsSupported=canConsumeSparseVoxelPbrMaterials(sparseSceneSource);
      const assembledDrySceneData:SparseVoxelDrySceneData={
        primitiveRecords:scenePrimitives.packedRecords,
        ownerBase:scene.rigidBodies.length,skippedOwnerId:scenePrimitives.openShellOwnerId,
        terrainMaterialId:scenePrimitives.analyticTerrain?.materialId,terrainMaterialMetadata:terrainMaterial?.packedMetadata,terrainMaterialCacheKey:terrainMaterial?.cacheKey,
        glassRecords:sceneGlass.packedRecords,glassCacheKey:sceneGlass.cacheKey,
        thickGlassRecords:sceneThickGlass.packedRecords,thickGlassRevision:sceneThickGlass.revision,thickGlassCacheKey:sceneThickGlass.cacheKey,thickGlassReplacedThinPaneId:thickReplacedPaneId,
        primaryCompositeOwnedGlassPaneIdBase:compositorOwnedGlass[0]?.paneId,primaryCompositeOwnedGlassPaneCount:compositorOwnedGlass.length,
        ...lightingMirrors,
      };
      const thickGlassBound=resolveSparseVoxelThickGlassBinderStatus(assembledDrySceneData)==="bound";
      this.svoGlassSupported=!sceneGlass.metadata.some(({key,opaqueCutoutKey})=>Boolean(opaqueCutoutKey)&&(!thickGlassBound||!thickReplacedPaneKeys.has(key)));
      this.svoLightingSupported=Boolean(lightingMirrors)&&canConsumeSparseVoxelLighting(sparseSceneSource,assembledDrySceneData);
      const drySceneData:SparseVoxelDrySceneData|undefined=this.svoTerrainSupported&&this.svoGlassSupported&&this.svoMaterialsSupported&&this.svoLightingSupported?assembledDrySceneData:undefined;
      this.svoSourceAvailable=canEncodeSparseVoxelDryScene(sparseSceneSource,drySceneData);
      this.svoDrySceneSource=sparseSceneSource;this.svoDrySceneData=drySceneData;
      this.sceneryAnimation=drySceneData?svoScenePrimitiveAnimation(scenePrimitives):undefined;this.sceneryAnimationOrigin_ms=undefined;
      this.svoDryScenePipeline?.setSource(sparseSceneSource,drySceneData);
      if(staticRenderScene){
        if(this.failedOptionalPipelines.has("svo-dry-scene"))this.failPendingStaticSvoPresentation();
        else if(this.svoDryScenePipeline)this.reportStaticSvoAttachment();
        else{const pipelineProgress=this.svoPipelineProgress??{label:"Compiling sparse dry-scene pipeline",completed:0};this.onStatus({state:"initializing",...pipelineProgress,phase:"presentation",total:4,startedAt_ms,kind:"startup",retainingPrevious:false});}
      }
      this.pausedPresentationRevision+=1;
      if(previous&&previous!==solver&&!previousDestroyedForReset)this.retireGPUFluid(previous);
      this.gpuInfoCallback?.(solver.info);
      if(!staticRenderScene&&fencedInitialRaster)this.onStatus({state:"initializing",label:"Warmed solver attached; publishing fenced t=0 raster surface",phase:"presentation",completed:reportedCompleted,total:reportedTotal+1,startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:false});
      else if(!staticRenderScene)this.onStatus({state:"ready",label:"WebGPU direct-field solver ready",adapter:this.adapterName});
    }).catch((error:unknown)=>{if(this.disposed||generation!==this.gpuFluidRequestGeneration)return;this.gpuFluidPendingKey="";this.pendingInitialRasterPresentation=undefined;this.pendingStaticSvoPresentation=undefined;if(isGPUInitializationAbort(error))return;if(previous)this.onStatus({state:"ready",label:error instanceof Error?`Solver rebuild failed; previous solver retained: ${error.message}`:"Solver rebuild failed; previous solver retained",adapter:this.adapterName});else this.onStatus({state:"unavailable",label:error instanceof Error?`GPU initialization failed: ${error.message}`:"GPU initialization failed"});}).finally(()=>{if(generation===this.gpuFluidRequestGeneration){this.gpuFluidPending=undefined;if(this.gpuFluidInitializationAbort===abort)this.gpuFluidInitializationAbort=undefined;}});
  }

  private currentGPUFluid(scene: SceneDescription, config: SimulationRunConfig, time_s: number) {
    if (!this.device || this.disposed || this.deviceLost) return undefined;
    if (!canInitializeGPUSceneSource(scene, config.methodId)) return undefined;
    const key=this.solverKey(scene,config);
    if(!this.gpuFluid||key!==this.gpuFluidKey){
      // A change confined to the seed tier can re-seed the live solver instead
      // of rebuilding it. The attempt is fire-and-forget against a generation
      // guard; if it declines or the solver moved on, the ordinary rebuild
      // below still runs, so this can only make the path faster, never wrong.
      if(this.gpuFluid&&this.gpuFluidPendingKey!==key&&this.tryReseedGPUFluid(scene,config,key))return undefined;
      if(this.gpuFluidPendingKey!==key)this.beginGPUFluidInitialization(scene,config,key);
      return undefined;
    }
    // A timeline reset is represented by simulationEpoch in the key above.
    // Never turn a timestamp anomaly into an unplanned second solver build.
    if (time_s < (this.gpuFluid.info.submittedTime_s ?? 0)) return undefined;
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
        if (this.gpuFluidPendingKey !== rebuildKey) this.beginGPUFluidInitialization(scene, config, rebuildKey);
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
    const outcome = diagnosticsRequired && !diagnostics
      ? { ready: false, state: "failed-closed" as const,
          label: "t=0 raster publication failed closed: bounded diagnostics readback was unavailable" }
      : initialRasterPresentationReadiness({
          solverAttached: true,
          initialSparseAuthorityReady: pending.solver.initialSparseAuthorityReady === true,
          globalFineAttached: Boolean(pending.solver.globalFineLevelSetSource),
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
    this.gpuInfoCallback?.(pending.solver.info);
    this.pausedPresentationRevision += 1;
    if (outcome.ready) this.onStatus({ state: "ready", label: outcome.label, adapter: this.adapterName });
    else this.onStatus({ state: "blocked", label: outcome.label });
  }

  private settleStaticSvoPresentation(pending: PendingStaticSvoPresentation) {
    if (this.disposed || this.deviceLost || this.pendingStaticSvoPresentation !== pending
      || this.gpuFluid !== pending.solver || this.gpuFluidGeneration !== pending.solverGeneration
      || this.gpuFluidRequestGeneration !== pending.requestGeneration || !pending.attached || !pending.submitted) return;
    this.pendingStaticSvoPresentation = undefined;
    this.pausedPresentationRevision += 1;
    this.onStatus({ state: "ready", label: "Static SVO renderer ready", adapter: this.adapterName });
  }

  private submitPreparedGPUFluid(fluid: GPUSolverInstance, time_s: number, bodies: RigidBodyState[], maximumPendingAdvances = 1) {
    const device = this.device;
    if (!device) return fluid.info;
    this.preparedGPUTime_s = Math.max(this.preparedGPUTime_s, time_s);
    this.preparedGPUBodies = bodies;
    // A completion fence is the scheduling boundary. Encoding the entire debt
    // here can put hundreds of milliseconds of GPU work between presentations.
    if (!canQueuePreparedGPUAdvance(this.gpuPendingBatches, maximumPendingAdvances)) return fluid.info;
    const pendingAtSubmit = this.gpuPendingBatches;
    const nonPhysicsSequenceAtSubmit = this.nonPhysicsQueueWorkSequence;
    const queueWasIdleAtSubmit = pendingAtSubmit === 0
      && !this.presentationPending && this.diagnosticQueueWorkPending === 0;
    const { previousSubmittedTime, submittedTime } = submitNextPreparedGPUAdvance(fluid, this.preparedGPUTime_s, this.preparedGPUBodies);
    if (submittedTime > previousSubmittedTime) {
      const generation = this.gpuFluidGeneration;
      const submittedAt_ms = performance.now();
      this.physicsQueueWorkSequence += 1;
      this.gpuPendingBatches += 1;
      fluid.info.gpuPendingBatches = this.gpuPendingBatches;
      fluid.info.gpuInFlightSimulation_s = Math.max(0, submittedTime - (fluid.info.completedTime_s ?? 0));
      void device.queue.onSubmittedWorkDone().then(() => {
        if (this.disposed || this.deviceLost || this.gpuFluid !== fluid || this.gpuFluidGeneration !== generation) return;
        this.advanceWallEstimator.observeCompletion(performance.now(), submittedAt_ms, {
          pendingTargetWorkAtSubmit: pendingAtSubmit,
          interveningWorkSequence: nonPhysicsSequenceAtSubmit,
          queueWasIdleAtSubmit,
        });
        this.gpuPendingBatches = Math.max(0, this.gpuPendingBatches - 1);
        fluid.info.completedTime_s = Math.max(fluid.info.completedTime_s ?? 0, submittedTime);
        fluid.info.gpuPendingBatches = this.gpuPendingBatches;
        fluid.info.gpuInFlightSimulation_s = Math.max(0, (fluid.info.submittedTime_s ?? submittedTime) - fluid.info.completedTime_s);
        this.gpuInfoCallback?.({ ...fluid.info });
        this.gpuAdvanceCompletedCallback?.(submittedTime);
        this.continuePreparedGPUWork(fluid, generation);
      }).catch(() => { /* Device loss is reported by device.lost. */ });
    }
    // Functional diagnostics use a bounded cadence and remain independent of
    // the generic performance trace sampled by the solver itself.
    const now_ms=performance.now();if(now_ms-this.lastGPUInfoPollAt_ms>=250){
      this.lastGPUInfoPollAt_ms=now_ms;
      this.nonPhysicsQueueWorkSequence += 1;
      this.diagnosticQueueWorkPending += 1;
      void fluid.readStats().then(info=>this.gpuInfoCallback?.({...info}))
        .catch(()=>{ /* Device loss is reported by device.lost. */ })
        .finally(()=>{this.diagnosticQueueWorkPending=Math.max(0,this.diagnosticQueueWorkPending-1);});
    }
    return fluid.info;
  }

  /**
   * Refill the queue back to its rolling in-flight ceiling.
   *
   * An empty queue is never the right state for a GPU-limited solver: the next
   * refill would then have to wait for an animation-frame callback, which is
   * most of the gap between this lane and the Dawn harness. This used to gate
   * on presentation slack, but slack is a presentation-deadline guard, not
   * evidence about the queue — and it is unsatisfiable until a physics trace
   * exists, so an uninstrumented session never refilled at all. The ceiling
   * still bounds Reset's drain to the same depth the presentation path admits.
   */
  private continuePreparedGPUWork(fluid: GPUSolverInstance, generation: number) {
    if (this.disposed || this.deviceLost || this.gpuFluid !== fluid || this.gpuFluidGeneration !== generation) return;
    if (!this.simulationRunning) return;
    const depth = this.preparedGPUQueueDepth(fluid);
    if (this.gpuPendingBatches >= depth) return;
    this.submitPreparedGPUFluid(fluid, this.preparedGPUTime_s, this.preparedGPUBodies, depth);
  }

  /**
   * Re-pose the scene's authored scenery motion for this frame.
   *
   * One buffer write over one contiguous span of analytic primitive records:
   * the sparse world, its material owners and its baked occupancy mips are all
   * untouched, because the motion was bounded at authoring time to stay inside
   * the cell ownership they already describe (lib/scenery-sway.ts). The frame
   * therefore costs a few kilobytes and no re-voxelization, and the tree's
   * silhouette, normals, exact shadows and every distance-dependent light term
   * follow it. Its cone-marched soft shadow does not, and stays at the
   * reference pose the mip pyramid was built from.
   */
  private advanceSceneryAnimation(): void {
    const animation = this.sceneryAnimation;
    if (!animation || !this.svoDryScenePipeline) return;
    const now_ms = performance.now();
    this.sceneryAnimationOrigin_ms ??= now_ms;
    const time_s = Math.max(0, now_ms - this.sceneryAnimationOrigin_ms) / 1000;
    this.svoDryScenePipeline.updatePrimitiveRecords(
      packSvoScenePrimitiveAnimation(animation, time_s),
      animation.firstPrimitiveIndex,
    );
  }

  private uploadFluid(fluid?: EulerianRenderState) {
    if (!this.device || this.disposed || this.deviceLost || !fluid) return;
    const key = `${fluid.nx}x${fluid.ny}x${fluid.nz}`;
    if (key !== this.fluidTextureKey) {
      this.fluidTexture?.destroy();
      this.fluidTexture = this.device.createTexture({ label: "Eulerian occupied cells", size: [fluid.nx, fluid.ny, fluid.nz], dimension: "3d", format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.fluidTextureKey = key; this.fluidRevision = -1; this.updateRenderSources();
    }
    if (fluid.revision === this.fluidRevision || !this.fluidTexture) return;
    const bytesPerRow = Math.ceil(fluid.nx / 256) * 256;
    const packed = new Uint8Array(bytesPerRow * fluid.ny * fluid.nz);
    for (let k = 0; k < fluid.nz; k += 1) for (let j = 0; j < fluid.ny; j += 1) {
      const source = fluid.nx * (j + fluid.ny * k);
      packed.set(fluid.occupancy.subarray(source, source + fluid.nx), bytesPerRow * (j + fluid.ny * k));
    }
    this.device.queue.writeTexture({ texture: this.fluidTexture }, packed, { bytesPerRow, rowsPerImage: fluid.ny }, { width: fluid.nx, height: fluid.ny, depthOrArrayLayers: fluid.nz });
    this.fluidRevision = fluid.revision;
  }

  resize(renderScale = 1): void {
    if (this.disposed || this.deviceLost) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
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
    this.voxelDebugDepth?.destroy();
    this.presentationTexture = this.device.createTexture({label:"Water presentation target",size:[renderWidth,renderHeight],format:this.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
    this.voxelDebugDepth = this.device.createTexture({label:"Sparse voxel inspection depth",size:[renderWidth,renderHeight],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT});
    this.upscaleBindGroup=this.device.createBindGroup({layout:this.upscalePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this.presentationTexture.createView()},{binding:1,resource:this.upscaleSampler}]});
    this.presentationTextureKey=key;
    this.waterPipeline?.ensureSize(renderWidth, renderHeight);
    this.svoDryScenePipeline?.ensureSize(renderWidth, renderHeight);
  }

  get presentationResolution(): string {
    if (!this.presentationTexture) return `${this.canvas.width} × ${this.canvas.height}`;
    return `${this.presentationTexture.width} × ${this.presentationTexture.height} (${Math.round(this.activeRenderScale * 100)}%)`;
  }

  draw(time_s: number, scene: SceneDescription, camera: CameraState, bodies: RigidBodyState[], selectedBodyId: string | undefined, fluid: EulerianRenderState | undefined, backend: SimulationBackend, config: SimulationRunConfig, gridOverlay?: GridOverlayConfig, environmentId: EnvironmentId = defaultEnvironmentId, voxelRenderMode: VoxelRenderMode = "smooth", svoRenderMode: SvoRenderMode = DEFAULT_SVO_RENDER_MODE, svoLightingMode: SvoLightingMode = DEFAULT_SVO_LIGHTING_MODE, svoLightingOptions: SvoLightingOptions = DEFAULT_SVO_LIGHTING_OPTIONS, svoDiagnostics: SvoRenderDiagnostics = DEFAULT_SVO_RENDER_DIAGNOSTICS, svoTuning: SvoRenderTuning = DEFAULT_SVO_RENDER_TUNING, pixelTrace?: PixelTraceConfig): RendererFrameMetrics {
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
    this.svoTemporalAccumulationEnabled = activeSvoTuning.temporalEnabled
      && (typeof location === "undefined" || new URLSearchParams(location.search).get("svoTemporal") !== "0");
    this.resize(activeSvoTuning.resolutionScale);
    if (!this.presentationTexture || !this.upscalePipeline || !this.upscaleBindGroup) return this.currentFrameMetrics(config.methodId, config.methodId, false, cpuTrace?.finish());
    if (svoRenderMode !== "svo" || voxelRenderMode !== "smooth") { this.svoPickingAvailable = false; this.lastSvoPickingBodies = []; }
    const requestedSvoDiagnostics = normalizeSvoRenderDiagnostics(svoDiagnostics);
    const activeSvoDiagnostics = svoRenderMode !== "svo" || voxelRenderMode !== "smooth"
      ? DEFAULT_SVO_RENDER_DIAGNOSTICS : requestedSvoDiagnostics;
    const tuningKey = svoRenderTuningKey(activeSvoTuning);
    const diagnosticsKey = `${activeSvoDiagnostics.overlay}:${activeSvoDiagnostics.maximumTraversalDepth}:${activeSvoDiagnostics.maximumNodeVisits}:${tuningKey}`;
    if (diagnosticsKey !== this.svoRenderDiagnosticsKey) {
      this.svoRenderDiagnosticsKey = diagnosticsKey;
      this.svoDryScenePipeline?.invalidateTemporalHistory();
      this.resetPresentationTrace();
    }
    const presentationContext = `${config.methodId}:${config.quality}:shadow-${svoLightingOptions.shadowsEnabled ? "on" : "off"}:ao-${svoLightingOptions.ambientOcclusionEnabled ? "on" : "off"}:temporal-${this.svoTemporalAccumulationEnabled ? "on" : "off"}:lighting-${svoLightingMode}:${voxelRenderMode}:${svoRenderMode}:tuning-${tuningKey}:${this.simulationRunning ? "running" : "paused"}`;
    if (presentationContext !== this.presentationContext) {
      this.presentationContext = presentationContext;
      this.resetPresentationTrace();
    }
    const basis = cameraBasis(camera), position = basis.position;
    if (backend === "webgpu" && gridOverlay?.axis !== "off") this.gpuFluid?.ensureGridDiagnosticTextures?.();
    const sceneRuntime = planSceneRuntime(scene, { methodId: config.methodId, renderMode: svoRenderMode });
    const readyGPUFluid = backend === "webgpu" || !sceneRuntime.fluidSolver
      ? this.currentGPUFluid(scene, config, time_s)
      : undefined;
    const pixelTraceRequested = Boolean(pixelTrace) && svoRenderMode === "svo" && voxelRenderMode === "smooth";
    this.ensureRequestedOptionalPipelines(optionalRendererPipelineRequests(
      gridOverlay, voxelRenderMode, svoRenderMode, this.simulationRunning,
      Boolean((readyGPUFluid ?? this.gpuFluid)?.secondaryParticles),
      pixelTraceRequested,
    ));
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
    } else if (!pixelTraceRequested) {
      this.svoDryScenePipeline?.clearPixelTraceRequest();
      this.pixelTraceOverlayPipeline?.clear();
      if (this.latestPixelTraceValue) { this.latestPixelTraceValue = undefined; this.pixelTraceRevisionValue += 1; }
      this.pixelTraceGeometryKey = "";
    }
    // Raw voxel/brick inspection is opt-in. Keeping the source detached in
    // smooth presentation avoids a second capacity-sized GPU instance arena
    // (about 295 MB for the widened ocean) while SVO continues to consume the
    // structural source directly.
    const requestedVoxelDebugGeneration = voxelRenderMode !== "smooth" && this.gpuFluid
      ? this.gpuFluidGeneration
      : -1;
    if (requestedVoxelDebugGeneration !== this.voxelDebugSourceGeneration) {
      this.voxelInspectionSource?.inspectionPublication?.setEnabled(false);
      this.voxelInspectionSource = requestedVoxelDebugGeneration >= 0 ? this.gpuFluid?.sparseVoxelRenderSource : undefined;
      this.voxelInspectionSource?.inspectionPublication?.setEnabled(true);
      this.voxelDebugPipeline?.setSource(this.voxelInspectionSource);
      this.voxelDebugSourceGeneration = requestedVoxelDebugGeneration;
    }
    if (readyGPUFluid) { this.preparedGPUTime_s = Math.max(this.preparedGPUTime_s, time_s); this.preparedGPUBodies = bodies; }
    if (this.presentationPending) {
      // One presentation at a time, but physics keeps flowing. Returning here
      // without admitting work used to leave the device idle for the rest of
      // the interval, because the presentation fence covers the whole queue.
      if (readyGPUFluid) {
        if (pausedTargetRequiresGPUAdvance(this.simulationRunning, time_s, readyGPUFluid.info.submittedTime_s ?? 0)) {
          this.submitPreparedGPUFluid(readyGPUFluid, time_s, bodies);
        } else this.continuePreparedGPUWork(readyGPUFluid, this.gpuFluidGeneration);
      }
      return this.currentFrameMetrics(config.methodId, presentationContext, false, cpuTrace?.finish());
    }
    const observedStep_ms=this.advanceWallEstimator.estimate_ms;
    const renderBeforePhysics = backend === "webgpu" && !presentationHasPhysicsSlack(this.lastPresentationCompletedAt_ms, performance.now(), observedStep_ms, this.presentationWallEstimator.estimate_ms ?? 0);
    let gpuInfo = readyGPUFluid?.info;
    const explicitPausedAdvance = readyGPUFluid && pausedTargetRequiresGPUAdvance(this.simulationRunning, time_s, readyGPUFluid.info.submittedTime_s ?? 0);
    if (readyGPUFluid && (explicitPausedAdvance || !renderBeforePhysics)) gpuInfo = this.submitPreparedGPUFluid(readyGPUFluid, time_s, bodies);
    // The global fine narrow band double-buffers generations. Refresh its
    // tagged renderer binding after each admitted solver encode so extraction
    // follows the newly published generation without any CPU field copy.
    if (readyGPUFluid?.globalFineLevelSetSource) {
      this.waterPipeline.setGlobalFineLevelSet(createGlobalFineLevelSetConsumerSource(readyGPUFluid.globalFineLevelSetSource));
    }
    const globalFineWaterReady = Boolean(readyGPUFluid
      && readyGPUFluid.initialSparseAuthorityReady
      && readyGPUFluid.globalFineLevelSetSource);
    if (readyGPUFluid && globalFineWaterReady !== this.globalFineWaterAttached) {
      this.globalFineWaterAttached = globalFineWaterReady;
      this.updateRenderSources(
        globalFineWaterReady ? this.scalarFallbackTexture : readyGPUFluid.surfaceFieldTexture ?? readyGPUFluid.volumeTexture,
        readyGPUFluid.columnBaseTexture,
        readyGPUFluid.gridCellTexture ?? this.gridCellTexture,
        readyGPUFluid.velocityTexture ?? this.velocityFallbackTexture,
        readyGPUFluid.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture,
        readyGPUFluid.gridDivergenceTexture ?? this.scalarFallbackTexture,
        readyGPUFluid.gridPressureTexture ?? this.scalarFallbackTexture,
      );
    }
    if (gpuInfo && this.gpuFluid && this.columnBaseTexture && this.gridCellTexture && this.velocityFallbackTexture && this.pressureSamplesFallbackTexture && this.scalarFallbackTexture) {const compactSurface=Boolean(this.gpuFluid.globalFineLevelSetSource);this.gridOverlayPipeline?.setVolume(compactSurface?this.scalarFallbackTexture:this.gpuFluid.surfaceFieldTexture??this.gpuFluid.volumeTexture, this.gpuFluid.columnBaseTexture ?? this.columnBaseTexture, this.gpuFluid.gridCellTexture ?? this.gridCellTexture, this.gpuFluid.velocityTexture ?? this.velocityFallbackTexture, this.gpuFluid.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture, this.gpuFluid.gridDivergenceTexture ?? this.scalarFallbackTexture, this.gpuFluid.gridPressureTexture ?? this.scalarFallbackTexture);}
    cpuTrace?.transition({ id: "scene-upload", label: "Scene and field uploads" });
    if (backend === "cpu-reference") this.uploadFluid(fluid);
    const cameraStabilityKey = [
      basis.position.x, basis.position.y, basis.position.z,
      basis.forward.x, basis.forward.y, basis.forward.z,
      basis.right.x, basis.right.y, basis.right.z,
      basis.up.x, basis.up.y, basis.up.z,
    ].join("|");
    if (cameraStabilityKey !== this.svoCameraStabilityKey) {
      this.svoCameraStabilityKey = cameraStabilityKey;
      this.svoCameraStableFrames = 0;
    } else this.svoCameraStableFrames += 1;
    const shadowStabilityKey = [
      this.gpuFluidGeneration, scene.sceneId, scene.randomSeed, environmentId, diagnosticsKey, selectedBodyId ?? "",
      cameraStabilityKey,
      ...bodies.flatMap((body) => [
        body.description.id, body.description.shape,
        body.description.dimensions_m.x, body.description.dimensions_m.y, body.description.dimensions_m.z,
        body.position_m.x, body.position_m.y, body.position_m.z,
        body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z,
      ]),
    ].join("|");
    const checkerboardShadowsEligible = this.svoTemporalAccumulationEnabled && activeSvoTuning.checkerboardShadowsEnabled && svoLightingOptions.shadowsEnabled
      && svoRenderMode === "svo" && voxelRenderMode === "smooth" && requestedSvoDiagnostics.overlay === "off";
    if (!checkerboardShadowsEligible || shadowStabilityKey !== this.svoShadowStabilityKey) {
      this.svoShadowStabilityKey = checkerboardShadowsEligible ? shadowStabilityKey : "";
      this.svoShadowStableFrames = 0;
      this.svoDryScenePipeline?.invalidateTemporalHistory();
    } else this.svoShadowStableFrames += 1;
    const shadowTemporalFrame = svoShadowTemporalFrame(checkerboardShadowsEligible, this.svoShadowStableFrames, this.presentationFrameIndex);
    const drySceneTemporalFrame = svoDrySceneTemporalFrame(shadowTemporalFrame, this.svoCameraStableFrames);
    this.presentationFrameIndex += 1;
    const techniqueModeCode = gridOverlay?.mode && isOctreeTechniqueOverlayMode(gridOverlay.mode)
      ? OCTREE_TECHNIQUE_OVERLAY_CODES[gridOverlay.mode]
      : 0;
    // The compact pressure solve ping-pongs its row buffer. Refresh the
    // diagnostic bundle only while a technique view is visible so pressure
    // updates never keep a bind group pointed at the preceding solve bank.
    if (techniqueModeCode) {
      this.techniqueOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    }
    const uniform = new Float32Array([
      this.presentationTexture.width, this.presentationTexture.height, time_s, drySceneTemporalFrame,
      position.x, position.y, position.z, svoCostOverlayCode(activeSvoDiagnostics.overlay),
      camera.target_m.x, camera.target_m.y, camera.target_m.z, 0,
      scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction,
      // options.w carries the largest represented adaptive pressure-cell
      // width. The grid overlay uses it to normalize its categorical scale
      // palette to the hierarchy that can actually exist in this solver.
      activeSvoDiagnostics.maximumTraversalDepth * 512 + activeSvoDiagnostics.maximumNodeVisits, scene.voxelDomain.finestCellSize_m, Math.min(bodies.length, 12), gpuInfo?.quadtreeMaximumFluidScale ?? 1,
      // Field mode: 1 = raw occupancy, 2 = packed tall-cell level set,
      // 3 = uniform-layout level set (quadtree resident phi).
      gpuInfo?.nx ?? fluid?.nx ?? 1, gpuInfo?.ny ?? fluid?.ny ?? 1, gpuInfo?.nz ?? fluid?.nz ?? 1, gpuInfo ? (gpuInfo.gridKind === "restricted-tall-cell" ? 2 : gpuInfo.gridKind === "quadtree-tall-cell" || gpuInfo.gridKind === "octree" ? 3 : 1) : fluid ? 1 : 0,
      gridOverlay?.axis === "z" ? 1 : gridOverlay?.axis === "x" ? 2 : gridOverlay?.axis === "y" ? 3 : gridOverlay?.axis === "volume" ? 4 : 0, gridOverlay?.position ?? 0.5, gpuInfo?.gridKind === "quadtree-tall-cell" || gpuInfo?.gridKind === "octree" ? 1 : 0,
      techniqueModeCode || (gridOverlay?.mode === "cfl" ? 1 : gridOverlay?.mode === "speed" ? 2 : gridOverlay?.mode === "phi" ? 3 : gridOverlay?.mode === "divergence" ? 4 : gridOverlay?.mode === "pressure" ? 5 : gridOverlay?.mode === "representation" ? 6 : gridOverlay?.mode === "optical" ? 7 : gridOverlay?.mode === "projection" && gpuInfo?.gridKind === "octree" ? 8 : gridOverlay?.mode === "resolution" && (gpuInfo?.gridKind === "quadtree-tall-cell" || gpuInfo?.gridKind === "octree") ? 9 : 0),
      environmentIndex(environmentId), gpuInfo?.lastDt_s ?? 0, gpuInfo?.maxSpeed_m_s ?? 0,
      gpuInfo?.gridKind === "quadtree-tall-cell" ? (gpuInfo.quadtreeOpticalLayerMode === "adaptive-motion" ? 2 : 1) : 0
    ]);
    // Terrain heightfield mirror for the environment shaders: meta lane plus
    // two vec4 lanes per feature, matching lib/terrain.ts semantics exactly.
    const packed = new Float32Array(100);
    packed.set(uniform, 0);
    if (sceneHasTerrain(scene) && scene.terrain) {
      const terrain = scene.terrain;
      const features = terrain.features.slice(0, MAX_TERRAIN_FEATURES);
      packed.set([1, terrain.baseHeight_m, features.length, TERRAIN_UNION_EXPONENT], 32);
      features.forEach((feature, index) => {
        packed.set([feature.center_m.x, feature.center_m.z, feature.radius_m.x, feature.radius_m.z], 36 + index * 8);
        packed.set([(feature.kind === "mound" ? 1 : -1) * feature.amount_m, feature.rotation_rad ?? 0, feature.flat ?? TERRAIN_DEFAULT_FLAT, 0], 40 + index * 8);
      });
    }
    this.device.queue.writeBuffer(this.uniformBuffer, 0, packed);
    const residentRigidBuffer = backend === "webgpu" ? this.gpuFluid?.rigidRenderBuffer : undefined;
    if (residentRigidBuffer) {
      this.gpuFluid?.setSelectedRigidBody?.(bodies.findIndex((body) => body.description.id === selectedBodyId));
    } else {
      const bodyData = new Float32Array(12 * 16);
      const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
      const palette = [[0.95, 0.63, 0.29], [0.48, 0.66, 0.96], [0.84, 0.42, 0.48], [0.66, 0.52, 0.92]];
      bodies.slice(0, 12).forEach((body, index) => {
        const offset = index * 16;
        const d = body.description.dimensions_m;
        const half = body.description.shape === "box" ? [d.x / 2, d.y / 2, d.z / 2] : body.description.shape === "sphere" ? [d.x, d.x, d.x] : [d.x, d.y / 2, d.x];
        const color = palette[shapeIndex[body.description.shape]];
        bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, boundingRadius(body)], offset);
        bodyData.set([half[0], half[1], half[2], shapeIndex[body.description.shape]], offset + 4);
        bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
        bodyData.set([color[0], color[1], color[2], body.description.id === selectedBodyId ? 1 : 0], offset + 12);
      });
      this.device.queue.writeBuffer(this.bodyBuffer, 0, bodyData);
    }
    this.advanceSceneryAnimation();
    this.svoDryScenePipeline?.setLightingMode(svoLightingMode);
    // Reduced-rate cone lighting is the production default: half-resolution
    // prepass + guided upsample, measured within the visibility-error gates.
    this.svoDryScenePipeline?.setLightingOptions({ ...svoLightingOptions, coneLightingScale: activeSvoTuning.coneLightingScale });
    this.svoDryScenePipeline?.setRenderTuning(activeSvoTuning);
    cpuTrace?.transition({ id: "command-encoding", label: "Presentation command encoding" });
    const traceRequestedAt_ms = measurementInstrumentationEnabled ? performance.now() : 0;
    const shouldTracePresentation = measurementInstrumentationEnabled
      && !this.presentationTracePending
      && traceRequestedAt_ms - this.lastPresentationTraceAt_ms >= 250;
    const presentationTraceSampleId = shouldTracePresentation
      ? ++this.presentationTraceSampleId
      : 0;
    const traceDetailedSvoRenderPath = svoRenderMode === "svo"
      && voxelRenderMode === "smooth"
      && svoLightingMode === "cone";
    const presentationTrace = shouldTracePresentation
      && !this.hardwarePresentationTraceInvalid
      && GPUStageTimestampRecorder.supported(this.device)
      ? new GPUStageTimestampRecorder(
        this.device,
        presentationTraceSampleId,
        "presentation",
        presentationContext,
        64,
      )
      : undefined;
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
    const encoder = presentationTrace?.instrument(rawEncoder) ?? rawEncoder;
    presentationTrace?.begin();
    const detailedPresentationTrace = traceDetailedSvoRenderPath ? presentationTrace : undefined;
    // The SVO cone path names its own stages; every other path walks the fixed
    // presentation partition in encode order.
    let fixedPresentationPhase = 0;
    const closeFixedPresentationPhase = () => {
      const phase = PRESENTATION_TRACE_PHASES[fixedPresentationPhase];
      fixedPresentationPhase += 1;
      if (phase && !traceDetailedSvoRenderPath) presentationTrace?.completePhase(encoder, phase);
    };
    const completeDetailedPresentationPhase = (phase: GPUTimestampPhase) =>
      detailedPresentationTrace?.completePhase(encoder, phase);
    // A raw/brick toggle while paused still needs one fresh materialization;
    // regular solver encodes clear this pending request, avoiding duplication.
    this.voxelInspectionSource?.inspectionPublication?.encodePending?.(encoder);
    if (residentRigidBuffer) encoder.copyBufferToBuffer(residentRigidBuffer, 0, this.bodyBuffer, 0, 12 * 16 * 4);
    this.svoDryScenePipeline?.setRigidMotionSource(backend === "webgpu" ? this.gpuFluid?.rigidMotionBuffer : undefined);
    this.svoDryScenePipeline?.setFluidCoverage(this.ensureFluidCoverage(readyGPUFluid, scene));
    this.secondaryParticlePipeline?.setSource(backend === "webgpu" ? this.gpuFluid?.secondaryParticles : undefined);
    let svoEncoded = false;
    const useSvoDryScene = svoRenderMode === "svo" && voxelRenderMode === "smooth";
    if (!useSvoDryScene) this.svoDryScenePipeline?.invalidateTemporalHistory();
    const drySceneReplacement = useSvoDryScene
      ? (replacementEncoder: GPUCommandEncoder, target: GPUTexture | GPUTextureView, tracePhase?: (phase: GPUTimestampPhase) => void) => {
        const cellSize_m = this.svoDryScenePipeline?.temporalCellSize_m ?? 0;
        const temporalFrame: SparseVoxelTemporalFrameState | undefined = this.svoTemporalAccumulationEnabled ? {
          camera: {
            position_m: [basis.position.x, basis.position.y, basis.position.z],
            forward: [basis.forward.x, basis.forward.y, basis.forward.z],
            right: [basis.right.x, basis.right.y, basis.right.z],
            up: [basis.up.x, basis.up.y, basis.up.z],
          },
          deltaTime_s: this.simulationRunning ? gpuInfo?.lastDt_s ?? 0 : 0,
          cellSize_m,
          paused: !this.simulationRunning,
          composition: "dry-before-raster-water",
          maximumSamples: activeSvoTuning.temporalMaximumSamples,
          varianceSigma: activeSvoTuning.temporalVarianceSigma,
          depthToleranceScale: activeSvoTuning.temporalDepthToleranceScale,
        } : undefined;
        if (!temporalFrame) this.svoDryScenePipeline?.invalidateTemporalHistory();
        // SVO visibility and shading are presentation work, not an on-change
        // cache. Execute them for every submitted frame so dynamic sparse
        // publications, lighting, and temporal sampling can never be hidden
        // behind a host-computed scene key.
        const replacementResult = this.svoDryScenePipeline?.encode(replacementEncoder, target, temporalFrame, undefined, tracePhase) ?? false;
        svoEncoded = Boolean(replacementResult);
        if (!replacementResult) this.svoDryScenePipeline?.invalidateTemporalHistory();
        return replacementResult;
      }
      : undefined;
    this.waterPipeline.setSceneHasFluid(Boolean(sceneRuntime.fluidSolver));
    // Sparse presentation owns the dry scene for the whole session. Frames it
    // cannot publish yet show the environment's own ambient light, never the
    // legacy procedural room, which is a different set from the authored one.
    // A scene that has actually fallen back keeps the raster room, because
    // there it is the whole picture rather than a placeholder.
    const svoPresentationExpected = useSvoDryScene
      && !this.failedOptionalPipelines.has("svo-dry-scene")
      && this.svoTerrainSupported && this.svoGlassSupported && this.svoMaterialsSupported
      && this.svoLightingSupported;
    this.waterPipeline.setPendingSvoBackground(
      svoPresentationExpected ? svoEnvironmentAmbientBackgroundLinear(environmentId, scene.lighting?.environment) : undefined,
    );
    // Before the dry pass samples it, and outside the water pipeline's own
    // passes so the volume is complete for the whole frame.
    this.svoFluidCoverage?.encode(encoder);
    const rasterResult = this.waterPipeline.encode(
      encoder, this.presentationTexture,
      gpuInfo?.nx ?? fluid?.nx ?? 1, gpuInfo?.ny ?? fluid?.ny ?? 1, gpuInfo?.nz ?? fluid?.nz ?? 1,
      gpuInfo?.gridKind === "restricted-tall-cell", gpuInfo?.maximumNeighborDelta ?? 0,
      gpuInfo?.encodedSteps ?? fluid?.revision ?? 0,
      drySceneReplacement,
      closeFixedPresentationPhase,
      detailedPresentationTrace ? completeDetailedPresentationPhase : undefined,
    );
    if (!rasterResult) throw new Error("Raster optics pipeline is not ready");
    const pendingInitialRaster = this.pendingInitialRasterPresentation;
    const initialRasterSubmission = pendingInitialRaster
      && !pendingInitialRaster.submitted
      && pendingInitialRaster.solver === readyGPUFluid
      && readyGPUFluid.initialSparseAuthorityReady === true
      && Boolean(readyGPUFluid.globalFineLevelSetSource)
      && this.globalFineWaterAttached
      && rasterResult.surfaceUpdated
      ? pendingInitialRaster
      : undefined;
    if (initialRasterSubmission) initialRasterSubmission.submitted = true;
    const pendingStaticSvo = this.pendingStaticSvoPresentation;
    const initialStaticSvoSubmission = pendingStaticSvo
      && !pendingStaticSvo.submitted
      && pendingStaticSvo.attached
      && pendingStaticSvo.solver === readyGPUFluid
      && svoEncoded
      ? pendingStaticSvo
      : undefined;
    if (initialStaticSvoSubmission) {
      initialStaticSvoSubmission.submitted = true;
      this.onStatus({
        state: "initializing", label: "Submitting first sparse garden frame", phase: "presentation",
        completed: 3, total: 4, startedAt_ms: initialStaticSvoSubmission.startedAt_ms, kind: "startup", retainingPrevious: false,
      });
    }
    this.svoPickingAvailable = useSvoDryScene && svoEncoded;
    this.lastSvoPickingBodies = this.svoPickingAvailable ? bodies.slice(0, 12) : [];
    this.publishEffectiveRendererStatus(resolveEffectiveRendererStatus(svoRenderMode, {
      pipelineAvailable: this.svoPipelineAvailable,
      sourceAvailable: this.svoSourceAvailable,
      terrainSupported: this.svoTerrainSupported,
      glassSupported: this.svoGlassSupported,
      materialsSupported: this.svoMaterialsSupported,
      lightingSupported: this.svoLightingSupported,
      inspectionMode: voxelRenderMode !== "smooth",
      svoEncoded,
    }));
    let inspectionOverlayEncoded = false;
    if (voxelRenderMode !== "smooth" && this.voxelDebugDepth) {
      const sceneExtent = Math.hypot(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
      this.voxelDebugPipeline?.encode(encoder, {
        mode: voxelRenderMode,
        colorTarget: this.presentationTexture.createView(),
        depthTarget: this.voxelDebugDepth.createView(),
        depthLoadOp: "clear",
        // Inspection is a representation switch, not a subtle overlay. Clear
        // the smooth hybrid frame so contiguous voxels and brick bounds remain
        // unmistakable even for a still, fully filled region.
        colorLoadOp: "clear",
        viewProjection: voxelViewProjectionMatrix(camera, this.presentationTexture.width / Math.max(1, this.presentationTexture.height), 0.01, camera.distance_m + sceneExtent * 3),
        cameraPosition: [position.x, position.y, position.z],
        containerBounds: {
          min: [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2],
          max: [scene.container.width_m / 2, scene.container.height_m, scene.container.depth_m / 2]
        },
        containerClosedTop: scene.container.top === "closed",
        exposure: 1,
        gridOpacity: 0.88
      });
      inspectionOverlayEncoded = true;
    }
    if (gridOverlay && gridOverlay.axis !== "off") {
      const overlayView=this.presentationTexture.createView();
      // Generic texture fields and compact paper publications each own both
      // their slice and ray-integrated volume presentation.
      if(!techniqueModeCode)this.gridOverlayPipeline?.encode(encoder,overlayView);
      if(techniqueModeCode){
        this.techniqueOverlayPipeline?.encode(encoder,overlayView,techniqueModeCode);
        this.techniqueAuditOverlayPipeline?.encode(encoder,overlayView,techniqueModeCode);
      }
      inspectionOverlayEncoded = true;
    }
    if (inspectionOverlayEncoded) {
      completeDetailedPresentationPhase({ id: "inspection-overlay", label: "Inspection overlays" });
    }
    if (pixelTraceRequested && pixelTrace) {
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
      }
      this.encodePixelTraceOverlay(encoder, basis, pixelTrace);
    }
    closeFixedPresentationPhase();
    const upscalePass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0.01,g:0.025,b:0.024,a:1},loadOp:"clear",storeOp:"store"}]});
    upscalePass.setPipeline(this.upscalePipeline);upscalePass.setBindGroup(0,this.upscaleBindGroup);upscalePass.draw(3);upscalePass.end();
    completeDetailedPresentationPhase({ id: "present", label: "Final upscale + present" });
    closeFixedPresentationPhase();
    // A frame that skipped part of the raster path would leave the remaining
    // stages named by position rather than by the work they contain. Drop that
    // sample to the queue-wall observation instead of publishing a mislabelled
    // partition — and without retiring the recorder, which is still healthy.
    const hardwarePresentationTrace = traceDetailedSvoRenderPath
      || fixedPresentationPhase === PRESENTATION_TRACE_PHASES.length
      ? presentationTrace
      : undefined;
    if (hardwarePresentationTrace) hardwarePresentationTrace.resolve(encoder);
    else presentationTrace?.destroy();
    presentationQueueTrace?.begin();
    const presentationSubmittedAt_ms=performance.now();
    const presentationQueueWasIdleAtSubmit=this.gpuPendingBatches===0
      && this.diagnosticQueueWorkPending===0;
    const physicsSequenceAtPresentationSubmit=this.physicsQueueWorkSequence;
    this.device.queue.submit([encoder.finish()]);
    this.nonPhysicsQueueWorkSequence+=1;
    this.presentationPending=true;
    if (pixelTraceProbing) this.pumpPixelTraceReadback();
    const presentationQueueTraceRead = presentationQueueTrace?.read(this.device.queue);
    const presentationTraceRead = hardwarePresentationTrace
      ? hardwarePresentationTrace.read()
        .then((trace) => {
          this.hardwarePresentationTraceInvalid = !trace;
          return trace ?? presentationQueueTraceRead;
        })
        .catch(() => {
          this.hardwarePresentationTraceInvalid = true;
          return presentationQueueTraceRead;
        })
      : presentationQueueTraceRead;
    if (presentationTraceRead) {
      this.lastPresentationTraceAt_ms = traceRequestedAt_ms;
      this.presentationTracePending = true;
      const sampledContext = presentationContext;
      void presentationTraceRead.then((trace) => {
        const instrumentation = usePerformanceInstrumentationStore.getState();
        if (!trace || this.disposed || this.deviceLost || this.presentationContext !== sampledContext
          || !instrumentation.enabled || instrumentation.enabledAt_ms > traceRequestedAt_ms) return;
        this.latestPresentationTrace = trace;
        if (!this.simulationRunning) this.pausedPresentationRevision += 1;
      }).catch(() => {
        hardwarePresentationTrace?.destroy();
      }).finally(() => {
        this.presentationTracePending = false;
      });
    }
    const surfaceDiagnosticsRequired = true;
    const surfaceDiagnosticsCompletion = this.waterPipeline.completeSurfaceDiagnostics();
    const presentationDevice=this.device;
    void this.device.queue.onSubmittedWorkDone().then(async()=>{
      if(this.disposed||this.deviceLost||this.device!==presentationDevice)return;
      const completedAt_ms=performance.now();
      this.presentationWallEstimator.observeCompletion(completedAt_ms,presentationSubmittedAt_ms,{
        pendingTargetWorkAtSubmit:0,
        interveningWorkSequence:physicsSequenceAtPresentationSubmit,
        queueWasIdleAtSubmit:presentationQueueWasIdleAtSubmit,
      });
      this.presentationPending=false;this.lastPresentationCompletedAt_ms=completedAt_ms;
      if(initialStaticSvoSubmission)this.settleStaticSvoPresentation(initialStaticSvoSubmission);
      if(initialRasterSubmission){
        const initialDiagnostics=await surfaceDiagnosticsCompletion;
        this.settleInitialRasterPresentation(initialRasterSubmission,surfaceDiagnosticsRequired,initialDiagnostics);
      }
      if(this.gpuFluid)this.continuePreparedGPUWork(this.gpuFluid,this.gpuFluidGeneration);
    }).catch(()=>{this.presentationPending=false;});
    if(readyGPUFluid&&this.simulationRunning){
      cpuTrace?.transition({ id: "other", label: "Post-submit physics scheduling" });
      const postPresentationDepth=this.preparedGPUQueueDepth(readyGPUFluid);
      // postPresentationDepth is a ceiling, not an increment. Adding the
      // current pending count here admitted another full window every frame,
      // so slow 16/32-leaf solvers accumulated seconds of work that Reset then
      // had to drain before it could replace the solver.
      const maximumPendingAdvances=postPresentationDepth;
      for(let queued=0;queued<postPresentationDepth;queued+=1){
        const before=readyGPUFluid.info.submittedTime_s??0;
        this.submitPreparedGPUFluid(readyGPUFluid,time_s,bodies,maximumPendingAdvances);
        if((readyGPUFluid.info.submittedTime_s??0)<=before)break;
      }
    }
    return this.currentFrameMetrics(config.methodId, presentationContext, true, cpuTrace?.finish());
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    const fluid = this.gpuFluid;
    this.gpuFluid = undefined;
    this.pendingInitialRasterPresentation = undefined;
    this.pendingStaticSvoPresentation = undefined;
    this.svoPickingAvailable = false;
    this.lastSvoPickingBodies = [];
    this.gpuFluidRequestGeneration += 1;
    this.gpuFluidInitializationAbort?.abort();
    this.gpuFluidInitializationAbort = undefined;
    this.gpuFluidPendingKey = "";
    this.resetGPUQueueTracking();
    this.gpuFluidGeneration += 1;
    try { fluid?.destroy(); } catch { /* Device loss can invalidate solver resources first. */ }
    for (const retired of this.retiredGPUFluids) { try { retired.destroy(); } catch { /* Best-effort cleanup after device loss. */ } }
    this.retiredGPUFluids.clear();
    try { this.waterPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.gridOverlayPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.voxelDebugPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.svoDryScenePipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    for (const resource of [this.presentationTexture, this.voxelDebugDepth, this.fluidTexture, this.columnBaseTexture, this.gridCellTexture, this.velocityFallbackTexture, this.pressureSamplesFallbackTexture, this.scalarFallbackTexture, this.uniformBuffer, this.bodyBuffer]) {
      try { resource?.destroy(); } catch { /* Best-effort cleanup during hot reload. */ }
    }
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
