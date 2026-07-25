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
import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";
import { buildSparseVoxelDrySceneLightingMirrors, canConsumeSparseVoxelLighting, canConsumeSparseVoxelPbrMaterials, canConsumeSparseVoxelPrimitiveCandidates, canEncodeSparseVoxelDryScene, resolveSparseVoxelThickGlassBinderStatus, SparseVoxelDrySceneRenderer, type SparseVoxelDrySceneData } from "./webgpu-svo-dry-scene";
import { buildSvoScenePrimitives } from "./svo-scene-primitives";
import { buildSvoSceneGlass } from "./svo-scene-glass";
import { buildSvoSceneThickGlass } from "./svo-scene-thick-glass";
import { buildSvoTerrainMaterial } from "./svo-terrain-material";
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
import { isGPUInitializationAbort } from "./gpu-initialization";
import { createGlobalFineLevelSetConsumerSource } from "./octree-consumer-sampling";
import { OCTREE_TECHNIQUE_OVERLAY_CODES, isOctreeTechniqueOverlayMode, type OctreeTechniqueOverlayMode } from "./octree-technique-debug";
import { OctreeTechniqueOverlayPipeline } from "./webgpu-octree-technique-overlay";
import { automaticGPURecoveryEnabled, performanceTraceDeviceFeatures } from "./gpu-startup";
import { OctreeTechniqueAuditOverlayPipeline } from "./webgpu-octree-technique-audit-overlay";
import { initialRasterPresentationReadiness, requiresFencedInitialRasterPresentation } from "./gpu-t0-presentation";
import { WebGPUStaticSvoScene } from "./webgpu-static-svo-scene";
import { planSceneRuntime } from "./scene-runtime";
import type { GPUFailureReproduction } from "./webgpu-failure-reproduction";
import {
  CPUPerformanceTrace,
  DynamicGPUPerformanceTraceRecorder,
  GPUPerformanceTraceRecorder,
  GPUQueueWallPerformanceTraceRecorder,
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

/** Use the exact generic physics partition as the rolling scheduler estimate. */
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
  | "secondary-particles";

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
): OptionalRendererPipeline[] {
  const requested: OptionalRendererPipeline[] = [];
  if (gridOverlay && gridOverlay.axis !== "off") {
    if (gridOverlay.axis !== "volume") requested.push("grid-overlay");
    if (gridOverlay.mode && isOctreeTechniqueOverlayMode(gridOverlay.mode)) {
      requested.push("technique-overlay", "technique-audit-overlay");
    }
  }
  if (voxelRenderMode !== "smooth") requested.push("voxel-debug");
  if (svoRenderMode === "svo" && voxelRenderMode === "smooth") requested.push("svo-dry-scene");
  if (simulationRunning && secondaryParticlesAvailable) requested.push("secondary-particles");
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

/** Content identity for every construction-time input captured by the GPU scene source. */
export function gpuSceneSolverKey(scene: SceneDescription, config: SimulationRunConfig): string {
  return `${config.simulationEpoch ?? 0}:fluid-${planSceneRuntime(scene, { methodId: config.methodId }).fluidSolver}:${config.methodId}:${config.quality}:${JSON.stringify(structuralMethodValues(config))}:${scene.environment ?? "default"}:${JSON.stringify(scene.lighting ?? null)}:${JSON.stringify(scene.voxelDomain)}:${scene.container.width_m}:${scene.container.height_m}:${scene.container.depth_m}:${scene.container.fillFraction}:${scene.container.top}:${scene.container.fluidWallMode}:${JSON.stringify(scene.rigidBodies)}:${scene.fluid.initialCondition}:${JSON.stringify(scene.fluid.initialBrickSeeds_m ?? null)}:${scene.fluid.density_kg_m3}:${scene.fluid.dynamicViscosity_Pa_s}:${scene.fluid.surfaceTension_N_m}:${scene.fluid.gravity_m_s2.y}:${JSON.stringify(scene.fluid.inflow ?? null)}`;
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
  | "missing-primitive-candidates"
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
  primitiveCandidatesSupported?: boolean;
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
  if (conditions.primitiveCandidatesSupported === false) return { requestedMode, effectiveMode: "raster", fallbackReason: "missing-primitive-candidates" };
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
  private gridOverlayPipeline?: GridOverlayPipeline;
  private techniqueOverlayPipeline?: OctreeTechniqueOverlayPipeline;
  private techniqueAuditOverlayPipeline?: OctreeTechniqueAuditOverlayPipeline;
  /** Optional programs compile only after their explicit presentation mode is used. */
  private readonly optionalPipelineTasks = new Map<OptionalRendererPipeline, Promise<void>>();
  /** A compile failure is sticky for this device; do not hammer a fragile driver every frame. */
  private readonly failedOptionalPipelines = new Set<OptionalRendererPipeline>();
  private presentationTexture?: GPUTexture;
  private voxelDebugDepth?: GPUTexture;
  private presentationTextureKey = "";
  private activeRenderScale = 1;
  private readonly rasterRenderScale = 0.72;
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
  private svoPrimitiveCandidatesSupported = true;
  private svoLightingSupported = true;
  private svoPipelineAvailable = false;
  /** Internal A/B: temporal-off also restores full-rate shadow visibility. */
  private readonly svoTemporalAccumulationEnabled = typeof location === "undefined" || new URLSearchParams(location.search).get("svoTemporal") !== "0";
  private presentationFrameIndex = 0;
  private svoCameraStabilityKey = "";
  private svoCameraStableFrames = 0;
  private svoShadowStabilityKey = "";
  private svoShadowStableFrames = 0;
  private svoRenderDiagnosticsKey = "";
  private gpuPendingBatches = 0;
  private lastPresentationCompletedAt_ms = -Infinity;
  private presentationPending = false;
  private simulationRunning = true;
  private preparedGPUTime_s = 0;
  private preparedGPUBodies: RigidBodyState[] = [];
  private gpuFluidGeneration = 0;
  /** True only while both compact t=0 raster sources are attached. */
  private globalFineWaterAttached = false;
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
    const requiredFeatures: GPUFeatureName[] = [
      ...performanceTraceDeviceFeatures(adapter.features),
    ];
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
    this.optionalPipelineTasks.clear(); this.failedOptionalPipelines.clear(); this.svoDrySceneSource = undefined; this.svoDrySceneData = undefined; this.svoPipelineProgress = undefined; this.svoPipelineStartedAt_ms = undefined; this.pendingStaticSvoPresentation = undefined;
    this.svoPipelineAvailable = false; this.svoSourceAvailable = false; this.svoTerrainSupported = true; this.svoGlassSupported = true; this.svoMaterialsSupported = true; this.svoPrimitiveCandidatesSupported = true; this.svoLightingSupported = true;
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

  private resetGPUQueueTracking() {
    this.gpuPendingBatches = 0;
    this.preparedGPUTime_s = 0;
    this.preparedGPUBodies = [];
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
        this.svoDrySceneSource=undefined;this.svoDrySceneData=undefined;this.svoDryScenePipeline?.setSource(undefined,undefined);
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
      this.gpuFluid=solver;this.gpuFluidKey=key;this.gpuFluidPendingKey="";this.resetGPUQueueTracking();this.gpuFluidGeneration+=1;this.lastGPUInfoPollAt_ms=-Infinity;this.globalFineWaterAttached=false;
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
      const candidateDrySceneData:SparseVoxelDrySceneData={
        primitiveRecords:scenePrimitives.packedRecords,primitiveCandidates:scenePrimitives.primitiveCandidates,
        ownerBase:scene.rigidBodies.length,skippedOwnerId:scenePrimitives.openShellOwnerId,
        terrainMaterialId:scenePrimitives.analyticTerrain?.materialId,terrainMaterialMetadata:terrainMaterial?.packedMetadata,terrainMaterialCacheKey:terrainMaterial?.cacheKey,
        glassRecords:sceneGlass.packedRecords,glassCacheKey:sceneGlass.cacheKey,
        thickGlassRecords:sceneThickGlass.packedRecords,thickGlassRevision:sceneThickGlass.revision,thickGlassCacheKey:sceneThickGlass.cacheKey,thickGlassReplacedThinPaneId:thickReplacedPaneId,
        primaryCompositeOwnedGlassPaneIdBase:compositorOwnedGlass[0]?.paneId,primaryCompositeOwnedGlassPaneCount:compositorOwnedGlass.length,
        ...lightingMirrors,
      };
      const thickGlassBound=resolveSparseVoxelThickGlassBinderStatus(candidateDrySceneData)==="bound";
      this.svoGlassSupported=!sceneGlass.metadata.some(({key,opaqueCutoutKey})=>Boolean(opaqueCutoutKey)&&(!thickGlassBound||!thickReplacedPaneKeys.has(key)));
      this.svoPrimitiveCandidatesSupported=canConsumeSparseVoxelPrimitiveCandidates(candidateDrySceneData);
      this.svoLightingSupported=Boolean(lightingMirrors)&&canConsumeSparseVoxelLighting(sparseSceneSource,candidateDrySceneData);
      const drySceneData:SparseVoxelDrySceneData|undefined=this.svoTerrainSupported&&this.svoGlassSupported&&this.svoMaterialsSupported&&this.svoPrimitiveCandidatesSupported&&this.svoLightingSupported?candidateDrySceneData:undefined;
      this.svoSourceAvailable=canEncodeSparseVoxelDryScene(sparseSceneSource,drySceneData);
      this.svoDrySceneSource=sparseSceneSource;this.svoDrySceneData=drySceneData;
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
    if(!this.gpuFluid||key!==this.gpuFluidKey){if(this.gpuFluidPendingKey!==key)this.beginGPUFluidInitialization(scene,config,key);return undefined;}
    // A timeline reset is represented by simulationEpoch in the key above.
    // Never turn a timestamp anomaly into an unplanned second solver build.
    if (time_s < (this.gpuFluid.info.submittedTime_s ?? 0)) return undefined;
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
    const { previousSubmittedTime, submittedTime } = submitNextPreparedGPUAdvance(fluid, this.preparedGPUTime_s, this.preparedGPUBodies);
    const advanceCompletion = fluid.pendingAdvanceCompletion;
    if (submittedTime > previousSubmittedTime) {
      const generation = this.gpuFluidGeneration;
      this.gpuPendingBatches += 1;
      fluid.info.gpuPendingBatches = this.gpuPendingBatches;
      fluid.info.gpuInFlightSimulation_s = Math.max(0, submittedTime - (fluid.info.completedTime_s ?? 0));
      void (advanceCompletion ?? device.queue.onSubmittedWorkDone()).then(() => {
        if (this.disposed || this.deviceLost || this.gpuFluid !== fluid || this.gpuFluidGeneration !== generation) return;
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
    const now_ms=performance.now();if(now_ms-this.lastGPUInfoPollAt_ms>=250){this.lastGPUInfoPollAt_ms=now_ms;void fluid.readStats().then(info=>this.gpuInfoCallback?.({...info})).catch(()=>{ /* Device loss is reported by device.lost. */ });}
    return fluid.info;
  }

  /** Keep the GPU occupied with a rolling advance, but yield at the 60 Hz presentation boundary. */
  private continuePreparedGPUWork(fluid: GPUSolverInstance, generation: number) {
    if (this.disposed || this.deviceLost || this.gpuFluid !== fluid || this.gpuFluidGeneration !== generation) return;
    if (!this.simulationRunning || this.gpuPendingBatches > 0 || this.presentationPending) return;
    const observedStep_ms=observedGPUAdvanceTime_ms(fluid.info.physicsTrace);
    if (!presentationHasPhysicsSlack(this.lastPresentationCompletedAt_ms, performance.now(), observedStep_ms, this.latestPresentationTrace?.total_ms ?? 0)) return;
    this.submitPreparedGPUFluid(fluid, this.preparedGPUTime_s, this.preparedGPUBodies);
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

  draw(time_s: number, scene: SceneDescription, camera: CameraState, bodies: RigidBodyState[], selectedBodyId: string | undefined, fluid: EulerianRenderState | undefined, backend: SimulationBackend, config: SimulationRunConfig, gridOverlay?: GridOverlayConfig, environmentId: EnvironmentId = defaultEnvironmentId, voxelRenderMode: VoxelRenderMode = "smooth", svoRenderMode: SvoRenderMode = DEFAULT_SVO_RENDER_MODE, svoLightingMode: SvoLightingMode = DEFAULT_SVO_LIGHTING_MODE, svoLightingOptions: SvoLightingOptions = DEFAULT_SVO_LIGHTING_OPTIONS, svoDiagnostics: SvoRenderDiagnostics = DEFAULT_SVO_RENDER_DIAGNOSTICS): RendererFrameMetrics {
    const measurementInstrumentationEnabled = usePerformanceInstrumentationStore.getState().enabled;
    const cpuTrace = measurementInstrumentationEnabled
      ? new CPUPerformanceTrace(
        ++this.cpuTraceSampleId,
        `${config.methodId}:${config.quality}`,
        { id: "frame-control", label: "Frame control + physics submission" },
      )
      : undefined;
    if (!this.device || this.disposed || this.deviceLost || !this.context || !this.uniformBuffer || !this.bodyBuffer || !this.waterPipeline) return this.currentFrameMetrics(config.methodId, config.methodId, false, cpuTrace?.finish());
    this.resize(this.rasterRenderScale);
    if (!this.presentationTexture || !this.upscalePipeline || !this.upscaleBindGroup) return this.currentFrameMetrics(config.methodId, config.methodId, false, cpuTrace?.finish());
    if (svoRenderMode !== "svo" || voxelRenderMode !== "smooth") { this.svoPickingAvailable = false; this.lastSvoPickingBodies = []; }
    const requestedSvoDiagnostics = normalizeSvoRenderDiagnostics(svoDiagnostics);
    const activeSvoDiagnostics = requestedSvoDiagnostics.overlay === "off" || svoRenderMode !== "svo" || voxelRenderMode !== "smooth"
      ? DEFAULT_SVO_RENDER_DIAGNOSTICS : requestedSvoDiagnostics;
    const diagnosticsKey = `${activeSvoDiagnostics.overlay}:${activeSvoDiagnostics.maximumTraversalDepth}:${activeSvoDiagnostics.maximumNodeVisits}:${activeSvoDiagnostics.overlayOpacity}`;
    if (diagnosticsKey !== this.svoRenderDiagnosticsKey) {
      this.svoRenderDiagnosticsKey = diagnosticsKey;
      this.svoDryScenePipeline?.invalidateTemporalHistory();
      this.resetPresentationTrace();
    }
    const presentationContext = `${config.methodId}:${config.quality}:shadow-${svoLightingOptions.shadowsEnabled ? "on" : "off"}:ao-${svoLightingOptions.ambientOcclusionEnabled ? "on" : "off"}:temporal-${this.svoTemporalAccumulationEnabled ? "on" : "off"}:lighting-${svoLightingMode}:${voxelRenderMode}:${svoRenderMode}:${this.simulationRunning ? "running" : "paused"}`;
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
    this.ensureRequestedOptionalPipelines(optionalRendererPipelineRequests(
      gridOverlay, voxelRenderMode, svoRenderMode, this.simulationRunning,
      Boolean((readyGPUFluid ?? this.gpuFluid)?.secondaryParticles),
    ));
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
    if (this.presentationPending) return this.currentFrameMetrics(config.methodId, presentationContext, false, cpuTrace?.finish());
    const observedStep_ms=measurementInstrumentationEnabled?observedGPUAdvanceTime_ms(readyGPUFluid?.info.physicsTrace):undefined;
    const renderBeforePhysics = backend === "webgpu" && !presentationHasPhysicsSlack(this.lastPresentationCompletedAt_ms, performance.now(), observedStep_ms, measurementInstrumentationEnabled ? this.latestPresentationTrace?.total_ms ?? 0 : 0);
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
    const checkerboardShadowsEligible = this.svoTemporalAccumulationEnabled && svoLightingOptions.shadowsEnabled
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
    const uniform = new Float32Array([
      this.presentationTexture.width, this.presentationTexture.height, time_s, drySceneTemporalFrame,
      position.x, position.y, position.z, svoCostOverlayCode(activeSvoDiagnostics.overlay),
      camera.target_m.x, camera.target_m.y, camera.target_m.z, activeSvoDiagnostics.overlayOpacity,
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
    this.svoDryScenePipeline?.setLightingMode(svoLightingMode);
    // Reduced-rate cone lighting is the production default: half-resolution
    // prepass + guided upsample, measured within the visibility-error gates.
    this.svoDryScenePipeline?.setLightingOptions({ ...svoLightingOptions, coneLightingScale: 0.5 });
    cpuTrace?.transition({ id: "command-encoding", label: "Presentation command encoding" });
    const traceRequestedAt_ms = measurementInstrumentationEnabled ? performance.now() : 0;
    const encoder = this.device.createCommandEncoder({ label: "Fluid Lab frame" });
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
      && !traceDetailedSvoRenderPath
      && GPUPerformanceTraceRecorder.supported(this.device)
      ? new GPUPerformanceTraceRecorder(
        this.device,
        presentationTraceSampleId,
        "presentation",
        presentationContext,
        PRESENTATION_TRACE_PHASES,
      )
      : undefined;
    const detailedPresentationTrace = shouldTracePresentation
      && traceDetailedSvoRenderPath
      && GPUPerformanceTraceRecorder.supported(this.device)
      ? new DynamicGPUPerformanceTraceRecorder(
        this.device,
        presentationTraceSampleId,
        "presentation",
        presentationContext,
        32,
      )
      : undefined;
    const presentationQueueTrace = shouldTracePresentation
      ? new GPUQueueWallPerformanceTraceRecorder(
        presentationTraceSampleId,
        "presentation",
        presentationContext,
      )
      : undefined;
    presentationTrace?.boundary(encoder, "Presentation trace start");
    detailedPresentationTrace?.begin(encoder);
    const completeDetailedPresentationPhase = (phase: GPUTimestampPhase) =>
      detailedPresentationTrace?.completePhase(encoder, phase);
    // A raw/brick toggle while paused still needs one fresh materialization;
    // regular solver encodes clear this pending request, avoiding duplication.
    this.voxelInspectionSource?.inspectionPublication?.encodePending?.(encoder);
    if (residentRigidBuffer) encoder.copyBufferToBuffer(residentRigidBuffer, 0, this.bodyBuffer, 0, 12 * 16 * 4);
    this.svoDryScenePipeline?.setRigidMotionSource(backend === "webgpu" ? this.gpuFluid?.rigidMotionBuffer : undefined);
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
    const rasterResult = this.waterPipeline.encode(
      encoder, this.presentationTexture,
      gpuInfo?.nx ?? fluid?.nx ?? 1, gpuInfo?.ny ?? fluid?.ny ?? 1, gpuInfo?.nz ?? fluid?.nz ?? 1,
      gpuInfo?.gridKind === "restricted-tall-cell", gpuInfo?.maximumNeighborDelta ?? 0,
      gpuInfo?.encodedSteps ?? fluid?.revision ?? 0,
      drySceneReplacement,
      () => presentationTrace?.boundary(encoder, "Presentation trace boundary"),
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
      primitiveCandidatesSupported: this.svoPrimitiveCandidatesSupported,
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
      // The legacy grid pass is a planar inspector. Full-volume paper modes
      // ray-integrate their compact structures in the technique pass itself.
      if(gridOverlay.axis!=="volume")this.gridOverlayPipeline?.encode(encoder,overlayView);
      if(techniqueModeCode){
        this.techniqueOverlayPipeline?.encode(encoder,overlayView,techniqueModeCode);
        this.techniqueAuditOverlayPipeline?.encode(encoder,overlayView,techniqueModeCode);
      }
      inspectionOverlayEncoded = true;
    }
    if (inspectionOverlayEncoded) {
      completeDetailedPresentationPhase({ id: "inspection-overlay", label: "Inspection overlays" });
    }
    presentationTrace?.boundary(encoder, "Presentation overlays complete");
    const upscalePass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0.01,g:0.025,b:0.024,a:1},loadOp:"clear",storeOp:"store"}]});
    upscalePass.setPipeline(this.upscalePipeline);upscalePass.setBindGroup(0,this.upscaleBindGroup);upscalePass.draw(3);upscalePass.end();
    completeDetailedPresentationPhase({ id: "present", label: "Final upscale + present" });
    presentationTrace?.boundary(encoder, "Presentation trace end");
    presentationTrace?.resolve(encoder);
    detailedPresentationTrace?.resolve(encoder);
    presentationQueueTrace?.begin();
    this.device.queue.submit([encoder.finish()]);
    const presentationQueueTraceRead = presentationQueueTrace?.read(this.device.queue);
    const hardwarePresentationTrace = detailedPresentationTrace ?? presentationTrace;
    const presentationTraceRead = hardwarePresentationTrace
      ? hardwarePresentationTrace.read()
        .then((trace) => trace ?? presentationQueueTraceRead)
        .catch(() => presentationQueueTraceRead)
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
    this.presentationPending=true;
    const presentationDevice=this.device;
    void this.device.queue.onSubmittedWorkDone().then(async()=>{
      if(this.disposed||this.deviceLost||this.device!==presentationDevice)return;
      const completedAt_ms=performance.now();
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
      const observedPostPresentationStep_ms=measurementInstrumentationEnabled?observedGPUAdvanceTime_ms(readyGPUFluid.info.physicsTrace):undefined;
      const postPresentationDepth=presentationPhysicsQueueDepth(observedPostPresentationStep_ms,measurementInstrumentationEnabled?this.latestPresentationTrace?.total_ms??0:0);
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
