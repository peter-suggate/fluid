import { WebGpuSvoFluidCoverage } from "./webgpu-svo-fluid-coverage";
import { cameraBasis, dot } from "./math";
import { sceneLatticeDimensions } from "./scene-lattice";
import { canonicalScene, type CameraState, type SceneDescription } from "./model";
import { boundingRadius, type RigidBodyState } from "./rigid-body";
import type { EulerianRenderState } from "./eulerian-solver";
import type { GPUEulerianInfo, GPURigidLoad, GPUQuality } from "./webgpu-eulerian";
import { getMethod, type GPUSolverInstance, type MethodParamValues } from "./methods";
import { GridOverlayPipeline } from "./webgpu-grid-overlay";
import { FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE, requiredFluidDeviceLimits } from "./webgpu-device-limits";
import { RasterWaterPipeline, type WaterRenderDiagnostics, type WaterSurfacePresentationDiagnostics } from "./webgpu-water-pipeline";
import { environmentIndex, type EnvironmentId, defaultEnvironmentId } from "./environments";
import { MAX_TERRAIN_FEATURES, TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT, sceneHasTerrain } from "./terrain";
import { SecondaryParticleRenderPipeline } from "./webgpu-secondary-particles";
import { SparseVoxelDebugRenderer, type SparseVoxelRenderSource, type SparseVoxelSceneRenderSource, type VoxelRenderMode } from "./webgpu-voxel-debug";
import { CAMERA_TAN_HALF_FOV, viewportAspect, viewportRayForPixel } from "./webgpu-camera";
import {
  type SvoPixelTrace,
  type SvoPixelTraceLayer,
} from "./svo-pixel-trace";
import { DecorationOverlay } from "./webgpu-decoration-overlay";
import { VISUALIZATION_CATALOG } from "./visualization-catalog";
import { assembleDecorations } from "./visualization-registry";
import { containerDecorationSpace } from "./visualization-decorations";
import { WebGPUFluidCellTrace } from "./webgpu-fluid-cell-trace";
import type { FluidCellTrace } from "./fluid-cell-trace";
import type { FineBandCellContext } from "./fine-band-cell-model";
import { buildSparseVoxelDrySceneLightingMirrors, canConsumeSparseVoxelLighting, packSvoDrySceneTerrainHeightfield, resolveSparseVoxelThickGlassBinderStatus, sparseVoxelDrySceneContractFailure, SparseVoxelDrySceneRenderer, SVO_DRY_SCENE_REVERSED_Z_NEAR_M, SVO_PRESENTATION_STARTUP_STAGES, svoPresentationResourcePlugin, type SparseVoxelDrySceneData, type SvoDryRigidBounds, type SvoDrySceneDirtyBounds } from "./webgpu-svo-dry-scene";
import {
  buildSvoScenePrimitives,
} from "./svo-scene-primitives";
import { buildSvoPrimitiveCandidates, refitSvoPrimitiveCandidates, svoPrimitiveCandidateBounds, type SvoPrimitiveCandidatePublication } from "./svo-primitive-candidates";
import { packSvoPrimitiveRecords, type SvoPrimitiveDescriptor } from "./svo-primitive-abi";
import { swayedPrimitiveDescriptor, type EnvironmentProxySway } from "./scenery-sway";
import { sparseScenePrimitiveForSvoDescriptor } from "./webgpu-sparse-scene-proxies";
import {
  buildDefaultSvoMaterialRecords,
  packSvoMaterialTable,
  svoMaterialFromEnvironmentProxyMaterial,
  svoMaterialFunctionIdForEnvironmentProxy,
} from "./svo-material-abi";
import { buildSvoSceneGlass } from "./svo-scene-glass";
import { buildSvoSceneThickGlass } from "./svo-scene-thick-glass";
import { buildSvoTerrainMaterial, sceneTerrainSurfaceModel } from "./svo-terrain-material";
import {
  DEFAULT_SVO_LIGHTING_OPTIONS,
  type SvoLightingOptions,
  type SvoLightingVisibilityStatus,
  type SvoPrimaryTraversalMode,
  type SvoSilhouetteRefinementStatus,
} from "./svo-render-options";
import { DEFAULT_SVO_RENDER_DIAGNOSTICS, normalizeSvoRenderDiagnostics, type SvoRenderDiagnostics } from "./svo-render-diagnostics";
import { SparseVoxelRenderStageOverlay } from "./webgpu-svo-stage-overlay";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, svoRenderTuningKey, type SvoRenderTuning } from "./svo-render-tuning";
import { isGPUInitializationAbort } from "./gpu-initialization";
import { createGlobalFineLevelSetConsumerSource } from "./octree-consumer-sampling";
import { OCTREE_TECHNIQUE_OVERLAY_CODES, isOctreeTechniqueOverlayMode, type OctreeTechniqueOverlayMode } from "./octree-technique-debug";
import { OctreeTechniqueOverlayPipeline } from "./webgpu-octree-technique-overlay";
import {
  automaticGPURecoveryEnabled,
  fluidExecutionDeviceFeatures,
} from "./gpu-startup";
import { OctreeTechniqueAuditOverlayPipeline } from "./webgpu-octree-technique-audit-overlay";
import { initialRasterPresentationReadiness, requiresFencedInitialRasterPresentation } from "./gpu-t0-presentation";
import { liveSvoSceneResourcePlugin, WebGPULiveSvoScene } from "./webgpu-live-svo-scene";
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
import type { ResourcePluginDefinition } from "./resource-readiness";
import {
  invalidateGPUCompilationManager,
  managedGPUDevice,
} from "./gpu-compilation-manager";

/** Device/canvas resources owned by FluidLabRenderer, declared beside their initializer. */
export const webGPUPlatformResourcePlugin: ResourcePluginDefinition = Object.freeze({
  id: "platform.webgpu-renderer",
  lane: "platform",
  label: "WebGPU renderer platform",
  provides: ["renderer"] as const,
  blocks: "viewport",
  phaseCopy: {
    planning: "Acquiring the browser GPU and selecting device capabilities.",
    renderer: "Preparing the canvas and minimum presentation resources.",
    "water-renderer": "Compiling rasterized water interfaces and optical compositing.",
  },
});

export type SimulationBackend = "webgpu" | "cpu-reference";
/** One item executing and one queued keeps the GPU busy without visible FIFO bursts. */
export const BROWSER_GPU_THROUGHPUT_DEPTH = 2;
export const SVO_CAMERA_CHANGING_FRAME = -2;

/**
 * Exact old/new coverage for each procedurally moving analytic primitive.
 *
 * The list stays per primitive rather than becoming one scene-sized union, so
 * the unified sparse publisher can map a small branch motion to only its local
 * pages. Intermediate poses are not render generations; old and new bounds are
 * the complete dependency set for this atomic presentation update.
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
    if (before.kind === "terrain-heightfield" || after.kind === "terrain-heightfield") continue;
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
  | "decoration-overlay"
  | "fluid-cell-trace"
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
  voxelRenderMode: VoxelRenderMode,
  simulationRunning: boolean,
  secondaryParticlesAvailable: boolean,
  pixelTraceActive = false,
  fluidCellTraceActive = false,
  stageViewActive = false,
  sparsePresentationRequired = true,
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
  if (sparsePresentationRequired) requested.push("svo-dry-scene");
  if (simulationRunning && secondaryParticlesAvailable) requested.push("secondary-particles");
  // The trace overlay is only meaningful over the sparse path it explains.
  if (pixelTraceActive && voxelRenderMode === "smooth") requested.push("decoration-overlay");
  // The cell gather reads published octree topology, not the presentation, so
  // it is requested independently of the voxel representation in use.
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
  // The lattice is keyed in cells, not in metres. Scaling the world multiplies
  // the container extents and the finest cell size by the same factor, so the
  // cell counts, every arena sized from them, and every compiled pipeline are
  // untouched — the structural tier must not claim otherwise and force a
  // rebuild for what is really a re-seed at a new scale. Authored domain bounds
  // are measured the same way, in cells, so they survive a scale too.
  const cellSize_m = scene.voxelDomain.finestCellSize_m;
  const bounds = scene.voxelDomain.bounds_m;
  const boundsCells = bounds
    ? [bounds.min.x, bounds.min.y, bounds.min.z, bounds.max.x, bounds.max.y, bounds.max.z]
      .map((value) => Math.round(value / cellSize_m)).join(",")
    : "none";
  const lattice = `${sceneLatticeDimensions(scene).join("x")}:${scene.voxelDomain.brickSize_cells}:${boundsCells}`;
  return `fluid-${planSceneRuntime(scene).fluidSolver}:${lattice}:${scene.container.top}:${scene.container.fluidWallMode}`;
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
  return `${c.width_m}:${c.height_m}:${c.depth_m}:${c.fillFraction}:${rigidBodyAllocationKey(scene.rigidBodies)}:${scene.fluid.initialCondition}:${JSON.stringify(scene.fluid.initialDamBreakDimensions_m ?? null)}:${JSON.stringify(scene.fluid.initialDamBreakOrigin_m ?? null)}:${JSON.stringify(scene.fluid.initialBrickSeeds_m ?? null)}:${scene.fluid.initialBrickSeedsAdditive ?? false}:${JSON.stringify(scene.terrain ?? null)}:${inflowBudgetKey(scene.fluid.inflow)}`;
}

/**
 * The rigid-body facts an allocation, a compiled policy, or the owner numbering
 * is shaped by.
 *
 * *Which* bodies, where they are, and what shape they take are absent on
 * purpose: every step already writes the whole roster into the rigid state
 * buffer through `syncBodies`, and the solid vertex SDF is re-encoded from that
 * buffer, so moving, reshaping or re-materialing a body is adopted live. That
 * is what keeps `rigidBodyRosterKey` in the uniform tier.
 *
 * What remains is three facts, and the third is the interesting one:
 *
 *  - Are there any solids? `octreeSparseWorldRequired`,
 *    `planOctreeSolidCellAllocation` and `planOctreeSurfaceStateAllocation`
 *    each allocate a dense solid field for the whole lattice or skip it.
 *  - Does any of them move? `planOctreeSolveTail` scores one extra iteration
 *    for a moving immersed boundary, regardless of how many there are.
 *  - **How many are there?** Not an allocation — a numbering.
 *    `svo-scene-primitives`, `svo-scene-thick-glass`, the sparse brick world
 *    and the dry-scene publication all number environment proxies as
 *    `rigidBodies.length + ownerIndex`, so the roster length is part of the
 *    render source's ABI. Adding a body renumbers every scenery object, which
 *    is why the count is here and why adding one still costs a re-seed.
 *
 * That third one is a decoupling worth doing and is not done: pinning the
 * environment base at the fixed rigid capacity (`GPU_RIGID_BODY_CAPACITY`, 12)
 * would take the count out of this key entirely, at the cost of resizing every
 * per-owner arena from `bodies.length + proxies` to `12 + proxies`. Until then,
 * "drop a body into running water" is warm only for a body that already exists.
 */
function rigidBodyAllocationKey(bodies: SceneDescription["rigidBodies"]): string {
  return `${bodies.length}:${bodies.some((body) => body.motion !== "static")}`;
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
export function sceneEditRequiresReset(before: SceneDescription, after: SceneDescription): boolean {
  return sceneStructuralKey(before) !== sceneStructuralKey(after)
    || gpuSceneSeedKey(before) !== gpuSceneSeedKey(after);
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
  return `${scene.fluid.density_kg_m3}:${scene.fluid.dynamicViscosity_Pa_s}:${scene.fluid.surfaceTension_N_m}:${scene.fluid.gravity_m_s2.y}:${inflowAimKey(scene.fluid.inflow)}:${rigidBodyRosterKey(scene.rigidBodies)}`;
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
      resource?: ResourcePluginDefinition;
    }
  | { state: "ready"; label: string; adapter: string; resource?: ResourcePluginDefinition }
  | { state: "blocked"; label: string; resource?: ResourcePluginDefinition }
  | { state: "manual"; label: string; resource?: ResourcePluginDefinition }
  | { state: "stopping"; label: string; resource?: ResourcePluginDefinition }
  | { state: "unavailable"; label: string; reproduction?: GPUFailureReproduction; resource?: ResourcePluginDefinition }
  | { state: "lost"; label: string; resource?: ResourcePluginDefinition };

export type SvoRendererFailureReason =
  | "missing-source"
  | "unsupported-terrain"
  | "unsupported-glass-cutout"
  | "missing-pbr-materials"
  | "missing-lighting-publications"
  | "pipeline-compile-failure"
  | "pipeline-compiling"
  | "frame-rejected";

export interface EffectiveRendererStatus {
  state: "active" | "not-required" | "pending" | "failed";
  failureReason?: SvoRendererFailureReason;
  /** Exact publication-contract cause; never used to select another renderer. */
  detail?: string;
  /** The requested refinement lifecycle is independent and never a fallback selector. */
  silhouetteRefinement?: SvoSilhouetteRefinementStatus;
  /** Requested versus effective shadows/AO/GI path. */
  lightingVisibility?: SvoLightingVisibilityStatus;
}

export interface EffectiveRendererConditions {
  /** False when the authoritative water renderer completely owns this scene. */
  required?: boolean;
  pipelineAvailable: boolean;
  /** A compile is in flight, so an absent pipeline is a rebuild rather than a failure. */
  pipelineCompiling?: boolean;
  /** Exact constructor/compile rejection retained for a fail-closed device. */
  pipelineFailure?: string;
  /** Exact requested-bundle transition while the owning pipeline remains attached. */
  pipelinePending?: string;
  sourceAvailable: boolean;
  terrainSupported: boolean;
  glassSupported?: boolean;
  materialsSupported?: boolean;
  lightingSupported?: boolean;
  svoEncoded: boolean;
  contractFailure?: string;
  silhouetteRefinement?: SvoSilhouetteRefinementStatus;
  lightingVisibility?: SvoLightingVisibilityStatus;
}

/** Resolve one frame's production renderer without changing simulation state. */
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

export function resolveEffectiveRendererStatus(
  conditions: EffectiveRendererConditions,
): EffectiveRendererStatus {
  const status = (renderer: Omit<EffectiveRendererStatus, "silhouetteRefinement" | "lightingVisibility">): EffectiveRendererStatus => ({
    ...renderer,
    ...(conditions.silhouetteRefinement ? { silhouetteRefinement: conditions.silhouetteRefinement } : {}),
    ...(conditions.lightingVisibility ? { lightingVisibility: conditions.lightingVisibility } : {}),
  });
  if (conditions.required === false) return status({ state: "not-required" });
  // An absent pipeline means two very different things. Startup and a primary
  // traversal swap both retire it while the replacement compiles, and reporting
  // that as a compile failure tells the user their renderer broke when it is
  // merely busy.
  if (!conditions.pipelineAvailable) {
    return status(conditions.pipelineCompiling
      ? { state: "pending", failureReason: "pipeline-compiling" }
      : {
          state: "failed",
          failureReason: "pipeline-compile-failure",
          ...(conditions.pipelineFailure ? { detail: conditions.pipelineFailure } : {}),
        });
  }
  if (!conditions.terrainSupported) return status({ state: "failed", failureReason: "unsupported-terrain" });
  if (conditions.glassSupported === false) return status({ state: "failed", failureReason: "unsupported-glass-cutout" });
  if (conditions.materialsSupported === false) return status({ state: "failed", failureReason: "missing-pbr-materials" });
  if (conditions.lightingSupported === false) return status({ state: "failed", failureReason: "missing-lighting-publications" });
  if (!conditions.sourceAvailable) {
    const sourceMissing = !conditions.contractFailure
      || conditions.contractFailure === "live sparse source is not attached";
    if (sourceMissing) return status(conditions.contractFailure
      ? { state: "pending", failureReason: "missing-source", detail: conditions.contractFailure }
      : { state: "pending", failureReason: "missing-source" });
    return status({ state: "failed", failureReason: "frame-rejected", detail: conditions.contractFailure });
  }
  if (!conditions.svoEncoded && conditions.pipelineCompiling) return status({
    state: "pending", failureReason: "pipeline-compiling",
    ...(conditions.pipelinePending ? { detail: conditions.pipelinePending } : {}),
  });
  if (!conditions.svoEncoded && conditions.pipelineFailure) return status({
    state: "failed", failureReason: "pipeline-compile-failure", detail: conditions.pipelineFailure,
  });
  if (!conditions.svoEncoded) return status({
    state: "failed", failureReason: "frame-rejected",
    detail: conditions.contractFailure ?? "live SVO renderer declined the frame",
  });
  return status({ state: "active" });
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
  { id: "dry-scene", label: "Dry scene lighting" },
  { id: "water-interfaces", label: "Front/back water interfaces" },
  { id: "optical-composite", label: "Optical composite" },
  { id: "inspection-overlay", label: "Inspection overlays" },
  { id: "present", label: "Final presentation" },
] as const;

interface PendingInitialRasterPresentation {
  readonly solver: GPUSolverInstance;
  readonly solverGeneration: number;
  readonly requestGeneration: number;
  readonly resource?: ResourcePluginDefinition;
  submitted: boolean;
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
  /** Live render publication identity; deliberately independent of solver identity. */
  private renderSceneKey = "";
  private renderSceneRevision = 0;
  private liveSceneAnimation?: {
    readonly rest: readonly SvoPrimitiveDescriptor[];
    readonly sway: readonly (EnvironmentProxySway | undefined)[];
    readonly keys: readonly string[];
    current: readonly SvoPrimitiveDescriptor[];
    candidates: SvoPrimitiveCandidatePublication;
    origin_ms?: number;
  };
  /** Deduplicates a visible failed-closed diagnostic while the previous pose remains authoritative. */
  private liveSceneAnimationFailure?: string;
  private svoPublicationFailure?: string;
  private gridOverlayPipeline?: GridOverlayPipeline;
  private techniqueOverlayPipeline?: OctreeTechniqueOverlayPipeline;
  private techniqueAuditOverlayPipeline?: OctreeTechniqueAuditOverlayPipeline;
  private decorationOverlayPipeline?: DecorationOverlay;
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
   * user-facing toggle is recorded here and takes effect by retiring the
   * pipeline: the next `ensureOptionalPipeline` sweep rebuilds against it.
   */
  private requestedPrimaryTraversal: SvoPrimaryTraversalMode =
    DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal ?? "raster";
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
  private presentationsInFlight = 0;
  private completedPresentations = 0;
  private simulationRunning = true;
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
  /** A sparse presentation becomes ready only after its first source-matched frame completes. */
  private pendingLiveSvoPresentation?: PendingLiveSvoPresentation;
  private svoPipelineProgress?: { label: string; completed: number; total: number };
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
  /** CSS viewport supplied by the main thread when this renderer owns an
   * OffscreenCanvas. HTML canvases continue to derive it locally. */
  private workerViewport?: { width: number; height: number; devicePixelRatio: number };

  get presentationRevision(): number { return this.pausedPresentationRevision; }

  constructor(private readonly canvas: HTMLCanvasElement | OffscreenCanvas, private readonly onStatus: (status: GPUStatus) => void, onGPUInfo?: (info: GPUEulerianInfo) => void, onGPURigidLoads?: (loads: GPURigidLoad[]) => void, onGPUAdvanceCompleted?: (time_s: number) => void, onEffectiveRendererStatus?: (status: EffectiveRendererStatus) => void) { this.gpuInfoCallback = onGPUInfo; this.gpuRigidLoadCallback = onGPURigidLoads; this.gpuAdvanceCompletedCallback = onGPUAdvanceCompleted; this.effectiveRendererStatusCallback = onEffectiveRendererStatus; }

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
      && previous.lightingVisibility?.detail === status.lightingVisibility?.detail) return;
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
   */
  private applyPrimaryTraversalRequest(requested: SvoPrimaryTraversalMode): void {
    if (requested === this.requestedPrimaryTraversal) return;
    this.requestedPrimaryTraversal = requested;
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
      // Relight already pays a full-resolution material/BRDF pass. Isolating
      // primary traversal raises Metal occupancy substantially. Exact-keyed
      // coherence can then retain that G-buffer for static/paused dry scenes.
      // Canonical-parametric removes the wide/canonical hybrid cursor and was
      // revalidated against split full-res relighting: 12.55 ms versus
      // 22.10-23.60 ms hybrid at 660x662 with identical output hashes.
      // Raster-primary replaces the full-screen traversal megakernel with a
      // rasterized brick-proxy pass: 4.89 ms of GPU against the megakernel's
      // 26.95 ms for primary visibility at 1500x1500 garden, whole frame
      // 49.62 -> 29.01 ms. It needs four depth-tested colour planes; a request
      // that the device cannot honor fails through the optional-pipeline
      // diagnostic instead of silently selecting another traversal.
      (device) => {
        if (this.requestedPrimaryTraversal === "raster"
          && device.limits.maxColorAttachmentBytesPerSample < FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE) {
          throw new RangeError(
            `Requested SVO raster primary needs maxColorAttachmentBytesPerSample >= ${FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE}; device exposes ${device.limits.maxColorAttachmentBytesPerSample}`,
          );
        }
        const traversal = this.requestedPrimaryTraversal === "raster"
          ? "raster-primary" as const : "canonical-parametric" as const;
        // Stationary primary reuse buys whatever the primary costs, and the two
        // traversals price that very differently. Skipping the megakernel is
        // worth 28.5 ms of a 49.6 ms frame; skipping the raster primary is worth
        // 7.9 ms of 29.0, and the
        // rigid impostor pass blocks the reuse anyway whenever the scene has
        // bodies. Asking for it there would only make a dead mode look live.
        const coherence = traversal === "raster-primary" ? "off" as const : "static-primary" as const;
        return new SparseVoxelDrySceneRenderer(device, this.uniformBuffer!, this.bodyBuffer!, "rgba16float",
          traversal, "off", "split", 0, coherence, true, true, true);
      },
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
    scene: SceneDescription,
    pixelTrace: PixelTraceConfig | undefined,
    fluidCellTrace: FluidCellTraceConfig | undefined,
  ): void {
    const overlay = this.decorationOverlayPipeline;
    if (!overlay?.ready || !this.presentationTexture) return;
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
    if (subjects.length === 0) {
      if (this.decorationGeometryKey !== "") { overlay.clear(); this.decorationGeometryKey = ""; }
      return;
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
    if (assembled.key !== this.decorationGeometryKey) {
      overlay.setGeometry(assembled.geometry);
      this.decorationGeometryKey = assembled.key;
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
      // Only the ray probe's work is a sequence, so only it has a sweep to
      // reveal; everything else is emitted at order zero and always drawn.
      reveal: pixelTrace?.reveal ?? 1,
      occludedAlpha: pixelTrace?.occludedAlpha,
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
    this.waterPipeline = undefined; this.gridOverlayPipeline = undefined; this.techniqueOverlayPipeline = undefined; this.techniqueAuditOverlayPipeline = undefined; this.voxelDebugPipeline = undefined; this.svoDryScenePipeline = undefined; this.secondaryParticlePipeline = undefined; this.svoStageOverlay = undefined;
    this.optionalPipelineTasks.clear(); this.failedOptionalPipelines.clear(); this.optionalPipelineFailures.clear(); this.svoDrySceneSource = undefined; this.svoSceneSidecar = undefined; this.svoDrySceneData = undefined; this.liveSceneAnimation = undefined; this.liveSceneAnimationFailure = undefined; this.renderSceneKey = ""; this.svoPipelineProgress = undefined; this.svoPipelineStartedAt_ms = undefined; this.pendingLiveSvoPresentation = undefined;
    this.svoPipelineAvailable = false; this.svoSourceAvailable = false; this.svoPublicationFailure = undefined; this.svoTerrainSupported = true; this.svoGlassSupported = true; this.svoMaterialsSupported = true; this.svoLightingSupported = true;
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
      this.onStatus({ state: "unavailable", label: error instanceof Error ? `GPU recovery failed: ${error.message}` : "GPU recovery failed", resource: webGPUPlatformResourcePlugin });
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
      void this.svoFluidCoverage.initializePipelines();
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
    this.waterPipeline?.setCoarseLevelSet(this.gpuFluid?.coarseLevelSetSource);
    this.gridOverlayPipeline?.setVolume(texture, columnBases, gridCells, velocity, pressureSamples, divergence, pressure);
    this.techniqueOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    this.techniqueOverlayPipeline?.setOwnerRows(pressureSamples);
    this.techniqueAuditOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    this.techniqueAuditOverlayPipeline?.setOwnerRows(pressureSamples);
    this.fluidCellTracePipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource, pressureSamples);
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
    const resource=getMethod(config.methodId).resource;
    this.onStatus({state:"initializing",label:"Re-seeding fenced t=0 solver authority",phase:"warmup",completed:0,total:1,startedAt_ms:performance.now(),kind:"rebuild",retainingPrevious:false,resource});
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
      this.pendingInitialRasterPresentation={solver,solverGeneration:generation,requestGeneration,submitted:false,resource};
      this.gpuInfoCallback?.({...solver.info});
      this.onStatus({state:"initializing",label:"Re-seeded solver ready; publishing fenced t=0 raster surface",phase:"presentation",completed:0,total:1,startedAt_ms:performance.now(),kind:"rebuild",retainingPrevious:false,resource});
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

  /** Change simulation admission while preserving already-submitted queue work. */
  setSimulationRunning(running: boolean): number | undefined {
    if (running !== this.simulationRunning) this.resetPresentationTrace();
    this.simulationRunning = running;
    const submittedTime_s = this.gpuFluid?.info.submittedTime_s;
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
    const rendererOnlyScene=!planSceneRuntime(scene).fluidSolver;
    const initializationResource=rendererOnlyScene?liveSvoSceneResourcePlugin:method.resource;
    this.gpuFluidInitializationAbort?.abort();
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
    const report=(progress:{phase:string;taskId?:string;label:string;completed:number;total:number})=>{if(this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration)return;reportedCompleted=progress.completed;reportedTotal=progress.total;this.onStatus({state:"initializing",...progress,startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:Boolean(previous),resource:initializationResource});};
    let previousDestroyedForReset=false;
    let previousSidecarDestroyedForReset=false;
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
        this.svoDrySceneSource=undefined;this.svoDrySceneData=undefined;this.liveSceneAnimation=undefined;this.liveSceneAnimationFailure=undefined;this.renderSceneKey="";this.svoDryScenePipeline?.setSource(undefined);
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
        solver=await WebGPULiveSvoScene.create(device, scene, config.quality, report, abort.signal, {
          environmentBrickRefinementLevels: typeof refinement === "number" ? refinement : undefined,
        });
      } else {
        solver=await (method.createSolverAsync
          ? method.createSolverAsync(device,scene,config.quality,config.values,this.gpuRigidLoadCallback,report,abort.signal)
          : new Promise<GPUSolverInstance>((resolve,reject)=>setTimeout(()=>{try{resolve(method.createSolver!(device,scene,config.quality,config.values,this.gpuRigidLoadCallback));}catch(error){reject(error);}},0)));
      }
      if (solver.sparseVoxelSceneSource) return {solver};
      try {
        const refinement=config.values.svoEnvironmentBrickRefinementLevels;
        const sidecar=await WebGPULiveSvoScene.create(device,scene,config.quality,report,abort.signal,{
          environmentBrickRefinementLevels:typeof refinement==="number"?refinement:undefined,
        });
        return {solver,sidecar};
      } catch(error) {
        solver.destroy();
        throw error;
      }
    });
    this.gpuFluidPending=create.then(({solver,sidecar})=>{
      if(this.disposed||this.deviceLost||generation!==this.gpuFluidRequestGeneration){solver.destroy();sidecar?.destroy();return;}
      if(config.methodId==="octree"&&solver.initialSparseAuthorityReady!==true){solver.destroy();sidecar?.destroy();throw new Error("Octree solver returned before fenced sparse t=0 authority");}
      report({phase:"attach",taskId:"solver.attach",label:"Attach warmed solver",completed:reportedCompleted,total:reportedTotal+1});
      solver.applyRuntimeValues?.(config.values);
      this.gpuFluid=solver;this.svoSceneSidecar=sidecar;this.gpuFluidKey=key;this.attachedStructuralKey=gpuSceneStructuralKey(scene,config);this.gpuFluidPendingKey="";this.resetGPUQueueTracking();this.gpuFluidGeneration+=1;this.lastGPUInfoPollAt_ms=-Infinity;this.globalFineWaterAttached=false;
      const fencedInitialRaster=requiresFencedInitialRasterPresentation(config.methodId);
      if(rendererOnlyScene){solver.info.initialRasterSurfaceReady=true;solver.info.initialRasterSurfaceState="gpu-authoritative";solver.info.initialRasterSurfaceDiagnostic="Live scene source ready; fluid authority intentionally absent";this.pendingInitialRasterPresentation=undefined;}
      else if(fencedInitialRaster){solver.info.initialRasterSurfaceReady=false;solver.info.initialRasterSurfaceState="pending";solver.info.initialRasterSurfaceDiagnostic="Waiting for the first fenced t=0 raster publication";this.pendingInitialRasterPresentation={solver,solverGeneration:this.gpuFluidGeneration,requestGeneration:generation,submitted:false,resource:method.resource};}
      else{solver.info.initialRasterSurfaceReady=true;solver.info.initialRasterSurfaceState="gpu-authoritative";solver.info.initialRasterSurfaceDiagnostic="Direct solver field attached; sparse raster fence not required";this.pendingInitialRasterPresentation=undefined;}
      this.updateRenderSources(solver.surfaceFieldTexture??solver.volumeTexture,solver.columnBaseTexture,solver.gridCellTexture??this.gridCellTexture,solver.velocityTexture??this.velocityFallbackTexture,solver.gridPressureSamplesTexture??this.pressureSamplesFallbackTexture,solver.gridDivergenceTexture??this.scalarFallbackTexture,solver.gridPressureTexture??this.scalarFallbackTexture);this.secondaryParticlePipeline?.setSource(solver.secondaryParticles);this.voxelInspectionSource?.inspectionPublication?.setEnabled(false);this.voxelInspectionSource=undefined;this.voxelDebugPipeline?.setSource(undefined);this.voxelDebugSourceGeneration=-1;
      this.attachSparsePresentationSource(solver,generation,startedAt_ms,sidecar?.sparseVoxelSceneSource??solver.sparseVoxelSceneSource);
      if(rendererOnlyScene)
        this.onStatus({state:"ready",label:"Live sparse scene source ready",adapter:this.adapterName,resource:liveSvoSceneResourcePlugin});
      this.pausedPresentationRevision+=1;
      if(previous&&previous!==solver&&!previousDestroyedForReset)this.retireGPUFluid(previous);
      if(previousSidecar&&previousSidecar!==sidecar&&!previousSidecarDestroyedForReset)this.retireGPUFluid(previousSidecar);
      this.gpuInfoCallback?.(solver.info);
      if(!rendererOnlyScene&&fencedInitialRaster)this.onStatus({state:"initializing",label:"Warmed solver attached; publishing fenced t=0 raster surface",phase:"presentation",completed:reportedCompleted,total:reportedTotal+1,startedAt_ms,kind:previous?"rebuild":"startup",retainingPrevious:false,resource:method.resource});
      else if(!rendererOnlyScene)this.onStatus({state:"ready",label:"WebGPU direct-field solver ready",adapter:this.adapterName,resource:method.resource});
    }).catch((error:unknown)=>{if(this.disposed||generation!==this.gpuFluidRequestGeneration)return;this.gpuFluidPendingKey="";this.pendingInitialRasterPresentation=undefined;this.pendingLiveSvoPresentation=undefined;if(isGPUInitializationAbort(error))return;if(previous)this.onStatus({state:"ready",label:error instanceof Error?`Solver rebuild failed; previous solver retained: ${error.message}`:"Solver rebuild failed; previous solver retained",adapter:this.adapterName,resource:initializationResource});else this.onStatus({state:"unavailable",label:error instanceof Error?`GPU initialization failed: ${error.message}`:"GPU initialization failed",resource:initializationResource});}).finally(()=>{if(generation===this.gpuFluidRequestGeneration){this.gpuFluidPending=undefined;if(this.gpuFluidInitializationAbort===abort)this.gpuFluidInitializationAbort=undefined;}});
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

  /**
   * Publish the presentation scene independently from the solver identity.
   * The solver receives the same revision through its live-maintenance seam,
   * while renderer-owned analytic/material/light arenas become visible on the
   * next submitted frame without allocation or bind-group churn.
   */
  private publishRenderScene(scene: SceneDescription, solver: GPUSolverInstance | undefined): void {
    const sceneKey = canonicalScene(scene);
    if (sceneKey === this.renderSceneKey) return;
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
        svoMaterialFunctionIdForEnvironmentProxy(primitive),
      )),
    ]);
    const sceneGlass = buildSvoSceneGlass(scene, { cellSize_m: source.structural?.domain.cellSize_m });
    const sceneThickGlass = buildSvoSceneThickGlass(scene, { revision });
    const thickReplacedPaneKey = sceneThickGlass.metadata.find(({ replacesThinPaneKey }) => Boolean(replacesThinPaneKey))?.replacesThinPaneKey;
    const thickReplacedPaneId = sceneGlass.metadata.find(({ key }) => key === thickReplacedPaneKey)?.paneId;
    // A porcelain ground publishes no terrain metadata at all: the metadata is
    // what configures the lawn closure, and an unread region is a region that
    // can go stale. Absent, the shader's policy word stays zero and its terrain
    // branch is unreachable by construction rather than by agreement.
    const terrainMaterial = scenePrimitives.analyticTerrain && terrainSurface === "garden-terrain"
      ? buildSvoTerrainMaterial(scene)
      : undefined;
    const compositorOwnedGlass = sceneGlass.metadata.filter(({ role }) => role === "container-pane" || role === "container-top");
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
      ownerBase: scene.rigidBodies.length,
      skippedOwnerId: scenePrimitives.openShellOwnerId,
      terrainMaterialId: scenePrimitives.analyticTerrain?.materialId,
      terrainMaterialMetadata: terrainMaterial?.packedMetadata,
      terrainMaterialCacheKey: terrainMaterial?.cacheKey,
      // A sculpted vessel is terrain the eight-feature uniform mirror cannot
      // express, so the grid the solver already consumes is published into the
      // scene arena as well. Analytic scenes pass undefined and keep the
      // closed-form evaluator they have always used.
      terrainHeightfield: packSvoDrySceneTerrainHeightfield(scene.terrain?.grid),
      glassRecords: sceneGlass.packedRecords,
      glassCacheKey: sceneGlass.cacheKey,
      thickGlassRecords: sceneThickGlass.packedRecords,
      thickGlassRevision: sceneThickGlass.revision,
      thickGlassCacheKey: sceneThickGlass.cacheKey,
      thickGlassReplacedThinPaneId: thickReplacedPaneId,
      primaryCompositeOwnedGlassPaneIdBase: compositorOwnedGlass[0]?.paneId,
      primaryCompositeOwnedGlassPaneCount: compositorOwnedGlass.length,
      ...lightingMirrors,
    };
    const thickGlassBound = resolveSparseVoxelThickGlassBinderStatus(publication) === "bound";
    const replacedPaneKeys = new Set(sceneThickGlass.metadata.flatMap(({ replacesThinPaneKey }) => replacesThinPaneKey ? [replacesThinPaneKey] : []));
    this.svoTerrainSupported = !scenePrimitives.requiresRasterTerrainFallback && (!sceneHasTerrain(scene) || Boolean(scenePrimitives.analyticTerrain));
    this.svoGlassSupported = !sceneGlass.metadata.some(({ key, opaqueCutoutKey }) => Boolean(opaqueCutoutKey) && (!thickGlassBound || !replacedPaneKeys.has(key)));
    this.svoMaterialsSupported = materialRecords.byteLength > 0;
    this.svoLightingSupported = canConsumeSparseVoxelLighting(publication);
    const supported = this.svoTerrainSupported && this.svoGlassSupported && this.svoMaterialsSupported && this.svoLightingSupported;
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
    this.liveSceneAnimation = sway.some(Boolean) ? {
      rest: scenePrimitives.descriptors,
      sway,
      keys: scenePrimitives.metadata.map(({ key }) => key),
      current: scenePrimitives.descriptors,
      candidates: primitiveCandidates,
    } : undefined;
    this.renderSceneRevision = revision;
    this.renderSceneKey = sceneKey;
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

  /** Refit continuously-authored motion through the same exact live arena. */
  private advanceLiveSceneAnimation(solver: GPUSolverInstance | undefined): void {
    const animation = this.liveSceneAnimation;
    const pipeline = this.svoDryScenePipeline;
    if (!animation || !pipeline || !this.svoDrySceneData) return;
    const now_ms = performance.now();
    animation.origin_ms ??= now_ms;
    const time_s = Math.max(0, now_ms - animation.origin_ms) / 1000;
    const descriptors = animation.rest.map((descriptor, index) => {
      const sway = animation.sway[index];
      return sway ? swayedPrimitiveDescriptor(descriptor, sway, time_s) : descriptor;
    });
    const candidates = refitSvoPrimitiveCandidates(
      descriptors as Parameters<typeof refitSvoPrimitiveCandidates>[0],
      animation.candidates,
    );
    const records = packSvoPrimitiveRecords(descriptors);
    const dirtyBounds = svoSwayDirtyBounds(animation.current, descriptors, animation.sway);
    const sparseUpdates = descriptors.flatMap((descriptor, index) => animation.sway[index] ? [{
      key: animation.keys[index], primitive: sparseScenePrimitiveForSvoDescriptor(descriptor),
    }] : []);
    try {
      if (sparseUpdates.length > 0 && !solver?.stageLivePrimitiveUpdates) {
        throw new Error("the active scene producer has no keyed live-update seam");
      }
      solver?.stageLivePrimitiveUpdates?.(sparseUpdates);
    } catch (error) {
      // Sparse and analytic publications remain atomic: retain the previous
      // pose and expose the failed-closed state instead of silently degrading.
      const reason = error instanceof Error ? error.message : "unknown sparse maintenance failure";
      if (reason !== this.liveSceneAnimationFailure) {
        this.liveSceneAnimationFailure = reason;
        this.onStatus({
          state: "blocked",
          label: `Live scene update failed closed: ${reason}`,
          resource: svoPresentationResourcePlugin,
        });
      }
      return;
    }
    this.liveSceneAnimationFailure = undefined;
    const revision = this.renderSceneRevision >= 0xffff_fffe ? 1 : this.renderSceneRevision + 1;
    if (!pipeline.publishPrimitiveArena(records, candidates, revision, {
      dirtyBounds,
      // Sparse node-mip/radiance payloads change in the dirty pages too. Their
      // stable atlas slots and page-local validity keep unrelated cache entries
      // warm without pretending the derived payload itself stayed unchanged.
      derivedLighting: "unchanged",
    })) {
      const reason = "the analytic primitive arena is not ready for the staged sparse generation";
      this.liveSceneAnimationFailure = reason;
      this.onStatus({ state: "blocked", label: `Live scene update failed closed: ${reason}`, resource: svoPresentationResourcePlugin });
      return;
    }
    animation.current = descriptors;
    animation.candidates = candidates;
    this.renderSceneRevision = revision;
    this.svoDrySceneData = { ...this.svoDrySceneData, renderRevision: revision, primitiveRecords: records, primitiveCandidates: candidates };
  }

  /** The solver's sparse world when it has one, otherwise the renderer sidecar. */
  private sparseSceneProducer(fluid: GPUSolverInstance | undefined): GPUSolverInstance | undefined {
    return fluid?.sparseVoxelSceneSource ? fluid : this.svoSceneSidecar;
  }

  private currentGPUFluid(scene: SceneDescription, config: SimulationRunConfig) {
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
          globalFineAttached: Boolean(pending.solver.globalFineLevelSetSource
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
    this.gpuInfoCallback?.(pending.solver.info);
    this.pausedPresentationRevision += 1;
    if (outcome.ready) this.onStatus({ state: "ready", label: outcome.label, adapter: this.adapterName, resource: pending.resource });
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

  private submitPreparedGPUFluid(fluid: GPUSolverInstance, time_s: number, bodies: RigidBodyState[], maximumPendingAdvances = 1) {
    const device = this.device;
    if (!device) return fluid.info;
    // Admit at most the single advance paired with this draw. The following
    // presentation is submitted immediately after it, so no later simulation
    // work can overtake the frame that visualizes this state transition.
    if (!canQueuePreparedGPUAdvance(this.gpuPendingBatches, maximumPendingAdvances)) return fluid.info;
    const { previousSubmittedTime, submittedTime } = submitNextPreparedGPUAdvance(fluid, time_s, bodies);
    if (submittedTime > previousSubmittedTime) {
      const generation = this.gpuFluidGeneration;
      this.gpuPendingBatches += 1;
      fluid.info.gpuPendingBatches = this.gpuPendingBatches;
      fluid.info.gpuInFlightSimulation_s = Math.max(0, submittedTime - (fluid.info.completedTime_s ?? 0));
      void device.queue.onSubmittedWorkDone().then(() => {
        if (this.disposed || this.deviceLost || this.gpuFluid !== fluid || this.gpuFluidGeneration !== generation) return;
        this.gpuPendingBatches = Math.max(0, this.gpuPendingBatches - 1);
        fluid.info.completedTime_s = Math.max(fluid.info.completedTime_s ?? 0, submittedTime);
        fluid.info.gpuPendingBatches = this.gpuPendingBatches;
        fluid.info.gpuInFlightSimulation_s = Math.max(0, (fluid.info.submittedTime_s ?? submittedTime) - fluid.info.completedTime_s);
        this.gpuInfoCallback?.({ ...fluid.info });
        this.gpuAdvanceCompletedCallback?.(submittedTime);
      }).catch(() => { /* Device loss is reported by device.lost. */ });
    }
    // Functional diagnostics use a bounded cadence and remain independent of
    // the generic performance trace sampled by the solver itself.
    const now_ms=performance.now();if(now_ms-this.lastGPUInfoPollAt_ms>=250){
      this.lastGPUInfoPollAt_ms=now_ms;
      void fluid.readStats().then(info=>this.gpuInfoCallback?.({...info}))
        .catch(()=>{ /* Device loss is reported by device.lost. */ });
    }
    return fluid.info;
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

  /** Frames whose presentation submission has actually finished on the GPU. */
  get completedPresentationCount(): number { return this.completedPresentations; }

  /** The described object the cursor is over, or undefined to clear the rim. */
  setHoverHighlight(range: { readonly first: number; readonly last: number } | undefined): void {
    if (range?.first === this.hoverHighlight?.first && range?.last === this.hoverHighlight?.last) return;
    this.hoverHighlight = range && range.last >= range.first ? range : undefined;
  }

  draw(time_s: number, scene: SceneDescription, camera: CameraState, bodies: RigidBodyState[], selectedBodyId: string | undefined, fluid: EulerianRenderState | undefined, backend: SimulationBackend, config: SimulationRunConfig, gridOverlay?: GridOverlayConfig, environmentId: EnvironmentId = defaultEnvironmentId, voxelRenderMode: VoxelRenderMode = "smooth", svoLightingOptions: SvoLightingOptions = DEFAULT_SVO_LIGHTING_OPTIONS, svoDiagnostics: SvoRenderDiagnostics = DEFAULT_SVO_RENDER_DIAGNOSTICS, svoTuning: SvoRenderTuning = DEFAULT_SVO_RENDER_TUNING, pixelTrace?: PixelTraceConfig, fluidCellTrace?: FluidCellTraceConfig): RendererFrameMetrics {
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
    const presentationContext = `${config.methodId}:${config.quality}:shadow-${svoLightingOptions.shadowsEnabled ? "on" : "off"}:ao-${svoLightingOptions.ambientOcclusionEnabled ? "on" : "off"}:cones-${svoLightingOptions.coneTracingMode ?? "cones"}:primary-${svoLightingOptions.primaryTraversal ?? "raster"}:gi:${voxelRenderMode}:tuning-${tuningKey}:${this.simulationRunning ? "running" : "paused"}`;
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
    if (backend === "webgpu" && (gridOverlay?.axis !== "off" || fluidCellTrace)) {
      this.gpuFluid?.ensureGridDiagnosticTextures?.();
    }
    const sceneRuntime = planSceneRuntime(scene);
    const svoSceneConfig: SimulationRunConfig = {
      ...config,
      values: {
        ...config.values,
        svoEnvironmentBrickRefinementLevels: activeSvoTuning.environmentBrickRefinementLevels,
      },
    };
    const readyGPUFluid = backend === "webgpu" || !sceneRuntime.fluidSolver
      ? this.currentGPUFluid(this.simulationScene ?? scene, svoSceneConfig)
      : undefined;
    const fluidSource = readyGPUFluid ?? this.gpuFluid;
    const sparseSceneProducer = this.sparseSceneProducer(fluidSource);
    // Every authored scene has a room or terrain shell. Fluid methods without
    // their own sparse hierarchy receive a renderer-owned source sidecar, so a
    // missing method capability can never turn the dry scene into magenta.
    const sparsePresentationRequired = true;
    this.publishRenderScene(scene, sparseSceneProducer);
    const pixelTraceRequested = Boolean(pixelTrace) && voxelRenderMode === "smooth";
    // Ahead of the sweep, so a toggled traversal retires the old pipeline and is
    // rebuilt in the same frame rather than one frame later.
    this.applyPrimaryTraversalRequest(svoLightingOptions.primaryTraversal ?? "raster");
    this.ensureRequestedOptionalPipelines(optionalRendererPipelineRequests(
      gridOverlay, voxelRenderMode, this.simulationRunning,
      Boolean((readyGPUFluid ?? this.gpuFluid)?.secondaryParticles),
      pixelTraceRequested, Boolean(fluidCellTrace),
      activeSvoDiagnostics.stageView !== "off",
      sparsePresentationRequired,
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
    // Raw voxel/brick inspection is opt-in. Keeping the source detached in
    // smooth presentation avoids a second capacity-sized GPU instance arena
    // (about 295 MB for the widened ocean) while SVO continues to consume the
    // structural source directly.
    const requestedVoxelDebugGeneration = voxelRenderMode !== "smooth" && sparseSceneProducer
      ? this.gpuFluidGeneration
      : -1;
    if (requestedVoxelDebugGeneration !== this.voxelDebugSourceGeneration) {
      this.voxelInspectionSource?.inspectionPublication?.setEnabled(false);
      this.voxelInspectionSource = requestedVoxelDebugGeneration >= 0 ? sparseSceneProducer?.sparseVoxelRenderSource : undefined;
      this.voxelInspectionSource?.inspectionPublication?.setEnabled(true);
      this.voxelDebugPipeline?.setSource(this.voxelInspectionSource);
      this.voxelDebugSourceGeneration = requestedVoxelDebugGeneration;
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
      );
    }
    // The global fine narrow band double-buffers generations. Refresh its
    // tagged renderer binding after each admitted solver encode so extraction
    // follows the newly published generation without any CPU field copy.
    if (readyGPUFluid?.globalFineLevelSetSource) {
      this.waterPipeline.setGlobalFineLevelSet(createGlobalFineLevelSetConsumerSource(readyGPUFluid.globalFineLevelSetSource));
    } else {
      this.waterPipeline.setGlobalFineLevelSet(undefined);
    }
    this.waterPipeline.setCoarseLevelSet(readyGPUFluid?.coarseLevelSetSource);
    const globalFineWaterReady = Boolean(readyGPUFluid
      && readyGPUFluid.initialSparseAuthorityReady
      && (readyGPUFluid.globalFineLevelSetSource || readyGPUFluid.coarseLevelSetSource));
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
    if (gpuInfo && this.gpuFluid && this.columnBaseTexture && this.gridCellTexture && this.velocityFallbackTexture && this.pressureSamplesFallbackTexture && this.scalarFallbackTexture) {const compactSurface=Boolean(this.gpuFluid.globalFineLevelSetSource||this.gpuFluid.coarseLevelSetSource);this.gridOverlayPipeline?.setVolume(compactSurface?this.scalarFallbackTexture:this.gpuFluid.surfaceFieldTexture??this.gpuFluid.volumeTexture, this.gpuFluid.columnBaseTexture ?? this.columnBaseTexture, this.gpuFluid.gridCellTexture ?? this.gridCellTexture, this.gpuFluid.velocityTexture ?? this.velocityFallbackTexture, this.gpuFluid.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture, this.gpuFluid.gridDivergenceTexture ?? this.scalarFallbackTexture, this.gpuFluid.gridPressureTexture ?? this.scalarFallbackTexture);}
    cpuTrace?.transition({ id: "scene-upload", label: "Scene and field uploads" });
    if (backend === "cpu-reference") this.uploadFluid(fluid);
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
    // The compact pressure solve ping-pongs its row buffer. Refresh the
    // diagnostic bundle only while a technique view is visible so pressure
    // updates never keep a bind group pointed at the preceding solve bank.
    if (techniqueModeCode) {
      this.techniqueOverlayPipeline?.setSource(this.gpuFluid?.octreeTechniqueDebugSource);
    }
    const uniform = new Float32Array([
      this.presentationTexture.width, this.presentationTexture.height, time_s, cameraChanging ? SVO_CAMERA_CHANGING_FRAME : -1,
      position.x, position.y, position.z, 0,
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
    const packed = new Float32Array(104);
    packed.set(uniform, 0);
    // Lanes 100..103 are the hover rim: the owner range under the cursor, its
    // strength, and the falloff exponent. Appended past the terrain mirror so
    // every other shader's view of this buffer is byte-identical.
    if (this.hoverHighlight) packed.set([this.hoverHighlight.first, this.hoverHighlight.last, .45, 2.6], 100);
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
    this.advanceLiveSceneAnimation(sparseSceneProducer);
    this.svoDryScenePipeline?.setRigidBodyCount(bodies.length, svoDryRigidBounds(bodies));
    // Reduced-rate cone lighting is the production default: quarter-axis-rate
    // prepass + full-resolution relight, with 0.5 retained by the quality tier.
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
    const traceDetailedSvoRenderPath = true;
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
    fluidSource?.encodeSceneMaintenance?.(encoder);
    if (sparseSceneProducer !== fluidSource) sparseSceneProducer?.encodeSceneMaintenance?.(encoder);
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
    let requestedBundleStatus = this.svoDryScenePipeline?.presentationBundleStatus;
    const drySceneReplacement = (
      replacementEncoder: GPUCommandEncoder,
      target: GPUTexture | GPUTextureView,
      tracePhase?: (phase: GPUTimestampPhase) => void,
    ) => {
      // Reuse is legal only while the complete live render generation and all
      // frame inputs remain unchanged. Source replacement and authored motion
      // advance sceneEpoch; this key also covers camera, viewport, bodies,
      // selection, tuning and environment.
      const primaryCoherenceKey = activeSvoTuning.stationaryPrimaryReuseEnabled
        && (!sceneRuntime.fluidSolver || !this.simulationRunning)
        ? `${presentationCoherenceKey}|viewport=${this.presentationTexture!.width}x${this.presentationTexture!.height}|scene=${this.svoDryScenePipeline?.sceneEpoch ?? 0}`
        : undefined;
      const replacementResult = this.svoDryScenePipeline?.encode(replacementEncoder, target, primaryCoherenceKey, tracePhase) ?? false;
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
    if (!rasterResult) throw new Error("Water optics pipeline is not ready");
    const pendingInitialRaster = this.pendingInitialRasterPresentation;
    const initialRasterSubmission = pendingInitialRaster
      && !pendingInitialRaster.submitted
      && pendingInitialRaster.solver === readyGPUFluid
      && readyGPUFluid.initialSparseAuthorityReady === true
      && Boolean(readyGPUFluid.globalFineLevelSetSource || readyGPUFluid.coarseLevelSetSource)
      && this.globalFineWaterAttached
      && rasterResult.surfaceUpdated
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
      terrainSupported: this.svoTerrainSupported,
      glassSupported: this.svoGlassSupported,
      materialsSupported: this.svoMaterialsSupported,
      lightingSupported: this.svoLightingSupported,
      svoEncoded,
      contractFailure: this.svoPublicationFailure,
      silhouetteRefinement: silhouetteRefinementStatus,
      lightingVisibility: lightingVisibilityStatus,
    }));
    let inspectionOverlayEncoded = false;
    // Render stage views replace the composited image with a decode of a plane
    // an earlier pass published. Encoded first among the inspection overlays so
    // structural overlays and the ray-trace decoration still draw over it, and
    // strictly read-only, so the frame it explains is the frame that shipped.
    if (activeSvoDiagnostics.stageView !== "off" && this.svoStageOverlay?.ready && svoEncoded) {
      const sceneExtent = Math.hypot(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
      inspectionOverlayEncoded = this.svoStageOverlay.encode(
        encoder, this.presentationTexture.createView(),
        activeSvoDiagnostics.stageView, activeSvoDiagnostics.lightSlot,
        this.presentationTexture.width, this.presentationTexture.height,
        camera.distance_m + sceneExtent,
        { ...(this.svoDryScenePipeline?.stagePlanes ?? {}), sceneRadiance: this.waterPipeline.drySceneRadianceView },
      ) || inspectionOverlayEncoded;
    }
    if (voxelRenderMode !== "smooth" && this.voxelDebugDepth) {
      const sceneExtent = Math.hypot(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
      this.voxelDebugPipeline?.encode(encoder, {
        mode: voxelRenderMode,
        colorTarget: this.presentationTexture.createView(),
        depthTarget: this.voxelDebugDepth.createView(),
        depthLoadOp: "clear",
        // Structural views diagnose the same GLOBAL frame. Alpha-blended
        // geometry must preserve the cone-traced presentation underneath.
        colorLoadOp: "load",
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
    // The gather is a compute pass over published topology, so it is encoded
    // whether or not any overlay drew this frame.
    // The readback is pumped after the submit, not here: `mapAsync` puts the
    // buffer into a pending map state immediately, and a buffer with a map
    // pending may not appear in a submit — including the very submit that
    // carries this copy. Mapping before submitting rejects the whole command
    // buffer, so the frame that was gathering the cell also fails to present.
    if (fluidCellTrace) {
      // Lazy allocation means the owner map may have arrived after the last
      // source refresh; `setSource` is a no-op once it stops changing.
      this.fluidCellTracePipeline?.setSource(
        this.gpuFluid?.octreeTechniqueDebugSource,
        this.gpuFluid?.gridPressureSamplesTexture ?? this.pressureSamplesFallbackTexture);
      this.fluidCellTracePipeline?.encode(encoder);
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
    }
    // One assembled draw for every decoration any pass contributed this frame.
    this.encodeDecorationOverlay(
      encoder, basis, scene, pixelTraceRequested ? pixelTrace : undefined, fluidCellTrace);
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
    this.device.queue.submit([encoder.finish()]);
    this.presentationsInFlight+=1;
    const completedPresentationDevice=this.device;
    let presentationRetired=false;
    const retirePresentation=()=>{
      if(presentationRetired)return;
      presentationRetired=true;
      this.presentationsInFlight=Math.max(0,this.presentationsInFlight-1);
    };
    void completedPresentationDevice.queue.onSubmittedWorkDone().then(()=>{
      retirePresentation();
      if(!this.disposed&&!this.deviceLost&&this.device===completedPresentationDevice)this.completedPresentations+=1;
    }).catch(retirePresentation);
    if (pixelTraceProbing) this.pumpPixelTraceReadback();
    if (fluidCellTrace) this.pumpFluidCellTraceReadback();
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
    // The ordinary completion callback only retires a throughput slot and feeds
    // FPS evidence. First-frame startup additionally needs proof that its exact
    // submission completed before publishing renderer authority.
    if(initialLiveSvoSubmission||initialRasterSubmission){
      const presentationDevice=this.device;
      void this.device.queue.onSubmittedWorkDone().then(async()=>{
        if(this.disposed||this.deviceLost||this.device!==presentationDevice)return;
        if(initialLiveSvoSubmission)this.settleLiveSvoPresentation(initialLiveSvoSubmission);
        if(initialRasterSubmission){
          const initialDiagnostics=await surfaceDiagnosticsCompletion;
          this.settleInitialRasterPresentation(initialRasterSubmission,surfaceDiagnosticsRequired,initialDiagnostics);
        }
      }).catch(()=>{ /* Device loss is reported by device.lost. */ });
    }
    return this.currentFrameMetrics(config.methodId, presentationContext, true, cpuTrace?.finish());
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
    try { this.voxelDebugPipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    try { this.svoDryScenePipeline?.destroy(); } catch { /* Best-effort cleanup after device loss. */ }
    for (const resource of [this.presentationTexture, this.voxelDebugDepth, this.fluidTexture, this.columnBaseTexture, this.gridCellTexture, this.velocityFallbackTexture, this.pressureSamplesFallbackTexture, this.scalarFallbackTexture, this.uniformBuffer, this.bodyBuffer]) {
      try { resource?.destroy(); } catch { /* Best-effort cleanup during hot reload. */ }
    }
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
