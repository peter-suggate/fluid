import { GPUInitializationTaskRunner } from "../../core/gpu-initialization";
import type { GPUQuality } from "../../core/gpu-quality";
import type {
  GPUInitializationReporter,
  GPUSolverInstance,
  InjectedLiquidBall,
} from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import type { RigidBodyState } from "../../core/rigid-body";
import { sceneLatticeDimensions } from "../../core/scene-lattice";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { CM12_PAPER_DT_S } from "../../core/cm12-numerics";
import type { GPUEulerianInfo, GPURigidLoad } from "../../core/webgpu-eulerian";
import {
  ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS,
  AdaptiveMassFrameCapture,
} from "./adaptive-mass-frame-pipeline";
import type { AdaptiveMassSolverOptions } from "./method";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
  sparseBrickKey,
  sparseBrickAtlasStats,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";
import {
  initializeSparseAtlasDynamics,
  type SparseAtlasDynamicsState,
} from "./sparse-atlas-dynamics";
import { WebGPUAdaptiveMassAtlasPresentation } from "./webgpu-adaptive-mass-atlas-presentation";
import {
  WebGPUSparseCM12Resident,
  type SparseCM12GPUActivityRecord,
} from "./webgpu-sparse-cm12-resident";

/** Method-local long-run physics receipt carried through the generic info bag. */
export interface AdaptiveMassStepTelemetry {
  adaptiveKineticEnergyBeforeFineUnits?: number;
  adaptiveKineticEnergyAfterFineUnits?: number;
  adaptiveProjectionKineticEnergyBeforeFineUnits?: number;
  adaptiveProjectionKineticEnergyAfterFineUnits?: number;
  adaptiveInactiveFaceCount?: number;
  adaptiveMaximumInactiveFaceSpeedBefore_m_s?: number;
  adaptiveMaximumInactiveFaceSpeedAfter_m_s?: number;
  adaptiveMaximumMixedSeamDivergence_s?: number;
  adaptiveMaximumDensityAfterTransport?: number;
  adaptiveMaximumDensityAfterConditioning?: number;
}

export interface AdaptiveMassGPUActivityBrick extends SparseCM12GPUActivityRecord {
  readonly key: number;
  readonly coordinate: SparseBrickVec3;
  readonly resolution: SparseBrickResolution;
}

/** Select a fine receiver and its strongly graded outward support rung. */
export function dormantReceiverResolution(
  mode: AdaptiveMassSolverOptions["resolutionMode"],
  distance = 0,
  sourceResolution: SparseBrickResolution = 8,
): SparseBrickResolution {
  if (mode === "all-fine") return 8;
  if (mode === "all-coarse") return 4;
  let resolution = sourceResolution;
  for (let step = 0; step < distance; step += 1) {
    resolution = resolution === 8 ? 4 : resolution === 4 ? 2 : 1;
  }
  return resolution;
}

/** Reserve a compact receiver halo for the fixed-topology GPU control arms. */
function residentSupportAtlas(
  source: SparseAdaptiveMassAtlas,
  mode: AdaptiveMassSolverOptions["resolutionMode"],
): SparseAdaptiveMassAtlas {
  const bricks = new Map(source.bricks.map((brick) => [brick.key, brick] as const));
  const receiverResolution: SparseBrickResolution = mode === "all-coarse" ? 4 : 8;
  for (const brick of source.bricks) {
    for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1)
      for (let dx = -1; dx <= 1; dx += 1) {
        const coordinate: SparseBrickVec3 = [
          brick.coordinate[0] + dx,
          brick.coordinate[1] + dy,
          brick.coordinate[2] + dz,
        ];
        if (coordinate.some((value, axis) => value < 0
          || value >= source.brickDimensions[axis])) continue;
        const key = sparseBrickKey(coordinate, source.brickDimensions);
        if (bricks.has(key)) continue;
        const count = receiverResolution ** 3;
        const receiver: SparseAdaptiveMassBrick = {
          key, coordinate, resolution: receiverResolution,
          density: new Float64Array(count),
          gamma: new Float64Array(count).fill(1),
        };
        bricks.set(key, receiver);
      }
  }
  return createSparseAdaptiveMassAtlas(
    source.dimensions,
    [...bricks.values()].sort((left, right) => left.key - right.key),
    source.generation,
  );
}

/**
 * Preallocate dormant receivers over the bounded brick domain. A one-axis
 * corridor leaves corner-authored dams capped by the transverse support halo
 * (cell 47 in the canonical 64-cubed mini dam).
 */
export function dormantReceiverDomain(
  source: SparseAdaptiveMassAtlas,
  mode: AdaptiveMassSolverOptions["resolutionMode"],
): SparseAdaptiveMassAtlas {
  const bricks = new Map(source.bricks.map((brick) => [brick.key, brick] as const));
  // A receiver becomes part of the represented domain precisely because the
  // swept surface can enter it. Rung A therefore authors it at the finest
  // level; grading may only coarsen outward support, never the receiver itself.
  const receiverResolution: SparseBrickResolution = mode === "all-coarse" ? 4 : 8;
  for (let z = 0; z < source.brickDimensions[2]; z += 1)
    for (let y = 0; y < source.brickDimensions[1]; y += 1)
      for (let x = 0; x < source.brickDimensions[0]; x += 1) {
    const coordinate: SparseBrickVec3 = [x, y, z];
    const key = sparseBrickKey(coordinate, source.brickDimensions);
    if (bricks.has(key)) continue;
    const resolution = receiverResolution;
    const count = resolution ** 3;
    bricks.set(key, {
      key, coordinate, resolution,
      density: new Float64Array(count),
      gamma: new Float64Array(count).fill(1),
    });
  }
  return createSparseAdaptiveMassAtlas(
    source.dimensions,
    [...bricks.values()].sort((left, right) => left.key - right.key),
    source.generation,
  );
}

/**
 * GPU-resident Sparse CM12 authority. Construction may build compact topology
 * on the host, but every accepted frame is device-only simulation work: the
 * host writes one small uniform block and encodes a fixed dispatch schedule.
 */
export class WebGPUAdaptiveMassSolver implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly gridCellTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly gridPressureTexture: GPUTexture;
  readonly gridDivergenceTexture: GPUTexture;

  private atlas: SparseAdaptiveMassAtlas;
  private lastTime_s = 0;
  private disposed = false;
  private physicsTraceSampleId = 0;
  private physicsTracePending = false;
  private lastPhysicsTraceAt_ms = -Infinity;

  private constructor(
    private readonly device: GPUDevice,
    private readonly scene: SceneDescription,
    private readonly options: AdaptiveMassSolverOptions,
    private readonly presentation: WebGPUAdaptiveMassAtlasPresentation,
    private readonly resident: WebGPUSparseCM12Resident,
    adaptiveMixedSeamFaceCount: number,
    atlas: SparseAdaptiveMassAtlas,
    quality: GPUQuality,
  ) {
    this.atlas = atlas;
    this.volumeTexture = presentation.densityTexture;
    this.surfaceFieldTexture = presentation.levelSetTexture;
    this.gridCellTexture = presentation.gridCellTexture;
    this.velocityTexture = presentation.velocityTexture;
    this.gridPressureTexture = presentation.pressureTexture;
    this.gridDivergenceTexture = presentation.divergenceTexture;
    const stats = sparseBrickAtlasStats(atlas);
    const [nx, ny, nz] = atlas.dimensions;
    const representedFraction = stats.leafCount / Math.max(1, stats.equivalentFinestCellCount);
    const cellSize_m = Math.min(
      scene.container.width_m / nx,
      scene.container.height_m / ny,
      scene.container.depth_m / nz,
    );
    this.info = {
      nx,
      ny,
      nz,
      storedNy: ny,
      cellCount: stats.leafCount,
      equivalentUniformCells: stats.equivalentFinestCellCount,
      compressionRatio: representedFraction,
      activeCompressionRatio: representedFraction,
      activeSampleCount: stats.leafCount,
      regularLayers: ny,
      maximumNeighborDelta: 1,
      gridKind: "octree",
      cellSize_m,
      pressureIterations: 0,
      pressureSolver: "GPU-resident sparse composite GᵀWG Jacobi-PCG",
      allocatedBytes: presentation.allocatedBytes + resident.allocatedBytes,
      quality,
      volumeCellSum: stats.integratedMassFineCells,
      representedVolumeCellSum: stats.integratedMassFineCells,
      representedVolumeDrift: 0,
      volumeTelemetrySource: "adaptive-conservative-mass",
      fluidBrickCapacity: stats.logicalBrickCount,
      fluidBrickResidentCount: stats.residentBrickCount,
      fluidBrickCoreCount: stats.residentBrickCount,
      fluidBrickHaloCount: 0,
      fluidBrickGeneration: stats.generation,
      adaptiveFineBrickCount: stats.fineBrickCount,
      adaptiveCoarseBrickCount: stats.coarseBrickCount,
      adaptiveFineCoarseFaceConnectedPairCount:
        stats.fineCoarseFaceConnectedPairCount,
      adaptiveMixedSeamFaceCount,
      quadtreeMaximumFluidScale: 2,
      quadtreeMaximumNeighborRatio: 2,
      submittedTime_s: 0,
      simulatedTime_s: 0,
      completedTime_s: 0,
      simulationLag_s: 0,
      encodedSteps: 0,
      lastSubsteps: 1,
      maximumTallCellHeight: 2,
      surfaceField: "levelset",
      volumeControl: false,
      hostFluidAuthority: "gpu-resident",
      hostSimulationSizedWorkItems: 0,
      hostSchedulingUsesReadback: false,
    };
  }

  static async createAsync(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    _onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: AdaptiveMassSolverOptions,
    onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUAdaptiveMassSolver> {
    const runner = new GPUInitializationTaskRunner(onProgress, signal);
    let dimensions: SparseBrickVec3 | undefined;
    let atlas: SparseAdaptiveMassAtlas | undefined;
    let presentation: WebGPUAdaptiveMassAtlasPresentation | undefined;
    let dynamics: SparseAtlasDynamicsState | undefined;
    let resident: WebGPUSparseCM12Resident | undefined;
    let initiallyActiveBrickKeys: ReadonlySet<number> | undefined;
    try {
      await runner.run([{
        id: "adaptive-mass.plan",
        phase: "planning",
        label: "Bound the arbitrary-scene presentation lattice",
        run: () => { dimensions = adaptiveMassPresentationDimensionsForScene(scene); },
      }, {
        id: "adaptive-mass.atlas",
        phase: "adaptive-topology",
        label: "Build resident 4³/8³ sparse bricks",
        dependencies: ["adaptive-mass.plan"],
        run: () => {
          atlas = initializeSparseBrickAtlasFromScene(scene, {
            finestDimensions: dimensions!,
            resolutionForBrick: options.resolutionMode === "all-fine"
              ? () => 8
              : () => 4,
          });
          const supported = residentSupportAtlas(atlas, options.resolutionMode);
          initiallyActiveBrickKeys = new Set(supported.bricks.map((brick) => brick.key));
          atlas = dormantReceiverDomain(supported, options.resolutionMode);
          dynamics = initializeSparseAtlasDynamics(atlas);
        },
      }, {
        id: "adaptive-mass.presentation",
        phase: "allocation",
        label: "Allocate adaptive water and ownership textures",
        dependencies: ["adaptive-mass.atlas"],
        run: () => { presentation = new WebGPUAdaptiveMassAtlasPresentation(device, dimensions!); },
      }, {
        id: "adaptive-mass.resident",
        phase: "allocation",
        label: "Pack compact GPU topology and allocate resident frame state",
        dependencies: ["adaptive-mass.atlas", "adaptive-mass.presentation"],
        run: async () => {
          resident = await WebGPUSparseCM12Resident.create(
            device, atlas!, dynamics!.grid, presentation!,
            initiallyActiveBrickKeys,
          );
        },
      }, {
        id: "adaptive-mass.upload",
        phase: "upload",
        label: "Publish sparse atlas generation zero",
        dependencies: ["adaptive-mass.resident"],
        run: () => {
          const encoder = device.createCommandEncoder({
            label: "Sparse CM12 initial GPU publication",
          });
          resident!.encodeInitialPresentation(encoder, finestCellSize(scene, atlas!));
          device.queue.submit([encoder.finish()]);
        },
      }, {
        id: "adaptive-mass.warmup",
        phase: "warmup",
        label: "Fence adaptive presentation generation zero",
        dependencies: ["adaptive-mass.upload"],
        run: () => device.queue.onSubmittedWorkDone(),
      }]);
      return new WebGPUAdaptiveMassSolver(
        device, scene, options, presentation!, resident!,
        dynamics!.grid.mixedSeamRowCount, atlas!, quality,
      );
    } catch (error) {
      resident?.destroy();
      presentation?.destroy();
      throw error;
    }
  }

  /**
   * Add a ball of liquid to the atlas the running solve is stepping.
   *
   * Without this the editor authors the ball into the scene document instead,
   * which re-seeds this solver at t = 0 — the user drops water into a running
   * tank and the tank restarts. The ball is converted from metres to finest
   * cells here because the atlas is the only thing that knows the lattice, and
   * its radius becomes three radii: a metric sphere is an ellipsoid on any
   * lattice whose cells are not cubes.
   *
   * The drop is applied to the atlas immediately rather than deferred to the
   * next step, and the presentation is republished from it so the fields the
   * renderer and the diagnostics read agree with the atlas before that step.
   * The ball is drawn from the step after the drop, like any other water.
   */
  injectLiquidBall(ball: InjectedLiquidBall): void {
    if (this.disposed || !(ball.radius_m > 0)) return;
    const container = this.scene.container;
    const [nx, ny, nz] = this.atlas.dimensions;
    const encoder = this.device.createCommandEncoder({
      label: "Sparse CM12 GPU liquid injection",
    });
    this.resident.encodeLiquidInjection(
      encoder,
      finestCellSize(this.scene, this.atlas),
      [
        (ball.centre_m.x + 0.5 * container.width_m) * nx / container.width_m,
        ball.centre_m.y * ny / container.height_m,
        (ball.centre_m.z + 0.5 * container.depth_m) * nz / container.depth_m,
      ],
      [
        ball.radius_m * nx / container.width_m,
        ball.radius_m * ny / container.height_m,
        (ball.halfHeight_m ?? ball.radius_m) * nz / container.depth_m,
      ],
    );
    this.device.queue.submit([encoder.finish()]);
  }

  advanceTo(time_s: number, _bodies: RigidBodyState[]): boolean {
    void _bodies;
    if (this.disposed || !Number.isFinite(time_s) || time_s <= this.lastTime_s + 1e-9) return false;
    const paperTimeStep = this.options.timeStep === "paper";
    if (paperTimeStep
      && time_s - this.lastTime_s < CM12_PAPER_DT_S - 1e-9) return false;
    const dt_s = paperTimeStep
      ? CM12_PAPER_DT_S
      : Math.min(this.scene.numerics.maxDt_s, time_s - this.lastTime_s);
    if (!(dt_s > 0)) return false;
    const cellSize_m = finestCellSize(this.scene, this.atlas);
    const gravity = this.scene.fluid.gravity_m_s2;
    const instrumentation = usePerformanceInstrumentationStore.getState();
    const traceRequestedAt_ms = instrumentation.enabled ? performance.now() : 0;
    const shouldTracePhysics = instrumentation.enabled && !this.physicsTracePending
      && traceRequestedAt_ms - this.lastPhysicsTraceAt_ms
        >= ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS;
    const traceSampleId = shouldTracePhysics ? ++this.physicsTraceSampleId : 0;
    const frameCapture = shouldTracePhysics
      ? new AdaptiveMassFrameCapture(
        traceSampleId,
        `adaptive-mass:sim-${(this.lastTime_s + dt_s).toFixed(6)}`,
      )
      : undefined;
    const encoder = this.device.createCommandEncoder({
      label: `Sparse CM12 resident frame ${(this.lastTime_s + dt_s).toFixed(6)}`,
    });
    this.resident.encode(
      encoder,
      dt_s,
      cellSize_m,
      this.scene.fluid.density_kg_m3 * cellSize_m * cellSize_m / dt_s,
      [gravity.x / cellSize_m, gravity.y / cellSize_m, gravity.z / cellSize_m],
    );
    // These seams describe host encoding only.  The corresponding numerical
    // stages are ordered compute dispatches in the command buffer above.
    for (const stage of ["receiver-topology", "coupled-transport",
      "surface-conditioning", "activity-resolution", "retain-rebuild", "force"] as const) {
      frameCapture?.completeDynamicsStage(stage);
    }
    for (const stage of ["topology", "rhs", "solve", "projection", "diagnostics"] as const) {
      frameCapture?.completeProjectionStage(stage);
    }
    frameCapture?.completeStateCommit();
    frameCapture?.completeMaterialization();
    frameCapture?.beginQueueUpload();
    this.device.queue.submit([encoder.finish()]);

    this.lastTime_s += dt_s;
    const nextTime_s = this.lastTime_s;
    this.info.submittedTime_s = nextTime_s;
    this.info.simulatedTime_s = nextTime_s;
    this.info.simulationLag_s = Math.max(0, time_s - nextTime_s);
    this.info.lastDt_s = dt_s;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    this.info.lastSubsteps = 1;
    this.info.pressureIterations = 128;
    this.info.hostSimulationSizedWorkItems = 0;
    const captured = frameCapture?.finish(this.device.queue);
    this.finishFrameCapture(captured, traceRequestedAt_ms);
    return true;
  }

  private finishFrameCapture(
    captured: ReturnType<AdaptiveMassFrameCapture["finish"]> | undefined,
    traceRequestedAt_ms: number,
  ): void {
    if (captured) {
      this.lastPhysicsTraceAt_ms = traceRequestedAt_ms;
      this.physicsTracePending = true;
      this.info.physicsCPUTrace = captured.cpuTrace;
      this.info.physicsCaptureIdentity = captured.identity;
      void captured.queueTrace.then((trace) => {
        const current = usePerformanceInstrumentationStore.getState();
        if (!this.disposed && current.enabled
          && current.enabledAt_ms <= traceRequestedAt_ms) {
          this.info.physicsTrace = trace;
        }
      }).catch(() => {}).finally(() => {
        this.physicsTracePending = false;
      });
    }
  }

  async readStats(): Promise<GPUEulerianInfo> {
    await this.device.queue.onSubmittedWorkDone();
    const diagnostics = await this.resident.readDiagnostics();
    this.info.pressureRelativeResidual = diagnostics.pressureRelativeResidual;
    this.info.maxDivergenceAfter_s = diagnostics.maximumDivergence_s;
    this.info.maxDivergence_s = diagnostics.maximumDivergence_s;
    const adaptiveInfo = this.info as typeof this.info & AdaptiveMassStepTelemetry;
    adaptiveInfo.adaptiveMaximumMixedSeamDivergence_s =
      diagnostics.maximumMixedSeamDivergence_s;
    adaptiveInfo.adaptiveMaximumInactiveFaceSpeedAfter_m_s = 0;
    this.info.adaptiveActivityMaximumScore = diagnostics.activityMaximumScore;
    this.info.adaptiveActivityMeasuredBrickCount = diagnostics.activityMeasuredBrickCount;
    this.info.adaptiveActivitySurfaceBrickCount = diagnostics.activitySurfaceBrickCount;
    this.info.adaptiveActivityHotBrickCount = diagnostics.activityHotBrickCount;
    this.info.adaptiveActivityQuietBrickCount = diagnostics.activityQuietBrickCount;
    this.info.adaptiveResolutionTopologyEpoch = diagnostics.activityTopologyEpoch;
    this.info.activeSampleCount = diagnostics.activeCellCount;
    this.info.activeCompressionRatio = diagnostics.activeCellCount
      / Math.max(1, this.info.equivalentUniformCells ?? diagnostics.activeCellCount);
    this.info.fluidBrickResidentCount = diagnostics.activeBrickCount;
    this.info.fluidBrickCoreCount = diagnostics.activeBrickCount;
    this.info.fluidBrickGeneration = this.atlas.generation + diagnostics.residencyGeneration;
    // This rung measures and retains history only. Candidate topology has no
    // authority yet, so an epoch must publish exact zero transition counts.
    this.info.adaptiveResolutionPromotedBrickCount = 0;
    this.info.adaptiveResolutionDemotedBrickCount = 0;
    this.info.adaptiveResolutionDeferredPromotionCount = 0;
    this.info.completedTime_s = Math.max(
      this.info.completedTime_s ?? 0,
      this.info.submittedTime_s ?? 0,
    );
    return { ...this.info };
  }

  /** Explicit acceptance/debug readback; never consulted by advanceTo. */
  async readGPUActivityPolicy(): Promise<{
    readonly acceptedSteps: number;
    readonly bricks: readonly AdaptiveMassGPUActivityBrick[];
  }> {
    const snapshot = await this.resident.readActivitySnapshot();
    if (snapshot.records.length !== this.atlas.bricks.length) {
      throw new Error("Sparse CM12 GPU activity record count does not match resident bricks");
    }
    return {
      acceptedSteps: snapshot.acceptedSteps,
      bricks: snapshot.records.map((record, index) => {
        const brick = this.atlas.bricks[index]!;
        return { ...record, key: brick.key, coordinate: brick.coordinate,
          resolution: brick.resolution };
      }),
    };
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.resident.destroy();
    this.presentation.destroy();
  }
}

export function adaptiveMassPresentationDimensionsForScene(
  scene: SceneDescription,
): SparseBrickVec3 {
  return sceneLatticeDimensions(scene);
}

function finestCellSize(scene: SceneDescription, atlas: SparseAdaptiveMassAtlas): number {
  return Math.min(
    scene.container.width_m / atlas.dimensions[0],
    scene.container.height_m / atlas.dimensions[1],
    scene.container.depth_m / atlas.dimensions[2],
  );
}
