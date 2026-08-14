import { GPUInitializationTaskRunner } from "../../core/gpu-initialization";
import type { GPUQuality } from "../../core/gpu-quality";
import type {
  GPUInitializationReporter,
  GPUSolverInstance,
} from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import type { RigidBodyState } from "../../core/rigid-body";
import { sceneLatticeDimensions } from "../../core/scene-lattice";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import type { GPUEulerianInfo, GPURigidLoad } from "../../core/webgpu-eulerian";
import {
  ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS,
  AdaptiveMassFrameCapture,
} from "./adaptive-mass-frame-pipeline";
import type { AdaptiveMassSolverOptions } from "./method";
import {
  coarsenLargeQuiescentComponents,
  initializeSparseBrickAtlasFromScene,
  sparseBrickAtlasStats,
  type SparseAdaptiveMassAtlas,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";
import {
  materializeSparseAtlasDivergence,
  materializeSparseAtlasPressure,
  type SparseAtlasProjectionResult,
} from "./sparse-atlas-composite-projection";
import {
  initializeSparseAtlasDynamics,
  materializeSparseAtlasDynamicsVelocityRgba,
  stepSparseAtlasDynamics,
  type SparseAtlasDynamicsState,
} from "./sparse-atlas-dynamics";
import {
  materializeAdaptiveMassPresentationAtlas,
  WebGPUAdaptiveMassAtlasPresentation,
  type AdaptiveMassAtlasMaterialization,
} from "./webgpu-adaptive-mass-atlas-presentation";

/**
 * Browser milestone for the adaptive-mass method.
 *
 * Physics authority is a compact CPU f64 atlas for now. It owns force,
 * conservative transport, and global composite projection across arbitrary
 * resident 4³/8³ bricks; WebGPU owns the consumer textures and ordinary
 * renderer lifecycle. This does not yet claim the final GPU page pool, GPU
 * sparse execution, or dynamic camera/activity resolution policy.
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
  private dynamics: SparseAtlasDynamicsState;
  private lastTime_s = 0;
  private initialMassFineCells: number;
  private disposed = false;
  private physicsTraceSampleId = 0;
  private physicsTracePending = false;
  private lastPhysicsTraceAt_ms = -Infinity;

  private constructor(
    private readonly device: GPUDevice,
    private readonly scene: SceneDescription,
    private readonly options: AdaptiveMassSolverOptions,
    private readonly presentation: WebGPUAdaptiveMassAtlasPresentation,
    atlas: SparseAdaptiveMassAtlas,
    dynamics: SparseAtlasDynamicsState,
    quality: GPUQuality,
  ) {
    this.atlas = atlas;
    this.dynamics = dynamics;
    this.volumeTexture = presentation.densityTexture;
    this.surfaceFieldTexture = presentation.levelSetTexture;
    this.gridCellTexture = presentation.gridCellTexture;
    this.velocityTexture = presentation.velocityTexture;
    this.gridPressureTexture = presentation.pressureTexture;
    this.gridDivergenceTexture = presentation.divergenceTexture;
    const stats = sparseBrickAtlasStats(atlas);
    this.initialMassFineCells = stats.integratedMassFineCells;
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
      pressureSolver: "Sparse composite GᵀWG Jacobi-PCG",
      allocatedBytes: presentation.allocatedBytes + stats.leafCount * 16,
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
      adaptiveMixedSeamFaceCount: dynamics.grid.mixedSeamRowCount,
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
      hostSimulationSizedWorkItems: stats.leafCount,
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
    let materialization: AdaptiveMassAtlasMaterialization | undefined;
    let dynamics: SparseAtlasDynamicsState | undefined;
    try {
      await runner.run([{
        id: "adaptive-mass.plan",
        phase: "planning",
        label: "Bound the arbitrary-scene presentation lattice",
        run: () => { dimensions = boundedPresentationDimensions(scene); },
      }, {
        id: "adaptive-mass.atlas",
        phase: "adaptive-topology",
        label: "Build resident 4³/8³ sparse bricks",
        dependencies: ["adaptive-mass.plan"],
        run: () => {
          atlas = initializeSparseBrickAtlasFromScene(scene, {
            finestDimensions: dimensions!,
            maximumFinestCells: 64 ** 3,
            fineHalf: {
              axis: options.seamAxis === "x" ? 0 : options.seamAxis === "y" ? 1 : 2,
              side: options.fineSide,
            },
          });
          atlas = coarsenLargeQuiescentComponents(atlas, 8, {
            axis: options.seamAxis === "x" ? 0 : options.seamAxis === "y" ? 1 : 2,
            side: options.fineSide,
          });
          dynamics = initializeSparseAtlasDynamics(atlas);
          materialization = atlasMaterialization(atlas, scene, dynamics);
        },
      }, {
        id: "adaptive-mass.presentation",
        phase: "allocation",
        label: "Allocate adaptive water and ownership textures",
        dependencies: ["adaptive-mass.atlas"],
        run: () => { presentation = new WebGPUAdaptiveMassAtlasPresentation(device, dimensions!); },
      }, {
        id: "adaptive-mass.upload",
        phase: "upload",
        label: "Publish sparse atlas generation zero",
        dependencies: ["adaptive-mass.presentation"],
        run: () => { presentation!.upload(materialization!); },
      }, {
        id: "adaptive-mass.warmup",
        phase: "warmup",
        label: "Fence adaptive presentation generation zero",
        dependencies: ["adaptive-mass.upload"],
        run: () => device.queue.onSubmittedWorkDone(),
      }]);
      return new WebGPUAdaptiveMassSolver(
        device, scene, options, presentation!, atlas!, dynamics!, quality,
      );
    } catch (error) {
      presentation?.destroy();
      throw error;
    }
  }

  advanceTo(time_s: number, _bodies: RigidBodyState[]): boolean {
    void _bodies;
    if (this.disposed || !Number.isFinite(time_s) || time_s <= this.lastTime_s + 1e-9) return false;
    const dt_s = Math.min(this.scene.numerics.maxDt_s, time_s - this.lastTime_s);
    if (!(dt_s > 0)) return false;
    const cellSize_m = finestCellSize(this.scene, this.atlas);
    const gravity = this.scene.fluid.gravity_m_s2;
    const instrumentation = usePerformanceInstrumentationStore.getState();
    const traceRequestedAt_ms = instrumentation.enabled ? performance.now() : 0;
    const shouldTracePhysics = instrumentation.enabled && !this.physicsTracePending
      && traceRequestedAt_ms - this.lastPhysicsTraceAt_ms
        >= ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS;
    const traceSampleId = shouldTracePhysics ? ++this.physicsTraceSampleId : 0;
    const traceContext = `adaptive-mass:sim-${(this.lastTime_s + dt_s).toFixed(6)}`;
    const frameCapture = shouldTracePhysics
      ? new AdaptiveMassFrameCapture(traceSampleId, traceContext)
      : undefined;
    const step = stepSparseAtlasDynamics(this.dynamics, {
      dt_s,
      accelerationFinePerSecond2: [
        gravity.x / cellSize_m,
        gravity.y / cellSize_m,
        gravity.z / cellSize_m,
      ],
      maximumCfl: 0.8,
      projection: {
        // The public physics gate is 1e-8. A 100x guard band avoids spending
        // CPU-reference frame time on invisible residual digits while keeping
        // the accepted solution comfortably inside that unchanged gate.
        relativeTolerance: 1e-10,
        absoluteTolerance: 1e-12,
        onStageComplete: frameCapture?.completeProjectionStage,
      },
      onStageComplete: frameCapture?.completeDynamicsStage,
    });
    this.atlas = step.atlas;
    this.dynamics = step.state;
    this.lastTime_s = step.state.time_s;
    const stats = sparseBrickAtlasStats(this.atlas);
    const projection = step.projection;
    const nextTime_s = step.state.time_s;
    this.info.submittedTime_s = nextTime_s;
    this.info.simulatedTime_s = nextTime_s;
    this.info.simulationLag_s = Math.max(0, time_s - nextTime_s);
    this.info.lastDt_s = dt_s;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    this.info.volumeCellSum = stats.integratedMassFineCells;
    this.info.representedVolumeCellSum = stats.integratedMassFineCells;
    this.info.representedVolumeDrift = stats.integratedMassFineCells - this.initialMassFineCells;
    this.info.fluidBrickGeneration = stats.generation;
    this.info.adaptiveFineBrickCount = stats.fineBrickCount;
    this.info.adaptiveCoarseBrickCount = stats.coarseBrickCount;
    this.info.adaptiveFineCoarseFaceConnectedPairCount =
      stats.fineCoarseFaceConnectedPairCount;
    this.info.fluidBrickResidentCount = stats.residentBrickCount;
    this.info.fluidBrickCoreCount = stats.residentBrickCount;
    this.info.cellCount = stats.leafCount;
    this.info.activeSampleCount = stats.leafCount;
    this.info.compressionRatio = stats.leafCompressionRatio;
    this.info.activeCompressionRatio = stats.leafCompressionRatio;
    this.info.lastSubsteps = step.stats.transportSubsteps;
    this.info.maxComponentCfl = step.stats.maximumOutgoingCfl;
    this.info.adaptiveMixedSeamFaceCount = projection?.receipt.mixedSeamRowCount ?? 0;
    this.info.hostSimulationSizedWorkItems = step.stats.workCellCount
      + step.stats.workFaceCount;
    frameCapture?.completeStateCommit();
    const materialization = atlasMaterialization(
      this.atlas, this.scene, this.dynamics, step.projection, dt_s,
    );
    if (projection) this.publishProjectionInfo(projection, materialization);
    frameCapture?.completeMaterialization();
    frameCapture?.beginQueueUpload();
    this.presentation.upload(materialization);
    const captured = frameCapture?.finish(this.device.queue);
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
    return true;
  }

  private publishProjectionInfo(
    projection: SparseAtlasProjectionResult,
    materialization: AdaptiveMassAtlasMaterialization,
  ): void {
    const receipt = projection.receipt;
    this.info.pressureIterations = receipt.iterations;
    this.info.pressureRelativeResidual = receipt.relativeResidualL2;
    this.info.pressureResidual = receipt.maximumResidual;
    this.info.maxDivergenceBefore_s = receipt.preDivergenceMaximum;
    this.info.maxDivergenceAfter_s = receipt.postDivergenceMaximum;
    this.info.maxDivergence_s = this.info.maxDivergenceAfter_s;
    this.info.projectionDivergenceRatio = receipt.divergenceReduction;
    this.info.maxPressure_Pa = maximumAbsolute(materialization.pressure ?? []);
    this.info.maxSpeed_m_s = maximumVelocityMagnitude(materialization.velocity ?? []);
    this.info.pressureRequiredRows = receipt.activeRowCount;
    this.info.pressureRowCapacity = projection.grid.gradientRows.length;
    this.info.pressureCapacityOverflow = false;
    this.info.nonFiniteCount = countNonFinite(
      materialization.velocity ?? [], materialization.pressure ?? [],
      materialization.divergence ?? [],
    );
  }

  async readStats(): Promise<GPUEulerianInfo> {
    await this.device.queue.onSubmittedWorkDone();
    this.info.completedTime_s = Math.max(
      this.info.completedTime_s ?? 0,
      this.info.submittedTime_s ?? 0,
    );
    return { ...this.info };
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.presentation.destroy();
  }
}

function boundedPresentationDimensions(scene: SceneDescription): SparseBrickVec3 {
  const authored = sceneLatticeDimensions(scene);
  const scale = Math.min(1, 64 / Math.max(...authored));
  return authored.map((value) => Math.max(8, Math.min(64,
    Math.max(1, Math.round(value * scale / 8)) * 8))) as [number, number, number];
}

function atlasMaterialization(
  atlas: SparseAdaptiveMassAtlas,
  scene: SceneDescription,
  dynamics?: SparseAtlasDynamicsState,
  projection?: SparseAtlasProjectionResult,
  dt_s?: number,
): AdaptiveMassAtlasMaterialization {
  const finestCell_m = finestCellSize(scene, atlas);
  const base = materializeAdaptiveMassPresentationAtlas({
    dimensions: atlas.dimensions,
    emptyLevelSet: 4 * finestCell_m,
    densityProxyBand: 4 * finestCell_m,
    bricks: atlas.bricks.map((brick) => ({
      originFine: brick.coordinate.map((value) => value * 8) as [number, number, number],
      resolution: brick.resolution,
      fineSpan: 8,
      density: brick.density,
    })),
  });
  const velocity = dynamics
    ? materializeSparseAtlasDynamicsVelocityRgba(dynamics) : undefined;
  if (velocity) scaleInPlace(velocity, finestCell_m, 4, 3);
  const pressure = projection && dt_s
    ? materializeSparseAtlasPressure(projection) : undefined;
  if (pressure && dt_s) {
    scaleInPlace(
      pressure,
      scene.fluid.density_kg_m3 * finestCell_m * finestCell_m / dt_s,
    );
  }
  const divergence = projection
    ? materializeSparseAtlasDivergence(projection) : undefined;
  return { ...base, velocity, pressure, divergence };
}

function finestCellSize(scene: SceneDescription, atlas: SparseAdaptiveMassAtlas): number {
  return Math.min(
    scene.container.width_m / atlas.dimensions[0],
    scene.container.height_m / atlas.dimensions[1],
    scene.container.depth_m / atlas.dimensions[2],
  );
}

function scaleInPlace(
  values: Float32Array,
  scale: number,
  stride = 1,
  scaledChannels = stride,
): void {
  for (let base = 0; base < values.length; base += stride) {
    for (let channel = 0; channel < scaledChannels; channel += 1) {
      values[base + channel] *= scale;
    }
  }
}

function maximumAbsolute(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < values.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(values[index]));
  }
  return maximum;
}

function maximumVelocityMagnitude(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index + 2 < values.length; index += 4) {
    maximum = Math.max(maximum, Math.hypot(
      values[index], values[index + 1], values[index + 2],
    ));
  }
  return maximum;
}

function countNonFinite(...fields: readonly ArrayLike<number>[]): number {
  let count = 0;
  for (const field of fields) for (let index = 0; index < field.length; index += 1) {
    if (!Number.isFinite(field[index])) count += 1;
  }
  return count;
}
