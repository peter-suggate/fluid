import { GPUInitializationTaskRunner } from "../../core/gpu-initialization";
import type { GPUQuality } from "../../core/gpu-quality";
import type {
  GPUInitializationReporter,
  GPUSolverInstance,
  InjectedLiquidBall,
  MethodParamValues,
} from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import { initializeRigidBodies, type RigidBodyState } from "../../core/rigid-body";
import { sceneCellSizes_m, sceneLatticeDimensions } from "../../core/scene-lattice";
import {
  refinementRegionLattice,
  sceneRefinementRegions,
} from "../../core/refinement-regions";
import { GPUStageTimestampRecorder } from "../../core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { CM12_PAPER_DT_S } from "../../core/cm12-numerics";
import { averageInflowStrength, inflowOutletCenter } from "../../core/inflow-boundary";
import {
  GPU_RIGID_EXCHANGE_BYTES,
  type GPUEulerianInfo,
  type GPURigidLoad,
} from "../../core/webgpu-eulerian";
import { WebGPURigidBodySystem } from "../../core/webgpu-rigid-body";
import { fluidSolidWorldForScene } from
  "../../core/solid-world";
import {
  ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS,
  AdaptiveMassFrameCapture,
} from "./adaptive-mass-frame-pipeline";
import type { AdaptiveMassSolverOptions } from "./method";
import {
  initializeSparseBrickAtlasFromScene,
  materializeSparseBrickAtlasDensity,
  sparseBrickFromDense,
  sparseBrickAtlasStats,
  type SparseAdaptiveMassAtlas,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";
import {
  buildSparseAtlasCompositeGrid,
  type SparseAtlasCompositeGrid,
} from "./sparse-atlas-composite-projection";
import { SparseCM12PressureTopologyAttributionTracker } from
  "./sparse-cm12-pressure-topology-attribution";
import { packSparseCM12RefinementRegions } from
  "./sparse-cm12-refinement-regions";
import { WebGPUAdaptiveMassSparsePresentation } from
  "./webgpu-adaptive-mass-atlas-presentation";
import type { SparseWorld, SparseWorldDevice, SparseWorldUI } from "../../sparse-world";
import {
  createCM12SparseWorld,
  type CM12SparseWorldDeveloperTrace,
  type CM12SparseWorldRuntime,
  type CM12SparseWorldStepConfiguration,
} from "../../sparse-world/internal/cm12-adapter";
import {
  sparseCM12ActivityPolicy,
  sparseCM12PressureIterations,
  sparseCM12PressureIterationsFromReceipt,
  sparseCM12PressureRelativeTolerance,
  sparseCM12SharpeningDistance,
  sparseCM12SharpeningTraceSteps,
  type SparseCM12GPUActivityRecord,
} from "./webgpu-sparse-cm12-resident";

const PRESENTATION_PUBLISHER_ORACLE_QA_TOKEN: unique symbol =
  Symbol("Sparse CM12 presentation publisher oracle QA");
const PHASE1_TRANSPORT_RECEIPT_QA_TOKEN: unique symbol =
  Symbol("Sparse CM12 Phase-1 transport receipt QA");

export interface AdaptiveMassFluidDomain {
  readonly dimensions: SparseBrickVec3;
  readonly origin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
}

/** The authored lattice is only the initial fluid world; growth is demand-led. */
export function adaptiveMassFluidDomainForScene(
  scene: SceneDescription,
): AdaptiveMassFluidDomain {
  const tankDimensions = sceneLatticeDimensions(scene) as SparseBrickVec3;
  const cell = sceneCellSizes_m(scene);
  return {
    dimensions: tankDimensions,
    origin_m: [-0.5 * tankDimensions[0] * cell[0], 0,
      -0.5 * tankDimensions[2] * cell[2]],
    cellSize_m: cell,
  };
}

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

/**
 * Read-only receipt from the device scheduler. `advanceTo` never consumes this
 * shape: it exists only so explicit diagnostics can explain what the fixed GPU
 * dispatch chain accepted and what remains queued.
 */
interface SparseCM12TopologySchedulerDiagnostics {
  readonly acceptedTopologyGeneration?: number;
  readonly topologyUrgentQueuedBrickCount?: number;
  readonly topologyOrdinaryQueuedBrickCount?: number;
  readonly topologyPreparedBrickCount?: number;
  readonly topologyCommittedBrickCount?: number;
  readonly topologyDeferredBrickCount?: number;
  readonly acceptedFineBrickCount?: number;
  readonly acceptedCoarseBrickCount?: number;
}

/**
 * GPU-resident Sparse CM12 authority. Construction may build compact topology
 * on the host, but every accepted frame is device-only simulation work: the
 * host writes one small uniform block and encodes a fixed dispatch schedule.
 */
export class WebGPUAdaptiveMassSolver implements GPUSolverInstance {
  readonly sparseWorld: SparseWorld;
  readonly sparseWorldDevice: SparseWorldDevice;
  readonly sparseWorldUI: SparseWorldUI;
  readonly info: GPUEulerianInfo;
  readonly fluidDomain: NonNullable<GPUSolverInstance["fluidDomain"]>;
  get simulationReady(): boolean {
    return this.sparseWorldDevice.status === "ready";
  }
  async waitForSimulationReady(): Promise<void> {
    await this.sparseRuntime.waitForSimulationPipelines();
    void this.simulationReady;
  }
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly gridCellTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly gridPressureTexture: GPUTexture;
  readonly gridDivergenceTexture: GPUTexture;
  readonly initialSparseAuthorityReady = true;
  /** Compatibility getters backed exclusively by the public world view. */
  get sparseAdaptiveGridSource() { return this.sparseWorld.presentation().adaptiveGrid; }
  get globalFineLevelSetSource() { return this.sparseWorld.presentation().fineLevelSet; }
  readPresentationPageAllocatorReceiptQA() {
    return this.sparseWorldTrace.readPresentationPageAllocatorReceiptQA();
  }
  readWorldGrowthReceiptQA() {
    return this.sparseWorldTrace.readWorldGrowthReceiptQA();
  }
  private atlas: SparseAdaptiveMassAtlas;
  private lastTime_s = 0;
  private disposed = false;
  private physicsTraceSampleId = 0;
  private physicsTracePending = false;
  private lastPhysicsTraceAt_ms = -Infinity;
  /** Small staging ring carrying completed-frame pressure demand to the host. */
  private readonly pressureIterationReadbacks: GPUBuffer[] = [];
  private pressureIterationReceipt?: Readonly<{ executed: number; encoded: number }>;
  private pressureIterationReceiptSequence = 0;
  private pressureIterationReceiptAppliedSequence = 0;
  private pressureIterationControlGeneration = 0;
  /** One undecodable hardware sample retires the chain for this solver. */
  private hardwarePhysicsTraceInvalid = false;
  /** Diagnostics-only prior terminal tuple. Pressure topology precedes the
   * current frame's topology commit, so UI attribution must lag that commit. */
  private readonly pressureTopologyAttribution =
    new SparseCM12PressureTopologyAttributionTracker();

  private constructor(
    private readonly device: GPUDevice,
    // Not readonly: `applySceneUniforms` swaps in scalar-only scene revisions,
    // and `applyRuntimeValues` swaps the clock lane. Both are read fresh on
    // every advance rather than baked into an allocation, which is the whole
    // reason they can be adopted instead of rebuilt for.
    private scene: SceneDescription,
    private options: AdaptiveMassSolverOptions,
    private readonly presentation: WebGPUAdaptiveMassSparsePresentation,
    sparseWorldDevice: SparseWorldDevice,
    sparseWorld: SparseWorld,
    sparseWorldUI: SparseWorldUI,
    private readonly sparseRuntime: CM12SparseWorldRuntime,
    readonly sparseWorldTrace: CM12SparseWorldDeveloperTrace,
    private readonly sparseWorldNumerics: { current: CM12SparseWorldStepConfiguration },
    private readonly rigidSystem: WebGPURigidBodySystem | undefined,
    private readonly rigidExchange: GPUBuffer | undefined,
    private readonly rigidCouplingEnabled: boolean,
    adaptiveMixedSeamFaceCount: number,
    atlas: SparseAdaptiveMassAtlas,
    quality: GPUQuality,
  ) {
    this.sparseWorldDevice = sparseWorldDevice;
    this.sparseWorld = sparseWorld;
    this.sparseWorldUI = sparseWorldUI;
    this.atlas = atlas;
    const tankCellSize_m = sceneCellSizes_m(scene);
    this.fluidDomain = {
      origin_m: [-0.5 * atlas.dimensions[0] * tankCellSize_m[0], 0,
        -0.5 * atlas.dimensions[2] * tankCellSize_m[2]],
      cellSize_m: tankCellSize_m,
      dimensions: atlas.dimensions,
    };
    this.volumeTexture = presentation.densityTexture;
    this.surfaceFieldTexture = presentation.levelSetTexture;
    this.gridCellTexture = presentation.gridCellTexture;
    this.velocityTexture = presentation.velocityTexture;
    this.gridPressureTexture = presentation.pressureTexture;
    this.gridDivergenceTexture = presentation.divergenceTexture;
    const stats = sparseBrickAtlasStats(atlas);
    const [nx, ny, nz] = atlas.dimensions;
    const representedFraction = stats.leafCount / Math.max(1, stats.equivalentFinestCellCount);
    const cellSize_m = Math.min(...tankCellSize_m);
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
      pressureSolver: "GPU-resident one-reduction composite GᵀWG sparse MGPCG",
      allocatedBytes: presentation.allocatedBytes + sparseRuntime.allocatedBytes,
      quality,
      volumeCellSum: stats.integratedMassFineCells,
      representedVolumeCellSum: stats.integratedMassFineCells,
      representedVolumeDrift: 0,
      volumeTelemetrySource: "adaptive-conservative-mass",
      fluidBrickCapacity: stats.residentBrickCount,
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

  /** QA-only immutable HEAD presentation publisher construction. */
  static createPresentationPublisherOracleForQA(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: AdaptiveMassSolverOptions,
    onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUAdaptiveMassSolver> {
    return this.createAsync(device, scene, quality, onRigidLoads, options,
      onProgress, signal, PRESENTATION_PUBLISHER_ORACLE_QA_TOKEN);
  }

  /** QA-only construction that reserves the raw Phase-1 transport receipt
   * arena. Ordinary solver options cannot enable this instrumentation. */
  static createPhase1TransportReceiptOracleForQA(
    device: GPUDevice, scene: SceneDescription, quality: GPUQuality,
    onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: AdaptiveMassSolverOptions, onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUAdaptiveMassSolver> {
    return this.createAsync(device, scene, quality, onRigidLoads, options,
      onProgress, signal, PHASE1_TRANSPORT_RECEIPT_QA_TOKEN);
  }

  /** Explicit construction surface for production compiled topology transport. */
  static createCompiledTopologyTransport(
    device: GPUDevice, scene: SceneDescription, quality: GPUQuality,
    onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: AdaptiveMassSolverOptions, onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUAdaptiveMassSolver> {
    return this.createAsync(device, scene, quality, onRigidLoads, options,
      onProgress, signal);
  }

  static async createAsync(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    _onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: AdaptiveMassSolverOptions,
    onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
    qaToken?: typeof PRESENTATION_PUBLISHER_ORACLE_QA_TOKEN
      | typeof PHASE1_TRANSPORT_RECEIPT_QA_TOKEN,
  ): Promise<WebGPUAdaptiveMassSolver> {
    const runner = new GPUInitializationTaskRunner(onProgress, signal);
    const fluidDomainPlan = adaptiveMassFluidDomainForScene(scene);
    const initialSolidWorld = fluidSolidWorldForScene(scene);
    // Compile the boundary chain's closing marker while the scene builds. A
    // recorder constructed before it exists closes on an empty pass, which
    // Metal never samples, and that one bad sample would retire hardware
    // timing for the whole run.
    void GPUStageTimestampRecorder.prepare(device);
    let dimensions: SparseBrickVec3 | undefined;
    let atlas: SparseAdaptiveMassAtlas | undefined;
    let presentation: WebGPUAdaptiveMassSparsePresentation | undefined;
    let grid: SparseAtlasCompositeGrid | undefined;
    let sparseRuntime: Awaited<ReturnType<
      typeof createCM12SparseWorld>> | undefined;
    const sparseWorldNumerics: { current: CM12SparseWorldStepConfiguration } = {
      current: { finestCellSize_m: 1, pressureScale: 1 },
    };
    // Static terrain is compiled into SolidWorld before resident construction.
    // This sidecar owns only moving-body voxelization and bilateral reaction.
    const rigidCouplingEnabled = scene.rigidBodies.length > 0;
    let rigidExchange: GPUBuffer | undefined;
    let rigidSystem: WebGPURigidBodySystem | undefined;
    if (rigidCouplingEnabled) {
      rigidExchange = device.createBuffer({
        label: "Sparse CM12 rigid exchange",
        size: GPU_RIGID_EXCHANGE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      rigidSystem = new WebGPURigidBodySystem(device, scene, rigidExchange);
      rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
    }
    const rigidInitializationTasks = (rigidSystem?.initializationTasks() ?? []).map(
      (task) => ({ ...task, dependencies: [...(task.dependencies ?? []),
        "adaptive-mass.resident"] }),
    );
    let initiallyActiveBrickKeys: ReadonlySet<number> | undefined;
    try {
      await runner.run([{
        id: "adaptive-mass.plan",
        phase: "planning",
        label: "Bound the arbitrary-scene presentation lattice",
        run: () => { dimensions = fluidDomainPlan.dimensions; },
      }, {
        id: "adaptive-mass.atlas",
        phase: "adaptive-topology",
        label: "Build resident dyadic sparse bricks",
        dependencies: ["adaptive-mass.plan"],
        run: () => {
          const fineResolution = options.brickFineResolution ?? 8;
          const coarseResolution = (fineResolution / 2) as SparseBrickResolution;
          const resolutionForBrick = options.resolutionMode === "all-fine"
            ? () => fineResolution
            : options.resolutionMode === "all-coarse"
              ? () => coarseResolution
              : undefined;
          atlas = initializeSparseBrickAtlasFromScene(scene, {
            finestDimensions: dimensions!,
            brickFineResolution: fineResolution,
            maximumMacroSpanBricks: options.maximumMacroSpanBricks,
            surfaceFineRings: options.surfaceFineRings,
            ...(resolutionForBrick ? { resolutionForBrick } : {}),
          });
          // Generation zero contains only authored fluid. Dry face neighbours
          // are admitted by the GPU frontier if and when a swept fluid course
          // demands them; logical extent never becomes a topology allocation.
          initiallyActiveBrickKeys = new Set(atlas.bricks.filter((brick) =>
            brick.density.some((density) => density > 0)).map((brick) => brick.key));
          // The runtime is GPU-resident from generation zero. Construct only
          // the topology oracle needed by the packer; the CPU dynamics state
          // used to allocate duplicate velocity, pressure, policy and
          // workspace graphs that were never stepped or returned.
          grid = buildSparseAtlasCompositeGrid(atlas);
        },
      }, {
        id: "adaptive-mass.presentation",
        phase: "allocation",
        label: "Allocate adaptive water and ownership textures",
        dependencies: ["adaptive-mass.atlas"],
        run: () => {
          presentation = new WebGPUAdaptiveMassSparsePresentation(device);
        },
      }, {
        id: "adaptive-mass.resident",
        phase: "allocation",
        label: "Pack compact GPU topology and allocate resident frame state",
        dependencies: ["adaptive-mass.atlas", "adaptive-mass.presentation"],
        run: async () => {
          const cellSize_m = finestCellSize(scene, atlas!);
          sparseWorldNumerics.current = {
            finestCellSize_m: cellSize_m,
            pressureScale: 1,
            origin_m: fluidDomainPlan.origin_m,
          };
          sparseRuntime = await createCM12SparseWorld({
            device,
            atlas: atlas!,
            grid: grid!,
            numerics: () => sparseWorldNumerics.current,
            initiallyActiveBrickKeys,
            rigid: rigidCouplingEnabled ? {
              bodies: rigidSystem!.stateBuffer,
              exchange: rigidExchange!,
              worldDimensions_m: fluidDomainPlan.dimensions.map((value, axis) =>
                value * fluidDomainPlan.cellSize_m[axis]) as [number, number, number],
            } : undefined,
            rigidSystem,
            // Sized from the iteration ceiling this solver was built with, so
            // the journal can hold the longest solve it will ever encode.
            journal: options.pressureJournal
              ? { iterationCapacity: sparseCM12PressureIterations(
                options.pressureIterations) }
              : undefined,
            presentationPageResolution:
              options.presentationPageResolution ?? options.brickFineResolution ?? 8,
            report: (label: string) => onProgress({
              phase: "allocation",
              taskId: "adaptive-mass.resident",
              label,
              completed: 3,
              total: 6,
            }),
            solidWorld: initialSolidWorld,
            refinementRegionParameters: packSparseCM12RefinementRegions(
              sceneRefinementRegions(scene), refinementRegionLattice(scene)),
            mode: qaToken === PRESENTATION_PUBLISHER_ORACLE_QA_TOKEN
              ? "presentation-publisher-qa"
              : qaToken === PHASE1_TRANSPORT_RECEIPT_QA_TOKEN
                ? "phase1-transport-receipt-qa"
                : "production",
          });
          if (rigidSystem) {
            const occupancy = sparseRuntime.runtime.solidWorldCollisionSource;
            if (!occupancy) {
              throw new Error("Adaptive rigid contact requires resident SolidWorld occupancy");
            }
            rigidSystem.setSolidWorldCollisionSource({
              ...occupancy,
              origin_m: fluidDomainPlan.origin_m,
              cellSize_m: fluidDomainPlan.cellSize_m,
            });
          }
        },
      }, ...rigidInitializationTasks, {
        id: "adaptive-mass.upload",
        phase: "upload",
        label: "Publish sparse atlas generation zero",
        dependencies: ["adaptive-mass.resident"],
        run: () => {
          const encoder = device.createCommandEncoder({
            label: "Sparse CM12 initial GPU publication",
          });
          sparseRuntime!.runtime.encodeInitialPresentation(
            encoder, finestCellSize(scene, atlas!));
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
        device, scene, options, presentation!, sparseRuntime!.device, sparseRuntime!.world,
        sparseRuntime!.ui,
        sparseRuntime!.runtime, sparseRuntime!.developerTrace, sparseWorldNumerics,
        rigidSystem, rigidExchange, rigidCouplingEnabled,
        grid!.mixedSeamRowCount, atlas!, quality,
      );
    } catch (error) {
      sparseRuntime?.world.destroy();
      presentation?.destroy();
      rigidSystem?.destroy();
      rigidExchange?.destroy();
      throw error;
    }
  }

  /** Add a semantic liquid interaction through the public sparse-world API. */
  injectLiquidBall(ball: InjectedLiquidBall): void {
    if (this.disposed || !(ball.radius_m > 0)) return;
    this.sparseWorld.edit({
      kind: "liquid-ellipsoid",
      center_m: [ball.centre_m.x, ball.centre_m.y, ball.centre_m.z],
      radii_m: [ball.radius_m, ball.radius_m,
        ball.halfHeight_m ?? ball.radius_m],
    });
  }

  /**
   * Adopt scene scalars on the running solver.
   *
   * Everything this method reads out of the document below — `numerics.maxDt_s`,
   * `fluid.gravity_m_s2`, `fluid.density_kg_m3` — is read per advance, never
   * baked into a buffer, a pipeline or an atlas. Without this the renderer had
   * no way to deliver a changed scalar except by constructing a new solver, so
   * nudging the step slider rebuilt the whole sparse world to arrive at an
   * identical one. The renderer only calls this once the structural and seed
   * tiers already match, so the incoming document differs in scalars alone.
   */
  applySceneUniforms(scene: SceneDescription): void {
    const receipt = this.sparseWorld.edit({ kind: "set-scene", scene });
    if (receipt.disposition !== "applied") {
      throw new Error(receipt.reason ?? "Sparse world requires a rebuild for this scene edit");
    }
    this.scene = scene;
    this.resetPressureIterationFeedback();
  }

  /**
   * Adopt the controls that only change what the next advance asks for.
   *
   * `timeStep` picks between the paper 1/30 s operating step and the scene's
   * authored `maxDt_s`; both are consulted at the top of `advanceTo`, so the
   * switch is a live one. Activity thresholds are likewise copied into the
   * next frame's small policy uniform. Structural capacity controls still
   * rebuild, while accepted resolution changes publish at topology epochs.
   */
  applyRuntimeValues(values: MethodParamValues): void {
    const timeStep = values.timeStep === "scene" ? "scene" : "paper";
    const sharpeningDistance = sparseCM12SharpeningDistance(values.sharpeningDistance);
    const sharpeningTraceSteps = sparseCM12SharpeningTraceSteps(values.sharpeningTraceSteps);
    const gammaDiffusionEnabled = values.gammaDiffusion !== "off";
    const surfaceSharpeningEnabled = values.surfaceSharpening !== "off";
    const pressureIterations = sparseCM12PressureIterations(values.pressureIterations);
    const pressureRelativeTolerance =
      sparseCM12PressureRelativeTolerance(values.pressureRelativeTolerance);
    if (pressureIterations !== this.options.pressureIterations
      || pressureRelativeTolerance !== this.options.pressureRelativeTolerance) {
      this.resetPressureIterationFeedback();
    }
    const activityPolicy = sparseCM12ActivityPolicy({
      ...values,
      activitySignals: values.selectorMode === "activity",
    });
    this.options = { ...this.options, timeStep, sharpeningDistance, sharpeningTraceSteps,
      gammaDiffusionEnabled, surfaceSharpeningEnabled,
      pressureIterations, pressureRelativeTolerance, activityPolicy };
  }

  private resetPressureIterationFeedback(): void {
    this.pressureIterationReceipt = undefined;
    this.pressureIterationControlGeneration += 1;
    delete this.info.pressureIterationsExecuted;
    delete this.info.pressureIterationsEncoded;
  }

  advanceTo(time_s: number, bodies: RigidBodyState[]): boolean {
    if (this.disposed || this.sparseWorldDevice.status !== "ready" || !Number.isFinite(time_s)
      || time_s <= this.lastTime_s + 1e-9) return false;
    const paperTimeStep = this.options.timeStep === "paper";
    if (paperTimeStep
      && time_s - this.lastTime_s < CM12_PAPER_DT_S - 1e-9) return false;
    const dt_s = paperTimeStep
      ? CM12_PAPER_DT_S
      : Math.min(this.scene.numerics.maxDt_s, time_s - this.lastTime_s);
    if (!(dt_s > 0)) return false;
    const pressureIterationMaximum = sparseCM12PressureIterations(
      this.options.pressureIterations);
    const pressureRelativeTolerance = sparseCM12PressureRelativeTolerance(
      this.options.pressureRelativeTolerance);
    const pressureIterations = this.sparseWorldUI.control.pressureFilm?.captureEnabled
      ? pressureIterationMaximum
      : sparseCM12PressureIterationsFromReceipt(
        pressureIterationMaximum,
        pressureRelativeTolerance,
        this.pressureIterationReceipt,
      );
    const cellSize_m = finestCellSize(this.scene, this.atlas);
    const inflow = this.scene.fluid.inflow;
    const inflowStrength = inflow
      ? averageInflowStrength(inflow, this.lastTime_s, this.lastTime_s + dt_s) : 0;
    const liquidInflow = inflow && inflowStrength > 0 ? (() => {
      const outlet = inflowOutletCenter(inflow);
      return {
        outlet_m: [
          outlet.x,
          outlet.y,
          outlet.z,
        ] as const,
        radius_m: inflow.radius_m,
        velocity_m_s: [
          inflow.velocity_m_s.x * inflowStrength,
          inflow.velocity_m_s.y * inflowStrength,
          inflow.velocity_m_s.z * inflowStrength,
        ] as const,
      };
    })() : undefined;
    const activeBodies = bodies.slice(0, 12);
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
    const rawEncoder = this.device.createCommandEncoder({
      label: `Sparse CM12 resident frame ${(this.lastTime_s + dt_s).toFixed(6)}`,
    });
    // The stage partition is the encoder's own: boundaries ride the passes the
    // advance already encodes, so a sampled advance dispatches exactly the
    // physics an unsampled one does. `markersReady` gates the recorder's
    // fallback closing marker, which an advance whose final stage encoded
    // nothing would fall back to; an unsampled boundary there would retire
    // hardware timing for the whole run.
    const hardwareTrace = frameCapture && !this.hardwarePhysicsTraceInvalid
      && GPUStageTimestampRecorder.supported(this.device)
      && GPUStageTimestampRecorder.markersReady(this.device)
      ? new GPUStageTimestampRecorder(this.device, traceSampleId, "physics", traceContext)
      : undefined;
    const encoder = frameCapture
      ? frameCapture.instrument(rawEncoder, hardwareTrace)
      : rawEncoder;
    if (this.pressureIterationReadbacks.length === 0) {
      for (let index = 0; index < 3; index += 1) {
        this.pressureIterationReadbacks.push(this.device.createBuffer({
          label: `Sparse CM12 pressure-iteration receipt ${index}`,
          size: 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }));
      }
    }
    const pressureIterationReadback = this.pressureIterationReadbacks.find(
      (candidate) => candidate.mapState === "unmapped",
    );
    const pressureIterationReceiptSequence = pressureIterationReadback
      ? ++this.pressureIterationReceiptSequence : 0;
    const pressureIterationControlGeneration = this.pressureIterationControlGeneration;
    this.sparseWorldNumerics.current = {
      finestCellSize_m: cellSize_m,
      pressureScale: this.scene.fluid.density_kg_m3 * cellSize_m * cellSize_m / dt_s,
      origin_m: this.fluidDomain.origin_m,
      accelerationFinePerSecond2: [
        gravity.x / cellSize_m,
        gravity.y / cellSize_m,
        gravity.z / cellSize_m,
      ],
      sharpening: {
        distanceCells: this.options.sharpeningDistance,
        traceSteps: this.options.sharpeningTraceSteps,
        gammaDiffusionEnabled: this.options.gammaDiffusionEnabled,
        surfaceSharpeningEnabled: this.options.surfaceSharpeningEnabled,
      },
      activityPolicy: this.options.activityPolicy,
      pressureControl: {
        iterations: pressureIterations,
        relativeTolerance: pressureRelativeTolerance,
      },
      seams: frameCapture?.residentStageSeams,
      worldDimensions_m: this.fluidDomain.dimensions.map((value, axis) =>
        value * this.fluidDomain.cellSize_m[axis]) as [number, number, number],
    };
    this.sparseWorld.encodeStep(encoder, {
      time: this.lastTime_s + dt_s,
      dt: dt_s,
      gravity: [gravity.x, gravity.y, gravity.z],
      rigidBodies: activeBodies,
      liquidInflow,
    });
    if (pressureIterationReadback) {
      this.sparseRuntime.encodePressureIterationReceipt(
        encoder, pressureIterationReadback);
    }
    frameCapture?.closeCommands();
    this.device.queue.submit([encoder.finish()]);
    if (liquidInflow) {
      this.sparseWorld.edit({
        kind: "liquid-jet",
        ...liquidInflow,
        dt: dt_s,
      });
      // Create only this step's nozzle-swept volume. The next CM12 step owns
      // all downstream transport, projection, and gravitational curvature.
    }
    if (pressureIterationReadback) {
      this.readPressureIterationReceipt(
        pressureIterationReadback,
        pressureIterations,
        pressureIterationReceiptSequence,
        pressureIterationControlGeneration,
      );
    }

    this.lastTime_s += dt_s;
    const nextTime_s = this.lastTime_s;
    this.info.submittedTime_s = nextTime_s;
    this.info.simulatedTime_s = nextTime_s;
    this.info.simulationLag_s = Math.max(0, time_s - nextTime_s);
    this.info.lastDt_s = dt_s;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    this.info.lastSubsteps = 1;
    this.info.pressureIterations = pressureIterationMaximum;
    this.info.pressureIterationsEncoded = pressureIterations;
    this.info.hostSimulationSizedWorkItems = 0;
    const captured = frameCapture?.finish(this.device.queue);
    this.finishFrameCapture(captured, traceRequestedAt_ms);
    return true;
  }

  /** Publish the receipt and use it only as the next frame's encoded ceiling hint. */
  private readPressureIterationReceipt(
    readback: GPUBuffer,
    encoded: number,
    sequence: number,
    controlGeneration: number,
  ): void {
    void readback.mapAsync(GPUMapMode.READ).then(() => {
      const executed = Math.max(0, Math.round(
        new Float32Array(readback.getMappedRange(), 0, 1)[0]!,
      ));
      readback.unmap();
      if (this.disposed || !this.pressureIterationReadbacks.includes(readback)
        || controlGeneration !== this.pressureIterationControlGeneration
        || sequence <= this.pressureIterationReceiptAppliedSequence) return;
      const boundedExecuted = Math.min(encoded, executed);
      this.pressureIterationReceiptAppliedSequence = sequence;
      this.pressureIterationReceipt = { executed: boundedExecuted, encoded };
      this.info.pressureIterationsExecuted = boundedExecuted;
      this.info.pressureIterationsEncoded = encoded;
    }).catch(() => {
      if (readback.mapState === "mapped") readback.unmap();
    });
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
      // Prefer the hardware partition: it is the only lane that can put a
      // figure on an individual stage. One unusable sample retires it for this
      // solver and the queue-wall observation carries the advance from then on.
      const resolved = captured.hardwareTrace
        ? captured.hardwareTrace.then((trace) => {
          this.hardwarePhysicsTraceInvalid = !trace;
          return trace ?? captured.queueTrace;
        }).catch(() => {
          this.hardwarePhysicsTraceInvalid = true;
          return captured.queueTrace;
        })
        : captured.queueTrace;
      void Promise.resolve(resolved).then((trace) => {
        const current = usePerformanceInstrumentationStore.getState();
        if (trace && !this.disposed && current.enabled
          && current.enabledAt_ms <= traceRequestedAt_ms) {
          this.info.physicsTrace = trace;
        }
      }).catch(() => {}).finally(() => {
        this.physicsTracePending = false;
      });
    }
  }

  /**
   * Host-only view of asynchronously published timing. Unlike `readStats`,
   * this never fences the queue or maps the resident diagnostics buffer, so a
   * timestamp poll cannot accidentally submit dozens of full readbacks.
   */
  readPerformanceTraceSnapshot(): Pick<GPUEulerianInfo,
    "physicsTrace" | "physicsCPUTrace" | "physicsCaptureIdentity"> {
    return {
      physicsTrace: this.info.physicsTrace,
      physicsCPUTrace: this.info.physicsCPUTrace,
      physicsCaptureIdentity: this.info.physicsCaptureIdentity,
    };
  }

  async readStats(): Promise<GPUEulerianInfo> {
    await this.device.queue.onSubmittedWorkDone();
    const diagnostics = await this.sparseWorldTrace.readDiagnostics();
    // This full diagnostics readback remains downstream of simulation and only
    // updates panels. Adaptive encoding consumes the separate four-byte
    // completed-frame receipt, never this topology/physics diagnostics packet.
    const topology = diagnostics as typeof diagnostics
      & SparseCM12TopologySchedulerDiagnostics;
    this.info.adaptivePressureTopologyAttribution =
      this.pressureTopologyAttribution.observe({
        current: {
          encodedStep: this.info.encodedSteps ?? 0,
          topologyGeneration: topology.acceptedTopologyGeneration ?? 0,
          committedBrickCount: topology.topologyCommittedBrickCount ?? 0,
        },
        work: {
          acceptedCellCount: diagnostics.acceptedCellCount,
          acceptedRowCount: diagnostics.acceptedRowCount,
          pressureCellCount: diagnostics.pressureCellCount,
          pressureActiveRowCount: diagnostics.pressureActiveRowCount,
          pcm: diagnostics.pressureCanonicalMembership,
          authorities: diagnostics.pressureCutoverAuthorities,
        },
      });
    this.info.pressureRelativeResidual = diagnostics.pressureRelativeResidual;
    this.info.pressureRecursiveRelativeResidual =
      diagnostics.pressureRecursiveRelativeResidual;
    this.info.pressureTrueResidualMaximum = diagnostics.pressureTrueResidualMaximum;
    this.info.pressureInitialTrueRelativeResidual =
      diagnostics.pressureInitialTrueRelativeResidual;
    this.info.pressureIterationsExecuted = diagnostics.pressureIterationsExecuted;
    this.info.pressureIterationsEncoded = diagnostics.pressureIterationsEncoded;
    this.info.pressureFirstToleranceCrossingIteration =
      diagnostics.pressureFirstToleranceCrossingIteration;
    this.info.pressureSolveConverged = diagnostics.pressureSolveConverged;
    this.info.pressureIterationCapReached = diagnostics.pressureIterationCapReached;
    this.info.pressureConvergenceReason = diagnostics.pressureConvergenceReason;
    this.info.pressureCurvatureBreakdown = diagnostics.pressureCurvatureBreakdown;
    this.info.pressureCurvatureRecoveryCount =
      diagnostics.pressureCurvatureRecoveryCount;
    this.info.pressureRecursiveToTrueResidualRatio =
      diagnostics.pressureRecursiveToTrueResidualRatio;
    this.info.pressureResidualDrift = diagnostics.pressureResidualDrift;
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
    this.info.activeSampleCount = diagnostics.acceptedCellCount;
    this.info.activeCompressionRatio = diagnostics.acceptedCellCount
      / Math.max(1, this.info.equivalentUniformCells ?? diagnostics.acceptedCellCount);
    this.info.fluidBrickResidentCount = diagnostics.activeBrickCount;
    this.info.fluidBrickCoreCount = diagnostics.activeBrickCount;
    // Residency and accepted split/merge publication are independent GPU
    // generations. Their sum is a monotonic renderer-facing revision.
    this.info.fluidBrickGeneration = this.atlas.generation
      + diagnostics.residencyGeneration + (topology.acceptedTopologyGeneration ?? 0);
    this.info.adaptiveTopologyShadowGeneration =
      topology.acceptedTopologyGeneration ?? 0;
    this.info.adaptiveTopologyUrgentQueuedBrickCount =
      topology.topologyUrgentQueuedBrickCount ?? 0;
    this.info.adaptiveTopologyOrdinaryQueuedBrickCount =
      topology.topologyOrdinaryQueuedBrickCount ?? 0;
    this.info.adaptiveTopologyPreparedBrickCount = topology.topologyPreparedBrickCount ?? 0;
    this.info.adaptiveTopologyCommittedBrickCount = topology.topologyCommittedBrickCount ?? 0;
    this.info.adaptiveTopologyDeferredBrickCount = topology.topologyDeferredBrickCount ?? 0;
    this.info.adaptiveTopologyShadowFineBrickCount = topology.acceptedFineBrickCount;
    this.info.adaptiveTopologyShadowCoarseBrickCount = topology.acceptedCoarseBrickCount;
    this.info.adaptiveAcceptedCellCount = diagnostics.acceptedCellCount;
    this.info.adaptiveAcceptedRowCount = diagnostics.acceptedRowCount;
    this.info.adaptiveAcceptedSameLevelCoarseRowCount =
      diagnostics.acceptedSameLevelCoarseRowCount;
    this.info.adaptiveAcceptedMixedSeamRowCount = diagnostics.acceptedMixedSeamRowCount;
    this.info.adaptivePressureActiveRowCount = diagnostics.pressureActiveRowCount;
    this.info.adaptivePressureCellCount = diagnostics.pressureCellCount;
    this.info.adaptivePressureCanonicalMembership =
      diagnostics.pressureCanonicalMembership;
    this.info.adaptivePressureTopologyRepair = diagnostics.pressureTopologyRepair;
    // Keep the established diagnostics/benchmark field live while callers
    // migrate to the pressure-specific name above.
    this.info.adaptiveMixedSeamFaceCount = diagnostics.acceptedMixedSeamRowCount;
    this.info.adaptiveFineBrickCount = topology.acceptedFineBrickCount;
    this.info.adaptiveCoarseBrickCount = topology.acceptedCoarseBrickCount;
    this.info.adaptiveResolutionPromotedBrickCount = 0;
    this.info.adaptiveResolutionDemotedBrickCount = 0;
    this.info.adaptiveResolutionDeferredPromotionCount = 0;
    this.info.completedTime_s = Math.max(
      this.info.completedTime_s ?? 0,
      this.info.submittedTime_s ?? 0,
    );
    return { ...this.info };
  }

  /** Explicit Dawn/QA materialization; production rendering stays sparse. */
  readDiagnosticFields(includeWorldLeaves = false) {
    return this.sparseWorldTrace.readDiagnosticFields(includeWorldLeaves);
  }
  readPhase1TransportReceiptQA() {
    return this.sparseWorldTrace.readPhase1TransportReceiptQA();
  }
  readPhase1TransportProfileQA() {
    return this.sparseWorldTrace.readPhase1TransportProfileQA();
  }
  readCandidateEffectsTransactionQA() {
    return this.sparseWorldTrace.readCandidateEffectsTransactionQA();
  }
  readFramePlanPresentationHeaderQA() {
    return this.sparseWorldTrace.readFramePlanPresentationHeaderQA();
  }
  readFramePlanPresentationFaultRecordQA() {
    return this.sparseWorldTrace.readFramePlanPresentationFaultRecordQA();
  }
  /** Explicit FCA1 QA materialization; never consulted by frame scheduling. */
  readFrameControlQA() { return this.sparseWorldTrace.readFrameControlQA(); }
  readTransportPacketIndirectQA() {
    return this.sparseWorldTrace.readTransportPacketIndirectQA();
  }
  readDynamicTransportPacketsQA() {
    return this.sparseWorldTrace.readDynamicTransportPacketsQA();
  }
  /** Header-only FSM1 receipt; never consulted by frame scheduling. */
  readFinalScalarMaskHeaderQA() {
    return this.sparseWorldTrace.readFinalScalarMaskHeaderQA();
  }
  readSparseWorkShapeQA() { return this.sparseWorldTrace.readWorkShapeQA(); }
  readAdaptiveRepresentationQA() {
    return this.sparseWorldTrace.readAdaptiveRepresentationQA();
  }
  readAcceptedIndirectQA() { return this.sparseWorldTrace.readAcceptedIndirectQA(); }
  readFrameControlIndirectQA() {
    return this.sparseWorldTrace.readFrameControlIndirectQA();
  }
  readVelocityExtensionHeaderQA() {
    return this.sparseWorldTrace.readVelocityExtensionHeaderQA();
  }
  readVelocityExtensionQA() { return this.sparseWorldTrace.readVelocityExtensionQA(); }
  readPressureCanonicalMembershipQA() {
    return this.sparseWorldTrace.readPressureCanonicalMembershipQA();
  }

  get rigidRenderBuffer(): GPUBuffer | undefined { return this.rigidSystem?.renderBuffer; }
  get rigidMotionBuffer(): GPUBuffer | undefined { return this.rigidSystem?.motionBuffer; }
  setSelectedRigidBody(index: number): void { this.rigidSystem?.setSelectedIndex(index); }
  async pickRigidBody(origin: RigidBodyState["position_m"],
    direction: RigidBodyState["position_m"]) {
    return this.rigidSystem?.pick(origin, direction);
  }
  async readRigidBodyPoses() { return this.rigidSystem?.readPoses(); }

  /** Explicit acceptance/debug readback; never consulted by advanceTo. */
  async readGPUActivityPolicy(): Promise<{
    readonly acceptedSteps: number;
    readonly acceptedTopologyGeneration: number;
    readonly residentBrickCount: number;
    readonly faultFlags: number;
    readonly newlyActivatedBrickCount: number;
    readonly preparedBrickCount: number;
    readonly committedBrickCount: number;
    readonly commitFailed: boolean;
    readonly bricks: readonly AdaptiveMassGPUActivityBrick[];
  }> {
    const snapshot = await this.sparseWorldTrace.readActivitySnapshot(true);
    return {
      acceptedSteps: snapshot.acceptedSteps,
      acceptedTopologyGeneration: snapshot.acceptedTopologyGeneration,
      residentBrickCount: snapshot.residentBrickCount,
      faultFlags: snapshot.faultFlags,
      newlyActivatedBrickCount: snapshot.newlyActivatedBrickCount,
      preparedBrickCount: snapshot.preparedBrickCount,
      committedBrickCount: snapshot.committedBrickCount,
      commitFailed: snapshot.commitFailed,
      bricks: snapshot.records.map((record) => {
        const brick = this.atlas.bricks[record.leafId];
        if (brick) return { ...record, key: brick.key, coordinate: brick.coordinate,
          resolution: brick.resolution };
        if (!record.coordinate) {
          throw new Error(`Sparse CM12 dynamic leaf ${record.leafId} has no WDR coordinate`);
        }
        return { ...record, key: record.leafId, coordinate: record.coordinate,
          resolution: record.acceptedResolution };
      }),
    };
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.sparseWorld.destroy();
    this.presentation.destroy();
    this.rigidSystem?.destroy();
    this.rigidExchange?.destroy();
    for (const readback of this.pressureIterationReadbacks) readback.destroy();
    this.pressureIterationReadbacks.length = 0;
  }
}

export function adaptiveMassPresentationDimensionsForScene(
  scene: SceneDescription,
): SparseBrickVec3 {
  return adaptiveMassFluidDomainForScene(scene).dimensions;
}

function finestCellSize(scene: SceneDescription, _atlas: SparseAdaptiveMassAtlas): number {
  return Math.min(...sceneCellSizes_m(scene));
}
