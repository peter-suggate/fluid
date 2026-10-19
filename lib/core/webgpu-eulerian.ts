import type { SpatialPerformanceReport } from "./spatial-performance";
import type { SimulationFailure } from "./simulation-failure";
import type { PerformanceTrace } from "./performance-trace";
import type { GPUQuality } from "./gpu-quality";
import type { SolverDetailDiagnostics } from "./solver-detail";
export type GPUGridMethod = "octree";
export type GPUVelocityTransport = "semi-lagrangian" | "maccormack";

export interface GPUFieldLocation {
  x: number;
  y: number;
  z: number;
}

export interface GPUEulerianInfo {
  /** Narrow-band FLIP: the accepted detail plan (policy, requested/admitted, support, residency). */
  uniformDetail?: SolverDetailDiagnostics;
  uniformMixedGeneration?: number;
  uniformMixedFineTiles?: number;
  /** h tiles the owner-indexed storage holds, and the builder's admission under it (a staged return). */
  uniformMixedFineCapacity?: number;
  uniformMixedAdmission?: number;
  /** Builds deferred over the admission capacity, the h tiles they left waiting (summed per build), and capacity changes. */
  uniformMixedDeferredBuilds?: number;
  uniformMixedDeferredTiles?: number;
  uniformMixedCapacityChanges?: number;
  /** Completed dynamic ownership changes, including their field remap. */
  uniformMixedDynamicRelayouts?: number;
  uniformMixedRegularTiles?: number;
  uniformMixedGeneralTiles?: number;
  uniformMixedCoarseTiles?: number;
  uniformMixedOwners?: number;
  uniformCoarsePrepared?: boolean;
  uniformSimulationCellScale?: 1 | 4;
  /** Root solver field-load audit; not yet extension/pressure hierarchy coverage. */
  uniformPageMissingReadFields?: number;
  uniformPageMissingReads?: number;
  uniformDomainAuthority?: "pages";
  uniformDomainPages?: number;
  uniformDomainMigration?: string;
  uniformVolumeTransportWorkgroups?: number;
  uniformVolumeSharpenWorkgroups?: number;
  uniformVolumePageEdge?: number;
  uniformVolumePagesActive?: number;
  uniformVolumePagesTotal?: number;
  uniformVolumePageBytes?: number;
  uniformVolumePageStage?: string;
  uniformVolumePageTransportReceiptMs?: number;
  uniformVolumePageSharpenReceiptMs?: number;
  nx: number;
  ny: number;
  nz: number;
  /**
   * The cells of the lattice that lie in the tank, counted from the lattice's
   * low corner, when the solver's lattice (nx, ny, nz) runs past the tank's
   * high faces. A frame contours and slices these, at the tank's extents over
   * them a cell; absent, the lattice is the tank's.
   */
  tankCells?: readonly [number, number, number];
  storedNy: number;
  cellCount: number;
  equivalentUniformCells: number;
  compressionRatio: number;
  activeCompressionRatio?: number;
  activeSampleCount?: number;
  regularLayers: number;
  maximumNeighborDelta: number;
  gridKind: "octree" | "uniform";
  /** True only after the complete sparse t=0 authority passed its queue fence. */
  initialSparseAuthorityReady?: boolean;
  /** Whether the renderer has attached the initial water surface. */
  initialRasterSurfaceReady?: boolean;
  /** Presentation may be ready while non-critical simulation pipelines finish in the worker. */
  simulationPipelinesReady?: boolean;
  /** Background compilation failure, if a presentation-only solver could not become runnable. */
  simulationPipelineError?: string;
  simulationFailure?: SimulationFailure;
  /** Latest compact activity census, weighted by represented liquid volume. */
  adaptivePhysicalWidthCensus?: readonly { width: number; leaves: number; liquidVolumeFineCells: number }[];
  adaptivePhysicalWidthCensusStep?: number;
  topologyGenerationPending?: boolean;
  topologyPreparationMaximumSliceMs?: number;
  topologyPreparationMaximumSliceOperation?: string;
  topologyPublicationMaximumDurationMs?: number;
  topologyGenerationStaleCount?: number;
  topologyPreparationDurationMs?: number;
  topologyGenerationError?: string;
  topologyGenerationMaximumBytes?: number;
  topologyGenerationCount?: number;
  topologyGenerationRequestedLeaves?: number;
  topologyGenerationDeferred?: {
    leaves: number; cells: number; requestedBytes?: number; availableBytes?: number;
    reason?: "volume-capacity"; detail?: string;
  };
  /** Honest distinction between GPU-only authority, a readback-confirmed crossing, and fail-closed startup. */
  initialRasterSurfaceState?: "pending" | "gpu-authoritative" | "compact-confirmed" | "crossing-confirmed" | "failed-closed";
  initialRasterSurfaceDiagnostic?: string;
  cellSize_m: number;
  pressureIterations: number;
  pressureSolver?: string;
  allocatedBytes: number;
  /** First validation error captured by the solver's diagnostic error scope.
   * Reporting is asynchronous and never feeds simulation state. */
  gpuValidationError?: string;
  quality: GPUQuality;
  volumeCellSum?: number;
  representedVolumeCellSum?: number;
  representedVolumeDrift?: number;
  /** GPU field which supplied the displayed physical volume. */
  volumeTelemetrySource?: "dense-volume" | "narrow-band-phi" | "initial-condition" | "unavailable";
  front_m?: number;
  /** GPU field which supplied the displayed dam-front location. */
  frontTelemetrySource?: "dense-volume" | "initial-condition" | "unavailable";
  maxSpeed_m_s?: number;
  maxDivergence_s?: number;
  maxDivergenceBefore_s?: number;
  maxDivergenceAfter_s?: number;
  projectionDivergenceRatio?: number;
  maxAirSpeed_m_s?: number;
  /** Maximum physical pressure in Pa. */
  maxPressure_Pa?: number;
  pressureResidual?: number;
  /** Initial fresh infinity norm and effective per-step target, in s^-1. */
  pressureInitialResidual?: number;
  pressureResidualTarget?: number;
  /** Fresh b-Ap relative L2 residual. */
  pressureRelativeResidual?: number;
  /** Recursively updated CG residual, retained only to diagnose f32 drift. */
  pressureRecursiveRelativeResidual?: number;
  pressureTrueResidualMaximum?: number;
  pressureInitialTrueRelativeResidual?: number;
  pressureIterationsExecuted?: number;
  pressureIterationsEncoded?: number;
  pressureFirstToleranceCrossingIteration?: number;
  pressureSolveConverged?: boolean;
  pressureIterationCapReached?: boolean;
  pressureConvergenceReason?: "tolerance" | "iteration-cap" | "fixed-budget";
  pressureCurvatureBreakdown?: boolean;
  pressureCurvatureRecoveryCount?: number;
  pressureRecursiveToTrueResidualRatio?: number;
  pressureResidualDrift?: boolean;
  pressureRowCapacity?: number;
  pressureRequiredRows?: number;
  pressureCapacityOverflow?: boolean;
  frontierListCapacity?: number;
  frontierRequiredLeaves?: number;
  frontierCapacityOverflow?: boolean;
  maxComponentCfl?: number;
  highCflCellCount?: number;
  nonFiniteCount?: number;
  stabilityFlags?: string[];
  maxSpeedLocation?: GPUFieldLocation;
  maxDivergenceBeforeLocation?: GPUFieldLocation;
  maxDivergenceAfterLocation?: GPUFieldLocation;
  maxAirSpeedLocation?: GPUFieldLocation;
  maxPressureLocation?: GPUFieldLocation;
  maxPressureResidualLocation?: GPUFieldLocation;
  lastDt_s?: number;
  /** Level-set transport substeps encoded by the latest advance (1 when calm). */
  lastSubsteps?: number;
  /** Latest GPU step submitted to the device queue. */
  submittedTime_s?: number;
  /** Latest GPU step encoded by the solver. */
  simulatedTime_s?: number;
  /** Latest GPU step confirmed complete by the device queue. */
  completedTime_s?: number;
  simulationLag_s?: number;
  maximumTallCellHeight?: number;
  encodedSteps?: number;
  /** Monotonic generation of the contoured surface publication. A source may
   * advance this for paused edits without claiming another physics step. */
  surfaceRevision?: number;
  /** Presentation-sized physics batches submitted but not yet queue-confirmed. */
  gpuPendingBatches?: number;
  /** Simulation time represented by submitted, unconfirmed GPU work. */
  gpuInFlightSimulation_s?: number;
  initialVolumeCellSum?:number;
  volumeDrift?:number;
  rawVolumeDrift?:number;
  referenceLiquidVolume_cells?: number;
  /** NB scalar volume budget, independent of geometric V. The shift is the one
   * that would return the liquid to its budget: measured, never applied. */
  narrowBandTargetVolume_cells?:number;
  narrowBandOutflowVolume_cells?:number;
  narrowBandVolumeShift_cells?:number;
  narrowBandVolumeBeforeCorrection_cells?:number;
  narrowBandInitialVolume_cells?:number;
  narrowBandVolumeBudgetDrift?:number;
  /** The divergence volume control asked of the liquid at the last gauge, in 1/s. */
  narrowBandVolumeControlRate_per_s?:number;
  /** CM11a coarsest-grid convergence and finest-grid post-cycle residuals. */
  uniformCM11aResidualInfinity?: number;
  uniformCM11aConverged?: boolean;
  uniformCM11aCoarseIterations?: number;
  uniformCM11aCapFailure?: boolean;
  uniformCM11aFailingCoarseInvocation?: number;
  uniformCM11aCoarseMaxAbsRhs?: number;
  uniformCM11aCoarseMaxDiagonalPressure?: number;
  uniformCM11aCoarseMaxAbsPressure?: number;
  uniformCM11aCoarseProjectedGapPressure?: number;
  uniformCM11aCoarseNormalizedProjectedResidual?: number;
  uniformCM11aFineResidualInfinity?: number;
  uniformCM11aFineProjectedGapPressure?: number;
  uniformCM11aCoarseActiveRows?: number;
  uniformCM11aCoarseFreeRows?: number;
  uniformCM11aCoarseWorstRow?: number;
  uniformCM11aCoarseWorstRowActive?: boolean;
  uniformCM11aCoarseWorstRowHalo?: boolean;
  /**
   * CM11a host-side cycle budgeting. "fixed" encodes the configured schedule
   * every step; "lagged" encodes only as many cycles as the latest
   * asynchronous diagnostics sample says the solve needed, so the tail's
   * launch floor and CPU encode are never paid. Encoded counts are host facts
   * known at encode time; executed counts arrive with the readback.
   */
  uniformPressureCycleBudget?: "lagged" | "fixed" | "adaptive";
  uniformPressureBudgetHeadroom?: number;
  uniformPressureCyclesEncoded?: number;
  uniformPressureCyclesConfigured?: number;
  uniformPressurePassesEncoded?: number;
  uniformPressureFinishPassesEncoded?: number;
  /** Actual coarse work in the current adaptive frame, summed across invocations. */
  uniformPressureCoarseSweepsTotal?: number;
  uniformPressureCoarseSolvesExecuted?: number;
  uniformPressurePassesConfigured?: number;
  uniformPressureCyclesExecuted?: number;
  uniformPressureCyclesConverged?: boolean;
  uniformCM11aFullCyclesExecuted?: number;
  uniformCM11aVCyclesExecuted?: number;
  uniformPressureAcceptedResidual?: number;
  /** Geometric's h pressure band, from the frame receipt: tiles re-solved,
   * local V-cycles completed, final band residual. */
  uniformPressureBandTiles?: number;
  uniformPressureBandCycles?: number;
  uniformPressureBandResidual?: number;
  uniformPressureInitialResidual?: number;
  uniformPressureRejectedCycles?: number;
  uniformPressureRecoverySweeps?: number;
  uniformPressureRecoveryExhausted?: boolean;
  /** Sec. 3.3 FIM must terminate with an empty active list. */
  uniformFIMTerminalActiveFaces?: number;
  uniformFIMConverged?: boolean;
  uniformFIMExecutedPasses?: number;
  /** The 4h work map selected the Sec. 3.5 sweeps in the latest step. False is
   * the dense control, or sharpening being off; the counts are then absent. */
  uniformSharpenWorkMap?: boolean;
  uniformSharpenTilesActive?: number;
  uniformSharpenTilesTotal?: number;
  /** Volume dust floor: the live threshold, and what it discarded this step. */
  uniformVolumeDustThreshold?: number;
  uniformVolumeOrphanDustThreshold?: number;
  uniformVolumeOrphanDustCells?: number;
  uniformVolumeOrphanDustMass_cells?: number;
  uniformVolumeDustCells?: number;
  uniformVolumeDustMass_cells?: number;
  /** Experiment E1's two-level velocity sampler and its 4h fine map. */
  uniformTwoLevelVelocity?: boolean;
  uniformTwoLevelFineReach?: number;
  uniformTwoLevelFineTiles?: number;
  uniformTwoLevelTilesTotal?: number;
  /**
   * Experiment E3's live transport set: whether the twelve conservative
   * transport passes ran on it, how many 4h tiles it held, the reach this
   * step's measured maximum backward displacement required, the authored
   * margin on top of it, and the reach the dilation actually used. Used below
   * required means the shader's cap bit and a front can stall at the set
   * boundary; it is otherwise required + margin by construction.
   */
  uniformTransportWorkMap?: boolean;
  uniformTransportTiles?: number;
  uniformTransportTilesTotal?: number;
  uniformTransportReachTiles?: number;
  uniformTransportReachMargin?: number;
  uniformTransportRequiredReachTiles?: number;
  uniformTransportMaxDisplacement_cells?: number;
  /** Latest rolling uniform work box, copied only by the existing diagnostics readback. */
  uniformActiveRegionMinimum?: GPUFieldLocation;
  uniformActiveRegionMaximum?: GPUFieldLocation;
  uniformActiveRegionCellCount?: number;
  uniformActiveRegionFraction?: number;
  /**
   * How the solve window's dispatches were sized: "host" means the CPU chose
   * the group counts from a lagged box (direct launches, exact origins), and
   * "indirect" means the GPU's own records drove every launch.
   */
  uniformSolveWindowDispatch?: "host" | "indirect";
  /** Steps whose exact window outgrew the host's counts and were clipped. */
  uniformSolveWindowClippedSteps?: number;
  /** Steps the host sized at whole-domain counts (start-up, edits, penalty). */
  uniformSolveWindowDenseSteps?: number;
  /** Worst observed age, in steps, of the box the host sized a dispatch from. */
  uniformSolveWindowMaxLagSteps?: number;
  /** Window-local CM11a lattice: capacity and origin in simulation cells. */
  uniformPressureLattice?: string;
  /** False while that lattice is the whole domain (fallback or startup). */
  uniformPressureLatticeWindowed?: boolean;
  /** Instances built since load, and how long the last one took. */
  uniformPressureLatticeReplans?: number;
  uniformPressureLatticeReplanMs?: number;
  phiInterfaceCellCount?: number;
  volumeCorrectionNormalSpeed_cells_s?: number;
  /** Diagnostic divergence-rate equivalent of the normal volume correction. */
  volumeCorrectionDivergenceRate_s?: number;
  /** 1 = global controller; <1 concentrates the push on phi/VOF disagreement. */
  volumeControlAgreeWeight?: number;
  surfaceField?: "levelset";
  /** Global normal level-set volume controller. Defaults to enabled. */
  volumeControl?: boolean;
  /** Smoke-only step-response probe; scales the initial volume reference. */
  referenceVolumeScale?: number;
  quadtreeLeafCount?: number;
  quadtreePressureSampleCount?: number;
  quadtreeLiquidDofCount?: number;
  quadtreeOpticalLayerMode?: "fixed" | "adaptive-motion";
  quadtreeOpticalAlpha?: number;
  quadtreeOpticalMinimumCells?: number;
  quadtreeOpticalMaximumCells?: number;
  quadtreeFaceCount?: number;
  quadtreeMLSProjectionRowCount?: number;
  quadtreeTallSegmentCount?: number;
  quadtreeGhostFaceCount?: number;
  quadtreeMaximumFluidScale?: number;
  quadtreeMaximumNeighborRatio?: number;
  quadtreeLevelSetMismatchFraction?: number;
  quadtreeCulledDebrisCells?: number;
  quadtreeVofReconciliationActive?: boolean;
  quadtreeVelocityClampCount?: number;
  quadtreeTopologyReused?: boolean;
  quadtreeTopologyReuseCount?: number;
  quadtreePressureIterationsUsed?: number;
  /**
   * Outer CG iterations the last encoded solve actually scheduled.
   *
   * This is the solve tail as encoded — `planOctreeSolveTail`'s envelope at
   * construction, and the executor's own budget once one is attached — not a
   * request or a paper constant. A consumer that needs to know the shape of
   * the encoded solve (the dependency-cone diagnostic grows its stages from
   * it) reads it here rather than re-deriving it from scene facts it would
   * have to guess at, which is a fact about the solver that only the solver
   * holds.
   */
  quadtreePressureIterationBudget?: number;
  quadtreePressureIterationHardBudget?: number;
  quadtreePressureConverged?: boolean;
  /** Step-coherent graph/phi/velocity/pressure receipt summary for a rejected
   * Losasso update. Undefined again after a fully admitted tuple. */
  quadtreePressureRejectionSummary?: string;
  quadtreeFactorLevelCount?: number;
  quadtreeMultigridLevelCount?: number;
  quadtreeMultigridCoarsestDofs?: number;
  quadtreeRebuildCadenceSteps?: number;
  quadtreeTopologyStaleLimit?: number;
  quadtreeTopologyStaleSteps?: number;
  /** Bytes transferred from GPU for the latest adaptive topology update. */
  quadtreeTopologyReadbackBytes?: number;
  /** Whether an adaptive topology readback/rebuild is currently in flight. */
  quadtreeRebuildPending?: boolean;
  /** True when topology construction is encoded in each fluid advance rather than rebuilt by the asynchronous host path. */
  quadtreeInlineRebuild?: boolean;
  /** Render frames whose physics advance was blocked by the latest rebuild. */
  quadtreeRebuildBlockedFrames?: number;
  quadtreeRebuildCompletedCount?: number;
  /** Fluid authority remains in resident GPU resources between submissions. */
  hostFluidAuthority?: "gpu-resident";
  /** Simulation-sized host work performed by one authoritative fluid frame. */
  hostSimulationSizedWorkItems?: number;
  /** Must remain false for authoritative octree scheduling. */
  hostSchedulingUsesReadback?: boolean;
  /**
   * Static per-advance pass structure published by the uniform reference
   * solver once its programs are built, so the fluid pipeline panel's chips
   * state the counts the instance actually encodes rather than re-deriving
   * them from grid maths that could drift from the plan.
   */
  uniformPipelineFacts?: {
    readonly extrapolationFrontSweeps: number;
    readonly extrapolationHierarchyLevels: number;
    readonly extrapolationPassesPerInvocation: number;
    readonly multigridLevels: number;
    readonly multigridPasses: Readonly<Record<"setup" | "full-cycle" | "v-cycle" | "finish", number>>;
    readonly multigridPassesTotal: number;
    readonly pressureSchedule: Readonly<{
      fullCycles: number;
      vCycles: number;
      preSweeps: number;
      postSweeps: number;
    }>;
  };
  /** Latest exhaustive, exclusive GPU physics partition. */
  physicsTrace?: PerformanceTrace;
  spatialPerformance?: SpatialPerformanceReport;
  spatialPerformanceStages?: readonly { id: string; label: string }[];
  spatialPerformanceError?: string;
  physicsTraceUnavailable?: string;
  /** Main-thread command-encoding partition captured under the exact same
   * sample ID and context as `physicsTrace`. It deliberately excludes the
   * controller/render ticks which merely happen to report the completed GPU
   * sample later. */
  physicsCPUTrace?: PerformanceTrace;
  /** Stable join identity published when the sampled advance finishes
   * encoding, before asynchronous GPU trace readback completes. */
  physicsCaptureIdentity?: {
    sampleId: number;
    context: string;
    frameId: string;
  };
}

export interface GPURigidLoad {
  bodyId: string;
  impulse_N_s: { x: number; y: number; z: number };
  angularImpulse_N_m_s: { x: number; y: number; z: number };
  couplingInterval_s: number;
  displacedVolume_m3: number;
  meanFluidVelocity_m_s: { x: number; y: number; z: number };
}

export const GPU_RIGID_BODY_LIMIT = 12;
export const GPU_RIGID_EXCHANGE_WORDS = 12;
export const GPU_RIGID_EXCHANGE_BYTES = GPU_RIGID_BODY_LIMIT * GPU_RIGID_EXCHANGE_WORDS * Int32Array.BYTES_PER_ELEMENT;

/** Decode one fixed-point rigid-exchange record. Snapshot fields are averaged
 * when a solver encoded more than one fluid substep; impulse fields remain the
 * sum over the whole coupling interval. */
export function decodeGPURigidLoad(bodyId: string, words: Int32Array, index: number, couplingInterval_s: number, cellVolume_m3: number, snapshotCount = 1): GPURigidLoad {
  const base = index * GPU_RIGID_EXCHANGE_WORDS, snapshots = Math.max(1, snapshotCount);
  const wetCellWeight = words[base + 6] / 65536 / snapshots;
  const weightedVelocity = {
    x: words[base + 7] / 1e4 / snapshots,
    y: words[base + 8] / 1e4 / snapshots,
    z: words[base + 9] / 1e4 / snapshots
  };
  const meanFluidVelocity_m_s = wetCellWeight > 0 ? {
    x: weightedVelocity.x / wetCellWeight,
    y: weightedVelocity.y / wetCellWeight,
    z: weightedVelocity.z / wetCellWeight
  } : { x: 0, y: 0, z: 0 };
  return {
    bodyId,
    impulse_N_s: { x: words[base] / 1e6, y: words[base + 1] / 1e6, z: words[base + 2] / 1e6 },
    angularImpulse_N_m_s: { x: words[base + 3] / 1e6, y: words[base + 4] / 1e6, z: words[base + 5] / 1e6 },
    couplingInterval_s,
    displacedVolume_m3: wetCellWeight * cellVolume_m3,
    meanFluidVelocity_m_s
  };
}

const addLoadVector = (a: GPURigidLoad["impulse_N_s"], b: GPURigidLoad["impulse_N_s"]) => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });

export function mergeGPURigidLoads(current: GPURigidLoad[], incoming: GPURigidLoad[]): GPURigidLoad[] {
  const pending = new Map(current.map((load) => [load.bodyId, load]));
  for (const load of incoming) {
    const previous = pending.get(load.bodyId);
    pending.set(load.bodyId, previous ? {
      ...load,
      impulse_N_s: addLoadVector(previous.impulse_N_s, load.impulse_N_s),
      angularImpulse_N_m_s: addLoadVector(previous.angularImpulse_N_m_s, load.angularImpulse_N_m_s),
      couplingInterval_s: previous.couplingInterval_s + load.couplingInterval_s
    } : load);
  }
  return [...pending.values()];
}

export function consumeGPURigidLoad(load: GPURigidLoad, dt: number) {
  const deliveryTime = Math.max(load.couplingInterval_s, dt), fraction = Math.min(1, dt / deliveryTime);
  const impulse_N_s = { x: load.impulse_N_s.x * fraction, y: load.impulse_N_s.y * fraction, z: load.impulse_N_s.z * fraction };
  const angularImpulse_N_m_s = { x: load.angularImpulse_N_m_s.x * fraction, y: load.angularImpulse_N_m_s.y * fraction, z: load.angularImpulse_N_m_s.z * fraction };
  load.impulse_N_s = { x: load.impulse_N_s.x - impulse_N_s.x, y: load.impulse_N_s.y - impulse_N_s.y, z: load.impulse_N_s.z - impulse_N_s.z };
  load.angularImpulse_N_m_s = { x: load.angularImpulse_N_m_s.x - angularImpulse_N_m_s.x, y: load.angularImpulse_N_m_s.y - angularImpulse_N_m_s.y, z: load.angularImpulse_N_m_s.z - angularImpulse_N_m_s.z };
  load.couplingInterval_s = Math.max(0, load.couplingInterval_s - dt);
  return { impulse_N_s, angularImpulse_N_m_s };
}
