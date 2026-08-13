import type { SceneDescription } from "./model";
import type { PerformanceTrace } from "./performance-trace";
import type { GPUQuality, TallCellLayout } from "./tall-cell-grid";

export type { GPUQuality } from "./tall-cell-grid";
export type GPUGridMethod = "octree";
export type GPUVelocityTransport = "semi-lagrangian" | "maccormack";

export interface GPUFieldLocation {
  x: number;
  y: number;
  z: number;
}

export interface GPUEulerianInfo {
  nx: number;
  ny: number;
  nz: number;
  storedNy: number;
  cellCount: number;
  equivalentUniformCells: number;
  compressionRatio: number;
  activeCompressionRatio?: number;
  activeSampleCount?: number;
  regularLayers: number;
  maximumNeighborDelta: number;
  gridKind: "restricted-tall-cell" | "quadtree-tall-cell" | "octree" | "uniform";
  /** True only after the complete sparse t=0 authority passed its queue fence. */
  initialSparseAuthorityReady?: boolean;
  /** Renderer-owned gate: a warmed octree is not transport-ready until its first t=0 raster publication is fenced. */
  initialRasterSurfaceReady?: boolean;
  /** Honest distinction between GPU-only authority, a readback-confirmed crossing, and fail-closed startup. */
  initialRasterSurfaceState?: "pending" | "gpu-authoritative" | "compact-confirmed" | "crossing-confirmed" | "failed-closed";
  initialRasterSurfaceDiagnostic?: string;
  cellSize_m: number;
  pressureIterations: number;
  pressureSolver?: string;
  /** Construction-time octree coarse-dynamics selection. */
  coarseDynamicsBackend?: "losasso" | "power2017";
  allocatedBytes: number;
  /** Fixed GPU ring capacity for one-way escaped spray droplets. */
  secondaryParticleCapacity?: number;
  /** Logical sparse-fluid page table capacity and latest GPU-owned lifecycle counts. */
  fluidBrickCapacity?: number;
  fluidBrickResidentCount?: number;
  fluidBrickCoreCount?: number;
  fluidBrickHaloCount?: number;
  fluidBrickActivatedCount?: number;
  fluidBrickRetiredCount?: number;
  fluidBrickGeneration?: number;
  /** Wet-domain storage residency, independent from the narrow surface band. */
  fluidBulkBrickResidentCount?: number;
  fluidBulkBrickHaloCount?: number;
  fluidBulkBrickActivatedCount?: number;
  fluidBulkBrickRetiredCount?: number;
  powerDiagramReady?: boolean;
  powerDiagramAuthoritative?: boolean;
  /** Host-known generation stamped into the live GPU power topology/face publication. */
  powerDiagramGeneration?: number;
  powerDiagramAllocatedBytes?: number;
  globalFineLevelSetAllocatedBytes?: number;
  globalFineLevelSetResidentBrickCapacity?: number;
  globalFineLevelSetLogicalBrickCount?: number;
  /** Global, uniformly indexed sparse fine narrow-band level set. */
  globalFineLevelSetEnabled?: boolean;
  globalFineLevelSetFactor?: 1 | 4 | 8;
  /** Static sparse fine-transport schedule; these host-known values require no readback. */
  globalFineTransportQueryCapacity?: number;
  globalFineTransportChunkCapacity?: number;
  globalFineTransportChunkCount?: number;
  globalFineTransportSegmentCount?: number;
  globalFineTransportEncodedPasses?: number;
  globalFineTransportPrepassScratchBytes?: number;
  globalFineTransportVertexScratchBytes?: number;
  /** QA-only global-fine handoff/publication counters. */
  globalFineSeedCount?: number;
  globalFineSeedError?: number;
  globalFineTopologyFlags?: number;
  /** Bit mask: 1 topology, 2 redistance, 4 volume, 8 transport. */
  globalFineDownstreamFinalizeReason?: number;
  globalFineRedistanceUnresolvedCells?: number;
  globalFineRedistanceSeeds?: number;
  globalFineRedistanceCommitted?: boolean;
  globalFineVolumeFlags?: number;
  globalFineTransportDepartureOutsideBand?: number;
  globalFineTransportNonfiniteVelocity?: number;
  globalFineTransportCommitted?: boolean;
  globalFineTransportStructuredAuthorityUnavailable?: number;
  globalFineTransportVelocityUnavailable?: number;
  /** Invalid Stage-B velocity statuses observed while tracing the fine band. */
  globalFineTransportInvalidVelocityStatus?: number;
  /** Velocity samples that returned a non-positive validity weight. */
  globalFineTransportNonpositiveVelocityResult?: number;
  /** Bitwise union of invalid Stage-B velocity status reasons. */
  globalFineTransportVelocityStatusReasonOr?: number;
  /** Exact status and chunk-local sample index for the first invalid velocity. */
  globalFineTransportFirstInvalidVelocityStatus?: number;
  globalFineTransportFirstInvalidVelocityLocalIndex?: number;
  /** Exact solver-local position in metres at which that status was observed.
   * Solver-local x/z begin at the negative-world container walls. */
  globalFineTransportFirstInvalidVelocityPosition_m?: GPUFieldLocation;
  /** Accepted structured velocity publication sampled by fine transport. */
  structuredVelocityGeneration?: number;
  structuredVelocityRows?: number;
  structuredVelocitySlots?: number;
  structuredVelocityValid?: boolean;
  /** Exact terminal Section-5 sparse-air publication workload. These are
   * successful GPU-owned counts, not capacities or host estimates. */
  structuredAirSupportRows?: number;
  structuredAirSupportCells?: number;
  structuredAirSupportCapacity?: number;
  structuredAirSupportFaceItems?: number;
  structuredAirSupportSeedFaces?: number;
  structuredAirSupportMarchDepth?: number;
  /** First live or latched Section-5 air-support publication rejection. */
  structuredAirSupportFailureFlags?: number;
  structuredAirSupportFailureItem?: number;
  /** Decoded structured-dynamics reject carry. A rejection zeroes every class
   * dispatch, so the step silently freezes; these name the responsible stage
   * instead of leaving it to surface as a step-count or volume-drift failure. */
  structuredRejectStage?: number;
  structuredRejectIndex?: number;
  structuredRejectSummary?: string;
  structuredBoundaryGeneration?: number;
  structuredBoundaryValid?: boolean;
  /** Exact same-generation face-weighted kinetic-energy stages across the
   * substep: start-of-step (post-remap), post-advection, post-force
   * (pre-projection), post-projection. */
  structuredStartKineticEnergyProxy?: number;
  structuredPostAdvectionKineticEnergyProxy?: number;
  structuredPreProjectionKineticEnergyProxy?: number;
  structuredPostProjectionKineticEnergyProxy?: number;
  /** Sampler-path census over wet faces (transition / staggered / cell). */
  structuredWetFaceCount?: number;
  structuredWetStartThetaEnergyProxy?: number;
  structuredWetPostAdvectionThetaEnergyProxy?: number;
  structuredWetPreProjectionThetaEnergyProxy?: number;
  structuredWetPostProjectionThetaEnergyProxy?: number;
  structuredStaggeredPathCount?: number;
  /** Same stages restricted to faces with a liquid incident row. */
  structuredWetStartKineticEnergyProxy?: number;
  structuredWetPostAdvectionKineticEnergyProxy?: number;
  structuredWetPreProjectionKineticEnergyProxy?: number;
  structuredWetPostProjectionKineticEnergyProxy?: number;
  structuredProjectionEnergyRatio?: number;
  structuredProjectionEnergySampleCount?: number;
  /** Power-coarse φ authority failure bits and first compact row. Bit 512
   * identifies a missing causal non-obtuse Delaunay simplex. */
  globalFineCoarseLevelSetFlags?: number;
  globalFineCoarseLevelSetFirstErrorRow?: number;
  globalFineInterfaceBricks?: number;
  globalFineDesiredBricks?: number;
  globalFineActivatedBricks?: number;
  globalFinePublished?: boolean;
  globalFineRolledBack?: boolean;
  globalFineActiveBricks?: number;
  globalFineGeneration?: number;
  /** First validation error captured by the solver's diagnostic error scope.
   * Reporting is asynchronous and never feeds simulation state. */
  gpuValidationError?: string;
  quality: GPUQuality;
  volumeCellSum?: number;
  representedVolumeCellSum?: number;
  representedVolumeDrift?: number;
  /** GPU field which supplied the displayed physical volume. */
  volumeTelemetrySource?: "global-fine" | "adaptive-conservative-mass" | "adaptive-pages"
    | "dense-volume" | "initial-condition" | "unavailable";
  front_m?: number;
  /** GPU field which supplied the displayed dam-front location. */
  frontTelemetrySource?: "dense-volume" | "initial-condition" | "unavailable";
  maxSpeed_m_s?: number;
  maxDivergence_s?: number;
  maxDivergenceBefore_s?: number;
  maxDivergenceAfter_s?: number;
  projectionDivergenceRatio?: number;
  maxAirSpeed_m_s?: number;
  /** Physical pressure in Pa for dense/tall-cell methods. The octree power
   * projection stores dt·p/rho (m²/s); consumers must branch on gridKind. */
  maxPressure_Pa?: number;
  pressureResidual?: number;
  pressureRelativeResidual?: number;
  pressureRowCapacity?: number;
  pressureRequiredRows?: number;
  pressureCapacityOverflow?: boolean;
  frontierListCapacity?: number;
  frontierRequiredLeaves?: number;
  frontierCapacityOverflow?: boolean;
  maxComponentCfl?: number;
  /** Faces processed by the latest compact octree velocity transport pass. */
  adaptiveFaceTransportedCount?: number;
  highCflCellCount?: number;
  nonFiniteCount?: number;
  stabilityFlags?: string[];
  /** Nonempty when the last advance's encoded stage order deviated from the
   * declared physics step program (lib/physics-step-program.ts). */
  stepSequenceDeviations?: string[];
  /** Step index of the latest step-coherent structured snapshot consumed by
   * diagnostics, and its exact whole-step authority lag (0 = current). */
  structuredSnapshotStep?: number;
  structuredAuthorityLagSteps?: number;
  /** Fine active-brick count from the same record: worklist header word ONE.
   * Word zero is the generation; the overflow flag is the 0xFFFFFFFF sentinel
   * that silently no-ops the solver (P0.4 / docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md
   * A3, A4.3). */
  structuredSnapshotActiveFineBricks?: number;
  structuredSnapshotFineBandOverflow?: boolean;
  /** MGPCG counters from the same record: outer iterations the GPU executed
   * and whether it converged inside the encoded budget. */
  structuredSnapshotExecutedSolveIterations?: number;
  structuredSnapshotSolveConverged?: boolean;
  /** Nonempty when a step's encode predicted zero work for a stage it deleted
   * and the GPU's own counters for that step disagreed (P0.5 lag-k check). */
  stepPredictionFailures?: string[];
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
  /** Presentation-sized physics batches submitted but not yet queue-confirmed. */
  gpuPendingBatches?: number;
  /** Simulation time represented by submitted, unconfirmed GPU work. */
  gpuInFlightSimulation_s?: number;
  initialVolumeCellSum?:number;
  volumeDrift?:number;
  rawVolumeDrift?:number;
  referenceLiquidVolume_cells?: number;
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
  /** Sec. 3.3 FIM must terminate with an empty active list. */
  uniformFIMTerminalActiveFaces?: number;
  uniformFIMConverged?: boolean;
  uniformFIMExecutedPasses?: number;
  /** Latest rolling uniform work box, copied only by the existing diagnostics readback. */
  uniformActiveRegionMinimum?: GPUFieldLocation;
  uniformActiveRegionMaximum?: GPUFieldLocation;
  uniformActiveRegionCellCount?: number;
  uniformActiveRegionFraction?: number;
  /** Chentanez--Mueller mass stored above rho=1 after the latest transport. */
  adaptiveCompressedExcessVolume_cells?: number;
  /** Conserved mass currently below the rho=.5 visible-surface threshold. */
  adaptiveSubIsoVolume_cells?: number;
  adaptiveOverfullLeafCount?: number;
  adaptiveSubIsoLeafCount?: number;
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
  hostFluidAuthority?: "gpu-resident" | "cpu-reference";
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

export interface WebGPUEulerianSolverOptions {
  velocityTransport?: GPUVelocityTransport;
  /** Layout overrides applied on top of the quality preset. */
  tallCellSettings?: Partial<import("./tall-cell-grid").TallCellSettings>;
  /** Multigrid refinement V-cycles after the initial full cycle. */
  pressureCycles?: number;
  /** Reuse the previous frame's pressure as the fine-grid initial guess. */
  pressureWarmStart?: boolean;
  /** Rebuild divergence after projection and apply a second pressure solve. */
  pressureDefectCorrection?: boolean;
  /** Mass-Conserving Eulerian Liquid Simulation Sec 3.5 density sharpening
   * after conservative advection. Defaults to on. */
  densitySharpening?: boolean;
  /** Restricted tall cells now use the paper's signed-distance surface.
   * Retained as an informational compatibility parameter. */
  surfaceField?: "levelset";
  /** Apply the narrow-band global normal level-set volume controller on the
   * restricted tall-cell level set. Defaults to on. */
  volumeControl?: boolean;
  /** Smoke-only step-response probe; scales the initial volume reference. */
  referenceVolumeScale?: number;
  /** Tall-cell paper Sec 3.3.1 hierarchical velocity extrapolation beyond the
   * two-cell narrow band. Defaults to on; off reverts to the legacy repeated
   * neighbor passes (a documented diagnostic departure). */
  hierarchicalExtrapolation?: boolean;
  /** Internal browser path: allocate resources now, compile pipelines with the
   * asynchronous WebGPU API before exposing the solver. */
  deferPipelineCompilation?: boolean;
  /** Test-only layout injection for the one-tall-cell differential probe. */
  layoutOverride?: TallCellLayout;
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

export const uniformTargetCells: Record<GPUQuality, number> = { balanced: 110_000, high: 500_000, ultra: 1_200_000 };
