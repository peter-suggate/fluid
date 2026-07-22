import type { SceneDescription } from "./model";
import { damBreakFractions } from "./initial-fluid";
import { initializeRigidBodies, type RigidBodyState } from "./rigid-body";
import { tallCellComputeShader } from "./tall-cell-kernels";
import { createTallCellLayout, type GPUQuality, type TallCellLayout } from "./tall-cell-grid";
import { TallCellMultigrid } from "./tall-cell-multigrid";
import { TallCellVelocityHierarchy } from "./tall-cell-extrapolation";
import { classifyTallCellStability, planGPUAdvance } from "./tall-cell-diagnostics";
import { averageInflowStrength, createInflowGridBoundary, inflowBoundaryWGSL, type InflowGridBoundary } from "./inflow-boundary";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import { WebGPURigidBodySystem } from "./webgpu-rigid-body";
import { encodeGPUStageTextureCapture, type PendingGPUStageCapture } from "./gpu-stage-capture";

export type { GPUQuality } from "./tall-cell-grid";
export type GPUGridMethod = "tall-cell" | "quadtree-tall-cell" | "octree" | "uniform";
export type GPUVelocityTransport = "semi-lagrangian" | "maccormack";

export type GPUPhysicsStageId =
  | "preparation"
  | "topology"
  | "advection"
  | "conditioning"
  | "remeshing"
  | "pressure"
  | "powerAssembly"
  | "pressureSolve"
  | "projection"
  | "powerProjection"
  | "velocityProjection"
  | "faceBand"
  | "faceMarch"
  | "powerPublication"
  | "extrapolation"
  | "materialization"
  | "surfaceUpdate"
  | "fineTopology"
  | "fineTransport"
  | "fineRedistance"
  | "rigidCoupling"
  | "spray"
  | "fluidResidency"
  | "sparsePublication"
  | "diagnostics";

export interface GPUPhysicsTimings {
  preparation_ms: number;
  layerConstruction_ms: number;
  advection_ms: number;
  conditioning_ms: number;
  remeshing_ms: number;
  pressure_ms: number;
  /** Diagnostic subdivision of pressure_ms for power-row construction and operator assembly. */
  powerAssembly_ms: number;
  /** Diagnostic subdivision of pressure_ms for the pressure iteration/V-cycle work. */
  pressureSolve_ms: number;
  projection_ms: number;
  /** Diagnostic subdivision of projection_ms for generalized-face projection and publication. */
  powerProjection_ms: number;
  /** Diagnostic subdivision of projection_ms for the regular finite-volume compatibility field. */
  velocityProjection_ms: number;
  /** Section 5 face-band construction, including the generalized-face setup preceding it. */
  faceBand_ms: number;
  /** Section 5 regular-face narrow-band fast march. */
  faceMarch_ms: number;
  /** Section 5 regular-face-to-power-face publication and validation tail. */
  powerPublication_ms: number;
  extrapolation_ms: number;
  materialization_ms: number;
  surfaceUpdate_ms: number;
  /** Section 5 fine-page topology range, including surface setup assigned to its shared interval. */
  fineTopology_ms: number;
  /** Section 5 fine-level-set transport range. */
  fineTransport_ms: number;
  /** Section 5 fine redistance and transactional publication tail. */
  fineRedistance_ms: number;
  rigidCoupling_ms: number;
  spray_ms: number;
  fluidResidency_ms: number;
  sparsePublication_ms: number;
  diagnostics_ms: number;
  overhead_ms: number;
  total_ms: number;
  /** Stages actually encoded in the sampled advance; zero-duration entries can therefore be shown as idle. */
  activeStages: GPUPhysicsStageId[];
}

export type GPUPhysicsTimingField = Exclude<keyof GPUPhysicsTimings, "activeStages">;
export const GPU_PHYSICS_TIMESTAMP_CAPACITY = 2048;

export interface GPUPhysicsTimestampSegment {
  name: GPUPhysicsTimingField;
  start: number;
  end: number;
  /**
   * Complete temporally ordered boundary chain shared by a partitioned
   * aggregate. A child is publishable only when every boundary in its parent
   * partition resolved and the whole chain is monotonic.
   */
  requiredBoundaries?: readonly number[];
}

export function emptyGPUPhysicsTimings(activeStages: GPUPhysicsStageId[] = []): GPUPhysicsTimings {
  return {
    preparation_ms: 0,
    layerConstruction_ms: 0,
    advection_ms: 0,
    conditioning_ms: 0,
    remeshing_ms: 0,
    pressure_ms: 0,
    powerAssembly_ms: 0,
    pressureSolve_ms: 0,
    projection_ms: 0,
    powerProjection_ms: 0,
    velocityProjection_ms: 0,
    faceBand_ms: 0,
    faceMarch_ms: 0,
    powerPublication_ms: 0,
    extrapolation_ms: 0,
    materialization_ms: 0,
    surfaceUpdate_ms: 0,
    fineTopology_ms: 0,
    fineTransport_ms: 0,
    fineRedistance_ms: 0,
    rigidCoupling_ms: 0,
    spray_ms: 0,
    fluidResidency_ms: 0,
    sparsePublication_ms: 0,
    diagnostics_ms: 0,
    overhead_ms: 0,
    total_ms: 0,
    activeStages
  };
}

/**
 * Decode timestamp ranges defensively. WebGPU query resolve leaves an
 * unwritten query at zero on the backends we use; subtracting a valid shared
 * boundary from that zero produces a finite but impossible multi-hour stage.
 * Treat an entire field as unavailable when any of its encoded ranges is
 * unresolved, out of bounds, or temporally out of order.
 */
export function decodeGPUPhysicsTimestampSegments(
  timestamps: ArrayLike<bigint>,
  segments: readonly GPUPhysicsTimestampSegment[],
): Partial<Record<GPUPhysicsTimingField, number>> {
  const decoded: Partial<Record<GPUPhysicsTimingField, number>> = {};
  const invalid = new Set<GPUPhysicsTimingField>();
  for (const segment of segments) {
    const boundaries = segment.requiredBoundaries ?? [segment.start, segment.end];
    let previous: bigint | undefined;
    let valid = boundaries.length >= 2
      && boundaries.includes(segment.start)
      && boundaries.includes(segment.end)
      && boundaries.indexOf(segment.start) <= boundaries.indexOf(segment.end);
    for (const index of boundaries) {
      if (!valid || !Number.isInteger(index) || index < 0 || index >= timestamps.length) {
        valid = false;
        break;
      }
      const timestamp = timestamps[index];
      if (timestamp === undefined || timestamp === 0n
        || (previous !== undefined && timestamp < previous)) {
        valid = false;
        break;
      }
      previous = timestamp;
    }
    if (!valid) {
      invalid.add(segment.name);
      continue;
    }
    const elapsed_ms = Number(timestamps[segment.end] - timestamps[segment.start]) / 1e6;
    if (!Number.isFinite(elapsed_ms) || elapsed_ms < 0) {
      invalid.add(segment.name);
      continue;
    }
    decoded[segment.name] = (decoded[segment.name] ?? 0) + elapsed_ms;
  }
  for (const name of invalid) decoded[name] = 0;
  return decoded;
}

export function categorizedGPUPhysicsTime_ms(timings: GPUPhysicsTimings) {
  return timings.preparation_ms + timings.layerConstruction_ms + timings.advection_ms
    + timings.conditioning_ms + timings.remeshing_ms + timings.pressure_ms
    + timings.projection_ms + timings.extrapolation_ms + timings.materialization_ms
    + timings.surfaceUpdate_ms + timings.rigidCoupling_ms + timings.spray_ms
    + timings.fluidResidency_ms + timings.sparsePublication_ms + timings.diagnostics_ms;
}

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
  initialRasterSurfaceState?: "pending" | "gpu-authoritative" | "crossing-confirmed" | "failed-closed";
  initialRasterSurfaceDiagnostic?: string;
  cellSize_m: number;
  pressureIterations: number;
  pressureSolver?: string;
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
  /** Brick-pooled atlas tiles mirrored from the dense fields plus their validation error. */
  fluidBrickAtlasCapacity?: number;
  fluidBrickAtlasResidentTiles?: number;
  fluidBrickAtlasOverflow?: number;
  fluidBrickAtlasMaxPhiError?: number;
  fluidBrickAtlasMaxVelocityError?: number;
  fluidBrickAtlasMaxPhiErrorManual?: number;
  fluidBrickAtlasMaxVelocityErrorManual?: number;
  sparseSurfaceLogicalPages?: number;
  sparseSurfacePageCapacity?: number;
  adaptiveSurfacePageCapacity?: number;
  adaptiveSurfaceActivePages?: number;
  adaptiveSurfaceCandidatePages?: number;
  adaptiveSurfaceAdapterCandidateRows?: number;
  adaptiveSurfaceAdapterDispatchX?: number;
  adaptiveSurfaceOverflow?: boolean;
  adaptiveSurfaceOverflowCode?: number;
  adaptiveSurfaceDepartureFallbacks?: number;
  adaptiveSurfaceFinestResidentPages?: number;
  adaptiveSurfaceCoarseResidentPages?: number;
  adaptiveSurfaceMaximumResidentLeafSize?: number;
  /** Requested and effective GPU power-diagram projection state. */
  powerDiagramProjection?: "off" | "mirror" | "authoritative";
  powerDiagramReady?: boolean;
  powerDiagramAuthoritative?: boolean;
  /** Host-known generation stamped into the live GPU power topology/face publication. */
  powerDiagramGeneration?: number;
  powerDiagramFallbackReason?: string;
  powerDiagramAllocatedBytes?: number;
  globalFineLevelSetAllocatedBytes?: number;
  globalFineLevelSetResidentBrickCapacity?: number;
  globalFineLevelSetLogicalBrickCount?: number;
  /** Global, uniformly indexed sparse fine narrow-band level set. */
  globalFineLevelSetEnabled?: boolean;
  globalFineLevelSetFactor?: 4 | 8;
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
  globalFineTransportFaceBandUnavailable?: number;
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
  globalFineFaceBandFlags?: number;
  globalFineFaceBandTransitionFlags?: number;
  globalFineFaceBandPowerPublicationFlags?: number;
  globalFineFaceBandTransientPowerFlags?: number;
  globalFineFaceBandPointFieldFlags?: number;
  /** Power-coarse φ authority failure bits and first compact row. Bit 512
   * identifies a missing causal non-obtuse Delaunay simplex. */
  globalFineCoarseLevelSetFlags?: number;
  globalFineCoarseLevelSetFirstErrorRow?: number;
  /** Bounded, observational Section 5 transaction details. These values come
   * from the control headers already read for authority validation; exposing
   * them does not add a simulation-sized readback. */
  globalFineFaceBandFirstError?: number;
  globalFineFaceBandRowCount?: number;
  globalFineFaceBandFaceCount?: number;
  globalFineFaceBandIncidenceCount?: number;
  globalFineFaceBandSeedCount?: number;
  globalFineFaceBandAcceptedCount?: number;
  globalFineFaceBandUnresolvedCount?: number;
  globalFineFaceBandSampleFailures?: number;
  globalFineFaceBandCoarsePhiFallbacks?: number;
  globalFineFaceBandCoarsePhiFailures?: number;
  globalFineFaceBandPhiExtensions?: number;
  globalFineFaceBandMarchHeapHighWater?: number;
  globalFineFaceBandMarchPops?: number;
  globalFineFaceBandMarchTrials?: number;
  globalFineFaceBandMarchChunks?: number;
  globalFineFaceBandMarchChunkBound?: number;
  globalFineFaceBandMarchCapExhausted?: number;
  globalFineFaceBandMarchUnresolvedWithPredecessor?: number;
  globalFineFaceBandMarchDisconnected?: number;
  globalFineFaceBandTransitionFirstError?: number;
  globalFineFaceBandTransitionRowCount?: number;
  globalFineFaceBandTransitionRows?: number;
  globalFineFaceBandTransitionAdjacencyCount?: number;
  globalFineFaceBandTransitionCoreRows?: number;
  globalFineFaceBandTransitionSupport1Rows?: number;
  globalFineFaceBandTransitionSupport2Rows?: number;
  globalFineFaceBandTransitionSupport3Rows?: number;
  globalFineFaceBandTransitionEndpointRows?: number;
  globalFineFaceBandBoundaryGhostRequests?: number;
  /** Existing bounded transition-failure payload, decoded when an excluded
   * same/coarser mask escapes acute-simplex grading into the dry band. */
  globalFineFaceBandAcuteGradingFailure?: {
    readonly band: number;
    readonly rowCell: number;
    readonly rowSize: number;
    readonly descriptor: number;
    readonly coarseMask: number;
  };
  globalFineFaceBandPhiFailureCounts?: {
    readonly missingRow: number;
    readonly exactCoarseMiss: number;
    readonly invalidMetric: number;
    readonly invalidSelector: number;
  };
  globalFineFaceBandPhiFailure?: {
    readonly cause: number;
    readonly faceIndex: number;
    readonly globalFace: number;
    readonly negativeRow: number;
    readonly positiveRow: number;
    readonly anchorRow: number;
    readonly centroid: readonly [number, number, number];
    readonly interpolantPath: number;
    readonly missingOrigin: readonly [number, number, number];
    readonly missingSize: number;
    readonly selectorOrCorner: number;
    readonly detail: number;
  };
  globalFineFaceBandTransientPowerFirstError?: number;
  globalFineFaceBandTransientPowerRows?: number;
  globalFineFaceBandTransientPowerEmitted?: number;
  globalFineFaceBandTransientPowerSampled?: number;
  globalFineFaceBandTransientPowerValidated?: number;
  globalFineFaceBandPointFieldFirstError?: number;
  globalFineFaceBandPointFieldRows?: number;
  globalFineFaceBandPointFieldSolved?: number;
  globalFineFaceBandPointFieldWallContributions?: number;
  globalFineFaceBandPowerPublicationFirstError?: number;
  globalFineFaceBandPowerPublicationFaces?: number;
  globalFineFaceBandPowerPublicationTargets?: number;
  globalFineFaceBandPowerPublicationInterpolated?: number;
  globalFineFaceBandPowerPublicationCommitted?: number;
  /** Exact live Section 5 transaction identities and validity, decoded from
   * the existing bounded diagnostics readback. These never steer authority. */
  globalFineFaceBandGeneration?: number;
  globalFineFaceBandValid?: boolean;
  globalFineFaceBandTransitionValid?: boolean;
  globalFineFaceBandPointFieldValid?: boolean;
  globalFineFaceBandTransientPowerValid?: boolean;
  globalFineFaceBandPowerPublicationValid?: boolean;
  globalFineFaceBandPowerFineGeneration?: number;
  globalFineFaceBandPowerGeneration?: number;
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
  pagedPhiDifferentialSamples?: number;
  pagedPhiDifferentialComparedSamples?: number;
  pagedPhiDifferentialMaxAbs?: number;
  pagedPhiDifferentialMeanAbs?: number;
  pagedPhiDifferentialSignMismatches?: number;
  pagedPhiDifferentialHashMisses?: number;
  pagedPhiDifferentialAffineFallbacks?: number;
  pagedPhiDifferentialMaxCell?: readonly [number, number, number];
  pagedPhiDifferentialMaxDensePhi?: number;
  pagedPhiDifferentialMaxPagedPhi?: number;
  sparseSurfaceResidentPages?: number;
  sparseSurfaceCorePages?: number;
  sparseSurfaceHaloPages?: number;
  sparseSurfaceActivatedPages?: number;
  sparseSurfaceRetiredPages?: number;
  sparseSurfaceOverflow?: number;
  sparseSurfacePeakPages?: number;
  quality: GPUQuality;
  volumeCellSum?: number;
  representedVolumeCellSum?: number;
  representedVolumeDrift?: number;
  /** GPU field which supplied the displayed physical volume. */
  volumeTelemetrySource?: "global-fine" | "adaptive-pages" | "dense-volume" | "initial-condition" | "unavailable";
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
  pressureEntryCapacity?: number;
  pressureRequiredRows?: number;
  pressureRequiredEntries?: number;
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
  gpuStep_ms?: number;
  /** Monotonic identity of the latest decoded physics timestamp query set. */
  gpuPhysicsTimingSampleId?: number;
  /** Submitted simulation time represented by the latest timestamp sample. */
  gpuPhysicsTimingSimulation_s?: number;
  /** Host wall latency from the timestamp copy submission until map completion. */
  gpuPhysicsTimingReadbackWall_ms?: number;
  /** Full host wall duration of the latest periodic diagnostics/readback fan-out. */
  gpuTelemetryWall_ms?: number;
  gpuQueueWall_ms?: number;
  gpuQueueSimulation_s?: number;
  /** Presentation-sized physics batches submitted but not yet queue-confirmed. */
  gpuPendingBatches?: number;
  /** Simulation time represented by submitted, unconfirmed GPU work. */
  gpuInFlightSimulation_s?: number;
  /** Wall latency from submission to completion of the latest confirmed batch. */
  gpuBatchWall_ms?: number;
  /** Main-thread time spent encoding and submitting the latest GPU advance. */
  cpuAdvanceEncode_ms?: number;
  /** Intrusive pressure-only submission-to-completion sample. Unlike timestamp
   * queries this includes WebGPU implementation and driver command processing. */
  gpuPressureSolveObservedWall_ms?: number;
  /** Host encode time for the isolated pressure-only probe command buffer. */
  cpuPressureSolveProbeEncode_ms?: number;
  gpuPressureSolveObservedSampleId?: number;
  gpuPressureSolveObservedSimulation_s?: number;
  /** Monotonic wall spent in intrusive pressure replays. The renderer uses
   * its delta to keep profiler self-time out of production cadence. */
  gpuProfilerWallTotal_ms?: number;
  /** Intrusive production sample split at real command-buffer submission
   * boundaries. Unlike shader timestamps these intervals include Dawn, driver,
   * queue scheduling, and GPU completion work for each production phase. */
  gpuAdvancePhaseWall?: {
    sampleId: number;
    simulation_s: number;
    topologyAdvection_ms: number;
    pressureProjection_ms: number;
    surfaceCoupling_ms: number;
    publicationDiagnostics_ms: number;
    total_ms: number;
  };
  /** Solver-side attribution for the latest command encode. These regions are
   * host timings only; shader execution remains covered by timestamp queries. */
  cpuAdvanceEncodeBreakdown?: {
    setup_ms: number;
    topology_ms: number;
    /** Exact host time spent emitting the pressure solver itself. */
    pressureSolve_ms: number;
    /** Compute dispatches in the last pressure solver schedule. */
    pressureSolvePassCount: number;
    /** Actual compute-pass begin/end transitions around those dispatches. */
    pressureSolvePassTransitionCount: number;
    pressureProjection_ms: number;
    surface_ms: number;
    publication_ms: number;
    finalize_ms: number;
  };
  /** End-to-end wall time from starting command encoding through queue completion. */
  gpuAdvanceWall_ms?: number;
  /** Simulation time advanced by the latest confirmed batch. */
  gpuBatchSimulation_s?: number;
  /** Wall interval between the two latest ordered batch completions. */
  gpuCompletionWall_ms?: number;
  /** Portion of gpuCompletionWall_ms consumed by intrusive profiler work. */
  gpuCompletionProfilerWall_ms?: number;
  /** Completion interval with known intrusive profiler work removed. */
  gpuCompletionProductionWall_ms?: number;
  /** Simulation time confirmed during gpuCompletionWall_ms. */
  gpuCompletionSimulation_s?: number;
  /** Wall interval between queue-confirmed presentation completions. */
  gpuPresentationWall_ms?: number;
  /** Host-side gap between an empty queue completing and its next physics submission. */
  gpuQueueStarved_ms?: number;
  initialVolumeCellSum?:number;
  volumeDrift?:number;
  rawVolumeDrift?:number;
  referenceLiquidVolume_cells?: number;
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
  /** Latest asynchronous adaptive topology readback + rebuild latency. */
  quadtreeRebuildWall_ms?: number;
  /** GPU construction, vertical sizing, subdivision, smoothing, and compact readback latency. */
  quadtreeGPUConstruction_ms?: number;
  quadtreeGPUConstructionKernel_ms?: number;
  quadtreeGPUSparsePack_ms?: number;
  /** Remaining host time for exact redistance and sparse variational packing. */
  quadtreeCPUTopologyPack_ms?: number;
  quadtreeCPURedistance_ms?: number;
  quadtreeCPUQuadtreeDecode_ms?: number;
  quadtreeCPUTallGrid_ms?: number;
  quadtreeCPUVariationalAssembly_ms?: number;
  quadtreeCPUSystemPack_ms?: number;
  quadtreeCPUICFactorization_ms?: number;
  quadtreeCPUResourceUpload_ms?: number;
  quadtreeTopologyReused?: boolean;
  quadtreeTopologyReuseCount?: number;
  quadtreePressureIterationsUsed?: number;
  quadtreePressureIterationBudget?: number;
  quadtreePressureIterationHardBudget?: number;
  quadtreePressureConverged?: boolean;
  quadtreeFactorLevelCount?: number;
  quadtreeMultigridLevelCount?: number;
  quadtreeMultigridCoarsestDofs?: number;
  quadtreePressurePhaseTimings?: { setup_ms: number; firstIterations_ms: number; remainingIterations_ms: number; project_ms: number };
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
  gpuTimings?: GPUPhysicsTimings;
  /** Fluid authority remains in resident GPU resources between submissions. */
  hostFluidAuthority?: "gpu-resident" | "cpu-reference";
  /** Simulation-sized host work performed by one authoritative fluid frame. */
  hostSimulationSizedWorkItems?: number;
  /** Must remain false for authoritative octree scheduling. */
  hostSchedulingUsesReadback?: boolean;
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

export const legacyUniformComputeShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  inflowTiming: vec4f,
}
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var pressureIn: texture_3d<f32>;
@group(0) @binding(3) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var volumeIn: texture_3d<f32>;
@group(0) @binding(5) var volumeOut: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var heightIn: texture_2d<f32>;
@group(0) @binding(8) var heightOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(9) var<storage,read_write> reductions:array<atomic<u32>,4>;
struct RigidBody {
  positionShape: vec4f,
  dimensions: vec4f,
  orientation: vec4f,
  linearVelocity: vec4f,
  angularVelocity: vec4f,
  inverseMassInertia: vec4f,
  angularMomentumRestitution: vec4f,
  material: vec4f,
}
@group(0) @binding(10) var<storage,read> rigidBodies:array<RigidBody,12>;
@group(0) @binding(11) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(12) var predictedVelocityIn: texture_3d<f32>;
@group(0) @binding(13) var reversedVelocityIn: texture_3d<f32>;
// Precomputed transport velocity with a one-texel zero shell so hardware
// trilinear sampling reproduces the zero wall-face boundary condition.
@group(0) @binding(14) var transportIn: texture_3d<f32>;
@group(0) @binding(15) var transportSampler: sampler;
@group(0) @binding(16) var transportOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(17) var fluxScalesIn: texture_3d<f32>;
@group(0) @binding(18) var fluxScalesOut: texture_storage_3d<rg32float, write>;
@group(0) @binding(19) var<storage,read_write> sharpenDeposits:array<atomic<i32>>;
// The adaptive method binds its resident signed-distance field here. Uniform
// reference solvers bind volumeIn instead, preserving their VOF formulation.
@group(0) @binding(20) var surfaceIn: texture_3d<f32>;
// Per-column terrain heights in cell units; params.container.w enables it so
// terrain-free scenes never pay the extra load. Static for the whole run.
@group(0) @binding(21) var terrainIn: texture_2d<f32>;
// Optional bulk-field brick atlas. The page table is INVALID-only for uniform,
// tall-cell, atlas-off, and authoritative-infrastructure-only paths, making the
// helper fall back to the established dense transport texture.
struct BulkAtlasParams {
  dims: vec4u,
  brickDims: vec4u,
  tileGrid: vec4u,
  capacitySeed: vec4u,
  cell: vec4f,
}
@group(0) @binding(22) var<storage,read> bulkAtlasPageTable: array<u32>;
@group(0) @binding(23) var bulkAtlasVelocity: texture_3d<f32>;
@group(0) @binding(24) var<uniform> bulkAtlasParams: BulkAtlasParams;
// x enables atlas reads; y independently enables GPU-authored sparse targets.
@group(0) @binding(25) var<uniform> bulkAtlasControl: vec4u;
// Residency ABI: words 0/count, 12..14/cell64 indirect dispatch, then active
// (brick,leaf) pairs from word 16. The retired pair stream follows capacity.
@group(0) @binding(26) var<storage,read> bulkWorklist: array<u32>;
// Sparse occupancy reduces resident cells into one atomic maximum per x/z
// column, then publishes the same dense r32float height texture consumed by
// the unported advection kernels. A zero word represents the historical -1
// (empty-column) sentinel; occupied y is stored as y+1.
@group(0) @binding(27) var<storage,read_write> occupancyColumns: array<atomic<u32>>;

fn dims() -> vec3i { return vec3i(textureDimensions(volumeIn)); }
fn inflowGridDims()->vec3i{return dims();}
fn valid(p: vec3i) -> bool { let d=dims(); return all(p >= vec3i(0)) && all(p < d); }
fn clampCell(p: vec3i) -> vec3i { return clamp(p, vec3i(0), dims()-vec3i(1)); }
fn worldCell(id:vec3i)->vec3f{let h=params.cellGravity.xyz;return vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);}
fn hasTerrain()->bool{return params.container.w>0.5;}
fn terrainHeightCells(x:i32,z:i32)->f32{let d=dims();return textureLoad(terrainIn,vec2i(clamp(x,0,d.x-1),clamp(z,0,d.z-1)),0).x;}
// Ground handling mirrors the rigid-body solid treatment with zero velocity:
// the heightfield closes faces, drops pressure unknowns, and blocks deposits.
fn cellInsideTerrain(p:vec3i)->bool{if(!hasTerrain()){return false;}return f32(p.y)+0.5<terrainHeightCells(p.x,p.z);}
fn cellTerrainFraction(p:vec3i)->f32{if(!hasTerrain()){return 0.0;}return clamp(terrainHeightCells(p.x,p.z)-f32(p.y),0.0,1.0);}
${inflowBoundaryWGSL}
fn volume(p: vec3i) -> f32 { if (!valid(p)) { return 0.0; } return textureLoad(volumeIn,p,0).x; }
fn transportConservativeVolume() -> bool { return params.physical.z > 0.5; }
fn levelSetAuthority() -> bool { return params.physical.w > 0.5; }
fn hydrostaticSplit() -> bool { return params.inflowTiming.y > 0.5; }
fn surfaceValue(p: vec3i) -> f32 {
  if (!valid(p)) { return select(0.0, 5.0 * min(params.cellGravity.x, min(params.cellGravity.y, params.cellGravity.z)), levelSetAuthority()); }
  return textureLoad(surfaceIn, p, 0).x;
}
fn surfaceOccupancy(p: vec3i) -> f32 {
  if (!valid(p)) { return 0.0; }
  let value = surfaceValue(p);
  return select(clamp(value, 0.0, 1.0), clamp(0.5 - value / (4.0 * params.cellGravity.y), 0.0, 1.0), levelSetAuthority());
}
fn surfaceLiquid(p: vec3i) -> bool { return valid(p) && select(surfaceValue(p) >= 0.5, surfaceValue(p) < 0.0, levelSetAuthority()); }
fn velocity(p: vec3i) -> vec3f { return textureLoad(velocityIn,clampCell(p),0).xyz; }
fn faceVelocity(p:vec3i)->vec3f{if(!valid(p)){return vec3f(0.0);}return textureLoad(velocityIn,p,0).xyz;}
fn liquid(p:vec3i)->bool{return surfaceLiquid(p);}
fn pressureValue(p:vec3i)->f32{return textureLoad(pressureIn,clampCell(p),0).x;}
fn transportVelocity(id:vec3i)->vec3f{
  var v=velocity(id);if(surfaceOccupancy(id)>=0.01){return v;}var sum=vec3f(0.0);var weight=0.0;
  let px=surfaceOccupancy(id+vec3i(1,0,0));let nx=surfaceOccupancy(id-vec3i(1,0,0));let py=surfaceOccupancy(id+vec3i(0,1,0));let ny=surfaceOccupancy(id-vec3i(0,1,0));let pz=surfaceOccupancy(id+vec3i(0,0,1));let nz=surfaceOccupancy(id-vec3i(0,0,1));
  sum+=velocity(id+vec3i(1,0,0))*px+velocity(id-vec3i(1,0,0))*nx+velocity(id+vec3i(0,1,0))*py+velocity(id-vec3i(0,1,0))*ny+velocity(id+vec3i(0,0,1))*pz+velocity(id-vec3i(0,0,1))*nz;weight=px+nx+py+ny+pz+nz;if(weight>0.001){v=sum/weight;}return v;
}
fn sampledFaceVelocity(p:vec3i,component:u32)->f32{
  let d=dims();if(p[component]<0||p[component]>=d[component]){return 0.0;}
  return textureLoad(transportIn,clampCell(p)+vec3i(1),0)[component];
}
fn transportCoordinate(q:vec3f)->vec3f{return (q+vec3f(1.5))/vec3f(dims()+vec3i(2));}
const BULK_ATLAS_INVALID:u32=0xffffffffu;
fn bulkAtlasSlot(position:vec3f)->vec4u{
  if(bulkAtlasControl.x==0u||bulkAtlasParams.capacitySeed.x==0u){return vec4u(BULK_ATLAS_INVALID);}
  if(any(position<vec3f(0.0))||any(position>vec3f(bulkAtlasParams.dims.xyz-vec3u(1u)))){return vec4u(BULK_ATLAS_INVALID);}
  let p=clamp(position,vec3f(0.0),vec3f(bulkAtlasParams.dims.xyz-vec3u(1u)));
  let brick=min(vec3u(floor(p/f32(bulkAtlasParams.dims.w))),bulkAtlasParams.brickDims.xyz-vec3u(1u));
  let brickIndex=brick.x+bulkAtlasParams.brickDims.x*(brick.y+bulkAtlasParams.brickDims.y*brick.z);
  if(brickIndex>=arrayLength(&bulkAtlasPageTable)){return vec4u(BULK_ATLAS_INVALID,brick);}
  let slot=bulkAtlasPageTable[brickIndex];
  if(slot>=bulkAtlasParams.capacitySeed.x){return vec4u(BULK_ATLAS_INVALID,brick);}
  return vec4u(slot,brick);
}
fn bulkAtlasTileOrigin(slot:u32)->vec3u{
  let g=bulkAtlasParams.tileGrid;
  return vec3u(slot%g.x,(slot/g.x)%g.y,slot/(g.x*g.y))*g.w;
}
fn bulkAtlasTexel(slot:u32,brick:vec3u,cell:vec3i)->vec3i{
  return vec3i(bulkAtlasTileOrigin(slot))+cell-vec3i(brick*bulkAtlasParams.dims.w)+vec3i(1);
}
fn bulkAtlasSampleVelocity(position:vec3f,slotBrick:vec4u)->vec3f{
  let p=clamp(position,vec3f(0.0),vec3f(bulkAtlasParams.dims.xyz-vec3u(1u)));
  let a=vec3i(floor(p));let t=fract(p);let slot=slotBrick.x;let brick=slotBrick.yzw;
  let v000=textureLoad(bulkAtlasVelocity,bulkAtlasTexel(slot,brick,a),0).xyz;
  let v100=textureLoad(bulkAtlasVelocity,bulkAtlasTexel(slot,brick,a+vec3i(1,0,0)),0).xyz;
  let v010=textureLoad(bulkAtlasVelocity,bulkAtlasTexel(slot,brick,a+vec3i(0,1,0)),0).xyz;
  let v110=textureLoad(bulkAtlasVelocity,bulkAtlasTexel(slot,brick,a+vec3i(1,1,0)),0).xyz;
  let v001=textureLoad(bulkAtlasVelocity,bulkAtlasTexel(slot,brick,a+vec3i(0,0,1)),0).xyz;
  let v101=textureLoad(bulkAtlasVelocity,bulkAtlasTexel(slot,brick,a+vec3i(1,0,1)),0).xyz;
  let v011=textureLoad(bulkAtlasVelocity,bulkAtlasTexel(slot,brick,a+vec3i(0,1,1)),0).xyz;
  let v111=textureLoad(bulkAtlasVelocity,bulkAtlasTexel(slot,brick,a+vec3i(1,1,1)),0).xyz;
  return mix(mix(mix(v000,v100,t.x),mix(v010,v110,t.x),t.y),mix(mix(v001,v101,t.x),mix(v011,v111,t.x),t.y),t.z);
}
struct ScheduledCell { id: vec3i, scheduled: u32 }
fn sparseCell64Ready()->bool{return bulkWorklist[0]!=0u&&bulkWorklist[12]!=0u&&bulkWorklist[13]!=0u;}
fn scheduledVelocityCell(wid:vec3u,localIndex:u32,denseGid:vec3u)->ScheduledCell{
  if(bulkAtlasControl.y==0u){return ScheduledCell(vec3i(denseGid),1u);}
  let workgroupLinear=wid.x+wid.y*bulkWorklist[12];
  let stream=workgroupLinear*64u+localIndex;
  let brickVoxels=bulkAtlasParams.dims.w*bulkAtlasParams.dims.w*bulkAtlasParams.dims.w;
  let activeIndex=stream/brickVoxels;
  if(activeIndex>=bulkWorklist[0]){return ScheduledCell(vec3i(0),0u);}
  let entry=16u+activeIndex*2u;
  if(entry>=arrayLength(&bulkWorklist)){return ScheduledCell(vec3i(0),0u);}
  let brickIndex=bulkWorklist[entry];
  if(brickIndex>=bulkAtlasParams.brickDims.w){return ScheduledCell(vec3i(0),0u);}
  let b=bulkAtlasParams.brickDims;
  let brick=vec3u(brickIndex%b.x,(brickIndex/b.x)%b.y,brickIndex/(b.x*b.y));
  let localLinear=stream-activeIndex*brickVoxels;
  let size=bulkAtlasParams.dims.w;
  let local=vec3u(localLinear%size,(localLinear/size)%size,localLinear/(size*size));
  return ScheduledCell(vec3i(brick*size+local),1u);
}
fn interfaceFraction(a:f32,b:f32)->f32{
  // Distance from the liquid cell centre to alpha=0.5 along a grid edge.
  return clamp((a-0.5)/max(abs(a-b),1e-6),0.05,1.0);
}
fn sampleVolume(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);let c000=volume(b);let c100=volume(b+vec3i(1,0,0));let c010=volume(b+vec3i(0,1,0));let c110=volume(b+vec3i(1,1,0));let c001=volume(b+vec3i(0,0,1));let c101=volume(b+vec3i(1,0,1));let c011=volume(b+vec3i(0,1,1));let c111=volume(b+vec3i(1,1,1));return mix(mix(mix(c000,c100,f.x),mix(c010,c110,f.x),f.y),mix(mix(c001,c101,f.x),mix(c011,c111,f.x),f.y),f.z);
}
fn sampleVelocityComponent(p:vec3f,component:u32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;var lower=vec3f(0.0);lower[component]=-1.0;let q=clamp(p-offset,lower,vec3f(dims()-vec3i(1)));
  let slotBrick=bulkAtlasSlot(q);if(slotBrick.x!=BULK_ATLAS_INVALID){return bulkAtlasSampleVelocity(q,slotBrick)[component];}
  return textureSampleLevel(transportIn,transportSampler,transportCoordinate(q),0.0)[component];
}
fn sampleVelocity(p:vec3f)->vec3f{return vec3f(sampleVelocityComponent(p,0u),sampleVelocityComponent(p,1u),sampleVelocityComponent(p,2u));}
// One collocated vector fetch per RK2 stage; the half-texel stagger error only
// perturbs where the trace samples, not the sampled face values themselves.
fn transportVectorEstimate(p:vec3f)->vec3f{
  let q=clamp(p-vec3f(0.75),vec3f(-1.0),vec3f(dims()-vec3i(1)));
  let slotBrick=bulkAtlasSlot(q);if(slotBrick.x!=BULK_ATLAS_INVALID){return bulkAtlasSampleVelocity(q,slotBrick);}
  return textureSampleLevel(transportIn,transportSampler,transportCoordinate(q),0.0).xyz;
}
fn departurePoint(position:vec3f,dt:f32,h:vec3f)->vec3f{let first=transportVectorEstimate(position);let midpoint=position-0.5*first*dt/h;return position-transportVectorEstimate(midpoint)*dt/h;}
fn advectVelocityComponent(position:vec3f,component:u32,dt:f32,h:vec3f)->f32{
  return sampleVelocityComponent(departurePoint(position,dt,h),component);
}
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
fn insideRigid(body:RigidBody,world:vec3f)->bool{
  let p=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)<=d.x;}
  if(shape==1){return all(abs(p)<=0.5*d);}
  if(shape==2){let cy=clamp(p.y,-0.5*d.y,0.5*d.y);return length(vec3f(p.x,p.y-cy,p.z))<=d.x;}
  return p.x*p.x+p.z*p.z<=d.x*d.x&&abs(p.y)<=0.5*d.y;
}
fn rigidBodyIndexAt(world:vec3f)->i32{
  let bodyCount=u32(round(params.boundary.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],world)){return i32(bodyIndex);}}
  return -1;
}
fn rigidVelocityAt(bodyIndex:i32,world:vec3f)->vec3f{
  let body=rigidBodies[u32(bodyIndex)];
  return body.linearVelocity.xyz+cross(body.angularVelocity.xyz,world-body.positionShape.xyz);
}
// Conservative bounding-sphere reject so cells away from every body (and
// body-free scenes) skip the per-cell primitive tests in the solid-aware
// pressure, projection, and coupling kernels.
fn nearAnyBody(world:vec3f)->bool{
  let bodyCount=u32(round(params.boundary.z));
  let margin=2.0*max(params.cellGravity.x,max(params.cellGravity.y,params.cellGravity.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}
    let body=rigidBodies[bodyIndex];let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
    var radius=0.5*length(d);
    if(shape==0){radius=d.x;}
    if(shape==2){radius=d.x+0.5*d.y;}
    if(shape==3){radius=sqrt(d.x*d.x+0.25*d.y*d.y);}
    if(distance(world,body.positionShape.xyz)<=radius+margin){return true;}
  }
  return false;
}
// Paper Sec 3.9.1 treats a cell as solid in the divergence when its solid
// fraction is high; the cell-centre point-in-primitive test is our s>0.9.
fn cellRigidBody(p:vec3i)->i32{
  if(!valid(p)){return -1;}
  return rigidBodyIndexAt(worldCell(p));
}
// Sub-cell solid fraction with the CPU voxelizer's 8-corner sampling
// (solidFieldsFromBodies), so mixed cells blend rather than snap.
fn bodySolidFraction(body:RigidBody,p:vec3i)->f32{
  var inside=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
    if(insideRigid(body,worldCell(p)+offset*params.cellGravity.xyz)){inside+=1.0;}
  }
  return inside/8.0;
}
fn cellSolidFraction(p:vec3i)->f32{
  let bodyCount=u32(round(params.boundary.z));var fraction=0.0;
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}fraction=max(fraction,bodySolidFraction(rigidBodies[bodyIndex],p));}
  return fraction;
}
// Projection enforces body velocity on interior faces, so that value cannot be
// used as the undisturbed fluid velocity for form drag. Sample six wet, open
// points just beyond the body's bounding sphere instead.
fn ambientFluidVelocity(body:RigidBody,p:vec3i,fallback:vec3f)->vec3f{
  let h=params.cellGravity.xyz;let radius=max(body.dimensions.w,0.0);let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
  let offsets=array<vec3i,6>(vec3i(-reach.x,0,0),vec3i(reach.x,0,0),vec3i(0,-reach.y,0),vec3i(0,reach.y,0),vec3i(0,0,-reach.z),vec3i(0,0,reach.z));
  var total=vec3f(0.0);var weight=0.0;
  for(var n=0;n<6;n+=1){let q=p+offsets[n];if(!valid(q)||cellRigidBody(q)>=0||cellInsideTerrain(q)){continue;}let wet=surfaceOccupancy(q);total+=wet*velocity(q);weight+=wet;}
  return select(fallback,total/max(weight,1e-6),weight>0.0);
}
fn columnHeight(x:i32,z:i32)->f32{
  let d=dims();if(x<0||x>=d.x||z<0||z>=d.z){return 0.0;}return textureLoad(heightIn,vec2i(x,z),0).x;
}
fn hydrostaticSurfaceCells(x:i32,z:i32)->f32{
  let d=dims();if(x<0||x>=d.x||z<0||z>=d.z){return -1.0;}return textureLoad(heightIn,vec2i(x,z),0).y;
}
fn hydrostaticColumnContains(id:vec3i)->bool{
  let eta=hydrostaticSurfaceCells(id.x,id.z);
  return eta>=0.0&&f32(id.y)+0.5<eta;
}
fn fixedHydrostaticPotentialAtY(yCells:f32)->f32{
  return -params.cellGravity.w*max((params.inflowTiming.z-yCells)*params.cellGravity.y,0.0);
}
fn fixedHydrostaticAcceleration(id:vec3i)->f32{
  let neighbor=id+vec3i(0,1,0);
  if(!hydrostaticColumnContains(id)&&!hydrostaticColumnContains(neighbor)){return params.cellGravity.w;}
  let y0=f32(id.y)+0.5;var y1=f32(neighbor.y)+0.5;
  let phi0=surfaceValue(id);let phi1=surfaceValue(neighbor);
  var distance=params.cellGravity.y;
  if((phi0<0.0)!=(phi1<0.0)){
    let crossing=clamp(-phi0/(phi1-phi0),0.01,1.0);
    y1=mix(y0,y1,crossing);distance*=crossing;
  }
  let gradient=(fixedHydrostaticPotentialAtY(y1)-fixedHydrostaticPotentialAtY(y0))/max(distance,1e-7);
  return params.cellGravity.w-gradient;
}
fn upwind(face:f32,negative:f32,positive:f32)->f32{return face*select(positive,negative,face>=0.0);}
fn normalSurfaceOccupancy(id:vec3i)->f32{
  if(valid(id)){return surfaceOccupancy(id);}
  // Side walls and the floor are solids, so extend alpha with a zero-normal
  // gradient instead of inventing an air interface at the wall. Only an open
  // top (boundary.w) is allowed to expose liquid to exterior air.
  if(id.y>=dims().y&&params.boundary.w>0.5){return 0.0;}
  return surfaceOccupancy(clampCell(id));
}
fn surfaceGradient(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(normalSurfaceOccupancy(id+vec3i(1,0,0))-normalSurfaceOccupancy(id-vec3i(1,0,0)),normalSurfaceOccupancy(id+vec3i(0,1,0))-normalSurfaceOccupancy(id-vec3i(0,1,0)),normalSurfaceOccupancy(id+vec3i(0,0,1))-normalSurfaceOccupancy(id-vec3i(0,0,1)))/(2.0*h);
}
fn interfaceNormal(id:vec3i)->vec3f{
  let gradient=surfaceGradient(id);
  return gradient/max(length(gradient),1e-6);
}
// The diagnostic/emergency VOF still sharpens along its own density gradient;
// this field does not classify the adaptive pressure or velocity solve.
fn normalVolume(id:vec3i)->f32{
  if(valid(id)){return volume(id);}
  if(id.y>=dims().y&&params.boundary.w>0.5){return 0.0;}
  return textureLoad(volumeIn,clampCell(id),0).x;
}
fn volumeGradient(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(normalVolume(id+vec3i(1,0,0))-normalVolume(id-vec3i(1,0,0)),normalVolume(id+vec3i(0,1,0))-normalVolume(id-vec3i(0,1,0)),normalVolume(id+vec3i(0,0,1))-normalVolume(id-vec3i(0,0,1)))/(2.0*h);
}
fn rawVolumeFlux(id:vec3i,axis:u32,dt:f32)->f32{
  if(!valid(id)){return 0.0;}
  let neighbor=id+select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u);
  let speed=faceVelocity(id)[axis];
  return dt/params.cellGravity.xyz[axis]*upwind(speed,volume(id),volume(neighbor));
}
fn outwardFlux(id:vec3i,dt:f32)->f32{
  if(!valid(id)){return 0.0;}
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  return max(rawVolumeFlux(id,0u,dt),0.0)+max(-rawVolumeFlux(id-ex,0u,dt),0.0)
       + max(rawVolumeFlux(id,1u,dt),0.0)+max(-rawVolumeFlux(id-ey,1u,dt),0.0)
       + max(rawVolumeFlux(id,2u,dt),0.0)+max(-rawVolumeFlux(id-ez,2u,dt),0.0);
}
fn inwardFlux(id:vec3i,dt:f32)->f32{
  if(!valid(id)){return 0.0;}
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  return max(-rawVolumeFlux(id,0u,dt),0.0)+max(rawVolumeFlux(id-ex,0u,dt),0.0)
       + max(-rawVolumeFlux(id,1u,dt),0.0)+max(rawVolumeFlux(id-ey,1u,dt),0.0)
       + max(-rawVolumeFlux(id,2u,dt),0.0)+max(rawVolumeFlux(id-ez,2u,dt),0.0);
}
fn donorScale(id:vec3i,dt:f32)->f32{return min(1.0,volume(id)/max(outwardFlux(id,dt),1e-9));}
fn receiverScale(id:vec3i,dt:f32)->f32{return min(1.0,max(0.0,1.0-volume(id))/max(inwardFlux(id,dt),1e-9));}
// Scales are precomputed once per cell by buildFluxScales; invalid neighbors
// keep the historical donor 0 / receiver 1 limits.
fn cellFluxScales(id:vec3i)->vec2f{if(!valid(id)){return vec2f(0.0,1.0);}return textureLoad(fluxScalesIn,id,0).xy;}
fn limitedVolumeFlux(id:vec3i,axis:u32,dt:f32)->f32{
  let offset=select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u);
  let neighbor=id+offset;let flux=rawVolumeFlux(id,axis,dt);
  let donor=cellFluxScales(id);let receiver=cellFluxScales(neighbor);
  if(flux>=0.0){return flux*min(donor.x,receiver.y);}
  return flux*min(receiver.x,donor.y);
}
fn advectedVolume(id:vec3i,dt:f32)->f32{
  let centre=volume(id);
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  let fxp=limitedVolumeFlux(id,0u,dt);let fxm=limitedVolumeFlux(id-ex,0u,dt);
  let fyp=limitedVolumeFlux(id,1u,dt);let fym=limitedVolumeFlux(id-ey,1u,dt);
  let fzp=limitedVolumeFlux(id,2u,dt);let fzm=limitedVolumeFlux(id-ez,2u,dt);
  // No upper clamp on the transported value: a clamp here would destroy
  // sharpening deposits above one, which drain through the correction
  // divergence instead. The inflow source alone is bounded by the cell's
  // remaining capacity, as the old clamp did implicitly.
  let bounded=max(centre-(fxp-fxm+fyp-fym+fzp-fzm),0.0);
  return bounded+min(inflowReceiverSource(id,dt),max(0.0,1.0-bounded));
}

fn diffusionVelocity(p:vec3i)->vec3f{let v=textureLoad(velocityIn,clampCell(p),0).xyz;if(params.boundary.y>0.5&&!valid(p)){return -v;}return v;}
fn strainMagnitude(id:vec3i)->f32{
  let h=params.cellGravity.xyz;let dx=(diffusionVelocity(id+vec3i(1,0,0))-diffusionVelocity(id-vec3i(1,0,0)))/(2.0*h.x);let dy=(diffusionVelocity(id+vec3i(0,1,0))-diffusionVelocity(id-vec3i(0,1,0)))/(2.0*h.y);let dz=(diffusionVelocity(id+vec3i(0,0,1))-diffusionVelocity(id-vec3i(0,0,1)))/(2.0*h.z);let sxy=0.5*(dx.y+dy.x);let sxz=0.5*(dx.z+dz.x);let syz=0.5*(dy.z+dz.y);
  return sqrt(2.0*(dx.x*dx.x+dy.y*dy.y+dz.z*dz.z+2.0*(sxy*sxy+sxz*sxz+syz*syz)));
}
fn velocityLaplacian(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;let centre=diffusionVelocity(id);
  return (diffusionVelocity(id+vec3i(1,0,0))-2.0*centre+diffusionVelocity(id-vec3i(1,0,0)))/(h.x*h.x)+(diffusionVelocity(id+vec3i(0,1,0))-2.0*centre+diffusionVelocity(id-vec3i(0,1,0)))/(h.y*h.y)+(diffusionVelocity(id+vec3i(0,0,1))-2.0*centre+diffusionVelocity(id-vec3i(0,0,1)))/(h.z*h.z);
}

fn applyVelocityForces(id:vec3i,inputVelocity:vec3f,dt:f32,h:vec3f)->vec3f{
  var v=inputVelocity;let occupancy=surfaceOccupancy(id);if(occupancy>0.0){let molecular=params.physical.y/params.physical.x;v+=dt*molecular*velocityLaplacian(id);}
  // Body force lives on faces. A face participates whenever liquid exists on
  // either side; this is the same rule during impact and at equilibrium.
  let qy=id+vec3i(0,1,0);let yOccupancy=surfaceOccupancy(qy);
  let centerLiquid=select(occupancy>=0.5,occupancy>0.5,levelSetAuthority());
  let yLiquid=select(yOccupancy>=0.5,yOccupancy>0.5,levelSetAuthority());
  if(centerLiquid||yLiquid){
    if(hydrostaticSplit()){v.y+=fixedHydrostaticAcceleration(id)*dt;}
    else{v.y+=params.cellGravity.w*dt;}
  }
  let qx=id+vec3i(1,0,0);let qz=id+vec3i(0,0,1);
  let xOccupancy=surfaceOccupancy(qx);let zOccupancy=surfaceOccupancy(qz);
  // Balanced-force CSF: pressure and capillary acceleration use the same
  // positive-face locations and alpha differences. Curvature is a deep
  // stencil, so evaluate the centre once and only on faces whose occupancy
  // difference can produce a non-zero force. The previous formulation
  // evaluated centre curvature three times and paid six curvature stencils in
  // every bulk cell even though the final multiplication was exactly zero.
  let sigmaOverRho=params.boundary.x/params.physical.x;
  if(sigmaOverRho>0.0){
    let dx=select(0.0,xOccupancy-occupancy,valid(qx));
    let dy=select(0.0,yOccupancy-occupancy,valid(qy));
    let dz=select(0.0,zOccupancy-occupancy,valid(qz));
    if(dx!=0.0||dy!=0.0||dz!=0.0){
      let centreCurvature=curvatureAt(id);
      if(dx!=0.0){v.x+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qx))*dx/h.x;}
      if(dy!=0.0){v.y+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qy))*dy/h.y;}
      if(dz!=0.0){v.z+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qz))*dz/h.z;}
    }
  }
  v=applyInflowVelocity(id,v);let d=dims();if(id.x==d.x-1){v.x=0.0;}if(id.y==d.y-1){v.y=0.0;}if(id.z==d.z-1){v.z=0.0;}return v;
}

@compute @workgroup_size(4,4,4)
fn buildTransport(@builtin(global_invocation_id) gid:vec3u,@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) localIndex:u32){
  var padded=vec3i(gid);let d=dims();var id=padded-vec3i(1);
  if(bulkAtlasControl.y!=0u){
    let scheduled=scheduledVelocityCell(wid,localIndex,gid);if(scheduled.scheduled==0u){return;}
    id=scheduled.id;if(!valid(id)){return;}padded=id+vec3i(1);
  }else{
    if(any(padded>=d+vec3i(2))){return;}
    if(!valid(id)){textureStore(transportOut,padded,vec4f(0.0));return;}
  }
  textureStore(transportOut,padded,vec4f(transportVelocity(id),0.0));
}
@compute @workgroup_size(4,4,4)
fn buildFluxScales(@builtin(global_invocation_id) gid:vec3u,@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) localIndex:u32){
  var id=vec3i(gid);
  if(bulkAtlasControl.y!=0u){let scheduled=scheduledVelocityCell(wid,localIndex,gid);if(scheduled.scheduled==0u){return;}id=scheduled.id;}
  if(!valid(id)){return;}let dt=params.dimsDt.w;
  textureStore(fluxScalesOut,id,vec4f(donorScale(id,dt),receiverScale(id,dt),0.0,0.0));
}
// Highest cell supported by the authoritative surface in each column;
// advection skips cells well above it after projection zeroes their faces.
// The second channel is a separate bottom/terrain-connected zero crossing for
// the hydrostatic reference. Floating sheets and spray never enter that field.
@compute @workgroup_size(8,8,1)
fn buildOccupancy(@builtin(global_invocation_id) gid:vec3u){
  let d=dims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}
  // The GPU-authored list is allowed to be empty during first-publication or
  // overflow recovery. Sparse mode always launches this area-only sentinel;
  // it becomes the historical dense y scan only when no indirect work exists.
  if(bulkAtlasControl.y!=0u&&sparseCell64Ready()){return;}
  var highest=-1.0;
  if(!hydrostaticSplit()){
    for(var y:i32=d.y-1;y>=0;y-=1){if(surfaceOccupancy(vec3i(i32(gid.x),y,i32(gid.y)))>0.0001){highest=f32(y);break;}}
    textureStore(heightOut,vec2i(gid.xy),vec4f(highest,-1.0,0.0,0.0));return;
  }
  var referenceTop=-1;var referenceStarted=false;var referenceEnded=false;
  for(var y:i32=0;y<d.y;y+=1){
    let p=vec3i(i32(gid.x),y,i32(gid.y));let occupied=surfaceOccupancy(p)>0.0001;
    if(occupied){highest=f32(y);}
    if(!referenceEnded&&!cellInsideTerrain(p)){
      let wet=surfaceLiquid(p);
      if(!referenceStarted){referenceStarted=wet;if(wet){referenceTop=y;}else{referenceEnded=true;}}
      else if(wet){referenceTop=y;}else{referenceEnded=true;}
    }
  }
  var eta=-1.0;
  if(referenceTop>=0){
    let p=vec3i(i32(gid.x),referenceTop,i32(gid.y));let phi0=surfaceValue(p);let phi1=surfaceValue(p+vec3i(0,1,0));let difference=phi1-phi0;
    let crossing=select(1.0,clamp(-phi0/difference,0.0,1.0),difference>1e-7);
    eta=f32(referenceTop)+0.5+crossing;
  }
  textureStore(heightOut,vec2i(gid.xy),vec4f(highest,eta,0.0,0.0));
}
// Cell64 deliberately visits every resident payload cell. Atomics collapse
// all y bricks sharing a column without races, including floating/disconnected
// liquid; the following area-only resolve preserves dense texture semantics.
@compute @workgroup_size(4,4,4)
fn buildSparseOccupancy(@builtin(global_invocation_id) gid:vec3u,@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let scheduled=scheduledVelocityCell(wid,localIndex,gid);if(scheduled.scheduled==0u){return;}
  let id=scheduled.id;if(!valid(id)||surfaceOccupancy(id)<=0.0001){return;}
  let d=dims();let column=u32(id.x+d.x*id.z);if(column>=arrayLength(&occupancyColumns)){return;}
  atomicMax(&occupancyColumns[column],u32(id.y+1));
}
@compute @workgroup_size(8,8,1)
fn resolveSparseOccupancy(@builtin(global_invocation_id) gid:vec3u){
  let d=dims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}
  if(!sparseCell64Ready()){return;}
  let column=gid.x+u32(d.x)*gid.y;if(column>=arrayLength(&occupancyColumns)){return;}
  textureStore(heightOut,vec2i(gid.xy),vec4f(f32(atomicLoad(&occupancyColumns[column]))-1.0,-1.0,0.0,0.0));
}
fn nearInflow(id:vec3i)->bool{
  if(inflowStrength()<=0.0){return false;}
  let axis=inflowAxis();let face=inflowFaceIndex(axis);
  return id[axis]>=face-1&&id[axis]<=face+2&&inflowApertureFraction(id)>0.0;
}
fn aboveOccupancy(id:vec3i)->bool{
  let d=dims();var occupancy=-1.0;
  for(var dz:i32=-1;dz<=1;dz+=1){for(var dx:i32=-1;dx<=1;dx+=1){
    occupancy=max(occupancy,textureLoad(heightIn,vec2i(clamp(id.x+dx,0,d.x-1),clamp(id.z+dz,0,d.z-1)),0).x);
  }}
  return f32(id.y)>occupancy+4.0&&!nearInflow(id);
}
@compute @workgroup_size(4,4,4)
fn semiLagrangianAdvection(@builtin(global_invocation_id) gid:vec3u,@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let scheduled=scheduledVelocityCell(wid,localIndex,gid);if(scheduled.scheduled==0u){return;}let id=scheduled.id;if(!valid(id)){return;}let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  if(aboveOccupancy(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(volume(id),0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
  var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));v=applyVelocityForces(id,v,dt,h);
  var advected=volume(id);if(transportConservativeVolume()){advected=advectedVolume(id,dt);}textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(advected,0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn advect(@builtin(global_invocation_id) gid: vec3u,@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) localIndex:u32) {
  let scheduled=scheduledVelocityCell(wid,localIndex,gid);if(scheduled.scheduled==0u){return;}let id=scheduled.id; if (!valid(id)) { return; }
  if(aboveOccupancy(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(volume(id),0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
  let dt=params.dimsDt.w; let h=params.cellGravity.xyz;
  let cell=vec3f(id);var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));
  var advected=volume(id);if(transportConservativeVolume()){advected=advectedVolume(id,dt);}let d=dims();
  if (id.x==d.x-1) { v.x=0.0; }
  if (id.y==d.y-1) { v.y=0.0; }
  if (id.z==d.z-1) { v.z=0.0; }
  textureStore(velocityOut,id,vec4f(v,0.0));
  textureStore(volumeOut,id,vec4f(advected,0.0,0.0,0.0));
  textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn reverseAdvection(@builtin(global_invocation_id) gid:vec3u,@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let scheduled=scheduledVelocityCell(wid,localIndex,gid);if(scheduled.scheduled==0u){return;}let id=scheduled.id;if(!valid(id)){return;}
  if(aboveOccupancy(id)){textureStore(velocityOut,id,vec4f(0.0));return;}
  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,-dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,-dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,-dt,h));let d=dims();
  if(id.x==d.x-1){v.x=0.0;}if(id.y==d.y-1){v.y=0.0;}if(id.z==d.z-1){v.z=0.0;}textureStore(velocityOut,id,vec4f(v,0.0));
}

fn boundedMacCormack(id:vec3i,position:vec3f,component:u32,dt:f32,h:vec3f,predicted:f32,original:f32,reversed:f32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;var lowerCoordinate=vec3f(0.0);lowerCoordinate[component]=-1.0;
  let q=clamp(departurePoint(position,dt,h)-offset,lowerCoordinate,vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));
  var lower=sampledFaceVelocity(b,component);var upper=lower;
  for(var corner:u32=1u;corner<8u;corner+=1u){let cornerOffset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let value=sampledFaceVelocity(b+cornerOffset,component);lower=min(lower,value);upper=max(upper,value);}
  let corrected=predicted+0.5*(original-reversed);
  return select(corrected,predicted,corrected<lower||corrected>upper);
}

@compute @workgroup_size(4,4,4)
fn correctAdvection(@builtin(global_invocation_id) gid:vec3u,@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let scheduled=scheduledVelocityCell(wid,localIndex,gid);if(scheduled.scheduled==0u){return;}let id=scheduled.id;if(!valid(id)){return;}
  if(aboveOccupancy(id)){textureStore(velocityOut,id,vec4f(0.0));return;}
  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  let predicted=textureLoad(predictedVelocityIn,id,0).xyz;let original=textureLoad(velocityIn,id,0).xyz;let reversed=textureLoad(reversedVelocityIn,id,0).xyz;
  var v=vec3f(boundedMacCormack(id,cell+vec3f(1.0,0.5,0.5),0u,dt,h,predicted.x,original.x,reversed.x),boundedMacCormack(id,cell+vec3f(0.5,1.0,0.5),1u,dt,h,predicted.y,original.y,reversed.y),boundedMacCormack(id,cell+vec3f(0.5,0.5,1.0),2u,dt,h,predicted.z,original.z,reversed.z));v=applyVelocityForces(id,v,dt,h);
  textureStore(velocityOut,id,vec4f(v,0.0));
}

@compute @workgroup_size(8,8,1)
fn buildHeight(@builtin(global_invocation_id) gid:vec3u){let d=dims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}var total=0.0;for(var y:i32=0;y<d.y;y+=1){total+=volume(vec3i(i32(gid.x),y,i32(gid.y)))*params.cellGravity.y;}textureStore(heightOut,vec2i(gid.xy),vec4f(total));}

fn faceWorld(id:vec3i,axis:u32)->vec3f{
  var world=worldCell(id);world[axis]+=0.5*params.cellGravity.xyz[axis];return world;
}
// Positive-side face velocity with the paper's VOS constraint (Sec 3.9.1): a
// face touching a rigid-solid cell carries the solid velocity, which is what
// makes a moving body sweep water out of its path instead of ignoring it.
fn constrainedFaceVelocity(id:vec3i,axis:u32,checkSolid:bool)->f32{
  var neighbor=id;neighbor[axis]+=1;
  // The terrain heightfield is a static solid: a face touching it carries the
  // ground's zero velocity in the divergence, exactly like a wall.
  if(cellInsideTerrain(id)||cellInsideTerrain(neighbor)){return 0.0;}
  if(checkSolid){
    let body=max(cellRigidBody(id),cellRigidBody(neighbor));
    if(body>=0){return rigidVelocityAt(body,faceWorld(id,axis))[axis];}
  }
  return faceVelocity(id)[axis];
}
fn divergenceAt(id: vec3i, checkSolid: bool) -> f32 {
  let h=params.cellGravity.xyz;
  return (constrainedFaceVelocity(id,0u,checkSolid)-constrainedFaceVelocity(id-vec3i(1,0,0),0u,checkSolid))/h.x
       + (constrainedFaceVelocity(id,1u,checkSolid)-constrainedFaceVelocity(id-vec3i(0,1,0),1u,checkSolid))/h.y
       + (constrainedFaceVelocity(id,2u,checkSolid)-constrainedFaceVelocity(id-vec3i(0,0,1),2u,checkSolid))/h.z;
}
// Mass-Conserving Eulerian Liquid Simulation Sec 3.7: cells holding more
// density than they represent add min(lambda (rho'-1), eta) artificial
// divergence (lambda = 0.5, eta = 1 per the paper, expressed as a rate
// against its 1/30 s step) so the pressure solve pushes the excess out.
fn volumeCorrectionDivergence(id: vec3i) -> f32 {
  let excess=max(0.0,volume(id)-1.0);
  if(excess<=0.0){return 0.0;}
  return min(0.5*excess,1.0)*30.0;
}

fn curvatureAt(id:vec3i)->f32{
  let h=params.cellGravity.xyz;
  return -((interfaceNormal(id+vec3i(1,0,0)).x-interfaceNormal(id-vec3i(1,0,0)).x)/(2.0*h.x)+(interfaceNormal(id+vec3i(0,1,0)).y-interfaceNormal(id-vec3i(0,1,0)).y)/(2.0*h.y)+(interfaceNormal(id+vec3i(0,0,1)).z-interfaceNormal(id-vec3i(0,0,1)).z)/(2.0*h.z));
}

fn stencilCoefficient(id:vec3i,neighbor:vec3i,axis:u32,checkSolid:bool)->f32{
  if(!valid(neighbor)){return 0.0;}
  // A rigid-solid or terrain neighbor is a Neumann boundary exactly like a
  // wall; its motion enters through the divergence, not the stencil.
  if(cellInsideTerrain(neighbor)){return 0.0;}
  if(checkSolid&&cellRigidBody(neighbor)>=0){return 0.0;}
  let h=params.cellGravity.xyz[axis];
  if(liquid(neighbor)){return 1.0/(h*h);}
  return 1.0/(interfaceFraction(volume(id),volume(neighbor))*h*h);
}

fn stencilPressure(id:vec3i,neighbor:vec3i,axis:u32,checkSolid:bool)->f32{
  if(!valid(neighbor)||!liquid(neighbor)){return 0.0;}
  return stencilCoefficient(id,neighbor,axis,checkSolid)*pressureValue(neighbor);
}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  if (!liquid(id)) { textureStore(pressureOut,id,vec4f(0.0)); return; }
  // Ground cells are solid, not pressure unknowns, like body interiors below.
  if(cellInsideTerrain(id)){textureStore(pressureOut,id,vec4f(0.0));return;}
  let checkSolid=nearAnyBody(worldCell(id));
  // Paper Sec 3.9.1: cells occupied by a rigid body are solid, not pressure
  // unknowns. Without this the sphere interior stays "water" and the solve
  // never displaces it.
  if(checkSolid&&cellRigidBody(id)>=0){textureStore(pressureOut,id,vec4f(0.0));return;}
  let old=textureLoad(pressureIn,id,0).x;let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  let diagonal=stencilCoefficient(id,id-ex,0u,checkSolid)+stencilCoefficient(id,id+ex,0u,checkSolid)+stencilCoefficient(id,id-ey,1u,checkSolid)+stencilCoefficient(id,id+ey,1u,checkSolid)+stencilCoefficient(id,id-ez,2u,checkSolid)+stencilCoefficient(id,id+ez,2u,checkSolid);
  let sum=stencilPressure(id,id-ex,0u,checkSolid)+stencilPressure(id,id+ex,0u,checkSolid)+stencilPressure(id,id-ey,1u,checkSolid)+stencilPressure(id,id+ey,1u,checkSolid)+stencilPressure(id,id-ez,2u,checkSolid)+stencilPressure(id,id+ez,2u,checkSolid);
  // Subtracted so the projection leaves div_new = +c at overfull cells (an
  // outward drain); added it would leave div_new = -c and feed the excess.
  let rhs=params.physical.x*(divergenceAt(id,checkSolid)-volumeCorrectionDivergence(id))/params.dimsDt.w;
  // A liquid cell sealed on all six sides (tight body/wall gap) has no
  // stencil; leave it unconstrained instead of dividing by epsilon.
  if(diagonal<=0.0){textureStore(pressureOut,id,vec4f(0.0));return;}
  let next=(sum-rhs)/max(diagonal,1e-9);
  textureStore(pressureOut,id,vec4f(mix(old,next,0.8),0.0,0.0,0.0));
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  let h=params.cellGravity.xyz;let scale=params.dimsDt.w/params.physical.x;var v=velocity(id);let d=dims();
  let p0=select(0.0,pressureValue(id),liquid(id));
  let ex=id+vec3i(1,0,0);let ey=id+vec3i(0,1,0);let ez=id+vec3i(0,0,1);
  if(id.x==d.x-1){v.x=0.0;}else if(liquid(id)||liquid(ex)){let p1=select(0.0,pressureValue(ex),liquid(ex));let theta=select(interfaceFraction(volume(ex),volume(id)),interfaceFraction(volume(id),volume(ex)),liquid(id));v.x-=scale*(p1-p0)/(h.x*select(theta,1.0,liquid(id)&&liquid(ex)));}else{v.x=0.0;}
  if(id.y==d.y-1){v.y=0.0;}else if(liquid(id)||liquid(ey)){let p1=select(0.0,pressureValue(ey),liquid(ey));let theta=select(interfaceFraction(volume(ey),volume(id)),interfaceFraction(volume(id),volume(ey)),liquid(id));v.y-=scale*(p1-p0)/(h.y*select(theta,1.0,liquid(id)&&liquid(ey)));}else{v.y=0.0;}
  if(id.z==d.z-1){v.z=0.0;}else if(liquid(id)||liquid(ez)){let p1=select(0.0,pressureValue(ez),liquid(ez));let theta=select(interfaceFraction(volume(ez),volume(id)),interfaceFraction(volume(id),volume(ez)),liquid(id));v.z-=scale*(p1-p0)/(h.z*select(theta,1.0,liquid(id)&&liquid(ez)));}else{v.z=0.0;}
  // Faces the terrain heightfield covers are no-flux ground, like the floor.
  if(hasTerrain()){
    if(cellInsideTerrain(id)||cellInsideTerrain(ex)){v.x=0.0;}
    if(cellInsideTerrain(id)||cellInsideTerrain(ey)){v.y=0.0;}
    if(cellInsideTerrain(id)||cellInsideTerrain(ez)){v.z=0.0;}
  }
  // Faces covered by a rigid body move with the body (paper Sec 3.9.1); the
  // VOF fluxes then transport volume out of the body's path. Domain-edge
  // faces stay walls.
  if(nearAnyBody(worldCell(id))){
    let bodyX=max(cellRigidBody(id),cellRigidBody(ex));
    let bodyY=max(cellRigidBody(id),cellRigidBody(ey));
    let bodyZ=max(cellRigidBody(id),cellRigidBody(ez));
    if(bodyX>=0&&id.x<d.x-1){v.x=rigidVelocityAt(bodyX,faceWorld(id,0u)).x;}
    if(bodyY>=0&&id.y<d.y-1){v.y=rigidVelocityAt(bodyY,faceWorld(id,1u)).y;}
    if(bodyZ>=0&&id.z<d.z-1){v.z=rigidVelocityAt(bodyZ,faceWorld(id,2u)).z;}
  }
  v=applyInflowVelocity(id,v);textureStore(velocityOut,id,vec4f(v,0.0)); textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
}

// Brinkman-style immersed boundary: drive wet cells inside each moving primitive
// toward the local solid velocity and accumulate the exact opposite impulse.
@compute @workgroup_size(4,4,4)
fn coupleRigid(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let phi=volume(id);let wetFraction=surfaceOccupancy(id);var v=velocity(id);let h=params.cellGravity.xyz;
  let world=vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);
  let bodyCount=u32(round(params.boundary.z));let cellMass=params.physical.x*h.x*h.y*h.z*wetFraction;let blend=clamp(45.0*params.dimsDt.w,0.0,1.0);var coupledBody=12u;var solidFraction=0.0;
  // Match the adaptive voxelizer's overlap rule: the body with the greatest
  // sub-cell coverage owns this cell, so displaced volume is never counted
  // twice and does not depend on body-array order.
  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let candidate=bodySolidFraction(rigidBodies[bodyIndex],id);if(candidate>solidFraction){solidFraction=candidate;coupledBody=bodyIndex;}}
  if(coupledBody<12u){
    let bodyIndex=coupledBody;let body=rigidBodies[bodyIndex];
    let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let fluidVelocity=v;let ambientVelocity=ambientFluidVelocity(body,id,fluidVelocity);let fluidImpulse=cellMass*solidFraction*(solidVelocity-fluidVelocity)*blend;v+=fluidImpulse/max(cellMass,1e-9);
    let reaction=-fluidImpulse;let torque=cross(arm,reaction);let base=bodyIndex*12u;
    atomicAdd(&rigidExchange[base],i32(round(reaction.x*1000000.0)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1000000.0)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1000000.0)));
    atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1000000.0)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1000000.0)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1000000.0)));
    let displacedWeight=wetFraction*solidFraction;
    atomicAdd(&rigidExchange[base+6u],i32(round(displacedWeight*65536.0)));
    atomicAdd(&rigidExchange[base+7u],i32(round(displacedWeight*ambientVelocity.x*10000.0)));atomicAdd(&rigidExchange[base+8u],i32(round(displacedWeight*ambientVelocity.y*10000.0)));atomicAdd(&rigidExchange[base+9u],i32(round(displacedWeight*ambientVelocity.z*10000.0)));
  }
  // Paper Sec 3.9.1 phi-s: inside a body the advected field is meaningless, so
  // blend it toward the (1-s)-weighted neighbor average. This is what lets the
  // body displace its water column instead of sealing a phantom plug of
  // liquid inside and carrying it around.
  var phiNext=phi;
  if(nearAnyBody(world)){
    let s=cellSolidFraction(id);
    if(s>0.0){
      var open=0.0;var openSum=0.0;var total=0.0;
      let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
      for(var index=0;index<6;index+=1){
        let np=clampCell(id+offsets[index]);
        let neighborVolume=volume(np);let neighborOpen=(1.0-cellSolidFraction(np))*(1.0-cellTerrainFraction(np));total+=neighborVolume;
        open+=neighborOpen;openSum+=neighborOpen*neighborVolume;
      }
      // A one-cell stencil diffuses a carried interior plug over several body
      // radii. Direct lateral open samples preserve the same local phi-s target
      // while making it follow a fast body through the interface in one step.
      if(coupledBody<12u&&i32(round(rigidBodies[coupledBody].positionShape.w))==0&&length(rigidBodies[coupledBody].linearVelocity.xyz)>0.25){
        let radius=max(rigidBodies[coupledBody].dimensions.w,0.0);let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
        let far=array<vec3i,4>(vec3i(-reach.x,0,0),vec3i(reach.x,0,0),vec3i(0,0,-reach.z),vec3i(0,0,reach.z));
        for(var index=0;index<4;index+=1){let np=id+far[index];if(valid(np)){let neighborOpen=(1.0-cellSolidFraction(np))*(1.0-cellTerrainFraction(np));open+=neighborOpen;openSum+=neighborOpen*volume(np);}}
      }
      let relaxTarget=select(total/6.0,openSum/max(open,1.0),open>0.0);
      phiNext=mix(phi,relaxTarget,s);
    }
  }
  // The nozzle mouth is an open boundary. Coupling the visual nozzle body
  // must not replace the prescribed reservoir velocity at that opening.
  v=applyInflowVelocity(id,v);
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(phiNext));
}

// Paper Sec 3.9.1 phi-s for the resident adaptive level set. While an adaptive
// projection owns the pressure solve the uniform pressure textures are idle,
// so this pass aliases them: pressureIn is a copy of the signed-distance field
// and pressureOut is the resident level-set texture itself.
@compute @workgroup_size(4,4,4)
fn relaxSolidPhi(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let phi=textureLoad(pressureIn,id,0).x;
  var result=phi;
  if(nearAnyBody(worldCell(id))){
    let s=cellSolidFraction(id);
    if(s>0.0){
      var open=0.0;var openSum=0.0;var total=0.0;var exteriorOpen=0.0;var exteriorSum=0.0;
      let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
      for(var index=0;index<6;index+=1){
        let np=clampCell(id+offsets[index]);
        let neighborPhi=textureLoad(pressureIn,np,0).x;let neighborOpen=(1.0-cellSolidFraction(np))*(1.0-cellTerrainFraction(np));total+=neighborPhi;
        open+=neighborOpen;openSum+=neighborOpen*neighborPhi;
        // Extend phi from the first genuinely open sample on each coordinate
        // ray. A one-cell relaxation takes many frames to cross a large solid
        // and leaves a newly submerged body falsely dry, under-reporting its
        // displaced volume. Six exterior samples establish the correct phase
        // throughout the solid in this pass while retaining a local fallback
        // for bodies wider than the bounded search.
        for(var step=1;step<=64;step+=1){
          let exterior=id+step*offsets[index];if(!valid(exterior)){break;}
          let exteriorWeight=(1.0-cellSolidFraction(exterior))*(1.0-cellTerrainFraction(exterior));
          if(exteriorWeight>0.5){exteriorOpen+=exteriorWeight;exteriorSum+=exteriorWeight*textureLoad(pressureIn,exterior,0).x;break;}
        }
      }
      let localTarget=select(total/6.0,openSum/max(open,1.0),open>0.0);
      let relaxTarget=select(localTarget,exteriorSum/max(exteriorOpen,1.0),exteriorOpen>0.0);
      result=mix(phi,relaxTarget,s);
    }
  }
  textureStore(pressureOut,id,vec4f(result));
}

// --- Density sharpening (Mass-Conserving Eulerian Liquid Simulation Sec 3.5,
// Eq 4-17 and Algorithm 2; docs/TALL_CELLS_PAPER.md Appendix B.3). Pass 1
// applies the local correction (Eq 17 keeps it non-positive: mass only moves
// from the air side to the liquid side); pass 2 returns the removed mass by
// tracing along the density gradient to the 0.5 iso-contour and depositing
// fixed-point trilinear weights; pass 3 folds the deposits back in.
fn cellInsideSolid(p:vec3i)->bool{
  if(cellInsideTerrain(p)){return true;}
  let bodyCount=u32(round(params.boundary.z));if(bodyCount==0u){return false;}
  let world=worldCell(p);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],world)){return true;}}
  return false;
}
fn sharpenDeltaRho(q:vec3i)->f32{
  let rho=volume(q);
  if(cellInsideSolid(q)){return 0.0;}
  let h=params.cellGravity.xyz;let deltaT=3.0*params.dimsDt.w;let tau=0.4;
  let sxp=-(rho-volume(q-vec3i(1,0,0)))*deltaT/h.x;let sxm=-(volume(q+vec3i(1,0,0))-rho)*deltaT/h.x;
  let syp=-(rho-volume(q-vec3i(0,1,0)))*deltaT/h.y;let sym=-(volume(q+vec3i(0,1,0))-rho)*deltaT/h.y;
  let szp=-(rho-volume(q-vec3i(0,0,1)))*deltaT/h.z;let szm=-(volume(q+vec3i(0,0,1))-rho)*deltaT/h.z;
  let gradPlus=sqrt(max(max(sxp,0.0)*max(sxp,0.0),min(sxm,0.0)*min(sxm,0.0))+max(max(syp,0.0)*max(syp,0.0),min(sym,0.0)*min(sym,0.0))+max(max(szp,0.0)*max(szp,0.0),min(szm,0.0)*min(szm,0.0)));
  let gradMinus=sqrt(max(min(sxp,0.0)*min(sxp,0.0),max(sxm,0.0)*max(sxm,0.0))+max(min(syp,0.0)*min(syp,0.0),max(sym,0.0)*max(sym,0.0))+max(min(szp,0.0)*min(szp,0.0),max(szm,0.0)*max(szm,0.0)));
  var maximumDifference=0.0;
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var index=0;index<6;index+=1){maximumDifference=max(maximumDifference,abs(rho-volume(q+offsets[index])));}
  let weight=(rho-0.5)*(rho-0.5)*(rho-0.5)*(1.0-min(1.0,maximumDifference/tau));
  var deltaRho=select(weight*gradMinus,weight*gradPlus,weight>=0.0);
  if(rho+deltaRho<0.0||rho<1e-5){deltaRho=-rho;}else if(rho>0.5){deltaRho=0.0;}
  return deltaRho;
}
@compute @workgroup_size(4,4,4)
fn sharpenCompute(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let stored=textureLoad(volumeIn,id,0).x;
  let deltaRho=sharpenDeltaRho(id);
  textureStore(volumeOut,id,vec4f(stored+deltaRho));
  textureStore(pressureOut,id,vec4f(deltaRho));
}
@compute @workgroup_size(4,4,4)
fn sharpenScatter(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let deltaRho=textureLoad(pressureIn,id,0).x;if(deltaRho>=0.0){return;}
  var p=vec3f(id)+vec3f(0.5);let maximumDistance=2.1;var travelled=0.0;let stepLength=0.5;
  for(var stepIndex=0;stepIndex<5;stepIndex+=1){
    if(sampleVolume(p)>=0.5||travelled>=maximumDistance){break;}
    let g=volumeGradient(vec3i(floor(p)));let magnitude=length(g);
    if(magnitude<1e-6){break;}
    let candidate=p+g/magnitude*stepLength;
    if(cellInsideSolid(vec3i(floor(candidate)))){break;}
    p=candidate;travelled+=stepLength;
  }
  // The paper assumes the 0.5 iso-contour lies within D cells of every
  // sharpened cell. In diffused low-density regions no contour exists
  // nearby, and depositing at the trace end concentrates fog at its local
  // maxima until free-floating droplets nucleate above the water. When the
  // trace fails to reach liquid, return the mass to its own cell instead.
  if(sampleVolume(p)<0.5){
    let dd=dims();let ownIndex=id.x+dd.x*(id.y+dd.y*id.z);
    atomicAdd(&sharpenDeposits[u32(ownIndex)],i32(round(-deltaRho*1048576.0)));return;
  }
  let anchor=p-vec3f(0.5);let cell=vec3i(floor(anchor));let f=fract(anchor);
  var weights=array<f32,8>();var indices=array<i32,8>();var total=0.0;
  let d=dims();
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let destination=cell+offset;
    var w=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    var index=-1;
    if(valid(destination)&&!cellInsideSolid(destination)){index=destination.x+d.x*(destination.y+d.y*destination.z);}else{w=0.0;}
    // Corners without remaining capacity are skipped so deposits cannot push
    // a cell past one, where the advection clamp would destroy the excess;
    // any residual overshoot drains through the correction divergence below.
    if(w>0.0&&volume(destination)>=1.0){w=0.0;}
    weights[corner]=w;indices[corner]=index;total+=w;
  }
  if(total<=1e-8){
    let ownIndex=id.x+d.x*(id.y+d.y*id.z);
    atomicAdd(&sharpenDeposits[u32(ownIndex)],i32(round(-deltaRho*1048576.0)));return;
  }
  for(var corner=0u;corner<8u;corner+=1u){
    if(weights[corner]<=0.0){continue;}
    atomicAdd(&sharpenDeposits[u32(indices[corner])],i32(round(-deltaRho*weights[corner]/total*1048576.0)));
  }
}
@compute @workgroup_size(4,4,4)
fn sharpenResolve(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let d=dims();let index=u32(id.x+d.x*(id.y+d.y*id.z));
  let deposit=f32(atomicLoad(&sharpenDeposits[index]))/1048576.0;
  textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x+deposit));
}
@compute @workgroup_size(4,4,4)
fn reduceDiagnostics(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!valid(id)){return;}let open=(1.0-cellSolidFraction(id))*(1.0-cellTerrainFraction(id));let represented=surfaceOccupancy(id)*open;let conservative=volume(id)*open;atomicAdd(&reductions[0],u32(represented*2048.0+0.5));if(surfaceLiquid(id)){atomicMax(&reductions[1],u32(id.x+1));}let speed=length(faceVelocity(id));atomicMax(&reductions[2],bitcast<u32>(speed));atomicAdd(&reductions[3],u32(clamp(conservative,0.0,8.0)*2048.0+0.5));}
`;

/** Clear every dense velocity scratch cell belonging to newly retired bricks. */
export const retiredBulkVelocityClearShader = /* wgsl */ `
struct BulkAtlasParams {
  dims: vec4u,
  brickDims: vec4u,
  tileGrid: vec4u,
  capacitySeed: vec4u,
  cell: vec4f,
}
@group(0) @binding(0) var velocityOut: texture_storage_3d<rgba32float,write>;
@group(0) @binding(1) var<storage,read> worklist: array<u32>;
@group(0) @binding(2) var<uniform> atlasParams: BulkAtlasParams;
@compute @workgroup_size(256)
fn clearRetiredVelocity(@builtin(global_invocation_id) gid:vec3u){
  let stream=gid.x+gid.y*worklist[5]*256u;
  let brickSize=atlasParams.dims.w;
  let brickVoxels=brickSize*brickSize*brickSize;
  let retiredIndex=stream/brickVoxels;
  if(retiredIndex>=worklist[4]){return;}
  let entry=16u+atlasParams.brickDims.w*2u+retiredIndex*2u;
  if(entry>=arrayLength(&worklist)){return;}
  let brickIndex=worklist[entry];
  if(brickIndex>=atlasParams.brickDims.w){return;}
  let b=atlasParams.brickDims;
  let brick=vec3u(brickIndex%b.x,(brickIndex/b.x)%b.y,brickIndex/(b.x*b.y));
  let localLinear=stream-retiredIndex*brickVoxels;
  let local=vec3u(localLinear%brickSize,(localLinear/brickSize)%brickSize,localLinear/(brickSize*brickSize));
  let cell=brick*brickSize+local;
  if(any(cell>=atlasParams.dims.xyz)){return;}
  textureStore(velocityOut,cell,vec4f(0.0));
}
`;

/**
 * Clear retired dense transport payloads without touching the permanent
 * one-texel zero shell. Sparse buildTransport writes only id+1, so the shell
 * remains at WebGPU's zero-initialized resource value for the texture's life.
 */
export const retiredBulkTransportClearShader = /* wgsl */ `
struct BulkAtlasParams {
  dims: vec4u,
  brickDims: vec4u,
  tileGrid: vec4u,
  capacitySeed: vec4u,
  cell: vec4f,
}
@group(0) @binding(0) var transportOut: texture_storage_3d<rgba16float,write>;
@group(0) @binding(1) var<storage,read> worklist: array<u32>;
@group(0) @binding(2) var<uniform> atlasParams: BulkAtlasParams;
@compute @workgroup_size(256)
fn clearRetiredTransport(@builtin(global_invocation_id) gid:vec3u){
  let stream=gid.x+gid.y*worklist[5]*256u;
  let brickSize=atlasParams.dims.w;
  let brickVoxels=brickSize*brickSize*brickSize;
  let retiredIndex=stream/brickVoxels;
  if(retiredIndex>=worklist[4]){return;}
  let entry=16u+atlasParams.brickDims.w*2u+retiredIndex*2u;
  if(entry>=arrayLength(&worklist)){return;}
  let brickIndex=worklist[entry];
  if(brickIndex>=atlasParams.brickDims.w){return;}
  let b=atlasParams.brickDims;
  let brick=vec3u(brickIndex%b.x,(brickIndex/b.x)%b.y,brickIndex/(b.x*b.y));
  let localLinear=stream-retiredIndex*brickVoxels;
  let local=vec3u(localLinear%brickSize,(localLinear/brickSize)%brickSize,localLinear/(brickSize*brickSize));
  let cell=brick*brickSize+local;
  if(any(cell>=atlasParams.dims.xyz)){return;}
  textureStore(transportOut,vec3i(cell)+vec3i(1),vec4f(0.0));
}
`;

/** Clear compatibility flux limits for cells in newly retired wet bricks. */
export const retiredBulkFluxScaleClearShader = /* wgsl */ `
struct BulkAtlasParams {
  dims: vec4u,
  brickDims: vec4u,
  tileGrid: vec4u,
  capacitySeed: vec4u,
  cell: vec4f,
}
@group(0) @binding(0) var fluxScalesOut: texture_storage_3d<rg32float,write>;
@group(0) @binding(1) var<storage,read> worklist: array<u32>;
@group(0) @binding(2) var<uniform> atlasParams: BulkAtlasParams;
@compute @workgroup_size(256)
fn clearRetiredFluxScales(@builtin(global_invocation_id) gid:vec3u){
  let stream=gid.x+gid.y*worklist[5]*256u;
  let brickSize=atlasParams.dims.w;
  let brickVoxels=brickSize*brickSize*brickSize;
  let retiredIndex=stream/brickVoxels;
  if(retiredIndex>=worklist[4]){return;}
  let entry=16u+atlasParams.brickDims.w*2u+retiredIndex*2u;
  if(entry>=arrayLength(&worklist)){return;}
  let brickIndex=worklist[entry];
  if(brickIndex>=atlasParams.brickDims.w){return;}
  let b=atlasParams.brickDims;
  let brick=vec3u(brickIndex%b.x,(brickIndex/b.x)%b.y,brickIndex/(b.x*b.y));
  let localLinear=stream-retiredIndex*brickVoxels;
  let local=vec3u(localLinear%brickSize,(localLinear/brickSize)%brickSize,localLinear/(brickSize*brickSize));
  let cell=brick*brickSize+local;
  if(any(cell>=atlasParams.dims.xyz)){return;}
  // Invalid-neighbor semantics are donor=0, receiver=1.
  textureStore(fluxScalesOut,cell,vec4f(0.0,1.0,0.0,0.0));
}
`;

export class WebGPUEulerianSolver {
  readonly info: GPUEulerianInfo;
  private readonly layout: TallCellLayout;
  private velocityA: GPUTexture; private velocityB: GPUTexture;private velocityC:GPUTexture;private velocityD:GPUTexture;
  private pressureA: GPUTexture; private pressureB: GPUTexture;
  private volumeA: GPUTexture; private volumeB: GPUTexture;private solidA:GPUTexture;private solidB:GPUTexture;
  private heightA: GPUTexture; private heightB: GPUTexture;
  private terrainTexture: GPUTexture;
  private params: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private planBindGroupLayout:GPUBindGroupLayout;
  private extrapolatePipeline!:GPUComputePipeline;private predictPipeline!:GPUComputePipeline;private reversePipeline!:GPUComputePipeline;private planSubstepsPipeline!:GPUComputePipeline;private transportPhiPipeline!:GPUComputePipeline;private clampPhiPipeline!:GPUComputePipeline;private reinitializePhiPipeline!:GPUComputePipeline;private advectPipeline!: GPUComputePipeline;private buildRhsPipeline!:GPUComputePipeline; private jacobiPipeline!: GPUComputePipeline; private projectPipeline!: GPUComputePipeline; private rigidPipeline!:GPUComputePipeline;private preReductionPipeline!:GPUComputePipeline;private reductionPipeline!:GPUComputePipeline;private planRemeshPipeline!:GPUComputePipeline;private smoothRemeshPipeline!:GPUComputePipeline;private remapPipeline!:GPUComputePipeline;
  private shaderModule:GPUShaderModule;private pipelineLayout:GPUPipelineLayout;private planPipelineLayout:GPUPipelineLayout;
  private planSubstepsGroup:GPUBindGroup;private extrapolateFirstGroup:GPUBindGroup;private extrapolateSecondGroup:GPUBindGroup;private extrapolateBackGroup:GPUBindGroup;private phiABGroup:GPUBindGroup;private phiBAGroup:GPUBindGroup;private predictGroup:GPUBindGroup;private reverseGroup:GPUBindGroup;private advectGroup: GPUBindGroup;private pressureRhsGroup:GPUBindGroup; private jacobiABGroup: GPUBindGroup; private jacobiBAGroup: GPUBindGroup; private projectGroup: GPUBindGroup;private rigidGroup:GPUBindGroup; private reductionGroup:GPUBindGroup;private planRemeshGroup:GPUBindGroup;private smoothRemeshFirstGroup:GPUBindGroup;private smoothRemeshSecondGroup:GPUBindGroup;private remapGroup:GPUBindGroup;
  private multigrid:TallCellMultigrid;
  private velocityHierarchy?:TallCellVelocityHierarchy;
  private reductionBuffer:GPUBuffer;private governorBuffer:GPUBuffer;private phiDispatchBuffer:GPUBuffer;private rigidBuffer:GPUBuffer;private rigidExchangeBuffer:GPUBuffer;private rigidSystem:WebGPURigidBodySystem;private nextColumnBases:GPUBuffer;private smoothedColumnBases:GPUBuffer;private querySet?:GPUQuerySet;private queryResolve?:GPUBuffer;
  private querySegments: GPUPhysicsTimestampSegment[]=[];private queryCount=0;
  private lastTime = 0;
  private lastFrameDt = 0;
  private readbackPending = false;
  private wallTimingPending = false;
  private performanceReadbacksEnabled = true;
  private validationChecked = false;
  private stepIndex = 0;
  private readonly inflowBoundary?: InflowGridBoundary;
  private readonly velocityTransport: GPUVelocityTransport;
  private readonly hierarchicalExtrapolation: boolean;
  private readonly volumeControl: boolean;
  private readonly pressureWarmStart:boolean;
  private readonly pressureDefectCorrection:boolean;
  private referenceLiquidVolumeCells = 0;
  private volumeCorrectionNormalSpeed = 0;

  constructor(private device: GPUDevice, readonly scene: SceneDescription, quality: GPUQuality, private onRigidLoads?: (loads: GPURigidLoad[]) => void, private readonly options:WebGPUEulerianSolverOptions={}) {
    this.layout=options.layoutOverride??createTallCellLayout(scene,quality,device.limits.maxTextureDimension3D,options.tallCellSettings);const {nx,packedNy,nz,fineNy}=this.layout;
    this.velocityTransport=options.velocityTransport??"maccormack";this.hierarchicalExtrapolation=options.hierarchicalExtrapolation??true;this.volumeControl=options.volumeControl??true;this.pressureWarmStart=options.pressureWarmStart??true;this.pressureDefectCorrection=options.pressureDefectCorrection??false;
    this.inflowBoundary=scene.fluid.inflow?createInflowGridBoundary(scene.fluid.inflow,scene.container,[nx,fineNy,nz]):undefined;
    const usage=GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
    const texture=(format: GPUTextureFormat)=>device.createTexture({size:[nx,packedNy,nz],dimension:"3d",format,usage});
    this.velocityA=texture("rgba32float"); this.velocityB=texture("rgba32float");this.velocityC=texture("rgba32float");this.velocityD=texture("rgba32float"); this.pressureA=texture("r32float"); this.pressureB=texture("r32float"); this.volumeA=texture("r32float"); this.volumeB=texture("r32float");this.solidA=texture("r32float");this.solidB=texture("r32float");
    this.heightA=device.createTexture({label:"Tall-cell column bases A",size:[nx,nz],format:"r32float",usage});this.heightB=device.createTexture({label:"Tall-cell column bases B",size:[nx,nz],format:"r32float",usage});
    this.terrainTexture=device.createTexture({label:"Tall-cell terrain heights",size:[nx,nz],format:"r32float",usage});
    this.params=device.createBuffer({size:144,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.reductionBuffer=device.createBuffer({size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.governorBuffer=device.createBuffer({label:"Tall-cell phi governor",size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.phiDispatchBuffer=device.createBuffer({label:"Tall-cell phi indirect dispatches",size:8*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
    device.queue.writeBuffer(this.governorBuffer,0,new Uint32Array(4));
    this.rigidExchangeBuffer=device.createBuffer({size:GPU_RIGID_EXCHANGE_BYTES,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.rigidSystem=new WebGPURigidBodySystem(device,scene,this.rigidExchangeBuffer,this.terrainTexture);this.rigidBuffer=this.rigidSystem.stateBuffer;this.rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
    if(device.features.has("timestamp-query")){this.querySet=device.createQuerySet({type:"timestamp",count:GPU_PHYSICS_TIMESTAMP_CAPACITY});this.queryResolve=device.createBuffer({size:GPU_PHYSICS_TIMESTAMP_CAPACITY*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});}
    this.nextColumnBases=device.createBuffer({label:"Next tall-cell column bases",size:nx*nz*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.smoothedColumnBases=device.createBuffer({label:"Smoothed tall-cell column bases",size:nx*nz*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    // The restricted solve expands one pressure stage into many multigrid
    // passes. Queue-completion timing remains reliable across browsers; the
    // timestamp ring is reserved for the single-pass uniform baseline.
    if(!options.deferPipelineCompilation)device.pushErrorScope("validation");this.shaderModule=device.createShaderModule({label:"Fluid Lab restricted tall-cell kernels",code:tallCellComputeShader});
    this.bindGroupLayout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}
      ,{binding:7,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}}
      ,{binding:8,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"2d"}}
      ,{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}
      ,{binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}
      ,{binding:11,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}
      ,{binding:12,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}
      ,{binding:13,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}}
      ,{binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}}
      ,{binding:15,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}}
      ,{binding:16,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}
      ,{binding:17,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}
      ,{binding:19,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}}
    ]});
    this.pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[this.bindGroupLayout]});
    this.planBindGroupLayout=device.createBindGroupLayout({entries:[
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:17,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:18,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}
    ]});
    this.planPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[this.planBindGroupLayout]});
    this.planSubstepsGroup=device.createBindGroup({layout:this.planBindGroupLayout,entries:[{binding:6,resource:{buffer:this.params}},{binding:17,resource:{buffer:this.governorBuffer}},{binding:18,resource:{buffer:this.phiDispatchBuffer}}]});
    if(!options.deferPipelineCompilation){this.createPipelinesSync();void device.popErrorScope().then(error=>{if(error)console.error(`Tall-cell pipeline creation: ${error.message}`);}).catch(()=>{ /* Device loss is handled by the renderer. */ });}
    if(this.hierarchicalExtrapolation)this.velocityHierarchy=new TallCellVelocityHierarchy(device,this.layout,this.velocityC,this.velocityD,this.heightA);
    this.multigrid=new TallCellMultigrid(device,this.layout,{pressureA:this.pressureA,pressureB:this.pressureB,volume:this.volumeA,solid:this.solidA,base:this.heightA,diagnostics:this.reductionBuffer},options.pressureCycles??8,options.deferPipelineCompilation);
    const pressureIterations=this.pressureDefectCorrection?2:1,cellSize=this.layout.cellSize_m,columnCount=nx*nz,allocatedBytes=this.layout.packedSampleCount*76+columnCount*12+128+16+8*16+12*80+GPU_RIGID_EXCHANGE_BYTES+this.multigrid.allocatedBytes;
    const primaryPressureSolver=this.pressureWarmStart?`warm ${this.multigrid.refinementCycles} V-cycles · ${this.multigrid.levelCount} levels · RBGS`:`1 full + ${this.multigrid.refinementCycles} V-cycles · ${this.multigrid.levelCount} levels · RBGS`;
    this.info={nx,ny:fineNy,nz,storedNy:packedNy,cellCount:this.layout.packedSampleCount,equivalentUniformCells:this.layout.equivalentUniformCellCount,compressionRatio:this.layout.compressionRatio,activeCompressionRatio:this.layout.activeCompressionRatio,activeSampleCount:this.layout.activeSampleCount,regularLayers:this.layout.settings.regularLayers,maximumNeighborDelta:this.layout.settings.maximumNeighborDelta,gridKind:"restricted-tall-cell",surfaceField:"levelset",cellSize_m:Math.min(cellSize.x,cellSize.y,cellSize.z),pressureIterations,pressureSolver:this.pressureDefectCorrection?`${primaryPressureSolver} + cold defect correction`:primaryPressureSolver,allocatedBytes,quality,encodedSteps:0,submittedTime_s:0,simulatedTime_s:0,completedTime_s:0,simulationLag_s:0};
    this.initializeVolume();
    this.extrapolateFirstGroup=this.group(this.velocityA,this.velocityD,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.extrapolateSecondGroup=this.group(this.velocityD,this.velocityC,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.extrapolateBackGroup=this.group(this.velocityC,this.velocityD,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.phiABGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.phiBAGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeB,this.volumeA,this.heightA,this.heightB);
    this.predictGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.reverseGroup=this.group(this.velocityB,this.velocityC,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.advectGroup=this.group(this.velocityA,this.velocityD,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB,this.velocityB,this.velocityC);
    this.pressureRhsGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.multigrid.fineRhs,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.jacobiABGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.jacobiBAGroup=this.group(this.velocityA,this.velocityB,this.pressureB,this.pressureA,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.projectGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.rigidGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.solidA,this.volumeA,this.volumeB,this.heightA,this.heightB,this.velocityA,this.velocityA,this.solidB);
    this.reductionGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.planRemeshGroup=this.group(this.velocityB,this.velocityA,this.pressureA,this.pressureB,this.volumeB,this.volumeA,this.heightA,this.heightB);
    this.smoothRemeshFirstGroup=this.group(this.velocityB,this.velocityA,this.pressureA,this.pressureB,this.volumeB,this.volumeA,this.heightA,this.heightB,this.velocityB,this.velocityB,this.solidA,this.nextColumnBases,this.smoothedColumnBases);
    this.smoothRemeshSecondGroup=this.group(this.velocityB,this.velocityA,this.pressureA,this.pressureB,this.volumeB,this.volumeA,this.heightA,this.heightB,this.velocityB,this.velocityB,this.solidA,this.smoothedColumnBases,this.nextColumnBases);
    this.remapGroup=this.group(this.velocityB,this.velocityA,this.pressureA,this.pressureB,this.volumeB,this.volumeA,this.heightA,this.heightB);
  }

  private pipelineDescriptor(entryPoint:string):GPUComputePipelineDescriptor{return{layout:entryPoint==="planSubsteps"?this.planPipelineLayout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  private createPipelinesSync(){const pipeline=(entryPoint:string)=>this.device.createComputePipeline(this.pipelineDescriptor(entryPoint));this.extrapolatePipeline=pipeline("extrapolateVelocity");this.predictPipeline=pipeline("predictVelocity");this.reversePipeline=pipeline("reverseVelocity");this.planSubstepsPipeline=pipeline("planSubsteps");this.transportPhiPipeline=pipeline("transportPhi");this.clampPhiPipeline=pipeline("clampPhi");this.reinitializePhiPipeline=pipeline("reinitializePhi");this.advectPipeline=pipeline(this.velocityTransport==="maccormack"?"finishAdvection":"finishSemiLagrangianAdvection");this.buildRhsPipeline=pipeline("buildPressureRhs");this.jacobiPipeline=pipeline("jacobi");this.projectPipeline=pipeline("project");this.rigidPipeline=pipeline("coupleRigid");this.preReductionPipeline=pipeline("reduceBeforeProjection");this.reductionPipeline=pipeline("reduceDiagnostics");this.planRemeshPipeline=pipeline("planRemesh");this.smoothRemeshPipeline=pipeline("smoothRemesh");this.remapPipeline=pipeline("remap");}
  static async createAsync(device:GPUDevice,scene:SceneDescription,quality:GPUQuality,onRigidLoads:((loads:GPURigidLoad[])=>void)|undefined,options:WebGPUEulerianSolverOptions,onProgress:(label:string,completed:number,total:number)=>void){const solver=new WebGPUEulerianSolver(device,scene,quality,onRigidLoads,{...options,deferPipelineCompilation:true});await solver.initializePipelines(onProgress);return solver;}
  private async initializePipelines(onProgress:(label:string,completed:number,total:number)=>void){
    const definitions=[
      ["Extrapolate velocity","extrapolateVelocity"],["Predict velocity","predictVelocity"],["Reverse velocity","reverseVelocity"],
      ["Transport level set","transportPhi"],["Clamp level set","clampPhi"],["Reinitialize level set","reinitializePhi"],
      ["Advect velocity",this.velocityTransport==="maccormack"?"finishAdvection":"finishSemiLagrangianAdvection"],
      ["Build pressure right-hand side","buildPressureRhs"],["Pressure relaxation","jacobi"],["Project velocity","project"],
      ["Couple rigid bodies","coupleRigid"],["Pre-projection diagnostics","reduceBeforeProjection"],["Fluid diagnostics","reduceDiagnostics"],
      ["Plan tall-cell remesh","planRemesh"],["Smooth remesh plan","smoothRemesh"],["Remap tall cells","remap"],
      ["Plan phi substeps","planSubsteps"]
    ] as const;
    const total=definitions.length+11,compiled:GPUComputePipeline[]=[];
    for(let index=0;index<definitions.length;index+=1){const [label,entryPoint]=definitions[index];onProgress(label,index,total);compiled.push(await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint)));onProgress(label,index+1,total);}
    this.extrapolatePipeline=compiled[0];this.predictPipeline=compiled[1];this.reversePipeline=compiled[2];this.transportPhiPipeline=compiled[3];this.clampPhiPipeline=compiled[4];this.reinitializePhiPipeline=compiled[5];this.advectPipeline=compiled[6];this.buildRhsPipeline=compiled[7];this.jacobiPipeline=compiled[8];this.projectPipeline=compiled[9];this.rigidPipeline=compiled[10];this.preReductionPipeline=compiled[11];this.reductionPipeline=compiled[12];this.planRemeshPipeline=compiled[13];this.smoothRemeshPipeline=compiled[14];this.remapPipeline=compiled[15];this.planSubstepsPipeline=compiled[16];
    await this.multigrid.initializePipelines(onProgress,definitions.length,total);
  }

  get volumeTexture(){return this.volumeA;}
  get rigidRenderBuffer(){return this.rigidSystem.renderBuffer;}
  get rigidMotionBuffer(){return this.rigidSystem.motionBuffer;}
  setSelectedRigidBody(index:number){this.rigidSystem.setSelectedIndex(index);}
  pickRigidBody(origin:RigidBodyState["position_m"],direction:RigidBodyState["position_m"]){return this.rigidSystem.pick(origin,direction);}
  get columnBaseTexture(){return this.heightA;}
  get velocityTexture(){return this.velocityA;}
  // velocityC is dead between finishAdvection's reverse-trace read and the
  // next step's extrapolation write, so it doubles as the pre-projection
  // snapshot captured right before the projection pass. Both textures are in
  // packed layout; readers must reconstruct the cubic field through the
  // column bases.
  get preProjectionVelocityTexture(){return this.velocityC;}
  private initializeVolume(){
    const {nx,nz,packedNy,initialPhi,columnBases,initialVolumeCellSum,referenceLiquidVolume_cells}=this.layout,c=this.scene.container,dam=damBreakFractions(c.fillFraction);const initiallyRepresented=referenceLiquidVolume_cells;this.referenceLiquidVolumeCells=initiallyRepresented*(this.options.referenceVolumeScale??1);const initialDrift=(initiallyRepresented-this.referenceLiquidVolumeCells)/Math.max(1,this.referenceLiquidVolumeCells);
    // This solver transports phi, so its represented-volume diagnostics and
    // controller reference use the smooth-Heaviside/fixed-point functional
    // from reduceDiagnostics. Keep initialVolumeCellSum as distinct binary
    // seed metadata; using it as the controller reference fabricated a ~6.6%
    // first-step loss on the 24x18x16 dam break before meaningful transport.
    this.info.initialVolumeCellSum=initialVolumeCellSum;this.info.volumeCellSum=initiallyRepresented;this.info.representedVolumeCellSum=initiallyRepresented;this.info.representedVolumeDrift=initialDrift;this.info.volumeDrift=initialDrift;this.info.rawVolumeDrift=initialDrift;this.info.maxSpeed_m_s=0;this.info.maxDivergence_s=0;this.info.maxDivergenceBefore_s=0;this.info.maxDivergenceAfter_s=0;this.info.maxAirSpeed_m_s=0;this.info.maxPressure_Pa=0;this.info.pressureResidual=0;this.info.pressureRelativeResidual=0;this.info.maxComponentCfl=0;this.info.highCflCellCount=0;this.info.nonFiniteCount=0;this.info.stabilityFlags=[];this.info.front_m=this.scene.fluid.initialCondition==="dam-break"?-c.width_m/2+dam.width*c.width_m:c.width_m/2;
    this.info.referenceLiquidVolume_cells=this.referenceLiquidVolumeCells;this.info.volumeCorrectionNormalSpeed_cells_s=0;
    const rowBytes=nx*4,padded=Math.ceil(rowBytes/256)*256,packed=new Uint8Array(padded*packedNy*nz),source=new Uint8Array(initialPhi.buffer,initialPhi.byteOffset,initialPhi.byteLength);
    for(let k=0;k<nz;k++)for(let j=0;j<packedNy;j++)packed.set(source.subarray(rowBytes*(j+packedNy*k),rowBytes*(j+packedNy*k+1)),padded*(j+packedNy*k));
    for(const texture of [this.volumeA,this.volumeB])this.device.queue.writeTexture({texture},packed,{bytesPerRow:padded,rowsPerImage:packedNy},{width:nx,height:packedNy,depthOrArrayLayers:nz});
    const basePacked=new Uint8Array(padded*nz),baseSource=new Uint8Array(columnBases.buffer,columnBases.byteOffset,columnBases.byteLength);
    for(let z=0;z<nz;z++)basePacked.set(baseSource.subarray(rowBytes*z,rowBytes*(z+1)),padded*z);
    for(const texture of [this.heightA,this.heightB])this.device.queue.writeTexture({texture},basePacked,{bytesPerRow:padded,rowsPerImage:nz},{width:nx,height:nz});
    // Static per-column terrain heights H_{i,k} in fine-cell units (paper
    // Fig. 2), consumed analytically by every kernel and re-stamped into the
    // packed solid texture each step by coupleRigid for the multigrid.
    const terrainHeights=terrainColumnHeights(this.scene,nx,nz),cellHeight=this.layout.cellSize_m.y;
    const terrainCells=new Float32Array(nx*nz);for(let index=0;index<terrainCells.length;index++)terrainCells[index]=terrainHeights[index]/cellHeight;
    const terrainPacked=new Uint8Array(padded*nz),terrainSource=new Uint8Array(terrainCells.buffer);
    for(let z=0;z<nz;z++)terrainPacked.set(terrainSource.subarray(rowBytes*z,rowBytes*(z+1)),padded*z);
    this.device.queue.writeTexture({texture:this.terrainTexture},terrainPacked,{bytesPerRow:padded,rowsPerImage:nz},{width:nx,height:nz});
  }

  private group(velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture,heightIn:GPUTexture,heightOut:GPUTexture,predictedVelocity:GPUTexture=velocityIn,reversedVelocity:GPUTexture=velocityIn,solidIn:GPUTexture=this.solidA,columnPlanIn:GPUBuffer=this.nextColumnBases,columnPlanOut:GPUBuffer=this.smoothedColumnBases){return this.device.createBindGroup({layout:this.bindGroupLayout,entries:[{binding:0,resource:velocityIn.createView()},{binding:1,resource:velocityOut.createView()},{binding:2,resource:pressureIn.createView()},{binding:3,resource:pressureOut.createView()},{binding:4,resource:volumeIn.createView()},{binding:5,resource:volumeOut.createView()},{binding:6,resource:{buffer:this.params}},{binding:7,resource:heightIn.createView()},{binding:8,resource:heightOut.createView()},{binding:9,resource:{buffer:this.reductionBuffer}},{binding:10,resource:{buffer:this.rigidBuffer}},{binding:11,resource:{buffer:this.rigidExchangeBuffer}},{binding:12,resource:{buffer:columnPlanIn}},{binding:13,resource:predictedVelocity.createView()},{binding:14,resource:reversedVelocity.createView()},{binding:15,resource:solidIn.createView()},{binding:16,resource:{buffer:columnPlanOut}},{binding:17,resource:{buffer:this.governorBuffer}},{binding:19,resource:this.terrainTexture.createView()}]});}
  private dispatch(pass: GPUComputePassEncoder,pipeline:GPUComputePipeline,group:GPUBindGroup){pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(this.info.nx/4),Math.ceil(this.info.storedNy/4),Math.ceil(this.info.nz/4));}
  private dispatchPhiIndirect(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,group:GPUBindGroup,slot:number){pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroupsIndirect(this.phiDispatchBuffer,slot*16);}
  private dispatchColumns(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,group:GPUBindGroup){pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(this.info.nx/8),Math.ceil(this.info.nz/8));}
  setPerformanceReadbacksEnabled(enabled:boolean){this.performanceReadbacksEnabled=enabled;if(!enabled)this.info.gpuStep_ms=undefined;}
  private timing(name:GPUPhysicsTimingField){if(!this.performanceReadbacksEnabled||!this.querySet||this.queryCount+2>GPU_PHYSICS_TIMESTAMP_CAPACITY)return undefined;const segment={name,start:this.queryCount++,end:this.queryCount++};this.querySegments.push(segment);return segment;}

  advanceTo(time_s:number,bodies:RigidBodyState[]=[]){
    const advance=planGPUAdvance(time_s,this.lastTime,this.scene.numerics.maxDt_s);if(!advance)return false;const delta=advance.dt_s;if(delta<1e-6){this.info.simulatedTime_s=this.lastTime;this.info.simulationLag_s=advance.lag_s;return true;}this.lastTime=advance.nextTime_s;this.info.submittedTime_s=this.lastTime;this.info.simulatedTime_s=this.lastTime;this.info.simulationLag_s=advance.lag_s;const c=this.scene.container,rho=this.scene.fluid.density_kg_m3,sigma=0;
    // The signed-distance interface retains adaptive transport subdivision,
    // while the expensive momentum, coupling, and pressure stages run once
    // for the CPU-planned frame delta. The absolute speed rail remains based
    // on maxDt_s, so changing this subdivision cannot relax that guard.
    this.lastFrameDt=delta;
    const activeBodies=bodies.slice(0,12);this.rigidSystem.syncBodies(activeBodies);
    const h=this.layout.cellSize_m,s=this.layout.settings,inflow=this.scene.fluid.inflow,outlet=this.inflowBoundary?.outletCenter_m,inflowStepStrength=inflow?averageInflowStrength(inflow,this.lastTime-delta,this.lastTime):0;if(this.inflowBoundary){const cellVolume=h.x*h.y*h.z;this.referenceLiquidVolumeCells+=this.inflowBoundary.flowRate_m3_s*inflowStepStrength*delta/cellVolume;this.info.referenceLiquidVolume_cells=this.referenceLiquidVolumeCells;}this.device.queue.writeBuffer(this.params,0,new Float32Array([this.info.nx,this.info.storedNy,this.info.nz,delta,h.x,h.y,h.z,this.scene.fluid.gravity_m_s2.y,c.width_m,c.height_m,c.depth_m,4*Math.min(h.x,h.y,h.z)/Math.max(this.scene.numerics.maxDt_s,1e-6),rho,this.scene.fluid.dynamicViscosity_Pa_s,0,this.volumeCorrectionNormalSpeed,sigma,c.fluidWallMode==="no-slip"?1:0,activeBodies.length,this.info.ny,s.regularLayers,s.liquidHalo,s.airHalo,s.maximumNeighborDelta,outlet?.x??0,outlet?.y??0,outlet?.z??0,inflow?.radius_m??0,inflow?.velocity_m_s.x??0,inflow?.velocity_m_s.y??0,inflow?.velocity_m_s.z??0,this.inflowBoundary?.apertureScale??0,inflowStepStrength,sceneHasTerrain(this.scene)?1:0,0,0]));
    this.querySegments=[];this.queryCount=0;if(!this.validationChecked)this.device.pushErrorScope("validation");const encoder=this.device.createCommandEncoder({label:"GPU fluid step"}),totalTiming=this.timing("total_ms");let stageCapture:PendingGPUStageCapture|undefined;const captureTexture=(stageKey:string,texture:GPUTexture)=>{if(stageCapture)return;stageCapture=encodeGPUStageTextureCapture({device:this.device,encoder,lane:"physics",stageKey,texture,dimension:"3d",dimensions:[this.info.nx,this.info.storedNy,this.info.nz],identity:{methodId:"tall-cell",sceneId:this.scene.sceneId,simulationTime_s:this.lastTime,step:this.stepIndex}});};if(totalTiming&&this.querySet){const pass=encoder.beginComputePass({timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:totalTiming.start}});pass.end();}{const plan=encoder.beginComputePass({label:"Plan phi substeps"});plan.setPipeline(this.planSubstepsPipeline);plan.setBindGroup(0,this.planSubstepsGroup);plan.dispatchWorkgroups(1);plan.end();}encoder.clearBuffer(this.rigidExchangeBuffer);encoder.clearBuffer(this.reductionBuffer);
    const preparationTiming=this.timing("preparation_ms"),reinitialize=this.stepIndex%10===0;
    if(this.velocityHierarchy){
      // The hierarchy fills the whole packed field. The fallback's eight-cell
      // halo also exceeds the four-cell full-frame speed rail.
      {const pass=encoder.beginComputePass(preparationTiming&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:preparationTiming.start}}:undefined);this.dispatch(pass,this.extrapolatePipeline,this.extrapolateFirstGroup);pass.end();}
      {const pass=encoder.beginComputePass();this.dispatch(pass,this.extrapolatePipeline,this.extrapolateSecondGroup);pass.end();}
      this.velocityHierarchy.encode(encoder,!reinitialize&&preparationTiming&&this.querySet?{querySet:this.querySet,endOfPassWriteIndex:preparationTiming.end}:undefined);
      encoder.copyTextureToTexture({texture:this.velocityD},{texture:this.velocityA},[this.info.nx,this.info.storedNy,this.info.nz]);
    }else{const passes=Math.max(2,s.airHalo);for(let passIndex=0;passIndex<passes;passIndex+=1){const first=passIndex===0,last=passIndex===passes-1&&!reinitialize;const pass=encoder.beginComputePass(preparationTiming&&this.querySet&&(first||last)?{timestampWrites:{querySet:this.querySet,...(first?{beginningOfPassWriteIndex:preparationTiming.start}:{}),...(last?{endOfPassWriteIndex:preparationTiming.end}:{})}}:undefined);if(passIndex===0)this.dispatch(pass,this.extrapolatePipeline,this.extrapolateFirstGroup);else if(passIndex%2===1)this.dispatch(pass,this.extrapolatePipeline,this.extrapolateSecondGroup);else this.dispatch(pass,this.extrapolatePipeline,this.extrapolateBackGroup);pass.end();}encoder.copyTextureToTexture({texture:passes%2===0?this.velocityC:this.velocityD},{texture:this.velocityA},[this.info.nx,this.info.storedNy,this.info.nz]);}
    if(reinitialize){const first=encoder.beginComputePass();this.dispatch(first,this.reinitializePhiPipeline,this.phiABGroup);first.end();const second=encoder.beginComputePass(preparationTiming&&this.querySet?{timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:preparationTiming.end}}:undefined);this.dispatch(second,this.reinitializePhiPipeline,this.phiBAGroup);second.end();}
    const advectionTiming=this.timing("advection_ms");
    for(let slot=0;slot<8;slot+=1){const transport=encoder.beginComputePass(slot===0&&advectionTiming&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:advectionTiming.start}}:undefined);this.dispatchPhiIndirect(transport,this.transportPhiPipeline,this.phiABGroup,slot);transport.end();const clamp=encoder.beginComputePass();this.dispatchPhiIndirect(clamp,this.clampPhiPipeline,this.phiBAGroup,slot);clamp.end();}
    {const pass=encoder.beginComputePass();this.dispatch(pass,this.predictPipeline,this.predictGroup);pass.end();if(this.velocityTransport==="maccormack"){const reverse=encoder.beginComputePass();this.dispatch(reverse,this.reversePipeline,this.reverseGroup);reverse.end();}const finish=encoder.beginComputePass(advectionTiming&&this.querySet?{timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:advectionTiming.end}}:undefined);this.dispatch(finish,this.advectPipeline,this.advectGroup);finish.end();encoder.copyTextureToTexture({texture:this.velocityD},{texture:this.velocityB},[this.info.nx,this.info.storedNy,this.info.nz]);}captureTexture("advection",this.velocityB);
    const remeshed=this.stepIndex>0&&this.stepIndex%s.remeshInterval===0;
    if(remeshed){const timing=this.timing("remeshing_ms"),extent:[number,number,number]=[this.info.nx,this.info.storedNy,this.info.nz];encoder.copyTextureToTexture({texture:this.volumeA},{texture:this.volumeB},extent);const plan=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start}}:undefined);this.dispatchColumns(plan,this.planRemeshPipeline,this.planRemeshGroup);plan.end();for(let passIndex=0;passIndex<8;passIndex+=1){const smooth=encoder.beginComputePass();this.dispatchColumns(smooth,this.smoothRemeshPipeline,passIndex%2===0?this.smoothRemeshFirstGroup:this.smoothRemeshSecondGroup);smooth.end();}const remap=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:timing.end}}:undefined);this.dispatch(remap,this.remapPipeline,this.remapGroup);remap.end();if(this.pressureWarmStart)encoder.copyTextureToTexture({texture:this.pressureB},{texture:this.pressureA},extent);encoder.copyTextureToTexture({texture:this.heightB},{texture:this.heightA},[this.info.nx,this.info.nz,1]);}else{encoder.copyTextureToTexture({texture:this.velocityB},{texture:this.velocityA},[this.info.nx,this.info.storedNy,this.info.nz]);}
    {const timing=this.timing("rigidCoupling_ms");const pass=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start,endOfPassWriteIndex:timing.end}}:undefined);this.dispatch(pass,this.rigidPipeline,this.rigidGroup);pass.end();encoder.copyTextureToTexture({texture:this.velocityB},{texture:this.velocityA},[this.info.nx,this.info.storedNy,this.info.nz]);encoder.copyTextureToTexture({texture:this.volumeB},{texture:this.volumeA},[this.info.nx,this.info.storedNy,this.info.nz]);}
    {const pass=encoder.beginComputePass();this.dispatch(pass,this.preReductionPipeline,this.reductionGroup);pass.end();}
    {const timing=this.timing("pressure_ms");const rhs=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start}}:undefined);this.dispatch(rhs,this.buildRhsPipeline,this.pressureRhsGroup);rhs.end();this.multigrid.encode(encoder,{warmStart:this.pressureWarmStart&&this.stepIndex>0,topologyChanged:this.stepIndex===0||remeshed});if(timing&&this.querySet){const end=encoder.beginComputePass({timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:timing.end}});end.end();}}
    {const timing=this.timing("projection_ms");encoder.copyTextureToTexture({texture:this.velocityA},{texture:this.velocityC},[this.info.nx,this.info.storedNy,this.info.nz]);const pass=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start,endOfPassWriteIndex:timing.end}}:undefined);this.dispatch(pass,this.projectPipeline,this.projectGroup);pass.end();encoder.copyTextureToTexture({texture:this.velocityB},{texture:this.velocityA},[this.info.nx,this.info.storedNy,this.info.nz]);encoder.copyTextureToTexture({texture:this.volumeB},{texture:this.volumeA},[this.info.nx,this.info.storedNy,this.info.nz]);}
    if(this.pressureDefectCorrection){const pressureTiming=this.timing("pressure_ms");const rhs=encoder.beginComputePass(pressureTiming&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:pressureTiming.start}}:undefined);this.dispatch(rhs,this.buildRhsPipeline,this.pressureRhsGroup);rhs.end();this.multigrid.encode(encoder,{warmStart:false,topologyChanged:false});if(pressureTiming&&this.querySet){const end=encoder.beginComputePass({timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:pressureTiming.end}});end.end();}const projectionTiming=this.timing("projection_ms");const pass=encoder.beginComputePass(projectionTiming&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:projectionTiming.start,endOfPassWriteIndex:projectionTiming.end}}:undefined);this.dispatch(pass,this.projectPipeline,this.projectGroup);pass.end();encoder.copyTextureToTexture({texture:this.velocityB},{texture:this.velocityA},[this.info.nx,this.info.storedNy,this.info.nz]);encoder.copyTextureToTexture({texture:this.volumeB},{texture:this.volumeA},[this.info.nx,this.info.storedNy,this.info.nz]);}captureTexture("pressure",this.pressureA);captureTexture("projection",this.velocityA);
    if(activeBodies.length>0){const cellVolume=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);this.rigidSystem.encode(encoder,delta,cellVolume,1,h.y);}
    this.stepIndex+=1;
    {const timing=this.timing("diagnostics_ms");const pass=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start,endOfPassWriteIndex:timing.end}}:undefined);this.dispatch(pass,this.reductionPipeline,this.reductionGroup);pass.end();}if(totalTiming&&this.querySet){const pass=encoder.beginComputePass({timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:totalTiming.end}});pass.end();}if(this.querySet&&this.queryResolve&&this.queryCount>0)encoder.resolveQuerySet(this.querySet,0,this.queryCount,this.queryResolve,0);
    const submittedAt=performance.now();this.device.queue.submit([encoder.finish()]);stageCapture?.afterSubmit();if(this.performanceReadbacksEnabled&&!this.wallTimingPending){this.wallTimingPending=true;void this.device.queue.onSubmittedWorkDone().then(()=>{this.info.gpuQueueWall_ms=performance.now()-submittedAt;this.info.gpuQueueSimulation_s=delta;}).catch(()=>{ /* Device loss is handled by the renderer. */ }).finally(()=>{this.wallTimingPending=false;});}
    if(!this.validationChecked){this.validationChecked=true;void this.device.popErrorScope().then(error=>{if(error)console.error(`GPU fluid validation: ${error.message}`);}).catch(()=>{ /* Device loss is handled by the renderer. */ });}return true;
  }

  async readStats(){
    if(!this.performanceReadbacksEnabled||this.stepIndex===0)return this.info;
    if(this.readbackPending)return this.info;
    this.readbackPending=true;
    const diagnosticBytes=128,governorBytes=16,queryOffset=diagnosticBytes+governorBytes,querySegments=[...this.querySegments],queryBytes=this.queryResolve?this.queryCount*8:0;
    const buffer=this.device.createBuffer({size:queryOffset+queryBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.reductionBuffer,0,buffer,0,diagnosticBytes);
    encoder.copyBufferToBuffer(this.governorBuffer,0,buffer,diagnosticBytes,governorBytes);
    if(this.queryResolve&&queryBytes>0)encoder.copyBufferToBuffer(this.queryResolve,0,buffer,queryOffset,queryBytes);
    this.device.queue.submit([encoder.finish()]);
    try{
      await buffer.mapAsync(GPUMapMode.READ);
      const words=new Uint32Array(buffer.getMappedRange(0,diagnosticBytes)),governorWords=new Uint32Array(buffer.getMappedRange(diagnosticBytes,governorBytes)),reference=Math.max(1,this.referenceLiquidVolumeCells);
      const decodePositiveFloat=(word:number)=>new Float32Array(new Uint32Array([word]).buffer)[0];
      const location=(slot:number):GPUFieldLocation=>({x:words[slot],y:words[slot+1],z:words[slot+2]});
      this.info.volumeCellSum=words[0]/256;
      this.info.volumeDrift=(this.info.volumeCellSum-reference)/reference;
      this.info.rawVolumeDrift=this.info.volumeDrift;
      this.info.representedVolumeCellSum=this.info.volumeCellSum;
      this.info.representedVolumeDrift=this.info.volumeDrift;
      this.info.phiInterfaceCellCount=words[7]/256;
      this.volumeCorrectionNormalSpeed=this.volumeControl?Math.max(-30,Math.min(30,0.5*(reference-this.info.volumeCellSum)/Math.max(this.info.phiInterfaceCellCount,1)/(1/30))):0;
      this.info.volumeCorrectionNormalSpeed_cells_s=this.volumeCorrectionNormalSpeed;
      this.info.front_m=-this.scene.container.width_m/2+words[1]*this.scene.container.width_m/this.info.nx;
      this.info.maxSpeed_m_s=decodePositiveFloat(words[2]);
      this.info.maximumTallCellHeight=words[3];
      this.info.maxDivergenceBefore_s=decodePositiveFloat(words[4]);
      this.info.maxDivergenceAfter_s=decodePositiveFloat(words[5]);
      this.info.maxDivergence_s=this.info.maxDivergenceAfter_s;
      this.info.projectionDivergenceRatio=this.info.maxDivergenceAfter_s/Math.max(this.info.maxDivergenceBefore_s,1e-30);
      this.info.maxAirSpeed_m_s=decodePositiveFloat(words[6]);
      this.info.maxSpeedLocation=location(8);
      this.info.maxDivergenceBeforeLocation=location(11);
      this.info.maxDivergenceAfterLocation=location(14);
      this.info.maxAirSpeedLocation=location(17);
      this.info.nonFiniteCount=words[20];
      this.info.maxPressure_Pa=decodePositiveFloat(words[21]);
      this.info.pressureResidual=decodePositiveFloat(words[22]);
      const rhsMaximum=decodePositiveFloat(words[23]);
      this.info.pressureRelativeResidual=this.info.pressureResidual/Math.max(rhsMaximum,1e-30);
      this.info.maxPressureLocation=location(24);
      this.info.maxPressureResidualLocation=location(27);
      this.info.maxComponentCfl=decodePositiveFloat(words[30]);
      this.info.highCflCellCount=words[31];
      this.info.lastSubsteps=governorWords[1];this.info.lastDt_s=decodePositiveFloat(governorWords[2]);this.info.encodedSteps=governorWords[3];
      const measuredFrameDt=(this.info.lastDt_s??0)*(this.info.lastSubsteps??1);this.info.stabilityFlags=classifyTallCellStability({nonFiniteCount:this.info.nonFiniteCount,pressureRelativeResidual:this.info.pressureRelativeResidual,maxComponentCfl:this.info.maxComponentCfl,highCflCellCount:this.info.highCflCellCount,maxDivergenceBefore_s:this.info.maxDivergenceBefore_s,maxDivergenceAfter_s:this.info.maxDivergenceAfter_s,dt_s:measuredFrameDt||this.lastFrameDt});
      if(queryBytes>0){
        const times=new BigUint64Array(buffer.getMappedRange(queryOffset,queryBytes));
        const stageByField:Partial<Record<GPUPhysicsTimingField,GPUPhysicsStageId>>={preparation_ms:"preparation",layerConstruction_ms:"topology",advection_ms:"advection",conditioning_ms:"conditioning",remeshing_ms:"remeshing",pressure_ms:"pressure",powerAssembly_ms:"powerAssembly",pressureSolve_ms:"pressureSolve",projection_ms:"projection",powerProjection_ms:"powerProjection",velocityProjection_ms:"velocityProjection",faceBand_ms:"faceBand",faceMarch_ms:"faceMarch",powerPublication_ms:"powerPublication",extrapolation_ms:"extrapolation",materialization_ms:"materialization",surfaceUpdate_ms:"surfaceUpdate",fineTopology_ms:"fineTopology",fineTransport_ms:"fineTransport",fineRedistance_ms:"fineRedistance",rigidCoupling_ms:"rigidCoupling",spray_ms:"spray",fluidResidency_ms:"fluidResidency",sparsePublication_ms:"sparsePublication",diagnostics_ms:"diagnostics"};
        const activeStages=[...new Set(querySegments.map(segment=>stageByField[segment.name]).filter((stage):stage is GPUPhysicsStageId=>Boolean(stage)))];
        const timings=emptyGPUPhysicsTimings(activeStages);
        const decoded=decodeGPUPhysicsTimestampSegments(times,querySegments);
        for(const name of new Set(querySegments.map(segment=>segment.name)))timings[name]=decoded[name]??0;
        const categorized=categorizedGPUPhysicsTime_ms(timings);timings.total_ms=Math.max(timings.total_ms,categorized);timings.overhead_ms=Math.max(0,timings.total_ms-categorized);this.info.gpuTimings=timings;this.info.gpuStep_ms=timings.total_ms;
      }
      buffer.unmap();
    }catch(error){console.error("Tall-cell diagnostics readback failed",error);}finally{buffer.destroy();this.readbackPending=false;}
    return this.info;
  }

  destroy(){this.multigrid.destroy();this.velocityHierarchy?.destroy();for(const t of [this.velocityA,this.velocityB,this.velocityC,this.velocityD,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.solidA,this.solidB,this.heightA,this.heightB,this.terrainTexture])t.destroy();this.params.destroy();this.reductionBuffer.destroy();this.governorBuffer.destroy();this.phiDispatchBuffer.destroy();this.rigidSystem.destroy();this.rigidExchangeBuffer.destroy();this.nextColumnBases.destroy();this.smoothedColumnBases.destroy();this.querySet?.destroy();this.queryResolve?.destroy();}
}
