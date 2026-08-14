import type { GPUInitializationTask } from "../../core/gpu-initialization";
import type { WebGPUFineLevelSetBrickSource } from "../../core/levelset-consumer-abi";
import type { SceneDescription } from "../../core/model";
import type { GPUFluidBrickResidency } from "../../core/webgpu-fluid-brick-residency";
import type { PassBroker } from "../../core/webgpu-pass-broker";
import type { SurfaceInflowState, WebGPUQuadtreeSurfaceState } from "./surface-state";
import type { OctreeCoarseBackend, OctreeCoarseDynamicsConfiguration } from "./octree-coarse-backend";
import type { OctreeLeafFrontierAllocationPlan } from "./octree-arena-allocation";
import type { OctreePressureCapacityPlan } from "./octree-pressure-capacity";
import type {
  OctreeAllocationProgress,
  OctreeProjectionResources,
  OctreeSemanticBoundary,
} from "./octree-projection-contract";
import type { OctreeProjectionLane } from "./octree-projection.wgsl";
// Type-only, and therefore erased: the engine owns the failure layout that
// names these regions, and a lane may only refer to them by name.
import type { OctreeFrontierFailureRegion } from "./webgpu-octree";
import type { OctreeRuntimeDials } from "./octree-runtime-dials";
import type { OctreeSolveTailPolicy } from "./octree-solve-tail-policy";
import type { WebGPUOctreeCoarseSummary } from "./webgpu-octree-coarse-summary";
import type {
  WebGPUFineLevelSetBricks,
} from "./webgpu-octree-fine-levelset-bricks";
import type { WebGPUFineLevelSetRedistance } from "./webgpu-octree-fine-levelset-redistance";
import type { WebGPUFineLevelSetRigidCarve } from "./webgpu-octree-fine-levelset-rigid-carve";
import type { WebGPUFineLevelSetSummaries } from "./webgpu-octree-fine-levelset-summary";
import type { WebGPUFineLevelSetTopology } from "./webgpu-octree-fine-levelset-topology";
import type {
  FineLevelSetTransportTopologyDelta,
  WebGPUFineLevelSetTransport,
} from "./webgpu-octree-fine-levelset-transport";
import type { WebGPUFineLevelSetVolumeCorrection } from "./webgpu-octree-fine-levelset-volume";
import type { WebGPUOctreeFineSeedAdapter } from "./webgpu-octree-fine-seed-adapter";
import type {
  OctreeOwnerLeafSize,
  WebGPUOctreeSimulationOwnerPages,
} from "./webgpu-octree-owner-pages";
import type { OctreeWorkAccounting } from "./webgpu-octree-work-accounting";

/**
 * One counter blob for the whole projection.
 *
 * Both coarse dynamics lanes contribute to it and the solver publishes it
 * verbatim, so it is engine-owned state a lane writes into rather than
 * something either lane may own: a lane that kept its own copy would leave
 * the other lane's fields at their constructor defaults and the performance
 * strip would read a live number as zero.
 */
export interface OctreeProjectionTelemetry {
  leafCount: number;
  pressureSampleCount: number;
  liquidDofCount: number;
  faceCount: number;
  mlsProjectionRowCount: number;
  tallSegmentCount: number;
  ghostFaceCount: number;
  maximumNeighborRatio: number;
  maximumFluidScale: number;
  compressionRatio: number;
  allocatedBytes: number;
  pressureIterationsUsed: number;
  pressureIterationBudget: number;
  pressureIterationHardBudget: number;
  pressureConverged?: boolean;
  pressureRowCapacity: number;
  pressureRequiredRows?: number;
  pressureCapacityOverflow?: boolean;
  frontierListCapacity: number;
  frontierRequiredLeaves?: number;
  frontierCapacityOverflow?: boolean;
  velocityClampCount: number;
  factorLevelCount: number;
  multigridLevelCount: number;
  multigridCoarsestDofs: number;
  topologyReadbackBytes: number;
  topologyReused: boolean;
  topologyReuseCount: number;
  /** Host-scheduled cadence skips. These omit the entire candidate graph. */
  topologyCadenceSkipCount: number;
  /** Distinct GPU epochs whose row-identity receipt reported exact reuse.
   * This currently observes, but does not yet elide, the cadence-one graph
   * candidate construction. */
  topologyExactIdentityCount: number;
  powerDiagramReady: boolean;
  powerDiagramAuthoritative: boolean;
  powerDiagramAllocatedBytes: number;
  globalFineLevelSetAllocatedBytes: number;
  globalFineLevelSetResidentBrickCapacity: number;
  globalFineLevelSetLogicalBrickCount: number;
  globalFineTransportQueryCapacity: number;
  globalFineTransportChunkCapacity: number;
  globalFineTransportChunkCount: number;
  globalFineTransportSegmentCount: number;
  globalFineTransportEncodedPasses: number;
  globalFineTransportPrepassScratchBytes: number;
  globalFineTransportVertexScratchBytes: number;
}

/**
 * The fine-band transport stage, named by what the shared publication path
 * needs from it rather than by which lane built it.
 *
 * Power's `WebGPUFineLevelSetTransport` and Losasso's direct face transport
 * take different encode options and bind different velocity authorities, so
 * only the published control and the compact page-delta producer are common.
 * Encoding one is a lane hook (`encodeFineTransport`); consuming its receipt
 * is not.
 */
export interface OctreeFineLevelSetTransportStage {
  readonly control: GPUBuffer;
  /** Present only on the staged Power transport; the work-accounting view
   * tests for it rather than for a concrete class. */
  readonly governor?: GPUBuffer;
  readonly topologyDelta: FineLevelSetTransportTopologyDelta;
  readonly plan: { readonly allocatedBytes: number };
  initializePipelines(): Promise<void>;
  destroy(): void;
}

/**
 * The shared topology engine, as its coarse dynamics lane sees it.
 *
 * This is deliberately a wide surface: the engine owns the topology,
 * frontier, owner pages, pressure banks, bind groups and the whole fine-band
 * A/B ladder, and a lane's allocation stage has to reach all of it to build
 * its own authority against those exact buffer identities. What the interface
 * buys is that the reach is *named* — every member here is a documented
 * engine/lane seam, and a lane cannot silently start depending on engine
 * internals that are not listed.
 */
export interface OctreeTopologyEngine {
  readonly device: GPUDevice;
  readonly scene: SceneDescription;
  readonly dims: { nx: number; ny: number; nz: number };
  readonly resources: OctreeProjectionResources;
  readonly deferPipelineCompilation: boolean;
  readonly coarseDynamics: OctreeCoarseDynamicsConfiguration;
  readonly info: OctreeProjectionTelemetry;
  readonly workAccounting: OctreeWorkAccounting;

  // Topology, frontier and pressure identities.
  readonly ownerPages: WebGPUOctreeSimulationOwnerPages;
  readonly topologyResidency: GPUFluidBrickResidency;
  readonly surfaceState: WebGPUQuadtreeSurfaceState;
  readonly pressureA: GPUBuffer;
  readonly pressureB: GPUBuffer;
  readonly compaction: GPUBuffer;
  readonly leafHeaders: GPUBuffer;
  readonly candidateLeafHeaders: GPUBuffer;
  readonly candidatePressure: GPUBuffer;
  readonly leafFrontier: GPUBuffer;
  readonly solidCells: GPUBuffer;
  readonly params: GPUBuffer;
  readonly topologyCandidateDispatch: GPUBuffer;
  /** Solve feedback staging, copied inside the lane's own solve encoder. */
  readonly solveStats: GPUBuffer;
  readonly compactionByteLength: number;
  readonly unpublishedFineSummaryDirectory: GPUBuffer;
  readonly powerRowDelta: {
    readonly rows: GPUBuffer;
    readonly rowCapacity: number;
    readonly controlOffsetWords: number;
    readonly newToOldOffsetWords: number;
    readonly oldToNewOffsetWords: number;
    readonly dirtyRowsOffsetWords: number;
    readonly affectedRowsOffsetWords: number;
  };

  // Allocation plans and sizing law.
  readonly pressureCapacity: OctreePressureCapacityPlan;
  readonly frontierAllocation: OctreeLeafFrontierAllocationPlan;
  readonly solveTailPolicy: OctreeSolveTailPolicy;
  readonly maxLeafSize: 2 | 4 | 8 | 16 | 32;
  readonly topologyMaximumLeafSize: OctreeOwnerLeafSize;
  readonly coarseOnlySurfaceTracking: boolean;
  readonly analyticSparseBootstrap: boolean;
  /** Set by `destroy`; a lane initialization task racing teardown must fail closed. */
  readonly powerLifecycleDisposed: boolean;
  readonly fineLevelSetBandCells: number;
  readonly interfaceBandCellsEffective: number;
  readonly surfaceGradingLayersEffective: number;
  readonly compactionAllocationRowDeltaScratchOffsetBytes: number;
  readonly dirtyFailureOffsetBytes: number;
  readonly dirtyAuthorityStateOffsetBytes: number;
  readonly frontierPublicationOffsetBytes: number;
  readonly topologyTileChangeFlagsOffsetBytes: number;
  readonly topologyTileChangeFlagsByteLength: number;

  // The fine-band ladder. Both lanes fill these slots from their own coarse
  // authority, so they are engine state a lane writes rather than lane state.
  readonly globalFineLevelSet?: WebGPUFineLevelSetBricks;
  readonly globalFineSummaries?: WebGPUFineLevelSetSummaries;
  readonly fineSeedAdapter?: WebGPUOctreeFineSeedAdapter;
  globalFineSourceA?: WebGPUFineLevelSetBrickSource;
  globalFineSourceB?: WebGPUFineLevelSetBrickSource;
  globalFineTopologyAB?: WebGPUFineLevelSetTopology;
  globalFineTopologyBA?: WebGPUFineLevelSetTopology;
  globalFineRedistanceA?: WebGPUFineLevelSetRedistance;
  globalFineRedistanceB?: WebGPUFineLevelSetRedistance;
  globalFineVolumeA?: WebGPUFineLevelSetVolumeCorrection;
  globalFineVolumeB?: WebGPUFineLevelSetVolumeCorrection;
  globalFineRigidCarveA?: WebGPUFineLevelSetRigidCarve;
  globalFineRigidCarveB?: WebGPUFineLevelSetRigidCarve;
  globalFineTransportA?: WebGPUFineLevelSetTransport;
  globalFineTransportB?: WebGPUFineLevelSetTransport;
  coarseOnlySummary?: WebGPUOctreeCoarseSummary;
  globalFineCurrentIsA: boolean;
  globalFinePublishedIsA: boolean;
  globalFineBootstrapped: boolean;

  // Step state the lanes read and advance.
  latestPressureInA: boolean;
  /** Inflow cells staged by the host, in cubic metres, awaiting a volume
   * authority that can absorb them. The lane that owns the surface clears it. */
  pendingSurfaceReferenceVolume_m3: number;
  powerTimestep_s: number;
  powerAdvancingPressureSteps: number;
  dynamicCouplingBodyCount: number;
  surfaceInflow?: SurfaceInflowState;
  activePowerGeneration: number;
  candidatePowerGeneration: number;
  adaptiveSurfaceGeneration: number;
  topologyReusePending: boolean;
  lastObservedExactTopologyReuseEpoch: number;

  // Bind groups over binding 15, whose ABI is whichever lane published it.
  groups: { ab: GPUBindGroup; ba: GPUBindGroup };
  candidateRowGroups: { fromA: GPUBindGroup; fromB: GPUBindGroup };
  fineSummarySizingGroup: GPUBindGroup;
  topologyDecisionGroup?: GPUBindGroup;
  readonly diagnosticGroups?: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
  readonly materializePipeline?: GPUComputePipeline;
  /** The lane's solve control, republished whenever its authority is rebuilt. */
  pressureSolverControl: GPUBuffer;

  createProjectionGroup(
    pressureIn: GPUBuffer | GPUBufferBinding,
    pressureOut: GPUBuffer | GPUBufferBinding,
    binding15Override?: GPUBuffer,
    leafHeadersOverride?: GPUBuffer,
  ): GPUBindGroup;
  /** Shared fine-band settlement; both lanes call it from `encodeAdvance`. */
  encodePendingFineSettlement(
    encoder: GPUCommandEncoder,
    productionBoundary?: OctreeSemanticBoundary,
  ): GPUCommandEncoder;
  encodeOverlayMaterialization(encoder: GPUCommandEncoder, pressureInA?: boolean): boolean;
  pressureHardIterationCeiling(): number;
  /** Re-emit the uniform parameter block after a live dial changed the band. */
  writeParams(): void;
  /** Fold a solver control block into the shared iteration/residual telemetry. */
  applyMGPCGDiagnostics(words: Uint32Array): void;
  /** Residual triple the engine publishes beside `info`; a lane clears it when
   * its own receipt supersedes the shared reduction. */
  residualRms?: number;
  initialResidualRms?: number;
  relativeResidual?: number;
  readOwnerPageControl(): Promise<readonly number[]>;
  /** Admit the scalar generation proven by a coherent adaptive step receipt. */
  applyAdaptiveSurfaceGenerationReceipt(generation: number): void;
}

/**
 * Read-only buffer identities a lane publishes for the QA/harness surface.
 *
 * The engine re-exposes these verbatim as its own getters so that neither the
 * solver shell nor the smoke harness has to know which lane is running: a
 * configuration that never built the authority simply leaves the slot absent,
 * which is exactly what those readers already test for.
 */
export interface OctreeLaneDebugSources {
  /**
   * The pressure bank this lane's own solve published, when it keeps one.
   *
   * A lane with frame-local pressure views owns which of them is authoritative
   * this step; the engine's A/B latch names the bank only for lanes that solve
   * into the shared banks directly.
   */
  readonly pressureFrameView?: GPUBufferBinding;
  /** The adaptive nodal velocity/phi pair, for lanes that publish a graph. */
  readonly adaptiveVelocity?: OctreeLaneAdaptiveVelocityDebug;
  readonly topologyEpochState?: GPUBuffer;
  readonly airSupportScratch?: GPUBuffer;
  readonly spgridLevelDelta?: GPUBuffer;
  readonly structuredVelocityControl?: GPUBuffer;
  readonly structuredBoundaryControl?: GPUBuffer;
  readonly structuredRowVelocities?: GPUBuffer;
  readonly structuredAuthority?: GPUBuffer;
  readonly structuredWorksets?: GPUBuffer;
  readonly structuredProjectionEnergyStats?: GPUBuffer;
  readonly powerDescriptorControl?: GPUBuffer;
  readonly powerTopologyControl?: GPUBuffer;
  readonly powerDescriptorRows?: GPUBuffer;
  readonly powerTopologyMetrics?: GPUBuffer;
  readonly powerCatalogEntryHeaders?: GPUBuffer;
  readonly powerCatalogFaces?: GPUBuffer;
  readonly globalFineCoarseLevelSetControl?: GPUBuffer;
  readonly globalFineRestrictionControl?: GPUBuffer;
  readonly losassoAuthorityControl?: GPUBuffer;
  readonly losassoCoarsePhiControl?: GPUBuffer;
  readonly losassoExtensionControl?: GPUBuffer;
  readonly losassoAdaptiveAcceptedGraphControl?: GPUBuffer;
  readonly losassoAdaptiveCandidateGraphControl?: GPUBuffer;
  readonly losassoAdaptivePhiControl?: GPUBuffer;
  readonly losassoAdaptivePhiReceipts?: GPUBuffer;
  readonly losassoAdaptiveVelocityReceipts?: GPUBuffer;
  readonly losassoAdaptiveRendererDirectory?: GPUBuffer;
  readonly losassoCandidateAuthorityControl?: GPUBuffer;
  readonly losassoAdaptiveMassControl?: GPUBuffer;
  readonly losassoAdaptiveMassReceipts?: GPUBuffer;
  readonly losassoAdaptiveCandidateMassControl?: GPUBuffer;
  readonly losassoAdaptiveCandidateMassReceipts?: GPUBuffer;
  readonly losassoCandidateVelocityMigrationReceipt?: GPUBuffer;
  readonly rigidCouplingDiagnosticBuffer?: GPUBuffer;
  readonly rigidBoundaryRefreshDiagnosticBuffer?: GPUBuffer;
}

/** Rejection-only velocity forensics over the lane's own face authority. */
export interface OctreeLaneVelocityDebug {
  readonly control: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  /** The band's own metric row and capacity, when an extension band exists. */
  readonly faceMetrics?: GPUBuffer;
  readonly faceCapacity?: number;
  readonly projectedVelocity: GPUBuffer;
  readonly extendedVelocity: GPUBuffer;
  readonly wetControl: GPUBuffer;
  readonly wetFaceGeometry: GPUBuffer;
  readonly wetAdvectedVelocity: GPUBuffer;
  readonly wetPredictedVelocity: GPUBuffer;
  readonly wetProjectedVelocity: GPUBuffer;
  readonly wetExtendedVelocity: GPUBuffer;
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
}

/** Rejection-only pressure-operator forensics. */
export interface OctreeLanePressureDebug {
  readonly control: GPUBuffer;
  readonly rightHandSide: GPUBuffer;
  readonly diagonal: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  readonly leafHeaders: GPUBuffer;
  readonly rowPhi: GPUBuffer;
  readonly ghostDistances: GPUBuffer;
}

/** Frontier/dirty-tile forensics: leaf-frontier header plus compaction scratch. */
export interface OctreeLaneFrontierDebug {
  readonly frontier: GPUBuffer;
  readonly compaction: GPUBuffer;
  readonly dirtyFailureOffsetBytes: number;
}

/** Rejection-only coarse-phi forensics over the lane's row directory. */
export interface OctreeLaneCoarsePhiDebug {
  readonly control: GPUBuffer;
  readonly rowPhi: GPUBuffer;
  readonly leafHeaders: GPUBuffer;
  readonly dimensions: readonly [number, number, number];
}

/** The adaptive nodal velocity/phi read model, for lanes that publish one. */
export interface OctreeLaneAdaptiveVelocityDebug {
  readonly control: GPUBuffer;
  readonly leaves: GPUBuffer;
  readonly nodalVelocity: GPUBuffer;
  readonly phiControl: GPUBuffer;
  readonly rowPhi: GPUBuffer;
  readonly extensionReach_m: number;
}

/**
 * The lane's factor-one renderer surface authority.
 *
 * Factor-one physics and rendering must select the same scalar authority. The
 * lane that owns the surface names the whole tuple at once so no consumer can
 * pair one lane's generation with another lane's row directory -- the mixed
 * tuple the raster gate rejects, which showed as an empty tank after step one.
 */
export interface OctreeLaneCoarseLevelSetPublication {
  readonly directory: GPUBuffer;
  readonly control: GPUBuffer;
  readonly rowCapacity: number;
  readonly generation: number;
  readonly gradients?: GPUBuffer;
}

/** The symmetry-audit staging buffers `FLUID_SYMMETRY_STAGE_AUDIT=1` captures. */
/**
 * Deep receipts for the smoke harness and the `tools/probe-*` scripts.
 *
 * Nothing the solver consumes reads them, and every consumer already declares
 * its own structural view of the words it decodes, so this contract states
 * only that the lane answers -- not the shape it answers with. Every entry is
 * optional: a lane without the machinery must be absent here rather than
 * fabricate a receipt.
 */
/** The lane-owned half of the technique overlay's cell-trace source. */
export interface OctreeLaneTechniqueDebugSources {
  readonly topologyMetrics: GPUBuffer;
  readonly catalogEntryHeaders: GPUBuffer;
  readonly catalogFaces: GPUBuffer;
  readonly tetrahedronHeaders: GPUBuffer;
  readonly tetrahedra: GPUBuffer;
  readonly tetrahedronVertices: GPUBuffer;
  readonly structuredAuthority: GPUBuffer;
  readonly structuredParams: GPUBuffer;
  readonly structuredRowGeometry: GPUBuffer;
  readonly structuredRowVelocities: GPUBuffer;
  readonly structuredControl: GPUBuffer;
  readonly coarsePhiValues?: GPUBuffer;
}

/**
 * One copy into the shared frontier-failure readback.
 *
 * The engine owns the layout table so a region can never be copied to one
 * offset and decoded from another; a lane names regions and hands over
 * buffers, and never computes an offset of its own.
 */
export type OctreeLaneFrontierFailureCapture = (
  region: OctreeFrontierFailureRegion,
  source: GPUBuffer,
  sourceOffset?: number,
) => void;

/** The lane-owned tail of the shared frontier-failure receipt. */
export interface OctreeLaneFrontierFailureReceipt {
  spgridLevelDelta: readonly number[];
  spgridCandidateDispatch: readonly number[];
  descriptorFailureRow?: unknown;
  boundaryFailureRow?: unknown;
  coarseFailureRow?: unknown;
}

export interface OctreeLaneForensicReadbacks {
  /** ||r0 - A*M*r0|| / ||r0||: the preconditioner's error-propagation factor. */
  preconditionerContraction?(): Promise<unknown>;
  /** Per-level publication census for the lane's multigrid hierarchy. */
  hierarchyCensus?(): Promise<unknown>;
  /** The lane's own end-of-step authority/solver control words. */
  authorityDiagnostics?(): Promise<unknown>;
  /** Complete adaptive graph/phi/velocity publication snapshot. */
  adaptiveSurfacePublication?(): Promise<unknown>;
  /** The GPU-authored unique shared-node worklist receipt. */
  adaptiveNodeReceipt?(): Promise<unknown>;
  /** Receipt for a rejected adaptive candidate transaction. */
  adaptiveCandidateGraphReceipt?(): Promise<unknown>;
  adaptiveVelocityReceipts?(): Promise<unknown>;
  adaptiveVelocityDiagnostics?(): Promise<unknown>;
  /** Bounded owner-page lookup for one pressure row. */
  ownerPageForRow?(row: number): Promise<unknown>;
  /** Control words of the compact interface seed chain. */
  seedChainControls?(): Promise<unknown>;
  /** Decode of one rejected candidate descriptor row. */
  descriptorCandidateFailure?(row: number): Promise<unknown>;
  /** Decode of the compact row a fail-closed coarse-phi transaction named. */
  coarseFailureRow?(row: number): Promise<unknown>;
  /** Section 5 support topology behind a stage-6 first-error word. */
  airSupportFailureTopology?(
    firstError: number,
    latch: readonly number[],
  ): Promise<Readonly<Record<string, unknown>> | undefined>;
  /** Per-level publication census for the lane's multigrid hierarchy. */
  spgridHierarchyCensus?(): Promise<unknown>;
  /** Terminal-only proof that the compact directory differential executed. */
  spgridTouchedDirectoryTripwire?(): Promise<unknown>;
}

export interface OctreeLaneSymmetryStageAuditBuffers {
  readonly initialResidual: GPUBuffer;
  readonly initialPreconditioned: GPUBuffer;
  readonly initialPreconditionedImage: GPUBuffer;
  readonly preconditionerPreSmoothed: GPUBuffer;
  readonly preconditionerZeroSmoothed: GPUBuffer;
  readonly preconditionerFirstOperatorImage: GPUBuffer;
  readonly preconditionerFirstSmoothed: GPUBuffer;
  readonly preconditionerInnerResidual: GPUBuffer;
  readonly preconditionerInnerCorrection: GPUBuffer;
  readonly preconditionerPostCorrected: GPUBuffer;
}

/** What `encode` hands the lane for one advance. */
export interface OctreeAdvanceContext {
  readonly productionBoundary?: OctreeSemanticBoundary;
  /** Stamp shared with the end-of-step snapshot ring. */
  readonly step?: number;
  /**
   * `power-operator-only` stops after the projected velocity publication.
   * That is the explicit t=0 dependency chain's lifecycle boundary, not an
   * alternate simulation path.
   */
  readonly scope: "complete" | "power-operator-only";
}

/** Fine-band transport encode arguments the shared surface step already owns. */
export interface OctreeFineTransportRequest {
  readonly timestep: number;
  readonly inflow?: SurfaceInflowState;
  readonly transportBandCells: number;
  readonly maximumBacktraceFineCells: number;
  readonly openTopBoundary: boolean;
  readonly dynamicBoundary: boolean;
}

/**
 * One coarse dynamics backend, as a strategy over the shared topology engine.
 *
 * Every member here replaced an inline `coarseDynamics.backend === "losasso"`
 * test inside the engine. The tests were spread across the constructor's
 * allocation stages, the initialization task list, the advance, the fine-band
 * surface step and the readback surface; each one was an invitation for the
 * two lanes to drift apart in a shared code path where only one of them was
 * ever exercised.
 */
export interface OctreeCoarseDynamicsLane {
  readonly backend: OctreeCoarseBackend;

  /** The lane's assembled projection module, plus the two clauses it differs by. */
  readonly wgsl: {
    readonly projectionShader: string;
    readonly fragments: OctreeProjectionLane;
  };

  // --- Allocation-stage hooks (constructor) --------------------------------
  /** Largest leaf the topology may hold under this lane's grading law. */
  topologyLeafCeiling(
    maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    dims: { nx: number; ny: number; nz: number },
    exactTilingLeafSize: OctreeOwnerLeafSize,
  ): OctreeOwnerLeafSize;
  /** Top rung of the refinement ladder; must span exactly the held sizes. */
  refinementLadderLeafSize(
    maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    dims: { nx: number; ny: number; nz: number },
    topologyMaximumLeafSize: OctreeOwnerLeafSize,
  ): OctreeOwnerLeafSize;
  /**
   * Whether balance propagation budgets both halves of the mixed-ring rule.
   *
   * Power's stronger exclusive-Delaunay rule can renew an ordinary imbalance,
   * so it pays twice the ordinary closure rounds; a generalized face graph
   * needs only ordinary strict-2:1 propagation and the second half is
   * guaranteed no-op work.
   */
  readonly balanceRoundsUseExclusiveMixedRing: boolean;
  /** Reject a band this lane's velocity extension cannot cover. */
  validateInterfaceBand(interfaceRefinementBandCells: number): void;
  /** Residency tile edge, and the leaf-inside-tile invariant it establishes. */
  topologyTileSize(
    maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    effectiveLeafSize: OctreeOwnerLeafSize,
    exactTilingLeafSize: OctreeOwnerLeafSize,
    topologyMaximumLeafSize: OctreeOwnerLeafSize,
  ): number;
  /** Physical-area reserve multiplier for the fine narrow-band page pool. */
  readonly fineBandSurfaceGrowthSafety: number;
  /** Translated-envelope floor under that reserve; zero where none applies. */
  fineBandBrickFloor(input: {
    readonly brickDimensions: readonly [number, number, number];
    readonly minimumBrick: readonly [number, number, number];
    readonly maximumBrick: readonly [number, number, number];
    readonly capacityDilationBrickRings: number;
  }): number;
  /** Allocation stage 8: build the lane's coarse authority, if it is synchronous. */
  constructAuthority(): void;
  /** Every shader/allocation task this lane still owes after the constructor. */
  initializationTasks(): GPUInitializationTask[];
  /** Lane override for the solve tail's hard iteration ceiling. */
  pressureHardIterationCeiling(): number | undefined;
  /** Human-readable description of the pressure solver actually constructed. */
  pressureSolverLabel(): string;

  // --- Topology transaction hooks -----------------------------------------
  /**
   * Complete the inactive epoch after frontier/owner publication. Every
   * component writes only candidate storage or the inactive structured bank;
   * the final singleton is the sole cross-component validation reduction.
   */
  encodeInactiveCandidate(encoder: GPUCommandEncoder): void;
  /** Beginning of substep N+1: this lane's half of the owner/frontier flip. */
  encodeReadyTopologyFlip(encoder: GPUCommandEncoder): void;
  /**
   * Queue this lane's expected coarse-directory generation for the candidate
   * command buffer, before it is encoded. Directory generation N is produced
   * after topology N and is the authority for the next rebuild; a lane
   * without a scheduled directory writes nothing.
   */
  stampCoarseDirectoryGeneration(): void;
  /** Whether this lane rebuilds topology on every advance regardless of cadence. */
  readonly topologyCadenceIsEveryAdvance: boolean;
  /**
   * Whether the live coarse-band dials describe machinery this lane has.
   *
   * Checked before the engine latches the dial bag, so a lane without the seam
   * never records an applied set it did not act on.
   */
  runtimeDialsApplicable(): boolean;
  applyRuntimeDials(dials: OctreeRuntimeDials): void;

  // --- Advance ------------------------------------------------------------
  encodeAdvance(encoder: GPUCommandEncoder, ctx: OctreeAdvanceContext): GPUCommandEncoder;
  /** Pressure bank the overlay materialization must read this step. */
  overlayPressureAuthorityIsA(pressureInA: boolean): boolean;

  /**
   * Release whatever this lane staged for an encoder that has now submitted.
   * Invocation-stable parameter slots are the case that made this a hook: a
   * host write into shared storage would make every queued invocation observe
   * the final value instead of its own.
   */
  retireSubmittedEncoder(encoder: GPUCommandEncoder): void;
  /** Drop the analytic bootstrap sign from this lane's boundary authority. */
  retireAnalyticBootstrap(): void;

  // --- Fine-band surface step hooks ---------------------------------------
  /**
   * Cold coarse-tracker bootstrap at the head of the surface step. Returns
   * whether it ran, because the pre-force correction below is skipped on the
   * step that bootstrapped (it would transport the same surface twice).
   */
  encodeSurfaceCoarseBootstrap(broker: PassBroker): boolean;
  /**
   * Coarse-only surface advance that is the complete moving-surface authority.
   * Losasso's factor-one lane returns the encoder it finished on; every other
   * configuration returns undefined and falls through to the shared sparse
   * fine-band path below it.
   */
  encodeCoarseOnlySurfaceAdvance(
    broker: PassBroker,
    dt_s: number,
    inflow: SurfaceInflowState | undefined,
  ): GPUCommandEncoder | undefined;
  /**
   * Surface advance for a configuration that allocated no fine band at all.
   * Returning undefined means this lane has no coarse-only authority either,
   * which the engine reports as an incomplete pipeline rather than a no-op.
   */
  encodeCoarseOnlyFallbackAdvance(
    broker: PassBroker,
    dt_s: number,
    coarseBootstrappedThisStep: boolean,
  ): GPUCommandEncoder | undefined;
  /** Binding 9 of the fine topology publication: this lane's coarse phi view. */
  fineTopologyCoarseEntry(binding: number): GPUBindGroupEntry;
  /** The transport stage that owns the currently published fine generation. */
  fineTransportStage(currentIsA: boolean): OctreeFineLevelSetTransportStage | undefined;
  encodeFineTransport(
    broker: PassBroker,
    stage: OctreeFineLevelSetTransportStage,
    request: OctreeFineTransportRequest,
  ): PassBroker;
  /** Coarse phi refresh from the transported target, before this step's forces. */
  encodeCoarsePhiBeforeForces(
    broker: PassBroker,
    target: WebGPUFineLevelSetBrickSource,
    topology: WebGPUFineLevelSetTopology,
    dt_s: number,
  ): void;
  /** Fail closed when a settled generation is missing its air-support epoch. */
  requireSettledSupport(): void;
  /** Section 5 support refresh over the settled fine generation. */
  encodeSettledSupport(
    encoder: GPUCommandEncoder,
    dt_s: number,
    phase: "t0" | "recurring",
  ): GPUCommandEncoder;

  // --- Fine-band settlement hooks -----------------------------------------
  /** Coarse authority bootstrap before the first volume correction. */
  encodeSettlementBootstrap(broker: PassBroker, target: WebGPUFineLevelSetBrickSource): void;
  /**
   * Whether a settled volume correction moves phi or only measures it.
   *
   * The adaptive Losasso lane treats phi as geometric signed-distance
   * authority: volume loss is measured there but never repaired by a post-step
   * scalar offset. Power retains its own policy.
   */
  readonly settlementVolumePolicy: "measure" | "correct";
  /** Coarse authority republication from the fully redistanced band. */
  encodeSettlementCoarseRefresh(broker: PassBroker, target: WebGPUFineLevelSetBrickSource): void;
  /** Restriction/summary/extension publication after the topology finalize. */
  encodeSettlementRestriction(
    broker: PassBroker,
    target: WebGPUFineLevelSetBrickSource,
    topology: WebGPUFineLevelSetTopology,
  ): void;

  // --- Readback and diagnostics -------------------------------------------
  readonly debug: OctreeLaneDebugSources;
  debugSources(): Record<string, unknown>;
  /** The A/B pressure banks the overlay's diagnostic bind groups must read. */
  readonly diagnosticPressureBanks: Readonly<{
    pressureA: GPUBufferBinding;
    pressureB: GPUBufferBinding;
  }> | undefined;
  /** Symmetry-audit staging, wherever this lane's preconditioner captured it. */
  readonly solverSymmetryStageAuditBuffers: OctreeLaneSymmetryStageAuditBuffers | undefined;
  /**
   * The backend-neutral coarse directory pair for generic render/QA consumers.
   *
   * A lane's private topology-sampling structure must never leak through this
   * source; publishing the neutral pair is how that stays true by construction.
   */
  genericCoarseDirectory(): Readonly<{
    directory: GPUBuffer;
    rowCapacity: number;
  }> | undefined;
  /** This lane's factor-one renderer surface authority, generation included. */
  coarseLevelSetPublication(): OctreeLaneCoarseLevelSetPublication | undefined;
  /** Producer state paired with an unpublished fine summary. */
  summaryCoarseDebug(): Readonly<{ control: GPUBuffer; delta: GPUBuffer }> | undefined;
  /** Whether a coherent adaptive step receipt may advance the scalar clock. */
  acceptsAdaptiveSurfaceGenerationReceipt(): boolean;
  /**
   * Whether this lane holds a moving-surface authority of its own.
   *
   * The sparse render publication needs some settled surface: either the
   * shared fine band, or this. A lane that has neither leaves the publication
   * with nothing to name, which is the error it raises.
   */
  hasCoarseSurfaceAuthority(): boolean;
  /**
   * Whether this lane's compact velocity authority can source the renderer.
   *
   * Dense bootstrap phi may only be released once every recurring consumer
   * reads page buffers instead, and this is the consumer the engine cannot
   * see: it lives entirely inside the lane.
   */
  compactRendererSourceReady(): boolean;
  /** Whether this lane published the coarse projection bind groups. */
  readonly hasCoarseProjectionGroups: boolean;
  /** This lane's own outer-iteration budget, when its solver names one. */
  solverIterationBudget(): number | undefined;
  /** Lane-owned rows of the production work-accounting telemetry. */
  workAccountingBuffers(): Readonly<{
    pressureRhs?: GPUBufferBinding;
    section63Coefficients?: GPUBufferBinding;
  }>;
  /** This lane's half of the technique overlay cell trace, when it has one. */
  techniqueDebugSources(): OctreeLaneTechniqueDebugSources | undefined;
  /**
   * QA-only symmetry sources for this lane's boundary and air-support banks.
   *
   * Deliberately opaque: the shape names this lane's own planner types, which
   * octree-shared may not import. Every consumer already re-declares the
   * fields it reads structurally, so widening here costs it nothing.
   */
  readonly structuredBoundarySymmetryDebug: unknown;
  /**
   * The six candidate publication controls the frontier-failure receipt
   * reports. A lane whose single reduced authority replaces all six captures
   * it into each region, which keeps the long-standing receipt shape.
   */
  captureFrontierFailureAuthorityControls(capture: OctreeLaneFrontierFailureCapture): void;
  /** This lane's coarse directory/delta sources for the same receipt. */
  captureFrontierFailureCoarseSources(capture: OctreeLaneFrontierFailureCapture): void;
  /** This lane's candidate row and indirect-dispatch sources for that receipt. */
  captureFrontierFailureCandidateSources(capture: OctreeLaneFrontierFailureCapture): void;
  /** Post-map, lane-owned decodes appended to the frontier-failure receipt. */
  decodeFrontierFailure(controls: Readonly<{
    descriptorCandidate: readonly number[];
    boundaryCandidate: readonly number[];
    coarseControl: readonly number[];
  }>): Promise<OctreeLaneFrontierFailureReceipt>;
  /**
   * Copy this lane's regions of the shared global-fine QA readback.
   *
   * Every destination range a lane writes belongs to it alone, so the engine
   * can emit them in one block without changing what the mapped words hold.
   */
  encodeGlobalFineDiagnosticCopies(encoder: GPUCommandEncoder, readback: GPUBuffer): void;
  /** Live solve receipt this lane owns; false leaves the shared staging path. */
  readSolveDiagnostics(): Promise<boolean>;
  /** Adopt an end-of-step receipt the snapshot ring captured for this lane. */
  applyStepDiagnostics(authority: Uint32Array, solver: Uint32Array): void;
  readonly velocityDebug: OctreeLaneVelocityDebug | undefined;
  readonly pressureDebug: OctreeLanePressureDebug | undefined;
  readonly frontierDebug: OctreeLaneFrontierDebug | undefined;
  readonly coarsePhiDebug: OctreeLaneCoarsePhiDebug | undefined;
  /**
   * Fail-closed t=0 validation of this lane's own published authority.
   *
   * Throws with the lane's own receipt words when the paused startup tuple is
   * rejected; the shell cannot decode those words without naming the lane's
   * receipt ABI, which is exactly the coupling this hook removes. Returning
   * undefined means the lane has no dedicated gate and the shell's shared
   * structured/boundary validation applies instead.
   */
  validateInitialAuthority(context: {
    readonly dimensions: readonly [number, number, number];
    /** Re-read the engine telemetry between readbacks, as the shell does. */
    readonly refreshInfo: () => void;
  }): Promise<Readonly<{ converged: boolean; iterationsUsed: number }> | undefined>;
  /** Diagnostic-only receipt for the lane's factor-one surface authority. */
  readCoarseSurfaceTrackerReceipt(): Promise<unknown>;
  readonly forensics: OctreeLaneForensicReadbacks;

  destroy(): void;
}

/**
 * A lane is built against an engine that is still under construction: it must
 * hold the reference and read nothing until `constructAuthority`.
 */
export type OctreeCoarseDynamicsLaneFactory = (engine: OctreeTopologyEngine) => OctreeCoarseDynamicsLane;

const laneFactories = new Map<OctreeCoarseBackend, OctreeCoarseDynamicsLaneFactory>();

/**
 * Installed by the method plugin, which is the only place allowed to name both
 * backends. The engine resolves a lane by id so that adding a coarse dynamics
 * backend never edits `octree-shared`.
 */
export function registerOctreeCoarseDynamicsLane(
  backend: OctreeCoarseBackend,
  factory: OctreeCoarseDynamicsLaneFactory,
): void {
  laneFactories.set(backend, factory);
}

export function octreeCoarseDynamicsLaneFactory(
  backend: OctreeCoarseBackend,
): OctreeCoarseDynamicsLaneFactory {
  const factory = laneFactories.get(backend);
  if (!factory) {
    throw new Error(`No coarse dynamics lane is installed for backend ${backend}`
      + " -- import the octree method plugin before constructing a solver");
  }
  return factory;
}

export type { OctreeAllocationProgress };
