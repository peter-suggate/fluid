import type { WebGPUFineLevelSetBrickSource } from "../../core/levelset-consumer-abi";
import type { SceneDescription } from "../../core/model";
import {
  OCTREE_REFINEMENT_REGION_PARAMS_BYTES,
  OCTREE_REFINEMENT_REGION_PARAMS_OFFSET,
  packOctreeRefinementRegions,
} from "./octree-refinement-regions";
import { sceneRefinementRegions } from "../../core/refinement-regions";
import {
  OCTREE_COLD_AUTHORED_SURFACE_PARAMS_BYTES,
  octreeColdAuthoredSurfaceBoxes,
  packOctreeColdAuthoredSurfaceBoxes,
} from "./octree-cold-authored-surface";
import { WebGPUOctreeFineSeedAdapter } from "./webgpu-octree-fine-seed-adapter";
import {
  WebGPUOctreeSimulationOwnerPages,
  type OctreeOwnerLeafSize,
} from "./webgpu-octree-owner-pages";
import { PassBroker } from "../../core/webgpu-pass-broker";
import {
  octreeCoarseDynamicsLaneFactory,
  type OctreeCoarseDynamicsLane,
  type OctreeFineLevelSetTransportStage,
  type OctreeProjectionTelemetry,
  type OctreeTopologyEngine,
} from "./octree-coarse-dynamics-lane";
import { planOctreeSurfaceStateAllocation } from "./octree-surface-allocation";
import { planOctreeAnalyticBootstrapBounds } from "./octree-analytic-bootstrap";
import { WebGPUOctreeAnalyticBootstrapWorklist } from "./webgpu-octree-analytic-bootstrap";
import {
  damBreakBoxContains,
  damBreakSignedDistanceAtNode,
  initialFluidBrickComponentBounds,
  initialFluidBrickSignedDistanceAtCell,
  initialFluidBrickSignedDistanceAtNode,
  initialFluidBrickUnionBounds,
  initialLiquidContainsCell,
  sceneDamBreakBox,
  sceneDamBreakFractions,
  sceneDamBreakIsOffsetFromCorner,
  sceneHasInitialLiquidVolumes,
} from "../../core/initial-fluid";
import { signedDistanceFromVolume } from "../../core/volume-signed-distance";
import { sceneHasTerrain, terrainColumnHeights } from "../../core/terrain";
import {
  WebGPUQuadtreeSurfaceState,
  type SurfaceInflowState,
} from "./surface-state";
import {
  createOctreeSparseResidencyWorld,
  type OctreeSparseResidencyWorld,
} from "../../core/octree-sparse-residency-world";
import type { SparseScenePrimitiveUpdate } from "../../core/webgpu-sparse-scene-proxies";
import {
  resolveOctreeCoarseDynamics,
  type OctreeCoarseDynamicsConfiguration,
} from "./octree-coarse-backend";
import {
  FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES,
  GPUFluidBrickResidency,
  planFineSeedCandidateResidencyPools,
} from "../../core/webgpu-fluid-brick-residency";
import type { GPUInitializationTask } from "../../core/gpu-initialization";
import {
  planGPUShaderCapabilities,
  planGPUShaderTasks,
  type GPUShaderCapabilityPlan,
  type GPUShaderTaskDefinition,
} from "../../core/gpu-shader-plan";
import { planOctreeSolveTail, type OctreeSolveTailPolicy } from "./octree-solve-tail-policy";
import {
  octreeDialledSurfaceBand,
  octreeRuntimeDialsEqual,
  octreeSurfaceProtectionWidthCells,
  type OctreeRuntimeDials,
} from "./octree-runtime-dials";
import { OctreeWorkAccounting } from "./webgpu-octree-work-accounting";
import { WebGPUOctreeCoarseSummary } from "./webgpu-octree-coarse-summary";
import { planFineLevelSetBricks } from "./octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "./webgpu-octree-fine-levelset-bricks";
import { WebGPUFineLevelSetRedistance } from "./webgpu-octree-fine-levelset-redistance";
import { WebGPUFineLevelSetTransport } from "./webgpu-octree-fine-levelset-transport";
import { WebGPUFineLevelSetVolumeCorrection } from "./webgpu-octree-fine-levelset-volume";
import { WebGPUFineLevelSetRigidCarve } from "./webgpu-octree-fine-levelset-rigid-carve";
import {
  createGPULogicalActivityAdoptionContext,
  type GPULogicalActivityAdoptionContext,
} from "../../core/gpu-logical-activity-adoption";
import { performanceShaderVariant } from "../../core/stores/performance-instrumentation-store";
import {
  planFineLevelSetGPUSummaries,
  WebGPUFineLevelSetSummaries,
} from "./webgpu-octree-fine-levelset-summary";
import {
  planFineLevelSetBandFineCells,
  planFineLevelSetCapacityDilationBrickRings,
  planFineLevelSetTopologyBand,
  WebGPUFineLevelSetLeafSeeds,
  WebGPUFineLevelSetTopology,
} from "./webgpu-octree-fine-levelset-topology";
import {
  OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS,
  OCTREE_PROJECTION_ACTIVITY_MODULE_ID,
  OCTREE_PROJECTION_ACTIVITY_TASKS,
  OCTREE_PROJECTION_BASE_ENTRY_POINTS,
  octreeProjectionPipelineRequired,
  type OctreeProjectionPipelineReachability,
} from "./octree-projection-entry-points";
import {
  OCTREE_ALLOCATION_STAGES,
  type OctreeAllocationProgress,
  type OctreeEnginePhase,
  type OctreeFineSemanticPhase,
  type OctreeInitialSparseAuthorityPhaseId,
  type OctreeProjectionOptions,
  type OctreeSemanticBoundary,
  type OctreeProjectionResources,
} from "./octree-projection-contract";
import {
  octreeFineEngineSplitsEnabled,
  octreeGradingMembershipLoadEnabled,
  octreeGradingPageFillEnabled,
  octreeGradingSplitHelpersEnabled,
} from "./octree-projection-gates";
import {
  octreeDensePhiReleaseReady,
  octreeSparseWorldRequired,
  planOctreeSolidCellAllocation,
} from "./octree-dense-field-residency";
import {
  octreeDeviceRowCapacityCeiling,
  planOctreeFluidFootprintBudget,
  planOctreePressureCapacity,
  type OctreePressureCapacityPlan,
} from "./octree-pressure-capacity";
import {
  planFluidFootprintFineNarrowBandBrickCapacity,
  resolveGlobalFineBrickCapacity,
} from "./octree-fine-band-capacity";
import {
  planOctreeCompactionAllocation,
  planOctreeLeafFrontierAllocation,
  type OctreeLeafFrontierAllocationPlan,
} from "./octree-arena-allocation";
import {
  censusOctreeTopologyLeaves,
  type OctreeTopologyLeafCensus,
} from "./octree-topology-census";
import {
  octreeBalanceRounds,
  octreeLeafSize,
  octreeLosassoTopologyLeafSize,
} from "./octree-leaf-sizing";
import {
  OCTREE_PROJECTION_CORE_BUFFER_LAYOUT,
  OCTREE_PROJECTION_CORE_TEXTURE_LAYOUT,
  OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT,
  projectionBufferLayoutEntries,
} from "./octree-projection-bind-layouts";

type OctreePipelineVariants = { full: GPUComputePipeline; delta: GPUComputePipeline };

interface OctreePipelineCacheEntry {
  base: GPUComputePipeline[];
  frontierSort: GPUComputePipeline[];
  refine: Map<number, OctreePipelineVariants>;
  refineCoarse: Map<number, OctreePipelineVariants>;
  balanceCoarse: Map<number, OctreePipelineVariants>;
}
const octreePipelineCache = new WeakMap<GPUDevice, Map<string, OctreePipelineCacheEntry>>();
const octreeDiagnosticPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();

interface PendingFinePublication {
  readonly topology: WebGPUFineLevelSetTopology;
  readonly redistance: WebGPUFineLevelSetRedistance;
  readonly volume?: WebGPUFineLevelSetVolumeCorrection;
  readonly transport?: OctreeFineLevelSetTransportStage;
  readonly target: WebGPUFineLevelSetBrickSource;
  readonly targetIsA: boolean;
  readonly redistanceBandCells: number;
  readonly maximumDisplacementFineCells: number;
  readonly warmClosestPoints: boolean;
}

/**
 * Failure-only readback layout, in words. Both the copy offsets and the decode
 * slices are derived from this one table, so a region cannot be copied to one
 * offset and decoded from another. Reserved entries hold historical offsets
 * stable; a region whose source is absent this run simply stays zero.
 */
const OCTREE_FRONTIER_FAILURE_REGIONS = [
  ["frontier", 16], ["compaction", 16], ["reservedControl", 16],
  ["frontierFailure", 8], ["frontierPublication", 14], ["dirtyAuthorityState", 1],
  ["descriptorCandidate", 16], ["topologyCandidate", 16], ["structuredCandidate", 16],
  ["boundaryCandidate", 16], ["spgridCandidate", 16], ["epoch", 16], ["rowDelta", 16],
  ["ownerCandidate", 32], ["carryFlags", 64], ["fineSummaryDirectory", 16],
  ["fineSummaryWorkState", 32], ["coarseControl", 16], ["coarseDirectory", 8],
  ["coarseDelta", 16], ["finePageDelta", 16], ["rowDeltaNewToOld", 64],
  ["rowDeltaAffectedRows", 64], ["descriptorCandidates", 64], ["descriptorStatuses", 64],
  ["structuredDispatch", 30], ["candidateSchedules", 9], ["reservedAlignment", 1],
  ["frontierCandidates", 32],
] as const satisfies ReadonlyArray<readonly [string, number]>;

export type OctreeFrontierFailureRegion = (typeof OCTREE_FRONTIER_FAILURE_REGIONS)[number][0];

const OCTREE_FRONTIER_FAILURE_LAYOUT = (() => {
  const spans = new Map<string, { readonly words: number; readonly bytes: number; readonly count: number }>();
  let words = 0;
  for (const [name, count] of OCTREE_FRONTIER_FAILURE_REGIONS) {
    spans.set(name, { words, bytes: words * 4, count });
    words += count;
  }
  return Object.freeze({
    totalBytes: words * 4,
    span(name: OctreeFrontierFailureRegion) {
      const found = spans.get(name);
      if (!found) throw new RangeError(`Unknown octree frontier failure region ${name}`);
      return found;
    },
  });
})();

/**
 * A GPU-resident, pressure-only octree projection.
 *
 * Ownership is paged and pressure exists only at live leaf origins, resolved
 * through the compact frontier hash.
 */
export class WebGPUOctreeProjection implements OctreeTopologyEngine {
  readonly preconditioner = "section43-hybrid" as const;
  /** The coarse dynamics strategy. Every seam where the two backends
   * genuinely differ is a named member of it. */
  readonly lane: OctreeCoarseDynamicsLane;
  /** Observational only; no simulation branch consumes these counters. */
  readonly workAccounting = new OctreeWorkAccounting();
  readonly info: OctreeProjectionTelemetry;
  levelSetMismatchFraction = 0;
  relativeResidual?: number;
  residualRms?: number;
  initialResidualRms?: number;

  private readonly topology: GPUBuffer;
  readonly ownerPages: WebGPUOctreeSimulationOwnerPages;
  readonly pressureA: GPUBuffer;
  readonly pressureB: GPUBuffer;
  readonly compaction: GPUBuffer;
  readonly leafHeaders: GPUBuffer;
  readonly candidateLeafHeaders: GPUBuffer;
  readonly candidatePressure: GPUBuffer;
  /** Plain-u32 ping/pong scratch used only by the cold large-frontier merge sort. */
  private readonly frontierSortScratch: GPUBuffer;
  readonly leafFrontier: GPUBuffer;
  readonly fineSeedAdapter?: WebGPUOctreeFineSeedAdapter;
  readonly globalFineLevelSet?: WebGPUFineLevelSetBricks;
  private readonly globalFineSeeds?: WebGPUFineLevelSetLeafSeeds;
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
  private lastGlobalFineTransport?: OctreeFineLevelSetTransportStage;
  readonly globalFineSummaries?: WebGPUFineLevelSetSummaries;
  coarseOnlySummary?: WebGPUOctreeCoarseSummary;
  readonly unpublishedFineSummaryDirectory: GPUBuffer;
  private surfaceStateAccountingBytes = 0;
  globalFineSourceA?: WebGPUFineLevelSetBrickSource;
  globalFineSourceB?: WebGPUFineLevelSetBrickSource;
  /** Slot consumed by the command stream currently being encoded. */
  globalFineCurrentIsA = true;
  /** Last slot whose producing encoder has actually been queue-submitted. */
  globalFinePublishedIsA = true;
  private readonly globalFinePublicationByEncoder = new WeakMap<GPUCommandEncoder, boolean>();
  private readonly analyticBootstrapRetirementByEncoder = new WeakSet<GPUCommandEncoder>();
  /** Once the t=0 authority is retired, scalar scene revisions must never
   * re-arm the analytic selector: `writeParams` runs on every
   * `applySceneUniforms`, and a re-armed selector silently rebuilds all
   * later topology candidates from the authored t=0 surface. Only a re-seed
   * (which re-runs the fenced bootstrap phases) may arm it again. */
  private analyticBootstrapRetired = false;
  /** Recurring fine generation transported before forces and settled only
   * after projection/extension has published. */
  private pendingFinePublication?: PendingFinePublication;
  globalFineBootstrapped = false;
  /** Monotone host mirror of the topology shader's `currentFinePopulated()`
   * (POWER_LIQUIDS_ULTIMATE_M1MAX.md B1 / P1.1). Set once a delta publication
   * has been encoded, i.e. after the cold bootstrap and its retry step. Only
   * the seed-chain encode consults it; a stale `false` merely re-encodes the
   * chain harmlessly, and it is never cleared short of teardown. */
  private finePopulated = false;
  private globalFineGeneration = 2;
  powerTimestep_s = 0;
  surfaceInflow?: SurfaceInflowState;
  pendingSurfaceReferenceVolume_m3 = 0;
  /** Host-side generation admitted for the factor-one renderer adapter.
   * Only a coherent step receipt updates this mirror; rejected GPU work never
   * becomes a current render publication merely because it was encoded. */
  adaptiveSurfaceGeneration = 0;
  /** Bodies that integrate this step; zero keeps the adjoint off the graph. */
  dynamicCouplingBodyCount = 0;
  powerAdvancingPressureSteps = 0;
  private readonly solveDispatch: GPUBuffer;
  readonly topologyCandidateDispatch: GPUBuffer;
  readonly topologyTileChangeFlagsOffsetBytes: number;
  readonly topologyTileChangeFlagsByteLength: number;
  readonly compactionAllocationRowDeltaScratchOffsetBytes: number;
  readonly dirtyFailureOffsetBytes: number;
  readonly frontierPublicationOffsetBytes: number;
  readonly dirtyAuthorityStateOffsetBytes: number;
  readonly solidCells: GPUBuffer;
  private readonly hasDenseSolidCells: boolean;
  readonly params: GPUBuffer;
  private readonly projectionActivity: GPULogicalActivityAdoptionContext;
  private readonly projectionActivityShaderKey?: string;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly frontierSortLayout: GPUBindGroupLayout;
  private readonly frontierSortPipelineLayout: GPUPipelineLayout;
  private readonly frontierSortStageParams?: GPUBuffer;
  private shader?: GPUShaderModule;
  private readonly projectionShaderCode: string;
  private readonly diagnosticLayout: GPUBindGroupLayout;
  private readonly diagnosticPipelineLayout: GPUPipelineLayout;
  private diagnosticShader?: GPUShaderModule;
  readonly surfaceState: WebGPUQuadtreeSurfaceState;
  private readonly sparseBrickWorld?: OctreeSparseResidencyWorld;
  readonly topologyResidency: GPUFluidBrickResidency;
  private readonly analyticBootstrapWorklist?: WebGPUOctreeAnalyticBootstrapWorklist;
  groups: { ab: GPUBindGroup; ba: GPUBindGroup };
  candidateRowGroups: { fromA: GPUBindGroup; fromB: GPUBindGroup };
  fineSummarySizingGroup: GPUBindGroup;
  private readonly frontierSortGroups: readonly GPUBindGroup[];
  /** Current fine/coarse classification plus persistent topology-tile membership. */
  topologyDecisionGroup?: GPUBindGroup;
  private denseBootstrapPhiReleased = false;
  private topologyDiagnosticTexture?: GPUTexture;
  private pressureSamplesDiagnosticTexture?: GPUTexture;
  private pressureDiagnosticTexture?: GPUTexture;
  diagnosticGroups?: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
  private rasterizeSolidsPipeline!: GPUComputePipeline;
  private resetPipeline!: GPUComputePipeline;
  private refinePipeline!: GPUComputePipeline;
  private balancePipeline!: GPUComputePipeline;
  private rasterizeSolidsDeltaPipeline!: GPUComputePipeline;
  private resetDeltaPipeline!: GPUComputePipeline;
  private refineDeltaPipeline!: GPUComputePipeline;
  private balanceDeltaPipeline!: GPUComputePipeline;
  private readonly refineLevelPipelines = new Map<number, OctreePipelineVariants>();
  private readonly refineCoarsePipelines = new Map<number, OctreePipelineVariants>();
  private readonly balanceCoarsePipelines = new Map<number, OctreePipelineVariants>();
  private planPipeline!: GPUComputePipeline;
  private scanPipeline!: GPUComputePipeline;
  private emitPipeline!: GPUComputePipeline;
  private stampFrontierAttemptPipeline!: GPUComputePipeline;
  private beginFrontierPipeline!: GPUComputePipeline;
  private classifyFrontierCandidatesPipeline!: GPUComputePipeline;
  private classifyFrontierCandidatesDeltaPipeline!: GPUComputePipeline;
  private prefixFrontierCandidateBlocksPipeline!: GPUComputePipeline;
  private prefixFrontierCandidateBlocksDeltaPipeline!: GPUComputePipeline;
  private emitFrontierCandidatesPipeline!: GPUComputePipeline;
  private emitFrontierCandidatesDeltaPipeline!: GPUComputePipeline;
  private prepareFrontierDispatchPipeline!: GPUComputePipeline;
  private sortFrontierCandidatesLocalPipeline!: GPUComputePipeline;
  private frontierCandidateSortPipelines: GPUComputePipeline[] = [];
  private classifyFrontierCarryPipeline!: GPUComputePipeline;
  private scanFrontierCarryBlocksPipeline!: GPUComputePipeline;
  private prefixFrontierCarryBlocksPipeline!: GPUComputePipeline;
  private mergeFrontierRowsPipeline!: GPUComputePipeline;
  private finalizeFrontierPipeline!: GPUComputePipeline;
  private prepareRowDeltaPipeline!: GPUComputePipeline;
  private classifyRowDeltaPipeline!: GPUComputePipeline;
  private finalizeRowDeltaClassificationPipeline!: GPUComputePipeline;
  private scanDirtyRowDeltaBlocksPipeline!: GPUComputePipeline;
  private prefixDirtyRowDeltaBlocksPipeline!: GPUComputePipeline;
  private scatterDirtyRowDeltaPipeline!: GPUComputePipeline;
  private markRowDeltaRingPipeline!: GPUComputePipeline;
  private markRowDeltaRingBlocksPipeline!: GPUComputePipeline;
  private scanAffectedRowDeltaBlocksPipeline!: GPUComputePipeline;
  private prefixAffectedRowDeltaBlocksPipeline!: GPUComputePipeline;
  private compactRowDeltaPipeline!: GPUComputePipeline;
  private publishRowDeltaPipeline!: GPUComputePipeline;
  private publishReusedRowDeltaPipeline!: GPUComputePipeline;
  private classifyTopologyTileSignaturePipeline!: GPUComputePipeline;
  private buildDirtyTileDeltaPipeline!: GPUComputePipeline;
  private buildDirtyFrontierDeltaPipeline!: GPUComputePipeline;
  materializePipeline?: GPUComputePipeline;
  readonly maxLeafSize: 2 | 4 | 8 | 16 | 32;
  /** Backend-normalized maximum consumed by the structural topology. */
  readonly topologyMaximumLeafSize: OctreeOwnerLeafSize;
  /** Coarsest span that tiles the domain exactly; the residency tile edge only. */
  private readonly losassoExactTilingLeafSize: OctreeOwnerLeafSize;
  readonly coarseDynamics: OctreeCoarseDynamicsConfiguration;
  private topologyCadenceCursor = 0;
  /** A tail deliberately omitted its candidate because the accepted topology
   * remains authoritative for another cadence advance. Distinct from an
   * accidental missing candidate. */
  topologyReusePending = false;
  /** Last accepted epoch counted from authority word 5's exact row-identity
   * receipt. Diagnostics can sample one epoch more than once. */
  lastObservedExactTopologyReuseEpoch = 0;
  /** Live cadence dial; undefined defers to the construction-time policy. */
  private topologyCadenceOverride?: number;
  /** Last bag applied, so a per-frame call with an unchanged bag costs a compare. */
  private appliedRuntimeDials?: OctreeRuntimeDials;
  private readonly fluidGatedBoundaryRefinement: boolean;
  private readonly topologyTileSize: number;
  private readonly adaptivity: number;
  /** Authored band. Sized the row capacity, the halo and the redistance reach. */
  private readonly interfaceRefinementBandCells: number;
  /**
   * The band the shader is currently running, which the live dial may thin.
   * Only `writeParams` reads it: every other consumer of the authored band is
   * an allocation that was made once and must keep describing itself.
   */
  interfaceBandCellsEffective = 0;
  /**
   * The grading layers the shader is currently running, which the live dial may
   * reduce toward the sharp 2:1 transition. The authored value stays put: it
   * sized the redistance reach and describes what was allocated.
   */
  surfaceGradingLayersEffective = 1;
  /** Live closed-container look-ahead; three preserves the measured default. */
  private wallBandCellsEffective = 1;
  /** Live factor-one cut floor. Two is the explicit coarse-cut experiment. */
  private finestSurfaceCellSizeEffective = 1;
  private readonly surfaceRefinementGradingLayers: number;
  readonly fineLevelSetBandCells: number;
  /** Factor-one uses the compact octree phi as the sole moving surface;
   * factors four/eight allocate the separate sparse fine band. */
  readonly coarseOnlySurfaceTracking: boolean;
  pressureSolverControl!: GPUBuffer;
  readonly pressureCapacity: OctreePressureCapacityPlan;
  readonly frontierAllocation: OctreeLeafFrontierAllocationPlan;
  /** A 4096-word shared sort occupies exactly WebGPU's portable 16 KiB floor. */
  private readonly useLocalFrontierCandidateSort: boolean;
  /** One workgroup per row is exact only while the 1-D indirect extent fits. */
  private readonly useCooperativeRowDeltaRing: boolean;
  /** Immutable cold-bootstrap and optional diagnostic dispatch records. */
  private readonly coldDispatch: GPUBuffer;
  private readonly coldDispatchOffsetBySize = new Map<number, number>();
  private readonly effectiveLeafSize: OctreeOwnerLeafSize;
  private readonly refinementSizes: readonly number[];
  private readonly coarseRefinementSizes: readonly number[];
  private readonly balanceRounds: number;
  private readonly linearBlocks: number;
  compactionByteLength = 0;
  solveStats!: GPUBuffer;
  private topologyWorklistReady = false;
  latestPressureInA = true;
  /** No dense phi exists; non-page topology groups must retain analytic sign until coarse correction publishes. */
  readonly analyticSparseBootstrap: boolean;
  /** Host-side encode serial used only for API validation/diagnostics. The
   * physics generation is stamped in command-buffer order by the GPU. */
  private powerAttemptGeneration = 0;
  candidatePowerGeneration = 0;
  activePowerGeneration = 0;
  readonly solveTailPolicy: OctreeSolveTailPolicy;
  powerLifecycleDisposed = false;

  constructor(
    readonly device: GPUDevice,
    // Not readonly: `applySceneUniforms` swaps in scalar-only scene revisions
    // so a density or gravity edit is a uniform write, not a rebuild.
    public scene: SceneDescription,
    readonly dims: { nx: number; ny: number; nz: number },
    readonly resources: OctreeProjectionResources,
    options: OctreeProjectionOptions,
    readonly deferPipelineCompilation = false,
    allocationProgress?: OctreeAllocationProgress,
  ) {
    this.deferPipelineCompilation = true;
    const reportAllocation = (stage: number) => allocationProgress?.(
      OCTREE_ALLOCATION_STAGES[stage]!, stage, OCTREE_ALLOCATION_STAGES.length,
    );
    reportAllocation(0);
    const count = dims.nx * dims.ny * dims.nz;
    const globalFineFactor = options.globalFineLevelSetFactor ?? 1;
    this.coarseDynamics = options.coarseDynamics ?? resolveOctreeCoarseDynamics({
      // Direct construction follows the product default. Frozen Power
      // reference lanes must opt in explicitly at their call site.
      backend: "losasso",
    });
    // The lane is built against an engine that is still under construction.
    // It holds the reference and reads nothing until `constructAuthority`.
    this.lane = octreeCoarseDynamicsLaneFactory(this.coarseDynamics.backend)(this);
    this.maxLeafSize = octreeLeafSize(options.maximumLeafSize ?? 16);
    // The exact tiling is retained, but only as the tile ABI -- see
    // `octreeLosassoTopologyLeafSize`. Losasso's leaf ceiling is now the same
    // "largest leaf the domain can hold" rule Power uses, so the deep interior
    // coarsens by distance to the interface instead of by gcd(nx, ny, nz).
    this.losassoExactTilingLeafSize = octreeLosassoTopologyLeafSize(this.maxLeafSize, dims);
    this.topologyMaximumLeafSize = this.lane.topologyLeafCeiling(
      this.maxLeafSize, dims, this.losassoExactTilingLeafSize);
    this.fluidGatedBoundaryRefinement = options.fluidGatedBoundaryRefinement ?? true;
    this.solveTailPolicy = planOctreeSolveTail({
      finestDimensions: [dims.nx, dims.ny, dims.nz],
      maximumLeafSize: this.maxLeafSize as 2 | 4 | 8 | 16 | 32,
      initialCondition: scene.fluid.initialCondition,
      hasInflow: scene.fluid.inflow !== undefined,
      hasTerrain: sceneHasTerrain(scene),
      movingRigidBodyCount: scene.rigidBodies.filter((body) => body.motion !== "static").length,
      closedTop: scene.container.top === "closed",
      requestedRelativeTolerance: scene.numerics.pressureRelativeTolerance,
    });
    // The refinement ladder must span exactly the sizes the topology may hold.
    // A rung above the ceiling compiles and dispatches a level with no eligible
    // candidate; a rung below it strands leaves the seed produced.
    this.effectiveLeafSize = this.lane.refinementLadderLeafSize(
      this.maxLeafSize, dims, this.topologyMaximumLeafSize);
    this.refinementSizes = Object.freeze((() => {
      const sizes: number[] = [];
      for (let size = this.effectiveLeafSize; size >= 2; size >>= 1) sizes.push(size);
      return sizes;
    })());
    this.coarseRefinementSizes = Object.freeze(
      this.refinementSizes.filter((size) => size >= 16),
    );
    // One propagation per tree LEVEL, and the tree has no level above the
    // largest leaf the domain can hold.
    this.balanceRounds = octreeBalanceRounds(
      this.effectiveLeafSize,
      this.lane.balanceRoundsUseExclusiveMixedRing,
    );
    this.adaptivity = Math.max(0, Math.min(1, options.adaptivity ?? 1));
    this.interfaceRefinementBandCells = Math.max(0, Math.min(32, Math.round(options.interfaceRefinementBandCells ?? 4)));
    this.lane.validateInterfaceBand(this.interfaceRefinementBandCells);
    this.surfaceRefinementGradingLayers = Math.max(1, Math.min(4,
      Math.round(options.surfaceRefinementGradingLayers ?? 1)));
    const initialSurface = options.initialRuntimeDials
      ? octreeDialledSurfaceBand(
        this.interfaceRefinementBandCells,
        this.surfaceRefinementGradingLayers,
        globalFineFactor === 1 ? 1 : 4,
        options.initialRuntimeDials,
      )
      : undefined;
    this.interfaceBandCellsEffective = initialSurface?.bandCells
      ?? this.interfaceRefinementBandCells;
    this.surfaceGradingLayersEffective = initialSurface?.gradingLayers
      ?? this.surfaceRefinementGradingLayers;
    this.wallBandCellsEffective = options.initialRuntimeDials?.wallBandCells
      ?? 1;
    this.finestSurfaceCellSizeEffective = options.initialRuntimeDials?.finestSurfaceCellSize
      ?? 1;
    // Product configurations couple Section 5 surface reach to pressure reach.
    // A distinct value is admitted only for diagnostic fault injection; unset
    // follows the master band exactly.
    this.fineLevelSetBandCells = Math.max(0, Math.min(32,
      Math.round(options.fineLevelSetBandCells ?? this.interfaceRefinementBandCells)));
    // Ando--Batty remeshes the octree itself, keeps every free-surface cut at
    // its finest pressure/level-set tier, and assigns the old fields onto the
    // new tree. Factor one therefore uses the graph-owned adaptive mass/phi
    // authority: a crossing leaf may never coarsen, while pure wet/air leaves
    // grade rapidly away under the ordinary 2:1 closure. Factors 4/8 remain a
    // distinct sparse subcell-interface experiment.
    this.coarseOnlySurfaceTracking = globalFineFactor === 1;
    // Analytic dam/tank scenes can construct compact topology and first fine seeds
    // phi without allocating or uploading a box-sized bootstrap texture.
    // Explicitly seeded brick geometry is not one of those closed-form shapes,
    // so it joins terrain and rigid bodies on the dense bootstrap path: the
    // host rasterizes `initialOctreeLevelSet` once and `topologyResidency`
    // publishes exact t=0 residency from that imported SDF.
    // A reservoir dragged off the container corner is not one of those closed
    // forms either: `analyticInitialPhi` anchors the block at the container
    // minimum, so an authored origin would be silently ignored on the GPU while
    // the host honoured it. It joins the dense bootstrap path instead.
    // An authored ball of liquid is not a closed form either, for the same
    // reason: `analyticInitialPhi` knows a corner block and a fill plane, so a
    // sphere scene would run the box on the GPU while the host believed it had
    // authored a ball.
    const analyticSparseBootstrap = (scene.fluid.initialBrickSeeds_m?.length ?? 0) === 0
      && scene.rigidBodies.length === 0 && !sceneHasTerrain(scene)
      && !sceneDamBreakIsOffsetFromCorner(scene) && !sceneHasInitialLiquidVolumes(scene);
    this.analyticSparseBootstrap = analyticSparseBootstrap;
    const surfaceStateAllocation = planOctreeSurfaceStateAllocation(
      [dims.nx, dims.ny, dims.nz],
      scene.rigidBodies.length === 0 && !sceneHasTerrain(scene),
      analyticSparseBootstrap,
    );
    const cell = {
      x: scene.container.width_m / dims.nx,
      y: scene.container.height_m / dims.ny,
      z: scene.container.depth_m / dims.nz
    };
    const spacing = [cell.x, cell.y, cell.z];
    if (spacing.some((value) => !Number.isFinite(value) || value <= 0)
      || Math.max(...spacing) / Math.min(...spacing) > 1 + 1e-5) {
      throw new RangeError("Power catalog requires isotropic finest cells");
    }
    const maximumStorageBinding = Math.min(
      device.limits.maxStorageBufferBindingSize,
      device.limits.maxBufferSize,
    );
    const fluidFootprint = planOctreeFluidFootprintBudget(scene, dims);
    const plannedPressureCapacity = planOctreePressureCapacity(
      dims, this.topologyMaximumLeafSize, this.interfaceRefinementBandCells,
      options.pressureRowCapacity,
      scene.container.top === "closed",
      scene.container.fillFraction,
      Number.MAX_SAFE_INTEGER,
      fluidFootprint,
    );
    const deviceRowLimit = octreeDeviceRowCapacityCeiling({
      dimensions: [dims.nx, dims.ny, dims.nz],
      maximumStorageBindingBytes: maximumStorageBinding,
      plannedRowCapacity: plannedPressureCapacity.rowCapacity,
    });
    if (deviceRowLimit < 1) {
      throw new RangeError("Octree row authorities cannot fit one row in the storage binding limit");
    }
    this.pressureCapacity = planOctreePressureCapacity(
      dims, this.topologyMaximumLeafSize, this.interfaceRefinementBandCells,
      options.pressureRowCapacity,
      scene.container.top === "closed",
      scene.container.fillFraction,
      deviceRowLimit,
      fluidFootprint,
    );
    this.frontierAllocation = planOctreeLeafFrontierAllocation(
      count,
      this.pressureCapacity.rowCapacity,
    );
    this.useLocalFrontierCandidateSort = this.frontierAllocation.listCapacity <= 4096;
    this.useCooperativeRowDeltaRing = this.frontierAllocation.listCapacity
      <= device.limits.maxComputeWorkgroupsPerDimension;
    this.linearBlocks = Math.ceil(this.frontierAllocation.listCapacity / 256);
    reportAllocation(1);
    // Open ocean scenes have no solid fraction to publish. Keep a single
    // zero-initialized record so every bind group remains valid; shader-side
    // bounds checks make all logical cells read as `{0,-1}` and rasterization
    // is skipped. Terrain/body scenes retain the dense VOS field.
    const solidCellAllocation = planOctreeSolidCellAllocation(dims, sceneHasTerrain(scene), scene.rigidBodies.length);
    this.hasDenseSolidCells = solidCellAllocation.hasDenseField;
    this.solidCells = device.createBuffer({
      label: this.hasDenseSolidCells ? "Octree VOS solid fractions and owners" : "Octree zero-solid sentinel",
      size: solidCellAllocation.allocatedBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (!this.hasDenseSolidCells) device.queue.writeBuffer(this.solidCells, 0, new Int32Array([0, -1]));
    // Build residency before the surface state so phi transport can consume
    // the previous publication's active-brick worklist directly. The t=0
    // publication is encoded before the first advance, so the first dynamic
    // surface pass is sparse as well.
    const topologyHaloCells = this.interfaceRefinementBandCells;
    // The delta partition can fail to partition, and shrinking it is not yet a
    // safe fix. Tiles are the granularity of every exact structural delta in the
    // topology path, and sizing them by the AUTHORED leaf gives a 32 x 16 x 32
    // domain a lattice of ONE tile -- any change anywhere marks the whole domain
    // dirty, so the incremental path is a full rebuild that also pays the
    // delta's classification and compaction. droplet-256 gets 8 x 8 x 8 = 512
    // tiles from the same authored leaf, which is why the design looks sound
    // there and this is invisible at that scale.
    //
    // Sizing by the largest leaf the domain can hold gives that lane 4 tiles,
    // and it is DEFAULT OFF because it measured inert: the D4 oracle reported
    // contactSteps {} -- no wall reached in 250 steps -- with every symmetry
    // hook at maximumObserved 0 because nothing moves, and validation still
    // clean. Two mechanisms are already ruled out without a GPU: the analytic
    // bootstrap bounds are healthy at tile 16 (4 active tiles, not 0) and
    // planOctreeCompactionAllocation is sane. The flag exists so the remaining
    // localization costs one run rather than a red tree.
    // The tile ABI is the exact tiling, NOT the leaf ceiling. Deriving it from
    // the ceiling once the ceiling stopped being gcd-bound would have grown the
    // retention tile with it -- a 24x18x16 domain would become a single 32-cell
    // tile, so any evidence anywhere would pin the whole box fine for three
    // generations and undo exactly the coarsening this change buys. Power
    // retains its frozen authored-leaf default and the diagnostic clamp.
    //
    // The dependency runs the other way instead: `octreeLosassoLeafCeiling`
    // clamps the leaf to this edge, because the delta refine/grading passes are
    // dispatched per active tile and reach a coarse candidate only through the
    // tile that holds its origin.
    this.topologyTileSize = this.lane.topologyTileSize(this.maxLeafSize,
      this.effectiveLeafSize, this.losassoExactTilingLeafSize, this.topologyMaximumLeafSize);
    const allocateSparseWorld = octreeSparseWorldRequired(sceneHasTerrain(scene), scene.rigidBodies.length);
    const sparseWorldBrickSize = scene.voxelDomain.brickSize_cells;
    if (allocateSparseWorld) this.sparseBrickWorld = createOctreeSparseResidencyWorld(device, scene, [dims.nx, dims.ny, dims.nz], {
      brickSize: sparseWorldBrickSize,
      environmentBrickRefinementLevels: options.environmentBrickRefinementLevels,
      haloCells: topologyHaloCells,
      // Canonical faces/pages own the simulation fields. Retain only the wet
      // bulk worklist needed by owner-page lifecycle.
      bulkResidency: true,
      brickPreActivation: true,
      topologyTileBricks: this.topologyTileSize / sparseWorldBrickSize,
      includePressureBoundarySupport: true,
      pressureBoundaryTopClosed: scene.container.top === "closed",
      includeWholeDomainPressureSupport: scene.fluid.inflow !== undefined,
      // Tied to the refinement policy, never authored separately: the
      // scheduler must retain a dry wall tile exactly when `refineLeaf` would
      // split one. When these two disagreed, the scheduler won and published
      // topology for leaves that never existed.
      fluidGatedBoundarySupport: this.fluidGatedBoundaryRefinement,
    });
    reportAllocation(2);
    const analyticBootstrapPlan = analyticSparseBootstrap ? planOctreeAnalyticBootstrapBounds({
      dimensions: [dims.nx, dims.ny, dims.nz],
      containerSize: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
      tileSizeCells: this.topologyTileSize,
      initialCondition: scene.fluid.initialCondition,
      fillFraction: scene.container.fillFraction,
      ...(scene.fluid.initialDamBreakDimensions_m ? { damBreakDimensions: [
        scene.fluid.initialDamBreakDimensions_m.x,
        scene.fluid.initialDamBreakDimensions_m.y,
        scene.fluid.initialDamBreakDimensions_m.z,
      ] as const } : {}),
      interfaceBandCells: this.interfaceRefinementBandCells,
    }) : undefined;
    const schedulerBrickDimensions = [dims.nx, dims.ny, dims.nz]
      .map((value) => Math.ceil(value / 8)) as [number, number, number];
    const schedulerTileBricks = this.topologyTileSize / 8;
    const schedulerTileDimensions = schedulerBrickDimensions
      .map((value) => Math.ceil(value / schedulerTileBricks)) as [number, number, number];
    const sparseSchedulerPools = !allocateSparseWorld ? planFineSeedCandidateResidencyPools(
      schedulerBrickDimensions,
      schedulerTileDimensions,
      8,
      this.interfaceRefinementBandCells,
      this.pressureCapacity.rowCapacity,
      analyticBootstrapPlan?.activeTileCount ?? 1,
      true,
    ) : undefined;
    this.topologyResidency = this.sparseBrickWorld?.topologyResidency ?? new GPUFluidBrickResidency(
      device, [dims.nx, dims.ny, dims.nz], [cell.x, cell.y, cell.z], {
        brickSize: 8, haloCells: topologyHaloCells, retireAfterFrames: 3,
        topologyTileBricks: this.topologyTileSize / 8,
        // Pressure topology owns the whole wet volume, not just the refined
        // interface sheet. The sparse-world path already retains the interior
        // (`bulkResidency`); this fallback scheduler must match it, or every
        // deep-liquid tile more than one tile below the free surface publishes
        // no topology tile and its cells decode as unmapped owner pages.
        includeLiquidInterior: true,
        includePressureBoundarySupport: true,
        pressureBoundaryTopClosed: scene.container.top === "closed",
        includeWholeDomainPressureSupport: scene.fluid.inflow !== undefined,
        fluidGatedBoundarySupport: this.fluidGatedBoundaryRefinement,
        // Direct page candidates consume no sparse-world leaf publication.
        // Keep only format-valid sentinel words for those bindings.
        fineSeedCandidatesOnly: true,
        fineSeedCandidateBrickCapacity: sparseSchedulerPools?.brickCapacity,
        fineSeedCandidateTileCapacity: sparseSchedulerPools?.tileCapacity,
        deferPipelineCompilation,
      },
    );
    reportAllocation(3);
    if (analyticBootstrapPlan) {
      const bootstrapPlan = analyticBootstrapPlan;
      const minimum = bootstrapPlan.activeTileLimits.minimum;
      if (minimum[0] !== 0 || minimum[1] !== 0 || minimum[2] !== 0) {
        throw new Error("Analytic octree bootstrap requires an origin-anchored compact tile range");
      }
      this.analyticBootstrapWorklist = new WebGPUOctreeAnalyticBootstrapWorklist(
        device,
        this.topologyResidency.tileWorklist,
        this.topologyResidency.topologyTileStateBuffer,
        {
          tileDimensions: bootstrapPlan.tileDimensions,
          activeTileLimits: bootstrapPlan.activeTileLimits.maximumExclusive,
          tileSizeCells: bootstrapPlan.tileSizeCells,
          activeTileCount: bootstrapPlan.activeTileCount,
          sparseStateCapacity: this.topologyResidency.allocationPlan.sparseKeyPools
            ? this.topologyResidency.tilePublicationCapacity : undefined,
        },
        deferPipelineCompilation,
      );
    }
    this.surfaceState = new WebGPUQuadtreeSurfaceState(
      device, dims, cell, undefined,
      analyticSparseBootstrap
        ? new Float32Array([Math.max(cell.x, cell.y, cell.z) * this.topologyMaximumLeafSize])
        : initialOctreeLevelSet(scene, dims, cell), undefined,
      undefined, false, false, true, true, this.hasDenseSolidCells ? this.solidCells : undefined, {
        worklist: this.topologyResidency.worklist,
        states: this.topologyResidency.stateBuffer,
        brickSize: 8
      }, true, analyticSparseBootstrap
    );
    reportAllocation(4);
    // COPY_SRC on the sparse owner arena and pressure iterates exists solely for test
    // readbacks (leaf-size census, 2:1 balance, and finiteness audits); the
    // simulation itself never copies them out.
    // The frontier is allocated before owner pages because its GPU-resident
    // active generation is the sole clock for the next owner-page candidate.
    // Residency worklist generations describe their producer, not the epoch
    // that will consume the carried page set.
    this.leafFrontier = device.createBuffer({
      label: "Persistent octree leaf frontier",
      size: this.frontierAllocation.allocatedBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.ownerPages = new WebGPUOctreeSimulationOwnerPages(
      device, [dims.nx, dims.ny, dims.nz],
      {
        // Derive physical owner storage from the same bounded adaptive
        // authorities that can request it. Arena overflow is already part
        // of topologyOverflow(); missing pages decode as canonical coarse
        // owners instead of reading outside the physical payload.
        adaptiveBounds: {
          pressureRowCapacity: this.pressureCapacity.rowCapacity,
          fineSeedLeafCapacity: this.pressureCapacity.rowCapacity,
        },
      },
      this.analyticBootstrapWorklist ? {
        tileWorklist: this.topologyResidency.tileWorklist,
        tileSizeCells: this.analyticBootstrapWorklist.plan.tileSizeCells,
        activeTileLimits: analyticBootstrapPlan!.ownerPageTileLimits.maximumExclusive,
        activeTileCount: analyticBootstrapPlan!.ownerPageTileCount,
      } : undefined,
      {
        tileWorklist: this.topologyResidency.tileWorklist,
        tileSizeCells: this.topologyTileSize,
        tileListCapacity: this.topologyResidency.tilePublicationCapacity,
        candidateGeneration: {
          buffer: this.leafFrontier,
          offsetWords: 3,
          frontierListCapacity: this.frontierAllocation.listCapacity,
        },
      },
      deferPipelineCompilation,
    );
    this.topology = this.ownerPages.arena;
    const pressureSlots = this.pressureCapacity.rowCapacity;
    this.pressureA = device.createBuffer({ label: "Octree leaf pressure A", size: Math.max(4, pressureSlots * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.pressureB = device.createBuffer({ label: "Octree leaf pressure B", size: Math.max(4, pressureSlots * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    // The scan totals are dead after leaf emission. The tail then doubles as
    // twelve resident rank-six generalized body-response vectors, avoiding a
    // ninth storage binding on minimum-limit WebGPU devices.
    // Change-driven rebuild state (per-tile change flags, dirty marks, and
    // the compacted dirty list) occupies an exclusive additive tail so the
    // per-solve scan partials can never clobber it, followed by the 8-byte
    // PCG residual feedback staged out via solveStats.
    const tileCapacity = this.topologyResidency.tileCapacity;
    const compactionAllocation = planOctreeCompactionAllocation(
      dims,
      this.pressureCapacity.rowCapacity,
      this.topologyResidency.tileWorklistByteLength,
      tileCapacity,
      this.topologyTileSize,
    );
    this.compactionByteLength = compactionAllocation.allocatedBytes;
    this.topologyTileChangeFlagsOffsetBytes = compactionAllocation.tileChangeFlagsOffsetWords * 4;
    this.topologyTileChangeFlagsByteLength = tileCapacity * 4;
    this.compactionAllocationRowDeltaScratchOffsetBytes =
      compactionAllocation.rowDeltaScratchBaseWords * 4;
    this.dirtyFailureOffsetBytes = compactionAllocation.dirtyFailureOffsetWords * 4;
    this.frontierPublicationOffsetBytes =
      (compactionAllocation.frontierTopologyReuseWord - 13) * 4;
    this.dirtyAuthorityStateOffsetBytes = this.frontierPublicationOffsetBytes - 147 * 4;
    this.compaction = device.createBuffer({
      label: "Octree leaf compaction and resident topology worklist",
      size: this.compactionByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    // The common projection layout is constructed before the optional global
    // fine hierarchy.  An unpublished 64-byte directory keeps the binding
    // valid for non-fine configurations; it can never authorize coarsening.
    this.unpublishedFineSummaryDirectory = device.createBuffer({ label: "Unpublished fine-summary directory",
      size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.unpublishedFineSummaryDirectory, 0, new Uint32Array(16));
    // Copy-only staging keeps solve feedback readable without racing the next
    // rebuild's worklist copy and without a ninth storage binding.
    this.solveStats = device.createBuffer({
      label: "Octree solve feedback staging",
      size: 32,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.leafHeaders = device.createBuffer({ label: "Octree leaf row headers", size: Math.max(48, this.pressureCapacity.headerBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.candidateLeafHeaders = device.createBuffer({ label: "Inactive octree leaf row headers",
      size: Math.max(48, this.pressureCapacity.headerBytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.candidatePressure = device.createBuffer({ label: "Inactive octree remapped pressure seed",
      size: Math.max(4, pressureSlots * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.frontierSortScratch = device.createBuffer({
      label: "Cold frontier merge-sort ping/pong scratch",
      size: Math.max(4, this.frontierAllocation.listCapacity * 4),
      usage: GPUBufferUsage.STORAGE,
    });
    this.solveDispatch = device.createBuffer({ label: "Octree leaf solve and retired-topology dispatch", size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    device.queue.writeBuffer(this.solveDispatch, 32, new Uint32Array([0, 1, 1, 0, 0, 1, 1, 0]));
    // Words 8..15 hold one-workgroup-per-tile coarse topology dispatches: the
    // per-frame copies refresh only the x counts, so y/z stay 1 from here.
    this.topologyCandidateDispatch = device.createBuffer({
      label: "Octree topology, frontier, and row-delta dispatch",
      size: 48,
      // COPY_SRC is failure-path only: the compact schedules are unreadable
      // once a rejection has already consumed them.
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
    });
    this.params = device.createBuffer({
      label: "Octree projection parameters",
      // The fixed 160-byte head, plus the authored refinement-region tail. The
      // diagnostic shader declares only the head and binds the same buffer,
      // which is legal: a uniform binding may be larger than the struct.
      size: OCTREE_REFINEMENT_REGION_PARAMS_OFFSET + OCTREE_REFINEMENT_REGION_PARAMS_BYTES
        + OCTREE_COLD_AUTHORED_SURFACE_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const projectionActivityProfile = performanceShaderVariant();
    this.projectionActivity = createGPULogicalActivityAdoptionContext({
      moduleId: OCTREE_PROJECTION_ACTIVITY_MODULE_ID,
      profile: projectionActivityProfile,
    });
    for (const descriptor of Object.values(OCTREE_PROJECTION_ACTIVITY_TASKS)) {
      this.projectionActivity.describeTask(descriptor.task, {
        id: descriptor.id,
        label: descriptor.label,
        phaseId: descriptor.phaseId,
      });
    }
    // The recurring topology family omits the cold merge-sort scratch binding,
    // keeping its explicit layout at nine compute-visible storage buffers.
    // The only kernel that reaches binding 9 has the exact sort layout below.
    this.layout = device.createBindGroupLayout({ entries: [
      ...projectionBufferLayoutEntries(OCTREE_PROJECTION_CORE_BUFFER_LAYOUT),
      ...OCTREE_PROJECTION_CORE_TEXTURE_LAYOUT.map(({ binding, viewDimension }) => ({
        binding, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" as const, viewDimension },
      })),
    ] });
    this.frontierSortLayout = device.createBindGroupLayout({
      entries: projectionBufferLayoutEntries(OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT),
    });
    const activityLayoutSuffix = this.projectionActivity.enabled ? [
      device.createBindGroupLayout({ entries: [] }),
      device.createBindGroupLayout({ entries: [] }),
      device.createBindGroupLayout({ entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      }] }),
    ] : [];
    this.pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.layout, ...activityLayoutSuffix],
    });
    this.frontierSortPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.frontierSortLayout, ...activityLayoutSuffix],
    });
    const projectionShaderVariant = this.projectionActivity.module(
      octreeProjectionActivityShader(this.projectionActivity, this.lane.wgsl.projectionShader),
      `${OCTREE_PROJECTION_ACTIVITY_MODULE_ID}/${projectionActivityProfile.cacheKey}`,
    );
    this.projectionActivityShaderKey = this.projectionActivity.enabled
      ? projectionShaderVariant.cacheKey : undefined;
    this.projectionShaderCode = projectionShaderVariant.code;
    this.groups = {
      ab: this.createProjectionGroup(this.pressureA, this.pressureB),
      ba: this.createProjectionGroup(this.pressureB, this.pressureA),
    };
    this.candidateRowGroups = {
      fromA: this.createProjectionGroup(this.pressureA, this.candidatePressure, undefined, this.candidateLeafHeaders),
      fromB: this.createProjectionGroup(this.pressureB, this.candidatePressure, undefined, this.candidateLeafHeaders),
    };
    this.fineSummarySizingGroup = this.createProjectionGroup(
      this.unpublishedFineSummaryDirectory, this.pressureB);
    reportAllocation(5);
    const frontierSortStageCount = this.useLocalFrontierCandidateSort
      ? 0 : Math.ceil(Math.log2(Math.max(1, this.frontierAllocation.listCapacity))) + 1;
    const frontierSortStageStride = Math.max(
      16, this.device.limits.minUniformBufferOffsetAlignment ?? 256,
    );
    this.frontierSortStageParams = frontierSortStageCount > 0
      ? this.device.createBuffer({
        label: "Immutable frontier merge-sort stage records",
        size: frontierSortStageCount * frontierSortStageStride,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      : undefined;
    if (this.frontierSortStageParams) {
      const records = new Uint32Array(frontierSortStageCount * frontierSortStageStride / 4);
      for (let stage = 0; stage < frontierSortStageCount; stage += 1) {
        records[stage * frontierSortStageStride / 4] = stage;
      }
      this.device.queue.writeBuffer(this.frontierSortStageParams, 0, records);
    }
    this.frontierSortGroups = this.frontierSortStageParams
      ? Array.from({ length: frontierSortStageCount }, (_, stage) => this.device.createBindGroup({
        layout: this.frontierSortLayout,
        entries: [
          { binding: 2, resource: { buffer: this.compaction } },
          { binding: 3, resource: { buffer: this.topology } },
          { binding: 6, resource: { buffer: this.params } },
          { binding: 7, resource: { buffer: this.frontierSortStageParams!,
            offset: stage * frontierSortStageStride, size: 16 } },
          { binding: 9, resource: { buffer: this.frontierSortScratch } },
          { binding: 13, resource: { buffer: this.leafFrontier } },
        ],
      }))
      : [];
    this.diagnosticLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32uint", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    this.diagnosticPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.diagnosticLayout] });
    const coldRecords: number[] = [
      Math.ceil(dims.nx / 4), Math.ceil(dims.ny / 4), Math.ceil(dims.nz / 4),
      Math.ceil(dims.nx / 8), Math.ceil(dims.ny / 8), Math.ceil(dims.nz / 8),
    ];
    for (const size of [...new Set(this.coarseRefinementSizes)]) {
      this.coldDispatchOffsetBySize.set(size, coldRecords.length * 4);
      coldRecords.push(Math.ceil(dims.nx / size), Math.ceil(dims.ny / size), Math.ceil(dims.nz / size));
    }
    this.coldDispatch = device.createBuffer({
      label: "Immutable cold topology and diagnostic dispatch records",
      size: coldRecords.length * 4,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.coldDispatch, 0, new Uint32Array(coldRecords));
    const fullyCoarseEstimate = Math.ceil(count / (this.topologyMaximumLeafSize ** 3));
    const approximateLeaves = Math.ceil(count * (1 - this.adaptivity) + fullyCoarseEstimate * this.adaptivity);
    this.info = {
      leafCount: approximateLeaves, pressureSampleCount: approximateLeaves, liquidDofCount: approximateLeaves,
      faceCount: 0, mlsProjectionRowCount: 0, tallSegmentCount: 0, ghostFaceCount: 0,
      maximumNeighborRatio: 2, maximumFluidScale: this.topologyMaximumLeafSize,
      compressionRatio: approximateLeaves / Math.max(1, count),
      allocatedBytes: this.ownerPages.allocatedBytes + this.solidCells.size
        + surfaceStateAllocation.allocatedBytes
        + this.pressureA.size + this.pressureB.size + this.candidatePressure.size
        + this.leafHeaders.size + this.candidateLeafHeaders.size + this.frontierSortScratch.size
        + (this.frontierSortStageParams?.size ?? 0)
        + this.leafFrontier.size + this.compaction.size + this.unpublishedFineSummaryDirectory.size
        + this.solveStats.size + this.solveDispatch.size + this.topologyCandidateDispatch.size
        + this.params.size + this.coldDispatch.size
        + (this.sparseBrickWorld?.allocatedBytes ?? this.topologyResidency.allocatedBytes)
        + (this.analyticBootstrapWorklist?.allocatedBytes ?? 0),
      pressureIterationsUsed: 0,
      pressureIterationBudget: this.solveTailPolicy.encodedOuterIterations,
      pressureIterationHardBudget: this.pressureHardIterationCeiling(),
      pressureConverged: undefined,
      pressureRowCapacity: pressureSlots,
      pressureCapacityOverflow: false,
      frontierListCapacity: this.frontierAllocation.listCapacity,
      frontierCapacityOverflow: false,
      velocityClampCount: 0,
      factorLevelCount: Math.ceil(Math.log2(Math.max(dims.nx, dims.ny, dims.nz))) + 1,
      multigridLevelCount: Math.ceil(Math.log2(Math.max(dims.nx, dims.ny, dims.nz))) + 1,
      multigridCoarsestDofs: 1,
      topologyReadbackBytes: 0,
      topologyReused: false, topologyReuseCount: 0,
      topologyCadenceSkipCount: 0, topologyExactIdentityCount: 0,
      powerDiagramReady: false,
      powerDiagramAuthoritative: false,
      powerDiagramAllocatedBytes: 0,
      globalFineLevelSetAllocatedBytes: 0,
      globalFineLevelSetResidentBrickCapacity: 0,
      globalFineLevelSetLogicalBrickCount: 0,
      globalFineTransportQueryCapacity: 0,
      globalFineTransportChunkCapacity: 0,
      globalFineTransportChunkCount: 0,
      globalFineTransportSegmentCount: 0,
      globalFineTransportEncodedPasses: 0,
      globalFineTransportPrepassScratchBytes: 0,
      globalFineTransportVertexScratchBytes: 0,
    };
    this.surfaceStateAccountingBytes = surfaceStateAllocation.allocatedBytes;
    this.workAccounting.setAuthorityBytes("owner-pages", this.ownerPages.allocatedBytes);
    this.workAccounting.setAuthorityBytes("solid-cells", this.solidCells.size);
    this.workAccounting.setAuthorityBytes("surface-state", this.surfaceStateAccountingBytes);
    this.workAccounting.setAuthorityBytes("pressure-topology-state",
      this.pressureA.size + this.pressureB.size + this.candidatePressure.size
      + this.leafHeaders.size + this.candidateLeafHeaders.size + this.frontierSortScratch.size
      + (this.frontierSortStageParams?.size ?? 0)
      + this.leafFrontier.size + this.compaction.size + this.unpublishedFineSummaryDirectory.size
      + this.solveStats.size + this.solveDispatch.size + this.topologyCandidateDispatch.size
      + this.params.size + this.coldDispatch.size);
    this.workAccounting.setAuthorityBytes("sparse-world",
      this.sparseBrickWorld?.allocatedBytes ?? 0);
    this.workAccounting.setAuthorityBytes("topology-residency",
      this.sparseBrickWorld ? 0 : this.topologyResidency.allocatedBytes);
    this.workAccounting.setAuthorityBytes("analytic-bootstrap",
      this.analyticBootstrapWorklist?.allocatedBytes ?? 0);
    reportAllocation(6);
    this.fineSeedAdapter = new WebGPUOctreeFineSeedAdapter(device, {
        leafHeaders: this.leafHeaders,
        rowCount: this.compaction,
        publicationControl: this.compaction,
        frontier: this.leafFrontier,
        dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
        cellSize: [cell.x, cell.y, cell.z],
        // Fine seed values and band residency are refreshed independently of
        // coarse structural decisions. Do not route the coarse dirty-tile mask
        // into this producer: a valid zero structural delta is not an empty
        // fine-band replacement publication.
      }, this.pressureCapacity.rowCapacity, {
          // Global fine bricks are keyed independently of octree resolution.
          // Every live core/halo leaf may therefore seed the narrow band; a
          // coarse interface leaf must not be discarded merely because its
          // pressure degree of freedom spans more than one finest cell.
          finestLeafSize: this.topologyMaximumLeafSize,
          // Seeding must reach at least as far as the widest band that
          // consumes the seeds. An over-wide halo only produces seeds the
          // redistance cutoff later invalidates; an under-wide one leaves the
          // outer band unseeded and fails the publication closed.
          haloCells: Math.max(this.interfaceRefinementBandCells, this.fineLevelSetBandCells),
          // Always bound; the analytic condition below selects the authority.
          // Without one of the two, no leaf seeds the band at all and every
          // downstream fine/coarse publication is empty.
          bootstrapLevelSet: this.surfaceState.texture,
          openTopBoundary: scene.container.top !== "closed",
          deferPipelineCompilation,
          ...(analyticSparseBootstrap ? {
            analyticInitialCondition: scene.fluid.initialCondition,
            initialFillFraction: scene.container.fillFraction,
            initialDamBreakDimensions: scene.fluid.initialDamBreakDimensions_m ? [
              scene.fluid.initialDamBreakDimensions_m.x,
              scene.fluid.initialDamBreakDimensions_m.y,
              scene.fluid.initialDamBreakDimensions_m.z,
            ] : undefined,
          } : {}),
      });
      this.info.allocatedBytes += this.fineSeedAdapter.plan.allocatedBytes;
      this.workAccounting.setAuthorityBytes("fine-seed-adapter",
        this.fineSeedAdapter.plan.allocatedBytes);
        reportAllocation(7);
        if (!this.coarseOnlySurfaceTracking) {
          if (!this.fineSeedAdapter) {
            throw new RangeError("Global fine level-set authority requires compact fine-seed leaves");
          }
          const minimumCell = Math.min(cell.x, cell.y, cell.z);
          const maximumCell = Math.max(cell.x, cell.y, cell.z);
          if (maximumCell - minimumCell > 1e-5 * maximumCell) {
            throw new RangeError("Global fine level-set authority currently requires isotropic finest octree cells");
          }
          const brickResolution = 4 as const;
          const brickDimensions = [dims.nx, dims.ny, dims.nz]
            .map((value) => Math.ceil(value * globalFineFactor / brickResolution)) as [number, number, number];
          const logicalBrickCount = brickDimensions.reduce((product, value) => product * value, 1);
          // Section 5 transports every sample in the authored narrow band.
          // The resident topology must therefore also hold the complete
          // backtrace and trilinear stencil beyond that band. A 3-D trilinear
          // corner can be sqrt(3) fine cells from the query, so its
          // signed-distance support needs two cells rather than one. An
          // unreachable cutoff sentinel is still rejected by seed identity.
          // These widths are re-derived per step from the same planner; the
          // shared helper is what keeps allocation and encode in agreement.
          const fineBandPlan = planFineLevelSetBandFineCells(
            this.fineLevelSetBandCells, globalFineFactor,
          );
          const { transportBandFineCells, redistanceBandFineCells, maximumBacktraceFineCells }
            = fineBandPlan;
          const physicalBand = planFineLevelSetTopologyBand(brickResolution, {
            maximumBacktraceFineCells,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells,
            safetyBrickRings: 1 + this.coarseDynamics.topology.extraDilationRings,
            transportBandFineCells,
          });
          const capacityDilationBrickRings = Math.max(
            physicalBand.dilationBrickRings,
            planFineLevelSetCapacityDilationBrickRings(
              brickResolution, this.fineLevelSetBandCells, globalFineFactor,
            ) + this.coarseDynamics.topology.extraDilationRings,
          );
          const footprintBrickDimensions = [0, 1, 2].map((axis) => Math.max(1,
            Math.ceil((fluidFootprint.maximumCell[axis]! - fluidFootprint.minimumCell[axis]!)
              * globalFineFactor / brickResolution))) as [number, number, number];
          const inflowFineBricks = Math.ceil(fluidFootprint.inflowLiquidCells
            * (globalFineFactor / brickResolution) ** 3);
          // Losasso transports the authored band from a wider valid donor
          // envelope (backtrace plus trilinear support). In a shallow D4 tank
          // that envelope plus the mandatory publication ring can span the
          // entire short axis; a fractional area reserve merely rejects one
          // dilation round later. Permit the physical plan to reach the full
          // logical lattice (the capacity planner still clamps there). Keep
          // Power historically used 1.5x. The shared candidate-local topology
          // can publish a larger legal interface shell than the old
          // tile-retention topology: on symmetric expansion it grows through
          // 14,400 pages as additional dilation rounds become representable.
          // The same 2x physical-area reserve already used by LoSasso admits
          // that complete shell. Large domains remain sparse because this is
          // an area-times-band-width estimate, clamped only on compact grids.
          const surfaceGrowthSafety = this.lane.fineBandSurfaceGrowthSafety;
          const defaultCapacity = planFluidFootprintFineNarrowBandBrickCapacity(
            brickDimensions, footprintBrickDimensions, capacityDilationBrickRings,
            inflowFineBricks, surfaceGrowthSafety,
          ).maximumResidentBricks;
          // The sparse area estimate intentionally ignores edge/corner terms,
          // while LoSasso's cold publisher dilates the complete authored box.
          // Recurring direct transport then tags the one-page near-zero shell
          // as interface membership before applying the same physical band.
          // Reserve the largest translated envelope plus one deformation ring
          // on each side as a floor. A rigid translated-box proof was 308 pages
          // short by generation 11 of ceiling-slab-drop as its surface settled;
          // this extra ring also covers that Section-5 interface growth without
          // changing the frozen Power allocation policy.
          const bandFloor = this.lane.fineBandBrickFloor({
            brickDimensions,
            minimumBrick: [0, 1, 2].map((axis) => Math.floor(
              fluidFootprint.minimumCell[axis]! * globalFineFactor / brickResolution,
            )) as [number, number, number],
            maximumBrick: [0, 1, 2].map((axis) => Math.ceil(
              fluidFootprint.maximumCell[axis]! * globalFineFactor / brickResolution,
            )) as [number, number, number],
            capacityDilationBrickRings,
          });
          const plannedCapacity = Math.max(defaultCapacity, bandFloor);
          const requestedCapacity = Math.min(logicalBrickCount,
            options.globalFineLevelSetMaximumBricks ?? plannedCapacity);
          const requestedPlan = planFineLevelSetBricks({
            domainOrigin: [0, 0, 0], finestCellDimensions: [dims.nx, dims.ny, dims.nz],
            finestCellWidth: minimumCell, fineFactor: globalFineFactor, brickResolution,
            maximumResidentBricks: requestedCapacity,
          });
          const requestedSummary = planFineLevelSetGPUSummaries(
            requestedPlan, this.pressureCapacity.rowCapacity);
          // Per-brick kernels tile over two dispatch dimensions. Capacity is a
          // physical narrow-band estimate and is clamped only by actual buffer
          // binding feasibility; a true page overflow remains fail-closed.
          const kernelBrickLimit = device.limits.maxComputeWorkgroupsPerDimension;
          const configuredCapacity = resolveGlobalFineBrickCapacity(
            plannedCapacity, options.globalFineLevelSetMaximumBricks, kernelBrickLimit, 64,
            Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize),
            brickResolution ** 3,
            (() => { let levels = 1, levelDims = brickDimensions;
              while (!levelDims.every((value) => value === 1)) {
                levelDims = levelDims.map((value) => Math.ceil(value / 2)) as [number, number, number];
                levels += 1;
              }
              return levels;
            })(),
            requestedSummary.entryCapacity,
          );
          const globalPlan = planFineLevelSetBricks({
            domainOrigin: [0, 0, 0], finestCellDimensions: [dims.nx, dims.ny, dims.nz],
            finestCellWidth: minimumCell, fineFactor: globalFineFactor, brickResolution,
            maximumResidentBricks: Math.min(logicalBrickCount, configuredCapacity),
          });
          this.globalFineLevelSet = new WebGPUFineLevelSetBricks(device, globalPlan);
          this.globalFineSourceA = this.globalFineLevelSet.initializeEmptyGPUGeneration(1);
          this.globalFineSourceB = this.globalFineLevelSet.prepareGPUGeneration(2);
          const brickUnionBounds = initialFluidBrickUnionBounds(
            this.scene, [dims.nx, dims.ny, dims.nz], this.scene.voxelDomain.brickSize_cells,
          );
          const brickComponentBounds = brickUnionBounds ? undefined : initialFluidBrickComponentBounds(
            this.scene, [dims.nx, dims.ny, dims.nz], this.scene.voxelDomain.brickSize_cells,
          );
          // Dense topology is still required around terrain and rigid bodies,
          // but those solids do not make the authored liquid SDF non-analytic.
          // Preserve the exact tank/dam phi for every cold fine halo page so
          // Losasso bootstrap does not depend circularly on its coarse-phi
          // exchange, which is first published after fine redistance.
          const exactAnalyticFineSeed = brickUnionBounds
            ? { initialCondition: "box" as const,
              // Fine SPGrid coordinates are container-local while authored
              // scene x/z coordinates are centred about the tank origin.
              minimum: [brickUnionBounds.minimum.x + 0.5 * this.scene.container.width_m,
                brickUnionBounds.minimum.y,
                brickUnionBounds.minimum.z + 0.5 * this.scene.container.depth_m] as const,
              maximum: [brickUnionBounds.maximum.x + 0.5 * this.scene.container.width_m,
                brickUnionBounds.maximum.y,
                brickUnionBounds.maximum.z + 0.5 * this.scene.container.depth_m] as const }
            : brickComponentBounds
              ? { initialCondition: "boxes" as const, boxes: brickComponentBounds.map((bounds) => ({
                minimum: [bounds.minimum.x + 0.5 * this.scene.container.width_m,
                  bounds.minimum.y,
                  bounds.minimum.z + 0.5 * this.scene.container.depth_m] as const,
                maximum: [bounds.maximum.x + 0.5 * this.scene.container.width_m,
                  bounds.maximum.y,
                  bounds.maximum.z + 0.5 * this.scene.container.depth_m] as const,
              })) }
            : (this.scene.fluid.initialBrickSeeds_m?.length ?? 0) === 0
            && !sceneDamBreakIsOffsetFromCorner(this.scene)
            && !sceneHasInitialLiquidVolumes(this.scene)
            ? { initialCondition: this.scene.fluid.initialCondition,
              fillFraction: this.scene.container.fillFraction,
              damBreakDimensions: this.scene.fluid.initialDamBreakDimensions_m
                ? [this.scene.fluid.initialDamBreakDimensions_m.x,
                  this.scene.fluid.initialDamBreakDimensions_m.y,
                  this.scene.fluid.initialDamBreakDimensions_m.z] as const
                : undefined }
            : undefined;
          this.globalFineSeeds = new WebGPUFineLevelSetLeafSeeds(
            device, this.globalFineSourceB, exactAnalyticFineSeed, {
              maximumSourceLeaves: this.pressureCapacity.rowCapacity,
            }, deferPipelineCompilation);
          this.globalFineSummaries = new WebGPUFineLevelSetSummaries(device, globalPlan,
            this.pressureCapacity.rowCapacity, deferPipelineCompilation);
          // The core topology layout is already at the activity-eligible
          // nine-storage budget. Refinement reuses pressure binding 4 for this
          // raw read-only directory instead of adding a tenth binding.
          this.fineSummarySizingGroup = this.createProjectionGroup(
            this.globalFineSummaries.directory, this.pressureB);
          const allocated = this.globalFineLevelSet.allocatedBytes + this.globalFineSeeds.allocatedBytes
            + this.globalFineSummaries.plan.allocatedBytes;
          this.info.allocatedBytes += allocated;
          this.info.globalFineLevelSetAllocatedBytes += allocated;
          this.workAccounting.setAuthorityBytes("fine-page-pool", allocated);
          this.info.globalFineLevelSetResidentBrickCapacity = globalPlan.maximumResidentBricks;
          this.info.globalFineLevelSetLogicalBrickCount = globalPlan.logicalBrickCount;
        }
    this.lane.constructAuthority();
    reportAllocation(8);
    this.writeParams();
  }

  pressureHardIterationCeiling(): number {
    return this.lane.pressureHardIterationCeiling()
      ?? this.solveTailPolicy.hardOuterIterationCeiling;
  }

  createProjectionGroup(
    pressureIn: GPUBuffer | GPUBufferBinding,
    pressureOut: GPUBuffer | GPUBufferBinding,
    binding15Override?: GPUBuffer,
    leafHeadersOverride: GPUBuffer = this.leafHeaders,
  ): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 2, resource: { buffer: this.compaction } },
      { binding: 3, resource: { buffer: this.topology } },
      { binding: 4, resource: "buffer" in pressureIn ? pressureIn : { buffer: pressureIn } },
      { binding: 5, resource: "buffer" in pressureOut ? pressureOut : { buffer: pressureOut } },
      { binding: 6, resource: { buffer: this.params } },
      { binding: 8, resource: { buffer: leafHeadersOverride } },
      { binding: 10, resource: { buffer: this.resources.rigidBodies } },
      { binding: 11, resource: { buffer: this.solidCells } },
      { binding: 12, resource: this.resources.terrain.createView() },
      { binding: 13, resource: { buffer: this.leafFrontier } },
      // Re-seeding rewrites this texture in place, so the view stays valid.
      { binding: 14, resource: this.surfaceState.texture.createView({ dimension: "3d" }) },
      { binding: 15, resource: { buffer: binding15Override
        ?? this.sparseBrickWorld?.bulkResidencyWorklist
        ?? this.topologyResidency.worklist } },
    ] });
  }

  private descriptor(entryPoint: string): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: {
      module: this.requireProjectionShader(),
      entryPoint,
      constants: this.pipelineConstants(this.topologyCandidateEntryPoint(entryPoint)),
    } };
  }
  private refinementDescriptor(entryPoint: string, size: number): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.requireProjectionShader(), entryPoint, constants: {
      ...this.pipelineConstants(true), targetRefinementSize: size,
    } } };
  }
  private frontierSortDescriptor(): GPUComputePipelineDescriptor {
    return {
      layout: this.frontierSortPipelineLayout,
      compute: {
        module: this.requireProjectionShader(),
        entryPoint: "sortFrontierCandidates",
        constants: this.pipelineConstants(true),
      },
    };
  }
  private topologyCandidateEntryPoint(entryPoint: string): boolean {
    return /^(?:rasterizeSolids|resetTopology|refineTopology|balanceTopology|stampFrontier|beginFrontier|classifyFrontier|scanFrontier|prefixFrontier|emitFrontier|prepareFrontier|sortFrontier|mergeFrontier|finalizeFrontier|planLeaves|emitLeaves|markRowDeltaRing)/.test(entryPoint);
  }
  private pipelineConstants(candidateTopology = false): Record<string, number> {
    return {
      rowIndexedPressure: 1,
      sparseTopologyTileStates: this.topologyResidency.allocationPlan.sparseKeyPools ? 1 : 0,
      denseSolidField: this.hasDenseSolidCells ? 1 : 0,
      fluidGatedBoundaryRefinement: this.fluidGatedBoundaryRefinement ? 1 : 0,
      topologyCandidateView: candidateTopology ? 1 : 0,
      fineSummaryFactor: this.coarseOnlySurfaceTracking
        ? 1 : this.globalFineLevelSet?.plan.fineFactor ?? 4,
      adaptiveCoarseSurface: this.coarseOnlySurfaceTracking ? 1 : 0,
      topologyTileCells: this.topologyTileSize,
      gradingPageFill: octreeGradingPageFillEnabled() ? 1 : 0,
      gradingSplitHelpers: octreeGradingSplitHelpersEnabled() ? 1 : 0,
      gradingMembershipLoad: octreeGradingMembershipLoadEnabled() ? 1 : 0,
    };
  }
  private diagnosticDescriptor(): GPUComputePipelineDescriptor {
    this.diagnosticShader ??= this.device.createShaderModule({
      label: "GPU octree overlay materialization",
      code: octreeDiagnosticShader,
    });
    return { layout: this.diagnosticPipelineLayout, compute: { module: this.diagnosticShader, entryPoint: "materializeOctreeFields", constants: { rowIndexedPressure: 1 } } };
  }

  private requireProjectionShader(): GPUShaderModule {
    if (!this.shader) throw new Error("Octree projection shader module has not been initialized");
    return this.shader;
  }

  private createProjectionShaderModule(): void {
    this.shader ??= this.device.createShaderModule({
      label: "GPU-resident octree projection",
      code: this.projectionShaderCode,
    });
  }

  private shaderCapabilities(diagnosticOverlays = false): GPUShaderCapabilityPlan {
    return planGPUShaderCapabilities(this.scene, {
      solver: "octree",
      fineInterface: Boolean(this.globalFineLevelSet),
      distributedFrontierSort: !this.useLocalFrontierCandidateSort,
      diagnosticOverlays,
      logicalActivity: this.projectionActivity.enabled,
    });
  }

  private registerProjectionPipeline<T extends GPUComputePipeline>(pipeline: T): T {
    return this.projectionActivity.registerPipeline(pipeline);
  }

  private static readonly pipelineEntryPoints = OCTREE_PROJECTION_BASE_ENTRY_POINTS;

  private assignPipelines(compiled: GPUComputePipeline[]) {
    [
      this.rasterizeSolidsPipeline, this.resetPipeline, this.refinePipeline, this.balancePipeline,
      this.rasterizeSolidsDeltaPipeline, this.resetDeltaPipeline, this.refineDeltaPipeline, this.balanceDeltaPipeline,
      this.stampFrontierAttemptPipeline, this.beginFrontierPipeline,
      this.classifyFrontierCandidatesPipeline, this.classifyFrontierCandidatesDeltaPipeline,
      this.prefixFrontierCandidateBlocksPipeline, this.prefixFrontierCandidateBlocksDeltaPipeline,
      this.emitFrontierCandidatesPipeline, this.emitFrontierCandidatesDeltaPipeline,
      this.prepareFrontierDispatchPipeline, this.sortFrontierCandidatesLocalPipeline,
      this.classifyFrontierCarryPipeline, this.scanFrontierCarryBlocksPipeline,
      this.prefixFrontierCarryBlocksPipeline, this.mergeFrontierRowsPipeline, this.finalizeFrontierPipeline,
      this.prepareRowDeltaPipeline,
      this.classifyRowDeltaPipeline, this.finalizeRowDeltaClassificationPipeline,
      this.scanDirtyRowDeltaBlocksPipeline, this.prefixDirtyRowDeltaBlocksPipeline,
      this.scatterDirtyRowDeltaPipeline, this.markRowDeltaRingPipeline,
      this.markRowDeltaRingBlocksPipeline,
      this.scanAffectedRowDeltaBlocksPipeline, this.prefixAffectedRowDeltaBlocksPipeline,
      this.compactRowDeltaPipeline, this.publishRowDeltaPipeline, this.publishReusedRowDeltaPipeline,
      this.planPipeline, this.scanPipeline, this.emitPipeline,
      this.classifyTopologyTileSignaturePipeline, this.buildDirtyTileDeltaPipeline,
      this.buildDirtyFrontierDeltaPipeline
    ] = compiled;
  }

  private pipelineReachability(): OctreeProjectionPipelineReachability {
    return {
      solidRasterization: this.hasDenseSolidCells,
      localFrontierCandidateSort: this.useLocalFrontierCandidateSort,
      cooperativeRowDeltaRing: this.useCooperativeRowDeltaRing,
    };
  }

  private basePipelineRequired(entryPoint: string) {
    return octreeProjectionPipelineRequired(entryPoint, this.pipelineReachability());
  }

  private pipelineCacheKey() {
    const stableEntries = (values: object) => Object.entries(values)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0);
    const reachability = this.pipelineReachability();
    return JSON.stringify({
      constants: stableEntries(this.pipelineConstants()),
      candidateConstants: stableEntries(this.pipelineConstants(true)),
      reachability: stableEntries(reachability),
      shaderCapabilities: this.shaderCapabilities().cacheKey,
      maximumLeafSize: this.maxLeafSize,
      topologyMaximumLeafSize: this.topologyMaximumLeafSize,
      effectiveLeafSize: this.effectiveLeafSize,
      coarseBackend: this.coarseDynamics.backend,
      requiredEntryPoints: WebGPUOctreeProjection.pipelineEntryPoints
        .filter((entryPoint) => octreeProjectionPipelineRequired(entryPoint, reachability)),
      ...(this.projectionActivityShaderKey
        ? { activityShader: this.projectionActivityShaderKey }
        : {}),
    });
  }

  private applyPipelineCache(entry: OctreePipelineCacheEntry) {
    this.assignPipelines(entry.base);
    this.frontierCandidateSortPipelines = entry.frontierSort;
    this.refineLevelPipelines.clear(); entry.refine.forEach((value, key) => this.refineLevelPipelines.set(key, value));
    this.refineCoarsePipelines.clear(); entry.refineCoarse.forEach((value, key) => this.refineCoarsePipelines.set(key, value));
    this.balanceCoarsePipelines.clear(); entry.balanceCoarse.forEach((value, key) => this.balanceCoarsePipelines.set(key, value));
  }

  private publishPipelineCache() {
    let cache = octreePipelineCache.get(this.device);
    if (!cache) { cache = new Map(); octreePipelineCache.set(this.device, cache); }
    cache.set(this.pipelineCacheKey(), {
      base: WebGPUOctreeProjection.pipelineEntryPoints.map((_, index) => [
        this.rasterizeSolidsPipeline, this.resetPipeline, this.refinePipeline, this.balancePipeline,
        this.rasterizeSolidsDeltaPipeline, this.resetDeltaPipeline, this.refineDeltaPipeline, this.balanceDeltaPipeline,
        this.stampFrontierAttemptPipeline, this.beginFrontierPipeline,
        this.classifyFrontierCandidatesPipeline, this.classifyFrontierCandidatesDeltaPipeline,
        this.prefixFrontierCandidateBlocksPipeline, this.prefixFrontierCandidateBlocksDeltaPipeline,
        this.emitFrontierCandidatesPipeline, this.emitFrontierCandidatesDeltaPipeline,
        this.prepareFrontierDispatchPipeline, this.sortFrontierCandidatesLocalPipeline,
        this.classifyFrontierCarryPipeline, this.scanFrontierCarryBlocksPipeline,
        this.prefixFrontierCarryBlocksPipeline, this.mergeFrontierRowsPipeline, this.finalizeFrontierPipeline,
        this.prepareRowDeltaPipeline,
        this.classifyRowDeltaPipeline, this.finalizeRowDeltaClassificationPipeline,
        this.scanDirtyRowDeltaBlocksPipeline, this.prefixDirtyRowDeltaBlocksPipeline,
        this.scatterDirtyRowDeltaPipeline, this.markRowDeltaRingPipeline,
        this.markRowDeltaRingBlocksPipeline,
        this.scanAffectedRowDeltaBlocksPipeline, this.prefixAffectedRowDeltaBlocksPipeline,
        this.compactRowDeltaPipeline, this.publishRowDeltaPipeline, this.publishReusedRowDeltaPipeline,
        this.planPipeline, this.scanPipeline, this.emitPipeline,
        this.classifyTopologyTileSignaturePipeline, this.buildDirtyTileDeltaPipeline,
        this.buildDirtyFrontierDeltaPipeline,
      ][index]),
      frontierSort: [...this.frontierCandidateSortPipelines],
      refine: new Map(this.refineLevelPipelines), refineCoarse: new Map(this.refineCoarsePipelines), balanceCoarse: new Map(this.balanceCoarsePipelines),
    });
  }

  get topologyTexture() { return this.topologyDiagnosticTexture; }
  get coarseBackend() { return this.coarseDynamics.backend; }
  get pressureSamplesTexture() { return this.pressureSamplesDiagnosticTexture; }
  get pressureTexture() { return this.pressureDiagnosticTexture; }
  get hasDiagnosticTextures() { return this.diagnosticGroups !== undefined; }

  /** Allocate the dense scientific-overlay fields only after inspection asks for them. */
  async ensureDiagnosticTextures(): Promise<boolean> {
    if (this.diagnosticGroups) return false;
    this.materializePipeline = octreeDiagnosticPipelineCache.get(this.device);
    if (!this.materializePipeline) {
      this.materializePipeline = await this.device.createComputePipelineAsync(this.diagnosticDescriptor());
      octreeDiagnosticPipelineCache.set(this.device, this.materializePipeline);
    }
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    const size: GPUExtent3D = [this.dims.nx, this.dims.ny, this.dims.nz];
    this.topologyDiagnosticTexture = this.device.createTexture({ label: "Octree overlay topology", size, dimension: "3d", format: "rg32uint", usage });
    this.pressureSamplesDiagnosticTexture = this.device.createTexture({ label: "Octree overlay pressure ownership", size, dimension: "3d", format: "rgba32uint", usage });
    this.pressureDiagnosticTexture = this.device.createTexture({ label: "Octree mapped leaf pressure", size, dimension: "3d", format: "r32float", usage });
    const diagnosticGroup = (pressure: GPUBuffer | GPUBufferBinding) => this.device.createBindGroup({ layout: this.diagnosticLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } },
      { binding: 1, resource: "buffer" in pressure ? pressure : { buffer: pressure } },
      { binding: 4, resource: this.topologyDiagnosticTexture!.createView() },
      { binding: 5, resource: this.pressureSamplesDiagnosticTexture!.createView() },
      { binding: 6, resource: this.pressureDiagnosticTexture!.createView() },
      { binding: 8, resource: { buffer: this.params } },
      { binding: 11, resource: { buffer: this.leafFrontier } }
    ] });
    const framePressure = this.lane.diagnosticPressureBanks;
    this.diagnosticGroups = {
      pressureA: diagnosticGroup(framePressure?.pressureA ?? this.pressureA),
      pressureB: diagnosticGroup(framePressure?.pressureB ?? this.pressureB)
    };
    this.info.allocatedBytes += this.dims.nx * this.dims.ny * this.dims.nz * 28;
    return true;
  }

  initializationTasks(): GPUInitializationTask[] {
    const cached = octreePipelineCache.get(this.device)?.get(this.pipelineCacheKey());
    const entries = WebGPUOctreeProjection.pipelineEntryPoints;
    const tasks: GPUInitializationTask[] = [
      ...this.topologyResidency.initializationTasks(),
      ...(this.analyticBootstrapWorklist?.initializationTasks() ?? []),
      ...this.ownerPages.initializationTasks(),
      ...(this.fineSeedAdapter?.initializationTasks() ?? []),
      ...(this.globalFineSeeds?.initializationTasks() ?? []),
      ...(this.globalFineSummaries?.initializationTasks() ?? []),
      ...(cached
        ? [{ id: "octree.pipeline-cache", phase: "adaptive-topology" as const,
          label: "Reuse compiled adaptive programs", run: () => this.applyPipelineCache(cached) }]
        : []),
    ];
    if (this.sparseBrickWorld) {
      tasks.push({
        id: "octree.sparse-world.pipelines",
        phase: "solver-pipelines",
        label: "Compile sparse voxel publication programs",
        run: () => this.sparseBrickWorld!.initializePipelines(),
      });
    }
    const compiled = new Array<GPUComputePipeline>(entries.length);
    const capabilities = this.shaderCapabilities();
    const shaderDefinitions: GPUShaderTaskDefinition[] = [];
    let lastRequiredBaseIndex = -1;
    if (!cached) entries.forEach((entryPoint, index) => {
      if (this.basePipelineRequired(entryPoint)) lastRequiredBaseIndex = index;
    });
    if (!cached) {
      shaderDefinitions.push({
        id: "octree.shader.projection",
        phase: "adaptive-topology",
        label: "Create reachable octree shader module",
        requires: ["adaptive-topology"],
        compile: () => this.createProjectionShaderModule(),
      });
    }
    if (!cached) entries.forEach((entryPoint, index) => {
      if (!this.basePipelineRequired(entryPoint)) return;
      shaderDefinitions.push({
        id: `octree.pipeline.${entryPoint}`,
        phase: "adaptive-topology",
        label: `Compile octree ${entryPoint}`,
        requires: entryPoint === "rasterizeSolids" || entryPoint === "rasterizeSolidsDelta"
          ? ["solid-fields"] : ["adaptive-topology"],
        dependencies: ["octree.shader.projection"],
        compile: async () => {
          compiled[index] = this.registerProjectionPipeline(
            await this.device.createComputePipelineAsync(this.descriptor(entryPoint)),
          );
          if (index === lastRequiredBaseIndex) {
            // Unreachable tuple slots remain unpublished; the immutable
            // reachability key prevents this sparse tuple from serving a
            // configuration that requires them.
            this.assignPipelines(compiled);
          }
        },
      });
    });
    if (!cached && !this.useLocalFrontierCandidateSort) {
      shaderDefinitions.push({
        id: "octree.pipeline.frontier-sort",
        phase: "adaptive-topology",
        label: "Compile octree frontier merge sort",
        requires: ["distributed-frontier-sort"],
        dependencies: ["octree.shader.projection"],
        compile: async () => {
          this.frontierCandidateSortPipelines = [this.registerProjectionPipeline(
            await this.device.createComputePipelineAsync(this.frontierSortDescriptor()),
          )];
        },
      });
    }
    for (let size = Math.min(8, this.maxLeafSize); size >= 2; size >>= 1) {
      if (cached?.refine.has(size)) continue;
      const level: Partial<OctreePipelineVariants> = {};
      const definitions = [
        ["full", "refineTopology"],
        ["delta", "refineTopologyDelta"],
      ] as const;
      definitions.forEach(([variant, entryPoint], index) => shaderDefinitions.push({
        id: `octree.pipeline.refine.${size}.${variant}`,
        phase: "adaptive-topology",
        label: `Compile octree refinement ${size} · ${variant}`,
        requires: ["adaptive-topology"],
        dependencies: ["octree.shader.projection"],
        compile: async () => {
          level[variant] = this.registerProjectionPipeline(
            await this.device.createComputePipelineAsync(this.refinementDescriptor(entryPoint, size)),
          );
          if (index === definitions.length - 1) {
            this.refineLevelPipelines.set(size, level as OctreePipelineVariants);
          }
        },
      }));
    }
    for (let size = this.effectiveLeafSize; size >= 16; size >>= 1) {
      for (const operation of ["refine", "balance"] as const) {
        if ((operation === "refine" ? cached?.refineCoarse : cached?.balanceCoarse)?.has(size)) continue;
        const pipelines: Partial<OctreePipelineVariants> = {};
        const prefix = operation === "refine" ? "refineTopologyCoarse" : "balanceTopologyCoarse";
        const definitions = [["full", prefix], ["delta", `${prefix}Delta`]] as const;
        definitions.forEach(([variant, entryPoint], index) => shaderDefinitions.push({
          id: `octree.pipeline.${operation}-coarse.${size}.${variant}`,
          phase: "adaptive-topology",
          label: `Compile octree coarse ${operation} ${size} · ${variant}`,
          requires: ["adaptive-topology"],
          dependencies: ["octree.shader.projection"],
          compile: async () => {
            pipelines[variant] = this.registerProjectionPipeline(
              await this.device.createComputePipelineAsync(this.refinementDescriptor(entryPoint, size)),
            );
            if (index === definitions.length - 1) {
              const complete = pipelines as OctreePipelineVariants;
              if (operation === "refine") this.refineCoarsePipelines.set(size, complete);
              else this.balanceCoarsePipelines.set(size, complete);
            }
          },
        }));
      }
    }
    tasks.push(...planGPUShaderTasks(capabilities, shaderDefinitions));
    if (!cached) {
      tasks.push({ id: "octree.pipeline-cache.publish", phase: "adaptive-topology", label: "Publish compiled octree pipelines", run: () => this.publishPipelineCache() });
    } else if (tasks.length > 1) {
      tasks.push({ id: "octree.pipeline-cache.publish", phase: "adaptive-topology", label: "Publish compiled adaptive variants", run: () => this.publishPipelineCache() });
    }
    tasks.push(...this.lane.initializationTasks());
    return tasks;
  }


  async initializePipelines(onProgress: (label: string, completed: number, total?: number) => void) {
    const tasks = this.initializationTasks();
    const signal = new AbortController().signal;
    for (let index = 0; index < tasks.length; index += 1) {
      onProgress(tasks[index].label, index, tasks.length);
      await tasks[index].run(signal);
      onProgress(tasks[index].label, index + 1, tasks.length);
    }
  }

  /**
   * Adopt scalar-only scene revisions. Every consumer of these values either
   * reads `this.scene` when encoding a step or reads the params buffer, so
   * re-writing params is the whole update; no allocation, seed, or topology
   * depends on them.
   */
  applySceneUniforms(scene: SceneDescription) {
    this.scene = scene;
    this.writeParams();
  }

  /**
   * Warm re-seed: adopt a new scene and overwrite the resident level set in
   * place, leaving every allocation, pipeline, and arena as-is. The caller
   * re-runs the fenced cold-bootstrap phases afterwards, which is what turns
   * the new phi into topology, structured authority, and a render world.
   *
   * Returns false when the seed cannot be produced against the existing
   * allocations, so the caller can fall back to a full rebuild rather than
   * running the solver on a stale or half-written seed.
   */
  reseed(scene: SceneDescription): boolean {
    const surfaceState = this.surfaceState;
    if (!surfaceState) return false;
    // Cell size derives from the incoming container extent and the resident
    // dims. The extent is a seed-tier input — scaling the world moves it, and
    // the whole point is that the dims do not follow — so this must be read
    // from `scene`, never from the extent captured at construction.
    const cell = {
      x: scene.container.width_m / this.dims.nx,
      y: scene.container.height_m / this.dims.ny,
      z: scene.container.depth_m / this.dims.nz,
    };
    const phi = initialOctreeLevelSet(scene, this.dims, cell);
    if (!surfaceState.reseedLevelSet(this.device, phi)) return false;
    this.scene = scene;
    // The caller re-runs the fenced cold-bootstrap phases, whose
    // structured-authority submission retires the selector again.
    this.analyticBootstrapRetired = false;
    this.writeParams();
    return true;
  }

  /**
   * Publish the authored refinement regions.
   *
   * Split from `writeParams` because it is the one part of the uniform that a
   * *scene edit* moves rather than construction: dragging a region's face
   * rewrites 272 bytes on the running solver and nothing else happens. The
   * regions live in cell coordinates, so the lattice — not the container
   * extent — is what they are resolved against.
   */
  private writeRefinementRegionParams() {
    this.device.queue.writeBuffer(this.params, OCTREE_REFINEMENT_REGION_PARAMS_OFFSET,
      packOctreeRefinementRegions(
        sceneRefinementRegions(this.scene),
        {
          dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
          cellSize_m: [
            this.scene.container.width_m / this.dims.nx,
            this.scene.container.height_m / this.dims.ny,
            this.scene.container.depth_m / this.dims.nz,
          ],
          origin_m: {
            x: -0.5 * this.scene.container.width_m,
            y: 0,
            z: -0.5 * this.scene.container.depth_m,
          },
        },
        this.topologyMaximumLeafSize,
      ));
  }

  private writeColdAuthoredSurfaceParams() {
    const boxes = octreeColdAuthoredSurfaceBoxes(
      this.scene,
      [this.dims.nx, this.dims.ny, this.dims.nz],
      this.scene.voxelDomain.brickSize_cells,
    );
    this.device.queue.writeBuffer(
      this.params,
      OCTREE_REFINEMENT_REGION_PARAMS_OFFSET + OCTREE_REFINEMENT_REGION_PARAMS_BYTES,
      packOctreeColdAuthoredSurfaceBoxes(boxes),
    );
  }

  writeParams() {
    const data = new ArrayBuffer(OCTREE_REFINEMENT_REGION_PARAMS_OFFSET);
    new Uint32Array(data, 0, 4).set([
      this.dims.nx, this.dims.ny, this.dims.nz, this.topologyMaximumLeafSize,
    ]);
    new Float32Array(data, 16, 4).set([
      this.scene.container.width_m / this.dims.nx,
      this.scene.container.height_m / this.dims.ny,
      this.scene.container.depth_m / this.dims.nz,
      0.8
    ]);
    const fineChangedKeysOffsetWords =
      this.globalFineTopologyAB?.pageDeltaLayout.changedKeysOffsetWords ?? 0;
    new Uint32Array(data, 32, 4).set([
      Math.round(this.adaptivity * 1000),
      fineChangedKeysOffsetWords,
      this.linearBlocks,
      0,
    ]);
    // Megakernel residual tolerance and compact pressure-solve controls.
    // solve.w is the ONLY live consumer of the band: the refinement protection
    // width, the boundary gate and the closed-wall strip all read it here, so
    // rewriting this uniform is the whole of applying the band dial.
    new Float32Array(data, 48, 4).set([1e-8, 0.01, 2.2, this.interfaceBandCellsEffective]);
    // container.w is an exactly representable small bit mask shared with the
    // topology shader: terrain and closed ceiling. The projection has one
    // native power topology, so no authority-selector bit is retained.
    const containerFlags = (sceneHasTerrain(this.scene) ? 1 : 0)
      | (this.scene.container.top === "closed" ? 2 : 0);
    new Float32Array(data, 64, 4).set([
      this.scene.container.width_m,
      this.scene.container.height_m,
      this.scene.container.depth_m,
      containerFlags,
    ]);
    const inflow = this.scene.fluid.inflow;
    const speed = inflow ? Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z) : 0;
    new Float32Array(data, 80, 4).set([inflow?.center_m.x ?? 0, inflow?.center_m.y ?? 0, inflow?.center_m.z ?? 0, inflow?.radius_m ?? 0]);
    new Float32Array(data, 96, 4).set([
      speed > 0 ? inflow!.velocity_m_s.x / speed : 0,
      speed > 0 ? inflow!.velocity_m_s.y / speed : 0,
      speed > 0 ? inflow!.velocity_m_s.z / speed : 0,
      inflow?.length_m ?? 0
    ]);
    // Every scene needs a t=0 phi authority, not just the analytic ones.
    // Non-analytic scenes read the imported dense level set until the same
    // retirement hands over to published coarse rows.
    const analyticBootstrapSelector = this.analyticBootstrapRetired
      ? 0
      : !this.analyticSparseBootstrap
        ? -30
        : this.scene.fluid.initialCondition === "dam-break" ? -20 : -10;
    new Float32Array(data, 112, 4).set([
      this.scene.fluid.density_kg_m3,
      this.scene.fluid.surfaceTension_N_m,
      this.scene.numerics.maxDt_s,
      analyticBootstrapSelector
    ]);
    // pressureCapacity.z carries three live topology experiments because the
    // projection uniform ABI reserves this word. Bytes are respectively:
    // grading layers, closed-wall band, and finest factor-one surface cell.
    // Keep pressureCapacity.w untouched: it carries warm-start bit zero and a
    // dynamically stamped corrected-coarse generation above bit one.
    const topologyDials = (this.surfaceGradingLayersEffective & 0xff)
      | ((this.wallBandCellsEffective & 0xff) << 8)
      | ((this.finestSurfaceCellSizeEffective & 0xff) << 16);
    new Uint32Array(data, 128, 4).set([
      this.pressureCapacity.rowCapacity,
      this.analyticSparseBootstrap
        ? (this.scene.fluid.initialCondition === "dam-break" ? 2 : 1)
        : 0,
      // Live: layer one is the mandatory sharp 2:1 transition supplied by the
      // balance closure. Only layers above one add progressive distance
      // padding, and the surface-band dial removes those before the band.
      topologyDials,
      1,
    ]);
    const dam = sceneDamBreakFractions(this.scene);
    new Float32Array(data, 144, 4).set([
      dam.width * this.scene.container.width_m,
      dam.height * this.scene.container.height_m,
      dam.depth * this.scene.container.depth_m,
      this.scene.container.fillFraction * this.dims.ny,
    ]);
    this.device.queue.writeBuffer(this.params, 0, data);
    this.writeRefinementRegionParams();
    this.writeColdAuthoredSurfaceParams();
  }

  setTimestep(dt_s: number) {
    this.powerTimestep_s = Math.max(0, Number.isFinite(dt_s) ? dt_s : 0);
  }

  setCouplingBodies(count: number, hasDynamicBodies: boolean) {
    const bounded = Math.max(0, Math.min(12, Math.floor(count)));
    this.device.queue.writeBuffer(this.params, 44, new Uint32Array([bounded]));
    this.device.queue.writeBuffer(this.params, 116, new Float32Array([hasDynamicBodies ? 1 : 0]));
    // Only a body that integrates can consume a reaction. A scene of authored
    // static solids still cuts apertures and imposes its normal velocity, but
    // the fluid-to-solid adjoint would write an exchange nobody reads, so the
    // whole pass stays off its command graph.
    this.dynamicCouplingBodyCount = hasDynamicBodies ? bounded : 0;
  }

  /**
   * Encode the one-time full-domain rebuild after bootstrap residency has been
   * written into the command stream.  Residency must run first so the owner
   * page lifecycle below can consume its active-brick worklist, while the
   * rebuild itself must still take the cold (full-domain) path because no
   * adaptive frontier has been published yet.
   */
  encodeColdBootstrapRebuild(encoder: GPUCommandEncoder) {
    if (this.analyticBootstrapWorklist) {
      // Analytic dam/tank scenes have a provably bounded liquid/interface box.
      // Publish the resident topology ABI on-GPU and immediately consume it;
      // no finest-domain scan or topology count readback is required. Missing
      // tiles are analytically non-negative air, while cold analytic phi retains the
      // authored SDF until compact coarse phi has published.
      this.analyticBootstrapWorklist.encode(encoder);
      this.topologyWorklistReady = true;
      this.encodeInactiveTopologyCandidate(encoder, true);
      // Cold bootstrap has no prior live epoch. It is the sole lifecycle
      // exception that publishes in the same command stream so the following
      // t=0 pressure checkpoint has an authority to consume.
      this.encodeReadyTopologyFlip(encoder);
      return;
    }
    // Non-analytic authored surfaces (terrain or rigid bodies) publish their
    // exact t=0 brick/tile residency from the imported dense SDF once. Owner
    // pages consume that bounded worklist, while refinement deliberately uses
    // the cold full-domain kernels because no prior adaptive frontier exists.
    this.topologyResidency.encode(encoder, this.surfaceState.texture);
    this.topologyWorklistReady = true;
    this.encodeInactiveTopologyCandidate(encoder, false, true);
    this.encodeReadyTopologyFlip(encoder);
  }

  /**
   * Seed Bet 1's exact structural/wet-decision fingerprints from the accepted
   * t=0 authority. Without this census the first no-time-advanced candidate
   * compares against zero-initialized signatures, marks every tile dirty, and
   * cannot distinguish an unchanged frontier from an empty replacement. This
   * is startup-only; recurring generations consume the same fingerprints and
   * retain the bounded changed-tile path.
   */
  private encodeColdTopologySignatureBaseline(encoder: GPUCommandEncoder): void {
    const decisionGroup = this.topologyDecisionGroup;
    if (!decisionGroup) throw new Error("Cold topology signature authority is unavailable");
    const broker = new PassBroker(encoder);
    broker.copyBufferToBuffer(
      this.topologyResidency.tileWorklist, 0,
      this.compaction, 0,
      this.topologyResidency.tileWorklistByteLength,
    );
    broker.copyBufferToBuffer(
      this.topologyResidency.tileWorklist, 0,
      this.solveDispatch, 48, 4,
    );
    const census = broker.compute({ label: "Seed accepted t=0 topology decision signatures" });
    census.setPipeline(this.classifyTopologyTileSignaturePipeline);
    census.setBindGroup(0, decisionGroup);
    census.dispatchWorkgroupsIndirect(this.solveDispatch, 48);
    broker.fence("accepted t=0 topology decision signatures seeded");
  }

  /** Encode one dependency-ordered t=0 checkpoint. Safe bring-up submits and
   * fences these separately so a driver failure is localized to one bounded
   * phase; product startup appends all checkpoints to one command buffer. */
  encodeInitialSparseAuthorityPhase(encoder: GPUCommandEncoder, phase: OctreeInitialSparseAuthorityPhaseId) {
    switch (phase) {
      case "cold-topology": {
        // Factor one has no fine-summary bootstrap. Publish its complete
        // analytic coarse lattice first so the cold pressure tree receives
        // the same interface-band refinement evidence as factor four, rather
        // than assembling free-surface T-junctions from an empty summary.
        if (this.coarseOnlySurfaceTracking && this.coarseOnlySummary) {
          const broker = new PassBroker(encoder);
          this.coarseOnlySummary.encode(broker);
          broker.fence("factor-one analytic coarse summary published before cold topology");
          encoder = broker.commandEncoder();
        }
        this.encodeColdBootstrapRebuild(encoder);
        break;
      }
      case "structured-authority":
        this.encode(
          encoder, this.dims.nx, this.dims.ny, this.dims.nz,
          undefined, "power-operator-only",
        );
        // The bootstrap selector remains invocation-stable throughout the
        // first structured solve. The submission-retirement hook writes zero
        // selector only after this encoder is submitted, so no command buffer
        // can observe a mixture of bootstrap and published sparse phi. Both
        // bootstrap authorities hand over here: the imported dense level set
        // is exactly as stale as the analytic form once coarse rows publish.
        this.analyticBootstrapRetirementByEncoder.add(encoder);
        break;
      case "surface-global-fine": this.encodeSurface(encoder, 0); break;
      case "sparse-render-world":
        this.encodeSparseBrickWorld(encoder);
        // Coarse phi is authoritative only after the preceding t=0 surface
        // checkpoint. Seed the structural/wet signature baseline here, at the
        // same scratch-lifecycle seam the candidate builder already owns.
        this.encodeColdTopologySignatureBaseline(encoder);
        // Warmup is the prior substep for the first live advance.  Prepare
        // generation 2 after every t=0 consumer has finished, but leave its
        // selector pending so the ordinary beginning-of-substep flip remains
        // the sole recurring publication operation.
        this.encodeInactiveTopologyCandidate(encoder);
        break;
      default: phase satisfies never;
    }
  }

  /** Retire invocation-stable coarse-phi parameter slots after queue submit. */
  retireSubmittedEncoder(encoder: GPUCommandEncoder) {
    const publishedIsA = this.globalFinePublicationByEncoder.get(encoder);
    if (publishedIsA !== undefined) {
      this.globalFinePublishedIsA = publishedIsA;
      this.globalFinePublicationByEncoder.delete(encoder);
    }
    if (this.analyticBootstrapRetirementByEncoder.delete(encoder)) {
      this.analyticBootstrapRetired = true;
      this.device.queue.writeBuffer(this.params, 124, new Float32Array([0]));
      this.lane.retireAnalyticBootstrap();
    }
    this.lane.retireSubmittedEncoder(encoder);
  }

  /** Tail of substep N: build and validate only the inactive frontier epoch. */
  encodeInactiveTopologyCandidate(
    encoder: GPUCommandEncoder,
    analyticColdBootstrap = false,
    coldFullRebuild = false,
  ) {
    this.topologyReusePending = false;
    this.powerAttemptGeneration = ((this.powerAttemptGeneration + 1) >>> 0) || 1;
    this.candidatePowerGeneration = this.powerAttemptGeneration;
    // Stamp the attempt in GPU command order. Multiple substeps may be
    // encoded into one command buffer; queue.writeBuffer on shared storage
    // would make every invocation observe the final host value instead.
    const stampBroker = new PassBroker(encoder);
    const stamp = stampBroker.compute({ label: "Stamp octree topology attempt generation" });
    stamp.setPipeline(this.stampFrontierAttemptPipeline);
    stamp.setBindGroup(0, this.groups.ab);
    stamp.dispatchWorkgroups(1);
    stampBroker.fence("topology attempt generation stamped");
    // Directory generation N is produced after topology N and is the authority
    // for the next topology rebuild. Only the lane knows whether it schedules
    // one, and its generation counter is its own.
    this.lane.stampCoarseDirectoryGeneration();
    // The first rebuild initializes every owner and, when present, solid cell. Thereafter the
    // previous publication's GPU-owned topology-tile list is the rebuild
    // domain: tiles span max(brick, maximumLeaf) cells, so every leaf lies
    // inside exactly one tile and partial rebuilds can never split a leaf.
    const residencyReady = this.topologyWorklistReady;
    const active = residencyReady && !coldFullRebuild;
    // The owner-page lifecycle consumes the same exact topology-tile
    // publication as the partial topology path. It publishes the complete
    // sorted page set before any refinement kernel may write payload owners;
    // missing support thereafter fails the generation closed.
    if (analyticColdBootstrap) {
      this.ownerPages.encodeAnalyticBootstrap(new PassBroker(encoder));
    } else if (residencyReady) {
      this.ownerPages.encodeInactiveCandidate(new PassBroker(encoder));
    }
    const broker = new PassBroker(encoder);
    if (active) {
      broker.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, 0,
        this.compaction, 0,
        this.topologyResidency.tileWorklistByteLength
      );
      if (analyticColdBootstrap) {
        // The analytic publisher is the cold generation's immutable tile
        // authority. Stage its three exact dispatch records before any
        // indirect consumer; the recurring dirty-tile singleton owns this
        // staging on every later generation.
        broker.copyBufferToBuffer(
          this.topologyResidency.tileWorklist,
          FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES,
          this.solveDispatch, 0, 12,
        );
        broker.copyBufferToBuffer(
          this.topologyResidency.tileWorklist,
          FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
          this.topologyCandidateDispatch, 0, 12,
        );
        broker.copyBufferToBuffer(
          this.topologyResidency.tileWorklist, 0,
          this.solveDispatch, 48, 4,
        );
      }
    }
    // Fine values may change every step without changing pressure topology or
    // frontier membership. Compare the actual per-tile owner/wet decisions in
    // parallel, then compact only changed signatures, residency transitions,
    // and rigid-body bounds. The active-count copy forms a one-workgroup-per-
    // tile schedule; y/z were initialized to one with solveDispatch.
    if (active && !analyticColdBootstrap) {
      const decisionGroup = this.topologyDecisionGroup;
      if (!decisionGroup) throw new Error("Topology decision signature authority is unavailable");
      broker.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, 0,
        this.solveDispatch, 48, 4,
      );
      const signatures = broker.compute({ label: "Compare topology-tile refinement signatures" });
      signatures.setPipeline(this.classifyTopologyTileSignaturePipeline);
      signatures.setBindGroup(0, decisionGroup);
      signatures.dispatchWorkgroupsIndirect(this.solveDispatch, 48);
      const mark = broker.compute({ label: "Build exact structural topology-tile delta" });
      mark.setPipeline(this.buildDirtyTileDeltaPipeline);
      mark.setBindGroup(0, decisionGroup);
      mark.dispatchWorkgroups(1);
      // The singleton publishes one exact union schedule over dirty active
      // tiles and retired tiles. Staging its three immutable argument records
      // avoids separate active/retired dispatches in every downstream level.
      broker.copyBufferToBuffer(this.compaction, 4, this.solveDispatch, 0, 12);
      broker.copyBufferToBuffer(this.compaction, 20, this.solveDispatch, 48, 12);
      broker.copyBufferToBuffer(this.compaction, 32, this.topologyCandidateDispatch, 0, 12);
    }
    let pass = broker.compute({ label: "Octree reset and refinement" });
    const dispatch = (full: GPUComputePipeline, delta: GPUComputePipeline) => {
      pass.setPipeline(active ? delta : full);
      pass.setBindGroup(0, this.groups.ab);
      if (active) pass.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      else pass.dispatchWorkgroupsIndirect(this.coldDispatch, 0);
    };
    const dispatchCandidates = (full: GPUComputePipeline, delta: GPUComputePipeline,
      group = this.groups.ab) => {
      pass.setPipeline(active ? delta : full);
      pass.setBindGroup(0, group);
      if (active) pass.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      else pass.dispatchWorkgroupsIndirect(this.coldDispatch, 12);
    };
    if (this.hasDenseSolidCells) {
      dispatch(this.rasterizeSolidsPipeline, this.rasterizeSolidsDeltaPipeline);
    }
    dispatch(this.resetPipeline, this.resetDeltaPipeline);
    const dispatchCoarse = (size: number, pipelines: OctreePipelineVariants,
      group = this.groups.ab) => {
      pass.setPipeline(active ? pipelines.delta : pipelines.full);
      pass.setBindGroup(0, group);
      if (active) pass.dispatchWorkgroupsIndirect(this.solveDispatch, 48);
      else {
        const offset = this.coldDispatchOffsetBySize.get(size);
        if (offset === undefined) throw new Error(`Missing immutable cold dispatch for size ${size}`);
        pass.dispatchWorkgroupsIndirect(this.coldDispatch, offset);
      }
    };
    for (const size of this.refinementSizes) {
      if (size >= 16) {
        dispatchCoarse(size, this.refineCoarsePipelines.get(size)!, this.fineSummarySizingGroup);
      } else {
        const level = this.refineLevelPipelines.get(size)!;
        dispatchCandidates(level.full, level.delta, this.fineSummarySizingGroup);
      }
    }
    if (active && !analyticColdBootstrap && this.balanceRounds > 0) {
      // Refinement is an exact decision delta, but grading is its closure: a
      // split at the edge of that delta can make an otherwise unchanged
      // resident support tile mixed. Restore the immutable active-residency
      // stream for balance only. The wet-frontier delta is rebuilt below, so
      // dry support tiles inspected here never become pressure rows.
      broker.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, 0,
        this.compaction, 0,
        this.topologyResidency.tileWorklistByteLength,
      );
      broker.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, 0,
        this.solveDispatch, 48, 4,
      );
      broker.copyBufferToBuffer(
        this.topologyResidency.tileWorklist,
        FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
        this.topologyCandidateDispatch, 0, 12,
      );
      pass = broker.compute({ label: "Octree resident grading closure" });
    }
    const gradingRoundProbe = typeof process !== "undefined"
      && process.env?.FLUID_GRADING_ROUND_PROBE === "1";
    for (let round = 0; round < this.balanceRounds; round += 1) {
      const tag = String(round).padStart(2, "0");
      for (const size of this.coarseRefinementSizes) {
        if (gradingRoundProbe) {
          pass = broker.compute({ label: `Octree grading r${tag} coarse ${size}` });
        }
        dispatchCoarse(size, this.balanceCoarsePipelines.get(size)!);
      }
      if (gradingRoundProbe) {
        pass = broker.compute({ label: `Octree grading r${tag} candidates` });
      }
      dispatchCandidates(this.balancePipeline, this.balanceDeltaPipeline);
    }
    if (active && !analyticColdBootstrap) {
      const decisionGroup = this.topologyDecisionGroup;
      if (!decisionGroup) throw new Error("Topology decision signature authority is unavailable");
      // Structural topology and liquid-row membership are independent deltas.
      // Restore the immutable residency worklist after topology consumed its
      // compact schedule, then publish only tiles whose wet-frontier decision
      // changed (plus tiles already stamped by structural/rigid work).
      broker.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, 0,
        this.compaction, 0,
        this.topologyResidency.tileWorklistByteLength,
      );
      const frontierDelta = broker.compute({ label: "Build exact wet-frontier tile delta" });
      frontierDelta.setPipeline(this.buildDirtyFrontierDeltaPipeline);
      frontierDelta.setBindGroup(0, decisionGroup);
      frontierDelta.dispatchWorkgroups(1);
      broker.copyBufferToBuffer(this.compaction, 32, this.topologyCandidateDispatch, 0, 12);
    }
    // Publish the immutable liquid-leaf frontier. Cold initialization emits
    // the bounded whole-domain candidate stream once; recurring generations
    // emit only dirty topology tiles and sorted-merge them with clean old rows.
    const begin = broker.compute({ label: "Begin persistent octree leaf frontier" });
    begin.setPipeline(this.beginFrontierPipeline); begin.setBindGroup(0, this.groups.ab); begin.dispatchWorkgroups(1);
    const candidates = broker.compute({ label: "Classify exact dirty-tile frontier candidates" });
    candidates.setBindGroup(0, active ? this.fineSummarySizingGroup : this.groups.ab);
    candidates.setPipeline(active
      ? this.classifyFrontierCandidatesDeltaPipeline
      : this.classifyFrontierCandidatesPipeline);
    if (active) candidates.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
    else candidates.dispatchWorkgroupsIndirect(this.coldDispatch, 12);
    candidates.setPipeline(active
      ? this.prefixFrontierCandidateBlocksDeltaPipeline
      : this.prefixFrontierCandidateBlocksPipeline);
    candidates.dispatchWorkgroups(1);
    candidates.setPipeline(active
      ? this.emitFrontierCandidatesDeltaPipeline
      : this.emitFrontierCandidatesPipeline);
    if (active) candidates.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
    else candidates.dispatchWorkgroupsIndirect(this.coldDispatch, 12);
    // Candidate emission owns the exact live count. Turn it and the previous
    // frontier count into three compact schedules (sort, carry, merge), then
    // stage them with one pass boundary. A valid zero-delta transaction writes
    // three zero dispatches and keeps the immutable frontier publication.
    candidates.setPipeline(this.prepareFrontierDispatchPipeline);
    candidates.dispatchWorkgroups(1);
    broker.copyBufferToBuffer(this.compaction, 4, this.topologyCandidateDispatch, 0, 36);
    const candidateSort = broker.compute({ label: "Sort dirty frontier candidates by level and Morton" });
    if (this.useLocalFrontierCandidateSort) {
      candidateSort.setBindGroup(0, active ? this.fineSummarySizingGroup : this.groups.ab);
      candidateSort.setPipeline(this.sortFrontierCandidatesLocalPipeline);
      candidateSort.dispatchWorkgroups(1);
    } else {
      const pipeline = this.frontierCandidateSortPipelines[0];
      if (!pipeline || this.frontierSortGroups.length === 0) {
        throw new Error("Distributed frontier sort pipeline is unavailable");
      }
      candidateSort.setPipeline(pipeline);
      for (const group of this.frontierSortGroups) {
        candidateSort.setBindGroup(0, group);
        candidateSort.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      }
    }
    const merge = broker.compute({ label: "Sorted old/new frontier merge" });
    merge.setBindGroup(0, active ? this.fineSummarySizingGroup : this.groups.ab);
    merge.setPipeline(this.classifyFrontierCarryPipeline);
    merge.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    merge.setPipeline(this.scanFrontierCarryBlocksPipeline);
    merge.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    merge.setPipeline(this.prefixFrontierCarryBlocksPipeline); merge.dispatchWorkgroups(1);
    merge.setPipeline(this.mergeFrontierRowsPipeline);
    merge.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 24);
    merge.setPipeline(this.finalizeFrontierPipeline); merge.dispatchWorkgroups(1);
    // Reuse the indirect buffer for pressure-row plan/emit and exact row
    // comparison. The finalizer publishes both records contiguously enough to
    // stage them without reopening a compute pass between the copies. Record
    // three is an exact one-workgroup-per-row extent for the cooperative
    // one-ring kernel; large capacities retain record one's 256-row fallback.
    broker.copyBufferToBuffer(this.compaction, 48, this.topologyCandidateDispatch, 0, 12);
    broker.copyBufferToBuffer(this.compaction, 4, this.topologyCandidateDispatch, 12, 12);
    broker.copyBufferToBuffer(this.compaction, 16, this.topologyCandidateDispatch, 36, 12);
    broker.fence("octree topology and frontier publication complete");
    this.encodeFrontierRows(
      encoder,
      "Inactive octree pressure-row candidate",
      this.lane.overlayPressureAuthorityIsA(this.latestPressureInA)
        ? this.candidateRowGroups.fromA : this.candidateRowGroups.fromB,
    );
    this.lane.encodeInactiveCandidate(encoder);
    return true;
  }


  /** Beginning of substep N+1: sole coupled owner/frontier epoch flip. */
  encodeReadyTopologyFlip(encoder: GPUCommandEncoder): void {
    this.lane.encodeReadyTopologyFlip(encoder);
  }

  finishTopologyCandidate() { this.info.topologyReuseCount += 1; }

  /**
   * Tail scheduling for the Losasso k-advance epoch. The fine band is still
   * transported every advance; its construction-time page plan includes the
   * corresponding extra dilation rings. Power 2017 always takes the legacy
   * every-advance path.
   */
  /**
   * Adopt the live coarse-band accuracy/cost dials.
   *
   * Cheap and idempotent by construction: every branch below is either a
   * queue write into an already-allocated buffer or a host field the next
   * encode reads, so the renderer can call this every frame with the current
   * parameter bag and only a changed dial does any work. Nothing here touches
   * the lattice, the arenas, or the accepted topology epoch, which is what
   * lets these keys stay out of the structural fingerprint.
   *
   * Power 2017 keeps its frozen reference behaviour: the dials describe
   * Losasso machinery and there is no equivalent seam on that backend.
   */
  applyRuntimeDials(dials: OctreeRuntimeDials): void {
    if (!this.lane.runtimeDialsApplicable()) return;
    if (this.appliedRuntimeDials && octreeRuntimeDialsEqual(this.appliedRuntimeDials, dials)) {
      return;
    }
    this.appliedRuntimeDials = dials;
    // The three topology dials change what the next LATTICE epoch is rather
    // than how hard it is worked, and reach the GPU as one uniform write --
    // `writeParams` is otherwise only called at construction and on reseed, so
    // there is no per-frame cost to pay for making it live. The topology is
    // rebuilt from the new width on the next candidate epoch; nothing is
    // reallocated and t=0 is not disturbed, which is what keeps this key out of
    // the structural fingerprint.
    const surface = octreeDialledSurfaceBand(
      this.interfaceRefinementBandCells, this.surfaceRefinementGradingLayers,
      this.coarseOnlySurfaceTracking ? 1 : 4, dials);
    const surfaceChanged = surface.bandCells !== this.interfaceBandCellsEffective
      || surface.gradingLayers !== this.surfaceGradingLayersEffective;
    const topologyDialsChanged = surfaceChanged
      || dials.wallBandCells !== this.wallBandCellsEffective
      || dials.finestSurfaceCellSize !== this.finestSurfaceCellSizeEffective;
    if (topologyDialsChanged) {
      this.interfaceBandCellsEffective = surface.bandCells;
      this.surfaceGradingLayersEffective = surface.gradingLayers;
      this.wallBandCellsEffective = dials.wallBandCells;
      this.finestSurfaceCellSizeEffective = dials.finestSurfaceCellSize;
      if (surfaceChanged) {
        this.coarseOnlySummary?.setRedistanceReachCells(
          octreeSurfaceProtectionWidthCells(
            surface.bandCells, surface.gradingLayers,
            this.topologyMaximumLeafSize, 1,
          ),
        );
      }
      this.writeParams();
    }
    this.lane.applyRuntimeDials(dials);
    // Zero keeps the construction-time cadence, whose value is also what sized
    // the candidate's extra dilation rings. A larger runtime cadence is
    // deliberately allowed to exceed that padding: it spends the canonical
    // band's own slack, so the surface can outrun its refined region rather
    // than the epoch failing to publish.
    this.topologyCadenceOverride = dials.topologyRebuildCadence > 0
      ? dials.topologyRebuildCadence : undefined;
  }

  encodeInactiveTopologyCandidateIfDue(encoder: GPUCommandEncoder): boolean {
    const cadence = this.topologyCadenceOverride
      ?? this.coarseDynamics.topology.advancesPerEpoch;
    if (this.lane.topologyCadenceIsEveryAdvance || cadence === 1) {
      return this.encodeInactiveTopologyCandidate(encoder);
    }
    this.topologyCadenceCursor += 1;
    if (this.topologyCadenceCursor < cadence) {
      if (this.candidatePowerGeneration !== 0) {
        throw new Error("Topology cadence cannot reuse while an inactive candidate is pending");
      }
      this.topologyReusePending = true;
      this.info.topologyReused = true;
      this.info.topologyReuseCount += 1;
      this.info.topologyCadenceSkipCount += 1;
      return false;
    }
    this.topologyCadenceCursor = 0;
    return this.encodeInactiveTopologyCandidate(encoder);
  }
  get pressureSolverLabel() { return this.lane.pressureSolverLabel(); }


  private encodeFrontierRows(
    encoder: GPUCommandEncoder,
    label: string,
    group = this.groups.ab,
  ): void {
    const broker = new PassBroker(encoder);
    const dirty = broker.compute({ label: `${label} dirty-row deterministic scan` });
    dirty.setBindGroup(0, group);
    dirty.setPipeline(this.scanDirtyRowDeltaBlocksPipeline);
    dirty.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    dirty.setPipeline(this.prefixDirtyRowDeltaBlocksPipeline);
    dirty.dispatchWorkgroups(1);
    dirty.setPipeline(this.scatterDirtyRowDeltaPipeline);
    dirty.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    const compact = broker.compute({ label });
    compact.setPipeline(this.planPipeline); compact.setBindGroup(0, group);
    compact.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
    compact.setPipeline(this.scanPipeline); compact.dispatchWorkgroups(1, 1, 1);
    compact.setPipeline(this.emitPipeline); compact.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
    const deltaPublish = broker.compute({ label: `${label} row-delta one-ring publication` });
    deltaPublish.setBindGroup(0, group);
    deltaPublish.setPipeline(this.useCooperativeRowDeltaRing
      ? this.markRowDeltaRingPipeline : this.markRowDeltaRingBlocksPipeline);
    deltaPublish.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch,
      this.useCooperativeRowDeltaRing ? 36 : 12);
    const deltaCompact = broker.compute({ label: `${label} row-delta compact publication` });
    deltaCompact.setBindGroup(0, group);
    deltaCompact.setPipeline(this.scanAffectedRowDeltaBlocksPipeline);
    deltaCompact.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    deltaCompact.setPipeline(this.prefixAffectedRowDeltaBlocksPipeline);
    deltaCompact.dispatchWorkgroups(1);
    deltaCompact.setPipeline(this.compactRowDeltaPipeline);
    deltaCompact.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    const deltaFinalize = broker.compute({ label: `${label} row-delta validate publication` });
    deltaFinalize.setBindGroup(0, group);
    deltaFinalize.setPipeline(this.publishRowDeltaPipeline);
    deltaFinalize.dispatchWorkgroups(1);
    deltaFinalize.setPipeline(this.publishReusedRowDeltaPipeline);
    // The established two-level validation lane consumes record 2 and is
    // fingerprinted against that publication order. Larger adaptive octrees
    // need record 1's full previous-row schedule to refresh every identity;
    // using record 2 there leaves stale descriptor/topology diagnostics.
    deltaFinalize.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch,
      this.maxLeafSize <= 2 ? 24 : 12);
    broker.copyBufferToBuffer(this.compaction, 8, this.solveDispatch, 0, 24);
  }

  encode(
    encoder: GPUCommandEncoder,
    _nx: number,
    _ny: number,
    _nz: number,
    options?: {
      productionBoundary?: OctreeSemanticBoundary;
      /** Stamp shared with the end-of-step snapshot ring. */
      step?: number;
    },
    scope: "complete" | "power-operator-only" = "complete",
  ): GPUCommandEncoder {
    return this.lane.encodeAdvance(encoder, {
      productionBoundary: options?.productionBoundary, step: options?.step, scope,
    });
  }

  /** Publish lazily allocated diagnostic textures from the live owner map.
   * The first overlay request materializes immediately, so reset-time grid
   * inspection never decodes zero-initialized topology storage as finest 1^3. */
  encodeOverlayMaterialization(encoder: GPUCommandEncoder, pressureInA = this.latestPressureInA) {
    if (!this.diagnosticGroups || !this.materializePipeline) return false;
    const broker = new PassBroker(encoder);
    const materialize = broker.compute({ label: "Materialize octree overlay fields" });
    materialize.setPipeline(this.materializePipeline);
    const authorityInA = this.lane.overlayPressureAuthorityIsA(pressureInA);
    materialize.setBindGroup(0, authorityInA
      ? this.diagnosticGroups.pressureA : this.diagnosticGroups.pressureB);
    materialize.dispatchWorkgroupsIndirect(this.coldDispatch, 0);
    broker.fence("octree overlay fields materialized");
    return true;
  }


  /** Settle the transported fine generation after projected velocity and CPT
   * seeds exist, then re-correct coarse phi without a second advection. */
  encodePendingFineSettlement(
    encoder: GPUCommandEncoder,
    productionBoundary?: OctreeSemanticBoundary,
  ): GPUCommandEncoder {
    const pending = this.pendingFinePublication;
    if (!pending) return encoder;
    const fineEngineSplits = octreeFineEngineSplitsEnabled();
    const split = (enginePhase: OctreeEnginePhase | undefined,
      finePhase: Extract<OctreeFineSemanticPhase, "fineRedistance" | "fineRestriction">) => {
      const phase = fineEngineSplits ? finePhase : enginePhase;
      if (productionBoundary && phase) encoder = productionBoundary(phase, encoder);
    };
    const redistanceBroker = new PassBroker(encoder);
    const rigidCarve = pending.targetIsA
      ? this.globalFineRigidCarveA : this.globalFineRigidCarveB;
    rigidCarve?.encode(redistanceBroker);
    pending.redistance.encode(redistanceBroker, {
      bandCells: pending.redistanceBandCells,
      maximumDisplacementFineCells: pending.maximumDisplacementFineCells,
      warmStart: pending.warmClosestPoints,
      residualTolerance: 1,
      // Mirrors the transport kernel's closed-Neumann boundary policy: closed
      // walls extend phi with unit outward slope, so seeding can represent a
      // surface separating from a wall (lid films otherwise dry only by
      // lateral erosion — the free-fall drop oracles).
      closedBoundary: true,
      openTopBoundary: this.scene.container.top !== "closed",
    });
    this.lane.encodeSettlementBootstrap(redistanceBroker, pending.target);
    if (pending.volume) {
      // A lane that treats phi as geometric signed-distance authority measures
      // the loss without repairing it; one that owns a volume budget corrects.
      if (this.lane.settlementVolumePolicy === "measure") {
        pending.volume.encodeMeasurement(redistanceBroker);
      } else {
        pending.volume.encode(redistanceBroker);
      }
    }
    this.lane.encodeSettlementCoarseRefresh(redistanceBroker, pending.target);
    encoder = redistanceBroker.commandEncoder();
    split("closestPointWaves", "fineRedistance");

    const restrictionBroker = new PassBroker(encoder);
    pending.topology.encodeFinalizePublication(restrictionBroker, {
      redistance: pending.redistance.control,
      ...(pending.volume ? { volume: pending.volume.control } : {}),
      ...(pending.transport ? { transport: pending.transport.control } : {}),
    });
    this.lane.encodeSettlementRestriction(restrictionBroker, pending.target,
      pending.topology);
    encoder = restrictionBroker.commandEncoder();
    this.globalFinePublicationByEncoder.set(encoder, pending.targetIsA);
    split(undefined, "fineRestriction");
    this.globalFineCurrentIsA = pending.targetIsA;
    this.globalFineBootstrapped = true;
    this.pendingFinePublication = undefined;
    return encoder;
  }

  /** Transport fine and coarse phi with the previous projected velocity.
   * Recurring redistance is deferred until projection has seeded the next CPT
   * extension. Segmented callbacks may replace the encoder, so the returned
   * encoder always owns the continuation. */
  encodeSurface(encoder: GPUCommandEncoder, dt_s: number, inflow?: SurfaceInflowState,
    _maximumDt_s?: number, productionBoundary?: OctreeSemanticBoundary): GPUCommandEncoder {
    this.surfaceInflow = inflow;
    const fineEngineSplits = octreeFineEngineSplitsEnabled();
    const splitProductionPhase = (
      enginePhase: OctreeEnginePhase | undefined,
      finePhase: Extract<OctreeFineSemanticPhase,
        "finePreparation" | "fineTransport" | "fineTopology" | "fineRedistance" | "fineRestriction">,
    ) => {
      const phase = fineEngineSplits ? finePhase : enginePhase;
      if (productionBoundary && phase) encoder = productionBoundary(phase, encoder);
    };
    if (this.pendingFinePublication) {
      throw new Error("A transported fine generation must settle before another surface step");
    }
    if (this.fineSeedAdapter) {
      const preparationBroker = new PassBroker(encoder);
      // Fine seeds are rebuilt from current compact coarse phi before each
      // global-fine publication transaction.
      this.fineSeedAdapter.encode(preparationBroker);
      // The paper's Section 6 grading invariant applies to every leaf that
      // can support the current Section 5 fine band. Publish that bounded
      // face/edge tile ring now, before the inactive topology candidate is
      // balanced later in this substep. Deferring this transaction to sparse
      // render publication made balance consume generation N-1 while fine
      // topology N could already demand a newly reached tile.
      const fineSeedSource = this.fineSeedAdapter.source;
      this.topologyResidency.encodeFineSeedCandidates(
        preparationBroker.commandEncoder(),
        fineSeedSource.leaves,
        fineSeedSource.candidates.candidates,
        fineSeedSource.candidates.countAndDispatch,
      );
      this.topologyWorklistReady = true;
      const coarseBootstrappedThisStep
        = this.lane.encodeSurfaceCoarseBootstrap(preparationBroker);
      const coarseOnlyAdvance
        = this.lane.encodeCoarseOnlySurfaceAdvance(preparationBroker, dt_s, inflow);
      if (coarseOnlyAdvance) {
        encoder = coarseOnlyAdvance;
        if (productionBoundary) {
          encoder = productionBoundary("structuredProjectionTail", encoder);
        }
        return encoder;
      }
      if (this.globalFineSeeds && this.globalFineTopologyAB && this.globalFineTopologyBA
        && this.globalFineRedistanceA && this.globalFineRedistanceB) {
        // Re-emitting compact interface seeds is intentional: the GPU
        // publication transaction, not this host-side scheduling latch,
        // decides whether the first sparse authority exists. A rejected cold
        // generation can therefore retry on the next encoded step.
        const seedBroker = preparationBroker;
        // Once a delta publication has been encoded the 8-dispatch seed chain
        // is provably unread (POWER_LIQUIDS_ULTIMATE_M1MAX.md B1 / P1.1):
        // `insertExternalSeeds` and `externalAffineInterfaceBrick` are only
        // dispatched by the `kind: "bootstrap"` branch of the topology encode,
        // and the sole remaining reader, `externalSeedPhi` inside
        // `initializeDesiredSamples`, returns its non-finite sentinel whenever
        // `currentFinePopulated()`. The buffer identity is unchanged, so the
        // publication still binds the same affine seed source.
        const seeds = this.globalFineBootstrapped
          ? (this.finePopulated
            ? { buffer: this.globalFineSeeds.buffer, affineValues: true }
            : this.globalFineSeeds.encode(
              seedBroker,
              { buffer: this.fineSeedAdapter.leaves },
              { buffer: this.fineSeedAdapter.source.candidates.candidates },
              { buffer: this.fineSeedAdapter.source.candidateCount },
            ))
          : this.globalFineSeeds.encodeFromAllInterfaceLeaves(
            seedBroker, { buffer: this.fineSeedAdapter.leaves }, { buffer: this.compaction },
          );
        const compactCoarseEntry: GPUBindGroupEntry = this.lane.fineTopologyCoarseEntry(9);
        // Same planner as allocation. The final three cells cover the complete
        // 3-D trilinear stencil and its centre on the closed cutoff.
        const fineBandPlan = planFineLevelSetBandFineCells(
          this.fineLevelSetBandCells, this.globalFineLevelSet!.plan.fineFactor,
        );
        const { transportBandFineCells: bandCells,
          redistanceBandFineCells: redistanceBandCells, maximumBacktraceFineCells }
          = fineBandPlan;
        const transport = this.lane.fineTransportStage(this.globalFineCurrentIsA);
        let transportEncoded = false;
        // Adapter publication, coarse bootstrap and compact interface seeding
        // precede characteristic transport. Keep them out of the transport
        // bucket so the generic trace names the measured work.
        seedBroker.fence("fine interface seed publication complete");
        splitProductionPhase(undefined, "finePreparation");
        if (this.globalFineBootstrapped && transport) {
          const transportBroker = new PassBroker(encoder);
          this.lastGlobalFineTransport = transport;
          const completedTransportBroker = this.lane.encodeFineTransport(
            transportBroker, transport, {
              timestep: dt_s,
              ...(inflow ? { inflow } : {}),
              transportBandCells: bandCells,
              maximumBacktraceFineCells,
              openTopBoundary: this.scene.container.top !== "closed",
              dynamicBoundary: this.scene.rigidBodies.length > 0,
            });
          // Topology may reuse the shared physical payload pool. Capture the
          // transported old phi by logical sample before that reuse, then
          // intersect it with the new generation after topology publication.
          encoder = completedTransportBroker.commandEncoder();
          transportEncoded = true;
          splitProductionPhase(undefined, "fineTransport");
        }
        let publicationTopology: WebGPUFineLevelSetTopology;
        let publicationRedistance: WebGPUFineLevelSetRedistance;
        let publicationVolume: WebGPUFineLevelSetVolumeCorrection | undefined;
        let publicationTarget: WebGPUFineLevelSetBrickSource;
        const publicationTransport = transportEncoded ? transport : undefined;
        if (this.globalFineBootstrapped && !publicationTransport) {
          throw new Error("Recurring fine topology requires the transport phase-mask delta authority");
        }
        if (this.globalFineCurrentIsA) {
          if (this.globalFineBootstrapped) {
            this.globalFineGeneration += 1;
            this.globalFineLevelSet!.repurposeGPUGeneration(this.globalFineSourceB!, this.globalFineGeneration);
          }
          publicationTopology = this.globalFineTopologyAB;
          publicationRedistance = this.globalFineRedistanceB;
          publicationVolume = this.globalFineVolumeB;
          publicationTarget = this.globalFineSourceB!;
          const topologyBroker = new PassBroker(encoder);
          publicationTopology.encode(topologyBroker, seeds, [compactCoarseEntry], {
            // Match the two-finest-cell residency reserved at allocation.
            maximumBacktraceFineCells,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells: redistanceBandCells,
            safetyBrickRings: 1 + this.coarseDynamics.topology.extraDilationRings,
          }, true, this.globalFineBootstrapped
            ? { kind: "delta", producer: publicationTransport!.topologyDelta }
            : { kind: "bootstrap" }, inflow, this.scene.container.top !== "closed");
          encoder = topologyBroker.commandEncoder();
          splitProductionPhase("brickEngineA", "fineTopology");
        } else {
          this.globalFineGeneration += 1;
          this.globalFineLevelSet!.repurposeGPUGeneration(this.globalFineSourceA!, this.globalFineGeneration);
          publicationTopology = this.globalFineTopologyBA;
          publicationRedistance = this.globalFineRedistanceA;
          publicationVolume = this.globalFineVolumeA;
          publicationTarget = this.globalFineSourceA!;
          const topologyBroker = new PassBroker(encoder);
          publicationTopology.encode(topologyBroker, seeds, [compactCoarseEntry], {
            maximumBacktraceFineCells,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells: redistanceBandCells,
            safetyBrickRings: 1 + this.coarseDynamics.topology.extraDilationRings,
          }, true, this.globalFineBootstrapped
            ? { kind: "delta", producer: publicationTransport!.topologyDelta }
            : { kind: "bootstrap" }, inflow, this.scene.container.top !== "closed");
          encoder = topologyBroker.commandEncoder();
          splitProductionPhase("brickEngineA", "fineTopology");
        }
        if (publicationVolume && this.pendingSurfaceReferenceVolume_m3 > 0) {
          publicationVolume.addReferenceVolume(this.pendingSurfaceReferenceVolume_m3);
          this.pendingSurfaceReferenceVolume_m3 = 0;
        }
        const wasBootstrapped = this.globalFineBootstrapped;
        // Latch after the seed decision above, so the first delta publication
        // still carries a freshly emitted chain: that is the pre-acceptance
        // retry window the comment above describes. From the next step on the
        // chain is skipped (B1 / P1.1).
        if (wasBootstrapped) this.finePopulated = true;
        this.pendingFinePublication = {
          topology: publicationTopology,
          redistance: publicationRedistance,
          ...(publicationVolume ? { volume: publicationVolume } : {}),
          ...(publicationTransport ? { transport: publicationTransport } : {}),
          target: publicationTarget,
          targetIsA: !this.globalFineCurrentIsA,
          redistanceBandCells,
          maximumDisplacementFineCells: maximumBacktraceFineCells,
          warmClosestPoints: wasBootstrapped,
        };
        // On recurring steps, coarse phi consumes the transported target before
        // any current-step force. Bootstrap first needs redistance to populate
        // the complete narrow band, so its sole correction occurs at settlement.
        if (wasBootstrapped && !coarseBootstrappedThisStep) {
          const coarseBroker = new PassBroker(encoder);
          this.lane.encodeCoarsePhiBeforeForces(coarseBroker, publicationTarget,
            publicationTopology, dt_s);
          coarseBroker.fence("transported fine and coarse phi published before forces");
          encoder = coarseBroker.commandEncoder();
        }
        if (!wasBootstrapped || dt_s === 0) {
          encoder = this.encodePendingFineSettlement(encoder, productionBoundary);
          this.lane.requireSettledSupport();
          encoder = this.lane.encodeSettledSupport(encoder, dt_s, "t0");
          if (productionBoundary) {
            encoder = productionBoundary("structuredProjectionTail", encoder);
          }
        }
      } else {
        // No fine band was allocated. The lane's own coarse authority is the
        // sole moving surface here; a lane without one leaves the pipeline
        // incomplete rather than silently advancing nothing.
        const fallback = this.lane.encodeCoarseOnlyFallbackAdvance(
          preparationBroker, dt_s, coarseBootstrappedThisStep);
        if (!fallback) {
          throw new Error("Authoritative Section 5 fine-band pipeline is incomplete");
        }
        encoder = fallback;
        if (productionBoundary) {
          encoder = productionBoundary("structuredProjectionTail", encoder);
        }
      }
      return encoder;
    }
    return encoder;
  }
  addSurfaceReferenceVolumeCells(cells: number) {
    if (!Number.isFinite(cells) || cells < 0) {
      throw new RangeError("Octree inflow reference cells must be finite and non-negative");
    }
    this.surfaceState.addReferenceVolumeCells(cells);
    const cellVolume = (this.scene.container.width_m / this.dims.nx)
      * (this.scene.container.height_m / this.dims.ny)
      * (this.scene.container.depth_m / this.dims.nz);
    this.pendingSurfaceReferenceVolume_m3 += cells * cellVolume;
  }
  async readSolveDiagnostics() {
    // A lane with its own solve control publishes its own receipt; only the
    // shared row/residual staging below is engine-owned.
    if (await this.lane.readSolveDiagnostics()) return;
    // The staging buffer was copied inside the solve encoder, so it can never
    // race the next rebuild's worklist copy over the compaction header. It
    // carries [overflow, required rows, required entries, exact dispatch xyz,
    // sum r^2, sum b^2] from the latest solve.
    const solverControl = this.pressureSolverControl;
    const solverBytes = 64;
    const readback = this.device.createBuffer({
      label: "Octree live pressure-row diagnostics",
      size: 32 + solverBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read octree pressure-row diagnostics" });
    encoder.copyBufferToBuffer(this.solveStats, 0, readback, 0, 32);
    encoder.copyBufferToBuffer(solverControl, 0, readback, 32, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange(0, 32 + solverBytes);
      const words = new Uint32Array(mapped, 0, 8);
      const residuals = new Float32Array(words.buffer, words.byteOffset + 24, 2);
      const overflow = words[0] !== 0;
      const liquidRows = words[1];
      this.info.pressureCapacityOverflow = overflow;
      this.info.frontierCapacityOverflow = (words[0] & 2) !== 0;
      this.info.frontierRequiredLeaves = words[1];
      this.info.pressureRequiredRows = words[1];
      this.info.pressureSampleCount = liquidRows;
      this.info.liquidDofCount = liquidRows;
      this.info.compressionRatio = liquidRows / Math.max(1, this.dims.nx * this.dims.ny * this.dims.nz);
      if (!overflow && liquidRows > 0) {
        const rr = residuals[0], bb = residuals[1];
        if (Number.isFinite(rr) && Number.isFinite(bb) && rr >= 0 && bb >= 0) {
          this.residualRms = Math.sqrt(rr / liquidRows);
          this.initialResidualRms = Math.sqrt(bb / liquidRows);
          this.relativeResidual = Math.sqrt(rr / Math.max(bb, 1e-30));
        } else {
          this.residualRms = undefined;
          this.initialResidualRms = undefined;
          this.relativeResidual = undefined;
        }
      } else {
        this.residualRms = undefined;
        this.initialResidualRms = undefined;
        this.relativeResidual = undefined;
      }
      this.applyMGPCGDiagnostics(new Uint32Array(mapped, 32, 16));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** One-time startup proof for the paper's Section 4.3 pressure authority.
   * Regular simulation scheduling never consumes this readback; the paused
   * t=0 transport gate uses it only after every initialization phase fenced. */
  async readMGPCGDiagnostics() {
    const solverControl = this.pressureSolverControl;
    const readback = this.device.createBuffer({
      label: "Octree t=0 MGPCG authority diagnostics",
      size: 128,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read octree t=0 MGPCG authority" });
    encoder.copyBufferToBuffer(solverControl, 0, readback, 0, 128);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = Uint32Array.from(new Uint32Array(readback.getMappedRange(0, 128)));
      this.applyMGPCGDiagnostics(words);
      return words;
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  applyMGPCGDiagnostics(words: Uint32Array) {
    if (words.length < 16) return;
    const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
    const rr = floats[10] + floats[11], bb = floats[8] + floats[9];
    this.info.pressureIterationsUsed = words[2];
    this.info.pressureConverged = words[0] === 0 && words[1] !== 0;
    if (Number.isFinite(rr) && Number.isFinite(bb) && rr >= 0 && bb >= 0) {
      const rows = Math.max(1, words[4]);
      this.residualRms = Math.sqrt(rr / rows);
      this.initialResidualRms = Math.sqrt(bb / rows);
      this.relativeResidual = Math.sqrt(rr / Math.max(bb, 1e-30));
    }
  }

  get surfaceDiagnostics() {
    return this.surfaceState.volumeDiagnostics;
  }
  async readSurfaceDiagnostics() {
    return this.surfaceState.readVolumeDiagnostics();
  }
  /** Presentation-only texture identity. The sparse octree solver never samples it. */
  get levelSetTexture() { return this.surfaceState.texture; }
  encodeBodyImpulseReadback() { return undefined; }
  readBodyImpulseReadback() { return Promise.resolve([]); }
  destroySharedSurface() { /* The octree owns its surface for its full lifetime. */ }
  get hasDenseLevelSetPublication() { return !this.denseBootstrapPhiReleased; }
  /** Release the last box-sized phi field after its bootstrap commands submit. */
  releaseDenseBootstrapPhi() {
    if (this.denseBootstrapPhiReleased) return 0;
    // Rigid/terrain coupling, the differential, and scientific overlays still
    // consume dense phi and therefore explicitly gate lifetime cutover. Every
    // recurring compact consumer must also attest that its bind group was
    // rebuilt onto page buffers plus the live format-only presentation texture.
    if (!octreeDensePhiReleaseReady({
      globalFineBootstrapped: this.globalFineBootstrapped,
      coarseProjectionGroupsActive: this.lane.hasCoarseProjectionGroups,
      fineSeedCoarseNative: this.fineSeedAdapter?.hasCoarsePhiBindings === true,
      topologyUsesFineSeedCandidates: this.topologyWorklistReady,
      compactRendererSourceReady: this.lane.compactRendererSourceReady() && this.fineSeedAuthority,
      incompatibleDenseConsumer: Boolean(this.diagnosticGroups
        || !this.globalFineBootstrapped
        || this.scene.rigidBodies.length > 0 || sceneHasTerrain(this.scene)),
    })) return 0;
    const releasedBytes = this.surfaceState.releasePresentationTexture();
    if (releasedBytes > 0) {
      this.denseBootstrapPhiReleased = true;
      this.info.allocatedBytes = Math.max(0, this.info.allocatedBytes - releasedBytes);
      this.surfaceStateAccountingBytes = Math.max(0,
        this.surfaceStateAccountingBytes - releasedBytes);
    }
    return releasedBytes;
  }
  get sparseVoxelSceneSource() { return this.sparseBrickWorld?.sceneSource; }
  rescaleSparsePresentation(scene: SceneDescription) {
    this.sparseBrickWorld?.rescaleRenderDomain(scene);
  }
  stageSceneUpdate(scene: SceneDescription) {
    return this.sparseBrickWorld?.stageSceneUpdate(scene) ?? false;
  }
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]) {
    return this.sparseBrickWorld?.stageLivePrimitiveUpdates(updates) ?? false;
  }
  encodeSceneMaintenance(encoder: GPUCommandEncoder) {
    return this.sparseBrickWorld?.encodeSceneMaintenance(encoder) ?? false;
  }
  // QA-only lane structure names. Every consumer of these reaches the
  // projection structurally, so the names have to survive the lane split even
  // though nothing behind them belongs to the engine any more.
  get structuredVelocityControl() { return this.lane.debug.structuredVelocityControl; }
  get structuredBoundaryControl() { return this.lane.debug.structuredBoundaryControl; }
  get structuredRowVelocities() { return this.lane.debug.structuredRowVelocities; }
  get structuredAuthority() { return this.lane.debug.structuredAuthority; }
  get structuredWorksets() { return this.lane.debug.structuredWorksets; }
  /** QA-only MGPCG status; simulation authority consumes this buffer directly on GPU. */
  get mgpcgControl() { return this.pressureSolverControl; }
  get powerDescriptorControl() { return this.lane.debug.powerDescriptorControl; }
  get powerTopologyControl() { return this.lane.debug.powerTopologyControl; }
  get powerDescriptorRows() { return this.lane.debug.powerDescriptorRows; }
  get powerTopologyMetrics() { return this.lane.debug.powerTopologyMetrics; }
  get powerCatalogEntryHeaders() { return this.lane.debug.powerCatalogEntryHeaders; }
  get powerCatalogFaces() { return this.lane.debug.powerCatalogFaces; }
  get techniqueDebugSource() {
    const surface = this.fineSeedAdapter?.source;
    const technique = this.lane.techniqueDebugSources();
    const adaptiveVelocity = this.lane.debug.adaptiveVelocity;
    if (!surface || !technique) return undefined;
    const fine = this.globalFineBootstrapped
      ? (this.globalFinePublishedIsA ? this.globalFineSourceA : this.globalFineSourceB)
      : undefined;
    const fineTopology = this.globalFinePublishedIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    const fineRedistance = this.globalFinePublishedIsA ? this.globalFineRedistanceA : this.globalFineRedistanceB;
    const fineBandLifecycle = fine && fineTopology && fineRedistance ? {
      params: { buffer: fine.params },
      metadata: { buffer: fine.metadata },
      worklist: { buffer: fine.worklist },
      samples: { buffer: fine.samples },
      topologyControl: { buffer: fineTopology.control },
      redistanceControl: { buffer: fineRedistance.control },
      seeds: { buffer: fine.workA },
      // The derived widths come from the planner the solver itself runs, so the
      // view cannot drift from the band that was actually allocated.
      bands: {
        pressureBandCells: this.interfaceRefinementBandCells,
        surfaceBandCells: this.fineLevelSetBandCells,
        ...planFineLevelSetBandFineCells(this.fineLevelSetBandCells, fine.plan.fineFactor),
        // The ladder the redistancer actually emitted, not a re-derivation: the
        // warm and cold paths choose different repair counts.
        ladderStrides: fineRedistance.lastEncodedStrides,
      },
    } : undefined;
    return {
      leaves: { buffer: surface.leaves },
      topologyMetrics: { buffer: technique.topologyMetrics },
      catalogEntryHeaders: { buffer: technique.catalogEntryHeaders },
      catalogFaces: { buffer: technique.catalogFaces },
      tetrahedronHeaders: { buffer: technique.tetrahedronHeaders },
      tetrahedra: { buffer: technique.tetrahedra },
      tetrahedronVertices: { buffer: technique.tetrahedronVertices },
      structuredAuthority: { buffer: technique.structuredAuthority },
      structuredParams: { buffer: technique.structuredParams },
      structuredRowGeometry: { buffer: technique.structuredRowGeometry },
      structuredRowVelocities: { buffer: technique.structuredRowVelocities },
      structuredControl: { buffer: technique.structuredControl },
      ...(adaptiveVelocity ? {
        losassoAdaptiveVelocity: {
          control: { buffer: adaptiveVelocity.control },
          leaves: { buffer: adaptiveVelocity.leaves },
          nodalVelocity: { buffer: adaptiveVelocity.nodalVelocity },
          phiControl: { buffer: adaptiveVelocity.phiControl },
          rowPhi: { buffer: adaptiveVelocity.rowPhi },
          extensionReach_m: adaptiveVelocity.extensionReach_m,
        },
      } : {}),
      // A lane that runs its own solve owns the authoritative bank this step;
      // otherwise the engine's own A/B latch names it.
      pressure: this.lane.debug.pressureFrameView
        ?? { buffer: this.latestPressureInA ? this.pressureA : this.pressureB },
      leafHeaders: { buffer: this.leafHeaders },
      ...(technique.coarsePhiValues ? { coarsePhi: { buffer: technique.coarsePhiValues } } : {}),
      topologyLifecycle: {
        tileWorklist: { buffer: this.topologyResidency.tileWorklist },
        tileDimensions: [
          Math.ceil(this.dims.nx / this.topologyTileSize),
          Math.ceil(this.dims.ny / this.topologyTileSize),
          Math.ceil(this.dims.nz / this.topologyTileSize),
        ] as const,
        tileSizeCells: this.topologyTileSize,
        tileCapacity: this.topologyResidency.tileCapacity,
      },
      ...(fineBandLifecycle ? { fineBandLifecycle } : {}),
      pressureRows: {
        dimensions: [this.dims.nx, this.dims.ny, this.dims.nz] as const,
        rowCapacity: this.pressureCapacity.rowCapacity,
      },
      generation: this.powerAttemptGeneration,
    };
  }
  /** Paired same-generation energy reduction produced around pressure projection. */
  get structuredProjectionEnergyStats(): GPUBuffer | undefined {
    return this.lane.debug.structuredProjectionEnergyStats;
  }
  /** QA-only boundary/air-support symmetry sources, whichever lane owns them. */
  get structuredBoundarySymmetryDebug() { return this.lane.structuredBoundarySymmetryDebug; }
  /** Failure-only decode of the compact row a lane's candidate publication rejected. */
  readPowerCoarseFailureRow(row: number) { return this.lane.forensics.coarseFailureRow?.(row); }
  /** Accepted generation is intentionally not guessed from the host attempt
   * counter. GPU audit paths obtain it from the accepted structured control. */
  get powerPublicationGeneration(): number | undefined { return undefined; }
  get powerLeafHeaders() { return this.leafHeaders; }
  /** QA-only inactive compact pressure headers for failed candidate diagnosis. */
  get powerCandidateLeafHeaders() { return this.candidateLeafHeaders; }
  /** QA-only active compact pressure potential, indexed by leaf row. */
  get powerPressureBuffer() { return this.latestPressureInA ? this.pressureA : this.pressureB; }
  /** QA-only buffers for the cold-to-recurring sparse-topology acceptance gate. */
  get powerLeafFrontier() { return this.leafFrontier; }
  /** QA-only compact row-publication header and fail-closed control tail. */
  get powerCompactionControl() { return this.compaction; }
  /** QA-only per-tile authority-change stamps used to build the exact dirty list. */
  get powerTopologyTileChangeFlags() {
    return {
      buffer: this.compaction,
      offsetBytes: this.topologyTileChangeFlagsOffsetBytes,
      byteLength: this.topologyTileChangeFlagsByteLength,
    };
  }
  /** QA-only raw sparse topology-tile membership used by the exact
   * fine-page-delta scheduler. Read only after a rejected publication. */
  get powerTopologyTileStates() {
    return {
      buffer: this.topologyResidency.topologyTileStateBuffer,
      byteLength: this.topologyResidency.allocationPlan.tileStateBytes,
      sparse: this.topologyResidency.allocationPlan.sparseKeyPools,
    };
  }
  /** QA-only carry-classification flags. Bit zero is keep, bits 1..4 encode
   * clean-row identity/wetness rejection, and bit five marks a dirty row. */
  get powerFrontierCarryFlags() {
    return {
      buffer: this.compaction,
      offsetBytes: this.compactionAllocationRowDeltaScratchOffsetBytes,
      byteLength: this.pressureCapacity.rowCapacity * 4,
    };
  }
  get powerRowDelta() {
    return {
      rows: this.leafFrontier,
      rowCapacity: this.frontierAllocation.listCapacity,
      controlOffsetWords: this.frontierAllocation.rowDeltaControlOffsetWords,
      newToOldOffsetWords: this.frontierAllocation.rowDeltaNewToOldOffsetWords,
      oldToNewOffsetWords: this.frontierAllocation.rowDeltaOldToNewOffsetWords,
      dirtyRowsOffsetWords: this.frontierAllocation.rowDeltaDirtyRowsOffsetWords,
      affectedRowsOffsetWords: this.frontierAllocation.rowDeltaAffectedRowsOffsetWords,
    };
  }
  get topologyTileWorklist() { return this.topologyResidency.tileWorklist; }
  /** Failure-only cold/recurring frontier headers. The immutable frontier
   * selector/counts and compact scheduler words identify the first zero
   * publication without reading any row payload or influencing authority. */
  async readPowerFrontierFailure() {
    const layout = OCTREE_FRONTIER_FAILURE_LAYOUT;
    const readback = this.device.createBuffer({
      label: "Octree power-frontier failure readback",
      size: layout.totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read octree power-frontier failure headers",
    });
    const capture = (region: OctreeFrontierFailureRegion, source: GPUBuffer, sourceOffset = 0) => {
      const span = layout.span(region);
      const byteLength = Math.min(span.count * 4, Math.max(0, source.size - sourceOffset));
      if (byteLength > 0) {
        encoder.copyBufferToBuffer(source, sourceOffset, readback, span.bytes, byteLength);
      }
    };
    capture("frontier", this.leafFrontier);
    capture("compaction", this.compaction);
    capture("reservedControl", this.compaction, this.dirtyFailureOffsetBytes);
    capture("frontierFailure", this.compaction, this.compaction.size - 8 * 4);
    capture("frontierPublication", this.compaction, this.frontierPublicationOffsetBytes);
    capture("dirtyAuthorityState", this.compaction, this.dirtyAuthorityStateOffsetBytes);
    this.lane.captureFrontierFailureAuthorityControls(capture);
    capture("rowDelta", this.leafFrontier,
      this.frontierAllocation.rowDeltaControlOffsetWords * 4);
    capture("ownerCandidate", this.ownerPages.candidateTransaction);
    capture("carryFlags", this.compaction, this.compactionAllocationRowDeltaScratchOffsetBytes);
    if (this.globalFineSummaries) {
      capture("fineSummaryDirectory", this.globalFineSummaries.directory);
      capture("fineSummaryWorkState", this.globalFineSummaries.diagnosticBuffers.workState);
    }
    this.lane.captureFrontierFailureCoarseSources(capture);
    const fineTopology = this.globalFinePublishedIsA
      ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    if (fineTopology) capture("finePageDelta", fineTopology.pageDelta);
    // Keep the first descriptor wave beside its exact row-delta inputs. A
    // malformed compact-list publication otherwise gets overwritten by the
    // coupled poison flag before a failure-only inspection can distinguish
    // list ordering from descriptor geometry.
    capture("rowDeltaNewToOld", this.leafFrontier,
      this.frontierAllocation.rowDeltaNewToOldOffsetWords * 4);
    capture("rowDeltaAffectedRows", this.leafFrontier,
      this.frontierAllocation.rowDeltaAffectedRowsOffsetWords * 4);
    this.lane.captureFrontierFailureCandidateSources(capture);
    // The three compact schedules the emission/sort/carry stages actually
    // consume, beside the head of the compact candidate list they fill. A
    // published row count with an empty candidate record is invisible in the
    // control words alone.
    capture("candidateSchedules", this.topologyCandidateDispatch);
    capture("frontierCandidates", this.leafFrontier,
      this.frontierAllocation.candidateOffsetWords * 4);
    this.device.queue.submit([encoder.finish()]);
    let result: {
      frontier: number[];
      compaction: number[];
      dirtyAuthority: number[];
      frontierFailure: number[];
      frontierPublication: number[];
      dirtyAuthorityState: number[];
      descriptorCandidate: number[];
      topologyCandidate: number[];
      structuredCandidate: number[];
      boundaryCandidate: number[];
      spgridCandidate: number[];
      epoch: number[];
      rowDelta: number[];
      ownerCandidate: number[];
      carryFlags: number[];
      fineSummaryDirectory: number[];
      fineSummaryWorkState: number[];
      coarseControl: number[];
      coarseDirectory: number[];
      coarseDelta: number[];
      finePageDelta: number[];
      rowDeltaNewToOld: number[];
      rowDeltaAffectedRows: number[];
      descriptorCandidates: number[];
      descriptorStatuses: number[];
      structuredDispatch: number[];
      candidateSchedules: number[];
      frontierCandidates: number[];
      spgridLevelDelta: readonly number[];
      spgridCandidateDispatch: readonly number[];
      controlSummary?: Record<string, number[]>;
      descriptorFailureRow?: unknown;
      boundaryFailureRow?: unknown;
      coarseFailureRow?: unknown;
    };
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const decode = (region: OctreeFrontierFailureRegion) => {
        const span = layout.span(region);
        return Array.from(words.slice(span.words, span.words + span.count));
      };
      result = {
        frontier: decode("frontier"),
        compaction: decode("compaction"),
        dirtyAuthority: decode("reservedControl"),
        frontierFailure: decode("frontierFailure"),
        frontierPublication: decode("frontierPublication"),
        dirtyAuthorityState: decode("dirtyAuthorityState"),
        descriptorCandidate: decode("descriptorCandidate"),
        topologyCandidate: decode("topologyCandidate"),
        structuredCandidate: decode("structuredCandidate"),
        boundaryCandidate: decode("boundaryCandidate"),
        spgridCandidate: decode("spgridCandidate"),
        epoch: decode("epoch"),
        rowDelta: decode("rowDelta"),
        ownerCandidate: decode("ownerCandidate"),
        carryFlags: decode("carryFlags"),
        fineSummaryDirectory: decode("fineSummaryDirectory"),
        fineSummaryWorkState: decode("fineSummaryWorkState"),
        coarseControl: decode("coarseControl"),
        coarseDirectory: decode("coarseDirectory"),
        coarseDelta: decode("coarseDelta"),
        finePageDelta: decode("finePageDelta"),
        rowDeltaNewToOld: decode("rowDeltaNewToOld"),
        rowDeltaAffectedRows: decode("rowDeltaAffectedRows"),
        descriptorCandidates: decode("descriptorCandidates"),
        descriptorStatuses: decode("descriptorStatuses"),
        structuredDispatch: decode("structuredDispatch"),
        candidateSchedules: decode("candidateSchedules"),
        frontierCandidates: decode("frontierCandidates"),
        spgridLevelDelta: [],
        spgridCandidateDispatch: [],
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    // The lane's own follow-up readbacks run after the shared block is
    // unmapped, so a row it fetches is chased from the same control words the
    // caller sees rather than from a second, later publication.
    Object.assign(result, await this.lane.decodeFrontierFailure(result));
    // Repeat only the compact producer controls at the tail of the serialized
    // error. Large row samples are commonly truncated by consoles precisely
    // where the originating publisher verdict would otherwise be lost.
    result.controlSummary = {
      frontier: result.frontier,
      dirtyAuthority: result.dirtyAuthority,
      frontierFailure: result.frontierFailure,
      frontierPublication: result.frontierPublication,
      dirtyAuthorityState: result.dirtyAuthorityState,
      descriptor: result.descriptorCandidate,
      topology: result.topologyCandidate,
      structured: result.structuredCandidate,
      boundary: result.boundaryCandidate,
      spgrid: result.spgridCandidate,
      epoch: result.epoch,
      ownerCandidate: result.ownerCandidate,
    };
    return result;
  }
  /** Failure-only readback of the immutable sparse owner-page control header. */
  async readOwnerPageControl(): Promise<readonly number[]> {
    const readback = this.device.createBuffer({
      label: "Octree owner-page control readback",
      size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read octree owner-page control" });
    encoder.copyBufferToBuffer(this.ownerPages.arena, 0, readback, 0, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      return Array.from(new Uint32Array(readback.getMappedRange()));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  /** Debug-only QA readback of the canonical sparse owner-page arena. */
  get ownerLatticeDebug(): {
    buffer: GPUBuffer;
    maximumLeafSize: number;
    dimensions: readonly [number, number, number];
  } {
    return { buffer: this.ownerPages.arena,
      maximumLeafSize: this.topologyMaximumLeafSize,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz] };
  }
  /** QA-only compact surface producer header feeding recurring topology residency. */
  get fineSeedCandidateControl() { return this.fineSeedAdapter?.source.candidateCount; }
  /** QA-only compact affine leaves classified by the recurring topology producer. */
  get fineSeedLeaves() { return this.fineSeedAdapter?.source.leaves; }
  get powerOwnerArena() { return this.ownerPages.arena; }
  get nativePowerVelocityAuthority() { return this.lane.compactRendererSourceReady(); }
  // Copy sources for the step-snapshot ring (POWER_LIQUIDS_ULTIMATE_M1MAX A4).
  // The ring appends these copies after every producer in the step's own
  // encoder, so a mapped record shows the step's own verdicts. Absent is not
  // zero: a missing source decodes as absent and refuses to authorize a skip.
  get topologyEpochState(): GPUBuffer | undefined { return this.lane.debug.topologyEpochState; }
  get airSupportScratch(): GPUBuffer | undefined { return this.lane.debug.airSupportScratch; }
  get spgridLevelDelta(): GPUBuffer | undefined { return this.lane.debug.spgridLevelDelta; }

  // --- Coarse dynamics lane diagnostics -----------------------------------
  // These names are the published diagnostic ABI. The step-snapshot ring, the
  // smoke harness and the `tools/probe-*` scripts all reach this object
  // structurally, so the projection keeps the surface and the lane that owns
  // the machinery answers. A lane without it answers `undefined` rather than
  // a fabricated receipt, which is exactly what the ring reads as "absent".
  get losassoAuthorityControl() { return this.lane.debug.losassoAuthorityControl; }
  get losassoCoarsePhiControl() { return this.lane.debug.losassoCoarsePhiControl; }
  get losassoExtensionControl() { return this.lane.debug.losassoExtensionControl; }
  get losassoAdaptiveAcceptedGraphControl() {
    return this.lane.debug.losassoAdaptiveAcceptedGraphControl;
  }
  get losassoAdaptiveCandidateGraphControl() {
    return this.lane.debug.losassoAdaptiveCandidateGraphControl;
  }
  get losassoAdaptivePhiControl() { return this.lane.debug.losassoAdaptivePhiControl; }
  get losassoAdaptivePhiReceipts() { return this.lane.debug.losassoAdaptivePhiReceipts; }
  get losassoAdaptiveVelocityReceipts() {
    return this.lane.debug.losassoAdaptiveVelocityReceipts;
  }
  get losassoAdaptiveRendererDirectory() {
    return this.lane.debug.losassoAdaptiveRendererDirectory;
  }
  get losassoCandidateAuthorityControl() {
    return this.lane.debug.losassoCandidateAuthorityControl;
  }
  get losassoAdaptiveMassControl() { return this.lane.debug.losassoAdaptiveMassControl; }
  get losassoAdaptiveMassReceipts() { return this.lane.debug.losassoAdaptiveMassReceipts; }
  get losassoAdaptiveCandidateMassControl() {
    return this.lane.debug.losassoAdaptiveCandidateMassControl;
  }
  get losassoAdaptiveCandidateMassReceipts() {
    return this.lane.debug.losassoAdaptiveCandidateMassReceipts;
  }
  get losassoCandidateVelocityMigrationReceipt() {
    return this.lane.debug.losassoCandidateVelocityMigrationReceipt;
  }
  /** Observational sealed-wet-owner tripwire, never consumed by simulation. */
  get rigidCouplingDiagnosticBuffer() {
    return this.lane.debug.rigidCouplingDiagnosticBuffer;
  }
  get rigidBoundaryRefreshDiagnosticBuffer() {
    return this.lane.debug.rigidBoundaryRefreshDiagnosticBuffer;
  }
  get losassoVelocityDebug() { return this.lane.velocityDebug; }
  get losassoPressureDebug() { return this.lane.pressureDebug; }
  get losassoFrontierDebug() { return this.lane.frontierDebug; }
  get losassoCoarsePhiDebug() { return this.lane.coarsePhiDebug; }
  /** Decode the pressure receipt copied by the production step snapshot. */
  applyLosassoStepDiagnostics(authority: Uint32Array, solver: Uint32Array): void {
    this.lane.applyStepDiagnostics(authority, solver);
  }
  /** Diagnostic-only receipt for the factor-one surface authority. */
  readCoarseSurfaceTrackerReceipt() { return this.lane.readCoarseSurfaceTrackerReceipt(); }
  /**
   * Fail-closed t=0 validation of the lane's own published authority. The
   * receipt words are the lane's private ABI, so the lane -- not this shell --
   * decides whether the paused startup tuple is admissible.
   */
  validateInitialLaneAuthority(context: {
    readonly dimensions: readonly [number, number, number];
    readonly refreshInfo: () => void;
  }) {
    return this.lane.validateInitialAuthority(context);
  }
  readLosassoPreconditionerContraction() {
    return this.lane.forensics.preconditionerContraction?.();
  }
  readLosassoHierarchyCensus() { return this.lane.forensics.hierarchyCensus?.(); }
  readLosassoAuthorityDiagnostics() { return this.lane.forensics.authorityDiagnostics?.(); }
  readAdaptiveSurfacePublicationDiagnostics() {
    return this.lane.forensics.adaptiveSurfacePublication?.();
  }
  readAdaptiveNodeReceipt() { return this.lane.forensics.adaptiveNodeReceipt?.(); }
  readAdaptiveCandidateGraphReceipt() {
    return this.lane.forensics.adaptiveCandidateGraphReceipt?.();
  }
  readAdaptiveVelocityReceipts() { return this.lane.forensics.adaptiveVelocityReceipts?.(); }
  readAdaptiveVelocityDiagnostics() {
    return this.lane.forensics.adaptiveVelocityDiagnostics?.();
  }
  /** End-of-step diagnostic source for the fine generation whose publication
   * was just encoded. `globalFinePublishedIsA` advances only after submission,
   * so it still names the preceding generation while the snapshot copies run
   * at the tail of the current command encoder. */
  get globalFineCurrentWorklist(): GPUBuffer | undefined {
    if (!this.globalFineBootstrapped) return undefined;
    return (this.globalFineCurrentIsA
      ? this.globalFineSourceA : this.globalFineSourceB)?.worklist;
  }
  /** End-of-step verdict for the same fine slot returned above. Unlike the
   * public diagnostic getter, this follows the encoder-time current slot so
   * the snapshot cannot pair generation N's worklist with generation N-1's
   * topology transaction. */
  get globalFineCurrentTopologyControl(): GPUBuffer | undefined {
    if (!this.globalFineBootstrapped) return undefined;
    return (this.globalFineCurrentIsA
      ? this.globalFineTopologyBA : this.globalFineTopologyAB)?.control;
  }
  get globalFineCurrentRedistanceControl(): GPUBuffer | undefined {
    if (!this.globalFineBootstrapped) return undefined;
    return (this.globalFineCurrentIsA
      ? this.globalFineRedistanceA : this.globalFineRedistanceB)?.control;
  }
  get globalFineCurrentVolumeControl(): GPUBuffer | undefined {
    if (!this.globalFineBootstrapped) return undefined;
    // Both physical helpers intentionally share this transaction control.
    return this.globalFineVolumeA?.control;
  }
  /** Minimal production telemetry retained after hierarchical accounting was removed. */
  get workAccountingBuffers(): Readonly<{
    fineTransportGovernor?: GPUBufferBinding;
    pressureRhs?: GPUBufferBinding;
    section63Coefficients?: GPUBufferBinding;
    symmetryInitialResidual?: GPUBufferBinding;
    symmetryInitialPreconditioned?: GPUBufferBinding;
    symmetryInitialPreconditionedImage?: GPUBufferBinding;
    symmetryPreconditionerPreSmoothed?: GPUBufferBinding;
    symmetryPreconditionerZeroSmoothed?: GPUBufferBinding;
    symmetryPreconditionerFirstOperatorImage?: GPUBufferBinding;
    symmetryPreconditionerFirstSmoothed?: GPUBufferBinding;
    symmetryPreconditionerInnerResidual?: GPUBufferBinding;
    symmetryPreconditionerInnerCorrection?: GPUBufferBinding;
    symmetryPreconditionerPostCorrected?: GPUBufferBinding;
  }> | undefined {
    // Only the staged Power transport keeps a governor; the direct face
    // transport has none, and its absence is what this view tests for.
    const governor = this.lastGlobalFineTransport?.governor;
    const fineTransportGovernor = governor
      ? { buffer: governor, size: 4 * (4 + 64) } : undefined;
    const { pressureRhs, section63Coefficients } = this.lane.workAccountingBuffers();
    const symmetry = this.lane.solverSymmetryStageAuditBuffers;
    return fineTransportGovernor || pressureRhs || section63Coefficients || symmetry ? Object.freeze({
      ...(fineTransportGovernor ? { fineTransportGovernor } : {}),
      ...(pressureRhs ? { pressureRhs } : {}),
      ...(section63Coefficients ? { section63Coefficients } : {}),
      ...(symmetry ? {
        symmetryInitialResidual: { buffer: symmetry.initialResidual },
        symmetryInitialPreconditioned: { buffer: symmetry.initialPreconditioned },
        symmetryInitialPreconditionedImage: { buffer: symmetry.initialPreconditionedImage },
        symmetryPreconditionerPreSmoothed: { buffer: symmetry.preconditionerPreSmoothed },
        symmetryPreconditionerZeroSmoothed: { buffer: symmetry.preconditionerZeroSmoothed },
        symmetryPreconditionerFirstOperatorImage: {
          buffer: symmetry.preconditionerFirstOperatorImage,
        },
        symmetryPreconditionerFirstSmoothed: { buffer: symmetry.preconditionerFirstSmoothed },
        symmetryPreconditionerInnerResidual: { buffer: symmetry.preconditionerInnerResidual },
        symmetryPreconditionerInnerCorrection: { buffer: symmetry.preconditionerInnerCorrection },
        symmetryPreconditionerPostCorrected: { buffer: symmetry.preconditionerPostCorrected },
      } : {}),
    }) : undefined;
  }
  get workAccountingPlan(): Readonly<{
    pressure: Readonly<{ maximumOuterIterations: number }>;
  }> {
    return Object.freeze({ pressure: Object.freeze({
      maximumOuterIterations: this.lane.solverIterationBudget()
        ?? this.solveTailPolicy.encodedOuterIterations,
    }) });
  }
  captureWorkAccounting() {
    return Object.freeze({
      pressure: Object.freeze({ report: null,
        blocker: "hierarchical work accounting was retired" }),
      snapshot: this.workAccounting.snapshot(),
    });
  }

  /** Authoritative narrow-band fine phi for rendering and surface transport.
   * Topology sizing and pressure fractions still require the terminal coarse-phi cutover. */
  get globalFineLevelSetSource(): WebGPUFineLevelSetBrickSource | undefined {
    if (!this.globalFineLevelSet || !this.globalFineBootstrapped) return undefined;
    const fine = this.globalFinePublishedIsA ? this.globalFineSourceA : this.globalFineSourceB;
    if (!fine) return undefined;
    // Generic rendering and QA consumers use the shared eight-word coarse
    // directory ABI. A lane's private topology-sampling structure (the Losasso
    // arena) must never leak through this backend-neutral source, so the lane
    // publishes the neutral pair rather than its own object.
    const coarse = this.lane.genericCoarseDirectory();
    const topology = this.globalFinePublishedIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    return { ...fine,
      ...(coarse ? { coarsePhiDirectory: coarse.directory, coarsePhiRowCapacity: coarse.rowCapacity } : {}),
      ...(topology ? { topologyControl: topology.control } : {}),
      ...(this.globalFineSeeds ? { seedControl: this.globalFineSeeds.buffer } : {}),
    };
  }
  /** Authored surface resolution even when factor one deliberately allocates
   * no separate global-fine source. */
  get surfaceTrackingFactor(): 1 | 4 | 8 {
    return this.coarseOnlySurfaceTracking ? 1 : this.globalFineLevelSet?.plan.fineFactor ?? 4;
  }
  /** Renderer-only view of the sole moving surface in coarse-1 mode. */
  get coarseLevelSetSource() {
    if (!this.coarseOnlySurfaceTracking) {
      return undefined;
    }
    // Factor-one physics and rendering must select the same scalar authority,
    // so the lane that owns the surface names it -- the engine never picks
    // between two candidate directories and can never pair one lane's
    // generation with another's row directory.
    const coarse = this.lane.coarseLevelSetPublication();
    if (!coarse || coarse.generation < 1) return undefined;
    return {
      kind: "coarse-levelset-sampling" as const,
      directory: { buffer: coarse.directory },
      control: { buffer: coarse.control },
      ...(coarse.gradients ? { gradients: { buffer: coarse.gradients } } : {}),
      rowCapacity: coarse.rowCapacity,
      sampleDimensions: [this.dims.nx, this.dims.ny, this.dims.nz] as const,
      physicalCellSize: this.scene.container.width_m / this.dims.nx,
      domainOrigin: [0, 0, 0] as const,
      generation: coarse.generation,
    };
  }

  /**
   * The published fine generation paired with the redistancer that produced it.
   *
   * Flood-provenance diagnostics need both: the buffers hold the seed links,
   * and only the redistancer knows which ladder the last encode emitted after
   * the warm/cold arguments were applied. Returning them together stops a
   * consumer pairing a generation with a schedule that did not build it.
   */
  get globalFineFloodProvenanceSource(): Readonly<{
    source: WebGPUFineLevelSetBrickSource;
    encodedStrides: readonly number[];
  }> | undefined {
    const source = this.globalFineLevelSetSource;
    const redistance = this.globalFinePublishedIsA
      ? this.globalFineRedistanceA : this.globalFineRedistanceB;
    if (!source || !redistance || redistance.lastEncodedStrides.length === 0) return undefined;
    return Object.freeze({ source, encodedStrides: redistance.lastEncodedStrides });
  }
  /** Diagnostic-only status for the transport most recently encoded. */
  get globalFineTransportControl(): GPUBuffer | undefined { return this.lastGlobalFineTransport?.control; }
  /** Rejection-only raw producer deltas for both retained transport slots. */
  get globalFineTransportDeltaDebugPair() {
    if (!this.globalFineTransportA || !this.globalFineTransportB) return undefined;
    return {
      a: this.globalFineTransportA.topologyDelta.buffer,
      b: this.globalFineTransportB.topologyDelta.buffer,
      pageCapacity: this.globalFineTransportA.topologyDelta.pageCapacity,
      changedKeysOffsetWords: this.globalFineTransportA.topologyDelta.changedKeysOffsetWords,
    };
  }
  /** Rejection-only A/B page-table identity evidence. Payload channels are
   * shared; this only reveals which logical key names each physical page. */
  get globalFineSourceDebugPair() {
    if (!this.globalFineSourceA || !this.globalFineSourceB) return undefined;
    const source = (fine: WebGPUFineLevelSetBrickSource) => ({
      generation: fine.generation,
      plan: fine.plan,
      metadata: fine.metadata,
      worklist: fine.worklist,
      samples: fine.samples,
      workA: fine.workA,
      rollbackSamples: fine.rollbackSamples,
      pageCapacity: fine.plan.maximumResidentBricks,
      samplesPerBrick: fine.plan.samplesPerBrick,
      brickResolution: fine.plan.brickResolution,
    });
    return {
      a: source(this.globalFineSourceA),
      b: source(this.globalFineSourceB),
      publishedIsA: this.globalFinePublishedIsA,
    };
  }
  /** Diagnostic-only status for the redistance transaction that produced the current fine slot. */
  get globalFineRedistanceControl(): GPUBuffer | undefined {
    return this.globalFinePublishedIsA ? this.globalFineRedistanceA?.control : this.globalFineRedistanceB?.control;
  }
  /** Rejection-only visibility into the exact dirty/support streams consumed
   * by redistance. This is not an authority selector and is never read during
   * the simulation schedule. */
  get globalFinePageDeltaDebug() {
    const topology = this.globalFinePublishedIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    if (!topology) return undefined;
    return {
      buffer: topology.pageDelta,
      params: topology.debugParameterBuffer,
      sparseCandidates: topology.debugSparseCandidateBuffer,
      sparseCandidateCapacity: topology.sparseCandidateCapacity,
      pageCapacity: topology.current.plan.maximumResidentBricks,
      changedKeysOffsetWords: topology.pageDeltaLayout.changedKeysOffsetWords,
      dirtyPagesOffsetWords: topology.pageDeltaLayout.dirtyPagesOffsetWords,
      supportPagesOffsetWords: topology.pageDeltaLayout.supportPagesOffsetWords,
      promotionCountsOffsetWords: topology.pageDeltaLayout.promotionCountsOffsetWords,
    };
  }
  /** Rejection-only parity evidence for the two immutable fine publications. */
  get globalFinePageDeltaDebugPair() {
    if (!this.globalFineTopologyAB || !this.globalFineTopologyBA) return undefined;
    return {
      ab: this.globalFineTopologyAB.pageDelta,
      ba: this.globalFineTopologyBA.pageDelta,
      publishedIsA: this.globalFinePublishedIsA,
    };
  }
  /** Diagnostic-only shared total-volume transaction for both fine slots. */
  get globalFineVolumeControl(): GPUBuffer | undefined { return this.globalFineVolumeA?.control; }
  /** Diagnostic-only compact coarse-phi transaction control. */
  get globalFineCoarseLevelSetControl(): GPUBuffer | undefined { return this.lane.debug.globalFineCoarseLevelSetControl; }
  /** Diagnostic-only fine-to-coarse restriction transaction consumed by the
   * compact coarse-phi publication gate. */
  get globalFineRestrictionControl(): GPUBuffer | undefined { return this.lane.debug.globalFineRestrictionControl; }
  /** Diagnostic-only raw sparse summary header; topology consumes this GPU-side. */
  get globalFineSummaryDirectory(): GPUBuffer | undefined { return this.globalFineSummaries?.directory; }
  /** Admit the scalar generation proven by a coherent adaptive step receipt. */
  applyAdaptiveSurfaceGenerationReceipt(generation: number): void {
    if (!this.lane.acceptsAdaptiveSurfaceGenerationReceipt()
      || !Number.isSafeInteger(generation) || generation < 1 || generation > 0xffff_ffff) {
      throw new RangeError("Adaptive surface generation receipt is invalid for this projection");
    }
    this.adaptiveSurfaceGeneration = generation;
  }
  /** Rejection-only producer state for an unpublished fine summary. */
  get globalFineSummaryDebug() {
    const summaries = this.globalFineSummaries;
    const coarse = this.lane.summaryCoarseDebug();
    if (!summaries || !coarse) return undefined;
    return {
      ...summaries.diagnosticBuffers,
      coarseControl: coarse.control,
      coarseDelta: coarse.delta,
    };
  }
  get fineSeedAuthority() { return Boolean(this.fineSeedAdapter && this.globalFineLevelSet); }
  /**
   * Diagnostic-only census of the accepted structural owner topology.
   *
   * Every resident payload cell resolves to its accepted leaf identity. The
   * identity set deduplicates leaves that span several physical pages; missing
   * canonical-air pages are excluded because they contain no authored owner
   * topology.
   */
  async readTopologyLeafCensus(): Promise<OctreeTopologyLeafCensus> {
    const arena = this.ownerPages.arena;
    const readback = this.device.createBuffer({
      label: "Octree structural leaf census readback",
      size: arena.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read octree structural leaf census",
    });
    encoder.copyBufferToBuffer(arena, 0, readback, 0, arena.size);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return censusOctreeTopologyLeaves(
        words, this.ownerPages.plan, this.topologyMaximumLeafSize,
      );
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  /** QA-only readback of the actual adapter-to-global-fine publication chain.
   * These counters are observational and never participate in simulation
   * scheduling or authority selection. */
  async readGlobalFineLevelSetDiagnostics() {
    const fine = this.globalFinePublishedIsA ? this.globalFineSourceA : this.globalFineSourceB;
    const topology = this.globalFinePublishedIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    if ((!fine || !topology || !this.globalFineSeeds) && !this.coarseOnlySurfaceTracking) {
      return undefined;
    }
    const readback = this.device.createBuffer({ label: "Global fine structured QA diagnostics", size: 952,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read global fine structured QA diagnostics" });
    if (this.globalFineSeeds) {
      encoder.copyBufferToBuffer(this.globalFineSeeds.buffer, 0, readback, 0, 8);
    }
    if (topology) encoder.copyBufferToBuffer(topology.control, 0, readback, 8, 36);
    if (fine) encoder.copyBufferToBuffer(fine.worklist, 0, readback, 44, 20);
    // The lane's regions of this readback are disjoint from the engine's and
    // from each other, so one block of copies holds exactly what interleaving
    // them would have.
    this.lane.encodeGlobalFineDiagnosticCopies(encoder, readback);
    if (this.fineSeedAdapter) {
      encoder.copyBufferToBuffer(this.fineSeedAdapter.source.candidateCount, 0, readback, 248, 36);
      encoder.copyBufferToBuffer(this.fineSeedAdapter.source.leaves, 0, readback, 284, 64);
    }
    encoder.copyBufferToBuffer(this.compaction, 0, readback, 396, 16);
    if (this.globalFineVolumeA) {
      encoder.copyBufferToBuffer(this.globalFineVolumeA.control, 0, readback, 512, 64);
    }
    if (topology) {
      encoder.copyBufferToBuffer(topology.pageDelta, 0, readback, 608, 64);
      encoder.copyBufferToBuffer(topology.control, 48, readback, 764, 16);
    }
    // Transport governor schedule (state[0..7]) and sleep forensics
    // (state[46..56]: first-schedule latch, repairs, sleeping bit, and the
    // why-not-sleeping bitmask + measured displacement the prepare kernel
    // writes unconditionally) for both banks — the uncommitted-delta root of
    // a recurring-band rejection names its blocking term here.
    if (this.globalFineTransportA) {
      encoder.copyBufferToBuffer(this.globalFineTransportA.governor, 0, readback, 792, 32);
      encoder.copyBufferToBuffer(this.globalFineTransportA.governor, 46 * 4, readback, 824, 44);
    }
    if (this.globalFineTransportB) {
      encoder.copyBufferToBuffer(this.globalFineTransportB.governor, 0, readback, 868, 32);
      encoder.copyBufferToBuffer(this.globalFineTransportB.governor, 46 * 4, readback, 900, 44);
    }
    this.device.queue.submit([encoder.finish()]);
    let copiedWords: number[];
    try {
      await readback.mapAsync(GPUMapMode.READ);
      copiedWords = Array.from(new Uint32Array(readback.getMappedRange()));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    const words = copiedWords;
    const diagnostics = { seedControl: Array.from(words.slice(0, 2)),
        topologyControl: Array.from(words.slice(2, 11)),
        worklistHeader: Array.from(words.slice(11, 16)),
        coarseControl: Array.from(words.slice(16, 32)),
        fineRestrictionControl: Array.from(words.slice(32, 40)),
        structuredVelocityControl: Array.from(words.slice(40, 46)),
        structuredBoundaryControl: Array.from(words.slice(46, 62)),
        fineSeedAdapterControl: Array.from(words.slice(62, 71)),
        firstFineSeedLeaf: Array.from(words.slice(71, 87)),
        firstStructuredVelocityA: Array.from(words.slice(87, 91)),
        firstStructuredVelocityB: Array.from(words.slice(91, 95)),
        firstCoarsePhi: Array.from(words.slice(95, 99)),
        compactRowPrefix: Array.from(words.slice(99, 103)),
        firstStructuredApertureAB: Array.from(words.slice(103, 105)),
        firstStructuredSolidNormalVelocityAB: Array.from(words.slice(105, 107)),
        airSupportControl: Array.from(words.slice(107, 123)),
        precedingAirSupportTerminal: Array.from(words.slice(123, 126)),
        firstAirSupportFailure: Array.from(words.slice(126, 128)),
        fineVolumeControl: Array.from(words.slice(128, 144)),
        airSupportTerminalScratch: Array.from(words.slice(144, 152)),
        finePageDeltaHeader: Array.from(words.slice(152, 168)),
        structuredRejectCarry: Array.from(words.slice(168, 179)),
        airSupportFallbacks: Array.from(words.slice(180, 182)),
        airSupportTopologyFailureLatch: Array.from(words.slice(182, 191)),
        airSupportFailureCounts: Array.from(words.slice(195, 198)),
        fineTopologyFailureLatch: Array.from(words.slice(191, 195)),
        fineTransportScheduleA: Array.from(words.slice(198, 206)),
        fineTransportSleepA: Array.from(words.slice(206, 217)),
        fineTransportScheduleB: Array.from(words.slice(217, 225)),
        fineTransportSleepB: Array.from(words.slice(225, 236)),
        configuredFineGeneration: fine?.generation ?? 0, fineGenerationSlot: fine?.generationSlot ?? 0,
        scheduledFineGeneration: this.globalFineGeneration, currentFineIsA: this.globalFinePublishedIsA };
    const liveFirstError = diagnostics.airSupportControl[1] ?? 0xffff_ffff;
    const precedingFirstError = diagnostics.firstAirSupportFailure[1] ?? 0xffff_ffff;
    // Once a failed support publication poisons the next structured candidate,
    // the current transaction exits at stage zero. The producer deliberately
    // keeps the originating topology word in recordArena[14], so prefer the
    // live word only when it still names stage 6 and otherwise inspect that
    // one-shot preceding latch.
    const topologyFirstError = (liveFirstError >>> 24) === 6
      ? liveFirstError : (precedingFirstError >>> 24) === 6 ? precedingFirstError : liveFirstError;
    const airSupportFailureTopology = await this.lane.forensics.airSupportFailureTopology?.(
      topologyFirstError, diagnostics.airSupportTopologyFailureLatch);
    return airSupportFailureTopology ? { ...diagnostics, airSupportFailureTopology } : diagnostics;
  }
  /** Post-submit diagnostic census; never participates in pressure scheduling. */
  readSPGridHierarchyCensus() {
    return this.lane.forensics.spgridHierarchyCensus?.();
  }
  /** Terminal-only proof that the compact directory differential executed. */
  readSPGridTouchedDirectoryTripwire() {
    return this.lane.forensics.spgridTouchedDirectoryTripwire?.();
  }
  /** Post-submit Bet-4 machinery census from the shipping persistent solve. */
  readPowerHybridCensus() {
    return Promise.resolve(null);
  }
  /**
   * Post-submit class-0 census of the Section 4.3 band shell. `null` unless
   * `FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS` selected a mode — it is the only
   * emission that authors those words, so it cannot report a stale arena.
   */
  readPersistentBandCensus() {
    return Promise.resolve(null);
  }
  /** The retired persistent arena owned the optional class-level D4 snapshot.
   * Scene field symmetry remains covered by the authoritative smoke oracle. */
  async readPowerHybridClassSymmetry() { return undefined; }

  get fluidBrickCapacity() { return this.topologyResidency.capacity; }
  get fluidBrickResidencyWorklist() { return this.topologyResidency.worklist; }
  get fluidBulkBrickCapacity() { return this.sparseBrickWorld?.bulkResidency?.capacity ?? 0; }
  get fluidBulkBrickResidencyWorklist() {
    return this.sparseBrickWorld?.bulkResidencyWorklist;
  }
  readFluidBrickResidencyStats() { return this.topologyResidency.readStats(); }
  readFluidBulkBrickResidencyStats() { return this.sparseBrickWorld?.readBulkResidencyStats(); }
  encodeSparseBrickWorld(encoder: GPUCommandEncoder, _dt_s = 0) {
    void _dt_s;
    // Either the shared fine band or the lane's own coarse authority may be
    // the settled surface; a configuration with neither has nothing to name.
    if ((!this.globalFineBootstrapped && !this.lane.hasCoarseSurfaceAuthority())
      || !this.fineSeedAdapter) {
      throw new Error("Sparse render publication requires a settled surface authority and compact seeds");
    }
    const source=this.fineSeedAdapter.source;
    const bulkResidency = this.sparseBrickWorld?.bulkResidency;
    if (bulkResidency && bulkResidency !== this.topologyResidency) {
      bulkResidency.encodeFineSeedCandidates(
        encoder, source.leaves, source.candidates.candidates, source.candidates.countAndDispatch,
      );
    }
    // Scene maintenance remains live even when this physics checkpoint has no
    // dense fluid payload to publish. The presentation loop calls the same
    // entry point while paused; this bootstrap call merely starts convergence
    // for the initial scene revision.
    this.sparseBrickWorld?.encodeSceneMaintenance(encoder);
    // Publication is GPU-transactional. Failed, stale, and overflowing
    // generations retain the last good (including analytic t=0) tile stream;
    // a published zero-count generation is the distinct valid-empty case.
  }

  destroy() {
    this.powerLifecycleDisposed = true;
    // The lane is built by an initialization task rather than the constructor,
    // so a failure during initialization reaches this cleanup with parts of it
    // still unassigned. It releases only what it holds, so cleanup never throws
    // a TypeError that would replace the failure the caller is about to rethrow.
    this.lane.destroy();
    this.ownerPages.destroy();
    this.pressureA.destroy(); this.pressureB.destroy(); this.params.destroy();
    this.topologyCandidateDispatch.destroy();
    this.coldDispatch.destroy();
    this.compaction.destroy(); this.leafHeaders.destroy(); this.candidateLeafHeaders.destroy();
    this.candidatePressure.destroy();
    this.frontierSortScratch.destroy(); this.frontierSortStageParams?.destroy();
    this.leafFrontier.destroy();
    this.solveDispatch.destroy(); this.solidCells.destroy(); this.solveStats.destroy();
    this.unpublishedFineSummaryDirectory.destroy();
    this.globalFineRedistanceA?.destroy(); this.globalFineRedistanceB?.destroy();
    this.analyticBootstrapWorklist?.destroy();
    this.globalFineVolumeA?.destroy(); this.globalFineVolumeB?.destroy();
    this.globalFineRigidCarveA?.destroy(); this.globalFineRigidCarveB?.destroy();
    this.globalFineTransportA?.destroy(); this.globalFineTransportB?.destroy();
    this.globalFineTopologyAB?.destroy(); this.globalFineTopologyBA?.destroy();
    this.globalFineSeeds?.destroy(); this.globalFineLevelSet?.destroy();
    this.globalFineSummaries?.destroy();
    this.coarseOnlySummary?.destroy();
    this.fineSeedAdapter?.destroy();
    this.topologyDiagnosticTexture?.destroy(); this.pressureSamplesDiagnosticTexture?.destroy(); this.pressureDiagnosticTexture?.destroy();
    this.surfaceState.destroy();
    if (this.sparseBrickWorld) this.sparseBrickWorld.destroy(); else this.topologyResidency.destroy();
  }
}


export function initialOctreeLevelSet(
  scene: SceneDescription,
  dims: { nx: number; ny: number; nz: number },
  cell: { x: number; y: number; z: number }
) {
  const { nx, ny, nz } = dims;
  // Explicit brick seeds are a union of exact axis-aligned boxes. Preserve
  // that analytic signed distance at cell centres instead of rebuilding it
  // from binary occupancy, whose Euclidean transform rounds the very corners
  // used by the symmetry oracle before the first GPU command is submitted.
  if ((scene.fluid.initialBrickSeeds_m?.length ?? 0) > 0 && !sceneHasTerrain(scene)
    && !scene.fluid.initialBrickSeedsAdditive && !sceneHasInitialLiquidVolumes(scene)) {
    const phi = new Float32Array(nx * ny * nz);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      phi[x + nx * (y + ny * z)] = initialFluidBrickSignedDistanceAtCell(
        scene, x, y, z, [nx, ny, nz])!;
    }
    return phi;
  }
  const alpha = new Float32Array(nx * ny * nz);
  const dam = sceneDamBreakBox(scene);
  const terrainHeights = terrainColumnHeights(scene, nx, nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const aboveGround = (y + 0.5) * cell.y > terrainHeights[x + nx * z];
    const wet = aboveGround && initialLiquidContainsCell(scene, x, y, z, [nx, ny, nz],
      scene.fluid.initialCondition === "dam-break"
        ? damBreakBoxContains(dam, (x + 0.5) / nx, (y + 0.5) / ny, (z + 0.5) / nz)
        : (y + 0.5) / ny <= scene.container.fillFraction);
    alpha[x + nx * (y + ny * z)] = wet ? 1 : 0;
  }
  return signedDistanceFromVolume(alpha, nx, ny, nz, cell);
}

/** One-time node-lattice bootstrap for an exact, non-additive brick union.
 *
 * Returns undefined for fields that do not have this analytic representation;
 * those retain the existing cell-centred bootstrap.  The `(n + 1)^3` array is
 * construction-only and is released after the first adaptive graph commit.
 * No recurring physics or renderer work is dense-domain sized. */
export function initialOctreeNodalLevelSet(
  scene: SceneDescription,
  dims: { nx: number; ny: number; nz: number },
): Float32Array | undefined {
  const seeded = (scene.fluid.initialBrickSeeds_m?.length ?? 0) > 0;
  const proceduralDam = !seeded && scene.fluid.initialCondition === "dam-break";
  // A ball unions with the base condition, so neither branch below is the whole
  // field once one is authored. Declining leaves the cell-centred rasterization,
  // which is the same answer terrain and additive seeds already get.
  if ((!seeded && !proceduralDam) || sceneHasTerrain(scene)
      || scene.fluid.initialBrickSeedsAdditive || sceneHasInitialLiquidVolumes(scene)) return undefined;
  const { nx, ny, nz } = dims;
  const values = new Float32Array((nx + 1) * (ny + 1) * (nz + 1));
  for (let z = 0; z <= nz; z += 1) {
    for (let y = 0; y <= ny; y += 1) {
      for (let x = 0; x <= nx; x += 1) {
        values[x + (nx + 1) * (y + (ny + 1) * z)] = seeded
          ? initialFluidBrickSignedDistanceAtNode(scene, x, y, z, [nx, ny, nz])!
          : damBreakSignedDistanceAtNode(scene, x, y, z, [nx, ny, nz])!;
      }
    }
  }
  return values;
}

function projectionWGSLClosingDelimiter(
  source: string,
  openIndex: number,
  openCharacter: string,
  closeCharacter: string,
): number {
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === openCharacter) depth += 1;
    else if (source[index] === closeCharacter && --depth === 0) return index;
  }
  throw new Error(`Projection activity WGSL has an unterminated ${openCharacter}${closeCharacter} region`);
}

/** One bounded workgroup progress sample per projection dispatch. Disabled
 * mode returns the production shader byte-for-byte. */
export function octreeProjectionActivityShader(
  activity: GPULogicalActivityAdoptionContext,
  shaderSource: string,
): string {
  if (!activity.enabled) return shaderSource;
  const edits: Array<{ start: number; end: number; replacement: string }> = [];
  for (const entryPoint of OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS) {
    const declaration = new RegExp(
      `@compute\\s+@workgroup_size\\s*\\(([^)]*)\\)\\s*fn\\s+${entryPoint}\\s*\\(`,
    ).exec(shaderSource);
    if (!declaration) throw new Error(`Projection activity entry point is missing: ${entryPoint}`);
    const functionIndex = shaderSource.indexOf("fn", declaration.index);
    const parametersOpen = shaderSource.indexOf("(", functionIndex);
    const parametersClose = projectionWGSLClosingDelimiter(
      shaderSource, parametersOpen, "(", ")",
    );
    const bodyOpen = shaderSource.indexOf("{", parametersClose);
    if (bodyOpen < 0) throw new Error(`Projection activity entry point has no body: ${entryPoint}`);
    const parameters = shaderSource.slice(parametersOpen + 1, parametersClose);
    const builtinName = (builtin: string) => new RegExp(
      `@builtin\\(${builtin}\\)\\s*([A-Za-z_]\\w*)\\s*:`,
    ).exec(parameters)?.[1];
    const workgroupId = builtinName("workgroup_id") ?? "activityWorkgroupId";
    const localInvocationIndex = builtinName("local_invocation_index")
      ?? "activityLocalInvocationIndex";
    const numWorkgroups = builtinName("num_workgroups") ?? "activityNumWorkgroups";
    const additions = [
      ...(builtinName("workgroup_id") ? []
        : ["@builtin(workgroup_id) activityWorkgroupId:vec3u"]),
      ...(builtinName("local_invocation_index") ? []
        : ["@builtin(local_invocation_index) activityLocalInvocationIndex:u32"]),
      ...(builtinName("num_workgroups") ? []
        : ["@builtin(num_workgroups) activityNumWorkgroups:vec3u"]),
    ];
    if (additions.length > 0) {
      const trimmedParameters = parameters.trim();
      const separator = trimmedParameters.length === 0 || trimmedParameters.endsWith(",") ? "" : ",";
      edits.push({
        start: parametersOpen + 1,
        end: parametersClose,
        replacement: `${parameters}${separator}${additions.join(",")}`,
      });
    }
    const workgroupDimensions = declaration[1]!.split(",")
      .map((value) => Number(value.trim().replace(/u$/, "")));
    if (workgroupDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
      throw new Error(`Projection activity workgroup size is not literal: ${entryPoint}`);
    }
    const workgroupLaneCount = workgroupDimensions.reduce((product, value) => product * value, 1);
    const descriptor = OCTREE_PROJECTION_ACTIVITY_TASKS[entryPoint];
    const progress = activity.workgroup(descriptor.task, "progress", {
      workgroupId,
      numWorkgroups,
      localInvocationIndex,
      workgroupLaneCount,
    });
    edits.push({ start: bodyOpen + 1, end: bodyOpen + 1, replacement: progress });
  }
  return edits.sort((left, right) => right.start - left.start).reduce(
    (source, edit) => source.slice(0, edit.start) + edit.replacement + source.slice(edit.end),
    shaderSource,
  );
}

/** GPU-only adapter from packed owner authority to on-demand scientific overlays. */
export const octreeDiagnosticShader = /* wgsl */ `
override rowIndexedPressure: bool = true;
struct Owner { packedOrigin: u32, size: u32 }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container:vec4f, inflowPositionRadius:vec4f, inflowDirectionLength:vec4f, physical:vec4f, pressureCapacity:vec4u }
@group(0) @binding(0) var<storage, read> owners: array<u32>;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(4) var topologyOut: texture_storage_3d<rg32uint, write>;
@group(0) @binding(5) var pressureSamplesOut: texture_storage_3d<rgba32uint, write>;
@group(0) @binding(6) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(8) var<uniform> params: Params;
@group(0) @binding(11) var<storage, read> frontier: array<u32>;
fn dims() -> vec3u { return params.dimsMax.xyz; }
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn packOrigin(p: vec3u) -> u32 { return index(p); }
fn unpackOrigin(word: u32) -> vec3u { let plane=params.dimsMax.x*params.dimsMax.y;return vec3u(word%params.dimsMax.x,(word/params.dimsMax.x)%params.dimsMax.y,word/plane); }
fn invalidOwner()->Owner{return Owner(0u,0u);}
fn ownerValid(owner:Owner)->bool{if(owner.size==0u||owner.size>params.dimsMax.w
  ||(owner.size&(owner.size-1u))!=0u){return false;}let origin=unpackOrigin(owner.packedOrigin);
  return all(origin+vec3u(owner.size)<=dims());}
fn decodePagedOwner(word:u32,cell:vec3u)->Owner{
  if((word&0x80000000u)==0u){return invalidOwner();}let exponent=(word>>18u)&7u;
  if(exponent>5u){return invalidOwner();}let brickOrigin=vec3i((cell/vec3u(8u))*vec3u(8u));
  let delta=vec3i(i32(word&63u)-32,i32((word>>6u)&63u)-32,i32((word>>12u)&63u)-32);
  let signedOrigin=brickOrigin+delta;if(any(signedOrigin<vec3i(0))){return invalidOwner();}
  let origin=vec3u(signedOrigin);let size=1u<<exponent;
  if(any(cell<origin)||any(cell>=origin+vec3u(size))||any(origin+vec3u(size)>dims())){return invalidOwner();}
  return Owner(packOrigin(origin),size);
}
fn ownerPageEncoded(logical:u32)->u32{let pageOffset=owners[5];let capacity=owners[3];let logicalCount=owners[4];let table=owners[10]>>31u;let directoryOffset=pageOffset+3u*capacity+table*logicalCount;if(pageOffset!=16u+capacity||logical>=logicalCount||owners[6]!=pageOffset+3u*capacity+2u*logicalCount){return 0u;}return owners[directoryOffset+logical];}
fn ownerAt(cell: vec3u) -> Owner {
  let bd=(dims()+vec3u(7u))/8u;let b=cell/8u;let logical=b.x+b.y*bd.x+b.z*bd.x*bd.y;let encoded=ownerPageEncoded(logical);let capacity=owners[3];
  if(encoded==0u||encoded==0xffffffffu||encoded>capacity){return invalidOwner();}let local=cell%vec3u(8u);let table=owners[10]>>31u;let payload=owners[6]+table*capacity*512u;let word=owners[payload+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u];
  if(word==0u){return invalidOwner();}return decodePagedOwner(word,cell);
}
fn frontierBase(which:u32)->u32{return 10u+which*params.pressureCapacity.x;}
fn mortonPart10(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn rowMorton(cell:u32)->u32{let p=unpackOrigin(cell);return mortonPart10(p.x)|(mortonPart10(p.y)<<1u)|(mortonPart10(p.z)<<2u);}
fn pressureRow(owner: Owner) -> u32 {
  if(!ownerValid(owner)){return 0xffffffffu;}
  let cell = index(unpackOrigin(owner.packedOrigin));
  if (!rowIndexedPressure) { return cell; }
  let current=frontier[2];let count=min(frontier[current],params.pressureCapacity.x);
  let level=u32(firstTrailingBit(owner.size));let morton=rowMorton(cell);var lo=0u;var hi=count;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontier[frontierBase(current)+mid];
    let otherOwner=ownerAt(unpackOrigin(other));if(!ownerValid(otherOwner)){return 0xffffffffu;}
    let otherLevel=u32(firstTrailingBit(otherOwner.size));
    let otherMorton=rowMorton(other);if(otherLevel<level||(otherLevel==level&&(otherMorton<morton
      ||(otherMorton==morton&&other<cell)))){lo=mid+1u;}else{hi=mid;}}
  return select(0xffffffffu,lo,lo<count&&frontier[frontierBase(current)+lo]==cell);
}
@compute @workgroup_size(4,4,4)
fn materializeOctreeFields(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(gid);
  if(!ownerValid(owner)){
    textureStore(topologyOut,vec3i(gid),vec4u(0xffffffffu));
    textureStore(pressureSamplesOut,vec3i(gid),vec4u(0xffffffffu));
    textureStore(pressureOut,vec3i(gid),vec4f(0.0));return;
  }
  let origin = unpackOrigin(owner.packedOrigin);
  let horizontal = origin.x | (origin.z << 10u) | (owner.size << 20u);
  let vertical = origin.y | ((origin.y + owner.size) << 10u);
  textureStore(topologyOut, vec3i(gid), vec4u(horizontal, vertical, 0u, 0u));
  let invalid = 0xffffffffu; let row = pressureRow(owner); let wet = row != invalid;
  let q = vec3i(gid);
  // Pressure ownership remains useful to generic scientific slices. Compact
  // velocity and Projection Δu are rendered by the native technique overlay.
  textureStore(pressureSamplesOut, q, select(vec4u(invalid), vec4u(row, 0u, vertical, horizontal), wet));
  var centrePressure = 0.0;
  if (row < arrayLength(&pressure)) { centrePressure = pressure[row]; }
  textureStore(pressureOut, vec3i(gid), vec4f(select(0.0, centrePressure, wet)));
}
`;
