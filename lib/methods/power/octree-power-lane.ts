import type { GPUInitializationTask } from "../../core/gpu-initialization";
import type { WebGPUFineLevelSetBrickSource } from "../../core/levelset-consumer-abi";
import { PassBroker } from "../../core/webgpu-pass-broker";
import { sceneHasTerrain } from "../../core/terrain";
import type { SurfaceInflowState } from "../octree-shared/surface-state";
import type {
  OctreeAdvanceContext,
  OctreeCoarseDynamicsLane,
  OctreeFineLevelSetTransportStage,
  OctreeFineTransportRequest,
  OctreeLaneCoarseLevelSetPublication,
  OctreeLaneDebugSources,
  OctreeLaneFrontierFailureCapture,
  OctreeLaneFrontierFailureReceipt,
  OctreeLaneSymmetryStageAuditBuffers,
  OctreeLaneTechniqueDebugSources,
  OctreeTopologyEngine,
} from "../octree-shared/octree-coarse-dynamics-lane";
import { octreeEffectiveLeafSize } from "../octree-shared/octree-leaf-sizing";
import { POWER2017_FINE_BAND_SURFACE_GROWTH_SAFETY } from "../octree-shared/octree-fine-band-capacity";
import { initialOctreeLevelSet } from "../octree-shared/webgpu-octree";
import { WebGPUOctreeCoarseSummary } from "../octree-shared/webgpu-octree-coarse-summary";
import {
  maximumFineLevelSetJFAStride,
  WebGPUFineLevelSetRedistance,
} from "../octree-shared/webgpu-octree-fine-levelset-redistance";
import { WebGPUFineLevelSetVolumeCorrection } from "../octree-shared/webgpu-octree-fine-levelset-volume";
import {
  planFineLevelSetBandFineCells,
  WebGPUFineLevelSetTopology,
} from "../octree-shared/webgpu-octree-fine-levelset-topology";
import { sumOctreePowerAllocationBreakdown } from "../octree-shared/octree-arena-allocation";
import {
  createGPULogicalActivityAdoptionContext,
  type GPULogicalActivityAdoptionContext,
} from "../../core/gpu-logical-activity-adoption";
import { performanceShaderVariant } from "../../core/stores/performance-instrumentation-store";
import type { OctreeRuntimeDials } from "../octree-shared/octree-runtime-dials";
import {
  OCTREE_FINE_SEMANTIC_PHASES,
  type OctreeEnginePhase,
  type OctreeFineSemanticPhase,
  type OctreeSemanticBoundary,
} from "../octree-shared/octree-projection-contract";
import {
  octreeFineEngineSplitsEnabled,
  octreeTopologyTileClampEnabled,
} from "../octree-shared/octree-projection-gates";
import type { OctreeOwnerLeafSize } from "../octree-shared/webgpu-octree-owner-pages";
import {
  lookupOctreeOwnerPage,
  OCTREE_OWNER_PAGE_LOOKUP_STATUS,
} from "../octree-shared/webgpu-octree-owner-pages";
import { WebGPUOctreeCoarseLevelSet } from "../octree-shared/webgpu-octree-coarse-levelset";
import { WebGPUFineToCoarseLevelSet } from "../octree-shared/webgpu-octree-fine-to-coarse-levelset";
import { WebGPUFineLevelSetTransport } from "../octree-shared/webgpu-octree-fine-levelset-transport";
import {
  normalizeOctreeSection43BoundarySmoothing,
  type OctreeFirstOrderSPDVCycle,
} from "../octree-shared/webgpu-octree-section43-contract";
import {
  WebGPUOctreePipelinedMGPCG,
  type OctreePipelinedMGPCGVectors,
  type OctreePipelinedWorksetLinearOperator,
} from "../octree-shared/webgpu-octree-pipelined-mgpcg";
import { WebGPUOctreeSolidVertexSdf } from "../octree-shared/webgpu-octree-solid-vertex-sdf";
import { WebGPUOctreeTopologyEpoch } from "../octree-shared/webgpu-octree-topology-epoch";
import {
  decodeGeneratedOctreePowerCatalog,
  fetchGeneratedOctreePowerCatalog,
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
  type GeneratedOctreePowerCatalogViews,
} from "./generated/octree-power-catalog";
import { OCTREE_POWER_NEIGHBOR_DIRECTIONS } from "./octree-power-descriptor";
import { octreePowerProjectionShader, POWER_PROJECTION_LANE } from "./octree-power-projection.wgsl";
import { planOctreeAirVelocitySupport } from "./webgpu-octree-air-velocity-support";
import {
  octreeAirSupportFootprintCapacity,
  WebGPUOctreeAirVelocitySupportProducer,
} from "./webgpu-octree-air-velocity-support-gpu";
import { WebGPUOctreePowerCoarseLevelSet } from "./webgpu-octree-power-coarse-levelset";
import { WebGPUOctreePowerDescriptor } from "./webgpu-octree-power-descriptor";
import { WebGPUOctreePowerTopology } from "./webgpu-octree-power-topology";
import { WebGPUOctreeSection43HybridPreconditioner } from "./webgpu-octree-section43-preconditioner";
import {
  WebGPUOctreeSPGridVCycle,
  type OctreeSPGridAccurateAuthority,
} from "./webgpu-octree-spgrid-vcycle";
import { WebGPUStructuredBoundaryCoefficients } from "./webgpu-octree-structured-boundary";
import { WebGPUStructuredVelocityDynamics } from "./webgpu-octree-structured-dynamics";
import { WebGPUDirectStructuredVelocityAuthority } from "./webgpu-octree-structured-velocity-gpu";

/** Structured world-boundary bits are x-/x+/y-/y+/z-/z+. */
function structuredClosedBoundaryMask(closedTop: boolean): number {
  return closedTop ? 0b11_1111 : 0b11_0111;
}

/**
 * The generated catalog is a 14 MB device-independent constant, so fetching,
 * decoding and re-viewing it once per solver build is pure waste — an editor
 * session rebuilds many times per minute. Memoizing the in-flight promise also
 * collapses concurrent builds onto one decode.
 *
 * Safe because the views are read-only inputs: nothing mutates them, and the
 * asset is fixed for the lifetime of the module (it is checked in and
 * version-guarded by `verify:octree-power-catalog`).
 */
let generatedOctreePowerCatalog: Promise<GeneratedOctreePowerCatalogViews> | undefined;

async function readGeneratedOctreePowerCatalog(): Promise<GeneratedOctreePowerCatalogViews> {
  const url = new URL("./generated/octree-power-catalog.bin", import.meta.url);
  if (url.protocol !== "file:") return fetchGeneratedOctreePowerCatalog(url);
  // Node's fetch deliberately rejects file: URLs. Keep the browser asset path
  // unchanged while letting the production-equivalent Dawn harness initialize
  // the same checked-in binary instead of silently exercising rollback only.
  // The indirection through a variable is what keeps the browser bundle free of
  // a node builtin; `@vite-ignore` is telling the bundler that is deliberate
  // rather than a specifier it failed to understand.
  const nodeFs = "node:fs/promises";
  const { readFile } = await import(/* @vite-ignore */ nodeFs) as { readFile(path: URL): Promise<Uint8Array> };
  const bytes = await readFile(url);
  return decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
}

function loadGeneratedOctreePowerCatalog(): Promise<GeneratedOctreePowerCatalogViews> {
  // A failed load must not be cached, or one transient error poisons the
  // session; clear the memo so the next build retries.
  return generatedOctreePowerCatalog ??= readGeneratedOctreePowerCatalog()
    .catch((error: unknown) => { generatedOctreePowerCatalog = undefined; throw error; });
}

type OctreeFirstOrderVCycleImplementation = OctreeFirstOrderSPDVCycle & {
  readonly plan: { readonly levelCount: number };
  readonly accurateOperator: OctreePipelinedWorksetLinearOperator;
  configureAccurateAuthority(authority: OctreeSPGridAccurateAuthority): void;
  readPublishedHierarchyForDiagnostics(): ReturnType<
    WebGPUOctreeSPGridVCycle["readPublishedHierarchyForDiagnostics"]
  >;
  readHierarchyCensus(): Promise<Readonly<{ levels: readonly Readonly<Record<string, number>>[] }>>;
  readTouchedDirectoryTripwireDiagnostics(): ReturnType<
    WebGPUOctreeSPGridVCycle["readTouchedDirectoryTripwireDiagnostics"]
  >;
  readCandidateFailureDiagnostics(): Promise<Readonly<{
    levelDelta: readonly number[]; candidateDispatch: readonly number[];
  }>>;
  initializePipelines(onProgress?: (label: string, completed: number, total: number) => void): Promise<void>;
  encodeCapture(broker: PassBroker): void;
  readonly candidateControl: GPUBuffer;
  /** Copy source for the step-snapshot ring (A4). Never bound by this file. */
  readonly levelDelta: GPUBuffer;
  // `dispatch` is the per-level dispatch metadata the Section 4.3 shell reads as
  // `spgridDispatch` (binding 26) for pCount/pPageCount/pTransferCount. The
  // implementation already returns it; omitting it here narrowed it away from
  // the spread below, so the preconditioner's source was missing a member it
  // declares as required.
  readonly section63Topology: Readonly<{
    topology: GPUBuffer; state: GPUBuffer; geometry: GPUBuffer; layout: GPUBuffer;
    dispatch: GPUBuffer;
  }>;
  encodeCandidateSetup(broker: PassBroker, input: { solverControl: GPUBuffer; rowCount: GPUBuffer;
    sourceControl: GPUBuffer; topologyMetrics: GPUBuffer }): void;
  encodeReadySetupCommit(broker: PassBroker, input: { solverControl: GPUBuffer;
    rowCount: GPUBuffer }): void;
  destroy(): void;
};

/**
 * Aanjaneya et al. 2017 power-diagram coarse dynamics, as an octree lane.
 *
 * Everything here was an inline `coarseDynamics.backend === "power2017"` test
 * inside the shared topology engine. The power descriptor, its Delaunay
 * catalog, the structured velocity/boundary authorities, the Section 5 air
 * support producer and the SPGrid V-cycle are this lane's private machinery:
 * no engine code may name them, which is what keeps the shared topology,
 * frontier and fine-band ladder honestly shared.
 */
export class OctreePowerCoarseDynamicsLane implements OctreeCoarseDynamicsLane {
  readonly backend = "power2017" as const;

  readonly wgsl = {
    projectionShader: octreePowerProjectionShader,
    fragments: POWER_PROJECTION_LANE,
  } as const;

  constructor(private readonly engine: OctreeTopologyEngine) {}

  private powerCoarseLevelSet?: WebGPUOctreeCoarseLevelSet;
  private powerCoarseLevelSetSchedule?: WebGPUOctreePowerCoarseLevelSet;
  private fineToPowerCoarseLevelSet?: WebGPUFineToCoarseLevelSet;
  private powerCoarseLevelSetBootstrapped = false;
  /** Generation of the currently scheduled coarse-octree phi publication.
   * It advances independently when the optional global fine band is off. */
  private powerCoarseLevelSetGeneration = 2;
  /** Row-parallel production pressure executor with exact integer reductions. */
  private pipelinedMGPCG?: WebGPUOctreePipelinedMGPCG;
  private section43HybridPreconditioner?: WebGPUOctreeSection43HybridPreconditioner;
  private pipelinedMGPCGVectors?: OctreePipelinedMGPCGVectors;
  private firstOrderVCycle!: OctreeFirstOrderVCycleImplementation;
  private structuredVelocity?: WebGPUDirectStructuredVelocityAuthority;
  private structuredBoundary?: WebGPUStructuredBoundaryCoefficients;
  private topologyEpoch?: WebGPUOctreeTopologyEpoch;
  private readyEpochAudit?: GPUBuffer;
  private readyFrontierAudit?: GPUBuffer;
  private readyCompactionAudit?: GPUBuffer;
  private structuredDynamics?: WebGPUStructuredVelocityDynamics;
  private airVelocitySupport?: WebGPUOctreeAirVelocitySupportProducer;
  private structuredDivergenceRhs?: GPUBuffer;
  private structuredSeparationMask?: GPUBuffer;
  private powerDescriptor?: WebGPUOctreePowerDescriptor;
  private powerTopology?: WebGPUOctreePowerTopology;
  private powerSolidVertices?: WebGPUOctreeSolidVertexSdf;
  private powerVolumes?: GPUBuffer;
  private powerVolumeParams?: GPUBuffer;
  private powerVolumePipeline?: GPUComputePipeline;
  private powerVolumeGroup?: GPUBindGroup;
  private initializePowerVolumePipeline?: () => Promise<void>;

  // --- Allocation-stage hooks ----------------------------------------------
  /**
   * Power grades by owner page, not by leaf size, so the authored maximum is
   * already the ceiling: no exact-tiling ladder constrains which sizes the
   * arena can address.
   */
  topologyLeafCeiling(
    maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    _dims: { nx: number; ny: number; nz: number },
    _exactTilingLeafSize: OctreeOwnerLeafSize,
  ): OctreeOwnerLeafSize {
    return maximumLeafSize;
  }

  /** The frozen Power ladder: the largest size that tiles these dimensions. */
  refinementLadderLeafSize(
    maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    dims: { nx: number; ny: number; nz: number },
    _topologyMaximumLeafSize: OctreeOwnerLeafSize,
  ): OctreeOwnerLeafSize {
    return octreeEffectiveLeafSize(maximumLeafSize, dims);
  }

  /**
   * Power's exclusive-Delaunay rule can renew an ordinary imbalance, so its
   * closure budget pays for both halves of the mixed-ring propagation.
   */
  readonly balanceRoundsUseExclusiveMixedRing = true;

  /**
   * The structured velocity extension is sized from the accepted row set
   * rather than from a compiled band width, so no authored interface band can
   * outrun it.
   */
  validateInterfaceBand(_interfaceRefinementBandCells: number): void {}

  /**
   * The residency tile is the refinement ladder's top rung (or the authored
   * maximum where the clamp gate is off). Power leaves cannot exceed it
   * because the ladder is what produced them.
   */
  topologyTileSize(
    maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    effectiveLeafSize: OctreeOwnerLeafSize,
    _exactTilingLeafSize: OctreeOwnerLeafSize,
    _topologyMaximumLeafSize: OctreeOwnerLeafSize,
  ): number {
    return Math.max(8, octreeTopologyTileClampEnabled() ? effectiveLeafSize : maximumLeafSize);
  }

  readonly fineBandSurfaceGrowthSafety = POWER2017_FINE_BAND_SURFACE_GROWTH_SAFETY;

  /**
   * The frozen Power allocation policy has no translated-envelope floor: its
   * cold publisher never dilates the complete authored box the way the
   * adaptive lane's does, so the sparse area estimate stands alone.
   */
  fineBandBrickFloor(_input: {
    readonly brickDimensions: readonly [number, number, number];
    readonly minimumBrick: readonly [number, number, number];
    readonly maximumBrick: readonly [number, number, number];
    readonly capacityDilationBrickRings: number;
  }): number {
    return 0;
  }

  /** No override: the shared solve tail policy owns Power's hard ceiling. */
  pressureHardIterationCeiling(): number | undefined { return undefined; }

  pressureSolverLabel(): string {
    const budget = this.pipelinedMGPCG?.iterationBudget ?? this.engine.info.pressureIterationBudget;
    const levels = this.firstOrderVCycle?.plan.levelCount ?? 0;
    return `Octree power MGPCG · row-parallel exact-reduction executor · Section 4.3 fixed schedule · up to ${budget} iterations · ${levels}-level L1 V-cycle`;
  }

  /**
   * Nothing to build here. The power descriptor needs the generated Delaunay
   * catalog, which is a 14 MB asynchronous asset; this lane therefore builds
   * its authority from an initialization task instead of the constructor.
   */
  constructAuthority(): void {}
  private initializeNativePowerAuthority(catalog: GeneratedOctreePowerCatalogViews): void {
    if (this.powerDescriptor || this.engine.powerLifecycleDisposed) return;
    const rowCapacity = this.engine.pressureCapacity.rowCapacity;
    this.powerDescriptor = new WebGPUOctreePowerDescriptor(this.engine.device, rowCapacity);
    this.powerTopology = new WebGPUOctreePowerTopology(this.engine.device, rowCapacity, catalog);
    const structured = new WebGPUDirectStructuredVelocityAuthority(this.engine.device, {
      leafHeaders: this.engine.leafHeaders,
      topology: this.powerTopology.source, rowDelta: this.engine.powerRowDelta,
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
      closedBoundaryMask: structuredClosedBoundaryMask(this.engine.scene.container.top === "closed"),
    });
    this.structuredVelocity = structured;
    const structuredSource = structured.source;
    const section63Source = structuredSource.section63;
    this.structuredDivergenceRhs = this.engine.device.createBuffer({
      label: "Structured divergence RHS SoA",
      size: rowCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const rowDelta = { rows: this.engine.leafFrontier,
      rowCapacity: this.engine.frontierAllocation.listCapacity,
      controlOffsetWords: this.engine.frontierAllocation.rowDeltaControlOffsetWords,
      newToOldOffsetWords: this.engine.frontierAllocation.rowDeltaNewToOldOffsetWords,
      oldToNewOffsetWords: this.engine.frontierAllocation.rowDeltaOldToNewOffsetWords,
      // SPGrid caches row-indexed pages. Insertions and retirements therefore
      // require its wider positional influence stream even though remapped
      // power descriptors can carry the same immutable identities exactly.
      dirtyRowsOffsetWords: this.engine.frontierAllocation.rowDeltaAffectedRowsOffsetWords,
      dirtyCountControlWord: 6 as const };
    this.firstOrderVCycle = new WebGPUOctreeSPGridVCycle(this.engine.device, {
      ...section63Source, rowGeometry: structuredSource.rowGeometry, rowDelta,
    }, { dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz], rowCapacity,
      finestCellWidth: this.engine.scene.container.width_m / this.engine.dims.nx,
      compileHierarchicalExecutor: true,
      deferPipelineCompilation: this.engine.deferPipelineCompilation,
    });
    if (this.firstOrderVCycle.smootherContract?.degree === undefined) {
      throw new Error("Wide MGPCG requires the published V-cycle smoother contract");
    }
    const spgrid = this.firstOrderVCycle.section63Topology;
    // The paper evolves coarse octree phi regardless of whether the optional
    // factor-1/factor-4/factor-8 interface band exists. It is also the complete
    // inside/outside and cell-centre boundary authority in coarse-only mode.
    this.powerCoarseLevelSet = new WebGPUOctreeCoarseLevelSet(this.engine.device, rowCapacity);
    const airSupportLayout = planOctreeAirVelocitySupport(
      rowCapacity, structured.plan.slotCapacity, this.engine.device.limits.minStorageBufferOffsetAlignment,
      this.engine.dims.nx * this.engine.dims.ny * this.engine.dims.nz,
      octreeAirSupportFootprintCapacity(rowCapacity,
        this.engine.dims.nx * this.engine.dims.ny * this.engine.dims.nz),
    );
    this.powerCoarseLevelSetSchedule = new WebGPUOctreePowerCoarseLevelSet(
      this.engine.device, this.powerCoarseLevelSet, this.powerTopology.source,
      structured.plan.slotCapacity * 16,
      airSupportLayout,
      this.engine.coarseOnlySurfaceTracking ? airSupportLayout.ownerDirectoryCellCapacity : 0,
    );
    if (this.engine.coarseOnlySurfaceTracking) {
      const coarseCell = {
        x: this.engine.scene.container.width_m / this.engine.dims.nx,
        y: this.engine.scene.container.height_m / this.engine.dims.ny,
        z: this.engine.scene.container.depth_m / this.engine.dims.nz,
      };
      this.engine.coarseOnlySummary = new WebGPUOctreeCoarseSummary(this.engine.device,
        this.powerCoarseLevelSetSchedule.sampleSource,
        [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz], {
          arena: this.powerCoarseLevelSetSchedule.selectorRows,
          layout: airSupportLayout,
          rowVelocities: structuredSource.rowVelocities,
          initialPhi: initialOctreeLevelSet(this.engine.scene, this.engine.dims, coarseCell),
          physicalCellSize: coarseCell.x,
          timestep_s: this.engine.scene.numerics.maxDt_s,
          maximumLeafSize: this.engine.maxLeafSize,
          ownerPages: { arena: this.engine.ownerPages.arena, plan: this.engine.ownerPages.plan },
          ...(this.engine.scene.rigidBodies.length > 0 ? { rigid: {
            rigidBodies: this.engine.resources.rigidBodies,
            immersedVolumes: this.engine.resources.rigidImmersedVolumes,
            bodyCount: this.engine.scene.rigidBodies.length,
          } } : {}),
        });
      this.engine.info.allocatedBytes += this.engine.coarseOnlySummary.plan.allocatedBytes;
      this.engine.workAccounting.setAuthorityBytes("coarse-summary",
        this.engine.coarseOnlySummary.plan.allocatedBytes);
    }
    if (this.engine.coarseOnlySurfaceTracking && !this.engine.coarseOnlySummary) {
      throw new Error("Factor-one allocation did not construct the dense coarse tracker");
    }
    const coarseDirectory = this.powerCoarseLevelSetSchedule.sampleSource.directory;
    // Binding 15 is the compact coarse-phi directory for the mandatory power
    // topology/pressure authority.
    this.engine.groups = {
      ab: this.engine.createProjectionGroup(this.engine.pressureA, this.engine.pressureB, coarseDirectory),
      ba: this.engine.createProjectionGroup(this.engine.pressureB, this.engine.pressureA, coarseDirectory),
    };
    this.engine.candidateRowGroups = {
      fromA: this.engine.createProjectionGroup(this.engine.pressureA, this.engine.candidatePressure, coarseDirectory,
        this.engine.candidateLeafHeaders),
      fromB: this.engine.createProjectionGroup(this.engine.pressureB, this.engine.candidatePressure, coarseDirectory,
        this.engine.candidateLeafHeaders),
    };
    const pressureSummaryDirectory = this.engine.globalFineSummaries?.directory
      ?? this.engine.coarseOnlySummary?.directory ?? this.engine.unpublishedFineSummaryDirectory;
    this.engine.fineSummarySizingGroup = this.engine.createProjectionGroup(
      pressureSummaryDirectory, this.engine.pressureB,
      coarseDirectory);
    // Structural dirtiness compares the current pressure-owner decisions, not
    // the fine payload transaction. This stable group keeps the fine summary,
    // persistent topology-tile membership, and compact coarse authority
    // together across A/B fine generations.
    this.engine.topologyDecisionGroup = this.engine.createProjectionGroup(
      pressureSummaryDirectory,
      this.engine.topologyResidency.topologyTileStateBuffer,
      coarseDirectory,
    );
    this.engine.fineSeedAdapter?.setCoarsePhiSource(
      this.powerCoarseLevelSetSchedule.sampleSource,
    );
    this.engine.fineSeedAdapter?.setStructuredVelocitySource(structuredSource);
    if (this.engine.globalFineSourceA && this.engine.globalFineSourceB) {
      this.fineToPowerCoarseLevelSet = new WebGPUFineToCoarseLevelSet(this.engine.device, rowCapacity,
        this.engine.globalFineSourceA.plan.maximumResidentBricks * this.engine.globalFineSourceA.plan.samplesPerBrick);
      const compactCoarse = this.powerCoarseLevelSetSchedule.sampleSource;
      this.engine.globalFineTopologyAB = new WebGPUFineLevelSetTopology(
        this.engine.device, this.engine.globalFineSourceA, this.engine.globalFineSourceB, compactCoarse.wgsl(9),
        this.engine.deferPipelineCompilation,
      );
      this.engine.globalFineTopologyBA = new WebGPUFineLevelSetTopology(
        this.engine.device, this.engine.globalFineSourceB, this.engine.globalFineSourceA, compactCoarse.wgsl(9),
        this.engine.deferPipelineCompilation,
      );
      const changedKeysOffsetWords = this.engine.globalFineTopologyAB.pageDeltaLayout.changedKeysOffsetWords;
      if (changedKeysOffsetWords !== this.engine.globalFineTopologyBA.pageDeltaLayout.changedKeysOffsetWords) {
        throw new Error("Fine topology A/B page-delta layouts disagree");
      }
      // Deferred Power resources are created after the projection's initial
      // parameter upload. Publish the exact producer ABI only now, when both
      // page-delta layouts exist; leaving the constructor's zero sentinel in
      // this word rejects every recurring dirty-tile transaction.
      this.engine.device.queue.writeBuffer(this.engine.params, 36, new Uint32Array([changedKeysOffsetWords]));
      // Each destination generation consumes the exact delta published by
      // the topology transaction that authors that destination.
      this.engine.globalFineRedistanceA = new WebGPUFineLevelSetRedistance(
        this.engine.device, this.engine.globalFineSourceA, this.engine.globalFineTopologyBA,
        {
          deferPipelineCompilation: this.engine.deferPipelineCompilation,
          maximumRequiredJfaStride: maximumFineLevelSetJFAStride(
            planFineLevelSetBandFineCells(this.engine.fineLevelSetBandCells,
              this.engine.globalFineSourceA.plan.fineFactor).redistanceBandFineCells),
        },
      );
      this.engine.globalFineRedistanceB = new WebGPUFineLevelSetRedistance(
        this.engine.device, this.engine.globalFineSourceB, this.engine.globalFineTopologyAB,
        {
          deferPipelineCompilation: this.engine.deferPipelineCompilation,
          maximumRequiredJfaStride: maximumFineLevelSetJFAStride(
            planFineLevelSetBandFineCells(this.engine.fineLevelSetBandCells,
              this.engine.globalFineSourceB.plan.fineFactor).redistanceBandFineCells),
        },
      );
    }
    if (sceneHasTerrain(this.engine.scene) || this.engine.scene.rigidBodies.length > 0) {
      this.powerSolidVertices = new WebGPUOctreeSolidVertexSdf(
        this.engine.device, rowCapacity, this.engine.candidateLeafHeaders,
        this.engine.powerRowDelta.rows, this.engine.resources.terrain,
        this.engine.resources.rigidBodies, this.engine.powerRowDelta.controlOffsetWords,
      );
    }
    // One-step-lagged unilateral-contact active set: the projection stage
    // marks liquid rows holding tension against the closed ceiling; the next
    // step's boundary rebuild opens those rows' world faces so the solve
    // itself computes the separation with consistent divergence.
    this.structuredSeparationMask = this.engine.device.createBuffer({
      label: "Structured ceiling separation mask",
      size: this.engine.dims.nx * this.engine.dims.ny * this.engine.dims.nz * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.structuredBoundary = new WebGPUStructuredBoundaryCoefficients(this.engine.device, {
      structured: structuredSource,
      separationMask: this.structuredSeparationMask,
      coarse: this.powerCoarseLevelSetSchedule.sampleSource,
      solid: this.powerSolidVertices?.source,
      rigidBodies: this.engine.resources.rigidBodies,
      bodyCount: this.engine.scene.rigidBodies.length,
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
      closedBoundaryMask: structuredClosedBoundaryMask(this.engine.scene.container.top === "closed"),
      // Always bound; `analyticBootstrap` above decides whether it is ever
      // sampled, exactly as the projection layout does.
      bootstrapLevelSet: this.engine.surfaceState.texture,
      ...(this.engine.analyticSparseBootstrap ? { analyticBootstrap: {
        initialCondition: this.engine.scene.fluid.initialCondition,
        fillFraction: this.engine.scene.container.fillFraction,
        damBreakDimensions: this.engine.scene.fluid.initialDamBreakDimensions_m,
      } } : {}),
    });
    this.firstOrderVCycle.configureAccurateAuthority({
      control: this.structuredBoundary.control,
      worksets: this.structuredBoundary.worksets,
      coefficients: section63Source.coefficients,
      worksetStrideWords: this.structuredBoundary.worksetStrideWords,
      worksetBankStrideWords: this.structuredBoundary.worksetBankStrideWords,
      epochControlWord: 4,
      bankControlWord: 5,
    });
    // All stencil and smoother stages run over the GPU-published row/page
    // schedules. Only their scalar dependencies cross dispatch boundaries,
    // and those reductions use exact integer superaccumulators.
    const vector = (label: string) => this.engine.device.createBuffer({
      label, size: rowCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.pipelinedMGPCGVectors = Object.freeze({
      pressure: vector("Octree wide MGPCG pressure"),
      residual: vector("Octree wide MGPCG residual"),
      preconditioned: vector("Octree wide MGPCG preconditioned residual"),
      preconditionedImage: vector("Octree wide MGPCG A(Mr)"),
      direction: vector("Octree wide MGPCG direction"),
      directionImage: vector("Octree wide MGPCG A(direction)"),
    });
    const accurateOperator = this.firstOrderVCycle.accurateOperator;
    this.section43HybridPreconditioner = new WebGPUOctreeSection43HybridPreconditioner(
      this.engine.device, {
        rowCount: this.engine.compaction,
        firstOrderVCycle: this.firstOrderVCycle,
        secondOrderOperator: accurateOperator,
        section63: {
          coefficients: section63Source.coefficients,
          control: this.structuredBoundary.control,
          metrics: section63Source.topologyMetrics,
          ...spgrid,
        },
      }, {
        rowCapacity,
        boundarySmoothingIterations: normalizeOctreeSection43BoundarySmoothing(
          this.engine.solveTailPolicy.boundarySmoothingIterations,
        ),
      },
    );
    this.pipelinedMGPCG = new WebGPUOctreePipelinedMGPCG(this.engine.device, {
      coefficients: section63Source.coefficients,
      rhs: this.structuredDivergenceRhs,
      rowCount: this.engine.compaction,
      rowDispatch: structuredSource.liveRowDispatch,
      acceptedAuthority: this.structuredBoundary.control,
      operator: accurateOperator,
      preconditioner: this.section43HybridPreconditioner,
      vectors: this.pipelinedMGPCGVectors,
    }, {
      rowCapacity,
      relativeTolerance: this.engine.solveTailPolicy.relativeTolerance,
      maximumIterations: this.engine.solveTailPolicy.encodedOuterIterations,
      hardIterationCeiling: this.engine.solveTailPolicy.hardOuterIterationCeiling,
    });
    this.engine.pressureSolverControl = this.pipelinedMGPCG.control;
    this.topologyEpoch = new WebGPUOctreeTopologyEpoch(this.engine.device, {
      ownerArena: this.engine.ownerPages.arena,
      ownerCandidate: this.engine.ownerPages.candidateTransaction,
      frontier: this.engine.leafFrontier,
      descriptorCandidateControl: this.powerDescriptor.control,
      topologyCandidateControl: this.powerTopology.control,
      structuredCandidateControl: structured.candidateControl,
      structuredAcceptedControl: structured.control,
      boundaryCandidateControl: this.structuredBoundary.candidateControl,
      spgridCandidateControl: this.firstOrderVCycle.candidateControl,
      candidateLeafHeaders: this.engine.candidateLeafHeaders,
      acceptedLeafHeaders: this.engine.leafHeaders,
      candidatePressure: this.engine.candidatePressure,
      pressureA: this.engine.pressureA,
      pressureB: this.engine.pressureB,
      rowCountControl: this.engine.compaction,
    }, { rowCapacity, slotCapacity: structured.plan.slotCapacity,
      catalogVersion: OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version,
      carryPressureHistory: false });
    if (typeof process !== "undefined" && process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1") {
      this.readyEpochAudit = this.engine.device.createBuffer({
        label: "Diagnostic coupled epoch state after ready commit",
        size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.readyFrontierAudit = this.engine.device.createBuffer({
        label: "Diagnostic frontier state after ready commit",
        size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.readyCompactionAudit = this.engine.device.createBuffer({
        label: "Diagnostic compaction state after ready commit",
        size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
    this.airVelocitySupport = new WebGPUOctreeAirVelocitySupportProducer(this.engine.device, {
      structured: structuredSource,
      topology: this.powerTopology.source,
      owners: this.engine.ownerPages,
      boundaryEpoch: { buffer: this.structuredBoundary.control, offsetWords: 4 },
      liquidMask: this.structuredBoundary.liquidMask,
      sharedArena: this.powerCoarseLevelSetSchedule.selectorRows,
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      closedBoundaryMask: structuredClosedBoundaryMask(this.engine.scene.container.top === "closed"),
      maximumLeafSize: this.engine.maxLeafSize,
      // Air-extension demand must cover the complete characteristic and the
      // terminal interpolation owner around it. Using one fine factor happened
      // to cover the oversampled factor-4/8 bricks, but factor 1 packs four
      // coarse cells into each brick: its RK2 midpoint/backtrace reaches two
      // owners away and the velocity interpolation at that point reaches one
      // owner farther. Omitting that last owner leaves otherwise-valid
      // transported samples with no velocity authority.
      maximumDisplacementFineCells: this.engine.globalFineLevelSet
        ? (this.engine.globalFineLevelSet.plan.fineFactor === 1
          ? planFineLevelSetBandFineCells(this.engine.fineLevelSetBandCells, 1)
            .maximumBacktraceFineCells + 1
          : this.engine.globalFineLevelSet.plan.fineFactor)
        : 4,
      ...(this.engine.globalFineSourceA && this.engine.globalFineSourceB ? {
        fineSources: [this.engine.globalFineSourceA, this.engine.globalFineSourceB] as const,
        transportBandFineCells: planFineLevelSetBandFineCells(this.engine.fineLevelSetBandCells,
          this.engine.globalFineSourceA.plan.fineFactor).transportBandFineCells,
      } : {}),
    }, this.engine.deferPipelineCompilation);
    const producedSupport = this.airVelocitySupport.plan.support;
    if (producedSupport.totalBytes !== airSupportLayout.totalBytes
      || producedSupport.rowCapacity !== airSupportLayout.rowCapacity
      || producedSupport.slotCapacity !== airSupportLayout.slotCapacity
      || producedSupport.selectorTagOffsetWords !== airSupportLayout.selectorTagOffsetWords
      || producedSupport.regularTagOffsetWords !== airSupportLayout.regularTagOffsetWords
      || producedSupport.controlOffsetWords !== airSupportLayout.controlOffsetWords
      || producedSupport.supportVectorOffsetWords !== airSupportLayout.supportVectorOffsetWords
      || producedSupport.ownerDirectoryOffsetWords !== airSupportLayout.ownerDirectoryOffsetWords
      || producedSupport.ownerDirectoryCellCapacity !== airSupportLayout.ownerDirectoryCellCapacity
      || producedSupport.supportCapacity !== airSupportLayout.supportCapacity) {
      throw new Error("Structured air-support producer and shared suffix layouts disagree");
    }
    this.structuredDynamics = new WebGPUStructuredVelocityDynamics(this.engine.device, {
      structured: structuredSource, topology: this.powerTopology!.source, pressure: this.engine.pressureA,
      separationMask: this.structuredSeparationMask,
      divergenceRhs: this.structuredDivergenceRhs,
      liquidMask: this.structuredBoundary!.liquidMask,
      solidNormalVelocities: this.structuredBoundary!.solidNormalVelocities,
      rigidBodies: this.engine.resources.rigidBodies,
      rigidExchange: this.engine.resources.rigidExchange,
      boundaryWorksets: this.structuredBoundary!.worksets,
      boundaryControl: this.structuredBoundary!.control,
      selectorRows: this.powerCoarseLevelSetSchedule!.selectorRows,
      selectorStride: this.powerCoarseLevelSetSchedule!.selectorStride,
      selectorOffsetWords: this.powerCoarseLevelSetSchedule!.plan.selectorOffsetWords,
      airSupportLayout,
      bodyCount: this.engine.scene.rigidBodies.length,
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
      closedBoundaryMask: structuredClosedBoundaryMask(this.engine.scene.container.top === "closed"),
    }, this.engine.deferPipelineCompilation);
    if (this.engine.globalFineSourceA && this.engine.globalFineSourceB) {
      const fineTransportResources = {
        structured: structuredSource, topology: this.powerTopology.source,
        airSupport: {
          arena: this.airVelocitySupport.source.arena,
          layout: this.airVelocitySupport.source.plan.support,
          boundaryControl: this.structuredBoundary.control,
        },
        dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz] as const,
        physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
        maximumLeafSize: this.engine.maxLeafSize,
      };
      this.engine.globalFineTransportA = new WebGPUFineLevelSetTransport(
        this.engine.device, this.engine.globalFineSourceA, fineTransportResources,
        this.engine.deferPipelineCompilation);
      this.engine.globalFineTransportB = new WebGPUFineLevelSetTransport(
        this.engine.device, this.engine.globalFineSourceB, fineTransportResources,
        this.engine.deferPipelineCompilation);
    }
    this.powerVolumes = this.engine.device.createBuffer({ label: "Octree physical power-cell volumes", size: rowCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    if (this.engine.globalFineSourceA && this.engine.globalFineSourceB && this.powerCoarseLevelSet) {
      const coarseVolumeSource = { headers: this.engine.leafHeaders, records: this.powerCoarseLevelSet.records,
        physicalVolumes: this.powerVolumes,
        sampleDirectory: this.powerCoarseLevelSetSchedule!.sampleSource.directory,
        publicationControl: this.powerCoarseLevelSetSchedule!.control,
        rowCount: this.engine.compaction,
        dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz] as const,
        physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
        maximumLeafSize: this.engine.maxLeafSize,
        sampleRowCapacity: this.powerCoarseLevelSetSchedule!.sampleSource.rowCapacity };
      this.engine.globalFineVolumeA = new WebGPUFineLevelSetVolumeCorrection(
        this.engine.device, this.engine.globalFineSourceA, coarseVolumeSource,
        undefined, this.engine.deferPipelineCompilation,
      );
      this.engine.globalFineVolumeB = new WebGPUFineLevelSetVolumeCorrection(
        this.engine.device, this.engine.globalFineSourceB, coarseVolumeSource, this.engine.globalFineVolumeA.control,
        this.engine.deferPipelineCompilation,
      );
    }
    this.powerVolumeParams = this.engine.device.createBuffer({ label: "Octree power-volume parameters", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const cellVolume = (this.engine.scene.container.width_m / this.engine.dims.nx)
      * (this.engine.scene.container.height_m / this.engine.dims.ny)
      * (this.engine.scene.container.depth_m / this.engine.dims.nz);
    const data = new Float32Array(4); data[0] = cellVolume;
    this.engine.device.queue.writeBuffer(this.powerVolumeParams, 0, data);
    const powerVolumeProfile = performanceShaderVariant();
    const powerVolumeActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "octree/power-volume",
      profile: powerVolumeProfile,
    });
    const powerVolumeVariant = powerVolumeActivity.module(
      octreePowerVolumeActivityShader(powerVolumeActivity),
      `octree/power-volume/${powerVolumeProfile.cacheKey}`,
    );
    const shaderModule = this.engine.device.createShaderModule({
      label: "Publish physical octree power volumes",
      code: powerVolumeVariant.code,
    });
    this.initializePowerVolumePipeline = async () => {
      if (this.powerVolumePipeline) return;
      this.powerVolumePipeline = powerVolumeActivity.registerPipeline(
        await this.engine.device.createComputePipelineAsync({
          label: "Publish physical octree power volumes", layout: "auto",
          compute: { module: shaderModule, entryPoint: "publishPowerVolumes" },
        }),
      );
      this.powerVolumeGroup = this.engine.device.createBindGroup({
        layout: this.powerVolumePipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: this.powerVolumeParams! } },
          { binding: 1, resource: { buffer: this.powerTopology!.metrics } },
          { binding: 2, resource: { buffer: this.engine.leafHeaders } },
          { binding: 3, resource: { buffer: this.engine.compaction } },
          { binding: 4, resource: { buffer: this.powerVolumes! } },
        ],
      });
    };
    const powerAllocated = sumOctreePowerAllocationBreakdown({
      descriptors: this.powerDescriptor.plan.allocatedBytes,
      topology: this.powerTopology.plan.allocatedBytes,
      structuredVelocity: structured.allocatedBytes,
      structuredBoundary: this.structuredBoundary?.allocatedBytes ?? 0,
      structuredDynamics: this.structuredDynamics?.allocatedBytes ?? 0,
      airVelocitySupport: this.airVelocitySupport?.allocatedBytes ?? 0,
      solidVertices: this.powerSolidVertices?.plan.allocatedBytes ?? 0,
      coarseLevelSet: this.powerCoarseLevelSet?.plan.allocatedBytes ?? 0,
      coarseSchedule: this.powerCoarseLevelSetSchedule?.plan.allocatedBytes ?? 0,
      physicalVolumes: rowCapacity * 4,
      physicalVolumeParams: 16,
    });
    const fineAllocated = sumOctreePowerAllocationBreakdown({
      restriction: this.fineToPowerCoarseLevelSet?.plan.allocatedBytes ?? 0,
      topologyAB: this.engine.globalFineTopologyAB?.allocatedBytes ?? 0,
      topologyBA: this.engine.globalFineTopologyBA?.allocatedBytes ?? 0,
      redistanceA: this.engine.globalFineRedistanceA?.allocatedBytes ?? 0,
      redistanceB: this.engine.globalFineRedistanceB?.allocatedBytes ?? 0,
      transportA: this.engine.globalFineTransportA?.plan.allocatedBytes ?? 0,
      transportB: this.engine.globalFineTransportB?.plan.allocatedBytes ?? 0,
      volumeA: this.engine.globalFineVolumeA?.allocatedBytes ?? 0,
      volumeB: this.engine.globalFineVolumeB?.allocatedBytes ?? 0,
    });
    this.engine.info.powerDiagramAllocatedBytes = powerAllocated;
    this.engine.info.allocatedBytes += powerAllocated + fineAllocated;
    this.engine.info.globalFineLevelSetAllocatedBytes += fineAllocated;
    this.engine.workAccounting.setAuthorityBytes("power", powerAllocated);
    this.engine.workAccounting.setAuthorityBytes("fine-level-set", fineAllocated);
    this.engine.workAccounting.setScratchBytes("pressure-mgpcg",
      (this.pipelinedMGPCG?.allocatedBytes ?? 0)
      + (this.section43HybridPreconditioner?.allocatedBytes ?? 0));
    this.engine.workAccounting.setScratchBytes("multigrid", this.firstOrderVCycle.allocatedBytes);
    this.engine.workAccounting.sealAllocationInventory();
    this.engine.info.powerDiagramReady = true;
    this.engine.info.powerDiagramAuthoritative = Boolean(this.structuredVelocity && this.structuredBoundary)
      && (!sceneHasTerrain(this.engine.scene) || Boolean(this.powerSolidVertices));
  }

  /**
   * Every shader and allocation task this lane still owes after the
   * constructor. The generated Delaunay catalog is a 14 MB asynchronous
   * asset, so it loads as a task and the allocation that consumes it depends
   * on that task explicitly.
   */
  initializationTasks(): GPUInitializationTask[] {
    const tasks: GPUInitializationTask[] = [];
    // A rebuilt lane must not allocate a second power authority over the
    // first. The descriptor's presence is what proves this lane already owns
    // one, exactly as the engine's `else if` guard did.
    if (this.powerDescriptor) return tasks;
    let catalog: GeneratedOctreePowerCatalogViews | undefined;
    tasks.push({
    id: "octree.power-catalog.load",
    phase: "adaptive-topology",
    label: "Load octree power-diagram catalog",
    run: async (signal) => {
      try {
        const trace = typeof process !== "undefined" && process.env?.FLUID_POWER_INIT_TRACE === "1";
        if (trace) console.log(JSON.stringify({ phase: "power-init", label: "catalog-load", status: "started" }));
        if (signal.aborted) throw new DOMException("Power catalog initialization aborted", "AbortError");
        catalog = await loadGeneratedOctreePowerCatalog();
        if (trace) console.log(JSON.stringify({ phase: "power-init", label: "catalog-load", status: "finished" }));
        if (signal.aborted) throw new DOMException("Power catalog initialization aborted", "AbortError");
      } catch (error) {
        this.engine.info.powerDiagramReady = false;
        throw error;
      }
    },
    });
    tasks.push({
      id: "octree.power-authority.allocate",
      phase: "allocation",
      label: "Allocate reachable power-solver capabilities",
      dependencies: ["octree.power-catalog.load"],
      run: () => {
        if (!catalog) throw new Error("Octree power catalog was not loaded");
        try { this.initializeNativePowerAuthority(catalog); }
        catch (error) { this.engine.info.powerDiagramReady = false; throw error; }
      },
    });
    tasks.push({
      id: "octree.power-pipelines.publication",
      phase: "solver-pipelines",
      label: "Compile power descriptor and topology publication programs",
      dependencies: ["octree.power-authority.allocate"],
      run: async (signal, report) => {
        await this.powerDescriptor!.initializePipelines();
        await this.powerTopology!.initializePipelines();
        await this.structuredVelocity!.initializePipelines();
        await this.powerCoarseLevelSet!.initializePipeline();
        await this.powerCoarseLevelSetSchedule!.initializePipelines();
        await this.fineToPowerCoarseLevelSet?.initializePipelines();
        await this.section43HybridPreconditioner!.initializePipelines();
        await this.pipelinedMGPCG!.initializePipelines();
        await this.powerSolidVertices?.initializePipelines();
        await this.structuredBoundary!.initializePipelines();
        await this.topologyEpoch!.initializePipelines();
        await this.initializePowerVolumePipeline?.();
        const coarseSummaryTasks = this.engine.coarseOnlySummary?.initializationTasks() ?? [];
        for (let index = 0; index < coarseSummaryTasks.length; index += 1) {
          if (signal.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
          const task = coarseSummaryTasks[index]!;
          report?.(`Compile coarse summary: ${task.label} (${index}/${coarseSummaryTasks.length})`);
          await task.run(signal);
        }
      },
    });
    if (this.engine.deferPipelineCompilation) {
      tasks.push({
        id: "octree.power-pipelines.spgrid",
        phase: "solver-pipelines",
        label: "Compile persistent SPGrid topology programs",
        dependencies: ["octree.power-authority.allocate"],
        run: () => this.firstOrderVCycle!.initializePipelines(),
      });
      if (this.engine.globalFineSourceA && this.engine.globalFineSourceB) {
        tasks.push({
          id: "octree.power-pipelines.fine-topology",
          phase: "solver-pipelines",
          label: "Compile fine topology programs",
          dependencies: ["octree.power-authority.allocate"],
          run: async () => {
            await this.engine.globalFineTopologyAB!.initializePipelines();
            await this.engine.globalFineTopologyBA!.initializePipelines();
          },
        }, {
          id: "octree.power-pipelines.fine-redistance",
          phase: "solver-pipelines",
          label: "Compile fine redistance programs",
          dependencies: ["octree.power-authority.allocate"],
          run: async () => {
            await this.engine.globalFineRedistanceA!.initializePipelines();
            await this.engine.globalFineRedistanceB!.initializePipelines();
          },
        }, {
          id: "octree.power-pipelines.fine-transport",
          phase: "solver-pipelines",
          label: "Compile fine transport programs",
          dependencies: ["octree.power-authority.allocate"],
          run: async () => {
            await this.engine.globalFineTransportA!.initializePipelines();
            await this.engine.globalFineTransportB!.initializePipelines();
          },
        }, {
          id: "octree.power-pipelines.fine-volume",
          phase: "solver-pipelines",
          label: "Compile fine volume-correction programs",
          dependencies: ["octree.power-authority.allocate"],
          run: async () => {
            await this.engine.globalFineVolumeA!.initializePipelines();
            await this.engine.globalFineVolumeB!.initializePipelines();
          },
        });
      }
      tasks.push({
        id: "octree.power-pipelines.air-support",
        phase: "solver-pipelines",
        label: "Compile structured air-support programs",
        dependencies: ["octree.power-authority.allocate"],
        run: () => this.airVelocitySupport!.initializePipelines(),
      }, {
        id: "octree.power-pipelines.structured-dynamics",
        phase: "solver-pipelines",
        label: "Compile structured dynamics programs",
        dependencies: ["octree.power-authority.allocate"],
        run: (_signal, report) => this.structuredDynamics!.initializePipelines(
          (entryPoint, completed, total) => report?.(
            `Compile structured dynamics: ${entryPoint} (${completed}/${total})`)),
      });
    }
    return tasks;
  }

  encodeInactiveCandidate(encoder: GPUCommandEncoder): void {
    const descriptor = this.powerDescriptor, topology = this.powerTopology;
    const structured = this.structuredVelocity, boundary = this.structuredBoundary;
    const epoch = this.topologyEpoch;
    // A pending target is not redistanced/committed yet. The ready flip may
    // only demand support from the currently settled fine publication.
    const fine = this.engine.globalFineCurrentIsA ? this.engine.globalFineSourceA : this.engine.globalFineSourceB;
    if (!descriptor || !topology || !structured || !boundary || !epoch) {
      throw new Error("Inactive topology epoch requires every coupled power authority");
    }
    const generation = this.engine.candidatePowerGeneration;
    if (generation === 0) throw new Error("Inactive topology candidate has no attempt generation");
    const broker = new PassBroker(encoder);
    const dimensions: [number, number, number] = [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz];
    const spacing: [number, number, number] = [
      this.engine.scene.container.width_m / this.engine.dims.nx,
      this.engine.scene.container.height_m / this.engine.dims.ny,
      this.engine.scene.container.depth_m / this.engine.dims.nz,
    ];
    descriptor.encodeCandidate(broker, this.engine.candidateLeafHeaders, this.engine.ownerPages.arena, {
      dimensions, maximumLeafSize: this.engine.maxLeafSize,
      ownerCandidateControl: this.engine.ownerPages.candidateTransaction, generation,
      rowDelta: this.engine.powerRowDelta,
    });
    topology.encodeCandidate(broker, descriptor.candidateDescriptors, spacing,
      this.engine.powerRowDelta);
    structured.encodeCandidate(broker, generation, 0, topology.candidateMetrics,
      this.engine.candidateLeafHeaders);
    if (!this.structuredDynamics) {
      throw new Error("Inactive topology candidate requires structured velocity transfer");
    }
    this.structuredDynamics.encodeTopologyTransferCandidate(broker);
    structured.encodeCandidateReconstruction(broker, topology.candidateMetrics,
      this.engine.candidateLeafHeaders);
    this.powerSolidVertices?.encode(broker, { dimensions, physicalSpacing: spacing, generation,
      terrainEnabled: sceneHasTerrain(this.engine.scene), bodyCount: this.engine.scene.rigidBodies.length });
    boundary.encodeCandidate(broker, fine, structured.candidateControl);
    this.firstOrderVCycle.encodeCandidateSetup(broker, {
      solverControl: this.engine.pressureSolverControl, rowCount: this.engine.compaction,
      sourceControl: structured.candidateControl,
      topologyMetrics: topology.candidateMetrics,
    });
    epoch.encodeCandidateValidation(broker, generation);
  }


  encodeReadyTopologyFlip(encoder: GPUCommandEncoder): void {
    const descriptor = this.powerDescriptor, topology = this.powerTopology;
    const structured = this.structuredVelocity, boundary = this.structuredBoundary;
    const epoch = this.topologyEpoch;
    if (!descriptor || !topology || !structured || !boundary || !epoch
      || this.engine.candidatePowerGeneration === 0) {
      throw new Error("Ready topology flip requires a complete inactive coupled candidate");
    }
    const acceptedGeneration = this.engine.candidatePowerGeneration;
    const broker = new PassBroker(encoder);
    epoch.encodeReadyCommitGate(broker, acceptedGeneration);
    this.engine.ownerPages.encodeReadyCommit(broker);
    descriptor.encodeReadyCommit(broker);
    topology.encodeReadyCommit(broker);
    structured.encodeReadyCommit(broker);
    boundary.encodeReadyCommit(broker);
    this.firstOrderVCycle.encodeReadySetupCommit(broker, {
      solverControl: this.engine.pressureSolverControl, rowCount: this.engine.compaction,
    });
    if (!this.airVelocitySupport) {
      throw new Error("Ready topology flip requires the Section 5 air-support producer");
    }
    // Aanjaneya et al. Section 5 first maps the migrated projected power-face
    // field to ordinary faces, extends it outside liquid, and maps it back
    // before the newly accepted epoch may be sampled.
    this.airVelocitySupport.encode(broker, acceptedGeneration,
      this.engine.globalFineBootstrapped ? (this.engine.globalFineCurrentIsA ? 0 : 1) : undefined,
      this.airSupportGravityImpulse(), "topology-commit");
    // This helper owns its broker. Close the publication pass before returning
    // so both the cold checkpoint (which finishes the encoder immediately) and
    // the recurring caller can safely append or finish commands.
    broker.fence("accepted Section 5 air-support epoch published");
    if (this.readyEpochAudit) {
      broker.copyBufferToBuffer(epoch.state, 0, this.readyEpochAudit, 0, 64);
      broker.copyBufferToBuffer(this.engine.leafFrontier, 0, this.readyFrontierAudit!, 0, 64);
      broker.copyBufferToBuffer(this.engine.compaction, 0, this.readyCompactionAudit!, 0, 64);
    }
    this.engine.activePowerGeneration = acceptedGeneration;
    this.engine.candidatePowerGeneration = 0;
    this.engine.info.topologyReused = false;
  }


  /**
   * Power rebuilds the coupled candidate on every advance: the descriptor,
   * the Delaunay topology and the structured banks are all row-indexed, so a
   * skipped epoch would leave them addressing a retired row set.
   */
  readonly topologyCadenceIsEveryAdvance = true;

  /**
   * The live coarse-band dials describe the adaptive lane's solve tuning and
   * velocity extension. Power 2017 keeps its frozen reference behaviour and
   * has no equivalent seam, so it never records an applied set.
   */
  runtimeDialsApplicable(): boolean { return false; }

  applyRuntimeDials(_dials: OctreeRuntimeDials): void {}

  /**
   * Directory generation N is produced after topology N and is the authority
   * for the next topology rebuild. Queue this expected generation before the
   * command buffer; later surface publication uses its own parameter buffer.
   */
  stampCoarseDirectoryGeneration(): void {
    if (!this.powerCoarseLevelSetSchedule) return;
    const generation = this.powerCoarseLevelSetGeneration & 0x3fff_ffff;
    const flags = 1 | (generation << 2);
    this.engine.device.queue.writeBuffer(this.engine.params, 140, new Uint32Array([flags >>> 0]));
  }

  encodeAdvance(encoder: GPUCommandEncoder, ctx: OctreeAdvanceContext): GPUCommandEncoder {
    const options = ctx, scope = ctx.scope;
    const solveBudget = this.pipelinedMGPCG?.iterationBudget
      ?? this.engine.solveTailPolicy.encodedOuterIterations;
    this.engine.workAccounting.beginSubstep();
    this.engine.info.pressureIterationBudget = solveBudget;
    this.engine.info.pressureIterationHardBudget = this.engine.pressureHardIterationCeiling();
    const fineEngineSplits = octreeFineEngineSplitsEnabled();
    const splitProductionPhase = (
      enginePhase: OctreeEnginePhase | undefined,
      finePhase: OctreeFineSemanticPhase,
    ) => {
      const phase = fineEngineSplits ? finePhase : enginePhase;
      if (options?.productionBoundary && phase) {
        encoder = options.productionBoundary(phase, encoder);
      }
    };
    // Candidate rows were compacted at the previous tail and became visible
    // only through the coupled beginning-of-substep commit.
    const initialInA = !this.engine.latestPressureInA;
    const pressureBroker = new PassBroker(encoder);
    encoder = this.encodeNativePowerAssembly(
      encoder,
      options?.productionBoundary,
      options?.productionBoundary && fineEngineSplits ? undefined : pressureBroker,
      fineEngineSplits,
    );
    const pressureIn = initialInA ? this.engine.pressureA : this.engine.pressureB;
    const pressureOut = initialInA ? this.engine.pressureB : this.engine.pressureA;
    if (!this.structuredVelocity) throw new Error("Pressure solve requires the accepted structured authority");
    const solveBroker = new PassBroker(encoder);
    const pipelined = this.pipelinedMGPCG;
    if (!pipelined) throw new Error("Row-parallel pressure executor was not constructed");
    // The SPGrid hierarchy and compiled A2 images are GPU-published before
    // the wide recurrence consumes them. Every following stencil/smoother
    // launch is row/page parallel; only the exact integer scalar finish is a
    // singleton.
    this.firstOrderVCycle.encodeSetup(solveBroker, {
      solverControl: this.engine.pressureSolverControl, rowCount: this.engine.compaction,
    });
    pipelined.encode(solveBroker, {
      pressureSeed: pressureIn,
      pressureOut,
      encodedIterationBudget: solveBudget,
    });
    this.engine.latestPressureInA = !initialInA;
    // Stage solve feedback (residual sums + row/entry counts) while this
    // encoder still owns write ordering on compaction; the async diagnostics
    // poll then reads the staging buffer without racing the next rebuild.
    solveBroker.copyBufferToBuffer(
      this.engine.compaction, this.engine.compactionByteLength - 32, this.engine.solveStats, 0, 32);
    splitProductionPhase("solveEngine", "mgpcgSolve");
    const finalInA = this.engine.latestPressureInA;
    const projectionBroker = new PassBroker(encoder);
    this.encodeStructuredProjection(
      projectionBroker,
      finalInA ? this.engine.pressureA : this.engine.pressureB,
    );
    projectionBroker.fence("projected structured velocity published");
    if (scope === "power-operator-only") {
      // The explicit t=0 dependency chain publishes the fine level set next,
      // then owns each Section 5 transaction in its own named checkpoint.
      // This is a lifecycle boundary, not an alternate simulation path.
      return encoder;
    }
    splitProductionPhase(undefined, "structuredProjection");
    encoder = this.engine.encodePendingFineSettlement(encoder, options?.productionBoundary);
    if (!this.engine.coarseOnlySurfaceTracking && this.engine.powerTimestep_s > 0) {
      if (!this.airVelocitySupport || !this.engine.globalFineBootstrapped) {
        throw new Error("Live Section 5 support refresh requires the settled fine generation");
      }
      const supportBroker = new PassBroker(encoder);
      this.airVelocitySupport.encode(supportBroker, this.engine.activePowerGeneration,
        this.engine.globalFineCurrentIsA ? 0 : 1, this.airSupportGravityImpulse(), "settled-fine");
      supportBroker.fence("settled fine-demand air support published");
      encoder = supportBroker.commandEncoder();
    }
    const projectionTailBroker = new PassBroker(encoder);
    projectionTailBroker.fence("structured projection tail published");
    this.engine.encodeOverlayMaterialization(encoder, finalInA);
    if (this.engine.powerTimestep_s > 0) this.engine.powerAdvancingPressureSteps += 1;
    splitProductionPhase("rowEngineB", "structuredProjectionTail");
    return encoder;
  }

  /** The substep's body-force increment for the Section 5 extension: air
   * vectors are rebuilt from projected liquid seeds every epoch, so the
   * producer folds exactly one g*dt into each reconstructed air vector. */
  private airSupportGravityImpulse(
    dt_s = this.engine.powerTimestep_s,
  ): [number, number, number] {
    const gravity = this.engine.scene.fluid.gravity_m_s2;
    return [gravity.x * dt_s, gravity.y * dt_s, gravity.z * dt_s];
  }
  private encodeNativePowerAssembly(
    encoder: GPUCommandEncoder,
    productionBoundary?: OctreeSemanticBoundary,
    sharedBroker?: PassBroker,
    fineEngineSplits = octreeFineEngineSplitsEnabled(),
  ): GPUCommandEncoder {
    const structured = this.structuredVelocity;
    const dynamics = this.structuredDynamics;
    const volumes = this.powerVolumes;
    const volumePipeline = this.powerVolumePipeline;
    const volumeGroup = this.powerVolumeGroup;
    if (!structured || !dynamics
      || !volumes || !volumePipeline || !volumeGroup) {
      throw new Error("Power assembly requires the complete direct structured authority");
    }
    let broker = sharedBroker ?? new PassBroker(encoder);
    const splitProductionPhase = (
      enginePhase: OctreeEnginePhase | undefined,
      finePhase: OctreeFineSemanticPhase,
      closeForRawContinuation = false,
    ) => {
      const phase = fineEngineSplits ? finePhase : enginePhase;
      if (productionBoundary && phase) {
        broker.fence(`production phase ${phase}`);
        encoder = productionBoundary(phase, encoder);
        broker = new PassBroker(encoder);
      } else if (closeForRawContinuation) {
        broker.fence(`raw continuation after ${finePhase}`);
      }
    };
    // The direct structured topology was committed and timestamped at the
    // beginning-of-substep seam. Active dynamics never rebuild rows/pages in
    // the solve graph, so no synthetic boundary belongs here.
    // The t=0 authority warmup reconstructs, solves, and projects but has no
    // transport interval. Its exact advection map is identity; invoking the
    // departure sampler here can only reject boundary stencils that no
    // positive-time characteristic ever requested.
    if (this.engine.powerTimestep_s > 0) dynamics.encodeAdvection(
      broker, this.engine.powerTimestep_s, this.engine.surfaceInflow,
    );
    dynamics.encodeForcesAndDivergence(
      broker, this.engine.powerTimestep_s, this.engine.scene.fluid.density_kg_m3, [
        this.engine.scene.fluid.gravity_m_s2.x,
        this.engine.scene.fluid.gravity_m_s2.y,
        this.engine.scene.fluid.gravity_m_s2.z,
      ], this.engine.surfaceInflow);
    splitProductionPhase(undefined, "structuredAdvectionBoundaryRhs");
    const pass = broker.compute({ label: "Publish physical power-cell volumes" });
    pass.setPipeline(volumePipeline); pass.setBindGroup(0, volumeGroup);
    pass.dispatchWorkgroupsIndirect(structured.source.liveRowDispatch,
      structured.source.section63.liveRowDispatchOffsetBytes);
    splitProductionPhase(undefined, "structuredVolumeCapture");
    // The inactive hierarchy was built at the preceding tail and committed at
    // this substep's head. Nothing in advection/RHS assembly mutates its row
    // geometry or accepted Section 6.3 coefficients, so an accepted recapture
    // here only defeats encodeSetup's already-committed fast path and encodes
    // the full candidate chain a second time. The next tail captures the next
    // candidate from its own explicit source mode.
    splitProductionPhase("rowEngineA", "finalPressureRowAssembly", true);
    return encoder;
  }
  private encodeStructuredProjection(
    broker: PassBroker,
    pressure: GPUBuffer,
  ): void {
    const dynamics = this.structuredDynamics;
    if (!dynamics || pressure !== this.engine.pressureA && pressure !== this.engine.pressureB) {
      throw new Error("Structured projection pressure buffer is not an accepted solve target");
    }
    dynamics.encodeProjection(broker, this.engine.powerTimestep_s, this.engine.scene.fluid.density_kg_m3, [
      this.engine.scene.fluid.gravity_m_s2.x,
      this.engine.scene.fluid.gravity_m_s2.y,
      this.engine.scene.fluid.gravity_m_s2.z,
    ], pressure, this.engine.dynamicCouplingBodyCount, this.engine.surfaceInflow);
    if (!this.airVelocitySupport || this.engine.activePowerGeneration === 0) {
      throw new Error("Structured projection requires an accepted Section 5 air-support epoch");
    }
  }
  private encodeCoarsePhiCorrection(
    broker: PassBroker,
    fine: WebGPUFineLevelSetBrickSource,
    topology: WebGPUFineLevelSetTopology,
    dt_s: number,
    allowValidatedProvisional = false,
  ): void {
    const structured = this.structuredVelocity?.source;
    if (!this.fineToPowerCoarseLevelSet || !this.powerCoarseLevelSetSchedule || !structured) {
      throw new Error("Coarse phi correction requires fine restriction and structured velocity authorities");
    }
    const correction = this.fineToPowerCoarseLevelSet.encode(broker, fine, {
      headers: this.engine.leafHeaders,
      // Adaptive candidate construction updates `compaction` before the
      // accepted row headers/geometry flip. Restrict the transported fine
      // generation over the immutable accepted structured epoch instead of
      // mixing candidate N+1's count with accepted N's row identities.
      rowCount: structured.control,
      rowCountOffsetWords: 2,
      topologyControl: topology.control,
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
      maximumLeafSize: this.engine.maxLeafSize,
      allowValidatedProvisional,
    });
    this.powerCoarseLevelSetSchedule.encode(broker, {
      headers: this.engine.leafHeaders,
      structured,
      rowCount: { buffer: structured.control, offset: 2 * 4, size: 4 },
      fineCorrection: {
        rowOffsets: correction.rowOffsets,
        contributions: correction.contributions,
        contributionCount: correction.counts,
        aggregated: correction.aggregated,
      },
    }, {
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
      dt: dt_s,
      maximumLeafSize: this.engine.maxLeafSize,
      generation: fine.generation & 0x3fff_ffff,
    });
    this.powerCoarseLevelSetGeneration = fine.generation & 0x3fff_ffff;
  }

  /** Power solves into the shared A/B banks, so the engine's latch names them. */
  overlayPressureAuthorityIsA(pressureInA: boolean): boolean { return pressureInA; }

  /**
   * Cold bootstrap of the compact power-coarse tracker from the current
   * surface leaves. Returns whether it ran: the caller skips this step's
   * pre-force correction when it did, which would otherwise transport the
   * same surface twice on step zero.
   */
  encodeSurfaceCoarseBootstrap(preparationBroker: PassBroker): boolean {
    const structuredSource = this.structuredVelocity?.source;
    if (!this.powerCoarseLevelSet || !this.powerCoarseLevelSetSchedule || !structuredSource) {
      return false;
    }
    if (this.powerCoarseLevelSetBootstrapped) return false;
    this.powerCoarseLevelSet.encodeBootstrapFromSurfaceLeaves(
      preparationBroker, this.engine.fineSeedAdapter!.leaves, structuredSource.liveRowDispatch,
    );
    this.powerCoarseLevelSetSchedule.encode(preparationBroker, {
      headers: this.engine.leafHeaders, structured: structuredSource,
      rowCount: this.engine.compaction,
    }, {
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
      dt: 0,
      maximumLeafSize: this.engine.maxLeafSize,
      generation: this.powerCoarseLevelSetGeneration & 0x3fff_ffff,
    });
    // The dense coarse tracker consumes the backend that owns the surface
    // step, which for this lane is the bootstrap itself.
    this.engine.coarseOnlySummary?.encode(preparationBroker);
    this.powerCoarseLevelSetBootstrapped = true;
    return true;
  }

  /**
   * The compact tracker is never the whole moving surface while a fine band
   * exists; the coarse-only configuration is handled by the fallback below,
   * which the engine reaches only when no fine band was allocated.
   */
  encodeCoarseOnlySurfaceAdvance(
    _broker: PassBroker,
    _dt_s: number,
    _inflow: SurfaceInflowState | undefined,
  ): GPUCommandEncoder | undefined {
    return undefined;
  }

  encodeCoarseOnlyFallbackAdvance(
    preparationBroker: PassBroker,
    dt_s: number,
    coarseBootstrappedThisStep: boolean,
  ): GPUCommandEncoder | undefined {
    const structuredSource = this.structuredVelocity?.source;
    if (!this.engine.coarseOnlySurfaceTracking || !this.powerCoarseLevelSetSchedule
      || !structuredSource) {
      return undefined;
    }
    // Historical coarse-only mode: compact octree phi is the sole moving
    // surface authority. Advance it directly; no global fine topology,
    // page publication, transport, redistance, summary, restriction or
    // volume-correction object exists in this configuration.
    if (!coarseBootstrappedThisStep && this.powerCoarseLevelSetBootstrapped) {
      this.powerCoarseLevelSetGeneration =
        (this.powerCoarseLevelSetGeneration + 1) & 0x3fff_ffff;
      if (this.powerCoarseLevelSetGeneration === 0) {
        this.powerCoarseLevelSetGeneration = 1;
      }
      this.powerCoarseLevelSetSchedule.encode(preparationBroker, {
        headers: this.engine.leafHeaders,
        structured: structuredSource,
        rowCount: this.engine.compaction,
      }, {
        dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
        physicalCellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
        dt: dt_s,
        maximumLeafSize: this.engine.maxLeafSize,
        generation: this.powerCoarseLevelSetGeneration,
      });
      this.engine.coarseOnlySummary?.encode(preparationBroker);
    }
    let encoder = preparationBroker.commandEncoder();
    if (!this.airVelocitySupport || this.engine.activePowerGeneration === 0) {
      throw new Error("Coarse-only surface tracking requires Section 5 air support");
    }
    const supportBroker = new PassBroker(encoder);
    this.airVelocitySupport.encode(supportBroker, this.engine.activePowerGeneration,
      undefined, this.airSupportGravityImpulse(dt_s), "settled-fine");
    // Cold bootstrap precedes the first air-support publication. Rebuild
    // the compact coarse tracker immediately afterward so generation zero
    // starts from complete velocity/owner support rather than an empty bank.
    if (coarseBootstrappedThisStep) this.engine.coarseOnlySummary?.encode(supportBroker);
    supportBroker.fence("coarse-only air support published");
    encoder = supportBroker.commandEncoder();
    return encoder;
  }

  fineTopologyCoarseEntry(binding: number): GPUBindGroupEntry {
    return { binding,
      resource: { buffer: this.powerCoarseLevelSetSchedule!.sampleSource.directory } };
  }

  /**
   * The staged transport samples the accepted structured face authority, so a
   * configuration without one has no transport to encode. That absence is
   * what the engine reads as "nothing was transported this step".
   */
  fineTransportStage(currentIsA: boolean): OctreeFineLevelSetTransportStage | undefined {
    if (!this.structuredVelocity?.source) return undefined;
    return currentIsA ? this.engine.globalFineTransportA : this.engine.globalFineTransportB;
  }

  encodeFineTransport(
    broker: PassBroker,
    stage: OctreeFineLevelSetTransportStage,
    request: OctreeFineTransportRequest,
  ): PassBroker {
    return (stage as WebGPUFineLevelSetTransport).encode(broker, {
      timestep: request.timestep,
      ...(request.inflow ? { inflow: request.inflow } : {}),
      boundaryPolicy: "closed-neumann",
      openTopBoundary: request.openTopBoundary,
      dynamicBoundary: request.dynamicBoundary,
      transportBandCells: request.transportBandCells,
      maximumBacktraceFineCells: request.maximumBacktraceFineCells,
    });
  }

  encodeCoarsePhiBeforeForces(
    broker: PassBroker,
    target: WebGPUFineLevelSetBrickSource,
    topology: WebGPUFineLevelSetTopology,
    dt_s: number,
  ): void {
    this.encodeCoarsePhiCorrection(broker, target, topology, dt_s, true);
  }

  requireSettledSupport(): void {
    if (!this.airVelocitySupport || !this.engine.globalFineBootstrapped
      || this.engine.activePowerGeneration === 0) {
      throw new Error("Settled t=0 fine authority requires Section 5 air support");
    }
  }

  encodeSettledSupport(
    encoder: GPUCommandEncoder,
    dt_s: number,
    _phase: "t0" | "recurring",
  ): GPUCommandEncoder {
    if (!this.airVelocitySupport) return encoder;
    const supportBroker = new PassBroker(encoder);
    this.airVelocitySupport.encode(supportBroker, this.engine.activePowerGeneration,
      this.engine.globalFineCurrentIsA ? 0 : 1, this.airSupportGravityImpulse(dt_s));
    supportBroker.fence("settled t=0 fine-demand air support published");
    return supportBroker.commandEncoder();
  }

  /**
   * Nothing to bootstrap: this lane's coarse authority is published from the
   * surface step's own bootstrap above, before any settlement runs.
   */
  encodeSettlementBootstrap(_broker: PassBroker, _target: WebGPUFineLevelSetBrickSource): void {}

  /** Power repairs measured volume loss with its own post-step correction. */
  readonly settlementVolumePolicy = "correct" as const;

  /** The restriction hook below republishes coarse phi; nothing precedes it. */
  encodeSettlementCoarseRefresh(
    _broker: PassBroker,
    _target: WebGPUFineLevelSetBrickSource,
  ): void {}

  encodeSettlementRestriction(
    restrictionBroker: PassBroker,
    target: WebGPUFineLevelSetBrickSource,
    topology: WebGPUFineLevelSetTopology,
  ): void {
    this.encodeCoarsePhiCorrection(restrictionBroker, target, topology, 0);
    if (!this.powerCoarseLevelSetSchedule) return;
    const coarse = this.powerCoarseLevelSetSchedule.sampleSource;
    this.engine.globalFineSummaries?.encode(restrictionBroker, target, {
      buffer: topology.pageDelta,
      layout: topology.pageDeltaLayout,
    }, {
      directory: coarse.directory,
      control: coarse.control,
      delta: coarse.delta,
      deltaHeaderWords: coarse.deltaHeaderWords,
      deltaRecordWords: coarse.deltaRecordWords,
    });
  }

  // --- Readback and diagnostics --------------------------------------------
  get debug(): OctreeLaneDebugSources {
    const structured = this.structuredVelocity;
    const topology = this.powerTopology;
    return {
      ...(this.topologyEpoch ? { topologyEpochState: this.topologyEpoch.state } : {}),
      ...(this.airVelocitySupport ? { airSupportScratch: this.airVelocitySupport.scratch } : {}),
      ...(this.firstOrderVCycle ? { spgridLevelDelta: this.firstOrderVCycle.levelDelta } : {}),
      ...(structured ? {
        structuredVelocityControl: structured.control,
        structuredRowVelocities: structured.source.rowVelocities,
        structuredAuthority: structured.source.authority,
        structuredWorksets: structured.source.familyWorksets.regularInterior.buffer,
      } : {}),
      ...(this.structuredBoundary ? { structuredBoundaryControl: this.structuredBoundary.control } : {}),
      ...(this.structuredDynamics
        ? { structuredProjectionEnergyStats: this.structuredDynamics.projectionEnergyStats } : {}),
      ...(this.powerDescriptor ? {
        powerDescriptorControl: this.powerDescriptor.control,
        powerDescriptorRows: this.powerDescriptor.descriptors,
      } : {}),
      ...(topology ? {
        powerTopologyControl: topology.control,
        powerTopologyMetrics: topology.metrics,
        powerCatalogEntryHeaders: topology.catalogEntryHeaders,
        powerCatalogFaces: topology.catalogFaces,
      } : {}),
      ...(this.powerCoarseLevelSetSchedule
        ? { globalFineCoarseLevelSetControl: this.powerCoarseLevelSetSchedule.control } : {}),
      ...(this.fineToPowerCoarseLevelSet
        ? { globalFineRestrictionControl: this.fineToPowerCoarseLevelSet.control } : {}),
    };
  }

  debugSources(): Record<string, unknown> { return { ...this.debug }; }

  /** Power solves into the engine's shared A/B banks; it keeps no frame views. */
  get diagnosticPressureBanks() { return undefined; }

  get solverSymmetryStageAuditBuffers(): OctreeLaneSymmetryStageAuditBuffers | undefined {
    return this.pipelinedMGPCG?.symmetryStageAuditBuffers;
  }

  genericCoarseDirectory() {
    const coarse = this.powerCoarseLevelSetSchedule?.sampleSource;
    return coarse ? { directory: coarse.directory, rowCapacity: coarse.rowCapacity } : undefined;
  }

  coarseLevelSetPublication(): OctreeLaneCoarseLevelSetPublication | undefined {
    const coarse = this.powerCoarseLevelSetSchedule?.sampleSource;
    // The bootstrap is what makes the directory a surface rather than an
    // empty bank, so an unbootstrapped schedule publishes nothing.
    if (!coarse || !this.powerCoarseLevelSetBootstrapped) return undefined;
    return {
      directory: coarse.directory,
      control: coarse.control,
      rowCapacity: coarse.rowCapacity,
      generation: this.powerCoarseLevelSetGeneration & 0x3fff_ffff,
    };
  }

  summaryCoarseDebug() {
    const coarse = this.powerCoarseLevelSetSchedule;
    return coarse ? { control: coarse.control, delta: coarse.sampleSource.delta } : undefined;
  }

  /**
   * The adaptive step receipt is the Losasso graph's scalar clock. This lane
   * advances its own generation on the host, so admitting one would pair a
   * foreign clock with this directory.
   */
  acceptsAdaptiveSurfaceGenerationReceipt(): boolean { return false; }

  hasCoarseSurfaceAuthority(): boolean { return this.powerCoarseLevelSetBootstrapped; }

  /** Whether the compact velocity authority can source the sparse renderer. */
  compactRendererSourceReady(): boolean {
    return Boolean(this.structuredVelocity && this.structuredBoundary && this.structuredDynamics);
  }

  /** Coarse projection groups exist exactly while the schedule does. */
  get hasCoarseProjectionGroups(): boolean {
    return this.powerCoarseLevelSetSchedule !== undefined;
  }

  /** The pipelined executor's own budget, when it has been constructed. */
  solverIterationBudget(): number | undefined { return this.pipelinedMGPCG?.iterationBudget; }

  workAccountingBuffers(): Readonly<{
    pressureRhs?: GPUBufferBinding;
    section63Coefficients?: GPUBufferBinding;
  }> {
    return {
      ...(this.structuredDivergenceRhs ? { pressureRhs: { buffer: this.structuredDivergenceRhs } } : {}),
      ...(this.structuredVelocity ? {
        section63Coefficients: {
          buffer: this.structuredVelocity.source.section63.coefficients,
          size: this.structuredVelocity.source.section63.coefficientBankStrideWords * 4,
        },
      } : {}),
    };
  }

  /**
   * Retire the invocation-stable coarse-phi parameter slots this lane staged
   * for the submitted encoder. A lane without a schedule has none.
   */
  retireSubmittedEncoder(encoder: GPUCommandEncoder): void {
    this.powerCoarseLevelSetSchedule?.retireSubmittedEncoder(encoder);
  }

  /** The analytic bootstrap sign is retired from this lane's boundary bank. */
  retireAnalyticBootstrap(): void {
    this.structuredBoundary?.retireAnalyticBootstrap();
  }

  techniqueDebugSources(): OctreeLaneTechniqueDebugSources | undefined {
    const topology = this.powerTopology?.source;
    const structured = this.structuredVelocity?.source;
    const tetrahedronHeaders = topology?.catalogTetrahedronHeaders;
    const tetrahedra = topology?.catalogTetrahedra;
    const tetrahedronVertices = topology?.catalogTetrahedronVertices;
    if (!topology || !structured || !tetrahedronHeaders || !tetrahedra || !tetrahedronVertices) {
      return undefined;
    }
    return {
      topologyMetrics: topology.metrics,
      catalogEntryHeaders: topology.catalogEntryHeaders,
      catalogFaces: topology.catalogFaces,
      tetrahedronHeaders,
      tetrahedra,
      tetrahedronVertices,
      structuredAuthority: structured.authority,
      structuredParams: structured.params,
      structuredRowGeometry: structured.rowGeometry,
      structuredRowVelocities: structured.rowVelocities,
      structuredControl: structured.control,
      // Optional: a scene can reach a publication before the power-coarse
      // level set has one, and the cell trace reads zero flags as "never
      // corrected" rather than as a phi of zero.
      ...(this.powerCoarseLevelSetSchedule
        ? { coarsePhiValues: this.powerCoarseLevelSetSchedule.sampleSource.values } : {}),
    };
  }

  captureFrontierFailureAuthorityControls(capture: OctreeLaneFrontierFailureCapture): void {
    capture("descriptorCandidate", this.powerDescriptor!.control);
    capture("topologyCandidate", this.powerTopology!.control);
    capture("structuredCandidate", this.structuredVelocity!.candidateControl);
    capture("boundaryCandidate", this.structuredBoundary!.candidateControl);
    capture("spgridCandidate", this.firstOrderVCycle.candidateControl);
    capture("epoch", this.topologyEpoch!.state);
  }

  captureFrontierFailureCoarseSources(capture: OctreeLaneFrontierFailureCapture): void {
    if (!this.powerCoarseLevelSetSchedule) return;
    const coarse = this.powerCoarseLevelSetSchedule.sampleSource;
    capture("coarseControl", coarse.control);
    capture("coarseDirectory", coarse.directory);
    capture("coarseDelta", coarse.delta);
  }

  captureFrontierFailureCandidateSources(capture: OctreeLaneFrontierFailureCapture): void {
    if (this.powerDescriptor) {
      capture("descriptorCandidates", this.powerDescriptor.candidateDescriptors);
      capture("descriptorStatuses", this.powerDescriptor.dispatch, 4 * 4);
    }
    // The structured publication's nine indirect records. Words 3..5 are the
    // slot dispatch consumed by `classifyStructuredCatalogSlots`, while words
    // 18..20 are the exact changed-face transfer record. A record Dawn's
    // indirect-args validator zeroed raises no error and simply never runs the
    // stage, which is indistinguishable from a physics rejection in the
    // control words alone.
    if (this.structuredVelocity) {
      capture("structuredDispatch", this.structuredVelocity.liveRowDispatch);
    }
  }

  async decodeFrontierFailure(controls: Readonly<{
    descriptorCandidate: readonly number[];
    boundaryCandidate: readonly number[];
    coarseControl: readonly number[];
  }>): Promise<OctreeLaneFrontierFailureReceipt> {
    const receipt: OctreeLaneFrontierFailureReceipt = {
      spgridLevelDelta: [], spgridCandidateDispatch: [],
    };
    if (this.firstOrderVCycle) {
      const spgridFailure = await this.firstOrderVCycle.readCandidateFailureDiagnostics();
      receipt.spgridLevelDelta = spgridFailure.levelDelta;
      receipt.spgridCandidateDispatch = spgridFailure.candidateDispatch;
    }
    const descriptorFirstError = Number(controls.descriptorCandidate[3]) >>> 0;
    if (this.powerDescriptor && Number(controls.descriptorCandidate[2]) !== 0
      && descriptorFirstError < this.engine.pressureCapacity.rowCapacity) {
      receipt.descriptorFailureRow =
        await this.readPowerDescriptorCandidateFailure(descriptorFirstError);
    }
    const boundary = controls.boundaryCandidate;
    const boundaryFirstError = Number(boundary[1]) >>> 0;
    if (this.powerDescriptor && (Number(boundary[0]) & 2) !== 0
      && boundaryFirstError < this.engine.pressureCapacity.rowCapacity) {
      receipt.boundaryFailureRow = await this.readPowerDescriptorCandidateFailure(boundaryFirstError);
    }
    const coarseFirstError = Number(controls.coarseControl[1]) >>> 0;
    if (Number(controls.coarseControl[0]) !== 0
      && coarseFirstError < this.engine.pressureCapacity.rowCapacity) {
      receipt.coarseFailureRow = await this.readPowerCoarseFailureRow(coarseFirstError);
    }
    return receipt;
  }

  /**
   * Copy this lane's disjoint regions of the shared 952-byte global-fine QA
   * readback. Every destination range below belongs to this lane alone, so
   * emitting them together rather than interleaved with the engine's own
   * copies cannot change what the mapped words hold.
   */
  encodeGlobalFineDiagnosticCopies(encoder: GPUCommandEncoder, readback: GPUBuffer): void {
    if (this.powerCoarseLevelSetSchedule) {
      encoder.copyBufferToBuffer(this.powerCoarseLevelSetSchedule.control, 0, readback, 64, 64);
    }
    if (this.fineToPowerCoarseLevelSet) {
      encoder.copyBufferToBuffer(this.fineToPowerCoarseLevelSet.control, 0, readback, 128, 32);
    }
    if (this.structuredVelocity) {
      encoder.copyBufferToBuffer(this.structuredVelocity.control, 0, readback, 160, 24);
      // Full 11-word reject carry (words 0..10): the stage-1/2 detail vec4
      // lives in words 6..9 and the workset class in word 10, which the
      // 6-word control slice above cannot carry.
      encoder.copyBufferToBuffer(this.structuredVelocity.control, 0, readback, 672, 44);
    }
    if (this.structuredBoundary) {
      encoder.copyBufferToBuffer(this.structuredBoundary.control, 0, readback, 184, 64);
    }
    if (this.structuredVelocity) {
      encoder.copyBufferToBuffer(this.structuredVelocity.rowVelocitiesA, 0, readback, 348, 16);
      encoder.copyBufferToBuffer(this.structuredVelocity.rowVelocitiesA,
        this.structuredVelocity.plan.rowCapacity * 16, readback, 364, 16);
    }
    if (this.powerCoarseLevelSet) {
      encoder.copyBufferToBuffer(this.powerCoarseLevelSet.records, 0, readback, 380, 16);
    }
    if (this.structuredVelocity && this.structuredBoundary) {
      const fractionOffset = this.structuredVelocity.plan.offsets.fractions * 4;
      const authorityBankBytes = this.structuredVelocity.plan.authorityWords * 4;
      const solidBankBytes = this.structuredVelocity.plan.slotCapacity * 4;
      encoder.copyBufferToBuffer(this.structuredVelocity.source.authority,
        fractionOffset, readback, 412, 4);
      encoder.copyBufferToBuffer(this.structuredVelocity.source.authority,
        authorityBankBytes + fractionOffset, readback, 416, 4);
      encoder.copyBufferToBuffer(this.structuredBoundary.solidNormalVelocities,
        0, readback, 420, 4);
      encoder.copyBufferToBuffer(this.structuredBoundary.solidNormalVelocities,
        solidBankBytes, readback, 424, 4);
    }
    if (this.airVelocitySupport) {
      const support = this.airVelocitySupport.source;
      encoder.copyBufferToBuffer(support.arena, support.controlOffsetWords * 4,
        readback, 428, 64);
      encoder.copyBufferToBuffer(support.recordArena, 13 * 4, readback, 492, 12);
      encoder.copyBufferToBuffer(this.airVelocitySupport.scratch, 38 * 4, readback, 504, 8);
      // Exact terminal wave counts and row/support cardinalities from the most
      // recent Section 5 transaction. This is diagnostic-only, after the
      // measured simulation, and never feeds scheduling.
      encoder.copyBufferToBuffer(this.airVelocitySupport.scratch, 32 * 4,
        readback, 576, 32);
      // Stationary-air fallback latch: unreached-patch count and the first
      // (cell<<3)|axis identity from the most recent march.
      encoder.copyBufferToBuffer(this.airVelocitySupport.scratch, 41 * 4,
        readback, 720, 8);
      // First stage-6 rejected identity/reason, copied by the failure's own
      // finalize pass before any later support transaction can reuse its slot.
      encoder.copyBufferToBuffer(this.airVelocitySupport.scratch, 51 * 4,
        readback, 728, 36);
      // Candidate/support cardinalities and the reuse flag as the rejection saw
      // them. Sited past the topology control copy at 764 rather than extending
      // the latch in place, which would have overlapped it.
      encoder.copyBufferToBuffer(this.airVelocitySupport.scratch, 60 * 4,
        readback, 780, 12);
    }
  }

  /** Power's own solve receipt is the shared row/residual staging path. */
  async readSolveDiagnostics(): Promise<boolean> { return false; }

  /** The step-snapshot ring carries no Power-owned end-of-step receipt. */
  applyStepDiagnostics(_authority: Uint32Array, _solver: Uint32Array): void {}

  readonly velocityDebug = undefined;
  readonly pressureDebug = undefined;
  readonly frontierDebug = undefined;
  readonly coarsePhiDebug = undefined;

  /** No dedicated gate: the shell's shared structured/boundary validation is
   * this lane's t=0 receipt, and it decodes those words itself. */
  async validateInitialAuthority(): Promise<undefined> { return undefined; }

  async readCoarseSurfaceTrackerReceipt(): Promise<unknown> {
    return this.engine.coarseOnlySummary?.readReceipt();
  }

  readonly forensics = {
    ownerPageForRow: (row: number) => this.readOwnerPageForPowerRow(row),
    seedChainControls: () => this.readPowerSeedChainControls(),
    descriptorCandidateFailure: (row: number) => this.readPowerDescriptorCandidateFailure(row),
    coarseFailureRow: (row: number) => this.readPowerCoarseFailureRow(row),
    airSupportFailureTopology: (firstError: number, latch: readonly number[]) =>
      this.readAirSupportFailureTopology(firstError, latch),
    spgridHierarchyCensus: () => this.readSPGridHierarchyCensus(),
    spgridTouchedDirectoryTripwire: () => this.readSPGridTouchedDirectoryTripwire(),
  };

  /** Rejection-only symmetry view over the accepted structured banks. */
  get structuredBoundarySymmetryDebug() {
    const boundary = this.structuredBoundary, structured = this.structuredVelocity?.source;
    const dynamics = this.structuredDynamics;
    if (!boundary || !structured || !dynamics) return undefined;
    return { control: boundary.control, candidateControl: boundary.candidateControl,
      epochState: this.topologyEpoch!.state, structuredControl: structured.control,
      readyEpochAudit: this.readyEpochAudit,
      readyFrontierAudit: this.readyFrontierAudit,
      readyCompactionAudit: this.readyCompactionAudit,
      candidates: boundary.candidates,
      authority: structured.authority, plan: structured.plan,
      rowGeometry: structured.rowGeometry,
      rowVelocities: structured.rowVelocities,
      selectorRows: this.powerCoarseLevelSetSchedule!.selectorRows,
      selectorOffsetWords: dynamics.selectorOffsetWords,
      selectorStride: dynamics.selectorStride,
      supportVectorOffsetWords: this.airVelocitySupport!.plan.support.supportVectorOffsetWords,
      ownerDirectoryOffsetWords: this.airVelocitySupport!.plan.support.ownerDirectoryOffsetWords,
      supportCapacity: this.airVelocitySupport!.plan.support.supportCapacity,
      supportRecordArena: this.airVelocitySupport!.recordArena,
      supportRecordOffsetWords: this.airVelocitySupport!.plan.records.recordOffsetWords,
      supportFaces: this.airVelocitySupport!.faceA,
      supportScratch: this.airVelocitySupport!.scratch,
      supportFaceAdjacency: this.airVelocitySupport!.faceAdjacency,
      supportFaceAdjacencyStride: this.airVelocitySupport!.plan.faceAdjacencyStride,
      topologyTransferAudit: dynamics.topologyTransferAudit,
      advectionSymmetryAudit: dynamics.advectionSymmetryAudit };
  }
  /** QA-only bounded readback for the compact row that caused a fail-closed
   * coarse-phi transaction. This is used only while constructing a rejected
   * t=0 solver, so it cannot enter the recurring simulation schedule. */
  async readPowerCoarseFailureRow(row: number) {
    if (!Number.isSafeInteger(row) || row < 0 || row >= this.engine.pressureCapacity.rowCapacity
      || !this.powerCoarseLevelSet || !this.powerCoarseLevelSetSchedule
      || !this.powerTopology || !this.structuredVelocity) return undefined;
    const readback = this.engine.device.createBuffer({ label: "Power coarse-phi failure-row QA", size: 160,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.engine.device.createCommandEncoder({ label: "Read power coarse-phi failure row" });
    encoder.copyBufferToBuffer(this.engine.leafHeaders, row * 48, readback, 0, 48);
    encoder.copyBufferToBuffer(this.powerTopology.metrics, row * 16, readback, 48, 16);
    encoder.copyBufferToBuffer(this.powerCoarseLevelSet.records, row * 16, readback, 64, 16);
    encoder.copyBufferToBuffer(this.powerCoarseLevelSetSchedule.diagnosticRowStatus,
      row * 32, readback, 80, 32);
    encoder.copyBufferToBuffer(this.powerCoarseLevelSetSchedule.diagnosticCandidateSampleDirectory,
      32 + row * 32, readback, 112, 32);
    encoder.copyBufferToBuffer(this.structuredVelocity.source.rowGeometry,
      row * 16, readback, 144, 16);
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange().slice(0);
      const words = new Uint32Array(bytes), floats = new Float32Array(bytes);
      return {
        row,
        header: { cell: words[0], entryStart: words[1], entryCount: words[2], size: words[3],
          diagonal: floats[4], rhs: floats[5], gradient: Array.from(floats.slice(8, 12)) },
        metric: { topologyCode: words[12], transformAndFlags: words[13], volume: floats[14] },
        coarsePhi: { phi: floats[16], minimumPhi: floats[17], maximumPhi: floats[18], flags: words[19] },
        rowStatus: { flags: words[20], advected: words[21], uniform: words[22],
          transition: words[23], corrected: words[24], interface: words[25],
          physicalVolume: floats[26], pad: words[27] },
        candidate: { cellPlusOne: words[28], size: words[29], phi: floats[30],
          minimumPhi: floats[31], maximumPhi: floats[32], flags: words[33],
          row: words[34], physicalVolume: floats[35] },
        rowGeometry: { cell: words[36], size: words[37], page: words[38], local: words[39] },
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  async readPowerDescriptorCandidateFailure(row: number) {
    if (!this.powerDescriptor || !Number.isSafeInteger(row) || row < 0
      || row >= this.engine.pressureCapacity.rowCapacity) return undefined;
    const headerReadback = this.engine.device.createBuffer({
      label: "Rejected power descriptor candidate leaf header",
      size: 48 + this.engine.ownerPages.arena.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({
      label: "Read rejected power descriptor candidate leaf header",
    });
    encoder.copyBufferToBuffer(this.engine.candidateLeafHeaders, row * 48, headerReadback, 0, 48);
    encoder.copyBufferToBuffer(
      this.engine.ownerPages.arena, 0, headerReadback, 48, this.engine.ownerPages.arena.size,
    );
    this.engine.device.queue.submit([encoder.finish()]);
    const [descriptor, rowRecord] = await Promise.all([
      this.powerDescriptor.readCandidateFailure(row),
      this.readPowerCoarseFailureRow(row),
      headerReadback.mapAsync(GPUMapMode.READ),
    ]);
    try {
      const bytes = headerReadback.getMappedRange().slice(0);
      const words = new Uint32Array(bytes), floats = new Float32Array(bytes);
      const owner = words.subarray(12);
      const cell = words[0] ?? 0;
      const q = [
        cell % this.engine.dims.nx,
        Math.floor(cell / this.engine.dims.nx) % this.engine.dims.ny,
        Math.floor(cell / (this.engine.dims.nx * this.engine.dims.ny)),
      ];
      const brickDimensions = [
        Math.ceil(this.engine.dims.nx / 8),
        Math.ceil(this.engine.dims.ny / 8),
        Math.ceil(this.engine.dims.nz / 8),
      ];
      const brick = q.map((value) => Math.floor(value / 8));
      const logical = brick[0] + brick[1] * brickDimensions[0]
        + brick[2] * brickDimensions[0] * brickDimensions[1];
      const capacity = owner[3] ?? 0;
      const logicalCount = owner[4] ?? 0;
      const activeTable = (owner[10] ?? 0) >>> 31;
      const local = q.map((value) => value % 8);
      const inspectOwnerTable = (table: number) => {
        const directoryOffset = (owner[5] ?? 0) + 3 * capacity + table * logicalCount;
        const encodedPage = owner[directoryOffset + logical] ?? 0;
        const payloadOffset = (owner[6] ?? 0) + table * capacity * 512;
        const payloadWord = encodedPage > 0 && encodedPage !== 0xffff_ffff
          ? owner[payloadOffset + (encodedPage - 1) * 512
            + local[0] + local[1] * 8 + local[2] * 64]
          : undefined;
        return { table, directoryOffset, encodedPage, payloadOffset, payloadWord };
      };
      return {
        descriptor,
        candidateHeader: {
          cell: words[0], entryStart: words[1], entryCount: words[2], size: words[3],
          diagonal: floats[4], rhs: floats[5], gradient: Array.from(floats.slice(8, 12)),
        },
        ownerControl: Array.from(owner.slice(0, 16)),
        candidateOwnerPages: {
          q, logical, active: inspectOwnerTable(activeTable),
          inactive: inspectOwnerTable(1 - activeTable),
        },
        row: rowRecord,
      };
    } finally {
      if (headerReadback.mapState === "mapped") headerReadback.unmap();
      headerReadback.destroy();
    }
  }
  /** Failure-only CPU mirror of Section 5's 18-neighbor descriptor audit.
   * The support transaction deliberately publishes a zero count on failure,
   * but its scattered identity record remains available at the stage-6 item
   * index. Reading that record and the immutable accepted owner pages here
   * identifies the grading invariant that rejected the leaf without adding a
   * diagnostic binding to the production shader. */
  private async readAirSupportFailureTopology(firstError: number, latch?: readonly number[]) {
    const support = this.airVelocitySupport;
    const stage = firstError >>> 24;
    const item = firstError & 0x00ff_ffff;
    if (!support || stage !== 6 || item >= support.plan.records.capacity) return undefined;
    const recordWord = support.plan.records.recordOffsetWords + item * 8;
    const recordByte = recordWord * 4;
    if (recordByte + 32 > support.recordArena.size) return undefined;
    const ownerArena = this.engine.ownerPages.arena;
    const readback = this.engine.device.createBuffer({
      label: "Section 5 failed support topology readback",
      size: 32 + ownerArena.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({
      label: "Read Section 5 failed support topology",
    });
    encoder.copyBufferToBuffer(support.recordArena, recordByte, readback, 0, 32);
    encoder.copyBufferToBuffer(ownerArena, 0, readback, 32, ownerArena.size);
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const latched = latch?.length === 9 && latch[8] === firstError ? latch : undefined;
      const record = latched ? [latched[0], latched[1], latched[2], latched[3],
        latched[4], latched[7], latched[5], latched[6]] : Array.from(words.slice(0, 8));
      const origin = record.slice(0, 3) as [number, number, number];
      const size = record[3] ?? 0;
      const validIdentity = size >= 1 && size <= this.engine.maxLeafSize
        && (size & (size - 1)) === 0
        && origin.every((value, axis) => value + size <= [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz][axis]);
      if (!validIdentity) return { stage, item, record, invalidIdentity: true };
      const ownerWords = words.subarray(8);
      let finer = false;
      let coarser = false;
      let invalidOwner = false;
      let ratioViolation = false;
      const dimensions = [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz] as const;
      const neighbors = OCTREE_POWER_NEIGHBOR_DIRECTIONS.map((direction, directionIndex) => {
        const coordinate = (axis: 0 | 1 | 2) => direction[axis] < 0
          ? origin[axis] - 1
          : direction[axis] > 0 ? origin[axis] + size : origin[axis] + Math.floor(size / 2);
        const probe: [number, number, number] = [coordinate(0), coordinate(1), coordinate(2)];
        const boundary = probe.some((value, axis) => value < 0 || value >= dimensions[axis]);
        if (boundary) return { directionIndex, direction, probe, boundary: true, size };
        const owner = lookupOctreeOwnerPage(ownerWords, this.engine.ownerPages.plan, probe,
          this.engine.maxLeafSize as OctreeOwnerLeafSize);
        const invalid = (owner.status & OCTREE_OWNER_PAGE_LOOKUP_STATUS.invalid) !== 0;
        const badRatio = owner.size * 2 < size || owner.size > size * 2;
        finer ||= owner.size < size;
        coarser ||= owner.size > size;
        invalidOwner ||= invalid;
        ratioViolation ||= badRatio;
        return { directionIndex, direction, probe, boundary: false,
          owner: { origin: owner.origin, size: owner.size, missing: owner.missing,
            status: owner.status }, invalid, ratioViolation: badRatio };
      });
      const reasons = [
        invalidOwner ? "invalid-owner" : undefined,
        ratioViolation ? "ratio-over-2:1" : undefined,
        finer && coarser ? "mixed-finer-coarser" : undefined,
      ].filter((reason): reason is string => reason !== undefined);
      const durableReasonCode = record[4] ?? 0;
      const durableDetail = record[6] ?? 0;
      const durableFailure = durableReasonCode === 1 ? {
        reason: "invalid-owner", directionIndex: durableDetail & 31,
        ownerSize: (durableDetail >>> 8) & 63, ownerStatus: (durableDetail >>> 16) & 0xffff,
      } : durableReasonCode === 2 ? {
        reason: "ratio-over-2:1", directionIndex: durableDetail & 31,
        ownerSize: (durableDetail >>> 8) & 63, anchorSize: (durableDetail >>> 16) & 63,
      } : durableReasonCode === 3 ? {
        reason: "mixed-finer-coarser", firstFinerDirectionIndex: durableDetail & 31,
        firstCoarserDirectionIndex: (durableDetail >>> 5) & 31,
      } : undefined;
      return { stage, item, record, identity: { origin, size },
        recordCase: record[4], recordFlagsTransform: record[5], recordLayer: record[6],
        recordGeneration: record[7], durableFailure, finer, coarser, reasons, neighbors };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  /** Post-submit diagnostic census; never participates in pressure scheduling. */
  readSPGridHierarchyCensus() {
    return this.firstOrderVCycle?.readHierarchyCensus();
  }
  /** Terminal-only proof that the compact directory differential executed. */
  readSPGridTouchedDirectoryTripwire() {
    return this.firstOrderVCycle?.readTouchedDirectoryTripwireDiagnostics();
  }
  /** Failure-only cross-check of one published Power row against its owner page. */
  async readOwnerPageForPowerRow(row: number): Promise<Record<string, unknown> | undefined> {
    const structured = this.structuredVelocity?.source;
    if (!structured || !Number.isSafeInteger(row) || row < 0
      || row >= structured.plan.rowCapacity) return undefined;
    const arena = this.engine.ownerPages.arena;
    const readback = this.engine.device.createBuffer({
      label: "Octree Power-row owner-page failure readback",
      size: arena.size + 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({
      label: "Read Octree Power-row owner-page failure",
    });
    encoder.copyBufferToBuffer(structured.rowGeometry, row * 16, readback, 0, 16);
    encoder.copyBufferToBuffer(arena, 0, readback, 16, arena.size);
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const geometry = Array.from(words.slice(0, 4));
      const owner = words.subarray(4);
      const cell = geometry[0] ?? 0;
      if ((geometry[1] ?? 0) === 0) {
        return { row, geometry, ownerControl: Array.from(owner.slice(0, 16)) };
      }
      const q = [
        cell % this.engine.dims.nx,
        Math.floor(cell / this.engine.dims.nx) % this.engine.dims.ny,
        Math.floor(cell / (this.engine.dims.nx * this.engine.dims.ny)),
      ];
      const brickDimensions = [
        Math.ceil(this.engine.dims.nx / 8),
        Math.ceil(this.engine.dims.ny / 8),
        Math.ceil(this.engine.dims.nz / 8),
      ];
      const brick = q.map((value) => Math.floor(value / 8));
      const logical = brick[0] + brick[1] * brickDimensions[0]
        + brick[2] * brickDimensions[0] * brickDimensions[1];
      const capacity = owner[3] ?? 0;
      const resident = Math.min(owner[1] ?? 0, capacity);
      let low = 0, high = resident;
      while (low < high) {
        const middle = low + Math.floor((high - low) / 2);
        if ((owner[16 + middle] ?? 0) < logical + 1) low = middle + 1;
        else high = middle;
      }
      const record = low < resident && owner[16 + low] === logical + 1 ? low : -1;
      const encodedPage = record >= 0 ? owner[(owner[5] ?? 0) + record] : 0;
      const local = q.map((value) => value % 8);
      const payloadWord = encodedPage && encodedPage !== 0xffff_ffff
        ? owner[(owner[6] ?? 0) + (encodedPage - 1) * 512
          + local[0] + local[1] * 8 + local[2] * 64]
        : undefined;
      return {
        row, geometry, q, logical, record, encodedPage, payloadWord,
        ownerControl: Array.from(owner.slice(0, 16)),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  /** Failure-only structured publication headers. */
  async readPowerSeedChainControls(): Promise<Record<string, readonly number[]> | undefined> {
    if (!this.structuredVelocity || !this.structuredBoundary) return undefined;
    const readback = this.engine.device.createBuffer({
      label: "Octree Power seed-chain control readback",
      size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({
      label: "Read Octree Power seed-chain controls",
    });
    encoder.copyBufferToBuffer(this.structuredVelocity.control, 0, readback, 0, 24);
    encoder.copyBufferToBuffer(this.structuredBoundary.control, 0, readback, 24, 40);
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return {
        structuredVelocity: Array.from(words.slice(0, 6)),
        structuredBoundary: Array.from(words.slice(6, 16)),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    // These are built by an initialization task rather than the constructor,
    // so a failure during initialization reaches this cleanup with them still
    // unassigned. Destroying them unconditionally throws a TypeError that
    // replaces the failure the caller is about to rethrow.
    this.pipelinedMGPCG?.destroy();
    this.section43HybridPreconditioner?.destroy();
    for (const buffer of Object.values(this.pipelinedMGPCGVectors ?? {})) buffer.destroy();
    this.firstOrderVCycle?.destroy();
    this.topologyEpoch?.destroy();
    this.readyEpochAudit?.destroy();
    this.readyFrontierAudit?.destroy();
    this.readyCompactionAudit?.destroy();
    this.structuredDynamics?.destroy();
    this.structuredBoundary?.destroy(); this.structuredVelocity?.destroy();
    this.structuredDivergenceRhs?.destroy();
    this.structuredSeparationMask?.destroy();
    this.fineToPowerCoarseLevelSet?.destroy();
    this.airVelocitySupport?.destroy(); this.powerCoarseLevelSetSchedule?.destroy();
    this.powerCoarseLevelSet?.destroy(); this.powerSolidVertices?.destroy();
    this.powerDescriptor?.destroy(); this.powerTopology?.destroy();
    this.powerVolumes?.destroy(); this.powerVolumeParams?.destroy();
  }
}

export const octreePowerVolumeShader = /* wgsl */ `
struct Params { cellVolume:f32,pad0:u32,pad1:u32,pad2:u32 }
struct PowerRowMetric { topologyCode:u32,transformAndFlags:u32,volume:f32,reserved:u32 }
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> metrics:array<PowerRowMetric>;
@group(0) @binding(2) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(3) var<storage,read> rowCountSource:array<u32>;
@group(0) @binding(4) var<storage,read_write> volumes:array<f32>;
@compute @workgroup_size(64) fn publishPowerVolumes(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;let count=select(0u,rowCountSource[0],arrayLength(&rowCountSource)>0u);
  if(row>=count||row>=arrayLength(&metrics)||row>=arrayLength(&headers)||row>=arrayLength(&volumes)){return;}
  let size=f32(headers[row].size);let volume=metrics[row].volume*size*size*size*params.cellVolume;
  volumes[row]=select(0.0,volume,volume==volume&&volume>0.0&&abs(volume)<=3.402823e38);
}
`;

/** Activity-only variant; the exported production shader above is never rewritten. */
export function octreePowerVolumeActivityShader(activity: GPULogicalActivityAdoptionContext): string {
  // Fine semantic phase 2 is bracketed exactly by the completion seams at
  // encodeNativePowerAssembly: structuredAdvectionBoundaryRhs ->
  // structuredVolumeCapture -> finalPressureRowAssembly. The hardware trace
  // arms those boundaries onto this pass and its immediate successor.
  const phaseIndex = OCTREE_FINE_SEMANTIC_PHASES.indexOf("structuredVolumeCapture");
  if (phaseIndex < 0) throw new Error("Structured-volume semantic phase is missing");
  const fineSemanticTicks = octreeFineEngineSplitsEnabled();
  const entry = activity.workgroup("publish-power-cell-volumes", "enter", {
    // Logical boundary tick zero closes trace phase zero. This dispatch is
    // phase two, so it begins at boundary one and closes at boundary two.
    tick: fineSemanticTicks ? `${phaseIndex - 1}u` : undefined,
    workgroupId: "activityWorkgroupId",
    numWorkgroups: "activityNumWorkgroups",
    localInvocationIndex: "activityLocalInvocationIndex",
    workgroupLaneCount: 64,
  });
  const exit = activity.workgroup("publish-power-cell-volumes", "exit", {
    tick: fineSemanticTicks ? `${phaseIndex}u` : undefined,
    workgroupId: "activityWorkgroupId",
    numWorkgroups: "activityNumWorkgroups",
    localInvocationIndex: "activityLocalInvocationIndex",
    workgroupLaneCount: 64,
  });
  if (!entry && !exit) return octreePowerVolumeShader;
  const signature = "fn publishPowerVolumes(@builtin(global_invocation_id) gid:vec3u)";
  const instrumentedSignature = "fn publishPowerVolumes(@builtin(global_invocation_id) gid:vec3u,@builtin(workgroup_id) activityWorkgroupId:vec3u,@builtin(local_invocation_index) activityLocalInvocationIndex:u32,@builtin(num_workgroups) activityNumWorkgroups:vec3u)";
  const start = octreePowerVolumeShader.indexOf(signature);
  if (start < 0) throw new Error("Power-volume activity entry point is missing");
  const bodyStart = octreePowerVolumeShader.indexOf("{", start + signature.length);
  let depth = 0, bodyEnd = -1;
  for (let index = bodyStart; index < octreePowerVolumeShader.length; index += 1) {
    if (octreePowerVolumeShader[index] === "{") depth += 1;
    else if (octreePowerVolumeShader[index] === "}" && --depth === 0) { bodyEnd = index; break; }
  }
  if (bodyStart < 0 || bodyEnd < 0) throw new Error("Power-volume activity body is malformed");
  const body = octreePowerVolumeShader.slice(bodyStart + 1, bodyEnd)
    .replace(/\breturn;/g, `${exit}return;`);
  return `${octreePowerVolumeShader.slice(0, start)}${instrumentedSignature}{${entry}${body}${exit}${octreePowerVolumeShader.slice(bodyEnd)}`;
}
