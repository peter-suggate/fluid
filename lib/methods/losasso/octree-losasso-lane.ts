import type { GPUInitializationTask } from "../../core/gpu-initialization";
import type { WebGPUFineLevelSetBrickSource } from "../../core/levelset-consumer-abi";
import { PassBroker } from "../../core/webgpu-pass-broker";
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
import type { OctreeRuntimeDials } from "../octree-shared/octree-runtime-dials";
import {
  octreeDialledIterationCap,
  octreeDialledRelativeTolerance,
} from "../octree-shared/octree-runtime-dials";
import { initialFluidBrickContainsCell, sceneHasInitialLiquidVolumes }
  from "../../core/initial-fluid";
import { planFluidFootprintFineBandBrickFloor }
  from "../octree-shared/octree-fine-band-capacity";
import { octreeLosassoLeafCeiling } from "../octree-shared/octree-leaf-sizing";
import { sceneHasUniformFinestCellCeiling }
  from "../octree-shared/octree-refinement-regions";
import { octreeSurfaceProtectionWidthCells }
  from "../octree-shared/octree-runtime-dials";
import { initialOctreeLevelSet, initialOctreeNodalLevelSet }
  from "../octree-shared/webgpu-octree";
import {
  maximumFineLevelSetJFAStride,
  WebGPUFineLevelSetRedistance,
} from "../octree-shared/webgpu-octree-fine-levelset-redistance";
import { WebGPUFineLevelSetRigidCarve }
  from "../octree-shared/webgpu-octree-fine-levelset-rigid-carve";
import {
  planFineLevelSetBandFineCells,
  WebGPUFineLevelSetTopology,
} from "../octree-shared/webgpu-octree-fine-levelset-topology";
import { WebGPUFineLevelSetVolumeCorrection }
  from "../octree-shared/webgpu-octree-fine-levelset-volume";
import type { OctreeOwnerLeafSize } from "../octree-shared/webgpu-octree-owner-pages";
import { OCTREE_PIPELINED_PCG_MAXIMUM_HARD_ITERATION_CEILING }
  from "../octree-shared/webgpu-octree-pipelined-mgpcg";
import { WebGPUOctreeRigidCouplingDiagnostics }
  from "../octree-shared/webgpu-octree-rigid-coupling-diagnostics";
import { OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS }
  from "./webgpu-octree-losasso-adaptive-phi";
import {
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_HEADER_WORDS,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_RECORDS,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_WORDS,
} from "./webgpu-octree-losasso-adaptive-velocity";
import { WebGPUOctreeLosassoCoarseBackend } from "./webgpu-octree-losasso-backend";
import {
  makeOctreeLosassoCoarsePhiSampleWGSL,
  WebGPUOctreeLosassoCoarsePhiExchange,
  type WebGPUOctreeLosassoCoarsePhiInput,
} from "./webgpu-octree-losasso-coarse-phi";
import { OCTREE_LOSASSO_COARSE_PHI_MAGIC }
  from "./webgpu-octree-losasso-coarse-phi.wgsl";
import { WebGPUOctreeLosassoConditionedOperator }
  from "./webgpu-octree-losasso-conditioned-operator";
import { WebGPUOctreeLosassoFineTransport }
  from "./webgpu-octree-losasso-fine-transport";
import { OCTREE_LOSASSO_CONTROL_WORDS } from "./octree-losasso-operator";
import { WebGPUOctreeLosassoReadyCommit }
  from "./webgpu-octree-losasso-ready-commit";
import { octreeLosassoResidentSolveEnabled }
  from "./webgpu-octree-losasso-resident-mgpcg";
import { WebGPUOctreeLosassoRowMotion } from "./webgpu-octree-losasso-row-motion";
import { LOSASSO_SURFACE_GRAPH_CONTROL_WORDS }
  from "./webgpu-octree-losasso-surface-graph";
import { OCTREE_LOSASSO_EXTENSION_WIDTH }
  from "./webgpu-octree-losasso-velocity-extension";
import { LOSASSO_PROJECTION_LANE, octreeLosassoProjectionShader }
  from "./octree-losasso-projection.wgsl";

/**
 * The Losasso 2004 octree coarse dynamics lane.
 *
 * `WebGPUOctreeProjection` owns the topology, the frontier, the fine narrow
 * band and every structure both backends share; this class owns the half that
 * is specific to the reduced axis-face authority -- its construction, its
 * advance, its receipts, and the named seams the engine calls instead of
 * re-testing `coarseDynamics.backend` at twenty-five separate sites.
 *
 * See docs/papers/losasso-2004-octree-water-smoke.txt for the method itself.
 */
export class OctreeLosassoCoarseDynamicsLane implements OctreeCoarseDynamicsLane {
  readonly backend = "losasso" as const;

  readonly wgsl = {
    projectionShader: octreeLosassoProjectionShader,
    fragments: LOSASSO_PROJECTION_LANE,
  } as const;

  /**
   * The engine is still under construction when a lane is built. Hold the
   * reference and read nothing from it until `constructAuthority`.
   */
  constructor(private readonly engine: OctreeTopologyEngine) {}

  private losassoFineTransportA?: WebGPUOctreeLosassoFineTransport;

  private losassoFineTransportB?: WebGPUOctreeLosassoFineTransport;

  /** Buffer identities captured by the currently published Losasso projection
   * groups. Coarse-phi generations rewrite their stable arenas in place, so
   * repeated refreshes within an advance must not rebuild identical groups. */
  private losassoProjectionGroupSources?: {
    readonly directory: GPUBuffer;
    readonly summary: GPUBuffer;
  };

  private losassoBackend?: WebGPUOctreeLosassoCoarseBackend;

  private losassoReadyCommit?: WebGPUOctreeLosassoReadyCommit;

  private losassoCoarsePhi?: WebGPUOctreeLosassoCoarsePhiExchange;

  private losassoRowMotion?: WebGPUOctreeLosassoRowMotion;

  private losassoConditionedOperator?: WebGPUOctreeLosassoConditionedOperator;

  private rigidCouplingDiagnostics?: WebGPUOctreeRigidCouplingDiagnostics;

  /**
   * Losasso grades by leaf SIZE, so the ceiling is the largest size the
   * exact-tiling ladder can reach for these dimensions rather than the
   * authored maximum. A leaf above it has no owner page that can hold it.
   */
  topologyLeafCeiling(
    maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    dims: { nx: number; ny: number; nz: number },
    exactTilingLeafSize: OctreeOwnerLeafSize,
  ): OctreeOwnerLeafSize {
    return octreeLosassoLeafCeiling(maximumLeafSize, dims, exactTilingLeafSize);
  }

  /**
   * The refinement ladder must span exactly the sizes the topology can hold,
   * so this lane's ladder tops out at its own ceiling. A wider ladder would
   * emit candidate leaves the owner arena cannot address.
   */
  refinementLadderLeafSize(
    _maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    _dims: { nx: number; ny: number; nz: number },
    topologyMaximumLeafSize: OctreeOwnerLeafSize,
  ): OctreeOwnerLeafSize {
    return topologyMaximumLeafSize;
  }

  /** A generalized face graph closes under ordinary strict-2:1 propagation. */
  readonly balanceRoundsUseExclusiveMixedRing = false;

  /**
   * The velocity extension must reach the whole interface band, the longest
   * backtrace out of it, and the MAC stencil around the landing cell. A band
   * wider than the compiled extension silently samples unextended air.
   */
  validateInterfaceBand(interfaceRefinementBandCells: number): void {
    if (OCTREE_LOSASSO_EXTENSION_WIDTH
      < Math.max(1, interfaceRefinementBandCells) + 3) {
      throw new RangeError("Losasso velocity extension must cover interface band + backtrace + MAC stencil");
    }
  }

  /**
   * Residency tiles are the exact-tiling rung, which is also what establishes
   * the invariant delta refinement depends on: a leaf never exceeds the tile
   * holding its origin, so that tile is the whole candidate neighbourhood.
   */
  topologyTileSize(
    _maximumLeafSize: 2 | 4 | 8 | 16 | 32,
    _effectiveLeafSize: OctreeOwnerLeafSize,
    exactTilingLeafSize: OctreeOwnerLeafSize,
    topologyMaximumLeafSize: OctreeOwnerLeafSize,
  ): number {
    const tileSize = Math.max(8, exactTilingLeafSize);
    if (topologyMaximumLeafSize > tileSize) {
      throw new Error(`Losasso leaf ceiling ${topologyMaximumLeafSize}`
        + ` exceeds the ${tileSize}-cell residency tile`);
    }
    return tileSize;
  }

  /** The adaptive band is a shell; two translated envelopes cover its growth. */
  readonly fineBandSurfaceGrowthSafety = 2;

  /**
   * The band must also survive a translation of the whole envelope, so its
   * page pool carries a floor two brick rings above the dilation the shared
   * plan already budgets. Power has no equivalent floor.
   */
  fineBandBrickFloor(input: {
    readonly brickDimensions: readonly [number, number, number];
    readonly minimumBrick: readonly [number, number, number];
    readonly maximumBrick: readonly [number, number, number];
    readonly capacityDilationBrickRings: number;
  }): number {
    return planFluidFootprintFineBandBrickFloor(input.brickDimensions,
      input.minimumBrick, input.maximumBrick, input.capacityDilationBrickRings + 2);
  }

  /**
   * A uniform finest-cell scene tracked coarse-only has no fine band to hide
   * an unconverged tail behind, so it may spend the pipelined solver's whole
   * hard ceiling rather than the shared tail policy's.
   */
  pressureHardIterationCeiling(): number | undefined {
    return this.engine.coarseOnlySurfaceTracking
      && sceneHasUniformFinestCellCeiling(this.engine.scene)
      ? OCTREE_PIPELINED_PCG_MAXIMUM_HARD_ITERATION_CEILING
      : undefined;
  }

  pressureSolverLabel(): string {
    const budget = this.losassoBackend?.solverIterationBudget
      ?? this.engine.info.pressureIterationBudget;
    return `Octree Losasso MGPCG · exact-reduction wide solve · plain first-order V-cycle · up to ${budget} iterations`;
  }

  /** Construct the reduced Losasso graph synchronously so initialization can
   * enumerate only its own shader tasks. No Power catalogue or structured
   * authority is reachable from this construction branch. */
  constructAuthority(): void {
    if (this.losassoBackend || this.engine.powerLifecycleDisposed) return;
    const rowCapacity = this.engine.pressureCapacity.rowCapacity;
    const extensionBandBrickCapacity = this.engine.globalFineSourceA?.plan.maximumResidentBricks
      ?? (this.engine.coarseOnlySurfaceTracking ? 1 : undefined);
    if (extensionBandBrickCapacity === undefined) {
      throw new Error("Losasso factor-4/8 construction requires the sparse fine-phi page plan");
    }
    // A 2:1 row can expose four subfaces on each positive axis plus the
    // corresponding negative/free-surface patches. Twenty-four per row is a
    // conservative pre-deduplication bound; using the ordinary six-face count
    // would fail exactly at graded T-junctions.
    const faceCapacity = 24 * rowCapacity;
    const largestFaceArenaBytes = 32 * faceCapacity;
    const maximumStorageBytes = Math.min(this.engine.device.limits.maxStorageBufferBindingSize,
      this.engine.device.limits.maxBufferSize);
    if (!Number.isSafeInteger(faceCapacity) || largestFaceArenaBytes > maximumStorageBytes) {
      throw new RangeError("Losasso 2:1 axis-face capacity exceeds this device's storage binding limit");
    }
    const cellSize = this.engine.scene.container.width_m / this.engine.dims.nx;
    const closedTop = this.engine.scene.container.top === "closed";
    // Factor one owns phi on the compact shared-node graph. The legacy coarse
    // exchange remains only as the factor-4/8 restriction bridge from sparse
    // fine phi; constructing it here would reintroduce a dense recurring bank.
    if (!this.engine.coarseOnlySurfaceTracking) {
      this.losassoCoarsePhi = new WebGPUOctreeLosassoCoarsePhiExchange(
        this.engine.device, rowCapacity, faceCapacity, 0,
      );
    }
    const adaptiveInitialNodalPhi = this.engine.coarseOnlySurfaceTracking
      ? initialOctreeNodalLevelSet(this.engine.scene, this.engine.dims) : undefined;
    const adaptiveInitialPhi = this.engine.coarseOnlySurfaceTracking
      ? adaptiveInitialNodalPhi ?? initialOctreeLevelSet(this.engine.scene, this.engine.dims, {
        x: this.engine.scene.container.width_m / this.engine.dims.nx,
        y: this.engine.scene.container.height_m / this.engine.dims.ny,
        z: this.engine.scene.container.depth_m / this.engine.dims.nz,
      })
      : undefined;
    // A non-additive brick scene authors an exact union of finest-grid cells.
    // Its conservative density must therefore start from those cell volumes,
    // not from a smoothed Heaviside reconstruction of the presentation phi.
    // Other adaptive scene representations retain their existing bootstrap
    // until they expose an equally exact authored volume-fraction source.
    const adaptiveInitialCellFractions = this.engine.coarseOnlySurfaceTracking
      && (this.engine.scene.fluid.initialBrickSeeds_m?.length ?? 0) > 0
      && !this.engine.scene.fluid.initialBrickSeedsAdditive
      && !sceneHasInitialLiquidVolumes(this.engine.scene)
      ? (() => {
        const values = new Float32Array(this.engine.dims.nx * this.engine.dims.ny * this.engine.dims.nz);
        for (let z = 0; z < this.engine.dims.nz; z += 1) {
          for (let y = 0; y < this.engine.dims.ny; y += 1) {
            for (let x = 0; x < this.engine.dims.nx; x += 1) {
              values[x + this.engine.dims.nx * (y + this.engine.dims.ny * z)] =
                initialFluidBrickContainsCell(this.engine.scene, x, y, z,
                  [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz]) ? 1 : 0;
            }
          }
        }
        return values;
      })()
      : undefined;
    // A full-fine authored ceiling removes the coarse levels that make the
    // ordinary adaptive dam system cheap to precondition. The 24x18x16 water
    // box first exceeds the legacy 40-iteration envelope during the advancing
    // front transient (not during construction), so compiling only that tail
    // makes a valid uniform-grid experiment fail closed at step 41. Keep the
    // same residual target and MGPCG algorithm; widen only its fail-safe tail.
    // The resident solver exits as soon as the same-step residual converges, so
    // ordinary steps and scenes without an authored unit ceiling pay no extra
    // iterations.
    const pressureHardIterationCeiling = this.engine.pressureHardIterationCeiling();
    this.losassoBackend = new WebGPUOctreeLosassoCoarseBackend({
      device: this.engine.device,
      capacities: { rows: rowCapacity, faces: faceCapacity, incidences: 2 * faceCapacity },
      topology: {
        dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
        maximumLeafSize: this.engine.topologyMaximumLeafSize,
        physicalCellSize: [cellSize, cellSize, cellSize],
        domainOrigin: [0, 0, 0],
        ownerPages: this.engine.ownerPages.plan,
      },
      density: this.engine.scene.fluid.density_kg_m3,
      extensionBandBrickCapacity,
      velocityExtensionMode: this.engine.coarseDynamics.losassoVelocityExtension,
      closedBoundaries: [true, true, true, closedTop, true, true],
      ...(this.engine.coarseOnlySurfaceTracking ? {
        adaptiveSurface: {
          candidateLeafHeaders: this.engine.candidateLeafHeaders,
          candidateOwnerArena: this.engine.ownerPages.arena,
          candidateOwnerTransaction: this.engine.ownerPages.candidateTransaction,
          frontier: this.engine.leafFrontier,
          initialPhi: adaptiveInitialPhi!,
          initialPhiLayout: adaptiveInitialNodalPhi ? "nodal-lattice" : "cell-centred",
          ...(adaptiveInitialCellFractions
            ? { initialCellFractions: adaptiveInitialCellFractions } : {}),
          redistanceBandWorld: cellSize * octreeSurfaceProtectionWidthCells(
            this.engine.interfaceBandCellsEffective, this.engine.surfaceGradingLayersEffective,
            this.engine.topologyMaximumLeafSize, 1,
          ),
          // The uniform-fine experiment has no coarse jump to shorten the
          // compact Jacobi propagation path. Its three-cell redistance shell
          // needs a wider fixed envelope during the highly folded dam front;
          // the residual gate remains the acceptance authority.
          ...(sceneHasUniformFinestCellCeiling(this.engine.scene)
            ? { redistanceIterations: 128, fullGraphRedistance: true } : {}),
          openTop: !closedTop,
        },
      } : {}),
      // The ≤4K-row coarse-only tier runs the whole warm-started MGPCG loop —
      // V-cycle, operator, exact reductions, convergence — in one resident
      // dispatch, so it executes exactly the iterations the residual gate
      // needs instead of launching the encoded envelope.
      residentSolver: octreeLosassoResidentSolveEnabled()
        && this.engine.coarseOnlySurfaceTracking && rowCapacity <= 4_096,
      solver: {
        relativeTolerance: this.engine.solveTailPolicy.relativeTolerance,
        // The <=4K-row factor-one production tier has a measured late-dam
        // requirement of 31 iterations. Keep two iterations of headroom while
        // preserving the wider hard ceiling as the fallback contract.
        maximumIterations: this.engine.coarseOnlySurfaceTracking && rowCapacity <= 4_096
          ? 33 : this.engine.solveTailPolicy.hardOuterIterationCeiling,
        hardIterationCeiling: pressureHardIterationCeiling,
        // The cooperative drain is bounded by the compact coarse row arena,
        // not by the independent fine level-set factor. Keep larger coarse
        // problems on the ordinary partial-plus-finish schedule.
        factorOneCombinedReductionDrains:
          this.engine.coarseOnlySurfaceTracking && rowCapacity <= 4_096,
      },
      rigidPressureReaction: {
        solidCells: this.engine.solidCells,
        rigidBodies: this.engine.resources.rigidBodies,
        rigidImmersedVolumes: this.engine.resources.rigidImmersedVolumes,
        rigidExchange: this.engine.resources.rigidExchange,
        rigidWorldOrigin: [
          -0.5 * this.engine.scene.container.width_m,
          0,
          -0.5 * this.engine.scene.container.depth_m,
        ],
        hydrostaticReferenceY_m:
          this.engine.scene.container.fillFraction * this.engine.scene.container.height_m,
      },
    });
    this.engine.pressureSolverControl = this.losassoBackend.solverControl
      ?? this.losassoBackend.sources.rowCount;
    this.losassoReadyCommit = new WebGPUOctreeLosassoReadyCommit(this.engine.device, {
      candidateAuthority: this.losassoBackend.candidateAuthorityControl,
      ownerCandidateTransaction: this.engine.ownerPages.candidateTransaction,
      frontier: this.engine.leafFrontier,
      candidateLeafHeaders: this.engine.candidateLeafHeaders,
      acceptedLeafHeaders: this.engine.leafHeaders,
      candidatePressure: this.engine.candidatePressure,
      pressureA: this.losassoBackend.pressureFrameViews?.pressureA ?? this.engine.pressureA,
      pressureB: this.losassoBackend.pressureFrameViews?.pressureB ?? this.engine.pressureB,
      acceptedRowCount: this.engine.compaction,
    }, rowCapacity);
    const finest = this.losassoBackend.sources.operator;
    this.losassoRowMotion = new WebGPUOctreeLosassoRowMotion(this.engine.device, {
      authority: finest.control,
      rowFaceOffsets: finest.rowFaceOffsets,
      rowFaces: finest.rowFaces,
      faces: finest.faces,
      extendedVelocity: this.losassoBackend.sources.extension.extendedVelocity,
    }, rowCapacity);
    this.engine.fineSeedAdapter?.setRowMotionSource(this.losassoRowMotion.source);
    const wide = this.losassoBackend.sources.wideSolver;
    if (!wide) throw new Error("Losasso wide solver authority was not published");
    this.losassoConditionedOperator = new WebGPUOctreeLosassoConditionedOperator(this.engine.device, {
      authority: finest.control,
      rowFaceOffsets: finest.rowFaceOffsets,
      rowFaces: finest.rowFaces,
      faces: finest.faces,
      diagonal: this.losassoBackend.pressureFrameViews?.diagonal ?? wide.diagonal,
      solverAuthority: wide.acceptedAuthority,
    }, rowCapacity);
    if (this.engine.scene.rigidBodies.length > 0) {
      this.rigidCouplingDiagnostics = new WebGPUOctreeRigidCouplingDiagnostics(this.engine.device, {
        authority: finest.control,
        leafHeaders: this.engine.leafHeaders,
        solidCells: this.engine.solidCells,
        dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      }, rowCapacity);
    }
    const adaptivePhi = this.losassoBackend.adaptivePhiSource;
    const acceptedAdaptiveGraph = this.losassoBackend.adaptiveSurfaceGraphSources?.accepted;
    if (this.engine.coarseOnlySurfaceTracking) {
      if (!adaptivePhi || !acceptedAdaptiveGraph) {
        throw new Error("Losasso factor-one adaptive scalar graph was not constructed");
      }
      this.engine.fineSeedAdapter?.setCoarsePhiSource({
        values: adaptivePhi.rowPhi,
        gradients: adaptivePhi.rowGradient,
        control: acceptedAdaptiveGraph.control,
      });
    } else {
      this.engine.fineSeedAdapter?.setCoarsePhiSource(
        this.losassoCoarsePhi!.fineSeedCoarsePhiSource());
    }
    this.refreshLosassoProjectionGroups();

    const fineA = this.engine.globalFineSourceA, fineB = this.engine.globalFineSourceB;
    const sampler = this.losassoBackend.sources.velocitySampler;
    if (fineA && fineB) {
      if (!sampler) throw new Error("Losasso factor-4 transport requires its reduced velocity sampler");
      const coarseWGSL = makeOctreeLosassoCoarsePhiSampleWGSL(9);
      this.engine.globalFineTopologyAB = new WebGPUFineLevelSetTopology(
        this.engine.device, fineA, fineB, coarseWGSL, this.engine.deferPipelineCompilation,
      );
      this.engine.globalFineTopologyBA = new WebGPUFineLevelSetTopology(
        this.engine.device, fineB, fineA, coarseWGSL, this.engine.deferPipelineCompilation,
      );
      const changedKeysOffsetWords = this.engine.globalFineTopologyAB.pageDeltaLayout.changedKeysOffsetWords;
      if (changedKeysOffsetWords !== this.engine.globalFineTopologyBA.pageDeltaLayout.changedKeysOffsetWords) {
        throw new Error("Losasso fine topology A/B page-delta layouts disagree");
      }
      this.engine.device.queue.writeBuffer(this.engine.params, 36, new Uint32Array([changedKeysOffsetWords]));
      const redistanceOptions = (source: WebGPUFineLevelSetBrickSource) => ({
        deferPipelineCompilation: this.engine.deferPipelineCompilation,
        axisPermutationInvariantSeeds: true,
        maximumRequiredJfaStride: maximumFineLevelSetJFAStride(
          planFineLevelSetBandFineCells(this.engine.fineLevelSetBandCells,
            source.plan.fineFactor).redistanceBandFineCells),
      });
      this.engine.globalFineRedistanceA = new WebGPUFineLevelSetRedistance(
        this.engine.device, fineA, this.engine.globalFineTopologyBA, redistanceOptions(fineA),
      );
      this.engine.globalFineRedistanceB = new WebGPUFineLevelSetRedistance(
        this.engine.device, fineB, this.engine.globalFineTopologyAB, redistanceOptions(fineB),
      );
      this.losassoFineTransportA = new WebGPUOctreeLosassoFineTransport(
        this.engine.device, fineA, sampler);
      this.losassoFineTransportB = new WebGPUOctreeLosassoFineTransport(
        this.engine.device, fineB, sampler);
      if (this.engine.scene.rigidBodies.length > 0) {
        const carveOptions = {
          bodyCount: this.engine.scene.rigidBodies.length,
          rigidWorldOrigin: [-0.5 * this.engine.scene.container.width_m, 0,
            -0.5 * this.engine.scene.container.depth_m] as const,
        };
        this.engine.globalFineRigidCarveA = new WebGPUFineLevelSetRigidCarve(
          this.engine.device, fineA, this.engine.resources.rigidBodies, carveOptions);
        this.engine.globalFineRigidCarveB = new WebGPUFineLevelSetRigidCarve(
          this.engine.device, fineB, this.engine.resources.rigidBodies, carveOptions);
      }
      const coarseInput = this.losassoCoarsePhiInput();
      const coarseVolume = this.losassoCoarsePhi!.volumeCoarseSource(coarseInput);
      const rigidVolumeTarget = this.engine.scene.rigidBodies.length > 0 ? {
        immersedVolumes: this.engine.resources.rigidImmersedVolumes,
        bodyCount: this.engine.scene.rigidBodies.length,
      } : undefined;
      this.engine.globalFineVolumeA = new WebGPUFineLevelSetVolumeCorrection(
        this.engine.device, fineA, coarseVolume, undefined, this.engine.deferPipelineCompilation,
        "moving-pages", rigidVolumeTarget,
      );
      this.engine.globalFineVolumeB = new WebGPUFineLevelSetVolumeCorrection(
        this.engine.device, fineB, coarseVolume, this.engine.globalFineVolumeA.control,
        this.engine.deferPipelineCompilation, "moving-pages", rigidVolumeTarget,
      );
    }
    const coarseAllocated = this.losassoBackend.allocatedBytes
      + this.losassoReadyCommit.allocatedBytes + (this.losassoCoarsePhi?.plan.allocatedBytes ?? 0)
      + this.losassoConditionedOperator.allocatedBytes + this.losassoRowMotion.plan.allocatedBytes
      + (this.engine.coarseOnlySummary?.plan.allocatedBytes ?? 0)
      + (this.rigidCouplingDiagnostics?.allocatedBytes ?? 0);
    const allocated = coarseAllocated
      + (this.engine.globalFineTopologyAB?.allocatedBytes ?? 0)
      + (this.engine.globalFineTopologyBA?.allocatedBytes ?? 0)
      + (this.engine.globalFineRedistanceA?.allocatedBytes ?? 0)
      + (this.engine.globalFineRedistanceB?.allocatedBytes ?? 0)
      + (this.losassoFineTransportA?.plan.allocatedBytes ?? 0)
      + (this.losassoFineTransportB?.plan.allocatedBytes ?? 0)
      + (this.engine.globalFineRigidCarveA?.allocatedBytes ?? 0)
      + (this.engine.globalFineRigidCarveB?.allocatedBytes ?? 0)
      + (this.engine.globalFineVolumeA?.allocatedBytes ?? 0)
      + (this.engine.globalFineVolumeB?.allocatedBytes ?? 0);
    this.engine.info.allocatedBytes += allocated;
    this.engine.info.globalFineLevelSetAllocatedBytes += allocated - coarseAllocated;
    this.engine.info.powerDiagramReady = false;
    this.engine.info.powerDiagramAuthoritative = false;
    this.engine.workAccounting.setAuthorityBytes("losasso", coarseAllocated);
    this.engine.workAccounting.setAuthorityBytes("fine-level-set", Math.max(0,
      allocated - coarseAllocated));
    this.engine.workAccounting.sealAllocationInventory();
  }

  private losassoCoarsePhiInput(): WebGPUOctreeLosassoCoarsePhiInput {
    const backend = this.losassoBackend;
    if (!backend) throw new Error("Losasso coarse authority was not constructed");
    return {
      leafHeaders: this.engine.leafHeaders,
      coarseControl: backend.sources.operator.control,
      faces: backend.sources.projection.faces,
      faceDispatch: backend.sources.projection.faceDispatch,
      faceGeometry: backend.sources.dynamics.faceGeometry,
      solidCells: this.engine.solidCells,
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz],
      maximumLeafSize: this.engine.topologyMaximumLeafSize,
      cellSize: this.engine.scene.container.width_m / this.engine.dims.nx,
    };
  }


  private refreshLosassoProjectionGroups(): void {
    const directory = this.engine.coarseOnlySurfaceTracking
      ? this.losassoBackend?.adaptivePhiSource?.topologyEvidence
      : this.losassoCoarsePhi?.source.arena;
    if (!directory) return;
    const summary = this.engine.globalFineSummaries?.directory
      ?? this.engine.coarseOnlySummary?.directory ?? this.engine.unpublishedFineSummaryDirectory;
    if (this.losassoProjectionGroupSources?.directory === directory
      && this.losassoProjectionGroupSources.summary === summary) return;
    this.engine.groups = {
      ab: this.engine.createProjectionGroup(this.engine.pressureA, this.engine.pressureB, directory),
      ba: this.engine.createProjectionGroup(this.engine.pressureB, this.engine.pressureA, directory),
    };
    const pressureViews = this.losassoBackend?.pressureFrameViews;
    this.engine.candidateRowGroups = {
      fromA: this.engine.createProjectionGroup(pressureViews?.pressureA ?? this.engine.pressureA,
        this.engine.candidatePressure, directory,
        this.engine.candidateLeafHeaders),
      fromB: this.engine.createProjectionGroup(pressureViews?.pressureB ?? this.engine.pressureB,
        this.engine.candidatePressure, directory,
        this.engine.candidateLeafHeaders),
    };
    this.engine.fineSummarySizingGroup = this.engine.createProjectionGroup(summary, this.engine.pressureB, directory);
    this.engine.topologyDecisionGroup = this.engine.createProjectionGroup(
      summary, this.engine.topologyResidency.topologyTileStateBuffer, directory,
    );
    this.losassoProjectionGroupSources = { directory, summary };
  }

  initializationTasks(): GPUInitializationTask[] {
    const tasks: GPUInitializationTask[] = [];
    const reducedTasks = [
      ...(this.losassoBackend ? [{ label: "Compile complete Losasso coarse backend",
        run: () => this.losassoBackend!.initialize() }] : []),
      ...(this.losassoReadyCommit?.initializationTasks ?? []),
      ...(this.losassoCoarsePhi?.initializationTasks ?? []),
      ...(this.losassoRowMotion?.initializationTasks ?? []),
      ...(this.losassoConditionedOperator?.initializationTasks ?? []),
      ...(this.rigidCouplingDiagnostics?.initializationTasks ?? []),
      ...(this.engine.coarseOnlySummary?.initializationTasks() ?? []),
    ];
    reducedTasks.forEach((task, index) => tasks.push({
      id: `octree.losasso.pipeline.${index}`,
      phase: "solver-pipelines",
      label: task.label,
      run: (signal) => task.run(signal),
    }));
    if (this.engine.globalFineSourceA && this.engine.globalFineSourceB) {
      tasks.push({
        id: "octree.losasso.fine-topology",
        phase: "solver-pipelines",
        label: "Compile Losasso factor-4 fine topology",
        run: async () => {
          await this.engine.globalFineTopologyAB!.initializePipelines();
          await this.engine.globalFineTopologyBA!.initializePipelines();
        },
      }, {
        id: "octree.losasso.fine-redistance",
        phase: "solver-pipelines",
        label: "Compile Losasso factor-4 redistance",
        run: async () => {
          await this.engine.globalFineRedistanceA!.initializePipelines();
          await this.engine.globalFineRedistanceB!.initializePipelines();
        },
      }, {
        id: "octree.losasso.fine-transport",
        phase: "solver-pipelines",
        label: "Compile Losasso factor-4 direct face transport",
        run: async () => {
          await this.losassoFineTransportA!.initializePipelines();
          await this.losassoFineTransportB!.initializePipelines();
        },
      }, {
        id: "octree.losasso.fine-rigid-carve",
        phase: "solver-pipelines",
        label: "Compile Losasso fine rigid-body carve",
        run: async () => {
          await this.engine.globalFineRigidCarveA?.initializePipelines();
          await this.engine.globalFineRigidCarveB?.initializePipelines();
        },
      }, {
        id: "octree.losasso.fine-volume",
        phase: "solver-pipelines",
        label: "Compile Losasso factor-4 volume bridge",
        run: async () => {
          await this.engine.globalFineVolumeA!.initializePipelines();
          await this.engine.globalFineVolumeB!.initializePipelines();
        },
      });
    }
    return tasks;
  }

  encodeInactiveCandidate(encoder: GPUCommandEncoder): void {
    const backend = this.losassoBackend;
    if (!backend || this.engine.candidatePowerGeneration === 0) {
      throw new Error("Inactive topology candidate requires the reduced Losasso authority");
    }
    const broker = new PassBroker(encoder);
    backend.encodeCandidatePublication(broker, {
      leafHeaders: this.engine.candidateLeafHeaders,
      frontier: this.engine.leafFrontier,
      ownerArena: this.engine.ownerPages.arena,
      ownerCandidateTransaction: this.engine.ownerPages.candidateTransaction,
      solidCells: this.engine.solidCells,
      rigidBodies: this.engine.resources.rigidBodies,
    });
    broker.fence("inactive Losasso axis-face candidate published");
  }

  encodeReadyTopologyFlip(encoder: GPUCommandEncoder): void {
    if (this.engine.candidatePowerGeneration === 0 && this.engine.topologyReusePending) {
      // Retain the accepted graph, phi banks, face authority and hierarchy.
      // The adaptive advance has already paired scalar/velocity clocks, so
      // this boundary needs no graph copy, filter, hierarchy refresh or
      // extension-topology remap.
      this.engine.topologyReusePending = false;
      this.engine.info.topologyReused = true;
      return;
    }
    if (!this.losassoReadyCommit || !this.losassoBackend
      || this.engine.candidatePowerGeneration === 0) {
      throw new Error("Ready topology flip requires a complete inactive Losasso candidate");
    }
    const broker = new PassBroker(encoder);
    // The row/pressure and reduced-operator copies validate the exact same
    // candidate transaction immediately before the owner selector flips.
    this.losassoReadyCommit.encodeReadyCommit(broker);
    this.losassoBackend.encodeReadyCommit(broker, {
      frontier: this.engine.leafFrontier,
      ownerCandidateTransaction: this.engine.ownerPages.candidateTransaction,
    });
    this.engine.ownerPages.encodeReadyCommit(broker);
    const currentFine = this.engine.globalFineCurrentIsA
      ? this.engine.globalFineSourceA : this.engine.globalFineSourceB;
    if (this.engine.globalFineBootstrapped && currentFine && this.losassoCoarsePhi) {
      // Losasso et al. 2004 Section 4, equations 5-6, requires the pressure
      // operator to use the same face boundary state as divergence. A ready
      // topology commit replaces the accepted face records, so reapply the
      // current fine-phi ghost distances (including unilateral closed-wall
      // separation) before building the hierarchy or solving this epoch.
      // See docs/papers/losasso-2004-octree-water-smoke.txt, Sections 4.1-4.2.
      this.losassoCoarsePhi.encode(broker, currentFine,
        this.losassoCoarsePhiInput());
      this.losassoConditionedOperator?.encodeAfterGhostDistances(broker);
    } else if (this.engine.coarseOnlySurfaceTracking) {
      // The backend commit has already re-derived row evidence and face
      // ghosts after the matching accepted face bank. A topology handoff
      // must not transport, redistance, correct, or flip phi a second time.
      if (!this.losassoBackend.adaptivePhiSource) {
        throw new Error("Ready factor-one topology requires adaptive phi");
      }
      // The GPU ready gate may still reject this candidate. Do not predict
      // its scalar clock on the host: the step-coherent adaptive receipt is
      // the sole authority that admits a renderer generation.
      this.losassoConditionedOperator?.encodeAfterGhostDistances(broker);
    }
    this.losassoBackend.encodeHierarchyRefresh(broker, this.engine.leafHeaders);
    this.refreshLosassoProjectionGroups();
    // Candidate publication migrates the lagged wet-face field by geometric
    // identity. Reconstruct row motion immediately so encodeSurface never
    // observes row indices from the retired epoch.
    this.losassoRowMotion?.encode(broker);
    broker.fence("accepted Losasso row and owner epoch published");
    this.engine.activePowerGeneration = this.engine.candidatePowerGeneration;
    this.engine.candidatePowerGeneration = 0;
    this.engine.info.topologyReused = false;
  }

  /**
   * This lane rebuilds its candidate graph on a host-scheduled cadence, so the
   * engine's skip path is live for it: unlike Power it can legitimately reuse
   * an accepted epoch across several advances.
   */
  readonly topologyCadenceIsEveryAdvance = false;

  runtimeDialsApplicable(): boolean { return this.losassoBackend !== undefined; }

  applyRuntimeDials(dials: OctreeRuntimeDials): void {
    const backend = this.losassoBackend;
    if (!backend) return;
    backend.applySolveTuning({
      relativeTolerance: octreeDialledRelativeTolerance(
        this.engine.solveTailPolicy.relativeTolerance, dials),
      maximumIterations: octreeDialledIterationCap(
        backend.solverIterationBudget ?? this.engine.solveTailPolicy.hardOuterIterationCeiling,
        dials),
      bottomSweeps: dials.vcycleBottomSweeps,
      smoothingSweeps: dials.vcycleSmoothingSweeps,
    });
    backend.setVelocityExtensionSweeps(dials.velocityExtensionSweeps);
  }

  encodeAdvance(encoder: GPUCommandEncoder, ctx: OctreeAdvanceContext): GPUCommandEncoder {
    const options = ctx, scope = ctx.scope;
    const backend = this.losassoBackend;
    if (!backend || this.engine.activePowerGeneration === 0) {
      throw new Error("Losasso pressure step requires an accepted compact authority");
    }
    const solveBudget = backend.solverIterationBudget
      ?? this.engine.solveTailPolicy.encodedOuterIterations;
    this.engine.workAccounting.beginSubstep();
    this.engine.info.pressureIterationBudget = solveBudget;
    this.engine.info.pressureIterationHardBudget = this.engine.pressureHardIterationCeiling();
    const step = {
      dt_s: this.engine.powerTimestep_s,
      gravity_m_s2: [
        this.engine.scene.fluid.gravity_m_s2.x,
        this.engine.scene.fluid.gravity_m_s2.y,
        this.engine.scene.fluid.gravity_m_s2.z,
      ] as const,
      inflow: this.engine.surfaceInflow,
    };
    let broker = new PassBroker(encoder);
    if (this.engine.scene.rigidBodies.length > 0 && backend.encodeRigidBoundaryRefresh(broker)) {
      const currentFine = this.engine.globalFineCurrentIsA
        ? this.engine.globalFineSourceA : this.engine.globalFineSourceB;
      if (currentFine && this.losassoCoarsePhi) {
        this.losassoCoarsePhi.encodeGhostRefresh(broker, currentFine,
          this.losassoCoarsePhiInput());
      }
      this.losassoConditionedOperator?.encodeAfterGhostDistances(broker);
      backend.encodeHierarchyCoefficientRefresh(broker);
    }
    this.rigidCouplingDiagnostics?.encode(broker);
    backend.encodeAdvection(broker, step);
    backend.encodeForcesAndDivergence(broker, step);
    broker.fence("Losasso first-order axis-face RHS published");
    if (options?.productionBoundary) {
      encoder = options.productionBoundary("structuredAdvectionBoundaryRhs", encoder);
      broker = new PassBroker(encoder);
    }
    const initialInA = !this.engine.latestPressureInA;
    const pressureIn = initialInA ? this.engine.pressureA : this.engine.pressureB;
    const pressureOut = initialInA ? this.engine.pressureB : this.engine.pressureA;
    backend.encodeSolve(broker, { pressureSeed: pressureIn, pressureOut });
    this.engine.latestPressureInA = !initialInA;
    broker.fence("Losasso wide exact-reduction pressure solve complete");
    if (options?.productionBoundary) {
      encoder = options.productionBoundary("mgpcgSolve", encoder);
      broker = new PassBroker(encoder);
    }
    // Static oracle bodies do not integrate, but their pressure reaction is
    // still observable evidence (and validates the same K^T p used by dynamic
    // bodies). The reaction pass therefore sees every authored body.
    backend.encodeProjection(broker, pressureOut, step,
      Math.min(12, this.engine.scene.rigidBodies.length));
    broker.fence("Losasso projected wet axis faces published");
    if (this.engine.coarseOnlySurfaceTracking) {
      // Projection is the accepted face seed for the next characteristic
      // trace. Reconstruct the graph-owned accepted/predictor node fields now;
      // the backend also retains the compact face seed for topology migration.
      const advanceSerial = this.engine.powerTimestep_s > 0
        ? this.engine.powerAdvancingPressureSteps + 1 : 0;
      backend.encodeExtension(broker, advanceSerial, this.engine.activePowerGeneration);
      this.losassoRowMotion?.encode(broker);
      broker.fence("Losasso adaptive post-projection nodal velocity published");
    }
    if (scope === "power-operator-only") return encoder;
    if (options?.productionBoundary) {
      encoder = options.productionBoundary("structuredProjection", encoder);
    }
    encoder = this.engine.encodePendingFineSettlement(encoder, options?.productionBoundary);
    this.engine.encodeOverlayMaterialization(encoder, this.engine.latestPressureInA);
    if (this.engine.powerTimestep_s > 0) this.engine.powerAdvancingPressureSteps += 1;
    if (options?.productionBoundary) {
      encoder = options.productionBoundary("structuredProjectionTail", encoder);
    }
    return encoder;
  }

  /**
   * The reduced backend keeps its own frame-local pressure banks, so it names
   * the authoritative one; the engine's A/B latch only describes the shared
   * banks this lane does not solve into.
   */
  overlayPressureAuthorityIsA(pressureInA: boolean): boolean {
    return this.losassoBackend?.pressureAuthorityIsA ?? pressureInA;
  }

  /**
   * Nothing to bootstrap. The dense coarse tracker consumes the backend that
   * owns the surface step; this lane publishes its surface below, after its
   * staggered velocity source is ready. Advancing the tracker here as part of
   * the legacy Power bootstrap transported the same surface twice on step zero.
   */
  encodeSurfaceCoarseBootstrap(_broker: PassBroker): boolean { return false; }

  encodeCoarseOnlySurfaceAdvance(
    preparationBroker: PassBroker,
    dt_s: number,
    inflow: SurfaceInflowState | undefined,
  ): GPUCommandEncoder | undefined {
    if (!this.engine.coarseOnlySurfaceTracking) return undefined;
    const backend = this.losassoBackend;
    if (this.engine.pendingSurfaceReferenceVolume_m3 > 0) {
      if (!backend?.addAdaptiveSurfaceReferenceVolume(
        this.engine.pendingSurfaceReferenceVolume_m3)) {
        throw new Error("Losasso factor-one adaptive volume authority is unavailable");
      }
      this.engine.pendingSurfaceReferenceVolume_m3 = 0;
    }
    if (!backend?.encodeAdaptiveSurfaceAdvance(preparationBroker, dt_s, inflow)) {
      throw new Error("Losasso factor-one adaptive surface authority is unavailable");
    }
    // The GPU transaction may retain its prior scalar bank. Do not predict
    // its generation on the host: the coherent step receipt advances this
    // mirror only after graph, phi, velocity and renderer all commit.
    this.losassoConditionedOperator?.encodeAfterGhostDistances(preparationBroker);
    backend.encodeHierarchyCoefficientRefresh(preparationBroker);
    this.refreshLosassoProjectionGroups();
    preparationBroker.fence("Losasso adaptive nodal phi and pressure ghosts published");
    return preparationBroker.commandEncoder();
  }

  /**
   * This lane allocates a fine band whenever it is not factor one, and the
   * factor-one path returned above. Reaching here means the configuration has
   * neither authority, which the engine reports as an incomplete pipeline
   * rather than silently advancing nothing.
   */
  encodeCoarseOnlyFallbackAdvance(
    _broker: PassBroker,
    _dt_s: number,
    _coarseBootstrappedThisStep: boolean,
  ): GPUCommandEncoder | undefined {
    return undefined;
  }

  fineTopologyCoarseEntry(binding: number): GPUBindGroupEntry {
    return this.losassoCoarsePhi!.fineTopologyEntry(binding);
  }

  fineTransportStage(currentIsA: boolean): OctreeFineLevelSetTransportStage | undefined {
    return currentIsA ? this.losassoFineTransportA : this.losassoFineTransportB;
  }

  encodeFineTransport(
    broker: PassBroker,
    stage: OctreeFineLevelSetTransportStage,
    request: OctreeFineTransportRequest,
  ): PassBroker {
    // The direct face transport carries no dynamic-boundary clause; the shared
    // request declares one because the staged Power transport needs it.
    void request.dynamicBoundary;
    return (stage as WebGPUOctreeLosassoFineTransport).encode(broker, {
      timestep: request.timestep,
      ...(request.inflow ? { inflow: request.inflow } : {}),
      // S1 deliberately consumes the previous advance's settled W7
      // field. A ready topology flip may already have advanced the
      // wet-face epoch, while this lagged sampler remains physically
      // valid by geometric identity until S3e rebuilds it below.
      velocityEpoch: 0,
      boundaryPolicy: "closed-neumann",
      openTopBoundary: request.openTopBoundary,
      transportBandCells: request.transportBandCells,
      maximumBacktraceFineCells: request.maximumBacktraceFineCells,
    });
  }

  encodeCoarsePhiBeforeForces(
    broker: PassBroker,
    target: WebGPUFineLevelSetBrickSource,
    _topology: WebGPUFineLevelSetTopology,
    _dt_s: number,
  ): void {
    if (!this.losassoCoarsePhi) throw new Error("Losasso coarse-phi exchange is unavailable");
    this.losassoCoarsePhi.encodeFieldRefresh(broker, target,
      this.losassoCoarsePhiInput());
    this.losassoConditionedOperator?.encodeAfterGhostDistances(broker);
    this.losassoBackend?.encodeHierarchyCoefficientRefresh(broker);
    this.refreshLosassoProjectionGroups();
  }

  /**
   * The Section 5 air-support producer is Power machinery. This lane extends
   * its own velocity inside `encodeExtension`, so a settled generation here
   * owes no support epoch and there is nothing to require.
   */
  requireSettledSupport(): void {}

  encodeSettledSupport(
    encoder: GPUCommandEncoder,
    _dt_s: number,
    _phase: "t0" | "recurring",
  ): GPUCommandEncoder { return encoder; }

  encodeSettlementBootstrap(
    redistanceBroker: PassBroker,
    target: WebGPUFineLevelSetBrickSource,
  ): void {
    if (this.engine.globalFineBootstrapped) return;
    if (!this.losassoCoarsePhi) throw new Error("Losasso coarse-phi exchange is unavailable");
    // Bootstrap the generic coarse-volume directory before the first volume
    // correction. Later settlements already have the pre-force exchange.
    this.losassoCoarsePhi.encode(redistanceBroker, target,
      this.losassoCoarsePhiInput());
    this.losassoConditionedOperator?.encodeAfterGhostDistances(redistanceBroker);
    this.losassoBackend?.encodeHierarchyCoefficientRefresh(redistanceBroker);
    this.refreshLosassoProjectionGroups();
  }

  /**
   * Phi is geometric signed-distance authority in this lane: volume loss is
   * measured here but never repaired by a post-step scalar offset.
   */
  readonly settlementVolumePolicy = "measure" as const;

  encodeSettlementCoarseRefresh(
    redistanceBroker: PassBroker,
    target: WebGPUFineLevelSetBrickSource,
  ): void {
    if (!this.losassoCoarsePhi) throw new Error("Losasso coarse-phi exchange is unavailable");
    // Republish the final conditioned operator and hierarchy from the fully
    // redistanced band. Losasso volume telemetry does not move fine phi.
    this.losassoCoarsePhi.encodeFieldRefresh(redistanceBroker, target,
      this.losassoCoarsePhiInput());
    this.losassoConditionedOperator?.encodeAfterGhostDistances(redistanceBroker);
    this.losassoBackend?.encodeHierarchyCoefficientRefresh(redistanceBroker);
    this.refreshLosassoProjectionGroups();
  }

  encodeSettlementRestriction(
    restrictionBroker: PassBroker,
    target: WebGPUFineLevelSetBrickSource,
    topology: WebGPUFineLevelSetTopology,
  ): void {
    if (this.losassoCoarsePhi) {
      this.engine.globalFineSummaries?.encode(restrictionBroker, target, {
        buffer: topology.pageDelta,
        layout: topology.pageDeltaLayout,
      }, this.losassoCoarsePhi.summaryCoarseSource());
    }
    const backend = this.losassoBackend;
    if (!backend) throw new Error("Losasso extension-band backend is unavailable");
    backend.encodeExtensionBandPublication(restrictionBroker, target);
    const advanceSerial = this.engine.powerTimestep_s > 0
      ? this.engine.powerAdvancingPressureSteps + 1 : 0;
    backend.encodeExtension(restrictionBroker, advanceSerial,
      this.engine.activePowerGeneration);
    this.losassoRowMotion?.encode(restrictionBroker);
  }
  get debug(): OctreeLaneDebugSources {
    const adaptiveGraph = this.losassoBackend?.adaptiveSurfaceGraphSources?.accepted;
    const adaptivePhi = this.losassoBackend?.adaptivePhiSource;
    const reach = this.losassoBackend?.adaptiveVelocityExtensionReach_m;
    const frameViews = this.losassoBackend?.pressureFrameViews;
    return {
      ...(frameViews ? { pressureFrameView: this.losassoBackend!.pressureAuthorityIsA
        ? frameViews.pressureA : frameViews.pressureB } : {}),
      ...(adaptiveGraph && adaptivePhi && reach !== undefined && Number.isFinite(reach)
        ? { adaptiveVelocity: {
          control: adaptiveGraph.control,
          leaves: adaptiveGraph.leaves,
          nodalVelocity: adaptiveGraph.nodalVelocity,
          phiControl: adaptivePhi.control,
          rowPhi: adaptivePhi.rowPhi,
          extensionReach_m: reach,
        } } : {}),
      losassoAuthorityControl: this.losassoAuthorityControl,
      losassoCoarsePhiControl: this.losassoCoarsePhiControl,
      losassoExtensionControl: this.losassoExtensionControl,
      losassoAdaptiveAcceptedGraphControl: this.losassoAdaptiveAcceptedGraphControl,
      losassoAdaptiveCandidateGraphControl: this.losassoAdaptiveCandidateGraphControl,
      losassoAdaptivePhiControl: this.losassoAdaptivePhiControl,
      losassoAdaptivePhiReceipts: this.losassoAdaptivePhiReceipts,
      losassoAdaptiveVelocityReceipts: this.losassoAdaptiveVelocityReceipts,
      losassoAdaptiveRendererDirectory: this.losassoAdaptiveRendererDirectory,
      losassoCandidateAuthorityControl: this.losassoCandidateAuthorityControl,
      losassoAdaptiveMassControl: this.losassoAdaptiveMassControl,
      losassoAdaptiveMassReceipts: this.losassoAdaptiveMassReceipts,
      losassoAdaptiveCandidateMassControl: this.losassoAdaptiveCandidateMassControl,
      losassoAdaptiveCandidateMassReceipts: this.losassoAdaptiveCandidateMassReceipts,
      losassoCandidateVelocityMigrationReceipt: this.losassoCandidateVelocityMigrationReceipt,
      rigidCouplingDiagnosticBuffer: this.rigidCouplingDiagnosticBuffer,
      rigidBoundaryRefreshDiagnosticBuffer: this.rigidBoundaryRefreshDiagnosticBuffer,
    };
  }

  debugSources(): Record<string, unknown> { return { ...this.debug }; }

  get diagnosticPressureBanks() { return this.losassoBackend?.pressureFrameViews; }

  get solverSymmetryStageAuditBuffers(): OctreeLaneSymmetryStageAuditBuffers | undefined {
    return this.losassoBackend?.solverSymmetryStageAuditBuffers;
  }

  /**
   * Generic rendering and QA consumers use the shared eight-word coarse
   * directory ABI. The Losasso arena is this lane's private topology-sampling
   * hash table and must never leak through that backend-neutral source.
   */
  genericCoarseDirectory() {
    const source = this.losassoCoarsePhi?.source;
    return source
      ? { directory: source.volumeDirectory, rowCapacity: source.rowCapacity }
      : undefined;
  }

  coarseLevelSetPublication(): OctreeLaneCoarseLevelSetPublication | undefined {
    if (!this.engine.coarseOnlySurfaceTracking) return undefined;
    const legacyLosasso = this.losassoCoarsePhi;
    const adaptive = this.losassoBackend?.adaptivePhiSource;
    // Factor-one physics and rendering must select the same scalar authority.
    // The legacy coarse-phi object remains allocated during the cutover, so
    // testing it first silently paired the adaptive generation below with the
    // retired row directory. The raster gate correctly rejected that mixed
    // tuple and the browser showed an empty tank after the first step.
    const coarse = adaptive ? {
      // Output-only compatibility publication. Physics consumes nodal phi and
      // topologyEvidence; this Power-directory ABI is rebuilt one-way solely
      // for renderer/view consumers and never feeds an adaptive solve.
      directory: adaptive.rendererDirectory,
      control: adaptive.control,
      rowCapacity: adaptive.topologyEvidenceRowCapacity,
    } : legacyLosasso ? {
      directory: legacyLosasso.source.volumeDirectory,
      control: legacyLosasso.source.volumePublication,
      rowCapacity: legacyLosasso.source.rowCapacity,
    } : undefined;
    const generation = adaptive
      ? this.engine.adaptiveSurfaceGeneration
      : legacyLosasso?.coarseOnlyPublishedGeneration ?? this.engine.activePowerGeneration;
    // The adaptive renderer directory is graph-leaf indexed and carries its
    // own eight-corner nodal records. `rowGradient` is pressure-row indexed;
    // advertising it beside a larger owner-support directory gives view/QA
    // consumers two incompatible cardinalities and can overrun the buffer.
    const gradients = adaptive ? undefined : legacyLosasso?.source.rowGradient;
    if (!coarse) return undefined;
    return { ...coarse, generation, ...(gradients ? { gradients } : {}) };
  }

  summaryCoarseDebug() {
    const coarse = this.losassoCoarsePhi;
    return coarse
      ? { control: coarse.source.volumePublication, delta: coarse.source.summaryDelta }
      : undefined;
  }

  acceptsAdaptiveSurfaceGenerationReceipt(): boolean {
    return this.engine.coarseOnlySurfaceTracking;
  }

  hasCoarseSurfaceAuthority(): boolean {
    return this.engine.coarseOnlySurfaceTracking
      && this.losassoBackend?.adaptivePhiSource !== undefined;
  }

  /**
   * The renderer never sources from this lane's compact velocity authority:
   * the factor-one path publishes through the adaptive surface graph instead,
   * so releasing dense bootstrap phi on its account would strand it.
   */
  compactRendererSourceReady(): boolean { return false; }

  /** The reduced operator carries its own groups; there is no coarse pair. */
  readonly hasCoarseProjectionGroups = false;

  /** The reduced backend names its own budget once it has been constructed. */
  solverIterationBudget(): number | undefined {
    return this.losassoBackend?.solverIterationBudget;
  }

  /**
   * The production work-accounting rows below are Power sizing law: this lane
   * has no compact divergence RHS bank and no Section 6.3 coefficient bank,
   * so it contributes neither rather than substituting a same-shaped buffer.
   */
  workAccountingBuffers(): Readonly<{
    pressureRhs?: GPUBufferBinding;
    section63Coefficients?: GPUBufferBinding;
  }> {
    return {};
  }

  /**
   * The technique cell trace reads the power catalog and structured banks,
   * neither of which this lane allocates. Returning undefined suppresses the
   * overlay rather than publishing a same-named buffer with other contents.
   */
  techniqueDebugSources(): OctreeLaneTechniqueDebugSources | undefined { return undefined; }

  /** No structured boundary bank and no Section 5 air-support producer. */
  readonly structuredBoundarySymmetryDebug = undefined;

  /**
   * Preserve the long-standing failure receipt shape for callers while
   * exposing the single reduced authority that replaces the five Power
   * publication controls. Unused regions remain zero-initialized.
   */
  captureFrontierFailureAuthorityControls(capture: OctreeLaneFrontierFailureCapture): void {
    const control = this.losassoAuthorityControl;
    if (!control) return;
    for (const region of ["descriptorCandidate", "topologyCandidate",
      "structuredCandidate", "boundaryCandidate", "spgridCandidate",
      "epoch"] as const) capture(region, control);
  }

  /** The coarse phi arena is this lane's private hash table, not the neutral
   * directory triple this receipt region names. */
  captureFrontierFailureCoarseSources(_capture: OctreeLaneFrontierFailureCapture): void {}

  /** No compact descriptor candidates and no structured indirect records. */
  captureFrontierFailureCandidateSources(_capture: OctreeLaneFrontierFailureCapture): void {}

  /** No SPGrid hierarchy and no compact candidate rows to chase. */
  async decodeFrontierFailure(): Promise<OctreeLaneFrontierFailureReceipt> {
    return { spgridLevelDelta: [], spgridCandidateDispatch: [] };
  }

  /** Every lane-owned region of that readback belongs to the Power banks. */
  encodeGlobalFineDiagnosticCopies(_encoder: GPUCommandEncoder, _readback: GPUBuffer): void {}

  /** No coarse-directory schedule, and therefore no generation to stamp. */
  stampCoarseDirectoryGeneration(): void {}

  /** Nothing invocation-stable is staged per encoder on this lane. */
  retireSubmittedEncoder(_encoder: GPUCommandEncoder): void {}

  /** The analytic bootstrap sign lives in the Power structured boundary bank. */
  retireAnalyticBootstrap(): void {}

  async readSolveDiagnostics(): Promise<boolean> {
    const backend = this.losassoBackend;
    if (!backend) throw new Error("Losasso solve diagnostics require the reduced backend");
    const solverControl = backend.solverControl ?? backend.sources.rowCount;
    const readback = this.engine.device.createBuffer({
      label: "Octree Losasso live pressure diagnostics",
      size: 32 + 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({
      label: "Read octree Losasso pressure diagnostics",
    });
    encoder.copyBufferToBuffer(backend.sources.operator.control, 0, readback, 0, 32);
    encoder.copyBufferToBuffer(solverControl, 0, readback, 32,
      Math.min(64, solverControl.size));
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange();
      this.applyStepDiagnostics(
        new Uint32Array(mapped, 0, 8),
        new Uint32Array(mapped, 32, 16),
      );
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    return true;
  }

  /** Decode the pressure receipt copied by the production step snapshot.
   * This is the same ABI as readSolveDiagnostics(), but consumes bytes already
   * ordered by the physics submission instead of adding a telemetry submit. */
  applyStepDiagnostics(authority: Uint32Array, solver: Uint32Array): void {
    const epoch = authority[0] ?? 0;
    const exactIdentity = authority[5] === 1 && epoch !== 0;
    this.engine.info.topologyReused = exactIdentity || this.engine.topologyReusePending;
    if (exactIdentity && epoch !== this.engine.lastObservedExactTopologyReuseEpoch) {
      this.engine.lastObservedExactTopologyReuseEpoch = epoch;
      this.engine.info.topologyExactIdentityCount += 1;
    }
    const rows = authority[1] ?? 0;
    const valid = authority[3] === 1 && authority[4] === 0;
    this.engine.info.pressureCapacityOverflow = !valid;
    this.engine.info.frontierCapacityOverflow = !valid;
    this.engine.info.frontierRequiredLeaves = rows;
    this.engine.info.pressureRequiredRows = rows;
    this.engine.info.pressureSampleCount = rows;
    this.engine.info.liquidDofCount = rows;
    this.engine.info.faceCount = authority[2] ?? 0;
    this.engine.info.compressionRatio = rows
      / Math.max(1, this.engine.dims.nx * this.engine.dims.ny * this.engine.dims.nz);
    this.engine.residualRms = undefined;
    this.engine.initialResidualRms = undefined;
    this.engine.relativeResidual = undefined;
    this.engine.applyMGPCGDiagnostics(solver);
  }

  /** End-of-step copy sources for the Losasso-native diagnostic receipt. */
  get losassoAuthorityControl(): GPUBuffer | undefined {
    return this.losassoBackend?.sources.operator.control;
  }

  get losassoCoarsePhiControl(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptivePhiSource?.topologyEvidence
      ?? this.losassoCoarsePhi?.source.arena;
  }

  get losassoExtensionControl(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptiveVelocityReceiptSource
      ?? this.losassoBackend?.extensionBand?.source.control;
  }

  get losassoAdaptiveAcceptedGraphControl(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptiveSurfaceGraphSources?.accepted.control;
  }

  get losassoAdaptiveCandidateGraphControl(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptiveSurfaceGraphSources?.candidate.control;
  }

  get losassoAdaptivePhiControl(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptivePhiSource?.control;
  }

  get losassoAdaptivePhiReceipts(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptivePhiSource?.receipts;
  }

  get losassoAdaptiveVelocityReceipts(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptiveVelocityReceiptSource;
  }

  get losassoAdaptiveRendererDirectory(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptivePhiSource?.rendererDirectory;
  }

  get losassoCandidateAuthorityControl(): GPUBuffer | undefined {
    return this.losassoBackend?.candidateAuthorityControl;
  }

  get losassoAdaptiveMassControl(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptiveMassSource?.control;
  }

  get losassoAdaptiveMassReceipts(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptiveMassSource?.receipts;
  }

  get losassoAdaptiveCandidateMassControl(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptiveMassSource?.candidateControl;
  }

  get losassoAdaptiveCandidateMassReceipts(): GPUBuffer | undefined {
    return this.losassoBackend?.adaptiveMassSource?.candidateReceipts;
  }

  get losassoCandidateVelocityMigrationReceipt(): GPUBuffer | undefined {
    return this.losassoBackend?.candidateVelocityMigrationReceipt;
  }

  get velocityDebug() {
    const source = this.losassoBackend?.sources.velocitySampler;
    const extension = this.losassoBackend?.extensionBand?.source;
    const wet = this.losassoBackend?.sources;
    if (source) return {
      control: source.control,
      faceGeometry: source.faceGeometry,
      // The band's own metric row records which authority published a face and
      // at which dilation layer, which is what a support asymmetry has to be
      // read against. Capacity comes along so a dropped face is separable from
      // a face that was never proposed.
      faceMetrics: extension!.faceMetrics,
      faceCapacity: extension!.faceCapacity,
      projectedVelocity: extension!.projectedVelocity,
      extendedVelocity: source.extendedVelocity,
      wetControl: wet!.operator.control,
      wetFaceGeometry: wet!.dynamics.faceGeometry,
      wetAdvectedVelocity: wet!.dynamics.advectedVelocity,
      wetPredictedVelocity: wet!.dynamics.predictedVelocity,
      wetProjectedVelocity: wet!.projection.projectedVelocity,
      wetExtendedVelocity: wet!.extension.extendedVelocity,
      dimensions: source.dimensions,
      maximumLeafSize: source.maximumLeafSize,
    };
    // QA reconstructs a cubic field one-way from the accepted compact faces.
    // Adaptive physics itself samples the graph-owned nodal velocity arena.
    if (this.losassoBackend?.adaptiveVelocitySamplerSource && wet) return {
      control: wet.operator.control,
      faceGeometry: wet.dynamics.faceGeometry,
      projectedVelocity: wet.projection.projectedVelocity,
      extendedVelocity: wet.extension.extendedVelocity,
      wetControl: wet.operator.control,
      wetFaceGeometry: wet.dynamics.faceGeometry,
      wetAdvectedVelocity: wet.dynamics.advectedVelocity,
      wetPredictedVelocity: wet.dynamics.predictedVelocity,
      wetProjectedVelocity: wet.projection.projectedVelocity,
      wetExtendedVelocity: wet.extension.extendedVelocity,
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz] as const,
      maximumLeafSize: this.engine.topologyMaximumLeafSize,
    };
    return undefined;
    return undefined;
  }

  /** Frontier/dirty-tile forensics: the leaf-frontier header plus the shared
   * compaction scratch header. Read-only diagnostic surface. */
  get frontierDebug() {
    return { frontier: this.engine.leafFrontier, compaction: this.engine.compaction,
      dirtyFailureOffsetBytes: this.engine.dirtyFailureOffsetBytes };
  }

  get pressureDebug() {
    const source = this.losassoBackend?.sources;
    const wide = source?.wideSolver;
    const coarsePhi = this.losassoBackend?.adaptivePhiSource
      ?? this.losassoCoarsePhi?.source;
    return source && wide && coarsePhi ? {
      control: source.operator.control,
      rightHandSide: source.rightHandSide,
      diagonal: wide.diagonal,
      faces: source.operator.faces,
      faceGeometry: source.dynamics.faceGeometry,
      leafHeaders: this.engine.leafHeaders,
      rowPhi: coarsePhi.rowPhi,
      ghostDistances: coarsePhi.ghostDistances,
    } : undefined;
  }

  get coarsePhiDebug() {
    const source = this.losassoBackend?.adaptivePhiSource
      ?? this.losassoCoarsePhi?.source;
    const control = this.losassoBackend?.sources.operator.control;
    return source && control ? {
      control, rowPhi: source.rowPhi, leafHeaders: this.engine.leafHeaders,
      dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz] as const,
    } : undefined;
  }

  /**
   * Rejection-only forensics for the smoke harness and the `tools/probe-*`
   * scripts. The engine republishes these under their historical public names;
   * routing them through one named bag keeps the shared contract from growing
   * a member every time this lane adds a probe.
   */
  readonly forensics = {
    preconditionerContraction: async () => this.readLosassoPreconditionerContraction(),
    hierarchyCensus: async () => this.readLosassoHierarchyCensus(),
    authorityDiagnostics: async () => this.readLosassoAuthorityDiagnostics(),
    adaptiveSurfacePublication: async () => this.readAdaptiveSurfacePublicationDiagnostics(),
    adaptiveNodeReceipt: async () => this.readAdaptiveNodeReceipt(),
    adaptiveCandidateGraphReceipt: async () => this.readAdaptiveCandidateGraphReceipt(),
    adaptiveVelocityReceipts: async () => this.readAdaptiveVelocityReceipts(),
    adaptiveVelocityDiagnostics: async () => this.readAdaptiveVelocityDiagnostics(),
  };
  /**
   * How much of the initial residual one preconditioner application removes.
   *
   * `FLUID_SYMMETRY_STAGE_AUDIT=1` already captures r0, M*r0 and A*M*r0 for the
   * first solve of an advance. The error-propagation factor of the stationary
   * iteration built on M is ||r0 - A*M*r0|| / ||r0||: a working V-cycle sits
   * around 0.05-0.2, while a smoother-only preconditioner sits at 0.9 or above.
   * Iteration COUNT cannot distinguish those two — a degraded preconditioner
   * still converges, just with a count that tracks resolution — so this ratio,
   * not the count, is the preconditioner's regression metric.
   */
  async readLosassoPreconditionerContraction() {
    const symmetry = this.solverSymmetryStageAuditBuffers;
    const control = this.losassoBackend?.sources.operator.control;
    if (!symmetry || !control) return undefined;
    const words = OCTREE_LOSASSO_CONTROL_WORDS;
    const header = this.engine.device.createBuffer({
      label: "Losasso contraction row count",
      size: words * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let rowCount = 0;
    {
      const encoder = this.engine.device.createCommandEncoder({
        label: "Read Losasso contraction row count",
      });
      encoder.copyBufferToBuffer(control, 0, header, 0, words * 4);
      this.engine.device.queue.submit([encoder.finish()]);
      try {
        await header.mapAsync(GPUMapMode.READ);
        const controlWords = new Uint32Array(header.getMappedRange().slice(0));
        if (controlWords[3] !== 1) return undefined;
        rowCount = controlWords[1] ?? 0;
      } finally {
        header.destroy();
      }
    }
    const vectorBytes = rowCount * 4;
    if (rowCount === 0 || vectorBytes > symmetry.initialResidual.size) return undefined;
    const readback = this.engine.device.createBuffer({
      label: "Losasso preconditioner contraction",
      size: vectorBytes * 3, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({
      label: "Read Losasso preconditioner contraction",
    });
    encoder.copyBufferToBuffer(symmetry.initialResidual, 0, readback, 0, vectorBytes);
    encoder.copyBufferToBuffer(symmetry.initialPreconditioned, 0, readback, vectorBytes, vectorBytes);
    encoder.copyBufferToBuffer(
      symmetry.initialPreconditionedImage, 0, readback, vectorBytes * 2, vectorBytes);
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const all = new Float32Array(readback.getMappedRange().slice(0));
      const residual = all.subarray(0, rowCount);
      const preconditioned = all.subarray(rowCount, rowCount * 2);
      const image = all.subarray(rowCount * 2, rowCount * 3);
      let residualSquared = 0, remainderSquared = 0, imageSquared = 0, preconditionedDotResidual = 0;
      let nonFinite = 0;
      for (let row = 0; row < rowCount; row += 1) {
        const r = residual[row]!, m = preconditioned[row]!, a = image[row]!;
        if (!Number.isFinite(r) || !Number.isFinite(m) || !Number.isFinite(a)) {
          nonFinite += 1;
          continue;
        }
        residualSquared += r * r;
        remainderSquared += (r - a) * (r - a);
        imageSquared += a * a;
        preconditionedDotResidual += m * r;
      }
      const residualNorm = Math.sqrt(residualSquared);
      return {
        rows: rowCount,
        nonFiniteRows: nonFinite,
        residualNorm,
        // The headline number. >= ~0.9 means the V-cycle is not acting as one.
        contraction: residualNorm > 0 ? Math.sqrt(remainderSquared) / residualNorm : Number.NaN,
        imageNorm: Math.sqrt(imageSquared),
        // Must be > 0 for CG: it is the first gamma the recurrence divides by.
        preconditionedDotResidual,
      };
    } finally {
      readback.destroy();
    }
  }

  async readLosassoAuthorityDiagnostics(): Promise<Readonly<{
    authority: readonly number[];
    candidate: readonly number[];
    candidateHeader: readonly number[];
    solver: readonly number[];
    coarsePhi: readonly number[];
    adaptiveGraph: readonly number[];
    adaptivePhiControl: readonly number[];
    adaptivePhi: readonly number[];
    adaptiveRenderer: readonly number[];
    candidateAdaptiveGraph: readonly number[];
    ownerCandidate: readonly number[];
    frontierControl: readonly number[];
    adaptiveMassControl: readonly number[];
    adaptiveMassReceipts: readonly number[];
    velocityMigration: readonly number[];
  }> | undefined> {
    const backend = this.losassoBackend;
    const coarseControl = backend?.adaptivePhiSource?.topologyEvidence
      ?? this.losassoCoarsePhi?.source.arena;
    if (!backend || !coarseControl) return undefined;
    const adaptivePhiReceiptBytes = 4 * OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS;
    const adaptiveRendererOffset = 352 + adaptivePhiReceiptBytes;
    const adaptivePhiControlOffset = adaptiveRendererOffset + 32;
    const candidateAdaptiveGraphOffset = adaptivePhiControlOffset + 80;
    const ownerCandidateOffset = candidateAdaptiveGraphOffset + 128;
    const frontierControlOffset = ownerCandidateOffset + 128;
    const adaptiveMassControlOffset = frontierControlOffset + 64;
    const adaptiveMassReceiptsOffset = adaptiveMassControlOffset + 128;
    const velocityMigrationOffset = adaptiveMassReceiptsOffset + 128;
    const readback = this.engine.device.createBuffer({
      label: "Read Losasso reduced authority",
      size: velocityMigrationOffset + 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({ label: "Read Losasso authority controls" });
    encoder.copyBufferToBuffer(backend.sources.operator.control, 0, readback, 0, 32);
    encoder.copyBufferToBuffer(backend.candidateAuthorityControl, 0, readback, 32, 32);
    encoder.copyBufferToBuffer(this.engine.candidateLeafHeaders, 0, readback, 64, 48);
    const solver = backend.solverControl ?? backend.sources.rowCount;
    encoder.copyBufferToBuffer(solver, 0, readback, 112, Math.min(32, solver.size));
    encoder.copyBufferToBuffer(coarseControl, 0, readback, 144, 80);
    const adaptiveGraph = backend.adaptiveSurfaceGraphSources?.accepted.control;
    const candidateAdaptiveGraph = backend.adaptiveSurfaceGraphSources?.candidate.control;
    const adaptivePhiControl = backend.adaptivePhiSource?.control;
    const adaptivePhi = backend.adaptivePhiSource?.receipts;
    const adaptiveRenderer = backend.adaptivePhiSource?.rendererDirectory;
    const adaptiveMassControl = backend.adaptiveMassSource?.control;
    const adaptiveMassReceipts = backend.adaptiveMassSource?.receipts;
    if (adaptiveGraph) encoder.copyBufferToBuffer(adaptiveGraph, 0, readback, 224, 128);
    if (adaptivePhi) encoder.copyBufferToBuffer(adaptivePhi, 0, readback, 352,
      adaptivePhiReceiptBytes);
    if (adaptiveRenderer) encoder.copyBufferToBuffer(adaptiveRenderer, 0, readback,
      adaptiveRendererOffset, 32);
    if (adaptivePhiControl) encoder.copyBufferToBuffer(adaptivePhiControl, 0, readback,
      adaptivePhiControlOffset, 80);
    if (candidateAdaptiveGraph) encoder.copyBufferToBuffer(candidateAdaptiveGraph, 0,
      readback, candidateAdaptiveGraphOffset, 128);
    encoder.copyBufferToBuffer(this.engine.ownerPages.candidateTransaction, 0, readback,
      ownerCandidateOffset, 128);
    encoder.copyBufferToBuffer(this.engine.leafFrontier, 0, readback, frontierControlOffset, 64);
    if (adaptiveMassControl) encoder.copyBufferToBuffer(adaptiveMassControl, 0, readback,
      adaptiveMassControlOffset, Math.min(128, adaptiveMassControl.size));
    if (adaptiveMassReceipts) encoder.copyBufferToBuffer(adaptiveMassReceipts, 0, readback,
      adaptiveMassReceiptsOffset, Math.min(128, adaptiveMassReceipts.size));
    encoder.copyBufferToBuffer(backend.candidateVelocityMigrationReceipt, 0, readback,
      velocityMigrationOffset, 32);
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return Object.freeze({
        authority: Array.from(words.slice(0, 8)),
        candidate: Array.from(words.slice(8, 16)),
        candidateHeader: Array.from(words.slice(16, 28)),
        solver: Array.from(words.slice(28, 36)),
        coarsePhi: Array.from(words.slice(36, 56)),
        adaptiveGraph: Array.from(words.slice(56, 88)),
        adaptivePhiControl: Array.from(words.slice(adaptivePhiControlOffset / 4,
          adaptivePhiControlOffset / 4 + 20)),
        adaptivePhi: Array.from(words.slice(88,
          88 + OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS)),
        adaptiveRenderer: Array.from(words.slice(adaptiveRendererOffset / 4,
          adaptiveRendererOffset / 4 + 8)),
        candidateAdaptiveGraph: Array.from(words.slice(candidateAdaptiveGraphOffset / 4,
          candidateAdaptiveGraphOffset / 4 + 32)),
        ownerCandidate: Array.from(words.slice(ownerCandidateOffset / 4,
          ownerCandidateOffset / 4 + 32)),
        frontierControl: Array.from(words.slice(frontierControlOffset / 4,
          frontierControlOffset / 4 + 16)),
        adaptiveMassControl: Array.from(words.slice(adaptiveMassControlOffset / 4,
          adaptiveMassControlOffset / 4 + 32)),
        adaptiveMassReceipts: Array.from(words.slice(adaptiveMassReceiptsOffset / 4,
          adaptiveMassReceiptsOffset / 4 + 32)),
        velocityMigration: Array.from(words.slice(velocityMigrationOffset / 4,
          velocityMigrationOffset / 4 + 8)),
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Diagnostic-only factor-one graph/scalar/renderer snapshot. The returned
   * arrays are trimmed to live GPU counts, excluding capacity padding. */
  async readAdaptiveSurfacePublicationDiagnostics() {
    const backend = this.losassoBackend;
    const graph = backend?.adaptiveSurfaceGraphSources?.accepted;
    const candidateGraph = backend?.adaptiveSurfaceGraphSources?.candidate;
    const phi = backend?.adaptivePhiSource;
    const mass = backend?.adaptiveMassSource;
    const stencil = backend?.adaptiveVelocityStencilDiagnosticSources;
    const candidateVelocity = backend?.adaptiveVelocityCandidateDiagnosticSources;
    if (!graph || !candidateGraph || !phi || !mass || !stencil || !candidateVelocity) {
      return undefined;
    }
    const authorityControl = backend.sources.operator.control;
    const pressureFaces = backend.sources.operator.faces;
    const pressureRowToGraphLeaf = graph.pressureRowToGraphLeaf;
    const faceGeometry = backend.sources.dynamics.faceGeometry;
    const projectedFaceVelocity = backend.sources.dynamics.projectedVelocity;
    const predictedFaceVelocity = backend.sources.dynamics.predictedVelocity;
    const advectedFaceVelocity = backend.sources.dynamics.advectedVelocity;
    const extendedFaceVelocity = backend.sources.dynamics.extendedVelocity;
    const graphControlBytes = LOSASSO_SURFACE_GRAPH_CONTROL_WORDS * 4;
    const phiControlBytes = 80;
    const authorityControlOffset = graphControlBytes + phiControlBytes;
    const faceGeometryOffset = authorityControlOffset + authorityControl.size;
    const pressureFacesOffset = faceGeometryOffset + faceGeometry.size;
    const pressureRowToGraphLeafOffset = pressureFacesOffset + pressureFaces.size;
    const projectedFaceVelocityOffset = pressureRowToGraphLeafOffset
      + pressureRowToGraphLeaf.size;
    const predictedFaceVelocityOffset = projectedFaceVelocityOffset
      + projectedFaceVelocity.size;
    const advectedFaceVelocityOffset = predictedFaceVelocityOffset
      + predictedFaceVelocity.size;
    const extendedFaceVelocityOffset = advectedFaceVelocityOffset
      + advectedFaceVelocity.size;
    const stencilControlOffset = extendedFaceVelocityOffset + extendedFaceVelocity.size;
    const stencilOffset = stencilControlOffset + stencil.control.size;
    const candidateAuthorityControlOffset = stencilOffset + stencil.records.size;
    const candidateFaceGeometryOffset = candidateAuthorityControlOffset
      + candidateVelocity.authorityControl.size;
    const candidateExtendedFaceVelocityOffset = candidateFaceGeometryOffset
      + candidateVelocity.faceGeometry.size;
    const candidateNodalVelocityOffset = candidateExtendedFaceVelocityOffset
      + candidateVelocity.extendedVelocity.size;
    const leavesOffset = candidateNodalVelocityOffset + candidateVelocity.nodalVelocity.size;
    const nodalPhiOffset = leavesOffset + graph.leaves.size;
    const nodesOffset = nodalPhiOffset + graph.phi.size;
    const constraintsOffset = nodesOffset + graph.nodes.size;
    const adjacencyOffset = constraintsOffset + graph.constraints.size;
    const nodalVelocityOffset = adjacencyOffset + graph.adjacency.size;
    const nodeValidityOffset = nodalVelocityOffset + graph.nodalVelocity.size;
    const rendererOffset = nodeValidityOffset + graph.nodeValidity.size;
    const transportBandMaskOffset = rendererOffset + phi.rendererDirectory.size;
    const redistanceDistanceAOffset = transportBandMaskOffset + phi.transportBandMask.size;
    const redistanceDistanceBOffset = redistanceDistanceAOffset + phi.redistanceDistanceA.size;
    const phiReceiptsOffset = redistanceDistanceBOffset + phi.redistanceDistanceB.size;
    const candidateGraphControlOffset = phiReceiptsOffset + phi.receipts.size;
    const candidateNodesOffset = candidateGraphControlOffset + candidateGraph.control.size;
    const candidateNodalPhiOffset = candidateNodesOffset + candidateGraph.nodes.size;
    const acceptedMassOffset = candidateNodalPhiOffset + candidateGraph.phi.size;
    const leafRhoPhiOffset = acceptedMassOffset + mass.acceptedMass.size;
    const readback = this.engine.device.createBuffer({
      label: "Read adaptive surface publication",
      size: leafRhoPhiOffset + mass.leafRhoPhi.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({
      label: "Read adaptive graph scalar and renderer publication",
    });
    encoder.copyBufferToBuffer(graph.control, 0, readback, 0, graphControlBytes);
    encoder.copyBufferToBuffer(phi.control, 0, readback, graphControlBytes, phiControlBytes);
    encoder.copyBufferToBuffer(authorityControl, 0, readback, authorityControlOffset,
      authorityControl.size);
    encoder.copyBufferToBuffer(faceGeometry, 0, readback, faceGeometryOffset,
      faceGeometry.size);
    encoder.copyBufferToBuffer(pressureFaces, 0, readback, pressureFacesOffset,
      pressureFaces.size);
    encoder.copyBufferToBuffer(pressureRowToGraphLeaf, 0, readback,
      pressureRowToGraphLeafOffset, pressureRowToGraphLeaf.size);
    encoder.copyBufferToBuffer(projectedFaceVelocity, 0, readback,
      projectedFaceVelocityOffset, projectedFaceVelocity.size);
    encoder.copyBufferToBuffer(predictedFaceVelocity, 0, readback,
      predictedFaceVelocityOffset, predictedFaceVelocity.size);
    encoder.copyBufferToBuffer(advectedFaceVelocity, 0, readback,
      advectedFaceVelocityOffset, advectedFaceVelocity.size);
    encoder.copyBufferToBuffer(extendedFaceVelocity, 0, readback,
      extendedFaceVelocityOffset, extendedFaceVelocity.size);
    encoder.copyBufferToBuffer(stencil.control, 0, readback, stencilControlOffset,
      stencil.control.size);
    encoder.copyBufferToBuffer(stencil.records, 0, readback, stencilOffset,
      stencil.records.size);
    encoder.copyBufferToBuffer(candidateVelocity.authorityControl, 0, readback,
      candidateAuthorityControlOffset, candidateVelocity.authorityControl.size);
    encoder.copyBufferToBuffer(candidateVelocity.faceGeometry, 0, readback,
      candidateFaceGeometryOffset, candidateVelocity.faceGeometry.size);
    encoder.copyBufferToBuffer(candidateVelocity.extendedVelocity, 0, readback,
      candidateExtendedFaceVelocityOffset, candidateVelocity.extendedVelocity.size);
    encoder.copyBufferToBuffer(candidateVelocity.nodalVelocity, 0, readback,
      candidateNodalVelocityOffset, candidateVelocity.nodalVelocity.size);
    encoder.copyBufferToBuffer(graph.leaves, 0, readback, leavesOffset, graph.leaves.size);
    encoder.copyBufferToBuffer(graph.phi, 0, readback, nodalPhiOffset, graph.phi.size);
    encoder.copyBufferToBuffer(graph.nodes, 0, readback, nodesOffset, graph.nodes.size);
    encoder.copyBufferToBuffer(graph.constraints, 0, readback, constraintsOffset,
      graph.constraints.size);
    encoder.copyBufferToBuffer(graph.adjacency, 0, readback, adjacencyOffset,
      graph.adjacency.size);
    encoder.copyBufferToBuffer(graph.nodalVelocity, 0, readback, nodalVelocityOffset,
      graph.nodalVelocity.size);
    encoder.copyBufferToBuffer(graph.nodeValidity, 0, readback, nodeValidityOffset,
      graph.nodeValidity.size);
    encoder.copyBufferToBuffer(phi.rendererDirectory, 0, readback, rendererOffset,
      phi.rendererDirectory.size);
    encoder.copyBufferToBuffer(phi.transportBandMask, 0, readback, transportBandMaskOffset,
      phi.transportBandMask.size);
    encoder.copyBufferToBuffer(phi.redistanceDistanceA, 0, readback,
      redistanceDistanceAOffset, phi.redistanceDistanceA.size);
    encoder.copyBufferToBuffer(phi.redistanceDistanceB, 0, readback,
      redistanceDistanceBOffset, phi.redistanceDistanceB.size);
    encoder.copyBufferToBuffer(phi.receipts, 0, readback,
      phiReceiptsOffset, phi.receipts.size);
    encoder.copyBufferToBuffer(candidateGraph.control, 0, readback,
      candidateGraphControlOffset, candidateGraph.control.size);
    encoder.copyBufferToBuffer(candidateGraph.nodes, 0, readback,
      candidateNodesOffset, candidateGraph.nodes.size);
    encoder.copyBufferToBuffer(candidateGraph.phi, 0, readback,
      candidateNodalPhiOffset, candidateGraph.phi.size);
    encoder.copyBufferToBuffer(mass.acceptedMass, 0, readback,
      acceptedMassOffset, mass.acceptedMass.size);
    encoder.copyBufferToBuffer(mass.leafRhoPhi, 0, readback,
      leafRhoPhiOffset, mass.leafRhoPhi.size);
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange();
      const graphControl = Uint32Array.from(new Uint32Array(mapped, 0,
        LOSASSO_SURFACE_GRAPH_CONTROL_WORDS));
      const phiControl = Uint32Array.from(new Uint32Array(mapped, graphControlBytes, 20));
      const authorityWords = authorityControl.size / 4;
      const authority = Uint32Array.from(new Uint32Array(mapped, authorityControlOffset,
        authorityWords));
      const faceCount = Math.min(authority[2] ?? 0, faceGeometry.size / 16,
        projectedFaceVelocity.size / 4, predictedFaceVelocity.size / 4,
        advectedFaceVelocity.size / 4,
        extendedFaceVelocity.size / 4);
      const acceptedStencilControl = Uint32Array.from(new Uint32Array(mapped,
        stencilControlOffset, stencil.control.size / 4));
      const candidateAuthority = Uint32Array.from(new Uint32Array(mapped,
        candidateAuthorityControlOffset, candidateVelocity.authorityControl.size / 4));
      const leafCount = Math.min(graphControl[1] ?? 0, graph.leafCapacity);
      const nodeCount = Math.min(graphControl[2] ?? 0, graph.nodeCapacity);
      const stencilWords = Math.min(stencil.records.size / 4, 3 * 36 * nodeCount);
      const acceptedFaceGeometry = Uint32Array.from(new Uint32Array(mapped,
        faceGeometryOffset, 4 * faceCount));
      const acceptedPressureFaces = Uint32Array.from(new Uint32Array(mapped,
        pressureFacesOffset, 8 * faceCount));
      const acceptedPressureRowToGraphLeaf = Uint32Array.from(new Uint32Array(mapped,
        pressureRowToGraphLeafOffset, pressureRowToGraphLeaf.size / 4));
      const projectedFaces = Uint32Array.from(new Uint32Array(mapped,
        projectedFaceVelocityOffset, faceCount));
      const predictedFaces = Uint32Array.from(new Uint32Array(mapped,
        predictedFaceVelocityOffset, faceCount));
      const advectedFaces = Uint32Array.from(new Uint32Array(mapped,
        advectedFaceVelocityOffset, faceCount));
      const extendedFaces = Uint32Array.from(new Uint32Array(mapped,
        extendedFaceVelocityOffset, faceCount));
      const acceptedStencils = Uint32Array.from(new Uint32Array(mapped,
        stencilOffset, stencilWords));
      const candidateFaceCount = Math.min(candidateAuthority[2] ?? 0,
        candidateVelocity.faceGeometry.size / 16, candidateVelocity.extendedVelocity.size / 4);
      const candidateGeometry = Uint32Array.from(new Uint32Array(mapped,
        candidateFaceGeometryOffset, 4 * candidateFaceCount));
      const candidateExtended = Uint32Array.from(new Uint32Array(mapped,
        candidateExtendedFaceVelocityOffset, candidateFaceCount));
      const candidateGraphControl = Uint32Array.from(new Uint32Array(mapped,
        candidateGraphControlOffset, LOSASSO_SURFACE_GRAPH_CONTROL_WORDS));
      const candidateNodeCount = Math.min(candidateGraphControl[2] ?? 0,
        candidateGraph.nodeCapacity);
      const candidateNodalVelocity = Uint32Array.from(new Uint32Array(mapped,
        candidateNodalVelocityOffset, 8 * candidateNodeCount));
      const candidateNodes = Uint32Array.from(new Uint32Array(mapped,
        candidateNodesOffset, 4 * candidateNodeCount));
      const candidateNodalPhi = Uint32Array.from(new Uint32Array(mapped,
        candidateNodalPhiOffset, 2 * candidateNodeCount));
      const acceptedMass = Float32Array.from(new Float32Array(mapped,
        acceptedMassOffset, leafCount));
      const leafRhoPhi = Float32Array.from(new Float32Array(mapped,
        leafRhoPhiOffset, 4 * leafCount));
      const leaves = Uint32Array.from(new Uint32Array(mapped, leavesOffset, 16 * leafCount));
      const nodalPhi = Uint32Array.from(new Uint32Array(mapped, nodalPhiOffset, 2 * nodeCount));
      const nodes = Uint32Array.from(new Uint32Array(mapped, nodesOffset, 4 * nodeCount));
      const constraints = Uint32Array.from(new Uint32Array(mapped, constraintsOffset,
        12 * nodeCount));
      const adjacency = Uint32Array.from(new Uint32Array(mapped, adjacencyOffset,
        12 * nodeCount));
      const nodalVelocity = Uint32Array.from(new Uint32Array(mapped, nodalVelocityOffset,
        8 * nodeCount));
      const nodeValidity = Uint32Array.from(new Uint32Array(mapped, nodeValidityOffset,
        nodeCount));
      const renderer = Uint32Array.from(new Uint32Array(mapped, rendererOffset,
        phi.rendererDirectory.size / 4));
      const transportBandMask = Uint32Array.from(new Uint32Array(mapped,
        transportBandMaskOffset, nodeCount));
      const redistanceDistanceA = Float32Array.from(new Float32Array(mapped,
        redistanceDistanceAOffset, nodeCount));
      const redistanceDistanceB = Float32Array.from(new Float32Array(mapped,
        redistanceDistanceBOffset, nodeCount));
      const phiReceipts = Uint32Array.from(new Uint32Array(mapped,
        phiReceiptsOffset, phi.receipts.size / 4));
      return Object.freeze({ authorityControl: authority,
        faceGeometry: acceptedFaceGeometry,
        pressureFaces: acceptedPressureFaces,
        pressureRowToGraphLeaf: acceptedPressureRowToGraphLeaf,
        projectedFaceVelocity: projectedFaces,
        predictedFaceVelocity: predictedFaces,
        advectedFaceVelocity: advectedFaces,
        extendedFaceVelocity: extendedFaces,
        stencilControl: acceptedStencilControl, stencils: acceptedStencils,
        candidateAuthorityControl: candidateAuthority,
        candidateFaceGeometry: candidateGeometry,
        candidateExtendedFaceVelocity: candidateExtended,
        candidateNodalVelocity,
        candidateGraphControl, candidateNodes, candidateNodalPhi,
        acceptedMass, leafRhoPhi,
        graphControl, phiControl, leaves, nodalPhi, nodes, constraints,
        adjacency, nodalVelocity, nodeValidity, renderer, transportBandMask,
        redistanceDistanceA, redistanceDistanceB, phiReceipts,
        dimensions: [this.engine.dims.nx, this.engine.dims.ny, this.engine.dims.nz] as const });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /**
   * Per-level publication census for the Losasso multigrid hierarchy.
   *
   * Nothing else reads these words. An unpublished sub-level silently disables
   * the fused sub-L0 cycle (its enable predicate requires word 3 == 1 on every
   * level), which degrades the preconditioner to four damped-Jacobi sweeps on
   * L0 without failing anything — CG then simply takes more iterations. This
   * census is the only surface that can tell that apart from a healthy solve.
   *
   * Both the level control buffer and the fused arena's mirror of it are read:
   * the coefficient-refresh path can set the level's error word without ever
   * writing words 3/4 back into the arena, so they can legitimately disagree,
   * and it is the arena copy the V-cycle actually gates on.
   */
  async readLosassoHierarchyCensus() {
    const vcycle = this.losassoBackend?.sources.vcycle;
    if (!vcycle) return undefined;
    const words = OCTREE_LOSASSO_CONTROL_WORDS;
    const bytes = words * 4;
    const levels = vcycle.levels;
    const fused = vcycle.fusedSubL0;
    // Transitions mirror L1..Ln, so the arena holds one fewer record than levels.
    const arenaRecords = fused ? Math.max(0, levels.length - 1) : 0;
    const readback = this.engine.device.createBuffer({
      label: "Losasso hierarchy publication census",
      size: (levels.length + arenaRecords) * bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.engine.device.createCommandEncoder({
      label: "Read Losasso hierarchy publication census",
    });
    levels.forEach((level, index) => {
      encoder.copyBufferToBuffer(level.control, 0, readback, index * bytes, bytes);
    });
    for (let transition = 0; transition < arenaRecords; transition += 1) {
      const layout = fused!.levelLayouts[transition + 1]!;
      encoder.copyBufferToBuffer(
        fused!.arena,
        (fused!.acceptedBankWordOffset + layout.baseWords
          + layout.controlOffsetWords) * 4,
        readback, (levels.length + transition) * bytes, bytes,
      );
    }
    this.engine.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const all = new Uint32Array(readback.getMappedRange().slice(0));
      const record = (base: number) => ({
        generation: all[base]!, rows: all[base + 1]!, faces: all[base + 2]!,
        published: all[base + 3]!, errorBits: all[base + 4]!,
        topologyReused: all[base + 5]!, directoryCapacity: all[base + 6]!,
      });
      const levelRecords = levels.map((_, index) => ({
        level: index, ...record(index * words),
      }));
      const arena = Array.from({ length: arenaRecords }, (_, transition) => ({
        level: transition + 1, ...record((levels.length + transition) * words),
      }));
      // The fused predicate's exact conjunction, reproduced on the host so a
      // census can state the consequence rather than leave it to be inferred.
      const capacities = fused?.levelRowCapacities ?? [];
      const cycleEnabled = arena.length > 0 && arena.every((entry) => entry.published === 1
        && entry.rows <= (capacities[entry.level] ?? 0));
      return {
        levelCount: levels.length,
        levels: levelRecords,
        arena,
        levelRowCapacities: [...capacities],
        cycleEnabled,
        firstUnpublishedLevel: levelRecords.find((entry) =>
          entry.level > 0 && entry.published !== 1)?.level,
        firstErroredLevel: levelRecords.find((entry) => entry.errorBits !== 0)?.level,
      };
    } finally {
      readback.destroy();
    }
  }

  /** Observational sealed-wet-owner tripwire, never consumed by simulation. */
  get rigidCouplingDiagnosticBuffer(): GPUBuffer | undefined {
    return this.rigidCouplingDiagnostics?.diagnosticBuffer;
  }

  get rigidBoundaryRefreshDiagnosticBuffer(): GPUBuffer | undefined {
    return this.losassoBackend?.rigidBoundaryRefreshDiagnostics;
  }

  /** Diagnostic-only receipt for the factor-one surface authority. */
  async readCoarseSurfaceTrackerReceipt() {
    if (this.engine.coarseOnlySurfaceTracking) {
      return this.losassoBackend?.readAdaptiveSurfaceGraphReceipt("accepted");
    }
    return this.engine.coarseOnlySummary?.readReceipt();
  }

  /** Diagnostic-only receipt for a rejected adaptive candidate transaction. */
  readAdaptiveCandidateGraphReceipt() {
    return this.losassoBackend?.readAdaptiveSurfaceGraphReceipt("candidate");
  }

  readAdaptiveVelocityReceipts() {
    return this.losassoBackend?.readAdaptiveVelocityReceipts();
  }

  readAdaptiveVelocityDiagnostics() {
    return this.losassoBackend?.readAdaptiveVelocityDiagnostics();
  }

  /** Diagnostic receipt for the GPU-authored unique shared-node worklist. */
  async readAdaptiveNodeReceipt() {
    const graph = await this.losassoBackend?.readAdaptiveSurfaceGraphReceipt("accepted");
    const source = this.losassoBackend?.adaptiveSurfaceGraphSources?.accepted;
    if (graph && source) {
      return Object.freeze({
        count: graph.nodeCount,
        generation: graph.epoch,
        published: graph.published,
        errors: graph.errors,
        capacity: source.nodeCapacity,
        dispatch: graph.nodeDispatch,
        leafCount: graph.leafCount,
        independentNodes: graph.independentNodeCount,
        constrainedNodes: graph.edgeHangingNodeCount + graph.faceHangingNodeCount,
        edgeHangingNodes: graph.edgeHangingNodeCount,
        faceHangingNodes: graph.faceHangingNodeCount,
        missingLookups: graph.missingLookupCount,
        coverageErrors: graph.coverageErrors,
        adjacencyErrors: graph.reciprocalAdjacencyErrors,
      });
    }
    return undefined;
  }

  /**
   * Fail-closed t=0 validation of this lane's own published authority.
   *
   * The paper path must be complete before the first trajectory can be
   * requested. These are one-time post-fence readbacks for UI readiness and
   * diagnostics; recurring frame scheduling remains GPU-resident. The words
   * are this lane's own receipt ABI, which is why the shell cannot decode them.
   */
  async validateInitialAuthority(context: {
    readonly dimensions: readonly [number, number, number];
    readonly refreshInfo: () => void;
  }): Promise<Readonly<{ converged: boolean; iterationsUsed: number }> | undefined> {
    const reduced = await this.readLosassoAuthorityDiagnostics();
    context.refreshInfo();
    const authority = reduced?.authority ?? [];
    const candidate = reduced?.candidate ?? [];
    const candidateHeader = reduced?.candidateHeader ?? [];
    const solver = reduced?.solver ?? [];
    const coarsePhi = reduced?.coarsePhi ?? [];
    const adaptiveGraph = reduced?.adaptiveGraph ?? [];
    const adaptivePhiControl = reduced?.adaptivePhiControl ?? [];
    const adaptiveRenderer = reduced?.adaptiveRenderer ?? [];
    const ready = authority.length >= 5 && authority[0] !== 0
      && authority[1] > 0 && authority[2] > 0
      && authority[3] === 1 && authority[4] === 0;
    const solverReady = solver.length >= 3 && solver[0] === 0 && solver[1] !== 0;
    if (!ready || !solverReady) {
      const [owner, adaptiveCandidate, adaptiveVelocity] = await Promise.all([
        this.engine.readOwnerPageControl(),
        this.readAdaptiveCandidateGraphReceipt(),
        this.readAdaptiveVelocityReceipts(),
      ]);
      throw new Error("Paused t=0 Losasso authority rejected: authority="
        + JSON.stringify(authority) + "; candidate=" + JSON.stringify(candidate)
        + "; candidateHeader=" + JSON.stringify(candidateHeader)
        + "; solver=" + JSON.stringify(solver)
        + "; coarsePhi=" + JSON.stringify(coarsePhi)
        + "; owner=" + JSON.stringify(owner)
        + "; adaptiveCandidate=" + JSON.stringify(adaptiveCandidate)
        + "; adaptiveVelocity=" + JSON.stringify(adaptiveVelocity));
    }
    if ((adaptiveGraph[0] ?? 0) !== 0) {
      const adaptiveReady = adaptiveGraph.length >= 32
        && adaptivePhiControl.length >= 20 && adaptiveRenderer.length >= 8
        && adaptiveGraph[0] === authority[0] && adaptiveGraph[1] > 0
        && adaptiveGraph[2] > 0 && adaptiveGraph[3] === adaptiveGraph[0]
        && adaptiveGraph[4] === 0 && adaptiveGraph[5] !== 0
        && adaptiveGraph[6] === adaptiveGraph[5]
        && adaptiveGraph[28] === authority[1] && adaptiveGraph[29] === 0
        && adaptivePhiControl[0] === 0x4150_4849
        && adaptivePhiControl[1] === adaptiveGraph[0]
        && adaptivePhiControl[2] === adaptiveGraph[5]
        && adaptivePhiControl[4] === adaptiveGraph[2]
        && adaptivePhiControl[5] === adaptiveGraph[1]
        && adaptivePhiControl[7] === 1 && adaptivePhiControl[12] === 0
        && adaptiveRenderer[0] === 0x8000_0000
        && adaptiveRenderer[1] === adaptiveGraph[5]
        && adaptiveRenderer[2] === adaptiveGraph[1]
        && coarsePhi[0] === OCTREE_LOSASSO_COARSE_PHI_MAGIC
        && coarsePhi[1] === adaptiveGraph[0] && coarsePhi[2] === adaptiveGraph[1]
        && coarsePhi[12] === adaptiveGraph[5] && coarsePhi[13] === 0
        && coarsePhi[14] === adaptiveGraph[5];
      if (!adaptiveReady) {
        const [adaptiveVelocity, adaptiveVelocityDiagnostics] = await Promise.all([
          this.readAdaptiveVelocityReceipts(),
          this.readAdaptiveVelocityDiagnostics(),
        ]);
        const adaptiveVelocityFailures = summarizeAdaptiveVelocityFailureDiagnostics(
          adaptiveVelocityDiagnostics, context.dimensions);
        throw new Error("Paused t=0 adaptive surface publication rejected: graph="
          + JSON.stringify(adaptiveGraph) + "; phiControl="
          + JSON.stringify(adaptivePhiControl) + "; renderer="
          + JSON.stringify(adaptiveRenderer) + "; coarsePhi="
          + JSON.stringify(coarsePhi) + "; velocity="
          + JSON.stringify(adaptiveVelocity) + "; velocityFailures="
          + JSON.stringify(adaptiveVelocityFailures));
      }
      // This is a post-queue-fence adoption of the current GPU tuple, not a
      // predicted host generation. It breaks the startup dependency cycle:
      // the renderer view may now attach before the first recurring step.
      this.engine.applyAdaptiveSurfaceGenerationReceipt(adaptiveGraph[5]!);
      context.refreshInfo();
    }
    return { converged: solverReady, iterationsUsed: solver[2] ?? 0 };
  }

  destroy(): void {
    this.losassoReadyCommit?.destroy();
    this.losassoCoarsePhi?.destroy();
    this.losassoRowMotion?.destroy();
    this.losassoConditionedOperator?.destroy();
    this.rigidCouplingDiagnostics?.destroy();
    this.losassoBackend?.destroy();
    this.losassoFineTransportA?.destroy(); this.losassoFineTransportB?.destroy();
  }
}

/**
 * Decode the adaptive velocity extension's unresolved-node bank into a shape a
 * rejection message can carry. Nothing consumes it but the two t=0 throws: a
 * raw word dump cannot say which of the five causes stopped a node.
 */
function summarizeAdaptiveVelocityFailureDiagnostics(words: readonly number[] | undefined,
  dimensions: readonly [number, number, number]) {
  if (!words) return undefined;
  const bits = new Uint32Array(1);
  const float = (word: number) => {
    bits[0] = word >>> 0;
    return new Float32Array(bits.buffer)[0]!;
  };
  const position = (item: number) => {
    if (item === 0xffff_ffff) return undefined;
    const dx = dimensions[0] + 1, dy = dimensions[1] + 1;
    return [item % dx, Math.floor(item / dx) % dy,
      Math.floor(item / (dx * dy))] as const;
  };
  const names = ["accepted", "predictor", "candidate", "candidatePredictor"] as const;
  return names.map((name, bank) => {
    const base = bank * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS;
    const captured = Math.min(words[base + 6] ?? 0,
      OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_RECORDS);
    const records = Array.from({ length: Math.min(captured, 12) }, (_unused, record) => {
      const at = base + OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_HEADER_WORDS
        + record * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_WORDS;
      const causeBits = words[at + 5] ?? 0;
      const item = words[at + 1] ?? 0xffff_ffff;
      return { node: words[at] ?? 0xffff_ffff, position: position(item),
        missingComponents: words[at + 2] ?? 0,
        constraintCount: words[at + 3] ?? 0xffff_ffff,
        phi: float(words[at + 4] ?? 0), causeBits,
        causes: [...(causeBits & 1 ? ["no-adjacency"] : []),
          ...(causeBits & 2 ? ["no-nonfar-adjacency"] : []),
          ...(causeBits & 4 ? ["nonfar-neighbors-miss-components"] : []),
          ...(causeBits & 8 ? ["constrained"] : []),
          ...(causeBits & 16 ? ["invalid-masters"] : [])],
        neighbors: Array.from({ length: 6 }, (_none, direction) => ({ direction,
          slot: words[at + 6 + direction] ?? 0xffff_ffff,
          phi: (words[at + 12 + direction] ?? 0xffff_ffff) === 0xffff_ffff
            ? undefined : float(words[at + 12 + direction]!),
          mask: (words[at + 18 + direction] ?? 0) & 7,
        })),
      };
    });
    const invalidOutsideReach = words[base + 9] ?? 0;
    const invalidOutsideIndependent = words[base + 12] ?? 0;
    return { name, header: { unresolved: words[base] ?? 0,
      noAdjacency: words[base + 1] ?? 0,
      noNonfarAdjacency: words[base + 2] ?? 0,
      nonfarNeighborsMissComponents: words[base + 3] ?? 0,
      constrained: words[base + 4] ?? 0,
      invalidMasters: words[base + 5] ?? 0, captured,
      demandedMasters: words[base + 7] ?? 0,
      dilatedSupport: words[base + 8] ?? 0,
      invalidOutsideReach,
      minimumInvalidOutsideAbsPhi: (words[base + 10] ?? 0xffff_ffff) === 0xffff_ffff
        ? undefined : float(words[base + 10]!),
      maximumInvalidOutsideAbsPhi: float(words[base + 11] ?? 0),
      invalidOutsideIndependent,
      invalidOutsideConstrained: invalidOutsideReach - invalidOutsideIndependent }, records };
  });
}
