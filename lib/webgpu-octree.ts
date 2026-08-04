import type { SceneDescription } from "./model";
import { WebGPUOctreeFineSeedAdapter } from "./webgpu-octree-fine-seed-adapter";
import {
  lookupOctreeOwnerPage,
  OCTREE_OWNER_ARENA_PROJECTION_WORDS,
  OCTREE_OWNER_PAGE_LOOKUP_STATUS,
  WebGPUOctreeSimulationOwnerPages,
  type OctreeOwnerLeafSize,
  type OctreeOwnerPagePlan,
} from "./webgpu-octree-owner-pages";
import { PassBroker } from "./webgpu-pass-broker";
import { planOctreeSurfaceStateAllocation } from "./octree-surface-allocation";
import { planOctreeAnalyticBootstrapBounds } from "./octree-analytic-bootstrap";
import { WebGPUOctreeAnalyticBootstrapWorklist } from "./webgpu-octree-analytic-bootstrap";
import { combineInitialBrickWet, damBreakBoxContains, initialFluidBrickContainsCell,
  initialFluidBrickSignedDistanceAtCell, initialFluidBrickUnionBounds,
  sceneDamBreakBox, sceneDamBreakFractions, sceneDamBreakIsOffsetFromCorner } from "./initial-fluid";
import { integratedInflowVolume } from "./inflow-boundary";
import { signedDistanceFromVolume } from "./volume-signed-distance";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import { WebGPUQuadtreeSurfaceState, type SurfaceInflowState } from "./webgpu-quadtree-builder";
import { OctreeSparseBrickWorld } from "./webgpu-octree-sparse-bricks";
import type { SparseScenePrimitiveUpdate } from "./webgpu-sparse-scene-proxies";
import {
  DEFAULT_OCTREE_COARSE_BACKEND,
  resolveOctreeCoarseDynamics,
  type OctreeCoarseDynamicsConfiguration,
} from "./octree-coarse-backend";
import { WebGPUOctreeLosassoCoarseBackend } from "./webgpu-octree-losasso-backend";
import { WebGPUOctreeLosassoReadyCommit } from "./webgpu-octree-losasso-ready-commit";
import {
  makeOctreeLosassoCoarsePhiSampleWGSL,
  WebGPUOctreeLosassoCoarsePhiExchange,
  type WebGPUOctreeLosassoCoarsePhiInput,
} from "./webgpu-octree-losasso-coarse-phi";
import { OCTREE_LOSASSO_COARSE_PHI_MAGIC } from "./webgpu-octree-losasso-coarse-phi.wgsl";
import { WebGPUOctreeLosassoFineTransport } from "./webgpu-octree-losasso-fine-transport";
import { WebGPUOctreeLosassoRowMotion } from "./webgpu-octree-losasso-row-motion";
import { WebGPUOctreeLosassoConditionedOperator } from
  "./webgpu-octree-losasso-conditioned-operator";
import {
  FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES,
  GPUFluidBrickResidency,
  planFineSeedCandidateResidencyPools,
} from "./webgpu-fluid-brick-residency";
import type { GPUInitializationTask } from "./gpu-initialization";
import {
  planGPUShaderCapabilities,
  planGPUShaderTasks,
  type GPUShaderCapabilityPlan,
  type GPUShaderTaskDefinition,
} from "./gpu-shader-plan";
import {
  fetchGeneratedOctreePowerCatalog,
  decodeGeneratedOctreePowerCatalog,
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
  type GeneratedOctreePowerCatalogViews,
} from "./generated/octree-power-catalog";
import { WebGPUOctreePowerDescriptor } from "./webgpu-octree-power-descriptor";
import { OCTREE_POWER_NEIGHBOR_DIRECTIONS } from "./octree-power-descriptor";
import { WebGPUOctreePowerTopology } from "./webgpu-octree-power-topology";
import {
  structuredVelocityRowCapacityForBindingLimit,
  WebGPUDirectStructuredVelocityAuthority,
} from "./webgpu-octree-structured-velocity-gpu";
import { WebGPUStructuredBoundaryCoefficients } from "./webgpu-octree-structured-boundary";
import { WebGPUStructuredVelocityDynamics } from "./webgpu-octree-structured-dynamics";
import {
  planOctreeAirVelocitySupport,
} from "./webgpu-octree-air-velocity-support";
import {
  octreeAirSupportFootprintCapacity,
  WebGPUOctreeAirVelocitySupportProducer,
} from "./webgpu-octree-air-velocity-support-gpu";
import { WebGPUOctreeSolidVertexSdf } from "./webgpu-octree-solid-vertex-sdf";
import {
  normalizeOctreeSection43BoundarySmoothing,
  type OctreeFirstOrderSPDVCycle,
} from "./webgpu-octree-section43-contract";
import {
  WebGPUOctreePipelinedMGPCG,
  type OctreePipelinedMGPCGVectors,
  type OctreePipelinedWorksetLinearOperator,
} from "./webgpu-octree-pipelined-mgpcg";
import { WebGPUOctreeSection43HybridPreconditioner } from
  "./webgpu-octree-section43-preconditioner";
import {
  spgridRowCapacityForBindingLimit,
  type OctreeSPGridAccurateAuthority,
  WebGPUOctreeSPGridVCycle,
} from "./webgpu-octree-spgrid-vcycle";
import { WebGPUOctreeTopologyEpoch } from "./webgpu-octree-topology-epoch";
import {
  planOctreeSolveTail,
  type OctreeSolveTailPolicy,
} from "./octree-solve-tail-policy";
import {
  OctreeWorkAccounting,
} from "./webgpu-octree-work-accounting";
import { WebGPUOctreeCoarseLevelSet } from "./webgpu-octree-coarse-levelset";
import { WebGPUOctreePowerCoarseLevelSet } from "./webgpu-octree-power-coarse-levelset";
import { WebGPUOctreeCoarseSummary } from "./webgpu-octree-coarse-summary";
import { WebGPUFineToCoarseLevelSet } from "./webgpu-octree-fine-to-coarse-levelset";
import { planFineLevelSetBricks } from "./octree-fine-levelset-bricks";
import {
  WebGPUFineLevelSetBricks,
  type WebGPUFineLevelSetBrickSource,
} from "./webgpu-octree-fine-levelset-bricks";
import {
  maximumFineLevelSetJFAStride,
  WebGPUFineLevelSetRedistance,
} from "./webgpu-octree-fine-levelset-redistance";
import {
  planFineLevelSetGPUTransportPasses,
  WebGPUFineLevelSetTransport,
} from "./webgpu-octree-fine-levelset-transport";
import { WebGPUFineLevelSetVolumeCorrection } from "./webgpu-octree-fine-levelset-volume";
import {
  createGPULogicalActivityAdoptionContext,
  type GPULogicalActivityAdoptionContext,
} from "./gpu-logical-activity-adoption";
import { performanceShaderVariant } from "./stores/performance-instrumentation-store";
import { FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE, FINE_LEVELSET_SUMMARY_ENTRY_WORDS,
  planFineLevelSetGPUSummaries,
  WebGPUFineLevelSetSummaries } from "./webgpu-octree-fine-levelset-summary";
import {
  planFineLevelSetBandFineCells,
  planFineLevelSetCapacityDilationBrickRings,
  planFineLevelSetTopologyBand,
  WebGPUFineLevelSetLeafSeeds,
  WebGPUFineLevelSetTopology,
} from "./webgpu-octree-fine-levelset-topology";

type OctreePipelineVariants = { full: GPUComputePipeline; delta: GPUComputePipeline };

export const OCTREE_PROJECTION_ACTIVITY_MODULE_ID = "octree/projection-topology";

export const OCTREE_PROJECTION_BASE_ENTRY_POINTS = [
  "rasterizeSolids", "resetTopology", "refineTopology", "balanceTopology",
  "rasterizeSolidsDelta", "resetTopologyDelta", "refineTopologyDelta", "balanceTopologyDelta",
  "stampFrontierAttempt", "beginFrontier", "classifyFrontierCandidates", "classifyFrontierCandidatesDelta",
  "prefixFrontierCandidateBlocks", "prefixFrontierCandidateBlocksDelta",
  "emitFrontierCandidates", "emitFrontierCandidatesDelta",
  "prepareFrontierDispatch", "sortFrontierCandidatesLocal",
  "classifyFrontierCarry",
  "scanFrontierCarryBlocks", "prefixFrontierCarryBlocks", "mergeFrontierRows", "finalizeFrontier",
  "prepareRowDelta", "classifyRowDelta",
  "finalizeRowDeltaClassification", "scanDirtyRowDeltaBlocks", "prefixDirtyRowDeltaBlocks",
  "scatterDirtyRowDelta", "markRowDeltaRing", "markRowDeltaRingBlocks",
  "scanAffectedRowDeltaBlocks",
  "prefixAffectedRowDeltaBlocks", "compactRowDelta", "publishRowDelta", "publishReusedRowDelta",
  "planLeaves", "scanLeafBlocks", "emitLeaves",
  "classifyTopologyTileSignature", "buildDirtyTileDelta", "buildDirtyFrontierDelta",
  "advanceGradingRound",
] as const;

export const OCTREE_PROJECTION_VARIANT_ENTRY_POINTS = [
  "refineTopologyCoarse", "refineTopologyCoarseDelta",
  "balanceTopologyCoarse", "balanceTopologyCoarseDelta",
  "sortFrontierCandidates",
] as const;

export const OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS = Object.freeze([
  // The instrumentation transform requires every listed entry point to exist in
  // the module, which is the exhaustiveness guard. A source-gated experiment is
  // absent from the unflagged shader by construction, so it joins the list only
  // when its own variable puts it in the shader.
  ...OCTREE_PROJECTION_BASE_ENTRY_POINTS.filter((entryPoint) =>
    entryPoint !== "advanceGradingRound" || octreeGradingFixpointEnabled()),
  ...OCTREE_PROJECTION_VARIANT_ENTRY_POINTS,
]);

export type OctreeProjectionActivityEntryPoint =
  typeof OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS[number];

const projectionActivityTaskName = (entryPoint: string) => entryPoint
  .replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();
const projectionActivityLabel = (entryPoint: string) => entryPoint
  .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
  .replace(/^./, (character) => character.toUpperCase());

export const OCTREE_PROJECTION_ACTIVITY_TASKS = Object.freeze(Object.fromEntries(
  OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS.map((entryPoint) => {
    const task = projectionActivityTaskName(entryPoint);
    return [entryPoint, Object.freeze({
      task,
      id: `gpu.physics.coarse-grid.${task}`,
      label: `Coarse grid · ${projectionActivityLabel(entryPoint)}`,
      phaseId: "coarse-grid",
    })];
  }),
)) as Readonly<Record<OctreeProjectionActivityEntryPoint, Readonly<{
  task: string; id: string; label: string; phaseId: "coarse-grid";
}>>>;

/** Structured world-boundary bits are x-/x+/y-/y+/z-/z+. */
function structuredClosedBoundaryMask(closedTop: boolean): number {
  return closedTop ? 0b11_1111 : 0b11_0111;
}

/** Ordered, individually fenced t=0 authority checkpoints.
 * Aanjaneya et al. (2017), Section 5 p.8, first constructs regular octree-face
 * neighborhoods, augments T-junctions with local Delaunay tetrahedra, extends
 * air velocities from physical closest points, and only then interpolates the
 * result back to power faces. */
export const OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES = [
  { id: "cold-topology", label: "Cold octree topology" },
  { id: "structured-authority", label: "Direct structured velocity and pressure authority" },
  { id: "surface-global-fine", label: "Surface and global-fine redistance publication" },
  { id: "sparse-render-world", label: "Sparse render world publication" },
] as const;
export type OctreeInitialSparseAuthorityPhaseId = typeof OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES[number]["id"];

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
  const nodeFs = "node:fs/promises";
  const { readFile } = await import(nodeFs) as { readFile(path: URL): Promise<Uint8Array> };
  const bytes = await readFile(url);
  return decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
}

function loadGeneratedOctreePowerCatalog(): Promise<GeneratedOctreePowerCatalogViews> {
  // A failed load must not be cached, or one transient error poisons the
  // session; clear the memo so the next build retries.
  return generatedOctreePowerCatalog ??= readGeneratedOctreePowerCatalog()
    .catch((error: unknown) => { generatedOctreePowerCatalog = undefined; throw error; });
}
interface OctreePipelineCacheEntry {
  base: GPUComputePipeline[];
  frontierSort: GPUComputePipeline[];
  refine: Map<number, OctreePipelineVariants>;
  refineCoarse: Map<number, OctreePipelineVariants>;
  balanceCoarse: Map<number, OctreePipelineVariants>;
}
const octreePipelineCache = new WeakMap<GPUDevice, Map<string, OctreePipelineCacheEntry>>();
const octreeDiagnosticPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();

/**
 * Paper-compatible adaptive boundary policy.
 *
 * The default keeps interface/inflow/hysteresis protection unchanged, but
 * permits a boundary-crossing leaf to remain coarse until liquid approaches
 * within the authored interface band. The authored method option can retain
 * unconditional unit-cell wall and terrain refinement as a control.
 */
/** CPU mirror of the shader's final boundary branch for unit tests/tooling. */
export function octreeFluidGatedBoundaryWouldRefine(input: {
  readonly boundaryIntersects: boolean;
  readonly liquidProximityProtected: boolean;
  readonly minimumPhi: number;
  readonly protectionWidth: number;
  readonly fluidGated: boolean;
}): boolean {
  if (!Number.isFinite(input.minimumPhi)
      || !Number.isFinite(input.protectionWidth)
      || input.protectionWidth < 0) {
    throw new RangeError("Boundary refinement phi and protection width must be finite");
  }
  return input.liquidProximityProtected
    || (input.boundaryIntersects
      && (!input.fluidGated || input.minimumPhi <= input.protectionWidth));
}

export interface OctreeProjectionOptions {
  /** Construction-time coarse backend seam; never represented in WGSL. */
  coarseDynamics?: OctreeCoarseDynamicsConfiguration;
  maximumLeafSize?: 2 | 4 | 8 | 16 | 32;
  /** Default liquid-proximity boundary gate; false selects the unconditional control. */
  fluidGatedBoundaryRefinement?: boolean;
  /** Renderer-owned refinement of authored-environment bricks. */
  environmentBrickRefinementLevels?: number;
  /** 0 = finest cells everywhere; 1 = full distance-graded coarsening. */
  adaptivity?: number;
  /** Pure-phase cells farther from liquid/solid interfaces than this finest-cell band may remain coarse. */
  interfaceRefinementBandCells?: number;
  /**
   * Number of candidate-cell widths retained at each dyadic pressure level
   * around the surface. One preserves the sharpest legal 2:1 transition;
   * larger values create progressively wider intermediate-level shells.
   */
  surfaceRefinementGradingLayers?: number;
  /**
   * Half-width of the Section 5 high-resolution surface-tracking band, in
   * finest octree cells. The product master control supplies the same authored
   * reach as the pressure band above; the separate option remains only for
   * diagnostic fault injection. An unset value follows
   * `interfaceRefinementBandCells`.
   */
  fineLevelSetBandCells?: number;
  /** Authoritative domain-global Section 5 narrow-band factor. */
  globalFineLevelSetFactor?: 1 | 4 | 8;
  /** Explicit physical brick cap for the global factor-1/factor-4/factor-8 publication. */
  globalFineLevelSetMaximumBricks?: number;
  /** Advanced safety override for the compact pressure-row arena. */
  pressureRowCapacity?: number;
}

/** Allocation milestones owned by the octree resource graph. Keep these next
 * to the constructor that performs the work so product progress cannot drift
 * away from the actual GPU boundaries. */
export const OCTREE_ALLOCATION_STAGES = Object.freeze([
  "Plan octree domain and capacity",
  "Allocate sparse brick-world resources",
  "Allocate topology residency scheduler",
  "Allocate analytic bootstrap and surface state",
  "Allocate topology owners and pressure rows",
  "Build octree layouts and bind groups",
  "Allocate fine-interface seed resources",
  "Allocate global fine level-set pages",
  "Finalize octree resource graph",
] as const);

export type OctreeAllocationProgress = (label: string, completed: number, total: number) => void;

/** The seven data-domain boundaries of the collapsed recurring frame. */
export const OCTREE_ENGINE_PHASES = [
  "structureEpoch",
  "rowEngineA",
  "solveEngine",
  "rowEngineB",
  "brickEngineA",
  "closestPointWaves",
  "brickEngineB",
] as const;
export type OctreeEnginePhase = typeof OCTREE_ENGINE_PHASES[number];

/** Fine-grained checkpoints for direct structured attribution. */
export const OCTREE_FINE_SEMANTIC_PHASES = [
  "powerDescriptorTopology",
  "structuredAdvectionBoundaryRhs",
  "structuredVolumeCapture",
  "finalPressureRowAssembly",
  "mgpcgSolve",
  "structuredProjection",
  "structuredProjectionTail",
  "finePreparation",
  "fineTransport",
  "fineTopology",
  "fineRedistance",
  "fineRestriction",
] as const;
export type OctreeFineSemanticPhase = typeof OCTREE_FINE_SEMANTIC_PHASES[number];
/** Semantic checkpoints consumed by the generic adjacent-boundary trace.
 * Instrumented runs use numerical seams by default. Set
 * `FLUID_ENGINE_SPLIT=collapsed` only for the seven-bucket engine view. */
export type OctreeSemanticPhase = OctreeEnginePhase | OctreeFineSemanticPhase;
export type OctreeSemanticBoundary = (
  phase: OctreeSemanticPhase,
  encoder: GPUCommandEncoder,
) => GPUCommandEncoder;

interface PendingFinePublication {
  readonly topology: WebGPUFineLevelSetTopology;
  readonly redistance: WebGPUFineLevelSetRedistance;
  readonly volume?: WebGPUFineLevelSetVolumeCorrection;
  readonly transport?: WebGPUFineLevelSetTransport | WebGPUOctreeLosassoFineTransport;
  readonly target: WebGPUFineLevelSetBrickSource;
  readonly targetIsA: boolean;
  readonly redistanceBandCells: number;
  readonly maximumDisplacementFineCells: number;
  readonly warmClosestPoints: boolean;
}

/** Read at encode time so benchmark processes can select attribution without
 * changing construction or numerical behavior. */
export function octreeFineEngineSplitsEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_ENGINE_SPLIT !== "collapsed";
}

/**
 * Whole-page owner fills inside the split materializer.
 *
 * Selects the page-claimed form of splitLeaf over the per-cell walk. Both write
 * the same cells the same words through the same idempotent atomicMin; the page
 * form only stops the inner loop from recomputing a page-invariant constant 512
 * times through three runtime integer divisions. Off restores the original walk
 * verbatim, including its per-cell membership load, so an interleaved A/B can
 * score them from one shader module in separate processes.
 */
export function octreeGradingPageFillEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_GRADING_PAGE_FILL !== "0";
}

/**
 * Losing askers take one page of the split they asked for.
 *
 * Requires the page-claimed materializer above; with the page fill off this has
 * nothing to divide. The write set is unchanged either way -- the same cells
 * receive the same words through the same idempotent atomicMin -- so this only
 * chooses how many lanes carry it.
 */
export function octreeGradingSplitHelpersEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_GRADING_SPLIT_HELPERS !== "0";
}

/**
 * Restore the per-cell membership load inside the split materializer.
 *
 * Off by default because the bit it preserves is provably clear on every cell
 * the topology candidate view can address; see splitOwnerWord. Kept as an arm
 * so the interleaved A/B can price the load rather than assert it.
 */
export function octreeGradingMembershipLoadEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_GRADING_MEMBERSHIP_LOAD === "1";
}

/**
 * Retire grading rounds once the balance closure has converged.
 *
 * A balance round writes owner state ONLY through a split, so a round that
 * claims none leaves the next round's input bit-identical and the next round
 * provably claims none either. Skipping is therefore exact, not conservative.
 *
 * Default OFF and gated at the SOURCE, not by an override: with the variable
 * unset `octreeProjectionShader` is byte-identical to the unflagged tree, the
 * `advanceGradingRound` pipeline is not reachable, and the owner arena keeps
 * its 16-word control block. Nothing about the unflagged path can move.
 */
export function octreeGradingFixpointEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_GRADING_FIXPOINT === "1";
}

/**
 * Size the topology delta tile by the largest leaf the domain can hold.
 *
 * Default OFF: it currently leaves symmetric-expansion inert. See the comment
 * at the assignment for what has already been ruled out.
 */
export function octreeTopologyTileClampEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_TOPOLOGY_TILE_CLAMP === "1";
}

export function octreeSparseWorldRequired(
  hasTerrain: boolean,
  rigidBodyCount: number,
): boolean {
  return hasTerrain || rigidBodyCount > 0;
}

export interface OctreeDensePhiReleaseState {
  globalFineBootstrapped: boolean;
  coarseProjectionGroupsActive: boolean;
  fineSeedCoarseNative: boolean;
  topologyUsesFineSeedCandidates: boolean;
  compactRendererSourceReady: boolean;
  incompatibleDenseConsumer: boolean;
}

/** All recurring consumers must complete their bind-group handoff before destroy. */
export function octreeDensePhiReleaseReady(state: OctreeDensePhiReleaseState): boolean {
  return state.globalFineBootstrapped
    && state.coarseProjectionGroupsActive
    && state.fineSeedCoarseNative
    && state.topologyUsesFineSeedCandidates
    && state.compactRendererSourceReady
    && !state.incompatibleDenseConsumer;
}

export interface OctreePressureCapacityPlan {
  rowCapacity: number;
  pressureBytes: number;
  headerBytes: number;
}

/**
 * The paper's Section 5 interpolant needs a complete local octree
 * neighbourhood wherever a trajectory can sample velocity.  The generated
 * interior Delaunay catalog has no clipped/ghost sites outside the domain, so
 * the bounded production extension keeps closed walls in the regular
 * unit-cell case.  Three cells match the paper's Section 4.3 boundary-band
 * scale; Section 5 requires the advection band to contain the trajectory, so
 * the configured interface support is used whenever it is larger.
 */
export const OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS = 3;

export interface OctreePowerBoundaryStripPlan {
  readonly widthCells: number;
  /** Exact number of finest cells in the union of the selected closed-wall strips. */
  readonly unitCellUpperBound: number;
  /** Exact number of 8-cubed owner pages intersected by that union. */
  readonly ownerPageUpperBound: number;
}

export function planOctreePowerBoundaryStrip(
  dims: { nx: number; ny: number; nz: number },
  interfaceBandCells: number,
  closedTop = false,
): OctreePowerBoundaryStripPlan {
  const dimensions = [dims.nx, dims.ny, dims.nz];
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Octree power boundary-strip dimensions must be positive safe integers");
  }
  if (!Number.isFinite(interfaceBandCells) || interfaceBandCells < 0) {
    throw new RangeError("Octree power boundary-strip interface band must be finite and non-negative");
  }
  const widthCells = Math.max(OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS, Math.ceil(interfaceBandCells));
  const lowWidths = [widthCells, widthCells, widthCells];
  const highWidths = [widthCells, closedTop ? widthCells : 0, widthCells];
  const interiorCells = dimensions.map((value, axis) => Math.max(0,
    value - Math.min(value, lowWidths[axis]) - Math.min(value, highWidths[axis])));
  const volume = dimensions[0] * dimensions[1] * dimensions[2];
  const interiorVolume = interiorCells[0] * interiorCells[1] * interiorCells[2];

  const pageDimensions = dimensions.map((value) => Math.ceil(value / 8));
  const interiorPages = dimensions.map((value, axis) => {
    const first = Math.ceil(Math.min(value, lowWidths[axis]) / 8);
    // A partial terminal page is interior when that side is open; with a
    // closed high wall, only complete pages ending before its strip qualify.
    const lastExclusive = highWidths[axis] === 0
      ? Math.ceil(value / 8)
      : Math.floor((value - Math.min(value, highWidths[axis])) / 8);
    return Math.max(0, lastExclusive - first);
  });
  return {
    widthCells,
    unitCellUpperBound: volume - interiorVolume,
    ownerPageUpperBound: pageDimensions[0] * pageDimensions[1] * pageDimensions[2]
      - interiorPages[0] * interiorPages[1] * interiorPages[2],
  };
}

export interface OctreeProjectionPipelineReachability {
  readonly solidRasterization: boolean;
  readonly localFrontierCandidateSort: boolean;
  readonly cooperativeRowDeltaRing: boolean;
}

/** Compile only entry points reachable from the immutable solver configuration. */
export function octreeProjectionPipelineRequired(
  entryPoint: string,
  config: OctreeProjectionPipelineReachability,
): boolean {
  // These entry points exist only as specialization templates. Every encoded
  // fine refinement selects a target-size pipeline from `refineLevelPipelines`.
  if (entryPoint === "refineTopology" || entryPoint === "refineTopologyDelta") {
    return false;
  }
  if (entryPoint === "advanceGradingRound") {
    return octreeGradingFixpointEnabled();
  }
  if (entryPoint === "rasterizeSolids" || entryPoint === "rasterizeSolidsDelta") {
    return config.solidRasterization;
  }
  if (entryPoint === "sortFrontierCandidatesLocal") {
    return config.localFrontierCandidateSort;
  }
  if (entryPoint === "markRowDeltaRing") {
    return config.cooperativeRowDeltaRing;
  }
  if (entryPoint === "markRowDeltaRingBlocks") {
    return !config.cooperativeRowDeltaRing;
  }
  return true;
}

export interface OctreeLeafFrontierAllocationPlan {
  cellCount: number;
  listCapacity: number;
  /** Third immutable-sort stream used only while merging dirty candidates. */
  candidateOffsetWords: number;
  /** Fixed control header for the exact old/new row-delta transaction. */
  rowDeltaControlOffsetWords: number;
  /** Exact `newRow -> oldRow|INVALID` map. */
  rowDeltaNewToOldOffsetWords: number;
  /** Exact `oldRow -> newRow|INVALID` map. */
  rowDeltaOldToNewOffsetWords: number;
  /** Rows whose exact identity was added this generation. */
  rowDeltaDirtyRowsOffsetWords: number;
  /** Dirty rows plus their exact current/retired one-ring. */
  rowDeltaAffectedRowsOffsetWords: number;
  candidateBytes: number;
  allocatedBytes: number;
}

export interface OctreePowerRowIdentity {
  readonly cell: number;
  readonly size: number;
  readonly morton: number;
}

export interface OctreePowerRowDeltaOracle {
  readonly newToOld: readonly number[];
  readonly oldToNew: readonly number[];
  readonly dirtyRows: readonly number[];
  readonly carried: number;
  readonly added: number;
  readonly retired: number;
}

export interface OctreeTopologyLeafCensus {
  readonly generation: number;
  readonly residentOwnerPages: number;
  readonly topologyLeaves: number;
  readonly representedCells: number;
  readonly leafCountsBySize: Readonly<Record<string, number>>;
  /** Coarse leaf origins per finest-grid Y layer; diagnostic spatial profile. */
  readonly coarseLeafCountsByOriginY: readonly number[];
  /** Leaf-size histograms intersecting each three-cell world-boundary strip. */
  readonly boundaryStripLeafCountsBySize: Readonly<Record<
    "xLow" | "xHigh" | "yLow" | "yHigh" | "zLow" | "zHigh",
    Readonly<Record<string, number>>
  >>;
}

export function censusOctreeTopologyLeaves(
  ownerWords: ArrayLike<number>,
  plan: OctreeOwnerPagePlan,
  maximumLeafSize: OctreeOwnerLeafSize,
): OctreeTopologyLeafCensus {
  if (ownerWords.length < plan.allocatedWords) {
    throw new RangeError("Octree topology census owner arena is truncated");
  }
  const counts = new Map<number, number>();
  const coarseLeafCountsByOriginY = new Array<number>(plan.dimensions[1]).fill(0);
  const boundaryCounts = {
    xLow: new Map<number, number>(), xHigh: new Map<number, number>(),
    yLow: new Map<number, number>(), yHigh: new Map<number, number>(),
    zLow: new Map<number, number>(), zHigh: new Map<number, number>(),
  };
  const addBoundary = (face: keyof typeof boundaryCounts, size: number) => {
    const faceCounts = boundaryCounts[face];
    faceCounts.set(size, (faceCounts.get(size) ?? 0) + 1);
  };
  const identities = new Set<string>();
  let representedCells = 0;
  for (let z = 0; z < plan.dimensions[2]; z += 1) {
    for (let y = 0; y < plan.dimensions[1]; y += 1) {
      for (let x = 0; x < plan.dimensions[0]; x += 1) {
        const owner = lookupOctreeOwnerPage(
          ownerWords, plan, [x, y, z], maximumLeafSize,
        );
        if ((owner.status & (OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing
          | OCTREE_OWNER_PAGE_LOOKUP_STATUS.invalid)) !== 0) continue;
        const identity = `${owner.origin[0]},${owner.origin[1]},${owner.origin[2]},${owner.size}`;
        if (identities.has(identity)) continue;
        identities.add(identity);
        counts.set(owner.size, (counts.get(owner.size) ?? 0) + 1);
        if (owner.size > 1) coarseLeafCountsByOriginY[owner.origin[1]]! += 1;
        const high = owner.origin.map((coordinate) => coordinate + owner.size);
        const strip = OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS;
        if (owner.origin[0] < strip) addBoundary("xLow", owner.size);
        if (high[0]! > plan.dimensions[0] - strip) addBoundary("xHigh", owner.size);
        if (owner.origin[1] < strip) addBoundary("yLow", owner.size);
        if (high[1]! > plan.dimensions[1] - strip) addBoundary("yHigh", owner.size);
        if (owner.origin[2] < strip) addBoundary("zLow", owner.size);
        if (high[2]! > plan.dimensions[2] - strip) addBoundary("zHigh", owner.size);
        representedCells += owner.size ** 3;
      }
    }
  }
  const leafCountsBySize = Object.fromEntries(
    [...counts.entries()]
      .sort(([left], [right]) => left - right)
      .map(([size, count]) => [String(size), count]),
  );
  const boundaryStripLeafCountsBySize = Object.fromEntries(
    Object.entries(boundaryCounts).map(([face, faceCounts]) => [face,
      Object.freeze(Object.fromEntries([...faceCounts.entries()]
        .sort(([left], [right]) => left - right)
        .map(([size, count]) => [String(size), count]))),
    ]),
  ) as OctreeTopologyLeafCensus["boundaryStripLeafCountsBySize"];
  return Object.freeze({
    generation: Number(ownerWords[7] ?? 0) >>> 0,
    residentOwnerPages: Number(ownerWords[1] ?? 0) >>> 0,
    topologyLeaves: identities.size,
    representedCells,
    leafCountsBySize: Object.freeze(leafCountsBySize),
    coarseLeafCountsByOriginY: Object.freeze(coarseLeafCountsByOriginY),
    boundaryStripLeafCountsBySize: Object.freeze(boundaryStripLeafCountsBySize),
  });
}

export const OCTREE_PRESSURE_ROW_ONE_RING_DIRECTION_COUNT = 18;

/**
 * CPU oracle for the pressure row-delta publication's face-and-edge one-ring.
 *
 * `initiallyAffected` is an immutable snapshot. Reading only that snapshot is
 * essential: reading the output while walking rows would turn a chain into an
 * unbounded in-dispatch flood. Each row supplies the shader's exact eighteen
 * face/edge neighbours, with -1 for an out-of-domain direction.
 */
export function expandOctreePressureRowAffectedOneRing(
  initiallyAffected: readonly boolean[],
  directionalNeighbours: readonly (readonly number[])[],
  membershipChanged = false,
): readonly boolean[] {
  if (directionalNeighbours.length !== initiallyAffected.length
    || directionalNeighbours.some((neighbours) =>
      neighbours.length !== OCTREE_PRESSURE_ROW_ONE_RING_DIRECTION_COUNT)) {
    throw new RangeError("Pressure row one-ring requires exactly eighteen directions per row");
  }
  for (const neighbours of directionalNeighbours) for (const neighbour of neighbours) {
    if (!Number.isSafeInteger(neighbour)
      || neighbour < -1 || neighbour >= initiallyAffected.length) {
      throw new RangeError("Pressure row one-ring neighbour is out of range");
    }
  }
  if (membershipChanged) return initiallyAffected.map(() => true);
  return initiallyAffected.map((affected, row) => affected
    || directionalNeighbours[row]!.some((neighbour) =>
      neighbour >= 0 && initiallyAffected[neighbour]!));
}

/** CPU oracle for the shader's 4³ lanes × 2³ cells cold candidate lattice. */
export function enumerateOctreeFrontierCandidateLattice(
  dimensions: readonly [number, number, number],
): readonly number[] {
  if (!dimensions.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("Octree candidate dimensions must be positive integers");
  }
  const [nx, ny, nz] = dimensions;
  const cells: number[] = [];
  for (let bz = 0; bz < Math.ceil(nz / 8); bz += 1) {
    for (let by = 0; by < Math.ceil(ny / 8); by += 1) {
      for (let bx = 0; bx < Math.ceil(nx / 8); bx += 1) {
        for (let lz = 0; lz < 4; lz += 1) {
          for (let ly = 0; ly < 4; ly += 1) {
            for (let lx = 0; lx < 4; lx += 1) {
              for (let octant = 0; octant < 8; octant += 1) {
                const x = bx * 8 + lx * 2 + (octant & 1);
                const y = by * 8 + ly * 2 + ((octant >> 1) & 1);
                const z = bz * 8 + lz * 2 + ((octant >> 2) & 1);
                if (x < nx && y < ny && z < nz) cells.push(x + nx * (y + ny * z));
              }
            }
          }
        }
      }
    }
  }
  return cells;
}

/** CPU oracle for the GPU's sorted exact `(level, Morton)` row merge. */
export function mergeOctreePowerRowIdentities(
  previous: readonly OctreePowerRowIdentity[],
  current: readonly OctreePowerRowIdentity[],
  authorityDirtyRows: ReadonlySet<number> = new Set(),
): OctreePowerRowDeltaOracle {
  const valid = (row: OctreePowerRowIdentity) => Number.isSafeInteger(row.cell) && row.cell >= 0
    && Number.isSafeInteger(row.size) && row.size > 0 && (row.size & (row.size - 1)) === 0
    && Number.isSafeInteger(row.morton) && row.morton >= 0;
  if (!previous.every(valid) || !current.every(valid)) throw new RangeError("Power row identities must be finite integer octree keys");
  const key = (row: OctreePowerRowIdentity) => [Math.log2(row.size), row.morton] as const;
  const less = (a: OctreePowerRowIdentity, b: OctreePowerRowIdentity) => {
    const ka = key(a), kb = key(b);
    return ka[0] < kb[0] || (ka[0] === kb[0] && ka[1] < kb[1]);
  };
  const ordered = (rows: readonly OctreePowerRowIdentity[]) =>
    rows.every((row, index) => index === 0 || less(rows[index - 1], row));
  if (!ordered(previous) || !ordered(current)) throw new RangeError("Power row identities must be strictly sorted by (level, Morton)");
  const newToOld = new Array(current.length).fill(-1);
  const oldToNew = new Array(previous.length).fill(-1);
  const dirtyRows: number[] = [];
  let oldRow = 0, newRow = 0, carried = 0, added = 0, retired = 0;
  while (oldRow < previous.length || newRow < current.length) {
    const oldIdentity = previous[oldRow], newIdentity = current[newRow];
    if (oldIdentity && newIdentity && oldIdentity.cell === newIdentity.cell && oldIdentity.size === newIdentity.size) {
      newToOld[newRow] = oldRow; oldToNew[oldRow] = newRow; carried += 1;
      if (authorityDirtyRows.has(newRow)) dirtyRows.push(newRow);
      oldRow += 1; newRow += 1;
    } else if (!oldIdentity || (newIdentity && less(newIdentity, oldIdentity))) {
      dirtyRows.push(newRow); added += 1; newRow += 1;
    } else {
      retired += 1; oldRow += 1;
    }
  }
  if (current.length !== carried + added || current.length !== previous.length + added - retired) {
    throw new Error("Power row delta transaction count mismatch");
  }
  return { newToOld, oldToNew, dirtyRows, carried, added, retired };
}

/** Allocate only the sorted A/B publication, dirty candidate stream, and row delta.
 *
 * The ten-word header deliberately separates active authority (0..3) from the
 * inactive candidate transaction (4..9).  Candidate validation may populate
 * the latter at the tail of substep N, but only the coupled owner/frontier
 * boundary commit may change the active selector/generation at N+1.
 */
export function planOctreeLeafFrontierAllocation(
  cellCount: number,
  rowCapacity: number,
): OctreeLeafFrontierAllocationPlan {
  if (!Number.isSafeInteger(cellCount) || cellCount < 1) throw new Error("Octree frontier cell count must be a positive integer");
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) throw new Error("Octree frontier row capacity must be a positive integer");
  const listCapacity = Math.min(cellCount, rowCapacity);
  const candidateOffsetWords = 10 + 2 * listCapacity;
  const candidateBytes = listCapacity * 4;
  // The persistent frontier owns the row-delta publication because it is the
  // only stage that can still see both exact old and new `(cell,size)`
  // identities.  Sixteen control words are followed by two total maps and two
  // compact worklists.  Downstream descriptor/topology/face stages consume
  // these offsets directly; there is no topology-wide reuse branch.
  const rowDeltaControlOffsetWords = candidateOffsetWords + listCapacity;
  const rowDeltaNewToOldOffsetWords = rowDeltaControlOffsetWords + 16;
  const rowDeltaOldToNewOffsetWords = rowDeltaNewToOldOffsetWords + listCapacity;
  const rowDeltaDirtyRowsOffsetWords = rowDeltaOldToNewOffsetWords + listCapacity;
  const rowDeltaAffectedRowsOffsetWords = rowDeltaDirtyRowsOffsetWords + listCapacity;
  const allocatedBytes = (rowDeltaAffectedRowsOffsetWords + listCapacity) * 4;
  return {
    cellCount,
    listCapacity,
    candidateOffsetWords,
    rowDeltaControlOffsetWords,
    rowDeltaNewToOldOffsetWords,
    rowDeltaOldToNewOffsetWords,
    rowDeltaDirtyRowsOffsetWords,
    rowDeltaAffectedRowsOffsetWords,
    candidateBytes,
    allocatedBytes,
  };
}

export interface OctreeSolidCellAllocationPlan {
  allocatedBytes: number;
  denseBytes: number;
  savedBytes: number;
  hasDenseField: boolean;
}

/** Keep one valid `{ fraction, owner }` binding when a scene has no solids. */
export function planOctreeSolidCellAllocation(
  dims: { nx: number; ny: number; nz: number },
  hasTerrain: boolean,
  rigidBodyCount: number,
): OctreeSolidCellAllocationPlan {
  const denseBytes = Math.max(8, dims.nx * dims.ny * dims.nz * 8);
  const hasDenseField = hasTerrain || rigidBodyCount > 0;
  const allocatedBytes = hasDenseField ? denseBytes : 8;
  return { allocatedBytes, denseBytes, savedBytes: denseBytes - allocatedBytes, hasDenseField };
}

/** Resolve a physically planned capacity against 2D dispatch and binding limits. */
export function resolveGlobalFineBrickCapacity(
  defaultCapacity: number,
  override: number | undefined,
  maximumWorkgroupsPerDimension: number,
  transportWorkgroupQuantum = 64,
  maximumStorageBufferBindingSize = Number.MAX_SAFE_INTEGER,
  samplesPerBrick = 64,
  summaryLevelCount = 1,
  exactSummaryEntryCapacity?: number,
): number {
  if (!Number.isSafeInteger(defaultCapacity) || defaultCapacity < 1
    || !Number.isSafeInteger(maximumWorkgroupsPerDimension) || maximumWorkgroupsPerDimension < 1
    || !Number.isSafeInteger(transportWorkgroupQuantum) || transportWorkgroupQuantum < 1
    || !Number.isSafeInteger(maximumStorageBufferBindingSize) || maximumStorageBufferBindingSize < 16
    || !Number.isSafeInteger(samplesPerBrick) || samplesPerBrick < 1
    || !Number.isSafeInteger(summaryLevelCount) || summaryLevelCount < 1
    || (exactSummaryEntryCapacity !== undefined
      && (!Number.isSafeInteger(exactSummaryEntryCapacity) || exactSummaryEntryCapacity < 1))) {
    throw new RangeError("Global fine level-set default/device capacities must be positive integers");
  }
  // Per-brick work is tiled over x/y; dispatch shape no longer truncates the
  // physical capacity. Bound the largest persistent buffers instead: one
  // four-byte fine channel per sample and the compact sorted summary directory containing
  // at most one entry per resident brick per hierarchy level at load <= 0.5.
  const twoDimensionalDispatchMaximum = maximumWorkgroupsPerDimension ** 2;
  const payloadSafe = Math.floor(maximumStorageBufferBindingSize / (samplesPerBrick * 4));
  let summaryHashSlots = 1;
  while ((summaryHashSlots * 2) * 32 + 64 <= maximumStorageBufferBindingSize) summaryHashSlots *= 2;
  const summarySafe = exactSummaryEntryCapacity === undefined
    ? Math.floor(summaryHashSlots / (2 * summaryLevelCount))
    : Number.MAX_SAFE_INTEGER;
  const rawDeviceMaximum = Math.min(twoDimensionalDispatchMaximum, payloadSafe, summarySafe);
  const deviceMaximum = Math.floor(rawDeviceMaximum / transportWorkgroupQuantum) * transportWorkgroupQuantum;
  const configured = override ?? defaultCapacity;
  if (!Number.isSafeInteger(configured) || configured < 1) {
    throw new RangeError("Global fine level-set brick capacity must be a positive integer");
  }
  if (configured > deviceMaximum) {
    throw new RangeError(`Global fine level-set brick capacity ${configured} exceeds the sparse binding/dispatch limit ${deviceMaximum}; the physical narrow-band estimate is not reduced implicitly`);
  }
  if (exactSummaryEntryCapacity !== undefined) {
    let exactHashCapacity = 1;
    while (exactHashCapacity < exactSummaryEntryCapacity * 2) exactHashCapacity *= 2;
    if (64 + exactHashCapacity * 32 > maximumStorageBufferBindingSize) {
      throw new RangeError(`Global fine sparse summary requires ${64 + exactHashCapacity * 32} bytes, exceeding the storage binding limit ${maximumStorageBufferBindingSize}`);
    }
  }
  return configured;
}

export interface GlobalFineNarrowBandCapacityPlan {
  readonly logicalBrickCount: number;
  readonly maximumInterfaceAreaBricks: number;
  readonly bandLayers: number;
  readonly bandBrickCount: number;
  readonly surfaceGrowthSafety: number;
  readonly surfaceGrowthHeadroomBricks: number;
  readonly maximumResidentBricks: number;
}

export interface OctreeFluidFootprintBudget {
  readonly initialLiquidCells: number;
  readonly inflowLiquidCells: number;
  readonly maximumLiquidCells: number;
  /** Inclusive/exclusive finest-cell bounds of the authored t=0 liquid. */
  readonly minimumCell: readonly [number, number, number];
  readonly maximumCell: readonly [number, number, number];
}

/** Exact authored liquid-volume budget.  This is construction-time work only:
 * it deliberately shares the initial-condition predicates with the bootstrap
 * so capacity follows fluid rather than the surrounding air arena.  Inflow is
 * integrated over the authored scene duration and converted conservatively to
 * finest-cell volumes.  Runtime overflow remains a rejected GPU publication. */
export function planOctreeFluidFootprintBudget(
  scene: SceneDescription,
  dims: { nx: number; ny: number; nz: number },
): OctreeFluidFootprintBudget {
  const dimensions = [dims.nx, dims.ny, dims.nz] as const;
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Octree fluid-footprint dimensions must be positive integers");
  }
  const dam = sceneDamBreakBox(scene); let initialLiquidCells = 0;
  const minimum = [dims.nx, dims.ny, dims.nz] as [number, number, number];
  const maximum = [0, 0, 0] as [number, number, number];
  for (let z = 0; z < dims.nz; z += 1) for (let y = 0; y < dims.ny; y += 1) {
    for (let x = 0; x < dims.nx; x += 1) {
      const baseWet = scene.fluid.initialCondition === "dam-break"
        ? damBreakBoxContains(dam, (x + 0.5) / dims.nx, (y + 0.5) / dims.ny,
          (z + 0.5) / dims.nz)
        : (y + 0.5) / dims.ny <= scene.container.fillFraction;
      const wet = combineInitialBrickWet(scene,
        initialFluidBrickContainsCell(scene, x, y, z, dimensions), baseWet);
      if (!wet) continue;
      initialLiquidCells += 1;
      minimum[0] = Math.min(minimum[0], x); minimum[1] = Math.min(minimum[1], y);
      minimum[2] = Math.min(minimum[2], z);
      maximum[0] = Math.max(maximum[0], x + 1); maximum[1] = Math.max(maximum[1], y + 1);
      maximum[2] = Math.max(maximum[2], z + 1);
    }
  }
  if (initialLiquidCells === 0) { minimum.fill(0); maximum.fill(1); }
  const cellVolume = scene.container.width_m * scene.container.height_m
    * scene.container.depth_m / (dims.nx * dims.ny * dims.nz);
  const inflowVolume = scene.fluid.inflow
    ? integratedInflowVolume(scene.fluid.inflow, 0, Math.max(0, scene.duration_s)) : 0;
  const inflowLiquidCells = Math.ceil(Math.max(0, inflowVolume) / cellVolume);
  const maximumLiquidCells = Math.min(dims.nx * dims.ny * dims.nz,
    initialLiquidCells + inflowLiquidCells);
  return Object.freeze({ initialLiquidCells, inflowLiquidCells, maximumLiquidCells,
    minimumCell: Object.freeze(minimum), maximumCell: Object.freeze(maximum) });
}

/** Footprint-specialized version of the physical band plan.  Only the three
 * exposed faces of the authored liquid box are needed for a corner/tank seed;
 * the deformation safety and inflow surface term are explicit headroom, not a
 * numerical assumption. */
export function planFluidFootprintFineNarrowBandBrickCapacity(
  logicalBrickDimensions: readonly [number, number, number],
  footprintBrickDimensions: readonly [number, number, number],
  dilationBrickRings: number,
  inflowBrickBudget = 0,
  surfaceGrowthSafety = 1.5,
): GlobalFineNarrowBandCapacityPlan {
  const base = planGlobalFineNarrowBandBrickCapacity(logicalBrickDimensions,
    dilationBrickRings, surfaceGrowthSafety);
  if (footprintBrickDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
    || !Number.isSafeInteger(inflowBrickBudget) || inflowBrickBudget < 0) {
    throw new RangeError("Fluid-footprint fine-band dimensions are invalid");
  }
  const [x, y, z] = footprintBrickDimensions;
  const authoredArea = x * y + x * z + y * z;
  const inflowArea = inflowBrickBudget === 0 ? 0
    : Math.ceil(3 * inflowBrickBudget ** (2 / 3));
  const maximumInterfaceAreaBricks = Math.min(base.logicalBrickCount,
    authoredArea + inflowArea);
  const bandBrickCount = Math.min(base.logicalBrickCount,
    maximumInterfaceAreaBricks * base.bandLayers);
  const maximumResidentBricks = Math.min(base.logicalBrickCount,
    Math.ceil(bandBrickCount * surfaceGrowthSafety));
  return Object.freeze({ ...base, maximumInterfaceAreaBricks, bandBrickCount,
    surfaceGrowthHeadroomBricks: maximumResidentBricks - bandBrickCount,
    maximumResidentBricks });
}

/**
 * Exact publication floor for a rectangular authored liquid footprint.
 *
 * The rolling area-times-width estimate above is the right asymptotic reserve,
 * but it omits the edge and corner terms of the first Chebyshev dilation. That
 * omission is material for compact seeds: an 8-cubed drop with a six-brick
 * publication radius occupies a clipped box-shell volume, not six unrelated
 * face slabs. The GPU publisher discovers that volume exactly and rejects the
 * generation when the page pool was sized only from the slab estimate. Callers
 * include the extra transport membership ring when planning recurring updates.
 *
 * Bounds are half-open in logical fine-brick coordinates. The returned shell
 * is the largest translated expanded box that fits in the domain minus the
 * box interior farther than `dilationBrickRings` from every authored face.
 * Planning the initial clipped position is insufficient: a lid- or wall-flush
 * seed consumes the un-clipped envelope as soon as it separates.
 */
export function planFluidFootprintFineBandBrickFloor(
  logicalBrickDimensions: readonly [number, number, number],
  footprintMinimumBrick: readonly [number, number, number],
  footprintMaximumBrick: readonly [number, number, number],
  dilationBrickRings: number,
): number {
  if (logicalBrickDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
    || !Number.isSafeInteger(dilationBrickRings) || dilationBrickRings < 1
    || footprintMinimumBrick.some((value, axis) => !Number.isSafeInteger(value)
      || value < 0 || value >= footprintMaximumBrick[axis]!)
    || footprintMaximumBrick.some((value, axis) => !Number.isSafeInteger(value)
      || value > logicalBrickDimensions[axis]!)) {
    throw new RangeError("Fluid-footprint fine-band bounds are invalid");
  }
  const outerExtents = logicalBrickDimensions.map((dimension, axis) => Math.min(
    dimension,
    footprintMaximumBrick[axis]! - footprintMinimumBrick[axis]!
      + 2 * dilationBrickRings,
  ));
  const innerExtents = logicalBrickDimensions.map((_dimension, axis) => Math.max(0,
    footprintMaximumBrick[axis]! - footprintMinimumBrick[axis]!
      - 2 * dilationBrickRings));
  const volume = (extents: readonly number[]) => extents.reduce((product, value) => {
    const next = product * value;
    if (!Number.isSafeInteger(next)) {
      throw new RangeError("Fluid-footprint fine-band volume exceeds exact integer range");
    }
    return next;
  }, 1);
  return volume(outerExtents) - volume(innerExtents);
}

/**
 * Physical single-interface narrow-band capacity, in global fine bricks.
 *
 * This is deliberately an area-times-width plan. Increasing all logical
 * dimensions while holding the physical brick-band width fixed grows the
 * reserve quadratically rather than materializing the cubic fine lattice.
 * `surfaceGrowthSafety` is explicit deformation/topology headroom; fixed-size
 * physical pages themselves do not incur allocator fragmentation. The 1.5
 * default preserves the evolving fine SPGrid required by Aanjaneya et al.
 * 2017 Section 5 (`docs/papers/aanjaneya-2017-power-liquids.txt`); their
 * two-grid construction updates the fine surface grid every advection step
 * and does not assume that its interface remains planar.
 */
export function planGlobalFineNarrowBandBrickCapacity(
  brickDimensions: readonly [number, number, number],
  dilationBrickRings: number,
  surfaceGrowthSafety = 1.5,
): GlobalFineNarrowBandCapacityPlan {
  if (brickDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
    || !Number.isSafeInteger(dilationBrickRings) || dilationBrickRings < 1
    || !Number.isFinite(surfaceGrowthSafety) || surfaceGrowthSafety < 1) {
    throw new RangeError("Global fine narrow-band estimate inputs are invalid");
  }
  const [x, y, z] = brickDimensions;
  const logicalBrickCount = x * y * z;
  const maximumInterfaceAreaBricks = Math.max(x * y, x * z, y * z);
  const bandLayers = 2 * dilationBrickRings + 1;
  if (![logicalBrickCount, maximumInterfaceAreaBricks, bandLayers,
    maximumInterfaceAreaBricks * bandLayers].every(Number.isSafeInteger)) {
    throw new RangeError("Global fine narrow-band estimate exceeds exact integer range");
  }
  const bandBrickCount = Math.min(logicalBrickCount, maximumInterfaceAreaBricks * bandLayers);
  const plannedWithHeadroom = Math.ceil(bandBrickCount * surfaceGrowthSafety);
  const maximumResidentBricks = Math.min(logicalBrickCount, plannedWithHeadroom);
  return {
    logicalBrickCount, maximumInterfaceAreaBricks, bandLayers, bandBrickCount,
    surfaceGrowthSafety,
    surfaceGrowthHeadroomBricks: maximumResidentBricks - bandBrickCount,
    maximumResidentBricks,
  };
}

/** Scalar form of the physical narrow-band plan. */
export function estimateGlobalFineNarrowBandBrickCapacity(
  brickDimensions: readonly [number, number, number],
  dilationBrickRings: number,
  surfaceGrowthSafety = 1.5,
): number {
  return planGlobalFineNarrowBandBrickCapacity(
    brickDimensions, dilationBrickRings, surfaceGrowthSafety,
  ).maximumResidentBricks;
}

/** Exact named-resource sum used by production allocation telemetry. */
export function sumOctreePowerAllocationBreakdown(
  breakdown: Readonly<Record<string, number>>,
): number {
  let total = 0;
  for (const [name, bytes] of Object.entries(breakdown)) {
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new RangeError(`Octree power allocation ${name} must be non-negative safe bytes`);
    }
    total += bytes;
    if (!Number.isSafeInteger(total)) throw new RangeError("Octree power allocation total exceeds safe integer range");
  }
  return total;
}

/**
 * Deformation/topology headroom on the interface band, matching the physical
 * narrow-band plan above.  Fixed-size row records do not fragment, so this is
 * pure interface-growth reserve.
 */
export const OCTREE_PRESSURE_SURFACE_GROWTH_SAFETY = 1.25;
/** Total row headroom over the authored liquid + finite inflow volume.  A
 * rejected generation is the growth signal; no kernel silently truncates to
 * this budget. */
export const OCTREE_PRESSURE_FLUID_FOOTPRINT_HEADROOM = 2;

/**
 * A sloshing or collapsing liquid redistributes within at least this fraction
 * of the container, so the wettable envelope never shrinks below it however
 * shallow the authored fill is.
 */
export const OCTREE_PRESSURE_WETTABLE_FLOOR_FRACTION = 0.5;

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

type OctreeFrontierFailureRegion = (typeof OCTREE_FRONTIER_FAILURE_REGIONS)[number][0];

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
 * Capacity for the compact pressure publication.  The interface contribution
 * is an area-times-width band, the wall term reserves the closed-wall unit
 * strip that can carry rows, and the fully-coarse term covers the calm bulk.
 * Overflow is detected on-GPU and fail-closed; this is a capacity, not an
 * assumption used by the numerical kernels.
 */
export function planOctreePressureCapacity(
  dims: { nx: number; ny: number; nz: number },
  maximumLeafSize: number,
  interfaceBandCells: number,
  override?: number,
  closedTop = false,
  liquidFillFraction = 1,
  rowCapacityLimit = Number.MAX_SAFE_INTEGER,
  fluidFootprint?: OctreeFluidFootprintBudget,
): OctreePressureCapacityPlan {
  const count = dims.nx * dims.ny * dims.nz;
  const aligned = (value: number) => Math.ceil(value / 256) * 256;
  const bandLayers = Math.max(2, Math.ceil(interfaceBandCells) + 2);
  // One connected interface has the area of a single cross-section.  Summing
  // all three modelled a surface that is simultaneously maximal in every
  // orientation, which no single interface can be, and at the widened ocean it
  // reserved 384k rows against a 25.6k-cell free surface.  This is the same
  // area-times-width shape `planGlobalFineNarrowBandBrickCapacity` uses for the
  // physical band, including its explicit deformation headroom.
  const interfaceArea = Math.max(dims.nx * dims.ny, dims.nx * dims.nz, dims.ny * dims.nz);
  const surfaceRows = Math.ceil(interfaceArea * bandLayers * OCTREE_PRESSURE_SURFACE_GROWTH_SAFETY);
  const coarseRows = 8 * Math.ceil(count / Math.max(1, maximumLeafSize ** 3));
  // Power authority currently uses the generated interior catalog.  Reserve
  // the closed-wall unit-strip bound in addition to the moving interface
  // bound; overlap only makes this conservative.  This prevents the
  // correctness strip from silently converting into a row-arena rollback.
  //
  // `powerClosedWallStripIntersects` splits the whole strip to unit owners,
  // but splitting the tree is not publishing a row: only wet leaves reach the
  // row arena, and the strip above the free surface stays dry.  The widened
  // ocean measures 148,600 published rows against a 480,768-cell closed strip,
  // so reserving the strip over the full container height was reserving air.
  // The wettable envelope is the rest waterline plus the strip width and the
  // interface band, and never less than half the container, which covers the
  // authored collapse and seiche cases.  Beyond it the GPU fails closed on
  // arena overflow rather than corrupting the solve.
  const stripWidth = Math.max(OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS, Math.ceil(interfaceBandCells));
  const clampedFill = Math.max(0, Math.min(1, liquidFillFraction));
  const wettableCellsY = Math.min(dims.ny,
    Math.max(Math.ceil(dims.ny * OCTREE_PRESSURE_WETTABLE_FLOOR_FRACTION),
      Math.ceil(clampedFill * dims.ny) + stripWidth + bandLayers));
  const wallRows = wettableCellsY >= dims.ny
    ? planOctreePowerBoundaryStrip(dims, interfaceBandCells, closedTop).unitCellUpperBound
    : planOctreePowerBoundaryStrip({ nx: dims.nx, ny: wettableCellsY, nz: dims.nz },
      interfaceBandCells, false).unitCellUpperBound;
  const sceneShapedRequest = surfaceRows + wallRows + coarseRows;
  const footprintRequest = fluidFootprint === undefined ? sceneShapedRequest
    : Math.ceil(Math.max(1, fluidFootprint.maximumLiquidCells)
      * OCTREE_PRESSURE_FLUID_FOOTPRINT_HEADROOM);
  const requested = override === undefined
    ? Math.min(sceneShapedRequest, footprintRequest) : override;
  if (!Number.isSafeInteger(rowCapacityLimit) || rowCapacityLimit < 1) {
    throw new RangeError("Octree pressure row-capacity limit must be a positive safe integer");
  }
  const rowCapacity = Math.max(1,
    Math.min(count, rowCapacityLimit, aligned(Math.max(1, Math.floor(requested)))));
  return {
    rowCapacity,
    pressureBytes: rowCapacity * 2 * 4,
    headerBytes: rowCapacity * 48,
  };
}

export interface OctreeCompactionAllocationPlan {
  scanBlockCapacity: number;
  candidateBlockCapacity: number;
  scanAndTaskBytes: number;
  activeTileBytes: number;
  /** Stable tail offsets consumed by GPU-only downstream delta publishers. */
  changeStateBaseWords: number;
  tileChangeFlagsOffsetWords: number;
  tileRefinementSignaturesOffsetWords: number;
  tileFrontierSignaturesOffsetWords: number;
  tileSignatureChangedOffsetWords: number;
  tileFrontierChangeFlagsOffsetWords: number;
  frontierTopologyReuseWord: number;
  dirtyFailureOffsetWords: number;
  /** Plain-storage scratch for cooperative row classification and scans. */
  rowDeltaScratchBaseWords: number;
  rowDeltaScratchWords: number;
  allocatedBytes: number;
}

/**
 * Size the shared scan/task arena from the authorities that can actually
 * publish work. Compact pressure owns one scan record per frontier block;
 * active/retired topology tiles own the exact candidate scan records. The
 * resident tile list remains an independent lower bound because it is copied
 * into the same buffer before topology rebuilds.
 */
export function planOctreeCompactionAllocation(
  dims: { nx: number; ny: number; nz: number },
  pressureRowCapacity: number,
  activeTileWorklistBytes: number,
  activeTileCapacity: number,
  topologyTileSize: number,
): OctreeCompactionAllocationPlan {
  if (![dims.nx, dims.ny, dims.nz].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new Error("Octree compaction dimensions must be positive integers");
  }
  if (!Number.isSafeInteger(pressureRowCapacity) || pressureRowCapacity < 1) {
    throw new Error("Octree compaction pressure capacity must be a positive integer");
  }
  if (!Number.isSafeInteger(activeTileWorklistBytes) || activeTileWorklistBytes < 0
    || !Number.isSafeInteger(activeTileCapacity) || activeTileCapacity < 0
    || !Number.isSafeInteger(topologyTileSize) || topologyTileSize < 8 || topologyTileSize % 8 !== 0) {
    throw new Error("Octree compaction active-tile bounds must be non-negative integers");
  }
  const scanBlockCapacity = Math.ceil(pressureRowCapacity / 256);
  const candidateBlocksPerTile = (topologyTileSize / 8) ** 3;
  const candidateBlockCapacity = 2 * activeTileCapacity * candidateBlocksPerTile;
  const scanAndTaskBytes = 4 * (15 + 3 * scanBlockCapacity
    + 2 * candidateBlockCapacity);
  // The recurring state retains generation-stamped dirty membership, the
  // compact dirty list, and independent five-word structural-refinement and
  // wet-frontier signatures per logical topology tile. A one-word bitmask
  // lets the parallel comparison publish both decisions for the compacting
  // singletons without atomics.
  const tileSignatureWords = 5;
  const tileFrontierSignatureWords = 5;
  const tileSignatureChangedWords = 1;
  const tileFrontierChangeFlagWords = 1;
  const rigidSnapshotWords = 2 + 12 * 12;
  const dirtyAuthorityWords = 1;
  const dirtyFailureWords = 8;
  // The fourteen publication words are a last-good row-control snapshot plus
  // an independent exact-topology reuse bit.  The latter survives restoring
  // words 0..11, so downstream topology consumers can distinguish immutable
  // row reuse from a freshly emitted row set without a host readback.
  const activeTileBytes = 4 * ((2 + tileSignatureWords + tileFrontierSignatureWords
    + tileSignatureChangedWords + tileFrontierChangeFlagWords) * activeTileCapacity
    + dirtyAuthorityWords + rigidSnapshotWords + 14 + dirtyFailureWords) + 32;
  const rowDeltaBlockCount = Math.ceil(pressureRowCapacity / 256);
  // Two row-sized streams (flags and exclusive ranks), one block-total stream
  // plus its exact total, and two words per classification block.
  const rowDeltaScratchWords = 2 * pressureRowCapacity + 3 * rowDeltaBlockCount + 1;
  const allocatedBytes = Math.max(60, scanAndTaskBytes, activeTileWorklistBytes)
    + rowDeltaScratchWords * 4 + activeTileBytes;
  const changeStateWords = (2 + tileSignatureWords + tileFrontierSignatureWords
    + tileSignatureChangedWords + tileFrontierChangeFlagWords) * activeTileCapacity
    + dirtyAuthorityWords + rigidSnapshotWords + 14 + dirtyFailureWords;
  const changeStateBaseWords = allocatedBytes / 4 - 8 - changeStateWords;
  const tileChangeFlagsOffsetWords = changeStateBaseWords;
  const tileRefinementSignaturesOffsetWords = changeStateBaseWords + 2 * activeTileCapacity;
  const tileFrontierSignaturesOffsetWords =
    tileRefinementSignaturesOffsetWords + tileSignatureWords * activeTileCapacity;
  const tileSignatureChangedOffsetWords =
    tileFrontierSignaturesOffsetWords + tileFrontierSignatureWords * activeTileCapacity;
  const tileFrontierChangeFlagsOffsetWords =
    tileSignatureChangedOffsetWords + tileSignatureChangedWords * activeTileCapacity;
  const frontierTopologyReuseWord = changeStateBaseWords
    + (2 + tileSignatureWords + tileFrontierSignatureWords
      + tileSignatureChangedWords + tileFrontierChangeFlagWords) * activeTileCapacity
    + dirtyAuthorityWords + rigidSnapshotWords + 13;
  const dirtyFailureOffsetWords = frontierTopologyReuseWord + 1;
  const rowDeltaScratchBaseWords = changeStateBaseWords - rowDeltaScratchWords;
  return { scanBlockCapacity, candidateBlockCapacity, scanAndTaskBytes, activeTileBytes,
    changeStateBaseWords, tileChangeFlagsOffsetWords, tileRefinementSignaturesOffsetWords,
    tileFrontierSignaturesOffsetWords, tileSignatureChangedOffsetWords,
    tileFrontierChangeFlagsOffsetWords, frontierTopologyReuseWord, dirtyFailureOffsetWords,
    rowDeltaScratchBaseWords, rowDeltaScratchWords, allocatedBytes };
}

interface OctreeProjectionResources {
  rigidBodies: GPUBuffer;
  rigidExchange: GPUBuffer;
  terrain: GPUTexture;
}

function octreeLeafSize(value: number): 2 | 4 | 8 | 16 | 32 {
  const rounded = Math.max(2, Math.round(value));
  if (rounded >= 32) return 32;
  if (rounded >= 16) return 16;
  if (rounded >= 8) return 8;
  return rounded <= 2 ? 2 : 4;
}

/**
 * CPU mirror of the three grading predicates used by balanceTopologyAt and
 * balanceCoarseBlock. Neighbor entries may repeat because distinct face/edge
 * probes can resolve to the same leaf.
 */
export function octreeBalancePredicatesWouldSplit(
  anchorSize: number,
  neighborSizes: readonly number[],
): boolean {
  const ratioViolation = anchorSize <= 16
    && neighborSizes.some((neighborSize) => neighborSize > 2 * anchorSize);
  const mixedPaperRing = anchorSize >= 2 && anchorSize <= 16
    && neighborSizes.some((neighborSize) => neighborSize < anchorSize)
    && neighborSizes.some((neighborSize) => neighborSize > anchorSize);
  const faceNeighborTooFine = anchorSize > 2
    && neighborSizes.some((neighborSize) => neighborSize * 2 < anchorSize);
  return ratioViolation || mixedPaperRing || faceNeighborTooFine;
}

/**
 * The largest leaf the domain can actually hold.
 *
 * A leaf is dyadic and size-aligned, so `origin + size <= dims` must hold on
 * every axis; `resetTopologyAt` enforces exactly that by halving until it does.
 * A domain whose shortest axis is 16 therefore contains no size-32 leaf ANYWHERE
 * and can never grow one, because refinement only ever makes leaves finer.
 *
 * The authored maximum still sizes the plans, the tile lattice and the shader
 * params; this only bounds the refinement and grading LADDER, whose every rung
 * above this size is a dispatch that provably matches no leaf. On the
 * symmetric-expansion lane (32 x 16 x 32, authored leaf 32) that is one coarse
 * refinement dispatch plus one coarse balance dispatch in every one of the
 * balance rounds, and two whole rounds of the fixed-point budget -- fixed costs
 * that do not shrink with the domain and are invisible on a cubic lane.
 */
export function octreeEffectiveLeafSize(
  maximumLeafSize: 2 | 4 | 8 | 16 | 32,
  dims: { nx: number; ny: number; nz: number },
): 2 | 4 | 8 | 16 | 32 {
  const shortest = Math.min(dims.nx, dims.ny, dims.nz);
  let size: number = maximumLeafSize;
  while (size > 2 && size > shortest) size >>= 1;
  return size as 2 | 4 | 8 | 16 | 32;
}

/**
 * Losasso's geometric parent rows are complete dyadic cubes. Unlike the
 * boundary-clipped Power topology, every hierarchy span must therefore divide
 * all three finest-grid dimensions. Keep the authored maximum as a ceiling and
 * select the coarsest exact tiling; a unit-leaf result is the intentional
 * single-level hierarchy for domains with an odd axis.
 */
export function octreeLosassoTopologyLeafSize(
  maximumLeafSize: 2 | 4 | 8 | 16 | 32,
  dims: { nx: number; ny: number; nz: number },
): OctreeOwnerLeafSize {
  let size: number = maximumLeafSize;
  while (size > 1 && [dims.nx, dims.ny, dims.nz].some((value) => value % size !== 0)) {
    size >>= 1;
  }
  return size as OctreeOwnerLeafSize;
}

export function octreeBalanceRounds(maximumLeafSize: OctreeOwnerLeafSize): number {
  if (maximumLeafSize <= 2) return 0;
  // Ordinary 2:1 balance needs at most one propagation per tree level. The
  // paper's stronger mixed-ring rule can renew an ordinary imbalance, so
  // budget both halves of that fixed-point iteration.
  return 2 * Math.ceil(Math.log2(maximumLeafSize));
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

export const OCTREE_PROJECTION_CORE_BUFFER_LAYOUT = Object.freeze([
  { binding: 2, type: "storage" },
  { binding: 3, type: "storage" },
  { binding: 4, type: "storage" },
  { binding: 5, type: "storage" },
  { binding: 6, type: "uniform" },
  { binding: 8, type: "storage" },
  { binding: 10, type: "storage" },
  { binding: 11, type: "storage" },
  { binding: 13, type: "storage" },
  { binding: 15, type: "read-only-storage" },
] as const);

/** The core family's non-buffer bindings. Exported beside the buffer layout so
 * the reachability audit can check an entry point against the complete family
 * rather than against the buffers alone, which reads a legitimate texture as an
 * uncovered binding. */
export const OCTREE_PROJECTION_CORE_TEXTURE_LAYOUT = Object.freeze([
  { binding: 12, viewDimension: "2d" },
  { binding: 14, viewDimension: "3d" },
] as const);

export const OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT = Object.freeze([
  { binding: 2, type: "storage" },
  { binding: 3, type: "storage" },
  { binding: 6, type: "uniform" },
  { binding: 7, type: "uniform" },
  { binding: 9, type: "storage" },
  { binding: 13, type: "storage" },
] as const);

function projectionBufferLayoutEntries(
  entries: readonly { readonly binding: number;
    readonly type: "uniform" | "storage" | "read-only-storage" }[],
): GPUBindGroupLayoutEntry[] {
  return entries.map(({ binding, type }) => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type },
  }));
}

/**
 * A GPU-resident, pressure-only octree projection.
 *
 * Ownership is paged and pressure exists only at live leaf origins, resolved
 * through the compact frontier hash.
 */
export class WebGPUOctreeProjection {
  readonly preconditioner = "section43-hybrid" as const;
  /** Observational only; no simulation branch consumes these counters. */
  readonly workAccounting = new OctreeWorkAccounting();
  readonly info: {
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
  };
  levelSetMismatchFraction = 0;
  relativeResidual?: number;
  residualRms?: number;
  initialResidualRms?: number;

  private readonly topology: GPUBuffer;
  private readonly ownerPages: WebGPUOctreeSimulationOwnerPages;
  private readonly pressureA: GPUBuffer;
  private readonly pressureB: GPUBuffer;
  private readonly compaction: GPUBuffer;
  private readonly leafHeaders: GPUBuffer;
  private readonly candidateLeafHeaders: GPUBuffer;
  private readonly candidatePressure: GPUBuffer;
  /** Plain-u32 ping/pong scratch used only by the cold large-frontier merge sort. */
  private readonly frontierSortScratch: GPUBuffer;
  private readonly leafFrontier: GPUBuffer;
  private readonly fineSeedAdapter?: WebGPUOctreeFineSeedAdapter;
  private readonly globalFineLevelSet?: WebGPUFineLevelSetBricks;
  private readonly globalFineSeeds?: WebGPUFineLevelSetLeafSeeds;
  private globalFineTopologyAB?: WebGPUFineLevelSetTopology;
  private globalFineTopologyBA?: WebGPUFineLevelSetTopology;
  private globalFineRedistanceA?: WebGPUFineLevelSetRedistance;
  private globalFineRedistanceB?: WebGPUFineLevelSetRedistance;
  private globalFineVolumeA?: WebGPUFineLevelSetVolumeCorrection;
  private globalFineVolumeB?: WebGPUFineLevelSetVolumeCorrection;
  private globalFineTransportA?: WebGPUFineLevelSetTransport;
  private globalFineTransportB?: WebGPUFineLevelSetTransport;
  private losassoFineTransportA?: WebGPUOctreeLosassoFineTransport;
  private losassoFineTransportB?: WebGPUOctreeLosassoFineTransport;
  private lastGlobalFineTransport?: WebGPUFineLevelSetTransport | WebGPUOctreeLosassoFineTransport;
  private readonly globalFineSummaries?: WebGPUFineLevelSetSummaries;
  private coarseOnlySummary?: WebGPUOctreeCoarseSummary;
  private readonly unpublishedFineSummaryDirectory: GPUBuffer;
  private surfaceStateAccountingBytes = 0;
  private powerCoarseLevelSet?: WebGPUOctreeCoarseLevelSet;
  private powerCoarseLevelSetSchedule?: WebGPUOctreePowerCoarseLevelSet;
  private fineToPowerCoarseLevelSet?: WebGPUFineToCoarseLevelSet;
  private powerCoarseLevelSetBootstrapped = false;
  /** Generation of the currently scheduled coarse-octree phi publication.
   * It advances independently when the optional global fine band is off. */
  private powerCoarseLevelSetGeneration = 2;
  private globalFineSourceA?: WebGPUFineLevelSetBrickSource;
  private globalFineSourceB?: WebGPUFineLevelSetBrickSource;
  /** Slot consumed by the command stream currently being encoded. */
  private globalFineCurrentIsA = true;
  /** Last slot whose producing encoder has actually been queue-submitted. */
  private globalFinePublishedIsA = true;
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
  private globalFineBootstrapped = false;
  /** Monotone host mirror of the topology shader's `currentFinePopulated()`
   * (POWER_LIQUIDS_ULTIMATE_M1MAX.md B1 / P1.1). Set once a delta publication
   * has been encoded, i.e. after the cold bootstrap and its retry step. Only
   * the seed-chain encode consults it; a stale `false` merely re-encodes the
   * chain harmlessly, and it is never cleared short of teardown. */
  private finePopulated = false;
  private globalFineGeneration = 2;
  private lastPowerBoundaryFineSource?: { generation: number; generationSlot: 0 | 1 };
  private powerTimestep_s = 0;
  private surfaceInflow?: SurfaceInflowState;
  private pendingSurfaceReferenceVolume_m3 = 0;
  /** Bodies that integrate this step; zero keeps the adjoint off the graph. */
  private dynamicCouplingBodyCount = 0;
  private powerAdvancingPressureSteps = 0;
  private readonly solveDispatch: GPUBuffer;
  private readonly topologyCandidateDispatch: GPUBuffer;
  private readonly topologyTileChangeFlagsOffsetBytes: number;
  private readonly topologyTileChangeFlagsByteLength: number;
  private readonly compactionAllocationRowDeltaScratchOffsetBytes: number;
  private readonly dirtyFailureOffsetBytes: number;
  private readonly frontierPublicationOffsetBytes: number;
  private readonly dirtyAuthorityStateOffsetBytes: number;
  private readonly solidCells: GPUBuffer;
  private readonly hasDenseSolidCells: boolean;
  private readonly params: GPUBuffer;
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
  private readonly surfaceState: WebGPUQuadtreeSurfaceState;
  private readonly sparseBrickWorld?: OctreeSparseBrickWorld;
  private readonly topologyResidency: GPUFluidBrickResidency;
  private readonly analyticBootstrapWorklist?: WebGPUOctreeAnalyticBootstrapWorklist;
  private sparseBrickWorldAccountedBytes = 0;
  private groups: { ab: GPUBindGroup; ba: GPUBindGroup };
  private candidateRowGroups: { fromA: GPUBindGroup; fromB: GPUBindGroup };
  private fineSummarySizingGroup: GPUBindGroup;
  private readonly frontierSortGroups: readonly GPUBindGroup[];
  /** Current fine/coarse classification plus persistent topology-tile membership. */
  private topologyDecisionGroup?: GPUBindGroup;
  private denseBootstrapPhiReleased = false;
  private topologyDiagnosticTexture?: GPUTexture;
  private pressureSamplesDiagnosticTexture?: GPUTexture;
  private pressureDiagnosticTexture?: GPUTexture;
  private diagnosticGroups?: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
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
  private advanceGradingRoundPipeline!: GPUComputePipeline;
  private materializePipeline?: GPUComputePipeline;
  private readonly maxLeafSize: 2 | 4 | 8 | 16 | 32;
  /** Backend-normalized maximum consumed by the structural topology. */
  private readonly topologyMaximumLeafSize: OctreeOwnerLeafSize;
  private readonly coarseDynamics: OctreeCoarseDynamicsConfiguration;
  private topologyCadenceCursor = 0;
  private readonly fluidGatedBoundaryRefinement: boolean;
  private readonly topologyTileSize: number;
  private readonly adaptivity: number;
  private readonly interfaceRefinementBandCells: number;
  private readonly surfaceRefinementGradingLayers: number;
  private readonly fineLevelSetBandCells: number;
  /** Factor-one uses the compact octree phi as the sole moving surface;
   * factors four/eight allocate the separate sparse fine band. */
  private readonly coarseOnlySurfaceTracking: boolean;
  private pressureSolverControl!: GPUBuffer;
  /** Row-parallel production pressure executor with exact integer reductions. */
  private pipelinedMGPCG?: WebGPUOctreePipelinedMGPCG;
  private section43HybridPreconditioner?: WebGPUOctreeSection43HybridPreconditioner;
  private pipelinedMGPCGVectors?: OctreePipelinedMGPCGVectors;
  private firstOrderVCycle!: OctreeFirstOrderVCycleImplementation;
  private losassoBackend?: WebGPUOctreeLosassoCoarseBackend;
  private losassoReadyCommit?: WebGPUOctreeLosassoReadyCommit;
  private losassoCoarsePhi?: WebGPUOctreeLosassoCoarsePhiExchange;
  private losassoRowMotion?: WebGPUOctreeLosassoRowMotion;
  private losassoConditionedOperator?: WebGPUOctreeLosassoConditionedOperator;
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
  private readonly pressureCapacity: OctreePressureCapacityPlan;
  private readonly frontierAllocation: OctreeLeafFrontierAllocationPlan;
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
  private compactionByteLength = 0;
  private solveStats!: GPUBuffer;
  private topologyWorklistReady = false;
  private latestPressureInA = true;
  /** No dense phi exists; non-page topology groups must retain analytic sign until coarse correction publishes. */
  private readonly analyticSparseBootstrap: boolean;
  private powerDescriptor?: WebGPUOctreePowerDescriptor;
  private powerTopology?: WebGPUOctreePowerTopology;
  private powerSolidVertices?: WebGPUOctreeSolidVertexSdf;
  private powerVolumes?: GPUBuffer;
  private powerVolumeParams?: GPUBuffer;
  private powerVolumePipeline?: GPUComputePipeline;
  private powerVolumeGroup?: GPUBindGroup;
  private initializePowerVolumePipeline?: () => Promise<void>;
  /** Host-side encode serial used only for API validation/diagnostics. The
   * physics generation is stamped in command-buffer order by the GPU. */
  private powerAttemptGeneration = 0;
  private candidatePowerGeneration = 0;
  private activePowerGeneration = 0;
  private readonly solveTailPolicy: OctreeSolveTailPolicy;
  private powerLifecycleDisposed = false;

  constructor(
    private readonly device: GPUDevice,
    // Not readonly: `applySceneUniforms` swaps in scalar-only scene revisions
    // so a density or gravity edit is a uniform write, not a rebuild.
    private scene: SceneDescription,
    private readonly dims: { nx: number; ny: number; nz: number },
    private readonly resources: OctreeProjectionResources,
    options: OctreeProjectionOptions,
    private readonly deferPipelineCompilation = false,
    allocationProgress?: OctreeAllocationProgress,
  ) {
    this.deferPipelineCompilation = true;
    const reportAllocation = (stage: number) => allocationProgress?.(
      OCTREE_ALLOCATION_STAGES[stage]!, stage, OCTREE_ALLOCATION_STAGES.length,
    );
    reportAllocation(0);
    const count = dims.nx * dims.ny * dims.nz;
    this.coarseDynamics = options.coarseDynamics ?? resolveOctreeCoarseDynamics({
      // Direct construction follows the product default. Frozen Power
      // reference lanes must opt in explicitly at their call site.
      backend: DEFAULT_OCTREE_COARSE_BACKEND,
      globalFineLevelSetFactor: options.globalFineLevelSetFactor ?? 4,
    });
    this.maxLeafSize = octreeLeafSize(options.maximumLeafSize ?? 16);
    this.topologyMaximumLeafSize = this.coarseDynamics.backend === "losasso"
      ? octreeLosassoTopologyLeafSize(this.maxLeafSize, dims)
      : this.maxLeafSize;
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
    this.effectiveLeafSize = this.coarseDynamics.backend === "losasso"
      ? this.topologyMaximumLeafSize
      : octreeEffectiveLeafSize(this.maxLeafSize, dims);
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
    this.balanceRounds = octreeBalanceRounds(this.effectiveLeafSize);
    this.adaptivity = Math.max(0, Math.min(1, options.adaptivity ?? 1));
    this.interfaceRefinementBandCells = Math.max(0, Math.min(32, Math.round(options.interfaceRefinementBandCells ?? 4)));
    this.surfaceRefinementGradingLayers = Math.max(1, Math.min(4,
      Math.round(options.surfaceRefinementGradingLayers ?? 1)));
    // Product configurations couple Section 5 surface reach to pressure reach.
    // A distinct value is admitted only for diagnostic fault injection; unset
    // follows the master band exactly.
    this.fineLevelSetBandCells = Math.max(0, Math.min(32,
      Math.round(options.fineLevelSetBandCells ?? this.interfaceRefinementBandCells)));
    // Factor one is represented only by compact octree rows. Enforce this at
    // the allocation boundary so no caller or authored scene can accidentally
    // pay for a redundant same-resolution fine grid.
    this.coarseOnlySurfaceTracking = options.globalFineLevelSetFactor === 1;
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
    const analyticSparseBootstrap = (scene.fluid.initialBrickSeeds_m?.length ?? 0) === 0
      && scene.rigidBodies.length === 0 && !sceneHasTerrain(scene)
      && !sceneDamBreakIsOffsetFromCorner(scene);
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
    const structuredVelocityRowLimit = structuredVelocityRowCapacityForBindingLimit(maximumStorageBinding);
    const spgridRowLimit = spgridRowCapacityForBindingLimit(
      [dims.nx, dims.ny, dims.nz],
      maximumStorageBinding,
      plannedPressureCapacity.rowCapacity,
    );
    const deviceRowLimit = Math.min(structuredVelocityRowLimit, spgridRowLimit);
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
    // The Losasso hierarchy requires an exact domain tiling, so its normalized
    // topology maximum is also the tile ABI consumed by the host residency
    // plan and `topologyTileSize()` in WGSL. Power retains its frozen authored-
    // leaf default and the existing diagnostic clamp experiment.
    this.topologyTileSize = Math.max(8, this.coarseDynamics.backend === "losasso"
      ? this.topologyMaximumLeafSize
      : octreeTopologyTileClampEnabled() ? this.effectiveLeafSize : this.maxLeafSize);
    const allocateSparseWorld = octreeSparseWorldRequired(sceneHasTerrain(scene), scene.rigidBodies.length);
    const sparseWorldBrickSize = scene.voxelDomain.brickSize_cells;
    if (allocateSparseWorld) this.sparseBrickWorld = new OctreeSparseBrickWorld(device, scene, [dims.nx, dims.ny, dims.nz], {
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
    this.params = device.createBuffer({ label: "Octree projection parameters", size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
    const backendProjectionShader = this.coarseDynamics.backend === "losasso"
      ? octreeLosassoSurfaceGradingShader(octreeProjectionShader)
      : octreeProjectionShader;
    const projectionShaderVariant = this.projectionActivity.module(
      octreeProjectionActivityShader(this.projectionActivity, backendProjectionShader),
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
      pressureIterationHardBudget: this.solveTailPolicy.hardOuterIterationCeiling,
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
        const globalFineFactor = options.globalFineLevelSetFactor ?? 4;
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
          // the frozen Power reference's historical 1.5x reserve unchanged.
          const surfaceGrowthSafety = this.coarseDynamics.backend === "losasso" ? 2 : 1.5;
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
          const bandFloor = this.coarseDynamics.backend === "losasso"
            ? planFluidFootprintFineBandBrickFloor(
              brickDimensions,
              [0, 1, 2].map((axis) => Math.floor(
                fluidFootprint.minimumCell[axis]! * globalFineFactor / brickResolution,
              )) as [number, number, number],
              [0, 1, 2].map((axis) => Math.ceil(
                fluidFootprint.maximumCell[axis]! * globalFineFactor / brickResolution,
              )) as [number, number, number],
              capacityDilationBrickRings + 2,
            ) : 0;
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
            : this.analyticSparseBootstrap
            && (this.scene.fluid.initialBrickSeeds_m?.length ?? 0) === 0
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
    this.sparseBrickWorldAccountedBytes = this.sparseBrickWorld?.allocatedBytes ?? 0;
    if (this.coarseDynamics.backend === "losasso") this.initializeLosassoAuthority();
    reportAllocation(8);
    this.writeParams();
  }

  /** Construct the reduced Losasso graph synchronously so initialization can
   * enumerate only its own shader tasks. No Power catalogue or structured
   * authority is reachable from this construction branch. */
  private initializeLosassoAuthority(): void {
    if (this.losassoBackend || this.powerLifecycleDisposed) return;
    const rowCapacity = this.pressureCapacity.rowCapacity;
    const extensionBandBrickCapacity = this.globalFineSourceA?.plan.maximumResidentBricks
      ?? (this.coarseOnlySurfaceTracking ? 1 : undefined);
    if (extensionBandBrickCapacity === undefined) {
      throw new Error("Losasso factor-4/8 construction requires the sparse fine-phi page plan");
    }
    // A 2:1 row can expose four subfaces on each positive axis plus the
    // corresponding negative/free-surface patches. Twenty-four per row is a
    // conservative pre-deduplication bound; using the ordinary six-face count
    // would fail exactly at graded T-junctions.
    const faceCapacity = 24 * rowCapacity;
    const largestFaceArenaBytes = 32 * faceCapacity;
    const maximumStorageBytes = Math.min(this.device.limits.maxStorageBufferBindingSize,
      this.device.limits.maxBufferSize);
    if (!Number.isSafeInteger(faceCapacity) || largestFaceArenaBytes > maximumStorageBytes) {
      throw new RangeError("Losasso 2:1 axis-face capacity exceeds this device's storage binding limit");
    }
    const cellSize = this.scene.container.width_m / this.dims.nx;
    const closedTop = this.scene.container.top === "closed";
    this.losassoCoarsePhi = new WebGPUOctreeLosassoCoarsePhiExchange(
      this.device, rowCapacity, faceCapacity,
    );
    this.losassoBackend = new WebGPUOctreeLosassoCoarseBackend({
      device: this.device,
      capacities: { rows: rowCapacity, faces: faceCapacity, incidences: 2 * faceCapacity },
      topology: {
        dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
        maximumLeafSize: this.topologyMaximumLeafSize,
        physicalCellSize: [cellSize, cellSize, cellSize],
        domainOrigin: [0, 0, 0],
        ownerPages: this.ownerPages.plan,
      },
      density: this.scene.fluid.density_kg_m3,
      extensionBandBrickCapacity,
      closedBoundaries: [true, true, true, closedTop, true, true],
      solver: {
        relativeTolerance: this.solveTailPolicy.relativeTolerance,
        maximumIterations: this.solveTailPolicy.hardOuterIterationCeiling,
        hardIterationCeiling: this.solveTailPolicy.hardOuterIterationCeiling,
        // The cooperative drain is bounded by the compact coarse row arena,
        // not by the independent fine level-set factor. Keep larger coarse
        // problems on the ordinary partial-plus-finish schedule.
        factorOneCombinedReductionDrains:
          this.coarseOnlySurfaceTracking && rowCapacity <= 4_096,
      },
      rigidPressureReaction: {
        solidCells: this.solidCells,
        rigidBodies: this.resources.rigidBodies,
        rigidExchange: this.resources.rigidExchange,
        rigidWorldOrigin: [
          -0.5 * this.scene.container.width_m,
          0,
          -0.5 * this.scene.container.depth_m,
        ],
      },
    });
    this.pressureSolverControl = this.losassoBackend.solverControl
      ?? this.losassoBackend.sources.rowCount;
    this.losassoReadyCommit = new WebGPUOctreeLosassoReadyCommit(this.device, {
      candidateAuthority: this.losassoBackend.candidateAuthorityControl,
      ownerCandidateTransaction: this.ownerPages.candidateTransaction,
      frontier: this.leafFrontier,
      candidateLeafHeaders: this.candidateLeafHeaders,
      acceptedLeafHeaders: this.leafHeaders,
      candidatePressure: this.candidatePressure,
      pressureA: this.pressureA,
      pressureB: this.pressureB,
      acceptedRowCount: this.compaction,
    }, rowCapacity);
    const finest = this.losassoBackend.sources.operator;
    this.losassoRowMotion = new WebGPUOctreeLosassoRowMotion(this.device, {
      authority: finest.control,
      rowFaceOffsets: finest.rowFaceOffsets,
      rowFaces: finest.rowFaces,
      faces: finest.faces,
      extendedVelocity: this.losassoBackend.sources.extension.extendedVelocity,
    }, rowCapacity);
    this.fineSeedAdapter?.setRowMotionSource(this.losassoRowMotion.source);
    const wide = this.losassoBackend.sources.wideSolver;
    if (!wide) throw new Error("Losasso wide solver authority was not published");
    this.losassoConditionedOperator = new WebGPUOctreeLosassoConditionedOperator(this.device, {
      authority: finest.control,
      rowFaceOffsets: finest.rowFaceOffsets,
      rowFaces: finest.rowFaces,
      faces: finest.faces,
      diagonal: wide.diagonal,
      solverAuthority: wide.acceptedAuthority,
    }, rowCapacity);
    this.fineSeedAdapter?.setCoarsePhiSource(
      this.losassoCoarsePhi.fineSeedCoarsePhiSource());
    this.refreshLosassoProjectionGroups();

    const fineA = this.globalFineSourceA, fineB = this.globalFineSourceB;
    const sampler = this.losassoBackend.sources.velocitySampler;
    if (fineA && fineB) {
      if (!sampler) throw new Error("Losasso factor-4 transport requires its reduced velocity sampler");
      const coarseWGSL = makeOctreeLosassoCoarsePhiSampleWGSL(9);
      this.globalFineTopologyAB = new WebGPUFineLevelSetTopology(
        this.device, fineA, fineB, coarseWGSL, this.deferPipelineCompilation,
      );
      this.globalFineTopologyBA = new WebGPUFineLevelSetTopology(
        this.device, fineB, fineA, coarseWGSL, this.deferPipelineCompilation,
      );
      const changedKeysOffsetWords = this.globalFineTopologyAB.pageDeltaLayout.changedKeysOffsetWords;
      if (changedKeysOffsetWords !== this.globalFineTopologyBA.pageDeltaLayout.changedKeysOffsetWords) {
        throw new Error("Losasso fine topology A/B page-delta layouts disagree");
      }
      this.device.queue.writeBuffer(this.params, 36, new Uint32Array([changedKeysOffsetWords]));
      const redistanceOptions = (source: WebGPUFineLevelSetBrickSource) => ({
        deferPipelineCompilation: this.deferPipelineCompilation,
        axisPermutationInvariantSeeds: true,
        maximumRequiredJfaStride: maximumFineLevelSetJFAStride(
          planFineLevelSetBandFineCells(this.fineLevelSetBandCells,
            source.plan.fineFactor).redistanceBandFineCells),
      });
      this.globalFineRedistanceA = new WebGPUFineLevelSetRedistance(
        this.device, fineA, this.globalFineTopologyBA, redistanceOptions(fineA),
      );
      this.globalFineRedistanceB = new WebGPUFineLevelSetRedistance(
        this.device, fineB, this.globalFineTopologyAB, redistanceOptions(fineB),
      );
      this.losassoFineTransportA = new WebGPUOctreeLosassoFineTransport(
        this.device, fineA, sampler);
      this.losassoFineTransportB = new WebGPUOctreeLosassoFineTransport(
        this.device, fineB, sampler);
      const coarseInput = this.losassoCoarsePhiInput();
      const coarseVolume = this.losassoCoarsePhi.volumeCoarseSource(coarseInput);
      this.globalFineVolumeA = new WebGPUFineLevelSetVolumeCorrection(
        this.device, fineA, coarseVolume, undefined, this.deferPipelineCompilation,
      );
      this.globalFineVolumeB = new WebGPUFineLevelSetVolumeCorrection(
        this.device, fineB, coarseVolume, this.globalFineVolumeA.control,
        this.deferPipelineCompilation,
      );
    }
    const coarseAllocated = this.losassoBackend.allocatedBytes
      + this.losassoReadyCommit.allocatedBytes + this.losassoCoarsePhi.plan.allocatedBytes
      + this.losassoConditionedOperator.allocatedBytes + this.losassoRowMotion.plan.allocatedBytes;
    const allocated = coarseAllocated
      + (this.globalFineTopologyAB?.allocatedBytes ?? 0)
      + (this.globalFineTopologyBA?.allocatedBytes ?? 0)
      + (this.globalFineRedistanceA?.allocatedBytes ?? 0)
      + (this.globalFineRedistanceB?.allocatedBytes ?? 0)
      + (this.losassoFineTransportA?.plan.allocatedBytes ?? 0)
      + (this.losassoFineTransportB?.plan.allocatedBytes ?? 0)
      + (this.globalFineVolumeA?.allocatedBytes ?? 0)
      + (this.globalFineVolumeB?.allocatedBytes ?? 0);
    this.info.allocatedBytes += allocated;
    this.info.globalFineLevelSetAllocatedBytes += allocated - coarseAllocated;
    this.info.powerDiagramReady = false;
    this.info.powerDiagramAuthoritative = false;
    this.workAccounting.setAuthorityBytes("losasso", coarseAllocated);
    this.workAccounting.setAuthorityBytes("fine-level-set", Math.max(0,
      allocated - coarseAllocated));
    this.workAccounting.sealAllocationInventory();
  }

  private losassoCoarsePhiInput(): WebGPUOctreeLosassoCoarsePhiInput {
    const backend = this.losassoBackend;
    if (!backend) throw new Error("Losasso coarse authority was not constructed");
    return {
      leafHeaders: this.leafHeaders,
      coarseControl: backend.sources.operator.control,
      faces: backend.sources.projection.faces,
      faceGeometry: backend.sources.dynamics.faceGeometry,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      maximumLeafSize: this.topologyMaximumLeafSize,
      cellSize: this.scene.container.width_m / this.dims.nx,
    };
  }

  private refreshLosassoProjectionGroups(): void {
    const directory = this.losassoCoarsePhi?.source.arena;
    if (!directory) return;
    this.groups = {
      ab: this.createProjectionGroup(this.pressureA, this.pressureB, directory),
      ba: this.createProjectionGroup(this.pressureB, this.pressureA, directory),
    };
    this.candidateRowGroups = {
      fromA: this.createProjectionGroup(this.pressureA, this.candidatePressure, directory,
        this.candidateLeafHeaders),
      fromB: this.createProjectionGroup(this.pressureB, this.candidatePressure, directory,
        this.candidateLeafHeaders),
    };
    const summary = this.globalFineSummaries?.directory
      ?? this.unpublishedFineSummaryDirectory;
    this.fineSummarySizingGroup = this.createProjectionGroup(summary, this.pressureB, directory);
    this.topologyDecisionGroup = this.createProjectionGroup(
      summary, this.topologyResidency.topologyTileStateBuffer, directory,
    );
  }

  private createProjectionGroup(
    pressureIn: GPUBuffer,
    pressureOut: GPUBuffer,
    binding15Override?: GPUBuffer,
    leafHeadersOverride: GPUBuffer = this.leafHeaders,
  ): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 2, resource: { buffer: this.compaction } },
      { binding: 3, resource: { buffer: this.topology } },
      { binding: 4, resource: { buffer: pressureIn } },
      { binding: 5, resource: { buffer: pressureOut } },
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
    return /^(?:rasterizeSolids|resetTopology|refineTopology|balanceTopology|advanceGradingRound|stampFrontier|beginFrontier|classifyFrontier|scanFrontier|prefixFrontier|emitFrontier|prepareFrontier|sortFrontier|mergeFrontier|finalizeFrontier|planLeaves|emitLeaves|markRowDeltaRing)/.test(entryPoint);
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
      this.buildDirtyFrontierDeltaPipeline, this.advanceGradingRoundPipeline
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
        this.buildDirtyFrontierDeltaPipeline, this.advanceGradingRoundPipeline,
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
    const diagnosticGroup = (pressure: GPUBuffer) => this.device.createBindGroup({ layout: this.diagnosticLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } },
      { binding: 1, resource: { buffer: pressure } },
      { binding: 4, resource: this.topologyDiagnosticTexture!.createView() },
      { binding: 5, resource: this.pressureSamplesDiagnosticTexture!.createView() },
      { binding: 6, resource: this.pressureDiagnosticTexture!.createView() },
      { binding: 8, resource: { buffer: this.params } },
      { binding: 11, resource: { buffer: this.leafFrontier } }
    ] });
    this.diagnosticGroups = {
      pressureA: diagnosticGroup(this.pressureA),
      pressureB: diagnosticGroup(this.pressureB)
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
    if (this.coarseDynamics.backend === "losasso") {
      const reducedTasks = [
        ...(this.losassoBackend ? [{ label: "Compile complete Losasso coarse backend",
          run: () => this.losassoBackend!.initialize() }] : []),
        ...(this.losassoReadyCommit?.initializationTasks ?? []),
        ...(this.losassoCoarsePhi?.initializationTasks ?? []),
        ...(this.losassoRowMotion?.initializationTasks ?? []),
        ...(this.losassoConditionedOperator?.initializationTasks ?? []),
      ];
      reducedTasks.forEach((task, index) => tasks.push({
        id: `octree.losasso.pipeline.${index}`,
        phase: "solver-pipelines",
        label: task.label,
        run: () => task.run(),
      }));
      if (this.globalFineSourceA && this.globalFineSourceB) {
        tasks.push({
          id: "octree.losasso.fine-topology",
          phase: "solver-pipelines",
          label: "Compile Losasso factor-4 fine topology",
          run: async () => {
            await this.globalFineTopologyAB!.initializePipelines();
            await this.globalFineTopologyBA!.initializePipelines();
          },
        }, {
          id: "octree.losasso.fine-redistance",
          phase: "solver-pipelines",
          label: "Compile Losasso factor-4 redistance",
          run: async () => {
            await this.globalFineRedistanceA!.initializePipelines();
            await this.globalFineRedistanceB!.initializePipelines();
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
          id: "octree.losasso.fine-volume",
          phase: "solver-pipelines",
          label: "Compile Losasso factor-4 volume bridge",
          run: async () => {
            await this.globalFineVolumeA!.initializePipelines();
            await this.globalFineVolumeB!.initializePipelines();
          },
        });
      }
    } else if (!this.powerDescriptor) {
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
          this.info.powerDiagramReady = false;
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
          catch (error) { this.info.powerDiagramReady = false; throw error; }
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
          const coarseSummaryTasks = this.coarseOnlySummary?.initializationTasks() ?? [];
          for (let index = 0; index < coarseSummaryTasks.length; index += 1) {
            if (signal.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
            const task = coarseSummaryTasks[index]!;
            report?.(`Compile coarse summary: ${task.label} (${index}/${coarseSummaryTasks.length})`);
            await task.run(signal);
          }
        },
      });
      if (this.deferPipelineCompilation) {
        tasks.push({
          id: "octree.power-pipelines.spgrid",
          phase: "solver-pipelines",
          label: "Compile persistent SPGrid topology programs",
          dependencies: ["octree.power-authority.allocate"],
          run: () => this.firstOrderVCycle!.initializePipelines(),
        });
        if (this.globalFineSourceA && this.globalFineSourceB) {
          tasks.push({
            id: "octree.power-pipelines.fine-topology",
            phase: "solver-pipelines",
            label: "Compile fine topology programs",
            dependencies: ["octree.power-authority.allocate"],
            run: async () => {
              await this.globalFineTopologyAB!.initializePipelines();
              await this.globalFineTopologyBA!.initializePipelines();
            },
          }, {
            id: "octree.power-pipelines.fine-redistance",
            phase: "solver-pipelines",
            label: "Compile fine redistance programs",
            dependencies: ["octree.power-authority.allocate"],
            run: async () => {
              await this.globalFineRedistanceA!.initializePipelines();
              await this.globalFineRedistanceB!.initializePipelines();
            },
          }, {
            id: "octree.power-pipelines.fine-transport",
            phase: "solver-pipelines",
            label: "Compile fine transport programs",
            dependencies: ["octree.power-authority.allocate"],
            run: async () => {
              await this.globalFineTransportA!.initializePipelines();
              await this.globalFineTransportB!.initializePipelines();
            },
          }, {
            id: "octree.power-pipelines.fine-volume",
            phase: "solver-pipelines",
            label: "Compile fine volume-correction programs",
            dependencies: ["octree.power-authority.allocate"],
            run: async () => {
              await this.globalFineVolumeA!.initializePipelines();
              await this.globalFineVolumeB!.initializePipelines();
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
    }
    return tasks;
  }

  private initializeNativePowerAuthority(catalog: GeneratedOctreePowerCatalogViews): void {
    if (this.powerDescriptor || this.powerLifecycleDisposed) return;
    const rowCapacity = this.pressureCapacity.rowCapacity;
    this.powerDescriptor = new WebGPUOctreePowerDescriptor(this.device, rowCapacity);
    this.powerTopology = new WebGPUOctreePowerTopology(this.device, rowCapacity, catalog);
    const structured = new WebGPUDirectStructuredVelocityAuthority(this.device, {
      leafHeaders: this.leafHeaders,
      topology: this.powerTopology.source, rowDelta: this.powerRowDelta,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      physicalCellSize: this.scene.container.width_m / this.dims.nx,
      closedBoundaryMask: structuredClosedBoundaryMask(this.scene.container.top === "closed"),
    });
    this.structuredVelocity = structured;
    const structuredSource = structured.source;
    const section63Source = structuredSource.section63;
    this.structuredDivergenceRhs = this.device.createBuffer({
      label: "Structured divergence RHS SoA",
      size: rowCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const rowDelta = { rows: this.leafFrontier,
      rowCapacity: this.frontierAllocation.listCapacity,
      controlOffsetWords: this.frontierAllocation.rowDeltaControlOffsetWords,
      newToOldOffsetWords: this.frontierAllocation.rowDeltaNewToOldOffsetWords,
      oldToNewOffsetWords: this.frontierAllocation.rowDeltaOldToNewOffsetWords,
      // SPGrid caches row-indexed pages. Insertions and retirements therefore
      // require its wider positional influence stream even though remapped
      // power descriptors can carry the same immutable identities exactly.
      dirtyRowsOffsetWords: this.frontierAllocation.rowDeltaAffectedRowsOffsetWords,
      dirtyCountControlWord: 6 as const };
    this.firstOrderVCycle = new WebGPUOctreeSPGridVCycle(this.device, {
      ...section63Source, rowGeometry: structuredSource.rowGeometry, rowDelta,
    }, { dimensions: [this.dims.nx, this.dims.ny, this.dims.nz], rowCapacity,
      finestCellWidth: this.scene.container.width_m / this.dims.nx,
      compileHierarchicalExecutor: true,
      deferPipelineCompilation: this.deferPipelineCompilation,
    });
    if (this.firstOrderVCycle.smootherContract?.degree === undefined) {
      throw new Error("Wide MGPCG requires the published V-cycle smoother contract");
    }
    const spgrid = this.firstOrderVCycle.section63Topology;
    // The paper evolves coarse octree phi regardless of whether the optional
    // factor-1/factor-4/factor-8 interface band exists. It is also the complete
    // inside/outside and cell-centre boundary authority in coarse-only mode.
    this.powerCoarseLevelSet = new WebGPUOctreeCoarseLevelSet(this.device, rowCapacity);
    const airSupportLayout = planOctreeAirVelocitySupport(
      rowCapacity, structured.plan.slotCapacity, this.device.limits.minStorageBufferOffsetAlignment,
      this.dims.nx * this.dims.ny * this.dims.nz,
      octreeAirSupportFootprintCapacity(rowCapacity,
        this.dims.nx * this.dims.ny * this.dims.nz),
    );
    this.powerCoarseLevelSetSchedule = new WebGPUOctreePowerCoarseLevelSet(
      this.device, this.powerCoarseLevelSet, this.powerTopology.source,
      structured.plan.slotCapacity * 16,
      airSupportLayout,
      this.coarseOnlySurfaceTracking ? airSupportLayout.ownerDirectoryCellCapacity : 0,
    );
    if (this.coarseOnlySurfaceTracking) {
      const coarseCell = {
        x: this.scene.container.width_m / this.dims.nx,
        y: this.scene.container.height_m / this.dims.ny,
        z: this.scene.container.depth_m / this.dims.nz,
      };
      this.coarseOnlySummary = new WebGPUOctreeCoarseSummary(this.device,
        this.powerCoarseLevelSetSchedule.sampleSource,
        [this.dims.nx, this.dims.ny, this.dims.nz], {
          arena: this.powerCoarseLevelSetSchedule.selectorRows,
          layout: airSupportLayout,
          rowVelocities: structuredSource.rowVelocities,
          initialPhi: initialOctreeLevelSet(this.scene, this.dims, coarseCell),
          physicalCellSize: coarseCell.x,
          timestep_s: this.scene.numerics.maxDt_s,
          maximumLeafSize: this.maxLeafSize,
        });
      this.info.allocatedBytes += this.coarseOnlySummary.plan.allocatedBytes;
      this.workAccounting.setAuthorityBytes("coarse-summary",
        this.coarseOnlySummary.plan.allocatedBytes);
    }
    const coarseDirectory = this.powerCoarseLevelSetSchedule.sampleSource.directory;
    // Binding 15 is the compact coarse-phi directory for the mandatory power
    // topology/pressure authority.
    this.groups = {
      ab: this.createProjectionGroup(this.pressureA, this.pressureB, coarseDirectory),
      ba: this.createProjectionGroup(this.pressureB, this.pressureA, coarseDirectory),
    };
    this.candidateRowGroups = {
      fromA: this.createProjectionGroup(this.pressureA, this.candidatePressure, coarseDirectory,
        this.candidateLeafHeaders),
      fromB: this.createProjectionGroup(this.pressureB, this.candidatePressure, coarseDirectory,
        this.candidateLeafHeaders),
    };
    const pressureSummaryDirectory = this.globalFineSummaries?.directory
      ?? this.coarseOnlySummary?.directory ?? this.unpublishedFineSummaryDirectory;
    this.fineSummarySizingGroup = this.createProjectionGroup(
      pressureSummaryDirectory, this.pressureB,
      coarseDirectory);
    // Structural dirtiness compares the current pressure-owner decisions, not
    // the fine payload transaction. This stable group keeps the fine summary,
    // persistent topology-tile membership, and compact coarse authority
    // together across A/B fine generations.
    this.topologyDecisionGroup = this.createProjectionGroup(
      pressureSummaryDirectory,
      this.topologyResidency.topologyTileStateBuffer,
      coarseDirectory,
    );
    this.fineSeedAdapter?.setCoarsePhiSource(
      this.powerCoarseLevelSetSchedule.sampleSource,
    );
    this.fineSeedAdapter?.setStructuredVelocitySource(structuredSource);
    if (this.globalFineSourceA && this.globalFineSourceB) {
      this.fineToPowerCoarseLevelSet = new WebGPUFineToCoarseLevelSet(this.device, rowCapacity,
        this.globalFineSourceA.plan.maximumResidentBricks * this.globalFineSourceA.plan.samplesPerBrick);
      const compactCoarse = this.powerCoarseLevelSetSchedule.sampleSource;
      this.globalFineTopologyAB = new WebGPUFineLevelSetTopology(
        this.device, this.globalFineSourceA, this.globalFineSourceB, compactCoarse.wgsl(9),
        this.deferPipelineCompilation,
      );
      this.globalFineTopologyBA = new WebGPUFineLevelSetTopology(
        this.device, this.globalFineSourceB, this.globalFineSourceA, compactCoarse.wgsl(9),
        this.deferPipelineCompilation,
      );
      const changedKeysOffsetWords = this.globalFineTopologyAB.pageDeltaLayout.changedKeysOffsetWords;
      if (changedKeysOffsetWords !== this.globalFineTopologyBA.pageDeltaLayout.changedKeysOffsetWords) {
        throw new Error("Fine topology A/B page-delta layouts disagree");
      }
      // Deferred Power resources are created after the projection's initial
      // parameter upload. Publish the exact producer ABI only now, when both
      // page-delta layouts exist; leaving the constructor's zero sentinel in
      // this word rejects every recurring dirty-tile transaction.
      this.device.queue.writeBuffer(this.params, 36, new Uint32Array([changedKeysOffsetWords]));
      // Each destination generation consumes the exact delta published by
      // the topology transaction that authors that destination.
      this.globalFineRedistanceA = new WebGPUFineLevelSetRedistance(
        this.device, this.globalFineSourceA, this.globalFineTopologyBA,
        {
          deferPipelineCompilation: this.deferPipelineCompilation,
          maximumRequiredJfaStride: maximumFineLevelSetJFAStride(
            planFineLevelSetBandFineCells(this.fineLevelSetBandCells,
              this.globalFineSourceA.plan.fineFactor).redistanceBandFineCells),
        },
      );
      this.globalFineRedistanceB = new WebGPUFineLevelSetRedistance(
        this.device, this.globalFineSourceB, this.globalFineTopologyAB,
        {
          deferPipelineCompilation: this.deferPipelineCompilation,
          maximumRequiredJfaStride: maximumFineLevelSetJFAStride(
            planFineLevelSetBandFineCells(this.fineLevelSetBandCells,
              this.globalFineSourceB.plan.fineFactor).redistanceBandFineCells),
        },
      );
    }
    if (sceneHasTerrain(this.scene) || this.scene.rigidBodies.length > 0) {
      this.powerSolidVertices = new WebGPUOctreeSolidVertexSdf(
        this.device, rowCapacity, this.candidateLeafHeaders,
        this.powerRowDelta.rows, this.resources.terrain,
        this.resources.rigidBodies, this.powerRowDelta.controlOffsetWords,
      );
    }
    // One-step-lagged unilateral-contact active set: the projection stage
    // marks liquid rows holding tension against the closed ceiling; the next
    // step's boundary rebuild opens those rows' world faces so the solve
    // itself computes the separation with consistent divergence.
    this.structuredSeparationMask = this.device.createBuffer({
      label: "Structured ceiling separation mask",
      size: this.dims.nx * this.dims.ny * this.dims.nz * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.structuredBoundary = new WebGPUStructuredBoundaryCoefficients(this.device, {
      structured: structuredSource,
      separationMask: this.structuredSeparationMask,
      coarse: this.powerCoarseLevelSetSchedule.sampleSource,
      solid: this.powerSolidVertices?.source,
      rigidBodies: this.resources.rigidBodies,
      bodyCount: this.scene.rigidBodies.length,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      physicalCellSize: this.scene.container.width_m / this.dims.nx,
      closedBoundaryMask: structuredClosedBoundaryMask(this.scene.container.top === "closed"),
      // Always bound; `analyticBootstrap` above decides whether it is ever
      // sampled, exactly as the projection layout does.
      bootstrapLevelSet: this.surfaceState.texture,
      ...(this.analyticSparseBootstrap ? { analyticBootstrap: {
        initialCondition: this.scene.fluid.initialCondition,
        fillFraction: this.scene.container.fillFraction,
        damBreakDimensions: this.scene.fluid.initialDamBreakDimensions_m,
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
    const vector = (label: string) => this.device.createBuffer({
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
      this.device, {
        rowCount: this.compaction,
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
          this.solveTailPolicy.boundarySmoothingIterations,
        ),
      },
    );
    this.pipelinedMGPCG = new WebGPUOctreePipelinedMGPCG(this.device, {
      coefficients: section63Source.coefficients,
      rhs: this.structuredDivergenceRhs,
      rowCount: this.compaction,
      rowDispatch: structuredSource.liveRowDispatch,
      acceptedAuthority: this.structuredBoundary.control,
      operator: accurateOperator,
      preconditioner: this.section43HybridPreconditioner,
      vectors: this.pipelinedMGPCGVectors,
    }, {
      rowCapacity,
      relativeTolerance: this.solveTailPolicy.relativeTolerance,
      maximumIterations: this.solveTailPolicy.encodedOuterIterations,
      hardIterationCeiling: this.solveTailPolicy.hardOuterIterationCeiling,
    });
    this.pressureSolverControl = this.pipelinedMGPCG.control;
    this.topologyEpoch = new WebGPUOctreeTopologyEpoch(this.device, {
      ownerArena: this.ownerPages.arena,
      ownerCandidate: this.ownerPages.candidateTransaction,
      frontier: this.leafFrontier,
      descriptorCandidateControl: this.powerDescriptor.control,
      topologyCandidateControl: this.powerTopology.control,
      structuredCandidateControl: structured.candidateControl,
      structuredAcceptedControl: structured.control,
      boundaryCandidateControl: this.structuredBoundary.candidateControl,
      spgridCandidateControl: this.firstOrderVCycle.candidateControl,
      candidateLeafHeaders: this.candidateLeafHeaders,
      acceptedLeafHeaders: this.leafHeaders,
      candidatePressure: this.candidatePressure,
      pressureA: this.pressureA,
      pressureB: this.pressureB,
      rowCountControl: this.compaction,
    }, { rowCapacity, slotCapacity: structured.plan.slotCapacity,
      catalogVersion: OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version,
      carryPressureHistory: false });
    if (typeof process !== "undefined" && process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1") {
      this.readyEpochAudit = this.device.createBuffer({
        label: "Diagnostic coupled epoch state after ready commit",
        size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.readyFrontierAudit = this.device.createBuffer({
        label: "Diagnostic frontier state after ready commit",
        size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.readyCompactionAudit = this.device.createBuffer({
        label: "Diagnostic compaction state after ready commit",
        size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
    this.airVelocitySupport = new WebGPUOctreeAirVelocitySupportProducer(this.device, {
      structured: structuredSource,
      topology: this.powerTopology.source,
      owners: this.ownerPages,
      boundaryEpoch: { buffer: this.structuredBoundary.control, offsetWords: 4 },
      liquidMask: this.structuredBoundary.liquidMask,
      sharedArena: this.powerCoarseLevelSetSchedule.selectorRows,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      closedBoundaryMask: structuredClosedBoundaryMask(this.scene.container.top === "closed"),
      maximumLeafSize: this.maxLeafSize,
      // Air-extension demand must cover the complete characteristic and the
      // terminal interpolation owner around it. Using one fine factor happened
      // to cover the oversampled factor-4/8 bricks, but factor 1 packs four
      // coarse cells into each brick: its RK2 midpoint/backtrace reaches two
      // owners away and the velocity interpolation at that point reaches one
      // owner farther. Omitting that last owner leaves otherwise-valid
      // transported samples with no velocity authority.
      maximumDisplacementFineCells: this.globalFineLevelSet
        ? (this.globalFineLevelSet.plan.fineFactor === 1
          ? planFineLevelSetBandFineCells(this.fineLevelSetBandCells, 1)
            .maximumBacktraceFineCells + 1
          : this.globalFineLevelSet.plan.fineFactor)
        : 4,
      ...(this.globalFineSourceA && this.globalFineSourceB ? {
        fineSources: [this.globalFineSourceA, this.globalFineSourceB] as const,
        transportBandFineCells: planFineLevelSetBandFineCells(this.fineLevelSetBandCells,
          this.globalFineSourceA.plan.fineFactor).transportBandFineCells,
      } : {}),
    }, this.deferPipelineCompilation);
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
    this.structuredDynamics = new WebGPUStructuredVelocityDynamics(this.device, {
      structured: structuredSource, topology: this.powerTopology!.source, pressure: this.pressureA,
      separationMask: this.structuredSeparationMask,
      divergenceRhs: this.structuredDivergenceRhs,
      liquidMask: this.structuredBoundary!.liquidMask,
      solidNormalVelocities: this.structuredBoundary!.solidNormalVelocities,
      rigidBodies: this.resources.rigidBodies,
      rigidExchange: this.resources.rigidExchange,
      boundaryWorksets: this.structuredBoundary!.worksets,
      boundaryControl: this.structuredBoundary!.control,
      selectorRows: this.powerCoarseLevelSetSchedule!.selectorRows,
      selectorStride: this.powerCoarseLevelSetSchedule!.selectorStride,
      selectorOffsetWords: this.powerCoarseLevelSetSchedule!.plan.selectorOffsetWords,
      airSupportLayout,
      bodyCount: this.scene.rigidBodies.length,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      physicalCellSize: this.scene.container.width_m / this.dims.nx,
      closedBoundaryMask: structuredClosedBoundaryMask(this.scene.container.top === "closed"),
    }, this.deferPipelineCompilation);
    if (this.globalFineSourceA && this.globalFineSourceB) {
      const fineTransportResources = {
        structured: structuredSource, topology: this.powerTopology.source,
        airSupport: {
          arena: this.airVelocitySupport.source.arena,
          layout: this.airVelocitySupport.source.plan.support,
          boundaryControl: this.structuredBoundary.control,
        },
        dimensions: [this.dims.nx, this.dims.ny, this.dims.nz] as const,
        physicalCellSize: this.scene.container.width_m / this.dims.nx,
        maximumLeafSize: this.maxLeafSize,
      };
      this.globalFineTransportA = new WebGPUFineLevelSetTransport(
        this.device, this.globalFineSourceA, fineTransportResources,
        this.deferPipelineCompilation);
      this.globalFineTransportB = new WebGPUFineLevelSetTransport(
        this.device, this.globalFineSourceB, fineTransportResources,
        this.deferPipelineCompilation);
    }
    this.powerVolumes = this.device.createBuffer({ label: "Octree physical power-cell volumes", size: rowCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    if (this.globalFineSourceA && this.globalFineSourceB && this.powerCoarseLevelSet) {
      const coarseVolumeSource = { headers: this.leafHeaders, records: this.powerCoarseLevelSet.records,
        physicalVolumes: this.powerVolumes,
        sampleDirectory: this.powerCoarseLevelSetSchedule!.sampleSource.directory,
        publicationControl: this.powerCoarseLevelSetSchedule!.control,
        rowCount: this.compaction,
        dimensions: [this.dims.nx, this.dims.ny, this.dims.nz] as const,
        physicalCellSize: this.scene.container.width_m / this.dims.nx,
        maximumLeafSize: this.maxLeafSize,
        sampleRowCapacity: this.powerCoarseLevelSetSchedule!.sampleSource.rowCapacity };
      this.globalFineVolumeA = new WebGPUFineLevelSetVolumeCorrection(
        this.device, this.globalFineSourceA, coarseVolumeSource,
        undefined, this.deferPipelineCompilation,
      );
      this.globalFineVolumeB = new WebGPUFineLevelSetVolumeCorrection(
        this.device, this.globalFineSourceB, coarseVolumeSource, this.globalFineVolumeA.control,
        this.deferPipelineCompilation,
      );
    }
    this.powerVolumeParams = this.device.createBuffer({ label: "Octree power-volume parameters", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const cellVolume = (this.scene.container.width_m / this.dims.nx)
      * (this.scene.container.height_m / this.dims.ny)
      * (this.scene.container.depth_m / this.dims.nz);
    const data = new Float32Array(4); data[0] = cellVolume;
    this.device.queue.writeBuffer(this.powerVolumeParams, 0, data);
    const powerVolumeProfile = performanceShaderVariant();
    const powerVolumeActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "octree/power-volume",
      profile: powerVolumeProfile,
    });
    const powerVolumeVariant = powerVolumeActivity.module(
      octreePowerVolumeActivityShader(powerVolumeActivity),
      `octree/power-volume/${powerVolumeProfile.cacheKey}`,
    );
    const shaderModule = this.device.createShaderModule({
      label: "Publish physical octree power volumes",
      code: powerVolumeVariant.code,
    });
    this.initializePowerVolumePipeline = async () => {
      if (this.powerVolumePipeline) return;
      this.powerVolumePipeline = powerVolumeActivity.registerPipeline(
        await this.device.createComputePipelineAsync({
          label: "Publish physical octree power volumes", layout: "auto",
          compute: { module: shaderModule, entryPoint: "publishPowerVolumes" },
        }),
      );
      this.powerVolumeGroup = this.device.createBindGroup({
        layout: this.powerVolumePipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: this.powerVolumeParams! } },
          { binding: 1, resource: { buffer: this.powerTopology!.metrics } },
          { binding: 2, resource: { buffer: this.leafHeaders } },
          { binding: 3, resource: { buffer: this.compaction } },
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
      topologyAB: this.globalFineTopologyAB?.allocatedBytes ?? 0,
      topologyBA: this.globalFineTopologyBA?.allocatedBytes ?? 0,
      redistanceA: this.globalFineRedistanceA?.allocatedBytes ?? 0,
      redistanceB: this.globalFineRedistanceB?.allocatedBytes ?? 0,
      transportA: this.globalFineTransportA?.plan.allocatedBytes ?? 0,
      transportB: this.globalFineTransportB?.plan.allocatedBytes ?? 0,
      volumeA: this.globalFineVolumeA?.allocatedBytes ?? 0,
      volumeB: this.globalFineVolumeB?.allocatedBytes ?? 0,
    });
    this.info.powerDiagramAllocatedBytes = powerAllocated;
    this.info.allocatedBytes += powerAllocated + fineAllocated;
    this.info.globalFineLevelSetAllocatedBytes += fineAllocated;
    this.workAccounting.setAuthorityBytes("power", powerAllocated);
    this.workAccounting.setAuthorityBytes("fine-level-set", fineAllocated);
    this.workAccounting.setScratchBytes("pressure-mgpcg",
      (this.pipelinedMGPCG?.allocatedBytes ?? 0)
      + (this.section43HybridPreconditioner?.allocatedBytes ?? 0));
    this.workAccounting.setScratchBytes("multigrid", this.firstOrderVCycle.allocatedBytes);
    this.workAccounting.sealAllocationInventory();
    this.info.powerDiagramReady = true;
    this.info.powerDiagramAuthoritative = Boolean(this.structuredVelocity && this.structuredBoundary)
      && (!sceneHasTerrain(this.scene) || Boolean(this.powerSolidVertices));
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

  private writeParams() {
    const data = new ArrayBuffer(160);
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
    new Float32Array(data, 48, 4).set([1e-8, 0.01, 2.2, this.interfaceRefinementBandCells]);
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
    // pressureCapacity.z carries the surface grading width because the
    // projection uniform ABI already reserves this otherwise-unused word.
    new Uint32Array(data, 128, 4).set([
      this.pressureCapacity.rowCapacity,
      this.analyticSparseBootstrap
        ? (this.scene.fluid.initialCondition === "dam-break" ? 2 : 1)
        : 0,
      this.surfaceRefinementGradingLayers,
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
      case "cold-topology": this.encodeColdBootstrapRebuild(encoder); break;
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
      this.structuredBoundary?.retireAnalyticBootstrap();
    }
    this.powerCoarseLevelSetSchedule?.retireSubmittedEncoder(encoder);
  }

  /** Tail of substep N: build and validate only the inactive frontier epoch. */
  encodeInactiveTopologyCandidate(
    encoder: GPUCommandEncoder,
    analyticColdBootstrap = false,
    coldFullRebuild = false,
  ) {
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
    // for the next topology rebuild. Queue this expected generation before the
    // command buffer; later surface publication uses its own parameter buffer.
    if (this.powerCoarseLevelSetSchedule) {
      const generation = this.powerCoarseLevelSetGeneration & 0x3fff_ffff;
      const flags = 1 | (generation << 2);
      this.device.queue.writeBuffer(this.params, 140, new Uint32Array([flags >>> 0]));
    }
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
    const gradingFixpoint = octreeGradingFixpointEnabled();
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
      // Carry this round's verdict to the next. Dispatches inside one compute
      // pass are ordered, so the balance kernels read what this writes with no
      // copy, no pass boundary and no indirect republication -- a producer and
      // a consumer in separate dispatches, never a barrier. The last round has
      // no successor to inform.
      if (gradingFixpoint && round + 1 < this.balanceRounds) {
        pass.setPipeline(this.advanceGradingRoundPipeline);
        pass.setBindGroup(0, this.groups.ab);
        pass.dispatchWorkgroups(1);
      }
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
      this.latestPressureInA ? this.candidateRowGroups.fromA : this.candidateRowGroups.fromB,
    );
    this.encodeInactiveCoupledPowerCandidate(encoder);
    return true;
  }

  /** Complete the inactive epoch after frontier/owner publication. Every
   * component writes only candidate storage or the inactive structured bank;
   * the final singleton is the sole cross-component validation reduction. */
  private encodeInactiveCoupledPowerCandidate(encoder: GPUCommandEncoder): void {
    if (this.coarseDynamics.backend === "losasso") {
      const backend = this.losassoBackend;
      if (!backend || this.candidatePowerGeneration === 0) {
        throw new Error("Inactive topology candidate requires the reduced Losasso authority");
      }
      const broker = new PassBroker(encoder);
      backend.encodeCandidatePublication(broker, {
        leafHeaders: this.candidateLeafHeaders,
        frontier: this.leafFrontier,
        ownerArena: this.ownerPages.arena,
        ownerCandidateTransaction: this.ownerPages.candidateTransaction,
        solidCells: this.solidCells,
        rigidBodies: this.resources.rigidBodies,
      });
      broker.fence("inactive Losasso axis-face candidate published");
      return;
    }
    const descriptor = this.powerDescriptor, topology = this.powerTopology;
    const structured = this.structuredVelocity, boundary = this.structuredBoundary;
    const epoch = this.topologyEpoch;
    // A pending target is not redistanced/committed yet. The ready flip may
    // only demand support from the currently settled fine publication.
    const fine = this.globalFineCurrentIsA ? this.globalFineSourceA : this.globalFineSourceB;
    if (!descriptor || !topology || !structured || !boundary || !epoch) {
      throw new Error("Inactive topology epoch requires every coupled power authority");
    }
    const generation = this.candidatePowerGeneration;
    if (generation === 0) throw new Error("Inactive topology candidate has no attempt generation");
    const broker = new PassBroker(encoder);
    const dimensions: [number, number, number] = [this.dims.nx, this.dims.ny, this.dims.nz];
    const spacing: [number, number, number] = [
      this.scene.container.width_m / this.dims.nx,
      this.scene.container.height_m / this.dims.ny,
      this.scene.container.depth_m / this.dims.nz,
    ];
    descriptor.encodeCandidate(broker, this.candidateLeafHeaders, this.ownerPages.arena, {
      dimensions, maximumLeafSize: this.maxLeafSize,
      ownerCandidateControl: this.ownerPages.candidateTransaction, generation,
      rowDelta: this.powerRowDelta,
    });
    topology.encodeCandidate(broker, descriptor.candidateDescriptors, spacing,
      this.powerRowDelta);
    structured.encodeCandidate(broker, generation, 0, topology.candidateMetrics,
      this.candidateLeafHeaders);
    if (!this.structuredDynamics) {
      throw new Error("Inactive topology candidate requires structured velocity transfer");
    }
    this.structuredDynamics.encodeTopologyTransferCandidate(broker);
    structured.encodeCandidateReconstruction(broker, topology.candidateMetrics,
      this.candidateLeafHeaders);
    this.powerSolidVertices?.encode(broker, { dimensions, physicalSpacing: spacing, generation,
      terrainEnabled: sceneHasTerrain(this.scene), bodyCount: this.scene.rigidBodies.length });
    boundary.encodeCandidate(broker, fine, structured.candidateControl);
    this.firstOrderVCycle.encodeCandidateSetup(broker, {
      solverControl: this.pressureSolverControl, rowCount: this.compaction,
      sourceControl: structured.candidateControl,
      topologyMetrics: topology.candidateMetrics,
    });
    epoch.encodeCandidateValidation(broker, generation);
  }

  /** Beginning of substep N+1: sole coupled owner/frontier epoch flip. */
  /** The substep's body-force increment for the Section 5 extension: air
   * vectors are rebuilt from projected liquid seeds every epoch, so the
   * producer folds exactly one g*dt into each reconstructed air vector. */
  private airSupportGravityImpulse(
    dt_s = this.powerTimestep_s,
  ): [number, number, number] {
    const gravity = this.scene.fluid.gravity_m_s2;
    return [gravity.x * dt_s, gravity.y * dt_s, gravity.z * dt_s];
  }

  encodeReadyTopologyFlip(encoder: GPUCommandEncoder): void {
    if (this.coarseDynamics.backend === "losasso") {
      if (this.candidatePowerGeneration === 0
        && this.coarseDynamics.topology.advancesPerEpoch > 1) {
        this.info.topologyReused = true;
        return;
      }
      if (!this.losassoReadyCommit || !this.losassoBackend
        || this.candidatePowerGeneration === 0) {
        throw new Error("Ready topology flip requires a complete inactive Losasso candidate");
      }
      const broker = new PassBroker(encoder);
      // The row/pressure and reduced-operator copies validate the exact same
      // candidate transaction immediately before the owner selector flips.
      this.losassoReadyCommit.encodeReadyCommit(broker);
      this.losassoBackend.encodeReadyCommit(broker, {
        frontier: this.leafFrontier,
        ownerCandidateTransaction: this.ownerPages.candidateTransaction,
      });
      this.ownerPages.encodeReadyCommit(broker);
      const currentFine = this.globalFineCurrentIsA
        ? this.globalFineSourceA : this.globalFineSourceB;
      if (this.globalFineBootstrapped && currentFine && this.losassoCoarsePhi) {
        // Losasso et al. 2004 Section 4, equations 5-6, requires the pressure
        // operator to use the same face boundary state as divergence. A ready
        // topology commit replaces the accepted face records, so reapply the
        // current fine-phi ghost distances (including unilateral closed-wall
        // separation) before building the hierarchy or solving this epoch.
        // See docs/papers/losasso-2004-octree-water-smoke.txt, Sections 4.1-4.2.
        this.losassoCoarsePhi.encode(broker, currentFine,
          this.losassoCoarsePhiInput());
        this.losassoConditionedOperator?.encodeAfterGhostDistances(broker);
      }
      this.losassoBackend.encodeHierarchyRefresh(broker, this.leafHeaders);
      this.refreshLosassoProjectionGroups();
      // Candidate publication migrates the lagged wet-face field by geometric
      // identity. Reconstruct row motion immediately so encodeSurface never
      // observes row indices from the retired epoch.
      this.losassoRowMotion?.encode(broker);
      broker.fence("accepted Losasso row and owner epoch published");
      this.activePowerGeneration = this.candidatePowerGeneration;
      this.candidatePowerGeneration = 0;
      this.info.topologyReused = false;
      return;
    }
    const descriptor = this.powerDescriptor, topology = this.powerTopology;
    const structured = this.structuredVelocity, boundary = this.structuredBoundary;
    const epoch = this.topologyEpoch;
    if (!descriptor || !topology || !structured || !boundary || !epoch
      || this.candidatePowerGeneration === 0) {
      throw new Error("Ready topology flip requires a complete inactive coupled candidate");
    }
    const acceptedGeneration = this.candidatePowerGeneration;
    const broker = new PassBroker(encoder);
    epoch.encodeReadyCommitGate(broker, acceptedGeneration);
    this.ownerPages.encodeReadyCommit(broker);
    descriptor.encodeReadyCommit(broker);
    topology.encodeReadyCommit(broker);
    structured.encodeReadyCommit(broker);
    boundary.encodeReadyCommit(broker);
    this.firstOrderVCycle.encodeReadySetupCommit(broker, {
      solverControl: this.pressureSolverControl, rowCount: this.compaction,
    });
    if (!this.airVelocitySupport) {
      throw new Error("Ready topology flip requires the Section 5 air-support producer");
    }
    // Aanjaneya et al. Section 5 first maps the migrated projected power-face
    // field to ordinary faces, extends it outside liquid, and maps it back
    // before the newly accepted epoch may be sampled.
    this.airVelocitySupport.encode(broker, acceptedGeneration,
      this.globalFineBootstrapped ? (this.globalFineCurrentIsA ? 0 : 1) : undefined,
      this.airSupportGravityImpulse(), "topology-commit");
    // This helper owns its broker. Close the publication pass before returning
    // so both the cold checkpoint (which finishes the encoder immediately) and
    // the recurring caller can safely append or finish commands.
    broker.fence("accepted Section 5 air-support epoch published");
    if (this.readyEpochAudit) {
      broker.copyBufferToBuffer(epoch.state, 0, this.readyEpochAudit, 0, 64);
      broker.copyBufferToBuffer(this.leafFrontier, 0, this.readyFrontierAudit!, 0, 64);
      broker.copyBufferToBuffer(this.compaction, 0, this.readyCompactionAudit!, 0, 64);
    }
    this.activePowerGeneration = acceptedGeneration;
    this.candidatePowerGeneration = 0;
    this.info.topologyReused = false;
  }

  finishTopologyCandidate() { this.info.topologyReuseCount += 1; }

  /**
   * Tail scheduling for the Losasso k-advance epoch. The fine band is still
   * transported every advance; its construction-time page plan includes the
   * corresponding extra dilation rings. Power 2017 always takes the legacy
   * every-advance path.
   */
  encodeInactiveTopologyCandidateIfDue(encoder: GPUCommandEncoder): boolean {
    const cadence = this.coarseDynamics.topology.advancesPerEpoch;
    if (this.coarseDynamics.backend === "power2017" || cadence === 1) {
      return this.encodeInactiveTopologyCandidate(encoder);
    }
    this.topologyCadenceCursor += 1;
    if (this.topologyCadenceCursor < cadence) {
      this.info.topologyReused = true;
      this.info.topologyReuseCount += 1;
      return false;
    }
    this.topologyCadenceCursor = 0;
    return this.encodeInactiveTopologyCandidate(encoder);
  }
  get pressureSolverLabel() {
    if (this.coarseDynamics.backend === "losasso") {
      const budget = this.losassoBackend?.solverIterationBudget
        ?? this.info.pressureIterationBudget;
      return `Octree Losasso MGPCG · exact-reduction wide solve · plain first-order V-cycle · up to ${budget} iterations`;
    }
    const budget = this.pipelinedMGPCG?.iterationBudget ?? this.info.pressureIterationBudget;
    const levels = this.firstOrderVCycle?.plan.levelCount ?? 0;
    return `Octree power MGPCG · row-parallel exact-reduction executor · Section 4.3 fixed schedule · up to ${budget} iterations · ${levels}-level L1 V-cycle`;
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
    if (this.powerTimestep_s > 0) dynamics.encodeAdvection(
      broker, this.powerTimestep_s, this.surfaceInflow,
    );
    dynamics.encodeForcesAndDivergence(
      broker, this.powerTimestep_s, this.scene.fluid.density_kg_m3, [
        this.scene.fluid.gravity_m_s2.x,
        this.scene.fluid.gravity_m_s2.y,
        this.scene.fluid.gravity_m_s2.z,
      ], this.surfaceInflow);
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
    if (!dynamics || pressure !== this.pressureA && pressure !== this.pressureB) {
      throw new Error("Structured projection pressure buffer is not an accepted solve target");
    }
    dynamics.encodeProjection(broker, this.powerTimestep_s, this.scene.fluid.density_kg_m3, [
      this.scene.fluid.gravity_m_s2.x,
      this.scene.fluid.gravity_m_s2.y,
      this.scene.fluid.gravity_m_s2.z,
    ], pressure, this.dynamicCouplingBodyCount, this.surfaceInflow);
    if (!this.airVelocitySupport || this.activePowerGeneration === 0) {
      throw new Error("Structured projection requires an accepted Section 5 air-support epoch");
    }
  }

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
    if (this.coarseDynamics.backend === "losasso") {
      return this.encodeLosasso(encoder, options, scope);
    }
    const solveBudget = this.pipelinedMGPCG?.iterationBudget
      ?? this.solveTailPolicy.encodedOuterIterations;
    this.workAccounting.beginSubstep();
    this.info.pressureIterationBudget = solveBudget;
    this.info.pressureIterationHardBudget = this.solveTailPolicy.hardOuterIterationCeiling;
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
    const initialInA = !this.latestPressureInA;
    const pressureBroker = new PassBroker(encoder);
    encoder = this.encodeNativePowerAssembly(
      encoder,
      options?.productionBoundary,
      options?.productionBoundary && fineEngineSplits ? undefined : pressureBroker,
      fineEngineSplits,
    );
    const pressureIn = initialInA ? this.pressureA : this.pressureB;
    const pressureOut = initialInA ? this.pressureB : this.pressureA;
    if (!this.structuredVelocity) throw new Error("Pressure solve requires the accepted structured authority");
    const solveBroker = new PassBroker(encoder);
    const pipelined = this.pipelinedMGPCG;
    if (!pipelined) throw new Error("Row-parallel pressure executor was not constructed");
    // The SPGrid hierarchy and compiled A2 images are GPU-published before
    // the wide recurrence consumes them. Every following stencil/smoother
    // launch is row/page parallel; only the exact integer scalar finish is a
    // singleton.
    this.firstOrderVCycle.encodeSetup(solveBroker, {
      solverControl: this.pressureSolverControl, rowCount: this.compaction,
    });
    pipelined.encode(solveBroker, {
      pressureSeed: pressureIn,
      pressureOut,
      encodedIterationBudget: solveBudget,
    });
    this.latestPressureInA = !initialInA;
    // Stage solve feedback (residual sums + row/entry counts) while this
    // encoder still owns write ordering on compaction; the async diagnostics
    // poll then reads the staging buffer without racing the next rebuild.
    solveBroker.copyBufferToBuffer(
      this.compaction, this.compactionByteLength - 32, this.solveStats, 0, 32);
    splitProductionPhase("solveEngine", "mgpcgSolve");
    const finalInA = this.latestPressureInA;
    const projectionBroker = new PassBroker(encoder);
    this.encodeStructuredProjection(
      projectionBroker,
      finalInA ? this.pressureA : this.pressureB,
    );
    projectionBroker.fence("projected structured velocity published");
    if (scope === "power-operator-only") {
      // The explicit t=0 dependency chain publishes the fine level set next,
      // then owns each Section 5 transaction in its own named checkpoint.
      // This is a lifecycle boundary, not an alternate simulation path.
      return encoder;
    }
    splitProductionPhase(undefined, "structuredProjection");
    encoder = this.encodePendingFineSettlement(encoder, options?.productionBoundary);
    if (!this.coarseOnlySurfaceTracking && this.powerTimestep_s > 0) {
      if (!this.airVelocitySupport || !this.globalFineBootstrapped) {
        throw new Error("Live Section 5 support refresh requires the settled fine generation");
      }
      const supportBroker = new PassBroker(encoder);
      this.airVelocitySupport.encode(supportBroker, this.activePowerGeneration,
        this.globalFineCurrentIsA ? 0 : 1, this.airSupportGravityImpulse(), "settled-fine");
      supportBroker.fence("settled fine-demand air support published");
      encoder = supportBroker.commandEncoder();
    }
    const projectionTailBroker = new PassBroker(encoder);
    projectionTailBroker.fence("structured projection tail published");
    this.encodeOverlayMaterialization(encoder, finalInA);
    if (this.powerTimestep_s > 0) this.powerAdvancingPressureSteps += 1;
    splitProductionPhase("rowEngineB", "structuredProjectionTail");
    return encoder;
  }

  private encodeLosasso(
    encoder: GPUCommandEncoder,
    options?: { productionBoundary?: OctreeSemanticBoundary; step?: number },
    scope: "complete" | "power-operator-only" = "complete",
  ): GPUCommandEncoder {
    const backend = this.losassoBackend;
    if (!backend || this.activePowerGeneration === 0) {
      throw new Error("Losasso pressure step requires an accepted compact authority");
    }
    const solveBudget = backend.solverIterationBudget
      ?? this.solveTailPolicy.encodedOuterIterations;
    this.workAccounting.beginSubstep();
    this.info.pressureIterationBudget = solveBudget;
    this.info.pressureIterationHardBudget = this.solveTailPolicy.hardOuterIterationCeiling;
    const step = {
      dt_s: this.powerTimestep_s,
      gravity_m_s2: [
        this.scene.fluid.gravity_m_s2.x,
        this.scene.fluid.gravity_m_s2.y,
        this.scene.fluid.gravity_m_s2.z,
      ] as const,
      inflow: this.surfaceInflow,
    };
    let broker = new PassBroker(encoder);
    backend.encodeAdvection(broker, step);
    backend.encodeForcesAndDivergence(broker, step);
    broker.fence("Losasso first-order axis-face RHS published");
    if (options?.productionBoundary) {
      encoder = options.productionBoundary("structuredAdvectionBoundaryRhs", encoder);
      broker = new PassBroker(encoder);
    }
    const initialInA = !this.latestPressureInA;
    const pressureIn = initialInA ? this.pressureA : this.pressureB;
    const pressureOut = initialInA ? this.pressureB : this.pressureA;
    backend.encodeSolve(broker, { pressureSeed: pressureIn, pressureOut });
    this.latestPressureInA = !initialInA;
    broker.fence("Losasso wide exact-reduction pressure solve complete");
    if (options?.productionBoundary) {
      encoder = options.productionBoundary("mgpcgSolve", encoder);
      broker = new PassBroker(encoder);
    }
    backend.encodeProjection(broker, pressureOut, step, this.dynamicCouplingBodyCount);
    broker.fence("Losasso projected wet axis faces published");
    if (scope === "power-operator-only") return encoder;
    if (options?.productionBoundary) {
      encoder = options.productionBoundary("structuredProjection", encoder);
    }
    encoder = this.encodePendingFineSettlement(encoder, options?.productionBoundary);
    this.encodeOverlayMaterialization(encoder, this.latestPressureInA);
    if (this.powerTimestep_s > 0) this.powerAdvancingPressureSteps += 1;
    if (options?.productionBoundary) {
      encoder = options.productionBoundary("structuredProjectionTail", encoder);
    }
    return encoder;
  }

  /** Publish lazily allocated diagnostic textures from the live owner map.
   * The first overlay request materializes immediately, so reset-time grid
   * inspection never decodes zero-initialized topology storage as finest 1^3. */
  encodeOverlayMaterialization(encoder: GPUCommandEncoder, pressureInA = this.latestPressureInA) {
    if (!this.diagnosticGroups || !this.materializePipeline) return false;
    const broker = new PassBroker(encoder);
    const materialize = broker.compute({ label: "Materialize octree overlay fields" });
    materialize.setPipeline(this.materializePipeline);
    materialize.setBindGroup(0, pressureInA ? this.diagnosticGroups.pressureA : this.diagnosticGroups.pressureB);
    materialize.dispatchWorkgroupsIndirect(this.coldDispatch, 0);
    broker.fence("octree overlay fields materialized");
    return true;
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
      headers: this.leafHeaders,
      // Adaptive candidate construction updates `compaction` before the
      // accepted row headers/geometry flip. Restrict the transported fine
      // generation over the immutable accepted structured epoch instead of
      // mixing candidate N+1's count with accepted N's row identities.
      rowCount: structured.control,
      rowCountOffsetWords: 2,
      topologyControl: topology.control,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      physicalCellSize: this.scene.container.width_m / this.dims.nx,
      maximumLeafSize: this.maxLeafSize,
      allowValidatedProvisional,
    });
    this.powerCoarseLevelSetSchedule.encode(broker, {
      headers: this.leafHeaders,
      structured,
      rowCount: { buffer: structured.control, offset: 2 * 4, size: 4 },
      fineCorrection: {
        rowOffsets: correction.rowOffsets,
        contributions: correction.contributions,
        contributionCount: correction.counts,
        aggregated: correction.aggregated,
      },
    }, {
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      physicalCellSize: this.scene.container.width_m / this.dims.nx,
      dt: dt_s,
      maximumLeafSize: this.maxLeafSize,
      generation: fine.generation & 0x3fff_ffff,
    });
    this.powerCoarseLevelSetGeneration = fine.generation & 0x3fff_ffff;
  }

  /** Settle the transported fine generation after projected velocity and CPT
   * seeds exist, then re-correct coarse phi without a second advection. */
  private encodePendingFineSettlement(
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
    if (this.coarseDynamics.backend === "losasso" && !this.globalFineBootstrapped) {
      if (!this.losassoCoarsePhi) throw new Error("Losasso coarse-phi exchange is unavailable");
      // Bootstrap the generic coarse-volume directory before the first volume
      // correction. Later settlements already have the pre-force exchange.
      this.losassoCoarsePhi.encode(redistanceBroker, pending.target,
        this.losassoCoarsePhiInput());
      this.losassoConditionedOperator?.encodeAfterGhostDistances(redistanceBroker);
      this.losassoBackend?.encodeHierarchyCoefficientRefresh(redistanceBroker);
      this.refreshLosassoProjectionGroups();
    }
    pending.volume?.encode(redistanceBroker);
    if (this.coarseDynamics.backend === "losasso") {
      if (!this.losassoCoarsePhi) throw new Error("Losasso coarse-phi exchange is unavailable");
      // Volume correction may move fine phi. Republish the final conditioned
      // operator and hierarchy from the corrected, fully redistanced band.
      this.losassoCoarsePhi.encodeFieldRefresh(redistanceBroker, pending.target,
        this.losassoCoarsePhiInput());
      this.losassoConditionedOperator?.encodeAfterGhostDistances(redistanceBroker);
      this.losassoBackend?.encodeHierarchyCoefficientRefresh(redistanceBroker);
      this.refreshLosassoProjectionGroups();
    }
    encoder = redistanceBroker.commandEncoder();
    split("closestPointWaves", "fineRedistance");

    const restrictionBroker = new PassBroker(encoder);
    pending.topology.encodeFinalizePublication(restrictionBroker, {
      redistance: pending.redistance.control,
      ...(pending.volume ? { volume: pending.volume.control } : {}),
      ...(pending.transport ? { transport: pending.transport.control } : {}),
    });
    if (this.coarseDynamics.backend === "power2017") {
      this.encodeCoarsePhiCorrection(restrictionBroker, pending.target, pending.topology, 0);
    }
    if (this.coarseDynamics.backend === "power2017" && this.powerCoarseLevelSetSchedule) {
      const coarse = this.powerCoarseLevelSetSchedule.sampleSource;
      this.globalFineSummaries?.encode(restrictionBroker, pending.target, {
        buffer: pending.topology.pageDelta,
        layout: pending.topology.pageDeltaLayout,
      }, {
        directory: coarse.directory,
        control: coarse.control,
        delta: coarse.delta,
        deltaHeaderWords: coarse.deltaHeaderWords,
        deltaRecordWords: coarse.deltaRecordWords,
      });
    } else if (this.coarseDynamics.backend === "losasso" && this.losassoCoarsePhi) {
      this.globalFineSummaries?.encode(restrictionBroker, pending.target, {
        buffer: pending.topology.pageDelta,
        layout: pending.topology.pageDeltaLayout,
      }, this.losassoCoarsePhi.summaryCoarseSource());
    }
    if (this.coarseDynamics.backend === "losasso") {
      const backend = this.losassoBackend;
      if (!backend) throw new Error("Losasso extension-band backend is unavailable");
      backend.encodeExtensionBandPublication(restrictionBroker, pending.target);
      const advanceSerial = this.powerTimestep_s > 0
        ? this.powerAdvancingPressureSteps + 1 : 0;
      backend.encodeExtension(restrictionBroker, advanceSerial,
        this.activePowerGeneration);
      this.losassoRowMotion?.encode(restrictionBroker);
    }
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
      let coarseBootstrappedThisStep = false;
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
      const structuredSource = this.structuredVelocity?.source;
      if (this.powerCoarseLevelSet && this.powerCoarseLevelSetSchedule && structuredSource) {
        if (!this.powerCoarseLevelSetBootstrapped) {
          this.powerCoarseLevelSet.encodeBootstrapFromSurfaceLeaves(
            preparationBroker, this.fineSeedAdapter.leaves, structuredSource.liveRowDispatch,
          );
          this.powerCoarseLevelSetSchedule.encode(preparationBroker, {
            headers: this.leafHeaders, structured: structuredSource, rowCount: this.compaction,
          }, {
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            dt: 0,
            maximumLeafSize: this.maxLeafSize,
            generation: this.powerCoarseLevelSetGeneration & 0x3fff_ffff,
          });
          this.coarseOnlySummary?.encode(preparationBroker);
          this.powerCoarseLevelSetBootstrapped = true;
          coarseBootstrappedThisStep = true;
        }
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
        const compactCoarseEntry: GPUBindGroupEntry = this.coarseDynamics.backend === "losasso"
          ? this.losassoCoarsePhi!.fineTopologyEntry(9)
          : { binding: 9,
            resource: { buffer: this.powerCoarseLevelSetSchedule!.sampleSource.directory } };
        // Same planner as allocation. The final three cells cover the complete
        // 3-D trilinear stencil and its centre on the closed cutoff.
        const fineBandPlan = planFineLevelSetBandFineCells(
          this.fineLevelSetBandCells, this.globalFineLevelSet!.plan.fineFactor,
        );
        const { transportBandFineCells: bandCells,
          redistanceBandFineCells: redistanceBandCells, maximumBacktraceFineCells }
          = fineBandPlan;
        const transport = this.coarseDynamics.backend === "losasso"
          ? (this.globalFineCurrentIsA ? this.losassoFineTransportA : this.losassoFineTransportB)
          : (this.globalFineCurrentIsA ? this.globalFineTransportA : this.globalFineTransportB);
        let transportEncoded = false;
        // Adapter publication, coarse bootstrap and compact interface seeding
        // precede characteristic transport. Keep them out of the transport
        // bucket so the generic trace names the measured work.
        seedBroker.fence("fine interface seed publication complete");
        splitProductionPhase(undefined, "finePreparation");
        if (this.globalFineBootstrapped && transport
          && (structuredSource || this.coarseDynamics.backend === "losasso")) {
          const transportBroker = new PassBroker(encoder);
          this.lastGlobalFineTransport = transport;
          const completedTransportBroker = this.coarseDynamics.backend === "losasso"
            ? (transport as WebGPUOctreeLosassoFineTransport).encode(transportBroker, {
              timestep: dt_s,
              // S1 deliberately consumes the previous advance's settled W7
              // field. A ready topology flip may already have advanced the
              // wet-face epoch, while this lagged sampler remains physically
              // valid by geometric identity until S3e rebuilds it below.
              velocityEpoch: 0,
              boundaryPolicy: "closed-neumann",
              openTopBoundary: this.scene.container.top !== "closed",
              transportBandCells: bandCells,
              maximumBacktraceFineCells,
            })
            : (transport as WebGPUFineLevelSetTransport).encode(transportBroker, {
              timestep: dt_s,
              ...(inflow ? { inflow } : {}),
              boundaryPolicy: "closed-neumann",
              openTopBoundary: this.scene.container.top !== "closed",
              dynamicBoundary: this.scene.rigidBodies.length > 0,
              transportBandCells: bandCells,
              maximumBacktraceFineCells,
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
          warmClosestPoints: wasBootstrapped && this.coarseDynamics.backend !== "losasso",
        };
        // On recurring steps, coarse phi consumes the transported target before
        // any current-step force. Bootstrap first needs redistance to populate
        // the complete narrow band, so its sole correction occurs at settlement.
        if (wasBootstrapped && !coarseBootstrappedThisStep) {
          const coarseBroker = new PassBroker(encoder);
          if (this.coarseDynamics.backend === "losasso") {
            if (!this.losassoCoarsePhi) throw new Error("Losasso coarse-phi exchange is unavailable");
            this.losassoCoarsePhi.encodeFieldRefresh(coarseBroker, publicationTarget,
              this.losassoCoarsePhiInput());
            this.losassoConditionedOperator?.encodeAfterGhostDistances(coarseBroker);
            this.losassoBackend?.encodeHierarchyCoefficientRefresh(coarseBroker);
            this.refreshLosassoProjectionGroups();
          } else {
            this.encodeCoarsePhiCorrection(coarseBroker, publicationTarget,
              publicationTopology, dt_s, true);
          }
          coarseBroker.fence("transported fine and coarse phi published before forces");
          encoder = coarseBroker.commandEncoder();
        }
        if (!wasBootstrapped || dt_s === 0) {
          encoder = this.encodePendingFineSettlement(encoder, productionBoundary);
          if (this.coarseDynamics.backend === "power2017"
            && (!this.airVelocitySupport || !this.globalFineBootstrapped
              || this.activePowerGeneration === 0)) {
            throw new Error("Settled t=0 fine authority requires Section 5 air support");
          }
          if (this.airVelocitySupport) {
            const supportBroker = new PassBroker(encoder);
            this.airVelocitySupport.encode(supportBroker, this.activePowerGeneration,
              this.globalFineCurrentIsA ? 0 : 1, this.airSupportGravityImpulse(dt_s));
            supportBroker.fence("settled t=0 fine-demand air support published");
            encoder = supportBroker.commandEncoder();
          }
          if (productionBoundary) {
            encoder = productionBoundary("structuredProjectionTail", encoder);
          }
        }
      } else if (this.coarseOnlySurfaceTracking && this.powerCoarseLevelSetSchedule
        && structuredSource) {
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
            headers: this.leafHeaders,
            structured: structuredSource,
            rowCount: this.compaction,
          }, {
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            dt: dt_s,
            maximumLeafSize: this.maxLeafSize,
            generation: this.powerCoarseLevelSetGeneration,
          });
          this.coarseOnlySummary?.encode(preparationBroker);
        }
        encoder = preparationBroker.commandEncoder();
        if (!this.airVelocitySupport || this.activePowerGeneration === 0) {
          throw new Error("Coarse-only surface tracking requires Section 5 air support");
        }
        const supportBroker = new PassBroker(encoder);
        this.airVelocitySupport.encode(supportBroker, this.activePowerGeneration,
          undefined, this.airSupportGravityImpulse(dt_s), "settled-fine");
        // Cold bootstrap precedes the first air-support publication. Rebuild
        // the compact coarse tracker immediately afterward so generation zero
        // starts from complete velocity/owner support rather than an empty bank.
        if (coarseBootstrappedThisStep) this.coarseOnlySummary?.encode(supportBroker);
        supportBroker.fence("coarse-only air support published");
        encoder = supportBroker.commandEncoder();
        if (productionBoundary) {
          encoder = productionBoundary("structuredProjectionTail", encoder);
        }
      } else {
        throw new Error("Authoritative Section 5 fine-band pipeline is incomplete");
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
    if (this.coarseBackend === "losasso") {
      const backend = this.losassoBackend;
      if (!backend) throw new Error("Losasso solve diagnostics require the reduced backend");
      const solverControl = backend.solverControl ?? backend.sources.rowCount;
      const readback = this.device.createBuffer({
        label: "Octree Losasso live pressure diagnostics",
        size: 32 + 64,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = this.device.createCommandEncoder({
        label: "Read octree Losasso pressure diagnostics",
      });
      encoder.copyBufferToBuffer(backend.sources.operator.control, 0, readback, 0, 32);
      encoder.copyBufferToBuffer(solverControl, 0, readback, 32,
        Math.min(64, solverControl.size));
      this.device.queue.submit([encoder.finish()]);
      try {
        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const authority = new Uint32Array(mapped, 0, 8);
        const rows = authority[1] ?? 0;
        const valid = authority[3] === 1 && authority[4] === 0;
        this.info.pressureCapacityOverflow = !valid;
        this.info.frontierCapacityOverflow = !valid;
        this.info.frontierRequiredLeaves = rows;
        this.info.pressureRequiredRows = rows;
        this.info.pressureSampleCount = rows;
        this.info.liquidDofCount = rows;
        this.info.faceCount = authority[2] ?? 0;
        this.info.compressionRatio = rows
          / Math.max(1, this.dims.nx * this.dims.ny * this.dims.nz);
        this.residualRms = undefined;
        this.initialResidualRms = undefined;
        this.relativeResidual = undefined;
        this.applyMGPCGDiagnostics(new Uint32Array(mapped, 32, 16));
      } finally {
        if (readback.mapState === "mapped") readback.unmap();
        readback.destroy();
      }
      return;
    }
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

  private applyMGPCGDiagnostics(words: Uint32Array) {
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
  /** Diagnostic-only. Undefined unless the factor-one coarse tracker is the
   * surface authority. `completions < advances` means the raster consumed a
   * held surface on the difference, which is what intermittent publication
   * looks like from outside. */
  async readCoarseSurfaceTrackerReceipt() {
    return this.coarseOnlySummary?.readReceipt();
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
      coarseProjectionGroupsActive: this.powerCoarseLevelSetSchedule !== undefined,
      fineSeedCoarseNative: this.fineSeedAdapter?.hasCoarsePhiBindings === true,
      topologyUsesFineSeedCandidates: this.topologyWorklistReady,
      compactRendererSourceReady: this.nativePowerVelocityAuthority && this.fineSeedAuthority,
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
  stageSceneUpdate(scene: SceneDescription) {
    return this.sparseBrickWorld?.stageSceneUpdate(scene) ?? false;
  }
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]) {
    return this.sparseBrickWorld?.stageLivePrimitiveUpdates(updates) ?? false;
  }
  encodeSceneMaintenance(encoder: GPUCommandEncoder) {
    return this.sparseBrickWorld?.encodeSceneMaintenance(encoder) ?? false;
  }
  get sparseVoxelRenderSource() {
    if (this.sparseBrickWorld) {
      const source = this.sparseBrickWorld.inspectionSource;
      if (!source) void this.sparseBrickWorld.ensureInspectionSource();
      const currentBytes = this.sparseBrickWorld.allocatedBytes;
      this.info.allocatedBytes += currentBytes - this.sparseBrickWorldAccountedBytes;
      this.sparseBrickWorldAccountedBytes = currentBytes;
      return source;
    }
    return undefined;
  }
  get structuredVelocityControl() { return this.structuredVelocity?.control; }
  get structuredBoundaryControl() { return this.structuredBoundary?.control; }
  get structuredRowVelocities() { return this.structuredVelocity?.source.rowVelocities; }
  get structuredAuthority() { return this.structuredVelocity?.source.authority; }
  get structuredWorksets() { return this.structuredVelocity?.source.familyWorksets.regularInterior.buffer; }
  /** QA-only MGPCG status; simulation authority consumes this buffer directly on GPU. */
  get mgpcgControl() { return this.pressureSolverControl; }
  /** QA-only identity of the sparse fine source sampled by the last face build. */
  get powerBoundaryFineSource() { return this.lastPowerBoundaryFineSource; }
  /** QA-only exact sparse source sampled by the last power-boundary build.
   * The surface phase may already have toggled the public current slot when a
   * later publication fails, so generation/slot metadata alone is not enough
   * to reproduce a boundary-authority disagreement. */
  get powerBoundaryFineLevelSetSource(): WebGPUFineLevelSetBrickSource | undefined {
    const sampled = this.lastPowerBoundaryFineSource;
    if (!sampled) return undefined;
    const source = this.globalFineSourceA?.generationSlot === sampled.generationSlot
      ? this.globalFineSourceA : this.globalFineSourceB;
    return source?.generation === sampled.generation ? source : undefined;
  }
  get powerDescriptorControl() { return this.powerDescriptor?.control; }
  get powerTopologyControl() { return this.powerTopology?.control; }
  get powerDescriptorRows() { return this.powerDescriptor?.descriptors; }
  get powerTopologyMetrics() { return this.powerTopology?.metrics; }
  get powerCatalogEntryHeaders() { return this.powerTopology?.catalogEntryHeaders; }
  get powerCatalogFaces() { return this.powerTopology?.catalogFaces; }
  get techniqueDebugSource() {
    const surface = this.fineSeedAdapter?.source;
    const topology = this.powerTopology?.source;
    const structured = this.structuredVelocity?.source;
    const tetrahedronHeaders = topology?.catalogTetrahedronHeaders;
    const tetrahedra = topology?.catalogTetrahedra;
    const tetrahedronVertices = topology?.catalogTetrahedronVertices;
    if (!surface || !topology || !structured || !tetrahedronHeaders || !tetrahedra || !tetrahedronVertices) return undefined;
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
      topologyMetrics: { buffer: topology.metrics },
      catalogEntryHeaders: { buffer: topology.catalogEntryHeaders },
      catalogFaces: { buffer: topology.catalogFaces },
      tetrahedronHeaders: { buffer: tetrahedronHeaders },
      tetrahedra: { buffer: tetrahedra },
      tetrahedronVertices: { buffer: tetrahedronVertices },
      structuredAuthority: { buffer: structured.authority },
      structuredParams: { buffer: structured.params },
      structuredRowGeometry: { buffer: structured.rowGeometry },
      structuredRowVelocities: { buffer: structured.rowVelocities },
      structuredControl: { buffer: structured.control },
      pressure: { buffer: this.latestPressureInA ? this.pressureA : this.pressureB },
      leafHeaders: { buffer: this.leafHeaders },
      // Optional: a scene can reach a publication before the power-coarse
      // level set has one, and the cell trace reads zero flags as "never
      // corrected" rather than as a phi of zero.
      ...(this.powerCoarseLevelSetSchedule
        ? { coarsePhi: { buffer: this.powerCoarseLevelSetSchedule.sampleSource.values } }
        : {}),
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
  /** Accepted generation is intentionally not guessed from the host attempt
   * counter. GPU audit paths obtain it from the accepted structured control. */
  get powerPublicationGeneration(): number | undefined { return undefined; }
  get powerLeafHeaders() { return this.leafHeaders; }
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
    const losassoControl = this.losassoBackend?.sources.operator.control;
    if (losassoControl) {
      // Preserve the long-standing failure receipt shape for callers while
      // exposing the single reduced authority that replaces the five Power
      // publication controls. Unused regions remain zero-initialized.
      for (const region of ["descriptorCandidate", "topologyCandidate",
        "structuredCandidate", "boundaryCandidate", "spgridCandidate",
        "epoch"] as const) capture(region, losassoControl);
    } else {
      capture("descriptorCandidate", this.powerDescriptor!.control);
      capture("topologyCandidate", this.powerTopology!.control);
      capture("structuredCandidate", this.structuredVelocity!.candidateControl);
      capture("boundaryCandidate", this.structuredBoundary!.candidateControl);
      capture("spgridCandidate", this.firstOrderVCycle.candidateControl);
      capture("epoch", this.topologyEpoch!.state);
    }
    capture("rowDelta", this.leafFrontier,
      this.frontierAllocation.rowDeltaControlOffsetWords * 4);
    capture("ownerCandidate", this.ownerPages.candidateTransaction);
    capture("carryFlags", this.compaction, this.compactionAllocationRowDeltaScratchOffsetBytes);
    if (this.globalFineSummaries) {
      capture("fineSummaryDirectory", this.globalFineSummaries.directory);
      capture("fineSummaryWorkState", this.globalFineSummaries.diagnosticBuffers.workState);
    }
    if (this.powerCoarseLevelSetSchedule) {
      const coarse = this.powerCoarseLevelSetSchedule.sampleSource;
      capture("coarseControl", coarse.control);
      capture("coarseDirectory", coarse.directory);
      capture("coarseDelta", coarse.delta);
    }
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
    if (this.firstOrderVCycle) {
      const spgridFailure = await this.firstOrderVCycle.readCandidateFailureDiagnostics();
      result.spgridLevelDelta = spgridFailure.levelDelta;
      result.spgridCandidateDispatch = spgridFailure.candidateDispatch;
    }
    const descriptorFirstError = Number(result.descriptorCandidate[3]) >>> 0;
    if (this.powerDescriptor && Number(result.descriptorCandidate[2]) !== 0
      && descriptorFirstError < this.pressureCapacity.rowCapacity) {
      result.descriptorFailureRow =
        await this.readPowerDescriptorCandidateFailure(descriptorFirstError);
    }
    const boundary = result.boundaryCandidate;
    const boundaryFirstError = Number(boundary[1]) >>> 0;
    if (this.powerDescriptor && (Number(boundary[0]) & 2) !== 0
      && boundaryFirstError < this.pressureCapacity.rowCapacity) {
      result.boundaryFailureRow = await this.readPowerDescriptorCandidateFailure(boundaryFirstError);
    }
    const coarseFirstError = Number(result.coarseControl[1]) >>> 0;
    if (Number(result.coarseControl[0]) !== 0 && coarseFirstError < this.pressureCapacity.rowCapacity) {
      result.coarseFailureRow = await this.readPowerCoarseFailureRow(coarseFirstError);
    }
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
  /** Failure-only cross-check of one published Power row against its owner page. */
  async readOwnerPageForPowerRow(row: number): Promise<Record<string, unknown> | undefined> {
    const structured = this.structuredVelocity?.source;
    if (!structured || !Number.isSafeInteger(row) || row < 0
      || row >= structured.plan.rowCapacity) return undefined;
    const arena = this.ownerPages.arena;
    const readback = this.device.createBuffer({
      label: "Octree Power-row owner-page failure readback",
      size: arena.size + 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read Octree Power-row owner-page failure",
    });
    encoder.copyBufferToBuffer(structured.rowGeometry, row * 16, readback, 0, 16);
    encoder.copyBufferToBuffer(arena, 0, readback, 16, arena.size);
    this.device.queue.submit([encoder.finish()]);
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
        cell % this.dims.nx,
        Math.floor(cell / this.dims.nx) % this.dims.ny,
        Math.floor(cell / (this.dims.nx * this.dims.ny)),
      ];
      const brickDimensions = [
        Math.ceil(this.dims.nx / 8),
        Math.ceil(this.dims.ny / 8),
        Math.ceil(this.dims.nz / 8),
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
    const readback = this.device.createBuffer({
      label: "Octree Power seed-chain control readback",
      size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read Octree Power seed-chain controls",
    });
    encoder.copyBufferToBuffer(this.structuredVelocity.control, 0, readback, 0, 24);
    encoder.copyBufferToBuffer(this.structuredBoundary.control, 0, readback, 24, 40);
    this.device.queue.submit([encoder.finish()]);
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
  get nativePowerVelocityAuthority() {
    return Boolean(this.structuredVelocity && this.structuredBoundary
      && this.structuredDynamics);
  }
  // Copy sources for the step-snapshot ring (POWER_LIQUIDS_ULTIMATE_M1MAX A4).
  // The ring appends these copies after every producer in the step's own
  // encoder, so a mapped record shows the step's own verdicts. Absent is not
  // zero: a missing source decodes as absent and refuses to authorize a skip.
  get topologyEpochState(): GPUBuffer | undefined { return this.topologyEpoch?.state; }
  get airSupportScratch(): GPUBuffer | undefined { return this.airVelocitySupport?.scratch; }
  get spgridLevelDelta(): GPUBuffer | undefined { return this.firstOrderVCycle?.levelDelta; }
  /** End-of-step copy sources for the Losasso-native diagnostic receipt. */
  get losassoAuthorityControl(): GPUBuffer | undefined {
    return this.losassoBackend?.sources.operator.control;
  }
  get losassoCoarsePhiControl(): GPUBuffer | undefined {
    return this.losassoCoarsePhi?.source.arena;
  }
  get losassoExtensionControl(): GPUBuffer | undefined {
    return this.losassoBackend?.extensionBand.source.control;
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
  get losassoVelocityDebug() {
    const source = this.losassoBackend?.sources.velocitySampler;
    const extension = this.losassoBackend?.extensionBand.source;
    const wet = this.losassoBackend?.sources;
    return source ? {
      control: source.control,
      faceGeometry: source.faceGeometry,
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
    } : undefined;
  }
  /** Frontier/dirty-tile forensics: the leaf-frontier header plus the shared
   * compaction scratch header. Read-only diagnostic surface. */
  get losassoFrontierDebug() {
    return this.coarseDynamics.backend === "losasso"
      ? { frontier: this.leafFrontier, compaction: this.compaction,
        dirtyFailureOffsetBytes: this.dirtyFailureOffsetBytes }
      : undefined;
  }
  get losassoPressureDebug() {
    const source = this.losassoBackend?.sources;
    const wide = source?.wideSolver;
    return source && wide ? {
      control: source.operator.control,
      rightHandSide: source.rightHandSide,
      diagonal: wide.diagonal,
    } : undefined;
  }
  get losassoCoarsePhiDebug() {
    const source = this.losassoCoarsePhi?.source;
    const control = this.losassoBackend?.sources.operator.control;
    return source && control ? {
      control, rowPhi: source.rowPhi, leafHeaders: this.leafHeaders,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz] as const,
    } : undefined;
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
    const fineTransportGovernor = this.lastGlobalFineTransport
      && "governor" in this.lastGlobalFineTransport ? {
        buffer: this.lastGlobalFineTransport.governor,
        size: 4 * (4 + 64),
      } : undefined;
    const pressureRhs = this.structuredDivergenceRhs
      ? { buffer: this.structuredDivergenceRhs } : undefined;
    const section63Coefficients = this.structuredVelocity ? {
      buffer: this.structuredVelocity.source.section63.coefficients,
      size: this.structuredVelocity.source.section63.coefficientBankStrideWords * 4,
    } : undefined;
    const symmetry = this.pipelinedMGPCG?.symmetryStageAuditBuffers
      ?? this.losassoBackend?.solverSymmetryStageAuditBuffers;
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
      maximumOuterIterations: this.pipelinedMGPCG?.iterationBudget
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
    const powerCoarse = this.powerCoarseLevelSetSchedule?.sampleSource;
    const losassoCoarse = this.losassoCoarsePhi?.source;
    const coarse = powerCoarse ?? (losassoCoarse
      // Generic rendering and QA consumers use the shared eight-word coarse
      // directory ABI. The Losasso arena is its private topology-sampling
      // hash table and must never leak through this backend-neutral source.
      ? { directory: losassoCoarse.volumeDirectory, rowCapacity: losassoCoarse.rowCapacity }
      : undefined);
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
    const coarse = this.powerCoarseLevelSetSchedule?.sampleSource;
    if (!this.coarseOnlySurfaceTracking || !this.powerCoarseLevelSetBootstrapped || !coarse) {
      return undefined;
    }
    return {
      kind: "coarse-levelset-sampling" as const,
      directory: { buffer: coarse.directory },
      control: { buffer: coarse.control },
      rowCapacity: coarse.rowCapacity,
      sampleDimensions: [this.dims.nx, this.dims.ny, this.dims.nz] as const,
      physicalCellSize: this.scene.container.width_m / this.dims.nx,
      domainOrigin: [0, 0, 0] as const,
      generation: this.powerCoarseLevelSetGeneration & 0x3fff_ffff,
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
  get globalFineCoarseLevelSetControl(): GPUBuffer | undefined { return this.powerCoarseLevelSetSchedule?.control; }
  /** Diagnostic-only fine-to-coarse restriction transaction consumed by the
   * compact coarse-phi publication gate. */
  get globalFineRestrictionControl(): GPUBuffer | undefined { return this.fineToPowerCoarseLevelSet?.control; }
  /** QA-only bounded readback for the compact row that caused a fail-closed
   * coarse-phi transaction. This is used only while constructing a rejected
   * t=0 solver, so it cannot enter the recurring simulation schedule. */
  async readPowerCoarseFailureRow(row: number) {
    if (!Number.isSafeInteger(row) || row < 0 || row >= this.pressureCapacity.rowCapacity
      || !this.powerCoarseLevelSet || !this.powerCoarseLevelSetSchedule
      || !this.powerTopology || !this.structuredVelocity) return undefined;
    const readback = this.device.createBuffer({ label: "Power coarse-phi failure-row QA", size: 160,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read power coarse-phi failure row" });
    encoder.copyBufferToBuffer(this.leafHeaders, row * 48, readback, 0, 48);
    encoder.copyBufferToBuffer(this.powerTopology.metrics, row * 16, readback, 48, 16);
    encoder.copyBufferToBuffer(this.powerCoarseLevelSet.records, row * 16, readback, 64, 16);
    encoder.copyBufferToBuffer(this.powerCoarseLevelSetSchedule.diagnosticRowStatus,
      row * 32, readback, 80, 32);
    encoder.copyBufferToBuffer(this.powerCoarseLevelSetSchedule.diagnosticCandidateSampleDirectory,
      32 + row * 32, readback, 112, 32);
    encoder.copyBufferToBuffer(this.structuredVelocity.source.rowGeometry,
      row * 16, readback, 144, 16);
    this.device.queue.submit([encoder.finish()]);
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
      || row >= this.pressureCapacity.rowCapacity) return undefined;
    const headerReadback = this.device.createBuffer({
      label: "Rejected power descriptor candidate leaf header",
      size: 48 + this.ownerPages.arena.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read rejected power descriptor candidate leaf header",
    });
    encoder.copyBufferToBuffer(this.candidateLeafHeaders, row * 48, headerReadback, 0, 48);
    encoder.copyBufferToBuffer(
      this.ownerPages.arena, 0, headerReadback, 48, this.ownerPages.arena.size,
    );
    this.device.queue.submit([encoder.finish()]);
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
        cell % this.dims.nx,
        Math.floor(cell / this.dims.nx) % this.dims.ny,
        Math.floor(cell / (this.dims.nx * this.dims.ny)),
      ];
      const brickDimensions = [
        Math.ceil(this.dims.nx / 8),
        Math.ceil(this.dims.ny / 8),
        Math.ceil(this.dims.nz / 8),
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
  /** Diagnostic-only raw sparse summary header; topology consumes this GPU-side. */
  get globalFineSummaryDirectory(): GPUBuffer | undefined { return this.globalFineSummaries?.directory; }
  /** Rejection-only producer state for an unpublished fine summary. */
  get globalFineSummaryDebug() {
    const summaries = this.globalFineSummaries;
    const powerCoarse = this.powerCoarseLevelSetSchedule;
    const losassoCoarse = this.losassoCoarsePhi;
    if (!summaries || (!powerCoarse && !losassoCoarse)) return undefined;
    return {
      ...summaries.diagnosticBuffers,
      coarseControl: powerCoarse?.control ?? losassoCoarse!.source.volumePublication,
      coarseDelta: powerCoarse?.sampleSource.delta ?? losassoCoarse!.source.summaryDelta,
    };
  }
  get fineSeedAuthority() { return Boolean(this.fineSeedAdapter && this.globalFineLevelSet); }
  /** Paired same-generation energy reduction produced around pressure projection. */
  get structuredProjectionEnergyStats(): GPUBuffer | undefined {
    return this.structuredDynamics?.projectionEnergyStats;
  }

  async readLosassoAuthorityDiagnostics(): Promise<Readonly<{
    authority: readonly number[];
    candidate: readonly number[];
    candidateHeader: readonly number[];
    solver: readonly number[];
    coarsePhi: readonly number[];
  }> | undefined> {
    const backend = this.losassoBackend, coarse = this.losassoCoarsePhi;
    if (!backend || !coarse) return undefined;
    const readback = this.device.createBuffer({
      label: "Read Losasso reduced authority",
      size: 32 + 32 + 48 + 32 + 80,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read Losasso authority controls" });
    encoder.copyBufferToBuffer(backend.sources.operator.control, 0, readback, 0, 32);
    encoder.copyBufferToBuffer(backend.candidateAuthorityControl, 0, readback, 32, 32);
    encoder.copyBufferToBuffer(this.candidateLeafHeaders, 0, readback, 64, 48);
    const solver = backend.solverControl ?? backend.sources.rowCount;
    encoder.copyBufferToBuffer(solver, 0, readback, 112, Math.min(32, solver.size));
    encoder.copyBufferToBuffer(coarse.source.arena, 0, readback, 144, 80);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return Object.freeze({
        authority: Array.from(words.slice(0, 8)),
        candidate: Array.from(words.slice(8, 16)),
        candidateHeader: Array.from(words.slice(16, 28)),
        solver: Array.from(words.slice(28, 36)),
        coarsePhi: Array.from(words.slice(36, 56)),
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
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
    const ownerArena = this.ownerPages.arena;
    const readback = this.device.createBuffer({
      label: "Section 5 failed support topology readback",
      size: 32 + ownerArena.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read Section 5 failed support topology",
    });
    encoder.copyBufferToBuffer(support.recordArena, recordByte, readback, 0, 32);
    encoder.copyBufferToBuffer(ownerArena, 0, readback, 32, ownerArena.size);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const latched = latch?.length === 9 && latch[8] === firstError ? latch : undefined;
      const record = latched ? [latched[0], latched[1], latched[2], latched[3],
        latched[4], latched[7], latched[5], latched[6]] : Array.from(words.slice(0, 8));
      const origin = record.slice(0, 3) as [number, number, number];
      const size = record[3] ?? 0;
      const validIdentity = size >= 1 && size <= this.maxLeafSize
        && (size & (size - 1)) === 0
        && origin.every((value, axis) => value + size <= [this.dims.nx, this.dims.ny, this.dims.nz][axis]);
      if (!validIdentity) return { stage, item, record, invalidIdentity: true };
      const ownerWords = words.subarray(8);
      let finer = false;
      let coarser = false;
      let invalidOwner = false;
      let ratioViolation = false;
      const dimensions = [this.dims.nx, this.dims.ny, this.dims.nz] as const;
      const neighbors = OCTREE_POWER_NEIGHBOR_DIRECTIONS.map((direction, directionIndex) => {
        const coordinate = (axis: 0 | 1 | 2) => direction[axis] < 0
          ? origin[axis] - 1
          : direction[axis] > 0 ? origin[axis] + size : origin[axis] + Math.floor(size / 2);
        const probe: [number, number, number] = [coordinate(0), coordinate(1), coordinate(2)];
        const boundary = probe.some((value, axis) => value < 0 || value >= dimensions[axis]);
        if (boundary) return { directionIndex, direction, probe, boundary: true, size };
        const owner = lookupOctreeOwnerPage(ownerWords, this.ownerPages.plan, probe,
          this.maxLeafSize as OctreeOwnerLeafSize);
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
    if (this.fineSeedAdapter) {
      encoder.copyBufferToBuffer(this.fineSeedAdapter.source.candidateCount, 0, readback, 248, 36);
      encoder.copyBufferToBuffer(this.fineSeedAdapter.source.leaves, 0, readback, 284, 64);
    }
    if (this.structuredVelocity) {
      encoder.copyBufferToBuffer(this.structuredVelocity.rowVelocitiesA, 0, readback, 348, 16);
      encoder.copyBufferToBuffer(this.structuredVelocity.rowVelocitiesA,
        this.structuredVelocity.plan.rowCapacity * 16, readback, 364, 16);
    }
    if (this.powerCoarseLevelSet) {
      encoder.copyBufferToBuffer(this.powerCoarseLevelSet.records, 0, readback, 380, 16);
    }
    encoder.copyBufferToBuffer(this.compaction, 0, readback, 396, 16);
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
    }
    if (this.globalFineVolumeA) {
      encoder.copyBufferToBuffer(this.globalFineVolumeA.control, 0, readback, 512, 64);
    }
    if (this.airVelocitySupport) {
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
    const airSupportFailureTopology = await this.readAirSupportFailureTopology(topologyFirstError,
      diagnostics.airSupportTopologyFailureLatch);
    return airSupportFailureTopology ? { ...diagnostics, airSupportFailureTopology } : diagnostics;
  }
  /** Post-submit diagnostic census; never participates in pressure scheduling. */
  readSPGridHierarchyCensus() {
    return this.firstOrderVCycle?.readHierarchyCensus();
  }
  /** Terminal-only proof that the compact directory differential executed. */
  readSPGridTouchedDirectoryTripwire() {
    return this.firstOrderVCycle?.readTouchedDirectoryTripwireDiagnostics();
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
  readFluidBrickResidencyStats() { return this.topologyResidency.readStats(); }
  readFluidBulkBrickResidencyStats() { return this.sparseBrickWorld?.readBulkResidencyStats(); }
  encodeSparseBrickWorld(encoder: GPUCommandEncoder, _dt_s = 0) {
    if ((!this.globalFineBootstrapped && !this.powerCoarseLevelSetBootstrapped)
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
    // These three are built by an initialization task rather than the
    // constructor, so a failure during initialization reaches this cleanup
    // with them still unassigned. Destroying them unconditionally throws a
    // TypeError that replaces the failure the caller is about to rethrow.
    this.pipelinedMGPCG?.destroy();
    this.losassoReadyCommit?.destroy();
    this.losassoCoarsePhi?.destroy();
    this.losassoRowMotion?.destroy();
    this.losassoConditionedOperator?.destroy();
    this.losassoBackend?.destroy();
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
    this.globalFineTransportA?.destroy(); this.globalFineTransportB?.destroy();
    this.losassoFineTransportA?.destroy(); this.losassoFineTransportB?.destroy();
    this.globalFineTopologyAB?.destroy(); this.globalFineTopologyBA?.destroy();
    this.globalFineSeeds?.destroy(); this.globalFineLevelSet?.destroy();
    this.globalFineSummaries?.destroy();
    this.coarseOnlySummary?.destroy();
    this.fineSeedAdapter?.destroy();
    this.fineToPowerCoarseLevelSet?.destroy();
    this.airVelocitySupport?.destroy(); this.powerCoarseLevelSetSchedule?.destroy();
    this.powerCoarseLevelSet?.destroy(); this.powerSolidVertices?.destroy();
    this.powerDescriptor?.destroy(); this.powerTopology?.destroy();
    this.powerVolumes?.destroy(); this.powerVolumeParams?.destroy();
    this.topologyDiagnosticTexture?.destroy(); this.pressureSamplesDiagnosticTexture?.destroy(); this.pressureDiagnosticTexture?.destroy();
    this.surfaceState.destroy();
    if (this.sparseBrickWorld) this.sparseBrickWorld.destroy(); else this.topologyResidency.destroy();
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
    && !scene.fluid.initialBrickSeedsAdditive) {
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
    const brickWet = initialFluidBrickContainsCell(scene, x, y, z, [nx, ny, nz]);
    const wet = aboveGround && combineInitialBrickWet(scene, brickWet, scene.fluid.initialCondition === "dam-break"
      ? damBreakBoxContains(dam, (x + 0.5) / nx, (y + 0.5) / ny, (z + 0.5) / nz)
      : (y + 0.5) / ny <= scene.container.fillFraction);
    alpha[x + nx * (y + ny * z)] = wet ? 1 : 0;
  }
  return signedDistanceFromVolume(alpha, nx, ny, nz, cell);
}

const octreeProjectionShaderBase = /* wgsl */ `
override targetRefinementSize: u32 = 0u;
override rowIndexedPressure: bool = true;
override sparseTopologyTileStates: u32 = 0u;
override denseSolidField: bool = true;
override fluidGatedBoundaryRefinement: bool = true;
override topologyCandidateView: u32 = 0u;
override fineSummaryFactor: u32 = 4u;
override gradingPageFill: bool = false;
override gradingSplitHelpers: bool = false;
override gradingMembershipLoad: bool = false;
struct Owner { packedOrigin: u32, size: u32 }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowDirectionLength: vec4f, physical: vec4f, pressureCapacity: vec4u, hydrostatic: vec4f }
struct LeafHeader { cell: u32, entryStart: u32, entryCount: u32, size: u32, diagonal: f32, rhs: f32, pad0: u32, pad1: u32, gradient: vec4f }
struct RigidBody { positionShape: vec4f, dimensions: vec4f, orientation: vec4f, linearVelocity: vec4f, angularVelocity: vec4f, inverseMassInertia: vec4f, angularMomentumRestitution: vec4f, material: vec4f }
struct SolidCell { fraction: f32, owner: i32 }
// [0] = row count, [1] = reserved, [2..4] = row-parallel indirect args,
// [5..7] = reserved, [8] = reserved,
// [9..11] = one-workgroup-per-leaf args, [12..14] = frontier row-plan args;
// per-block row totals and reserved words (later exclusive offsets) start
// at word 15. The dispatch words are copied out after their producing pass because one
// buffer cannot be writable storage and indirect in the same dispatch scope.
@group(0) @binding(2) var<storage, read_write> compaction: array<u32>;
@group(0) @binding(3) var<storage, read_write> owners: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> pressureIn: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<uniform> frontierSortParams: vec4u;
@group(0) @binding(8) var<storage, read_write> leafHeaders: array<LeafHeader>;
@group(0) @binding(9) var<storage, read_write> frontierSortScratch: array<u32>;
@group(0) @binding(10) var<storage, read_write> rigidBodies: array<RigidBody, 12>;
@group(0) @binding(11) var<storage, read_write> solidCells: array<u32>;
@group(0) @binding(12) var terrainIn: texture_2d<f32>;
// The host-rasterized t=0 level set. Authoritative only while the -30
// bootstrap sentinel is live; analytic scenes bind a 1-cubed placeholder here
// and never sample it. A sampled texture does not consume the storage-buffer
// budget this layout is already at.
@group(0) @binding(14) var bootstrapLevelSetIn: texture_3d<f32>;
// [0..1] immutable A/B counts, [2] active selector, [3] active generation,
// [4..5] candidate count/carry scratch, [6] candidate ready,
// [7] candidate selector, [8] candidate generation, [9] rejection reason,
// followed by sorted A/B publications,
// the bounded dirty candidate stream, and the exact row-delta ABI.
@group(0) @binding(13) var<storage, read_write> frontier: array<u32>;
// Dual ABI. Sparse-extrapolation groups bind the bulk-residency worklist;
// global-fine topology/pressure groups bind the corrected compact coarse-phi
// directory (8-word header followed by 8-word hash entries).
@group(0) @binding(15) var<storage, read> bulkWorklist: array<u32>;

fn dims() -> vec3u {
  return params.dimsMax.xyz;
}
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
struct CorrectedCoarsePhi { authority:bool, phi:f32, minimumPhi:f32, maximumPhi:f32, leafSize:u32 }
fn coarseWord(index:u32)->u32{return bulkWorklist[index];}
fn coarseFinite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn coarseDirectoryAuthority()->bool{
  let expected=params.pressureCapacity.w>>2u;
  if(expected==0u||arrayLength(&bulkWorklist)<16u||coarseWord(0u)!=0x80000000u
      ||(coarseWord(1u)&0x3fffffffu)!=expected){return false;}
  let directoryDims=vec3u(coarseWord(4u),coarseWord(5u),coarseWord(6u));
  let physicalCellSize=bitcast<f32>(coarseWord(7u));let rowCount=coarseWord(2u);
  let actualCapacity=(arrayLength(&bulkWorklist)-8u)/8u;
  return all(directoryDims==dims())&&coarseFinite(physicalCellSize)&&physicalCellSize>0.0
    &&abs(physicalCellSize-params.cellRelax.x)<=1e-5*max(physicalCellSize,params.cellRelax.x)
    &&rowCount<=actualCapacity&&rowCount>0u
    &&coarseWord(3u)>0u&&(coarseWord(3u)&(coarseWord(3u)-1u))==0u;
}
// The reduced Losasso backend binds its compact coarse-phi arena at this
// slot instead of the Power corrected directory. The arena magic is stamped
// only after a fault-free fine-to-coarse exchange and cleared on any fault,
// so arena liveness carries the same fail-closed meaning as the Power
// generation gate above.
fn losassoCoarseArenaAuthority()->bool{
  if(arrayLength(&bulkWorklist)<20u
      ||coarseWord(0u)!=${OCTREE_LOSASSO_COARSE_PHI_MAGIC}u){return false;}
  let directoryDims=vec3u(coarseWord(5u),coarseWord(6u),coarseWord(7u));
  let physicalCellSize=bitcast<f32>(coarseWord(8u));
  let rowCount=coarseWord(2u);let maximumLeaf=coarseWord(4u);
  return all(directoryDims==dims())&&coarseFinite(physicalCellSize)&&physicalCellSize>0.0
    &&abs(physicalCellSize-params.cellRelax.x)<=1e-5*max(physicalCellSize,params.cellRelax.x)
    &&rowCount>0u&&maximumLeaf>0u&&(maximumLeaf&(maximumLeaf-1u))==0u;
}
// Hash lookup into the Losasso arena's row directory (header words 10/11,
// 4-word entries: cell+1, size, row+1, hash) — the same scheme
// sampleCoarseOctreePhi uses from the fine-topology binding.
fn losassoArenaLookup(cell:u32,size:u32)->u32{
  let capacity=coarseWord(11u);if(capacity==0u||(capacity&(capacity-1u))!=0u){return 0xffffffffu;}
  let hash=((cell*0x9e3779b1u)^size)*0x85ebca6bu;let base=coarseWord(10u);let mask=capacity-1u;
  for(var probe=0u;probe<32u;probe+=1u){let at=base+4u*((hash+probe)&mask);let key=coarseWord(at);
   if(key==0u){return 0xffffffffu;}
   if(key==cell+1u&&coarseWord(at+1u)==size&&coarseWord(at+3u)==hash){return coarseWord(at+2u)-1u;}}
  return 0xffffffffu;
}
fn coarseMortonPart(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn coarseMorton(cell:u32)->u32{let d=dims();let q=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));return coarseMortonPart(q.x)|(coarseMortonPart(q.y)<<1u)|(coarseMortonPart(q.z)<<2u);}
fn coarseLookup(cell:u32,size:u32)->u32{let count=min(coarseWord(2u),(arrayLength(&bulkWorklist)-8u)/8u);let wantedLevel=31u-countLeadingZeros(size);let wantedMorton=coarseMorton(cell);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let base=8u+middle*8u;let entryLevel=31u-countLeadingZeros(coarseWord(base+1u));let entryMorton=coarseMorton(coarseWord(base)-1u);if(entryLevel<wantedLevel||(entryLevel==wantedLevel&&entryMorton<wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let base=8u+low*8u;if(coarseWord(base)==cell+1u&&coarseWord(base+1u)==size){return base;}}return 0xffffffffu;}
fn correctedCoarsePhi(point:vec3f)->CorrectedCoarsePhi{
  if(any(point<vec3f(0.0))||any(point>=vec3f(dims()))){return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u);}
  // Losasso branch: classify from the live arena's restricted row phi. This is
  // the coarse backstop that keeps wet rows carried through a one-generation
  // fine-summary gap; without it every unsummarized cell reads dry, a single
  // hiccup validly retires the whole frontier, and a zero-row topology is
  // terminal (dirty marking only visits active tiles).
  if(losassoCoarseArenaAuthority()){
    let arenaQ=vec3u(floor(point));var size=1u;let maximumLeaf=coarseWord(4u);
    loop{let origin=(arenaQ/vec3u(size))*vec3u(size);
     let cell=origin.x+dims().x*(origin.y+dims().y*origin.z);
     let row=losassoArenaLookup(cell,size);
     if(row!=0xffffffffu&&row<coarseWord(2u)){let entry=coarseWord(9u)+8u*row;
      let flags=coarseWord(entry+5u);let value=bitcast<f32>(coarseWord(entry+2u));
      if((flags&3u)==3u&&coarseFinite(value)){return CorrectedCoarsePhi(true,value,value,value,size);}
      return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u);}
     if(size>=maximumLeaf){break;}size*=2u;}
    // A valid sparse arena defines every directory miss as coarse air, matching
    // sampleCoarseOctreePhi's half-width convention.
    let air=0.5*bitcast<f32>(coarseWord(8u));
    return CorrectedCoarsePhi(true,air,air,air,0u);
  }
  if(!coarseDirectoryAuthority()){return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u);}
  let q=vec3u(floor(point));let denseCell=q.x+dims().x*(q.y+dims().y*q.z);
  let volume=dims().x*dims().y*dims().z;let actualCapacity=(arrayLength(&bulkWorklist)-8u)/8u;
  if((coarseWord(1u)&0x40000000u)!=0u&&actualCapacity>=volume){let denseBase=8u+(actualCapacity-volume+denseCell)*8u;
    let value=bitcast<f32>(coarseWord(denseBase+2u));let flags=coarseWord(denseBase+5u);
    if(coarseWord(denseBase)==denseCell+1u&&coarseWord(denseBase+1u)==1u&&(flags&9u)==9u&&coarseFinite(value)){
      return CorrectedCoarsePhi(true,value,value,value,1u);}}
  var size=1u;let maximumLeaf=coarseWord(3u);
  loop{let origin=(q/vec3u(size))*vec3u(size);let cell=origin.x+dims().x*(origin.y+dims().y*origin.z);let base=coarseLookup(cell,size);
    if(base!=0xffffffffu){let value=bitcast<f32>(coarseWord(base+2u));let minimum=bitcast<f32>(coarseWord(base+3u));let maximum=bitcast<f32>(coarseWord(base+4u));let flags=coarseWord(base+5u);
      if((flags&9u)!=9u||!coarseFinite(value)||!coarseFinite(minimum)||!coarseFinite(maximum)||minimum>maximum||value<minimum||value>maximum){return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u);}
      return CorrectedCoarsePhi(true,value,minimum,maximum,size);}
    if(size>=maximumLeaf){break;}size*=2u;
  }
  // Fine-backed modes have no dense complement. Their valid sparse directory
  // still defines every miss as positive air.
  let air=0.5*bitcast<f32>(coarseWord(7u));
  return CorrectedCoarsePhi(true,air,air,air,0u);
}
fn coarseClassificationPhi(sample:CorrectedCoarsePhi)->f32{
  return select(sample.phi,min(sample.phi,sample.minimumPhi),sample.minimumPhi<0.0&&sample.maximumPhi>=0.0);
}
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn packOrigin(p: vec3u) -> u32 { return index(p); }
fn unpackOrigin(word: u32) -> vec3u {
  let plane = params.dimsMax.x * params.dimsMax.y;
  return vec3u(word % params.dimsMax.x, (word / params.dimsMax.x) % params.dimsMax.y, word / plane);
}
const OWNER_WORD_VALID: u32 = 0x80000000u;
const OWNER_WORD_TOPOLOGY: u32 = 0x00200000u;
fn encodePagedOwner(cell: vec3u, origin: vec3u, size: u32) -> u32 {
  let brickOrigin = (cell / vec3u(8u)) * vec3u(8u);
  let delta = vec3i(origin) - vec3i(brickOrigin);
  return OWNER_WORD_VALID
    | (u32(delta.x + 32) & 63u)
    | ((u32(delta.y + 32) & 63u) << 6u)
    | ((u32(delta.z + 32) & 63u) << 12u)
    | ((u32(firstTrailingBit(size)) & 7u) << 18u);
}
fn invalidOwner() -> Owner { return Owner(0u, 0u); }
fn ownerValid(owner: Owner) -> bool {
  if (owner.size == 0u || owner.size > params.dimsMax.w
      || (owner.size & (owner.size - 1u)) != 0u) { return false; }
  let origin = unpackOrigin(owner.packedOrigin);
  return all(origin + vec3u(owner.size) <= dims());
}
fn rejectOwnerAuthority() -> Owner {
  atomicStore(&owners[2], 1u);
  return invalidOwner();
}
fn decodePagedOwner(word: u32, cell: vec3u) -> Owner {
  if ((word & OWNER_WORD_VALID) == 0u) { return rejectOwnerAuthority(); }
  let exponent = (word >> 18u) & 7u;
  if (exponent > 5u) { return rejectOwnerAuthority(); }
  let brickOrigin = vec3i((cell / vec3u(8u)) * vec3u(8u));
  let delta = vec3i(i32(word & 63u) - 32, i32((word >> 6u) & 63u) - 32,
    i32((word >> 12u) & 63u) - 32);
  let signedOrigin = brickOrigin + delta;
  if (any(signedOrigin < vec3i(0))) { return rejectOwnerAuthority(); }
  let origin = vec3u(signedOrigin); let size = 1u << exponent;
  if (any(cell < origin) || any(cell >= origin + vec3u(size))
      || any(origin + vec3u(size) > dims())) { return rejectOwnerAuthority(); }
  return Owner(packOrigin(origin), size);
}
// The owner-page arena layout -- capacity, logical count, directory offset,
// payload offset and the active table bit -- is published by the page
// authority before any topology dispatch and is never written by this shader.
// It is therefore dispatch-invariant, yet every owner read and every owner
// store re-derived it through five device atomics on the SAME five words, plus
// three more inside ownerPayloadBase. Those addresses cannot live in L1, so a
// single owner lookup cost nine round trips to a handful of contended lines
// and splitLeaf paid them once per cell of the leaf it materializes.
//
// Resolve it once per invocation instead. The header is read with the same
// atomic loads the first time it is needed, so a caller that runs before the
// authority publishes still observes exactly what it observed before.
struct OwnerPageMap {
  directoryOffset: u32,
  payloadBase: u32,
  capacity: u32,
  logicalCount: u32,
  consistent: u32,
}
var<private> ownerPageMapCache: OwnerPageMap;
var<private> ownerPageMapResolved: bool = false;
fn ownerPageMap() -> OwnerPageMap {
  if (!ownerPageMapResolved) {
    let pageIndexOffset = atomicLoad(&owners[5]);
    let capacity = atomicLoad(&owners[3]);
    let logicalCount = atomicLoad(&owners[4]);
    let activeTable = atomicLoad(&owners[10]) >> 31u;
    let table = activeTable ^ min(topologyCandidateView, 1u);
    let payloadOffset = atomicLoad(&owners[6]);
    let consistent = pageIndexOffset == 16u + capacity
      && payloadOffset == pageIndexOffset + 3u * capacity + 2u * logicalCount;
    ownerPageMapCache = OwnerPageMap(
      pageIndexOffset + 3u * capacity + table * logicalCount,
      payloadOffset + table * capacity * 512u,
      capacity, logicalCount, select(0u, 1u, consistent));
    ownerPageMapResolved = true;
  }
  return ownerPageMapCache;
}
// Memoizing this lookup does NOT pay; see E6's recorded negative result.
//
// The directory IS dispatch-invariant -- ownerPageMap caches the arena header
// on exactly that reasoning, and every atomic store in this module targets the
// rejection latch at owners[2] or a payload word -- and lookups do arrive in
// page-local bursts, so a single-entry var<private> memo is both sound and
// hits about seven times in eight. It measured -0.41 ms in favour of NOT
// memoizing, with the grading probe floor consistently ~5% worse. The load it
// removes was already the hottest line in the arena, while the two extra
// thread-local registers and the compare are paid on every lookup.
fn ownerPageEncoded(logical: u32) -> u32 {
  let map = ownerPageMap();
  if (map.consistent == 0u || logical >= map.logicalCount) { return 0u; }
  let directoryOffset = map.directoryOffset;
  return atomicLoad(&owners[directoryOffset + logical]);
}
fn requireOwnerPageEncoded(logical: u32) -> u32 {
  let encoded = ownerPageEncoded(logical);
  if (encoded == 0u) { atomicStore(&owners[2], 1u); }
  return encoded;
}
fn ownerPageWord(cell: vec3u) -> u32 {
  let brickDims = (dims() + vec3u(7u)) / 8u;
  let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = ownerPageEncoded(logical);
  let map = ownerPageMap();
  if (encoded == 0u || encoded == 0xffffffffu || encoded > map.capacity) { return 0xffffffffu; }
  let local = cell % vec3u(8u);
  return atomicLoad(&owners[map.payloadBase + (encoded - 1u) * 512u + local.x + local.y * 8u + local.z * 64u]);
}
fn ownerAt(p: vec3i) -> Owner {
  if (!valid(p)) { return rejectOwnerAuthority(); }
  let cell = vec3u(p);
  let word = ownerPageWord(cell);
  if (word == 0xffffffffu || word == 0u) { return rejectOwnerAuthority(); }
  return decodePagedOwner(word, cell);
}
fn ownerAtIndex(cell: u32) -> Owner { return ownerAt(vec3i(cellCoord(cell))); }
fn storeOwner(cell: vec3u, origin: vec3u, size: u32) {
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = ownerPageEncoded(logical); let map = ownerPageMap();
  if (encoded == 0u || encoded == 0xffffffffu || encoded > map.capacity) { return; }
  let local = cell % vec3u(8u);
  let at=map.payloadBase+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u;
  let membership=atomicLoad(&owners[at])&OWNER_WORD_TOPOLOGY;
  atomicStore(&owners[at], encodePagedOwner(cell, origin, size) | membership);
}
fn storeOwnerRequired(cell: vec3u, origin: vec3u, size: u32) {
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = requireOwnerPageEncoded(logical); if (encoded == 0u) { return; }
  let local = cell % vec3u(8u);
  // Balance can split a freshly published child while the neighbouring
  // parent split is still writing its coarser children in the same dispatch.
  // The exponent occupies bits 18..20, above every origin delta, so every
  // valid finer dyadic encoding is numerically smaller than every overlapping
  // coarser encoding. Atomic min therefore makes that race deterministic and
  // leaves one non-overlapping owner partition instead of a torn parent.
  let at=ownerPageMap().payloadBase+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u;
  let membership=atomicLoad(&owners[at])&OWNER_WORD_TOPOLOGY;
  atomicMin(&owners[at], encodePagedOwner(cell, origin, size) | membership);
}
fn markAcceptedOwner(origin:vec3u){
  let brickDims=(dims()+vec3u(7u))/8u;let brick=origin/8u;
  let logical=brick.x+brick.y*brickDims.x+brick.z*brickDims.x*brickDims.y;
  let encoded=ownerPageEncoded(logical);let map=ownerPageMap();
  if(encoded==0u||encoded==0xffffffffu||encoded>map.capacity){return;}
  let local=origin%vec3u(8u);let at=map.payloadBase+(encoded-1u)*512u
    +local.x+local.y*8u+local.z*64u;
  atomicOr(&owners[at],OWNER_WORD_TOPOLOGY);
}
fn requireLeafOwnerPages(origin: vec3u, size: u32, lane: u32, lanes: u32) {
  let brickDims = (dims() + vec3u(7u)) / 8u; let first = origin / 8u; let last = (origin + vec3u(size - 1u)) / 8u;
  let shape = last - first + vec3u(1u); let count = shape.x * shape.y * shape.z;
  for (var item = lane; item < count; item += lanes) {
    let local = vec3u(item % shape.x, (item / shape.x) % shape.y, item / (shape.x * shape.y)); let brick = first + local;
    let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
    _ = requireOwnerPageEncoded(logical);
  }
}
// Negative sentinels encode the bootstrap-only initial phi authority:
// tank = -10, dam-break = -20, imported dense level set = -30.
//
// One of these must be live during cold bootstrap. The coarse rows that phi()
// ordinarily reads are produced downstream of the topology this stage builds,
// so with no bootstrap sentinel correctedCoarsePhi has no authority and every
// cell reads air -- the frontier then publishes zero liquid rows.
// Analytic dam/tank scenes answer from closed form; every other scene
// (terrain, rigid bodies, explicitly seeded bricks) answers from the dense
// SDF the host already rasterized and uploaded for topology residency.
fn bootstrapPhiEnabled() -> bool { return params.physical.w < 0.0; }
fn authoredAnalyticPhiAvailable() -> bool { return params.pressureCapacity.y != 0u; }
fn analyticInitialPhiEnabled() -> bool {
  return params.physical.w < 0.0 && params.physical.w > -25.0;
}
fn bootstrapTexturePhiEnabled() -> bool { return params.physical.w <= -25.0; }
fn analyticInitialDamBreak() -> bool {
  return params.pressureCapacity.y == 2u
    ||(params.physical.w < -15.0 && params.physical.w > -25.0);
}
fn bootstrapTexturePhi(p: vec3i) -> f32 {
  return textureLoad(bootstrapLevelSetIn,
    clamp(p, vec3i(0), vec3i(dims()) - vec3i(1)), 0).x;
}
fn analyticInitialPhi(point: vec3f) -> f32 {
  let fill = clamp(params.hydrostatic.w / f32(max(1u, dims().y)), 0.0, 1.0);
  let world = vec3f(-0.5 * params.container.x + point.x * params.cellRelax.x,
    point.y * params.cellRelax.y,
    -0.5 * params.container.z + point.z * params.cellRelax.z);
  if (!analyticInitialDamBreak()) { return world.y - fill * params.container.y; }
  let heightFraction = max(0.92, fill);
  let footprintFraction = sqrt(fill / max(heightFraction, 1e-9));
  let fallback = vec3f(footprintFraction * params.container.x,
    heightFraction * params.container.y, footprintFraction * params.container.z);
  let damDimensions = select(fallback, params.hydrostatic.xyz,
    any(params.hydrostatic.xyz > vec3f(0.0)));
  let exposedMaximum = vec3f(-0.5 * params.container.x + damDimensions.x,
    damDimensions.y, -0.5 * params.container.z + damDimensions.z);
  let q = world - exposedMaximum;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn phi(p: vec3i) -> f32 {
  if (!valid(p)) { return 3.402823e38; }
  if(analyticInitialPhiEnabled()){
    return analyticInitialPhi(vec3f(p)+vec3f(0.5));
  }
  if(bootstrapTexturePhiEnabled()){return bootstrapTexturePhi(p);}
  let coarse=correctedCoarsePhi(vec3f(p)+vec3f(0.5));
  if(coarse.authority){return coarseClassificationPhi(coarse);}
  // A corrected coarse directory is GPU-published after transported fine phi.
  // During the one-generation handoff window, retain the exact authored SDF
  // instead of turning every unresolved sample into air. The fallback is
  // unreachable as soon as the moving coarse authority validates.
  if(authoredAnalyticPhiAvailable()){
    return analyticInitialPhi(vec3f(p)+vec3f(0.5));
  }
  return 3.402823e38;
}
fn samplePhiPoint(point:vec3f)->f32{
  let bounded=clamp(point,vec3f(0.0),vec3f(dims()-vec3u(1u)));let a=vec3u(floor(bounded));let b=min(a+vec3u(1u),dims()-vec3u(1u));let t=fract(bounded);
  let p000=phi(vec3i(a));let p100=phi(vec3i(vec3u(b.x,a.y,a.z)));let p010=phi(vec3i(vec3u(a.x,b.y,a.z)));let p110=phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001=phi(vec3i(vec3u(a.x,a.y,b.z)));let p101=phi(vec3i(vec3u(b.x,a.y,b.z)));let p011=phi(vec3i(vec3u(a.x,b.y,b.z)));let p111=phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y),mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y),t.z);}
// Whether this generation can classify liquid at all.
//
// liquidOwner has no third state: when correctedCoarsePhi reports no
// authority it answers "air". A transaction encoded against a missing
// authority therefore classifies the whole domain dry, carries no row, emits
// no candidate, and publishes an empty topology -- and that state is
// terminal, because dirty marking only visits ACTIVE tiles, so a topology
// that ever reaches zero rows can never dirty a tile again and never
// re-refines. Measured: a solid crossing out of the interface band leaves the
// corrected-coarse publication one generation behind the candidate, this
// predicate went false for one step, and the run published zero pressure rows
// for the remaining 95 steps.
//
// The bootstrap authorities answer from closed form or the uploaded dense SDF
// and never consult the directory, so they are always available.
fn liquidAuthorityAvailable()->bool{
  if(bootstrapPhiEnabled()){return true;}
  // The Losasso lane classifies wetness from the fine summaries backed by its
  // coarse-phi arena; a live arena is that lane's liquid authority. Without
  // this clause the first topology candidate that needs row additions is
  // rejected on availability, the rejection retries forever, and the frozen
  // t=0 wet-row set holds the collapse front statically at the authored dam
  // boundary while the projected field recirculates inside it.
  return coarseDirectoryAuthority()||losassoCoarseArenaAuthority()
    ||authoredAnalyticPhiAvailable();
}
fn liquidOwner(owner: Owner) -> bool {
  if (!ownerValid(owner)) { return false; }
  let centre=vec3f(unpackOrigin(owner.packedOrigin))+vec3f(0.5*f32(owner.size));
  if(analyticInitialPhiEnabled()){return analyticInitialPhi(centre)<0.0;}
  // Sample the leaf centre exactly as the analytic branch above does, so both
  // bootstrap authorities classify a leaf by the same rule.
  if(bootstrapTexturePhiEnabled()){return bootstrapTexturePhi(vec3i(floor(centre)))<0.0;}
  let coarse=correctedCoarsePhi(centre);
  if(coarse.authority&&coarse.leafSize==owner.size){return coarse.phi<0.0;}
  if(coarse.authority&&coarse.leafSize==0u){return false;}
  if(coarse.authority&&coarse.maximumPhi<0.0){return true;}
  if(coarse.authority&&coarse.minimumPhi>=0.0){return false;}
  if(!coarse.authority&&authoredAnalyticPhiAvailable()){
    return analyticInitialPhi(centre)<0.0;
  }
  return false;
}
fn isOrigin(id: vec3u, owner: Owner) -> bool {
  return ownerValid(owner) && all(id == unpackOrigin(owner.packedOrigin));
}
fn cellCount() -> u32 { return params.dimsMax.x * params.dimsMax.y * params.dimsMax.z; }
fn frontierListCapacity() -> u32 { return params.pressureCapacity.x; }
fn frontierBase(which: u32) -> u32 { return 10u + which * frontierListCapacity(); }
fn frontierCandidateBase() -> u32 { return 10u + 2u * frontierListCapacity(); }
fn rowDeltaControlBase()->u32{return frontierCandidateBase()+frontierListCapacity();}
fn rowDeltaNewToOldBase()->u32{return rowDeltaControlBase()+16u;}
fn rowDeltaOldToNewBase()->u32{return rowDeltaNewToOldBase()+frontierListCapacity();}
fn rowDeltaDirtyRowsBase()->u32{return rowDeltaOldToNewBase()+frontierListCapacity();}
fn rowDeltaAffectedRowsBase()->u32{return rowDeltaDirtyRowsBase()+frontierListCapacity();}
fn frontierCurrent() -> u32 { return frontier[2]; }
fn frontierGeneration() -> u32 { return frontier[3]; }
fn frontierCount(which: u32) -> u32 { return min(frontier[which], frontierListCapacity()); }
fn frontierCell(which: u32, slot: u32) -> u32 { return frontier[frontierBase(which) + slot]; }
fn frontierRowIdentityIn(cell:u32,size:u32,current:u32)->u32{
  if(size==0u){return 0xffffffffu;}
  let count=frontierCount(current);
  let level=u32(firstTrailingBit(size));let morton=rowMorton(cell);
  var lo=0u;var hi=count;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(current,mid);
    let otherOwner=ownerAtIndex(other);if(!ownerValid(otherOwner)){return 0xffffffffu;}
    let otherSize=otherOwner.size;let otherLevel=u32(firstTrailingBit(otherSize));
    let otherMorton=rowMorton(other);
    if(otherLevel<level||(otherLevel==level&&(otherMorton<morton
      ||(otherMorton==morton&&other<cell)))){lo=mid+1u;}else{hi=mid;}}
  if(lo<count&&frontierCell(current,lo)==cell&&ownerAtIndex(cell).size==size){return lo;}
  return 0xffffffffu;
}
fn frontierRowIdentity(cell:u32,size:u32)->u32{return frontierRowIdentityIn(cell,size,frontierCurrent());}
fn frontierRow(cell:u32)->u32{let owner=ownerAtIndex(cell);return select(0xffffffffu,
  frontierRowIdentity(cell,owner.size),ownerValid(owner));}
fn candidateFrontierCurrent()->u32{return select(frontierCurrent(),frontier[7u],frontier[6u]==1u);}
fn candidateFrontierRow(cell:u32)->u32{let owner=ownerAtIndex(cell);return select(0xffffffffu,
  frontierRowIdentityIn(cell,owner.size,candidateFrontierCurrent()),ownerValid(owner));}
fn frontierAlive(cell:u32)->bool{return frontierRow(cell)!=0xffffffffu;}
fn pressureIndex(owner: Owner) -> u32 {
  return frontierRowIdentity(index(unpackOrigin(owner.packedOrigin)),owner.size);
}
// Section 4.3 requires the L1 and L2 operators to use exactly the same set of
// pressure variables. Recurring frontier publication may classify a leaf from
// the newer complete fine-summary interval, while topology construction reads
// the published compact coarse/page authority. Once the
// compact frontier is published, membership in that frontier is therefore the
// pressure-variable authority; reclassifying an incident leaf here can create
// an L1 entry for a row that does not exist in L2.
fn pressureVariableExists(owner: Owner) -> bool {
  if (!rowIndexedPressure) { return liquidOwner(owner); }
  return pressureIndex(owner) < compaction[0];
}
// The trailing eight words are isolated from topology-change state and scan
// partials: overflow, required rows, required entries, exact dispatch xyz,
// then residual sums rr/bb.
fn pressureControlBase() -> u32 { return arrayLength(&compaction) - 8u; }
fn pressureOverflowed() -> bool {
  // Owner probes made while refining dry support may reject pages which never
  // become pressure rows. Do not let that transaction-wide diagnostic suppress
  // every row write. Each emitted row revalidates its own owner below, and the
  // downstream operator publisher fails closed if any header remains absent.
  return compaction[pressureControlBase()] != 0u;
}
fn axisVector(axis: u32) -> vec3i { return select(select(vec3i(0,0,1), vec3i(0,1,0), axis == 1u), vec3i(1,0,0), axis == 0u); }
fn worldCell(p: vec3i) -> vec3f {
  let h = params.cellRelax.xyz;
  return vec3f(-0.5 * params.container.x + (f32(p.x) + 0.5) * h.x, (f32(p.y) + 0.5) * h.y, -0.5 * params.container.z + (f32(p.z) + 0.5) * h.z);
}
fn quaternionRotate(q: vec4f, v: vec3f) -> vec3f { let uv = cross(q.yzw, v); let uuv = cross(q.yzw, uv); return v + 2.0 * (q.x * uv + uuv); }
fn quaternionInverseRotate(q: vec4f, v: vec3f) -> vec3f { return quaternionRotate(vec4f(q.x, -q.yzw), v); }
fn insideRigid(body: RigidBody, world: vec3f) -> bool {
  let p = quaternionInverseRotate(body.orientation, world - body.positionShape.xyz); let d = body.dimensions.xyz; let shape = i32(round(body.positionShape.w));
  if (shape == 0) { return length(p) <= d.x; }
  if (shape == 1) { return all(abs(p) <= 0.5 * d); }
  if (shape == 2) { let cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y); return length(vec3f(p.x, p.y - cy, p.z)) <= d.x; }
  return p.x * p.x + p.z * p.z <= d.x * d.x && abs(p.y) <= 0.5 * d.y;
}
fn insideInflowChannel(world: vec3f) -> bool {
  if (params.inflowPositionRadius.w <= 0.0 || params.inflowDirectionLength.w <= 0.0) { return false; }
  let delta = world - params.inflowPositionRadius.xyz;
  let along = dot(delta, params.inflowDirectionLength.xyz);
  let radial = delta - along * params.inflowDirectionLength.xyz;
  let margin = max(params.cellRelax.x, max(params.cellRelax.y, params.cellRelax.z));
  return abs(along) <= 0.5 * params.inflowDirectionLength.w + margin && length(radial) <= params.inflowPositionRadius.w + margin;
}
fn bodySolidFraction(body: RigidBody, p: vec3i) -> f32 {
  let center = worldCell(p); let h = params.cellRelax.xyz; var inside = 0.0;
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let offset = vec3f(select(-0.4, 0.4, (corner & 1u) != 0u), select(-0.4, 0.4, (corner & 2u) != 0u), select(-0.4, 0.4, (corner & 4u) != 0u));
    if (insideRigid(body, center + offset * h)) { inside += 1.0; }
  }
  return inside / 8.0;
}
// Evaluate the current authored occupancy without mutating the retained dense
// publication. The previous record and this evaluator form the exact old/new
// transaction consumed by topology dirty marking.
fn currentSolidAt(p: vec3i) -> SolidCell {
  if (!valid(p)) { return SolidCell(1.0, -1); }
  var fraction = 0.0; var owner = -1;
  if ((u32(round(params.container.w)) & 1u) != 0u) {
    fraction = clamp(textureLoad(terrainIn, vec2i(p.x, p.z), 0).x - f32(p.y), 0.0, 1.0);
  }
  if (!insideInflowChannel(worldCell(p))) {
    for (var bodyIndex = 0u; bodyIndex < 12u; bodyIndex += 1u) {
      if (bodyIndex >= params.control.w) { break; }
      let candidate = bodySolidFraction(rigidBodies[bodyIndex], p);
      if (candidate > fraction) { fraction = candidate; owner = i32(bodyIndex); }
    }
  }
  return SolidCell(fraction, owner);
}
fn solidAt(p: vec3i) -> SolidCell {
  if (!valid(p)) { return SolidCell(1.0, -1); }
  let i = index(vec3u(p));
  let word = 2u * i;
  if (word + 1u >= arrayLength(&solidCells)) { return SolidCell(0.0, -1); }
  return SolidCell(bitcast<f32>(solidCells[word]), bitcast<i32>(solidCells[word + 1u]));
}
fn candidateScanScratchBase() -> u32 { return 15u + 3u * params.control.z; }

// Topology-tile worklist header occupies words 0..15 of the copied buffer:
// word 0 the active tile count, word 1 the active dispatch x width, word 4
// and word 5 the retired equivalents. A tile spans max(8, maximumLeaf) cells
// per axis so every dyadic pressure leaf lies inside exactly one tile; each
// tile decomposes into (tileSize/4)^3 of the existing 4^3 cell workgroups.
fn topologyTileSize() -> u32 { return max(8u, params.dimsMax.w); }
fn deltaTopologyCell(workgroup: vec3u, local: vec3u) -> vec3u {
  let tileSize = topologyTileSize();
  let blocks = tileSize / 4u;
  let groupsPerTile = blocks * blocks * blocks;
  let linearWorkgroup = workgroup.x + workgroup.y * compaction[1];
  let streamIndex = linearWorkgroup / groupsPerTile;
  let total = compaction[0] + compaction[4];
  if (streamIndex >= total) { return vec3u(0xffffffffu); }
  let tile = deltaTileOrigin(streamIndex) / tileSize;
  let sub = linearWorkgroup % groupsPerTile;
  let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
  return tile * tileSize + subCoord * 4u + local;
}

// Refinement and balancing can only act on leaves of size >= 2. Their origins
// are even-aligned, so candidate passes cover an 8^3 cell region with each
// 4^3 workgroup instead of launching one invocation for every finest cell.
fn deltaTopologyCandidate(workgroup: vec3u, local: vec3u) -> vec3u {
  let tileSize = topologyTileSize();
  let blocks = max(1u, tileSize / 8u);
  let groupsPerTile = blocks * blocks * blocks;
  let linearWorkgroup = workgroup.x + workgroup.y * compaction[8];
  let streamIndex = linearWorkgroup / groupsPerTile;
  let total = compaction[0] + compaction[4];
  if (streamIndex >= total) { return vec3u(0xffffffffu); }
  let sub = linearWorkgroup % groupsPerTile;
  let tile = deltaTileOrigin(streamIndex) / tileSize;
  let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
  return tile * tileSize + subCoord * 8u + local * 2u;
}

fn deltaTileOrigin(slot: u32) -> vec3u {
  let retired = slot >= compaction[0];
  let localSlot = select(slot, slot - compaction[0], retired);
  return worklistTileOrigin(localSlot, select(16u, retiredTileIndexBase(), retired));
}

// The coarse cooperative kernels dispatch exactly one workgroup per worklist
// tile (the header tile counts are copied into dedicated indirect x slots on
// the CPU timeline), so wid.x always names a valid tile slot. Each workgroup
// walks its (tileSize/targetRefinementSize)^3 sub-blocks internally; the loop
// bound derives from override constants, keeping barrier control flow uniform.
fn worklistTileOrigin(slot: u32, indexBase: u32) -> vec3u {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tileIndex = compaction[indexBase + slot];
  return vec3u(tileIndex % tx, (tileIndex / tx) % ty, tileIndex / (tx * ty)) * tileSize;
}

fn retiredTileIndexBase() -> u32 {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return 16u + tx * ty * tz;
}

// ---- Exact structural topology/frontier delta -------------------------------
// Fine payload values are refreshed every step, but pressure topology and row
// membership depend only on the resulting owner/wet decisions. One parallel
// workgroup per active topology tile hashes those discrete decisions; the
// singleton below compacts only changed signatures, residency transitions, and
// rigid-body bounds. Unchanged phi magnitudes therefore publish zero work.

fn topologyTileCapacity() -> u32 {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return tx * ty * tz;
}
const RIGID_SNAPSHOT_WORDS: u32 = 146u;
const TILE_SIGNATURE_WORDS: u32 = 5u;
const TILE_SIGNATURE_STRUCTURAL_CHANGED: u32 = 1u;
const TILE_SIGNATURE_FRONTIER_CHANGED: u32 = 2u;
const DIRTY_TILE_VALID_MAGIC: u32 = 0x44544c54u;
const RIGID_SNAPSHOT_MAGIC: u32 = 0x52424744u;
// Structural word 4 packs a 24-bit validity tag and an 8-bit temporal
// retention counter. Current refinement evidence refreshes the counter to
// three; absence only decrements it, so retained protection cannot self-refresh.
// Boundary gating is deliberately orthogonal to this pressure-side policy.
const TILE_SIGNATURE_VALID_MAGIC: u32 = 0x0053474eu;
const TILE_SIGNATURE_VALID_MASK: u32 = 0x00ffffffu;
const PRESSURE_RETENTION_GENERATIONS: u32 = 3u;
const TILE_SIGNATURE_FAILED: u32 = 0xffffffffu;
fn changeStateWords() -> u32 {
  return 14u * topologyTileCapacity() + 1u + RIGID_SNAPSHOT_WORDS + 22u;
}
fn changeStateBase() -> u32 { return arrayLength(&compaction) - 8u - changeStateWords(); }
fn tileChangeFlagsBase() -> u32 { return changeStateBase(); }
fn dirtyListBase() -> u32 { return changeStateBase() + topologyTileCapacity(); }
fn tileSignatureBase() -> u32 { return changeStateBase() + 2u * topologyTileCapacity(); }
fn tileFrontierSignatureBase() -> u32 {
  return tileSignatureBase() + TILE_SIGNATURE_WORDS * topologyTileCapacity();
}
fn tileSignatureChangedBase() -> u32 {
  return tileFrontierSignatureBase() + TILE_SIGNATURE_WORDS * topologyTileCapacity();
}
fn tileFrontierChangeFlagsBase() -> u32 {
  return tileSignatureChangedBase() + topologyTileCapacity();
}
fn dirtyAuthorityBase() -> u32 {
  return tileFrontierChangeFlagsBase() + topologyTileCapacity();
}
fn rigidSnapshotBase() -> u32 { return dirtyAuthorityBase() + 1u; }
fn frontierPublicationBase() -> u32 {
  return rigidSnapshotBase() + RIGID_SNAPSHOT_WORDS;
}
fn frontierTopologyReuseBase() -> u32 { return frontierPublicationBase() + 13u; }
fn dirtyFailureBase() -> u32 { return frontierTopologyReuseBase() + 1u; }
const FRONTIER_REUSE_MAGIC: u32 = 0x46525553u;
const FRONTIER_FAILED_MAGIC: u32 = 0x4641494cu;
const COARSE_PREDICTED_WET_MAGIC: u32 = 0x43505754u;
const DIRTY_FAILURE_TILE_COUNTS: u32 = 1u;
const DIRTY_FAILURE_TILE_SIGNATURE: u32 = 2u;
const DIRTY_FAILURE_RETIRED_TILE: u32 = 3u;
const DIRTY_FAILURE_TILE_OVERFLOW: u32 = 4u;
const DIRTY_FAILURE_FRONTIER_COUNTS: u32 = 5u;
const DIRTY_FAILURE_FRONTIER_SIGNATURE: u32 = 6u;
const DIRTY_FAILURE_FRONTIER_OVERFLOW: u32 = 7u;
fn clearDirtyFailure() {
  for (var word = 0u; word < 8u; word += 1u) {
    compaction[dirtyFailureBase() + word] = 0u;
  }
}
fn rejectDirtyAuthority(reason: u32, stage: u32, slot: u32, tileIndex: u32,
    activeCount: u32, retiredCount: u32, capacity: u32) {
  compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
  if (compaction[dirtyFailureBase()] != 0u) { return; }
  compaction[dirtyFailureBase()] = reason;
  compaction[dirtyFailureBase() + 1u] = stage;
  compaction[dirtyFailureBase() + 2u] = slot;
  compaction[dirtyFailureBase() + 3u] = tileIndex;
  compaction[dirtyFailureBase() + 4u] = activeCount;
  compaction[dirtyFailureBase() + 5u] = retiredCount;
  compaction[dirtyFailureBase() + 6u] = capacity;
  compaction[dirtyFailureBase() + 7u] = frontier[3];
}
fn frontierGenerationReused() -> bool {
  return compaction[11] == FRONTIER_REUSE_MAGIC
    || compaction[frontierTopologyReuseBase()] != 0u;
}
// Between structural-delta classification and beginFrontier this word is the
// exact quiescence latch.  The resident grading closure restores compaction's
// capacity-shaped active-tile header, so it cannot recover the prior zero
// dirty count from words 0..10.  beginFrontier clears the latch before the
// same word resumes its existing frontier-publication meaning.
fn topologyStructurallyQuiescent() -> bool {
  return compaction[frontierTopologyReuseBase()] != 0u;
}

fn residencyTiledDispatch(blocks: u32) -> vec2u {
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  return vec2u(x, y);
}

fn tileStateWord(index: u32) -> u32 { return bitcast<u32>(pressureOut[index]); }
fn tileStateHash(key: u32) -> u32 {
  var x = key * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}
fn topologyTileActive(key: u32) -> bool {
  if (key >= topologyTileCapacity()) { return false; }
  if (sparseTopologyTileStates == 0u) {
    return key < arrayLength(&pressureOut) && tileStateWord(key) != 0u;
  }
  let slots = arrayLength(&pressureOut) / 2u;
  if (slots == 0u) { return false; }
  let encoded = key + 1u;
  let start = tileStateHash(key) % slots;
  for (var probe = 0u; probe < slots; probe += 1u) {
    let slot = (start + probe) % slots;
    let stored = tileStateWord(2u * slot);
    if (stored == encoded) { return tileStateWord(2u * slot + 1u) != 0u; }
    if (stored == 0u) { return false; }
  }
  return false;
}
fn appendDirtyTile(tileIndex: u32, generation: u32, count: ptr<function, u32>) {
  if (!topologyTileActive(tileIndex)
      || compaction[tileChangeFlagsBase() + tileIndex] == generation) { return; }
  if (*count >= topologyTileCapacity()) {
    rejectDirtyAuthority(DIRTY_FAILURE_TILE_OVERFLOW, 1u, *count, tileIndex,
      compaction[0], compaction[4], topologyTileCapacity());
    return;
  }
  compaction[tileChangeFlagsBase() + tileIndex] = generation;
  compaction[dirtyListBase() + *count] = tileIndex;
  *count += 1u;
}
fn appendDirtyTileRing(tileIndex: u32, generation: u32, count: ptr<function, u32>) {
  let tileSize = topologyTileSize();
  let td = vec3u((dims().x + tileSize - 1u) / tileSize,
    (dims().y + tileSize - 1u) / tileSize, (dims().z + tileSize - 1u) / tileSize);
  let tile = vec3i(i32(tileIndex % td.x), i32((tileIndex / td.x) % td.y),
    i32(tileIndex / (td.x * td.y)));
  for (var z = -1; z <= 1; z += 1) { for (var y = -1; y <= 1; y += 1) {
    for (var x = -1; x <= 1; x += 1) {
      let q = tile + vec3i(x, y, z);
      if (any(q < vec3i(0)) || any(q >= vec3i(td))) { continue; }
      appendDirtyTile(u32(q.x) + td.x * (u32(q.y) + td.y * u32(q.z)),
        generation, count);
    }
  } }
}
fn topologyDecisionHash(value: u32) -> u32 {
  var x = value * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}
var<workgroup> tileSignatureReduction: array<vec4u, 256>;
var<workgroup> tileFrontierSignatureReduction: array<vec4u, 256>;
var<workgroup> tileEvidenceReduction: array<u32, 256>;
@compute @workgroup_size(256)
fn classifyTopologyTileSignature(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let capacity = topologyTileCapacity();
  let activeCount = min(compaction[0], capacity);
  let validSlot = wid.x < activeCount;
  let safeSlot = select(0u, wid.x, validSlot);
  let tileIndex = compaction[16u + safeSlot];
  let validTile = validSlot && tileIndex < capacity && topologyTileActive(tileIndex);
  let safeTileIndex = select(0u, tileIndex, validTile);
  let tileSize = topologyTileSize();
  let td = (dims() + vec3u(tileSize - 1u)) / tileSize;
  let tile = vec3u(safeTileIndex % td.x, (safeTileIndex / td.x) % td.y,
    safeTileIndex / (td.x * td.y));
  let origin = tile * tileSize;
  let cellCount = tileSize * tileSize * tileSize;
  var signature = vec4u(0u);
  var frontierSignature = vec4u(0u);
  var refinementEvidenceCount = 0u;
  if (validTile) {
    for (var flat = lid; flat < cellCount; flat += 256u) {
      let local = vec3u(flat % tileSize, (flat / tileSize) % tileSize,
        flat / (tileSize * tileSize));
      let q = origin + local;
      if (any(q >= dims())) { continue; }
      let cell = index(q);
      let owner = ownerAtIndex(cell);
      if (!ownerValid(owner) || !isOrigin(q, owner)) { continue; }
      let wet = currentPressureOwnerWet(owner);
      // Fine-interface and inflow protection are structural sizing inputs.
      // Folding them into the retained signature is what turns surface motion
      // into an exact dirty-tile transaction instead of silently carrying a
      // coarse pressure topology underneath a moving fine band.
      let refinementEvidence = pressureRefinementEvidence(unpackOrigin(owner.packedOrigin), owner.size);
      let structuralDecision = cell ^ (owner.size * 0x9e3779b9u)
        ^ select(0u, 0x27d4eb2du, refinementEvidence);
      let frontierDecision = structuralDecision ^ select(0u, 0x85ebca6bu, wet);
      signature.x ^= topologyDecisionHash(structuralDecision);
      signature.y += topologyDecisionHash(structuralDecision ^ 0xc2b2ae35u);
      signature.z += 1u;
      signature.w += owner.size;
      refinementEvidenceCount += select(0u, 1u, refinementEvidence);
      frontierSignature.x ^= topologyDecisionHash(frontierDecision);
      frontierSignature.y += topologyDecisionHash(frontierDecision ^ 0xc2b2ae35u);
      frontierSignature.z += 1u;
      frontierSignature.w += select(0u, 1u, wet);
    }
  }
  tileSignatureReduction[lid] = signature;
  tileFrontierSignatureReduction[lid] = frontierSignature;
  tileEvidenceReduction[lid] = refinementEvidenceCount;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) {
      let right = tileSignatureReduction[lid + stride];
      tileSignatureReduction[lid] = vec4u(
        tileSignatureReduction[lid].x ^ right.x,
        tileSignatureReduction[lid].yzw + right.yzw,
      );
      let frontierRight = tileFrontierSignatureReduction[lid + stride];
      tileFrontierSignatureReduction[lid] = vec4u(
        tileFrontierSignatureReduction[lid].x ^ frontierRight.x,
        tileFrontierSignatureReduction[lid].yzw + frontierRight.yzw,
      );
      tileEvidenceReduction[lid] += tileEvidenceReduction[lid + stride];
    }
  }
  workgroupBarrier();
  if (lid != 0u) { return; }
  if (!validSlot) { return; }
  if (!validTile) {
    compaction[tileSignatureChangedBase() + wid.x] = TILE_SIGNATURE_FAILED;
    return;
  }
  let base = tileSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
  var next = tileSignatureReduction[0];
  let priorState = compaction[base + 4u];
  let valid = (priorState & TILE_SIGNATURE_VALID_MASK) == TILE_SIGNATURE_VALID_MAGIC;
  let priorRetention = select(0u, priorState >> 24u, valid);
  // Factor 4/8 already carry a fine pressure shell. Publishing tile-wide
  // hysteresis for their fluid-gated topology grows that shell after the
  // first advance and changes the mini-dam impulse response. Factor 1 needs
  // the retention because it has no finer pressure-support lattice. Boundary
  // policy stays orthogonal: the unconditional control may still split dry
  // wall leaves without widening the liquid pressure shell.
  let retainPressureHysteresis = fineSummaryFactor == 1u;
  let currentEvidence = retainPressureHysteresis
    && tileEvidenceReduction[0] != 0u;
  let retention = select(select(0u, priorRetention - 1u, priorRetention > 0u),
    PRESSURE_RETENTION_GENERATIONS, currentEvidence);
  next.x ^= topologyDecisionHash(0x68bc21ebu ^ retention);
  let structuralUnchanged = valid && all(vec4u(compaction[base], compaction[base + 1u],
    compaction[base + 2u], compaction[base + 3u]) == next);
  compaction[base] = next.x; compaction[base + 1u] = next.y;
  compaction[base + 2u] = next.z; compaction[base + 3u] = next.w;
  compaction[base + 4u] = TILE_SIGNATURE_VALID_MAGIC | (retention << 24u);
  let frontierBase = tileFrontierSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
  let frontierNext = tileFrontierSignatureReduction[0];
  let frontierValid = compaction[frontierBase + 4u] == TILE_SIGNATURE_VALID_MAGIC;
  let frontierUnchanged = frontierValid && all(vec4u(compaction[frontierBase],
    compaction[frontierBase + 1u], compaction[frontierBase + 2u],
    compaction[frontierBase + 3u]) == frontierNext);
  compaction[frontierBase] = frontierNext.x;
  compaction[frontierBase + 1u] = frontierNext.y;
  compaction[frontierBase + 2u] = frontierNext.z;
  compaction[frontierBase + 3u] = frontierNext.w;
  compaction[frontierBase + 4u] = TILE_SIGNATURE_VALID_MAGIC;
  compaction[tileSignatureChangedBase() + wid.x] =
    select(TILE_SIGNATURE_STRUCTURAL_CHANGED, 0u, structuralUnchanged)
    | select(TILE_SIGNATURE_FRONTIER_CHANGED, 0u, frontierUnchanged);
}
fn currentRigidWord(body: u32, word: u32) -> u32 {
  let value = rigidBodies[body];
  switch word {
    case 0u: { return bitcast<u32>(value.positionShape.x); }
    case 1u: { return bitcast<u32>(value.positionShape.y); }
    case 2u: { return bitcast<u32>(value.positionShape.z); }
    case 3u: { return bitcast<u32>(value.positionShape.w); }
    case 4u: { return bitcast<u32>(value.dimensions.x); }
    case 5u: { return bitcast<u32>(value.dimensions.y); }
    case 6u: { return bitcast<u32>(value.dimensions.z); }
    case 7u: { return bitcast<u32>(value.dimensions.w); }
    case 8u: { return bitcast<u32>(value.orientation.x); }
    case 9u: { return bitcast<u32>(value.orientation.y); }
    case 10u: { return bitcast<u32>(value.orientation.z); }
    default: { return bitcast<u32>(value.orientation.w); }
  }
}
fn snapshotRigidBody(body: u32) -> RigidBody {
  let base = rigidSnapshotBase() + 2u + 12u * body;
  return RigidBody(
    vec4f(bitcast<f32>(compaction[base]), bitcast<f32>(compaction[base + 1u]),
      bitcast<f32>(compaction[base + 2u]), bitcast<f32>(compaction[base + 3u])),
    vec4f(bitcast<f32>(compaction[base + 4u]), bitcast<f32>(compaction[base + 5u]),
      bitcast<f32>(compaction[base + 6u]), bitcast<f32>(compaction[base + 7u])),
    vec4f(bitcast<f32>(compaction[base + 8u]), bitcast<f32>(compaction[base + 9u]),
      bitcast<f32>(compaction[base + 10u]), bitcast<f32>(compaction[base + 11u])),
    vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
}
fn rigidBodyChanged(body: u32, currentCount: u32, previousCount: u32) -> bool {
  if (body >= currentCount || body >= previousCount) { return true; }
  let base = rigidSnapshotBase() + 2u + 12u * body;
  for (var word = 0u; word < 12u; word += 1u) {
    if (compaction[base + word] != currentRigidWord(body, word)) { return true; }
  }
  return false;
}
fn rigidHalfExtent(body: RigidBody) -> vec3f {
  let d = max(vec3f(0.0), body.dimensions.xyz);
  let axisX = abs(quaternionRotate(body.orientation, vec3f(1.0, 0.0, 0.0)));
  let axisY = abs(quaternionRotate(body.orientation, vec3f(0.0, 1.0, 0.0)));
  let axisZ = abs(quaternionRotate(body.orientation, vec3f(0.0, 0.0, 1.0)));
  let shape = i32(round(body.positionShape.w));
  if (shape == 0) { return vec3f(d.x); }
  if (shape == 1) { return 0.5 * (d.x * axisX + d.y * axisY + d.z * axisZ); }
  if (shape == 2) { return vec3f(d.x) + 0.5 * d.y * axisY; }
  return vec3f(d.x) * sqrt(max(vec3f(0.0), vec3f(1.0) - axisY * axisY))
    + 0.5 * d.y * axisY;
}
fn appendRigidBounds(body: RigidBody, generation: u32, count: ptr<function, u32>) {
  let extent = rigidHalfExtent(body) + 0.5 * params.cellRelax.xyz;
  let domainMinimum = vec3f(-0.5 * params.container.x, 0.0, -0.5 * params.container.z);
  let firstCell = clamp(vec3i(floor((body.positionShape.xyz - extent - domainMinimum)
    / params.cellRelax.xyz)), vec3i(0), vec3i(dims()) - vec3i(1));
  let lastCell = clamp(vec3i(floor((body.positionShape.xyz + extent - domainMinimum)
    / params.cellRelax.xyz)), vec3i(0), vec3i(dims()) - vec3i(1));
  let first = firstCell / i32(topologyTileSize());
  let last = lastCell / i32(topologyTileSize());
  let td = (dims() + vec3u(topologyTileSize() - 1u)) / topologyTileSize();
  for (var z = first.z; z <= last.z; z += 1) { for (var y = first.y; y <= last.y; y += 1) {
    for (var x = first.x; x <= last.x; x += 1) {
      let tileIndex = u32(x) + td.x * (u32(y) + td.y * u32(z));
      appendDirtyTileRing(tileIndex, generation, count);
    }
  } }
}
@compute @workgroup_size(1)
fn buildDirtyTileDelta() {
  // Dirty membership belongs to the candidate attempt, not to the last
  // accepted frontier plus one. A rejected attempt deliberately leaves
  // frontier[3] unchanged while stampFrontierAttempt advances frontier[8].
  // Using the accepted clock here made every retry stamp the old generation;
  // carry validation then treated genuinely changed rows as clean and turned
  // one recoverable rejection into a permanent topology freeze.
  let generation = frontier[8u];
  var dirtyCount = 0u;
  clearDirtyFailure();
  compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
  let capacity = topologyTileCapacity();
  let activeCount = compaction[0];
  if (activeCount > capacity || compaction[4] > capacity) {
    rejectDirtyAuthority(DIRTY_FAILURE_TILE_COUNTS, 1u, 0u, 0u,
      activeCount, compaction[4], capacity);
    compaction[0] = 0u;
    compaction[4] = 0u;
    compaction[1] = 0u; compaction[2] = 1u; compaction[3] = 1u;
    compaction[5] = 0u; compaction[6] = 1u; compaction[7] = 1u;
    compaction[8] = 0u; compaction[9] = 1u; compaction[10] = 1u;
    return;
  }
  compaction[dirtyAuthorityBase()] = DIRTY_TILE_VALID_MAGIC;
  for (var slot = 0u; slot < activeCount; slot += 1u) {
    let tileIndex = compaction[16u + slot];
    let changed = compaction[tileSignatureChangedBase() + slot];
    if (tileIndex >= capacity || changed == TILE_SIGNATURE_FAILED) {
      rejectDirtyAuthority(DIRTY_FAILURE_TILE_SIGNATURE, 1u, slot, tileIndex,
        activeCount, compaction[4], capacity);
      break;
    }
    if ((changed & TILE_SIGNATURE_STRUCTURAL_CHANGED) != 0u) {
      appendDirtyTileRing(tileIndex, generation, &dirtyCount);
    }
  }
  // A retired residency tile dirties its surviving active neighbors. The
  // retired tile itself is reset by the independent retired dispatch and its
  // persistent decision signature is invalidated before possible reuse.
  for (var slot = 0u; slot < compaction[4]; slot += 1u) {
    let tileIndex = compaction[retiredTileIndexBase() + slot];
    if (tileIndex >= capacity) {
      rejectDirtyAuthority(DIRTY_FAILURE_RETIRED_TILE, 1u, slot, tileIndex,
        activeCount, compaction[4], capacity);
      break;
    }
    let signature = tileSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
    compaction[signature + 4u] = 0u;
    let frontierSignature = tileFrontierSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
    compaction[frontierSignature + 4u] = 0u;
    // The tile is no longer active, so appendDirtyTileRing deliberately will
    // not add the tile itself to the active dirty list. Its old frontier rows
    // still belong to this generation's changed authority, however: without
    // this stamp classifyFrontierCarry treats those identities as clean and
    // carries headers whose owners were reset through the retired dispatch.
    compaction[tileChangeFlagsBase() + tileIndex] = generation;
    appendDirtyTileRing(tileIndex, generation, &dirtyCount);
  }
  let snapshotValid = compaction[rigidSnapshotBase()] == RIGID_SNAPSHOT_MAGIC;
  let previousBodies = select(0u, min(12u, compaction[rigidSnapshotBase() + 1u]), snapshotValid);
  let currentBodies = min(12u, params.control.w);
  for (var body = 0u; body < max(previousBodies, currentBodies); body += 1u) {
    if (!snapshotValid || rigidBodyChanged(body, currentBodies, previousBodies)) {
      if (body < previousBodies) { appendRigidBounds(snapshotRigidBody(body), generation, &dirtyCount); }
      if (body < currentBodies) { appendRigidBounds(rigidBodies[body], generation, &dirtyCount); }
    }
  }
  if (compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC) {
    compaction[rigidSnapshotBase()] = RIGID_SNAPSHOT_MAGIC;
    compaction[rigidSnapshotBase() + 1u] = currentBodies;
    for (var body = 0u; body < 12u; body += 1u) {
      let base = rigidSnapshotBase() + 2u + 12u * body;
      for (var word = 0u; word < 12u; word += 1u) {
        compaction[base + word] = select(0u, currentRigidWord(body, word), body < currentBodies);
      }
    }
  } else {
    dirtyCount = 0u;
  }
  for (var slot = 0u; slot < dirtyCount; slot += 1u) {
    compaction[16u + slot] = compaction[dirtyListBase() + slot];
  }
  compaction[0] = dirtyCount;
  let validDelta = compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC;
  compaction[4] = select(0u, compaction[4], validDelta);
  let totalTiles = dirtyCount + compaction[4];
  let blocks = topologyTileSize() / 4u;
  let tileDispatch = residencyTiledDispatch(totalTiles * blocks * blocks * blocks);
  compaction[1] = tileDispatch.x; compaction[2] = tileDispatch.y; compaction[3] = 1u;
  compaction[5] = totalTiles; compaction[6] = 1u; compaction[7] = 1u;
  let candidateBlocks = max(1u, topologyTileSize() / 8u);
  let candidateDispatch = residencyTiledDispatch(
    totalTiles * candidateBlocks * candidateBlocks * candidateBlocks);
  compaction[8] = candidateDispatch.x; compaction[9] = candidateDispatch.y; compaction[10] = 1u;
  // Persist the zero-delta decision across the full-residency worklist restore
  // used by grading.  This is GPU-authored and consumed by the grading
  // kernels; the host continues to encode the same static closure.
  compaction[frontierTopologyReuseBase()] = select(0u, 1u,
    validDelta && totalTiles == 0u);
  if (validDelta && compaction[dirtyFailureBase()] == 0u) {
    compaction[dirtyFailureBase()] = 0x100u;
  }
}

@compute @workgroup_size(1)
fn buildDirtyFrontierDelta() {
  let generation = frontier[8u];
  let capacity = topologyTileCapacity();
  let activeCount = compaction[0];
  var dirtyCount = 0u;
  clearDirtyFailure();
  compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
  if (activeCount > capacity || compaction[4] > capacity) {
    rejectDirtyAuthority(DIRTY_FAILURE_FRONTIER_COUNTS, 2u, 0u, 0u,
      activeCount, compaction[4], capacity);
    compaction[0] = 0u; compaction[4] = 0u;
    compaction[8] = 0u; compaction[9] = 1u; compaction[10] = 1u;
    return;
  }
  compaction[dirtyAuthorityBase()] = DIRTY_TILE_VALID_MAGIC;
  // The residency worklist is unique and sorted, so exact frontier tiles need
  // no atomic deduplication. Structural/rigid rings are already stamped for
  // this generation; wet-only changes stamp just their own tile and let the
  // later row-delta one-ring expand pressure consumers.
  for (var slot = 0u; slot < activeCount; slot += 1u) {
    let tileIndex = compaction[16u + slot];
    let changed = compaction[tileSignatureChangedBase() + slot];
    compaction[tileSignatureChangedBase() + slot] = 0u;
    if (tileIndex >= capacity || changed == TILE_SIGNATURE_FAILED) {
      rejectDirtyAuthority(DIRTY_FAILURE_FRONTIER_SIGNATURE, 2u, slot, tileIndex,
        activeCount, compaction[4], capacity);
      break;
    }
    let structural = compaction[tileChangeFlagsBase() + tileIndex] == generation;
    // The row frontier is a function of the exact structural and wet/dry
    // decision fingerprints. A membership-only shortcut is not sound here:
    // free-surface fractions and Section 4 coefficients can change while row
    // identity remains stable.
    let wet = (changed & TILE_SIGNATURE_FRONTIER_CHANGED) != 0u;
    if (structural || wet) {
      if (dirtyCount >= capacity) {
        rejectDirtyAuthority(DIRTY_FAILURE_FRONTIER_OVERFLOW, 2u, slot, tileIndex,
          activeCount, compaction[4], capacity);
        break;
      }
      if (wet) { compaction[tileFrontierChangeFlagsBase() + tileIndex] = generation; }
      compaction[dirtyListBase() + dirtyCount] = tileIndex;
      dirtyCount += 1u;
    }
  }
  if (compaction[dirtyAuthorityBase()] != DIRTY_TILE_VALID_MAGIC) {
    dirtyCount = 0u;
    compaction[4] = 0u;
  }
  for (var slot = 0u; slot < dirtyCount; slot += 1u) {
    compaction[16u + slot] = compaction[dirtyListBase() + slot];
  }
  compaction[0] = dirtyCount;
  let totalTiles = dirtyCount + compaction[4];
  let candidateBlocks = max(1u, topologyTileSize() / 8u);
  let candidateDispatch = residencyTiledDispatch(
    totalTiles * candidateBlocks * candidateBlocks * candidateBlocks);
  compaction[8] = candidateDispatch.x;
  compaction[9] = candidateDispatch.y;
  compaction[10] = 1u;
  if (compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC
      && compaction[dirtyFailureBase()] == 0u) {
    compaction[dirtyFailureBase()] = 0x200u;
  }
}
// -----------------------------------------------------------------------------

fn rasterizeSolidsAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let solid = currentSolidAt(vec3i(gid));
  let word = 2u * index(gid);
  solidCells[word] = bitcast<u32>(solid.fraction);
  solidCells[word + 1u] = bitcast<u32>(solid.owner);
}

@compute @workgroup_size(4,4,4)
fn rasterizeSolids(@builtin(global_invocation_id) gid: vec3u) { rasterizeSolidsAt(gid); }

@compute @workgroup_size(4,4,4)
fn rasterizeSolidsDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  rasterizeSolidsAt(deltaTopologyCell(wid, lid));
}

fn resetTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  var size = params.dimsMax.w;
  var origin = (gid / vec3u(size)) * vec3u(size);
  loop {
    if (all(origin + vec3u(size) <= dims()) || size == 1u) { break; }
    size = size / 2u; origin = (gid / vec3u(size)) * vec3u(size);
  }
  storeOwner(gid, origin, size);
}

@compute @workgroup_size(4,4,4)
fn resetTopology(@builtin(global_invocation_id) gid: vec3u) { resetTopologyAt(gid); }

@compute @workgroup_size(4,4,4)
fn resetTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  resetTopologyAt(deltaTopologyCell(wid, lid));
}

struct FineLeafSummary {
  found: bool,
  complete: bool,
  coarseAuthority: bool,
  centerValid: bool,
  exactCellValid: bool,
  exactCellNegative: bool,
  centerPhi: f32,
  minimumPhi: f32,
  maximumPhi: f32,
  minimumAbsolutePhi: f32,
}
fn fineSummaryFinite(value: f32) -> bool { return value == value && abs(value) < 3.402823e38; }
fn fineSummaryOrderedFloat(value: u32) -> f32 {
  let mask = select(0x80000000u, 0xffffffffu, (value & 0x80000000u) == 0u);
  return bitcast<f32>(value ^ mask);
}
// Refinement-only bind groups alias pressureIn (binding 4) with the raw
// summary directory. Other entry points retain the normal pressure buffer.
fn fineSummaryLength() -> u32 { return arrayLength(&pressureIn); }
fn fineSummaryWord(index: u32) -> u32 { return bitcast<u32>(pressureIn[index]); }
fn fineLeafSummary(origin: vec3u, size: u32) -> FineLeafSummary {
  var result = FineLeafSummary(false, false, false, false, false, false, 0.0,
    3.402823e38, -3.402823e38, 3.402823e38);
  if (fineSummaryLength() < 16u || fineSummaryWord(0u) != 0u
      || fineSummaryWord(9u) != 0x80000000u) { return result; }
  let baseDims = vec3u(fineSummaryWord(4u), fineSummaryWord(5u), fineSummaryWord(6u));
  let cellDims = dims();
  if (any(cellDims == vec3u(0u))) { return result; }
  let factorOne = fineSummaryFactor == 1u;
  var bricksPerCell = 0u;
  if (factorOne) {
    if (any(baseDims != (cellDims + vec3u(3u)) / 4u)) { return result; }
  } else {
    if (any(baseDims % cellDims != vec3u(0u))) { return result; }
    let ratios = baseDims / cellDims; bricksPerCell = ratios.x;
    if (bricksPerCell == 0u || any(ratios != vec3u(bricksPerCell))) { return result; }
  }
  // One factor-1 B4 leaf spans four finest cells per axis. Sizes 1 and 2
  // deliberately consume that containing leaf's conservative interval; size
  // 4 is the first exact geometric match, and each larger dyadic size climbs
  // one summary level per doubling.
  var brickSide = select(size * bricksPerCell, max(1u, size / 4u), factorOne);
  var level = 0u;
  if (brickSide == 0u || (brickSide & (brickSide - 1u)) != 0u) { return result; }
  var levelOffset = 0u; var levelDims = baseDims;
  var remaining = brickSide;
  loop {
    if (remaining == 1u) { break; }
    levelOffset += levelDims.x * levelDims.y * levelDims.z;
    levelDims = (levelDims + vec3u(1u)) / 2u;
    remaining >>= 1u; level += 1u;
  }
  if (level > fineSummaryWord(7u)) { return result; }
  let brickOrigin = select(origin * bricksPerCell, origin / 4u, factorOne);
  if (factorOne && size >= 4u && any(origin % vec3u(size) != vec3u(0u))) { return result; }
  if (any(brickOrigin % vec3u(brickSide) != vec3u(0u))) { return result; }
  let coordinate = brickOrigin / brickSide;
  if (any(coordinate >= levelDims)) { return result; }
  let key = levelOffset + coordinate.x + levelDims.x * (coordinate.y + levelDims.y * coordinate.z);
  let hierarchyCapacity = fineSummaryWord(10u);
  let count = fineSummaryWord(2u); let capacity = fineSummaryWord(3u);
  let entryOffset = fineSummaryWord(8u);
  let pageSize = fineSummaryWord(14u); let topLevelPages = fineSummaryWord(15u);
  let expectedTopLevelPages = hierarchyCapacity / ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u
    + select(0u, 1u, hierarchyCapacity % ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u != 0u);
  let pagePoolOffset = 16u + topLevelPages;
  if (key >= hierarchyCapacity || count > capacity || pageSize != ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u
      || topLevelPages != expectedTopLevelPages || pagePoolOffset > fineSummaryLength()
      || entryOffset < pagePoolOffset
      || (entryOffset - pagePoolOffset) % ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u != 0u
      || capacity > (fineSummaryLength() - entryOffset) / ${FINE_LEVELSET_SUMMARY_ENTRY_WORDS}u) { return result; }
  let directoryPageCapacity = (entryOffset - pagePoolOffset) / ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u;
  // The publisher owns a bounded sparse two-level hierarchy-key -> active-rank directory.
  // Refinement therefore performs one page load, one rank load, and one compact entry load;
  // the recurring sort/merge stream and binary search do not exist.
  let pageRankPlusOne = fineSummaryWord(16u + key / ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u);
  if (pageRankPlusOne == 0u || pageRankPlusOne > directoryPageCapacity) { return result; }
  let pageWord = pagePoolOffset + (pageRankPlusOne - 1u) * ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u
    + (key & ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE - 1}u);
  if (pageWord >= entryOffset) { return result; }
  let rankPlusOne = fineSummaryWord(pageWord);
  if (rankPlusOne != 0u && rankPlusOne <= capacity) {
    let base = entryOffset + (rankPlusOne - 1u) * ${FINE_LEVELSET_SUMMARY_ENTRY_WORDS}u;
    if (fineSummaryWord(base) != key) { return result; }
    let minimumPhi = fineSummaryOrderedFloat(fineSummaryWord(base + 1u));
    let maximumPhi = fineSummaryOrderedFloat(fineSummaryWord(base + 2u));
    let minimumAbsolutePhi = bitcast<f32>(fineSummaryWord(base + 3u));
    let entryFlags = fineSummaryWord(base + 6u);
    if ((entryFlags & 0x003fffffu) != 0u || !fineSummaryFinite(minimumPhi)
        || !fineSummaryFinite(maximumPhi) || !fineSummaryFinite(minimumAbsolutePhi)) { return result; }
    let expectedBricks = brickSide * brickSide * brickSide;
    result.found = true; result.minimumPhi = minimumPhi; result.maximumPhi = maximumPhi;
    result.minimumAbsolutePhi = minimumAbsolutePhi;
    result.coarseAuthority = (entryFlags & 0x80000000u) != 0u;
    let samplesPerBrick = fineSummaryWord(11u);
    let fineComplete = fineSummaryWord(base + 5u) == expectedBricks
      && (samplesPerBrick == 64u || samplesPerBrick == 512u)
      && fineSummaryWord(base + 4u) == expectedBricks * samplesPerBrick;
    result.complete = result.coarseAuthority || fineComplete;
    result.centerPhi = bitcast<f32>(fineSummaryWord(base + 7u));
    // A corrected-coarse interval and complete fine samples intentionally
    // coexist in the unified entry. Coarse authority must not hide the exact
    // fine centre: pure-coarse entries have zero fine counts and therefore
    // fail fineComplete without overloading centerPhi=0 as evidence.
    // Section 5 requires the new octree to consume the current advected level
    // set. At factor 4/8 the summary node and pressure leaf share a geometric
    // centre. At factor 1 the B4 node is larger than size-1/2 pressure leaves,
    // so only its interval is conservative for them; its centre becomes exact
    // at size 4 and above. Requiring fineComplete also excludes a pure-coarse
    // collision from masquerading as that fine centre.
    let centerMatchesLeaf = !factorOne || size >= 4u;
    result.centerValid = centerMatchesLeaf
      && (!factorOne || fineComplete)
      && (entryFlags & 0x3fc00000u) == 0x3fc00000u
      && fineSummaryFinite(result.centerPhi);
    // Factor 1 packs the exact validity and phase of all 4^3 finest-cell
    // samples in its level-zero B4 entry. A unit pressure owner can therefore
    // consume its own advected phi sign without inventing a finer surface
    // hierarchy or falling back to the previous coarse frontier.
    if (factorOne && size == 1u
        && ((samplesPerBrick == 64u && fineSummaryWord(base + 5u) == 1u)
          || (result.coarseAuthority && fineSummaryWord(base + 4u) == 64u
            && fineSummaryWord(base + 5u) == 1u))) {
      let local = origin & vec3u(3u);
      let bit = local.x + 4u * (local.y + 4u * local.z);
      let word = bit >> 5u; let mask = 1u << (bit & 31u);
      result.exactCellValid = (fineSummaryWord(base + 8u + word) & mask) != 0u;
      result.exactCellNegative = (fineSummaryWord(base + 10u + word) & mask) != 0u;
    }
    return result;
  }
  return result;
}

fn powerClosedWallStripIntersects(origin: vec3u, size: u32) -> bool {
  let flags = u32(round(params.container.w));
  let width = max(${OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS}u,
    u32(ceil(max(0.0, params.solve.w))));
  let high = origin + vec3u(size);
  let d = dims();
  // x+/-, z+/-, and the floor are closed for every container. The ceiling
  // participates only for an authored closed-top scene (flag bit 1). This
  // identifies candidates for the regular-cube Section 5 wall path; the
  // unconditional control splits all of them to unit owners, while the
  // adaptive policy retains that resolution only when liquid is nearby.
  return origin.x < min(width, d.x) || high.x > d.x - min(width, d.x)
    || origin.z < min(width, d.z) || high.z > d.z - min(width, d.z)
    || origin.y < min(width, d.y)
    || ((flags & 2u) != 0u && high.y > d.y - min(width, d.y));
}

fn inflowProtectionIntersects(origin: vec3u, size: u32) -> bool {
  if (params.inflowPositionRadius.w <= 0.0 || params.inflowDirectionLength.w <= 0.0) {
    return false;
  }
  let h = params.cellRelax.xyz;
  let halfExtent = 0.5 * f32(size) * h;
  let center = worldCell(vec3i(origin)) + 0.5 * f32(size - 1u) * h;
  let delta = center - params.inflowPositionRadius.xyz;
  let direction = params.inflowDirectionLength.xyz;
  let along = dot(delta, direction);
  let radial = delta - along * direction;
  let alongRadius = dot(abs(direction), halfExtent);
  let radialRadius = length(halfExtent);
  let authoredHalfLength = 0.5 * params.inflowDirectionLength.w;
  return abs(along) <= authoredHalfLength + alongRadius
    && length(radial) <= params.inflowPositionRadius.w + radialRadius;
}

fn pressureRefinementEvidence(origin: vec3u, size: u32) -> bool {
  if (inflowProtectionIntersects(origin, size)) { return true; }
  let summary = fineLeafSummary(origin, size);
  if (!summary.found) { return false; }
  // Factor one has no finer surface lattice from which to recover outward
  // motion. Keep both wet and dry children of each represented B4 block at
  // unit pressure resolution; this is the coarse air-side support halo, not a
  // second level-set field.
  // The fine-summary values and cell spacing are physical. Two authored
  // bands are retained here: the requested interface band and one additional
  // displacement/support ring the width of this leaf's own edge. This spatial
  // retention is the pressure-side coarsening hysteresis and prevents
  // alternating split/carry decisions as the zero set crosses a dyadic
  // boundary. It also keeps the coarse and fine publication clocks coherent
  // across transported frames.
  //
  let cellWidth = max(params.cellRelax.x, max(params.cellRelax.y, params.cellRelax.z));
  let gradingLayers = f32(max(1u, params.pressureCapacity.z));
  let retainedProtectionWidth = (max(1.0, params.solve.w)
    + gradingLayers * max(2.0, f32(size))) * cellWidth;
  // The fine factor-4/8 gated path already owns sub-cell interface support.
  // Preserve its compact pressure band: only the reach beyond the smallest
  // merge candidate needs to scale with this leaf. Factor 1 keeps the wider
  // shell used by its sole coarse surface tracker.
  let compactProtectionWidth = (max(1.0, params.solve.w)
    + gradingLayers * max(0.0, f32(size) - 2.0)) * cellWidth;
  let protectionWidth = select(compactProtectionWidth, retainedProtectionWidth,
    fineSummaryFactor == 1u);
  let crossesInterface = summary.minimumPhi <= 0.0 && summary.maximumPhi >= 0.0;
  // A sign crossing is positive refinement evidence even when the narrow-band
  // publication does not fill the candidate leaf's entire volume. Requiring a
  // complete size-8/16 summary here strands factor-1 surface bricks inside the
  // coarse leaf and prevents the later per-level passes from ever seeing them.
  if (crossesInterface) { return true; }
  // A size-two adaptive pressure row can represent the factor-4/8 free-surface
  // cut directly. Splitting every merely-near row to unit size inflated the
  // first recurring mini-dam frontier from 1,248 to 1,500 rows and exhausted
  // the solve tail, damping the bottom-front expansion from step two onward.
  if (fineSummaryFactor != 1u && size <= 2u) {
    return false;
  }
  if (summary.coarseAuthority) { return false; }
  let observedNearInterface = summary.minimumAbsolutePhi <= protectionWidth;
  // Factor 4/8 can use the merged corrected-coarse interval to prove complete
  // distance evidence. Factor 1 deliberately publishes a fine-only hierarchy
  // because size-1/2/4 coarse rows collide on a B4 key; an observed finite
  // near-interface sample is still safe positive evidence. Incomplete absence
  // remains false and therefore never invents refinement away from the band.
  return observedNearInterface && (summary.complete || fineSummaryFactor == 1u);
}

fn pressureRetentionAt(origin: vec3u) -> u32 {
  let tileSize = topologyTileSize();
  let td = (dims() + vec3u(tileSize - 1u)) / tileSize;
  let tile = min(origin / tileSize, td - vec3u(1u));
  let tileIndex = tile.x + td.x * (tile.y + td.y * tile.z);
  if (tileIndex >= topologyTileCapacity()) { return 0u; }
  let state = compaction[tileSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex + 4u];
  return select(0u, state >> 24u,
    (state & TILE_SIGNATURE_VALID_MASK) == TILE_SIGNATURE_VALID_MAGIC);
}

// The recurring summary/coarse hierarchies already carry a conservative
// minimum over this candidate. Only bootstrap lacks that compact authority,
// so it pays the exact cell scan once while the imported/analytic phi is live.
fn boundaryLiquidMinimumPhi(origin: vec3u, size: u32, bootstrapMinimum: f32) -> f32 {
  if (bootstrapPhiEnabled()) { return bootstrapMinimum; }
  let summary = fineLeafSummary(origin, size);
  if (summary.found) { return summary.minimumPhi; }
  let centre = vec3f(origin) + vec3f(0.5 * f32(size));
  let coarse = correctedCoarsePhi(centre);
  if (coarse.authority) {
    // A leafSize-zero result is the authoritative positive-air complement of
    // the sparse liquid/interface directory, not a measured distance sample.
    // Its nominal value is one maximum-leaf width (0.10 m for mini16), which
    // lies inside the authored three-cell look-ahead (0.15 m) and formerly
    // made every dry wall/ceiling strip refine forever after bootstrap. A
    // missing sparse row instead proves this candidate is outside the active
    // fluid band; when liquid approaches, the fine summary becomes present
    // and the branch above supplies its conservative minimum before contact.
    return select(coarse.minimumPhi, 3.402823e38, coarse.leafSize == 0u);
  }
  return phi(vec3i(min(origin + vec3u(size / 2u), dims() - vec3u(1u))));
}

fn leafNeedsRefinement(origin: vec3u, size: u32) -> bool {
  // Current spatial pressure evidence always wins. Temporal retention is
  // considered only after classifying the candidate: it may preserve an
  // interior pressure shell, but must not let unrelated evidence elsewhere in
  // the same 8-cubed tile pin a locally dry wall or terrain crossing.
  if (pressureRefinementEvidence(origin, size)) { return true; }
  let pressureRetained = pressureRetentionAt(origin) > 0u
    && fineSummaryFactor == 1u;
  let adaptivity = f32(params.control.x) / 1000.0;
  if (adaptivity <= 0.0) { return true; }
  let crossesClosedWall = powerClosedWallStripIntersects(origin, size);
  // Empty/open mini-dam scenes bind only a format-valid solid sentinel. Fine
  // interface and inflow protection have already been resolved above, so the
  // remaining expensive predicate is solid-only.
  if (!denseSolidField && !crossesClosedWall) { return pressureRetained; }
  var minimumSolid = 1.0; var maximumSolid = 0.0;
  var minimumPhi = 3.402823e38;
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q = origin + vec3u(x,y,z);
    if (denseSolidField) {
      let solid = solidAt(vec3i(q)).fraction;
      minimumSolid = min(minimumSolid, solid); maximumSolid = max(maximumSolid, solid);
    }
    if (fluidGatedBoundaryRefinement && bootstrapPhiEnabled()) {
      minimumPhi = min(minimumPhi, phi(vec3i(q)));
    }
  } } }
  let crossesSolidBoundary = maximumSolid - minimumSolid > 1e-5 || (maximumSolid > 1e-5 && maximumSolid < 1.0 - 1e-5);
  let crossesBoundary = crossesClosedWall || (denseSolidField && crossesSolidBoundary);
  if (crossesBoundary) {
    if (!fluidGatedBoundaryRefinement) { return true; }
    minimumPhi = boundaryLiquidMinimumPhi(origin, size, minimumPhi);
    return minimumPhi <= params.solve.w * params.cellRelax.x;
  }
  if (pressureRetained) { return true; }
  if (minimumSolid >= 1.0 - 1e-5) { return false; }
  return false;
}

// Claim the split of a leaf by publishing its own origin cell, which is the
// first cell splitLeaf would write anyway, and report whether this invocation
// was the one that lowered it.
//
// Grading is a neighbour repair: every leaf on the ring around a coarse
// neighbour asks for the SAME neighbour split, and each asker then writes the
// identical size-cubed owner partition serially in one lane. The writes are
// idempotent atomicMin, so the duplicates never change the published topology
// -- they only multiply a 32-cubed materialization by the ring population and
// pile every copy onto the same words. Deduplicating on the origin cell keeps
// the published state identical (the winner performs every write the losers
// would have) while making the cost proportional to splits rather than to
// askers.
//
// A missing owner page is answered by materializing, not claiming: that path
// already latches the rejection flag inside storeOwnerRequired, and the loop
// must keep visiting the pages that do exist exactly as before.
fn claimLeafSplit(origin: vec3u, size: u32) -> bool {
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = origin / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = requireOwnerPageEncoded(logical); if (encoded == 0u) { return true; }
  let local = origin % vec3u(8u);
  let at = ownerPageMap().payloadBase + (encoded - 1u) * 512u
    + local.x + local.y * 8u + local.z * 64u;
  let membership = atomicLoad(&owners[at]) & OWNER_WORD_TOPOLOGY;
  let word = encodePagedOwner(origin, origin, size / 2u) | membership;
  return atomicMin(&owners[at], word) > word;
}

// Materialize a split, one owner page at a time.
//
// This is the same write set storeOwnerRequired produced cell by cell, in the
// same page-local order, with the page lookup lifted out of the inner loop.
// A leaf is dyadic and its origin is size-aligned, so it either covers whole
// 8-cubed pages or lies inside one -- either way the directory only has to be
// consulted once per page instead of once per cell, which is 512 fewer
// dependent device loads on the serial chain of a size-32 split. The
// rejection latch and the absent-page skip keep the exact behaviour of
// storeOwnerRequired, which likewise only tests for a zero page index.
// Materialize a claimed split across lanes cooperating invocations.
//
// lanes == 1 reproduces the original serial walk exactly, term for term and
// page for page; it stays the definition of the result. Wider lane counts
// divide the SAME write set, and dividing it is observationally free: the write
// is atomicMin against a value that depends only on (origin, size, cell), so
// it is commutative and idempotent and no lane can see another's order. The
// claim is deliberately NOT here -- claimLeafSplit elects one materializer per
// split before this is ever reached, so a fanned-out call cannot duplicate work
// that the serial one deduplicated.
//
// The division is over PAGES, and the per-page body is byte-for-byte the serial
// one. That matters more than it looks: an earlier revision flattened the inner
// triple loop so it could stride cells as well as pages, which put three
// integer div/mods on every one of a size-32 split's 32,768 cells. At lanes == 1
// that alone measured +6.05 ms/advance on droplet-256 (86.97 -> 93.02) -- a
// direct measurement of how ALU-sensitive this loop is. Striding pages needs no
// arithmetic the serial walk did not already do.
//
// A size-32 leaf is 4x4x4 = 64 pages and puts exactly one lane on each; size 16
// is 8 pages over 8 lanes. Smaller leaves are one page and stay serial, which is
// the right trade -- they are at most 512 cells against the size-32 case's
// 32,768, and those big coarse-neighbour splits are what the profile is made of.
fn materializeSplitStrided(origin: vec3u, size: u32, lane: u32, lanes: u32) {
  let child = size / 2u;
  let payloadBase = ownerPageMap().payloadBase;
  let brickDims = (dims() + vec3u(7u)) / 8u;
  let first = origin / 8u;
  let last = (origin + vec3u(size - 1u)) / 8u;
  let shape = last - first + vec3u(1u);
  let pages = shape.x * shape.y * shape.z;
  for (var page = lane; page < pages; page += lanes) {
    let brick = first + vec3u(page % shape.x, (page / shape.x) % shape.y,
      page / (shape.x * shape.y));
    let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
    let encoded = requireOwnerPageEncoded(logical);
    if (encoded == 0u) { continue; }
    let base = payloadBase + (encoded - 1u) * 512u;
    let brickOrigin = brick * vec3u(8u);
    let lo = max(brickOrigin, origin) - brickOrigin;
    let hi = min(brickOrigin + vec3u(8u), origin + vec3u(size)) - brickOrigin;
    for (var lz = lo.z; lz < hi.z; lz += 1u) {
      for (var ly = lo.y; ly < hi.y; ly += 1u) {
        for (var lx = lo.x; lx < hi.x; lx += 1u) {
          let local = vec3u(lx, ly, lz);
          let cell = brickOrigin + local;
          let childOrigin = origin + ((cell - origin) / vec3u(child)) * vec3u(child);
          let at = base + local.x + local.y * 8u + local.z * 64u;
          let membership = atomicLoad(&owners[at]) & OWNER_WORD_TOPOLOGY;
          atomicMin(&owners[at], encodePagedOwner(cell, childOrigin, child) | membership);
        }
      }
    }
  }
}

// --- Page-claimed split materialization ------------------------------------
//
// One page of a split: where its 512 owner words live, and the single word
// every one of them receives.
struct SplitPage { base: u32, word: u32 }

// Resolve one page of the split rooted at (origin, size).
//
// Defined for size >= 16, where the child is 8 or larger. The leaf is
// size-aligned and the page is 8-aligned, so the page lies wholly inside ONE
// child and its 512 cells share a SINGLE owner word: encodePagedOwner keys the
// origin delta off the CELL's brick origin, which is page-invariant too. The
// per-cell loop this replaces recomputed that constant 512 times through three
// runtime integer divisions by child plus a firstTrailingBit -- the exact class
// of arithmetic an earlier revision measured at +6.05 ms/advance for three
// div/mods on this very loop.
//
// base == 0 is the absent-page sentinel: the payload can never start at word
// zero because the arena header and the page directory precede it.
fn splitPageAt(origin: vec3u, size: u32, page: u32) -> SplitPage {
  let child = size / 2u;
  let span = size / 8u;
  let first = origin / 8u;
  let brick = first + vec3u(page % span, (page / span) % span, page / (span * span));
  let brickDims = (dims() + vec3u(7u)) / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = requireOwnerPageEncoded(logical);
  if (encoded == 0u) { return SplitPage(0u, 0u); }
  let brickOrigin = brick * vec3u(8u);
  let childOrigin = origin + ((brickOrigin - origin) & vec3u(~(child - 1u)));
  return SplitPage(ownerPageMap().payloadBase + (encoded - 1u) * 512u,
    encodePagedOwner(brickOrigin, childOrigin, child));
}

// Membership is not readable state here, so the fill does not read it.
//
// storeOwnerRequired preserves OWNER_WORD_TOPOLOGY by loading the current word
// and OR-ing the bit back into the atomicMin candidate. That load is the second
// device round trip on a dependent chain -- half of the traffic of a size-cubed
// materialization -- and inside the topology candidate view it can never
// observe a set bit. Membership is a leaf property published by
// markAcceptedOwner during frontier emission, and commitOwnerPageCandidate
// rewrites the whole inactive payload bank with word &= ~OWNER_WORD_TOPOLOGY
// (webgpu-octree-owner-pages.ts, "Membership is a leaf property, not a
// resident-page property") in the pass immediately before the topology
// dispatches. Every page reachable through the candidate directory is in that
// candidate key set by construction, so the bit is clear on every cell this
// path can address, and OR-ing zero is a no-op.
//
// The guard is the override, not a comment: outside the candidate view the
// loading form is kept verbatim.
fn splitOwnerWord(at: u32, word: u32) -> u32 {
  if (topologyCandidateView == 1u && !gradingMembershipLoad) { return word; }
  return word | (atomicLoad(&owners[at]) & OWNER_WORD_TOPOLOGY);
}

// Claim a page the same way the split itself is claimed: write the first cell
// the fill would write anyway and report whether this invocation lowered it.
// A loser retires having spent exactly the three device ops it already spent,
// and it knows the page has a materializer.
fn claimSplitPage(plan: SplitPage) -> bool {
  let word = splitOwnerWord(plan.base, plan.word);
  return atomicMin(&owners[plan.base], word) > word;
}

// The remaining 511 cells, contiguous and in the same x-fastest order the
// triple loop used. Slot zero belongs to the claim.
fn fillSplitPage(plan: SplitPage) {
  for (var slot = 1u; slot < 512u; slot += 1u) {
    let at = plan.base + slot;
    atomicMin(&owners[at], splitOwnerWord(at, plan.word));
  }
}

// The elected materializer sweeps every page, so coverage never depends on how
// many helpers showed up. Pages a helper already claimed cost three device ops
// to skip.
fn materializeSplitPages(origin: vec3u, size: u32) {
  let span = size / 8u;
  let pages = span * span * span;
  for (var page = 0u; page < pages; page += 1u) {
    let plan = splitPageAt(origin, size, page);
    if (plan.base == 0u) { continue; }
    // Page zero's first cell IS the leaf origin and its word is byte-identical
    // to the leaf claim's, so re-claiming it would always lose and would strand
    // the 511 cells behind it.
    if (page == 0u || claimSplitPage(plan)) { fillSplitPage(plan); }
  }
}

// Give a losing asker exactly one page.
//
// Grading is a neighbour repair: every leaf on the ring around a coarse
// neighbour asks for the SAME neighbour split. Deduplicating on the origin cell
// made the cost proportional to splits rather than askers, but it also left ONE
// lane walking the whole size-cubed partition -- 32,768 dependent atomics for a
// size-32 leaf -- while every other asker retired after three device ops and 63
// of its own workgroup siblings idled.
//
// The partition divides over pages and the write is an idempotent atomicMin of
// a value depending only on (origin, size, cell), so any lane may perform any
// page and no lane can observe another's order. Each loser therefore takes one
// page, chosen by hashing its own anchor so that askers spread over the pages
// instead of colliding on the first one. Nothing is shared, nothing blocks, and
// the loser's cost is unchanged unless it actually wins work -- which is the
// property a workgroup-local queue drained behind a barrier could not have.
fn helpSplitPage(origin: vec3u, size: u32, seed: u32) {
  let span = size / 8u;
  let pages = span * span * span;
  var mixed = seed * 0x9e3779b1u;
  mixed = mixed ^ (mixed >> 16u);
  let plan = splitPageAt(origin, size, mixed & (pages - 1u));
  if (plan.base == 0u) { return; }
  if (claimSplitPage(plan)) { fillSplitPage(plan); }
}

fn splitLeafSeeded(origin: vec3u, size: u32, seed: u32) {
  let claimed = claimLeafSplit(origin, size);
  if (gradingPageFill && size >= 16u) {
    if (claimed) { materializeSplitPages(origin, size); }
    else if (gradingSplitHelpers) { helpSplitPage(origin, size, seed); }
    return;
  }
  if (!claimed) { return; }
  materializeSplitStrided(origin, size, 0u, 1u);
}

fn splitLeaf(origin: vec3u, size: u32) { splitLeafSeeded(origin, size, 0u); }

fn refineTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(vec3i(gid));
  if (ownerValid(owner) && owner.size > 1u && (targetRefinementSize == 0u || owner.size == targetRefinementSize) && isOrigin(gid, owner) && leafNeedsRefinement(gid, owner.size)) { splitLeaf(gid, owner.size); }
}

@compute @workgroup_size(4,4,4)
fn refineTopology(@builtin(global_invocation_id) gid: vec3u) { refineTopologyAt(gid * 2u); }

@compute @workgroup_size(4,4,4)
fn refineTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  refineTopologyAt(deltaTopologyCandidate(wid, lid));
}

// Large leaves are deliberately rare. One 128-lane workgroup evaluates the
// exact scalar sizing predicate once, then publishes child owners
// cooperatively. The predicate's cubic scan remains runtime-bounded below so
// browser Metal compilers cannot specialize 16^3/32^3 into a giant kernel.
var<workgroup> refineEligible: atomic<u32>;
var<workgroup> refineDecision: atomic<u32>;
var<workgroup> refineRuntimeSize: atomic<u32>;
// min/max solid fraction plus minimum liquid phi. A vec4 retains the same
// naturally aligned reduction shape on every backend.
var<workgroup> refineBoundaryRange: array<vec4f, 128>;

@compute @workgroup_size(128)
fn refineTopologyCoarse(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  refineCoarseBlock(wid * vec3u(targetRefinementSize), lid);
}

@compute @workgroup_size(128)
fn refineTopologyCoarseDelta(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let tile = deltaTileOrigin(wid.x);
  let blocks = max(1u, topologyTileSize() / targetRefinementSize);
  for (var sub = 0u; sub < blocks * blocks * blocks; sub += 1u) {
    let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
    refineCoarseBlock(tile + subCoord * vec3u(targetRefinementSize), lid);
  }
}

fn refineCoarseBlock(origin: vec3u, lid: u32) {
  // Eligibility is scalar; the potentially size^3 solid predicate is not.
  // All lanes cooperatively reduce the leaf's solid range.
  if (lid == 0u) {
    let inBounds = all(origin < dims());
    let owner = ownerAt(vec3i(min(origin, dims() - vec3u(1u))));
    let eligible = inBounds && ownerValid(owner)
      && owner.size == targetRefinementSize && isOrigin(origin, owner);
    atomicStore(&refineEligible, select(0u, 1u, eligible));
    atomicStore(&refineDecision, 0u);
    // Preserve the storage-loaded size across the barrier. Using the pipeline
    // override as the cubic loop bound lets some browser Metal compilers fully
    // specialize size^3 at 16/32 and produce a watchdog-scale kernel.
    atomicStore(&refineRuntimeSize, max(1u, owner.size));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&refineEligible) == 0u) { return; }
  let size = workgroupUniformLoad(&refineRuntimeSize);
  var boundaryRange = vec4f(1.0, 0.0, 3.402823e38, 0.0);
  let crossesClosedWall = powerClosedWallStripIntersects(origin, size);
  if (denseSolidField
      || (fluidGatedBoundaryRefinement && bootstrapPhiEnabled() && crossesClosedWall)) {
    let solidCellsInLeaf = size * size * size;
    for (var flat = lid; flat < solidCellsInLeaf; flat += 128u) {
      let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
      let q = origin + local;
      if (denseSolidField) {
        let solid = solidAt(vec3i(q)).fraction;
        boundaryRange = vec4f(
          min(boundaryRange.x, solid), max(boundaryRange.y, solid),
          boundaryRange.z, 0.0);
      }
      if (fluidGatedBoundaryRefinement && bootstrapPhiEnabled()) {
        boundaryRange.z = min(boundaryRange.z, phi(vec3i(q)));
      }
    }
  }
  refineBoundaryRange[lid] = boundaryRange;
  for (var stride = 64u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) {
      let right = refineBoundaryRange[lid + stride];
      refineBoundaryRange[lid] = vec4f(
        min(refineBoundaryRange[lid].x, right.x),
        max(refineBoundaryRange[lid].y, right.y),
        min(refineBoundaryRange[lid].z, right.z),
        0.0,
      );
    }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let range = refineBoundaryRange[0];
    let adaptivity = f32(params.control.x) / 1000.0;
    let crossesSolid = range.y - range.x > 1e-5
      || (range.y > 1e-5 && range.y < 1.0 - 1e-5);
    let crossesBoundary = crossesClosedWall || (denseSolidField && crossesSolid);
    var boundaryDecision = crossesBoundary;
    if (fluidGatedBoundaryRefinement && crossesBoundary) {
      boundaryDecision = boundaryLiquidMinimumPhi(origin, size, range.z)
        <= params.solve.w * params.cellRelax.x;
    }
    let pressureEvidence = pressureRefinementEvidence(origin, size);
    let pressureRetained = pressureRetentionAt(origin) > 0u
      && fineSummaryFactor == 1u;
    // Preserve the production pressure hysteresis everywhere except a locally
    // dry boundary crossing. Current spatial evidence still overrides the gate
    // and splits before liquid contact.
    let pressureDecision = pressureEvidence
      || (pressureRetained && (!fluidGatedBoundaryRefinement || !crossesBoundary));
    let decision = pressureDecision || adaptivity <= 0.0 || boundaryDecision;
    atomicStore(&refineDecision, select(0u, 1u, decision));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&refineDecision) == 0u) { return; }
  let cells = size * size * size;
  let child = size / 2u;
  requireLeafOwnerPages(origin, size, lid, 128u);
  workgroupBarrier();
  for (var flat = lid; flat < cells; flat += 128u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerRequired(origin + local, childOrigin, child);
  }
}

fn ownerAtIsTooFine(p: vec3i, size: u32) -> bool {
  if (!valid(p)) { return false; }
  let neighbor = ownerAt(p);
  return ownerValid(neighbor) && neighbor.size * 2u < size;
}
fn neighborTooFine(origin: vec3u, size: u32) -> bool {
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) {
    let q0 = vec3i(origin + vec3u(0u,y,z)); let q1 = vec3i(origin + vec3u(size-1u,y,z));
    if (ownerAtIsTooFine(q0-vec3i(1,0,0), size) || ownerAtIsTooFine(q1+vec3i(1,0,0), size)) { return true; }
  } }
  for (var z = 0u; z < size; z += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q0 = vec3i(origin + vec3u(x,0u,z)); let q1 = vec3i(origin + vec3u(x,size-1u,z));
    if (ownerAtIsTooFine(q0-vec3i(0,1,0), size) || ownerAtIsTooFine(q1+vec3i(0,1,0), size)) { return true; }
  } }
  for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q0 = vec3i(origin + vec3u(x,y,0u)); let q1 = vec3i(origin + vec3u(x,y,size-1u));
    if (ownerAtIsTooFine(q0-vec3i(0,0,1), size) || ownerAtIsTooFine(q1+vec3i(0,0,1), size)) { return true; }
  } }
  return false;
}

const PAPER_DIRECTIONS: array<vec3i,18> = array<vec3i,18>(
  vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0),
  vec3i(-1,-1,0),vec3i(-1,0,-1),vec3i(-1,0,1),vec3i(-1,1,0),vec3i(0,-1,-1),vec3i(0,-1,1),
  vec3i(0,1,-1),vec3i(0,1,1),vec3i(1,-1,0),vec3i(1,0,-1),vec3i(1,0,1),vec3i(1,1,0));
fn paperProbe(origin: vec3u, size: u32, direction: vec3i) -> vec3i {
  var probe = vec3i(0);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    probe[axis] = select(select(i32(origin[axis] + size / 2u), i32(origin[axis] + size), direction[axis] > 0),
      i32(origin[axis]) - 1, direction[axis] < 0);
  }
  return probe;
}
fn repairPaperMixedNeighbors(origin: vec3u, size: u32) {
  var finer = false; var coarser = false;
  for (var bit = 0u; bit < 18u; bit += 1u) {
    let probe = paperProbe(origin, size, PAPER_DIRECTIONS[bit]); if (!valid(probe)) { continue; }
    let neighbor = ownerAt(probe);if(!ownerValid(neighbor)){continue;}
    let neighborSize = neighbor.size; finer = finer || neighborSize < size; coarser = coarser || neighborSize > size;
  }
  if (!finer || !coarser) { return; }
  // This is the exact deterministic rule in plan section 7.3 and the CPU
  // oracle: split every coarse face/edge neighbor of the mixed anchor once.
  for (var bit = 0u; bit < 18u; bit += 1u) {
    let probe = paperProbe(origin, size, PAPER_DIRECTIONS[bit]); if (!valid(probe)) { continue; }
    let neighbor = ownerAt(probe); if (ownerValid(neighbor) && neighbor.size > size) { splitLeafSeeded(unpackOrigin(neighbor.packedOrigin), neighbor.size, packOrigin(origin) + bit); }
  }
}

fn repairPaperRatioNeighbors(origin: vec3u, size: u32) {
  for (var bit = 0u; bit < 18u; bit += 1u) {
    let probe = paperProbe(origin, size, PAPER_DIRECTIONS[bit]); if (!valid(probe)) { continue; }
    let neighbor = ownerAt(probe);
    if (ownerValid(neighbor) && neighbor.size > 2u * size) {
      splitLeafSeeded(unpackOrigin(neighbor.packedOrigin), neighbor.size, packOrigin(origin) + bit);
    }
  }
}

fn balanceTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(vec3i(gid));
  if (!ownerValid(owner)) { return; }
  if (owner.size <= 16u && isOrigin(gid, owner)) { repairPaperRatioNeighbors(gid, owner.size); }
  if (owner.size >= 2u && owner.size <= 16u && isOrigin(gid, owner)) { repairPaperMixedNeighbors(gid, owner.size); }
  // Size-16+ leaves use the cooperative entry point below.
  if (owner.size > 2u && owner.size < 16u && isOrigin(gid, owner) && neighborTooFine(gid, owner.size)) { splitLeaf(gid, owner.size); }
}

@compute @workgroup_size(4,4,4)
fn balanceTopology(@builtin(global_invocation_id) gid: vec3u) {
  let base = gid * 2u;
  for (var parity = 0u; parity < 8u; parity += 1u) {
    balanceTopologyAt(base + vec3u(parity & 1u, (parity >> 1u) & 1u, (parity >> 2u) & 1u));
  }
}

@compute @workgroup_size(4,4,4)
fn balanceTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  if (topologyStructurallyQuiescent()) { return; }
  let base = deltaTopologyCandidate(wid, lid);
  for (var parity = 0u; parity < 8u; parity += 1u) {
    balanceTopologyAt(base + vec3u(parity & 1u, (parity >> 1u) & 1u, (parity >> 2u) & 1u));
  }
}

var<workgroup> balanceEligible: atomic<u32>;
var<workgroup> balanceRuntimeSize: atomic<u32>;
var<workgroup> balanceFlags: array<u32, 256>;

@compute @workgroup_size(256)
fn balanceTopologyCoarse(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  balanceCoarseBlock(wid * vec3u(targetRefinementSize), lid);
}

@compute @workgroup_size(256)
fn balanceTopologyCoarseDelta(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let tile = deltaTileOrigin(wid.x);
  let blocks = max(1u, topologyTileSize() / targetRefinementSize);
  for (var sub = 0u; sub < blocks * blocks * blocks; sub += 1u) {
    let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
    balanceCoarseBlock(tile + subCoord * vec3u(targetRefinementSize), lid);
  }
}

fn balanceCoarseBlock(origin: vec3u, lid: u32) {
  // See refineCoarseBlock: bounds rejection flows through the
  // lane-0 eligibility store to keep barrier control flow formally uniform.
  if (lid == 0u) {
    let inBounds = all(origin < dims()) && !topologyStructurallyQuiescent();
    let owner = ownerAt(vec3i(min(origin, dims() - vec3u(1u))));
    atomicStore(&balanceEligible, select(0u, 1u, inBounds && ownerValid(owner)
      && owner.size == targetRefinementSize && isOrigin(origin, owner)));
    atomicStore(&balanceRuntimeSize, max(1u, owner.size));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&balanceEligible) == 0u) { return; }
  // Keep size-dependent loops dynamic for the same browser-Metal reason as
  // refineCoarseBlock; targetRefinementSize remains only an eligibility key.
  let size = workgroupUniformLoad(&balanceRuntimeSize);
  var needsSplit = 0u;
  let faceSamples = size * size;
  for (var sample = lid; sample < 6u * faceSamples; sample += 256u) {
    let face = sample / faceSamples;
    let axis = face / 2u;
    let positive = (face & 1u) == 1u;
    let within = sample % faceSamples;
    let a = within % size;
    let b = within / size;
    var local = vec3u(0u);
    local[axis] = select(0u, size - 1u, positive);
    local[(axis + 1u) % 3u] = a;
    local[(axis + 2u) % 3u] = b;
    let outside = vec3i(origin + local) + select(-1, 1, positive) * axisVector(axis);
    if (ownerAtIsTooFine(outside, size)) { needsSplit = 1u; }
  }
  balanceFlags[lid] = needsSplit;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) { balanceFlags[lid] = max(balanceFlags[lid], balanceFlags[lid + stride]); }
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&balanceFlags[0]) == 0u) { return; }
  let cells = size * size * size;
  let child = size / 2u;
  requireLeafOwnerPages(origin, size, lid, 256u);
  workgroupBarrier();
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerRequired(origin + local, childOrigin, child);
  }
}

// --- Compact liquid-frontier publication ----------------------------------

// The leaf frontier is an immutable sorted A/B publication. Recurring
// generations emit fixed dirty-tile candidate records, sort that bounded
// stream, and merge it with clean rows from the previous publication. No
// claim table, tombstone, append counter, or whole-active-list branch exists.
@compute @workgroup_size(1)
fn stampFrontierAttempt() {
  // frontier[8] is the last attempted generation, including rejected
  // attempts. Preserve a non-zero, wrap-safe monotonic clock without any
  // host-visible readback or shared-uniform mutation.
  var next = frontier[8] + 1u;
  if (next == 0u) { next = 1u; }
  frontier[8] = next;
}

@compute @workgroup_size(1)
fn beginFrontier() {
  let current = frontierCurrent();
  frontier[4] = 0u;
  frontier[5] = 0u;
  // A tail builder owns exactly one inactive transaction.  Clearing ready
  // here cannot affect the active selector/generation consumed by this
  // substep; only the coupled owner/frontier commit changes those words.
  frontier[6] = 0u;
  frontier[9] = 0u;
  let control = pressureControlBase();
  compaction[control] = 0u;
  compaction[control + 1u] = 0u;
  compaction[control + 2u] = 0u;
  let failed = compaction[dirtyAuthorityBase()] == FRONTIER_FAILED_MAGIC;
  let reuse = compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC
    && compaction[0] == 0u && compaction[4] == 0u;
  frontier[7] = 1u - current;
  let blocks = select((frontierCount(current) + 255u) / 256u, 0u, reuse || failed);
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  compaction[12] = x; compaction[13] = y; compaction[14] = 1u;
  // A malformed structural transaction never advances the immutable frontier
  // selector. Downstream scan/emit observes the failure magic and restores
  // the last complete row-control publication.
  compaction[11] = select(
    select(0u, FRONTIER_REUSE_MAGIC, reuse),
    FRONTIER_FAILED_MAGIC,
    failed,
  );
  compaction[frontierTopologyReuseBase()] = 0u;
  if (reuse) {
    // Geometry is unchanged, but every downstream publication still advances
    // one power generation. Publish the exact identity row delta for that
    // generation instead of leaving consumers on the previous control epoch.
    let count = frontierCount(current);
    let base = rowDeltaControlBase();
    frontier[base] = count; frontier[base + 1u] = count;
    frontier[base + 2u] = count; frontier[base + 3u] = 0u;
    frontier[base + 4u] = 0u; frontier[base + 5u] = 0u;
    frontier[base + 6u] = 0u; frontier[base + 7u] = frontier[8];
    frontier[base + 8u] = 0x52444c54u;
    frontier[base + 9u] = 0u; frontier[base + 10u] = 1u;
    frontier[base + 11u] = 1u; frontier[base + 12u] = 0u;
    frontier[base + 13u] = 1u; frontier[base + 14u] = 1u;
    frontier[base + 15u] = 1u;
    frontier[7] = current;
    frontier[6] = 1u;
  }
}

fn currentPressureOwnerWet(owner: Owner) -> bool {
  let origin=unpackOrigin(owner.packedOrigin);let fine=fineLeafSummary(origin,owner.size);
  var wet=liquidOwner(owner);
  // Neither bootstrap authority may be second-guessed by a fine summary that
  // does not exist yet at t=0.
  if(bootstrapPhiEnabled()){return wet;}
  if(fine.found){
    if(fine.exactCellValid){wet=fine.exactCellNegative;}
    else if(fine.centerValid){wet=fine.centerPhi<0.0;}
    // A coarse-only summary is the paper's separate octree level set, not a
    // license to reclassify the same cell through a second surface authority.
    // Keep liquidOwner's exact coarse-centre decision so frontier membership
    // and the power-boundary ghost-fluid sample consume one generation and
    // one authority.  Only a complete fine interval may refine that decision.
    else if(fine.complete&&!fine.coarseAuthority){
      if(fine.maximumPhi<0.0){wet=true;}
      else if(fine.minimumPhi>=0.0){wet=false;}
      else{let centre=vec3f(origin)+vec3f(0.5*f32(owner.size-1u));wet=samplePhiPoint(centre)<0.0;}
    }
  }
  return wet;
}

fn previousFrontierHasExactIdentity(cell:u32,size:u32)->bool{
  let previous=frontierCurrent();let previousCount=frontierCount(previous);
  let old=previousLowerBound(cell,size,previous,previousCount);
  return old<previousCount&&frontierCell(previous,old)==cell
    &&old<arrayLength(&leafHeaders)&&leafHeaders[old].cell==cell
    &&leafHeaders[old].size==size;
}

fn frontierCandidateAt(gid:vec3u,additionsOnly:bool)->u32{
  if(any(gid>=dims())){return 0xffffffffu;}
  let cell = index(gid);
  let owner = ownerAtIndex(cell);
  if(!isOrigin(gid,owner)||!currentPressureOwnerWet(owner)){return 0xffffffffu;}
  // A recurring dirty tile usually republishes the same wet leaves. Preserve
  // those identities in the already-sorted previous frontier and sort only
  // genuine additions. The carried row remains marked affected below, so
  // value/operator consumers still recompute it from the current generation.
  if(additionsOnly&&previousFrontierHasExactIdentity(cell,owner.size)){
    return 0xffffffffu;
  }
  return cell;
}

var<workgroup> frontierCandidateScan:array<u32,64>;
fn candidateBlockIndex(wid:vec3u,deltaMode:bool)->u32{
  let tileBlocks=max(1u,topologyTileSize()/8u);let perTile=tileBlocks*tileBlocks*tileBlocks;
  if(deltaMode){return wid.x+wid.y*compaction[8];}
  let denseBlocks=(dims()+vec3u(7u))/8u;
  return wid.x+denseBlocks.x*(wid.y+denseBlocks.y*wid.z);
}
fn candidateBlockCount(deltaMode:bool)->u32{
  let blocks=max(1u,topologyTileSize()/8u);let perTile=blocks*blocks*blocks;
  let dense=(dims()+vec3u(7u))/8u;
  return select(dense.x*dense.y*dense.z,(compaction[0]+compaction[4])*perTile,deltaMode);
}
fn candidateLaneBase(wid:vec3u,lid:vec3u,deltaMode:bool)->vec3u{
  if(deltaMode){return deltaTopologyCandidate(wid,lid);}
  return wid*8u+lid*2u;
}
fn classifyFrontierCandidateBlock(wid:vec3u,lid:vec3u,deltaMode:bool){
  let local=lid.x+4u*(lid.y+4u*lid.z);
  let base=candidateLaneBase(wid,lid,deltaMode);var count=0u;
  // Each lane owns an exact 2^3 sub-block. Odd coordinates matter because
  // interface refinement legitimately publishes size-one leaves.
  for(var octant=0u;octant<8u;octant+=1u){
    let offset=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u);
    var cell=0xffffffffu;
    if(all(base<dims())){cell=frontierCandidateAt(base+offset,deltaMode);}
    count+=select(0u,1u,cell!=0xffffffffu);
  }
  frontierCandidateScan[local]=count;
  for(var stride=32u;stride>0u;stride>>=1u){workgroupBarrier();
    if(local<stride){frontierCandidateScan[local]+=frontierCandidateScan[local+stride];}}
  workgroupBarrier();
  if(local==0u){let block=candidateBlockIndex(wid,deltaMode);
    compaction[candidateScanScratchBase()+2u*block]=frontierCandidateScan[0];}
}
@compute @workgroup_size(4,4,4)
fn classifyFrontierCandidates(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  classifyFrontierCandidateBlock(wid,lid,false);
}
@compute @workgroup_size(4,4,4)
fn classifyFrontierCandidatesDelta(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  classifyFrontierCandidateBlock(wid,lid,true);
}

fn prefixFrontierCandidateBlockStream(lid:u32,deltaMode:bool){
  let blocks=candidateBlockCount(deltaMode);let chunk=(blocks+255u)/256u;
  let begin=min(blocks,lid*chunk);let end=min(blocks,begin+chunk);var subtotal=0u;
  for(var block=begin;block<end;block+=1u){subtotal+=compaction[candidateScanScratchBase()+2u*block];}
  // Cooperative Hillis-Steele scan over the 256 lane subtotals.
  rowDeltaScan[lid]=subtotal;workgroupBarrier();
  for(var stride=1u;stride<256u;stride<<=1u){
    var add=0u;if(lid>=stride){add=rowDeltaScan[lid-stride];}
    workgroupBarrier();rowDeltaScan[lid]+=add;workgroupBarrier();
  }
  var cursor=rowDeltaScan[lid]-subtotal;
  for(var block=begin;block<end;block+=1u){let count=compaction[candidateScanScratchBase()+2u*block];
    compaction[candidateScanScratchBase()+2u*block+1u]=cursor;cursor+=count;}
  if(lid==255u){frontier[4]=rowDeltaScan[255u];}
}
@compute @workgroup_size(256)
fn prefixFrontierCandidateBlocks(@builtin(local_invocation_index)lid:u32){
  prefixFrontierCandidateBlockStream(lid,false);
}
@compute @workgroup_size(256)
fn prefixFrontierCandidateBlocksDelta(@builtin(local_invocation_index)lid:u32){
  prefixFrontierCandidateBlockStream(lid,true);
}

fn emitFrontierCandidateBlock(wid:vec3u,lid:vec3u,deltaMode:bool){
  let local=lid.x+4u*(lid.y+4u*lid.z);
  let base=candidateLaneBase(wid,lid,deltaMode);
  var laneCandidates:array<u32,8>;var laneCount=0u;
  for(var octant=0u;octant<8u;octant+=1u){
    let offset=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u);
    var cell=0xffffffffu;
    if(all(base<dims())){cell=frontierCandidateAt(base+offset,deltaMode);}
    laneCandidates[octant]=cell;
    laneCount+=select(0u,1u,cell!=0xffffffffu);
  }
  frontierCandidateScan[local]=laneCount;
  for(var stride=1u;stride<64u;stride<<=1u){workgroupBarrier();var add=0u;
    if(local>=stride){add=frontierCandidateScan[local-stride];}
    workgroupBarrier();frontierCandidateScan[local]+=add;}
  workgroupBarrier();
  let block=candidateBlockIndex(wid,deltaMode);
  let outputBase=compaction[candidateScanScratchBase()+2u*block+1u]
    +frontierCandidateScan[local]-laneCount;
  var rank=0u;
  for(var octant=0u;octant<8u;octant+=1u){
    let cell=laneCandidates[octant];if(cell==0xffffffffu){continue;}
    let output=outputBase+rank;rank+=1u;
    if(output<frontierListCapacity()){frontier[frontierCandidateBase()+output]=cell;}
  }
}
@compute @workgroup_size(4,4,4)
fn emitFrontierCandidates(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  emitFrontierCandidateBlock(wid,lid,false);
}
@compute @workgroup_size(4,4,4)
fn emitFrontierCandidatesDelta(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  emitFrontierCandidateBlock(wid,lid,true);
}

// Candidate emission is the first point where the exact dirty frontier size is
// known. Publish compact live schedules into header words whose topology uses
// are complete. One contiguous copy then feeds the three indirect consumers.
@compute @workgroup_size(1)
fn prepareFrontierDispatch() {
  let reused = compaction[11] == FRONTIER_REUSE_MAGIC;
  let failed = compaction[11] == FRONTIER_FAILED_MAGIC;
  let candidateBlocks = select(
    (min(frontier[4], frontierListCapacity()) + 255u) / 256u, 0u, reused || failed);
  // Reuse still needs one row-sized launch to publish identity old/new maps.
  let carryBlocks = select((frontierCount(frontierCurrent()) + 255u) / 256u, 0u, failed);
  let mergeBlocks = select(max(candidateBlocks, carryBlocks), 0u, reused || failed);
  compaction[1] = candidateBlocks; compaction[2] = 1u; compaction[3] = 1u;
  compaction[4] = carryBlocks; compaction[5] = 1u; compaction[6] = 1u;
  compaction[7] = mergeBlocks; compaction[8] = 1u; compaction[9] = 1u;
}

fn cellCoord(c: u32) -> vec3u {
  let nx = params.dimsMax.x; let ny = params.dimsMax.y;
  return vec3u(c % nx, (c / nx) % ny, c / (nx * ny));
}

fn mortonPart10(value:u32)->u32{
  var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;
  x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;
}
fn rowMorton(cell:u32)->u32{let p=cellCoord(cell);return mortonPart10(p.x)|(mortonPart10(p.y)<<1u)|(mortonPart10(p.z)<<2u);}
fn candidateSortLoad(index:u32,fromCandidate:bool)->u32{
  return select(frontierSortScratch[index],frontier[frontierCandidateBase()+index],fromCandidate);
}
fn candidateSortStore(index:u32,value:u32,toCandidate:bool){
  if(toCandidate){frontier[frontierCandidateBase()+index]=value;}
  else{frontierSortScratch[index]=value;}
}
const ROW_DELTA_VALID:u32=0x52444c54u;
const ROW_DELTA_AFFECTED:u32=0x80000000u;
const ROW_DELTA_STRUCTURAL:u32=0x40000000u;
fn rowDeltaMapOld(encoded:u32)->u32{
  let value=encoded&0x3fffffffu;
  return select(0xffffffffu,value-1u,value!=0u);
}
fn rowDeltaBlockCount()->u32{return (frontierListCapacity()+255u)/256u;}
// The topology frontier must be declared atomic because its earlier append
// phase performs contended claims. Row-delta scan scratch therefore lives in
// the plain-u32 compaction tail. Only the immutable public maps, compact lists,
// and sixteen-word transaction header remain in the frontier; every such store
// has one statically unique writer and is never used as synchronization or RMW.
fn rowDeltaScratchWords()->u32{return 2u*frontierListCapacity()+3u*rowDeltaBlockCount()+1u;}
fn rowDeltaFlagsBase()->u32{return changeStateBase()-rowDeltaScratchWords();}
fn rowDeltaPrefixBase()->u32{return rowDeltaFlagsBase()+frontierListCapacity();}
fn rowDeltaBlockTotalsBase()->u32{return rowDeltaPrefixBase()+frontierListCapacity();}
fn rowDeltaCarriedBlocksBase()->u32{return rowDeltaBlockTotalsBase()+rowDeltaBlockCount()+1u;}
fn rowSortKeyLess(cellA:u32,sizeA:u32,cellB:u32,sizeB:u32)->bool{
  if(sizeA==0u){return false;}if(sizeB==0u){return true;}
  let levelA=u32(firstTrailingBit(sizeA));let levelB=u32(firstTrailingBit(sizeB));
  let mortonA=rowMorton(cellA);let mortonB=rowMorton(cellB);
  return levelA<levelB||(levelA==levelB&&(mortonA<mortonB||(mortonA==mortonB&&cellA<cellB)));
}
fn frontierSortStageCount(count:u32)->u32{
  var stages=0u;var width=1u;
  while(width<count&&stages<31u){width*=2u;stages+=1u;}
  return stages;
}

// The mini/default pressure frontier is bounded by the complete 16^3 domain.
// Keep that immutable-capacity lane resident in one portable 16 KiB
// workgroup allocation and replace O(log rows) global dispatch barriers with
// workgroup barriers. Larger frontiers retain the parallel merge-sort entry
// point below.
const FRONTIER_LOCAL_SORT_CAPACITY:u32=4096u;
var<workgroup> frontierLocalSortCells:array<u32,4096>;
fn frontierLocalCellLess(left:u32,right:u32)->bool{
  if(left==0xffffffffu){return false;}
  if(right==0xffffffffu){return true;}
  return rowSortKeyLess(left,ownerAtIndex(left).size,right,ownerAtIndex(right).size);
}
@compute @workgroup_size(256)
fn sortFrontierCandidatesLocal(@builtin(local_invocation_index)lid:u32){
  if(lid==0u){
    // Borrow slot zero for the uniform count before the cooperative load. It
    // is overwritten with candidate zero after every lane snapshots the count,
    // keeping the complete 4096-record lane at the portable 16 KiB limit.
    frontierLocalSortCells[0u]=
      min(min(frontier[4],frontierListCapacity()),FRONTIER_LOCAL_SORT_CAPACITY);
  }
  workgroupBarrier();
  let count=workgroupUniformLoad(&frontierLocalSortCells[0u]);
  if(count<2u){return;}
  var span=1u;
  while(span<count){span<<=1u;}
  for(var slot=lid;slot<span;slot+=256u){
    frontierLocalSortCells[slot]=select(
      0xffffffffu,frontier[frontierCandidateBase()+slot],slot<count);
  }
  workgroupBarrier();
  for(var width=2u;width<=span;width<<=1u){
    for(var stride=width>>1u;stride>0u;stride>>=1u){
      for(var slot=lid;slot<span;slot+=256u){
        let other=slot^stride;
        if(other>slot){
          let left=frontierLocalSortCells[slot];
          let right=frontierLocalSortCells[other];
          let ascending=(slot&width)==0u;
          let swap=select(frontierLocalCellLess(left,right),
            frontierLocalCellLess(right,left),ascending);
          if(swap){
            frontierLocalSortCells[slot]=right;
            frontierLocalSortCells[other]=left;
          }
        }
      }
      workgroupBarrier();
    }
  }
  for(var slot=lid;slot<count;slot+=256u){
    frontier[frontierCandidateBase()+slot]=frontierLocalSortCells[slot];
  }
}

// One invocation owns each fixed record. The header clear is bounded to
// sixteen words and the row payload is overwritten exactly by later stages;
// no capacity-sized serial reset remains.
@compute @workgroup_size(256)
fn prepareRowDelta(@builtin(global_invocation_id)gid:vec3u){
  if(frontierGenerationReused()){
    let row=gid.x;let count=frontierCount(frontierCurrent());
    if(row<count){
      frontier[rowDeltaNewToOldBase()+row]=row+1u;
      frontier[rowDeltaOldToNewBase()+row]=row+1u;
      compaction[rowDeltaFlagsBase()+row]=0u;
    }
    return;
  }
  if(gid.x<16u){frontier[rowDeltaControlBase()+gid.x]=0u;}
  if(gid.x==0u){compaction[rowDeltaFlagsBase()]=0u;}
}

// Stable bottom-up merge sorting needs O(log rows) dispatch barriers rather
// than a data-sized singleton loop. Every source record independently computes
// its unique merge rank, preserving exact (level, Morton, cell) identity.
@compute @workgroup_size(256)
fn sortFrontierCandidates(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;let count=min(frontier[4],frontierListCapacity());
  if(any(dims()>vec3u(1024u))||count>arrayLength(&frontierSortScratch)){
    if(row==0u){compaction[pressureControlBase()]=4u;}return;
  }
  let stage=frontierSortParams.x;
  let stages=frontierSortStageCount(count);
  if(stage==stages){
    if((stages&1u)!=0u&&row<count){candidateSortStore(row,candidateSortLoad(row,false),true);}
    return;
  }
  if(stage>stages||row>=count){return;}
  let fromCandidate=(stage&1u)==0u;let width=1u<<stage;let span=2u*width;
  let runBase=(row/span)*span;let split=min(runBase+width,count);let runEnd=min(runBase+span,count);
  let cell=candidateSortLoad(row,fromCandidate);let size=ownerAtIndex(cell).size;
  var lo=runBase;var hi=split;var local=0u;
  if(row<split){
    lo=split;hi=runEnd;local=row-runBase;
    while(lo<hi){let mid=lo+(hi-lo)/2u;let other=candidateSortLoad(mid,fromCandidate);
      if(rowSortKeyLess(other,ownerAtIndex(other).size,cell,size)){lo=mid+1u;}else{hi=mid;}}
    candidateSortStore(runBase+local+(lo-split),cell,!fromCandidate);
  }else{
    lo=runBase;hi=split;local=row-split;
    while(lo<hi){let mid=lo+(hi-lo)/2u;let other=candidateSortLoad(mid,fromCandidate);
      if(!rowSortKeyLess(cell,size,other,ownerAtIndex(other).size)){lo=mid+1u;}else{hi=mid;}}
    candidateSortStore(runBase+local+(lo-runBase),cell,!fromCandidate);
  }
}

fn rowAuthorityFrontierDirtyGeneration(cell:u32,generation:u32)->bool{
  let origin=cellCoord(cell);let tileSize=topologyTileSize();
  let td=(dims()+vec3u(tileSize-1u))/tileSize;let tile=vec3i(origin/tileSize);
  let ownIndex=u32(tile.x)+td.x*(u32(tile.y)+td.y*u32(tile.z));
  return compaction[tileFrontierChangeFlagsBase()+ownIndex]==generation;
}
fn rowAuthorityStructuralDirtyGeneration(cell:u32,generation:u32)->bool{
  let origin=cellCoord(cell);let size=ownerAtIndex(cell).size;
  let tileSize=topologyTileSize();let td=(dims()+vec3u(tileSize-1u))/tileSize;
  // A descriptor reads exactly the anchor owner plus the paper's 18
  // face/edge owner probes. Test those authority tiles directly. The old
  // maximum-leaf cube admitted unrelated changes from as many as 27 tiles
  // and made the structural workset nearly indistinguishable from the wet
  // influence set.
  let ownTile=origin/tileSize;
  let ownIndex=ownTile.x+td.x*(ownTile.y+td.y*ownTile.z);
  if(compaction[tileChangeFlagsBase()+ownIndex]==generation){return true;}
  for(var bit=0u;bit<18u;bit+=1u){
    let probe=paperProbe(origin,size,PAPER_DIRECTIONS[bit]);
    if(!valid(probe)){continue;}
    let tile=vec3u(probe)/tileSize;
    let index=tile.x+td.x*(tile.y+td.y*tile.z);
    if(compaction[tileChangeFlagsBase()+index]==generation){return true;}
  }
  return false;
}
fn rowAuthorityDirtyGeneration(cell:u32,generation:u32)->bool{
  return rowAuthorityFrontierDirtyGeneration(cell,generation)
    ||rowAuthorityStructuralDirtyGeneration(cell,generation);
}
fn rowAuthorityDirty(cell:u32)->bool{return rowAuthorityDirtyGeneration(cell,frontierGeneration());}
fn rowKeyLess(levelA:u32,mortonA:u32,levelB:u32,mortonB:u32)->bool{
  return levelA<levelB||(levelA==levelB&&mortonA<mortonB);
}
fn rowIdentityLess(cellA:u32,sizeA:u32,cellB:u32,sizeB:u32)->bool{
  if(sizeA==0u){return false;}if(sizeB==0u){return true;}
  return rowKeyLess(u32(firstTrailingBit(sizeA)),rowMorton(cellA),
    u32(firstTrailingBit(sizeB)),rowMorton(cellB));
}
fn findPreviousRow(cell:u32,size:u32,previous:u32,previousCount:u32)->u32{
  var lo=0u;var hi=previousCount;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(previous,mid);
    let otherSize=select(0u,leafHeaders[mid].size,mid<arrayLength(&leafHeaders));
    if(rowIdentityLess(other,otherSize,cell,size)){lo=mid+1u;}else{hi=mid;}}
  if(lo<previousCount&&lo<arrayLength(&leafHeaders)&&frontierCell(previous,lo)==cell
    &&leafHeaders[lo].cell==cell&&leafHeaders[lo].size==size){return lo;}
  return 0xffffffffu;
}
fn findCurrentRow(cell:u32,size:u32,current:u32,currentCount:u32)->u32{
  var lo=0u;var hi=currentCount;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(current,mid);
    if(rowIdentityLess(other,ownerAtIndex(other).size,cell,size)){lo=mid+1u;}else{hi=mid;}}
  if(lo<currentCount){let other=frontierCell(current,lo);
    if(other==cell&&ownerAtIndex(other).size==size){return lo;}}
  return 0xffffffffu;
}
var<workgroup> rowDeltaReduce:array<vec4u,256>;
var<workgroup> rowDeltaScan:array<u32,256>;
var<workgroup> rowDeltaScanTotal:u32;
var<workgroup> rowDeltaRingVotes:array<u32,32>;
fn rowDeltaExclusiveScan(value:u32,lid:u32)->u32{
  rowDeltaScan[lid]=value;workgroupBarrier();
  for(var stride=1u;stride<256u;stride*=2u){
    let index=(lid+1u)*2u*stride-1u;
    if(index<256u){rowDeltaScan[index]+=rowDeltaScan[index-stride];}
    workgroupBarrier();
  }
  if(lid==0u){rowDeltaScanTotal=rowDeltaScan[255u];rowDeltaScan[255u]=0u;}
  workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){
    let index=(lid+1u)*2u*stride-1u;
    if(index<256u){let left=rowDeltaScan[index-stride];
      rowDeltaScan[index-stride]=rowDeltaScan[index];rowDeltaScan[index]+=left;}
    workgroupBarrier();
  }
  return rowDeltaScan[lid];
}

@compute @workgroup_size(256)
fn classifyFrontierCarry(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;let previous=frontierCurrent();let previousCount=frontierCount(previous);
  if(row>=previousCount){return;}
  // The sorted merge below is the exact old/new identity join. Clear the
  // reverse map here so retired identities remain explicitly unmapped without
  // a second capacity-sized preparation/classification pass.
  frontier[rowDeltaOldToNewBase()+row]=0u;
  let cell=frontierCell(previous,row);let old=leafHeaders[row];
  let exactStructural=rowAuthorityStructuralDirtyGeneration(cell,frontier[8u]);
  let dirty=exactStructural||rowAuthorityFrontierDirtyGeneration(cell,frontier[8u]);
  let structuralDirty=dirty;
  let owner=ownerAtIndex(cell);
  let cellMatches=old.cell==cell;
  let sizeMatches=old.size==owner.size;
  let originMatches=isOrigin(cellCoord(cell),owner);
  let exact=cellMatches&&sizeMatches&&originMatches;
  let wet=currentPressureOwnerWet(owner);
  // Exact wet identities retain their previous canonical order even when
  // their authority tile is dirty. Dirty is an affected-payload bit, not a
  // reason to discard and re-sort an otherwise unchanged row identity.
  let keep=exact&&wet;
  // A supposedly clean identity is never silently retired. That indicates
  // incomplete dirty evidence and rejects the candidate generation.
  let reason=select(0u,1u,!cellMatches)|select(0u,2u,!sizeMatches)
    |select(0u,4u,!originMatches)|select(0u,8u,exact&&!wet);
  compaction[rowDeltaFlagsBase()+row]=select(0u,1u,keep)
    |select(0u,reason<<1u,!dirty)|select(0u,32u,dirty)
    |select(0u,64u,structuralDirty);
}

@compute @workgroup_size(256)
fn scanFrontierCarryBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){
  let row=wid.x*256u+lid;let previousCount=frontierCount(frontierCurrent());
  let flag=select(0u,compaction[rowDeltaFlagsBase()+row]&1u,row<previousCount);
  let rank=rowDeltaExclusiveScan(flag,lid);
  if(row<previousCount){compaction[rowDeltaPrefixBase()+row]=rank;}
  if(lid==0u){compaction[rowDeltaBlockTotalsBase()+wid.x]=rowDeltaScanTotal;}
}

@compute @workgroup_size(256)
fn prefixFrontierCarryBlocks(@builtin(local_invocation_index)lid:u32){
  let halted=frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC;
  let blocks=select((frontierCount(frontierCurrent())+255u)/256u,0u,halted);
  let chunk=(blocks+255u)/256u;
  let begin=min(blocks,lid*chunk);let end=min(blocks,begin+chunk);var subtotal=0u;
  for(var block=begin;block<end;block+=1u){subtotal+=compaction[rowDeltaBlockTotalsBase()+block];}
  var cursor=rowDeltaExclusiveScan(subtotal,lid);
  for(var block=begin;block<end;block+=1u){let count=compaction[rowDeltaBlockTotalsBase()+block];
    compaction[rowDeltaBlockTotalsBase()+block]=cursor;cursor+=count;}
  workgroupBarrier();
  if(lid==0u&&!halted){frontier[5]=rowDeltaScanTotal;}
}

fn keptRowsBefore(index:u32,previousCount:u32)->u32{
  if(index>=previousCount){return frontier[5];}
  return compaction[rowDeltaPrefixBase()+index]
    +compaction[rowDeltaBlockTotalsBase()+index/256u];
}
fn candidateLowerBound(cell:u32,size:u32,candidateCount:u32)->u32{
  var lo=0u;var hi=candidateCount;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontier[frontierCandidateBase()+mid];
    if(rowIdentityLess(other,ownerAtIndex(other).size,cell,size)){lo=mid+1u;}else{hi=mid;}}
  return lo;
}
fn previousLowerBound(cell:u32,size:u32,previous:u32,previousCount:u32)->u32{
  var lo=0u;var hi=previousCount;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(previous,mid);
    let otherSize=leafHeaders[mid].size;
    if(rowIdentityLess(other,otherSize,cell,size)){lo=mid+1u;}else{hi=mid;}}
  return lo;
}

@compute @workgroup_size(256)
fn mergeFrontierRows(@builtin(global_invocation_id)gid:vec3u){
  let slot=gid.x;let previous=frontierCurrent();let next=1u-previous;
  let previousCount=frontierCount(previous);let candidateCount=min(frontier[4],frontierListCapacity());
  if(slot<previousCount&&(compaction[rowDeltaFlagsBase()+slot]&1u)!=0u){
    let cell=frontierCell(previous,slot);let size=leafHeaders[slot].size;
    let output=keptRowsBefore(slot,previousCount)+candidateLowerBound(cell,size,candidateCount);
    if(output<frontierListCapacity()){
      frontier[frontierBase(next)+output]=cell;
      let dirty=output!=slot||(compaction[rowDeltaFlagsBase()+slot]&32u)!=0u;
      let structural=dirty;
      frontier[rowDeltaNewToOldBase()+output]=(slot+1u)
        |select(0u,ROW_DELTA_AFFECTED,dirty)
        |select(0u,ROW_DELTA_STRUCTURAL,structural);
      frontier[rowDeltaOldToNewBase()+slot]=output+1u;
    }
  }
  if(slot<candidateCount){
    let cell=frontier[frontierCandidateBase()+slot];let size=ownerAtIndex(cell).size;
    let old=previousLowerBound(cell,size,previous,previousCount);
    // Recurring candidates are additions only and must never collide with a
    // carried exact identity. Leave any malformed collision to the carried
    // writer; the final validator rejects it without a storage race.
    let carriedCollision=old<previousCount&&frontierCell(previous,old)==cell
      &&leafHeaders[old].size==size&&(compaction[rowDeltaFlagsBase()+old]&1u)!=0u;
    if(!carriedCollision){
      let output=slot+keptRowsBefore(old,previousCount);
      if(output<frontierListCapacity()){
        frontier[frontierBase(next)+output]=cell;
        let exact=old<previousCount&&frontierCell(previous,old)==cell
          &&leafHeaders[old].size==size;
        let dirty=!exact||old!=output
          ||rowAuthorityDirtyGeneration(cell,frontier[8u]);
        let structural=dirty;
        frontier[rowDeltaNewToOldBase()+output]=select(old+1u,0u,!exact)
          |select(0u,ROW_DELTA_AFFECTED,dirty)
          |select(0u,ROW_DELTA_STRUCTURAL,structural);
        if(exact){frontier[rowDeltaOldToNewBase()+old]=output+1u;}
      }
    }
  }
}

@compute @workgroup_size(256)
fn finalizeFrontier(@builtin(local_invocation_index)lid:u32){
  // Storage-buffer values are not workgroup-uniform in WebGPU's static
  // uniformity analysis, even though beginFrontier gives this word one
  // writer. Keep every lane alive through the cooperative reduction and gate
  // only lane-local validation work. The rejected generation is discarded by
  // lane zero after the last barrier, leaving the immutable selector intact.
  let frontierRejected=compaction[11]==FRONTIER_FAILED_MAGIC;
  let frontierReused=compaction[11]==FRONTIER_REUSE_MAGIC;
  let previous=frontierCurrent();let next=1u-previous;
  let previousCount=frontierCount(previous);let candidateCount=frontier[4];
  let boundedCandidates=min(candidateCount,frontierListCapacity());
  let required=frontier[5]+candidateCount;
  // A candidate whose liquid authority is unavailable is rejected, never
  // published. Rejection retains the previous frontier selector and retries on
  // the next generation, which is exactly what the lagging coarse publication
  // needs; publishing would instead freeze the topology at zero rows forever.
  var matched=0u;var invalid=select(0u,1u,candidateCount>frontierListCapacity()
    ||required>frontierListCapacity()||!liquidAuthorityAvailable());
  var firstFailure=0xffffffffu;var exactFailures=0u;
  if(!frontierRejected&&!frontierReused){
    for(var row=lid;row<previousCount;row+=256u){
      let flags=compaction[rowDeltaFlagsBase()+row];
      let reason=(flags>>1u)&15u;
      invalid|=select(0u,1u,reason!=0u);
      firstFailure=min(firstFailure,select(0xffffffffu,row*16u+reason,reason!=0u));
      exactFailures+=select(0u,1u,(reason&7u)!=0u);
      if(row>0u){
        let cell=frontierCell(previous,row);let prior=frontierCell(previous,row-1u);
        invalid|=select(0u,1u,!rowIdentityLess(
          prior,leafHeaders[row-1u].size,cell,leafHeaders[row].size));
      }
    }
    for(var row=lid;row<boundedCandidates;row+=256u){
      let cell=frontier[frontierCandidateBase()+row];let size=ownerAtIndex(cell).size;
      if(row>0u){let prior=frontier[frontierCandidateBase()+row-1u];
        let unordered=!rowIdentityLess(prior,ownerAtIndex(prior).size,cell,size);
        invalid|=select(0u,1u,unordered);
        firstFailure=min(firstFailure,select(0xffffffffu,0x10000000u|row,unordered));}
      let old=previousLowerBound(cell,size,previous,previousCount);
      let exact=old<previousCount&&frontierCell(previous,old)==cell&&leafHeaders[old].size==size;
      matched+=select(0u,1u,exact);
      // Delta candidate generation filters every exact previous identity.
      // Seeing one here means the temporal-coherence partition was malformed.
      invalid|=select(0u,1u,exact);
      firstFailure=min(firstFailure,select(0xffffffffu,0x18000000u|row,exact));
    }
    for(var row=lid;row<min(required,frontierListCapacity());row+=256u){
      let cell=frontier[frontierBase(next)+row];let size=ownerAtIndex(cell).size;
      let invalidMember=!isOrigin(cellCoord(cell),ownerAtIndex(cell))
        ||!currentPressureOwnerWet(ownerAtIndex(cell));
      invalid|=select(0u,1u,invalidMember);
      firstFailure=min(firstFailure,select(0xffffffffu,0x20000000u|row,invalidMember));
      if(row>0u){let prior=frontier[frontierBase(next)+row-1u];
        let unordered=!rowIdentityLess(prior,ownerAtIndex(prior).size,cell,size);
        invalid|=select(0u,1u,unordered);
        firstFailure=min(firstFailure,select(0xffffffffu,0x30000000u|row,unordered));}
    }
  }
  rowDeltaReduce[lid]=vec4u(matched,invalid,firstFailure,exactFailures);workgroupBarrier();
  for(var stride=128u;stride>0u;stride>>=1u){
    if(lid<stride){
      let right=rowDeltaReduce[lid+stride];
      rowDeltaReduce[lid]=vec4u(rowDeltaReduce[lid].xy+right.xy,
        min(rowDeltaReduce[lid].z,right.z),rowDeltaReduce[lid].w+right.w);
    }workgroupBarrier();}
  if(lid!=0u){return;}
  if(frontierRejected){
    if(lid==0u){frontier[6]=0u;frontier[9]=1u;}
    return;
  }
  if(frontierReused){
    // The immutable frontier selector and leaf payload remain valid, but the
    // identity maps still need one bounded dispatch so value-refresh
    // consumers can accept the new generation with zero dirty rows.
    let blocks=(previousCount+255u)/256u;
    compaction[1]=blocks;compaction[2]=1u;compaction[3]=1u;
    compaction[4]=0u;compaction[5]=1u;compaction[6]=1u;
    return;
  }
  let carried=frontier[5];
  let added=candidateCount;
  let retired=select(previousCount-carried,0u,carried>previousCount);
  let valid=rowDeltaReduce[0].x==0u&&rowDeltaReduce[0].y==0u&&carried<=previousCount
    &&required==carried+added&&required==previousCount+added-retired
    // A transient wetness-authority gap must never turn a live topology into
    // the terminal zero-row state. Dirty discovery only visits active tiles,
    // so accepting this transition would make recovery impossible even when
    // the next generation's fine/coarse publications are healthy again.
    &&(previousCount==0u||required>0u);
  if(!valid){
    compaction[dirtyFailureBase()]=0x300u;
    compaction[dirtyFailureBase()+1u]=required;
    compaction[dirtyFailureBase()+2u]=carried;
    compaction[dirtyFailureBase()+3u]=rowDeltaReduce[0].z;
    compaction[dirtyFailureBase()+4u]=rowDeltaReduce[0].y;
    compaction[dirtyFailureBase()+5u]=candidateCount;
    compaction[dirtyFailureBase()+6u]=previousCount;
    compaction[dirtyFailureBase()+7u]=boundedCandidates;
    let control=pressureControlBase();compaction[control]=4u;
    compaction[control+1u]=required;compaction[control+2u]=carried;
    // Words 6/7 are later reused for residual floats even on a rejected
    // solve, so preserve the bounded carry-rejection classification in the
    // three control words that remain stable through downstream fail-closed
    // stages. Previous/candidate/kept counts are already present in the
    // frontier header and required/carried words above.
    compaction[control+3u]=rowDeltaReduce[0].z;
    compaction[control+4u]=rowDeltaReduce[0].w;
    compaction[control+5u]=rowDeltaReduce[0].y;
    compaction[control+5u]=coarseWord(0u);
    compaction[control+6u]=coarseWord(1u);
    compaction[control+7u]=params.pressureCapacity.w;
    frontier[6]=0u;
    frontier[9]=4u;
    compaction[11]=FRONTIER_FAILED_MAGIC;compaction[frontierTopologyReuseBase()]=0u;
    compaction[1]=0u;compaction[2]=1u;compaction[3]=1u;
    compaction[4]=0u;compaction[5]=1u;compaction[6]=1u;
    compaction[12]=0u;compaction[13]=1u;compaction[14]=1u;return;
  }
  frontier[next]=required;
  frontier[7]=next;
  frontier[9]=0u;
  frontier[6]=1u;
  let base=rowDeltaControlBase();
  frontier[base]=required;frontier[base+1u]=previousCount;
  frontier[base+2u]=carried;frontier[base+3u]=added;
  frontier[base+4u]=retired;frontier[base+7u]=frontier[8];
  frontier[base+15u]=1u;
  let blocks=(required+255u)/256u;compaction[8]=blocks;
  let x=min(blocks,65535u);var y=1u;if(x>0u){y=(blocks+x-1u)/x;}
  compaction[12]=x;compaction[13]=y;compaction[14]=1u;
  let rowBlocks=max(blocks,(previousCount+255u)/256u);
  compaction[1]=rowBlocks;compaction[2]=1u;compaction[3]=1u;
  // The fourth immutable indirect record is exact, not block-shaped: the
  // cooperative ring kernel owns one 32-lane workgroup for each current row.
  compaction[4]=required;compaction[5]=1u;compaction[6]=1u;
}

@compute @workgroup_size(256)
fn classifyRowDelta(
  @builtin(global_invocation_id)gid:vec3u,
  @builtin(local_invocation_index)lid:u32,
  @builtin(workgroup_id)wid:vec3u,
){
  let reused=frontierGenerationReused();
  let current=frontierCurrent();let previous=1u-current;
  let currentCount=frontierCount(current);let previousCount=frontierCount(previous);
  let row=gid.x;var carried=0u;var invalid=0u;
  if(!reused&&row<currentCount){
    let cell=frontierCell(current,row);let size=ownerAtIndex(cell).size;
    let old=findPreviousRow(cell,size,previous,previousCount);
    // Positional L1 consumers publish by row page. A carried identity that
    // moved because of an insertion/retirement must therefore enter the exact
    // dirty stream even when its spatial authority tile is unchanged.
    let exactStructural=old==0xffffffffu
      ||rowAuthorityStructuralDirtyGeneration(cell,frontierGeneration());
    let affected=exactStructural
      ||rowAuthorityFrontierDirtyGeneration(cell,frontierGeneration());
    let structuralDirty=affected;
    frontier[rowDeltaNewToOldBase()+row]=
      select(old+1u,0u,old==0xffffffffu)|select(0u,ROW_DELTA_AFFECTED,affected)
      |select(0u,ROW_DELTA_STRUCTURAL,structuralDirty);
    compaction[rowDeltaFlagsBase()+row]=select(0u,1u,structuralDirty);
    carried=select(1u,0u,old==0xffffffffu);
    if(row>0u){let prior=frontierCell(current,row-1u);
      if(!rowIdentityLess(prior,ownerAtIndex(prior).size,cell,size)){invalid=1u;}}
  }
  if(!reused&&row<previousCount){
    if(row>=arrayLength(&leafHeaders)||leafHeaders[row].cell!=frontierCell(previous,row)){invalid=1u;}
    else{
      let cell=frontierCell(previous,row);let size=leafHeaders[row].size;
      let mapped=findCurrentRow(cell,size,current,currentCount);
      frontier[rowDeltaOldToNewBase()+row]=select(mapped+1u,0u,mapped==0xffffffffu);
      if(row>0u){let prior=frontierCell(previous,row-1u);
        if(row-1u>=arrayLength(&leafHeaders)
          ||!rowIdentityLess(prior,leafHeaders[row-1u].size,cell,size)){invalid=1u;}}
    }
  }
  rowDeltaReduce[lid]=vec4u(carried,invalid,0u,0u);workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){
    if(lid<stride){rowDeltaReduce[lid]+=rowDeltaReduce[lid+stride];}workgroupBarrier();
  }
  if(lid==0u&&!reused){let at=rowDeltaCarriedBlocksBase()+2u*wid.x;
    compaction[at]=rowDeltaReduce[0].x;compaction[at+1u]=rowDeltaReduce[0].y;}
}
@compute @workgroup_size(256)
fn finalizeRowDeltaClassification(@builtin(local_invocation_index)lid:u32){
  let halted=frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC;
  let currentCount=frontierCount(frontierCurrent());
  let previousCount=frontierCount(1u-frontierCurrent());
  let blocks=select((max(currentCount,previousCount)+255u)/256u,0u,halted);
  let chunk=(blocks+255u)/256u;
  let begin=min(blocks,lid*chunk);let end=min(blocks,begin+chunk);var subtotal=vec2u(0u);
  for(var block=begin;block<end;block+=1u){let at=rowDeltaCarriedBlocksBase()+2u*block;
    subtotal+=vec2u(compaction[at],compaction[at+1u]);}
  rowDeltaReduce[lid]=vec4u(subtotal,0u,0u);workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){
    if(lid<stride){rowDeltaReduce[lid]+=rowDeltaReduce[lid+stride];}workgroupBarrier();
  }
  if(lid==0u&&!halted){
    let base=rowDeltaControlBase();let carried=rowDeltaReduce[0].x;
    let added=select(currentCount-carried,0u,carried>currentCount);
    let retired=select(previousCount-carried,0u,carried>previousCount);
    let valid=rowDeltaReduce[0].y==0u&&carried<=min(currentCount,previousCount)
      &&currentCount==carried+added&&currentCount==previousCount+added-retired;
    frontier[base]=currentCount;frontier[base+1u]=previousCount;
    frontier[base+2u]=carried;frontier[base+3u]=added;
    frontier[base+4u]=retired;frontier[base+7u]=frontierGeneration();
    frontier[base+15u]=select(0u,1u,valid);
  }
}

fn scanRowDeltaBlock(affected:bool,wid:u32,lid:u32){
  let row=wid*256u+lid;let count=frontier[rowDeltaControlBase()];
  var flag=0u;
  if(row<count){
    flag=select(select(0u,1u,(frontier[rowDeltaNewToOldBase()+row]&ROW_DELTA_STRUCTURAL)!=0u),
      select(0u,1u,(frontier[rowDeltaNewToOldBase()+row]&ROW_DELTA_AFFECTED)!=0u),affected);
  }
  let rank=rowDeltaExclusiveScan(flag,lid);
  if(row<count){compaction[rowDeltaPrefixBase()+row]=rank;}
  if(lid==0u){compaction[rowDeltaBlockTotalsBase()+wid]=rowDeltaScanTotal;}
}
fn prefixRowDeltaBlocks(affected:bool,lid:u32){
  let halted=frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC;
  let blocks=select((frontier[rowDeltaControlBase()]+255u)/256u,0u,halted);
  let chunk=(blocks+255u)/256u;
  let begin=min(blocks,lid*chunk);let end=min(blocks,begin+chunk);var subtotal=0u;
  for(var block=begin;block<end;block+=1u){subtotal+=compaction[rowDeltaBlockTotalsBase()+block];}
  var cursor=rowDeltaExclusiveScan(subtotal,lid);
  for(var block=begin;block<end;block+=1u){let value=compaction[rowDeltaBlockTotalsBase()+block];
    compaction[rowDeltaBlockTotalsBase()+block]=cursor;cursor+=value;}
  workgroupBarrier();
  if(lid==0u&&!halted){compaction[rowDeltaBlockTotalsBase()+blocks]=rowDeltaScanTotal;
    frontier[rowDeltaControlBase()+select(5u,6u,affected)]=rowDeltaScanTotal;}
}
@compute @workgroup_size(256)
fn scanDirtyRowDeltaBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){
  scanRowDeltaBlock(false,wid.x,lid);
}
@compute @workgroup_size(256)
fn prefixDirtyRowDeltaBlocks(@builtin(local_invocation_index)lid:u32){prefixRowDeltaBlocks(false,lid);}
@compute @workgroup_size(256)
fn scatterDirtyRowDelta(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;let count=frontier[rowDeltaControlBase()];
  if(row<count){
    let encoded=frontier[rowDeltaNewToOldBase()+row];
    // The next dispatch may set ROW_DELTA_AFFECTED. Preserve its complete
    // input image in plain scratch first so no workgroup can observe another
    // row's in-dispatch publication and accidentally flood through a chain.
    compaction[rowDeltaFlagsBase()+row]=
      select(0u,1u,(encoded&ROW_DELTA_AFFECTED)!=0u);
    if((encoded&ROW_DELTA_STRUCTURAL)!=0u){
      let output=compaction[rowDeltaPrefixBase()+row]
        +compaction[rowDeltaBlockTotalsBase()+row/256u];
      frontier[rowDeltaDirtyRowsBase()+output]=row;
    }
  }
}

const ROW_DELTA_RING_DIRECTION_COUNT:u32=18u;
const ROW_DELTA_RING_DIRECTIONS=array<vec3i,18>(
  vec3i(0,-1,-1),vec3i(-1,0,-1),vec3i(0,0,-1),vec3i(1,0,-1),vec3i(0,1,-1),
  vec3i(-1,-1,0),vec3i(0,-1,0),vec3i(1,-1,0),vec3i(-1,0,0),vec3i(1,0,0),
  vec3i(-1,1,0),vec3i(0,1,0),vec3i(1,1,0),
  vec3i(0,-1,1),vec3i(-1,0,1),vec3i(0,0,1),vec3i(1,0,1),vec3i(0,1,1));
fn rowDeltaRingDirectionAffected(row:u32,d:vec3i)->u32{
  let h=leafHeaders[row];let origin=cellCoord(h.cell);var probe=vec3i(0);
  for(var axis=0u;axis<3u;axis+=1u){
    probe[axis]=select(select(i32(origin[axis]+h.size/2u),
      i32(origin[axis]+h.size),d[axis]>0),i32(origin[axis])-1,d[axis]<0);
  }
  if(!valid(probe)){return 0u;}
  let owner=ownerAt(probe);
  if(!ownerValid(owner)){return 0u;}
  let neighbor=candidateFrontierRow(index(unpackOrigin(owner.packedOrigin)));
  let count=frontier[rowDeltaControlBase()];
  if(neighbor==0xffffffffu||neighbor>=count){return 0u;}
  return compaction[rowDeltaFlagsBase()+neighbor];
}

// Mini/default lane: one 32-lane workgroup owns one row. Lanes 0..17 resolve
// the exact face/edge directions, then a deterministic OR tree gives lane zero
// the only write. All neighbour tests read the preceding dispatch's snapshot.
@compute @workgroup_size(32)
fn markRowDeltaRing(
  @builtin(workgroup_id)wid:vec3u,
  @builtin(local_invocation_index)lane:u32,
){
  let base=rowDeltaControlBase();let count=frontier[base];let row=wid.x;
  let live=!frontierGenerationReused()&&compaction[11]!=FRONTIER_FAILED_MAGIC&&row<count;
  let membershipChanged=live&&(frontier[base+3u]!=0u||frontier[base+4u]!=0u);
  var vote=select(0u,1u,membershipChanged);
  if(live&&!membershipChanged){
    if(lane==0u){vote=compaction[rowDeltaFlagsBase()+row];}
    if(lane<ROW_DELTA_RING_DIRECTION_COUNT){
      vote|=rowDeltaRingDirectionAffected(row,ROW_DELTA_RING_DIRECTIONS[lane]);
    }
  }
  rowDeltaRingVotes[lane]=vote;workgroupBarrier();
  for(var stride=16u;stride>0u;stride/=2u){
    if(lane<stride){rowDeltaRingVotes[lane]|=rowDeltaRingVotes[lane+stride];}
    workgroupBarrier();
  }
  if(lane==0u&&live&&rowDeltaRingVotes[0u]!=0u){
    frontier[rowDeltaNewToOldBase()+row]|=ROW_DELTA_AFFECTED;
  }
}

// Very large row capacities cannot encode an exact one-dimensional
// workgroup-per-row extent. Retain a block-shaped large-capacity kernel that
// reads the same immutable snapshot so its semantics remain exactly one ring.
@compute @workgroup_size(256)
fn markRowDeltaRingBlocks(@builtin(global_invocation_id)gid:vec3u){
  let base=rowDeltaControlBase();let count=frontier[base];let row=gid.x;
  if(frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC||row>=count){return;}
  let membershipChanged=frontier[base+3u]!=0u||frontier[base+4u]!=0u;
  var affected=membershipChanged||compaction[rowDeltaFlagsBase()+row]!=0u;
  for(var direction=0u;direction<ROW_DELTA_RING_DIRECTION_COUNT&&!affected;direction+=1u){
    affected=rowDeltaRingDirectionAffected(row,ROW_DELTA_RING_DIRECTIONS[direction])!=0u;
  }
  if(affected){frontier[rowDeltaNewToOldBase()+row]|=ROW_DELTA_AFFECTED;}
}

@compute @workgroup_size(256)
fn scanAffectedRowDeltaBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){
  scanRowDeltaBlock(true,wid.x,lid);
}
@compute @workgroup_size(256)
fn prefixAffectedRowDeltaBlocks(@builtin(local_invocation_index)lid:u32){prefixRowDeltaBlocks(true,lid);}
@compute @workgroup_size(256)
fn compactRowDelta(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;let count=frontier[rowDeltaControlBase()];
  if(row<count&&(frontier[rowDeltaNewToOldBase()+row]&ROW_DELTA_AFFECTED)!=0u){
    let output=compaction[rowDeltaPrefixBase()+row]
      +compaction[rowDeltaBlockTotalsBase()+row/256u];
    frontier[rowDeltaAffectedRowsBase()+output]=row;
  }
}
@compute @workgroup_size(1)
fn publishRowDelta(){
  if(frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC){return;}
  let base=rowDeltaControlBase();let count=frontier[base];
  let previous=frontier[base+1u];let carried=frontier[base+2u];
  let added=frontier[base+3u];let retired=frontier[base+4u];
  let dirty=frontier[base+5u];let affected=frontier[base+6u];
  let valid=frontier[base+15u]==1u&&carried<=previous
    &&count==carried+added&&count==previous+added-retired
    &&dirty<=affected&&affected<=count;
  frontier[base+8u]=select(0u,ROW_DELTA_VALID,valid);
  frontier[base+9u]=(dirty+63u)/64u;frontier[base+10u]=1u;frontier[base+11u]=1u;
  frontier[base+12u]=(affected+63u)/64u;frontier[base+13u]=1u;frontier[base+14u]=1u;
}
@compute @workgroup_size(256)
fn publishReusedRowDelta(@builtin(global_invocation_id)gid:vec3u){
  if(!frontierGenerationReused()){return;}
  let row=gid.x;let count=frontierCount(candidateFrontierCurrent());
  if(row<count){
    frontier[rowDeltaNewToOldBase()+row]=row+1u;
    frontier[rowDeltaOldToNewBase()+row]=row+1u;
    compaction[rowDeltaFlagsBase()+row]=0u;
  }
  if(row==0u){
    let base=rowDeltaControlBase();
    frontier[base]=count;frontier[base+1u]=count;frontier[base+2u]=count;
    frontier[base+3u]=0u;frontier[base+4u]=0u;frontier[base+5u]=0u;
    frontier[base+6u]=0u;frontier[base+7u]=frontier[8u];
    frontier[base+8u]=ROW_DELTA_VALID;
    frontier[base+9u]=0u;frontier[base+10u]=1u;frontier[base+11u]=1u;
    frontier[base+12u]=0u;frontier[base+13u]=1u;frontier[base+14u]=1u;
    frontier[base+15u]=1u;
  }
}

fn candidateLeafInfo(c: u32) -> vec3u {
  let owner = ownerAtIndex(c);
  if (candidateFrontierRow(c)==0xffffffffu || !isOrigin(cellCoord(c), owner)) { return vec3u(0u); }
  return vec3u(1u, 0u, 0u);
}

var<workgroup> scanPairs: array<vec3u, 256>;
var<workgroup> emitOverflow: atomic<u32>;

@compute @workgroup_size(256)
fn planLeaves(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  var value = vec3u(0u);
  let current = candidateFrontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  if (slot < frontierCount(current)) { value = candidateLeafInfo(frontierCell(current, slot)); }
  scanPairs[lid] = value;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) { scanPairs[lid] += scanPairs[lid + stride]; }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let total = scanPairs[0];
    let block = wid.x + wid.y * compaction[12];
    compaction[15u + 3u * block] = total.x;
    compaction[16u + 3u * block] = total.y;
    compaction[17u + 3u * block] = total.z;
  }
}

@compute @workgroup_size(256)
fn scanLeafBlocks(@builtin(local_invocation_id) lid3: vec3u) {
  let lid = lid3.x;
  if (lid == 0u) {
    atomicStore(&emitOverflow, select(0u, 1u,
      compaction[11] == FRONTIER_REUSE_MAGIC || compaction[11] == FRONTIER_FAILED_MAGIC));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&emitOverflow) != 0u) {
    if (lid == 0u) {
      let publication = frontierPublicationBase();
      // Word zero is written last by a normal publication, so a visible magic
      // value proves all twelve control words belong to one complete row set.
      if (compaction[11] == FRONTIER_REUSE_MAGIC
          && compaction[publication] == FRONTIER_REUSE_MAGIC) {
        for (var word = 0u; word < 12u; word += 1u) {
          compaction[word] = compaction[publication + 1u + word];
        }
        compaction[frontierTopologyReuseBase()] = 1u;
      } else {
        let control = pressureControlBase();
        compaction[control] = 4u;
        compaction[control + 1u] = compaction[11u];
        compaction[control + 2u] = compaction[publication];
        compaction[control + 3u] = compaction[dirtyAuthorityBase()];
        compaction[control + 4u] = compaction[frontierTopologyReuseBase()];
        compaction[0] = 0u; compaction[2] = 0u; compaction[5] = 0u;
        compaction[9] = 0u;
      }
    }
    return;
  }
  let blocks = compaction[8];
  let chunk = (blocks + 255u) / 256u;
  let base = lid * chunk;
  var sum = vec3u(0u);
  for (var i = 0u; i < chunk; i += 1u) {
    let b = base + i;
    if (b < blocks) { sum += vec3u(compaction[15u + 3u * b], compaction[16u + 3u * b], compaction[17u + 3u * b]); }
  }
  scanPairs[lid] = sum;
  for (var stride = 1u; stride < 256u; stride <<= 1u) {
    workgroupBarrier();
    var add = vec3u(0u);
    if (lid >= stride) { add = scanPairs[lid - stride]; }
    workgroupBarrier();
    scanPairs[lid] += add;
  }
  workgroupBarrier();
  var running = scanPairs[lid] - sum;
  for (var i = 0u; i < chunk; i += 1u) {
    let b = base + i;
    if (b < blocks) {
      let pair = vec3u(compaction[15u + 3u * b], compaction[16u + 3u * b], compaction[17u + 3u * b]);
      compaction[15u + 3u * b] = running.x;
      compaction[16u + 3u * b] = running.y;
      compaction[17u + 3u * b] = running.z;
      running += pair;
    }
  }
  if (lid == 255u) {
    let total = scanPairs[255];
    let control = pressureControlBase();
    let frontierOverflow = (compaction[control] & 2u) != 0u;
    let rowOverflow = rowIndexedPressure && total.x > params.pressureCapacity.x;
    let overflow = frontierOverflow || rowOverflow;
    let publishedRows = select(total.x, 0u, overflow);
    compaction[0] = publishedRows; compaction[1] = 0u;
    compaction[control] = select(0u, 2u, frontierOverflow) | select(0u, 1u, rowOverflow);
    compaction[control + 1u] = max(total.x, select(0u, compaction[control + 1u], frontierOverflow));
    compaction[control + 2u] = 0u;
    compaction[control + 3u] = select(0u, (dims().x + 3u) / 4u, overflow);
    compaction[control + 4u] = select(1u, (dims().y + 3u) / 4u, overflow);
    compaction[control + 5u] = select(1u, (dims().z + 3u) / 4u, overflow);
    let blocks = (publishedRows + 255u) / 256u;
    let x = min(blocks, 65535u);
    var y = 1u;
    if (x > 0u) { y = (blocks + x - 1u) / x; }
    compaction[2] = x; compaction[3] = y; compaction[4] = 1u;
    compaction[5] = 0u; compaction[6] = 1u; compaction[7] = 1u; compaction[8] = 0u;
    let leafX = min(publishedRows, 65535u);
    var leafY = 1u;
    if (leafX > 0u) { leafY = (publishedRows + leafX - 1u) / leafX; }
    compaction[9] = leafX; compaction[10] = leafY; compaction[11] = 1u;
    if (!overflow) {
      let publication = frontierPublicationBase();
      for (var word = 0u; word < 12u; word += 1u) {
        compaction[publication + 1u + word] = compaction[word];
      }
      compaction[publication] = FRONTIER_REUSE_MAGIC;
    }
  }
}

@compute @workgroup_size(256)
fn emitLeaves(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  if (lid == 0u) { atomicStore(&emitOverflow, select(0u, 1u, pressureOverflowed())); }
  workgroupBarrier();
  if (workgroupUniformLoad(&emitOverflow) != 0u) { return; }
  var value = vec3u(0u);
  let current = candidateFrontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  var cell = 0u;
  if (slot < frontierCount(current)) { cell = frontierCell(current, slot); value = candidateLeafInfo(cell); }
  scanPairs[lid] = value;
  for (var stride = 1u; stride < 256u; stride <<= 1u) {
    workgroupBarrier();
    var add = vec3u(0u);
    if (lid >= stride) { add = scanPairs[lid - stride]; }
    workgroupBarrier();
    scanPairs[lid] += add;
  }
  workgroupBarrier();
  if (value.x == 1u) {
    let exclusive = scanPairs[lid] - value;
    let block = wid.x + wid.y * compaction[12];
    let row = compaction[15u + 3u * block] + exclusive.x;
    let previousRow = rowDeltaMapOld(frontier[rowDeltaNewToOldBase()+row]);
    var warm = 0.0;
    if (rowIndexedPressure && previousRow < arrayLength(&pressureIn)) { warm = pressureIn[previousRow]; }
    if (rowIndexedPressure) {
      pressureOut[row] = select(0.0, warm, (params.pressureCapacity.w & 1u) != 0u);
    }
    let acceptedOwner=ownerAtIndex(cell);
    // A genuinely new factor-one row was admitted by the compact advected
    // summary. Preserve that classification across the topology flip so the
    // sole coarse phi field can seed the row even though no prior directory
    // entry exists at this identity.
    let predictedWet = fineSummaryFactor == 1u && previousRow == 0xffffffffu;
    // Crossing happens under a sub-cell CFL step, so seed just inside the
    // interface. Redistance owns the magnitude after the sign handoff.
    let predictedPhi = -0.05 * f32(acceptedOwner.size) * params.cellRelax.x;
    leafHeaders[row] = LeafHeader(cell, 0u, 0u, acceptedOwner.size, 0.0, 0.0,
      select(0u, bitcast<u32>(predictedPhi), predictedWet),
      select(0u, COARSE_PREDICTED_WET_MAGIC, predictedWet), vec4f(0.0));
    markAcceptedOwner(unpackOrigin(acceptedOwner.packedOrigin));
  }
}

`;

/**
 * Splice the grading-fixpoint experiment into the projection shader.
 *
 * Applied only when `FLUID_OCTREE_GRADING_FIXPOINT=1`. Every edit is an exact
 * string replacement asserted to match once, so the transform either produces
 * the whole experiment or throws at module evaluation -- it can never emit a
 * half-patched shader. With the flag unset this is not called at all and the
 * module is byte-identical to the unflagged tree.
 */
function octreeGradingFixpointShader(source: string): string {
  const splice = (from: string, to: string): void => {
    const occurrences = source.split(from).length - 1;
    if (occurrences !== 1) {
      throw new Error(
        `Grading-fixpoint shader splice matched ${occurrences} times, expected 1`);
    }
    source = source.replace(from, to);
  };
  splice(
    "const OWNER_WORD_TOPOLOGY: u32 = 0x00200000u;",
    `const OWNER_WORD_TOPOLOGY: u32 = 0x00200000u;
// Projection-owned scratch inside the owner arena's reserved block. The page
// authority owns words 0-15 and never touches these.
const GRADING_SPLIT_FLAG: u32 = ${OCTREE_OWNER_ARENA_PROJECTION_WORDS.gradingSplitFlag}u;
const GRADING_CONVERGED: u32 = ${OCTREE_OWNER_ARENA_PROJECTION_WORDS.gradingConverged}u;

// The grading closure is a fixpoint, and it reaches it long before it stops.
//
// A balance round writes owner state ONLY through a split, so a round that
// claims none leaves the state every later round would read bit-identical, and
// every later round is therefore guaranteed to claim none either. The rounds
// are consecutive dispatches in one compute pass, where WebGPU orders each
// dispatch's writes before the next dispatch's reads, so round r's verdict is
// readable at the top of round r+1 with no barrier and no readback.
//
// ZERO means keep grading. That polarity is load-bearing: a freshly created,
// zero-filled arena must run the full unconditional closure, because a flag
// that had to be seeded before it was safe disables grading on any path that
// reaches the rounds first -- and a tree that is never 2:1 balanced publishes
// no liquid-row frontier at all.
fn markGradingSplit() { atomicStore(&owners[GRADING_SPLIT_FLAG], 1u); }
fn gradingRoundActive() -> bool {
  return atomicLoad(&owners[GRADING_CONVERGED]) == 0u;
}`);
  splice(
    `  let word = encodePagedOwner(origin, origin, size / 2u) | membership;
  return atomicMin(&owners[at], word) > word;`,
    `  let word = encodePagedOwner(origin, origin, size / 2u) | membership;
  let claimed = atomicMin(&owners[at], word) > word;
  if (claimed) { markGradingSplit(); }
  return claimed;`);
  splice(
    `fn balanceTopology(@builtin(global_invocation_id) gid: vec3u) {
  let base = gid * 2u;`,
    `fn balanceTopology(@builtin(global_invocation_id) gid: vec3u) {
  if (!gradingRoundActive()) { return; }
  let base = gid * 2u;`);
  splice(
    `fn balanceTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let base = deltaTopologyCandidate(wid, lid);`,
    `fn balanceTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  if (!gradingRoundActive()) { return; }
  let base = deltaTopologyCandidate(wid, lid);`);
  // The cooperative block ends in a barrier. WGSL uniformity analysis rejects a
  // barrier reachable only under a storage-derived condition, and under
  // skip_validation that surfaces as a Dawn abort inside entry-point lookup
  // rather than as a shader error -- so this gate rides the lane-zero
  // eligibility store that already exists for exactly this reason.
  splice(
    `    let inBounds = all(origin < dims()) && !topologyStructurallyQuiescent();
    let owner = ownerAt(vec3i(min(origin, dims() - vec3u(1u))));
    atomicStore(&balanceEligible, select(0u, 1u, inBounds && ownerValid(owner)`,
    `    let inBounds = all(origin < dims()) && !topologyStructurallyQuiescent()
      && gradingRoundActive();
    let owner = ownerAt(vec3i(min(origin, dims() - vec3u(1u))));
    atomicStore(&balanceEligible, select(0u, 1u, inBounds && ownerValid(owner)`);
  splice(
    `  if (workgroupUniformLoad(&balanceFlags[0]) == 0u) { return; }
  let cells = size * size * size;`,
    `  if (workgroupUniformLoad(&balanceFlags[0]) == 0u) { return; }
  // This path splits without going through claimLeafSplit, so it raises the
  // flag itself; one lane is enough and the store is idempotent.
  if (lid == 0u) { markGradingSplit(); }
  let cells = size * size * size;`);
  splice(
    `@compute @workgroup_size(1)
fn beginFrontier() {`,
    `// Carry one round's verdict into the next: one lane, between rounds, inside
// the open pass.
@compute @workgroup_size(1)
fn advanceGradingRound() {
  let raised = atomicLoad(&owners[GRADING_SPLIT_FLAG]);
  atomicStore(&owners[GRADING_CONVERGED], select(1u, 0u, raised != 0u));
  atomicStore(&owners[GRADING_SPLIT_FLAG], 0u);
}

@compute @workgroup_size(1)
fn beginFrontier() {`);
  splice(
    `  var next = frontier[8] + 1u;
  if (next == 0u) { next = 1u; }
  frontier[8] = next;`,
    `  var next = frontier[8] + 1u;
  if (next == 0u) { next = 1u; }
  frontier[8] = next;
  // Re-arm per advance. Not load-bearing -- zero already means keep grading --
  // but it stops a converged verdict from leaking across generations.
  atomicStore(&owners[GRADING_CONVERGED], 0u);
  atomicStore(&owners[GRADING_SPLIT_FLAG], 0u);`);
  return source;
}

/**
 * The projection module. Byte-identical to the unflagged tree unless
 * `FLUID_OCTREE_GRADING_FIXPOINT=1`, which is asserted by
 * `tests/octree-balance-elision.test.ts`.
 */
export const octreeProjectionShader = octreeGradingFixpointEnabled()
  ? octreeGradingFixpointShader(octreeProjectionShaderBase)
  : octreeProjectionShaderBase;

/**
 * Losasso owns a distinct topology shader module. Its free-surface shell is
 * uniformly finest, so a size-two leaf with finite near-interface evidence is
 * allowed to fall through to the normal distance predicate and split. The
 * frozen Power module keeps the compact size-two surface rows verbatim.
 */
export function octreeLosassoSurfaceGradingShader(source: string): string {
  const compactSurfaceRows = `  if (fineSummaryFactor != 1u && size <= 2u) {
    return false;
  }`;
  if (!source.includes(compactSurfaceRows)) {
    throw new Error("Losasso grading transform could not locate the Power surface-row clause");
  }
  return source.replace(compactSurfaceRows,
    `  // Losasso: size-two rows inside the two-fine-cell shell split to unit rows.`);
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
  shaderSource = octreeProjectionShader,
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
