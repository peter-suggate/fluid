import type { SceneDescription } from "./model";
import { OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS } from "./octree-face-fragments";
import { WebGPUOctreeFaceMirror } from "./webgpu-octree-face-mirror";
import { WebGPUOctreeFaceTransport } from "./webgpu-octree-face-transport";
import { WebGPUOctreeSolidFaces } from "./webgpu-octree-solid-faces";
import { WebGPUOctreeSurfaceAdapter } from "./webgpu-octree-surface-adapter";
import {
  OCTREE_UNIFIED_SURFACE_RESIDENT_FRACTION,
  WebGPUOctreeSurfacePages,
} from "./webgpu-octree-surface-pages";
import { WebGPUOctreeSimulationOwnerPages } from "./webgpu-octree-owner-pages";
import { WebGPUOctreePhiDifferential, type OctreePagedPhiDifferential } from "./webgpu-octree-phi-diagnostic";
import { planOctreeSurfaceStateAllocation } from "./octree-surface-allocation";
import { planOctreeAnalyticBootstrapBounds } from "./octree-analytic-bootstrap";
import { WebGPUOctreeAnalyticBootstrapWorklist } from "./webgpu-octree-analytic-bootstrap";
import { combineInitialBrickWet, damBreakFractions, initialFluidBrickContainsCell } from "./initial-fluid";
import { signedDistanceFromVolume } from "./quadtree-tall-cell-grid";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import { WebGPUQuadtreeSurfaceState, type SurfaceInflowState } from "./webgpu-quadtree-builder";
import { OctreeSparseBrickWorld } from "./webgpu-octree-sparse-bricks";
import { CompactOctreeVoxelInspection } from "./webgpu-octree-voxel-inspection";
import type { FluidBrickAtlasMode, FluidBrickAtlasSamplingSource } from "./webgpu-brick-atlas";
import {
  GPUFluidBrickResidency,
  FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES,
  FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_RETIRED_CANDIDATE_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_RETIRED_DISPATCH_OFFSET_BYTES,
  planSurfaceCandidateResidencyPools,
} from "./webgpu-fluid-brick-residency";
import {
  WebGPUSparseSurfaceBand,
  type SparseSurfaceBandGPUSource,
  type SparseSurfaceBandMode,
} from "./webgpu-sparse-surface-band";
import type { GPUInitializationTask } from "./gpu-initialization";
import {
  fetchGeneratedOctreePowerCatalog,
  decodeGeneratedOctreePowerCatalog,
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
  type GeneratedOctreePowerCatalogViews,
} from "./generated/octree-power-catalog";
import { WebGPUOctreePowerDescriptor } from "./webgpu-octree-power-descriptor";
import { WebGPUOctreePowerTopology } from "./webgpu-octree-power-topology";
import {
  OCTREE_POWER_FACE_QUADRATURE_BYTES,
  WebGPUOctreePowerFaces,
  octreePowerClosedBoundaryMask,
  planOctreePowerFaces,
} from "./webgpu-octree-power-faces";
import { WebGPUOctreePowerOperator, planOctreePowerGPUOperator } from "./webgpu-octree-power-operator";
import { WebGPUOctreePowerFaceSeed } from "./webgpu-octree-power-face-seed";
import { WebGPUOctreePowerFaceTransfer } from "./webgpu-octree-power-face-transfer";
import { WebGPUOctreePowerSolidFaces } from "./webgpu-octree-power-solid-faces";
import { WebGPUOctreeSolidVertexSdf } from "./webgpu-octree-solid-vertex-sdf";
import { WebGPUOctreePowerVelocity } from "./webgpu-octree-power-velocity";
import {
  planOctreePowerVelocityChunkCapacity,
  WebGPUOctreePowerVelocityPrepass,
} from "./webgpu-octree-power-velocity-prepass";
import { WebGPUOctreeMGPCG } from "./webgpu-octree-mgpcg";
import { WebGPUOctreeFirstOrderVCycle } from "./webgpu-octree-first-order-vcycle";
import {
  OCTREE_FACE_BAND_ENCODE_PHASES,
  unpackOctreeFaceBandPointFieldControl,
  unpackOctreeFaceBandTransientPowerControl,
  WebGPUOctreeFaceFastMarch,
  type OctreeFaceBandEncodePhase,
} from "./webgpu-octree-face-fast-march";
import { WebGPUOctreeCoarseLevelSet } from "./webgpu-octree-coarse-levelset";
import { WebGPUOctreePowerCoarseLevelSet } from "./webgpu-octree-power-coarse-levelset";
import { WebGPUFineToCoarseLevelSet } from "./webgpu-octree-fine-to-coarse-levelset";
import { planFineLevelSetBricks } from "./octree-fine-levelset-bricks";
import {
  WebGPUFineLevelSetBricks,
  type WebGPUFineLevelSetBrickSource,
} from "./webgpu-octree-fine-levelset-bricks";
import { WebGPUFineLevelSetRedistance } from "./webgpu-octree-fine-levelset-redistance";
import { WebGPUFineLevelSetTransport } from "./webgpu-octree-fine-levelset-transport";
import { WebGPUFineLevelSetVolumeCorrection } from "./webgpu-octree-fine-levelset-volume";
import { planFineLevelSetGPUSummaries, WebGPUFineLevelSetSummaries } from "./webgpu-octree-fine-levelset-summary";
import {
  planFineLevelSetTopologyBand,
  WebGPUFineLevelSetLeafSeeds,
  WebGPUFineLevelSetTopology,
} from "./webgpu-octree-fine-levelset-topology";

type OctreePipelineVariants = { full: GPUComputePipeline; active: GPUComputePipeline; retired: GPUComputePipeline };

/** Ordered t=0 authority checkpoints; async startup fences every phase.
 * Aanjaneya et al. (2017), Section 5 p.8, first constructs regular octree-face
 * neighborhoods, augments T-junctions with local Delaunay tetrahedra, marches
 * velocities on that face graph, and only then interpolates the result back
 * to power faces. */
export const OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES = [
  { id: "cold-topology", label: "Cold octree topology" },
  { id: "power-operator-authority", label: "Power faces and operator authority" },
  { id: "surface-global-fine", label: "Surface and global-fine redistance publication" },
  { id: "section5-face-band-topology", label: "Section 5 face-band row topology" },
  { id: "section5-face-band-transitions", label: "Section 5 Delaunay transition adjacency and regular faces" },
  { id: "section5-face-band-fast-march", label: "Section 5 face-centered velocity fast march" },
  { id: "section5-face-band-power-publication", label: "Section 5 regular-face to power-face publication" },
  { id: "sparse-render-world", label: "Sparse render world publication" },
] as const;
export type OctreeInitialSparseAuthorityPhaseId = typeof OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES[number]["id"];

async function loadGeneratedOctreePowerCatalog(): Promise<GeneratedOctreePowerCatalogViews> {
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
interface OctreePipelineCacheEntry {
  base: GPUComputePipeline[];
  refine: Map<number, OctreePipelineVariants>;
  refineCoarse: Map<number, OctreePipelineVariants>;
  balanceCoarse: Map<number, OctreePipelineVariants>;
  materialize: GPUComputePipeline;
  pressureImpulse: GPUComputePipeline;
}
const octreePipelineCache = new WeakMap<GPUDevice, Map<string, OctreePipelineCacheEntry>>();

export interface OctreeProjectionOptions {
  pressureIterations: number;
  /** Default-off migration mirror for the canonical adaptive velocity-face ABI. */
  faceVelocityMirror?: boolean;
  /** Default-off A/B: replace assembled divergence RHS with the face mirror. */
  faceVelocityRhs?: boolean;
  /** Method-default U3: advect and force canonical face velocities compactly. */
  faceVelocityTransport?: boolean;
  /** Solve for pressure relative to a fixed tank-fill rest-surface reference. */
  hydrostaticSplit?: boolean;
  maximumLeafSize?: 2 | 4 | 8 | 16 | 32;
  /** 0 = finest cells everywhere; 1 = full distance-graded coarsening. */
  adaptivity?: number;
  /** Pure-phase cells farther from liquid/solid interfaces than this finest-cell band may remain coarse. */
  interfaceRefinementBandCells?: number;
  /** Adds up to eight finest cells of refinement support around curved or strongly straining surface regions. */
  surfaceDetailStrength?: number;
  /** Dynamic fine level-set pages around phi=0; off preserves the dense-only path. */
  sparseSurfaceBand?: "off" | SparseSurfaceBandMode;
  /** Fine level-set samples per coarse transport cell edge. */
  surfaceRefinementFactor?: 1 | 2 | 4;
  /** Opt-in domain-global narrow-band mirror; leaf-attached pages remain the renderer rollback. */
  globalFineLevelSetFactor?: 4 | 8;
  /** Explicit physical brick cap for the global factor-4/factor-8 mirror. */
  globalFineLevelSetMaximumBricks?: number;
  /** Two-sided sparse phi support measured in fine cells. */
  sparseSurfaceBandCells?: number;
  /** Fraction of the logical fine-page lattice backed by physical slots. */
  sparseSurfacePageFraction?: number;
  /** Brick-pooled phi/velocity atlas ownership; off avoids atlas allocation. */
  brickAtlas?: "off" | FluidBrickAtlasMode;
  /** Velocity-swept brick residency support plus downstream activation. */
  brickPreActivation?: boolean;
  /** Execute coarse level-set transport/redistancing only on resident bricks. */
  brickSparseSurface?: boolean;
  /** Execute post-projection velocity extrapolation only on wet-domain resident bricks. */
  brickSparseExtrapolation?: boolean;
  jacobiRelaxation?: number;
  extrapolationSweeps?: number;
  /**
   * Leaf pressure solve strategy. "auto" selects the Section 4.3 hybrid PCG
   * for admitted power authority and Chebyshev otherwise. An explicit
   * non-authoritative "mgpcg" request retains the aggregate preconditioner as
   * a rollback path. Both consume the same compact rows.
   * the liquid leaf origins, assembles each row's diagonal / flux / merged
   * neighbor table once per solve, then applies a Chebyshev-accelerated
   * polynomial in row-parallel indirect dispatches. "compact" retains the
   * warm-started weighted-Jacobi ladder for A/B. "megakernel" instead
   * runs the whole iteration loop in one persistent single-workgroup dispatch
   * with a residual early exit — it wins only when solves converge within a
   * few iterations. "dense" is the legacy one-thread-per-finest-cell ladder
   * kept for A/B comparison.
   */
  leafSolver?: "auto" | "dense" | "compact" | "chebyshev" | "mgpcg" | "megakernel";
  /**
   * Start each compacted solve from the previous step's pressure field
   * (default) instead of clearing to zero. The legacy "dense" ladder always
   * cold-starts so it remains a faithful pre-compaction baseline.
   */
  pressureWarmStart?: boolean;
  /** Advanced safety override for the compact pressure-row arena. */
  pressureRowCapacity?: number;
  /** Migration switch for the generalized power-diagram pressure path. */
  powerDiagramProjection?: "off" | "mirror" | "authoritative";
}

export interface OctreePowerProjectionPolicy {
  readonly requested: "off" | "mirror" | "authoritative";
  readonly mirrorEnabled: boolean;
  readonly authoritative: boolean;
  readonly fallbackReason?: string;
}

/** Authority remains fail-closed until canonical power-face velocity seeding is complete. */
export function resolveOctreePowerProjectionPolicy(
  requested: "off" | "mirror" | "authoritative" | undefined,
  physicalSpacing: readonly [number, number, number],
  hasTerrain: boolean,
  rigidBodyCount: number,
  canonicalVelocitySeedAvailable = false,
  generalizedRigidFacesAvailable = false,
  importedOrSeededGeometry = false,
  terrainVertexSdfAvailable = false,
  terrainRollbackSeedAvailable = false,
): OctreePowerProjectionPolicy {
  const mode = requested ?? "off";
  if (mode === "off") return { requested: mode, mirrorEnabled: false, authoritative: false };
  const isotropic = physicalSpacing.every((value) => Number.isFinite(value) && value > 0)
    && Math.max(...physicalSpacing) / Math.min(...physicalSpacing) <= 1 + 1e-5;
  const limitations = [
    !isotropic && "power catalog requires isotropic finest cells",
    hasTerrain && !terrainVertexSdfAvailable
      && "terrain authority lacks paper-required cell-vertex solid SDF embedded-boundary data",
    hasTerrain && !terrainRollbackSeedAvailable
      && "terrain authority lacks a canonical compact-face rollback seed",
    importedOrSeededGeometry && "imported/seeded geometry requires the compatibility bootstrap and axis projection",
    rigidBodyCount > 0 && !generalizedRigidFacesAvailable && "general power-face rigid apertures are not implemented",
    mode === "authoritative" && !canonicalVelocitySeedAvailable
      && "authoritative power projection requires transferred compact face velocity",
  ].filter(Boolean);
  const authoritative = mode === "authoritative" && limitations.length === 0;
  return { requested: mode, mirrorEnabled: true, authoritative,
    ...(limitations.length > 0 ? { fallbackReason: limitations.join("; ") } : {}) };
}

export function adaptiveFaceRhsIsSupported(
  requested: boolean,
  hasTerrain: boolean,
  rigidBodyCount: number,
  hydrostaticSplit: boolean,
  adaptiveSolidFaces = false,
  terrainEmbeddedBoundaryAvailable = false,
): boolean {
  return requested && (!hasTerrain || terrainEmbeddedBoundaryAvailable)
    && (rigidBodyCount === 0 || adaptiveSolidFaces || terrainEmbeddedBoundaryAvailable) && !hydrostaticSplit;
}

export function octreeSparseWorldRequired(
  directPagedTopology: boolean,
  surfacePagesRequested: boolean,
  hasTerrain: boolean,
  rigidBodyCount: number,
  compatibilityRequested = false,
): boolean {
  return compatibilityRequested || !directPagedTopology || !surfacePagesRequested || hasTerrain || rigidBodyCount > 0;
}

export interface OctreeDensePhiReleaseState {
  directPagedTopology: boolean;
  surfacePagesBootstrapped: boolean;
  pagedProjectionGroupsActive: boolean;
  faceGroupsPageNative: boolean;
  surfaceAdapterPageNative: boolean;
  topologyUsesSurfaceCandidates: boolean;
  compactRendererSourceReady: boolean;
  incompatibleDenseConsumer: boolean;
}

/** All recurring consumers must complete their bind-group handoff before destroy. */
export function octreeDensePhiReleaseReady(state: OctreeDensePhiReleaseState): boolean {
  return state.directPagedTopology
    && state.surfacePagesBootstrapped
    && state.pagedProjectionGroupsActive
    && state.faceGroupsPageNative
    && state.surfaceAdapterPageNative
    && state.topologyUsesSurfaceCandidates
    && state.compactRendererSourceReady
    && !state.incompatibleDenseConsumer;
}

export interface OctreePressureCapacityPlan {
  rowCapacity: number;
  entryCapacity: number;
  pressureBytes: number;
  headerBytes: number;
  entryBytes: number;
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

export interface OctreeOwnerAllocationPlan {
  readonly cellCount: number;
  readonly allocatedBytes: number;
  readonly legacyDenseBytes: number;
  readonly savedBytes: number;
}

/**
 * One self-decodable word per finest cell. The queried cell plus dyadic size
 * determines the aligned origin, so no bounded coordinate packing is needed.
 */
export function planOctreeOwnerAllocation(cellCount: number): OctreeOwnerAllocationPlan {
  if (!Number.isSafeInteger(cellCount) || cellCount < 1) throw new Error("Octree owner cell count must be a positive integer");
  const allocatedBytes = cellCount * 4;
  const legacyDenseBytes = cellCount * 8;
  return { cellCount, allocatedBytes, legacyDenseBytes, savedBytes: legacyDenseBytes - allocatedBytes };
}

export interface OctreePackedOwner { readonly origin: readonly [number, number, number]; readonly size: number }

export function encodeOctreeOwnerWord(origin: readonly [number, number, number], size: number): number {
  if (![1, 2, 4, 8, 16, 32].includes(size)) throw new RangeError("Octree owner size must be dyadic from 1 through 32");
  if (origin.some((value) => !Number.isSafeInteger(value) || value < 0 || value > 0xffffffff || value % size !== 0)) {
    throw new RangeError("Octree owner origin must be an aligned unsigned 32-bit coordinate");
  }
  if (size === 1) return 0x8000_0000;
  const exponent = Math.log2(size);
  return exponent;
}

export function decodeOctreeOwnerWord(word: number, cell: readonly [number, number, number]): OctreePackedOwner {
  const packed = word >>> 0;
  if ((packed & 0x8000_0000) !== 0) return { origin: [...cell] as [number, number, number], size: 1 };
  const exponent = packed & 7;
  if (exponent < 1 || exponent > 5) return { origin: [...cell] as [number, number, number], size: 1 };
  const size = 1 << exponent;
  return { origin: cell.map((value) => Math.floor(value / size) * size) as [number, number, number], size };
}

export interface OctreeProjectionPipelineReachability {
  readonly leafSolver: "dense" | "compact" | "chebyshev" | "mgpcg" | "megakernel";
  readonly segmentedProjection: boolean;
  readonly extrapolationSweeps: number;
  readonly sparseExtrapolation: boolean;
  readonly hasDensePhiSnapshot: boolean;
}

/** Compile only entry points reachable from the immutable solver configuration. */
export function octreeProjectionPipelineRequired(
  entryPoint: string,
  config: OctreeProjectionPipelineReachability,
): boolean {
  if (entryPoint === "jacobi") return config.leafSolver === "dense";
  if (entryPoint === "iterateLeaves") return config.leafSolver === "compact";
  if (entryPoint === "iterateChebyshev") return config.leafSolver === "chebyshev";
  if (entryPoint === "solveLeaves") return config.leafSolver === "megakernel";
  if (entryPoint === "reduceResidualPartials" || entryPoint === "reduceResidualTotal") {
    return config.leafSolver === "chebyshev";
  }
  if (entryPoint === "project") return !config.segmentedProjection;
  if (entryPoint === "projectSmallLeaves" || entryPoint === "projectLeaves") return config.segmentedProjection;
  if (entryPoint === "extrapolateSeed" || entryPoint === "extrapolate") {
    return config.extrapolationSweeps > 0 && !config.sparseExtrapolation;
  }
  if (entryPoint === "extrapolateSeedSparse" || entryPoint === "extrapolateSparse") {
    return config.extrapolationSweeps > 0 && config.sparseExtrapolation;
  }
  if (entryPoint === "copyExtrapolatedSparse") {
    return config.extrapolationSweeps % 2 === 1 && config.sparseExtrapolation;
  }
  if (["markChangedTiles", "buildDirtyWorklist", "refreshSnapshotTiles", "refreshSnapshotDense"].includes(entryPoint)) {
    return config.hasDensePhiSnapshot;
  }
  return true;
}

export interface OctreeLeafFrontierAllocationPlan {
  cellCount: number;
  listCapacity: number;
  /** Exact-key row hash slots on compact authority; zero on dense compatibility. */
  hashCapacity: number;
  denseOriginMapBytes: number;
  rowMapBytes: number;
  allocatedBytes: number;
  denseCompatibilityBytes: number;
  savedBytes: number;
}

/**
 * Compact authority resolves a packed leaf origin through a bounded exact-key
 * hash. Unsupported compatibility modes retain the direct finest-cell map.
 */
export function planOctreeLeafFrontierAllocation(
  cellCount: number,
  rowCapacity: number,
  compactAuthority: boolean,
): OctreeLeafFrontierAllocationPlan {
  if (!Number.isSafeInteger(cellCount) || cellCount < 1) throw new Error("Octree frontier cell count must be a positive integer");
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) throw new Error("Octree frontier row capacity must be a positive integer");
  const listCapacity = compactAuthority ? Math.min(cellCount, rowCapacity) : cellCount;
  const denseOriginMapBytes = cellCount * 4;
  let hashCapacity = 0;
  if (compactAuthority) {
    hashCapacity = 1;
    // Keep load <= 75%. A 32-probe shader lookup then remains bounded and any
    // pathological cluster is reported as a fail-closed pressure overflow.
    const requested = Math.ceil(listCapacity * 4 / 3);
    while (hashCapacity < requested) hashCapacity *= 2;
  }
  const rowMapBytes = compactAuthority ? hashCapacity * 2 * 4 : denseOriginMapBytes;
  const allocatedBytes = (4 + 2 * listCapacity) * 4 + rowMapBytes;
  const denseCompatibilityBytes = (4 + 3 * cellCount) * 4;
  return {
    cellCount,
    listCapacity,
    hashCapacity,
    denseOriginMapBytes,
    rowMapBytes,
    allocatedBytes,
    denseCompatibilityBytes,
    savedBytes: denseCompatibilityBytes - allocatedBytes,
  };
}

export interface OctreeSolidCellAllocationPlan {
  allocatedBytes: number;
  denseBytes: number;
  savedBytes: number;
  hasDenseField: boolean;
}

export interface OctreePhiSnapshotAllocationPlan {
  allocatedBytes: number;
  denseBytes: number;
  savedBytes: number;
  hasDenseField: boolean;
}

/**
 * Paged surface authority invalidates change-driven topology through its
 * compact residency lists, so it does not retain a second dense phi field.
 */
export function planOctreePhiSnapshotAllocation(
  dims: { nx: number; ny: number; nz: number },
  adaptiveSurfaceAuthority: boolean,
): OctreePhiSnapshotAllocationPlan {
  const denseBytes = Math.max(4, dims.nx * dims.ny * dims.nz * 4);
  const hasDenseField = !adaptiveSurfaceAuthority;
  const allocatedBytes = hasDenseField ? denseBytes : 4;
  return { allocatedBytes, denseBytes, savedBytes: denseBytes - allocatedBytes, hasDenseField };
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
  // four-byte fine channel per sample and the sparse summary hash containing
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

/**
 * Physical single-interface narrow-band capacity, in global fine bricks.
 *
 * This is deliberately an area-times-width plan. Increasing all logical
 * dimensions while holding the physical brick-band width fixed grows the
 * reserve quadratically rather than materializing the cubic fine lattice.
 * `surfaceGrowthSafety` is explicit deformation/topology headroom; fixed-size
 * physical pages themselves do not incur allocator fragmentation.
 */
export function planGlobalFineNarrowBandBrickCapacity(
  brickDimensions: readonly [number, number, number],
  dilationBrickRings: number,
  surfaceGrowthSafety = 1.25,
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

/** Backward-compatible scalar form of the physical narrow-band plan. */
export function estimateGlobalFineNarrowBandBrickCapacity(
  brickDimensions: readonly [number, number, number],
  dilationBrickRings: number,
  surfaceGrowthSafety = 1.25,
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
 * Capacity for the compact pressure publication.  The interface contribution
 * scales with domain surface area, while the fully-coarse term covers the calm
 * bulk.  Overflow is detected on-GPU and fail-closed; this is a capacity, not
 * an assumption used by the numerical kernels.
 */
export function planOctreePressureCapacity(
  dims: { nx: number; ny: number; nz: number },
  maximumLeafSize: number,
  interfaceBandCells: number,
  override?: number,
  powerDiagramAuthority = false,
  closedTop = false,
): OctreePressureCapacityPlan {
  const count = dims.nx * dims.ny * dims.nz;
  const aligned = (value: number) => Math.ceil(value / 256) * 256;
  const surfaceArea = dims.nx * dims.ny + dims.nx * dims.nz + dims.ny * dims.nz;
  const surfaceRows = surfaceArea * Math.max(2, Math.ceil(interfaceBandCells) + 2);
  const coarseRows = 8 * Math.ceil(count / Math.max(1, maximumLeafSize ** 3));
  // Power authority currently uses the generated interior catalog.  Reserve
  // the exact closed-wall unit-strip upper bound in addition to the moving
  // interface bound; overlap only makes this conservative.  This prevents the
  // correctness strip from silently converting into a row-arena rollback.
  const wallRows = powerDiagramAuthority
    ? planOctreePowerBoundaryStrip(dims, interfaceBandCells, closedTop).unitCellUpperBound
    : 0;
  const requested = override === undefined ? surfaceRows + wallRows + coarseRows : override;
  const rowCapacity = Math.max(1, Math.min(count, aligned(Math.max(1, Math.floor(requested)))));
  // Do not charge the rollback/mirror modes for the larger generalized row.
  // An authoritative power solve selects a single shared arena sized to the
  // catalog's proven bound instead of retaining two authoritative operators.
  const entryCapacity = rowCapacity * (powerDiagramAuthority
    ? Math.max(6 * OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS,
      OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows)
    : 6 * OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS);
  return {
    rowCapacity,
    entryCapacity,
    pressureBytes: rowCapacity * 2 * 4,
    headerBytes: rowCapacity * 48,
    entryBytes: entryCapacity * 8,
  };
}

export interface OctreeCompactionAllocationPlan {
  scanBlockCapacity: number;
  coarseTaskCapacity: number;
  scanAndTaskBytes: number;
  activeTileBytes: number;
  allocatedBytes: number;
}

/**
 * Size the shared scan/task arena from the authorities that can actually
 * publish work. Compact pressure owns at most one frontier row and one
 * cooperative coarse task per pressure slot. Dense compatibility retains the
 * historical finest-lattice scan and independently padded 8-cell task tiles.
 * The resident active/retired tile list remains an independent lower bound
 * because it is copied into the same buffer before topology rebuilds.
 */
export function planOctreeCompactionAllocation(
  dims: { nx: number; ny: number; nz: number },
  pressureRowCapacity: number,
  activeTileWorklistBytes: number,
  activeTileCapacity: number,
  topologyTileSize: number,
  compactAuthority: boolean,
): OctreeCompactionAllocationPlan {
  const count = dims.nx * dims.ny * dims.nz;
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
  const scanBlockCapacity = Math.ceil((compactAuthority ? pressureRowCapacity : count) / 256);
  const coarseTasksPerTile = (topologyTileSize / 8) ** 3;
  const coarseTaskCapacity = compactAuthority
    ? Math.min(pressureRowCapacity, activeTileCapacity * coarseTasksPerTile)
    : Math.ceil(dims.nx / 8) * Math.ceil(dims.ny / 8) * Math.ceil(dims.nz / 8);
  const scanAndTaskBytes = 4 * (15 + 3 * scanBlockCapacity + 12 * 8 + 2 * coarseTaskCapacity);
  const activeTileBytes = 4 * 3 * activeTileCapacity + 32;
  const allocatedBytes = Math.max(60, scanAndTaskBytes, activeTileWorklistBytes) + activeTileBytes;
  return { scanBlockCapacity, coarseTaskCapacity, scanAndTaskBytes, activeTileBytes, allocatedBytes };
}

interface OctreeProjectionResources {
  velocityIn: GPUTexture;
  velocityOut: GPUTexture;
  velocityScratch: GPUTexture;
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
 * A GPU-resident, pressure-only octree projection.
 *
 * Ownership is paged on the supported compact path, while compatibility modes
 * retain one packed owner word per finest cell. Pressure exists only at live
 * leaf origins and resolves those rows through the compact frontier hash.
 */
export class WebGPUOctreeProjection {
  readonly preconditioner: "jacobi" | "additive-aggregate" | "section43-hybrid";
  readonly canEncodeInlineRebuild = true;
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
    pressureEntryCapacity: number;
    pressureRequiredRows?: number;
    pressureRequiredEntries?: number;
    pressureCapacityOverflow?: boolean;
    frontierListCapacity: number;
    frontierRequiredLeaves?: number;
    frontierCapacityOverflow?: boolean;
    velocityClampCount: number;
    factorLevelCount: number;
    multigridLevelCount: number;
    multigridCoarsestDofs: number;
    pressurePhaseTimings?: Record<string, number>;
    topologyReadbackBytes: number;
    topologyReused: boolean;
    topologyReuseCount: number;
    gpuConstruction_ms: number;
    gpuConstructionKernel_ms: number;
    gpuSparsePack_ms: number;
    cpuTopologyPack_ms: number;
    cpuRedistance_ms: number;
    cpuQuadtreeDecode_ms: number;
    cpuTallGrid_ms: number;
    cpuVariationalAssembly_ms: number;
    cpuSystemPack_ms: number;
    cpuICFactorization_ms: number;
    cpuResourceUpload_ms: number;
    powerDiagramProjection: "off" | "mirror" | "authoritative";
    powerDiagramReady: boolean;
    powerDiagramAuthoritative: boolean;
    powerDiagramFallbackReason?: string;
    powerDiagramAllocatedBytes: number;
    globalFineLevelSetAllocatedBytes: number;
    globalFineLevelSetResidentBrickCapacity: number;
    globalFineLevelSetLogicalBrickCount: number;
  };
  levelSetMismatchFraction = 0;
  relativeResidual?: number;
  residualRms?: number;
  initialResidualRms?: number;

  private readonly topology: GPUBuffer;
  private readonly ownerPages?: WebGPUOctreeSimulationOwnerPages;
  private readonly pressureA: GPUBuffer;
  private readonly pressureB: GPUBuffer;
  private readonly compaction: GPUBuffer;
  private readonly leafHeaders: GPUBuffer;
  private readonly leafEntries: GPUBuffer;
  private readonly leafFrontier: GPUBuffer;
  private readonly faceMirror?: WebGPUOctreeFaceMirror;
  private readonly faceTransport?: WebGPUOctreeFaceTransport;
  private readonly solidFaces?: WebGPUOctreeSolidFaces;
  private readonly adaptiveSurfaceAdapter?: WebGPUOctreeSurfaceAdapter;
  private readonly adaptiveSurfacePages?: WebGPUOctreeSurfacePages;
  private readonly globalFineLevelSet?: WebGPUFineLevelSetBricks;
  private readonly globalFineSeeds?: WebGPUFineLevelSetLeafSeeds;
  private globalFineTopologyAB?: WebGPUFineLevelSetTopology;
  private globalFineTopologyBA?: WebGPUFineLevelSetTopology;
  private readonly globalFineRedistanceA?: WebGPUFineLevelSetRedistance;
  private readonly globalFineRedistanceB?: WebGPUFineLevelSetRedistance;
  private globalFineVolumeA?: WebGPUFineLevelSetVolumeCorrection;
  private globalFineVolumeB?: WebGPUFineLevelSetVolumeCorrection;
  private globalFineVelocityPrepass?: WebGPUOctreePowerVelocityPrepass;
  private globalFineFaceFastMarch?: WebGPUOctreeFaceFastMarch;
  private globalFineTransportA?: WebGPUFineLevelSetTransport;
  private globalFineTransportB?: WebGPUFineLevelSetTransport;
  private lastGlobalFineTransport?: WebGPUFineLevelSetTransport;
  private readonly globalFineSummaries?: WebGPUFineLevelSetSummaries;
  private readonly fineSummaryFallback: GPUBuffer;
  private powerVelocity?: WebGPUOctreePowerVelocity;
  private powerCoarseLevelSet?: WebGPUOctreeCoarseLevelSet;
  private powerCoarseLevelSetSchedule?: WebGPUOctreePowerCoarseLevelSet;
  private fineToPowerCoarseLevelSet?: WebGPUFineToCoarseLevelSet;
  private powerCoarseLevelSetBootstrapped = false;
  private globalFineSourceA?: WebGPUFineLevelSetBrickSource;
  private globalFineSourceB?: WebGPUFineLevelSetBrickSource;
  private globalFineCurrentIsA = true;
  private globalFineBootstrapped = false;
  private globalFineGeneration = 2;
  private readonly faceRhsAuthority: boolean;
  private faceTransportDt_s = 0;
  private readonly solveDispatch: GPUBuffer;
  private readonly topologyCandidateDispatch: GPUBuffer;
  private readonly solidCells: GPUBuffer;
  private readonly hasDenseSolidCells: boolean;
  private readonly params: GPUBuffer;
  private readonly pagedParams?: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly shader: GPUShaderModule;
  private readonly diagnosticLayout: GPUBindGroupLayout;
  private readonly diagnosticPipelineLayout: GPUPipelineLayout;
  private readonly diagnosticShader: GPUShaderModule;
  private readonly couplingLayout: GPUBindGroupLayout;
  private readonly couplingPipelineLayout: GPUPipelineLayout;
  private readonly couplingShader: GPUShaderModule;
  private readonly surfaceState: WebGPUQuadtreeSurfaceState;
  private readonly sparseSurfaceBand?: WebGPUSparseSurfaceBand;
  private readonly sparseBrickWorld?: OctreeSparseBrickWorld;
  private readonly topologyResidency: GPUFluidBrickResidency;
  private readonly analyticBootstrapWorklist?: WebGPUOctreeAnalyticBootstrapWorklist;
  private sparseBrickWorldAccountedBytes = 0;
  private groups: { ab: GPUBindGroup; ba: GPUBindGroup; extrapolateOut: GPUBindGroup; extrapolateScratch: GPUBindGroup };
  private fineSummarySizingGroup: GPUBindGroup;
  private pagedGroups?: { ab: GPUBindGroup; ba: GPUBindGroup; extrapolateOut: GPUBindGroup; extrapolateScratch: GPUBindGroup };
  private surfacePagesBootstrapped = false;
  private denseBootstrapPhiReleased = false;
  private compactVoxelInspection?: CompactOctreeVoxelInspection;
  private readonly levelSetFallbackTexture?: GPUTexture;
  private topologyDiagnosticTexture?: GPUTexture;
  private pressureSamplesDiagnosticTexture?: GPUTexture;
  private pressureDiagnosticTexture?: GPUTexture;
  private divergenceDiagnosticTexture?: GPUTexture;
  private diagnosticGroups?: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
  private couplingGroups: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
  private rasterizeSolidsPipeline!: GPUComputePipeline;
  private resetPipeline!: GPUComputePipeline;
  private refinePipeline!: GPUComputePipeline;
  private balancePipeline!: GPUComputePipeline;
  private rasterizeSolidsActivePipeline!: GPUComputePipeline;
  private resetActivePipeline!: GPUComputePipeline;
  private refineActivePipeline!: GPUComputePipeline;
  private balanceActivePipeline!: GPUComputePipeline;
  private rasterizeSolidsRetiredPipeline!: GPUComputePipeline;
  private resetRetiredPipeline!: GPUComputePipeline;
  private refineRetiredPipeline!: GPUComputePipeline;
  private balanceRetiredPipeline!: GPUComputePipeline;
  private readonly refineLevelPipelines = new Map<number, {
    full: GPUComputePipeline;
    active: GPUComputePipeline;
    retired: GPUComputePipeline;
  }>();
  private readonly refineCoarsePipelines = new Map<number, {
    full: GPUComputePipeline; active: GPUComputePipeline; retired: GPUComputePipeline;
  }>();
  private readonly balanceCoarsePipelines = new Map<number, {
    full: GPUComputePipeline; active: GPUComputePipeline; retired: GPUComputePipeline;
  }>();
  private jacobiPipeline!: GPUComputePipeline;
  private planPipeline!: GPUComputePipeline;
  private scanPipeline!: GPUComputePipeline;
  private emitPipeline!: GPUComputePipeline;
  private beginFrontierPipeline!: GPUComputePipeline;
  private filterFrontierPipeline!: GPUComputePipeline;
  private appendFrontierPipeline!: GPUComputePipeline;
  private appendFrontierActivePipeline!: GPUComputePipeline;
  private appendFrontierRetiredPipeline!: GPUComputePipeline;
  private finalizeFrontierPipeline!: GPUComputePipeline;
  private assemblePipeline!: GPUComputePipeline;
  private assembleCoarsePipeline!: GPUComputePipeline;
  private gatherBodyCouplingPipeline!: GPUComputePipeline;
  private applyBodyCouplingPipeline!: GPUComputePipeline;
  private iteratePipeline!: GPUComputePipeline;
  private iterateChebyshevPipeline!: GPUComputePipeline;
  private reduceResidualPartialsPipeline!: GPUComputePipeline;
  private reduceResidualTotalPipeline!: GPUComputePipeline;
  private markChangedTilesPipeline!: GPUComputePipeline;
  private buildDirtyWorklistPipeline!: GPUComputePipeline;
  private refreshSnapshotTilesPipeline!: GPUComputePipeline;
  private refreshSnapshotDensePipeline!: GPUComputePipeline;
  private solvePipeline!: GPUComputePipeline;
  private pressureImpulsePipeline!: GPUComputePipeline;
  private reconstructSmallGradientsPipeline!: GPUComputePipeline;
  private reconstructGradientsPipeline!: GPUComputePipeline;
  private projectPipeline!: GPUComputePipeline;
  private projectSmallLeavesPipeline!: GPUComputePipeline;
  private projectLeavesPipeline!: GPUComputePipeline;
  private pressureOverflowPipeline!: GPUComputePipeline;
  private extrapolateSeedPipeline!: GPUComputePipeline;
  private extrapolatePipeline!: GPUComputePipeline;
  private extrapolateSeedSparsePipeline!: GPUComputePipeline;
  private extrapolateSparsePipeline!: GPUComputePipeline;
  private copyExtrapolatedSparsePipeline!: GPUComputePipeline;
  private materializePipeline!: GPUComputePipeline;
  private readonly maxLeafSize: number;
  private readonly topologyTileSize: number;
  private readonly adaptivity: number;
  private readonly interfaceRefinementBandCells: number;
  private readonly surfaceDetailStrength: number;
  private readonly iterations: number;
  private readonly extrapolationSweeps: number;
  private readonly sparseExtrapolationRequested: boolean;
  private readonly leafSolver: "dense" | "compact" | "chebyshev" | "mgpcg" | "megakernel";
  private mgpcg?: WebGPUOctreeMGPCG;
  private firstOrderVCycle?: WebGPUOctreeFirstOrderVCycle;
  private readonly pressureWarmStart: boolean;
  private readonly hydrostaticSplit: boolean;
  private readonly rowIndexedPressure: boolean;
  private readonly pressureCapacity: OctreePressureCapacityPlan;
  private readonly frontierAllocation: OctreeLeafFrontierAllocationPlan;
  private readonly workgroups: [number, number, number];
  private readonly linearBlocks: number;
  private readonly coarseTaskCapacity: number;
  private couplingHasDynamicBodies = false;
  private couplingBodyCount = 0;
  private compactionByteLength = 0;
  private solveStats!: GPUBuffer;
  private readonly phiSnapshotTexture: GPUTexture;
  /** Compact surface authority has no dense topology snapshot allocation. */
  private readonly hasDensePhiSnapshot: boolean;
  /** Encoded Chebyshev passes, adapted from async residual feedback. */
  private encodedSolvePasses = 0;
  private lastFeedbackRows = 0;
  private topologyWorklistReady = false;
  private latestPressureInA = true;
  private readonly pressureOverflowDispatch: GPUBuffer;
  /** Experimental page-to-topology authority; enabled after its long gate. */
  private readonly directPagedTopology: boolean;
  /** No dense phi exists; non-page topology groups must retain analytic sign until coarse correction publishes. */
  private readonly analyticSparseBootstrap: boolean;
  private pagedPhiDifferential?: WebGPUOctreePhiDifferential;
  private latestPagedPhiDifferential?: OctreePagedPhiDifferential;
  private readonly powerPolicy: OctreePowerProjectionPolicy;
  private powerDescriptor?: WebGPUOctreePowerDescriptor;
  private powerTopology?: WebGPUOctreePowerTopology;
  private powerFaces?: WebGPUOctreePowerFaces;
  private powerOperator?: WebGPUOctreePowerOperator;
  private powerFaceSeed?: WebGPUOctreePowerFaceSeed;
  private powerFaceTransfer?: WebGPUOctreePowerFaceTransfer;
  private powerSolidFaces?: WebGPUOctreePowerSolidFaces;
  private powerSolidVertices?: WebGPUOctreeSolidVertexSdf;
  private powerVolumes?: GPUBuffer;
  private powerVolumeParams?: GPUBuffer;
  private powerVolumePipeline?: GPUComputePipeline;
  private powerVolumeGroup?: GPUBindGroup;
  private powerGeneration = 0;
  private powerLifecycleDisposed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly scene: SceneDescription,
    private readonly dims: { nx: number; ny: number; nz: number },
    private readonly resources: OctreeProjectionResources,
    options: OctreeProjectionOptions,
    deferPipelineCompilation = false
  ) {
    const count = dims.nx * dims.ny * dims.nz;
    this.directPagedTopology = typeof process === "undefined"
      || process.env?.FLUID_OCTREE_DIRECT_PAGED_PHI !== "0";
    this.maxLeafSize = octreeLeafSize(options.maximumLeafSize ?? 16);
    this.adaptivity = Math.max(0, Math.min(1, options.adaptivity ?? 1));
    this.interfaceRefinementBandCells = Math.max(0, Math.min(32, Math.round(options.interfaceRefinementBandCells ?? 4)));
    this.surfaceDetailStrength = Math.max(0, Math.min(1, options.surfaceDetailStrength ?? 0));
    this.iterations = Math.max(8, Math.min(400, Math.round(options.pressureIterations)));
    const requestedExtrapolationSweeps = Math.max(0, Math.min(8, Math.round(options.extrapolationSweeps ?? 4)));
    // Kept opt-in until the widened-ocean A/B demonstrates that worklist
    // decoding beats the dense 4^3 dispatch for this short stencil ladder.
    this.sparseExtrapolationRequested = options.brickSparseExtrapolation ?? false;
    // The row-parallel Chebyshev path is the default: it retains occupancy but
    // replaces four dispatch-bound Jacobi sweeps with one polynomial pass.
    // The single-workgroup megakernel is kept for very small/calm A/B cases.
    const requested = options.leafSolver ?? "auto";
    this.pressureWarmStart = options.pressureWarmStart ?? true;
    // A fixed rest-surface decomposition is well-defined for tank fills. A
    // dam-break reservoir has no domain-wide hydrostatic datum, so fail closed
    // to the absolute-pressure path even if an experimental flag is supplied.
    this.hydrostaticSplit = options.hydrostaticSplit === true
      && scene.fluid.initialCondition === "tank-fill"
      && scene.fluid.inflow === undefined
      && scene.rigidBodies.length === 0;
    // Power authority requires a transferred compact velocity source for
    // topology changes. Treat that dependency as part of the authority mode,
    // rather than requiring a second hidden feature switch from callers.
    const hasTerrain = sceneHasTerrain(scene);
    // Terrain authority owns a generation-tagged sparse vertex-SDF stage and
    // retains the same compact Cartesian seed used by every cold/new power
    // face. Runtime validation still suppresses publication if either input is
    // absent or stale; this flag records construction support only.
    const terrainEmbeddedBoundaryAvailable = hasTerrain
      && options.powerDiagramProjection === "authoritative";
    const faceTransportRequested = options.faceVelocityTransport === true
      || options.powerDiagramProjection === "authoritative";
    const adaptiveSolidFaces = scene.rigidBodies.length > 0 && !sceneHasTerrain(scene);
    this.faceRhsAuthority = adaptiveFaceRhsIsSupported(
      options.faceVelocityRhs === true || faceTransportRequested,
      sceneHasTerrain(scene),
      scene.rigidBodies.length,
      this.hydrostaticSplit,
      adaptiveSolidFaces,
      terrainEmbeddedBoundaryAvailable,
    );
    // A default transport request must not reserve the face store in a scene
    // whose boundary operators cannot consume it. Explicit mirror/RHS flags
    // remain available as observational A/B modes on those scenes.
    const faceTransportEnabled = faceTransportRequested && this.faceRhsAuthority;
    // Canonical face transport reconstructs velocity from bounded incidence
    // directly. The legacy texture extrapolator is both redundant and a
    // finest-box dispatch against 1x1 compatibility textures after cutover.
    this.extrapolationSweeps = faceTransportEnabled ? 0 : requestedExtrapolationSweeps;
    // Compact face transport and paged phi are one authority cutover. Keeping
    // the dense publication texture is transitional; its three transport
    // copies and two jump-flood seed arenas have no work once pages own phi.
    const surfacePagesRequested = faceTransportEnabled
      && (typeof process === "undefined" || process.env?.FLUID_OCTREE_SURFACE_PAGES !== "0");
    const ownerPageOverride = typeof process === "undefined" ? undefined : process.env?.FLUID_OCTREE_OWNER_PAGES;
    const ownerPagesRequested = (surfacePagesRequested || ownerPageOverride === "1") && ownerPageOverride !== "0";
    // Analytic dam/tank scenes can construct compact topology and first-page
    // phi without allocating or uploading a box-sized bootstrap texture.
    // Explicit seeded/imported shapes remain on the compatibility staging path
    // until their bounded sparse voxelizer is available.
    const analyticSparseBootstrap = surfacePagesRequested && this.directPagedTopology
      && (scene.fluid.initialBrickSeeds_m?.length ?? 0) === 0
      && scene.rigidBodies.length === 0 && !sceneHasTerrain(scene);
    this.analyticSparseBootstrap = analyticSparseBootstrap;
    // Change-driven rebuild compares every cell in a topology tile against a
    // dense phi snapshot. Paged phi already bounds rebuild work by its active
    // topology/residency lists, so retaining another box-sized r32 field is a
    // net loss on the large scenes this path targets. Bind one format-correct
    // texel for layout compatibility and keep the snapshot kernels disabled.
    const phiSnapshotAllocation = planOctreePhiSnapshotAllocation(dims, surfacePagesRequested);
    this.hasDensePhiSnapshot = phiSnapshotAllocation.hasDenseField;
    const surfaceStateAllocation = planOctreeSurfaceStateAllocation(
      [dims.nx, dims.ny, dims.nz], surfacePagesRequested,
      surfacePagesRequested && this.directPagedTopology
        && scene.rigidBodies.length === 0 && !sceneHasTerrain(scene),
      analyticSparseBootstrap,
    );
    const cell = {
      x: scene.container.width_m / dims.nx,
      y: scene.container.height_m / dims.ny,
      z: scene.container.depth_m / dims.nz
    };
    this.powerPolicy = resolveOctreePowerProjectionPolicy(options.powerDiagramProjection,
      [cell.x, cell.y, cell.z], sceneHasTerrain(scene), scene.rigidBodies.length, faceTransportEnabled,
      !sceneHasTerrain(scene) || terrainEmbeddedBoundaryAvailable,
      (scene.fluid.initialBrickSeeds_m?.length ?? 0) > 0,
      terrainEmbeddedBoundaryAvailable,
      terrainEmbeddedBoundaryAvailable && faceTransportEnabled);
    this.leafSolver = requested === "auto"
      ? (this.powerPolicy.authoritative ? "mgpcg" : "chebyshev")
      : requested;
    this.preconditioner = this.leafSolver === "mgpcg"
      ? (this.powerPolicy.authoritative ? "section43-hybrid" : "additive-aggregate")
      : "jacobi";
    this.rowIndexedPressure = this.leafSolver !== "dense";
    this.encodedSolvePasses = Math.ceil(this.iterations / 4);
    this.pressureCapacity = planOctreePressureCapacity(
      dims, this.maxLeafSize, this.interfaceRefinementBandCells,
      options.pressureRowCapacity,
      this.powerPolicy.authoritative,
      scene.container.top === "closed",
    );
    const adaptiveSurfacePageFraction = options.sparseSurfacePageFraction
      ?? OCTREE_UNIFIED_SURFACE_RESIDENT_FRACTION;
    if (!Number.isFinite(adaptiveSurfacePageFraction)
      || adaptiveSurfacePageFraction <= 0 || adaptiveSurfacePageFraction > 1) {
      throw new RangeError("Octree surface page fraction must be in (0, 1]");
    }
    const compactFrontierAuthority = faceTransportEnabled && this.rowIndexedPressure;
    this.frontierAllocation = planOctreeLeafFrontierAllocation(
      count,
      this.pressureCapacity.rowCapacity,
      compactFrontierAuthority,
    );
    this.linearBlocks = Math.ceil(this.frontierAllocation.listCapacity / 256);
    // Open ocean scenes have no solid fraction to publish. Keep a single
    // zero-initialized record so every bind group remains valid; shader-side
    // bounds checks make all logical cells read as `{0,-1}` and rasterization
    // is skipped. Terrain/body scenes retain the dense VOS field.
    const solidCellAllocation = planOctreeSolidCellAllocation(dims, sceneHasTerrain(scene), scene.rigidBodies.length);
    this.hasDenseSolidCells = solidCellAllocation.hasDenseField;
    this.solidCells = device.createBuffer({
      label: this.hasDenseSolidCells ? "Octree VOS solid fractions and owners" : "Octree zero-solid fallback",
      size: solidCellAllocation.allocatedBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (!this.hasDenseSolidCells) device.queue.writeBuffer(this.solidCells, 0, new Int32Array([0, -1]));
    // Build residency before the surface state so phi transport can consume
    // the previous publication's active-brick worklist directly. The t=0
    // publication is encoded before the first advance, so the first dynamic
    // surface pass is sparse as well.
    const topologyHaloCells = this.interfaceRefinementBandCells
      + 8 * this.surfaceDetailStrength;
    this.topologyTileSize = Math.max(8, this.maxLeafSize);
    const compactAtlasDiagnostic = typeof process !== "undefined"
      && process.env?.FLUID_OCTREE_COMPACT_BRICK_ATLAS === "1";
    const sparseWorldCompatibilityRequested = typeof process !== "undefined"
      && process.env?.FLUID_OCTREE_SPARSE_WORLD === "1";
    const allocateSparseWorld = octreeSparseWorldRequired(
      this.directPagedTopology, surfacePagesRequested, sceneHasTerrain(scene), scene.rigidBodies.length,
      sparseWorldCompatibilityRequested,
    );
    if (allocateSparseWorld) this.sparseBrickWorld = new OctreeSparseBrickWorld(device, scene, [dims.nx, dims.ny, dims.nz], {
      brickSize: 8,
      haloCells: topologyHaloCells,
      // Canonical faces/pages own the simulation fields. Retain only the wet
      // bulk worklist needed by owner-page lifecycle, not a duplicate 3D
      // phi/velocity atlas. The env switch is an explicit parity diagnostic.
      brickAtlas: faceTransportEnabled && !compactAtlasDiagnostic ? "off" : options.brickAtlas ?? "mirror",
      bulkResidencyOnly: faceTransportEnabled,
      brickPreActivation: options.brickPreActivation ?? true,
      topologyTileBricks: this.topologyTileSize / 8
    });
    const analyticBootstrapPlan = analyticSparseBootstrap ? planOctreeAnalyticBootstrapBounds({
      dimensions: [dims.nx, dims.ny, dims.nz],
      containerSize: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
      tileSizeCells: this.topologyTileSize,
      initialCondition: scene.fluid.initialCondition,
      fillFraction: scene.container.fillFraction,
      interfaceBandCells: this.interfaceRefinementBandCells,
      surfaceDetailStrength: this.surfaceDetailStrength,
    }) : undefined;
    const schedulerBrickDimensions = [dims.nx, dims.ny, dims.nz]
      .map((value) => Math.ceil(value / 8)) as [number, number, number];
    const schedulerTileBricks = this.topologyTileSize / 8;
    const schedulerTileDimensions = schedulerBrickDimensions
      .map((value) => Math.ceil(value / schedulerTileBricks)) as [number, number, number];
    const sparseSchedulerPools = !allocateSparseWorld ? planSurfaceCandidateResidencyPools(
      schedulerBrickDimensions,
      schedulerTileDimensions,
      8,
      this.interfaceRefinementBandCells,
      this.pressureCapacity.rowCapacity,
      analyticBootstrapPlan?.activeTileCount ?? 1,
    ) : undefined;
    this.topologyResidency = this.sparseBrickWorld?.topologyResidency ?? new GPUFluidBrickResidency(
      device, [dims.nx, dims.ny, dims.nz], [cell.x, cell.y, cell.z], {
        brickSize: 8, haloCells: topologyHaloCells, retireAfterFrames: 3,
        topologyTileBricks: this.topologyTileSize / 8,
        // Direct page candidates never consume the legacy sparse-world leaf
        // mirror. Keep only format-valid sentinel words for those bindings.
        surfaceCandidatesOnly: true,
        surfaceCandidateBrickCapacity: sparseSchedulerPools?.brickCapacity,
        surfaceCandidateTileCapacity: sparseSchedulerPools?.tileCapacity,
      },
    );
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
      );
    }
    this.surfaceState = new WebGPUQuadtreeSurfaceState(
      device, dims, cell, resources.velocityOut,
      analyticSparseBootstrap
        ? new Float32Array([Math.max(cell.x, cell.y, cell.z) * this.maxLeafSize])
        : initialOctreeLevelSet(scene, dims, cell), undefined,
      undefined, false, false, true, true, this.hasDenseSolidCells ? this.solidCells : undefined, options.brickSparseSurface === false ? undefined : {
        worklist: this.topologyResidency.worklist,
        states: this.topologyResidency.stateBuffer,
        brickSize: 8
      }, surfacePagesRequested, analyticSparseBootstrap
    );
    // Compact face transport owns the leaf-attached surface-page arena below;
    // do not allocate the older dense-velocity sparse-band mirror as well.
    const sparseSurfaceMode = faceTransportEnabled ? "off" : options.sparseSurfaceBand ?? "off";
    if (sparseSurfaceMode !== "off") {
      this.sparseSurfaceBand = new WebGPUSparseSurfaceBand(
        device,
        [dims.nx, dims.ny, dims.nz],
        [cell.x, cell.y, cell.z],
        this.surfaceState.texture,
        resources.velocityOut,
        this.solidCells,
        {
          mode: sparseSurfaceMode,
          refinementFactor: options.surfaceRefinementFactor ?? 2,
          bandCells: options.sparseSurfaceBandCells ?? 4,
          stencilCells: 5,
          retireAfterFrames: 3,
          maximumResidentFraction: options.sparseSurfacePageFraction ?? 0.75,
        },
      );
    }
    // COPY_SRC on the packed owner lattice and pressure iterates exists solely for test
    // readbacks (leaf-size census, 2:1 balance, and finiteness audits); the
    // simulation itself never copies them out.
    const ownerAllocation = planOctreeOwnerAllocation(count);
    if (ownerPagesRequested) {
      this.ownerPages = new WebGPUOctreeSimulationOwnerPages(
        device, [dims.nx, dims.ny, dims.nz],
        this.sparseBrickWorld?.bulkResidencyWorklist ?? this.topologyResidency.worklist,
        surfacePagesRequested && this.rowIndexedPressure ? {
          // Derive physical owner storage from the same bounded adaptive
          // authorities that can request it. Arena overflow is already part
          // of topologyOverflow(); missing pages decode as canonical coarse
          // owners instead of reading outside the physical payload.
          adaptiveBounds: {
            pressureRowCapacity: this.pressureCapacity.rowCapacity,
            surfacePageCapacity: Math.max(1,
              Math.ceil(this.pressureCapacity.rowCapacity * adaptiveSurfacePageFraction)),
          },
        } : {
          // Explicit compatibility overrides have no compact surface bound.
          maximumResidentFraction: 0.90,
        },
        this.analyticBootstrapWorklist ? {
          tileWorklist: this.topologyResidency.tileWorklist,
          tileSizeCells: this.analyticBootstrapWorklist.plan.tileSizeCells,
          activeTileLimits: this.analyticBootstrapWorklist.plan.activeTileLimits,
          activeTileCount: this.analyticBootstrapWorklist.plan.activeTileCount,
        } : undefined,
      );
      this.topology = this.ownerPages.arena;
    } else {
      this.topology = device.createBuffer({ label: "Octree packed owner lattice", size: Math.max(4, ownerAllocation.allocatedBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    }
    const pressureSlots = this.rowIndexedPressure ? this.pressureCapacity.rowCapacity : count;
    this.pressureA = device.createBuffer({ label: "Octree leaf pressure A", size: Math.max(4, pressureSlots * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.pressureB = device.createBuffer({ label: "Octree leaf pressure B", size: Math.max(4, pressureSlots * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const compactBuffers = this.leafSolver !== "dense";
    // The scan totals are dead after leaf emission. The tail then doubles as
    // twelve resident rank-six generalized body-response vectors, avoiding a
    // ninth storage binding on minimum-limit WebGPU devices.
    // Change-driven rebuild state (per-tile change flags, dirty marks, and
    // the compacted dirty list) occupies an exclusive additive tail so the
    // per-solve scan partials can never clobber it, followed by the 8-byte
    // Chebyshev residual feedback staged out via solveStats.
    const tileCapacity = this.topologyResidency.tileCapacity;
    const compactionAllocation = planOctreeCompactionAllocation(
      dims,
      this.pressureCapacity.rowCapacity,
      this.topologyResidency.tileWorklistByteLength,
      tileCapacity,
      this.topologyTileSize,
      compactFrontierAuthority,
    );
    this.coarseTaskCapacity = compactionAllocation.coarseTaskCapacity;
    this.compactionByteLength = compactionAllocation.allocatedBytes;
    this.phiSnapshotTexture = device.createTexture({
      label: this.hasDensePhiSnapshot ? "Octree topology phi snapshot" : "Octree topology phi snapshot fallback",
      size: this.hasDensePhiSnapshot ? [dims.nx, dims.ny, dims.nz] : [1, 1, 1],
      dimension: "3d",
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING
    });
    this.compaction = device.createBuffer({
      label: "Octree leaf compaction, body coupling, and resident topology worklist",
      size: this.compactionByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    // The common projection layout is constructed before the optional global
    // fine hierarchy.  An unpublished 64-byte directory keeps the binding
    // valid for non-fine configurations; it can never authorize coarsening.
    this.fineSummaryFallback = device.createBuffer({ label: "Unpublished fine-summary directory",
      size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.fineSummaryFallback, 0, new Uint32Array(16));
    // Copy-only staging keeps solve feedback readable without racing the next
    // rebuild's worklist copy and without a ninth storage binding.
    this.solveStats = device.createBuffer({
      label: "Octree solve feedback staging",
      size: 32,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.leafHeaders = device.createBuffer({ label: "Octree leaf row headers", size: compactBuffers ? Math.max(48, this.pressureCapacity.headerBytes) : Math.max(48, count * 48), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.leafEntries = device.createBuffer({ label: "Octree leaf matrix entries", size: compactBuffers ? Math.max(8, this.pressureCapacity.entryBytes) : Math.max(8, count * 48), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    // Header + capacity-bounded ping-pong leaf-origin lists + an exact-key
    // origin-to-row hash. Compatibility solvers retain the dense direct map.
    this.leafFrontier = device.createBuffer({ label: "Persistent octree leaf frontier", size: this.frontierAllocation.allocatedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (this.leafSolver === "mgpcg") {
      if (this.powerPolicy.authoritative) {
        this.firstOrderVCycle = new WebGPUOctreeFirstOrderVCycle(device, {
          leafHeaders: this.leafHeaders,
          leafEntries: this.leafEntries,
        }, {
          dimensions: [dims.nx, dims.ny, dims.nz],
          rowCapacity: this.pressureCapacity.rowCapacity,
          finestCellWidth: cell.x,
        });
      }
      this.mgpcg = new WebGPUOctreeMGPCG(device, {
        leafHeaders: this.leafHeaders,
        leafEntries: this.leafEntries,
        rowCount: this.compaction,
        firstOrderVCycle: this.firstOrderVCycle,
      }, {
        dimensions: [dims.nx, dims.ny, dims.nz],
        rowCapacity: this.pressureCapacity.rowCapacity,
        maximumLeafSize: this.maxLeafSize,
        maximumIterations: this.iterations,
        relativeTolerance: scene.numerics.pressureRelativeTolerance,
        preconditionerKind: this.powerPolicy.authoritative ? "section43-hybrid" : "aggregate",
      });
    }
    // Words 8..15 hold one-workgroup-per-tile coarse topology dispatches: the
    // per-frame copies refresh only the x counts, so y/z stay 1 from here.
    this.solveDispatch = device.createBuffer({ label: "Octree leaf solve and retired-topology dispatch", size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    device.queue.writeBuffer(this.solveDispatch, 32, new Uint32Array([0, 1, 1, 0, 0, 1, 1, 0]));
    this.topologyCandidateDispatch = device.createBuffer({ label: "Octree topology candidate-origin dispatch", size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.pressureOverflowDispatch = device.createBuffer({ label: "Octree pressure overflow fail-closed dispatch", size: 12, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.params = device.createBuffer({ label: "Octree projection parameters", size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // These bindings exist solely for the experimental direct topology
    // sampler. The stable compact-page path must not pay their allocation or
    // bind-group cost while the differential gate remains disabled.
    if (surfacePagesRequested && (this.directPagedTopology || options.globalFineLevelSetFactor !== undefined)) {
      this.pagedParams = device.createBuffer({ label: "Octree paged-phi projection parameters", size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.levelSetFallbackTexture = device.createTexture({
        label: "Octree paged-phi sampled fallback", size: [1, 1, 1], dimension: "3d", format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
    }
    this.layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      // Both pressure buffers are writable so the persistent megakernel can
      // ping-pong iterates inside a single dispatch.
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } }
      ,{ binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
      ,{ binding: 14, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "read-write", format: "r32float", viewDimension: "3d" } }
      ,{ binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.shader = device.createShaderModule({ label: "GPU-resident octree projection", code: octreeProjectionShader });
    this.groups = {
      ab: this.createProjectionGroup(resources.velocityIn, resources.velocityOut, this.pressureA, this.pressureB),
      ba: this.createProjectionGroup(resources.velocityIn, resources.velocityOut, this.pressureB, this.pressureA),
      extrapolateOut: this.createProjectionGroup(resources.velocityOut, resources.velocityScratch, this.pressureA, this.pressureB),
      extrapolateScratch: this.createProjectionGroup(resources.velocityScratch, resources.velocityOut, this.pressureA, this.pressureB)
    };
    this.fineSummarySizingGroup = this.createProjectionGroup(
      resources.velocityIn, resources.velocityOut, this.fineSummaryFallback, this.pressureB);
    this.diagnosticLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32uint", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    this.diagnosticPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.diagnosticLayout] });
    this.diagnosticShader = device.createShaderModule({ label: "GPU octree overlay materialization", code: octreeDiagnosticShader });
    this.couplingLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    this.couplingPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.couplingLayout] });
    this.couplingShader = device.createShaderModule({ label: "GPU octree pressure-to-body coupling", code: octreePressureCouplingShader });
    this.couplingGroups = {
      pressureA: this.createCouplingGroup(this.pressureA, this.levelSetTexture),
      pressureB: this.createCouplingGroup(this.pressureB, this.levelSetTexture),
    };
    this.workgroups = [Math.ceil(dims.nx / 4), Math.ceil(dims.ny / 4), Math.ceil(dims.nz / 4)];
    const fullyCoarseEstimate = Math.ceil(count / (this.maxLeafSize ** 3));
    const approximateLeaves = Math.ceil(count * (1 - this.adaptivity) + fullyCoarseEstimate * this.adaptivity);
    const initialSolvePasses = this.leafSolver === "chebyshev" ? Math.ceil(this.iterations / 4) : this.iterations;
    this.info = {
      leafCount: approximateLeaves, pressureSampleCount: approximateLeaves, liquidDofCount: approximateLeaves,
      faceCount: 0, mlsProjectionRowCount: 0, tallSegmentCount: 0, ghostFaceCount: 0,
      maximumNeighborRatio: 2, maximumFluidScale: this.maxLeafSize, compressionRatio: approximateLeaves / Math.max(1, count),
      allocatedBytes: this.topology.size + (this.ownerPages ? 32 : 0) + this.solidCells.size + phiSnapshotAllocation.allocatedBytes + surfaceStateAllocation.allocatedBytes
        + this.pressureA.size + this.pressureB.size + this.leafHeaders.size + this.leafEntries.size
        + this.leafFrontier.size + this.compaction.size + this.fineSummaryFallback.size + 192
        + (this.mgpcg?.plan.allocatedBytes ?? 0)
        + (this.firstOrderVCycle?.allocatedBytes ?? 0)
        + (this.sparseBrickWorld?.allocatedBytes ?? this.topologyResidency.allocatedBytes) + (this.sparseSurfaceBand?.allocatedBytes ?? 0)
        + (this.analyticBootstrapWorklist?.allocatedBytes ?? 0),
      pressureIterationsUsed: initialSolvePasses, pressureIterationBudget: initialSolvePasses,
      pressureIterationHardBudget: initialSolvePasses, pressureConverged: undefined,
      pressureRowCapacity: pressureSlots, pressureEntryCapacity: this.rowIndexedPressure ? this.pressureCapacity.entryCapacity : count * 6,
      pressureCapacityOverflow: false,
      frontierListCapacity: this.frontierAllocation.listCapacity,
      frontierCapacityOverflow: false,
      velocityClampCount: 0,
      // These legacy telemetry fields describe actual V-cycle state. The
      // experimental aggregate-PCG hierarchy is reported in pressureSolverLabel
      // and must not masquerade as multigrid levels/coarsest DOFs.
      factorLevelCount: this.firstOrderVCycle?.plan.levelCount ?? 0,
      multigridLevelCount: this.firstOrderVCycle?.plan.levelCount ?? 0,
      multigridCoarsestDofs: 0, topologyReadbackBytes: 0,
      topologyReused: false, topologyReuseCount: 0, gpuConstruction_ms: 0, gpuConstructionKernel_ms: 0,
      gpuSparsePack_ms: 0, cpuTopologyPack_ms: 0, cpuRedistance_ms: 0, cpuQuadtreeDecode_ms: 0,
      cpuTallGrid_ms: 0, cpuVariationalAssembly_ms: 0, cpuSystemPack_ms: 0,
      cpuICFactorization_ms: 0, cpuResourceUpload_ms: 0,
      powerDiagramProjection: this.powerPolicy.requested,
      powerDiagramReady: false,
      powerDiagramAuthoritative: false,
      powerDiagramFallbackReason: this.powerPolicy.fallbackReason,
      powerDiagramAllocatedBytes: 0,
      globalFineLevelSetAllocatedBytes: 0,
      globalFineLevelSetResidentBrickCapacity: 0,
      globalFineLevelSetLogicalBrickCount: 0,
    };
    // An unsupported authority request still builds the observational mirror;
    // only the RHS replacement is suppressed.
    if (options.faceVelocityMirror || options.faceVelocityRhs || faceTransportEnabled) {
      this.faceMirror = new WebGPUOctreeFaceMirror(device, {
        velocity: resources.velocityIn,
        levelSet: this.levelSetTexture,
        owners: this.topology,
        leafHeaders: this.leafHeaders,
        frontier: this.leafFrontier,
        compaction: this.compaction,
        params: this.params,
        pressureA: this.pressureA,
        pressureB: this.pressureB,
        projectedVelocity: resources.velocityOut,
      }, this.pressureCapacity.rowCapacity, {
        preserveTopologyVelocities: faceTransportEnabled
          && (typeof process === "undefined" || process.env?.FLUID_OCTREE_FACE_TRANSFER !== "0"),
      });
      this.info.allocatedBytes += this.faceMirror.plan.allocatedBytes;
      if (this.faceRhsAuthority && adaptiveSolidFaces) {
        this.solidFaces = new WebGPUOctreeSolidFaces(device, {
          faces: this.faceMirror.source,
          rigidBodies: resources.rigidBodies,
          rigidExchange: resources.rigidExchange,
          params: this.params,
          pressureA: this.pressureA,
          pressureB: this.pressureB,
        });
        this.info.allocatedBytes += this.solidFaces.plan.allocatedBytes;
      }
      if (faceTransportEnabled) {
        this.faceTransport = new WebGPUOctreeFaceTransport(device, this.faceMirror.source, [
          scene.container.width_m / this.dims.nx,
          scene.container.height_m / this.dims.ny,
          scene.container.depth_m / this.dims.nz,
        ], [this.dims.nx, this.dims.ny, this.dims.nz]);
        this.info.allocatedBytes += this.faceTransport.plan.allocatedBytes;
        if (surfacePagesRequested) this.adaptiveSurfaceAdapter = new WebGPUOctreeSurfaceAdapter(device, {
          leafHeaders: this.leafHeaders,
          rowCount: this.compaction,
          publicationControl: this.compaction,
          frontier: this.leafFrontier,
          levelSet: this.levelSetTexture,
          dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
          cellSize: [cell.x, cell.y, cell.z],
        }, this.faceMirror.source, this.pressureCapacity.rowCapacity, {
          // Global fine bricks are keyed independently of octree resolution.
          // Every live core/halo leaf may therefore seed the narrow band; a
          // coarse interface leaf must not be discarded merely because its
          // pressure degree of freedom spans more than one finest cell.
          finestLeafSize: this.maxLeafSize,
          haloCells: this.interfaceRefinementBandCells,
          directPageSampling: this.directPagedTopology,
          ...(analyticSparseBootstrap ? {
            analyticInitialCondition: scene.fluid.initialCondition,
            initialFillFraction: scene.container.fillFraction,
          } : {}),
        });
        if (this.adaptiveSurfaceAdapter) this.adaptiveSurfacePages = new WebGPUOctreeSurfacePages(
          device,
          this.adaptiveSurfaceAdapter.source,
          this.pressureCapacity.rowCapacity,
          [this.dims.nx, this.dims.ny, this.dims.nz],
          [cell.x, cell.y, cell.z],
          // Dedicated row-attached authority budget. Direct paged topology
          // publishes the complete compact core/halo set, so use the same
          // bounded fraction that sized owner-page support above. Overflow
          // remains an explicit fail-closed diagnostic.
          {
            maximumResidentFraction: adaptiveSurfacePageFraction,
            // 2^3 is the bandwidth-oriented default; explicit factor-4
            // refinement retains the higher-resolution page ABI.
            pageResolution: options.surfaceRefinementFactor === 4 ? 4 : 2,
            airHaloCells: Math.max(1, Math.min(8, this.interfaceRefinementBandCells)),
          },
          { texture: this.levelSetTexture, dimensions: [this.dims.nx, this.dims.ny, this.dims.nz] },
        );
        if (this.adaptiveSurfaceAdapter && this.adaptiveSurfacePages) this.info.allocatedBytes += this.adaptiveSurfaceAdapter.plan.allocatedBytes
          + this.adaptiveSurfacePages.allocatedBytes;
        if (this.directPagedTopology && this.adaptiveSurfaceAdapter && this.adaptiveSurfacePages) {
          this.faceMirror.setSurfacePageSource(this.adaptiveSurfacePages.source);
          this.adaptiveSurfaceAdapter.setSurfacePageSource(this.adaptiveSurfacePages.source);
          const paged = { surfaceData: this.adaptiveSurfacePages.source.arena.buffer,
            surfaceIndex: this.adaptiveSurfaceAdapter.leaves, params: this.pagedParams! };
          this.pagedGroups = {
            ab: this.createProjectionGroup(resources.velocityIn, resources.velocityOut, this.pressureA, this.pressureB, paged),
            ba: this.createProjectionGroup(resources.velocityIn, resources.velocityOut, this.pressureB, this.pressureA, paged),
            extrapolateOut: this.createProjectionGroup(resources.velocityOut, resources.velocityScratch, this.pressureA, this.pressureB, paged),
            extrapolateScratch: this.createProjectionGroup(resources.velocityScratch, resources.velocityOut, this.pressureA, this.pressureB, paged),
          };
          if (typeof process !== "undefined" && process.env?.FLUID_OCTREE_PHI_DIAGNOSTIC === "1") {
            this.pagedPhiDifferential = new WebGPUOctreePhiDifferential(
              device, this.levelSetTexture, this.adaptiveSurfacePages.source,
              this.adaptiveSurfaceAdapter.leaves, this.compaction,
              [this.dims.nx, this.dims.ny, this.dims.nz], this.topologyTileSize,
            );
          }
        }
        const globalFineFactor = options.globalFineLevelSetFactor;
        if (globalFineFactor !== undefined) {
          if (!this.adaptiveSurfaceAdapter || !this.adaptiveSurfacePages) {
            throw new RangeError("Global fine level-set mirror requires compact surface adapter/pages");
          }
          const minimumCell = Math.min(cell.x, cell.y, cell.z);
          const maximumCell = Math.max(cell.x, cell.y, cell.z);
          if (maximumCell - minimumCell > 1e-5 * maximumCell) {
            throw new RangeError("Global fine level-set mirror currently requires isotropic finest octree cells");
          }
          const brickResolution = 4 as const;
          const brickDimensions = [dims.nx, dims.ny, dims.nz]
            .map((value) => Math.ceil(value * globalFineFactor / brickResolution)) as [number, number, number];
          const logicalBrickCount = brickDimensions.reduce((product, value) => product * value, 1);
          const legacyCapacity = Math.ceil(this.adaptiveSurfacePages.plan.pageCapacity
            * (globalFineFactor / brickResolution) ** 3 * 2);
          const redistanceBandFineCells = Math.min(256, Math.max(4,
            this.interfaceRefinementBandCells * globalFineFactor));
          const physicalBand = planFineLevelSetTopologyBand(brickResolution, {
            maximumBacktraceFineCells: globalFineFactor,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells,
            safetyBrickRings: 1,
          });
          const defaultCapacity = Math.max(legacyCapacity,
            estimateGlobalFineNarrowBandBrickCapacity(brickDimensions, physicalBand.dilationBrickRings));
          const requestedCapacity = Math.min(logicalBrickCount,
            options.globalFineLevelSetMaximumBricks ?? defaultCapacity);
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
            defaultCapacity, options.globalFineLevelSetMaximumBricks, kernelBrickLimit, 64,
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
          this.globalFineSeeds = new WebGPUFineLevelSetLeafSeeds(device, this.globalFineSourceB);
          this.globalFineRedistanceA = new WebGPUFineLevelSetRedistance(device, this.globalFineSourceA);
          this.globalFineRedistanceB = new WebGPUFineLevelSetRedistance(device, this.globalFineSourceB);
          this.globalFineSummaries = new WebGPUFineLevelSetSummaries(device, globalPlan,
            this.pressureCapacity.rowCapacity);
          // The common layout is already at WebGPU's portable ten-storage-
          // buffer limit. Refinement reuses pressure binding 4 for this raw
          // read-only directory instead of adding an eleventh binding.
          this.fineSummarySizingGroup = this.createProjectionGroup(
            resources.velocityIn, resources.velocityOut, this.globalFineSummaries.directory, this.pressureB);
          const allocated = this.globalFineLevelSet.allocatedBytes + this.globalFineSeeds.allocatedBytes
            + this.globalFineRedistanceA.allocatedBytes + this.globalFineRedistanceB.allocatedBytes
            + this.globalFineSummaries.plan.allocatedBytes;
          this.info.allocatedBytes += allocated;
          this.info.globalFineLevelSetAllocatedBytes += allocated;
          this.info.globalFineLevelSetResidentBrickCapacity = globalPlan.maximumResidentBricks;
          this.info.globalFineLevelSetLogicalBrickCount = globalPlan.logicalBrickCount;
        }
      }
    }
    this.sparseBrickWorldAccountedBytes = this.sparseBrickWorld?.allocatedBytes ?? 0;
    if (!deferPipelineCompilation) this.createPipelinesSync();
    this.writeParams(options.jacobiRelaxation ?? 0.8);
  }

  private createProjectionGroup(
    velocityIn: GPUTexture,
    velocityOut: GPUTexture,
    pressureIn: GPUBuffer,
    pressureOut: GPUBuffer,
    sparse?: { surfaceData: GPUBuffer; surfaceIndex: GPUBuffer; params?: GPUBuffer },
    binding15Override?: GPUBuffer,
  ): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: velocityOut.createView() },
      { binding: 2, resource: { buffer: this.compaction } },
      { binding: 3, resource: { buffer: this.topology } },
      { binding: 4, resource: { buffer: pressureIn } },
      { binding: 5, resource: { buffer: pressureOut } },
      { binding: 6, resource: { buffer: sparse?.params ?? (sparse ? this.pagedParams! : this.params) } },
      { binding: 7, resource: (sparse ? this.levelSetFallbackTexture! : this.levelSetTexture).createView() },
      { binding: 8, resource: { buffer: this.leafHeaders } },
      { binding: 9, resource: { buffer: this.leafEntries } },
      { binding: 10, resource: { buffer: this.resources.rigidBodies } },
      { binding: 11, resource: { buffer: sparse?.surfaceData ?? this.solidCells } },
      { binding: 12, resource: this.resources.terrain.createView() },
      { binding: 13, resource: { buffer: this.leafFrontier } },
      { binding: 14, resource: this.phiSnapshotTexture.createView() },
      { binding: 15, resource: { buffer: sparse?.surfaceIndex
        ?? binding15Override
        ?? this.sparseBrickWorld?.bulkResidencyWorklist
        ?? this.topologyResidency.worklist } },
    ] });
  }

  private createCouplingGroup(pressure: GPUBuffer, levelSet: GPUTexture): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.couplingLayout, entries: [
      { binding: 0, resource: { buffer: pressure } },
      { binding: 1, resource: { buffer: this.topology } },
      { binding: 2, resource: { buffer: this.solidCells } },
      { binding: 3, resource: { buffer: this.resources.rigidBodies } },
      { binding: 4, resource: { buffer: this.resources.rigidExchange } },
      { binding: 5, resource: { buffer: this.params } },
      { binding: 6, resource: levelSet.createView() },
      { binding: 7, resource: { buffer: this.leafFrontier } },
      { binding: 8, resource: { buffer: this.compaction } },
    ] });
  }

  private descriptor(entryPoint: string): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.shader, entryPoint, constants: this.pipelineConstants() } };
  }
  private refinementDescriptor(entryPoint: string, size: number): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.shader, entryPoint, constants: { ...this.pipelineConstants(), targetRefinementSize: size } } };
  }
  private pipelineConstants(): Record<string, number> {
    const plan = this.adaptiveSurfacePages?.plan;
    return {
      rowIndexedPressure: this.rowIndexedPressure ? 1 : 0,
      surfaceHashOffset: plan?.hashOffsetWords ?? 0,
      surfaceHashCapacity: plan?.hashCapacity ?? 1,
      surfaceAirHashOffset: plan?.airHashOffsetWords ?? 0,
      surfaceAirHashCapacity: plan?.airHashCapacity ?? 1,
      surfacePageTableOffset: plan?.pageTableOffsetWords ?? 0,
      surfacePhiOffset: plan?.phiAOffsetWords ?? 0,
      surfacePageCapacity: plan?.pageCapacity ?? 1,
      surfaceLeafCapacity: plan?.leafCapacity ?? 1,
      surfacePageResolution: plan?.pageResolution ?? 2,
    };
  }
  private diagnosticDescriptor(): GPUComputePipelineDescriptor {
    return { layout: this.diagnosticPipelineLayout, compute: { module: this.diagnosticShader, entryPoint: "materializeOctreeFields", constants: { rowIndexedPressure: this.rowIndexedPressure ? 1 : 0 } } };
  }
  private couplingDescriptor(): GPUComputePipelineDescriptor {
    return { layout: this.couplingPipelineLayout, compute: { module: this.couplingShader, entryPoint: "accumulatePressureImpulse", constants: { rowIndexedPressure: this.rowIndexedPressure ? 1 : 0 } } };
  }

  private static readonly pipelineEntryPoints = [
    "rasterizeSolids", "resetTopology", "refineTopology", "balanceTopology", "jacobi",
    "rasterizeSolidsActive", "resetTopologyActive", "refineTopologyActive", "balanceTopologyActive",
    "rasterizeSolidsRetired", "resetTopologyRetired", "refineTopologyRetired", "balanceTopologyRetired",
    "beginFrontier", "filterFrontier", "appendFrontier", "appendFrontierActive", "appendFrontierRetired", "finalizeFrontier",
    "planLeaves", "scanLeafBlocks", "emitLeaves", "assembleSystem", "assembleCoarseSystem", "gatherBodyCoupling", "applyBodyCoupling", "iterateLeaves", "iterateChebyshev", "solveLeaves", "reduceResidualPartials", "reduceResidualTotal",
    "reconstructSmallGradients", "reconstructGradients", "project", "projectSmallLeaves", "projectLeaves", "passThroughPressureOverflow", "extrapolateSeed", "extrapolate",
    "extrapolateSeedSparse", "extrapolateSparse", "copyExtrapolatedSparse",
    "markChangedTiles", "buildDirtyWorklist", "refreshSnapshotTiles", "refreshSnapshotDense"
  ] as const;

  private assignPipelines(compiled: GPUComputePipeline[]) {
    [
      this.rasterizeSolidsPipeline, this.resetPipeline, this.refinePipeline, this.balancePipeline, this.jacobiPipeline,
      this.rasterizeSolidsActivePipeline, this.resetActivePipeline, this.refineActivePipeline, this.balanceActivePipeline,
      this.rasterizeSolidsRetiredPipeline, this.resetRetiredPipeline, this.refineRetiredPipeline, this.balanceRetiredPipeline,
      this.beginFrontierPipeline, this.filterFrontierPipeline, this.appendFrontierPipeline, this.appendFrontierActivePipeline, this.appendFrontierRetiredPipeline, this.finalizeFrontierPipeline,
      this.planPipeline, this.scanPipeline, this.emitPipeline, this.assemblePipeline, this.assembleCoarsePipeline, this.gatherBodyCouplingPipeline, this.applyBodyCouplingPipeline, this.iteratePipeline, this.iterateChebyshevPipeline, this.solvePipeline, this.reduceResidualPartialsPipeline, this.reduceResidualTotalPipeline,
      this.reconstructSmallGradientsPipeline, this.reconstructGradientsPipeline, this.projectPipeline, this.projectSmallLeavesPipeline, this.projectLeavesPipeline, this.pressureOverflowPipeline, this.extrapolateSeedPipeline, this.extrapolatePipeline,
      this.extrapolateSeedSparsePipeline, this.extrapolateSparsePipeline, this.copyExtrapolatedSparsePipeline,
      this.markChangedTilesPipeline, this.buildDirtyWorklistPipeline, this.refreshSnapshotTilesPipeline, this.refreshSnapshotDensePipeline
    ] = compiled;
  }

  private pipelineReachability(): OctreeProjectionPipelineReachability {
    return {
      leafSolver: this.leafSolver,
      segmentedProjection: Boolean(this.faceTransport) || this.extrapolationSweeps > 0,
      extrapolationSweeps: this.extrapolationSweeps,
      sparseExtrapolation: this.usesSparseVelocityExtrapolation,
      hasDensePhiSnapshot: this.hasDensePhiSnapshot,
    };
  }

  private basePipelineRequired(entryPoint: string) {
    return octreeProjectionPipelineRequired(entryPoint, this.pipelineReachability());
  }

  private assignCompletePipelines(compiled: GPUComputePipeline[]) {
    const fallback = compiled[0];
    if (!fallback) throw new Error("Octree base pipeline compilation did not publish a fallback slot");
    WebGPUOctreeProjection.pipelineEntryPoints.forEach((_, index) => { compiled[index] ??= fallback; });
    this.assignPipelines(compiled);
  }

  private pipelineCacheKey() {
    return WebGPUOctreeProjection.pipelineEntryPoints.filter((entryPoint) => this.basePipelineRequired(entryPoint)).join("|");
  }

  private applyPipelineCache(entry: OctreePipelineCacheEntry) {
    this.assignPipelines(entry.base);
    this.refineLevelPipelines.clear(); entry.refine.forEach((value, key) => this.refineLevelPipelines.set(key, value));
    this.refineCoarsePipelines.clear(); entry.refineCoarse.forEach((value, key) => this.refineCoarsePipelines.set(key, value));
    this.balanceCoarsePipelines.clear(); entry.balanceCoarse.forEach((value, key) => this.balanceCoarsePipelines.set(key, value));
    this.materializePipeline = entry.materialize;
    this.pressureImpulsePipeline = entry.pressureImpulse;
  }

  private publishPipelineCache() {
    let cache = octreePipelineCache.get(this.device);
    if (!cache) { cache = new Map(); octreePipelineCache.set(this.device, cache); }
    cache.set(this.pipelineCacheKey(), {
      base: WebGPUOctreeProjection.pipelineEntryPoints.map((_, index) => [
        this.rasterizeSolidsPipeline, this.resetPipeline, this.refinePipeline, this.balancePipeline, this.jacobiPipeline,
        this.rasterizeSolidsActivePipeline, this.resetActivePipeline, this.refineActivePipeline, this.balanceActivePipeline,
        this.rasterizeSolidsRetiredPipeline, this.resetRetiredPipeline, this.refineRetiredPipeline, this.balanceRetiredPipeline,
        this.beginFrontierPipeline, this.filterFrontierPipeline, this.appendFrontierPipeline, this.appendFrontierActivePipeline, this.appendFrontierRetiredPipeline, this.finalizeFrontierPipeline,
        this.planPipeline, this.scanPipeline, this.emitPipeline, this.assemblePipeline, this.assembleCoarsePipeline, this.gatherBodyCouplingPipeline, this.applyBodyCouplingPipeline, this.iteratePipeline, this.iterateChebyshevPipeline, this.solvePipeline, this.reduceResidualPartialsPipeline, this.reduceResidualTotalPipeline,
        this.reconstructSmallGradientsPipeline, this.reconstructGradientsPipeline, this.projectPipeline, this.projectSmallLeavesPipeline, this.projectLeavesPipeline, this.pressureOverflowPipeline, this.extrapolateSeedPipeline, this.extrapolatePipeline,
        this.extrapolateSeedSparsePipeline, this.extrapolateSparsePipeline, this.copyExtrapolatedSparsePipeline,
        this.markChangedTilesPipeline, this.buildDirtyWorklistPipeline, this.refreshSnapshotTilesPipeline, this.refreshSnapshotDensePipeline,
      ][index]),
      refine: new Map(this.refineLevelPipelines), refineCoarse: new Map(this.refineCoarsePipelines), balanceCoarse: new Map(this.balanceCoarsePipelines),
      materialize: this.materializePipeline, pressureImpulse: this.pressureImpulsePipeline,
    });
  }

  private createPipelinesSync() {
    const compiled: GPUComputePipeline[] = [];
    WebGPUOctreeProjection.pipelineEntryPoints.forEach((entryPoint, index) => {
      if (this.basePipelineRequired(entryPoint)) {
        compiled[index] = this.device.createComputePipeline(this.descriptor(entryPoint));
      }
    });
    // Keep the fixed assignment/cache ABI without asking Dawn to specialize
    // entry points unreachable from this immutable solver configuration.
    this.assignCompletePipelines(compiled);
    // Warm every user-selectable leaf specialization once per device. A leaf
    // size change should rebuild data, not stop the interface to compile a
    // program variant the settings UI already advertised.
    for (let size = 32; size >= 2; size >>= 1) this.refineLevelPipelines.set(size, {
      full: this.device.createComputePipeline(this.refinementDescriptor("refineTopology", size)),
      active: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyActive", size)),
      retired: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyRetired", size)),
    });
    for (let size = 32; size >= 16; size >>= 1) {
      this.refineCoarsePipelines.set(size, {
        full: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyCoarse", size)),
        active: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyCoarseActive", size)),
        retired: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyCoarseRetired", size)),
      });
      this.balanceCoarsePipelines.set(size, {
        full: this.device.createComputePipeline(this.refinementDescriptor("balanceTopologyCoarse", size)),
        active: this.device.createComputePipeline(this.refinementDescriptor("balanceTopologyCoarseActive", size)),
        retired: this.device.createComputePipeline(this.refinementDescriptor("balanceTopologyCoarseRetired", size)),
      });
    }
    this.materializePipeline = this.device.createComputePipeline(this.diagnosticDescriptor());
    this.pressureImpulsePipeline = this.device.createComputePipeline(this.couplingDescriptor());
    this.publishPipelineCache();
  }

  get topologyTexture() { return this.topologyDiagnosticTexture; }
  get pressureSamplesTexture() { return this.pressureSamplesDiagnosticTexture; }
  get pressureTexture() { return this.pressureDiagnosticTexture; }
  get divergenceTexture() { return this.divergenceDiagnosticTexture; }
  get hasDiagnosticTextures() { return this.diagnosticGroups !== undefined; }

  /** Allocate the dense scientific-overlay fields only after inspection asks for them. */
  ensureDiagnosticTextures(): boolean {
    // Scientific overlays are an explicit dense materialization request. If
    // the bootstrap publication has already been retired, the renderer uses
    // the adaptive page/leaf overlays instead of silently sampling the 1x1
    // format fallback as though it covered the domain.
    if (this.denseBootstrapPhiReleased) return false;
    if (this.diagnosticGroups) return false;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    const size: GPUExtent3D = [this.dims.nx, this.dims.ny, this.dims.nz];
    this.topologyDiagnosticTexture = this.device.createTexture({ label: "Octree overlay topology", size, dimension: "3d", format: "rg32uint", usage });
    this.pressureSamplesDiagnosticTexture = this.device.createTexture({ label: "Octree overlay pressure ownership", size, dimension: "3d", format: "rgba32uint", usage });
    this.pressureDiagnosticTexture = this.device.createTexture({ label: "Octree mapped leaf pressure", size, dimension: "3d", format: "r32float", usage });
    this.divergenceDiagnosticTexture = this.device.createTexture({ label: "Octree projected divergence", size, dimension: "3d", format: "r32float", usage });
    const diagnosticGroup = (pressure: GPUBuffer, gradients: GPUBuffer) => this.device.createBindGroup({ layout: this.diagnosticLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } },
      { binding: 1, resource: { buffer: pressure } },
      { binding: 2, resource: this.resources.velocityOut.createView() },
      { binding: 3, resource: this.levelSetTexture.createView() },
      { binding: 4, resource: this.topologyDiagnosticTexture!.createView() },
      { binding: 5, resource: this.pressureSamplesDiagnosticTexture!.createView() },
      { binding: 6, resource: this.pressureDiagnosticTexture!.createView() },
      { binding: 7, resource: this.divergenceDiagnosticTexture!.createView() },
      { binding: 8, resource: { buffer: this.params } },
      { binding: 9, resource: { buffer: this.rowIndexedPressure ? this.leafHeaders : gradients } },
      { binding: 10, resource: this.resources.velocityIn.createView() },
      { binding: 11, resource: { buffer: this.leafFrontier } }
    ] });
    this.diagnosticGroups = {
      pressureA: diagnosticGroup(this.pressureA, this.pressureB),
      pressureB: diagnosticGroup(this.pressureB, this.pressureA)
    };
    this.info.allocatedBytes += this.dims.nx * this.dims.ny * this.dims.nz * 40;
    return true;
  }

  initializationTasks(): GPUInitializationTask[] {
    const cached = octreePipelineCache.get(this.device)?.get(this.pipelineCacheKey());
    const entries = WebGPUOctreeProjection.pipelineEntryPoints;
    const tasks: GPUInitializationTask[] = cached
      ? [{ id: "octree.pipeline-cache", phase: "adaptive-topology", label: "Reuse compiled adaptive programs", run: () => this.applyPipelineCache(cached) }]
      : [];
    const compiled = new Array<GPUComputePipeline>(entries.length);
    let lastRequiredBaseIndex = -1;
    if (!cached) entries.forEach((entryPoint, index) => {
      if (this.basePipelineRequired(entryPoint)) lastRequiredBaseIndex = index;
    });
    if (!cached) entries.forEach((entryPoint, index) => {
      if (!this.basePipelineRequired(entryPoint)) return;
      tasks.push({
        id: `octree.pipeline.${entryPoint}`,
        phase: "adaptive-topology",
        label: `Compile octree ${entryPoint}`,
        run: async () => {
          compiled[index] = await this.device.createComputePipelineAsync(this.descriptor(entryPoint));
          if (index === lastRequiredBaseIndex) {
            // Reusing a live pipeline for unreachable slots keeps the fixed
            // assignment/cache tuple stable; those properties are never read
            // by this configuration's encode branches.
            this.assignCompletePipelines(compiled);
          }
        },
      });
    });
    for (let size = 32; size >= 2; size >>= 1) {
      if (cached?.refine.has(size)) continue;
      const level: Partial<{ full: GPUComputePipeline; active: GPUComputePipeline; retired: GPUComputePipeline }> = {};
      const definitions = [
        ["full", "refineTopology"],
        ["active", "refineTopologyActive"],
        ["retired", "refineTopologyRetired"],
      ] as const;
      definitions.forEach(([variant, entryPoint], index) => tasks.push({
        id: `octree.pipeline.refine.${size}.${variant}`,
        phase: "adaptive-topology",
        label: `Compile octree refinement ${size} · ${variant}`,
        run: async () => {
          level[variant] = await this.device.createComputePipelineAsync(this.refinementDescriptor(entryPoint, size));
          if (index === definitions.length - 1) this.refineLevelPipelines.set(size, level as { full: GPUComputePipeline; active: GPUComputePipeline; retired: GPUComputePipeline });
        },
      }));
    }
    for (let size = 32; size >= 16; size >>= 1) {
      for (const operation of ["refine", "balance"] as const) {
        if ((operation === "refine" ? cached?.refineCoarse : cached?.balanceCoarse)?.has(size)) continue;
        const pipelines: Partial<{ full: GPUComputePipeline; active: GPUComputePipeline; retired: GPUComputePipeline }> = {};
        const prefix = operation === "refine" ? "refineTopologyCoarse" : "balanceTopologyCoarse";
        const definitions = [["full", prefix], ["active", `${prefix}Active`], ["retired", `${prefix}Retired`]] as const;
        definitions.forEach(([variant, entryPoint], index) => tasks.push({
          id: `octree.pipeline.${operation}-coarse.${size}.${variant}`,
          phase: "adaptive-topology",
          label: `Compile octree coarse ${operation} ${size} · ${variant}`,
          run: async () => {
            pipelines[variant] = await this.device.createComputePipelineAsync(this.refinementDescriptor(entryPoint, size));
            if (index === definitions.length - 1) {
              const complete = pipelines as { full: GPUComputePipeline; active: GPUComputePipeline; retired: GPUComputePipeline };
              if (operation === "refine") this.refineCoarsePipelines.set(size, complete);
              else this.balanceCoarsePipelines.set(size, complete);
            }
          },
        }));
      }
    }
    if (!cached) {
      tasks.push({ id: "octree.pipeline.materialize", phase: "adaptive-topology", label: "Compile octree overlay materialization", run: async () => { this.materializePipeline = await this.device.createComputePipelineAsync(this.diagnosticDescriptor()); } });
      tasks.push({ id: "octree.pipeline.pressure-impulse", phase: "adaptive-topology", label: "Compile octree pressure-to-body coupling", run: async () => { this.pressureImpulsePipeline = await this.device.createComputePipelineAsync(this.couplingDescriptor()); this.publishPipelineCache(); } });
    } else if (tasks.length > 1) {
      tasks.push({ id: "octree.pipeline-cache.publish", phase: "adaptive-topology", label: "Publish compiled adaptive variants", run: () => this.publishPipelineCache() });
    }
    if (this.powerPolicy.mirrorEnabled && !this.powerDescriptor) tasks.push({
      id: "octree.power-catalog",
      phase: "adaptive-topology",
      label: "Load and allocate octree power-diagram catalog",
      run: async (signal) => {
        try {
          const trace = typeof process !== "undefined" && process.env?.FLUID_POWER_INIT_TRACE === "1";
          if (trace) console.log(JSON.stringify({ phase: "power-init", label: "catalog-load", status: "started" }));
          if (signal.aborted) throw new DOMException("Power catalog initialization aborted", "AbortError");
          const catalog = await loadGeneratedOctreePowerCatalog();
          if (trace) console.log(JSON.stringify({ phase: "power-init", label: "catalog-load", status: "finished" }));
          if (signal.aborted) throw new DOMException("Power catalog initialization aborted", "AbortError");
          this.initializePowerMirror(catalog);
        } catch (error) {
          if (signal.aborted) throw error;
          const detail = error instanceof Error ? error.message : String(error);
          this.info.powerDiagramFallbackReason = [this.info.powerDiagramFallbackReason, `catalog initialization failed: ${detail}`]
            .filter(Boolean).join("; ");
          this.info.powerDiagramReady = false;
        }
      },
    });
    return tasks;
  }

  private initializePowerMirror(catalog: GeneratedOctreePowerCatalogViews): void {
    if (this.powerDescriptor || this.powerLifecycleDisposed) return;
    const tracePowerInit = <T>(label: string, create: () => T): T => {
      const enabled = typeof process !== "undefined" && process.env?.FLUID_POWER_INIT_TRACE === "1";
      const started = performance.now();
      if (enabled) console.log(JSON.stringify({ phase: "power-init", label, status: "started" }));
      const value = create();
      if (enabled) console.log(JSON.stringify({ phase: "power-init", label, status: "finished", elapsed_ms: performance.now() - started }));
      return value;
    };
    const rowCapacity = this.pressureCapacity.rowCapacity;
    const maximumIncidence = OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence;
    const maximumNeighbors = OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows;
    const maximumStorageBytes = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
    // A physical interior face contributes two incidences. Reserve the proven
    // worst-case average when the adapter supports it, otherwise consume the
    // largest binding-safe compact arena and let WP4 suppress publication on
    // a genuine scene overflow. Never create an invalid >limit binding.
    const faceCapacity = Math.max(1, Math.min(
      rowCapacity * Math.ceil(maximumIncidence / 2),
      Math.floor(maximumStorageBytes / 32),
      Math.floor(maximumStorageBytes / OCTREE_POWER_FACE_QUADRATURE_BYTES),
    ));
    const incidenceCapacity = Math.min(faceCapacity * 2, rowCapacity * maximumIncidence);
    const entryCapacity = rowCapacity * maximumNeighbors;
    const facePlan = planOctreePowerFaces(rowCapacity, faceCapacity, incidenceCapacity);
    const operatorPlan = planOctreePowerGPUOperator(rowCapacity, faceCapacity, entryCapacity, maximumIncidence);
    if (facePlan.faceBytes > maximumStorageBytes || facePlan.normalBytes > maximumStorageBytes
      || facePlan.centroidBytes > maximumStorageBytes || facePlan.quadratureBytes > maximumStorageBytes
      || facePlan.incidenceBytes > maximumStorageBytes || facePlan.boundaryQueryBytes > maximumStorageBytes
      || operatorPlan.arenaBytes > maximumStorageBytes) {
      throw new RangeError("Power-diagram compact arenas exceed this adapter's storage-buffer binding limit");
    }
    this.powerDescriptor = tracePowerInit("descriptor", () => new WebGPUOctreePowerDescriptor(this.device, rowCapacity));
    this.powerTopology = tracePowerInit("topology", () => new WebGPUOctreePowerTopology(this.device, rowCapacity, catalog));
    this.powerFaces = tracePowerInit("faces", () => new WebGPUOctreePowerFaces(this.device, rowCapacity, faceCapacity, this.powerTopology!.source, incidenceCapacity));
    this.powerOperator = tracePowerInit("operator", () => new WebGPUOctreePowerOperator(this.device, rowCapacity, faceCapacity, entryCapacity, maximumIncidence));
    this.powerVelocity = tracePowerInit("velocity", () => new WebGPUOctreePowerVelocity(this.device, rowCapacity));
    if (this.globalFineSourceA && this.globalFineSourceB) {
      this.powerCoarseLevelSet = new WebGPUOctreeCoarseLevelSet(this.device, rowCapacity);
      this.powerCoarseLevelSetSchedule = new WebGPUOctreePowerCoarseLevelSet(
        this.device, this.powerCoarseLevelSet, this.powerTopology.source,
      );
      const coarseDirectory = this.powerCoarseLevelSetSchedule.sampleSource.directory;
      // Binding 15 is a dual ABI. Global-fine pressure/topology groups consume
      // the compact coarse-phi directory; sparse extrapolation retains its
      // dedicated bulk-worklist groups and therefore never observes it.
      this.groups = {
        ab: this.createProjectionGroup(this.resources.velocityIn, this.resources.velocityOut,
          this.pressureA, this.pressureB, undefined, coarseDirectory),
        ba: this.createProjectionGroup(this.resources.velocityIn, this.resources.velocityOut,
          this.pressureB, this.pressureA, undefined, coarseDirectory),
        extrapolateOut: this.groups.extrapolateOut,
        extrapolateScratch: this.groups.extrapolateScratch,
      };
      this.fineSummarySizingGroup = this.createProjectionGroup(
        this.resources.velocityIn, this.resources.velocityOut,
        this.globalFineSummaries?.directory ?? this.fineSummaryFallback, this.pressureB,
        undefined, coarseDirectory);
      this.fineToPowerCoarseLevelSet = new WebGPUFineToCoarseLevelSet(this.device, rowCapacity,
        this.globalFineSourceA.plan.maximumResidentBricks * this.globalFineSourceA.plan.samplesPerBrick);
      const compactCoarse = this.powerCoarseLevelSetSchedule.sampleSource;
      this.globalFineTopologyAB = new WebGPUFineLevelSetTopology(
        this.device, this.globalFineSourceA, this.globalFineSourceB, compactCoarse.wgsl(9), false,
      );
      this.globalFineTopologyBA = new WebGPUFineLevelSetTopology(
        this.device, this.globalFineSourceB, this.globalFineSourceA, compactCoarse.wgsl(9), false,
      );
    }
    if (this.powerPolicy.authoritative && this.faceMirror) {
      this.powerFaceSeed = tracePowerInit("face-seed", () => new WebGPUOctreePowerFaceSeed(this.device, this.faceMirror!.source, this.powerFaces!.source));
      this.powerFaceTransfer = tracePowerInit("face-transfer", () => new WebGPUOctreePowerFaceTransfer(this.device, this.powerFaces!.source, this.leafHeaders,
        [this.dims.nx, this.dims.ny, this.dims.nz]));
      if (sceneHasTerrain(this.scene)) {
        this.powerSolidVertices = tracePowerInit("solid-vertex-sdf", () => new WebGPUOctreeSolidVertexSdf(
          this.device, rowCapacity, this.leafHeaders, this.compaction, this.resources.terrain, this.powerFaceSeed!.control,
        ));
      }
      if (this.scene.rigidBodies.length > 0 || sceneHasTerrain(this.scene)) {
        this.powerSolidFaces = new WebGPUOctreePowerSolidFaces(this.device, {
          faces: this.powerFaces.source,
          rigidBodies: this.resources.rigidBodies,
          terrain: this.resources.terrain,
          pressureA: this.pressureA,
          pressureB: this.pressureB,
          rigidExchange: this.resources.rigidExchange,
          solidVertices: this.powerSolidVertices?.source,
        });
      }
    }
    if (this.powerFaceSeed && this.powerVelocity && this.globalFineLevelSet && this.globalFineSourceA && this.globalFineSourceB) {
      const queryCapacity = this.globalFineLevelSet.plan.maximumResidentBricks
        * this.globalFineLevelSet.plan.samplesPerBrick;
      const velocityChunkCapacity = planOctreePowerVelocityChunkCapacity(queryCapacity, {
        maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize,
        maxBufferSize: this.device.limits.maxBufferSize,
        maxComputeWorkgroupsPerDimension: this.device.limits.maxComputeWorkgroupsPerDimension,
        minStorageBufferOffsetAlignment: this.device.limits.minStorageBufferOffsetAlignment,
      });
      this.globalFineVelocityPrepass = new WebGPUOctreePowerVelocityPrepass(
        this.device, velocityChunkCapacity, this.powerTopology.source, this.powerFaces.source,
      );
      if (this.faceMirror) {
        // Factor-4 has one B4 page per finest octree cell; factor-8 has a
        // deduplicated 2 x 2 x 2 page group per cell. The bounded graph depth
        // spans both sides of the configured physical interface band plus
        // topology-publication support; validation suppresses the generation
        // if this conservative bound is ever insufficient.
        const maximumBandPhiRounds = Math.max(1, 2 * this.interfaceRefinementBandCells + 4);
        this.globalFineFaceFastMarch = new WebGPUOctreeFaceFastMarch(
          this.device, this.globalFineSourceA, rowCapacity, maximumBandPhiRounds, this.powerFaces.plan.faceCapacity,
        );
      }
      this.globalFineTransportA = new WebGPUFineLevelSetTransport(
        this.device, this.globalFineSourceA, this.globalFineVelocityPrepass, this.globalFineFaceFastMarch,
      );
      this.globalFineTransportB = new WebGPUFineLevelSetTransport(
        this.device, this.globalFineSourceB, this.globalFineVelocityPrepass, this.globalFineFaceFastMarch,
      );
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
        sampleHashCapacity: this.powerCoarseLevelSetSchedule!.sampleSource.hashCapacity };
      this.globalFineVolumeA = new WebGPUFineLevelSetVolumeCorrection(
        this.device, this.globalFineSourceA, coarseVolumeSource,
      );
      this.globalFineVolumeB = new WebGPUFineLevelSetVolumeCorrection(
        this.device, this.globalFineSourceB, coarseVolumeSource, this.globalFineVolumeA.control,
      );
    }
    this.powerVolumeParams = this.device.createBuffer({ label: "Octree power-volume parameters", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const cellVolume = (this.scene.container.width_m / this.dims.nx)
      * (this.scene.container.height_m / this.dims.ny)
      * (this.scene.container.depth_m / this.dims.nz);
    const data = new ArrayBuffer(16); new Float32Array(data)[0] = cellVolume; new Uint32Array(data)[1] = rowCapacity;
    this.device.queue.writeBuffer(this.powerVolumeParams, 0, data);
    const shaderModule = this.device.createShaderModule({ label: "Publish physical octree power volumes", code: octreePowerVolumeShader });
    this.powerVolumePipeline = tracePowerInit("volume-pipeline", () => this.device.createComputePipeline({ label: "Publish physical octree power volumes", layout: "auto",
      compute: { module: shaderModule, entryPoint: "publishPowerVolumes" } }));
    this.powerVolumeGroup = this.device.createBindGroup({ layout: this.powerVolumePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.powerVolumeParams } }, { binding: 1, resource: { buffer: this.powerTopology.metrics } },
      { binding: 2, resource: { buffer: this.leafHeaders } }, { binding: 3, resource: { buffer: this.compaction } },
      { binding: 4, resource: { buffer: this.powerVolumes } },
    ] });
    const powerAllocated = sumOctreePowerAllocationBreakdown({
      descriptors: this.powerDescriptor.plan.allocatedBytes,
      topology: this.powerTopology.plan.allocatedBytes,
      faces: this.powerFaces.plan.allocatedBytes,
      operator: this.powerOperator.plan.allocatedBytes,
      faceSeed: this.powerFaceSeed?.plan.allocatedBytes ?? 0,
      faceTransfer: this.powerFaceTransfer?.plan.allocatedBytes ?? 0,
      solidVertices: this.powerSolidVertices?.plan.allocatedBytes ?? 0,
      solidFaces: this.powerSolidFaces?.plan.allocatedBytes ?? 0,
      velocity: this.powerVelocity.plan.allocatedBytes,
      coarseLevelSet: this.powerCoarseLevelSet?.plan.allocatedBytes ?? 0,
      coarseSchedule: this.powerCoarseLevelSetSchedule?.plan.allocatedBytes ?? 0,
      physicalVolumes: rowCapacity * 4,
      physicalVolumeParams: 16,
    });
    const fineAllocated = sumOctreePowerAllocationBreakdown({
      restriction: this.fineToPowerCoarseLevelSet?.plan.allocatedBytes ?? 0,
      topologyAB: this.globalFineTopologyAB?.allocatedBytes ?? 0,
      topologyBA: this.globalFineTopologyBA?.allocatedBytes ?? 0,
      velocityPrepass: this.globalFineVelocityPrepass?.plan.allocatedBytes ?? 0,
      faceFastMarch: this.globalFineFaceFastMarch?.plan.gpuAllocatedBytes ?? 0,
      transportA: this.globalFineTransportA?.plan.allocatedBytes ?? 0,
      transportB: this.globalFineTransportB?.plan.allocatedBytes ?? 0,
      volumeA: this.globalFineVolumeA?.allocatedBytes ?? 0,
      volumeB: this.globalFineVolumeB?.allocatedBytes ?? 0,
    });
    this.info.powerDiagramAllocatedBytes = powerAllocated;
    this.info.allocatedBytes += powerAllocated + fineAllocated;
    this.info.globalFineLevelSetAllocatedBytes += fineAllocated;
    this.info.powerDiagramReady = true;
    this.info.powerDiagramAuthoritative = this.powerPolicy.authoritative && Boolean(this.powerFaceSeed)
      && (!sceneHasTerrain(this.scene) || Boolean(this.powerSolidVertices && this.powerSolidFaces))
      && (this.scene.rigidBodies.length === 0 || Boolean(this.powerSolidFaces));
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

  private writeParams(relaxation = 0.8) {
    const data = new ArrayBuffer(160);
    new Uint32Array(data, 0, 4).set([this.dims.nx, this.dims.ny, this.dims.nz, this.maxLeafSize]);
    new Float32Array(data, 16, 4).set([
      this.scene.container.width_m / this.dims.nx,
      this.scene.container.height_m / this.dims.ny,
      this.scene.container.depth_m / this.dims.nz,
      relaxation
    ]);
    new Uint32Array(data, 32, 4).set([Math.round(this.adaptivity * 1000), this.iterations, this.linearBlocks, 0]);
    // Megakernel residual tolerance, followed by the scaled Chebyshev spectrum.
    new Float32Array(data, 48, 4).set([1e-8, 0.01, 2.2, this.interfaceRefinementBandCells]);
    // container.w is an exactly representable small bit mask shared with the
    // topology shader: terrain, closed ceiling, authoritative power wall
    // strip.  Encoding these together avoids widening the long-lived params
    // ABI while keeping open-top dam-break semantics explicit.
    const containerFlags = (sceneHasTerrain(this.scene) ? 1 : 0)
      | (this.scene.container.top === "closed" ? 2 : 0)
      | (this.powerPolicy.authoritative ? 4 : 0);
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
    const hasImportedInitialSeeds = (this.scene.fluid.initialBrickSeeds_m?.length ?? 0) > 0;
    const encodedSurfaceDetail = hasImportedInitialSeeds || !this.analyticSparseBootstrap
      ? this.surfaceDetailStrength
      : this.scene.fluid.initialCondition === "dam-break"
        ? -20 - this.surfaceDetailStrength
        : -10 - this.surfaceDetailStrength;
    new Float32Array(data, 112, 4).set([
      this.scene.fluid.density_kg_m3,
      this.scene.fluid.surfaceTension_N_m,
      this.scene.numerics.maxDt_s,
      encodedSurfaceDetail
    ]);
    new Uint32Array(data, 128, 4).set([
      this.rowIndexedPressure ? this.pressureCapacity.rowCapacity : this.dims.nx * this.dims.ny * this.dims.nz,
      this.rowIndexedPressure ? this.pressureCapacity.entryCapacity : this.dims.nx * this.dims.ny * this.dims.nz * 6,
      this.coarseTaskCapacity,
      this.pressureWarmStart ? 1 : 0,
    ]);
    new Float32Array(data, 144, 4).set([
      this.scene.fluid.gravity_m_s2.y,
      0,
      this.hydrostaticSplit ? 1 : 0,
      this.scene.container.fillFraction * this.dims.ny,
    ]);
    this.device.queue.writeBuffer(this.params, 0, data);
    if (this.pagedParams) {
      const paged = data.slice(0);
      const flags = new Uint32Array(paged, 140, 1);
      flags[0] |= 2;
      this.device.queue.writeBuffer(this.pagedParams, 0, paged);
    }
  }

  /** The split reference is a velocity potential, so it scales with this substep's dt. */
  setHydrostaticTimestep(dt_s: number) {
    this.faceTransportDt_s = Math.max(0, Number.isFinite(dt_s) ? dt_s : 0);
    if (this.hydrostaticSplit) {
      const value = new Float32Array([dt_s]);
      this.device.queue.writeBuffer(this.params, 148, value);
      if (this.pagedParams) this.device.queue.writeBuffer(this.pagedParams, 148, value);
    }
  }

  setCouplingBodies(count: number, hasDynamicBodies: boolean) {
    this.couplingHasDynamicBodies = hasDynamicBodies;
    this.couplingBodyCount = Math.max(0, Math.min(12, Math.floor(count)));
    this.device.queue.writeBuffer(this.params, 44, new Uint32Array([Math.max(0, Math.min(12, Math.floor(count)))]));
    this.device.queue.writeBuffer(this.params, 116, new Float32Array([hasDynamicBodies ? 1 : 0]));
    if (this.pagedParams) {
      this.device.queue.writeBuffer(this.pagedParams, 44, new Uint32Array([Math.max(0, Math.min(12, Math.floor(count)))]));
      this.device.queue.writeBuffer(this.pagedParams, 116, new Float32Array([hasDynamicBodies ? 1 : 0]));
    }
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group = this.groups.ab) {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(...this.workgroups);
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
      // tiles are analytically non-negative air, while legacyPhi retains the
      // authored SDF until compact coarse phi has published.
      this.analyticBootstrapWorklist.encode(encoder);
      this.topologyWorklistReady = true;
      this.encodeInlineRebuild(encoder, undefined, true);
      return;
    }
    const topologyWorklistReady = this.topologyWorklistReady;
    this.topologyWorklistReady = false;
    try {
      this.encodeInlineRebuild(encoder);
    } finally {
      this.topologyWorklistReady = topologyWorklistReady;
    }
  }

  /** Encode one dependency-ordered t=0 checkpoint. Async startup submits and
   * fences these separately so a driver failure is localized to one bounded
   * phase instead of one giant command buffer. */
  encodeInitialSparseAuthorityPhase(encoder: GPUCommandEncoder, phase: OctreeInitialSparseAuthorityPhaseId) {
    switch (phase) {
      case "cold-topology": this.encodeColdBootstrapRebuild(encoder); break;
      case "power-operator-authority": this.encode(encoder, this.dims.nx, this.dims.ny, this.dims.nz); break;
      case "surface-global-fine": this.encodeSurface(encoder, 0); break;
      case "section5-face-band-topology": this.encodeGlobalFineFaceBandPhase(encoder, "topology-build"); break;
      case "section5-face-band-transitions": this.encodeGlobalFineFaceBandPhase(encoder, "transition-adjacency"); break;
      case "section5-face-band-fast-march": this.encodeGlobalFineFaceBandPhase(encoder, "fast-march"); break;
      case "section5-face-band-power-publication": this.encodeGlobalFineFaceBandPhase(encoder, "power-publication"); break;
      case "sparse-render-world": this.encodeSparseBrickWorld(encoder); break;
      default: phase satisfies never;
    }
  }

  /** Preserve the combined encoder contract for synchronous/diagnostic callers. */
  encodeInitialSparseAuthority(encoder: GPUCommandEncoder) {
    for (const phase of OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES) {
      this.encodeInitialSparseAuthorityPhase(encoder, phase.id);
    }
  }

  /** Retire invocation-stable coarse-phi parameter slots after queue submit. */
  retireSubmittedEncoder(encoder: GPUCommandEncoder) {
    this.powerCoarseLevelSetSchedule?.retireSubmittedEncoder(encoder);
  }

  /**
   * Publish the Section 5 regular-face and transition-tetra velocity band from
   * the current indexed fine generation.  Warmup invokes this after the first
   * fine publication so a paused t=0 scene already owns every interpolation
   * structure required by its first trajectory; regular steps refresh it from
   * the newly projected power velocities before transport.
   */
  private encodeGlobalFineFaceBand(encoder: GPUCommandEncoder) {
    for (const phase of OCTREE_FACE_BAND_ENCODE_PHASES) {
      this.encodeGlobalFineFaceBandPhase(encoder, phase);
    }
  }

  /** Encode one independently fenceable Section 5 face-band checkpoint. */
  private encodeGlobalFineFaceBandPhase(
    encoder: GPUCommandEncoder,
    phase: OctreeFaceBandEncodePhase,
  ) {
    if (!this.globalFineFaceFastMarch || !this.globalFineBootstrapped || !this.powerVelocity
      || !this.powerFaces || !this.powerTopology || !this.powerCoarseLevelSetSchedule) return;
    const fine = this.globalFineCurrentIsA ? this.globalFineSourceA : this.globalFineSourceB;
    const fineTopology = this.globalFineCurrentIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    if (!fine || !fineTopology) return;
    this.globalFineFaceFastMarch.encodePhase(encoder, {
      fine,
      fineTopologyControl: fineTopology.control,
      owners: this.topology,
      coarsePhiDirectory: this.powerCoarseLevelSetSchedule.sampleSource.directory,
      powerRowVelocities: this.powerVelocity.velocities,
      powerVelocityControl: this.powerVelocity.control,
      powerVelocityGeneration: this.powerGeneration,
      powerTopology: this.powerTopology.source,
      powerFaces: this.powerFaces.source,
      siteIndex: this.powerFaces.source.siteIndex,
      siteHashCapacity: this.powerFaces.plan.hashCapacity,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      maximumLeafSize: this.maxLeafSize,
      generation: this.globalFineGeneration,
      closedTop: this.scene.container.top === "closed",
    }, phase);
    if (phase !== "power-publication") return;
    // Complete the paper's round trip. The marcher commits only after its
    // whole narrow-band power subset validates; reconstructing the compact
    // regular field afterwards makes that extrapolated result the next
    // face-transport input, and the exact generalized-face snapshot preserves
    // it across a topology rebuild. On marcher failure the power records are
    // untouched, so these guarded publications reproduce the projected
    // rollback rather than exposing partial scratch values.
    if (this.powerFaceSeed && this.powerOperator) {
      this.powerFaceSeed.encodePowerToAxis(encoder, this.powerOperator.control, true);
      this.powerFaceTransfer?.encodeCapture(encoder);
    }
  }

  encodeInlineRebuild(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites,
    analyticColdBootstrap = false) {
    // Directory generation N is produced after topology N and is the authority
    // for the next topology rebuild. Queue this expected generation before the
    // command buffer; later surface publication uses its own parameter buffer.
    if (this.powerCoarseLevelSetSchedule) {
      const generation = this.globalFineGeneration & 0x3fff_ffff;
      const flags = (this.pressureWarmStart ? 1 : 0) | (generation << 2);
      this.device.queue.writeBuffer(this.params, 140, new Uint32Array([flags >>> 0]));
    }
    // The first rebuild initializes every owner and, when present, solid cell. Thereafter the
    // previous publication's GPU-owned topology-tile list is the rebuild
    // domain: tiles span max(brick, maximumLeaf) cells, so every leaf lies
    // inside exactly one tile and partial rebuilds can never split a leaf.
    const active = this.topologyWorklistReady;
    // The owner-page lifecycle consumes the same residency publication as the
    // partial topology path. While direct-paged topology deliberately retains
    // a full-domain rebuild, an empty/unpublished candidate list must not retire
    // the previous coarse owner map before narrow-band phi misses consult it.
    // Full rebuild refinement still allocates missing pages on-GPU through
    // ensureLeafOwnerPages/storeOwnerEnsure. Re-enable lifecycle retirement
    // when candidate worklists gain a transactional publication predicate.
    if (analyticColdBootstrap && this.ownerPages) this.ownerPages.encodeAnalyticBootstrap(encoder);
    else if (active || !this.directPagedTopology) this.ownerPages?.encode(encoder);
    if (active) {
      encoder.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, 0,
        this.compaction, 0,
        this.topologyResidency.tileWorklistByteLength
      );
      // Dawn forbids one buffer being both writable storage and INDIRECT in a
      // pass; the dispatch args are staged into the dedicated indirect buffer.
      encoder.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES,
        this.solveDispatch, 0, 12
      );
      encoder.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, FLUID_TILE_RETIRED_DISPATCH_OFFSET_BYTES,
        this.solveDispatch, 16, 12
      );
      encoder.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
        this.topologyCandidateDispatch, 0, 12
      );
      encoder.copyBufferToBuffer(
        this.topologyResidency.tileWorklist, FLUID_TILE_RETIRED_CANDIDATE_DISPATCH_OFFSET_BYTES,
        this.topologyCandidateDispatch, 16, 12
      );
      // Coarse cooperative kernels launch exactly one workgroup per worklist
      // tile (header word 0 = active count, word 4 = retired count) and walk
      // the per-size sub-blocks internally, so no surplus workgroups launch.
      encoder.copyBufferToBuffer(this.topologyResidency.tileWorklist, 0, this.solveDispatch, 32, 4);
      encoder.copyBufferToBuffer(this.topologyResidency.tileWorklist, 16, this.solveDispatch, 48, 4);
      if (this.surfacePagesBootstrapped) this.pagedPhiDifferential?.encode(encoder, this.solveDispatch);
    }
    // Change-driven scheduling: compare phi against each active tile's last
    // rebuilt snapshot, compact the active list down to dirty tiles (plus
    // their neighbors — grading reach is under one tile), and re-stage the
    // filtered dispatch args over the active-side slots. Retired tiles keep
    // their full rebuild. Conservative fallbacks (bodies, detail refinement)
    // leave the classic every-tile rebuild untouched.
    if (active && this.changeDrivenEligible()) {
      const mark = encoder.beginComputePass({ label: "Octree topology change detection" });
      mark.setPipeline(this.markChangedTilesPipeline);
      mark.setBindGroup(0, this.groups.ab);
      mark.dispatchWorkgroupsIndirect(this.solveDispatch, 32);
      mark.setPipeline(this.buildDirtyWorklistPipeline);
      mark.dispatchWorkgroups(1);
      mark.end();
      encoder.copyBufferToBuffer(this.compaction, 4, this.solveDispatch, 0, 12);
      encoder.copyBufferToBuffer(this.compaction, 32, this.topologyCandidateDispatch, 0, 12);
      encoder.copyBufferToBuffer(this.compaction, 0, this.solveDispatch, 32, 4);
      const refresh = encoder.beginComputePass({ label: "Octree topology snapshot refresh" });
      refresh.setPipeline(this.refreshSnapshotTilesPipeline);
      refresh.setBindGroup(0, this.groups.ab);
      refresh.dispatchWorkgroupsIndirect(this.solveDispatch, 32);
      refresh.end();
    }
    const pass = encoder.beginComputePass({
      label: "Octree reset and refinement",
      ...(timestampWrites?.beginningOfPassWriteIndex !== undefined ? {
        timestampWrites: {
          querySet: timestampWrites.querySet,
          beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex,
        },
      } : {}),
    });
    const dispatch = (full: GPUComputePipeline, resident: GPUComputePipeline) => {
      pass.setPipeline(active ? resident : full);
      pass.setBindGroup(0, this.groups.ab);
      if (active) pass.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      else pass.dispatchWorkgroups(...this.workgroups);
    };
    const dispatchRetired = (pipeline: GPUComputePipeline) => {
      if (!active) return;
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.groups.ab);
      pass.dispatchWorkgroupsIndirect(this.solveDispatch, 16);
    };
    const candidateWorkgroups: [number, number, number] = [
      Math.ceil(this.dims.nx / 8), Math.ceil(this.dims.ny / 8), Math.ceil(this.dims.nz / 8)
    ];
    const dispatchCandidates = (full: GPUComputePipeline, resident: GPUComputePipeline,
      group = this.groups.ab) => {
      pass.setPipeline(active ? resident : full);
      pass.setBindGroup(0, group);
      if (active) pass.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      else pass.dispatchWorkgroups(...candidateWorkgroups);
    };
    const dispatchRetiredCandidates = (pipeline: GPUComputePipeline, group = this.groups.ab) => {
      if (!active) return;
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 16);
    };
    if (this.hasDenseSolidCells) {
      dispatch(this.rasterizeSolidsPipeline, this.rasterizeSolidsActivePipeline);
      dispatchRetired(this.rasterizeSolidsRetiredPipeline);
    }
    dispatch(this.resetPipeline, this.resetActivePipeline);
    dispatchRetired(this.resetRetiredPipeline);
    // Coarse (size >= 16) cooperative kernels follow the same worklist contract
    // as the fine candidate levels: after initialization only leaves inside
    // active or retired topology tiles can change, so stable coarse leaves in
    // the vast calm interior are never rescanned. The resident variants ride
    // the candidate indirect args ((tileSize/8)^3 groups per tile) and retire
    // surplus workgroups against the per-size block count in the kernel.
    const coarseWorklistGating = typeof process === "undefined" || process.env?.FLUID_OCTREE_COARSE_WORKLIST !== "0";
    const dispatchCoarse = (size: number, pipelines: { full: GPUComputePipeline; active: GPUComputePipeline; retired: GPUComputePipeline },
      group = this.groups.ab) => {
      if (active && coarseWorklistGating) {
        pass.setPipeline(pipelines.active);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroupsIndirect(this.solveDispatch, 32);
        pass.setPipeline(pipelines.retired);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroupsIndirect(this.solveDispatch, 48);
      } else {
        pass.setPipeline(pipelines.full);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(this.dims.nx / size), Math.ceil(this.dims.ny / size), Math.ceil(this.dims.nz / size));
      }
    };
    for (let size = this.maxLeafSize; size >= 2; size >>= 1) {
      if (size >= 16) {
        dispatchCoarse(size, this.refineCoarsePipelines.get(size)!, this.fineSummarySizingGroup);
      } else {
        const level = this.refineLevelPipelines.get(size)!;
        dispatchCandidates(level.full, level.active, this.fineSummarySizingGroup);
        dispatchRetiredCandidates(level.retired, this.fineSummarySizingGroup);
      }
    }
    const balanceRounds = Math.max(1, Math.ceil(Math.log2(this.maxLeafSize)));
    for (let round = 0; round < balanceRounds; round += 1) {
      for (let size = this.maxLeafSize; size >= 16; size >>= 1) {
        dispatchCoarse(size, this.balanceCoarsePipelines.get(size)!);
      }
      dispatchCandidates(this.balancePipeline, this.balanceActivePipeline);
      dispatchRetiredCandidates(this.balanceRetiredPipeline);
    }
    pass.end();

    // Evolve the persistent liquid-leaf frontier. Only cold initialization
    // walks the full finest lattice; subsequent frames filter the old compact
    // list and append replacement origins from the rebuilt topology tiles.
    const begin = encoder.beginComputePass({ label: "Begin persistent octree leaf frontier" });
    begin.setPipeline(this.beginFrontierPipeline); begin.setBindGroup(0, this.groups.ab); begin.dispatchWorkgroups(1);
    begin.end();
    encoder.copyBufferToBuffer(this.compaction, 48, this.topologyCandidateDispatch, 0, 12);
    const evolve = encoder.beginComputePass({
      label: "Evolve persistent octree leaf frontier",
      ...(timestampWrites?.endOfPassWriteIndex !== undefined ? {
        timestampWrites: {
          querySet: timestampWrites.querySet,
          endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex,
        },
      } : {}),
    });
    evolve.setBindGroup(0, this.groups.ab);
    if (active) {
      evolve.setPipeline(this.filterFrontierPipeline); evolve.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      evolve.setPipeline(this.appendFrontierActivePipeline); evolve.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      evolve.setPipeline(this.appendFrontierRetiredPipeline); evolve.dispatchWorkgroupsIndirect(this.solveDispatch, 16);
    } else {
      evolve.setPipeline(this.appendFrontierPipeline); evolve.dispatchWorkgroups(...this.workgroups);
    }
    evolve.setPipeline(this.finalizeFrontierPipeline); evolve.dispatchWorkgroups(1);
    evolve.end();
    // Reuse the candidate indirect buffer for pressure-row plan/emit; topology
    // candidate dispatches have already completed by this point.
    encoder.copyBufferToBuffer(this.compaction, 48, this.topologyCandidateDispatch, 0, 12);
    if (!active && this.hasDensePhiSnapshot) {
      // Seed the change-detection snapshot from the cold rebuild's phi so
      // eligibility can engage on any later frame without a stale baseline.
      const seed = encoder.beginComputePass({ label: "Octree topology snapshot seed" });
      seed.setPipeline(this.refreshSnapshotDensePipeline);
      seed.setBindGroup(0, this.groups.ab);
      seed.dispatchWorkgroups(...this.workgroups);
      seed.end();
    }
    return true;
  }

  /**
   * Change-driven rebuild is provably safe only while the refinement inputs
   * are phi alone: bodies rewrite solid fractions and detail refinement reads
   * velocity strain, so either forces the classic every-tile rebuild.
   */
  private changeDrivenEligible() {
    if (!this.hasDensePhiSnapshot) return false;
    // Fine summaries are a new per-generation sizing input.  Until dirty-tile
    // marking compares summary generations, rebuilding every active tile is
    // the conservative schedule.
    if (this.globalFineSummaries) return false;
    if (this.couplingBodyCount > 0 || this.surfaceDetailStrength > 0) return false;
    return typeof process === "undefined" || process.env?.FLUID_OCTREE_CHANGE_DRIVEN !== "0";
  }

  finishInlineRebuild() { this.info.topologyReuseCount += 1; }
  get extrapolationSweepCount() { return this.extrapolationSweeps; }
  get pressureSolverLabel() {
    if (this.leafSolver === "mgpcg" && this.mgpcg?.plan.preconditionerKind === "section43-hybrid") {
      return `Octree power PCG · Section 4.3 hybrid · up to ${this.mgpcg.iterationBudget} iterations · k=8 paired L2 boundary smoothing on a 3 graph-ring band approximation · ${this.firstOrderVCycle?.plan.levelCount ?? 0}-level L1 V-cycle`;
    }
    if (this.leafSolver === "mgpcg") return `Octree matrix-free aggregate PCG · up to ${this.iterations} iterations · ${this.mgpcg?.plan.hierarchyLevelCount ?? 0} additive levels · experimental`;
    if (this.leafSolver === "chebyshev") return `Octree Chebyshev-Jacobi · ${Math.ceil(this.iterations / 4)} parallel polynomial passes${this.couplingHasDynamicBodies ? " · frame-lagged rigid coupling" : ""}`;
    if (this.leafSolver === "megakernel" && !this.couplingHasDynamicBodies) return `Octree persistent Jacobi · up to ${this.iterations} sweeps`;
    return `Octree weighted Jacobi · ${this.iterations} fixed GPU sweeps`;
  }

  private encodePowerAssemblyMirror(encoder: GPUCommandEncoder): void {
    const descriptor = this.powerDescriptor, topology = this.powerTopology, faces = this.powerFaces;
    const operator = this.powerOperator, volumes = this.powerVolumes;
    const volumePipeline = this.powerVolumePipeline;
    const volumeGroup = this.powerVolumeGroup;
    if (!descriptor || !topology || !faces || !operator || !volumes || !volumePipeline || !volumeGroup) return;
    const dimensions: [number, number, number] = [this.dims.nx, this.dims.ny, this.dims.nz];
    const spacing: [number, number, number] = [
      this.scene.container.width_m / this.dims.nx,
      this.scene.container.height_m / this.dims.ny,
      this.scene.container.depth_m / this.dims.nz,
    ];
    this.powerGeneration = (this.powerGeneration + 1) >>> 0;
    const boundaryFine = this.globalFineBootstrapped
      ? (this.globalFineCurrentIsA ? this.globalFineSourceA : this.globalFineSourceB)
      : this.globalFineSourceA;
    // Paper Sections 4.1/5 require free-surface pressure to evaluate signed
    // distance at both actual cell centres.  Before the first fine-band
    // publication the authored analytic field is that authority; recurring
    // generations consume the current two-sided sparse fine field directly.
    // If neither exists, internal boundary publication fails closed in the
    // face builder instead of synthesizing an affine air value.
    const boundaryPhi = boundaryFine ? {
      mode: this.globalFineBootstrapped ? "fine" as const : "analytic" as const,
      fine: boundaryFine,
      container: [this.scene.container.width_m, this.scene.container.height_m,
        this.scene.container.depth_m] as const,
      fillFraction: this.scene.container.fillFraction,
      initialCondition: this.scene.fluid.initialCondition,
    } : undefined;
    const faceOptions = { dimensions, rowCount: this.compaction,
      physicalCellSize: spacing[0], generation: this.powerGeneration,
      closedBoundaryMask: octreePowerClosedBoundaryMask(this.scene.container.top === "closed"),
      ...(boundaryPhi ? { boundaryPhi } : {}),
    } as const;
    // Geometry descriptors must come from the octree topology authority, not
    // the phase-row index used to resolve incident pressure rows.  A missing
    // phase row does not mean that the spatial leaf is absent: synthesizing a
    // miss at the querying row's preferred size makes the same air location
    // appear coarse to one row and fine to another, producing impossible,
    // non-reciprocal descriptor pairs.  Owner-page residency includes the
    // bounded face/edge halo required by the paper's local encoding.
    faces.encodeSiteIndex(encoder, this.leafHeaders, faceOptions);
    descriptor.encode(encoder, this.leafHeaders, this.topology, {
      dimensions, maximumLeafSize: this.maxLeafSize, rowCountBuffer: this.compaction,
      generation: this.powerGeneration, ownerMode: this.ownerPages ? "paged" : "dense",
    });
    topology.encode(encoder, descriptor.descriptors, this.compaction, spacing);
    faces.encode(encoder, this.leafHeaders, faceOptions, true);
    // Cold/new-connectivity rollback is seeded first. Stable generalized
    // identities then recover the previous projected DOF directly, avoiding
    // a lossy power -> axis -> power round trip on unchanged topology.
    this.powerFaceSeed?.encode(encoder);
    this.powerFaceTransfer?.encodeApply(encoder);
    this.powerSolidVertices?.encode(encoder, {
      dimensions,
      physicalSpacing: spacing,
      generation: this.powerGeneration,
      terrainEnabled: sceneHasTerrain(this.scene),
    });
    this.powerSolidFaces?.encodeClassifyAndConstrain(encoder, {
      dimensions, physicalSpacing: spacing,
      container: [this.scene.container.width_m, this.scene.container.height_m, this.scene.container.depth_m],
      rigidBodyCount: this.scene.rigidBodies.length,
      terrainEnabled: sceneHasTerrain(this.scene),
      pressureImpulseScale: this.faceTransportDt_s,
    });
    const pass = encoder.beginComputePass({ label: "Publish physical power-cell volumes" });
    pass.setPipeline(volumePipeline); pass.setBindGroup(0, volumeGroup);
    pass.dispatchWorkgroups(Math.ceil(this.pressureCapacity.rowCapacity / 64)); pass.end();
    operator.encodeAssemblyFromControl(encoder, faces.faces, faces.source, volumes, faces.control,
      this.powerPolicy.authoritative ? this.powerFaceSeed?.control : undefined,
      this.powerPolicy.authoritative ? this.powerSolidFaces?.control : undefined);
    if (this.powerPolicy.authoritative) {
      operator.encodeLeafRowPublication(encoder, this.leafHeaders, this.leafEntries);
    }
  }

  private encodePowerProjectionMirror(encoder: GPUCommandEncoder, pressure: GPUBuffer): void {
    if (!this.powerFaces || !this.powerOperator) return;
    this.powerOperator.encodeProjectionFromControl(encoder, this.powerFaces.faces, this.powerFaces.source,
      pressure, this.powerFaces.control, 1, this.leafSolver === "mgpcg" ? this.mgpcg?.control : undefined);
    this.powerSolidFaces?.encodePostProjectionConstraint(encoder);
  }

  private encodePowerVelocityPublication(encoder: GPUCommandEncoder): void {
    if (!this.powerPolicy.authoritative || !this.powerFaceSeed || !this.powerOperator
      || !this.powerFaces || !this.powerVelocity) return;
    const faces = this.powerFaces.source;
    this.powerVelocity.encodeFromFaceControl(encoder, {
      faces: faces.faces,
      faceNormals: faces.faceNormals,
      incidenceRows: faces.incidenceRows,
      incidences: faces.incidence,
    }, faces.control, {
      maximumIncidencePerRow: OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
      generation: this.powerGeneration,
      projectionControl: this.powerOperator.control,
    });
    this.powerFaceSeed.encodePowerToAxis(encoder, this.powerOperator.control, true);
    this.powerFaceTransfer?.encodeCapture(encoder);
  }

  private encodeFrontierRows(
    encoder: GPUCommandEncoder,
    label: string,
    group = this.groups.ab,
    beginning?: GPUComputePassTimestampWrites,
  ): void {
    const compact = encoder.beginComputePass({ label, ...(beginning ? { timestampWrites: beginning } : {}) });
    compact.setPipeline(this.planPipeline); compact.setBindGroup(0, group);
    compact.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
    compact.setPipeline(this.scanPipeline); compact.dispatchWorkgroups(1, 1, 1);
    compact.setPipeline(this.emitPipeline); compact.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
    compact.end();
    encoder.copyBufferToBuffer(this.compaction, 8, this.solveDispatch, 0, 24);
    encoder.copyBufferToBuffer(this.compaction, this.compactionByteLength - 20, this.pressureOverflowDispatch, 0, 12);
  }

  encode(
    encoder: GPUCommandEncoder,
    _nx: number,
    _ny: number,
    _nz: number,
    timestampWrites?: GPUComputePassTimestampWrites,
    detailedTimestampWrites?: {
      projection?: GPUComputePassTimestampWrites;
      extrapolation?: GPUComputePassTimestampWrites;
      materialization?: GPUComputePassTimestampWrites;
    }
  ) {
    const linear = (pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, blocks: number, group = this.groups.ab) => {
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(blocks, 1, 1);
    };
    const gatherBodyCoupling = (pass: GPUComputePassEncoder, group: GPUBindGroup) => {
      if (!this.couplingHasDynamicBodies) return;
      pass.setPipeline(this.gatherBodyCouplingPipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(12, 1, 1);
    };
    // A monolithic rank-six response needs one global K^T p reduction per
    // pressure iterate. The optional single-workgroup megakernel cannot launch
    // those reductions. Chebyshev instead treats the uploaded rigid velocity
    // as prescribed for this solve and returns the new pressure impulse for the
    // next presentation batch: a deliberately frame-lagged partitioned split.
    const useMegakernel = this.leafSolver === "megakernel" && !this.couplingHasDynamicBodies;
    const useChebyshev = this.leafSolver === "chebyshev";
    const useMGPCG = this.leafSolver === "mgpcg" && this.mgpcg !== undefined;
    const useLaggedRigidCoupling = useChebyshev && this.couplingHasDynamicBodies;
    const solvePasses = useChebyshev ? this.encodedSolvePasses
      : useMGPCG ? this.mgpcg!.iterationBudget : this.iterations;
    this.info.pressureIterationsUsed = solvePasses;
    this.info.pressureIterationBudget = solvePasses;
    this.info.pressureIterationHardBudget = useChebyshev ? Math.ceil(this.iterations / 4) : solvePasses;
    let pressureBufferSwaps = this.iterations;
    if (this.leafSolver === "dense") {
      encoder.clearBuffer(this.pressureA); encoder.clearBuffer(this.pressureB);
      const pressure = encoder.beginComputePass({ label: "Octree leaf Jacobi solve", ...(timestampWrites ? { timestampWrites } : {}) });
      for (let iteration = 0; iteration < this.iterations; iteration += 1) {
        const group = iteration % 2 === 0 ? this.groups.ab : this.groups.ba;
        gatherBodyCoupling(pressure, group);
        this.dispatch(pressure, this.jacobiPipeline, group);
      }
      pressure.end();
      // Dense Jacobi remains a validation baseline, but downstream affine
      // reconstruction/projection still consumes the persistent leaf rows.
      this.encodeFrontierRows(encoder, "Octree validation frontier rows");
    } else {
      // Compact the liquid leaf origins with a prefix-sum scan, assemble each
      // row's cached diagonal / flux / merged neighbor table once, then solve.
      // Emission remaps the previous converged row field into the opposite
      // buffer before publishing new origin->row words. This preserves warm
      // starts even when compaction order changes without a dense pressure map.
      const remapGroup = this.latestPressureInA ? this.groups.ab : this.groups.ba;
      this.encodeFrontierRows(encoder, "Octree leaf compaction", remapGroup, timestampWrites?.beginningOfPassWriteIndex !== undefined ? { querySet: timestampWrites.querySet, beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } : undefined);
      const initialInA = !this.latestPressureInA;
      const groupForIteration = (iteration: number) => (initialInA === (iteration % 2 === 0)) ? this.groups.ab : this.groups.ba;
      let pressure = encoder.beginComputePass({ label: "Octree leaf pressure assembly", ...(!this.faceMirror && !this.powerDescriptor && timestampWrites?.endOfPassWriteIndex !== undefined ? { timestampWrites: { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } } : {}) });
      pressure.setPipeline(this.assemblePipeline);
      pressure.setBindGroup(0, groupForIteration(0));
      pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      pressure.setPipeline(this.assembleCoarsePipeline);
      pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 12);
      if (this.faceMirror || this.powerDescriptor) {
        pressure.end();
        // Section 4.3's middle V-cycle uses L1, not the power matrix. Capture
        // the exact Cartesian/GFM rows before power publication replaces the
        // shared compact header/entry buffers with L2 coefficients.
        this.firstOrderVCycle?.encodeCapture(encoder);
        if (this.faceMirror) {
          this.faceMirror.encodeTopology(encoder, this.solveDispatch);
          this.faceTransport?.encode(encoder, {
            dt: this.faceTransportDt_s,
            acceleration: [
              this.scene.fluid.gravity_m_s2.x,
              this.scene.fluid.gravity_m_s2.y,
              this.scene.fluid.gravity_m_s2.z,
            ],
            reseedFromMirror: true,
          });
          this.solidFaces?.encodeClassifyAndConstrain(encoder);
          this.faceMirror.encodeRhs(encoder, this.solveDispatch, this.faceRhsAuthority);
        }
        this.encodePowerAssemblyMirror(encoder);
        pressure = encoder.beginComputePass({ label: "Octree leaf pressure solve", ...(timestampWrites?.endOfPassWriteIndex !== undefined ? { timestampWrites: { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } } : {}) });
      }
      if (useMGPCG) {
        // The row assembly pass must be closed before the standalone solver
        // emits its matrix-free hierarchy and Krylov passes.  Emission put the
        // warm start in initialInA; MGPCG publishes into the opposite existing
        // pressure buffer and keeps Chebyshev's two-buffer rollback intact.
        pressure.end();
        const pressureIn = initialInA ? this.pressureA : this.pressureB;
        const pressureOut = initialInA ? this.pressureB : this.pressureA;
        this.mgpcg!.encode(encoder, pressureIn, pressureOut);
        pressureBufferSwaps = 1;
        this.latestPressureInA = !initialInA;
      } else if (useMegakernel) {
        linear(pressure, this.solvePipeline, 1, groupForIteration(0));
        pressureBufferSwaps = 0;
      } else if (useChebyshev) {
        const acceleratedIterations = this.encodedSolvePasses;
        pressureBufferSwaps = acceleratedIterations;
        for (let iteration = 0; iteration < acceleratedIterations; iteration += 1) {
          pressure.setPipeline(this.iterateChebyshevPipeline);
          pressure.setBindGroup(0, groupForIteration(iteration));
          pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
        }
        // Residual feedback for the adaptive budget: bind the parity whose
        // pressureIn is the final iterate, reduce per-workgroup partials into
        // the dead scan words, then fold them into the trailing feedback words.
        const finalGroup = groupForIteration(acceleratedIterations);
        pressure.setPipeline(this.reduceResidualPartialsPipeline);
        pressure.setBindGroup(0, finalGroup);
        pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
        pressure.setPipeline(this.reduceResidualTotalPipeline);
        pressure.dispatchWorkgroups(1);
      } else {
        pressureBufferSwaps = this.iterations;
        for (let sweep = 0; sweep < this.iterations; sweep += 1) {
          const group = groupForIteration(sweep);
          gatherBodyCoupling(pressure, group);
          pressure.setPipeline(this.iteratePipeline); pressure.setBindGroup(0, group);
          pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
        }
      }
      if (!useMGPCG) {
        pressure.end();
        const finalInA = pressureBufferSwaps % 2 === 0 ? initialInA : !initialInA;
        this.latestPressureInA = finalInA;
      }
    }
    // Dense validation assembles its compact rows after solving, so it can
    // publish an observational mirror but cannot select face-derived RHS.
    if (this.leafSolver === "dense") this.faceMirror?.encode(encoder, this.solveDispatch, false);
    // Stage solve feedback (residual sums + row/entry counts) while this
    // encoder still owns write ordering on compaction; the async diagnostics
    // poll then reads the staging buffer without racing the next rebuild.
    encoder.copyBufferToBuffer(this.compaction, this.compactionByteLength - 32, this.solveStats, 0, 32);
    // The megakernel folds its final iterate back into pressure A; the fixed-
    // count ladders land in A exactly when the sweep count is even.
    const finalInA = this.rowIndexedPressure ? this.latestPressureInA : pressureBufferSwaps % 2 === 0;
    this.latestPressureInA = finalInA;
    this.encodePowerProjectionMirror(encoder, finalInA ? this.pressureA : this.pressureB);
    // U2 authority: update the canonical adaptive normal velocities with the
    // same matched pressure jump used by the compatibility texture projector.
    // Unsupported terrain/rigid/hydrostatic operators remain fail-closed at
    // construction; the observational mirror still computes parity.
    this.faceMirror?.encodeProjection(encoder, finalInA);
    // The axis projection above remains available only to explicit
    // compatibility modes. In authoritative mode the power publication either
    // replaces it completely or marks the compact face authority invalid.
    this.encodePowerVelocityPublication(encoder);
    this.solidFaces?.encodePostProjectionConstraint(encoder);
    // Paper Section 5 velocity extrapolation: projected power faces are first
    // reconstructed onto regular octree faces, constrained at solids, then
    // fast-marched through the current fine narrow band before factor-m
    // trajectories are traced.
    this.encodeGlobalFineFaceBand(encoder);
    // Diagnose the actual authority visible to fine transport: axis rollback
    // for off/mirror, or the all-or-nothing power-to-axis reconstruction.
    this.faceMirror?.encodeProjectedDivergence(encoder);
    const finalGroup = finalInA ? this.groups.ab : this.groups.ba;
    const project = encoder.beginComputePass({ label: "Octree finite-volume velocity projection", ...(detailedTimestampWrites?.projection ? { timestampWrites: detailedTimestampWrites.projection } : {}) });
    // The exact ladder refreshes M^-1 K^T p from the converged pressure so
    // projection sees the same-step solid response. The accelerated coupled
    // path intentionally skips that dependency: projection uses the rigid
    // velocity uploaded from the previous coupling batch, while the following
    // impulse pass publishes the current K^T p for the next batch.
    if (!useLaggedRigidCoupling) gatherBodyCoupling(project, finalGroup);
    if (this.couplingHasDynamicBodies && !useLaggedRigidCoupling) {
      project.setPipeline(this.applyBodyCouplingPipeline); project.setBindGroup(0, finalGroup); project.dispatchWorkgroups(1, 1, 1);
    }
    // With no registered coupling bodies every face fails the owner test, so
    // the dense impulse sweep would touch the whole grid to accumulate nothing.
    if (this.couplingBodyCount > 0 && !this.solidFaces) {
      project.setPipeline(this.pressureImpulsePipeline);
      project.setBindGroup(0, finalInA ? this.couplingGroups.pressureA : this.couplingGroups.pressureB);
      project.dispatchWorkgroups(...this.workgroups);
    }
    // A leaf-constant pressure basis constrains the integrated leaf flux but
    // has no gradient on the dense faces inside a coarse leaf. Reconstruct an
    // affine gradient into non-origin slots of the alternate pressure buffer;
    // project then applies it only to those internal faces, leaving every
    // solved coarse/fine boundary flux unchanged.
    project.setPipeline(this.reconstructSmallGradientsPipeline); project.setBindGroup(0, finalGroup);
    project.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
    project.setPipeline(this.reconstructGradientsPipeline); project.dispatchWorkgroupsIndirect(this.solveDispatch, 12);
    if (this.faceTransport || this.extrapolationSweeps > 0) {
      project.setPipeline(this.projectSmallLeavesPipeline); project.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      project.setPipeline(this.projectLeavesPipeline); project.dispatchWorkgroupsIndirect(this.solveDispatch, 12);
    } else {
      this.dispatch(project, this.projectPipeline, finalGroup);
    }
    project.setPipeline(this.pressureOverflowPipeline); project.setBindGroup(0, finalGroup);
    project.dispatchWorkgroupsIndirect(this.pressureOverflowDispatch, 0);
    project.end();
    if (this.powerPolicy.authoritative && this.powerSolidFaces) this.powerSolidFaces.encodePressureImpulses(encoder, finalInA);
    else this.solidFaces?.encodePressureImpulses(encoder, finalInA);
    if (this.adaptiveSurfacePages && this.faceMirror) encoder.clearBuffer(this.faceMirror.parity,16,16);
    else this.faceMirror?.encodeProjectionParity(encoder, finalInA);
    const sparseExtrapolation = this.usesSparseVelocityExtrapolation;
    const bulkWorklist = this.sparseBrickWorld?.bulkResidencyWorklist;
    for (let sweep = 0; sweep < this.extrapolationSweeps; sweep += 1) {
      const timing = detailedTimestampWrites?.extrapolation;
      const extrapolationTimestampWrites = timing && (sweep === 0 || sweep === this.extrapolationSweeps - 1) ? {
        querySet: timing.querySet,
        ...(sweep === 0 && timing.beginningOfPassWriteIndex !== undefined ? { beginningOfPassWriteIndex: timing.beginningOfPassWriteIndex } : {}),
        ...(sweep === this.extrapolationSweeps - 1 && timing.endOfPassWriteIndex !== undefined ? { endOfPassWriteIndex: timing.endOfPassWriteIndex } : {})
      } : undefined;
      const extrapolate = encoder.beginComputePass({ label: "Octree narrow-band velocity extrapolation", ...(extrapolationTimestampWrites ? { timestampWrites: extrapolationTimestampWrites } : {}) });
      const group = sweep % 2 === 0 ? this.groups.extrapolateOut : this.groups.extrapolateScratch;
      extrapolate.setPipeline(sparseExtrapolation
        ? (sweep === 0 ? this.extrapolateSeedSparsePipeline : this.extrapolateSparsePipeline)
        : (sweep === 0 ? this.extrapolateSeedPipeline : this.extrapolatePipeline));
      extrapolate.setBindGroup(0, group);
      if (sparseExtrapolation && bulkWorklist) {
        extrapolate.dispatchWorkgroupsIndirect(bulkWorklist, FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES);
      } else extrapolate.dispatchWorkgroups(...this.workgroups);
      extrapolate.end();
    }
    // An odd sweep ends in scratch; copy it back to the solver's authoritative velocity.
    if (this.extrapolationSweeps % 2 === 1) {
      if (sparseExtrapolation && bulkWorklist) {
        const copy = encoder.beginComputePass({ label: "Copy sparse extrapolated velocity" });
        copy.setPipeline(this.copyExtrapolatedSparsePipeline);
        copy.setBindGroup(0, this.groups.extrapolateScratch);
        copy.dispatchWorkgroupsIndirect(bulkWorklist, FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES);
        copy.end();
      } else encoder.copyTextureToTexture({ texture: this.resources.velocityScratch }, { texture: this.resources.velocityOut }, [this.dims.nx, this.dims.ny, this.dims.nz]);
    }
    this.encodeOverlayMaterialization(encoder, finalInA, detailedTimestampWrites?.materialization);
  }

  /** Publish lazily allocated diagnostic textures from the live owner map.
   * The first overlay request materializes immediately, so reset-time grid
   * inspection never decodes zero-initialized topology storage as finest 1^3. */
  encodeOverlayMaterialization(encoder: GPUCommandEncoder, pressureInA = this.latestPressureInA, timestampWrites?: GPUComputePassTimestampWrites) {
    if (!this.diagnosticGroups) return false;
    const materialize = encoder.beginComputePass({ label: "Materialize octree overlay fields", ...(timestampWrites ? { timestampWrites } : {}) });
    materialize.setPipeline(this.materializePipeline);
    materialize.setBindGroup(0, pressureInA ? this.diagnosticGroups.pressureA : this.diagnosticGroups.pressureB);
    materialize.dispatchWorkgroups(...this.workgroups);
    materialize.end();
    return true;
  }

  encodeSurface(encoder: GPUCommandEncoder, dt_s: number, inflow?: SurfaceInflowState, _maximumDt_s?: number, timestampWrites?: GPUComputePassTimestampWrites) {
    if (this.adaptiveSurfaceAdapter && this.adaptiveSurfacePages) {
      let coarseBootstrappedThisStep = false;
      // The page-native field remains the explicit differential/rollback
      // oracle during the global-fine migration.  Its recurrence is governed
      // by GPU publication validity in the hybrid consumer, never by the
      // optimistic CPU-side fine-generation counter.
      this.adaptiveSurfaceAdapter.encode(encoder);
      if (this.powerCoarseLevelSet && this.powerCoarseLevelSetSchedule && this.powerVelocity && this.powerFaces) {
        if (!this.powerCoarseLevelSetBootstrapped) {
          this.powerCoarseLevelSet.encodeBootstrapFromSurfaceLeaves(encoder, this.adaptiveSurfaceAdapter.leaves);
          this.powerCoarseLevelSetSchedule.encode(encoder, {
            headers: this.leafHeaders, cellVelocities: this.powerVelocity.velocities,
            siteIndex: this.powerFaces.source.siteIndex, rowCount: this.compaction,
          }, {
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            dt: 0, hashCapacity: this.powerFaces.plan.hashCapacity,
            maximumLeafSize: this.maxLeafSize, generation: this.globalFineGeneration & 0x3fff_ffff,
          });
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
        const seeds = this.globalFineSeeds.encodeFromAllInterfaceLeaves(
          encoder, { buffer: this.adaptiveSurfaceAdapter.leaves }, { buffer: this.compaction },
        );
        const compactCoarseEntry: GPUBindGroupEntry = { binding: 9,
          resource: { buffer: this.powerCoarseLevelSetSchedule!.sampleSource.directory } };
        const bandCells = Math.min(256, Math.max(4,
          this.interfaceRefinementBandCells * (this.globalFineLevelSet?.plan.fineFactor ?? 4)));
        const transport = this.globalFineCurrentIsA ? this.globalFineTransportA : this.globalFineTransportB;
        let transportEncoded = false;
        if (this.globalFineBootstrapped && transport && this.powerFaceSeed && this.powerVelocity) {
          this.lastGlobalFineTransport = transport;
          transport.encode(encoder, {
            timestep: dt_s,
            headers: this.leafHeaders,
            rowVelocities: this.powerVelocity.velocities,
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            maximumLeafSize: this.maxLeafSize,
            ownerTopology: this.topology,
            powerTopology: this.powerTopology!.source,
            generation: this.powerGeneration,
            boundaryPolicy: "closed-neumann",
            transportBandCells: Math.min(256, Math.max(4,
              this.interfaceRefinementBandCells * (this.globalFineLevelSet?.plan.fineFactor ?? 4))),
          });
          transportEncoded = true;
        }
        let publicationTopology: WebGPUFineLevelSetTopology;
        let publicationRedistance: WebGPUFineLevelSetRedistance;
        let publicationVolume: WebGPUFineLevelSetVolumeCorrection | undefined;
        const publicationTransport = transportEncoded ? transport : undefined;
        if (this.globalFineCurrentIsA) {
          if (this.globalFineBootstrapped) {
            this.globalFineGeneration += 1;
            this.globalFineLevelSet!.repurposeGPUGeneration(this.globalFineSourceB!, this.globalFineGeneration);
          }
          publicationTopology = this.globalFineTopologyAB;
          publicationRedistance = this.globalFineRedistanceB;
          publicationVolume = this.globalFineVolumeB;
          publicationTopology.encode(encoder, seeds, [compactCoarseEntry], {
            // The octree timestep is bounded at one finest effective cell.
            // Express that same physical displacement on the fine lattice.
            maximumBacktraceFineCells: this.globalFineLevelSet!.plan.fineFactor,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells: bandCells,
            safetyBrickRings: 1,
          }, true);
          publicationRedistance.encode(encoder, { bandCells: Math.min(256,
            bandCells + this.globalFineLevelSet!.plan.fineFactor + 1), residualTolerance: 1 });
          publicationVolume?.encode(encoder);
        } else {
          this.globalFineGeneration += 1;
          this.globalFineLevelSet!.repurposeGPUGeneration(this.globalFineSourceA!, this.globalFineGeneration);
          publicationTopology = this.globalFineTopologyBA;
          publicationRedistance = this.globalFineRedistanceA;
          publicationVolume = this.globalFineVolumeA;
          publicationTopology.encode(encoder, seeds, [compactCoarseEntry], {
            maximumBacktraceFineCells: this.globalFineLevelSet!.plan.fineFactor,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells: bandCells,
            safetyBrickRings: 1,
          }, true);
          publicationRedistance.encode(encoder, { bandCells: Math.min(256,
            bandCells + this.globalFineLevelSet!.plan.fineFactor + 1), residualTolerance: 1 });
          publicationVolume?.encode(encoder);
        }
        publicationTopology.encodeFinalizePublication(encoder, {
          redistance: publicationRedistance.control,
          ...(publicationVolume ? { volume: publicationVolume.control } : {}),
          ...(publicationTransport ? { transport: publicationTransport.control } : {}),
        });
        const correctedFine = this.globalFineCurrentIsA ? this.globalFineSourceB : this.globalFineSourceA;
        if (correctedFine && this.fineToPowerCoarseLevelSet && this.powerCoarseLevelSetSchedule
          && this.powerVelocity && this.powerFaces) {
          const correction = this.fineToPowerCoarseLevelSet.encode(encoder, correctedFine, {
            headers: this.leafHeaders, siteIndex: this.powerFaces.source.siteIndex, rowCount: this.compaction,
            topologyControl: publicationTopology.control,
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            maximumLeafSize: this.maxLeafSize, siteHashCapacity: this.powerFaces.plan.hashCapacity,
          });
          this.powerCoarseLevelSetSchedule.encode(encoder, {
            headers: this.leafHeaders, cellVelocities: this.powerVelocity.velocities,
            siteIndex: this.powerFaces.source.siteIndex, rowCount: this.compaction,
            fineCorrection: { rowOffsets: correction.rowOffsets, contributions: correction.contributions,
              contributionCount: correction.counts, aggregated: correction.aggregated },
          }, {
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            dt: coarseBootstrappedThisStep ? 0 : dt_s, hashCapacity: this.powerFaces.plan.hashCapacity,
            maximumLeafSize: this.maxLeafSize, generation: correctedFine.generation & 0x3fff_ffff,
          });
        }
        if (correctedFine && this.powerCoarseLevelSetSchedule) {
          const coarse = this.powerCoarseLevelSetSchedule.sampleSource;
          this.globalFineSummaries?.encode(encoder, correctedFine,
            { directory: coarse.directory, hashCapacity: coarse.hashCapacity });
        }
        this.globalFineCurrentIsA = !this.globalFineCurrentIsA;
        this.globalFineBootstrapped = true;
      }
      this.adaptiveSurfacePages.encodeLifecycle(encoder);
      this.adaptiveSurfacePages.encodeTransport(encoder, dt_s);
      // Keep the transported page field intact while the page-local
      // redistance operator is disabled; its cross-page stencil currently
      // creates false zero crossings at compact-leaf boundaries.
      this.adaptiveSurfacePages.encodeVolumeCorrection(encoder);
      // One bootstrap publication seeds legacy structural/SVO compatibility.
      // Once page authority is live, do not expand it back to a box-sized phi
      // texture merely so the brick scheduler can classify it again.
      if (!this.surfacePagesBootstrapped) this.adaptiveSurfacePages.encodeDensePublication(encoder);
      // Bootstrap once from the uploaded dense field, then make page/leaf phi
      // the topology authority. Commands already encoded in this submission
      // retain the legacy group; later rebuilds use the compact bindings.
      if (this.directPagedTopology && !this.surfacePagesBootstrapped && this.pagedGroups) {
        this.surfacePagesBootstrapped = true;
        this.groups = this.pagedGroups;
        // Rebuild every live page consumer with the format-only texture before
        // the dense bootstrap texture is retired. Even an untaken WGSL branch
        // cannot leave a destroyed resource attached to a submitted bind group.
        this.faceMirror?.setSurfacePageSource(this.adaptiveSurfacePages.source, this.levelSetFallbackTexture!);
        this.adaptiveSurfaceAdapter.setSurfacePageSource(this.adaptiveSurfacePages.source, this.levelSetFallbackTexture!);
        this.couplingGroups = {
          pressureA: this.createCouplingGroup(this.pressureA, this.levelSetFallbackTexture!),
          pressureB: this.createCouplingGroup(this.pressureB, this.levelSetFallbackTexture!),
        };
      }
      if (this.globalFineLevelSet && !this.surfacePagesBootstrapped) this.surfacePagesBootstrapped = true;
      return;
    }
    // Compact transport has no trustworthy dense velocity texture. Until the
    // paged arena can cover the live interface, preserve the last valid phi
    // instead of advecting it with a 1x1 compatibility texture.
    if (this.faceTransport) return;
    this.surfaceState.encode(encoder, dt_s, inflow, timestampWrites);
    this.encodeSurfaceBand(encoder, dt_s);
  }
  encodeSurfaceBand(encoder: GPUCommandEncoder, dt_s: number) {
    this.sparseSurfaceBand?.encode(encoder, dt_s);
  }
  addSurfaceReferenceVolumeCells(cells: number) { this.surfaceState.addReferenceVolumeCells(cells); }
  async readSolveDiagnostics() {
    // The staging buffer was copied inside the solve encoder, so it can never
    // race the next rebuild's worklist copy over the compaction header. It
    // carries [overflow, required rows, required entries, fallback dispatch xyz,
    // sum r^2, sum b^2] from the latest solve.
    const readback = this.device.createBuffer({
      label: "Octree live pressure-row diagnostics",
      size: 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read octree pressure-row diagnostics" });
    encoder.copyBufferToBuffer(this.solveStats, 0, readback, 0, 32);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange(0, 32));
      const residuals = new Float32Array(words.buffer, words.byteOffset + 24, 2);
      const overflow = words[0] !== 0;
      const liquidRows = words[1];
      this.info.pressureCapacityOverflow = overflow;
      this.info.frontierCapacityOverflow = (words[0] & 2) !== 0;
      this.info.frontierRequiredLeaves = words[1];
      this.info.pressureRequiredRows = words[1];
      this.info.pressureRequiredEntries = words[2];
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
        this.updateSolveBudget(rr, bb, liquidRows);
      } else {
        this.residualRms = undefined;
        this.initialResidualRms = undefined;
        this.relativeResidual = undefined;
      }
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** One-time startup proof for the paper's Section 4.3 pressure authority.
   * Regular simulation scheduling never consumes this readback; the paused
   * t=0 transport gate uses it only after every initialization phase fenced. */
  async readMGPCGDiagnostics() {
    if (!this.mgpcg) return undefined;
    const readback = this.device.createBuffer({
      label: "Octree t=0 MGPCG authority diagnostics",
      size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read octree t=0 MGPCG authority" });
    encoder.copyBufferToBuffer(this.mgpcg.control, 0, readback, 0, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      return Uint32Array.from(new Uint32Array(readback.getMappedRange(0, 64)));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /**
   * Adaptive Chebyshev budget from asynchronous relative-residual feedback.
   * Warm-started calm scenes converge in a couple of polynomial passes, so a
   * fixed ladder wastes most of its dispatches; conversely a topology shift or
   * a residual above tolerance snaps the budget straight back up.
   */
  private updateSolveBudget(rr: number, bb: number, rows: number) {
    if (this.leafSolver !== "chebyshev") return;
    // Compact face transport can submit several topology generations before
    // an asynchronous residual map resolves. Reducing the next ladders from
    // stale feedback makes stability depend on CPU/GPU scheduling. Keep its
    // full deterministic budget until feedback is consumed on-GPU.
    if (this.faceTransport) return;
    if (typeof process !== "undefined" && process.env?.FLUID_OCTREE_ADAPTIVE_PRESSURE === "0") return;
    const cap = Math.ceil(this.iterations / 4);
    const floor = Math.min(cap, 2);
    const rowsShifted = this.lastFeedbackRows > 0 && Math.abs(rows - this.lastFeedbackRows) > 0.01 * this.lastFeedbackRows;
    this.lastFeedbackRows = rows;
    if (rowsShifted) { this.encodedSolvePasses = cap; return; }
    if (!Number.isFinite(rr) || !Number.isFinite(bb) || rr < 0 || bb < 0) return;
    // ||r|| <= 1e-4 ||b|| — the same |b|-relative bar the megakernel and the
    // quadtree ladder use, with a small absolute slack for near-zero systems.
    const converged = rr <= 1e-8 * bb + 1e-12;
    this.encodedSolvePasses = converged
      ? Math.max(floor, Math.floor(this.encodedSolvePasses * 0.75))
      : Math.min(cap, Math.max(this.encodedSolvePasses + 1, this.encodedSolvePasses * 2));
  }
  get surfaceDiagnostics() {
    if (!this.adaptiveSurfacePages) return this.surfaceState.volumeDiagnostics;
    const base = this.surfaceState.volumeDiagnostics;
    const pages = this.adaptiveSurfacePages.volumeDiagnostics;
    const band = 4 * (this.scene.container.height_m / this.dims.ny);
    const correctedDelta = pages.transportedVolumeCells - pages.referenceVolumeCells
      - pages.interfacePages * pages.correctionShiftCells * Math.min(
        this.scene.container.width_m / this.dims.nx,
        this.scene.container.height_m / this.dims.ny,
        this.scene.container.depth_m / this.dims.nz,
      ) / Math.max(1e-9, band);
    return {
      ...base,
      volumeCells: base.referenceVolumeCells + correctedDelta,
      interfaceCells: pages.activePages,
      correctionSpeed: pages.correctionShiftCells / Math.max(1e-9, this.faceTransportDt_s),
      mismatchFraction: pages.overflow ? 1 : 0,
    };
  }
  async readSurfaceDiagnostics() {
    if (!this.adaptiveSurfacePages) return this.surfaceState.readVolumeDiagnostics();
    await this.adaptiveSurfacePages.readDiagnostics();
    const differential = await this.readPagedPhiDifferential();
    if (differential) console.info(JSON.stringify({ phase: "octree-paged-phi-differential", ...differential }));
    return this.surfaceDiagnostics;
  }
  encodeBodyImpulseReadback(_encoder: GPUCommandEncoder) { return undefined; }
  readBodyImpulseReadback(_buffer: GPUBuffer) { return Promise.resolve([]); }
  destroySharedSurface() { /* The octree owns its surface for its full lifetime. */ }
  get levelSetTexture() { return this.denseBootstrapPhiReleased ? this.levelSetFallbackTexture! : this.surfaceState.texture; }
  get hasDenseLevelSetPublication() { return !this.denseBootstrapPhiReleased; }
  /** Release the last box-sized phi field after its bootstrap commands submit. */
  releaseDenseBootstrapPhi() {
    if (this.denseBootstrapPhiReleased) return 0;
    // Rigid/terrain coupling, the differential, and scientific overlays still
    // consume dense phi and therefore explicitly gate lifetime cutover. Every
    // recurring compact consumer must also attest that its bind group was
    // rebuilt onto page buffers plus the live format-only fallback texture.
    if (!octreeDensePhiReleaseReady({
      directPagedTopology: this.directPagedTopology,
      surfacePagesBootstrapped: this.surfacePagesBootstrapped,
      pagedProjectionGroupsActive: this.pagedGroups !== undefined && this.groups === this.pagedGroups,
      faceGroupsPageNative: this.faceMirror?.hasPageNativePhiBindings === true,
      surfaceAdapterPageNative: this.adaptiveSurfaceAdapter?.hasPageNativePhiBindings === true,
      topologyUsesSurfaceCandidates: this.topologyWorklistReady,
      compactRendererSourceReady: this.adaptiveFaceVelocityAuthority && this.adaptiveSurfaceAuthority,
      incompatibleDenseConsumer: Boolean(this.pagedPhiDifferential || this.diagnosticGroups
        || (this.globalFineLevelSet && !this.globalFineBootstrapped)
        || this.scene.rigidBodies.length > 0 || sceneHasTerrain(this.scene)),
    })) return 0;
    this.adaptiveSurfacePages?.releaseDensePublicationBinding();
    const releasedBytes = this.surfaceState.releasePresentationTexture();
    if (releasedBytes > 0) {
      this.denseBootstrapPhiReleased = true;
      this.info.allocatedBytes = Math.max(0, this.info.allocatedBytes - releasedBytes);
    }
    return releasedBytes;
  }
  get sparseVoxelSceneSource() { return this.sparseBrickWorld?.sceneSource; }
  get sparseVoxelRenderSource() {
    if (this.sparseBrickWorld) {
      const source = this.sparseBrickWorld.ensureInspectionSource();
      const currentBytes = this.sparseBrickWorld.allocatedBytes;
      this.info.allocatedBytes += currentBytes - this.sparseBrickWorldAccountedBytes;
      this.sparseBrickWorldAccountedBytes = currentBytes;
      return source;
    }
    if (!this.rowIndexedPressure) return undefined;
    if (!this.compactVoxelInspection) {
      this.compactVoxelInspection = new CompactOctreeVoxelInspection(
        this.device,
        this.scene,
        [this.dims.nx, this.dims.ny, this.dims.nz],
        {
          leafHeaders: { buffer: this.leafHeaders },
          rowCount: { buffer: this.compaction },
          rowCapacity: this.pressureCapacity.rowCapacity,
        },
      );
      this.info.allocatedBytes += this.compactVoxelInspection.allocatedBytes;
      // Compact pressure headers are build-time scratch rather than an idle
      // publication. Rebuild once on first inspection and capture the rows in
      // the same submission, so paused t=0 scenes do not require STEP.
      this.compactVoxelInspection.source.inspectionPublication?.setEnabled(true);
      const encoder = this.device.createCommandEncoder({ label: "Bootstrap compact octree voxel inspection" });
      // The resident scheduler is allowed to be empty while paused. Force the
      // one-time debug rebuild over the full compact domain; later simulation
      // frames return to the normal resident scheduler immediately.
      const topologyWorklistReady = this.topologyWorklistReady;
      this.topologyWorklistReady = false;
      this.encodeInlineRebuild(encoder);
      this.topologyWorklistReady = topologyWorklistReady;
      this.encodeFrontierRows(encoder, "Octree inspection frontier rows",
        this.latestPressureInA ? this.groups.ab : this.groups.ba);
      this.compactVoxelInspection.encode(encoder);
      this.device.queue.submit([encoder.finish()]);
    }
    return this.compactVoxelInspection.source;
  }
  get fluidBrickAtlasSamplingSource(): FluidBrickAtlasSamplingSource | undefined { return this.sparseBrickWorld?.atlasSamplingSource; }
  get usesSparseVelocityExtrapolation() {
    const source = this.sparseBrickWorld?.atlasSamplingSource;
    return Boolean(this.sparseExtrapolationRequested && source?.mode === "mirror" && source.sparseDispatchSafe);
  }
  get sparseSurfaceBandSource(): SparseSurfaceBandGPUSource | undefined { return this.sparseSurfaceBand?.source; }
  get adaptiveFaceMirrorSource() { return this.faceMirror?.source; }
  get adaptiveFaceVelocitySource() { return this.faceTransport?.velocitySource; }
  get powerFaceTransferControl() { return this.powerFaceTransfer?.control; }
  get powerFaceSeedControl() { return this.powerFaceSeed?.control; }
  get powerOperatorControl() { return this.powerOperator?.control; }
  /** QA-only MGPCG status; simulation authority consumes this buffer directly on GPU. */
  get mgpcgControl() { return this.mgpcg?.control; }
  get powerFaceControl() { return this.powerFaces?.control; }
  get powerFaceSiteIndex() { return this.powerFaces?.source.siteIndex; }
  get powerDescriptorControl() { return this.powerDescriptor?.control; }
  get powerTopologyControl() { return this.powerTopology?.control; }
  get powerDescriptorRows() { return this.powerDescriptor?.descriptors; }
  get powerTopologyMetrics() { return this.powerTopology?.metrics; }
  get powerCatalogEntryHeaders() { return this.powerTopology?.catalogEntryHeaders; }
  get powerCatalogFaces() { return this.powerTopology?.catalogFaces; }
  get techniqueDebugSource() {
    const surface = this.adaptiveSurfacePages?.source;
    const topology = this.powerTopology?.source;
    const faces = this.powerFaces?.source;
    const tetrahedronHeaders = topology?.catalogTetrahedronHeaders;
    const tetrahedra = topology?.catalogTetrahedra;
    const tetrahedronVertices = topology?.catalogTetrahedronVertices;
    if (!surface || !topology || !faces || !tetrahedronHeaders || !tetrahedra || !tetrahedronVertices) return undefined;
    const fine = this.globalFineBootstrapped
      ? (this.globalFineCurrentIsA ? this.globalFineSourceA : this.globalFineSourceB)
      : undefined;
    const fineTopology = this.globalFineCurrentIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    const fineRedistance = this.globalFineCurrentIsA ? this.globalFineRedistanceA : this.globalFineRedistanceB;
    const fineBandLifecycle = fine && fineTopology && fineRedistance ? {
      params: { buffer: fine.params },
      hash: { buffer: fine.hash },
      metadata: { buffer: fine.metadata },
      worklist: { buffer: fine.worklist },
      sampleFlags: { buffer: fine.flags },
      phi: { buffer: fine.phi },
      topologyControl: { buffer: fineTopology.control },
      redistanceControl: { buffer: fineRedistance.control },
    } : undefined;
    const faceBand = this.globalFineFaceFastMarch?.source;
    const section5FaceBand = faceBand ? {
      rowHash: { buffer: faceBand.rowHash }, rows: { buffer: faceBand.rows },
      faces: { buffer: faceBand.faces }, incidence: { buffer: faceBand.incidence },
      states: { buffer: faceBand.state }, control: { buffer: faceBand.control },
      transitionControl: { buffer: faceBand.transitionControl },
    } : undefined;
    return {
      leaves: surface.leaves,
      topologyMetrics: { buffer: topology.metrics },
      tetrahedronHeaders: { buffer: tetrahedronHeaders },
      tetrahedra: { buffer: tetrahedra },
      tetrahedronVertices: { buffer: tetrahedronVertices },
      powerFaces: { buffer: faces.faces },
      faceNormals: { buffer: faces.faceNormals },
      faceCentroids: { buffer: faces.faceCentroids },
      incidenceRows: { buffer: faces.incidenceRows },
      incidence: { buffer: faces.incidence },
      faceControl: { buffer: faces.control },
      leafHeaders: { buffer: this.leafHeaders },
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
      ...(section5FaceBand ? { section5FaceBand } : {}),
      generation: this.powerGeneration,
    };
  }
  /** CPU already owns this counter to stamp GPU publications; exposing it in
   * observational UI telemetry adds no simulation-sized work or readback. */
  get powerPublicationGeneration() { return this.powerGeneration; }
  get powerLeafHeaders() { return this.leafHeaders; }
  /** QA-only buffers for the cold-to-recurring sparse-topology acceptance gate. */
  get powerLeafFrontier() { return this.leafFrontier; }
  get topologyTileWorklist() { return this.topologyResidency.tileWorklist; }
  get powerOwnerArena() { return this.ownerPages?.arena ?? this.topology; }
  get adaptiveSolidFaceSource() { return this.solidFaces?.source; }
  readAdaptiveSolidFaceDiagnostics() { return this.solidFaces?.readDiagnostics(); }
  readAdaptiveFaceVelocityDiagnostics() { return this.faceTransport?.readDiagnostics(); }
  get adaptiveFaceVelocityAuthority() { return Boolean(this.faceMirror && this.faceRhsAuthority); }
  get adaptiveSurfacePageSource() { return this.adaptiveSurfacePages?.source; }
  /** Authoritative narrow-band fine phi for rendering and surface transport.
   * Topology sizing and pressure fractions still require the terminal coarse-phi cutover. */
  get globalFineLevelSetSource(): WebGPUFineLevelSetBrickSource | undefined {
    if (!this.globalFineLevelSet || !this.globalFineBootstrapped) return undefined;
    const fine = this.globalFineCurrentIsA ? this.globalFineSourceA : this.globalFineSourceB;
    if (!fine) return undefined;
    const coarse = this.powerCoarseLevelSetSchedule?.sampleSource;
    const topology = this.globalFineCurrentIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    return { ...fine,
      ...(coarse ? { coarsePhiDirectory: coarse.directory, coarsePhiHashCapacity: coarse.hashCapacity } : {}),
      ...(topology ? { topologyControl: topology.control } : {}),
      ...(this.globalFineSeeds ? { seedControl: this.globalFineSeeds.buffer } : {}),
    };
  }
  /** Diagnostic-only status for the transport most recently encoded. */
  get globalFineTransportControl(): GPUBuffer | undefined { return this.lastGlobalFineTransport?.control; }
  /** Diagnostic-only status for the redistance transaction that produced the current fine slot. */
  get globalFineRedistanceControl(): GPUBuffer | undefined {
    return this.globalFineCurrentIsA ? this.globalFineRedistanceA?.control : this.globalFineRedistanceB?.control;
  }
  /** Diagnostic-only shared total-volume transaction for both fine slots. */
  get globalFineVolumeControl(): GPUBuffer | undefined { return this.globalFineVolumeA?.control; }
  /** Diagnostic-only Stage-A reconstruction status used by fine transport. */
  get globalFinePowerVelocityControl(): GPUBuffer | undefined { return this.powerVelocity?.control; }
  /** Diagnostic-only Stage-B point-sampler status used by fine transport. */
  get globalFinePowerVelocitySampleControl(): GPUBuffer | undefined {
    return this.globalFineVelocityPrepass?.source.control;
  }
  /** Diagnostic-only compact coarse-phi transaction control. */
  get globalFineCoarseLevelSetControl(): GPUBuffer | undefined { return this.powerCoarseLevelSetSchedule?.control; }
  /** Diagnostic-only Section 5 face-band status; never participates in publication decisions on the CPU. */
  get globalFineFaceBandControl(): GPUBuffer | undefined { return this.globalFineFaceFastMarch?.control; }
  /** Diagnostic-only catalog-Delaunay transition gate; never participates in CPU authority selection. */
  get globalFineFaceBandTransitionControl(): GPUBuffer | undefined {
    return this.globalFineFaceFastMarch?.transitionControl;
  }
  get globalFineFaceBandPointFieldControl(): GPUBuffer | undefined {
    return this.globalFineFaceFastMarch?.pointFieldControl;
  }
  get globalFineFaceBandTransientPowerControl(): GPUBuffer | undefined {
    return this.globalFineFaceFastMarch?.transientPowerControl;
  }
  get globalFineFaceBandPowerPublicationControl(): GPUBuffer | undefined {
    return this.globalFineFaceFastMarch?.powerPublicationControl;
  }
  get globalFineFaceBandPlan() { return this.globalFineFaceFastMarch?.plan; }
  /** Diagnostic-only raw sparse summary header; topology consumes this GPU-side. */
  get globalFineSummaryDirectory(): GPUBuffer | undefined { return this.globalFineSummaries?.directory; }
  get adaptiveSurfaceAuthority() { return Boolean(this.adaptiveSurfaceAdapter && this.adaptiveSurfacePages); }
  readAdaptiveSurfacePageDiagnostics() { return this.adaptiveSurfacePages?.readDiagnostics(); }
  /** QA-only readback of the actual adapter-to-global-fine publication chain.
   * These counters are observational and never participate in simulation
   * scheduling or authority selection. */
  async readGlobalFineLevelSetDiagnostics() {
    const fine = this.globalFineCurrentIsA ? this.globalFineSourceA : this.globalFineSourceB;
    const topology = this.globalFineCurrentIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    const redistance = this.globalFineCurrentIsA ? this.globalFineRedistanceA : this.globalFineRedistanceB;
    if (!fine || !topology || !this.globalFineSeeds) return undefined;
    // Exact packed layout (bytes): existing chain [0, 560), point-field
    // control [560, 592), transient physical graph control [592, 656),
    // transition first-owner-mismatch payload [656, 720), the complete
    // 48-byte redistance control [720, 768), then the face-march heap
    // completion diagnostics [768, 800). Existing prefixes remain ABI-stable.
    const readback = this.device.createBuffer({ label: "Global fine QA diagnostics", size: 800,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read global fine QA diagnostics" });
    encoder.copyBufferToBuffer(this.globalFineSeeds.buffer, 0, readback, 0, 8);
    encoder.copyBufferToBuffer(topology.control, 0, readback, 8, 32);
    encoder.copyBufferToBuffer(fine.worklist, 0, readback, 40, 20);
    if (this.powerCoarseLevelSetSchedule) {
      encoder.copyBufferToBuffer(this.powerCoarseLevelSetSchedule.sampleSource.directory, 0, readback, 64, 32);
      encoder.copyBufferToBuffer(this.powerCoarseLevelSetSchedule.control, 0, readback, 96, 64);
    }
    if (this.fineToPowerCoarseLevelSet) {
      encoder.copyBufferToBuffer(this.fineToPowerCoarseLevelSet.control, 0, readback, 160, 32);
    }
    if (this.lastGlobalFineTransport) {
      encoder.copyBufferToBuffer(this.lastGlobalFineTransport.control, 0, readback, 192, 32);
    }
    if (redistance) encoder.copyBufferToBuffer(redistance.control, 0, readback, 224, 16);
    if (this.globalFineVolumeA) {
      encoder.copyBufferToBuffer(this.globalFineVolumeA.control, 0, readback, 240, 64);
    }
    if (this.globalFineFaceFastMarch) {
      encoder.copyBufferToBuffer(this.globalFineFaceFastMarch.control, 0, readback, 304, 64);
      encoder.copyBufferToBuffer(this.globalFineFaceFastMarch.control, 64, readback, 768, 32);
    }
    if (this.powerVelocity) encoder.copyBufferToBuffer(this.powerVelocity.control, 0, readback, 368, 32);
    if (this.globalFineVelocityPrepass) {
      encoder.copyBufferToBuffer(this.globalFineVelocityPrepass.source.control, 0, readback, 400, 32);
    }
    if (this.globalFineFaceFastMarch) {
      encoder.copyBufferToBuffer(this.globalFineFaceFastMarch.powerPublicationControl, 0, readback, 432, 64);
      encoder.copyBufferToBuffer(this.globalFineFaceFastMarch.transitionControl, 0, readback, 496, 64);
      encoder.copyBufferToBuffer(this.globalFineFaceFastMarch.pointFieldControl, 0, readback, 560, 32);
      encoder.copyBufferToBuffer(this.globalFineFaceFastMarch.transientPowerControl, 0, readback, 592, 64);
      encoder.copyBufferToBuffer(this.globalFineFaceFastMarch.transitionControl, 64, readback, 656, 64);
    }
    if (redistance) encoder.copyBufferToBuffer(redistance.control, 0, readback, 720, 48);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return {
        seedControl: Array.from(words.slice(0, 2)),
        topologyControl: Array.from(words.slice(2, 10)),
        worklistHeader: Array.from(words.slice(10, 15)),
        seedCount: words[0], seedError: words[1], topologyFlags: words[2],
        interfaceBricks: words[3], desiredBricks: words[4], activatedBricks: words[5],
        published: words[6] !== 0, rolledBack: words[7] !== 0,
        downstreamFinalizeReason: words[9], activeBricks: words[10], generation: words[11],
        configuredFineGeneration: fine.generation, fineGenerationSlot: fine.generationSlot,
        scheduledFineGeneration: this.globalFineGeneration, currentFineIsA: this.globalFineCurrentIsA,
        coarseDirectoryHeader: Array.from(words.slice(16, 24)),
        coarseControl: Array.from(words.slice(24, 40)),
        fineRestrictionControl: Array.from(words.slice(40, 48)),
        coarseDirectoryState: words[16], coarseDirectoryGeneration: words[17],
        coarseControlFlags: words[24], coarseControlGeneration: words[35], coarseControlValid: words[36],
        fineRestrictionCount: words[40], fineRestrictionMaximumPerRow: words[41],
        fineRestrictionFlags: words[42], fineRestrictionUnowned: words[43],
        fineRestrictionRows: words[44], fineRestrictionValid: words[45],
        transportControl: Array.from(words.slice(48, 56)),
        transportDepartureOutsideBand: words[48], transportNonfiniteVelocity: words[49],
        transportProcessed: words[50], transportCommitted: words[51] !== 0,
        transportExtrapolatedVelocity: words[52], transportMaximumDisplacementFineCells: words[53],
        transportFaceBandUnavailable: words[54], transportVelocityUnavailable: words[55],
        redistanceControl: Array.from(words.slice(56, 60)),
        redistanceControlDetailed: Array.from(words.slice(180, 192)),
        volumeControl: Array.from(words.slice(60, 76)),
        faceBandControl: Array.from(words.slice(76, 92)),
        faceBandMarchControl: Array.from(words.slice(192, 200)),
        powerVelocityControl: Array.from(words.slice(92, 100)),
        powerVelocitySampleControl: Array.from(words.slice(100, 108)),
        faceBandPowerPublicationControl: Array.from(words.slice(108, 124)),
        faceBandTransitionControl: Array.from(words.slice(124, 140)),
        faceBandTransitionOwnerFailure: Array.from(words.slice(164, 180)),
        faceBandPointFieldControl: Array.from(words.slice(140, 148)),
        faceBandTransientPowerControl: Array.from(words.slice(148, 164)),
        faceBandPointField: unpackOctreeFaceBandPointFieldControl(words.slice(140, 148)),
        faceBandTransientPower: unpackOctreeFaceBandTransientPowerControl(words.slice(148, 164)),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  async readPagedPhiDifferential() {
    if (!this.pagedPhiDifferential) return undefined;
    this.latestPagedPhiDifferential = await this.pagedPhiDifferential.read(this.device);
    return this.latestPagedPhiDifferential;
  }
  get sparseSurfaceRefinementFactor() { return this.sparseSurfaceBand?.plan.refinementFactor ?? 1; }
  get requiresFineSurfaceTimestep() { return this.sparseSurfaceBand?.requiresFineTimestep ?? false; }
  get fluidBrickCapacity() { return this.topologyResidency.capacity; }
  readFluidBrickResidencyStats() { return this.topologyResidency.readStats(); }
  readFluidBulkBrickResidencyStats() { return this.sparseBrickWorld?.readBulkResidencyStats(); }
  readFluidBrickAtlasStats() { return this.sparseBrickWorld?.readAtlasStats(); }
  readSparseSurfaceBandStats() { return this.sparseSurfaceBand?.readStats(); }
  encodeSparseBrickWorld(encoder: GPUCommandEncoder, timings: {
    residency?: GPUComputePassTimestampWrites;
    publication?: GPUComputePassTimestampWrites;
  } = {}, dt_s = 0, bulkAlreadyRefreshed = false) {
    if (this.surfacePagesBootstrapped && this.adaptiveSurfaceAdapter) {
      const source=this.adaptiveSurfaceAdapter.source;
      this.topologyResidency.encodeSurfaceCandidates(
        encoder, source.leaves, source.candidates.candidates, source.candidates.countAndDispatch,
      );
      this.sparseBrickWorld?.bulkResidency?.encodeSurfaceCandidates(
        encoder, source.leaves, source.candidates.candidates, source.candidates.countAndDispatch,
      );
      // Publication is GPU-transactional. Failed, stale, and overflowing
      // generations retain the last good (including analytic t=0) tile stream;
      // a published zero-count generation is the distinct valid-empty case.
      this.topologyWorklistReady = true;
      this.compactVoxelInspection?.encode(encoder);
      return;
    }
    if (!this.sparseBrickWorld) {
      this.topologyResidency.encode(encoder, this.levelSetTexture, this.resources.velocityOut, {
        dt_s, preActivation: true,
      });
      this.topologyWorklistReady = true;
      this.compactVoxelInspection?.encode(encoder);
      return;
    }
    this.sparseBrickWorld.encode(encoder, {
      levelSet: this.levelSetTexture,
      velocity: this.resources.velocityOut,
      solidCells: this.solidCells
    }, timings, dt_s, bulkAlreadyRefreshed);
    this.topologyWorklistReady = true;
  }

  destroy() {
    this.powerLifecycleDisposed = true;
    this.mgpcg?.destroy();
    this.firstOrderVCycle?.destroy();
    if (this.ownerPages) this.ownerPages.destroy(); else this.topology.destroy();
    this.pressureA.destroy(); this.pressureB.destroy(); this.params.destroy(); this.pagedParams?.destroy();
    this.topologyCandidateDispatch.destroy(); this.pressureOverflowDispatch.destroy();
    this.compaction.destroy(); this.leafHeaders.destroy(); this.leafEntries.destroy(); this.leafFrontier.destroy(); this.solveDispatch.destroy(); this.solidCells.destroy(); this.solveStats.destroy(); this.fineSummaryFallback.destroy();
    this.faceMirror?.destroy();
    this.faceTransport?.destroy();
    this.solidFaces?.destroy();
    this.compactVoxelInspection?.destroy();
    this.globalFineRedistanceA?.destroy(); this.globalFineRedistanceB?.destroy();
    this.analyticBootstrapWorklist?.destroy();
    this.globalFineVolumeA?.destroy(); this.globalFineVolumeB?.destroy();
    this.globalFineTransportA?.destroy(); this.globalFineTransportB?.destroy(); this.globalFineVelocityPrepass?.destroy();
    this.globalFineFaceFastMarch?.destroy();
    this.globalFineTopologyAB?.destroy(); this.globalFineTopologyBA?.destroy();
    this.globalFineSeeds?.destroy(); this.globalFineLevelSet?.destroy();
    this.globalFineSummaries?.destroy();
    this.adaptiveSurfacePages?.destroy();
    this.adaptiveSurfaceAdapter?.destroy();
    this.pagedPhiDifferential?.destroy();
    this.fineToPowerCoarseLevelSet?.destroy(); this.powerCoarseLevelSetSchedule?.destroy(); this.powerCoarseLevelSet?.destroy(); this.powerVelocity?.destroy(); this.powerSolidFaces?.destroy(); this.powerSolidVertices?.destroy(); this.powerFaceTransfer?.destroy(); this.powerFaceSeed?.destroy(); this.powerDescriptor?.destroy(); this.powerTopology?.destroy(); this.powerFaces?.destroy(); this.powerOperator?.destroy();
    this.powerVolumes?.destroy(); this.powerVolumeParams?.destroy();
    this.phiSnapshotTexture.destroy();
    this.levelSetFallbackTexture?.destroy();
    this.topologyDiagnosticTexture?.destroy(); this.pressureSamplesDiagnosticTexture?.destroy(); this.pressureDiagnosticTexture?.destroy(); this.divergenceDiagnosticTexture?.destroy();
    this.surfaceState.destroy();
    this.sparseSurfaceBand?.destroy();
    if (this.sparseBrickWorld) this.sparseBrickWorld.destroy(); else this.topologyResidency.destroy();
  }
}

export const octreePowerVolumeShader = /* wgsl */ `
struct Params { cellVolume:f32,rowCapacity:u32,pad0:u32,pad1:u32 }
struct PowerRowMetric { topologyCode:u32,transformAndFlags:u32,volume:f32,reserved:u32 }
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> metrics:array<PowerRowMetric>;
@group(0) @binding(2) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(3) var<storage,read> rowCountSource:array<u32>;
@group(0) @binding(4) var<storage,read_write> volumes:array<f32>;
@compute @workgroup_size(64) fn publishPowerVolumes(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;let count=min(select(0u,rowCountSource[0],arrayLength(&rowCountSource)>0u),params.rowCapacity);
  if(row>=count||row>=arrayLength(&metrics)||row>=arrayLength(&headers)||row>=arrayLength(&volumes)){return;}
  let size=f32(headers[row].size);let volume=metrics[row].volume*size*size*size*params.cellVolume;
  volumes[row]=select(0.0,volume,volume==volume&&volume>0.0&&abs(volume)<=3.402823e38);
}
`;

function initialOctreeLevelSet(
  scene: SceneDescription,
  dims: { nx: number; ny: number; nz: number },
  cell: { x: number; y: number; z: number }
) {
  const { nx, ny, nz } = dims;
  const alpha = new Float32Array(nx * ny * nz);
  const dam = damBreakFractions(scene.container.fillFraction);
  const terrainHeights = terrainColumnHeights(scene, nx, nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const aboveGround = (y + 0.5) * cell.y > terrainHeights[x + nx * z];
    const brickWet = initialFluidBrickContainsCell(scene, x, y, z, [nx, ny, nz]);
    const wet = aboveGround && combineInitialBrickWet(scene, brickWet, scene.fluid.initialCondition === "dam-break"
      ? (x + 0.5) / nx <= dam.width && (y + 0.5) / ny <= dam.height && (z + 0.5) / nz <= dam.depth
      : (y + 0.5) / ny <= scene.container.fillFraction);
    alpha[x + nx * (y + ny * z)] = wet ? 1 : 0;
  }
  return signedDistanceFromVolume(alpha, nx, ny, nz, cell);
}

export const octreeProjectionShader = /* wgsl */ `
override targetRefinementSize: u32 = 0u;
override rowIndexedPressure: bool = true;
override surfaceHashOffset: u32 = 0u;
override surfaceHashCapacity: u32 = 1u;
override surfaceAirHashOffset: u32 = 0u;
override surfaceAirHashCapacity: u32 = 1u;
override surfacePageTableOffset: u32 = 0u;
override surfacePhiOffset: u32 = 0u;
override surfacePageCapacity: u32 = 1u;
override surfaceLeafCapacity: u32 = 1u;
override surfacePageResolution: u32 = 2u;

struct Owner { packedOrigin: u32, size: u32 }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowDirectionLength: vec4f, physical: vec4f, pressureCapacity: vec4u, hydrostatic: vec4f }
struct LeafHeader { cell: u32, entryStart: u32, entryCount: u32, size: u32, diagonal: f32, rhs: f32, pad0: u32, pad1: u32, gradient: vec4f }
struct LeafEntry { row: u32, coefficient: f32 }
struct RigidBody { positionShape: vec4f, dimensions: vec4f, orientation: vec4f, linearVelocity: vec4f, angularVelocity: vec4f, inverseMassInertia: vec4f, angularMomentumRestitution: vec4f, material: vec4f }
struct SolidCell { fraction: f32, owner: i32 }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
// [0] = row count, [1] = entry count, [2..4] = row-parallel indirect args,
// [5..7] = cooperative coarse-assembly indirect args, [8] = coarse row count,
// [9..11] = one-workgroup-per-leaf args, [12..14] = frontier row-plan args;
// per-block (rows, entries, coarse rows) totals (later exclusive offsets) start
// at word 15. The dispatch words are copied out after their producing pass because one
// buffer cannot be writable storage and indirect in the same dispatch scope.
@group(0) @binding(2) var<storage, read_write> compaction: array<u32>;
@group(0) @binding(3) var<storage, read_write> owners: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> pressureIn: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var levelSetIn: texture_3d<f32>;
@group(0) @binding(8) var<storage, read_write> leafHeaders: array<LeafHeader>;
@group(0) @binding(9) var<storage, read_write> leafEntries: array<LeafEntry>;
@group(0) @binding(10) var<storage, read_write> rigidBodies: array<RigidBody, 12>;
// Dense-solid words on compatibility scenes; compact surface arena after the
// solid-free bootstrap. Both are writable-storage bindings with raw 32-bit ABI.
@group(0) @binding(11) var<storage, read_write> solidOrSurface: array<atomic<u32>>;
@group(0) @binding(12) var terrainIn: texture_2d<f32>;
// [0..1] ping-pong counts, [2] current-list selector, [3] generation,
// followed by capacity-bounded list A/list B and either an interleaved compact
// [cell+1,row+2] hash or the compatibility one-word-per-cell row map.
@group(0) @binding(13) var<storage, read_write> frontier: array<atomic<u32>>;
// Phi as of each tile's last rebuild; drives change-driven rebuild scheduling.
@group(0) @binding(14) var phiSnapshot: texture_storage_3d<r32float, read_write>;
// Dual ABI. Sparse-extrapolation groups bind the bulk-residency worklist;
// global-fine topology/pressure groups bind the corrected compact coarse-phi
// directory (8-word header followed by 8-word hash entries).
@group(0) @binding(15) var<storage, read> bulkWorklist: array<u32>;

fn dims() -> vec3u {
  let specializationDependency = surfaceHashOffset + surfaceHashCapacity + surfaceAirHashOffset + surfaceAirHashCapacity + surfacePageTableOffset
    + surfacePhiOffset + surfacePageCapacity + surfaceLeafCapacity + surfacePageResolution;
  return params.dimsMax.xyz + vec3u(specializationDependency & 0u);
}
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
struct CorrectedCoarsePhi { authority:bool, phi:f32, minimumPhi:f32, maximumPhi:f32 }
fn coarseWord(index:u32)->u32{return bulkWorklist[index];}
fn coarseFinite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn coarseDirectoryAuthority()->bool{
  let expected=params.pressureCapacity.w>>2u;
  if(expected==0u||arrayLength(&bulkWorklist)<16u||coarseWord(0u)!=0x80000000u
      ||(coarseWord(1u)&0x3fffffffu)!=expected){return false;}
  let directoryDims=vec3u(coarseWord(4u),coarseWord(5u),coarseWord(6u));
  let physicalCellSize=bitcast<f32>(coarseWord(7u));let capacity=coarseWord(2u);
  let actualCapacity=(arrayLength(&bulkWorklist)-8u)/8u;
  return all(directoryDims==dims())&&coarseFinite(physicalCellSize)&&physicalCellSize>0.0
    &&abs(physicalCellSize-params.cellRelax.x)<=1e-5*max(physicalCellSize,params.cellRelax.x)
    &&capacity==actualCapacity&&capacity>0u&&(capacity&(capacity-1u))==0u
    &&coarseWord(3u)>0u&&(coarseWord(3u)&(coarseWord(3u)-1u))==0u;
}
fn coarseHash(cell:u32,size:u32)->u32{var value=cell^(size*0x9e3779b9u);value=(value^(value>>16u))*0x7feb352du;value=(value^(value>>15u))*0x846ca68bu;return value^(value>>16u);}
fn coarseLookup(cell:u32,size:u32)->u32{let capacity=coarseWord(2u);let start=coarseHash(cell,size)&(capacity-1u);for(var probe=0u;probe<min(32u,capacity);probe+=1u){let slot=(start+probe)&(capacity-1u);let base=8u+slot*8u;let observed=coarseWord(base);if(observed==0u){return 0xffffffffu;}if(observed==cell+1u&&coarseWord(base+1u)==size){return base;}}return 0xffffffffu;}
fn correctedCoarsePhi(point:vec3f)->CorrectedCoarsePhi{
  if(!coarseDirectoryAuthority()||any(point<vec3f(0.0))||any(point>=vec3f(dims()))){return CorrectedCoarsePhi(false,0.0,0.0,0.0);}
  let q=vec3u(floor(point));var size=1u;let maximumLeaf=coarseWord(3u);
  loop{let origin=(q/vec3u(size))*vec3u(size);let cell=origin.x+dims().x*(origin.y+dims().y*origin.z);let base=coarseLookup(cell,size);
    if(base!=0xffffffffu){let value=bitcast<f32>(coarseWord(base+2u));let minimum=bitcast<f32>(coarseWord(base+3u));let maximum=bitcast<f32>(coarseWord(base+4u));let flags=coarseWord(base+5u);
      if((flags&9u)!=9u||!coarseFinite(value)||!coarseFinite(minimum)||!coarseFinite(maximum)||minimum>maximum||value<minimum||value>maximum){return CorrectedCoarsePhi(false,0.0,0.0,0.0);}
      return CorrectedCoarsePhi(true,value,minimum,maximum);}
    if(size>=maximumLeaf){break;}size*=2u;
  }
  // Publication is all-or-nothing: every live liquid/interface row inserts a
  // key before state becomes PUBLISHED. A miss in a valid directory is the
  // explicit positive-air complement, never an unknown sparse hole.
  let air=bitcast<f32>(coarseWord(7u))*f32(maximumLeaf);
  return CorrectedCoarsePhi(true,air,air,air);
}
fn coarseClassificationPhi(sample:CorrectedCoarsePhi)->f32{
  return select(sample.phi,min(sample.phi,sample.minimumPhi),sample.minimumPhi<0.0&&sample.maximumPhi>=0.0);
}
fn bulkResidentCell(workgroup: vec3u, localIndex: u32) -> vec3u {
  let dispatchWidth = bulkWorklist[12];
  if (dispatchWidth == 0u) { return vec3u(0xffffffffu); }
  let stream = (workgroup.x + workgroup.y * dispatchWidth) * 64u + localIndex;
  let brickSize = 8u;
  let brickVoxels = brickSize * brickSize * brickSize;
  let activeIndex = stream / brickVoxels;
  if (activeIndex >= bulkWorklist[0]) { return vec3u(0xffffffffu); }
  let entry = 16u + 2u * activeIndex;
  if (entry >= arrayLength(&bulkWorklist)) { return vec3u(0xffffffffu); }
  let brickDims = (dims() + vec3u(brickSize - 1u)) / brickSize;
  let brickCapacity = brickDims.x * brickDims.y * brickDims.z;
  let brickIndex = bulkWorklist[entry];
  if (brickIndex >= brickCapacity) { return vec3u(0xffffffffu); }
  let brick = vec3u(brickIndex % brickDims.x, (brickIndex / brickDims.x) % brickDims.y, brickIndex / (brickDims.x * brickDims.y));
  let localLinear = stream - activeIndex * brickVoxels;
  let local = vec3u(localLinear % brickSize, (localLinear / brickSize) % brickSize, localLinear / (brickSize * brickSize));
  return brick * brickSize + local;
}
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn packOrigin(p: vec3u) -> u32 { return index(p); }
fn unpackOrigin(word: u32) -> vec3u {
  let plane = params.dimsMax.x * params.dimsMax.y;
  return vec3u(word % params.dimsMax.x, (word / params.dimsMax.x) % params.dimsMax.y, word / plane);
}
const PACKED_FINE_OWNER: u32 = 0x80000000u;
fn encodeOwner(origin: vec3u, size: u32) -> u32 {
  if (size == 1u) { return PACKED_FINE_OWNER; }
  return u32(firstTrailingBit(size));
}
fn decodeOwner(word: u32, cell: vec3u) -> Owner {
  if ((word & PACKED_FINE_OWNER) != 0u) { return Owner(packOrigin(cell), 1u); }
  let exponent = word & 7u;
  if (exponent == 0u || exponent > 5u) { return Owner(packOrigin(cell), 1u); }
  let origin = (cell >> vec3u(exponent)) << vec3u(exponent);
  return Owner(packOrigin(origin), 1u << exponent);
}
fn canonicalOwner(cell: vec3u) -> Owner {
  // A nonresident owner page must never synthesize a leaf spanning into a
  // resident page: that creates two overlapping owners at the page seam.
  // Eight is the owner-page edge length; resident multi-page leaves still
  // retain their explicitly published larger owner words.
  var size = min(params.dimsMax.w, 8u);
  var origin = (cell / vec3u(size)) * vec3u(size);
  loop {
    if (all(origin + vec3u(size) <= dims()) || size == 1u) { break; }
    size >>= 1u; origin = (cell / vec3u(size)) * vec3u(size);
  }
  return Owner(packOrigin(origin), size);
}
fn ownerPagesEnabled() -> bool { return arrayLength(&owners) > 15u && atomicLoad(&owners[15]) == 0x4f574e52u; }
fn ownerPageEncoded(logical: u32) -> u32 {
  let freeListOffset = atomicLoad(&owners[5]);
  if (freeListOffset <= 16u || ((freeListOffset - 16u) & 1u) != 0u) { return 0u; }
  let hashCapacity = (freeListOffset - 16u) / 2u;
  let key = logical + 1u;
  var slot = (logical * 0x9e3779b1u) % hashCapacity;
  for (var probe = 0u; probe < hashCapacity; probe += 1u) {
    let observed = atomicLoad(&owners[16u + slot]);
    if (observed == key) { return atomicLoad(&owners[16u + hashCapacity + slot]); }
    if (observed == 0u) { break; }
    slot = select(slot + 1u, 0u, slot + 1u == hashCapacity);
  }
  return 0u;
}
fn popOwnerPage() -> u32 {
  var physical = 0xffffffffu;
  loop {
    let count = atomicLoad(&owners[0]); if (count == 0u) { break; }
    let claim = atomicCompareExchangeWeak(&owners[0], count, count - 1u);
    if (claim.exchanged) { physical = atomicLoad(&owners[atomicLoad(&owners[5]) + count - 1u]); break; }
  }
  return physical;
}
// Refinement/grading may discover a required neighbor just outside the
// residency publication. Allocate that 8^3 owner page on demand instead of
// silently dropping the split and leaving an immutable synthetic coarse leaf.
fn ensureOwnerPageEncoded(logical: u32) -> u32 {
  let freeListOffset = atomicLoad(&owners[5]);
  if (freeListOffset <= 16u || ((freeListOffset - 16u) & 1u) != 0u) { return 0u; }
  let hashCapacity = (freeListOffset - 16u) / 2u; let key = logical + 1u;
  var slot = (logical * 0x9e3779b1u) % hashCapacity;
  for (var probe = 0u; probe < hashCapacity; probe += 1u) {
    let keyWord = 16u + slot; let valueWord = 16u + hashCapacity + slot;
    let observed = atomicLoad(&owners[keyWord]);
    if (observed == key) {
      // Never spin behind a sibling lane that is clearing the page. The fixed
      // grading ladder retries this idempotent split in the next dispatch.
      let encoded = atomicLoad(&owners[valueWord]);
      return select(encoded, 0u, encoded == 0u || encoded == 0xffffffffu);
    }
    if (observed == 0u || observed == 0xffffffffu) {
      let claim = atomicCompareExchangeWeak(&owners[keyWord], observed, key);
      if (claim.exchanged) {
        atomicStore(&owners[valueWord], 0xffffffffu);
        let physical = popOwnerPage(); let capacity = atomicLoad(&owners[3]);
        if (physical == 0xffffffffu || physical >= capacity) {
          atomicStore(&owners[valueWord], 0u); atomicStore(&owners[keyWord], 0xffffffffu);
          atomicStore(&owners[2], 1u); return 0u;
        }
        let base = atomicLoad(&owners[6]) + physical * 512u;
        for (var local = 0u; local < 512u; local += 1u) { atomicStore(&owners[base + local], 0u); }
        atomicStore(&owners[valueWord], physical + 1u); atomicAdd(&owners[1], 1u); return physical + 1u;
      }
      if (claim.old_value == key) { return 0u; }
    }
    slot = select(slot + 1u, 0u, slot + 1u == hashCapacity);
  }
  atomicStore(&owners[2], 1u); return 0u;
}
fn ownerPageWord(cell: vec3u) -> u32 {
  let brickDims = (dims() + vec3u(7u)) / 8u;
  let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = ownerPageEncoded(logical);
  let capacity = atomicLoad(&owners[3]);
  if (encoded == 0u || encoded == 0xffffffffu || encoded > capacity) { return 0xffffffffu; }
  let local = cell % vec3u(8u);
  return atomicLoad(&owners[atomicLoad(&owners[6]) + (encoded - 1u) * 512u + local.x + local.y * 8u + local.z * 64u]);
}
fn ownerAt(p: vec3i) -> Owner {
  let cell = vec3u(p);
  if (!ownerPagesEnabled()) { return decodeOwner(atomicLoad(&owners[index(cell)]), cell); }
  let word = ownerPageWord(cell);
  if (word == 0xffffffffu || word == 0u) { return canonicalOwner(cell); }
  return decodeOwner(word, cell);
}
fn ownerAtIndex(cell: u32) -> Owner { return ownerAt(vec3i(cellCoord(cell))); }
fn storeOwner(cell: vec3u, origin: vec3u, size: u32) {
  if (!ownerPagesEnabled()) { atomicStore(&owners[index(cell)], encodeOwner(origin, size)); return; }
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = ownerPageEncoded(logical); let capacity = atomicLoad(&owners[3]);
  if (encoded == 0u || encoded == 0xffffffffu || encoded > capacity) { return; }
  let local = cell % vec3u(8u);
  atomicStore(&owners[atomicLoad(&owners[6]) + (encoded - 1u) * 512u + local.x + local.y * 8u + local.z * 64u], encodeOwner(origin, size));
}
fn storeOwnerEnsure(cell: vec3u, origin: vec3u, size: u32) {
  if (!ownerPagesEnabled()) { atomicStore(&owners[index(cell)], encodeOwner(origin, size)); return; }
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = ensureOwnerPageEncoded(logical); if (encoded == 0u) { return; }
  let local = cell % vec3u(8u);
  atomicStore(&owners[atomicLoad(&owners[6]) + (encoded - 1u) * 512u + local.x + local.y * 8u + local.z * 64u], encodeOwner(origin, size));
}
fn ensureLeafOwnerPages(origin: vec3u, size: u32, lane: u32, lanes: u32) {
  if (!ownerPagesEnabled()) { return; }
  let brickDims = (dims() + vec3u(7u)) / 8u; let first = origin / 8u; let last = (origin + vec3u(size - 1u)) / 8u;
  let shape = last - first + vec3u(1u); let count = shape.x * shape.y * shape.z;
  for (var item = lane; item < count; item += lanes) {
    let local = vec3u(item % shape.x, (item / shape.x) % shape.y, item / (shape.x * shape.y)); let brick = first + local;
    let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
    _ = ensureOwnerPageEncoded(logical);
  }
}
// The host-selected binding group only identifies the ABI of binding 11.  The
// sparse page arena becomes phi authority after the GPU has published a
// non-empty, fault-free generation.  Keeping those predicates separate lets a
// cold/failed page build fall back to the authored analytic initial condition
// without ever interpreting the page arena as the legacy solid-cell buffer.
fn pagedSurfaceBindings() -> bool { return (params.pressureCapacity.w & 2u) != 0u; }
fn pagedSurfaceAuthority() -> bool {
  return pagedSurfaceBindings() && arrayLength(&solidOrSurface) > 7u
    && atomicLoad(&solidOrSurface[3]) == 0u
    && atomicLoad(&solidOrSurface[6]) > 0u
    && atomicLoad(&solidOrSurface[7]) > 0u;
}
// Negative sentinels encode a bootstrap-only analytic initial condition while
// preserving the runtime surface-detail scalar: tank = -10-detail,
// dam-break = -20-detail. Imported/seeded shapes retain the non-negative
// compatibility value until their bounded sparse voxelizer is implemented.
fn analyticInitialPhiEnabled() -> bool { return params.physical.w < 0.0; }
fn analyticInitialDamBreak() -> bool { return params.physical.w < -15.0; }
fn surfaceDetailStrengthValue() -> f32 {
  return select(params.physical.w, select(-params.physical.w - 10.0, -params.physical.w - 20.0,
    analyticInitialDamBreak()), analyticInitialPhiEnabled());
}
fn analyticInitialPhi(point: vec3f) -> f32 {
  let fill = clamp(params.hydrostatic.w / f32(max(1u, dims().y)), 0.0, 1.0);
  let world = vec3f(-0.5 * params.container.x + point.x * params.cellRelax.x,
    point.y * params.cellRelax.y,
    -0.5 * params.container.z + point.z * params.cellRelax.z);
  if (!analyticInitialDamBreak()) { return world.y - fill * params.container.y; }
  let heightFraction = max(0.92, fill);
  let footprintFraction = sqrt(fill / max(heightFraction, 1e-9));
  let halfExtent = 0.5 * vec3f(footprintFraction * params.container.x,
    heightFraction * params.container.y, footprintFraction * params.container.z);
  let centre = vec3f(-0.5 * params.container.x + halfExtent.x, halfExtent.y,
    -0.5 * params.container.z + halfExtent.z);
  let q = abs(world - centre) - halfExtent;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
fn surfaceWord(row: u32, word: u32) -> u32 { return bulkWorklist[row * 12u + word]; }
fn surfaceLeafOrigin(row: u32) -> vec3u { return unpackOrigin(surfaceWord(row, 0u)); }
fn surfaceLeafSize(row: u32) -> u32 { return surfaceWord(row, 1u); }
fn surfaceHashCoord(q: vec3u) -> u32 { var h=(q.x*73856093u)^(q.y*19349663u)^(q.z*83492791u);h^=h>>16u;return h; }
fn surfaceArenaWord(word: u32) -> u32 { return atomicLoad(&solidOrSurface[word]); }
fn findSurfaceLeaf(p: vec3u) -> u32 {
  let mask = surfaceHashCapacity - 1u;
  for (var size = 1u; size <= 32u; size <<= 1u) {
    var slot = surfaceHashCoord(p / size) & mask;
    for (var probe = 0u; probe < 32u; probe += 1u) {
      let encoded = surfaceArenaWord(surfaceHashOffset + slot);
      if (encoded == 0u) { break; }
      let row = encoded - 1u;
      if (row < surfaceLeafCapacity) {
        let origin = surfaceLeafOrigin(row); let leafSize = surfaceLeafSize(row);
        if (leafSize == size && all(p >= origin) && all(p < origin + vec3u(size))) { return row; }
      }
      slot = (slot + 1u) & mask;
    }
  }
  return 0xffffffffu;
}
fn findAirAlias(p:vec3u)->u32{let mask=surfaceAirHashCapacity-1u;var slot=surfaceHashCoord(p)&mask;let key=packOrigin(p)+1u;for(var probe=0u;probe<32u;probe+=1u){let at=surfaceAirHashOffset+2u*slot;let stored=surfaceArenaWord(at);if(stored==0u){break;}if(stored==key){let encoded=surfaceArenaWord(at+1u);if(encoded!=0u){let row=0xffffffffu-encoded;if(row<surfaceLeafCapacity){return row;}}}slot=(slot+1u)&mask;}return 0xffffffffu;}
fn surfaceLeafFallback(row: u32, point: vec3f) -> f32 {
  let origin = vec3f(surfaceLeafOrigin(row)); let size = f32(surfaceLeafSize(row));
  let centre = origin + vec3f(0.5 * size);
  let base = bitcast<f32>(surfaceWord(row, 4u));
  let gradient = vec3f(bitcast<f32>(surfaceWord(row, 5u)), bitcast<f32>(surfaceWord(row, 6u)), bitcast<f32>(surfaceWord(row, 7u)));
  let physicalGradient=gradient/max(params.cellRelax.xyz,vec3f(1e-9));
  let boundedGradient=physicalGradient/max(1.0,length(physicalGradient))*params.cellRelax.xyz;
  return base + dot(boundedGradient, point - centre);
}
fn surfacePagePhi(row: u32, point: vec3f) -> f32 {
  let page = surfaceArenaWord(surfacePageTableOffset + row);
  if (page == 0xffffffffu || page >= surfacePageCapacity) { return surfaceLeafFallback(row, point); }
  let origin = vec3f(surfaceLeafOrigin(row)); let size = f32(surfaceLeafSize(row));
  let grid = clamp((point-origin)/size*f32(surfacePageResolution)-vec3f(0.5),vec3f(0.0),vec3f(f32(surfacePageResolution-1u)));
  let a=vec3u(floor(grid));let b=min(a+vec3u(1u),vec3u(surfacePageResolution-1u));let t=fract(grid);
  let samples=surfacePageResolution*surfacePageResolution*surfacePageResolution;let base=surfacePhiOffset+page*samples;
  let i000=a.x+surfacePageResolution*(a.y+surfacePageResolution*a.z);let i100=b.x+surfacePageResolution*(a.y+surfacePageResolution*a.z);let i010=a.x+surfacePageResolution*(b.y+surfacePageResolution*a.z);let i110=b.x+surfacePageResolution*(b.y+surfacePageResolution*a.z);
  let i001=a.x+surfacePageResolution*(a.y+surfacePageResolution*b.z);let i101=b.x+surfacePageResolution*(a.y+surfacePageResolution*b.z);let i011=a.x+surfacePageResolution*(b.y+surfacePageResolution*b.z);let i111=b.x+surfacePageResolution*(b.y+surfacePageResolution*b.z);
  let c000=bitcast<f32>(surfaceArenaWord(base+i000));let c100=bitcast<f32>(surfaceArenaWord(base+i100));let c010=bitcast<f32>(surfaceArenaWord(base+i010));let c110=bitcast<f32>(surfaceArenaWord(base+i110));
  let c001=bitcast<f32>(surfaceArenaWord(base+i001));let c101=bitcast<f32>(surfaceArenaWord(base+i101));let c011=bitcast<f32>(surfaceArenaWord(base+i011));let c111=bitcast<f32>(surfaceArenaWord(base+i111));
  return mix(mix(mix(c000,c100,t.x),mix(c010,c110,t.x),t.y),mix(mix(c001,c101,t.x),mix(c011,c111,t.x),t.y),t.z);
}
fn surfaceIncidentPhi(row:u32,point:vec3f)->f32{
  let origin=vec3f(surfaceLeafOrigin(row));let size=f32(surfaceLeafSize(row));
  // Air-halo entries retain only their incident row. A page is authoritative
  // inside that row; outside it, use the incident affine field rather than a
  // clamped page-edge value.
  if(all(point>=origin)&&all(point<origin+vec3f(size))){return surfacePagePhi(row,point);}
  return surfaceLeafFallback(row,point);
}
fn legacyPhi(p:vec3i)->f32{
  if (!valid(p)) { return 3.402823e38; }
  if (analyticInitialPhiEnabled() &&
      (frontierGeneration() == 0u || !pagedSurfaceBindings() || !pagedSurfaceAuthority())) {
    return analyticInitialPhi(vec3f(p) + vec3f(0.5));
  }
  if (!pagedSurfaceAuthority()) { return textureLoad(levelSetIn, p, 0).x; }
  let row = findSurfaceLeaf(vec3u(p));
  if (row == 0xffffffffu) { return pagedPhiMiss(vec3u(p)); }
  return surfacePagePhi(row, vec3f(p)+vec3f(0.5));
}
fn phi(p: vec3i) -> f32 {
  if (!valid(p)) { return 3.402823e38; }
  let coarse=correctedCoarsePhi(vec3f(p)+vec3f(0.5));
  if(coarse.authority){return coarseClassificationPhi(coarse);}
  return legacyPhi(p);
}
fn liquidCell(p: vec3i) -> bool { return valid(p) && phi(p) < 0.0; }
fn samplePhiPoint(point:vec3f)->f32{
  let bounded=clamp(point,vec3f(0.0),vec3f(dims()-vec3u(1u)));let a=vec3u(floor(bounded));let b=min(a+vec3u(1u),dims()-vec3u(1u));let t=fract(bounded);
  let p000=phi(vec3i(a));let p100=phi(vec3i(vec3u(b.x,a.y,a.z)));let p010=phi(vec3i(vec3u(a.x,b.y,a.z)));let p110=phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001=phi(vec3i(vec3u(a.x,a.y,b.z)));let p101=phi(vec3i(vec3u(b.x,a.y,b.z)));let p011=phi(vec3i(vec3u(a.x,b.y,b.z)));let p111=phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y),mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y),t.z);}
fn ownerPhi(owner: Owner) -> f32 {
  // Pressure lives at the geometric leaf centre. Even-sized leaves therefore
  // sit between fine-cell samples; trilinear reconstruction avoids the
  // upper-corner classification bias of origin + size/2.
  let centre = vec3f(unpackOrigin(owner.packedOrigin)) + vec3f(0.5 * f32(owner.size - 1u));
  let coarse=correctedCoarsePhi(centre+vec3f(0.5));
  if(coarse.authority){return coarseClassificationPhi(coarse);}
  return samplePhiPoint(centre);
}
fn ownerPhiGradient(owner:Owner)->vec3f{let c=vec3f(unpackOrigin(owner.packedOrigin))+vec3f(0.5*f32(owner.size-1u));let step=max(0.5,0.5*f32(owner.size));
  return vec3f(samplePhiPoint(c+vec3f(step,0,0))-samplePhiPoint(c-vec3f(step,0,0)),samplePhiPoint(c+vec3f(0,step,0))-samplePhiPoint(c-vec3f(0,step,0)),samplePhiPoint(c+vec3f(0,0,step))-samplePhiPoint(c-vec3f(0,0,step)))/(2.0*step);}
fn liquidOwner(owner: Owner) -> bool {
  let centre=vec3f(unpackOrigin(owner.packedOrigin))+vec3f(0.5*f32(owner.size));
  let coarse=correctedCoarsePhi(centre);
  if(coarse.authority){return coarse.minimumPhi<0.0;}
  return ownerPhi(owner) < 0.0;
}
fn isOrigin(id: vec3u, owner: Owner) -> bool { return all(id == unpackOrigin(owner.packedOrigin)); }
fn cellCount() -> u32 { return params.dimsMax.x * params.dimsMax.y * params.dimsMax.z; }
fn frontierListCapacity() -> u32 { return select(cellCount(), params.pressureCapacity.x, rowIndexedPressure); }
fn frontierBase(which: u32) -> u32 { return 4u + which * frontierListCapacity(); }
fn frontierMapBase() -> u32 { return 4u + 2u * frontierListCapacity(); }
fn frontierHashCapacity() -> u32 { return select(0u, (arrayLength(&frontier)-frontierMapBase())/2u, rowIndexedPressure); }
fn frontierHash(cell:u32)->u32{var h=cell*747796405u+2891336453u;h^=h>>16u;h*=2246822519u;h^=h>>13u;return h;}
fn frontierCurrent() -> u32 { return atomicLoad(&frontier[2]); }
fn frontierGeneration() -> u32 { return atomicLoad(&frontier[3]); }
fn frontierCount(which: u32) -> u32 { return min(atomicLoad(&frontier[which]), frontierListCapacity()); }
fn frontierCell(which: u32, slot: u32) -> u32 { return atomicLoad(&frontier[frontierBase(which) + slot]); }
fn frontierHashSlot(cell:u32)->u32{
  let cap=frontierHashCapacity();if(cap==0u){return 0xffffffffu;}var slot=frontierHash(cell)&(cap-1u);let key=cell+1u;
  for(var probe=0u;probe<32u;probe+=1u){let stored=atomicLoad(&frontier[frontierMapBase()+2u*slot]);if(stored==0u){break;}if(stored==key){return slot;}slot=(slot+1u)&(cap-1u);}return 0xffffffffu;
}
fn frontierRowWord(cell: u32) -> u32 {
  if(!rowIndexedPressure){return atomicLoad(&frontier[frontierMapBase()+cell]);}
  let slot=frontierHashSlot(cell);if(slot==0xffffffffu){return 0u;}return atomicLoad(&frontier[frontierMapBase()+2u*slot+1u]);
}
fn frontierAlive(cell: u32) -> bool { return frontierRowWord(cell) != 0u; }
fn frontierRow(cell: u32) -> u32 {
  let word = frontierRowWord(cell);
  return select(0xffffffffu, word - 2u, word >= 2u);
}
// Sparse narrow-band level sets require a signed background on both sides of
// the active band. The page hash contains pressure (liquid) rows, so an air
// cell is normally absent. Preserve deep-air sign with a finite positive
// background and extrapolate bounded ghost samples from the nearest incident
// liquid leaf. The previous owner/frontier map supplies deep-liquid sign even
// if a bounded hash probe misses a highly clustered row.
fn pagedPhiMiss(p: vec3u) -> f32 {
  let point=vec3f(p)+vec3f(0.5);
  let band=max(1u,min(32u,u32(ceil(max(1.0,params.solve.w)))));
  let h=max(params.cellRelax.x,max(params.cellRelax.y,params.cellRelax.z));
  let background=f32(band+1u)*h;
  let owner=ownerAt(vec3i(p));
  let ownRow=frontierRow(index(unpackOrigin(owner.packedOrigin)));
  if(ownRow!=0xffffffffu&&ownRow<surfaceLeafCapacity){return clamp(surfacePagePhi(ownRow,point),-background,-0.01*h);}
  let airRow=findAirAlias(p);if(airRow!=0xffffffffu){return clamp(surfaceIncidentPhi(airRow,point),0.01*h,background);}
  return background;
}
fn pressureIndex(owner: Owner) -> u32 {
  let cell = index(unpackOrigin(owner.packedOrigin));
  return select(cell, frontierRow(cell), rowIndexedPressure);
}
fn frontierInvalidate(cell: u32) {
  if(!rowIndexedPressure){atomicStore(&frontier[frontierMapBase()+cell],0u);return;}
  let slot=frontierHashSlot(cell);if(slot!=0xffffffffu){atomicStore(&frontier[frontierMapBase()+2u*slot+1u],0u);atomicStore(&frontier[frontierMapBase()+2u*slot],0xffffffffu);}
}
fn frontierClaimValue(at:u32)->bool{
  for(var attempt=0u;attempt<32u;attempt+=1u){let claim=atomicCompareExchangeWeak(&frontier[at+1u],0u,1u);if(claim.exchanged){return true;}if(claim.old_value!=0u){return false;}}
  return false;
}
fn frontierClaim(cell:u32)->bool{
  if(!rowIndexedPressure){return atomicCompareExchangeWeak(&frontier[frontierMapBase()+cell],0u,1u).exchanged;}
  let cap=frontierHashCapacity();let base=frontierHash(cell)&(cap-1u);let key=cell+1u;
  // A tombstone cannot be reused until the rest of its probe cluster has been
  // searched for this key. Otherwise topology churn can insert the same cell
  // before its still-live later slot and publish duplicate compact rows.
  for(var restart=0u;restart<8u;restart+=1u){
    var firstTombstone=0xffffffffu;var emptySlot=0xffffffffu;var slot=base;
    for(var probe=0u;probe<32u;probe+=1u){let at=frontierMapBase()+2u*slot;let stored=atomicLoad(&frontier[at]);
      if(stored==key){return frontierClaimValue(at);}
      if(stored==0xffffffffu&&firstTombstone==0xffffffffu){firstTombstone=slot;}
      if(stored==0u){emptySlot=slot;break;}
      slot=(slot+1u)&(cap-1u);
    }
    let chosenSlot=select(emptySlot,firstTombstone,firstTombstone!=0xffffffffu);
    if(chosenSlot==0xffffffffu){break;}
    let expected=select(0u,0xffffffffu,firstTombstone!=0xffffffffu);
    let at=frontierMapBase()+2u*chosenSlot;
    for(var attempt=0u;attempt<32u;attempt+=1u){let claim=atomicCompareExchangeWeak(&frontier[at],expected,key);
      if(claim.exchanged||claim.old_value==key){return frontierClaimValue(at);}
      if(claim.old_value!=expected){break;}
    }
  }
  let control=pressureControlBase();compaction[control]=3u;compaction[control+1u]=atomicLoad(&frontier[1u-frontierCurrent()]);return false;
}
fn frontierSetRow(cell:u32,word:u32){
  if(!rowIndexedPressure){atomicStore(&frontier[frontierMapBase()+cell],word);return;}
  let slot=frontierHashSlot(cell);if(slot!=0xffffffffu){atomicStore(&frontier[frontierMapBase()+2u*slot+1u],word);}else{compaction[pressureControlBase()]=3u;}
}
fn frontierAppend(which: u32, cell: u32) {
  if (!frontierClaim(cell)) { return; }
  let slot = atomicAdd(&frontier[which], 1u);
  if (slot < frontierListCapacity()) {
    atomicStore(&frontier[frontierBase(which) + slot], cell);
  } else {
    // Keep an unlisted origin eligible for a future rebuild after a transient
    // overflow; the current solve is failed closed by finalizeFrontier.
    frontierInvalidate(cell);
  }
}
// The trailing eight words are isolated from topology-change state and scan
// partials: overflow, required rows, required entries, fallback dispatch xyz,
// then residual sums rr/bb.
fn pressureControlBase() -> u32 { return arrayLength(&compaction) - 8u; }
fn pressureOverflowed() -> bool {
  return compaction[pressureControlBase()] != 0u || (ownerPagesEnabled() && atomicLoad(&owners[2]) != 0u);
}
fn component(v: vec3f, axis: u32) -> f32 { return select(select(v.z, v.y, axis == 1u), v.x, axis == 0u); }
fn axisVector(axis: u32) -> vec3i { return select(select(vec3i(0,0,1), vec3i(0,1,0), axis == 1u), vec3i(1,0,0), axis == 0u); }
fn faceArea(axis: u32) -> f32 {
  let h = params.cellRelax.xyz;
  return select(select(h.x * h.y, h.x * h.z, axis == 1u), h.y * h.z, axis == 0u);
}
fn cellWidth(axis: u32) -> f32 { return component(params.cellRelax.xyz, axis); }
fn pressureDistance(a: Owner, b: Owner, axis: u32) -> f32 {
  return pressureDistanceFromPhi(a, b, axis, ownerPhi(a), ownerPhi(b));
}
fn pressureDistanceFromPhi(a: Owner, b: Owner, axis: u32, phiA: f32, phiB: f32) -> f32 {
  let full = 0.5 * f32(a.size + b.size) * cellWidth(axis);
  if ((phiA < 0.0) == (phiB < 0.0)) { return full; }
  // Ghost-fluid/Ando--Batty distance: p=0 lies at the zero crossing of the
  // resident level set, not at the neighbouring air leaf centre. The lower
  // bound is a geometric degeneracy guard equivalent to the quadtree's
  // bounded free-surface weight, not a pressure tuning coefficient.
  let liquidPhi = select(phiB, phiA, phiA < 0.0);
  let airPhi = select(phiA, phiB, phiA < 0.0);
  let theta = clamp(abs(liquidPhi) / max(abs(liquidPhi) + abs(airPhi), 1e-12), 0.01, 1.0);
  return theta * full;
}
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
fn solidAt(p: vec3i) -> SolidCell {
  if (!valid(p)) { return SolidCell(1.0, -1); }
  if (pagedSurfaceBindings()) { return SolidCell(0.0, -1); }
  let i = index(vec3u(p));
  let word = 2u * i;
  if (word + 1u >= arrayLength(&solidOrSurface)) { return SolidCell(0.0, -1); }
  return SolidCell(bitcast<f32>(atomicLoad(&solidOrSurface[word])), bitcast<i32>(atomicLoad(&solidOrSurface[word + 1u])));
}
fn faceSolid(a: vec3i, b: vec3i) -> SolidCell { let sa = solidAt(a); let sb = solidAt(b); if (sa.fraction >= sb.fraction) { return sa; } return sb; }
fn faceWorld(faceCell: vec3i, axis: u32) -> vec3f { var result = worldCell(faceCell); result[axis] += 0.5 * params.cellRelax[axis]; return result; }
fn solidVelocity(solid: SolidCell, world: vec3f) -> vec3f {
  if (solid.owner < 0) { return vec3f(0.0); }
  let body = rigidBodies[u32(solid.owner)];
  return body.linearVelocity.xyz + cross(body.angularVelocity.xyz, world - body.positionShape.xyz);
}
fn couplingBase() -> u32 { return 15u + 3u * params.control.z; }
fn coarseTaskListBase() -> u32 { return couplingBase() + 12u * 8u; }
fn coarseTaskCapacity() -> u32 {
  return params.pressureCapacity.z;
}
fn coarseTaskRow(task: u32) -> u32 { return compaction[coarseTaskListBase() + 2u * task]; }
fn coarseTaskTile(task: u32) -> u32 { return compaction[coarseTaskListBase() + 2u * task + 1u]; }
fn coarseTaskIndex(workgroup: vec3u) -> u32 { return workgroup.x + workgroup.y * compaction[5]; }
fn couplingAcceleration(body: u32, component: u32) -> f32 { return bitcast<f32>(compaction[couplingBase() + body * 8u + component]); }
fn constrainedFaceVelocity(faceCell: vec3i, axis: u32, solid: SolidCell) -> f32 {
  let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
  return open * component(velocityAt(faceCell), axis) + solid.fraction * component(solidVelocity(solid, faceWorld(faceCell, axis)), axis);
}

// Topology-tile worklist header occupies words 0..15 of the copied buffer:
// word 0 the active tile count, word 1 the active dispatch x width, word 4
// and word 5 the retired equivalents. A tile spans max(8, maximumLeaf) cells
// per axis so every dyadic pressure leaf lies inside exactly one tile; each
// tile decomposes into (tileSize/4)^3 of the existing 4^3 cell workgroups.
fn topologyTileSize() -> u32 { return max(8u, params.dimsMax.w); }
fn topologyTileCell(workgroup: vec3u, local: vec3u, countWord: u32, widthWord: u32, indexBase: u32) -> vec3u {
  let tileSize = topologyTileSize();
  let blocks = tileSize / 4u;
  let groupsPerTile = blocks * blocks * blocks;
  let linearWorkgroup = workgroup.x + workgroup.y * compaction[widthWord];
  let streamIndex = linearWorkgroup / groupsPerTile;
  // The 2D-tiled indirect dispatch may round up; out-of-list workgroups map
  // to an out-of-domain cell that every kernel's bounds guard rejects.
  if (streamIndex >= compaction[countWord]) { return vec3u(0xffffffffu); }
  let sub = linearWorkgroup % groupsPerTile;
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tileIndex = compaction[indexBase + streamIndex];
  let tile = vec3u(tileIndex % tx, (tileIndex / tx) % ty, tileIndex / (tx * ty));
  let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
  return tile * tileSize + subCoord * 4u + local;
}

fn residentTopologyCell(workgroup: vec3u, local: vec3u) -> vec3u {
  return topologyTileCell(workgroup, local, 0u, 1u, 16u);
}

// Retired tiles hold just-retired bricks with no resident sibling. Resetting
// and rebuilding them prevents old interface leaves from fossilizing outside
// residency; their indices follow the active capacity in the copied list.
fn retiredTopologyCell(workgroup: vec3u, local: vec3u) -> vec3u {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return topologyTileCell(workgroup, local, 4u, 5u, 16u + tx * ty * tz);
}

// Refinement and balancing can only act on leaves of size >= 2. Their origins
// are even-aligned, so candidate passes cover an 8^3 cell region with each
// 4^3 workgroup instead of launching one invocation for every finest cell.
fn topologyCandidateCell(workgroup: vec3u, local: vec3u, countWord: u32, widthWord: u32, indexBase: u32) -> vec3u {
  let tileSize = topologyTileSize();
  let blocks = max(1u, tileSize / 8u);
  let groupsPerTile = blocks * blocks * blocks;
  let linearWorkgroup = workgroup.x + workgroup.y * compaction[widthWord];
  let streamIndex = linearWorkgroup / groupsPerTile;
  if (streamIndex >= compaction[countWord]) { return vec3u(0xffffffffu); }
  let sub = linearWorkgroup % groupsPerTile;
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tileIndex = compaction[indexBase + streamIndex];
  let tile = vec3u(tileIndex % tx, (tileIndex / tx) % ty, tileIndex / (tx * ty));
  let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
  return tile * tileSize + subCoord * 8u + local * 2u;
}

fn residentTopologyCandidate(workgroup: vec3u, local: vec3u) -> vec3u {
  return topologyCandidateCell(workgroup, local, 0u, 8u, 16u);
}

fn retiredTopologyCandidate(workgroup: vec3u, local: vec3u) -> vec3u {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return topologyCandidateCell(workgroup, local, 4u, 12u, 16u + tx * ty * tz);
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

// ---- Change-driven topology rebuild -----------------------------------------
// A calm ocean's tiles stay resident (the surface band never retires), yet
// their refinement decisions cannot change while phi is static. Per-tile
// change flags live past the copied worklist (words 16 + 2*capacity ...),
// stamped with the worklist generation so stale flags from formerly active
// tiles never read as fresh. The dirty pass then filters the active tile list
// in place so reset/refine/balance and the frontier append touch only tiles
// whose phi moved (or whose neighbor's did — grading travels under one tile).

fn topologyTileCapacity() -> u32 {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return tx * ty * tz;
}
// The change-tracking regions live in an exclusive tail of the compaction
// buffer (sized additively by the constructor), just ahead of the eight-word
// pressure control/residual tail. Placing them anywhere inside the general
// header/partials span is unsafe: the per-solve plan/scan partials sweep
// words 15+3*linearBlocks and would clobber flags between frames.
fn changeStateBase() -> u32 { return arrayLength(&compaction) - 8u - 3u * topologyTileCapacity(); }
fn tileChangeFlagsBase() -> u32 { return changeStateBase(); }
fn dirtyMarksBase() -> u32 { return changeStateBase() + topologyTileCapacity(); }
fn dirtyListBase() -> u32 { return changeStateBase() + 2u * topologyTileCapacity(); }

fn residencyTiledDispatch(blocks: u32) -> vec2u {
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  return vec2u(x, y);
}

var<workgroup> changeReduce: array<f32, 256>;
var<workgroup> changeReduceMin: array<f32, 256>;

@compute @workgroup_size(256)
fn markChangedTiles(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3u) {
  let tileIndex = compaction[16u + wid.x];
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let origin = vec3u(tileIndex % tx, (tileIndex / tx) % ty, tileIndex / (tx * ty)) * tileSize;
  var maxDelta = 0.0;
  var minAbsPhi = 3.402823e38;
  let cells = tileSize * tileSize * tileSize;
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % tileSize, (flat / tileSize) % tileSize, flat / (tileSize * tileSize));
    let q = origin + local;
    if (any(q >= dims())) { continue; }
    let sample = phi(vec3i(q));
    maxDelta = max(maxDelta, abs(sample - textureLoad(phiSnapshot, vec3i(q)).x));
    minAbsPhi = min(minAbsPhi, abs(sample));
  }
  changeReduce[lid] = maxDelta;
  changeReduceMin[lid] = minAbsPhi;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) {
      changeReduce[lid] = max(changeReduce[lid], changeReduce[lid + stride]);
      changeReduceMin[lid] = min(changeReduceMin[lid], changeReduceMin[lid + stride]);
    }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let h = min(params.cellRelax.x, min(params.cellRelax.y, params.cellRelax.z));
    // Two tiers. Away from the interface, 0.1h of accumulated drift rebuilds
    // well before any band or grading decision can flip (their scales are
    // >= 1h). A tile holding interface cells must react to ANY real phi
    // motion: leaf wetness follows the interpolated leaf-centre sign, which
    // can flip on a sub-0.1h creep, and a missed flip leaves a wet leaf with
    // no pressure row. The 2h margin bounds every cell that interpolation can
    // consult; a bit-static calm tile (delta exactly 0) still never rebuilds.
    let changed = changeReduce[0] > 0.1 * h
      || (changeReduceMin[0] < 2.0 * h && changeReduce[0] > 0.001 * h);
    // Stamp with the frontier rebuild counter: it advances every rebuild, so
    // flags from earlier frames (or formerly active tiles) never read fresh.
    // The copied tile-worklist header has NO generation word — its word 15 is
    // permanently zero, which is why it cannot serve as the stamp.
    let generation = atomicLoad(&frontier[3]) + 1u;
    compaction[tileChangeFlagsBase() + tileIndex] = select(0u, generation, changed);
  }
}

var<workgroup> dirtyScan: array<u32, 256>;

@compute @workgroup_size(256)
fn buildDirtyWorklist(@builtin(local_invocation_index) lid: u32) {
  let activeCount = compaction[0];
  let generation = atomicLoad(&frontier[3]) + 1u;
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  let chunk = (activeCount + 255u) / 256u;
  let base = lid * chunk;
  var count = 0u;
  for (var i = 0u; i < chunk; i += 1u) {
    let slot = base + i;
    if (slot >= activeCount) { break; }
    let tileIndex = compaction[16u + slot];
    let tile = vec3i(i32(tileIndex % tx), i32((tileIndex / tx) % ty), i32(tileIndex / (tx * ty)));
    var dirty = false;
    for (var dz = -1; dz <= 1; dz += 1) { for (var dy = -1; dy <= 1; dy += 1) { for (var dx = -1; dx <= 1; dx += 1) {
      let n = tile + vec3i(dx, dy, dz);
      if (any(n < vec3i(0)) || n.x >= i32(tx) || n.y >= i32(ty) || n.z >= i32(tz)) { continue; }
      let nIndex = u32(n.x) + tx * (u32(n.y) + ty * u32(n.z));
      if (generation != 0u && compaction[tileChangeFlagsBase() + nIndex] == generation) { dirty = true; }
    } } }
    compaction[dirtyMarksBase() + slot] = select(0u, 1u, dirty);
    count += select(0u, 1u, dirty);
  }
  dirtyScan[lid] = count;
  for (var stride = 1u; stride < 256u; stride <<= 1u) {
    workgroupBarrier();
    var add = 0u;
    if (lid >= stride) { add = dirtyScan[lid - stride]; }
    workgroupBarrier();
    dirtyScan[lid] += add;
  }
  workgroupBarrier();
  var offset = dirtyScan[lid] - count;
  for (var i = 0u; i < chunk; i += 1u) {
    let slot = base + i;
    if (slot >= activeCount) { break; }
    if (compaction[dirtyMarksBase() + slot] == 1u) {
      compaction[dirtyListBase() + offset] = compaction[16u + slot];
      offset += 1u;
    }
  }
  storageBarrier();
  workgroupBarrier();
  let total = dirtyScan[255];
  for (var i = lid; i < total; i += 256u) {
    compaction[16u + i] = compaction[dirtyListBase() + i];
  }
  if (lid == 0u) {
    compaction[0] = total;
    let blocks = tileSize / 4u;
    let tileDispatch = residencyTiledDispatch(total * blocks * blocks * blocks);
    compaction[1] = tileDispatch.x; compaction[2] = tileDispatch.y; compaction[3] = 1u;
    let candidateBlocks = max(1u, tileSize / 8u);
    let candidateDispatch = residencyTiledDispatch(total * candidateBlocks * candidateBlocks * candidateBlocks);
    compaction[8] = candidateDispatch.x; compaction[9] = candidateDispatch.y; compaction[10] = 1u;
  }
}

@compute @workgroup_size(256)
fn refreshSnapshotTiles(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3u) {
  let origin = worklistTileOrigin(wid.x, 16u);
  let tileSize = topologyTileSize();
  let cells = tileSize * tileSize * tileSize;
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % tileSize, (flat / tileSize) % tileSize, flat / (tileSize * tileSize));
    let q = origin + local;
    if (any(q >= dims())) { continue; }
    textureStore(phiSnapshot, vec3i(q), vec4f(phi(vec3i(q)), 0.0, 0.0, 0.0));
  }
}

@compute @workgroup_size(4,4,4)
fn refreshSnapshotDense(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  textureStore(phiSnapshot, vec3i(gid), vec4f(phi(vec3i(gid)), 0.0, 0.0, 0.0));
}
// -----------------------------------------------------------------------------

fn rasterizeSolidsAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  if (pagedSurfaceBindings()) { return; }
  let p = vec3i(gid); var fraction = 0.0; var owner = -1;
  if ((u32(round(params.container.w)) & 1u) != 0u) { fraction = clamp(textureLoad(terrainIn, vec2i(p.x, p.z), 0).x - f32(p.y), 0.0, 1.0); }
  if (!insideInflowChannel(worldCell(p))) {
    for (var bodyIndex = 0u; bodyIndex < 12u; bodyIndex += 1u) {
      if (bodyIndex >= params.control.w) { break; }
      let candidate = bodySolidFraction(rigidBodies[bodyIndex], p);
      if (candidate > fraction) { fraction = candidate; owner = i32(bodyIndex); }
    }
  }
  let word = 2u * index(gid);
  atomicStore(&solidOrSurface[word], bitcast<u32>(fraction));
  atomicStore(&solidOrSurface[word + 1u], bitcast<u32>(owner));
}

@compute @workgroup_size(4,4,4)
fn rasterizeSolids(@builtin(global_invocation_id) gid: vec3u) { rasterizeSolidsAt(gid); }

@compute @workgroup_size(4,4,4)
fn rasterizeSolidsActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  rasterizeSolidsAt(residentTopologyCell(wid, lid));
}

@compute @workgroup_size(4,4,4)
fn rasterizeSolidsRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  rasterizeSolidsAt(retiredTopologyCell(wid, lid));
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

fn invalidateFrontierAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let old = ownerAt(vec3i(gid));
  if (old.size > 0u && isOrigin(gid, old)) { frontierInvalidate(index(gid)); }
}

@compute @workgroup_size(4,4,4)
fn resetTopology(@builtin(global_invocation_id) gid: vec3u) { resetTopologyAt(gid); }

@compute @workgroup_size(4,4,4)
fn resetTopologyActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let gid = residentTopologyCell(wid, lid); invalidateFrontierAt(gid); resetTopologyAt(gid);
}

@compute @workgroup_size(4,4,4)
fn resetTopologyRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let gid = retiredTopologyCell(wid, lid); invalidateFrontierAt(gid); resetTopologyAt(gid);
}

struct FineLeafSummary {
  found: bool,
  complete: bool,
  coarseAuthority: bool,
  minimumPhi: f32,
  maximumPhi: f32,
  minimumAbsolutePhi: f32,
}
fn fineSummaryFinite(value: f32) -> bool { return value == value && abs(value) < 3.402823e38; }
fn fineSummaryHash(key: u32) -> u32 {
  var value = key * 0x9e3779b1u;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  return value ^ (value >> 15u);
}
fn fineSummaryOrderedFloat(value: u32) -> f32 {
  let mask = select(0x80000000u, 0xffffffffu, (value & 0x80000000u) == 0u);
  return bitcast<f32>(value ^ mask);
}
// Refinement-only bind groups alias pressureIn (binding 4) with the raw
// summary directory. Other entry points retain the normal pressure buffer.
fn fineSummaryLength() -> u32 { return arrayLength(&pressureIn); }
fn fineSummaryWord(index: u32) -> u32 { return bitcast<u32>(pressureIn[index]); }
fn fineLeafSummary(origin: vec3u, size: u32) -> FineLeafSummary {
  var result = FineLeafSummary(false, false, false, 3.402823e38, -3.402823e38, 3.402823e38);
  if (fineSummaryLength() < 16u || fineSummaryWord(0u) != 0u
      || fineSummaryWord(9u) != 0x80000000u) { return result; }
  let baseDims = vec3u(fineSummaryWord(4u), fineSummaryWord(5u), fineSummaryWord(6u));
  let cellDims = dims();
  if (any(cellDims == vec3u(0u)) || any(baseDims % cellDims != vec3u(0u))) { return result; }
  let ratios = baseDims / cellDims; let bricksPerCell = ratios.x;
  if (bricksPerCell == 0u || any(ratios != vec3u(bricksPerCell))) { return result; }
  var brickSide = size * bricksPerCell; var level = 0u;
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
  let brickOrigin = origin * bricksPerCell;
  if (any(brickOrigin % vec3u(brickSide) != vec3u(0u))) { return result; }
  let coordinate = brickOrigin / brickSide;
  if (any(coordinate >= levelDims)) { return result; }
  let key = levelOffset + coordinate.x + levelDims.x * (coordinate.y + levelDims.y * coordinate.z);
  if (key >= fineSummaryWord(10u)) { return result; }
  let hashCapacity = fineSummaryWord(3u); let maxProbes = min(fineSummaryWord(8u), 32u);
  if (hashCapacity == 0u || (hashCapacity & (hashCapacity - 1u)) != 0u || maxProbes == 0u) { return result; }
  let start = fineSummaryHash(key) & (hashCapacity - 1u);
  for (var probe = 0u; probe < 32u; probe += 1u) {
    if (probe >= maxProbes) { break; }
    let slot = (start + probe) & (hashCapacity - 1u); let base = 16u + slot * 8u;
    if (base + 7u >= fineSummaryLength()) { return result; }
    let observed = fineSummaryWord(base);
    if (observed == 0xffffffffu) { break; }
    if (observed != key) { continue; }
    let minimumPhi = fineSummaryOrderedFloat(fineSummaryWord(base + 1u));
    let maximumPhi = fineSummaryOrderedFloat(fineSummaryWord(base + 2u));
    let minimumAbsolutePhi = bitcast<f32>(fineSummaryWord(base + 3u));
    if (fineSummaryWord(base + 6u) != 0u || !fineSummaryFinite(minimumPhi)
        || !fineSummaryFinite(maximumPhi) || !fineSummaryFinite(minimumAbsolutePhi)) { return result; }
    let expectedBricks = brickSide * brickSide * brickSide;
    result.found = true; result.minimumPhi = minimumPhi; result.maximumPhi = maximumPhi;
    result.minimumAbsolutePhi = minimumAbsolutePhi;
    result.coarseAuthority = (fineSummaryWord(base + 7u) & 1u) != 0u;
    result.complete = result.coarseAuthority || (fineSummaryWord(base + 5u) == expectedBricks
      && fineSummaryWord(base + 4u) == expectedBricks * 64u);
    return result;
  }
  return result;
}

fn powerClosedWallStripIntersects(origin: vec3u, size: u32) -> bool {
  let flags = u32(round(params.container.w));
  if ((flags & 4u) == 0u) { return false; }
  let width = max(${OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS}u,
    u32(ceil(max(0.0, params.solve.w))));
  let high = origin + vec3u(size);
  let d = dims();
  // x+/-, z+/-, and the floor are closed for every container.  The ceiling
  // participates only for an authored closed-top scene (flag bit 1).  Any
  // intersecting leaf splits all the way to unit owners, putting wall samples
  // on the regular-cube Section 5 interpolation path instead of asking the
  // interior Delaunay catalog for sites outside the domain.
  return origin.x < min(width, d.x) || high.x > d.x - min(width, d.x)
    || origin.z < min(width, d.z) || high.z > d.z - min(width, d.z)
    || origin.y < min(width, d.y)
    || ((flags & 2u) != 0u && high.y > d.y - min(width, d.y));
}

fn leafNeedsRefinement(origin: vec3u, size: u32) -> bool {
  if (powerClosedWallStripIntersects(origin, size)) { return true; }
  var closestSurface = 3.402823e38; var minimumPhi = 3.402823e38; var maximumPhi = -3.402823e38; var minimumSolid = 1.0; var maximumSolid = 0.0;
  var minimumSurfaceVelocity = vec3f(0.0); var maximumSurfaceVelocity = vec3f(0.0); var maximumCurvatureProxy = 0.0; var hasSurfaceSample = false;
  let finestWidth = max(params.cellRelax.x, max(params.cellRelax.y, params.cellRelax.z));
  let baseBand = max(0.0, params.solve.w);
  let fineSummary = fineLeafSummary(origin, size);
  // A crossing observed by any published sparse descendant is already enough
  // to force refinement.  Absence is never interpreted as air or liquid.
  if (fineSummary.found && fineSummary.minimumPhi < 0.0 && fineSummary.maximumPhi >= 0.0) { return true; }
  if (fineSummary.complete) {
    closestSurface = fineSummary.minimumAbsolutePhi;
    minimumPhi = fineSummary.minimumPhi; maximumPhi = fineSummary.maximumPhi;
  }
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q = origin + vec3u(x,y,z); let solid = solidAt(vec3i(q)).fraction;
    var samplePhi = 0.0;
    if (!fineSummary.complete) {
      samplePhi=legacyPhi(vec3i(q));closestSurface=min(closestSurface,abs(samplePhi));
      minimumPhi=min(minimumPhi,samplePhi);maximumPhi=max(maximumPhi,samplePhi);
    } else if (surfaceDetailStrengthValue() > 0.0) { samplePhi = legacyPhi(vec3i(q)); }
    // Interval extrema above, rather than centre phi alone, preserve a thin
    // fine-corrected crossing during the summary-inconclusive fallback.
    minimumSolid = min(minimumSolid, solid); maximumSolid = max(maximumSolid, solid);
    if (surfaceDetailStrengthValue() > 0.0 && abs(samplePhi) < (baseBand + 2.0) * finestWidth) {
      let velocity = textureLoad(velocityIn, vec3i(q), 0).xyz;
      if (!hasSurfaceSample) { minimumSurfaceVelocity = velocity; maximumSurfaceVelocity = velocity; }
      else { minimumSurfaceVelocity = min(minimumSurfaceVelocity, velocity); maximumSurfaceVelocity = max(maximumSurfaceVelocity, velocity); }
      hasSurfaceSample = true;
      let p = vec3i(q);
      let laplacian = phi(p + vec3i(1,0,0)) + phi(p - vec3i(1,0,0))
        + phi(p + vec3i(0,1,0)) + phi(p - vec3i(0,1,0))
        + phi(p + vec3i(0,0,1)) + phi(p - vec3i(0,0,1)) - 6.0 * samplePhi;
      maximumCurvatureProxy = max(maximumCurvatureProxy, abs(laplacian) / max(finestWidth, 1e-6));
    }
  } } }
  // Interface-graded sizing: leaves that cross the liquid surface or a solid
  // boundary split, while uniform air, liquid, and solid bulk may stay coarse
  // once they are outside the explicit finest-cell interface band.
  let adaptivity = f32(params.control.x) / 1000.0;
  if (adaptivity <= 0.0) { return true; }
  let crossesSurface = minimumPhi < 0.0 && maximumPhi >= 0.0;
  let crossesSolidBoundary = maximumSolid - minimumSolid > 1e-5 || (maximumSolid > 1e-5 && maximumSolid < 1.0 - 1e-5);
  if (crossesSurface || crossesSolidBoundary) { return true; }
  if (minimumSolid >= 1.0 - 1e-5) { return false; }
  // This only widens the fine support band; it cannot coarsen a leaf selected
  // by the baseline signed-distance rule. Velocity span estimates deformation
  // over one configured maximum step, while the phi Laplacian detects curved
  // features. Keeping it inside the full-domain rebuild avoids stale brick
  // lists and preserves the dense transport/pressure authority.
  let strainActivity = select(0.0, length(maximumSurfaceVelocity - minimumSurfaceVelocity) * params.physical.z / max(finestWidth, 1e-6), hasSurfaceSample);
  let detailActivity = surfaceDetailStrengthValue() * clamp(max(strainActivity, 2.0 * maximumCurvatureProxy), 0.0, 1.0);
  let effectiveBand = baseBand + 8.0 * detailActivity;
  return closestSurface < effectiveBand * finestWidth;
}

fn splitLeaf(origin: vec3u, size: u32) {
  let child = size / 2u;
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let local = vec3u(x,y,z); let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerEnsure(origin + local, childOrigin, child);
  } } }
}

fn refineTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(vec3i(gid));
  if (owner.size > 1u && (targetRefinementSize == 0u || owner.size == targetRefinementSize) && isOrigin(gid, owner) && leafNeedsRefinement(gid, owner.size)) { splitLeaf(gid, owner.size); }
}

@compute @workgroup_size(4,4,4)
fn refineTopology(@builtin(global_invocation_id) gid: vec3u) { refineTopologyAt(gid * 2u); }

@compute @workgroup_size(4,4,4)
fn refineTopologyActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  refineTopologyAt(residentTopologyCandidate(wid, lid));
}

@compute @workgroup_size(4,4,4)
fn refineTopologyRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  refineTopologyAt(retiredTopologyCandidate(wid, lid));
}

// Large leaves are deliberately rare. One 128-lane workgroup evaluates the
// exact scalar sizing predicate once, then publishes child owners
// cooperatively. The predicate's cubic scan remains runtime-bounded below so
// browser Metal compilers cannot specialize 16^3/32^3 into a giant kernel.
var<workgroup> refineEligible: atomic<u32>;
var<workgroup> refineDecision: atomic<u32>;
var<workgroup> refineRuntimeSize: atomic<u32>;

@compute @workgroup_size(128)
fn refineTopologyCoarse(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  refineCoarseBlock(wid * vec3u(targetRefinementSize), lid);
}

@compute @workgroup_size(128)
fn refineTopologyCoarseActive(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let tile = worklistTileOrigin(wid.x, 16u);
  let blocks = max(1u, topologyTileSize() / targetRefinementSize);
  for (var sub = 0u; sub < blocks * blocks * blocks; sub += 1u) {
    let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
    refineCoarseBlock(tile + subCoord * vec3u(targetRefinementSize), lid);
  }
}

@compute @workgroup_size(128)
fn refineTopologyCoarseRetired(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let tile = worklistTileOrigin(wid.x, retiredTileIndexBase());
  let blocks = max(1u, topologyTileSize() / targetRefinementSize);
  for (var sub = 0u; sub < blocks * blocks * blocks; sub += 1u) {
    let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
    refineCoarseBlock(tile + subCoord * vec3u(targetRefinementSize), lid);
  }
}

fn refineCoarseBlock(origin: vec3u, lid: u32) {
  // Keep the exact scalar sizing predicate as the single source of truth.
  // Lane 0 evaluates the same predicate used by the portable candidate
  // kernel, then the whole workgroup publishes the selected children.
  if (lid == 0u) {
    let inBounds = all(origin < dims());
    let owner = ownerAt(vec3i(min(origin, dims() - vec3u(1u))));
    let eligible = inBounds && owner.size == targetRefinementSize && isOrigin(origin, owner);
    var decision = false;
    if (eligible) { decision = leafNeedsRefinement(origin, owner.size); }
    atomicStore(&refineEligible, select(0u, 1u, eligible));
    atomicStore(&refineDecision, select(0u, 1u, decision));
    // Preserve the storage-loaded size across the barrier. Using the pipeline
    // override as the cubic loop bound lets some browser Metal compilers fully
    // specialize size^3 at 16/32 and produce a watchdog-scale kernel.
    atomicStore(&refineRuntimeSize, max(1u, owner.size));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&refineEligible) == 0u || workgroupUniformLoad(&refineDecision) == 0u) { return; }
  let size = workgroupUniformLoad(&refineRuntimeSize);
  let cells = size * size * size;
  let child = size / 2u;
  ensureLeafOwnerPages(origin, size, lid, 128u);
  workgroupBarrier();
  for (var flat = lid; flat < cells; flat += 128u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerEnsure(origin + local, childOrigin, child);
  }
}

fn neighborTooFine(origin: vec3u, size: u32) -> bool {
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) {
    let q0 = vec3i(origin + vec3u(0u,y,z)); let q1 = vec3i(origin + vec3u(size-1u,y,z));
    if ((valid(q0-vec3i(1,0,0)) && ownerAt(q0-vec3i(1,0,0)).size * 2u < size) || (valid(q1+vec3i(1,0,0)) && ownerAt(q1+vec3i(1,0,0)).size * 2u < size)) { return true; }
  } }
  for (var z = 0u; z < size; z += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q0 = vec3i(origin + vec3u(x,0u,z)); let q1 = vec3i(origin + vec3u(x,size-1u,z));
    if ((valid(q0-vec3i(0,1,0)) && ownerAt(q0-vec3i(0,1,0)).size * 2u < size) || (valid(q1+vec3i(0,1,0)) && ownerAt(q1+vec3i(0,1,0)).size * 2u < size)) { return true; }
  } }
  for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q0 = vec3i(origin + vec3u(x,y,0u)); let q1 = vec3i(origin + vec3u(x,y,size-1u));
    if ((valid(q0-vec3i(0,0,1)) && ownerAt(q0-vec3i(0,0,1)).size * 2u < size) || (valid(q1+vec3i(0,0,1)) && ownerAt(q1+vec3i(0,0,1)).size * 2u < size)) { return true; }
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
fn paperStrictlyObtuseCoarseMask(mask: u32) -> bool {
  return mask == 25u || mask == 42u || mask == 52u || mask == 57u || mask == 58u || mask == 60u;
}
fn repairPaperAcuteNeighbors(origin: vec3u, size: u32) {
  // The six same/coarser bits are X/Y/Z faces followed by XY/XZ/YZ edges,
  // oriented away from this child's parent. Exactly six masks contain a
  // unique ordinary-Delaunay simplex with current-cell solid angle > pi/2.
  // Split its sole coarse face before descriptor publication; this changes
  // the local link to the paper's nonobtuse Cartesian limiting case without
  // inventing a redistance fallback.
  let child = (origin / vec3u(size)) & vec3u(1u);
  let outward = vec3i(select(-1, 1, child.x == 1u), select(-1, 1, child.y == 1u), select(-1, 1, child.z == 1u));
  let wanted = array<vec3i,6>(vec3i(outward.x,0,0), vec3i(0,outward.y,0), vec3i(0,0,outward.z),
    vec3i(outward.x,outward.y,0), vec3i(outward.x,0,outward.z), vec3i(0,outward.y,outward.z));
  var mask = 0u;
  for (var bit = 0u; bit < 6u; bit += 1u) {
    let probe = paperProbe(origin, size, wanted[bit]);
    if (valid(probe) && ownerAt(probe).size == size * 2u) { mask |= 1u << bit; }
  }
  if (!paperStrictlyObtuseCoarseMask(mask)) { return; }
  let faceBit = u32(firstTrailingBit(mask & 7u));
  if (faceBit >= 3u) { return; }
  let coarse = ownerAt(paperProbe(origin, size, wanted[faceBit]));
  if (coarse.size == size * 2u) { splitLeaf(unpackOrigin(coarse.packedOrigin), coarse.size); }
}
fn repairPaperMixedNeighbors(origin: vec3u, size: u32) {
  repairPaperAcuteNeighbors(origin, size);
  var finer = false; var coarser = false;
  for (var bit = 0u; bit < 18u; bit += 1u) {
    let probe = paperProbe(origin, size, PAPER_DIRECTIONS[bit]); if (!valid(probe)) { continue; }
    let neighborSize = ownerAt(probe).size; finer = finer || neighborSize < size; coarser = coarser || neighborSize > size;
  }
  if (!finer || !coarser) { return; }
  // This is the exact deterministic rule in plan section 7.3 and the CPU
  // oracle: split every coarse face/edge neighbor of the mixed anchor once.
  for (var bit = 0u; bit < 18u; bit += 1u) {
    let probe = paperProbe(origin, size, PAPER_DIRECTIONS[bit]); if (!valid(probe)) { continue; }
    let neighbor = ownerAt(probe); if (neighbor.size > size) { splitLeaf(unpackOrigin(neighbor.packedOrigin), neighbor.size); }
  }
}

fn balanceTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  // The portable balance dispatch is rooted at even coordinates. Inspect its
  // complete 2^3 child block so a unit-size anchor can request splitting the
  // implicated size-two coarse face before descriptor publication.
  let unitBase = gid & vec3u(0xfffffffeu);
  for (var childIndex = 0u; childIndex < 8u; childIndex += 1u) {
    let child = vec3u(childIndex & 1u, (childIndex >> 1u) & 1u, (childIndex >> 2u) & 1u);
    let q = unitBase + child;
    if (any(q >= dims())) { continue; }
    let unitOwner = ownerAt(vec3i(q));
    if (unitOwner.size == 1u && isOrigin(q, unitOwner)) { repairPaperAcuteNeighbors(q, 1u); }
  }
  let owner = ownerAt(vec3i(gid));
  if (owner.size >= 2u && owner.size <= 16u && isOrigin(gid, owner)) { repairPaperMixedNeighbors(gid, owner.size); }
  // Size-16+ leaves use the cooperative entry point below.
  if (owner.size > 2u && owner.size < 16u && isOrigin(gid, owner) && neighborTooFine(gid, owner.size)) { splitLeaf(gid, owner.size); }
}


@compute @workgroup_size(4,4,4)
fn balanceTopology(@builtin(global_invocation_id) gid: vec3u) { balanceTopologyAt(gid * 2u); }

@compute @workgroup_size(4,4,4)
fn balanceTopologyActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  balanceTopologyAt(residentTopologyCandidate(wid, lid));
}

@compute @workgroup_size(4,4,4)
fn balanceTopologyRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  balanceTopologyAt(retiredTopologyCandidate(wid, lid));
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
fn balanceTopologyCoarseActive(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let tile = worklistTileOrigin(wid.x, 16u);
  let blocks = max(1u, topologyTileSize() / targetRefinementSize);
  for (var sub = 0u; sub < blocks * blocks * blocks; sub += 1u) {
    let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
    balanceCoarseBlock(tile + subCoord * vec3u(targetRefinementSize), lid);
  }
}

@compute @workgroup_size(256)
fn balanceTopologyCoarseRetired(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let tile = worklistTileOrigin(wid.x, retiredTileIndexBase());
  let blocks = max(1u, topologyTileSize() / targetRefinementSize);
  for (var sub = 0u; sub < blocks * blocks * blocks; sub += 1u) {
    let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
    balanceCoarseBlock(tile + subCoord * vec3u(targetRefinementSize), lid);
  }
}

fn balanceCoarseBlock(origin: vec3u, lid: u32) {
  // See refineCoarseBlock: the sentinel/bounds rejection flows through the
  // lane-0 eligibility store to keep barrier control flow formally uniform.
  if (lid == 0u) {
    let inBounds = all(origin < dims());
    let owner = ownerAt(vec3i(min(origin, dims() - vec3u(1u))));
    atomicStore(&balanceEligible, select(0u, 1u, inBounds && owner.size == targetRefinementSize && isOrigin(origin, owner)));
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
    if (valid(outside) && ownerAt(outside).size * 2u < size) { needsSplit = 1u; }
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
  ensureLeafOwnerPages(origin, size, lid, 256u);
  workgroupBarrier();
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerEnsure(origin + local, childOrigin, child);
  }
}

fn hydrostaticSplit() -> bool { return params.hydrostatic.z > 0.5; }
fn ownerCentreY(owner: Owner) -> f32 {
  let origin = unpackOrigin(owner.packedOrigin);
  return (f32(origin.y) + 0.5 * f32(owner.size)) * params.cellRelax.y;
}
fn fixedHydrostaticPotential(y_m: f32) -> f32 {
  let restY_m = params.hydrostatic.w * params.cellRelax.y;
  return -params.hydrostatic.x * params.hydrostatic.y * max(restY_m - y_m, 0.0);
}
// Total pressure is zero in air, so the perturbation Dirichlet value is -p_h
// at the resident level-set crossing. Keeping this known value on the RHS lets
// the octree solve surface elevation without ever representing p_h in a leaf.
fn hydrostaticAirPressure(a: Owner, b: Owner, phiA: f32, phiB: f32) -> f32 {
  if (!hydrostaticSplit() || (phiA < 0.0) == (phiB < 0.0)) { return 0.0; }
  let t = clamp(-phiA / (phiB - phiA), 0.0, 1.0);
  let surfaceY = mix(ownerCentreY(a), ownerCentreY(b), t);
  return -fixedHydrostaticPotential(surfaceY);
}
fn pressureOf(owner: Owner) -> f32 {
  let slot = pressureIndex(owner);
  if (slot >= arrayLength(&pressureIn)) { return 0.0; }
  return pressureIn[slot];
}
fn velocityAt(p: vec3i) -> vec3f { if (!valid(p)) { return vec3f(0.0); } return textureLoad(velocityIn, p, 0).xyz; }

fn inverseInertiaResponse(body: RigidBody, value: vec3f) -> vec3f {
  let local = quaternionInverseRotate(body.orientation, value);
  return quaternionRotate(body.orientation, local * body.inverseMassInertia.yzw);
}

// One row of K M^-1 K^T, where
// K = grad^T V (1-A) L. The first component is the current matrix action
// against the globally gathered M^-1 K^T p vector; the second is this row's
// exact rank-six diagonal, used by weighted Jacobi.
fn leafBodyCoupling(origin: vec3u, size: u32) -> vec2f {
  if (params.physical.y < 0.5) { return vec2f(0.0); }
  var linear: array<vec3f, 12>;
  var angular: array<vec3f, 12>;
  for (var face = 0u; face < 6u; face += 1u) {
    let axis = face / 2u; let side = select(-1, 1, (face & 1u) == 1u); let e = axisVector(axis); let area = faceArea(axis);
    for (var b = 0u; b < size; b += 1u) { for (var a = 0u; a < size; a += 1u) {
      var local = vec3u(0u); local[axis] = select(0u, size - 1u, side > 0);
      local[(axis + 1u) % 3u] = a; local[(axis + 2u) % 3u] = b;
      let inside = vec3i(origin + local); let outside = inside + side * e;
      if (!valid(outside)) { continue; }
      let solid = faceSolid(inside, outside);
      if (solid.owner < 0 || u32(solid.owner) >= params.control.w || solid.fraction <= 0.0) { continue; }
      let bodyIndex = u32(solid.owner); let factor = -f32(side) * area * solid.fraction;
      var generator = vec3f(0.0); generator[axis] = factor;
      let faceCell = select(outside, inside, side > 0); let arm = faceWorld(faceCell, axis) - rigidBodies[bodyIndex].positionShape.xyz;
      linear[bodyIndex] += generator; angular[bodyIndex] += cross(arm, generator);
    } }
  }
  var action = 0.0; var diagonal = 0.0;
  for (var bodyIndex = 0u; bodyIndex < 12u; bodyIndex += 1u) {
    if (bodyIndex >= params.control.w) { break; }
    let body = rigidBodies[bodyIndex]; let linearGenerator = linear[bodyIndex]; let angularGenerator = angular[bodyIndex];
    let linearResponse = body.inverseMassInertia.x * linearGenerator;
    let angularResponse = inverseInertiaResponse(body, angularGenerator);
    diagonal += dot(linearGenerator, linearResponse) + dot(angularGenerator, angularResponse);
    let accelerated = vec3f(couplingAcceleration(bodyIndex, 0u), couplingAcceleration(bodyIndex, 1u), couplingAcceleration(bodyIndex, 2u));
    let angularAccelerated = vec3f(couplingAcceleration(bodyIndex, 3u), couplingAcceleration(bodyIndex, 4u), couplingAcceleration(bodyIndex, 5u));
    action += dot(linearGenerator, accelerated) + dot(angularGenerator, angularAccelerated);
  }
  return vec2f(action, diagonal);
}

fn accumulateFace(origin: vec3u, size: u32, axis: u32, side: i32, diagonal: ptr<function, f32>, sum: ptr<function, f32>, boundarySum: ptr<function, f32>, flux: ptr<function, f32>) {
  let owner = ownerAt(vec3i(origin));
  let e = axisVector(axis); let area = faceArea(axis);
  for (var b = 0u; b < size; b += 1u) { for (var a = 0u; a < size; a += 1u) {
    var local = vec3u(0u); local[axis] = select(0u, size - 1u, side > 0);
    local[(axis + 1u) % 3u] = a; local[(axis + 2u) % 3u] = b;
    let inside = vec3i(origin + local); let outside = inside + side * e;
    if (!valid(outside)) { continue; }
    let neighbor = ownerAt(outside); let distance = pressureDistance(ownerAt(vec3i(origin)), neighbor, axis);
    let solid = faceSolid(inside, outside); let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
    let coefficient = open * area / max(distance, 1e-7);
    (*diagonal) += coefficient;
    if (liquidOwner(neighbor)) {
      (*sum) += coefficient * pressureOf(neighbor);
    } else if (hydrostaticSplit()) {
      (*boundarySum) += coefficient * hydrostaticAirPressure(owner, neighbor, ownerPhi(owner), ownerPhi(neighbor));
    }
    let faceCell = select(outside, inside, side > 0);
    (*flux) += f32(side) * area * constrainedFaceVelocity(faceCell, axis, solid);
  } }
}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(vec3i(gid)); let idx = index(gid);
  if (!isOrigin(gid, owner) || !liquidOwner(owner)) { pressureOut[idx] = 0.0; return; }
  var diagonal = 0.0; var sum = 0.0; var boundarySum = 0.0; var flux = 0.0;
  accumulateFace(gid, owner.size, 0u, -1, &diagonal, &sum, &boundarySum, &flux); accumulateFace(gid, owner.size, 0u, 1, &diagonal, &sum, &boundarySum, &flux);
  accumulateFace(gid, owner.size, 1u, -1, &diagonal, &sum, &boundarySum, &flux); accumulateFace(gid, owner.size, 1u, 1, &diagonal, &sum, &boundarySum, &flux);
  accumulateFace(gid, owner.size, 2u, -1, &diagonal, &sum, &boundarySum, &flux); accumulateFace(gid, owner.size, 2u, 1, &diagonal, &sum, &boundarySum, &flux);
  let coupling = leafBodyCoupling(gid, owner.size); let old = pressureIn[idx]; let effectiveDiagonal = diagonal + coupling.y;
  let rhs = flux - boundarySum;
  let next = select(0.0, old + (sum - rhs - diagonal * old - coupling.x) / effectiveDiagonal, effectiveDiagonal > 1e-8);
  pressureOut[idx] = mix(pressureIn[idx], next, params.cellRelax.w);
}

// --- Compacted leaf solve -------------------------------------------------
// The dense jacobi above launches one thread per finest cell and rebuilds the
// entire row (coefficients, neighbor lookups, velocity flux) every sweep. The
// kernels below run the classic GPU-octree pipeline instead: a prefix-sum
// stream compaction of liquid leaf origins, a one-time assembly of each row's
// diagonal / RHS flux / merged neighbor table, then iteration kernels that
// only gather cached entries. 2:1 balance bounds a leaf's distinct neighbors
// at 4 per face (24 total; 6 for finest leaves), which also bounds the entry
// pool at 6 entries per finest cell.

// The leaf frontier survives topology rebuilds. Partial rebuilds invalidate
// only origins inside active/retired topology tiles; filtering preserves every
// untouched liquid leaf and tile-local append publishes the replacement
// origins. The dense finest lattice is scanned only for cold initialization.
@compute @workgroup_size(1)
fn beginFrontier() {
  let current = frontierCurrent();
  let next = 1u - current;
  atomicStore(&frontier[next], 0u);
  let control = pressureControlBase();
  compaction[control] = 0u;
  compaction[control + 1u] = 0u;
  compaction[control + 2u] = 0u;
  let blocks = (frontierCount(current) + 255u) / 256u;
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  compaction[12] = x; compaction[13] = y; compaction[14] = 1u;
}

@compute @workgroup_size(256)
fn filterFrontier(@builtin(global_invocation_id) gid: vec3u) {
  let current = frontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  if (slot >= frontierCount(current)) { return; }
  let cell = frontierCell(current, slot);
  if (frontierAlive(cell) && isOrigin(cellCoord(cell), ownerAtIndex(cell))) {
    let next = 1u - current;
    let output = atomicAdd(&frontier[next], 1u);
    if (output < frontierListCapacity()) {
      atomicStore(&frontier[frontierBase(next) + output], cell);
    } else {
      frontierInvalidate(cell);
    }
  } else { frontierInvalidate(cell); }
}

fn appendFrontierAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let cell = index(gid);
  let owner = ownerAtIndex(cell);
  if (isOrigin(gid, owner) && liquidOwner(owner)) { frontierAppend(1u - frontierCurrent(), cell); }
}

@compute @workgroup_size(4,4,4)
fn appendFrontier(@builtin(global_invocation_id) gid: vec3u) { appendFrontierAt(gid); }

@compute @workgroup_size(4,4,4)
fn appendFrontierActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  appendFrontierAt(residentTopologyCell(wid, lid));
}

@compute @workgroup_size(4,4,4)
fn appendFrontierRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  appendFrontierAt(retiredTopologyCell(wid, lid));
}

@compute @workgroup_size(1)
fn finalizeFrontier() {
  let next = 1u - frontierCurrent();
  let required = atomicLoad(&frontier[next]);
  let count = frontierCount(next);
  atomicStore(&frontier[2], next);
  atomicAdd(&frontier[3], 1u);
  if (rowIndexedPressure && required > frontierListCapacity()) {
    let control = pressureControlBase();
    compaction[control] = 2u;
    compaction[control + 1u] = required;
    compaction[control + 2u] = 0u;
    compaction[control + 3u] = (dims().x + 3u) / 4u;
    compaction[control + 4u] = (dims().y + 3u) / 4u;
    compaction[control + 5u] = (dims().z + 3u) / 4u;
  }
  let blocks = (count + 255u) / 256u;
  compaction[8] = blocks;
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  compaction[12] = x; compaction[13] = y; compaction[14] = 1u;
}

fn cellCoord(c: u32) -> vec3u {
  let nx = params.dimsMax.x; let ny = params.dimsMax.y;
  return vec3u(c % nx, (c / nx) % ny, c / (nx * ny));
}

var<workgroup> bodyCouplingScratch: array<f32, 256>;
var<workgroup> bodyGeneralized: array<f32, 6>;
var<workgroup> bodyCouplingOverflow: atomic<u32>;

// One workgroup per body evaluates K^T p over every finest positive face,
// then stores rho M^-1 K^T p in the dead tail of the compaction buffer. A
// following dispatch applies the rank-six response without atomics or a host
// synchronization point.
@compute @workgroup_size(256)
fn gatherBodyCoupling(@builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x; let bodyIndex = wid.x;
  if (lid == 0u) { atomicStore(&bodyCouplingOverflow, select(0u, 1u, pressureOverflowed())); }
  workgroupBarrier();
  if (workgroupUniformLoad(&bodyCouplingOverflow) != 0u || bodyIndex >= params.control.w) { return; }
  let body = rigidBodies[bodyIndex];
  var force = vec3f(0.0); var torque = vec3f(0.0);
  for (var flat = lid; flat < cellCount() * 3u; flat += 256u) {
    let axis = flat % 3u; let p = vec3i(cellCoord(flat / 3u)); let q = p + axisVector(axis);
    if (!valid(q)) { continue; }
    let solid = faceSolid(p, q);
    if (solid.owner < 0 || u32(solid.owner) != bodyIndex || solid.fraction <= 0.0) { continue; }
    let left = ownerAt(p); let right = ownerAt(q); let leftWet = liquidOwner(left); let rightWet = liquidOwner(right);
    if (!leftWet && !rightWet) { continue; }
    let p0 = select(0.0, pressureOf(left), leftWet); let p1 = select(0.0, pressureOf(right), rightWet);
    let magnitude = faceArea(axis) * solid.fraction * (p1 - p0);
    var generator = vec3f(0.0); generator[axis] = magnitude;
    force += generator; torque += cross(faceWorld(p, axis) - body.positionShape.xyz, generator);
  }
  let values = array<f32, 6>(force.x, force.y, force.z, torque.x, torque.y, torque.z);
  for (var component = 0u; component < 6u; component += 1u) {
    bodyCouplingScratch[lid] = values[component];
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) { bodyCouplingScratch[lid] += bodyCouplingScratch[lid + stride]; }
    }
    workgroupBarrier();
    if (lid == 0u) { bodyGeneralized[component] = bodyCouplingScratch[0]; }
    workgroupBarrier();
  }
  if (lid == 0u) {
    let linear = body.inverseMassInertia.x * vec3f(bodyGeneralized[0], bodyGeneralized[1], bodyGeneralized[2]);
    let angular = inverseInertiaResponse(body, vec3f(bodyGeneralized[3], bodyGeneralized[4], bodyGeneralized[5]));
    let response = array<f32, 6>(linear.x, linear.y, linear.z, angular.x, angular.y, angular.z);
    for (var component = 0u; component < 6u; component += 1u) { compaction[couplingBase() + bodyIndex * 8u + component] = bitcast<u32>(response[component]); }
  }
}

// Commit the same-step pressure response to the GPU-resident body velocity.
// Subsequent substeps and the shared tangential Brinkman pass therefore see
// the same state whose equal-and-opposite impulse is returned to the CPU.
@compute @workgroup_size(64)
fn applyBodyCoupling(@builtin(global_invocation_id) gid: vec3u) {
  let bodyIndex = gid.x;
  if (pressureOverflowed() || params.physical.y < 0.5 || bodyIndex >= params.control.w) { return; }
  var body = rigidBodies[bodyIndex];
  let linear = vec3f(couplingAcceleration(bodyIndex, 0u), couplingAcceleration(bodyIndex, 1u), couplingAcceleration(bodyIndex, 2u));
  let angular = vec3f(couplingAcceleration(bodyIndex, 3u), couplingAcceleration(bodyIndex, 4u), couplingAcceleration(bodyIndex, 5u));
  body.linearVelocity = vec4f(body.linearVelocity.xyz - linear, body.linearVelocity.w);
  body.angularVelocity = vec4f(body.angularVelocity.xyz - angular, body.angularVelocity.w);
  rigidBodies[bodyIndex] = body;
}

fn leafInfo(c: u32) -> vec3u {
  let owner = ownerAtIndex(c);
  if (!frontierAlive(c) || !isOrigin(cellCoord(c), owner)) { return vec3u(0u); }
  var coarseTasks = 0u;
  if (owner.size >= 8u) {
    let tiles = owner.size / 8u;
    coarseTasks = select(tiles * tiles * tiles, 1u, rowIndexedPressure);
  }
  return vec3u(1u, select(24u, 6u, owner.size == 1u), coarseTasks);
}

var<workgroup> scanPairs: array<vec3u, 256>;
var<workgroup> emitOverflow: atomic<u32>;

@compute @workgroup_size(256)
fn planLeaves(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  var value = vec3u(0u);
  let current = frontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  if (slot < frontierCount(current)) { value = leafInfo(frontierCell(current, slot)); }
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
    let rowOverflow = rowIndexedPressure && (total.x > params.pressureCapacity.x || total.y > params.pressureCapacity.y || total.z > coarseTaskCapacity());
    let overflow = frontierOverflow || rowOverflow;
    let publishedRows = select(total.x, 0u, overflow);
    let publishedEntries = select(total.y, 0u, overflow);
    compaction[0] = publishedRows; compaction[1] = publishedEntries;
    compaction[control] = select(0u, 2u, frontierOverflow) | select(0u, 1u, rowOverflow);
    compaction[control + 1u] = max(total.x, select(0u, compaction[control + 1u], frontierOverflow));
    compaction[control + 2u] = total.y;
    compaction[control + 3u] = select(0u, (dims().x + 3u) / 4u, overflow);
    compaction[control + 4u] = select(1u, (dims().y + 3u) / 4u, overflow);
    compaction[control + 5u] = select(1u, (dims().z + 3u) / 4u, overflow);
    let blocks = (publishedRows + 255u) / 256u;
    let x = min(blocks, 65535u);
    var y = 1u;
    if (x > 0u) { y = (blocks + x - 1u) / x; }
    compaction[2] = x; compaction[3] = y; compaction[4] = 1u;
    // Coarse velocity work is tiled into 8^3 chunks. This both bounds the work
    // per workgroup and restores occupancy for a handful of very large leaves.
    let cooperativeTasks = select(total.z, 0u, overflow);
    let coarseX = min(cooperativeTasks, 65535u);
    var coarseY = 1u;
    if (coarseX > 0u) { coarseY = (cooperativeTasks + coarseX - 1u) / coarseX; }
    compaction[5] = coarseX; compaction[6] = coarseY; compaction[7] = 1u; compaction[8] = cooperativeTasks;
    let leafX = min(publishedRows, 65535u);
    var leafY = 1u;
    if (leafX > 0u) { leafY = (publishedRows + leafX - 1u) / leafX; }
    compaction[9] = leafX; compaction[10] = leafY; compaction[11] = 1u;
  }
}

@compute @workgroup_size(256)
fn emitLeaves(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  if (lid == 0u) { atomicStore(&emitOverflow, select(0u, 1u, pressureOverflowed())); }
  workgroupBarrier();
  if (workgroupUniformLoad(&emitOverflow) != 0u) { return; }
  var value = vec3u(0u);
  let current = frontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  var cell = 0u;
  if (slot < frontierCount(current)) { cell = frontierCell(current, slot); value = leafInfo(cell); }
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
    let start = compaction[16u + 3u * block] + exclusive.y;
    let taskStart = compaction[17u + 3u * block] + exclusive.z;
    let cooperative = select(0u, 1u, value.z > 0u && taskStart + value.z <= coarseTaskCapacity());
    let previousWord = frontierRowWord(cell);
    let previousRow = select(0xffffffffu, previousWord - 2u, previousWord >= 2u);
    var warm = 0.0;
    if (rowIndexedPressure && previousRow < arrayLength(&pressureIn)) { warm = pressureIn[previousRow]; }
    if (rowIndexedPressure) {
      pressureOut[row] = select(0.0, warm, (params.pressureCapacity.w & 1u) != 0u);
      frontierSetRow(cell, row + 2u);
    }
    leafHeaders[row] = LeafHeader(cell, start, 0u, ownerAtIndex(cell).size, 0.0, 0.0, cooperative, 0u, vec4f(0.0));
    if (cooperative == 1u) {
      for (var tile = 0u; tile < value.z; tile += 1u) {
        let task = taskStart + tile;
        compaction[coarseTaskListBase() + 2u * task] = row;
        compaction[coarseTaskListBase() + 2u * task + 1u] = tile;
      }
    }
  }
}

fn compactRowIndex(gid: vec3u) -> u32 { return gid.x + gid.y * compaction[2] * 256u; }

@compute @workgroup_size(256)
fn assembleSystem(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  var header = leafHeaders[row];
  // Size>=8 rows are emitted to a separate deterministic stream and assembled
  // by one cooperative workgroup each. Tiny rows retain this occupancy-friendly
  // one-invocation path instead of paying for 64 mostly idle lanes.
  if (header.pad0 == 1u) { return; }
  let origin = cellCoord(header.cell);
  let owner = ownerAtIndex(header.cell);
  let size = header.size;
  var neighborCells: array<u32, 24>;
  var neighborCoefficients: array<f32, 24>;
  var neighborCount = 0u;
  var diagonal = 0.0;
  var boundarySum = 0.0;
  var flux = 0.0;
  for (var face = 0u; face < 6u; face += 1u) {
    let axis = face / 2u;
    let side = select(-1, 1, (face & 1u) == 1u);
    let e = axisVector(axis); let area = faceArea(axis);
    for (var b = 0u; b < size; b += 1u) { for (var a = 0u; a < size; a += 1u) {
      var local = vec3u(0u); local[axis] = select(0u, size - 1u, side > 0);
      local[(axis + 1u) % 3u] = a; local[(axis + 2u) % 3u] = b;
      let inside = vec3i(origin + local); let outside = inside + side * e;
      if (!valid(outside)) { continue; }
      let neighbor = ownerAt(outside);
      let distance = pressureDistance(ownerAt(vec3i(origin)), neighbor, axis);
      let solid = faceSolid(inside, outside); let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
      let coefficient = open * area / max(distance, 1e-7);
      diagonal += coefficient;
      if (liquidOwner(neighbor)) {
        let neighborCell = pressureIndex(neighbor);
        var found = false;
        for (var j = 0u; j < neighborCount; j += 1u) {
          if (neighborCells[j] == neighborCell) { neighborCoefficients[j] += coefficient; found = true; break; }
        }
        if (!found && neighborCount < 24u) {
          neighborCells[neighborCount] = neighborCell;
          neighborCoefficients[neighborCount] = coefficient;
          neighborCount += 1u;
        }
      } else if (hydrostaticSplit()) { boundarySum += coefficient * hydrostaticAirPressure(owner, neighbor, ownerPhi(owner), ownerPhi(neighbor)); }
      let faceCell = select(outside, inside, side > 0);
      flux += f32(side) * area * constrainedFaceVelocity(faceCell, axis, solid);
    } }
  }
  header.entryCount = neighborCount; header.diagonal = diagonal; header.rhs = flux - boundarySum;
  header.gradient = vec4f(ownerPhiGradient(owner),ownerPhi(owner));
  leafHeaders[row] = header;
  for (var j = 0u; j < neighborCount; j += 1u) { leafEntries[header.entryStart + j] = LeafEntry(neighborCells[j], neighborCoefficients[j]); }
}

// A balanced neighbor is no smaller than half this leaf, so each face quadrant
// touches at most one neighbor. Sixty-four lanes can therefore reduce all
// size^2 finest subfaces into four deterministic coefficients per face without
// atomics. Duplicate quadrant entries for a same-size/coarser neighbor are
// algebraically identical to the merged serial entry and retain a fixed layout.
var<workgroup> coarseDiagonalScratch: array<f32, 64>;
var<workgroup> coarseFluxScratch: array<f32, 64>;
var<workgroup> coarseBoundaryScratch: array<f32, 64>;
var<workgroup> coarseCoefficientScratch: array<f32, 256>;
var<workgroup> coarseTaskEligible: atomic<u32>;

@compute @workgroup_size(64)
fn assembleCoarseSystem(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let task = coarseTaskIndex(wid);
  if (lid == 0u) { atomicStore(&coarseTaskEligible, select(0u, 1u, coarseTaskTile(task) == 0u)); }
  workgroupBarrier();
  if (workgroupUniformLoad(&coarseTaskEligible) == 0u) { return; }
  let row = coarseTaskRow(task);
  var header = leafHeaders[row];
  let origin = cellCoord(header.cell);
  let owner = ownerAtIndex(header.cell);
  let size = header.size;
  let half = size / 2u;
  var diagonal = 0.0;
  var boundarySum = 0.0;
  var flux = 0.0;

  for (var face = 0u; face < 6u; face += 1u) {
    let axis = face / 2u;
    let side = select(-1, 1, (face & 1u) == 1u);
    let e = axisVector(axis);
    let area = faceArea(axis);
    var laneDiagonal = 0.0;
    var laneBoundary = 0.0;
    var laneFlux = 0.0;
    var laneCoefficients = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    let faceSamples = size * size;
    for (var sample = lid; sample < faceSamples; sample += 64u) {
      let a = sample % size;
      let b = sample / size;
      var local = vec3u(0u);
      local[axis] = select(0u, size - 1u, side > 0);
      local[(axis + 1u) % 3u] = a;
      local[(axis + 2u) % 3u] = b;
      let inside = vec3i(origin + local);
      let outside = inside + side * e;
      if (!valid(outside)) { continue; }
      let neighbor = ownerAt(outside);
      let solid = faceSolid(inside, outside);
      let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
      let coefficient = open * area / max(pressureDistance(owner, neighbor, axis), 1e-7);
      laneDiagonal += coefficient;
      if (liquidOwner(neighbor)) {
        let quadrant = select(0u, 1u, a >= half) + select(0u, 2u, b >= half);
        laneCoefficients[quadrant] += coefficient;
      } else if (hydrostaticSplit()) {
        laneBoundary += coefficient * hydrostaticAirPressure(owner, neighbor, ownerPhi(owner), ownerPhi(neighbor));
      }
      let faceCell = select(outside, inside, side > 0);
      laneFlux += f32(side) * area * constrainedFaceVelocity(faceCell, axis, solid);
    }
    coarseDiagonalScratch[lid] = laneDiagonal;
    coarseBoundaryScratch[lid] = laneBoundary;
    coarseFluxScratch[lid] = laneFlux;
    for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) {
      coarseCoefficientScratch[quadrant * 64u + lid] = laneCoefficients[quadrant];
    }
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) {
        coarseDiagonalScratch[lid] += coarseDiagonalScratch[lid + stride];
        coarseBoundaryScratch[lid] += coarseBoundaryScratch[lid + stride];
        coarseFluxScratch[lid] += coarseFluxScratch[lid + stride];
        for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) {
          let slot = quadrant * 64u + lid;
          coarseCoefficientScratch[slot] += coarseCoefficientScratch[slot + stride];
        }
      }
    }
    workgroupBarrier();
    if (lid == 0u) {
      diagonal += coarseDiagonalScratch[0];
      boundarySum += coarseBoundaryScratch[0];
      flux += coarseFluxScratch[0];
      for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) {
        let a = select(0u, half, (quadrant & 1u) != 0u);
        let b = select(0u, half, (quadrant & 2u) != 0u);
        var local = vec3u(0u);
        local[axis] = select(0u, size - 1u, side > 0);
        local[(axis + 1u) % 3u] = a;
        local[(axis + 2u) % 3u] = b;
        let outside = vec3i(origin + local) + side * e;
        var neighborCell = header.cell;
        let coefficient = coarseCoefficientScratch[quadrant * 64u];
        if (coefficient > 0.0 && valid(outside)) {
          let neighbor = ownerAt(outside);
          if (liquidOwner(neighbor)) { neighborCell = pressureIndex(neighbor); }
        }
        leafEntries[header.entryStart + face * 4u + quadrant] = LeafEntry(neighborCell, coefficient);
      }
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    header.entryCount = 24u;
    header.diagonal = diagonal;
    header.rhs = flux - boundarySum;
    header.gradient = vec4f(ownerPhiGradient(owner),ownerPhi(owner));
    leafHeaders[row] = header;
  }
}

@compute @workgroup_size(256)
fn iterateLeaves(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  let header = leafHeaders[row];
  var sum = 0.0;
  for (var j = 0u; j < header.entryCount; j += 1u) {
    let entry = leafEntries[header.entryStart + j];
    sum += entry.coefficient * pressureIn[entry.row];
  }
  let coupling = leafBodyCoupling(cellCoord(header.cell), header.size); let pressureRow = select(header.cell, row, rowIndexedPressure); let old = pressureIn[pressureRow]; let effectiveDiagonal = header.diagonal + coupling.y;
  let next = select(0.0, old + (sum - header.rhs - header.diagonal * old - coupling.x) / effectiveDiagonal, effectiveDiagonal > 1e-8);
  pressureOut[pressureRow] = mix(old, next, params.cellRelax.w);
}

// Chebyshev semi-iteration applies an accelerated polynomial to D^-1 A. Each
// row keeps its previous correction and recurrence scalar in the otherwise
// unused header padding, so every iteration remains one row-parallel SpMV.
@compute @workgroup_size(256)
fn iterateChebyshev(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  let header = leafHeaders[row];
  var sum = 0.0;
  for (var j = 0u; j < header.entryCount; j += 1u) {
    let entry = leafEntries[header.entryStart + j];
    sum += entry.coefficient * pressureIn[entry.row];
  }
  let pressureRow = select(header.cell, row, rowIndexedPressure);
  let old = pressureIn[pressureRow];
  let residual = select(0.0, (sum - header.rhs - header.diagonal * old) / header.diagonal, header.diagonal > 1e-8);
  let lower = params.solve.y; let upper = params.solve.z;
  let theta = 0.5 * (upper + lower); let delta = 0.5 * (upper - lower); let sigma = theta / delta;
  let previousSearch = bitcast<f32>(header.pad0); let previousRho = bitcast<f32>(header.pad1);
  var rho = 1.0 / sigma;
  var search = residual / theta;
  if (previousRho > 0.0) {
    rho = 1.0 / (2.0 * sigma - previousRho);
    search = rho * previousRho * previousSearch + (2.0 * rho / delta) * residual;
  }
  pressureOut[pressureRow] = old + search;
  // Only the two recurrence words change in the hot loop. Assign the members
  // directly so the backend does not conservatively write the complete
  // 48-byte header (including immutable topology and the projected gradient)
  // on every polynomial pass.
  leafHeaders[row].pad0 = bitcast<u32>(search);
  leafHeaders[row].pad1 = bitcast<u32>(rho);
}

// Relative-residual feedback for the adaptive Chebyshev budget. Stage one
// folds each row workgroup into the scan-partial words (dead once emitLeaves
// has consumed them); stage two reduces those into the two trailing feedback
// words that encode() stages into a copy-only buffer for async readback.
var<workgroup> residualPartials: array<vec2f, 256>;

@compute @workgroup_size(256)
fn reduceResidualPartials(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let row = compactRowIndex(gid);
  var partial = vec2f(0.0);
  if (row < compaction[0]) {
    let header = leafHeaders[row];
    if (header.diagonal > 1e-8) {
      var sum = 0.0;
      for (var j = 0u; j < header.entryCount; j += 1u) {
        let entry = leafEntries[header.entryStart + j];
        sum += entry.coefficient * pressureIn[entry.row];
      }
      let pressureRow = select(header.cell, row, rowIndexedPressure);
      let residual = sum - header.rhs - header.diagonal * pressureIn[pressureRow];
      partial = vec2f(residual * residual, header.rhs * header.rhs);
    }
  }
  residualPartials[lid] = partial;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) { residualPartials[lid] += residualPartials[lid + stride]; }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let block = wid.x + wid.y * compaction[2];
    compaction[15u + 3u * block] = bitcast<u32>(residualPartials[0].x);
    compaction[16u + 3u * block] = bitcast<u32>(residualPartials[0].y);
  }
}

@compute @workgroup_size(256)
fn reduceResidualTotal(@builtin(local_invocation_index) lid: u32) {
  let blocks = (compaction[0] + 255u) / 256u;
  var sum = vec2f(0.0);
  for (var b = lid; b < blocks; b += 256u) {
    sum += vec2f(bitcast<f32>(compaction[15u + 3u * b]), bitcast<f32>(compaction[16u + 3u * b]));
  }
  residualPartials[lid] = sum;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) { residualPartials[lid] += residualPartials[lid + stride]; }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let base = pressureControlBase() + 6u;
    compaction[base] = bitcast<u32>(residualPartials[0].x);
    compaction[base + 1u] = bitcast<u32>(residualPartials[0].y);
  }
}

fn leafPressureLoad(parity: u32, cell: u32) -> f32 {
  if (parity == 0u) { return pressureIn[cell]; }
  return pressureOut[cell];
}
fn leafPressureStore(parity: u32, cell: u32, value: f32) {
  if (parity == 0u) { pressureOut[cell] = value; } else { pressureIn[cell] = value; }
}

var<workgroup> reduceScalars: array<f32, 256>;
var<workgroup> rowCountShared: u32;
var<workgroup> convergedShared: u32;

// The whole weighted-Jacobi loop in ONE dispatch. Single-workgroup execution
// makes storageBarrier sufficient for ping-pong coherence; the loop break must
// come from workgroupUniformLoad so every barrier stays in uniform control
// flow. Warm-started from the persistent pressure buffer and exits early once
// the residual falls below params.solve.x relative to |b|^2.
@compute @workgroup_size(256)
fn solveLeaves(@builtin(local_invocation_id) lid3: vec3u) {
  let lid = lid3.x;
  if (lid == 0u) { rowCountShared = compaction[0]; convergedShared = 0u; }
  let n = workgroupUniformLoad(&rowCountShared);
  let cap = params.control.y;
  let relaxation = params.cellRelax.w;
  let tolerance2 = params.solve.x;
  var normB = 1e-30;
  var parity = 0u;
  var executed = 0u;
  for (var it = 0u; it < cap; it += 1u) {
    if (workgroupUniformLoad(&convergedShared) != 0u) { break; }
    var rr = 0.0;
    var bb = 0.0;
    for (var row = lid; row < n; row += 256u) {
      let header = leafHeaders[row];
      var sum = 0.0;
      for (var j = 0u; j < header.entryCount; j += 1u) {
        let entry = leafEntries[header.entryStart + j];
        sum += entry.coefficient * leafPressureLoad(parity, entry.row);
      }
      let pressureRow = select(header.cell, row, rowIndexedPressure);
      let previous = leafPressureLoad(parity, pressureRow);
      let rowLive = header.diagonal > 1e-8;
      let next = select(0.0, (sum - header.rhs) / header.diagonal, rowLive);
      leafPressureStore(parity, pressureRow, mix(previous, next, relaxation));
      let residual = select(0.0, sum - header.rhs - header.diagonal * previous, rowLive);
      rr += residual * residual;
      bb += header.rhs * header.rhs;
    }
    storageBarrier();
    if (it == 0u) {
      reduceScalars[lid] = bb;
      for (var stride = 128u; stride > 0u; stride >>= 1u) {
        workgroupBarrier();
        if (lid < stride) { reduceScalars[lid] += reduceScalars[lid + stride]; }
      }
      workgroupBarrier();
      normB = max(reduceScalars[0], 1e-30);
      workgroupBarrier();
    }
    reduceScalars[lid] = rr;
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) { reduceScalars[lid] += reduceScalars[lid + stride]; }
    }
    workgroupBarrier();
    if (lid == 0u && reduceScalars[0] <= tolerance2 * normB) { convergedShared = 1u; }
    parity ^= 1u;
    executed += 1u;
  }
  // Downstream stages always read pressureIn; fold an odd final parity back.
  storageBarrier();
  workgroupBarrier();
  if ((executed & 1u) == 1u) {
    for (var row = lid; row < n; row += 256u) {
      let header = leafHeaders[row];
      let pressureRow = select(header.cell, row, rowIndexedPressure);
      pressureIn[pressureRow] = pressureOut[pressureRow];
    }
  }
}

fn reconstructedAxisGradient(owner: Owner, axis: u32) -> f32 {
  let origin = unpackOrigin(owner.packedOrigin); let size = owner.size;
  let centrePressure = pressureOf(owner); let e = axisVector(axis);
  var gradientSum = 0.0; var weightSum = 0.0;
  for (var sideIndex = 0u; sideIndex < 2u; sideIndex += 1u) {
    let side = select(-1, 1, sideIndex == 1u);
    for (var b = 0u; b < size; b += 1u) { for (var a = 0u; a < size; a += 1u) {
      var local = vec3u(0u); local[axis] = select(0u, size - 1u, side > 0);
      local[(axis + 1u) % 3u] = a; local[(axis + 2u) % 3u] = b;
      let inside = vec3i(origin + local); let outside = inside + side * e;
      if (!valid(outside)) { continue; }
      let neighbor = ownerAt(outside);
      if (neighbor.packedOrigin == owner.packedOrigin) { continue; }
      let solid = faceSolid(inside, outside); let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
      let neighborPressure = select(0.0, pressureOf(neighbor), liquidOwner(neighbor));
      let gradient = f32(side) * (neighborPressure - centrePressure) / max(pressureDistance(owner, neighbor, axis), 1e-7);
      gradientSum += open * gradient; weightSum += open;
    } }
  }
  return select(0.0, gradientSum / weightSum, weightSum > 1e-6);
}

// Coarse-leaf affine reconstruction is surface-area work. Running the size^2
// face traversal in the single invocation at the leaf origin under-fills the
// GPU precisely for the largest leaves, where there are few origins and each
// invocation has thousands of subfaces. The 4^3 workgroup containing an
// aligned size>=4 origin owns that leaf and reduces its boundary cooperatively.
// vec3 components keep the three axis reductions independent while requiring
// only one workgroup reduction and one storage publication per leaf.
var<workgroup> gradientPartials: array<vec3f, 64>;
var<workgroup> gradientWeightPartials: array<vec3f, 64>;
var<workgroup> gradientOwnerOrigin: atomic<u32>;
var<workgroup> gradientOwnerSize: atomic<u32>;
var<workgroup> gradientOwnerEligible: atomic<u32>;

fn coarseGradientContribution(owner: Owner, sample: u32) -> array<vec3f, 2> {
  let origin = unpackOrigin(owner.packedOrigin);
  let faceSamples = owner.size * owner.size;
  let face = sample / faceSamples;
  let axis = face / 2u;
  let side = select(-1, 1, (face & 1u) == 1u);
  let inFace = sample - face * faceSamples;
  let a = inFace % owner.size;
  let b = inFace / owner.size;
  var local = vec3u(0u);
  local[axis] = select(0u, owner.size - 1u, side > 0);
  local[(axis + 1u) % 3u] = a;
  local[(axis + 2u) % 3u] = b;
  let inside = vec3i(origin + local);
  let outside = inside + side * axisVector(axis);
  var weighted = vec3f(0.0);
  var weight = vec3f(0.0);
  if (valid(outside)) {
    let neighbor = ownerAt(outside);
    if (neighbor.packedOrigin != owner.packedOrigin) {
      let solid = faceSolid(inside, outside);
      let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
      let neighborPressure = select(0.0, pressureOf(neighbor), liquidOwner(neighbor));
      let gradient = f32(side) * (neighborPressure - pressureOf(owner))
        / max(pressureDistance(owner, neighbor, axis), 1e-7);
      weighted[axis] = open * gradient;
      weight[axis] = open;
    }
  }
  return array<vec3f, 2>(weighted, weight);
}

fn reconstructedGradient(owner: Owner, axis: u32) -> f32 {
  if (owner.size <= 1u) { return 0.0; }
  if (rowIndexedPressure) {
    let row = pressureIndex(owner);
    if (row >= arrayLength(&leafHeaders)) { return 0.0; }
    return leafHeaders[row].gradient[axis];
  }
  var slot = unpackOrigin(owner.packedOrigin); slot[axis] += 1u;
  return pressureOut[index(slot)];
}

fn storeReconstructedGradient(row: u32, owner: Owner, gradient: vec3f) {
  if (rowIndexedPressure) {
    var header = leafHeaders[row];
    header.gradient = vec4f(gradient, header.gradient.w);
    leafHeaders[row] = header;
    return;
  }
  let origin = unpackOrigin(owner.packedOrigin);
  pressureOut[index(origin + vec3u(1u,0u,0u))] = gradient.x;
  pressureOut[index(origin + vec3u(0u,1u,0u))] = gradient.y;
  pressureOut[index(origin + vec3u(0u,0u,1u))] = gradient.z;
}

@compute @workgroup_size(64)
fn reconstructGradients(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let task = coarseTaskIndex(wid);
  let row = coarseTaskRow(task);
  if (lid == 0u) {
    var packed = 0u; var size = 0u;
    let eligible = row < compaction[0] && coarseTaskTile(task) == 0u;
    if (eligible) { let header = leafHeaders[row]; packed = ownerAtIndex(header.cell).packedOrigin; size = header.size; }
    atomicStore(&gradientOwnerOrigin, packed);
    atomicStore(&gradientOwnerSize, size);
    atomicStore(&gradientOwnerEligible, select(0u, 1u, eligible));
  }
  workgroupBarrier();
  let coarseOwner = Owner(workgroupUniformLoad(&gradientOwnerOrigin), workgroupUniformLoad(&gradientOwnerSize));
  if (workgroupUniformLoad(&gradientOwnerEligible) == 0u || coarseOwner.size <= 1u) { return; }
  let workgroupOrigin = unpackOrigin(coarseOwner.packedOrigin);
  if (coarseOwner.size >= 4u) {
    var weighted = vec3f(0.0);
    var weight = vec3f(0.0);
    let sampleCount = 6u * coarseOwner.size * coarseOwner.size;
    for (var sample = lid; sample < sampleCount; sample += 64u) {
      let contribution = coarseGradientContribution(coarseOwner, sample);
      weighted += contribution[0];
      weight += contribution[1];
    }
    gradientPartials[lid] = weighted;
    gradientWeightPartials[lid] = weight;
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) {
        gradientPartials[lid] += gradientPartials[lid + stride];
        gradientWeightPartials[lid] += gradientWeightPartials[lid + stride];
      }
    }
    workgroupBarrier();
    if (lid == 0u) {
      let gradient = vec3f(
        select(0.0, gradientPartials[0].x / gradientWeightPartials[0].x, gradientWeightPartials[0].x > 1e-6),
        select(0.0, gradientPartials[0].y / gradientWeightPartials[0].y, gradientWeightPartials[0].y > 1e-6),
        select(0.0, gradientPartials[0].z / gradientWeightPartials[0].z, gradientWeightPartials[0].z > 1e-6)
      );
      storeReconstructedGradient(row, coarseOwner, gradient);
    }
    return;
  }
  if (lid == 0u) {
    storeReconstructedGradient(row, coarseOwner, vec3f(
      reconstructedAxisGradient(coarseOwner, 0u),
      reconstructedAxisGradient(coarseOwner, 1u),
      reconstructedAxisGradient(coarseOwner, 2u)
    ));
  }
}

@compute @workgroup_size(256)
fn reconstructSmallGradients(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  let header = leafHeaders[row];
  // Chebyshev reuses pad0/pad1 as spectral state after assembly. Classify
  // projection work from the immutable leaf size, never those scratch words.
  if (header.size >= 8u || header.size <= 1u) { return; }
  let owner = ownerAtIndex(header.cell);
  storeReconstructedGradient(row, owner, vec3f(
    reconstructedAxisGradient(owner, 0u),
    reconstructedAxisGradient(owner, 1u),
    reconstructedAxisGradient(owner, 2u)
  ));
}

struct ProjectionOwners {
  center: Owner,
  x: Owner,
  y: Owner,
  z: Owner,
  live: vec4u,
}
struct ProjectionNeighborhood {
  center: Owner,
  x: Owner,
  y: Owner,
  z: Owner,
  ownerPhi: vec4f,
  centerSolid: SolidCell,
  xSolid: SolidCell,
  ySolid: SolidCell,
  zSolid: SolidCell,
}

// Projection used to call ownerPhi independently from liquidOwner and again
// from pressureDistance for all three axes. ownerPhi is an eight-sample
// trilinear reconstruction, so one output texel could issue about one hundred
// level-set reads. Load the four adjacent owners/cells once and reconstruct phi
// only on wet/dry owner boundaries. Same-phase leaf interiors need no phi
// samples at all instead of repeating the same eight-sample lookup twelve
// times; the exact phi=0 distance is retained wherever phases differ.
fn projectionOwners(id: vec3i) -> ProjectionOwners {
  let d = vec3i(dims());
  let center = ownerAt(id);
  var x = center; var y = center; var z = center;
  if (id.x + 1 < d.x) { x = ownerAt(id + vec3i(1,0,0)); }
  if (id.y + 1 < d.y) { y = ownerAt(id + vec3i(0,1,0)); }
  if (id.z + 1 < d.z) { z = ownerAt(id + vec3i(0,0,1)); }
  let centerLive = select(0u, 1u, liveOwner(center));
  var xLive = centerLive; var yLive = centerLive; var zLive = centerLive;
  if (x.packedOrigin != center.packedOrigin) { xLive = select(0u, 1u, liveOwner(x)); }
  if (y.packedOrigin == center.packedOrigin) { yLive = centerLive; }
  else if (y.packedOrigin == x.packedOrigin) { yLive = xLive; }
  else { yLive = select(0u, 1u, liveOwner(y)); }
  if (z.packedOrigin == center.packedOrigin) { zLive = centerLive; }
  else if (z.packedOrigin == x.packedOrigin) { zLive = xLive; }
  else if (z.packedOrigin == y.packedOrigin) { zLive = yLive; }
  else { zLive = select(0u, 1u, liveOwner(z)); }
  return ProjectionOwners(center, x, y, z, vec4u(centerLive, xLive, yLive, zLive));
}

fn projectionNeighborhood(id: vec3i, loaded: ProjectionOwners) -> ProjectionNeighborhood {
  let d = vec3i(dims());
  let center = loaded.center; let x = loaded.x; let y = loaded.y; let z = loaded.z;
  var centerPhi = select(1.0, -1.0, loaded.live.x != 0u);
  var xPhi = select(1.0, -1.0, loaded.live.y != 0u);
  var yPhi = select(1.0, -1.0, loaded.live.z != 0u);
  var zPhi = select(1.0, -1.0, loaded.live.w != 0u);
  var centerPhiSampled = false; var xPhiSampled = false; var yPhiSampled = false;
  let centerSolid = solidAt(id);
  var xSolid = centerSolid; var ySolid = centerSolid; var zSolid = centerSolid;
  if (id.x + 1 < d.x) {
    let q = id + vec3i(1,0,0); xSolid = solidAt(q);
    if (loaded.live.y != loaded.live.x) {
      centerPhi = ownerPhi(center); centerPhiSampled = true;
      xPhi = ownerPhi(x); xPhiSampled = true;
    }
  }
  if (id.y + 1 < d.y) {
    let q = id + vec3i(0,1,0); ySolid = solidAt(q);
    if (loaded.live.z != loaded.live.x) {
      if (!centerPhiSampled) { centerPhi = ownerPhi(center); centerPhiSampled = true; }
      if (y.packedOrigin == x.packedOrigin && xPhiSampled) { yPhi = xPhi; }
      else { yPhi = ownerPhi(y); }
      yPhiSampled = true;
    }
  }
  if (id.z + 1 < d.z) {
    let q = id + vec3i(0,0,1); zSolid = solidAt(q);
    if (loaded.live.w != loaded.live.x) {
      if (!centerPhiSampled) { centerPhi = ownerPhi(center); }
      if (z.packedOrigin == x.packedOrigin && xPhiSampled) { zPhi = xPhi; }
      else if (z.packedOrigin == y.packedOrigin && yPhiSampled) { zPhi = yPhi; }
      else { zPhi = ownerPhi(z); }
    }
  }
  return ProjectionNeighborhood(center, x, y, z, vec4f(centerPhi, xPhi, yPhi, zPhi), centerSolid, xSolid, ySolid, zSolid);
}

fn faceSolidFromCells(a: SolidCell, b: SolidCell) -> SolidCell {
  if (a.fraction >= b.fraction) { return a; }
  return b;
}

fn projectedComponentCached(id: vec3i, axis: u32, input: f32, left: Owner, right: Owner, leftPhi: f32, rightPhi: f32, solid: SolidCell) -> f32 {
  let d = vec3i(dims());
  if (id[axis] == d[axis] - 1) { return 0.0; }
  let leftWet = leftPhi < 0.0; let rightWet = rightPhi < 0.0;
  var fluid = input;
  if (leftWet || rightWet) {
    if (left.packedOrigin != right.packedOrigin) {
      let airPressure = hydrostaticAirPressure(left, right, leftPhi, rightPhi);
      let p0 = select(airPressure, pressureOf(left), leftWet); let p1 = select(airPressure, pressureOf(right), rightWet);
      let distance = pressureDistanceFromPhi(left, right, axis, leftPhi, rightPhi);
      fluid -= (p1 - p0) / max(distance, 1e-7);
    } else if (leftWet && rightWet) {
      fluid -= reconstructedGradient(left, axis);
    }
  } else { fluid = 0.0; }
  let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
  let constrained = open * fluid + solid.fraction * component(solidVelocity(solid, faceWorld(id, axis)), axis);
  return clamp(constrained, -50.0, 50.0);
}

fn liveOwner(owner: Owner) -> bool { return frontierAlive(index(unpackOrigin(owner.packedOrigin))); }
fn projectionControllerCached(id: vec3i, n: ProjectionOwners) -> u32 {
  let d = vec3i(dims());
  var best = select(0xffffffffu, n.center.packedOrigin, n.live.x != 0u);
  if (id.x + 1 < d.x && n.x.packedOrigin != n.center.packedOrigin && n.live.y != 0u) { best = min(best, n.x.packedOrigin); }
  if (id.y + 1 < d.y && n.y.packedOrigin != n.center.packedOrigin && n.y.packedOrigin != n.x.packedOrigin && n.live.z != 0u) { best = min(best, n.y.packedOrigin); }
  if (id.z + 1 < d.z && n.z.packedOrigin != n.center.packedOrigin && n.z.packedOrigin != n.x.packedOrigin && n.z.packedOrigin != n.y.packedOrigin && n.live.w != 0u) { best = min(best, n.z.packedOrigin); }
  return best;
}

fn projectedVelocityCached(id: vec3i, input: vec3f, n: ProjectionNeighborhood) -> vec3f {
  return vec3f(
    projectedComponentCached(id, 0u, input.x, n.center, n.x, n.ownerPhi.x, n.ownerPhi.y, faceSolidFromCells(n.centerSolid, n.xSolid)),
    projectedComponentCached(id, 1u, input.y, n.center, n.y, n.ownerPhi.x, n.ownerPhi.z, faceSolidFromCells(n.centerSolid, n.ySolid)),
    projectedComponentCached(id, 2u, input.z, n.center, n.z, n.ownerPhi.x, n.ownerPhi.w, faceSolidFromCells(n.centerSolid, n.zSolid))
  );
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let id = vec3i(gid); let input = textureLoad(velocityIn, id, 0).xyz;
  let loaded = projectionOwners(id);
  let neighborhood = projectionNeighborhood(id, loaded);
  let v = projectedVelocityCached(id, input, neighborhood);
  let known = select(0.0, 1.0, liquidCell(id) || liquidCell(id+vec3i(1,0,0)) || liquidCell(id+vec3i(0,1,0)) || liquidCell(id+vec3i(0,0,1)));
  textureStore(velocityOut, id, vec4f(v, known));
}

fn projectLeafCell(owner: Owner, id: vec3i) {
  if (!valid(id)) { return; }
  let loaded = projectionOwners(id);
  if (projectionControllerCached(id, loaded) != owner.packedOrigin) { return; }
  let neighborhood = projectionNeighborhood(id, loaded);
  let input = textureLoad(velocityIn, id, 0).xyz;
  let v = projectedVelocityCached(id, input, neighborhood);
  textureStore(velocityOut, id, vec4f(v, f32(frontierGeneration())));
}

// Small leaves retain row-parallel occupancy: one invocation owns a row and
// performs at most 5^3 compatibility writes. Overflow coarse rows use this
// path too, so the cooperative-list capacity never affects correctness.
@compute @workgroup_size(256)
fn projectSmallLeaves(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  let header = leafHeaders[row];
  if (header.size >= 8u) { return; }
  let owner = ownerAtIndex(header.cell);
  let width = owner.size + 1u;
  let origin = vec3i(unpackOrigin(owner.packedOrigin));
  for (var sample = 0u; sample < width * width * width; sample += 1u) {
    let local = vec3i(i32(sample % width), i32((sample / width) % width), i32(sample / (width * width)));
    projectLeafCell(owner, origin + local - vec3i(1));
  }
}

// Compact authority assigns one cooperative workgroup per coarse pressure row,
// so persistent task storage follows the pressure bound rather than the box's
// 8^3 tile count. Dense compatibility retains the historical per-tile jobs.
@compute @workgroup_size(256)
fn projectLeaves(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let task = coarseTaskIndex(wid);
  let row = coarseTaskRow(task);
  let tile = coarseTaskTile(task);
  let header = leafHeaders[row];
  let owner = ownerAtIndex(header.cell);
  if (rowIndexedPressure) {
    let width = owner.size + 1u;
    let origin = vec3i(unpackOrigin(owner.packedOrigin)) - vec3i(1);
    for (var sample = lid; sample < width * width * width; sample += 256u) {
      let local = vec3i(i32(sample % width), i32((sample / width) % width), i32(sample / (width * width)));
      projectLeafCell(owner, origin + local);
    }
    return;
  }
  let tiles = owner.size / 8u;
  let tileCoord = vec3u(tile % tiles, (tile / tiles) % tiles, tile / (tiles * tiles));
  let halo = vec3u(select(0u, 1u, tileCoord.x == 0u), select(0u, 1u, tileCoord.y == 0u), select(0u, 1u, tileCoord.z == 0u));
  let extent = vec3u(8u) + halo;
  let samples = extent.x * extent.y * extent.z;
  let tileOrigin = vec3i(unpackOrigin(owner.packedOrigin) + tileCoord * 8u) - vec3i(halo);
  for (var sample = lid; sample < samples; sample += 256u) {
    let local = vec3i(i32(sample % extent.x), i32((sample / extent.x) % extent.y), i32(sample / (extent.x * extent.y)));
    projectLeafCell(owner, tileOrigin + local);
  }
}

// An overflowing compact publication never indexes a partial row arena.  The
// scan emits this dense indirect dispatch only on overflow; it preserves the
// predicted velocity for one frame and stamps it as known for extrapolation.
@compute @workgroup_size(4,4,4)
fn passThroughPressureOverflow(@builtin(global_invocation_id) gid: vec3u) {
  if (!pressureOverflowed() || any(gid >= dims())) { return; }
  let value = textureLoad(velocityIn, vec3i(gid), 0);
  textureStore(velocityOut, vec3i(gid), vec4f(value.xyz, f32(frontierGeneration())));
}

// The sparse projector deliberately leaves unknown air texels untouched. The
// projector stamps known cells with the current frontier generation. The first
// extrapolation sweep ignores stale alpha from prior frames and expands the
// band by one cell. Dense A/B materializes the whole compatibility texture;
// sparse execution relies on explicit retirement clears for omitted bricks.
fn extrapolateSeedAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let id = vec3i(gid);
  let center = textureLoad(velocityIn, id, 0);
  let generation = frontierGeneration();
  if (u32(round(center.w)) == generation) { textureStore(velocityOut, id, vec4f(center.xyz, 1.0)); return; }
  let offsets = array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var sum = vec3f(0.0); var weight = 0.0;
  for (var n = 0u; n < 6u; n += 1u) {
    let p = id + offsets[n];
    if (valid(p)) { let q = textureLoad(velocityIn, p, 0); if (u32(round(q.w)) == generation) { sum += q.xyz; weight += 1.0; } }
  }
  textureStore(velocityOut, id, select(vec4f(0.0), vec4f(sum / max(weight,1.0),1.0), weight > 0.0));
}

fn extrapolateAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let id = vec3i(gid); let center = textureLoad(velocityIn, id, 0);
  if (center.w > 0.5) { textureStore(velocityOut, id, center); return; }
  var sum = vec3f(0.0); var weight = 0.0;
  let offsets = array<vec3i,6>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1));
  for (var n = 0u; n < 6u; n += 1u) { let p = id + offsets[n]; if (valid(p)) { let q = textureLoad(velocityIn,p,0); if (q.w > 0.5) { sum += q.xyz; weight += 1.0; } } }
  textureStore(velocityOut, id, select(vec4f(0.0), vec4f(sum / max(weight,1.0),1.0), weight > 0.0));
}

@compute @workgroup_size(4,4,4)
fn extrapolateSeed(@builtin(global_invocation_id) gid: vec3u) { extrapolateSeedAt(gid); }

@compute @workgroup_size(4,4,4)
fn extrapolate(@builtin(global_invocation_id) gid: vec3u) { extrapolateAt(gid); }

@compute @workgroup_size(64)
fn extrapolateSeedSparse(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  extrapolateSeedAt(bulkResidentCell(wid, lid));
}

@compute @workgroup_size(64)
fn extrapolateSparse(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  extrapolateAt(bulkResidentCell(wid, lid));
}

@compute @workgroup_size(64)
fn copyExtrapolatedSparse(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let gid = bulkResidentCell(wid, lid);
  if (any(gid >= dims())) { return; }
  textureStore(velocityOut, vec3i(gid), textureLoad(velocityIn, vec3i(gid), 0));
}
`;

/** Pressure part of the octree VOS coupling, accumulated into the shared
 * fixed-point body exchange alongside the post-projection tangential blend. */
export const octreePressureCouplingShader = /* wgsl */ `
override rowIndexedPressure: bool = true;
struct Owner { packedOrigin: u32, size: u32 }
struct SolidCell { fraction: f32, owner: i32 }
struct RigidBody { positionShape: vec4f, dimensions: vec4f, orientation: vec4f, linearVelocity: vec4f, angularVelocity: vec4f, inverseMassInertia: vec4f, angularMomentumRestitution: vec4f, material: vec4f }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowDirectionLength: vec4f, physical: vec4f, pressureCapacity: vec4u }
@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> owners: array<u32>;
@group(0) @binding(2) var<storage, read> solidCells: array<SolidCell>;
@group(0) @binding(3) var<storage, read> rigidBodies: array<RigidBody, 12>;
@group(0) @binding(4) var<storage, read_write> rigidExchange: array<atomic<i32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var levelSetIn: texture_3d<f32>;
@group(0) @binding(7) var<storage, read> frontier: array<u32>;
@group(0) @binding(8) var<storage, read> compaction: array<u32>;
fn dims() -> vec3u { return params.dimsMax.xyz; }
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn packOrigin(p: vec3u) -> u32 { return index(p); }
fn unpackOrigin(word: u32) -> vec3u { let plane=params.dimsMax.x*params.dimsMax.y;return vec3u(word%params.dimsMax.x,(word/params.dimsMax.x)%params.dimsMax.y,word/plane); }
fn decodeOwner(word: u32, cell: vec3u) -> Owner {
  if ((word & 0x80000000u) != 0u) { return Owner(packOrigin(cell), 1u); }
  let exponent = word & 7u;
  if (exponent == 0u || exponent > 5u) { return Owner(packOrigin(cell), 1u); }
  let origin = (cell >> vec3u(exponent)) << vec3u(exponent);
  return Owner(packOrigin(origin), 1u << exponent);
}
fn canonicalOwner(cell: vec3u) -> Owner { var size=min(params.dimsMax.w,8u);var origin=(cell/vec3u(size))*vec3u(size);loop{if(all(origin+vec3u(size)<=dims())||size==1u){break;}size>>=1u;origin=(cell/vec3u(size))*vec3u(size);}return Owner(packOrigin(origin),size); }
fn ownerPageEncoded(logical:u32)->u32{let freeOffset=owners[5];if(freeOffset<=16u||((freeOffset-16u)&1u)!=0u){return 0u;}let hashCapacity=(freeOffset-16u)/2u;let key=logical+1u;var slot=(logical*0x9e3779b1u)%hashCapacity;for(var probe=0u;probe<hashCapacity;probe+=1u){let observed=owners[16u+slot];if(observed==key){return owners[16u+hashCapacity+slot];}if(observed==0u){break;}slot=select(slot+1u,0u,slot+1u==hashCapacity);}return 0u;}
fn ownerAt(p: vec3i) -> Owner {
  let cell=vec3u(p);if(arrayLength(&owners)<=15u||owners[15]!=0x4f574e52u){return decodeOwner(owners[index(cell)],cell);}
  let bd=(dims()+vec3u(7u))/8u;let b=cell/8u;let logical=b.x+b.y*bd.x+b.z*bd.x*bd.y;let encoded=ownerPageEncoded(logical);let capacity=owners[3];
  if(encoded==0u||encoded==0xffffffffu||encoded>capacity){return canonicalOwner(cell);}let local=cell%vec3u(8u);let word=owners[owners[6]+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u];
  if(word==0u){return canonicalOwner(cell);}return decodeOwner(word,cell);
}
fn phi(p: vec3i) -> f32 { return textureLoad(levelSetIn, p, 0).x; }
fn ownerPhi(owner: Owner) -> f32 {
  let centre = vec3f(unpackOrigin(owner.packedOrigin)) + vec3f(0.5 * f32(owner.size - 1u));
  let a = vec3u(floor(centre)); let b = min(a + vec3u(1u), dims() - vec3u(1u)); let t = fract(centre);
  let p000 = phi(vec3i(a)); let p100 = phi(vec3i(vec3u(b.x,a.y,a.z)));
  let p010 = phi(vec3i(vec3u(a.x,b.y,a.z))); let p110 = phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001 = phi(vec3i(vec3u(a.x,a.y,b.z))); let p101 = phi(vec3i(vec3u(b.x,a.y,b.z)));
  let p011 = phi(vec3i(vec3u(a.x,b.y,b.z))); let p111 = phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn liquidOwner(owner: Owner) -> bool { return ownerPhi(owner) < 0.0; }
fn frontierMapBase()->u32{return 4u+2u*params.pressureCapacity.x;}
fn frontierHash(cell:u32)->u32{var h=cell*747796405u+2891336453u;h^=h>>16u;h*=2246822519u;h^=h>>13u;return h;}
fn compactFrontierWord(cell:u32)->u32{let cap=(arrayLength(&frontier)-frontierMapBase())/2u;var slot=frontierHash(cell)&(cap-1u);let key=cell+1u;for(var probe=0u;probe<32u;probe+=1u){let stored=frontier[frontierMapBase()+2u*slot];if(stored==0u){break;}if(stored==key){return frontier[frontierMapBase()+2u*slot+1u];}slot=(slot+1u)&(cap-1u);}return 0u;}
fn frontierWord(cell:u32)->u32{if(rowIndexedPressure){return compactFrontierWord(cell);}return frontier[frontierMapBase()+cell];}
fn pressureOf(owner: Owner) -> f32 {
  let cell = index(unpackOrigin(owner.packedOrigin));
  let word = frontierWord(cell);
  let row = select(0xffffffffu, word - 2u, word >= 2u);
  let slot = select(cell, row, rowIndexedPressure);
  if (slot >= arrayLength(&pressure)) { return 0.0; }
  return pressure[slot];
}
fn axisVector(axis: u32) -> vec3i { return select(select(vec3i(0,0,1), vec3i(0,1,0), axis == 1u), vec3i(1,0,0), axis == 0u); }
fn faceArea(axis: u32) -> f32 { let h = params.cellRelax.xyz; return select(select(h.x * h.y, h.x * h.z, axis == 1u), h.y * h.z, axis == 0u); }
fn worldCell(p: vec3i) -> vec3f { let h = params.cellRelax.xyz; return vec3f(-0.5 * params.container.x + (f32(p.x) + 0.5) * h.x, (f32(p.y) + 0.5) * h.y, -0.5 * params.container.z + (f32(p.z) + 0.5) * h.z); }
fn faceWorld(p: vec3i, axis: u32) -> vec3f { var result = worldCell(p); result[axis] += 0.5 * params.cellRelax[axis]; return result; }
fn faceSolid(a: vec3i, b: vec3i) -> SolidCell { let sa = solidCells[index(vec3u(a))]; let sb = solidCells[index(vec3u(b))]; if (sa.fraction >= sb.fraction) { return sa; } return sb; }
fn addImpulse(bodyIndex: u32, impulse: vec3f, world: vec3f) {
  let body = rigidBodies[bodyIndex]; let torque = cross(world - body.positionShape.xyz, impulse); let base = bodyIndex * 12u;
  atomicAdd(&rigidExchange[base], i32(round(impulse.x * 1e6))); atomicAdd(&rigidExchange[base + 1u], i32(round(impulse.y * 1e6))); atomicAdd(&rigidExchange[base + 2u], i32(round(impulse.z * 1e6)));
  atomicAdd(&rigidExchange[base + 3u], i32(round(torque.x * 1e6))); atomicAdd(&rigidExchange[base + 4u], i32(round(torque.y * 1e6))); atomicAdd(&rigidExchange[base + 5u], i32(round(torque.z * 1e6)));
}
@compute @workgroup_size(4,4,4)
fn accumulatePressureImpulse(@builtin(global_invocation_id) gid: vec3u) {
  if (compaction[arrayLength(&compaction) - 8u] != 0u || any(gid >= dims())) { return; }
  let p = vec3i(gid);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let q = p + axisVector(axis); if (!valid(q)) { continue; }
    let solid = faceSolid(p, q); if (solid.owner < 0 || solid.fraction <= 0.0 || u32(solid.owner) >= params.control.w) { continue; }
    let left = ownerAt(p); let right = ownerAt(q); let leftWet = liquidOwner(left); let rightWet = liquidOwner(right);
    if (!leftWet && !rightWet) { continue; }
    let p0 = select(0.0, pressureOf(left), leftWet); let p1 = select(0.0, pressureOf(right), rightWet);
    let scalar = params.physical.x * faceArea(axis) * solid.fraction * (p0 - p1);
    var impulse = vec3f(0.0); impulse[axis] = scalar;
    addImpulse(u32(solid.owner), impulse, faceWorld(p, axis));
  }
}
`;

/** GPU-only adapter from packed owner authority to on-demand scientific overlays. */
export const octreeDiagnosticShader = /* wgsl */ `
override rowIndexedPressure: bool = true;
struct Owner { packedOrigin: u32, size: u32 }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container:vec4f, inflowPositionRadius:vec4f, inflowDirectionLength:vec4f, physical:vec4f, pressureCapacity:vec4u }
@group(0) @binding(0) var<storage, read> owners: array<u32>;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(2) var velocity: texture_3d<f32>;
@group(0) @binding(3) var levelSetIn: texture_3d<f32>;
@group(0) @binding(4) var topologyOut: texture_storage_3d<rg32uint, write>;
@group(0) @binding(5) var pressureSamplesOut: texture_storage_3d<rgba32uint, write>;
@group(0) @binding(6) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var divergenceOut: texture_storage_3d<r32float, write>;
@group(0) @binding(8) var<uniform> params: Params;
@group(0) @binding(9) var<storage, read> gradients: array<f32>;
@group(0) @binding(10) var velocityBeforeProjection: texture_3d<f32>;
@group(0) @binding(11) var<storage, read> frontier: array<u32>;
fn dims() -> vec3u { return params.dimsMax.xyz; }
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn packOrigin(p: vec3u) -> u32 { return index(p); }
fn unpackOrigin(word: u32) -> vec3u { let plane=params.dimsMax.x*params.dimsMax.y;return vec3u(word%params.dimsMax.x,(word/params.dimsMax.x)%params.dimsMax.y,word/plane); }
fn decodeOwner(word: u32, cell: vec3u) -> Owner {
  if ((word & 0x80000000u) != 0u) { return Owner(packOrigin(cell), 1u); }
  let exponent = word & 7u;
  if (exponent == 0u || exponent > 5u) { return Owner(packOrigin(cell), 1u); }
  let origin = (cell >> vec3u(exponent)) << vec3u(exponent);
  return Owner(packOrigin(origin), 1u << exponent);
}
fn canonicalOwner(cell: vec3u) -> Owner { var size=min(params.dimsMax.w,8u);var origin=(cell/vec3u(size))*vec3u(size);loop{if(all(origin+vec3u(size)<=dims())||size==1u){break;}size>>=1u;origin=(cell/vec3u(size))*vec3u(size);}return Owner(packOrigin(origin),size); }
fn ownerPageEncoded(logical:u32)->u32{let freeOffset=owners[5];if(freeOffset<=16u||((freeOffset-16u)&1u)!=0u){return 0u;}let hashCapacity=(freeOffset-16u)/2u;let key=logical+1u;var slot=(logical*0x9e3779b1u)%hashCapacity;for(var probe=0u;probe<hashCapacity;probe+=1u){let observed=owners[16u+slot];if(observed==key){return owners[16u+hashCapacity+slot];}if(observed==0u){break;}slot=select(slot+1u,0u,slot+1u==hashCapacity);}return 0u;}
fn ownerAt(cell: vec3u) -> Owner {
  if(arrayLength(&owners)<=15u||owners[15]!=0x4f574e52u){return decodeOwner(owners[index(cell)],cell);}
  let bd=(dims()+vec3u(7u))/8u;let b=cell/8u;let logical=b.x+b.y*bd.x+b.z*bd.x*bd.y;let encoded=ownerPageEncoded(logical);let capacity=owners[3];
  if(encoded==0u||encoded==0xffffffffu||encoded>capacity){return canonicalOwner(cell);}let local=cell%vec3u(8u);let word=owners[owners[6]+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u];
  if(word==0u){return canonicalOwner(cell);}return decodeOwner(word,cell);
}
fn phi(p: vec3i) -> f32 { return textureLoad(levelSetIn, p, 0).x; }
fn liquidCell(p: vec3i) -> bool { return valid(p) && phi(p) < 0.0; }
fn ownerPhi(owner: Owner) -> f32 {
  let centre = vec3f(unpackOrigin(owner.packedOrigin)) + vec3f(0.5 * f32(owner.size - 1u));
  let a = vec3u(floor(centre)); let b = min(a + vec3u(1u), dims() - vec3u(1u)); let t = fract(centre);
  let p000 = phi(vec3i(a)); let p100 = phi(vec3i(vec3u(b.x,a.y,a.z)));
  let p010 = phi(vec3i(vec3u(a.x,b.y,a.z))); let p110 = phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001 = phi(vec3i(vec3u(a.x,a.y,b.z))); let p101 = phi(vec3i(vec3u(b.x,a.y,b.z)));
  let p011 = phi(vec3i(vec3u(a.x,b.y,b.z))); let p111 = phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn liquidOwner(owner: Owner) -> bool {
  return ownerPhi(owner) < 0.0;
}
fn frontierMapBase()->u32{return 4u+2u*params.pressureCapacity.x;}
fn frontierHash(cell:u32)->u32{var h=cell*747796405u+2891336453u;h^=h>>16u;h*=2246822519u;h^=h>>13u;return h;}
fn compactFrontierWord(cell:u32)->u32{let cap=(arrayLength(&frontier)-frontierMapBase())/2u;var slot=frontierHash(cell)&(cap-1u);let key=cell+1u;for(var probe=0u;probe<32u;probe+=1u){let stored=frontier[frontierMapBase()+2u*slot];if(stored==0u){break;}if(stored==key){return frontier[frontierMapBase()+2u*slot+1u];}slot=(slot+1u)&(cap-1u);}return 0u;}
fn pressureRow(owner: Owner) -> u32 {
  let cell = index(unpackOrigin(owner.packedOrigin));
  if (!rowIndexedPressure) { return cell; }
  let word = compactFrontierWord(cell);
  return select(0xffffffffu, word - 2u, word >= 2u);
}
fn leafGradient(owner: Owner) -> vec3f {
  if (owner.size <= 1u) { return vec3f(0.0); }
  let origin = unpackOrigin(owner.packedOrigin);
  if (rowIndexedPressure) {
    let row = pressureRow(owner);
    if (row == 0xffffffffu) { return vec3f(0.0); }
    let base = row * 12u + 8u;
    return vec3f(gradients[base], gradients[base + 1u], gradients[base + 2u]);
  }
  return vec3f(gradients[index(origin + vec3u(1u,0u,0u))], gradients[index(origin + vec3u(0u,1u,0u))], gradients[index(origin + vec3u(0u,0u,1u))]);
}
fn velocityAt(p: vec3i) -> vec3f { if (!valid(p)) { return vec3f(0.0); } return textureLoad(velocity, p, 0).xyz; }
@compute @workgroup_size(4,4,4)
fn materializeOctreeFields(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(gid); let origin = unpackOrigin(owner.packedOrigin);
  let horizontal = origin.x | (origin.z << 10u) | (owner.size << 20u);
  let vertical = origin.y | ((origin.y + owner.size) << 10u);
  textureStore(topologyOut, vec3i(gid), vec4u(horizontal, vertical, 0u, 0u));
  let wet = liquidOwner(owner); let invalid = 0xffffffffu; let row = pressureRow(owner);
  let q = vec3i(gid);
  let pressureUpdate = length(velocityAt(q) - textureLoad(velocityBeforeProjection, q, 0).xyz);
  // The overlay needs only x/z/w for legacy ownership. Publish the compact
  // row in x so hierarchy, page, and face audits reuse this existing pass;
  // y carries a bitcast live scalar for Projection Δu.
  textureStore(pressureSamplesOut, q, select(vec4u(invalid), vec4u(row, bitcast<u32>(pressureUpdate), vertical, horizontal), wet));
  let centre = vec3f(origin) + vec3f(0.5 * f32(owner.size - 1u));
  let offset = (vec3f(gid) - centre) * params.cellRelax.xyz;
  var centrePressure = 0.0;
  if (row < arrayLength(&pressure)) { centrePressure = pressure[row]; }
  let mappedPressure = centrePressure + dot(leafGradient(owner), offset);
  textureStore(pressureOut, vec3i(gid), vec4f(select(0.0, mappedPressure, wet)));
  let h = params.cellRelax.xyz;
  let divergence = (velocityAt(q).x - velocityAt(q-vec3i(1,0,0)).x) / h.x
                 + (velocityAt(q).y - velocityAt(q-vec3i(0,1,0)).y) / h.y
                 + (velocityAt(q).z - velocityAt(q-vec3i(0,0,1)).z) / h.z;
  textureStore(divergenceOut, q, vec4f(select(0.0, divergence, wet)));
}
`;
