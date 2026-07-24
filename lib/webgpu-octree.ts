import type { SceneDescription } from "./model";
import { OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS } from "./octree-face-fragments";
import { WebGPUOctreeFineSeedAdapter } from "./webgpu-octree-fine-seed-adapter";
import { WebGPUOctreeSimulationOwnerPages } from "./webgpu-octree-owner-pages";
import { PassBroker } from "./webgpu-pass-broker";
import { planOctreeSurfaceStateAllocation } from "./octree-surface-allocation";
import { planOctreeAnalyticBootstrapBounds } from "./octree-analytic-bootstrap";
import { WebGPUOctreeAnalyticBootstrapWorklist } from "./webgpu-octree-analytic-bootstrap";
import { combineInitialBrickWet, damBreakFractions, initialFluidBrickContainsCell } from "./initial-fluid";
import { signedDistanceFromVolume } from "./quadtree-tall-cell-grid";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import { WebGPUQuadtreeSurfaceState, type SurfaceInflowState } from "./webgpu-quadtree-builder";
import { OctreeSparseBrickWorld } from "./webgpu-octree-sparse-bricks";
import { CompactOctreeVoxelInspection } from "./webgpu-octree-voxel-inspection";
import {
  FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES,
  GPUFluidBrickResidency,
  planFineSeedCandidateResidencyPools,
} from "./webgpu-fluid-brick-residency";
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
  OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES,
  OCTREE_POWER_FACE_QUADRATURE_BYTES,
  WebGPUOctreePowerFaces,
  octreePowerClosedBoundaryMask,
  planOctreePowerFaces,
} from "./webgpu-octree-power-faces";
import { WebGPUOctreePowerOperator, planOctreePowerGPUOperator } from "./webgpu-octree-power-operator";
import { WebGPUOctreePowerFaceSeed } from "./webgpu-octree-power-face-seed";
import { WebGPUOctreePowerFaceAdvection } from "./webgpu-octree-power-face-advection";
import { WebGPUOctreePowerSolidFaces } from "./webgpu-octree-power-solid-faces";
import { WebGPUOctreeSolidVertexSdf } from "./webgpu-octree-solid-vertex-sdf";
import { WebGPUOctreePowerVelocity } from "./webgpu-octree-power-velocity";
import {
  planOctreePowerVelocityChunkCapacity,
  WebGPUOctreePowerVelocityPrepass,
} from "./webgpu-octree-power-velocity-prepass";
import {
  WebGPUOctreeMGPCG,
  type OctreeFirstOrderSPDVCycle,
} from "./webgpu-octree-mgpcg";
import { WebGPUOctreeSPGridVCycle } from "./webgpu-octree-spgrid-vcycle";
import {
  refreshOctreePowerGalerkinOperators,
  type OctreePowerGalerkinHierarchy,
} from "./octree-power-galerkin";
import { WebGPUOctreePowerGalerkin } from "./webgpu-octree-power-galerkin";
import {
  OCTREE_FACE_BAND_ENCODE_PHASES,
  unpackOctreeFaceBandPointFieldControl,
  unpackOctreeFaceBandTransientPowerControl,
  WebGPUOctreeFaceClosestPointExtension,
  type OctreeFaceBandEncodePhase,
} from "./webgpu-octree-face-closest-point";
import { WebGPUOctreeCoarseLevelSet } from "./webgpu-octree-coarse-levelset";
import { WebGPUOctreePowerCoarseLevelSet } from "./webgpu-octree-power-coarse-levelset";
import { WebGPUFineToCoarseLevelSet } from "./webgpu-octree-fine-to-coarse-levelset";
import { planFineLevelSetBricks } from "./octree-fine-levelset-bricks";
import {
  WebGPUFineLevelSetBricks,
  type WebGPUFineLevelSetBrickSource,
} from "./webgpu-octree-fine-levelset-bricks";
import {
  FINE_LEVELSET_REDISTANCE_CONTROL_BYTES,
  WebGPUFineLevelSetRedistance,
} from "./webgpu-octree-fine-levelset-redistance";
import {
  planFineLevelSetGPUTransportPasses,
  WebGPUFineLevelSetTransport,
} from "./webgpu-octree-fine-levelset-transport";
import { WebGPUFineLevelSetVolumeCorrection } from "./webgpu-octree-fine-levelset-volume";
import { WebGPUOctreeEnergyLedger, type OctreeEnergyLedgerSnapshot } from "./webgpu-octree-energy-ledger";
import { planFineLevelSetGPUSummaries, WebGPUFineLevelSetSummaries } from "./webgpu-octree-fine-levelset-summary";
import {
  planFineLevelSetTopologyBand,
  WebGPUFineLevelSetLeafSeeds,
  WebGPUFineLevelSetTopology,
} from "./webgpu-octree-fine-levelset-topology";

type OctreePipelineVariants = { full: GPUComputePipeline; delta: GPUComputePipeline };

/** Ordered, individually fenced t=0 authority checkpoints.
 * Aanjaneya et al. (2017), Section 5 p.8, first constructs regular octree-face
 * neighborhoods, augments T-junctions with local Delaunay tetrahedra, extends
 * air velocities from physical closest points, and only then interpolates the
 * result back to power faces. */
export const OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES = [
  { id: "cold-topology", label: "Cold octree topology" },
  { id: "power-operator-authority", label: "Power faces and operator authority" },
  { id: "surface-global-fine", label: "Surface and global-fine redistance publication" },
  { id: "section5-face-band-topology", label: "Section 5 face-band row topology" },
  { id: "section5-face-band-transitions", label: "Section 5 Delaunay transition adjacency and regular faces" },
  { id: "section5-face-band-closest-point", label: "Section 5 closest-point face velocity extension" },
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
  frontierSort: GPUComputePipeline[];
  refine: Map<number, OctreePipelineVariants>;
  refineCoarse: Map<number, OctreePipelineVariants>;
  balanceCoarse: Map<number, OctreePipelineVariants>;
  materialize: GPUComputePipeline;
}
const octreePipelineCache = new WeakMap<GPUDevice, Map<string, OctreePipelineCacheEntry>>();

export interface OctreeProjectionOptions {
  /** Immutable pressure-solver selection. Neither lane can invoke the other
   * after construction or use it as a numerical fallback. */
  powerPressureSolver?: "galerkin" | "section43-mgpcg";
  /** Required fixed symbolic hierarchy for the Galerkin lane. */
  powerGalerkinHierarchy?: OctreePowerGalerkinHierarchy;
  /** Matching pre/post L2 boundary-band sweeps. Kept as one control so the
   * Section 4.3 preconditioner cannot be made asymmetric from the UI. */
  powerBoundarySmoothingIterations?: number;
  maximumLeafSize?: 2 | 4 | 8 | 16 | 32;
  /** 0 = finest cells everywhere; 1 = full distance-graded coarsening. */
  adaptivity?: number;
  /** Pure-phase cells farther from liquid/solid interfaces than this finest-cell band may remain coarse. */
  interfaceRefinementBandCells?: number;
  /** Authoritative domain-global Section 5 narrow-band factor. */
  globalFineLevelSetFactor?: 4 | 8;
  /** Explicit physical brick cap for the global factor-4/factor-8 publication. */
  globalFineLevelSetMaximumBricks?: number;
  /** Advanced safety override for the compact pressure-row arena. */
  pressureRowCapacity?: number;
  /** Observational GPU-only stage energy reductions. Disabled by default and
   * never consumed by a simulation publication or acceptance gate. */
  energyLedger?: boolean;
  /** Fixed retained-step ring for the optional energy ledger. */
  energyLedgerStepCapacity?: number;
}

/** Semantic checkpoints consumed by the generic adjacent-boundary trace. */
export type OctreeSemanticPhase = "mgpcgSolve" | "pressureLeafCompactionL1Capture"
  | "powerDescriptorTopologyFaces" | "powerFaceRegularCompletion"
  | "powerOperatorRhsAssembly" | "finalPressureRowAssembly"
  | "powerProjectionPublication" | "faceBandTopologyBuild" | "faceBandTransitionAdjacency"
  | "faceBandClosestPointExtension" | "faceBandPowerPublicationCapture" | "powerProjectionTail";
export type OctreeSemanticBoundary = (
  phase: OctreeSemanticPhase,
  encoder: GPUCommandEncoder,
) => GPUCommandEncoder;

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

export interface OctreeProjectionPipelineReachability {
  readonly solidRasterization: boolean;
}

/** Compile only entry points reachable from the immutable solver configuration. */
export function octreeProjectionPipelineRequired(
  entryPoint: string,
  config: OctreeProjectionPipelineReachability,
): boolean {
  if (entryPoint === "rasterizeSolids" || entryPoint === "rasterizeSolidsDelta") {
    return config.solidRasterization;
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

/** Allocate only the sorted A/B publication, dirty candidate stream, and row delta. */
export function planOctreeLeafFrontierAllocation(
  cellCount: number,
  rowCapacity: number,
): OctreeLeafFrontierAllocationPlan {
  if (!Number.isSafeInteger(cellCount) || cellCount < 1) throw new Error("Octree frontier cell count must be a positive integer");
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) throw new Error("Octree frontier row capacity must be a positive integer");
  const listCapacity = Math.min(cellCount, rowCapacity);
  const candidateOffsetWords = 6 + 2 * listCapacity;
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
  const wallRows = planOctreePowerBoundaryStrip(dims, interfaceBandCells, closedTop).unitCellUpperBound;
  const requested = override === undefined ? surfaceRows + wallRows + coarseRows : override;
  const rowCapacity = Math.max(1, Math.min(count, aligned(Math.max(1, Math.floor(requested)))));
  const entryCapacity = rowCapacity * Math.max(
    6 * OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS,
    OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
  );
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
  /** Plain-storage scratch for cooperative row classification and scans. */
  rowDeltaScratchBaseWords: number;
  rowDeltaScratchWords: number;
  allocatedBytes: number;
}

/**
 * Size the shared scan/task arena from the authorities that can actually
 * publish work. Compact pressure owns at most one frontier row and one
 * cooperative coarse task per pressure slot. The resident active/retired tile
 * list remains an independent lower bound because it is copied into the same
 * buffer before topology rebuilds.
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
  const coarseTasksPerTile = (topologyTileSize / 8) ** 3;
  const candidateBlockCapacity = 2 * activeTileCapacity * coarseTasksPerTile;
  const coarseTaskCapacity = Math.min(pressureRowCapacity, activeTileCapacity * coarseTasksPerTile);
  const scanAndTaskBytes = 4 * (15 + 3 * scanBlockCapacity
    + 2 * Math.max(coarseTaskCapacity, candidateBlockCapacity));
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
  // The fourteen publication words are a last-good row-control snapshot plus
  // an independent exact-topology reuse bit.  The latter survives restoring
  // words 0..11, so downstream topology consumers can distinguish immutable
  // row reuse from a freshly emitted row set without a host readback.
  const activeTileBytes = 4 * ((2 + tileSignatureWords + tileFrontierSignatureWords
    + tileSignatureChangedWords + tileFrontierChangeFlagWords) * activeTileCapacity
    + dirtyAuthorityWords + rigidSnapshotWords + 14) + 32;
  const rowDeltaBlockCount = Math.ceil(pressureRowCapacity / 256);
  // Two row-sized streams (flags and exclusive ranks), one block-total stream
  // plus its exact total, and two words per classification block.
  const rowDeltaScratchWords = 2 * pressureRowCapacity + 3 * rowDeltaBlockCount + 1;
  const allocatedBytes = Math.max(60, scanAndTaskBytes, activeTileWorklistBytes)
    + rowDeltaScratchWords * 4 + activeTileBytes;
  const changeStateWords = (2 + tileSignatureWords + tileFrontierSignatureWords
    + tileSignatureChangedWords + tileFrontierChangeFlagWords) * activeTileCapacity
    + dirtyAuthorityWords + rigidSnapshotWords + 14;
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
  const rowDeltaScratchBaseWords = changeStateBaseWords - rowDeltaScratchWords;
  return { scanBlockCapacity, coarseTaskCapacity, candidateBlockCapacity, scanAndTaskBytes, activeTileBytes,
    changeStateBaseWords, tileChangeFlagsOffsetWords, tileRefinementSignaturesOffsetWords,
    tileFrontierSignaturesOffsetWords, tileSignatureChangedOffsetWords,
    tileFrontierChangeFlagsOffsetWords, frontierTopologyReuseWord,
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

type OctreeFirstOrderVCycleImplementation = OctreeFirstOrderSPDVCycle & {
  readonly plan: { readonly levelCount: number };
  encodeCapture(broker: PassBroker): void;
  destroy(): void;
};

/**
 * A GPU-resident, pressure-only octree projection.
 *
 * Ownership is paged and pressure exists only at live leaf origins, resolved
 * through the compact frontier hash.
 */
export class WebGPUOctreeProjection {
  readonly preconditioner: "fixed-galerkin" | "section43-hybrid";
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
  private readonly leafEntries: GPUBuffer;
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
  private globalFineVelocityPrepass?: WebGPUOctreePowerVelocityPrepass;
  private globalFineFaceExtension?: WebGPUOctreeFaceClosestPointExtension;
  private globalFineTransportA?: WebGPUFineLevelSetTransport;
  private globalFineTransportB?: WebGPUFineLevelSetTransport;
  private lastGlobalFineTransport?: WebGPUFineLevelSetTransport;
  private readonly globalFineSummaries?: WebGPUFineLevelSetSummaries;
  private readonly unpublishedFineSummaryDirectory: GPUBuffer;
  private powerVelocity?: WebGPUOctreePowerVelocity;
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
  private globalFineBootstrapped = false;
  private globalFineGeneration = 2;
  private lastPowerBoundaryFineSource?: { generation: number; generationSlot: 0 | 1 };
  private powerTimestep_s = 0;
  private powerAdvancingPressureSteps = 0;
  private readonly solveDispatch: GPUBuffer;
  private readonly topologyCandidateDispatch: GPUBuffer;
  private readonly topologyTileChangeFlagsOffsetBytes: number;
  private readonly topologyTileChangeFlagsByteLength: number;
  private readonly compactionAllocationRowDeltaScratchOffsetBytes: number;
  private readonly solidCells: GPUBuffer;
  private readonly hasDenseSolidCells: boolean;
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly shader: GPUShaderModule;
  private readonly diagnosticLayout: GPUBindGroupLayout;
  private readonly diagnosticPipelineLayout: GPUPipelineLayout;
  private readonly diagnosticShader: GPUShaderModule;
  private readonly surfaceState: WebGPUQuadtreeSurfaceState;
  private readonly sparseBrickWorld?: OctreeSparseBrickWorld;
  private readonly topologyResidency: GPUFluidBrickResidency;
  private readonly analyticBootstrapWorklist?: WebGPUOctreeAnalyticBootstrapWorklist;
  private sparseBrickWorldAccountedBytes = 0;
  private groups: { ab: GPUBindGroup; ba: GPUBindGroup };
  private fineSummarySizingGroup: GPUBindGroup;
  /** Current fine/coarse classification plus persistent topology-tile membership. */
  private topologyDecisionGroup?: GPUBindGroup;
  private denseBootstrapPhiReleased = false;
  private compactVoxelInspection?: CompactOctreeVoxelInspection;
  private readonly levelSetFallbackTexture?: GPUTexture;
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
  private scanAffectedRowDeltaBlocksPipeline!: GPUComputePipeline;
  private prefixAffectedRowDeltaBlocksPipeline!: GPUComputePipeline;
  private compactRowDeltaPipeline!: GPUComputePipeline;
  private publishRowDeltaPipeline!: GPUComputePipeline;
  private publishReusedRowDeltaPipeline!: GPUComputePipeline;
  private assemblePipeline!: GPUComputePipeline;
  private assembleCoarsePipeline!: GPUComputePipeline;
  private classifyTopologyTileSignaturePipeline!: GPUComputePipeline;
  private buildDirtyTileDeltaPipeline!: GPUComputePipeline;
  private buildDirtyFrontierDeltaPipeline!: GPUComputePipeline;
  private materializePipeline!: GPUComputePipeline;
  private readonly maxLeafSize: number;
  private readonly topologyTileSize: number;
  private readonly adaptivity: number;
  private readonly interfaceRefinementBandCells: number;
  private readonly pressureSolverMode: "galerkin" | "section43-mgpcg";
  private readonly mgpcg?: WebGPUOctreeMGPCG;
  private readonly galerkin?: WebGPUOctreePowerGalerkin;
  private readonly firstOrderVCycle?: OctreeFirstOrderVCycleImplementation;
  private readonly pressureCapacity: OctreePressureCapacityPlan;
  private readonly frontierAllocation: OctreeLeafFrontierAllocationPlan;
  /** A 4096-word shared sort occupies exactly WebGPU's portable 16 KiB floor. */
  private readonly useLocalFrontierCandidateSort: boolean;
  private readonly workgroups: [number, number, number];
  /** Immutable whole-domain candidate geometry and dyadic schedules. */
  private readonly candidateWorkgroups: [number, number, number];
  private readonly refinementSizes: readonly number[];
  private readonly coarseRefinementSizes: readonly number[];
  private readonly balanceRounds: number;
  private readonly linearBlocks: number;
  private readonly coarseTaskCapacity: number;
  private compactionByteLength = 0;
  private solveStats!: GPUBuffer;
  private topologyWorklistReady = false;
  private latestPressureInA = true;
  /** No dense phi exists; non-page topology groups must retain analytic sign until coarse correction publishes. */
  private readonly analyticSparseBootstrap: boolean;
  private powerDescriptor?: WebGPUOctreePowerDescriptor;
  private powerTopology?: WebGPUOctreePowerTopology;
  private powerFaces?: WebGPUOctreePowerFaces;
  private powerOperator?: WebGPUOctreePowerOperator;
  private powerFaceSeed?: WebGPUOctreePowerFaceSeed;
  private powerFaceAdvection?: WebGPUOctreePowerFaceAdvection;
  private powerSolidFaces?: WebGPUOctreePowerSolidFaces;
  private powerSolidVertices?: WebGPUOctreeSolidVertexSdf;
  private powerVolumes?: GPUBuffer;
  private powerVolumeParams?: GPUBuffer;
  private powerVolumePipeline?: GPUComputePipeline;
  private powerVolumeGroup?: GPUBindGroup;
  private powerGeneration = 0;
  private energyLedger?: WebGPUOctreeEnergyLedger;
  private readonly energyLedgerRequested: boolean;
  private readonly energyLedgerStepCapacity: number;
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
    this.pressureSolverMode = options.powerPressureSolver ?? "section43-mgpcg";
    this.preconditioner = this.pressureSolverMode === "galerkin"
      ? "fixed-galerkin"
      : "section43-hybrid";
    this.energyLedgerRequested = options.energyLedger === true;
    this.energyLedgerStepCapacity = options.energyLedgerStepCapacity ?? 512;
    this.maxLeafSize = octreeLeafSize(options.maximumLeafSize ?? 16);
    this.refinementSizes = Object.freeze((() => {
      const sizes: number[] = [];
      for (let size = this.maxLeafSize; size >= 2; size >>= 1) sizes.push(size);
      return sizes;
    })());
    this.coarseRefinementSizes = Object.freeze(
      this.refinementSizes.filter((size) => size >= 16),
    );
    this.balanceRounds = Math.max(1, Math.ceil(Math.log2(this.maxLeafSize)));
    this.adaptivity = Math.max(0, Math.min(1, options.adaptivity ?? 1));
    this.interfaceRefinementBandCells = Math.max(0, Math.min(32, Math.round(options.interfaceRefinementBandCells ?? 4)));
    // Analytic dam/tank scenes can construct compact topology and first fine seeds
    // phi without allocating or uploading a box-sized bootstrap texture.
    // Explicit seeded/imported shapes require the bounded sparse voxelizer and
    // are rejected below until that native bootstrap producer is available.
    const analyticSparseBootstrap = (scene.fluid.initialBrickSeeds_m?.length ?? 0) === 0
      && scene.rigidBodies.length === 0 && !sceneHasTerrain(scene);
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
    if ((scene.fluid.initialBrickSeeds_m?.length ?? 0) > 0) {
      throw new RangeError("Power projection does not support imported or explicitly seeded bootstrap geometry");
    }
    this.pressureCapacity = planOctreePressureCapacity(
      dims, this.maxLeafSize, this.interfaceRefinementBandCells,
      options.pressureRowCapacity,
      scene.container.top === "closed",
    );
    // The immutable pressure-row capacity selects the production lane. Every
    // capacity owned by one persistent workgroup must fit that executor's
    // literal iteration ceiling; larger capacities use bounded parallel PCG.
    this.frontierAllocation = planOctreeLeafFrontierAllocation(
      count,
      this.pressureCapacity.rowCapacity,
    );
    this.useLocalFrontierCandidateSort = this.frontierAllocation.listCapacity <= 4096;
    this.linearBlocks = Math.ceil(this.frontierAllocation.listCapacity / 256);
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
    this.topologyTileSize = Math.max(8, this.maxLeafSize);
    const allocateSparseWorld = octreeSparseWorldRequired(sceneHasTerrain(scene), scene.rigidBodies.length);
    const sparseWorldBrickSize = scene.voxelDomain.brickSize_cells;
    if (allocateSparseWorld) this.sparseBrickWorld = new OctreeSparseBrickWorld(device, scene, [dims.nx, dims.ny, dims.nz], {
      brickSize: sparseWorldBrickSize,
      haloCells: topologyHaloCells,
      // Canonical faces/pages own the simulation fields. Retain only the wet
      // bulk worklist needed by owner-page lifecycle.
      bulkResidency: true,
      brickPreActivation: true,
      topologyTileBricks: this.topologyTileSize / sparseWorldBrickSize
    });
    const analyticBootstrapPlan = analyticSparseBootstrap ? planOctreeAnalyticBootstrapBounds({
      dimensions: [dims.nx, dims.ny, dims.nz],
      containerSize: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
      tileSizeCells: this.topologyTileSize,
      initialCondition: scene.fluid.initialCondition,
      fillFraction: scene.container.fillFraction,
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
    ) : undefined;
    this.topologyResidency = this.sparseBrickWorld?.topologyResidency ?? new GPUFluidBrickResidency(
      device, [dims.nx, dims.ny, dims.nz], [cell.x, cell.y, cell.z], {
        brickSize: 8, haloCells: topologyHaloCells, retireAfterFrames: 3,
        topologyTileBricks: this.topologyTileSize / 8,
        // Direct page candidates consume no sparse-world leaf publication.
        // Keep only format-valid sentinel words for those bindings.
        fineSeedCandidatesOnly: true,
        fineSeedCandidateBrickCapacity: sparseSchedulerPools?.brickCapacity,
        fineSeedCandidateTileCapacity: sparseSchedulerPools?.tileCapacity,
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
      device, dims, cell, undefined,
      analyticSparseBootstrap
        ? new Float32Array([Math.max(cell.x, cell.y, cell.z) * this.maxLeafSize])
        : initialOctreeLevelSet(scene, dims, cell), undefined,
      undefined, false, false, true, true, this.hasDenseSolidCells ? this.solidCells : undefined, {
        worklist: this.topologyResidency.worklist,
        states: this.topologyResidency.stateBuffer,
        brickSize: 8
      }, true, analyticSparseBootstrap
    );
    // COPY_SRC on the sparse owner arena and pressure iterates exists solely for test
    // readbacks (leaf-size census, 2:1 balance, and finiteness audits); the
    // simulation itself never copies them out.
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
        activeTileLimits: this.analyticBootstrapWorklist.plan.activeTileLimits,
        activeTileCount: this.analyticBootstrapWorklist.plan.activeTileCount,
      } : undefined,
      {
        tileWorklist: this.topologyResidency.tileWorklist,
        tileSizeCells: this.topologyTileSize,
        tileListCapacity: this.topologyResidency.tilePublicationCapacity,
      },
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
    this.coarseTaskCapacity = compactionAllocation.coarseTaskCapacity;
    this.compactionByteLength = compactionAllocation.allocatedBytes;
    this.topologyTileChangeFlagsOffsetBytes = compactionAllocation.tileChangeFlagsOffsetWords * 4;
    this.topologyTileChangeFlagsByteLength = tileCapacity * 4;
    this.compactionAllocationRowDeltaScratchOffsetBytes =
      compactionAllocation.rowDeltaScratchBaseWords * 4;
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
    this.leafEntries = device.createBuffer({ label: "Octree leaf matrix entries", size: Math.max(8, this.pressureCapacity.entryBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    // Header + immutable A/B sorted publications + a bounded dirty-candidate
    // stream. Exact row lookup is a binary search over the live publication.
    this.leafFrontier = device.createBuffer({ label: "Persistent octree leaf frontier", size: this.frontierAllocation.allocatedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const source = {
      leafHeaders: this.leafHeaders,
      leafEntries: this.leafEntries,
      rowDelta: {
        rows: this.leafFrontier,
        rowCapacity: this.frontierAllocation.listCapacity,
        controlOffsetWords: this.frontierAllocation.rowDeltaControlOffsetWords,
        newToOldOffsetWords: this.frontierAllocation.rowDeltaNewToOldOffsetWords,
        dirtyRowsOffsetWords: this.frontierAllocation.rowDeltaDirtyRowsOffsetWords,
      },
    };
    if (this.pressureSolverMode === "galerkin") {
      const hierarchy = options.powerGalerkinHierarchy;
      if (!hierarchy) {
        throw new Error("fixed Galerkin pressure mode requires an explicitly supplied hierarchy");
      }
      if (hierarchy.dimensions[0] !== dims.nx || hierarchy.dimensions[1] !== dims.ny
        || hierarchy.dimensions[2] !== dims.nz) {
        throw new RangeError("fixed Galerkin hierarchy dimensions disagree with the UI pressure domain");
      }
      const finest = hierarchy.levels[0];
      const operators = refreshOctreePowerGalerkinOperators(
        hierarchy,
        new Float64Array(finest.nodeCount).fill(1),
        new Float64Array(hierarchy.fineOffDiagonalEntries.length),
      );
      this.galerkin = new WebGPUOctreePowerGalerkin(device, hierarchy, operators, {
        // Newly refined/coarsened rows can exceed eight cycles in the
        // mini-dam even with an exact remapped warm start. The GPU convergence
        // latch makes the encoded tail a no-op for ordinary short solves.
        cycles: 20,
        // Measured wall-neutral on the isolated Dawn benchmark (glue kernels,
        // not smoothers, own the cycle cost) while roughly halving the cycles
        // needed to reach the residual gate.
        smoothingIterations: 4,
        // Trilinear RAP rows do not retain the fine-grid M-matrix Jacobi
        // spectral bound. One quarter remains stable through both normal- and
        // mini-dam topology transitions; one half amplifies the normal dam.
        damping: 0.25,
        // The experimental single-dispatch three-level lane is not yet a
        // publication-equivalent replacement for the staged Galerkin cycle:
        // it can reject an otherwise valid mini-dam solve after a topology
        // transition. Keep the UI/default constructor on the audited staged
        // path until that lane has its own full-generation acceptance.
        persistentThreeLevel: false,
        relativeTolerance: scene.numerics.pressureRelativeTolerance,
      });
    } else {
      const cycleOptions = { dimensions: [dims.nx, dims.ny, dims.nz] as const,
        rowCapacity: this.pressureCapacity.rowCapacity, finestCellWidth: cell.x };
      this.firstOrderVCycle = new WebGPUOctreeSPGridVCycle(device, source, cycleOptions);
      this.mgpcg = new WebGPUOctreeMGPCG(device, {
        leafHeaders: this.leafHeaders,
        leafEntries: this.leafEntries,
        rowCount: this.compaction,
        firstOrderVCycle: this.firstOrderVCycle,
      }, {
        dimensions: [dims.nx, dims.ny, dims.nz],
        rowCapacity: this.pressureCapacity.rowCapacity,
        boundarySmoothingIterations: options.powerBoundarySmoothingIterations,
        relativeTolerance: scene.numerics.pressureRelativeTolerance,
      });
    }
    this.solveDispatch = device.createBuffer({ label: "Octree leaf solve and retired-topology dispatch", size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    device.queue.writeBuffer(this.solveDispatch, 32, new Uint32Array([0, 1, 1, 0, 0, 1, 1, 0]));
    // Words 8..15 hold one-workgroup-per-tile coarse topology dispatches: the
    // per-frame copies refresh only the x counts, so y/z stay 1 from here.
    this.topologyCandidateDispatch = device.createBuffer({
      label: "Octree topology, frontier, and row-delta dispatch",
      size: 48,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });
    this.params = device.createBuffer({ label: "Octree projection parameters", size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // The factor-m global fine band is buffer-native. Retain one format-only
    // sampled texture only because the common diagnostic layout requires it.
    this.levelSetFallbackTexture = device.createTexture({
      label: "Octree format-only sampled phi", size: [1, 1, 1], dimension: "3d", format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.layout = device.createBindGroupLayout({ entries: [
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
      ,{ binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.shader = device.createShaderModule({ label: "GPU-resident octree projection", code: octreeProjectionShader });
    this.groups = {
      ab: this.createProjectionGroup(this.pressureA, this.pressureB),
      ba: this.createProjectionGroup(this.pressureB, this.pressureA),
    };
    this.fineSummarySizingGroup = this.createProjectionGroup(
      this.unpublishedFineSummaryDirectory, this.pressureB);
    this.diagnosticLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32uint", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    this.diagnosticPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.diagnosticLayout] });
    this.diagnosticShader = device.createShaderModule({ label: "GPU octree overlay materialization", code: octreeDiagnosticShader });
    this.workgroups = [Math.ceil(dims.nx / 4), Math.ceil(dims.ny / 4), Math.ceil(dims.nz / 4)];
    this.candidateWorkgroups = [
      Math.ceil(dims.nx / 8), Math.ceil(dims.ny / 8), Math.ceil(dims.nz / 8),
    ];
    const fullyCoarseEstimate = Math.ceil(count / (this.maxLeafSize ** 3));
    const approximateLeaves = Math.ceil(count * (1 - this.adaptivity) + fullyCoarseEstimate * this.adaptivity);
    this.info = {
      leafCount: approximateLeaves, pressureSampleCount: approximateLeaves, liquidDofCount: approximateLeaves,
      faceCount: 0, mlsProjectionRowCount: 0, tallSegmentCount: 0, ghostFaceCount: 0,
      maximumNeighborRatio: 2, maximumFluidScale: this.maxLeafSize, compressionRatio: approximateLeaves / Math.max(1, count),
      allocatedBytes: this.ownerPages.allocatedBytes + this.solidCells.size
        + surfaceStateAllocation.allocatedBytes
        + this.pressureA.size + this.pressureB.size + this.leafHeaders.size + this.leafEntries.size
        + this.leafFrontier.size + this.compaction.size + this.unpublishedFineSummaryDirectory.size + 208
        + (this.mgpcg?.plan.allocatedBytes ?? this.galerkin?.allocatedBytes ?? 0)
        + (this.firstOrderVCycle?.allocatedBytes ?? 0)
        + (this.sparseBrickWorld?.allocatedBytes ?? this.topologyResidency.allocatedBytes)
        + (this.analyticBootstrapWorklist?.allocatedBytes ?? 0),
      pressureIterationsUsed: 0,
      pressureIterationBudget: this.mgpcg?.iterationBudget ?? this.galerkin?.plan.cycles ?? 0,
      pressureIterationHardBudget: this.mgpcg?.iterationBudget ?? this.galerkin?.plan.cycles ?? 0,
      pressureConverged: undefined,
      pressureRowCapacity: pressureSlots, pressureEntryCapacity: this.pressureCapacity.entryCapacity,
      pressureCapacityOverflow: false,
      frontierListCapacity: this.frontierAllocation.listCapacity,
      frontierCapacityOverflow: false,
      velocityClampCount: 0,
      factorLevelCount: this.firstOrderVCycle?.plan.levelCount ?? this.galerkin?.plan.levelCount ?? 0,
      multigridLevelCount: this.firstOrderVCycle?.plan.levelCount ?? this.galerkin?.plan.levelCount ?? 0,
      multigridCoarsestDofs: this.firstOrderVCycle instanceof WebGPUOctreeSPGridVCycle
        ? this.firstOrderVCycle.diagnostics.coarsestDegreesOfFreedom : 0,
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
          finestLeafSize: this.maxLeafSize,
          haloCells: this.interfaceRefinementBandCells,
          ...(analyticSparseBootstrap ? {
            analyticInitialCondition: scene.fluid.initialCondition,
            initialFillFraction: scene.container.fillFraction,
          } : {}),
      });
      this.info.allocatedBytes += this.fineSeedAdapter.plan.allocatedBytes;
        const globalFineFactor = options.globalFineLevelSetFactor ?? 4;
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
          const transportBandFineCells = Math.min(256, Math.max(4,
            this.interfaceRefinementBandCells * globalFineFactor));
          // Section 5 transports every sample in the authored narrow band.
          // The resident topology must therefore also hold the complete
          // backtrace and trilinear stencil beyond that band.  A 3-D
          // trilinear corner can be sqrt(3) fine cells from the query, so its
          // signed-distance support needs two cells rather than one.
          // The redistancer retains reachable samples on the closed authored
          // cutoff. Two cells cover ceil(sqrt(3)) interpolation reach, and one
          // final cell keeps all eight samples around a pressure-cell centre
          // valid when that centre itself lands on the outer support shell.
          // An unreachable cutoff sentinel is still rejected by seed identity.
          const redistanceBandFineCells = Math.min(256,
            transportBandFineCells + globalFineFactor + 3);
          const physicalBand = planFineLevelSetTopologyBand(brickResolution, {
            maximumBacktraceFineCells: globalFineFactor,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells,
            safetyBrickRings: 1,
          });
          const defaultCapacity = estimateGlobalFineNarrowBandBrickCapacity(
            brickDimensions, physicalBand.dilationBrickRings,
          );
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
          const exactAnalyticFineSeed = this.analyticSparseBootstrap
            && (this.scene.fluid.initialBrickSeeds_m?.length ?? 0) === 0
            ? { initialCondition: this.scene.fluid.initialCondition,
              fillFraction: this.scene.container.fillFraction }
            : undefined;
          this.globalFineSeeds = new WebGPUFineLevelSetLeafSeeds(
            device, this.globalFineSourceB, exactAnalyticFineSeed);
          this.globalFineSummaries = new WebGPUFineLevelSetSummaries(device, globalPlan,
            this.pressureCapacity.rowCapacity);
          // The common layout is already at WebGPU's portable ten-storage-
          // buffer limit. Refinement reuses pressure binding 4 for this raw
          // read-only directory instead of adding an eleventh binding.
          this.fineSummarySizingGroup = this.createProjectionGroup(
            this.globalFineSummaries.directory, this.pressureB);
          const allocated = this.globalFineLevelSet.allocatedBytes + this.globalFineSeeds.allocatedBytes
            + this.globalFineSummaries.plan.allocatedBytes;
          this.info.allocatedBytes += allocated;
          this.info.globalFineLevelSetAllocatedBytes += allocated;
          this.info.globalFineLevelSetResidentBrickCapacity = globalPlan.maximumResidentBricks;
          this.info.globalFineLevelSetLogicalBrickCount = globalPlan.logicalBrickCount;
    this.sparseBrickWorldAccountedBytes = this.sparseBrickWorld?.allocatedBytes ?? 0;
    if (!deferPipelineCompilation) this.createPipelinesSync();
    this.writeParams();
  }

  private createProjectionGroup(
    pressureIn: GPUBuffer,
    pressureOut: GPUBuffer,
    binding15Override?: GPUBuffer,
  ): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 2, resource: { buffer: this.compaction } },
      { binding: 3, resource: { buffer: this.topology } },
      { binding: 4, resource: { buffer: pressureIn } },
      { binding: 5, resource: { buffer: pressureOut } },
      { binding: 6, resource: { buffer: this.params } },
      { binding: 7, resource: this.levelSetTexture.createView() },
      { binding: 8, resource: { buffer: this.leafHeaders } },
      { binding: 9, resource: { buffer: this.leafEntries } },
      { binding: 10, resource: { buffer: this.resources.rigidBodies } },
      { binding: 11, resource: { buffer: this.solidCells } },
      { binding: 12, resource: this.resources.terrain.createView() },
      { binding: 13, resource: { buffer: this.leafFrontier } },
      { binding: 15, resource: { buffer: binding15Override
        ?? this.sparseBrickWorld?.bulkResidencyWorklist
        ?? this.topologyResidency.worklist } },
    ] });
  }

  private descriptor(entryPoint: string): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.shader, entryPoint, constants: this.pipelineConstants() } };
  }
  private refinementDescriptor(entryPoint: string, size: number): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.shader, entryPoint, constants: { ...this.pipelineConstants(), targetRefinementSize: size } } };
  }
  private frontierSortDescriptor(stage: number): GPUComputePipelineDescriptor {
    return {
      layout: this.pipelineLayout,
      compute: {
        module: this.shader,
        entryPoint: "sortFrontierCandidates",
        constants: { ...this.pipelineConstants(), frontierSortStage: stage },
      },
    };
  }
  private pipelineConstants(): Record<string, number> {
    return {
      rowIndexedPressure: 1,
      sparseTopologyTileStates: this.topologyResidency.allocationPlan.sparseKeyPools ? 1 : 0,
      denseSolidField: this.hasDenseSolidCells ? 1 : 0,
    };
  }
  private diagnosticDescriptor(): GPUComputePipelineDescriptor {
    return { layout: this.diagnosticPipelineLayout, compute: { module: this.diagnosticShader, entryPoint: "materializeOctreeFields", constants: { rowIndexedPressure: 1 } } };
  }

  private static readonly pipelineEntryPoints = [
    "rasterizeSolids", "resetTopology", "refineTopology", "balanceTopology",
    "rasterizeSolidsDelta", "resetTopologyDelta", "refineTopologyDelta", "balanceTopologyDelta",
    "beginFrontier", "classifyFrontierCandidates", "classifyFrontierCandidatesDelta",
    "prefixFrontierCandidateBlocks", "prefixFrontierCandidateBlocksDelta",
    "emitFrontierCandidates", "emitFrontierCandidatesDelta",
    "prepareFrontierDispatch", "sortFrontierCandidatesLocal",
    "classifyFrontierCarry",
    "scanFrontierCarryBlocks", "prefixFrontierCarryBlocks", "mergeFrontierRows", "finalizeFrontier",
    "prepareRowDelta", "classifyRowDelta",
    "finalizeRowDeltaClassification", "scanDirtyRowDeltaBlocks", "prefixDirtyRowDeltaBlocks",
    "scatterDirtyRowDelta", "markRowDeltaRing", "scanAffectedRowDeltaBlocks",
    "prefixAffectedRowDeltaBlocks", "compactRowDelta", "publishRowDelta", "publishReusedRowDelta",
    "planLeaves", "scanLeafBlocks", "emitLeaves", "assembleSystem", "assembleCoarseSystem",
    "classifyTopologyTileSignature", "buildDirtyTileDelta", "buildDirtyFrontierDelta"
  ] as const;

  private assignPipelines(compiled: GPUComputePipeline[]) {
    [
      this.rasterizeSolidsPipeline, this.resetPipeline, this.refinePipeline, this.balancePipeline,
      this.rasterizeSolidsDeltaPipeline, this.resetDeltaPipeline, this.refineDeltaPipeline, this.balanceDeltaPipeline,
      this.beginFrontierPipeline, this.classifyFrontierCandidatesPipeline, this.classifyFrontierCandidatesDeltaPipeline,
      this.prefixFrontierCandidateBlocksPipeline, this.prefixFrontierCandidateBlocksDeltaPipeline,
      this.emitFrontierCandidatesPipeline, this.emitFrontierCandidatesDeltaPipeline,
      this.prepareFrontierDispatchPipeline, this.sortFrontierCandidatesLocalPipeline,
      this.classifyFrontierCarryPipeline, this.scanFrontierCarryBlocksPipeline,
      this.prefixFrontierCarryBlocksPipeline, this.mergeFrontierRowsPipeline, this.finalizeFrontierPipeline,
      this.prepareRowDeltaPipeline,
      this.classifyRowDeltaPipeline, this.finalizeRowDeltaClassificationPipeline,
      this.scanDirtyRowDeltaBlocksPipeline, this.prefixDirtyRowDeltaBlocksPipeline,
      this.scatterDirtyRowDeltaPipeline, this.markRowDeltaRingPipeline,
      this.scanAffectedRowDeltaBlocksPipeline, this.prefixAffectedRowDeltaBlocksPipeline,
      this.compactRowDeltaPipeline, this.publishRowDeltaPipeline, this.publishReusedRowDeltaPipeline,
      this.planPipeline, this.scanPipeline, this.emitPipeline, this.assemblePipeline, this.assembleCoarsePipeline,
      this.classifyTopologyTileSignaturePipeline, this.buildDirtyTileDeltaPipeline,
      this.buildDirtyFrontierDeltaPipeline
    ] = compiled;
  }

  private pipelineReachability(): OctreeProjectionPipelineReachability {
    return {
      solidRasterization: this.hasDenseSolidCells,
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
      reachability: stableEntries(reachability),
      frontierSortStages: Math.ceil(Math.log2(Math.max(1, this.frontierAllocation.listCapacity))),
      requiredEntryPoints: WebGPUOctreeProjection.pipelineEntryPoints
        .filter((entryPoint) => octreeProjectionPipelineRequired(entryPoint, reachability)),
    });
  }

  private applyPipelineCache(entry: OctreePipelineCacheEntry) {
    this.assignPipelines(entry.base);
    this.frontierCandidateSortPipelines = entry.frontierSort;
    this.refineLevelPipelines.clear(); entry.refine.forEach((value, key) => this.refineLevelPipelines.set(key, value));
    this.refineCoarsePipelines.clear(); entry.refineCoarse.forEach((value, key) => this.refineCoarsePipelines.set(key, value));
    this.balanceCoarsePipelines.clear(); entry.balanceCoarse.forEach((value, key) => this.balanceCoarsePipelines.set(key, value));
    this.materializePipeline = entry.materialize;
  }

  private publishPipelineCache() {
    let cache = octreePipelineCache.get(this.device);
    if (!cache) { cache = new Map(); octreePipelineCache.set(this.device, cache); }
    cache.set(this.pipelineCacheKey(), {
      base: WebGPUOctreeProjection.pipelineEntryPoints.map((_, index) => [
        this.rasterizeSolidsPipeline, this.resetPipeline, this.refinePipeline, this.balancePipeline,
        this.rasterizeSolidsDeltaPipeline, this.resetDeltaPipeline, this.refineDeltaPipeline, this.balanceDeltaPipeline,
        this.beginFrontierPipeline, this.classifyFrontierCandidatesPipeline, this.classifyFrontierCandidatesDeltaPipeline,
        this.prefixFrontierCandidateBlocksPipeline, this.prefixFrontierCandidateBlocksDeltaPipeline,
        this.emitFrontierCandidatesPipeline, this.emitFrontierCandidatesDeltaPipeline,
        this.prepareFrontierDispatchPipeline, this.sortFrontierCandidatesLocalPipeline,
        this.classifyFrontierCarryPipeline, this.scanFrontierCarryBlocksPipeline,
        this.prefixFrontierCarryBlocksPipeline, this.mergeFrontierRowsPipeline, this.finalizeFrontierPipeline,
        this.prepareRowDeltaPipeline,
        this.classifyRowDeltaPipeline, this.finalizeRowDeltaClassificationPipeline,
        this.scanDirtyRowDeltaBlocksPipeline, this.prefixDirtyRowDeltaBlocksPipeline,
        this.scatterDirtyRowDeltaPipeline, this.markRowDeltaRingPipeline,
        this.scanAffectedRowDeltaBlocksPipeline, this.prefixAffectedRowDeltaBlocksPipeline,
        this.compactRowDeltaPipeline, this.publishRowDeltaPipeline, this.publishReusedRowDeltaPipeline,
        this.planPipeline, this.scanPipeline, this.emitPipeline, this.assemblePipeline, this.assembleCoarsePipeline,
        this.classifyTopologyTileSignaturePipeline, this.buildDirtyTileDeltaPipeline,
        this.buildDirtyFrontierDeltaPipeline,
      ][index]),
      frontierSort: [...this.frontierCandidateSortPipelines],
      refine: new Map(this.refineLevelPipelines), refineCoarse: new Map(this.refineCoarsePipelines), balanceCoarse: new Map(this.balanceCoarsePipelines),
      materialize: this.materializePipeline,
    });
  }

  private createPipelinesSync() {
    const compiled: GPUComputePipeline[] = [];
    WebGPUOctreeProjection.pipelineEntryPoints.forEach((entryPoint, index) => {
      if (this.basePipelineRequired(entryPoint)) {
        compiled[index] = this.device.createComputePipeline(this.descriptor(entryPoint));
      }
    });
    // Unreachable tuple slots stay unpublished. Accidentally selecting one
    // therefore fails construction/encoding instead of running an unrelated
    // successfully compiled program as a compatibility substitute.
    this.assignPipelines(compiled);
    if (!this.useLocalFrontierCandidateSort) {
      const frontierSortStages = Math.ceil(Math.log2(Math.max(1, this.frontierAllocation.listCapacity)));
      this.frontierCandidateSortPipelines = Array.from(
        { length: frontierSortStages + 1 },
        (_, stage) => this.device.createComputePipeline(this.frontierSortDescriptor(stage)),
      );
    }
    // Sizes 16 and 32 use the coarse worklist kernels below. The immutable
    // maximum participates in the cache key, so variants above it cannot be
    // reused by the replacement solver created for a settings change.
    for (let size = Math.min(8, this.maxLeafSize); size >= 2; size >>= 1) this.refineLevelPipelines.set(size, {
      full: this.device.createComputePipeline(this.refinementDescriptor("refineTopology", size)),
      delta: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyDelta", size)),
    });
    for (let size = this.maxLeafSize; size >= 16; size >>= 1) {
      this.refineCoarsePipelines.set(size, {
        full: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyCoarse", size)),
        delta: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyCoarseDelta", size)),
      });
      this.balanceCoarsePipelines.set(size, {
        full: this.device.createComputePipeline(this.refinementDescriptor("balanceTopologyCoarse", size)),
        delta: this.device.createComputePipeline(this.refinementDescriptor("balanceTopologyCoarseDelta", size)),
      });
    }
    this.materializePipeline = this.device.createComputePipeline(this.diagnosticDescriptor());
    this.publishPipelineCache();
  }

  get topologyTexture() { return this.topologyDiagnosticTexture; }
  get pressureSamplesTexture() { return this.pressureSamplesDiagnosticTexture; }
  get pressureTexture() { return this.pressureDiagnosticTexture; }
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
    const diagnosticGroup = (pressure: GPUBuffer) => this.device.createBindGroup({ layout: this.diagnosticLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } },
      { binding: 1, resource: { buffer: pressure } },
      { binding: 3, resource: this.levelSetTexture.createView() },
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
            // Unreachable tuple slots remain unpublished; the immutable
            // reachability key prevents this sparse tuple from serving a
            // configuration that requires them.
            this.assignPipelines(compiled);
          }
        },
      });
    });
    if (!cached && !this.useLocalFrontierCandidateSort) {
      const frontierSortStages = Math.ceil(Math.log2(Math.max(1, this.frontierAllocation.listCapacity)));
      const frontierSort = new Array<GPUComputePipeline>(frontierSortStages + 1);
      for (let stage = 0; stage <= frontierSortStages; stage += 1) {
        tasks.push({
          id: `octree.pipeline.frontier-sort.${stage}`,
          phase: "adaptive-topology",
          label: `Compile octree frontier merge stage ${stage}`,
          run: async () => {
            frontierSort[stage] = await this.device.createComputePipelineAsync(
              this.frontierSortDescriptor(stage),
            );
            if (stage === frontierSortStages) this.frontierCandidateSortPipelines = frontierSort;
          },
        });
      }
    }
    for (let size = Math.min(8, this.maxLeafSize); size >= 2; size >>= 1) {
      if (cached?.refine.has(size)) continue;
      const level: Partial<OctreePipelineVariants> = {};
      const definitions = [
        ["full", "refineTopology"],
        ["delta", "refineTopologyDelta"],
      ] as const;
      definitions.forEach(([variant, entryPoint], index) => tasks.push({
        id: `octree.pipeline.refine.${size}.${variant}`,
        phase: "adaptive-topology",
        label: `Compile octree refinement ${size} · ${variant}`,
        run: async () => {
          level[variant] = await this.device.createComputePipelineAsync(this.refinementDescriptor(entryPoint, size));
          if (index === definitions.length - 1) {
            this.refineLevelPipelines.set(size, level as OctreePipelineVariants);
          }
        },
      }));
    }
    for (let size = this.maxLeafSize; size >= 16; size >>= 1) {
      for (const operation of ["refine", "balance"] as const) {
        if ((operation === "refine" ? cached?.refineCoarse : cached?.balanceCoarse)?.has(size)) continue;
        const pipelines: Partial<OctreePipelineVariants> = {};
        const prefix = operation === "refine" ? "refineTopologyCoarse" : "balanceTopologyCoarse";
        const definitions = [["full", prefix], ["delta", `${prefix}Delta`]] as const;
        definitions.forEach(([variant, entryPoint], index) => tasks.push({
          id: `octree.pipeline.${operation}-coarse.${size}.${variant}`,
          phase: "adaptive-topology",
          label: `Compile octree coarse ${operation} ${size} · ${variant}`,
          run: async () => {
            pipelines[variant] = await this.device.createComputePipelineAsync(this.refinementDescriptor(entryPoint, size));
            if (index === definitions.length - 1) {
              const complete = pipelines as OctreePipelineVariants;
              if (operation === "refine") this.refineCoarsePipelines.set(size, complete);
              else this.balanceCoarsePipelines.set(size, complete);
            }
          },
        }));
      }
    }
    if (!cached) {
      tasks.push({ id: "octree.pipeline.materialize", phase: "adaptive-topology", label: "Compile octree overlay materialization", run: async () => { this.materializePipeline = await this.device.createComputePipelineAsync(this.diagnosticDescriptor()); } });
      tasks.push({ id: "octree.pipeline-cache.publish", phase: "adaptive-topology", label: "Publish compiled octree pipelines", run: () => this.publishPipelineCache() });
    } else if (tasks.length > 1) {
      tasks.push({ id: "octree.pipeline-cache.publish", phase: "adaptive-topology", label: "Publish compiled adaptive variants", run: () => this.publishPipelineCache() });
    }
    if (!this.powerDescriptor) tasks.push({
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
          this.initializeNativePowerAuthority(catalog);
        } catch (error) {
          this.info.powerDiagramReady = false;
          throw error;
        }
      },
    });
    return tasks;
  }

  private initializeNativePowerAuthority(catalog: GeneratedOctreePowerCatalogViews): void {
    if (this.powerDescriptor || this.powerLifecycleDisposed) return;
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
    this.powerDescriptor = new WebGPUOctreePowerDescriptor(this.device, rowCapacity);
    this.powerTopology = new WebGPUOctreePowerTopology(this.device, rowCapacity, catalog);
    this.powerFaces = new WebGPUOctreePowerFaces(this.device, rowCapacity, faceCapacity, this.powerTopology.source, incidenceCapacity);
    this.powerOperator = new WebGPUOctreePowerOperator(
      this.device, rowCapacity, faceCapacity, entryCapacity, maximumIncidence,
      { topology: this.powerTopology!.source, leafHeaders: this.leafHeaders,
        physicalCellSize: this.scene.container.width_m / this.dims.nx,
        physicalCellVolume: (this.scene.container.width_m / this.dims.nx)
          * (this.scene.container.height_m / this.dims.ny) * (this.scene.container.depth_m / this.dims.nz) },
    );
    this.powerVelocity = new WebGPUOctreePowerVelocity(this.device, rowCapacity);
    // The paper evolves coarse octree phi regardless of whether the optional
    // factor-4/factor-8 interface band exists. It is also the complete
    // inside/outside and cell-centre boundary authority in coarse-only mode.
    this.powerCoarseLevelSet = new WebGPUOctreeCoarseLevelSet(this.device, rowCapacity);
    this.powerCoarseLevelSetSchedule = new WebGPUOctreePowerCoarseLevelSet(
      this.device, this.powerCoarseLevelSet, this.powerTopology.source,
    );
    const coarseDirectory = this.powerCoarseLevelSetSchedule.sampleSource.directory;
    // Binding 15 is the compact coarse-phi directory for the mandatory power
    // topology/pressure authority.
    this.groups = {
      ab: this.createProjectionGroup(this.pressureA, this.pressureB, coarseDirectory),
      ba: this.createProjectionGroup(this.pressureB, this.pressureA, coarseDirectory),
    };
    this.fineSummarySizingGroup = this.createProjectionGroup(
      this.globalFineSummaries?.directory ?? this.unpublishedFineSummaryDirectory, this.pressureB,
      coarseDirectory);
    // Structural dirtiness compares the current pressure-owner decisions, not
    // the fine payload transaction. This stable group keeps the fine summary,
    // persistent topology-tile membership, and canonical coarse fallback
    // together across A/B fine generations.
    this.topologyDecisionGroup = this.createProjectionGroup(
      this.globalFineSummaries?.directory ?? this.unpublishedFineSummaryDirectory,
      this.topologyResidency.topologyTileStateBuffer,
      coarseDirectory,
    );
    this.fineSeedAdapter?.setCoarsePhiSource(
      this.powerCoarseLevelSetSchedule.sampleSource,
    );
    this.fineSeedAdapter?.setPowerVelocitySource(this.powerVelocity.source);
    if (this.globalFineSourceA && this.globalFineSourceB) {
      this.fineToPowerCoarseLevelSet = new WebGPUFineToCoarseLevelSet(this.device, rowCapacity,
        this.globalFineSourceA.plan.maximumResidentBricks * this.globalFineSourceA.plan.samplesPerBrick);
      const compactCoarse = this.powerCoarseLevelSetSchedule.sampleSource;
      this.globalFineTopologyAB = new WebGPUFineLevelSetTopology(
        this.device, this.globalFineSourceA, this.globalFineSourceB, compactCoarse.wgsl(9),
      );
      this.globalFineTopologyBA = new WebGPUFineLevelSetTopology(
        this.device, this.globalFineSourceB, this.globalFineSourceA, compactCoarse.wgsl(9),
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
      );
      this.globalFineRedistanceB = new WebGPUFineLevelSetRedistance(
        this.device, this.globalFineSourceB, this.globalFineTopologyAB,
      );
    }
    this.powerFaceSeed = new WebGPUOctreePowerFaceSeed(this.device, this.powerFaces.source);
    this.powerFaceAdvection = new WebGPUOctreePowerFaceAdvection(
      this.device, this.powerTopology.source, this.powerFaces.source);
    if (sceneHasTerrain(this.scene)) {
      this.powerSolidVertices = new WebGPUOctreeSolidVertexSdf(
        this.device, rowCapacity, this.leafHeaders, this.compaction, this.resources.terrain, this.powerFaceSeed!.control,
      );
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
      // Aanjaneya et al. (2017), Section 5 extrapolates velocity only through
      // the fine interface band needed by characteristic backtraces. The
      // row-phi extension remains bounded by that authored physical band.
      const bandPhiRelaxationRounds = Math.min(256, Math.max(4,
        this.interfaceRefinementBandCells * (this.globalFineLevelSet?.plan.fineFactor ?? 4)));
      const faceExtension = new WebGPUOctreeFaceClosestPointExtension(
        this.device, this.globalFineSourceA, rowCapacity, bandPhiRelaxationRounds,
        this.powerFaces.plan.faceCapacity,
      );
      this.globalFineFaceExtension = faceExtension;
      this.globalFineTransportA = new WebGPUFineLevelSetTransport(
        this.device, this.globalFineSourceA, this.globalFineVelocityPrepass, faceExtension,
      );
      this.globalFineTransportB = new WebGPUFineLevelSetTransport(
        this.device, this.globalFineSourceB, this.globalFineVelocityPrepass, faceExtension,
      );
      const transportPasses = planFineLevelSetGPUTransportPasses(
        this.globalFineTransportA.plan, this.globalFineSourceA.plan.fineFactor);
      this.info.globalFineTransportQueryCapacity = this.globalFineTransportA.queryCapacity;
      this.info.globalFineTransportChunkCapacity = this.globalFineTransportA.plan.velocityChunkCapacity;
      this.info.globalFineTransportChunkCount = transportPasses.chunkCount;
      this.info.globalFineTransportSegmentCount = transportPasses.segmentCount;
      this.info.globalFineTransportEncodedPasses = transportPasses.encodedPasses;
      this.info.globalFineTransportPrepassScratchBytes = this.globalFineVelocityPrepass.plan.scratchBytes;
      this.info.globalFineTransportVertexScratchBytes = this.globalFineVelocityPrepass.plan.vertexVelocityBytes;
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
      );
      this.globalFineVolumeB = new WebGPUFineLevelSetVolumeCorrection(
        this.device, this.globalFineSourceB, coarseVolumeSource, this.globalFineVolumeA.control,
      );
    }
    if (this.energyLedgerRequested && this.globalFineSourceA) {
      const fineSampleCapacity = this.globalFineSourceA.plan.maximumResidentBricks
        * this.globalFineSourceA.plan.samplesPerBrick;
      this.energyLedger = new WebGPUOctreeEnergyLedger(
        this.device,
        this.powerFaces.plan.faceCapacity,
        fineSampleCapacity,
        [this.scene.fluid.gravity_m_s2.x, this.scene.fluid.gravity_m_s2.y, this.scene.fluid.gravity_m_s2.z],
        this.energyLedgerStepCapacity,
      );
      this.info.allocatedBytes += this.energyLedger.plan.allocatedBytes;
    }
    this.powerVolumeParams = this.device.createBuffer({ label: "Octree power-volume parameters", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const cellVolume = (this.scene.container.width_m / this.dims.nx)
      * (this.scene.container.height_m / this.dims.ny)
      * (this.scene.container.depth_m / this.dims.nz);
    const data = new Float32Array(4); data[0] = cellVolume;
    this.device.queue.writeBuffer(this.powerVolumeParams, 0, data);
    const shaderModule = this.device.createShaderModule({ label: "Publish physical octree power volumes", code: octreePowerVolumeShader });
    this.powerVolumePipeline = this.device.createComputePipeline({ label: "Publish physical octree power volumes", layout: "auto",
      compute: { module: shaderModule, entryPoint: "publishPowerVolumes" } });
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
      faceAdvection: this.powerFaceAdvection?.plan.allocatedBytes ?? 0,
      faceTransfer: 0,
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
      redistanceA: this.globalFineRedistanceA?.allocatedBytes ?? 0,
      redistanceB: this.globalFineRedistanceB?.allocatedBytes ?? 0,
      velocityPrepass: this.globalFineVelocityPrepass?.plan.allocatedBytes ?? 0,
      faceClosestPointExtension: this.globalFineFaceExtension?.plan.gpuAllocatedBytes ?? 0,
      transportA: this.globalFineTransportA?.plan.allocatedBytes ?? 0,
      transportB: this.globalFineTransportB?.plan.allocatedBytes ?? 0,
      volumeA: this.globalFineVolumeA?.allocatedBytes ?? 0,
      volumeB: this.globalFineVolumeB?.allocatedBytes ?? 0,
    });
    this.info.powerDiagramAllocatedBytes = powerAllocated;
    this.info.allocatedBytes += powerAllocated + fineAllocated;
    this.info.globalFineLevelSetAllocatedBytes += fineAllocated;
    this.info.powerDiagramReady = true;
    this.info.powerDiagramAuthoritative = Boolean(this.powerFaceSeed)
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

  private writeParams() {
    const data = new ArrayBuffer(160);
    new Uint32Array(data, 0, 4).set([this.dims.nx, this.dims.ny, this.dims.nz, this.maxLeafSize]);
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
    const analyticBootstrapSelector = !this.analyticSparseBootstrap
      ? 0
      : this.scene.fluid.initialCondition === "dam-break" ? -20 : -10;
    new Float32Array(data, 112, 4).set([
      this.scene.fluid.density_kg_m3,
      this.scene.fluid.surfaceTension_N_m,
      this.scene.numerics.maxDt_s,
      analyticBootstrapSelector
    ]);
    new Uint32Array(data, 128, 4).set([
      this.pressureCapacity.rowCapacity,
      this.pressureCapacity.entryCapacity,
      this.coarseTaskCapacity,
      1,
    ]);
    new Float32Array(data, 144, 4).set([
      this.scene.fluid.gravity_m_s2.y,
      0,
      0,
      this.scene.container.fillFraction * this.dims.ny,
    ]);
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  setTimestep(dt_s: number) {
    this.powerTimestep_s = Math.max(0, Number.isFinite(dt_s) ? dt_s : 0);
  }

  setCouplingBodies(count: number, hasDynamicBodies: boolean) {
    this.device.queue.writeBuffer(this.params, 44, new Uint32Array([Math.max(0, Math.min(12, Math.floor(count)))]));
    this.device.queue.writeBuffer(this.params, 116, new Float32Array([hasDynamicBodies ? 1 : 0]));
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
      // tiles are analytically non-negative air, while cold analytic phi retains the
      // authored SDF until compact coarse phi has published.
      this.analyticBootstrapWorklist.encode(encoder);
      this.topologyWorklistReady = true;
      this.encodeInlineRebuild(encoder, true);
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

  /** Encode one dependency-ordered t=0 checkpoint. Safe bring-up submits and
   * fences these separately so a driver failure is localized to one bounded
   * phase; product startup appends all checkpoints to one command buffer. */
  encodeInitialSparseAuthorityPhase(encoder: GPUCommandEncoder, phase: OctreeInitialSparseAuthorityPhaseId) {
    switch (phase) {
      case "cold-topology": this.encodeColdBootstrapRebuild(encoder); break;
      case "power-operator-authority": this.encode(
        encoder, this.dims.nx, this.dims.ny, this.dims.nz,
        undefined, "power-operator-only",
      ); break;
      case "surface-global-fine": this.encodeSurface(encoder, 0); break;
      case "section5-face-band-topology": {
        const broker = new PassBroker(encoder);
        this.encodeGlobalFineFaceBandPhase(broker, "topology-build");
        broker.fence("t=0 Section 5 topology checkpoint");
        break;
      }
      case "section5-face-band-transitions": {
        const broker = new PassBroker(encoder);
        this.encodeGlobalFineFaceBandPhase(broker, "transition-adjacency");
        broker.fence("t=0 Section 5 transition checkpoint");
        break;
      }
      case "section5-face-band-closest-point": {
        const broker = new PassBroker(encoder);
        this.encodeGlobalFineFaceBandPhase(broker, "closest-point-extension");
        broker.fence("t=0 Section 5 closest-point checkpoint");
        break;
      }
      case "section5-face-band-power-publication": {
        const broker = new PassBroker(encoder);
        this.encodeGlobalFineFaceBandPhase(broker, "power-publication");
        broker.fence("t=0 Section 5 power publication checkpoint");
        break;
      }
      case "sparse-render-world": this.encodeSparseBrickWorld(encoder); break;
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
    this.powerCoarseLevelSetSchedule?.retireSubmittedEncoder(encoder);
  }

  /**
   * Publish the Section 5 regular-face and transition-tetra velocity band from
   * the current indexed fine generation.  Warmup invokes this after the first
   * fine publication so a paused t=0 scene already owns every interpolation
   * structure required by its first trajectory; regular steps refresh it from
   * the newly projected power velocities before transport.
   */
  private encodeGlobalFineFaceBand(
    encoder: GPUCommandEncoder,
    productionBoundary?: OctreeSemanticBoundary,
  ): GPUCommandEncoder {
    let broker = new PassBroker(encoder);
    for (const phase of OCTREE_FACE_BAND_ENCODE_PHASES) {
      this.encodeGlobalFineFaceBandPhase(broker, phase);
      if (productionBoundary) {
        const semanticPhase: OctreeSemanticPhase = phase === "topology-build"
          ? "faceBandTopologyBuild"
          : phase === "transition-adjacency" ? "faceBandTransitionAdjacency"
          : phase === "closest-point-extension" ? "faceBandClosestPointExtension"
          : "faceBandPowerPublicationCapture";
        broker.fence(`Section 5 semantic trace boundary: ${semanticPhase}`);
        encoder = productionBoundary(semanticPhase, encoder);
        broker = new PassBroker(encoder);
      }
    }
    broker.fence("Section 5 power-face publication committed");
    return encoder;
  }

  /** Encode one caller-fenceable Section 5 face-band checkpoint. */
  private encodeGlobalFineFaceBandPhase(
    broker: PassBroker,
    phase: OctreeFaceBandEncodePhase,
  ) {
    if (!this.globalFineFaceExtension || !this.globalFineBootstrapped || !this.powerVelocity
      || !this.powerFaces || !this.powerTopology || !this.powerCoarseLevelSetSchedule) {
      throw new Error(`Section 5 ${phase} requires the complete native fine/power authority`);
    }
    const fine = this.globalFineCurrentIsA ? this.globalFineSourceA : this.globalFineSourceB;
    const fineTopology = this.globalFineCurrentIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    if (!fine || !fineTopology) {
      throw new Error(`Section 5 ${phase} requires the current fine publication and topology transaction`);
    }
    this.globalFineFaceExtension.encodePhase(broker, {
      fine,
      fineTopologyControl: fineTopology.control,
      owners: this.topology,
      coarsePhiDirectory: this.powerCoarseLevelSetSchedule.sampleSource.directory,
      powerRowVelocities: this.powerVelocity.velocities,
      powerVelocityControl: this.powerVelocity.control,
      powerVelocityGeneration: this.powerGeneration,
      powerTopology: this.powerTopology.source,
      rowDelta: this.powerRowDelta,
      powerFaces: this.powerFaces.source,
      powerRowDirectory: this.powerFaces.source.rowDirectory,
      powerRowDirectoryCapacity: this.powerFaces.plan.rowDirectoryCapacity,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
      maximumLeafSize: this.maxLeafSize,
      generation: this.globalFineGeneration,
      closedTop: this.scene.container.top === "closed",
    }, phase);
    if (phase !== "power-publication") return;
    // The completed regular-to-power round trip is an immutable publication.
    // Complete the paper's round trip. Closest-point extension commits only after its
    // whole narrow-band power subset validates; reconstructing the compact
    // regular field afterwards makes that extrapolated result the next
    // face-transport input, and the exact generalized-face snapshot preserves
    // it across a topology rebuild. On extension failure the power records are
    // untouched, so these guarded publications reproduce the projected
    // rollback rather than exposing partial scratch values.
    if (!this.powerFaceSeed || !this.powerOperator || !this.powerFaceAdvection) {
      throw new Error("Section 5 power publication requires seed, advection, and operator authorities");
    }
    {
      // Aanjaneya et al. (2017), Section 5 extrapolates the regular octree-face
      // field and interpolates it back to the power faces before the next
      // characteristic trace. Reconstruct and retain the complete OLD
      // cell-centred interpolation mesh from that committed extrapolated field;
      // retaining the earlier post-projection/pre-extrapolation vectors leaves
      // air-side departure points without the paper's velocity extension.
      this.powerVelocity.encodeFromFaceControl(broker, {
        faces: this.powerFaces.source.faces,
        faceNormals: this.powerFaces.source.faceNormals,
        incidenceRows: this.powerFaces.source.incidenceRows,
        incidences: this.powerFaces.source.incidence,
      }, this.powerFaces.source.control, {
        maximumIncidencePerRow: OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
        generation: this.powerGeneration,
        projectionControl: this.powerOperator.control,
      });
      this.powerFaceAdvection.encodeCapture(broker, {
        leafHeaders: this.leafHeaders,
        rowVelocities: this.powerVelocity.velocities,
        velocityControl: this.powerVelocity.control,
      });
    }
  }

  encodeInlineRebuild(encoder: GPUCommandEncoder, analyticColdBootstrap = false) {
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
    const active = this.topologyWorklistReady;
    // The owner-page lifecycle consumes the same exact topology-tile
    // publication as the partial topology path. It publishes the complete
    // sorted page set before any refinement kernel may write payload owners;
    // missing support thereafter fails the generation closed.
    if (analyticColdBootstrap) {
      this.ownerPages.encodeAnalyticBootstrap(new PassBroker(encoder));
    } else if (active) {
      this.ownerPages.encode(new PassBroker(encoder));
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
    const pass = broker.compute({ label: "Octree reset and refinement" });
    const dispatch = (full: GPUComputePipeline, delta: GPUComputePipeline) => {
      pass.setPipeline(active ? delta : full);
      pass.setBindGroup(0, this.groups.ab);
      if (active) pass.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      else pass.dispatchWorkgroups(...this.workgroups);
    };
    const dispatchCandidates = (full: GPUComputePipeline, delta: GPUComputePipeline,
      group = this.groups.ab) => {
      pass.setPipeline(active ? delta : full);
      pass.setBindGroup(0, group);
      if (active) pass.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      else pass.dispatchWorkgroups(...this.candidateWorkgroups);
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
      else pass.dispatchWorkgroups(Math.ceil(this.dims.nx / size),
        Math.ceil(this.dims.ny / size), Math.ceil(this.dims.nz / size));
    };
    for (const size of this.refinementSizes) {
      if (size >= 16) {
        dispatchCoarse(size, this.refineCoarsePipelines.get(size)!, this.fineSummarySizingGroup);
      } else {
        const level = this.refineLevelPipelines.get(size)!;
        dispatchCandidates(level.full, level.delta, this.fineSummarySizingGroup);
      }
    }
    for (let round = 0; round < this.balanceRounds; round += 1) {
      for (const size of this.coarseRefinementSizes) {
        dispatchCoarse(size, this.balanceCoarsePipelines.get(size)!);
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
    else candidates.dispatchWorkgroups(...this.candidateWorkgroups);
    candidates.setPipeline(active
      ? this.prefixFrontierCandidateBlocksDeltaPipeline
      : this.prefixFrontierCandidateBlocksPipeline);
    candidates.dispatchWorkgroups(1);
    candidates.setPipeline(active
      ? this.emitFrontierCandidatesDeltaPipeline
      : this.emitFrontierCandidatesPipeline);
    if (active) candidates.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
    else candidates.dispatchWorkgroups(...this.candidateWorkgroups);
    // Candidate emission owns the exact live count. Turn it and the previous
    // frontier count into three compact schedules (sort, carry, merge), then
    // stage them with one pass boundary. A valid zero-delta transaction writes
    // three zero dispatches and keeps the immutable frontier publication.
    candidates.setPipeline(this.prepareFrontierDispatchPipeline);
    candidates.dispatchWorkgroups(1);
    broker.copyBufferToBuffer(this.compaction, 4, this.topologyCandidateDispatch, 0, 36);
    const candidateSort = broker.compute({ label: "Sort dirty frontier candidates by level and Morton" });
    candidateSort.setBindGroup(0, active ? this.fineSummarySizingGroup : this.groups.ab);
    if (this.useLocalFrontierCandidateSort) {
      candidateSort.setPipeline(this.sortFrontierCandidatesLocalPipeline);
      candidateSort.dispatchWorkgroups(1);
    } else {
      for (const pipeline of this.frontierCandidateSortPipelines) {
        candidateSort.setPipeline(pipeline);
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
    // stage them without reopening a compute pass between the two copies.
    broker.copyBufferToBuffer(this.compaction, 48, this.topologyCandidateDispatch, 0, 12);
    broker.copyBufferToBuffer(this.compaction, 4, this.topologyCandidateDispatch, 12, 12);
    broker.fence("octree topology and frontier publication complete");
    return true;
  }

  finishInlineRebuild() { this.info.topologyReuseCount += 1; }
  get pressureSolverLabel() {
    if (this.galerkin) {
      return `Octree power fixed native-L2 Galerkin · ${this.galerkin.plan.levelCount} levels`
        + ` · ${this.galerkin.plan.cycles} bounded V-cycles · no fallback`;
    }
    if (!this.mgpcg || !this.firstOrderVCycle) {
      throw new Error("pressure solver was not constructed");
    }
    const outer = this.mgpcg.executionMode === "persistent-small-domain"
      ? "persistent PCG"
      : "parallel PCG";
    return `Octree power ${outer} · Section 4.3 hybrid · paper sparse-grid pyramid A/B · up to ${this.mgpcg.iterationBudget} iterations · k=${this.mgpcg.boundarySmoothingIterations} paired L2 boundary smoothing · ${this.firstOrderVCycle.plan.levelCount}-level L1 V-cycle`;
  }

  private encodeNativePowerAssembly(
    encoder: GPUCommandEncoder,
    productionBoundary?: OctreeSemanticBoundary,
    sharedBroker?: PassBroker,
  ): GPUCommandEncoder {
    const descriptor = this.powerDescriptor, topology = this.powerTopology, faces = this.powerFaces;
    const operator = this.powerOperator, volumes = this.powerVolumes;
    const volumePipeline = this.powerVolumePipeline;
    const volumeGroup = this.powerVolumeGroup;
    const seed = this.powerFaceSeed, advection = this.powerFaceAdvection;
    if (!descriptor || !topology || !faces || !operator || !volumes || !volumePipeline || !volumeGroup
      || !seed || !advection) {
      throw new Error("Power assembly requires the complete native descriptor/topology/face/operator authority");
    }
    this.energyLedger?.beginStep();
    const oldPowerGeneration = this.powerGeneration;
    const targetPowerGeneration = (this.powerGeneration + 1) >>> 0;
    let broker = sharedBroker ?? new PassBroker(encoder);
    this.energyLedger?.encodeFaceMetric(broker, "oldFaceCapture", oldPowerGeneration, faces.source);
    const splitProductionPhase = (
      phase: OctreeSemanticPhase,
      closeForRawContinuation = false,
    ) => {
      if (productionBoundary) {
        broker.fence(`production phase ${phase}`);
        encoder = productionBoundary(phase, encoder);
        broker = new PassBroker(encoder);
      } else if (closeForRawContinuation) {
        broker.fence(`raw continuation after ${phase}`);
      }
    };
    const dimensions: [number, number, number] = [this.dims.nx, this.dims.ny, this.dims.nz];
    const spacing: [number, number, number] = [
      this.scene.container.width_m / this.dims.nx,
      this.scene.container.height_m / this.dims.ny,
      this.scene.container.depth_m / this.dims.nz,
    ];
    this.powerGeneration = targetPowerGeneration;
    // The first advancing pressure solve is still at the authored t=0
    // interface. Warmup publishes the fine-band data structure, but that
    // administrative publication is not an advection step and must not move
    // the initial Dirichlet surface used by the analytic cold solve.
    const useCurrentFineBoundary = this.globalFineBootstrapped && this.powerAdvancingPressureSteps > 0;
    const useCurrentCoarseBoundary = !useCurrentFineBoundary
      && this.powerCoarseLevelSetBootstrapped && this.powerAdvancingPressureSteps > 0;
    // Frontier membership, L1 capture, and the Power ghost-fluid coefficient
    // must consume one signed-distance generation. Warmup and the first
    // advancing solve retain the authored analytic t=0 field; the next solve
    // cuts over completely to the published fine/coarse authority.
    const analyticBootstrapSelector = !useCurrentFineBoundary && !useCurrentCoarseBoundary
      ? this.scene.fluid.initialCondition === "dam-break" ? -20 : -10
      : 0;
    this.device.queue.writeBuffer(this.params, 124,
      new Float32Array([analyticBootstrapSelector]));
    const boundaryFine = useCurrentFineBoundary
      ? (this.globalFineCurrentIsA ? this.globalFineSourceA : this.globalFineSourceB)
      : undefined;
    this.lastPowerBoundaryFineSource = boundaryFine
      ? { generation: boundaryFine.generation, generationSlot: boundaryFine.generationSlot }
      : undefined;
    // Paper Sections 4.1/5 require free-surface pressure to evaluate signed
    // distance at both actual cell centres.  Before the first fine-band
    // publication the authored analytic field is that authority; recurring
    // generations consume the current two-sided sparse fine field directly.
    // If neither exists, internal boundary publication fails closed in the
    // face builder instead of synthesizing an affine air value.
    // Paper Section 5: beyond the fine narrow band the coarse octree level
    // set is the signed-distance authority at cell centres, so fine-mode
    // boundary sampling resolves band-exterior query centres from the
    // published coarse directory instead of rejecting the generation.
    const boundaryCoarseDirectory = this.powerCoarseLevelSetSchedule?.sampleSource.directory;
    const boundaryPhi = {
      mode: useCurrentFineBoundary ? "fine" as const
        : useCurrentCoarseBoundary ? "coarse" as const : "analytic" as const,
      ...(boundaryFine ? { fine: boundaryFine } : {}),
      ...((useCurrentFineBoundary || useCurrentCoarseBoundary) && boundaryCoarseDirectory
        ? { coarse: { directory: boundaryCoarseDirectory } } : {}),
      container: [this.scene.container.width_m, this.scene.container.height_m,
        this.scene.container.depth_m] as const,
      fillFraction: this.scene.container.fillFraction,
      initialCondition: this.scene.fluid.initialCondition,
    };
    const faceOptions = { dimensions, rowCount: this.compaction,
      physicalCellSize: spacing[0], generation: this.powerGeneration,
      rowDelta: this.powerRowDelta,
      closedBoundaryMask: octreePowerClosedBoundaryMask(this.scene.container.top === "closed"),
      boundaryPhi,
    } as const;
    // Geometry descriptors must come from the octree topology authority, not
    // the phase-row index used to resolve incident pressure rows.  A missing
    // phase row does not mean that the spatial leaf is absent: synthesizing a
    // miss at the querying row's preferred size makes the same air location
    // appear coarse to one row and fine to another, producing impossible,
    // non-reciprocal descriptor pairs.  Owner-page residency includes the
    // bounded face/edge halo required by the paper's local encoding.
    descriptor.encode(broker, this.leafHeaders, this.ownerPages.arena, {
      dimensions, maximumLeafSize: this.maxLeafSize, rowCountBuffer: this.compaction,
      generation: this.powerGeneration,
      rowDelta: {
        rows: this.leafFrontier,
        rowCapacity: this.frontierAllocation.listCapacity,
        controlOffsetWords: this.frontierAllocation.rowDeltaControlOffsetWords,
        newToOldOffsetWords: this.frontierAllocation.rowDeltaNewToOldOffsetWords,
        oldToNewOffsetWords: this.frontierAllocation.rowDeltaOldToNewOffsetWords,
        dirtyRowsOffsetWords: this.frontierAllocation.rowDeltaDirtyRowsOffsetWords,
        affectedRowsOffsetWords: this.frontierAllocation.rowDeltaAffectedRowsOffsetWords,
      },
    });
    topology.encode(broker, descriptor.descriptors, this.compaction, spacing, this.powerRowDelta);
    faces.encode(broker, this.leafHeaders, faceOptions);
    splitProductionPhase("powerDescriptorTopologyFaces");
    // Cold generation 1 is initialized from the authored regular field.  On
    // every recurring generation Aanjaneya et al. (2017), Section 5 instead
    // traces each new generalized-face centroid into the captured OLD power
    // mesh, interpolates the old full vector, and projects onto the new normal.
    // The seed is deliberately encoded first only as cold-start storage; the
    // recurrent advector overwrites every face or invalidates the seed gate.
    seed.encode(broker);
    advection.encodeAdvect(broker, {
      seedControl: seed.control,
      dimensions,
      physicalCellSize: spacing[0],
      maximumLeafSize: this.maxLeafSize,
      generation: this.powerGeneration,
      timestep: this.powerTimestep_s,
    });
    if (this.powerGeneration > 1) {
      if (!this.globalFineBootstrapped || !this.globalFineFaceExtension) {
        throw new Error("Recurring Section 5 advection requires the retained regular-face publication");
      }
      this.globalFineFaceExtension.encodeCompletePowerFaceAdvectionFromRegularBand(broker, {
        faces: faces.source,
        advectionControl: advection.control,
        seedControl: seed.control,
        dimensions,
        maximumLeafSize: this.maxLeafSize,
        physicalCellSize: spacing[0],
        timestep: this.powerTimestep_s,
        // The face band is published before fine transport increments the
        // live generation.  Old-mesh tracing therefore consumes the retained
        // band epoch: generation 2 at cold start, then live minus one.
        fineGeneration: Math.max(2, this.globalFineGeneration - 1),
        powerGeneration: this.powerGeneration,
        powerTopology: topology.source,
        closedTop: this.scene.container.top === "closed",
      });
    }
    this.energyLedger?.encodeFaceMetric(broker, "postRemap", this.powerGeneration, faces.source);
    // Equation (1) splits external forces from advection, while Section 4.1
    // stores the authoritative normal velocity at every power face. Apply
    // gravity in that native basis so oblique transition faces receive g.n dt
    // exactly and closed walls retain their prescribed no-through-flow value.
    seed.encodeAcceleration(broker, [
      this.scene.fluid.gravity_m_s2.x,
      this.scene.fluid.gravity_m_s2.y,
      this.scene.fluid.gravity_m_s2.z,
    ], this.powerTimestep_s);
    this.energyLedger?.encodeFaceMetric(broker, "postGravity", this.powerGeneration, faces.source);
    this.powerSolidVertices?.encode(broker, {
      dimensions,
      physicalSpacing: spacing,
      generation: this.powerGeneration,
      terrainEnabled: sceneHasTerrain(this.scene),
    });
    this.powerSolidFaces?.encodeClassifyAndConstrain(broker, {
      dimensions, physicalSpacing: spacing,
      container: [this.scene.container.width_m, this.scene.container.height_m, this.scene.container.depth_m],
      rigidBodyCount: this.scene.rigidBodies.length,
      terrainEnabled: sceneHasTerrain(this.scene),
      pressureImpulseScale: this.powerTimestep_s,
    });
    this.energyLedger?.encodeFaceMetric(broker, "postSolidConstraint", this.powerGeneration, faces.source);
    splitProductionPhase("powerFaceRegularCompletion");
    const pass = broker.compute({ label: "Publish physical power-cell volumes" });
    pass.setPipeline(volumePipeline); pass.setBindGroup(0, volumeGroup);
    pass.dispatchWorkgroupsIndirect(
      faces.source.liveFaceDispatch,
      OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES,
    );
    operator.encodeAssemblyFromControl(broker, faces.faces, faces.source, faces.control,
      this.powerFaceSeed?.control,
      this.powerSolidFaces?.control);
    splitProductionPhase("powerOperatorRhsAssembly");
    operator.encodeLeafRowPublication(broker, this.leafHeaders, this.leafEntries);
    splitProductionPhase("finalPressureRowAssembly", true);
    return encoder;
  }

  private encodeNativePowerProjection(
    broker: PassBroker,
    pressure: GPUBuffer,
  ): void {
    if (!this.powerFaces || !this.powerOperator) {
      throw new Error("Power projection requires the native face and operator authorities");
    }
    const solverControl = this.galerkin?.control ?? this.mgpcg?.control;
    if (!solverControl) throw new Error("Power projection requires one immutable pressure solver");
    this.powerOperator.encodeProjectionFromControl(broker, this.powerFaces.faces, this.powerFaces.source,
      pressure, this.powerFaces.control, 1, solverControl);
    // Ordered compute dispatches publish the projected face field directly to
    // the diagnostic and solid consumers below; no pass transition is needed.
    this.energyLedger?.encodeFaceMetric(broker, "postProjection", this.powerGeneration, this.powerFaces.source);
    this.powerSolidFaces?.encodePostProjectionConstraint(broker);
  }

  private encodePowerVelocityPublication(broker: PassBroker): void {
    if (!this.powerFaceSeed || !this.powerOperator
      || !this.powerFaces || !this.powerVelocity) {
      throw new Error("Power velocity publication requires the complete native face/operator/vector authority");
    }
    const faces = this.powerFaces.source;
    this.powerVelocity.encodeFromFaceControl(broker, {
      faces: faces.faces,
      faceNormals: faces.faceNormals,
      incidenceRows: faces.incidenceRows,
      incidences: faces.incidence,
    }, faces.control, {
      maximumIncidencePerRow: OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
      generation: this.powerGeneration,
      projectionControl: this.powerOperator.control,
    });
    // The reconstructed native power-cell vectors are consumed directly by
    // Section 5. No Cartesian staggered-field republish exists at this seam.
  }

  private encodeFrontierRows(
    encoder: GPUCommandEncoder,
    label: string,
    group = this.groups.ab,
  ): void {
    const broker = new PassBroker(encoder);
    const deltaPrepare = broker.compute({ label: `${label} row-delta prepare` });
    deltaPrepare.setBindGroup(0, group);
    deltaPrepare.setPipeline(this.prepareRowDeltaPipeline);
    deltaPrepare.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    const deltaClassify = broker.compute({ label: `${label} exact row-delta classification` });
    deltaClassify.setBindGroup(0, group);
    deltaClassify.setPipeline(this.classifyRowDeltaPipeline);
    deltaClassify.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    deltaClassify.setPipeline(this.finalizeRowDeltaClassificationPipeline);
    deltaClassify.dispatchWorkgroups(1);
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
    deltaPublish.setPipeline(this.markRowDeltaRingPipeline);
    deltaPublish.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
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
    deltaFinalize.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 12);
    broker.copyBufferToBuffer(this.compaction, 8, this.solveDispatch, 0, 24);
  }

  encode(
    encoder: GPUCommandEncoder,
    _nx: number,
    _ny: number,
    _nz: number,
    options?: {
      productionBoundary?: OctreeSemanticBoundary;
    },
    scope: "complete" | "power-operator-only" = "complete",
  ): GPUCommandEncoder {
    const solveBudget = this.mgpcg?.iterationBudget ?? this.galerkin?.plan.cycles ?? 0;
    this.info.pressureIterationBudget = solveBudget;
    this.info.pressureIterationHardBudget = solveBudget;
    const splitProductionPhase = (phase: OctreeSemanticPhase) => {
      if (options?.productionBoundary) {
        encoder = options.productionBoundary(phase, encoder);
      }
    };
    // Compact and assemble L1 once, then replace it with the power operator
    // before the sole Section 4.3 MGPCG authority runs.
    const remapGroup = this.latestPressureInA ? this.groups.ab : this.groups.ba;
    this.encodeFrontierRows(encoder, "Octree leaf compaction", remapGroup);
    const initialInA = !this.latestPressureInA;
    const assemblyGroup = initialInA ? this.groups.ab : this.groups.ba;
    const pressureBroker = new PassBroker(encoder);
    const pressure = pressureBroker.compute({ label: "Octree leaf pressure assembly" });
    pressure.setPipeline(this.assemblePipeline);
    pressure.setBindGroup(0, assemblyGroup);
    pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
    pressure.setPipeline(this.assembleCoarsePipeline);
    pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 12);
    this.firstOrderVCycle?.encodeCapture(pressureBroker);
    if (options?.productionBoundary) {
      pressureBroker.fence("pressure capture trace boundary");
    }
    splitProductionPhase("pressureLeafCompactionL1Capture");
    encoder = this.encodeNativePowerAssembly(
      encoder,
      options?.productionBoundary,
      options?.productionBoundary ? undefined : pressureBroker,
    );
    const pressureIn = initialInA ? this.pressureA : this.pressureB;
    const pressureOut = initialInA ? this.pressureB : this.pressureA;
    if (!this.powerOperator) throw new Error("Pressure solve requires the native power operator authority");
    const solveBroker = new PassBroker(encoder);
    if (this.galerkin) {
      this.galerkin.encode(solveBroker, {
        liveOperator: {
          leafHeaders: this.leafHeaders,
          leafEntries: this.leafEntries,
          authorityControl: this.powerOperator.control,
        },
        initialCorrection: pressureIn,
        correction: pressureOut,
      });
    } else if (this.mgpcg) {
      this.mgpcg.encode(solveBroker, pressureIn, pressureOut, this.powerOperator.control);
    } else {
      throw new Error("No pressure solver was selected");
    }
    this.latestPressureInA = !initialInA;
    // Stage solve feedback (residual sums + row/entry counts) while this
    // encoder still owns write ordering on compaction; the async diagnostics
    // poll then reads the staging buffer without racing the next rebuild.
    solveBroker.copyBufferToBuffer(
      this.compaction, this.compactionByteLength - 32, this.solveStats, 0, 32);
    splitProductionPhase("mgpcgSolve");
    const finalInA = this.latestPressureInA;
    const projectionBroker = new PassBroker(encoder);
    this.encodeNativePowerProjection(
      projectionBroker,
      finalInA ? this.pressureA : this.pressureB,
    );
    this.encodePowerVelocityPublication(projectionBroker);
    projectionBroker.fence("projected native power velocity published");
    if (scope === "power-operator-only") {
      // The explicit t=0 dependency chain publishes the fine level set next,
      // then owns each Section 5 transaction in its own named checkpoint.
      // This is a lifecycle boundary, not an alternate simulation path.
      return encoder;
    }
    // Paper Section 5 velocity extrapolation: projected power faces are first
    // reconstructed onto regular octree faces, constrained at solids, then
    // extended from physical closest points before factor-m
    // trajectories are traced.
    splitProductionPhase("powerProjectionPublication");
    encoder = this.encodeGlobalFineFaceBand(encoder, options?.productionBoundary);
    const projectionTailBroker = new PassBroker(encoder);
    if (this.powerFaces) this.energyLedger?.encodeFaceMetric(
      projectionTailBroker, "postFaceBandPublication", this.powerGeneration, this.powerFaces.source,
    );
    this.powerSolidFaces?.encodePressureImpulses(projectionTailBroker, finalInA);
    projectionTailBroker.fence("projection diagnostics and solid pressure exchange published");
    this.encodeOverlayMaterialization(encoder, finalInA);
    if (this.powerTimestep_s > 0) this.powerAdvancingPressureSteps += 1;
    splitProductionPhase("powerProjectionTail");
    return encoder;
  }

  /** Publish lazily allocated diagnostic textures from the live owner map.
   * The first overlay request materializes immediately, so reset-time grid
   * inspection never decodes zero-initialized topology storage as finest 1^3. */
  encodeOverlayMaterialization(encoder: GPUCommandEncoder, pressureInA = this.latestPressureInA) {
    if (!this.diagnosticGroups) return false;
    const broker = new PassBroker(encoder);
    const materialize = broker.compute({ label: "Materialize octree overlay fields" });
    materialize.setPipeline(this.materializePipeline);
    materialize.setBindGroup(0, pressureInA ? this.diagnosticGroups.pressureA : this.diagnosticGroups.pressureB);
    materialize.dispatchWorkgroups(...this.workgroups);
    broker.fence("octree overlay fields materialized");
    return true;
  }

  /** Returns the encoder owning any work after the final production boundary.
   * Segmented boundary callbacks finish the encoder they receive, so callers
   * supplying productionBoundary must continue encoding on the returned
   * encoder, never on the argument. */
  encodeSurface(encoder: GPUCommandEncoder, dt_s: number, inflow?: SurfaceInflowState,
    _maximumDt_s?: number, productionBoundary?: (phase: "finePreparation" | "fineTransport" | "fineTopology" | "fineRedistance"
      | "fineRestriction", encoder: GPUCommandEncoder) => GPUCommandEncoder): GPUCommandEncoder {
    const splitProductionPhase = (phase: "finePreparation" | "fineTransport" | "fineTopology" | "fineRedistance"
      | "fineRestriction") => {
      if (productionBoundary) encoder = productionBoundary(phase, encoder);
    };
    if (this.fineSeedAdapter) {
      let coarseBootstrappedThisStep = false;
      const preparationBroker = new PassBroker(encoder);
      // Fine seeds are rebuilt from current compact coarse phi before each
      // global-fine publication transaction.
      this.fineSeedAdapter.encode(preparationBroker);
      if (this.powerCoarseLevelSet && this.powerCoarseLevelSetSchedule && this.powerVelocity && this.powerFaces) {
        if (!this.powerCoarseLevelSetBootstrapped) {
          this.powerCoarseLevelSet.encodeBootstrapFromSurfaceLeaves(
            preparationBroker, this.fineSeedAdapter.leaves,
          );
          this.powerCoarseLevelSetSchedule.encode(preparationBroker, {
            headers: this.leafHeaders, cellVelocities: this.powerVelocity.velocities,
            rowDirectory: this.powerFaces.source.rowDirectory, rowCount: this.compaction,
          }, {
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            dt: 0, rowDirectoryCapacity: this.powerFaces.plan.rowDirectoryCapacity,
            maximumLeafSize: this.maxLeafSize,
            generation: this.powerCoarseLevelSetGeneration & 0x3fff_ffff,
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
        const seedBroker = preparationBroker;
        const seeds = this.globalFineBootstrapped
          ? this.globalFineSeeds.encode(
            seedBroker,
            { buffer: this.fineSeedAdapter.leaves },
            { buffer: this.fineSeedAdapter.source.candidates.candidates },
            { buffer: this.fineSeedAdapter.source.candidateCount },
          )
          : this.globalFineSeeds.encodeFromAllInterfaceLeaves(
            seedBroker, { buffer: this.fineSeedAdapter.leaves }, { buffer: this.compaction },
            this.powerFaces ? {
              queries: { buffer: this.powerFaces.source.boundaryPhiQueries },
              control: { buffer: this.powerFaces.source.control },
            } : undefined,
          );
        const compactCoarseEntry: GPUBindGroupEntry = { binding: 9,
          resource: { buffer: this.powerCoarseLevelSetSchedule!.sampleSource.directory } };
        const bandCells = Math.min(256, Math.max(4,
          this.interfaceRefinementBandCells * (this.globalFineLevelSet?.plan.fineFactor ?? 4)));
        // Match allocation planning above. The final three cells cover the
        // complete 3-D trilinear stencil and its centre on the closed cutoff.
        const redistanceBandCells = Math.min(256,
          bandCells + this.globalFineLevelSet!.plan.fineFactor + 3);
        const transport = this.globalFineCurrentIsA ? this.globalFineTransportA : this.globalFineTransportB;
        let transportEncoded = false;
        // Adapter publication, coarse bootstrap and compact interface seeding
        // precede characteristic transport. Keep them out of the transport
        // bucket so the generic trace names the measured work.
        seedBroker.fence("fine interface seed publication complete");
        splitProductionPhase("finePreparation");
        if (this.globalFineBootstrapped && transport && this.powerFaceSeed && this.powerVelocity) {
          const transportBroker = new PassBroker(encoder);
          this.energyLedger?.encodeFinePotential(transportBroker, "preFineTransport", transport.source);
          this.lastGlobalFineTransport = transport;
          const completedTransportBroker = transport.encode(transportBroker, {
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
            openTopBoundary: this.scene.container.top !== "closed",
            transportBandCells: Math.min(256, Math.max(4,
              this.interfaceRefinementBandCells * (this.globalFineLevelSet?.plan.fineFactor ?? 4))),
          });
          this.energyLedger?.encodeFinePotential(completedTransportBroker, "postFineTransport", transport.source);
          // Topology may reuse the shared physical payload pool. Capture the
          // transported old phi by logical sample before that reuse, then
          // intersect it with the new generation after topology publication.
          this.energyLedger?.encodeFineCommonCapture(completedTransportBroker, transport.source);
          encoder = completedTransportBroker.commandEncoder();
          transportEncoded = true;
          splitProductionPhase("fineTransport");
        }
        let publicationTopology: WebGPUFineLevelSetTopology;
        let publicationRedistance: WebGPUFineLevelSetRedistance;
        let publicationVolume: WebGPUFineLevelSetVolumeCorrection | undefined;
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
          const topologyBroker = new PassBroker(encoder);
          publicationTopology.encode(topologyBroker, seeds, [compactCoarseEntry], {
            // The octree timestep is bounded at one finest effective cell.
            // Express that same physical displacement on the fine lattice.
            maximumBacktraceFineCells: this.globalFineLevelSet!.plan.fineFactor,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells: redistanceBandCells,
            safetyBrickRings: 1,
          }, true, this.globalFineBootstrapped
            ? { kind: "delta", producer: publicationTransport!.topologyDelta }
            : { kind: "bootstrap" });
          this.energyLedger?.encodeFinePotential(topologyBroker, "postFineTopology", this.globalFineSourceB!);
          if (publicationTransport) this.energyLedger?.encodeFineCommonTopologyPair(
            topologyBroker, publicationTransport.source, this.globalFineSourceB!,
          );
          encoder = topologyBroker.commandEncoder();
          splitProductionPhase("fineTopology");
          const redistanceBroker = new PassBroker(encoder);
          publicationRedistance.encode(redistanceBroker, { bandCells: redistanceBandCells, residualTolerance: 1 });
          this.energyLedger?.encodeFinePotential(redistanceBroker, "postFineRedistance", this.globalFineSourceB!);
          if (publicationTransport) this.energyLedger?.encodeFineCommonPotential(
            redistanceBroker, "postFineRedistanceCommon",
            this.globalFineSourceB!.generation, this.globalFineSourceB!,
          );
          publicationVolume?.encode(redistanceBroker);
          this.energyLedger?.encodeFinePotential(
            redistanceBroker, "postFineVolumeCorrection", this.globalFineSourceB!,
          );
          encoder = redistanceBroker.commandEncoder();
          splitProductionPhase("fineRedistance");
        } else {
          this.globalFineGeneration += 1;
          this.globalFineLevelSet!.repurposeGPUGeneration(this.globalFineSourceA!, this.globalFineGeneration);
          publicationTopology = this.globalFineTopologyBA;
          publicationRedistance = this.globalFineRedistanceA;
          publicationVolume = this.globalFineVolumeA;
          const topologyBroker = new PassBroker(encoder);
          publicationTopology.encode(topologyBroker, seeds, [compactCoarseEntry], {
            maximumBacktraceFineCells: this.globalFineLevelSet!.plan.fineFactor,
            interpolationSupportFineCells: 1,
            redistanceBandFineCells: redistanceBandCells,
            safetyBrickRings: 1,
          }, true, this.globalFineBootstrapped
            ? { kind: "delta", producer: publicationTransport!.topologyDelta }
            : { kind: "bootstrap" });
          this.energyLedger?.encodeFinePotential(topologyBroker, "postFineTopology", this.globalFineSourceA!);
          if (publicationTransport) this.energyLedger?.encodeFineCommonTopologyPair(
            topologyBroker, publicationTransport.source, this.globalFineSourceA!,
          );
          encoder = topologyBroker.commandEncoder();
          splitProductionPhase("fineTopology");
          const redistanceBroker = new PassBroker(encoder);
          publicationRedistance.encode(redistanceBroker, { bandCells: redistanceBandCells, residualTolerance: 1 });
          this.energyLedger?.encodeFinePotential(redistanceBroker, "postFineRedistance", this.globalFineSourceA!);
          if (publicationTransport) this.energyLedger?.encodeFineCommonPotential(
            redistanceBroker, "postFineRedistanceCommon",
            this.globalFineSourceA!.generation, this.globalFineSourceA!,
          );
          publicationVolume?.encode(redistanceBroker);
          this.energyLedger?.encodeFinePotential(
            redistanceBroker, "postFineVolumeCorrection", this.globalFineSourceA!,
          );
          encoder = redistanceBroker.commandEncoder();
          splitProductionPhase("fineRedistance");
        }
        const restrictionBroker = new PassBroker(encoder);
        publicationTopology.encodeFinalizePublication(restrictionBroker, {
          redistance: publicationRedistance.control,
          ...(publicationVolume ? { volume: publicationVolume.control } : {}),
          ...(publicationTransport ? { transport: publicationTransport.control } : {}),
        });
        const correctedFine = this.globalFineCurrentIsA ? this.globalFineSourceB : this.globalFineSourceA;
        if (correctedFine && this.fineToPowerCoarseLevelSet && this.powerCoarseLevelSetSchedule
          && this.powerVelocity && this.powerFaces) {
          const correction = this.fineToPowerCoarseLevelSet.encode(restrictionBroker, correctedFine, {
            headers: this.leafHeaders, rowDirectory: this.powerFaces.candidateRowDirectoryForCoarse,
            rowCount: this.compaction,
            topologyControl: publicationTopology.control,
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            maximumLeafSize: this.maxLeafSize,
            rowDirectoryCapacity: this.powerFaces.plan.rowDirectoryCapacity,
          });
          this.powerCoarseLevelSetSchedule.encode(restrictionBroker, {
            headers: this.leafHeaders, cellVelocities: this.powerVelocity.velocities,
            rowDirectory: this.powerFaces.candidateRowDirectoryForCoarse, rowCount: this.compaction,
            fineCorrection: { rowOffsets: correction.rowOffsets, contributions: correction.contributions,
              contributionCount: correction.counts, aggregated: correction.aggregated },
          }, {
            dimensions: [this.dims.nx, this.dims.ny, this.dims.nz],
            physicalCellSize: this.scene.container.width_m / this.dims.nx,
            dt: coarseBootstrappedThisStep ? 0 : dt_s,
            rowDirectoryCapacity: this.powerFaces.plan.rowDirectoryCapacity,
            maximumLeafSize: this.maxLeafSize, generation: correctedFine.generation & 0x3fff_ffff,
          });
          this.powerCoarseLevelSetGeneration = correctedFine.generation & 0x3fff_ffff;
        }
        if (correctedFine && this.powerCoarseLevelSetSchedule) {
          const coarse = this.powerCoarseLevelSetSchedule.sampleSource;
          this.globalFineSummaries?.encode(restrictionBroker, correctedFine, {
            buffer: publicationTopology.pageDelta,
            layout: publicationTopology.pageDeltaLayout,
          },
            { directory: coarse.directory, control: coarse.control,
              delta: coarse.delta, deltaHeaderWords: coarse.deltaHeaderWords,
              deltaRecordWords: coarse.deltaRecordWords });
        }
        encoder = restrictionBroker.commandEncoder();
        const publicationTargetIsA = !this.globalFineCurrentIsA;
        // Register on the encoder that owns finalize/restriction before a
        // production boundary can finish and replace it. Public parity moves
        // only when retireSubmittedEncoder sees this exact encoder submitted.
        this.globalFinePublicationByEncoder.set(encoder, publicationTargetIsA);
        splitProductionPhase("fineRestriction");
        this.globalFineCurrentIsA = publicationTargetIsA;
        this.globalFineBootstrapped = true;
      } else {
        throw new Error("Authoritative Section 5 fine-band pipeline is incomplete");
      }
      return encoder;
    }
    return encoder;
  }
  addSurfaceReferenceVolumeCells(cells: number) { this.surfaceState.addReferenceVolumeCells(cells); }
  async readSolveDiagnostics() {
    // The staging buffer was copied inside the solve encoder, so it can never
    // race the next rebuild's worklist copy over the compaction header. It
    // carries [overflow, required rows, required entries, fallback dispatch xyz,
    // sum r^2, sum b^2] from the latest solve.
    const solverControl = this.galerkin?.control ?? this.mgpcg?.control;
    const solverBytes = solverControl ? 64 : 0;
    const readback = this.device.createBuffer({
      label: "Octree live pressure-row diagnostics",
      size: 32 + solverBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read octree pressure-row diagnostics" });
    encoder.copyBufferToBuffer(this.solveStats, 0, readback, 0, 32);
    if (solverControl) encoder.copyBufferToBuffer(solverControl, 0, readback, 32, 64);
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
      } else {
        this.residualRms = undefined;
        this.initialResidualRms = undefined;
        this.relativeResidual = undefined;
      }
      if (solverControl) this.applyMGPCGDiagnostics(new Uint32Array(mapped, 32, 16));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** One-time startup proof for the paper's Section 4.3 pressure authority.
   * Regular simulation scheduling never consumes this readback; the paused
   * t=0 transport gate uses it only after every initialization phase fenced. */
  async readMGPCGDiagnostics() {
    const solverControl = this.galerkin?.control ?? this.mgpcg?.control;
    if (!solverControl) return undefined;
    const readback = this.device.createBuffer({
      label: "Octree t=0 MGPCG authority diagnostics",
      size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read octree t=0 MGPCG authority" });
    encoder.copyBufferToBuffer(solverControl, 0, readback, 0, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = Uint32Array.from(new Uint32Array(readback.getMappedRange(0, 64)));
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
    const rr = floats[4], bb = floats[5];
    this.info.pressureIterationsUsed = words[2];
    this.info.pressureConverged = words[0] === 0 && words[1] !== 0;
    if (Number.isFinite(rr) && Number.isFinite(bb) && rr >= 0 && bb >= 0) {
      const rows = Math.max(1, words[3]);
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
  encodeBodyImpulseReadback() { return undefined; }
  readBodyImpulseReadback() { return Promise.resolve([]); }
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
      // Inspection consumes the current immutable frontier. It must never
      // mutate topology or revive the cold full-domain builder merely because
      // a diagnostic view was opened.
      this.compactVoxelInspection.source.inspectionPublication?.setEnabled(true);
      const encoder = this.device.createCommandEncoder({ label: "Bootstrap compact octree voxel inspection" });
      this.encodeFrontierRows(encoder, "Octree inspection frontier rows",
        this.latestPressureInA ? this.groups.ab : this.groups.ba);
      this.compactVoxelInspection.encode(encoder);
      this.device.queue.submit([encoder.finish()]);
    }
    return this.compactVoxelInspection.source;
  }
  get powerFaceSeedControl() { return this.powerFaceSeed?.control; }
  /** QA-only Section 5 recurrent old-mesh advection publication status. */
  get powerFaceAdvectionControl() { return this.powerFaceAdvection?.control; }
  /** QA-only generalized solid-aperture prerequisite consumed by row assembly. */
  get powerSolidFaceControl() { return this.powerSolidFaces?.control; }
  get powerOperatorControl() { return this.powerOperator?.control; }
  /** QA-only MGPCG status; simulation authority consumes this buffer directly on GPU. */
  get mgpcgControl() { return this.galerkin?.control ?? this.mgpcg?.control; }
  get powerFaceControl() { return this.powerFaces?.control; }
  /** QA-only rejected candidate header; never a simulation authority. */
  get powerFaceCandidateControl() { return this.powerFaces?.candidateControlForDiagnostics; }
  /** QA-only generalized face records used to localize recurrent advection failures. */
  get powerFaceSource() { return this.powerFaces?.source; }
  /** Explicit one-shot readback of the opt-in observational energy ring. */
  readEnergyLedger(): Promise<OctreeEnergyLedgerSnapshot> | undefined { return this.energyLedger?.read(); }
  /** QA-only exact liquid/air cell-centre queries authored before boundary-phi sampling. */
  get powerBoundaryPhiQueries() { return this.powerFaces?.source.boundaryPhiQueries; }
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
  get powerFaceRowDirectory() { return this.powerFaces?.source.rowDirectory; }
  get powerDescriptorControl() { return this.powerDescriptor?.control; }
  get powerTopologyControl() { return this.powerTopology?.control; }
  get powerDescriptorRows() { return this.powerDescriptor?.descriptors; }
  get powerTopologyMetrics() { return this.powerTopology?.metrics; }
  get powerCatalogEntryHeaders() { return this.powerTopology?.catalogEntryHeaders; }
  get powerCatalogFaces() { return this.powerTopology?.catalogFaces; }
  get techniqueDebugSource() {
    const surface = this.fineSeedAdapter?.source;
    const topology = this.powerTopology?.source;
    const faces = this.powerFaces?.source;
    const tetrahedronHeaders = topology?.catalogTetrahedronHeaders;
    const tetrahedra = topology?.catalogTetrahedra;
    const tetrahedronVertices = topology?.catalogTetrahedronVertices;
    if (!surface || !topology || !faces || !tetrahedronHeaders || !tetrahedra || !tetrahedronVertices) return undefined;
    const fine = this.globalFineBootstrapped
      ? (this.globalFinePublishedIsA ? this.globalFineSourceA : this.globalFineSourceB)
      : undefined;
    const fineTopology = this.globalFinePublishedIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    const fineRedistance = this.globalFinePublishedIsA ? this.globalFineRedistanceA : this.globalFineRedistanceB;
    const fineBandLifecycle = fine && fineTopology && fineRedistance ? {
      params: { buffer: fine.params },
      metadata: { buffer: fine.metadata },
      worklist: { buffer: fine.worklist },
      sampleFlags: { buffer: fine.flags },
      phi: { buffer: fine.phi },
      topologyControl: { buffer: fineTopology.control },
      redistanceControl: { buffer: fineRedistance.control },
    } : undefined;
    const faceBand = this.globalFineFaceExtension?.source;
    const section5FaceBand = faceBand ? {
      rowDirectory: { buffer: faceBand.rowDirectory }, rows: { buffer: faceBand.rows },
      faces: { buffer: faceBand.faces }, incidence: { buffer: faceBand.incidence },
      control: { buffer: faceBand.control },
      transitionControl: { buffer: faceBand.transitionControl },
    } : undefined;
    return {
      leaves: { buffer: surface.leaves },
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
  /** Exposes the already-live Section 5 cell-vector reconstruction to bounded
   * smoke readback; this does not allocate or materialize a dense GPU field. */
  get powerCellVelocityBuffer() { return this.powerVelocity?.velocities; }
  get powerLeafEntries() { return this.leafEntries; }
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
  async readPowerFrontierFailure(): Promise<Record<string, readonly number[]>> {
    const readback = this.device.createBuffer({
      label: "Octree power-frontier failure readback",
      size: 128,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read octree power-frontier failure headers",
    });
    encoder.copyBufferToBuffer(this.leafFrontier, 0, readback, 0, 64);
    encoder.copyBufferToBuffer(this.compaction, 0, readback, 64, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return {
        frontier: Array.from(words.slice(0, 16)),
        compaction: Array.from(words.slice(16, 32)),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
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
    if (!this.powerFaces || !Number.isSafeInteger(row) || row < 0
      || row >= this.powerFaces.plan.rowDirectoryCapacity) return undefined;
    const arena = this.ownerPages.arena;
    const readback = this.device.createBuffer({
      label: "Octree Power-row owner-page failure readback",
      size: arena.size + 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read Octree Power-row owner-page failure",
    });
    encoder.copyBufferToBuffer(this.powerFaces.source.rowDirectory, row * 16, readback, 0, 16);
    encoder.copyBufferToBuffer(arena, 0, readback, 16, arena.size);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const directory = Array.from(words.slice(0, 4));
      const owner = words.subarray(4);
      const cell = (directory[0] ?? 0) - 1;
      if (directory[0] === 0 || cell < 0) {
        return { row, directory, ownerControl: Array.from(owner.slice(0, 16)) };
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
        row, directory, q, logical, record, encodedPage, payloadWord,
        ownerControl: Array.from(owner.slice(0, 16)),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  /** Failure-only native generalized-face seed publication header. */
  async readPowerSeedChainControls(): Promise<Record<string, readonly number[]> | undefined> {
    if (!this.powerFaceSeed) return undefined;
    const readback = this.device.createBuffer({
      label: "Octree Power seed-chain control readback",
      size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read Octree Power seed-chain controls",
    });
    encoder.copyBufferToBuffer(this.powerFaceSeed.control, 0, readback, 0, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return {
        powerFaceSeed: Array.from(words.slice(0, 16)),
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
      maximumLeafSize: this.maxLeafSize,
      dimensions: [this.dims.nx, this.dims.ny, this.dims.nz] };
  }
  /** QA-only compact surface producer header feeding recurring topology residency. */
  get fineSeedCandidateControl() { return this.fineSeedAdapter?.source.candidateCount; }
  /** QA-only compact affine leaves classified by the recurring topology producer. */
  get fineSeedLeaves() { return this.fineSeedAdapter?.source.leaves; }
  get powerOwnerArena() { return this.ownerPages.arena; }
  get nativePowerVelocityAuthority() {
    return Boolean(this.powerFaces && this.powerFaceSeed && this.powerFaceAdvection && this.powerVelocity);
  }
  /** Authoritative narrow-band fine phi for rendering and surface transport.
   * Topology sizing and pressure fractions still require the terminal coarse-phi cutover. */
  get globalFineLevelSetSource(): WebGPUFineLevelSetBrickSource | undefined {
    if (!this.globalFineLevelSet || !this.globalFineBootstrapped) return undefined;
    const fine = this.globalFinePublishedIsA ? this.globalFineSourceA : this.globalFineSourceB;
    if (!fine) return undefined;
    const coarse = this.powerCoarseLevelSetSchedule?.sampleSource;
    const topology = this.globalFinePublishedIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    return { ...fine,
      ...(coarse ? { coarsePhiDirectory: coarse.directory, coarsePhiRowCapacity: coarse.rowCapacity } : {}),
      ...(topology ? { topologyControl: topology.control } : {}),
      ...(this.globalFineSeeds ? { seedControl: this.globalFineSeeds.buffer } : {}),
    };
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
      metadata: fine.metadata,
      worklist: fine.worklist,
      phi: fine.phi,
      workA: fine.workA,
      rollbackPhi: fine.rollbackPhi,
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
  /** Diagnostic-only Stage-A reconstruction status used by fine transport. */
  get globalFinePowerVelocityControl(): GPUBuffer | undefined { return this.powerVelocity?.control; }
  get globalFinePowerProjectionControl(): GPUBuffer | undefined { return this.powerOperator?.control; }
  /** Diagnostic-only Stage-B point-sampler status used by fine transport. */
  get globalFinePowerVelocitySampleControl(): GPUBuffer | undefined {
    return this.globalFineVelocityPrepass?.source.control;
  }
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
      || !this.powerTopology || !this.powerFaces) return undefined;
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
    encoder.copyBufferToBuffer(this.powerFaces.source.rowDirectory,
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
        rowDirectory: { cellPlusOne: words[36], row: words[37], size: words[38], morton: words[39] },
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  /** Diagnostic-only bounded readback for a rejected reciprocal power-face
   * pair. The candidate header supplies both exact row IDs; this method never
   * participates in topology or publication decisions. */
  async readPowerFaceCandidateFailurePair(row: number, neighbor: number) {
    if (!Number.isSafeInteger(row) || row < 0 || row >= this.pressureCapacity.rowCapacity
      || !Number.isSafeInteger(neighbor) || neighbor < 0
      || neighbor >= this.pressureCapacity.rowCapacity || !this.powerTopology
      || !this.powerDescriptor) return undefined;
    const readback = this.device.createBuffer({
      label: "Rejected power-face reciprocal pair QA",
      size: 136,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read rejected power-face reciprocal pair",
    });
    encoder.copyBufferToBuffer(this.leafHeaders, row * 48, readback, 0, 48);
    encoder.copyBufferToBuffer(this.powerTopology.metrics, row * 16, readback, 48, 16);
    encoder.copyBufferToBuffer(this.powerDescriptor.descriptors, row * 4, readback, 64, 4);
    encoder.copyBufferToBuffer(this.leafHeaders, neighbor * 48, readback, 68, 48);
    encoder.copyBufferToBuffer(this.powerTopology.metrics, neighbor * 16, readback, 116, 16);
    encoder.copyBufferToBuffer(this.powerDescriptor.descriptors, neighbor * 4, readback, 132, 4);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange().slice(0);
      const record = (offsetBytes: number, descriptorOffsetBytes: number, index: number) => {
        const words = new Uint32Array(bytes, offsetBytes, 16);
        const floats = new Float32Array(bytes, offsetBytes, 16);
        return {
          row: index,
          header: {
            cell: words[0], entryStart: words[1], entryCount: words[2], size: words[3],
            diagonal: floats[4], rhs: floats[5], gradient: Array.from(floats.slice(8, 12)),
          },
          metric: {
            topologyCode: words[12], transformAndFlags: words[13], volume: floats[14],
          },
          descriptor: new Uint32Array(bytes, descriptorOffsetBytes, 1)[0],
        };
      };
      return { row: record(0, 64, row), neighbor: record(68, 132, neighbor) };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  readPowerFaceCandidateFailure(faceIndex: number) {
    return this.powerFaces?.readCandidateFailure(faceIndex);
  }
  async readPowerDescriptorCandidateFailure(row: number) {
    if (!this.powerDescriptor || !Number.isSafeInteger(row) || row < 0
      || row >= this.pressureCapacity.rowCapacity) return undefined;
    const [descriptor, rowRecord] = await Promise.all([
      this.powerDescriptor.readCandidateFailure(row),
      this.readPowerCoarseFailureRow(row),
    ]);
    return { descriptor, row: rowRecord };
  }
  /** Diagnostic-only Section 5 face-band status; never participates in publication decisions on the CPU. */
  get globalFineFaceBandControl(): GPUBuffer | undefined { return this.globalFineFaceExtension?.control; }
  /** Rejection-only candidate header; the simulation never consumes it. */
  get globalFineFaceBandCandidateControl(): GPUBuffer | undefined {
    return this.globalFineFaceExtension?.candidateControlForDiagnostics;
  }
  /** Diagnostic-only catalog-Delaunay transition gate; never participates in CPU authority selection. */
  get globalFineFaceBandTransitionControl(): GPUBuffer | undefined {
    return this.globalFineFaceExtension?.transitionControl;
  }
  /** Rejection-only candidate transition header. */
  get globalFineFaceBandCandidateTransitionControl(): GPUBuffer | undefined {
    return this.globalFineFaceExtension?.candidateTransitionControlForDiagnostics;
  }
  get globalFineFaceBandPointFieldControl(): GPUBuffer | undefined {
    return this.globalFineFaceExtension?.pointFieldControl;
  }
  get globalFineFaceBandTransientPowerControl(): GPUBuffer | undefined {
    return this.globalFineFaceExtension?.transientPowerControl;
  }
  get globalFineFaceBandPowerPublicationControl(): GPUBuffer | undefined {
    return this.globalFineFaceExtension?.powerPublicationControl;
  }
  get globalFineFaceBandPlan() { return this.globalFineFaceExtension?.plan; }
  readGlobalFineBandRowFailure(index: number) {
    return this.globalFineFaceExtension?.readBandRowFailure(index);
  }
  readGlobalFineCandidateBandRowFailure(index: number) {
    return this.globalFineFaceExtension?.readBandRowFailure(index, true);
  }
  readGlobalFineBandFaceFailure(slot: number) {
    return this.globalFineFaceExtension?.readBandFaceFailure(slot);
  }
  readGlobalFineCandidateBandFaceFailure(slot: number) {
    return this.globalFineFaceExtension?.readBandFaceFailure(slot, true);
  }
  readGlobalFineCandidateBandIncidenceFailure(rowCount: number) {
    return this.globalFineFaceExtension?.readCandidateBandIncidenceFailure(rowCount);
  }
  readGlobalFineTransientPowerFaceFailure(slot: number) {
    return this.globalFineFaceExtension?.readTransientPowerFaceFailure(slot);
  }
  readGlobalFineBandAcuteTetraFailure(tagged: number) {
    return this.globalFineFaceExtension?.readBandAcuteTetraFailure(tagged);
  }
  readGlobalFineCandidateBandAcuteTetraFailure(tagged: number) {
    return this.globalFineFaceExtension?.readBandAcuteTetraFailure(tagged, true);
  }
  readGlobalFinePowerPublicationFailure(index: number) {
    const extension = this.globalFineFaceExtension, faces = this.powerFaces?.source;
    return extension && faces ? extension.readPowerPublicationFailure(index, faces) : undefined;
  }
  /** Diagnostic-only raw sparse summary header; topology consumes this GPU-side. */
  get globalFineSummaryDirectory(): GPUBuffer | undefined { return this.globalFineSummaries?.directory; }
  /** Rejection-only producer state for an unpublished fine summary. */
  get globalFineSummaryDebug() {
    const summaries = this.globalFineSummaries;
    const coarse = this.powerCoarseLevelSetSchedule;
    if (!summaries || !coarse) return undefined;
    return {
      ...summaries.diagnosticBuffers,
      coarseControl: coarse.control,
      coarseDelta: coarse.sampleSource.delta,
    };
  }
  get fineSeedAuthority() { return Boolean(this.fineSeedAdapter && this.globalFineLevelSet); }
  /** QA-only readback of the actual adapter-to-global-fine publication chain.
   * These counters are observational and never participate in simulation
   * scheduling or authority selection. */
  async readGlobalFineLevelSetDiagnostics() {
    const fine = this.globalFinePublishedIsA ? this.globalFineSourceA : this.globalFineSourceB;
    const topology = this.globalFinePublishedIsA ? this.globalFineTopologyBA : this.globalFineTopologyAB;
    const redistance = this.globalFinePublishedIsA ? this.globalFineRedistanceA : this.globalFineRedistanceB;
    if (!fine || !topology || !this.globalFineSeeds) return undefined;
    // Exact packed layout (bytes): existing chain [0, 560), point-field
    // control [560, 592), transient physical graph control [592, 656),
    // transition first-owner-mismatch payload [656, 720), the complete
    // complete JFA-CPT redistance control [720, 760) in its ABI-stable slot,
    // then closest-point completion diagnostics [768, 800), the power projection producer
    // control [800, 864), then the appended transport detail suffix
    // [864, 896), then the topology's pre-dilation Section 5 prefix count at
    // [896, 900), then the four mutually exclusive CPT interpolation-failure
    // reasons from face-control words 24..27 at [900, 916), then the complete
    // power-face publication control at [916, 980), candidate/row-delta
    // evidence through [1108), then the four direct CPT-failure face slots
    // at [1108, 1124), followed by the Power descriptor and topology
    // candidate/retained controls at [1124, 1252), followed by the complete
    // rejected Section 5 candidate controls at [1252, 1540). Existing
    // prefixes remain ABI-stable.
    const readback = this.device.createBuffer({ label: "Global fine QA diagnostics", size: 1540,
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
    if (this.globalFineFaceExtension) {
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.control, 0, readback, 304, 64);
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.control, 64, readback, 768, 32);
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.control, 96, readback, 900, 16);
    }
    if (this.powerVelocity) encoder.copyBufferToBuffer(this.powerVelocity.control, 0, readback, 368, 32);
    if (this.globalFineVelocityPrepass) {
      encoder.copyBufferToBuffer(this.globalFineVelocityPrepass.source.control, 0, readback, 400, 32);
    }
    if (this.globalFineFaceExtension) {
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.powerPublicationControl, 0, readback, 432, 64);
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.transitionControl, 0, readback, 496, 64);
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.pointFieldControl, 0, readback, 560, 32);
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.transientPowerControl, 0, readback, 592, 64);
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.transitionControl, 64, readback, 656, 64);
    }
    if (redistance) encoder.copyBufferToBuffer(redistance.control, 0, readback, 720,
      FINE_LEVELSET_REDISTANCE_CONTROL_BYTES);
    if (this.powerOperator) encoder.copyBufferToBuffer(this.powerOperator.control, 0, readback, 800, 64);
    if (this.lastGlobalFineTransport) {
      encoder.copyBufferToBuffer(this.lastGlobalFineTransport.control, 32, readback, 864, 32);
    }
    encoder.copyBufferToBuffer(topology.control, 32, readback, 896, 4);
    if (this.powerFaces) encoder.copyBufferToBuffer(this.powerFaces.control, 0, readback, 916, 64);
    if (this.powerFaces) {
      encoder.copyBufferToBuffer(this.powerFaces.candidateControlForDiagnostics, 0, readback, 980, 64);
    }
    encoder.copyBufferToBuffer(this.leafFrontier,
      this.frontierAllocation.rowDeltaControlOffsetWords * 4, readback, 1044, 64);
    if (this.globalFineFaceExtension) {
      encoder.copyBufferToBuffer(this.globalFineFaceExtension.control, 112, readback, 1108, 16);
    }
    if (this.powerDescriptor) {
      encoder.copyBufferToBuffer(this.powerDescriptor.control, 0, readback, 1124, 64);
    }
    if (this.powerTopology) {
      encoder.copyBufferToBuffer(this.powerTopology.control, 0, readback, 1188, 64);
    }
    if (this.globalFineFaceExtension) {
      encoder.copyBufferToBuffer(
        this.globalFineFaceExtension.candidateControlForDiagnostics, 0, readback, 1252, 128);
      encoder.copyBufferToBuffer(
        this.globalFineFaceExtension.candidateTransitionControlForDiagnostics, 0, readback, 1380, 160);
    }
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return {
        seedControl: Array.from(words.slice(0, 2)),
        topologyControl: [...words.slice(2, 10), words[224]],
        worklistHeader: Array.from(words.slice(10, 15)),
        seedCount: words[0], seedError: words[1], topologyFlags: words[2],
        interfaceBricks: words[3], desiredBricks: words[4], activatedBricks: words[5],
        interfaceSeedBricks: words[224],
        published: words[6] !== 0, rolledBack: words[7] !== 0,
        downstreamFinalizeReason: words[9], activeBricks: words[10], generation: words[11],
        configuredFineGeneration: fine.generation, fineGenerationSlot: fine.generationSlot,
        scheduledFineGeneration: this.globalFineGeneration, currentFineIsA: this.globalFinePublishedIsA,
        coarseDirectoryHeader: Array.from(words.slice(16, 24)),
        coarseControl: Array.from(words.slice(24, 40)),
        fineRestrictionControl: Array.from(words.slice(40, 48)),
        coarseDirectoryState: words[16], coarseDirectoryGeneration: words[17],
        coarseControlFlags: words[24], coarseControlGeneration: words[34], coarseControlValid: words[35],
        fineRestrictionCount: words[40], fineRestrictionMaximumPerRow: words[41],
        fineRestrictionFlags: words[42], fineRestrictionUnowned: words[43],
        fineRestrictionRows: words[44], fineRestrictionValid: words[45],
        transportControl: this.lastGlobalFineTransport
          ? [...words.slice(48, 56), ...words.slice(216, 224)]
          : Array.from(words.slice(48, 56)),
        transportDepartureOutsideBand: words[48], transportNonfiniteVelocity: words[49],
        transportProcessed: words[50], transportCommitted: words[51] !== 0,
        transportExtrapolatedVelocity: words[52], transportMaximumDisplacementFineCells: words[53],
        transportFaceBandUnavailable: words[54], transportVelocityUnavailable: words[55],
        redistanceControl: Array.from(words.slice(56, 60)),
        redistanceControlDetailed: Array.from(words.slice(180, 190)),
        volumeControl: Array.from(words.slice(60, 76)),
        faceBandControl: [...words.slice(76, 92), ...words.slice(192, 200),
          ...words.slice(225, 229), ...words.slice(277, 281)],
        powerFaceControl: Array.from(words.slice(229, 245)),
        powerFaceCandidateControl: Array.from(words.slice(245, 261)),
        powerRowDeltaControl: Array.from(words.slice(261, 277)),
        powerDescriptorControl: Array.from(words.slice(281, 297)),
        powerTopologyControl: Array.from(words.slice(297, 313)),
        faceBandCandidateControl: Array.from(words.slice(313, 345)),
        faceBandCandidateTransitionControl: Array.from(words.slice(345, 385)),
        powerVelocityControl: Array.from(words.slice(92, 100)),
        powerProjectionControl: Array.from(words.slice(200, 216)),
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
  get fluidBrickCapacity() { return this.topologyResidency.capacity; }
  readFluidBrickResidencyStats() { return this.topologyResidency.readStats(); }
  readFluidBulkBrickResidencyStats() { return this.sparseBrickWorld?.readBulkResidencyStats(); }
  encodeSparseBrickWorld(encoder: GPUCommandEncoder, _dt_s = 0) {
    if (!this.globalFineBootstrapped || !this.fineSeedAdapter) {
      throw new Error("Sparse render publication requires the compact fine-seed authority");
    }
    const source=this.fineSeedAdapter.source;
    this.topologyResidency.encodeFineSeedCandidates(
      encoder, source.leaves, source.candidates.candidates, source.candidates.countAndDispatch,
    );
    this.sparseBrickWorld?.bulkResidency?.encodeFineSeedCandidates(
      encoder, source.leaves, source.candidates.candidates, source.candidates.countAndDispatch,
    );
    // Publication is GPU-transactional. Failed, stale, and overflowing
    // generations retain the last good (including analytic t=0) tile stream;
    // a published zero-count generation is the distinct valid-empty case.
    this.topologyWorklistReady = true;
    this.compactVoxelInspection?.encode(encoder);
  }

  destroy() {
    this.powerLifecycleDisposed = true;
    this.mgpcg?.destroy();
    this.galerkin?.destroy();
    this.firstOrderVCycle?.destroy();
    this.ownerPages.destroy();
    this.pressureA.destroy(); this.pressureB.destroy(); this.params.destroy();
    this.topologyCandidateDispatch.destroy();
    this.compaction.destroy(); this.leafHeaders.destroy(); this.leafEntries.destroy(); this.leafFrontier.destroy(); this.solveDispatch.destroy(); this.solidCells.destroy(); this.solveStats.destroy(); this.unpublishedFineSummaryDirectory.destroy();
    this.compactVoxelInspection?.destroy();
    this.globalFineRedistanceA?.destroy(); this.globalFineRedistanceB?.destroy();
    this.analyticBootstrapWorklist?.destroy();
    this.globalFineVolumeA?.destroy(); this.globalFineVolumeB?.destroy();
    this.globalFineTransportA?.destroy(); this.globalFineTransportB?.destroy(); this.globalFineVelocityPrepass?.destroy();
    this.globalFineFaceExtension?.destroy();
    this.globalFineTopologyAB?.destroy(); this.globalFineTopologyBA?.destroy();
    this.globalFineSeeds?.destroy(); this.globalFineLevelSet?.destroy();
    this.globalFineSummaries?.destroy();
    this.fineSeedAdapter?.destroy();
    this.energyLedger?.destroy(); this.fineToPowerCoarseLevelSet?.destroy(); this.powerCoarseLevelSetSchedule?.destroy(); this.powerCoarseLevelSet?.destroy(); this.powerVelocity?.destroy(); this.powerSolidFaces?.destroy(); this.powerSolidVertices?.destroy(); this.powerFaceAdvection?.destroy(); this.powerFaceSeed?.destroy(); this.powerDescriptor?.destroy(); this.powerTopology?.destroy(); this.powerFaces?.destroy(); this.powerOperator?.destroy();
    this.powerVolumes?.destroy(); this.powerVolumeParams?.destroy();
    this.levelSetFallbackTexture?.destroy();
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
override sparseTopologyTileStates: u32 = 0u;
override denseSolidField: bool = true;
override frontierSortStage: u32 = 0u;
struct Owner { packedOrigin: u32, size: u32 }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowDirectionLength: vec4f, physical: vec4f, pressureCapacity: vec4u, hydrostatic: vec4f }
struct LeafHeader { cell: u32, entryStart: u32, entryCount: u32, size: u32, diagonal: f32, rhs: f32, pad0: u32, pad1: u32, gradient: vec4f }
struct LeafEntry { row: u32, coefficient: f32 }
struct RigidBody { positionShape: vec4f, dimensions: vec4f, orientation: vec4f, linearVelocity: vec4f, angularVelocity: vec4f, inverseMassInertia: vec4f, angularMomentumRestitution: vec4f, material: vec4f }
struct SolidCell { fraction: f32, owner: i32 }
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
@group(0) @binding(11) var<storage, read_write> solidCells: array<u32>;
@group(0) @binding(12) var terrainIn: texture_2d<f32>;
// [0..1] immutable A/B counts, [2] current selector, [3] generation,
// [4..5] candidate transaction state, followed by sorted A/B publications,
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
fn coarseMortonPart(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn coarseMorton(cell:u32)->u32{let d=dims();let q=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));return coarseMortonPart(q.x)|(coarseMortonPart(q.y)<<1u)|(coarseMortonPart(q.z)<<2u);}
fn coarseLookup(cell:u32,size:u32)->u32{let count=min(coarseWord(2u),(arrayLength(&bulkWorklist)-8u)/8u);let wantedLevel=31u-countLeadingZeros(size);let wantedMorton=coarseMorton(cell);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let base=8u+middle*8u;let entryLevel=31u-countLeadingZeros(coarseWord(base+1u));let entryMorton=coarseMorton(coarseWord(base)-1u);if(entryLevel<wantedLevel||(entryLevel==wantedLevel&&entryMorton<wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let base=8u+low*8u;if(coarseWord(base)==cell+1u&&coarseWord(base+1u)==size){return base;}}return 0xffffffffu;}
fn correctedCoarsePhi(point:vec3f)->CorrectedCoarsePhi{
  if(!coarseDirectoryAuthority()||any(point<vec3f(0.0))||any(point>=vec3f(dims()))){return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u);}
  let q=vec3u(floor(point));var size=1u;let maximumLeaf=coarseWord(3u);
  loop{let origin=(q/vec3u(size))*vec3u(size);let cell=origin.x+dims().x*(origin.y+dims().y*origin.z);let base=coarseLookup(cell,size);
    if(base!=0xffffffffu){let value=bitcast<f32>(coarseWord(base+2u));let minimum=bitcast<f32>(coarseWord(base+3u));let maximum=bitcast<f32>(coarseWord(base+4u));let flags=coarseWord(base+5u);
      if((flags&9u)!=9u||!coarseFinite(value)||!coarseFinite(minimum)||!coarseFinite(maximum)||minimum>maximum||value<minimum||value>maximum){return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u);}
      return CorrectedCoarsePhi(true,value,minimum,maximum,size);}
    if(size>=maximumLeaf){break;}size*=2u;
  }
  // Publication is all-or-nothing: every live liquid/interface row writes its
  // sorted slot before state becomes PUBLISHED. A miss in a valid directory is the
  // explicit positive-air complement, never an unknown sparse hole.
  let air=bitcast<f32>(coarseWord(7u))*f32(maximumLeaf);
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
fn encodePagedOwner(cell: vec3u, origin: vec3u, size: u32) -> u32 {
  let brickOrigin = (cell / vec3u(8u)) * vec3u(8u);
  let delta = vec3i(origin) - vec3i(brickOrigin);
  return OWNER_WORD_VALID
    | (u32(delta.x + 32) & 63u)
    | ((u32(delta.y + 32) & 63u) << 6u)
    | ((u32(delta.z + 32) & 63u) << 12u)
    | ((u32(firstTrailingBit(size)) & 7u) << 18u);
}
fn decodePagedOwner(word: u32, cell: vec3u) -> Owner {
  if ((word & OWNER_WORD_VALID) == 0u) { return canonicalOwner(cell); }
  let exponent = (word >> 18u) & 7u;
  if (exponent > 5u) { return canonicalOwner(cell); }
  let brickOrigin = vec3i((cell / vec3u(8u)) * vec3u(8u));
  let delta = vec3i(i32(word & 63u) - 32, i32((word >> 6u) & 63u) - 32,
    i32((word >> 12u) & 63u) - 32);
  let signedOrigin = brickOrigin + delta;
  if (any(signedOrigin < vec3i(0))) { return canonicalOwner(cell); }
  let origin = vec3u(signedOrigin); let size = 1u << exponent;
  if (any(cell < origin) || any(cell >= origin + vec3u(size))
      || any(origin + vec3u(size) > dims())) { return canonicalOwner(cell); }
  return Owner(packOrigin(origin), size);
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
fn ownerPageEncoded(logical: u32) -> u32 {
  let pageIndexOffset = atomicLoad(&owners[5]);
  if (pageIndexOffset <= 16u) { return 0u; }
  let recordCapacity = pageIndexOffset - 16u;
  let resident = min(atomicLoad(&owners[1]), recordCapacity);
  let key = logical + 1u;
  var low = 0u;
  var high = resident;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (atomicLoad(&owners[16u + middle]) < key) { low = middle + 1u; }
    else { high = middle; }
  }
  if (low >= resident || atomicLoad(&owners[16u + low]) != key) { return 0u; }
  return atomicLoad(&owners[pageIndexOffset + low]);
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
  let capacity = atomicLoad(&owners[3]);
  if (encoded == 0u || encoded == 0xffffffffu || encoded > capacity) { return 0xffffffffu; }
  let local = cell % vec3u(8u);
  return atomicLoad(&owners[atomicLoad(&owners[6]) + (encoded - 1u) * 512u + local.x + local.y * 8u + local.z * 64u]);
}
fn ownerAt(p: vec3i) -> Owner {
  let cell = vec3u(p);
  let word = ownerPageWord(cell);
  if (word == 0xffffffffu || word == 0u) { return canonicalOwner(cell); }
  return decodePagedOwner(word, cell);
}
fn ownerAtIndex(cell: u32) -> Owner { return ownerAt(vec3i(cellCoord(cell))); }
fn storeOwner(cell: vec3u, origin: vec3u, size: u32) {
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = ownerPageEncoded(logical); let capacity = atomicLoad(&owners[3]);
  if (encoded == 0u || encoded == 0xffffffffu || encoded > capacity) { return; }
  let local = cell % vec3u(8u);
  atomicStore(&owners[atomicLoad(&owners[6]) + (encoded - 1u) * 512u + local.x + local.y * 8u + local.z * 64u], encodePagedOwner(cell, origin, size));
}
fn storeOwnerRequired(cell: vec3u, origin: vec3u, size: u32) {
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = requireOwnerPageEncoded(logical); if (encoded == 0u) { return; }
  let local = cell % vec3u(8u);
  atomicStore(&owners[atomicLoad(&owners[6]) + (encoded - 1u) * 512u + local.x + local.y * 8u + local.z * 64u], encodePagedOwner(cell, origin, size));
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
// Negative sentinels encode the bootstrap-only analytic initial condition:
// tank = -10, dam-break = -20.
fn analyticInitialPhiEnabled() -> bool { return params.physical.w < 0.0; }
fn analyticInitialDamBreak() -> bool { return params.physical.w < -15.0; }
fn analyticInitialPhi(point: vec3f) -> f32 {
  let fill = clamp(params.hydrostatic.w / f32(max(1u, dims().y)), 0.0, 1.0);
  let world = vec3f(-0.5 * params.container.x + point.x * params.cellRelax.x,
    point.y * params.cellRelax.y,
    -0.5 * params.container.z + point.z * params.cellRelax.z);
  if (!analyticInitialDamBreak()) { return world.y - fill * params.container.y; }
  let heightFraction = max(0.92, fill);
  let footprintFraction = sqrt(fill / max(heightFraction, 1e-9));
  let exposedMaximum = vec3f(-0.5 * params.container.x + footprintFraction * params.container.x,
    heightFraction * params.container.y,
    -0.5 * params.container.z + footprintFraction * params.container.z);
  let q = world - exposedMaximum;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn phi(p: vec3i) -> f32 {
  if (!valid(p)) { return 3.402823e38; }
  if(analyticInitialPhiEnabled()){
    return analyticInitialPhi(vec3f(p)+vec3f(0.5));
  }
  let coarse=correctedCoarsePhi(vec3f(p)+vec3f(0.5));
  if(coarse.authority){return coarseClassificationPhi(coarse);}
  return 3.402823e38;
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
  if(analyticInitialPhiEnabled()){return analyticInitialPhi(centre+vec3f(0.5));}
  let coarse=correctedCoarsePhi(centre+vec3f(0.5));
  if(coarse.authority){return coarseClassificationPhi(coarse);}
  return samplePhiPoint(centre);
}
fn ownerPhiGradient(owner:Owner)->vec3f{let c=vec3f(unpackOrigin(owner.packedOrigin))+vec3f(0.5*f32(owner.size-1u));let step=max(0.5,0.5*f32(owner.size));
  return vec3f(samplePhiPoint(c+vec3f(step,0,0))-samplePhiPoint(c-vec3f(step,0,0)),samplePhiPoint(c+vec3f(0,step,0))-samplePhiPoint(c-vec3f(0,step,0)),samplePhiPoint(c+vec3f(0,0,step))-samplePhiPoint(c-vec3f(0,0,step)))/(2.0*step);}
fn liquidOwner(owner: Owner) -> bool {
  let centre=vec3f(unpackOrigin(owner.packedOrigin))+vec3f(0.5*f32(owner.size));
  if(analyticInitialPhiEnabled()){return analyticInitialPhi(centre)<0.0;}
  let coarse=correctedCoarsePhi(centre);
  if(coarse.authority&&coarse.leafSize==owner.size){return coarse.phi<0.0;}
  if(coarse.authority&&coarse.leafSize==0u){return false;}
  if(coarse.authority&&coarse.maximumPhi<0.0){return true;}
  if(coarse.authority&&coarse.minimumPhi>=0.0){return false;}
  return false;
}
fn isOrigin(id: vec3u, owner: Owner) -> bool { return all(id == unpackOrigin(owner.packedOrigin)); }
fn cellCount() -> u32 { return params.dimsMax.x * params.dimsMax.y * params.dimsMax.z; }
fn frontierListCapacity() -> u32 { return params.pressureCapacity.x; }
fn frontierBase(which: u32) -> u32 { return 6u + which * frontierListCapacity(); }
fn frontierCandidateBase() -> u32 { return 6u + 2u * frontierListCapacity(); }
fn rowDeltaControlBase()->u32{return frontierCandidateBase()+frontierListCapacity();}
fn rowDeltaNewToOldBase()->u32{return rowDeltaControlBase()+16u;}
fn rowDeltaOldToNewBase()->u32{return rowDeltaNewToOldBase()+frontierListCapacity();}
fn rowDeltaDirtyRowsBase()->u32{return rowDeltaOldToNewBase()+frontierListCapacity();}
fn rowDeltaAffectedRowsBase()->u32{return rowDeltaDirtyRowsBase()+frontierListCapacity();}
fn frontierCurrent() -> u32 { return frontier[2]; }
fn frontierGeneration() -> u32 { return frontier[3]; }
fn frontierCount(which: u32) -> u32 { return min(frontier[which], frontierListCapacity()); }
fn frontierCell(which: u32, slot: u32) -> u32 { return frontier[frontierBase(which) + slot]; }
fn frontierRowIdentity(cell:u32,size:u32)->u32{
  let current=frontierCurrent();let count=frontierCount(current);
  let level=u32(firstTrailingBit(size));let morton=rowMorton(cell);
  var lo=0u;var hi=count;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(current,mid);
    let otherSize=ownerAtIndex(other).size;let otherLevel=u32(firstTrailingBit(otherSize));
    let otherMorton=rowMorton(other);
    if(otherLevel<level||(otherLevel==level&&(otherMorton<morton
      ||(otherMorton==morton&&other<cell)))){lo=mid+1u;}else{hi=mid;}}
  if(lo<count&&frontierCell(current,lo)==cell&&ownerAtIndex(cell).size==size){return lo;}
  return 0xffffffffu;
}
fn frontierRow(cell:u32)->u32{return frontierRowIdentity(cell,ownerAtIndex(cell).size);}
fn frontierAlive(cell:u32)->bool{return frontierRow(cell)!=0xffffffffu;}
fn pressureIndex(owner: Owner) -> u32 {
  return frontierRowIdentity(index(unpackOrigin(owner.packedOrigin)),owner.size);
}
// Section 4.3 requires the L1 and L2 operators to use exactly the same set of
// pressure variables. Recurring frontier publication may classify a leaf from
// the newer complete fine-summary interval, while liquidOwner() deliberately
// retains the older coarse/page fallbacks for topology construction. Once the
// compact frontier is published, membership in that frontier is therefore the
// pressure-variable authority; reclassifying an incident leaf here can create
// an L1 entry for a row that does not exist in L2.
fn pressureVariableExists(owner: Owner) -> bool {
  if (!rowIndexedPressure) { return liquidOwner(owner); }
  return pressureIndex(owner) < compaction[0];
}
// The trailing eight words are isolated from topology-change state and scan
// partials: overflow, required rows, required entries, fallback dispatch xyz,
// then residual sums rr/bb.
fn pressureControlBase() -> u32 { return arrayLength(&compaction) - 8u; }
fn pressureOverflowed() -> bool {
  return compaction[pressureControlBase()] != 0u || atomicLoad(&owners[2]) != 0u;
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
  // First-order ghost-fluid distance used by the L1 preconditioner: p=0 lies
  // at the resident level-set crossing, not at the neighbouring air centre.
  // The lower bound is an implementation degeneracy guard, not an equation
  // attributed to Aanjaneya et al. 2017.
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
fn faceSolid(a: vec3i, b: vec3i) -> SolidCell { let sa = solidAt(a); let sb = solidAt(b); if (sa.fraction >= sb.fraction) { return sa; } return sb; }
fn coarseTaskListBase() -> u32 { return 15u + 3u * params.control.z; }
fn coarseTaskCapacity() -> u32 {
  return params.pressureCapacity.z;
}
fn coarseTaskRow(task: u32) -> u32 { return compaction[coarseTaskListBase() + 2u * task]; }
fn coarseTaskTile(task: u32) -> u32 { return compaction[coarseTaskListBase() + 2u * task + 1u]; }
fn coarseTaskIndex(workgroup: vec3u) -> u32 { return workgroup.x + workgroup.y * compaction[5]; }

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
const TILE_SIGNATURE_VALID_MAGIC: u32 = 0x5453474eu;
const TILE_SIGNATURE_FAILED: u32 = 0xffffffffu;
fn changeStateWords() -> u32 {
  return 14u * topologyTileCapacity() + 1u + RIGID_SNAPSHOT_WORDS + 14u;
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
const FRONTIER_REUSE_MAGIC: u32 = 0x46525553u;
const FRONTIER_FAILED_MAGIC: u32 = 0x4641494cu;
fn frontierGenerationReused() -> bool {
  return compaction[11] == FRONTIER_REUSE_MAGIC
    || compaction[frontierTopologyReuseBase()] != 0u;
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
    compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
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
  if (validTile) {
    for (var flat = lid; flat < cellCount; flat += 256u) {
      let local = vec3u(flat % tileSize, (flat / tileSize) % tileSize,
        flat / (tileSize * tileSize));
      let q = origin + local;
      if (any(q >= dims())) { continue; }
      let cell = index(q);
      let owner = ownerAtIndex(cell);
      if (!isOrigin(q, owner)) { continue; }
      let wet = currentPressureOwnerWet(owner);
      // The coarse sizing decision is structural: owner identity/scale,
      // residency, solids, and rigid state. Fine phi may change wet row
      // membership, but it must not force reset/refine/balance.
      let structuralDecision = cell ^ (owner.size * 0x9e3779b9u);
      let frontierDecision = structuralDecision ^ select(0u, 0x85ebca6bu, wet);
      signature.x ^= topologyDecisionHash(structuralDecision);
      signature.y += topologyDecisionHash(structuralDecision ^ 0xc2b2ae35u);
      signature.z += 1u;
      signature.w += owner.size;
      frontierSignature.x ^= topologyDecisionHash(frontierDecision);
      frontierSignature.y += topologyDecisionHash(frontierDecision ^ 0xc2b2ae35u);
      frontierSignature.z += 1u;
      frontierSignature.w += select(0u, 1u, wet);
    }
  }
  tileSignatureReduction[lid] = signature;
  tileFrontierSignatureReduction[lid] = frontierSignature;
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
  let next = tileSignatureReduction[0];
  let valid = compaction[base + 4u] == TILE_SIGNATURE_VALID_MAGIC;
  let structuralUnchanged = valid && all(vec4u(compaction[base], compaction[base + 1u],
    compaction[base + 2u], compaction[base + 3u]) == next);
  compaction[base] = next.x; compaction[base + 1u] = next.y;
  compaction[base + 2u] = next.z; compaction[base + 3u] = next.w;
  compaction[base + 4u] = TILE_SIGNATURE_VALID_MAGIC;
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
  let generation = max(1u, frontier[3] + 1u);
  var dirtyCount = 0u;
  compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
  let capacity = topologyTileCapacity();
  let activeCount = compaction[0];
  if (activeCount > capacity || compaction[4] > capacity) {
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
      compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
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
      compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
      break;
    }
    let signature = tileSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
    compaction[signature + 4u] = 0u;
    let frontierSignature = tileFrontierSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
    compaction[frontierSignature + 4u] = 0u;
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
}

@compute @workgroup_size(1)
fn buildDirtyFrontierDelta() {
  let generation = max(1u, frontier[3] + 1u);
  let capacity = topologyTileCapacity();
  let activeCount = compaction[0];
  var dirtyCount = 0u;
  compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
  if (activeCount > capacity || compaction[4] > capacity) {
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
      compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
      break;
    }
    let structural = compaction[tileChangeFlagsBase() + tileIndex] == generation;
    let wet = (changed & TILE_SIGNATURE_FRONTIER_CHANGED) != 0u;
    if (structural || wet) {
      if (dirtyCount >= capacity) {
        compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
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
  var result = FineLeafSummary(false, false, false, false, 0.0,
    3.402823e38, -3.402823e38, 3.402823e38);
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
  let count = fineSummaryWord(2u); let capacity = fineSummaryWord(3u);
  if (count > capacity || 16u + count * 8u > fineSummaryLength()) { return result; }
  var low = 0u; var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    let observed = fineSummaryWord(16u + middle * 8u);
    if (observed < key) { low = middle + 1u; } else { high = middle; }
  }
  if (low < count) {
    let base = 16u + low * 8u;
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
    // set. The summary publication evaluates the eight fine samples around
    // this dyadic node's own geometric centre, so the evidence applies to
    // every pressure-leaf size and remains independent of interval coverage.
    result.centerValid = (entryFlags & 0x3fc00000u) == 0x3fc00000u
      && fineSummaryFinite(result.centerPhi);
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
  let adaptivity = f32(params.control.x) / 1000.0;
  if (adaptivity <= 0.0) { return true; }
  // Empty/open mini-dam scenes bind only a format-valid solid sentinel. This
  // immutable override removes the otherwise dominant size^3 storage walk.
  if (!denseSolidField) { return false; }
  var minimumSolid = 1.0; var maximumSolid = 0.0;
  // Section 4 power cells may cross the free surface. Section 5 owns its
  // separate factor-m narrow band, so pressure sizing has no surface-driven
  // compatibility branch or dense-phi fallback.
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q = origin + vec3u(x,y,z); let solid = solidAt(vec3i(q)).fraction;
    minimumSolid = min(minimumSolid, solid); maximumSolid = max(maximumSolid, solid);
  } } }
  // Solid boundaries split; liquid interface resolution belongs exclusively
  // to the global-fine Section 5 publication.
  let crossesSolidBoundary = maximumSolid - minimumSolid > 1e-5 || (maximumSolid > 1e-5 && maximumSolid < 1.0 - 1e-5);
  if (crossesSolidBoundary) { return true; }
  if (minimumSolid >= 1.0 - 1e-5) { return false; }
  return false;
}

fn splitLeaf(origin: vec3u, size: u32) {
  let child = size / 2u;
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let local = vec3u(x,y,z); let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerRequired(origin + local, childOrigin, child);
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
var<workgroup> refineSolidRange: array<vec2f, 128>;

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
    let eligible = inBounds && owner.size == targetRefinementSize && isOrigin(origin, owner);
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
  var solidRange = vec2f(1.0, 0.0);
  if (denseSolidField) {
    let solidCellsInLeaf = size * size * size;
    for (var flat = lid; flat < solidCellsInLeaf; flat += 128u) {
      let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
      let solid = solidAt(vec3i(origin + local)).fraction;
      solidRange = vec2f(min(solidRange.x, solid), max(solidRange.y, solid));
    }
  }
  refineSolidRange[lid] = solidRange;
  for (var stride = 64u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) {
      let right = refineSolidRange[lid + stride];
      refineSolidRange[lid] = vec2f(
        min(refineSolidRange[lid].x, right.x),
        max(refineSolidRange[lid].y, right.y),
      );
    }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let range = refineSolidRange[0];
    let adaptivity = f32(params.control.x) / 1000.0;
    let crossesSolid = range.y - range.x > 1e-5
      || (range.y > 1e-5 && range.y < 1.0 - 1e-5);
    let decision = powerClosedWallStripIntersects(origin, size)
      || adaptivity <= 0.0 || (denseSolidField && crossesSolid);
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
fn repairPaperMixedNeighbors(origin: vec3u, size: u32) {
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
  let owner = ownerAt(vec3i(gid));
  if (owner.size >= 2u && owner.size <= 16u && isOrigin(gid, owner)) { repairPaperMixedNeighbors(gid, owner.size); }
  // Size-16+ leaves use the cooperative entry point below.
  if (owner.size > 2u && owner.size < 16u && isOrigin(gid, owner) && neighborTooFine(gid, owner.size)) { splitLeaf(gid, owner.size); }
}

@compute @workgroup_size(4,4,4)
fn balanceTopology(@builtin(global_invocation_id) gid: vec3u) { balanceTopologyAt(gid * 2u); }

@compute @workgroup_size(4,4,4)
fn balanceTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  balanceTopologyAt(deltaTopologyCandidate(wid, lid));
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
  requireLeafOwnerPages(origin, size, lid, 256u);
  workgroupBarrier();
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerRequired(origin + local, childOrigin, child);
  }
}

fn pressureOf(owner: Owner) -> f32 {
  let slot = pressureIndex(owner);
  if (slot >= arrayLength(&pressureIn)) { return 0.0; }
  return pressureIn[slot];
}

// --- Compacted leaf solve -------------------------------------------------
// The production path compacts liquid leaf origins, assembles each row once,
// then hands the compact operator to the mandatory power MGPCG solver. 2:1
// balance bounds a leaf's distinct neighbors at 4 per face (24 total; 6 for
// finest leaves), which also bounds the entry pool at 6 entries per cell.

// The leaf frontier is an immutable sorted A/B publication. Recurring
// generations emit fixed dirty-tile candidate records, sort that bounded
// stream, and merge it with clean rows from the previous publication. No
// claim table, tombstone, append counter, or whole-active-list fallback exists.
@compute @workgroup_size(1)
fn beginFrontier() {
  let current = frontierCurrent();
  frontier[4] = 0u;
  frontier[5] = 0u;
  let control = pressureControlBase();
  compaction[control] = 0u;
  compaction[control + 1u] = 0u;
  compaction[control + 2u] = 0u;
  let failed = compaction[dirtyAuthorityBase()] == FRONTIER_FAILED_MAGIC;
  let reuse = compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC
    && compaction[0] == 0u && compaction[4] == 0u;
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
    frontier[3] += 1u;
    let count = frontierCount(current);
    let base = rowDeltaControlBase();
    frontier[base] = count; frontier[base + 1u] = count;
    frontier[base + 2u] = count; frontier[base + 3u] = 0u;
    frontier[base + 4u] = 0u; frontier[base + 5u] = 0u;
    frontier[base + 6u] = 0u; frontier[base + 7u] = frontier[3];
    frontier[base + 8u] = 0x52444c54u;
    frontier[base + 9u] = 0u; frontier[base + 10u] = 1u;
    frontier[base + 11u] = 1u; frontier[base + 12u] = 0u;
    frontier[base + 13u] = 1u; frontier[base + 14u] = 1u;
    frontier[base + 15u] = 1u;
  }
}

fn currentPressureOwnerWet(owner: Owner) -> bool {
  let origin=unpackOrigin(owner.packedOrigin);let fine=fineLeafSummary(origin,owner.size);
  var wet=liquidOwner(owner);
  if(analyticInitialPhiEnabled()){return wet;}
  if(fine.found){
    if(fine.centerValid){wet=fine.centerPhi<0.0;}
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

fn frontierCandidateAt(gid:vec3u)->u32{
  if(any(gid>=dims())){return 0xffffffffu;}
  let cell = index(gid);
  let owner = ownerAtIndex(cell);
  if(!isOrigin(gid,owner)||!currentPressureOwnerWet(owner)){return 0xffffffffu;}
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
    if(all(base<dims())){cell=frontierCandidateAt(base+offset);}
    count+=select(0u,1u,cell!=0xffffffffu);
  }
  frontierCandidateScan[local]=count;
  for(var stride=32u;stride>0u;stride>>=1u){workgroupBarrier();
    if(local<stride){frontierCandidateScan[local]+=frontierCandidateScan[local+stride];}}
  workgroupBarrier();
  if(local==0u){let block=candidateBlockIndex(wid,deltaMode);
    compaction[coarseTaskListBase()+2u*block]=frontierCandidateScan[0];}
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
  for(var block=begin;block<end;block+=1u){subtotal+=compaction[coarseTaskListBase()+2u*block];}
  // Cooperative Hillis-Steele scan over the 256 lane subtotals.
  rowDeltaScan[lid]=subtotal;workgroupBarrier();
  for(var stride=1u;stride<256u;stride<<=1u){
    var add=0u;if(lid>=stride){add=rowDeltaScan[lid-stride];}
    workgroupBarrier();rowDeltaScan[lid]+=add;workgroupBarrier();
  }
  var cursor=rowDeltaScan[lid]-subtotal;
  for(var block=begin;block<end;block+=1u){let count=compaction[coarseTaskListBase()+2u*block];
    compaction[coarseTaskListBase()+2u*block+1u]=cursor;cursor+=count;}
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
    if(all(base<dims())){cell=frontierCandidateAt(base+offset);}
    laneCandidates[octant]=cell;
    laneCount+=select(0u,1u,cell!=0xffffffffu);
  }
  frontierCandidateScan[local]=laneCount;
  for(var stride=1u;stride<64u;stride<<=1u){workgroupBarrier();var add=0u;
    if(local>=stride){add=frontierCandidateScan[local-stride];}
    workgroupBarrier();frontierCandidateScan[local]+=add;}
  workgroupBarrier();
  let block=candidateBlockIndex(wid,deltaMode);
  let outputBase=compaction[coarseTaskListBase()+2u*block+1u]
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
  return select(leafEntries[index].row,frontier[frontierCandidateBase()+index],fromCandidate);
}
fn candidateSortStore(index:u32,value:u32,toCandidate:bool){
  if(toCandidate){frontier[frontierCandidateBase()+index]=value;}
  else{leafEntries[index].row=value;}
}
const ROW_DELTA_VALID:u32=0x52444c54u;
const ROW_DELTA_AFFECTED:u32=0x80000000u;
fn rowDeltaMapOld(encoded:u32)->u32{
  let value=encoded&0x7fffffffu;
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
  if(any(dims()>vec3u(1024u))||count>arrayLength(&leafEntries)){
    if(row==0u){compaction[pressureControlBase()]=4u;}return;
  }
  let stage=frontierSortStage;
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

fn rowAuthorityDirtyGeneration(cell:u32,generation:u32)->bool{
  let origin=cellCoord(cell);let tileSize=topologyTileSize();
  let td=(dims()+vec3u(tileSize-1u))/tileSize;let tile=vec3i(origin/tileSize);
  let ownIndex=u32(tile.x)+td.x*(u32(tile.y)+td.y*u32(tile.z));
  if(compaction[tileFrontierChangeFlagsBase()+ownIndex]==generation){return true;}
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
    let q=tile+vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=vec3i(td))){continue;}
    let index=u32(q.x)+td.x*(u32(q.y)+td.y*u32(q.z));
    if(compaction[tileChangeFlagsBase()+index]==generation){return true;}
  }}}
  return false;
}
fn rowAuthorityDirty(cell:u32)->bool{return rowAuthorityDirtyGeneration(cell,frontierGeneration());}
fn rowKeyLess(levelA:u32,mortonA:u32,levelB:u32,mortonB:u32)->bool{
  return levelA<levelB||(levelA==levelB&&mortonA<mortonB);
}
fn rowIdentityLess(cellA:u32,sizeA:u32,cellB:u32,sizeB:u32)->bool{
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
  let cell=frontierCell(previous,row);let old=leafHeaders[row];
  let dirty=rowAuthorityDirtyGeneration(cell,frontierGeneration()+1u);
  let owner=ownerAtIndex(cell);
  let cellMatches=old.cell==cell;
  let sizeMatches=old.size==owner.size;
  let originMatches=isOrigin(cellCoord(cell),owner);
  let exact=cellMatches&&sizeMatches&&originMatches;
  let wet=currentPressureOwnerWet(owner);
  let keep=!dirty&&exact&&wet;
  // A supposedly clean identity is never silently retired. That indicates
  // incomplete dirty evidence and rejects the candidate generation.
  let reason=select(0u,1u,!cellMatches)|select(0u,2u,!sizeMatches)
    |select(0u,4u,!originMatches)|select(0u,8u,exact&&!wet);
  compaction[rowDeltaFlagsBase()+row]=select(0u,1u,keep)
    |select(0u,reason<<1u,!dirty)|select(0u,32u,dirty);
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
    if(output<frontierListCapacity()){frontier[frontierBase(next)+output]=cell;}
  }
  if(slot<candidateCount){
    let cell=frontier[frontierCandidateBase()+slot];let size=ownerAtIndex(cell).size;
    let old=previousLowerBound(cell,size,previous,previousCount);
    // A candidate must replace a dirty old identity, never collide with a
    // clean carried row. Leave that destination to the carried writer; the
    // final validator rejects the overlap without a storage race.
    let cleanCollision=old<previousCount&&frontierCell(previous,old)==cell
      &&leafHeaders[old].size==size&&(compaction[rowDeltaFlagsBase()+old]&1u)!=0u;
    if(!cleanCollision){
      let output=slot+keptRowsBefore(old,previousCount);
      if(output<frontierListCapacity()){frontier[frontierBase(next)+output]=cell;}
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
  var matched=0u;var invalid=select(0u,1u,candidateCount>frontierListCapacity()
    ||required>frontierListCapacity());
  var firstFailure=0xffffffffu;var exactFailures=0u;
  if(!frontierRejected&&!frontierReused){
    for(var row=lid;row<previousCount;row+=256u){
      let flags=compaction[rowDeltaFlagsBase()+row];
      let reason=(flags>>1u)&15u;
      invalid|=select(0u,1u,reason!=0u);
      firstFailure=min(firstFailure,select(0xffffffffu,row*16u+reason,reason!=0u));
      exactFailures+=select(0u,1u,(reason&7u)!=0u);
    }
    for(var row=lid;row<boundedCandidates;row+=256u){
      let cell=frontier[frontierCandidateBase()+row];let size=ownerAtIndex(cell).size;
      if(row>0u){let prior=frontier[frontierCandidateBase()+row-1u];
        invalid|=select(0u,1u,!rowIdentityLess(prior,ownerAtIndex(prior).size,cell,size));}
      let old=previousLowerBound(cell,size,previous,previousCount);
      let exact=old<previousCount&&frontierCell(previous,old)==cell&&leafHeaders[old].size==size;
      matched+=select(0u,1u,exact);
      invalid|=select(0u,1u,exact&&(compaction[rowDeltaFlagsBase()+old]&1u)!=0u);
    }
    for(var row=lid;row<min(required,frontierListCapacity());row+=256u){
      let cell=frontier[frontierBase(next)+row];let size=ownerAtIndex(cell).size;
      invalid|=select(0u,1u,!isOrigin(cellCoord(cell),ownerAtIndex(cell))
        ||!currentPressureOwnerWet(ownerAtIndex(cell)));
      if(row>0u){let prior=frontier[frontierBase(next)+row-1u];
        invalid|=select(0u,1u,!rowIdentityLess(prior,ownerAtIndex(prior).size,cell,size));}
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
  if(frontierRejected){return;}
  if(frontierReused){
    // The immutable frontier selector and leaf payload remain valid, but the
    // identity maps still need one bounded dispatch so value-refresh
    // consumers can accept the new generation with zero dirty rows.
    let blocks=(previousCount+255u)/256u;
    compaction[1]=blocks;compaction[2]=1u;compaction[3]=1u;
    return;
  }
  let carried=frontier[5]+rowDeltaReduce[0].x;
  let added=select(candidateCount-rowDeltaReduce[0].x,0u,rowDeltaReduce[0].x>candidateCount);
  let retired=select(previousCount-carried,0u,carried>previousCount);
  let valid=rowDeltaReduce[0].y==0u&&carried<=previousCount
    &&required==carried+added&&required==previousCount+added-retired;
  if(!valid){
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
    compaction[11]=FRONTIER_FAILED_MAGIC;compaction[frontierTopologyReuseBase()]=0u;
    compaction[1]=0u;compaction[2]=1u;compaction[3]=1u;
    compaction[12]=0u;compaction[13]=1u;compaction[14]=1u;return;
  }
  frontier[next]=required;frontier[2]=next;frontier[3]+=1u;
  let blocks=(required+255u)/256u;compaction[8]=blocks;
  let x=min(blocks,65535u);var y=1u;if(x>0u){y=(blocks+x-1u)/x;}
  compaction[12]=x;compaction[13]=y;compaction[14]=1u;
  let rowBlocks=max(blocks,(previousCount+255u)/256u);
  compaction[1]=rowBlocks;compaction[2]=1u;compaction[3]=1u;
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
    let dirty=old==0xffffffffu||old!=row||rowAuthorityDirty(cell);
    frontier[rowDeltaNewToOldBase()+row]=
      select(old+1u,0u,old==0xffffffffu)|select(0u,ROW_DELTA_AFFECTED,dirty);
    compaction[rowDeltaFlagsBase()+row]=select(0u,1u,dirty);
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
  if(row<count){flag=select(compaction[rowDeltaFlagsBase()+row]&1u,
    select(0u,1u,(frontier[rowDeltaNewToOldBase()+row]&ROW_DELTA_AFFECTED)!=0u),affected);}
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
  if(row<count&&compaction[rowDeltaFlagsBase()+row]!=0u){
    let output=compaction[rowDeltaPrefixBase()+row]
      +compaction[rowDeltaBlockTotalsBase()+row/256u];
    frontier[rowDeltaDirtyRowsBase()+output]=row;
  }
}

@compute @workgroup_size(256)
fn markRowDeltaRing(@builtin(global_invocation_id)gid:vec3u){
  let base=rowDeltaControlBase();let count=frontier[base];
  let row=gid.x;if(row<count){
    let h=leafHeaders[row];
    // A liquid-row insertion or retirement changes the compact row numbering
    // and may alter which side owns a generalized face. Rebuild the bounded
    // current face graph in that uncommon case; positional carry remains
    // exact and cheap for the overwhelmingly common identity generation.
    let membershipChanged=frontier[base+3u]!=0u||frontier[base+4u]!=0u;
    var affected=membershipChanged
      ||(frontier[rowDeltaNewToOldBase()+row]&ROW_DELTA_AFFECTED)!=0u;
    let origin=cellCoord(h.cell);
    for(var z=-1;z<=1&&!affected;z++){for(var y=-1;y<=1&&!affected;y++){for(var x=-1;x<=1&&!affected;x++){
      let nonzero=select(0u,1u,x!=0)+select(0u,1u,y!=0)+select(0u,1u,z!=0);
      if(nonzero==0u||nonzero==3u){continue;}let d=vec3i(x,y,z);var probe=vec3i(0);
      for(var axis=0u;axis<3u;axis+=1u){probe[axis]=select(select(i32(origin[axis]+h.size/2u),i32(origin[axis]+h.size),d[axis]>0),i32(origin[axis])-1,d[axis]<0);}
      if(!valid(probe)){continue;}let owner=ownerAt(probe);let neighbor=frontierRow(index(unpackOrigin(owner.packedOrigin)));
      if(neighbor!=0xffffffffu&&compaction[rowDeltaFlagsBase()+neighbor]!=0u){affected=true;}
    }}}
    if(affected){frontier[rowDeltaNewToOldBase()+row]|=ROW_DELTA_AFFECTED;}
  }
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
  let row=gid.x;let count=frontierCount(frontierCurrent());
  if(row<count){
    frontier[rowDeltaNewToOldBase()+row]=row+1u;
    frontier[rowDeltaOldToNewBase()+row]=row+1u;
    compaction[rowDeltaFlagsBase()+row]=0u;
  }
  if(row==0u){
    let base=rowDeltaControlBase();
    frontier[base]=count;frontier[base+1u]=count;frontier[base+2u]=count;
    frontier[base+3u]=0u;frontier[base+4u]=0u;frontier[base+5u]=0u;
    frontier[base+6u]=0u;frontier[base+7u]=frontierGeneration();
    frontier[base+8u]=ROW_DELTA_VALID;
    frontier[base+9u]=0u;frontier[base+10u]=1u;frontier[base+11u]=1u;
    frontier[base+12u]=0u;frontier[base+13u]=1u;frontier[base+14u]=1u;
    frontier[base+15u]=1u;
  }
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
        compaction[pressureControlBase()] = 4u;
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
    let previousRow = rowDeltaMapOld(frontier[rowDeltaNewToOldBase()+row]);
    var warm = 0.0;
    if (rowIndexedPressure && previousRow < arrayLength(&pressureIn)) { warm = pressureIn[previousRow]; }
    if (rowIndexedPressure) {
      pressureOut[row] = select(0.0, warm, (params.pressureCapacity.w & 1u) != 0u);
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
      if (pressureVariableExists(neighbor)) {
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
      }
    } }
  }
  header.entryCount = neighborCount; header.diagonal = diagonal; header.rhs = 0.0;
  // The compact power-face operator consumes pressure directly. Retain inert
  // header padding without pulling projection work into assembly.
  header.gradient = vec4f(0.0);
  leafHeaders[row] = header;
  for (var j = 0u; j < neighborCount; j += 1u) { leafEntries[header.entryStart + j] = LeafEntry(neighborCells[j], neighborCoefficients[j]); }
}

// A balanced neighbor is no smaller than half this leaf, so each face quadrant
// touches at most one neighbor. Sixty-four lanes can therefore reduce all
// size^2 finest subfaces into four deterministic coefficients per face without
// atomics. Duplicate quadrant entries for a same-size/coarser neighbor are
// algebraically identical to the merged serial entry and retain a fixed layout.
var<workgroup> coarseDiagonalScratch: array<f32, 64>;
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

  for (var face = 0u; face < 6u; face += 1u) {
    let axis = face / 2u;
    let side = select(-1, 1, (face & 1u) == 1u);
    let e = axisVector(axis);
    let area = faceArea(axis);
    var laneDiagonal = 0.0;
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
      if (pressureVariableExists(neighbor)) {
        let quadrant = select(0u, 1u, a >= half) + select(0u, 2u, b >= half);
        laneCoefficients[quadrant] += coefficient;
      }
    }
    coarseDiagonalScratch[lid] = laneDiagonal;
    for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) {
      coarseCoefficientScratch[quadrant * 64u + lid] = laneCoefficients[quadrant];
    }
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) {
        coarseDiagonalScratch[lid] += coarseDiagonalScratch[lid + stride];
        for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) {
          let slot = quadrant * 64u + lid;
          coarseCoefficientScratch[slot] += coarseCoefficientScratch[slot + stride];
        }
      }
    }
    workgroupBarrier();
    if (lid == 0u) {
      diagonal += coarseDiagonalScratch[0];
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
          if (pressureVariableExists(neighbor)) { neighborCell = pressureIndex(neighbor); }
        }
        leafEntries[header.entryStart + face * 4u + quadrant] = LeafEntry(neighborCell, coefficient);
      }
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    header.entryCount = 24u;
    header.diagonal = diagonal;
    header.rhs = 0.0;
    // The compact power-face operator consumes pressure directly; gradient
    // padding remains inert for ABI stability.
    header.gradient = vec4f(0.0);
    leafHeaders[row] = header;
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
@group(0) @binding(3) var levelSetIn: texture_3d<f32>;
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
fn decodePagedOwner(word:u32,cell:vec3u)->Owner{
  if((word&0x80000000u)==0u){return canonicalOwner(cell);}let exponent=(word>>18u)&7u;
  if(exponent>5u){return canonicalOwner(cell);}let brickOrigin=vec3i((cell/vec3u(8u))*vec3u(8u));
  let delta=vec3i(i32(word&63u)-32,i32((word>>6u)&63u)-32,i32((word>>12u)&63u)-32);
  let signedOrigin=brickOrigin+delta;if(any(signedOrigin<vec3i(0))){return canonicalOwner(cell);}
  let origin=vec3u(signedOrigin);let size=1u<<exponent;
  if(any(cell<origin)||any(cell>=origin+vec3u(size))||any(origin+vec3u(size)>dims())){return canonicalOwner(cell);}
  return Owner(packOrigin(origin),size);
}
fn canonicalOwner(cell: vec3u) -> Owner { var size=min(params.dimsMax.w,8u);var origin=(cell/vec3u(size))*vec3u(size);loop{if(all(origin+vec3u(size)<=dims())||size==1u){break;}size>>=1u;origin=(cell/vec3u(size))*vec3u(size);}return Owner(packOrigin(origin),size); }
fn ownerPageEncoded(logical:u32)->u32{let pageOffset=owners[5];if(pageOffset<=16u){return 0u;}let cap=pageOffset-16u;let count=min(owners[1],cap);let key=logical+1u;var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;if(owners[16u+middle]<key){low=middle+1u;}else{high=middle;}}if(low>=count||owners[16u+low]!=key){return 0u;}return owners[pageOffset+low];}
fn ownerAt(cell: vec3u) -> Owner {
  let bd=(dims()+vec3u(7u))/8u;let b=cell/8u;let logical=b.x+b.y*bd.x+b.z*bd.x*bd.y;let encoded=ownerPageEncoded(logical);let capacity=owners[3];
  if(encoded==0u||encoded==0xffffffffu||encoded>capacity){return canonicalOwner(cell);}let local=cell%vec3u(8u);let word=owners[owners[6]+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u];
  if(word==0u){return canonicalOwner(cell);}return decodePagedOwner(word,cell);
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
fn frontierBase(which:u32)->u32{return 6u+which*params.pressureCapacity.x;}
fn mortonPart10(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn rowMorton(cell:u32)->u32{let p=unpackOrigin(cell);return mortonPart10(p.x)|(mortonPart10(p.y)<<1u)|(mortonPart10(p.z)<<2u);}
fn pressureRow(owner: Owner) -> u32 {
  let cell = index(unpackOrigin(owner.packedOrigin));
  if (!rowIndexedPressure) { return cell; }
  let current=frontier[2];let count=min(frontier[current],params.pressureCapacity.x);
  let level=u32(firstTrailingBit(owner.size));let morton=rowMorton(cell);var lo=0u;var hi=count;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontier[frontierBase(current)+mid];
    let otherOwner=ownerAt(unpackOrigin(other));let otherLevel=u32(firstTrailingBit(otherOwner.size));
    let otherMorton=rowMorton(other);if(otherLevel<level||(otherLevel==level&&(otherMorton<morton
      ||(otherMorton==morton&&other<cell)))){lo=mid+1u;}else{hi=mid;}}
  return select(0xffffffffu,lo,lo<count&&frontier[frontierBase(current)+lo]==cell);
}
@compute @workgroup_size(4,4,4)
fn materializeOctreeFields(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(gid); let origin = unpackOrigin(owner.packedOrigin);
  let horizontal = origin.x | (origin.z << 10u) | (owner.size << 20u);
  let vertical = origin.y | ((origin.y + owner.size) << 10u);
  textureStore(topologyOut, vec3i(gid), vec4u(horizontal, vertical, 0u, 0u));
  let wet = liquidOwner(owner); let invalid = 0xffffffffu; let row = pressureRow(owner);
  let q = vec3i(gid);
  // Pressure ownership remains useful to generic scientific slices. Compact
  // velocity and Projection Δu are rendered by the native technique overlay,
  // never reconstructed through a dense compatibility texture.
  textureStore(pressureSamplesOut, q, select(vec4u(invalid), vec4u(row, 0u, vertical, horizontal), wet));
  var centrePressure = 0.0;
  if (row < arrayLength(&pressure)) { centrePressure = pressure[row]; }
  textureStore(pressureOut, vec3i(gid), vec4f(select(0.0, centrePressure, wet)));
}
`;
