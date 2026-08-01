/**
 * GPU producer for the sparse positive-air identity domain used by the
 * Section 5 face-velocity extrapolation path.
 *
 * The producer completes the Section 5 chain: it publishes deduplicated cell
 * identities, seeds ordinary octree faces by copying the coplanar accepted
 * projected power-face samples, copies the closest-interface value through
 * the face graph, and then
 * reconstructs regular/power-cell vectors.  The suffix publication is
 * committed only after every demanded support vector validates.
 */

import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import {
  STRUCTURED_AIR_SUPPORT_ARENA_MAGIC,
  STRUCTURED_AIR_SUPPORT_ARENA_VERSION,
  STRUCTURED_AIR_SUPPORT_ARENA_FLAGS,
  STRUCTURED_AIR_SUPPORT_CONTROL_WORDS,
  STRUCTURED_AIR_SUPPORT_INVALID,
  STRUCTURED_AIR_SUPPORT_RECORD_FLAGS,
  STRUCTURED_AIR_SUPPORT_RECORD_WORDS,
  STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS,
  STRUCTURED_AIR_SUPPORT_VECTOR_WORDS,
  planStructuredAirSupportArena,
  type StructuredAirSupportArenaLayout,
} from "./octree-structured-air-support";
import {
  OCTREE_AIR_SUPPORT_CONTROL_WORDS,
  OCTREE_AIR_SUPPORT_INVALID,
  OCTREE_AIR_SUPPORT_LAYOUT_VERSION,
  OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE,
  OCTREE_AIR_SUPPORT_SELECTOR_STRIDE,
  OCTREE_AIR_SUPPORT_TAG,
  OCTREE_AIR_SUPPORT_VALID,
  planOctreeAirVelocitySupport,
  type OctreeAirVelocitySupportLayout,
} from "./webgpu-octree-air-velocity-support";
import {
  OCTREE_OWNER_PAGE_CONTROL_WORDS,
  OCTREE_OWNER_PAGE_PUBLICATION_STATUS,
  OCTREE_OWNER_PAGE_VOXELS,
  octreeOwnerPageLookupWgsl,
  type WebGPUOctreeSimulationOwnerPages,
} from "./webgpu-octree-owner-pages";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import type { DirectStructuredVelocitySource } from "./webgpu-octree-structured-velocity-gpu";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE = 256;
export const OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX = 12;
export const OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS =
  3 * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra;
export const OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE =
  OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE + OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS;
/** `{cell, size, tagWord}`. The demand flags a candidate used to carry were
 * write-only: every reader (mark, scatter, tag resolution) takes the flags from
 * the per-cell `directoryFlags` word, which is the deduplicated authority. */
export const OCTREE_AIR_SUPPORT_GPU_CANDIDATE_WORDS = 3;
export const OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_WAVES = 12;
export const OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_GROUP = 2;
/** Occupancy-wide sparse waves before the exact residual frontier tail. */
export const OCTREE_AIR_SUPPORT_GPU_PARALLEL_FRONTIER_WAVES = 12;
/** Words 41/42 are the stationary-air fallback latch: count of face patches
 * the march never reached, and the first such (cell<<3)|axis identity.
 * Words 43-46 retain the construction-stable dense oracle's march ledger; the
 * sparse production path keeps its hot counters in frontier words 0-10.
 * Words 47/48 are the same-epoch topology-reuse latch and preceding support-row
 * count. Word 49 retains the preceding seed-list count; the list itself reuses
 * dead candidate scratch after identity publication. Word 50 admits the exact
 * retained-solution refresh only when that list and the reciprocal graph were
 * published by the preceding sparse transaction. Words 51-59 durably latch
 * the first stage-6 rejected identity/reason before later transactions may
 * reuse its record slot. */
export const OCTREE_AIR_SUPPORT_GPU_SCRATCH_CONTROL_WORDS = 60;
export const OCTREE_AIR_SUPPORT_GPU_INDIRECT_RECORDS = 6;
export const OCTREE_AIR_SUPPORT_GPU_FACE_WORDS = 4;
/**
 * Origin coordinate (3 words) and extent (1 word) of a face row's cell,
 * resolved once by `resolveAirSupportFaceAdjacency` and appended to that row's
 * adjacency record. `faceCenter` used to re-derive both on every candidate of
 * the march's 30x4 scan, and re-deriving the origin means `coord()`'s three
 * emulated integer divisions by the runtime domain dimensions.
 */
export const OCTREE_AIR_SUPPORT_GPU_FACE_GEOMETRY_WORDS = 4;
export const OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE =
  1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence + 2 * STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS
  + OCTREE_AIR_SUPPORT_GPU_FACE_GEOMETRY_WORDS;

export const OCTREE_AIR_SUPPORT_GPU_ERROR = Object.freeze({
  source: 1 << 0,
  generation: 1 << 1,
  capacity: 1 << 2,
  topology: 1 << 3,
  catalog: 1 << 4,
  tag: 1 << 5,
} as const);

/** High-byte namespace for the existing bounded first-error word. */
export const OCTREE_AIR_SUPPORT_GPU_TOPOLOGY_STAGE = Object.freeze({
  demandIdentity: 1,
  fineClosureIdentity: 2,
  fineClosureTopology: 3,
  fineCandidateTopology: 4,
  tagIdentity: 5,
  supportTopology: 6,
  faceTopology: 7,
  faceReconstruction: 8,
  frontierReciprocity: 9,
  retainedRefresh: 10,
} as const);

export function decodeOctreeAirSupportGPUFirstError(packed: number) {
  const word = Number(packed) >>> 0;
  return Object.freeze({ stage: word >>> 24, item: word & 0x00ff_ffff });
}

/**
 * CPU oracle for the factor-1 three-axis demand dilation. A separable
 * max/boolean dilation by radius r is exactly the cubic Chebyshev
 * neighbourhood `[-r,r]^3`; this helper pins the GPU transformation
 * bit-for-bit without depending on shader execution.
 */
export function dilateFactorOneAirSupportDemand(
  dimensions: readonly [number, number, number],
  source: Uint8Array,
  radius: number,
): Uint8Array {
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
    || !Number.isSafeInteger(radius) || radius < 0) {
    throw new RangeError("Factor-1 demand dilation requires positive dimensions and a non-negative radius");
  }
  const [nx, ny, nz] = dimensions;
  const volume = nx * ny * nz;
  if (source.length !== volume) {
    throw new RangeError("Factor-1 demand mask size does not match its dimensions");
  }
  const axisPass = (input: Uint8Array, axis: 0 | 1 | 2): Uint8Array => {
    const output = new Uint8Array(volume);
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          const q = [x, y, z];
          for (let delta = -radius; delta <= radius; delta += 1) {
            q[axis] = (axis === 0 ? x : axis === 1 ? y : z) + delta;
            if (q[axis] < 0 || q[axis] >= dimensions[axis]) continue;
            const index = q[0] + nx * (q[1] + ny * q[2]);
            if (input[index] !== 0) {
              output[x + nx * (y + ny * z)] = 1;
              break;
            }
          }
        }
      }
    }
    return output;
  };
  return axisPass(axisPass(axisPass(source, 0), 1), 2);
}

/** CPU oracle for the three factor-1 frontier records published by the
 * singleton wave advance: occupancy-wide phase work, the next singleton, and
 * the exact residual tail. A converged or failed wave publishes three zero
 * records, so every later host-authored wave dispatches zero workgroups. */
export function factorOneAirSupportFrontierIndirectRecords(
  faceRows: number,
  changedFaces: number,
  clean = true,
): readonly number[] {
  if (!Number.isSafeInteger(faceRows) || faceRows < 0
    || !Number.isSafeInteger(changedFaces) || changedFaces < 0) {
    throw new RangeError("Factor-1 frontier schedules require non-negative integer counts");
  }
  const active = clean && changedFaces > 0;
  const dispatchFor = (count: number, size: number): readonly [number, number, number] => {
    const groups = Math.ceil(count / size);
    const x = Math.min(groups, 65_535);
    return [x, x > 0 ? Math.ceil(groups / x) : 1, 1];
  };
  return Object.freeze([
    ...(active ? dispatchFor(12 * faceRows, OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE) : [0, 1, 1]),
    ...(active ? [1, 1, 1] : [0, 1, 1]),
    ...(active ? [3, 1, 1] : [0, 1, 1]),
  ]);
}

/** Same-epoch Section 5 identity/topology reuse is the production path. An
 * explicit zero retains the full rebuild as a process-local A/B oracle. */
export function octreeAirSupportTopologyReuseEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_AIR_SUPPORT_TOPOLOGY_REUSE !== "0";
}

/** Sparse changed-frontier marching is the production path. Explicit zero
 * restores the preceding fixed 12+12+exact-tail schedule as a construction-
 * stable A/B oracle: both pipeline families and all arenas are still built. */
export function octreeAirSupportChangedFrontierEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_AIR_SUPPORT_CHANGED_FRONTIER !== "0";
}

/** Factor-1 wave convergence can be copied into indirect records for profiling.
 * It is opt-in because the 24 extra pass boundaries cost more than the empty
 * post-convergence dispatches on the coarse mini lane. Factor 4/8 never select
 * this specialization. */
export function octreeAirSupportIndirectFrontierGateEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE === "1";
}

/** GPU-authored live-page demand dispatch is the production path. Explicit
 * zero preserves the former provisioned-capacity launch as an exact A/B oracle. */
export function octreeAirSupportCompactFineDemandEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND !== "0";
}

/** Compact reuse-only demand-cell listing is retained as an exact experiment.
 * It is not the default: broad closures make its append atomics wall-neutral. */
export function octreeAirSupportCompactFineCellsEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_CELLS === "1";
}

/** The march and reconstruction exchange only face/storage payloads while
 * reading the same immutable indirect record. Zero restores the old split. */
export function octreeAirSupportReconstructionCompactPassEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_AIR_SUPPORT_RECONSTRUCT_COMPACT_PASS !== "0";
}

/** Encode the optional boundary between the Section 5 fixed point and its
 * storage-only reconstruction. Tests call this production helper directly. */
export function encodeOctreeAirSupportReconstructionHandoff(
  broker: PassBroker,
  environment?: Readonly<Record<string, string | undefined>>,
): void {
  if (!octreeAirSupportReconstructionCompactPassEnabled(environment)) {
    broker.fence("Section 5 closest-face fixed point published");
  }
}

/** Exact per-entry bind reachability. Binding zero/eleven are uniforms; all
 * other entries are storage resources and no pipeline reaches more than ten. */
export const OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS = Object.freeze({
  beginAirSupportPublication: Object.freeze([0,1,3,7,8,9,10,29]),
  clearAirSupportDirectory: Object.freeze([0,7]),
  clearAirSupportCandidates: Object.freeze([0,2,7]),
  clearAirSupportTags: Object.freeze([0,7,9]),
  emitAirSupportCandidates: Object.freeze([0,2,3,4,5,6,7,9,11,18]),
  markAndScanAirSupportCandidates: Object.freeze([0,7]),
  prefixAirSupportBlocks: Object.freeze([0,7]),
  scatterAirSupportRecords: Object.freeze([0,7,8]),
  resolveAirSupportTags: Object.freeze([0,7,8,9]),
  resolveAirSupportTopology: Object.freeze([0,3,7,8,11,12,13,14]),
  prepareFineBandAirSupportDemand: Object.freeze([0,7,26]),
  markFineBandAirSupportDemand: Object.freeze([0,7,25,26,27,28]),
  dilateFineBandAirSupportDemandX: Object.freeze([0,7]),
  dilateFineBandAirSupportDemandY: Object.freeze([0,7]),
  dilateFineBandAirSupportDemandZ: Object.freeze([0,7]),
  closeFineBandAirSupportInterpolationDemand: Object.freeze([0,2,3,4,5,6,7,11,12,13,14]),
  emitFineBandAirSupportCandidates: Object.freeze([0,2,3,7,11]),
  publishAirSupportOwnerDirectory: Object.freeze([0,2,3,7,8,9,11]),
  prepareAirSupportFaces: Object.freeze([0,7,29]),
  resolveAirSupportFaceAdjacency: Object.freeze([0,2,3,7,8,11,15,16,23]),
  validateAirSupportFrontierReciprocity: Object.freeze([0,7,23]),
  seedAirSupportFaces: Object.freeze([0,1,2,7,8,15,16,18,19,21,23]),
  seedRetainedAirSupportFaces: Object.freeze([0,1,2,7,8,15,16,18,20,21,23]),
  compactAirSupportSeedFrontier: Object.freeze([0,7,19,29]),
  refreshRetainedAirSupportFaceValues: Object.freeze([7,19,20]),
  finalizeRetainedAirSupportMarchSchedule: Object.freeze([0,7,29]),
  expandAirSupportChangedFrontier: Object.freeze([0,7,23,29]),
  relaxAirSupportChangedFrontier: Object.freeze([0,2,7,8,19,20,23,29]),
  commitAirSupportChangedFrontier: Object.freeze([0,7,19,20,29]),
  advanceAirSupportChangedFrontier: Object.freeze([0,7,29]),
  marchAirSupportFacesChangedFrontier: Object.freeze([0,2,7,8,19,20,23,29]),
  extendAirSupportFacesAtoB: Object.freeze([0,2,7,8,19,20,23]),
  extendAirSupportFacesBtoA: Object.freeze([0,2,7,8,19,20,23]),
  advanceAirSupportMarchWave: Object.freeze([7]),
  marchAirSupportFacesToFixedPoint: Object.freeze([0,2,7,8,19,20,23]),
  completeAirSupportIncidentFaces: Object.freeze([0,2,7,8,19,23,30]),
  reconstructAirSupportVectors: Object.freeze([0,2,7,8,15,16,19,22,23,24,30]),
  finalizeAirSupportMetadata: Object.freeze([0,2,7,8,9,22]),
  commitAirSupportDirectRows: Object.freeze([0,2,7,17,22]),
  commitAirSupportPublication: Object.freeze([0,7,8,9]),
} as const);
export type OctreeAirSupportGPUEntryPoint = keyof typeof OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS;

type OctreeAirSupportGPUPipelineBundle = Readonly<Record<string, GPUComputePipeline>>;

/** Pipelines contain no instance resources; all buffers live in the bind
 * groups assembled by `assignPipelineState`. Keep one exact entry-point bundle
 * per device and let concurrently initialized producers join the same work. */
const octreeAirSupportPipelineCache = new WeakMap<GPUDevice,
  Map<string, OctreeAirSupportGPUPipelineBundle>>();
const octreeAirSupportPipelineCompilations = new WeakMap<GPUDevice,
  Map<string, Promise<OctreeAirSupportGPUPipelineBundle>>>();

export interface OctreeAirVelocitySupportGPUPlan {
  readonly rowCapacity: number;
  readonly slotCapacity: number;
  readonly domainVolume: number;
  readonly candidateStride: number;
  readonly fineCandidateOffset: number;
  readonly candidateCapacity: number;
  readonly candidateBlockCapacity: number;
  readonly faceCellCapacity: number;
  readonly faceCapacity: number;
  readonly faceBytes: number;
  readonly faceAdjacencyStride: number;
  readonly faceAdjacencyBytes: number;
  readonly faceFrontierBytes: number;
  readonly directAirVectorBytes: number;
  readonly support: OctreeAirVelocitySupportLayout;
  readonly records: StructuredAirSupportArenaLayout;
  readonly scratchWords: number;
  readonly scratchBytes: number;
  readonly indirectBytes: number;
  readonly offsets: Readonly<{
    control: 0;
    candidates: number;
    ranks: number;
    directoryWinners: number;
    directoryFlags: number;
    /** Dense cell -> published-row slot. Derived in WGSL as
     * `directoryFlags + domainVolume`, so it costs no uniform word. */
    rowIndex: number;
    blockCounts: number;
    blockOffsets: number;
  }>;
  readonly allocatedBytes: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function checkedProduct(label: string, ...values: number[]): number {
  const value = values.reduce((product, part) => product * part, 1);
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} exceeds exact host addressing`);
  return value;
}

export function planOctreeAirVelocitySupportGPU(
  rowCapacityValue: number,
  slotCapacityValue: number,
  dimensionsValue: readonly [number, number, number],
  alignment = 256,
): OctreeAirVelocitySupportGPUPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Air-support GPU row capacity");
  const slotCapacity = positiveInteger(slotCapacityValue, "Air-support GPU slot capacity");
  const dimensions = dimensionsValue.map((value, axis) =>
    positiveInteger(value, `Air-support GPU dimension ${axis}`)) as [number, number, number];
  const domainVolume = checkedProduct("Air-support GPU domain volume", ...dimensions);
  const candidateStride = OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE;
  const fineCandidateOffset = checkedProduct("Air-support GPU row candidate capacity", rowCapacity, candidateStride);
  const candidateCapacity = fineCandidateOffset + domainVolume;
  const candidateBlockCapacity = Math.ceil(candidateCapacity / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE);
  // The encoded schedule uses bounded 2-D dispatch records. The shader's
  // linearItem helper preserves canonical row-major candidate order.
  if (Math.ceil(candidateCapacity / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE) > 65_535 ** 2
      || Math.ceil(rowCapacity * (OCTREE_AIR_SUPPORT_SELECTOR_STRIDE
        + OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE) / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE)
        > 65_535 ** 2) {
    throw new RangeError("Air-support GPU schedule exceeds the 2-D indirect dispatch limit");
  }
  const support = planOctreeAirVelocitySupport(rowCapacity, slotCapacity, alignment, domainVolume);
  const records = planStructuredAirSupportArena(support.supportCapacity);
  const faceCellCapacity = rowCapacity + support.supportCapacity;
  const faceCapacity = checkedProduct("Air-support ordinary face capacity",
    faceCellCapacity, STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS);
  const faceBytes = checkedProduct("Air-support ordinary face bytes",
    faceCapacity, OCTREE_AIR_SUPPORT_GPU_FACE_WORDS, 4);
  const faceAdjacencyStride = OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE;
  const faceAdjacencyBytes = checkedProduct("Air-support face-adjacency bytes",
    faceCellCapacity, faceAdjacencyStride, 4);
  // 16 control words, two face-capacity queues, and one generation mark per
  // face. The three axis queues partition each face-capacity bank exactly.
  const faceFrontierBytes = checkedProduct("Air-support changed-frontier bytes",
    16 + 3 * faceCapacity, 4);
  const directAirVectorBytes = checkedProduct("Air-support direct-air staging bytes", rowCapacity, 16);
  const offsets = {
    control: 0 as const,
    candidates: OCTREE_AIR_SUPPORT_GPU_SCRATCH_CONTROL_WORDS,
    ranks: 0,
    directoryWinners: 0,
    directoryFlags: 0,
    rowIndex: 0,
    blockCounts: 0,
    blockOffsets: 0,
  };
  offsets.ranks = offsets.candidates + OCTREE_AIR_SUPPORT_GPU_CANDIDATE_WORDS * candidateCapacity;
  offsets.directoryWinners = offsets.ranks + candidateCapacity;
  offsets.directoryFlags = offsets.directoryWinners + domainVolume;
  // Kept adjacent to directoryFlags on purpose: the shader addresses it as
  // `directoryFlagOffset + domainVolume` instead of spending a uniform word,
  // and the uniform block's vec3u members stay 16-byte aligned.
  offsets.rowIndex = offsets.directoryFlags + domainVolume;
  offsets.blockCounts = offsets.rowIndex + domainVolume;
  offsets.blockOffsets = offsets.blockCounts + candidateBlockCapacity;
  // A dense one-bit (stored as u32) source mask backs the factor-1 demand
  // dilation. Keeping it as a separate terminal arena lets the closure read a
  // stable mask while it publishes the final directory flags.
  const scratchWords = offsets.blockOffsets + candidateBlockCapacity + 2 * domainVolume;
  const scratchBytes = scratchWords * 4;
  const indirectBytes = OCTREE_AIR_SUPPORT_GPU_INDIRECT_RECORDS * 12;
  return Object.freeze({ rowCapacity, slotCapacity, domainVolume, candidateStride, fineCandidateOffset,
    candidateCapacity, candidateBlockCapacity, faceCellCapacity, faceCapacity, faceBytes,
    faceAdjacencyStride, faceAdjacencyBytes, faceFrontierBytes, directAirVectorBytes,
    support, records, scratchWords,
    scratchBytes, indirectBytes, offsets: Object.freeze(offsets),
    allocatedBytes: support.totalBytes + records.allocatedBytes + 3 * faceBytes + faceAdjacencyBytes + faceFrontierBytes + directAirVectorBytes
      + scratchBytes + indirectBytes + 512 });
}

export interface OctreeAirVelocitySupportGPUInputs {
  /**
   * Accepted structured authority for the current topology epoch.  Encode
   * this producer immediately after topology/structured/boundary acceptance,
   * while rowVelocities still carries the migrated prior projected field.
   */
  readonly structured: DirectStructuredVelocitySource;
  readonly topology: OctreePowerTopologySource;
  readonly owners: Pick<WebGPUOctreeSimulationOwnerPages, "plan" | "arena">;
  readonly boundaryEpoch: { readonly buffer: GPUBuffer; readonly offsetWords: number };
  /** Accepted A/B liquid bit paired with the boundary epoch. */
  readonly liquidMask: GPUBuffer;
  /**
   * Optional production selector arena (the power-coarse-level-set schedule's
   * selectorRows buffer). Its transport-metric prefix is preserved verbatim.
   */
  readonly sharedArena?: GPUBuffer;
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  /** Same configured upper bound supplied to fine transport backtraces. */
  readonly maximumDisplacementFineCells: number;
  /** Structured world-boundary bits in x-/x+/y-/y+/z-/z+ order. */
  readonly closedBoundaryMask: number;
  /** Stable A/B fine payloads; the producer selects the currently published
   * generation without creating bind groups on the recurring path. */
  readonly fineSources?: readonly [WebGPUFineLevelSetBrickSource, WebGPUFineLevelSetBrickSource];
  /** Complete authored destination band from Section 5, in fine cells. */
  readonly transportBandFineCells?: number;
}

export interface OctreeAirVelocitySupportGPUSource {
  readonly plan: OctreeAirVelocitySupportGPUPlan;
  /** Suffix-compatible transport metrics, selector/regular tags, control, vectors. */
  readonly arena: GPUBuffer;
  /** Immutable support identities plus a mirror of committed support vectors. */
  readonly recordArena: GPUBuffer;
  readonly selectorTagOffsetWords: number;
  readonly regularTagOffsetWords: number;
  readonly controlOffsetWords: number;
  readonly supportVectorOffsetWords: number;
  readonly recordOffsetWords: number;
  readonly recordVectorOffsetWords: number;
  /** Canonical banked full-vector output for every accepted direct row. */
  readonly canonicalRowVelocities: GPUBuffer;
}

/**
 * Standalone producer. Production consumers must resolve positive-air tags
 * against `arena`/`supportVectorOffsetWords`; compact wet rows continue to use
 * the canonical accepted `structured.rowVelocities`.  In particular, no
 * unbanked row-CPT buffer is a substitute for this suffix authority.
 */
export class WebGPUOctreeAirVelocitySupportProducer {
  readonly plan: OctreeAirVelocitySupportGPUPlan;
  readonly arena: GPUBuffer;
  readonly recordArena: GPUBuffer;
  readonly scratch: GPUBuffer;
  readonly indirect: GPUBuffer;
  readonly faceA: GPUBuffer;
  readonly faceB: GPUBuffer;
  readonly incidentFaces: GPUBuffer;
  readonly faceAdjacency: GPUBuffer;
  readonly faceFrontier: GPUBuffer;
  readonly directAirVectors: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: readonly [GPUBuffer, GPUBuffer];
  private readonly ownerParams: GPUBuffer;
  private readonly shaderModule: GPUShaderModule;
  private readonly pipelineCacheKey: string;
  private pipelines!: Readonly<Record<string, GPUComputePipeline>>;
  private groups!: readonly [Readonly<Record<string, GPUBindGroup>>,
    Readonly<Record<string, GPUBindGroup>>];
  private fineDemandGroups?: readonly [readonly [GPUBindGroup, GPUBindGroup],
    readonly [GPUBindGroup, GPUBindGroup]];
  private fineDemandScheduleGroups?: readonly [readonly [GPUBindGroup, GPUBindGroup],
    readonly [GPUBindGroup, GPUBindGroup]];
  private readonly ownsArena: boolean;
  private pipelinesInitialized = false;
  private pipelineInitialization?: Promise<void>;
  private destroyed = false;
  private publicationCount = 0;
  private parameterSlot: 0 | 1 = 0;

  constructor(private readonly device: GPUDevice, private readonly inputs: OctreeAirVelocitySupportGPUInputs,
    deferPipelineCompilation = false) {
    const { structured, topology, owners } = inputs;
    const finePlansMatch = !inputs.fineSources || (() => {
      const [a, b] = inputs.fineSources.map((source) => source.plan);
      return a.fineFactor === b.fineFactor && a.brickResolution === b.brickResolution
        && a.samplesPerBrick === b.samplesPerBrick
        && a.maximumResidentBricks === b.maximumResidentBricks
        && a.fineCellWidth === b.fineCellWidth
        && a.brickDimensions.every((value, axis) => value === b.brickDimensions[axis])
        && a.sampleDimensions.every((value, axis) => value === b.sampleDimensions[axis])
        && a.domainOrigin.every((value, axis) => value === b.domainOrigin[axis]);
    })();
    this.plan = planOctreeAirVelocitySupportGPU(structured.plan.rowCapacity,
      structured.plan.slotCapacity, inputs.dimensions, device.limits.minStorageBufferOffsetAlignment);
    if (!topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra
      || !topology.catalogTetrahedronVertices || !topology.catalogTetrahedronVertexCount
      || !Number.isSafeInteger(inputs.maximumLeafSize) || inputs.maximumLeafSize < 1
      || (inputs.maximumLeafSize & (inputs.maximumLeafSize - 1)) !== 0
      || !Number.isSafeInteger(inputs.maximumDisplacementFineCells)
      || inputs.maximumDisplacementFineCells < 1
      || !Number.isSafeInteger(inputs.closedBoundaryMask) || inputs.closedBoundaryMask < 0
      || inputs.closedBoundaryMask > 0x3f
      || owners.plan.dimensions.some((value, axis) => value !== inputs.dimensions[axis])
      || !Number.isSafeInteger(inputs.boundaryEpoch.offsetWords) || inputs.boundaryEpoch.offsetWords < 0
      || inputs.boundaryEpoch.offsetWords * 4 + 4 > inputs.boundaryEpoch.buffer.size
      || inputs.liquidMask.size < 2 * structured.plan.rowCapacity * 4
      || structured.rowVelocities.size < 2 * structured.plan.rowCapacity * 16
      || inputs.fineSources && (!Number.isSafeInteger(inputs.transportBandFineCells)
        || inputs.transportBandFineCells! < 1 || inputs.transportBandFineCells! > 256
        || !finePlansMatch)
      || inputs.sharedArena && inputs.sharedArena.size < this.plan.support.totalBytes) {
      throw new RangeError("Air-support GPU inputs are invalid, incomplete, or exceed the published extension depth");
    }
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    for (const [label, bytes] of [["support", this.plan.support.totalBytes],
      ["record", this.plan.records.allocatedBytes], ["scratch", this.plan.scratchBytes],
      ["face", this.plan.faceBytes], ["face-adjacency", this.plan.faceAdjacencyBytes],
      ["face-frontier", this.plan.faceFrontierBytes],
      ["direct-air", this.plan.directAirVectorBytes]] as const) {
      if (bytes > maximumBinding) throw new RangeError(`Air-support GPU ${label} arena exceeds binding limits`);
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.ownsArena = !inputs.sharedArena;
    this.arena = inputs.sharedArena ?? device.createBuffer({
      label: "Structured air-velocity support tags and vectors",
      size: this.plan.support.totalBytes, usage: storage });
    this.recordArena = device.createBuffer({ label: "Structured air-support identity records",
      size: this.plan.records.allocatedBytes, usage: storage });
    this.scratch = device.createBuffer({ label: "Structured air-support mark scan scatter scratch",
      size: this.plan.scratchBytes, usage: storage });
    this.faceA = device.createBuffer({ label: "Structured ordinary-face extension A",
      size: this.plan.faceBytes, usage: storage });
    this.faceB = device.createBuffer({ label: "Structured ordinary-face extension B",
      size: this.plan.faceBytes, usage: storage });
    this.incidentFaces = device.createBuffer({ label: "Retained missing incident-face closure",
      size: this.plan.faceBytes, usage: storage });
    this.faceAdjacency = device.createBuffer({ label: "Published structured ordinary-face adjacency",
      size: this.plan.faceAdjacencyBytes, usage: storage });
    this.faceFrontier = device.createBuffer({ label: "Structured ordinary-face changed frontier",
      size: this.plan.faceFrontierBytes, usage: storage });
    this.directAirVectors = device.createBuffer({ label: "Staged structured direct-air vectors",
      size: this.plan.directAirVectorBytes, usage: storage });
    this.indirect = device.createBuffer({ label: "Structured air-support indirect schedules",
      size: this.plan.indirectBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.params = Object.freeze([0, 1].map((slot) => device.createBuffer({
      label: `Structured air-support publication parameters ${slot}`, size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }))) as unknown as readonly [GPUBuffer, GPUBuffer];
    this.ownerParams = device.createBuffer({ label: "Structured air-support owner lookup parameters", size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const ownerPlan = owners.plan;
    device.queue.writeBuffer(this.ownerParams, 0, Uint32Array.from([
      ...inputs.dimensions, inputs.maximumLeafSize,
      ...ownerPlan.brickDimensions, ownerPlan.logicalBrickCount,
      ownerPlan.ownerDirectoryOffsetWords, ownerPlan.ownerPagesOffsetWords,
      ownerPlan.capacity, ownerPlan.pageVoxels,
    ]));
    this.shaderModule = device.createShaderModule({ label: "Structured positive-air identity publication",
      code: octreeAirVelocitySupportPublicationWGSL });
    this.pipelineCacheKey = this.pipelineEntryPoints().join("\0");
    if (!deferPipelineCompilation) this.createPipelinesSync();
    this.allocatedBytes = this.plan.allocatedBytes + 256
      - (inputs.sharedArena ? this.plan.support.totalBytes : 0);
  }

  private pipelineEntryPoints(): readonly OctreeAirSupportGPUEntryPoint[] {
    return Object.keys(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS) as OctreeAirSupportGPUEntryPoint[];
  }

  private pipelineDescriptor(entryPoint: OctreeAirSupportGPUEntryPoint): GPUComputePipelineDescriptor {
    return { label: entryPoint, layout: "auto",
      compute: { module: this.shaderModule, entryPoint } };
  }

  private assignPipelineState(pipelines: Readonly<Record<string, GPUComputePipeline>>): void {
    const { structured, topology, owners } = this.inputs;
    const entries = this.pipelineEntryPoints();
    const buffers = new Map<number, GPUBuffer>([
      [1, structured.control], [2, structured.rowGeometry],
      [3, owners.arena], [4, topology.catalogTetrahedronHeaders!],
      [5, topology.catalogTetrahedra!], [6, topology.catalogTetrahedronVertices!],
      [7, this.scratch], [8, this.recordArena], [9, this.arena],
      [10, this.inputs.boundaryEpoch.buffer], [11, this.ownerParams],
      [12, topology.sameOrFinerDirect], [13, topology.sameOrCoarserDirect],
      [14, topology.catalogLookup], [15, topology.catalogFaces],
      [16, topology.reconstructionData], [17, structured.rowVelocities],
      [18, this.inputs.liquidMask], [19, this.faceA], [20, this.faceB],
      [21, structured.authority], [22, this.directAirVectors],
      [23, this.faceAdjacency], [24, this.arena], [29, this.faceFrontier], [30, this.incidentFaces],
      ...(this.inputs.fineSources ? [[25, this.inputs.fineSources[0].metadata],
        [26, this.inputs.fineSources[0].worklist], [27, this.inputs.fineSources[0].flags],
        [28, this.inputs.fineSources[0].phi]] as const : []),
    ]);
    const resource = (params: GPUBuffer, entry: OctreeAirSupportGPUEntryPoint,
      binding: number): GPUBufferBinding => {
      if (binding === 0) return { buffer: params };
      if (entry === "reconstructAirSupportVectors" && binding === 24) {
        return { buffer: this.arena, offset: this.plan.support.supportVectorOffsetBytes,
          size: this.plan.support.supportVectorBytes };
      }
      return { buffer: buffers.get(binding)! };
    };
    const fineOnlyEntries = new Set<OctreeAirSupportGPUEntryPoint>([
      "prepareFineBandAirSupportDemand", "markFineBandAirSupportDemand",
      "dilateFineBandAirSupportDemandX", "dilateFineBandAirSupportDemandY",
      "dilateFineBandAirSupportDemandZ", "closeFineBandAirSupportInterpolationDemand",
      "emitFineBandAirSupportCandidates",
    ]);
    const configuredEntries = entries.filter((entry) => this.inputs.fineSources
      || !fineOnlyEntries.has(entry));
    const makeGroups = (params: GPUBuffer) => Object.freeze(Object.fromEntries(
      configuredEntries.map((entry) => [entry, this.device.createBindGroup({
        layout: pipelines[entry]!.getBindGroupLayout(0),
        entries: OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS[entry]
          .map((binding) => ({ binding, resource: resource(params, entry, binding) })),
      })])));
    const groups = Object.freeze(this.params.map(makeGroups)) as unknown as
      readonly [Readonly<Record<string, GPUBindGroup>>, Readonly<Record<string, GPUBindGroup>>];
    let fineDemandGroups: typeof this.fineDemandGroups;
    let fineDemandScheduleGroups: typeof this.fineDemandScheduleGroups;
    if (this.inputs.fineSources) {
      const makeFineGroup = (entry: "prepareFineBandAirSupportDemand" | "markFineBandAirSupportDemand",
        params: GPUBuffer, fine: WebGPUFineLevelSetBrickSource) => this.device.createBindGroup({
        layout: pipelines[entry]!.getBindGroupLayout(0),
        entries: OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS[entry].map((binding) => ({
          binding, resource: { buffer: binding === 0 ? params
          : binding === 25 ? fine.metadata
          : binding === 26 ? fine.worklist : binding === 27 ? fine.flags : binding === 28 ? fine.phi
            : buffers.get(binding)! } })),
      });
      fineDemandGroups = Object.freeze(this.params.map((params) => Object.freeze(
        this.inputs.fineSources!.map((fine) => makeFineGroup("markFineBandAirSupportDemand", params, fine))))) as unknown as
          NonNullable<typeof this.fineDemandGroups>;
      fineDemandScheduleGroups = Object.freeze(this.params.map((params) => Object.freeze(
        this.inputs.fineSources!.map((fine) => makeFineGroup("prepareFineBandAirSupportDemand", params, fine))))) as unknown as
          NonNullable<typeof this.fineDemandScheduleGroups>;
    }
    // Publish the pipeline-dependent state only after every pipeline and bind
    // group has been created, so encode can never observe a partial set.
    this.pipelines = Object.freeze(pipelines);
    this.groups = groups;
    this.fineDemandGroups = fineDemandGroups;
    this.fineDemandScheduleGroups = fineDemandScheduleGroups;
    this.pipelinesInitialized = true;
  }

  private createPipelinesSync(): void {
    let deviceCache = octreeAirSupportPipelineCache.get(this.device);
    if (!deviceCache) {
      deviceCache = new Map();
      octreeAirSupportPipelineCache.set(this.device, deviceCache);
    }
    let pipelines = deviceCache.get(this.pipelineCacheKey);
    if (!pipelines) {
      const compiled: Record<string, GPUComputePipeline> = {};
      for (const entryPoint of this.pipelineEntryPoints()) {
        compiled[entryPoint] = this.device.createComputePipeline(this.pipelineDescriptor(entryPoint));
      }
      pipelines = Object.freeze(compiled);
      deviceCache.set(this.pipelineCacheKey, pipelines);
    }
    this.assignPipelineState(pipelines);
  }

  async initializePipelines(): Promise<void> {
    if (this.destroyed) throw new Error("Air-support GPU producer is destroyed");
    if (this.pipelinesInitialized) return;
    if (this.pipelineInitialization) return this.pipelineInitialization;
    this.pipelineInitialization = (async () => {
      let deviceCache = octreeAirSupportPipelineCache.get(this.device);
      if (!deviceCache) {
        deviceCache = new Map();
        octreeAirSupportPipelineCache.set(this.device, deviceCache);
      }
      let pipelines = deviceCache.get(this.pipelineCacheKey);
      if (!pipelines) {
        let compilations = octreeAirSupportPipelineCompilations.get(this.device);
        if (!compilations) {
          compilations = new Map();
          octreeAirSupportPipelineCompilations.set(this.device, compilations);
        }
        let compilation = compilations.get(this.pipelineCacheKey);
        if (!compilation) {
          compilation = (async () => {
            const compiled: Record<string, GPUComputePipeline> = {};
            for (const entryPoint of this.pipelineEntryPoints()) {
              compiled[entryPoint] = await this.device.createComputePipelineAsync(
                this.pipelineDescriptor(entryPoint));
            }
            return Object.freeze(compiled);
          })().then((compiled) => {
            const published = deviceCache!.get(this.pipelineCacheKey) ?? compiled;
            deviceCache!.set(this.pipelineCacheKey, published);
            return published;
          }).finally(() => { compilations!.delete(this.pipelineCacheKey); });
          compilations.set(this.pipelineCacheKey, compilation);
        }
        pipelines = await compilation;
      }
      this.assignPipelineState(pipelines);
    })();
    return this.pipelineInitialization;
  }

  private parameterData(expectedEpoch: number, fineSlot?: 0 | 1,
    gravityDt: readonly [number, number, number] = [0, 0, 0],
    changedFrontier = octreeAirSupportChangedFrontierEnabled(),
    compactFineDemand = octreeAirSupportCompactFineDemandEnabled(),
    compactFineCells = octreeAirSupportCompactFineCellsEnabled(),
    indirectFrontierGate = octreeAirSupportIndirectFrontierGateEnabled()): ArrayBuffer {
    if (!Number.isSafeInteger(expectedEpoch) || expectedEpoch < 1 || expectedEpoch > 0xffff_ffff) {
      throw new RangeError("Air-support expected epoch must be a published uint32 generation");
    }
    const bytes = new ArrayBuffer(256), words = new Uint32Array(bytes);
    const fine = fineSlot === undefined ? undefined : this.inputs.fineSources?.[fineSlot];
    words.set([this.plan.rowCapacity, this.plan.slotCapacity, this.plan.domainVolume,
      this.plan.candidateStride, this.plan.candidateCapacity, this.plan.candidateBlockCapacity,
      this.plan.support.supportCapacity, expectedEpoch,
      ...this.inputs.dimensions, this.inputs.maximumLeafSize,
      this.plan.support.selectorTagOffsetWords, this.plan.support.regularTagOffsetWords,
      this.plan.support.controlOffsetWords, this.plan.support.supportVectorOffsetWords,
      this.plan.records.recordOffsetWords, this.plan.records.vectorOffsetWords,
      this.inputs.boundaryEpoch.offsetWords, this.inputs.topology.catalogTetrahedronVertexCount!,
      // Explicit, not `Object.values(offsets)`: `rowIndex` is derived in WGSL
      // and adding a word here would break the uniform block's vec3u alignment.
      this.plan.offsets.control, this.plan.offsets.candidates, this.plan.offsets.ranks,
      this.plan.offsets.directoryWinners, this.plan.offsets.directoryFlags,
      this.plan.offsets.blockCounts, this.plan.offsets.blockOffsets,
      this.plan.records.allocatedWords, this.plan.support.totalBytes / 4,
      this.plan.faceCellCapacity, this.plan.faceCapacity,
      this.inputs.topology.plan.lookupCount, this.inputs.topology.plan.entryCount,
      this.inputs.topology.reconstructionDataOffsetBytes / 4,
      this.inputs.topology.rowTemplateHeaderOffsetBytes / 4,
      this.inputs.structured.authorityBankStrideWords,
      this.inputs.structured.plan.maximumCaseSlots,
      this.inputs.structured.plan.offsets.values,
      this.inputs.structured.plan.offsets.rowSlotHandles,
      this.inputs.structured.plan.offsets.rowSlotSigns,
      this.inputs.structured.plan.offsets.rowCatalogSlots,
      (this.publicationCount > 0 ? 1 : 0) | (changedFrontier ? 2 : 0)
        | (compactFineCells ? 4 : 0) | (indirectFrontierGate ? 8 : 0),
      this.plan.faceAdjacencyStride,
      this.plan.support.ownerDirectoryOffsetWords,
      this.plan.fineCandidateOffset,
    ], 0);
    words[45] = fine?.plan.maximumResidentBricks ?? 0;
    words[46] = fine?.plan.fineFactor ?? 0;
    words[47] = fine ? this.inputs.transportBandFineCells ?? 0 : 0;
    words.set(fine?.plan.brickDimensions ?? [0, 0, 0], 48);
    words[51] = fine?.plan.brickResolution ?? 0;
    words.set(fine?.plan.sampleDimensions ?? [0, 0, 0], 52);
    words[55] = fine?.plan.samplesPerBrick ?? 0;
    words[56] = this.inputs.maximumDisplacementFineCells;
    words[57] = fine?.generation ?? 0;
    new Float32Array(bytes)[58] = fine?.plan.fineCellWidth ?? 0;
    words[59] = this.inputs.closedBoundaryMask;
    if (!gravityDt.every((component) => Number.isFinite(component))) {
      throw new RangeError("Air-support gravity impulse must be finite");
    }
    new Float32Array(bytes).set(gravityDt, 60);
    words[63] = octreeAirSupportTopologyReuseEnabled() ? 1 : 0;
    return bytes;
  }

  encode(broker: PassBroker, expectedEpoch: number, fineSlot?: 0 | 1,
    gravityDt?: readonly [number, number, number],
    site: "topology-commit" | "settled-fine" = "settled-fine"): void {
    if (this.destroyed) throw new Error("Air-support GPU producer is destroyed");
    if (!this.pipelinesInitialized) throw new Error("Air-support GPU pipelines are not initialized");
    if (fineSlot !== undefined && !this.inputs.fineSources) {
      throw new Error("Air-support fine-demand slot requires configured A/B fine sources");
    }
    const parameterSlot = this.parameterSlot;
    this.parameterSlot = parameterSlot === 0 ? 1 : 0;
    const changedFrontier = octreeAirSupportChangedFrontierEnabled();
    const compactFineDemand = octreeAirSupportCompactFineDemandEnabled();
    const indirectFrontierGate = changedFrontier
      && octreeAirSupportIndirectFrontierGateEnabled()
      && fineSlot !== undefined
      && this.inputs.fineSources![fineSlot].plan.fineFactor === 1;
    const params = this.params[parameterSlot], groups = this.groups[parameterSlot];
    this.device.queue.writeBuffer(params, 0,
      this.parameterData(expectedEpoch, fineSlot, gravityDt ?? [0, 0, 0], changedFrontier,
        compactFineDemand, octreeAirSupportCompactFineCellsEnabled(), indirectFrontierGate));
    this.publicationCount += 1;
    const siteLabel = (label: string) => `${label} · ${site}`;
    let pass = broker.compute({ label: siteLabel("Initialize structured air-support publication") });
    pass.setPipeline(this.pipelines.beginAirSupportPublication!);
    pass.setBindGroup(0, groups.beginAirSupportPublication!);
    pass.dispatchWorkgroups(1);
    if (fineSlot !== undefined && compactFineDemand && this.fineDemandScheduleGroups) {
      pass.setPipeline(this.pipelines.prepareFineBandAirSupportDemand!);
      pass.setBindGroup(0, this.fineDemandScheduleGroups[parameterSlot][fineSlot]);
      pass.dispatchWorkgroups(1);
    }
    // Storage-authored schedules are copied into an INDIRECT-only buffer. The
    // second copy below is the only other required pass boundary.
    broker.updateIndirectBuffer(this.scratch, 10 * 4, this.indirect, 0,
      (fineSlot === undefined || !compactFineDemand ? 4 : 5) * 12);
    broker.updateIndirectBuffer(this.scratch, 43 * 4, this.indirect, 60, 12);
    pass = broker.compute({ label: siteLabel("Publish structured air-support identities") });
    const run = (name: keyof typeof this.pipelines, indirectOffset?: number) => {
      pass.setPipeline(this.pipelines[name]!); pass.setBindGroup(0, groups[name]!);
      if (indirectOffset === undefined) pass.dispatchWorkgroups(1);
      else pass.dispatchWorkgroupsIndirect(this.indirect, indirectOffset);
    };
    run("clearAirSupportDirectory", 0);
    run("clearAirSupportCandidates", 36);
    run("clearAirSupportTags", 12);
    run("emitAirSupportCandidates", 24);
    if (fineSlot !== undefined && this.fineDemandGroups) {
      pass.setPipeline(this.pipelines.markFineBandAirSupportDemand!);
      pass.setBindGroup(0, this.fineDemandGroups[parameterSlot][fineSlot]);
      if (compactFineDemand) pass.dispatchWorkgroupsIndirect(this.indirect, 48);
      else {
        const capacity = this.inputs.fineSources![fineSlot].plan.maximumResidentBricks;
        const x = Math.min(capacity, this.device.limits.maxComputeWorkgroupsPerDimension);
        pass.dispatchWorkgroups(x, Math.ceil(capacity / x));
      }
      if (this.inputs.fineSources![fineSlot].plan.fineFactor === 1) {
        const dilationGroups = Math.ceil(this.plan.domainVolume / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE);
        for (const name of ["dilateFineBandAirSupportDemandX",
          "dilateFineBandAirSupportDemandY", "dilateFineBandAirSupportDemandZ"] as const) {
          pass.setPipeline(this.pipelines[name]!);
          pass.setBindGroup(0, groups[name]!);
          pass.dispatchWorkgroups(dilationGroups);
        }
      }
      pass.setPipeline(this.pipelines.closeFineBandAirSupportInterpolationDemand!);
      pass.setBindGroup(0, groups.closeFineBandAirSupportInterpolationDemand!);
      pass.dispatchWorkgroups(Math.ceil(this.plan.domainVolume / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE));
      pass.setPipeline(this.pipelines.emitFineBandAirSupportCandidates!);
      pass.setBindGroup(0, groups.emitFineBandAirSupportCandidates!);
      pass.dispatchWorkgroups(Math.ceil(this.plan.domainVolume / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE));
    }
    run("markAndScanAirSupportCandidates", 36);
    run("prefixAirSupportBlocks");
    run("scatterAirSupportRecords", 36);
    // Topology resolution is exact support-row work. The GPU reuse decision
    // and live count authored this record alongside the other identity
    // schedules; the later face-schedule copy safely overwrites it.
    pass.setPipeline(this.pipelines.resolveAirSupportTopology!);
    pass.setBindGroup(0, groups.resolveAirSupportTopology!);
    pass.dispatchWorkgroupsIndirect(this.indirect, 60);
    run("resolveAirSupportTags", 24);
    pass.setPipeline(this.pipelines.publishAirSupportOwnerDirectory!);
    pass.setBindGroup(0, groups.publishAirSupportOwnerDirectory!);
    pass.dispatchWorkgroups(Math.ceil(this.plan.domainVolume / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE));
    run("prepareAirSupportFaces");
    broker.updateIndirectBuffer(this.scratch, 32 * 4, this.indirect, 4 * 12, 2 * 12);
    // These indirect-publication words have completed their schedule lifetime.
    // Reuse them for terminal march depth and convergence state.
    broker.clearBuffer(this.scratch, 32 * 4, 6 * 4);
    pass = broker.compute({ label: siteLabel("Extrapolate structured ordinary faces and reconstruct support vectors") });
    pass.setPipeline(this.pipelines.resolveAirSupportFaceAdjacency!);
    pass.setBindGroup(0, groups.resolveAirSupportFaceAdjacency!);
    pass.dispatchWorkgroupsIndirect(this.indirect, 60);
    if (changedFrontier) {
      pass.setPipeline(this.pipelines.validateAirSupportFrontierReciprocity!);
      pass.setBindGroup(0, groups.validateAirSupportFrontierReciprocity!);
      pass.dispatchWorkgroupsIndirect(this.indirect, 60);
    }
    pass.setPipeline(this.pipelines.seedAirSupportFaces!); pass.setBindGroup(0, groups.seedAirSupportFaces!);
    pass.dispatchWorkgroupsIndirect(this.indirect, 48);
    if (changedFrontier) {
      pass.setPipeline(this.pipelines.seedRetainedAirSupportFaces!);
      pass.setBindGroup(0, groups.seedRetainedAirSupportFaces!);
      pass.dispatchWorkgroupsIndirect(this.indirect, 48);
      pass.setPipeline(this.pipelines.compactAirSupportSeedFrontier!);
      pass.setBindGroup(0, groups.compactAirSupportSeedFrontier!);
      pass.dispatchWorkgroupsIndirect(this.indirect, 48);
      // On an admitted same-topology publication, the winning seed identity
      // and squared distance of every settled face are immutable. Refresh the
      // seed values and update each retained carrier directly; no graph wave
      // can change its ordering tuple. Fresh publications still compact seeds
      // and author the ordinary sparse march schedule below.
      pass.setPipeline(this.pipelines.refreshRetainedAirSupportFaceValues!);
      pass.setBindGroup(0, groups.refreshRetainedAirSupportFaceValues!);
      pass.dispatchWorkgroupsIndirect(this.indirect, 60);
      pass.setPipeline(this.pipelines.finalizeRetainedAirSupportMarchSchedule!);
      pass.setBindGroup(0, groups.finalizeRetainedAirSupportMarchSchedule!);
      pass.dispatchWorkgroups(1);
    } else {
      for (let wave = 0; wave < OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX; wave += 1) {
        const name = (wave & 1) === 0 ? "extendAirSupportFacesAtoB" : "extendAirSupportFacesBtoA";
        pass.setPipeline(this.pipelines[name]!);
        pass.setBindGroup(0, groups[name]!);
        pass.dispatchWorkgroupsIndirect(this.indirect, 48);
      }
    }
    broker.fence("Section 5 ordinary-face seeds published");
    if (changedFrontier) {
      // The retained refresh publishes a zero schedule; a fresh publication
      // republishes the original face schedule. The INDIRECT copy is the
      // required visibility boundary between that GPU decision and the march.
      broker.updateIndirectBuffer(this.scratch, 32 * 4, this.indirect, 48, 12);
      if (indirectFrontierGate) {
        broker.updateIndirectBuffer(this.scratch, 10 * 4, this.indirect, 0, 3 * 12);
      }
      broker.clearBuffer(this.scratch, 32 * 4, 6 * 4);
    }
    pass = broker.compute({ label: siteLabel(changedFrontier
      ? "March Section 5 sparse changed frontier to a fixed point"
      : "March Section 5 fixed domain-wide oracle to a fixed point") });
    // One workgroup owns each independent velocity-axis graph. Seeds authored
    // the initial queues; every subsequent queue contains only faces whose
    // value actually changed. The shader expands those changes through the
    // reciprocal incidence graph, deduplicates destinations on the GPU, and
    // stops only when the next changed queue is empty. Thus settled and
    // disconnected air never enters another relaxation wave, while the same
    // |V_axis| proof bound and final no-change ledger remain authoritative.
    if (changedFrontier) {
      // Occupancy-wide sparse waves keep the GPU fed while the frontier is
      // broad. Each phase maps packed (axis,local-frontier-index) lanes across
      // the live face schedule; settled faces never execute extendFace. The
      // one-thread advance only publishes counts/reset state between dispatch
      // barriers. It cannot safely be folded into the last commit workgroup:
      // WGSL has no device-wide barrier, and a hand-rolled last-workgroup latch
      // would allow the winner to observe incomplete storage writes. Factor 1
      // instead lets this proven singleton publish the next phase, singleton,
      // and residual-tail indirect records. The persistent tail remains the
      // exact unbounded authority.
      for (let wave = 0; wave < OCTREE_AIR_SUPPORT_GPU_PARALLEL_FRONTIER_WAVES; wave += 1) {
        pass.setPipeline(this.pipelines.expandAirSupportChangedFrontier!);
        pass.setBindGroup(0, groups.expandAirSupportChangedFrontier!);
        pass.dispatchWorkgroupsIndirect(this.indirect, indirectFrontierGate ? 0 : 48);
        pass.setPipeline(this.pipelines.relaxAirSupportChangedFrontier!);
        pass.setBindGroup(0, groups.relaxAirSupportChangedFrontier!);
        pass.dispatchWorkgroupsIndirect(this.indirect, indirectFrontierGate ? 0 : 48);
        pass.setPipeline(this.pipelines.commitAirSupportChangedFrontier!);
        pass.setBindGroup(0, groups.commitAirSupportChangedFrontier!);
        pass.dispatchWorkgroupsIndirect(this.indirect, indirectFrontierGate ? 0 : 48);
        pass.setPipeline(this.pipelines.advanceAirSupportChangedFrontier!);
        pass.setBindGroup(0, groups.advanceAirSupportChangedFrontier!);
        if (indirectFrontierGate) {
          pass.dispatchWorkgroupsIndirect(this.indirect, 12);
          // The singleton's storage-authored schedules need a usage-scope
          // boundary before becoming INDIRECT. This copy is deliberately host
          // encoded; it replaces an unsafe cross-workgroup completion latch.
          broker.updateIndirectBuffer(this.scratch, 10 * 4, this.indirect, 0, 3 * 12);
          pass = broker.compute({ label: siteLabel(
            "March Section 5 sparse changed frontier to a fixed point") });
        } else {
          pass.dispatchWorkgroups(1);
        }
      }
      pass.setPipeline(this.pipelines.marchAirSupportFacesChangedFrontier!);
      pass.setBindGroup(0, groups.marchAirSupportFacesChangedFrontier!);
      if (indirectFrontierGate) pass.dispatchWorkgroupsIndirect(this.indirect, 24);
      else pass.dispatchWorkgroups(3);
    } else {
      for (let wave = 0; wave < OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_WAVES; wave += 1) {
        if (wave % OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_GROUP === 0) {
          pass.setPipeline(this.pipelines.advanceAirSupportMarchWave!);
          pass.setBindGroup(0, groups.advanceAirSupportMarchWave!);
          pass.dispatchWorkgroups(1);
        }
        const name = ((OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX + wave) & 1) === 0
          ? "extendAirSupportFacesAtoB" : "extendAirSupportFacesBtoA";
        pass.setPipeline(this.pipelines[name]!);
        pass.setBindGroup(0, groups[name]!);
        pass.dispatchWorkgroupsIndirect(this.indirect, 48);
      }
      pass.setPipeline(this.pipelines.advanceAirSupportMarchWave!);
      pass.setBindGroup(0, groups.advanceAirSupportMarchWave!);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.pipelines.marchAirSupportFacesToFixedPoint!);
      pass.setBindGroup(0, groups.marchAirSupportFacesToFixedPoint!);
      pass.dispatchWorkgroups(3);
    }
    pass.setPipeline(this.pipelines.completeAirSupportIncidentFaces!);
    pass.setBindGroup(0, groups.completeAirSupportIncidentFaces!);
    pass.dispatchWorkgroupsIndirect(this.indirect, 48);
    encodeOctreeAirSupportReconstructionHandoff(broker);
    pass = broker.compute({ label: siteLabel("Reconstruct Section 5 air-support vectors") });
    pass.setPipeline(this.pipelines.reconstructAirSupportVectors!);
    pass.setBindGroup(0, groups.reconstructAirSupportVectors!);
    pass.dispatchWorkgroupsIndirect(this.indirect, 60);
    run("finalizeAirSupportMetadata");
    pass.setPipeline(this.pipelines.commitAirSupportDirectRows!);
    pass.setBindGroup(0, groups.commitAirSupportDirectRows!);
    pass.dispatchWorkgroupsIndirect(this.indirect, 60);
    run("commitAirSupportPublication");
  }

  get source(): OctreeAirVelocitySupportGPUSource {
    return { plan: this.plan, arena: this.arena, recordArena: this.recordArena,
      selectorTagOffsetWords: this.plan.support.selectorTagOffsetWords,
      regularTagOffsetWords: this.plan.support.regularTagOffsetWords,
      controlOffsetWords: this.plan.support.controlOffsetWords,
      supportVectorOffsetWords: this.plan.support.supportVectorOffsetWords,
      recordOffsetWords: this.plan.records.recordOffsetWords,
      recordVectorOffsetWords: this.plan.records.vectorOffsetWords,
      canonicalRowVelocities: this.inputs.structured.rowVelocities };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.ownsArena) this.arena.destroy();
    for (const buffer of [this.recordArena, this.scratch, this.faceA, this.faceB, this.incidentFaces, this.faceAdjacency, this.faceFrontier,
      this.directAirVectors, this.indirect, this.ownerParams, ...this.params]) buffer.destroy();
  }
}

export const octreeAirVelocitySupportPublicationWGSL = /* wgsl */ `
struct P {
  rowCapacity:u32,slotCapacity:u32,domainVolume:u32,candidateStride:u32,
  candidateCapacity:u32,blockCapacity:u32,supportCapacity:u32,expectedEpoch:u32,
  dimensions:vec3u,maxLeaf:u32,
  selectorTagOffset:u32,regularTagOffset:u32,airControlOffset:u32,supportVectorOffset:u32,
  recordOffset:u32,recordVectorOffset:u32,boundaryEpochOffset:u32,tetraVertexCount:u32,
  scratchControl:u32,candidateOffset:u32,rankOffset:u32,directoryWinnerOffset:u32,
  directoryFlagOffset:u32,blockCountOffset:u32,blockOffsetOffset:u32,
  recordArenaWords:u32,supportArenaWords:u32,
  faceCellCapacity:u32,faceCapacity:u32,lookupCount:u32,catalogEntryCount:u32,
  reconstructionOffset:u32,templateHeaderOffset:u32,
  authorityBankStride:u32,maxSlots:u32,valuesOffset:u32,rowHandleOffset:u32,
  rowSignOffset:u32,rowCatalogOffset:u32,
  capturePreceding:u32,faceAdjacencyStride:u32,
  ownerDirectoryOffset:u32,
  fineCandidateOffset:u32,finePageCapacity:u32,fineFactor:u32,transportBandFineCells:u32,
  fineBrickDims:vec3u,fineR:u32,fineSampleDims:vec3u,fineSamplesPerBrick:u32,
  maxDisplacementFineCells:u32,expectedFineGeneration:u32,fineWidth:f32,closedBoundaryMask:u32,
  airGravityDt:vec3f,reuseTopology:u32,
}
struct Accepted {flags:atomic<u32>,firstError:atomic<u32>,rowCount:u32,epoch:u32,bank:u32,slotCount:u32}
// No demand-flags word: every consumer reads the deduplicated per-cell
// directoryFlags entry instead, so a per-candidate copy was write-only.
struct Candidate {cell:u32,size:u32,tagWord:u32}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read_write>accepted:Accepted;
@group(0)@binding(2)var<storage,read>rowGeometry:array<vec4u>;
@group(0)@binding(3)var<storage,read>ownerPageArena:array<u32>;
@group(0)@binding(4)var<storage,read>tetraHeaders:array<u32>;
@group(0)@binding(5)var<storage,read>tetrahedra:array<u32>;
@group(0)@binding(6)var<storage,read>tetraVertices:array<vec4f>;
@group(0)@binding(7)var<storage,read_write>scratch:array<atomic<u32>>;
@group(0)@binding(8)var<storage,read_write>recordArena:array<atomic<u32>>;
@group(0)@binding(9)var<storage,read_write>supportArena:array<atomic<u32>>;
@group(0)@binding(10)var<storage,read>boundaryEpoch:array<u32>;
@group(0)@binding(11)var<uniform>ownerPageLookupParams:OctreeOwnerPageLookupParams;
@group(0)@binding(12)var<storage,read>sameOrFinerDirect:array<u32>;
@group(0)@binding(13)var<storage,read>sameOrCoarserDirect:array<u32>;
@group(0)@binding(14)var<storage,read>catalogLookup:array<u32>;
struct CatalogSlotGeometry {neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f}
@group(0)@binding(15)var<storage,read>catalogFaces:array<CatalogSlotGeometry>;
@group(0)@binding(16)var<storage,read>denseCatalog:array<u32>;
@group(0)@binding(17)var<storage,read_write>rowVelocities:array<vec4f>;
@group(0)@binding(18)var<storage,read>liquidMask:array<u32>;
@group(0)@binding(19)var<storage,read_write>faceA:array<vec4u>;
@group(0)@binding(20)var<storage,read_write>faceB:array<vec4u>;
@group(0)@binding(21)var<storage,read>structuredAuthority:array<u32>;
@group(0)@binding(22)var<storage,read_write>directAirVectors:array<vec4f>;
@group(0)@binding(23)var<storage,read_write>faceAdjacency:array<u32>;
// This binding exposes only the aligned vector suffix of the consumer arena.
// Each reconstruction invocation owns one element, so its production write is
// a plain vec4 store rather than four atomics to non-conflicting addresses.
@group(0)@binding(24)var<storage,read_write>supportVectors:array<vec4f>;
@group(0)@binding(25)var<storage,read>fineMetadata:array<u32>;
@group(0)@binding(26)var<storage,read>fineWorklist:array<u32>;
@group(0)@binding(27)var<storage,read>fineFlags:array<u32>;
@group(0)@binding(28)var<storage,read>finePhi:array<f32>;
// GPU-authored sparse propagation queues. Layout: 16 control words, two
// face-capacity queue banks, then one generation mark per face slot.
@group(0)@binding(29)var<storage,read_write>faceFrontier:array<atomic<u32>>;
@group(0)@binding(30)var<storage,read_write>incidentFaces:array<vec4u>;
${octreeOwnerPageLookupWgsl}
const INVALID:u32=${OCTREE_AIR_SUPPORT_INVALID}u;
const SUPPORT_TAG:u32=${OCTREE_AIR_SUPPORT_TAG}u;
const OWNER_READY:u32=${OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready}u;
const OWNER_OVERFLOW:u32=${OCTREE_OWNER_PAGE_PUBLICATION_STATUS.overflow}u;
const ERROR_SOURCE:u32=${OCTREE_AIR_SUPPORT_GPU_ERROR.source}u;
const ERROR_GENERATION:u32=${OCTREE_AIR_SUPPORT_GPU_ERROR.generation}u;
const ERROR_CAPACITY:u32=${OCTREE_AIR_SUPPORT_GPU_ERROR.capacity}u;
const ERROR_TOPOLOGY:u32=${OCTREE_AIR_SUPPORT_GPU_ERROR.topology}u;
const ERROR_CATALOG:u32=${OCTREE_AIR_SUPPORT_GPU_ERROR.catalog}u;
const ERROR_TAG:u32=${OCTREE_AIR_SUPPORT_GPU_ERROR.tag}u;
// Retained in an otherwise unused frontier-control word across publications.
// A VALID support receipt plus this marker proves both the seed list and the
// reciprocal adjacency graph came from a completed sparse construction.
const RETAINED_GRAPH_VALID:u32=0x53524631u;
const RECORD_INTERFACE:u32=${STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.interfaceSource}u;
const RECORD_SELECTOR:u32=${STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.transitionSelector}u;
const RECORD_REGULAR:u32=${STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.regularInterpolationStencil}u;
const RECORD_EXTENSION:u32=${STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.extensionClosure}u;
const RECORD_FINE:u32=${STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.fineBandDemand}u;
// Transient scratch-only query marker. It is never copied into records and
// prevents same-dispatch VALUE_ONLY writes from recursively expanding.
const QUERY_FINE:u32=0x40000000u;
const DIRECTIONS:array<vec3i,18>=array<vec3i,18>(
  vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0),
  vec3i(-1,-1,0),vec3i(-1,0,-1),vec3i(-1,0,1),vec3i(-1,1,0),vec3i(0,-1,-1),vec3i(0,-1,1),
  vec3i(0,1,-1),vec3i(0,1,1),vec3i(1,-1,0),vec3i(1,0,-1),vec3i(1,0,1),vec3i(1,1,0));
fn s(index:u32)->u32{return atomicLoad(&scratch[index]);}
fn sw(index:u32,value:u32){atomicStore(&scratch[index],value);}
fn r(index:u32)->u32{return atomicLoad(&recordArena[index]);}
fn fail(item:u32,flag:u32){atomicOr(&scratch[0],flag);atomicMin(&scratch[1],item);}
fn failTopology(stage:u32,item:u32){atomicOr(&scratch[0],ERROR_TOPOLOGY);
  atomicMin(&scratch[1],(stage<<24u)|(item&0x00ffffffu));}
fn linearItem(wid:vec3u,lane:u32,workgroups:vec3u,size:u32)->u32{
  return (wid.x+wid.y*workgroups.x)*size+lane;
}
fn dispatchFor(count:u32,size:u32)->vec3u{let groups=(count+size-1u)/size;let x=min(groups,65535u);
  return vec3u(x,select(1u,(groups+x-1u)/x,x>0u),1u);}
fn writeDispatch(at:u32,value:vec3u){sw(at,value.x);sw(at+1u,value.y);sw(at+2u,value.z);}
fn coord(cell:u32)->vec3u{return vec3u(cell%p.dimensions.x,
  (cell/p.dimensions.x)%p.dimensions.y,cell/(p.dimensions.x*p.dimensions.y));}
fn cellOf(q:vec3u)->u32{return q.x+p.dimensions.x*(q.y+p.dimensions.y*q.z);}
fn inverseTransform(value:vec3f,code:u32)->vec3f{let bits=code&7u;let q=value*vec3f(
  select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));
  let permutation=(code/8u)%6u;if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}
  if(permutation==2u){return q.yxz;}if(permutation==3u){return q.zxy;}
  if(permutation==4u){return q.yzx;}return q.zyx;}
fn powerTransformVector(value:vec3i,code:u32)->vec3i{let signs=vec3i(select(1,-1,(code&1u)!=0u),
  select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));let permutation=(code/8u)%6u;var q=value;
  if(permutation==1u){q=value.xzy;}else if(permutation==2u){q=value.yxz;}else if(permutation==3u){q=value.yzx;}
  else if(permutation==4u){q=value.zxy;}else if(permutation==5u){q=value.zyx;}return q*signs;}
fn level(size:u32)->u32{return 31u-countLeadingZeros(size);}
// Dense cell -> published-row slot, authored once per publication immediately
// after the directory clear. It replaces a 12-iteration ordered binary search
// over rowGeometry that this producer performed ~1.5M times per encode (every
// demand, every fine-band closure, every face-adjacency identity). Because it
// still verifies the complete (cell,size) identity against rowGeometry it
// returns exactly the row the search returned; the only way the two could
// disagree is two rows claiming one origin cell, which is not an octree and is
// failed closed where the slot is authored.
fn rowIndexAt(cell:u32)->u32{return p.directoryFlagOffset+p.domainVolume+cell;}
fn fineDemandSourceAt(cell:u32)->u32{return arrayLength(&scratch)-2u*p.domainVolume+cell;}
fn fineDemandTemporaryAt(cell:u32)->u32{return arrayLength(&scratch)-p.domainVolume+cell;}
fn publishedRow(cell:u32,size:u32)->u32{if(cell>=p.domainVolume){return INVALID;}
  let row=s(rowIndexAt(cell));if(row>=s(2u)){return INVALID;}
  let g=rowGeometry[s(4u)*p.rowCapacity+row];
  return select(INVALID,row,g.x==cell&&g.y==size);}
// Base of the domain-sized fine-band candidate block. It follows the LAST
// accepted row's candidates rather than the last provisionable row's, so the
// live candidate index space [0, rows*stride + domainVolume) is contiguous and
// the sweeps below never walk the dead tail. Relocating the block cannot
// reorder anything: every row candidate index stays < rows*candidateStride and
// therefore still precedes every fine candidate, exactly as when the block sat
// at rowCapacity*candidateStride.
// p.fineCandidateOffset (rowCapacity*candidateStride) remains the fail-closed
// ceiling: rows never exceeds rowCapacity, so the min only ever fires on a
// corrupt accepted row count.
fn fineCandidateBase()->u32{return min(s(2u)*p.candidateStride,p.fineCandidateOffset);}
fn candidateBoundFor(rows:u32)->u32{return min(rows*p.candidateStride+p.domainVolume,p.candidateCapacity);}
fn candidateAt(item:u32)->Candidate{let at=p.candidateOffset+${OCTREE_AIR_SUPPORT_GPU_CANDIDATE_WORDS}u*item;
  return Candidate(s(at),s(at+1u),s(at+2u));}
fn setCandidate(item:u32,value:Candidate){let at=p.candidateOffset+${OCTREE_AIR_SUPPORT_GPU_CANDIDATE_WORDS}u*item;
  sw(at,value.cell);sw(at+1u,value.size);sw(at+2u,value.tagWord);}
fn compactFineDemandActive()->bool{return s(47u)!=0u&&(p.capturePreceding&4u)!=0u;}
// On a reused topology the fine candidate block is dead. Reuse it as an exact
// list of cells carrying this generation's dynamic RECORD_FINE bit. Query
// cells are appended by the mark dispatch first; value-only closure cells are
// appended by the following closure dispatch. atomicOr is the uniqueness
// authority, so every cell appears exactly once despite overlapping bricks.
fn publishFineDemand(cell:u32,bits:u32){
  let old=atomicOr(&scratch[p.directoryFlagOffset+cell],bits);
  if(compactFineDemandActive()&&(old&RECORD_FINE)==0u){
    let rank=atomicAdd(&scratch[30u],1u);
    if(rank>=p.domainVolume){fail(cell,ERROR_CAPACITY);return;}
    setCandidate(fineCandidateBase()+rank,Candidate(cell,0u,INVALID));
    if((bits&QUERY_FINE)!=0u){atomicAdd(&scratch[28u],1u);}
  }
}
fn recordAt(index:u32)->vec4u{let at=p.recordOffset+index*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;
  return vec4u(r(at),r(at+1u),r(at+2u),r(at+3u));}
fn recordCell(index:u32)->u32{return cellOf(recordAt(index).xyz);}
fn tagForIdentity(cell:u32,size:u32)->u32{let direct=publishedRow(cell,size);if(direct!=INVALID){return direct;}
  let tag=s(p.directoryWinnerOffset+cell);if((tag&SUPPORT_TAG)==0u){return INVALID;}let index=tag&0x7fffffffu;
  if(index>=s(8u)){return INVALID;}let identity=recordAt(index);return select(INVALID,tag,cellOf(identity.xyz)==cell&&identity.w==size);}
fn faceRowForTag(tag:u32)->u32{if(tag==INVALID){return INVALID;}return select(tag,s(2u)+(tag&0x7fffffffu),(tag&SUPPORT_TAG)!=0u);}
fn faceRowForIdentity(identity:vec2u)->u32{if(identity.x==INVALID||identity.y==INVALID){return INVALID;}
  return faceRowForTag(tagForIdentity(identity.x,identity.y));}
fn demand(row:u32,cell:i32,size:u32,flags:u32,tagWord:u32,item:u32){
  if(cell<0||u32(cell)>=p.domainVolume){atomicStore(&supportArena[tagWord],INVALID);return;}
  let origin=coord(u32(cell));let owner=octreeOwnerPageLookup(vec3i(origin));
  if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){
    failTopology(1u,item);atomicStore(&supportArena[tagWord],INVALID);return;}
  let resolvedCell=cellOf(owner.origin);let resolvedSize=owner.size;
  // Every cube or tetrahedron vertex is an exact octree-cell identity. A
  // containing coarser owner is not the requested vertex and cannot be
  // substituted without changing the paper's interpolant.
  if(resolvedCell!=u32(cell)||resolvedSize!=size){
    failTopology(1u,item);atomicStore(&supportArena[tagWord],INVALID);return;}
  // Mark demanded identities even when the accepted direct-row table already
  // owns them. Demand flags gate the interpolation closures; the march
  // destination itself is the whole accepted air partition, enrolled by
  // emitFineBandAirSupportCandidates so the extension domain stays contiguous.
  atomicOr(&scratch[p.directoryFlagOffset+resolvedCell],flags);
  let direct=publishedRow(resolvedCell,resolvedSize);if(direct!=INVALID){atomicStore(&supportArena[tagWord],direct);return;}
  setCandidate(item,Candidate(resolvedCell,resolvedSize,tagWord));
  atomicMin(&scratch[p.directoryWinnerOffset+resolvedCell],item);
}

@compute @workgroup_size(1)fn beginAirSupportPublication(){
  let precedingFlags=atomicLoad(&supportArena[p.airControlOffset]);
  let storedDetail=atomicLoad(&recordArena[14u]);
  let precedingDetail=select(0u,storedDetail,(storedDetail&0x80000000u)!=0u);
  if((p.capturePreceding&1u)!=0u&&precedingFlags!=0u&&s(38u)==0u
      &&atomicLoad(&supportArena[p.airControlOffset+14u])==${OCTREE_AIR_SUPPORT_LAYOUT_VERSION}u){
    sw(38u,precedingFlags);sw(39u,atomicLoad(&supportArena[p.airControlOffset+1u]));
  }
  // Preserve the preceding transaction's terminal control in reserved record
  // words. Failure-only QA can then distinguish a bad producer publication
  // from a later structured-stage rejection without adding a pass or readback
  // to the simulation path.
  if((p.capturePreceding&1u)!=0u&&atomicLoad(&recordArena[15u])==0u){
    atomicStore(&recordArena[13u],atomicLoad(&supportArena[p.airControlOffset]));
    atomicStore(&recordArena[14u],precedingDetail);
    atomicStore(&recordArena[15u],atomicLoad(&supportArena[p.airControlOffset+14u]));
  }
  if(atomicLoad(&accepted.flags)!=0u){
    if((precedingDetail&0x80000000u)==0u){
      atomicStore(&recordArena[13u],atomicLoad(&supportArena[p.airControlOffset]));
      atomicStore(&recordArena[14u],precedingDetail);}
    atomicStore(&recordArena[15u],atomicLoad(&supportArena[p.airControlOffset+14u]));
  }
  var boundaryNow=0u;if(p.boundaryEpochOffset<arrayLength(&boundaryEpoch)){boundaryNow=boundaryEpoch[p.boundaryEpochOffset];}
  let existingReady=atomicLoad(&supportArena[p.airControlOffset+13u])==${OCTREE_AIR_SUPPORT_VALID}u
    &&atomicLoad(&supportArena[p.airControlOffset+14u])==${OCTREE_AIR_SUPPORT_LAYOUT_VERSION}u
    &&atomicLoad(&supportArena[p.airControlOffset+2u])==accepted.epoch
    &&atomicLoad(&supportArena[p.airControlOffset+3u])==accepted.bank
    &&atomicLoad(&supportArena[p.airControlOffset+4u])==boundaryNow;
  let precedingSupportRows=atomicLoad(&supportArena[p.airControlOffset+6u]);
  // The complete support identity domain is a function of accepted topology,
  // bank and liquid-mask epoch. Fine demand changes RECORD_FINE flags but not
  // membership: emitFineBandAirSupportCandidates enrolls every accepted air
  // leaf. A VALID receipt for the exact accepted authority therefore proves
  // that record order, tags, owner directory and face adjacency are reusable.
  // Keep the decision on the GPU so a rejected/partial publication can never
  // make the host reuse stale topology.
  let reuseTopology=p.reuseTopology!=0u&&existingReady&&precedingSupportRows<=p.supportCapacity;
  sw(47u,select(0u,1u,reuseTopology));sw(48u,select(0u,precedingSupportRows,reuseTopology));
  let precedingSeeds=atomicLoad(&supportArena[p.airControlOffset+11u]);
  let retainedGraph=reuseTopology&&(p.capturePreceding&2u)!=0u
    &&atomicLoad(&faceFrontier[11u])==RETAINED_GRAPH_VALID
    &&precedingSeeds<=3u*p.candidateCapacity&&precedingSeeds<=p.faceCapacity;
  sw(49u,select(0u,precedingSeeds,retainedGraph));sw(50u,select(0u,1u,retainedGraph));
  // The candidate generation is only a request. If the accepted authority
  // itself is invalid, preserve a matching prior receipt. A rejected candidate
  // that leaves a clean older epoch is different: rebuild fine-band support
  // against that accepted epoch and the new fine generation, so Section 5 can
  // continue on one coherent (temporarily reused) power topology.
  if(atomicLoad(&accepted.flags)!=0u&&existingReady){
    sw(0u,ERROR_SOURCE|ERROR_GENERATION);sw(1u,0u);sw(31u,2u);
    writeDispatch(10u,vec3u(0u,1u,1u));writeDispatch(13u,vec3u(0u,1u,1u));
    writeDispatch(16u,vec3u(0u,1u,1u));writeDispatch(19u,vec3u(0u,1u,1u));
    writeDispatch(43u,vec3u(0u,1u,1u));return;
  }
  sw(0u,0u);sw(1u,INVALID);sw(31u,0u);let rows=min(accepted.rowCount,p.rowCapacity);sw(2u,rows);sw(3u,accepted.epoch);
  sw(4u,accepted.bank);var boundary=0u;if(p.boundaryEpochOffset<arrayLength(&boundaryEpoch)){boundary=boundaryEpoch[p.boundaryEpochOffset];}sw(5u,boundary);
  // The candidate arena is provisioned for rowCapacity rows, but only the
  // accepted rows ever emit into it, and the fine block is relocated to
  // sit immediately after them (see fineCandidateBase). Bounding the sweeps by
  // that live extent stops clearAirSupportCandidates, markAndScan, scatter and
  // resolveAirSupportTopology walking (rowCapacity-rows)*candidateStride dead
  // slots -- the arena is 3 words/slot, so the dead tail is what makes this pass
  // address-translation bound rather than compute bound.
  //
  // The emitted records are bit-identical to the capacity-wide sweep: row
  // candidates keep their indices, the fine block still follows every row
  // candidate, so the surviving candidates keep their relative order and
  // therefore their ranks, directory winners and support-record slots.
  let candidates=candidateBoundFor(rows);
  let blocks=(candidates+255u)/256u;sw(6u,candidates);sw(7u,blocks);sw(8u,0u);sw(9u,p.supportCapacity);
  if(reuseTopology){sw(8u,precedingSupportRows);}
  sw(25u,0u);
  sw(26u,select(0u,atomicLoad(&supportArena[p.airControlOffset+8u]),reuseTopology));
  sw(27u,select(0u,atomicLoad(&supportArena[p.airControlOffset+9u]),reuseTopology));
  sw(28u,0u);sw(30u,0u);sw(40u,p.expectedFineGeneration);
  sw(41u,0u);sw(42u,INVALID);
  if(atomicLoad(&accepted.flags)!=0u||accepted.epoch==0u||accepted.bank>1u
      ||accepted.rowCount==0u||accepted.rowCount>p.rowCapacity||accepted.slotCount>p.slotCapacity){fail(0u,ERROR_SOURCE|ERROR_GENERATION);}
  let ownerStatus=ownerPageArena[${OCTREE_OWNER_PAGE_CONTROL_WORDS.status}u];
  if(ownerPageArena[${OCTREE_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration}u]!=accepted.epoch
      ||(ownerStatus&OWNER_READY)==0u||(ownerStatus&OWNER_OVERFLOW)!=0u
      ||ownerPageArena[${OCTREE_OWNER_PAGE_CONTROL_WORDS.invalidEntryCount}u]!=0u||boundary!=accepted.epoch){fail(0u,ERROR_GENERATION);}
  if(p.recordOffset+p.supportCapacity*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u>arrayLength(&recordArena)
      ||p.recordVectorOffset+p.supportCapacity*${STRUCTURED_AIR_SUPPORT_VECTOR_WORDS}u>arrayLength(&recordArena)
      ||p.airControlOffset+${OCTREE_AIR_SUPPORT_CONTROL_WORDS}u>arrayLength(&supportArena)){fail(0u,ERROR_CAPACITY);}
  atomicStore(&recordArena[5u],0u);atomicStore(&recordArena[3u],0u);
  atomicStore(&supportArena[p.airControlOffset],ERROR_SOURCE);atomicStore(&supportArena[p.airControlOffset+1u],0u);
  atomicStore(&supportArena[p.airControlOffset+13u],0u);
  let clean=s(0u)==0u;
  writeDispatch(10u,select(vec3u(0u,1u,1u),dispatchFor(select(3u*p.domainVolume,p.domainVolume,reuseTopology),256u),clean));
  writeDispatch(13u,select(vec3u(0u,1u,1u),dispatchFor(rows*(${OCTREE_AIR_SUPPORT_SELECTOR_STRIDE}u+${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u),256u),clean&&!reuseTopology));
  // One workgroup owns one structured row. Its lanes cooperatively prove the
  // 27-site cube closure once before emitting either cube or tetra candidates.
  writeDispatch(16u,select(vec3u(0u,1u,1u),dispatchFor(rows,1u),clean&&!reuseTopology));
  writeDispatch(19u,select(vec3u(0u,1u,1u),dispatchFor(candidates,256u),clean&&!reuseTopology));
  // Words 43-45 are identity-phase scratch until prepareAirSupportFaces
  // resets them. Publish the exact support-row schedule there so topology
  // refresh never launches over provisioned support capacity.
  let topologyWork=select(p.supportCapacity,s(8u),reuseTopology);
  writeDispatch(43u,select(vec3u(0u,1u,1u),dispatchFor(topologyWork,256u),clean));
}

@compute @workgroup_size(256)fn clearAirSupportDirectory(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  if(s(47u)!=0u){if(item<p.domainVolume){let at=p.directoryFlagOffset+item;
      sw(at,s(at)&~(QUERY_FINE|RECORD_FINE));sw(fineDemandSourceAt(item),0u);
      sw(fineDemandTemporaryAt(item),0u);}return;}
  if(item<p.domainVolume){sw(p.directoryWinnerOffset+item,INVALID);sw(fineDemandSourceAt(item),0u);
    sw(fineDemandTemporaryAt(item),0u);}else if(item<2u*p.domainVolume){sw(p.directoryFlagOffset+item-p.domainVolume,0u);}
  else if(item<3u*p.domainVolume){sw(rowIndexAt(item-2u*p.domainVolume),INVALID);}}
// The dense row index is authored here, in the dispatch after the clear above,
// so the atomicMin below never races that clear. One accepted row owns one
// origin cell; a second claimant is a topology fault, not a tie to resolve.
@compute @workgroup_size(256)fn clearAirSupportCandidates(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  if(item<s(6u)){setCandidate(item,Candidate(INVALID,0u,INVALID));}
  if(item<s(2u)){let g=rowGeometry[s(4u)*p.rowCapacity+item];
    if(g.x>=p.domainVolume){fail(item,ERROR_SOURCE);}
    else if(atomicMin(&scratch[rowIndexAt(g.x)],item)!=INVALID){fail(item,ERROR_TOPOLOGY);}}}

@compute @workgroup_size(256)fn clearAirSupportTags(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  let selectors=s(2u)*${OCTREE_AIR_SUPPORT_SELECTOR_STRIDE}u;let regular=s(2u)*${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u;
  if(item<selectors){atomicStore(&supportArena[p.selectorTagOffset+item],INVALID);}
  else if(item<selectors+regular){atomicStore(&supportArena[p.regularTagOffset+item-selectors],INVALID);}}

fn unpackFineBrick(key:u32)->vec3u{let xy=p.fineBrickDims.x*p.fineBrickDims.y;let z=key/xy;let rem=key-z*xy;
  let y=rem/p.fineBrickDims.x;return vec3u(rem-y*p.fineBrickDims.x,y,z);}
fn fineLocal(local:u32)->vec3u{let z=local/(p.fineR*p.fineR);let rem=local-z*p.fineR*p.fineR;
  let y=rem/p.fineR;return vec3u(rem-y*p.fineR,y,z);}
// The fine worklist's resident dispatch is sample-kernel shaped
// (ceil(livePages/64)). Section 5 deliberately assigns one whole workgroup to
// each brick so its 64 lanes can reduce that brick's samples. Publish the exact
// page-shaped schedule into the otherwise-dead fifth indirect record during
// the existing initialization pass; no CPU count, capacity launch, or extra
// pass boundary is required.
@compute @workgroup_size(1)fn prepareFineBandAirSupportDemand(){
  let valid=p.fineFactor>0u&&p.transportBandFineCells>0u&&p.expectedFineGeneration>0u
    &&arrayLength(&fineWorklist)>=7u+p.finePageCapacity
    &&fineWorklist[0]==p.expectedFineGeneration&&fineWorklist[2]==p.finePageCapacity
    &&(fineWorklist[3]&3u)==3u&&fineWorklist[5]==1u&&fineWorklist[6]==1u
    &&fineWorklist[1]<=p.finePageCapacity;
  if(!valid){fail(0u,ERROR_SOURCE|ERROR_GENERATION);writeDispatch(22u,vec3u(0u,1u,1u));return;}
  writeDispatch(22u,select(vec3u(0u,1u,1u),dispatchFor(fineWorklist[1],1u),s(0u)==0u));
}
fn markFineBandDemandNeighborhood(base:vec3u){
  let radius=(p.maxDisplacementFineCells+p.fineFactor-1u)/p.fineFactor;
  for(var dz=-i32(radius);dz<=i32(radius);dz+=1){for(var dy=-i32(radius);dy<=i32(radius);dy+=1){for(var dx=-i32(radius);dx<=i32(radius);dx+=1){
    let demandCell=vec3i(base)+vec3i(dx,dy,dz);if(all(demandCell>=vec3i(0))&&all(demandCell<vec3i(p.dimensions))){
      publishFineDemand(cellOf(vec3u(demandCell)),QUERY_FINE|RECORD_FINE|RECORD_EXTENSION);}}}}}
// The resident brick page this workgroup owns (INVALID when the workgroup is
// retired), the minimum base owner cell any of its in-band samples demanded,
// and whether the brick ever spanned more than one such cell.
var<workgroup> markFineBrickPage:atomic<u32>;
var<workgroup> markFineBaseCell:atomic<u32>;
var<workgroup> markFineBaseSplit:atomic<u32>;
@compute @workgroup_size(64)fn markFineBandAirSupportDemand(@builtin(workgroup_id)wid:vec3u,
  @builtin(num_workgroups)groups:vec3u,@builtin(local_invocation_index)lane:u32){
  // Header/residency admission moved into lane 0 and behind workgroup state:
  // the reduction below needs workgroup barriers, and a barrier may not sit
  // downstream of a storage-derived early return.
  if(lane==0u){
    atomicStore(&markFineBrickPage,INVALID);atomicStore(&markFineBaseCell,INVALID);
    atomicStore(&markFineBaseSplit,0u);
    let headerValid=p.fineFactor>0u&&p.transportBandFineCells>0u&&p.expectedFineGeneration>0u
      &&arrayLength(&fineWorklist)>=7u+p.finePageCapacity&&fineWorklist[0]==p.expectedFineGeneration
      &&fineWorklist[2]==p.finePageCapacity&&(fineWorklist[3]&3u)==3u&&fineWorklist[5]==1u
      &&fineWorklist[6]==1u&&fineWorklist[1]<=p.finePageCapacity;
    if(!headerValid){fail(0u,ERROR_SOURCE|ERROR_GENERATION);}
    else{let work=wid.x+wid.y*groups.x;let live=fineWorklist[1];
      if(work<live){let id=fineWorklist[7u+work];
        if(id>=p.finePageCapacity||id*10u+2u>=arrayLength(&fineMetadata)
            ||fineMetadata[id*10u]!=id||fineMetadata[id*10u+2u]!=p.expectedFineGeneration){
          fail(work,ERROR_SOURCE|ERROR_GENERATION);
        }else{atomicStore(&markFineBrickPage,id);}}}}
  workgroupBarrier();
  let id=workgroupUniformLoad(&markFineBrickPage);
  // The demanded cell set is the union of every in-band sample's
  // (2*radius+1)^3 owner-cell neighbourhood, and atomicOr of one constant bit
  // pattern is both commutative and idempotent -- so emitting a neighbourhood
  // once per DISTINCT base cell publishes exactly the same flags as emitting it
  // once per in-band sample.
  //
  // That is worth proving because a brick is brickResolution^3 samples over a
  // lattice refined by fineFactor >= brickResolution (the planner admits only
  // brickResolution 4 with fineFactor 4 or 8, and brick origins are multiples
  // of brickResolution), so q/fineFactor is CONSTANT across a brick: all 64
  // lanes were recomputing and re-issuing the SAME 27 atomics, contending on
  // the same 27 words. Uniformity is established here rather than assumed, and
  // a brick that ever spans two base cells falls back to the per-sample emit.
  let sampleCount=select(0u,p.fineSamplesPerBrick,id!=INVALID);
  var base=vec3u(0u);var inBand=false;
  for(var local=lane;local<sampleCount;local+=64u){let index=id*p.fineSamplesPerBrick+local;
    if(index>=arrayLength(&fineFlags)||index>=arrayLength(&finePhi)){fail(index,ERROR_CAPACITY);continue;}
    if((fineFlags[index]&1u)==0u){continue;}let value=finePhi[index];if(!finiteValue(value)){fail(index,ERROR_SOURCE);continue;}
    if(abs(value)>f32(p.transportBandFineCells)*p.fineWidth){continue;}
    let q=unpackFineBrick(fineMetadata[id*10u+1u])*p.fineR+fineLocal(local);
    if(any(q>=p.fineSampleDims)){fail(index,ERROR_SOURCE);continue;}
    let sampleBase=q/p.fineFactor;
    if(inBand&&any(sampleBase!=base)){atomicStore(&markFineBaseSplit,1u);}
    base=sampleBase;inBand=true;
    // At factor one every sample is already a distinct base cell. Publish a
    // stable source mask now; the dense closure below performs one
    // neighbourhood read per destination and only one final flag atomic.
    if(p.fineFactor==1u){atomicStore(&scratch[fineDemandSourceAt(cellOf(base))],1u);}}
  if(inBand){atomicMin(&markFineBaseCell,cellOf(base));}
  workgroupBarrier();
  let sharedCell=atomicLoad(&markFineBaseCell);
  if(inBand&&cellOf(base)!=sharedCell){atomicStore(&markFineBaseSplit,1u);}
  workgroupBarrier();
  if(p.fineFactor==1u){return;}
  if(atomicLoad(&markFineBaseSplit)!=0u){if(inBand){markFineBandDemandNeighborhood(base);}return;}
  if(sharedCell==INVALID){return;}
  // Uniform brick: spread its one neighbourhood across the workgroup's lanes.
  let radius=(p.maxDisplacementFineCells+p.fineFactor-1u)/p.fineFactor;
  let width=2u*radius+1u;let count=width*width*width;let origin=coord(sharedCell);
  for(var n=lane;n<count;n+=64u){
    let dx=i32(n%width)-i32(radius);let dy=i32((n/width)%width)-i32(radius);let dz=i32(n/(width*width))-i32(radius);
    let demandCell=vec3i(origin)+vec3i(dx,dy,dz);if(all(demandCell>=vec3i(0))&&all(demandCell<vec3i(p.dimensions))){
      publishFineDemand(cellOf(vec3u(demandCell)),QUERY_FINE|RECORD_FINE|RECORD_EXTENSION);}}}

fn markFineResolvedOwner(expectedCenter:vec3f,expectedSize:u32,item:u32){
  if(expectedSize==0u){fail(item,ERROR_CATALOG);return;}
  let owner=octreeOwnerPageLookup(vec3i(floor(expectedCenter)));
  if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){failTopology(2u,item);return;}
  let physical=p.fineWidth*f32(p.fineFactor);if(!finiteValue(physical)||physical<=0.){fail(item,ERROR_SOURCE);return;}
  let ownerCenter=vec3f(owner.origin)+.5*f32(owner.size);
  let tolerance=max(1e-5/physical,f32(expectedSize)*2e-5);
  if(owner.size!=expectedSize||any(abs(ownerCenter-expectedCenter)>vec3f(tolerance))){failTopology(2u,item);return;}
  publishFineDemand(cellOf(owner.origin),RECORD_FINE|RECORD_EXTENSION);
}

fn fineResolvedOwnerMatches(expectedCenter:vec3f,expectedSize:u32,item:u32)->bool{
  if(expectedSize==0u){fail(item,ERROR_CATALOG);return false;}
  let owner=octreeOwnerPageLookup(vec3i(floor(expectedCenter)));
  if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){failTopology(2u,item);return false;}
  let ownerCenter=vec3f(owner.origin)+.5*f32(owner.size);
  let tolerance=max(1e-5,f32(expectedSize)*2e-5);
  return owner.size==expectedSize&&all(abs(ownerCenter-expectedCenter)<=vec3f(tolerance));
}

fn markExactRegularNeighborhood(origin:vec3u,size:u32,item:u32)->bool{
  let center=vec3f(origin)+.5*f32(size);let half=.5*f32(size);
  var exact=true;
  for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){for(var dx=-1;dx<=1;dx+=1){
    let expectedCenter=clamp(center+vec3f(f32(dx),f32(dy),f32(dz))*f32(size),
      vec3f(half),vec3f(p.dimensions)-vec3f(half));
    let resolved=octreeOwnerPageLookup(vec3i(floor(expectedCenter)));
    if((resolved.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){failTopology(2u,item);exact=false;continue;}
    let resolvedCenter=vec3f(resolved.origin)+.5*f32(resolved.size);
    let tolerance=max(1e-5,f32(size)*2e-5);
    let matches=resolved.size==size&&all(abs(resolvedCenter-expectedCenter)<=vec3f(tolerance));
    exact=exact&&matches;
    if(matches){
      publishFineDemand(cellOf(resolved.origin),RECORD_FINE|RECORD_EXTENSION);
    }}}}
  return exact;
}

fn fineDemandAxisTap(q:vec3i,axis:u32,delta:i32)->vec3i{
 if(axis==0u){return q+vec3i(delta,0,0);}
 if(axis==1u){return q+vec3i(0,delta,0);}
 return q+vec3i(0,0,delta);
}
fn dilateFineDemandAxis(item:u32,axis:u32,sourceTemporary:bool){
 if(item>=p.domainVolume||p.fineFactor!=1u||s(0u)!=0u){return;}
 let q=vec3i(coord(item));
 let radius=(p.maxDisplacementFineCells+p.fineFactor-1u)/p.fineFactor;
 var demanded=false;
 for(var delta=-i32(radius);delta<=i32(radius)&&!demanded;delta+=1){
  let source=fineDemandAxisTap(q,axis,delta);
  if(all(source>=vec3i(0))&&all(source<vec3i(p.dimensions))){
   let cell=cellOf(vec3u(source));
   demanded=select(s(fineDemandSourceAt(cell)),s(fineDemandTemporaryAt(cell)),
     sourceTemporary)!=0u;
  }
 }
 let value=select(0u,1u,demanded);
 if(sourceTemporary){sw(fineDemandSourceAt(item),value);}
 else{sw(fineDemandTemporaryAt(item),value);}
}
@compute @workgroup_size(256)fn dilateFineBandAirSupportDemandX(
 @builtin(global_invocation_id)g:vec3u){dilateFineDemandAxis(g.x,0u,false);}
@compute @workgroup_size(256)fn dilateFineBandAirSupportDemandY(
 @builtin(global_invocation_id)g:vec3u){dilateFineDemandAxis(g.x,1u,true);}
@compute @workgroup_size(256)fn dilateFineBandAirSupportDemandZ(
 @builtin(global_invocation_id)g:vec3u){dilateFineDemandAxis(g.x,2u,false);}

// Section 5 samples the dual mesh: trilinear interpolation needs the 27-cell
// logical stencil, while transition interpolation needs every tetra selector
// of the locally resolved power case. Publish that exact one-hop closure once;
// recurring transport then performs only dense owner/tag gathers.
@compute @workgroup_size(256)fn closeFineBandAirSupportInterpolationDemand(@builtin(global_invocation_id)g:vec3u){
  let invocation=g.x;var item=invocation;
  if(p.fineFactor==1u){
    if(item>=p.domainVolume||s(0u)!=0u){return;}
    let demanded=s(fineDemandTemporaryAt(item))!=0u;
    if(demanded){publishFineDemand(item,QUERY_FINE|RECORD_FINE|RECORD_EXTENSION);}
  }else if(compactFineDemandActive()){
    if(invocation>=s(28u)){return;}item=candidateAt(fineCandidateBase()+invocation).cell;
  }
  if(item>=p.domainVolume||s(0u)!=0u||(s(p.directoryFlagOffset+item)&QUERY_FINE)==0u){return;}
  let owner=octreeOwnerPageLookup(vec3i(coord(item)));if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){failTopology(3u,item);return;}
  let originCell=cellOf(owner.origin);let direct=publishedRow(originCell,owner.size);var caseId=INVALID;var transform=0u;
  if(direct!=INVALID){let geometry=rowGeometry[s(4u)*p.rowCapacity+direct];caseId=geometry.z;transform=geometry.w&63u;
  }else{let descriptor=descriptorForIdentity(owner.origin,owner.size);if(descriptor.x==INVALID){failTopology(3u,item);return;}
    let resolved=resolveDescriptor(descriptor.x);caseId=resolved.x;transform=resolved.y;}
  if(caseId==INVALID||caseId>=p.catalogEntryCount||transform>=48u){fail(item,ERROR_CATALOG);return;}
  let center=vec3f(owner.origin)+.5*f32(owner.size);
  // A nominal case-zero descriptor does not see a body-diagonal coarse owner.
  // Preserve exact regular samples for the uniform octants, then publish the
  // retained case-zero Delaunay fan used by every nonuniform octant.
  if(caseId==0u&&markExactRegularNeighborhood(owner.origin,owner.size,item)){return;}
  if(caseId>0xffffffffu/3u){fail(item,ERROR_CATALOG);return;}let headerAt=3u*caseId;
  if(headerAt>arrayLength(&tetraHeaders)||arrayLength(&tetraHeaders)-headerAt<3u){fail(item,ERROR_CATALOG);return;}
  let first=tetraHeaders[headerAt];let count=tetraHeaders[headerAt+1u];
  if(first>arrayLength(&tetrahedra)||count>arrayLength(&tetrahedra)-first){fail(item,ERROR_CATALOG);return;}
  // Section 6.2 stores the local Delaunay tetrahedralization per topology case
  // precisely because it is scene independent. Everything below the selector
  // byte -- selectorSize, selectorCenter, the resolved owner page and the flag
  // word it is OR-ed with -- is a function of the selector alone; owner.size,
  // center and transform are loop invariant. The generated catalog puts 136.4
  // vertex occurrences but only 24.7 DISTINCT selectors in the mean case (5.51x,
  // 199,872 distinct over 1,102,236 occurrences across all 8,083 fanned
  // entries), so walking occurrences re-resolved every owner ~5.5 times.
  // Walk the distinct selector set instead. Both terminal operations --
  // fail(item,ERROR_CATALOG) and markFineResolvedOwner's atomicOr/failTopology
  // -- are idempotent and receive identical arguments on every occurrence of a
  // selector, so the published flag set and the first-error word are bit-for-bit
  // what the occurrence walk produced, independent of order.
  var seen=array<u32,8>(0u,0u,0u,0u,0u,0u,0u,0u);
  for(var tetra=0u;tetra<count;tetra+=1u){let packed=tetrahedra[first+tetra];for(var vertex=0u;vertex<3u;vertex+=1u){
    let selector=(packed>>(8u*vertex))&255u;let selectorBit=1u<<(selector&31u);
    if((seen[selector>>5u]&selectorBit)!=0u){continue;}
    seen[selector>>5u]|=selectorBit;
    if(selector>=p.tetraVertexCount||selector>=arrayLength(&tetraVertices)){fail(item,ERROR_CATALOG);continue;}
    let v=tetraVertices[selector];if(v.w<=0.||!finiteValue(v.x)||!finiteValue(v.y)||!finiteValue(v.z)||!finiteValue(v.w)){
      fail(item,ERROR_CATALOG);continue;}
    let sizef=f32(owner.size)*v.w;let selectorSize=u32(round(sizef));
    if(selectorSize==0u||abs(sizef-f32(selectorSize))>2e-4){fail(item,ERROR_CATALOG);continue;}
    let selectorCenter=center+f32(owner.size)*inverseTransform(v.xyz,transform);
    let originf=selectorCenter-vec3f(.5*sizef);let origin=vec3i(round(originf));
    if(any(abs(originf-vec3f(origin))>vec3f(2e-4))){fail(item,ERROR_CATALOG);continue;}
    if(any(origin<vec3i(0))||any(origin+vec3i(i32(selectorSize))>vec3i(p.dimensions))){continue;}
    markFineResolvedOwner(selectorCenter,selectorSize,item);}}}

@compute @workgroup_size(256)fn emitFineBandAirSupportCandidates(@builtin(global_invocation_id)g:vec3u){
  let invocation=g.x;var item=invocation;let reuse=s(47u)!=0u;
  if(compactFineDemandActive()){
    if(invocation>=s(30u)){return;}item=candidateAt(fineCandidateBase()+invocation).cell;
  }
  if(item>=p.domainVolume||s(0u)!=0u){return;}let output=fineCandidateBase()+item;
  let demanded=s(p.directoryFlagOffset+item);
  if(reuse&&demanded==0u){return;}
  if(!reuse){setCandidate(output,Candidate(INVALID,0u,INVALID));}
  // Paper Section 5 marches the closest-face extension over the octree's whole
  // air region, so its domain is contiguous by construction and every fine-band
  // face has a path to a seeded liquid face. A demanded-cells-only destination
  // broke that invariant: a thin splash film whose coarse rows have all gone
  // air demands only a 1-ring around its own fine band, which islands away
  // from the liquid and freezes at stationary air — the stuck ceiling/corner
  // fluid artifact. Every accepted air leaf therefore joins the march graph:
  // the leaf's origin cell emits its one candidate, while non-origin cells
  // only forward their demand flags to the owning leaf as before.
  let owner=octreeOwnerPageLookup(vec3i(coord(item)));
  if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){
    if((demanded&RECORD_FINE)!=0u){failTopology(4u,output);}return;}
  let resolvedCell=cellOf(owner.origin);
  let topologyMember=(owner.status&OWNER_PAGE_LOOKUP_TOPOLOGY)!=0u;
  if(demanded!=0u||(topologyMember&&resolvedCell==item)){
    atomicOr(&scratch[p.directoryFlagOffset+resolvedCell],demanded|RECORD_EXTENSION);}
  if(reuse){return;}
  let direct=publishedRow(resolvedCell,owner.size);
  if(direct!=INVALID){return;}
  if(demanded==0u&&!topologyMember){return;}
  if((demanded&RECORD_FINE)==0u&&resolvedCell!=item){return;}
  setCandidate(output,Candidate(resolvedCell,owner.size,INVALID));
  atomicMin(&scratch[p.directoryWinnerOffset+resolvedCell],output);}

var<workgroup> emitRowActive:atomic<u32>;
var<workgroup> emitRowRegular:atomic<u32>;
var<workgroup> emitRowDemand:atomic<u32>;
var<workgroup> emitRowGeometry:array<u32,4>;
var<workgroup> emitRowTetraHeader:array<u32,2>;
@compute @workgroup_size(256)fn emitAirSupportCandidates(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  let row=wid.x+wid.y*groups.x;let itemBase=row*p.candidateStride;
  // Transport demand is a disjunction over the row's own liquid bit and the
  // liquid bits of its 18 logical neighbours. Every probe is a pure read and
  // OR is associative and commutative, so spreading the 18 probes over 18
  // lanes yields the same predicate the serial short-circuit produced — it
  // only stops one lane from running ~18 dependent owner-page walks while the
  // other 255 wait at the barrier below.
  if(lane==0u){atomicStore(&emitRowDemand,0u);
    atomicStore(&emitRowActive,select(0u,1u,row<s(2u)&&s(0u)==0u));}
  workgroupBarrier();
  let inRange=workgroupUniformLoad(&emitRowActive)!=0u;
  if(inRange&&lane<18u){let anchor=rowGeometry[s(4u)*p.rowCapacity+row];
    var demanded=lane==0u&&liquidRow(row);
    if(!demanded){let identity=neighborIdentity(anchor,DIRECTIONS[lane]);
      if(identity.x!=INVALID){let other=publishedRow(identity.x,identity.y);
        demanded=other!=INVALID&&liquidRow(other);}}
    if(demanded){atomicStore(&emitRowDemand,1u);}}
  workgroupBarrier();
  let demandRow=workgroupUniformLoad(&emitRowDemand)!=0u;
  if(lane==0u){
    var enabled=inRange&&demandRow;var g=vec4u(0u);
    if(enabled){g=rowGeometry[s(4u)*p.rowCapacity+row];if(g.y==0u||g.x>=p.domainVolume){fail(itemBase,ERROR_SOURCE);enabled=false;}}
    emitRowGeometry[0]=g.x;emitRowGeometry[1]=g.y;emitRowGeometry[2]=g.z;emitRowGeometry[3]=g.w;
    // Regular-closure eligibility is geometric: any row whose boundary-clamped
    // 27-neighbourhood is uniform owns axis-normal cube faces, including rows
    // whose caseId is nonzero only because a domain wall enters the descriptor.
    // The paper's per-axis face interpolation covers exactly these regular
    // regions; keying the closure off caseId==0 disabled it for every
    // wall-touching row, which is where a dam break keeps its kinetic energy.
    atomicStore(&emitRowActive,select(0u,1u,enabled));atomicStore(&emitRowRegular,select(0u,1u,enabled));
  }
  workgroupBarrier();if(workgroupUniformLoad(&emitRowActive)==0u){return;}
  let g=vec4u(emitRowGeometry[0],emitRowGeometry[1],emitRowGeometry[2],emitRowGeometry[3]);
  if(lane<${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u){
    let dx=i32(lane%3u)-1;let dy=i32((lane/3u)%3u)-1;let dz=i32(lane/9u)-1;
    let center=vec3f(coord(g.x))+.5*f32(g.y);let half=.5*f32(g.y);
    let expectedCenter=clamp(center+vec3f(f32(dx),f32(dy),f32(dz))*f32(g.y),
      vec3f(half),vec3f(p.dimensions)-vec3f(half));
    if(!fineResolvedOwnerMatches(expectedCenter,g.y,itemBase+lane)){atomicStore(&emitRowRegular,0u);}
  }
  workgroupBarrier();let regular=workgroupUniformLoad(&emitRowRegular)!=0u;let q=vec3i(coord(g.x));
  // Transition consumers (fine transport, tet-fan sampling) still address the
  // selector tags of every nonzero-case row, so a regular wall row publishes
  // BOTH closures: cube tags for the staggered basis and its retained tet fan.
  let needsSelectors=!regular||g.z!=0u;
  if(lane==0u&&needsSelectors){var valid=g.z<=0xffffffffu/3u;let headerAt=select(0u,3u*g.z,valid);
    valid=valid&&headerAt<=arrayLength(&tetraHeaders)&&arrayLength(&tetraHeaders)-headerAt>=3u;
    var first=0u;var count=0u;if(valid){first=tetraHeaders[headerAt];count=tetraHeaders[headerAt+1u];
      valid=first<=arrayLength(&tetrahedra)&&count<=arrayLength(&tetrahedra)-first
        &&count<=${OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS / 3}u;}
    if(!valid&&!regular){fail(itemBase,ERROR_CATALOG);}emitRowTetraHeader[0]=first;emitRowTetraHeader[1]=count;
    atomicStore(&emitRowActive,select(0u,1u,valid));}
  workgroupBarrier();let transitionActive=workgroupUniformLoad(&emitRowActive)!=0u;
  if(regular&&lane<${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u){let local=lane;let dx=i32(local%3u)-1;let dy=i32((local/3u)%3u)-1;let dz=i32(local/9u)-1;
    let requestedOrigin=q+vec3i(dx,dy,dz)*i32(g.y);let inDomain=all(requestedOrigin>=vec3i(0))&&all(requestedOrigin+vec3i(i32(g.y))<=vec3i(p.dimensions));
    let tag=p.regularTagOffset+row*${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u+local;
    if(!inDomain){atomicStore(&supportArena[tag],INVALID);}else{let cell=cellOf(vec3u(requestedOrigin));atomicAdd(&scratch[27u],1u);
    demand(row,i32(cell),g.y,RECORD_REGULAR|RECORD_EXTENSION,tag,itemBase+local);}}
  if(!needsSelectors||!transitionActive){return;}
  let occurrence=lane;let first=emitRowTetraHeader[0];let count=emitRowTetraHeader[1];if(occurrence>=3u*count){return;}
  let item=itemBase+${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u+occurrence;
  let packed=tetrahedra[first+occurrence/3u];let selector=(packed>>(8u*(occurrence%3u)))&255u;
    if(selector>=p.tetraVertexCount||selector>=arrayLength(&tetraVertices)){fail(item,ERROR_CATALOG);return;}let v=tetraVertices[selector];
    if(!finiteValue(v.x)||!finiteValue(v.y)||!finiteValue(v.z)||!finiteValue(v.w)||v.w<=0.){fail(item,ERROR_CATALOG);return;}
    if(length(v.xyz)<1e-7&&!(all(v.xyz==vec3f(0.))&&v.w==1.)){fail(item,ERROR_CATALOG);return;}
    let sizef=f32(g.y)*v.w;let size=u32(round(sizef));
    let center=vec3f(q)+.5*f32(g.y);let originf=center+f32(g.y)*inverseTransform(v.xyz,g.w&63u)-.5*sizef;let origin=vec3i(round(originf));
    let tag=p.selectorTagOffset+row*${OCTREE_AIR_SUPPORT_SELECTOR_STRIDE}u+selector;atomicAdd(&scratch[26u],1u);
    if(size==0u||abs(sizef-f32(size))>2e-4||any(abs(originf-vec3f(origin))>vec3f(2e-4))){fail(item,ERROR_CATALOG);return;}
    if(any(origin<vec3i(0))||any(origin+vec3i(i32(size))>vec3i(p.dimensions))){atomicStore(&supportArena[tag],INVALID);return;}
    demand(row,i32(cellOf(vec3u(origin))),size,RECORD_SELECTOR|RECORD_EXTENSION,tag,item);}

var<workgroup> marks:array<u32,256>;
@compute @workgroup_size(256)fn markAndScanAirSupportCandidates(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let block=wid.x+wid.y*groups.x;let item=block*256u+lane;
  var mark=0u;if(item<s(6u)){let c=candidateAt(item);mark=select(0u,1u,c.cell!=INVALID&&s(p.directoryWinnerOffset+c.cell)==item);}
  marks[lane]=mark;workgroupBarrier();for(var offset=1u;offset<256u;offset<<=1u){var add=0u;if(lane>=offset){add=marks[lane-offset];}
    workgroupBarrier();marks[lane]+=add;workgroupBarrier();}if(item<s(6u)){sw(p.rankOffset+item,select(INVALID,marks[lane]-1u,mark!=0u));}
  if(lane==255u&&block<p.blockCapacity){sw(p.blockCountOffset+block,marks[255u]);}}

var<workgroup> blockScan:array<u32,256>;
@compute @workgroup_size(256)fn prefixAirSupportBlocks(@builtin(local_invocation_index)lane:u32){let reuse=s(47u)!=0u;
  // Barriers must remain in uniform control flow. A reuse dispatch therefore
  // scans zero blocks and suppresses only the terminal count store.
  let blocks=select(s(7u),0u,reuse);let chunk=(blocks+255u)/256u;
  let first=min(blocks,lane*chunk);let last=min(blocks,first+chunk);var total=0u;for(var block=first;block<last;block+=1u){total+=s(p.blockCountOffset+block);}blockScan[lane]=total;workgroupBarrier();
  for(var offset=1u;offset<256u;offset<<=1u){var add=0u;if(lane>=offset){add=blockScan[lane-offset];}workgroupBarrier();blockScan[lane]+=add;workgroupBarrier();}
  var cursor=select(0u,blockScan[lane-1u],lane>0u);for(var block=first;block<last;block+=1u){sw(p.blockOffsetOffset+block,cursor);cursor+=s(p.blockCountOffset+block);}
  if(lane==255u&&!reuse){let count=blockScan[255u];sw(8u,count);if(count>p.supportCapacity){fail(count,ERROR_CAPACITY);}}}

@compute @workgroup_size(256)fn scatterAirSupportRecords(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let block=wid.x+wid.y*groups.x;let item=block*256u+lane;
  if(item>=s(6u)||s(0u)!=0u){return;}let localRank=s(p.rankOffset+item);if(localRank==INVALID){return;}let output=s(p.blockOffsetOffset+block)+localRank;
  if(output>=p.supportCapacity){fail(item,ERROR_CAPACITY);return;}let c=candidateAt(item);let flags=s(p.directoryFlagOffset+c.cell)&0x3fffffffu;let q=coord(c.cell);let at=p.recordOffset+output*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;
  atomicStore(&recordArena[at],q.x);atomicStore(&recordArena[at+1u],q.y);atomicStore(&recordArena[at+2u],q.z);atomicStore(&recordArena[at+3u],c.size);
  // Air topology is not inferred from the wet demand source. INVALID remains
  // explicit until a topology descriptor for this exact identity is proven.
  atomicStore(&recordArena[at+4u],INVALID);atomicStore(&recordArena[at+5u],flags<<6u);
  atomicStore(&recordArena[at+6u],INVALID);atomicStore(&recordArena[at+7u],s(3u));
  sw(p.directoryWinnerOffset+c.cell,SUPPORT_TAG|output);}

@compute @workgroup_size(256)fn resolveAirSupportTags(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  // This entry point is dispatched over rows*256 items, which now overlaps the
  // relocated fine-band block. Fine candidates carry no tag word and the final
  // store below already skipped them, so retiring them here keeps the integrity
  // gates below applying to exactly the tagged row candidates they always did.
  if(item>=s(6u)||s(0u)!=0u){return;}let c=candidateAt(item);
  if(c.cell==INVALID||c.tagWord==INVALID){return;}let tag=s(p.directoryWinnerOffset+c.cell);
  if((tag&SUPPORT_TAG)==0u){fail(item,ERROR_TAG);return;}let support=tag&0x7fffffffu;if(support>=s(8u)){fail(item,ERROR_TAG);return;}
  let at=p.recordOffset+support*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;if(r(at)!=coord(c.cell).x||r(at+1u)!=coord(c.cell).y||r(at+2u)!=coord(c.cell).z||r(at+3u)!=c.size){failTopology(5u,item);return;}
  if(c.tagWord!=INVALID){atomicStore(&supportArena[c.tagWord],tag);}}

fn boundaryDirectionBit(direction:vec3i)->u32{if(direction.x<0){return 0u;}if(direction.y<0){return 1u;}
  if(direction.z<0){return 2u;}if(direction.z>0){return 3u;}if(direction.y>0){return 4u;}return 5u;}
fn transformBoundaryMask(mask:u32,transform:u32)->u32{let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(0,-1,0),
  vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0));var result=0u;for(var bit=0u;bit<6u;bit+=1u){
  if((mask&(1u<<bit))!=0u){result|=1u<<boundaryDirectionBit(powerTransformVector(directions[bit],transform));}}return result;}
fn resolveBoundaryEntry(interior:u32,mask:u32)->u32{let key=interior*64u+mask;var lo=0u;var hi=min(p.lookupCount,arrayLength(&catalogLookup)/3u);
  while(lo<hi){let mid=lo+(hi-lo)/2u;let candidate=catalogLookup[3u*mid];if(candidate<key){lo=mid+1u;}else{hi=mid;}}
  if(lo>=min(p.lookupCount,arrayLength(&catalogLookup)/3u)||catalogLookup[3u*lo]!=key){return INVALID;}return catalogLookup[3u*lo+1u];}
fn resolveDescriptor(descriptor:u32)->vec2u{let boundary=(descriptor>>24u)&63u;let geometry=descriptor&0xc0ffffffu;var packed=INVALID;
  if((geometry&0x80000000u)!=0u){let index=geometry&0x1ffu;if((geometry&0x40fffe00u)==0u&&index<arrayLength(&sameOrCoarserDirect)){packed=sameOrCoarserDirect[index];}}
  else{let index=geometry&0x3ffffu;if((geometry&0x40fc0000u)==0u&&index<arrayLength(&sameOrFinerDirect)){packed=sameOrFinerDirect[index];}}
  if(packed==INVALID){return vec2u(INVALID);}let transform=packed>>16u;var entry=packed&0xffffu;if(boundary!=0u){
    entry=resolveBoundaryEntry(entry,transformBoundaryMask(boundary,transform));}return vec2u(entry,transform);}
// Returns [descriptor, failure reason, failure detail]. A rejected record has
// no catalog case or extension layer, so resolveAirSupportTopology persists
// reason/detail in those two otherwise-unused words before failing closed.
// reason 1 = malformed owner, 2 = ratio beyond 2:1, 3 = mixed finer/coarser.
fn descriptorForIdentity(origin:vec3u,size:u32)->vec3u{var sizes:array<u32,18>;var finer=false;var coarser=false;
  var firstFiner=31u;var firstCoarser=31u;
  for(var bit=0u;bit<18u;bit+=1u){let direction=DIRECTIONS[bit];var probe=vec3i(0);
    for(var axis=0u;axis<3u;axis+=1u){probe[axis]=select(select(i32(origin[axis]+size/2u),i32(origin[axis]+size),direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}
    // Air rows use the interior Delaunay fan and apply authored wall/open
    // behavior while constructing ordinary faces below. Encoding a clipped
    // boundary case here asks the liquid power catalog for positive-air-only
    // transition combinations which it intentionally does not contain.
    if(any(probe<vec3i(0))||any(probe>=vec3i(p.dimensions))){sizes[bit]=size;continue;}
    let owner=octreeOwnerPageLookup(probe);if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){return vec3u(INVALID,1u,(bit&31u)|((owner.size&63u)<<8u)|((owner.status&0xffffu)<<16u));}
    if(owner.size*2u<size||owner.size>size*2u){return vec3u(INVALID,2u,(bit&31u)|((owner.size&63u)<<8u)|((size&63u)<<16u));}
    sizes[bit]=owner.size;if(owner.size<size){finer=true;firstFiner=min(firstFiner,bit);}if(owner.size>size){coarser=true;firstCoarser=min(firstCoarser,bit);}}
  if(finer&&coarser){return vec3u(INVALID,3u,(firstFiner&31u)|((firstCoarser&31u)<<5u));}var descriptor=0u;if(!coarser){for(var bit=0u;bit<18u;bit+=1u){if(sizes[bit]==size){descriptor|=1u<<bit;}}}
  else{let child=(origin/vec3u(size))&vec3u(1u);descriptor|=0x80000000u|child.x|(child.y<<1u)|(child.z<<2u);
    let outward=vec3i(select(-1,1,child.x==1u),select(-1,1,child.y==1u),select(-1,1,child.z==1u));
    let wanted=array<vec3i,6>(vec3i(outward.x,0,0),vec3i(0,outward.y,0),vec3i(0,0,outward.z),
      vec3i(outward.x,outward.y,0),vec3i(outward.x,0,outward.z),vec3i(0,outward.y,outward.z));
    for(var coarse=0u;coarse<6u;coarse+=1u){for(var bit=0u;bit<18u;bit+=1u){if(all(DIRECTIONS[bit]==wanted[coarse])&&sizes[bit]==size*2u){descriptor|=1u<<(coarse+3u);}}}}
  return vec3u(descriptor,0u,0u);}

@compute @workgroup_size(256)fn resolveAirSupportTopology(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  if(item>=s(8u)||s(0u)!=0u){return;}let identity=recordAt(item);
  if(s(47u)!=0u){let at=p.recordOffset+item*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;
    let stable=r(at+5u)&~(RECORD_FINE<<6u);let dynamic=s(p.directoryFlagOffset+recordCell(item))&RECORD_FINE;
    atomicStore(&recordArena[at+5u],stable|(dynamic<<6u));return;}
  let descriptor=descriptorForIdentity(identity.xyz,identity.w);
  if(descriptor.x==INVALID){let at=p.recordOffset+item*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;
    atomicStore(&recordArena[at+4u],descriptor.y);atomicStore(&recordArena[at+6u],descriptor.z);failTopology(6u,item);return;}let resolved=resolveDescriptor(descriptor.x);
  if(resolved.x==INVALID||resolved.x>=p.catalogEntryCount||resolved.y>=48u){fail(item,ERROR_CATALOG);return;}
  let at=p.recordOffset+item*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;atomicStore(&recordArena[at+4u],resolved.x);
  let old=r(at+5u);atomicStore(&recordArena[at+5u],resolved.y|(old&0xffffffc0u));}

// Fine transport consumes this identity-keyed view through the already-bound
// support arena. Each finest cell names its complete accepted octree owner;
// positive-air owners retain their support tag instead of acquiring a compact
// momentum row or falling back to a neighbouring wet row.
@compute @workgroup_size(256)fn publishAirSupportOwnerDirectory(@builtin(global_invocation_id)g:vec3u){
  let item=g.x;if(item>=p.domainVolume||s(0u)!=0u||s(47u)!=0u){return;}let output=p.ownerDirectoryOffset+4u*item;
  if(output+3u>=arrayLength(&supportArena)){fail(item,ERROR_CAPACITY);return;}
  let owner=octreeOwnerPageLookup(vec3i(coord(item)));var tag=INVALID;var originCell=INVALID;var size=0u;var packed=INVALID;
  if((owner.status&OWNER_PAGE_LOOKUP_INVALID)==0u){originCell=cellOf(owner.origin);size=owner.size;let direct=publishedRow(originCell,size);
    if(direct!=INVALID){let geometry=rowGeometry[s(4u)*p.rowCapacity+direct];tag=direct;packed=(geometry.z&0xffffu)|((geometry.w&63u)<<16u);
    }else{let candidate=s(p.directoryWinnerOffset+originCell);if((candidate&SUPPORT_TAG)!=0u){let support=candidate&0x7fffffffu;
      if(support<s(8u)){let identity=recordAt(support);let at=p.recordOffset+support*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;
        if(cellOf(identity.xyz)==originCell&&identity.w==size){let caseId=r(at+4u);let transform=r(at+5u)&63u;
          if(caseId!=INVALID){tag=candidate;packed=(caseId&0xffffu)|(transform<<16u);}}}}}}
  atomicStore(&supportArena[output],tag);atomicStore(&supportArena[output+1u],originCell);
  atomicStore(&supportArena[output+2u],size);atomicStore(&supportArena[output+3u],packed);}

// directRows/bank/supportRows are the dispatch-uniform scratch control words
// s(2u)/s(4u)/s(8u). They are authored before the face passes and never
// written during them, so a caller may read them once and pass them down; the
// resolved cell is identical either way. That matters because s() is an
// atomicLoad the compiler may not hoist, and the march's 30x4 candidate scan
// used to re-issue all three on every candidate of every lane of every sweep.
fn faceCellIn(faceRow:u32,directRows:u32,bank:u32,supportRows:u32)->vec4u{
  if(faceRow<directRows){return rowGeometry[bank*p.rowCapacity+faceRow];}
  let support=faceRow-directRows;if(support>=supportRows){return vec4u(INVALID);}let identity=recordAt(support);
  let at=p.recordOffset+support*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;return vec4u(cellOf(identity.xyz),identity.w,r(at+4u),r(at+5u));}
fn faceCell(faceRow:u32)->vec4u{return faceCellIn(faceRow,s(2u),s(4u),s(8u));}
fn liquidRow(row:u32)->bool{if(row>=s(2u)){return false;}let at=s(4u)*p.rowCapacity+row;
  if(at>=arrayLength(&liquidMask)){return false;}return liquidMask[at]!=0u;}
// The serial 18-direction transport-demand walk that used to live here is now
// evaluated by 18 lanes of the emitting row's own workgroup; see the comment in
// emitAirSupportCandidates for why the disjunction is order-free.
fn publishedLiquidRow(row:u32)->bool{if(row>=s(2u)){return false;}let cell=faceCell(row).x;
  return (s(p.directoryFlagOffset+cell)&0x80000000u)!=0u;}
fn publishedDirectLiquidRow(row:u32)->bool{if(row>=s(2u)){return false;}let geometry=s(4u)*p.rowCapacity+row;
  if(geometry>=arrayLength(&rowGeometry)){return false;}let cell=rowGeometry[geometry].x;
  return (s(p.directoryFlagOffset+cell)&0x80000000u)!=0u;}
fn publishedDirectDemandedRow(row:u32)->bool{if(row>=s(2u)){return false;}let geometry=s(4u)*p.rowCapacity+row;
  if(geometry>=arrayLength(&rowGeometry)){return false;}let cell=rowGeometry[geometry].x;
  return (s(p.directoryFlagOffset+cell)&RECORD_EXTENSION)!=0u;}
fn finiteValue(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn validVector(value:vec4f)->bool{return value.w>0.&&finiteValue(value.x)&&finiteValue(value.y)&&finiteValue(value.z);}
fn bitsf(index:u32)->f32{return bitcast<f32>(denseCatalog[index]);}
fn caseHeader(caseId:u32)->vec2u{let at=p.templateHeaderOffset+4u*caseId;
  if(caseId>=p.catalogEntryCount||at+1u>=arrayLength(&denseCatalog)){return vec2u(INVALID);}
  return vec2u(denseCatalog[at],denseCatalog[at+1u]);}
// Section 5 seed source: the exact accepted projected power-face normal sample
// whose power face is coplanar with the requested ordinary-face patch. The
// paper's extension copies face values; the seed therefore selects ONE face
// (nearest coplanar axis-normal face, lowest local slot on exact ties) and
// never averages, and never treats a reconstructed cell-centred vector as a
// face quantity. Returns (axis component, status): status 1 found, 0 no
// coplanar axis-normal face, -1 authority/catalog fault.
fn projectedAxisFaceValue(row:u32,axis:u32,patchCenter:vec3f)->vec2f{if(row>=s(2u)){return vec2f(0.,-1.);}
  let cell=faceCell(row);let header=caseHeader(cell.z);if(header.x==INVALID||header.y>p.maxSlots){return vec2f(0.,-1.);}
  let bank=s(4u)*p.authorityBankStride;let rowBase=row*p.maxSlots;let transform=cell.w&63u;
  let anchorCenter=vec3f(coord(cell.x))+.5*f32(cell.y);let tolerance=2e-4*f32(cell.y);
  var bestLocal=INVALID;var bestDistance=0.;var bestValue=0.;
  for(var local=0u;local<header.y;local+=1u){let localAt=rowBase+local;
    let handleAt=bank+p.rowHandleOffset+localAt;let signAt=bank+p.rowSignOffset+localAt;
    let catalogAt=bank+p.rowCatalogOffset+localAt;if(handleAt>=arrayLength(&structuredAuthority)
        ||signAt>=arrayLength(&structuredAuthority)||catalogAt>=arrayLength(&structuredAuthority)){return vec2f(0.,-1.);}
    let handle=structuredAuthority[handleAt];let global=structuredAuthority[catalogAt];
    let valueAt=bank+p.valuesOffset+handle;if(handle>=accepted.slotCount
        ||valueAt>=arrayLength(&structuredAuthority)||global>=arrayLength(&catalogFaces)){return vec2f(0.,-1.);}
    let slot=catalogFaces[global];let normal=normalize(inverseTransform(slot.normalInverseDistance.xyz,transform));
    // Positive comparisons so a NaN normal or centroid rejects the slot.
    let aligned=normal[axis];if(!(abs(aligned)>=0.999)){continue;}
    let centroid=anchorCenter+f32(cell.y)*inverseTransform(slot.areaCentroid.yzw,transform);
    if(!(abs(centroid[axis]-patchCenter[axis])<=tolerance)){continue;}
    let separation=distance(centroid,patchCenter);if(!finiteValue(separation)){continue;}
    // Strict less-than keeps the lowest local slot index on exact ties: the
    // same stable index discipline betterFace applies during the march.
    if(bestLocal==INVALID||separation<bestDistance){
      let sample=f32(bitcast<i32>(structuredAuthority[signAt]))*bitcast<f32>(structuredAuthority[valueAt]);
      bestLocal=local;bestDistance=separation;bestValue=select(sample,-sample,aligned<0.);}}
  if(bestLocal==INVALID){return vec2f(0.,0.);}
  if(!finiteValue(bestValue)){return vec2f(0.,-1.);}
  return vec2f(bestValue,1.);}
fn neighborIdentity(cell:vec4u,direction:vec3i)->vec2u{let origin=coord(cell.x);var probe=vec3i(0);for(var axis=0u;axis<3u;axis+=1u){
  probe[axis]=select(select(i32(origin[axis]+cell.y/2u),i32(origin[axis]+cell.y),direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}
  if(any(probe<vec3i(0))||any(probe>=vec3i(p.dimensions))){return vec2u(INVALID);}let owner=octreeOwnerPageLookup(probe);
  if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){return vec2u(INVALID);}return vec2u(cellOf(owner.origin),owner.size);}
fn signedFaceNeighborIdentity(cell:vec4u,axis:u32,quadrant:u32,positive:bool)->vec2u{let origin=coord(cell.x);var probe=vec3i(origin);
  var transverse=0u;for(var a=0u;a<3u;a+=1u){if(a==axis){probe[a]=select(i32(origin[a])-1,i32(origin[a]+cell.y),positive);continue;}
    let high=(quadrant&(1u<<transverse))!=0u;let offset=min(cell.y-1u,(cell.y*(select(1u,3u,high)))/4u);
    probe[a]=i32(origin[a]+offset);transverse+=1u;}if(any(probe<vec3i(0))||any(probe>=vec3i(p.dimensions))){return vec2u(INVALID);}
  let owner=octreeOwnerPageLookup(probe);if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){return vec2u(INVALID);}
  return vec2u(cellOf(owner.origin),owner.size);}
fn catalogNeighbor(cell:vec4u,global:u32)->vec2u{if(global>=arrayLength(&catalogFaces)){return vec2u(INVALID);}
  let slot=catalogFaces[global];let sizeF=f32(cell.y)*slot.neighborOffsetSize.w;if(sizeF<=0.){return vec2u(INVALID);}
  let size=u32(round(sizeF));let center=vec3f(coord(cell.x))+.5*f32(cell.y);
  let originF=center+f32(cell.y)*inverseTransform(slot.neighborOffsetSize.xyz,cell.w&63u)-.5*sizeF;
  let origin=vec3i(round(originF));if(size==0u||abs(sizeF-f32(size))>2e-4||any(abs(originF-vec3f(origin))>vec3f(2e-4))
      ||any(origin<vec3i(0))||any(origin+vec3i(i32(size))>vec3i(p.dimensions))){return vec2u(INVALID);}
  return vec2u(cellOf(vec3u(origin)),size);}

@compute @workgroup_size(1)fn prepareAirSupportFaces(){var clean=s(0u)==0u;let faceRows=s(2u)+s(8u);let count=faceRows*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  if(faceRows>p.faceCellCapacity||count>p.faceCapacity){fail(faceRows,ERROR_CAPACITY);clean=false;}sw(29u,select(0u,count,clean));sw(25u,0u);sw(28u,0u);sw(30u,0u);sw(32u,0u);sw(37u,0u);
  sw(43u,0u);sw(44u,0u);sw(45u,0u);sw(46u,0u);
  for(var word=0u;word<16u;word+=1u){atomicStore(&faceFrontier[word],0u);}
  writeDispatch(32u,select(vec3u(0u,1u,1u),dispatchFor(count,256u),clean));
  writeDispatch(35u,select(vec3u(0u,1u,1u),dispatchFor(faceRows,256u),clean));}

fn adjacencyBase(faceRow:u32)->u32{return faceRow*p.faceAdjacencyStride;}
fn adjacencyIncidentCount(faceRow:u32)->u32{return faceAdjacency[adjacencyBase(faceRow)];}
fn adjacencyIncident(faceRow:u32,local:u32)->u32{return faceAdjacency[adjacencyBase(faceRow)+1u+local];}
fn adjacencyNegative(faceRow:u32,axis:u32,quadrant:u32)->u32{
  return faceAdjacency[adjacencyBase(faceRow)+${1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u+4u*axis+quadrant];}
fn adjacencyPositive(faceRow:u32,axis:u32,quadrant:u32)->u32{
  return faceAdjacency[adjacencyBase(faceRow)+${1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence + STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant];}
// This face row's cell origin coordinate and extent, resolved once by
// resolveAirSupportFaceAdjacency into the tail of the same adjacency record the
// march already reads for its incidence list. See faceCenter.
fn adjacencyGeometryBase(faceRow:u32)->u32{
  return adjacencyBase(faceRow)+${1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence + 2 * STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;}
fn setAdjacencyGeometry(faceRow:u32,cell:vec4u){let at=adjacencyGeometryBase(faceRow);let q=coord(cell.x);
  faceAdjacency[at]=q.x;faceAdjacency[at+1u]=q.y;faceAdjacency[at+2u]=q.z;faceAdjacency[at+3u]=cell.y;}
fn adjacencyGeometry(faceRow:u32)->vec4u{let at=adjacencyGeometryBase(faceRow);
  return vec4u(faceAdjacency[at],faceAdjacency[at+1u],faceAdjacency[at+2u],faceAdjacency[at+3u]);}

// Resolve all topology identities once. The six extrapolation waves consume
// only this compact indexed graph; catalog and owner-page traversal is kept in
// the topology-publication stage as required by the sparse Section 5 design.
@compute @workgroup_size(256)fn resolveAirSupportFaceAdjacency(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let faceRow=linearItem(wid,lane,groups,256u);
  let faceRows=s(2u)+s(8u);if(faceRow>=faceRows||s(0u)!=0u||s(47u)!=0u){return;}let base=adjacencyBase(faceRow);
  if(base+p.faceAdjacencyStride>arrayLength(&faceAdjacency)){fail(faceRow,ERROR_CAPACITY);return;}
  for(var local=0u;local<p.faceAdjacencyStride;local+=1u){faceAdjacency[base+local]=INVALID;}
  let cell=faceCell(faceRow);if(cell.x==INVALID){failTopology(7u,faceRow);return;}setAdjacencyGeometry(faceRow,cell);
  let header=caseHeader(cell.z);
  if(header.x==INVALID||header.y>${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u
      ||header.x>arrayLength(&catalogFaces)||header.y>arrayLength(&catalogFaces)-header.x){fail(faceRow,ERROR_CATALOG);return;}
  var count=0u;for(var localFace=0u;localFace<header.y;localFace+=1u){let identity=catalogNeighbor(cell,header.x+localFace);
    if(identity.x==INVALID){continue;}let otherRow=faceRowForIdentity(identity);if(otherRow==INVALID){continue;}
    var duplicate=false;for(var prior=0u;prior<count;prior+=1u){duplicate=duplicate||faceAdjacency[base+1u+prior]==otherRow;}
    if(!duplicate){if(count>=${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u){fail(faceRow,ERROR_CAPACITY);return;}
      faceAdjacency[base+1u+count]=otherRow;count+=1u;}}
  faceAdjacency[base]=count;
  for(var axis=0u;axis<3u;axis+=1u){for(var quadrant=0u;quadrant<4u;quadrant+=1u){
    let patchIndex=4u*axis+quadrant;let negative=signedFaceNeighborIdentity(cell,axis,quadrant,false);
    let positive=signedFaceNeighborIdentity(cell,axis,quadrant,true);
    faceAdjacency[base+${1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u+patchIndex]=faceRowForIdentity(negative);
    faceAdjacency[base+${1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence + STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+patchIndex]=faceRowForIdentity(positive);
  }}}

// The changed-frontier walks outward from a changed SOURCE, whereas extendFace
// gathers sources from a DESTINATION's incidence list. The catalog graph is
// undirected, but prove that publication invariant before relying on it: a
// missing reverse edge would otherwise strand a demanded destination. This
// runs after adjacency publication (and on reused topology) and fails closed.
@compute @workgroup_size(256)fn validateAirSupportFrontierReciprocity(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let faceRow=linearItem(wid,lane,groups,256u);
  // The retained-graph marker proves this exact adjacency was already checked
  // by the preceding VALID sparse publication. A dense-oracle publication
  // deliberately leaves no marker, so its first sparse successor validates.
  let faceRows=s(2u)+s(8u);if(faceRow>=faceRows||s(0u)!=0u||s(50u)!=0u){return;}
  let count=adjacencyIncidentCount(faceRow);if(count>${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u){fail(faceRow,ERROR_CAPACITY);return;}
  for(var local=0u;local<count;local+=1u){let other=adjacencyIncident(faceRow,local);
    if(other>=faceRows){failTopology(9u,faceRow);continue;}let otherCount=adjacencyIncidentCount(other);
    if(otherCount>${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u){fail(other,ERROR_CAPACITY);continue;}
    var reverse=false;for(var back=0u;back<otherCount;back+=1u){reverse=reverse||adjacencyIncident(other,back)==faceRow;}
    if(!reverse){failTopology(9u,faceRow);}}}

// The patch centre of one owned face slot. The origin coordinate and extent
// come from the adjacency record resolveAirSupportFaceAdjacency authored for
// this face row, not from re-resolving the row's cell.
//
// This is the hot half of the march. extendFace evaluates it once for its own
// patch and then again for the seed patch of every one of up to 30x4 = 120
// candidates, on every sweep -- 12 prefix waves, up to 12 wide waves, plus the
// persistent tail. Each evaluation used to run faceCellIn (a rowGeometry gather
// for a direct row; SIX recordArena atomicLoads for a support row) and then
// coord(), whose three divisions and modulos are by p.dimensions.x/.y, runtime
// values, so the backend emits emulated integer division rather than the
// multiply-shift it can use for a literal. That is the ALU the 'Extrapolate
// structured ordinary faces' pass (3.06 ms, twice per advance, ALU limiter
// 61%) and 'March Section 5 closest faces' (1.01 ms, twice) are spending.
//
// The stored words ARE coord(cell.x) and cell.y, written by the same faceCellIn
// call the reader used to make, so vec3f(g.xyz) and f32(g.w) reproduce
// vec3f(coord(cell.x)) and f32(cell.y) exactly and every float operation below
// is unchanged and in the same order. Nothing is stored as f32 and reloaded, so
// no expression is ended and no multiply-add is unfused: this moves an INTEGER
// address derivation, not a float value. Gate A.
//
// The three scratch control words the old signature threaded through
// (directRows/bank/supportRows) only ever selected which cell faceCellIn
// resolved. They are settled before this stage and unwritten during it -- which
// is exactly why the resolution can be hoisted to the publication at all -- so
// with the resolution gone the parameters go too, and extendFace stops issuing
// three atomicLoads per invocation to feed them.
fn faceCenter(item:u32)->vec3f{
  let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  let local=item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let axis=local/4u;let quadrant=local%4u;
  let g=adjacencyGeometry(faceRow);let origin=vec3f(g.xyz);let extent=f32(g.w);var center=origin+vec3f(.5*extent);
  center[axis]=origin[axis]+extent;var transverse=0u;for(var a=0u;a<3u;a+=1u){if(a==axis){continue;}
    center[a]=origin[a]+select(.25,.75,(quadrant&(1u<<transverse))!=0u)*extent;transverse+=1u;}return center;}
fn faceCenterQuarter(item:u32)->vec3i{
  let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  let local=item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let axis=local/4u;let quadrant=local%4u;
  let g=adjacencyGeometry(faceRow);let origin=vec3i(g.xyz);let extent=i32(g.w);
  var center=4*origin+vec3i(2*extent);center[axis]=4*origin[axis]+4*extent;
  var transverse=0u;for(var a=0u;a<3u;a+=1u){if(a==axis){continue;}
    center[a]=4*origin[a]+select(extent,3*extent,(quadrant&(1u<<transverse))!=0u);transverse+=1u;}
  return center;}
fn signedFaceCenterQuarter(faceRow:u32,axis:u32,quadrant:u32,positive:bool)->vec3i{
  let g=adjacencyGeometry(faceRow);let origin=vec3i(g.xyz);let extent=i32(g.w);
  var center=4*origin+vec3i(2*extent);
  center[axis]=4*origin[axis]+select(0,4*extent,positive);
  var transverse=0u;for(var a=0u;a<3u;a+=1u){if(a==axis){continue;}
    center[a]=4*origin[a]+select(extent,3*extent,(quadrant&(1u<<transverse))!=0u);transverse+=1u;}
  return center;}

fn frontierAxisCapacity()->u32{return 4u*p.faceCellCapacity;}
fn frontierQueueBase(bank:u32,axis:u32)->u32{return 16u+bank*p.faceCapacity+axis*frontierAxisCapacity();}
fn frontierMarkBase()->u32{return 16u+2u*p.faceCapacity;}
fn appendSeedFrontier(axis:u32,item:u32){let at=atomicAdd(&faceFrontier[axis],1u);
  if(at>=frontierAxisCapacity()){fail(item,ERROR_CAPACITY);return;}
  atomicStore(&faceFrontier[frontierQueueBase(0u,axis)+at],item);}

var<workgroup> seedCounts:array<u32,256>;
fn airSupportSeedCarrier(item:u32)->vec4u{let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  let local=item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let axis=local/4u;let cell=faceCell(faceRow);
  if(cell.x==INVALID){failTopology(7u,item);return vec4u(0u,INVALID,INVALID,0u);}
  if(faceRow<s(2u)&&liquidRow(faceRow)){atomicOr(&scratch[p.directoryFlagOffset+cell.x],0x80000000u);}
  // Paper Section 5: seed each patch by COPYING the exact projected
  // power-face value on its plane — the owning liquid row's face first,
  // else the positive liquid neighbour's coincident face. The distance
  // origin stays this patch centre so the march still orders sources by
  // proximity to the free surface.
  let patchCenter=faceCenter(item);var seed=vec2f(0.,0.);
  if(faceRow<s(2u)&&liquidRow(faceRow)){seed=projectedAxisFaceValue(faceRow,axis,patchCenter);}
  if(seed.y==0.){let otherRow=adjacencyPositive(faceRow,axis,local%4u);
    if(otherRow!=INVALID&&otherRow<s(2u)&&liquidRow(otherRow)){seed=projectedAxisFaceValue(otherRow,axis,patchCenter);}}
  if(seed.y<0.){fail(item,ERROR_SOURCE);return vec4u(0u,INVALID,INVALID,0u);}
  if(seed.y>0.){return vec4u(bitcast<u32>(seed.x),0u,item,1u);}
  return vec4u(0u,INVALID,INVALID,0u);}
@compute @workgroup_size(256)fn seedAirSupportFaces(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);var seeded=0u;
  if(item<s(29u)&&s(0u)==0u&&s(50u)==0u){faceA[item]=vec4u(0u,INVALID,INVALID,0u);
    let carrier=airSupportSeedCarrier(item);if(carrier.w!=0u){faceA[item]=carrier;seeded=1u;}}
  seedCounts[lane]=seeded;workgroupBarrier();for(var width=128u;width>0u;width>>=1u){if(lane<width){seedCounts[lane]+=seedCounts[lane+width];}workgroupBarrier();}
  if(lane==0u&&seedCounts[0]!=0u){atomicAdd(&scratch[25u],seedCounts[0]);}}
@compute @workgroup_size(256)fn seedRetainedAirSupportFaces(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let invocation=linearItem(wid,lane,groups,256u);var seeded=0u;
  if(invocation<s(49u)&&s(0u)==0u&&s(50u)!=0u){let item=s(p.candidateOffset+invocation);
    if(item>=s(29u)){failTopology(10u,invocation);}else{let carrier=airSupportSeedCarrier(item);
      if(carrier.w==0u){failTopology(10u,item);}else{faceB[item]=carrier;seeded=1u;}}}
  seedCounts[lane]=seeded;workgroupBarrier();for(var width=128u;width>0u;width>>=1u){if(lane<width){seedCounts[lane]+=seedCounts[lane+width];}workgroupBarrier();}
  if(lane==0u&&seedCounts[0]!=0u){atomicAdd(&scratch[25u],seedCounts[0]);}}
@compute @workgroup_size(256)fn compactAirSupportSeedFrontier(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  if(item>=s(29u)||s(0u)!=0u||s(50u)!=0u){return;}atomicStore(&faceFrontier[frontierMarkBase()+item],0u);
  if(faceA[item].w!=0u){let at=atomicAdd(&scratch[49u],1u);
    if(at>=3u*p.candidateCapacity){fail(item,ERROR_CAPACITY);return;}
    sw(p.candidateOffset+at,item);appendSeedFrontier((item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u)/4u,item);}}

// Same topology and liquid-mask authority imply the closest-source ordering
// tuple (squared distance, seed identity) is immutable. Only the projected
// velocity carried by each seed is dynamic. Each row refreshes its twelve
// settled carriers from the newly staged value of its retained winning seed;
// unchanged values do not write, and no domain/frontier relaxation is needed.
@compute @workgroup_size(256)fn refreshRetainedAirSupportFaceValues(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let faceRow=linearItem(wid,lane,groups,256u);
  let faceRows=s(2u)+s(8u);if(faceRow>=faceRows||s(0u)!=0u||s(50u)==0u){return;}
  for(var local=0u;local<${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;local+=1u){let item=faceRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+local;
    var carrier=faceA[item];if(carrier.w==0u){continue;}let seedItem=carrier.z;
    if(seedItem>=s(29u)){failTopology(10u,item);continue;}let seed=faceB[seedItem];
    if(seed.w==0u||seed.y!=0u||seed.z!=seedItem){failTopology(10u,item);continue;}
    if(carrier.x!=seed.x){carrier.x=seed.x;faceA[item]=carrier;}}}

// Publish the next march schedule after seed recount. A retained solution must
// reproduce exactly the preceding seed membership; any mismatch fails closed.
// Fresh sparse construction keeps the ordinary face schedule and leaves the
// exact changed-frontier fixed point as the publication authority.
@compute @workgroup_size(1)fn finalizeRetainedAirSupportMarchSchedule(){let retained=s(50u)!=0u;
  var clean=s(0u)==0u;if(retained&&s(25u)!=s(49u)){failTopology(10u,s(25u));clean=false;}
  if(retained){atomicStore(&faceFrontier[10u],1u);}
  writeDispatch(32u,select(vec3u(0u,1u,1u),dispatchFor(s(29u),256u),clean&&!retained));
  if(p.fineFactor==1u&&(p.capturePreceding&8u)!=0u){
    let waveActive=clean&&!retained;
    writeDispatch(10u,select(vec3u(0u,1u,1u),dispatchFor(s(29u),256u),waveActive));
    writeDispatch(13u,select(vec3u(0u,1u,1u),vec3u(1u),waveActive));
    writeDispatch(16u,select(vec3u(0u,1u,1u),vec3u(3u,1u,1u),waveActive));}
  atomicStore(&faceFrontier[11u],select(0u,RETAINED_GRAPH_VALID,clean));}
// The carrier stores squared Euclidean distance. sqrt is strictly monotone on
// non-negative finite values, so it cannot change the closest-seed ordering;
// doing it in every candidate visit was pure work. The one consumer that needs
// physical distance (the detached-air gravity ramp) takes one sqrt per row.
fn canonicalSeedOffset(item:u32,seed:u32)->vec3i{
  let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  return powerTransformVector(faceCenterQuarter(seed)-faceCenterQuarter(item),faceCell(faceRow).w&63u);}
fn canonicalOffsetLess(a:vec3i,b:vec3i)->bool{
  return a.x<b.x||(a.x==b.x&&(a.y<b.y||(a.y==b.y&&a.z<b.z)));}
fn faceDistanceSquared(item:u32,seed:u32)->f32{
  let delta=faceCenterQuarter(item)-faceCenterQuarter(seed);
  let squared=delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
  return .0625*f32(squared);}
fn betterFace(item:u32,candidate:vec4u,best:vec4u)->bool{let candidateDistanceSquared=bitcast<f32>(candidate.y);
  let bestDistanceSquared=bitcast<f32>(best.y);
  // Section 5 copies the value of the face closest to the free surface but
  // leaves exactly equidistant seeds unspecified. A spatial ordering cannot
  // resolve a seed orbit without breaking one of its reflection stabilizers.
  // Normal-velocity magnitude is invariant under every axis reflection and
  // permutation, so use it before the canonical spatial fallback.
  let candidateMagnitude=abs(bitcast<f32>(candidate.x));let bestMagnitude=abs(bitcast<f32>(best.x));
  return candidate.w!=0u&&finiteValue(candidateDistanceSquared)&&(best.w==0u||candidateDistanceSquared<bestDistanceSquared
    ||(candidateDistanceSquared==bestDistanceSquared
      &&(candidateMagnitude<bestMagnitude||(candidateMagnitude==bestMagnitude
        &&(canonicalOffsetLess(canonicalSeedOffset(item,candidate.z),canonicalSeedOffset(item,best.z))
          ||(all(canonicalSeedOffset(item,candidate.z)==canonicalSeedOffset(item,best.z))&&candidate.z<best.z))))));}
// An air leaf may be demanded even when the octree has no allocated leaf on
// its negative side. Positive-only patch ownership then has no stored record
// for one of the ordinary faces incident to this dual-mesh node. Recover that
// face directly from the same Section 5 closest-free-surface seed authority;
// duplicating the opposite face would make the answer depend on ownership.
fn closestSeedFaceAt(faceRow:u32,axis:u32,quadrant:u32,positive:bool)->vec4u{
  let requested=signedFaceCenterQuarter(faceRow,axis,quadrant,positive);
  let transform=faceCell(faceRow).w&63u;var best=vec4u(0u);var bestSquared=0x7fffffffu;
  var bestMagnitude=3.402823e38;var bestOffset=vec3i(0);
  let faceCount=s(29u);
  for(var item=4u*axis;item<faceCount;item+=${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u){
    for(var candidateQuadrant=0u;candidateQuadrant<4u;candidateQuadrant+=1u){
      let candidateItem=item+candidateQuadrant;let candidate=faceA[candidateItem];
      if(candidate.w==0u||bitcast<f32>(candidate.y)!=0.){continue;}
      let delta=faceCenterQuarter(candidate.z)-requested;
      let squared=u32(delta.x*delta.x+delta.y*delta.y+delta.z*delta.z);
      let magnitude=abs(bitcast<f32>(candidate.x));
      let offset=powerTransformVector(delta,transform);
      if(best.w==0u||squared<bestSquared||(squared==bestSquared
          &&(magnitude<bestMagnitude||(magnitude==bestMagnitude
            &&(canonicalOffsetLess(offset,bestOffset)
              ||(all(offset==bestOffset)&&candidate.z<best.z)))))){
        best=candidate;bestSquared=squared;bestMagnitude=magnitude;bestOffset=offset;}
    }
  }
  if(best.w!=0u){best.y=bitcast<u32>(.0625*f32(bestSquared));}
  return best;}
@compute @workgroup_size(256)fn completeAirSupportIncidentFaces(
  @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u,
  @builtin(num_workgroups)groups:vec3u){
  let item=linearItem(wid,lane,groups,256u);if(item>=s(29u)||s(0u)!=0u){return;}
  let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  let local=item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  let axis=local/4u;let quadrant=local%4u;let cell=faceCell(faceRow);
  var completion=vec4u(0u);
  // Section 5 extrapolates the liquid velocity through the air band even when
  // that band meets a solid wall.  Publish the missing negative-side carrier
  // at a closed wall too; regularVectorAt then uses the same marched trace on
  // both domain signs, with stationary solid as the no-carrier fallback.
  if(adjacencyNegative(faceRow,axis,quadrant)==INVALID){
    let retained=incidentFaces[item];
    if(s(50u)!=0u&&retained.w!=0u&&retained.z<s(29u)&&faceA[retained.z].w!=0u){
      completion=faceA[retained.z];completion.y=retained.y;
    }else{completion=closestSeedFaceAt(faceRow,axis,quadrant,false);}
  }
  incidentFaces[item]=completion;
}
// Everything the 30x4 candidate scan needs about the marching patch itself is
// loop-invariant: the published face count and this patch's own centre. They
// used to be re-derived on every candidate — up to 119 redundant repeats per
// lane per sweep of an atomic load, a faceCell gather, and coord()'s emulated
// integer divisions. Hoisting recomputes nothing and reorders nothing, so the
// marched field is bit-identical; it only removes work from the single hottest
// loop in Section 5. The per-candidate faceCenter below is now a four-word read
// of the candidate's own adjacency record — the gather and the divisions are
// resolved once per face row by resolveAirSupportFaceAdjacency.
fn extendFace(item:u32,readA:bool)->bool{let current=select(faceB[item],faceA[item],readA);
  let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let local=item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let axis=local/4u;var best=current;
  let faceCount=s(29u);
  let incidence=adjacencyIncidentCount(faceRow);if(incidence>${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u){fail(item,ERROR_CAPACITY);return false;}
  for(var localFace=0u;localFace<incidence;localFace+=1u){let otherRow=adjacencyIncident(faceRow,localFace);
    if(otherRow==INVALID){continue;}let sourceBase=otherRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis;
    for(var quadrant=0u;quadrant<4u;quadrant+=1u){let source=sourceBase+quadrant;
      if(source>=faceCount){continue;}var candidate=select(faceB[source],faceA[source],readA);if(candidate.w!=0u){
        // candidate.z is the ORIGINAL seed patch, preserved verbatim through
        // every copy, so the carrier metric is the true Euclidean distance to
        // the free surface — the reference lane's closest-point transform.
        // Accumulating per-hop path length instead made the metric an
        // axis-graph geodesic (Manhattan-like), which under-drives diagonal
        // spreading and squares off the dam front.
        let distanceSquared=faceDistanceSquared(item,candidate.z);
        if(!finiteValue(distanceSquared)){fail(item,ERROR_SOURCE);continue;}candidate.y=bitcast<u32>(distanceSquared);}
      if(betterFace(item,candidate,best)){best=candidate;}}}
  let changed=any(best!=current);if(readA){faceB[item]=best;}else{faceA[item]=best;}return changed;}

// Construction-stable dense oracle retained for exact A/B isolation. It is
// not selected unless FLUID_OCTREE_AIR_SUPPORT_CHANGED_FRONTIER=0.
var<workgroup> sweepChanged:atomic<u32>;
var<workgroup> sweepFaceCount:u32;
@compute @workgroup_size(256)fn extendAirSupportFacesAtoB(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  if(lane==0u){atomicStore(&sweepChanged,0u);sweepFaceCount=select(0u,s(29u),s(0u)==0u&&s(44u)==0u);}
  workgroupBarrier();let faceCount=workgroupUniformLoad(&sweepFaceCount);let item=linearItem(wid,lane,groups,256u);
  if(item<faceCount&&extendFace(item,true)){atomicStore(&sweepChanged,1u);}workgroupBarrier();
  if(lane==0u&&atomicLoad(&sweepChanged)!=0u){atomicOr(&scratch[43u],1u);}}
@compute @workgroup_size(256)fn extendAirSupportFacesBtoA(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  if(lane==0u){atomicStore(&sweepChanged,0u);sweepFaceCount=select(0u,s(29u),s(0u)==0u&&s(44u)==0u);}
  workgroupBarrier();let faceCount=workgroupUniformLoad(&sweepFaceCount);let item=linearItem(wid,lane,groups,256u);
  if(item<faceCount&&extendFace(item,false)){atomicStore(&sweepChanged,1u);}workgroupBarrier();
  if(lane==0u&&atomicLoad(&sweepChanged)!=0u){atomicOr(&scratch[43u],1u);}}

@compute @workgroup_size(1)fn advanceAirSupportMarchWave(){let group=s(45u);
  if(group>0u){if(s(43u)==0u){sw(44u,1u);}
    else if(s(44u)==0u){sw(46u,s(46u)+${OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_GROUP}u);}}
  sw(43u,0u);sw(45u,group+1u);}

var<workgroup> relaxationChanged:atomic<u32>;
var<workgroup> relaxationFaceRows:atomic<u32>;
var<workgroup> relaxationFailed:atomic<u32>;
@compute @workgroup_size(256)fn marchAirSupportFacesToFixedPoint(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u){
  if(lane==0u){atomicStore(&relaxationFaceRows,s(2u)+s(8u));
    atomicStore(&relaxationFailed,select(0u,1u,s(0u)!=0u||s(44u)!=0u));
    if(s(0u)==0u){sw(32u+wid.x,${OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX}u+s(46u));}}
  workgroupBarrier();let faceRows=workgroupUniformLoad(&relaxationFaceRows);
  let failed=workgroupUniformLoad(&relaxationFailed);if(failed!=0u){return;}
  let axis=wid.x;let count=4u*faceRows;var readA=${((OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX
    + OCTREE_AIR_SUPPORT_GPU_WIDE_MARCH_WAVES) & 1) === 0 ? "true" : "false"};var tailWave=0u;
  loop{if(lane==0u){atomicStore(&relaxationChanged,0u);}workgroupBarrier();
    for(var local=lane;local<count;local+=256u){let faceRow=local/4u;let quadrant=local%4u;
      let item=faceRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant;
      if(extendFace(item,readA)){atomicStore(&relaxationChanged,1u);}}
    storageBarrier();workgroupBarrier();let changed=workgroupUniformLoad(&relaxationChanged);tailWave+=1u;
    if(lane==0u){sw(32u+axis,${OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX}u+s(46u)+tailWave);sw(35u+axis,changed);}workgroupBarrier();
    if(changed==0u||tailWave>=max(1u,count)){break;}readA=!readA;
  }}

// The compact propagation authority. Queue A contains faces that actually
// changed in the preceding wave (the seed faces for wave zero). Their
// reciprocal neighbours are deduplicated into queue B with a generation mark;
// only those destinations run the expensive 30x4 closest-face gather. All
// proposals read authoritative faceA, then a barrier separates the commit back
// to faceA, preserving the same synchronous monotone relaxation as a dense
// Jacobi wave without copying untouched faces between banks.
var<workgroup> frontierFaceRows:u32;
var<workgroup> frontierCurrentCount:u32;
var<workgroup> frontierActiveCount:u32;
var<workgroup> frontierChangedCount:u32;
var<workgroup> frontierFailed:u32;
fn appendFrontierDestination(axis:u32,item:u32,generation:u32){
  let prior=atomicExchange(&faceFrontier[frontierMarkBase()+item],generation);
  if(prior==generation){return;}let at=atomicAdd(&faceFrontier[3u+axis],1u);
  if(at>=frontierAxisCapacity()){fail(item,ERROR_CAPACITY);return;}
  atomicStore(&faceFrontier[frontierQueueBase(1u,axis)+at],item);}

// Packed work mapping for the occupancy-wide frontier prefix. Consecutive
// lanes alternate axes and advance one queue slot every three lanes, avoiding
// the two provisioned-capacity holes an axis-major mapping would launch when
// the live face-row count is smaller than faceCellCapacity.
fn packedFrontierLane(packed:u32)->vec2u{return vec2u(packed%3u,packed/3u);}
@compute @workgroup_size(256)fn expandAirSupportChangedFrontier(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  let packed=linearItem(wid,lane,groups,256u);let liveAxisCapacity=4u*(s(2u)+s(8u));
  if(packed>=3u*liveAxisCapacity||s(0u)!=0u||atomicLoad(&faceFrontier[10u])!=0u){return;}
  let work=packedFrontierLane(packed);let axis=work.x;let local=work.y;
  let current=atomicLoad(&faceFrontier[axis]);if(local>=current){return;}
  let source=atomicLoad(&faceFrontier[frontierQueueBase(0u,axis)+local]);
  let sourceRow=source/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  if(source>=p.faceCapacity||sourceRow>=s(2u)+s(8u)
      ||((source%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u)/4u)!=axis){fail(source,ERROR_CAPACITY);return;}
  let incidence=adjacencyIncidentCount(sourceRow);
  if(incidence>${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u){fail(source,ERROR_CAPACITY);return;}
  let generation=atomicLoad(&faceFrontier[9u])+1u;
  for(var edge=0u;edge<incidence;edge+=1u){let destinationRow=adjacencyIncident(sourceRow,edge);
    if(destinationRow>=s(2u)+s(8u)){failTopology(9u,sourceRow);continue;}
    for(var quadrant=0u;quadrant<4u;quadrant+=1u){
      appendFrontierDestination(axis,destinationRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant,generation);}}}
@compute @workgroup_size(256)fn relaxAirSupportChangedFrontier(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  let packed=linearItem(wid,lane,groups,256u);let liveAxisCapacity=4u*(s(2u)+s(8u));
  if(packed>=3u*liveAxisCapacity||s(0u)!=0u||atomicLoad(&faceFrontier[10u])!=0u){return;}
  let work=packedFrontierLane(packed);let axis=work.x;let local=work.y;
  if(local>=atomicLoad(&faceFrontier[3u+axis])){return;}
  let item=atomicLoad(&faceFrontier[frontierQueueBase(1u,axis)+local]);extendFace(item,true);}
@compute @workgroup_size(256)fn commitAirSupportChangedFrontier(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  let packed=linearItem(wid,lane,groups,256u);let liveAxisCapacity=4u*(s(2u)+s(8u));
  if(packed>=3u*liveAxisCapacity||s(0u)!=0u||atomicLoad(&faceFrontier[10u])!=0u){return;}
  let work=packedFrontierLane(packed);let axis=work.x;let local=work.y;
  if(local>=atomicLoad(&faceFrontier[3u+axis])){return;}
  let item=atomicLoad(&faceFrontier[frontierQueueBase(1u,axis)+local]);let proposal=faceB[item];let previous=faceA[item];
  if(any(proposal!=previous)){faceA[item]=proposal;let at=atomicAdd(&faceFrontier[6u+axis],1u);
    if(at>=frontierAxisCapacity()){fail(item,ERROR_CAPACITY);}
    else{atomicStore(&faceFrontier[frontierQueueBase(0u,axis)+at],item);}}}
@compute @workgroup_size(1)fn advanceAirSupportChangedFrontier(){
  var changed=0u;for(var axis=0u;axis<3u;axis+=1u){let count=atomicLoad(&faceFrontier[6u+axis]);changed+=count;
    atomicStore(&faceFrontier[axis],count);atomicStore(&faceFrontier[3u+axis],0u);atomicStore(&faceFrontier[6u+axis],0u);}
  atomicAdd(&faceFrontier[9u],1u);if(changed==0u){atomicStore(&faceFrontier[10u],1u);}
  if(p.fineFactor==1u&&(p.capturePreceding&8u)!=0u){let waveActive=s(0u)==0u&&changed!=0u;
    writeDispatch(10u,select(vec3u(0u,1u,1u),dispatchFor(12u*(s(2u)+s(8u)),256u),waveActive));
    writeDispatch(13u,select(vec3u(0u,1u,1u),vec3u(1u),waveActive));
    writeDispatch(16u,select(vec3u(0u,1u,1u),vec3u(3u,1u,1u),waveActive));}}
@compute @workgroup_size(256)fn marchAirSupportFacesChangedFrontier(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u){let axis=wid.x;
  if(lane==0u){frontierFaceRows=s(2u)+s(8u);frontierCurrentCount=atomicLoad(&faceFrontier[axis]);
    frontierFailed=select(0u,1u,s(0u)!=0u||axis>=3u);sw(32u+axis,0u);sw(35u+axis,0u);}
  workgroupBarrier();if(workgroupUniformLoad(&frontierFailed)!=0u){return;}
  let faceRows=workgroupUniformLoad(&frontierFaceRows);let axisCapacity=4u*faceRows;
  let prefixWaves=atomicLoad(&faceFrontier[9u]);var wave=0u;
  loop{let current=workgroupUniformLoad(&frontierCurrentCount);if(current==0u){break;}
    if(lane==0u){atomicStore(&faceFrontier[3u+axis],0u);}workgroupBarrier();
    let generation=prefixWaves+wave+1u;for(var local=lane;local<current;local+=256u){
      let source=atomicLoad(&faceFrontier[frontierQueueBase(0u,axis)+local]);
      if(source>=p.faceCapacity||((source/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u)>=faceRows)
          ||((source%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u)/4u)!=axis){fail(source,ERROR_CAPACITY);continue;}
      let sourceRow=source/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let incidence=adjacencyIncidentCount(sourceRow);
      if(incidence>${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u){fail(source,ERROR_CAPACITY);continue;}
      for(var edge=0u;edge<incidence;edge+=1u){let destinationRow=adjacencyIncident(sourceRow,edge);
        if(destinationRow>=faceRows){failTopology(9u,sourceRow);continue;}
        for(var quadrant=0u;quadrant<4u;quadrant+=1u){
          appendFrontierDestination(axis,destinationRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant,generation);}}}
    storageBarrier();workgroupBarrier();if(lane==0u){frontierActiveCount=atomicLoad(&faceFrontier[3u+axis]);
      atomicStore(&faceFrontier[6u+axis],0u);}workgroupBarrier();let activeCount=workgroupUniformLoad(&frontierActiveCount);
    for(var local=lane;local<activeCount;local+=256u){let item=atomicLoad(&faceFrontier[frontierQueueBase(1u,axis)+local]);
      extendFace(item,true);}
    storageBarrier();workgroupBarrier();
    for(var local=lane;local<activeCount;local+=256u){let item=atomicLoad(&faceFrontier[frontierQueueBase(1u,axis)+local]);
      let proposal=faceB[item];let previous=faceA[item];if(any(proposal!=previous)){faceA[item]=proposal;
        let at=atomicAdd(&faceFrontier[6u+axis],1u);if(at>=frontierAxisCapacity()){fail(item,ERROR_CAPACITY);}
        else{atomicStore(&faceFrontier[frontierQueueBase(0u,axis)+at],item);}}}
    storageBarrier();workgroupBarrier();if(lane==0u){frontierChangedCount=atomicLoad(&faceFrontier[6u+axis]);
      frontierCurrentCount=frontierChangedCount;wave+=1u;sw(32u+axis,prefixWaves+wave);
      if(frontierChangedCount!=0u&&prefixWaves+wave>=max(1u,axisCapacity)){sw(35u+axis,1u);frontierCurrentCount=0u;}}
    workgroupBarrier();}
}

fn quadrantAt(cell:vec4u,axis:u32,point:vec3f)->u32{let origin=vec3f(coord(cell.x));var result=0u;var transverse=0u;
  for(var a=0u;a<3u;a+=1u){if(a==axis){continue;}if(point[a]>=origin[a]+.5*f32(cell.y)){result|=1u<<transverse;}transverse+=1u;}return result;}
fn ownedFaceQuadrant(faceRow:u32,axis:u32,quadrant:u32)->vec4u{
  return faceA[faceRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant];}
fn canonicalAirSupportSum(values:array<f32,31>,count:u32)->f32{
  var sorted=values;
  for(var i=1u;i<count;i+=1u){let value=sorted[i];var j=i;
    loop{if(j==0u||abs(sorted[j-1u])<=abs(value)){break;}sorted[j]=sorted[j-1u];j-=1u;}
    sorted[j]=value;}
  var sum=0.;var i=0u;
  loop{if(i>=count){break;}let magnitude=abs(sorted[i]);var balance=0;var j=i;
    loop{if(j>=count||abs(sorted[j])!=magnitude){break;}
      if(sorted[j]>0.){balance+=1;}else if(sorted[j]<0.){balance-=1;}j+=1u;}
    sum+=f32(balance)*magnitude;i=j;}
  return sum;}
fn canonicalAirSupportPair(a:f32,b:f32)->f32{
  var values:array<f32,31>;values[0]=a;values[1]=b;
  return canonicalAirSupportSum(values,2u);}
fn canonicalAirSupportDot(a:vec3f,b:vec3f)->f32{
  var values:array<f32,31>;values[0]=a.x*b.x;values[1]=a.y*b.y;values[2]=a.z*b.z;
  return canonicalAirSupportSum(values,3u);}
// Interpolate the marched regular-octree face field at an arbitrary power-face
// centroid, then project that vector onto the power-face normal.
fn regularVectorAt(faceRow:u32,point:vec3f)->vec4f{let cell=faceCell(faceRow);let origin=vec3f(coord(cell.x));var result=vec3f(0.);
  for(var axis=0u;axis<3u;axis+=1u){
    var fixedMask=0u;var fixedBits=0u;var transverse=0u;
    for(var a=0u;a<3u;a+=1u){if(a==axis){continue;}
      let coordinate=round(clamp((point[a]-origin[a])/f32(cell.y),0.,1.)*65536.)/65536.;
      let bit=1u<<transverse;if(coordinate<.5){fixedMask|=bit;}
      else if(coordinate>.5){fixedMask|=bit;fixedBits|=bit;}transverse+=1u;}
    var terms:array<f32,31>;var termCount=0u;
    for(var quadrant=0u;quadrant<4u;quadrant+=1u){if((quadrant&fixedMask)!=fixedBits){continue;}
      var positive=ownedFaceQuadrant(faceRow,axis,quadrant);
      // In an AIR reconstruction Section 5's marched face is authoritative up
      // to the wall.  A closed wall supplies the stationary-solid value only
      // when this side has no carrier; applying zero unconditionally pins a
      // receding lid film while the opposite domain sign consumes extension.
      if(positive.w==0u&&u32(origin[axis])+cell.y==p.dimensions[axis]
          &&(p.closedBoundaryMask&(1u<<(2u*axis+1u)))!=0u){positive=vec4u(bitcast<u32>(0.),0u,INVALID,1u);}
      let negativeRow=adjacencyNegative(faceRow,axis,quadrant);var negative=vec4u(0u);
      if(negativeRow!=INVALID){let negativeCell=faceCell(negativeRow);
        let negativeQuadrant=select(quadrant,quadrantAt(negativeCell,axis,point),negativeCell.y!=cell.y);
        negative=ownedFaceQuadrant(negativeRow,axis,negativeQuadrant);}
      else{negative=incidentFaces[faceRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant];
        if(negative.w==0u&&u32(origin[axis])==0u&&(p.closedBoundaryMask&(1u<<(2u*axis)))!=0u){negative=vec4u(bitcast<u32>(0.),0u,INVALID,1u);}}
      if(positive.w==0u&&negative.w==0u){
      // No marched value on either side: this cell's face-graph component
      // holds no seeded liquid face. Since emitFineBandAirSupportCandidates
      // now enrolls every accepted air leaf, the march domain is the paper's
      // contiguous air partition and this can only be air sealed away from
      // all liquid by solids or closed walls (a fine film that has never had
      // liquid rows still reaches the bulk's seeds through the far-air
      // leaves). Stationary air is the correct content for a sealed pocket;
      // failing closed instead froze the whole epoch on the first island.
      atomicAdd(&scratch[41u],1u);
      atomicMin(&scratch[42u],(((cell.w>>6u)&0xffu)<<16u)|(cell.x<<3u)|axis);
        terms[termCount]=0.;termCount+=1u;continue;}
      let positiveValue=bitcast<f32>(select(negative.x,positive.x,positive.w!=0u));let negativeValue=bitcast<f32>(select(positive.x,negative.x,negative.w!=0u));
      let t=round(clamp((point[axis]-origin[axis])/f32(cell.y),0.,1.)*65536.)/65536.;
      terms[termCount]=canonicalAirSupportPair((1.-t)*negativeValue,t*positiveValue);termCount+=1u;}
    result[axis]=canonicalAirSupportSum(terms,termCount)/f32(termCount);}
  return vec4f(result,1.);}

// Euclidean distance from this row's nearest marched face patch to its
// original seed patch — the free-surface proximity the march already carries.
fn rowSeedDistance(faceRow:u32)->f32{var bestSquared=3.402823e38;
  // faceA owns only each cell's positive-axis patches.  A reflection reverses
  // that ownership: this row's positive face maps to the reflected row's
  // negative face, which is stored on its negative neighbour.  The gravity
  // ramp is a cell-centred scalar, so measure the closest carrier over both
  // sides of every incident face.  Looking at the owned half alone gives
  // reflected cells different ramps even when the marched field is exact.
  for(var axis=0u;axis<3u;axis+=1u){for(var quadrant=0u;quadrant<4u;quadrant+=1u){
    let positive=ownedFaceQuadrant(faceRow,axis,quadrant);
    if(positive.w!=0u){bestSquared=min(bestSquared,bitcast<f32>(positive.y));}
    let negativeRow=adjacencyNegative(faceRow,axis,quadrant);
    if(negativeRow!=INVALID){
      let point=faceCenter(faceRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant);
      let negativeQuadrant=quadrantAt(faceCell(negativeRow),axis,point);
      let negative=ownedFaceQuadrant(negativeRow,axis,negativeQuadrant);
      if(negative.w!=0u){bestSquared=min(bestSquared,bitcast<f32>(negative.y));}
    }
  }}
  return sqrt(bestSquared);}
fn reconstructedFaceVector(faceRow:u32)->vec4f{let cell=faceCell(faceRow);let caseId=cell.z;let transform=cell.w&63u;let header=caseHeader(caseId);
  if(header.x==INVALID||header.x>arrayLength(&catalogFaces)||header.y>arrayLength(&catalogFaces)-header.x){fail(faceRow,ERROR_CATALOG);return vec4f(0.,0.,0.,-1.);}
  var termsX:array<f32,31>;var termsY:array<f32,31>;var termsZ:array<f32,31>;
  let anchorCenter=vec3f(coord(cell.x))+.5*f32(cell.y);
  for(var local=0u;local<header.y;local+=1u){let global=header.x+local;let slot=catalogFaces[global];
    let centroid=anchorCenter+f32(cell.y)*inverseTransform(slot.areaCentroid.yzw,transform);let interpolated=regularVectorAt(faceRow,centroid);
    if(!validVector(interpolated)){failTopology(8u,faceRow);
      let detail=0x80000000u|(local&0xffu)|((u32(interpolated.x)&3u)<<8u)
        |((u32(interpolated.y+16.*interpolated.z)&0x3fu)<<10u);
      loop{let exchange=atomicCompareExchangeWeak(&recordArena[14u],0u,detail);
        if(exchange.exchanged||exchange.old_value!=0u){break;}}
      return vec4f(f32(local),interpolated.x,interpolated.y+16.*interpolated.z,-1.);}let normal=normalize(inverseTransform(slot.normalInverseDistance.xyz,transform));
    let sample=canonicalAirSupportDot(interpolated.xyz,normal);let coefficient=p.reconstructionOffset+3u*global;
    if(coefficient+2u>=arrayLength(&denseCatalog)||!finiteValue(sample)){fail(faceRow,ERROR_CATALOG);return vec4f(0.,0.,0.,-1.);}
    let term=vec3f(bitsf(coefficient),bitsf(coefficient+1u),bitsf(coefficient+2u))*sample;
    termsX[local]=term.x;termsY[local]=term.y;termsZ[local]=term.z;}
  let canonical=vec3f(canonicalAirSupportSum(termsX,header.y),
    canonicalAirSupportSum(termsY,header.y),canonicalAirSupportSum(termsZ,header.y));
  // Gravity over the extension band. A sub-grid film (ceiling sheet, wall-seam
  // band) owns no liquid rows, so nothing ever integrated the body force into
  // the only field that transports its phi — and near seams the closest liquid
  // seeds are wall-constrained faces whose values are ~0, leaving the film in
  // frozen equilibrium. Add this substep's body-force increment at the one
  // reconstruction site every air consumer shares (fine transport, the dry-row
  // staggered substitution, committed direct-air rows). Bounded by
  // construction: the march rebuilds every air vector from the projected
  // liquid seeds each epoch, so this is one g*dt, never an accumulating field
  // — the pathology that keeps forceFamily's wet gate cannot recur here.
  //
  // Ramped by the marched seed distance: air hugging the liquid must stay the
  // paper's exact closest-face copy — the surface sampler ingests it, and an
  // un-projected impulse there pumps splash energy (measured +30% peak speed
  // on the minimal dam) and biases hydrostatic rest. Only detached air, whose
  // dynamics no liquid row carries, receives the body force.
  let gravityRamp=clamp((rowSeedDistance(faceRow)-1.5)/1.5,0.,1.);
  let result=inverseTransform(canonical,transform)+gravityRamp*p.airGravityDt;
  if(!finiteValue(result.x)||!finiteValue(result.y)||!finiteValue(result.z)){fail(faceRow,ERROR_SOURCE);return vec4f(0.,0.,0.,-1.);}
  return vec4f(result,1.);}

var<workgroup> reconstructExpected:array<u32,256>;
var<workgroup> reconstructCompleted:array<u32,256>;
@compute @workgroup_size(256)fn reconstructAirSupportVectors(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let faceRow=linearItem(wid,lane,groups,256u);
  let faceRows=s(2u)+s(8u);var expected=0u;var completed=0u;
  if(faceRow<faceRows&&s(0u)==0u&&!publishedLiquidRow(faceRow)
      &&(faceRow>=s(2u)||publishedDirectDemandedRow(faceRow))){expected=1u;let result=reconstructedFaceVector(faceRow);
    if(validVector(result)){if(faceRow<s(2u)){if(faceRow>=arrayLength(&directAirVectors)){fail(faceRow,ERROR_CAPACITY);}
        else{directAirVectors[faceRow]=result;completed=1u;}}
      else{let support=faceRow-s(2u);if(support>=arrayLength(&supportVectors)){fail(faceRow,ERROR_CAPACITY);}
        else{supportVectors[support]=result;let mirror=p.recordVectorOffset+4u*support;
          atomicStore(&recordArena[mirror],bitcast<u32>(result.x));atomicStore(&recordArena[mirror+1u],bitcast<u32>(result.y));
          atomicStore(&recordArena[mirror+2u],bitcast<u32>(result.z));atomicStore(&recordArena[mirror+3u],bitcast<u32>(result.w));completed=1u;}}}
    else if(faceRow<s(2u)){if(faceRow<arrayLength(&directAirVectors)){directAirVectors[faceRow]=result;}}
    else{let support=faceRow-s(2u);if(support<arrayLength(&supportVectors)){supportVectors[support]=result;
      let mirror=p.recordVectorOffset+4u*support;if(mirror+3u<arrayLength(&recordArena)){
        atomicStore(&recordArena[mirror],bitcast<u32>(result.x));atomicStore(&recordArena[mirror+1u],bitcast<u32>(result.y));
        atomicStore(&recordArena[mirror+2u],bitcast<u32>(result.z));atomicStore(&recordArena[mirror+3u],bitcast<u32>(result.w));}}}}
  reconstructExpected[lane]=expected;reconstructCompleted[lane]=completed;workgroupBarrier();
  for(var width=128u;width>0u;width>>=1u){if(lane<width){reconstructExpected[lane]+=reconstructExpected[lane+width];
      reconstructCompleted[lane]+=reconstructCompleted[lane+width];}workgroupBarrier();}
  if(lane==0u){if(reconstructExpected[0]!=0u){atomicAdd(&scratch[30u],reconstructExpected[0]);}
    if(reconstructCompleted[0]!=0u){atomicAdd(&scratch[28u],reconstructCompleted[0]);}}}

@compute @workgroup_size(1)fn finalizeAirSupportMetadata(){if(s(31u)==2u){return;}
  let first=s(1u);if((first>>24u)==6u&&s(59u)==0u){let item=first&0x00ffffffu;
    if(item<s(8u)){let failedAt=p.recordOffset+item*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;
      sw(51u,r(failedAt));sw(52u,r(failedAt+1u));sw(53u,r(failedAt+2u));sw(54u,r(failedAt+3u));
      sw(55u,r(failedAt+4u));sw(56u,r(failedAt+6u));sw(57u,r(failedAt+7u));sw(58u,r(failedAt+5u));
      sw(59u,first);}}
  if((first>>24u)==8u){let row=first&0x00ffffffu;let directRows=s(2u);var rejected=vec4f(0.);
    if(row<directRows&&row<arrayLength(&directAirVectors)){rejected=directAirVectors[row];}
    else if(row>=directRows){let support=row-directRows;let mirror=p.recordVectorOffset+4u*support;
      if(mirror+3u<arrayLength(&recordArena)){rejected=vec4f(bitcast<f32>(r(mirror)),bitcast<f32>(r(mirror+1u)),
        bitcast<f32>(r(mirror+2u)),bitcast<f32>(r(mirror+3u)));}}
    if(rejected.w<0.){let changed=select(0u,1u,(s(35u)|s(36u)|s(37u))!=0u);
      let detail=0x80000000u|(u32(rejected.x)&0xffu)|((u32(rejected.y)&3u)<<8u)
        |((u32(rejected.z)&0x3fu)<<10u)|((min(max(s(32u),max(s(33u),s(34u))),63u)&0x3fu)<<16u)
        |(changed<<22u);let cell=faceCell(row);
      let identity=(min(directRows,0x3fffu)&0x3fffu)|((min(cell.x,0x3fffu)&0x3fffu)<<14u)
        |((min(level(cell.y),15u)&15u)<<28u);
      atomicStore(&recordArena[13u],identity);atomicStore(&recordArena[14u],detail);}}
  let errors=s(0u);let count=s(8u);
  let expectedVectors=s(30u);let clean=errors==0u&&(s(35u)|s(36u)|s(37u))==0u&&count<=p.supportCapacity&&expectedVectors>=count
    &&s(28u)==expectedVectors&&select(true,s(25u)>0u,expectedVectors>0u);
  sw(31u,select(0u,1u,clean));
  atomicStore(&recordArena[0u],${STRUCTURED_AIR_SUPPORT_ARENA_MAGIC}u);atomicStore(&recordArena[1u],${STRUCTURED_AIR_SUPPORT_ARENA_VERSION}u);
  atomicStore(&recordArena[2u],s(3u));atomicStore(&recordArena[3u],select(0u,count,clean));atomicStore(&recordArena[4u],p.supportCapacity);
  atomicStore(&recordArena[5u],0u);
  atomicStore(&recordArena[6u],0u);atomicStore(&recordArena[7u],p.recordOffset);atomicStore(&recordArena[8u],p.recordVectorOffset);
  atomicStore(&recordArena[9u],p.recordArenaWords);atomicStore(&recordArena[10u],${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u);
  atomicStore(&recordArena[11u],${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u);atomicStore(&recordArena[12u],${STRUCTURED_AIR_SUPPORT_VECTOR_WORDS}u);
  let at=p.airControlOffset;atomicStore(&supportArena[at],select(errors,0u,clean));atomicStore(&supportArena[at+1u],select(s(1u),INVALID,clean));
  atomicStore(&supportArena[at+2u],s(3u));atomicStore(&supportArena[at+3u],s(4u));atomicStore(&supportArena[at+4u],s(5u));
  atomicStore(&supportArena[at+5u],s(2u));atomicStore(&supportArena[at+6u],select(0u,count,clean));atomicStore(&supportArena[at+7u],p.supportCapacity);
  atomicStore(&supportArena[at+8u],s(26u));atomicStore(&supportArena[at+9u],s(27u));
  atomicStore(&supportArena[at+10u],s(29u));atomicStore(&supportArena[at+11u],s(25u));
  atomicStore(&supportArena[at+12u],max(s(32u),max(s(33u),s(34u))));
  atomicStore(&supportArena[at+13u],0u);atomicStore(&supportArena[at+14u],${OCTREE_AIR_SUPPORT_LAYOUT_VERSION}u);
  atomicStore(&supportArena[at+15u],select(0u,p.expectedFineGeneration,clean));}

@compute @workgroup_size(256)fn commitAirSupportDirectRows(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let row=linearItem(wid,lane,groups,256u);
  if(s(31u)!=1u||row>=s(2u)||publishedDirectLiquidRow(row)||!publishedDirectDemandedRow(row)){return;}let output=s(4u)*p.rowCapacity+row;
  // Host construction proves both banked destination and staging extents, and
  // reconstruction completeness proves every direct-air staging lane exists.
  rowVelocities[output]=directAirVectors[row];}

@compute @workgroup_size(1)fn commitAirSupportPublication(){if(s(31u)==2u){return;}let clean=s(31u)==1u&&s(0u)==0u;
  // All non-valid metadata and every clean direct-air vector precede these
  // publication-last flags. The suffix VALID store is literally last.
  atomicStore(&recordArena[5u],select(0u,${STRUCTURED_AIR_SUPPORT_ARENA_FLAGS.ready | STRUCTURED_AIR_SUPPORT_ARENA_FLAGS.validated}u,clean));
  atomicStore(&supportArena[p.airControlOffset+13u],select(0u,${OCTREE_AIR_SUPPORT_VALID}u,clean));}
`;
