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
/** Occupancy prefix only. Publication still depends on the persistent tail's
 * GPU-observed fixed point, so this is not a propagation-radius bound. */
export const OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX = 12;
export const OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS =
  3 * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra;
export const OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE =
  OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE + OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS;
export const OCTREE_AIR_SUPPORT_GPU_SCRATCH_CONTROL_WORDS = 41;
export const OCTREE_AIR_SUPPORT_GPU_INDIRECT_RECORDS = 6;
export const OCTREE_AIR_SUPPORT_GPU_FACE_WORDS = 4;
export const OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE =
  1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence + 2 * STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS;

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
} as const);

export function decodeOctreeAirSupportGPUFirstError(packed: number) {
  const word = Number(packed) >>> 0;
  return Object.freeze({ stage: word >>> 24, item: word & 0x00ff_ffff });
}

/** Exact per-entry bind reachability. Binding zero/eleven are uniforms; all
 * other entries are storage resources and no pipeline reaches more than ten. */
export const OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS = Object.freeze({
  beginAirSupportPublication: Object.freeze([0,1,3,7,8,9,10]),
  clearAirSupportDirectory: Object.freeze([0,7]),
  clearAirSupportCandidates: Object.freeze([0,7]),
  clearAirSupportTags: Object.freeze([0,7,9]),
  emitAirSupportCandidates: Object.freeze([0,2,3,4,5,6,7,9,11,18]),
  markAndScanAirSupportCandidates: Object.freeze([0,7]),
  prefixAirSupportBlocks: Object.freeze([0,7]),
  scatterAirSupportRecords: Object.freeze([0,7,8]),
  resolveAirSupportTags: Object.freeze([0,7,8,9]),
  resolveAirSupportTopology: Object.freeze([0,3,7,8,11,12,13,14]),
  markFineBandAirSupportDemand: Object.freeze([0,7,25,26,27,28]),
  closeFineBandAirSupportInterpolationDemand: Object.freeze([0,2,3,4,5,6,7,11,12,13,14]),
  emitFineBandAirSupportCandidates: Object.freeze([0,2,3,7,11]),
  publishAirSupportOwnerDirectory: Object.freeze([0,2,3,7,8,9,11]),
  prepareAirSupportFaces: Object.freeze([0,7]),
  resolveAirSupportFaceAdjacency: Object.freeze([0,2,3,7,8,11,15,16,23]),
  seedAirSupportFaces: Object.freeze([0,1,2,7,8,15,16,18,19,21,23]),
  extendAirSupportFacesAtoB: Object.freeze([0,2,7,8,19,20,23]),
  extendAirSupportFacesBtoA: Object.freeze([0,2,7,8,19,20,23]),
  marchAirSupportFacesToFixedPoint: Object.freeze([0,2,7,8,19,20,23]),
  reconstructAirSupportVectors: Object.freeze([0,2,7,8,15,16,19,22,23,24]),
  finalizeAirSupportMetadata: Object.freeze([0,2,7,8,9,22]),
  commitAirSupportDirectRows: Object.freeze([0,2,7,17,22]),
  commitAirSupportPublication: Object.freeze([0,7,8,9]),
} as const);
export type OctreeAirSupportGPUEntryPoint = keyof typeof OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS;

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
  const directAirVectorBytes = checkedProduct("Air-support direct-air staging bytes", rowCapacity, 16);
  const offsets = {
    control: 0 as const,
    candidates: OCTREE_AIR_SUPPORT_GPU_SCRATCH_CONTROL_WORDS,
    ranks: 0,
    directoryWinners: 0,
    directoryFlags: 0,
    blockCounts: 0,
    blockOffsets: 0,
  };
  offsets.ranks = offsets.candidates + 4 * candidateCapacity;
  offsets.directoryWinners = offsets.ranks + candidateCapacity;
  offsets.directoryFlags = offsets.directoryWinners + domainVolume;
  offsets.blockCounts = offsets.directoryFlags + domainVolume;
  offsets.blockOffsets = offsets.blockCounts + candidateBlockCapacity;
  const scratchWords = offsets.blockOffsets + candidateBlockCapacity;
  const scratchBytes = scratchWords * 4;
  const indirectBytes = OCTREE_AIR_SUPPORT_GPU_INDIRECT_RECORDS * 12;
  return Object.freeze({ rowCapacity, slotCapacity, domainVolume, candidateStride, fineCandidateOffset,
    candidateCapacity, candidateBlockCapacity, faceCellCapacity, faceCapacity, faceBytes,
    faceAdjacencyStride, faceAdjacencyBytes, directAirVectorBytes,
    support, records, scratchWords,
    scratchBytes, indirectBytes, offsets: Object.freeze(offsets),
    allocatedBytes: support.totalBytes + records.allocatedBytes + 2 * faceBytes + faceAdjacencyBytes + directAirVectorBytes
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
  readonly faceAdjacency: GPUBuffer;
  readonly directAirVectors: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: readonly [GPUBuffer, GPUBuffer];
  private readonly ownerParams: GPUBuffer;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private readonly groups: readonly [Readonly<Record<string, GPUBindGroup>>,
    Readonly<Record<string, GPUBindGroup>>];
  private readonly fineDemandGroups?: readonly [readonly [GPUBindGroup, GPUBindGroup],
    readonly [GPUBindGroup, GPUBindGroup]];
  private readonly ownsArena: boolean;
  private destroyed = false;
  private publicationCount = 0;
  private parameterSlot: 0 | 1 = 0;

  constructor(private readonly device: GPUDevice, private readonly inputs: OctreeAirVelocitySupportGPUInputs) {
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
    this.faceAdjacency = device.createBuffer({ label: "Published structured ordinary-face adjacency",
      size: this.plan.faceAdjacencyBytes, usage: storage });
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
    const module = device.createShaderModule({ label: "Structured positive-air identity publication",
      code: octreeAirVelocitySupportPublicationWGSL });
    const make = (entryPoint: string) => device.createComputePipeline({ label: entryPoint,
      layout: "auto", compute: { module, entryPoint } });
    const entries = Object.keys(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS) as OctreeAirSupportGPUEntryPoint[];
    this.pipelines = Object.freeze(Object.fromEntries(entries.map((entry) => [entry, make(entry)])));
    const buffers = new Map<number, GPUBuffer>([
      [1, structured.control], [2, structured.rowGeometry],
      [3, owners.arena], [4, topology.catalogTetrahedronHeaders],
      [5, topology.catalogTetrahedra], [6, topology.catalogTetrahedronVertices],
      [7, this.scratch], [8, this.recordArena], [9, this.arena],
      [10, inputs.boundaryEpoch.buffer], [11, this.ownerParams],
      [12, topology.sameOrFinerDirect], [13, topology.sameOrCoarserDirect],
      [14, topology.catalogLookup], [15, topology.catalogFaces],
      [16, topology.reconstructionData], [17, structured.rowVelocities],
      [18, inputs.liquidMask], [19, this.faceA], [20, this.faceB],
      [21, structured.authority], [22, this.directAirVectors],
      [23, this.faceAdjacency], [24, this.arena],
      ...(inputs.fineSources ? [[25, inputs.fineSources[0].metadata], [26, inputs.fineSources[0].worklist],
        [27, inputs.fineSources[0].flags], [28, inputs.fineSources[0].phi]] as const : []),
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
      "markFineBandAirSupportDemand", "closeFineBandAirSupportInterpolationDemand",
      "emitFineBandAirSupportCandidates",
    ]);
    const configuredEntries = entries.filter((entry) => inputs.fineSources || !fineOnlyEntries.has(entry));
    const makeGroups = (params: GPUBuffer) => Object.freeze(Object.fromEntries(
      configuredEntries.map((entry) => [entry,
        device.createBindGroup({ layout: this.pipelines[entry]!.getBindGroupLayout(0),
          entries: OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS[entry]
            .map((binding) => ({ binding, resource: resource(params, entry, binding) })) }),
      ])));
    this.groups = Object.freeze(this.params.map(makeGroups)) as unknown as
      readonly [Readonly<Record<string, GPUBindGroup>>, Readonly<Record<string, GPUBindGroup>>];
    if (inputs.fineSources) {
      const bindings = OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS.markFineBandAirSupportDemand;
      const makeFineGroup = (params: GPUBuffer, fine: WebGPUFineLevelSetBrickSource) => device.createBindGroup({
        layout: this.pipelines.markFineBandAirSupportDemand!.getBindGroupLayout(0),
        entries: bindings.map((binding) => ({ binding, resource: { buffer: binding === 0 ? params
          : binding === 25 ? fine.metadata
          : binding === 26 ? fine.worklist : binding === 27 ? fine.flags : binding === 28 ? fine.phi
            : buffers.get(binding)! } })),
      });
      this.fineDemandGroups = Object.freeze(this.params.map((params) => Object.freeze(
        inputs.fineSources!.map((fine) => makeFineGroup(params, fine))))) as unknown as
          readonly [readonly [GPUBindGroup, GPUBindGroup], readonly [GPUBindGroup, GPUBindGroup]];
    }
    this.allocatedBytes = this.plan.allocatedBytes + 256
      - (inputs.sharedArena ? this.plan.support.totalBytes : 0);
  }

  private parameterData(expectedEpoch: number, fineSlot?: 0 | 1): ArrayBuffer {
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
      ...Object.values(this.plan.offsets),
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
      this.publicationCount > 0 ? 1 : 0,
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
    return bytes;
  }

  encode(broker: PassBroker, expectedEpoch: number, fineSlot?: 0 | 1): void {
    if (this.destroyed) throw new Error("Air-support GPU producer is destroyed");
    if (fineSlot !== undefined && !this.inputs.fineSources) {
      throw new Error("Air-support fine-demand slot requires configured A/B fine sources");
    }
    const parameterSlot = this.parameterSlot;
    this.parameterSlot = parameterSlot === 0 ? 1 : 0;
    const params = this.params[parameterSlot], groups = this.groups[parameterSlot];
    this.device.queue.writeBuffer(params, 0, this.parameterData(expectedEpoch, fineSlot));
    this.publicationCount += 1;
    let pass = broker.compute({ label: "Initialize structured air-support publication" });
    pass.setPipeline(this.pipelines.beginAirSupportPublication!);
    pass.setBindGroup(0, groups.beginAirSupportPublication!);
    pass.dispatchWorkgroups(1);
    // Storage-authored schedules are copied into an INDIRECT-only buffer. The
    // second copy below is the only other required pass boundary.
    broker.updateIndirectBuffer(this.scratch, 10 * 4, this.indirect, 0, 4 * 12);
    pass = broker.compute({ label: "Publish structured air-support identities" });
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
      const capacity = this.inputs.fineSources![fineSlot].plan.maximumResidentBricks;
      const x = Math.min(capacity, this.device.limits.maxComputeWorkgroupsPerDimension);
      const y = Math.ceil(capacity / x);
      pass.setPipeline(this.pipelines.markFineBandAirSupportDemand!);
      pass.setBindGroup(0, this.fineDemandGroups[parameterSlot][fineSlot]);
      pass.dispatchWorkgroups(x, y);
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
    run("resolveAirSupportTopology", 36);
    run("resolveAirSupportTags", 24);
    pass.setPipeline(this.pipelines.publishAirSupportOwnerDirectory!);
    pass.setBindGroup(0, groups.publishAirSupportOwnerDirectory!);
    pass.dispatchWorkgroups(Math.ceil(this.plan.domainVolume / OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE));
    run("prepareAirSupportFaces");
    broker.updateIndirectBuffer(this.scratch, 32 * 4, this.indirect, 4 * 12, 2 * 12);
    // These indirect-publication words have completed their schedule lifetime.
    // Reuse them for terminal march depth and convergence state.
    broker.clearBuffer(this.scratch, 32 * 4, 6 * 4);
    pass = broker.compute({ label: "Extrapolate structured ordinary faces and reconstruct support vectors" });
    pass.setPipeline(this.pipelines.resolveAirSupportFaceAdjacency!);
    pass.setBindGroup(0, groups.resolveAirSupportFaceAdjacency!);
    pass.dispatchWorkgroupsIndirect(this.indirect, 60);
    pass.setPipeline(this.pipelines.seedAirSupportFaces!); pass.setBindGroup(0, groups.seedAirSupportFaces!);
    pass.dispatchWorkgroupsIndirect(this.indirect, 48);
    // Run a short fully parallel Jacobi prefix to occupy the GPU. It is not a
    // convergence bound: the persistent tail below still marches until a
    // GPU-observed no-change wave and the final gate still rejects otherwise.
    for (let wave = 0; wave < OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX; wave += 1) {
      const name = (wave & 1) === 0 ? "extendAirSupportFacesAtoB" : "extendAirSupportFacesBtoA";
      pass.setPipeline(this.pipelines[name]!);
      pass.setBindGroup(0, groups[name]!);
      pass.dispatchWorkgroupsIndirect(this.indirect, 48);
    }
    broker.fence("Section 5 ordinary-face seeds published");
    // One persistent workgroup owns each independent velocity-axis face graph.
    // That gives WGSL a real global barrier between relaxation waves without host-unrolling
    // a scene/domain bound into thousands of empty passes. It terminates only
    // on a GPU-observed no-change wave; |V| is the Bellman-Ford simple-path
    // bound and therefore a fail-closed proof bound, not a widened band.
    pass = broker.compute({ label: "March Section 5 closest faces to a fixed point" });
    pass.setPipeline(this.pipelines.marchAirSupportFacesToFixedPoint!);
    pass.setBindGroup(0, groups.marchAirSupportFacesToFixedPoint!);
    pass.dispatchWorkgroups(3);
    broker.fence("Section 5 closest-face fixed point published");
    pass = broker.compute({ label: "Reconstruct Section 5 air-support vectors" });
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
    for (const buffer of [this.recordArena, this.scratch, this.faceA, this.faceB, this.faceAdjacency,
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
}
struct Accepted {flags:atomic<u32>,firstError:atomic<u32>,rowCount:u32,epoch:u32,bank:u32,slotCount:u32}
struct Candidate {cell:u32,size:u32,flags:u32,tagWord:u32}
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
fn mortonPart10(v:u32)->u32{var x=v&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;
  x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn morton(cell:u32)->u32{let q=coord(cell);return mortonPart10(q.x)|(mortonPart10(q.y)<<1u)|(mortonPart10(q.z)<<2u);}
fn level(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn lessGeometry(a:vec4u,b:vec4u)->bool{let al=level(a.y);let bl=level(b.y);let am=morton(a.x);let bm=morton(b.x);
  return al<bl||(al==bl&&am<bm);}
fn publishedRow(cell:u32,size:u32)->u32{let wanted=vec4u(cell,size,0u,0u);var lo=0u;var hi=s(2u);
  let base=s(4u)*p.rowCapacity;while(lo<hi){let mid=lo+(hi-lo)/2u;let g=rowGeometry[base+mid];
    if(lessGeometry(g,wanted)){lo=mid+1u;}else{hi=mid;}}if(lo<s(2u)){let g=rowGeometry[base+lo];
    if(g.x==cell&&g.y==size){return lo;}}return INVALID;}
fn candidateAt(item:u32)->Candidate{let at=p.candidateOffset+4u*item;return Candidate(s(at),s(at+1u),s(at+2u),s(at+3u));}
fn setCandidate(item:u32,value:Candidate){let at=p.candidateOffset+4u*item;sw(at,value.cell);sw(at+1u,value.size);sw(at+2u,value.flags);sw(at+3u,value.tagWord);}
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
  // owns them. The extrapolation destination is the demanded closure, not all
  // remote air rows present in the sparse topology.
  atomicOr(&scratch[p.directoryFlagOffset+resolvedCell],flags);
  let direct=publishedRow(resolvedCell,resolvedSize);if(direct!=INVALID){atomicStore(&supportArena[tagWord],direct);return;}
  setCandidate(item,Candidate(resolvedCell,resolvedSize,flags,tagWord));
  atomicMin(&scratch[p.directoryWinnerOffset+resolvedCell],item);
}

@compute @workgroup_size(1)fn beginAirSupportPublication(){
  let precedingFlags=atomicLoad(&supportArena[p.airControlOffset]);
  let storedDetail=atomicLoad(&recordArena[14u]);
  let precedingDetail=select(0u,storedDetail,(storedDetail&0x80000000u)!=0u);
  if(p.capturePreceding!=0u&&precedingFlags!=0u&&s(38u)==0u
      &&atomicLoad(&supportArena[p.airControlOffset+14u])==${OCTREE_AIR_SUPPORT_LAYOUT_VERSION}u){
    sw(38u,precedingFlags);sw(39u,atomicLoad(&supportArena[p.airControlOffset+1u]));
  }
  // Preserve the preceding transaction's terminal control in reserved record
  // words. Failure-only QA can then distinguish a bad producer publication
  // from a later structured-stage rejection without adding a pass or readback
  // to the simulation path.
  if(p.capturePreceding!=0u&&atomicLoad(&recordArena[15u])==0u){
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
  // The candidate generation is only a request. If the accepted authority
  // itself is invalid, preserve a matching prior receipt. A rejected candidate
  // that leaves a clean older epoch is different: rebuild fine-band support
  // against that accepted epoch and the new fine generation, so Section 5 can
  // continue on one coherent (temporarily reused) power topology.
  if(atomicLoad(&accepted.flags)!=0u&&existingReady){
    sw(0u,ERROR_SOURCE|ERROR_GENERATION);sw(1u,0u);sw(31u,2u);
    writeDispatch(10u,vec3u(0u,1u,1u));writeDispatch(13u,vec3u(0u,1u,1u));
    writeDispatch(16u,vec3u(0u,1u,1u));writeDispatch(19u,vec3u(0u,1u,1u));return;
  }
  sw(0u,0u);sw(1u,INVALID);sw(31u,0u);let rows=min(accepted.rowCount,p.rowCapacity);sw(2u,rows);sw(3u,accepted.epoch);
  sw(4u,accepted.bank);var boundary=0u;if(p.boundaryEpochOffset<arrayLength(&boundaryEpoch)){boundary=boundaryEpoch[p.boundaryEpochOffset];}sw(5u,boundary);
  let candidates=p.candidateCapacity;let blocks=p.blockCapacity;sw(6u,candidates);sw(7u,blocks);sw(8u,0u);sw(9u,p.supportCapacity);
  sw(25u,0u);sw(26u,0u);sw(27u,0u);sw(28u,0u);sw(40u,p.expectedFineGeneration);
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
  let clean=s(0u)==0u;writeDispatch(10u,select(vec3u(0u,1u,1u),dispatchFor(2u*p.domainVolume,256u),clean));
  writeDispatch(13u,select(vec3u(0u,1u,1u),dispatchFor(rows*(${OCTREE_AIR_SUPPORT_SELECTOR_STRIDE}u+${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u),256u),clean));
  // One workgroup owns one structured row. Its lanes cooperatively prove the
  // 27-site cube closure once before emitting either cube or tetra candidates.
  writeDispatch(16u,select(vec3u(0u,1u,1u),dispatchFor(rows,1u),clean));
  writeDispatch(19u,select(vec3u(0u,1u,1u),dispatchFor(candidates,256u),clean));
}

@compute @workgroup_size(256)fn clearAirSupportDirectory(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  if(item<p.domainVolume){sw(p.directoryWinnerOffset+item,INVALID);}else if(item<2u*p.domainVolume){sw(p.directoryFlagOffset+item-p.domainVolume,0u);}}
@compute @workgroup_size(256)fn clearAirSupportCandidates(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  if(item<p.candidateCapacity){setCandidate(item,Candidate(INVALID,0u,0u,INVALID));}}

@compute @workgroup_size(256)fn clearAirSupportTags(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  let selectors=s(2u)*${OCTREE_AIR_SUPPORT_SELECTOR_STRIDE}u;let regular=s(2u)*${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u;
  if(item<selectors){atomicStore(&supportArena[p.selectorTagOffset+item],INVALID);}
  else if(item<selectors+regular){atomicStore(&supportArena[p.regularTagOffset+item-selectors],INVALID);}}

fn unpackFineBrick(key:u32)->vec3u{let xy=p.fineBrickDims.x*p.fineBrickDims.y;let z=key/xy;let rem=key-z*xy;
  let y=rem/p.fineBrickDims.x;return vec3u(rem-y*p.fineBrickDims.x,y,z);}
fn fineLocal(local:u32)->vec3u{let z=local/(p.fineR*p.fineR);let rem=local-z*p.fineR*p.fineR;
  let y=rem/p.fineR;return vec3u(rem-y*p.fineR,y,z);}
@compute @workgroup_size(64)fn markFineBandAirSupportDemand(@builtin(workgroup_id)wid:vec3u,
  @builtin(num_workgroups)groups:vec3u,@builtin(local_invocation_index)lane:u32){
  let headerValid=p.fineFactor>0u&&p.transportBandFineCells>0u&&p.expectedFineGeneration>0u
    &&arrayLength(&fineWorklist)>=7u+p.finePageCapacity&&fineWorklist[0]==p.expectedFineGeneration
    &&fineWorklist[2]==p.finePageCapacity&&(fineWorklist[3]&3u)==3u&&fineWorklist[5]==1u
    &&fineWorklist[6]==1u&&fineWorklist[1]<=p.finePageCapacity;
  if(!headerValid){if(lane==0u){fail(0u,ERROR_SOURCE|ERROR_GENERATION);}return;}
  let work=wid.x+wid.y*groups.x;let live=fineWorklist[1];if(work>=live){return;}let id=fineWorklist[7u+work];
  if(id>=p.finePageCapacity||id*10u+2u>=arrayLength(&fineMetadata)
      ||fineMetadata[id*10u]!=id||fineMetadata[id*10u+2u]!=p.expectedFineGeneration){
    if(lane==0u){fail(work,ERROR_SOURCE|ERROR_GENERATION);}return;}
  for(var local=lane;local<p.fineSamplesPerBrick;local+=64u){let index=id*p.fineSamplesPerBrick+local;
    if(index>=arrayLength(&fineFlags)||index>=arrayLength(&finePhi)){fail(index,ERROR_CAPACITY);continue;}
    if((fineFlags[index]&1u)==0u){continue;}let value=finePhi[index];if(!finiteValue(value)){fail(index,ERROR_SOURCE);continue;}
    if(abs(value)>f32(p.transportBandFineCells)*p.fineWidth){continue;}
    let q=unpackFineBrick(fineMetadata[id*10u+1u])*p.fineR+fineLocal(local);
    if(any(q>=p.fineSampleDims)){fail(index,ERROR_SOURCE);continue;}
    let base=q/p.fineFactor;let radius=(p.maxDisplacementFineCells+p.fineFactor-1u)/p.fineFactor;
    for(var dz=-i32(radius);dz<=i32(radius);dz+=1){for(var dy=-i32(radius);dy<=i32(radius);dy+=1){for(var dx=-i32(radius);dx<=i32(radius);dx+=1){
      let demandCell=vec3i(base)+vec3i(dx,dy,dz);if(all(demandCell>=vec3i(0))&&all(demandCell<vec3i(p.dimensions))){
        atomicOr(&scratch[p.directoryFlagOffset+cellOf(vec3u(demandCell))],QUERY_FINE|RECORD_FINE|RECORD_EXTENSION);}}}}}}

fn markFineResolvedOwner(expectedCenter:vec3f,expectedSize:u32,item:u32){
  if(expectedSize==0u){fail(item,ERROR_CATALOG);return;}
  let owner=octreeOwnerPageLookup(vec3i(floor(expectedCenter)));
  if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){failTopology(2u,item);return;}
  let physical=p.fineWidth*f32(p.fineFactor);if(!finiteValue(physical)||physical<=0.){fail(item,ERROR_SOURCE);return;}
  let ownerCenter=vec3f(owner.origin)+.5*f32(owner.size);
  let tolerance=max(1e-5/physical,f32(expectedSize)*2e-5);
  if(owner.size!=expectedSize||any(abs(ownerCenter-expectedCenter)>vec3f(tolerance))){failTopology(2u,item);return;}
  atomicOr(&scratch[p.directoryFlagOffset+cellOf(owner.origin)],RECORD_FINE|RECORD_EXTENSION);
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
      atomicOr(&scratch[p.directoryFlagOffset+cellOf(resolved.origin)],RECORD_FINE|RECORD_EXTENSION);
    }}}}
  return exact;
}

// Section 5 samples the dual mesh: trilinear interpolation needs the 27-cell
// logical stencil, while transition interpolation needs every tetra selector
// of the locally resolved power case. Publish that exact one-hop closure once;
// recurring transport then performs only dense owner/tag gathers.
@compute @workgroup_size(256)fn closeFineBandAirSupportInterpolationDemand(@builtin(global_invocation_id)g:vec3u){
  let item=g.x;if(item>=p.domainVolume||s(0u)!=0u||(s(p.directoryFlagOffset+item)&QUERY_FINE)==0u){return;}
  let owner=octreeOwnerPageLookup(vec3i(coord(item)));if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){failTopology(3u,item);return;}
  let originCell=cellOf(owner.origin);let direct=publishedRow(originCell,owner.size);var caseId=INVALID;var transform=0u;
  if(direct!=INVALID){let geometry=rowGeometry[s(4u)*p.rowCapacity+direct];caseId=geometry.z;transform=geometry.w&63u;
  }else{let descriptor=descriptorForIdentity(owner.origin,owner.size);if(descriptor==INVALID){failTopology(3u,item);return;}
    let resolved=resolveDescriptor(descriptor);caseId=resolved.x;transform=resolved.y;}
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
  for(var tetra=0u;tetra<count;tetra+=1u){let packed=tetrahedra[first+tetra];for(var vertex=0u;vertex<3u;vertex+=1u){
    let selector=(packed>>(8u*vertex))&255u;if(selector>=p.tetraVertexCount||selector>=arrayLength(&tetraVertices)){fail(item,ERROR_CATALOG);continue;}
    let v=tetraVertices[selector];if(v.w<=0.||!finiteValue(v.x)||!finiteValue(v.y)||!finiteValue(v.z)||!finiteValue(v.w)){
      fail(item,ERROR_CATALOG);continue;}
    let sizef=f32(owner.size)*v.w;let selectorSize=u32(round(sizef));
    if(selectorSize==0u||abs(sizef-f32(selectorSize))>2e-4){fail(item,ERROR_CATALOG);continue;}
    let selectorCenter=center+f32(owner.size)*inverseTransform(v.xyz,transform);
    let originf=selectorCenter-vec3f(.5*sizef);let origin=vec3i(round(originf));
    if(any(abs(originf-vec3f(origin))>vec3f(2e-4))){fail(item,ERROR_CATALOG);continue;}
    if(any(origin<vec3i(0))||any(origin+vec3i(i32(selectorSize))>vec3i(p.dimensions))){continue;}
    markFineResolvedOwner(selectorCenter,selectorSize,item);}}}

@compute @workgroup_size(256)fn emitFineBandAirSupportCandidates(@builtin(global_invocation_id)g:vec3u){let item=g.x;
  if(item>=p.domainVolume||s(0u)!=0u){return;}let output=p.fineCandidateOffset+item;
  setCandidate(output,Candidate(INVALID,0u,0u,INVALID));let demanded=s(p.directoryFlagOffset+item);
  if((demanded&RECORD_FINE)==0u){return;}let owner=octreeOwnerPageLookup(vec3i(coord(item)));
  if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u){failTopology(4u,output);return;}let resolvedCell=cellOf(owner.origin);
  atomicOr(&scratch[p.directoryFlagOffset+resolvedCell],demanded);let direct=publishedRow(resolvedCell,owner.size);
  if(direct!=INVALID){return;}setCandidate(output,Candidate(resolvedCell,owner.size,demanded,INVALID));
  atomicMin(&scratch[p.directoryWinnerOffset+resolvedCell],output);}

var<workgroup> emitRowActive:atomic<u32>;
var<workgroup> emitRowRegular:atomic<u32>;
var<workgroup> emitRowGeometry:array<u32,4>;
var<workgroup> emitRowTetraHeader:array<u32,2>;
@compute @workgroup_size(256)fn emitAirSupportCandidates(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  let row=wid.x+wid.y*groups.x;let itemBase=row*p.candidateStride;
  if(lane==0u){
    var enabled=row<s(2u)&&s(0u)==0u&&transportDemandRow(row);var g=vec4u(0u);
    if(enabled){g=rowGeometry[s(4u)*p.rowCapacity+row];if(g.y==0u||g.x>=p.domainVolume){fail(itemBase,ERROR_SOURCE);enabled=false;}}
    emitRowGeometry[0]=g.x;emitRowGeometry[1]=g.y;emitRowGeometry[2]=g.z;emitRowGeometry[3]=g.w;
    atomicStore(&emitRowActive,select(0u,1u,enabled));atomicStore(&emitRowRegular,select(0u,1u,enabled&&g.z==0u));
  }
  workgroupBarrier();if(workgroupUniformLoad(&emitRowActive)==0u){return;}
  let g=vec4u(emitRowGeometry[0],emitRowGeometry[1],emitRowGeometry[2],emitRowGeometry[3]);
  if(g.z==0u&&lane<${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u){
    let dx=i32(lane%3u)-1;let dy=i32((lane/3u)%3u)-1;let dz=i32(lane/9u)-1;
    let center=vec3f(coord(g.x))+.5*f32(g.y);let half=.5*f32(g.y);
    let expectedCenter=clamp(center+vec3f(f32(dx),f32(dy),f32(dz))*f32(g.y),
      vec3f(half),vec3f(p.dimensions)-vec3f(half));
    if(!fineResolvedOwnerMatches(expectedCenter,g.y,itemBase+lane)){atomicStore(&emitRowRegular,0u);}
  }
  workgroupBarrier();let regular=workgroupUniformLoad(&emitRowRegular)!=0u;let q=vec3i(coord(g.x));
  if(lane==0u&&!regular){var valid=g.z<=0xffffffffu/3u;let headerAt=select(0u,3u*g.z,valid);
    valid=valid&&headerAt<=arrayLength(&tetraHeaders)&&arrayLength(&tetraHeaders)-headerAt>=3u;
    var first=0u;var count=0u;if(valid){first=tetraHeaders[headerAt];count=tetraHeaders[headerAt+1u];
      valid=first<=arrayLength(&tetrahedra)&&count<=arrayLength(&tetrahedra)-first
        &&count<=${OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS / 3}u;}
    if(!valid){fail(itemBase,ERROR_CATALOG);}emitRowTetraHeader[0]=first;emitRowTetraHeader[1]=count;
    atomicStore(&emitRowActive,select(0u,1u,valid));}
  workgroupBarrier();let transitionActive=workgroupUniformLoad(&emitRowActive)!=0u;
  if(regular&&lane<${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u){let local=lane;let dx=i32(local%3u)-1;let dy=i32((local/3u)%3u)-1;let dz=i32(local/9u)-1;
    let requestedOrigin=q+vec3i(dx,dy,dz)*i32(g.y);let inDomain=all(requestedOrigin>=vec3i(0))&&all(requestedOrigin+vec3i(i32(g.y))<=vec3i(p.dimensions));
    let tag=p.regularTagOffset+row*${OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE}u+local;
    if(!inDomain){atomicStore(&supportArena[tag],INVALID);return;}let cell=cellOf(vec3u(requestedOrigin));atomicAdd(&scratch[27u],1u);
    demand(row,i32(cell),g.y,RECORD_REGULAR|RECORD_EXTENSION,tag,itemBase+local);return;}
  if(regular){return;}
  if(!transitionActive){return;}
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
    workgroupBarrier();marks[lane]+=add;workgroupBarrier();}if(item<p.candidateCapacity){sw(p.rankOffset+item,select(INVALID,marks[lane]-1u,mark!=0u));}
  if(lane==255u&&block<p.blockCapacity){sw(p.blockCountOffset+block,marks[255u]);}}

var<workgroup> blockScan:array<u32,256>;
@compute @workgroup_size(256)fn prefixAirSupportBlocks(@builtin(local_invocation_index)lane:u32){let blocks=s(7u);let chunk=(blocks+255u)/256u;
  let first=min(blocks,lane*chunk);let last=min(blocks,first+chunk);var total=0u;for(var block=first;block<last;block+=1u){total+=s(p.blockCountOffset+block);}blockScan[lane]=total;workgroupBarrier();
  for(var offset=1u;offset<256u;offset<<=1u){var add=0u;if(lane>=offset){add=blockScan[lane-offset];}workgroupBarrier();blockScan[lane]+=add;workgroupBarrier();}
  var cursor=select(0u,blockScan[lane-1u],lane>0u);for(var block=first;block<last;block+=1u){sw(p.blockOffsetOffset+block,cursor);cursor+=s(p.blockCountOffset+block);}
  if(lane==255u){let count=blockScan[255u];sw(8u,count);if(count>p.supportCapacity){fail(count,ERROR_CAPACITY);}}}

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
  if(item>=s(6u)||s(0u)!=0u){return;}let c=candidateAt(item);if(c.cell==INVALID){return;}let tag=s(p.directoryWinnerOffset+c.cell);
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
fn descriptorForIdentity(origin:vec3u,size:u32)->u32{var sizes:array<u32,18>;var boundary=0u;var finer=false;var coarser=false;
  for(var bit=0u;bit<18u;bit+=1u){let direction=DIRECTIONS[bit];var probe=vec3i(0);
    for(var axis=0u;axis<3u;axis+=1u){probe[axis]=select(select(i32(origin[axis]+size/2u),i32(origin[axis]+size),direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}
    if(any(probe<vec3i(0))||any(probe>=vec3i(p.dimensions))){if(bit<6u){boundary|=1u<<bit;}sizes[bit]=size;continue;}
    let owner=octreeOwnerPageLookup(probe);if((owner.status&OWNER_PAGE_LOOKUP_INVALID)!=0u||owner.size*2u<size||owner.size>size*2u){return INVALID;}
    sizes[bit]=owner.size;finer=finer||owner.size<size;coarser=coarser||owner.size>size;}
  if(finer&&coarser){return INVALID;}var descriptor=boundary<<24u;if(!coarser){for(var bit=0u;bit<18u;bit+=1u){if(sizes[bit]==size){descriptor|=1u<<bit;}}}
  else{let child=(origin/vec3u(size))&vec3u(1u);descriptor|=0x80000000u|child.x|(child.y<<1u)|(child.z<<2u);
    let outward=vec3i(select(-1,1,child.x==1u),select(-1,1,child.y==1u),select(-1,1,child.z==1u));
    let wanted=array<vec3i,6>(vec3i(outward.x,0,0),vec3i(0,outward.y,0),vec3i(0,0,outward.z),
      vec3i(outward.x,outward.y,0),vec3i(outward.x,0,outward.z),vec3i(0,outward.y,outward.z));
    for(var coarse=0u;coarse<6u;coarse+=1u){for(var bit=0u;bit<18u;bit+=1u){if(all(DIRECTIONS[bit]==wanted[coarse])&&sizes[bit]==size*2u){descriptor|=1u<<(coarse+3u);}}}}
  return descriptor;}

@compute @workgroup_size(256)fn resolveAirSupportTopology(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);
  if(item>=s(8u)||s(0u)!=0u){return;}let identity=recordAt(item);let descriptor=descriptorForIdentity(identity.xyz,identity.w);
  if(descriptor==INVALID){failTopology(6u,item);return;}let resolved=resolveDescriptor(descriptor);
  if(resolved.x==INVALID||resolved.x>=p.catalogEntryCount||resolved.y>=48u){fail(item,ERROR_CATALOG);return;}
  let at=p.recordOffset+item*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;atomicStore(&recordArena[at+4u],resolved.x);
  let old=r(at+5u);atomicStore(&recordArena[at+5u],resolved.y|(old&0xffffffc0u));}

// Fine transport consumes this identity-keyed view through the already-bound
// support arena. Each finest cell names its complete accepted octree owner;
// positive-air owners retain their support tag instead of acquiring a compact
// momentum row or falling back to a neighbouring wet row.
@compute @workgroup_size(256)fn publishAirSupportOwnerDirectory(@builtin(global_invocation_id)g:vec3u){
  let item=g.x;if(item>=p.domainVolume||s(0u)!=0u){return;}let output=p.ownerDirectoryOffset+4u*item;
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

fn faceCell(faceRow:u32)->vec4u{if(faceRow<s(2u)){return rowGeometry[s(4u)*p.rowCapacity+faceRow];}
  let support=faceRow-s(2u);if(support>=s(8u)){return vec4u(INVALID);}let identity=recordAt(support);
  let at=p.recordOffset+support*${STRUCTURED_AIR_SUPPORT_RECORD_WORDS}u;return vec4u(cellOf(identity.xyz),identity.w,r(at+4u),r(at+5u));}
fn liquidRow(row:u32)->bool{if(row>=s(2u)){return false;}let at=s(4u)*p.rowCapacity+row;
  if(at>=arrayLength(&liquidMask)){return false;}return liquidMask[at]!=0u;}
fn transportDemandRow(row:u32)->bool{if(row>=s(2u)){return false;}if(liquidRow(row)){return true;}
  let cell=rowGeometry[s(4u)*p.rowCapacity+row];
  for(var direction=0u;direction<18u;direction+=1u){let identity=neighborIdentity(cell,DIRECTIONS[direction]);
    if(identity.x==INVALID){continue;}let other=publishedRow(identity.x,identity.y);
    if(other!=INVALID&&liquidRow(other)){return true;}}
  return false;}
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
  writeDispatch(32u,select(vec3u(0u,1u,1u),dispatchFor(count,256u),clean));
  writeDispatch(35u,select(vec3u(0u,1u,1u),dispatchFor(faceRows,256u),clean));}

fn adjacencyBase(faceRow:u32)->u32{return faceRow*p.faceAdjacencyStride;}
fn adjacencyIncidentCount(faceRow:u32)->u32{return faceAdjacency[adjacencyBase(faceRow)];}
fn adjacencyIncident(faceRow:u32,local:u32)->u32{return faceAdjacency[adjacencyBase(faceRow)+1u+local];}
fn adjacencyNegative(faceRow:u32,axis:u32,quadrant:u32)->u32{
  return faceAdjacency[adjacencyBase(faceRow)+${1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u+4u*axis+quadrant];}
fn adjacencyPositive(faceRow:u32,axis:u32,quadrant:u32)->u32{
  return faceAdjacency[adjacencyBase(faceRow)+${1 + OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence + STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant];}

// Resolve all topology identities once. The six extrapolation waves consume
// only this compact indexed graph; catalog and owner-page traversal is kept in
// the topology-publication stage as required by the sparse Section 5 design.
@compute @workgroup_size(256)fn resolveAirSupportFaceAdjacency(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let faceRow=linearItem(wid,lane,groups,256u);
  let faceRows=s(2u)+s(8u);if(faceRow>=faceRows||s(0u)!=0u){return;}let base=adjacencyBase(faceRow);
  if(base+p.faceAdjacencyStride>arrayLength(&faceAdjacency)){fail(faceRow,ERROR_CAPACITY);return;}
  for(var local=0u;local<p.faceAdjacencyStride;local+=1u){faceAdjacency[base+local]=INVALID;}
  let cell=faceCell(faceRow);if(cell.x==INVALID){failTopology(7u,faceRow);return;}let header=caseHeader(cell.z);
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

fn faceCenter(item:u32)->vec3f{let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
  let local=item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let axis=local/4u;let quadrant=local%4u;
  let cell=faceCell(faceRow);let origin=vec3f(coord(cell.x));let extent=f32(cell.y);var center=origin+vec3f(.5*extent);
  center[axis]=origin[axis]+extent;var transverse=0u;for(var a=0u;a<3u;a+=1u){if(a==axis){continue;}
    center[a]=origin[a]+select(.25,.75,(quadrant&(1u<<transverse))!=0u)*extent;transverse+=1u;}return center;}

var<workgroup> seedCounts:array<u32,256>;
@compute @workgroup_size(256)fn seedAirSupportFaces(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){let item=linearItem(wid,lane,groups,256u);var seeded=0u;
  if(item<s(29u)&&s(0u)==0u){faceA[item]=vec4u(0u,INVALID,INVALID,0u);let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;
    let local=item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let axis=local/4u;let cell=faceCell(faceRow);
    if(cell.x==INVALID){failTopology(7u,item);}else{if(faceRow<s(2u)&&liquidRow(faceRow)){atomicOr(&scratch[p.directoryFlagOffset+cell.x],0x80000000u);}
      // Paper Section 5: seed each patch by COPYING the exact projected
      // power-face value on its plane — the owning liquid row's face first,
      // else the positive liquid neighbour's coincident face. The distance
      // origin stays this patch centre so the march still orders sources by
      // proximity to the free surface.
      let patchCenter=faceCenter(item);var seed=vec2f(0.,0.);
      if(faceRow<s(2u)&&liquidRow(faceRow)){seed=projectedAxisFaceValue(faceRow,axis,patchCenter);}
      if(seed.y==0.){let otherRow=adjacencyPositive(faceRow,axis,local%4u);
        if(otherRow!=INVALID&&otherRow<s(2u)&&liquidRow(otherRow)){seed=projectedAxisFaceValue(otherRow,axis,patchCenter);}}
      if(seed.y<0.){fail(item,ERROR_SOURCE);}
      else if(seed.y>0.){faceA[item]=vec4u(bitcast<u32>(seed.x),0u,item,1u);seeded=1u;}}}
  seedCounts[lane]=seeded;workgroupBarrier();for(var width=128u;width>0u;width>>=1u){if(lane<width){seedCounts[lane]+=seedCounts[lane+width];}workgroupBarrier();}
  if(lane==0u&&seedCounts[0]!=0u){atomicAdd(&scratch[25u],seedCounts[0]);}}
fn betterFace(candidate:vec4u,best:vec4u)->bool{let candidateDistance=bitcast<f32>(candidate.y);let bestDistance=bitcast<f32>(best.y);
  return candidate.w!=0u&&finiteValue(candidateDistance)&&(best.w==0u||candidateDistance<bestDistance
    ||(candidateDistance==bestDistance&&candidate.z<best.z));}
fn extendFace(item:u32,readA:bool)->bool{let current=select(faceB[item],faceA[item],readA);
  let faceRow=item/${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let local=item%${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u;let axis=local/4u;var best=current;
  let incidence=adjacencyIncidentCount(faceRow);if(incidence>${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u){fail(item,ERROR_CAPACITY);return false;}
  for(var localFace=0u;localFace<incidence;localFace+=1u){let otherRow=adjacencyIncident(faceRow,localFace);
    if(otherRow==INVALID){continue;}for(var quadrant=0u;quadrant<4u;quadrant+=1u){let source=otherRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant;
      if(source>=s(29u)){continue;}var candidate=select(faceB[source],faceA[source],readA);if(candidate.w!=0u){
        let distance=bitcast<f32>(candidate.y)+length(faceCenter(item)-faceCenter(source));
        if(!finiteValue(distance)){fail(item,ERROR_SOURCE);continue;}candidate.y=bitcast<u32>(distance);}
      if(betterFace(candidate,best)){best=candidate;}}}
  let changed=any(best!=current);if(readA){faceB[item]=best;}else{faceA[item]=best;}return changed;}

@compute @workgroup_size(256)fn extendAirSupportFacesAtoB(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  let item=linearItem(wid,lane,groups,256u);if(item<s(29u)&&s(0u)==0u){_ = extendFace(item,true);}}
@compute @workgroup_size(256)fn extendAirSupportFacesBtoA(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)groups:vec3u){
  let item=linearItem(wid,lane,groups,256u);if(item<s(29u)&&s(0u)==0u){_ = extendFace(item,false);}}

var<workgroup> relaxationChanged:atomic<u32>;
var<workgroup> relaxationFaceRows:atomic<u32>;
var<workgroup> relaxationFailed:atomic<u32>;
@compute @workgroup_size(256)fn marchAirSupportFacesToFixedPoint(@builtin(local_invocation_index)lane:u32,
  @builtin(workgroup_id)wid:vec3u){
  if(lane==0u){atomicStore(&relaxationFaceRows,s(2u)+s(8u));atomicStore(&relaxationFailed,select(0u,1u,s(0u)!=0u));}
  workgroupBarrier();let faceRows=workgroupUniformLoad(&relaxationFaceRows);
  let failed=workgroupUniformLoad(&relaxationFailed);if(failed!=0u){return;}
  let axis=wid.x;let count=4u*faceRows;
  var readA=${(OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX & 1) === 0 ? "true" : "false"};
  var tailWave=0u;
  loop{
    if(lane==0u){atomicStore(&relaxationChanged,0u);}workgroupBarrier();
    for(var local=lane;local<count;local+=256u){let faceRow=local/4u;let quadrant=local%4u;
      let item=faceRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant;
      if(extendFace(item,readA)){atomicStore(&relaxationChanged,1u);}}
    storageBarrier();workgroupBarrier();let changed=workgroupUniformLoad(&relaxationChanged);tailWave+=1u;
    if(lane==0u){sw(32u+axis,${OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX}u+tailWave);sw(35u+axis,changed);}workgroupBarrier();
    if(changed==0u||tailWave>=max(1u,count)){break;}readA=!readA;
  }}

fn quadrantAt(cell:vec4u,axis:u32,point:vec3f)->u32{let origin=vec3f(coord(cell.x));var result=0u;var transverse=0u;
  for(var a=0u;a<3u;a+=1u){if(a==axis){continue;}if(point[a]>=origin[a]+.5*f32(cell.y)){result|=1u<<transverse;}transverse+=1u;}return result;}
fn ownedFace(faceRow:u32,axis:u32,point:vec3f)->vec4u{let cell=faceCell(faceRow);let quadrant=quadrantAt(cell,axis,point);
  return faceA[faceRow*${STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS}u+4u*axis+quadrant];}
// Interpolate the marched regular-octree face field at an arbitrary power-face
// centroid, then project that vector onto the power-face normal.
fn regularVectorAt(faceRow:u32,point:vec3f)->vec4f{let cell=faceCell(faceRow);let origin=vec3f(coord(cell.x));var result=vec3f(0.);
  for(var axis=0u;axis<3u;axis+=1u){var positive=ownedFace(faceRow,axis,point);
    if(positive.w==0u&&u32(origin[axis])+cell.y==p.dimensions[axis]
        &&(p.closedBoundaryMask&(1u<<(2u*axis+1u)))!=0u){positive=vec4u(bitcast<u32>(0.),0u,INVALID,1u);}
    let quadrant=quadrantAt(cell,axis,point);let negativeRow=adjacencyNegative(faceRow,axis,quadrant);var negative=vec4u(0u);
    if(negativeRow!=INVALID){negative=ownedFace(negativeRow,axis,point);}
    else if(u32(origin[axis])==0u&&(p.closedBoundaryMask&(1u<<(2u*axis)))!=0u){negative=vec4u(bitcast<u32>(0.),0u,INVALID,1u);}
    if(positive.w==0u&&negative.w==0u){
      return vec4f(f32(axis),f32(quadrant),f32((positive.w&1u)|((negative.w&1u)<<1u)),-1.);}
    let positiveValue=bitcast<f32>(select(negative.x,positive.x,positive.w!=0u));let negativeValue=bitcast<f32>(select(positive.x,negative.x,negative.w!=0u));
    let t=clamp((point[axis]-origin[axis])/f32(cell.y),0.,1.);result[axis]=mix(negativeValue,positiveValue,t);}
  return vec4f(result,1.);}

fn reconstructedFaceVector(faceRow:u32)->vec4f{let cell=faceCell(faceRow);let caseId=cell.z;let transform=cell.w&63u;let header=caseHeader(caseId);
  if(header.x==INVALID||header.x>arrayLength(&catalogFaces)||header.y>arrayLength(&catalogFaces)-header.x){fail(faceRow,ERROR_CATALOG);return vec4f(0.,0.,0.,-1.);}
  var canonical=vec3f(0.);let anchorCenter=vec3f(coord(cell.x))+.5*f32(cell.y);
  for(var local=0u;local<header.y;local+=1u){let global=header.x+local;let slot=catalogFaces[global];
    let centroid=anchorCenter+f32(cell.y)*inverseTransform(slot.areaCentroid.yzw,transform);let interpolated=regularVectorAt(faceRow,centroid);
    if(!validVector(interpolated)){failTopology(8u,faceRow);
      let detail=0x80000000u|(local&0xffu)|((u32(interpolated.x)&3u)<<8u)
        |((u32(interpolated.y+16.*interpolated.z)&0x3fu)<<10u);
      loop{let exchange=atomicCompareExchangeWeak(&recordArena[14u],0u,detail);
        if(exchange.exchanged||exchange.old_value!=0u){break;}}
      return vec4f(f32(local),interpolated.x,interpolated.y+16.*interpolated.z,-1.);}let normal=normalize(inverseTransform(slot.normalInverseDistance.xyz,transform));
    let sample=dot(interpolated.xyz,normal);let coefficient=p.reconstructionOffset+3u*global;
    if(coefficient+2u>=arrayLength(&denseCatalog)||!finiteValue(sample)){fail(faceRow,ERROR_CATALOG);return vec4f(0.,0.,0.,-1.);}
    canonical+=vec3f(bitsf(coefficient),bitsf(coefficient+1u),bitsf(coefficient+2u))*sample;}
  let result=inverseTransform(canonical,transform);
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
  let first=s(1u);if((first>>24u)==8u){let row=first&0x00ffffffu;let directRows=s(2u);var rejected=vec4f(0.);
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
