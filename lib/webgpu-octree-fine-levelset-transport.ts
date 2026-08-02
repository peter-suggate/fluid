/** Direct structured-velocity transport for the sparse fine level set.
 *
 * The fine lattice owns phi only. Every characteristic samples the accepted
 * packed A/B structured row publication; there is no face graph, sampling
 * prepass, row directory, or second velocity authority.
 */

import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import type { DirectStructuredVelocitySource } from "./webgpu-octree-structured-velocity-gpu";
import {
  OCTREE_AIR_SUPPORT_SELECTOR_STRIDE,
  type OctreeAirVelocitySupportLayout,
} from "./webgpu-octree-air-velocity-support";
import type { PassBroker } from "./webgpu-pass-broker";
import type { SurfaceInflowState } from "./webgpu-quadtree-builder";
export { structuredFineLevelSetTransportWGSL } from "./webgpu-octree-fine-levelset-transport.wgsl";
import { structuredFineLevelSetTransportWGSL } from "./webgpu-octree-fine-levelset-transport.wgsl";
import { octreeAlgorithmDiagnosticsEnabled } from "./octree-algorithm-diagnostics";

/**
 * Diagnostic-only, shares `FLUID_WORKSET_CENSUS=1` with the structured dynamics
 * census (`webgpu-octree-structured-dynamics.ts`); duplicated here rather than
 * imported so the fine lane keeps no dependency on the family-class module.
 *
 * `publishStructuredFineTransportWorksets` dispatches ONE workgroup per page,
 * so the published class count is literally the workgroup count of `Advect fine
 * phi <class>` -- and 64 lanes per page means a class holding 214 pages can put
 * at most 13,696 threads on a 98,304-slot GPU however wide the lattice is.
 */
export function fineTransportWorksetCensusEnabled(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_WORKSET_CENSUS === "1";
}

export const FLUID_FINE_TRANSPORT_STAGED_ADDRESSING_ENV = "FLUID_FINE_TRANSPORT_STAGED_ADDRESSING";
/** Exact address-only E-5 specialization. Kept switchable for interleaved
 * measurements; incompatible geometries fail back to direct lookup in WGSL. */
export function fineTransportStagedAddressingRequested(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.[FLUID_FINE_TRANSPORT_STAGED_ADDRESSING_ENV] === "1";
}

export const FLUID_FINE_TRANSPORT_B4_ADDRESSING_ENV = "FLUID_FINE_TRANSPORT_B4_ADDRESSING";
/** Exact B4/64-sample integer specialization. The generic arithmetic remains
 * selectable as a differential oracle and for every non-B4 geometry. */
export function fineTransportB4AddressingEnabled(
  plan: Readonly<{ brickResolution: number; samplesPerBrick: number }>,
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return plan.brickResolution === 4 && plan.samplesPerBrick === 64
    && environment?.[FLUID_FINE_TRANSPORT_B4_ADDRESSING_ENV] !== "0";
}

export const FLUID_COARSE_PHI_BFECC_ENV = "FLUID_COARSE_PHI_BFECC";
/** Optional factor-1 quality ladder from the coarse-only plan.  It remains
 * opt-in until its moving-surface benefit clears the volume/energy acceptance
 * gates; factor-4/8 always retain the established transport regardless of the
 * switch. */
export function coarsePhiBFECCEnabled(
  fineFactor: number,
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return fineFactor === 1 && environment?.[FLUID_COARSE_PHI_BFECC_ENV] === "1";
}

export const FINE_LEVELSET_TRANSPORT_CONTROL_BYTES = 64;
export const FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP = 4_096;
export const FINE_LEVELSET_TRANSPORT_MAXIMUM_ENCODED_SUBSTEPS = 64;
/** A sleeping region may move by at most this fraction of one fine cell per
 * step. Topology/row identity and the preceding exact repair set must also be
 * unchanged; this scalar is never sufficient on its own. */
export const FINE_LEVELSET_QUIESCENCE_DISPLACEMENT_EPSILON_CELLS = 1e-5;
export const FLUID_FINE_TRANSPORT_QUIESCENCE_ENV = "FLUID_FINE_TRANSPORT_QUIESCENCE";
/** Default-on exact A/B switch. Zero supplies an impossible negative sleep
 * tolerance while preserving the identical shader and publication graph. */
export function fineTransportQuiescenceEnabled(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.[FLUID_FINE_TRANSPORT_QUIESCENCE_ENV] !== "0";
}
/** Binding zero is uniform; every remaining entry is storage. The dense owner
 * directory coalesces the former row-geometry lookup into air support, leaving
 * transition classes one buffer below Dawn's portable ten-buffer ceiling. */
export const FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS = Object.freeze([
  Object.freeze([0,1,2,3,4,5,6,12,13,20]),
  Object.freeze([0,1,2,3,4,5,6,12,13,20]),
] as const);
const FINE_LEVELSET_TRANSPORT_WORKSET_CLASSES = Object.freeze([0, 2] as const);

interface FineLevelSetTransportPipelineBundle {
  readonly planPipeline: GPUComputePipeline;
  readonly classifyPipeline: GPUComputePipeline;
  readonly reduceWorksetsPipeline: GPUComputePipeline;
  readonly scanWorksetsPipeline: GPUComputePipeline;
  readonly publishWorksetsPipeline: GPUComputePipeline;
  readonly compactWorksetsPipeline: GPUComputePipeline;
  readonly transportPipelines: readonly GPUComputePipeline[];
  readonly reversePipelines: readonly GPUComputePipeline[];
  readonly correctionPipelines: readonly GPUComputePipeline[];
  readonly reduceStatusPipeline: GPUComputePipeline;
  readonly summarizePipeline: GPUComputePipeline;
  readonly commitPipeline: GPUComputePipeline;
  readonly clearDeltaPipeline: GPUComputePipeline;
  readonly reduceDeltaPipeline: GPUComputePipeline;
  readonly deltaPipeline: GPUComputePipeline;
  readonly compactDeltaPipeline: GPUComputePipeline;
}

const fineLevelSetTransportPipelineCache = new WeakMap<GPUDevice,
  Map<string, FineLevelSetTransportPipelineBundle>>();
const fineLevelSetTransportPipelineCompilations = new WeakMap<GPUDevice,
  Map<string, Promise<FineLevelSetTransportPipelineBundle>>>();

export interface FineLevelSetGPUTransportControl {
  departureOutsideBand: number;
  nonfiniteVelocity: number;
  processed: number;
  committed: boolean;
  extrapolatedVelocity: number;
  maximumDisplacementFineCells: number;
  /** Kept as a telemetry ABI field; now counts rejected structured epochs. */
  structuredAuthorityUnavailable: number;
  velocityUnavailable: number;
  invalidVelocityStatus?: number;
  nonpositiveVelocityResult?: number;
  velocityStatusReasonOr?: number;
  firstInvalidVelocityStatus?: number;
  firstInvalidVelocityLocalIndex?: number;
  firstInvalidVelocityPosition?: readonly [number, number, number];
}


export type FineLevelSetGPUBoundaryPolicy = "strict" | "closed-neumann";

export interface FineLevelSetGPUTransportOptions {
  timestep: number;
  /** Authored nozzle source applied directly to resident fine phi. */
  inflow?: SurfaceInflowState;
  generation?: number;
  transportBandCells?: number;
  /** Fine-cell closure proven resident by the destination topology publication. */
  maximumBacktraceFineCells?: number;
  boundaryPolicy?: FineLevelSetGPUBoundaryPolicy;
  openTopBoundary?: boolean;
  /** Conservatively disables sleeping when an authored moving solid can wake
   * a dependency cone without first producing liquid velocity. */
  dynamicBoundary?: boolean;
}

export interface FineLevelSetGPUTransportPlan {
  readonly queryCapacity: number;
  readonly velocityChunkCapacity: number;
  readonly chunkCount: 1;
  readonly topologyDeltaBytes: number;
  readonly pageStatusBytes: number;
  readonly worksetBytes: number;
  readonly activitySnapshotBytes: number;
  readonly controlBytes: typeof FINE_LEVELSET_TRANSPORT_CONTROL_BYTES;
  readonly allocatedBytes: number;
}

export interface FineLevelSetGPUTransportPassPlan {
  readonly chunkCount: number;
  readonly segmentCount: 64;
  readonly passesPerSegment: 0;
  readonly passesPerChunk: 1;
  readonly encodedPasses: 10;
}

export function planFineLevelSetGPUTransport(queryCapacity: number,
  _velocityChunkCapacity = queryCapacity, pageCapacity = 0): FineLevelSetGPUTransportPlan {
  if (!Number.isSafeInteger(queryCapacity) || queryCapacity < 1
    || !Number.isSafeInteger(pageCapacity) || pageCapacity < 0) {
    throw new RangeError("Fine transport capacities must be positive integers");
  }
  const pages = Math.max(1, pageCapacity);
  const topologyDeltaBytes = (8 + 3 * pages) * 4;
  // Three words per page: classification uses {class, work-id}; transport
  // reuses them as two packed u16 counter pairs plus exact displacement.
  const pageStatusBytes = pages * 12;
  const worksetBytes = 128 + pages * 4;
  const activitySnapshotBytes = pages * 4;
  return { queryCapacity, velocityChunkCapacity: queryCapacity, chunkCount: 1,
    topologyDeltaBytes, pageStatusBytes, worksetBytes, activitySnapshotBytes,
    controlBytes: FINE_LEVELSET_TRANSPORT_CONTROL_BYTES,
    allocatedBytes: 320 + pageStatusBytes + worksetBytes + activitySnapshotBytes + topologyDeltaBytes
      + FINE_LEVELSET_TRANSPORT_CONTROL_BYTES + 2 * 512 };
}

export function planFineLevelSetGPUTransportPasses(
  _plan: Pick<FineLevelSetGPUTransportPlan, "chunkCount">,
  _segmentCount: 1 | 4 | 8,
): FineLevelSetGPUTransportPassPlan {
  return { chunkCount: 1, segmentCount: FINE_LEVELSET_TRANSPORT_MAXIMUM_ENCODED_SUBSTEPS,
    passesPerSegment: 0, passesPerChunk: 1, encodedPasses: 10 };
}

export function unpackFineLevelSetGPUTransportControl(words: ArrayLike<number>): FineLevelSetGPUTransportControl {
  if (words.length < 8) throw new RangeError("Fine transport control needs eight words");
  return { departureOutsideBand: Number(words[0]) >>> 0, nonfiniteVelocity: Number(words[1]) >>> 0,
    processed: Number(words[2]) >>> 0, committed: Number(words[3]) !== 0,
    extrapolatedVelocity: Number(words[4]) >>> 0, maximumDisplacementFineCells: Number(words[5]) >>> 0,
    structuredAuthorityUnavailable: Number(words[6]) >>> 0, velocityUnavailable: Number(words[7]) >>> 0,
    ...(words.length >= 12 ? { invalidVelocityStatus: Number(words[8]) >>> 0,
      nonpositiveVelocityResult: Number(words[9]) >>> 0,
      velocityStatusReasonOr: Number(words[10]) >>> 0,
      firstInvalidVelocityStatus: Number(words[11]) >>> 0 } : {}),
    ...(words.length >= 16 ? { firstInvalidVelocityLocalIndex: Number(words[12]) >>> 0,
      firstInvalidVelocityPosition: [new Float32Array(new Uint32Array([Number(words[13]) >>> 0]).buffer)[0],
        new Float32Array(new Uint32Array([Number(words[14]) >>> 0]).buffer)[0],
        new Float32Array(new Uint32Array([Number(words[15]) >>> 0]).buffer)[0]] as const } : {}) };
}

export interface FineLevelSetTransportTopologyDelta {
  readonly buffer: GPUBuffer;
  readonly pageCapacity: number;
  /** Header word seven: measured complete-characteristic displacement in fine cells. */
  readonly maximumDisplacementOffsetWords: 7;
  readonly candidateKeysOffsetWords: 8;
  readonly changedKeysOffsetWords: number;
}

export interface StructuredFineTransportResources {
  readonly structured: DirectStructuredVelocitySource;
  readonly topology: OctreePowerTopologySource;
  /** Section 5 positive-air vectors and direct selector/stencil tags. When
   * omitted, the transport schedule deliberately publishes zero work. */
  readonly airSupport?: Readonly<{
    readonly arena: GPUBuffer;
    readonly layout: OctreeAirVelocitySupportLayout;
    /** Accepted structured-boundary control; epoch is word four. */
    readonly boundaryControl: GPUBuffer;
  }>;
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly maximumLeafSize: number;
}

export class WebGPUFineLevelSetTransport {
  readonly control: GPUBuffer;
  readonly queryCapacity: number;
  readonly plan: FineLevelSetGPUTransportPlan;
  readonly topologyDelta: FineLevelSetTransportTopologyDelta;
  private readonly params: GPUBuffer;
  /** Post-submit accounting control; never consumed by host scheduling. */
  readonly governor: GPUBuffer;
  /** Copy-published indirect records. Keeping this buffer INDIRECT-only avoids
   * aliasing the writable governor in any compute synchronization scope. */
  private readonly indirectDispatch: GPUBuffer;
  private planPipeline!: GPUComputePipeline;
  private classifyPipeline!: GPUComputePipeline;
  private reduceWorksetsPipeline!: GPUComputePipeline;
  private scanWorksetsPipeline!: GPUComputePipeline;
  private publishWorksetsPipeline!: GPUComputePipeline;
  private compactWorksetsPipeline!: GPUComputePipeline;
  private transportPipelines!: readonly GPUComputePipeline[];
  private reversePipelines!: readonly GPUComputePipeline[];
  private correctionPipelines!: readonly GPUComputePipeline[];
  private reduceStatusPipeline!: GPUComputePipeline;
  private summarizePipeline!: GPUComputePipeline;
  private commitPipeline!: GPUComputePipeline;
  private clearDeltaPipeline!: GPUComputePipeline;
  private reduceDeltaPipeline!: GPUComputePipeline;
  private deltaPipeline!: GPUComputePipeline;
  private compactDeltaPipeline!: GPUComputePipeline;
  private readonly samplingCatalog: GPUBuffer;
  /** Exact logical fine-page identities from the preceding use of this A/B
   * transport instance. This is a wake oracle, never a phi cache. */
  private readonly activitySnapshot: GPUBuffer;
  private readonly samplingOffsets: readonly number[];
  private planGroup!: GPUBindGroup;
  private classifyGroup!: GPUBindGroup;
  private reduceWorksetsGroup!: GPUBindGroup;
  private scanWorksetsGroup!: GPUBindGroup;
  private publishWorksetsGroup!: GPUBindGroup;
  private compactWorksetsGroup!: GPUBindGroup;
  private transportGroups!: readonly GPUBindGroup[];
  private reverseGroups!: readonly GPUBindGroup[];
  private correctionGroups!: readonly GPUBindGroup[];
  private reduceStatusGroup!: GPUBindGroup;
  private summarizeGroup!: GPUBindGroup;
  private commitGroup!: GPUBindGroup;
  private clearDeltaGroup!: GPUBindGroup;
  private reduceDeltaGroup!: GPUBindGroup;
  private deltaGroup!: GPUBindGroup;
  private compactDeltaGroup!: GPUBindGroup;
  private readonly pipelineCacheKey: string;
  private readonly pipelineConstants: Readonly<Record<string, number>>;
  private pipelineInitialization?: Promise<void>;
  /** 256-page blocks the widened publication trio classifies and scatters. */
  private readonly scanBlocks: number;
  /** Diagnostic-only per-class page-count census; see `censusTick`. */
  private readonly censusEnabled = fineTransportWorksetCensusEnabled();
  private censusStaging?: GPUBuffer;
  private censusPhase: "idle" | "copied" | "mapping" = "idle";
  private censusStep = 0;
  private readonly bfeccEnabled: boolean;
  private readonly reversePhi?: GPUBuffer;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource,
    private readonly resources: StructuredFineTransportResources,
    deferPipelineCompilation = false) {
    const { structured, topology } = resources;
    if (!(resources.physicalCellSize > 0) || !Number.isFinite(resources.physicalCellSize)
      || !Number.isSafeInteger(resources.maximumLeafSize) || resources.maximumLeafSize < 1
      || !topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra
      || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Structured fine transport resources are invalid or undersized");
    }
    const air = resources.airSupport;
    if (air && (air.layout.rowCapacity !== structured.plan.rowCapacity
      || air.layout.slotCapacity !== structured.plan.slotCapacity
      || air.layout.selectorStride !== OCTREE_AIR_SUPPORT_SELECTOR_STRIDE
      || air.layout.ownerDirectoryCellCapacity < resources.dimensions[0]
        * resources.dimensions[1] * resources.dimensions[2]
      || air.layout.ownerDirectorySlotCapacity
        < 2 * (structured.plan.rowCapacity + air.layout.supportCapacity)
      || air.layout.supportVectorOffsetWords % 4 !== 0
      || air.arena.size < air.layout.totalBytes
      || air.boundaryControl.size < 7 * 4)) {
      throw new RangeError("Structured fine transport air-support authority is invalid or undersized");
    }
    this.queryCapacity = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    this.bfeccEnabled = coarsePhiBFECCEnabled(source.plan.fineFactor);
    const basePlan = planFineLevelSetGPUTransport(this.queryCapacity, this.queryCapacity,
      source.plan.maximumResidentBricks);
    const bfeccScratchBytes = this.bfeccEnabled ? this.queryCapacity * 4 : 0;
    this.plan = Object.freeze({ ...basePlan,
      allocatedBytes: basePlan.allocatedBytes + bfeccScratchBytes });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const pieces = [topology.catalogVolumes, topology.catalogFaces,
      topology.catalogTetrahedronHeaders, topology.catalogTetrahedronVertices,
      topology.catalogTetrahedra] as const;
    const samplingOffsets: number[] = [0];
    let samplingBytes = structured.plan.rowCapacity * 16;
    for (const piece of pieces) { samplingOffsets.push(samplingBytes); samplingBytes += piece.size; }
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (samplingBytes > maximumBinding) throw new RangeError("Fine transport sampling catalog exceeds binding limits");
    this.samplingOffsets = Object.freeze(samplingOffsets);
    this.samplingCatalog = device.createBuffer({ label: "Fine transport row metrics and immutable interpolation catalog",
      size: samplingBytes, usage: storage });
    this.activitySnapshot = device.createBuffer({ label: "Fine transport exact page-activity snapshot",
      size: this.plan.activitySnapshotBytes, usage: storage });
    const catalogCopy = device.createCommandEncoder({ label: "Install fine transport interpolation catalog" });
    pieces.forEach((piece, index) => catalogCopy.copyBufferToBuffer(
      piece, 0, this.samplingCatalog, samplingOffsets[index + 1]!, piece.size));
    device.queue.submit([catalogCopy.finish()]);
    this.params = device.createBuffer({ label: "Structured fine transport parameters",
      size: 320,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.control = device.createBuffer({ label: "Structured fine transport control",
      size: FINE_LEVELSET_TRANSPORT_CONTROL_BYTES, usage: storage });
    this.scanBlocks = Math.ceil(source.plan.maximumResidentBricks / 256);
    // Block-scan scratch for the widened workset, status, and delta
    // publications: SCAN_BLOCK_WORDS (16) per 256-page block, one global
    // record, then one prefix word per scanned item so the scatters carry the
    // reduce dispatch's prefix instead of recomputing it. Addressed past the
    // class payload by the shader's scanBase().
    const scanScratchBytes = 4 * (16 * (this.scanBlocks + 1) + 256 * this.scanBlocks);
    this.governor = device.createBuffer({ label: "Structured fine transport GPU substep governor",
      size: 512 + this.plan.pageStatusBytes + this.plan.worksetBytes + scanScratchBytes,
      usage: storage });
    this.indirectDispatch = device.createBuffer({ label: "Structured fine transport indirect dispatch publication",
      size: 512, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.topologyDelta = { buffer: device.createBuffer({
      label: "Structured fine transport interface delta", size: this.plan.topologyDeltaBytes, usage: storage }),
      pageCapacity: source.plan.maximumResidentBricks, maximumDisplacementOffsetWords: 7,
      candidateKeysOffsetWords: 8,
      changedKeysOffsetWords: 8 + source.plan.maximumResidentBricks };
    if (this.bfeccEnabled) {
      this.reversePhi = device.createBuffer({ label: "Factor-1 bounded MacCormack reverse phi",
        size: bfeccScratchBytes, usage: storage });
    }
    const stagedFineAddressing = fineTransportStagedAddressingRequested() ? 1 : 0;
    const b4FineAddressing = fineTransportB4AddressingEnabled(source.plan) ? 1 : 0;
    this.pipelineCacheKey = JSON.stringify({ stagedFineAddressing, b4FineAddressing,
      bfeccEnabled: this.bfeccEnabled });
    this.pipelineConstants = Object.freeze({ stagedFineAddressing, b4FineAddressing });
    if (deferPipelineCompilation) return;
    let deviceCache = fineLevelSetTransportPipelineCache.get(device);
    if (!deviceCache) {
      deviceCache = new Map();
      fineLevelSetTransportPipelineCache.set(device, deviceCache);
    }
    let pipelines = deviceCache.get(this.pipelineCacheKey);
    if (!pipelines) {
      const shaderModule = device.createShaderModule({ label: "Direct structured fine level-set transport",
        code: structuredFineLevelSetTransportWGSL });
      const make = (entryPoint: string) => device.createComputePipeline({ label: entryPoint, layout: "auto",
        compute: { module: shaderModule, entryPoint, constants: {
          stagedFineAddressing: fineTransportStagedAddressingRequested() ? 1 : 0,
          b4FineAddressing: fineTransportB4AddressingEnabled(source.plan) ? 1 : 0,
        } } });
      pipelines = {
        planPipeline: make("planStructuredFineTransportSubsteps"),
        classifyPipeline: make("classifyStructuredFineTransportBlocks"),
        reduceWorksetsPipeline: make("reduceStructuredFineTransportWorksetBlocks"),
        scanWorksetsPipeline: make("scanStructuredFineTransportWorksetGroups"),
        publishWorksetsPipeline: make("publishStructuredFineTransportWorksets"),
        compactWorksetsPipeline: make("compactStructuredFineTransportWorksets"),
        // transitionSample dynamically selects trilinear or tetrahedral sampling
        // at every trajectory point. Only validity-aware common/rare variants are
        // required; page-anchor transition specialization would be unsound.
        transportPipelines: ["transportRegularCommonPhi", "transportRegularRarePhi"].map(make),
        reversePipelines: this.bfeccEnabled
          ? ["reverseRegularCommonPhi", "reverseRegularRarePhi"].map(make) : [],
        correctionPipelines: this.bfeccEnabled
          ? ["correctRegularCommonPhi", "correctRegularRarePhi"].map(make) : [],
        reduceStatusPipeline: make("reduceStructuredFineTransportStatus"),
        summarizePipeline: make("summarizeStructuredFineTransport"),
        commitPipeline: make("commitStructuredFineTransport"),
        clearDeltaPipeline: make("clearStructuredFineDelta"),
        reduceDeltaPipeline: make("reduceStructuredFineDeltaBlocks"),
        deltaPipeline: make("publishStructuredFineDelta"),
        compactDeltaPipeline: make("compactStructuredFineDelta"),
      };
      deviceCache.set(this.pipelineCacheKey, pipelines);
    }
    this.installPipelineBundle(pipelines);
  }

  private installPipelineBundle(pipelines: FineLevelSetTransportPipelineBundle): void {
    const { device, source } = this;
    const { structured, topology } = this.resources;
    const air = this.resources.airSupport;
    this.planPipeline = pipelines.planPipeline;
    this.classifyPipeline = pipelines.classifyPipeline;
    this.reduceWorksetsPipeline = pipelines.reduceWorksetsPipeline;
    this.scanWorksetsPipeline = pipelines.scanWorksetsPipeline;
    this.publishWorksetsPipeline = pipelines.publishWorksetsPipeline;
    this.compactWorksetsPipeline = pipelines.compactWorksetsPipeline;
    this.transportPipelines = pipelines.transportPipelines;
    this.reversePipelines = pipelines.reversePipelines;
    this.correctionPipelines = pipelines.correctionPipelines;
    this.reduceStatusPipeline = pipelines.reduceStatusPipeline;
    this.summarizePipeline = pipelines.summarizePipeline;
    this.commitPipeline = pipelines.commitPipeline;
    this.clearDeltaPipeline = pipelines.clearDeltaPipeline;
    this.reduceDeltaPipeline = pipelines.reduceDeltaPipeline;
    this.deltaPipeline = pipelines.deltaPipeline;
    this.compactDeltaPipeline = pipelines.compactDeltaPipeline;
    const all = new Map<number, GPUBuffer>([
      [0, this.params], [1, source.metadata], [2, source.worklist], [3, source.flags], [4, source.phi],
      // workA is persistent closest-point identity. The distance lane workB is
      // the sanctioned transport scratch: topology reconstructs its carried
      // magnitude from transported phi before any JFA consumer can observe it.
      [5, source.workB], [6, this.samplingCatalog], [7, this.control], [8, this.topologyDelta.buffer],
      [9, structured.control],
      [12, structured.rowVelocities], [13, this.governor], [14, topology.metrics],
      // Missing support remains constructible for isolated tests, but the
      // shader's enabled bit makes that state publish zero work.
      [16, this.activitySnapshot], [20, air?.arena ?? structured.rowVelocities],
      [21, air?.boundaryControl ?? structured.control],
      ...(this.reversePhi ? [[10, this.reversePhi] as const] : []),
    ]);
    const group = (pipeline: GPUComputePipeline, bindings: readonly number[]) =>
      device.createBindGroup({ label: `${pipeline.label} fine transport bindings ${bindings.join(",")}`,
        layout: pipeline.getBindGroupLayout(0), entries: bindings.map((binding) => ({
        binding, resource: { buffer: all.get(binding)! },
      })) });
    this.planGroup = group(this.planPipeline, [0,1,2,6,9,12,13,14,16,20,21]);
    this.classifyGroup = group(this.classifyPipeline, [0,1,2,3,4,13]);
    // Bindings must match what the entry point statically uses: these
    // pipelines are created with `layout: "auto"`, so the derived layout omits
    // any binding the entry point does not reference, and supplying an extra
    // one makes the whole bind group invalid. The group scan touches only the
    // governor, and the header publication no longer walks the worklist.
    this.reduceWorksetsGroup = group(this.reduceWorksetsPipeline, [0,2,9,13]);
    this.scanWorksetsGroup = group(this.scanWorksetsPipeline, [0,13]);
    this.publishWorksetsGroup = group(this.publishWorksetsPipeline, [0,9,13]);
    this.compactWorksetsGroup = group(this.compactWorksetsPipeline, [0,2,9,13]);
    this.transportGroups = this.transportPipelines.map((pipeline, index) => group(pipeline,
      FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS[index]!));
    this.reverseGroups = this.reversePipelines.map((pipeline) => group(pipeline,
      [0,1,2,3,5,6,10,12,13,20]));
    this.correctionGroups = this.correctionPipelines.map((pipeline) => group(pipeline,
      [0,1,2,3,4,5,6,10,12,13,20]));
    // Summary uses metadata only to retain the first rejected sample's stable
    // fine-lattice position; only the final reduction publishes the control.
    this.reduceStatusGroup = group(this.reduceStatusPipeline, [0,1,2,13]);
    this.summarizeGroup = group(this.summarizePipeline, [0,1,2,7,13]);
    this.commitGroup = group(this.commitPipeline, [0,1,2,3,4,5,7,8]);
    this.clearDeltaGroup = group(this.clearDeltaPipeline, [8]);
    this.reduceDeltaGroup = group(this.reduceDeltaPipeline, [0,2,8,13]);
    this.deltaGroup = group(this.deltaPipeline, [0,2,7,8,13]);
    this.compactDeltaGroup = group(this.compactDeltaPipeline, [0,2,8,13]);
  }

  private async compilePipelineBundleAsync(): Promise<FineLevelSetTransportPipelineBundle> {
    const shaderModule = this.device.createShaderModule({
      label: "Direct structured fine level-set transport",
      code: structuredFineLevelSetTransportWGSL,
    });
    const make = (entryPoint: string) => this.device.createComputePipelineAsync({
      label: entryPoint,
      layout: "auto",
      compute: { module: shaderModule, entryPoint, constants: { ...this.pipelineConstants } },
    });
    const planPipeline = await make("planStructuredFineTransportSubsteps");
    const classifyPipeline = await make("classifyStructuredFineTransportBlocks");
    const reduceWorksetsPipeline = await make("reduceStructuredFineTransportWorksetBlocks");
    const scanWorksetsPipeline = await make("scanStructuredFineTransportWorksetGroups");
    const publishWorksetsPipeline = await make("publishStructuredFineTransportWorksets");
    const compactWorksetsPipeline = await make("compactStructuredFineTransportWorksets");
    const transportPipelines: GPUComputePipeline[] = [];
    for (const entryPoint of ["transportRegularCommonPhi", "transportRegularRarePhi"]) {
      transportPipelines.push(await make(entryPoint));
    }
    const reversePipelines: GPUComputePipeline[] = [];
    const correctionPipelines: GPUComputePipeline[] = [];
    if (this.bfeccEnabled) {
      for (const entryPoint of ["reverseRegularCommonPhi", "reverseRegularRarePhi"]) {
        reversePipelines.push(await make(entryPoint));
      }
      for (const entryPoint of ["correctRegularCommonPhi", "correctRegularRarePhi"]) {
        correctionPipelines.push(await make(entryPoint));
      }
    }
    return {
      planPipeline,
      classifyPipeline,
      reduceWorksetsPipeline,
      scanWorksetsPipeline,
      publishWorksetsPipeline,
      compactWorksetsPipeline,
      transportPipelines,
      reversePipelines,
      correctionPipelines,
      reduceStatusPipeline: await make("reduceStructuredFineTransportStatus"),
      summarizePipeline: await make("summarizeStructuredFineTransport"),
      commitPipeline: await make("commitStructuredFineTransport"),
      clearDeltaPipeline: await make("clearStructuredFineDelta"),
      reduceDeltaPipeline: await make("reduceStructuredFineDeltaBlocks"),
      deltaPipeline: await make("publishStructuredFineDelta"),
      compactDeltaPipeline: await make("compactStructuredFineDelta"),
    };
  }

  initializePipelines(): Promise<void> {
    if (this.planPipeline) return Promise.resolve();
    if (this.pipelineInitialization) return this.pipelineInitialization;
    this.pipelineInitialization = (async () => {
      let deviceCache = fineLevelSetTransportPipelineCache.get(this.device);
      if (!deviceCache) {
        deviceCache = new Map();
        fineLevelSetTransportPipelineCache.set(this.device, deviceCache);
      }
      let pipelines = deviceCache.get(this.pipelineCacheKey);
      if (!pipelines) {
        let compilations = fineLevelSetTransportPipelineCompilations.get(this.device);
        if (!compilations) {
          compilations = new Map();
          fineLevelSetTransportPipelineCompilations.set(this.device, compilations);
        }
        let compilation = compilations.get(this.pipelineCacheKey);
        if (!compilation) {
          compilation = this.compilePipelineBundleAsync().then((compiled) => {
            const published = deviceCache!.get(this.pipelineCacheKey) ?? compiled;
            deviceCache!.set(this.pipelineCacheKey, published);
            return published;
          }).finally(() => { compilations!.delete(this.pipelineCacheKey); });
          compilations.set(this.pipelineCacheKey, compilation);
        }
        pipelines = await compilation;
      }
      this.installPipelineBundle(pipelines);
    })();
    return this.pipelineInitialization;
  }

  encode(broker: PassBroker, options: FineLevelSetGPUTransportOptions): PassBroker {
    if (this.destroyed) throw new Error("Fine level-set transport is destroyed");
    if (!this.planPipeline) throw new Error("Fine transport pipelines are not initialized");
    if (!Number.isFinite(options.timestep) || options.timestep < 0) {
      throw new RangeError("Fine level-set transport timestep must be finite and non-negative");
    }
    const band = options.transportBandCells ?? 0xffff;
    if (!Number.isSafeInteger(band) || band < 1 || band > 0xffff) {
      throw new RangeError("Fine level-set transport band must be a positive integer");
    }
    const { structured } = this.resources, plan = this.source.plan;
    const maximumBacktraceFineCells = options.maximumBacktraceFineCells ?? plan.fineFactor;
    if (!Number.isSafeInteger(maximumBacktraceFineCells) || maximumBacktraceFineCells < 1
      || maximumBacktraceFineCells > 2 * plan.fineFactor) {
      throw new RangeError("Fine transport displacement bound exceeds its configured support depth");
    }
    const bytes = new ArrayBuffer(320), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    u.set(plan.brickDimensions, 0); u[3] = plan.brickResolution;
    u.set(plan.sampleDimensions, 4); u[7] = plan.samplesPerBrick;
    f.set(plan.domainOrigin, 8); f[11] = plan.fineCellWidth;
    u.set([plan.maximumResidentBricks, options.generation ?? this.source.generation,
      plan.fineFactor, this.resources.maximumLeafSize], 12);
    u.set(this.resources.dimensions, 16); u[19] = structured.plan.rowCapacity;
    f[20] = this.resources.physicalCellSize; f[21] = options.timestep;
    u[22] = band; u[23] = options.boundaryPolicy === "closed-neumann" ? 1 : 0;
    u[24] = options.openTopBoundary ? 1 : 0;
    u[25] = structured.plan.maximumCaseSlots; u[26] = structured.plan.authorityWords;
    u[27] = structured.rowBankStrideWords;
    u.set(Object.values(structured.plan.offsets), 28);
    u[48] = this.resources.topology.catalogTetrahedronVertexCount ?? 0;
    u[49] = maximumBacktraceFineCells;
    u.set(this.samplingOffsets.map((offset) => offset / 4), 50);
    u[56] = this.resources.topology.rowTemplateHeaderOffsetBytes / 4;
    const air = this.resources.airSupport;
    u[50] = air?.layout.ownerDirectoryOffsetWords ?? 0;
    u.set(air ? [1, air.layout.selectorTagOffsetWords, air.layout.regularTagOffsetWords,
      air.layout.controlOffsetWords, air.layout.supportVectorOffsetWords,
      air.layout.supportCapacity, air.layout.selectorStride] : [0, 0, 0, 0, 0, 0, 0], 57);
    const inflow = options.inflow;
    if (inflow) {
      f.set([inflow.outletCenter_m.x, inflow.outletCenter_m.y,
        inflow.outletCenter_m.z, inflow.radius_m], 64);
      f.set([inflow.velocity_m_s.x, inflow.velocity_m_s.y,
        inflow.velocity_m_s.z, inflow.apertureScale], 68);
    }
    f.set([inflow?.strength ?? 0,
      fineTransportQuiescenceEnabled()
        ? FINE_LEVELSET_QUIESCENCE_DISPLACEMENT_EPSILON_CELLS : -1,
      options.dynamicBoundary ? 1 : 0, 0], 72);
    this.device.queue.writeBuffer(this.params, 0, bytes);
    const run = (pipeline: GPUComputePipeline, group: GPUBindGroup, label: string, groups: number) => {
      const pass = broker.compute({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups);
    };
    const runLiveBlocks = (pipeline: GPUComputePipeline, group: GPUBindGroup, label: string) => {
      const pass = broker.compute({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      // Governor words 43..45 are the GPU-authored ceil(livePages / 256)
      // dispatch. The allocation-time scanBlocks value sizes scratch only.
      pass.dispatchWorkgroupsIndirect(this.indirectDispatch, 43 * 4);
    };
    run(this.planPipeline, this.planGroup, "Plan GPU-resident fine transport substeps", 1);
    broker.fence("structured fine substep governor published");
    broker.updateIndirectBuffer(this.governor, 0, this.indirectDispatch, 0, 512);
    const classify = broker.compute({ label: "Classify direct structured fine transport blocks" });
    classify.setPipeline(this.classifyPipeline); classify.setBindGroup(0, this.classifyGroup);
    classify.dispatchWorkgroupsIndirect(this.indirectDispatch, 160);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after fine transport block classification");
    }
    // Classify -> block prefix -> scatter, keyed by page index so the class
    // streams keep the ascending order the retired serial loops produced.
    runLiveBlocks(this.reduceWorksetsPipeline, this.reduceWorksetsGroup,
      "Reduce direct structured fine transport workset blocks");
    run(this.scanWorksetsPipeline, this.scanWorksetsGroup,
      "Scan direct structured fine transport workset groups", 1);
    run(this.publishWorksetsPipeline, this.publishWorksetsGroup,
      "Publish direct structured fine transport worksets", 1);
    runLiveBlocks(this.compactWorksetsPipeline, this.compactWorksetsGroup,
      "Compact direct structured fine transport worksets");
    // Storage dependencies are ordered between dispatches in one compute pass.
    // End the pass only for the storage-write -> INDIRECT-only copy below.
    broker.fence("structured fine transport worksets published");
    broker.updateIndirectBuffer(this.governor, 0, this.indirectDispatch, 0, 512);
    this.censusTick(broker);
    const classNames = ["common", "rare"];
    this.transportPipelines.forEach((pipeline, index) => {
      const transport = broker.compute({ label: `Advect fine phi ${classNames[index]}` });
      transport.setPipeline(pipeline); transport.setBindGroup(0, this.transportGroups[index]!);
      transport.dispatchWorkgroupsIndirect(this.indirectDispatch,
        (4 + 7 * FINE_LEVELSET_TRANSPORT_WORKSET_CLASSES[index]! + 4) * 4);
    });
    this.reversePipelines.forEach((pipeline, index) => {
      const reverse = broker.compute({ label: `Reverse factor-1 predicted phi ${classNames[index]}` });
      reverse.setPipeline(pipeline); reverse.setBindGroup(0, this.reverseGroups[index]!);
      reverse.dispatchWorkgroupsIndirect(this.indirectDispatch,
        (4 + 7 * FINE_LEVELSET_TRANSPORT_WORKSET_CLASSES[index]! + 4) * 4);
    });
    this.correctionPipelines.forEach((pipeline, index) => {
      const correction = broker.compute({
        label: `Apply bounded factor-1 MacCormack correction ${classNames[index]}`,
      });
      correction.setPipeline(pipeline); correction.setBindGroup(0, this.correctionGroups[index]!);
      correction.dispatchWorkgroupsIndirect(this.indirectDispatch,
        (4 + 7 * FINE_LEVELSET_TRANSPORT_WORKSET_CLASSES[index]! + 4) * 4);
    });
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after fine characteristic transport");
    }
    runLiveBlocks(this.reduceStatusPipeline, this.reduceStatusGroup,
      "Reduce structured fine transport status blocks");
    run(this.summarizePipeline, this.summarizeGroup, "Publish structured fine transport status", 1);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after fine transport summary");
    }
    const commit = broker.compute({ label: "Commit structured fine phi and phase delta" });
    commit.setPipeline(this.commitPipeline); commit.setBindGroup(0, this.commitGroup);
    commit.dispatchWorkgroupsIndirect(this.indirectDispatch, 160);
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after fine transport commit");
    }
    run(this.clearDeltaPipeline, this.clearDeltaGroup,
      "Clear structured fine transport delta header", 1);
    runLiveBlocks(this.reduceDeltaPipeline, this.reduceDeltaGroup,
      "Reduce structured fine transport delta blocks");
    run(this.deltaPipeline, this.deltaGroup, "Publish structured fine transport delta", 1);
    runLiveBlocks(this.compactDeltaPipeline, this.compactDeltaGroup,
      "Compact structured fine topology delta");
    return broker;
  }

  /** Copy this frame's published class headers, decode the previous frame's.
   * The two-frame split is required for the same reason as the structured
   * dynamics census: an in-frame `mapAsync` resolves against a queue serial
   * that predates this encoder's submission. */
  private censusTick(broker: PassBroker): void {
    if (!this.censusEnabled || this.destroyed) return;
    this.censusStep += 1;
    const staging = this.censusStaging ??= this.device.createBuffer({
      label: "Structured fine transport workset census staging",
      size: 57 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    if (this.censusPhase === "copied") {
      this.censusPhase = "mapping";
      const step = this.censusStep;
      void staging.mapAsync(GPUMapMode.READ).then(() => {
        const words = [...new Uint32Array(staging.getMappedRange())];
        staging.unmap();
        this.censusPhase = "idle";
        // Governor layout: word 1 is the accepted substep count, then four
        // seven-word class headers from word 4 -- see HEADER_BASE/HEADER_WORDS
        // in `webgpu-octree-fine-levelset-transport.wgsl.ts`. Class order is
        // regular-common, transition-common, regular-rare, transition-rare.
        console.log(JSON.stringify({ phase: "fine-transport-workset-census", step,
          flags: words[0] ?? 0, substeps: words[1] ?? 0,
          pagesByClass: [0, 1, 2, 3].map((cls) => words[4 + 7 * cls + 1] ?? 0),
          validByClass: [0, 1, 2, 3].map((cls) => words[4 + 7 * cls + 3] ?? 0),
          samplesPerBrick: this.source.plan.samplesPerBrick,
          sleep: { priorSnapshot: words[46] ?? 0, priorExactRepairs: words[47] ?? 0,
            active: words[50] ?? 0, blockers: words[52] ?? 0,
            displacementFineCells: new Float32Array(new Uint32Array([words[53] ?? 0]).buffer)[0],
            rows: words[54] ?? 0, supports: words[55] ?? 0, pages: words[56] ?? 0 },
        }));
      }).catch(() => { this.censusPhase = "idle"; });
      return;
    }
    if (this.censusPhase !== "idle") return;
    broker.copyBufferToBuffer(this.governor, 0, staging, 0, 57 * 4);
    this.censusPhase = "copied";
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.control.destroy(); this.governor.destroy(); this.indirectDispatch.destroy();
    this.samplingCatalog.destroy();
    this.activitySnapshot.destroy();
    this.reversePhi?.destroy();
    this.topologyDelta.buffer.destroy();
  }
}
