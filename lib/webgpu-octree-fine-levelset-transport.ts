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

export const FINE_LEVELSET_TRANSPORT_CONTROL_BYTES = 64;
export const FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP = 4_096;
export const FINE_LEVELSET_TRANSPORT_MAXIMUM_ENCODED_SUBSTEPS = 64;
/** Binding zero is uniform; every remaining entry is storage. The dense owner
 * directory coalesces the former row-geometry lookup into air support, leaving
 * transition classes one buffer below Dawn's portable ten-buffer ceiling. */
export const FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS = Object.freeze([
  Object.freeze([0,1,2,3,4,5,6,12,13,20]),
  Object.freeze([0,1,2,3,4,5,6,12,13,20]),
] as const);
const FINE_LEVELSET_TRANSPORT_WORKSET_CLASSES = Object.freeze([0, 2] as const);

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
  generation?: number;
  transportBandCells?: number;
  /** Fine-cell closure proven resident by the destination topology publication. */
  maximumBacktraceFineCells?: number;
  boundaryPolicy?: FineLevelSetGPUBoundaryPolicy;
  openTopBoundary?: boolean;
}

export interface FineLevelSetGPUTransportPlan {
  readonly queryCapacity: number;
  readonly velocityChunkCapacity: number;
  readonly chunkCount: 1;
  readonly topologyDeltaBytes: number;
  readonly pageStatusBytes: number;
  readonly worksetBytes: number;
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
  const topologyDeltaBytes = (8 + 2 * pages) * 4;
  // Three words per page: classification uses {class, work-id}; transport
  // reuses them as two packed u16 counter pairs plus exact displacement.
  const pageStatusBytes = pages * 12;
  const worksetBytes = 128 + pages * 4;
  return { queryCapacity, velocityChunkCapacity: queryCapacity, chunkCount: 1,
    topologyDeltaBytes, pageStatusBytes, worksetBytes,
    controlBytes: FINE_LEVELSET_TRANSPORT_CONTROL_BYTES,
    allocatedBytes: 256 + pageStatusBytes + worksetBytes + topologyDeltaBytes
      + FINE_LEVELSET_TRANSPORT_CONTROL_BYTES + 2 * 512 };
}

export function planFineLevelSetGPUTransportPasses(
  _plan: Pick<FineLevelSetGPUTransportPlan, "chunkCount">,
  _segmentCount: 4 | 8,
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
  private readonly planPipeline: GPUComputePipeline;
  private readonly classifyPipeline: GPUComputePipeline;
  private readonly reduceWorksetsPipeline: GPUComputePipeline;
  private readonly scanWorksetsPipeline: GPUComputePipeline;
  private readonly publishWorksetsPipeline: GPUComputePipeline;
  private readonly compactWorksetsPipeline: GPUComputePipeline;
  private readonly transportPipelines: readonly GPUComputePipeline[];
  private readonly reduceStatusPipeline: GPUComputePipeline;
  private readonly summarizePipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;
  private readonly clearDeltaPipeline: GPUComputePipeline;
  private readonly reduceDeltaPipeline: GPUComputePipeline;
  private readonly deltaPipeline: GPUComputePipeline;
  private readonly compactDeltaPipeline: GPUComputePipeline;
  private readonly samplingCatalog: GPUBuffer;
  private readonly samplingOffsets: readonly number[];
  private readonly planGroup: GPUBindGroup;
  private readonly classifyGroup: GPUBindGroup;
  private readonly reduceWorksetsGroup: GPUBindGroup;
  private readonly scanWorksetsGroup: GPUBindGroup;
  private readonly publishWorksetsGroup: GPUBindGroup;
  private readonly compactWorksetsGroup: GPUBindGroup;
  private readonly transportGroups: readonly GPUBindGroup[];
  private readonly reduceStatusGroup: GPUBindGroup;
  private readonly summarizeGroup: GPUBindGroup;
  private readonly commitGroup: GPUBindGroup;
  private readonly clearDeltaGroup: GPUBindGroup;
  private readonly reduceDeltaGroup: GPUBindGroup;
  private readonly deltaGroup: GPUBindGroup;
  private readonly compactDeltaGroup: GPUBindGroup;
  /** 256-page blocks the widened publication trio classifies and scatters. */
  private readonly scanBlocks: number;
  /** 256-word blocks covering the interface delta's cleared header and stream. */
  private readonly deltaClearBlocks: number;
  /** Diagnostic-only per-class page-count census; see `censusTick`. */
  private readonly censusEnabled = fineTransportWorksetCensusEnabled();
  private censusStaging?: GPUBuffer;
  private censusPhase: "idle" | "copied" | "mapping" = "idle";
  private censusStep = 0;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource,
    private readonly resources: StructuredFineTransportResources) {
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
      || air.layout.supportVectorOffsetWords % 4 !== 0
      || air.arena.size < air.layout.totalBytes
      || air.boundaryControl.size < 7 * 4)) {
      throw new RangeError("Structured fine transport air-support authority is invalid or undersized");
    }
    this.queryCapacity = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    this.plan = planFineLevelSetGPUTransport(this.queryCapacity, this.queryCapacity,
      source.plan.maximumResidentBricks);
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
    const catalogCopy = device.createCommandEncoder({ label: "Install fine transport interpolation catalog" });
    pieces.forEach((piece, index) => catalogCopy.copyBufferToBuffer(
      piece, 0, this.samplingCatalog, samplingOffsets[index + 1]!, piece.size));
    device.queue.submit([catalogCopy.finish()]);
    this.params = device.createBuffer({ label: "Structured fine transport parameters",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.control = device.createBuffer({ label: "Structured fine transport control",
      size: FINE_LEVELSET_TRANSPORT_CONTROL_BYTES, usage: storage });
    this.scanBlocks = Math.ceil(source.plan.maximumResidentBricks / 256);
    this.deltaClearBlocks = Math.ceil((8 + 2 * source.plan.maximumResidentBricks) / 256);
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
      pageCapacity: source.plan.maximumResidentBricks, candidateKeysOffsetWords: 8,
      changedKeysOffsetWords: 8 + source.plan.maximumResidentBricks };
    const module = device.createShaderModule({ label: "Direct structured fine level-set transport",
      code: structuredFineLevelSetTransportWGSL });
    const make = (entryPoint: string) => device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module, entryPoint } });
    this.planPipeline = make("planStructuredFineTransportSubsteps");
    this.classifyPipeline = make("classifyStructuredFineTransportBlocks");
    this.reduceWorksetsPipeline = make("reduceStructuredFineTransportWorksetBlocks");
    this.scanWorksetsPipeline = make("scanStructuredFineTransportWorksetGroups");
    this.publishWorksetsPipeline = make("publishStructuredFineTransportWorksets");
    this.compactWorksetsPipeline = make("compactStructuredFineTransportWorksets");
    // transitionSample dynamically selects trilinear or tetrahedral sampling
    // at every trajectory point. Only validity-aware common/rare variants are
    // required; page-anchor transition specialization would be unsound.
    this.transportPipelines = ["transportRegularCommonPhi", "transportRegularRarePhi"].map(make);
    this.reduceStatusPipeline = make("reduceStructuredFineTransportStatus");
    this.summarizePipeline = make("summarizeStructuredFineTransport");
    this.commitPipeline = make("commitStructuredFineTransport");
    this.clearDeltaPipeline = make("clearStructuredFineDelta");
    this.reduceDeltaPipeline = make("reduceStructuredFineDeltaBlocks");
    this.deltaPipeline = make("publishStructuredFineDelta");
    this.compactDeltaPipeline = make("compactStructuredFineDelta");
    const all = new Map<number, GPUBuffer>([
      [0, this.params], [1, source.metadata], [2, source.worklist], [3, source.flags], [4, source.phi],
      [5, source.workA], [6, this.samplingCatalog], [7, this.control], [8, this.topologyDelta.buffer],
      [9, structured.control],
      [12, structured.rowVelocities], [13, this.governor], [14, topology.metrics],
      // Missing support remains constructible for isolated tests, but the
      // shader's enabled bit makes that state publish zero work.
      [20, air?.arena ?? structured.rowVelocities],
      [21, air?.boundaryControl ?? structured.control],
    ]);
    const group = (pipeline: GPUComputePipeline, bindings: readonly number[]) =>
      device.createBindGroup({ label: `${pipeline.label} fine transport bindings ${bindings.join(",")}`,
        layout: pipeline.getBindGroupLayout(0), entries: bindings.map((binding) => ({
        binding, resource: { buffer: all.get(binding)! },
      })) });
    this.planGroup = group(this.planPipeline, [0,2,6,9,12,13,14,20,21]);
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
    // Summary uses metadata only to retain the first rejected sample's stable
    // fine-lattice position; only the final reduction publishes the control.
    this.reduceStatusGroup = group(this.reduceStatusPipeline, [0,1,2,13]);
    this.summarizeGroup = group(this.summarizePipeline, [0,1,2,7,13]);
    this.commitGroup = group(this.commitPipeline, [0,1,2,3,4,5,7,8]);
    this.clearDeltaGroup = group(this.clearDeltaPipeline, [0,8]);
    this.reduceDeltaGroup = group(this.reduceDeltaPipeline, [0,2,8,13]);
    this.deltaGroup = group(this.deltaPipeline, [0,2,7,8,13]);
    this.compactDeltaGroup = group(this.compactDeltaPipeline, [0,2,8,13]);
  }

  encode(broker: PassBroker, options: FineLevelSetGPUTransportOptions): PassBroker {
    if (this.destroyed) throw new Error("Fine level-set transport is destroyed");
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
    const bytes = new ArrayBuffer(256), u = new Uint32Array(bytes), f = new Float32Array(bytes);
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
    this.device.queue.writeBuffer(this.params, 0, bytes);
    const run = (pipeline: GPUComputePipeline, group: GPUBindGroup, label: string, groups: number) => {
      const pass = broker.compute({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups);
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
    run(this.reduceWorksetsPipeline, this.reduceWorksetsGroup,
      "Reduce direct structured fine transport workset blocks", this.scanBlocks);
    run(this.scanWorksetsPipeline, this.scanWorksetsGroup,
      "Scan direct structured fine transport workset groups", 1);
    run(this.publishWorksetsPipeline, this.publishWorksetsGroup,
      "Publish direct structured fine transport worksets", 1);
    run(this.compactWorksetsPipeline, this.compactWorksetsGroup,
      "Compact direct structured fine transport worksets", this.scanBlocks);
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
    if (octreeAlgorithmDiagnosticsEnabled()) {
      broker.fence("algorithm diagnostic after fine characteristic transport");
    }
    run(this.reduceStatusPipeline, this.reduceStatusGroup,
      "Reduce structured fine transport status blocks", this.scanBlocks);
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
      "Clear structured fine transport delta", this.deltaClearBlocks);
    run(this.reduceDeltaPipeline, this.reduceDeltaGroup,
      "Reduce structured fine transport delta blocks", this.scanBlocks);
    run(this.deltaPipeline, this.deltaGroup, "Publish structured fine transport delta", 1);
    run(this.compactDeltaPipeline, this.compactDeltaGroup,
      "Compact structured fine topology delta", this.scanBlocks);
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
      size: 128, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
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
        }));
      }).catch(() => { this.censusPhase = "idle"; });
      return;
    }
    if (this.censusPhase !== "idle") return;
    broker.copyBufferToBuffer(this.governor, 0, staging, 0, 128);
    this.censusPhase = "copied";
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.control.destroy(); this.governor.destroy(); this.indirectDispatch.destroy();
    this.samplingCatalog.destroy();
    this.topologyDelta.buffer.destroy();
  }
}
