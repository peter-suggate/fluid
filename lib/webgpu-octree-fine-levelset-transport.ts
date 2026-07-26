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
  private readonly publishWorksetsPipeline: GPUComputePipeline;
  private readonly transportPipelines: readonly GPUComputePipeline[];
  private readonly summarizePipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;
  private readonly deltaPipeline: GPUComputePipeline;
  private readonly samplingCatalog: GPUBuffer;
  private readonly samplingOffsets: readonly number[];
  private readonly planGroup: GPUBindGroup;
  private readonly classifyGroup: GPUBindGroup;
  private readonly publishWorksetsGroup: GPUBindGroup;
  private readonly transportGroups: readonly GPUBindGroup[];
  private readonly summarizeGroup: GPUBindGroup;
  private readonly commitGroup: GPUBindGroup;
  private readonly deltaGroup: GPUBindGroup;
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
    this.governor = device.createBuffer({ label: "Structured fine transport GPU substep governor",
      size: 512 + this.plan.pageStatusBytes + this.plan.worksetBytes,
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
    this.publishWorksetsPipeline = make("publishStructuredFineTransportWorksets");
    // transitionSample dynamically selects trilinear or tetrahedral sampling
    // at every trajectory point. Only validity-aware common/rare variants are
    // required; page-anchor transition specialization would be unsound.
    this.transportPipelines = ["transportRegularCommonPhi", "transportRegularRarePhi"].map(make);
    this.summarizePipeline = make("summarizeStructuredFineTransport");
    this.commitPipeline = make("commitStructuredFineTransport");
    this.deltaPipeline = make("publishStructuredFineDelta");
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
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bindings.map((binding) => ({
        binding, resource: { buffer: all.get(binding)! },
      })) });
    this.planGroup = group(this.planPipeline, [0,2,6,9,12,13,14,20,21]);
    this.classifyGroup = group(this.classifyPipeline, [0,1,2,3,4,13]);
    this.publishWorksetsGroup = group(this.publishWorksetsPipeline, [0,2,9,13]);
    this.transportGroups = this.transportPipelines.map((pipeline, index) => group(pipeline,
      FINE_LEVELSET_TRANSPORT_CLASS_BINDINGS[index]!));
    this.summarizeGroup = group(this.summarizePipeline, [0,2,7,13]);
    this.commitGroup = group(this.commitPipeline, [0,1,2,3,4,5,7,8]);
    this.deltaGroup = group(this.deltaPipeline, [0,2,7,8]);
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
      || maximumBacktraceFineCells > plan.fineFactor) {
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
    run(this.publishWorksetsPipeline, this.publishWorksetsGroup,
      "Publish direct structured fine transport worksets", 1);
    // Storage dependencies are ordered between dispatches in one compute pass.
    // End the pass only for the storage-write -> INDIRECT-only copy below.
    broker.fence("structured fine transport worksets published");
    broker.updateIndirectBuffer(this.governor, 0, this.indirectDispatch, 0, 512);
    const classNames = ["common", "rare"];
    this.transportPipelines.forEach((pipeline, index) => {
      const transport = broker.compute({ label: `Advect fine phi ${classNames[index]}` });
      transport.setPipeline(pipeline); transport.setBindGroup(0, this.transportGroups[index]!);
      transport.dispatchWorkgroupsIndirect(this.indirectDispatch,
        (4 + 7 * FINE_LEVELSET_TRANSPORT_WORKSET_CLASSES[index]! + 4) * 4);
    });
    run(this.summarizePipeline, this.summarizeGroup, "Publish structured fine transport status", 1);
    const commit = broker.compute({ label: "Commit structured fine phi and phase delta" });
    commit.setPipeline(this.commitPipeline); commit.setBindGroup(0, this.commitGroup);
    commit.dispatchWorkgroupsIndirect(this.indirectDispatch, 160);
    run(this.deltaPipeline, this.deltaGroup, "Compact structured fine topology delta", 1);
    return broker;
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.control.destroy(); this.governor.destroy(); this.indirectDispatch.destroy();
    this.samplingCatalog.destroy();
    this.topologyDelta.buffer.destroy();
  }
}
