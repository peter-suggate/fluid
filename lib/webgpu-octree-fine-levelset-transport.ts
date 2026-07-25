import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL } from "./webgpu-fine-levelset-dispatch";
import type {
  WebGPUOctreePowerVelocityPrepass,
} from "./webgpu-octree-power-velocity-prepass";
import type {
  WebGPUOctreeFaceClosestPointExtension,
} from "./webgpu-octree-face-closest-point";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import {
  makeFineLevelSetProductionFusedTransportWGSL,
  type FineTransportPackedSamplerPlan,
} from "./webgpu-octree-fine-levelset-fused-transport";
import type { PassBroker } from "./webgpu-pass-broker";

export const FINE_LEVELSET_TRANSPORT_CONTROL_BYTES = 64;
/** Each summary workgroup reduces this many trajectory outcomes before a
 * deterministic second-level chunk reduction. */
export const FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP = 4_096;

export interface FineLevelSetGPUTransportControl {
  departureOutsideBand: number;
  nonfiniteVelocity: number;
  processed: number;
  committed: boolean;
  extrapolatedVelocity: number;
  maximumDisplacementFineCells: number;
  faceBandUnavailable: number;
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
  headers: GPUBuffer;
  rowVelocities: GPUBuffer;
  dimensions: readonly [number, number, number];
  physicalCellSize: number;
  maximumLeafSize: number;
  /** Adaptive owner authority used to resolve Section 5 regular-face rows. */
  ownerTopology: GPUBuffer;
  /** Current catalog/Delaunay authority used by the Section 5 point field. */
  powerTopology: OctreePowerTopologySource;
  generation?: number;
  /** Destination band transported this step; outer valid samples are interpolation support only. */
  transportBandCells?: number;
  /** Closed boundaries extend cell-centred phi constantly through the in-domain wall half-cell. */
  boundaryPolicy?: FineLevelSetGPUBoundaryPolicy;
  /** Open ceilings extend the top air field normally for outflow characteristics. */
  openTopBoundary?: boolean;
}

export interface FineLevelSetGPUTransportPlan {
  readonly queryCapacity: number;
  readonly velocityChunkCapacity: number;
  readonly positionCapacity: number;
  readonly positionBytes: number;
  readonly outcomeBytes: number;
  readonly summaryBytes: number;
  readonly partialSummaryBytes: number;
  readonly partialSummaryGroupsPerChunk: number;
  readonly dispatchMetadataBytes: number;
  readonly dispatchArgumentsOffsetBytes: number;
  readonly indirectBytes: number;
  /** Compact producer-owned interface-membership delta consumed by recurring topology. */
  readonly topologyDeltaBytes: number;
  readonly chunkCount: number;
  readonly chunkParameterStride: number;
  readonly chunkParameterBytes: number;
  readonly controlBytes: number;
  readonly allocatedBytes: number;
}

export interface FineLevelSetGPUTransportPassPlan {
  readonly chunkCount: number;
  readonly segmentCount: number;
  readonly passesPerSegment: number;
  readonly passesPerChunk: number;
  readonly encodedPasses: number;
}

export function planFineLevelSetGPUTransport(queryCapacity: number,
  velocityChunkCapacity: number, pageCapacity = 0): FineLevelSetGPUTransportPlan {
  if (!Number.isSafeInteger(queryCapacity) || queryCapacity < 1
    || !Number.isSafeInteger(velocityChunkCapacity) || velocityChunkCapacity < 1
    || !Number.isSafeInteger(pageCapacity) || pageCapacity < 0) {
    throw new RangeError("Fine transport capacities must be positive integers");
  }
  const positionCapacity = velocityChunkCapacity;
  const positionBytes = positionCapacity * 16;
  const outcomeBytes = positionCapacity * 8;
  const chunkCount = Math.ceil(queryCapacity / velocityChunkCapacity);
  const summaryBytes = chunkCount * FINE_LEVELSET_TRANSPORT_CONTROL_BYTES;
  const partialSummaryGroupsPerChunk = Math.ceil(
    velocityChunkCapacity / FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP,
  );
  const partialSummaryBytes = chunkCount * partialSummaryGroupsPerChunk
    * FINE_LEVELSET_TRANSPORT_CONTROL_BYTES;
  const dispatchArgumentsOffsetBytes = (8 + chunkCount) * 4;
  // Trace/summary records are followed by the sample commit and one page
  // commit record. The latter lets the producer publish one phase mask per
  // resident page without a catalog-sized host dispatch.
  const indirectBytes = (2 * chunkCount + 2) * 12;
  const dispatchMetadataBytes = dispatchArgumentsOffsetBytes + indirectBytes;
  const topologyDeltaBytes = pageCapacity === 0 ? 0 : (8 + 2 * pageCapacity) * 4;
  const chunkParameterStride = 128, chunkParameterBytes = chunkCount * chunkParameterStride;
  return { queryCapacity, velocityChunkCapacity, positionCapacity, positionBytes, outcomeBytes, summaryBytes,
    partialSummaryBytes, partialSummaryGroupsPerChunk,
    dispatchMetadataBytes, dispatchArgumentsOffsetBytes, indirectBytes, topologyDeltaBytes,
    chunkCount, chunkParameterStride, chunkParameterBytes,
    controlBytes: FINE_LEVELSET_TRANSPORT_CONTROL_BYTES,
    allocatedBytes: positionBytes + outcomeBytes + summaryBytes + partialSummaryBytes + chunkParameterBytes
      + dispatchMetadataBytes + indirectBytes + topologyDeltaBytes
      + 16 + FINE_LEVELSET_TRANSPORT_CONTROL_BYTES };
}

/** Producer-owned, zero-atomic interface-membership delta.
 *
 * Header words are count, generation, validity, active page count, followed
 * by one xyz indirect record. Candidate keys occupy `8..8+capacity`; compact
 * sorted keys occupy the following `capacity` words.
 */
export interface FineLevelSetTransportTopologyDelta {
  readonly buffer: GPUBuffer;
  readonly pageCapacity: number;
  readonly candidateKeysOffsetWords: 8;
  readonly changedKeysOffsetWords: number;
}

/**
 * Static command-count contract for Section 5's piecewise-linear trace.
 * Production packs the minimum portable direct/air sampler publications, then
 * one dense kernel performs every m sample and final phi interpolation. The
 * owner-page publication is consumed directly; outcome reduction/publication
 * remains dense and atomic-free.
 */
export function planFineLevelSetGPUTransportPasses(
  plan: Pick<FineLevelSetGPUTransportPlan, "chunkCount">,
  segmentCount: 4 | 8,
): FineLevelSetGPUTransportPassPlan {
  if (!Number.isSafeInteger(plan.chunkCount) || plan.chunkCount < 1) {
    throw new RangeError("Fine transport pass plan requires at least one chunk");
  }
  const passesPerSegment = 0;
  const passesPerChunk = 2;
  return {
    chunkCount: plan.chunkCount,
    segmentCount,
    passesPerSegment,
    passesPerChunk,
    encodedPasses: 1 + plan.chunkCount * passesPerChunk + 2,
  };
}

export function unpackFineLevelSetGPUTransportControl(words: ArrayLike<number>): FineLevelSetGPUTransportControl {
  if (words.length < 8) throw new RangeError("Fine transport control needs eight words");
  return { departureOutsideBand: Number(words[0]) >>> 0, nonfiniteVelocity: Number(words[1]) >>> 0,
    processed: Number(words[2]) >>> 0, committed: Number(words[3]) !== 0,
    extrapolatedVelocity: Number(words[4]) >>> 0, maximumDisplacementFineCells: Number(words[5]) >>> 0,
    faceBandUnavailable: Number(words[6]) >>> 0, velocityUnavailable: Number(words[7]) >>> 0,
    ...(words.length >= 12 ? { invalidVelocityStatus: Number(words[8]) >>> 0,
      nonpositiveVelocityResult: Number(words[9]) >>> 0,
      velocityStatusReasonOr: Number(words[10]) >>> 0,
      firstInvalidVelocityStatus: Number(words[11]) >>> 0 } : {}),
    ...(words.length >= 16 ? { firstInvalidVelocityLocalIndex: Number(words[12]) >>> 0,
      firstInvalidVelocityPosition: [new Float32Array(new Uint32Array([Number(words[13]) >>> 0]).buffer)[0],
        new Float32Array(new Uint32Array([Number(words[14]) >>> 0]).buffer)[0],
        new Float32Array(new Uint32Array([Number(words[15]) >>> 0]).buffer)[0]] as const } : {}) };
}

/**
 * Paper section 5 transport on the global uniform fine lattice.  Trajectory
 * positions stay GPU-resident and Stage-B octree velocity is queried again
 * before every one of the m=fineFactor piecewise-linear segments.
 */
export class WebGPUFineLevelSetTransport {
  readonly control: GPUBuffer;
  readonly queryCapacity: number;
  readonly plan: FineLevelSetGPUTransportPlan;
  readonly topologyDelta: FineLevelSetTransportTopologyDelta;
  private readonly positions: GPUBuffer;
  private readonly outcomes: GPUBuffer;
  private readonly chunkSummaries: GPUBuffer;
  private readonly partialSummaries: GPUBuffer;
  private readonly dispatchMetadata: GPUBuffer;
  private readonly indirectDispatch: GPUBuffer;
  private readonly dispatchParams: GPUBuffer;
  private readonly prepareDispatchPipeline: GPUComputePipeline;
  private readonly summarizePipeline: GPUComputePipeline;
  private readonly finalizeSummaryPipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;
  private readonly publishTopologyDeltaPipeline: GPUComputePipeline;
  private readonly fused: {
    readonly plan: FineTransportPackedSamplerPlan;
    readonly directAuthority: GPUBuffer;
    readonly airAuthority: GPUBuffer;
    readonly params: readonly GPUBuffer[];
    readonly pipeline: GPUComputePipeline;
  };
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly source: WebGPUFineLevelSetBrickSource,
    private readonly velocityPrepass: Pick<WebGPUOctreePowerVelocityPrepass,
      "encodeRowDescriptors" | "prepareFusedSampling" | "encodeDirectoryCount"
      | "encodeFusedAuthority" | "fusedSamplerSource" | "source">,
    /** Paper Section 5 face-band velocity authority. Stage B remains the
     * primary liquid interpolant; positive-air samples and exact local-catalog
     * coverage misses are completed from closest-point-extended regular octree faces. */
    private readonly faceBand: Pick<WebGPUOctreeFaceClosestPointExtension,
      "prepareFusedAirSampling" | "encodeFusedAuthority" | "fusedSamplerSource">,
  ) {
    this.queryCapacity = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    if (velocityPrepass.source.queryCapacity < this.queryCapacity
      && (velocityPrepass.source.queryCapacity * 16) % device.limits.minStorageBufferOffsetAlignment !== 0) {
      throw new RangeError("Fine transport velocity chunk must satisfy storage-buffer offset alignment");
    }
    this.plan = planFineLevelSetGPUTransport(this.queryCapacity, velocityPrepass.source.queryCapacity,
      source.plan.maximumResidentBricks);
    this.positions = device.createBuffer({ label: "fine-levelset trajectory positions",
      size: this.plan.positionBytes, usage: GPUBufferUsage.STORAGE });
    this.outcomes = device.createBuffer({ label: "fine-levelset trajectory outcomes",
      size: this.plan.outcomeBytes, usage: GPUBufferUsage.STORAGE });
    this.chunkSummaries = device.createBuffer({ label: "fine-levelset transport chunk summaries",
      size: this.plan.summaryBytes, usage: GPUBufferUsage.STORAGE });
    this.partialSummaries = device.createBuffer({ label: "fine-levelset transport parallel summary partials",
      size: this.plan.partialSummaryBytes, usage: GPUBufferUsage.STORAGE });
    this.dispatchMetadata = device.createBuffer({ label: "fine-levelset validated live dispatch publication",
      size: this.plan.dispatchMetadataBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.indirectDispatch = device.createBuffer({ label: "fine-levelset immutable live dispatches",
      size: this.plan.indirectBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.dispatchParams = device.createBuffer({ label: "fine-levelset live dispatch parameters",
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.dispatchParams, 0, new Uint32Array([
      this.plan.velocityChunkCapacity, this.plan.chunkCount,
      device.limits.maxComputeWorkgroupsPerDimension, this.plan.dispatchArgumentsOffsetBytes / 4,
    ]));
    this.control = device.createBuffer({ label: "fine-levelset transport control",
      size: FINE_LEVELSET_TRANSPORT_CONTROL_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.topologyDelta = {
      buffer: device.createBuffer({ label: "fine transport compact interface-membership delta",
        size: this.plan.topologyDeltaBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      pageCapacity: source.plan.maximumResidentBricks,
      candidateKeysOffsetWords: 8,
      changedKeysOffsetWords: 8 + source.plan.maximumResidentBricks,
    };
    const shaderModule = device.createShaderModule({ label: "fine-levelset fused transport publication",
      code: fineLevelSetFusedTransportPublicationWGSL });
    const pipeline = (entryPoint: string, label: string) => device.createComputePipeline({ label, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.prepareDispatchPipeline = pipeline("prepareFineTransportDispatch",
      "Publish validated fine transport dispatches");
    this.summarizePipeline = pipeline("summarizeFineTransportChunk", "Summarize fine transport chunk");
    this.finalizeSummaryPipeline = pipeline("finalizeFineTransportChunkSummary",
      "Finalize fine transport chunk summary");
    this.publishPipeline = pipeline("publishFineTransport", "Publish fine transport status");
    this.commitPipeline = pipeline("commitFineTransport", "Commit fine transport");
    this.publishTopologyDeltaPipeline = pipeline("publishFineTransportTopologyDelta",
      "Publish compact fine transport topology delta");
    {
      const direct = velocityPrepass.fusedSamplerSource, air = faceBand.fusedSamplerSource;
      const packed: FineTransportPackedSamplerPlan = {
        direct: direct.regions, air: air.regions,
        directBytes: direct.authority.size, airBytes: air.authority.size,
      };
      const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
      if (packed.directBytes > maximumBinding || packed.airBytes > maximumBinding) {
        throw new RangeError("Fine fused characteristic sampler exceeds the portable packed-binding limit");
      }
      const fusedParams = Array.from({ length: this.plan.chunkCount }, (_, chunk) => device.createBuffer({
        label: `Fine fused characteristic parameters ${chunk + 1}/${this.plan.chunkCount}`, size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
      const fusedModule = device.createShaderModule({ label: "Fused production fine characteristic",
        code: makeFineLevelSetProductionFusedTransportWGSL() });
      this.fused = { plan: packed, directAuthority: direct.authority, airAuthority: air.authority, params: fusedParams,
        pipeline: device.createComputePipeline({ label: "Trace and sample complete fine characteristic",
          layout: "auto", compute: { module: fusedModule, entryPoint: "transportFineCharacteristic" } }) };
    }
  }

  encode(initialBroker: PassBroker, options: FineLevelSetGPUTransportOptions): PassBroker {
    if (this.destroyed) throw new Error("Fine level-set transport is destroyed");
    if (!Number.isFinite(options.timestep) || options.timestep < 0) {
      throw new RangeError("Fine level-set transport timestep must be finite and non-negative");
    }
    const transportBandCells = options.transportBandCells ?? 0xffff;
    if (!Number.isSafeInteger(transportBandCells) || transportBandCells < 1 || transportBandCells > 0xffff) {
      throw new RangeError("Fine level-set transport band must be a positive integer");
    }
    const closedDomainBoundary = options.boundaryPolicy === "closed-neumann" ? 1 : 0;
    this.device.queue.writeBuffer(this.source.params, 76, new Float32Array([options.timestep]));
    const binding = (buffer: GPUBuffer): GPUBufferBinding => ({ buffer });
    const broker = initialBroker;
    const run = (pipeline: GPUComputePipeline, entries: readonly GPUBindGroupEntry[], label: string,
      workgroups: number) => {
      const pass = broker.compute({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(workgroups);
    };
    const runIndirect = (pipeline: GPUComputePipeline, entries: readonly GPUBindGroupEntry[], label: string,
      indirectOffset: number) => {
      const pass = broker.compute({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroupsIndirect(this.indirectDispatch, indirectOffset);
    };
    const prepassOptions = { dimensions: options.dimensions, physicalCellSize: options.physicalCellSize,
      maximumLeafSize: options.maximumLeafSize, queryCount: this.queryCapacity,
      generation: options.generation };
    this.velocityPrepass.encodeRowDescriptors(broker, options.headers);
    const chunkCapacity = this.velocityPrepass.source.queryCapacity;
    {
      const direct = this.velocityPrepass.fusedSamplerSource;
      const air = this.faceBand.fusedSamplerSource;
      if (options.rowVelocities.size > this.fused.plan.direct.velocities.sizeBytes
        || options.ownerTopology.size > this.device.limits.maxStorageBufferBindingSize) {
        throw new RangeError("Fine fused characteristic authority exceeds its immutable packed plan");
      }
      this.velocityPrepass.prepareFusedSampling(prepassOptions, chunkCapacity);
      this.velocityPrepass.encodeDirectoryCount(broker);
      this.faceBand.prepareFusedAirSampling({
        dimensions: options.dimensions, maximumLeafSize: options.maximumLeafSize,
        queryCount: chunkCapacity, physicalCellSize: options.physicalCellSize,
        owners: options.ownerTopology,
        // Transport consumes the face band retained for this live source
        // generation; the destination fine generation is allocated only after
        // the characteristic transaction has completed.
        fineGeneration: this.source.generation,
        powerTopology: options.powerTopology,
      });
      // Keep authority publication and characteristic work in distinct GPU
      // passes. Besides making timestamp attribution honest, this gives Metal
      // an explicit producer/consumer boundary for the packed storage arenas.
      broker.compute({ label: "Publish grouped Stage-B and face-band transport authorities" });
      this.velocityPrepass.encodeFusedAuthority(broker, options.rowVelocities);
      this.faceBand.encodeFusedAuthority(broker);
      broker.fence("grouped transport authorities published");
    }
    run(this.prepareDispatchPipeline, [
      { binding: 0, resource: binding(this.source.params) },
      { binding: 2, resource: binding(this.source.metadata) },
      { binding: 3, resource: binding(this.source.worklist) },
      { binding: 14, resource: binding(this.dispatchMetadata) },
      { binding: 16, resource: binding(this.dispatchParams) },
    ], "Validate and publish live global fine dispatches", 1);
    broker.fence("fine transport dispatches validated");
    // The resident narrow band is normally within a few percent of its fixed
    // allocation (4,024 of 4,096 pages in the UI dam). Replaying its compact
    // GPU-authored indirect records added a pass boundary and, on the current
    // recurring path, could publish a valid zero-work transport. Dispatch the
    // immutable per-chunk/page capacities instead; activeSample and the
    // validated metadata prefix retain exact fail-closed bounds in the shader.
    for (let chunk = 0; chunk < this.plan.chunkCount; chunk += 1) {
      {
        const a = this.fused.plan.direct, b = this.fused.plan.air;
        const data = new ArrayBuffer(128), words = new Uint32Array(data), floats = new Float32Array(data);
        words.set([
          a.rows.offsetBytes / 4, a.rows.sizeBytes / 16,
          // Retired duplicate direct row-directory slots remain zero in the
          // shared 128-byte ABI. Identity comes from the face-band directory.
          0, 0,
          a.velocities.offsetBytes / 4, options.rowVelocities.size / 16,
          a.tetraHeaders.offsetBytes / 4, a.tetraHeaders.sizeBytes / 12,
          a.tetraVertices.offsetBytes / 4, a.tetraVertices.sizeBytes / 16,
          a.tetrahedra.offsetBytes / 4, a.tetrahedra.sizeBytes / 4,
          b.control.offsetBytes / 4,
          b.rows.offsetBytes / 4, b.rows.sizeBytes / 32,
          b.rowDirectory.offsetBytes / 4, b.rowDirectory.sizeBytes / 4,
          b.velocities.offsetBytes / 4, b.velocities.sizeBytes / 16,
          b.metrics.offsetBytes / 4, b.metrics.sizeBytes / 16,
          b.transition.offsetBytes / 4, b.point.offsetBytes / 4,
          chunk * this.plan.velocityChunkCapacity, closedDomainBoundary,
          options.openTopBoundary ? 1 : 0,
        ]);
        // Pack words 23..26 are chunkBase, closed, open-top, and band
        // distance. Keep the float after both boundary-policy words.
        floats[26] = transportBandCells * this.source.plan.fineCellWidth;
        this.device.queue.writeBuffer(this.fused.params[chunk], 0, data);
        run(this.fused.pipeline, [
          { binding: 0, resource: binding(this.source.params) },
          { binding: 2, resource: binding(this.source.metadata) },
          { binding: 3, resource: binding(this.source.worklist) },
          { binding: 4, resource: binding(this.source.flags) },
          { binding: 5, resource: binding(this.source.phi) },
          { binding: 6, resource: binding(this.source.workA) },
          { binding: 10, resource: binding(this.positions) },
          { binding: 12, resource: binding(this.outcomes) },
          { binding: 13, resource: binding(this.fused.directAuthority) },
          { binding: 14, resource: binding(this.fused.airAuthority) },
          { binding: 15, resource: binding(this.fused.params[chunk]) },
          { binding: 16, resource: binding(this.velocityPrepass.fusedSamplerSource.params) },
          { binding: 17, resource: binding(this.faceBand.fusedSamplerSource.params) },
          { binding: 18, resource: binding(this.faceBand.fusedSamplerSource.sampleParams) },
          { binding: 19, resource: binding(options.ownerTopology) },
        ], `Trace and sample global fine characteristic ${chunk + 1}/${this.plan.chunkCount}`,
        Math.ceil(this.plan.velocityChunkCapacity / 64));
        broker.fence("fine characteristic traced");
        run(this.summarizePipeline, [
          { binding: 0, resource: binding(this.source.params) },
          { binding: 10, resource: binding(this.positions) }, { binding: 12, resource: binding(this.outcomes) },
          { binding: 14, resource: binding(this.dispatchMetadata) },
          { binding: 15, resource: binding(this.fused.params[chunk]) },
          { binding: 18, resource: binding(this.partialSummaries) },
        ], `Summarize global fine departure chunk ${chunk + 1}/${this.plan.chunkCount}`,
        this.plan.partialSummaryGroupsPerChunk);
        run(this.finalizeSummaryPipeline, [
          { binding: 13, resource: binding(this.chunkSummaries) },
          { binding: 14, resource: binding(this.dispatchMetadata) },
          { binding: 15, resource: binding(this.fused.params[chunk]) },
          { binding: 18, resource: binding(this.partialSummaries) },
        ], `Finalize global fine departure chunk ${chunk + 1}/${this.plan.chunkCount}`, 1);
        broker.fence("fine departure chunk summarized");
      }
    }
    run(this.publishPipeline, [{ binding: 0, resource: binding(this.source.params) },
      { binding: 7, resource: binding(this.control) },
      { binding: 13, resource: binding(this.chunkSummaries) },
      { binding: 14, resource: binding(this.dispatchMetadata) }],
      "Publish global fine transport", 1);
    broker.fence("fine transport status published");
    run(this.commitPipeline, [
      { binding: 0, resource: binding(this.source.params) },
      { binding: 2, resource: binding(this.source.metadata) },
      { binding: 3, resource: binding(this.source.worklist) },
      { binding: 4, resource: binding(this.source.flags) },
      { binding: 5, resource: binding(this.source.phi) }, { binding: 6, resource: binding(this.source.workA) },
      { binding: 7, resource: binding(this.control) },
      { binding: 17, resource: binding(this.topologyDelta.buffer) }],
    "Commit global fine transport phase masks", this.source.plan.maximumResidentBricks);
    broker.fence("fine transport phase masks committed");
    run(this.publishTopologyDeltaPipeline, [
      { binding: 0, resource: binding(this.source.params) },
      { binding: 2, resource: binding(this.source.metadata) },
      { binding: 3, resource: binding(this.source.worklist) },
      { binding: 7, resource: binding(this.control) },
      { binding: 17, resource: binding(this.topologyDelta.buffer) },
    ], "Publish compact global fine transport topology delta", 1);
    return broker;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.positions.destroy();
    this.outcomes.destroy();
    this.chunkSummaries.destroy();
    this.partialSummaries.destroy();
    this.dispatchMetadata.destroy();
    this.indirectDispatch.destroy();
    this.dispatchParams.destroy();
    this.control.destroy();
    this.topologyDelta.buffer.destroy();
    this.fused.params.forEach((buffer) => buffer.destroy());
  }
}

export const fineLevelSetFusedTransportPublicationWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;
struct Params{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,
 activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
struct Control{departureOutsideBand:u32,nonfiniteVelocity:u32,processed:u32,committed:u32,extrapolatedVelocity:u32,maximumDisplacementFineCells:u32,faceBandUnavailable:u32,velocityUnavailable:u32,invalidVelocityStatus:u32,nonpositiveVelocityResult:u32,velocityStatusReasonOr:u32,firstInvalidVelocityStatus:u32,firstInvalidVelocityLocalIndex:u32,firstInvalidVelocityX:u32,firstInvalidVelocityY:u32,firstInvalidVelocityZ:u32}
struct Pack{directRowOffset:u32,directRowCount:u32,directDirectoryOffset:u32,directDirectoryCount:u32,
 directVelocityOffset:u32,directVelocityCount:u32,tetraHeaderOffset:u32,tetraHeaderCount:u32,
 tetraVertexOffset:u32,tetraVertexCount:u32,tetrahedronOffset:u32,tetrahedronCount:u32,
 bandControlOffset:u32,bandRowOffset:u32,bandRowCount:u32,bandDirectoryOffset:u32,bandDirectoryWordCount:u32,
 bandVelocityOffset:u32,bandVelocityCount:u32,bandMetricOffset:u32,bandMetricCount:u32,
 transitionOffset:u32,pointOffset:u32,chunkBase:u32,closedDomainBoundary:u32,
 openTopBoundary:u32,transportBandDistance:f32,pad0:u32,pad1:u32,pad2:u32,pad3:u32,pad4:u32}
struct DispatchP{chunkCapacity:u32,chunkCount:u32,maxDispatchX:u32,argsWordOffset:u32}
@group(0)@binding(0)var<uniform>params:Params;
@group(0)@binding(2)var<storage,read_write>metadata:array<u32>;@group(0)@binding(3)var<storage,read>worklist:array<u32>;
@group(0)@binding(4)var<storage,read>sampleFlags:array<u32>;@group(0)@binding(5)var<storage,read_write>phi:array<f32>;
@group(0)@binding(6)var<storage,read_write>workA:array<f32>;@group(0)@binding(7)var<storage,read_write>control:Control;
@group(0)@binding(10)var<storage,read_write>positions:array<vec4f>;
@group(0)@binding(12)var<storage,read_write>outcomes:array<vec2u>;
@group(0)@binding(13)var<storage,read_write>chunkSummaries:array<u32>;
@group(0)@binding(14)var<storage,read_write>dispatchMetadata:array<u32>;
@group(0)@binding(15)var<uniform>pack:Pack;
@group(0)@binding(16)var<uniform>dispatchP:DispatchP;
@group(0)@binding(17)var<storage,read_write>topologyDelta:array<u32>;
@group(0)@binding(18)var<storage,read_write>partialSummaries:array<u32>;
const DEPARTURE:u32=1u;const NONFINITE:u32=2u;const PROCESSED:u32=4u;const FACE_UNAVAILABLE:u32=8u;const VELOCITY_UNAVAILABLE:u32=16u;const INVALID_STATUS:u32=32u;const NONPOSITIVE:u32=64u;
const PAGE_INTERFACE:u32=2u;const PAGE_PHASE_DIRTY:u32=4u;
fn finiteScalar(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn outcomeFlags(v:u32)->u32{return v&255u;}fn outcomeExtrapolated(v:u32)->u32{return(v>>8u)&15u;}fn outcomeDisplacement(v:u32)->u32{return v>>12u;}fn packOutcome(flags:u32,extrapolated:u32,displacement:u32)->u32{return(flags&255u)|((extrapolated&15u)<<8u)|(min(displacement,0xfffffu)<<12u);}
fn activeSample(flat:u32)->vec2u{if(arrayLength(&worklist)<5u||worklist[1]!=params.generation||worklist[3]!=1u||worklist[4]!=1u||worklist[0]>params.pageCapacity||5u+worklist[0]>arrayLength(&worklist)){return vec2u(INVALID);}let count=worklist[0];if(flat>=count*params.samplesPerBrick){return vec2u(INVALID);}
 let w=flat/params.samplesPerBrick;let local=flat-w*params.samplesPerBrick;let id=worklist[5u+w];
 if(id>=params.pageCapacity||id*10u+9u>=arrayLength(&metadata)||metadata[id*10u]!=id||metadata[id*10u+2u]!=params.generation){return vec2u(INVALID);}return vec2u(id,local);}
var<workgroup>dispatchFailure:array<u32,64>;
fn writeDispatch(record:u32,items:u32){let base=dispatchP.argsWordOffset+record*3u;if(items==0u){dispatchMetadata[base]=0u;dispatchMetadata[base+1u]=0u;dispatchMetadata[base+2u]=0u;return;}let groups=(items+63u)/64u;let x=min(groups,dispatchP.maxDispatchX);dispatchMetadata[base]=x;dispatchMetadata[base+1u]=(groups+x-1u)/x;dispatchMetadata[base+2u]=1u;}
@compute @workgroup_size(64)fn prepareFineTransportDispatch(@builtin(local_invocation_index)lid:u32){
 let headerValid=arrayLength(&worklist)>=5u&&params.samplesPerBrick>0u&&dispatchP.chunkCapacity>0u&&dispatchP.chunkCount>0u&&dispatchP.maxDispatchX>0u&&worklist[1]==params.generation&&worklist[3]==1u&&worklist[4]==1u&&worklist[0]<=params.pageCapacity&&worklist[0]<=params.worklistCapacity&&5u+worklist[0]<=arrayLength(&worklist)&&worklist[0]<=INVALID/params.samplesPerBrick;
 let count=select(0u,worklist[0],headerValid);var failure=select(1u,0u,headerValid);for(var i=lid;i<count;i+=64u){let id=worklist[5u+i];if(id>=params.pageCapacity||id*10u+9u>=arrayLength(&metadata)||metadata[id*10u]!=id||metadata[id*10u+2u]!=params.generation){failure=1u;}}
 dispatchFailure[lid]=failure;workgroupBarrier();var stride=32u;loop{if(stride==0u){break;}if(lid<stride){dispatchFailure[lid]|=dispatchFailure[lid+stride];}workgroupBarrier();stride/=2u;}let valid=dispatchFailure[0]==0u;let live=select(0u,count*params.samplesPerBrick,valid);let capacity=dispatchP.chunkCapacity*dispatchP.chunkCount;if(lid==0u){dispatchMetadata[0]=select(0u,1u,valid&&live<=capacity);dispatchMetadata[1]=params.generation;dispatchMetadata[2]=select(0u,live,valid&&live<=capacity);dispatchMetadata[3]=select(0u,(live+dispatchP.chunkCapacity-1u)/dispatchP.chunkCapacity,valid&&live<=capacity);dispatchMetadata[4]=dispatchP.chunkCapacity;dispatchMetadata[5]=dispatchP.chunkCount;dispatchMetadata[6]=select(1u,0u,valid&&live<=capacity);dispatchMetadata[7]=0u;}workgroupBarrier();
 let published=dispatchMetadata[0]==1u;for(var chunk=lid;chunk<dispatchP.chunkCount;chunk+=64u){let base=chunk*dispatchP.chunkCapacity;let chunkLive=select(0u,min(dispatchP.chunkCapacity,live-base),published&&base<live);dispatchMetadata[8u+chunk]=chunkLive;writeDispatch(chunk,chunkLive);let summaryRecord=dispatchP.chunkCount+chunk;if(chunkLive==0u){writeDispatch(summaryRecord,0u);}else{let at=dispatchP.argsWordOffset+summaryRecord*3u;dispatchMetadata[at]=1u;dispatchMetadata[at+1u]=1u;dispatchMetadata[at+2u]=1u;}}if(lid==0u){writeDispatch(2u*dispatchP.chunkCount,select(0u,live,published));let pageRecord=2u*dispatchP.chunkCount+1u;let pages=select(0u,count,published);let at=dispatchP.argsWordOffset+pageRecord*3u;let x=min(pages,dispatchP.maxDispatchX);dispatchMetadata[at]=x;dispatchMetadata[at+1u]=select(1u,(pages+x-1u)/x,x>0u);dispatchMetadata[at+2u]=1u;}}
var<workgroup>s0:array<u32,64>;var<workgroup>s1:array<u32,64>;var<workgroup>s2:array<u32,64>;var<workgroup>s4:array<u32,64>;var<workgroup>s5:array<u32,64>;var<workgroup>s6:array<u32,64>;var<workgroup>s7:array<u32,64>;var<workgroup>s8:array<u32,64>;var<workgroup>s9:array<u32,64>;var<workgroup>s10:array<u32,64>;var<workgroup>s11:array<u32,64>;var<workgroup>s12:array<u32,64>;var<workgroup>s13:array<u32,64>;var<workgroup>s14:array<u32,64>;var<workgroup>s15:array<u32,64>;
fn reduceLane(lid:u32){var stride=32u;loop{if(stride==0u){break;}if(lid<stride){s0[lid]+=s0[lid+stride];s1[lid]+=s1[lid+stride];s2[lid]+=s2[lid+stride];s4[lid]+=s4[lid+stride];s5[lid]=max(s5[lid],s5[lid+stride]);s6[lid]+=s6[lid+stride];s7[lid]+=s7[lid+stride];s8[lid]+=s8[lid+stride];s9[lid]+=s9[lid+stride];s10[lid]|=s10[lid+stride];if(s12[lid+stride]<s12[lid]){s11[lid]=s11[lid+stride];s12[lid]=s12[lid+stride];s13[lid]=s13[lid+stride];s14[lid]=s14[lid+stride];s15[lid]=s15[lid+stride];}}workgroupBarrier();stride/=2u;}}
fn initializeSummaryLane(lid:u32,departure:u32,nonfinite:u32,processed:u32,extrapolated:u32,displacement:u32,faceUnavailable:u32,velocityUnavailable:u32,invalidStatus:u32,nonpositive:u32,reasonOr:u32,firstStatus:u32,firstIndex:u32,x:u32,y:u32,z:u32){s0[lid]=departure;s1[lid]=nonfinite;s2[lid]=processed;s4[lid]=extrapolated;s5[lid]=displacement;s6[lid]=faceUnavailable;s7[lid]=velocityUnavailable;s8[lid]=invalidStatus;s9[lid]=nonpositive;s10[lid]=reasonOr;s11[lid]=firstStatus;s12[lid]=firstIndex;s13[lid]=x;s14[lid]=y;s15[lid]=z;}
@compute @workgroup_size(64)fn summarizeFineTransportChunk(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){var departure=0u;var nonfinite=0u;var processed=0u;var extrapolated=0u;var displacement=0u;var faceUnavailable=0u;var velocityUnavailable=0u;var invalidStatus=0u;var nonpositive=0u;var reasonOr=0u;var firstStatus=INVALID;var firstIndex=INVALID;var firstX=0u;var firstY=0u;var firstZ=0u;let chunk=pack.chunkBase/dispatchMetadata[4];let live=select(0u,dispatchMetadata[8u+chunk],dispatchMetadata[0]==1u&&dispatchMetadata[1]==params.generation&&chunk<dispatchMetadata[3]);let groups=(dispatchMetadata[4]+${FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP - 1}u)/${FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP}u;let begin=wg.x*${FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP}u;let end=min(begin+${FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP}u,live);for(var i=begin+lid;i<end;i+=64u){let outcome=outcomes[i];let flags=outcomeFlags(outcome.x);departure+=select(0u,1u,(flags&DEPARTURE)!=0u);nonfinite+=select(0u,1u,(flags&NONFINITE)!=0u);processed+=select(0u,1u,(flags&PROCESSED)!=0u);extrapolated+=outcomeExtrapolated(outcome.x);displacement=max(displacement,outcomeDisplacement(outcome.x));faceUnavailable+=select(0u,1u,(flags&FACE_UNAVAILABLE)!=0u);velocityUnavailable+=select(0u,1u,(flags&VELOCITY_UNAVAILABLE)!=0u);invalidStatus+=select(0u,1u,(flags&INVALID_STATUS)!=0u);nonpositive+=select(0u,1u,(flags&NONPOSITIVE)!=0u);if((flags&INVALID_STATUS)!=0u){reasonOr|=outcome.y;}if((flags&(INVALID_STATUS|DEPARTURE))!=0u&&pack.chunkBase+i<firstIndex){firstIndex=pack.chunkBase+i;firstStatus=outcome.y;firstX=bitcast<u32>(positions[i].x);firstY=bitcast<u32>(positions[i].y);firstZ=bitcast<u32>(positions[i].z);}}initializeSummaryLane(lid,departure,nonfinite,processed,extrapolated,displacement,faceUnavailable,velocityUnavailable,invalidStatus,nonpositive,reasonOr,firstStatus,firstIndex,firstX,firstY,firstZ);workgroupBarrier();reduceLane(lid);if(lid==0u&&wg.x<groups){let base=(chunk*groups+wg.x)*16u;if(base+15u<arrayLength(&partialSummaries)){partialSummaries[base]=s0[0];partialSummaries[base+1u]=s1[0];partialSummaries[base+2u]=s2[0];partialSummaries[base+3u]=0u;partialSummaries[base+4u]=s4[0];partialSummaries[base+5u]=s5[0];partialSummaries[base+6u]=s6[0];partialSummaries[base+7u]=s7[0];partialSummaries[base+8u]=s8[0];partialSummaries[base+9u]=s9[0];partialSummaries[base+10u]=s10[0];partialSummaries[base+11u]=s11[0];partialSummaries[base+12u]=s12[0];partialSummaries[base+13u]=s13[0];partialSummaries[base+14u]=s14[0];partialSummaries[base+15u]=s15[0];}}}
@compute @workgroup_size(64)fn finalizeFineTransportChunkSummary(@builtin(local_invocation_index)lid:u32){var departure=0u;var nonfinite=0u;var processed=0u;var extrapolated=0u;var displacement=0u;var faceUnavailable=0u;var velocityUnavailable=0u;var invalidStatus=0u;var nonpositive=0u;var reasonOr=0u;var firstStatus=INVALID;var firstIndex=INVALID;var firstX=0u;var firstY=0u;var firstZ=0u;let chunk=pack.chunkBase/dispatchMetadata[4];let groups=(dispatchMetadata[4]+${FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP - 1}u)/${FINE_LEVELSET_TRANSPORT_SUMMARY_ITEMS_PER_WORKGROUP}u;for(var group=lid;group<groups;group+=64u){let base=(chunk*groups+group)*16u;if(base+15u>=arrayLength(&partialSummaries)){continue;}departure+=partialSummaries[base];nonfinite+=partialSummaries[base+1u];processed+=partialSummaries[base+2u];extrapolated+=partialSummaries[base+4u];displacement=max(displacement,partialSummaries[base+5u]);faceUnavailable+=partialSummaries[base+6u];velocityUnavailable+=partialSummaries[base+7u];invalidStatus+=partialSummaries[base+8u];nonpositive+=partialSummaries[base+9u];reasonOr|=partialSummaries[base+10u];if(partialSummaries[base+12u]<firstIndex){firstStatus=partialSummaries[base+11u];firstIndex=partialSummaries[base+12u];firstX=partialSummaries[base+13u];firstY=partialSummaries[base+14u];firstZ=partialSummaries[base+15u];}}initializeSummaryLane(lid,departure,nonfinite,processed,extrapolated,displacement,faceUnavailable,velocityUnavailable,invalidStatus,nonpositive,reasonOr,firstStatus,firstIndex,firstX,firstY,firstZ);workgroupBarrier();reduceLane(lid);if(lid==0u){let base=chunk*16u;if(base+15u<arrayLength(&chunkSummaries)){chunkSummaries[base]=s0[0];chunkSummaries[base+1u]=s1[0];chunkSummaries[base+2u]=s2[0];chunkSummaries[base+3u]=0u;chunkSummaries[base+4u]=s4[0];chunkSummaries[base+5u]=s5[0];chunkSummaries[base+6u]=s6[0];chunkSummaries[base+7u]=s7[0];chunkSummaries[base+8u]=s8[0];chunkSummaries[base+9u]=s9[0];chunkSummaries[base+10u]=s10[0];chunkSummaries[base+11u]=s11[0];chunkSummaries[base+12u]=s12[0];chunkSummaries[base+13u]=s13[0];chunkSummaries[base+14u]=s14[0];chunkSummaries[base+15u]=s15[0];}}}
@compute @workgroup_size(64)fn publishFineTransport(@builtin(local_invocation_index)lid:u32){var departure=0u;var nonfinite=0u;var processed=0u;var extrapolated=0u;var displacement=0u;var faceUnavailable=0u;var velocityUnavailable=0u;var invalidStatus=0u;var nonpositive=0u;var reasonOr=0u;var firstStatus=INVALID;var firstIndex=INVALID;var firstX=0u;var firstY=0u;var firstZ=0u;let dispatchValid=dispatchMetadata[0]==1u&&dispatchMetadata[1]==params.generation;let count=select(0u,min(dispatchMetadata[3],arrayLength(&chunkSummaries)/16u),dispatchValid);for(var i=lid;i<count;i+=64u){let base=i*16u;departure+=chunkSummaries[base];nonfinite+=chunkSummaries[base+1u];processed+=chunkSummaries[base+2u];extrapolated+=chunkSummaries[base+4u];displacement=max(displacement,chunkSummaries[base+5u]);faceUnavailable+=chunkSummaries[base+6u];velocityUnavailable+=chunkSummaries[base+7u];invalidStatus+=chunkSummaries[base+8u];nonpositive+=chunkSummaries[base+9u];reasonOr|=chunkSummaries[base+10u];if(chunkSummaries[base+12u]<firstIndex){firstStatus=chunkSummaries[base+11u];firstIndex=chunkSummaries[base+12u];firstX=chunkSummaries[base+13u];firstY=chunkSummaries[base+14u];firstZ=chunkSummaries[base+15u];}}initializeSummaryLane(lid,departure,nonfinite,processed,extrapolated,displacement,faceUnavailable,velocityUnavailable,invalidStatus,nonpositive,reasonOr,firstStatus,firstIndex,firstX,firstY,firstZ);workgroupBarrier();reduceLane(lid);if(lid==0u){control.departureOutsideBand=s0[0];control.nonfiniteVelocity=s1[0];control.processed=s2[0];control.committed=select(0u,1u,dispatchValid&&s0[0]==0u&&s1[0]==0u&&s7[0]==0u);control.extrapolatedVelocity=s4[0];control.maximumDisplacementFineCells=s5[0];control.faceBandUnavailable=s6[0];control.velocityUnavailable=s7[0];control.invalidVelocityStatus=s8[0];control.nonpositiveVelocityResult=s9[0];control.velocityStatusReasonOr=s10[0];control.firstInvalidVelocityStatus=s11[0];control.firstInvalidVelocityLocalIndex=s12[0];control.firstInvalidVelocityX=s13[0];control.firstInvalidVelocityY=s14[0];control.firstInvalidVelocityZ=s15[0];}}
var<workgroup>pageChanged:array<u32,64>;var<workgroup>pageOldInterface:array<u32,64>;var<workgroup>pageInterface:array<u32,64>;var<workgroup>pageNeighbors:array<u32,6>;
fn localCoord(local:u32)->vec3u{let x=local%params.brickResolution;let yz=local/params.brickResolution;return vec3u(x,yz%params.brickResolution,yz/params.brickResolution);}
fn localIndex(q:vec3u)->u32{return q.x+params.brickResolution*(q.y+params.brickResolution*q.z);}
fn neighborSample(id:u32,local:u32,q:vec3u,direction:u32)->u32{var inside=true;var next=q;if(direction==0u){inside=q.x>0u;next.x=select(params.brickResolution-1u,q.x-1u,inside);}else if(direction==1u){inside=q.x+1u<params.brickResolution;next.x=select(0u,q.x+1u,inside);}else if(direction==2u){inside=q.y>0u;next.y=select(params.brickResolution-1u,q.y-1u,inside);}else if(direction==3u){inside=q.y+1u<params.brickResolution;next.y=select(0u,q.y+1u,inside);}else if(direction==4u){inside=q.z>0u;next.z=select(params.brickResolution-1u,q.z-1u,inside);}else{inside=q.z+1u<params.brickResolution;next.z=select(0u,q.z+1u,inside);}if(inside){return id*params.samplesPerBrick+localIndex(next);}let neighbor=pageNeighbors[direction];if(neighbor==INVALID){return INVALID;}return neighbor*params.samplesPerBrick+localIndex(next);}
@compute @workgroup_size(64)fn commitFineTransport(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let work=fineLinearWorkgroup(w,n);var pageLive=control.committed!=0u&&work<worklist[0];var changed=0u;var hadInterface=0u;var hasInterface=0u;var id=INVALID;if(pageLive){id=worklist[5u+work];if(id>=params.pageCapacity||metadata[id*10u+2u]!=params.generation){pageLive=false;}}if(lid<6u){var neighbor=INVALID;if(pageLive){let candidate=metadata[id*10u+4u+lid];if(candidate<params.pageCapacity&&metadata[candidate*10u+2u]==params.generation){neighbor=candidate;}}pageNeighbors[lid]=neighbor;}workgroupBarrier();if(pageLive){for(var local=lid;local<params.samplesPerBrick;local+=64u){let index=id*params.samplesPerBrick+local;let valid=(sampleFlags[index]&VALID)!=0u;let oldValue=phi[index];let newValue=workA[index];if(valid&&finiteScalar(oldValue)&&finiteScalar(newValue)){changed|=select(0u,1u,(oldValue<0.0)!=(newValue<0.0));let q=localCoord(local);for(var direction=0u;direction<6u;direction+=1u){let neighbor=neighborSample(id,local,q,direction);if(neighbor!=INVALID&&(sampleFlags[neighbor]&VALID)!=0u){let oldOther=phi[neighbor];let other=workA[neighbor];hadInterface|=select(0u,1u,finiteScalar(oldOther)&&(oldOther<0.0)!=(oldValue<0.0));hasInterface|=select(0u,1u,finiteScalar(other)&&(other<0.0)!=(newValue<0.0));}}}else if(valid){changed=1u;} }}pageChanged[lid]=changed;pageOldInterface[lid]=hadInterface;pageInterface[lid]=hasInterface;workgroupBarrier();var width=32u;loop{if(width==0u){break;}if(lid<width){pageChanged[lid]|=pageChanged[lid+width];pageOldInterface[lid]|=pageOldInterface[lid+width];pageInterface[lid]|=pageInterface[lid+width];}workgroupBarrier();width>>=1u;}if(pageLive&&lid==0u){let previous=metadata[id*10u+3u];let interfaceBefore=(previous&PAGE_INTERFACE)!=0u||pageOldInterface[0]!=0u;let interfaceNow=pageInterface[0]!=0u;
// The fine summary provides the exact cell-centre phase required by
// Aanjaneya et al. section 5. Retain every old/current interface page and
// sample-phase transition, then expand only its bounded dirty halo.
let membershipChanged=pageChanged[0]!=0u||interfaceBefore||interfaceNow;metadata[id*10u+3u]=VALID|select(0u,PAGE_INTERFACE,interfaceNow)|select(0u,PAGE_PHASE_DIRTY,membershipChanged);topologyDelta[8u+work]=select(INVALID,metadata[id*10u+1u],membershipChanged);}workgroupBarrier();if(pageLive){for(var local=lid;local<params.samplesPerBrick;local+=64u){let index=id*params.samplesPerBrick+local;if((sampleFlags[index]&VALID)!=0u){phi[index]=workA[index];}}}}
var<workgroup>deltaCounts:array<u32,256>;var<workgroup>deltaMaximumKeys:array<u32,256>;var<workgroup>deltaFirstKeys:array<u32,256>;var<workgroup>deltaFailures:array<u32,256>;
@compute @workgroup_size(256)fn publishFineTransportTopologyDelta(@builtin(local_invocation_index)lid:u32){
  let capacity=params.pageCapacity;
  if(lid==0u){topologyDelta[0]=0u;topologyDelta[1]=params.generation;topologyDelta[2]=0u;
    topologyDelta[3]=0u;topologyDelta[4]=0u;topologyDelta[5]=1u;topologyDelta[6]=1u;topologyDelta[7]=0u;}
  workgroupBarrier();
  let headerValid=control.committed!=0u&&arrayLength(&worklist)>=5u
    &&worklist[1]==params.generation&&worklist[0]<=capacity
    &&5u+worklist[0]<=arrayLength(&worklist)&&arrayLength(&topologyDelta)>=8u+2u*capacity;
  let livePages=select(0u,worklist[0],headerValid);
  let chunk=(livePages+255u)/256u;let firstWork=lid*chunk;
  let lastWork=min(firstWork+chunk,livePages);var count=0u;var firstKey=INVALID;
  var previousKey=0u;var maximumKey=0u;var failure=select(1u,0u,headerValid);
  for(var work=firstWork;work<lastWork;work+=1u){let id=worklist[5u+work];
    if(id>=capacity||id*10u+2u>=arrayLength(&metadata)||metadata[id*10u+2u]!=params.generation){
      failure=1u;continue;}
    let candidate=topologyDelta[8u+work];if(candidate!=INVALID){
      if(count>0u&&candidate<=previousKey){failure=1u;}if(count==0u){firstKey=candidate;}
      previousKey=candidate;maximumKey=max(maximumKey,candidate);count+=1u;}}
  deltaCounts[lid]=count;deltaFirstKeys[lid]=firstKey;
  deltaMaximumKeys[lid]=maximumKey;deltaFailures[lid]=failure;
  for(var width=1u;width<256u;width<<=1u){workgroupBarrier();var addCount=0u;
    var priorMaximum=0u;if(lid>=width){addCount=deltaCounts[lid-width];
      priorMaximum=deltaMaximumKeys[lid-width];}workgroupBarrier();
    deltaCounts[lid]+=addCount;deltaMaximumKeys[lid]=max(deltaMaximumKeys[lid],priorMaximum);}
  workgroupBarrier();
  if(lid>0u&&deltaCounts[lid-1u]>0u&&firstKey!=INVALID&&deltaMaximumKeys[lid-1u]>=firstKey){
    deltaFailures[lid]=1u;}
  for(var width=128u;width>0u;width>>=1u){workgroupBarrier();
    if(lid<width){deltaFailures[lid]|=deltaFailures[lid+width];}}
  workgroupBarrier();
  var output=0u;if(lid>0u){output=deltaCounts[lid-1u];}
  if(deltaFailures[0]==0u){for(var work=firstWork;work<lastWork;work+=1u){
    let candidate=topologyDelta[8u+work];if(candidate!=INVALID){
      topologyDelta[8u+capacity+output]=candidate;output+=1u;}}}
  workgroupBarrier();
  if(lid==0u&&deltaFailures[0]==0u){let total=deltaCounts[255u];topologyDelta[0]=total;
    topologyDelta[2]=1u;topologyDelta[3]=livePages;topologyDelta[4]=total;}
}
`;
