import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { PassBroker } from "./webgpu-pass-broker";
import { octreeLosassoFineTransportWGSL } from "./webgpu-octree-losasso-fine-transport.wgsl";
import type { WebGPUOctreeLosassoVelocitySamplerSource } from "./webgpu-octree-losasso-velocity-sampler";
import type { SurfaceInflowState } from "./webgpu-quadtree-builder";

export const OCTREE_LOSASSO_FINE_TRANSPORT_CONTROL_BYTES = 64;
export const OCTREE_LOSASSO_FINE_TRANSPORT_DISPATCH_BYTES = 16;

export type OctreeLosassoFineTransportBoundaryPolicy = "strict" | "closed-neumann";

export interface OctreeLosassoFineTransportOptions {
  readonly timestep: number;
  /** Authored nozzle source applied directly to every recurring fine-phi generation. */
  readonly inflow?: SurfaceInflowState;
  readonly generation?: number;
  readonly velocityEpoch?: number;
  readonly transportBandCells?: number;
  readonly maximumBacktraceFineCells?: number;
  readonly boundaryPolicy?: OctreeLosassoFineTransportBoundaryPolicy;
  readonly openTopBoundary?: boolean;
}

export interface OctreeLosassoFineTransportPlan {
  readonly pageCapacity: number;
  readonly sampleCapacity: number;
  readonly encodedDispatchCount: 4;
  readonly topologyDeltaBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreeLosassoFineTransportTopologyDelta {
  readonly buffer: GPUBuffer;
  readonly pageCapacity: number;
  readonly maximumDisplacementOffsetWords: 7;
  readonly candidateKeysOffsetWords: 8;
  readonly changedKeysOffsetWords: number;
}

export function planOctreeLosassoFineTransport(
  source: Pick<WebGPUFineLevelSetBrickSource, "plan">,
): OctreeLosassoFineTransportPlan {
  const pages = source.plan.maximumResidentBricks;
  if (!Number.isSafeInteger(pages) || pages < 1 || Math.ceil(pages / 64) > 65_535) {
    throw new RangeError("Losasso fine transport page capacity exceeds its dispatch shape");
  }
  const topologyDeltaBytes = (8 + 3 * pages) * 4;
  return Object.freeze({ pageCapacity: pages,
    sampleCapacity: pages * source.plan.samplesPerBrick,
    encodedDispatchCount: 4 as const, topologyDeltaBytes,
    allocatedBytes: 160 + OCTREE_LOSASSO_FINE_TRANSPORT_CONTROL_BYTES
      + OCTREE_LOSASSO_FINE_TRANSPORT_DISPATCH_BYTES + topologyDeltaBytes });
}

/**
 * Losasso-only sparse fine-phi transport. Its layout has ten storage bindings:
 * five shared fine lifecycle buffers, four reduced axis-face sampler buffers,
 * and the topology delta. No Power topology, catalogue, tetrahedron, selector,
 * or air-support resource can be supplied to this constructor.
 */
export class WebGPUOctreeLosassoFineTransport {
  readonly plan: OctreeLosassoFineTransportPlan;
  readonly control: GPUBuffer;
  readonly topologyDelta: OctreeLosassoFineTransportTopologyDelta;
  readonly bindingCount = 11;
  readonly storageBindingCount = 10;

  private readonly params: GPUBuffer;
  private readonly liveDispatch: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly prepareLayout: GPUBindGroupLayout;
  private readonly preparePipelineLayout: GPUPipelineLayout;
  private prepare?: GPUComputePipeline;
  private advect?: GPUComputePipeline;
  private commit?: GPUComputePipeline;
  private finalize?: GPUComputePipeline;
  private group?: GPUBindGroup;
  private prepareGroup?: GPUBindGroup;
  private initialization?: Promise<void>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    readonly source: WebGPUFineLevelSetBrickSource,
    readonly velocity: WebGPUOctreeLosassoVelocitySamplerSource) {
    this.plan = planOctreeLosassoFineTransport(source);
    if (!(velocity.fineCellSize > 0) || !Number.isFinite(velocity.fineCellSize)
      || velocity.dimensions.some((value, axis) => !Number.isSafeInteger(value) || value < 1
        || Math.abs(value * velocity.fineCellSize
          - source.plan.sampleDimensions[axis]! * source.plan.fineCellWidth)
          > 1e-5 * Math.max(velocity.fineCellSize, source.plan.fineCellWidth))
      || !Number.isSafeInteger(velocity.maximumLeafSize) || velocity.maximumLeafSize < 1
      || (velocity.maximumLeafSize & (velocity.maximumLeafSize - 1)) !== 0
      || velocity.maximumLeafSize > Math.max(...velocity.dimensions)
      || !Number.isSafeInteger(velocity.directoryCapacity) || velocity.directoryCapacity < 2
      || (velocity.directoryCapacity & (velocity.directoryCapacity - 1)) !== 0
      || velocity.axisFaceDirectory.size < velocity.directoryCapacity * 8) {
      throw new RangeError("Losasso fine transport sampler does not match the uniform-fine lattice");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.params = device.createBuffer({ label: "Losasso fine transport constants", size: 160,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.control = device.createBuffer({ label: "Losasso fine transport control",
      size: OCTREE_LOSASSO_FINE_TRANSPORT_CONTROL_BYTES, usage: storage });
    this.liveDispatch = device.createBuffer({ label: "Losasso fine transport live dispatch",
      size: OCTREE_LOSASSO_FINE_TRANSPORT_DISPATCH_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const delta = device.createBuffer({ label: "Losasso fine topology delta",
      size: this.plan.topologyDeltaBytes, usage: storage });
    this.topologyDelta = Object.freeze({ buffer: delta, pageCapacity: this.plan.pageCapacity,
      maximumDisplacementOffsetWords: 7 as const, candidateKeysOffsetWords: 8 as const,
      changedKeysOffsetWords: 8 + this.plan.pageCapacity });
    const readOnly = { type: "read-only-storage" as const };
    this.layout = device.createBindGroupLayout({ label: "Losasso fine transport reduced layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
    this.pipelineLayout = device.createPipelineLayout({ label: "Losasso fine transport pipeline layout",
      bindGroupLayouts: [this.layout] });
    this.prepareLayout = device.createBindGroupLayout({
      label: "Losasso fine transport prepare layout", entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.preparePipelineLayout = device.createPipelineLayout({
      label: "Losasso fine transport prepare pipeline layout",
      bindGroupLayouts: [this.prepareLayout],
    });
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso fine transport", run: () => this.initialize() }];
  }

  initializePipelines(): Promise<void> { return this.initialize(); }
  initialize(): Promise<void> {
    this.assertLive();
    if (this.prepare) return Promise.resolve();
    if (this.initialization) return this.initialization;
    this.initialization = (async () => {
      const shaderModule = this.device.createShaderModule({
        label: "Losasso fine transport and axis-face sampler shader",
        code: octreeLosassoFineTransportWGSL,
      });
      const make = (entryPoint: string, layout = this.pipelineLayout) =>
        this.device.createComputePipelineAsync({
        label: entryPoint, layout, compute: { module: shaderModule, entryPoint },
      });
      [this.prepare, this.advect, this.commit, this.finalize] = await Promise.all([
        make("prepareLosassoFineTransport", this.preparePipelineLayout), make("advectLosassoFinePhi"),
        make("commitLosassoFinePhi"), make("finalizeLosassoFineTransport"),
      ]);
      this.group = this.device.createBindGroup({ label: "Losasso fine transport reduced bindings",
        layout: this.layout, entries: [
          this.params, this.source.metadata, this.source.worklist, this.source.samples, this.source.workB,
          this.control, this.velocity.control, this.velocity.faceGeometry, this.velocity.extendedVelocity,
          this.velocity.axisFaceDirectory, this.topologyDelta.buffer,
        ].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      this.prepareGroup = this.device.createBindGroup({
        label: "Losasso fine transport prepare bindings", layout: this.prepareLayout,
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 2, resource: { buffer: this.source.worklist } },
          { binding: 5, resource: { buffer: this.control } },
          { binding: 6, resource: { buffer: this.velocity.control } },
          { binding: 7, resource: { buffer: this.velocity.faceGeometry } },
          { binding: 8, resource: { buffer: this.velocity.extendedVelocity } },
          { binding: 9, resource: { buffer: this.velocity.axisFaceDirectory } },
          { binding: 10, resource: { buffer: this.topologyDelta.buffer } },
          { binding: 11, resource: { buffer: this.liveDispatch } },
        ],
      });
    })();
    return this.initialization;
  }

  encode(broker: PassBroker, options: OctreeLosassoFineTransportOptions): PassBroker {
    this.assertLive();
    if (!this.prepare || !this.advect || !this.commit || !this.finalize
      || !this.group || !this.prepareGroup) {
      throw new Error("Losasso fine transport pipelines are not initialized");
    }
    if (!Number.isFinite(options.timestep) || options.timestep < 0) {
      throw new RangeError("Losasso fine transport timestep must be finite and non-negative");
    }
    const generation = options.generation ?? this.source.generation;
    const velocityEpoch = options.velocityEpoch ?? 0;
    const band = options.transportBandCells ?? 0xffff;
    const maximumBacktrace = options.maximumBacktraceFineCells ?? this.source.plan.fineFactor;
    for (const [label, value, minimum, maximum] of [
      ["generation", generation, 1, 0xffff_ffff], ["velocity epoch", velocityEpoch, 0, 0xffff_ffff],
      ["transport band", band, 1, 0xffff],
      ["maximum backtrace", maximumBacktrace, 1, 4 * this.source.plan.fineFactor],
    ] as const) {
      if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
        throw new RangeError(`Losasso fine transport ${label} is outside [${minimum},${maximum}]`);
      }
    }
    const bytes = new ArrayBuffer(160), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
    words.set(this.source.plan.brickDimensions, 0); words[3] = this.source.plan.brickResolution;
    words.set(this.source.plan.sampleDimensions, 4); words[7] = this.source.plan.samplesPerBrick;
    floats.set(this.source.plan.domainOrigin, 8); floats[11] = this.source.plan.fineCellWidth;
    words.set([this.plan.pageCapacity, generation, this.plan.pageCapacity, 7], 12);
    words.set(this.velocity.dimensions, 16); words[19] = this.velocity.directoryCapacity;
    floats[20] = this.velocity.fineCellSize; words[21] = this.velocity.maximumLeafSize;
    words.set([band, maximumBacktrace,
      options.boundaryPolicy === "closed-neumann" ? 1 : 0,
      options.openTopBoundary ? 1 : 0], 24);
    floats[28] = options.timestep; words[29] = velocityEpoch;
    // The bound sizes donor residency; it is not a measured CFL. Advancing a
    // fixed number of stages derived from that bound adds rounding and diffusion
    // even when the accepted 4 ms characteristic moves less than one fine cell.
    // The direct LoSasso lane therefore uses the same single midpoint trace as
    // the paper path and rejects an actual trace that exceeds the bound.
    words[30] = 1;
    const inflow = options.inflow;
    if (inflow) {
      floats.set([inflow.outletCenter_m.x, inflow.outletCenter_m.y,
        inflow.outletCenter_m.z, inflow.radius_m], 32);
      floats.set([inflow.velocity_m_s.x, inflow.velocity_m_s.y,
        inflow.velocity_m_s.z, inflow.strength], 36);
    }
    this.device.queue.writeBuffer(this.params, 0, bytes);
    const prepare = broker.compute({ label: "Losasso fine transport - prepare live dispatch" });
    prepare.setPipeline(this.prepare); prepare.setBindGroup(0, this.prepareGroup);
    prepare.dispatchWorkgroups(1);
    broker.fence("Losasso fine transport live dispatch publication");
    const pass = broker.compute({ label: "Losasso fine transport - direct axis-face sampling" });
    pass.setBindGroup(0, this.group);
    pass.setPipeline(this.advect); pass.dispatchWorkgroupsIndirect(
      this.liveDispatch, 0);
    pass.setPipeline(this.commit); pass.dispatchWorkgroupsIndirect(
      this.liveDispatch, 0);
    pass.setPipeline(this.finalize); pass.dispatchWorkgroups(1);
    return broker;
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.control.destroy(); this.liveDispatch.destroy();
    this.topologyDelta.buffer.destroy();
    this.prepare = undefined; this.advect = undefined; this.commit = undefined;
    this.finalize = undefined; this.group = undefined; this.prepareGroup = undefined;
  }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso fine transport is destroyed"); }
}
