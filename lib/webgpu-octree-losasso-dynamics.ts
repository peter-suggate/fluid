import type { PassBroker } from "./webgpu-pass-broker";
import type { SurfaceInflowState } from "./webgpu-quadtree-builder";
import {
  makeOctreeLosassoAdaptiveDynamicsWGSL,
  octreeLosassoDynamicsWGSL,
} from "./webgpu-octree-losasso-dynamics.wgsl";

export interface WebGPUOctreeLosassoDynamicsSource {
  readonly faceCapacity: number;
  readonly rowCapacity: number;
  readonly control: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  readonly axisFaceDirectory: GPUBuffer;
  readonly faceDirectoryCapacity: number;
  /** Previous advance's fixed-K extended velocity. */
  readonly extendedVelocity: GPUBuffer;
  readonly advectedVelocity: GPUBuffer;
  /** Projection input after body force and cut-cell boundary enforcement. */
  readonly predictedVelocity: GPUBuffer;
  /** Projection output, constrained again at authored nozzle faces. */
  readonly projectedVelocity: GPUBuffer;
  readonly rowFaceOffsets: GPUBuffer;
  readonly rowFaces: GPUBuffer;
  readonly rightHandSide: GPUBuffer;
  readonly faceDispatch: GPUBuffer;
  readonly rowDispatch: GPUBuffer;
}

export interface WebGPUOctreeLosassoDynamicsOptions {
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  readonly physicalCellSize: readonly [number, number, number];
  readonly domainOrigin?: readonly [number, number, number];
  readonly density: number;
  /** Row-indexed vec4f(rho,mass,leafVolume,compression). When present, the
   * paper's bounded rho>1 divergence term returns locally compressed mass to
   * visible volume during projection. */
  readonly surfaceDensityRows?: GPUBuffer;
  /** Production resident solve RHS authority inside the shared frame arena. */
  readonly rightHandSideTarget?: GPUBufferBinding;
  /** x-, x+, y-, y+, z-, z+. Defaults to a closed box. */
  readonly closedBoundaries?: readonly [boolean, boolean, boolean, boolean, boolean, boolean];
}

export interface WebGPUOctreeLosassoDynamicsStep {
  readonly dt_s: number;
  readonly gravity_m_s2: readonly [number, number, number];
  /** Backend-neutral authored nozzle state; absent/zero strength is a no-op. */
  readonly inflow?: SurfaceInflowState;
}

export interface WebGPUOctreeLosassoDynamicsSamplerSource {
  readonly kind?: "legacy";
  readonly control: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  readonly axisFaceDirectory: GPUBuffer;
  readonly extendedVelocity: GPUBuffer;
  /** Forward predictor extended over the same sampler geometry. */
  readonly predictorExtendedVelocity: GPUBuffer;
  /** Two contiguous vec4u nodal banks: accepted first, then predictor. */
  readonly stagedVelocity: GPUBuffer;
  readonly predictorStagedVelocity: GPUBuffer;
  readonly faceCapacity: number;
  readonly directoryCapacity: number;
}

/** Compact graph-owned sampler used by the fully adaptive factor-one lane. */
export interface WebGPUOctreeLosassoAdaptiveDynamicsSamplerSource {
  readonly kind: "adaptive";
  readonly leaves: GPUBuffer;
  readonly ownerArena: GPUBuffer;
  readonly leafLocator: GPUBuffer;
  readonly velocityArena: GPUBuffer;
  readonly wgsl: string;
}

export type WebGPUOctreeLosassoAnyDynamicsSamplerSource =
  | WebGPUOctreeLosassoDynamicsSamplerSource
  | WebGPUOctreeLosassoAdaptiveDynamicsSamplerSource;

const ENTRY_POINTS = [
  "advectLosassoFaces",
  "forceLosassoFaces",
  "divergenceLosassoRows",
  "constrainLosassoInflowFaces",
] as const;
type DynamicsEntryPoint = typeof ENTRY_POINTS[number];

const BINDINGS = Object.freeze({
  advectLosassoFaces: [0, 1, 2, 3, 6, 12, 15],
  forceLosassoFaces: [0, 1, 2, 3, 6, 7],
  divergenceLosassoRows: [0, 1, 2, 7, 8, 9, 10, 16],
  constrainLosassoInflowFaces: [0, 1, 3, 11],
} as const);

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) {
    throw new RangeError(`Losasso dynamics ${label} must be a positive u32`);
  }
  return value;
}

export class WebGPUOctreeLosassoDynamics {
  readonly allocatedBytes: number;
  readonly predictedVelocity: GPUBuffer;
  readonly rightHandSide: GPUBuffer;

  private readonly params: GPUBuffer;
  private pipelines?: Readonly<Record<DynamicsEntryPoint, GPUComputePipeline>>;
  private groups?: Readonly<Record<DynamicsEntryPoint, GPUBindGroup>>;
  private step?: WebGPUOctreeLosassoDynamicsStep;
  private uploadedStep?: WebGPUOctreeLosassoDynamicsStep;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly source: WebGPUOctreeLosassoDynamicsSource,
    private readonly options: WebGPUOctreeLosassoDynamicsOptions,
    private readonly sampler: WebGPUOctreeLosassoAnyDynamicsSamplerSource,
  ) {
    positiveInteger(source.faceCapacity, "face capacity");
    positiveInteger(source.rowCapacity, "row capacity");
    positiveInteger(source.faceDirectoryCapacity, "directory capacity");
    if ((source.faceDirectoryCapacity & (source.faceDirectoryCapacity - 1)) !== 0) {
      throw new RangeError("Losasso dynamics directory capacity must be a power of two");
    }
    for (const [axis, dimension] of options.dimensions.entries()) {
      positiveInteger(dimension, `dimension ${axis}`);
    }
    positiveInteger(options.maximumLeafSize, "maximum leaf size");
    if ((options.maximumLeafSize & (options.maximumLeafSize - 1)) !== 0) {
      throw new RangeError("Losasso dynamics maximum leaf size must be dyadic");
    }
    for (const [axis, width] of options.physicalCellSize.entries()) {
      if (!Number.isFinite(width) || width <= 0) {
        throw new RangeError(`Losasso dynamics cell width ${axis} must be positive and finite`);
      }
    }
    if (!Number.isFinite(options.density) || options.density <= 0) {
      throw new RangeError("Losasso dynamics density must be positive and finite");
    }
    const origin = options.domainOrigin ?? [0, 0, 0];
    if (origin.some((value) => !Number.isFinite(value))) {
      throw new RangeError("Losasso dynamics domain origin must be finite");
    }
    this.params = device.createBuffer({ label: "Losasso axis-face dynamics parameters",
      size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const words = new Uint32Array(32);
    words.set([...options.dimensions, options.maximumLeafSize], 0);
    const closed = options.closedBoundaries ?? [true, true, true, true, true, true];
    const boundaryMask = closed.reduce((mask, value, index) =>
      mask | (value ? 1 << index : 0), 0);
    words.set([source.faceCapacity, source.rowCapacity, source.faceDirectoryCapacity,
      boundaryMask], 4);
    const floats = new Float32Array(words.buffer);
    floats.set([...origin, 0], 8);
    floats.set([...options.physicalCellSize, options.density], 12);
    floats.set([0, 0, 0, 0], 16);
    floats.set([0, 0, 0, 0], 20);
    floats.set([0, 0, 0, 0], 24);
    words.set(sampler.kind === "adaptive"
      ? [0, 0, 1, options.surfaceDensityRows ? 1 : 0]
      : [sampler.faceCapacity, sampler.directoryCapacity, 0, 0], 28);
    device.queue.writeBuffer(this.params, 0, words);
    this.predictedVelocity = source.predictedVelocity;
    this.rightHandSide = source.rightHandSide;
    this.allocatedBytes = this.params.size;
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso axis-face dynamics", run: () => this.initialize() }];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.pipelines) return;
    const adaptive = this.sampler.kind === "adaptive";
    const shaderModule = this.device.createShaderModule({
      label: "Losasso reduced axis-face dynamics shader",
      code: adaptive
        ? makeOctreeLosassoAdaptiveDynamicsWGSL(this.sampler.wgsl)
        : octreeLosassoDynamicsWGSL,
    });
    const pairs = await Promise.all(ENTRY_POINTS.map(async (entryPoint) => [entryPoint,
      await this.device.createComputePipelineAsync({
        label: `Losasso dynamics - ${entryPoint}`,
        layout: "auto",
        compute: { module: shaderModule, entryPoint },
      })] as const));
    this.pipelines = Object.freeze(Object.fromEntries(pairs)) as
      Readonly<Record<DynamicsEntryPoint, GPUComputePipeline>>;
    const samplerBuffers = adaptive
      ? [this.sampler.leaves, this.sampler.ownerArena, this.sampler.leafLocator,
        this.sampler.velocityArena]
      : [this.sampler.control, this.sampler.faceGeometry,
        this.sampler.axisFaceDirectory, this.sampler.stagedVelocity];
    const buffers: readonly (GPUBuffer | GPUBufferBinding)[] = [this.params, this.source.control, this.source.faces,
      this.source.faceGeometry, this.source.axisFaceDirectory, this.source.extendedVelocity,
      this.source.advectedVelocity, this.source.predictedVelocity,
      this.source.rowFaceOffsets, this.source.rowFaces,
      this.options.rightHandSideTarget ?? this.source.rightHandSide,
      this.source.projectedVelocity, ...samplerBuffers,
      // The density branch is statically disabled when no rows are supplied,
      // but Dawn still validates the declared binding usages. Binding the RHS
      // here aliases its read_write binding 10 with read-only binding 16 in one
      // synchronization scope. The already read-only authority control is a
      // format-valid sentinel and cannot create that storage-usage conflict.
      this.options.surfaceDensityRows ?? this.source.control];
    this.groups = Object.freeze(Object.fromEntries(ENTRY_POINTS.map((entryPoint) => [entryPoint,
      this.device.createBindGroup({
        label: `Losasso dynamics bindings - ${entryPoint}`,
        layout: this.pipelines![entryPoint].getBindGroupLayout(0),
        entries: (adaptive && entryPoint === "advectLosassoFaces"
          ? [...BINDINGS[entryPoint].filter((binding) => binding < 12), 12, 13, 14, 15]
          : BINDINGS[entryPoint]).map((binding) =>
          ({ binding, resource: "buffer" in buffers[binding]!
            ? buffers[binding]! as GPUBufferBinding
            : { buffer: buffers[binding]! as GPUBuffer } })),
      })]))) as Readonly<Record<DynamicsEntryPoint, GPUBindGroup>>;
  }

  encodeAdvection(
    broker: PassBroker,
    step: WebGPUOctreeLosassoDynamicsStep,
  ): void {
    this.assertReady();
    this.writeStep(step);
    const predict = broker.compute({ label: "Losasso S1a - advect graded axis faces" });
    predict.setPipeline(this.pipelines!.advectLosassoFaces);
    predict.setBindGroup(0, this.groups!.advectLosassoFaces);
    predict.dispatchWorkgroupsIndirect(this.source.faceDispatch, 0);
  }

  encodeForcesAndDivergence(
    broker: PassBroker,
    step: WebGPUOctreeLosassoDynamicsStep,
  ): void {
    this.assertReady();
    if (this.step && this.step.dt_s !== step.dt_s) {
      throw new Error("Losasso advection and divergence must use one timestep");
    }
    this.writeStep(step);
    const pass = broker.compute({ label: "Losasso S2 - force faces and assemble divergence" });
    pass.setPipeline(this.pipelines!.forceLosassoFaces);
    pass.setBindGroup(0, this.groups!.forceLosassoFaces);
    pass.dispatchWorkgroupsIndirect(this.source.faceDispatch, 0);
    pass.setPipeline(this.pipelines!.divergenceLosassoRows);
    pass.setBindGroup(0, this.groups!.divergenceLosassoRows);
    pass.dispatchWorkgroupsIndirect(this.source.rowDispatch, 0);
    this.step = undefined;
  }

  /** Re-impose the nozzle Dirichlet value after pressure projection. */
  encodeInflowConstraint(broker: PassBroker, step: WebGPUOctreeLosassoDynamicsStep): void {
    this.assertReady();
    const inflow = step.inflow;
    if (!inflow || inflow.strength <= 0 || inflow.radius_m <= 0
      || Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y,
        inflow.velocity_m_s.z) <= 1e-6) return;
    this.writeStep(step);
    const pass = broker.compute({ label: "Losasso S3 - constrain authored inflow faces" });
    pass.setPipeline(this.pipelines!.constrainLosassoInflowFaces);
    pass.setBindGroup(0, this.groups!.constrainLosassoInflowFaces);
    pass.dispatchWorkgroupsIndirect(this.source.faceDispatch, 0);
    this.step = undefined;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy();
    this.pipelines = undefined;
    this.groups = undefined;
  }

  private writeStep(step: WebGPUOctreeLosassoDynamicsStep): void {
    if (this.uploadedStep === step) {
      this.step = step;
      return;
    }
    if (!Number.isFinite(step.dt_s) || step.dt_s < 0) {
      throw new RangeError("Losasso dynamics timestep must be non-negative and finite");
    }
    if (step.gravity_m_s2.some((value) => !Number.isFinite(value))) {
      throw new RangeError("Losasso dynamics gravity must be finite");
    }
    const inflow = step.inflow;
    if (inflow && (!Number.isFinite(inflow.outletCenter_m.x)
      || !Number.isFinite(inflow.outletCenter_m.y)
      || !Number.isFinite(inflow.outletCenter_m.z)
      || !Number.isFinite(inflow.radius_m) || inflow.radius_m < 0
      || !Number.isFinite(inflow.velocity_m_s.x)
      || !Number.isFinite(inflow.velocity_m_s.y)
      || !Number.isFinite(inflow.velocity_m_s.z)
      || !Number.isFinite(inflow.apertureScale) || inflow.apertureScale < 0
      || !Number.isFinite(inflow.strength) || inflow.strength < 0)) {
      throw new RangeError("Losasso dynamics inflow must be finite and non-negative");
    }
    const active = inflow && inflow.strength > 0 && inflow.radius_m > 0
      && Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y,
        inflow.velocity_m_s.z) > 1e-6;
    // SurfaceInflowState uses the scene's centered X/Z coordinates; compact
    // Losasso face geometry uses the non-negative topology lattice.
    const origin = this.options.domainOrigin ?? [0, 0, 0];
    const latticeInflow = active ? [
      origin[0] + inflow.outletCenter_m.x
        + 0.5 * this.options.dimensions[0] * this.options.physicalCellSize[0],
      origin[1] + inflow.outletCenter_m.y,
      origin[2] + inflow.outletCenter_m.z
        + 0.5 * this.options.dimensions[2] * this.options.physicalCellSize[2],
    ] as const : [0, 0, 0] as const;
    const scale = active ? inflow.strength * inflow.apertureScale : 0;
    // One contiguous upload covers dynamic words 8..27. The immutable cell
    // widths/density occupy words 12..15 inside that span, so restate them
    // byte-for-byte rather than issuing four driver-visible subrange writes.
    const uploaded = new Float32Array(20);
    uploaded.set([...origin, step.dt_s], 0);
    uploaded.set([...this.options.physicalCellSize, this.options.density], 4);
    uploaded.set([...step.gravity_m_s2, 0], 8);
    if (active) {
      uploaded.set([...latticeInflow, inflow.radius_m], 12);
      uploaded.set([inflow.velocity_m_s.x * scale, inflow.velocity_m_s.y * scale,
        inflow.velocity_m_s.z * scale, 1], 16);
    }
    this.device.queue.writeBuffer(this.params, 8 * 4, uploaded);
    this.uploadedStep = step;
    this.step = step;
  }

  private assertReady(): void {
    this.assertLive();
    if (!this.pipelines || !this.groups) throw new Error("Losasso dynamics is not initialized");
  }
  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso dynamics is destroyed");
  }
}
