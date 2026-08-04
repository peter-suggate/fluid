import type { PassBroker } from "./webgpu-pass-broker";
import type { LosassoVelocityExtensionMode } from "./octree-coarse-backend";
import { octreeLosassoVelocityExtensionWGSL } from "./webgpu-octree-losasso-velocity-extension.wgsl";

/** Production D4 support: four-cell band + two-cell backtrace + one MAC stencil cell. */
export const OCTREE_LOSASSO_EXTENSION_WIDTH = 7;
/** One extra synchronous relaxation beyond the graph radius. */
export const OCTREE_LOSASSO_EXTENSION_SWEEPS = 8;
export const OCTREE_LOSASSO_FACE_METRIC_WORDS = 4;
export const OCTREE_LOSASSO_EXTENSION_BINDINGS = 7;

export const OCTREE_LOSASSO_FACE_FLAGS = Object.freeze({ seed: 1 << 0 } as const);

export interface WebGPUOctreeLosassoVelocityExtensionSource {
  readonly faceCapacity: number;
  /** Shared coarse authority `[epoch,rowCount,faceCount,valid,...]`. */
  readonly control: GPUBuffer;
  /** vec4u(axis, flags, bitcast(abs(phi)), graphLayer). */
  readonly faceMetrics: GPUBuffer;
  readonly adjacencyOffsets: GPUBuffer;
  /** Ascending face ids within every CSR row. */
  readonly adjacencyFaces: GPUBuffer;
  /** Projected seed field at the S3e position. */
  readonly projectedVelocity: GPUBuffer;
  /** Published extension consumed by the next advance's S1 sampling. */
  readonly extendedVelocity: GPUBuffer;
}

export interface OctreeLosassoVelocityExtensionPlan {
  readonly faceCapacity: number;
  readonly width: typeof OCTREE_LOSASSO_EXTENSION_WIDTH;
  readonly sweeps: typeof OCTREE_LOSASSO_EXTENSION_SWEEPS;
  readonly dispatch: readonly [number, number, 1];
  readonly scratchBytes: number;
}

export function planOctreeLosassoVelocityExtension(
  faceCapacity: number,
): OctreeLosassoVelocityExtensionPlan {
  if (!Number.isSafeInteger(faceCapacity) || faceCapacity < 1) {
    throw new RangeError("Losasso velocity-extension face capacity must be positive");
  }
  const groups = Math.ceil(faceCapacity / 64);
  const groupsX = Math.min(groups, 65_535), groupsY = Math.ceil(groups / groupsX);
  if (groupsY > 65_535) throw new RangeError("Losasso velocity extension exceeds two-dimensional dispatch");
  return Object.freeze({ faceCapacity, width: OCTREE_LOSASSO_EXTENSION_WIDTH,
    sweeps: OCTREE_LOSASSO_EXTENSION_SWEEPS,
    dispatch: [groupsX, groupsY, 1] as const, scratchBytes: 2 * faceCapacity * 4 });
}

/**
 * Fixed-K synchronous Jacobi extension over same-axis face adjacency.
 * `encodeOncePerAdvance` refuses a second S3e build for the same advance; it
 * owns no convergence counter, reduction buffer, or fixed-point frontier.
 */
export class WebGPUOctreeLosassoVelocityExtension {
  readonly plan: OctreeLosassoVelocityExtensionPlan;
  readonly bindingCount = OCTREE_LOSASSO_EXTENSION_BINDINGS;
  readonly scheduling = "once-per-advance-lagged-consumer" as const;
  readonly allocatedBytes: number;

  private readonly params: GPUBuffer;
  private readonly uniformStride: number;
  private readonly scratchA: GPUBuffer;
  private readonly scratchB: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private pipeline?: GPUComputePipeline;
  private groups?: readonly GPUBindGroup[];
  private lastAdvance = -1;
  private lastEpoch = 0;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly source: WebGPUOctreeLosassoVelocityExtensionSource,
    private readonly mode: LosassoVelocityExtensionMode = "fixed-jacobi",
  ) {
    this.plan = planOctreeLosassoVelocityExtension(source.faceCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.uniformStride = device.limits.minUniformBufferOffsetAlignment;
    this.params = device.createBuffer({ label: "Losasso extension constants",
      size: this.uniformStride * OCTREE_LOSASSO_EXTENSION_SWEEPS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.scratchA = device.createBuffer({ label: "Losasso extension Jacobi A",
      size: source.faceCapacity * 4, usage: storage });
    this.scratchB = device.createBuffer({ label: "Losasso extension Jacobi B",
      size: source.faceCapacity * 4, usage: storage });
    const parameterBytes = new ArrayBuffer(this.params.size);
    const parameterWords = new Uint32Array(parameterBytes);
    for (let sweep = 0; sweep < OCTREE_LOSASSO_EXTENSION_SWEEPS; sweep += 1) {
      const at = sweep * this.uniformStride / 4;
      parameterWords.set([source.faceCapacity, OCTREE_LOSASSO_EXTENSION_WIDTH,
        sweep + 1, mode === "causal-front" ? 1 : 0], at);
    }
    device.queue.writeBuffer(this.params, 0, parameterBytes);
    const readOnly = { type: "read-only-storage" as const };
    this.layout = device.createBindGroupLayout({
      label: "Losasso axis-face extension reduced layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {
          type: "uniform", hasDynamicOffset: true, minBindingSize: 16,
        } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({
      label: "Losasso axis-face extension pipeline layout", bindGroupLayouts: [this.layout],
    });
    this.allocatedBytes = this.params.size + this.scratchA.size + this.scratchB.size;
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso fixed-K velocity extension", run: () => this.initialize() }];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.pipeline) return;
    const shaderModule = this.device.createShaderModule({
      label: "Losasso fixed-K axis-face extension shader",
      code: octreeLosassoVelocityExtensionWGSL,
    });
    this.pipeline = await this.device.createComputePipelineAsync({
      label: "Losasso fixed-K axis-face Jacobi sweep", layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: "jacobiLosassoAxisFaces" },
    });
    const velocities = [this.source.projectedVelocity, this.scratchA, this.scratchB,
      this.scratchA, this.scratchB, this.scratchA, this.scratchB, this.scratchA,
      this.source.extendedVelocity] as const;
    this.groups = Array.from({ length: OCTREE_LOSASSO_EXTENSION_SWEEPS }, (_, sweep) =>
      this.device.createBindGroup({ label: `Losasso Jacobi sweep ${sweep + 1}`,
        layout: this.layout, entries: [
          { binding: 0, resource: { buffer: this.params, size: 16 } },
          { binding: 1, resource: { buffer: this.source.control } },
          { binding: 2, resource: { buffer: this.source.faceMetrics } },
          { binding: 3, resource: { buffer: this.source.adjacencyOffsets } },
          { binding: 4, resource: { buffer: this.source.adjacencyFaces } },
          { binding: 5, resource: { buffer: velocities[sweep]! } },
          { binding: 6, resource: { buffer: velocities[sweep + 1]! } },
        ] }));
  }

  encodeOncePerAdvance(broker: PassBroker, advance: number, topologyEpoch: number): boolean {
    this.assertLive();
    if (!this.pipeline || !this.groups) throw new Error("Losasso extension pipeline is not initialized");
    if (!Number.isSafeInteger(advance) || advance < 0
      || !Number.isSafeInteger(topologyEpoch) || topologyEpoch < 1) {
      throw new RangeError("Losasso extension scheduling needs a non-negative advance and positive epoch");
    }
    if (advance < this.lastAdvance) throw new Error("Losasso extension advances must be monotonic");
    if (advance === this.lastAdvance) {
      if (topologyEpoch !== this.lastEpoch) {
        throw new Error("Losasso topology epoch changed twice inside one advance");
      }
      return false;
    }
    this.lastAdvance = advance;
    this.lastEpoch = topologyEpoch;
    const pass = broker.compute({ label: `Losasso S3e - ${this.mode} axis-face extension` });
    for (const [sweep, group] of this.groups.entries()) {
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, group, [sweep * this.uniformStride]);
      pass.dispatchWorkgroups(...this.plan.dispatch);
    }
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy(); this.scratchA.destroy(); this.scratchB.destroy();
    this.pipeline = undefined; this.groups = undefined;
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso velocity extension is destroyed");
  }
}
