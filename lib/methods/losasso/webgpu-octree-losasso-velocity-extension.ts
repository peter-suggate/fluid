import type { PassBroker } from "../../core/webgpu-pass-broker";
import type { LosassoVelocityExtensionMode } from "../octree-shared/octree-coarse-backend";
import { octreeLosassoVelocityExtensionWGSL } from "./webgpu-octree-losasso-velocity-extension.wgsl";

/** Production D4 support: four-cell band + two-cell backtrace + one MAC stencil cell. */
export const OCTREE_LOSASSO_EXTENSION_WIDTH = 7;
/** The W7 radius is Euclidean while the compact graph has six-neighbour
 * (Manhattan) edges. A point inside the physical band can therefore be
 * ceil(sqrt(3) * 7) graph edges from a seed. One final settling sweep makes
 * every published W7 value available to the following characteristic. */
export const OCTREE_LOSASSO_EXTENSION_SWEEPS = 14;
export const OCTREE_LOSASSO_FACE_METRIC_WORDS = 4;
export const OCTREE_LOSASSO_EXTENSION_BINDINGS = 7;

export const OCTREE_LOSASSO_FACE_FLAGS = Object.freeze({ seed: 1 << 0 } as const);

export interface WebGPUOctreeLosassoVelocityExtensionSource {
  readonly faceCapacity: number;
  /** Shared coarse authority `[epoch,rowCount,faceCount,valid,...]`. */
  readonly control: GPUBuffer;
  /** Exact GPU-authored dispatch for the published W7 graph. */
  readonly faceDispatch: GPUBuffer;
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

/** Cover the full 3-D W=7 extension band, plus one settling sweep. */
export const OCTREE_LOSASSO_MINIMUM_EXTENSION_SWEEPS =
  OCTREE_LOSASSO_EXTENSION_SWEEPS;

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
  private predictorGroups?: readonly GPUBindGroup[];
  /**
   * Groups that write the DESTINATION field instead of the next scratch bank,
   * indexed by which scratch bank the last sweep reads. A shortened chain has
   * to land in `extendedVelocity` (or, for the predictor, back in the
   * projected field) whatever its length, and only the terminal sweep differs
   * from the full-length schedule.
   */
  private terminalGroups?: Readonly<Record<"a" | "b", GPUBindGroup>>;
  private predictorTerminalGroups?: Readonly<Record<"a" | "b", GPUBindGroup>>;
  private sweeps = OCTREE_LOSASSO_EXTENSION_SWEEPS;
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
    const sweepFields = (destination: GPUBuffer): readonly GPUBuffer[] => {
      const fields: GPUBuffer[] = [this.source.projectedVelocity];
      for (let sweep = 1; sweep < OCTREE_LOSASSO_EXTENSION_SWEEPS; sweep += 1) {
        fields.push((sweep & 1) !== 0 ? this.scratchA : this.scratchB);
      }
      fields.push(destination);
      return Object.freeze(fields);
    };
    const velocities = sweepFields(this.source.extendedVelocity);
    const predictorVelocities = sweepFields(this.source.projectedVelocity);
    const group = (label: string, read: GPUBuffer, write: GPUBuffer) =>
      this.device.createBindGroup({ label, layout: this.layout, entries: [
        { binding: 0, resource: { buffer: this.params, size: 16 } },
        { binding: 1, resource: { buffer: this.source.control } },
        { binding: 2, resource: { buffer: this.source.faceMetrics } },
        { binding: 3, resource: { buffer: this.source.adjacencyOffsets } },
        { binding: 4, resource: { buffer: this.source.adjacencyFaces } },
        { binding: 5, resource: { buffer: read } },
        { binding: 6, resource: { buffer: write } },
      ] });
    const makeGroups = (fields: readonly GPUBuffer[], label: string) =>
      Array.from({ length: OCTREE_LOSASSO_EXTENSION_SWEEPS }, (_, sweep) =>
        group(`Losasso ${label} Jacobi sweep ${sweep + 1}`, fields[sweep]!, fields[sweep + 1]!));
    const makeTerminal = (destination: GPUBuffer, label: string) => Object.freeze({
      a: group(`Losasso ${label} Jacobi terminal sweep from A`, this.scratchA, destination),
      b: group(`Losasso ${label} Jacobi terminal sweep from B`, this.scratchB, destination),
    });
    this.groups = makeGroups(velocities, "published");
    this.predictorGroups = makeGroups(predictorVelocities, "predictor");
    this.terminalGroups = makeTerminal(this.source.extendedVelocity, "published");
    this.predictorTerminalGroups = makeTerminal(this.source.projectedVelocity, "predictor");
  }

  /**
   * Jacobi sweeps per extension, for both the published field and the
   * predictor. The default 14 covers the 3-D W7 adjacency graph plus one settling
   * sweep; a shorter chain leaves the outermost air faces holding whatever the
   * previous advance published, which the next semi-Lagrangian backtrace then
   * samples. The terminal sweep always writes the destination field, so the
   * consumers see a complete publication at any length.
   */
  setSweeps(sweeps: number): void {
    this.assertLive();
    if (!Number.isSafeInteger(sweeps)) {
      this.sweeps = OCTREE_LOSASSO_EXTENSION_SWEEPS;
      return;
    }
    // The published graph is W7. Leaving any of it stale makes the apparent
    // transport reach depend on a performance dial, so the graph contract is
    // width + one settling sweep, not an independently shrinkable K.
    this.sweeps = Math.min(OCTREE_LOSASSO_EXTENSION_SWEEPS,
      Math.max(OCTREE_LOSASSO_MINIMUM_EXTENSION_SWEEPS, sweeps));
  }

  /** Sweeps the next encode will emit. */
  get encodedSweeps(): number { return this.sweeps; }

  /**
   * Emit `sweeps` chained dispatches, the last of which writes `destination`.
   * At full length this is exactly the prebuilt chain; shortened, only the
   * final bind group differs.
   */
  private encodeChain(
    pass: GPUComputePassEncoder,
    chain: readonly GPUBindGroup[],
    terminal: Readonly<Record<"a" | "b", GPUBindGroup>>,
  ): void {
    const last = this.sweeps - 1;
    for (let sweep = 0; sweep < last; sweep += 1) {
      pass.setPipeline(this.pipeline!);
      pass.setBindGroup(0, chain[sweep]!, [sweep * this.uniformStride]);
      pass.dispatchWorkgroupsIndirect(this.source.faceDispatch, 0);
    }
    // Odd read indices are scratch A, even ones scratch B; index 0 is the
    // projected seed, which `setSweeps` keeps out of terminal position.
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, last % 2 === 1 ? terminal.a : terminal.b,
      [last * this.uniformStride]);
    pass.dispatchWorkgroupsIndirect(this.source.faceDispatch, 0);
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
    this.encodeChain(pass, this.groups, this.terminalGroups!);
    return true;
  }

  /** Extend a gathered MacCormack predictor in place without consuming S3e's once-per-advance slot. */
  encodePredictor(broker: PassBroker): void {
    this.assertLive();
    if (!this.pipeline || !this.predictorGroups) {
      throw new Error("Losasso velocity extension pipeline is not initialized");
    }
    const pass = broker.compute({ label: `Losasso S1b - ${this.mode} predictor extension` });
    this.encodeChain(pass, this.predictorGroups, this.predictorTerminalGroups!);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy(); this.scratchA.destroy(); this.scratchB.destroy();
    this.pipeline = undefined; this.groups = undefined; this.predictorGroups = undefined;
    this.terminalGroups = undefined; this.predictorTerminalGroups = undefined;
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso velocity extension is destroyed");
  }
}
