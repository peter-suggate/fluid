import type { TwoTileConservativeOperator } from "./two-tile-conservative-transport";
import { twoTileConservativeTransportWGSL } from "./two-tile-conservative-transport.wgsl";

export interface PackedTwoTileConservativeTransport {
  readonly receiverCount: number;
  readonly coefficientCount: number;
  /** Four u32 values matching the WGSL Parameters structure. */
  readonly parameters: Uint32Array;
  /** Receiver CSR offsets, including the terminating offset. */
  readonly rowOffsets: Uint32Array;
  /** Interleaved 8-byte `{ donorCellId, coefficient }` records. */
  readonly coefficients: ArrayBuffer;
  /** Persistent gamma state authored by the CPU row-construction oracle. */
  readonly nextGamma: Float32Array;
}

/**
 * Losslessly pack the CPU-built sparsity pattern, with numerical values rounded
 * once to the f32 representation used by WebGPU.
 *
 * This function does not build traces or reproduce CM12 conditioning. A new
 * packed operator is required whenever the source persistent-gamma snapshot
 * changes, exactly like rebuilding the CPU operator for the next step.
 */
export function packTwoTileConservativeTransport(
  operator: TwoTileConservativeOperator,
): PackedTwoTileConservativeTransport {
  const receiverCount = operator.grid.cells.length;
  if (operator.rows.length !== receiverCount || operator.nextGamma.length !== receiverCount) {
    throw new RangeError("two-tile transport operator dimensions do not match its grid");
  }
  const rowOffsets = new Uint32Array(receiverCount + 1);
  let coefficientCount = 0;
  for (let receiver = 0; receiver < receiverCount; receiver += 1) {
    rowOffsets[receiver] = coefficientCount;
    coefficientCount += operator.rows[receiver].length;
  }
  rowOffsets[receiverCount] = coefficientCount;

  const coefficientRecords = new ArrayBuffer(Math.max(8, coefficientCount * 8));
  const coefficientView = new DataView(coefficientRecords);
  let coefficientIndex = 0;
  for (const row of operator.rows) {
    let previousDonor = -1;
    for (const entry of row) {
      if (!Number.isSafeInteger(entry.donorCellId)
        || entry.donorCellId < 0 || entry.donorCellId >= receiverCount) {
        throw new RangeError(`transport row references invalid donor ${entry.donorCellId}`);
      }
      if (entry.donorCellId <= previousDonor) {
        throw new Error("two-tile transport donor IDs must be strictly sorted within a row");
      }
      if (!Number.isFinite(entry.coefficient) || entry.coefficient < 0) {
        throw new RangeError(`transport coefficient must be finite and nonnegative; received ${entry.coefficient}`);
      }
      const byte = coefficientIndex * 8;
      coefficientView.setUint32(byte, entry.donorCellId, true);
      coefficientView.setFloat32(byte + 4, entry.coefficient, true);
      previousDonor = entry.donorCellId;
      coefficientIndex += 1;
    }
  }

  const nextGamma = Float32Array.from(operator.nextGamma);
  for (let cell = 0; cell < nextGamma.length; cell += 1) {
    if (!Number.isFinite(nextGamma[cell]) || nextGamma[cell] < 0) {
      throw new RangeError(`nextGamma[${cell}] is invalid after f32 conversion`);
    }
  }
  return {
    receiverCount,
    coefficientCount,
    parameters: new Uint32Array([receiverCount, coefficientCount, 0, 0]),
    rowOffsets,
    coefficients: coefficientRecords,
    nextGamma,
  };
}

interface BufferSourceView {
  readonly buffer: ArrayBufferLike;
  readonly byteOffset: number;
  readonly byteLength: number;
}

type OutputGroupCache = WeakMap<GPUBuffer,
  WeakMap<GPUBuffer, WeakMap<GPUBuffer, GPUBindGroup>>>;

/** GPU executor for one CPU-built, persistent-gamma-specific transport operator. */
export class WebGPUTwoTileConservativeTransport {
  readonly receiverCount: number;
  readonly coefficientCount: number;

  private readonly parameters: GPUBuffer;
  private readonly rowOffsets: GPUBuffer;
  private readonly coefficients: GPUBuffer;
  private readonly nextGamma: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipeline: GPUComputePipeline;
  private readonly groups: OutputGroupCache = new WeakMap();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    packed: PackedTwoTileConservativeTransport,
    buffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer],
    layout: GPUBindGroupLayout,
    pipeline: GPUComputePipeline,
  ) {
    this.receiverCount = packed.receiverCount;
    this.coefficientCount = packed.coefficientCount;
    [this.parameters, this.rowOffsets, this.coefficients, this.nextGamma] = buffers;
    this.layout = layout;
    this.pipeline = pipeline;
  }

  static async create(
    device: GPUDevice,
    operator: TwoTileConservativeOperator,
  ): Promise<WebGPUTwoTileConservativeTransport> {
    const packed = packTwoTileConservativeTransport(operator);
    const upload = (
      label: string,
      source: ArrayBuffer | BufferSourceView,
      usage: GPUBufferUsageFlags,
    ): GPUBuffer => {
      const buffer = device.createBuffer({ label, size: Math.max(4, source.byteLength), usage });
      if (source instanceof ArrayBuffer) {
        device.queue.writeBuffer(buffer, 0, source);
      } else {
        device.queue.writeBuffer(buffer, 0, source.buffer as ArrayBuffer,
          source.byteOffset, source.byteLength);
      }
      return buffer;
    };
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const buffers = [
      upload("Adaptive mass two-tile transport parameters", packed.parameters,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      upload("Adaptive mass two-tile transport row offsets", packed.rowOffsets, storage),
      upload("Adaptive mass two-tile transport coefficients", packed.coefficients, storage),
      upload("Adaptive mass two-tile transport next gamma", packed.nextGamma, storage),
    ] as const;

    const readOnly = { type: "read-only-storage" as const };
    const layout = device.createBindGroupLayout({
      label: "Adaptive mass two-tile conservative transport layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const shader = device.createShaderModule({
      label: "Adaptive mass two-tile conservative transport shader",
      code: twoTileConservativeTransportWGSL,
    });
    const pipeline = await device.createComputePipelineAsync({
      label: "Apply adaptive mass two-tile conservative transport",
      layout: device.createPipelineLayout({
        label: "Adaptive mass two-tile conservative transport pipeline layout",
        bindGroupLayouts: [layout],
      }),
      compute: { module: shader, entryPoint: "applyTwoTileConservativeTransport" },
    });
    return new WebGPUTwoTileConservativeTransport(device, packed, buffers, layout, pipeline);
  }

  /**
   * Apply the frozen density rows and publish their paired authoritative gamma.
   * All three value buffers contain `receiverCount` f32 values.
   */
  encode(
    encoder: GPUCommandEncoder,
    sourceDensity: GPUBuffer,
    destinationDensity: GPUBuffer,
    destinationGamma: GPUBuffer,
  ): void {
    this.assertLive();
    let densities = this.groups.get(sourceDensity);
    if (!densities) {
      densities = new WeakMap();
      this.groups.set(sourceDensity, densities);
    }
    let gammas = densities.get(destinationDensity);
    if (!gammas) {
      gammas = new WeakMap();
      densities.set(destinationDensity, gammas);
    }
    let group = gammas.get(destinationGamma);
    if (!group) {
      const topology = [this.parameters, this.rowOffsets, this.coefficients, this.nextGamma];
      group = this.device.createBindGroup({
        label: "Adaptive mass two-tile conservative transport bindings",
        layout: this.layout,
        entries: [
          ...topology.map((buffer, binding) => ({ binding, resource: { buffer } })),
          { binding: 4, resource: { buffer: sourceDensity } },
          { binding: 5, resource: { buffer: destinationDensity } },
          { binding: 6, resource: { buffer: destinationGamma } },
        ],
      });
      gammas.set(destinationGamma, group);
    }
    const pass = encoder.beginComputePass({
      label: "Adaptive mass two-tile conservative transport apply",
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.receiverCount / 64));
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of [this.parameters, this.rowOffsets, this.coefficients, this.nextGamma]) {
      buffer.destroy();
    }
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("adaptive mass two-tile transport operator is destroyed");
  }
}
