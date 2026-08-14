import type { TwoTileCompositeGrid } from "./two-tile-composite-grid";
import { twoTilePressureOperatorWGSL } from "./two-tile-pressure-operator.wgsl";

export interface PackedTwoTilePressureOperator {
  readonly cellCount: number;
  readonly gradientRowCount: number;
  readonly termCount: number;
  readonly incidenceCount: number;
  /** Four u32 values matching the WGSL Parameters structure. */
  readonly parameters: Uint32Array;
  /** 16-byte GradientRow records. */
  readonly gradientRows: ArrayBuffer;
  /** 8-byte GradientTerm records. */
  readonly gradientTerms: ArrayBuffer;
  /** One CSR offset per cell plus the terminating offset. */
  readonly cellIncidenceOffsets: Uint32Array;
  /** 8-byte CellIncidence records. */
  readonly cellIncidences: ArrayBuffer;
}

/**
 * Pack the exact rows produced by `buildTwoTileCompositeGrid` for the GPU.
 * No matrix is assembled: an incidence stores the row and the cell's term in
 * that row, which is sufficient to evaluate one cell of G^T W G independently.
 */
export function packTwoTilePressureOperator(
  grid: TwoTileCompositeGrid,
): PackedTwoTilePressureOperator {
  const rowCount = grid.gradientRows.length;
  let termCount = 0;
  for (const row of grid.gradientRows) {
    termCount += row.terms.length;
  }

  const rowRecords = new ArrayBuffer(Math.max(16, rowCount * 16));
  const rowView = new DataView(rowRecords);
  const termRecords = new ArrayBuffer(Math.max(8, termCount * 8));
  const termView = new DataView(termRecords);
  const byCell: { rowId: number; termIndex: number }[][] = Array.from(
    { length: grid.cells.length },
    () => [],
  );

  let nextTerm = 0;
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const row = grid.gradientRows[rowIndex];
    if (row.id !== rowIndex) {
      throw new Error(`gradient row ${rowIndex} has non-canonical ID ${row.id}`);
    }
    const rowByte = rowIndex * 16;
    rowView.setUint32(rowByte, nextTerm, true);
    rowView.setUint32(rowByte + 4, row.terms.length, true);
    rowView.setFloat32(rowByte + 8, row.dualWeight, true);
    rowView.setUint32(rowByte + 12, 0, true);
    for (const term of row.terms) {
      if (!Number.isSafeInteger(term.cellId) || term.cellId < 0 || term.cellId >= grid.cells.length) {
        throw new RangeError(`gradient row ${row.id} references invalid cell ${term.cellId}`);
      }
      const termByte = nextTerm * 8;
      termView.setUint32(termByte, term.cellId, true);
      termView.setFloat32(termByte + 4, term.coefficient, true);
      byCell[term.cellId].push({ rowId: row.id, termIndex: nextTerm });
      nextTerm += 1;
    }
  }

  const offsets = new Uint32Array(grid.cells.length + 1);
  let incidenceCount = 0;
  for (let cellId = 0; cellId < grid.cells.length; cellId += 1) {
    offsets[cellId] = incidenceCount;
    incidenceCount += byCell[cellId].length;
  }
  offsets[grid.cells.length] = incidenceCount;
  const incidenceRecords = new ArrayBuffer(Math.max(8, incidenceCount * 8));
  const incidenceView = new DataView(incidenceRecords);
  let incidenceIndex = 0;
  for (const incidences of byCell) {
    for (const incidence of incidences) {
      const byte = incidenceIndex * 8;
      incidenceView.setUint32(byte, incidence.rowId, true);
      incidenceView.setUint32(byte + 4, incidence.termIndex, true);
      incidenceIndex += 1;
    }
  }

  return {
    cellCount: grid.cells.length,
    gradientRowCount: rowCount,
    termCount,
    incidenceCount,
    parameters: new Uint32Array([grid.cells.length, rowCount, termCount, incidenceCount]),
    gradientRows: rowRecords,
    gradientTerms: termRecords,
    cellIncidenceOffsets: offsets,
    cellIncidences: incidenceRecords,
  };
}

type BindGroupCache = WeakMap<GPUBuffer, WeakMap<GPUBuffer, GPUBindGroup>>;

/** GPU owner for the frozen two-tile G^T W G operator and its packed topology. */
export class WebGPUTwoTilePressureOperator {
  readonly cellCount: number;
  readonly gradientRowCount: number;
  readonly termCount: number;
  readonly incidenceCount: number;

  private readonly parameters: GPUBuffer;
  private readonly gradientRows: GPUBuffer;
  private readonly gradientTerms: GPUBuffer;
  private readonly cellIncidenceOffsets: GPUBuffer;
  private readonly cellIncidences: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipeline: GPUComputePipeline;
  private readonly groups: BindGroupCache = new WeakMap();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    packed: PackedTwoTilePressureOperator,
    buffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer],
    layout: GPUBindGroupLayout,
    pipeline: GPUComputePipeline,
  ) {
    this.cellCount = packed.cellCount;
    this.gradientRowCount = packed.gradientRowCount;
    this.termCount = packed.termCount;
    this.incidenceCount = packed.incidenceCount;
    [this.parameters, this.gradientRows, this.gradientTerms,
      this.cellIncidenceOffsets, this.cellIncidences] = buffers;
    this.layout = layout;
    this.pipeline = pipeline;
  }

  static async create(
    device: GPUDevice,
    grid: TwoTileCompositeGrid,
  ): Promise<WebGPUTwoTilePressureOperator> {
    const packed = packTwoTilePressureOperator(grid);
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const upload = (label: string, source: ArrayBuffer | ArrayBufferView, usage: GPUBufferUsageFlags) => {
      const byteLength = source instanceof ArrayBuffer ? source.byteLength : source.byteLength;
      const buffer = device.createBuffer({ label, size: Math.max(4, byteLength), usage });
      if (source instanceof ArrayBuffer) {
        device.queue.writeBuffer(buffer, 0, source);
      } else {
        device.queue.writeBuffer(buffer, 0, source.buffer as ArrayBuffer,
          source.byteOffset, source.byteLength);
      }
      return buffer;
    };
    const buffers = [
      upload("Adaptive mass two-tile pressure parameters", packed.parameters,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      upload("Adaptive mass two-tile gradient rows", packed.gradientRows, storageUsage),
      upload("Adaptive mass two-tile gradient terms", packed.gradientTerms, storageUsage),
      upload("Adaptive mass two-tile cell incidence offsets", packed.cellIncidenceOffsets, storageUsage),
      upload("Adaptive mass two-tile cell incidences", packed.cellIncidences, storageUsage),
    ] as const;

    const readOnly = { type: "read-only-storage" as const };
    const layout = device.createBindGroupLayout({
      label: "Adaptive mass two-tile pressure operator layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const shader = device.createShaderModule({
      label: "Adaptive mass two-tile pressure operator shader",
      code: twoTilePressureOperatorWGSL,
    });
    const pipeline = await device.createComputePipelineAsync({
      label: "Apply adaptive mass two-tile pressure operator",
      layout: device.createPipelineLayout({
        label: "Adaptive mass two-tile pressure operator pipeline layout",
        bindGroupLayouts: [layout],
      }),
      compute: { module: shader, entryPoint: "applyTwoTilePressureOperator" },
    });
    return new WebGPUTwoTilePressureOperator(device, packed, buffers, layout, pipeline);
  }

  /** Encode one matrix-free apply. Input and output contain `cellCount` f32 values. */
  encode(encoder: GPUCommandEncoder, input: GPUBuffer, output: GPUBuffer): void {
    this.assertLive();
    let outputs = this.groups.get(input);
    if (!outputs) {
      outputs = new WeakMap();
      this.groups.set(input, outputs);
    }
    let group = outputs.get(output);
    if (!group) {
      const topology = [this.parameters, this.gradientRows, this.gradientTerms,
        this.cellIncidenceOffsets, this.cellIncidences];
      group = this.device.createBindGroup({
        label: "Adaptive mass two-tile pressure operator bindings",
        layout: this.layout,
        entries: [
          ...topology.map((buffer, binding) => ({ binding, resource: { buffer } })),
          { binding: 5, resource: { buffer: input } },
          { binding: 6, resource: { buffer: output } },
        ],
      });
      outputs.set(output, group);
    }
    const pass = encoder.beginComputePass({ label: "Adaptive mass two-tile G^T W G apply" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.cellCount / 64));
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of [this.parameters, this.gradientRows, this.gradientTerms,
      this.cellIncidenceOffsets, this.cellIncidences]) buffer.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("adaptive mass two-tile pressure operator is destroyed");
  }
}
