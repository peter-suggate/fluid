import { OCTREE_LOSASSO_CONTROL_WORDS, OCTREE_LOSASSO_FACE_BYTES } from
  "./octree-losasso-operator";
import {
  planSignedRadix256F32Reduction,
  SIGNED_RADIX_256_F32_MAX_TERMS,
} from "./webgpu-exact-f32-reduction";
import { octreeLosassoHierarchyWGSL } from "./webgpu-octree-losasso-hierarchy.wgsl";
import type {
  OctreeLosassoVCycleHierarchySource,
  OctreeLosassoVCycleLevelSource,
  OctreeLosassoVCycleTransferSource,
} from "./webgpu-octree-losasso-vcycle";
import type { PassBroker } from "./webgpu-pass-broker";

export interface WebGPUOctreeLosassoHierarchyOptions {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  readonly physicalCellSize: readonly [number, number, number];
  readonly finest: OctreeLosassoVCycleLevelSource;
}

export interface WebGPUOctreeLosassoHierarchyCandidate {
  readonly leafHeaders: GPUBuffer;
  readonly finestControl: GPUBuffer;
  readonly finestFaces: GPUBuffer;
}

interface OwnedLevel extends OctreeLosassoVCycleLevelSource {
  readonly cells: GPUBuffer;
}

interface OwnedTransfer extends OctreeLosassoVCycleTransferSource {
  readonly directory: GPUBuffer;
  readonly scratch: GPUBuffer;
  readonly params: GPUBuffer;
}

const ENTRY_POINTS = [
  "extractLosassoFinestCells",
  "buildLosassoParentRows",
  "finishLosassoParentRows",
  "buildLosassoCoarseFaces",
  "buildLosassoCoarseCSR",
] as const;

function positive(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) {
    throw new RangeError(`Losasso hierarchy ${label} must be a positive u32`);
  }
  return value;
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

function parameterData(options: WebGPUOctreeLosassoHierarchyOptions,
  targetSpan: number, directoryCapacity: number): ArrayBuffer {
  const words = new Uint32Array(12);
  words.set([...options.dimensions, targetSpan], 0);
  words.set([options.rowCapacity, options.rowCapacity, options.faceCapacity,
    directoryCapacity], 4);
  new Float32Array(words.buffer).set([...options.physicalCellSize, 0], 8);
  return words.buffer;
}

/**
 * GPU-owned geometric hierarchy built only from compact leaf rows and their
 * first-order axis faces. Each transition groups rows in the next dyadic cell,
 * drops faces internal to that aggregate, and recomputes parent-centre
 * distances while retaining cut/open area patches. No Power authority enters
 * this allocation or publication path.
 */
export class WebGPUOctreeLosassoHierarchyPublisher {
  readonly hierarchy: OctreeLosassoVCycleHierarchySource;
  readonly initializationTasks: readonly { label: string; run: () => Promise<void> }[];
  readonly allocatedBytes: number;

  private readonly finestCells: GPUBuffer;
  private readonly ownedLevels: readonly OwnedLevel[];
  private readonly ownedTransfers: readonly OwnedTransfer[];
  private pipelines?: Readonly<Record<typeof ENTRY_POINTS[number], GPUComputePipeline>>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    private readonly options: WebGPUOctreeLosassoHierarchyOptions) {
    const rows = positive(options.rowCapacity, "row capacity");
    const faces = positive(options.faceCapacity, "face capacity");
    const [nx, ny, nz] = options.dimensions;
    [nx, ny, nz].forEach((value, axis) => positive(value, `dimension ${axis}`));
    const maximumLeafSize = positive(options.maximumLeafSize, "maximum leaf size");
    if ((maximumLeafSize & (maximumLeafSize - 1)) !== 0) {
      throw new RangeError("Losasso hierarchy needs a dyadic maximum leaf size");
    }
    if (options.dimensions.some((value) => value % maximumLeafSize !== 0)) {
      throw new RangeError("Losasso hierarchy dimensions must be divisible by maximum leaf size");
    }
    if (rows > SIGNED_RADIX_256_F32_MAX_TERMS) {
      throw new RangeError("Losasso hierarchy row capacity exceeds exact restriction capacity");
    }
    for (const [axis, size] of options.physicalCellSize.entries()) {
      if (!Number.isFinite(size) || size <= 0) {
        throw new RangeError(`Losasso hierarchy cell size ${axis} must be positive and finite`);
      }
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      | GPUBufferUsage.COPY_DST;
    const indirect = storage | GPUBufferUsage.INDIRECT;
    const make = (label: string, size: number, usage = storage) => device.createBuffer({
      label: `Losasso hierarchy ${label}`, size: Math.max(16, size), usage,
    });
    this.finestCells = make("L0 row geometry", rows * 16);
    const levels: OwnedLevel[] = [];
    const transfers: OwnedTransfer[] = [];
    const directoryCapacity = nextPowerOfTwo(2 * rows);
    const exact = planSignedRadix256F32Reduction(rows, 1);
    let fineDispatch = options.finest.rowDispatch;
    for (let targetSpan = 2, level = 1; targetSpan <= maximumLeafSize;
      targetSpan *= 2, level += 1) {
      const control = make(`L${level} control`, OCTREE_LOSASSO_CONTROL_WORDS * 4);
      const rowDispatch = make(`L${level} row dispatch`, 12, indirect);
      const ownedLevel: OwnedLevel = Object.freeze({
        rowCapacity: rows,
        control,
        rowFaceOffsets: make(`L${level} row offsets`, (rows + 1) * 4),
        rowFaces: make(`L${level} row faces`, 2 * faces * 4),
        faces: make(`L${level} axis faces`, faces * OCTREE_LOSASSO_FACE_BYTES),
        rowDispatch,
        cells: make(`L${level} row geometry`, rows * 16),
      });
      const params = make(`L${level} publication parameters`, 48,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(params, 0,
        parameterData(options, targetSpan, directoryCapacity));
      transfers.push(Object.freeze({
        fineParents: make(`L${level - 1} parent rows`, rows * 4),
        fineVolumes: make(`L${level - 1} voxel volumes`, rows * 4),
        coarseInverseVolumes: make(`L${level} inverse wet volumes`, rows * 4),
        restrictionPartials: make(`L${level} exact restriction limbs`, exact.byteLength),
        fineRowDispatch: fineDispatch,
        directory: make(`L${level} parent directory`, directoryCapacity * 8),
        scratch: make(`L${level} integer volume and CSR scratch`, rows * 4),
        params,
      }));
      levels.push(ownedLevel);
      fineDispatch = rowDispatch;
    }
    this.ownedLevels = Object.freeze(levels);
    this.ownedTransfers = Object.freeze(transfers);
    this.hierarchy = Object.freeze({
      levels: Object.freeze([options.finest, ...levels]),
      transfers: Object.freeze([...transfers]),
    });
    this.allocatedBytes = this.finestCells.size
      + [...levels.flatMap((level) => [level.control, level.rowFaceOffsets,
        level.rowFaces, level.faces, level.rowDispatch, level.cells]),
      ...transfers.flatMap((transfer) => [transfer.fineParents, transfer.fineVolumes,
        transfer.coarseInverseVolumes, transfer.restrictionPartials,
        transfer.directory, transfer.scratch, transfer.params])]
        .filter((buffer, index, all) => all.indexOf(buffer) === index)
        .reduce((sum, buffer) => sum + buffer.size, 0);
    this.initializationTasks = [{ label: "Compile reduced Losasso hierarchy publisher",
      run: () => this.initialize() }];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.pipelines) return;
    const shaderModule = this.device.createShaderModule({
      label: "Losasso reduced hierarchy publication shader",
      code: octreeLosassoHierarchyWGSL,
    });
    const entries = await Promise.all(ENTRY_POINTS.map(async (entryPoint) => [entryPoint,
      await this.device.createComputePipelineAsync({
        label: `Losasso hierarchy - ${entryPoint}`,
        layout: "auto", compute: { module: shaderModule, entryPoint },
      })] as const));
    this.pipelines = Object.fromEntries(entries) as
      Record<typeof ENTRY_POINTS[number], GPUComputePipeline>;
  }

  encodeCandidatePublication(broker: PassBroker,
    candidate: WebGPUOctreeLosassoHierarchyCandidate): void {
    this.assertLive();
    if (!this.pipelines) throw new Error("Losasso hierarchy publisher is not initialized");
    // The factor-1 lane is a useful compact diagnostic of the same reduced
    // operator. Its finest grid is already the complete one-level hierarchy,
    // so there is no transfer geometry to publish.
    if (this.ownedLevels.length === 0) return;
    let fineControl = candidate.finestControl;
    let fineCells = this.finestCells;
    let fineFaces = candidate.finestFaces;
    const extract = this.pipelines.extractLosassoFinestCells;
    const extractGroup = this.device.createBindGroup({
      label: "Losasso hierarchy finest geometry bindings",
      layout: extract.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.ownedTransfers[0]!.params } },
        { binding: 1, resource: { buffer: candidate.leafHeaders } },
        { binding: 2, resource: { buffer: fineControl } },
        { binding: 3, resource: { buffer: fineCells } },
      ],
    });
    const pass = broker.compute({ label: "Losasso hierarchy - publish geometric levels" });
    pass.setPipeline(extract); pass.setBindGroup(0, extractGroup);
    pass.dispatchWorkgroups(Math.ceil(this.options.rowCapacity / 64));
    for (let index = 0; index < this.ownedLevels.length; index += 1) {
      const coarse = this.ownedLevels[index]!;
      const transfer = this.ownedTransfers[index]!;
      this.encodeTransition(pass, { fineControl, fineCells, fineFaces, coarse, transfer });
      fineControl = coarse.control; fineCells = coarse.cells; fineFaces = coarse.faces;
    }
  }

  private encodeTransition(pass: GPUComputePassEncoder, input: {
    readonly fineControl: GPUBuffer; readonly fineCells: GPUBuffer;
    readonly fineFaces: GPUBuffer; readonly coarse: OwnedLevel;
    readonly transfer: OwnedTransfer;
  }): void {
    const buffers = [input.transfer.params, input.fineControl, input.fineCells,
      input.fineFaces, input.coarse.control, input.coarse.cells, input.coarse.faces,
      input.coarse.rowFaceOffsets, input.coarse.rowFaces, input.coarse.rowDispatch,
      input.transfer.fineParents, input.transfer.fineVolumes,
      input.transfer.coarseInverseVolumes, input.transfer.directory,
      input.transfer.scratch];
    const bindingBuffers = new Map<number, GPUBuffer>([
      [0, buffers[0]!], [2, buffers[1]!], [3, buffers[2]!], [4, buffers[3]!],
      [5, buffers[4]!], [6, buffers[5]!], [7, buffers[6]!], [8, buffers[7]!],
      [9, buffers[8]!], [10, buffers[9]!], [11, buffers[10]!], [12, buffers[11]!],
      [13, buffers[12]!], [14, buffers[13]!], [15, buffers[14]!],
    ]);
    const bindings: Readonly<Record<Exclude<typeof ENTRY_POINTS[number],
      "extractLosassoFinestCells">, readonly number[]>> = {
        buildLosassoParentRows: [0, 2, 3, 5, 6, 11, 12, 14, 15],
        finishLosassoParentRows: [5, 10, 13, 15],
        buildLosassoCoarseFaces: [0, 2, 3, 4, 5, 6, 7, 11],
        buildLosassoCoarseCSR: [0, 5, 7, 8, 9, 10, 15],
      };
    const transitionEntryPoints = ENTRY_POINTS.slice(1) as readonly
      Exclude<typeof ENTRY_POINTS[number], "extractLosassoFinestCells">[];
    for (const entryPoint of transitionEntryPoints) {
      const pipeline = this.pipelines![entryPoint];
      const group = this.device.createBindGroup({
        label: `Losasso hierarchy bindings - ${entryPoint}`,
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings[entryPoint].map((binding) => ({ binding,
          resource: { buffer: bindingBuffers.get(binding)! } })),
      });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(entryPoint === "finishLosassoParentRows"
        ? Math.ceil(this.options.rowCapacity / 64) : 1);
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const buffers = [this.finestCells,
      ...this.ownedLevels.flatMap((level) => [level.control, level.rowFaceOffsets,
        level.rowFaces, level.faces, level.rowDispatch, level.cells]),
      ...this.ownedTransfers.flatMap((transfer) => [transfer.fineParents,
        transfer.fineVolumes, transfer.coarseInverseVolumes,
        transfer.restrictionPartials,
        transfer.directory, transfer.scratch, transfer.params])];
    for (const buffer of buffers.filter((value, index, all) => all.indexOf(value) === index)) {
      buffer.destroy();
    }
    this.pipelines = undefined;
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso hierarchy publisher is destroyed");
  }
}
