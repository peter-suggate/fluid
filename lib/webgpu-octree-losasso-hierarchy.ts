import { OCTREE_LOSASSO_CONTROL_WORDS, OCTREE_LOSASSO_FACE_BYTES } from
  "./octree-losasso-operator";
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
  readonly coarseFaceSources: GPUBuffer;
}

interface OwnedTransfer extends OctreeLosassoVCycleTransferSource {
  readonly directory: GPUBuffer;
  readonly params: GPUBuffer;
}

/**
 * Smallest coarse level worth building.
 *
 * A level below this has too few rows to relax with any parallelism while still
 * carrying tens of thousands of retained face patches, so it costs far more
 * than the extra contraction it buys. See the depth sweep in the constructor.
 */
export const MINIMUM_COARSE_LEVEL_ROWS = 1_024;

const ENTRY_POINTS = [
  "extractLosassoFinestCells",
  "buildLosassoParentRows",
  "buildLosassoParentRowsSmall",
  "buildLosassoCoarseFaces",
  "buildLosassoCoarseCSR",
  "buildLosassoCoarseCSRSmall",
  "refreshLosassoCoarseFaces",
  "refreshLosassoReusedCoarseFaces",
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
  targetSpan: number, directoryCapacity: number, fusedBaseWords: number): ArrayBuffer {
  const words = new Uint32Array(16);
  words.set([...options.dimensions, targetSpan], 0);
  words.set([options.rowCapacity, options.rowCapacity, options.faceCapacity,
    directoryCapacity], 4);
  new Float32Array(words.buffer).set([...options.physicalCellSize, 0], 8);
  words[12] = fusedBaseWords;
  return words.buffer;
}

function fusedLayout(rowCapacity: number, faceCapacity: number) {
  const controlOffsetWords = 0;
  const rowOffsetsOffsetWords = controlOffsetWords + OCTREE_LOSASSO_CONTROL_WORDS;
  const rowFacesOffsetWords = rowOffsetsOffsetWords + rowCapacity + 1;
  const facesOffsetWords = rowFacesOffsetWords + 2 * faceCapacity;
  const parentsOffsetWords = facesOffsetWords + 8 * faceCapacity;
  const childOffsetsOffsetWords = parentsOffsetWords + rowCapacity;
  const childListOffsetWords = childOffsetsOffsetWords + rowCapacity + 1;
  const transitionStrideWords = childListOffsetWords + rowCapacity;
  return Object.freeze({ rowCapacity, faceCapacity, transitionStrideWords,
    controlOffsetWords, rowOffsetsOffsetWords, rowFacesOffsetWords,
    facesOffsetWords, parentsOffsetWords, childOffsetsOffsetWords,
    childListOffsetWords });
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
  private readonly fusedArena: GPUBuffer;
  private readonly ownedLevels: readonly OwnedLevel[];
  private readonly ownedTransfers: readonly OwnedTransfer[];
  private readonly bindGroupCache = new Map<GPUComputePipeline, {
    readonly bindings: readonly number[]; readonly buffers: readonly GPUBuffer[];
    readonly group: GPUBindGroup;
  }[]>();
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
    // No divisibility requirement. `parentCell` snaps a fine origin down to a
    // multiple of the parent span and keys it through a hash directory, so a
    // span that does not divide the domain simply produces one partial parent
    // against each far face -- representable, and priced by the ceil-rounded
    // capacity below. The guard that used to stand here was an allocation
    // convenience, and enforcing it upstream capped the leaf ceiling at
    // gcd(nx, ny, nz), which is what kept deep interiors uniformly fine.
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
    // Aggregation collapses rows eightfold per transition but only retains and
    // re-parents face patches, so face count -- which is what the smoother
    // actually walks -- decays far slower. Measured on the 128^3 lane
    // (2026-08-06): rows 125,067 -> 30,315 -> 6,883 -> 1,311 -> 200 -> 36 while
    // faces went 406,004 -> 243,572 -> 145,348 -> 78,132 -> 53,556 -> 42,585.
    // The bottom two levels therefore carry 96,000 faces spread over 236 rows
    // -- almost no parallelism and almost no coarsening -- and a depth sweep
    // priced them at ~460 ms of the frame to buy 27 iterations down to 17:
    //
    //   depth 1: 72 iterations, 395 ms   depth 3: 27 iterations, 389 ms
    //   depth 2: 44 iterations, 376 ms   depth 5: 17 iterations, 850 ms
    //
    // Stop where a level still has enough rows to relax in parallel. This keeps
    // the 128^3 hierarchy at depth 3, its measured optimum and a win against
    // the 453 ms this lane cost while the V-cycle was inert.
    // Ceil, not exact division: a span that does not divide the domain leaves a
    // partial parent against each far face, and that parent is a row like any
    // other. Flooring it would under-reserve exactly the rows the relaxed
    // ceiling introduces.
    const levelRowCapacityAt = (targetSpan: number) => Math.min(rows,
      Math.ceil(nx / targetSpan) * Math.ceil(ny / targetSpan) * Math.ceil(nz / targetSpan));
    let transitionCount = 0;
    for (let targetSpan = 2; targetSpan <= maximumLeafSize; targetSpan *= 2) {
      if (levelRowCapacityAt(targetSpan) < MINIMUM_COARSE_LEVEL_ROWS) break;
      transitionCount += 1;
    }
    const packed = fusedLayout(rows, faces);
    const levelRowCapacities = [rows];
    this.fusedArena = make("packed fused sub-L0 hierarchy",
      Math.max(1, transitionCount) * packed.transitionStrideWords * 4);
    let fineDispatch = options.finest.rowDispatch;
    for (let targetSpan = 2, level = 1; level <= transitionCount;
      targetSpan *= 2, level += 1) {
      levelRowCapacities.push(levelRowCapacityAt(targetSpan));
      const control = make(`L${level} control`, OCTREE_LOSASSO_CONTROL_WORDS * 4);
      const rowDispatch = make(`L${level} row and face dispatch`, 24, indirect);
      const ownedLevel: OwnedLevel = Object.freeze({
        rowCapacity: rows,
        control,
        rowFaceOffsets: make(`L${level} row offsets`, (rows + 1) * 4),
        rowFaces: make(`L${level} row faces`, 2 * faces * 4),
        faces: make(`L${level} axis faces`, faces * OCTREE_LOSASSO_FACE_BYTES),
        rowDispatch,
        cells: make(`L${level} row geometry`, rows * 16),
        coarseFaceSources: make(`L${level} fine-face provenance`, faces * 4),
      });
      const params = make(`L${level} publication parameters`, 64,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(params, 0,
        parameterData(options, targetSpan, directoryCapacity,
          (level - 1) * packed.transitionStrideWords));
      transfers.push(Object.freeze({
        fineParents: make(`L${level - 1} parent rows`, rows * 4),
        childOffsets: make(`L${level} child offsets`, (rows + 1) * 4),
        childList: make(`L${level} ascending child rows`, rows * 4),
        fineRowDispatch: fineDispatch,
        directory: make(`L${level} parent directory`, directoryCapacity * 8),
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
      fusedSubL0: Object.freeze({ arena: this.fusedArena, ...packed,
        levelRowCapacities: Object.freeze(levelRowCapacities) }),
    });
    this.allocatedBytes = this.finestCells.size + this.fusedArena.size
      + [...levels.flatMap((level) => [level.control, level.rowFaceOffsets,
        level.rowFaces, level.faces, level.rowDispatch, level.cells,
        level.coarseFaceSources]),
      ...transfers.flatMap((transfer) => [transfer.fineParents, transfer.childOffsets,
        transfer.childList,
        transfer.directory, transfer.params])]
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

  private cachedBindGroup(pipeline: GPUComputePipeline, label: string,
    bindings: readonly number[], buffers: readonly GPUBuffer[]): GPUBindGroup {
    const variants = this.bindGroupCache.get(pipeline) ?? [];
    const cached = variants.find((variant) => variant.bindings.length === bindings.length
      && variant.bindings.every((binding, index) => binding === bindings[index]
        && variant.buffers[index] === buffers[index]));
    if (cached) return cached.group;
    const stableBindings = [...bindings], stableBuffers = [...buffers];
    const group = this.device.createBindGroup({ label,
      layout: pipeline.getBindGroupLayout(0),
      entries: stableBindings.map((binding, index) =>
        ({ binding, resource: { buffer: stableBuffers[index]! } })),
    });
    variants.push({ bindings: stableBindings, buffers: stableBuffers, group });
    this.bindGroupCache.set(pipeline, variants);
    return group;
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
    const extractGroup = this.cachedBindGroup(extract,
      "Losasso hierarchy finest geometry bindings", [0, 1, 2, 3],
      [this.ownedTransfers[0]!.params, candidate.leafHeaders, fineControl, fineCells]);
    const pass = broker.compute({ label: "Losasso hierarchy - extractLosassoFinestCells" });
    pass.setPipeline(extract); pass.setBindGroup(0, extractGroup);
    pass.dispatchWorkgroups(Math.ceil(this.options.rowCapacity / 64));
    for (let index = 0; index < this.ownedLevels.length; index += 1) {
      const coarse = this.ownedLevels[index]!;
      const transfer = this.ownedTransfers[index]!;
      this.encodeTransition(broker, { fineControl, fineCells, fineFaces, coarse, transfer });
      this.encodeReusedTransitionRefresh(broker,
        { fineControl, fineCells, fineFaces, coarse, transfer });
      fineControl = coarse.control; fineCells = coarse.cells; fineFaces = coarse.faces;
    }
  }

  /** Refresh face coefficients while retaining the topology built for this epoch. */
  encodeCoefficientRefresh(broker: PassBroker, finest: {
    readonly control: GPUBuffer;
    readonly faces: GPUBuffer;
  }): void {
    this.assertLive();
    if (!this.pipelines) throw new Error("Losasso hierarchy publisher is not initialized");
    if (this.ownedLevels.length === 0) return;
    const pipeline = this.pipelines.refreshLosassoCoarseFaces;
    const pass = broker.compute({ label: "Losasso hierarchy - refresh face coefficients" });
    let fineControl = finest.control;
    let fineCells = this.finestCells;
    let fineFaces = finest.faces;
    for (const [index, coarse] of this.ownedLevels.entries()) {
      const bindings = [0, 2, 3, 4, 5, 6, 7, 16, 17];
      const group = this.cachedBindGroup(pipeline,
        "Losasso hierarchy coefficient refresh bindings", bindings,
        [this.ownedTransfers[index]!.params, fineControl, fineCells, fineFaces,
          coarse.control, coarse.cells, coarse.faces, coarse.coarseFaceSources, this.fusedArena]);
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroupsIndirect(coarse.rowDispatch, 12);
      fineControl = coarse.control;
      fineCells = coarse.cells;
      fineFaces = coarse.faces;
    }
  }

  private encodeTransition(broker: PassBroker, input: {
    readonly fineControl: GPUBuffer; readonly fineCells: GPUBuffer;
    readonly fineFaces: GPUBuffer; readonly coarse: OwnedLevel;
    readonly transfer: OwnedTransfer;
  }): void {
    const buffers = [input.transfer.params, input.fineControl, input.fineCells,
      input.fineFaces, input.coarse.control, input.coarse.cells, input.coarse.faces,
      input.coarse.rowFaceOffsets, input.coarse.rowFaces, input.coarse.rowDispatch,
      input.transfer.fineParents, input.transfer.childOffsets,
      input.transfer.childList, input.transfer.directory,
      input.coarse.coarseFaceSources, this.fusedArena];
    const bindingBuffers = new Map<number, GPUBuffer>([
      [0, buffers[0]!], [2, buffers[1]!], [3, buffers[2]!], [4, buffers[3]!],
      [5, buffers[4]!], [6, buffers[5]!], [7, buffers[6]!], [8, buffers[7]!],
      [9, buffers[8]!], [10, buffers[9]!], [11, buffers[10]!], [12, buffers[11]!],
      [13, buffers[12]!], [14, buffers[13]!], [16, buffers[14]!],
      [17, buffers[15]!],
    ]);
    const bindings: Readonly<Record<Exclude<typeof ENTRY_POINTS[number],
      "extractLosassoFinestCells">, readonly number[]>> = {
        buildLosassoParentRows: [0, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17],
        buildLosassoParentRowsSmall: [0, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17],
        buildLosassoCoarseFaces: [0, 2, 3, 4, 5, 6, 7, 11, 16, 17],
        buildLosassoCoarseCSR: [0, 2, 5, 7, 8, 9, 10, 17],
        buildLosassoCoarseCSRSmall: [0, 2, 5, 7, 8, 9, 10, 17],
      refreshLosassoCoarseFaces: [0, 2, 3, 4, 5, 6, 7, 16, 17],
      refreshLosassoReusedCoarseFaces: [0, 2, 3, 4, 5, 6, 7, 16, 17],
    };
    const transitionEntryPoints = [this.options.rowCapacity <= 4096
      ? "buildLosassoParentRowsSmall" : "buildLosassoParentRows", "buildLosassoCoarseFaces",
      this.options.rowCapacity <= 4096 ? "buildLosassoCoarseCSRSmall"
        : "buildLosassoCoarseCSR"] as const;
    for (const entryPoint of transitionEntryPoints) {
      const pipeline = this.pipelines![entryPoint];
      const selected = bindings[entryPoint];
      const group = this.cachedBindGroup(pipeline,
        `Losasso hierarchy bindings - ${entryPoint}`, selected,
        selected.map((binding) => bindingBuffers.get(binding)!));
      const pass = broker.compute({ label: `Losasso hierarchy - ${entryPoint}` });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
    }
  }

  private encodeReusedTransitionRefresh(broker: PassBroker, input: {
    readonly fineControl: GPUBuffer; readonly fineCells: GPUBuffer;
    readonly fineFaces: GPUBuffer; readonly coarse: OwnedLevel;
    readonly transfer: OwnedTransfer;
  }): void {
    const pipeline = this.pipelines!.refreshLosassoReusedCoarseFaces;
    const group = this.cachedBindGroup(pipeline,
      "Losasso hierarchy exact-reuse coefficient refresh bindings",
      [0, 2, 3, 4, 5, 6, 7, 16, 17], [input.transfer.params, input.fineControl,
        input.fineCells, input.fineFaces, input.coarse.control, input.coarse.cells,
        input.coarse.faces, input.coarse.coarseFaceSources, this.fusedArena]);
    const pass = broker.compute({ label:
      "Losasso hierarchy - refreshLosassoReusedCoarseFaces" });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(input.coarse.rowDispatch, 12);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const buffers = [this.finestCells, this.fusedArena,
      ...this.ownedLevels.flatMap((level) => [level.control, level.rowFaceOffsets,
        level.rowFaces, level.faces, level.rowDispatch, level.cells,
        level.coarseFaceSources]),
      ...this.ownedTransfers.flatMap((transfer) => [transfer.fineParents,
        transfer.childOffsets, transfer.childList,
        transfer.directory, transfer.params])];
    for (const buffer of buffers.filter((value, index, all) => all.indexOf(value) === index)) {
      buffer.destroy();
    }
    this.pipelines = undefined;
    this.bindGroupCache.clear();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso hierarchy publisher is destroyed");
  }
}
