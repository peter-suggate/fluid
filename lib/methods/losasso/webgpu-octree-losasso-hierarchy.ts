import { OCTREE_LOSASSO_CONTROL_WORDS } from
  "./octree-losasso-operator";
import { octreeLosassoHierarchyWGSL } from "./webgpu-octree-losasso-hierarchy.wgsl";
import { OCTREE_LOSASSO_FRAME_ARENA_MAGIC, OCTREE_LOSASSO_FRAME_ARENA_VERSION,
  octreeLosassoArenaView, planOctreeLosassoFrameArenas,
  type OctreeLosassoBufferView, type OctreeLosassoFrameArenaPlan } from
  "./webgpu-octree-losasso-frame-arena";
import type {
  OctreeLosassoVCycleHierarchySource,
  OctreeLosassoVCycleLevelSource,
  OctreeLosassoVCycleTransferSource,
} from "./webgpu-octree-losasso-vcycle";
import type { PassBroker } from "../../core/webgpu-pass-broker";

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
  readonly finestRowFaceOffsets: GPUBuffer;
  readonly finestRowFaces: GPUBuffer;
  readonly finestFaces: GPUBuffer;
}

interface OwnedLevel extends OctreeLosassoVCycleLevelSource {
  readonly cells: GPUBuffer;
  readonly coarseFaceSources: GPUBuffer;
  readonly coarseFaceSourceOffsets: GPUBuffer;
}

interface OwnedTransfer extends OctreeLosassoVCycleTransferSource {
  readonly directory: GPUBuffer;
  readonly params: GPUBuffer;
}

/** Small dense algebraic bottom retained by the compact row-pair hierarchy. */
export const MINIMUM_COARSE_LEVEL_ROWS = 8;

const ENTRY_POINTS = [
  "extractLosassoFinestCells",
  "publishLosassoFinestDirectedCSR",
  "buildLosassoParentRows",
  "buildLosassoParentRowsSmall",
  "buildLosassoCoarseFaces",
  "buildLosassoCoarseCSR",
  "buildLosassoCoarseCSRSmall",
  "publishLosassoCoarseDirectedCSR",
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
  targetSpan: number, directoryCapacity: number, fusedBaseWords: number,
  sourceIsAlgebraic: boolean, edgeDirectoryCapacity: number,
  coarseRowCapacity = options.rowCapacity,
  directedEdgeCapacity = 2 * options.faceCapacity,
  fineRowCapacity = options.rowCapacity): ArrayBuffer {
  const words = new Uint32Array(16);
  words.set([...options.dimensions, targetSpan], 0);
  words.set([fineRowCapacity, coarseRowCapacity, directedEdgeCapacity,
    directoryCapacity], 4);
  new Float32Array(words.buffer).set([...options.physicalCellSize, 0], 8);
  words[12] = fusedBaseWords;
  words[13] = sourceIsAlgebraic ? 1 : 0;
  words[14] = edgeDirectoryCapacity;
  words[15] = options.faceCapacity;
  return words.buffer;
}

/** GPU-owned Galerkin hierarchy of unique algebraic row-pair edges. */
export class WebGPUOctreeLosassoHierarchyPublisher {
  readonly hierarchy: OctreeLosassoVCycleHierarchySource;
  readonly arenaPlan: OctreeLosassoFrameArenaPlan;
  readonly initializationTasks: readonly { label: string; run: () => Promise<void> }[];
  readonly allocatedBytes: number;

  private readonly finestCells: GPUBuffer;
  private readonly fusedArena: GPUBuffer;
  private readonly frameArena: GPUBuffer;
  private readonly controlArena: GPUBuffer;
  private readonly edgeScratch: GPUBuffer;
  private readonly finestParams: GPUBuffer;
  private frameViewCache?: Readonly<{
    rightHandSide: OctreeLosassoBufferView; diagonal: OctreeLosassoBufferView;
    pressureA: OctreeLosassoBufferView; pressureB: OctreeLosassoBufferView;
  }>;
  private readonly ownedLevels: readonly OwnedLevel[];
  private readonly ownedTransfers: readonly OwnedTransfer[];
  private readonly bindGroupCache = new Map<GPUComputePipeline, {
    readonly bindings: readonly number[]; readonly buffers: readonly GPUBuffer[];
    readonly group: GPUBindGroup;
  }[]>();
  private pipelines?: Readonly<Record<typeof ENTRY_POINTS[number], GPUComputePipeline>>;
  private destroyed = false;

  get frameViews(): Readonly<{
    rightHandSide: OctreeLosassoBufferView;
    diagonal: OctreeLosassoBufferView;
    pressureA: OctreeLosassoBufferView;
    pressureB: OctreeLosassoBufferView;
  }> {
    const frame = this.arenaPlan.frame;
    return this.frameViewCache ??= Object.freeze({
      rightHandSide: octreeLosassoArenaView(this.frameArena, frame.rightHandSide),
      diagonal: octreeLosassoArenaView(this.frameArena, frame.diagonal),
      pressureA: octreeLosassoArenaView(this.frameArena, frame.pressureA),
      pressureB: octreeLosassoArenaView(this.frameArena, frame.pressureB),
    });
  }

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
    const edgeDirectoryCapacity = nextPowerOfTwo(2 * faces);
    this.edgeScratch = make("algebraic edge compilation scratch",
      (2 * edgeDirectoryCapacity + faces + Math.max(rows, faces) + 1) * 4);
    // Algebraic reduction folds all fine patches connecting the same row pair,
    // so deeper levels no longer retain tens of thousands of duplicate faces.
    // Keep contracting until the bottom is genuinely small.
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
    const levelRowCapacities = [rows];
    for (let level = 1, targetSpan = 2; level <= transitionCount;
      level += 1, targetSpan *= 2) {
      levelRowCapacities.push(levelRowCapacityAt(targetSpan));
    }
    this.arenaPlan = planOctreeLosassoFrameArenas({ rowCapacity: rows,
      faceCapacity: faces, levelRowCapacities });
    const packed = this.arenaPlan.operator;
    this.fusedArena = make("shared dense operator arena", packed.bufferBytes);
    this.frameArena = make("shared mutable pressure frame arena", this.arenaPlan.frame.bufferBytes);
    device.queue.writeBuffer(this.frameArena, this.arenaPlan.frame.header.byteOffset,
      Uint32Array.of(OCTREE_LOSASSO_FRAME_ARENA_MAGIC,
        OCTREE_LOSASSO_FRAME_ARENA_VERSION));
    this.controlArena = make("shared pressure control and dispatch arena",
      this.arenaPlan.control.bufferBytes, indirect);
    device.queue.writeBuffer(this.controlArena, this.arenaPlan.control.header.byteOffset,
      Uint32Array.of(OCTREE_LOSASSO_FRAME_ARENA_MAGIC,
        OCTREE_LOSASSO_FRAME_ARENA_VERSION));
    device.queue.writeBuffer(this.fusedArena, 0, Uint32Array.of(
      OCTREE_LOSASSO_FRAME_ARENA_MAGIC, OCTREE_LOSASSO_FRAME_ARENA_VERSION,
      packed.banks[0].wordOffset, packed.banks[1].wordOffset));
    this.finestParams = make("L0 directed operator publication parameters", 64,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(this.finestParams, 0,
      parameterData(options, 1, directoryCapacity, packed.levelBasesWords[0]!,
        false, edgeDirectoryCapacity, packed.levels[0]!.rowCapacity,
        packed.levels[0]!.directedEdgeCapacity, packed.levels[0]!.rowCapacity));
    let fineDispatch = options.finest.rowDispatch;
    for (let targetSpan = 2, level = 1; level <= transitionCount;
      targetSpan *= 2, level += 1) {
      const control = make(`L${level} control`, OCTREE_LOSASSO_CONTROL_WORDS * 4);
      const rowDispatch = make(`L${level} row and face dispatch`, 24, indirect);
      const ownedLevel: OwnedLevel = Object.freeze({
        rowCapacity: rows,
        control,
        rowFaceOffsets: make(`L${level} algebraic row offsets`, (rows + 1) * 4),
        rowFaces: make(`L${level} row edge ids`, 2 * faces * 4),
        faces: make(`L${level} algebraic row-pair edges`, faces * 16),
        rowDispatch,
        cells: make(`L${level} row geometry`, rows * 16),
        coarseFaceSources: make(`L${level} fine-face provenance`, faces * 4),
        coarseFaceSourceOffsets: make(`L${level} edge-source offsets`, (faces + 1) * 4),
      });
      const params = make(`L${level} publication parameters`, 64,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(params, 0,
        parameterData(options, targetSpan, directoryCapacity,
          packed.levelBasesWords[level]!, level > 1, edgeDirectoryCapacity,
          packed.levels[level]!.rowCapacity, packed.levels[level]!.directedEdgeCapacity,
          packed.levels[level - 1]!.rowCapacity));
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
      fusedSubL0: Object.freeze({ arena: this.fusedArena,
        frameArena: this.frameArena, controlArena: this.controlArena,
        arenaPlan: this.arenaPlan,
        rowCapacity: rows, faceCapacity: faces,
        acceptedBankWordOffset: packed.banks[0].wordOffset,
        candidateBankWordOffset: packed.banks[1].wordOffset,
        levelLayouts: Object.freeze(packed.levels.map((layout, level) => Object.freeze({
          baseWords: packed.levelBasesWords[level]!,
          controlOffsetWords: layout.controlOffsetWords,
          rowOffsetsOffsetWords: layout.rowOffsetsOffsetWords,
          directedEdgesOffsetWords: layout.directedEdgesOffsetWords,
          parentsOffsetWords: layout.parentsOffsetWords,
          childOffsetsOffsetWords: layout.childOffsetsOffsetWords,
          childListOffsetWords: layout.childListOffsetWords,
          directedEdgeCapacity: layout.directedEdgeCapacity,
        }))),
        levelRowCapacities: Object.freeze(levelRowCapacities) }),
    });
    this.allocatedBytes = this.finestCells.size + this.fusedArena.size
      + this.frameArena.size + this.controlArena.size + this.edgeScratch.size + this.finestParams.size
      + [...levels.flatMap((level) => [level.control, level.rowFaceOffsets,
        level.rowFaces, level.faces, level.rowDispatch, level.cells,
        level.coarseFaceSources, level.coarseFaceSourceOffsets]),
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
    const publishFinest = this.pipelines.publishLosassoFinestDirectedCSR;
    const publishFinestGroup = this.cachedBindGroup(publishFinest,
      "Losasso hierarchy finest directed CSR bindings", [0, 2, 4, 17, 20, 21],
      [this.finestParams, candidate.finestControl, candidate.finestFaces,
        this.fusedArena, candidate.finestRowFaceOffsets, candidate.finestRowFaces]);
    const publishPass = broker.compute({ label: "Losasso hierarchy - publish finest directed CSR" });
    publishPass.setPipeline(publishFinest); publishPass.setBindGroup(0, publishFinestGroup);
    publishPass.dispatchWorkgroups(Math.ceil((this.options.rowCapacity + 1) / 64));
    broker.copyBufferToBuffer(this.options.finest.rowDispatch, 0, this.controlArena,
      this.arenaPlan.control.rowDispatchByteOffset, 12);
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
    readonly rowFaceOffsets: GPUBuffer;
    readonly rowFaces: GPUBuffer;
    readonly faces: GPUBuffer;
  }): void {
    this.assertLive();
    if (!this.pipelines) throw new Error("Losasso hierarchy publisher is not initialized");
    const publishFinest = this.pipelines.publishLosassoFinestDirectedCSR;
    const publishFinestGroup = this.cachedBindGroup(publishFinest,
      "Losasso hierarchy finest directed CSR refresh bindings", [0, 2, 4, 17, 20, 21],
      [this.finestParams, finest.control, finest.faces, this.fusedArena,
        finest.rowFaceOffsets, finest.rowFaces]);
    const finestPass = broker.compute({ label: "Losasso hierarchy - refresh finest directed CSR" });
    finestPass.setPipeline(publishFinest); finestPass.setBindGroup(0, publishFinestGroup);
    finestPass.dispatchWorkgroups(Math.ceil((this.options.rowCapacity + 1) / 64));
    broker.copyBufferToBuffer(this.options.finest.rowDispatch, 0, this.controlArena,
      this.arenaPlan.control.rowDispatchByteOffset, 12);
    if (this.ownedLevels.length === 0) return;
    const pipeline = this.pipelines.refreshLosassoCoarseFaces;
    const pass = broker.compute({ label: "Losasso hierarchy - refresh face coefficients" });
    let fineControl = finest.control;
    let fineFaces = finest.faces;
    for (const [index, coarse] of this.ownedLevels.entries()) {
      const bindings = [0, 2, 4, 5, 7, 16, 17, 19];
      const group = this.cachedBindGroup(pipeline,
        "Losasso hierarchy coefficient refresh bindings", bindings,
        [this.ownedTransfers[index]!.params, fineControl, fineFaces,
          coarse.control, coarse.faces, coarse.coarseFaceSources, this.fusedArena,
          coarse.coarseFaceSourceOffsets]);
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroupsIndirect(coarse.rowDispatch, 12);
      const publish = this.pipelines.publishLosassoCoarseDirectedCSR;
      const publishGroup = this.cachedBindGroup(publish,
        "Losasso hierarchy coarse directed CSR refresh bindings",
        [0, 5, 7, 8, 9, 17], [this.ownedTransfers[index]!.params,
          coarse.control, coarse.faces, coarse.rowFaceOffsets, coarse.rowFaces,
          this.fusedArena]);
      pass.setPipeline(publish); pass.setBindGroup(0, publishGroup);
      pass.dispatchWorkgroupsIndirect(coarse.rowDispatch, 0);
      fineControl = coarse.control;
      fineFaces = coarse.faces;
    }
  }

  /** Snapshot the compact solver epoch into the shared control page. Row
   * fields are already producer-owned frame-arena views and need no staging. */
  encodePressureFramePublication(broker: PassBroker, input: {
    readonly acceptedAuthority: GPUBuffer;
  }): void {
    this.assertLive();
    broker.copyBufferToBuffer(input.acceptedAuthority, 0, this.controlArena,
      32 * 4, 7 * 4);
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
      input.coarse.coarseFaceSources, this.fusedArena, this.edgeScratch,
      input.coarse.coarseFaceSourceOffsets];
    const bindingBuffers = new Map<number, GPUBuffer>([
      [0, buffers[0]!], [2, buffers[1]!], [3, buffers[2]!], [4, buffers[3]!],
      [5, buffers[4]!], [6, buffers[5]!], [7, buffers[6]!], [8, buffers[7]!],
      [9, buffers[8]!], [10, buffers[9]!], [11, buffers[10]!], [12, buffers[11]!],
      [13, buffers[12]!], [14, buffers[13]!], [16, buffers[14]!],
      [17, buffers[15]!], [18, buffers[16]!], [19, buffers[17]!],
    ]);
    const bindings: Readonly<Record<Exclude<typeof ENTRY_POINTS[number],
      "extractLosassoFinestCells" | "publishLosassoFinestDirectedCSR">, readonly number[]>> = {
        buildLosassoParentRows: [0, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17],
        buildLosassoParentRowsSmall: [0, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17],
        buildLosassoCoarseFaces: [0, 2, 4, 5, 7, 10, 11, 16, 18, 19],
        buildLosassoCoarseCSR: [0, 2, 5, 7, 8, 9, 10, 17, 18],
        buildLosassoCoarseCSRSmall: [0, 2, 5, 7, 8, 9, 10, 17, 18],
      refreshLosassoCoarseFaces: [0, 2, 4, 5, 7, 16, 17, 19],
      refreshLosassoReusedCoarseFaces: [0, 2, 4, 5, 7, 16, 17, 19],
      publishLosassoCoarseDirectedCSR: [0, 5, 7, 8, 9, 17],
    };
    const transitionEntryPoints = [this.options.rowCapacity <= 4096
      ? "buildLosassoParentRowsSmall" : "buildLosassoParentRows", "buildLosassoCoarseFaces",
      this.options.rowCapacity <= 4096 ? "buildLosassoCoarseCSRSmall"
        : "buildLosassoCoarseCSR", "publishLosassoCoarseDirectedCSR"] as const;
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
      [0, 2, 4, 5, 7, 16, 17, 19], [input.transfer.params, input.fineControl,
        input.fineFaces, input.coarse.control, input.coarse.faces,
        input.coarse.coarseFaceSources, this.fusedArena,
        input.coarse.coarseFaceSourceOffsets]);
    const pass = broker.compute({ label:
      "Losasso hierarchy - refreshLosassoReusedCoarseFaces" });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(input.coarse.rowDispatch, 12);
    const publish = this.pipelines!.publishLosassoCoarseDirectedCSR;
    const publishGroup = this.cachedBindGroup(publish,
      "Losasso hierarchy exact-reuse directed CSR bindings", [0, 5, 7, 8, 9, 17],
      [input.transfer.params, input.coarse.control, input.coarse.faces,
        input.coarse.rowFaceOffsets, input.coarse.rowFaces, this.fusedArena]);
    pass.setPipeline(publish); pass.setBindGroup(0, publishGroup);
    pass.dispatchWorkgroupsIndirect(input.coarse.rowDispatch, 0);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const buffers = [this.finestCells, this.fusedArena, this.frameArena, this.controlArena,
      this.edgeScratch, this.finestParams,
      ...this.ownedLevels.flatMap((level) => [level.control, level.rowFaceOffsets,
        level.rowFaces, level.faces, level.rowDispatch, level.cells,
        level.coarseFaceSources, level.coarseFaceSourceOffsets]),
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
