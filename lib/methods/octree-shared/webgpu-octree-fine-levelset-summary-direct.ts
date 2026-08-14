import type { FineLevelSetBrickPlan, FineLevelSetFactor } from "./octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL } from "../../core/webgpu-fine-levelset-dispatch";
import { fineLevelSetPackedSampleWGSL } from "../../core/fine-levelset-packed-sample";
import type { WebGPUFineLevelSetBrickSource } from "../../core/levelset-consumer-abi";
import type { FineLevelSetPageDeltaLayout } from "./webgpu-octree-fine-levelset-topology";
import { PassBroker } from "../../core/webgpu-pass-broker";
import type { GPUInitializationTask } from "../../core/gpu-initialization";

export const FINE_LEVELSET_SUMMARY_VALID = 0x8000_0000;
export const FINE_LEVELSET_SUMMARY_COARSE_AUTHORITY = 0x8000_0000;
export const FINE_LEVELSET_SUMMARY_CENTER_COMPLETE = 0x3fc0_0000;
export const FINE_LEVELSET_SUMMARY_SIZING_REFINEMENT = 0x4000_0000;
export const FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE = 32;
export const FINE_LEVELSET_SUMMARY_ENTRY_WORDS = 12;
export const FINE_LEVELSET_SUMMARY_ERROR = Object.freeze({ capacity: 1, staleGeneration: 4,
  nonfinite: 8 } as const);

export const FINE_LEVELSET_SUMMARY_CONSUMERS = Object.freeze([
  Object.freeze({ owner: "pressureRefinementEvidence", classification: "simulation-critical",
    access: "direct hierarchy key to active-mip rank" }),
  Object.freeze({ owner: "currentPressureOwnerWet", classification: "simulation-critical",
    access: "direct hierarchy key to active-mip rank" }),
  Object.freeze({ owner: "globalFineSummaryDirectory", classification: "diagnostics-only",
    access: "published raw buffer readback" }),
  Object.freeze({ owner: "globalFineSummaryDebug", classification: "diagnostics-only",
    access: "publication header and bounded internal state" }),
] as const);

export interface FineLevelSetGPUSummaryPlan {
  readonly maximumResidentBricks: number; readonly maximumLevel: number;
  readonly fineEntryCapacity: number; readonly coarseEntryCapacity: number; readonly entryCapacity: number;
  readonly hierarchyKeyCapacity: number; readonly directoryPageSize: number;
  readonly hierarchyTopLevelPages: number; readonly directoryPageCapacity: number;
  readonly directoryWords: number;
  readonly directoryBytes: number; readonly fineEntriesBytes: number;
  readonly keyStateBytes: number; readonly rankStateBytes: number; readonly pageStateBytes: number;
  readonly workStateBytes: number; readonly indirectBytes: number;
  readonly parameterBytes: number; readonly allocatedBytes: number;
  readonly levelOffsets: readonly number[]; readonly levelDimensions: readonly (readonly [number, number, number])[];
}

export interface FineLevelSetSummaryPageDeltaSource {
  readonly buffer: GPUBuffer;
  readonly layout: Pick<FineLevelSetPageDeltaLayout, "changedKeysOffsetWords">;
}

export interface FineLevelSetSummaryCoarseSource {
  readonly directory: GPUBuffer; readonly control: GPUBuffer; readonly delta: GPUBuffer;
  readonly deltaHeaderWords: 16; readonly deltaRecordWords: 4;
}

export interface FineLevelSetSummaryLeafLookup {
  readonly level: number; readonly key: number; readonly brickSide: number;
  readonly expectedBrickCount: number; readonly expectedSampleCount: number;
}

export function planFineLevelSetSummaryLeafLookup(
  baseDimensions: readonly [number, number, number],
  finestCellDimensions: readonly [number, number, number],
  origin: readonly [number, number, number], size: number, samplesPerBrick = 64,
  fineFactor?: FineLevelSetFactor,
): FineLevelSetSummaryLeafLookup {
  if (!Number.isInteger(size) || size < 1 || (size & (size - 1)) !== 0) {
    throw new RangeError("Fine-summary leaf size must be a positive power of two");
  }
  const factorOneShape = baseDimensions.every((value, axis) =>
    value === Math.ceil(finestCellDimensions[axis] / 4));
  const factorOne = fineFactor === 1 || (fineFactor === undefined && factorOneShape
    && baseDimensions.some((value, axis) => value < finestCellDimensions[axis]));
  if (factorOne && !factorOneShape) {
    throw new RangeError("Factor-1 fine-summary dimensions must contain four finest cells per B4 leaf");
  }
  const ratios = baseDimensions.map((value, axis) => value / finestCellDimensions[axis]);
  const bricksPerCell = ratios[0];
  if (!factorOne && (!Number.isInteger(bricksPerCell) || bricksPerCell < 1
    || ratios.some((value) => value !== bricksPerCell))) {
    throw new RangeError("Fine-summary lattice must be factor-1 or contain an equal integer brick count per finest cell");
  }
  // A factor-1 B4 summary leaf covers four finest cells per axis. Unit and
  // size-2 pressure leaves therefore share its conservative interval; size 4
  // is the first pressure leaf whose geometry exactly matches that node.
  const brickSide = factorOne ? Math.max(1, size / 4) : size * bricksPerCell;
  if (!Number.isSafeInteger(brickSide) || (brickSide & (brickSide - 1)) !== 0) {
    throw new RangeError("Fine-summary leaf brick span must be a safe power of two");
  }
  const level = Math.log2(brickSide); let levelOffset = 0;
  let dimensions = [...baseDimensions] as [number, number, number];
  for (let current = 0; current < level; current += 1) {
    levelOffset += dimensions[0] * dimensions[1] * dimensions[2];
    dimensions = dimensions.map((value) => Math.ceil(value / 2)) as [number, number, number];
  }
  const brickOrigin = origin.map((value) => factorOne ? Math.floor(value / 4) : value * bricksPerCell);
  if (factorOne && size >= 4 && origin.some((value) => value % size !== 0)) {
    throw new RangeError("Octree leaf origin is not aligned to its factor-1 fine-summary node");
  }
  const coordinate = brickOrigin.map((value) => value / brickSide);
  if (coordinate.some((value) => !Number.isInteger(value))) {
    throw new RangeError("Octree leaf origin is not aligned to its fine-summary node");
  }
  const key = levelOffset + coordinate[0] + dimensions[0] * (coordinate[1] + dimensions[1] * coordinate[2]);
  const expectedBrickCount = brickSide ** 3;
  return { level, key, brickSide, expectedBrickCount, expectedSampleCount: expectedBrickCount * samplesPerBrick };
}

export type FineLevelSetSummaryRefinementSignal = "refine" | "complete-no-crossing" | "invalid";

/** CPU mirror of the production one-load key-to-rank lookup. */
export function fineLevelSetSummaryDirectEntryBase(words: Uint32Array, key: number): number | undefined {
  if (words.length < 16 || words[9] !== FINE_LEVELSET_SUMMARY_VALID || words[0] !== 0) return undefined;
  const hierarchyKeyCapacity = words[10]!; const entryCapacity = words[3]!;
  const entryOffset = words[8]!; const pageSize = words[14]!; const topLevelPages = words[15]!;
  if (key < 0 || key >= hierarchyKeyCapacity || pageSize !== FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE
    || topLevelPages !== Math.ceil(hierarchyKeyCapacity / pageSize)
    || 16 + topLevelPages > entryOffset || entryOffset > words.length
    || (entryOffset - 16 - topLevelPages) % pageSize !== 0
    || entryCapacity > Math.floor((words.length - entryOffset) / FINE_LEVELSET_SUMMARY_ENTRY_WORDS)
    || 16 + Math.floor(key / pageSize) >= words.length) return undefined;
  const pageRankPlusOne = words[16 + Math.floor(key / pageSize)]!;
  const pagePoolOffset = 16 + topLevelPages;
  const pageCapacity = (entryOffset - pagePoolOffset) / pageSize;
  if (pageRankPlusOne === 0 || pageRankPlusOne > pageCapacity) return undefined;
  const rankWord = pagePoolOffset + (pageRankPlusOne - 1) * pageSize + key % pageSize;
  if (rankWord >= entryOffset || rankWord >= words.length) return undefined;
  const rankPlusOne = words[rankWord]!;
  if (rankPlusOne === 0 || rankPlusOne > entryCapacity) return undefined;
  const base = entryOffset + (rankPlusOne - 1) * FINE_LEVELSET_SUMMARY_ENTRY_WORDS;
  if (base + FINE_LEVELSET_SUMMARY_ENTRY_WORDS - 1 >= words.length || words[base] !== key) return undefined;
  return base;
}

export function fineLevelSetSummaryRefinementSignal(summary: {
  readonly published: boolean; readonly directoryFlags: number; readonly found: boolean;
  readonly entryFlags: number; readonly minimumPhi: number; readonly maximumPhi: number;
  readonly minimumAbsolutePhi: number; readonly brickCount: number; readonly sampleCount: number;
}, lookup: Pick<FineLevelSetSummaryLeafLookup, "expectedBrickCount" | "expectedSampleCount">,
bandWidth: number): FineLevelSetSummaryRefinementSignal {
  const coarseAuthority = (summary.entryFlags >>> 31) !== 0;
  if (!summary.published || summary.directoryFlags !== 0 || !summary.found
    || (summary.entryFlags & 0x003f_ffff) !== 0
    || !Number.isFinite(summary.minimumPhi) || !Number.isFinite(summary.maximumPhi)
    || !Number.isFinite(summary.minimumAbsolutePhi)) return "invalid";
  if (summary.minimumPhi < 0 && summary.maximumPhi >= 0) return "refine";
  if (!coarseAuthority && (summary.brickCount !== lookup.expectedBrickCount
    || summary.sampleCount !== lookup.expectedSampleCount)) return "invalid";
  return summary.minimumAbsolutePhi < bandWidth ? "refine" : "complete-no-crossing";
}

export function planFineLevelSetGPUSummaries(plan: FineLevelSetBrickPlan,
  coarseEntryCapacity = 0): FineLevelSetGPUSummaryPlan {
  if (!Number.isSafeInteger(coarseEntryCapacity) || coarseEntryCapacity < 0) {
    throw new RangeError("Fine summary coarse-entry capacity must be a non-negative integer");
  }
  const levelOffsets: number[] = []; const levelDimensions: Array<readonly [number, number, number]> = [];
  let dimensions = [...plan.brickDimensions] as [number, number, number]; let hierarchyKeyCapacity = 0;
  let fineEntryCapacity = 0;
  for (let level = 0; level < 32; level += 1) {
    const count = dimensions[0] * dimensions[1] * dimensions[2];
    levelOffsets.push(hierarchyKeyCapacity); levelDimensions.push(dimensions);
    hierarchyKeyCapacity += count; fineEntryCapacity += Math.min(plan.maximumResidentBricks, count);
    if (dimensions.every((value) => value === 1)) break;
    dimensions = dimensions.map((value) => Math.ceil(value / 2)) as [number, number, number];
  }
  if (!Number.isSafeInteger(hierarchyKeyCapacity) || hierarchyKeyCapacity >= 0xffff_ffff) {
    throw new RangeError("Fine summary hierarchy keys exceed the u32 ABI");
  }
  const entryCapacity = Math.min(hierarchyKeyCapacity, fineEntryCapacity + coarseEntryCapacity);
  const directoryPageSize = FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE;
  const hierarchyTopLevelPages = Math.ceil(hierarchyKeyCapacity / directoryPageSize);
  const directoryPageCapacity = Math.min(hierarchyTopLevelPages, Math.max(1, entryCapacity));
  const directoryWords = 16 + hierarchyTopLevelPages + directoryPageCapacity * directoryPageSize
    + Math.max(1, entryCapacity) * FINE_LEVELSET_SUMMARY_ENTRY_WORDS;
  if (!Number.isSafeInteger(directoryWords) || directoryWords >= 0xffff_ffff) {
    throw new RangeError("Fine summary sparse directory exceeds the u32 ABI");
  }
  const directoryBytes = directoryWords * 4;
  const fineEntriesBytes = Math.max(1, entryCapacity) * FINE_LEVELSET_SUMMARY_ENTRY_WORDS * 4;
  const keyStateBytes = Math.max(1, entryCapacity) * 8;
  const rankStateBytes = Math.max(1, entryCapacity) * 8;
  const pageStateBytes = Math.max(1, directoryPageCapacity) * 8;
  const workStateBytes = 256;
  // Three phase groups publish independently so a producer can bind the next
  // dispatch arena as STORAGE while the current arena remains INDIRECT in the
  // same compute pass. Sharing one buffer would violate WebGPU's pass-wide
  // storage/indirect usage scope; staging copies used to hide that conflict.
  const indirectBytes = 36 + 12 + 12;
  const parameterBytes = levelOffsets.length * 112;
  return { maximumResidentBricks: plan.maximumResidentBricks, maximumLevel: levelOffsets.length - 1,
    fineEntryCapacity, coarseEntryCapacity, entryCapacity, hierarchyKeyCapacity, directoryPageSize,
    hierarchyTopLevelPages, directoryPageCapacity, directoryWords, directoryBytes, fineEntriesBytes,
    keyStateBytes, rankStateBytes, pageStateBytes, workStateBytes, indirectBytes,
    parameterBytes, allocatedBytes: directoryBytes + fineEntriesBytes + keyStateBytes + rankStateBytes + pageStateBytes
      + workStateBytes + indirectBytes + parameterBytes, levelOffsets, levelDimensions };
}

export class WebGPUFineLevelSetSummaries {
  readonly plan: FineLevelSetGPUSummaryPlan; readonly directory: GPUBuffer;
  private readonly fineEntries: GPUBuffer; private readonly fineReferences: GPUBuffer;
  private readonly coarseRows: GPUBuffer;
  private readonly rankKeys: GPUBuffer; private readonly freeRanks: GPUBuffer;
  private readonly directoryPageReferences: GPUBuffer; private readonly freeDirectoryPages: GPUBuffer;
  private readonly workState: GPUBuffer;
  private readonly mutationDispatch: GPUBuffer;
  private readonly reclamationDispatch: GPUBuffer;
  private readonly recomputeDispatch: GPUBuffer;
  private readonly params: readonly GPUBuffer[];
  private readonly pipelines: Record<string, GPUComputePipeline> = {};
  private shaderModule?: GPUShaderModule;
  private static readonly entryPoints = [
    "prepareFineSummaryDirect", "validateFineSummaryDelta", "validateFineSummaryCoarse",
    "retireFineSummaryCoarse", "removeFineSummaryPages", "addFineSummaryPages",
    "prepareFineSummaryPageReclamation", "reclaimFineSummaryDirectoryPages",
    "ensureFineSummaryDirectoryPages", "ensureFineSummaryCoarseDirectoryPages",
    "ensureFineSummaryRanks", "ensureFineSummaryCoarseRanks",
    "publishFineSummaryCoarseRows", "prepareFineSummaryRecompute", "recomputeFineSummaryBase",
    "recomputeFineSummaryAllParents", "publishFineSummaryDirect",
  ] as const;
  private readonly pipelinesDeferred: boolean;
  private cachedGroups?: {
    readonly sourceParams: GPUBuffer; readonly metadata: GPUBuffer; readonly worklist: GPUBuffer;
    readonly samples: GPUBuffer; readonly delta: GPUBuffer;
    readonly coarseDirectory: GPUBuffer; readonly coarseControl: GPUBuffer; readonly coarseDelta: GPUBuffer;
    readonly prepare: GPUBindGroup; readonly validateFine: GPUBindGroup; readonly validateCoarse: GPUBindGroup;
    readonly retireCoarse: GPUBindGroup; readonly removeFine: GPUBindGroup;
    readonly preparePageReclamation: GPUBindGroup; readonly reclaimPages: GPUBindGroup;
    readonly ensureFinePages: GPUBindGroup; readonly ensureCoarsePages: GPUBindGroup;
    readonly ensureFineRanks: GPUBindGroup; readonly ensureCoarseRanks: GPUBindGroup;
    readonly addFine: GPUBindGroup;
    readonly publishCoarse: GPUBindGroup; readonly prepareRecompute: GPUBindGroup;
    readonly base: GPUBindGroup; readonly parents: GPUBindGroup; readonly publish: GPUBindGroup;
  };
  private destroyed = false;

  get diagnosticBuffers() {
    return { fineEntries: this.fineEntries, fineReferences: this.fineReferences,
      coarseRows: this.coarseRows, rankKeys: this.rankKeys, freeRanks: this.freeRanks,
      directoryPageReferences: this.directoryPageReferences, freeDirectoryPages: this.freeDirectoryPages,
      workState: this.workState };
  }

  constructor(private readonly device: GPUDevice, readonly finePlan: FineLevelSetBrickPlan,
    coarseEntryCapacity = 0, _deferPipelineCompilation = true) {
    this.pipelinesDeferred = true;
    this.plan = planFineLevelSetGPUSummaries(finePlan, coarseEntryCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const make = (label: string, size: number) => device.createBuffer({ label, size: Math.max(4, size), usage: storage });
    this.directory = make("global fine published direct summary directory", this.plan.directoryBytes);
    this.fineEntries = make("global fine active summary mip", this.plan.fineEntriesBytes);
    this.fineReferences = make("global fine active-rank reference counts", this.plan.entryCapacity * 4);
    this.coarseRows = make("global fine active-rank coarse rows", this.plan.entryCapacity * 4);
    this.rankKeys = make("global fine active mip rank keys", this.plan.entryCapacity * 4);
    this.freeRanks = make("global fine active mip free ranks", this.plan.entryCapacity * 4);
    this.directoryPageReferences = make("global fine sparse directory-page references",
      this.plan.directoryPageCapacity * 4);
    this.freeDirectoryPages = make("global fine sparse directory-page free ranks",
      this.plan.directoryPageCapacity * 4);
    this.workState = make("global fine direct publication state", this.plan.workStateBytes);
    const directIndirect = (label: string, size: number) => device.createBuffer({ label, size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC });
    this.mutationDispatch = directIndirect("global fine direct mutation dispatches", 36);
    this.reclamationDispatch = directIndirect("global fine direct reclamation dispatch", 12);
    this.recomputeDispatch = directIndirect("global fine direct recompute dispatch", 12);
    this.params = this.plan.levelOffsets.map((_, level) => device.createBuffer({
      label: `global fine direct summary parameters level ${level}`, size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
  }

  private descriptor(entryPoint: string): GPUComputePipelineDescriptor {
    this.shaderModule ??= this.device.createShaderModule({
      label: "global fine direct summary hierarchy", code: fineLevelSetSummaryWGSL,
    });
    return { label: entryPoint, layout: "auto",
      compute: { module: this.shaderModule, entryPoint } };
  }

  initializationTasks(): GPUInitializationTask[] {
    if (!this.pipelinesDeferred) return [];
    return WebGPUFineLevelSetSummaries.entryPoints.map((entryPoint) => ({
      id: `octree.fine-summary.pipeline.${entryPoint}`,
      phase: "adaptive-topology" as const,
      label: `Compile fine summary · ${entryPoint}`,
      run: async () => { this.pipelines[entryPoint] =
        await this.device.createComputePipelineAsync(this.descriptor(entryPoint)); },
    }));
  }

  encode(broker: PassBroker, source: WebGPUFineLevelSetBrickSource,
    pageDelta: FineLevelSetSummaryPageDeltaSource, coarse: FineLevelSetSummaryCoarseSource): void {
    if (this.destroyed) throw new Error("Fine summary hierarchy is destroyed");
    if (source.plan !== this.finePlan && JSON.stringify(source.plan) !== JSON.stringify(this.finePlan)) {
      throw new RangeError("Fine summary source does not match its configured lattice");
    }
    if (coarse.deltaHeaderWords !== 16 || coarse.deltaRecordWords !== 4) {
      throw new RangeError("Fine summary coarse-delta ABI is invalid");
    }
    for (let level = 0; level < this.params.length; level += 1) {
      // Match WGSL uniform layout exactly: each vec3u begins on a 16-byte
      // boundary, including the two vec3 fields that follow scalar runs.
      const data = new Uint32Array(28); const dims = this.plan.levelDimensions[level];
      data.set(this.finePlan.brickDimensions, 0); data[3] = this.finePlan.samplesPerBrick;
      data.set([this.finePlan.maximumResidentBricks, this.plan.entryCapacity, source.generation, level], 4);
      data.set([this.plan.levelOffsets[level], dims[0] * dims[1] * dims[2]], 8); data.set(dims, 12);
      data[15] = this.plan.maximumLevel; data[16] = this.plan.hierarchyKeyCapacity; data[17] = 7;
      data[18] = this.plan.coarseEntryCapacity; data.set(this.finePlan.finestCellDimensions, 20);
      data[23] = pageDelta.layout.changedKeysOffsetWords;
      data[24] = this.device.limits.maxComputeWorkgroupsPerDimension;
      data[25] = this.plan.directoryPageCapacity; data[26] = this.plan.hierarchyTopLevelPages;
      data[27] = this.finePlan.fineFactor;
      this.device.queue.writeBuffer(this.params[level], 0, data);
    }
    const delta = pageDelta.buffer; const coarseDirectory = coarse.directory;
    const coarseControl = coarse.control; const coarseDelta = coarse.delta;
    let groups = this.cachedGroups;
    const matches = groups?.sourceParams === source.params && groups.metadata === source.metadata
      && groups.worklist === source.worklist && groups.samples === source.samples
      && groups.delta === delta && groups.coarseDirectory === coarseDirectory
      && groups.coarseControl === coarseControl && groups.coarseDelta === coarseDelta;
    if (!matches) {
      const bind = (name: string, params: GPUBuffer | undefined,
        buffers: readonly (readonly [number, GPUBuffer])[]) => this.device.createBindGroup({
        layout: this.pipelines[name]!.getBindGroupLayout(0), entries: [
          ...(params ? [{ binding: 0, resource: { buffer: params } }] : []),
          ...buffers.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
        ],
      });
      const fineInput = [[1, source.metadata], [2, source.worklist], [5, this.directory],
        [12, this.workState], [17, delta]] as const;
      const baseSamples = [[1, source.metadata], [2, source.worklist], [3, source.samples],
        [7, this.fineEntries], [11, this.rankKeys], [12, this.workState]] as const;
      const parentSamples = [...baseSamples, [5, this.directory]] as const;
      groups = {
        sourceParams: source.params, metadata: source.metadata, worklist: source.worklist,
        samples: source.samples, delta, coarseDirectory, coarseControl, coarseDelta,
        prepare: bind("prepareFineSummaryDirect", this.params[0], [[2, source.worklist], [5, this.directory],
          [6, coarseDirectory], [12, this.workState], [14, coarseDelta], [16, coarseControl], [17, delta],
          [18, this.mutationDispatch]]),
        validateFine: bind("validateFineSummaryDelta", this.params[0],
          [[1, source.metadata], [2, source.worklist], [12, this.workState], [17, delta]]),
        validateCoarse: bind("validateFineSummaryCoarse", this.params[0],
          [[6, coarseDirectory], [12, this.workState]]),
        retireCoarse: bind("retireFineSummaryCoarse", this.params[0],
          [[5, this.directory], [8, this.fineReferences], [9, this.coarseRows],
            [10, this.directoryPageReferences], [12, this.workState], [13, this.freeRanks],
            [14, coarseDelta]]),
        removeFine: bind("removeFineSummaryPages", this.params[0], [...fineInput, [8, this.fineReferences],
          [9, this.coarseRows], [10, this.directoryPageReferences], [13, this.freeRanks]]),
        preparePageReclamation: bind("prepareFineSummaryPageReclamation", this.params[0],
          [[12, this.workState], [18, this.reclamationDispatch]]),
        reclaimPages: bind("reclaimFineSummaryDirectoryPages", this.params[0],
          [[5, this.directory], [10, this.directoryPageReferences], [11, this.rankKeys],
            [12, this.workState], [13, this.freeRanks], [15, this.freeDirectoryPages]]),
        ensureFinePages: bind("ensureFineSummaryDirectoryPages", this.params[0], [...fineInput,
          [10, this.directoryPageReferences], [15, this.freeDirectoryPages]]),
        ensureCoarsePages: bind("ensureFineSummaryCoarseDirectoryPages", this.params[0],
          [[5, this.directory], [6, coarseDirectory], [10, this.directoryPageReferences],
            [12, this.workState], [15, this.freeDirectoryPages]]),
        ensureFineRanks: bind("ensureFineSummaryRanks", this.params[0], [...fineInput,
          [8, this.fineReferences], [9, this.coarseRows], [10, this.directoryPageReferences],
          [11, this.rankKeys], [13, this.freeRanks]]),
        ensureCoarseRanks: bind("ensureFineSummaryCoarseRanks", this.params[0],
          [[5, this.directory], [6, coarseDirectory], [8, this.fineReferences], [9, this.coarseRows],
            [10, this.directoryPageReferences], [11, this.rankKeys], [12, this.workState],
            [13, this.freeRanks]]),
        addFine: bind("addFineSummaryPages", this.params[0], [...fineInput, [8, this.fineReferences]]),
        publishCoarse: bind("publishFineSummaryCoarseRows", this.params[0],
          [[5, this.directory], [6, coarseDirectory], [9, this.coarseRows], [12, this.workState]]),
        prepareRecompute: bind("prepareFineSummaryRecompute", this.params[0],
          [[12, this.workState], [18, this.recomputeDispatch]]),
        base: bind("recomputeFineSummaryBase", this.params[0], baseSamples),
        parents: bind("recomputeFineSummaryAllParents", this.params[0], parentSamples),
        publish: bind("publishFineSummaryDirect", this.params[0], [[5, this.directory], [6, coarseDirectory],
          [7, this.fineEntries], [9, this.coarseRows], [11, this.rankKeys], [12, this.workState]]),
      };
      this.cachedGroups = groups;
    }
    const boundGroups = groups!;
    const run = (name: string, group: GPUBindGroup, label: string) => {
      const pass = broker.compute({ label }); pass.setPipeline(this.pipelines[name]); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
    };
    const indirect = (name: string, group: GPUBindGroup, dispatch: GPUBuffer,
      offset: number, label: string) => {
      const pass = broker.compute({ label }); pass.setPipeline(this.pipelines[name]); pass.setBindGroup(0, group);
      pass.dispatchWorkgroupsIndirect(dispatch, offset);
    };
    run("prepareFineSummaryDirect", boundGroups.prepare, "Prepare direct fine-summary publication");
    broker.fence("fine-summary mutation dispatch publication");
    indirect("validateFineSummaryDelta", boundGroups.validateFine, this.mutationDispatch, 0,
      "Validate exact fine-summary delta");
    indirect("validateFineSummaryCoarse", boundGroups.validateCoarse, this.mutationDispatch, 12,
      "Validate fine-summary coarse rows");
    indirect("retireFineSummaryCoarse", boundGroups.retireCoarse, this.mutationDispatch, 24,
      "Retire direct coarse summary ranks");
    indirect("removeFineSummaryPages", boundGroups.removeFine, this.mutationDispatch, 0,
      "Remove direct fine-summary page references");
    run("prepareFineSummaryPageReclamation", boundGroups.preparePageReclamation,
      "Prepare sparse fine-summary directory-page reclamation");
    broker.fence("fine-summary reclamation dispatch publication");
    indirect("reclaimFineSummaryDirectoryPages", boundGroups.reclaimPages, this.reclamationDispatch, 0,
      "Reclaim empty fine-summary directory pages");
    indirect("ensureFineSummaryDirectoryPages", boundGroups.ensureFinePages, this.mutationDispatch, 0,
      "Ensure sparse fine-summary directory pages");
    indirect("ensureFineSummaryCoarseDirectoryPages", boundGroups.ensureCoarsePages, this.mutationDispatch, 12,
      "Ensure sparse coarse-summary directory pages");
    indirect("ensureFineSummaryRanks", boundGroups.ensureFineRanks, this.mutationDispatch, 0,
      "Ensure compact fine-summary ranks");
    indirect("ensureFineSummaryCoarseRanks", boundGroups.ensureCoarseRanks, this.mutationDispatch, 12,
      "Ensure compact coarse-summary ranks");
    indirect("addFineSummaryPages", boundGroups.addFine, this.mutationDispatch, 0,
      "Add direct fine-summary page references");
    indirect("publishFineSummaryCoarseRows", boundGroups.publishCoarse, this.mutationDispatch, 12,
      "Publish direct coarse summary ranks");
    run("prepareFineSummaryRecompute", boundGroups.prepareRecompute, "Prepare active fine-summary mip dispatch");
    broker.fence("fine-summary recompute dispatch publication");
    indirect("recomputeFineSummaryBase", boundGroups.base, this.recomputeDispatch, 0,
      "Recompute direct fine-summary bases");
    indirect("recomputeFineSummaryAllParents", boundGroups.parents, this.recomputeDispatch, 0,
      "Recompute all direct fine-summary parents");
    run("publishFineSummaryDirect", boundGroups.publish, "Publish direct fine-summary directory and active mip");
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.directory.destroy(); this.fineEntries.destroy(); this.fineReferences.destroy();
    this.coarseRows.destroy(); this.rankKeys.destroy(); this.freeRanks.destroy();
    this.directoryPageReferences.destroy(); this.freeDirectoryPages.destroy();
    this.workState.destroy(); this.mutationDispatch.destroy(); this.reclamationDispatch.destroy();
    this.recomputeDispatch.destroy(); this.params.forEach((buffer) => buffer.destroy());
  }
}

export const fineLevelSetSummaryWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const PUBLISHED:u32=0x80000000u;
const CAPACITY:u32=1u;const STALE:u32=4u;const NONFINITE:u32=8u;
const COARSE_AUTHORITY:u32=0x80000000u;const CENTER_COMPLETE:u32=0x3fc00000u;const STATE_READY:u32=0x51a7e001u;
const SIZING_REFINEMENT:u32=0x40000000u;const PAGE_SIZING_MASK:u32=0xff0000u;
struct P{baseDims:vec3u,samplesPerBrick:u32,pageCapacity:u32,entryCapacity:u32,generation:u32,level:u32,
 levelOffset:u32,levelKeyCount:u32,levelDims:vec3u,maximumLevel:u32,hierarchyKeyCapacity:u32,
 worklistHeaderWords:u32,coarseEntryCapacity:u32,finestDims:vec3u,changedKeysOffset:u32,maxWorkgroups:u32,
 directoryPageCapacity:u32,topLevelPageCount:u32,fineFactor:u32}
struct Entry{key:u32,minimumPhi:u32,maximumPhi:u32,minimumAbsolutePhi:u32,samples:u32,bricks:u32,flags:u32,centerPhi:u32,
 validMaskLow:u32,validMaskHigh:u32,negativeMaskLow:u32,negativeMaskHigh:u32}
struct CoarseEntry{cellPlusOne:u32,size:u32,phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32,row:u32,physicalVolume:f32}
struct CoarseDirectory{state:u32,generation:u32,rowCount:u32,maximumLeafSize:u32,dimensions:vec3u,physicalCellSize:f32,entries:array<CoarseEntry>}
struct CoarseDeltaRecord{cellPlusOne:u32,size:u32,row:u32,flags:u32}
struct CoarseDelta{count:u32,generation:u32,flags:u32,valid:u32,pad:array<u32,12>,items:array<CoarseDeltaRecord>}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>metadata:array<u32>;
@group(0)@binding(2)var<storage,read>worklist:array<u32>;@group(0)@binding(3)var<storage,read>samples:array<u32>;
${fineLevelSetPackedSampleWGSL("samples")}
@group(0)@binding(5)var<storage,read_write>directory:array<atomic<u32>>;
@group(0)@binding(6)var<storage,read>coarse:CoarseDirectory;@group(0)@binding(7)var<storage,read_write>fineEntries:array<Entry>;
@group(0)@binding(8)var<storage,read_write>fineReferences:array<atomic<u32>>;
@group(0)@binding(9)var<storage,read_write>coarseRows:array<atomic<u32>>;
@group(0)@binding(10)var<storage,read_write>directoryPageReferences:array<atomic<u32>>;
@group(0)@binding(11)var<storage,read_write>rankKeys:array<atomic<u32>>;
@group(0)@binding(12)var<storage,read_write>state:array<atomic<u32>>;
@group(0)@binding(13)var<storage,read_write>freeRanks:array<u32>;@group(0)@binding(14)var<storage,read>coarseDelta:CoarseDelta;
@group(0)@binding(15)var<storage,read_write>freeDirectoryPages:array<u32>;
@group(0)@binding(16)var<storage,read>coarseControl:array<u32>;@group(0)@binding(17)var<storage,read>pageDelta:array<u32>;
@group(0)@binding(18)var<storage,read_write>publishedDispatch:array<u32>;
var<workgroup>minimumPhi:array<f32,64>;var<workgroup>maximumPhi:array<f32,64>;
var<workgroup>minimumAbsolutePhi:array<f32,64>;var<workgroup>validSamples:array<u32,64>;
var<workgroup>errors:array<u32,64>;var<workgroup>children:array<Entry,8>;
var<workgroup>exactValid:array<u32,64>;var<workgroup>exactNegative:array<u32,64>;
var<workgroup>centerBits:array<u32,8>;var<workgroup>centerStates:array<u32,8>;
var<workgroup>publishErrors:array<u32,256>;
var<workgroup>parentPartials:array<Entry,64>;
var<workgroup>parentKeyPlusOne:u32;var<workgroup>parentRankValid:u32;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn ordered(v:f32)->u32{let bits=bitcast<u32>(v);return select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u);}
fn emptyEntry(key:u32)->Entry{return Entry(key,0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,0u,0u,0u,0u,0u,0u);}
fn present(e:Entry)->bool{return e.samples!=0u||e.bricks!=0u||(e.flags&CENTER_COMPLETE)!=0u;}
fn decodedOrdered(bits:u32)->f32{return bitcast<f32>(select(~bits,bits^0x80000000u,(bits&0x80000000u)!=0u));}
fn publishedEntryValid(e:Entry)->bool{let centerState=e.flags&CENTER_COMPLETE;
 if(centerState!=0u&&centerState!=CENTER_COMPLETE){return false;}
 if(centerState==CENTER_COMPLETE&&!finite(bitcast<f32>(e.centerPhi))){return false;}
 let hasSamples=e.samples!=0u;let hasBricks=e.bricks!=0u;if(hasSamples!=hasBricks){return false;}
 if(hasSamples){if(e.bricks>p.pageCapacity||e.bricks>INVALID/p.samplesPerBrick
   ||e.samples>e.bricks*p.samplesPerBrick){return false;}let lo=decodedOrdered(e.minimumPhi);
  let hi=decodedOrdered(e.maximumPhi);let ma=bitcast<f32>(e.minimumAbsolutePhi);
  var masksValid=(e.negativeMaskLow&~e.validMaskLow)==0u&&(e.negativeMaskHigh&~e.validMaskHigh)==0u;
  if(p.fineFactor==1u&&e.key<p.baseDims.x*p.baseDims.y*p.baseDims.z){
   masksValid=masksValid&&e.samples==countOneBits(e.validMaskLow)+countOneBits(e.validMaskHigh);
  }else{masksValid=masksValid&&(e.validMaskLow|e.validMaskHigh|e.negativeMaskLow|e.negativeMaskHigh)==0u;}
  return masksValid&&finite(lo)&&finite(hi)&&finite(ma)&&lo<=hi&&ma>=0.0;}
 if((e.flags&COARSE_AUTHORITY)!=0u){let lo=decodedOrdered(e.minimumPhi);let hi=decodedOrdered(e.maximumPhi);
  let ma=bitcast<f32>(e.minimumAbsolutePhi);return finite(lo)&&finite(hi)&&finite(ma)&&lo<=hi&&ma>=0.0;}
 return e.minimumPhi==INVALID&&e.maximumPhi==0u&&e.minimumAbsolutePhi==bitcast<u32>(3.402823e38);}
fn combine(a:Entry,b:Entry)->Entry{return Entry(a.key,min(a.minimumPhi,b.minimumPhi),max(a.maximumPhi,b.maximumPhi),
 min(a.minimumAbsolutePhi,b.minimumAbsolutePhi),a.samples+b.samples,a.bricks+b.bricks,a.flags|b.flags,0u,0u,0u,0u,0u);}
fn dirLoad(word:u32)->u32{return atomicLoad(&directory[word]);}fn dirStore(word:u32,value:u32){atomicStore(&directory[word],value);}
fn topWord(key:u32)->u32{return 16u+key/${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u;}fn pagePoolOffset()->u32{return 16u+p.topLevelPageCount;}
fn pageWord(page:u32,key:u32)->u32{return pagePoolOffset()+page*${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u+(key&${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE - 1}u);}
fn entryOffset()->u32{return pagePoolOffset()+p.directoryPageCapacity*${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u;}
fn entryBase(rank:u32)->u32{return entryOffset()+rank*${FINE_LEVELSET_SUMMARY_ENTRY_WORDS}u;}
fn rankForKey(key:u32)->u32{if(key>=p.hierarchyKeyCapacity){return INVALID;}let pagePlusOne=dirLoad(topWord(key));
 if(pagePlusOne==0u||pagePlusOne==INVALID||pagePlusOne>p.directoryPageCapacity){return INVALID;}
 let rankPlusOne=dirLoad(pageWord(pagePlusOne-1u,key));return select(INVALID,rankPlusOne-1u,rankPlusOne>0u&&rankPlusOne<=p.entryCapacity);}
fn storePublic(rank:u32,e:Entry){let base=entryBase(rank);dirStore(base,e.key);dirStore(base+1u,e.minimumPhi);
 dirStore(base+2u,e.maximumPhi);dirStore(base+3u,e.minimumAbsolutePhi);dirStore(base+4u,e.samples);
 dirStore(base+5u,e.bricks);dirStore(base+6u,e.flags);dirStore(base+7u,e.centerPhi);
 dirStore(base+8u,e.validMaskLow);dirStore(base+9u,e.validMaskHigh);
 dirStore(base+10u,e.negativeMaskLow);dirStore(base+11u,e.negativeMaskHigh);}
fn validWorklist()->bool{return p.worklistHeaderWords==7u&&arrayLength(&worklist)>=7u&&worklist[0]==p.generation
 &&worklist[2]==p.pageCapacity&&(worklist[3]&3u)==3u&&worklist[5]==1u&&worklist[6]==1u
 &&worklist[1]<=p.pageCapacity&&7u+worklist[1]<=arrayLength(&worklist);}
fn finePage(key:u32)->u32{if(!validWorklist()){return INVALID;}let count=p.baseDims.x*p.baseDims.y*p.baseDims.z;
 let base=7u+p.pageCapacity;if(key>=count||base+key>=arrayLength(&worklist)){return INVALID;}let id=worklist[base+key];
 let m=id*4u;return select(INVALID,id,id<p.pageCapacity&&m+2u<arrayLength(&metadata)&&metadata[m]==id
  &&metadata[m+1u]==key&&metadata[m+2u]==p.generation);}
fn writeDispatch(base:u32,count:u32,itemsPerGroup:u32){let groups=(count+itemsPerGroup-1u)/itemsPerGroup;
 let width=min(groups,p.maxWorkgroups);let safe=max(1u,width);atomicStore(&state[base],width);
 atomicStore(&state[base+1u],select(1u,(groups+safe-1u)/safe,width!=0u));atomicStore(&state[base+2u],1u);}
fn publishDispatch(base:u32,count:u32,itemsPerGroup:u32){let groups=(count+itemsPerGroup-1u)/itemsPerGroup;
 let width=min(groups,p.maxWorkgroups);let safe=max(1u,width);publishedDispatch[base]=width;
 publishedDispatch[base+1u]=select(1u,(groups+safe-1u)/safe,width!=0u);publishedDispatch[base+2u]=1u;}
fn hierarchyKey(baseKey:u32,targetLevel:u32)->u32{let xy=p.baseDims.x*p.baseDims.y;let z=baseKey/xy;
 let rem=baseKey-z*xy;let y=rem/p.baseDims.x;var q=vec3u(rem-y*p.baseDims.x,y,z);var dims=p.baseDims;var offset=0u;
 for(var level=0u;level<targetLevel;level+=1u){offset+=dims.x*dims.y*dims.z;dims=(dims+vec3u(1u))/2u;q/=2u;}
 return offset+q.x+dims.x*(q.y+dims.y*q.z);}
fn coarseHierarchyKey(cellPlusOne:u32,size:u32)->u32{
 if(cellPlusOne==0u||size==0u||(size&(size-1u))!=0u){return INVALID;}
 let cell=cellPlusOne-1u;if(cell>=p.finestDims.x*p.finestDims.y*p.finestDims.z){return INVALID;}
 let origin=vec3u(cell%p.finestDims.x,(cell/p.finestDims.x)%p.finestDims.y,
  cell/(p.finestDims.x*p.finestDims.y));
 // At factor 1 a B4 summary leaf contains 4^3 finest cells. Sizes 1, 2,
 // and 4 map to that level-zero key, while larger dyadic leaves advance one
 // hierarchy level per doubling beyond size 4. Coarse rows are deliberately
 // not attached to these colliding keys; factor-1 consumers fall back to the
 // direct coarse authority when fine evidence is unavailable.
 if(p.fineFactor==1u){if(any(p.baseDims!=(p.finestDims+vec3u(3u))/4u)){return INVALID;}
  let side=max(1u,size/4u);let brickOrigin=origin/4u;
  if(size>=4u&&any(origin%vec3u(size)!=vec3u(0u))){return INVALID;}
  if(any(brickOrigin%vec3u(side)!=vec3u(0u))){return INVALID;}
  let level=31u-countLeadingZeros(side);let base=brickOrigin.x+p.baseDims.x*(brickOrigin.y+p.baseDims.y*brickOrigin.z);
  return hierarchyKey(base,level);}
 let ratio=p.baseDims/p.finestDims;
 if(ratio.x==0u||any(ratio!=vec3u(ratio.x))){return INVALID;}
 let side=size*ratio.x;if(side==0u||(side&(side-1u))!=0u){return INVALID;}let brickOrigin=origin*ratio.x;
 if(any(brickOrigin%vec3u(side)!=vec3u(0u))){return INVALID;}let level=31u-countLeadingZeros(side);
 let base=brickOrigin.x+p.baseDims.x*(brickOrigin.y+p.baseDims.y*brickOrigin.z);return hierarchyKey(base,level);}
fn changedKey(index:u32)->u32{if(atomicLoad(&state[1])==1u){let id=worklist[7u+index];let m=id*4u;
 return select(INVALID,metadata[m+1u],id<p.pageCapacity&&m+2u<arrayLength(&metadata)&&metadata[m]==id
  &&metadata[m+2u]==p.generation);}return pageDelta[p.changedKeysOffset+index];}
fn setError(value:u32,index:u32){atomicOr(&state[0],value);atomicMin(&state[9],index);}
fn coarseAuthoritative()->bool{let generation=p.generation&0x3fffffffu;return p.coarseEntryCapacity==0u||
 (arrayLength(&coarseControl)>=12u&&coarse.state==PUBLISHED&&(coarse.generation&0x3fffffffu)==generation
 &&coarse.rowCount==coarseControl[2]&&coarse.rowCount<=arrayLength(&coarse.entries)
 &&all(coarse.dimensions==p.finestDims)&&coarseControl[0]==0u&&(coarseControl[10]&0x3fffffffu)==generation
 &&coarseControl[11]==PUBLISHED&&coarseDelta.flags==0u&&coarseDelta.valid==PUBLISHED
 &&(coarseDelta.generation&0x3fffffffu)==generation&&coarseDelta.count<=arrayLength(&coarseDelta.items)
 &&coarseDelta.count<=2u*p.coarseEntryCapacity);}
@compute @workgroup_size(1)fn prepareFineSummaryDirect(){let cold=atomicLoad(&state[15])!=STATE_READY;
 let fineValid=validWorklist();let coarseValid=coarseAuthoritative();let deltaValid=arrayLength(&pageDelta)>p.changedKeysOffset
 &&pageDelta[1]==p.generation&&pageDelta[15]==1u&&pageDelta[0]<=2u*p.pageCapacity
 &&p.changedKeysOffset+pageDelta[0]<=arrayLength(&pageDelta);let mode=select(2u,1u,cold);
 let valid=fineValid&&coarseValid&&(cold||deltaValid);let fineCount=select(pageDelta[0],worklist[1],cold);
 atomicStore(&state[0],select(STALE,0u,valid));atomicStore(&state[1],mode);atomicStore(&state[2],select(0u,fineCount,valid));
 // Factor-1 size-1/2/4 rows collide on one B4 hierarchy key, so a single
 // atomic coarseRows slot cannot represent them without a nondeterministic
 // last-writer centre/interval. Publish a fine-only summary at factor 1;
 // consumers retain their direct coarse authority whenever that summary is
 // absent or lacks a geometrically matching centre.
 atomicStore(&state[3],select(0u,coarse.rowCount,valid&&p.coarseEntryCapacity!=0u&&p.fineFactor!=1u));
 atomicStore(&state[4],select(0u,coarseDelta.count,valid&&!cold&&p.coarseEntryCapacity!=0u&&p.fineFactor!=1u));
 atomicStore(&state[5],p.generation);atomicStore(&state[9],INVALID);atomicStore(&state[12],atomicLoad(&state[8]));dirStore(9u,0u);
 writeDispatch(16u,atomicLoad(&state[2]),256u);writeDispatch(19u,atomicLoad(&state[3]),256u);
 writeDispatch(22u,atomicLoad(&state[4]),256u);publishDispatch(0u,atomicLoad(&state[2]),256u);
 publishDispatch(3u,atomicLoad(&state[3]),256u);publishDispatch(6u,atomicLoad(&state[4]),256u);}
@compute @workgroup_size(256)fn validateFineSummaryDelta(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let index=fineLinearWorkgroup(wid,n)*256u+lid;
 if(index>=atomicLoad(&state[2])||atomicLoad(&state[0])!=0u){return;}let key=changedKey(index);
 if(key>=p.baseDims.x*p.baseDims.y*p.baseDims.z){setError(STALE,index);}}
@compute @workgroup_size(256)fn validateFineSummaryCoarse(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let row=fineLinearWorkgroup(wid,n)*256u+lid;
 if(row>=atomicLoad(&state[3])||atomicLoad(&state[0])!=0u){return;}let e=coarse.entries[row];
 let key=coarseHierarchyKey(e.cellPlusOne,e.size);if(key==INVALID||(e.flags&9u)!=9u||e.row!=row||!finite(e.phi)
  ||!finite(e.minimumPhi)||!finite(e.maximumPhi)||e.minimumPhi>e.phi||e.phi>e.maximumPhi){setError(STALE,row);}}
fn popRank()->u32{loop{let count=atomicLoad(&state[8]);if(count==0u){break;}let claimed=atomicCompareExchangeWeak(&state[8],count,count-1u);
 if(claimed.exchanged){return freeRanks[count-1u];}}return atomicAdd(&state[7],1u);}
fn popDirectoryPage()->u32{loop{let count=atomicLoad(&state[11]);if(count==0u){break;}
 let claimed=atomicCompareExchangeWeak(&state[11],count,count-1u);if(claimed.exchanged){return freeDirectoryPages[count-1u];}}
 return atomicAdd(&state[10],1u);}
fn ensureDirectoryPage(key:u32){if(key>=p.hierarchyKeyCapacity){setError(CAPACITY,key);return;}let word=topWord(key);
 var claimed=atomicCompareExchangeWeak(&directory[word],0u,INVALID);for(var retry=0u;retry<4u&&!claimed.exchanged&&claimed.old_value==0u;retry+=1u){
  claimed=atomicCompareExchangeWeak(&directory[word],0u,INVALID);}if(!claimed.exchanged){if(claimed.old_value==0u){setError(CAPACITY,key);}return;}
 let page=popDirectoryPage();
 if(page>=p.directoryPageCapacity){dirStore(word,0u);setError(CAPACITY,key);return;}
 for(var slot=0u;slot<${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u;slot+=1u){dirStore(pagePoolOffset()+page*${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u+slot,0u);}
 atomicStore(&directoryPageReferences[page],0u);dirStore(word,page+1u);}
fn ensureRank(key:u32){let pagePlusOne=dirLoad(topWord(key));if(pagePlusOne==0u||pagePlusOne==INVALID
 ||pagePlusOne>p.directoryPageCapacity){setError(CAPACITY,key);return;}let page=pagePlusOne-1u;let word=pageWord(page,key);
 var claimed=atomicCompareExchangeWeak(&directory[word],0u,INVALID);for(var retry=0u;retry<4u&&!claimed.exchanged&&claimed.old_value==0u;retry+=1u){
  claimed=atomicCompareExchangeWeak(&directory[word],0u,INVALID);}if(!claimed.exchanged){if(claimed.old_value==0u){setError(CAPACITY,key);}return;}
 let rank=popRank();
 if(rank>=p.entryCapacity){dirStore(word,0u);setError(CAPACITY,key);return;}atomicStore(&fineReferences[rank],0u);
 atomicStore(&coarseRows[rank],0u);atomicStore(&rankKeys[rank],key+1u);atomicAdd(&directoryPageReferences[page],1u);
 dirStore(word,rank+1u);atomicAdd(&state[6],1u);}
fn releaseKey(key:u32){let pagePlusOne=dirLoad(topWord(key));if(pagePlusOne==0u||pagePlusOne==INVALID
 ||pagePlusOne>p.directoryPageCapacity){setError(STALE,key);return;}let page=pagePlusOne-1u;let word=pageWord(page,key);
 let rankPlusOne=dirLoad(word);if(rankPlusOne==0u||rankPlusOne==INVALID||rankPlusOne>p.entryCapacity){setError(STALE,key);return;}
 let rank=rankPlusOne-1u;dirStore(word,0u);let slot=atomicAdd(&state[8],1u);
 if(slot<p.entryCapacity){freeRanks[slot]=rank;}else{setError(CAPACITY,key);}atomicSub(&state[6],1u);
 let pageOld=atomicSub(&directoryPageReferences[page],1u);if(pageOld==0u){setError(STALE,key);}}
fn removeFineBase(key:u32){let baseRank=rankForKey(key);if(baseRank==INVALID||atomicLoad(&fineReferences[baseRank])==0u){return;}
 for(var level=0u;level<=p.maximumLevel;level+=1u){let h=hierarchyKey(key,level);let rank=rankForKey(h);
  if(rank==INVALID){setError(STALE,h);continue;}let old=atomicSub(&fineReferences[rank],1u);if(old==0u){setError(STALE,h);}
  else if(old==1u&&atomicLoad(&coarseRows[rank])==0u){releaseKey(h);}}}
fn addFineBase(key:u32){let baseRank=rankForKey(key);if(baseRank==INVALID){setError(CAPACITY,key);return;}
 if(atomicLoad(&fineReferences[baseRank])!=0u){return;}for(var level=0u;level<=p.maximumLevel;level+=1u){let h=hierarchyKey(key,level);
  let rank=rankForKey(h);if(rank==INVALID){setError(CAPACITY,h);continue;}atomicAdd(&fineReferences[rank],1u);}}
@compute @workgroup_size(256)fn retireFineSummaryCoarse(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let index=fineLinearWorkgroup(wid,n)*256u+lid;
 if(index>=atomicLoad(&state[4])||atomicLoad(&state[0])!=0u){return;}let item=coarseDelta.items[index];
 let key=coarseHierarchyKey(item.cellPlusOne,item.size);if(key==INVALID){setError(STALE,index);return;}
 let rank=rankForKey(key);if(rank==INVALID){return;}
 if(atomicExchange(&coarseRows[rank],0u)!=0u&&atomicLoad(&fineReferences[rank])==0u){releaseKey(key);}}
@compute @workgroup_size(256)fn removeFineSummaryPages(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let index=fineLinearWorkgroup(wid,n)*256u+lid;
 if(index>=atomicLoad(&state[2])||atomicLoad(&state[0])!=0u||atomicLoad(&state[1])==1u){return;}
 let key=changedKey(index);if(finePage(key)==INVALID){removeFineBase(key);}}
@compute @workgroup_size(1)fn prepareFineSummaryPageReclamation(){let first=atomicLoad(&state[12]);let last=atomicLoad(&state[8]);
 let count=select(0u,last-first,atomicLoad(&state[0])==0u&&last>=first);writeDispatch(31u,count,256u);
 publishDispatch(0u,count,256u);}
@compute @workgroup_size(256)fn reclaimFineSummaryDirectoryPages(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let index=fineLinearWorkgroup(wid,n)*256u+lid;
 let first=atomicLoad(&state[12]);let last=atomicLoad(&state[8]);if(atomicLoad(&state[0])!=0u||first>last||index>=last-first){return;}
 let slot=first+index;if(slot>=p.entryCapacity){setError(CAPACITY,slot);return;}let rank=freeRanks[slot];
 if(rank>=p.entryCapacity||rank>=arrayLength(&rankKeys)){setError(CAPACITY,rank);return;}let keyPlusOne=atomicLoad(&rankKeys[rank]);
 if(keyPlusOne==0u){setError(STALE,rank);return;}let key=keyPlusOne-1u;if(key>=p.hierarchyKeyCapacity){setError(STALE,key);
  atomicStore(&rankKeys[rank],0u);return;}let top=topWord(key);let pagePlusOne=dirLoad(top);
 if(pagePlusOne==INVALID||pagePlusOne>p.directoryPageCapacity){setError(STALE,key);atomicStore(&rankKeys[rank],0u);return;}
 if(pagePlusOne!=0u&&pagePlusOne!=INVALID&&pagePlusOne<=p.directoryPageCapacity){let page=pagePlusOne-1u;
  if(atomicLoad(&directoryPageReferences[page])==0u){var released=atomicCompareExchangeWeak(&directory[top],pagePlusOne,0u);
   for(var retry=0u;retry<4u&&!released.exchanged&&released.old_value==pagePlusOne;retry+=1u){released=atomicCompareExchangeWeak(&directory[top],pagePlusOne,0u);}
   if(released.exchanged){let pageSlot=atomicAdd(&state[11],1u);if(pageSlot<p.directoryPageCapacity){freeDirectoryPages[pageSlot]=page;}
    else{setError(CAPACITY,key);}}else if(released.old_value==pagePlusOne){setError(CAPACITY,key);}}}
 atomicStore(&rankKeys[rank],0u);}
@compute @workgroup_size(256)fn ensureFineSummaryDirectoryPages(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let index=fineLinearWorkgroup(wid,n)*256u+lid;
 if(index>=atomicLoad(&state[2])||atomicLoad(&state[0])!=0u){return;}let key=changedKey(index);if(finePage(key)==INVALID){return;}
 for(var level=0u;level<=p.maximumLevel;level+=1u){ensureDirectoryPage(hierarchyKey(key,level));}}
@compute @workgroup_size(256)fn ensureFineSummaryCoarseDirectoryPages(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let row=fineLinearWorkgroup(wid,n)*256u+lid;
 if(row>=atomicLoad(&state[3])||atomicLoad(&state[0])!=0u){return;}let e=coarse.entries[row];
 ensureDirectoryPage(coarseHierarchyKey(e.cellPlusOne,e.size));}
@compute @workgroup_size(256)fn ensureFineSummaryRanks(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let index=fineLinearWorkgroup(wid,n)*256u+lid;
 if(index>=atomicLoad(&state[2])||atomicLoad(&state[0])!=0u){return;}let key=changedKey(index);if(finePage(key)==INVALID){return;}
 for(var level=0u;level<=p.maximumLevel;level+=1u){ensureRank(hierarchyKey(key,level));}}
@compute @workgroup_size(256)fn ensureFineSummaryCoarseRanks(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let row=fineLinearWorkgroup(wid,n)*256u+lid;
 if(row>=atomicLoad(&state[3])||atomicLoad(&state[0])!=0u){return;}let e=coarse.entries[row];
 ensureRank(coarseHierarchyKey(e.cellPlusOne,e.size));}
@compute @workgroup_size(256)fn addFineSummaryPages(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let index=fineLinearWorkgroup(wid,n)*256u+lid;
 if(index>=atomicLoad(&state[2])||atomicLoad(&state[0])!=0u){return;}let key=changedKey(index);
 if(finePage(key)!=INVALID){addFineBase(key);}}
@compute @workgroup_size(256)fn publishFineSummaryCoarseRows(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let row=fineLinearWorkgroup(wid,n)*256u+lid;
 if(row>=atomicLoad(&state[3])||atomicLoad(&state[0])!=0u){return;}let e=coarse.entries[row];
 let key=coarseHierarchyKey(e.cellPlusOne,e.size);let rank=rankForKey(key);if(rank==INVALID){setError(CAPACITY,key);return;}
 atomicStore(&coarseRows[rank],row+1u);}
@compute @workgroup_size(1)fn prepareFineSummaryRecompute(){if(atomicLoad(&state[0])==0u){atomicStore(&state[15],STATE_READY);}
 let count=select(0u,atomicLoad(&state[7]),atomicLoad(&state[0])==0u);writeDispatch(28u,count,1u);
 publishDispatch(0u,count,1u);}
fn centerSampleAtLevel(key:u32,corner:u32,level:u32,levelOffset:u32,levelDims:vec3u)->vec2u{
 let levelCount=levelDims.x*levelDims.y*levelDims.z;if(key<levelOffset||key>=levelOffset+levelCount){return vec2u(0u);}
 let local=key-levelOffset;let xy=levelDims.x*levelDims.y;let z=local/xy;let rem=local-z*xy;let y=rem/levelDims.x;
 let coord=vec3u(rem-y*levelDims.x,y,z);let resolution=select(4u,8u,p.samplesPerBrick==512u);
 let span=(1u<<level)*resolution;let low=coord*span+vec3u(span/2u-1u);let q=low+vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);
 let brick=q/resolution;if(any(brick>=p.baseDims)){return vec2u(0u);}let key0=brick.x+p.baseDims.x*(brick.y+p.baseDims.y*brick.z);
 let page=finePage(key0);if(page==INVALID){return vec2u(0u);}let within=q-brick*resolution;
 let index=page*p.samplesPerBrick+within.x+resolution*(within.y+resolution*within.z);
 if(index>=arrayLength(&samples)){return vec2u(0u,NONFINITE);}
 let value=finePackedPhi(index);return select(vec2u(bitcast<u32>(value),1u),vec2u(0u,NONFINITE),!finite(value));}
fn centerSampleAt(key:u32,corner:u32)->vec2u{return centerSampleAtLevel(key,corner,p.level,p.levelOffset,p.levelDims);}
fn finishCenter(value:Entry)->Entry{var result=value;var center=0.0;var mask=0u;result.flags&=~CENTER_COMPLETE;
 for(var corner=0u;corner<8u;corner+=1u){result.flags|=centerStates[corner]&NONFINITE;if((centerStates[corner]&1u)!=0u){
  center+=0.125*bitcast<f32>(centerBits[corner]);mask|=1u<<corner;}}if(mask==0xffu){result.centerPhi=bitcast<u32>(center);result.flags|=CENTER_COMPLETE;}return result;}
@compute @workgroup_size(64)fn recomputeFineSummaryBase(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let rank=fineLinearWorkgroup(wid,n);
 let high=atomicLoad(&state[7]);let rankInRange=rank<high&&rank<p.entryCapacity&&rank<arrayLength(&rankKeys);
 var keyPlusOne=0u;if(rankInRange){keyPlusOne=atomicLoad(&rankKeys[rank]);}let key=select(0u,keyPlusOne-1u,keyPlusOne!=0u);
 let enabled=rankInRange&&keyPlusOne!=0u&&key>=p.levelOffset&&key<p.levelOffset+p.levelKeyCount;
 if(lid==0u&&rank<high&&!rankInRange){setError(CAPACITY,rank);}var lo=3.402823e38;var hi=-3.402823e38;
 var ma=3.402823e38;var count=0u;var failure=0u;exactValid[lid]=0u;exactNegative[lid]=0u;
 if(enabled){let page=finePage(key);if(page!=INVALID){for(var local=lid;local<p.samplesPerBrick;local+=64u){
  let index=page*p.samplesPerBrick+local;if(index>=arrayLength(&samples)){failure|=CAPACITY;continue;}
  if((finePackedFlags(index)&VALID)==0u){continue;}let value=finePackedPhi(index);if(!finite(value)){failure|=NONFINITE;continue;}
  lo=min(lo,value);hi=max(hi,value);ma=min(ma,abs(value));count+=1u;
  if(p.fineFactor==1u&&p.samplesPerBrick==64u){exactValid[lid]=1u;exactNegative[lid]=select(0u,1u,value<0.0);}}}}
 minimumPhi[lid]=lo;maximumPhi[lid]=hi;minimumAbsolutePhi[lid]=ma;validSamples[lid]=count;errors[lid]=failure;workgroupBarrier();
 for(var stride=32u;stride>0u;stride>>=1u){if(lid<stride){minimumPhi[lid]=min(minimumPhi[lid],minimumPhi[lid+stride]);
  maximumPhi[lid]=max(maximumPhi[lid],maximumPhi[lid+stride]);minimumAbsolutePhi[lid]=min(minimumAbsolutePhi[lid],minimumAbsolutePhi[lid+stride]);
  validSamples[lid]+=validSamples[lid+stride];errors[lid]|=errors[lid+stride];}workgroupBarrier();}
 if(lid<8u){var center=vec2u(0u);if(enabled){center=centerSampleAt(key,lid);}centerBits[lid]=center.x;centerStates[lid]=center.y;}workgroupBarrier();
 if(lid==0u&&enabled){var value=emptyEntry(key);value.flags=errors[0];if(validSamples[0]>0u){value.minimumPhi=ordered(minimumPhi[0]);
  value.maximumPhi=ordered(maximumPhi[0]);value.minimumAbsolutePhi=bitcast<u32>(minimumAbsolutePhi[0]);value.samples=validSamples[0];value.bricks=1u;}
  let page=finePage(key);if(page!=INVALID&&(metadata[4u*page+3u]&PAGE_SIZING_MASK)>=0x400000u){value.flags|=SIZING_REFINEMENT;}
  if(p.fineFactor==1u&&p.samplesPerBrick==64u){for(var bit=0u;bit<32u;bit+=1u){
   value.validMaskLow|=exactValid[bit]<<bit;value.validMaskHigh|=exactValid[bit+32u]<<bit;
   value.negativeMaskLow|=exactNegative[bit]<<bit;value.negativeMaskHigh|=exactNegative[bit+32u]<<bit;}}
  fineEntries[rank]=finishCenter(value);}}
@compute @workgroup_size(64)fn recomputeFineSummaryParents(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let rank=fineLinearWorkgroup(wid,n);
 let high=atomicLoad(&state[7]);let rankInRange=rank<high&&rank<p.entryCapacity&&rank<arrayLength(&rankKeys);
 var keyPlusOne=0u;if(rankInRange){keyPlusOne=atomicLoad(&rankKeys[rank]);}let key=select(0u,keyPlusOne-1u,keyPlusOne!=0u);
 let enabled=rankInRange&&keyPlusOne!=0u&&key>=p.levelOffset&&key<p.levelOffset+p.levelKeyCount;
 if(lid==0u&&rank<high&&!rankInRange){setError(CAPACITY,rank);}var coord=vec3u(0u);if(enabled){let local=key-p.levelOffset;
  let xy=p.levelDims.x*p.levelDims.y;let z=local/xy;let rem=local-z*xy;let y=rem/p.levelDims.x;coord=vec3u(rem-y*p.levelDims.x,y,z);}
 var childDims=p.baseDims;var childOffset=0u;for(var level=0u;level+1u<p.level;level+=1u){childOffset+=childDims.x*childDims.y*childDims.z;childDims=(childDims+vec3u(1u))/2u;}
 if(lid<8u){var item=emptyEntry(key);var center=vec2u(0u);if(enabled){let q=coord*2u+vec3u(lid&1u,(lid>>1u)&1u,(lid>>2u)&1u);
   if(!any(q>=childDims)){let childKey=childOffset+q.x+childDims.x*(q.y+childDims.y*q.z);let childRank=rankForKey(childKey);
    if(childRank!=INVALID){item=fineEntries[childRank];}}center=centerSampleAt(key,lid);}children[lid]=item;
  centerBits[lid]=center.x;centerStates[lid]=center.y;}workgroupBarrier();
 if(lid==0u&&enabled){var value=emptyEntry(key);for(var child=0u;child<8u;child+=1u){if(present(children[child])){value=combine(value,children[child]);}}
  fineEntries[rank]=finishCenter(value);}}
// Parent fields use associative integer/min/max aggregates. Rebuilding each
// active parent directly from its published base descendants therefore emits
// the same bits without a global dispatch boundary between every mip level.
@compute @workgroup_size(64)fn recomputeFineSummaryAllParents(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let rank=fineLinearWorkgroup(wid,n);
 let high=atomicLoad(&state[7]);let inRange=rank<high&&rank<p.entryCapacity&&rank<arrayLength(&rankKeys);
 if(lid==0u){parentRankValid=select(0u,1u,inRange);parentKeyPlusOne=0u;if(inRange){parentKeyPlusOne=atomicLoad(&rankKeys[rank]);}
  if(rank<high&&!inRange){setError(CAPACITY,rank);}}
 let valid=workgroupUniformLoad(&parentRankValid);let keyPlusOne=workgroupUniformLoad(&parentKeyPlusOne);
 let key=select(0u,keyPlusOne-1u,keyPlusOne!=0u);if(valid==0u||keyPlusOne==0u){return;}
 var level=0u;var offset=0u;var dims=p.baseDims;var found=false;loop{let count=dims.x*dims.y*dims.z;
  if(key>=offset&&key<offset+count){found=true;break;}offset+=count;if(level>=p.maximumLevel){break;}
  dims=(dims+vec3u(1u))/2u;level+=1u;}if(!found){if(lid==0u){setError(STALE,key);}return;}if(level==0u){return;}
 let local=key-offset;let xy=dims.x*dims.y;let z=local/xy;let rem=local-z*xy;let y=rem/dims.x;
 let coord=vec3u(rem-y*dims.x,y,z);let span=1u<<level;let low=coord*span;
 let highCoord=min(low+vec3u(span),p.baseDims);let extent=highCoord-low;let count=extent.x*extent.y*extent.z;
 var partial=emptyEntry(key);for(var descendant=lid;descendant<count;descendant+=64u){let dz=descendant/(extent.x*extent.y);
  let rest=descendant-dz*extent.x*extent.y;let dy=rest/extent.x;let q=low+vec3u(rest-dy*extent.x,dy,dz);
  let baseKey=q.x+p.baseDims.x*(q.y+p.baseDims.y*q.z);let baseRank=rankForKey(baseKey);
  if(baseRank!=INVALID){let item=fineEntries[baseRank];if(present(item)){partial=combine(partial,item);}}}
 parentPartials[lid]=partial;workgroupBarrier();for(var stride=32u;stride>0u;stride>>=1u){
  if(lid<stride&&present(parentPartials[lid+stride])){parentPartials[lid]=combine(parentPartials[lid],parentPartials[lid+stride]);}
  workgroupBarrier();}if(lid<8u){let center=centerSampleAtLevel(key,lid,level,offset,dims);
  centerBits[lid]=center.x;centerStates[lid]=center.y;}workgroupBarrier();
 if(lid==0u){fineEntries[rank]=finishCenter(parentPartials[0]);}}
fn coarseEntryAt(key:u32,rank:u32)->Entry{var value=emptyEntry(key);
 // Defensive mirror of the factor-1 scheduling gate: never interpret a
 // colliding sub-brick coarseRows representative as B4 summary authority.
 if(p.fineFactor==1u){return value;}
 let rowPlusOne=atomicLoad(&coarseRows[rank]);if(rowPlusOne==0u){return value;}
 let row=rowPlusOne-1u;if(row>=coarse.rowCount||row>=arrayLength(&coarse.entries)){value.flags=STALE;return value;}let e=coarse.entries[row];
 if(coarseHierarchyKey(e.cellPlusOne,e.size)!=key||(e.flags&9u)!=9u||!finite(e.phi)||!finite(e.minimumPhi)||!finite(e.maximumPhi)){value.flags=STALE;return value;}
 value.minimumPhi=ordered(e.minimumPhi);value.maximumPhi=ordered(e.maximumPhi);value.minimumAbsolutePhi=bitcast<u32>(select(min(abs(e.minimumPhi),abs(e.maximumPhi)),0.0,e.minimumPhi<=0.0&&e.maximumPhi>=0.0));
 value.flags=COARSE_AUTHORITY|CENTER_COMPLETE;value.centerPhi=bitcast<u32>(e.phi);return value;}
@compute @workgroup_size(256)fn publishFineSummaryDirect(@builtin(local_invocation_index)lid:u32){var failure=atomicLoad(&state[0]);
 let high=select(0u,atomicLoad(&state[7]),failure==0u);for(var rank=lid;rank<high;rank+=256u){let keyPlusOne=atomicLoad(&rankKeys[rank]);
  if(keyPlusOne==0u){continue;}let key=keyPlusOne-1u;var value=fineEntries[rank];let fineCenter=value.centerPhi;
  let fineCenterComplete=(value.flags&CENTER_COMPLETE)==CENTER_COMPLETE;let coarseValue=coarseEntryAt(key,rank);if(present(coarseValue)){
   value=combine(value,coarseValue);value.centerPhi=select(coarseValue.centerPhi,fineCenter,fineCenterComplete);}
  let bad=key>=p.hierarchyKeyCapacity||rankForKey(key)!=rank||!publishedEntryValid(value)||(value.flags&0x003fffffu)!=0u;
  if(bad){atomicMin(&state[9],rank);}
  failure|=select(0u,STALE,bad);failure|=value.flags&0x003fffffu;storePublic(rank,value);}
 publishErrors[lid]=failure;workgroupBarrier();for(var stride=128u;stride>0u;stride>>=1u){if(lid<stride){publishErrors[lid]|=publishErrors[lid+stride];}workgroupBarrier();}
 if(lid==0u){let error=publishErrors[0];dirStore(0u,error);dirStore(1u,p.generation);dirStore(2u,atomicLoad(&state[6]));
  dirStore(3u,p.entryCapacity);dirStore(4u,p.baseDims.x);dirStore(5u,p.baseDims.y);dirStore(6u,p.baseDims.z);dirStore(7u,p.maximumLevel);
  dirStore(8u,entryOffset());dirStore(10u,p.hierarchyKeyCapacity);dirStore(11u,p.samplesPerBrick);
  dirStore(12u,16u);dirStore(13u,atomicLoad(&state[7]));dirStore(14u,${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u);dirStore(15u,p.topLevelPageCount);
  dirStore(9u,select(0u,PUBLISHED,error==0u));}}
`;
