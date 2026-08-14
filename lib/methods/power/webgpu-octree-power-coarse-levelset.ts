/** GPU-resident WP8 coarse-octree advection and exact fine restriction. */

import { OCTREE_COARSE_PHI_BYTES, OCTREE_COARSE_PHI_FLAG } from "../octree-shared/octree-coarse-levelset";
import {
  OCTREE_POWER_COARSE_LEVELSET_VALID,
} from "../octree-shared/octree-power-coarse-levelset-control-abi";
import {
  makeOctreePowerCoarseLevelSetSampleWGSL,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES,
} from "../../core/octree-power-coarse-levelset-sample-abi";
import { OCTREE_FINE_PHI_CONTRIBUTION_BYTES, type OctreeCoarsePhiCorrectionInput,
  type WebGPUOctreeCoarseLevelSet } from "../octree-shared/webgpu-octree-coarse-levelset";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import type { DirectStructuredVelocitySource } from "./webgpu-octree-structured-velocity-gpu";
import type { OctreeAirVelocitySupportLayout } from "./webgpu-octree-air-velocity-support";
import { PassBroker } from "../../core/webgpu-pass-broker";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";

/**
 * The sample directory's strides and its sampling WGSL are the consumer half
 * of this publication and live in
 * `lib/core/octree-power-coarse-levelset-sample-abi.ts`. Re-exported here so
 * the producer and its own tests still name them in one place.
 */
export {
  makeOctreePowerCoarseLevelSetSampleWGSL,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES,
} from "../../core/octree-power-coarse-levelset-sample-abi";

/**
 * The published control word is the consumer half of this producer, and the
 * engine's t=0 acceptance gate reads it without constructing this backend. It
 * lives in `octree-shared/octree-power-coarse-levelset-control-abi.ts`;
 * re-exported here so the producer and its tests keep naming it in one place.
 */
export {
  OCTREE_POWER_COARSE_LEVELSET_ERROR,
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  unpackOctreePowerCoarseLevelSetControl,
  type OctreePowerCoarseLevelSetControl,
} from "../octree-shared/octree-power-coarse-levelset-control-abi";

export const OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES = 64;
export const OCTREE_POWER_COARSE_LEVELSET_DELTA_HEADER_WORDS = 16;
export const OCTREE_POWER_COARSE_LEVELSET_DELTA_RECORD_WORDS = 4;
/** The selector bytes address the generated global table, whose exact extent
 * is smaller than the u8 ABI ceiling. Coarse rows and air tags time-alias this
 * direct-indexed region, so both use the same manifest-proven width. */
export const OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE =
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST.tetrahedronVertexCount;
/** One cold bootstrap plus the solver's bounded 64 encoded surface substeps. */
export const OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS = 65;
/** Portable dynamic-uniform/storage binding alignment. Each command encoder
 * owns one arena, so parameter data cannot alias another unsubmitted encoder. */
export const OCTREE_POWER_COARSE_LEVELSET_PARAM_STRIDE = 256;
export const OCTREE_POWER_COARSE_LEVELSET_DISPATCH_BYTES = 24;

const POWER_COARSE_PHASE_BINDINGS = [
  [0, 1, 8, 13, 15, 16, 17, 18, 19, 21, 24],
  [0, 1, 2, 5, 6, 13, 20, 21],
  [0, 1, 2, 5, 7, 8, 9, 13, 17, 20, 21],
  [0, 9, 11, 12, 13, 17, 21],
  [0, 1, 2, 3, 4, 5, 9, 13, 17, 20, 21],
  [0, 1, 6, 8, 9, 11, 12, 13, 17, 18, 21],
  [0, 13, 15, 16, 17, 18, 19, 21],
] as const;

const POWER_COARSE_STORAGE_BINDINGS = new Set([8, 9, 13, 15, 17, 18, 19, 20, 24]);

export interface OctreePowerCoarseLevelSetPlan {
  readonly rowCapacity: number;
  readonly scratchBytes: number;
  readonly rowStatusBytes: number;
  readonly sampleDirectoryBytes: number;
  readonly deltaBytes: number;
  readonly selectorRowBytes: number;
  readonly selectorOffsetWords: number;
  readonly parameterArenaBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreePowerCoarseLevelSetInput {
  readonly headers: GPUBuffer;
  readonly structured: Pick<DirectStructuredVelocitySource,
    "control" | "rowVelocities" | "rowGeometry" | "rowBankStrideWords" | "liveRowDispatch">;
  /** CPU count or GPU buffer whose first u32 is the compact live-row count. */
  readonly rowCount: number | GPUBuffer | GPUBufferBinding;
  readonly fineCorrection?: OctreeCoarsePhiCorrectionInput & {
    /** Numeric counts for host-authored tests, or a GPU buffer with
     * `(contributionCount, maximumContributionsPerRow)` at byte zero. */
    readonly contributionCount: number | GPUBuffer;
    readonly maximumContributionsPerRow?: number;
    /** One `{centerPhi,minimumPhi,maximumPhi,valid}` record per row. */
    readonly aggregated?: boolean;
  };
}

export interface OctreePowerCoarseLevelSetOptions {
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly dt: number;
  /** Largest power-of-two leaf extent in finest-cell units. */
  readonly maximumLeafSize?: number;
  readonly generation?: number;
}

export interface OctreePowerCoarseDirectoryHeader {
  readonly state: number; readonly generation: number; readonly rowCount: number;
  readonly maximumLeafSize: number; readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number; readonly actualRowCapacity: number;
}

/** CPU mirror of the topology shader's all-or-nothing directory gate. */
export function octreePowerCoarseDirectoryIsAuthoritative(
  header: OctreePowerCoarseDirectoryHeader,
  expectedGeneration: number,
  expectedDimensions: readonly [number, number, number],
  expectedPhysicalCellSize: number,
): boolean {
  const rowCount = header.rowCount;
  return expectedGeneration > 0
    && header.state === OCTREE_POWER_COARSE_LEVELSET_VALID
    && (header.generation & 0x3fff_ffff) === (expectedGeneration & 0x3fff_ffff)
    && header.dimensions.every((value, axis) => value === expectedDimensions[axis])
    && Number.isFinite(header.physicalCellSize) && header.physicalCellSize > 0
    && Math.abs(header.physicalCellSize - expectedPhysicalCellSize)
      <= 1e-5 * Math.max(header.physicalCellSize, expectedPhysicalCellSize)
    && Number.isSafeInteger(rowCount) && rowCount > 0
    && rowCount <= header.actualRowCapacity
    && Number.isSafeInteger(header.maximumLeafSize) && header.maximumLeafSize > 0
    && (header.maximumLeafSize & (header.maximumLeafSize - 1)) === 0;
}

/** One-binding, GPU-published source used to initialize missing fine bricks. */
export interface OctreePowerCoarseLevelSetSampleSource {
  readonly directory: GPUBuffer;
  /** Backing capacity for the compact sorted directory. Only the published
   * header row count is live; stale tail records are never consulted. */
  readonly rowCapacity: number;
  /** Immutable publication control; word two is the exact compact row count. */
  readonly control: GPUBuffer;
  /** Corrected values indexed by current compact row for the fine-seed bridge. */
  readonly values: GPUBuffer;
  /** Exact recurring value/phase delta. The fixed 16-word header is followed
   * by compact `{cellPlusOne,size,rowOrInvalid,flags}` records. */
  readonly delta: GPUBuffer;
  readonly deltaHeaderWords: typeof OCTREE_POWER_COARSE_LEVELSET_DELTA_HEADER_WORDS;
  readonly deltaRecordWords: typeof OCTREE_POWER_COARSE_LEVELSET_DELTA_RECORD_WORDS;
  readonly wgsl: (binding?: number) => string;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be positive`);
  return value;
}
function u32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`${label} must be an unsigned u32`);
  return value;
}

export function planOctreePowerCoarseLevelSet(rowCapacityValue: number, selectorPrefixBytes = 0,
  selectorSuffixBytes = 0, denseComplementCells = 0): OctreePowerCoarseLevelSetPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Power coarse-phi row capacity");
  if (!Number.isSafeInteger(selectorPrefixBytes) || selectorPrefixBytes < 0 || selectorPrefixBytes % 16 !== 0) {
    throw new RangeError("Power coarse-phi selector prefix must be vec4 aligned");
  }
  if (!Number.isSafeInteger(selectorSuffixBytes) || selectorSuffixBytes < 0 || selectorSuffixBytes % 16 !== 0) {
    throw new RangeError("Power coarse-phi selector suffix must be vec4 aligned");
  }
  if (!Number.isSafeInteger(denseComplementCells) || denseComplementCells < 0) {
    throw new RangeError("Power coarse-phi dense complement capacity must be non-negative");
  }
  // The coarse-only mode redistances directly on compact rows. Two row banks
  // provide the dispatch-wide synchronization boundary without allocating a
  // global fine lattice.
  const scratchBytes = 2 * rowCapacity * OCTREE_COARSE_PHI_BYTES;
  const rowStatusBytes = rowCapacity * OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES;
  const sampleDirectoryBytes = OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES
    + (rowCapacity + denseComplementCells) * OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES;
  const deltaBytes = (OCTREE_POWER_COARSE_LEVELSET_DELTA_HEADER_WORDS
    + 2 * rowCapacity * OCTREE_POWER_COARSE_LEVELSET_DELTA_RECORD_WORDS) * 4;
  const selectorOffsetWords = selectorPrefixBytes / 4;
  const selectorRowBytes = selectorPrefixBytes + rowCapacity * OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE * 4
    + selectorSuffixBytes;
  const parameterArenaBytes = OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS
    * OCTREE_POWER_COARSE_LEVELSET_PARAM_STRIDE;
  return { rowCapacity, scratchBytes, rowStatusBytes, sampleDirectoryBytes, deltaBytes,
    selectorRowBytes, selectorOffsetWords,
    parameterArenaBytes,
    allocatedBytes: scratchBytes + rowStatusBytes + 2 * sampleDirectoryBytes + deltaBytes + selectorRowBytes
      + OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + parameterArenaBytes + 32
      + 2 * OCTREE_POWER_COARSE_LEVELSET_DISPATCH_BYTES
      + (rowCapacity + 1) * 4 + OCTREE_FINE_PHI_CONTRIBUTION_BYTES };
}

export class WebGPUOctreePowerCoarseLevelSet {
  readonly plan: OctreePowerCoarseLevelSetPlan;
  readonly control: GPUBuffer;
  readonly sampleDirectory: GPUBuffer;
  private readonly candidateSampleDirectory: GPUBuffer;
  private readonly delta: GPUBuffer;
  private readonly scratch: GPUBuffer;
  /** Direct catalog-selector to current compact-row adjacency. Section 5's
   * coarse transport and structured transition advection consume the same
   * paper-defined local Delaunay vertices. */
  readonly selectorRows: GPUBuffer;
  readonly selectorStride = OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE;
  private readonly rowStatus: GPUBuffer;
  private readonly emptyOffsets: GPUBuffer; private readonly emptyContributions: GPUBuffer;
  private readonly validFineControl: GPUBuffer;
  private readonly dispatchMetadata: GPUBuffer;
  private readonly indirectDispatch: GPUBuffer;
  private readonly phaseLayouts: readonly GPUBindGroupLayout[];
  private preparePipeline!: GPUComputePipeline;
  private buildSelectorRowsPipeline!: GPUComputePipeline;
  private advectPipeline!: GPUComputePipeline;
  private correctPipeline!: GPUComputePipeline;
  private redistanceAtoBPipeline!: GPUComputePipeline;
  private redistanceBtoAPipeline!: GPUComputePipeline;
  private publishPipeline!: GPUComputePipeline;
  private commitPipeline!: GPUComputePipeline;
  private readonly pipelineLayouts: readonly GPUPipelineLayout[];
  private readonly encoderArenas = new WeakMap<GPUCommandEncoder, {
    readonly params: GPUBuffer; invocationCount: number;
  }>();
  /** Submitted command buffers return their arena here. Queue writes issued
   * for the next encoder are ordered after that submission, so recycling
   * avoids a create/destroy allocation on every advance without aliasing two
   * command buffers that are being encoded concurrently. */
  private readonly freeParameterArenas: GPUBuffer[] = [];
  private readonly liveParameterArenas = new Set<GPUBuffer>();
  private readonly bindingCache: {
    readonly params: GPUBuffer;
    readonly resources: readonly GPUBuffer[];
    readonly groups: readonly GPUBindGroup[];
  }[] = [];
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly coarse: WebGPUOctreeCoarseLevelSet,
    private readonly topology: OctreePowerTopologySource, selectorPrefixBytes = 0,
    readonly airSupportLayout?: OctreeAirVelocitySupportLayout,
    denseComplementCells = 0) {
    if (!topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra
      || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Power coarse level set requires the complete tetrahedron catalog");
    }
    const selectorCount = positiveInteger(topology.catalogTetrahedronVertexCount
      ?? OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE, "Power coarse-phi selector count");
    if (selectorCount > OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE) {
      throw new RangeError("Power coarse-phi selector count exceeds the u8 catalog domain");
    }
    if (airSupportLayout && (airSupportLayout.rowCapacity !== coarse.plan.rowCapacity
      || airSupportLayout.transportMetricBytes !== selectorPrefixBytes
      || airSupportLayout.selectorTagOffsetBytes !== selectorPrefixBytes)) {
      throw new RangeError("Power coarse-phi air-support layout does not preserve the selector prefix");
    }
    const selectorBaseBytes = selectorPrefixBytes
      + coarse.plan.rowCapacity * OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE * 4;
    const selectorSuffixBytes = airSupportLayout ? airSupportLayout.totalBytes - selectorBaseBytes : 0;
    if (selectorSuffixBytes < 0) throw new RangeError("Power coarse-phi air-support suffix overlaps selector rows");
    this.plan = planOctreePowerCoarseLevelSet(coarse.plan.rowCapacity, selectorPrefixBytes,
      selectorSuffixBytes, denseComplementCells);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.scratch = device.createBuffer({ label: "Power coarse phi advection and restriction", size: this.plan.scratchBytes, usage: storage });
    this.selectorRows = device.createBuffer({
      label: "Power coarse phi direct selector-row adjacency",
      size: this.plan.selectorRowBytes,
      usage: storage,
    });
    this.control = device.createBuffer({ label: "Power coarse phi schedule control", size: OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, usage: storage });
    this.sampleDirectory = device.createBuffer({ label: "Power coarse phi sorted row directory",
      size: this.plan.sampleDirectoryBytes, usage: storage });
    this.candidateSampleDirectory = device.createBuffer({
      label: "Power coarse phi immutable candidate row directory",
      size: this.plan.sampleDirectoryBytes, usage: storage,
    });
    this.delta = device.createBuffer({ label: "Power coarse phi exact value/phase row delta",
      size: this.plan.deltaBytes, usage: storage });
    this.rowStatus = device.createBuffer({ label: "Power coarse phi fixed row status",
      size: this.plan.rowStatusBytes, usage: storage });
    this.emptyOffsets = device.createBuffer({ label: "Empty fine correction offsets", size: (this.plan.rowCapacity + 1) * 4, usage: storage });
    this.emptyContributions = device.createBuffer({ label: "Empty fine correction contribution", size: OCTREE_FINE_PHI_CONTRIBUTION_BYTES, usage: storage });
    this.validFineControl = device.createBuffer({ label: "Valid host fine-correction control", size: 32, usage: storage });
    this.dispatchMetadata = device.createBuffer({
      label: "Power coarse phi exact dispatch metadata",
      size: OCTREE_POWER_COARSE_LEVELSET_DISPATCH_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.indirectDispatch = device.createBuffer({
      label: "Power coarse phi immutable indirect dispatch",
      size: OCTREE_POWER_COARSE_LEVELSET_DISPATCH_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });
    device.queue.writeBuffer(this.validFineControl, 20, new Uint32Array([OCTREE_POWER_COARSE_LEVELSET_VALID]));
    this.phaseLayouts = POWER_COARSE_PHASE_BINDINGS.map((bindings, phase) =>
      device.createBindGroupLayout({
        label: "Power coarse phi phase " + phase + " bindings",
        entries: bindings.map((binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: binding === 0
            ? { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: 64 }
            : { type: POWER_COARSE_STORAGE_BINDINGS.has(binding)
              ? "storage" as const : "read-only-storage" as const },
        })),
      }));
    this.pipelineLayouts = this.phaseLayouts.map((layout) =>
      device.createPipelineLayout({ bindGroupLayouts: [layout] }));
  }

  async initializePipelines(): Promise<void> {
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({
      label: "Power coarse phi schedule", code: octreePowerCoarseLevelSetShader });
    const pipeline = (phase: number, label: string, entryPoint = label) =>
      compiler.compileComputePipeline({
        label, layout: this.pipelineLayouts[phase]!,
        compute: { module: shaderModule, entryPoint },
      });
    [this.preparePipeline, this.buildSelectorRowsPipeline, this.advectPipeline,
      this.correctPipeline, this.redistanceAtoBPipeline, this.redistanceBtoAPipeline,
      this.publishPipeline, this.commitPipeline] = await Promise.all([
      pipeline(0, "preparePowerCoarsePhiSchedule"),
      pipeline(1, "buildPowerCoarseSelectorRows"),
      pipeline(2, "advectPowerCoarsePhiSchedule"),
      pipeline(3, "correctPowerCoarsePhiSchedule"),
      pipeline(4, "redistancePowerCoarsePhiAtoB"),
      pipeline(4, "redistancePowerCoarsePhiBtoA"),
      pipeline(5, "publishPowerCoarsePhiSchedule"),
      pipeline(6, "commitPowerCoarsePhiSchedule"),
    ]);
  }

  private cachedBindGroups(params: GPUBuffer, common: ReadonlyMap<number, GPUBufferBinding>):
  readonly GPUBindGroup[] {
    const resources = [...new Set(POWER_COARSE_PHASE_BINDINGS.flat())]
      .sort((a, b) => a - b).map((binding) => common.get(binding)!.buffer);
    const cached = this.bindingCache.find((entry) => entry.params === params
      && entry.resources.length === resources.length
      && entry.resources.every((buffer, index) => buffer === resources[index]));
    if (cached) return cached.groups;
    const groups = POWER_COARSE_PHASE_BINDINGS.map((bindings, phase) =>
      this.device.createBindGroup({
        label: "Power coarse phi cached phase " + phase + " bindings",
        layout: this.phaseLayouts[phase]!,
        entries: bindings.map((binding) => ({
          binding,
          resource: binding === 0 ? { buffer: params, size: 64 } : common.get(binding)!,
        })),
      }));
    this.bindingCache.push({ params, resources, groups });
    return groups;
  }

  /** Record immutable coarse evolution/publication and leave its final pass open. */
  encode(broker: PassBroker, input: OctreePowerCoarseLevelSetInput, options: OctreePowerCoarseLevelSetOptions): void {
    if (this.destroyed) throw new Error("Power coarse level-set schedule is destroyed");
    // The invocation arena is command-encoder scoped. Accessing the encoder
    // also closes any preceding compute pass before GPU-authored counts are
    // copied into aligned uniform slots below.
    const encoder = broker.commandEncoder();
    let arena = this.encoderArenas.get(encoder);
    if (!arena) {
      const params = this.freeParameterArenas.pop() ?? this.device.createBuffer({
        label: "Power coarse phi encoder parameter arena",
        size: this.plan.parameterArenaBytes,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      arena = { params, invocationCount: 0 };
      this.encoderArenas.set(encoder, arena);
      this.liveParameterArenas.add(params);
    }
    const encoderInvocation = arena.invocationCount;
    if (encoderInvocation >= OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS) {
      throw new RangeError("Power coarse level-set encoder exceeds its 65 parameter-arena invocations");
    }
    arena.invocationCount += 1;
    const invocationBase = encoderInvocation * OCTREE_POWER_COARSE_LEVELSET_PARAM_STRIDE;
    const maximumRows = typeof input.rowCount === "number" ? u32(input.rowCount, "Power coarse-phi row count") : this.plan.rowCapacity;
    if (maximumRows > this.plan.rowCapacity) throw new RangeError("Power coarse-phi row count exceeds capacity");
    const dimensions = options.dimensions.map((value) => positiveInteger(value, "Power coarse-phi dimension")) as [number, number, number];
    const maximumLeafSize = positiveInteger(options.maximumLeafSize ?? Math.max(...dimensions), "Power coarse-phi maximum leaf size");
    if ((maximumLeafSize & (maximumLeafSize - 1)) !== 0) throw new RangeError("Power coarse-phi maximum leaf size must be a power of two");
    if (!(options.physicalCellSize > 0) || !Number.isFinite(options.physicalCellSize)
      || !Number.isFinite(options.dt) || options.dt < 0) throw new RangeError("Power coarse-phi physical parameters are invalid");
    const fine = input.fineCorrection; const gpuFineCounts = fine !== undefined && typeof fine.contributionCount !== "number";
    const contributionCount = fine === undefined ? 0 : typeof fine.contributionCount === "number"
      ? u32(fine.contributionCount, "Fine correction contribution count") : 0;
    const maximumPerRow = positiveInteger(fine?.maximumContributionsPerRow ?? 1, "Fine correction row bound");
    const generation = u32(options.generation ?? 0, "Power coarse-phi generation");
    const data = new ArrayBuffer(64), words = new Uint32Array(data), floats = new Float32Array(data);
    words.set([...dimensions, this.plan.rowCapacity, maximumRows, contributionCount, maximumPerRow, generation]);
    floats.set([options.physicalCellSize, options.dt], 8); words.set([this.airSupportLayout?.regularTagOffsetWords ?? 0, this.plan.rowCapacity,
      fine ? (fine.aggregated ? 2 : 1) : 0, maximumLeafSize], 10);
    words[15] = this.plan.selectorOffsetWords;
    this.device.queue.writeBuffer(arena.params, invocationBase, data);
    if (typeof input.rowCount !== "number") {
      const binding = "buffer" in input.rowCount
        ? input.rowCount : { buffer: input.rowCount, offset: 0 };
      broker.copyBufferToBuffer(binding.buffer, binding.offset ?? 0,
        arena.params, invocationBase + 16, 4);
    }
    if (gpuFineCounts) broker.copyBufferToBuffer(fine!.contributionCount as GPUBuffer, 0,
      arena.params, invocationBase + 20, 8);
    // The topology publisher's compact affected-row count is a GPU-resident
    // cache invalidation word. Copy it into the existing uniform arena rather
    // than adding an eleventh storage binding to the prepare pipeline.
    broker.copyBufferToBuffer(this.topology.control, 22 * 4, arena.params, invocationBase + 56, 4);
    const offsets = fine?.rowOffsets ?? this.emptyOffsets, contributions = fine?.contributions ?? this.emptyContributions;
    const fineControl = gpuFineCounts ? fine!.contributionCount as GPUBuffer : this.validFineControl;
    const binding = (buffer: GPUBuffer, offset = 0, size?: number): GPUBufferBinding => ({ buffer, offset, ...(size ? { size } : {}) });
    const common = new Map<number, GPUBufferBinding>([[0, binding(arena.params, 0, 64)], [1, binding(input.headers)], [2, binding(this.topology.metrics)],
      [3, binding(this.topology.catalogTetrahedronHeaders!)], [4, binding(this.topology.catalogTetrahedra!)],
      [5, binding(this.topology.catalogTetrahedronVertices!)], [6, binding(input.structured.rowGeometry)], [7, binding(input.structured.rowVelocities)],
      [8, binding(this.coarse.records)], [9, binding(this.scratch)], [11, binding(offsets)], [12, binding(contributions)],
      [13, binding(this.control)], [15, binding(this.sampleDirectory)], [16, binding(fineControl)],
      [17, binding(this.rowStatus)], [18, binding(this.candidateSampleDirectory)],
      [19, binding(this.delta)], [20, binding(this.selectorRows)], [21, binding(input.structured.control)],
      [24, binding(this.dispatchMetadata)]]);
    const bindGroups = this.cachedBindGroups(arena.params, common);
    // Prepare/commit retain bounded control reductions. Every row-parallel
    // phase consumes the structured authority's immutable live-row dispatch;
    // selector reconstruction additionally returns immediately while its
    // topology-epoch cache remains current.
    const prepare = broker.compute({ label: "Power coarse level set · publish exact dispatch" });
    prepare.setPipeline(this.preparePipeline);
    prepare.setBindGroup(0, bindGroups[0]!, [invocationBase]);
    prepare.dispatchWorkgroups(1);
    broker.updateIndirectBuffer(this.dispatchMetadata, 0, this.indirectDispatch, 0,
      OCTREE_POWER_COARSE_LEVELSET_DISPATCH_BYTES);
    const pass = broker.compute({ label: "Power coarse level set · persistent schedule" });
    const runRows = (pipeline: GPUComputePipeline, phase: number, offset: number) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[phase]!, [invocationBase]);
      pass.dispatchWorkgroupsIndirect(this.indirectDispatch, offset);
    };
    // The Section 5 air-support producer owns the complete selector/regular
    // tag suffix as one publication-last transaction.  Letting this schedule
    // rewrite only the direct subset would expose a mixed-generation table to
    // momentum and fine transport.  Standalone coarse-phi users retain the
    // compact direct-row cache when no support layout is installed.
    if (!this.airSupportLayout) runRows(this.buildSelectorRowsPipeline, 1, 12);
    runRows(this.advectPipeline, 2, 0);
    runRows(this.correctPipeline, 3, 0);
    // Factor one has no fine field to restore signed-distance quality after
    // transport. Restore the paper path's eight compact coarse sweeps. The
    // even pass count leaves the accepted result in scratch bank A, which is
    // also the unchanged fine-restriction publication source.
    const coarseRedistance = !fine && options.dt > 0
      && (typeof process === "undefined"
        || process.env.FLUID_OCTREE_COARSE_REDISTANCE !== "0");
    if (coarseRedistance) {
      runRows(this.redistanceAtoBPipeline, 4, 0);
      runRows(this.redistanceBtoAPipeline, 4, 0);
      runRows(this.redistanceAtoBPipeline, 4, 0);
      runRows(this.redistanceBtoAPipeline, 4, 0);
      runRows(this.redistanceAtoBPipeline, 4, 0);
      runRows(this.redistanceBtoAPipeline, 4, 0);
      runRows(this.redistanceAtoBPipeline, 4, 0);
      runRows(this.redistanceBtoAPipeline, 4, 0);
    }
    runRows(this.publishPipeline, 5, 0);
    pass.setPipeline(this.commitPipeline);
    pass.setBindGroup(0, bindGroups[6]!, [invocationBase]);
    pass.dispatchWorkgroups(1);
    broker.fence("power coarse level-set schedule complete");
  }

  /** Call after submitting a finished encoder to release its private arena.
   * Other command buffers may be encoded before this call because their
   * parameter storage is disjoint. */
  retireSubmittedEncoder(encoder: GPUCommandEncoder): void {
    const arena = this.encoderArenas.get(encoder); if (!arena) return;
    this.encoderArenas.delete(encoder);
    this.freeParameterArenas.push(arena.params);
  }

  get sampleSource(): OctreePowerCoarseLevelSetSampleSource {
    return { directory: this.sampleDirectory, rowCapacity: this.plan.rowCapacity,
      control: this.control, values: this.coarse.records, delta: this.delta,
      deltaHeaderWords: OCTREE_POWER_COARSE_LEVELSET_DELTA_HEADER_WORDS,
      deltaRecordWords: OCTREE_POWER_COARSE_LEVELSET_DELTA_RECORD_WORDS,
      wgsl: makeOctreePowerCoarseLevelSetSampleWGSL };
  }

  /** Rejected-generation QA only; never consumed by simulation work. */
  get diagnosticRowStatus(): GPUBuffer { return this.rowStatus; }
  /** Rejected-generation QA only; never consumed by simulation work. */
  get diagnosticCandidateSampleDirectory(): GPUBuffer { return this.candidateSampleDirectory; }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.scratch.destroy(); this.selectorRows.destroy(); this.control.destroy(); this.sampleDirectory.destroy();
    this.dispatchMetadata.destroy(); this.indirectDispatch.destroy(); this.bindingCache.length = 0;
    this.candidateSampleDirectory.destroy(); this.delta.destroy(); this.rowStatus.destroy();
    this.liveParameterArenas.forEach((buffer) => buffer.destroy()); this.liveParameterArenas.clear();
    this.freeParameterArenas.length = 0;
    this.emptyOffsets.destroy(); this.emptyContributions.destroy(); this.validFineControl.destroy(); }
}

export const octreePowerCoarseLevelSetShader = /* wgsl */ `
struct Params { dimensionsCapacity:vec4u, countsGeneration:vec4u, physical:vec2f, regularTagOffsetWords:u32, structuredRowCapacity:u32, hasFine:u32, maximumLeafSize:u32, pad0:u32, selectorOffsetWords:u32 }
struct LeafHeader { cell:u32, entryStart:u32, entryCount:u32, size:u32, diagonal:f32, rhs:f32, pad0:u32, pad1:u32, gradient:vec4f }
struct Metric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct TetraHeader { first:u32, count:u32, flags:u32 }
struct TetraVertex { offsetSize:vec4f }
struct CoarsePhi { phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32 }
struct FineContribution { phi:f32, distanceSquared:f32, valid:u32, pad:u32 }
struct SampleEntry { cellPlusOne:u32, size:u32, phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32, row:u32, physicalVolume:f32 }
struct SampleDirectory { state:u32, generation:u32, rowCount:u32, maximumLeafSize:u32, dimensions:vec3u, physicalCellSize:f32, entries:array<SampleEntry> }
struct RowStatus { flags:u32, advected:u32, uniform:u32, transition:u32, corrected:u32, interfaceRow:u32, physicalVolume:f32, pad:u32 }
struct Control { flags:u32, firstError:u32, rowCount:u32, advected:u32, uniform:u32, transition:u32, passes:u32, corrected:u32, interfaces:u32, contributionCount:u32, generation:u32, valid:u32, pad0:u32, pad1:u32, pad2:u32, pad3:u32 }
@group(0) @binding(0) var<uniform> params:Params;@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;@group(0) @binding(2) var<storage,read> metrics:array<Metric>;
@group(0) @binding(3) var<storage,read> tetraHeaders:array<TetraHeader>;@group(0) @binding(4) var<storage,read> tetrahedra:array<u32>;
@group(0) @binding(5) var<storage,read> vertices:array<TetraVertex>;
@group(0) @binding(6) var<storage,read> rowGeometry:array<vec4u>;@group(0) @binding(7) var<storage,read> velocities:array<vec4f>;@group(0) @binding(8) var<storage,read_write> coarse:array<CoarsePhi>;
@group(0) @binding(9) var<storage,read_write> scratchA:array<CoarsePhi>;
@group(0) @binding(11) var<storage,read> fineOffsets:array<u32>;@group(0) @binding(12) var<storage,read> fine:array<FineContribution>;@group(0) @binding(13) var<storage,read_write> control:Control;
@group(0) @binding(15) var<storage,read_write> sampleDirectory:SampleDirectory;
@group(0) @binding(16) var<storage,read> fineControl:array<u32>;
@group(0) @binding(17) var<storage,read_write> rowStatus:array<RowStatus>;
@group(0) @binding(18) var<storage,read_write> candidateDirectory:SampleDirectory;
struct DeltaRecord { cellPlusOne:u32, size:u32, row:u32, flags:u32 }
struct DeltaPublication { count:u32, generation:u32, flags:u32, valid:u32, pad:array<u32,12>, records:array<DeltaRecord> }
@group(0) @binding(19) var<storage,read_write> coarseDeltaPublication:DeltaPublication;
@group(0) @binding(20) var<storage,read_write> selectorRows:array<u32>;
@group(0) @binding(21) var<storage,read> structuredControl:array<u32>;
@group(0) @binding(24) var<storage,read_write> dispatchMetadata:array<u32>;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;const CAPACITY:u32=1u;const INVALID_ROW:u32=2u;const INVALID_VELOCITY:u32=4u;const INVALID_CATALOG:u32=8u;const INVALID_FINE_OFFSETS:u32=16u;const INVALID_FINE_SAMPLE:u32=32u;const FINE_BOUND:u32=64u;const INVALID_SOURCE:u32=256u;const NO_CAUSAL_SIMPLEX:u32=512u;const UNIFORM:u32=1u;
const PHI_VALID:u32=${OCTREE_COARSE_PHI_FLAG.valid}u;const PHI_CORRECTED:u32=${OCTREE_COARSE_PHI_FLAG.correctedFromFine}u;const PHI_INTERFACE:u32=${OCTREE_COARSE_PHI_FLAG.containsInterface}u;const PHI_FINITE:u32=${OCTREE_COARSE_PHI_FLAG.finite}u;const MIGRATED_AIR:u32=16u;
const COARSE_PREDICTED_WET_MAGIC:u32=0x43505754u;
fn finite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;}fn structuredValid()->bool{return arrayLength(&structuredControl)>=6u&&structuredControl[0]==0u&&structuredControl[3]!=0u&&structuredControl[4]<=1u&&structuredControl[2]==params.countsGeneration.x&&structuredControl[2]<=params.dimensionsCapacity.w;}fn sourceRequested()->u32{return select(0u,params.countsGeneration.x,structuredValid());}fn requested()->u32{return sourceRequested();}fn rejectedFine()->bool{return params.hasFine!=0u&&(arrayLength(&fineControl)<6u||fineControl[0]==INVALID||fineControl[5]!=VALID);}fn dims()->vec3u{return params.dimensionsCapacity.xyz;}fn bankBase()->u32{return structuredControl[4]*params.structuredRowCapacity;}fn geometry(row:u32)->vec4u{return rowGeometry[bankBase()+row];}
fn coord(cell:u32)->vec3u{let d=dims();return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}fn center(row:u32)->vec3f{return (vec3f(coord(headers[row].cell))+0.5*f32(headers[row].size))*params.physical.x;}fn size(row:u32)->f32{return f32(headers[row].size)*params.physical.x;}
fn inverseTransform(value:vec3f,code:u32)->vec3f{let bits=code&7u;let q=value*vec3f(select(1.0,-1.0,(bits&1u)!=0u),select(1.0,-1.0,(bits&2u)!=0u),select(1.0,-1.0,(bits&4u)!=0u));let p=(code/8u)%6u;if(p==0u){return q.xyz;}if(p==1u){return q.xzy;}if(p==2u){return q.yxz;}if(p==3u){return q.zxy;}if(p==4u){return q.yzx;}return q.zyx;}
fn mortonPart10(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn morton(cell:u32)->u32{let q=coord(cell);return mortonPart10(q.x)|(mortonPart10(q.y)<<1u)|(mortonPart10(q.z)<<2u);}
fn level(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn directoryLess(aLevel:u32,aMorton:u32,bLevel:u32,bMorton:u32)->bool{return aLevel<bLevel||(aLevel==bLevel&&aMorton<bMorton);}
fn findSite(c:vec3f,s:f32)->u32{let grid=s/params.physical.x;let o=c/params.physical.x-0.5*grid;let rounded=round(o);if(abs(grid-round(grid))>2e-4||any(abs(o-rounded)>vec3f(2e-4))||any(rounded<vec3f(0.0))||any(rounded>=vec3f(dims()))){return INVALID;}let q=vec3u(rounded);let cell=q.x+dims().x*(q.y+dims().y*q.z);let wantedSize=u32(round(grid));let wantedLevel=level(wantedSize);let wantedMorton=morton(cell);let count=min(sourceRequested(),params.structuredRowCapacity);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let candidate=geometry(middle);if(directoryLess(level(candidate.y),morton(candidate.x),wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let candidate=geometry(low);if(candidate.x==cell&&candidate.y==wantedSize){return low;}}return INVALID;}
fn selectorRow(row:u32,selector:u32)->u32{let index=params.selectorOffsetWords+row*${OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE}u+selector;if(selector>=${OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE}u||index>=arrayLength(&selectorRows)){return INVALID;}return selectorRows[index];}
fn buildSelectorRow(row:u32){let selectorCount=min(arrayLength(&vertices),${OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE}u);if(control.pad3==0u||control.pad0==0u||row>=requested()||row>=arrayLength(&headers)||row>=arrayLength(&metrics)){return;}let metric=metrics[row];for(var selector=0u;selector<selectorCount;selector+=1u){let output=params.selectorOffsetWords+row*${OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE}u+selector;if(output>=arrayLength(&selectorRows)){return;}let vertex=vertices[selector].offsetSize;let c=center(row)+size(row)*inverseTransform(vertex.xyz,metric.transformAndFlags&63u);selectorRows[output]=findSite(c,size(row)*vertex.w);}for(var selector=selectorCount;selector<${OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE}u;selector+=1u){let output=params.selectorOffsetWords+row*${OCTREE_POWER_COARSE_LEVELSET_SELECTOR_STRIDE}u+selector;if(output<arrayLength(&selectorRows)){selectorRows[output]=INVALID;}}if(params.regularTagOffsetWords!=0u){for(var stencil=0u;stencil<27u;stencil+=1u){let output=params.regularTagOffsetWords+27u*row+stencil;if(output>=arrayLength(&selectorRows)){return;}let offset=vec3f(f32(i32(stencil%3u)-1),f32(i32((stencil/3u)%3u)-1),f32(i32(stencil/9u)-1));selectorRows[output]=findSite(center(row)+size(row)*offset,size(row));}}}
fn fail(row:u32,flag:u32){if(row<arrayLength(&rowStatus)){rowStatus[row].flags|=flag;}}fn solveGradient(m:mat3x3f,b:vec3f)->vec4f{let xx=m[0].x;let xy=m[0].y;let xz=m[0].z;let yy=m[1].y;let yz=m[1].z;let zz=m[2].z;let c00=yy*zz-yz*yz;let c01=xz*yz-xy*zz;let c02=xy*yz-xz*yy;let c11=xx*zz-xz*xz;let c12=xy*xz-xx*yz;let c22=xx*yy-xy*xy;let detValue=xx*c00+xy*c01+xz*c02;if(!finite(detValue)||abs(detValue)<=1e-9){return vec4f(0.0);}return vec4f(vec3f(c00*b.x+c01*b.y+c02*b.z,c01*b.x+c11*b.y+c12*b.z,c02*b.x+c12*b.y+c22*b.z)/detValue,1.0);}
fn previousSampleSlot(cell:u32,s:u32)->u32{let count=min(sampleDirectory.rowCount,arrayLength(&sampleDirectory.entries));let wantedLevel=level(s);let wantedMorton=morton(cell);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let entry=sampleDirectory.entries[middle];let entryMorton=morton(entry.cellPlusOne-1u);if(directoryLess(level(entry.size),entryMorton,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let entry=sampleDirectory.entries[low];if(entry.cellPlusOne==cell+1u&&entry.size==s){return low;}}return INVALID;}
fn candidateSampleSlot(cell:u32,s:u32)->u32{let count=min(candidateDirectory.rowCount,arrayLength(&candidateDirectory.entries));let wantedLevel=level(s);let wantedMorton=morton(cell);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let entry=candidateDirectory.entries[middle];let entryMorton=morton(entry.cellPlusOne-1u);if(directoryLess(level(entry.size),entryMorton,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let entry=candidateDirectory.entries[low];if(entry.cellPlusOne==cell+1u&&entry.size==s){return low;}}return INVALID;}
fn previousSampleAtCurrentLeaf(header:LeafHeader)->CoarsePhi{var slot=previousSampleSlot(header.cell,header.size);if(slot!=INVALID){let flags=sampleDirectory.entries[slot].flags;let phi=sampleDirectory.entries[slot].phi;let minimumPhi=sampleDirectory.entries[slot].minimumPhi;let maximumPhi=sampleDirectory.entries[slot].maximumPhi;if((flags&(PHI_VALID|PHI_FINITE))==(PHI_VALID|PHI_FINITE)&&finite(phi)&&finite(minimumPhi)&&finite(maximumPhi)&&minimumPhi<=maximumPhi){return CoarsePhi(phi,minimumPhi,maximumPhi,flags);}}
  let d=dims();let origin=coord(header.cell);let half=header.size/2u;var q=min(origin+vec3u(half),d-vec3u(1u));var scale=1u;loop{let owner=(q/vec3u(scale))*vec3u(scale);let cell=owner.x+d.x*(owner.y+d.y*owner.z);slot=previousSampleSlot(cell,scale);if(slot!=INVALID){let flags=sampleDirectory.entries[slot].flags;let phi=sampleDirectory.entries[slot].phi;let minimumPhi=sampleDirectory.entries[slot].minimumPhi;let maximumPhi=sampleDirectory.entries[slot].maximumPhi;if((flags&(PHI_VALID|PHI_FINITE))==(PHI_VALID|PHI_FINITE)&&finite(phi)&&finite(minimumPhi)&&finite(maximumPhi)&&minimumPhi<=maximumPhi){return CoarsePhi(phi,minimumPhi,maximumPhi,flags);}}if(scale>=params.maximumLeafSize){break;}scale*=2u;}
  if(header.pad1==COARSE_PREDICTED_WET_MAGIC){let seed=bitcast<f32>(header.pad0);if(finite(seed)&&seed<0.0){return CoarsePhi(seed,seed,seed,PHI_VALID|PHI_FINITE|MIGRATED_AIR);}}
  let air=params.physical.x*f32(max(1u,params.maximumLeafSize));return CoarsePhi(air,air,air,PHI_VALID|PHI_FINITE|MIGRATED_AIR);
}
fn migratePowerCoarsePhiSource(row:u32){if(sampleDirectory.state!=VALID||sampleDirectory.rowCount>arrayLength(&sampleDirectory.entries)){return;}let count=sourceRequested();if(row>=count||row>=params.dimensionsCapacity.w||row>=arrayLength(&headers)||row>=arrayLength(&coarse)){return;}let header=headers[row];if(header.size==0u||header.cell>=dims().x*dims().y*dims().z){return;}coarse[row]=previousSampleAtCurrentLeaf(header);}
fn preparePowerCoarsePhi(){let count=sourceRequested();control=Control(select(0u,CAPACITY,params.countsGeneration.x>params.dimensionsCapacity.w),0xffffffffu,count,0u,0u,0u,select(0u,8u,params.hasFine==0u&&params.physical.y>0.0),0u,0u,params.countsGeneration.y,params.countsGeneration.w,0u,0u,0u,0u,0u);candidateDirectory.state=0u;candidateDirectory.generation=params.countsGeneration.w;candidateDirectory.rowCount=count;candidateDirectory.maximumLeafSize=params.maximumLeafSize;candidateDirectory.dimensions=params.dimensionsCapacity.xyz;candidateDirectory.physicalCellSize=params.physical.x;coarseDeltaPublication.count=0u;coarseDeltaPublication.generation=params.countsGeneration.w;coarseDeltaPublication.flags=0u;coarseDeltaPublication.valid=0u;}
fn clearPowerCoarsePhiRowStatus(row:u32){if(row<arrayLength(&rowStatus)){rowStatus[row]=RowStatus(0u,0u,0u,0u,0u,0u,0.0,0u);}}
fn advectPowerCoarsePhi(row:u32){if(row>=requested()||row>=params.dimensionsCapacity.w){return;}let velocityAt=bankBase()+row;if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)||velocityAt>=arrayLength(&velocities)||row>=arrayLength(&coarse)||row>=arrayLength(&scratchA)||row>=arrayLength(&rowStatus)){fail(row,CAPACITY);return;}let metric=metrics[row];if((metric.transformAndFlags&VALID)==0u){fail(row,INVALID_ROW);return;}let extent=f32(headers[row].size)*params.physical.x;let physicalVolume=metric.volume*extent*extent*extent;if(!finite(physicalVolume)||physicalVolume<=0.0){fail(row,INVALID_ROW);return;}rowStatus[row].physicalVolume=physicalVolume;let velocity=velocities[velocityAt];if(params.physical.y>0.0&&(!finite(velocity.x)||!finite(velocity.y)||!finite(velocity.z)||velocity.w<=0.0)){fail(row,INVALID_VELOCITY);return;}let source=coarse[row];if((source.flags&(PHI_VALID|PHI_FINITE))!=(PHI_VALID|PHI_FINITE)||!finite(source.phi)||!finite(source.minimumPhi)||!finite(source.maximumPhi)||source.minimumPhi>source.maximumPhi||source.phi<source.minimumPhi||source.phi>source.maximumPhi){fail(row,INVALID_SOURCE);return;}var matrix=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));var rhs=vec3f(0.0);
/* Restriction-only schedule entry (POWER_LIQUIDS_ULTIMATE_M1MAX.md B3 / P1.3).
   The assembled least-squares gradient is consumed by exactly one expression,
   itself guarded by params.physical.y>0.0, so at dt==0 the whole assembly is
   dead. Skipping it leaves matrix/rhs at zero, solveGradient then returns
   vec4f(0.0) through its abs(detValue)<=1e-9 branch, and gradient.w==0.0
   selects the same false branch dt==0 already selected. Value-neutral: the
   dt==0 call site keeps correct -> publish -> commit, its coarse delta
   directory, its PHI_INTERFACE recomputation and its physical volume. */
if(params.physical.y>0.0){for(var selector=0u;selector<arrayLength(&vertices);selector+=1u){let neighbor=selectorRow(row,selector);if(neighbor==INVALID||neighbor>=requested()||neighbor>=arrayLength(&headers)||neighbor>=arrayLength(&metrics)||neighbor>=arrayLength(&coarse)){continue;}let delta=center(neighbor)-center(row);let length2=dot(delta,delta);let other=coarse[neighbor];if((other.flags&(PHI_VALID|PHI_FINITE))!=(PHI_VALID|PHI_FINITE)||length2<=1e-12||!finite(other.phi)){continue;}let weight=1.0/length2;matrix+=weight*mat3x3f(delta*delta.x,delta*delta.y,delta*delta.z);rhs+=weight*delta*(other.phi-source.phi);}}
let gradient=solveGradient(matrix,rhs);var shift=0.0;if(params.physical.y>0.0&&gradient.w>0.0){let proposed=-params.physical.y*dot(velocity.xyz,gradient.xyz);let displacement=params.physical.y*length(velocity.xyz);if(finite(proposed)&&finite(displacement)){shift=clamp(proposed,-displacement,displacement);}}let value=source.phi+shift;let shiftedMinimum=min(value,source.minimumPhi+shift);let shiftedMaximum=max(value,source.maximumPhi+shift);scratchA[row]=CoarsePhi(value,shiftedMinimum,shiftedMaximum,(source.flags&(~PHI_CORRECTED))|PHI_VALID|PHI_FINITE);rowStatus[row].advected=1u;}
fn applyExactFineCorrection(row:u32){if(params.hasFine!=2u||row>=requested()||row>=arrayLength(&rowStatus)||rowStatus[row].flags!=0u||row+1u>=arrayLength(&fineOffsets)){return;}let begin=fineOffsets[row];let end=fineOffsets[row+1u];if(end<=begin||begin>=arrayLength(&fine)){return;}let aggregate=fine[begin];if(aggregate.pad==0u){return;}var output=scratchA[row];output.phi=aggregate.phi;output.minimumPhi=aggregate.distanceSquared;output.maximumPhi=bitcast<f32>(aggregate.valid);output.flags=(output.flags&(~(MIGRATED_AIR|PHI_INTERFACE)))|PHI_CORRECTED|PHI_VALID|PHI_FINITE;scratchA[row]=output;}
fn solveTranspose(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let d=dot(a,cross(b,c));if(!finite(d)||abs(d)<=1e-10){return vec4f(0.0);}return vec4f((rhs.x*cross(b,c)+rhs.y*cross(c,a)+rhs.z*cross(a,b))/d,1.0);}
fn solveColumns(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let d=dot(a,cross(b,c));if(!finite(d)||abs(d)<=1e-10){return vec4f(0.0);}return vec4f(dot(rhs,cross(b,c)),dot(a,cross(rhs,c)),dot(a,cross(b,rhs)),d);}
fn nonobtuse(a:vec3f,b:vec3f,c:vec3f)->bool{let den=length(a)*length(b)*length(c)+dot(a,b)*length(c)+dot(a,c)*length(b)+dot(b,c)*length(a);let num=abs(dot(a,cross(b,c)));return den+2e-6*max(1.0,max(abs(den),num))>=num;}
fn causalTetraCandidate(q:mat3x3f,known:vec3f)->f32{let unavailable=1e30;let av=solveTranspose(q[0],q[1],q[2],known);let bv=solveTranspose(q[0],q[1],q[2],vec3f(1.0));if(av.w==0.0||bv.w==0.0){return unavailable;}let aa=dot(bv.xyz,bv.xyz);let bb=dot(av.xyz,bv.xyz);let cc=dot(av.xyz,av.xyz)-1.0;let disc=bb*bb-aa*cc;if(!finite(aa)||aa<=1e-12||!finite(disc)||disc<0.0){return unavailable;}let candidate=(bb+sqrt(disc))/aa;if(!finite(candidate)||candidate+2e-6<max(known.x,max(known.y,known.z))){return unavailable;}let ray=solveColumns(q[0],q[1],q[2],-(av.xyz-candidate*bv.xyz));if(ray.w==0.0){return unavailable;}let coefficients=ray.xyz/ray.w;let sum=coefficients.x+coefficients.y+coefficients.z;if(!finite(sum)||sum<=2e-6||any(coefficients/sum<vec3f(-2e-6))){return unavailable;}return candidate;}
fn causalTriangleCandidate(a:vec3f,b:vec3f,known:vec2f)->f32{let unavailable=1e30;let g00=dot(a,a);let g01=dot(a,b);let g11=dot(b,b);let determinant=g00*g11-g01*g01;if(!finite(determinant)||determinant<=1e-12||g01+2e-6*max(1.0,sqrt(g00*g11))<0.0){return unavailable;}let av=a*((g11*known.x-g01*known.y)/determinant)+b*((g00*known.y-g01*known.x)/determinant);let bv=a*((g11-g01)/determinant)+b*((g00-g01)/determinant);let aa=dot(bv,bv);let bb=dot(av,bv);let cc=dot(av,av)-1.0;let disc=bb*bb-aa*cc;if(!finite(aa)||aa<=1e-12||!finite(disc)||disc<0.0){return unavailable;}let candidate=(bb+sqrt(disc))/aa;if(!finite(candidate)||candidate+2e-6<max(known.x,known.y)){return unavailable;}let delta=vec2f(candidate)-known;let coefficients=vec2f(g11*delta.x-g01*delta.y,g00*delta.y-g01*delta.x)/determinant;let sum=coefficients.x+coefficients.y;if(!finite(sum)||sum<=2e-6||any(coefficients/sum<vec2f(-2e-6))){return unavailable;}return candidate;}
fn causalEdgeCandidate(offset:vec3f,known:f32)->f32{let distance=length(offset);if(!finite(distance)||distance<=1e-6||!finite(known)){return 1e30;}return known+distance;}
fn eikonal3(values:vec3f,h:f32)->f32{var a=values;if(a.x>a.y){a=vec3f(a.y,a.x,a.z);}if(a.y>a.z){a=vec3f(a.x,a.z,a.y);}if(a.x>a.y){a=vec3f(a.y,a.x,a.z);}var u=a.x+h;if(u>a.y){u=0.5*(a.x+a.y+sqrt(max(0.0,2.0*h*h-(a.x-a.y)*(a.x-a.y))));}if(u>a.z){let disc=3.0*h*h-(a.x-a.y)*(a.x-a.y)-(a.x-a.z)*(a.x-a.z)-(a.y-a.z)*(a.y-a.z);u=(a.x+a.y+a.z+sqrt(max(0.0,disc)))/3.0;}return u;}
fn redistancePowerCoarsePhi(row:u32,fromA:bool){
  if(params.hasFine!=0u||row>=requested()||row>=params.dimensionsCapacity.w){return;}
  let capacity=params.dimensionsCapacity.w;
  let sourceIndex=select(capacity+row,row,fromA);
  let destinationIndex=select(row,capacity+row,fromA);
  if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)||row>=arrayLength(&rowStatus)||sourceIndex>=arrayLength(&scratchA)||destinationIndex>=arrayLength(&scratchA)){fail(row,CAPACITY);return;}
  var source=scratchA[sourceIndex];
  var fixedSeed=(source.flags&(PHI_CORRECTED|PHI_INTERFACE))!=0u;
  if(!fixedSeed){for(var seedSelector=0u;seedSelector<arrayLength(&vertices);seedSelector+=1u){let seedNeighbor=selectorRow(row,seedSelector);if(seedNeighbor==INVALID||seedNeighbor>=requested()){continue;}let seedIndex=select(capacity+seedNeighbor,seedNeighbor,fromA);if(seedIndex>=arrayLength(&scratchA)){continue;}let seedPhi=scratchA[seedIndex].phi;if(finite(seedPhi)&&((source.phi<0.0&&seedPhi>=0.0)||(source.phi>=0.0&&seedPhi<0.0))){fixedSeed=true;break;}}}
  if(fixedSeed){source.flags|=PHI_INTERFACE;scratchA[destinationIndex]=source;return;}
  let metric=metrics[row];
  if(metric.topologyCode>=arrayLength(&tetraHeaders)){fail(row,INVALID_CATALOG);return;}
  let header=tetraHeaders[metric.topologyCode];
  var magnitude=1e30;var used=false;
  if((header.flags&UNIFORM)!=0u){
    var axes=vec3f(1e30);
    for(var selector=0u;selector<arrayLength(&vertices);selector+=1u){let v=vertices[selector].offsetSize;if(abs(v.w-1.0)>1e-5){continue;}let world=inverseTransform(v.xyz,metric.transformAndFlags&63u);if(abs(length(world)-1.0)>1e-5){continue;}let neighbor=selectorRow(row,selector);if(neighbor==INVALID||neighbor>=requested()){continue;}let index=select(capacity+neighbor,neighbor,fromA);if(index>=arrayLength(&scratchA)){continue;}let phi=abs(scratchA[index].phi);let axis=select(select(2u,1u,abs(world.y)>.5),0u,abs(world.x)>.5);axes[axis]=min(axes[axis],phi);}
    // Boundary rows may lack the outward neighbor on one or two axes. The
    // sorted eikonal update naturally reduces to its 2-D or 1-D form when an
    // axis remains unavailable, so require at least one causal neighbor rather
    // than incorrectly rejecting every domain-boundary row.
    if(any(axes<vec3f(1e29))){magnitude=eikonal3(axes,size(row));used=true;}
    rowStatus[row].uniform+=1u;
  }else{
    if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){fail(row,INVALID_CATALOG);return;}
    for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let s=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(s>=vec3u(arrayLength(&vertices)))){fail(row,INVALID_CATALOG);return;}let rows=vec3u(selectorRow(row,s.x),selectorRow(row,s.y),selectorRow(row,s.z));if(any(rows==vec3u(INVALID))||any(rows>=vec3u(requested()))){continue;}let indices=select(vec3u(capacity)+rows,rows,vec3<bool>(fromA));if(any(indices>=vec3u(arrayLength(&scratchA)))){continue;}let q=mat3x3f(size(row)*inverseTransform(vertices[s.x].offsetSize.xyz,metric.transformAndFlags&63u),size(row)*inverseTransform(vertices[s.y].offsetSize.xyz,metric.transformAndFlags&63u),size(row)*inverseTransform(vertices[s.z].offsetSize.xyz,metric.transformAndFlags&63u));if(!nonobtuse(q[0],q[1],q[2])){continue;}let known=vec3f(abs(scratchA[indices.x].phi),abs(scratchA[indices.y].phi),abs(scratchA[indices.z].phi));let full=causalTetraCandidate(q,known);let face0=causalTriangleCandidate(q[0],q[1],known.xy);let face1=causalTriangleCandidate(q[0],q[2],known.xz);let face2=causalTriangleCandidate(q[1],q[2],known.yz);let edge0=causalEdgeCandidate(q[0],known.x);let edge1=causalEdgeCandidate(q[1],known.y);let edge2=causalEdgeCandidate(q[2],known.z);let candidate=min(min(full,min(face0,min(face1,face2))),min(edge0,min(edge1,edge2)));if(candidate<1e29){magnitude=min(magnitude,candidate);used=true;}}
    rowStatus[row].transition+=1u;
  }
  // An isolated boundary/coarse leaf can have no same-level causal simplex.
  // A monotone sweep has no information with which to reduce its distance, so
  // its finite incoming value is already the local fixed point.
  if(!used){scratchA[destinationIndex]=source;return;}
  let sign=select(1.0,-1.0,source.phi<0.0);let value=sign*min(abs(source.phi),magnitude);
  scratchA[destinationIndex]=CoarsePhi(value,value,value,(source.flags&(~PHI_INTERFACE))|PHI_VALID|PHI_FINITE);
}
fn validatePowerCoarseFineCorrection(row:u32){if(params.hasFine==0u||row>=requested()||row>=arrayLength(&rowStatus)){return;}if(row+1u>=arrayLength(&fineOffsets)){fail(row,INVALID_FINE_OFFSETS);return;}let begin=fineOffsets[row];let end=fineOffsets[row+1u];if((row==0u&&begin!=0u)||(row+1u==requested()&&end!=params.countsGeneration.y)||end<begin||end>params.countsGeneration.y||end>arrayLength(&fine)){fail(row,INVALID_FINE_OFFSETS);return;}if(end-begin>params.countsGeneration.z){fail(row,FINE_BOUND);return;}for(var cursor=begin;cursor<end;cursor+=1u){let sample=fine[cursor];if(params.hasFine==2u){let maximum=bitcast<f32>(sample.valid);if(sample.pad!=0u&&(!finite(sample.phi)||!finite(sample.distanceSquared)||!finite(maximum)||sample.distanceSquared>maximum)){fail(row,INVALID_FINE_SAMPLE);return;}}else if(sample.valid!=0u&&(!finite(sample.phi)||!finite(sample.distanceSquared)||sample.distanceSquared<0.0)){fail(row,INVALID_FINE_SAMPLE);return;}}}
fn publishPowerCoarsePhi(slot:u32){if(slot>=requested()||bankBase()+slot>=arrayLength(&rowGeometry)||slot>=arrayLength(&candidateDirectory.entries)){return;}candidateDirectory.entries[slot]=SampleEntry(0u,0u,0.0,0.0,0.0,0u,INVALID,0.0);let descriptor=geometry(slot);let row=slot;if(row>=arrayLength(&headers)||row>=arrayLength(&coarse)||row>=arrayLength(&scratchA)||row>=arrayLength(&rowStatus)){return;}let header=headers[row];var descriptorValid=descriptor.x==header.cell&&descriptor.y==header.size;if(slot>0u){let prior=geometry(slot-1u);descriptorValid=descriptorValid&&directoryLess(level(prior.y),morton(prior.x),level(descriptor.y),morton(descriptor.x));}if(slot+1u<requested()){let following=geometry(slot+1u);descriptorValid=descriptorValid&&directoryLess(level(descriptor.y),morton(descriptor.x),level(following.y),morton(following.x));}if(!descriptorValid||rowStatus[row].flags!=0u){return;}var output=scratchA[row];if(params.hasFine!=0u){let begin=fineOffsets[row];let end=fineOffsets[row+1u];if(params.hasFine==2u){if(end>begin){let aggregate=fine[begin];if(aggregate.pad!=0u){output.phi=aggregate.phi;output.minimumPhi=aggregate.distanceSquared;output.maximumPhi=bitcast<f32>(aggregate.valid);output.flags=(output.flags&(~MIGRATED_AIR))|PHI_CORRECTED;rowStatus[row].corrected=1u;}}}else{var nearest=1e30;var minimum=1e30;var maximum=-1e30;var count=0u;for(var cursor=begin;cursor<end;cursor+=1u){let sample=fine[cursor];if(sample.valid==0u){continue;}minimum=min(minimum,sample.phi);maximum=max(maximum,sample.phi);if(sample.distanceSquared<nearest||(sample.distanceSquared==nearest&&sample.phi<output.phi)){nearest=sample.distanceSquared;output.phi=sample.phi;}count+=1u;}if(count>0u){output.minimumPhi=minimum;output.maximumPhi=maximum;output.flags=(output.flags&(~MIGRATED_AIR))|PHI_CORRECTED;rowStatus[row].corrected=1u;}}}output.flags&=~MIGRATED_AIR;output.minimumPhi=min(output.minimumPhi,output.phi);output.maximumPhi=max(output.maximumPhi,output.phi);if(output.minimumPhi<=0.0&&output.maximumPhi>=0.0){output.flags|=PHI_INTERFACE;rowStatus[row].interfaceRow=1u;}coarse[row]=output;candidateDirectory.entries[slot]=SampleEntry(descriptor.x+1u,descriptor.y,output.phi,output.minimumPhi,output.maximumPhi,output.flags,row,rowStatus[row].physicalVolume);}
var<workgroup> reduceFlags:array<u32,256>;var<workgroup> reduceFirst:array<u32,256>;var<workgroup> reduceAdvected:array<u32,256>;var<workgroup> reduceUniform:array<u32,256>;var<workgroup> reduceTransition:array<u32,256>;var<workgroup> reduceCorrected:array<u32,256>;var<workgroup> reduceInterfaces:array<u32,256>;var<workgroup> reduceDirectoryRows:array<u32,256>;var<workgroup> coarseFinalizeEnabled:u32;
fn finalizePowerCoarsePhi(lid:u32){if(lid==0u){coarseFinalizeEnabled=select(0u,1u,!rejectedFine());}let enabled=workgroupUniformLoad(&coarseFinalizeEnabled);var flags=0u;var first=INVALID;var advected=0u;var uniform=0u;var transition=0u;var corrected=0u;var interfaces=0u;var directoryRows=0u;let count=select(0u,requested(),enabled!=0u);for(var row=lid;row<count;row+=256u){if(row>=arrayLength(&rowStatus)||row>=arrayLength(&candidateDirectory.entries)){flags|=CAPACITY;first=min(first,row);continue;}let status=rowStatus[row];flags|=status.flags;if(status.flags!=0u){first=min(first,row);}advected+=status.advected;uniform+=status.uniform;transition+=status.transition;corrected+=status.corrected;interfaces+=status.interfaceRow;let entry=candidateDirectory.entries[row];let entryValid=entry.cellPlusOne!=0u&&entry.size!=0u&&(entry.flags&(PHI_VALID|PHI_FINITE))==(PHI_VALID|PHI_FINITE);directoryRows+=select(0u,1u,entryValid);if(!entryValid&&status.flags==0u){flags|=INVALID_ROW;first=min(first,row);}}reduceFlags[lid]=flags;reduceFirst[lid]=first;reduceAdvected[lid]=advected;reduceUniform[lid]=uniform;reduceTransition[lid]=transition;reduceCorrected[lid]=corrected;reduceInterfaces[lid]=interfaces;reduceDirectoryRows[lid]=directoryRows;workgroupBarrier();for(var stride=128u;stride>0u;stride>>=1u){if(lid<stride){reduceFlags[lid]|=reduceFlags[lid+stride];reduceFirst[lid]=min(reduceFirst[lid],reduceFirst[lid+stride]);reduceAdvected[lid]+=reduceAdvected[lid+stride];reduceUniform[lid]+=reduceUniform[lid+stride];reduceTransition[lid]+=reduceTransition[lid+stride];reduceCorrected[lid]+=reduceCorrected[lid+stride];reduceInterfaces[lid]+=reduceInterfaces[lid+stride];reduceDirectoryRows[lid]+=reduceDirectoryRows[lid+stride];}workgroupBarrier();}if(lid==0u&&enabled!=0u){let complete=count>0u&&count<=params.dimensionsCapacity.w&&reduceAdvected[0]==count&&reduceDirectoryRows[0]==count;control.flags|=reduceFlags[0];control.firstError=reduceFirst[0];control.advected=reduceAdvected[0];control.uniform=reduceUniform[0];control.transition=reduceTransition[0];control.corrected=reduceCorrected[0];control.interfaces=reduceInterfaces[0];if(control.flags==0u&&complete){control.valid=VALID;candidateDirectory.state=VALID;}else{control.valid=0u;candidateDirectory.state=0u;}}}
var<workgroup> deltaPrefix:array<u32,256>;
var<workgroup> coarseDeltaCommitState:array<u32,3>;
fn deltaCandidate(index:u32,currentCount:u32,previousCount:u32)->vec4u{if(index<currentCount){let entry=candidateDirectory.entries[index];let oldSlot=previousSampleSlot(entry.cellPlusOne-1u,entry.size);var changed=oldSlot==INVALID;var flags=1u;if(oldSlot!=INVALID){let old=sampleDirectory.entries[oldSlot];let valueChanged=bitcast<u32>(old.phi)!=bitcast<u32>(entry.phi)||bitcast<u32>(old.minimumPhi)!=bitcast<u32>(entry.minimumPhi)||bitcast<u32>(old.maximumPhi)!=bitcast<u32>(entry.maximumPhi);let phaseChanged=((old.flags^entry.flags)&(PHI_INTERFACE|PHI_CORRECTED))!=0u;changed=valueChanged||phaseChanged;flags|=select(0u,4u,valueChanged)|select(0u,8u,phaseChanged);}return vec4u(entry.cellPlusOne,entry.size,entry.row,select(0u,flags,changed));}let old=sampleDirectory.entries[index-currentCount];let retired=candidateSampleSlot(old.cellPlusOne-1u,old.size)==INVALID;return vec4u(old.cellPlusOne,old.size,INVALID,select(0u,2u,retired));}
fn publishPowerCoarsePhiDeltaAndCommit(lid:u32){if(lid==0u){let rejected=rejectedFine();let enabled=!rejected&&control.valid==VALID&&candidateDirectory.state==VALID;if(!rejected&&!enabled){sampleDirectory.state=0u;}coarseDeltaCommitState[0]=select(0u,1u,enabled);coarseDeltaCommitState[1]=select(0u,candidateDirectory.rowCount,enabled);let previous=select(0u,min(sampleDirectory.rowCount,arrayLength(&sampleDirectory.entries)),sampleDirectory.state==VALID);coarseDeltaCommitState[2]=select(0u,previous,enabled);}let enabled=workgroupUniformLoad(&coarseDeltaCommitState[0]);let currentCount=workgroupUniformLoad(&coarseDeltaCommitState[1]);let previousCount=workgroupUniformLoad(&coarseDeltaCommitState[2]);let count=currentCount+previousCount;let width=count/256u;let remainder=count%256u;let begin=lid*width+min(lid,remainder);let end=begin+width+select(0u,1u,lid<remainder);var local=0u;for(var index=begin;index<end;index+=1u){local+=select(0u,1u,deltaCandidate(index,currentCount,previousCount).w!=0u);}deltaPrefix[lid]=local;workgroupBarrier();for(var stride=1u;stride<256u;stride<<=1u){var add=0u;if(lid>=stride){add=deltaPrefix[lid-stride];}workgroupBarrier();deltaPrefix[lid]+=add;workgroupBarrier();}var cursor=deltaPrefix[lid]-local;for(var index=begin;index<end;index+=1u){let item=deltaCandidate(index,currentCount,previousCount);if(item.w!=0u&&cursor<arrayLength(&coarseDeltaPublication.records)){coarseDeltaPublication.records[cursor]=DeltaRecord(item.x,item.y,item.z,item.w);cursor+=1u;}}let changed=deltaPrefix[255u];let overflow=changed>arrayLength(&coarseDeltaPublication.records);if(lid==0u&&enabled!=0u){sampleDirectory.state=0u;}workgroupBarrier();if(enabled!=0u){for(var slot=lid;slot<currentCount;slot+=256u){sampleDirectory.entries[slot]=candidateDirectory.entries[slot];}}workgroupBarrier();if(lid==0u&&enabled!=0u){sampleDirectory.generation=candidateDirectory.generation;sampleDirectory.rowCount=currentCount;sampleDirectory.maximumLeafSize=candidateDirectory.maximumLeafSize;sampleDirectory.dimensions=candidateDirectory.dimensions;sampleDirectory.physicalCellSize=candidateDirectory.physicalCellSize;sampleDirectory.state=VALID;coarseDeltaPublication.count=min(changed,arrayLength(&coarseDeltaPublication.records));coarseDeltaPublication.generation=candidateDirectory.generation;coarseDeltaPublication.flags=select(0u,CAPACITY,overflow);coarseDeltaPublication.valid=select(0u,VALID,!overflow);}}
var<workgroup> coarseScheduleEnabled:u32;
fn scheduleEnabled(lid:u32)->u32{if(lid==0u){coarseScheduleEnabled=select(0u,1u,control.pad0==1u);}
 return workgroupUniformLoad(&coarseScheduleEnabled);}
@compute @workgroup_size(256) fn preparePowerCoarsePhiSchedule(@builtin(local_invocation_index)lid:u32){
 if(lid==0u){let accepted=!rejectedFine();coarseScheduleEnabled=select(0u,1u,accepted);
  dispatchMetadata[0]=0u;dispatchMetadata[1]=1u;dispatchMetadata[2]=1u;
  dispatchMetadata[3]=0u;dispatchMetadata[4]=1u;dispatchMetadata[5]=1u;}
 let enabled=workgroupUniformLoad(&coarseScheduleEnabled);
 if(enabled==0u){if(lid==0u){control.pad0=0u;}return;}
 for(var row=lid;row<sourceRequested();row+=256u){migratePowerCoarsePhiSource(row);}
 storageBarrier();workgroupBarrier();
 if(lid==0u){let priorValid=control.pad1;let priorRows=control.pad2;preparePowerCoarsePhi();control.pad0=1u;let rebuild=priorValid!=VALID||priorRows!=sourceRequested()||params.pad0!=0u;control.pad1=VALID;control.pad2=sourceRequested();control.pad3=select(0u,1u,rebuild);let groups=(sourceRequested()+63u)/64u;dispatchMetadata[0]=groups;dispatchMetadata[3]=select(0u,groups,rebuild);}
 storageBarrier();workgroupBarrier();
 for(var row=lid;row<arrayLength(&rowStatus);row+=256u){clearPowerCoarsePhiRowStatus(row);}
}
@compute @workgroup_size(64) fn buildPowerCoarseSelectorRows(@builtin(global_invocation_id)gid:vec3u){
 buildSelectorRow(gid.x);
}
@compute @workgroup_size(64) fn advectPowerCoarsePhiSchedule(@builtin(global_invocation_id)gid:vec3u,@builtin(local_invocation_index)lid:u32){
 if(scheduleEnabled(lid)==0u){return;}
 advectPowerCoarsePhi(gid.x);
}
@compute @workgroup_size(64) fn correctPowerCoarsePhiSchedule(@builtin(global_invocation_id)gid:vec3u,@builtin(local_invocation_index)lid:u32){
 if(scheduleEnabled(lid)==0u){return;}
 validatePowerCoarseFineCorrection(gid.x);applyExactFineCorrection(gid.x);
}
@compute @workgroup_size(64) fn redistancePowerCoarsePhiAtoB(@builtin(global_invocation_id)gid:vec3u,@builtin(local_invocation_index)lid:u32){
 if(scheduleEnabled(lid)==0u){return;}
 redistancePowerCoarsePhi(gid.x,true);
}
@compute @workgroup_size(64) fn redistancePowerCoarsePhiBtoA(@builtin(global_invocation_id)gid:vec3u,@builtin(local_invocation_index)lid:u32){
 if(scheduleEnabled(lid)==0u){return;}
 redistancePowerCoarsePhi(gid.x,false);
}
@compute @workgroup_size(64) fn publishPowerCoarsePhiSchedule(@builtin(global_invocation_id)gid:vec3u,@builtin(local_invocation_index)lid:u32){
 if(scheduleEnabled(lid)==0u){return;}
 publishPowerCoarsePhi(gid.x);
}
@compute @workgroup_size(256) fn commitPowerCoarsePhiSchedule(@builtin(local_invocation_index)lid:u32){
 if(scheduleEnabled(lid)==0u){return;}
 finalizePowerCoarsePhi(lid);
 storageBarrier();workgroupBarrier();
 publishPowerCoarsePhiDeltaAndCommit(lid);
}
`;
