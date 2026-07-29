/**
 * Single-dispatch persistent MGPCG executor for small pressure systems.
 *
 * The hierarchical path (`WebGPUOctreePipelinedMGPCG` +
 * `WebGPUOctreeSection43HybridPreconditioner` + `WebGPUOctreeSPGridVCycle`)
 * encodes 862-890 dispatches across ~91 compute passes to solve a system with
 * ~1,470 unknowns. This module runs the identical iteration — k=8 boundary
 * band sweeps, the first-order five-level V-cycle, k=8 matched sweeps, the
 * Section 6.3 A2 applies, the CG vector updates and the two compensated dot
 * products per iteration — inside ONE dispatch of ONE 256-lane workgroup,
 * with `storageBarrier(); workgroupBarrier();` where dispatch boundaries used
 * to be and `workgroupUniformLoad` on every barrier-carrying loop bound.
 *
 * Nothing here changes the algebra, the tolerances, or the fail-closed
 * semantics. The kernel writes the same MGPCG control words to the same
 * buffer, so `StructuredStepSnapshotRing` and the Part-A tripwires are
 * unaffected.
 *
 * Selection is by constructed row capacity against
 * {@link OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD};
 * the hierarchical path stays fully supported and is the big-domain path.
 */
import { PassBroker } from "./webgpu-pass-broker";
import {
  createGPULogicalActivityAdoptionContext,
} from "./gpu-logical-activity-adoption";
import { performanceShaderVariant } from "./stores/performance-instrumentation-store";
import {
  OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY,
  type OctreePersistentMGPCGExecutor,
} from "./webgpu-octree-section43-contract";
import {
  planOctreeSPGridVCycle,
  type OctreeSPGridVCyclePlan,
} from "./webgpu-octree-spgrid-vcycle";
import {
  OCTREE_PERSISTENT_MGPCG_CHANNEL,
  OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT,
  OCTREE_PERSISTENT_MGPCG_HEADER,
  OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES,
  OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES,
  octreePersistentMGPCGArenaWords,
  octreePersistentMGPCGWGSL,
} from "./webgpu-octree-persistent-mgpcg.wgsl";

export {
  OCTREE_PERSISTENT_MGPCG_CHANNEL,
  OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT,
  OCTREE_PERSISTENT_MGPCG_HEADER,
  OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES,
  octreePersistentMGPCGArenaWords,
  octreePersistentMGPCGWGSL,
};

/**
 * Authored row-count ceiling for the persistent executor.
 *
 * Single source of truth is the Section 4.3 contract constant, so the
 * selection gate and the contract can never drift. 4,096 retains the measured
 * mini-lane win; larger constructed systems use the hierarchical path because
 * a single workgroup is one GPU core and loses once row/page-parallel work can
 * saturate the device.
 */
export const OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD =
  OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY;

/** One 256-lane workgroup, one dispatch. */
export const OCTREE_PERSISTENT_MGPCG_LANES = 256;

/** Marker written to solve-control word 21 so a readback can identify the executor. */
export const OCTREE_PERSISTENT_MGPCG_CONTROL_MARKER = 0x5045_5253;

/** Extra solve-control word the persistent executor authors. */
export const OCTREE_PERSISTENT_MGPCG_CONTROL = Object.freeze({
  executorMarker: 21,
} as const);

/**
 * Keep the old capacity-strided arena available as a process-local timing and
 * correctness oracle. Production packs every hot vector to the accepted live
 * row prefix; the external staging ABI remains capacity-strided.
 */
export function octreePersistentMGPCGCompactLiveRowsEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_PERSISTENT_MGPCG_COMPACT_ROWS !== "0";
}

export const OCTREE_PERSISTENT_MGPCG_ACTIVITY_MODULE_ID = "octree/persistent-mgpcg";
const OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK = "whole-solve";

const STORAGE_BINDINGS = Object.freeze([1, 2, 3, 4, 5, 6, 7, 8] as const);
/** Explicit, asserted: eight storage bindings plus one uniform. */
export const OCTREE_PERSISTENT_MGPCG_STORAGE_BINDING_COUNT = STORAGE_BINDINGS.length;

const PARAMS_LEVEL_TABLE_SLOTS = 16;
const PARAMS_BYTES = 80 + 5 * PARAMS_LEVEL_TABLE_SLOTS * 4;
const DISPATCH_RECORD_WORDS_PER_LEVEL = 12;
const DISPATCH_LIFECYCLE_WORDS = 2;
const ACCEPTED_CONTROL_BYTES = 24;
const WORKSET_ROW_CLASS_COUNT = 4;
const WORKSET_HEADER_STAGE_BYTES = 8;

/**
 * Everything the persistent kernel binds or stages. Buffers are supplied by
 * the existing owners; this module allocates only its row arena and uniform.
 */
export interface OctreePersistentMGPCGSource {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
  /** SPGrid V-cycle 26-channel level arena (`section63Topology.state`). */
  readonly state: GPUBuffer;
  /** SPGrid sparse topology / worklists / transfers (`section63Topology.topology`). */
  readonly topology: GPUBuffer;
  /** SPGrid captured fixed row geometry (`section63Topology.geometry`). */
  readonly geometry: GPUBuffer;
  /** SPGrid per-level worklist counts and published dispatch words. */
  readonly dispatchMeta: GPUBuffer;
  /** Banked Section 6.3 diagonal + eighteen canonical coefficients. */
  readonly coefficients: GPUBuffer;
  readonly coefficientBankStrideWords: number;
  /** Structured power topology metrics, one `Metric` per row. */
  readonly metrics: GPUBuffer;
  /** Accepted structured control `[flags, firstError, rows, generation, bank, slots]`. */
  readonly acceptedAuthority: GPUBuffer;
  /** Packed A/B structured row and family worksets. */
  readonly worksets: GPUBuffer;
  readonly worksetStrideWords: number;
  readonly worksetBankStrideWords: number;
  /** Current divergence/RHS authority, one f32 per compact row. */
  readonly rhs: GPUBuffer;
  /** Shared MGPCG solve control; the persistent kernel authors the same words. */
  readonly control: GPUBuffer;
}

export interface OctreePersistentMGPCGOptions {
  readonly maximumIterations: number;
  readonly boundarySmoothingIterations: number;
  /** SPGrid smoother contract degree, 2 or 4. */
  readonly chebyshevDegree: 2 | 4;
  readonly boundaryBandLayers: number;
  readonly relativeTolerance: number;
  readonly absoluteTolerance?: number;
  readonly damping?: number;
  readonly maximumLevels?: number;
}

export interface OctreePersistentMGPCGSolve {
  readonly pressureSeed: GPUBuffer;
  readonly pressureOut: GPUBuffer;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value;
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

/**
 * Sixteen-slot level tables, byte-identical to the ones
 * `WebGPUOctreeSPGridVCycle` writes into its per-level uniform. The padding
 * entries above `levelCount` repeat the same closed form so an out-of-range
 * probe cannot read stale storage.
 */
function buildLevelTables(
  dimensions: readonly [number, number, number],
  plan: OctreeSPGridVCyclePlan,
): Readonly<{
  levelCaps: number[]; levelBases: number[]; brickOffsets: number[];
  pageOffsets: number[]; transferOffsets: number[];
}> {
  const maximumSparseSlots = nextPowerOfTwo(plan.rowCapacity * 16);
  const levelCapacityAt = (level: number) => {
    if (level < plan.levelCount) return plan.levelCapacities[level];
    const scale = 2 ** level;
    const domainCells = dimensions
      .map((value) => Math.ceil(value / scale))
      .reduce((product, value) => product * value, 1);
    return nextPowerOfTwo(Math.min(maximumSparseSlots, 2 * domainCells));
  };
  const levelCaps: number[] = [], levelBases: number[] = [], brickOffsets: number[] = [];
  const pageOffsets: number[] = [], transferOffsets: number[] = [];
  let slotBase = 0, brickBase = 0, pageBase = 0, transferBase = 0;
  for (let level = 0; level < PARAMS_LEVEL_TABLE_SLOTS; level += 1) {
    const capacity = levelCapacityAt(level);
    const levelDimensions = dimensions.map((value) => Math.ceil(value / 2 ** level));
    levelCaps.push(capacity);
    levelBases.push(slotBase); slotBase += capacity;
    brickOffsets.push(brickBase);
    brickBase += levelDimensions.map((value) => Math.ceil(value / 4))
      .reduce((product, value) => product * value, 1);
    pageOffsets.push(pageBase);
    pageBase += Math.ceil(levelDimensions[0] / 8) * Math.ceil(levelDimensions[1] / 8)
      * Math.ceil(levelDimensions[2] / 4);
    transferOffsets.push(transferBase);
    transferBase += Math.min(plan.transferStride, capacity * 8) * 4 + 4 * capacity;
  }
  if (levelBases[plan.levelCount] !== plan.totalLevelSlots
    || brickOffsets[plan.levelCount] !== plan.brickCount
    || pageOffsets[plan.levelCount] !== plan.pageDirectoryBytes / 4) {
    throw new RangeError("Persistent MGPCG level tables disagree with the SPGrid plan");
  }
  return { levelCaps, levelBases, brickOffsets, pageOffsets, transferOffsets };
}

/**
 * The persistent executor. It owns exactly two allocations: the consolidated
 * row arena (all row-shaped vectors, the staged authority tables and the
 * reduction partials) and its uniform parameter block.
 */
export class WebGPUOctreePersistentMGPCG implements OctreePersistentMGPCGExecutor {
  readonly plan: OctreeSPGridVCyclePlan;
  /** Contract surface declared by `OctreePersistentMGPCGExecutor`. */
  readonly maximumRowCapacity: number;
  readonly encodedDispatchCount = 1 as const;
  readonly dispatchShape = [1, 1, 1] as const;
  readonly invariantProof = Object.freeze({
    ghostRows: "spgrid-identical" as const,
    transfers: "validated-adjoint-pair" as const,
    invalidRows: "uniform-fail-closed-before-arithmetic" as const,
  });
  readonly encodedPassTransitionCount = 1;
  readonly allocatedBytes: number;
  readonly arenaWords: number;
  readonly storageBindingCount = OCTREE_PERSISTENT_MGPCG_STORAGE_BINDING_COUNT;
  readonly workgroupStorageBytes = OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES;

  private readonly params: GPUBuffer;
  private readonly arena: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly groups: Array<{ pressureOut: GPUBuffer; group: GPUBindGroup }> = [];
  private readonly dispatchMetaBytes: number;
  private destroyed = false;

  /** GPU-authored records suitable for the existing diagnostics readback. */
  get workAccountingBuffers(): Readonly<{ control: GPUBuffer; arena: GPUBuffer }> {
    return Object.freeze({ control: this.source.control, arena: this.arena });
  }

  get workAccountingPlan(): Readonly<{
    encodedDispatchCount: 1; lanes: number; rowThreshold: number;
    storageBindings: number; workgroupStorageBytes: number;
  }> {
    return Object.freeze({
      encodedDispatchCount: 1 as const,
      lanes: OCTREE_PERSISTENT_MGPCG_LANES,
      rowThreshold: this.maximumRowCapacity,
      storageBindings: this.storageBindingCount,
      workgroupStorageBytes: this.workgroupStorageBytes,
    });
  }

  constructor(
    private readonly device: GPUDevice,
    private readonly source: OctreePersistentMGPCGSource,
    private readonly options: OctreePersistentMGPCGOptions,
  ) {
    const rowCapacity = positiveInteger(source.rowCapacity, "Persistent MGPCG row capacity");
    if (rowCapacity > OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD) {
      throw new RangeError(
        "Persistent MGPCG row capacity exceeds the authored single-workgroup threshold",
      );
    }
    this.maximumRowCapacity = OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD;
    this.plan = planOctreeSPGridVCycle({
      dimensions: source.dimensions,
      rowCapacity,
      ...(options.maximumLevels === undefined ? {} : { maximumLevels: options.maximumLevels }),
    });
    if (options.maximumIterations < 4 || options.maximumIterations > 10
      || !Number.isSafeInteger(options.maximumIterations)) {
      throw new RangeError("Persistent MGPCG encoded iterations must remain in [4,10]");
    }
    if (options.chebyshevDegree !== 2 && options.chebyshevDegree !== 4) {
      throw new RangeError("Persistent MGPCG requires the exact SPGrid Chebyshev degree");
    }
    if ((options.boundarySmoothingIterations & 1) !== 0
      || options.boundarySmoothingIterations < 2) {
      throw new RangeError("Persistent MGPCG boundary smoothing must be even and at least two");
    }
    // The kernel transcribes encodeSetup's fixed classify + AtoB/BtoA/AtoB
    // dilation chain. A different layer count would need a different chain.
    if (options.boundaryBandLayers !== 3) {
      throw new RangeError("Persistent MGPCG transcribes the three-layer Section 4.3 shell only");
    }
    const damping = options.damping ?? 2 / 3;
    if (!Number.isFinite(damping) || damping <= 0 || damping >= 1) {
      throw new RangeError("Persistent MGPCG damping must be finite and in (0,1)");
    }
    const absoluteTolerance = options.absoluteTolerance ?? 0;
    if (!(options.relativeTolerance >= 0) || !(absoluteTolerance >= 0)
      || (options.relativeTolerance === 0 && absoluteTolerance === 0)) {
      throw new RangeError("Persistent MGPCG requires a non-zero stopping tolerance");
    }

    // Fail closed on capability before any pipeline exists. The 10-storage
    // ceiling and the workgroup-storage ceiling are both validation errors
    // that `skip_validation` turns into a SIGSEGV rather than a message.
    const limits = device.limits;
    if (limits) {
      if (OCTREE_PERSISTENT_MGPCG_LANES > limits.maxComputeInvocationsPerWorkgroup) {
        throw new RangeError("Persistent MGPCG 256-lane workgroup exceeds device limits");
      }
      if (this.workgroupStorageBytes > limits.maxComputeWorkgroupStorageSize) {
        throw new RangeError(`Persistent MGPCG requires ${this.workgroupStorageBytes} bytes of `
          + `workgroup storage; device exposes ${limits.maxComputeWorkgroupStorageSize}`);
      }
      if (this.storageBindingCount > limits.maxStorageBuffersPerShaderStage) {
        throw new RangeError("Persistent MGPCG storage bindings exceed device limits");
      }
    }

    this.dispatchMetaBytes =
      (this.plan.levelCount * DISPATCH_RECORD_WORDS_PER_LEVEL + DISPATCH_LIFECYCLE_WORDS) * 4;
    if (this.dispatchMetaBytes > OCTREE_PERSISTENT_MGPCG_HEADER.dispatchWords * 4) {
      throw new RangeError("Persistent MGPCG dispatch staging window is too small");
    }
    const copySource = GPUBufferUsage.COPY_SRC;
    for (const [label, buffer] of [
      ["accepted authority", source.acceptedAuthority],
      ["dispatch metadata", source.dispatchMeta],
      ["structured worksets", source.worksets],
      ["divergence RHS", source.rhs],
    ] as const) {
      if ((buffer.usage & copySource) === 0) {
        throw new RangeError(`Persistent MGPCG requires COPY_SRC on the ${label} buffer`);
      }
    }
    if (source.acceptedAuthority.size < ACCEPTED_CONTROL_BYTES
      || source.dispatchMeta.size < this.dispatchMetaBytes
      || source.rhs.size < rowCapacity * 4
      || source.coefficients.size < 2 * rowCapacity * 19 * 4
      || source.geometry.size < rowCapacity * 16
      || source.coefficientBankStrideWords < rowCapacity * 19) {
      throw new RangeError("Persistent MGPCG source capacity is too small");
    }
    const worksetSpan =
      (source.worksetBankStrideWords + WORKSET_ROW_CLASS_COUNT * source.worksetStrideWords) * 4;
    if (source.worksets.size < worksetSpan) {
      throw new RangeError("Persistent MGPCG workset staging exceeds the published banks");
    }

    this.arenaWords = octreePersistentMGPCGArenaWords(rowCapacity);
    this.arena = device.createBuffer({
      label: "Persistent MGPCG consolidated row arena",
      size: this.arenaWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.params = device.createBuffer({
      label: "Persistent MGPCG parameters",
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const tables = buildLevelTables(source.dimensions, this.plan);
    const words = new Uint32Array(PARAMS_BYTES / 4);
    const floats = new Float32Array(words.buffer);
    words.set([
      source.dimensions[0], source.dimensions[1], source.dimensions[2], rowCapacity,
      this.plan.levelCount, this.plan.levelStride, this.plan.totalLevelSlots,
      source.coefficientBankStrideWords,
      options.maximumIterations, options.chebyshevDegree,
      options.boundarySmoothingIterations, options.boundaryBandLayers,
      this.plan.transferStride, this.plan.brickCount, this.plan.pageDirectoryBytes / 4,
      Math.ceil(rowCapacity / OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES),
    ]);
    floats[16] = options.relativeTolerance;
    floats[17] = absoluteTolerance;
    floats[18] = 1e-30;
    floats[19] = damping;
    words.set(tables.levelCaps, 20);
    words.set(tables.levelBases, 20 + PARAMS_LEVEL_TABLE_SLOTS);
    words.set(tables.brickOffsets, 20 + 2 * PARAMS_LEVEL_TABLE_SLOTS);
    words.set(tables.pageOffsets, 20 + 3 * PARAMS_LEVEL_TABLE_SLOTS);
    words.set(tables.transferOffsets, 20 + 4 * PARAMS_LEVEL_TABLE_SLOTS);
    device.queue.writeBuffer(this.params, 0, words);

    const activityProfile = performanceShaderVariant();
    const activity = createGPULogicalActivityAdoptionContext({
      moduleId: OCTREE_PERSISTENT_MGPCG_ACTIVITY_MODULE_ID,
      profile: activityProfile,
      identity: "subgroup",
    });
    activity.describeTask(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, {
      id: "gpu.physics.persistent-mgpcg.whole-solve",
      label: "Persistent MGPCG · whole solve",
      phaseId: "pressure-solve",
      checkpoints: {
        enter: activity.checkpointId(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, "enter"),
        exit: activity.checkpointId(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, "exit"),
      },
    });
    const activitySite = {
      tick: "atomicLoad(&control[2])",
      workgroupId: "activityWorkgroupId",
      numWorkgroups: "activityNumWorkgroups",
      subgroupId: "(lane / activitySubgroupSize)",
      subgroupIdEvidence: "reconstructed" as const,
      subgroupLane: "activitySubgroupLane",
      subgroupSize: "activitySubgroupSize",
      active: "lane < rows() && !failed()",
    };
    const activityParameters = [
      "@builtin(workgroup_id) activityWorkgroupId:vec3u",
      "@builtin(num_workgroups) activityNumWorkgroups:vec3u",
      "@builtin(subgroup_invocation_id) activitySubgroupLane:u32",
      "@builtin(subgroup_size) activitySubgroupSize:u32",
    ].join(",\n ");
    const compactLiveRows = octreePersistentMGPCGCompactLiveRowsEnabled();
    const shaderSource = octreePersistentMGPCGWGSL({
      maximumIterations: options.maximumIterations,
      compactLiveRows,
      ...(activity.enabled ? { activity: {
        parameters: activityParameters,
        enter: activity.subgroup(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, "enter", activitySite),
        // Scalar exit markers preserve interval accounting without adding a
        // second ballot to the sampled-lane utilization denominator.
        exit: activity.workgroup(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, "exit", {
          tick: "atomicLoad(&control[2])",
          workgroupId: "activityWorkgroupId",
          numWorkgroups: "activityNumWorkgroups",
          localInvocationIndex: "lane",
          workgroupLaneCount: OCTREE_PERSISTENT_MGPCG_LANES,
        }),
      } } : {}),
    });
    const shaderVariant = activity.module(
      shaderSource,
      `${OCTREE_PERSISTENT_MGPCG_ACTIVITY_MODULE_ID}/${activityProfile.cacheKey}`
        + `/${compactLiveRows ? "compact-live" : "capacity-strided"}`,
    );
    const shaderModule = device.createShaderModule({
      label: `Octree persistent MGPCG · single-dispatch 256-lane solve · ${
        compactLiveRows ? "compact live rows" : "capacity-strided oracle"}`,
      code: shaderVariant.code,
    });
    this.pipeline = activity.registerPipeline(device.createComputePipeline({
      label: "Persistent MGPCG · persistentMGPCG",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "persistentMGPCG" },
    }));
    this.allocatedBytes = this.arena.size + this.params.size;
  }

  /** Byte offset of a row channel inside the arena. */
  channelByteOffset(channel: number): number {
    return (OCTREE_PERSISTENT_MGPCG_HEADER.totalWords
      + channel * this.source.rowCapacity) * 4;
  }

  /**
   * One clear, a fixed set of tiny staging copies, and one dispatch.
   *
   * ORDERING CONTRACT. Every staged source must already carry this step's
   * final value when this runs, which is true at the current call site: the
   * structured publication, the SPGrid capture/commit and the divergence
   * assembly all complete inside `encodeNativePowerAssembly`, which precedes
   * the solve broker. If a caller ever needs the SPGrid setup to run first,
   * it must pass `encodeInnerSetup`; that callback is invoked before the
   * `dispatchMeta` copy.
   *
   * The control clear precedes `encodeInnerSetup` for the same reason it is
   * the first statement of `WebGPUOctreePipelinedMGPCG.encode`: the SPGrid
   * commit stages (`finalizeLifecycle`, `publishCommittedInputs`) gate on
   * `control[0] == 0`, so a setup encoded against last step's error flags
   * would refuse to commit the hierarchy and keep refusing forever. The
   * callback still runs before every staging copy, and any word it authors
   * survives into the dispatch.
   */
  encodeSolve(
    broker: PassBroker,
    solve: OctreePersistentMGPCGSolve,
    encodeInnerSetup?: (broker: PassBroker) => void,
  ): void {
    this.assertLive();
    const rowBytes = this.source.rowCapacity * 4;
    if (solve.pressureSeed.size < rowBytes || solve.pressureOut.size < rowBytes) {
      throw new RangeError("Persistent MGPCG solve buffers are smaller than the row capacity");
    }
    if ((solve.pressureSeed.usage & GPUBufferUsage.COPY_SRC) === 0) {
      throw new RangeError("Persistent MGPCG requires COPY_SRC on the pressure seed");
    }
    broker.clearBuffer(this.source.control);
    encodeInnerSetup?.(broker);
    broker.copyBufferToBuffer(this.source.acceptedAuthority, 0, this.arena,
      OCTREE_PERSISTENT_MGPCG_HEADER.accepted * 4, ACCEPTED_CONTROL_BYTES);
    broker.copyBufferToBuffer(this.source.dispatchMeta, 0, this.arena,
      OCTREE_PERSISTENT_MGPCG_HEADER.dispatch * 4, this.dispatchMetaBytes);
    // Both banks' four accepted row-class headers. The kernel selects the
    // accepted bank GPU-side and fails closed unless every header carries the
    // accepted generation and the four counts sum to rows() — the proof that
    // iterating [0, rows()) equals the hierarchical four-class apply.
    for (let bank = 0; bank < 2; bank += 1) {
      for (let rowClass = 0; rowClass < WORKSET_ROW_CLASS_COUNT; rowClass += 1) {
        broker.copyBufferToBuffer(
          this.source.worksets,
          (bank * this.source.worksetBankStrideWords
            + rowClass * this.source.worksetStrideWords) * 4,
          this.arena,
          (OCTREE_PERSISTENT_MGPCG_HEADER.worksetHeaders
            + (bank * WORKSET_ROW_CLASS_COUNT + rowClass) * 2) * 4,
          WORKSET_HEADER_STAGE_BYTES,
        );
      }
    }
    broker.copyBufferToBuffer(this.source.rhs, 0, this.arena,
      this.channelByteOffset(OCTREE_PERSISTENT_MGPCG_CHANNEL.rhs), rowBytes);
    broker.copyBufferToBuffer(solve.pressureSeed, 0, this.arena,
      this.channelByteOffset(OCTREE_PERSISTENT_MGPCG_CHANNEL.pressureSeed), rowBytes);

    const pass = broker.compute({
      label: "Octree persistent MGPCG · whole solve in one workgroup",
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup(solve.pressureOut));
    pass.dispatchWorkgroups(...this.dispatchShape);
    broker.fence("persistent MGPCG pressure publication");
  }

  private bindGroup(pressureOut: GPUBuffer): GPUBindGroup {
    const cached = this.groups.find((candidate) => candidate.pressureOut === pressureOut);
    if (cached) return cached.group;
    const resources: Readonly<Record<number, GPUBuffer>> = {
      0: this.params,
      1: this.arena,
      2: this.source.state,
      3: this.source.topology,
      4: this.source.coefficients,
      5: this.source.geometry,
      6: this.source.metrics,
      7: this.source.control,
      8: pressureOut,
    };
    const group = this.device.createBindGroup({
      label: "Persistent MGPCG bindings",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [0, ...STORAGE_BINDINGS].map((binding) => ({
        binding, resource: { buffer: resources[binding] },
      })),
    });
    this.groups.push({ pressureOut, group });
    return group;
  }

  /** Row-count gate the encode site consults; see the threshold's docs. */
  static selects(rowCapacity: number): boolean {
    return Number.isSafeInteger(rowCapacity) && rowCapacity > 0
      && rowCapacity <= OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD;
  }

  get iterationBudget(): number { return this.options.maximumIterations; }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Persistent MGPCG executor is destroyed");
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.groups.length = 0;
    this.arena.destroy();
    this.params.destroy();
  }
}
