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
 * The GPU-published live row count is bounded by
 * {@link OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD}; provisioned capacity may be
 * larger because adaptive topology is not known at construction time.
 */
import { PassBroker } from "./webgpu-pass-broker";
import {
  createGPULogicalActivityAdoptionContext,
  type GPULogicalActivityAdoptionContext,
} from "./gpu-logical-activity-adoption";
import { performanceShaderVariant } from "./stores/performance-instrumentation-store";
import { gpuCompilationManagerFor } from "./gpu-compilation-manager";
import {
  OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY,
  type OctreePersistentMGPCGExecutor,
} from "./webgpu-octree-section43-contract";
import {
  planOctreeSPGridVCycle,
  type OctreeSPGridVCyclePlan,
} from "./webgpu-octree-spgrid-vcycle";
import {
  OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER,
  OCTREE_PERSISTENT_MGPCG_CHANNEL,
  OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT,
  OCTREE_PERSISTENT_MGPCG_HEADER,
  OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS,
  OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES,
  OCTREE_PERSISTENT_MGPCG_STAGED_SMOOTHER_WORKGROUP_BYTES,
  OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES,
  octreePersistentMGPCGArenaWords,
  octreePersistentMGPCGWGSL,
  octreePersistentMGPCGWorkgroupBytes,
} from "./webgpu-octree-persistent-mgpcg.wgsl";

export {
  OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER,
  OCTREE_PERSISTENT_MGPCG_CHANNEL,
  OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT,
  OCTREE_PERSISTENT_MGPCG_HEADER,
  OCTREE_PERSISTENT_MGPCG_STAGED_SMOOTHER_WORKGROUP_BYTES,
  OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES,
  octreePersistentMGPCGArenaWords,
  octreePersistentMGPCGWGSL,
  octreePersistentMGPCGWorkgroupBytes,
};

/**
 * Authored live-row ceiling for the persistent executor.
 *
 * Single source of truth is the persistent-executor contract constant, so the
 * runtime gate and executor contract cannot drift.
 */
export const OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD =
  OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY;
function persistentMGPCGRowThreshold(
  _environment?: Readonly<Record<string, string | undefined>>,
): number {
  return OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD;
}

function persistentMGPCGDiagnosticOutputChannel(): number | undefined {
  if (typeof process === "undefined") return undefined;
  const name = process.env?.FLUID_PERSISTENT_MGPCG_DIAGNOSTIC_OUTPUT;
  if (!name) return undefined;
  const channels = OCTREE_PERSISTENT_MGPCG_CHANNEL as Readonly<Record<string, number>>;
  const channel = channels[name];
  if (channel === undefined) {
    throw new RangeError(`Unknown persistent MGPCG diagnostic output channel ${name}`);
  }
  return channel;
}

/** One 256-lane workgroup, one dispatch. */
export const OCTREE_PERSISTENT_MGPCG_LANES = 256;

/** Marker written to solve-control word 21 so a readback can identify the executor. */
export const OCTREE_PERSISTENT_MGPCG_CONTROL_MARKER = 0x5045_5253;

/** Extra solve-control word the persistent executor authors. */
export const OCTREE_PERSISTENT_MGPCG_CONTROL = Object.freeze({
  executorMarker: 21,
  hybridRegularRows: 22,
  hybridIdentityRows: 23,
  hybridPowerRows: 24,
  hybridLiquidRows: 25,
  hybridPowerMachineryRows: 26,
  hybridFullPageSlotChains: 27,
  hybridPageSlotChains: 28,
  hybridEpoch: 29,
  hybridRowClassCount: 30,
  hybridCensusMarker: 31,
} as const);

export const OCTREE_PERSISTENT_MGPCG_HYBRID_CENSUS_MARKER = 0x4834_4234;

export interface OctreePersistentMGPCGHybridCensus {
  readonly regularRows: number;
  /** Exact dry diag=1/offdiag=0/RHS=0 rows; excluded from the Bet-4 score. */
  readonly identityRows: number;
  readonly powerRows: number;
  readonly liquidRows: number;
  readonly liveRows: number;
  readonly fullDescriptorRows: number;
  readonly hybridDescriptorRows: number;
  readonly fullCatalogRows: number;
  readonly hybridCatalogRows: number;
  readonly fullPageSlotChains: number;
  readonly hybridPageSlotChains: number;
  readonly epoch: number;
  readonly machineryReduction: number;
}

/** Decode only the census written after exact GPU partition validation. */
export function decodeOctreePersistentMGPCGHybridCensus(
  words: Uint32Array,
): OctreePersistentMGPCGHybridCensus | null {
  if (words.length < 10 || words[9] !== OCTREE_PERSISTENT_MGPCG_HYBRID_CENSUS_MARKER) return null;
  const regularRows = words[0]!, identityRows = words[1]!, powerRows = words[2]!;
  const liquidRows = regularRows + powerRows, liveRows = liquidRows + identityRows;
  if (liveRows < 1 || words[3] !== liquidRows || words[4] !== powerRows
    || words[5] !== 18 * liquidRows || words[6] !== 18 * powerRows
    || words[7] === 0 || words[8] !== 5) return null;
  return Object.freeze({
    regularRows, identityRows, powerRows, liquidRows, liveRows,
    fullDescriptorRows: liquidRows, hybridDescriptorRows: powerRows,
    fullCatalogRows: liquidRows, hybridCatalogRows: powerRows,
    fullPageSlotChains: words[5]!, hybridPageSlotChains: words[6]!,
    epoch: words[7]!,
    machineryReduction: powerRows === 0 ? Number.POSITIVE_INFINITY : liquidRows / powerRows,
  });
}

/** Shared solve-control ABI consumed by the persistent kernel and snapshots. */
export const OCTREE_PERSISTENT_MGPCG_CONTROL_BYTES = 128;

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

/**
 * Memory-only staging for the single-workgroup smoother. Default OFF.
 *
 * WHAT IT CHANGES. Nothing arithmetic. It (a) copies the two solve-invariant
 * header regions — the immutable dispatch header `arena[16..272)` and the
 * eight-word accepted-authority prefix of a `read` binding — into workgroup
 * storage once at kernel entry, so `count()`/`pageCount()`/`transferCount()`
 * and `acceptedBank()` stop re-reading storage on every call; and (b) issues
 * `applied()`'s eighteen stencil spokes as three waves of independent loads
 * held in eighteen named scalars, instead of one dependent load chain feeding a
 * dynamically indexed `array<f32,18>`. Same eighteen terms, same order, same
 * `canonical18Sum`, same `reportAt` sites — see the equality argument on
 * `stagedSmootherApplied` in the WGSL module.
 *
 * WHY IT DEFAULTS OFF. The tree's correctness contract for this kernel is
 * bitwise, not approximate: `test:webgpu:symmetric-expansion` pins the factor-1
 * lane D4-symmetric through step 67 and first divergent at step 68. The
 * equality argument above is a proof about the emitted WGSL, not about what
 * Tint and the Metal backend do with a shader whose register pressure and
 * unrolling shape both change. Until that oracle has been re-run against the
 * variant on real hardware, the claim is unwitnessed, and a silent one-ulp move
 * in the smoother would be indistinguishable from a topology regression in
 * every downstream lane. The A/B and the symmetry gate flip this, not review.
 *
 * The 1,056 extra bytes of workgroup storage are asserted against
 * `maxComputeWorkgroupStorageSize` before any pipeline exists, exactly as the
 * baseline footprint is: an overflow here is a validation error the
 * `skip_validation` harness turns into a SIGSEGV rather than a message.
 */
export function octreePersistentMGPCGStagedSmootherEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_MGPCG_STAGED_SMOOTHER === "1";
}

/** Selected mode for {@link octreePersistentMGPCGRegularBandRowsMode}. */
export type OctreePersistentMGPCGRegularBandRowsMode = "off" | "census" | "route";

/**
 * Class-0 routing for the Section 4.3 band sweep. Default OFF.
 *
 * WHAT IT CHANGES. `applyAllRows` already reads the accepted five-class
 * worksets and sends `cls == 0` rows to `applyRegularRow` — six channels,
 * `regularPageSlot`, no `finerAdjoint`. `applyBandRows` cannot, because a row's
 * class exists *only* as membership in one of those five lists and the band
 * list carries bare row indices. This publishes that membership once, in P4b,
 * as a per-row map in CH_BANDA (dead from the third dilation to the end of the
 * dispatch), and then the band sweep takes the same `applyRegularRow` for the
 * same rows. That sweep runs `2 * boundarySweeps - 1` times per Section 4.3
 * correction and is where the kernel's time is.
 *
 * `"census"` publishes the map and the four census words but leaves every band
 * row on `applyRow`. It exists so the class-0 hit rate on band rows, the coarse
 * share of those rows, and the map's own cost are all measurable *before* the
 * arithmetic moves. `"1"` selects the routing and keeps the census.
 *
 * WHAT IT IS NOT. It is NOT bit-identical, and this is Gate B, not Gate A:
 *
 *  - `applyRow` adds `finerAdjoint(...)`, which for a class-0 row returns
 *    `+0.0` (level 0 returns the literal; above it, a class-0 row owns no fine
 *    ghosts because `rowTransition` keeps every level-transition row out of
 *    class 0, so all eight `canonical18Sum` children fold to `+0.0`). Dropping
 *    a `+ (+0.0)` is value-preserving except that it can no longer turn a
 *    `-0.0` row image into `+0.0`. Numerical equivalence up to signed zero is
 *    the strongest claim available, and the argument about which rows own
 *    ghosts is a property of the publisher, not a local invariant.
 *  - `applyRegularRow` adds two authority checks (`caseId == 0`, no physical
 *    boundary bits) that `applyRow` does not make. Both are false by the
 *    definition of class 0, and — decisively — P2's `applyAllRows` already ran
 *    `applyRegularRow` over every class-0 row before any band exists, and every
 *    report either path can raise for a given row is a function of topology and
 *    row geometry only. So the routing adds no reachable report that the solve
 *    has not already made.
 *  - Its address resolver is `regularPageSlot`, not `opPageSlot`. That is not
 *    obviously fewer loads (it trades `decode`'s dependent page-origin probe
 *    for a dependent `S_KEY` probe and two dispatch-header reads); its
 *    measured value here is the twelve skipped edge channels and the removed
 *    `finerAdjoint`, so a lane whose rows are all at level 0 should expect much
 *    less than one whose band is coarse.
 *
 * WHY IT DEFAULTS OFF. The tree's contract for this kernel is bitwise: the
 * factor-1 `symmetric-expansion` lane is D4-symmetric through step 67 and first
 * diverges at 68. A signed-zero flip is invisible to that oracle's enforced
 * fields — every one of them gates on `maximumAbsoluteError`, and
 * `Math.abs(+0 - -0)` is `0`; only `topology`, an integer field, gates on
 * `Object.is` — but "invisible to the oracle" is an argument, and the argument
 * has not been witnessed on hardware. The A/B plus the symmetry gate flip this,
 * not review.
 */
export function octreePersistentMGPCGRegularBandRowsMode(
  environment?: Readonly<Record<string, string | undefined>>,
): OctreePersistentMGPCGRegularBandRowsMode {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  const value = resolved?.FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS;
  if (value === "1") return "route";
  if (value === "census") return "census";
  return "off";
}

/** The four `bandCensus` words, decoded only when this dispatch authored them. */
export interface OctreePersistentMGPCGBandCensus {
  readonly bandRows: number;
  readonly regularBandRows: number;
  /** Regular band rows whose native level is above zero — the only ones whose
   * `finerAdjoint` probe does more than return a literal. */
  readonly coarseRegularBandRows: number;
  /** `regularBandRows / bandRows`; the multiplier on any per-row saving. */
  readonly regularShare: number;
}

export function decodeOctreePersistentMGPCGBandCensus(
  words: Uint32Array,
): OctreePersistentMGPCGBandCensus | null {
  if (words.length < OCTREE_PERSISTENT_MGPCG_HEADER.bandCensusWords) return null;
  if (words[0] !== OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER) return null;
  const bandRows = words[1]!, regularBandRows = words[2]!;
  const coarseRegularBandRows = words[3]!;
  if (regularBandRows > bandRows || coarseRegularBandRows > regularBandRows) return null;
  return Object.freeze({
    bandRows, regularBandRows, coarseRegularBandRows,
    regularShare: bandRows === 0 ? 0 : regularBandRows / bandRows,
  });
}

export const OCTREE_PERSISTENT_MGPCG_ACTIVITY_MODULE_ID = "octree/persistent-mgpcg";
const OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK = "whole-solve";

const STORAGE_BINDINGS = Object.freeze([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as const);
/** Explicit, asserted: ten storage bindings plus one uniform. */
export const OCTREE_PERSISTENT_MGPCG_STORAGE_BINDING_COUNT = STORAGE_BINDINGS.length;

const PARAMS_LEVEL_TABLE_SLOTS = 16;
const PARAMS_BYTES = 96 + 5 * PARAMS_LEVEL_TABLE_SLOTS * 4;
const DISPATCH_RECORD_WORDS_PER_LEVEL = 12;
const DISPATCH_LIFECYCLE_WORDS = 2;
const WORKSET_ROW_CLASS_COUNT = 5;

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
  /** Accepted dynamic-boundary `C` publication. The solver consumes the
   * `[flags, firstError, rows, slots, epoch, bank, published]` prefix live. */
  readonly acceptedAuthority: GPUBuffer;
  /** Packed A/B dynamic-boundary row and family worksets. */
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
  readonly workgroupStorageBytes: number;

  private readonly params: GPUBuffer;
  private readonly arena: GPUBuffer;
  private pipeline!: GPUComputePipeline;
  private readonly activity: GPULogicalActivityAdoptionContext;
  private readonly shaderCode: string;
  private readonly shaderLabel: string;
  private readonly groups: Array<{ pressureOut: GPUBuffer; group: GPUBindGroup }> = [];
  private readonly dispatchMetaBytes: number;
  private readonly compactLiveRows: boolean;
  private readonly stagedSmoother: boolean;
  private readonly regularBandRows: OctreePersistentMGPCGRegularBandRowsMode;
  private destroyed = false;

  /** GPU-authored records suitable for the existing diagnostics readback. */
  get workAccountingBuffers(): Readonly<{ control: GPUBuffer; arena: GPUBuffer }> {
    return Object.freeze({ control: this.source.control, arena: this.arena });
  }

  diagnosticRowStride(liveRows: number): number {
    return this.compactLiveRows ? liveRows : this.source.rowCapacity;
  }

  diagnosticStageByteOffset(stage: "initialResidual" | "initialPreconditioned"): number {
    return this.channelByteOffset(stage === "initialResidual"
      ? OCTREE_PERSISTENT_MGPCG_CHANNEL.rhs
      : OCTREE_PERSISTENT_MGPCG_CHANNEL.pressureSeed);
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

  /** Post-submit Bet-4 work census. It is observational and never schedules. */
  async readHybridCensus(): Promise<OctreePersistentMGPCGHybridCensus | null> {
    this.assertLive();
    const readback = this.device.createBuffer({
      label: "Persistent MGPCG hybrid work census readback",
      size: 40,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read persistent hybrid work census" });
    encoder.copyBufferToBuffer(this.source.control,
      OCTREE_PERSISTENT_MGPCG_CONTROL.hybridRegularRows * 4, readback, 0, 40);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      return decodeOctreePersistentMGPCGHybridCensus(
        Uint32Array.from(new Uint32Array(readback.getMappedRange())),
      );
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /**
   * Post-submit class-0 band census. Observational, and `null` unless
   * `FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS` selected a mode for this executor —
   * the census words are only authored by that emission, so a stale arena
   * cannot be mistaken for a measurement.
   */
  async readBandCensus(): Promise<OctreePersistentMGPCGBandCensus | null> {
    this.assertLive();
    if (this.regularBandRows === "off") return null;
    const bytes = OCTREE_PERSISTENT_MGPCG_HEADER.bandCensusWords * 4;
    const readback = this.device.createBuffer({
      label: "Persistent MGPCG band census readback",
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read persistent band census" });
    encoder.copyBufferToBuffer(this.arena,
      OCTREE_PERSISTENT_MGPCG_HEADER.bandCensus * 4, readback, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      return decodeOctreePersistentMGPCGBandCensus(
        Uint32Array.from(new Uint32Array(readback.getMappedRange())),
      );
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  constructor(
    private readonly device: GPUDevice,
    private readonly source: OctreePersistentMGPCGSource,
    private readonly options: OctreePersistentMGPCGOptions,
  ) {
    // Resolved before the capability gate below, which asserts the declared
    // workgroup footprint against the device limit.
    this.stagedSmoother = octreePersistentMGPCGStagedSmootherEnabled();
    this.regularBandRows = octreePersistentMGPCGRegularBandRowsMode();
    this.workgroupStorageBytes = octreePersistentMGPCGWorkgroupBytes(this.stagedSmoother);
    const rowCapacity = positiveInteger(source.rowCapacity, "Persistent MGPCG row capacity");
    // This is provisioned arena capacity, not adaptive work. The exact live
    // row count is GPU-published only after topology construction, so the
    // persistent kernel gates that count before doing arithmetic.
    this.maximumRowCapacity = persistentMGPCGRowThreshold();
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
      ["dispatch metadata", source.dispatchMeta],
      ["divergence RHS", source.rhs],
    ] as const) {
      if ((buffer.usage & copySource) === 0) {
        throw new RangeError(`Persistent MGPCG requires COPY_SRC on the ${label} buffer`);
      }
    }
    if (source.acceptedAuthority.size < 7 * 4
      || source.dispatchMeta.size < this.dispatchMetaBytes
      || source.rhs.size < rowCapacity * 4
      || source.coefficients.size < 2 * rowCapacity * 19 * 4
      || source.geometry.size < rowCapacity * 16
      || source.coefficientBankStrideWords < rowCapacity * 19
      || source.worksetStrideWords < 7 + rowCapacity
      || source.worksetBankStrideWords < WORKSET_ROW_CLASS_COUNT * source.worksetStrideWords) {
      throw new RangeError("Persistent MGPCG source capacity is too small");
    }
    const worksetSpan =
      (source.worksetBankStrideWords + WORKSET_ROW_CLASS_COUNT * source.worksetStrideWords) * 4;
    if (source.worksets.size < worksetSpan) {
      throw new RangeError("Persistent MGPCG workset staging exceeds the published banks");
    }

    const partialCapacity = Math.ceil(
      rowCapacity / OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES,
    );
    const stagedInputBaseWords = OCTREE_PERSISTENT_MGPCG_HEADER.totalWords
      + OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT * rowCapacity
      + partialCapacity * OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS;
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
      stagedInputBaseWords,
      0, 0, 0, 0,
      source.worksetStrideWords, source.worksetBankStrideWords, 0, 0,
    ]);
    floats[16] = options.relativeTolerance;
    floats[17] = absoluteTolerance;
    floats[18] = 1e-30;
    floats[19] = damping;
    words.set(tables.levelCaps, 24);
    words.set(tables.levelBases, 24 + PARAMS_LEVEL_TABLE_SLOTS);
    words.set(tables.brickOffsets, 24 + 2 * PARAMS_LEVEL_TABLE_SLOTS);
    words.set(tables.pageOffsets, 24 + 3 * PARAMS_LEVEL_TABLE_SLOTS);
    words.set(tables.transferOffsets, 24 + 4 * PARAMS_LEVEL_TABLE_SLOTS);
    device.queue.writeBuffer(this.params, 0, words);

    const activityProfile = performanceShaderVariant();
    this.activity = createGPULogicalActivityAdoptionContext({
      moduleId: OCTREE_PERSISTENT_MGPCG_ACTIVITY_MODULE_ID,
      profile: activityProfile,
      identity: "subgroup",
    });
    this.activity.describeTask(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, {
      id: "gpu.physics.persistent-mgpcg.whole-solve",
      label: "Persistent MGPCG · whole solve",
      phaseId: "pressure-solve",
      checkpoints: {
        enter: this.activity.checkpointId(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, "enter"),
        exit: this.activity.checkpointId(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, "exit"),
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
    this.compactLiveRows = compactLiveRows;
    const diagnosticOutputChannel = persistentMGPCGDiagnosticOutputChannel();
    const regularAddressOracle = typeof process !== "undefined"
      && process.env.FLUID_POWER_HYBRID_FULL_ADDRESS_ORACLE === "1";
    const regularAdjointOracle = typeof process !== "undefined"
      && process.env.FLUID_POWER_HYBRID_FULL_ADJOINT_ORACLE === "1";
    const regularAllChannelsOracle = typeof process !== "undefined"
      && process.env.FLUID_POWER_HYBRID_ALL_CHANNELS_ORACLE === "1";
    const section63BandOracle = typeof process !== "undefined"
      && process.env.FLUID_POWER_HYBRID_SECTION63_BAND_ORACLE === "1";
    const fullApplyOracle = typeof process !== "undefined"
      && process.env.FLUID_POWER_HYBRID_FULL_APPLY_ORACLE === "1";
    const linearApplyOracle = typeof process !== "undefined"
      && process.env.FLUID_POWER_HYBRID_LINEAR_APPLY_ORACLE === "1";
    const stageCaptureNames = ["initialPreconditioned", "vcyclePhase0", "vcyclePresmooth",
      "vcycleRestrict", "vcycleProlong", "vcyclePostsmooth", "section43Boundary",
      "section43First", "section43Inputs"] as const;
    const requestedStageCapture = typeof process !== "undefined"
      ? process.env.FLUID_PERSISTENT_MGPCG_STAGE_CAPTURE : undefined;
    if (requestedStageCapture !== undefined
      && !stageCaptureNames.includes(requestedStageCapture as typeof stageCaptureNames[number])) {
      throw new RangeError(`Unknown persistent MGPCG stage capture ${requestedStageCapture}`);
    }
    const diagnosticStageCapture = typeof process !== "undefined"
      && process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1"
      ? (requestedStageCapture ?? "initialPreconditioned") as typeof stageCaptureNames[number]
      : undefined;
    const shaderSource = octreePersistentMGPCGWGSL({
      maximumIterations: options.maximumIterations,
      compactLiveRows,
      regularAddressOracle,
      regularAdjointOracle,
      regularAllChannelsOracle,
      section63BandOracle,
      fullApplyOracle,
      linearApplyOracle,
      stagedSmoother: this.stagedSmoother,
      ...(this.regularBandRows === "off" ? {} : { regularBandRows: this.regularBandRows }),
      diagnosticStageCapture,
      ...(diagnosticOutputChannel === undefined ? {} : { diagnosticOutputChannel }),
      ...(this.activity.enabled ? { activity: {
        parameters: activityParameters,
        enter: this.activity.subgroup(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, "enter", activitySite),
        // Scalar exit markers preserve interval accounting without adding a
        // second ballot to the sampled-lane utilization denominator.
        exit: this.activity.workgroup(OCTREE_PERSISTENT_MGPCG_ACTIVITY_TASK, "exit", {
          tick: "atomicLoad(&control[2])",
          workgroupId: "activityWorkgroupId",
          numWorkgroups: "activityNumWorkgroups",
          localInvocationIndex: "lane",
          workgroupLaneCount: OCTREE_PERSISTENT_MGPCG_LANES,
        }),
      } } : {}),
    });
    const shaderVariant = this.activity.module(
      shaderSource,
      `${OCTREE_PERSISTENT_MGPCG_ACTIVITY_MODULE_ID}/${activityProfile.cacheKey}`
        + `/${compactLiveRows ? "compact-live" : "capacity-strided"}`
        + `/${regularAddressOracle ? "full-address" : "regular-address"}`
        + `/${regularAdjointOracle ? "full-adjoint" : "no-adjoint"}`
        + `/${regularAllChannelsOracle ? "all-channels" : "axis-channels"}`
        + `/${section63BandOracle ? "section63-band" : "dynamic-band"}`
        + `/${fullApplyOracle ? "full-apply" : "hybrid-apply"}`
        + `/${linearApplyOracle ? "linear-apply" : "class-apply"}`
        + `/${this.stagedSmoother ? "staged-smoother" : "direct-smoother"}`
        + `/band-rows-${this.regularBandRows}`
        + `/diagnostic-output-${diagnosticOutputChannel ?? "pressure"}`,
    );
    this.shaderCode = shaderVariant.code;
    this.shaderLabel = `Octree persistent MGPCG · single-dispatch 256-lane solve · ${
      compactLiveRows ? "compact live rows" : "capacity-strided oracle"}${
      this.stagedSmoother ? " · staged smoother" : ""}${
      this.regularBandRows === "off" ? "" : ` · ${this.regularBandRows} band rows`}`;
    this.allocatedBytes = this.arena.size + this.params.size;
  }

  async initializePipeline(): Promise<void> {
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({
      label: this.shaderLabel,
      code: this.shaderCode,
    });
    this.pipeline = this.activity.registerPipeline(await compiler.compileComputePipeline({
      label: "Persistent MGPCG · persistentMGPCG",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "persistentMGPCG" },
    }));
  }

  /** Byte offset of an immutable capacity-strided solve input. */
  channelByteOffset(channel: number): number {
    if (channel !== OCTREE_PERSISTENT_MGPCG_CHANNEL.rhs
      && channel !== OCTREE_PERSISTENT_MGPCG_CHANNEL.pressureSeed) {
      throw new RangeError("Persistent MGPCG staging accepts only RHS and pressure seed channels");
    }
    const rowCapacity = this.source.rowCapacity;
    const partialCapacity = Math.ceil(
      rowCapacity / OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES,
    );
    const stagedInputBaseWords = OCTREE_PERSISTENT_MGPCG_HEADER.totalWords
      + OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT * rowCapacity
      + partialCapacity * OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS;
    return (stagedInputBaseWords
      + (channel - OCTREE_PERSISTENT_MGPCG_CHANNEL.rhs) * rowCapacity) * 4;
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
    broker.copyBufferToBuffer(this.source.dispatchMeta, 0, this.arena,
      OCTREE_PERSISTENT_MGPCG_HEADER.dispatch * 4, this.dispatchMetaBytes);
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
      9: this.source.worksets,
      10: this.source.acceptedAuthority,
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

  /** Whether an exact GPU-published live row count fits this executor. */
  static acceptsLiveRows(
    liveRowCount: number,
    environment?: Readonly<Record<string, string | undefined>>,
  ): boolean {
    return Number.isSafeInteger(liveRowCount) && liveRowCount > 0
      && liveRowCount <= persistentMGPCGRowThreshold(environment);
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
